"""End-to-end tests against the real llama-server builds and the Qwen3.5-0.8B test models.

Run with ``uv run pytest -m integration`` after ``uv run python scripts/fetch_test_assets.py``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TypeGuard

import httpx
import pytest

from dom.backend import LlamaInferenceService, LlamaServerProcess, ServerConfig
from dom.backend.presets import PresetFile
from dom.backend.service import AutoTuning
from dom.config import GenerationSettings, InferenceSettings, ServerBuild, find_server_builds
from dom.config.binary import server_executable_name
from dom.domain import ChatDelta, ChatMessage, ModelStatus, Role
from dom.hardware import HardwareProbe
from dom.hardware.probe import read_memory
from dom.tuning import BenchmarkCache, LlamaBench
from dom.tuning.tuner import CTX_CAP, CTX_MIN

DEV_DIR = Path(__file__).resolve().parents[2] / ".dev"
CPU_BINARY = DEV_DIR / "bin" / "cpu" / server_executable_name()
MODELS_DIR = DEV_DIR / "models"
Q4 = "Qwen3.5-0.8B-Q4_K_M"
Q8 = "Qwen3.5-0.8B-Q8_0"
REPLY_TIMEOUT_S = 300.0

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not CPU_BINARY.is_file()
        or not (MODELS_DIR / f"{Q4}.gguf").is_file()
        or not (MODELS_DIR / f"{Q8}.gguf").is_file(),
        reason="test assets missing: run `uv run python scripts/fetch_test_assets.py`",
    ),
]


async def _make_service(
    tmp_path: Path, builds: list[ServerBuild]
) -> tuple[LlamaInferenceService, LlamaServerProcess, AutoTuning]:
    """The same wiring as ``dom.app.build_service``, keeping the process for inspection."""
    hardware = await HardwareProbe().detect(builds)
    assert hardware.build is not None
    binary = hardware.build.path
    presets = PresetFile(tmp_path / "presets.ini", InferenceSettings())
    process = LlamaServerProcess(
        ServerConfig(
            binary=binary,
            models_dir=MODELS_DIR,
            cache_dir=tmp_path / "cache",
            log_path=tmp_path / "llama-server.log",
            presets=presets,
        )
    )
    tuning = AutoTuning(
        hardware=hardware,
        presets=presets,
        defaults=InferenceSettings(),
        memory_reader=read_memory,
        bench=LlamaBench(binary.with_name(binary.name.replace("llama-server", "llama-bench"))),
        bench_cache=BenchmarkCache(tmp_path / "benchmarks.json"),
        build_version="test",
    )
    return LlamaInferenceService(process, tuning), process, tuning


async def _reply(service: LlamaInferenceService, model_id: str) -> str:
    messages = [ChatMessage(Role.USER, "Say hello in one short sentence.")]
    parts: list[str] = []
    async with asyncio.timeout(REPLY_TIMEOUT_S):
        async for delta in service.stream_chat(model_id, messages):
            parts.append(delta.content)
    return "".join(parts).strip()


async def _check_generation_settings(service: LlamaInferenceService, model_id: str) -> None:
    """A profile's settings reach the server: no thinking, and the reply is cut at the cap."""
    cap = 12
    settings = GenerationSettings(max_reply_tokens=cap, thinking=False)
    messages = [ChatMessage(Role.USER, "Write a long story about a dragon.")]
    deltas: list[ChatDelta] = []
    async with asyncio.timeout(REPLY_TIMEOUT_S):
        async for delta in service.stream_chat(model_id, messages, settings):
            deltas.append(delta)
    assert not "".join(d.reasoning for d in deltas), "thinking was not turned off"
    assert "".join(d.content for d in deltas).strip(), "no answer without thinking"
    assert len([d for d in deltas if d.content]) <= cap, "the reply cap was ignored"


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_array(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


async def _effective_ctx(process: LlamaServerProcess, model_id: str) -> int | None:
    """The context size the server actually allocated for a loaded model."""
    endpoint = process.endpoint
    assert endpoint is not None
    async with httpx.AsyncClient(
        base_url=endpoint.base_url, headers={"Authorization": f"Bearer {endpoint.api_key}"}
    ) as client:
        payload: object = (await client.get("/models")).json()
    assert _is_object(payload)
    data = payload.get("data")
    assert _is_array(data)
    for entry in data:
        if _is_object(entry) and entry.get("id") == model_id:
            memory = entry.get("memory")
            if _is_object(memory) and _is_object(effective := memory.get("effective")):
                n_ctx = effective.get("n_ctx")
                return n_ctx if isinstance(n_ctx, int) else None
    return None


def _process_group_alive(pgid: int) -> bool:
    if sys.platform == "win32":
        return False  # the job object guarantees cleanup; nothing portable to probe here
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_cpu_build_tunes_loads_chats_and_switches(tmp_path: Path) -> None:
    service, process, tuning = await _make_service(tmp_path, [ServerBuild("cpu", CPU_BINARY)])
    await service.start()
    pgid = process.pid
    assert pgid is not None
    try:
        models = {m.id: m for m in await service.list_models()}
        assert set(models) == {Q4, Q8}, f"unexpected models: {sorted(models)}"

        loaded = await service.select_model(Q4)
        assert loaded.info.status == ModelStatus.LOADED
        assert loaded.device_label == "CPU"
        assert loaded.tuned, "settings were not based on the server's memory measurement"
        assert loaded.ctx_size is not None
        assert CTX_MIN <= loaded.ctx_size <= CTX_CAP
        assert f"ctx-size = {loaded.ctx_size}" in tuning.presets.path.read_text(encoding="utf-8")
        assert await _effective_ctx(process, Q4) == loaded.ctx_size
        assert await _reply(service, Q4), "Q4_K_M gave an empty reply"
        await _check_generation_settings(service, Q4)

        await service.select_model(Q8)
        statuses = {m.id: m.status for m in await service.list_models()}
        assert statuses == {Q4: ModelStatus.UNLOADED, Q8: ModelStatus.LOADED}
        assert await _reply(service, Q8), "Q8_0 gave an empty reply"
    finally:
        await service.stop()
    assert not process.is_running
    assert not _process_group_alive(pgid), "llama-server processes were left running"


async def test_gpu_build_is_chosen_and_benchmarked(tmp_path: Path) -> None:
    builds = find_server_builds(None, DEV_DIR / "bin")
    service, process, tuning = await _make_service(tmp_path, builds)
    gpu = tuning.hardware.gpu
    if gpu is None:
        pytest.skip("no GPU visible to any llama-server build on this machine")
    await service.start()
    progress: list[str] = []
    try:
        loaded = await service.select_model(Q4, progress.append)
        assert loaded.info.status == ModelStatus.LOADED
        assert loaded.tuned, "settings were not based on the server's memory measurement"
        if gpu.integrated:
            assert any("faster" in step for step in progress), progress
            assert (tmp_path / "benchmarks.json").is_file()
            # Second time: the decision is cached, so no benchmark.
            await service.select_model(Q8)
            progress.clear()
            await service.select_model(Q4, progress.append)
            assert not any("faster" in step for step in progress), progress
        assert await _effective_ctx(process, Q4) == loaded.ctx_size
        assert await _reply(service, Q4), "empty reply on the GPU build"
    finally:
        await service.stop()
    assert not process.is_running
