"""End-to-end test against the real llama-server and the Qwen3.5-0.8B test models.

Run with ``uv run pytest -m integration`` after ``uv run python scripts/fetch_test_assets.py``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from dom.backend import LlamaInferenceService, LlamaServerProcess, ServerConfig
from dom.config import InferenceSettings
from dom.config.binary import server_executable_name
from dom.domain import ChatMessage, ModelStatus, Role

DEV_DIR = Path(__file__).resolve().parents[2] / ".dev"
BINARY = DEV_DIR / "bin" / "cpu" / server_executable_name()
MODELS_DIR = DEV_DIR / "models"
Q4 = "Qwen3.5-0.8B-Q4_K_M"
Q8 = "Qwen3.5-0.8B-Q8_0"
REPLY_TIMEOUT_S = 300.0

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not BINARY.is_file()
        or not (MODELS_DIR / f"{Q4}.gguf").is_file()
        or not (MODELS_DIR / f"{Q8}.gguf").is_file(),
        reason="test assets missing: run `uv run python scripts/fetch_test_assets.py`",
    ),
]


async def _reply(service: LlamaInferenceService, model_id: str) -> str:
    messages = [ChatMessage(Role.USER, "Say hello in one short sentence.")]
    parts: list[str] = []
    async with asyncio.timeout(REPLY_TIMEOUT_S):
        async for delta in service.stream_chat(model_id, messages):
            parts.append(delta.content)
    return "".join(parts).strip()


def _process_group_alive(pgid: int) -> bool:
    if sys.platform == "win32":
        return False  # the job object guarantees cleanup; nothing portable to probe here
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_real_server_load_chat_and_switch(tmp_path: Path) -> None:
    process = LlamaServerProcess(
        ServerConfig(
            binary=BINARY,
            models_dir=MODELS_DIR,
            cache_dir=tmp_path / "cache",
            log_path=tmp_path / "llama-server.log",
            settings=InferenceSettings(),
        )
    )
    service = LlamaInferenceService(process)
    await service.start()
    pgid = process.pid
    assert pgid is not None
    try:
        models = {m.id: m for m in await service.list_models()}
        assert set(models) == {Q4, Q8}, f"unexpected models: {sorted(models)}"

        loaded = await service.select_model(Q4)
        assert loaded.status == ModelStatus.LOADED
        assert await _reply(service, Q4), "Q4_K_M gave an empty reply"

        await service.select_model(Q8)
        statuses = {m.id: m.status for m in await service.list_models()}
        assert statuses == {Q4: ModelStatus.UNLOADED, Q8: ModelStatus.LOADED}
        assert await _reply(service, Q8), "Q8_0 gave an empty reply"
    finally:
        await service.stop()
    assert not process.is_running
    assert not _process_group_alive(pgid), "llama-server processes were left running"
