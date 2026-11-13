"""Tests for the presets file and the router's memory measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dom.backend.client import parse_model_memory, parse_models
from dom.backend.errors import BackendError
from dom.backend.presets import PresetFile, render_presets
from dom.config import InferenceSettings
from dom.domain import MemoryUse, ModelMemory

if TYPE_CHECKING:
    from pathlib import Path

# Real measurement of Qwen3.5-0.8B-Q4_K_M from the b12110 Vulkan build on an Iris Xe.
VULKAN_MEMORY: dict[str, object] = {
    "status": "measured",
    "min": {
        "n_ctx": 4096,
        "devices": [
            {
                "kind": "device",
                "name": "Vulkan0",
                "model": 521555200,
                "context": 131137536,
                "compute": 521164816,
                "total": 11934910464,
                "free": 7959957504,
            },
            {"kind": "host", "name": "Host", "model": 208588800, "context": 0, "compute": 12619776},
        ],
    },
    "max": {
        "n_ctx": 262144,
        "devices": [
            {
                "kind": "device",
                "name": "Vulkan0",
                "model": 521555200,
                "context": 3302031360,
                "compute": 785405968,
                "total": 11934910464,
                "free": 7959957504,
            },
            {
                "kind": "host",
                "name": "Host",
                "model": 208588800,
                "context": 0,
                "compute": 276860928,
            },
        ],
    },
    "effective": None,
}


def test_parse_model_memory_splits_device_and_host() -> None:
    memory = parse_model_memory(VULKAN_MEMORY)
    assert memory == ModelMemory(
        low=MemoryUse(4096, 521555200 + 131137536 + 521164816, 208588800 + 12619776),
        high=MemoryUse(262144, 521555200 + 3302031360 + 785405968, 208588800 + 276860928),
    )


def test_cpu_buffer_types_count_as_host_memory() -> None:
    memory = parse_model_memory(
        {
            "status": "measured",
            "min": {
                "n_ctx": 4096,
                "devices": [
                    {"kind": "host", "model": 100, "context": 10, "compute": 1},
                    {
                        "kind": "buffer_type",
                        "name": "CPU_REPACK",
                        "model": 50,
                        "context": 0,
                        "compute": 0,
                    },
                ],
            },
            "max": {"n_ctx": 8192, "devices": [{"kind": "host", "model": 100, "context": 20}]},
        }
    )
    assert memory is not None
    assert memory.low == MemoryUse(4096, 0, 161)


@pytest.mark.parametrize("status", ["unmeasured", "measuring", "failed"])
def test_unfinished_measurement_is_none(status: str) -> None:
    assert parse_model_memory({**VULKAN_MEMORY, "status": status}) is None


def test_memory_interpolates_per_token() -> None:
    memory = parse_model_memory(VULKAN_MEMORY)
    assert memory is not None
    assert memory.can_extrapolate
    mid = memory.at(8192)
    assert memory.low.total_bytes < mid.total_bytes < memory.high.total_bytes
    per_token = (memory.high.total_bytes - memory.low.total_bytes) / (262144 - 4096)
    assert mid.total_bytes == pytest.approx(memory.low.total_bytes + per_token * 4096, abs=2)


def test_pinned_measurement_cannot_extrapolate() -> None:
    point = MemoryUse(6144, 1000, 10)
    memory = ModelMemory(point, point)
    assert not memory.can_extrapolate
    assert memory.at(32768) == point


def test_parse_models_reads_context_length_and_memory() -> None:
    models = parse_models(
        {
            "data": [
                {
                    "id": "m",
                    "source": "models_dir",
                    "status": {"value": "unloaded"},
                    "context_length": 262144,
                    "memory": VULKAN_MEMORY,
                }
            ]
        }
    )
    assert models[0].context_length == 262144  # noqa: PLR2004
    assert models[0].memory is not None


def test_render_presets() -> None:
    text = render_presets(
        InferenceSettings(),
        {
            "b-model": InferenceSettings(ctx_size=16384, device="none"),
            "a-model": InferenceSettings(ctx_size=4096, reasoning_budget=None),
        },
    )
    assert text.splitlines() == [
        "version = 1",
        "",
        "; Written by dom. Changes will be overwritten.",
        "",
        "[*]",
        "reasoning-budget = 512",
        "",
        "[a-model]",
        "ctx-size = 4096",
        "",
        "[b-model]",
        "ctx-size = 16384",
        "device = none",
        "reasoning-budget = 512",
    ]


def test_render_presets_rejects_unsafe_names() -> None:
    with pytest.raises(BackendError):
        render_presets(InferenceSettings(), {"bad]name": InferenceSettings()})


def test_preset_file_writes_and_resets(tmp_path: Path) -> None:
    presets = PresetFile(tmp_path / "data" / "presets.ini", InferenceSettings())
    presets.reset()
    assert "[*]" in presets.path.read_text(encoding="utf-8")
    assert presets.set_model("m", InferenceSettings(ctx_size=4096))
    assert not presets.set_model("m", InferenceSettings(ctx_size=4096))
    assert "[m]" in presets.path.read_text(encoding="utf-8")
    presets.reset()
    assert "[m]" not in presets.path.read_text(encoding="utf-8")


def test_model_path_comes_from_the_model_argument() -> None:
    models = parse_models(
        {
            "data": [
                {
                    "id": "m",
                    "source": "models_dir",
                    "status": {
                        "value": "unloaded",
                        "args": ["llama-server", "--alias", "m", "--model", "/x/m.gguf"],
                    },
                }
            ]
        }
    )
    assert models[0].path == "/x/m.gguf"
