"""Tests for the tuner and the CPU/GPU benchmark."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from dom.config import InferenceSettings
from dom.domain import MemoryUse, ModelInfo, ModelMemory, ModelStatus
from dom.hardware import GpuDevice, MemoryStatus
from dom.hardware.models import GIB, MIB
from dom.tuning import (
    BenchmarkCache,
    BenchmarkResult,
    Machine,
    ModelPlan,
    Placement,
    TooLarge,
    choose_faster,
    parse_bench_output,
    plan_model,
)
from dom.tuning.tuner import CTX_CAP, CTX_MIN, memory_budget, ram_margin

if TYPE_CHECKING:
    from pathlib import Path

DEFAULTS = InferenceSettings()
QWEN_08B = ModelInfo("Qwen3.5-0.8B-Q4_K_M", ModelStatus.UNLOADED, context_length=262144)
# Measured on the CPU build: ~0.88 GB at 4k context, ~4.4 GB at 262k (about 12.6 kB/token).
QWEN_08B_MEMORY = ModelMemory(
    low=MemoryUse(4096, 0, 352865536 + 168689664 + 131137536 + 525428768),
    high=MemoryUse(262144, 0, 352865536 + 168689664 + 3302031360 + 791701536),
)
# A made-up 30B-class model: 18 GB of weights, 100 kB per context token.
BIG = ModelInfo("Big-30B-Q4_K_M", ModelStatus.UNLOADED, context_length=131072)
BIG_MEMORY = ModelMemory(
    low=MemoryUse(4096, 0, 18 * GIB + 4096 * 100_000),
    high=MemoryUse(131072, 0, 18 * GIB + 131072 * 100_000),
)
IGPU = GpuDevice("Vulkan0", "Intel(R) Iris(R) Xe Graphics", 11 * GIB, 7 * GIB, integrated=True)
DGPU = GpuDevice("Vulkan0", "NVIDIA GeForce RTX 4070", 12 * GIB, 11 * GIB, integrated=False)


def _plan(result: ModelPlan | TooLarge) -> ModelPlan:
    assert isinstance(result, ModelPlan), result
    return result


def test_ram_margin_is_clamped() -> None:
    assert ram_margin(4 * GIB) == 1 * GIB
    assert ram_margin(16 * GIB) == pytest.approx(1.6 * GIB, rel=1e-3)
    assert ram_margin(64 * GIB) == 2 * GIB


def test_budget_adds_only_dedicated_gpu_memory() -> None:
    ram = MemoryStatus(16 * GIB, 10 * GIB)
    cpu_budget = memory_budget(ram, None, Placement.CPU)
    assert memory_budget(ram, IGPU, Placement.GPU) == cpu_budget
    assert memory_budget(ram, DGPU, Placement.GPU) == cpu_budget + 10 * GIB
    assert memory_budget(ram, DGPU, Placement.CPU) == cpu_budget


def test_plenty_of_ram_gets_the_context_cap() -> None:
    plan = _plan(
        plan_model(
            QWEN_08B,
            QWEN_08B_MEMORY,
            Machine(MemoryStatus(16 * GIB, 10 * GIB), None),
            Placement.CPU,
            DEFAULTS,
        )
    )
    assert plan.ctx_size == CTX_CAP
    assert plan.settings.ctx_size == CTX_CAP
    assert plan.settings.device is None  # CPU-only build: nothing to switch off
    assert plan.settings.reasoning_budget == 512  # noqa: PLR2004


def test_tight_ram_gets_a_smaller_context() -> None:
    # 8 GB machine with 2.2 GB free: budget = 2.2 - 1.0 = 1.2 GB.
    ram = MemoryStatus(8 * GIB, int(2.2 * GIB))
    plan = _plan(plan_model(QWEN_08B, QWEN_08B_MEMORY, Machine(ram, None), Placement.CPU, DEFAULTS))
    assert CTX_MIN < plan.ctx_size < CTX_CAP
    assert plan.ctx_size % 1024 == 0
    assert plan.required_bytes is not None
    assert plan.required_bytes <= plan.budget_bytes
    assert QWEN_08B_MEMORY.at(plan.ctx_size + 1024).total_bytes > plan.budget_bytes


def test_model_that_cannot_fit_is_too_large() -> None:
    result = plan_model(
        BIG, BIG_MEMORY, Machine(MemoryStatus(16 * GIB, 12 * GIB), None), Placement.CPU, DEFAULTS
    )
    assert isinstance(result, TooLarge)
    assert "needs about 18.4 GB" in result.message()


def test_dedicated_gpu_memory_lets_a_big_model_fit() -> None:
    machine = Machine(MemoryStatus(16 * GIB, 12 * GIB), DGPU)
    plan = _plan(plan_model(BIG, BIG_MEMORY, machine, Placement.GPU, DEFAULTS))
    assert plan.settings.device is None
    assert plan.settings.reasoning_budget == 2048  # noqa: PLR2004


def test_cpu_placement_with_a_gpu_present_disables_offload() -> None:
    plan = _plan(
        plan_model(
            QWEN_08B,
            QWEN_08B_MEMORY,
            Machine(MemoryStatus(16 * GIB, 9 * GIB), IGPU),
            Placement.CPU,
            DEFAULTS,
        )
    )
    assert plan.settings.device == "none"


def test_short_trained_context_is_respected() -> None:
    model = ModelInfo("tiny", ModelStatus.UNLOADED, context_length=2048)
    memory = ModelMemory(MemoryUse(2048, 0, 100 * MIB), MemoryUse(2048, 0, 100 * MIB))
    plan = _plan(
        plan_model(
            model, memory, Machine(MemoryStatus(16 * GIB, 9 * GIB), None), Placement.CPU, DEFAULTS
        )
    )
    assert plan.ctx_size == 2048  # noqa: PLR2004


def test_without_a_measurement_falls_back_to_defaults() -> None:
    plan = _plan(
        plan_model(
            QWEN_08B, None, Machine(MemoryStatus(16 * GIB, 9 * GIB), None), Placement.CPU, DEFAULTS
        )
    )
    assert plan.ctx_size == DEFAULTS.ctx_size
    assert plan.required_bytes is None


# ---- benchmark ---------------------------------------------------------------------------

BENCH_JSON = json.dumps([
    {"n_prompt": 256, "n_gen": 0, "avg_ts": 622.7},
    {"n_prompt": 0, "n_gen": 32, "avg_ts": 35.4},
])  # fmt: skip


def test_parse_bench_output() -> None:
    assert parse_bench_output(BENCH_JSON) == BenchmarkResult(prompt_tps=622.7, gen_tps=35.4)


@pytest.mark.parametrize("text", ["", "not json", "[]", '[{"n_prompt": 256, "n_gen": 0}]'])
def test_parse_bench_output_rejects_garbage(text: str) -> None:
    assert parse_bench_output(text) is None


def test_choose_faster_uses_a_typical_turn() -> None:
    # Measured on an Iris Xe: GPU reads prompts 3x faster, generates slightly slower.
    gpu = BenchmarkResult(prompt_tps=622.7, gen_tps=35.4)
    cpu = BenchmarkResult(prompt_tps=202.9, gen_tps=38.7)
    assert choose_faster(gpu, cpu) == Placement.GPU
    slow_gpu = BenchmarkResult(prompt_tps=150.0, gen_tps=10.0)
    assert choose_faster(slow_gpu, cpu) == Placement.CPU


def test_choose_faster_handles_failed_runs() -> None:
    result = BenchmarkResult(100.0, 10.0)
    assert choose_faster(None, result) == Placement.CPU
    assert choose_faster(result, None) == Placement.GPU
    assert choose_faster(None, None) == Placement.GPU


def test_benchmark_cache_round_trip(tmp_path: Path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"1234")
    cache = BenchmarkCache(tmp_path / "bench.json")
    key = BenchmarkCache.key(model, "Iris Xe", "b12110")
    assert key == "m.gguf|4|Iris Xe|b12110"
    assert cache.get(key) is None
    cache.put(key, Placement.CPU, BenchmarkResult(1.0, 2.0), None)
    assert BenchmarkCache(tmp_path / "bench.json").get(key) == Placement.CPU


def test_benchmark_cache_ignores_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "bench.json"
    path.write_text("{not json", encoding="utf-8")
    assert BenchmarkCache(path).get("x") is None
