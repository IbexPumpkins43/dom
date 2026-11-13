"""Choosing a model's settings from the hardware and llama-server's memory measurement.

llama-server's ``--fit`` already places layers so a model fits in GPU memory, but it treats
system RAM as unlimited. The tuner's job is RAM: pick the largest context that fits in what the
computer has free (capped), or report that the model is too large.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from dom.hardware.models import GIB

if TYPE_CHECKING:
    from dom.config import InferenceSettings
    from dom.domain import ModelInfo, ModelMemory
    from dom.hardware import GpuDevice, MemoryStatus

CTX_MIN = 4096
CTX_CAP = 32768
CTX_STEP = 1024
# RAM kept free for the OS, dom and other apps: 10% of RAM, between 1 and 2 GiB.
RAM_MARGIN_SHARE = 0.10
RAM_MARGIN_MIN = 1 * GIB
RAM_MARGIN_MAX = 2 * GIB
# GPU memory kept free (matches the router's --models-memory-margin).
GPU_MARGIN = 1 * GIB
# Thinking budget by model size (weights + minimum context, in bytes).
_REASONING_BUDGETS: tuple[tuple[int, int], ...] = ((3 * GIB, 512), (12 * GIB, 1024))
_REASONING_BUDGET_LARGE = 2048


@dataclass(frozen=True, slots=True)
class Machine:
    """The memory picture at load time."""

    ram: MemoryStatus
    gpu: GpuDevice | None


class Placement(StrEnum):
    """Where a model runs."""

    GPU = "gpu"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class ModelPlan:
    """The settings dom will load a model with, and why."""

    model_id: str
    settings: InferenceSettings
    placement: Placement
    ctx_size: int
    required_bytes: int | None
    budget_bytes: int


@dataclass(frozen=True, slots=True)
class TooLarge:
    """The model cannot run on this computer even with the smallest context."""

    model_id: str
    required_bytes: int
    budget_bytes: int

    def message(self) -> str:
        """Plain-English explanation for the UI."""
        return (
            f"{self.model_id} needs about {self.required_bytes / GIB:.1f} GB of memory, but only "
            f"about {max(self.budget_bytes, 0) / GIB:.1f} GB is free on this computer. "
            "Close other programs, or choose a smaller model."
        )


def ram_margin(total_bytes: int) -> int:
    """RAM to leave free for everything else."""
    return min(RAM_MARGIN_MAX, max(RAM_MARGIN_MIN, round(total_bytes * RAM_MARGIN_SHARE)))


def memory_budget(ram: MemoryStatus, gpu: GpuDevice | None, placement: Placement) -> int:
    """Bytes the model may use.

    A dedicated GPU adds its free memory (``--fit`` spills whatever does not fit there into
    RAM). An integrated GPU shares system RAM, so it adds nothing.
    """
    budget = ram.available_bytes - ram_margin(ram.total_bytes)
    if placement == Placement.GPU and gpu is not None and not gpu.integrated:
        budget += max(0, gpu.free_bytes - GPU_MARGIN)
    return budget


def _ctx_upper(model: ModelInfo) -> int:
    trained = model.context_length
    return min(CTX_CAP, trained) if trained else CTX_CAP


def _largest_fitting_ctx(memory: ModelMemory, budget: int, lower: int, upper: int) -> int:
    low, high = memory.low, memory.high
    per_token = (high.total_bytes - low.total_bytes) / (high.n_ctx - low.n_ctx)
    if per_token <= 0:
        return upper
    n = low.n_ctx + (budget - low.total_bytes) / per_token
    n = math.floor(n / CTX_STEP) * CTX_STEP
    return max(lower, min(upper, n))


def _reasoning_budget(size_bytes: int | None, default: int | None) -> int | None:
    if size_bytes is None:
        return default
    for limit, budget in _REASONING_BUDGETS:
        if size_bytes < limit:
            return budget
    return _REASONING_BUDGET_LARGE


def plan_model(
    model: ModelInfo,
    memory: ModelMemory | None,
    machine: Machine,
    placement: Placement,
    defaults: InferenceSettings,
) -> ModelPlan | TooLarge:
    """Settings for ``model`` on this computer, or ``TooLarge``.

    ``memory`` must be an unpinned measurement (two different context sizes); without one the
    plan falls back to the default context size and skips the fit check.
    """
    budget = memory_budget(machine.ram, machine.gpu, placement)
    upper = _ctx_upper(model)
    lower = min(CTX_MIN, upper)
    device = "none" if placement == Placement.CPU and machine.gpu is not None else None
    if memory is None or not memory.can_extrapolate:
        ctx = min(defaults.ctx_size or CTX_MIN, upper)
        settings = replace(defaults, ctx_size=ctx, device=device)
        return ModelPlan(model.id, settings, placement, ctx, None, budget)
    minimum = memory.at(lower).total_bytes
    if minimum > budget:
        return TooLarge(model.id, minimum, budget)
    ctx = _largest_fitting_ctx(memory, budget, lower, upper)
    settings = replace(
        defaults,
        ctx_size=ctx,
        device=device,
        reasoning_budget=_reasoning_budget(minimum, defaults.reasoning_budget),
    )
    return ModelPlan(model.id, settings, placement, ctx, memory.at(ctx).total_bytes, budget)
