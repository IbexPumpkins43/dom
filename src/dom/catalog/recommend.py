"""Which catalog models suit this computer.

A rough, explainable estimate (it is only used to suggest a download; once a model is loaded,
``dom.tuning`` sizes it exactly from llama.cpp's own measurement):

- Memory: RAM minus room for the OS and other programs, plus a dedicated GPU's memory. A model
  needs its file size plus room for an 8k-token conversation.
- Speed: generating a word means reading the model's active weights once, so tokens/second is
  about memory bandwidth / bytes read per token. Typical bandwidths are used; mixture-of-experts
  models read only their active part.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dom.hardware.models import GIB

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from dom.catalog.models import CatalogModel, CatalogQuant
    from dom.hardware import HardwareProfile

OS_RESERVE = 3 * GIB
GPU_RESERVE = 1 * GIB
CONTEXT_ALLOWANCE = 1 * GIB
WEIGHTS_OVERHEAD = 1.05
# bytes/s actually achieved when generating; conservative (a 2022 Intel laptop measured ~20 GB/s,
# a DDR5 desktop gets more), so suggestions lean towards models that feel responsive.
CPU_BANDWIDTH = 30e9
GPU_BANDWIDTH = 300e9  # bytes/s: a mid-range dedicated graphics card
MIN_COMFORTABLE_TPS = 8.0


@dataclass(frozen=True, slots=True)
class Suggestion:
    """A catalog model judged against this computer."""

    model: CatalogModel
    quant: CatalogQuant | None
    est_tokens_per_s: float | None
    installed: bool
    recommended: bool = False

    @property
    def fits(self) -> bool:
        """True if some file of this model fits in this computer's memory."""
        return self.quant is not None


def memory_for_models(hardware: HardwareProfile) -> int:
    """Bytes a model may use on this computer (not counting what is in use right now)."""
    budget = hardware.memory.total_bytes - OS_RESERVE
    gpu = hardware.gpu
    if gpu is not None and not gpu.integrated:
        budget += max(0, gpu.total_bytes - GPU_RESERVE)
    return budget


def needed_bytes(quant: CatalogQuant) -> int:
    """Memory a model file needs with a typical conversation."""
    return round(quant.size_bytes * WEIGHTS_OVERHEAD) + CONTEXT_ALLOWANCE


def estimate_tokens_per_s(model: CatalogModel, quant: CatalogQuant, hw: HardwareProfile) -> float:
    """Rough generation speed for ``quant`` on this computer."""
    gpu = hw.gpu
    on_gpu = gpu is not None and not gpu.integrated and needed_bytes(quant) <= gpu.total_bytes
    bandwidth = GPU_BANDWIDTH if on_gpu else CPU_BANDWIDTH
    bytes_per_token = quant.size_bytes * model.active_params_b / model.params_b
    return bandwidth / bytes_per_token


def _choose_quant(model: CatalogModel, hw: HardwareProfile, budget: int) -> CatalogQuant | None:
    """The first preferred file that fits and feels responsive, else the fastest that fits."""
    fitting = [q for q in model.quants if needed_bytes(q) <= budget]
    for quant in fitting:
        if estimate_tokens_per_s(model, quant, hw) >= MIN_COMFORTABLE_TPS:
            return quant
    return min(fitting, key=lambda q: q.size_bytes, default=None)


def suggest(
    catalog: Sequence[CatalogModel], hardware: HardwareProfile, installed: Collection[str]
) -> list[Suggestion]:
    """Judge every catalog model; ``installed`` holds the model files already downloaded."""
    budget = memory_for_models(hardware)
    suggestions: list[Suggestion] = []
    for model in catalog:
        have = next((q for q in model.quants if q.filename in installed), None)
        chosen = have or _choose_quant(model, hardware, budget)
        speed = estimate_tokens_per_s(model, chosen, hardware) if chosen else None
        suggestions.append(Suggestion(model, chosen, speed, installed=have is not None))
    comfortable = [
        s for s in suggestions
        if s.est_tokens_per_s is not None and s.est_tokens_per_s >= MIN_COMFORTABLE_TPS
    ]  # fmt: skip
    if comfortable:
        best = max(comfortable, key=lambda s: s.model.params_b)
        suggestions = [
            Suggestion(s.model, s.quant, s.est_tokens_per_s, s.installed, recommended=True)
            if s is best else s
            for s in suggestions
        ]  # fmt: skip
    return suggestions
