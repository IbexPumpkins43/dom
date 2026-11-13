"""Tuning layer: choosing per-model settings and CPU-vs-GPU placement for this computer."""

from dom.tuning.benchmark import (
    BenchmarkCache,
    BenchmarkResult,
    LlamaBench,
    choose_faster,
    parse_bench_output,
)
from dom.tuning.tuner import Machine, ModelPlan, Placement, TooLarge, plan_model

__all__ = [
    "BenchmarkCache",
    "BenchmarkResult",
    "LlamaBench",
    "Machine",
    "ModelPlan",
    "Placement",
    "TooLarge",
    "choose_faster",
    "parse_bench_output",
    "plan_model",
]
