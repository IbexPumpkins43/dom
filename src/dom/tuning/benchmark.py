"""A short llama-bench run to decide between CPU and an integrated GPU, cached per model."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from dom.config.binary import run_engine_tool
from dom.jsonshape import is_array, is_object
from dom.tuning.tuner import Placement

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# What a "typical turn" looks like when comparing speeds: reading this much conversation and
# writing this long a reply.
TYPICAL_PROMPT_TOKENS = 1024
TYPICAL_REPLY_TOKENS = 256
# What the benchmark itself runs (kept short: it runs once per model on this computer).
BENCH_PROMPT_TOKENS = 256
BENCH_GEN_TOKENS = 32
BENCH_TIMEOUT_S = 300.0


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Measured speeds, in tokens per second."""

    prompt_tps: float
    gen_tps: float

    def turn_seconds(self) -> float:
        """Estimated time for a typical turn."""
        return TYPICAL_PROMPT_TOKENS / self.prompt_tps + TYPICAL_REPLY_TOKENS / self.gen_tps


def parse_bench_output(text: str) -> BenchmarkResult | None:
    """Parse ``llama-bench -o json`` output (one prompt test and one generation test)."""
    try:
        payload: object = json.loads(text)
    except ValueError:
        return None
    if not is_array(payload):
        return None
    prompt_tps = gen_tps = 0.0
    for row in payload:
        if not is_object(row):
            continue
        speed = row.get("avg_ts")
        if not isinstance(speed, int | float) or speed <= 0:
            continue
        if row.get("n_gen") == 0:
            prompt_tps = float(speed)
        elif row.get("n_prompt") == 0:
            gen_tps = float(speed)
    if prompt_tps <= 0 or gen_tps <= 0:
        return None
    return BenchmarkResult(prompt_tps=prompt_tps, gen_tps=gen_tps)


def choose_faster(gpu: BenchmarkResult | None, cpu: BenchmarkResult | None) -> Placement:
    """GPU unless the CPU is measurably faster for a typical turn (or the GPU run failed)."""
    if gpu is None:
        return Placement.CPU if cpu is not None else Placement.GPU
    if cpu is None:
        return Placement.GPU
    return Placement.CPU if cpu.turn_seconds() < gpu.turn_seconds() else Placement.GPU


class LlamaBench:
    """Runs the ``llama-bench`` that ships next to ``llama-server``."""

    def __init__(self, binary: Path) -> None:
        """Use the ``llama-bench`` executable at ``binary``."""
        self._binary = binary

    @property
    def available(self) -> bool:
        """True if the executable exists."""
        return self._binary.is_file()

    async def run(self, model_path: Path, placement: Placement) -> BenchmarkResult | None:
        """Benchmark ``model_path`` on the GPU or CPU; ``None`` if the run failed."""
        return await asyncio.to_thread(self._run, model_path, placement)

    def _run(self, model_path: Path, placement: Placement) -> BenchmarkResult | None:
        command = [
            str(self._binary),
            "-m", str(model_path),
            "-p", str(BENCH_PROMPT_TOKENS),
            "-n", str(BENCH_GEN_TOKENS),
            "-r", "1",
            "-o", "json",
        ]  # fmt: skip
        if placement == Placement.CPU:
            command += ["-dev", "none"]
        logger.info("Benchmarking %s on %s", model_path.name, placement)
        try:
            result = run_engine_tool(command, BENCH_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("llama-bench failed for %s on %s: %s", model_path.name, placement, exc)
            return None
        parsed = parse_bench_output(result.stdout)
        if parsed is None:
            logger.warning(
                "llama-bench gave no result for %s on %s (exit %s): %s",
                model_path.name, placement, result.returncode, result.stderr.strip()[-500:],
            )  # fmt: skip
        else:
            logger.info("%s on %s: %s", model_path.name, placement, parsed)
        return parsed


class BenchmarkCache:
    """Remembers the CPU/GPU decision per model and GPU, in a small JSON file."""

    def __init__(self, path: Path) -> None:
        """Use the cache file at ``path`` (created on first save)."""
        self._path = path
        self._lock = threading.Lock()

    @staticmethod
    def key(model_path: Path, gpu_name: str, build_version: str) -> str:
        """Cache key: the exact model file, GPU and llama.cpp build."""
        try:
            size = model_path.stat().st_size
        except OSError:
            size = -1
        return f"{model_path.name}|{size}|{gpu_name}|{build_version}"

    def get(self, key: str) -> Placement | None:
        """The cached decision, if any."""
        entry = self._load().get(key)
        if is_object(entry) and isinstance(choice := entry.get("placement"), str):
            try:
                return Placement(choice)
            except ValueError:
                return None
        return None

    def put(
        self,
        key: str,
        placement: Placement,
        gpu: BenchmarkResult | None,
        cpu: BenchmarkResult | None,
    ) -> None:
        """Store a decision (and the measurements behind it)."""
        with self._lock:
            data = self._load()
            data[key] = {
                "placement": placement.value,
                "gpu": asdict(gpu) if gpu else None,
                "cpu": asdict(cpu) if cpu else None,
            }
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_name(self._path.name + ".tmp")
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                os.replace(tmp, self._path)
            except OSError as exc:
                logger.warning("Could not save benchmark results to %s: %s", self._path, exc)

    def _load(self) -> dict[str, object]:
        try:
            payload: object = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return payload if is_object(payload) else {}
