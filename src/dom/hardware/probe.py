"""Detecting RAM, CPU cores and the GPUs llama.cpp can use."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import subprocess
from typing import TYPE_CHECKING

import psutil

from dom.hardware.models import MIB, GpuDevice, HardwareProfile, MemoryStatus

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dom.config import ServerBuild

logger = logging.getLogger(__name__)

LIST_DEVICES_TIMEOUT_S = 30.0
CPU_BUILD_NAME = "cpu"

# "  Vulkan0: Intel(R) Iris(R) Xe Graphics (ADL GT2) (11382 MiB, 7653 MiB free)"
_DEVICE_LINE = re.compile(
    r"^\s*(?P<id>\S+):\s+(?P<name>.+?)\s+\((?P<total>\d+) MiB, (?P<free>\d+) MiB free\)\s*$"
)
# Software renderers are CPUs pretending to be GPUs; never offload to them.
_SOFTWARE_RENDERERS = re.compile(r"llvmpipe|lavapipe|swiftshader|microsoft basic render", re.I)
# Names of GPUs built into the CPU. llama.cpp does not report this, so it is a heuristic; it only
# decides whether dom runs a short speed test, so a wrong guess costs a few seconds, not quality.
_INTEGRATED_NAMES = re.compile(
    r"adreno|iris|uhd graphics|hd graphics|intel\(r\) graphics|intel\(r\) arc\(tm\) graphics$"
    r"|radeon\(tm\) graphics|radeon graphics|radeon \d{3,4}m|vega \d+ graphics|apple m\d",
    re.I,
)
# Names of dedicated graphics cards; checked first so a big card is never called integrated.
_DISCRETE_NAMES = re.compile(
    r"nvidia|geforce|rtx|gtx|quadro|tesla|radeon rx|radeon pro|instinct|arc\(tm\) [ab]\d{3}",
    re.I,
)
# An unrecognised GPU that can use most of system RAM is sharing it, i.e. integrated.
_SHARED_MEMORY_SHARE = 0.5

type MemoryReader = Callable[[], MemoryStatus]
type DeviceLister = Callable[[ServerBuild], list[tuple[str, str, int, int]]]


def parse_device_list(text: str) -> list[tuple[str, str, int, int]]:
    """Parse ``--list-devices`` output into ``(id, name, total_bytes, free_bytes)`` tuples."""
    devices: list[tuple[str, str, int, int]] = []
    for line in text.splitlines():
        match = _DEVICE_LINE.match(line)
        if match is None or _SOFTWARE_RENDERERS.search(match["name"]):
            continue
        devices.append(
            (match["id"], match["name"], int(match["total"]) * MIB, int(match["free"]) * MIB)
        )
    return devices


def is_integrated_gpu(name: str, total_bytes: int, ram_total_bytes: int) -> bool:
    """Guess whether a GPU shares system RAM (built into the CPU)."""
    if _DISCRETE_NAMES.search(name):
        return False
    if _INTEGRATED_NAMES.search(name.strip()):
        return True
    return ram_total_bytes > 0 and total_bytes >= ram_total_bytes * _SHARED_MEMORY_SHARE


def choose_build(
    builds: Sequence[ServerBuild], gpus_by_build: dict[str, list[GpuDevice]]
) -> tuple[ServerBuild, list[GpuDevice]]:
    """Use the first build that can see a GPU; otherwise the CPU build (or the first one)."""
    for build in builds:
        gpus = gpus_by_build.get(build.name, [])
        if gpus:
            return build, gpus
    cpu = next((b for b in builds if b.name == CPU_BUILD_NAME), builds[0])
    return cpu, []


def read_memory() -> MemoryStatus:
    """Current system RAM from psutil."""
    vm = psutil.virtual_memory()
    return MemoryStatus(total_bytes=vm.total, available_bytes=vm.available)


def list_devices(build: ServerBuild) -> list[tuple[str, str, int, int]]:
    """Run ``<build> --list-devices``; any failure means "no GPUs" (logged, not raised)."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("LLAMA_")}
    try:
        result = subprocess.run(
            [str(build.path), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=LIST_DEVICES_TIMEOUT_S,
            env=env,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not list devices of %s: %s", build.path, exc)
        return []
    if result.returncode != 0:
        logger.warning(
            "%s --list-devices failed (%s): %s",
            build.path,
            result.returncode,
            result.stderr.strip()[-500:],
        )
        return []
    return parse_device_list(result.stdout + "\n" + result.stderr)


class HardwareProbe:
    """Detects the hardware and picks the llama-server build to run."""

    def __init__(
        self,
        memory_reader: MemoryReader = read_memory,
        device_lister: DeviceLister = list_devices,
    ) -> None:
        """Use the given readers (tests pass fakes)."""
        self._memory_reader = memory_reader
        self._device_lister = device_lister

    def basic(self) -> HardwareProfile:
        """RAM and CPU only, for when no llama-server build is installed yet."""
        return HardwareProfile(
            os_name=platform.system(),
            arch=platform.machine(),
            physical_cores=psutil.cpu_count(logical=False),
            logical_cores=psutil.cpu_count(logical=True),
            memory=self._memory_reader(),
            gpus=(),
            build=None,
        )

    def memory(self) -> MemoryStatus:
        """Current RAM (fast; call it right before loading a model)."""
        return self._memory_reader()

    async def detect(self, builds: Sequence[ServerBuild]) -> HardwareProfile:
        """Probe RAM, CPU and each build's GPUs without blocking the event loop."""
        memory = await asyncio.to_thread(self._memory_reader)
        gpus_by_build: dict[str, list[GpuDevice]] = {}
        for build in builds:
            raw = await asyncio.to_thread(self._device_lister, build)
            gpus_by_build[build.name] = [
                GpuDevice(
                    dev_id, name, total, free, is_integrated_gpu(name, total, memory.total_bytes)
                )
                for dev_id, name, total, free in raw
            ]
        build, gpus = choose_build(builds, gpus_by_build)
        profile = HardwareProfile(
            os_name=platform.system(),
            arch=platform.machine(),
            physical_cores=psutil.cpu_count(logical=False),
            logical_cores=psutil.cpu_count(logical=True),
            memory=memory,
            gpus=tuple(gpus),
            build=build,
        )
        logger.info("Hardware: %s; using the %s build", profile.summary(), build.name)
        return profile
