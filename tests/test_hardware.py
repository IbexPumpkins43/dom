"""Tests for hardware detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from dom.config import ServerBuild
from dom.hardware import (
    GpuDevice,
    HardwareProbe,
    MemoryStatus,
    choose_build,
    is_integrated_gpu,
    parse_device_list,
)
from dom.hardware.models import GIB, MIB

LIST_DEVICES_OUTPUT = """\
load_backend: loaded Vulkan backend from libggml-vulkan.so
Available devices:
  Vulkan0: Intel(R) Iris(R) Xe Graphics (ADL GT2) (11382 MiB, 7653 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4070 (12282 MiB, 11800 MiB free)
  Vulkan2: llvmpipe (LLVM 19.1.0, 256 bits) (15000 MiB, 15000 MiB free)
"""


def test_parse_device_list() -> None:
    assert parse_device_list(LIST_DEVICES_OUTPUT) == [
        ("Vulkan0", "Intel(R) Iris(R) Xe Graphics (ADL GT2)", 11382 * MIB, 7653 * MIB),
        ("Vulkan1", "NVIDIA GeForce RTX 4070", 12282 * MIB, 11800 * MIB),
    ]


def test_parse_device_list_with_no_devices() -> None:
    assert parse_device_list("Available devices:\n  (none)\n") == []


@pytest.mark.parametrize(
    ("name", "total_gib", "expected"),
    [
        ("Intel(R) Iris(R) Xe Graphics (ADL GT2)", 11, True),
        ("Intel(R) UHD Graphics 620", 4, True),
        ("Intel(R) Arc(TM) Graphics", 8, True),
        ("Intel(R) Arc(TM) A770 Graphics", 16, False),
        ("Qualcomm(R) Adreno(TM) X1-85 GPU", 12, True),
        ("AMD Radeon(TM) Graphics", 8, True),
        ("AMD Radeon 780M", 8, True),
        ("NVIDIA GeForce RTX 4070", 12, False),
        ("AMD Radeon RX 7900 XTX", 24, False),
        ("Some Unknown GPU", 20, True),  # can use most of the 32 GB RAM: shared memory
    ],
)
def test_is_integrated_gpu(name: str, total_gib: int, expected: bool) -> None:
    assert is_integrated_gpu(name, total_gib * GIB, 32 * GIB) is expected


CPU = ServerBuild("cpu", Path("/bin/cpu/llama-server"))
VULKAN = ServerBuild("vulkan", Path("/bin/vulkan/llama-server"))
GPU = GpuDevice("Vulkan0", "RTX", 8 * GIB, 7 * GIB, integrated=False)


def test_choose_build_prefers_a_build_with_a_gpu() -> None:
    assert choose_build([CPU, VULKAN], {"cpu": [], "vulkan": [GPU]}) == (VULKAN, [GPU])


def test_choose_build_falls_back_to_cpu() -> None:
    assert choose_build([VULKAN, CPU], {"cpu": [], "vulkan": []}) == (CPU, [])


def test_choose_build_uses_the_only_build() -> None:
    other = ServerBuild("path", Path("/usr/bin/llama-server"))
    assert choose_build([other], {}) == (other, [])


async def test_probe_builds_a_profile() -> None:
    def lister(build: ServerBuild) -> list[tuple[str, str, int, int]]:
        if build.name == "vulkan":
            return [("Vulkan0", "Intel(R) Iris(R) Xe Graphics", 11 * GIB, 7 * GIB)]
        return []

    probe = HardwareProbe(
        memory_reader=lambda: MemoryStatus(16 * GIB, 9 * GIB), device_lister=lister
    )
    profile = await probe.detect([CPU, VULKAN])
    assert profile.build == VULKAN
    assert profile.gpu is not None
    assert profile.gpu.integrated
    assert profile.memory.available_bytes == 9 * GIB
    assert "Iris" in profile.summary()
    assert "16 GB RAM" in profile.summary()
