"""Hardware description objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dom.config import ServerBuild

GIB = 1024**3
MIB = 1024**2


@dataclass(frozen=True, slots=True)
class MemoryStatus:
    """System RAM, in bytes."""

    total_bytes: int
    available_bytes: int


@dataclass(frozen=True, slots=True)
class GpuDevice:
    """A GPU llama.cpp can offload to, as reported by ``llama-server --list-devices``."""

    id: str
    name: str
    total_bytes: int
    free_bytes: int
    integrated: bool


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """What dom detected about this computer, and the llama-server build it chose."""

    os_name: str
    arch: str
    physical_cores: int | None
    logical_cores: int | None
    memory: MemoryStatus
    gpus: tuple[GpuDevice, ...]
    build: ServerBuild

    @property
    def gpu(self) -> GpuDevice | None:
        """The GPU dom uses (the first one), if any."""
        return self.gpus[0] if self.gpus else None

    def summary(self) -> str:
        """A short plain-English description, for the UI."""
        cores = f"{self.physical_cores} CPU cores" if self.physical_cores else "CPU"
        ram = f"{self.memory.total_bytes / GIB:.0f} GB RAM"
        parts = [f"{self.os_name} {self.arch}", cores, ram]
        for gpu in self.gpus:
            kind = "integrated GPU" if gpu.integrated else f"{gpu.total_bytes / GIB:.0f} GB GPU"
            parts.append(f"{gpu.name} ({kind})")
        if not self.gpus:
            parts.append("no supported GPU")
        return ", ".join(parts)
