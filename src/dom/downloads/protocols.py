"""Download interfaces the UI depends on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable


class ModelDownloads(Protocol):
    """Downloads catalog models into the models folder."""

    async def installed_files(self) -> set[str]:
        """Names of the model files already present."""
        ...

    async def partial_bytes(self, filename: str) -> int:
        """Bytes of an unfinished earlier download of ``filename``."""
        ...

    async def download(
        self, model_key: str, filename: str, progress: Callable[[int, int], None]
    ) -> None:
        """Download (or resume) one file; cancel the task to pause."""
        ...


class EngineDownloads(Protocol):
    """Installs llama.cpp when it is missing."""

    async def install(self, progress: Callable[[str, int, int], None]) -> None:
        """Download and unpack the engine builds."""
        ...
