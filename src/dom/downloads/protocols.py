"""Download interfaces the UI depends on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from dom.downloads.hub import HubPage, ParamRange


class ModelSource(Protocol):
    """Finds models to download (Hugging Face search)."""

    async def search(self, query: str, size: ParamRange = (None, None)) -> HubPage:
        """The first page of models matching ``query`` ("" for the most popular) with ``size``
        (billions of parameters: at least, at most)."""
        ...

    async def page(self, url: str) -> HubPage:
        """A further page of results (a page's ``next_url``)."""
        ...


class ModelDownloads(Protocol):
    """The models folder: downloads into it and deletions from it."""

    async def installed_files(self) -> dict[str, int]:
        """The model files already present: name -> size in bytes."""
        ...

    async def sources(self) -> dict[str, str]:
        """Downloaded file name -> the repository it came from."""
        ...

    async def partial_bytes(self, filename: str) -> int:
        """Bytes of an unfinished earlier download of ``filename``."""
        ...

    async def download(self, repo: str, path: str, progress: Callable[[int, int], None]) -> None:
        """Download (or resume) file ``path`` of ``repo``; cancel the task to pause."""
        ...

    async def delete(self, filename: str) -> None:
        """Delete a downloaded model file."""
        ...


class EngineDownloads(Protocol):
    """Installs llama.cpp when it is missing."""

    async def install(self, progress: Callable[[str, int, int], None]) -> None:
        """Download and unpack the engine builds."""
        ...
