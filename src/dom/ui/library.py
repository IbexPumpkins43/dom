"""The "Get models" screen's behaviour (Qt-free, so it can be tested with fakes)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from dom.catalog import suggest
from dom.downloads import DownloadError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from dom.catalog import CatalogModel, Suggestion
    from dom.downloads.protocols import ModelDownloads
    from dom.hardware import HardwareProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CatalogRow:
    """One model on the screen."""

    suggestion: Suggestion
    downloading: bool
    partial_bytes: int


class LibraryView(Protocol):
    """What the library controller can tell the "Get models" screen to show."""

    def show_catalog(self, rows: Sequence[CatalogRow], hardware_text: str) -> None:
        """Every catalog model and how it suits this computer."""
        ...

    def show_download_progress(self, key: str, done: int, total: int) -> None:
        """Progress of one download."""
        ...

    def show_library_error(self, message: str) -> None:
        """A plain-English problem (e.g. a failed download)."""
        ...


class LibraryController:
    """Suggests catalog models for this computer and downloads them."""

    def __init__(
        self,
        catalog: Sequence[CatalogModel],
        downloads: ModelDownloads,
        hardware: Callable[[], HardwareProfile],
        on_installed: Callable[[], Awaitable[None]],
    ) -> None:
        """``hardware`` returns the best-known profile; ``on_installed`` runs after a download."""
        self._catalog = catalog
        self._downloads = downloads
        self._hardware = hardware
        self._on_installed = on_installed
        self._view: LibraryView | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def attach(self, view: LibraryView) -> None:
        """Show results in ``view``."""
        self._view = view

    def is_downloading(self, key: str) -> bool:
        """True while ``key`` is being downloaded."""
        task = self._tasks.get(key)
        return task is not None and not task.done()

    async def refresh(self) -> None:
        """Recompute suggestions and redraw the screen."""
        if self._view is None:
            return
        hardware = self._hardware()
        installed = await self._downloads.installed_files()
        rows: list[CatalogRow] = []
        for item in suggest(self._catalog, hardware, installed):
            partial = 0
            if item.quant is not None and not item.installed:
                partial = await self._downloads.partial_bytes(item.quant.filename)
            rows.append(CatalogRow(item, self.is_downloading(item.model.key), partial))
        self._view.show_catalog(rows, hardware.summary())

    async def download(self, key: str) -> None:
        """Download the suggested file of catalog model ``key`` (resuming if possible)."""
        if self.is_downloading(key):
            return
        hardware = self._hardware()
        installed = await self._downloads.installed_files()
        item = next(
            (s for s in suggest(self._catalog, hardware, installed) if s.model.key == key), None
        )
        if item is None or item.quant is None or item.installed:
            return
        task = asyncio.create_task(self._download(key, item.quant.filename))
        self._tasks[key] = task
        await self.refresh()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await self.refresh()

    async def cancel(self, key: str) -> None:
        """Stop a download; the part already downloaded is kept for resuming."""
        task = self._tasks.get(key)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def shutdown(self) -> None:
        """Stop every download (partial files are kept)."""
        for key in list(self._tasks):
            await self.cancel(key)

    async def _download(self, key: str, filename: str) -> None:
        def progress(done: int, total: int) -> None:
            if self._view is not None:
                self._view.show_download_progress(key, done, total)

        try:
            await self._downloads.download(key, filename, progress)
        except DownloadError as exc:
            logger.warning("Download of %s failed: %s", filename, exc)
            if self._view is not None:
                self._view.show_library_error(str(exc))
            return
        await self._on_installed()
