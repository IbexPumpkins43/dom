"""The models screen's behaviour (Qt-free, so it can be tested with fakes).

Lists GGUF chat models from Hugging Face (most downloaded first, or matching a search), judges
each against this computer, and downloads or deletes them. Models already downloaded are shown
first, so they can be deleted even when the search does not find them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from dom.catalog import CatalogQuant, quant_label, suggest, to_catalog
from dom.catalog.recommend import (
    MIN_COMFORTABLE_TPS,
    estimate_tokens_per_s,
    memory_for_models,
    needed_bytes,
)
from dom.downloads import DownloadError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from dom.catalog import CatalogModel, HubModel
    from dom.downloads.protocols import ModelDownloads, ModelSource
    from dom.hardware import HardwareProfile

logger = logging.getLogger(__name__)

LOCAL_PREFIX = "local:"  # card keys of downloaded files that are not in the search results


class Fit(StrEnum):
    """How a model version suits this computer."""

    GOOD = "good"
    SLOW = "slow"
    TOO_LARGE = "too_large"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Version:
    """One file of a model (a quantisation level) that can be downloaded."""

    path: str
    label: str
    size_bytes: int
    fit: Fit

    @property
    def filename(self) -> str:
        """Its name in the models folder."""
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Everything the screen shows for one model."""

    key: str
    title: str
    author: str | None
    params_b: float | None
    downloads: int | None
    likes: int | None
    versions: tuple[Version, ...]  # smallest first
    selected: str | None  # path of the version that Download would fetch
    installed: str | None  # file name, if a version is downloaded
    fit: Fit
    est_tokens_per_s: float | None
    downloading: bool
    partial_bytes: int

    def version(self, path: str | None) -> Version | None:
        """The version with this path."""
        return next((v for v in self.versions if v.path == path), None)


@dataclass(frozen=True, slots=True)
class ScreenState:
    """The parts of the screen around the cards."""

    query: str
    loading: bool
    error: str | None
    can_load_more: bool
    hardware_text: str


class LibraryView(Protocol):
    """What the library controller tells the models screen to show."""

    def show_cards(self, cards: Sequence[ModelCard], state: ScreenState) -> None:
        """The model cards and the state of the search."""
        ...

    def show_download_progress(self, key: str, done: int, total: int) -> None:
        """Progress of one download."""
        ...

    def show_library_error(self, message: str) -> None:
        """A plain-English problem (e.g. a failed download)."""
        ...


@dataclass(frozen=True, slots=True)
class _Download:
    repo: str
    path: str
    task: asyncio.Task[None]


class LibraryController:
    """Searches Hugging Face, and downloads and deletes models."""

    def __init__(
        self,
        source: ModelSource,
        downloads: ModelDownloads,
        hardware: Callable[[], HardwareProfile],
        on_changed: Callable[[], Awaitable[None]],
        before_delete: Callable[[str], Awaitable[None]],
    ) -> None:
        """``hardware`` returns the best-known profile; ``on_changed`` runs after a download or
        deletion; ``before_delete(model_id)`` releases a model before its file is deleted."""
        self._source = source
        self._downloads = downloads
        self._hardware = hardware
        self._on_changed = on_changed
        self._before_delete = before_delete
        self._view: LibraryView | None = None
        self._query = ""
        self._results: list[HubModel] = []
        self._next_url: str | None = None
        self._loaded = False
        self._error: str | None = None
        self._search_task: asyncio.Task[None] | None = None
        self._choices: dict[str, str] = {}  # card key -> path picked in the version list
        self._active: dict[str, _Download] = {}
        self._cards: list[ModelCard] = []

    def attach(self, view: LibraryView) -> None:
        """Show results in ``view``."""
        self._view = view

    @property
    def cards(self) -> list[ModelCard]:
        """The cards as last shown."""
        return list(self._cards)

    def is_downloading(self, key: str) -> bool:
        """True while a version of card ``key`` is being downloaded."""
        download = self._active.get(key)
        return download is not None and not download.task.done()

    # ---- searching -----------------------------------------------------------------------

    async def open(self) -> None:
        """The screen was opened: list the most popular models the first time, else redraw."""
        if self._loaded or self.is_searching:
            await self.refresh()
        else:
            await self.search("")

    @property
    def is_searching(self) -> bool:
        """True while a search request is running."""
        return self._search_task is not None and not self._search_task.done()

    async def search(self, query: str) -> None:
        """Show models matching ``query`` (replacing any search still running)."""
        if self._search_task is not None and not self._search_task.done():
            self._search_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._search_task
        self._query = query.strip()
        self._search_task = asyncio.create_task(self._fetch(None))
        await self.refresh()
        with contextlib.suppress(asyncio.CancelledError):
            await self._search_task
        await self.refresh()

    async def load_more(self) -> None:
        """Add the next page of results."""
        if self._next_url is None or self.is_searching:
            return
        self._search_task = asyncio.create_task(self._fetch(self._next_url))
        await self.refresh()
        with contextlib.suppress(asyncio.CancelledError):
            await self._search_task
        await self.refresh()

    async def _fetch(self, page_url: str | None) -> None:
        try:
            if page_url is None:
                page = await self._source.search(self._query)
            else:
                page = await self._source.page(page_url)
        except DownloadError as exc:
            logger.warning("Model search failed: %s", exc)
            self._error = str(exc)
            return
        self._error = None
        self._loaded = True
        if page_url is None:
            self._results = list(page.models)
        else:
            known = {m.repo for m in self._results}
            self._results += [m for m in page.models if m.repo not in known]
        self._next_url = page.next_url

    # ---- choosing, downloading and deleting -------------------------------------------------

    async def choose_version(self, key: str, path: str) -> None:
        """Download ``path`` (instead of the suggested version) for card ``key``."""
        self._choices[key] = path
        await self.refresh()

    async def download(self, key: str) -> None:
        """Download the chosen (or suggested) version of card ``key``, resuming if possible."""
        card = next((c for c in self._cards if c.key == key), None)
        if card is None or card.installed is not None or self.is_downloading(key):
            return
        version = card.version(card.selected)
        if version is None or version.fit == Fit.TOO_LARGE:
            return
        task = asyncio.create_task(self._download(key, version.path))
        self._active[key] = _Download(key, version.path, task)
        await self.refresh()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._active.pop(key, None)
        await self.refresh()

    async def cancel(self, key: str) -> None:
        """Stop a download; the part already downloaded is kept for resuming."""
        download = self._active.get(key)
        if download is not None and not download.task.done():
            download.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await download.task

    async def delete(self, filename: str) -> None:
        """Delete a downloaded model (unloading it first if it is in use)."""
        await self._before_delete(filename.removesuffix(".gguf"))
        try:
            await self._downloads.delete(filename)
        except DownloadError as exc:
            self._report(str(exc))
        await self._on_changed()
        await self.refresh()

    async def shutdown(self) -> None:
        """Stop every download (partial files are kept) and any search."""
        for key in list(self._active):
            await self.cancel(key)
        if self._search_task is not None:
            self._search_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._search_task

    async def _download(self, key: str, path: str) -> None:
        def progress(done: int, total: int) -> None:
            if self._view is not None:
                self._view.show_download_progress(key, done, total)

        try:
            await self._downloads.download(key, path, progress)
        except DownloadError as exc:
            logger.warning("Download of %s failed: %s", path, exc)
            self._report(str(exc))
            return
        await self._on_changed()

    def _report(self, message: str) -> None:
        if self._view is not None:
            self._view.show_library_error(message)

    # ---- building the cards ----------------------------------------------------------------

    async def refresh(self) -> None:
        """Recompute the cards and redraw the screen."""
        hardware = self._hardware()
        installed = await self._downloads.installed_files()
        sources = await self._downloads.sources()
        cards: list[ModelCard] = []
        shown_files: set[str] = set()
        for model in self._results:
            card = await self._hub_card(model, hardware, installed)
            if card is not None:
                cards.append(card)
                shown_files.update(v.filename for v in card.versions)
        local = [
            self._local_card(name, sources.get(name))
            for name in sorted(installed - shown_files)
            if self._matches(name, sources.get(name))
        ]
        # Downloaded models first, so they are easy to find (and delete).
        cards = local + sorted(cards, key=lambda c: c.installed is None)
        self._cards = cards
        if self._view is not None:
            state = ScreenState(
                query=self._query,
                loading=self.is_searching,
                error=self._error,
                can_load_more=self._next_url is not None,
                hardware_text=hardware.summary(),
            )
            self._view.show_cards(cards, state)

    def _matches(self, filename: str, repo: str | None) -> bool:
        words = self._query.lower().split()
        text = f"{filename} {repo or ''}".lower()
        return all(word in text for word in words)

    async def _hub_card(
        self, model: HubModel, hardware: HardwareProfile, installed: set[str]
    ) -> ModelCard | None:
        catalog = to_catalog(model)
        if catalog is None:
            return None
        [suggestion] = suggest([catalog], hardware, installed)
        budget = memory_for_models(hardware)
        versions = tuple(
            sorted(
                (self._version(catalog, q, hardware, budget) for q in catalog.quants),
                key=lambda v: v.size_bytes,
            )
        )
        installed_version = next((v for v in versions if v.filename in installed), None)
        download = self._active.get(model.repo)
        selected = (
            installed_version.path
            if installed_version is not None
            else download.path
            if download is not None
            else self._choices.get(model.repo)
            or (suggestion.quant or catalog.quants[0]).filename  # too large: the usual pick
        )
        chosen = next((v for v in versions if v.path == selected), None)
        partial = 0
        if chosen is not None and installed_version is None:
            partial = await self._downloads.partial_bytes(chosen.filename)
        if chosen is not None:
            fit = chosen.fit
            quant = next(q for q in catalog.quants if q.filename == chosen.path)
            speed = estimate_tokens_per_s(catalog, quant, hardware)
        else:
            fit, speed = Fit.TOO_LARGE, None
        return ModelCard(
            key=model.repo,
            title=model.name,
            author=model.author,
            params_b=catalog.params_b,
            downloads=model.downloads,
            likes=model.likes,
            versions=versions,
            selected=selected,
            installed=installed_version.filename if installed_version else None,
            fit=fit,
            est_tokens_per_s=speed,
            downloading=self.is_downloading(model.repo),
            partial_bytes=partial,
        )

    @staticmethod
    def _version(
        model: CatalogModel, quant: CatalogQuant, hardware: HardwareProfile, budget: int
    ) -> Version:
        if needed_bytes(quant) > budget:
            fit = Fit.TOO_LARGE
        elif estimate_tokens_per_s(model, quant, hardware) < MIN_COMFORTABLE_TPS:
            fit = Fit.SLOW
        else:
            fit = Fit.GOOD
        return Version(quant.filename, quant.label, quant.size_bytes, fit)

    @staticmethod
    def _local_card(filename: str, repo: str | None) -> ModelCard:
        label = quant_label(filename) or "GGUF"
        version = Version(filename, label, 0, Fit.UNKNOWN)
        title = filename.removesuffix(".gguf")
        return ModelCard(
            key=LOCAL_PREFIX + filename,
            title=title,
            author=repo.split("/", 1)[0] if repo else None,
            params_b=None,
            downloads=None,
            likes=None,
            versions=(version,),
            selected=filename,
            installed=filename,
            fit=Fit.UNKNOWN,
            est_tokens_per_s=None,
            downloading=False,
            partial_bytes=0,
        )
