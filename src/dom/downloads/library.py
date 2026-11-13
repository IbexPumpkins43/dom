"""Downloading catalog models into the models folder."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from dom.downloads.http import DownloadError, download_verified, make_client
from dom.downloads.huggingface import resolve_file

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import httpx

    from dom.catalog import CatalogModel
    from dom.downloads.http import Progress

logger = logging.getLogger(__name__)


class ModelLibrary:
    """The models folder, and downloads of catalog models into it."""

    def __init__(
        self,
        models_dir: Path,
        catalog: Sequence[CatalogModel],
        client_factory: Callable[[], httpx.AsyncClient] = make_client,
    ) -> None:
        """Manage ``models_dir``; ``client_factory`` lets tests fake the network."""
        self._models_dir = models_dir
        self._catalog = {model.key: model for model in catalog}
        self._client_factory = client_factory

    @property
    def models_dir(self) -> Path:
        """Where model files live."""
        return self._models_dir

    async def installed_files(self) -> set[str]:
        """Names of the ``.gguf`` files in the models folder."""

        def scan() -> set[str]:
            if not self._models_dir.is_dir():
                return set()
            return {p.name for p in self._models_dir.iterdir() if p.suffix == ".gguf"}

        return await asyncio.to_thread(scan)

    async def partial_bytes(self, filename: str) -> int:
        """Bytes of an earlier, unfinished download of ``filename`` (0 if none)."""
        part = self._models_dir / (filename + ".part")
        return await asyncio.to_thread(lambda: part.stat().st_size if part.is_file() else 0)

    async def download(self, model_key: str, filename: str, progress: Progress) -> None:
        """Download one file of a catalog model (resuming and verifying).

        Raises:
            DownloadError: with a plain-English message.
        """
        model = self._catalog.get(model_key)
        if model is None or not any(q.filename == filename for q in model.quants):
            raise DownloadError(f"{filename} is not a file dom knows how to download.")
        logger.info("Downloading %s from %s", filename, model.repo)
        async with self._client_factory() as client:
            remote = await resolve_file(client, model.repo, filename)
            await download_verified(client, remote, self._models_dir / filename, progress)
        logger.info("Downloaded and verified %s", filename)
