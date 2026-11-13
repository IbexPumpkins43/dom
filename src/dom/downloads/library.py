"""The models folder: downloading GGUF files from Hugging Face into it, and deleting them."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING

from dom.downloads.http import DownloadError, download_verified, make_client
from dom.downloads.huggingface import resolve_file
from dom.jsonshape import is_object

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import httpx

    from dom.downloads.http import Progress

logger = logging.getLogger(__name__)

# Which Hugging Face repository each downloaded file came from (for the models screen).
SOURCES_FILE = ".dom-sources.json"


def local_name(path: str) -> str:
    """The name a repository file gets in the models folder (its base name)."""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _check_name(filename: str) -> None:
    if (
        not filename.endswith(".gguf")
        or filename != local_name(filename)
        or filename.startswith(".")
    ):
        raise DownloadError(f"{filename} is not a model file dom can manage.")


class ModelLibrary:
    """The models folder, downloads into it and deletions from it."""

    def __init__(
        self,
        models_dir: Path,
        client_factory: Callable[[], httpx.AsyncClient] = make_client,
    ) -> None:
        """Manage ``models_dir``; ``client_factory`` lets tests fake the network."""
        self._models_dir = models_dir
        self._client_factory = client_factory
        self._sources_lock = asyncio.Lock()

    @property
    def models_dir(self) -> Path:
        """Where model files live."""
        return self._models_dir

    async def installed_files(self) -> dict[str, int]:
        """The ``.gguf`` files in the models folder: name -> size in bytes."""

        def scan() -> dict[str, int]:
            if not self._models_dir.is_dir():
                return {}
            return {
                p.name: p.stat().st_size
                for p in self._models_dir.iterdir()
                if p.suffix == ".gguf" and p.is_file()
            }

        return await asyncio.to_thread(scan)

    async def sources(self) -> dict[str, str]:
        """Downloaded file name -> the Hugging Face repository it came from."""
        return await asyncio.to_thread(self._read_sources)

    async def partial_bytes(self, filename: str) -> int:
        """Bytes of an earlier, unfinished download of ``filename`` (0 if none)."""
        part = self._models_dir / (local_name(filename) + ".part")
        return await asyncio.to_thread(lambda: part.stat().st_size if part.is_file() else 0)

    async def download(self, repo: str, path: str, progress: Progress) -> None:
        """Download ``path`` from Hugging Face repository ``repo`` (resuming and verifying).

        Raises:
            DownloadError: with a plain-English message.
        """
        filename = local_name(path)
        _check_name(filename)
        logger.info("Downloading %s from %s", path, repo)
        async with self._client_factory() as client:
            remote = await resolve_file(client, repo, path)
            await download_verified(client, remote, self._models_dir / filename, progress)
        await self._set_source(filename, repo)
        logger.info("Downloaded and verified %s", filename)

    async def delete(self, filename: str) -> None:
        """Delete a model file (and any unfinished download of it).

        Raises:
            DownloadError: if it could not be deleted (e.g. it is still in use).
        """
        _check_name(filename)

        def remove() -> None:
            for path in (self._models_dir / filename, self._models_dir / (filename + ".part")):
                path.unlink(missing_ok=True)

        try:
            await asyncio.to_thread(remove)
        except OSError as exc:
            raise DownloadError(f"Could not delete {filename}: {exc.strerror or exc}") from exc
        await self._set_source(filename, None)
        logger.info("Deleted %s", filename)

    def _read_sources(self) -> dict[str, str]:
        try:
            raw: object = json.loads((self._models_dir / SOURCES_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not is_object(raw):
            return {}
        return {k: v for k, v in raw.items() if isinstance(v, str)}

    async def _set_source(self, filename: str, repo: str | None) -> None:
        def update() -> None:
            sources = self._read_sources()
            if repo is None:
                sources.pop(filename, None)
            else:
                sources[filename] = repo
            target = self._models_dir / SOURCES_FILE
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(json.dumps(sources, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(temporary, target)

        async with self._sources_lock:
            try:
                await asyncio.to_thread(update)
            except OSError:
                logger.warning("Could not update %s", SOURCES_FILE, exc_info=True)
