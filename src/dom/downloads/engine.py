"""Downloading llama.cpp ("the AI engine") when it is missing."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from dom.downloads.http import download_verified, make_client
from dom.downloads.llama_release import detect_target, find_releases, install_release, is_installed

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import httpx

logger = logging.getLogger(__name__)

type EngineProgress = Callable[[str, int, int], None]


class EngineInstaller:
    """Installs the CPU and Vulkan builds of the latest llama.cpp release into ``bin_dir``."""

    def __init__(
        self, bin_dir: Path, client_factory: Callable[[], httpx.AsyncClient] = make_client
    ) -> None:
        """Install into ``bin_dir``; ``client_factory`` lets tests fake the network."""
        self._bin_dir = bin_dir
        self._client_factory = client_factory

    async def install(self, progress: EngineProgress) -> None:
        """Download, verify and unpack every build (``progress(label, done, total)``).

        Raises:
            DownloadError: with a plain-English message.
        """
        downloads = self._bin_dir / ".downloads"
        async with self._client_factory() as client:
            for release in await find_releases(client, detect_target()):
                if await asyncio.to_thread(is_installed, self._bin_dir, release):
                    continue
                label = f"{release.variant.value.upper()} engine"
                archive = downloads / release.asset.name
                await download_verified(
                    client, release.asset, archive, lambda d, t, lbl=label: progress(lbl, d, t)
                )
                dest = await asyncio.to_thread(install_release, archive, self._bin_dir, release)
                await asyncio.to_thread(archive.unlink)
                logger.info("Installed llama.cpp %s into %s", release.version_label, dest)
        await asyncio.to_thread(_remove_if_empty, downloads)


def _remove_if_empty(directory: Path) -> None:
    with contextlib.suppress(OSError):
        directory.rmdir()
