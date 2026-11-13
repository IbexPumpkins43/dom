"""Download the development/test assets into ``.dev/``.

Fetches two things, and nothing else:

1. The ``llama-server`` CPU and Vulkan (GPU) builds for this OS/CPU from the ggml-org/llama.cpp
   GitHub releases, unpacked into ``.dev/bin/cpu/`` and ``.dev/bin/vulkan/`` with the release tag
   written to ``VERSION`` in each.
2. The Q4_K_M and Q8_0 quants of ``unsloth/Qwen3.5-0.8B-GGUF`` from Hugging Face, into
   ``.dev/models/``.

Everything is checked against the SHA-256 the source publishes; interrupted downloads resume and
assets that are already present and verified are skipped. The download code is shared with the
app (``dom.downloads``).

Run with ``uv run python scripts/fetch_test_assets.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
import time
from pathlib import Path

from dom.downloads import DownloadError, RemoteFile, download_verified, make_client
from dom.downloads.http import sha256_of
from dom.downloads.huggingface import list_files
from dom.downloads.llama_release import (
    Target,
    detect_target,
    find_releases,
    install_release,
    is_installed,
)

logger = logging.getLogger("fetch_test_assets")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEV_DIR = PROJECT_ROOT / ".dev"
BIN_DIR = DEV_DIR / "bin"
MODELS_DIR = DEV_DIR / "models"
DOWNLOADS_DIR = DEV_DIR / "downloads"
# Hash markers live outside models/ so llama-server only ever sees model files there.
VERIFIED_DIR = DEV_DIR / ".verified"
HF_REPO = "unsloth/Qwen3.5-0.8B-GGUF"
MODEL_FILES = ("Qwen3.5-0.8B-Q4_K_M.gguf", "Qwen3.5-0.8B-Q8_0.gguf")
PROGRESS_INTERVAL_S = 0.2


class _ProgressBar:
    """A single-line progress display on stderr."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._last = 0.0

    def __call__(self, done: int, total: int) -> None:
        now = time.monotonic()
        if now - self._last < PROGRESS_INTERVAL_S and done < total:
            return
        self._last = now
        pct = 100.0 * done / total if total else 100.0
        sys.stderr.write(
            f"\r  {self._label}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)"
        )
        if done >= total:
            sys.stderr.write("\n")
        sys.stderr.flush()


async def fetch_binaries(target: Target) -> None:
    """Make sure ``.dev/bin/<variant>`` holds each build of the latest llama.cpp release."""
    if (BIN_DIR / "VERSION").exists():
        logger.info("Removing the old single-build layout in %s", BIN_DIR)
        shutil.rmtree(BIN_DIR)
    async with make_client() as client:
        for release in await find_releases(client, target):
            label = f"{release.variant} build {release.version_label}"
            if is_installed(BIN_DIR, release):
                logger.info("llama-server %s already installed, skipping.", label)
                continue
            archive = DOWNLOADS_DIR / release.asset.name
            await download_verified(client, release.asset, archive, _ProgressBar(archive.name))
            dest = await asyncio.to_thread(install_release, archive, BIN_DIR, release)
            archive.unlink()
            logger.info("Installed llama-server %s into %s", label, dest)


def _already_verified(remote: RemoteFile, dest: Path) -> bool:
    marker = VERIFIED_DIR / (dest.name + ".sha256")
    if not dest.is_file() or dest.stat().st_size != remote.size:
        return False
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == remote.sha256:
        return True
    logger.info("Verifying existing %s ...", dest.name)
    if sha256_of(dest) != remote.sha256:
        return False
    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(remote.sha256 + "\n", encoding="utf-8")
    return True


async def fetch_models() -> None:
    """Make sure both test quants are in ``.dev/models`` and verified."""
    async with make_client() as client:
        files = {f.name: f for f in await list_files(client, HF_REPO)}
        for name in MODEL_FILES:
            remote = files.get(name)
            if remote is None:
                raise DownloadError(f"{name} is not in {HF_REPO} any more.")
            dest = MODELS_DIR / name
            if await asyncio.to_thread(_already_verified, remote, dest):
                logger.info("%s already present and verified, skipping.", name)
                continue
            await download_verified(client, remote, dest, _ProgressBar(name))
            await asyncio.to_thread(_already_verified, remote, dest)
            logger.info("Downloaded and verified %s", name)


def main() -> int:
    """Fetch all test assets. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Download dom's development/test assets.")
    parser.add_argument(
        "--target",
        choices=[t.value for t in Target],
        help="Override the detected platform for the llama-server download.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        target = Target(args.target) if args.target else detect_target()
        logger.info("Platform: %s", target)
        asyncio.run(fetch_binaries(target))
        asyncio.run(fetch_models())
    except DownloadError as exc:
        logger.error("Error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted. Rerun the script to resume.")
        return 130
    logger.info("All test assets are in place under %s", DEV_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
