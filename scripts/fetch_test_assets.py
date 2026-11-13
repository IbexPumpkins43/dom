"""Download the development/test assets into ``.dev/``.

Fetches two things, and nothing else:

1. The ``llama-server`` CPU and Vulkan (GPU) builds for this OS/CPU from the ggml-org/llama.cpp
   GitHub releases, unpacked into ``.dev/bin/cpu/`` and ``.dev/bin/vulkan/`` with the release tag
   written to ``VERSION`` in each.
2. The Q4_K_M and Q8_0 quants of ``unsloth/Qwen3.5-0.8B-GGUF`` from Hugging Face, into
   ``.dev/models/``.

Everything is checked against the SHA-256 the source reports. Downloads go to a ``.part`` file
that is resumed on the next run and only renamed into place once verified, so a failed or
interrupted run never leaves a half-written file under the final name. Assets that are already
present and verified are skipped.

Run with ``uv run python scripts/fetch_test_assets.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeGuard

import httpx

logger = logging.getLogger("fetch_test_assets")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEV_DIR = PROJECT_ROOT / ".dev"
BIN_DIR = DEV_DIR / "bin"
MODELS_DIR = DEV_DIR / "models"
# Hash markers live outside models/ so llama-server only ever sees model files there.
VERIFIED_DIR = DEV_DIR / ".verified"

GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp"
HF_REPO = "unsloth/Qwen3.5-0.8B-GGUF"
HF_TREE_URL = f"https://huggingface.co/api/models/{HF_REPO}/tree/main"
HF_RESOLVE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"
MODEL_QUANTS = ("Q4_K_M", "Q8_0")

# Since the move to versioned releases, the "latest" release carries a text file naming the
# build tag (b####) whose release holds the actual binaries.
BUILD_TAG_ASSET = "nightly-tag.txt"

USER_AGENT = "dom-dev-asset-fetcher"
CHUNK_SIZE = 1024 * 1024
PROGRESS_INTERVAL_S = 0.2
HTTP_OK = 200
HTTP_PARTIAL_CONTENT = 206
TIMEOUT = httpx.Timeout(30.0, read=120.0)


class FetchError(Exception):
    """A test asset could not be found, downloaded or verified."""


class Target(StrEnum):
    """Platforms we know how to pick llama.cpp builds for."""

    LINUX_X64 = "linux-x64"
    LINUX_ARM64 = "linux-arm64"
    WINDOWS_X64 = "windows-x64"
    WINDOWS_ARM64 = "windows-arm64"


class Variant(StrEnum):
    """llama.cpp builds we install; each goes into ``.dev/bin/<variant>/``."""

    CPU = "cpu"
    VULKAN = "vulkan"


# Release asset names for each target and build variant.
ASSET_PATTERNS: dict[tuple[Target, Variant], re.Pattern[str]] = {
    (Target.LINUX_X64, Variant.CPU): re.compile(r"^llama-[^-]+-bin-ubuntu-x64\.tar\.gz$"),
    (Target.LINUX_ARM64, Variant.CPU): re.compile(r"^llama-[^-]+-bin-ubuntu-arm64\.tar\.gz$"),
    (Target.WINDOWS_X64, Variant.CPU): re.compile(r"^llama-[^-]+-bin-win-cpu-x64\.zip$"),
    (Target.WINDOWS_ARM64, Variant.CPU): re.compile(r"^llama-[^-]+-bin-win-cpu-arm64\.zip$"),
    (Target.LINUX_X64, Variant.VULKAN): re.compile(r"^llama-[^-]+-bin-ubuntu-vulkan-x64\.tar\.gz$"),
    (Target.LINUX_ARM64, Variant.VULKAN): re.compile(
        r"^llama-[^-]+-bin-ubuntu-vulkan-arm64\.tar\.gz$"
    ),
    (Target.WINDOWS_X64, Variant.VULKAN): re.compile(r"^llama-[^-]+-bin-win-vulkan-x64\.zip$"),
    (Target.WINDOWS_ARM64, Variant.VULKAN): re.compile(r"^llama-[^-]+-bin-win-vulkan-arm64\.zip$"),
}


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """A file to download, with the SHA-256 its source reports."""

    name: str
    url: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BinaryRelease:
    """The llama.cpp release we install, and one build's archive for this platform."""

    variant: Variant
    version_label: str
    asset: RemoteFile


# --------------------------------------------------------------------------------------------
# Small JSON helpers: the APIs return untyped JSON, so narrow it explicitly.
# --------------------------------------------------------------------------------------------


def _is_json_object(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_json_array(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _as_dict(value: object, what: str) -> dict[str, object]:
    if not _is_json_object(value):
        raise FetchError(f"Unexpected response shape for {what}: expected an object.")
    return {str(k): v for k, v in value.items()}


def _as_list(value: object, what: str) -> list[object]:
    if not _is_json_array(value):
        raise FetchError(f"Unexpected response shape for {what}: expected a list.")
    return list(value)


def _as_str(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise FetchError(f"Unexpected response shape for {what}: expected a string.")
    return value


def _as_int(value: object, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise FetchError(f"Unexpected response shape for {what}: expected an integer.")
    return value


def _get_json(client: httpx.Client, url: str) -> object:
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise FetchError(f"Request to {url} failed: {exc}") from exc
    result: object = response.json()
    return result


# --------------------------------------------------------------------------------------------
# Platform detection
# --------------------------------------------------------------------------------------------


def detect_target() -> Target:
    """Return the download target for the current OS and CPU architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_arm = machine in {"arm64", "aarch64"}
    is_x64 = machine in {"x86_64", "amd64"}
    if system == "linux" and is_x64:
        return Target.LINUX_X64
    if system == "linux" and is_arm:
        return Target.LINUX_ARM64
    if system == "windows" and is_x64:
        return Target.WINDOWS_X64
    if system == "windows" and is_arm:
        return Target.WINDOWS_ARM64
    raise FetchError(f"Unsupported platform: {platform.system()} {platform.machine()}.")


# --------------------------------------------------------------------------------------------
# Lookups
# --------------------------------------------------------------------------------------------


def _release_assets(release: dict[str, object]) -> list[dict[str, object]]:
    return [_as_dict(a, "release asset") for a in _as_list(release.get("assets"), "assets")]


def _asset_to_remote(asset: dict[str, object]) -> RemoteFile:
    name = _as_str(asset.get("name"), "asset name")
    digest = _as_str(asset.get("digest"), f"digest of {name}")
    algo, _, value = digest.partition(":")
    if algo != "sha256" or not value:
        raise FetchError(f"GitHub did not report a SHA-256 for {name} (got {digest!r}).")
    return RemoteFile(
        name=name,
        url=_as_str(asset.get("browser_download_url"), f"URL of {name}"),
        size=_as_int(asset.get("size"), f"size of {name}"),
        sha256=value.lower(),
    )


def _pick(assets: list[dict[str, object]], pattern: re.Pattern[str]) -> dict[str, object] | None:
    matches = [a for a in assets if pattern.match(str(a.get("name", "")))]
    if len(matches) > 1:
        raise FetchError(f"Several assets match {pattern.pattern}; refusing to guess.")
    return matches[0] if matches else None


def find_binary_releases(client: httpx.Client, target: Target) -> list[BinaryRelease]:
    """Resolve the latest stable llama.cpp release to each build variant's asset for ``target``."""
    latest = _as_dict(_get_json(client, f"{GITHUB_API}/releases/latest"), "latest release")
    latest_tag = _as_str(latest.get("tag_name"), "release tag")
    assets = _release_assets(latest)
    label = latest_tag
    if _pick(assets, ASSET_PATTERNS[target, Variant.CPU]) is None:
        build_tag = _resolve_build_tag(client, assets, latest_tag)
        build = _as_dict(_get_json(client, f"{GITHUB_API}/releases/tags/{build_tag}"), "build")
        assets = _release_assets(build)
        label = f"{latest_tag} ({build_tag})"
    releases: list[BinaryRelease] = []
    for variant in Variant:
        chosen = _pick(assets, ASSET_PATTERNS[target, variant])
        if chosen is None:
            raise FetchError(f"Release {label} has no {variant} build for {target}.")
        releases.append(
            BinaryRelease(variant=variant, version_label=label, asset=_asset_to_remote(chosen))
        )
    return releases


def _resolve_build_tag(
    client: httpx.Client, assets: list[dict[str, object]], latest_tag: str
) -> str:
    pointer = next((a for a in assets if a.get("name") == BUILD_TAG_ASSET), None)
    if pointer is None:
        raise FetchError(f"Release {latest_tag} has no binaries and no {BUILD_TAG_ASSET} pointer.")
    pointer_url = _as_str(pointer.get("browser_download_url"), BUILD_TAG_ASSET)
    try:
        response = client.get(pointer_url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise FetchError(f"Could not read {BUILD_TAG_ASSET}: {exc}") from exc
    build_tag = response.text.strip()
    if not re.fullmatch(r"b\d+", build_tag):
        raise FetchError(f"{BUILD_TAG_ASSET} has an unexpected build tag: {build_tag!r}.")
    return build_tag


def find_model_files(client: httpx.Client) -> list[RemoteFile]:
    """Look up the exact filenames, sizes and hashes of the two test quants."""
    entries = [_as_dict(e, "repo entry") for e in _as_list(_get_json(client, HF_TREE_URL), "tree")]
    found: list[RemoteFile] = []
    for quant in MODEL_QUANTS:
        pattern = re.compile(rf"^[^/]+-{re.escape(quant)}\.gguf$")
        matches = [
            e
            for e in entries
            if e.get("type") == "file"
            and pattern.match(str(e.get("path", "")))
            and not str(e.get("path", "")).startswith("mmproj")
        ]
        if len(matches) != 1:
            raise FetchError(
                f"Expected exactly one {quant} file in {HF_REPO}, found {len(matches)}."
            )
        entry = matches[0]
        path = _as_str(entry.get("path"), "file path")
        lfs = _as_dict(entry.get("lfs"), f"LFS info of {path}")
        found.append(
            RemoteFile(
                name=path,
                url=f"{HF_RESOLVE_URL}/{path}",
                size=_as_int(entry.get("size"), f"size of {path}"),
                sha256=_as_str(lfs.get("oid"), f"hash of {path}").lower(),
            )
        )
    return found


# --------------------------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


class _Progress:
    """A single-line progress display on stderr."""

    def __init__(self, label: str, total: int, start: int) -> None:
        self._label = label
        self._total = total
        self._done = start
        self._last_draw = 0.0
        self._started = time.monotonic()
        self._start_bytes = start

    def advance(self, n: int) -> None:
        self._done += n
        now = time.monotonic()
        if now - self._last_draw >= PROGRESS_INTERVAL_S or self._done >= self._total:
            self._last_draw = now
            self._draw(now)

    def _draw(self, now: float) -> None:
        pct = 100.0 * self._done / self._total if self._total else 100.0
        elapsed = max(now - self._started, 1e-6)
        speed = (self._done - self._start_bytes) / elapsed / 1e6
        line = (
            f"\r  {self._label}: {self._done / 1e6:8.1f} / {self._total / 1e6:.1f} MB "
            f"({pct:5.1f}%) {speed:6.1f} MB/s"
        )
        sys.stderr.write(line)
        sys.stderr.flush()

    def finish(self) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()


def download_verified(client: httpx.Client, remote: RemoteFile, dest: Path) -> None:
    """Download ``remote`` to ``dest``, resuming a ``.part`` file and verifying the SHA-256.

    ``dest`` only appears once the content is verified.
    """
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    if offset > remote.size:
        part.unlink()
        offset = 0

    if offset < remote.size:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with client.stream("GET", remote.url, headers=headers) as response:
                if response.status_code == HTTP_PARTIAL_CONTENT:
                    mode = "ab"
                elif response.status_code == HTTP_OK:
                    mode, offset = "wb", 0
                else:
                    response.raise_for_status()
                    raise FetchError(f"Unexpected HTTP {response.status_code} for {remote.name}.")
                if offset:
                    logger.info("Resuming %s from %.1f MB", remote.name, offset / 1e6)
                progress = _Progress(remote.name, remote.size, offset)
                with part.open(mode) as handle:
                    for chunk in response.iter_bytes(CHUNK_SIZE):
                        handle.write(chunk)
                        progress.advance(len(chunk))
                progress.finish()
        except httpx.HTTPError as exc:
            raise FetchError(
                f"Download of {remote.name} failed: {exc}. Rerun the script to resume."
            ) from exc

    actual = sha256_of(part)
    if actual != remote.sha256:
        part.unlink()
        raise FetchError(
            f"SHA-256 mismatch for {remote.name}: expected {remote.sha256}, got {actual}. "
            "The partial file was deleted; rerun the script to try again."
        )
    part.replace(dest)


# --------------------------------------------------------------------------------------------
# Unpacking
# --------------------------------------------------------------------------------------------


def _extract_archive(archive: Path, into: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            root = into.resolve()
            for member in zf.namelist():
                target = (into / member).resolve()
                if not target.is_relative_to(root):
                    raise FetchError(f"Archive entry escapes target directory: {member}")
            zf.extractall(into)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(into, filter="data")


def _server_binary_name() -> str:
    return "llama-server.exe" if platform.system() == "Windows" else "llama-server"


def install_binary(archive: Path, release: BinaryRelease) -> None:
    """Unpack into ``.dev/bin/<variant>/`` (flattening one top-level folder) and pin VERSION."""
    dest = BIN_DIR / release.variant
    with tempfile.TemporaryDirectory(dir=DEV_DIR, prefix="bin-") as tmp:
        staging = Path(tmp) / "unpacked"
        staging.mkdir()
        _extract_archive(archive, staging)
        children = list(staging.iterdir())
        content_root = children[0] if len(children) == 1 and children[0].is_dir() else staging
        if not (content_root / _server_binary_name()).exists():
            raise FetchError(f"{_server_binary_name()} not found in {archive.name}.")
        (content_root / "VERSION").write_text(release.version_label + "\n", encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        content_root.rename(dest)


def binary_is_current(release: BinaryRelease) -> bool:
    """True if ``.dev/bin/<variant>`` already holds this release (VERSION is written last)."""
    dest = BIN_DIR / release.variant
    version_file = dest / "VERSION"
    if not version_file.exists() or not (dest / _server_binary_name()).exists():
        return False
    return version_file.read_text(encoding="utf-8").strip() == release.version_label


# --------------------------------------------------------------------------------------------
# Main steps
# --------------------------------------------------------------------------------------------


def fetch_binaries(client: httpx.Client, target: Target) -> None:
    """Make sure ``.dev/bin/<variant>`` holds each llama-server build of the latest release."""
    if (BIN_DIR / "VERSION").exists():
        logger.info("Removing the old single-build layout in %s", BIN_DIR)
        shutil.rmtree(BIN_DIR)
    for release in find_binary_releases(client, target):
        label = f"{release.variant} build {release.version_label}"
        logger.info("llama.cpp %s: %s", label, release.asset.name)
        if binary_is_current(release):
            logger.info("llama-server %s already installed, skipping.", label)
            continue
        archive = DEV_DIR / "downloads" / release.asset.name
        if not (archive.exists() and sha256_of(archive) == release.asset.sha256):
            download_verified(client, release.asset, archive)
        install_binary(archive, release)
        archive.unlink()
        logger.info("Installed llama-server %s into %s", label, BIN_DIR / release.variant)


def _verified_marker(path: Path) -> Path:
    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    return VERIFIED_DIR / (path.name + ".sha256")


def fetch_models(client: httpx.Client) -> None:
    """Make sure both test quants are in ``.dev/models`` and verified."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for remote in find_model_files(client):
        dest = MODELS_DIR / remote.name
        marker = _verified_marker(dest)
        if (
            dest.exists()
            and dest.stat().st_size == remote.size
            and marker.exists()
            and marker.read_text(encoding="utf-8").strip() == remote.sha256
        ):
            logger.info("%s already present and verified, skipping.", remote.name)
            continue
        if dest.exists() and dest.stat().st_size == remote.size:
            logger.info("Verifying existing %s ...", remote.name)
            if sha256_of(dest) == remote.sha256:
                marker.write_text(remote.sha256 + "\n", encoding="utf-8")
                continue
            dest.unlink()
        download_verified(client, remote, dest)
        marker.write_text(remote.sha256 + "\n", encoding="utf-8")
        logger.info("Downloaded and verified %s", remote.name)


def main() -> int:
    """Fetch all test assets. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--target",
        choices=[t.value for t in Target],
        help="Override the detected platform for the llama-server download.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # httpx logs every request at INFO, including long signed download URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        target = Target(args.target) if args.target else detect_target()
        logger.info("Platform: %s", target)
        DEV_DIR.mkdir(exist_ok=True)
        headers = {"User-Agent": USER_AGENT}
        with httpx.Client(headers=headers, follow_redirects=True, timeout=TIMEOUT) as client:
            fetch_binaries(client, target)
            fetch_models(client)
    except FetchError as exc:
        logger.error("Error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted. Rerun the script to resume.")
        return 130
    logger.info("All test assets are in place under %s", DEV_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
