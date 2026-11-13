"""Finding and installing llama.cpp release builds from GitHub."""

from __future__ import annotations

import platform
import re
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from dom.downloads._json import as_array, as_int, as_object, as_str
from dom.downloads.http import DownloadError, RemoteFile, fetch_json, fetch_text

if TYPE_CHECKING:
    import httpx

GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp"
# Since the move to versioned releases, the "latest" release carries a text file naming the
# build tag (b####) whose release holds the actual binaries.
BUILD_TAG_ASSET = "nightly-tag.txt"
VERSION_FILE = "VERSION"
LLAMA_LICENSE_FILE = "LICENSE"


class Target(StrEnum):
    """Platforms dom ships llama.cpp builds for."""

    LINUX_X64 = "linux-x64"
    LINUX_ARM64 = "linux-arm64"
    WINDOWS_X64 = "windows-x64"
    WINDOWS_ARM64 = "windows-arm64"


class Variant(StrEnum):
    """llama.cpp builds dom installs; each goes into ``<bin>/<variant>/``."""

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
class BinaryRelease:
    """One build of a llama.cpp release, as a downloadable archive."""

    variant: Variant
    version_label: str
    asset: RemoteFile


def detect_target(system: str | None = None, machine: str | None = None) -> Target:
    """The download target for this OS and CPU."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    is_arm = machine in {"arm64", "aarch64"}
    is_x64 = machine in {"x86_64", "amd64"}
    targets = {
        ("linux", True, False): Target.LINUX_X64,
        ("linux", False, True): Target.LINUX_ARM64,
        ("windows", True, False): Target.WINDOWS_X64,
        ("windows", False, True): Target.WINDOWS_ARM64,
    }
    target = targets.get((system, is_x64, is_arm))
    if target is None:
        raise DownloadError(f"dom has no llama.cpp build for {system} {machine}.")
    return target


def _assets(release: dict[str, object]) -> list[dict[str, object]]:
    return [as_object(a, "release asset") for a in as_array(release.get("assets"), "assets")]


def _pick(assets: list[dict[str, object]], pattern: re.Pattern[str]) -> dict[str, object] | None:
    matches = [a for a in assets if pattern.match(str(a.get("name", "")))]
    if len(matches) > 1:
        raise DownloadError(f"Several release files match {pattern.pattern}; refusing to guess.")
    return matches[0] if matches else None


def _to_remote(asset: dict[str, object]) -> RemoteFile:
    name = as_str(asset.get("name"), "asset name")
    algo, _, value = as_str(asset.get("digest"), f"checksum of {name}").partition(":")
    if algo != "sha256" or not value:
        raise DownloadError(f"GitHub did not publish a SHA-256 checksum for {name}.")
    return RemoteFile(
        name=name,
        url=as_str(asset.get("browser_download_url"), f"address of {name}"),
        size=as_int(asset.get("size"), f"size of {name}"),
        sha256=value.lower(),
    )


async def find_releases(client: httpx.AsyncClient, target: Target) -> list[BinaryRelease]:
    """The latest stable llama.cpp release's CPU and Vulkan builds for ``target``."""
    latest = as_object(await fetch_json(client, f"{GITHUB_API}/releases/latest"), "release")
    latest_tag = as_str(latest.get("tag_name"), "release tag")
    assets = _assets(latest)
    label = latest_tag
    if _pick(assets, ASSET_PATTERNS[target, Variant.CPU]) is None:
        pointer = next((a for a in assets if a.get("name") == BUILD_TAG_ASSET), None)
        if pointer is None:
            raise DownloadError(f"llama.cpp release {latest_tag} has no downloads for {target}.")
        url = as_str(pointer.get("browser_download_url"), BUILD_TAG_ASSET)
        build_tag = (await fetch_text(client, url)).strip()
        if not re.fullmatch(r"b\d+", build_tag):
            raise DownloadError(f"Unexpected llama.cpp build tag {build_tag!r}.")
        build = await fetch_json(client, f"{GITHUB_API}/releases/tags/{build_tag}")
        assets = _assets(as_object(build, "build"))
        label = f"{latest_tag} ({build_tag})"
    releases: list[BinaryRelease] = []
    for variant in Variant:
        chosen = _pick(assets, ASSET_PATTERNS[target, variant])
        if chosen is None:
            raise DownloadError(f"llama.cpp {label} has no {variant} build for {target}.")
        releases.append(BinaryRelease(variant, label, _to_remote(chosen)))
    return releases


def server_name(system: str | None = None) -> str:
    """``llama-server`` executable name for this OS."""
    return "llama-server.exe" if (system or platform.system()) == "Windows" else "llama-server"


def is_installed(bin_dir: Path, release: BinaryRelease) -> bool:
    """True if ``bin_dir/<variant>`` holds exactly this release (VERSION is written last)."""
    dest = bin_dir / release.variant
    version = dest / VERSION_FILE
    if not version.is_file() or not (dest / server_name()).is_file():
        return False
    return version.read_text(encoding="utf-8").strip() == release.version_label


def _extract(archive: Path, into: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            root = into.resolve()
            for member in zf.namelist():
                if not (into / member).resolve().is_relative_to(root):
                    raise DownloadError(f"Unsafe path in {archive.name}: {member}")
            zf.extractall(into)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(into, filter="data")


def install_release(archive: Path, bin_dir: Path, release: BinaryRelease) -> Path:
    """Unpack into ``bin_dir/<variant>/`` (replacing what is there) and write VERSION.

    Blocking; call it from a worker thread. Returns the installed directory.

    Raises:
        DownloadError: if the archive is damaged or cannot be unpacked (plain-English message).
    """
    try:
        return _install(archive, bin_dir, release)
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise DownloadError(f"Could not unpack {archive.name} into {bin_dir}: {exc}") from exc


def _install(archive: Path, bin_dir: Path, release: BinaryRelease) -> Path:
    dest = bin_dir / release.variant
    bin_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=bin_dir, prefix=".unpack-") as tmp:
        staging = Path(tmp) / "unpacked"
        staging.mkdir()
        _extract(archive, staging)
        children = list(staging.iterdir())
        root = children[0] if len(children) == 1 and children[0].is_dir() else staging
        if not (root / server_name()).is_file():
            raise DownloadError(f"{server_name()} is missing from {archive.name}.")
        (root / VERSION_FILE).write_text(release.version_label + "\n", encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        root.rename(dest)
    return dest
