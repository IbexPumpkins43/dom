"""Tests for the downloads layer (all network access is faked with httpx.MockTransport)."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from typing import TYPE_CHECKING

import httpx
import pytest

from dom.downloads import DownloadError, RemoteFile, download_verified
from dom.downloads.http import host_allowed, make_client
from dom.downloads.huggingface import resolve_file
from dom.downloads.llama_release import (
    BinaryRelease,
    Target,
    Variant,
    detect_target,
    find_releases,
    install_release,
    is_installed,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

CONTENT = bytes(range(256)) * 4096  # 1 MiB
SHA = hashlib.sha256(CONTENT).hexdigest()
URL = "https://huggingface.co/org/repo/resolve/main/model.gguf"
REMOTE = RemoteFile("model.gguf", URL, len(CONTENT), SHA)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    client = make_client()
    client._transport = httpx.MockTransport(handler)  # pyright: ignore[reportPrivateUsage] -- inject fake network into the allow-listed client
    return client


def _ranged(requests: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        header = request.headers.get("Range")
        if header:
            start = int(header.removeprefix("bytes=").rstrip("-"))
            return httpx.Response(206, content=CONTENT[start:])
        return httpx.Response(200, content=CONTENT)

    return handler


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("huggingface.co", True),
        ("cas-bridge.xethub.hf.co", True),
        ("us.aws.cdn.hf.co", True),
        ("api.github.com", True),
        ("github.com", True),
        ("release-assets.githubusercontent.com", True),
        ("evil.com", False),
        ("huggingface.co.evil.com", False),
        ("nothf.co", False),
    ],
)
def test_host_allowed(host: str, allowed: bool) -> None:
    assert host_allowed(host) is allowed


async def test_full_download_is_verified_and_renamed(tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []
    requests: list[httpx.Request] = []
    dest = tmp_path / "models" / "model.gguf"
    async with _client(_ranged(requests)) as client:
        await download_verified(client, REMOTE, dest, lambda d, t: seen.append((d, t)))
    assert dest.read_bytes() == CONTENT
    assert not (tmp_path / "models" / "model.gguf.part").exists()
    assert seen[-1] == (len(CONTENT), len(CONTENT))


async def test_partial_download_resumes(tmp_path: Path) -> None:
    dest = tmp_path / "model.gguf"
    (tmp_path / "model.gguf.part").write_bytes(CONTENT[:300_000])
    requests: list[httpx.Request] = []
    async with _client(_ranged(requests)) as client:
        await download_verified(client, REMOTE, dest)
    assert requests[0].headers["Range"] == "bytes=300000-"
    assert dest.read_bytes() == CONTENT


async def test_server_ignoring_range_restarts_cleanly(tmp_path: Path) -> None:
    dest = tmp_path / "model.gguf"
    (tmp_path / "model.gguf.part").write_bytes(b"garbage" * 1000)
    async with _client(lambda _r: httpx.Response(200, content=CONTENT)) as client:
        await download_verified(client, REMOTE, dest)
    assert dest.read_bytes() == CONTENT


async def test_checksum_mismatch_deletes_the_partial_file(tmp_path: Path) -> None:
    dest = tmp_path / "model.gguf"
    corrupt = CONTENT[:-1] + b"\x00"
    async with _client(lambda _r: httpx.Response(200, content=corrupt)) as client:
        with pytest.raises(DownloadError, match="damaged"):
            await download_verified(client, REMOTE, dest)
    assert not dest.exists()
    assert not (tmp_path / "model.gguf.part").exists()


async def test_interrupted_download_keeps_the_partial_file(tmp_path: Path) -> None:
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):  # noqa: ANN204 -- async generator protocol
            yield CONTENT[:200_000]
            raise httpx.ReadError("connection reset")

    dest = tmp_path / "model.gguf"
    async with _client(lambda _r: httpx.Response(200, stream=Broken())) as client:
        with pytest.raises(DownloadError, match="resume"):
            await download_verified(client, REMOTE, dest)
    assert (tmp_path / "model.gguf.part").stat().st_size == 200_000  # noqa: PLR2004


async def test_disallowed_hosts_are_refused(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    remote = RemoteFile("x", "https://evil.example/x", 1, SHA)
    async with _client(_ranged(calls)) as client:
        with pytest.raises(DownloadError, match="refused"):
            await download_verified(client, remote, tmp_path / "x")
    assert calls == []


async def test_redirect_to_a_disallowed_host_is_refused(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "huggingface.co":
            return httpx.Response(302, headers={"Location": "https://evil.example/model"})
        return httpx.Response(200, content=CONTENT)

    async with _client(handler) as client:
        with pytest.raises(DownloadError, match="refused"):
            await download_verified(client, REMOTE, tmp_path / "model.gguf")


async def test_resolve_file_reads_size_and_hash() -> None:
    tree = [
        {"type": "file", "path": "README.md", "size": 10},
        {"type": "file", "path": "model.gguf", "size": len(CONTENT), "lfs": {"oid": SHA.upper()}},
    ]
    async with _client(lambda _r: httpx.Response(200, json=tree)) as client:
        remote = await resolve_file(client, "org/repo", "model.gguf")
        assert remote == RemoteFile("model.gguf", URL, len(CONTENT), SHA)
        with pytest.raises(DownloadError, match="no longer available"):
            await resolve_file(client, "org/repo", "missing.gguf")


# ---- llama.cpp releases ------------------------------------------------------------------


def _asset(name: str, url: str = "https://github.com/x") -> dict[str, object]:
    return {"name": name, "browser_download_url": f"{url}/{name}", "size": 10,
            "digest": "sha256:" + "ab" * 32}  # fmt: skip


async def test_find_releases_follows_the_build_tag_pointer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/releases/latest"):
            return httpx.Response(200, json={"tag_name": "v0.9.0",
                                             "assets": [_asset("nightly-tag.txt")]})  # fmt: skip
        if path.endswith("/nightly-tag.txt"):
            return httpx.Response(200, text="b12110\n")
        if path.endswith("/releases/tags/b12110"):
            return httpx.Response(200, json={"tag_name": "b12110", "assets": [
                _asset("llama-b12110-bin-win-cpu-arm64.zip"),
                _asset("llama-b12110-bin-win-vulkan-arm64.zip"),
                _asset("llama-b12110-bin-win-cpu-x64.zip"),
                _asset("llama-b12110-bin-win-opencl-adreno-arm64.zip"),
            ]})  # fmt: skip
        return httpx.Response(404)

    async with _client(handler) as client:
        releases = await find_releases(client, Target.WINDOWS_ARM64)
    assert [(r.variant, r.version_label, r.asset.name) for r in releases] == [
        (Variant.CPU, "v0.9.0 (b12110)", "llama-b12110-bin-win-cpu-arm64.zip"),
        (Variant.VULKAN, "v0.9.0 (b12110)", "llama-b12110-bin-win-vulkan-arm64.zip"),
    ]
    assert releases[0].asset.sha256 == "ab" * 32


@pytest.mark.parametrize(
    ("system", "machine", "target"),
    [
        ("Linux", "x86_64", Target.LINUX_X64),
        ("Linux", "aarch64", Target.LINUX_ARM64),
        ("Windows", "AMD64", Target.WINDOWS_X64),
        ("Windows", "ARM64", Target.WINDOWS_ARM64),
    ],
)
def test_detect_target(system: str, machine: str, target: Target) -> None:
    assert detect_target(system, machine) == target


def test_detect_target_rejects_unknown_platforms() -> None:
    with pytest.raises(DownloadError):
        detect_target("Darwin", "arm64")


def _tarball(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_install_release_flattens_and_pins_version(tmp_path: Path) -> None:
    archive = _tarball(
        tmp_path / "llama.tar.gz",
        {"llama-b1/llama-server": b"bin", "llama-b1/libggml.so": b"lib"},
    )
    release = BinaryRelease(Variant.CPU, "v1 (b1)", RemoteFile("x", "u", 1, "s"))
    bin_dir = tmp_path / "bin"
    dest = install_release(archive, bin_dir, release)
    assert dest == bin_dir / "cpu"
    assert (dest / "libggml.so").read_bytes() == b"lib"
    assert json.dumps((dest / "VERSION").read_text(encoding="utf-8")) == json.dumps("v1 (b1)\n")
    assert is_installed(bin_dir, release)
    assert not is_installed(bin_dir, BinaryRelease(Variant.CPU, "v2", release.asset))


def test_install_release_rejects_unsafe_archives(tmp_path: Path) -> None:
    archive = _tarball(tmp_path / "bad.tar.gz", {"../escape": b"x", "llama-server": b"bin"})
    release = BinaryRelease(Variant.CPU, "v1", RemoteFile("x", "u", 1, "s"))
    with pytest.raises((DownloadError, tarfile.TarError)):
        install_release(archive, tmp_path / "bin", release)
    assert not (tmp_path / "escape").exists()
