"""Resumable, SHA-256-verified downloads over an allow-listed HTTP client."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Protocol

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

# Hosts dom may contact (only for downloads the user asked for). Entries starting with "." also
# match subdomains. Hugging Face serves files from its CDN, GitHub from release-asset storage.
ALLOWED_HOSTS: tuple[str, ...] = (
    "huggingface.co",
    ".hf.co",
    ".huggingface.co",
    "github.com",
    "api.github.com",
    ".githubusercontent.com",
)
USER_AGENT = "dom"
CHUNK_SIZE = 1024 * 1024
TIMEOUT = httpx.Timeout(30.0, read=120.0)
HTTP_OK = 200
HTTP_PARTIAL_CONTENT = 206

type Progress = Callable[[int, int], None]


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...
    def hexdigest(self) -> str: ...


class DownloadError(Exception):
    """A download could not be found, completed or verified. The message is plain English."""


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """A file to download, with the SHA-256 its source publishes."""

    name: str
    url: str
    size: int
    sha256: str


def host_allowed(host: str) -> bool:
    """True if ``host`` is one dom may contact."""
    host = host.lower().rstrip(".")
    return any(
        host.endswith(allowed) if allowed.startswith(".") else host == allowed
        for allowed in ALLOWED_HOSTS
    )


async def _check_host(request: httpx.Request) -> None:
    if request.url.scheme != "https" or not host_allowed(request.url.host):
        raise DownloadError(f"dom refused to contact {request.url.host}: not an allowed source.")


def make_client(timeout: httpx.Timeout = TIMEOUT) -> httpx.AsyncClient:
    """An HTTP client that only talks HTTPS to ``ALLOWED_HOSTS`` (redirects included)."""
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=timeout,
        event_hooks={"request": [_check_host]},
    )


async def fetch_json(client: httpx.AsyncClient, url: str) -> object:
    """GET ``url`` and decode JSON."""
    try:
        response = await client.get(url)
        response.raise_for_status()
        payload: object = response.json()
    except httpx.HTTPError as exc:
        raise DownloadError(f"Could not reach {httpx.URL(url).host}: {exc}") from exc
    except ValueError as exc:
        raise DownloadError(f"{httpx.URL(url).host} sent an unreadable answer.") from exc
    return payload


async def fetch_text(client: httpx.AsyncClient, url: str) -> str:
    """GET ``url`` as text."""
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DownloadError(f"Could not reach {httpx.URL(url).host}: {exc}") from exc
    return response.text


def _hash_file(path: Path) -> _Digest:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file."""
    return _hash_file(path).hexdigest()


def _write_chunk(handle: IO[bytes], digest: _Digest, chunk: bytes) -> None:
    handle.write(chunk)
    digest.update(chunk)


async def download_verified(
    client: httpx.AsyncClient,
    remote: RemoteFile,
    dest: Path,
    progress: Progress | None = None,
) -> None:
    """Download ``remote`` to ``dest``, resuming ``dest.part`` and verifying the SHA-256.

    ``dest`` only appears once its content is verified. Disk writes and hashing run in a worker
    thread so the event loop (and UI) stays responsive. Cancelling the task stops the download
    and keeps the ``.part`` file for the next attempt.

    Raises:
        DownloadError: on network failure (the partial file is kept for resuming) or a hash
            mismatch (the partial file is deleted).
    """
    partial = _PartialFile(dest.with_name(dest.name + ".part"))
    try:
        await asyncio.to_thread(dest.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(partial.open_existing, remote.size)
        if partial.offset < remote.size:
            await _stream_into(client, remote, partial, progress)
        if partial.digest.hexdigest() != remote.sha256:
            partial.path.unlink(missing_ok=True)
            raise DownloadError(
                f"{remote.name} was damaged while downloading (its checksum does not match). "
                "Please try again."
            )
        await asyncio.to_thread(partial.path.replace, dest)
    except OSError as exc:
        raise DownloadError(
            f"Could not save {remote.name} to {dest.parent} ({exc.strerror or exc}). "
            "Is the disk full?"
        ) from exc


class _PartialFile:
    """A ``.part`` file being downloaded, and the running hash of its content."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.digest: _Digest = hashlib.sha256()

    def open_existing(self, expected_size: int) -> None:
        """Pick up an earlier partial download (hashing what is already there)."""
        if not self.path.exists():
            return
        size = self.path.stat().st_size
        if size > expected_size:
            self.path.unlink()
            return
        self.offset = size
        self.digest = _hash_file(self.path)

    def restart(self) -> None:
        """The server ignored the range request: start from zero."""
        self.offset = 0
        self.digest = hashlib.sha256()

    async def append(self, handle: IO[bytes], data: bytes) -> None:
        """Write and hash ``data`` in a worker thread."""
        await asyncio.to_thread(_write_chunk, handle, self.digest, data)
        self.offset += len(data)


async def _stream_into(
    client: httpx.AsyncClient,
    remote: RemoteFile,
    partial: _PartialFile,
    progress: Progress | None,
) -> None:
    headers = {"Range": f"bytes={partial.offset}-"} if partial.offset else {}
    try:
        async with client.stream("GET", remote.url, headers=headers) as response:
            if response.status_code == HTTP_PARTIAL_CONTENT:
                mode = "ab"
            elif response.status_code == HTTP_OK:
                mode = "wb"
                partial.restart()
            else:
                response.raise_for_status()
                raise DownloadError(f"Unexpected answer {response.status_code} for {remote.name}.")
            if partial.offset:
                logger.info("Resuming %s from %.1f MB", remote.name, partial.offset / 1e6)
            handle = await asyncio.to_thread(partial.path.open, mode)
            buffer = bytearray()
            try:
                async for data in response.aiter_bytes():
                    buffer += data
                    if len(buffer) >= CHUNK_SIZE:
                        await partial.append(handle, bytes(buffer))
                        buffer.clear()
                        if progress is not None:
                            progress(partial.offset, remote.size)
            finally:
                # Keep every byte that arrived, even if the connection broke, so a resume
                # continues exactly where this attempt stopped.
                if buffer:
                    await partial.append(handle, bytes(buffer))
                await asyncio.to_thread(handle.close)
            if progress is not None:
                progress(partial.offset, remote.size)
    except httpx.HTTPError as exc:
        raise DownloadError(
            f"The download of {remote.name} was interrupted ({exc}). Try again to resume it."
        ) from exc
