"""Looking up model files on Hugging Face."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dom.downloads._json import as_array, as_int, as_object, as_str
from dom.downloads.http import DownloadError, RemoteFile, fetch_json

if TYPE_CHECKING:
    import httpx

HF = "https://huggingface.co"


async def list_files(client: httpx.AsyncClient, repo: str) -> list[RemoteFile]:
    """Every LFS file at the top level of ``repo`` with its size and SHA-256."""
    entries = as_array(await fetch_json(client, f"{HF}/api/models/{repo}/tree/main"), "file list")
    files: list[RemoteFile] = []
    for raw in entries:
        entry = as_object(raw, "file entry")
        lfs = entry.get("lfs")
        if entry.get("type") != "file" or lfs is None:
            continue
        path = as_str(entry.get("path"), "file name")
        files.append(
            RemoteFile(
                name=path,
                url=f"{HF}/{repo}/resolve/main/{path}",
                size=as_int(entry.get("size"), f"size of {path}"),
                sha256=as_str(as_object(lfs, "file hash").get("oid"), f"hash of {path}").lower(),
            )
        )
    return files


async def resolve_file(client: httpx.AsyncClient, repo: str, filename: str) -> RemoteFile:
    """The exact file ``filename`` in ``repo``, with its current size and hash."""
    for remote in await list_files(client, repo):
        if remote.name == filename:
            return remote
    raise DownloadError(f"{filename} is no longer available from {repo}.")
