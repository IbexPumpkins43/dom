"""Downloads layer: verified, resumable downloads from Hugging Face and GitHub only.

dom makes network requests only when the user asks for a download (and in the dev-only
``scripts/fetch_test_assets.py``). Every request goes through ``make_client``, which refuses any
host outside ``ALLOWED_HOSTS``.
"""

from dom.downloads.http import (
    ALLOWED_HOSTS,
    DownloadError,
    RemoteFile,
    download_verified,
    fetch_json,
    make_client,
)

__all__ = [
    "ALLOWED_HOSTS",
    "DownloadError",
    "RemoteFile",
    "download_verified",
    "fetch_json",
    "make_client",
]
