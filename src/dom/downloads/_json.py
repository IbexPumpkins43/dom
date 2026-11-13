"""Narrowing untyped JSON from the download sources."""

from __future__ import annotations

from dom.downloads.http import DownloadError
from dom.jsonshape import is_array, is_object


def as_object(value: object, what: str) -> dict[str, object]:
    """``value`` as an object, or a ``DownloadError`` naming ``what``."""
    if not is_object(value):
        raise DownloadError(f"Unexpected answer from the server ({what}).")
    return value


def as_array(value: object, what: str) -> list[object]:
    """``value`` as an array, or a ``DownloadError`` naming ``what``."""
    if not is_array(value):
        raise DownloadError(f"Unexpected answer from the server ({what}).")
    return value


def as_str(value: object, what: str) -> str:
    """``value`` as a string, or a ``DownloadError`` naming ``what``."""
    if not isinstance(value, str):
        raise DownloadError(f"Unexpected answer from the server ({what}).")
    return value


def as_int(value: object, what: str) -> int:
    """``value`` as an integer, or a ``DownloadError`` naming ``what``."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise DownloadError(f"Unexpected answer from the server ({what}).")
    return value
