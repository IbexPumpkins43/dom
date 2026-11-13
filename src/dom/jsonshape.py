"""Narrowing untyped JSON (from llama-server, llama-bench, GitHub, Hugging Face) safely."""

from __future__ import annotations

from typing import TypeGuard


def is_object(value: object) -> TypeGuard[dict[str, object]]:
    """True for a JSON object."""
    return isinstance(value, dict)


def is_array(value: object) -> TypeGuard[list[object]]:
    """True for a JSON array."""
    return isinstance(value, list)


def str_or_none(value: object) -> str | None:
    """``value`` if it is a string, else ``None``."""
    return value if isinstance(value, str) else None


def int_or_zero(value: object) -> int:
    """``value`` if it is an integer (not a bool), else 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
