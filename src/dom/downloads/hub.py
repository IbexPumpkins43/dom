"""Browsing Hugging Face for GGUF chat models (only while the models screen is open).

Uses the public, anonymous ``/api/models`` search: GGUF repositories sorted by downloads, with
the model's GGUF metadata (parameter count, chat template) and file names. Only models that can
chat (a chat template, and a chat task tag or none), are not gated behind a licence
click-through, and have at least one single-file GGUF are kept.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import httpx

from dom.catalog.models import HubModel
from dom.catalog.quants import is_single_model_file
from dom.downloads.http import DownloadError, make_client
from dom.jsonshape import is_array, is_object

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api/models"
PAGE_SIZE = 40
_EXPAND = ("gguf", "downloads", "likes", "gated", "siblings", "pipeline_tag")
# Hugging Face task tags of models dom can chat with. Untagged repos are kept (many GGUF
# conversions have no tag); speech, audio, image-generation and embedding models are not.
CHAT_TASKS = frozenset({"text-generation", "image-text-to-text", "text2text-generation"})


@dataclass(frozen=True, slots=True)
class HubPage:
    """One page of search results, and where the next page is (if any)."""

    models: tuple[HubModel, ...]
    next_url: str | None


type ParamRange = tuple[float | None, float | None]  # billions of parameters: (at least, at most)
ANY_SIZE: ParamRange = (None, None)


def _billions(value: float) -> str:
    return f"{value:g}B"


def search_url(query: str, size: ParamRange = ANY_SIZE, limit: int = PAGE_SIZE) -> str:
    """The search request for ``query`` ("" lists the most downloaded GGUF models), limited to
    models with ``size`` parameters (Hugging Face's own filter; it is approximate)."""
    params: list[tuple[str, str]] = [
        ("filter", "gguf"),
        ("sort", "downloads"),
        ("direction", "-1"),
        ("limit", str(limit)),
    ]
    if query.strip():
        params.append(("search", query.strip()))
    low, high = size
    if low is not None or high is not None:
        bounds = [f"min:{_billions(low or 0)}"] + ([f"max:{_billions(high)}"] if high else [])
        params.append(("num_parameters", ",".join(bounds)))
    params += [("expand[]", field) for field in _EXPAND]
    return f"{HF_API}?{urlencode(params)}"


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def parse_model(entry: object) -> HubModel | None:
    """A search result as a ``HubModel``, or ``None`` if dom cannot offer it for chat."""
    if not is_object(entry):
        return None
    repo = entry.get("id")
    gguf = entry.get("gguf")
    siblings = entry.get("siblings")
    if not isinstance(repo, str) or not is_object(gguf) or not is_array(siblings):
        return None
    if entry.get("gated") not in (False, None) or not gguf.get("chat_template"):
        return None  # gated (needs a Hugging Face login) or not a chat model
    task = entry.get("pipeline_tag")
    if task is not None and task not in CHAT_TASKS:
        return None  # e.g. speech recognition: has a chat template, but does not chat
    files = tuple(
        name
        for sibling in siblings
        if is_object(sibling)
        and isinstance(name := sibling.get("rfilename"), str)
        and is_single_model_file(name)
    )
    if not files:
        return None
    architecture = gguf.get("architecture")
    return HubModel(
        repo=repo,
        downloads=_int(entry.get("downloads")) or 0,
        likes=_int(entry.get("likes")) or 0,
        params=_int(gguf.get("total")),
        files=files,
        architecture=architecture if isinstance(architecture, str) else None,
        context_length=_int(gguf.get("context_length")),
    )


def parse_page(payload: object, link_header: str | None) -> HubPage:
    """A page of results from the search's JSON and its ``Link`` header."""
    if not is_array(payload):
        raise DownloadError("Unexpected answer from Hugging Face (model list).")
    models = tuple(m for m in (parse_model(entry) for entry in payload) if m is not None)
    return HubPage(models, _next_link(link_header))


def _next_link(header: str | None) -> str | None:
    """The ``rel="next"`` URL of a ``Link`` header."""
    if not header:
        return None
    for part in header.split(","):
        url, _, rel = part.partition(";")
        if 'rel="next"' in rel:
            return url.strip().removeprefix("<").removesuffix(">")
    return None


class HubSearch:
    """Searches Hugging Face (through the allow-listed client)."""

    def __init__(self, client_factory: Callable[[], httpx.AsyncClient] = make_client) -> None:
        """``client_factory`` lets tests fake the network."""
        self._client_factory = client_factory

    async def search(self, query: str, size: ParamRange = ANY_SIZE) -> HubPage:
        """The first page of models matching ``query`` with ``size`` parameters."""
        return await self.page(search_url(query, size))

    async def page(self, url: str) -> HubPage:
        """The page of results at ``url`` (a search URL or a page's ``next_url``).

        Raises:
            DownloadError: with a plain-English message.
        """
        async with self._client_factory() as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                payload: object = response.json()
            except httpx.HTTPError as exc:
                raise DownloadError(
                    "Could not reach Hugging Face. Check your internet connection."
                ) from exc
            except ValueError as exc:
                raise DownloadError("Hugging Face sent an unreadable answer.") from exc
        page = parse_page(payload, response.headers.get("link"))
        logger.info("Hugging Face search: %d usable models", len(page.models))
        return page
