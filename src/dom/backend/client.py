"""Async HTTP client for llama-server in router mode."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, TypeGuard

import httpx

from dom.backend.errors import (
    ApiError,
    BackendError,
    ChatStreamError,
    ModelLoadError,
    ModelNotFoundError,
    ServerUnavailableError,
)
from dom.domain import ChatDelta, MemoryUse, ModelInfo, ModelMemory, ModelStatus

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from dom.domain import ChatMessage

logger = logging.getLogger(__name__)

# Model entries the router found in --models-dir. It also lists its download cache
# ("cache") and presets; dom only shows the models folder.
MODELS_DIR_SOURCE = "models_dir"

HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
# Prompt processing on a long conversation can take a while before the first token arrives,
# and a reply can stream for minutes; cancellation is how the user stops it.
STREAM_TIMEOUT = httpx.Timeout(None, connect=5.0)


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_array(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def parse_model_status(status: object) -> tuple[ModelStatus, int | None]:
    """Map a router ``status`` object to a ``ModelStatus`` and exit code.

    The router has no "failed" value: a failed load is ``unloaded`` with ``failed: true``.
    ``sleeping`` models wake on the next request, so they count as loaded.
    """
    if not _is_object(status):
        return ModelStatus.UNLOADED, None
    value = status.get("value")
    exit_code = status.get("exit_code")
    code = exit_code if isinstance(exit_code, int) else None
    match value:
        case "loaded" | "sleeping":
            return ModelStatus.LOADED, None
        case "loading" | "downloading":
            return ModelStatus.LOADING, None
        case _ if status.get("failed") is True:
            return ModelStatus.FAILED, code
        case _:
            return ModelStatus.UNLOADED, None


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _parse_memory_use(point: object) -> MemoryUse | None:
    if not _is_object(point) or not isinstance(n_ctx := point.get("n_ctx"), int):
        return None
    devices = point.get("devices")
    if not _is_array(devices):
        return None
    device_bytes = host_bytes = 0
    for entry in devices:
        if not _is_object(entry):
            continue
        used = _int(entry.get("model")) + _int(entry.get("context")) + _int(entry.get("compute"))
        if entry.get("kind") == "device":
            device_bytes += used
        else:  # "host" and host-side "buffer_type" entries (e.g. CPU_REPACK) live in RAM
            host_bytes += used
    return MemoryUse(n_ctx=n_ctx, device_bytes=device_bytes, host_bytes=host_bytes)


def parse_model_memory(memory: object) -> ModelMemory | None:
    """Parse the router's ``memory`` field; ``None`` unless it is fully measured."""
    if not _is_object(memory) or memory.get("status") != "measured":
        return None
    low = _parse_memory_use(memory.get("min"))
    high = _parse_memory_use(memory.get("max"))
    if low is None or high is None:
        return None
    return ModelMemory(low=low, high=high)


def parse_models(payload: object) -> list[ModelInfo]:
    """Parse a ``GET /models`` response into models from the models folder."""
    if not _is_object(payload) or not _is_array(data := payload.get("data")):
        raise BackendError("The AI server sent an unexpected model list.")
    models: list[ModelInfo] = []
    for entry in data:
        if not _is_object(entry) or not isinstance(model_id := entry.get("id"), str):
            continue
        source = entry.get("source")
        if source is not None and source != MODELS_DIR_SOURCE:
            continue
        status, exit_code = parse_model_status(entry.get("status"))
        context_length = entry.get("context_length")
        models.append(
            ModelInfo(
                id=model_id,
                status=status,
                path=_str_or_none(entry.get("path")),
                exit_code=exit_code,
                context_length=context_length if isinstance(context_length, int) else None,
                memory=parse_model_memory(entry.get("memory")),
            )
        )
    models.sort(key=lambda m: m.id.lower())
    return models


def _error_from_response(status_code: int, body: str) -> ApiError:
    message, error_type = body.strip() or "no details", None
    try:
        payload: object = json.loads(body)
    except ValueError:
        payload = None
    if _is_object(payload) and _is_object(err := payload.get("error")):
        message = _str_or_none(err.get("message")) or message
        error_type = _str_or_none(err.get("type"))
    if status_code == HTTP_NOT_FOUND:
        return ModelNotFoundError(status_code, message, error_type)
    return ApiError(status_code, message, error_type)


def parse_sse_line(line: str) -> ChatDelta | None:
    """Parse one Server-Sent Events line of a chat stream.

    Returns ``None`` for lines without content (comments, keep-alives, role-only chunks).

    Raises:
        _StreamDoneError: on the ``[DONE]`` end-of-stream marker.
        ChatStreamError: if the server sent an error event or invalid JSON.
    """
    if not line.startswith(_SSE_DATA_PREFIX):
        return None
    data = line[len(_SSE_DATA_PREFIX) :].strip()
    if data == _SSE_DONE:
        raise _StreamDoneError
    try:
        payload: object = json.loads(data)
    except ValueError as exc:
        raise ChatStreamError("The AI server sent a reply that could not be read.") from exc
    if not _is_object(payload):
        return None
    if _is_object(err := payload.get("error")):
        message = _str_or_none(err.get("message")) or "unknown error"
        raise ChatStreamError(f"The AI server stopped the reply: {message}")
    choices = payload.get("choices")
    if not _is_array(choices) or not choices or not _is_object(choice := choices[0]):
        return None
    delta = choice.get("delta")
    if not _is_object(delta):
        return None
    content = _str_or_none(delta.get("content")) or ""
    reasoning = _str_or_none(delta.get("reasoning_content")) or ""
    if not content and not reasoning:
        return None
    return ChatDelta(content=content, reasoning=reasoning)


class _StreamDoneError(Exception):
    """Internal: the ``[DONE]`` marker was received."""


class LlamaClient:
    """Talks to a running llama-server router over its HTTP API."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create a client for ``base_url`` (``transport`` lets tests inject a fake)."""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=DEFAULT_TIMEOUT, transport=transport
        )

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._http.aclose()

    async def health(self) -> bool:
        """True if the server answers ``/health`` with 200."""
        try:
            response = await self._http.get("/health")
        except httpx.HTTPError:
            return False
        return response.is_success

    async def list_models(self, *, reload: bool = False) -> list[ModelInfo]:
        """Return the models in the models folder with their current status.

        ``reload`` makes the router re-read the models folder and the presets file first.
        """
        response = await self._request("GET", "/models?reload=1" if reload else "/models")
        try:
            payload: object = response.json()
        except ValueError as exc:
            raise BackendError("The AI server sent an unreadable model list.") from exc
        return parse_models(payload)

    async def get_model(self, model_id: str) -> ModelInfo:
        """Return one model's current info."""
        for model in await self.list_models():
            if model.id == model_id:
                return model
        raise ModelNotFoundError(HTTP_NOT_FOUND, f"Model {model_id} is not in the models folder")

    async def load_model(self, model_id: str) -> None:
        """Ask the router to start loading a model (returns before it is loaded)."""
        await self._request("POST", "/models/load", json={"model": model_id})

    async def unload_model(self, model_id: str) -> None:
        """Ask the router to unload a model."""
        await self._request("POST", "/models/unload", json={"model": model_id})

    async def wait_for_status(
        self,
        model_id: str,
        wanted: ModelStatus,
        *,
        poll_interval: float = 0.25,
        timeout: float = 600.0,
    ) -> ModelInfo:
        """Poll until a model reaches ``wanted``.

        Raises:
            ModelLoadError: if the model fails, or stops loading without reaching ``wanted``,
                or the timeout passes.
        """
        try:
            async with asyncio.timeout(timeout):
                return await self._poll_status(model_id, wanted, poll_interval)
        except TimeoutError as exc:
            raise ModelLoadError(
                f"The model {model_id} took too long to become {wanted.value}."
            ) from exc

    async def _poll_status(
        self, model_id: str, wanted: ModelStatus, poll_interval: float
    ) -> ModelInfo:
        saw_loading = False
        while True:
            model = await self.get_model(model_id)
            if model.status == wanted:
                return model
            if model.status == ModelStatus.FAILED:
                detail = f" (exit code {model.exit_code})" if model.exit_code else ""
                raise ModelLoadError(f"The model {model_id} could not be loaded{detail}.")
            if model.status == ModelStatus.LOADING:
                saw_loading = True
            elif wanted == ModelStatus.LOADED and saw_loading:
                raise ModelLoadError(f"The model {model_id} stopped loading unexpectedly.")
            await asyncio.sleep(poll_interval)

    async def switch_model(self, model_id: str, *, poll_interval: float = 0.25) -> ModelInfo:
        """Unload any other model, then load ``model_id`` and wait until it is ready."""
        models = await self.list_models()
        if not any(m.id == model_id for m in models):
            raise ModelNotFoundError(
                HTTP_NOT_FOUND, f"Model {model_id} is not in the models folder"
            )
        for other in models:
            if other.id != model_id and other.status in (ModelStatus.LOADED, ModelStatus.LOADING):
                logger.info("Unloading %s", other.id)
                await self.unload_model(other.id)
                await self.wait_for_status(
                    other.id, ModelStatus.UNLOADED, poll_interval=poll_interval
                )
        current = next(m for m in models if m.id == model_id)
        if current.status == ModelStatus.LOADED:
            return current
        if current.status != ModelStatus.LOADING:
            logger.info("Loading %s", model_id)
            await self.load_model(model_id)
        return await self.wait_for_status(model_id, ModelStatus.LOADED, poll_interval=poll_interval)

    async def stream_chat(
        self, model_id: str, messages: Sequence[ChatMessage]
    ) -> AsyncGenerator[ChatDelta]:
        """Stream a chat reply as it is generated.

        Stop iterating (or cancel the task) to cancel the reply: the HTTP connection is closed
        and llama-server stops generating.
        """
        body = {
            "model": model_id,
            "stream": True,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        }
        try:
            async with self._http.stream(
                "POST", "/v1/chat/completions", json=body, timeout=STREAM_TIMEOUT
            ) as response:
                if not response.is_success:
                    text = (await response.aread()).decode("utf-8", errors="replace")
                    raise _error_from_response(response.status_code, text)
                # Keep reading after [DONE] until the server ends the response: returning early
                # would leave httpx's internal generators suspended, and they would then be
                # finalised from another task (which anyio rejects with a RuntimeError).
                done = False
                async for line in response.aiter_lines():
                    if done:
                        continue
                    try:
                        delta = parse_sse_line(line)
                    except _StreamDoneError:
                        done = True
                        continue
                    if delta is not None:
                        yield delta
        except httpx.HTTPError as exc:
            raise ServerUnavailableError(
                "Lost the connection to the AI server while it was replying."
            ) from exc

    async def _request(
        self, method: str, path: str, json: dict[str, str] | None = None
    ) -> httpx.Response:
        try:
            response = await self._http.request(method, path, json=json)
        except httpx.HTTPError as exc:
            raise ServerUnavailableError("Could not reach the AI server.") from exc
        if not response.is_success:
            raise _error_from_response(response.status_code, response.text)
        return response
