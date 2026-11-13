"""Tests for ``LlamaClient`` against a fake llama-server router (``httpx.MockTransport``)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeGuard

import httpx
import pytest

from dom.backend import (
    ApiError,
    ChatStreamError,
    LlamaClient,
    ModelLoadError,
    ModelNotFoundError,
    ServerUnavailableError,
)
from dom.backend.client import parse_models
from dom.domain import ChatDelta, ChatMessage, ModelStatus, Role

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

API_KEY = "test-key"


def _status(value: str, **extra: object) -> dict[str, object]:
    return {"value": value, **extra}


@dataclass
class FakeRouter:
    """Minimal stateful stand-in for llama-server in router mode."""

    statuses: dict[str, dict[str, object]] = field(default_factory=dict[str, dict[str, object]])
    polls_until_loaded: int = 2
    fail_load: set[str] = field(default_factory=set[str])
    chat_lines: list[str] = field(default_factory=list[str])
    requests: list[tuple[str, str, object]] = field(default_factory=list[tuple[str, str, object]])
    _loading_polls: dict[str, int] = field(default_factory=dict[str, int])

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("Authorization") != f"Bearer {API_KEY}":
            return _error(401, "Invalid API Key", "authentication_error")
        body: object = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        route = (request.method, request.url.path)
        if route == ("GET", "/models"):
            self._advance_loading()
            return httpx.Response(200, json={"data": self._entries(), "object": "list"})
        if route == ("POST", "/v1/chat/completions"):
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content="".join(f"{line}\n\n" for line in self.chat_lines).encode(),
            )
        if route in {("POST", "/models/load"), ("POST", "/models/unload")}:
            return self._load_or_unload(request.url.path, _model_of(body))
        return _error(404, "not found", "not_found_error")

    def _load_or_unload(self, path: str, model: str | None) -> httpx.Response:
        if model is None or model not in self.statuses:
            return _error(404, "File Not Found", "not_found_error")
        if path == "/models/unload":
            self.statuses[model] = _status("unloaded")
        elif self.statuses[model]["value"] == "loaded":
            return _error(400, "model is already running", "invalid_request_error")
        else:
            self.statuses[model] = _status("loading")
            self._loading_polls[model] = 0
        return httpx.Response(200, json={"success": True})

    def _advance_loading(self) -> None:
        for model, polls in list(self._loading_polls.items()):
            if polls >= self.polls_until_loaded:
                del self._loading_polls[model]
                self.statuses[model] = (
                    _status("unloaded", failed=True, exit_code=1)
                    if model in self.fail_load
                    else _status("loaded")
                )
            else:
                self._loading_polls[model] = polls + 1

    def _entries(self) -> list[dict[str, object]]:
        return [
            {"id": mid, "path": f"/models/{mid}.gguf", "source": "models_dir", "status": status}
            for mid, status in self.statuses.items()
        ]


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _model_of(body: object) -> str | None:
    if _is_object(body) and isinstance(model := body.get("model"), str):
        return model
    return None


def _error(code: int, message: str, error_type: str) -> httpx.Response:
    return httpx.Response(
        code, json={"error": {"code": code, "message": message, "type": error_type}}
    )


def _chunk(content: str | None = None, reasoning: str | None = None) -> str:
    delta: dict[str, object] = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    payload = {"choices": [{"index": 0, "finish_reason": None, "delta": delta}]}
    return "data: " + json.dumps(payload)


def _client(router: FakeRouter) -> LlamaClient:
    return LlamaClient("http://test", API_KEY, transport=httpx.MockTransport(router.handler))


@pytest.fixture
def router() -> FakeRouter:
    return FakeRouter(
        statuses={"Qwen-Q4_K_M": _status("unloaded"), "Qwen-Q8_0": _status("unloaded")}
    )


# ---- model list --------------------------------------------------------------------------


def test_parse_models_maps_every_status_and_filters_cache() -> None:
    payload = {
        "data": [
            {"id": "a-loaded", "source": "models_dir", "status": _status("loaded")},
            {"id": "b-loading", "source": "models_dir", "status": _status("loading")},
            {"id": "c-sleeping", "source": "models_dir", "status": _status("sleeping")},
            {
                "id": "d-failed",
                "source": "models_dir",
                "status": _status("unloaded", failed=True, exit_code=3),
            },
            {"id": "e-unloaded", "source": "models_dir", "status": _status("unloaded")},
            {"id": "f-weird", "source": "models_dir", "status": _status("something-new")},
            {"id": "org/cached:Q4", "source": "cache", "status": _status("unloaded")},
            {"no-id": True},
        ]
    }
    models = parse_models(payload)
    assert [(m.id, m.status) for m in models] == [
        ("a-loaded", ModelStatus.LOADED),
        ("b-loading", ModelStatus.LOADING),
        ("c-sleeping", ModelStatus.LOADED),
        ("d-failed", ModelStatus.FAILED),
        ("e-unloaded", ModelStatus.UNLOADED),
        ("f-weird", ModelStatus.UNLOADED),
    ]
    assert models[3].exit_code == 3  # noqa: PLR2004


async def test_list_models(router: FakeRouter) -> None:
    client = _client(router)
    models = await client.list_models()
    assert [m.id for m in models] == ["Qwen-Q4_K_M", "Qwen-Q8_0"]
    assert all(m.status == ModelStatus.UNLOADED for m in models)
    assert models[0].path == "/models/Qwen-Q4_K_M.gguf"
    await client.aclose()


# ---- loading -----------------------------------------------------------------------------


async def test_load_polls_until_loaded(router: FakeRouter) -> None:
    client = _client(router)
    info = await client.switch_model("Qwen-Q4_K_M", poll_interval=0)
    assert info.status == ModelStatus.LOADED
    assert ("POST", "/models/load", {"model": "Qwen-Q4_K_M"}) in router.requests
    model_polls = [r for r in router.requests if r[1] == "/models"]
    assert len(model_polls) >= router.polls_until_loaded
    await client.aclose()


async def test_switch_unloads_the_other_model_first(router: FakeRouter) -> None:
    router.statuses["Qwen-Q4_K_M"] = _status("loaded")
    client = _client(router)
    await client.switch_model("Qwen-Q8_0", poll_interval=0)
    calls = [(m, p, b) for m, p, b in router.requests if m == "POST"]
    assert calls == [
        ("POST", "/models/unload", {"model": "Qwen-Q4_K_M"}),
        ("POST", "/models/load", {"model": "Qwen-Q8_0"}),
    ]
    statuses = {m.id: m.status for m in await client.list_models()}
    assert statuses == {"Qwen-Q4_K_M": ModelStatus.UNLOADED, "Qwen-Q8_0": ModelStatus.LOADED}
    await client.aclose()


async def test_switch_to_already_loaded_model_is_a_no_op(router: FakeRouter) -> None:
    router.statuses["Qwen-Q4_K_M"] = _status("loaded")
    client = _client(router)
    info = await client.switch_model("Qwen-Q4_K_M", poll_interval=0)
    assert info.status == ModelStatus.LOADED
    assert not [r for r in router.requests if r[0] == "POST"]
    await client.aclose()


async def test_failed_load_raises_model_load_error(router: FakeRouter) -> None:
    router.fail_load.add("Qwen-Q8_0")
    client = _client(router)
    with pytest.raises(ModelLoadError, match="exit code 1"):
        await client.switch_model("Qwen-Q8_0", poll_interval=0)
    assert (await client.get_model("Qwen-Q8_0")).status == ModelStatus.FAILED
    await client.aclose()


async def test_load_timeout_raises_model_load_error(router: FakeRouter) -> None:
    router.polls_until_loaded = 10**9
    client = _client(router)
    await client.load_model("Qwen-Q4_K_M")
    with pytest.raises(ModelLoadError, match="too long"):
        await client.wait_for_status(
            "Qwen-Q4_K_M", ModelStatus.LOADED, poll_interval=0, timeout=0.05
        )
    await client.aclose()


async def test_unknown_model_raises_not_found(router: FakeRouter) -> None:
    client = _client(router)
    with pytest.raises(ModelNotFoundError):
        await client.switch_model("missing", poll_interval=0)
    with pytest.raises(ModelNotFoundError):
        await client.load_model("missing")
    await client.aclose()


# ---- chat streaming ----------------------------------------------------------------------


async def _collect(stream: AsyncIterator[ChatDelta]) -> list[ChatDelta]:
    return [delta async for delta in stream]


async def test_stream_chat_assembles_full_reply(router: FakeRouter) -> None:
    router.chat_lines = [
        'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}',
        _chunk(reasoning="Let me "),
        _chunk(reasoning="think."),
        ": keep-alive comment",
        _chunk(content="Hello"),
        _chunk(content=", **world**!"),
        'data: {"choices":[{"index":0,"finish_reason":"stop","delta":{}}],"timings":{}}',
        "data: [DONE]",
    ]
    client = _client(router)
    messages = [ChatMessage(Role.USER, "Hi")]
    deltas = await _collect(client.stream_chat("Qwen-Q4_K_M", messages))
    assert "".join(d.content for d in deltas) == "Hello, **world**!"
    assert "".join(d.reasoning for d in deltas) == "Let me think."
    method, path, body = router.requests[-1]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert body == {
        "model": "Qwen-Q4_K_M",
        "stream": True,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    await client.aclose()


async def test_stream_error_event_raises_chat_stream_error(router: FakeRouter) -> None:
    router.chat_lines = [
        _chunk(content="Partial"),
        'data: {"error":{"code":500,"message":"context full","type":"server_error"}}',
    ]
    client = _client(router)
    with pytest.raises(ChatStreamError, match="context full"):
        await _collect(client.stream_chat("m", [ChatMessage(Role.USER, "x")]))
    await client.aclose()


async def test_stream_malformed_json_raises_chat_stream_error(router: FakeRouter) -> None:
    router.chat_lines = ["data: {not json"]
    client = _client(router)
    with pytest.raises(ChatStreamError):
        await _collect(client.stream_chat("m", [ChatMessage(Role.USER, "x")]))
    await client.aclose()


class EndlessStream(httpx.AsyncByteStream):
    """An SSE body that never ends, and records whether it was closed."""

    def __init__(self) -> None:
        self.closed = False
        self.sent = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            self.sent += 1
            yield (_chunk(content=f"t{self.sent} ") + "\n\n").encode()
            await asyncio.sleep(0)

    async def aclose(self) -> None:
        self.closed = True


async def test_breaking_out_of_stream_closes_connection() -> None:
    body = EndlessStream()
    transport = httpx.MockTransport(
        lambda _req: httpx.Response(200, stream=body, headers={"Content-Type": "text/event-stream"})
    )
    client = LlamaClient("http://test", API_KEY, transport=transport)
    stream = client.stream_chat("m", [ChatMessage(Role.USER, "go")])
    received: list[str] = []
    async for delta in stream:
        received.append(delta.content)
        if len(received) == 3:  # noqa: PLR2004
            break
    await stream.aclose()
    assert received == ["t1 ", "t2 ", "t3 "]
    assert body.closed
    await client.aclose()


async def test_cancelling_the_task_stops_the_stream() -> None:
    body = EndlessStream()
    transport = httpx.MockTransport(
        lambda _req: httpx.Response(200, stream=body, headers={"Content-Type": "text/event-stream"})
    )
    client = LlamaClient("http://test", API_KEY, transport=transport)
    started = asyncio.Event()

    async def consume() -> None:
        async for _ in client.stream_chat("m", [ChatMessage(Role.USER, "go")]):
            started.set()

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert body.closed
    sent_at_cancel = body.sent
    await asyncio.sleep(0.01)
    assert body.sent == sent_at_cancel
    await client.aclose()


# ---- error mapping -----------------------------------------------------------------------


async def test_chat_http_error_becomes_api_error() -> None:
    transport = httpx.MockTransport(
        lambda _req: _error(400, "model is not loaded", "invalid_request_error")
    )
    client = LlamaClient("http://test", API_KEY, transport=transport)
    with pytest.raises(ApiError) as info:
        await _collect(client.stream_chat("m", [ChatMessage(Role.USER, "x")]))
    assert info.value.status_code == 400  # noqa: PLR2004
    assert info.value.server_message == "model is not loaded"
    await client.aclose()


async def test_server_error_with_non_json_body_becomes_api_error() -> None:
    transport = httpx.MockTransport(lambda _req: httpx.Response(500, text="boom"))
    client = LlamaClient("http://test", API_KEY, transport=transport)
    with pytest.raises(ApiError, match="boom"):
        await client.list_models()
    await client.aclose()


async def test_connection_failure_becomes_server_unavailable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = LlamaClient("http://test", API_KEY, transport=httpx.MockTransport(refuse))
    with pytest.raises(ServerUnavailableError):
        await client.list_models()
    with pytest.raises(ServerUnavailableError):
        await _collect(client.stream_chat("m", [ChatMessage(Role.USER, "x")]))
    assert await client.health() is False
    await client.aclose()


async def test_unexpected_model_list_shape_is_a_backend_error() -> None:
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, json=["not", "an", "object"]))
    client = LlamaClient("http://test", API_KEY, transport=transport)
    with pytest.raises(Exception, match="unexpected model list"):
        await client.list_models()
    await client.aclose()


async def test_api_key_is_sent(router: FakeRouter) -> None:
    unauthenticated = LlamaClient(
        "http://test", None, transport=httpx.MockTransport(router.handler)
    )
    with pytest.raises(ApiError) as info:
        await unauthenticated.list_models()
    assert info.value.status_code == 401  # noqa: PLR2004
    await unauthenticated.aclose()
