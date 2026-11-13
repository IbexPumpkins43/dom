"""``InferenceService`` backed by a local llama-server router."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dom.backend.client import LlamaClient
from dom.backend.errors import ServerUnavailableError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    import httpx

    from dom.backend.process import LlamaServerProcess
    from dom.domain import ChatDelta, ChatMessage, ModelInfo

logger = logging.getLogger(__name__)


class LlamaInferenceService:
    """Owns one ``LlamaServerProcess`` and the ``LlamaClient`` that talks to it."""

    def __init__(
        self,
        process: LlamaServerProcess,
        transport_factory: Callable[[], httpx.AsyncBaseTransport | None] = lambda: None,
    ) -> None:
        """Wrap ``process``; ``transport_factory`` lets tests swap the HTTP transport."""
        self._process = process
        self._transport_factory = transport_factory
        self._client: LlamaClient | None = None
        self._stopping = False

    async def start(self) -> None:
        """Start (or restart) the server and connect to it."""
        await self._close_client()
        self._stopping = False
        endpoint = await self._process.start()
        self._client = LlamaClient(
            endpoint.base_url, endpoint.api_key, transport=self._transport_factory()
        )

    async def stop(self) -> None:
        """Stop the server and all model processes."""
        self._stopping = True
        await self._close_client()
        await self._process.stop()

    def stop_now(self) -> None:
        """Stop the server synchronously (for interpreter/app shutdown)."""
        self._stopping = True
        self._client = None
        self._process.stop_now()

    async def wait_for_crash(self) -> int | None:
        """Wait until the server exits; ``None`` means it was stopped on purpose."""
        code = await self._process.wait_for_exit()
        if self._stopping:
            return None
        logger.error("llama-server exited unexpectedly with code %s", code)
        logger.error("Last server log lines:\n%s", self._process.log_tail())
        # The router is gone, but a model process it started may not be: clean up the tree.
        await self._close_client()
        await self._process.stop()
        return code

    async def list_models(self) -> list[ModelInfo]:
        """Return the local models and their status."""
        return await self._require_client().list_models()

    async def select_model(self, model_id: str) -> ModelInfo:
        """Make ``model_id`` the only loaded model and wait until it is ready."""
        return await self._require_client().switch_model(model_id)

    def stream_chat(
        self, model_id: str, messages: Sequence[ChatMessage]
    ) -> AsyncGenerator[ChatDelta]:
        """Stream a reply; stop iterating or cancel to stop generation."""
        return self._require_client().stream_chat(model_id, messages)

    def _require_client(self) -> LlamaClient:
        if self._client is None:
            raise ServerUnavailableError("The AI server is not running.")
        return self._client

    async def _close_client(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
