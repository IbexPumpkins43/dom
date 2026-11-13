"""An ``InferenceService`` that is built when it is first started."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dom.backend.errors import ServerUnavailableError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from dom.backend.protocols import InferenceService
    from dom.domain import ChatDelta, ChatMessage, LoadedModel, ModelInfo


class LazyInferenceService:
    """Creates the real service on ``start``, so "Try again" can pick up a newly added binary.

    ``factory`` raises a ``BackendError`` (for example ``ServerStartError``) with a friendly
    message when the real service cannot be created yet.
    """

    def __init__(self, factory: Callable[[], Awaitable[InferenceService]]) -> None:
        """Remember how to build the real service."""
        self._factory = factory
        self._inner: InferenceService | None = None

    async def start(self) -> None:
        """Build the real service if needed, then start it."""
        if self._inner is None:
            self._inner = await self._factory()
        await self._inner.start()

    async def stop(self) -> None:
        """Stop the real service, if there is one."""
        if self._inner is not None:
            await self._inner.stop()

    def stop_now(self) -> None:
        """Stop the real service synchronously, if there is one."""
        if self._inner is not None:
            self._inner.stop_now()

    async def wait_for_crash(self) -> int | None:
        """Wait for the real service to exit."""
        return await self._require().wait_for_crash()

    async def list_models(self) -> list[ModelInfo]:
        """The real service's models."""
        return await self._require().list_models()

    async def select_model(
        self, model_id: str, on_progress: Callable[[str], None] | None = None
    ) -> LoadedModel:
        """Load a model with the real service."""
        return await self._require().select_model(model_id, on_progress)

    def hardware_summary(self) -> str | None:
        """The real service's hardware summary, once it exists."""
        return self._inner.hardware_summary() if self._inner is not None else None

    def stream_chat(
        self, model_id: str, messages: Sequence[ChatMessage]
    ) -> AsyncGenerator[ChatDelta]:
        """Stream a reply from the real service."""
        return self._require().stream_chat(model_id, messages)

    def _require(self) -> InferenceService:
        if self._inner is None:
            raise ServerUnavailableError("The AI engine is not running.")
        return self._inner
