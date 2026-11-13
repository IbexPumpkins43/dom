"""Backend interfaces the UI depends on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from dom.domain import ChatDelta, ChatMessage, ModelInfo


class InferenceService(Protocol):
    """Runs local models and streams chat replies."""

    async def start(self) -> None:
        """Start the inference server and wait until it is ready."""
        ...

    async def stop(self) -> None:
        """Stop the inference server and every model process."""
        ...

    def stop_now(self) -> None:
        """Synchronous ``stop`` for shutdown paths that cannot await."""
        ...

    async def wait_for_crash(self) -> int | None:
        """Wait until the server exits; return its exit code, or ``None`` if it was stopped."""
        ...

    async def list_models(self) -> list[ModelInfo]:
        """Return the local models and their status."""
        ...

    async def select_model(self, model_id: str) -> ModelInfo:
        """Make ``model_id`` the only loaded model and wait until it is ready."""
        ...

    def stream_chat(
        self, model_id: str, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[ChatDelta]:
        """Stream a reply; stop iterating or cancel to stop generation."""
        ...
