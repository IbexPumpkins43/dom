"""Backend interfaces the UI depends on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    from dom.domain import ChatDelta, ChatMessage, LoadedModel, ModelInfo
    from dom.hardware import HardwareProfile


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

    async def select_model(
        self, model_id: str, on_progress: Callable[[str], None] | None = None
    ) -> LoadedModel:
        """Make ``model_id`` the only loaded model, tuned for this computer, and wait for it.

        ``on_progress`` receives plain-English steps (e.g. "Testing speed…") for the UI.
        """
        ...

    def hardware_profile(self) -> HardwareProfile | None:
        """What was detected about this computer, once the server has started."""
        ...

    async def reload_models(self) -> list[ModelInfo]:
        """Re-scan the models folder (after a download) and return the models."""
        ...

    def stream_chat(
        self, model_id: str, messages: Sequence[ChatMessage]
    ) -> AsyncGenerator[ChatDelta]:
        """Stream a reply; stop iterating or cancel to stop generation."""
        ...
