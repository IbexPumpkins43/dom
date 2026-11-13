"""The interface between the chat controller and whatever draws the UI."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dom.domain import Conversation, Message, ModelInfo


class ServerState(StrEnum):
    """What the inference server is doing, as far as the UI cares."""

    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    ENGINE_MISSING = "engine_missing"
    INSTALLING = "installing"
    CRASHED = "crashed"
    STOPPED = "stopped"


class ChatView(Protocol):
    """Everything the controller can tell the UI to show."""

    def show_server_state(self, state: ServerState, message: str | None) -> None:
        """Server status, with a plain-English explanation when something is wrong."""
        ...

    def show_models(self, models: Sequence[ModelInfo], selected_id: str | None) -> None:
        """The model picker contents and which model the user picked."""
        ...

    def show_conversations(
        self, conversations: Sequence[Conversation], current_id: str | None
    ) -> None:
        """The sidebar list (newest first) and which one is open."""
        ...

    def show_messages(self, messages: Sequence[Message]) -> None:
        """Replace the chat view with these messages."""
        ...

    def begin_reply(self) -> None:
        """An assistant reply has started streaming."""
        ...

    def update_reply(self, content: str, reasoning: str) -> None:
        """The full reply text so far (the view decides how often to redraw)."""
        ...

    def end_reply(self) -> None:
        """The streaming reply finished, was stopped, or failed."""
        ...

    def set_input_state(self, *, can_send: bool, streaming: bool) -> None:
        """Whether the input box accepts messages, and whether Stop should show."""
        ...

    def show_error(self, message: str) -> None:
        """A one-off plain-English error message."""
        ...

    def show_model_details(self, text: str | None) -> None:
        """A short line about the model in use or being prepared; ``None`` hides it."""
        ...

    def show_hardware(self, summary: str | None) -> None:
        """What was detected about this computer (e.g. for a tooltip)."""
        ...

    def show_notice(self, message: str | None) -> None:
        """A hint in the empty chat area, such as "pick a model"; ``None`` hides it."""
        ...
