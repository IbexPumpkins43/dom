"""Test doubles for the UI layer's protocols."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dom.backend.errors import BackendError, ModelLoadError, ServerStartError
from dom.domain import ChatDelta, ChatMessage, LoadedModel, ModelInfo, ModelStatus

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    from dom.domain import Conversation, Message
    from dom.ui.protocols import ServerState


@dataclass
class FakeInferenceService:
    """An in-memory ``InferenceService``."""

    model_ids: list[str] = field(default_factory=lambda: ["model-a", "model-b"])
    reply: list[ChatDelta] = field(
        default_factory=lambda: [
            ChatDelta(reasoning="Thinking "),
            ChatDelta(reasoning="hard."),
            ChatDelta(content="Hello"),
            ChatDelta(content=" there!"),
        ]
    )
    start_error: str | None = None
    load_error: str | None = None
    stream_error_after: int | None = None
    endless_stream: bool = False
    starts: int = 0
    stopped: bool = False
    stream_closed: bool = False
    received: list[list[ChatMessage]] = field(default_factory=list[list[ChatMessage]])
    _loaded: str | None = None
    _crash: asyncio.Future[int | None] | None = None
    first_token: asyncio.Event = field(default_factory=asyncio.Event)
    progress_seen: list[str] = field(default_factory=list[str])
    endless_started: asyncio.Event = field(default_factory=asyncio.Event)

    async def start(self) -> None:
        self.starts += 1
        if self.start_error is not None:
            raise ServerStartError(self.start_error)
        self._crash = asyncio.get_running_loop().create_future()
        self._loaded = None

    async def stop(self) -> None:
        self.stopped = True
        if self._crash is not None and not self._crash.done():
            self._crash.set_result(None)

    def stop_now(self) -> None:
        self.stopped = True

    async def wait_for_crash(self) -> int | None:
        assert self._crash is not None
        return await self._crash

    def crash(self, code: int) -> None:
        assert self._crash is not None
        self._crash.set_result(code)

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(m, ModelStatus.LOADED if m == self._loaded else ModelStatus.UNLOADED)
            for m in self.model_ids
        ]

    async def select_model(
        self, model_id: str, on_progress: Callable[[str], None] | None = None
    ) -> LoadedModel:
        if on_progress is not None:
            on_progress("Testing speed…")
            self.progress_seen.append("Testing speed…")
        await asyncio.sleep(0)
        if self.load_error is not None:
            raise ModelLoadError(self.load_error)
        self._loaded = model_id
        return LoadedModel(
            ModelInfo(model_id, ModelStatus.LOADED), 32768, "Test GPU (GPU)", tuned=True
        )

    def hardware_summary(self) -> str | None:
        return "Test OS, 8 CPU cores, 16 GB RAM"

    async def stream_chat(
        self, model_id: str, messages: Sequence[ChatMessage]
    ) -> AsyncGenerator[ChatDelta]:
        self.received.append(list(messages))
        try:
            for i, delta in enumerate(self.reply):
                if self.stream_error_after is not None and i >= self.stream_error_after:
                    raise BackendError("The connection dropped.")
                yield delta
                self.first_token.set()
                await asyncio.sleep(0)
            while self.endless_stream:
                self.endless_started.set()
                await asyncio.sleep(0.01)
                yield ChatDelta(content=".")
        finally:
            self.stream_closed = True


@dataclass
class RecordingView:
    """A ``ChatView`` that records what it was told to show."""

    server_states: list[tuple[ServerState, str | None]] = field(
        default_factory=list[tuple["ServerState", str | None]]
    )
    models: list[ModelInfo] = field(default_factory=list[ModelInfo])
    selected_model: str | None = None
    conversations: list[Conversation] = field(default_factory=list["Conversation"])
    current_conversation: str | None = None
    messages: list[Message] = field(default_factory=list["Message"])
    reply: tuple[str, str] = ("", "")
    reply_active: bool = False
    can_send: bool = False
    streaming: bool = False
    errors: list[str] = field(default_factory=list[str])
    notice: str | None = None
    model_details: list[str | None] = field(default_factory=list[str | None])
    hardware: str | None = None

    def show_server_state(self, state: ServerState, message: str | None) -> None:
        self.server_states.append((state, message))

    def show_models(self, models: Sequence[ModelInfo], selected_id: str | None) -> None:
        self.models = list(models)
        self.selected_model = selected_id

    def show_conversations(
        self, conversations: Sequence[Conversation], current_id: str | None
    ) -> None:
        self.conversations = list(conversations)
        self.current_conversation = current_id

    def show_messages(self, messages: Sequence[Message]) -> None:
        self.messages = list(messages)

    def begin_reply(self) -> None:
        self.reply_active = True
        self.reply = ("", "")

    def update_reply(self, content: str, reasoning: str) -> None:
        self.reply = (content, reasoning)

    def end_reply(self) -> None:
        self.reply_active = False

    def set_input_state(self, *, can_send: bool, streaming: bool) -> None:
        self.can_send = can_send
        self.streaming = streaming

    def show_error(self, message: str) -> None:
        self.errors.append(message)

    def show_notice(self, message: str | None) -> None:
        self.notice = message

    def show_model_details(self, text: str | None) -> None:
        self.model_details.append(text)

    def show_hardware(self, summary: str | None) -> None:
        self.hardware = summary
