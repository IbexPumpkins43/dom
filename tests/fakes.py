"""Test doubles for the UI layer's protocols."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dom.backend.errors import (
    BackendError,
    EngineMissingError,
    ModelLoadError,
    ServerStartError,
)
from dom.domain import ChatDelta, ChatMessage, LoadedModel, ModelInfo, ModelStatus
from dom.downloads import DownloadError
from dom.downloads.hub import HubPage, ParamRange
from dom.hardware import HardwareProfile, MemoryStatus
from dom.hardware.models import GIB

TEST_HARDWARE = HardwareProfile(
    os_name="Test OS",
    arch="x86_64",
    physical_cores=8,
    logical_cores=16,
    memory=MemoryStatus(16 * GIB, 10 * GIB),
    gpus=(),
    build=None,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    from dom.catalog import HubModel
    from dom.config.settings import GenerationSettings
    from dom.domain import Conversation, Message, Profile
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
    engine_missing: bool = False
    load_error: str | None = None
    stream_error_after: int | None = None
    endless_stream: bool = False
    starts: int = 0
    stopped: bool = False
    stream_closed: bool = False
    received: list[list[ChatMessage]] = field(default_factory=list[list[ChatMessage]])
    received_settings: list[GenerationSettings | None] = field(
        default_factory=list["GenerationSettings | None"]
    )
    _loaded: str | None = None
    _crash: asyncio.Future[int | None] | None = None
    first_token: asyncio.Event = field(default_factory=asyncio.Event)
    progress_seen: list[str] = field(default_factory=list[str])
    reloads: int = 0
    unloaded: list[str] = field(default_factory=list[str])
    endless_started: asyncio.Event = field(default_factory=asyncio.Event)

    async def start(self) -> None:
        self.starts += 1
        if self.engine_missing:
            raise EngineMissingError("llama-server is missing")
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

    async def forget_model(self, model_id: str) -> None:
        self.unloaded.append(model_id)
        if self._loaded == model_id:
            self._loaded = None

    def hardware_profile(self) -> HardwareProfile | None:
        return TEST_HARDWARE

    async def reload_models(self) -> list[ModelInfo]:
        self.reloads += 1
        return await self.list_models()

    async def stream_chat(
        self,
        model_id: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings | None = None,
    ) -> AsyncGenerator[ChatDelta]:
        self.received.append(list(messages))
        self.received_settings.append(settings)
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
class FakeEngine:
    """An ``EngineDownloads`` that pretends to install, optionally failing."""

    service: FakeInferenceService
    error: str | None = None
    installs: int = 0

    async def install(self, progress: Callable[[str, int, int], None]) -> None:
        self.installs += 1
        progress("CPU engine", 50, 100)
        await asyncio.sleep(0)
        if self.error is not None:
            raise DownloadError(self.error)
        self.service.start_error = None
        self.service.engine_missing = False


@dataclass
class FakeDownloads:
    """A ``ModelDownloads`` over an in-memory folder."""

    files: set[str] = field(default_factory=set[str])
    sizes: dict[str, int] = field(default_factory=dict[str, int])
    partial: dict[str, int] = field(default_factory=dict[str, int])
    origin: dict[str, str] = field(default_factory=dict[str, str])
    error: str | None = None
    delete_error: str | None = None
    block: asyncio.Event | None = None
    downloaded: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    deleted: list[str] = field(default_factory=list[str])

    async def installed_files(self) -> dict[str, int]:
        return {name: self.sizes.get(name, GIB) for name in self.files}

    async def sources(self) -> dict[str, str]:
        return dict(self.origin)

    async def partial_bytes(self, filename: str) -> int:
        return self.partial.get(filename, 0)

    async def download(self, repo: str, path: str, progress: Callable[[int, int], None]) -> None:
        progress(1, 2)
        if self.block is not None:
            await self.block.wait()
        if self.error is not None:
            raise DownloadError(self.error)
        progress(2, 2)
        filename = path.rsplit("/", 1)[-1]
        self.files.add(filename)
        self.origin[filename] = repo
        self.downloaded.append((repo, path))

    async def delete(self, filename: str) -> None:
        if self.delete_error is not None:
            raise DownloadError(self.delete_error)
        self.files.discard(filename)
        self.origin.pop(filename, None)
        self.deleted.append(filename)


@dataclass
class FakeSource:
    """A ``ModelSource`` returning fixed pages; ``more`` is the page after the first."""

    models: list[HubModel] = field(default_factory=list["HubModel"])
    more: list[HubModel] = field(default_factory=list["HubModel"])
    error: str | None = None
    queries: list[str] = field(default_factory=list[str])
    sizes: list[ParamRange] = field(default_factory=list["ParamRange"])

    async def search(self, query: str, size: ParamRange = (None, None)) -> HubPage:
        self.queries.append(query)
        self.sizes.append(size)
        await asyncio.sleep(0)
        if self.error is not None:
            raise DownloadError(self.error)
        words = query.lower().split()
        found = [m for m in self.models if all(w in m.repo.lower() for w in words)]
        return HubPage(tuple(found), "next" if self.more and not query else None)

    async def page(self, url: str) -> HubPage:
        assert url == "next"
        return HubPage(tuple(self.more), None)


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


@dataclass
class RecordingProfileView:
    """A ``ProfileView`` that records what it was told to show."""

    profiles: list[Profile] = field(default_factory=list["Profile"])
    current: Profile | None = None
    errors: list[str] = field(default_factory=list[str])

    def show_profiles(self, profiles: Sequence[Profile], current: Profile) -> None:
        self.profiles = list(profiles)
        self.current = current

    def show_error(self, message: str) -> None:
        self.errors.append(message)
