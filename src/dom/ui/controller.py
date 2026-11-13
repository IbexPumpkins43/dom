"""Chat controller: all UI behaviour that does not need Qt, so it can be tested with fakes."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from dom.backend.errors import BackendError, EngineMissingError
from dom.domain import ChatMessage, LoadedModel, Message, ModelInfo, ModelStatus, Role
from dom.downloads import DownloadError
from dom.storage.errors import StorageError
from dom.ui.protocols import ServerState

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from dom.backend.protocols import InferenceService
    from dom.downloads.protocols import EngineDownloads
    from dom.storage.protocols import ConversationRepository
    from dom.ui.protocols import ChatView

logger = logging.getLogger(__name__)

TITLE_MAX_CHARS = 60
MODEL_POLL_INTERVAL_S = 0.5


def describe_loaded(loaded: LoadedModel) -> str:
    """E.g. "Running on Intel Iris Xe (GPU) · 32k context"."""
    text = f"Running on {loaded.device_label}"
    if loaded.ctx_size:
        text += f" · {loaded.ctx_size // 1024}k context"
    return text


def make_title(text: str) -> str:
    """A sidebar title from the first line of the first message."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "New chat")
    if len(first_line) <= TITLE_MAX_CHARS:
        return first_line
    return first_line[: TITLE_MAX_CHARS - 1].rstrip() + "…"


class ChatController:
    """Coordinates the inference service, the conversation store and the view.

    All public coroutines catch expected errors and report them to the view in plain English;
    they are safe to fire-and-forget from UI callbacks.
    """

    def __init__(
        self,
        service: InferenceService,
        repository: ConversationRepository,
        view: ChatView,
        engine: EngineDownloads,
    ) -> None:
        """Wire up the controller; nothing starts until ``startup`` is awaited."""
        self._service = service
        self._engine = engine
        self._repo = repository
        self._view = view
        self._server_state = ServerState.STOPPED
        self._models: list[ModelInfo] = []
        self._selected_model: str | None = None
        self._conversation_id: str | None = None
        self._messages: list[Message] = []
        self._reply_task: asyncio.Task[None] | None = None
        self._load_task: asyncio.Task[None] | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._shut_down = False
        self._loading_model: str | None = None

    # ---- lifecycle -----------------------------------------------------------------------

    async def startup(self) -> None:
        """Load the sidebar and start the inference server."""
        await self._refresh_conversations()
        self._view.show_messages([])
        await self._start_server()

    async def restart_server(self) -> None:
        """Start the server again after a crash or failed start, reloading the picked model."""
        if self._server_state in (
            ServerState.STARTING, ServerState.READY, ServerState.INSTALLING
        ):  # fmt: skip
            return
        await self._start_server()
        if self.server_state == ServerState.READY and self._selected_model is not None:
            await self.select_model(self._selected_model)

    @property
    def server_state(self) -> ServerState:
        """The current server state."""
        return self._server_state

    async def shutdown(self) -> None:
        """Stop any reply, stop the server and close storage."""
        if self._shut_down:
            return
        self._shut_down = True
        for task in (self._reply_task, self._load_task, self._watch_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        try:
            await self._service.stop()
        except BackendError:
            logger.exception("Error while stopping the AI server")
        await asyncio.to_thread(self._repo.close)

    async def _start_server(self) -> None:
        self._set_server_state(ServerState.STARTING, None)
        try:
            await self._service.start()
        except EngineMissingError as exc:
            logger.warning("AI engine missing: %s", exc)
            self._set_server_state(ServerState.ENGINE_MISSING, str(exc))
            return
        except BackendError as exc:
            logger.warning("Server start failed: %s", exc)
            self._set_server_state(ServerState.FAILED, str(exc))
            return
        self._set_server_state(ServerState.READY, None)
        profile = self._service.hardware_profile()
        self._view.show_hardware(profile.summary() if profile is not None else None)
        self._watch_task = asyncio.create_task(self._watch_for_crash())
        await self._refresh_models()

    async def install_engine(self) -> None:
        """Download the AI engine (llama.cpp), then start it."""
        if self._server_state != ServerState.ENGINE_MISSING:
            return
        self._set_server_state(ServerState.INSTALLING, "Downloading the AI engine…")

        def progress(label: str, done: int, total: int) -> None:
            pct = 100 * done // total if total else 100
            mb = f"{done / 1e6:.0f} of {total / 1e6:.0f} MB"
            self._view.show_server_state(
                ServerState.INSTALLING, f"Downloading the {label}: {pct}% ({mb})"
            )

        try:
            await self._engine.install(progress)
        except DownloadError as exc:
            self._set_server_state(
                ServerState.ENGINE_MISSING, f"The AI engine could not be downloaded. {exc}"
            )
            return
        await self._start_server()

    async def models_changed(self) -> None:
        """Re-scan the models folder after a download."""
        if self._server_state == ServerState.READY:
            await self._refresh_models(rescan=True)

    async def _watch_for_crash(self) -> None:
        code = await self._service.wait_for_crash()
        if code is None or self._shut_down:
            return
        if self._reply_task is not None:
            self._reply_task.cancel()
        if self._load_task is not None:
            self._load_task.cancel()
        self._models = [replace(m, status=ModelStatus.UNLOADED) for m in self._models]
        self._view.show_models(self._models, self._selected_model)
        self._view.show_model_details(None)
        self._set_server_state(
            ServerState.CRASHED,
            f"The AI engine stopped unexpectedly (exit code {code}). "
            "Click Restart to start it again.",
        )

    # ---- models --------------------------------------------------------------------------

    async def select_model(self, model_id: str) -> None:
        """Load ``model_id`` (unloading any other) and show progress in the picker."""
        if self._server_state != ServerState.READY or self.is_streaming:
            return
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._load_task
        self._selected_model = model_id
        self._loading_model = model_id
        self._models = [self._as_loading(m) for m in self._models]
        self._view.show_models(self._models, model_id)
        self._update_input()
        self._load_task = asyncio.create_task(self._load(model_id))
        await self._load_task

    async def _load(self, model_id: str) -> None:
        self._view.show_model_details("Preparing the model…")
        poller = asyncio.create_task(self._poll_models_while_loading())
        try:
            loaded = await self._service.select_model(model_id, self._view.show_model_details)
        except BackendError as exc:
            self._view.show_model_details(None)
            self._view.show_error(f"Could not load {model_id}. {exc}")
        else:
            self._view.show_model_details(describe_loaded(loaded))
        finally:
            self._loading_model = None
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller
        await self._refresh_models()

    def _as_loading(self, model: ModelInfo) -> ModelInfo:
        if model.id == self._loading_model and model.status != ModelStatus.LOADED:
            return replace(model, status=ModelStatus.LOADING)
        return model

    async def _poll_models_while_loading(self) -> None:
        while True:
            await asyncio.sleep(MODEL_POLL_INTERVAL_S)
            await self._refresh_models()

    async def _refresh_models(self, *, rescan: bool = False) -> None:
        try:
            if rescan:
                models = await self._service.reload_models()
            else:
                models = await self._service.list_models()
        except BackendError as exc:
            self._view.show_error(str(exc))
            return
        # While dom measures, benchmarks or re-tunes a model the server still calls it unloaded.
        self._models = [self._as_loading(m) for m in models]
        self._view.show_models(self._models, self._selected_model)
        self._update_input()

    @property
    def loaded_model(self) -> str | None:
        """The selected model, if it is loaded and ready for chat."""
        for model in self._models:
            if model.id == self._selected_model and model.status == ModelStatus.LOADED:
                return model.id
        return None

    # ---- conversations -------------------------------------------------------------------

    async def new_chat(self) -> None:
        """Start a fresh conversation (saved once the first message is sent)."""
        await self.stop_reply()
        self._conversation_id = None
        self._messages = []
        self._view.show_messages([])
        await self._refresh_conversations()
        self._update_input()

    async def open_conversation(self, conversation_id: str) -> None:
        """Show a saved conversation."""
        if conversation_id == self._conversation_id:
            return
        await self.stop_reply()
        try:
            messages = await asyncio.to_thread(self._repo.get_messages, conversation_id)
        except StorageError as exc:
            self._view.show_error(str(exc))
            return
        self._conversation_id = conversation_id
        self._messages = messages
        self._view.show_messages(messages)
        await self._refresh_conversations()
        self._update_input()

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a saved conversation (stopping its reply first if it is the open one)."""
        is_open = conversation_id == self._conversation_id
        if is_open:
            await self.stop_reply()
        try:
            await asyncio.to_thread(self._repo.delete_conversation, conversation_id)
        except StorageError as exc:
            self._view.show_error(f"Could not delete the chat. {exc}")
            return
        if is_open:
            self._conversation_id = None
            self._messages = []
            self._view.show_messages([])
        await self._refresh_conversations()
        self._update_input()

    async def _refresh_conversations(self) -> None:
        try:
            conversations = await asyncio.to_thread(self._repo.list_conversations)
        except StorageError as exc:
            self._view.show_error(str(exc))
            return
        self._view.show_conversations(conversations, self._conversation_id)

    # ---- chatting ------------------------------------------------------------------------

    @property
    def is_streaming(self) -> bool:
        """True while a reply is being generated."""
        return self._reply_task is not None and not self._reply_task.done()

    async def send(self, text: str) -> None:
        """Save the user's message and stream the assistant's reply."""
        text = text.strip()
        model_id = self.loaded_model
        if not text or model_id is None or self.is_streaming:
            return
        try:
            if self._conversation_id is None:
                conversation = await asyncio.to_thread(
                    self._repo.create_conversation, make_title(text)
                )
                self._conversation_id = conversation.id
            message = await asyncio.to_thread(
                self._repo.add_message, self._conversation_id, Role.USER, text
            )
        except StorageError as exc:
            self._view.show_error(str(exc))
            return
        self._messages.append(message)
        self._view.show_messages(self._messages)
        await self._refresh_conversations()
        self._reply_task = asyncio.create_task(self._stream_reply(self._conversation_id, model_id))
        self._update_input()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reply_task
        self._update_input()

    async def stop_reply(self) -> None:
        """Stop the reply being generated; what was received so far is kept."""
        task = self._reply_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._update_input()

    async def _stream_reply(self, conversation_id: str, model_id: str) -> None:
        history = [ChatMessage(m.role, m.content) for m in self._messages]
        content: list[str] = []
        reasoning: list[str] = []
        self._view.begin_reply()
        try:
            # aclosing: close the stream in this task even when cancelled, so the HTTP
            # connection is released here rather than by the garbage collector in another task.
            async with contextlib.aclosing(self._service.stream_chat(model_id, history)) as stream:
                async for delta in stream:
                    content.append(delta.content)
                    reasoning.append(delta.reasoning)
                    self._view.update_reply("".join(content), "".join(reasoning))
        except asyncio.CancelledError:
            logger.info("Reply stopped by the user")
        except BackendError as exc:
            self._view.show_error(str(exc))
        finally:
            self._view.end_reply()
        await self._save_reply(conversation_id, model_id, "".join(content), "".join(reasoning))

    async def _save_reply(
        self, conversation_id: str, model_id: str, content: str, reasoning: str
    ) -> None:
        if not content and not reasoning:
            self._view.show_messages(self._messages)
            return
        try:
            message = await asyncio.to_thread(
                self._repo.add_message,
                conversation_id,
                Role.ASSISTANT,
                content,
                reasoning=reasoning or None,
                model_id=model_id,
            )
        except StorageError as exc:
            self._view.show_error(str(exc))
            return
        if conversation_id == self._conversation_id:
            self._messages.append(message)
            self._view.show_messages(self._messages)
        await self._refresh_conversations()

    # ---- helpers -------------------------------------------------------------------------

    def _set_server_state(self, state: ServerState, message: str | None) -> None:
        self._server_state = state
        self._view.show_server_state(state, message)
        self._update_input()

    def _update_input(self) -> None:
        streaming = self.is_streaming
        ready = self._server_state == ServerState.READY and self.loaded_model is not None
        self._view.set_input_state(can_send=ready and not streaming, streaming=streaming)
        self._view.show_notice(self._notice())

    def _notice(self) -> str | None:
        if self._server_state != ServerState.READY or self._messages:
            return None
        if not self._models:
            return "No models yet. Click “Get models” on the left to download one."
        if self._selected_model is None:
            return "Pick a model below to start chatting."
        if self.loaded_model is None:
            return "Loading the model…"
        return None


def spawn(coro: Coroutine[object, object, None], tasks: set[asyncio.Task[None]]) -> None:
    """Run a controller coroutine from a UI callback, keeping a reference until it finishes."""
    task = asyncio.ensure_future(coro)
    tasks.add(task)

    def _done(t: asyncio.Task[None]) -> None:
        tasks.discard(t)
        if not t.cancelled() and (exc := t.exception()) is not None:
            logger.error("Unexpected error in UI task", exc_info=exc)

    task.add_done_callback(_done)
