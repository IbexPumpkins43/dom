"""The main window: implements ``ChatView`` with Qt widgets and forwards input to the controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dom.ui.controller import spawn
from dom.ui.protocols import ServerState
from dom.ui.style import STYLESHEET
from dom.ui.widgets import Banner, ChatArea, ChatInput, ModelPicker, Sidebar

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable, Coroutine, Sequence

    from PySide6.QtGui import QCloseEvent

    from dom.domain import Conversation, Message, ModelInfo
    from dom.ui.controller import ChatController

ERROR_DISPLAY_MS = 10_000

_SERVER_LABELS: dict[ServerState, str] = {
    ServerState.STARTING: "Starting the AI engine…",
    ServerState.READY: "AI engine running",
    ServerState.FAILED: "AI engine unavailable",
    ServerState.CRASHED: "AI engine stopped",
    ServerState.STOPPED: "AI engine stopped",
}


class MainWindow(QMainWindow):
    """The chat window."""

    def __init__(self, on_closed: Callable[[], None]) -> None:
        """Build the window; call ``attach`` before showing it."""
        super().__init__()
        self._on_closed = on_closed
        self._controller: ChatController | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._closing = False
        self._closed = False
        self._server_state = ServerState.STOPPED
        self._streaming = False
        self.setWindowTitle("dom")
        self.resize(1180, 780)
        self.setMinimumSize(760, 480)
        self.setStyleSheet(STYLESHEET)

        self._sidebar = Sidebar()
        self._picker = ModelPicker()
        self._server_label = QLabel()
        self._server_label.setObjectName("serverLabel")
        self._details_label = QLabel()
        self._details_label.setObjectName("detailsLabel")
        self._banner = Banner()
        self._chat = ChatArea()
        self._error = QLabel()
        self._error.setObjectName("errorLabel")
        self._error.setWordWrap(True)
        self._error.hide()
        self._error_timer = QTimer(self)
        self._error_timer.setSingleShot(True)
        self._error_timer.timeout.connect(self._error.hide)
        self._input = ChatInput()
        self._send = QPushButton("Send")
        self._send.setObjectName("sendButton")
        self._send.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build_layout()

    def _build_layout(self) -> None:
        top_bar = QWidget()
        top_bar.setObjectName("topBar")
        top = QHBoxLayout(top_bar)
        top.setContentsMargins(16, 10, 16, 10)
        top.addWidget(self._picker)
        top.addSpacing(12)
        top.addWidget(self._details_label)
        top.addStretch(1)
        top.addWidget(self._server_label)

        composer = QWidget()
        composer.setObjectName("composer")
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(10, 8, 10, 8)
        composer_layout.addWidget(self._input, 1)
        composer_layout.addWidget(self._send, 0, Qt.AlignmentFlag.AlignBottom)
        composer_row = QHBoxLayout()
        composer_row.setContentsMargins(16, 0, 16, 16)
        composer_row.addStretch(1)
        composer_row.addWidget(composer, 100)
        composer_row.addStretch(1)
        composer.setMaximumWidth(900)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(top_bar)
        main_layout.addWidget(self._banner)
        main_layout.addWidget(self._chat, 1)
        error_row = QHBoxLayout()
        error_row.setContentsMargins(16, 0, 16, 6)
        error_row.addWidget(self._error)
        main_layout.addLayout(error_row)
        main_layout.addLayout(composer_row)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._sidebar)
        layout.addWidget(main, 1)
        self.setCentralWidget(central)

    def attach(self, controller: ChatController) -> None:
        """Connect UI events to ``controller``."""
        self._controller = controller
        self._input.submitted.connect(self._submit)
        self._send.clicked.connect(self._send_or_stop)
        self._picker.model_chosen.connect(self._on_model_chosen)
        self._sidebar.new_chat_requested.connect(lambda: self._run(controller.new_chat()))
        self._sidebar.conversation_selected.connect(self._on_conversation_selected)
        self._banner.action_clicked.connect(lambda: self._run(controller.restart_server()))
        self.set_input_state(can_send=False, streaming=False)

    # ---- ChatView ------------------------------------------------------------------------

    def show_server_state(self, state: ServerState, message: str | None) -> None:
        """Update the status label and the banner."""
        self._server_state = state
        self._server_label.setText(_SERVER_LABELS[state])
        self._server_label.setProperty("state", state.value)
        self._server_label.style().unpolish(self._server_label)
        self._server_label.style().polish(self._server_label)
        if state == ServerState.READY:
            self._banner.hide()
        elif state == ServerState.STARTING:
            self._banner.show_message("Starting the AI engine…", None, error=False)
        elif state == ServerState.CRASHED:
            self._banner.show_message(message or "The AI engine stopped.", "Restart", error=True)
        elif state == ServerState.FAILED:
            self._banner.show_message(
                message or "The AI engine could not start.", "Try again", error=True
            )
        self._picker.setEnabled(state == ServerState.READY and not self._streaming)

    def show_models(self, models: Sequence[ModelInfo], selected_id: str | None) -> None:
        """Fill the model picker."""
        self._picker.set_models(models, selected_id)

    def show_conversations(
        self, conversations: Sequence[Conversation], current_id: str | None
    ) -> None:
        """Fill the sidebar."""
        self._sidebar.set_conversations(conversations, current_id)

    def show_messages(self, messages: Sequence[Message]) -> None:
        """Show a conversation."""
        self._chat.set_messages(messages)

    def begin_reply(self) -> None:
        """Start showing a streaming reply."""
        self._chat.begin_reply()

    def update_reply(self, content: str, reasoning: str) -> None:
        """Update the streaming reply (redraws are throttled)."""
        self._chat.update_reply(content, reasoning)

    def end_reply(self) -> None:
        """Finish the streaming reply."""
        self._chat.end_reply()

    def set_input_state(self, *, can_send: bool, streaming: bool) -> None:
        """Enable or disable the input; swap Send for Stop while streaming."""
        self._streaming = streaming
        self._input.setEnabled(can_send)
        self._send.setText("Stop" if streaming else "Send")
        self._send.setProperty("stop", streaming)
        self._send.style().unpolish(self._send)
        self._send.style().polish(self._send)
        self._send.setEnabled(streaming or can_send)
        self._picker.setEnabled(self._server_state == ServerState.READY and not streaming)
        if can_send:
            self._input.setFocus()

    def show_error(self, message: str) -> None:
        """Show an error above the input for a few seconds."""
        self._error.setText(message)
        self._error.show()
        self._error_timer.start(ERROR_DISPLAY_MS)

    def show_notice(self, message: str | None) -> None:
        """Show or hide the hint in the empty chat area."""
        self._chat.set_notice(message)

    def show_model_details(self, text: str | None) -> None:
        """Show what the model runs on (or what dom is doing to prepare it)."""
        self._details_label.setText(text or "")

    def show_hardware(self, summary: str | None) -> None:
        """Put the detected hardware in tooltips."""
        tip = f"This computer: {summary}" if summary else ""
        self._server_label.setToolTip(tip)
        self._details_label.setToolTip(tip)

    # ---- events --------------------------------------------------------------------------

    def _submit(self) -> None:
        controller = self._controller
        if controller is None or not self._input.isEnabled():
            return
        text = self._input.toPlainText()
        if not text.strip():
            return
        self._input.clear()
        self._run(controller.send(text))

    def _on_model_chosen(self, model_id: str) -> None:
        if self._controller is not None:
            self._run(self._controller.select_model(model_id))

    def _on_conversation_selected(self, conversation_id: str) -> None:
        if self._controller is not None:
            self._run(self._controller.open_conversation(conversation_id))

    def _send_or_stop(self) -> None:
        if self._controller is not None and self._streaming:
            self._run(self._controller.stop_reply())
        else:
            self._submit()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 -- Qt override
        """Stop the AI server before the window actually closes."""
        if self._closed or self._controller is None:
            event.accept()
            self._finish_close()
            return
        event.ignore()
        if self._closing:
            return
        self._closing = True
        self.setEnabled(False)
        self.setWindowTitle("dom — closing…")
        self._run(self._shutdown_then_close(self._controller))

    async def _shutdown_then_close(self, controller: ChatController) -> None:
        try:
            await controller.shutdown()
        finally:
            self._closed = True
            self.close()

    def _finish_close(self) -> None:
        self._on_closed()

    def _run(self, coro: Coroutine[object, object, None]) -> None:
        spawn(coro, self._tasks)
