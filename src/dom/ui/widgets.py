"""Qt widgets for the chat window."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dom.domain import ModelStatus, Role

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PySide6.QtGui import QKeyEvent

    from dom.domain import Conversation, Message, ModelInfo

# Redraw a streaming reply at most this often; re-rendering Markdown per token is slow.
REPLY_REDRAW_MS = 50
INPUT_MAX_LINES = 8
SIDEBAR_WIDTH = 260
CHAT_MAX_WIDTH = 860
USER_BUBBLE_MAX_WIDTH = 620
_BUBBLE_H_PADDING = 28

_STATUS_LABELS: dict[ModelStatus, str] = {
    ModelStatus.UNLOADED: "not loaded",
    ModelStatus.LOADING: "loading…",
    ModelStatus.LOADED: "ready",
    ModelStatus.FAILED: "failed to load",
}
_STATUS_DOTS: dict[ModelStatus, str] = {
    ModelStatus.UNLOADED: "○",
    ModelStatus.LOADING: "◐",
    ModelStatus.LOADED: "●",
    ModelStatus.FAILED: "✕",
}


class ChatInput(QPlainTextEdit):
    """Multi-line message box: Enter sends, Shift+Enter adds a new line."""

    submitted = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the input box."""
        super().__init__(parent)
        self.setObjectName("chatInput")
        self.setPlaceholderText("Message dom…  (Enter to send, Shift+Enter for a new line)")
        self.setTabChangesFocus(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.document().contentsChanged.connect(self._fit_height)
        self._fit_height()

    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802 -- Qt override
        """Send on Enter; insert a newline on Shift+Enter."""
        is_enter = e.key() in (Qt.Key.Key_Return.value, Qt.Key.Key_Enter.value)
        shift = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if is_enter and not shift:
            e.accept()
            self.submitted.emit()
            return
        super().keyPressEvent(e)

    def _fit_height(self) -> None:
        line = self.fontMetrics().lineSpacing()
        lines = max(1, min(INPUT_MAX_LINES, int(self.document().size().height())))
        margins = self.contentsMargins()
        doc_margin = int(self.document().documentMargin() * 2)
        self.setFixedHeight(lines * line + doc_margin + margins.top() + margins.bottom() + 4)


class MessageBubble(QFrame):
    """One message: a user bubble on the right, or an assistant reply rendered as Markdown."""

    def __init__(self, role: Role, content: str = "", reasoning: str = "") -> None:
        """Create a bubble for ``role``."""
        super().__init__()
        self._role = role
        self.setObjectName("userBubble" if role == Role.USER else "assistantBubble")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        self._thinking_toggle = QToolButton()
        self._thinking_toggle.setObjectName("thinkingToggle")
        self._thinking_toggle.setCheckable(True)
        self._thinking_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._thinking_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._thinking_toggle.toggled.connect(self._toggle_thinking)
        self._thinking = _selectable_label(Qt.TextFormat.PlainText)
        self._thinking.setObjectName("thinkingText")
        self._thinking.hide()
        self._thinking_toggle.hide()
        layout.addWidget(self._thinking_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._thinking)

        text_format = Qt.TextFormat.PlainText if role == Role.USER else Qt.TextFormat.MarkdownText
        self._content = _selectable_label(text_format)
        layout.addWidget(self._content)
        if role == Role.USER:
            # A word-wrapped label sizes itself narrowly; give short messages their natural
            # width (up to a maximum) so they read like a chat bubble.
            self.setMaximumWidth(USER_BUBBLE_MAX_WIDTH)
            metrics = self._content.fontMetrics()
            natural = max((metrics.horizontalAdvance(ln) for ln in content.splitlines()), default=0)
            self._content.setMinimumWidth(
                min(natural + 2, USER_BUBBLE_MAX_WIDTH - _BUBBLE_H_PADDING)
            )
        self.set_text(content, reasoning, streaming=False)

    def set_text(self, content: str, reasoning: str, *, streaming: bool) -> None:
        """Show ``content`` and (collapsed) ``reasoning``."""
        has_reasoning = bool(reasoning.strip())
        self._thinking_toggle.setVisible(has_reasoning)
        if has_reasoning:
            thinking_now = streaming and not content
            self._thinking_toggle.setText("Thinking…" if thinking_now else "Thoughts")
            self._thinking.setText(reasoning.strip())
        self._content.setText(content)
        self._content.setVisible(bool(content) or not has_reasoning)

    def _toggle_thinking(self, shown: bool) -> None:
        self._thinking.setVisible(shown)
        arrow = Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow
        self._thinking_toggle.setArrowType(arrow)


def _selectable_label(text_format: Qt.TextFormat) -> QLabel:
    label = QLabel()
    label.setTextFormat(text_format)
    label.setWordWrap(True)
    # Links are not clickable on purpose: the app never opens network locations by itself.
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    return label


class ChatArea(QScrollArea):
    """The scrolling list of messages, plus the streaming reply."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty chat area."""
        super().__init__(parent)
        self.setObjectName("chatArea")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer = QWidget()
        outer.setObjectName("chatCanvas")
        outer_layout = QHBoxLayout(outer)
        outer_layout.setContentsMargins(16, 16, 16, 16)
        column = QWidget()
        column.setMaximumWidth(CHAT_MAX_WIDTH)
        self._layout = QVBoxLayout(column)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(14)
        self._layout.addStretch(1)
        outer_layout.addStretch(1)
        outer_layout.addWidget(column, 100)
        outer_layout.addStretch(1)
        self.setWidget(outer)

        self._notice = QLabel()
        self._notice.setObjectName("notice")
        self._notice.setWordWrap(True)
        self._notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._notice.hide()
        self._layout.insertWidget(0, self._notice)

        self._reply: MessageBubble | None = None
        self._latest: tuple[str, str] = ("", "")
        self._redraw = QTimer(self)
        self._redraw.setSingleShot(True)
        self._redraw.setInterval(REPLY_REDRAW_MS)
        self._redraw.timeout.connect(self._flush_reply)
        self._follow = True
        self.verticalScrollBar().rangeChanged.connect(self._on_range_changed)
        self.verticalScrollBar().valueChanged.connect(self._on_scrolled)

    def set_notice(self, message: str | None) -> None:
        """Show a hint in the empty chat, or hide it."""
        self._notice.setText(message or "")
        self._notice.setVisible(message is not None)

    def set_messages(self, messages: Sequence[Message]) -> None:
        """Replace everything with ``messages``."""
        self._reply = None
        self._redraw.stop()
        while self._layout.count() > 2:  # keep the notice (first) and the stretch (last)  # noqa: PLR2004
            item = self._layout.takeAt(1)
            if item is not None and (widget := item.widget()) is not None:
                widget.deleteLater()
        for message in messages:
            self._add_bubble(MessageBubble(message.role, message.content, message.reasoning or ""))
        self._follow = True

    def begin_reply(self) -> None:
        """Add an empty assistant bubble that will fill as the reply streams."""
        self._latest = ("", "")
        self._reply = MessageBubble(Role.ASSISTANT, "", "")
        self._reply.set_text("", "", streaming=True)
        self._add_bubble(self._reply)
        self._follow = True

    def update_reply(self, content: str, reasoning: str) -> None:
        """Queue a redraw of the streaming reply (throttled)."""
        self._latest = (content, reasoning)
        if not self._redraw.isActive():
            self._redraw.start()

    def end_reply(self) -> None:
        """Draw the final state of the streaming reply."""
        self._redraw.stop()
        self._flush_reply(streaming=False)
        self._reply = None

    def _flush_reply(self, *, streaming: bool = True) -> None:
        if self._reply is not None:
            content, reasoning = self._latest
            self._reply.set_text(content, reasoning, streaming=streaming)

    def _add_bubble(self, bubble: MessageBubble) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        if bubble.objectName() == "userBubble":
            bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)
            row_layout.addStretch(1)
            row_layout.addWidget(bubble, 4)
        else:
            row_layout.addWidget(bubble)
        self._layout.insertWidget(self._layout.count() - 1, row)

    def _on_scrolled(self, value: int) -> None:
        bar = self.verticalScrollBar()
        self._follow = value >= bar.maximum() - 40

    def _on_range_changed(self, _minimum: int, maximum: int) -> None:
        if self._follow:
            self.verticalScrollBar().setValue(maximum)


class Sidebar(QFrame):
    """Past conversations (newest first) and the New chat button."""

    new_chat_requested = Signal()
    get_models_requested = Signal()
    conversation_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the sidebar."""
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(SIDEBAR_WIDTH)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)
        self._new_chat = QPushButton("+  New chat")
        self._new_chat.setObjectName("newChat")
        self._new_chat.clicked.connect(self.new_chat_requested.emit)
        layout.addWidget(self._new_chat)
        self._get_models = QPushButton("↓  Get models")
        self._get_models.setObjectName("newChat")
        self._get_models.clicked.connect(self.get_models_requested.emit)
        layout.addWidget(self._get_models)
        heading = QLabel("Chats")
        heading.setObjectName("sidebarHeading")
        layout.addWidget(heading)
        self._list = QListWidget()
        self._list.setObjectName("conversationList")
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setUniformItemSizes(True)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.itemClicked.connect(self._on_clicked)
        layout.addWidget(self._list, 1)

    def set_conversations(
        self, conversations: Sequence[Conversation], current_id: str | None
    ) -> None:
        """Show ``conversations`` and highlight ``current_id``."""
        self._list.blockSignals(True)
        self._list.clear()
        for conversation in conversations:
            item = QListWidgetItem(conversation.title)
            item.setToolTip(conversation.title)
            item.setData(Qt.ItemDataRole.UserRole, conversation.id)
            item.setSizeHint(QSize(0, 34))
            self._list.addItem(item)
            if conversation.id == current_id:
                item.setSelected(True)
                self._list.setCurrentItem(item)
        if current_id is None:
            self._list.clearSelection()
        self._list.blockSignals(False)

    def _on_clicked(self, item: QListWidgetItem) -> None:
        conversation_id: object = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(conversation_id, str):
            self.conversation_selected.emit(conversation_id)


class ModelPicker(QComboBox):
    """Drop-down of local models, each with its status. Choosing one loads it."""

    model_chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty picker."""
        super().__init__(parent)
        self.setObjectName("modelPicker")
        self.setMinimumWidth(320)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.activated.connect(self._on_activated)
        self._has_placeholder = True
        self.addItem("No models yet", None)

    def set_models(self, models: Sequence[ModelInfo], selected_id: str | None) -> None:
        """Show ``models`` with their status; ``selected_id`` is the one the user picked."""
        self.blockSignals(True)
        self.clear()
        self._has_placeholder = selected_id is None
        if self._has_placeholder:
            self.addItem("Choose a model…" if models else "No models found", None)
        items = self.model()
        for model in models:
            status = _STATUS_LABELS[model.status]
            if model.unavailable_reason and model.status == ModelStatus.UNLOADED:
                status = "too large for this computer"
            self.addItem(f"{_STATUS_DOTS[model.status]}  {model.id}  —  {status}", model.id)
            index = self.count() - 1
            if model.unavailable_reason:
                self.setItemData(index, model.unavailable_reason, Qt.ItemDataRole.ToolTipRole)
                if model.id != selected_id and isinstance(items, QStandardItemModel):
                    items.item(index).setEnabled(False)
            if model.id == selected_id:
                self.setCurrentIndex(index)
        if self._has_placeholder:
            self.setCurrentIndex(0)
        self.blockSignals(False)

    def _on_activated(self, index: int) -> None:
        model_id: object = self.itemData(index)
        if isinstance(model_id, str):
            self.model_chosen.emit(model_id)


class Banner(QFrame):
    """A strip under the top bar for server problems, with an action button."""

    action_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create a hidden banner."""
        super().__init__(parent)
        self.setObjectName("banner")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        self._text = QLabel()
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._text, 1)
        self._button = QPushButton()
        self._button.clicked.connect(self.action_clicked.emit)
        layout.addWidget(self._button, 0, Qt.AlignmentFlag.AlignTop)
        self.hide()

    def show_message(self, text: str, action: str | None, *, error: bool) -> None:
        """Show ``text``, with a button labelled ``action`` if given."""
        self.setProperty("error", error)
        self.style().unpolish(self)
        self.style().polish(self)
        self._text.setText(text)
        self._button.setVisible(action is not None)
        self._button.setText(action or "")
        self.show()
