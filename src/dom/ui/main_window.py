"""The main window: implements ``ChatView`` with Qt widgets and forwards input to the controller."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QSize, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from dom.ui.controller import spawn
from dom.ui.dialogs import Question, ask
from dom.ui.icons import ICON_SIZE, Icon, make_icon
from dom.ui.library import Filters
from dom.ui.models_screen import ModelsScreen
from dom.ui.profile_widgets import ProfileButton, ProfileDialog
from dom.ui.protocols import ServerState
from dom.ui.theme import BRASA, apply_theme, theme_for
from dom.ui.titlebar import TitleBar, WindowFrame
from dom.ui.widgets import Banner, ChatArea, ChatInput, ElidedLabel, ModelPicker, Sidebar

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from PySide6.QtGui import QCloseEvent

    from dom.domain import Conversation, Message, ModelInfo, Profile, ProfileSettings
    from dom.hardware import SystemUsage
    from dom.ui.controller import ChatController
    from dom.ui.library import LibraryController
    from dom.ui.profiles import ProfileController
    from dom.ui.theme import Base16

ERROR_DISPLAY_MS = 10_000
USAGE_REFRESH_MS = 200
COMPOSER_MAX_WIDTH = 900
SIDEBAR_SHORTCUT = "Ctrl+B"

_SERVER_LABELS: dict[ServerState, str] = {
    ServerState.STARTING: "Starting the AI engine…",
    ServerState.READY: "AI engine running",
    ServerState.FAILED: "AI engine unavailable",
    ServerState.ENGINE_MISSING: "AI engine not installed",
    ServerState.INSTALLING: "Installing the AI engine…",
    ServerState.CRASHED: "AI engine stopped",
    ServerState.STOPPED: "AI engine stopped",
}


class MainWindow(QMainWindow):
    """The chat window (implements ``ChatView`` and ``ProfileView``)."""

    def __init__(
        self,
        on_closed: Callable[[], None],
        usage_reader: Callable[[], SystemUsage] | None = None,
    ) -> None:
        """Build the window; call ``attach`` before showing it.

        ``usage_reader`` (if given) is sampled every few seconds, off the UI thread, for the
        CPU/memory panel at the bottom of the sidebar.
        """
        super().__init__()
        self._usage_reader = usage_reader
        self._usage_busy = False
        self._on_closed = on_closed
        self._controller: ChatController | None = None
        self._library: LibraryController | None = None
        self._profiles: ProfileController | None = None
        self._offered_library = False
        self._chat_title = "New chat"
        self._tasks: set[asyncio.Task[None]] = set()
        self._closing = False
        self._closed = False
        self._server_state = ServerState.STOPPED
        self._streaming = False
        self.setWindowTitle("dom")
        # dom draws its own title bar in the app's colours (ui/titlebar.py). The min/max hints
        # keep the window minimisable from the taskbar and snappable on Windows.
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.resize(1180, 780)
        self.setMinimumSize(760, 480)

        self._sidebar = Sidebar()
        self._picker = ModelPicker()
        self._server_label = QLabel()
        self._server_label.setObjectName("serverLabel")
        self._details_label = ElidedLabel()
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
        self._composer = QWidget()
        self._input.focus_changed.connect(self._on_input_focus)
        self._send = QPushButton("Send")
        self._send.setObjectName("sendButton")
        self._send.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build_layout()
        self._usage_timer = QTimer(self)
        self._usage_timer.setInterval(USAGE_REFRESH_MS)
        self._usage_timer.timeout.connect(self._sample_usage)
        self._sidebar.usage.setVisible(usage_reader is not None)
        if usage_reader is not None:
            self._usage_timer.start()

    def _build_layout(self) -> None:
        composer = self._composer
        composer.setObjectName("composer")
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(10, 8, 10, 8)
        composer_layout.addWidget(self._input, 1)
        composer_layout.addWidget(self._send, 0, Qt.AlignmentFlag.AlignBottom)

        # Under the message box: which model, what it runs on, and the engine's state.
        footer = QWidget()
        footer.setObjectName("footer")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(4, 8, 4, 0)
        footer_layout.addWidget(self._picker)
        footer_layout.addSpacing(12)
        footer_layout.addWidget(self._details_label, 1)
        footer_layout.addSpacing(12)
        footer_layout.addWidget(self._server_label)

        bottom = QWidget()
        bottom.setMaximumWidth(COMPOSER_MAX_WIDTH)
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        bottom_layout.addWidget(self._error)
        bottom_layout.addWidget(composer)
        bottom_layout.addWidget(footer)
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(16, 0, 16, 14)
        bottom_row.addStretch(1)
        bottom_row.addWidget(bottom, 100)
        bottom_row.addStretch(1)

        main = QWidget()
        # The whole chat column (messages, message box, model picker) shares the chat's background.
        main.setObjectName("chatPane")
        main.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self._banner)
        main_layout.addWidget(self._chat, 1)
        main_layout.addLayout(bottom_row)

        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._profile_button = ProfileButton()
        self._profile_button.hide()  # until the profiles are known
        self._sidebar.set_profile_widget(self._profile_button)
        layout.addWidget(self._sidebar)
        layout.addWidget(self._build_pages(main), 1)
        self._frame = WindowFrame(self._build_title_bar(), body)
        self.setCentralWidget(self._frame)

    def _build_pages(self, chat: QWidget) -> QStackedWidget:
        """The chat, or the models screen in its place."""
        self._models_screen = ModelsScreen()
        self._models_screen.back_requested.connect(self.show_chat)
        self._pages = QStackedWidget()
        self._pages.addWidget(chat)
        self._pages.addWidget(self._models_screen)
        return self._pages

    def _build_title_bar(self) -> TitleBar:
        self._theme = BRASA
        # In place of the "dom" name: collapse / expand the sidebar (also Ctrl+B).
        self._sidebar_toggle = QPushButton()
        self._sidebar_toggle.setObjectName("sidebarToggle")
        self._sidebar_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sidebar_toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._sidebar_toggle.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self._sidebar_toggle.clicked.connect(self.toggle_sidebar)
        QShortcut(QKeySequence(SIDEBAR_SHORTCUT), self).activated.connect(self.toggle_sidebar)
        self._title_bar = TitleBar(self._theme, leading=self._sidebar_toggle)
        self._title_bar.set_title("New chat")
        self._set_icon_colour(self._theme.base05)
        self._update_sidebar_toggle()
        return self._title_bar

    def toggle_sidebar(self) -> None:
        """Switch the sidebar between full and icons-only."""
        compact = not self._sidebar.is_compact()
        self._sidebar.set_compact(compact)
        self._profile_button.set_compact(compact)
        self._update_sidebar_toggle()

    def sidebar_toggle(self) -> QPushButton:
        """The collapse / expand button in the title bar."""
        return self._sidebar_toggle

    def _update_sidebar_toggle(self) -> None:
        action = "Expand" if self._sidebar.is_compact() else "Collapse"
        self._sidebar_toggle.setToolTip(f"{action} the sidebar ({SIDEBAR_SHORTCUT})")

    def _set_icon_colour(self, colour: str) -> None:
        self._sidebar_toggle.setIcon(make_icon(Icon.SIDEBAR, colour))
        self._sidebar.set_icon_colour(colour)

    def attach(
        self,
        controller: ChatController,
        library: LibraryController,
        profiles: ProfileController,
    ) -> None:
        """Connect UI events to the chat, library and profile controllers."""
        self._controller = controller
        self._library = library
        self._profiles = profiles
        button = self._profile_button
        button.switch_requested.connect(self._switch_profile)
        button.add_requested.connect(self._add_profile)
        button.edit_requested.connect(self._edit_profile)
        button.delete_requested.connect(self._confirm_delete_profile)
        self._sidebar.get_models_requested.connect(self.show_models_screen)
        self._connect_models_screen(library)
        self._input.submitted.connect(self._submit)
        self._send.clicked.connect(self._send_or_stop)
        self._picker.model_chosen.connect(self._on_model_chosen)
        self._sidebar.new_chat_requested.connect(self._new_chat)
        self._sidebar.conversation_selected.connect(self._on_conversation_selected)
        self._sidebar.delete_requested.connect(self._confirm_delete)
        self._banner.action_clicked.connect(self._on_banner_action)
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
        elif state == ServerState.ENGINE_MISSING:
            text = message or "The AI engine (llama.cpp) is not installed yet."
            self._banner.show_message(text, "Download AI engine", error=True)
        elif state == ServerState.INSTALLING:
            self._banner.show_message(message or "Installing the AI engine…", None, error=False)
        self._picker.setEnabled(state == ServerState.READY and not self._streaming)

    def show_models(self, models: Sequence[ModelInfo], selected_id: str | None) -> None:
        """Fill the model picker (and offer downloads the first time there are none)."""
        self._picker.set_models(models, selected_id)
        if not models and not self._offered_library and self._server_state == ServerState.READY:
            self._offered_library = True
            self.show_models_screen()

    def show_conversations(
        self, conversations: Sequence[Conversation], current_id: str | None
    ) -> None:
        """Fill the sidebar, and show the open chat's title in the title bar."""
        self._sidebar.set_conversations(conversations, current_id)
        current = next((c for c in conversations if c.id == current_id), None)
        self._chat_title = current.title if current is not None else "New chat"
        if not self.showing_models():
            self._title_bar.set_title(self._chat_title)

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
        self._details_label.set_full_text(text or "")

    def show_hardware(self, summary: str | None) -> None:
        """Put the detected hardware in tooltips."""
        tip = f"This computer: {summary}" if summary else ""
        self._server_label.setToolTip(tip)
        self._details_label.setToolTip(tip)

    # ---- ProfileView ---------------------------------------------------------------------

    def show_profiles(self, profiles: Sequence[Profile], current: Profile) -> None:
        """Show the current profile in the sidebar and its menu, in the profile's colours."""
        self._profile_button.set_profiles(profiles, current)
        self._profile_button.show()
        self._apply_look(theme_for(current.settings))

    def theme(self) -> Base16:
        """The colours in use (Brasa, or Brasa tinted in the profile's colour)."""
        return self._theme

    def _apply_look(self, theme: Base16) -> None:
        if theme == self._theme:
            return  # re-applying the stylesheet re-polishes every widget: skip if unchanged
        self._theme = theme
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, theme)
        self._chat.set_logo_colour(theme.base09)
        self._set_icon_colour(theme.base05)

    def profile_button(self) -> ProfileButton:
        """The profile switcher at the bottom of the sidebar."""
        return self._profile_button

    # ---- events --------------------------------------------------------------------------

    def _switch_profile(self, profile_id: str) -> None:
        if self._profiles is not None:
            self._run(self._profiles.switch(profile_id))

    def _add_profile(self) -> None:
        profiles = self._profiles
        if profiles is None:
            return

        def save(settings: ProfileSettings) -> None:
            self._run(profiles.create(settings))

        self._open_profile_dialog(profiles.new_profile_settings(), save, adding=True)

    def _edit_profile(self) -> None:
        profiles = self._profiles
        if profiles is None:
            return
        profile_id = profiles.current.id

        def save(settings: ProfileSettings) -> None:
            self._run(profiles.update(profile_id, settings))

        self._open_profile_dialog(profiles.current.settings, save, adding=False)

    def _open_profile_dialog(
        self, initial: ProfileSettings, on_save: Callable[[ProfileSettings], None], *, adding: bool
    ) -> None:
        dialog = ProfileDialog(self, initial, on_save, adding=adding)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.open()  # not exec(): see dialogs.ask

    def _confirm_delete_profile(self) -> None:
        profiles = self._profiles
        if profiles is None:
            return
        profile = profiles.current
        question = Question(
            "Delete profile",
            f"Delete the profile “{profile.name}” and all of its chats?\n\n"
            "This cannot be undone. Downloaded models are kept.",
            "Delete profile",
            danger=True,
        )
        ask(self, question, lambda: self._run(profiles.delete(profile.id)))

    def _submit(self) -> None:
        controller = self._controller
        if controller is None or not self._input.isEnabled():
            return
        text = self._input.toPlainText()
        if not text.strip():
            return
        self._input.clear()
        self._run(controller.send(text))

    def _sample_usage(self) -> None:
        if self._usage_reader is None or self._usage_busy or self._closing:
            return
        self._usage_busy = True
        self._run(self._refresh_usage(self._usage_reader))

    async def _refresh_usage(self, reader: Callable[[], SystemUsage]) -> None:
        try:
            usage = await asyncio.to_thread(reader)
        finally:
            self._usage_busy = False
        self._sidebar.usage.set_usage(usage)

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 -- Qt override
        """Keep the title bar and resize border in step with maximise/restore."""
        if event.type() == QEvent.Type.WindowStateChange:
            maximized = self.isMaximized()
            self._title_bar.update_state(maximized)
            self._frame.set_maximized(maximized)
        super().changeEvent(event)

    def _on_input_focus(self, focused: bool) -> None:
        self._composer.setProperty("focused", focused)
        self._composer.style().unpolish(self._composer)
        self._composer.style().polish(self._composer)

    def _confirm_delete(self, conversation_id: str, title: str) -> None:
        controller = self._controller
        if controller is None:
            return
        question = Question(
            "Delete chat",
            f"Delete “{title or 'this chat'}”?\n\nThis cannot be undone.",
            "Delete",
            danger=True,
        )
        ask(self, question, lambda: self._run(controller.delete_conversation(conversation_id)))

    def _on_banner_action(self) -> None:
        if self._controller is None:
            return
        if self._server_state == ServerState.ENGINE_MISSING:
            self._run(self._controller.install_engine())
        else:
            self._run(self._controller.restart_server())

    # ---- models screen -------------------------------------------------------------------

    def _connect_models_screen(self, library: LibraryController) -> None:
        screen = self._models_screen
        library.attach(screen)
        screen.search_requested.connect(self._search_models)
        screen.load_more_requested.connect(lambda: self._run(library.load_more()))
        screen.download_requested.connect(self._download_model)
        screen.cancel_requested.connect(self._cancel_download)
        screen.version_chosen.connect(self._choose_version)
        screen.filters_changed.connect(self._set_filters)
        screen.delete_requested.connect(self._confirm_delete_model)

    def _search_models(self, text: str) -> None:
        if self._library is not None:
            self._run(self._library.search(text))

    def _download_model(self, key: str) -> None:
        if self._library is not None:
            self._run(self._library.download(key))

    def _cancel_download(self, key: str) -> None:
        if self._library is not None:
            self._run(self._library.cancel(key))

    def _set_filters(self, filters: object) -> None:
        if self._library is not None and isinstance(filters, Filters):
            self._run(self._library.set_filters(filters))

    def _choose_version(self, key: str, path: str) -> None:
        if self._library is not None:
            self._run(self._library.choose_version(key, path))

    def models_screen(self) -> ModelsScreen:
        """The models screen (shown in place of the chat)."""
        return self._models_screen

    def showing_models(self) -> bool:
        """True while the models screen is shown instead of the chat."""
        return self._pages.currentWidget() is self._models_screen

    def show_models_screen(self) -> None:
        """Show the models screen in place of the chat."""
        self._pages.setCurrentWidget(self._models_screen)
        self._title_bar.set_title("Models")
        self._models_screen.search.setFocus()
        if self._library is not None:
            self._run(self._library.open())

    def show_chat(self) -> None:
        """Go back from the models screen to the chat."""
        self._pages.setCurrentIndex(0)
        self._title_bar.set_title(self._chat_title)

    def _new_chat(self) -> None:
        self.show_chat()
        if self._controller is not None:
            self._run(self._controller.new_chat())

    def _confirm_delete_model(self, filename: str, title: str) -> None:
        library = self._library
        if library is None:
            return
        question = Question(
            "Delete model",
            f"Delete “{title}” from this computer?\n\n{filename} will be removed. You can "
            "download it again later.",
            "Delete",
            danger=True,
        )
        ask(self, question, lambda: self._run(library.delete(filename)))

    def _on_model_chosen(self, model_id: str) -> None:
        if self._controller is not None:
            self._run(self._controller.select_model(model_id))

    def _on_conversation_selected(self, conversation_id: str) -> None:
        self.show_chat()
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
            if self._library is not None:
                await self._library.shutdown()
            await controller.shutdown()
        finally:
            self._closed = True
            self.close()

    def _finish_close(self) -> None:
        self._on_closed()

    def _run(self, coro: Coroutine[object, object, None]) -> None:
        spawn(coro, self._tasks)
