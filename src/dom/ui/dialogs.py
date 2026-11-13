"""Child windows with dom's own title bar and frame (instead of the system's)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from dom.ui.theme import BRASA
from dom.ui.titlebar import TitleBar, WindowFrame

if TYPE_CHECKING:
    from dom.ui.theme import Base16

_MESSAGE_WIDTH = 420


class FramedDialog(QDialog):
    """A dialog drawn with dom's title bar (close button only) and border.

    Put the dialog's widgets in ``content()``'s layout.
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        resizable: bool = False,
        theme: Base16 = BRASA,
    ) -> None:
        """Create an empty framed dialog titled ``title``."""
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.title_bar = TitleBar(theme, maximizable=False)
        self.title_bar.set_title(title)
        self._content = QWidget()
        self._content.setObjectName("dialogContent")
        self._content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(WindowFrame(self.title_bar, self._content, resizable=resizable))

    def content(self) -> QWidget:
        """The area below the title bar."""
        return self._content


@dataclass(frozen=True, slots=True)
class _Buttons:
    """The buttons of a message dialog: an optional confirm button and a cancel/OK button."""

    cancel: str
    confirm: str | None = None
    danger: bool = False


class _MessageDialog(FramedDialog):
    """A short message with one or two buttons."""

    def __init__(self, parent: QWidget | None, title: str, text: str, buttons: _Buttons) -> None:
        super().__init__(title, parent)
        layout = QVBoxLayout(self.content())
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(16)
        message = QLabel(text)
        message.setObjectName("dialogMessage")
        message.setWordWrap(True)
        message.setMinimumWidth(_MESSAGE_WIDTH)
        message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(message)
        row = QHBoxLayout()
        row.addStretch(1)
        self.cancel_button = QPushButton(buttons.cancel)
        self.cancel_button.clicked.connect(self.reject)
        row.addWidget(self.cancel_button)
        self.confirm_button: QPushButton | None = None
        if buttons.confirm is not None:
            self.confirm_button = QPushButton(buttons.confirm)
            self.confirm_button.setObjectName("dangerButton" if buttons.danger else "primaryButton")
            self.confirm_button.clicked.connect(self.accept)
            row.addWidget(self.confirm_button)
        layout.addLayout(row)
        # The safe choice is the default: Enter cancels a destructive action.
        self.cancel_button.setDefault(True)
        self.cancel_button.setFocus()


def confirm(
    parent: QWidget | None, title: str, text: str, confirm_label: str, *, danger: bool = False
) -> bool:
    """Ask a yes/no question in a framed dialog; True if the user confirmed."""
    dialog = _MessageDialog(parent, title, text, _Buttons("Cancel", confirm_label, danger))
    return dialog.exec() == QDialog.DialogCode.Accepted


def show_message(parent: QWidget | None, title: str, text: str) -> None:
    """Show a message in a framed dialog and wait until it is closed."""
    _MessageDialog(parent, title, text, _Buttons("OK")).exec()
