"""Profile widgets: the avatar button in the title bar (with its menu) and the profile form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dom.config.settings import GenerationSettings
from dom.domain import PROFILE_NAME_MAX_CHARS, ProfileColour, ProfileSettings
from dom.ui.dialogs import FramedDialog
from dom.ui.theme import BRASA, profile_colour

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from PySide6.QtGui import QPaintEvent

    from dom.domain import Profile
    from dom.ui.theme import Base16

AVATAR_SIZE = 20
MENU_AVATAR_SIZE = 18
SWATCH_SIZE = 26
BUTTON_NAME_MAX_CHARS = 18
_ICON_PIXEL_RATIO = 2.0  # draw icons at 2x so they stay sharp on high-DPI screens


def avatar_pixmap(
    settings: ProfileSettings,
    size: int,
    *,
    theme: Base16 = BRASA,
    device_pixel_ratio: float = _ICON_PIXEL_RATIO,
) -> QPixmap:
    """A round avatar: the profile's initial on its colour."""
    pixels = round(size * device_pixel_ratio)
    pixmap = QPixmap(pixels, pixels)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(profile_colour(theme, settings.colour)))
    painter.drawEllipse(0, 0, pixels, pixels)
    font = QFont()
    font.setBold(True)
    font.setPixelSize(max(1, round(pixels * 0.55)))
    painter.setFont(font)
    painter.setPen(QColor(theme.base00))
    painter.drawText(QRectF(0, 0, pixels, pixels), Qt.AlignmentFlag.AlignCenter, settings.initial)
    painter.end()
    pixmap.setDevicePixelRatio(device_pixel_ratio)
    return pixmap


def _short(name: str, limit: int = BUTTON_NAME_MAX_CHARS) -> str:
    return name if len(name) <= limit else name[: limit - 1].rstrip() + "…"


class ProfileButton(QToolButton):
    """The current profile's avatar and name; click for the profile menu (like a browser's)."""

    switch_requested = Signal(str)
    add_requested = Signal()
    edit_requested = Signal()
    delete_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the button with an empty menu; ``set_profiles`` fills it."""
        super().__init__(parent)
        self.setObjectName("profileButton")
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setIconSize(QSize(AVATAR_SIZE, AVATAR_SIZE))
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self.setToolTip("Profiles")

    def menu_actions(self) -> list[str]:
        """The menu's item texts, in order (separators as ``---``); for tests."""
        return [a.text() or "---" for a in self._menu.actions()]

    def trigger(self, text: str) -> None:
        """Trigger the menu item with this text (for tests)."""
        for action in self._menu.actions():
            if action.text() == text:
                action.trigger()
                return
        raise KeyError(text)

    def set_profiles(self, profiles: Sequence[Profile], current: Profile) -> None:
        """Show ``current`` on the button and every profile in the menu."""
        self.setIcon(QIcon(avatar_pixmap(current.settings, AVATAR_SIZE)))
        self.setText(_short(current.name))
        self.setToolTip(f"Profile: {current.name}")
        menu = self._menu
        menu.clear()
        for profile in profiles:
            action = menu.addAction(
                QIcon(avatar_pixmap(profile.settings, MENU_AVATAR_SIZE)), profile.name
            )
            if profile.id == current.id:
                font = action.font()
                font.setBold(True)
                action.setFont(font)
            profile_id = profile.id
            action.triggered.connect(
                lambda _checked=False, pid=profile_id: self.switch_requested.emit(pid)
            )
        menu.addSeparator()
        menu.addAction("Edit this profile…").triggered.connect(self.edit_requested.emit)
        menu.addAction("Add profile…").triggered.connect(self.add_requested.emit)
        delete = menu.addAction("Delete this profile…")
        delete.setEnabled(len(profiles) > 1)
        delete.triggered.connect(self.delete_requested.emit)


class ColourSwatch(QAbstractButton):
    """A round, checkable colour choice."""

    def __init__(self, colour: ProfileColour, theme: Base16, parent: QWidget | None = None) -> None:
        """A swatch for ``colour`` in ``theme``."""
        super().__init__(parent)
        self.colour = colour
        self._theme = theme
        self.setCheckable(True)
        self.setFixedSize(SWATCH_SIZE, SWATCH_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(colour.value.capitalize())
        self.setAccessibleName(colour.value)

    def paintEvent(self, e: QPaintEvent) -> None:  # noqa: N802 -- Qt override
        """A filled circle, ringed when chosen."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        ring = 2.0
        inner = QRectF(self.rect()).adjusted(ring * 2, ring * 2, -ring * 2, -ring * 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(profile_colour(self._theme, self.colour)))
        painter.drawEllipse(inner)
        if self.isChecked() or self.hasFocus():
            pen = QPen(QColor(self._theme.base07 if self.isChecked() else self._theme.base03))
            pen.setWidthF(ring)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(self.rect()).adjusted(1, 1, -1, -1))
        painter.end()


@dataclass(frozen=True, slots=True)
class _Choice[T]:
    label: str
    value: T


TEMPERATURE_CHOICES: tuple[_Choice[float | None], ...] = (
    _Choice("The model's default", None),
    _Choice("Precise and predictable", 0.2),
    _Choice("Balanced", 0.7),
    _Choice("Creative and varied", 1.0),
)
# Tokens are word pieces: about three quarters of an English word each.
REPLY_LENGTH_CHOICES: tuple[_Choice[int | None], ...] = (
    _Choice("Short (about 750 words)", 1024),
    _Choice("Medium (about 3,000 words)", 4096),
    _Choice("Long (about 6,000 words)", 8192),
    _Choice("No limit", None),
)


def _fill_combo[T](
    combo: QComboBox, choices: tuple[_Choice[T], ...], current: T, custom: str
) -> list[_Choice[T]]:
    """Fill ``combo`` and select ``current`` (adding a "custom" entry if it is not a choice)."""
    options = list(choices)
    if all(c.value != current for c in options):
        options.append(_Choice(custom.format(current), current))
    for option in options:
        combo.addItem(option.label)
    combo.setCurrentIndex(next(i for i, c in enumerate(options) if c.value == current))
    return options


class ProfileDialog(FramedDialog):
    """Add or edit a profile: name, colour, instructions for the AI and how it replies."""

    def __init__(
        self,
        parent: QWidget | None,
        initial: ProfileSettings,
        on_save: Callable[[ProfileSettings], None],
        *,
        adding: bool,
    ) -> None:
        """A form filled with ``initial``; ``on_save`` gets the result when the user confirms."""
        super().__init__("Add profile" if adding else "Edit profile", parent)
        self.setObjectName("profileDialog")
        self._on_save = on_save
        layout = QVBoxLayout(self.content())
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(6)
        self._build_identity(layout, initial)
        self._build_replies(layout, initial)
        self._build_buttons(layout, "Add profile" if adding else "Save")
        self.setMinimumWidth(460)
        self.name_edit.textChanged.connect(self._validate)
        self.accepted.connect(lambda: self._on_save(self.settings()))
        self._validate()
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def _build_identity(self, layout: QVBoxLayout, initial: ProfileSettings) -> None:
        self.name_edit = QLineEdit(initial.name)
        self.name_edit.setMaxLength(PROFILE_NAME_MAX_CHARS)
        self.name_edit.setPlaceholderText("For example: Work, Homework, Sam")
        self._field(layout, "Name", self.name_edit)

        swatches = QHBoxLayout()
        swatches.setSpacing(4)
        self._colours = QButtonGroup(self)
        self._colours.setExclusive(True)
        for colour in ProfileColour:
            swatch = ColourSwatch(colour, BRASA)
            swatch.setChecked(colour == initial.colour)
            self._colours.addButton(swatch)
            swatches.addWidget(swatch)
        swatches.addStretch(1)
        self._field(layout, "Colour", swatches)
        layout.addSpacing(4)
        self.tint_check = QCheckBox("Tint dom in this colour")
        self.tint_check.setChecked(initial.tint)
        layout.addWidget(self.tint_check)

    def _build_replies(self, layout: QVBoxLayout, initial: ProfileSettings) -> None:
        self.prompt_edit = QPlainTextEdit(initial.system_prompt)
        self.prompt_edit.setPlaceholderText(
            "Optional. For example: Answer briefly, in British English."
        )
        self.prompt_edit.setTabChangesFocus(True)
        self.prompt_edit.setFixedHeight(84)
        self._field(
            layout,
            "Instructions for the AI",
            self.prompt_edit,
            "Given to the AI at the start of every chat in this profile.",
        )

        generation = initial.generation
        self.style_combo = QComboBox()
        self._temperatures = _fill_combo(
            self.style_combo, TEMPERATURE_CHOICES, generation.temperature, "Custom ({})"
        )
        self._field(layout, "Style of answers", self.style_combo)

        self.length_combo = QComboBox()
        self._lengths = _fill_combo(
            self.length_combo, REPLY_LENGTH_CHOICES, generation.max_reply_tokens, "{} tokens"
        )
        self._field(
            layout,
            "Longest reply",
            self.length_combo,
            "Stops replies that run on. Thinking counts towards it.",
        )

        layout.addSpacing(6)
        self.thinking_check = QCheckBox("Think before answering")
        self.thinking_check.setChecked(generation.thinking)
        layout.addWidget(self.thinking_check)
        hint = QLabel("Slower, but usually better answers. Turn off for quick replies.")
        hint.setObjectName("fieldHint")
        layout.addWidget(hint)

    def _build_buttons(self, layout: QVBoxLayout, confirm_label: str) -> None:
        layout.addSpacing(14)
        row = QHBoxLayout()
        row.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.save_button = QPushButton(confirm_label)
        self.save_button.setObjectName("primaryButton")
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.accept)
        row.addWidget(self.cancel_button)
        row.addWidget(self.save_button)
        layout.addLayout(row)

    @staticmethod
    def _field(
        layout: QVBoxLayout, label: str, field: QWidget | QHBoxLayout, hint: str | None = None
    ) -> None:
        layout.addSpacing(8)
        heading = QLabel(label)
        heading.setObjectName("fieldLabel")
        layout.addWidget(heading)
        if isinstance(field, QWidget):
            heading.setBuddy(field)
            layout.addWidget(field)
        else:
            layout.addLayout(field)
        if hint is not None:
            note = QLabel(hint)
            note.setObjectName("fieldHint")
            note.setWordWrap(True)
            layout.addWidget(note)

    def _validate(self) -> None:
        self.save_button.setEnabled(bool(self.name_edit.text().strip()))

    def settings(self) -> ProfileSettings:
        """The settings as currently entered."""
        checked = self._colours.checkedButton()
        colour = checked.colour if isinstance(checked, ColourSwatch) else ProfileColour.ORANGE
        return ProfileSettings(
            name=self.name_edit.text().strip(),
            colour=colour,
            system_prompt=self.prompt_edit.toPlainText().strip(),
            generation=GenerationSettings(
                temperature=self._temperatures[self.style_combo.currentIndex()].value,
                max_reply_tokens=self._lengths[self.length_combo.currentIndex()].value,
                thinking=self.thinking_check.isChecked(),
            ),
            tint=self.tint_check.isChecked(),
        )
