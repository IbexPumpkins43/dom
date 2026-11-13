"""A slider with two handles that picks a range of stops (Qt has no built-in one)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPalette, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

if TYPE_CHECKING:
    from PySide6.QtGui import QKeyEvent, QMouseEvent, QPaintEvent

HANDLE_RADIUS = 7
TRACK_HEIGHT = 4
_MARGIN = HANDLE_RADIUS + 2


class RangeSlider(QWidget):
    """Picks ``low <= high`` among ``stops`` positions by dragging either handle.

    ``range_moved`` fires while dragging (for a live label); ``range_changed`` once the user
    lets go or uses the keyboard (for work such as a new search). Colours come from the palette:
    the selected part uses ``Accent``.
    """

    range_moved = Signal(int, int)
    range_changed = Signal(int, int)

    def __init__(self, stops: int, parent: QWidget | None = None) -> None:
        """A slider with ``stops`` positions (at least 2), covering all of them."""
        super().__init__(parent)
        self._stops = max(2, stops)
        self._low = 0
        self._high = self._stops - 1
        self._active: int | None = None  # 0 = low handle, 1 = high handle, while dragging
        self._focused_handle = 1
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:  # noqa: N802 -- Qt override
        """Wide enough to tell the stops apart."""
        return QSize(220, HANDLE_RADIUS * 2 + 8)

    def values(self) -> tuple[int, int]:
        """The selected stops: (low, high)."""
        return (self._low, self._high)

    def set_values(self, low: int, high: int) -> None:
        """Select stops ``low`` to ``high`` (clamped, without signals)."""
        last = self._stops - 1
        self._low = max(0, min(low, last))
        self._high = max(self._low, min(high, last))
        self.update()

    # ---- geometry ----------------------------------------------------------------------------

    def _x(self, stop: int) -> float:
        span = max(1.0, self.width() - 2 * _MARGIN)
        return _MARGIN + span * stop / (self._stops - 1)

    def _stop_at(self, x: float) -> int:
        span = max(1.0, self.width() - 2 * _MARGIN)
        return max(0, min(self._stops - 1, round((x - _MARGIN) / span * (self._stops - 1))))

    # ---- drawing -----------------------------------------------------------------------------

    def paintEvent(self, e: QPaintEvent) -> None:  # noqa: N802 -- Qt override
        """Track, selected part, stop marks and the two handles."""
        palette = self.palette()
        accent = palette.color(QPalette.ColorRole.Accent)
        track = palette.color(QPalette.ColorRole.Midlight)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        middle = self.height() / 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(
            QRectF(_MARGIN, middle - TRACK_HEIGHT / 2, self.width() - 2 * _MARGIN, TRACK_HEIGHT),
            2,
            2,
        )
        low, high = self._x(self._low), self._x(self._high)
        painter.setBrush(accent)
        painter.drawRoundedRect(
            QRectF(low, middle - TRACK_HEIGHT / 2, high - low, TRACK_HEIGHT), 2, 2
        )
        painter.setBrush(palette.color(QPalette.ColorRole.Mid))
        for stop in range(self._stops):
            if not self._low <= stop <= self._high:
                painter.drawEllipse(QPointF(self._x(stop), middle), 1.5, 1.5)
        for index, x in enumerate((low, high)):
            painter.setBrush(accent)
            ring = QColor(palette.color(QPalette.ColorRole.BrightText))
            focused = self.hasFocus() and index == self._focused_handle
            painter.setPen(QPen(ring, 2) if focused else Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x, middle), HANDLE_RADIUS, HANDLE_RADIUS)
        painter.end()

    # ---- mouse and keyboard ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 -- Qt override
        """Grab the nearer handle (the one on the side clicked when they overlap)."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        x = event.position().x()
        to_low, to_high = abs(x - self._x(self._low)), abs(x - self._x(self._high))
        if self._low == self._high:
            self._active = 0 if x < self._x(self._low) else 1
        else:
            self._active = 0 if to_low < to_high else 1
        self._focused_handle = self._active
        self._move_active(x)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 -- Qt override
        """Drag the grabbed handle from stop to stop."""
        if self._active is not None:
            self._move_active(event.position().x())
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 -- Qt override
        """Let go: report the final range."""
        if self._active is not None and event.button() == Qt.MouseButton.LeftButton:
            self._active = None
            self.range_changed.emit(self._low, self._high)
            event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 -- Qt override
        """Left/Right move the focused handle; Tab-like Up/Down switch handles."""
        key = event.key()
        if key in (Qt.Key.Key_Up.value, Qt.Key.Key_Down.value):
            self._focused_handle = 1 - self._focused_handle
            self.update()
            return
        step = {Qt.Key.Key_Left.value: -1, Qt.Key.Key_Right.value: 1}.get(key)
        if step is None:
            super().keyPressEvent(event)
            return
        low, high = self._low, self._high
        if self._focused_handle == 0:
            low = max(0, min(high, low + step))
        else:
            high = min(self._stops - 1, max(low, high + step))
        if (low, high) != (self._low, self._high):
            self._low, self._high = low, high
            self.update()
            self.range_moved.emit(low, high)
            self.range_changed.emit(low, high)

    def _move_active(self, x: float) -> None:
        stop = self._stop_at(x)
        if self._active == 0:
            stop = min(stop, self._high)
            changed = stop != self._low
            self._low = stop
        else:
            stop = max(stop, self._low)
            changed = stop != self._high
            self._high = stop
        if changed:
            self.update()
            self.range_moved.emit(self._low, self._high)
