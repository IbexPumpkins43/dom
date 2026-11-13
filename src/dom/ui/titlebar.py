"""dom's own window frame: a themed title bar plus resize edges, replacing the system's.

Moving and resizing are handed to the operating system (``QWindow.startSystemMove`` /
``startSystemResize``), so window snapping, multi-monitor moves and Wayland all keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from dom.ui.widgets import ElidedLabel

if TYPE_CHECKING:
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QEnterEvent, QPaintEvent

    from dom.ui.theme import Base16

TITLE_BAR_HEIGHT = 36
RESIZE_MARGIN = 5
LEADING_MARGIN = 10
_BUTTON_SIZE = QSize(46, TITLE_BAR_HEIGHT)
_GLYPH = 10  # size of the drawn button symbols, in pixels


class CaptionButton(QAbstractButton):
    """Minimise / maximise / restore / close, drawn with lines so they look the same everywhere.

    Colours come from the application palette (so they follow a profile's tint), except the
    close button's red hover, which is ``theme.base08``.
    """

    def __init__(self, kind: str, theme: Base16, parent: QWidget | None = None) -> None:
        """``kind`` is ``minimize``, ``maximize``, ``restore`` or ``close``."""
        super().__init__(parent)
        self.kind = kind
        self._theme = theme
        self._hovered = False
        self.setFixedSize(_BUTTON_SIZE)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip({"minimize": "Minimise", "close": "Close"}.get(kind, "Maximise"))

    def enterEvent(self, event: QEnterEvent) -> None:  # noqa: N802 -- Qt override
        """Highlight on hover."""
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:  # noqa: N802 -- Qt override
        """Stop highlighting."""
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, e: QPaintEvent) -> None:  # noqa: N802 -- Qt override
        """Draw the background and the symbol."""
        palette = self.palette()
        closing = self._hovered and self.kind == "close"
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._hovered:
            hover = palette.color(QPalette.ColorRole.Midlight)  # base02
            painter.fillRect(self.rect(), QColor(self._theme.base08) if closing else hover)
        colour = palette.color(
            QPalette.ColorRole.Window if closing else QPalette.ColorRole.WindowText
        )
        pen = QPen(colour)
        pen.setWidthF(1.2)
        painter.setPen(pen)
        centre = QPointF(self.width() / 2, self.height() / 2)
        half = _GLYPH / 2
        box = QRectF(centre.x() - half, centre.y() - half, _GLYPH, _GLYPH)
        if self.kind == "minimize":
            painter.drawLine(QPointF(box.left(), centre.y()), QPointF(box.right(), centre.y()))
        elif self.kind == "maximize":
            painter.drawRect(box)
        elif self.kind == "restore":
            painter.drawRect(box.adjusted(0, 2, -2, 0))
            painter.drawPolyline(
                [
                    QPointF(box.left() + 2, box.top() + 2),
                    QPointF(box.left() + 2, box.top()),
                    QPointF(box.right(), box.top()),
                    QPointF(box.right(), box.bottom() - 2),
                    QPointF(box.right() - 2, box.bottom() - 2),
                ]
            )
        else:
            painter.drawLine(box.topLeft(), box.bottomRight())
            painter.drawLine(box.topRight(), box.bottomLeft())
        painter.end()


class TitleBar(QWidget):
    """The top bar: app name, current chat title and the window buttons. Drag it to move."""

    def __init__(
        self,
        theme: Base16,
        parent: QWidget | None = None,
        *,
        maximizable: bool = True,
        leading: QWidget | None = None,
    ) -> None:
        """Create the bar (it acts on the top-level window it ends up in).

        ``maximizable=False`` (dialogs) shows only the close button. ``leading`` (e.g. the main
        window's sidebar button) takes the place of the "dom" name on the left.
        """
        super().__init__(parent)
        self._maximizable = maximizable
        self.setObjectName("titleBar")
        self.setFixedHeight(TITLE_BAR_HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setSpacing(0)
        if leading is None:
            layout.setContentsMargins(14, 0, 0, 0)
            name = QLabel("dom")
            name.setObjectName("titleBarName")
            layout.addWidget(name)
        else:
            # Lines up with the sidebar's icon buttons, which sit 10 px from the edge.
            layout.setContentsMargins(LEADING_MARGIN, 0, 0, 0)
            layout.addWidget(leading)
        self._title = ElidedLabel()
        self._title.setObjectName("titleBarTitle")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title, 1)
        self.minimize_button = CaptionButton("minimize", theme)
        self.maximize_button = CaptionButton("maximize", theme)
        self.close_button = CaptionButton("close", theme)
        for button in (self.minimize_button, self.maximize_button, self.close_button):
            layout.addWidget(button)
        self.minimize_button.setVisible(maximizable)
        self.maximize_button.setVisible(maximizable)
        self.minimize_button.clicked.connect(lambda: self.window().showMinimized())
        self.maximize_button.clicked.connect(self.toggle_maximized)
        self.close_button.clicked.connect(lambda: self.window().close())

    def set_title(self, text: str) -> None:
        """The text in the middle of the bar (e.g. the open chat's title)."""
        self._title.set_full_text(text)

    def title(self) -> str:
        """The full (not elided) title."""
        return self._title.full_text()

    def toggle_maximized(self) -> None:
        """Maximise, or restore if already maximised."""
        if not self._maximizable:
            return
        window = self.window()
        if window.isMaximized():
            window.showNormal()
        else:
            window.showMaximized()

    def update_state(self, maximized: bool) -> None:
        """Show the restore symbol while maximised."""
        self.maximize_button.kind = "restore" if maximized else "maximize"
        self.maximize_button.setToolTip("Restore" if maximized else "Maximise")
        self.maximize_button.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 -- Qt override
        """Dragging the bar moves the window (the OS does it, so snapping works)."""
        window = self.window()
        if event.button() == Qt.MouseButton.LeftButton and window.isVisible():
            window.windowHandle().startSystemMove()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 -- Qt override
        """Double-clicking the bar maximises or restores the window."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


def edges_at(pos: QPoint, size: QSize, margin: int) -> Qt.Edge | None:
    """Which window edges (possibly a corner) ``pos`` is within ``margin`` pixels of."""
    edges = Qt.Edge(0)
    if pos.x() < margin:
        edges |= Qt.Edge.LeftEdge
    elif pos.x() >= size.width() - margin:
        edges |= Qt.Edge.RightEdge
    if pos.y() < margin:
        edges |= Qt.Edge.TopEdge
    elif pos.y() >= size.height() - margin:
        edges |= Qt.Edge.BottomEdge
    return edges if edges != Qt.Edge(0) else None


def _cursor_for(edges: Qt.Edge) -> Qt.CursorShape:
    diagonal = (Qt.Edge.LeftEdge | Qt.Edge.TopEdge, Qt.Edge.RightEdge | Qt.Edge.BottomEdge)
    anti_diagonal = (Qt.Edge.RightEdge | Qt.Edge.TopEdge, Qt.Edge.LeftEdge | Qt.Edge.BottomEdge)
    if edges in diagonal:
        return Qt.CursorShape.SizeFDiagCursor
    if edges in anti_diagonal:
        return Qt.CursorShape.SizeBDiagCursor
    if edges in (Qt.Edge.LeftEdge, Qt.Edge.RightEdge):
        return Qt.CursorShape.SizeHorCursor
    return Qt.CursorShape.SizeVerCursor


class WindowFrame(QWidget):
    """The window's content area with a thin border you can drag to resize the window."""

    def __init__(
        self,
        title_bar: TitleBar,
        body: QWidget,
        parent: QWidget | None = None,
        *,
        resizable: bool = True,
    ) -> None:
        """Stack ``title_bar`` above ``body`` inside a border (draggable if ``resizable``)."""
        super().__init__(parent)
        self._resizable = resizable
        self.setObjectName("windowFrame")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(0)
        self._layout.addWidget(title_bar)
        self._layout.addWidget(body, 1)
        self.set_maximized(False)

    def set_maximized(self, maximized: bool) -> None:
        """No resize border while maximised; a 1 px outline for fixed-size windows."""
        margin = 0 if maximized else (RESIZE_MARGIN if self._resizable else 1)
        self._layout.setContentsMargins(margin, margin, margin, margin)
        self._margin = margin

    def _edges(self, pos: QPoint) -> Qt.Edge | None:
        return edges_at(pos, self.size(), self._margin) if self._resizable else None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 -- Qt override
        """Show a resize cursor over the border."""
        edges = self._edges(event.position().toPoint())
        if edges is None:
            self.unsetCursor()
        else:
            self.setCursor(_cursor_for(edges))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QEvent) -> None:  # noqa: N802 -- Qt override
        """Back to the normal cursor."""
        self.unsetCursor()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 -- Qt override
        """Dragging the border resizes the window (done by the OS)."""
        edges = self._edges(event.position().toPoint())
        window = self.window()
        if event.button() == Qt.MouseButton.LeftButton and edges is not None and window.isVisible():
            window.windowHandle().startSystemResize(edges)
            event.accept()
            return
        super().mousePressEvent(event)
