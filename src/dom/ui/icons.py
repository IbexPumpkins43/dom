"""dom's line icons, drawn as small inline SVGs so they can take any colour (e.g. a tint).

24 x 24 grid, 1.8 px round strokes, no fill: the same look as the window buttons.
"""

from __future__ import annotations

import math
from enum import StrEnum

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

ICON_SIZE = 18
_PIXEL_RATIO = 2.0  # drawn at 2x so they stay sharp on high-DPI screens


class Icon(StrEnum):
    """The icons dom uses."""

    NEW_CHAT = "new_chat"
    DOWNLOAD = "download"
    SETTINGS = "settings"
    SIDEBAR = "sidebar"


def _gear(teeth: int = 8, outer: float = 9.5, inner: float = 7.2, centre: float = 12.0) -> str:
    """A gear outline: ``teeth`` flat-topped teeth around a ring, plus the hub."""
    step = 2 * math.pi / teeth
    points: list[tuple[float, float]] = []
    for i in range(teeth):
        a = i * step
        # valley, rise, flat top, fall: each tooth spans about 40% of its step
        for angle, radius in (
            (a - step * 0.30, inner),
            (a - step * 0.18, outer),
            (a + step * 0.18, outer),
            (a + step * 0.30, inner),
        ):
            points.append((centre + radius * math.cos(angle), centre + radius * math.sin(angle)))
    path = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in points) + " Z"
    return f'<path d="{path}"/><circle cx="12" cy="12" r="3"/>'


_SHAPES: dict[Icon, str] = {
    Icon.NEW_CHAT: '<path d="M12 5v14M5 12h14"/>',
    Icon.DOWNLOAD: '<path d="M12 4v11M7 10.5l5 5 5-5M5 20h14"/>',
    Icon.SETTINGS: _gear(),
    Icon.SIDEBAR: '<rect x="3" y="4.5" width="18" height="15" rx="2.5"/><path d="M9.5 4.5v15"/>',
}


def icon_svg(name: Icon, colour: str) -> str:
    """The icon as an SVG document, stroked in ``colour``."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{colour}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        f"{_SHAPES[name]}</svg>"
    )


def make_icon(name: Icon, colour: str, size: int = ICON_SIZE) -> QIcon:
    """The icon in ``colour`` as a ``QIcon`` of ``size`` pixels."""
    pixels = round(size * _PIXEL_RATIO)
    pixmap = QPixmap(pixels, pixels)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(icon_svg(name, colour).encode())).render(
        painter, QRectF(0, 0, pixels, pixels)
    )
    painter.end()
    pixmap.setDevicePixelRatio(_PIXEL_RATIO)
    return QIcon(pixmap)
