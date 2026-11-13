"""Images shipped with dom (package data, so they are found in packaged builds too)."""

from __future__ import annotations

from importlib.resources import files

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

LOGO_FILE = "logo.svg"


def logo_svg() -> bytes:
    """The dom logo (an ember in a hearth) as SVG."""
    return files("dom.ui").joinpath("assets", LOGO_FILE).read_bytes()


def render_logo(size: int, *, opacity: float = 1.0, device_pixel_ratio: float = 1.0) -> QPixmap:
    """The logo as a ``size`` x ``size`` pixmap, sharp on high-DPI screens."""
    renderer = QSvgRenderer(QByteArray(logo_svg()))
    pixels = round(size * device_pixel_ratio)
    pixmap = QPixmap(pixels, pixels)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setOpacity(opacity)
    renderer.render(painter, QRectF(0, 0, pixels, pixels))
    painter.end()
    pixmap.setDevicePixelRatio(device_pixel_ratio)
    return pixmap
