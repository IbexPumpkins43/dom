"""Images shipped with dom (package data, so they are found in packaged builds too)."""

from __future__ import annotations

from importlib.resources import files

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

LOGO_FILE = "logo.svg"
LOGO_COLOUR = "#FF7A4D"  # the fill in logo.svg (Brasa base09)


def logo_svg() -> bytes:
    """The dom logo (the word "dom" in Noto Sans Bold, as outlines) as SVG."""
    return files("dom.ui").joinpath("assets", LOGO_FILE).read_bytes()


def render_logo(
    width: int,
    *,
    opacity: float = 1.0,
    device_pixel_ratio: float = 1.0,
    colour: str = LOGO_COLOUR,
) -> QPixmap:
    """The logo ``width`` pixels wide (height keeps its proportions), sharp on high-DPI screens."""
    svg = logo_svg().replace(LOGO_COLOUR.encode(), colour.encode())
    renderer = QSvgRenderer(QByteArray(svg))
    view = renderer.viewBoxF()
    pixels = round(width * device_pixel_ratio)
    height = max(1, round(pixels * view.height() / view.width()))
    pixmap = QPixmap(pixels, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setOpacity(opacity)
    renderer.render(painter, QRectF(0, 0, pixels, height))
    painter.end()
    pixmap.setDevicePixelRatio(device_pixel_ratio)
    return pixmap
