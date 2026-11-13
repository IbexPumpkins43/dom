"""Images and fonts shipped with dom (package data, so they are found in packaged builds too)."""

from __future__ import annotations

import logging
from importlib.resources import files
from typing import TYPE_CHECKING

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QFontDatabase, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from dom.ui.theme import UI_FONT_FAMILY

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

logger = logging.getLogger(__name__)

# Figtree (SIL Open Font License 1.1, assets/fonts/OFL.txt): static Regular, Medium, SemiBold,
# Bold, Italic and Bold Italic cut from the variable font, which Qt handles unevenly per platform.
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


def load_fonts() -> set[str]:
    """Register the fonts shipped with dom with Qt; return their family names."""
    families: set[str] = set()
    for entry in files("dom.ui").joinpath("assets", "fonts").iterdir():
        if not entry.name.endswith(".ttf"):
            continue
        font_id = QFontDatabase.addApplicationFontFromData(QByteArray(entry.read_bytes()))
        if font_id < 0:
            logger.warning("Could not load the font %s", entry.name)
            continue
        families.update(QFontDatabase.applicationFontFamilies(font_id))
    return families


def use_ui_font(app: QApplication) -> bool:
    """Make Figtree the default font (keeping the system's font size); call before
    ``apply_theme``, whose stylesheet then applies it to every kind of widget.

    Returns False (and keeps the system font) if it could not be loaded.
    """
    if UI_FONT_FAMILY not in load_fonts():
        logger.warning("The %s font could not be loaded; using the system font", UI_FONT_FAMILY)
        return False
    font = app.font()
    font.setFamilies([UI_FONT_FAMILY])
    app.setFont(font)
    return True
