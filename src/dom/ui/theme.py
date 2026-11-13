"""The app's look: the base16 "Brasa" colour scheme, applied to Qt's palette and stylesheet.

Colours follow the base16 styling guidelines (https://github.com/tinted-theming/home):
base00-base03 are backgrounds from darkest to lightest, base04-base07 foregrounds, base08 red
(errors), base09 orange (dom's accent), base0A yellow, base0B green, base0C cyan, base0D blue
(links), base0E magenta, base0F brown.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from PySide6.QtGui import QColor, QFontDatabase, QPalette

from dom.domain import ProfileColour

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

    from dom.domain import ProfileSettings


@dataclass(frozen=True, slots=True)
class Base16:
    """A base16 colour scheme."""

    name: str
    base00: str
    base01: str
    base02: str
    base03: str
    base04: str
    base05: str
    base06: str
    base07: str
    base08: str
    base09: str
    base0A: str  # noqa: N815 -- base16's own names
    base0B: str  # noqa: N815
    base0C: str  # noqa: N815
    base0D: str  # noqa: N815
    base0E: str  # noqa: N815
    base0F: str  # noqa: N815


# https://github.com/tinted-theming/schemes/blob/spec-0.11/base16/brasa.yaml (author: Teshre)
BRASA = Base16(
    name="Brasa",
    base00="#1A0F0A",
    base01="#2B1C14",
    base02="#45261A",
    base03="#7A6150",
    base04="#B59D88",
    base05="#F0D8C0",
    base06="#F6E1CC",
    base07="#FBEAD8",
    base08="#F2685A",
    base09="#FF7A4D",
    base0A="#F0B23A",
    base0B="#B8C24A",
    base0C="#6BC8B8",
    base0D="#9AA6E0",
    base0E="#E68AA2",
    base0F="#5A3A28",
)


def profile_colour(t: Base16, colour: ProfileColour) -> str:
    """The scheme's accent colour for a profile avatar colour."""
    return {
        ProfileColour.ORANGE: t.base09,
        ProfileColour.RED: t.base08,
        ProfileColour.YELLOW: t.base0A,
        ProfileColour.GREEN: t.base0B,
        ProfileColour.CYAN: t.base0C,
        ProfileColour.BLUE: t.base0D,
        ProfileColour.MAGENTA: t.base0E,
    }[colour]


def _with_hue(colour: str, hue: float) -> str:
    """``colour`` with its hue replaced (lightness and saturation kept)."""
    original = QColor(colour)
    return QColor.fromHslF(hue, original.hslSaturationF(), original.lightnessF()).name().upper()


def tinted(t: Base16, accent: str) -> Base16:
    """``t`` recoloured around ``accent``: backgrounds and text take its hue, and it becomes
    the accent colour (base09). Lightness is kept, so contrast stays the same."""
    hue = max(0.0, QColor(accent).hslHueF())
    shift = {
        name: _with_hue(getattr(t, name), hue)
        for name in ("base00", "base01", "base02", "base03", "base04", "base05", "base06",
                     "base07", "base0F")
    }  # fmt: skip
    return replace(t, name=f"{t.name} ({accent})", base09=accent.upper(), **shift)


def theme_for(settings: ProfileSettings, base: Base16 = BRASA) -> Base16:
    """The look for a profile: ``base``, tinted in the profile's colour if it asks for that."""
    if not settings.tint:
        return base
    return tinted(base, profile_colour(base, settings.colour))


def build_palette(t: Base16) -> QPalette:
    """Qt palette roles from the scheme (covers widgets the stylesheet does not, e.g. menus)."""
    palette = QPalette()
    roles: dict[QPalette.ColorRole, str] = {
        QPalette.ColorRole.Window: t.base00,
        QPalette.ColorRole.WindowText: t.base05,
        QPalette.ColorRole.Base: t.base00,
        QPalette.ColorRole.AlternateBase: t.base01,
        QPalette.ColorRole.Text: t.base05,
        QPalette.ColorRole.Button: t.base01,
        QPalette.ColorRole.ButtonText: t.base05,
        QPalette.ColorRole.BrightText: t.base07,
        QPalette.ColorRole.Highlight: t.base02,
        QPalette.ColorRole.HighlightedText: t.base07,
        QPalette.ColorRole.ToolTipBase: t.base01,
        QPalette.ColorRole.ToolTipText: t.base05,
        QPalette.ColorRole.PlaceholderText: t.base04,
        QPalette.ColorRole.Link: t.base0D,
        QPalette.ColorRole.LinkVisited: t.base0E,
        QPalette.ColorRole.Light: t.base03,
        QPalette.ColorRole.Midlight: t.base02,
        QPalette.ColorRole.Mid: t.base03,
        QPalette.ColorRole.Dark: t.base00,
        QPalette.ColorRole.Shadow: t.base00,
    }
    for role, colour in roles.items():
        palette.setColor(role, QColor(colour))
    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(disabled, role, QColor(t.base03))
    return palette


UI_FONT_FAMILY = "Figtree"  # bundled in ui/assets/fonts, registered by ui.assets.use_ui_font


def build_stylesheet(t: Base16, font_family: str | None = None) -> str:
    """The app's stylesheet in the scheme's colours, in ``font_family`` if given.

    The font is set here, not only with ``QApplication.setFont``: desktop themes (e.g. GNOME)
    give Qt a font per widget type (labels, buttons, menus, lists, tooltips...), and only a
    stylesheet overrides all of them.
    """
    font_rule = f'* {{ font-family: "{font_family}"; }}\n' if font_family else ""
    return (
        font_rule
        + f"""
QToolTip {{ background: {t.base01}; color: {t.base05}; border: 1px solid {t.base03}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {t.base02}; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {t.base03}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QPushButton {{
    background: {t.base01};
    color: {t.base05};
    border: 1px solid {t.base02};
    border-radius: 4px;
    padding: 6px 12px;
}}
QPushButton:hover {{ background: {t.base02}; }}
QPushButton:disabled {{ color: {t.base03}; border-color: {t.base01}; }}

#windowFrame {{ background: {t.base01}; }}
#usagePanel {{ border-top: 1px solid {t.base02}; }}
#usageLabel {{ color: {t.base04}; font-size: 11px; font-weight: 600; }}
#usageValue {{ color: {t.base05}; font-size: 11px; }}
#cpuBar, #memoryBar {{ background: {t.base00}; border: none; border-radius: 2px; }}
#cpuBar::chunk {{ background: {t.base0A}; border-radius: 2px; }}
#memoryBar::chunk {{ background: {t.base09}; border-radius: 2px; }}
#cpuBar[high="true"]::chunk, #memoryBar[high="true"]::chunk {{ background: {t.base08}; }}
#titleBar {{ background: {t.base01}; border-bottom: 1px solid {t.base02}; }}
#titleBarName {{ color: {t.base09}; font-weight: 700; font-size: 13px; letter-spacing: 1px; }}
#titleBarTitle {{ color: {t.base04}; font-size: 16px; }}
#profileButton {{
    background: transparent;
    color: {t.base06};
    border: none;
    border-radius: 4px;
    padding: 6px 8px;
    text-align: left;
    font-size: 13px;
    font-weight: 600;
}}
#profileButton:hover, #profileButton:pressed {{ background: {t.base02}; }}
#profileButton[compact="true"] {{ text-align: center; padding: 6px 0; }}
#profileRow {{ border-top: 1px solid {t.base02}; }}
#sidebarIconButton, #sidebarToggle {{
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 0;
    min-width: 36px;
    max-width: 36px;
    min-height: 32px;
    max-height: 32px;
}}
#sidebarIconButton:hover, #sidebarToggle:hover {{ background: {t.base02}; }}
#sidebarToggle {{ min-height: 28px; max-height: 28px; }}
QMenu {{ background: {t.base01}; color: {t.base05}; border: 1px solid {t.base03}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px 6px 8px; border-radius: 3px; }}
QMenu::item:selected {{ background: {t.base02}; }}
QMenu::item:disabled {{ color: {t.base03}; }}
QMenu::separator {{ height: 1px; background: {t.base02}; margin: 4px 2px; }}
#sidebar {{ background: {t.base01}; border-right: 1px solid {t.base02}; }}
#sidebarHeading {{
    color: {t.base04};
    font-size: 11px;
    font-weight: 600;
    padding: 6px 4px 0 4px;
}}
#newChat {{ text-align: left; background: {t.base00}; padding: 8px 10px; }}
#newChat[compact="true"] {{ text-align: center; padding: 8px 0; }}
#newChat:hover {{ background: {t.base02}; }}
#conversationList {{ background: transparent; outline: none; border: none; }}
#conversationList::item {{ border-radius: 3px; }}
#conversationList::item:hover {{ background: {t.base02}; }}
#conversationList::item:selected {{ background: {t.base02}; }}
#conversationTitle {{ color: {t.base05}; }}
#deleteChat {{
    border: none;
    background: transparent;
    color: {t.base04};
    padding: 2px 6px;
    font-size: 13px;
}}
#deleteChat:hover {{ color: {t.base08}; background: {t.base00}; }}

#footer QComboBox, #modelPicker {{
    background: {t.base01};
    color: {t.base05};
    border: 1px solid {t.base02};
    border-radius: 4px;
    padding: 5px 10px;
}}
#modelPicker:hover {{ border-color: {t.base03}; }}
#modelPicker QAbstractItemView {{
    background: {t.base01};
    color: {t.base05};
    selection-background-color: {t.base02};
    border: 1px solid {t.base03};
}}
#serverLabel, #detailsLabel {{ color: {t.base04}; font-size: 12px; }}
#serverLabel[state="failed"], #serverLabel[state="crashed"],
#serverLabel[state="engine_missing"] {{ color: {t.base08}; }}

#banner {{ background: {t.base01}; border-bottom: 1px solid {t.base02}; color: {t.base05}; }}
#banner[error="true"] {{ background: {t.base02}; border-bottom: 1px solid {t.base08}; }}

#chatPane, #chatArea, #chatCanvas {{ background: {t.base00}; }}
#notice {{ color: {t.base04}; font-size: 15px; padding: 60px 20px; }}
#userBubble {{ background: {t.base02}; border: none; border-radius: 6px; color: {t.base06}; }}
#assistantBubble {{
    background: {t.base01};
    border: 1px solid {t.base02};
    border-radius: 6px;
    color: {t.base05};
}}
#thinkingToggle {{
    border: none;
    color: {t.base04};
    font-size: 12px;
    padding: 0;
    background: transparent;
}}
#thinkingText {{
    color: {t.base04};
    font-size: 12px;
    border-left: 2px solid {t.base03};
    padding-left: 10px;
}}

#composer {{
    background: {t.base01};
    border: 1px solid {t.base03};
    border-radius: 6px;
}}
#composer[focused="true"] {{ border-color: {t.base09}; }}
#chatInput {{
    border: none;
    background: transparent;
    color: {t.base06};
    font-size: 14px;
    selection-background-color: {t.base02};
}}
#sendButton {{
    padding: 7px 18px;
    border-radius: 5px;
    background: {t.base09};
    color: {t.base00};
    border: none;
    font-weight: 600;
}}
#sendButton:hover {{ background: {t.base0A}; }}
#sendButton:disabled {{ background: {t.base02}; color: {t.base03}; }}
#sendButton[stop="true"] {{ background: {t.base08}; color: {t.base00}; }}
#errorLabel {{ color: {t.base08}; }}

QDialog, #dialogContent {{ background: {t.base00}; }}
#dialogMessage {{ color: {t.base05}; font-size: 13px; }}
#fieldLabel {{ color: {t.base06}; font-weight: 600; font-size: 13px; }}
#fieldHint {{ color: {t.base04}; font-size: 12px; }}
#profileDialog QLineEdit, #profileDialog QPlainTextEdit, #profileDialog QComboBox {{
    background: {t.base01};
    color: {t.base06};
    border: 1px solid {t.base03};
    border-radius: 4px;
    padding: 5px 8px;
    selection-background-color: {t.base02};
}}
#profileDialog QLineEdit:focus, #profileDialog QPlainTextEdit:focus,
#profileDialog QComboBox:focus {{ border-color: {t.base09}; }}
#profileDialog QComboBox QAbstractItemView {{
    background: {t.base01};
    color: {t.base05};
    selection-background-color: {t.base02};
    border: 1px solid {t.base03};
}}
#profileDialog QCheckBox {{ color: {t.base06}; font-size: 13px; spacing: 8px; }}
#profileDialog QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {t.base03};
    border-radius: 3px;
    background: {t.base01};
}}
#profileDialog QCheckBox::indicator:hover {{ border-color: {t.base04}; }}
#profileDialog QCheckBox::indicator:checked {{ background: {t.base09}; border-color: {t.base09}; }}
#dangerButton {{ background: {t.base08}; color: {t.base00}; border: none; font-weight: 600; }}
#dangerButton:hover {{ background: {t.base09}; }}
#primaryButton {{ background: {t.base09}; color: {t.base00}; border: none; font-weight: 600; }}
#primaryButton:hover {{ background: {t.base0A}; }}
#dialogHeading {{ font-size: 18px; font-weight: 600; color: {t.base06}; }}
#modelRow {{ border: 1px solid {t.base02}; border-radius: 5px; background: {t.base01}; }}
#modelRowTitle {{ font-weight: 600; font-size: 14px; color: {t.base06}; }}
#modelRowDetails {{ color: {t.base04}; font-size: 12px; }}
#badge {{
    background: {t.base09};
    color: {t.base00};
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 11px;
    font-weight: 600;
}}
#badge[kind="installed"] {{ background: {t.base0B}; }}
QProgressBar {{
    background: {t.base00};
    border: 1px solid {t.base02};
    border-radius: 3px;
    color: {t.base05};
    text-align: center;
}}
QProgressBar::chunk {{ background: {t.base09}; border-radius: 3px; }}
"""
    )


def apply_theme(app: QApplication, theme: Base16 = BRASA) -> None:
    """Use ``theme`` for every window of ``app``."""
    app.setStyle("Fusion")
    app.setPalette(build_palette(theme))
    family = UI_FONT_FAMILY if QFontDatabase.hasFamily(UI_FONT_FAMILY) else None
    app.setStyleSheet(build_stylesheet(theme, family))
