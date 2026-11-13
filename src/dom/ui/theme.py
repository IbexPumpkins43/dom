"""The app's look: the base16 "Brasa" colour scheme, applied to Qt's palette and stylesheet.

Colours follow the base16 styling guidelines (https://github.com/tinted-theming/home):
base00-base03 are backgrounds from darkest to lightest, base04-base07 foregrounds, base08 red
(errors), base09 orange (dom's accent), base0A yellow, base0B green, base0C cyan, base0D blue
(links), base0E magenta, base0F brown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtGui import QColor, QPalette

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication


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


def build_stylesheet(t: Base16) -> str:
    """The app's stylesheet in the scheme's colours."""
    return f"""
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
#titleBar {{ background: {t.base01}; border-bottom: 1px solid {t.base02}; }}
#titleBarName {{ color: {t.base09}; font-weight: 700; font-size: 13px; letter-spacing: 1px; }}
#titleBarTitle {{ color: {t.base04}; font-size: 12px; }}
#sidebar {{ background: {t.base01}; border-right: 1px solid {t.base02}; }}
#sidebarHeading {{
    color: {t.base04};
    font-size: 11px;
    font-weight: 600;
    padding: 6px 4px 0 4px;
}}
#newChat {{ text-align: left; background: {t.base00}; padding: 8px 12px; }}
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
#assistantBubble {{ background: transparent; color: {t.base05}; }}
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

QDialog {{ background: {t.base00}; }}
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


def apply_theme(app: QApplication, theme: Base16 = BRASA) -> None:
    """Use ``theme`` for every window of ``app``."""
    app.setStyle("Fusion")
    app.setPalette(build_palette(theme))
    app.setStyleSheet(build_stylesheet(theme))
