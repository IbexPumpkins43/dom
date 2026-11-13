"""Offscreen tests for Qt widgets that carry logic."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QPalette, QStandardItemModel
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QListWidget,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dom.domain import Conversation, Message, ModelInfo, ModelStatus, Role
from dom.hardware import MemoryStatus, SystemUsage
from dom.hardware.models import GIB
from dom.hardware.probe import read_usage
from dom.ui.assets import logo_svg, render_logo
from dom.ui.dialogs import FramedDialog, Question, ask
from dom.ui.theme import BRASA, apply_theme
from dom.ui.titlebar import TitleBar, edges_at
from dom.ui.widgets import ChatArea, ModelPicker, Sidebar, UsagePanel


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    existing = QApplication.instance()
    return existing if isinstance(existing, QApplication) else QApplication([])


def test_picker_greys_out_models_that_are_too_large(qt_app: QApplication) -> None:
    picker = ModelPicker()
    reason = "Big needs about 20.0 GB of memory, but only about 6.0 GB is free."
    picker.set_models(
        [
            ModelInfo("Small", ModelStatus.UNLOADED),
            ModelInfo("Big", ModelStatus.UNLOADED, unavailable_reason=reason),
        ],
        selected_id=None,
    )
    model = picker.model()
    assert isinstance(model, QStandardItemModel)
    assert picker.itemText(0) == "Choose a model…"
    assert model.item(1).isEnabled()
    assert not model.item(2).isEnabled()
    assert "too large for this computer" in picker.itemText(2)
    assert picker.itemData(2, Qt.ItemDataRole.ToolTipRole) == reason
    assert qt_app is not None


def _conversation(conversation_id: str, title: str) -> Conversation:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Conversation(conversation_id, title, now, now)


def test_sidebar_delete_button_and_key_request_deletion(qt_app: QApplication) -> None:
    sidebar = Sidebar()
    requested: list[tuple[str, str]] = []

    def record(conversation_id: str, title: str) -> None:
        requested.append((conversation_id, title))

    sidebar.delete_requested.connect(record)
    sidebar.set_conversations([_conversation("a", "First"), _conversation("b", "Second")], "b")
    sidebar_list = sidebar.findChild(QListWidget, "conversationList")
    assert isinstance(sidebar_list, QListWidget)
    row = sidebar_list.itemWidget(sidebar_list.item(0))
    assert row is not None
    button = row.findChild(QToolButton, "deleteChat")
    assert isinstance(button, QToolButton)
    button.click()
    QTest.keyClick(sidebar, Qt.Key.Key_Delete)
    assert requested == [("a", "First"), ("b", "Second")]
    assert qt_app is not None


def test_brasa_theme_is_applied(qt_app: QApplication) -> None:
    apply_theme(qt_app, BRASA)
    window_colour = qt_app.palette().color(QPalette.ColorRole.Window).name().upper()
    assert window_colour == BRASA.base00
    assert BRASA.base09 in qt_app.styleSheet()


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (2, 50, Qt.Edge.LeftEdge),
        (397, 50, Qt.Edge.RightEdge),
        (50, 1, Qt.Edge.TopEdge),
        (50, 298, Qt.Edge.BottomEdge),
        (1, 1, Qt.Edge.LeftEdge | Qt.Edge.TopEdge),
        (399, 299, Qt.Edge.RightEdge | Qt.Edge.BottomEdge),
        (50, 50, None),
    ],
)
def test_edges_at(x: int, y: int, expected: Qt.Edge | None) -> None:
    assert edges_at(QPoint(x, y), QSize(400, 300), 5) == expected


def test_no_resize_edges_when_maximized() -> None:
    assert edges_at(QPoint(0, 0), QSize(400, 300), 0) is None


def test_title_bar_buttons_and_title(qt_app: QApplication) -> None:
    window = QWidget()
    bar = TitleBar(BRASA, window)
    layout = QVBoxLayout(window)
    layout.addWidget(bar)
    window.resize(600, 200)
    window.show()
    bar.set_title("A chat about embers")
    assert bar.title() == "A chat about embers"
    bar.toggle_maximized()
    assert window.isMaximized()
    bar.update_state(window.isMaximized())
    assert bar.maximize_button.kind == "restore"
    bar.toggle_maximized()
    assert not window.isMaximized()
    bar.close_button.click()
    assert not window.isVisible()
    assert qt_app is not None


def test_usage_panel_shows_cpu_and_memory(qt_app: QApplication) -> None:
    panel = UsagePanel()
    panel.set_usage(SystemUsage(23.4, MemoryStatus(16 * GIB, 6 * GIB)))
    assert panel.cpu_text.text() == "23%"
    assert panel.memory_text.text() == "10.0 / 16.0 GB"
    assert panel.cpu_bar.value() == 23  # noqa: PLR2004
    assert panel.memory_bar.value() == 62  # noqa: PLR2004
    assert panel.memory_bar.property("high") is False
    panel.set_usage(SystemUsage(97.0, MemoryStatus(16 * GIB, 1 * GIB)))
    assert panel.cpu_bar.property("high") is True
    assert panel.memory_bar.property("high") is True
    assert qt_app is not None


def test_read_usage_returns_sane_values() -> None:
    usage = read_usage()
    assert 0.0 <= usage.cpu_percent <= 100.0  # noqa: PLR2004
    assert 0 < usage.memory_used_bytes <= usage.memory.total_bytes


def test_framed_dialog_uses_dom_title_bar(qt_app: QApplication) -> None:
    dialog = FramedDialog("Get models", resizable=True)
    assert dialog.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert dialog.title_bar.title() == "Get models"
    assert not dialog.title_bar.maximize_button.isVisibleTo(dialog)
    assert dialog.title_bar.close_button.isVisibleTo(dialog)
    assert qt_app is not None


@pytest.mark.parametrize(("press", "expected"), [("Cancel", []), ("Delete", ["yes"])])
def test_ask_dialog(qt_app: QApplication, press: str, expected: list[str]) -> None:
    parent = QWidget()
    parent.show()
    answers: list[str] = []
    dialog = ask(
        parent,
        Question("Delete chat", "Delete it?", "Delete", danger=True),
        lambda: answers.append("yes"),
    )
    assert dialog.isVisible()
    assert dialog.isModal()
    buttons = dialog.findChildren(QPushButton)
    assert next(b for b in buttons if b.text() == "Cancel").isDefault()
    next(b for b in buttons if b.text() == press).click()
    assert answers == expected
    assert not dialog.isVisible()
    assert qt_app is not None


def test_logo_is_packaged_and_renders(qt_app: QApplication) -> None:
    assert b"<svg" in logo_svg()
    pixmap = render_logo(64, opacity=0.5, device_pixel_ratio=2.0)
    assert pixmap.width() == 128  # noqa: PLR2004 -- 64 px at 2x
    assert 0 < pixmap.height() < pixmap.width()  # a wordmark: wider than tall
    assert not pixmap.toImage().isNull()
    assert qt_app is not None


def test_logo_shows_only_on_empty_chats(qt_app: QApplication) -> None:
    area = ChatArea()
    area.resize(600, 400)
    area.show()
    assert area.logo_visible()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    area.set_messages([Message("m1", "c1", Role.USER, "hi", now)])
    assert not area.logo_visible()
    area.set_messages([])
    assert area.logo_visible()
    area.begin_reply()
    assert not area.logo_visible()
    assert qt_app is not None
