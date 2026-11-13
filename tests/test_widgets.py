"""Offscreen tests for Qt widgets that carry logic."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QStandardItemModel
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QListWidget, QToolButton

from dom.domain import Conversation, ModelInfo, ModelStatus
from dom.ui.theme import BRASA, apply_theme
from dom.ui.widgets import ModelPicker, Sidebar


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
