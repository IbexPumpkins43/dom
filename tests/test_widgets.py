"""Offscreen tests for Qt widgets that carry logic."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QApplication

from dom.domain import ModelInfo, ModelStatus
from dom.ui.widgets import ModelPicker


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
