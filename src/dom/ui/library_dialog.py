"""The "Get models" window."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from dom.catalog.recommend import needed_bytes
from dom.hardware.models import GIB

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dom.ui.library import CatalogRow

# Roughly how many words a token is, for showing speed in words rather than tokens.
WORDS_PER_TOKEN = 0.75


def _gb(size: int) -> str:
    return f"{size / 1e9:.1f} GB"


def describe_row(row: CatalogRow) -> str:
    """The details line under a model's name."""
    item = row.suggestion
    model = item.model
    size = f"{model.params_b:g}B parameters" + (
        f" ({model.active_params_b:g}B active)" if model.is_moe else ""
    )
    if item.quant is None:
        smallest = min(model.quants, key=lambda q: q.size_bytes)
        return (
            f"{model.maker} · {size} · too large for this computer "
            f"(needs about {needed_bytes(smallest) / GIB:.0f} GB of memory)"
        )
    parts = [model.maker, size, item.quant.label, f"{_gb(item.quant.size_bytes)} download"]
    if item.est_tokens_per_s is not None:
        parts.append(f"about {item.est_tokens_per_s * WORDS_PER_TOKEN:.0f} words/second")
    return " · ".join(parts)


class _ModelRow(QFrame):
    """One catalog model with its Download/Cancel button and progress bar."""

    def __init__(self, row: CatalogRow) -> None:
        super().__init__()
        self.setObjectName("modelRow")
        item = row.suggestion
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        text = QVBoxLayout()
        title = QLabel(item.model.name)
        title.setObjectName("modelRowTitle")
        badges = [
            b for b, on in (("Recommended", item.recommended), ("Downloaded", item.installed)) if on
        ]
        heading = QHBoxLayout()
        heading.addWidget(title)
        for badge in badges:
            label = QLabel(badge)
            label.setObjectName("badge")
            heading.addWidget(label)
        heading.addStretch(1)
        text.addLayout(heading)
        summary = QLabel(item.model.summary)
        summary.setWordWrap(True)
        text.addWidget(summary)
        details = QLabel(describe_row(row))
        details.setObjectName("modelRowDetails")
        details.setWordWrap(True)
        text.addWidget(details)
        layout.addLayout(text, 1)

        side = QVBoxLayout()
        self.button = QPushButton()
        self.button.setMinimumWidth(120)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(160)
        self.progress.setTextVisible(True)
        side.addWidget(self.button)
        side.addWidget(self.progress)
        side.addStretch(1)
        layout.addLayout(side)

        self.progress.setVisible(row.downloading or row.partial_bytes > 0)
        if item.quant is not None and row.partial_bytes:
            self.set_progress(row.partial_bytes, item.quant.size_bytes)
        if item.installed:
            self.button.setText("Downloaded")
            self.button.setEnabled(False)
        elif item.quant is None:
            self.button.setText("Too large")
            self.button.setEnabled(False)
            self.setEnabled(False)
        elif row.downloading:
            self.button.setText("Cancel")
        else:
            self.button.setText("Resume" if row.partial_bytes else "Download")

    def set_progress(self, done: int, total: int) -> None:
        self.progress.setVisible(True)
        self.progress.setMaximum(1000)
        self.progress.setValue(int(1000 * done / total) if total else 0)
        self.progress.setFormat(f"{done / 1e9:.1f} of {total / 1e9:.1f} GB")


class LibraryDialog(QDialog):
    """Implements ``LibraryView``: lists catalog models and their downloads."""

    download_requested = Signal(str)
    cancel_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the (empty) window."""
        super().__init__(parent)
        self.setWindowTitle("Get models")
        self.resize(760, 680)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        heading = QLabel("Get models")
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)
        self._hardware = QLabel()
        self._hardware.setWordWrap(True)
        self._hardware.setObjectName("modelRowDetails")
        layout.addWidget(self._hardware)
        self._error = QLabel()
        self._error.setObjectName("errorLabel")
        self._error.setWordWrap(True)
        self._error.hide()
        layout.addWidget(self._error)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._list = QWidget()
        self._list_layout = QVBoxLayout(self._list)
        self._list_layout.setSpacing(8)
        self._list_layout.addStretch(1)
        scroll.setWidget(self._list)
        layout.addWidget(scroll, 1)
        self._rows: dict[str, _ModelRow] = {}

    def show_catalog(self, rows: Sequence[CatalogRow], hardware_text: str) -> None:
        """Rebuild the list."""
        self._hardware.setText(
            f"This computer: {hardware_text}.\nModels download from Hugging Face and then run "
            "entirely on this computer. You can keep chatting while a model downloads."
        )
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item is not None and (widget := item.widget()) is not None:
                widget.deleteLater()
        self._rows.clear()
        for row in rows:
            widget = _ModelRow(row)
            key = row.suggestion.model.key
            downloading = row.downloading
            widget.button.clicked.connect(
                lambda _checked=False, k=key, d=downloading: self._clicked(k, d)
            )
            self._rows[key] = widget
            self._list_layout.insertWidget(self._list_layout.count() - 1, widget)

    def show_download_progress(self, key: str, done: int, total: int) -> None:
        """Update one progress bar."""
        row = self._rows.get(key)
        if row is not None:
            row.set_progress(done, total)

    def show_library_error(self, message: str) -> None:
        """Show a problem above the list."""
        self._error.setText(message)
        self._error.setVisible(bool(message))

    def _clicked(self, key: str, downloading: bool) -> None:
        self._error.hide()
        (self.cancel_requested if downloading else self.download_requested).emit(key)
