"""The models screen: a searchable grid of Hugging Face models, shown in place of the chat."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from dom.ui.library import QUANT_FILTERS, Filters, Fit, SizeRange
from dom.ui.widgets import ElidedLabel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PySide6.QtGui import QResizeEvent

    from dom.ui.library import ModelCard, ScreenState, Version

CARD_MIN_WIDTH = 290
CARD_HEIGHT = 168
GRID_SPACING = 12
SEARCH_DELAY_MS = 500  # search after typing pauses this long (or straight away on Enter)
WORDS_PER_TOKEN = 0.75
PROGRESS_STEPS = 1000

SIZE_LABELS: dict[SizeRange, str] = {
    SizeRange.ANY: "Any size",
    SizeRange.TINY: "Up to 2B parameters",
    SizeRange.SMALL: "2B to 5B parameters",
    SizeRange.MEDIUM: "5B to 10B parameters",
    SizeRange.LARGE: "10B to 20B parameters",
    SizeRange.VERY_LARGE: "20B to 40B parameters",
    SizeRange.HUGE: "Over 40B parameters",
}

_FIT_TEXT: dict[Fit, str] = {
    Fit.GOOD: "Runs well on this computer",
    Fit.SLOW: "Slow on this computer",
    Fit.TOO_LARGE: "Too large for this computer",
    Fit.UNKNOWN: "On this computer",
}


def _gb(size: int) -> str:
    return f"{size / 1e9:.1f} GB"


def _count(value: int) -> str:
    """1234567 -> "1.2M"."""
    for limit, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if value >= limit:
            return f"{value / limit:.1f}".rstrip("0").rstrip(".") + suffix
    return str(value)


def facts_text(card: ModelCard) -> str:
    """E.g. "4.2B parameters · 1.2M downloads · 340 likes"."""
    parts: list[str] = []
    if card.params_b is not None:
        parts.append(f"{card.params_b:.1f}B parameters".replace(".0B", "B"))
    if card.downloads is not None:
        parts.append(f"{_count(card.downloads)} downloads")
    if card.likes is not None:
        parts.append(f"{_count(card.likes)} likes")
    return " · ".join(parts)


def fit_text(card: ModelCard) -> str:
    """How the chosen version suits this computer, in plain English."""
    text = _FIT_TEXT[card.fit]
    if card.fit in (Fit.GOOD, Fit.SLOW) and card.est_tokens_per_s is not None:
        words = max(1, round(card.est_tokens_per_s * WORDS_PER_TOKEN))
        text = text.removesuffix(" on this computer") + f" · about {words} words/s"
    return text


def author_html(card: ModelCard) -> str:
    """ "by unsloth · View on Hugging Face" (a link), or "Your file" for unknown files."""
    who = f"by {html.escape(card.author)}" if card.author else "Your file"
    if not card.url:
        return who
    return f'{who} · <a href="{html.escape(card.url, quote=True)}">View on Hugging Face</a>'


def version_text(version: Version) -> str:
    """E.g. "Q4_K_M · 2.7 GB" (sizes are estimates until the download starts)."""
    if version.size_bytes <= 0:
        return version.label
    return f"{version.label} · {_gb(version.size_bytes)}"


class ModelCardWidget(QFrame):
    """One model: name, facts, how it suits this computer, version and Download/Delete."""

    download_requested = Signal(str)
    cancel_requested = Signal(str)
    delete_requested = Signal(str, str)  # file name, title
    version_chosen = Signal(str, str)  # card key, version path

    def __init__(self, card: ModelCard, parent: QWidget | None = None) -> None:
        """Show ``card``."""
        super().__init__(parent)
        self.card = card
        self.setObjectName("modelCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(CARD_HEIGHT)
        self.setMinimumWidth(CARD_MIN_WIDTH)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)
        self._build_text(layout, card)
        self.progress = QProgressBar()
        self.progress.setObjectName("modelCardProgress")
        self.progress.setTextVisible(True)
        self.progress.setFixedHeight(18)
        self.progress.setVisible(card.downloading or card.partial_bytes > 0)
        layout.addWidget(self.progress)
        chosen = card.version(card.selected)
        if chosen is not None and card.partial_bytes:
            self.set_progress(card.partial_bytes, chosen.size_bytes)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.versions = QComboBox()
        self.versions.setObjectName("versionPicker")
        self.versions.setToolTip(
            "Versions of this model: smaller ones need less memory and run faster; larger ones "
            "answer better. Sizes are estimates until the download starts."
        )
        self.version_label = QLabel()
        self.version_label.setObjectName("modelCardFacts")
        self.button = QPushButton()
        self.button.setMinimumWidth(96)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        row.addWidget(self.versions, 1)
        row.addWidget(self.version_label, 1)
        row.addWidget(self.button)
        layout.addLayout(row)
        self._fill(card)

    def _build_text(self, layout: QVBoxLayout, card: ModelCard) -> None:
        heading = QHBoxLayout()
        title = ElidedLabel(card.title)
        title.setObjectName("modelCardTitle")
        title.setToolTip(card.key if not card.key.startswith("local:") else card.title)
        heading.addWidget(title, 1)
        if card.installed is not None:
            badge = QLabel("Downloaded")
            badge.setObjectName("badge")
            badge.setProperty("kind", "installed")
            heading.addWidget(badge)
        layout.addLayout(heading)
        self.author = QLabel(author_html(card))
        self.author.setObjectName("modelCardAuthor")
        self.author.setTextFormat(Qt.TextFormat.RichText)
        self.author.setOpenExternalLinks(True)  # opens the page in the web browser
        if card.url:
            self.author.setToolTip(card.url)
        layout.addWidget(self.author)
        facts = ElidedLabel(facts_text(card))
        facts.setObjectName("modelCardFacts")
        layout.addWidget(facts)
        self.fit_label = ElidedLabel(fit_text(card))
        self.fit_label.setObjectName("modelCardFit")
        self.fit_label.setProperty("fit", card.fit.value)
        layout.addWidget(self.fit_label)
        layout.addStretch(1)

    def _fill(self, card: ModelCard) -> None:
        installed = card.installed is not None
        self.versions.setVisible(not installed)
        self.version_label.setVisible(installed)
        if installed:
            chosen = card.version(card.selected)
            self.version_label.setText(version_text(chosen) if chosen else "")
            self.button.setText("Delete")
            self.button.setObjectName("dangerButton")
            filename = card.installed or ""
            self.button.clicked.connect(
                lambda: self.delete_requested.emit(filename, self.card.title)
            )
            return
        for index, version in enumerate(card.versions):
            self.versions.addItem(version_text(version), version.path)
            if version.fit == Fit.TOO_LARGE:
                self.versions.setItemData(
                    index, "Too large for this computer", Qt.ItemDataRole.ToolTipRole
                )
            if version.path == card.selected:
                self.versions.setCurrentIndex(index)
        self.versions.setEnabled(not card.downloading)
        self.versions.activated.connect(self._on_version)
        if card.downloading:
            self.button.setText("Cancel")
            self.button.clicked.connect(lambda: self.cancel_requested.emit(self.card.key))
        else:
            self.button.setText("Resume" if card.partial_bytes else "Download")
            self.button.setObjectName("primaryButton")
            self.button.setEnabled(card.fit != Fit.TOO_LARGE and card.selected is not None)
            self.button.clicked.connect(lambda: self.download_requested.emit(self.card.key))

    def _on_version(self, index: int) -> None:
        path: object = self.versions.itemData(index)
        if isinstance(path, str) and path != self.card.selected:
            self.version_chosen.emit(self.card.key, path)

    def set_progress(self, done: int, total: int) -> None:
        """Show download progress."""
        self.progress.setVisible(True)
        self.progress.setMaximum(PROGRESS_STEPS)
        self.progress.setValue(
            min(PROGRESS_STEPS, int(PROGRESS_STEPS * done / total)) if total else 0
        )
        self.progress.setFormat(f"{done / 1e9:.1f} of {total / 1e9:.1f} GB")


class ModelsScreen(QWidget):
    """Implements ``LibraryView``: search box, model grid and "Show more"."""

    search_requested = Signal(str)
    load_more_requested = Signal()
    download_requested = Signal(str)
    cancel_requested = Signal(str)
    delete_requested = Signal(str, str)  # file name, title
    version_chosen = Signal(str, str)
    back_requested = Signal()
    filters_changed = Signal(object)  # Filters

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the (empty) screen."""
        super().__init__(parent)
        self.setObjectName("modelsScreen")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._cards: dict[str, ModelCardWidget] = {}
        self._columns = 0
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 22, 28, 0)
        outer.setSpacing(10)

        self._build_header(outer)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("modelGridScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        holder.setObjectName("modelGrid")
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 4, 0, 20)
        self._grid = QGridLayout()
        self._grid.setSpacing(GRID_SPACING)
        holder_layout.addLayout(self._grid)
        self.more_button = QPushButton("Show more")
        self.more_button.clicked.connect(self.load_more_requested.emit)
        holder_layout.addWidget(self.more_button, 0, Qt.AlignmentFlag.AlignHCenter)
        holder_layout.addStretch(1)
        self._scroll.setWidget(holder)
        outer.addWidget(self._scroll, 1)
        self._last_error: str | None = None

    def _build_header(self, outer: QVBoxLayout) -> None:
        top = QHBoxLayout()
        heading = QLabel("Models")
        heading.setObjectName("screenHeading")
        top.addWidget(heading)
        top.addStretch(1)
        back = QPushButton("Back to chat")
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.clicked.connect(self.back_requested.emit)
        top.addWidget(back)
        outer.addLayout(top)
        self.subtitle = ElidedLabel()
        self.subtitle.setObjectName("screenSubtitle")
        outer.addWidget(self.subtitle)

        self.search = QLineEdit()
        self.search.setObjectName("modelSearch")
        self.search.setPlaceholderText("Search Hugging Face models, e.g. qwen, gemma, coder…")
        self.search.setClearButtonEnabled(True)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(SEARCH_DELAY_MS)
        self._search_timer.timeout.connect(self._emit_search)
        self.search.textEdited.connect(self._typed)
        self.search.returnPressed.connect(self._emit_search)
        outer.addWidget(self.search)
        outer.addLayout(self._build_filters())
        self.status = QLabel()
        self.status.setObjectName("screenStatus")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

    def _build_filters(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        self.size_filter = QComboBox()
        self.size_filter.setObjectName("filterBox")
        self.size_filter.setToolTip("Model size: bigger models answer better but need more memory")
        for size, label in SIZE_LABELS.items():
            self.size_filter.addItem(label, size.value)
        self.quant_filter = QComboBox()
        self.quant_filter.setObjectName("filterBox")
        self.quant_filter.setToolTip(
            "Only models offering this version (quantisation). Q8_0 is closest to the original; "
            "lower numbers are smaller and faster but less accurate."
        )
        self.quant_filter.addItem("Any version", "")
        for quant in QUANT_FILTERS:
            self.quant_filter.addItem(quant, quant)
        self.fits_filter = QCheckBox("Only models that fit this computer")
        self.fits_filter.setObjectName("filterCheck")
        for box in (self.size_filter, self.quant_filter):
            box.activated.connect(self._filter_picked)
        self.fits_filter.toggled.connect(self._filter_toggled)
        row.addWidget(self.size_filter)
        row.addWidget(self.quant_filter)
        row.addSpacing(6)
        row.addWidget(self.fits_filter)
        row.addStretch(1)
        return row

    def filters(self) -> Filters:
        """The filters as set on screen."""
        size: object = self.size_filter.currentData()
        quant: object = self.quant_filter.currentData()
        return Filters(
            size=SizeRange(size) if isinstance(size, str) else SizeRange.ANY,
            quant=quant if isinstance(quant, str) and quant else None,
            fits_only=self.fits_filter.isChecked(),
        )

    def _filter_picked(self, _index: int) -> None:
        self.filters_changed.emit(self.filters())

    def _filter_toggled(self, _checked: bool) -> None:
        self.filters_changed.emit(self.filters())

    def _typed(self, _text: str) -> None:
        self._search_timer.start()

    def _emit_search(self) -> None:
        self._search_timer.stop()
        self.search_requested.emit(self.search.text())

    # ---- LibraryView ---------------------------------------------------------------------

    def show_cards(self, cards: Sequence[ModelCard], state: ScreenState) -> None:
        """Replace the grid with ``cards`` (keeping the scroll position)."""
        scroll = self._scroll.verticalScrollBar().value()
        for widget in self._cards.values():
            widget.deleteLater()
        self._cards = {}
        for card in cards:
            widget = ModelCardWidget(card)
            widget.download_requested.connect(self.download_requested.emit)
            widget.cancel_requested.connect(self.cancel_requested.emit)
            widget.delete_requested.connect(self.delete_requested.emit)
            widget.version_chosen.connect(self.version_chosen.emit)
            self._cards[card.key] = widget
        self._columns = 0
        self._arrange()
        self.subtitle.set_full_text(
            f"GGUF chat models from Hugging Face. This computer: {state.hardware_text}"
        )
        self.more_button.setVisible(state.can_load_more and not state.loading)
        self.status.setProperty("error", state.error is not None)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.status.setText(self._status_text(cards, state))
        self.status.setVisible(bool(self.status.text()))
        self._scroll.verticalScrollBar().setValue(scroll)

    @staticmethod
    def _status_text(cards: Sequence[ModelCard], state: ScreenState) -> str:
        if state.loading:
            return "Searching Hugging Face…"
        if state.error is not None:
            return state.error
        if not cards and state.query:
            return f"No chat models found for “{state.query}”."
        if not cards:
            return "No models match these filters."
        return ""

    def show_download_progress(self, key: str, done: int, total: int) -> None:
        """Move one card's progress bar."""
        widget = self._cards.get(key)
        if widget is not None:
            widget.set_progress(done, total)

    def show_library_error(self, message: str) -> None:
        """Show a problem above the grid."""
        self.status.setProperty("error", True)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.status.setText(message)
        self.status.show()

    # ---- layout ------------------------------------------------------------------------------

    def card_widgets(self) -> list[ModelCardWidget]:
        """The cards in grid order (for tests)."""
        return list(self._cards.values())

    def columns(self) -> int:
        """How many cards fit side by side."""
        return self._columns

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 -- Qt override
        """Re-flow the grid when the width changes."""
        super().resizeEvent(event)
        self._arrange()

    def _arrange(self) -> None:
        width = self._scroll.viewport().width()
        columns = max(1, (width + GRID_SPACING) // (CARD_MIN_WIDTH + GRID_SPACING))
        if columns == self._columns:
            return
        self._columns = columns
        while self._grid.count():
            self._grid.takeAt(0)
        for index, widget in enumerate(self._cards.values()):
            self._grid.addWidget(widget, index // columns, index % columns)
        for column in range(self._grid.columnCount()):
            self._grid.setColumnStretch(column, 1 if column < columns else 0)
