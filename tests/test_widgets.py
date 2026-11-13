"""Offscreen tests for Qt widgets that carry logic."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontInfo, QPalette, QStandardItemModel
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QListWidget,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dom.config.settings import GenerationSettings
from dom.domain import (
    Conversation,
    Message,
    ModelInfo,
    ModelStatus,
    Profile,
    ProfileColour,
    ProfileSettings,
    Role,
)
from dom.hardware import MemoryStatus, SystemUsage
from dom.hardware.models import GIB
from dom.hardware.probe import read_usage
from dom.ui.assets import load_fonts, logo_svg, render_logo
from dom.ui.dialogs import FramedDialog, Question, ask
from dom.ui.icons import Icon, icon_svg, make_icon
from dom.ui.main_window import MainWindow
from dom.ui.profile_widgets import (
    BUTTON_NAME_MAX_CHARS,
    ProfileButton,
    ProfileDialog,
    avatar_pixmap,
)
from dom.ui.theme import (
    BRASA,
    UI_FONT_FAMILY,
    apply_theme,
    profile_colour,
    theme_for,
    tinted,
)
from dom.ui.titlebar import TitleBar, edges_at
from dom.ui.widgets import (
    SIDEBAR_COMPACT_WIDTH,
    SIDEBAR_WIDTH,
    ChatArea,
    ModelPicker,
    Sidebar,
    UsagePanel,
)


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


def _profile(profile_id: str, name: str, colour: ProfileColour) -> Profile:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Profile(profile_id, ProfileSettings(name, colour), now, now)


def test_profile_button_menu(qt_app: QApplication) -> None:
    button = ProfileButton()
    home = _profile("p1", "Home", ProfileColour.ORANGE)
    work = _profile("p2", "A very long profile name indeed, much longer", ProfileColour.BLUE)
    button.set_profiles([home], home)
    assert button.text() == "Home"
    assert button.menu_actions() == [
        "Home", "---", "Edit this profile…", "Add profile…", "Delete this profile…",
    ]  # fmt: skip
    assert not next(
        a for a in button.profile_menu().actions() if a.text().startswith("Delete")
    ).isEnabled()

    button.set_profiles([home, work], work)
    assert button.text().endswith("…")
    assert len(button.text()) <= BUTTON_NAME_MAX_CHARS
    seen: list[str] = []
    button.switch_requested.connect(seen.append)
    button.add_requested.connect(lambda: seen.append("add"))
    button.edit_requested.connect(lambda: seen.append("edit"))
    button.delete_requested.connect(lambda: seen.append("delete"))
    for text in ("Home", "Add profile…", "Edit this profile…", "Delete this profile…"):
        button.trigger(text)
    assert seen == ["p1", "add", "edit", "delete"]
    assert qt_app is not None


def test_profile_button_is_at_the_bottom_of_the_sidebar(qt_app: QApplication) -> None:
    window = MainWindow(on_closed=lambda: None)
    home = _profile("p1", "Home", ProfileColour.ORANGE)
    window.show_profiles([home], home)
    window.resize(1000, 700)
    window.show()
    qt_app.processEvents()
    sidebar = window.findChild(Sidebar)
    assert sidebar is not None
    button = window.profile_button()
    assert sidebar.isAncestorOf(button)
    top_left = button.mapTo(sidebar, button.rect().topLeft())
    bottom = button.mapTo(sidebar, button.rect().bottomLeft()).y()
    assert top_left.y() > sidebar.usage.geometry().bottom()  # under the CPU/memory panel
    assert sidebar.height() - bottom < 20  # noqa: PLR2004 -- at the bottom
    settings = sidebar.settings_button
    assert settings.mapTo(sidebar, settings.rect().topLeft()).x() > top_left.x()  # beside it
    window.close()


def test_compact_sidebar_shows_icons_only(qt_app: QApplication) -> None:
    window = MainWindow(on_closed=lambda: None)
    home = _profile("p1", "Home", ProfileColour.ORANGE)
    window.show_profiles([home], home)
    window.resize(1000, 700)
    window.show()
    sidebar = window.findChild(Sidebar)
    assert sidebar is not None
    button = window.profile_button()
    assert "Collapse" in window.sidebar_toggle().toolTip()

    window.sidebar_toggle().click()
    qt_app.processEvents()
    assert sidebar.is_compact()
    assert sidebar.width() == SIDEBAR_COMPACT_WIDTH
    assert button.text() == ""
    chat_list = sidebar.findChild(QListWidget, "conversationList")
    assert chat_list is not None
    assert not chat_list.isVisible()
    buttons = [b for b in sidebar.findChildren(QPushButton) if b.isVisible()]
    assert all(b.text() == "" and not b.icon().isNull() for b in buttons)
    assert "Expand" in window.sidebar_toggle().toolTip()

    QTest.keyClick(window, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)  # Ctrl+B
    qt_app.processEvents()
    assert not sidebar.is_compact()
    assert sidebar.width() == SIDEBAR_WIDTH
    assert button.text() == "Home"
    window.close()


def test_main_window_title_bar_has_the_sidebar_button_not_the_name(qt_app: QApplication) -> None:
    window = MainWindow(on_closed=lambda: None)
    names = [label.text() for label in window.findChildren(QLabel, "titleBarName")]
    assert names == []
    dialog = FramedDialog("Dialog")  # dialogs keep the name
    assert [label.text() for label in dialog.findChildren(QLabel, "titleBarName")] == ["dom"]
    assert qt_app is not None


def test_icons_render_in_the_given_colour(qt_app: QApplication) -> None:
    for name in Icon:
        assert 'stroke="#123456"' in icon_svg(name, "#123456")
        assert not make_icon(name, "#123456").isNull()
    assert qt_app is not None


def test_avatar_is_a_coloured_circle(qt_app: QApplication) -> None:
    pixmap = avatar_pixmap(ProfileSettings("sam", ProfileColour.GREEN), 20, device_pixel_ratio=2)
    image = pixmap.toImage()
    assert image.width() == 40  # noqa: PLR2004
    assert image.pixelColor(2, 20).name().upper() == BRASA.base0B.upper()  # left edge: fill
    assert image.pixelColor(0, 0).alpha() == 0  # corner: outside the circle
    assert qt_app is not None


def test_profile_dialog_round_trip(qt_app: QApplication) -> None:
    initial = ProfileSettings(
        "Work",
        ProfileColour.CYAN,
        "Be brief.",
        GenerationSettings(temperature=0.35, max_reply_tokens=None, thinking=False),
    )
    saved: list[ProfileSettings] = []
    dialog = ProfileDialog(None, initial, saved.append, adding=False)
    assert dialog.settings() == initial  # custom temperature kept as a "Custom" choice
    assert dialog.save_button.text() == "Save"

    dialog.name_edit.setText("   ")
    assert not dialog.save_button.isEnabled()
    dialog.name_edit.setText("Office")
    assert dialog.save_button.isEnabled()
    dialog.style_combo.setCurrentIndex(1)  # precise
    dialog.length_combo.setCurrentIndex(0)  # short
    dialog.thinking_check.setChecked(True)
    dialog.tint_check.setChecked(True)
    QTest.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)
    assert saved == [
        ProfileSettings(
            "Office",
            ProfileColour.CYAN,
            "Be brief.",
            GenerationSettings(temperature=0.2, max_reply_tokens=1024, thinking=True),
            tint=True,
        )
    ]
    assert qt_app is not None


def test_add_profile_dialog_cancel_saves_nothing(qt_app: QApplication) -> None:
    saved: list[ProfileSettings] = []
    dialog = ProfileDialog(
        None, ProfileSettings("Profile 2", ProfileColour.RED), saved.append, adding=True
    )
    assert dialog.windowTitle() == "Add profile"
    assert dialog.save_button.text() == "Add profile"
    QTest.mouseClick(dialog.cancel_button, Qt.MouseButton.LeftButton)
    assert saved == []
    assert qt_app is not None


def test_tinted_theme_keeps_lightness_and_uses_the_profile_colour() -> None:
    blue = profile_colour(BRASA, ProfileColour.BLUE)
    theme = tinted(BRASA, blue)
    assert theme.base09 == blue.upper()
    assert theme.base08 == BRASA.base08  # errors stay red
    for name in ("base00", "base02", "base05"):
        before, after = QColor(getattr(BRASA, name)), QColor(getattr(theme, name))
        assert abs(before.lightnessF() - after.lightnessF()) < 0.01  # noqa: PLR2004
        assert abs(after.hslHueF() - QColor(blue).hslHueF()) < 0.02  # noqa: PLR2004
    plain = ProfileSettings("x", ProfileColour.BLUE)
    assert theme_for(plain) is BRASA
    assert theme_for(replace(plain, tint=True)) == theme


def test_window_follows_the_profile_tint(qt_app: QApplication) -> None:
    window = MainWindow(on_closed=lambda: None)
    plain = _profile("p1", "Home", ProfileColour.BLUE)
    tinted_profile = replace(plain, settings=replace(plain.settings, tint=True))
    try:
        window.show_profiles([tinted_profile], tinted_profile)
        expected = theme_for(tinted_profile.settings)
        assert window.theme() == expected
        assert qt_app.palette().color(QPalette.ColorRole.Window).name().upper() == expected.base00
        window.show_profiles([plain], plain)
        assert window.theme() is BRASA
        assert qt_app.palette().color(QPalette.ColorRole.Window).name().upper() == BRASA.base00
    finally:
        apply_theme(qt_app)


def test_logo_can_be_recoloured(qt_app: QApplication) -> None:
    image = render_logo(200, colour="#00FF00").toImage()
    colours = {
        image.pixelColor(x, y).name()
        for x in range(0, image.width(), 4)
        for y in range(0, image.height(), 4)
        if image.pixelColor(x, y).alpha() == 255  # noqa: PLR2004
    }
    assert colours == {"#00ff00"}
    assert qt_app is not None


def test_figtree_is_bundled_in_every_weight(qt_app: QApplication) -> None:
    assert UI_FONT_FAMILY in load_fonts()
    styles = set(QFontDatabase.styles(UI_FONT_FAMILY))
    assert {"Regular", "Medium", "SemiBold", "Bold", "Italic", "Bold Italic"} <= styles
    assert qt_app is not None


def test_ui_font_beats_per_widget_type_theme_fonts(qt_app: QApplication) -> None:
    # Desktop themes (e.g. GNOME) set a font per widget type; the stylesheet must win.
    load_fonts()
    classes = ("QLabel", "QPushButton", "QMenu")
    for cls in classes:
        QApplication.setFont(QFont("DejaVu Sans"), cls)  # pyright: ignore[reportArgumentType] -- stubs say bytes; PySide6 only accepts str
    try:
        apply_theme(qt_app)
        holder = QWidget()
        layout = QVBoxLayout(holder)
        widgets = [QLabel("text"), QPushButton("button"), QMenu()]
        for widget in widgets[:2]:
            layout.addWidget(widget)
        holder.show()
        for widget in widgets:
            widget.ensurePolished()
            assert QFontInfo(widget.font()).family() == UI_FONT_FAMILY, type(widget).__name__
        holder.close()
    finally:
        for cls in classes:
            QApplication.setFont(QApplication.font(), cls)  # pyright: ignore[reportArgumentType] -- as above


def test_profile_menu_opens_above_the_button(qt_app: QApplication) -> None:
    button = ProfileButton()
    home = _profile("p1", "Home", ProfileColour.ORANGE)
    button.set_profiles([home], home)
    button.show()
    button.move(100, 500)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    menu = button.profile_menu()
    assert menu.isVisible()
    assert menu.geometry().bottom() < button.mapToGlobal(button.rect().topLeft()).y()
    menu.hide()
    button.close()
    assert qt_app is not None
