"""Tests for the Qt-free profile controller and how the chat controller uses profiles."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pytest

from dom.config.settings import GenerationSettings
from dom.domain import ChatMessage, ProfileColour, ProfileSettings, Role
from dom.storage import SqliteStore
from dom.ui.controller import ChatController
from dom.ui.profiles import (
    ProfileController,
    clean_settings,
    initial_profile,
    new_profile_settings,
    next_colour,
)
from tests.fakes import FakeEngine, FakeInferenceService, RecordingProfileView, RecordingView

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

WORK = ProfileSettings(
    "Work",
    ProfileColour.BLUE,
    "Answer in one sentence.",
    GenerationSettings(temperature=0.2, max_reply_tokens=1024, thinking=False),
)


@dataclass
class App:
    store: SqliteStore
    service: FakeInferenceService
    chat_view: RecordingView
    profile_view: RecordingProfileView
    chat: ChatController
    profiles: ProfileController


@pytest.fixture
async def app(tmp_path: Path) -> AsyncIterator[App]:
    store = SqliteStore(tmp_path / "dom.sqlite3")
    service = FakeInferenceService()
    chat_view = RecordingView()
    profile_view = RecordingProfileView()
    chat = ChatController(service, store, chat_view, FakeEngine(service), initial_profile(store))
    profiles = ProfileController(store, profile_view, chat)
    await profiles.startup()
    await chat.startup()
    yield App(store, service, chat_view, profile_view, chat, profiles)
    await chat.shutdown()


async def _chat_in(app: App, model: str, text: str) -> None:
    await app.chat.select_model(model)
    await app.chat.send(text)


async def test_startup_opens_the_default_profile(app: App) -> None:
    assert [p.name for p in app.profile_view.profiles] == ["Default"]
    assert app.profile_view.current is not None
    assert app.profile_view.current.name == "Default"


def test_initial_profile_is_the_last_used(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "dom.sqlite3")
    work = store.create_profile(WORK)
    assert initial_profile(store).id == work.id  # created after Default, so used more recently
    store.mark_used("default")
    assert initial_profile(store).id == "default"
    store.close()


async def test_adding_a_profile_switches_to_it_with_an_empty_history(app: App) -> None:
    await _chat_in(app, "model-a", "hello from default")
    assert len(app.chat_view.conversations) == 1
    await app.profiles.create(WORK)
    assert app.chat.profile.name == "Work"
    assert app.chat_view.conversations == []
    assert app.chat_view.messages == []
    assert [p.name for p in app.profile_view.profiles] == ["Default", "Work"]
    assert app.profile_view.current is not None
    assert app.profile_view.current.name == "Work"


async def test_switching_back_shows_that_profiles_chats(app: App) -> None:
    await _chat_in(app, "model-a", "hello from default")
    await app.profiles.create(WORK)
    await app.chat.send("hello from work")
    await app.profiles.switch("default")
    assert [c.title for c in app.chat_view.conversations] == ["hello from default"]


async def test_each_profile_reopens_its_own_model(app: App) -> None:
    await app.chat.select_model("model-a")
    await app.profiles.create(WORK)
    await app.chat.select_model("model-b")
    await app.profiles.switch("default")
    assert app.chat.loaded_model == "model-a"
    work_id = next(p.id for p in app.profile_view.profiles if p.name == "Work")
    await app.profiles.switch(work_id)
    assert app.chat.loaded_model == "model-b"


async def test_a_new_profile_keeps_the_loaded_model(app: App) -> None:
    await app.chat.select_model("model-a")
    await app.profiles.create(WORK)
    assert app.chat.loaded_model == "model-a"


async def test_the_profiles_model_is_loaded_at_startup(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "dom.sqlite3")
    store.set_last_model("default", "model-b")
    service = FakeInferenceService()
    chat = ChatController(
        service, store, RecordingView(), FakeEngine(service), initial_profile(store)
    )
    await chat.startup()
    assert chat.loaded_model == "model-b"
    await chat.shutdown()


async def test_missing_last_model_is_ignored(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "dom.sqlite3")
    store.set_last_model("default", "deleted-model")
    service = FakeInferenceService()
    view = RecordingView()
    chat = ChatController(service, store, view, FakeEngine(service), initial_profile(store))
    await chat.startup()
    assert chat.loaded_model is None
    assert view.errors == []
    await chat.shutdown()


async def test_system_prompt_and_settings_are_sent(app: App) -> None:
    await app.profiles.create(WORK)
    await _chat_in(app, "model-a", "hi")
    assert [(m.role, m.content) for m in app.service.received[-1]] == [
        (Role.SYSTEM, "Answer in one sentence."),
        (Role.USER, "hi"),
    ]
    assert app.service.received_settings[-1] == WORK.generation
    # The system prompt is sent, not saved as part of the chat.
    assert [m.role for m in app.chat_view.messages] == [Role.USER, Role.ASSISTANT]


async def test_editing_the_current_profile_applies_to_the_next_reply(app: App) -> None:
    await _chat_in(app, "model-a", "first")
    edited = replace(WORK, name="Default renamed")
    await app.profiles.update("default", edited)
    assert len(app.chat_view.messages) == 2  # the open chat stays  # noqa: PLR2004
    await app.chat.send("second")
    assert app.service.received[-1][0] == ChatMessage(Role.SYSTEM, "Answer in one sentence.")
    assert app.service.received_settings[-1] == WORK.generation
    assert app.profile_view.current is not None
    assert app.profile_view.current.name == "Default renamed"


async def test_deleting_the_current_profile_opens_another(app: App) -> None:
    await app.profiles.create(WORK)
    await app.chat.select_model("model-a")
    await app.chat.send("work chat")
    work_id = app.chat.profile.id
    await app.profiles.delete(work_id)
    assert app.chat.profile.id == "default"
    assert [p.id for p in app.profile_view.profiles] == ["default"]
    assert app.store.list_conversations(work_id) == []


async def test_the_last_profile_cannot_be_deleted(app: App) -> None:
    await app.profiles.delete("default")
    assert app.profile_view.errors == ["The last profile cannot be deleted."]
    assert [p.id for p in app.store.list_profiles()] == ["default"]


async def test_switching_profiles_stops_a_reply(app: App) -> None:
    app.service.endless_stream = True
    await app.chat.select_model("model-a")
    await app.profiles.create(WORK)
    work_id = app.chat.profile.id
    await app.profiles.switch("default")
    task = asyncio.ensure_future(app.chat.send("go on forever"))
    await app.service.endless_started.wait()
    await app.profiles.switch(work_id)
    await task
    assert not app.chat.is_streaming
    assert app.chat_view.conversations == []


def test_clean_settings_and_defaults() -> None:
    cleaned = clean_settings(ProfileSettings("  Sam   the  Man ", ProfileColour.RED, "  hi \n"))
    assert cleaned.name == "Sam the Man"
    assert cleaned.system_prompt == "hi"
    assert clean_settings(ProfileSettings("   ", ProfileColour.RED)).name == "Profile"
    assert clean_settings(ProfileSettings("x" * 99, ProfileColour.RED)).name == "x" * 40


def test_new_profiles_get_unused_names_and_colours(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "dom.sqlite3")
    profiles = store.list_profiles()
    first = new_profile_settings(profiles)
    assert first.name == "Profile 2"
    assert first.colour != ProfileColour.ORANGE  # Default's colour
    store.create_profile(first)
    second = new_profile_settings(store.list_profiles())
    assert second.name == "Profile 3"
    assert second.colour not in (ProfileColour.ORANGE, first.colour)
    store.close()
    assert next_colour([]) == ProfileColour.ORANGE
