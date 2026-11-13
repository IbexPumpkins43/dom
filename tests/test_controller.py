"""Tests for the Qt-free chat controller, using fakes for the service and the view."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from dom.domain import ModelStatus, Role
from dom.storage import SqliteConversationRepository
from dom.ui.controller import ChatController, make_title
from dom.ui.protocols import ServerState
from tests.fakes import FakeInferenceService, RecordingView

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@pytest.fixture
def service() -> FakeInferenceService:
    return FakeInferenceService()


@pytest.fixture
def view() -> RecordingView:
    return RecordingView()


@pytest.fixture
async def controller(
    tmp_path: Path, service: FakeInferenceService, view: RecordingView
) -> AsyncIterator[ChatController]:
    repo = SqliteConversationRepository(tmp_path / "dom.sqlite3")
    ctrl = ChatController(service, repo, view, models_dir_hint="/models")
    yield ctrl
    await ctrl.shutdown()


async def _ready(controller: ChatController, model: str = "model-a") -> None:
    await controller.startup()
    await controller.select_model(model)


async def test_startup_shows_models_and_asks_to_pick_one(
    controller: ChatController, view: RecordingView
) -> None:
    await controller.startup()
    assert view.server_states[-1] == (ServerState.READY, None)
    assert [m.id for m in view.models] == ["model-a", "model-b"]
    assert view.notice is not None
    assert "Pick a model" in view.notice
    assert not view.can_send


async def test_failed_start_is_reported_in_plain_english(
    controller: ChatController, service: FakeInferenceService, view: RecordingView
) -> None:
    service.start_error = "dom needs the llama-server program"
    await controller.startup()
    assert view.server_states[-1] == (ServerState.FAILED, "dom needs the llama-server program")
    assert not view.can_send


async def test_no_models_shows_where_to_put_them(
    controller: ChatController, service: FakeInferenceService, view: RecordingView
) -> None:
    service.model_ids = []
    await controller.startup()
    assert view.notice is not None
    assert "/models" in view.notice


async def test_selecting_a_model_enables_input(
    controller: ChatController, view: RecordingView
) -> None:
    await _ready(controller)
    assert {m.id: m.status for m in view.models} == {
        "model-a": ModelStatus.LOADED,
        "model-b": ModelStatus.UNLOADED,
    }
    assert view.selected_model == "model-a"
    assert view.can_send
    assert view.notice is None


async def test_failed_load_shows_error_and_keeps_input_disabled(
    controller: ChatController, service: FakeInferenceService, view: RecordingView
) -> None:
    service.load_error = "out of memory"
    await _ready(controller)
    assert any("out of memory" in e for e in view.errors)
    assert not view.can_send


async def test_send_streams_and_saves_both_messages(
    controller: ChatController, view: RecordingView, tmp_path: Path
) -> None:
    await _ready(controller)
    await controller.send("  Hi there\nsecond line ")
    assert view.reply == ("Hello there!", "Thinking hard.")
    assert not view.reply_active
    assert [(m.role, m.content) for m in view.messages] == [
        (Role.USER, "Hi there\nsecond line"),
        (Role.ASSISTANT, "Hello there!"),
    ]
    assert view.messages[1].reasoning == "Thinking hard."
    assert view.messages[1].model_id == "model-a"
    assert [c.title for c in view.conversations] == ["Hi there"]
    assert view.current_conversation == view.conversations[0].id
    assert view.can_send
    assert not view.streaming

    reopened = SqliteConversationRepository(tmp_path / "dom.sqlite3")
    saved = reopened.get_messages(view.conversations[0].id)
    reopened.close()
    assert [m.content for m in saved] == ["Hi there\nsecond line", "Hello there!"]


async def test_history_is_sent_with_each_message(
    controller: ChatController, service: FakeInferenceService
) -> None:
    await _ready(controller)
    await controller.send("one")
    await controller.send("two")
    assert [(m.role, m.content) for m in service.received[-1]] == [
        (Role.USER, "one"),
        (Role.ASSISTANT, "Hello there!"),
        (Role.USER, "two"),
    ]


async def test_cannot_send_without_a_loaded_model(
    controller: ChatController, service: FakeInferenceService, view: RecordingView
) -> None:
    await controller.startup()
    await controller.send("hello?")
    assert service.received == []
    assert view.conversations == []


async def test_stop_cancels_the_stream_and_keeps_partial_reply(
    controller: ChatController, service: FakeInferenceService, view: RecordingView
) -> None:
    service.endless_stream = True
    await _ready(controller)
    send = asyncio.create_task(controller.send("tell me a long story"))
    await service.endless_started.wait()
    assert view.streaming
    assert not view.can_send
    await controller.stop_reply()
    await send
    assert service.stream_closed
    assert not view.streaming
    assert view.can_send
    assistant = view.messages[-1]
    assert assistant.role == Role.ASSISTANT
    assert assistant.content.startswith("Hello there!")
    assert assistant.reasoning == "Thinking hard."


async def test_stream_error_is_shown_and_partial_reply_saved(
    controller: ChatController, service: FakeInferenceService, view: RecordingView
) -> None:
    service.stream_error_after = 3
    await _ready(controller)
    await controller.send("hi")
    assert "The connection dropped." in view.errors
    assert view.messages[-1].content == "Hello"
    assert view.can_send


async def test_switching_conversations(controller: ChatController, view: RecordingView) -> None:
    await _ready(controller)
    await controller.send("first chat")
    first_id = view.current_conversation
    await controller.new_chat()
    assert view.messages == []
    assert view.current_conversation is None
    await controller.send("second chat")
    assert [c.title for c in view.conversations] == ["second chat", "first chat"]
    assert first_id is not None
    await controller.open_conversation(first_id)
    assert [m.content for m in view.messages] == ["first chat", "Hello there!"]
    assert view.current_conversation == first_id


async def test_crash_is_reported_and_restart_reloads_the_model(
    controller: ChatController, service: FakeInferenceService, view: RecordingView
) -> None:
    await _ready(controller, "model-b")
    service.crash(139)
    await asyncio.sleep(0.01)
    state, message = view.server_states[-1]
    assert state == ServerState.CRASHED
    assert message is not None
    assert "139" in message
    assert not view.can_send

    await controller.restart_server()
    assert service.starts == 2  # noqa: PLR2004
    assert view.server_states[-1] == (ServerState.READY, None)
    assert {m.id: m.status for m in view.models}["model-b"] == ModelStatus.LOADED
    assert view.can_send


async def test_shutdown_stops_server(
    controller: ChatController, service: FakeInferenceService
) -> None:
    await _ready(controller)
    await controller.shutdown()
    assert service.stopped


def test_make_title() -> None:
    assert make_title("\n  Hello world  \nmore") == "Hello world"
    long = "x" * 100
    assert len(make_title(long)) == 60  # noqa: PLR2004
    assert make_title(long).endswith("…")
    assert make_title("   ") == "New chat"
