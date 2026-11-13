"""Tests for the SQLite conversation repository."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dom.config.settings import GenerationSettings
from dom.domain import ProfileColour, ProfileSettings, Role
from dom.storage import (
    ConversationNotFoundError,
    ProfileNotFoundError,
    SchemaVersionError,
    SqliteStore,
)
from dom.storage.schema import MIGRATIONS, SCHEMA_VERSION

DEFAULT = "default"  # the profile migration 2 creates

if TYPE_CHECKING:
    from pathlib import Path


class FakeClock:
    """A clock that advances one second every time it is read."""

    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "dom.sqlite3"


def test_round_trip_survives_close_and_reopen(db_path: Path) -> None:
    repo = SqliteStore(db_path, clock=FakeClock())
    first = repo.create_conversation(DEFAULT, "First chat")
    repo.add_message(first.id, Role.USER, "Hello")
    repo.add_message(first.id, Role.ASSISTANT, "Hi! **How** can I help?", model_id="m1")
    second = repo.create_conversation(DEFAULT, "Second chat")
    repo.add_message(second.id, Role.USER, "Line one\nLine two")
    repo.add_message(
        second.id, Role.ASSISTANT, "Answer", reasoning="Let me think...", model_id="m2"
    )
    repo.add_message(second.id, Role.USER, "Thanks")
    before_convs = repo.list_conversations(DEFAULT)
    before_first = repo.get_messages(first.id)
    before_second = repo.get_messages(second.id)
    repo.close()

    reopened = SqliteStore(db_path)
    try:
        assert reopened.list_conversations(DEFAULT) == before_convs
        assert [c.id for c in before_convs] == [second.id, first.id]
        assert reopened.get_messages(first.id) == before_first
        assert reopened.get_messages(second.id) == before_second
        assert [(m.role, m.content) for m in reopened.get_messages(second.id)] == [
            (Role.USER, "Line one\nLine two"),
            (Role.ASSISTANT, "Answer"),
            (Role.USER, "Thanks"),
        ]
        answer = reopened.get_messages(second.id)[1]
        assert answer.reasoning == "Let me think..."
        assert answer.model_id == "m2"
    finally:
        reopened.close()


def test_conversations_sorted_by_last_activity(db_path: Path) -> None:
    repo = SqliteStore(db_path, clock=FakeClock())
    older = repo.create_conversation(DEFAULT, "older")
    newer = repo.create_conversation(DEFAULT, "newer")
    assert [c.id for c in repo.list_conversations(DEFAULT)] == [newer.id, older.id]

    repo.add_message(older.id, Role.USER, "bump")
    assert [c.id for c in repo.list_conversations(DEFAULT)] == [older.id, newer.id]
    repo.close()


def test_same_timestamp_falls_back_to_creation_order(db_path: Path) -> None:
    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    repo = SqliteStore(db_path, clock=lambda: fixed)
    a = repo.create_conversation(DEFAULT, "a")
    b = repo.create_conversation(DEFAULT, "b")
    assert [c.id for c in repo.list_conversations(DEFAULT)] == [b.id, a.id]
    repo.close()


def test_add_message_to_missing_conversation_raises(db_path: Path) -> None:
    repo = SqliteStore(db_path)
    with pytest.raises(ConversationNotFoundError):
        repo.add_message("does-not-exist", Role.USER, "hi")
    repo.close()


def test_empty_database_lists_nothing(db_path: Path) -> None:
    repo = SqliteStore(db_path)
    assert repo.list_conversations(DEFAULT) == []
    assert repo.get_messages("nope") == []
    repo.close()


def test_schema_version_recorded(db_path: Path) -> None:
    SqliteStore(db_path).close()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT version FROM schema_version").fetchall()
    assert rows == [(SCHEMA_VERSION,)]


def test_newer_schema_is_rejected(db_path: Path) -> None:
    SqliteStore(db_path).close()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION + 1,))
    conn.commit()
    conn.close()
    with pytest.raises(SchemaVersionError):
        SqliteStore(db_path)


def test_delete_conversation_removes_it_and_its_messages(db_path: Path) -> None:
    repo = SqliteStore(db_path, clock=FakeClock())
    keep = repo.create_conversation(DEFAULT, "keep")
    repo.add_message(keep.id, Role.USER, "stay")
    gone = repo.create_conversation(DEFAULT, "gone")
    repo.add_message(gone.id, Role.USER, "bye")
    repo.add_message(gone.id, Role.ASSISTANT, "bye!")
    repo.delete_conversation(gone.id)
    assert [c.id for c in repo.list_conversations(DEFAULT)] == [keep.id]
    assert repo.get_messages(gone.id) == []
    assert [m.content for m in repo.get_messages(keep.id)] == ["stay"]
    repo.close()
    with sqlite3.connect(db_path) as conn:
        orphans = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (gone.id,)
        ).fetchone()
    assert orphans == (0,)


def test_deleting_a_missing_conversation_raises(db_path: Path) -> None:
    repo = SqliteStore(db_path)
    with pytest.raises(ConversationNotFoundError):
        repo.delete_conversation("nope")
    repo.close()


# ---- profiles ----------------------------------------------------------------------------


def test_new_database_has_a_default_profile(db_path: Path) -> None:
    repo = SqliteStore(db_path)
    [profile] = repo.list_profiles()
    assert profile.id == DEFAULT
    assert profile.name == "Default"
    assert profile.settings.colour == ProfileColour.ORANGE
    assert profile.settings.generation == GenerationSettings()
    assert profile.last_model is None
    repo.close()


def test_version_1_chats_move_into_the_default_profile(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        f"BEGIN; {MIGRATIONS[0]}"
        "CREATE TABLE schema_version (version INTEGER NOT NULL);"
        "INSERT INTO schema_version VALUES (1);"
        "INSERT INTO conversations VALUES "
        "('c1', 'Old chat', '2025-01-01T00:00:00.000000Z', '2025-01-01T00:00:00.000000Z');"
        "INSERT INTO messages VALUES "
        "('m1', 'c1', 1, 'user', 'hello', NULL, NULL, '2025-01-01T00:00:00.000000Z');"
        "COMMIT;"
    )
    conn.close()
    repo = SqliteStore(db_path)
    assert [c.title for c in repo.list_conversations(DEFAULT)] == ["Old chat"]
    assert [m.content for m in repo.get_messages("c1")] == ["hello"]
    assert [p.name for p in repo.list_profiles()] == ["Default"]
    repo.close()


def test_profile_round_trip(db_path: Path) -> None:
    repo = SqliteStore(db_path, clock=FakeClock())
    settings = ProfileSettings(
        "Work",
        ProfileColour.BLUE,
        "Be brief.",
        GenerationSettings(temperature=0.2, max_reply_tokens=None, thinking=False),
    )
    work = repo.create_profile(settings)
    assert work.settings == settings
    renamed = replace(settings, name="Office", generation=GenerationSettings())
    assert repo.update_profile(work.id, renamed).settings == renamed
    repo.set_last_model(work.id, "model-a")
    used = repo.mark_used(work.id)
    assert used.last_model == "model-a"
    assert used.last_used_at > work.last_used_at
    repo.close()

    reopened = SqliteStore(db_path)
    assert [p.name for p in reopened.list_profiles()] == ["Default", "Office"]
    assert reopened.list_profiles()[1] == used
    reopened.close()


def test_profiles_keep_their_chats_apart(db_path: Path) -> None:
    repo = SqliteStore(db_path, clock=FakeClock())
    work = repo.create_profile(ProfileSettings("Work", ProfileColour.BLUE))
    home_chat = repo.create_conversation(DEFAULT, "home")
    work_chat = repo.create_conversation(work.id, "work")
    assert [c.id for c in repo.list_conversations(DEFAULT)] == [home_chat.id]
    assert [c.id for c in repo.list_conversations(work.id)] == [work_chat.id]
    repo.close()


def test_deleting_a_profile_deletes_its_chats(db_path: Path) -> None:
    repo = SqliteStore(db_path, clock=FakeClock())
    work = repo.create_profile(ProfileSettings("Work", ProfileColour.BLUE))
    chat = repo.create_conversation(work.id, "work")
    repo.add_message(chat.id, Role.USER, "secret")
    kept = repo.create_conversation(DEFAULT, "home")
    repo.delete_profile(work.id)
    assert [p.id for p in repo.list_profiles()] == [DEFAULT]
    assert repo.get_messages(chat.id) == []
    assert [c.id for c in repo.list_conversations(DEFAULT)] == [kept.id]
    repo.close()
    with sqlite3.connect(db_path) as conn:
        left = conn.execute("SELECT COUNT(*) FROM conversations WHERE id = ?", (chat.id,))
        assert left.fetchone() == (0,)


def test_missing_profiles_raise(db_path: Path) -> None:
    repo = SqliteStore(db_path)
    settings = ProfileSettings("x", ProfileColour.RED)
    with pytest.raises(ProfileNotFoundError):
        repo.update_profile("nope", settings)
    with pytest.raises(ProfileNotFoundError):
        repo.mark_used("nope")
    with pytest.raises(ProfileNotFoundError):
        repo.delete_profile("nope")
    with pytest.raises(ProfileNotFoundError):
        repo.create_conversation("nope", "chat")
    repo.close()
