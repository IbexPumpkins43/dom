"""Tests for the SQLite conversation repository."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from dom.domain import Role
from dom.storage import (
    ConversationNotFoundError,
    SchemaVersionError,
    SqliteConversationRepository,
)
from dom.storage.schema import SCHEMA_VERSION

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
    repo = SqliteConversationRepository(db_path, clock=FakeClock())
    first = repo.create_conversation("First chat")
    repo.add_message(first.id, Role.USER, "Hello")
    repo.add_message(first.id, Role.ASSISTANT, "Hi! **How** can I help?", model_id="m1")
    second = repo.create_conversation("Second chat")
    repo.add_message(second.id, Role.USER, "Line one\nLine two")
    repo.add_message(
        second.id, Role.ASSISTANT, "Answer", reasoning="Let me think...", model_id="m2"
    )
    repo.add_message(second.id, Role.USER, "Thanks")
    before_convs = repo.list_conversations()
    before_first = repo.get_messages(first.id)
    before_second = repo.get_messages(second.id)
    repo.close()

    reopened = SqliteConversationRepository(db_path)
    try:
        assert reopened.list_conversations() == before_convs
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
    repo = SqliteConversationRepository(db_path, clock=FakeClock())
    older = repo.create_conversation("older")
    newer = repo.create_conversation("newer")
    assert [c.id for c in repo.list_conversations()] == [newer.id, older.id]

    repo.add_message(older.id, Role.USER, "bump")
    assert [c.id for c in repo.list_conversations()] == [older.id, newer.id]
    repo.close()


def test_same_timestamp_falls_back_to_creation_order(db_path: Path) -> None:
    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    repo = SqliteConversationRepository(db_path, clock=lambda: fixed)
    a = repo.create_conversation("a")
    b = repo.create_conversation("b")
    assert [c.id for c in repo.list_conversations()] == [b.id, a.id]
    repo.close()


def test_add_message_to_missing_conversation_raises(db_path: Path) -> None:
    repo = SqliteConversationRepository(db_path)
    with pytest.raises(ConversationNotFoundError):
        repo.add_message("does-not-exist", Role.USER, "hi")
    repo.close()


def test_empty_database_lists_nothing(db_path: Path) -> None:
    repo = SqliteConversationRepository(db_path)
    assert repo.list_conversations() == []
    assert repo.get_messages("nope") == []
    repo.close()


def test_schema_version_recorded(db_path: Path) -> None:
    SqliteConversationRepository(db_path).close()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT version FROM schema_version").fetchall()
    assert rows == [(SCHEMA_VERSION,)]


def test_newer_schema_is_rejected(db_path: Path) -> None:
    SqliteConversationRepository(db_path).close()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION + 1,))
    conn.commit()
    conn.close()
    with pytest.raises(SchemaVersionError):
        SqliteConversationRepository(db_path)


def test_delete_conversation_removes_it_and_its_messages(db_path: Path) -> None:
    repo = SqliteConversationRepository(db_path, clock=FakeClock())
    keep = repo.create_conversation("keep")
    repo.add_message(keep.id, Role.USER, "stay")
    gone = repo.create_conversation("gone")
    repo.add_message(gone.id, Role.USER, "bye")
    repo.add_message(gone.id, Role.ASSISTANT, "bye!")
    repo.delete_conversation(gone.id)
    assert [c.id for c in repo.list_conversations()] == [keep.id]
    assert repo.get_messages(gone.id) == []
    assert [m.content for m in repo.get_messages(keep.id)] == ["stay"]
    repo.close()
    with sqlite3.connect(db_path) as conn:
        orphans = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (gone.id,)
        ).fetchone()
    assert orphans == (0,)


def test_deleting_a_missing_conversation_raises(db_path: Path) -> None:
    repo = SqliteConversationRepository(db_path)
    with pytest.raises(ConversationNotFoundError):
        repo.delete_conversation("nope")
    repo.close()
