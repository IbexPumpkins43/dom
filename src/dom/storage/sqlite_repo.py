"""SQLite implementation of ``ConversationRepository``."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from dom.domain import Conversation, Message, Role
from dom.storage.errors import ConversationNotFoundError, StorageError
from dom.storage.schema import migrate

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Fixed-width UTC timestamps, so text ordering in SQL matches time ordering.
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

type Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_ts(value: datetime) -> str:
    return value.astimezone(UTC).strftime(_TIMESTAMP_FORMAT)


def _parse_ts(value: str) -> datetime:
    return datetime.strptime(value, _TIMESTAMP_FORMAT).replace(tzinfo=UTC)


def _text(row: tuple[object, ...], index: int) -> str:
    value = row[index]
    if not isinstance(value, str):
        raise StorageError(f"Unexpected value in database column {index}: {value!r}")
    return value


def _optional_text(row: tuple[object, ...], index: int) -> str | None:
    return None if row[index] is None else _text(row, index)


class SqliteConversationRepository:
    """Stores conversations in a SQLite database file.

    Safe to call from any one thread at a time (calls are serialised with a lock), which is how
    the UI uses it through ``asyncio.to_thread``.
    """

    def __init__(self, db_path: Path, clock: Clock = _utc_now) -> None:
        """Open (creating if needed) and migrate the database at ``db_path``."""
        self._clock = clock
        self._lock = threading.Lock()
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            migrate(self._conn)
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(
                f"Could not open the conversation database at {db_path}: {exc}"
            ) from exc

    def create_conversation(self, title: str) -> Conversation:
        """Create and return a new, empty conversation."""
        now = self._clock()
        conversation = Conversation(
            id=str(uuid.uuid4()), title=title, created_at=now, updated_at=now
        )
        with self._lock, self._transaction():
            self._conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation.id, title, _format_ts(now), _format_ts(now)),
            )
        return conversation

    def list_conversations(self) -> list[Conversation]:
        """Return all conversations, most recently updated first."""
        with self._lock:
            rows: list[tuple[object, ...]] = self._query(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC, rowid DESC"
            )
        return [
            Conversation(
                id=_text(r, 0),
                title=_text(r, 1),
                created_at=_parse_ts(_text(r, 2)),
                updated_at=_parse_ts(_text(r, 3)),
            )
            for r in rows
        ]

    def get_messages(self, conversation_id: str) -> list[Message]:
        """Return a conversation's messages in the order they were added."""
        with self._lock:
            rows = self._query(
                "SELECT id, role, content, reasoning, model_id, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY seq",
                (conversation_id,),
            )
        return [
            Message(
                id=_text(r, 0),
                conversation_id=conversation_id,
                role=Role(_text(r, 1)),
                content=_text(r, 2),
                reasoning=_optional_text(r, 3),
                model_id=_optional_text(r, 4),
                created_at=_parse_ts(_text(r, 5)),
            )
            for r in rows
        ]

    def add_message(
        self,
        conversation_id: str,
        role: Role,
        content: str,
        *,
        reasoning: str | None = None,
        model_id: str | None = None,
    ) -> Message:
        """Append a message to a conversation and mark the conversation as updated."""
        now = self._clock()
        message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=now,
            reasoning=reasoning,
            model_id=model_id,
        )
        with self._lock, self._transaction():
            updated = self._conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_format_ts(now), conversation_id),
            )
            if updated.rowcount == 0:
                raise ConversationNotFoundError(f"No conversation with id {conversation_id}")
            self._conn.execute(
                "INSERT INTO messages "
                "(id, conversation_id, seq, role, content, reasoning, model_id, created_at) "
                "VALUES (?, ?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM messages "
                "WHERE conversation_id = ?), ?, ?, ?, ?, ?)",
                (
                    message.id,
                    conversation_id,
                    conversation_id,
                    role.value,
                    content,
                    reasoning,
                    model_id,
                    _format_ts(now),
                ),
            )
        return message

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation; its messages go with it (``ON DELETE CASCADE``).

        Raises:
            ConversationNotFoundError: if there is no such conversation.
        """
        with self._lock, self._transaction():
            deleted = self._conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
            if deleted.rowcount == 0:
                raise ConversationNotFoundError(f"No conversation with id {conversation_id}")

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def _query(self, sql: str, params: tuple[str, ...] = ()) -> list[tuple[object, ...]]:
        try:
            return self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Database query failed: {exc}") from exc

    def _transaction(self) -> _Transaction:
        return _Transaction(self._conn)


class _Transaction:
    """Commit on success, roll back on error, and wrap SQLite errors in ``StorageError``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> bool:
        if exc_type is None:
            try:
                self._conn.commit()
            except sqlite3.Error as commit_exc:
                self._conn.rollback()
                raise StorageError(f"Could not save to the database: {commit_exc}") from commit_exc
            return False
        self._conn.rollback()
        if isinstance(exc, sqlite3.Error):
            raise StorageError(f"Could not save to the database: {exc}") from exc
        return False
