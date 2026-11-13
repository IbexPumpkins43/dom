"""SQLite implementation of ``ConversationRepository`` and ``ProfileRepository``."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from dom.config.settings import GenerationSettings
from dom.domain import Conversation, Message, Profile, ProfileColour, ProfileSettings, Role
from dom.storage.errors import ConversationNotFoundError, ProfileNotFoundError, StorageError
from dom.storage.schema import migrate

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Fixed-width UTC timestamps, so text ordering in SQL matches time ordering.
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

_PROFILE_COLUMNS = (
    "id, name, colour, created_at, last_used_at, system_prompt, last_model, temperature, "
    "max_reply_tokens, thinking"
)

type Clock = Callable[[], datetime]
type SqlValue = str | int | float | None


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


def _optional_int(row: tuple[object, ...], index: int) -> int | None:
    value = row[index]
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    raise StorageError(f"Unexpected value in database column {index}: {value!r}")


def _optional_float(row: tuple[object, ...], index: int) -> float | None:
    value = row[index]
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise StorageError(f"Unexpected value in database column {index}: {value!r}")


def _colour(value: str) -> ProfileColour:
    try:
        return ProfileColour(value)
    except ValueError:
        return ProfileColour.ORANGE  # written by a newer dom with more colours


def _profile(row: tuple[object, ...]) -> Profile:
    settings = ProfileSettings(
        name=_text(row, 1),
        colour=_colour(_text(row, 2)),
        system_prompt=_text(row, 5),
        generation=GenerationSettings(
            temperature=_optional_float(row, 7),
            max_reply_tokens=_optional_int(row, 8),
            thinking=bool(_optional_int(row, 9)),
        ),
    )
    return Profile(
        id=_text(row, 0),
        settings=settings,
        created_at=_parse_ts(_text(row, 3)),
        last_used_at=_parse_ts(_text(row, 4)),
        last_model=_optional_text(row, 6),
    )


def _settings_values(settings: ProfileSettings) -> tuple[SqlValue, ...]:
    """name, colour, system_prompt, temperature, max_reply_tokens, thinking."""
    generation = settings.generation
    return (
        settings.name,
        settings.colour.value,
        settings.system_prompt,
        generation.temperature,
        generation.max_reply_tokens,
        int(generation.thinking),
    )


class SqliteStore:
    """Stores profiles, conversations and messages in one SQLite database file.

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

    # ---- profiles ------------------------------------------------------------------------

    def list_profiles(self) -> list[Profile]:
        """Return every profile in the order they were added."""
        with self._lock:
            rows = self._query(f"SELECT {_PROFILE_COLUMNS} FROM profiles ORDER BY rowid")
        return [_profile(r) for r in rows]

    def create_profile(self, settings: ProfileSettings) -> Profile:
        """Create and return a new profile (with no chats)."""
        now = _format_ts(self._clock())
        profile_id = str(uuid.uuid4())
        with self._lock, self._transaction():
            self._conn.execute(
                "INSERT INTO profiles (id, created_at, last_used_at, name, colour, system_prompt, "
                "temperature, max_reply_tokens, thinking) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (profile_id, now, now, *_settings_values(settings)),
            )
        return self._get_profile(profile_id)

    def update_profile(self, profile_id: str, settings: ProfileSettings) -> Profile:
        """Replace a profile's settings and return the updated profile.

        Raises:
            ProfileNotFoundError: if there is no such profile.
        """
        with self._lock, self._transaction():
            updated = self._conn.execute(
                "UPDATE profiles SET name = ?, colour = ?, system_prompt = ?, temperature = ?, "
                "max_reply_tokens = ?, thinking = ? WHERE id = ?",
                (*_settings_values(settings), profile_id),
            )
            self._require_row(updated.rowcount, profile_id)
        return self._get_profile(profile_id)

    def set_last_model(self, profile_id: str, model_id: str | None) -> None:
        """Remember the model a profile used last."""
        with self._lock, self._transaction():
            updated = self._conn.execute(
                "UPDATE profiles SET last_model = ? WHERE id = ?", (model_id, profile_id)
            )
            self._require_row(updated.rowcount, profile_id)

    def mark_used(self, profile_id: str) -> Profile:
        """Record that a profile was opened now (dom starts in the last-used profile)."""
        with self._lock, self._transaction():
            updated = self._conn.execute(
                "UPDATE profiles SET last_used_at = ? WHERE id = ?",
                (_format_ts(self._clock()), profile_id),
            )
            self._require_row(updated.rowcount, profile_id)
        return self._get_profile(profile_id)

    def delete_profile(self, profile_id: str) -> None:
        """Delete a profile with all of its chats.

        Raises:
            ProfileNotFoundError: if there is no such profile.
        """
        with self._lock, self._transaction():
            # Chats and their messages go with it (ON DELETE CASCADE, twice).
            deleted = self._conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
            self._require_row(deleted.rowcount, profile_id)

    def _get_profile(self, profile_id: str) -> Profile:
        with self._lock:
            rows = self._query(
                f"SELECT {_PROFILE_COLUMNS} FROM profiles WHERE id = ?",
                (profile_id,),
            )
        if not rows:
            raise ProfileNotFoundError(f"No profile with id {profile_id}")
        return _profile(rows[0])

    @staticmethod
    def _require_row(rowcount: int, profile_id: str) -> None:
        if rowcount == 0:
            raise ProfileNotFoundError(f"No profile with id {profile_id}")

    # ---- conversations -------------------------------------------------------------------

    def create_conversation(self, profile_id: str, title: str) -> Conversation:
        """Create and return a new, empty conversation in a profile."""
        now = self._clock()
        conversation = Conversation(
            id=str(uuid.uuid4()), title=title, created_at=now, updated_at=now
        )
        with self._lock, self._transaction():
            exists = self._conn.execute(
                "SELECT 1 FROM profiles WHERE id = ?", (profile_id,)
            ).fetchone()
            if exists is None:
                raise ProfileNotFoundError(f"No profile with id {profile_id}")
            self._conn.execute(
                "INSERT INTO conversations (id, profile_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation.id, profile_id, title, _format_ts(now), _format_ts(now)),
            )
        return conversation

    def list_conversations(self, profile_id: str) -> list[Conversation]:
        """Return a profile's conversations, most recently updated first."""
        with self._lock:
            rows: list[tuple[object, ...]] = self._query(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "WHERE profile_id = ? ORDER BY updated_at DESC, rowid DESC",
                (profile_id,),
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

    def _query(self, sql: str, params: tuple[SqlValue, ...] = ()) -> list[tuple[object, ...]]:
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
