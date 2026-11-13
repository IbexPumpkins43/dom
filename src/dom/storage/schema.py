"""SQLite schema and forward-only migrations."""

from __future__ import annotations

import logging
import sqlite3

from dom.storage.errors import SchemaVersionError, StorageError

logger = logging.getLogger(__name__)

# MIGRATIONS[i] upgrades the schema from version i to version i + 1. Never edit a released
# entry; append a new one instead.
MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE conversations (
        id          TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    );
    CREATE INDEX ix_conversations_updated ON conversations (updated_at DESC);
    CREATE TABLE messages (
        id               TEXT PRIMARY KEY,
        conversation_id  TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
        seq              INTEGER NOT NULL,
        role             TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
        content          TEXT NOT NULL,
        reasoning        TEXT,
        model_id         TEXT,
        created_at       TEXT NOT NULL,
        UNIQUE (conversation_id, seq)
    );
    """,
    # 2: profiles. Existing chats move into a "Default" profile. (SQLite cannot add a NOT NULL
    # foreign-key column with a default, so profile_id is nullable; the code always sets it.)
    """
    CREATE TABLE profiles (
        id                TEXT PRIMARY KEY,
        name              TEXT NOT NULL,
        colour            TEXT NOT NULL,
        created_at        TEXT NOT NULL,
        last_used_at      TEXT NOT NULL,
        system_prompt     TEXT NOT NULL DEFAULT '',
        last_model        TEXT,
        temperature       REAL,
        max_reply_tokens  INTEGER,
        thinking          INTEGER NOT NULL DEFAULT 1
    );
    INSERT INTO profiles (id, name, colour, created_at, last_used_at, max_reply_tokens)
    VALUES (
        'default', 'Default', 'orange',
        strftime('%Y-%m-%dT%H:%M:%S', 'now') || '.000000Z',
        strftime('%Y-%m-%dT%H:%M:%S', 'now') || '.000000Z',
        8192
    );
    ALTER TABLE conversations
        ADD COLUMN profile_id TEXT REFERENCES profiles (id) ON DELETE CASCADE;
    UPDATE conversations SET profile_id = 'default';
    CREATE INDEX ix_conversations_profile ON conversations (profile_id, updated_at DESC);
    """,
)

SCHEMA_VERSION = len(MIGRATIONS)


def current_version(conn: sqlite3.Connection) -> int:
    """Return the schema version stored in the database (0 for a new database)."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row: tuple[object, ...] | None = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        return 0
    version = row[0]
    if not isinstance(version, int):
        raise SchemaVersionError(f"Corrupt schema_version value: {version!r}")
    return version


def migrate(conn: sqlite3.Connection) -> None:
    """Bring the database up to ``SCHEMA_VERSION``."""
    version = current_version(conn)
    conn.commit()
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"The database uses schema version {version}, but this version of dom only "
            f"understands up to {SCHEMA_VERSION}. Please update dom."
        )
    for target in range(version + 1, SCHEMA_VERSION + 1):
        logger.info("Migrating database schema to version %d", target)
        try:
            conn.executescript(
                "BEGIN;\n"
                f"{MIGRATIONS[target - 1]}\n"
                "DELETE FROM schema_version;\n"
                f"INSERT INTO schema_version (version) VALUES ({target});\n"
                "COMMIT;"
            )
        except sqlite3.Error as exc:
            conn.rollback()
            raise StorageError(f"Database migration to version {target} failed: {exc}") from exc
