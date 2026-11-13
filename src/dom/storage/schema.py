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
