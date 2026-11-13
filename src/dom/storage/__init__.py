"""Storage layer: the conversation repository protocol and its SQLite implementation."""

from dom.storage.errors import ConversationNotFoundError, SchemaVersionError, StorageError
from dom.storage.protocols import ConversationRepository
from dom.storage.sqlite_repo import SqliteConversationRepository

__all__ = [
    "ConversationNotFoundError",
    "ConversationRepository",
    "SchemaVersionError",
    "SqliteConversationRepository",
    "StorageError",
]
