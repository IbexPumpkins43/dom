"""Storage-layer exceptions."""


class StorageError(Exception):
    """Base class for storage failures."""


class SchemaVersionError(StorageError):
    """The database was written by a newer version of dom than this one."""


class ConversationNotFoundError(StorageError):
    """A conversation id does not exist."""
