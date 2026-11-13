"""Domain objects shared by every layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


class Role(StrEnum):
    """Who wrote a chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ModelStatus(StrEnum):
    """Lifecycle state of a local model, as shown in the model picker."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Conversation:
    """A saved chat conversation."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Message:
    """One saved message in a conversation.

    ``reasoning`` holds the model's "thinking" text, if it produced any.
    """

    id: str
    conversation_id: str
    role: Role
    content: str
    created_at: datetime
    reasoning: str | None = None
    model_id: str | None = None


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A message as sent to the model: just a role and its text."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ChatDelta:
    """An incremental piece of a streamed reply."""

    content: str = ""
    reasoning: str = ""


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """A local GGUF model known to the inference server."""

    id: str
    status: ModelStatus
    path: str | None = None
    exit_code: int | None = None
