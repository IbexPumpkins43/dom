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
class MemoryUse:
    """Bytes a model needs at a given context size, split by where they live."""

    n_ctx: int
    device_bytes: int
    host_bytes: int

    @property
    def total_bytes(self) -> int:
        """Device plus host bytes."""
        return self.device_bytes + self.host_bytes


@dataclass(frozen=True, slots=True)
class ModelMemory:
    """llama-server's memory measurement of a model at two context sizes."""

    low: MemoryUse
    high: MemoryUse

    @property
    def can_extrapolate(self) -> bool:
        """True if the two points differ, so memory per context token is known."""
        return self.high.n_ctx > self.low.n_ctx

    def at(self, n_ctx: int) -> MemoryUse:
        """Estimated memory at ``n_ctx`` tokens (linear between and beyond the two points)."""
        if not self.can_extrapolate:
            return self.low
        span = self.high.n_ctx - self.low.n_ctx

        def interpolate(low: int, high: int) -> int:
            return round(low + (high - low) * (n_ctx - self.low.n_ctx) / span)

        return MemoryUse(
            n_ctx=n_ctx,
            device_bytes=interpolate(self.low.device_bytes, self.high.device_bytes),
            host_bytes=interpolate(self.low.host_bytes, self.high.host_bytes),
        )


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """A local GGUF model known to the inference server."""

    id: str
    status: ModelStatus
    path: str | None = None
    exit_code: int | None = None
    context_length: int | None = None
    memory: ModelMemory | None = None
    memory_state: str | None = None
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A model that is loaded and ready, and how it was set up."""

    info: ModelInfo
    ctx_size: int | None
    device_label: str
    tuned: bool
