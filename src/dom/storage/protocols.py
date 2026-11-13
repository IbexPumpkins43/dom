"""Storage interfaces the rest of the app depends on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dom.domain import Conversation, Message, Role


class ConversationRepository(Protocol):
    """Persists conversations and their messages.

    Methods are synchronous and may touch the disk; callers on the UI thread must run them in a
    worker thread (for example with ``asyncio.to_thread``).
    """

    def create_conversation(self, title: str) -> Conversation:
        """Create and return a new, empty conversation."""
        ...

    def list_conversations(self) -> list[Conversation]:
        """Return all conversations, most recently updated first."""
        ...

    def get_messages(self, conversation_id: str) -> list[Message]:
        """Return a conversation's messages in the order they were added."""
        ...

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
        ...

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages."""
        ...

    def close(self) -> None:
        """Release the underlying storage."""
        ...
