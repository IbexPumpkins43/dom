"""Storage interfaces the rest of the app depends on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dom.domain import Conversation, Message, Profile, ProfileSettings, Role


class ConversationRepository(Protocol):
    """Persists conversations and their messages.

    Methods are synchronous and may touch the disk; callers on the UI thread must run them in a
    worker thread (for example with ``asyncio.to_thread``).
    """

    def create_conversation(self, profile_id: str, title: str) -> Conversation:
        """Create and return a new, empty conversation in a profile."""
        ...

    def list_conversations(self, profile_id: str) -> list[Conversation]:
        """Return a profile's conversations, most recently updated first."""
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


class ProfileRepository(Protocol):
    """Persists profiles (each owns its conversations). Synchronous, like the above."""

    def list_profiles(self) -> list[Profile]:
        """Return every profile in the order they were added."""
        ...

    def create_profile(self, settings: ProfileSettings) -> Profile:
        """Create and return a new profile."""
        ...

    def update_profile(self, profile_id: str, settings: ProfileSettings) -> Profile:
        """Replace a profile's settings and return the updated profile."""
        ...

    def set_last_model(self, profile_id: str, model_id: str | None) -> None:
        """Remember the model a profile used last."""
        ...

    def mark_used(self, profile_id: str) -> Profile:
        """Record that a profile was opened now; return it."""
        ...

    def delete_profile(self, profile_id: str) -> None:
        """Delete a profile and all of its conversations."""
        ...


class Store(ConversationRepository, ProfileRepository, Protocol):
    """Conversations and profiles in one store."""
