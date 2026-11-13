"""Domain layer: frozen dataclasses and enums shared by every other layer."""

from dom.domain.models import (
    PROFILE_NAME_MAX_CHARS,
    ChatDelta,
    ChatMessage,
    Conversation,
    LoadedModel,
    MemoryUse,
    Message,
    ModelInfo,
    ModelMemory,
    ModelStatus,
    Profile,
    ProfileColour,
    ProfileSettings,
    Role,
)

__all__ = [
    "PROFILE_NAME_MAX_CHARS",
    "ChatDelta",
    "ChatMessage",
    "Conversation",
    "LoadedModel",
    "MemoryUse",
    "Message",
    "ModelInfo",
    "ModelMemory",
    "ModelStatus",
    "Profile",
    "ProfileColour",
    "ProfileSettings",
    "Role",
]
