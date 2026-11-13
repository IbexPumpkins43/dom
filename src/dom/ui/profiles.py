"""Profile controller: switching, adding, editing and deleting profiles (Qt-free, like a browser's).

Each profile has its own chats, system prompt, generation settings and last-used model. The
models themselves and the AI engine are shared. dom opens in the profile used last.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from dom.domain import PROFILE_NAME_MAX_CHARS, ProfileColour, ProfileSettings
from dom.storage.errors import StorageError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dom.domain import Profile
    from dom.storage.protocols import ProfileRepository
    from dom.ui.controller import ChatController

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_NAME = "Default"


class ProfileView(Protocol):
    """What the profile controller tells the UI to show."""

    def show_profiles(self, profiles: Sequence[Profile], current: Profile) -> None:
        """Every profile (for the switcher menu) and the one in use."""
        ...

    def show_error(self, message: str) -> None:
        """A one-off plain-English error message."""
        ...


def clean_settings(settings: ProfileSettings) -> ProfileSettings:
    """Trim the name (falling back to "Profile") and the system prompt."""
    name = " ".join(settings.name.split())[:PROFILE_NAME_MAX_CHARS] or "Profile"
    return replace(settings, name=name, system_prompt=settings.system_prompt.strip())


def next_colour(profiles: Sequence[Profile]) -> ProfileColour:
    """The first avatar colour no profile uses yet (cycling once all are taken)."""
    used = [p.settings.colour for p in profiles]
    colours = list(ProfileColour)
    unused = [c for c in colours if c not in used]
    return unused[0] if unused else colours[len(profiles) % len(colours)]


def new_profile_settings(profiles: Sequence[Profile]) -> ProfileSettings:
    """Starting values for the "Add profile" form: "Profile 2", an unused colour, defaults."""
    names = {p.name for p in profiles}
    number = len(profiles) + 1
    while f"Profile {number}" in names:
        number += 1
    return ProfileSettings(name=f"Profile {number}", colour=next_colour(profiles))


def initial_profile(repo: ProfileRepository) -> Profile:
    """The profile to open at startup: the one used last (creating one if there are none).

    Synchronous: called before the event loop starts.

    Raises:
        StorageError: if the profiles cannot be read or created.
    """
    profiles = repo.list_profiles()
    if not profiles:
        return repo.create_profile(ProfileSettings(DEFAULT_PROFILE_NAME, ProfileColour.ORANGE))
    return max(profiles, key=lambda p: p.last_used_at)


class ProfileController:
    """Manages profiles; the chat controller holds the one in use."""

    def __init__(self, repo: ProfileRepository, view: ProfileView, chat: ChatController) -> None:
        """Wire up the controller; call ``startup`` to show the profiles."""
        self._repo = repo
        self._view = view
        self._chat = chat
        self._profiles: list[Profile] = []

    @property
    def profiles(self) -> list[Profile]:
        """Every profile, oldest first (as last shown)."""
        return list(self._profiles)

    @property
    def current(self) -> Profile:
        """The profile in use."""
        return self._chat.profile

    def new_profile_settings(self) -> ProfileSettings:
        """Starting values for the "Add profile" form."""
        return new_profile_settings(self._profiles)

    async def startup(self) -> None:
        """Show the profiles."""
        await self._refresh()

    async def switch(self, profile_id: str) -> None:
        """Open another profile: its chats, settings and model."""
        if profile_id == self.current.id:
            return
        try:
            profile = await asyncio.to_thread(self._repo.mark_used, profile_id)
        except StorageError as exc:
            self._view.show_error(f"Could not open the profile. {exc}")
            return
        await self._chat.set_profile(profile)
        await self._refresh()

    async def create(self, settings: ProfileSettings) -> None:
        """Add a profile and switch to it."""
        try:
            profile = await asyncio.to_thread(self._repo.create_profile, clean_settings(settings))
        except StorageError as exc:
            self._view.show_error(f"Could not add the profile. {exc}")
            return
        await self._chat.set_profile(profile)
        await self._refresh()

    async def update(self, profile_id: str, settings: ProfileSettings) -> None:
        """Save a profile's settings (used from the next reply if it is the current one)."""
        try:
            profile = await asyncio.to_thread(
                self._repo.update_profile, profile_id, clean_settings(settings)
            )
        except StorageError as exc:
            self._view.show_error(f"Could not save the profile. {exc}")
            return
        if profile.id == self.current.id:
            await self._chat.set_profile(profile)
        await self._refresh()

    async def delete(self, profile_id: str) -> None:
        """Delete a profile and its chats; deleting the current one opens another first."""
        others = [p for p in self._profiles if p.id != profile_id]
        if not others:
            self._view.show_error("The last profile cannot be deleted.")
            return
        if profile_id == self.current.id:
            await self.switch(max(others, key=lambda p: p.last_used_at).id)
            if self.current.id == profile_id:
                return  # switching failed (already reported)
        try:
            await asyncio.to_thread(self._repo.delete_profile, profile_id)
        except StorageError as exc:
            self._view.show_error(f"Could not delete the profile. {exc}")
        await self._refresh()

    async def _refresh(self) -> None:
        try:
            self._profiles = await asyncio.to_thread(self._repo.list_profiles)
        except StorageError as exc:
            self._view.show_error(str(exc))
            return
        self._view.show_profiles(self._profiles, self.current)
