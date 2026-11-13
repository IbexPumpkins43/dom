"""Tests for LazyInferenceService (retrying after llama-server is added)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dom.backend import ServerStartError, ServerUnavailableError
from dom.backend.lazy import LazyInferenceService
from tests.fakes import FakeInferenceService

if TYPE_CHECKING:
    from dom.backend.protocols import InferenceService


async def test_start_retries_the_factory_until_it_succeeds() -> None:
    attempts: list[int] = []
    real = FakeInferenceService()

    def factory() -> InferenceService:
        attempts.append(1)
        if len(attempts) == 1:
            raise ServerStartError("llama-server is missing")
        return real

    lazy = LazyInferenceService(factory)
    with pytest.raises(ServerStartError, match="missing"):
        await lazy.start()
    with pytest.raises(ServerUnavailableError):
        await lazy.list_models()
    lazy.stop_now()  # nothing to stop yet; must not fail

    await lazy.start()
    assert [m.id for m in await lazy.list_models()] == ["model-a", "model-b"]
    await lazy.stop()
    assert real.stopped
    assert len(attempts) == 2  # noqa: PLR2004
