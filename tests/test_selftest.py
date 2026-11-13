"""Tests for ``dom --self-test``."""

from __future__ import annotations

import pytest

from dom.domain import ChatDelta
from dom.selftest import run_self_test
from tests.fakes import FakeInferenceService


async def test_self_test_passes_with_a_working_engine() -> None:
    service = FakeInferenceService()
    assert await run_self_test(service) == 0
    assert service.stopped


@pytest.mark.parametrize(
    "breakage",
    ["no models", "start fails", "load fails", "empty reply"],
)
async def test_self_test_fails_cleanly(breakage: str) -> None:
    service = FakeInferenceService()
    if breakage == "no models":
        service.model_ids = []
    elif breakage == "start fails":
        service.start_error = "llama-server crashed"
    elif breakage == "load fails":
        service.load_error = "out of memory"
    else:
        service.reply = [ChatDelta(reasoning="thinking only")]
    assert await run_self_test(service) == 1
    assert service.stopped
