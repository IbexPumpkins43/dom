"""``dom --self-test``: check an installation end to end without a window.

Starts the AI engine, loads the smallest model in the models folder, asks one question and
checks that an answer comes back, then stops everything. Used by CI on the packaged app for every
platform; also handy for support ("does the engine work on this computer?"). The result is
logged (a packaged Windows app has no console) and returned as the exit code.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from dom.backend.errors import BackendError
from dom.domain import ChatMessage, Role

if TYPE_CHECKING:
    from dom.backend.protocols import InferenceService

logger = logging.getLogger("dom.selftest")

REPLY_TIMEOUT_S = 600.0
PROMPT = "Reply with one short friendly sentence."


def _size(path: str | None) -> int:
    if path is None:
        return 0
    with contextlib.suppress(OSError):
        return Path(path).stat().st_size
    return 0


def _log_step(step: str) -> None:
    logger.info("SELF-TEST %s", step)


async def run_self_test(service: InferenceService) -> int:
    """Returns 0 if a model loaded and answered, 1 otherwise."""
    try:
        await service.start()
        profile = service.hardware_profile()
        logger.info("SELF-TEST hardware: %s", profile.summary() if profile else "unknown")
        models = await service.list_models()
        if not models:
            logger.error("SELF-TEST FAILED: no models in the models folder")
            return 1
        model = min(models, key=lambda m: _size(m.path))
        loaded = await service.select_model(model.id, _log_step)
        logger.info(
            "SELF-TEST loaded %s on %s, context %s", model.id, loaded.device_label, loaded.ctx_size
        )
        parts: list[str] = []
        async with asyncio.timeout(REPLY_TIMEOUT_S):
            async for delta in service.stream_chat(model.id, [ChatMessage(Role.USER, PROMPT)]):
                parts.append(delta.content)
        reply = "".join(parts).strip()
        if not reply:
            logger.error("SELF-TEST FAILED: empty reply")
            return 1
        logger.info("SELF-TEST PASSED: %r", reply[:200])
        return 0
    except (BackendError, TimeoutError) as exc:
        logger.error("SELF-TEST FAILED: %s", exc)
        return 1
    finally:
        await service.stop()
