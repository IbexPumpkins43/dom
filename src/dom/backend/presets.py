"""The llama-server presets file: global defaults plus one tuned section per model."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import replace
from typing import TYPE_CHECKING

from dom.backend.errors import BackendError

if TYPE_CHECKING:
    from pathlib import Path

    from dom.config import InferenceSettings

logger = logging.getLogger(__name__)

PRESETS_VERSION = "1"
_GLOBAL_SECTION = "*"


def render_presets(defaults: InferenceSettings, models: dict[str, InferenceSettings]) -> str:
    """The INI text for ``defaults`` (the ``[*]`` section) and per-model sections.

    The context size is never set globally: a pinned context makes llama-server measure a
    model's memory at that one size only, and tuning needs the unpinned measurement.
    """
    defaults = replace(defaults, ctx_size=None)
    lines = [f"version = {PRESETS_VERSION}", "", "; Written by dom. Changes will be overwritten."]
    sections = [(_GLOBAL_SECTION, defaults), *sorted(models.items())]
    for name, settings in sections:
        if any(ch in name for ch in "[]\n\r"):
            raise BackendError(f"Model name {name!r} cannot be used in the presets file.")
        lines += ["", f"[{name}]"]
        lines += [f"{key} = {value}" for key, value in settings.to_preset_items()]
    return "\n".join(lines) + "\n"


class PresetFile:
    """Owns the presets file llama-server reads (``--models-preset``)."""

    def __init__(self, path: Path, defaults: InferenceSettings) -> None:
        """Manage the file at ``path`` with ``defaults`` for every model."""
        self._path = path
        self._defaults = defaults
        self._models: dict[str, InferenceSettings] = {}
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Where the file is."""
        return self._path

    def reset(self) -> None:
        """Keep only the global defaults (so llama-server measures every model unpinned)."""
        with self._lock:
            self._models.clear()
            self._write()

    def set_model(self, model_id: str, settings: InferenceSettings) -> bool:
        """Set one model's settings; returns True if the file changed."""
        with self._lock:
            if self._models.get(model_id) == settings:
                return False
            self._models[model_id] = settings
            self._write()
            return True

    def remove_model(self, model_id: str) -> bool:
        """Drop one model's section (its file is being deleted); returns True if it had one.

        llama-server lists every model with a section, even when its file is gone.
        """
        with self._lock:
            if self._models.pop(model_id, None) is None:
                return False
            self._write()
            return True

    def _write(self) -> None:
        text = render_presets(self._defaults, self._models)
        tmp = self._path.with_name(self._path.name + ".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as exc:
            raise BackendError(
                f"Could not write the model settings file {self._path}: {exc}"
            ) from exc
        logger.debug("Wrote presets:\n%s", text)
