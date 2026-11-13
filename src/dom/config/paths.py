"""Where dom keeps its files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_path

APP_NAME = "dom"
DEV_ENV_VAR = "DOM_DEV"

# src/dom/config/paths.py -> project root (only meaningful when running from a checkout).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class AppPaths:
    """All file-system locations the app uses."""

    data_dir: Path
    bin_dir: Path
    models_dir: Path
    db_path: Path
    llama_cache_dir: Path
    log_dir: Path

    @classmethod
    def for_data_dir(
        cls, data_dir: Path, *, bin_dir: Path | None = None, models_dir: Path | None = None
    ) -> AppPaths:
        """Lay out the standard sub-folders under ``data_dir``."""
        return cls(
            data_dir=data_dir,
            bin_dir=bin_dir or data_dir / "bin",
            models_dir=models_dir or data_dir / "models",
            db_path=data_dir / "dom.sqlite3",
            # Kept empty on purpose: llama-server's router also lists models from its download
            # cache, and pointing it here means only models_dir shows up in the picker.
            llama_cache_dir=data_dir / "llama-cache",
            log_dir=data_dir / "logs",
        )

    def ensure_exist(self) -> None:
        """Create the folders the app writes to."""
        for directory in (self.data_dir, self.models_dir, self.llama_cache_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)


def is_dev_mode() -> bool:
    """True when ``DOM_DEV=1`` is set."""
    return os.environ.get(DEV_ENV_VAR, "") == "1"


def default_paths() -> AppPaths:
    """User-data paths, or the checkout's ``.dev/`` folders in development mode."""
    if is_dev_mode():
        dev = _PROJECT_ROOT / ".dev"
        return AppPaths.for_data_dir(dev / "data", bin_dir=dev / "bin", models_dir=dev / "models")
    return AppPaths.for_data_dir(user_data_path(APP_NAME, appauthor=False))
