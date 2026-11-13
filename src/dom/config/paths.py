"""Where dom keeps its files."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_path

APP_NAME = "dom"
DEV_ENV_VAR = "DOM_DEV"
DATA_DIR_ENV_VAR = "DOM_DATA_DIR"
# Folder next to the packaged executable that holds the llama.cpp builds shipped with dom.
BUNDLED_ENGINE_DIR = "llama"

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
    bundled_bin_dir: Path | None = None

    @property
    def engine_dirs(self) -> list[Path]:
        """Where to look for llama-server builds: downloaded ones first, then bundled ones."""
        return [self.bin_dir] + ([self.bundled_bin_dir] if self.bundled_bin_dir else [])

    @classmethod
    def for_data_dir(
        cls,
        data_dir: Path,
        *,
        bin_dir: Path | None = None,
        models_dir: Path | None = None,
        bundled_bin_dir: Path | None = None,
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
            bundled_bin_dir=bundled_bin_dir,
        )

    def ensure_exist(self) -> None:
        """Create the folders the app writes to."""
        for directory in (self.data_dir, self.models_dir, self.llama_cache_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)


def is_dev_mode() -> bool:
    """True when ``DOM_DEV=1`` is set."""
    return os.environ.get(DEV_ENV_VAR, "") == "1"


def bundled_engine_dir() -> Path | None:
    """The llama.cpp builds shipped next to a packaged (PyInstaller) dom, if this is one."""
    if not getattr(sys, "frozen", False):
        return None
    candidate = Path(sys.executable).resolve().parent / BUNDLED_ENGINE_DIR
    return candidate if candidate.is_dir() else None


def default_paths() -> AppPaths:
    """Where dom keeps its files.

    ``DOM_DEV=1`` uses the checkout's ``.dev/`` folders; ``DOM_DATA_DIR`` overrides the
    user-data folder (handy for testing or a portable install); otherwise the OS user-data folder.
    """
    if is_dev_mode():
        dev = _PROJECT_ROOT / ".dev"
        return AppPaths.for_data_dir(dev / "data", bin_dir=dev / "bin", models_dir=dev / "models")
    override = os.environ.get(DATA_DIR_ENV_VAR, "").strip()
    data_dir = Path(override) if override else user_data_path(APP_NAME, appauthor=False)
    return AppPaths.for_data_dir(data_dir, bundled_bin_dir=bundled_engine_dir())
