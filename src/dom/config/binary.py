"""Finding the llama-server executable."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

SERVER_STEM = "llama-server"
# llama.cpp programs read LLAMA_* variables (LLAMA_ARG_CTX_SIZE, ...) as settings. They are removed
# from every child environment so InferenceSettings stays the only source of settings.
_ENGINE_ENV_PREFIX = "LLAMA_"
# Keep llama.cpp programs from flashing a console window on Windows (0 elsewhere).
NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# Optional explicit location of llama-server (there is no settings screen yet).
CONFIGURED_PATH_ENV = "DOM_LLAMA_SERVER"


class LlamaServerNotFoundError(Exception):
    """No usable llama-server executable was found."""

    def __init__(self, searched: list[str], bin_dir: Path) -> None:
        self.searched = searched
        self.bin_dir = bin_dir
        super().__init__(
            "dom needs the llama-server program to run AI models, but it could not find it. "
            f"Put llama-server (and the files that come with it) in this folder: {bin_dir}"
        )


def engine_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """An environment for llama.cpp programs: ``base`` (default: ours) minus ``LLAMA_*``."""
    source = os.environ if base is None else base
    return {k: v for k, v in source.items() if not k.upper().startswith(_ENGINE_ENV_PREFIX)}


def run_engine_tool(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run a short-lived llama.cpp program (blocking; call from a worker thread).

    Raises:
        OSError: if it cannot be started.
        subprocess.TimeoutExpired: if it runs longer than ``timeout`` seconds.
    """
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=engine_env(),
        check=False,
        creationflags=NO_WINDOW,
    )


def server_executable_name(platform: str = sys.platform) -> str:
    """File name of the executable on this OS."""
    return f"{SERVER_STEM}.exe" if platform == "win32" else SERVER_STEM


def configured_server_path() -> Path | None:
    """The llama-server path set in ``DOM_LLAMA_SERVER``, if any."""
    value = os.environ.get(CONFIGURED_PATH_ENV, "").strip()
    return Path(value) if value else None


def _builds_in(directory: Path, exe: str) -> list[ServerBuild]:
    builds: list[ServerBuild] = []
    if directory.is_dir():
        for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
            if (sub / exe).is_file():
                builds.append(ServerBuild(sub.name, sub / exe))
    if (directory / exe).is_file():
        builds.append(ServerBuild("default", directory / exe))
    return builds


@dataclass(frozen=True, slots=True)
class ServerBuild:
    """One installed llama-server build, e.g. ``cpu`` or ``vulkan``."""

    name: str
    path: Path


def find_server_builds(
    configured: Path | None,
    bin_dirs: Path | Sequence[Path],
    which: Callable[[str], str | None] = shutil.which,
) -> list[ServerBuild]:
    """Locate llama-server builds.

    Order: the configured path (used alone if set and present); otherwise the first of
    ``bin_dirs`` that has any build (every ``<dir>/<name>/llama-server`` plus
    ``<dir>/llama-server``); otherwise ``PATH``.

    Raises:
        LlamaServerNotFoundError: if no build is found anywhere (names the first folder).
    """
    dirs = [bin_dirs] if isinstance(bin_dirs, Path) else list(bin_dirs)
    bin_dir = dirs[0]
    exe = server_executable_name()
    searched: list[str] = []
    if configured is not None:
        searched.append(str(configured))
        if configured.is_file():
            return [ServerBuild("configured", configured)]
    for directory in dirs:
        builds = _builds_in(directory, exe)
        searched.append(str(directory / exe))
        if builds:
            return builds
    searched.append(f"{SERVER_STEM} on PATH")
    on_path = which(SERVER_STEM)
    if on_path is not None:
        return [ServerBuild("path", Path(on_path))]
    raise LlamaServerNotFoundError(searched, bin_dir)
