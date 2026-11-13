"""Finding the llama-server executable."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

SERVER_STEM = "llama-server"
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


def server_executable_name(platform: str = sys.platform) -> str:
    """File name of the executable on this OS."""
    return f"{SERVER_STEM}.exe" if platform == "win32" else SERVER_STEM


def configured_server_path() -> Path | None:
    """The llama-server path set in ``DOM_LLAMA_SERVER``, if any."""
    value = os.environ.get(CONFIGURED_PATH_ENV, "").strip()
    return Path(value) if value else None


@dataclass(frozen=True, slots=True)
class ServerBuild:
    """One installed llama-server build, e.g. ``cpu`` or ``vulkan``."""

    name: str
    path: Path


def find_server_builds(
    configured: Path | None,
    bin_dir: Path,
    which: Callable[[str], str | None] = shutil.which,
) -> list[ServerBuild]:
    """Locate llama-server builds.

    Order: the configured path (used alone if set and present); otherwise every
    ``bin_dir/<name>/llama-server`` plus ``bin_dir/llama-server``; otherwise ``PATH``.

    Raises:
        LlamaServerNotFoundError: if no build is found anywhere.
    """
    exe = server_executable_name()
    searched: list[str] = []
    if configured is not None:
        searched.append(str(configured))
        if configured.is_file():
            return [ServerBuild("configured", configured)]
    builds: list[ServerBuild] = []
    if bin_dir.is_dir():
        for sub in sorted(p for p in bin_dir.iterdir() if p.is_dir()):
            if (sub / exe).is_file():
                builds.append(ServerBuild(sub.name, sub / exe))
    if (bin_dir / exe).is_file():
        builds.append(ServerBuild("default", bin_dir / exe))
    searched.append(str(bin_dir / exe))
    if builds:
        return builds
    searched.append(f"{SERVER_STEM} on PATH")
    on_path = which(SERVER_STEM)
    if on_path is not None:
        return [ServerBuild("path", Path(on_path))]
    raise LlamaServerNotFoundError(searched, bin_dir)
