"""Finding the llama-server executable."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

SERVER_STEM = "llama-server"


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


def find_llama_server(
    configured: Path | None,
    bin_dir: Path,
    which: Callable[[str], str | None] = shutil.which,
) -> Path:
    """Locate llama-server: the configured path, then ``bin_dir``, then ``PATH``.

    Raises:
        LlamaServerNotFoundError: if none of those has it.
    """
    searched: list[str] = []
    if configured is not None:
        searched.append(str(configured))
        if configured.is_file():
            return configured
    in_bin = bin_dir / server_executable_name()
    searched.append(str(in_bin))
    if in_bin.is_file():
        return in_bin
    searched.append(f"{SERVER_STEM} on PATH")
    on_path = which(SERVER_STEM)
    if on_path is not None:
        return Path(on_path)
    raise LlamaServerNotFoundError(searched, bin_dir)
