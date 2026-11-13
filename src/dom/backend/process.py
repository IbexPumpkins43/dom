"""Running llama-server as a child process in router mode."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING

from dom.backend.client import LlamaClient
from dom.backend.errors import ServerStartError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dom.backend._winjob import KillOnCloseJob
    from dom.config import InferenceSettings

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
API_KEY_ENV = "LLAMA_API_KEY"
CACHE_ENV = "LLAMA_CACHE"
# llama-server reads LLAMA_ARG_* (and other LLAMA_*) variables as settings. Strip any from the
# user's environment so InferenceSettings is the only source of flags.
_STRIPPED_ENV_PREFIX = "LLAMA_"
_START_ATTEMPTS = 3
_HEALTH_POLL_S = 0.1
_GRACEFUL_STOP_S = 5.0
_LOG_TAIL_LINES = 15
_PORT_IN_USE_HINTS = ("address already in use", "couldn't bind", "failed to bind", "bind failed")


@dataclass(frozen=True, slots=True)
class ServerEndpoint:
    """How to reach a running server."""

    base_url: str
    api_key: str


def pick_free_port(host: str = HOST) -> int:
    """Ask the OS for a currently free TCP port on ``host``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port: int = sock.getsockname()[1]
        return port


def build_command(
    binary: Path, models_dir: Path, port: int, settings: InferenceSettings
) -> list[str]:
    """The full llama-server command line (flags checked against b12110 ``--help``)."""
    return [
        str(binary),
        "--host",
        HOST,
        "--port",
        str(port),
        "--models-dir",
        str(models_dir.resolve()),
        "--no-models-autoload",
        "--models-max",
        "1",
        "--offline",
        "--no-webui",
        *settings.to_server_args(),
    ]


def build_env(cache_dir: Path, api_key: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """Child environment: no inherited ``LLAMA_*`` settings, a private cache, and the API key."""
    source = dict(os.environ) if base is None else base
    env = {k: v for k, v in source.items() if not k.upper().startswith(_STRIPPED_ENV_PREFIX)}
    env[CACHE_ENV] = str(cache_dir.resolve())
    env[API_KEY_ENV] = api_key
    return env


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Everything needed to launch llama-server."""

    binary: Path
    models_dir: Path
    cache_dir: Path
    log_path: Path
    settings: InferenceSettings
    startup_timeout: float = 60.0


class LlamaServerProcess:
    """Starts, health-checks and stops one llama-server router and all its model processes."""

    def __init__(
        self, config: ServerConfig, *, port_picker: Callable[[], int] = pick_free_port
    ) -> None:
        """Configure (but do not start) the server."""
        self._binary = config.binary
        self._models_dir = config.models_dir
        self._cache_dir = config.cache_dir
        self._log_path = config.log_path
        self._settings = config.settings
        self._startup_timeout = config.startup_timeout
        self._port_picker = port_picker
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_file: IO[bytes] | None = None
        self._job: KillOnCloseJob | None = None
        self._endpoint: ServerEndpoint | None = None

    @property
    def endpoint(self) -> ServerEndpoint | None:
        """Where the server listens, once started."""
        return self._endpoint

    @property
    def pid(self) -> int | None:
        """Process id of the router, once started (on Linux also its process group id)."""
        return self._proc.pid if self._proc is not None else None

    @property
    def is_running(self) -> bool:
        """True while the router process is alive."""
        return self._proc is not None and self._proc.poll() is None

    async def start(self) -> ServerEndpoint:
        """Start the server and wait until ``/health`` answers.

        Raises:
            ServerStartError: if it exits, never becomes healthy, or cannot be launched.
        """
        if self.is_running and self._endpoint is not None:
            return self._endpoint
        last_error: ServerStartError | None = None
        for _ in range(_START_ATTEMPTS):
            try:
                return await self._start_once()
            except _PortInUseError as exc:
                logger.warning("Port was taken, retrying: %s", exc)
                last_error = ServerStartError(str(exc))
        raise last_error or ServerStartError("Could not start the AI server.")

    async def _start_once(self) -> ServerEndpoint:
        port = self._port_picker()
        api_key = secrets.token_urlsafe(32)
        command = build_command(self._binary, self._models_dir, port, self._settings)
        env = build_env(self._cache_dir, api_key)
        logger.info("Starting llama-server: %s", " ".join(command))
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._launch(command, env)
        endpoint = ServerEndpoint(base_url=f"http://{HOST}:{port}", api_key=api_key)
        try:
            await self._wait_healthy(endpoint)
        except BaseException:
            self.stop_now()
            raise
        self._endpoint = endpoint
        logger.info("llama-server is ready at %s", endpoint.base_url)
        return endpoint

    def _launch(self, command: list[str], env: dict[str, str]) -> None:
        self._log_file = self._log_path.open("wb")
        try:
            if sys.platform == "win32":
                from dom.backend._winjob import KillOnCloseJob  # noqa: PLC0415

                self._job = KillOnCloseJob()
                self._proc = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
                )
                self._job.assign(self._proc.pid)
            else:
                # Own session/process group, so stopping can signal the router and its children.
                self._proc = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
        except OSError as exc:
            self._close_handles()
            raise ServerStartError(
                f"Could not start the AI server program at {command[0]}: {exc.strerror or exc}"
            ) from exc
        except ServerStartError:
            self.stop_now()
            raise

    async def _wait_healthy(self, endpoint: ServerEndpoint) -> None:
        client = LlamaClient(endpoint.base_url, endpoint.api_key)
        deadline = time.monotonic() + self._startup_timeout
        try:
            while time.monotonic() < deadline:
                if self._proc is None or self._proc.poll() is not None:
                    self._raise_exited()
                if await client.health():
                    return
                await asyncio.sleep(_HEALTH_POLL_S)
        finally:
            await client.aclose()
        raise ServerStartError(
            f"The AI server did not start within {self._startup_timeout:.0f} seconds. "
            f"Details are in {self._log_path}."
        )

    def _raise_exited(self) -> None:
        code = self._proc.returncode if self._proc is not None else None
        tail = self.log_tail()
        if any(hint in tail.lower() for hint in _PORT_IN_USE_HINTS):
            raise _PortInUseError(f"llama-server could not bind its port (exit code {code}).")
        raise ServerStartError(
            f"The AI server stopped while starting (exit code {code}). "
            f"Details are in {self._log_path}."
        )

    def log_tail(self, lines: int = _LOG_TAIL_LINES) -> str:
        """The last lines of the server log, for error messages."""
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    async def wait_for_exit(self) -> int:
        """Wait (without blocking the event loop) until the router exits; return its code."""
        proc = self._proc
        if proc is None:
            raise ServerStartError("The AI server is not running.")
        return await asyncio.to_thread(proc.wait)

    async def stop(self) -> None:
        """Stop the server and all model processes without blocking the event loop."""
        await asyncio.to_thread(self.stop_now)

    def stop_now(self) -> None:
        """Stop the server and all model processes (blocks for up to a few seconds)."""
        proc = self._proc
        if proc is None:
            self._close_handles()
            return
        if sys.platform == "win32":
            # No console to deliver Ctrl+C to, and the server keeps no state worth flushing:
            # end the whole job (router + model processes) at once.
            if self._job is not None:
                self._job.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=_GRACEFUL_STOP_S)
        else:
            if proc.poll() is None:
                # SIGTERM lets the router shut down its model processes cleanly.
                proc.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=_GRACEFUL_STOP_S)
            # Whatever is left in the process group (stuck router or model processes) is killed.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=_GRACEFUL_STOP_S)
        logger.info("llama-server stopped (exit code %s)", proc.returncode)
        self._proc = None
        self._endpoint = None
        self._close_handles()

    def _close_handles(self) -> None:
        if self._job is not None:
            self._job.close()
            self._job = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


class _PortInUseError(ServerStartError):
    """Internal: the chosen port was taken between picking and binding it."""
