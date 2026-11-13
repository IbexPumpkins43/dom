"""Composition root: builds the real implementations and runs the Qt app."""

from __future__ import annotations

import asyncio
import atexit
import logging
import signal
import sys
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

import qasync
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from dom.backend import LlamaInferenceService, LlamaServerProcess, ServerConfig
from dom.backend.errors import ServerStartError
from dom.backend.lazy import LazyInferenceService
from dom.config import (
    AppPaths,
    InferenceSettings,
    LlamaServerNotFoundError,
    configured_server_path,
    default_paths,
    find_server_builds,
    is_dev_mode,
)
from dom.storage import SqliteConversationRepository, StorageError
from dom.ui.controller import ChatController, spawn
from dom.ui.main_window import MainWindow

if TYPE_CHECKING:
    from dom.backend import InferenceService

logger = logging.getLogger("dom")

# Python only runs signal handlers between bytecodes; this timer keeps the interpreter
# ticking while Qt's event loop waits, so Ctrl+C / SIGTERM close the app promptly.
_SIGNAL_WAKE_MS = 250
_LOG_BYTES = 2_000_000


def configure_logging(paths: AppPaths) -> None:
    """Log to ``<data>/logs/dom.log`` and to stderr."""
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        paths.log_dir / "dom.log", maxBytes=_LOG_BYTES, backupCount=2, encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[file_handler, logging.StreamHandler()],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def build_service(paths: AppPaths, settings: InferenceSettings) -> InferenceService:
    """The llama-server service; it locates the binary each time it is (re)started."""

    def create() -> InferenceService:
        try:
            builds = find_server_builds(configured_server_path(), paths.bin_dir)
            binary = next((b.path for b in builds if b.name == "cpu"), builds[0].path)
        except LlamaServerNotFoundError as exc:
            logger.warning("llama-server not found; searched: %s", ", ".join(exc.searched))
            raise ServerStartError(str(exc)) from exc
        logger.info("Using llama-server at %s", binary)
        process = LlamaServerProcess(
            ServerConfig(
                binary=binary,
                models_dir=paths.models_dir,
                cache_dir=paths.llama_cache_dir,
                log_path=paths.log_dir / "llama-server.log",
                settings=settings,
            )
        )
        return LlamaInferenceService(process)

    return LazyInferenceService(create)


async def _run(
    service: InferenceService, repo: SqliteConversationRepository, paths: AppPaths
) -> int:
    closed = asyncio.Event()
    window = MainWindow(on_closed=closed.set)
    controller = ChatController(service, repo, window, models_dir_hint=str(paths.models_dir))
    window.attach(controller)
    window.show()

    loop = asyncio.get_running_loop()

    def request_close(_signum: int, _frame: object) -> None:
        loop.call_soon_threadsafe(window.close)

    signal.signal(signal.SIGINT, request_close)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, request_close)
    wake = QTimer()
    wake.start(_SIGNAL_WAKE_MS)
    wake.timeout.connect(lambda: None)

    tasks: set[asyncio.Task[None]] = set()
    spawn(controller.startup(), tasks)
    await closed.wait()
    wake.stop()
    return 0


def main() -> int:
    """Run dom. Returns the process exit code."""
    paths = default_paths()
    paths.ensure_exist()
    configure_logging(paths)
    logger.info("Starting dom (dev mode: %s), data in %s", is_dev_mode(), paths.data_dir)

    app = QApplication(sys.argv)
    app.setApplicationName("dom")
    app.setStyle("Fusion")
    # The window closes itself only after the server is stopped (see MainWindow.closeEvent).
    app.setQuitOnLastWindowClosed(False)

    try:
        repo = SqliteConversationRepository(paths.db_path)
    except StorageError as exc:
        logger.exception("Could not open the database")
        QMessageBox.critical(None, "dom", f"dom could not open its chat history.\n\n{exc}")
        return 1

    service = build_service(paths, InferenceSettings())
    # Last-resort cleanup if the event loop dies with an exception: never leave the server.
    atexit.register(service.stop_now)
    try:
        return asyncio.run(_run(service, repo, paths), loop_factory=lambda: qasync.QEventLoop(app))
    finally:
        service.stop_now()
