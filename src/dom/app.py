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
from PySide6.QtWidgets import QApplication

from dom.backend import LlamaInferenceService, LlamaServerProcess, ServerConfig
from dom.backend.errors import EngineMissingError
from dom.backend.lazy import LazyInferenceService
from dom.backend.presets import PresetFile
from dom.backend.service import AutoTuning
from dom.catalog import CATALOG
from dom.config import (
    AppPaths,
    InferenceSettings,
    LlamaServerNotFoundError,
    configured_server_path,
    default_paths,
    find_server_builds,
    is_dev_mode,
)
from dom.downloads.engine import EngineInstaller
from dom.downloads.library import ModelLibrary
from dom.hardware import HardwareProbe
from dom.hardware.probe import read_memory, read_usage
from dom.selftest import run_self_test
from dom.storage import SqliteStore, StorageError
from dom.tuning import BenchmarkCache, LlamaBench
from dom.ui.assets import use_ui_font
from dom.ui.controller import ChatController, spawn
from dom.ui.dialogs import show_message
from dom.ui.library import LibraryController
from dom.ui.main_window import MainWindow
from dom.ui.profiles import ProfileController, initial_profile
from dom.ui.theme import apply_theme

if TYPE_CHECKING:
    from pathlib import Path

    from dom.backend import InferenceService
    from dom.domain import Profile

logger = logging.getLogger("dom")

# Python only runs signal handlers between bytecodes; this timer keeps the interpreter
# ticking while Qt's event loop waits, so Ctrl+C / SIGTERM close the app promptly.
_SIGNAL_WAKE_MS = 250
_LOG_BYTES = 2_000_000
SELF_TEST_FLAG = "--self-test"


def configure_logging(paths: AppPaths) -> None:
    """Log to ``<data>/logs/dom.log`` and to stderr."""
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        paths.log_dir / "dom.log", maxBytes=_LOG_BYTES, backupCount=2, encoding="utf-8"
    )
    handlers: list[logging.Handler] = [file_handler]
    # A packaged Windows app has no console: sys.stderr is None there.
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _build_version(binary: Path) -> str:
    try:
        return (binary.parent / "VERSION").read_text(encoding="utf-8").strip() or str(binary)
    except OSError:
        return str(binary)


def build_service(paths: AppPaths, defaults: InferenceSettings) -> InferenceService:
    """The llama-server service; on each (re)start it finds the builds and probes hardware."""

    async def create() -> InferenceService:
        try:
            builds = find_server_builds(configured_server_path(), paths.engine_dirs)
        except LlamaServerNotFoundError as exc:
            logger.warning("llama-server not found; searched: %s", ", ".join(exc.searched))
            raise EngineMissingError(
                "The AI engine (llama.cpp) is not installed yet. dom can download it for you "
                "from GitHub (about 60 MB), or you can put llama-server and its files in "
                f"{paths.bin_dir}."
            ) from exc
        hardware = await HardwareProbe().detect(builds)
        build = hardware.build
        if build is None:
            raise EngineMissingError("No usable llama-server build was found.")
        binary = build.path
        logger.info("Using llama-server at %s", binary)
        presets = PresetFile(paths.data_dir / "presets.ini", defaults)
        process = LlamaServerProcess(
            ServerConfig(
                binary=binary,
                models_dir=paths.models_dir,
                cache_dir=paths.llama_cache_dir,
                log_path=paths.log_dir / "llama-server.log",
                presets=presets,
            )
        )
        tuning = AutoTuning(
            hardware=hardware,
            presets=presets,
            defaults=defaults,
            memory_reader=read_memory,
            bench=LlamaBench(binary.with_name(binary.name.replace("llama-server", "llama-bench"))),
            bench_cache=BenchmarkCache(paths.data_dir / "benchmarks.json"),
            build_version=_build_version(binary),
        )
        return LlamaInferenceService(process, tuning)

    return LazyInferenceService(create)


async def _run(
    service: InferenceService, store: SqliteStore, profile: Profile, paths: AppPaths
) -> int:
    closed = asyncio.Event()
    window = MainWindow(on_closed=closed.set, usage_reader=read_usage)
    controller = ChatController(service, store, window, EngineInstaller(paths.bin_dir), profile)
    profiles = ProfileController(store, window, controller)
    probe = HardwareProbe()
    library = LibraryController(
        CATALOG,
        ModelLibrary(paths.models_dir, CATALOG),
        hardware=lambda: service.hardware_profile() or probe.basic(),
        on_installed=controller.models_changed,
    )
    window.attach(controller, library, profiles)
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
    spawn(profiles.startup(), tasks)
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

    if SELF_TEST_FLAG in sys.argv[1:]:
        service = build_service(paths, InferenceSettings())
        atexit.register(service.stop_now)
        return asyncio.run(run_self_test(service))

    app = QApplication(sys.argv)
    app.setApplicationName("dom")
    use_ui_font(app)
    apply_theme(app)
    # The window closes itself only after the server is stopped (see MainWindow.closeEvent).
    app.setQuitOnLastWindowClosed(False)

    try:
        store = SqliteStore(paths.db_path)
        profile = initial_profile(store)
    except StorageError as exc:
        logger.exception("Could not open the database")
        show_message(None, "dom", f"dom could not open its chat history.\n\n{exc}")
        return 1

    service = build_service(paths, InferenceSettings())
    # Last-resort cleanup if the event loop dies with an exception: never leave the server.
    atexit.register(service.stop_now)
    try:
        return asyncio.run(
            _run(service, store, profile, paths), loop_factory=lambda: qasync.QEventLoop(app)
        )
    finally:
        service.stop_now()
