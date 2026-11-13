# Minimal local stubs for qasync, which ships no type information. Only what dom uses.
import asyncio

from PySide6.QtCore import QCoreApplication

class QEventLoop(asyncio.AbstractEventLoop):
    def __init__(
        self,
        app: QCoreApplication | None = None,
        set_running_loop: bool = False,
        already_running: bool = False,
    ) -> None: ...
