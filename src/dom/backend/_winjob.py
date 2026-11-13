"""Windows Job Objects: kill llama-server and every process it starts, even if dom crashes.

The router starts one child ``llama-server`` per loaded model. Putting the router in a job with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` means Windows kills the whole tree when the job handle is
closed, which also happens automatically if dom itself is killed.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from dom.backend.errors import ServerStartError

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


class _IoCounters(ctypes.Structure):
    _fields_ = (
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    )


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    )


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = (
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


class KillOnCloseJob:
    """A Windows job object that kills its processes when closed."""

    def __init__(self) -> None:
        """Create the job object."""
        if sys.platform != "win32":
            raise ServerStartError("Job objects are only available on Windows.")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        handle: int | None = self._kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ServerStartError(f"Could not create a job object ({ctypes.get_last_error()}).")
        self._handle: int | None = handle
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok: int = self._kernel32.SetInformationJobObject(
            wintypes.HANDLE(handle),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            error = ctypes.get_last_error()
            self.close()
            raise ServerStartError(f"Could not configure the job object ({error}).")

    def assign(self, pid: int) -> None:
        """Put a running process (and its future children) into the job."""
        if sys.platform != "win32" or self._handle is None:
            raise ServerStartError("The job object is closed.")
        process: int | None = self._kernel32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid
        )
        if not process:
            raise ServerStartError(
                f"Could not open the server process ({ctypes.get_last_error()})."
            )
        try:
            ok: int = self._kernel32.AssignProcessToJobObject(
                wintypes.HANDLE(self._handle), wintypes.HANDLE(process)
            )
            if not ok:
                raise ServerStartError(
                    f"Could not attach the server to a job object ({ctypes.get_last_error()})."
                )
        finally:
            self._kernel32.CloseHandle(wintypes.HANDLE(process))

    def terminate(self) -> None:
        """Kill every process in the job right away."""
        if sys.platform == "win32" and self._handle is not None:
            self._kernel32.TerminateJobObject(wintypes.HANDLE(self._handle), 1)

    def close(self) -> None:
        """Close the job handle, which kills any process still in it."""
        if sys.platform == "win32" and self._handle is not None:
            self._kernel32.CloseHandle(wintypes.HANDLE(self._handle))
            self._handle = None
