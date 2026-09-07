# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable native-Windows process identity and containment helpers."""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from ctypes import wintypes
from typing import Any

_CREATE_SUSPENDED = 0x00000004
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_WAIT_TIMEOUT = 258


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JobObjectBasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


def _windows_job_kernel32() -> ctypes.WinDLL:
    if os.name != "nt":
        raise RuntimeError("Windows Job Object containment requires Windows")
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    library.CreateJobObjectW.restype = wintypes.HANDLE
    library.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    library.SetInformationJobObject.restype = wintypes.BOOL
    library.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    library.AssignProcessToJobObject.restype = wintypes.BOOL
    library.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    library.TerminateJobObject.restype = wintypes.BOOL
    library.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    library.QueryInformationJobObject.restype = wintypes.BOOL
    library.CloseHandle.argtypes = [wintypes.HANDLE]
    library.CloseHandle.restype = wintypes.BOOL
    return library


def _raise_windows_error(message: str, error_number: int | None = None) -> None:
    number = ctypes.get_last_error() if error_number is None else error_number
    raise OSError(number, f"{message}: {ctypes.FormatError(number)}")


class WindowsKillOnCloseJob:
    """Own a native Windows process tree and terminate it when closed."""

    creation_flags = _CREATE_SUSPENDED

    def __init__(self, *, allow_breakaway: bool = False) -> None:
        self._kernel32 = _windows_job_kernel32()
        job = self._kernel32.CreateJobObjectW(None, None)
        if not job:
            _raise_windows_error("could not create Job Object")
        self._handle = int(job)
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if allow_breakaway:
            limits.BasicLimitInformation.LimitFlags |= _JOB_OBJECT_LIMIT_BREAKAWAY_OK
        if not self._kernel32.SetInformationJobObject(
            self._handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error_number = ctypes.get_last_error()
            self._kernel32.CloseHandle(self._handle)
            self._handle = 0
            _raise_windows_error("could not configure Job Object", error_number)

    def __enter__(self) -> WindowsKillOnCloseJob:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _require_handle(self) -> int:
        if not self._handle:
            raise RuntimeError("Windows Job Object is closed")
        return self._handle

    @staticmethod
    def _process_handle(process: subprocess.Popen[Any]) -> int:
        handle = getattr(process, "_handle", None)
        if not handle:
            raise RuntimeError("subprocess does not expose a live Windows handle")
        return int(handle)

    def assign_process(self, process: subprocess.Popen[Any]) -> None:
        """Assign a suspended ``Popen`` process before it may create children."""

        if not self._kernel32.AssignProcessToJobObject(
            self._require_handle(),
            self._process_handle(process),
        ):
            _raise_windows_error("could not assign process to Job Object")

    def resume_process(self, process: subprocess.Popen[Any]) -> None:
        """Resume a process created with :attr:`creation_flags`."""

        self._require_handle()
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        status = int(ntdll.NtResumeProcess(self._process_handle(process)))
        if status < 0:
            raise OSError(
                f"NtResumeProcess failed with NTSTATUS 0x{status & 0xFFFFFFFF:08x}"
            )

    def active_process_count(self) -> int:
        """Return the number of processes currently contained by the job."""

        accounting = _JobObjectBasicAccountingInformation()
        if not self._kernel32.QueryInformationJobObject(
            self._require_handle(),
            _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            None,
        ):
            _raise_windows_error("could not query Job Object")
        return int(accounting.ActiveProcesses)

    def terminate(self, *, exit_code: int = 1) -> None:
        """Terminate every process in the job, including orphaned descendants."""

        if (
            not isinstance(exit_code, int)
            or isinstance(exit_code, bool)
            or not 0 <= exit_code <= 0xFFFFFFFF
        ):
            raise ValueError("exit_code must be an unsigned 32-bit integer")
        handle = self._require_handle()
        if self.active_process_count() == 0:
            return
        if not self._kernel32.TerminateJobObject(handle, exit_code):
            _raise_windows_error("could not terminate Job Object")

    def wait_for_empty(
        self,
        *,
        timeout_s: float,
        poll_interval_s: float = 0.02,
    ) -> None:
        """Poll until every process has left the job or raise ``TimeoutError``."""

        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        deadline = time.monotonic() + timeout_s
        while self.active_process_count():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Job Object descendants remained after cleanup timeout"
                )
            time.sleep(min(poll_interval_s, max(0.0, deadline - time.monotonic())))

    def close(self) -> None:
        """Close the job handle, idempotently triggering kill-on-close cleanup."""

        if not self._handle:
            return
        if not self._kernel32.CloseHandle(self._handle):
            _raise_windows_error("could not close Job Object")
        self._handle = 0


def windows_process_start_token(pid: int) -> str | None:
    """Return a PID-reuse-resistant creation-time token for a live process."""

    if os.name != "nt" or not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(
        _PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE,
        False,
        pid,
    )
    if not handle:
        return None
    try:
        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        if int(kernel32.WaitForSingleObject(handle, 0)) != _WAIT_TIMEOUT:
            return None
        value = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        return f"w{value:x}"
    finally:
        kernel32.CloseHandle(handle)


def windows_process_is_live(pid: int) -> bool:
    """Return whether the exact numeric PID currently names a live process."""

    if os.name != "nt" or not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(_SYNCHRONIZE, False, pid)
    if not handle:
        return False
    try:
        return int(kernel32.WaitForSingleObject(handle, 0)) == _WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)
