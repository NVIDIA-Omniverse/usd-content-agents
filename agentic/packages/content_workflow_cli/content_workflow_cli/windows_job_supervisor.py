# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Windows Job Object supervisor for untrusted child-agent process trees.

The runner starts this module with two anonymous-pipe handles.  The supervisor
creates its kill-on-close Job Object before reporting readiness, and it does
not create the provider process until the runner acknowledges that readiness.
The provider is created suspended, assigned to the Job Object, and only then
resumed, so no provider descendant can escape during process startup.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

_SUPERVISOR_READY_TOKEN = b"R"
_SUPERVISOR_START_TOKEN = b"S"
_SUPERVISOR_ERROR = 125
_CLEANUP_TIMEOUT_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.02

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_PROCESS_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 258
_CREATE_SUSPENDED = 0x00000004


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


def _kernel32() -> ctypes.WinDLL:
    if os.name != "nt":
        raise RuntimeError("Windows Job Object supervision requires Windows")
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
    library.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    library.OpenProcess.restype = wintypes.HANDLE
    library.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    library.WaitForSingleObject.restype = wintypes.DWORD
    library.CloseHandle.argtypes = [wintypes.HANDLE]
    library.CloseHandle.restype = wintypes.BOOL
    return library


def _raise_last_windows_error(message: str) -> None:
    error_number = ctypes.get_last_error()
    raise OSError(error_number, f"{message}: {ctypes.FormatError(error_number)}")


def _create_kill_on_close_job(kernel32: ctypes.WinDLL) -> int:
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        _raise_last_windows_error("could not create Job Object")
    limits = _JobObjectExtendedLimitInformation()
    limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    ):
        kernel32.CloseHandle(job)
        _raise_last_windows_error("could not configure Job Object")
    return int(job)


def _open_parent_process(kernel32: ctypes.WinDLL, parent_pid: int) -> int:
    handle = kernel32.OpenProcess(_PROCESS_SYNCHRONIZE, False, parent_pid)
    if not handle:
        _raise_last_windows_error("could not monitor the workflow runner")
    return int(handle)


def _resume_process(process: subprocess.Popen[bytes]) -> None:
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    status = int(ntdll.NtResumeProcess(int(process._handle)))  # type: ignore[attr-defined]
    if status < 0:
        raise OSError(
            f"NtResumeProcess failed with NTSTATUS 0x{status & 0xFFFFFFFF:08x}"
        )


def _active_process_count(kernel32: ctypes.WinDLL, job: int) -> int:
    accounting = _JobObjectBasicAccountingInformation()
    if not kernel32.QueryInformationJobObject(
        job,
        _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
        ctypes.byref(accounting),
        ctypes.sizeof(accounting),
        None,
    ):
        _raise_last_windows_error("could not query Job Object")
    return int(accounting.ActiveProcesses)


def _terminate_and_wait_for_empty_job(kernel32: ctypes.WinDLL, job: int) -> None:
    if not kernel32.TerminateJobObject(job, _SUPERVISOR_ERROR):
        _raise_last_windows_error("could not terminate Job Object")
    deadline = time.monotonic() + _CLEANUP_TIMEOUT_SECONDS
    while _active_process_count(kernel32, job):
        if time.monotonic() >= deadline:
            raise RuntimeError("Job Object descendants remained after cleanup timeout")
        time.sleep(_POLL_INTERVAL_SECONDS)


def _open_inherited_pipe(handle: int, flags: int) -> int:
    import msvcrt

    if handle <= 0:
        raise ValueError("invalid inherited pipe handle")
    return msvcrt.open_osfhandle(handle, flags)


def _parse_arguments(arguments: list[str]) -> tuple[int, int, int, list[str]]:
    if (
        len(arguments) < 7
        or arguments[0] != "--parent-pid"
        or arguments[2] != "--ready-handle"
        or arguments[4] != "--start-handle"
        or arguments[6] != "--"
    ):
        raise ValueError("missing or invalid readiness protocol")
    parent_pid = int(arguments[1])
    ready_handle = int(arguments[3])
    start_handle = int(arguments[5])
    command = arguments[7:]
    if parent_pid <= 0 or ready_handle <= 0 or start_handle <= 0:
        raise ValueError("invalid readiness handles or parent process")
    if ready_handle == start_handle:
        raise ValueError("readiness handles must be distinct")
    if not command:
        raise ValueError("missing command")
    return parent_pid, ready_handle, start_handle, command


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if os.name != "nt":
        print("windows-job-supervisor: Windows is required", file=sys.stderr)
        return _SUPERVISOR_ERROR
    try:
        parent_pid, ready_handle, start_handle, command = _parse_arguments(arguments)
    except (TypeError, ValueError) as exc:
        print(f"windows-job-supervisor: {exc}", file=sys.stderr)
        return _SUPERVISOR_ERROR

    kernel32 = _kernel32()
    job = 0
    parent = 0
    ready_fd = -1
    start_fd = -1
    target: subprocess.Popen[bytes] | None = None
    cleanup_required = False
    try:
        job = _create_kill_on_close_job(kernel32)
        parent = _open_parent_process(kernel32, parent_pid)
        ready_fd = _open_inherited_pipe(ready_handle, os.O_WRONLY)
        start_fd = _open_inherited_pipe(start_handle, os.O_RDONLY)
        if os.write(ready_fd, _SUPERVISOR_READY_TOKEN) != len(_SUPERVISOR_READY_TOKEN):
            raise RuntimeError("short readiness write")
        os.close(ready_fd)
        ready_fd = -1
        if os.read(start_fd, len(_SUPERVISOR_START_TOKEN)) != _SUPERVISOR_START_TOKEN:
            raise RuntimeError("runner did not confirm supervisor readiness")
        os.close(start_fd)
        start_fd = -1

        target = subprocess.Popen(
            command,
            creationflags=_CREATE_SUSPENDED,
        )
        cleanup_required = True
        if not kernel32.AssignProcessToJobObject(job, int(target._handle)):  # type: ignore[attr-defined]
            _raise_last_windows_error("could not assign provider to Job Object")
        _resume_process(target)

        while target.poll() is None:
            parent_state = kernel32.WaitForSingleObject(parent, 0)
            if parent_state == _WAIT_OBJECT_0:
                raise RuntimeError("workflow runner exited before its child")
            if parent_state != _WAIT_TIMEOUT:
                _raise_last_windows_error("could not monitor the workflow runner")
            time.sleep(_POLL_INTERVAL_SECONDS)
        returncode = int(target.wait())
        _terminate_and_wait_for_empty_job(kernel32, job)
        cleanup_required = False
        return returncode
    except BaseException as exc:  # noqa: BLE001 - isolated process boundary
        print(f"windows-job-supervisor: {exc}", file=sys.stderr)
        return _SUPERVISOR_ERROR
    finally:
        for fd in (ready_fd, start_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if cleanup_required and job:
            try:
                _terminate_and_wait_for_empty_job(kernel32, job)
            except Exception as exc:  # noqa: BLE001 - preserve primary failure
                print(f"windows-job-supervisor: cleanup failed: {exc}", file=sys.stderr)
        if parent:
            kernel32.CloseHandle(parent)
        if job:
            kernel32.CloseHandle(job)


if __name__ == "__main__":
    raise SystemExit(main())
