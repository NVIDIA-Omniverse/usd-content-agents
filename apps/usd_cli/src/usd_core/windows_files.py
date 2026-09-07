# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small Windows primitives used by the in-tree usd-cli component."""

from __future__ import annotations

import os
import secrets
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path


def windows_process_start_token(pid: int) -> str | None:
    """Return a stable Windows process creation-time token for one live PID."""

    if os.name != "nt" or not 1 < pid <= 2_147_483_647:
        return None
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    get_exit_code.restype = wintypes.BOOL
    get_process_times = kernel32.GetProcessTimes
    get_process_times.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    get_process_times.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    handle = open_process(process_query_limited_information, False, pid)
    if not handle:
        return None
    try:
        exit_code = wintypes.DWORD()
        if not get_exit_code(handle, ctypes.byref(exit_code)) or exit_code.value != still_active:
            return None
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        if not get_process_times(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return None
        value = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return f"w{value:016x}"
    finally:
        close_handle(handle)


def windows_process_state(pid: int) -> str:
    """Return ``alive``, ``dead``, or ``unknown`` for one Windows PID."""

    if os.name != "nt" or not 1 < pid <= 2_147_483_647:
        return "dead"
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    get_exit_code.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    handle = open_process(0x1000, False, pid)
    if not handle:
        return "dead" if ctypes.get_last_error() == 87 else "unknown"
    try:
        exit_code = wintypes.DWORD()
        if not get_exit_code(handle, ctypes.byref(exit_code)):
            return "unknown"
        return "alive" if exit_code.value == 259 else "dead"
    finally:
        close_handle(handle)


def windows_process_parent_pid(pid: int) -> int | None:
    """Return the current parent PID recorded for one Windows process."""

    if os.name != "nt" or not 1 < pid <= 2_147_483_647:
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    create_snapshot = ctypes.windll.kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    create_snapshot.restype = wintypes.HANDLE
    process_first = ctypes.windll.kernel32.Process32FirstW
    process_first.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W)]
    process_first.restype = wintypes.BOOL
    process_next = ctypes.windll.kernel32.Process32NextW
    process_next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W)]
    process_next.restype = wintypes.BOOL
    close_handle = ctypes.windll.kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    snapshot = create_snapshot(0x00000002, 0)
    invalid_handle = wintypes.HANDLE(-1).value
    if snapshot == invalid_handle:
        return None
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        if not process_first(snapshot, ctypes.byref(entry)):
            return None
        while True:
            if entry.th32ProcessID == pid:
                return int(entry.th32ParentProcessID)
            if not process_next(snapshot, ctypes.byref(entry)):
                return None
    finally:
        close_handle(snapshot)


class _WindowsFileApi:
    """Lazily bound Win32/NT calls; importing this module remains portable."""

    FILE_LIST_DIRECTORY = 0x0001
    FILE_READ_DATA = 0x0001
    FILE_WRITE_DATA = 0x0002
    FILE_APPEND_DATA = 0x0004
    FILE_READ_ATTRIBUTES = 0x0080
    SYNCHRONIZE = 0x00100000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_OPEN = 0x00000001
    FILE_CREATE = 0x00000002
    FILE_OPEN_IF = 0x00000003
    FILE_DIRECTORY_FILE = 0x00000001
    FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
    FILE_NON_DIRECTORY_FILE = 0x00000040
    FILE_OPEN_REPARSE_POINT = 0x00200000
    FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    FILE_ATTRIBUTE_NORMAL = 0x00000080
    FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    OPEN_EXISTING = 3
    OBJ_CASE_INSENSITIVE = 0x00000040

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class UnicodeString(ctypes.Structure):
            _fields_ = [
                ("Length", wintypes.USHORT),
                ("MaximumLength", wintypes.USHORT),
                ("Buffer", wintypes.LPWSTR),
            ]

        class ObjectAttributes(ctypes.Structure):
            _fields_ = [
                ("Length", wintypes.ULONG),
                ("RootDirectory", wintypes.HANDLE),
                ("ObjectName", ctypes.POINTER(UnicodeString)),
                ("Attributes", wintypes.ULONG),
                ("SecurityDescriptor", wintypes.LPVOID),
                ("SecurityQualityOfService", wintypes.LPVOID),
            ]

        class IoStatusValue(ctypes.Union):
            _fields_ = [("Status", ctypes.c_long), ("Pointer", wintypes.LPVOID)]

        class IoStatusBlock(ctypes.Structure):
            _anonymous_ = ("value",)
            _fields_ = [("value", IoStatusValue), ("Information", ctypes.c_size_t)]

        class ByHandleFileInformation(ctypes.Structure):
            _fields_ = [
                ("FileAttributes", wintypes.DWORD),
                ("CreationTime", wintypes.FILETIME),
                ("LastAccessTime", wintypes.FILETIME),
                ("LastWriteTime", wintypes.FILETIME),
                ("VolumeSerialNumber", wintypes.DWORD),
                ("FileSizeHigh", wintypes.DWORD),
                ("FileSizeLow", wintypes.DWORD),
                ("NumberOfLinks", wintypes.DWORD),
                ("FileIndexHigh", wintypes.DWORD),
                ("FileIndexLow", wintypes.DWORD),
            ]

        self.ctypes = ctypes
        self.wintypes = wintypes
        self.UnicodeString = UnicodeString
        self.ObjectAttributes = ObjectAttributes
        self.IoStatusBlock = IoStatusBlock
        self.ByHandleFileInformation = ByHandleFileInformation
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.nt_create_file = ntdll.NtCreateFile
        self.nt_create_file.argtypes = [
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.DWORD,
            ctypes.POINTER(ObjectAttributes),
            ctypes.POINTER(IoStatusBlock),
            ctypes.POINTER(ctypes.c_longlong),
            wintypes.ULONG,
            wintypes.ULONG,
            wintypes.ULONG,
            wintypes.ULONG,
            wintypes.LPVOID,
            wintypes.ULONG,
        ]
        self.nt_create_file.restype = ctypes.c_long
        self.rtl_nt_status_to_dos_error = ntdll.RtlNtStatusToDosError
        self.rtl_nt_status_to_dos_error.argtypes = [ctypes.c_long]
        self.rtl_nt_status_to_dos_error.restype = wintypes.ULONG
        self.create_file = kernel32.CreateFileW
        self.create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self.create_file.restype = wintypes.HANDLE
        self.close_handle = kernel32.CloseHandle
        self.close_handle.argtypes = [wintypes.HANDLE]
        self.close_handle.restype = wintypes.BOOL
        self.get_file_information = kernel32.GetFileInformationByHandle
        self.get_file_information.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ByHandleFileInformation),
        ]
        self.get_file_information.restype = wintypes.BOOL
        self.get_final_path_name = kernel32.GetFinalPathNameByHandleW
        self.get_final_path_name.argtypes = [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        self.get_final_path_name.restype = wintypes.DWORD
        self.invalid_handle_value = wintypes.HANDLE(-1).value

    def reject_unsafe(self, handle: int, *, is_directory: bool) -> None:
        information = self.ByHandleFileInformation()
        if not self.get_file_information(handle, self.ctypes.byref(information)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        if information.FileAttributes & self.FILE_ATTRIBUTE_REPARSE_POINT:
            raise OSError("refusing to traverse a reparsed usd-cli state path")
        observed_directory = bool(
            information.FileAttributes & self.FILE_ATTRIBUTE_DIRECTORY
        )
        if observed_directory != is_directory:
            raise OSError("usd-cli state path has the wrong file type")
        if not is_directory and information.NumberOfLinks != 1:
            raise OSError("usd-cli state file must have exactly one link")

    def normalized_handle_path(self, handle: int) -> str:
        required = self.get_final_path_name(handle, None, 0, 0)
        if required == 0:
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        buffer = self.ctypes.create_unicode_buffer(required + 1)
        written = self.get_final_path_name(handle, buffer, len(buffer), 0)
        if written == 0 or written >= len(buffer):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        value = buffer.value
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
        return os.path.normcase(os.path.normpath(os.path.abspath(value)))

    def open_absolute_directory(self, path: Path) -> int:
        handle = self.create_file(
            str(path),
            self.FILE_LIST_DIRECTORY | self.FILE_READ_ATTRIBUTES | self.SYNCHRONIZE,
            self.FILE_SHARE_READ | self.FILE_SHARE_WRITE,
            None,
            self.OPEN_EXISTING,
            self.FILE_FLAG_BACKUP_SEMANTICS | self.FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if handle == self.invalid_handle_value:
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        try:
            self.reject_unsafe(handle, is_directory=True)
            expected = os.path.normcase(
                os.path.normpath(os.path.abspath(os.fspath(path)))
            )
            if self.normalized_handle_path(handle) != expected:
                raise OSError(
                    "refusing a usd-cli state directory reached through a reparse point"
                )
        except BaseException:
            self.close_handle(handle)
            raise
        return handle

    def open_relative(
        self,
        parent: int,
        component: str,
        *,
        is_directory: bool,
        readable: bool = False,
        writable: bool = False,
        append: bool = False,
        create: bool = False,
        exclusive: bool = False,
    ) -> int:
        _validate_component(component)
        buffer = self.ctypes.create_unicode_buffer(component)
        encoded_length = len(component.encode("utf-16-le"))
        name = self.UnicodeString(
            Length=encoded_length,
            MaximumLength=encoded_length + self.ctypes.sizeof(self.wintypes.WCHAR),
            Buffer=self.ctypes.cast(buffer, self.wintypes.LPWSTR),
        )
        attributes = self.ObjectAttributes(
            Length=self.ctypes.sizeof(self.ObjectAttributes),
            RootDirectory=parent,
            ObjectName=self.ctypes.pointer(name),
            Attributes=self.OBJ_CASE_INSENSITIVE,
            SecurityDescriptor=None,
            SecurityQualityOfService=None,
        )
        io_status = self.IoStatusBlock()
        handle = self.wintypes.HANDLE()
        access = self.FILE_READ_ATTRIBUTES | self.SYNCHRONIZE
        if is_directory:
            access |= self.FILE_LIST_DIRECTORY
        else:
            if readable:
                access |= self.FILE_READ_DATA
            if writable:
                access |= self.FILE_WRITE_DATA
            if append:
                access |= self.FILE_APPEND_DATA
        options = self.FILE_OPEN_REPARSE_POINT | self.FILE_SYNCHRONOUS_IO_NONALERT
        options |= self.FILE_DIRECTORY_FILE if is_directory else self.FILE_NON_DIRECTORY_FILE
        disposition = (
            self.FILE_CREATE
            if exclusive
            else self.FILE_OPEN_IF
            if create
            else self.FILE_OPEN
        )
        status_value = self.nt_create_file(
            self.ctypes.byref(handle),
            access,
            self.ctypes.byref(attributes),
            self.ctypes.byref(io_status),
            None,
            self.FILE_ATTRIBUTE_NORMAL,
            self.FILE_SHARE_READ | self.FILE_SHARE_WRITE,
            disposition,
            options,
            None,
            0,
        )
        if status_value < 0:
            error = int(self.rtl_nt_status_to_dos_error(status_value))
            raise self.ctypes.WinError(error)
        try:
            self.reject_unsafe(handle.value, is_directory=is_directory)
        except BaseException:
            self.close_handle(handle.value)
            raise
        return handle.value


@lru_cache(maxsize=1)
def _windows_api() -> _WindowsFileApi:
    if os.name != "nt":
        raise OSError("Windows handle confinement is unavailable")
    return _WindowsFileApi()


def _validate_component(component: str) -> None:
    if (
        not component
        or component in {".", ".."}
        or any(separator in component for separator in ("/", "\\"))
        or ":" in component
    ):
        raise ValueError("filename must be one canonical path component")


@contextmanager
def open_confined_directory(directory: str | Path) -> Iterator[int]:
    """Hold a directory and every traversed component without following reparses.

    The returned CRT descriptor can be used with :func:`os.fstat`.  All Windows
    handles intentionally omit ``FILE_SHARE_DELETE`` so directory replacement is
    denied for the entire context, providing the stable anchor that POSIX obtains
    from an open directory descriptor plus ``flock``.
    """

    import msvcrt

    api = _windows_api()
    absolute = Path(os.path.abspath(os.fspath(directory)))
    if not absolute.anchor:
        raise ValueError("state directory must be absolute")
    handles = [api.open_absolute_directory(Path(absolute.anchor))]
    descriptor = -1
    try:
        try:
            for component in absolute.parts[1:]:
                handles.append(
                    api.open_relative(handles[-1], component, is_directory=True)
                )
        except PermissionError:
            for handle in reversed(handles):
                api.close_handle(handle)
            handles.clear()
            handles.append(api.open_absolute_directory(absolute))
        final_handle = handles.pop()
        try:
            descriptor = msvcrt.open_osfhandle(
                final_handle,
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOINHERIT", 0),
            )
        except BaseException:
            api.close_handle(final_handle)
            raise
        yield descriptor
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for handle in reversed(handles):
            api.close_handle(handle)


def confined_directory_path(directory_descriptor: int) -> Path:
    """Return the verified current path of a confined Windows directory handle."""

    import msvcrt

    api = _windows_api()
    handle = msvcrt.get_osfhandle(directory_descriptor)
    api.reject_unsafe(handle, is_directory=True)
    return Path(api.normalized_handle_path(handle))


def open_confined_regular_file_at(
    directory_descriptor: int,
    filename: str,
    *,
    readable: bool = False,
    writable: bool = False,
    append: bool = False,
    create: bool = False,
    exclusive: bool = False,
) -> int:
    """Open one single-link non-reparse leaf relative to a held directory."""

    import msvcrt

    if exclusive and not create:
        raise ValueError("exclusive file open requires create=True")
    api = _windows_api()
    parent_handle = msvcrt.get_osfhandle(directory_descriptor)
    api.reject_unsafe(parent_handle, is_directory=True)
    leaf = api.open_relative(
        parent_handle,
        filename,
        is_directory=False,
        readable=readable,
        writable=writable,
        append=append,
        create=create,
        exclusive=exclusive,
    )
    if readable and (writable or append):
        access_flags = os.O_RDWR
    elif writable or append:
        access_flags = os.O_WRONLY
    else:
        access_flags = os.O_RDONLY
    access_flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    if append:
        access_flags |= os.O_APPEND
    try:
        descriptor = msvcrt.open_osfhandle(leaf, access_flags)
    except BaseException:
        api.close_handle(leaf)
        raise
    try:
        require_confined_regular_file(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def require_confined_regular_file(descriptor: int) -> os.stat_result:
    """Validate that a held Windows file is still regular and single-linked."""

    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise OSError("usd-cli state file must be a single-link regular file")
    return metadata


def read_confined_regular_file_at(
    directory_descriptor: int,
    filename: str,
    *,
    max_bytes: int,
) -> bytes:
    """Read one bounded file relative to an already confined directory."""

    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    descriptor = open_confined_regular_file_at(
        directory_descriptor,
        filename,
        readable=True,
    )
    try:
        before = require_confined_regular_file(descriptor)
        if before.st_size > max_bytes:
            raise OSError("unsafe or oversized usd-cli state file")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes - size + 1))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise OSError("usd-cli state file exceeds its size limit")
        after = require_confined_regular_file(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or size != after.st_size
        ):
            raise OSError("usd-cli state file changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def replace_confined_regular_file_at(
    directory_descriptor: int,
    filename: str,
    payload: bytes,
    *,
    max_bytes: int,
    remove_if_empty: bool = False,
) -> None:
    """Atomically replace one bounded state file beneath a confined directory."""

    _validate_component(filename)
    if len(payload) > max_bytes:
        raise OSError("usd-cli state file exceeds its size limit")
    directory = confined_directory_path(directory_descriptor)
    temporary_name = f".{filename}.{secrets.token_hex(8)}.tmp"
    temporary_path = directory / temporary_name
    descriptor = open_confined_regular_file_at(
        directory_descriptor,
        temporary_name,
        writable=True,
        create=True,
        exclusive=True,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("usd-cli state write made no forward progress")
            view = view[written:]
        os.fsync(descriptor)
        metadata = require_confined_regular_file(descriptor)
        if metadata.st_size != len(payload):
            raise OSError("usd-cli state write was incomplete")
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary_path, directory / filename)
        temporary_path = Path()
        installed = read_confined_regular_file_at(
            directory_descriptor,
            filename,
            max_bytes=max_bytes,
        )
        if installed != payload:
            raise OSError("usd-cli state file changed during publication")
        if remove_if_empty and not payload:
            (directory / filename).unlink()
    finally:
        if temporary_path != Path():
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


@contextmanager
def advisory_file_lock(descriptor: int) -> Iterator[None]:
    """Take a blocking one-byte advisory lock on a portable CRT descriptor."""

    if os.name != "nt":
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        return
    import errno
    import msvcrt

    retry_errors = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
    original_position = os.lseek(descriptor, 0, os.SEEK_CUR)
    while True:
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            break
        except OSError as exc:
            if exc.errno not in retry_errors:
                raise
            time.sleep(0.01)
    try:
        yield
    finally:
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        os.lseek(descriptor, original_position, os.SEEK_SET)


def read_confined_regular_file(
    directory: str | Path,
    filename: str,
    *,
    max_bytes: int,
) -> bytes:
    """Read one single-link file through a no-reparse Windows handle chain."""

    with open_confined_directory(directory) as directory_descriptor:
        return read_confined_regular_file_at(
            directory_descriptor,
            filename,
            max_bytes=max_bytes,
        )
