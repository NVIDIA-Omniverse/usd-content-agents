# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable USD export helpers for ABI-isolated Scene Optimizer workers.

This module is copied beside each worker and runs with the Scene Optimizer
bundle's ``pxr`` bindings. Keep it limited to the Python standard library and
``pxr``; importing the application package in that subprocess would break its
OpenUSD ABI isolation.
"""

import ctypes
import hashlib
import os
import secrets
import shutil
import stat
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_LOCALIZABLE_USD_LAYER_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
PORTABLE_SIDECAR_MARKER_NAME = ".usd_portable_sidecar"
PORTABLE_SIDECAR_MARKER_BYTES = b"world-understanding portable USD sidecar v1\n"
_RENAME_NOREPLACE = 1
_SUPPORTS_DIRECTORY_DESCRIPTORS = os.name == "posix"
_HOST_IS_NATIVE_WINDOWS = os.name == "nt"
_PATH_TRANSACTION_DIRECTORY_MODE = None if os.name == "nt" else 0o700
_BINARY_OPEN_FLAG = getattr(os, "O_BINARY", 0)
_WINDOWS_RESERVED_FILE_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{suffix}" for suffix in "123456789¹²³"}
    | {f"LPT{suffix}" for suffix in "123456789¹²³"}
)


class _WindowsDirectory:
    """A directory pinned by a no-delete Windows native handle chain."""

    __slots__ = ("_owned_handles", "handle", "path")

    def __init__(self, path: Path, handles: tuple[int, ...]) -> None:
        if not handles:
            raise ValueError("Windows directory requires a held handle")
        self.path = path
        self._owned_handles = handles
        self.handle = handles[-1]

    def close(self) -> None:
        first_error: OSError | None = None
        while self._owned_handles:
            handle, self._owned_handles = (
                self._owned_handles[-1],
                self._owned_handles[:-1],
            )
            if not _WINDOWS_CLOSE_HANDLE(handle) and first_error is None:
                first_error = ctypes.WinError(ctypes.get_last_error())
        if first_error is not None:
            raise first_error


if os.name == "nt":  # pragma: win32 cover
    import msvcrt
    from ctypes import wintypes

    _FILE_LIST_DIRECTORY = 0x0001
    _FILE_READ_DATA = 0x0001
    _FILE_READ_ATTRIBUTES = 0x0080
    _FILE_WRITE_ATTRIBUTES = 0x0100
    _DELETE = 0x00010000
    _SYNCHRONIZE = 0x00100000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_OPEN = 0x00000001
    _FILE_CREATE = 0x00000002
    _FILE_OPEN_IF = 0x00000003
    _FILE_DIRECTORY_FILE = 0x00000001
    _FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
    _FILE_NON_DIRECTORY_FILE = 0x00000040
    _FILE_OPEN_REPARSE_POINT = 0x00200000
    _FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    _FILE_ATTRIBUTE_NORMAL = 0x00000080
    _FILE_ATTRIBUTE_READONLY = 0x00000001
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _OPEN_EXISTING = 3
    _OBJ_CASE_INSENSITIVE = 0x00000040
    _FILE_BASIC_INFO_CLASS = 0
    _FILE_RENAME_INFO_CLASS = 3
    _FILE_DISPOSITION_INFO_CLASS = 4

    class _UnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        ]

    class _ObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(_UnicodeString)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", wintypes.LPVOID),
            ("SecurityQualityOfService", wintypes.LPVOID),
        ]

    class _IoStatusValue(ctypes.Union):
        _fields_ = [("Status", ctypes.c_long), ("Pointer", wintypes.LPVOID)]

    class _IoStatusBlock(ctypes.Structure):
        _anonymous_ = ("value",)
        _fields_ = [("value", _IoStatusValue), ("Information", ctypes.c_size_t)]

    class _ByHandleFileInformation(ctypes.Structure):
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

    class _FileId128(ctypes.Structure):
        _fields_ = [("Identifier", ctypes.c_ubyte * 16)]

    class _FileIdInformation(ctypes.Structure):
        _fields_ = [
            ("VolumeSerialNumber", ctypes.c_ulonglong),
            ("FileId", _FileId128),
        ]

    class _FileRenameInformation(ctypes.Structure):
        _fields_ = [
            ("ReplaceIfExists", wintypes.BOOLEAN),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]

    class _FileDispositionInformation(ctypes.Structure):
        _fields_ = [("DeleteFile", wintypes.BOOLEAN)]

    class _FileBasicInformation(ctypes.Structure):
        _fields_ = [
            ("CreationTime", ctypes.c_longlong),
            ("LastAccessTime", ctypes.c_longlong),
            ("LastWriteTime", ctypes.c_longlong),
            ("ChangeTime", ctypes.c_longlong),
            ("FileAttributes", wintypes.DWORD),
        ]

    _WINDOWS_NTDLL = ctypes.WinDLL("ntdll", use_last_error=True)
    _WINDOWS_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _WINDOWS_NT_CREATE_FILE = _WINDOWS_NTDLL.NtCreateFile
    _WINDOWS_NT_CREATE_FILE.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(_ObjectAttributes),
        ctypes.POINTER(_IoStatusBlock),
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.LPVOID,
        wintypes.ULONG,
    ]
    _WINDOWS_NT_CREATE_FILE.restype = ctypes.c_long
    _WINDOWS_STATUS_TO_ERROR = _WINDOWS_NTDLL.RtlNtStatusToDosError
    _WINDOWS_STATUS_TO_ERROR.argtypes = [ctypes.c_long]
    _WINDOWS_STATUS_TO_ERROR.restype = wintypes.ULONG
    _WINDOWS_CREATE_FILE = _WINDOWS_KERNEL32.CreateFileW
    _WINDOWS_CREATE_FILE.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _WINDOWS_CREATE_FILE.restype = wintypes.HANDLE
    _WINDOWS_CLOSE_HANDLE = _WINDOWS_KERNEL32.CloseHandle
    _WINDOWS_CLOSE_HANDLE.argtypes = [wintypes.HANDLE]
    _WINDOWS_CLOSE_HANDLE.restype = wintypes.BOOL
    _WINDOWS_GET_FILE_INFORMATION = _WINDOWS_KERNEL32.GetFileInformationByHandle
    _WINDOWS_GET_FILE_INFORMATION.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    ]
    _WINDOWS_GET_FILE_INFORMATION.restype = wintypes.BOOL
    _WINDOWS_GET_FILE_INFORMATION_EX = _WINDOWS_KERNEL32.GetFileInformationByHandleEx
    _WINDOWS_GET_FILE_INFORMATION_EX.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _WINDOWS_GET_FILE_INFORMATION_EX.restype = wintypes.BOOL
    _WINDOWS_GET_FINAL_PATH_NAME = _WINDOWS_KERNEL32.GetFinalPathNameByHandleW
    _WINDOWS_GET_FINAL_PATH_NAME.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    _WINDOWS_GET_FINAL_PATH_NAME.restype = wintypes.DWORD
    _WINDOWS_SET_FILE_INFORMATION = _WINDOWS_KERNEL32.SetFileInformationByHandle
    _WINDOWS_SET_FILE_INFORMATION.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _WINDOWS_SET_FILE_INFORMATION.restype = wintypes.BOOL
    _WINDOWS_INVALID_HANDLE = wintypes.HANDLE(-1).value
else:
    _WINDOWS_CLOSE_HANDLE = None


def _load_process_libc() -> Any:
    """Return the process' libc handle, or ``None`` off POSIX.

    ``ctypes.CDLL(None)`` is the POSIX ``dlopen(NULL)`` idiom for the main
    program handle. It raises ``TypeError`` on Windows, so importing this
    module would fail there before any caller reaches the Linux-only
    ``renameat2`` path guarded below.
    """
    if os.name != "posix":  # pragma: no cover - no libc handle to open
        return None
    try:
        return ctypes.CDLL(None, use_errno=True)
    except OSError:  # pragma: no cover - libc is always loadable on POSIX
        return None


_RENAMEAT2: Any = getattr(_load_process_libc(), "renameat2", None)


class _PathDirectory:
    """A verified directory used where Python exposes no ``dir_fd`` support.

    Windows cannot express the POSIX descriptor-confinement guarantee through
    the standard library. This backend rejects every observed symlink and
    reparse point and revalidates directory identities around mutations, but
    validation and use remain separate operations. The native Windows support
    plan records that unavoidable TOCTOU gap.

    Keeping this distinct from ``int`` makes any missed descriptor-only call
    fail loudly instead of silently treating an unrelated integer as a handle.
    """

    __slots__ = ("metadata", "path")

    def __init__(self, path: Path, metadata: os.stat_result) -> None:
        self.path = path
        self.metadata = metadata

    def __index__(self) -> int:
        raise TypeError("Path-backed USD export directories are not file descriptors")


_DirectoryHandle = int | _WindowsDirectory | _PathDirectory


class _OutputCommitRaceError(RuntimeError):
    """The destination changed during commit and the transaction is retained."""


def _lexical_absolute_path(path: str | os.PathLike[str]) -> Path:
    """Return an absolute path without resolving filesystem symlinks."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _windows_raise_status(status: int) -> None:
    error = int(_WINDOWS_STATUS_TO_ERROR(status))
    if error in {2, 3}:
        raise FileNotFoundError(error, os.strerror(error))
    if error in {80, 183}:
        raise FileExistsError(error, os.strerror(error))
    raise ctypes.WinError(error)


def _windows_validate_component(component: str) -> None:
    windows_forbidden = '<>:"/\\|?*'
    if (
        not component
        or component in {".", ".."}
        or any(character in windows_forbidden for character in component)
        or any(ord(character) < 32 for character in component)
        or component[-1] in {" ", "."}
        or component.partition(".")[0].rstrip(" ").upper()
        in _WINDOWS_RESERVED_FILE_STEMS
    ):
        raise ValueError("USD path contains a non-canonical Windows component")


def _windows_information(handle: int) -> Any:
    information = _ByHandleFileInformation()
    if not _WINDOWS_GET_FILE_INFORMATION(handle, ctypes.byref(information)):
        raise ctypes.WinError(ctypes.get_last_error())
    return information


def _windows_volume_serial_number(handle: int) -> int:
    information = _FileIdInformation()
    # FileIdInfo from FILE_INFO_BY_HANDLE_CLASS. Python's Windows ``st_dev``
    # uses this 64-bit volume serial rather than the legacy DWORD field in
    # BY_HANDLE_FILE_INFORMATION.
    if not _WINDOWS_GET_FILE_INFORMATION_EX(
        handle,
        18,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(information.VolumeSerialNumber)


def _windows_metadata(handle: int) -> os.stat_result:
    information = _windows_information(handle)
    attributes = information.FileAttributes
    if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise RuntimeError("Refusing output path containing a Windows reparse point")
    mode = (
        stat.S_IFDIR | 0o700
        if attributes & _FILE_ATTRIBUTE_DIRECTORY
        else stat.S_IFREG | 0o600
    )
    inode = (int(information.FileIndexHigh) << 32) | int(information.FileIndexLow)
    size = (int(information.FileSizeHigh) << 32) | int(information.FileSizeLow)
    return os.stat_result(
        (
            mode,
            inode,
            _windows_volume_serial_number(handle),
            int(information.NumberOfLinks),
            0,
            0,
            size,
            0,
            0,
            0,
        )
    )


def _windows_normalized_handle_path(handle: int) -> str:
    """Return the canonical DOS path for a pinned Windows handle."""
    required = _WINDOWS_GET_FINAL_PATH_NAME(handle, None, 0, 0)
    if required == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_unicode_buffer(required + 1)
    written = _WINDOWS_GET_FINAL_PATH_NAME(handle, buffer, len(buffer), 0)
    if written == 0 or written >= len(buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.normpath(os.path.abspath(value)))


def _windows_open_verified_absolute_directory_handle(directory: Path) -> int:
    """Pin an exact directory when a restricted ancestor cannot be listed."""
    handle = _WINDOWS_CREATE_FILE(
        str(directory),
        _FILE_LIST_DIRECTORY | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _WINDOWS_INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        _windows_metadata(handle)
        expected = os.path.normcase(os.path.normpath(os.path.abspath(directory)))
        if _windows_normalized_handle_path(handle) != expected:
            raise RuntimeError(
                "Refusing a Windows directory reached through a reparse point"
            )
    except BaseException:
        _WINDOWS_CLOSE_HANDLE(handle)
        raise
    return handle


def _windows_open_relative_handle(
    parent: _WindowsDirectory,
    component: str,
    *,
    directory: bool | None,
    create: bool,
    exclusive_create: bool = False,
    access: int | None = None,
) -> int:
    _windows_validate_component(component)
    buffer = ctypes.create_unicode_buffer(component)
    encoded_length = len(component.encode("utf-16-le"))
    name = _UnicodeString(
        Length=encoded_length,
        MaximumLength=encoded_length + ctypes.sizeof(wintypes.WCHAR),
        Buffer=ctypes.cast(buffer, wintypes.LPWSTR),
    )
    attributes = _ObjectAttributes(
        Length=ctypes.sizeof(_ObjectAttributes),
        RootDirectory=parent.handle,
        ObjectName=ctypes.pointer(name),
        Attributes=_OBJ_CASE_INSENSITIVE,
        SecurityDescriptor=None,
        SecurityQualityOfService=None,
    )
    io_status = _IoStatusBlock()
    handle = wintypes.HANDLE()
    desired_access = access
    if desired_access is None:
        desired_access = _FILE_LIST_DIRECTORY if directory else _FILE_READ_DATA
    desired_access |= _FILE_READ_ATTRIBUTES | _SYNCHRONIZE
    options = _FILE_OPEN_REPARSE_POINT | _FILE_SYNCHRONOUS_IO_NONALERT
    if directory is True:
        options |= _FILE_DIRECTORY_FILE
    elif directory is False:
        options |= _FILE_NON_DIRECTORY_FILE
    status_code = _WINDOWS_NT_CREATE_FILE(
        ctypes.byref(handle),
        desired_access,
        ctypes.byref(attributes),
        ctypes.byref(io_status),
        None,
        _FILE_ATTRIBUTE_NORMAL,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        _FILE_CREATE if exclusive_create else (_FILE_OPEN_IF if create else _FILE_OPEN),
        options,
        None,
        0,
    )
    if status_code < 0:
        _windows_raise_status(status_code)
    try:
        metadata = _windows_metadata(handle.value)
        if directory is True and not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(component)
        if directory is False and not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("Refusing to open a non-regular USD artifact")
    except BaseException:
        _WINDOWS_CLOSE_HANDLE(handle.value)
        raise
    return handle.value


def _windows_open_absolute_directory(path: Path, *, create: bool) -> _WindowsDirectory:
    directory = _lexical_absolute_path(path)
    if not directory.anchor:
        raise ValueError("USD output directory must be absolute")
    anchor_handle = _WINDOWS_CREATE_FILE(
        directory.anchor,
        _FILE_LIST_DIRECTORY | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if anchor_handle == _WINDOWS_INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    handles = [anchor_handle]
    current = Path(directory.anchor)
    try:
        anchor_metadata = _windows_metadata(anchor_handle)
        if not stat.S_ISDIR(anchor_metadata.st_mode):
            raise NotADirectoryError(directory.anchor)
        try:
            for component in directory.parts[1:]:
                parent = _WindowsDirectory(current, (handles[-1],))
                handle = _windows_open_relative_handle(
                    parent,
                    component,
                    directory=True,
                    create=create,
                )
                handles.append(handle)
                current /= component
        except PermissionError:
            # Restricted Windows tokens can traverse an inherited profile
            # ancestor without permission to list/open it. Pin and verify the
            # exact writable directory before beginning relative operations.
            for handle in reversed(handles):
                _WINDOWS_CLOSE_HANDLE(handle)
            handles = []
            if create:
                directory.mkdir(parents=True, exist_ok=True)
            handles.append(_windows_open_verified_absolute_directory_handle(directory))
        result = _WindowsDirectory(directory, tuple(handles))
        handles = []
        return result
    except BaseException as exc:
        for handle in reversed(handles):
            _WINDOWS_CLOSE_HANDLE(handle)
        if isinstance(exc, FileNotFoundError | ValueError):
            raise
        raise RuntimeError(
            "Refusing output path with a reparse point or non-directory ancestor: "
            f"{directory}"
        ) from exc


def _open_child_directory(
    parent: _DirectoryHandle,
    name: str,
    *,
    create: bool,
    exclusive_create: bool = False,
    delete_access: bool = False,
) -> _DirectoryHandle:
    if isinstance(parent, _WindowsDirectory):
        handle = _windows_open_relative_handle(
            parent,
            name,
            directory=True,
            create=create,
            exclusive_create=exclusive_create,
            access=_FILE_LIST_DIRECTORY | (_DELETE if delete_access else 0),
        )
        return _WindowsDirectory(parent.path / name, (handle,))
    if exclusive_create:
        os.mkdir(name, mode=0o700, dir_fd=parent)
    elif create:
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent)
        except FileExistsError:
            pass
    return os.open(name, _directory_open_flags(), dir_fd=parent)


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _is_reparse_point(metadata: os.stat_result) -> bool:
    """Return whether metadata identifies a symlink or Windows reparse point."""

    return bool(
        stat.S_ISLNK(metadata.st_mode)
        or (getattr(metadata, "st_file_attributes", 0) or 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _reject_path_reparse_points(path: Path, *, allow_missing: bool = False) -> None:
    """Reject every observed symlink or reparse point in an absolute path."""

    current = Path(path.anchor)
    components = [current]
    for part in path.parts[1:]:
        components.append(components[-1] / part)
    for current in components:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if allow_missing:
                return
            raise
        except OSError as exc:
            raise RuntimeError(
                f"Cannot inspect USD export path component: {current}"
            ) from exc
        if _is_reparse_point(metadata):
            raise RuntimeError(
                f"Refusing USD export path with a symlink or reparse point: {current}"
            )


def _open_directory_nofollow(
    path: Path,
    *,
    create: bool,
) -> _DirectoryHandle:
    """Open an absolute directory path without following any symlink."""
    directory = _lexical_absolute_path(path)
    if _HOST_IS_NATIVE_WINDOWS:
        return _windows_open_absolute_directory(directory, create=create)
    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        _reject_path_reparse_points(directory, allow_missing=create)
        if create:
            directory.mkdir(mode=0o755, parents=True, exist_ok=True)
        _reject_path_reparse_points(directory)
        metadata = directory.lstat()
        if _is_reparse_point(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"USD export parent is not a directory: {directory}")
        return _PathDirectory(directory, metadata)

    descriptor = os.open(directory.anchor, _directory_open_flags())
    try:
        for part in directory.parts[1:]:
            try:
                child_descriptor = os.open(
                    part,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child_descriptor = os.open(
                    part,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = child_descriptor
    except OSError as exc:
        os.close(descriptor)
        raise RuntimeError(
            "Refusing output path with a symlink or non-directory ancestor: "
            f"{directory}"
        ) from exc
    return descriptor


def _entry_metadata(
    directory_descriptor: _DirectoryHandle,
    name: str,
) -> os.stat_result | None:
    if isinstance(directory_descriptor, _WindowsDirectory):
        try:
            handle = _windows_open_relative_handle(
                directory_descriptor,
                name,
                directory=None,
                create=False,
                access=_FILE_READ_ATTRIBUTES,
            )
        except FileNotFoundError:
            return None
        try:
            return _windows_metadata(handle)
        finally:
            _WINDOWS_CLOSE_HANDLE(handle)
    if isinstance(directory_descriptor, _PathDirectory):
        child = _path_child(directory_descriptor, name)
        try:
            return child.lstat()
        except FileNotFoundError:
            return None
    try:
        return os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _unused_entry_name(
    directory_descriptor: _DirectoryHandle,
    prefix: str,
) -> str:
    """Return an unpredictable unused leaf name in a private directory."""
    for _ in range(100):
        name = f"{prefix}{secrets.token_hex(8)}"
        if _entry_metadata(directory_descriptor, name) is None:
            return name
    raise RuntimeError("Failed to allocate a unique USD transaction entry")


def _same_entry(
    expected: os.stat_result,
    actual: os.stat_result | None,
) -> bool:
    return bool(
        actual is not None
        and expected.st_dev == actual.st_dev
        and expected.st_ino == actual.st_ino
        and stat.S_IFMT(expected.st_mode) == stat.S_IFMT(actual.st_mode)
    )


def _path_directory_metadata(directory: _PathDirectory) -> os.stat_result:
    """Revalidate and return one path-backed directory identity."""

    try:
        _reject_path_reparse_points(directory.path)
        metadata = directory.path.lstat()
    except (OSError, RuntimeError) as exc:
        raise _OutputCommitRaceError(
            f"USD export directory changed or disappeared during use: {directory.path}"
        ) from exc
    if (
        _is_reparse_point(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
        or not _same_entry(
            directory.metadata,
            metadata,
        )
    ):
        raise _OutputCommitRaceError(
            f"USD export directory changed during use: {directory.path}"
        )
    return metadata


def _directory_metadata(
    directory: _DirectoryHandle,
) -> os.stat_result:
    if isinstance(directory, _WindowsDirectory):
        return _windows_metadata(directory.handle)
    if isinstance(directory, _PathDirectory):
        return _path_directory_metadata(directory)
    return os.fstat(directory)


def _close_directory(directory: _DirectoryHandle) -> None:
    if isinstance(directory, _WindowsDirectory):
        directory.close()
    elif not isinstance(directory, _PathDirectory):
        os.close(directory)


def _path_child(directory: _PathDirectory, name: str) -> Path:
    """Return one validated leaf beneath a current path-backed directory."""

    _path_directory_metadata(directory)
    candidate = Path(name)
    windows_forbidden = '<>:"/\\|?*'
    if (
        not name
        or candidate.name != name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or any(character in windows_forbidden for character in name)
        or any(ord(character) < 32 for character in name)
        or name.endswith((".", " "))
        or name.partition(".")[0].rstrip(" ").upper() in _WINDOWS_RESERVED_FILE_STEMS
    ):
        raise RuntimeError(f"Unsafe USD export transaction entry: {name!r}")
    return directory.path / name


def _rename_noreplace(
    source_descriptor: _DirectoryHandle,
    source_name: str,
    destination_descriptor: _DirectoryHandle,
    destination_name: str,
) -> None:
    """Atomically rename one entry without overwriting a concurrent entry."""
    if isinstance(source_descriptor, _WindowsDirectory) and isinstance(
        destination_descriptor, _WindowsDirectory
    ):
        metadata = _entry_metadata(source_descriptor, source_name)
        if metadata is None:
            raise FileNotFoundError(source_name)
        handle = _windows_open_relative_handle(
            source_descriptor,
            source_name,
            directory=stat.S_ISDIR(metadata.st_mode),
            create=False,
            access=_DELETE,
        )
        try:
            encoded_name = str(destination_descriptor.path / destination_name).encode(
                "utf-16-le"
            )
            size = (
                _FileRenameInformation.FileName.offset
                + len(encoded_name)
                + ctypes.sizeof(wintypes.WCHAR)
            )
            buffer = ctypes.create_string_buffer(size)
            information = ctypes.cast(
                buffer,
                ctypes.POINTER(_FileRenameInformation),
            ).contents
            information.ReplaceIfExists = False
            # SetFileInformationByHandle accepts an absolute target with a
            # null RootDirectory on every supported Windows version. The
            # destination ancestor chain remains pinned by native handles, and
            # ReplaceIfExists=False provides the no-clobber commit primitive.
            information.RootDirectory = None
            information.FileNameLength = len(encoded_name)
            ctypes.memmove(
                ctypes.addressof(buffer) + _FileRenameInformation.FileName.offset,
                encoded_name,
                len(encoded_name),
            )
            if not _WINDOWS_SET_FILE_INFORMATION(
                handle,
                _FILE_RENAME_INFO_CLASS,
                buffer,
                size,
            ):
                error = ctypes.get_last_error()
                if error in {80, 183}:
                    raise FileExistsError(error, os.strerror(error), destination_name)
                raise ctypes.WinError(error)
        finally:
            _WINDOWS_CLOSE_HANDLE(handle)
        return
    if isinstance(source_descriptor, _WindowsDirectory) or isinstance(
        destination_descriptor, _WindowsDirectory
    ):
        raise TypeError("Cannot mix Windows handles and other export directories")
    if isinstance(source_descriptor, _PathDirectory) or isinstance(
        destination_descriptor,
        _PathDirectory,
    ):
        if not isinstance(source_descriptor, _PathDirectory) or not isinstance(
            destination_descriptor,
            _PathDirectory,
        ):
            raise RuntimeError("Cannot mix descriptor and path export directories")
        source = _path_child(source_descriptor, source_name)
        destination = _path_child(destination_descriptor, destination_name)
        if _entry_metadata(destination_descriptor, destination_name) is not None:
            raise FileExistsError(
                f"USD export destination already exists: {destination}"
            )
        # Windows os.rename is a same-volume, atomic, no-replace move. The
        # existence check is still required for the Linux-hosted path-backend
        # tests, where os.rename would otherwise replace a file.
        os.rename(source, destination)
        return
    if _RENAMEAT2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required on Linux")
    _RENAMEAT2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    _RENAMEAT2.restype = ctypes.c_int
    result = _RENAMEAT2(
        source_descriptor,
        os.fsencode(source_name),
        destination_descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            source_name,
            destination_name,
        )


def _restore_checked_entry(
    source_descriptor: _DirectoryHandle,
    source_name: str,
    destination_descriptor: _DirectoryHandle,
    destination_name: str,
    expected: os.stat_result,
    description: str,
) -> None:
    """Restore a moved entry only while its inode and destination remain safe."""
    if not _same_entry(
        expected,
        _entry_metadata(source_descriptor, source_name),
    ):
        raise _OutputCommitRaceError(
            f"Cannot safely restore {description}; its moved inode changed"
        )
    if _entry_metadata(destination_descriptor, destination_name) is not None:
        raise _OutputCommitRaceError(
            f"Cannot safely restore {description}; destination was recreated"
        )
    _rename_noreplace(
        source_descriptor,
        source_name,
        destination_descriptor,
        destination_name,
    )
    if not _same_entry(
        expected,
        _entry_metadata(destination_descriptor, destination_name),
    ):
        raise _OutputCommitRaceError(
            f"Cannot verify restored {description}; transaction was preserved"
        )


def _move_checked_entry(
    source_descriptor: _DirectoryHandle,
    source_name: str,
    destination_descriptor: _DirectoryHandle,
    destination_name: str,
    expected: os.stat_result,
    description: str,
) -> os.stat_result:
    """Move exactly the prevalidated inode or restore and fail closed."""
    if _entry_metadata(destination_descriptor, destination_name) is not None:
        raise _OutputCommitRaceError(
            f"Cannot move {description}; transaction destination already exists"
        )
    try:
        _rename_noreplace(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )
        moved = _entry_metadata(destination_descriptor, destination_name)
    except BaseException as move_error:
        try:
            source_after = _entry_metadata(source_descriptor, source_name)
            destination_after = _entry_metadata(
                destination_descriptor,
                destination_name,
            )
        except BaseException as inspection_error:
            raise _OutputCommitRaceError(
                f"Cannot inspect {description} after an interrupted commit; "
                "transaction was preserved"
            ) from inspection_error
        if _same_entry(expected, destination_after) and source_after is None:
            try:
                _restore_checked_entry(
                    destination_descriptor,
                    destination_name,
                    source_descriptor,
                    source_name,
                    expected,
                    description,
                )
            except BaseException as restore_error:
                raise _OutputCommitRaceError(
                    f"{description} moved during an interrupted commit and could "
                    "not be restored; transaction was preserved"
                ) from restore_error
            raise
        if _same_entry(expected, source_after) and destination_after is None:
            raise
        raise _OutputCommitRaceError(
            f"{description} changed during an interrupted commit; transaction was "
            "preserved"
        ) from move_error
    if _same_entry(expected, moved):
        assert moved is not None
        return moved

    if moved is not None:
        try:
            _restore_checked_entry(
                destination_descriptor,
                destination_name,
                source_descriptor,
                source_name,
                moved,
                description,
            )
        except BaseException as restore_error:
            raise _OutputCommitRaceError(
                f"{description} changed during commit and could not be restored; "
                "transaction was preserved"
            ) from restore_error
    raise _OutputCommitRaceError(
        f"{description} changed during commit; transaction was preserved"
    )


def _undo_checked_move(
    source_descriptor: _DirectoryHandle,
    source_name: str,
    destination_descriptor: _DirectoryHandle,
    destination_name: str,
    expected: os.stat_result,
    description: str,
) -> None:
    """Undo a pre-armed move whether it landed or was already restored."""
    try:
        source_metadata = _entry_metadata(source_descriptor, source_name)
        destination_metadata = _entry_metadata(
            destination_descriptor,
            destination_name,
        )
    except BaseException as inspection_error:
        raise _OutputCommitRaceError(
            f"Cannot inspect {description} during rollback; transaction was preserved"
        ) from inspection_error

    if _same_entry(expected, source_metadata) and destination_metadata is None:
        return
    if _same_entry(expected, destination_metadata) and source_metadata is None:
        _restore_checked_entry(
            destination_descriptor,
            destination_name,
            source_descriptor,
            source_name,
            expected,
            description,
        )
        return
    raise _OutputCommitRaceError(
        f"Cannot safely undo {description}; transaction was preserved"
    )


def _require_replaceable_output(
    directory_descriptor: _DirectoryHandle,
    output_name: str,
    display_path: Path,
) -> os.stat_result | None:
    """Return a regular output's identity, rejecting all other leaf types."""
    metadata = _entry_metadata(directory_descriptor, output_name)
    if metadata is None:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"Refusing to replace a symlink USD output: {display_path}")
    if _is_reparse_point(metadata):
        raise RuntimeError(
            f"Refusing to replace a reparse-point USD output: {display_path}"
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"Refusing to replace a non-file USD output: {display_path}")
    return metadata


def _create_transaction_directory(
    parent_descriptor: _DirectoryHandle,
    prefix: str,
) -> tuple[str, _DirectoryHandle, Path]:
    """Create an exclusive randomized transaction directory under a destination."""
    if isinstance(parent_descriptor, _PathDirectory):
        for _ in range(100):
            name = f"{prefix}{secrets.token_hex(8)}"
            transaction = _path_child(parent_descriptor, name)
            try:
                if _PATH_TRANSACTION_DIRECTORY_MODE is None:
                    # Windows' special 0o700 mode creates a protected DACL that
                    # survives rename and can make the published bundle
                    # unreadable to identities allowed by the destination.
                    # The default mkdir ACL inherits that destination access.
                    transaction.mkdir()
                else:
                    transaction.mkdir(mode=_PATH_TRANSACTION_DIRECTORY_MODE)
            except FileExistsError:
                continue
            directory = _open_directory_nofollow(transaction, create=False)
            assert isinstance(directory, _PathDirectory)
            return name, directory, transaction
        raise RuntimeError("Failed to allocate a unique USD output transaction")

    for _ in range(100):
        name = f"{prefix}{secrets.token_hex(8)}"
        try:
            descriptor = _open_child_directory(
                parent_descriptor,
                name,
                create=True,
                exclusive_create=True,
                delete_access=True,
            )
        except FileExistsError:
            continue
        transaction_path = (
            descriptor.path
            if isinstance(descriptor, _WindowsDirectory)
            else Path(f"/proc/self/fd/{descriptor}")
        )
        return name, descriptor, transaction_path
    raise RuntimeError("Failed to allocate a unique USD output transaction")


def _delete_entry(
    directory_descriptor: _DirectoryHandle,
    name: str,
    *,
    directory: bool,
) -> None:
    if isinstance(directory_descriptor, _WindowsDirectory):
        handle = _windows_open_relative_handle(
            directory_descriptor,
            name,
            directory=directory,
            create=False,
            access=_DELETE | (0 if directory else _FILE_WRITE_ATTRIBUTES),
        )
        try:
            if not directory:
                attributes = int(_windows_information(handle).FileAttributes)
                if attributes & _FILE_ATTRIBUTE_READONLY:
                    basic_information = _FileBasicInformation(
                        FileAttributes=(attributes & ~_FILE_ATTRIBUTE_READONLY)
                        or _FILE_ATTRIBUTE_NORMAL
                    )
                    if not _WINDOWS_SET_FILE_INFORMATION(
                        handle,
                        _FILE_BASIC_INFO_CLASS,
                        ctypes.byref(basic_information),
                        ctypes.sizeof(basic_information),
                    ):
                        raise ctypes.WinError(ctypes.get_last_error())
            information = _FileDispositionInformation(DeleteFile=True)
            if not _WINDOWS_SET_FILE_INFORMATION(
                handle,
                _FILE_DISPOSITION_INFO_CLASS,
                ctypes.byref(information),
                ctypes.sizeof(information),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _WINDOWS_CLOSE_HANDLE(handle)
        return
    if directory:
        os.rmdir(name, dir_fd=directory_descriptor)
    else:
        os.unlink(name, dir_fd=directory_descriptor)


def _windows_mark_handle_for_delete(handle: int) -> None:
    information = _FileDispositionInformation(DeleteFile=True)
    if not _WINDOWS_SET_FILE_INFORMATION(
        handle,
        _FILE_DISPOSITION_INFO_CLASS,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _clear_directory_contents(directory_descriptor: _DirectoryHandle) -> bool:
    """Clear an opened private directory without traversing a pathname."""
    if isinstance(directory_descriptor, _PathDirectory):
        try:
            _path_directory_metadata(directory_descriptor)
            entries = list(os.scandir(directory_descriptor.path))
        except (OSError, RuntimeError):
            return False

        for entry in entries:
            entry_path = Path(entry.path)
            try:
                # On Windows, DirEntry.stat() reports zero for st_dev/st_ino.
                # Path.lstat() returns the file identity needed by _same_entry.
                expected = entry_path.lstat()
            except OSError:
                return False
            if _is_reparse_point(expected):
                return False

            if stat.S_ISDIR(expected.st_mode):
                try:
                    child = _open_directory_nofollow(entry_path, create=False)
                except (OSError, RuntimeError):
                    return False
                assert isinstance(child, _PathDirectory)
                try:
                    if not _same_entry(expected, _directory_metadata(child)):
                        return False
                    if not _clear_directory_contents(child):
                        return False
                    if not _same_entry(
                        expected,
                        _entry_metadata(directory_descriptor, entry.name),
                    ):
                        return False
                    entry_path.rmdir()
                except (OSError, RuntimeError):
                    return False
                continue

            if not stat.S_ISREG(expected.st_mode) or not _same_entry(
                expected,
                _entry_metadata(directory_descriptor, entry.name),
            ):
                return False
            try:
                entry_path.unlink()
            except PermissionError:
                # A sidecar written by the older exporter may carry the
                # Windows read-only attribute from chmod(0444).
                try:
                    entry_path.chmod(stat.S_IWRITE)
                    entry_path.unlink()
                except OSError:
                    return False
            except OSError:
                return False
        return True

    try:
        entries = list(
            os.scandir(
                directory_descriptor.path
                if isinstance(directory_descriptor, _WindowsDirectory)
                else directory_descriptor
            )
        )
    except OSError:
        return False

    for entry in entries:
        try:
            expected = (
                _entry_metadata(directory_descriptor, entry.name)
                if isinstance(directory_descriptor, _WindowsDirectory)
                else entry.stat(follow_symlinks=False)
            )
        except (OSError, RuntimeError):
            return False
        if expected is None:
            return False

        if stat.S_ISDIR(expected.st_mode):
            try:
                child_descriptor = _open_child_directory(
                    directory_descriptor,
                    entry.name,
                    create=False,
                    delete_access=True,
                )
            except (OSError, RuntimeError):
                return False
            try:
                opened = _directory_metadata(child_descriptor)
                if not _same_entry(expected, opened):
                    return False
                if not _clear_directory_contents(child_descriptor):
                    return False
                if not _same_entry(
                    opened,
                    _entry_metadata(directory_descriptor, entry.name),
                ):
                    return False
                if isinstance(child_descriptor, _WindowsDirectory):
                    _windows_mark_handle_for_delete(child_descriptor.handle)
                else:
                    _delete_entry(
                        directory_descriptor,
                        entry.name,
                        directory=True,
                    )
            except (OSError, RuntimeError):
                return False
            finally:
                _close_directory(child_descriptor)
            continue

        try:
            if not _same_entry(
                expected,
                _entry_metadata(directory_descriptor, entry.name),
            ):
                return False
            _delete_entry(
                directory_descriptor,
                entry.name,
                directory=False,
            )
        except (OSError, RuntimeError):
            return False
    return True


def _cleanup_transaction_directory(
    parent_descriptor: _DirectoryHandle,
    transaction_name: str,
    transaction_descriptor: _DirectoryHandle,
) -> bool:
    """Remove only the opened transaction, preserving any swapped replacement."""
    try:
        expected = _directory_metadata(transaction_descriptor)
        if not _same_entry(
            expected,
            _entry_metadata(parent_descriptor, transaction_name),
        ):
            return False
    except (OSError, RuntimeError):
        return False
    if not _clear_directory_contents(transaction_descriptor):
        return False
    try:
        if not _same_entry(
            expected,
            _entry_metadata(parent_descriptor, transaction_name),
        ):
            return False
        if isinstance(transaction_descriptor, _WindowsDirectory):
            _windows_mark_handle_for_delete(transaction_descriptor.handle)
        elif isinstance(parent_descriptor, _PathDirectory):
            assert isinstance(transaction_descriptor, _PathDirectory)
            transaction_descriptor.path.rmdir()
        else:
            _delete_entry(parent_descriptor, transaction_name, directory=True)
    except (OSError, RuntimeError):
        return False
    return True


def portable_sidecar_name(output_path: str | os.PathLike[str]) -> str:
    """Return the collision-safe sidecar name for one portable USD output."""
    return f"{Path(output_path).name}_assets"


def legacy_portable_sidecar_name(output_path: str | os.PathLike[str]) -> str:
    """Return the pre-collision-fix sidecar name for compatibility readers."""
    return f"{Path(output_path).stem}_assets"


def export_layer_for(stage: Any) -> Any:
    """Return the layer ``stage`` should be exported through.

    ``GetRootLayer().Export`` writes only the root layer and does not re-anchor
    relative composition arcs. Because the workers write to a different
    directory, those arcs dangle and can silently remove all composed geometry.

    Flattening fixes that at the cost of collapsing authored composition such
    as variant sets. Preserve a single-layer stage and flatten only when another
    non-session layer contributes to the composed result. See issue #963.
    """
    composed_layers = [
        layer for layer in stage.GetUsedLayers() if layer is not stage.GetSessionLayer()
    ]
    return stage.Flatten() if len(composed_layers) > 1 else stage.GetRootLayer()


def is_bare_mdl_token(asset_path: object) -> bool:
    """Return whether ``asset_path`` is a runtime-resolved MDL module token."""
    try:
        from world_understanding.utils.usd.asset_paths import is_bare_mdl_asset_path
    except ModuleNotFoundError:
        # This module also runs inside the ABI-isolated Scene Optimizer
        # worker (`python -S`, replaced PYTHONPATH), where the package is
        # unimportable by design. Mirror is_bare_mdl_asset_path inline.
        token = str(asset_path).strip("@")
        return (
            bool(token)
            and ":" not in token
            and "/" not in token
            and "\\" not in token
            and Path(token).suffix.lower() == ".mdl"
        )

    return is_bare_mdl_asset_path(asset_path)


def is_runtime_resolved_asset_path(asset_path: object) -> bool:
    """Return whether an asset intentionally remains resolver-owned at runtime."""
    path_text = str(asset_path).strip("@")
    if is_bare_mdl_token(path_text):
        return True
    parsed = urlparse(path_text)
    # A one-letter scheme is a Windows drive, not a resolver URI.
    return bool(
        parsed.scheme and len(parsed.scheme) > 1 and parsed.scheme.lower() != "file"
    )


def _anchored_asset_path(layer: Any, asset_path: object) -> str:
    """Anchor one authored path to the layer that owns it."""
    from pxr import Sdf

    return str(Sdf.ComputeAssetPathRelativeToLayer(layer, str(asset_path)))


def _resolved_path_string(resolved_path: Any) -> str:
    getter = getattr(resolved_path, "GetPathString", None)
    return str(getter() if getter is not None else resolved_path)


def _outer_asset_identifier(asset_path: str) -> str:
    """Return the outer package/file identifier for one resolved asset."""
    from pxr import Ar

    if not Ar.IsPackageRelativePath(asset_path):
        return asset_path
    parts = Ar.SplitPackageRelativePathOuter(asset_path)
    if isinstance(parts, tuple | list) and len(parts) == 2:
        return str(parts[0])
    raise RuntimeError(f"Could not parse resolved package asset: {asset_path}")


def _filesystem_dependency_path(asset_path: str) -> Path | None:
    """Return the local outer file for an asset, or ``None`` for resolver URIs."""
    outer = _outer_asset_identifier(asset_path)
    parsed = urlparse(outer)
    if parsed.scheme and len(parsed.scheme) > 1:
        if parsed.scheme.lower() != "file":
            return None
        path_text = url2pathname(parsed.path)
        if parsed.netloc:
            path_text = f"//{parsed.netloc}{path_text}"
    else:
        path_text = outer

    candidate = Path(path_text).expanduser()
    if not candidate.is_absolute():
        raise RuntimeError(
            f"Resolved filesystem dependency is not absolute: {asset_path}"
        )
    return candidate.resolve()


def _normalize_dependency_roots(
    roots: Iterable[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    normalized = tuple(
        dict.fromkeys(Path(root).expanduser().resolve() for root in roots)
    )
    if not normalized:
        raise ValueError("approved_dependency_roots must not be empty")
    filesystem_roots = [str(root) for root in normalized if root.parent == root]
    if filesystem_roots:
        raise ValueError(
            "approved_dependency_roots must not contain filesystem roots: "
            + ", ".join(filesystem_roots)
        )
    invalid = [str(root) for root in normalized if not root.is_dir()]
    if invalid:
        raise ValueError(
            "approved_dependency_roots must contain existing directories: "
            + ", ".join(invalid)
        )
    return normalized


def _require_approved_dependency(
    asset_path: str,
    approved_roots: tuple[Path, ...],
) -> None:
    """Reject filesystem dependencies outside explicitly approved roots."""
    candidate = _filesystem_dependency_path(asset_path)
    if candidate is None:
        return
    if any(candidate.is_relative_to(root) for root in approved_roots):
        return
    raise RuntimeError(
        "Cannot create a portable USD export; resolved dependency is outside "
        f"approved roots: {candidate}"
    )


def _asset_basename(asset_path: object) -> str:
    """Return a safe leaf name for a resolved file or package member."""
    from pxr import Ar

    path = str(asset_path)
    while Ar.IsPackageRelativePath(path):
        inner_parts = Ar.SplitPackageRelativePathInner(path)
        if isinstance(inner_parts, tuple | list) and len(inner_parts) == 2:
            inner = str(inner_parts[1])
        else:
            inner = str(inner_parts)
        if not inner or inner == path:
            break
        path = inner

    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    if not name or name in {".", ".."} or ":" in name:
        raise RuntimeError(f"Unsafe USD asset filename: {name!r}")
    return name


def _copy_resolved_asset(resolver: Any, resolved_path: Any, destination: Path) -> None:
    """Copy one resolver-backed asset, including a member inside a USDZ."""
    asset = resolver.OpenAsset(resolved_path)
    if asset is None:
        raise RuntimeError(
            f"Failed to open resolved USD asset: {_resolved_path_string(resolved_path)}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    size = int(asset.GetSize())
    offset = 0
    with open(destination, "wb") as stream:
        while offset < size:
            count = min(_COPY_CHUNK_BYTES, size - offset)
            chunk = asset.Read(count, offset)
            if not chunk:
                raise RuntimeError(
                    "Resolver returned a short read for USD asset: "
                    f"{_resolved_path_string(resolved_path)}"
                )
            stream.write(chunk)
            offset += len(chunk)
    if offset != size:
        raise RuntimeError(
            "Resolver returned the wrong byte count for USD asset: "
            f"{_resolved_path_string(resolved_path)}"
        )


def _discover_expanded_dependencies(
    stage: Any,
    is_runtime_asset_path: Callable[[str], bool],
    require_approved_dependency: Callable[[str], None],
) -> dict[str, list[str]]:
    """Return template expansions plus unresolved non-runtime dependencies."""
    from pxr import Sdf, UsdUtils

    expanded: dict[str, list[str]] = {}

    def record(layer: Any, dependency_info: Any) -> Any:
        asset_path = str(dependency_info.assetPath)
        if is_runtime_asset_path(asset_path):
            return UsdUtils.DependencyInfo()
        dependencies = list(dependency_info.dependencies)
        if dependencies:
            anchored = _anchored_asset_path(layer, asset_path)
            expanded.setdefault(anchored, [])
            for dependency in dependencies:
                candidate = _anchored_asset_path(layer, dependency)
                if candidate not in expanded[anchored]:
                    expanded[anchored].append(candidate)
        return dependency_info

    root_layer = stage.GetRootLayer()
    root_identifier = str(
        root_layer.resolvedPath or root_layer.realPath or root_layer.identifier
    )
    if not root_identifier:
        raise RuntimeError("Scene Optimizer input has no resolvable root layer")

    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
            Sdf.AssetPath(root_identifier), record
        )
    except Exception as exc:
        raise RuntimeError("Failed to enumerate USD asset dependencies") from exc

    rejected = sorted(
        str(path) for path in unresolved if not is_runtime_asset_path(str(path))
    )
    if rejected:
        raise RuntimeError(
            "Cannot create a portable USD export; unresolved asset dependencies: "
            + ", ".join(rejected)
        )

    for layer in layers:
        identifier = str(layer.realPath or layer.resolvedPath or layer.identifier)
        if identifier:
            require_approved_dependency(identifier)
    for asset in assets:
        asset_path = str(asset)
        if not is_runtime_asset_path(asset_path):
            require_approved_dependency(asset_path)
    return expanded


def _copy_layer(layer: Any, output_name: str) -> Any:
    """Copy ``layer`` so path rewriting never mutates the optimizer stage."""
    from pxr import Sdf

    copied = Sdf.Layer.CreateAnonymous(f"portable_{output_name}.usda")
    copied.TransferContent(layer)
    return copied


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _require_owned_sidecar_at(
    parent_descriptor: _DirectoryHandle,
    sidecar_name: str,
    display_path: Path,
) -> os.stat_result:
    """Refuse a sidecar not owned by this exporter, without following links."""
    message = f"Refusing to replace a non-exporter-owned USD sidecar: {display_path}"
    if isinstance(parent_descriptor, _WindowsDirectory):
        directory_descriptor: _DirectoryHandle | None = None
        marker_handle: int | None = None
        marker_descriptor: int | None = None
        try:
            directory_descriptor = _open_child_directory(
                parent_descriptor,
                sidecar_name,
                create=False,
            )
            metadata_directory = _directory_metadata(directory_descriptor)
            marker_handle = _windows_open_relative_handle(
                directory_descriptor,
                PORTABLE_SIDECAR_MARKER_NAME,
                directory=False,
                create=False,
            )
            metadata = _windows_metadata(marker_handle)
            marker_descriptor = msvcrt.open_osfhandle(
                marker_handle,
                os.O_RDONLY | getattr(os, "O_BINARY", 0),
            )
            marker_handle = None
            marker_bytes = os.read(
                marker_descriptor,
                len(PORTABLE_SIDECAR_MARKER_BYTES) + 1,
            )
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(message) from exc
        finally:
            if marker_descriptor is not None:
                os.close(marker_descriptor)
            elif marker_handle is not None:
                _WINDOWS_CLOSE_HANDLE(marker_handle)
            if directory_descriptor is not None:
                _close_directory(directory_descriptor)
        if marker_bytes != PORTABLE_SIDECAR_MARKER_BYTES:
            raise RuntimeError(message)
        return metadata_directory

    if isinstance(parent_descriptor, _PathDirectory):
        sidecar_metadata = _entry_metadata(parent_descriptor, sidecar_name)
        if (
            sidecar_metadata is None
            or _is_reparse_point(sidecar_metadata)
            or not stat.S_ISDIR(sidecar_metadata.st_mode)
        ):
            raise RuntimeError(message)
        sidecar_path = _path_child(parent_descriptor, sidecar_name)
        sidecar_directory: _DirectoryHandle | None = None
        marker_descriptor = -1
        try:
            sidecar_directory = _open_directory_nofollow(
                sidecar_path,
                create=False,
            )
            assert isinstance(sidecar_directory, _PathDirectory)
            marker_metadata = _entry_metadata(
                sidecar_directory,
                PORTABLE_SIDECAR_MARKER_NAME,
            )
            if (
                marker_metadata is None
                or _is_reparse_point(marker_metadata)
                or not stat.S_ISREG(marker_metadata.st_mode)
            ):
                raise RuntimeError(message)
            marker_path = _path_child(
                sidecar_directory,
                PORTABLE_SIDECAR_MARKER_NAME,
            )
            marker_descriptor = os.open(
                marker_path,
                os.O_RDONLY | _BINARY_OPEN_FLAG,
            )
            if not _same_entry(marker_metadata, os.fstat(marker_descriptor)):
                raise _OutputCommitRaceError(
                    "Portable USD sidecar marker changed during validation"
                )
            marker_bytes = os.read(
                marker_descriptor,
                len(PORTABLE_SIDECAR_MARKER_BYTES) + 1,
            )
            if not _same_entry(
                sidecar_metadata,
                _directory_metadata(sidecar_directory),
            ) or not _same_entry(
                sidecar_metadata,
                _entry_metadata(parent_descriptor, sidecar_name),
            ):
                raise _OutputCommitRaceError(
                    "Portable USD sidecar changed during validation"
                )
        except _OutputCommitRaceError:
            raise
        except RuntimeError:
            raise
        except OSError as exc:
            raise RuntimeError(message) from exc
        finally:
            if marker_descriptor >= 0:
                os.close(marker_descriptor)
            if sidecar_directory is not None:
                _close_directory(sidecar_directory)
        if marker_bytes != PORTABLE_SIDECAR_MARKER_BYTES:
            raise RuntimeError(message)
        return sidecar_metadata
    directory_descriptor = -1
    marker_descriptor = -1
    try:
        directory_descriptor = os.open(
            sidecar_name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        metadata_directory = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(metadata_directory.st_mode):
            raise RuntimeError(message)
        marker_descriptor = os.open(
            PORTABLE_SIDECAR_MARKER_NAME,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
        metadata = os.fstat(marker_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(message)
        marker_bytes = os.read(
            marker_descriptor,
            len(PORTABLE_SIDECAR_MARKER_BYTES) + 1,
        )
    except OSError as exc:
        raise RuntimeError(message) from exc
    finally:
        if marker_descriptor >= 0:
            os.close(marker_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
    if marker_bytes != PORTABLE_SIDECAR_MARKER_BYTES:
        raise RuntimeError(message)
    return metadata_directory


def _require_owned_sidecar(sidecar: Path) -> None:
    """Refuse to replace a directory not created by this exporter."""
    lexical_sidecar = _lexical_absolute_path(sidecar)
    parent_descriptor = _open_directory_nofollow(
        lexical_sidecar.parent,
        create=False,
    )
    try:
        _require_owned_sidecar_at(
            parent_descriptor,
            lexical_sidecar.name,
            lexical_sidecar,
        )
    finally:
        _close_directory(parent_descriptor)


def _write_sidecar_marker(sidecar: Path) -> None:
    """Mark a populated transaction sidecar as safe for future replacement."""
    if not sidecar.is_dir():
        return
    marker = sidecar / PORTABLE_SIDECAR_MARKER_NAME
    marker.write_bytes(PORTABLE_SIDECAR_MARKER_BYTES)
    if _SUPPORTS_DIRECTORY_DESCRIPTORS:
        marker.chmod(0o444)


def _commit_export(
    transaction_output: Path,
    transaction_sidecar: Path,
    output: Path,
    sidecar: Path,
    *,
    transaction_dir_fd: _DirectoryHandle | None = None,
    destination_dir_fd: _DirectoryHandle | None = None,
    overwrite: bool = True,
    post_commit: (
        Callable[
            [os.stat_result, os.stat_result, os.stat_result | None],
            None,
        ]
        | None
    ) = None,
) -> None:
    """Replace the prior bundle and run a finalizer before discarding backups."""
    own_transaction_descriptor = transaction_dir_fd is None
    own_destination_descriptor = destination_dir_fd is None
    if transaction_dir_fd is None:
        transaction_dir_fd = _open_directory_nofollow(
            transaction_output.parent,
            create=False,
        )
    if destination_dir_fd is None:
        try:
            destination_dir_fd = _open_directory_nofollow(
                output.parent,
                create=False,
            )
        except Exception:
            if own_transaction_descriptor:
                _close_directory(transaction_dir_fd)
            raise

    output_backup: tuple[str, os.stat_result] | None = None
    sidecar_backup: tuple[str, os.stat_result] | None = None
    committed_output: os.stat_result | None = None
    committed_sidecar: os.stat_result | None = None
    try:
        backup_output_name = _unused_entry_name(
            transaction_dir_fd,
            ".previous_output_",
        )
        backup_sidecar_name = _unused_entry_name(
            transaction_dir_fd,
            ".previous_sidecar_",
        )
        prior_output = _require_replaceable_output(
            destination_dir_fd,
            output.name,
            output,
        )
        if prior_output is not None and not overwrite:
            raise ValueError(
                "output_usd_path already exists; set overwrite=true to replace it"
            )
        sidecar_metadata = _entry_metadata(destination_dir_fd, sidecar.name)
        prior_sidecar = None
        if sidecar_metadata is not None:
            prior_sidecar = _require_owned_sidecar_at(
                destination_dir_fd,
                sidecar.name,
                sidecar,
            )
            if not overwrite:
                raise ValueError(
                    "Portable USD sidecar already exists; set overwrite=true to "
                    "replace it"
                )

        transaction_output_metadata = _entry_metadata(
            transaction_dir_fd,
            transaction_output.name,
        )
        if transaction_output_metadata is None or not stat.S_ISREG(
            transaction_output_metadata.st_mode
        ):
            raise RuntimeError(
                f"Portable USD transaction output is not a file: {transaction_output}"
            )
        transaction_sidecar_metadata = _entry_metadata(
            transaction_dir_fd,
            transaction_sidecar.name,
        )
        if transaction_sidecar_metadata is not None:
            owned_transaction_sidecar = _require_owned_sidecar_at(
                transaction_dir_fd,
                transaction_sidecar.name,
                transaction_sidecar,
            )
            if not _same_entry(
                transaction_sidecar_metadata,
                owned_transaction_sidecar,
            ):
                raise _OutputCommitRaceError(
                    "New USD sidecar changed during validation; transaction was "
                    "preserved"
                )

        if prior_sidecar is not None:
            sidecar_backup = (backup_sidecar_name, prior_sidecar)
            moved = _move_checked_entry(
                destination_dir_fd,
                sidecar.name,
                transaction_dir_fd,
                backup_sidecar_name,
                prior_sidecar,
                "existing USD sidecar",
            )
            sidecar_backup = (backup_sidecar_name, moved)
            moved_owned = _require_owned_sidecar_at(
                transaction_dir_fd,
                backup_sidecar_name,
                sidecar,
            )
            if not _same_entry(moved, moved_owned):
                raise _OutputCommitRaceError(
                    "Existing USD sidecar changed after backup; transaction was "
                    "preserved"
                )
        if prior_output is not None:
            output_backup = (backup_output_name, prior_output)
            moved = _move_checked_entry(
                destination_dir_fd,
                output.name,
                transaction_dir_fd,
                backup_output_name,
                prior_output,
                "existing USD output",
            )
            output_backup = (backup_output_name, moved)
        if transaction_sidecar_metadata is not None:
            committed_sidecar = transaction_sidecar_metadata
            committed_sidecar = _move_checked_entry(
                transaction_dir_fd,
                transaction_sidecar.name,
                destination_dir_fd,
                sidecar.name,
                transaction_sidecar_metadata,
                "new USD sidecar",
            )
        committed_output = transaction_output_metadata
        committed_output = _move_checked_entry(
            transaction_dir_fd,
            transaction_output.name,
            destination_dir_fd,
            output.name,
            transaction_output_metadata,
            "new USD output",
        )
        if post_commit is not None:
            post_commit(
                _directory_metadata(destination_dir_fd),
                committed_output,
                committed_sidecar,
            )
    except BaseException as commit_error:
        rollback_errors: list[BaseException] = []
        if committed_output is not None:
            try:
                _undo_checked_move(
                    transaction_dir_fd,
                    transaction_output.name,
                    destination_dir_fd,
                    output.name,
                    committed_output,
                    "new USD output",
                )
            except BaseException as exc:
                rollback_errors.append(exc)
        if committed_sidecar is not None:
            try:
                _undo_checked_move(
                    transaction_dir_fd,
                    transaction_sidecar.name,
                    destination_dir_fd,
                    sidecar.name,
                    committed_sidecar,
                    "new USD sidecar",
                )
            except BaseException as exc:
                rollback_errors.append(exc)
        if output_backup is not None:
            backup_name, backup_metadata = output_backup
            try:
                _undo_checked_move(
                    destination_dir_fd,
                    output.name,
                    transaction_dir_fd,
                    backup_name,
                    backup_metadata,
                    "existing USD output",
                )
            except BaseException as exc:
                rollback_errors.append(exc)
        if sidecar_backup is not None:
            backup_name, backup_metadata = sidecar_backup
            try:
                _undo_checked_move(
                    destination_dir_fd,
                    sidecar.name,
                    transaction_dir_fd,
                    backup_name,
                    backup_metadata,
                    "existing USD sidecar",
                )
            except BaseException as exc:
                rollback_errors.append(exc)
        if rollback_errors:
            raise _OutputCommitRaceError(
                "USD export commit could not be safely rolled back; transaction "
                "was preserved"
            ) from commit_error
        raise
    finally:
        if own_destination_descriptor:
            _close_directory(destination_dir_fd)
        if own_transaction_descriptor:
            _close_directory(transaction_dir_fd)


def _commit_output_file(
    transaction_output: Path,
    output: Path,
    *,
    transaction_dir_fd: _DirectoryHandle,
    destination_dir_fd: _DirectoryHandle,
) -> None:
    """Commit one regular file without following destination symlinks."""
    transaction_metadata = _entry_metadata(
        transaction_dir_fd,
        transaction_output.name,
    )
    if transaction_metadata is None or not stat.S_ISREG(transaction_metadata.st_mode):
        raise RuntimeError(
            f"USD output transaction did not create: {transaction_output}"
        )
    prior_output = _require_replaceable_output(
        destination_dir_fd,
        output.name,
        output,
    )
    backup_output_name = _unused_entry_name(
        transaction_dir_fd,
        ".previous_output_",
    )
    output_backup: tuple[str, os.stat_result] | None = None
    committed_output: os.stat_result | None = None
    try:
        if prior_output is not None:
            output_backup = (backup_output_name, prior_output)
            moved = _move_checked_entry(
                destination_dir_fd,
                output.name,
                transaction_dir_fd,
                backup_output_name,
                prior_output,
                "existing USD output",
            )
            output_backup = (backup_output_name, moved)
        committed_output = transaction_metadata
        committed_output = _move_checked_entry(
            transaction_dir_fd,
            transaction_output.name,
            destination_dir_fd,
            output.name,
            transaction_metadata,
            "new USD output",
        )
    except BaseException as commit_error:
        rollback_errors: list[BaseException] = []
        if committed_output is not None:
            try:
                _undo_checked_move(
                    transaction_dir_fd,
                    transaction_output.name,
                    destination_dir_fd,
                    output.name,
                    committed_output,
                    "new USD output",
                )
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if output_backup is not None:
            backup_name, backup_metadata = output_backup
            try:
                _undo_checked_move(
                    destination_dir_fd,
                    output.name,
                    transaction_dir_fd,
                    backup_name,
                    backup_metadata,
                    "existing USD output",
                )
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise _OutputCommitRaceError(
                "USD output commit could not be safely rolled back; transaction "
                "was preserved"
            ) from commit_error
        raise


@contextmanager
def _atomic_output_file(
    output_path: str | os.PathLike[str],
    *,
    clear_portable_sidecar: bool = False,
) -> Iterator[Path]:
    """Yield a confined temporary path and atomically publish it on success.

    ``clear_portable_sidecar`` is for writers that publish a standalone root
    layer. When enabled, an existing exporter-owned current-format sidecar is
    removed in the same rollback-safe transaction as the root replacement.
    The default deliberately retains sidecars for in-place metadata/UV edits.
    """
    output = _lexical_absolute_path(output_path)
    sidecar = output.parent / portable_sidecar_name(output)
    parent_descriptor = _open_directory_nofollow(output.parent, create=True)
    transaction_name = ""
    transaction_descriptor: _DirectoryHandle | None = None
    preserve_transaction = False
    try:
        _require_replaceable_output(parent_descriptor, output.name, output)
        if (
            clear_portable_sidecar
            and _entry_metadata(parent_descriptor, sidecar.name) is not None
        ):
            _require_owned_sidecar_at(
                parent_descriptor,
                sidecar.name,
                sidecar,
            )
        (
            transaction_name,
            transaction_descriptor,
            transaction_dir,
        ) = _create_transaction_directory(parent_descriptor, ".usd_output_")
        transaction_output = transaction_dir / output.name
        yield transaction_output
        if clear_portable_sidecar:
            _commit_export(
                transaction_output,
                transaction_dir / sidecar.name,
                output,
                sidecar,
                transaction_dir_fd=transaction_descriptor,
                destination_dir_fd=parent_descriptor,
            )
        else:
            _commit_output_file(
                transaction_output,
                output,
                transaction_dir_fd=transaction_descriptor,
                destination_dir_fd=parent_descriptor,
            )
    except _OutputCommitRaceError:
        preserve_transaction = True
        raise
    finally:
        if transaction_descriptor is not None:
            if transaction_name and not preserve_transaction:
                _cleanup_transaction_directory(
                    parent_descriptor,
                    transaction_name,
                    transaction_descriptor,
                )
            _close_directory(transaction_descriptor)
        _close_directory(parent_descriptor)


def _validate_portable_export(
    output: Path,
    sidecar: Path,
    is_runtime_asset_path: Callable[[str], bool],
) -> None:
    """Fail unless every non-runtime dependency stays inside ``sidecar``."""
    from pxr import Sdf, UsdUtils

    def ignore_runtime_asset(_layer: Any, dependency_info: Any) -> Any:
        if is_runtime_asset_path(str(dependency_info.assetPath)):
            return UsdUtils.DependencyInfo()
        return dependency_info

    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
            Sdf.AssetPath(str(output)), ignore_runtime_asset
        )
    except Exception as exc:
        raise RuntimeError("Failed to validate portable USD export") from exc

    if unresolved:
        raise RuntimeError(
            "Portable USD export still has unresolved dependencies: "
            + ", ".join(sorted(str(path) for path in unresolved))
        )

    output_resolved = output.resolve()
    sidecar_resolved = sidecar.resolve()
    external_layers = []
    for layer in layers:
        identifier = str(layer.realPath or layer.resolvedPath or layer.identifier)
        if not identifier:
            external_layers.append("<anonymous>")
            continue
        candidate = Path(identifier).resolve()
        if candidate == output_resolved:
            continue
        try:
            candidate.relative_to(sidecar_resolved)
        except ValueError:
            external_layers.append(identifier)
    if external_layers:
        raise RuntimeError(
            "Portable USD export still has external layers: "
            + ", ".join(sorted(external_layers))
        )

    escaped_assets = []
    for asset in assets:
        outer_identifier = _outer_asset_identifier(str(asset))
        parsed = urlparse(outer_identifier)
        if parsed.scheme and len(parsed.scheme) > 1:
            candidate = _filesystem_dependency_path(outer_identifier)
            if candidate is None:
                escaped_assets.append(str(asset))
                continue
        else:
            candidate = Path(outer_identifier).expanduser()
            if not candidate.is_absolute():
                candidate = output.parent / candidate
            candidate = candidate.resolve()
        try:
            candidate.relative_to(sidecar_resolved)
        except ValueError:
            escaped_assets.append(str(asset))
    if escaped_assets:
        raise RuntimeError(
            "Portable USD export still references assets outside its sidecar: "
            + ", ".join(sorted(escaped_assets))
        )


def export_stage_portably(
    stage: Any,
    output_path: str | os.PathLike[str],
    *,
    approved_dependency_roots: Iterable[str | os.PathLike[str]],
    export_layer: Any | None = None,
    is_runtime_asset_path: Callable[[str], bool] = is_bare_mdl_token,
    overwrite: bool = True,
    post_commit: (
        Callable[
            [os.stat_result, os.stat_result, os.stat_result | None],
            None,
        ]
        | None
    ) = None,
) -> bool:
    """Export ``stage`` with every resolvable dependency beside the result.

    Layered stages use their composed flattened layer (``export_layer_for``),
    while single-layer stages retain authored composition. Asset-valued paths
    are copied into ``<output-filename>_assets`` and rewritten relative to the
    output. Bare ``*.mdl`` module tokens are intentionally preserved because
    Omniverse resolves those at runtime. URI and custom-resolver dependencies
    are copied when the active resolver can resolve and open them; otherwise,
    like every other unresolved non-runtime path, the export fails closed.
    ``approved_dependency_roots`` is mandatory for filesystem dependencies;
    package members are authorized by their outer package path. Resolver URIs
    remain resolver-owned. Callers may also provide a prepared ``export_layer``
    and a broader runtime-path predicate while retaining the same resolver-copy
    and atomic-commit logic. ``post_commit`` receives identity metadata for the
    destination parent, root output, and optional sidecar
    after the new bundle is in place but before prior-output backups are
    discarded; an exception rolls the complete export transaction back.
    ``overwrite=False`` is enforced again by the identity-checked commit so a
    destination created while the export is being prepared is not replaced.
    """
    from pxr import Ar, Sdf, UsdUtils

    approved_roots = _normalize_dependency_roots(approved_dependency_roots)
    output = _lexical_absolute_path(output_path)
    sidecar = output.parent / portable_sidecar_name(output)
    resolver = Ar.GetResolver()
    destination_descriptor = _open_directory_nofollow(output.parent, create=True)
    transaction_name = ""
    transaction_descriptor: _DirectoryHandle | None = None
    preserve_transaction = False

    def require_approved_dependency(asset_path: str) -> None:
        _require_approved_dependency(asset_path, approved_roots)

    try:
        _require_replaceable_output(
            destination_descriptor,
            output.name,
            output,
        )
        if _entry_metadata(destination_descriptor, sidecar.name) is not None:
            _require_owned_sidecar_at(
                destination_descriptor,
                sidecar.name,
                sidecar,
            )

        with Ar.ResolverContextBinder(stage.GetPathResolverContext()):
            expanded = _discover_expanded_dependencies(
                stage,
                is_runtime_asset_path,
                require_approved_dependency,
            )
            source_layer = (
                export_layer if export_layer is not None else export_layer_for(stage)
            )
            portable_layer = _copy_layer(source_layer, output.name)

            (
                transaction_name,
                transaction_descriptor,
                transaction_dir,
            ) = _create_transaction_directory(
                destination_descriptor,
                f".{output.stem}_export_",
            )
            transaction_output = transaction_dir / output.name
            transaction_sidecar = transaction_dir / sidecar.name
            copied: dict[tuple[str, str], Path] = {}

            def copy_one(anchored_path: str, group_key: str | None = None) -> Path:
                resolved = resolver.Resolve(anchored_path)
                if not resolved:
                    raise RuntimeError(
                        "Cannot create a portable USD export; unresolved asset: "
                        f"{anchored_path}"
                    )
                resolved_text = _resolved_path_string(resolved)
                require_approved_dependency(resolved_text)
                cache_key = (resolved_text, str(group_key or ""))
                relative = copied.get(cache_key)
                if relative is None:
                    digest_source = str(group_key or resolved_text)
                    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[
                        :16
                    ]
                    relative = Path(digest) / _asset_basename(resolved_text)
                    destination = transaction_sidecar / relative
                    if destination.exists():
                        raise RuntimeError(
                            "Portable USD dependency destination collision: "
                            f"{destination}"
                        )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    suffix = Path(_asset_basename(resolved_text)).suffix.lower()
                    if suffix not in _LOCALIZABLE_USD_LAYER_SUFFIXES:
                        _copy_resolved_asset(resolver, resolved, destination)
                    copied[cache_key] = relative
                    localize_copied_layer(resolved_text, destination)
                return relative

            def rewritten_path(relative: Path, owner: Path) -> str:
                destination = transaction_sidecar / relative
                return Path(os.path.relpath(destination, owner.parent)).as_posix()

            def rewrite_for_layer(
                asset_path: object,
                source_layer: Any,
                destination_layer: Path,
            ) -> str:
                raw = str(asset_path)
                if not raw or is_runtime_asset_path(raw):
                    return raw

                anchored = _anchored_asset_path(source_layer, raw)
                dependencies = expanded.get(anchored)
                if dependencies:
                    for dependency in dependencies:
                        copy_one(dependency, group_key=anchored)
                    relative = Path(
                        hashlib.sha256(anchored.encode("utf-8")).hexdigest()[:16]
                    ) / _asset_basename(anchored)
                    return rewritten_path(relative, destination_layer)

                relative = copy_one(anchored)
                return rewritten_path(relative, destination_layer)

            def localize_copied_layer(
                source_identifier: str,
                destination_layer: Path,
            ) -> None:
                suffix = Path(_asset_basename(source_identifier)).suffix.lower()
                if suffix not in _LOCALIZABLE_USD_LAYER_SUFFIXES:
                    return

                source_dependency_layer = Sdf.Layer.FindOrOpen(source_identifier)
                if source_dependency_layer is None:
                    raise RuntimeError(
                        f"Failed to open copied USD dependency: {source_identifier}"
                    )
                portable_dependency_layer = _copy_layer(
                    source_dependency_layer,
                    destination_layer.name,
                )
                UsdUtils.ModifyAssetPaths(
                    portable_dependency_layer,
                    lambda path: rewrite_for_layer(
                        path,
                        source_dependency_layer,
                        destination_layer,
                    ),
                )
                if not portable_dependency_layer.Export(str(destination_layer)):
                    raise RuntimeError(
                        f"Failed to export copied USD dependency: {destination_layer}"
                    )

            UsdUtils.ModifyAssetPaths(
                portable_layer,
                lambda path: rewrite_for_layer(
                    path,
                    stage.GetRootLayer(),
                    transaction_output,
                ),
            )
            if not portable_layer.Export(str(transaction_output)):
                raise RuntimeError(f"Failed to export USD stage: {output}")
            _validate_portable_export(
                transaction_output,
                transaction_sidecar,
                is_runtime_asset_path,
            )
            _write_sidecar_marker(transaction_sidecar)
            _commit_export(
                transaction_output,
                transaction_sidecar,
                output,
                sidecar,
                transaction_dir_fd=transaction_descriptor,
                destination_dir_fd=destination_descriptor,
                overwrite=overwrite,
                post_commit=post_commit,
            )
    except _OutputCommitRaceError:
        preserve_transaction = True
        raise
    finally:
        if transaction_descriptor is not None:
            if transaction_name and not preserve_transaction:
                _cleanup_transaction_directory(
                    destination_descriptor,
                    transaction_name,
                    transaction_descriptor,
                )
            _close_directory(transaction_descriptor)
        _close_directory(destination_descriptor)

    return True
