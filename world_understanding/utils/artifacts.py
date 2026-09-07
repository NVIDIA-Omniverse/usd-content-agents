# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact-path safety helpers.

POSIX hosts keep every traversed directory descriptor open and use
``O_NOFOLLOW`` for ancestors and leaves. Windows normally uses relative
``NtCreateFile`` opens with ``FILE_OPEN_REPARSE_POINT`` while retaining
no-delete handles for the complete ancestor chain. Sandboxed Windows accounts
that may traverse but not open a profile ancestor fall back to a verified
full-path directory handle; descendant operations remain relative to that
pinned handle. Both backends therefore operate on the objects they validate
instead of trusting a path after a separate check.
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO
from uuid import uuid4

if os.name == "nt":  # pragma: win32 cover
    import ctypes
    import msvcrt
    from ctypes import wintypes

_PIPELINE_TEMP_COMPONENT = ".pipeline_temp"
_WINDOWS_RESERVED_FILE_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{suffix}" for suffix in "123456789¹²³"}
    | {f"LPT{suffix}" for suffix in "123456789¹²³"}
)
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class ArtifactPathError(RuntimeError):
    """Raised when a local artifact path cannot be traversed safely."""


@dataclass(frozen=True)
class OpenArtifactFile:
    """One regular artifact held open beneath a descriptor-confined root."""

    relative_key: str
    stream: BinaryIO
    metadata: os.stat_result


@dataclass
class ConfinedAtomicWrite:
    """State yielded while atomically publishing a confined artifact."""

    stream: BinaryIO | None
    published: bool = False


def is_pipeline_temp_path(path: str | Path) -> bool:
    """Return whether a relative or absolute path enters ``.pipeline_temp``."""
    return any(
        component.casefold() == _PIPELINE_TEMP_COMPONENT
        for component in str(path).replace("\\", "/").split("/")
    )


def validated_artifact_relative_key(key: object) -> str:
    """Return one canonical POSIX artifact key or reject it.

    Storage keys are intentionally narrower than host filesystem paths.  This
    rejects spellings that could alias on Windows even though production runs
    on Linux, keeping cross-platform fixtures deterministic.
    """

    if not isinstance(key, str):
        raise ValueError("Artifact key must be a string")
    parts = key.split("/")
    windows_path = PureWindowsPath(key)
    if (
        not key
        or key.startswith("/")
        or "\\" in key
        or "\x00" in key
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in parts)
        or any(part.casefold() == _PIPELINE_TEMP_COMPONENT for part in parts)
    ):
        raise ValueError("Artifact key is reserved or non-canonical")
    return key


def validated_s3_object_suffix(object_key: object, session_prefix: str) -> str:
    """Validate and return the canonical suffix of one session-owned S3 key."""

    if not isinstance(object_key, str) or not object_key.startswith(session_prefix):
        raise ValueError("S3 object key is outside the session key prefix")
    suffix = object_key[len(session_prefix) :]
    try:
        return validated_artifact_relative_key(suffix)
    except ValueError as exc:
        raise ValueError("S3 object key has an unsafe local path suffix") from exc


def _absolute_path_parts(path: str | Path) -> tuple[str, ...]:
    """Return lexical absolute path components without resolving symlinks."""

    raw_path = os.fspath(path)
    if "\x00" in raw_path:
        raise ValueError("Artifact path contains a null byte")
    absolute = Path(os.path.abspath(raw_path))
    return tuple(part for part in absolute.parts if part != absolute.anchor)


_SUPPORTS_DIRECTORY_DESCRIPTORS = os.name == "posix"

#: The Windows CRT opens descriptors in text-translation mode unless this is
#: set, which rewrites LF to CRLF (and CRLF to CR-CRLF) on the way out.
#: Artifacts are bytes, so every open in this module must be binary. Zero on
#: POSIX, where the flag does not exist and no translation happens.
_BINARY_OPEN_FLAG = getattr(os, "O_BINARY", 0)
_NO_INHERIT_OPEN_FLAG = getattr(os, "O_NOINHERIT", 0)
_SUPPORTS_WINDOWS_HANDLE_CONFINEMENT = os.name == "nt"

if os.name == "nt":  # pragma: win32 cover
    _FILE_LIST_DIRECTORY = 0x0001
    _FILE_READ_DATA = 0x0001
    _FILE_WRITE_DATA = 0x0002
    _FILE_APPEND_DATA = 0x0004
    _FILE_READ_ATTRIBUTES = 0x0080
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
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _OPEN_EXISTING = 3
    _OBJ_CASE_INSENSITIVE = 0x00000040
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

    class _FileRenameInformation(ctypes.Structure):
        _fields_ = [
            ("ReplaceIfExists", wintypes.BOOLEAN),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]

    class _FileDispositionInformation(ctypes.Structure):
        _fields_ = [("DeleteFile", wintypes.BOOLEAN)]

    _ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _nt_create_file = _ntdll.NtCreateFile
    _nt_create_file.argtypes = [
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
    _nt_create_file.restype = ctypes.c_long
    _rtl_nt_status_to_dos_error = _ntdll.RtlNtStatusToDosError
    _rtl_nt_status_to_dos_error.argtypes = [ctypes.c_long]
    _rtl_nt_status_to_dos_error.restype = wintypes.ULONG
    _create_file = _kernel32.CreateFileW
    _create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _create_file.restype = wintypes.HANDLE
    _close_handle = _kernel32.CloseHandle
    _close_handle.argtypes = [wintypes.HANDLE]
    _close_handle.restype = wintypes.BOOL
    _get_file_information = _kernel32.GetFileInformationByHandle
    _get_file_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    ]
    _get_file_information.restype = wintypes.BOOL
    _get_final_path_name = _kernel32.GetFinalPathNameByHandleW
    _get_final_path_name.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    _get_final_path_name.restype = wintypes.DWORD
    _set_file_information = _kernel32.SetFileInformationByHandle
    _set_file_information.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _set_file_information.restype = wintypes.BOOL
    _flush_file_buffers = _kernel32.FlushFileBuffers
    _flush_file_buffers.argtypes = [wintypes.HANDLE]
    _flush_file_buffers.restype = wintypes.BOOL
    _INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value


class _WindowsConfinedDirectory:
    """A Windows directory pinned by no-delete, no-reparse ancestor handles."""

    __slots__ = ("_owned_handles", "handle", "path")

    def __init__(self, path: Path, handles: tuple[int, ...]) -> None:
        if not handles:
            raise ValueError("Windows confined directory requires a held handle")
        self.path = path
        self._owned_handles = handles
        self.handle = handles[-1]

    def close(self) -> None:
        if os.name != "nt":  # pragma: no cover - defensive type guard
            return
        first_error: OSError | None = None
        while self._owned_handles:
            handle, self._owned_handles = (
                self._owned_handles[-1],
                self._owned_handles[:-1],
            )
            if not _close_handle(handle) and first_error is None:
                first_error = ctypes.WinError(ctypes.get_last_error())
        if first_error is not None:
            raise first_error

    def identity(self) -> tuple[int, int]:
        information = _windows_file_information(self.handle)
        file_index = (int(information.FileIndexHigh) << 32) | int(
            information.FileIndexLow
        )
        return int(information.VolumeSerialNumber), file_index

    def __index__(self) -> int:
        raise TypeError(
            "This confined Windows directory is a native handle chain, not a "
            "CRT file descriptor. Use a confined artifact operation."
        )


def _windows_raise_ntstatus(status: int) -> None:
    error = int(_rtl_nt_status_to_dos_error(status))
    raise ctypes.WinError(error)


def _windows_normalized_handle_path(handle: int) -> str:
    """Return one comparable DOS path for a pinned Windows handle."""

    required = _get_final_path_name(handle, None, 0, 0)
    if required == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_unicode_buffer(required + 1)
    written = _get_final_path_name(handle, buffer, len(buffer), 0)
    if written == 0 or written >= len(buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.normpath(os.path.abspath(value)))


def _windows_open_verified_absolute_directory_handle(directory: Path) -> int:
    """Pin an exact directory when an ancestor cannot itself be opened.

    A restricted Windows token can have traverse permission through a profile
    ancestor without permission to obtain a directory handle for that
    ancestor. ``CreateFileW`` can still open the final allowed directory. Its
    normalized handle path proves that no intermediate reparse point redirected
    the open before relative confined operations begin.
    """

    handle = _create_file(
        str(directory),
        _FILE_LIST_DIRECTORY | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        _windows_reject_unsafe_component(handle, directory=True)
        expected = os.path.normcase(
            os.path.normpath(os.path.abspath(os.fspath(directory)))
        )
        if _windows_normalized_handle_path(handle) != expected:
            raise ArtifactPathError(
                "Refusing a Windows directory reached through a reparse point"
            )
    except BaseException:
        _close_handle(handle)
        raise
    return handle


def _windows_file_information(handle: int) -> Any:
    information = _ByHandleFileInformation()
    if not _get_file_information(handle, ctypes.byref(information)):
        raise ctypes.WinError(ctypes.get_last_error())
    return information


def _windows_reject_unsafe_component(
    handle: int,
    *,
    directory: bool,
    allow_hardlinks: bool = False,
) -> None:
    information = _windows_file_information(handle)
    if information.FileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise ArtifactPathError("Refusing to traverse a reparsed artifact path")
    is_directory = bool(information.FileAttributes & _FILE_ATTRIBUTE_DIRECTORY)
    if directory != is_directory:
        if directory:
            raise NotADirectoryError("Artifact path component is not a directory")
        raise ArtifactPathError("Refusing to open a non-regular artifact")
    if not directory and information.NumberOfLinks != 1 and not allow_hardlinks:
        raise ArtifactPathError("Artifact must be a single-link regular file")


def _windows_validate_component(component: str) -> None:
    """Reject Windows aliases that are not canonical artifact components."""

    windows_forbidden = '<>:"/\\|?*'
    if (
        not component
        or component in {".", ".."}
        or any(character in windows_forbidden for character in component)
        or any(ord(character) < 32 for character in component)
        or component[-1] in {" ", "."}
        or component.partition(".")[0].upper() in _WINDOWS_RESERVED_FILE_STEMS
    ):
        raise ValueError("Artifact key contains a non-canonical Windows component")


def _windows_open_relative_handle(
    parent_handle: int,
    component: str,
    *,
    directory: bool,
    create: bool,
    exclusive_create: bool = False,
    access: int | None = None,
    allow_hardlinks: bool = False,
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
        RootDirectory=parent_handle,
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
    options |= _FILE_DIRECTORY_FILE if directory else _FILE_NON_DIRECTORY_FILE
    status = _nt_create_file(
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
    if status < 0:
        _windows_raise_ntstatus(status)
    try:
        _windows_reject_unsafe_component(
            handle.value,
            directory=directory,
            allow_hardlinks=allow_hardlinks,
        )
    except BaseException:
        _close_handle(handle.value)
        raise
    return handle.value


@contextmanager
def _windows_open_absolute_directory(
    path: str | Path,
    *,
    create: bool,
) -> Iterator[_WindowsConfinedDirectory]:
    directory = Path(os.path.abspath(os.fspath(path)))
    if not directory.anchor:
        raise ValueError("Confined Windows directory must be absolute")
    anchor_handle = _create_file(
        directory.anchor,
        _FILE_LIST_DIRECTORY | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if anchor_handle == _INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    handles = [anchor_handle]
    try:
        _windows_reject_unsafe_component(anchor_handle, directory=True)
        try:
            for component in directory.parts[1:]:
                handles.append(
                    _windows_open_relative_handle(
                        handles[-1],
                        component,
                        directory=True,
                        create=create,
                    )
                )
        except PermissionError:
            # Restricted child sandboxes can traverse an inherited profile
            # directory without being allowed to list/open that ancestor. The
            # exact final directory is still writable and can be pinned safely.
            for handle in reversed(handles):
                _close_handle(handle)
            handles = []
            if create:
                directory.mkdir(parents=True, exist_ok=True)
            handles.append(_windows_open_verified_absolute_directory_handle(directory))
        confined = _WindowsConfinedDirectory(directory, tuple(handles))
        handles = []
        try:
            yield confined
        finally:
            confined.close()
    finally:
        for handle in reversed(handles):
            _close_handle(handle)


@contextmanager
def _windows_open_relative_directory(
    root: _WindowsConfinedDirectory,
    components: tuple[str, ...],
    *,
    create: bool,
    exclusive_create: bool = False,
) -> Iterator[_WindowsConfinedDirectory]:
    if exclusive_create and (not create or not components):
        raise ValueError(
            "exclusive confined directory creation requires a relative leaf"
        )
    handles: list[int] = []
    parent_handle = root.handle
    current = root.path
    try:
        for index, component in enumerate(components):
            current /= component
            handle = _windows_open_relative_handle(
                parent_handle,
                component,
                directory=True,
                create=create,
                exclusive_create=(exclusive_create and index == len(components) - 1),
            )
            handles.append(handle)
            parent_handle = handle
        if not handles:
            yield root
            return
        confined = _WindowsConfinedDirectory(current, tuple(handles))
        handles = []
        try:
            yield confined
        finally:
            confined.close()
    finally:
        for handle in reversed(handles):
            _close_handle(handle)


def _windows_handle_to_descriptor(handle: int, flags: int) -> int:
    try:
        return msvcrt.open_osfhandle(handle, flags | _NO_INHERIT_OPEN_FLAG)
    except BaseException:
        _close_handle(handle)
        raise


def _windows_flush_confined_directory(root: _WindowsConfinedDirectory) -> None:
    """Flush one pinned directory through a write-capable native handle."""

    handle = _create_file(
        str(root.path),
        _FILE_WRITE_DATA | _SYNCHRONIZE,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        _windows_reject_unsafe_component(handle, directory=True)
        if not _flush_file_buffers(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _close_handle(handle)


class _PathConfinedDirectory:
    """A best-effort test/fallback directory where native confinement is absent.

    The POSIX implementation holds a directory descriptor and re-opens every
    child relative to it. The Windows implementation uses native relative
    handles. This distinct path-only type remains for tests and for an unknown
    non-POSIX host without either primitive; each component is checked and the
    verified absolute path is reused for the operation.

    That is a weaker guarantee. Validation and use are separate steps here, so
    an attacker who can write into an ancestor directory could still swap a
    component in between. Production Windows never selects this backend. This
    is a distinct type rather than an ``int`` so any caller that passes it to a
    ``dir_fd`` parameter fails loudly instead of silently operating on an
    unrelated descriptor.
    """

    __slots__ = ("path",)

    def __init__(self, path: Path) -> None:
        self.path = path

    def __index__(self) -> int:
        raise TypeError(
            "This confined directory is not a file descriptor. The confined "
            "artifact operation reached here has no implementation for this "
            "platform; only the pipeline checkpoint, lock, write, and tree "
            "removal paths are supported without directory descriptors."
        )


def _reject_reparse_points(path: Path) -> None:
    """Reject a path whose leaf or any ancestor is a reparse point."""

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            # Nothing exists from here down; creation validates the remainder.
            return
        except OSError as error:
            raise ArtifactPathError(
                "Cannot inspect an artifact path component"
            ) from error
        attributes = getattr(metadata, "st_file_attributes", 0)
        if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ArtifactPathError("Refusing to traverse a reparsed artifact path")


def _confined_child_path(root: _PathConfinedDirectory, relative_key: str) -> Path:
    """Return the validated absolute path of one key beneath a verified root."""

    canonical_key = validated_artifact_relative_key(relative_key)
    return root.path.joinpath(*canonical_key.split("/"))


@contextmanager
def _path_open_confined_directory_at(
    root: _PathConfinedDirectory,
    relative_key: str,
    *,
    create: bool,
    mode: int,
    exclusive_create: bool = False,
) -> Iterator[_PathConfinedDirectory]:
    """Verify a nested directory beneath a path-confined root."""

    canonical_key = validated_artifact_relative_key(relative_key)
    if exclusive_create and not create:
        raise ValueError("exclusive confined directory creation requires create=True")
    current = root.path
    components = canonical_key.split("/")
    for index, component in enumerate(components):
        current = current / component
        if exclusive_create and index == len(components) - 1:
            current.mkdir(mode=mode)
            metadata = current.lstat()
            if not stat.S_ISDIR(metadata.st_mode):  # pragma: no cover - mkdir invariant
                raise NotADirectoryError(str(current))
            continue
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if not create:
                raise
            try:
                current.mkdir(mode=mode)
            except FileExistsError:
                # Another creator won. The inspection below is authoritative.
                pass
            metadata = current.lstat()
        except OSError as error:
            raise ArtifactPathError(
                "Cannot inspect an artifact path component"
            ) from error
        attributes = getattr(metadata, "st_file_attributes", 0)
        if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ArtifactPathError("Refusing to traverse a reparsed artifact path")
        if not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(str(current))
    yield _PathConfinedDirectory(current)


@contextmanager
def _windows_open_confined_file_descriptor(
    root: _WindowsConfinedDirectory,
    relative_key: str,
    *,
    access: int,
    descriptor_flags: int,
    create: bool,
    exclusive_create: bool = False,
    allow_hardlinks: bool = False,
) -> Iterator[int]:
    if exclusive_create and not create:
        raise ValueError("exclusive confined file creation requires create=True")
    canonical_key = validated_artifact_relative_key(relative_key)
    parts = tuple(canonical_key.split("/"))
    with _windows_open_relative_directory(
        root,
        parts[:-1],
        create=create,
    ) as parent:
        handle = _windows_open_relative_handle(
            parent.handle,
            parts[-1],
            directory=False,
            create=create,
            exclusive_create=exclusive_create,
            access=access | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
            allow_hardlinks=allow_hardlinks,
        )
        descriptor = _windows_handle_to_descriptor(
            handle,
            descriptor_flags | _BINARY_OPEN_FLAG,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or (
                metadata.st_nlink != 1 and not allow_hardlinks
            ):
                raise ArtifactPathError("Artifact must be a single-link regular file")
            yield descriptor
        finally:
            os.close(descriptor)


@contextmanager
def _windows_open_confined_regular_file(
    root: _WindowsConfinedDirectory,
    relative_key: str,
    *,
    allow_hardlinks: bool = False,
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    with _windows_open_confined_file_descriptor(
        root,
        relative_key,
        access=_FILE_READ_DATA,
        descriptor_flags=os.O_RDONLY,
        create=False,
        allow_hardlinks=allow_hardlinks,
    ) as descriptor:
        stream = os.fdopen(os.dup(descriptor), "rb")
        try:
            yield stream, os.fstat(descriptor)
        finally:
            stream.close()


@contextmanager
def _windows_open_confined_lock_file(
    root: _WindowsConfinedDirectory,
    relative_key: str,
    *,
    exclusive_create: bool = False,
) -> Iterator[int]:
    with _windows_open_confined_file_descriptor(
        root,
        relative_key,
        access=_FILE_READ_DATA | _FILE_WRITE_DATA,
        descriptor_flags=os.O_RDWR,
        create=True,
        exclusive_create=exclusive_create,
    ) as descriptor:
        yield descriptor


def _windows_set_file_name(
    descriptor: int,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    encoded_name = str(destination).encode("utf-16-le")
    # Although FileNameLength excludes a terminator, SetFileInformationByHandle
    # has been observed to consume the following WCHAR for this information
    # class. Keep an explicit zero WCHAR in the backing allocation so the
    # variable-length name cannot acquire bytes from adjacent memory.
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
    information.ReplaceIfExists = overwrite
    information.RootDirectory = None
    information.FileNameLength = len(encoded_name)
    ctypes.memmove(
        ctypes.addressof(buffer) + _FileRenameInformation.FileName.offset,
        encoded_name,
        len(encoded_name),
    )
    handle = msvcrt.get_osfhandle(descriptor)
    if not _set_file_information(
        handle,
        _FILE_RENAME_INFO_CLASS,
        buffer,
        size,
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_mark_descriptor_for_delete(descriptor: int) -> None:
    information = _FileDispositionInformation(DeleteFile=True)
    handle = msvcrt.get_osfhandle(descriptor)
    if not _set_file_information(
        handle,
        _FILE_DISPOSITION_INFO_CLASS,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_try_open_existing_regular_file(
    parent: _WindowsConfinedDirectory,
    leaf_name: str,
) -> int | None:
    try:
        handle = _windows_open_relative_handle(
            parent.handle,
            leaf_name,
            directory=False,
            create=False,
            access=_FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
        )
    except FileNotFoundError:
        return None
    return handle


def _windows_write_bytes_to_confined(
    root: _WindowsConfinedDirectory,
    relative_key: str,
    data: bytes,
    *,
    overwrite: bool,
    file_mode: int,
) -> bool:
    del file_mode  # Windows privacy is governed by ACLs, not POSIX mode bits.
    canonical_key = validated_artifact_relative_key(relative_key)
    parts = tuple(canonical_key.split("/"))
    with _windows_open_relative_directory(
        root,
        parts[:-1],
        create=True,
    ) as parent:
        existing_handle = _windows_try_open_existing_regular_file(parent, parts[-1])
        if existing_handle is not None:
            _close_handle(existing_handle)
            if not overwrite:
                return False

        temporary_name = f".{parts[-1]}.{uuid4().hex}.tmp"
        handle = _windows_open_relative_handle(
            parent.handle,
            temporary_name,
            directory=False,
            create=True,
            exclusive_create=True,
            access=_FILE_WRITE_DATA | _DELETE | _SYNCHRONIZE,
        )
        descriptor = _windows_handle_to_descriptor(
            handle,
            os.O_WRONLY | _BINARY_OPEN_FLAG,
        )
        published = False
        try:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:  # pragma: no cover - regular-file invariant
                    raise OSError("Could not write artifact bytes")
                view = view[written:]
            os.fsync(descriptor)
            try:
                _windows_set_file_name(
                    descriptor,
                    parent.path / parts[-1],
                    overwrite=overwrite,
                )
            except FileExistsError:
                if overwrite:
                    raise
                existing_handle = _windows_try_open_existing_regular_file(
                    parent,
                    parts[-1],
                )
                if existing_handle is None:
                    raise ArtifactPathError(
                        "Artifact destination changed during publication"
                    ) from None
                _close_handle(existing_handle)
                return False
            published = True
            _windows_flush_confined_directory(parent)
            return True
        finally:
            if not published:
                try:
                    _windows_mark_descriptor_for_delete(descriptor)
                except OSError:
                    pass
            os.close(descriptor)


def _windows_copy_open_file_to_confined(
    root: _WindowsConfinedDirectory,
    relative_key: str,
    source: BinaryIO,
    source_metadata: os.stat_result,
    *,
    overwrite: bool,
) -> bool:
    canonical_key = validated_artifact_relative_key(relative_key)
    parts = tuple(canonical_key.split("/"))
    with _windows_open_relative_directory(root, parts[:-1], create=True) as parent:
        existing_handle = _windows_try_open_existing_regular_file(parent, parts[-1])
        if existing_handle is not None:
            _close_handle(existing_handle)
            if not overwrite:
                return False

        temporary_name = f".{parts[-1]}.{uuid4().hex}.tmp"
        handle = _windows_open_relative_handle(
            parent.handle,
            temporary_name,
            directory=False,
            create=True,
            exclusive_create=True,
            access=_FILE_WRITE_DATA | _DELETE | _SYNCHRONIZE,
        )
        descriptor = _windows_handle_to_descriptor(
            handle,
            os.O_WRONLY | _BINARY_OPEN_FLAG,
        )
        published = False
        try:
            before = os.fstat(source.fileno())
            expected_identity = (
                source_metadata.st_dev,
                source_metadata.st_ino,
                source_metadata.st_size,
                source_metadata.st_mtime_ns,
                source_metadata.st_ctime_ns,
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                != expected_identity
            ):
                raise ArtifactPathError("Artifact source identity changed")
            with os.fdopen(os.dup(descriptor), "wb") as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            after = os.fstat(source.fileno())
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != expected_identity:
                raise ArtifactPathError("Artifact source changed while copied")
            try:
                _windows_set_file_name(
                    descriptor,
                    parent.path / parts[-1],
                    overwrite=overwrite,
                )
            except FileExistsError:
                if overwrite:
                    raise
                existing_handle = _windows_try_open_existing_regular_file(
                    parent,
                    parts[-1],
                )
                if existing_handle is None:
                    raise ArtifactPathError(
                        "Artifact destination changed during publication"
                    ) from None
                _close_handle(existing_handle)
                return False
            published = True
            _windows_flush_confined_directory(parent)
            return True
        finally:
            if not published:
                try:
                    _windows_mark_descriptor_for_delete(descriptor)
                except OSError:
                    pass
            os.close(descriptor)


def _windows_append_bytes_to_confined(
    root: _WindowsConfinedDirectory,
    relative_key: str,
    data: bytes,
) -> None:
    from world_understanding.utils.file_locking import (
        blocking_exclusive_descriptor_lock,
    )

    with _windows_open_confined_file_descriptor(
        root,
        relative_key,
        access=_FILE_APPEND_DATA | _FILE_WRITE_DATA,
        descriptor_flags=os.O_WRONLY,
        create=True,
    ) as descriptor:
        with blocking_exclusive_descriptor_lock(descriptor):
            os.lseek(descriptor, 0, os.SEEK_END)
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:  # pragma: no cover - regular-file invariant
                    raise OSError("Could not append artifact bytes")
                view = view[written:]
            os.fsync(descriptor)
    canonical_key = validated_artifact_relative_key(relative_key)
    parts = tuple(canonical_key.split("/"))
    with _windows_open_relative_directory(
        root,
        parts[:-1],
        create=False,
    ) as parent:
        _windows_flush_confined_directory(parent)


def _windows_delete_confined_file(
    root: _WindowsConfinedDirectory,
    relative_key: str,
    *,
    missing_ok: bool,
) -> bool:
    canonical_key = validated_artifact_relative_key(relative_key)
    parts = tuple(canonical_key.split("/"))
    try:
        with _windows_open_relative_directory(
            root,
            parts[:-1],
            create=False,
        ) as parent:
            handle = _windows_open_relative_handle(
                parent.handle,
                parts[-1],
                directory=False,
                create=False,
                access=_DELETE | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
            )
            descriptor = _windows_handle_to_descriptor(
                handle,
                os.O_RDONLY | _BINARY_OPEN_FLAG,
            )
            try:
                _windows_mark_descriptor_for_delete(descriptor)
            finally:
                os.close(descriptor)
            _windows_flush_confined_directory(parent)
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    return True


@contextmanager
def _path_open_confined_directory(
    path: str | Path,
    *,
    create: bool,
    mode: int,
) -> Iterator[_PathConfinedDirectory]:
    """Verify and hold a directory where descriptors cannot confine it."""

    directory = Path(os.path.abspath(os.fspath(path)))
    if create:
        try:
            directory.mkdir(mode=mode, parents=True, exist_ok=True)
        except FileExistsError:
            pass
    _reject_reparse_points(directory)
    # Match what a descriptor open reports: absent paths are FileNotFoundError,
    # and only an existing non-directory is NotADirectoryError.
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(str(directory)) from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(str(directory))
    yield _PathConfinedDirectory(directory)


@contextmanager
def _path_open_confined_lock_file(
    root: _PathConfinedDirectory,
    relative_key: str,
    *,
    file_mode: int,
    exclusive_create: bool = False,
) -> Iterator[int]:
    """Open one regular lock file beneath a verified root."""

    target = _confined_child_path(root, relative_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_points(target.parent if exclusive_create else target)
    descriptor = os.open(
        target,
        os.O_CREAT
        | os.O_RDWR
        | (os.O_EXCL if exclusive_create else 0)
        | _BINARY_OPEN_FLAG,
        file_mode,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArtifactPathError("Lock artifact must be a regular file")
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _path_open_confined_regular_file(
    root: _PathConfinedDirectory,
    relative_key: str,
    *,
    allow_hardlinks: bool = False,
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Open one regular file beneath a verified path-backed root."""

    target = _confined_child_path(root, relative_key)
    _reject_reparse_points(target)
    descriptor = os.open(
        target,
        os.O_RDONLY | _BINARY_OPEN_FLAG | getattr(os, "O_NOINHERIT", 0),
    )
    stream: BinaryIO | None = None
    try:
        descriptor_metadata = os.fstat(descriptor)
        named_metadata = target.lstat()
        if (
            not stat.S_ISREG(descriptor_metadata.st_mode)
            or not stat.S_ISREG(named_metadata.st_mode)
            or (descriptor_metadata.st_nlink != 1 and not allow_hardlinks)
            or (named_metadata.st_nlink != 1 and not allow_hardlinks)
            or (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
            != (named_metadata.st_dev, named_metadata.st_ino)
        ):
            raise ArtifactPathError(
                "Artifact must be a single-link regular file beneath the confined root"
            )
        _reject_reparse_points(target)
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        yield stream, descriptor_metadata
    finally:
        if stream is not None:
            stream.close()
        if descriptor >= 0:
            os.close(descriptor)


def _path_write_bytes_to_confined(
    root: _PathConfinedDirectory,
    relative_key: str,
    data: bytes,
    *,
    overwrite: bool,
    file_mode: int,
) -> bool:
    """Publish bytes beneath a verified root through a same-directory rename."""

    target = _confined_child_path(root, relative_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_points(target)
    if not overwrite and target.exists():
        return False

    transaction = target.with_name(f"{target.name}.{uuid4().hex}.tmp")
    descriptor = os.open(
        transaction,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | _BINARY_OPEN_FLAG,
        file_mode,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - regular-file invariant
                raise OSError("Could not write artifact bytes")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        transaction.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    if overwrite:
        os.replace(transaction, target)
        return True
    # The existence check above is only a fast path: os.replace overwrites
    # unconditionally, so a writer that created the target in between would
    # be destroyed and this would still report success. os.link refuses to
    # clobber, which is the same no-clobber publish the descriptor backend
    # gets from confined_atomic_writer.
    try:
        os.link(transaction, target)
    except FileExistsError:
        return False
    finally:
        transaction.unlink(missing_ok=True)
    return True


def _path_append_bytes_to_confined(
    root: _PathConfinedDirectory,
    relative_key: str,
    data: bytes,
    *,
    file_mode: int,
) -> None:
    """Append bytes beneath a verified root where ``openat`` is unavailable."""

    target = _confined_child_path(root, relative_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_points(target)
    descriptor = os.open(
        target,
        os.O_APPEND
        | os.O_CREAT
        | os.O_WRONLY
        | _BINARY_OPEN_FLAG
        | getattr(os, "O_NOINHERIT", 0),
        file_mode,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ArtifactPathError(
                "Refusing to append to a non-regular or multiply linked artifact"
            )
        # Detect a reparse-point substitution that completed while the file was
        # being opened. The path backend's remaining check/use limitation is
        # documented on _PathConfinedDirectory.
        _reject_reparse_points(target)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - regular-file invariant
                raise OSError("Could not append artifact bytes")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_delete_confined_file(
    root: _PathConfinedDirectory,
    relative_key: str,
    *,
    missing_ok: bool,
) -> bool:
    """Delete one verified regular file where ``unlinkat`` is unavailable."""

    target = _confined_child_path(root, relative_key)
    try:
        _reject_reparse_points(target)
        metadata = target.lstat()
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ArtifactPathError(
            "Refusing to delete a non-regular or multiply linked artifact"
        )
    _reject_reparse_points(target.parent)
    target.unlink()
    return True


def _path_remove_confined_tree(target_path: Path) -> bool:
    """Remove one verified directory tree where descriptors cannot confine it."""

    try:
        metadata = target_path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise ValueError("Working directory cannot be a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Working directory must be a directory")
    _reject_reparse_points(target_path.parent)
    shutil.rmtree(target_path)
    return True


@contextmanager
def open_confined_directory(
    path: str | Path,
    *,
    create: bool = False,
    mode: int = 0o777,
) -> Iterator[int]:
    """Open a directory while holding and no-following every path component."""

    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        if _SUPPORTS_WINDOWS_HANDLE_CONFINEMENT:
            del mode  # Windows directory permissions are ACL-based.
            with _windows_open_absolute_directory(path, create=create) as root:
                yield root  # type: ignore[misc]
            return
        with _path_open_confined_directory(path, create=create, mode=mode) as root:
            yield root  # type: ignore[misc]
        return

    descriptor = os.open(os.sep, _DIRECTORY_OPEN_FLAGS)
    try:
        for component in _absolute_path_parts(path):
            try:
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=mode, dir_fd=descriptor)
                except FileExistsError:
                    # Another creator won. The no-follow open below is the
                    # authoritative type and confinement check.
                    pass
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ArtifactPathError(
                        "Refusing to traverse a symlinked artifact path"
                    ) from exc
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor
    finally:
        os.close(descriptor)


def confined_directory_identity(
    root_descriptor: int | _WindowsConfinedDirectory,
) -> tuple[int, int]:
    """Return the stable filesystem identity of one held confined directory."""

    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        if _SUPPORTS_WINDOWS_HANDLE_CONFINEMENT:
            if not isinstance(root_descriptor, _WindowsConfinedDirectory):
                raise ArtifactPathError(
                    "Confined root is not a held Windows directory handle"
                )
            return root_descriptor.identity()
        raise RuntimeError(
            "Directory identity requires descriptor or Windows handle confinement"
        )

    metadata = os.fstat(root_descriptor)  # type: ignore[arg-type]
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactPathError("Confined root is not a directory")
    return metadata.st_dev, metadata.st_ino


def fsync_directory(path: str | Path) -> None:
    """Persist a directory through the platform's confined handle primitive."""

    if _SUPPORTS_DIRECTORY_DESCRIPTORS:
        with open_confined_directory(path) as descriptor:
            os.fsync(descriptor)
        return
    if _SUPPORTS_WINDOWS_HANDLE_CONFINEMENT:
        with _windows_open_absolute_directory(path, create=False) as root:
            _windows_flush_confined_directory(root)
        return
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, directory_flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _open_relative_directory(
    root_descriptor: int,
    components: tuple[str, ...],
    *,
    create: bool,
    mode: int = 0o777,
    exclusive_create: bool = False,
) -> Iterator[int]:
    """Open relative directory components beneath an already-held root."""

    if exclusive_create and (not create or not components):
        raise ValueError(
            "exclusive confined directory creation requires a relative leaf"
        )
    descriptor = os.dup(root_descriptor)
    try:
        for index, component in enumerate(components):
            if exclusive_create and index == len(components) - 1:
                os.mkdir(component, mode=mode, dir_fd=descriptor)
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
                continue
            try:
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=mode, dir_fd=descriptor)
                except FileExistsError:
                    pass
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ArtifactPathError(
                        "Refusing to traverse a symlinked artifact path"
                    ) from exc
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def open_confined_directory_at(
    root_descriptor: int,
    relative_key: str,
    *,
    create: bool = False,
    mode: int = 0o777,
    exclusive_create: bool = False,
) -> Iterator[int]:
    """Open a canonical relative directory beneath an already-held root.

    When ``exclusive_create`` is true, missing ancestors may be created but the
    final directory component must not already exist.
    """

    if exclusive_create and not create:
        raise ValueError("exclusive confined directory creation requires create=True")

    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        if _SUPPORTS_WINDOWS_HANDLE_CONFINEMENT:
            del mode  # Windows directory permissions are ACL-based.
            canonical_key = validated_artifact_relative_key(relative_key)
            with _windows_open_relative_directory(
                root_descriptor,  # type: ignore[arg-type]
                tuple(canonical_key.split("/")),
                create=create,
                exclusive_create=exclusive_create,
            ) as directory:
                yield directory  # type: ignore[misc]
            return
        with _path_open_confined_directory_at(
            root_descriptor,  # type: ignore[arg-type]
            relative_key,
            create=create,
            mode=mode,
            exclusive_create=exclusive_create,
        ) as directory:
            yield directory  # type: ignore[misc]
        return

    canonical_key = validated_artifact_relative_key(relative_key)
    with _open_relative_directory(
        root_descriptor,
        tuple(canonical_key.split("/")),
        create=create,
        mode=mode,
        exclusive_create=exclusive_create,
    ) as descriptor:
        yield descriptor


@contextmanager
def open_confined_regular_file(
    root_descriptor: int,
    relative_key: str,
    *,
    allow_hardlinks: bool = False,
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Open and hold one canonical regular file beneath ``root_descriptor``."""

    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        if _SUPPORTS_WINDOWS_HANDLE_CONFINEMENT:
            with _windows_open_confined_regular_file(
                root_descriptor,  # type: ignore[arg-type]
                relative_key,
                allow_hardlinks=allow_hardlinks,
            ) as opened:
                yield opened
            return
        with _path_open_confined_regular_file(
            root_descriptor,  # type: ignore[arg-type]
            relative_key,
            allow_hardlinks=allow_hardlinks,
        ) as opened:
            yield opened
        return

    canonical_key = validated_artifact_relative_key(relative_key)
    parts = tuple(canonical_key.split("/"))
    with _open_relative_directory(
        root_descriptor,
        parts[:-1],
        create=False,
    ) as parent_descriptor:
        with open_confined_regular_file_leaf(
            parent_descriptor,
            parts[-1],
        ) as opened:
            yield opened


def open_confined_binary_writer(
    root_descriptor: int,
    relative_key: str,
) -> BinaryIO:
    """Create or truncate one confined regular file and return its held stream.

    Unlike the atomic publication helpers, this is intended for live logs whose
    bytes must remain visible while a child process is running. The returned leaf
    handle pins the validated file after the ancestor handle chain is released.
    """

    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        if not _SUPPORTS_WINDOWS_HANDLE_CONFINEMENT:
            raise RuntimeError(
                "Live confined writers require directory descriptors or Windows handles"
            )
        with _windows_open_confined_file_descriptor(
            root_descriptor,  # type: ignore[arg-type]
            relative_key,
            access=_FILE_WRITE_DATA,
            descriptor_flags=os.O_WRONLY,
            create=True,
        ) as descriptor:
            owned_descriptor = os.dup(descriptor)
        try:
            os.ftruncate(owned_descriptor, 0)
            os.lseek(owned_descriptor, 0, os.SEEK_SET)
            return os.fdopen(owned_descriptor, "wb")
        except BaseException:
            os.close(owned_descriptor)
            raise

    canonical_key = validated_artifact_relative_key(relative_key)
    parts = tuple(canonical_key.split("/"))
    with _open_relative_directory(
        root_descriptor,
        parts[:-1],
        create=True,
    ) as parent_descriptor:
        flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(parts[-1], flags, 0o600, dir_fd=parent_descriptor)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ArtifactPathError("Artifact must be a single-link regular file")
            os.ftruncate(descriptor, 0)
            return os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise


@contextmanager
def open_confined_regular_file_leaf(
    parent_descriptor: int,
    leaf_name: str,
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Open one Linux host leaf beneath an already-held parent descriptor.

    Unlike artifact keys, host leaf names may contain backslashes, colons, or
    the reserved pipeline-temp spelling. Directory traversal remains invalid.
    """

    if (
        type(leaf_name) is not str
        or not leaf_name
        or leaf_name in {".", ".."}
        or "/" in leaf_name
        or "\x00" in leaf_name
    ):
        raise ValueError("Artifact source leaf must be one exact host filename")
    try:
        descriptor = os.open(
            leaf_name,
            _FILE_READ_FLAGS,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ArtifactPathError("Refusing to read a symlinked artifact") from exc
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ArtifactPathError("Refusing to read a non-regular artifact")
    stream = os.fdopen(descriptor, "rb")
    try:
        yield stream, metadata
    finally:
        stream.close()


@contextmanager
def open_regular_file_no_follow(
    path: str | Path,
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Open a regular file while no-following and holding every ancestor."""

    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        absolute = Path(os.path.abspath(os.fspath(path)))
        with open_confined_directory(absolute.parent) as parent_descriptor:
            with open_confined_regular_file(
                parent_descriptor,
                absolute.name,
            ) as opened:
                yield opened
        return

    parts = _absolute_path_parts(path)
    if not parts:
        raise ArtifactPathError("Artifact source must be a regular file")
    parent = Path(os.sep).joinpath(*parts[:-1])
    with open_confined_directory(parent) as parent_descriptor:
        try:
            descriptor = os.open(
                parts[-1],
                _FILE_READ_FLAGS,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ArtifactPathError(
                    "Refusing to read a symlinked artifact source"
                ) from exc
            raise
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise ArtifactPathError("Artifact source must be a regular file")
        stream = os.fdopen(descriptor, "rb")
        try:
            yield stream, metadata
        finally:
            stream.close()


def visible_local_artifact_key(
    session_root: str | Path,
    key: str | Path,
) -> str | None:
    """Return a canonical session-relative key without resolving symlinks.

    This helper performs only lexical normalization. All filesystem access must
    subsequently use a descriptor-confined helper in this module; returning a
    pathname from a security check would reintroduce a check-then-use race.
    Symlinked artifacts are intentionally unsupported on direct-read surfaces.
    """
    if is_pipeline_temp_path(key):
        return None
    raw_key = os.fspath(key).replace("\\", "/")
    raw_path = Path(raw_key)
    if "\x00" in raw_key or ".." in raw_path.parts or PureWindowsPath(raw_key).drive:
        return None
    try:
        root = Path(os.path.abspath(os.fspath(session_root)))
        candidate = raw_path if raw_path.is_absolute() else root / raw_path
        relative = Path(os.path.abspath(os.fspath(candidate))).relative_to(root)
        return validated_artifact_relative_key(relative.as_posix())
    except (OSError, RuntimeError, ValueError):
        return None


def open_held_confined_artifact(
    session_root: str | Path,
    key: str | Path,
) -> OpenArtifactFile:
    """Open and hold one visible artifact through a stable dirfd chain."""
    relative_key = visible_local_artifact_key(session_root, key)
    if relative_key is None:
        raise FileNotFoundError("Artifact is not visible")
    with open_confined_directory(session_root) as root_descriptor:
        with open_confined_regular_file(
            root_descriptor,
            relative_key,
        ) as (source, metadata):
            descriptor = os.dup(source.fileno())
    return OpenArtifactFile(
        relative_key=relative_key,
        stream=os.fdopen(descriptor, "rb"),
        metadata=metadata,
    )


def confined_artifact_exists(
    session_root: str | Path,
    key: str | Path,
) -> bool:
    """Return whether one visible regular artifact can be opened safely."""
    try:
        artifact = open_held_confined_artifact(session_root, key)
    except (ArtifactPathError, FileNotFoundError, OSError, RuntimeError, ValueError):
        return False
    artifact.stream.close()
    return True


def read_confined_artifact_bytes(
    session_root: str | Path,
    key: str | Path,
) -> bytes:
    """Read one visible regular artifact from its already-confined descriptor."""
    artifact = open_held_confined_artifact(session_root, key)
    try:
        return artifact.stream.read()
    finally:
        artifact.stream.close()


def list_confined_artifact_keys(
    session_root: str | Path,
    *,
    prefix: str = "",
) -> list[str]:
    """List regular non-symlink artifacts through held directory descriptors."""
    if is_pipeline_temp_path(prefix):
        return []

    keys: list[str] = []

    def walk(directory_descriptor: int, parent_parts: tuple[str, ...]) -> None:
        with os.scandir(directory_descriptor) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            relative_parts = (*parent_parts, name)
            relative_key = "/".join(relative_parts)
            if name.casefold() == _PIPELINE_TEMP_COMPONENT or not _prefix_may_enter(
                relative_key,
                prefix,
            ):
                continue
            try:
                child_descriptor = os.open(
                    name,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=directory_descriptor,
                )
            except FileNotFoundError:
                continue
            except OSError as directory_error:
                if directory_error.errno not in {errno.ENOTDIR, errno.ELOOP}:
                    raise ArtifactPathError(
                        "Artifact changed during listing"
                    ) from directory_error
                try:
                    metadata = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(metadata.st_mode) and _prefix_matches(
                    relative_key,
                    prefix,
                ):
                    keys.append(validated_artifact_relative_key(relative_key))
                continue
            try:
                if not stat.S_ISDIR(os.fstat(child_descriptor).st_mode):
                    raise ArtifactPathError(
                        "Refusing to traverse a non-directory artifact"
                    )
                walk(child_descriptor, relative_parts)
            finally:
                os.close(child_descriptor)

    try:
        with open_confined_directory(session_root) as root_descriptor:
            walk(root_descriptor, ())
    except FileNotFoundError:
        return []
    return keys


def _prefix_matches(relative_key: str, prefix: str) -> bool:
    return not prefix or relative_key.startswith(prefix)


def _prefix_may_enter(relative_directory: str, prefix: str) -> bool:
    return (
        not prefix
        or relative_directory.startswith(prefix)
        or prefix.startswith(f"{relative_directory}/")
    )


def iter_open_regular_files(
    root_descriptor: int,
    *,
    prefix: str = "",
) -> Iterator[OpenArtifactFile]:
    """Yield held regular files beneath a held root without reopening paths."""

    def walk(
        directory_descriptor: int,
        parent_parts: tuple[str, ...],
    ) -> Iterator[OpenArtifactFile]:
        with os.scandir(directory_descriptor) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            relative_parts = (*parent_parts, name)
            relative_key = "/".join(relative_parts)
            if name.casefold() == _PIPELINE_TEMP_COMPONENT:
                continue
            potentially_selected = _prefix_may_enter(relative_key, prefix)
            if not potentially_selected:
                continue
            try:
                child_descriptor = os.open(
                    name,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=directory_descriptor,
                )
            except OSError as directory_error:
                if directory_error.errno not in {
                    errno.ENOTDIR,
                    errno.ELOOP,
                }:
                    raise ArtifactPathError(
                        "Artifact changed during traversal"
                    ) from directory_error
                if not _prefix_matches(relative_key, prefix):
                    continue
                validated_artifact_relative_key(relative_key)
                try:
                    file_descriptor = os.open(
                        name,
                        _FILE_READ_FLAGS,
                        dir_fd=directory_descriptor,
                    )
                except OSError as file_error:
                    if file_error.errno in {
                        errno.ELOOP,
                        errno.ENOTDIR,
                    }:
                        raise ArtifactPathError(
                            "Refusing to sync a symlinked session artifact"
                        ) from file_error
                    raise ArtifactPathError(
                        "Artifact changed during traversal"
                    ) from file_error
                metadata = os.fstat(file_descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    os.close(file_descriptor)
                    raise ArtifactPathError(
                        "Refusing to sync a special session artifact"
                    ) from directory_error
                stream = os.fdopen(file_descriptor, "rb")
                try:
                    yield OpenArtifactFile(relative_key, stream, metadata)
                finally:
                    stream.close()
            else:
                try:
                    metadata = os.fstat(child_descriptor)
                    if not stat.S_ISDIR(metadata.st_mode):
                        raise ArtifactPathError(
                            "Refusing to traverse a non-directory artifact"
                        )
                    yield from walk(child_descriptor, relative_parts)
                finally:
                    os.close(child_descriptor)

    yield from walk(root_descriptor, ())


def _validate_existing_destination(
    parent_descriptor: int,
    leaf_name: str,
) -> os.stat_result | None:
    try:
        metadata = os.stat(
            leaf_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        raise ArtifactPathError("Refusing to write a symlinked artifact destination")
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactPathError("Refusing to replace a non-regular artifact")
    return metadata


@contextmanager
def confined_atomic_writer(
    root_descriptor: int,
    relative_key: str,
    *,
    overwrite: bool,
    file_mode: int = 0o666,
    times_ns: tuple[int, int] | None = None,
    preserve_mode: bool = False,
) -> Iterator[ConfinedAtomicWrite]:
    """Write a canonical destination through held dirfds and publish atomically."""

    canonical_key = validated_artifact_relative_key(relative_key)
    parts = tuple(canonical_key.split("/"))
    with _open_relative_directory(
        root_descriptor,
        parts[:-1],
        create=True,
    ) as parent_descriptor:
        existing = _validate_existing_destination(
            parent_descriptor,
            parts[-1],
        )
        state = ConfinedAtomicWrite(stream=None)
        if existing is not None and not overwrite:
            yield state
            return

        temporary_name = f".{parts[-1]}.{uuid4().hex}.tmp"
        temporary_descriptor = os.open(
            temporary_name,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            file_mode,
            dir_fd=parent_descriptor,
        )
        stream = os.fdopen(temporary_descriptor, "wb")
        state.stream = stream
        try:
            yield state
            stream.flush()
            os.fsync(stream.fileno())
            if preserve_mode:
                # Preserve useful source mode bits without granting group/other
                # write access that the process umask would otherwise remove.
                os.fchmod(stream.fileno(), stat.S_IMODE(file_mode) & ~0o022)
            if times_ns is not None:
                os.utime(stream.fileno(), ns=times_ns)
            stream.close()

            if overwrite:
                os.replace(
                    temporary_name,
                    parts[-1],
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                state.published = True
            else:
                try:
                    os.link(
                        temporary_name,
                        parts[-1],
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    _validate_existing_destination(
                        parent_descriptor,
                        parts[-1],
                    )
                else:
                    state.published = True
        finally:
            if not stream.closed:
                stream.close()
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            if state.published:
                os.fsync(parent_descriptor)


def copy_open_file_to_confined(
    root_descriptor: int,
    relative_key: str,
    source: BinaryIO,
    source_metadata: os.stat_result,
    *,
    overwrite: bool,
) -> bool:
    """Atomically copy a held source stream beneath a held destination root."""

    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        if _SUPPORTS_WINDOWS_HANDLE_CONFINEMENT:
            return _windows_copy_open_file_to_confined(
                root_descriptor,  # type: ignore[arg-type]
                relative_key,
                source,
                source_metadata,
                overwrite=overwrite,
            )
        raise RuntimeError(
            "Atomic streaming copies require descriptor or Windows handle confinement"
        )

    state: ConfinedAtomicWrite
    with confined_atomic_writer(
        root_descriptor,
        relative_key,
        overwrite=overwrite,
        file_mode=stat.S_IMODE(source_metadata.st_mode),
        times_ns=(source_metadata.st_atime_ns, source_metadata.st_mtime_ns),
        preserve_mode=True,
    ) as state:
        if state.stream is not None:
            shutil.copyfileobj(source, state.stream)
    return state.published


def write_bytes_to_confined(
    root_descriptor: int,
    relative_key: str,
    data: bytes,
    *,
    overwrite: bool = True,
    file_mode: int = 0o666,
) -> bool:
    """Atomically publish bytes beneath a held destination root."""

    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        if _SUPPORTS_WINDOWS_HANDLE_CONFINEMENT:
            return _windows_write_bytes_to_confined(
                root_descriptor,  # type: ignore[arg-type]
                relative_key,
                data,
                overwrite=overwrite,
                file_mode=file_mode,
            )
        return _path_write_bytes_to_confined(
            root_descriptor,  # type: ignore[arg-type]
            relative_key,
            data,
            overwrite=overwrite,
            file_mode=file_mode,
        )

    state: ConfinedAtomicWrite
    with confined_atomic_writer(
        root_descriptor,
        relative_key,
        overwrite=overwrite,
        file_mode=file_mode,
    ) as state:
        if state.stream is not None:
            view = memoryview(data)
            while view:
                written = os.write(state.stream.fileno(), view)
                if written <= 0:  # pragma: no cover - regular-file invariant
                    raise OSError("Could not write artifact bytes")
                view = view[written:]
    return state.published


def delete_confined_file(
    root_descriptor: int,
    relative_key: str,
    *,
    missing_ok: bool = True,
) -> bool:
    """Delete a regular file beneath a held root without following aliases."""

    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        if _SUPPORTS_WINDOWS_HANDLE_CONFINEMENT:
            return _windows_delete_confined_file(
                root_descriptor,  # type: ignore[arg-type]
                relative_key,
                missing_ok=missing_ok,
            )
        return _path_delete_confined_file(
            root_descriptor,  # type: ignore[arg-type]
            relative_key,
            missing_ok=missing_ok,
        )

    canonical_key = validated_artifact_relative_key(relative_key)
    parts = tuple(canonical_key.split("/"))
    try:
        with _open_relative_directory(
            root_descriptor,
            parts[:-1],
            create=False,
        ) as parent_descriptor:
            metadata = _validate_existing_destination(
                parent_descriptor,
                parts[-1],
            )
            if metadata is None:
                if missing_ok:
                    return False
                raise FileNotFoundError(relative_key)
            os.unlink(parts[-1], dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            return True
    except FileNotFoundError:
        if missing_ok:
            return False
        raise


def append_bytes_to_confined(
    root_descriptor: int,
    relative_key: str,
    data: bytes,
    *,
    file_mode: int = 0o666,
) -> None:
    """Append bytes through held dirfds to a no-followed regular file."""

    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        if _SUPPORTS_WINDOWS_HANDLE_CONFINEMENT:
            del file_mode  # Windows privacy is governed by ACLs.
            _windows_append_bytes_to_confined(
                root_descriptor,  # type: ignore[arg-type]
                relative_key,
                data,
            )
            return
        _path_append_bytes_to_confined(
            root_descriptor,  # type: ignore[arg-type]
            relative_key,
            data,
            file_mode=file_mode,
        )
        return

    canonical_key = validated_artifact_relative_key(relative_key)
    parts = tuple(canonical_key.split("/"))
    with _open_relative_directory(
        root_descriptor,
        parts[:-1],
        create=True,
    ) as parent_descriptor:
        _validate_existing_destination(
            parent_descriptor,
            parts[-1],
        )
        descriptor = os.open(
            parts[-1],
            os.O_APPEND
            | os.O_CREAT
            | os.O_WRONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            file_mode,
            dir_fd=parent_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ArtifactPathError("Refusing to append to a non-regular artifact")
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:  # pragma: no cover - regular-file invariant
                    raise OSError("Could not append artifact bytes")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        # Persist either a newly created directory entry or the directory state
        # observed after a concurrent creator won the O_CREAT race.
        os.fsync(parent_descriptor)


@contextmanager
def open_confined_lock_file(
    root_descriptor: int,
    relative_key: str,
    *,
    file_mode: int = 0o600,
    exclusive_create: bool = False,
) -> Iterator[int]:
    """Open and hold one regular lock file beneath a held root.

    When ``exclusive_create`` is true, atomically create the leaf and raise
    ``FileExistsError`` rather than reopening an existing lock file.
    """

    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        if _SUPPORTS_WINDOWS_HANDLE_CONFINEMENT:
            del file_mode  # Windows privacy is governed by ACLs.
            with _windows_open_confined_lock_file(
                root_descriptor,  # type: ignore[arg-type]
                relative_key,
                exclusive_create=exclusive_create,
            ) as descriptor:
                yield descriptor
            return
        with _path_open_confined_lock_file(
            root_descriptor,  # type: ignore[arg-type]
            relative_key,
            file_mode=file_mode,
            exclusive_create=exclusive_create,
        ) as descriptor:
            yield descriptor
        return

    canonical_key = validated_artifact_relative_key(relative_key)
    parts = tuple(canonical_key.split("/"))
    with _open_relative_directory(
        root_descriptor,
        parts[:-1],
        create=True,
    ) as parent_descriptor:
        if not exclusive_create:
            _validate_existing_destination(
                parent_descriptor,
                parts[-1],
            )
        descriptor = os.open(
            parts[-1],
            os.O_CREAT
            | os.O_RDWR
            | (os.O_EXCL if exclusive_create else 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            file_mode,
            dir_fd=parent_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ArtifactPathError("Lock artifact must be a regular file")
            # Persist either a newly created directory entry or the directory
            # state observed after a concurrent creator won the O_CREAT race.
            os.fsync(parent_descriptor)
            yield descriptor
        finally:
            os.close(descriptor)


def prune_confined_snapshot(
    root_descriptor: int,
    prefix: str,
    source_relative_keys: set[str],
) -> None:
    """Prune one local snapshot without following or reopening path aliases."""

    canonical_source_keys = {
        validated_artifact_relative_key(key) for key in source_relative_keys
    }

    def prune(
        directory_descriptor: int,
        parent_parts: tuple[str, ...],
    ) -> None:
        directory_changed = False
        with os.scandir(directory_descriptor) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            relative_parts = (*parent_parts, name)
            relative_key = "/".join(relative_parts)
            if not _prefix_may_enter(relative_key, prefix):
                continue
            try:
                child_descriptor = os.open(
                    name,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=directory_descriptor,
                )
            except OSError as directory_error:
                if directory_error.errno not in {
                    errno.ENOTDIR,
                    errno.ELOOP,
                }:
                    if directory_error.errno == errno.ENOENT:
                        continue
                    raise ArtifactPathError(
                        "Artifact changed during snapshot pruning"
                    ) from directory_error
                if (
                    _prefix_matches(relative_key, prefix)
                    and relative_key not in canonical_source_keys
                ):
                    try:
                        os.unlink(name, dir_fd=directory_descriptor)
                    except FileNotFoundError:
                        pass
                    else:
                        directory_changed = True
                continue

            try:
                prune(child_descriptor, relative_parts)
            finally:
                os.close(child_descriptor)
            if _prefix_matches(relative_key, prefix) or relative_key == prefix.rstrip(
                "/"
            ):
                try:
                    os.rmdir(name, dir_fd=directory_descriptor)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                        raise
                else:
                    directory_changed = True

        if directory_changed:
            os.fsync(directory_descriptor)

    prune(root_descriptor, ())


def remove_confined_tree(
    working_dir: str | Path,
    allowed_root: str | Path,
) -> bool:
    """Remove one owned directory tree without following a swapped component."""

    root_path = Path(os.path.abspath(os.fspath(allowed_root)))
    target_path = Path(os.path.abspath(os.fspath(working_dir)))
    try:
        relative = target_path.relative_to(root_path)
    except ValueError:
        raise ValueError(
            "Working directory is outside the configured cleanup root"
        ) from None
    if not relative.parts:
        raise ValueError("Working directory must be a child of the cleanup root")
    if not _SUPPORTS_DIRECTORY_DESCRIPTORS:
        return _path_remove_confined_tree(target_path)
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("Descriptor-safe recursive cleanup is unavailable")

    try:
        with open_confined_directory(root_path) as root_descriptor:
            with _open_relative_directory(
                root_descriptor,
                tuple(relative.parts[:-1]),
                create=False,
            ) as parent_descriptor:
                leaf_name = relative.parts[-1]
                metadata = os.stat(
                    leaf_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(metadata.st_mode):
                    raise ValueError("Working directory cannot be a symlink")
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError("Working directory must be a directory")
                shutil.rmtree(leaf_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
                return True
    except FileNotFoundError:
        return False


def confined_cleanup_path(
    working_dir: str | Path,
    allowed_root: str | Path,
) -> Path:
    """Canonicalize a recursive-clean target under an explicit ownership root."""
    try:
        root = Path(allowed_root).resolve(strict=False)
        target = Path(working_dir).resolve(strict=False)
        relative = target.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise ValueError(
            "Working directory is outside the configured cleanup root"
        ) from None
    if not relative.parts:
        raise ValueError("Working directory must be a child of the cleanup root")
    return target


def remove_legacy_pipeline_temp(working_dir: str | Path) -> bool:
    """Remove the exact legacy temp-config entry beneath a pipeline workdir.

    The old handoff directory may contain credentials from a pre-fix run. A
    symlink (including a broken symlink) or non-directory entry is unlinked;
    only a real directory is traversed. Cleanup errors intentionally propagate
    so resume cannot continue while a known credential-bearing artifact remains.
    """
    target = Path(working_dir) / ".pipeline_temp"
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        return False

    if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
        shutil.rmtree(target)
    else:
        target.unlink()
    return True
