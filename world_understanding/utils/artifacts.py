# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact-path safety helpers.

The descriptor-based helpers in this module intentionally target the project's
supported Linux/WSL2 runtime.  They keep every traversed directory descriptor
open and use ``O_NOFOLLOW`` for both ancestors and leaves so a path swap cannot
redirect a read or write after validation.
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
from typing import BinaryIO
from uuid import uuid4

_PIPELINE_TEMP_COMPONENT = ".pipeline_temp"
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | os.O_DIRECTORY
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


@contextmanager
def open_confined_directory(
    path: str | Path,
    *,
    create: bool = False,
    mode: int = 0o777,
) -> Iterator[int]:
    """Open a directory while holding and no-following every path component."""

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


@contextmanager
def _open_relative_directory(
    root_descriptor: int,
    components: tuple[str, ...],
    *,
    create: bool,
    mode: int = 0o777,
) -> Iterator[int]:
    """Open relative directory components beneath an already-held root."""

    descriptor = os.dup(root_descriptor)
    try:
        for component in components:
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
) -> Iterator[int]:
    """Open a canonical relative directory beneath an already-held root."""

    canonical_key = validated_artifact_relative_key(relative_key)
    with _open_relative_directory(
        root_descriptor,
        tuple(canonical_key.split("/")),
        create=create,
        mode=mode,
    ) as descriptor:
        yield descriptor


@contextmanager
def open_confined_regular_file(
    root_descriptor: int,
    relative_key: str,
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Open and hold one canonical regular file beneath ``root_descriptor``."""

    canonical_key = validated_artifact_relative_key(relative_key)
    parts = tuple(canonical_key.split("/"))
    with _open_relative_directory(
        root_descriptor,
        parts[:-1],
        create=False,
    ) as parent_descriptor:
        try:
            descriptor = os.open(
                parts[-1],
                _FILE_READ_FLAGS,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ArtifactPathError(
                    "Refusing to read a symlinked artifact"
                ) from exc
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
) -> Iterator[int]:
    """Open and hold one regular lock file beneath a held root."""

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
            os.O_CREAT
            | os.O_RDWR
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
