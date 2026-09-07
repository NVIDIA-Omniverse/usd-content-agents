# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic JSON and content-addressed artifact helpers."""

from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import stat
import tempfile
import threading
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    copy_open_file_to_confined,
    delete_confined_file,
    fsync_directory,
    open_confined_binary_writer,
    open_confined_directory,
    open_confined_regular_file,
    write_bytes_to_confined,
)

_DIGEST_SCHEMA = "content-agent-workflows.artifact-set-digest.v1"
_CHUNK_SIZE = 1024 * 1024
_BINARY_OPEN_FLAG = getattr(os, "O_BINARY", 0)
_MAX_DURABLY_SYNCED_DIRECTORIES = 1024
_DURABLY_SYNCED_DIRECTORY_IDENTITIES: dict[str, tuple[int, int, int]] = {}
_PENDING_DURABLE_DIRECTORY_BOUNDARIES: dict[str, Path] = {}
_DURABLE_DIRECTORY_LOCK = threading.Lock()


def _stable_ctime_ns(metadata: os.stat_result) -> int:
    """Exclude Windows CRT ctime jitter from same-handle identity checks."""

    return 0 if os.name == "nt" else metadata.st_ctime_ns


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        fsync_directory(path)
        return
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _directory_identity(path: Path) -> tuple[int, int, int]:
    metadata = path.stat()
    return (metadata.st_dev, metadata.st_ino, _stable_ctime_ns(metadata))


def _record_durable_directory_locked(path: Path) -> None:
    key = str(path)
    if (
        key not in _DURABLY_SYNCED_DIRECTORY_IDENTITIES
        and len(_DURABLY_SYNCED_DIRECTORY_IDENTITIES) >= _MAX_DURABLY_SYNCED_DIRECTORIES
    ):
        _DURABLY_SYNCED_DIRECTORY_IDENTITIES.clear()
    _DURABLY_SYNCED_DIRECTORY_IDENTITIES[key] = _directory_identity(path)


def _remember_durable_directory(path: Path) -> None:
    """Refresh the cached generation only after its final directory fsync."""

    with _DURABLE_DIRECTORY_LOCK:
        _record_durable_directory_locked(path)


def _create_directory_tree_durably(path: Path) -> None:
    """Create ``path`` and durably repair its same-filesystem ancestry once."""

    windows_existing_boundary: Path | None = None
    if os.name == "nt":
        # Windows directory handles cannot generally be opened with
        # FILE_WRITE_DATA all the way through inherited profile ancestors.
        # Only the directories created by this call and their first existing
        # parent participate in this publication, so remember that durability
        # boundary before mkdir fills in the missing components.
        windows_existing_boundary = path
        while True:
            try:
                windows_existing_boundary.lstat()
            except FileNotFoundError:
                parent = windows_existing_boundary.parent
                if parent == windows_existing_boundary:
                    break
                windows_existing_boundary = parent
                continue
            break

    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    identity = _directory_identity(path)
    with _DURABLE_DIRECTORY_LOCK:
        cache_key = str(path)
        if existed and _DURABLY_SYNCED_DIRECTORY_IDENTITIES.get(cache_key) == identity:
            _PENDING_DURABLE_DIRECTORY_BOUNDARIES.pop(cache_key, None)
            return
        # A failed repair must not leave an older, colliding generation cached.
        _DURABLY_SYNCED_DIRECTORY_IDENTITIES.pop(cache_key, None)
        if windows_existing_boundary is not None:
            if (
                cache_key not in _PENDING_DURABLE_DIRECTORY_BOUNDARIES
                and len(_PENDING_DURABLE_DIRECTORY_BOUNDARIES)
                >= _MAX_DURABLY_SYNCED_DIRECTORIES
            ):
                _PENDING_DURABLE_DIRECTORY_BOUNDARIES.clear()
            windows_existing_boundary = (
                _PENDING_DURABLE_DIRECTORY_BOUNDARIES.setdefault(
                    cache_key,
                    windows_existing_boundary,
                )
            )
        current = path
        current_device = identity[0]
        while True:
            _fsync_directory(current)
            if current == windows_existing_boundary:
                break
            parent = current.parent
            if parent == current:
                break
            parent_device = parent.stat().st_dev
            # The mount point is the durability root for this filesystem.
            if parent_device != current_device:
                break
            current = parent
            current_device = parent_device
        _record_durable_directory_locked(path)
        _PENDING_DURABLE_DIRECTORY_BOUNDARIES.pop(cache_key, None)


@dataclass(frozen=True, slots=True)
class ContainedArtifactRead:
    """Stable metadata derived from one held, no-follow file descriptor."""

    path: Path
    sha256: str
    size_bytes: int
    data: bytes | None = None
    json_object: dict[str, Any] | None = None


def _write_all(fd: int, payload: bytes) -> None:
    """Write a whole payload, tolerating short writes."""

    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("Atomic artifact write made no forward progress")
        view = view[written:]


def _read_regular_fd(
    file_fd: int,
    artifact_path: Path,
    *,
    max_bytes: int | None,
    capture_bytes: bool,
    copy_fd: int | None = None,
    allow_hardlinks: bool = False,
) -> tuple[str, int, bytes | None]:
    """Read and digest one already-open regular file without path reopens.

    ``copy_fd`` streams the same validated bytes to an open destination so a
    caller can duplicate a large artifact without holding it in memory.
    """

    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")

    before = os.fstat(file_fd)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Workflow artifact is not a regular file: {artifact_path}")
    if before.st_nlink > 1 and not allow_hardlinks:
        # O_NOFOLLOW stops symlinks but not a same-UID hard link, which would
        # let a contained path alias bytes that live outside the run root.
        # nlink == 0 stays readable: an unlink after open cannot redirect the
        # already-validated descriptor. ``allow_hardlinks`` is reserved for
        # caller-owned source inputs read outside a child-writable run root.
        raise ValueError(
            f"Workflow artifact must be a single-link regular file: {artifact_path}"
        )
    if max_bytes is not None and before.st_size > max_bytes:
        raise ValueError(
            f"Workflow artifact exceeds {max_bytes} bytes: {artifact_path} "
            f"({before.st_size} bytes)"
        )

    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture_bytes else None
    size_bytes = 0
    while True:
        read_size = _CHUNK_SIZE
        if max_bytes is not None:
            read_size = min(read_size, max_bytes - size_bytes + 1)
        chunk = os.read(file_fd, read_size)
        if not chunk:
            break
        size_bytes += len(chunk)
        if max_bytes is not None and size_bytes > max_bytes:
            raise ValueError(
                f"Workflow artifact exceeds {max_bytes} bytes: {artifact_path} "
                f"(more than {max_bytes} bytes)"
            )
        digest.update(chunk)
        if copy_fd is not None:
            _write_all(copy_fd, chunk)
        if chunks is not None:
            chunks.append(chunk)

    after = os.fstat(file_fd)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        _stable_ctime_ns(before),
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        _stable_ctime_ns(after),
    )
    if identity_before != identity_after or size_bytes != after.st_size:
        raise ValueError(
            f"Workflow artifact changed while it was being read: {artifact_path}"
        )
    if after.st_nlink > 1 and not allow_hardlinks:
        raise ValueError(
            f"Workflow artifact gained a hard link while it was being read: "
            f"{artifact_path}"
        )

    return (
        digest.hexdigest(),
        size_bytes,
        b"".join(chunks) if chunks is not None else None,
    )


def _contained_artifact_location(
    run_dir: str | Path,
    candidate: str | Path,
) -> tuple[Path, Path]:
    """Return the trusted root and a validated lexical relative path."""

    root = Path(run_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Workflow run root is not a directory: {root}")
    value = Path(candidate).expanduser()
    path = value if value.is_absolute() else root / value
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Artifact path is outside the workflow run: {path}") from exc
    if not relative.parts:
        raise ValueError(f"Workflow artifact is not a regular file: {path}")
    for component in relative.parts:
        if component in {"", ".", ".."}:
            raise ValueError(f"Unsafe workflow artifact component: {component!r}")
    return root, relative


def read_contained_artifact(
    run_dir: str | Path,
    candidate: str | Path,
    *,
    max_bytes: int | None = None,
    image: bool = False,
    parse_json: bool = False,
    capture_bytes: bool = False,
    copy_fd: int | None = None,
    allow_hardlinks: bool = False,
) -> ContainedArtifactRead:
    """Read and validate one run-local artifact through held descriptors.

    Every path component is opened relative to the prior directory descriptor
    with ``O_NOFOLLOW``. The digest, byte count, optional JSON object, and
    optional image validation are therefore all derived from the same opened
    file rather than from a path that can be substituted between checks.
    ``capture_bytes`` returns those exact bytes for callers that must parse a
    JSON shape other than an object or process bounded text. ``copy_fd``
    streams them to an already-open destination instead, so duplicating a
    multi-hundred-megabyte stage does not require buffering it.
    """

    root, relative = _contained_artifact_location(run_dir, candidate)
    artifact_path = root / relative
    if os.name != "posix":
        try:
            with open_confined_directory(root) as root_descriptor:
                with open_confined_regular_file(
                    root_descriptor,
                    relative.as_posix(),
                    allow_hardlinks=allow_hardlinks,
                ) as (source, _metadata):
                    digest, size_bytes, data = _read_regular_fd(
                        source.fileno(),
                        artifact_path,
                        max_bytes=max_bytes,
                        capture_bytes=image or parse_json or capture_bytes,
                        copy_fd=copy_fd,
                        allow_hardlinks=allow_hardlinks,
                    )
        except FileNotFoundError as exc:
            raise ValueError(
                f"Workflow artifact does not exist: {artifact_path}"
            ) from exc
        except (ArtifactPathError, OSError) as exc:
            raise ValueError(
                "Workflow artifact path must not contain symlinks and must "
                f"name a regular file: {artifact_path}"
            ) from exc
    else:
        try:
            parent_fd = _open_contained_directory(root, relative.parent, create=False)
        except FileNotFoundError as exc:
            raise ValueError(
                f"Workflow artifact does not exist: {artifact_path}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                "Workflow artifact path must not contain symlinks or "
                f"non-directories: {artifact_path}"
            ) from exc

        file_fd = -1
        try:
            file_flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                file_fd = os.open(relative.name, file_flags, dir_fd=parent_fd)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"Workflow artifact does not exist: {artifact_path}"
                ) from exc
            except OSError as exc:
                raise ValueError(
                    "Workflow artifact path must not contain symlinks and must "
                    f"name a regular file: {artifact_path}"
                ) from exc
            digest, size_bytes, data = _read_regular_fd(
                file_fd,
                artifact_path,
                max_bytes=max_bytes,
                capture_bytes=image or parse_json or capture_bytes,
                copy_fd=copy_fd,
                allow_hardlinks=allow_hardlinks,
            )
        finally:
            if file_fd >= 0:
                os.close(file_fd)
            os.close(parent_fd)

    if image:
        assert data is not None
        try:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as opened:
                opened.verify()
        except Exception as exc:
            raise ValueError(
                f"Workflow image artifact is not decodable: {artifact_path}"
            ) from exc

    json_object: dict[str, Any] | None = None
    if parse_json:
        assert data is not None
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Workflow artifact is not valid UTF-8 JSON: {artifact_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {artifact_path}")
        json_object = payload

    return ContainedArtifactRead(
        path=artifact_path,
        sha256=digest,
        size_bytes=size_bytes,
        data=data if capture_bytes else None,
        json_object=json_object,
    )


def contained_regular_file(
    run_dir: str | Path,
    candidate: str | Path,
    *,
    max_bytes: int | None = None,
    image: bool = False,
) -> Path:
    """Return the path of a strict run-local regular file.

    Workflow traces and replay manifests are trust boundaries: child-authored
    artifact names must not make the wrapper disclose host files. Validation is
    descriptor-safe, but callers that need file contents or metadata from the
    validated file must use :func:`read_contained_artifact` rather than reopening
    the returned path.
    """

    return read_contained_artifact(
        run_dir,
        candidate,
        max_bytes=max_bytes,
        image=image,
    ).path


def snapshot_contained_artifact(
    run_dir: str | Path,
    source: str | Path,
    destination: str | Path,
    *,
    destination_run_dir: str | Path | None = None,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    allow_hardlinks: bool = True,
) -> ContainedArtifactRead:
    """Copy one contained artifact into an immutable no-follow snapshot.

    Both source and destination are opened relative to held directory
    descriptors. The source identity is checked before and after the streaming
    copy, and the destination is installed atomically only after its bytes match
    the caller's recorded digest and size. ``destination_run_dir`` permits a
    separately owned snapshot root; omitting it preserves the original
    same-workflow-root behavior.
    """

    root, source_relative = _contained_artifact_location(run_dir, source)
    destination_root, destination_relative = _contained_artifact_location(
        destination_run_dir if destination_run_dir is not None else run_dir,
        destination,
    )
    if destination_root == root and source_relative == destination_relative:
        raise ValueError("Checkpoint snapshot must not replace its source artifact")

    source_path = root / source_relative
    destination_path = destination_root / destination_relative
    if os.name != "posix":
        staging_relative = destination_relative.with_name(
            f".{destination_relative.name}.{secrets.token_hex(8)}.snapshot"
        )
        staging_path = destination_root / staging_relative
        staging_published = False
        validation_error: str | None = None
        try:
            with open_confined_directory(root) as source_root_descriptor:
                with open_confined_directory(
                    destination_root
                ) as destination_root_descriptor:
                    try:
                        with open_confined_regular_file(
                            source_root_descriptor,
                            source_relative.as_posix(),
                            allow_hardlinks=allow_hardlinks,
                        ) as (source_stream, source_metadata):
                            if not copy_open_file_to_confined(
                                destination_root_descriptor,
                                staging_relative.as_posix(),
                                source_stream,
                                source_metadata,
                                overwrite=False,
                            ):
                                raise RuntimeError(
                                    "Checkpoint staging destination already exists"
                                )
                        staging_published = True
                        with open_confined_regular_file(
                            destination_root_descriptor,
                            staging_relative.as_posix(),
                        ) as (staging_stream, staging_metadata):
                            sha256, size_bytes, _data = _read_regular_fd(
                                staging_stream.fileno(),
                                staging_path,
                                max_bytes=None,
                                capture_bytes=False,
                            )
                            if (
                                expected_sha256 is not None
                                and sha256 != expected_sha256
                            ):
                                validation_error = (
                                    "Checkpoint source digest changed before sealing: "
                                    f"{source_path}"
                                )
                            elif (
                                expected_size_bytes is not None
                                and size_bytes != expected_size_bytes
                            ):
                                validation_error = (
                                    "Checkpoint source size changed before sealing: "
                                    f"{source_path}"
                                )
                            else:
                                staging_stream.seek(0)
                                copy_open_file_to_confined(
                                    destination_root_descriptor,
                                    destination_relative.as_posix(),
                                    staging_stream,
                                    staging_metadata,
                                    overwrite=True,
                                )
                    finally:
                        if staging_published:
                            delete_confined_file(
                                destination_root_descriptor,
                                staging_relative.as_posix(),
                            )
        except (ArtifactPathError, OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                "Could not create a confined checkpoint snapshot: "
                f"{source_path} -> {destination_path}"
            ) from exc
        if validation_error is not None:
            raise ValueError(validation_error)
        return ContainedArtifactRead(
            path=destination_path,
            sha256=sha256,
            size_bytes=size_bytes,
        )
    try:
        source_parent_fd = _open_contained_directory(
            root,
            source_relative.parent,
            create=False,
        )
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(
            "Checkpoint source path must not contain symlinks or "
            f"non-directories: {source_path}"
        ) from exc

    source_fd = -1
    destination_parent_fd = -1
    temporary_fd = -1
    temporary_name = f".{destination_relative.name}.{secrets.token_hex(8)}.tmp"
    try:
        source_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            source_fd = os.open(
                source_relative.name,
                source_flags,
                dir_fd=source_parent_fd,
            )
        except OSError as exc:
            raise ValueError(
                "Checkpoint source must be a regular file and not a symlink: "
                f"{source_path}"
            ) from exc
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Checkpoint source is not a regular file: {source_path}")
        if before.st_nlink > 1 and not allow_hardlinks:
            raise ValueError(
                f"Checkpoint source must be a single-link regular file: {source_path}"
            )

        try:
            destination_parent_fd = _open_contained_directory(
                destination_root,
                destination_relative.parent,
                create=True,
            )
        except OSError as exc:
            raise ValueError(
                "Checkpoint destination path must not contain symlinks or "
                f"non-directories: {destination_path}"
            ) from exc
        try:
            existing = os.stat(
                destination_relative.name,
                dir_fd=destination_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError(
                "Checkpoint destination must be a regular file or absent: "
                f"{destination_path}"
            )

        destination_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        temporary_fd = os.open(
            temporary_name,
            destination_flags,
            0o600,
            dir_fd=destination_parent_fd,
        )
        digest = hashlib.sha256()
        size_bytes = 0
        while True:
            chunk = os.read(source_fd, _CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(temporary_fd, remaining)
                if written <= 0:
                    raise OSError("Could not write checkpoint snapshot")
                remaining = remaining[written:]

        after = os.fstat(source_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            _stable_ctime_ns(before),
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            _stable_ctime_ns(after),
        )
        if identity_before != identity_after or size_bytes != after.st_size:
            raise ValueError(
                f"Checkpoint source changed while it was copied: {source_path}"
            )
        if after.st_nlink > 1 and not allow_hardlinks:
            raise ValueError(
                f"Checkpoint source gained a hard link while it was copied: {source_path}"
            )
        sha256 = digest.hexdigest()
        if expected_sha256 is not None and sha256 != expected_sha256:
            raise ValueError(
                f"Checkpoint source digest changed before sealing: {source_path}"
            )
        if expected_size_bytes is not None and size_bytes != expected_size_bytes:
            raise ValueError(
                f"Checkpoint source size changed before sealing: {source_path}"
            )

        os.fsync(temporary_fd)
        os.fchmod(temporary_fd, 0o400)
        os.close(temporary_fd)
        temporary_fd = -1
        os.replace(
            temporary_name,
            destination_relative.name,
            src_dir_fd=destination_parent_fd,
            dst_dir_fd=destination_parent_fd,
        )
        try:
            os.fsync(destination_parent_fd)
        except OSError:
            pass
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if destination_parent_fd >= 0:
            try:
                os.unlink(temporary_name, dir_fd=destination_parent_fd)
            except FileNotFoundError:
                pass
            os.close(destination_parent_fd)
        if source_fd >= 0:
            os.close(source_fd)
        os.close(source_parent_fd)

    return ContainedArtifactRead(
        path=destination_path,
        sha256=sha256,
        size_bytes=size_bytes,
    )


def open_contained_text_writer(
    run_dir: str | Path,
    candidate: str | Path,
) -> TextIO:
    """Open one contained regular file for truncating text output, no-follow."""

    root, relative = _contained_artifact_location(run_dir, candidate)
    artifact_path = root / relative
    if os.name != "posix":
        try:
            with open_confined_directory(root) as root_descriptor:
                binary_stream = open_confined_binary_writer(
                    root_descriptor,
                    relative.as_posix(),
                )
            return io.TextIOWrapper(
                binary_stream,
                encoding="utf-8",
                errors="replace",
            )
        except (ArtifactPathError, OSError) as exc:
            raise ValueError(
                "Contained output target must be a regular file or absent: "
                f"{artifact_path}"
            ) from exc
    try:
        parent_fd = _open_contained_directory(root, relative.parent, create=True)
    except OSError as exc:
        raise ValueError(
            "Contained output path must not contain symlinks or "
            f"non-directories: {artifact_path}"
        ) from exc

    file_fd = -1
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_TRUNC
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            file_fd = os.open(relative.name, flags, 0o600, dir_fd=parent_fd)
        except OSError as exc:
            raise ValueError(
                "Contained output target must be a regular file or absent: "
                f"{artifact_path}"
            ) from exc
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ValueError(
                f"Contained output target must be a regular file: {artifact_path}"
            )
        os.fchmod(file_fd, 0o600)
        stream = os.fdopen(file_fd, "w", encoding="utf-8", errors="replace")
        file_fd = -1
        return stream
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


def chmod_contained_regular_file(
    run_dir: str | Path,
    candidate: str | Path,
    mode: int,
) -> None:
    """Change mode on a contained regular file through one no-follow FD."""

    root, relative = _contained_artifact_location(run_dir, candidate)
    artifact_path = root / relative
    if os.name != "posix":
        del mode  # Windows access is governed by the containing directory ACL.
        try:
            with open_confined_directory(root) as root_descriptor:
                with open_confined_regular_file(
                    root_descriptor,
                    relative.as_posix(),
                ):
                    return
        except (ArtifactPathError, OSError) as exc:
            raise ValueError(
                "Contained artifact must be a regular file and not a reparse "
                f"point: {artifact_path}"
            ) from exc
    try:
        parent_fd = _open_contained_directory(root, relative.parent, create=False)
    except OSError as exc:
        raise ValueError(
            "Contained artifact path must not contain symlinks or "
            f"non-directories: {artifact_path}"
        ) from exc
    file_fd = -1
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            file_fd = os.open(relative.name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ValueError(
                "Contained artifact must be a regular file and not a symlink: "
                f"{artifact_path}"
            ) from exc
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ValueError(
                f"Contained artifact is not a regular file: {artifact_path}"
            )
        os.fchmod(file_fd, mode)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


def resolve_artifact_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    """Resolve an artifact path, interpreting relative paths from ``base_dir``."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve()


def file_sha256(path: str | Path) -> str:
    """Return a SHA-256 digest derived from one held file descriptor."""

    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Artifact is not a file: {path}") from exc
    file_flags = (
        os.O_RDONLY
        | _BINARY_OPEN_FLAG
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        file_fd = os.open(resolved, file_flags)
    except OSError as exc:
        raise FileNotFoundError(f"Artifact is not a file: {resolved}") from exc
    try:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise FileNotFoundError(f"Artifact is not a file: {resolved}")
        # This is a pure digest helper over caller-named files; link-count
        # policy stays with the callers (several verify st_nlink themselves,
        # and the run-root containment reader rejects multi-link artifacts).
        digest, _, _ = _read_regular_fd(
            file_fd,
            resolved,
            max_bytes=None,
            capture_bytes=False,
            allow_hardlinks=True,
        )
    finally:
        os.close(file_fd)
    return digest


def _update_file_digest(digest: Any, path: Path, logical_path: str) -> None:
    digest.update(b"file\0")
    digest.update(logical_path.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    digest.update(b"\0")


def artifact_set_digest(
    paths: Iterable[str | Path],
    *,
    base_dir: Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Digest a deterministic set of files/directories plus optional metadata."""

    resolved_paths = sorted(
        {resolve_artifact_path(path, base_dir=base_dir) for path in paths},
        key=str,
    )
    digest = hashlib.sha256()
    digest.update(_DIGEST_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            dict(metadata or {}),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    digest.update(b"\0")

    for root in resolved_paths:
        if not root.exists():
            raise FileNotFoundError(f"Artifact does not exist: {root}")
        digest.update(b"root\0")
        digest.update(str(root).encode("utf-8"))
        digest.update(b"\0")
        if root.is_file():
            _update_file_digest(digest, root, root.name)
            continue
        if not root.is_dir():
            raise ValueError(f"Unsupported artifact type: {root}")

        digest.update(b"directory\0")
        entries = sorted(root.rglob("*"), key=lambda path: path.as_posix())
        for entry in entries:
            relative = entry.relative_to(root).as_posix()
            if entry.is_dir():
                digest.update(b"dir\0")
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
            elif entry.is_file():
                _update_file_digest(digest, entry, relative)
            else:
                raise ValueError(f"Unsupported artifact type: {entry}")
    return digest.hexdigest()


def _open_contained_directory(
    root: Path,
    relative: Path,
    *,
    create: bool,
) -> int:
    """Open a run-local directory without following any symlink component."""

    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(root, directory_flags)
    try:
        for component in relative.parts:
            if component in {"", ".", ".."}:
                raise ValueError(
                    f"Unsafe contained output directory component: {component!r}"
                )
            try:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=directory_fd)
                except FileExistsError:
                    # A concurrent creator won the race. The no-follow open
                    # below still verifies that it created a real directory.
                    pass
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except Exception:
        os.close(directory_fd)
        raise


def _atomic_write_text_contained(
    root: Path,
    relative: Path,
    text: str,
) -> Path:
    """Atomically replace one run-local regular file through directory FDs."""

    if relative.name in {"", ".", ".."} or relative.parent == Path(".."):
        raise ValueError(f"Unsafe contained output path: {relative}")
    if os.name != "posix":
        try:
            with open_confined_directory(root) as root_descriptor:
                write_bytes_to_confined(
                    root_descriptor,
                    relative.as_posix(),
                    text.encode("utf-8"),
                    file_mode=0o600,
                )
        except ArtifactPathError as exc:
            raise ValueError(
                "Contained output target must be a regular file or absent: "
                f"{root / relative}"
            ) from exc
        return root / relative
    parent_fd = _open_contained_directory(root, relative.parent, create=True)
    temporary_name = f".{relative.name}.{secrets.token_hex(8)}.tmp"
    temporary_fd = -1
    try:
        try:
            existing = os.stat(
                relative.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError(
                "Contained output target must be a regular file or absent: "
                f"{root / relative}"
            )

        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(
            temporary_name,
            file_flags,
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as stream:
            temporary_fd = -1
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary_name,
            relative.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        try:
            os.fsync(parent_fd)
        except OSError:
            # Some filesystems do not support directory fsync.
            pass
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
    return root / relative


def _absolute_artifact_path(path: str | Path) -> Path:
    """Resolve existing ancestors without following the final entry."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.parent.resolve() / candidate.name


DirectoryIdentityChain = tuple[tuple[str, tuple[int, int]], ...]


def _open_directory_no_symlinks(
    path: Path,
    *,
    create_missing: bool = True,
) -> tuple[int, DirectoryIdentityChain]:
    """Open an absolute directory one component at a time without symlinks."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(path.anchor, flags)
    root_stat = os.fstat(current_fd)
    identities: list[tuple[str, tuple[int, int]]] = [
        (path.anchor, (root_stat.st_dev, root_stat.st_ino))
    ]
    try:
        for component in path.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create_missing:
                    raise
                try:
                    os.mkdir(component, dir_fd=current_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(component, flags, dir_fd=current_fd)
            component_stat = os.fstat(next_fd)
            identities.append(
                (
                    component,
                    (component_stat.st_dev, component_stat.st_ino),
                )
            )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd, tuple(identities)
    except BaseException:
        os.close(current_fd)
        raise


def prepare_writable_directory(path: str | Path) -> Path:
    """Create and validate an absolute writable directory without following links.

    The returned path is lexical: no caller-supplied component is hidden by a
    prior ``resolve()``.  On POSIX, every component is opened relative to a
    held parent descriptor with ``O_NOFOLLOW`` and missing components are
    created through those descriptors.  This is the workflow entry-point
    primitive for output roots that may not exist yet.
    """

    absolute = Path(os.path.abspath(Path(path).expanduser()))
    try:
        resolved = absolute.resolve(strict=False)
    except OSError as exc:
        raise ValueError(
            f"Writable directory cannot be inspected safely: {absolute}"
        ) from exc
    if resolved != absolute:
        raise ValueError(
            "Writable directory path must resolve without traversing symlinks: "
            f"{absolute}"
        )

    if os.name != "posix":  # pragma: win32 cover
        absolute.mkdir(parents=True, exist_ok=True)
        if not absolute.is_dir() or absolute.is_symlink():
            raise ValueError(
                f"Writable directory must be a non-symlink directory: {absolute}"
            )
        return absolute

    directory_fd = -1
    try:
        directory_fd, identity_chain = _open_directory_no_symlinks(absolute)
        metadata = os.fstat(directory_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"Writable directory is not a directory: {absolute}")
        if not _directory_chain_matches(absolute, identity_chain):
            raise ValueError(
                f"Writable directory changed while it was prepared: {absolute}"
            )
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(
            "Writable directory path must resolve without traversing symlinks "
            f"and contain only directories: {absolute}"
        ) from exc
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
    return absolute


def prepare_writable_file_path(path: str | Path) -> Path:
    """Validate one writable file target and safely create its parent tree.

    The target itself remains absent or an existing single-link regular file;
    this function never creates or truncates it.  Its parent is prepared with
    :func:`prepare_writable_directory`, so intermediate and dangling symlinks
    fail closed before a workflow hands the path to an external authoring
    backend.
    """

    absolute = Path(os.path.abspath(Path(path).expanduser()))
    try:
        resolved = absolute.resolve(strict=False)
    except OSError as exc:
        raise ValueError(
            f"Writable file cannot be inspected safely: {absolute}"
        ) from exc
    if resolved != absolute:
        raise ValueError(
            f"Writable file path must resolve without traversing symlinks: {absolute}"
        )
    if absolute.name in {"", ".", ".."}:
        raise ValueError(f"Writable file target is invalid: {absolute}")

    parent = prepare_writable_directory(absolute.parent)
    target = parent / absolute.name
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return target
    except OSError as exc:
        raise ValueError(f"Writable file cannot be inspected safely: {target}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(
            "Writable file target must be a single-link regular file or absent: "
            f"{target}"
        )
    return target


def _directory_chain_matches(path: Path, chain: DirectoryIdentityChain) -> bool:
    if len(path.parts) != len(chain):
        return False
    current = Path(path.anchor)
    for index, (expected_name, expected_identity) in enumerate(chain):
        component = path.parts[index]
        if component != expected_name:
            return False
        if index:
            current /= component
        try:
            current_stat = current.lstat()
        except OSError:
            return False
        if (
            not stat.S_ISDIR(current_stat.st_mode)
            or (
                current_stat.st_dev,
                current_stat.st_ino,
            )
            != expected_identity
        ):
            return False
    return True


def _create_temporary_file_at(
    parent_fd: int,
    *,
    destination_name: str,
) -> tuple[int, str]:
    """Create a same-directory temporary file through an already pinned fd."""

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(128):
        temporary_name = f".{destination_name}.{secrets.token_hex(8)}.tmp"
        try:
            return (
                os.open(
                    temporary_name,
                    flags,
                    0o600,
                    dir_fd=parent_fd,
                ),
                temporary_name,
            )
        except FileExistsError:
            continue
    raise FileExistsError(
        f"Could not allocate a unique temporary artifact for {destination_name}"
    )


def _atomic_write_text_posix(path: Path, text: str) -> None:
    parent_fd, parent_chain = _open_directory_no_symlinks(path.parent)
    try:
        _atomic_write_text_at(parent_fd, destination_name=path.name, text=text)
        if not _directory_chain_matches(path.parent, parent_chain):
            raise RuntimeError(
                f"Atomic artifact parent directory changed during write: {path.parent}"
            )
    finally:
        os.close(parent_fd)


def _atomic_write_text_at(
    parent_fd: int,
    *,
    destination_name: str,
    text: str,
) -> None:
    """Atomically write one file relative to an already pinned directory."""

    if (
        not destination_name
        or destination_name in {".", ".."}
        or Path(destination_name).name != destination_name
    ):
        raise ValueError(
            f"Atomic artifact destination must be one file name: {destination_name}"
        )
    temporary_fd = -1
    temporary_name: str | None = None
    try:
        temporary_fd, temporary_name = _create_temporary_file_at(
            parent_fd,
            destination_name=destination_name,
        )
        _write_all(temporary_fd, text.encode("utf-8"))
        os.fsync(temporary_fd)
        temporary_stat = os.fstat(temporary_fd)
        staged_stat = os.stat(
            temporary_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        if (
            not stat.S_ISREG(temporary_stat.st_mode)
            or temporary_stat.st_nlink != 1
            or not stat.S_ISREG(staged_stat.st_mode)
            or staged_stat.st_nlink != 1
            or (staged_stat.st_dev, staged_stat.st_ino) != temporary_identity
        ):
            raise RuntimeError("Atomic artifact temporary file identity changed")
        os.replace(
            temporary_name,
            destination_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        installed_stat = os.stat(
            destination_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(installed_stat.st_mode)
            or installed_stat.st_nlink != 1
            or (installed_stat.st_dev, installed_stat.st_ino) != temporary_identity
        ):
            raise RuntimeError("Atomic artifact destination identity changed")
        os.fsync(parent_fd)
    finally:
        if temporary_name is not None:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=parent_fd)
        if temporary_fd >= 0:
            os.close(temporary_fd)


def _atomic_write_bytes_contained(
    root: Path,
    relative: Path,
    data: bytes,
) -> Path:
    """Atomically replace one run-local regular file with exact bytes."""

    if relative.name in {"", ".", ".."} or relative.parent == Path(".."):
        raise ValueError(f"Unsafe contained output path: {relative}")
    if os.name != "posix":
        try:
            with open_confined_directory(root) as root_descriptor:
                write_bytes_to_confined(
                    root_descriptor,
                    relative.as_posix(),
                    data,
                    file_mode=0o600,
                )
        except ArtifactPathError as exc:
            raise ValueError(
                "Contained output target must be a regular file or absent: "
                f"{root / relative}"
            ) from exc
        return root / relative
    parent_fd = _open_contained_directory(root, relative.parent, create=True)
    temporary_name = f".{relative.name}.{secrets.token_hex(8)}.tmp"
    temporary_fd = -1
    try:
        try:
            existing = os.stat(
                relative.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError(
                "Contained output target must be a regular file or absent: "
                f"{root / relative}"
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(temporary_fd, "wb") as stream:
            temporary_fd = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary_name,
            relative.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        try:
            os.fsync(parent_fd)
        except OSError:
            pass
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
    return root / relative


def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    within: str | Path | None = None,
) -> Path:
    """Write bytes through a same-directory atomic, no-follow replacement."""

    value = Path(path).expanduser()
    if within is not None:
        root = Path(within).expanduser().resolve(strict=True)
        absolute = (
            Path(os.path.abspath(value))
            if value.is_absolute()
            else Path(os.path.abspath(root / value))
        )
        try:
            relative = absolute.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Contained output path is outside the workflow run: {absolute}"
            ) from exc
        return _atomic_write_bytes_contained(root, relative, data)

    absolute = Path(os.path.abspath(value))
    absolute.parent.mkdir(parents=True, exist_ok=True)
    parent = absolute.parent.resolve(strict=True)
    target = parent / absolute.name
    if target.is_symlink():
        raise ValueError(f"Atomic output target must not be a symlink: {target}")
    return _atomic_write_bytes_contained(parent, Path(target.name), data)


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    within: str | Path | None = None,
) -> Path:
    """Write text through a same-directory atomic, no-follow replacement.

    When ``within`` is provided, every path component beneath that trusted
    directory is opened with ``O_NOFOLLOW``. This is the required mode for
    wrapper-owned files in a child-writable workflow run.
    """

    value = Path(path).expanduser()
    if within is not None:
        root = Path(within).expanduser().resolve(strict=True)
        absolute = (
            Path(os.path.abspath(value))
            if value.is_absolute()
            else Path(os.path.abspath(root / value))
        )
        try:
            relative = absolute.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Contained output path is outside the workflow run: {absolute}"
            ) from exc
        return _atomic_write_text_contained(root, relative, text)

    destination = _absolute_artifact_path(value)
    # Make the ancestor chain durable before anything is written, so an
    # interrupted ancestor sync fails the write instead of leaving a file
    # beneath directories that may not survive a crash.
    _create_directory_tree_durably(destination.parent)
    if os.name == "posix":
        # The write itself stays descriptor-pinned: it resolves without
        # following symlinks and replaces relative to a verified parent fd,
        # which also fsyncs that parent through the fd.
        _atomic_write_text_posix(destination, text)
        _remember_durable_directory(destination.parent)
        return destination

    temporary_path: Path | None = None
    try:  # pragma: win32 cover
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        _fsync_directory(destination.parent)
        _remember_durable_directory(destination.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def atomic_write_json(
    path: str | Path,
    payload: BaseModel | Mapping[str, Any],
    *,
    within: str | Path | None = None,
) -> Path:
    """Write stable JSON through a same-directory atomic replacement."""

    if isinstance(payload, BaseModel):
        document = payload.model_dump(mode="json")
    else:
        document = dict(payload)
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    return atomic_write_text(path, text, within=within)


def atomic_write_json_at(
    parent_fd: int,
    destination_name: str,
    payload: BaseModel | Mapping[str, Any],
) -> None:
    """Write stable JSON relative to an already pinned POSIX directory fd."""

    if os.name != "posix":  # pragma: win32 cover
        raise RuntimeError("Descriptor-relative atomic writes require POSIX")
    if isinstance(payload, BaseModel):
        document = payload.model_dump(mode="json")
    else:
        document = dict(payload)
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    _atomic_write_text_at(
        parent_fd,
        destination_name=destination_name,
        text=text,
    )


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {resolved}")
    return payload


def phase_result_digest(result: BaseModel, *, result_path: str | Path) -> str:
    """Digest a phase result's claims and every artifact it references."""

    artifact_paths = getattr(result, "artifact_paths", None)
    if not isinstance(artifact_paths, list):
        raise TypeError("Phase result must expose an artifact_paths list")
    metadata = result.model_dump(mode="json", exclude={"output_digest"})
    return artifact_set_digest(
        artifact_paths,
        base_dir=Path(result_path).expanduser().resolve().parent,
        metadata=metadata,
    )


def seal_phase_result[ModelT: BaseModel](result: ModelT, path: str | Path) -> ModelT:
    """Compute a phase output digest and atomically write the sealed result."""

    output_digest = phase_result_digest(result, result_path=path)
    sealed = result.model_copy(update={"output_digest": output_digest})
    atomic_write_json(path, sealed)
    return sealed
