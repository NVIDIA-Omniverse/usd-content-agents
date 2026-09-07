# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-safe local artifact reads and create-only materialization."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path

from .errors import ArtifactIntegrityError, ArtifactLimitError, UnsafeArtifactError
from .models import (
    MAX_OUTPUT_ARTIFACT_BYTES,
    SAFE_FILENAME_PATTERN,
    ArtifactRole,
    MaterializedArtifact,
    MaterializedWireSourceBundle,
    WireArtifact,
    WireSourceBundle,
)

_CHUNK_SIZE = 1024 * 1024


def validate_artifact_filename(filename: str) -> None:
    if (
        Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or re.fullmatch(SAFE_FILENAME_PATTERN, filename) is None
    ):
        raise UnsafeArtifactError("artifact filename is unsafe")


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _open_directory(path: Path, *, create: bool, label: str) -> int:
    """Open an absolute directory through no-follow descriptors."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current_fd = -1
    try:
        current_fd = os.open("/", flags)
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                raise UnsafeArtifactError(f"{label} contains an unsafe component")
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=current_fd)
                    os.fsync(current_fd)
                except FileExistsError:
                    pass
            child_fd = os.open(component, flags, dir_fd=current_fd)
            metadata = os.fstat(child_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child_fd)
                raise UnsafeArtifactError(f"{label} contains a non-directory component")
            os.close(current_fd)
            current_fd = child_fd
    except UnsafeArtifactError:
        if current_fd >= 0:
            os.close(current_fd)
        raise
    except OSError as exc:
        if current_fd >= 0:
            os.close(current_fd)
        raise UnsafeArtifactError(
            f"{label} contains a missing, non-directory, or symlink component"
        ) from exc
    return current_fd


def prepare_output_directory(path: str | Path) -> Path:
    absolute = _absolute(path)
    directory_fd = -1
    try:
        directory_fd = _open_directory(
            absolute,
            create=True,
            label="output directory",
        )
        os.fsync(directory_fd)
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
    return absolute


def read_regular_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int = MAX_OUTPUT_ARTIFACT_BYTES,
) -> bytes:
    """Read bytes through one no-follow descriptor and verify file identity."""

    absolute = _absolute(path)
    parent_fd = _open_directory(absolute.parent, create=False, label=label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    file_fd = -1
    try:
        file_fd = os.open(absolute.name, flags, dir_fd=parent_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise UnsafeArtifactError(f"{label} must be a single-link regular file")
        if before.st_size > max_bytes:
            raise ArtifactLimitError(f"{label} exceeds the {max_bytes}-byte limit")
        chunks: list[bytes] = []
        size_bytes = 0
        while True:
            chunk = os.read(file_fd, min(_CHUNK_SIZE, max_bytes - size_bytes + 1))
            if not chunk:
                break
            chunks.append(chunk)
            size_bytes += len(chunk)
            if size_bytes > max_bytes:
                raise ArtifactLimitError(f"{label} exceeds the {max_bytes}-byte limit")
        after = os.fstat(file_fd)
    except OSError as exc:
        raise UnsafeArtifactError(
            f"{label} cannot be read safely (symlink or invalid file)"
        ) from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if identity_before != identity_after or size_bytes != after.st_size:
        raise ArtifactIntegrityError(f"{label} changed while it was being read")
    return b"".join(chunks)


def read_worker_output(
    workspace: str | Path,
    candidate: str | Path,
    *,
    max_bytes: int = MAX_OUTPUT_ARTIFACT_BYTES,
) -> bytes:
    root = _absolute(workspace)
    root_fd = _open_directory(root, create=False, label="worker workspace")
    os.close(root_fd)
    path = Path(candidate)
    absolute = path if path.is_absolute() else root / path
    absolute = Path(os.path.abspath(absolute))
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise UnsafeArtifactError("worker output escapes its isolated workspace") from exc
    return read_regular_file(absolute, label="worker output", max_bytes=max_bytes)


def _write_create_only(root: Path, filename: str, content: bytes) -> Path:
    validate_artifact_filename(filename)
    root_fd = _open_directory(root, create=False, label="output directory")
    file_fd = -1
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        file_fd = os.open(filename, flags, 0o600, dir_fd=root_fd)
        view = memoryview(content)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:
                raise OSError("artifact write made no forward progress")
            view = view[written:]
        os.fsync(file_fd)
    except FileExistsError as exc:
        raise UnsafeArtifactError(f"artifact destination already exists: {filename}") from exc
    except OSError as exc:
        raise UnsafeArtifactError(f"artifact cannot be materialized: {filename}") from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(root_fd)
    return root / filename


def _unlink_created(root: Path, filename: str) -> None:
    root_fd = _open_directory(root, create=False, label="output directory")
    try:
        os.unlink(filename, dir_fd=root_fd)
    finally:
        os.close(root_fd)


def materialize_wire_bundle(
    bundle: WireSourceBundle,
    output_dir: str | Path,
) -> MaterializedWireSourceBundle:
    root = prepare_output_directory(output_dir)
    created: list[Path] = []
    artifacts: list[MaterializedArtifact] = []
    try:
        for wire in bundle.artifacts:
            content = wire.content()
            path = _write_create_only(root, wire.filename, content)
            created.append(path)
            observed = read_regular_file(path, label="materialized artifact")
            digest = hashlib.sha256(observed).hexdigest()
            if digest != wire.sha256 or len(observed) != wire.size_bytes:
                raise ArtifactIntegrityError(
                    "materialized artifact differs from its provider binding",
                    provider_id=bundle.provider_id,
                )
            artifacts.append(
                MaterializedArtifact(
                    filename=wire.filename,
                    role=wire.role,
                    media_type=wire.media_type,
                    sha256=wire.sha256,
                    size_bytes=wire.size_bytes,
                    path=path,
                )
            )
    except Exception:
        for path in reversed(created):
            try:
                _unlink_created(root, path.name)
            except (OSError, UnsafeArtifactError):
                pass
        raise
    return MaterializedWireSourceBundle(
        provider_id=bundle.provider_id,
        provider_version=bundle.provider_version,
        source_revision=bundle.source_revision,
        units=bundle.units,
        up_axis=bundle.up_axis,
        forward_axis=bundle.forward_axis,
        handedness=bundle.handedness,
        upstream_edit_uri=bundle.upstream_edit_uri,
        artifacts=tuple(artifacts),
        parts=bundle.parts,
        parameters=bundle.parameters,
        verification_assertions=bundle.verification_assertions,
        metadata=bundle.metadata,
    )


def wire_artifact_from_file(
    path: str | Path,
    *,
    filename: str,
    role: ArtifactRole,
    media_type: str,
) -> WireArtifact:
    content = read_regular_file(path, label=f"artifact {filename}")
    return WireArtifact.from_bytes(
        filename=filename,
        role=role,
        media_type=media_type,
        content=content,
    )
