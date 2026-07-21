# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integrity-bound local USD sources and private authoring projections.

Small root files can use kernel-sealed memory snapshots, while large roots stay
pinned to read-only source descriptors. Dependencies use unnamed disk-backed
snapshots so authoring retains exact bytes without making memory scale with the
source closure. ``WU_JOINT_RIGGER_SNAPSHOT_DIR`` can select an operator-managed
spill directory; otherwise known memory-backed filesystems are skipped. The
selected storage mode is explicit on every binding, and no mode imposes a
product byte limit.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import secrets
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Literal
from urllib.parse import urlparse

from world_understanding.functions.physics.joint_rigger.artifacts import (
    _open_bound_directory,
    _quarantine_descriptor_entry,
    _quarantine_recovery_note,
    _remove_descriptor_entry,
    _require_bound_directory_unchanged,
)
from world_understanding.functions.physics.joint_rigger.facade import (
    _MAX_OPAQUE_DEPENDENCY_FILES,
    _MAX_OPAQUE_DEPENDENCY_REFERENCES,
    _MAX_OPAQUE_DOCUMENT_BYTES,
    JointRiggerArtifactError,
    JointRiggerBackendIncompatibleError,
)
from world_understanding.functions.physics.joint_rigger.models import (
    ArtifactIdentityV1,
)

_MFD_CLOEXEC = 1
_MFD_ALLOW_SEALING = 2
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_SOURCE_MEMFD_SEALS = 1 | 2 | 4 | 8
_MAX_BOUND_DEPENDENCY_FILES = _MAX_OPAQUE_DEPENDENCY_FILES
_MAX_BOUND_DEPENDENCY_REFERENCES = _MAX_OPAQUE_DEPENDENCY_REFERENCES
_DISK_SNAPSHOT_DIRECTORY_ENV = "WU_JOINT_RIGGER_SNAPSHOT_DIR"
_MEMORY_BACKED_FILESYSTEM_TYPES = frozenset({"hugetlbfs", "ramfs", "tmpfs"})
# This is a storage-selection threshold, not an input limit. Small files are
# copied into immutable memfds. Larger files stay pinned by a read-only source
# descriptor and are rehashed while materializing the private authoring tree.
_MAX_MEMFD_SNAPSHOT_BYTES = _MAX_OPAQUE_DOCUMENT_BYTES
_LIBC = ctypes.CDLL(None, use_errno=True)
_MEMFD_CREATE: Any
try:
    _MEMFD_CREATE = _LIBC.memfd_create
except AttributeError:  # pragma: no cover - Linux runtime contract
    _MEMFD_CREATE = None
else:
    _MEMFD_CREATE.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    _MEMFD_CREATE.restype = ctypes.c_int


@dataclass(frozen=True)
class SealedDependencyBinding:
    """One exact local file binding with an explicit integrity mechanism.

    ``sealed_memfd`` is a kernel-immutable byte snapshot. ``anonymous_snapshot``
    is an unnamed disk-backed byte snapshot held through a read-only descriptor.
    ``pinned_file`` keeps the opened source inode read-only and detects any
    subsequent byte or descriptor-state change before materialization.
    """

    path: Path
    descriptor: int
    sha256: str
    projection_paths: tuple[Path, ...] = ()
    layer_projection_paths: frozenset[Path] = frozenset()
    storage_kind: Literal["sealed_memfd", "anonymous_snapshot", "pinned_file"] = (
        "sealed_memfd"
    )
    descriptor_state: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SealedSourceBinding:
    """Exact root bytes plus a streaming-verified local dependency closure."""

    path: Path
    descriptor: int
    sha256: str
    dependencies: tuple[SealedDependencyBinding, ...] = ()
    storage_kind: Literal["sealed_memfd", "anonymous_snapshot", "pinned_file"] = (
        "sealed_memfd"
    )
    descriptor_state: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FrozenProjectionRoot:
    """One post-validation projected root retained through descriptor copying."""

    path: Path
    descriptor: int
    identity: tuple[int, int]
    sha256: str
    state: tuple[int, ...]
    _parent: Any = field(repr=False, compare=False)


@dataclass
class BoundInputDirectory:
    """One caller-owned private projection retained from its creation."""

    path: Path
    descriptor: int
    identity: tuple[int, int]
    _parent: Any = field(repr=False, compare=False)
    closed: bool = field(default=False, repr=False, compare=False)


def create_sealed_source_binding(
    path: Path,
    *,
    expected: ArtifactIdentityV1,
) -> SealedSourceBinding:
    """Bind root and dependencies exactly and verify their request identity.

    Dependencies intentionally use unnamed disk-backed snapshots so memory use
    does not grow with package size and live namespace mutations cannot change
    the bytes copied into the private authoring tree.
    """

    from world_understanding.functions.physics.joint_rigger.reference import (
        _artifact_identity_from_captured_records,
        _capture_dependency_structure,
        _CapturedDependencyIdentityRecord,
    )

    root = _create_sealed_file_binding(
        path,
        expected_sha256=expected.root_sha256,
    )
    dependencies: list[SealedDependencyBinding] = []
    try:
        if path.suffix.lower() != ".usdz":
            first_structure = _capture_dependency_structure(
                path,
                logical_artifact_path=path,
            )
            if len(first_structure) > _MAX_BOUND_DEPENDENCY_REFERENCES:
                raise JointRiggerBackendIncompatibleError(
                    "Bound Joint Rigger source closure exceeds the fixed "
                    f"{_MAX_BOUND_DEPENDENCY_REFERENCES}-reference limit"
                )
            package_records = [
                record
                for record in first_structure
                if record.package_inner_locator is not None
            ]
            if package_records:
                raise JointRiggerBackendIncompatibleError(
                    "Bound Joint Rigger raw-source authoring does not support "
                    "package-relative dependencies"
                )
            backing_paths = sorted(
                {
                    record.backing_path
                    for record in first_structure
                    if record.backing_path is not None
                }
            )
            if len(backing_paths) > _MAX_BOUND_DEPENDENCY_FILES:
                raise JointRiggerBackendIncompatibleError(
                    "Bound Joint Rigger source closure exceeds the fixed "
                    f"{_MAX_BOUND_DEPENDENCY_FILES}-dependency-file limit"
                )
            for dependency_path in backing_paths:
                dependency = _create_sealed_file_binding(
                    dependency_path,
                    prefer_disk_snapshot=True,
                )
                dependencies.append(dependency)
            second_structure = _capture_dependency_structure(
                path,
                logical_artifact_path=path,
            )
            if second_structure != first_structure:
                raise JointRiggerArtifactError(
                    "Input USD dependency structure changed while it was bound"
                )
            projection_paths_by_backing: dict[Path, set[Path]] = {
                dependency.path: set() for dependency in dependencies
            }
            layer_projection_paths_by_backing: dict[Path, set[Path]] = {
                dependency.path: set() for dependency in dependencies
            }
            for record in first_structure:
                if record.backing_path is None:
                    continue
                locator_path = Path(record.locator)
                if locator_path.is_absolute():
                    raise JointRiggerBackendIncompatibleError(
                        "Bound Joint Rigger raw-source authoring does not support "
                        "absolute local dependencies"
                    )
                # USD asset locators are content, not shell input.  Preserve a
                # leading ``~`` literally so a valid relative dependency cannot
                # be redirected through the process user's home directory.
                logical_alias = Path(os.path.abspath(path.parent / locator_path))
                projection_paths_by_backing[record.backing_path].add(logical_alias)
                if record.kind == "used_layer":
                    layer_projection_paths_by_backing[record.backing_path].add(
                        logical_alias
                    )
            dependencies = [
                SealedDependencyBinding(
                    path=dependency.path,
                    descriptor=dependency.descriptor,
                    sha256=dependency.sha256,
                    projection_paths=tuple(
                        sorted(
                            projection_paths_by_backing[dependency.path],
                            key=lambda projection_path: projection_path.as_posix(),
                        )
                    ),
                    layer_projection_paths=frozenset(
                        layer_projection_paths_by_backing[dependency.path]
                    ),
                    storage_kind=dependency.storage_kind,
                    descriptor_state=dependency.descriptor_state,
                )
                for dependency in dependencies
            ]
            bindings_by_path = {
                dependency.path: dependency for dependency in dependencies
            }
            records: list[_CapturedDependencyIdentityRecord] = []
            for record in first_structure:
                sha256 = (
                    root.sha256
                    if record.backing_path is None
                    else bindings_by_path[record.backing_path].sha256
                )
                records.append(
                    _CapturedDependencyIdentityRecord(
                        kind=record.kind,
                        locator=record.locator,
                        sha256=sha256,
                        backing_path=record.backing_path,
                    )
                )
            captured_identity = _artifact_identity_from_captured_records(
                logical_artifact_path=path,
                uri=expected.uri,
                root_sha256=root.sha256,
                records=records,
            )
            if captured_identity != expected:
                raise JointRiggerArtifactError(
                    "Bound input dependency closure does not match the request"
                )
        binding = SealedSourceBinding(
            path=root.path,
            descriptor=root.descriptor,
            sha256=root.sha256,
            dependencies=tuple(dependencies),
            storage_kind=root.storage_kind,
            descriptor_state=root.descriptor_state,
        )
        require_sealed_source_binding(binding)
        return binding
    except BaseException as binding_error:
        close_errors = _close_descriptors(
            [root.descriptor, *(item.descriptor for item in dependencies)]
        )
        _add_cleanup_error_note(
            binding_error,
            label="Bound input cleanup also failed",
            errors=close_errors,
        )
        raise


def bound_input_dependency_snapshots(
    binding: SealedSourceBinding,
) -> tuple[tuple[str, int, str, str, bool], ...]:
    """Return every lexical projection alias backed by one sealed descriptor."""

    return tuple(
        (
            str(projection_path),
            dependency.descriptor,
            dependency.sha256,
            str(dependency.path),
            projection_path in dependency.layer_projection_paths,
        )
        for dependency in binding.dependencies
        for projection_path in dependency.projection_paths or (dependency.path,)
    )


def require_sealed_source_binding(binding: SealedSourceBinding | None) -> None:
    """Revalidate the immutable root snapshot before a trust boundary."""

    if binding is None:
        raise JointRiggerArtifactError("Bound input snapshot is missing")
    _require_sealed_file_binding(
        SealedDependencyBinding(
            path=binding.path,
            descriptor=binding.descriptor,
            sha256=binding.sha256,
            storage_kind=binding.storage_kind,
            descriptor_state=binding.descriptor_state,
        )
    )
    for dependency in binding.dependencies:
        _require_sealed_file_binding(dependency)


def close_source_binding(binding: SealedSourceBinding) -> list[Exception]:
    """Close every anonymous closure descriptor exactly once."""

    return _close_descriptors(
        [
            binding.descriptor,
            *(item.descriptor for item in binding.dependencies),
        ]
    )


def _create_bound_input_directory(
    *,
    parent_path: Path,
    prefix: str,
) -> BoundInputDirectory:
    """Create and retain one cryptographically named private directory."""

    parent = _open_bound_directory(parent_path)
    descriptor = -1
    created_name: str | None = None
    created_identity: tuple[int, int] | None = None
    active_error: BaseException | None = None
    try:
        _require_bound_directory_unchanged(parent)
        for _ in range(128):
            name = f"{prefix}{secrets.token_hex(12)}"
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent.descriptor)
            except FileExistsError:  # pragma: no cover - cryptographic collision
                continue
            created_name = name
            metadata = os.stat(
                name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            created_identity = (metadata.st_dev, metadata.st_ino)
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(name, flags, dir_fd=parent.descriptor)
            opened = os.fstat(descriptor)
            identity = (opened.st_dev, opened.st_ino)
            if not stat.S_ISDIR(opened.st_mode) or identity != created_identity:
                raise JointRiggerArtifactError(
                    "Created bound input directory changed while retained: "
                    f"{parent.opened_path / name}"
                )
            _require_bound_directory_unchanged(parent)
            named_after_open = os.stat(
                name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(named_after_open.st_mode)
                or (named_after_open.st_dev, named_after_open.st_ino) != identity
            ):
                raise JointRiggerArtifactError(
                    "Created bound input directory name changed after retention: "
                    f"{parent.opened_path / name}"
                )
            return BoundInputDirectory(
                path=parent.opened_path / name,
                descriptor=descriptor,
                identity=identity,
                _parent=parent,
            )
        raise JointRiggerArtifactError(
            "Could not allocate a private bound input directory"
        )
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        if active_error is not None:
            cleanup_errors: list[BaseException] = []
            if created_name is not None and created_identity is not None:
                try:
                    if descriptor >= 0:
                        try:
                            os.fstat(descriptor)
                        except OSError:
                            _remove_empty_created_bound_input_directory(
                                parent_descriptor=parent.descriptor,
                                parent_path=parent.opened_path,
                                name=created_name,
                                expected_identity=created_identity,
                            )
                        else:
                            _remove_descriptor_entry(
                                parent.descriptor,
                                created_name,
                                expected_identity=created_identity,
                                source_descriptor=descriptor,
                                label=(
                                    "failed bound input directory allocation "
                                    f"{parent.opened_path / created_name}"
                                ),
                            )
                    else:
                        _remove_empty_created_bound_input_directory(
                            parent_descriptor=parent.descriptor,
                            parent_path=parent.opened_path,
                            name=created_name,
                            expected_identity=created_identity,
                        )
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            elif created_name is not None:
                cleanup_errors.append(
                    JointRiggerArtifactError(
                        "Failed bound input directory identity was never retained; "
                        f"preserving its name: {parent.opened_path / created_name}"
                    )
                )
            for owned_descriptor in (descriptor, parent.descriptor):
                if owned_descriptor < 0:
                    continue
                try:
                    os.close(owned_descriptor)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                active_error.add_note(
                    "Bound input directory creation cleanup also failed: "
                    + "; ".join(str(error) for error in cleanup_errors)
                )


def _remove_empty_created_bound_input_directory(
    *,
    parent_descriptor: int,
    parent_path: Path,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    """Remove one just-created empty directory without allocating a child fd."""

    label = f"failed bound input directory allocation {parent_path / name}"
    try:
        metadata = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if (metadata.st_dev, metadata.st_ino) != expected_identity:
        raise JointRiggerArtifactError(
            f"{label} changed inode; refusing deletion; replacement preserved"
        )
    quarantine = _quarantine_descriptor_entry(
        parent_descriptor,
        name,
        expected_identity=expected_identity,
        label=label,
    )
    try:
        quarantined = os.stat(
            quarantine.quarantine_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(quarantined.st_mode)
            or (quarantined.st_dev, quarantined.st_ino) != expected_identity
        ):
            raise JointRiggerArtifactError(
                f"{label} quarantine changed inode; refusing deletion"
            )
        os.rmdir(quarantine.quarantine_name, dir_fd=parent_descriptor)
    except BaseException as cleanup_error:
        try:
            os.stat(
                quarantine.quarantine_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        restoration = _quarantine_recovery_note(
            parent_descriptor,
            quarantine.quarantine_name,
            name,
            preserved_identity=expected_identity,
            label=label,
        )
        cleanup_error.add_note(
            "Failed bound input directory quarantine recovery: " + restoration
        )
        raise


def materialize_bound_input(
    *,
    descriptor: int,
    expected_sha256: str,
    logical_input_path: Path,
    dependencies: tuple[tuple[str, int, str, str, bool], ...] = (),
    editable_root: bool = False,
) -> tuple[Path, BoundInputDirectory, dict[Path, Path]]:
    """Materialize a sealed source closure under one private mirrored tree.

    Mode sealing and cryptographic names close ordinary namespace races. They
    are not a sandbox against hostile same-UID or in-process code that can chmod
    private directories or manipulate OpenUSD's process-global layer cache.
    """

    if (
        isinstance(descriptor, bool)
        or not isinstance(descriptor, int)
        or descriptor < 0
    ):
        raise ValueError("bound input descriptor must be a non-negative integer")
    directory = _create_bound_input_directory(
        parent_path=Path(tempfile.gettempdir()),
        prefix="joint-rigger-bound-input-",
    )
    projection_root = directory.path / "filesystem"

    def projected_path(logical_path: Path) -> Path:
        absolute = Path(os.path.abspath(logical_path.expanduser()))
        return projection_root / absolute.relative_to(Path(absolute.anchor))

    try:
        path = projected_path(logical_input_path)
        root_logical_path = Path(os.path.abspath(logical_input_path.expanduser()))
        projection_entries: dict[Path, tuple[int, str, Path, Path, bool]] = {
            path: (
                descriptor,
                expected_sha256,
                root_logical_path,
                root_logical_path,
                True,
            )
        }
        for (
            dependency_path_value,
            dependency_descriptor,
            dependency_sha256,
            restore_path_value,
            is_layer,
        ) in dependencies:
            dependency_path = Path(
                os.path.abspath(Path(dependency_path_value).expanduser())
            )
            restore_path = Path(os.path.abspath(Path(restore_path_value).expanduser()))
            if dependency_path == root_logical_path:
                raise ValueError(
                    f"Bound dependency aliases the source root: {dependency_path}"
                )
            entry = (
                dependency_descriptor,
                dependency_sha256,
                dependency_path,
                restore_path,
                is_layer,
            )
            for target_path in {
                projected_path(dependency_path),
                projected_path(restore_path),
            }:
                existing = projection_entries.get(target_path)
                if existing is not None:
                    if existing[:2] != entry[:2] or existing[3] != entry[3]:
                        raise ValueError(
                            "Conflicting bound dependencies share a projection path: "
                            f"{dependency_path}"
                        )
                    projection_entries[target_path] = (
                        *existing[:4],
                        existing[4] or is_layer,
                    )
                    continue
                projection_entries[target_path] = entry

        ordered_targets = sorted(
            projection_entries,
            key=lambda target: (len(target.parts), target.as_posix()),
        )
        for index, target_path in enumerate(ordered_targets):
            for nested_path in ordered_targets[index + 1 :]:
                if target_path in nested_path.parents:
                    raise ValueError(
                        "Bound dependency projection has a file/ancestor collision: "
                        f"{target_path} and {nested_path}"
                    )

        for target_path in ordered_targets:
            source_descriptor, source_sha256, _, _, _ = projection_entries[target_path]
            _materialize_file(
                source_descriptor=source_descriptor,
                source_sha256=source_sha256,
                target_path=target_path,
            )

        restore_paths = {
            target_path: entry[3] for target_path, entry in projection_entries.items()
        }
        _validate_bound_projection_dependencies(
            path,
            projection_root=projection_root,
            materialized_paths=frozenset(projection_entries),
            layer_paths=frozenset(
                target_path
                for target_path, entry in projection_entries.items()
                if entry[4]
            ),
            restore_paths=restore_paths,
        )
        for child in projection_root.rglob("*"):
            child.chmod(0o500 if child.is_dir() else 0o400)
        projection_root.chmod(0o500)
        directory.path.chmod(0o500)
        if editable_root:
            path.parent.chmod(0o700)
            path.chmod(0o600)
        return path, directory, restore_paths
    except BaseException as materialization_error:
        try:
            remove_bound_input_directory(directory)
        except Exception as cleanup_error:
            materialization_error.add_note(
                "Bound input cleanup also failed: " + str(cleanup_error)
            )
        raise


def restore_bound_projection_paths(
    output_path: Path,
    *,
    projection_root: Path,
    logical_output_parent: Path,
    restore_paths: Mapping[Path, Path],
    output_descriptor: int | None = None,
) -> None:
    """Replace private locators, optionally preserving one retained root inode."""

    from pxr import Ar, Sdf, UsdUtils

    layer = Sdf.Layer.FindOrOpen(str(output_path))
    if not layer:
        raise RuntimeError(f"Could not open bound authored output: {output_path}")
    normalized_projection = projection_root.resolve(strict=True)
    informational_identifiers = _take_informational_asset_identifiers(layer)

    def remap_path(asset_path: str) -> str:
        if not asset_path or "://" in asset_path:
            return asset_path
        if Ar.IsPackageRelativePath(asset_path):
            raise RuntimeError(
                "Bound raw output does not support package-relative dependencies"
            )
        resolved = layer.ComputeAbsolutePath(asset_path)
        if not resolved:  # pragma: no cover - nonempty local resolver invariant
            return asset_path
        try:
            resolved_path = Path(resolved).resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                "Bound output local dependency could not be resolved in the private "
                f"projection: {asset_path}"
            ) from exc
        try:
            projected_relative = resolved_path.relative_to(normalized_projection)
        except ValueError as exc:
            raise RuntimeError(
                "Bound output local dependency resolves outside the private "
                f"projection: {asset_path}"
            ) from exc
        projected_path = normalized_projection / projected_relative
        original = restore_paths.get(projected_path)
        if original is None:
            raise RuntimeError(
                "Bound output contains an unmapped private projection path: "
                f"{projected_path}"
            )
        logical_target = Path(os.path.abspath(logical_output_parent / asset_path))
        normalized_original = Path(os.path.abspath(original))
        if logical_target == normalized_original:
            return asset_path
        return os.path.relpath(original, logical_output_parent).replace("\\", "/")

    try:
        UsdUtils.ModifyAssetPaths(
            layer,
            remap_path,
            keepEmptyPathsInArrays=True,
        )
    finally:
        _restore_informational_asset_identifiers(layer, informational_identifiers)
    if output_descriptor is None:
        if not layer.Save():
            raise RuntimeError(f"Could not save rebound authored output: {output_path}")
        return

    rebound_directory = _create_bound_input_directory(
        parent_path=output_path.parent,
        prefix=".joint-rigger-rebound-",
    )
    rebound_path = rebound_directory.path / output_path.name
    rebound_descriptor = -1
    active_error: BaseException | None = None
    try:
        file_format = _concrete_usd_export_format(
            layer,
            output_descriptor=output_descriptor,
            output_path=output_path,
        )
        arguments = {"format": file_format}
        if not layer.Export(str(rebound_path), args=arguments):
            raise RuntimeError(
                f"Could not export rebound authored output: {output_path}"
            )
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        rebound_descriptor = os.open(rebound_path, flags)
        rebound_state = os.fstat(rebound_descriptor)
        if not stat.S_ISREG(rebound_state.st_mode) or rebound_state.st_nlink != 1:
            raise JointRiggerArtifactError(
                f"Rebound authored output is not a private regular file: {output_path}"
            )
        os.ftruncate(output_descriptor, 0)
        os.lseek(output_descriptor, 0, os.SEEK_SET)
        _copy_descriptor_bytes(
            rebound_descriptor,
            output_descriptor,
            expected_source=rebound_state,
            label="rebound authored output",
        )
        os.fsync(output_descriptor)
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if rebound_descriptor >= 0:
            try:
                os.close(rebound_descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        try:
            remove_bound_input_directory(rebound_directory)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            detail = "; ".join(str(error) for error in cleanup_errors)
            if active_error is not None:
                active_error.add_note(f"Rebound output cleanup also failed: {detail}")
            else:
                raise cleanup_errors[0]


def _concrete_usd_export_format(
    layer: Any,
    *,
    output_descriptor: int,
    output_path: Path,
) -> str:
    """Resolve generic ``.usd`` to its concrete USDA or USDC encoding."""

    file_format = str(layer.GetFileFormat().formatId)
    if file_format in {"usda", "usdc"}:
        return file_format
    if file_format != "usd":
        raise JointRiggerArtifactError(
            f"Unsupported bound authored output format {file_format!r}: {output_path}"
        )
    before = os.fstat(output_descriptor)
    header = os.pread(output_descriptor, 16, 0)
    after = os.fstat(output_descriptor)
    if _descriptor_state(before) != _descriptor_state(after):
        raise JointRiggerArtifactError(
            f"Generic bound authored output changed while its format was read: "
            f"{output_path}"
        )
    if header.startswith(b"#usda"):
        return "usda"
    if header.startswith(b"PXR-USDC"):
        return "usdc"
    raise JointRiggerArtifactError(
        "Could not determine concrete USDA/USDC encoding for generic bound "
        f"authored output: {output_path}"
    )


@contextmanager
def _descriptor_projection_validation_path(
    *,
    path: Path,
    parent: Any,
    descriptor: int,
) -> Iterator[Path]:
    """Expose one exact descriptor at a same-parent USD-suffixed alias."""

    alias_name: str | None = None
    alias_descriptor = -1
    active_error: BaseException | None = None
    try:
        for _ in range(128):
            candidate = f".joint-rigger-validation-{secrets.token_hex(12)}{path.suffix}"
            try:
                os.symlink(
                    f"/proc/self/fd/{descriptor}",
                    candidate,
                    dir_fd=parent.descriptor,
                )
            except FileExistsError:  # pragma: no cover - cryptographic collision
                continue
            alias_name = candidate
            break
        if alias_name is None:
            raise JointRiggerArtifactError(
                "Could not allocate a descriptor-pinned validation alias"
            )
        o_path = getattr(os, "O_PATH", None)
        if o_path is None:  # pragma: no cover - official targets are Linux
            raise JointRiggerBackendIncompatibleError(
                "Descriptor-pinned validation requires Linux O_PATH"
            )
        alias_descriptor = os.open(
            alias_name,
            o_path | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent.descriptor,
        )
        alias_state = os.fstat(alias_descriptor)
        observed = os.stat(
            alias_name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISLNK(alias_state.st_mode)
            or (alias_state.st_dev, alias_state.st_ino)
            != (observed.st_dev, observed.st_ino)
            or os.readlink(alias_name, dir_fd=parent.descriptor)
            != f"/proc/self/fd/{descriptor}"
        ):
            raise JointRiggerArtifactError(
                "Descriptor-pinned validation alias changed while retained"
            )
        os.fchmod(parent.descriptor, 0o500)
        yield Path(f"/proc/self/fd/{parent.descriptor}") / alias_name
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        try:
            os.fchmod(parent.descriptor, 0o700)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if alias_name is not None and alias_descriptor >= 0:
            try:
                alias_state = os.fstat(alias_descriptor)
                _remove_descriptor_entry(
                    parent.descriptor,
                    alias_name,
                    expected_identity=(alias_state.st_dev, alias_state.st_ino),
                    source_descriptor=alias_descriptor,
                    label=f"descriptor-pinned validation alias {path}",
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        elif alias_name is not None:
            try:
                alias_state = os.stat(
                    alias_name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
                _remove_descriptor_entry(
                    parent.descriptor,
                    alias_name,
                    expected_identity=(alias_state.st_dev, alias_state.st_ino),
                    label=f"descriptor-pinned validation alias {path}",
                )
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if alias_descriptor >= 0:
            try:
                os.close(alias_descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        try:
            os.fchmod(parent.descriptor, 0o500)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            detail = "; ".join(str(error) for error in cleanup_errors)
            if active_error is not None:
                active_error.add_note(
                    "Descriptor-pinned validation alias cleanup also failed: " + detail
                )
            else:
                raise cleanup_errors[0]


@contextmanager
def freeze_bound_projection_root(
    path: Path,
    *,
    validate_frozen_projection: Callable[[Path], None] | None = None,
    prepare_before_freeze: Callable[[int], None] | None = None,
) -> Iterator[FrozenProjectionRoot]:
    """Bind, validate, prepare, and freeze one root before descriptor copying."""

    normalized = Path(os.path.abspath(path.expanduser()))
    parent: Any | None = None
    descriptor = -1
    active_error: BaseException | None = None
    try:
        parent = _open_bound_directory(normalized.parent)
        _require_bound_directory_unchanged(parent)
        access_mode = os.O_RDWR if prepare_before_freeze is not None else os.O_RDONLY
        flags = access_mode | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(
            normalized.name,
            flags,
            dir_fd=parent.descriptor,
        )
        opened = os.fstat(descriptor)
        observed = os.stat(
            normalized.name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or identity != (observed.st_dev, observed.st_ino)
        ):
            raise JointRiggerArtifactError(
                f"Editable projected root changed before freeze: {normalized}"
            )
        if prepare_before_freeze is not None:
            prepare_before_freeze(descriptor)
        prepared = os.fstat(descriptor)
        named_after_prepare = os.stat(
            normalized.name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(prepared.st_mode)
            or prepared.st_nlink != 1
            or (prepared.st_dev, prepared.st_ino) != identity
            or (named_after_prepare.st_dev, named_after_prepare.st_ino) != identity
        ):
            raise JointRiggerArtifactError(
                f"Editable projected root changed while prepared: {normalized}"
            )
        os.fchmod(descriptor, 0o400)
        if prepare_before_freeze is not None:
            readonly_flags = os.O_RDONLY | os.O_NOFOLLOW
            readonly_flags |= getattr(os, "O_CLOEXEC", 0)
            readonly_flags |= getattr(os, "O_NONBLOCK", 0)
            readonly_descriptor = os.open(
                normalized.name,
                readonly_flags,
                dir_fd=parent.descriptor,
            )
            readonly_state = os.fstat(readonly_descriptor)
            if (readonly_state.st_dev, readonly_state.st_ino) != identity:
                os.close(readonly_descriptor)
                raise JointRiggerArtifactError(
                    f"Frozen projected root changed while reopened: {normalized}"
                )
            writable_descriptor = descriptor
            descriptor = readonly_descriptor
            os.close(writable_descriptor)
        if validate_frozen_projection is not None:
            validation_state = _descriptor_state(os.fstat(descriptor))
            validation_sha256 = _stable_descriptor_sha256(
                descriptor,
                label=f"pre-validation frozen projected root {normalized}",
            )
            with _descriptor_projection_validation_path(
                path=normalized,
                parent=parent,
                descriptor=descriptor,
            ) as validation_path:
                validate_frozen_projection(validation_path)
            validated = os.fstat(descriptor)
            named_after_validation = os.stat(
                normalized.name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            if (
                _descriptor_state(validated) != validation_state
                or _descriptor_state(named_after_validation) != validation_state
                or _stable_descriptor_sha256(
                    descriptor,
                    label=f"validated frozen projected root {normalized}",
                )
                != validation_sha256
            ):
                raise JointRiggerArtifactError(
                    f"Projected root changed during final validation: {normalized}"
                )
        os.fchmod(parent.descriptor, 0o500)
        frozen_state = _descriptor_state(os.fstat(descriptor))
        sha256 = _stable_descriptor_sha256(
            descriptor,
            label=f"frozen projected root {normalized}",
        )
        binding = FrozenProjectionRoot(
            path=normalized,
            descriptor=descriptor,
            identity=identity,
            sha256=sha256,
            state=frozen_state,
            _parent=parent,
        )
        _require_frozen_projection_root(binding)
        yield binding
        _require_frozen_projection_root(binding)
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        for owned_descriptor in (
            descriptor,
            -1 if parent is None else parent.descriptor,
        ):
            if owned_descriptor < 0:
                continue
            try:
                os.close(owned_descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            detail = "; ".join(str(error) for error in cleanup_errors)
            if active_error is not None:
                active_error.add_note(
                    f"Frozen projected-root descriptor cleanup also failed: {detail}"
                )
            else:
                raise cleanup_errors[0]


def _require_frozen_projection_root(binding: FrozenProjectionRoot) -> None:
    """Require the retained root, its frozen state, and its live name to agree."""

    _require_bound_directory_unchanged(binding._parent)
    metadata = os.fstat(binding.descriptor)
    observed_state = _descriptor_state(metadata)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or (metadata.st_dev, metadata.st_ino) != binding.identity
        or observed_state != binding.state
    ):
        raise JointRiggerArtifactError(
            f"Frozen projected root changed through its descriptor: {binding.path}"
        )
    try:
        named = os.stat(
            binding.path.name,
            dir_fd=binding._parent.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise JointRiggerArtifactError(
            f"Frozen projected root disappeared before copy: {binding.path}"
        ) from exc
    if _descriptor_state(named) != binding.state:
        raise JointRiggerArtifactError(
            f"Frozen projected root changed pathname before copy: {binding.path}"
        )
    if (
        _stable_descriptor_sha256(
            binding.descriptor,
            label=f"frozen projected root {binding.path}",
        )
        != binding.sha256
    ):
        raise JointRiggerArtifactError(
            f"Frozen projected root changed content before copy: {binding.path}"
        )


def remove_bound_input_directory(directory: BoundInputDirectory) -> None:
    """Remove only the exact private directory retained when it was created."""

    if not isinstance(directory, BoundInputDirectory):
        raise TypeError("directory must be a retained BoundInputDirectory")
    if directory.closed:
        return
    parent = directory._parent
    descriptor = directory.descriptor
    active_error: BaseException | None = None
    try:
        _require_bound_directory_unchanged(parent)
        retained = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(retained.st_mode)
            or (retained.st_dev, retained.st_ino) != directory.identity
        ):
            raise JointRiggerArtifactError(
                f"Retained bound input directory changed inode: {directory.path}"
            )
        try:
            metadata = os.stat(
                directory.path.name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if retained.st_nlink == 0:
                return
            raise JointRiggerArtifactError(
                "Retained bound input directory was moved; preserving its linked "
                f"inode: {directory.path}"
            ) from None
        if (metadata.st_dev, metadata.st_ino) != directory.identity:
            raise JointRiggerArtifactError(
                "Bound input cleanup name changed inode; replacement preserved: "
                f"{directory.path}"
            )
        _remove_descriptor_entry(
            parent.descriptor,
            directory.path.name,
            expected_identity=directory.identity,
            source_descriptor=descriptor,
            label=f"bound input projection {directory.path}",
        )
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        directory.closed = True
        directory.descriptor = -1
        cleanup_errors: list[BaseException] = []
        for owned_descriptor in (descriptor, parent.descriptor):
            if owned_descriptor < 0:
                continue
            try:
                os.close(owned_descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            detail = "; ".join(str(error) for error in cleanup_errors)
            if active_error is not None:
                active_error.add_note(
                    f"Bound input cleanup descriptor close also failed: {detail}"
                )
            else:
                raise cleanup_errors[0]


def copy_regular_file_to_new_path(
    source_path: Path,
    target_path: Path,
    *,
    label: str,
    frozen_source: FrozenProjectionRoot | None = None,
    bind_created_file: Callable[[Path, os.stat_result], None] | None = None,
) -> None:
    """Copy stable regular-file bytes into a new no-follow target."""

    source_descriptor = -1
    close_source_descriptor = False
    target_descriptor = -1
    target_parent: Any | None = None
    target_identity: tuple[int, int] | None = None
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        if frozen_source is None:
            source_flags = os.O_RDONLY | os.O_NOFOLLOW
            source_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
            source_descriptor = os.open(source_path, source_flags)
            close_source_descriptor = True
        else:
            normalized_source = Path(os.path.abspath(source_path.expanduser()))
            if normalized_source != frozen_source.path:
                raise ValueError(
                    "frozen_source must describe the requested source_path"
                )
            _require_frozen_projection_root(frozen_source)
            source_descriptor = frozen_source.descriptor
        source_before = os.fstat(source_descriptor)
        if frozen_source is None:
            observed_source = os.stat(source_path, follow_symlinks=False)
            if not stat.S_ISREG(source_before.st_mode) or (
                source_before.st_dev,
                source_before.st_ino,
            ) != (observed_source.st_dev, observed_source.st_ino):
                raise JointRiggerArtifactError(f"{label} source changed before copy")
        target_parent = _open_bound_directory(target_path.parent)
        _require_bound_directory_unchanged(target_parent)
        target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        target_flags |= getattr(os, "O_CLOEXEC", 0)
        target_descriptor = os.open(
            target_path.name,
            target_flags,
            0o600,
            dir_fd=target_parent.descriptor,
        )
        target_metadata = os.fstat(target_descriptor)
        target_identity = (target_metadata.st_dev, target_metadata.st_ino)
        _copy_descriptor_bytes(
            source_descriptor,
            target_descriptor,
            expected_source=source_before,
            label=label,
        )
        if frozen_source is not None:
            _require_frozen_projection_root(frozen_source)
        os.fsync(target_descriptor)
        _require_bound_directory_unchanged(target_parent)
        if bind_created_file is not None:
            bind_created_file(target_path, os.fstat(target_descriptor))
    except BaseException as exc:
        primary_error = exc
    finally:
        if (
            primary_error is not None
            and target_identity is not None
            and target_parent is not None
        ):
            try:
                _remove_descriptor_entry(
                    target_parent.descriptor,
                    target_path.name,
                    expected_identity=target_identity,
                    source_descriptor=target_descriptor,
                    label=f"failed {label} target",
                )
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        # Keep the created target inode and its parent open until atomic
        # identity quarantine completes. A same-directory swap can then only
        # be preserved or rejected; it can never be mistaken for our target.
        descriptors = [target_descriptor]
        if close_source_descriptor:
            descriptors.append(source_descriptor)
        for descriptor in descriptors:
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if target_parent is not None:
            try:
                os.close(target_parent.descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
    if primary_error is not None:
        if cleanup_errors:
            primary_error.add_note(
                "Descriptor-copy cleanup also failed: "
                + "; ".join(str(error) for error in cleanup_errors)
            )
        raise primary_error
    if cleanup_errors:
        raise cleanup_errors[0]


def write_new_text_file(
    path: Path,
    payload: str,
    *,
    label: str,
    bind_created_file: Callable[[Path, os.stat_result], None] | None = None,
) -> None:
    """Write one UTF-8 report through an exclusive no-follow descriptor."""

    descriptor = -1
    parent: Any | None = None
    identity: tuple[int, int] | None = None
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    encoded = payload.encode("utf-8")
    try:
        parent = _open_bound_directory(path.parent)
        _require_bound_directory_unchanged(parent)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(
            path.name,
            flags,
            0o600,
            dir_fd=parent.descriptor,
        )
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:  # pragma: no cover - regular-file OS invariant
                raise OSError(f"Short write while creating {label}")
            remaining = remaining[written:]
        os.fsync(descriptor)
        _require_bound_directory_unchanged(parent)
        if bind_created_file is not None:
            bind_created_file(path, os.fstat(descriptor))
    except BaseException as exc:
        primary_error = exc
    finally:
        if primary_error is not None and identity is not None and parent is not None:
            try:
                _remove_descriptor_entry(
                    parent.descriptor,
                    path.name,
                    expected_identity=identity,
                    source_descriptor=descriptor,
                    label=f"failed {label}",
                )
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if parent is not None:
            try:
                os.close(parent.descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
    if primary_error is not None:
        if cleanup_errors:
            primary_error.add_note(
                f"{label} cleanup also failed: "
                + "; ".join(str(error) for error in cleanup_errors)
            )
        raise primary_error
    if cleanup_errors:
        raise cleanup_errors[0]


def _create_sealed_file_binding(
    path: Path,
    *,
    expected_sha256: str | None = None,
    prefer_disk_snapshot: bool = False,
) -> SealedDependencyBinding:
    """Bind one stable local file without imposing a product byte ceiling."""

    expanded = path.expanduser()
    resolved = expanded.resolve(strict=True)
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    source_descriptor = os.open(resolved, flags)
    binding_descriptor = -1
    snapshot_file: Any | None = None
    try:
        source_before = os.fstat(source_descriptor)
        observed = os.stat(resolved, follow_symlinks=False)
        if not stat.S_ISREG(source_before.st_mode) or (
            source_before.st_dev,
            source_before.st_ino,
        ) != (observed.st_dev, observed.st_ino):
            raise JointRiggerArtifactError(
                f"Input USD changed while it was bound: {path}"
            )
        use_memfd = (
            not prefer_disk_snapshot
            and _MEMFD_CREATE is not None
            and source_before.st_size <= _MAX_MEMFD_SNAPSHOT_BYTES
        )
        if use_memfd:
            binding_descriptor = _MEMFD_CREATE(
                b"joint-rigger-source",
                _MFD_CLOEXEC | _MFD_ALLOW_SEALING,
            )
            if binding_descriptor < 0:  # pragma: no cover - syscall failure
                error_number = ctypes.get_errno()
                raise OSError(error_number, os.strerror(error_number))
            write_descriptor = binding_descriptor
        elif prefer_disk_snapshot:
            snapshot_file = _create_disk_backed_snapshot_file(resolved)
            write_descriptor = snapshot_file.fileno()
        else:
            write_descriptor = -1
        digest = hashlib.sha256()
        offset = 0
        while offset < source_before.st_size:
            chunk = os.pread(
                source_descriptor,
                min(1024 * 1024, source_before.st_size - offset),
                offset,
            )
            if not chunk:
                raise JointRiggerArtifactError(
                    "Input USD changed while its root bytes were bound"
                )
            digest.update(chunk)
            if use_memfd or snapshot_file is not None:
                view = memoryview(chunk)
                while view:
                    written = os.write(write_descriptor, view)
                    if written <= 0:  # pragma: no cover - descriptor invariant
                        raise OSError("Could not write bound input snapshot")
                    view = view[written:]
            offset += len(chunk)
        if os.pread(source_descriptor, 1, offset):
            raise JointRiggerArtifactError(
                "Input USD grew while its root bytes were bound"
            )
        source_after = os.fstat(source_descriptor)
        if _descriptor_state(source_before) != _descriptor_state(source_after):
            raise JointRiggerArtifactError(
                "Input USD changed while its root bytes were bound"
            )
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise JointRiggerArtifactError(
                f"Input bytes do not match the expected identity: {path}"
            )
        if use_memfd:
            os.fsync(write_descriptor)
            fcntl.fcntl(binding_descriptor, _F_ADD_SEALS, _SOURCE_MEMFD_SEALS)
            binding = SealedDependencyBinding(
                path=resolved,
                descriptor=binding_descriptor,
                sha256=actual_sha256,
            )
        elif snapshot_file is not None:
            os.fsync(write_descriptor)
            snapshot_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            snapshot_flags |= getattr(os, "O_NONBLOCK", 0)
            binding_descriptor = os.open(
                f"/proc/self/fd/{write_descriptor}",
                snapshot_flags,
            )
            os.fchmod(write_descriptor, 0)
            if (
                _stable_descriptor_sha256(
                    binding_descriptor,
                    label=f"anonymous bound input snapshot {path}",
                )
                != actual_sha256
            ):
                raise JointRiggerArtifactError(
                    f"Anonymous input snapshot changed while it was bound: {path}"
                )
            snapshot_state = os.fstat(binding_descriptor)
            snapshot_file.close()
            snapshot_file = None
            binding = SealedDependencyBinding(
                path=resolved,
                descriptor=binding_descriptor,
                sha256=actual_sha256,
                storage_kind="anonymous_snapshot",
                descriptor_state=_descriptor_state(snapshot_state),
            )
        else:
            binding_descriptor = source_descriptor
            source_descriptor = -1
            binding = SealedDependencyBinding(
                path=resolved,
                descriptor=binding_descriptor,
                sha256=actual_sha256,
                storage_kind="pinned_file",
                descriptor_state=_descriptor_state(source_after),
            )
        _require_sealed_file_binding(binding)
        binding_descriptor = -1
        return binding
    finally:
        try:
            if binding_descriptor >= 0:
                owned_binding_descriptor = binding_descriptor
                binding_descriptor = -1
                os.close(owned_binding_descriptor)
        finally:
            try:
                if snapshot_file is not None:
                    snapshot_file.close()
            finally:
                if source_descriptor >= 0:
                    os.close(source_descriptor)


def _disk_snapshot_candidate_directories(source_path: Path) -> tuple[Path, ...]:
    configured = os.environ.get(_DISK_SNAPSHOT_DIRECTORY_ENV)
    if configured is not None:
        if not configured.strip():
            raise JointRiggerBackendIncompatibleError(
                f"{_DISK_SNAPSHOT_DIRECTORY_ENV} must name a non-empty directory"
            )
        return (Path(os.path.abspath(Path(configured).expanduser())),)

    candidates = (
        Path(tempfile.gettempdir()),
        Path("/var/tmp"),
        source_path.parent,
    )
    normalized: list[Path] = []
    for candidate in candidates:
        absolute = Path(os.path.abspath(candidate.expanduser()))
        if absolute not in normalized:
            normalized.append(absolute)
    return tuple(normalized)


def _descriptor_filesystem_types(descriptor: int) -> frozenset[str]:
    state = os.fstat(descriptor)
    device = f"{os.major(state.st_dev)}:{os.minor(state.st_dev)}"
    filesystem_types: set[str] = set()
    with Path("/proc/self/mountinfo").open(encoding="utf-8") as mount_info:
        for line in mount_info:
            fields = line.split()
            if len(fields) < 7 or fields[2] != device:
                continue
            try:
                separator = fields.index("-", 6)
            except ValueError:
                continue
            if separator + 1 < len(fields):
                filesystem_types.add(fields[separator + 1])
    if not filesystem_types:
        raise OSError(
            f"Could not identify filesystem type for descriptor device {device}"
        )
    return frozenset(filesystem_types)


def _create_disk_backed_snapshot_file(source_path: Path) -> Any:
    failures: list[str] = []
    for directory in _disk_snapshot_candidate_directories(source_path):
        try:
            candidate = tempfile.TemporaryFile(
                prefix="joint-rigger-source-snapshot-",
                dir=directory,
            )
        except OSError as error:
            failures.append(f"{directory}: {error}")
            continue
        try:
            filesystem_types = _descriptor_filesystem_types(candidate.fileno())
        except OSError as error:
            candidate.close()
            failures.append(f"{directory}: {error}")
            continue
        memory_backed = sorted(
            filesystem_types.intersection(_MEMORY_BACKED_FILESYSTEM_TYPES)
        )
        if memory_backed:
            candidate.close()
            failures.append(
                f"{directory}: memory-backed filesystem " + ", ".join(memory_backed)
            )
            continue
        return candidate

    detail = "; ".join(failures) or "no candidate directories"
    raise JointRiggerBackendIncompatibleError(
        "Joint Rigger requires disk-backed scratch storage for exact dependency "
        "snapshots. Set "
        f"{_DISK_SNAPSHOT_DIRECTORY_ENV} to an existing writable directory on "
        f"a non-memory-backed filesystem. Tried: {detail}"
    )


def _require_sealed_file_binding(binding: SealedDependencyBinding) -> None:
    if binding.storage_kind == "sealed_memfd":
        observed_seals = fcntl.fcntl(binding.descriptor, _F_GET_SEALS)
        if observed_seals & _SOURCE_MEMFD_SEALS != _SOURCE_MEMFD_SEALS:
            raise JointRiggerArtifactError("Bound input snapshot lost required seals")
        if binding.descriptor_state is not None:
            raise JointRiggerArtifactError(
                "Sealed memory snapshot unexpectedly records disk state"
            )
    elif binding.storage_kind in {"anonymous_snapshot", "pinned_file"}:
        observed = os.fstat(binding.descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise JointRiggerArtifactError(
                "Bound input descriptor no longer identifies a regular file"
            )
        if binding.descriptor_state is None:
            raise JointRiggerArtifactError(
                "Bound input descriptor is missing its captured state"
            )
        access_mode = fcntl.fcntl(binding.descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        if access_mode != os.O_RDONLY:
            raise JointRiggerArtifactError("Bound input descriptor is not read-only")
    else:  # pragma: no cover - frozen internal construction invariant
        raise JointRiggerArtifactError(
            f"Bound input snapshot has unknown storage kind {binding.storage_kind!r}"
        )
    if (
        _stable_descriptor_sha256(
            binding.descriptor,
            label=f"bound input snapshot {binding.path}",
        )
        != binding.sha256
    ):
        raise JointRiggerArtifactError("Bound input snapshot changed")
    if (
        binding.storage_kind in {"anonymous_snapshot", "pinned_file"}
        and _descriptor_state(os.fstat(binding.descriptor)) != binding.descriptor_state
    ):
        label = (
            "Anonymous input snapshot"
            if binding.storage_kind == "anonymous_snapshot"
            else "Pinned input file"
        )
        raise JointRiggerArtifactError(f"{label} changed")


def _materialize_file(
    *,
    source_descriptor: int,
    source_sha256: str,
    target_path: Path,
) -> None:
    if (
        isinstance(source_descriptor, bool)
        or not isinstance(source_descriptor, int)
        or source_descriptor < 0
    ):
        raise ValueError("bound dependency descriptor must be non-negative")
    before = os.fstat(source_descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("bound dependency descriptor must identify a file")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    output_descriptor = os.open(target_path, flags, 0o600)
    try:
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                source_descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise RuntimeError("Bound dependency changed while it was materialized")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(output_descriptor, view)
                if written <= 0:  # pragma: no cover - regular-file invariant
                    raise OSError("Could not materialize bound dependency")
                view = view[written:]
            offset += len(chunk)
        if os.pread(source_descriptor, 1, offset):
            raise RuntimeError("Bound dependency grew while it was materialized")
        after = os.fstat(source_descriptor)
        if _descriptor_state(before, include_nlink=False) != _descriptor_state(
            after,
            include_nlink=False,
        ):
            raise RuntimeError("Bound dependency changed while it was materialized")
        if digest.hexdigest() != source_sha256:
            raise RuntimeError("Bound dependency SHA-256 does not match its snapshot")
        os.fsync(output_descriptor)
    finally:
        os.close(output_descriptor)


def _validate_bound_projection_dependencies(
    root_path: Path,
    *,
    projection_root: Path,
    materialized_paths: frozenset[Path],
    layer_paths: frozenset[Path],
    restore_paths: Mapping[Path, Path],
) -> None:
    """Validate authored locators from sealed copies before opening a stage."""

    if root_path.suffix.lower() == ".usdz":
        from world_understanding.utils.usd.package import (
            USD_LAYER_EXTENSIONS,
            extract_usdz_package_for_edit,
        )

        try:
            with tempfile.TemporaryDirectory(
                dir=root_path.parent,
                prefix=f".{root_path.name}.sealed-validation-",
            ) as validation_dir_value:
                extraction_root = Path(validation_dir_value) / "contents"
                extracted_root = extract_usdz_package_for_edit(
                    root_path,
                    extraction_root,
                )
                extracted_files = frozenset(
                    path for path in extraction_root.rglob("*") if path.is_file()
                )
                extracted_layers = frozenset(
                    path
                    for path in extracted_files
                    if path.suffix.lower() in USD_LAYER_EXTENSIONS
                )
                _validate_bound_projection_dependencies(
                    extracted_root,
                    projection_root=extraction_root,
                    materialized_paths=extracted_files,
                    layer_paths=extracted_layers,
                    restore_paths={
                        path: Path("/") / path.relative_to(extraction_root)
                        for path in extracted_files
                    },
                )
        except JointRiggerBackendIncompatibleError:
            raise
        except Exception as exc:
            raise JointRiggerBackendIncompatibleError(
                f"Could not validate sealed USDZ dependency closure: {exc}"
            ) from exc
        return

    from pxr import Ar, Sdf, UsdUtils

    normalized_projection = projection_root.resolve(strict=True)
    absolute_locators: set[str] = set()
    remote_locators: set[str] = set()
    package_locators: set[str] = set()
    escaped_locators: set[str] = set()
    missing_locators: set[str] = set()
    nested_aliases: set[str] = set()
    cross_parent_layer_aliases: set[str] = set()
    for layer_path in sorted(layer_paths, key=lambda item: item.as_posix()):
        if not layer_path.is_relative_to(normalized_projection):
            raise RuntimeError(
                f"Bound layer is outside the private projection: {layer_path}"
            )
        try:
            layer = Sdf.Layer.OpenAsAnonymous(str(layer_path))
        except Exception as exc:
            raise RuntimeError(
                f"Could not inspect bound layer dependency locators: {layer_path}: {exc}"
            ) from exc
        if not layer:
            raise RuntimeError(
                f"Could not inspect bound layer dependency locators: {layer_path}"
            )
        _remove_informational_asset_identifiers(layer)

        # Bind the owner path explicitly: this callback is invoked after loop
        # construction and must not capture the next layer_path iteration.
        def inspect_locator(
            locator: str,
            *,
            owner_layer_path: Path = layer_path,
        ) -> str:
            if not locator:
                return locator
            if Ar.IsPackageRelativePath(locator):
                package_locators.add(locator)
                return locator
            windows_path = PureWindowsPath(locator)
            if windows_path.drive:
                absolute_locators.add(locator)
                return locator
            if urlparse(locator).scheme:
                remote_locators.add(locator)
                return locator
            locator_path = Path(locator)
            if locator_path.is_absolute() or windows_path.is_absolute():
                absolute_locators.add(locator)
                return locator
            projected_target = Path(
                os.path.abspath(owner_layer_path.parent / locator_path)
            )
            if not projected_target.is_relative_to(normalized_projection):
                escaped_locators.add(locator)
                return locator
            if projected_target.suffix.lower() == ".usdz":
                package_locators.add(locator)
                return locator
            if (
                projected_target not in materialized_paths
                or not projected_target.is_file()
            ):
                missing_locators.add(locator)
                return locator
            logical_alias = Path("/") / projected_target.relative_to(
                normalized_projection
            )
            restore_path = restore_paths.get(projected_target)
            if (
                projected_target in layer_paths
                and restore_path is not None
                and logical_alias.parent != restore_path.parent
            ):
                cross_parent_layer_aliases.add(locator)
            if owner_layer_path != root_path and restore_path != logical_alias:
                nested_aliases.add(locator)
            return locator

        try:
            UsdUtils.ModifyAssetPaths(
                layer,
                inspect_locator,
                keepEmptyPathsInArrays=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not inspect bound layer dependency locators: {layer_path}: {exc}"
            ) from exc

    unsupported = {
        "absolute": sorted(absolute_locators),
        "remote": sorted(remote_locators),
        "package": sorted(package_locators),
        "escaped": sorted(escaped_locators),
        "missing": sorted(missing_locators),
        "nested_symlink_alias": sorted(nested_aliases),
        "cross_parent_layer_alias": sorted(cross_parent_layer_aliases),
    }
    failures = {kind: values for kind, values in unsupported.items() if values}
    if failures:
        raise JointRiggerBackendIncompatibleError(
            f"Bound Joint Rigger raw-source dependency locators are unsupported: "
            f"{failures}"
        )


def _remove_informational_asset_identifiers(layer: Any) -> None:
    """Permanently exclude model identity metadata from locator inspection.

    ``assetInfo.identifier`` records where a model originated. OpenUSD does not
    compose or load that path, but ``UsdUtils.ModifyAssetPaths`` visits it along
    with runtime asset dependencies. Remove only that dictionary entry from the
    anonymous validation layer so stale provenance cannot be mistaken for a
    missing file while all load-bearing asset paths remain fail-closed.

    ``Sdf.Layer.Traverse`` reports variant-selection paths, but looking up such
    a path returns its ``VariantSpec`` instead of the associated ``PrimSpec``.
    Walk the prim-spec ownership graph directly so every nested variant root is
    sanitized along with its ordinary children.
    """

    _take_informational_asset_identifiers(layer)


def _iter_prim_specs(layer: Any) -> Iterator[Any]:
    """Yield every ordinary and variant-owned prim spec exactly once."""

    pending_prim_specs: list[Any] = list(layer.rootPrims)
    while pending_prim_specs:
        prim_spec = pending_prim_specs.pop()
        yield prim_spec
        pending_prim_specs.extend(prim_spec.nameChildren)
        for variant_set in prim_spec.variantSets.values():
            pending_prim_specs.extend(
                variant.primSpec for variant in variant_set.variants.values()
            )


def _take_informational_asset_identifiers(
    layer: Any,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Temporarily remove and return exact model-origin identifier metadata."""

    removed: list[tuple[str, dict[str, Any]]] = []
    for prim_spec in _iter_prim_specs(layer):
        asset_info = prim_spec.GetInfo("assetInfo")
        if isinstance(asset_info, dict) and "identifier" in asset_info:
            removed.append((str(prim_spec.path), dict(asset_info)))
            retained = dict(asset_info)
            retained.pop("identifier")
            if retained:
                prim_spec.SetInfo("assetInfo", retained)
            else:
                prim_spec.ClearInfo("assetInfo")
    return tuple(removed)


def _restore_informational_asset_identifiers(
    layer: Any,
    removed: tuple[tuple[str, dict[str, Any]], ...],
) -> None:
    """Restore exact metadata after re-resolving every captured prim spec."""

    prim_specs_by_path: dict[str, Any] = {}
    for prim_spec in _iter_prim_specs(layer):
        prim_path = str(prim_spec.path)
        if prim_path in prim_specs_by_path:
            raise RuntimeError(
                "Could not restore informational asset identifier because the "
                f"prim-spec path is ambiguous: {prim_path}"
            )
        prim_specs_by_path[prim_path] = prim_spec
    for prim_path, asset_info in removed:
        prim_spec = prim_specs_by_path.get(prim_path)
        if prim_spec is None:
            raise RuntimeError(
                "Could not restore informational asset identifier because the "
                f"prim spec no longer exists: {prim_path}"
            )
        prim_spec.SetInfo("assetInfo", asset_info)


def _copy_descriptor_bytes(
    source_descriptor: int,
    target_descriptor: int,
    *,
    expected_source: os.stat_result,
    label: str,
) -> None:
    offset = 0
    while offset < expected_source.st_size:
        chunk = os.pread(
            source_descriptor,
            min(1024 * 1024, expected_source.st_size - offset),
            offset,
        )
        if not chunk:
            raise JointRiggerArtifactError(f"{label} source changed during copy")
        remaining = memoryview(chunk)
        while remaining:
            written = os.write(target_descriptor, remaining)
            if written <= 0:  # pragma: no cover - regular-file OS invariant
                raise OSError(f"Short write while copying {label}")
            remaining = remaining[written:]
        offset += len(chunk)
    if os.pread(source_descriptor, 1, offset):
        raise JointRiggerArtifactError(f"{label} source grew during copy")
    if _descriptor_state(os.fstat(source_descriptor)) != _descriptor_state(
        expected_source
    ):
        raise JointRiggerArtifactError(f"{label} source changed during copy")


def _stable_descriptor_sha256(descriptor: int, *, label: str) -> str:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise JointRiggerArtifactError(f"Private {label} is not a regular file")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(
            descriptor,
            min(1024 * 1024, before.st_size - offset),
            offset,
        )
        if not chunk:
            raise JointRiggerArtifactError(f"Private {label} changed while hashing")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, offset):
        raise JointRiggerArtifactError(f"Private {label} grew while hashing")
    after = os.fstat(descriptor)
    if _descriptor_state(before) != _descriptor_state(after):
        raise JointRiggerArtifactError(f"Private {label} changed while hashing")
    return digest.hexdigest()


def _descriptor_state(
    value: os.stat_result,
    *,
    include_nlink: bool = True,
) -> tuple[int, ...]:
    fields = [
        value.st_dev,
        value.st_ino,
        value.st_mode,
    ]
    if include_nlink:
        fields.append(value.st_nlink)
    fields.extend(
        [
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        ]
    )
    return tuple(fields)


def _close_descriptors(descriptors: list[int]) -> list[Exception]:
    errors: list[Exception] = []
    for descriptor in dict.fromkeys(descriptors):
        try:
            os.close(descriptor)
        except Exception as exc:
            errors.append(exc)
    return errors


def _add_cleanup_error_note(
    primary_error: BaseException,
    *,
    label: str,
    errors: list[Exception],
) -> None:
    if errors:
        primary_error.add_note(f"{label}: " + "; ".join(str(error) for error in errors))


__all__ = [
    "BoundInputDirectory",
    "SealedDependencyBinding",
    "SealedSourceBinding",
    "bound_input_dependency_snapshots",
    "close_source_binding",
    "copy_regular_file_to_new_path",
    "create_sealed_source_binding",
    "freeze_bound_projection_root",
    "materialize_bound_input",
    "remove_bound_input_directory",
    "require_sealed_source_binding",
    "restore_bound_projection_paths",
    "write_new_text_file",
]
