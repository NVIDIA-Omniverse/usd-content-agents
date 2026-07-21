# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact staging and rollback for the shared Joint Rigger facade.

The generated asset is the commit point for a Joint Rigger run.  Reports and
an optional composition sidecar are promoted first; the generated root is
promoted last.  Consumers must therefore ignore reports or a sidecar when the
root is absent.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

_SIDECAR_BUNDLE_SCHEMA_VERSION = "world-understanding-joint-rigger-sidecar-v1"
_DIRECTORY_TREE_SCHEMA_VERSION = "world-understanding-artifact-tree-v1"
_CAPTURED_TARGET_TREE_SCHEMA_VERSION = (
    "world-understanding-joint-rigger-captured-target-tree-v1"
)
_ARTIFACT_TREE_MAX_DEPTH = 64
_ARTIFACT_TREE_MAX_ENTRIES = 100_000
_ARTIFACT_TREE_MAX_BYTES = 8 * 1024 * 1024 * 1024
_RENAME_NOREPLACE = 1
_PROC_SELF_FDINFO = Path("/proc/self/fdinfo")
_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEAT2: Any
try:
    _RENAMEAT2 = _LIBC.renameat2
except AttributeError:  # pragma: no cover - Linux runtime contract
    _RENAMEAT2 = None
else:
    _RENAMEAT2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    _RENAMEAT2.restype = ctypes.c_int


class ConcurrentArtifactPublicationError(RuntimeError):
    """Another writer is publishing to at least one requested target."""


class CommittedArtifactPublicationCleanupError(RuntimeError):
    """Publication committed, but one or more post-commit cleanups failed."""

    committed = True

    def __init__(self, cleanup_errors: Iterable[Exception]) -> None:
        self.cleanup_errors = tuple(cleanup_errors)
        super().__init__(
            "Artifact publication committed, but cleanup failed: "
            + "; ".join(str(error) for error in self.cleanup_errors)
        )


@dataclass
class _ArtifactTreeTraversalBudget:
    """Bound work across one caller-visible artifact-tree traversal."""

    label: str
    entries: int = 0
    total_bytes: int = 0

    def sorted_child_names(
        self,
        descriptor: int,
        *,
        relative_path: str,
    ) -> list[str]:
        """Collect direct child names without exceeding the remaining entries."""

        return _bounded_sorted_directory_names(
            descriptor,
            maximum_names=_ARTIFACT_TREE_MAX_ENTRIES - self.entries,
            overflow_message=(
                f"{self.label} exceeds artifact-tree entry limit at {relative_path}"
            ),
        )

    def require_depth(self, *, relative_path: str, depth: int) -> None:
        """Reject a descent before opening a child beyond the fixed ceiling."""

        if depth > _ARTIFACT_TREE_MAX_DEPTH:
            raise RuntimeError(
                f"{self.label} exceeds artifact-tree depth limit at {relative_path}"
            )

    def consume(
        self,
        *,
        relative_path: str,
        depth: int,
        byte_count: int = 0,
    ) -> None:
        """Account for one entry before opening, hashing, copying, or deleting it."""

        self.require_depth(relative_path=relative_path, depth=depth)
        next_entries = self.entries + 1
        if next_entries > _ARTIFACT_TREE_MAX_ENTRIES:
            raise RuntimeError(
                f"{self.label} exceeds artifact-tree entry limit at {relative_path}"
            )
        if byte_count < 0:
            raise RuntimeError(
                f"{self.label} has an invalid negative size at {relative_path}"
            )
        next_bytes = self.total_bytes + byte_count
        if next_bytes > _ARTIFACT_TREE_MAX_BYTES:
            raise RuntimeError(
                f"{self.label} exceeds artifact-tree byte limit at {relative_path}"
            )
        self.entries = next_entries
        self.total_bytes = next_bytes


def _bounded_sorted_directory_names(
    descriptor: int,
    *,
    maximum_names: int,
    overflow_message: str,
) -> list[str]:
    """Collect and sort at most ``maximum_names`` fd-relative child names."""

    if maximum_names < 0:
        raise RuntimeError(overflow_message)
    names: list[str] = []
    with os.scandir(descriptor) as entries:
        for entry in entries:
            if len(names) >= maximum_names:
                raise RuntimeError(overflow_message)
            names.append(entry.name)
    names.sort()
    return names


@dataclass
class _PublicationCleanupState:
    """Track the commit boundary and every subsequent cleanup failure."""

    committed: bool = False
    errors: list[Exception] = field(default_factory=list)


def _route_context_cleanup_errors(
    errors: list[Exception],
    cleanup_state: _PublicationCleanupState | None,
    *,
    label: str,
) -> None:
    """Route post-commit cleanup errors or raise ordinary cleanup failures."""

    _route_cleanup_failures(
        [(label, error) for error in errors],
        cleanup_state=cleanup_state,
        label=label,
    )


def _attach_cleanup_failure(
    primary_error: BaseException,
    cleanup_error: BaseException,
    *,
    context: str,
) -> None:
    """Attach one cleanup failure without replacing an active primary."""

    summary = f"{context}: {type(cleanup_error).__name__}"
    if str(cleanup_error):
        summary += f": {cleanup_error}"
    BaseException.add_note(primary_error, summary)
    for note in getattr(cleanup_error, "__notes__", ()):
        BaseException.add_note(primary_error, f"{context} detail: {note}")


def _collect_cleanup_failures(
    steps: Iterable[tuple[str, Callable[[], None]]],
) -> list[tuple[str, BaseException]]:
    """Run every independent cleanup step and retain exact failures."""

    failures: list[tuple[str, BaseException]] = []
    for context, cleanup in steps:
        try:
            cleanup()
        except BaseException as cleanup_error:
            failures.append((context, cleanup_error))
    return failures


def _route_cleanup_failures(
    failures: list[tuple[str, BaseException]],
    *,
    primary_error: BaseException | None = None,
    cleanup_state: _PublicationCleanupState | None = None,
    label: str,
) -> None:
    """Preserve an active primary or raise/route standalone cleanup failures."""

    recorded_errors = (
        tuple(cleanup_state.errors)
        if cleanup_state is not None and cleanup_state.committed
        else ()
    )
    if primary_error is not None:
        if not isinstance(primary_error, Exception) and recorded_errors:
            assert cleanup_state is not None
            cleanup_state.errors.clear()
            for recorded_error in recorded_errors:
                _attach_cleanup_failure(
                    primary_error,
                    recorded_error,
                    context="Earlier committed cleanup also failed",
                )
        for context, failure in failures:
            _attach_cleanup_failure(
                primary_error,
                failure,
                context=context,
            )
        return
    if not failures:
        return

    fatal_index = next(
        (
            index
            for index, (_, failure) in enumerate(failures)
            if not isinstance(failure, Exception)
        ),
        None,
    )
    if fatal_index is not None:
        fatal_error = failures[fatal_index][1]
        if recorded_errors:
            assert cleanup_state is not None
            cleanup_state.errors.clear()
            for recorded_error in recorded_errors:
                _attach_cleanup_failure(
                    fatal_error,
                    recorded_error,
                    context="Earlier committed cleanup also failed",
                )
        for index, (context, failure) in enumerate(failures):
            if index == fatal_index:
                continue
            _attach_cleanup_failure(
                fatal_error,
                failure,
                context=context,
            )
        raise fatal_error

    ordinary_errors = [
        failure for _, failure in failures if isinstance(failure, Exception)
    ]
    if cleanup_state is not None and cleanup_state.committed:
        cleanup_state.errors.extend(ordinary_errors)
        return
    if len(ordinary_errors) == 1:
        raise ordinary_errors[0]
    raise ExceptionGroup(label, ordinary_errors)


def _run_cleanup_steps(
    steps: Iterable[tuple[str, Callable[[], None]]],
    *,
    primary_error: BaseException | None = None,
    cleanup_state: _PublicationCleanupState | None = None,
    label: str,
) -> None:
    """Run all cleanup steps, then apply the shared ownership policy."""

    _route_cleanup_failures(
        _collect_cleanup_failures(steps),
        primary_error=primary_error,
        cleanup_state=cleanup_state,
        label=label,
    )


def _absolute_lexical_path(path: str | Path) -> Path:
    """Expand one path once without following any filesystem symlinks."""

    return Path(os.path.abspath(Path(path).expanduser()))


def _require_present_invariant[T](value: T | None, *, label: str) -> T:
    """Return an internal invariant value or fail closed under ``python -O``."""

    if value is None:
        raise RuntimeError(f"Internal Joint Rigger invariant is missing: {label}")
    return value


@dataclass(frozen=True)
class JointRiggerArtifactTargets:
    """Physical and publication paths for one complete artifact set.

    Backends write only to the physical output, report, and optional sidecar
    paths. When those paths are transaction staging locations, authored USD
    references, externally reported URIs, and destructive-read protection must
    instead use the corresponding ``publication_*`` paths. Caller-facing final
    targets default publication paths to their physical paths; staging preserves
    those exact final paths while changing only the physical write locations.

    Every coordinate is expanded to one absolute lexical path at construction
    without resolving symlinks. Instances are frozen so a backend cannot
    observe a publication layout that changes during authoring or after a
    working-directory change.
    """

    output_path: Path
    diagnostics_path: Path
    result_path: Path
    sidecar_path: Path | None = None
    publication_output_path: Path | None = None
    publication_sidecar_path: Path | None = None
    publication_diagnostics_path: Path | None = None
    publication_result_path: Path | None = None
    _created_file_binder: Callable[[Path, os.stat_result], None] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Normalize paths and materialize caller-facing publication defaults."""
        output_path = _absolute_lexical_path(self.output_path)
        object.__setattr__(self, "output_path", output_path)
        object.__setattr__(
            self,
            "diagnostics_path",
            _absolute_lexical_path(self.diagnostics_path),
        )
        object.__setattr__(
            self,
            "result_path",
            _absolute_lexical_path(self.result_path),
        )
        sidecar_path = None
        if self.sidecar_path is not None:
            sidecar_path = _absolute_lexical_path(self.sidecar_path)
        object.__setattr__(self, "sidecar_path", sidecar_path)

        publication_output_path = (
            output_path
            if self.publication_output_path is None
            else _absolute_lexical_path(self.publication_output_path)
        )
        object.__setattr__(
            self,
            "publication_output_path",
            publication_output_path,
        )
        publication_diagnostics_path = (
            self.diagnostics_path
            if self.publication_diagnostics_path is None
            else _absolute_lexical_path(self.publication_diagnostics_path)
        )
        object.__setattr__(
            self,
            "publication_diagnostics_path",
            publication_diagnostics_path,
        )
        publication_result_path = (
            self.result_path
            if self.publication_result_path is None
            else _absolute_lexical_path(self.publication_result_path)
        )
        object.__setattr__(
            self,
            "publication_result_path",
            publication_result_path,
        )
        if sidecar_path is None:
            if self.publication_sidecar_path is not None:
                raise ValueError(
                    "publication_sidecar_path requires a physical sidecar_path"
                )
            publication_sidecar_path = None
        else:
            publication_sidecar_path = (
                sidecar_path
                if self.publication_sidecar_path is None
                else _absolute_lexical_path(self.publication_sidecar_path)
            )
        object.__setattr__(
            self,
            "publication_sidecar_path",
            publication_sidecar_path,
        )

    def _bind_created_file(self, path: Path, metadata: os.stat_result) -> None:
        """Transfer one newly created staging inode to facade cleanup ownership."""

        binder = self._created_file_binder
        if binder is None:
            raise RuntimeError(
                "Combined authoring requires facade-owned staging cleanup"
            )
        binder(_absolute_lexical_path(path), metadata)


@dataclass
class _StagingPromotionState:
    """Identity-bound move evidence shared with one staging reservation."""

    source_identity: tuple[int, int] | None = None
    source_parent_identity: tuple[int, int] | None = None
    source_descriptor: int = -1
    source_is_directory: bool = False
    source_tree_sha256: str | None = None
    source_mount_id: int | None = None
    committed_identity: tuple[int, int] | None = None

    @property
    def committed(self) -> bool:
        """Return whether the exact staged inode reached its publication name."""

        return self.committed_identity is not None


@dataclass(frozen=True)
class _CapturedTargetHandle:
    """Shared, exactly-once close authority for one captured target inode."""

    descriptor: int
    closed: bool = False

    def close(self) -> None:
        """Close the retained descriptor at most once, even after close errors."""

        if self.closed:
            return
        descriptor = self.descriptor
        object.__setattr__(self, "closed", True)
        object.__setattr__(self, "descriptor", -1)
        os.close(descriptor)


_CapturedStatState = tuple[int, int, int, int, int, int, int, int, int]


@dataclass(frozen=True)
class _CapturedTargetState:
    """Descriptor-bound final-target state captured before backend authoring."""

    requested_path: Path
    parent_identity: tuple[int, int]
    entry_state: _CapturedStatState | None
    entry_handle: _CapturedTargetHandle | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    entry_mount_id: int | None = None
    content_sha256: str | None = None

    @property
    def entry_identity(self) -> tuple[int, int] | None:
        """Return the captured device/inode pair for promotion machinery."""

        if self.entry_state is None:
            return None
        return self.entry_state[:2]


@dataclass(frozen=True)
class StagedArtifact:
    """One staged artifact and its final transaction target.

    A paired ``source_descriptor`` and ``source_sha256`` opt into detached-copy
    publication. The descriptor must remain open read-only through promotion;
    regular files use a raw-byte digest and directories use the versioned exact
    tree digest returned by ``directory_tree_sha256``. Successful regular-file
    publication consumes the staged name; a directory source remains
    caller-owned so its facade-private tree can be cleaned by its creator.
    """

    staged_path: Path
    target_path: Path
    label: str
    source_descriptor: int | None = None
    source_sha256: str | None = None
    replace_existing: bool = True
    _initial_target_state: _CapturedTargetState | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _promotion_state: _StagingPromotionState | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate the optional descriptor-bound regular-file contract."""

        if self._initial_target_state is not None and (
            _absolute_lexical_path(self.target_path)
            != self._initial_target_state.requested_path
        ):
            raise ValueError(
                "initial target state must describe the artifact target path"
            )
        if (self.source_descriptor is None) != (self.source_sha256 is None):
            raise ValueError(
                "source_descriptor and source_sha256 must be provided together"
            )
        if self.source_descriptor is None:
            return
        if (
            isinstance(self.source_descriptor, bool)
            or not isinstance(self.source_descriptor, int)
            or self.source_descriptor < 0
        ):
            raise ValueError("source_descriptor must be a non-negative file descriptor")
        assert self.source_sha256 is not None
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in self.source_sha256
            )
        ):
            raise ValueError("source_sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "source_sha256", self.source_sha256.lower())


@dataclass(frozen=True)
class _BoundDirectory:
    """One directory inode held open across a publication transaction."""

    locator_path: Path
    opened_path: Path
    descriptor: int
    identity: tuple[int, int]


@dataclass
class _StagingCleanupReservation:
    """One backend staging name bound to its original parent directory."""

    parent: _BoundDirectory
    name: str
    owned_identity: tuple[int, int] | None = None
    owned_descriptor: int = -1
    is_owner_directory: bool = False
    publication_name: str | None = None
    promotion_state: _StagingPromotionState | None = None
    payload_name: str | None = None
    payload_identity: tuple[int, int] | None = None
    payload_descriptor: int = -1
    payload_is_directory: bool = False
    payload_binding_attempted: bool = False
    payload_promotion_state: _StagingPromotionState | None = None
    binding_revoked: bool = False
    closed: bool = False


@dataclass(frozen=True)
class _BoundEntry:
    """One lexical directory entry bound to a held parent descriptor."""

    parent: _BoundDirectory
    name: str

    @property
    def path(self) -> Path:
        """Return the opened-parent path used only for diagnostics."""

        return self.parent.opened_path / self.name


@dataclass(frozen=True)
class _QuarantinedDescriptorEntry:
    """One exact inode atomically moved away from its discoverable name."""

    original_name: str
    quarantine_name: str
    identity: tuple[int, int]


@dataclass(frozen=True)
class _LockedTarget:
    """One final target entry under a held and locked physical parent."""

    requested_path: Path
    entry: _BoundEntry
    identity: bytes


@dataclass(frozen=True)
class _BoundDescriptorSource:
    """One readable file or directory descriptor bound to expected content."""

    descriptor: int
    identity: tuple[int, int]
    sha256: str
    mode: int
    is_directory: bool
    mount_id: int | None = None


@dataclass(frozen=True)
class _DetachedTarget:
    """Identity and digest of one promoter-owned detached target copy."""

    identity: tuple[int, int]
    sha256: str
    mode: int
    is_directory: bool


@dataclass(frozen=True)
class _BoundArtifact:
    """One staged-to-final move bound to both parent directory inodes."""

    artifact: StagedArtifact
    staged_entry: _BoundEntry
    target_entry: _BoundEntry
    descriptor_source: _BoundDescriptorSource | None


@dataclass
class _ArtifactBackup:
    """One previous target moved into an fd-bound rollback directory."""

    bound_artifact: _BoundArtifact
    directory: _BoundDirectory
    directory_name: str
    artifact_entry: _BoundEntry
    artifact_identity: tuple[int, int]
    post_move_state: _CapturedStatState | None = None
    post_move_content_sha256: str | None = None


@dataclass(frozen=True)
class StagedJointRiggerArtifacts:
    """Physical staging paths, final publication layout, and owner state."""

    final_targets: JointRiggerArtifactTargets
    staged_targets: JointRiggerArtifactTargets
    sidecar_owner_path: Path | None = None
    _cleanup_reservations: tuple[_StagingCleanupReservation, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    _initial_target_states: tuple[_CapturedTargetState, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def cleanup(self) -> None:
        """Remove every staged path, including backend-created sidecars."""
        object.__setattr__(self.staged_targets, "_created_file_binder", None)
        steps: list[tuple[str, Callable[[], None]]] = []
        if self._cleanup_reservations:
            steps.append(
                (
                    "Staged artifact reservation cleanup failed",
                    partial(
                        _cleanup_staging_reservations,
                        self._cleanup_reservations,
                    ),
                )
            )
        else:
            cleanup_paths = _target_paths(self.staged_targets)
            if self.sidecar_owner_path is not None:
                cleanup_paths.append(self.sidecar_owner_path)
            steps.extend(
                (
                    f"Staged artifact cleanup failed for {path}",
                    partial(remove_artifact, path),
                )
                for path in cleanup_paths
            )
        handles = {
            id(state.entry_handle): state.entry_handle
            for state in self._initial_target_states
            if state.entry_handle is not None
        }
        steps.extend(
            (
                "Captured target descriptor cleanup failed",
                handle.close,
            )
            for handle in handles.values()
        )
        _run_cleanup_steps(
            steps,
            label="Staged artifact cleanup failed",
        )


def validate_artifact_targets(
    targets: JointRiggerArtifactTargets,
    *,
    read_paths: Iterable[tuple[str, str | Path]] = (),
) -> None:
    """Reject aliases, nested outputs, and destructive target shapes.

    Validation completes before a previous artifact set is touched. Read inputs
    may not alias an output or live below one. The ancestor check applies to
    file targets as well as the optional sidecar directory because an existing
    symlink at a file target can name an input directory and be replaced during
    publication. This public preflight accepts caller-facing final targets only:
    every ``publication_*`` coordinate must equal its corresponding physical
    path. ``create_staged_artifact_targets`` is the sole constructor for the
    intentionally different backend staging layout.
    """
    _validate_caller_publication_layout(targets)
    _validate_sidecar_parent(targets)
    target_items = _target_items(targets)
    normalized_targets: list[tuple[str, Path, Path, bool]] = []
    for label, path, expects_directory in target_items:
        lexical = _absolute_lexical_path(path)
        resolved = path.expanduser().resolve(strict=False)
        for previous_label, _, previous_resolved, _ in normalized_targets:
            if resolved == previous_resolved:
                raise ValueError(f"{label} must not alias {previous_label}: {path}")
            if _is_relative_to(resolved, previous_resolved) or _is_relative_to(
                previous_resolved, resolved
            ):
                raise ValueError(
                    "Nested Joint Rigger artifact targets are not supported: "
                    f"{label}={path} overlaps {previous_label}"
                )
        normalized_targets.append((label, lexical, resolved, expects_directory))

        if not path.exists() and not path.is_symlink():
            continue
        if expects_directory:
            if path.is_symlink() or not path.is_dir():
                raise ValueError(
                    f"Existing {label} must be a non-symlink directory: {path}"
                )
        elif path.is_dir() and not path.is_symlink():
            raise IsADirectoryError(
                f"Refusing to replace existing {label} directory: {path}"
            )

    normalized_reads = [
        (
            label,
            _absolute_lexical_path(path),
            Path(path).expanduser().resolve(strict=False),
        )
        for label, path in read_paths
    ]
    for read_label, read_lexical, read_resolved in normalized_reads:
        for (
            target_label,
            target_lexical,
            target_resolved,
            target_is_directory,
        ) in normalized_targets:
            if read_lexical == target_lexical or read_resolved == target_resolved:
                raise ValueError(
                    f"{target_label} must not alias {read_label}: {target_lexical}"
                )
            read_is_below_target = _is_relative_to(
                read_lexical, target_lexical
            ) or _is_relative_to(read_resolved, target_resolved)
            if target_is_directory and read_is_below_target:
                raise ValueError(
                    f"{read_label} must not be inside {target_label}: {read_lexical}"
                )
            if read_is_below_target:
                raise ValueError(
                    f"{target_label} must not be an ancestor of {read_label}: "
                    f"{target_lexical}"
                )


def invalidate_artifact_targets(targets: JointRiggerArtifactTargets) -> None:
    """Invalidate a previous run with its generated root removed first."""
    # The root is the commit point.  Remove it before reports or a sidecar so a
    # hard interruption cannot leave an apparently complete old result.
    remove_artifact(targets.output_path)
    if targets.sidecar_path is not None:
        remove_artifact(targets.sidecar_path)
    remove_artifact(targets.diagnostics_path)
    remove_artifact(targets.result_path)


def create_staged_artifact_targets(
    targets: JointRiggerArtifactTargets,
) -> StagedJointRiggerArtifacts:
    """Create unique same-filesystem staging paths for every final target.

    Staged roots and reports remain beside their final targets. A configured
    physical sidecar lives under a temporary owner directory beside the final
    root. Publication metadata always retains the exact final root and sidecar
    paths; conforming backends author relative references from that logical
    layout rather than from the intentionally different staging layout. Each
    final target's parent inode and absent-or-present entry inode are captured
    through the reservation's held parent descriptor. Promotion refuses any
    drift from that initial state before it backs up or replaces a target.
    """
    _validate_caller_publication_layout(targets)
    _validate_sidecar_parent(targets)

    cleanup_reservations: list[_StagingCleanupReservation] = []
    initial_target_states: list[_CapturedTargetState] = []
    sidecar_owner_path: Path | None = None
    try:
        staged_output, output_reservation = _reserve_backend_staging_name(
            targets.output_path,
            descriptor_owned=False,
        )
        cleanup_reservations.append(output_reservation)
        initial_target_states.append(
            _capture_target_state(output_reservation.parent, targets.output_path)
        )
        staged_sidecar = None
        if targets.sidecar_path is not None:
            sidecar_owner_path, sidecar_reservation = _create_sidecar_owner_reservation(
                targets.output_path.parent,
                target_name=targets.sidecar_path.name,
            )
            cleanup_reservations.append(sidecar_reservation)
            staged_sidecar = sidecar_owner_path / targets.sidecar_path.name
            initial_target_states.append(
                _capture_target_state(sidecar_reservation.parent, targets.sidecar_path)
            )

        staged_diagnostics, diagnostics_reservation = _reserve_backend_staging_name(
            targets.diagnostics_path
        )
        cleanup_reservations.append(diagnostics_reservation)
        initial_target_states.append(
            _capture_target_state(
                diagnostics_reservation.parent,
                targets.diagnostics_path,
            )
        )
        staged_result, result_reservation = _reserve_backend_staging_name(
            targets.result_path
        )
        cleanup_reservations.append(result_reservation)
        initial_target_states.append(
            _capture_target_state(result_reservation.parent, targets.result_path)
        )

        staged = JointRiggerArtifactTargets(
            output_path=staged_output,
            diagnostics_path=staged_diagnostics,
            result_path=staged_result,
            sidecar_path=staged_sidecar,
            publication_output_path=targets.publication_output_path,
            publication_sidecar_path=targets.publication_sidecar_path,
            publication_diagnostics_path=targets.publication_diagnostics_path,
            publication_result_path=targets.publication_result_path,
        )
    except BaseException as creation_error:
        _run_cleanup_steps(
            [
                (
                    "Partial staging reservation cleanup also failed",
                    partial(
                        _cleanup_staging_reservations,
                        tuple(cleanup_reservations),
                    ),
                ),
                *(
                    (
                        "Partial captured target descriptor cleanup also failed",
                        state.entry_handle.close,
                    )
                    for state in initial_target_states
                    if state.entry_handle is not None
                ),
            ],
            primary_error=creation_error,
            label="Partial staging target cleanup failed",
        )
        raise
    artifacts = StagedJointRiggerArtifacts(
        final_targets=targets,
        staged_targets=staged,
        sidecar_owner_path=sidecar_owner_path,
        _cleanup_reservations=tuple(cleanup_reservations),
        _initial_target_states=tuple(initial_target_states),
    )
    object.__setattr__(
        staged,
        "_created_file_binder",
        partial(_bind_staging_cleanup_identity, artifacts),
    )
    return artifacts


def _capture_target_state(
    parent: _BoundDirectory,
    target_path: Path,
) -> _CapturedTargetState:
    """Capture one final entry and retain exact content/state authority."""

    requested_path = _absolute_lexical_path(target_path)
    if requested_path.parent != parent.locator_path:
        raise RuntimeError(
            "Staging reservation parent does not own its publication target: "
            f"{requested_path}"
        )
    _require_bound_directory_unchanged(parent)
    entry_state = _optional_target_entry_state(
        parent.descriptor,
        requested_path.name,
    )
    if entry_state is None:
        _require_bound_directory_unchanged(parent)
        return _CapturedTargetState(
            requested_path=requested_path,
            parent_identity=parent.identity,
            entry_state=None,
        )

    mode = entry_state[2]
    if stat.S_ISREG(mode):
        flags = os.O_RDONLY | os.O_NOFOLLOW
        flags |= getattr(os, "O_NONBLOCK", 0)
    elif stat.S_ISDIR(mode):
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    elif stat.S_ISLNK(mode):
        o_path = getattr(os, "O_PATH", None)
        if o_path is None:  # pragma: no cover - official targets are Linux
            raise RuntimeError("Safe target capture requires Linux O_PATH")
        flags = o_path | os.O_NOFOLLOW
    else:
        raise ValueError(
            "Existing publication targets must be regular files, symlinks, or "
            f"directories: {requested_path}"
        )
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(
        requested_path.name,
        flags,
        dir_fd=parent.descriptor,
    )
    try:
        opened_state = _captured_stat_state(os.fstat(descriptor))
        if opened_state != entry_state:
            raise RuntimeError(
                f"Publication target changed while captured: {requested_path}"
            )
        parent_mount_id = _descriptor_mount_id(parent.descriptor)
        entry_mount_id = _descriptor_mount_id(descriptor)
        if entry_mount_id != parent_mount_id:
            raise ValueError(
                f"Existing publication target is a mount point: {requested_path}"
            )
        content_sha256 = _captured_target_content_sha256(
            descriptor,
            entry_state=entry_state,
            expected_mount_id=entry_mount_id,
            label=str(requested_path),
        )
        if _captured_stat_state(os.fstat(descriptor)) != entry_state:
            raise RuntimeError(
                f"Publication target changed while hashed: {requested_path}"
            )
        if (
            _optional_target_entry_state(
                parent.descriptor,
                requested_path.name,
            )
            != entry_state
        ):
            raise RuntimeError(
                f"Publication target changed while captured: {requested_path}"
            )
        _require_bound_directory_unchanged(parent)
    except BaseException as capture_error:
        _run_cleanup_steps(
            [
                (
                    f"Captured target descriptor cleanup failed for {requested_path}",
                    partial(os.close, descriptor),
                )
            ],
            primary_error=capture_error,
            label="Captured target descriptor cleanup failed",
        )
        raise
    return _CapturedTargetState(
        requested_path=requested_path,
        parent_identity=parent.identity,
        entry_state=entry_state,
        entry_handle=_CapturedTargetHandle(descriptor=descriptor),
        entry_mount_id=entry_mount_id,
        content_sha256=content_sha256,
    )


def _captured_stat_state(metadata: os.stat_result) -> _CapturedStatState:
    """Return mutation-sensitive state without access-time noise."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _optional_target_entry_state(
    parent_descriptor: int,
    entry_name: str,
) -> _CapturedStatState | None:
    """Return one fd-relative no-follow state or ``None`` when absent."""

    try:
        metadata = os.stat(
            entry_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    return _captured_stat_state(metadata)


def _captured_target_content_sha256(
    descriptor: int,
    *,
    entry_state: _CapturedStatState,
    expected_mount_id: int,
    label: str,
) -> str:
    """Hash one captured target's bytes, link payload, or exact tree state."""

    mode = entry_state[2]
    if stat.S_ISREG(mode):
        return _descriptor_sha256(descriptor, label=label)
    if stat.S_ISLNK(mode):
        payload = os.readlink("", dir_fd=descriptor)
        return hashlib.sha256(os.fsencode(payload)).hexdigest()
    if not stat.S_ISDIR(mode):  # pragma: no cover - capture rejects special files
        raise RuntimeError(f"Unsupported captured target type: {label}")
    entries: list[dict[str, str | int | list[int]]] = []
    _collect_captured_target_tree_state(
        descriptor,
        relative_directory=".",
        expected_mount_id=expected_mount_id,
        label=label,
        entries=entries,
    )
    tree_payload = json.dumps(
        {
            "schema_version": _CAPTURED_TARGET_TREE_SCHEMA_VERSION,
            "entries": entries,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(tree_payload).hexdigest()


def _collect_captured_target_tree_state(
    descriptor: int,
    *,
    relative_directory: str,
    expected_mount_id: int,
    label: str,
    entries: list[dict[str, str | int | list[int]]],
    traversal_budget: _ArtifactTreeTraversalBudget | None = None,
    depth: int = 0,
) -> None:
    """Collect content plus physical state for one stable directory tree."""

    if traversal_budget is None:
        traversal_budget = _ArtifactTreeTraversalBudget(
            label=f"Captured target tree {label}"
        )
    traversal_budget.consume(
        relative_path=relative_directory,
        depth=depth,
    )
    if _descriptor_mount_id(descriptor) != expected_mount_id:
        raise ValueError(
            f"Captured target tree {label} crossed a mount at {relative_directory}"
        )
    before = os.fstat(descriptor)
    before_state = _captured_stat_state(before)
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(f"Captured target tree changed type: {label}")
    names = traversal_budget.sorted_child_names(
        descriptor,
        relative_path=relative_directory,
    )
    serialized_state = list(before_state)
    if relative_directory == ".":
        # Moving the captured root into its rollback directory legitimately
        # changes only root ctime. Descendant physical state and all semantic
        # content must remain byte-for-byte identical across that rename.
        serialized_state[-1] = 0
    entries.append(
        {
            "path": relative_directory,
            "type": "directory",
            "state": serialized_state,
        }
    )
    for name in names:
        relative_path = (
            name if relative_directory == "." else f"{relative_directory}/{name}"
        )
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        child_state = _captured_stat_state(metadata)
        if stat.S_ISDIR(metadata.st_mode):
            traversal_budget.require_depth(
                relative_path=relative_path,
                depth=depth + 1,
            )
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            child_descriptor = os.open(name, flags, dir_fd=descriptor)
            try:
                if _captured_stat_state(os.fstat(child_descriptor)) != child_state:
                    raise RuntimeError(
                        f"Captured target tree changed inode: {relative_path}"
                    )
                _collect_captured_target_tree_state(
                    child_descriptor,
                    relative_directory=relative_path,
                    expected_mount_id=expected_mount_id,
                    label=label,
                    entries=entries,
                    traversal_budget=traversal_budget,
                    depth=depth + 1,
                )
            finally:
                os.close(child_descriptor)
            continue
        if stat.S_ISREG(metadata.st_mode):
            flags = os.O_RDONLY | os.O_NOFOLLOW
            flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
            child_descriptor = os.open(name, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child_descriptor)
                if _captured_stat_state(opened) != child_state:
                    raise RuntimeError(
                        f"Captured target tree changed inode: {relative_path}"
                    )
                if _descriptor_mount_id(child_descriptor) != expected_mount_id:
                    raise ValueError(
                        f"Captured target tree {label} crossed a mount at "
                        f"{relative_path}"
                    )
                traversal_budget.consume(
                    relative_path=relative_path,
                    depth=depth + 1,
                    byte_count=opened.st_size,
                )
                content_sha256 = _descriptor_sha256_from_state(
                    child_descriptor,
                    expected_state=opened,
                    label=f"{label} tree file {relative_path}",
                )
                if _captured_stat_state(os.fstat(child_descriptor)) != child_state:
                    raise RuntimeError(
                        f"Captured target tree changed while hashed: {relative_path}"
                    )
            finally:
                os.close(child_descriptor)
            entries.append(
                {
                    "path": relative_path,
                    "type": "file",
                    "state": list(child_state),
                    "sha256": content_sha256,
                }
            )
            continue
        if stat.S_ISLNK(metadata.st_mode):
            traversal_budget.consume(
                relative_path=relative_path,
                depth=depth + 1,
                byte_count=metadata.st_size,
            )
            o_path = getattr(os, "O_PATH", None)
            if o_path is None:  # pragma: no cover - official targets are Linux
                raise RuntimeError("Safe target capture requires Linux O_PATH")
            child_flags = o_path | os.O_NOFOLLOW
            child_flags |= getattr(os, "O_CLOEXEC", 0)
            child_descriptor = os.open(name, child_flags, dir_fd=descriptor)
            try:
                if _captured_stat_state(os.fstat(child_descriptor)) != child_state:
                    raise RuntimeError(
                        f"Captured target symlink changed inode: {relative_path}"
                    )
                if _descriptor_mount_id(child_descriptor) != expected_mount_id:
                    raise ValueError(
                        f"Captured target tree {label} crossed a mount at "
                        f"{relative_path}"
                    )
                link_target = os.readlink("", dir_fd=child_descriptor)
                if _captured_stat_state(os.fstat(child_descriptor)) != child_state:
                    raise RuntimeError(
                        f"Captured target symlink changed while read: {relative_path}"
                    )
            finally:
                os.close(child_descriptor)
            entries.append(
                {
                    "path": relative_path,
                    "type": "symlink",
                    "state": list(child_state),
                    "sha256": hashlib.sha256(os.fsencode(link_target)).hexdigest(),
                }
            )
            continue
        traversal_budget.consume(
            relative_path=relative_path,
            depth=depth + 1,
            byte_count=metadata.st_size,
        )
        raise ValueError(
            f"Captured target tree contains a special file: {relative_path}"
        )
    after_state = _captured_stat_state(os.fstat(descriptor))
    after_names = _bounded_sorted_directory_names(
        descriptor,
        maximum_names=len(names),
        overflow_message=(
            f"Captured target tree changed while hashed: {relative_directory}"
        ),
    )
    if after_state != before_state or after_names != names:
        raise RuntimeError(
            f"Captured target tree changed while hashed: {relative_directory}"
        )


def _initial_target_state(
    artifacts: StagedJointRiggerArtifacts,
    target_path: Path,
) -> _CapturedTargetState | None:
    """Return captured state for a generated bundle or legacy-untracked None."""

    requested_path = _absolute_lexical_path(target_path)
    for target_state in artifacts._initial_target_states:
        if target_state.requested_path == requested_path:
            return target_state
    if artifacts._initial_target_states:
        raise RuntimeError(
            f"Initial publication target state is missing for {requested_path}"
        )
    return None


def staged_promotion_artifacts(
    artifacts: StagedJointRiggerArtifacts,
) -> list[StagedArtifact]:
    """Validate a complete staged set and return root-last promotion order."""
    staged = artifacts.staged_targets
    final = artifacts.final_targets
    _require_staging_owner_unchanged(artifacts, staged.diagnostics_path)
    diagnostics_metadata = _require_regular_file(
        staged.diagnostics_path,
        "diagnostics report",
    )
    _bind_staging_cleanup_identity(
        artifacts,
        staged.diagnostics_path,
        diagnostics_metadata,
    )
    _require_staging_owner_unchanged(artifacts, staged.result_path)
    result_metadata = _require_regular_file(staged.result_path, "result report")
    _bind_staging_cleanup_identity(
        artifacts,
        staged.result_path,
        result_metadata,
    )
    _require_staging_owner_unchanged(artifacts, staged.output_path)
    output_metadata = _require_regular_file(staged.output_path, "generated root")
    _bind_staging_cleanup_identity(
        artifacts,
        staged.output_path,
        output_metadata,
    )
    output_promotion_state = _staging_promotion_state(
        artifacts,
        staged.output_path,
    )

    promotion = [
        StagedArtifact(
            staged_path=staged.diagnostics_path,
            target_path=final.diagnostics_path,
            label="diagnostics report",
            _initial_target_state=_initial_target_state(
                artifacts,
                final.diagnostics_path,
            ),
            _promotion_state=_staging_promotion_state(
                artifacts,
                staged.diagnostics_path,
            ),
        ),
        StagedArtifact(
            staged_path=staged.result_path,
            target_path=final.result_path,
            label="result report",
            _initial_target_state=_initial_target_state(
                artifacts,
                final.result_path,
            ),
            _promotion_state=_staging_promotion_state(
                artifacts,
                staged.result_path,
            ),
        ),
    ]
    if final.sidecar_path is not None:
        assert staged.sidecar_path is not None
        _require_staging_owner_unchanged(artifacts, staged.sidecar_path)
        sidecar_metadata = os.stat(staged.sidecar_path, follow_symlinks=False)
        if stat.S_ISLNK(sidecar_metadata.st_mode) or not stat.S_ISDIR(
            sidecar_metadata.st_mode
        ):
            raise RuntimeError(
                f"Staged composition sidecar is missing or invalid: {staged.sidecar_path}"
            )
        with _bound_sidecar_tree(
            staged.sidecar_path,
            label="Staged composition sidecar",
        ):
            pass
        _bind_staging_cleanup_identity(
            artifacts,
            staged.sidecar_path,
            sidecar_metadata,
            expects_directory=True,
        )
        sidecar_reservation, owner_payload = _find_staging_cleanup_reservation(
            artifacts,
            staged.sidecar_path,
        )
        if (
            not owner_payload
            or sidecar_reservation.payload_descriptor < 0
            or sidecar_reservation.payload_promotion_state is None
            or sidecar_reservation.payload_promotion_state.source_tree_sha256 is None
        ):
            raise RuntimeError(
                "Staged composition sidecar lacks a sealed descriptor source"
            )
        promotion.append(
            StagedArtifact(
                staged_path=staged.sidecar_path,
                target_path=final.sidecar_path,
                label="composition sidecar",
                source_descriptor=sidecar_reservation.payload_descriptor,
                source_sha256=(
                    sidecar_reservation.payload_promotion_state.source_tree_sha256
                ),
                _initial_target_state=_initial_target_state(
                    artifacts,
                    final.sidecar_path,
                ),
                _promotion_state=sidecar_reservation.payload_promotion_state,
            )
        )
    promotion.append(
        StagedArtifact(
            staged_path=staged.output_path,
            target_path=final.output_path,
            label="generated root",
            _initial_target_state=_initial_target_state(
                artifacts,
                final.output_path,
            ),
            _promotion_state=output_promotion_state,
        )
    )
    return promotion


def _require_staging_owner_unchanged(
    artifacts: StagedJointRiggerArtifacts,
    path: Path,
) -> None:
    """Require a staged path to remain beneath its descriptor-bound owner."""

    expected_owner = Path(os.path.abspath(path.parent.expanduser()))
    for reservation in artifacts._cleanup_reservations:
        reservation_parent = Path(
            os.path.abspath(reservation.parent.locator_path.expanduser())
        )
        if reservation_parent == expected_owner and reservation.name == path.name:
            _require_bound_directory_unchanged(reservation.parent)
            return
        reservation_owner = Path(
            os.path.abspath(
                (reservation.parent.locator_path / reservation.name).expanduser()
            )
        )
        if reservation_owner != expected_owner:
            continue
        identity = reservation.owned_identity
        descriptor = reservation.owned_descriptor
        opened = None if descriptor < 0 else os.fstat(descriptor)
        if (
            identity is None
            or opened is None
            or not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != identity
            or _optional_bound_entry_identity(
                _BoundEntry(parent=reservation.parent, name=reservation.name)
            )
            != identity
        ):
            raise RuntimeError(
                f"Staging owner changed inode before validation: {expected_owner}"
            )
        _require_bound_directory_unchanged(reservation.parent)
        return
    raise RuntimeError(f"Missing cleanup reservation for staged artifact: {path}")


def _bind_staging_cleanup_identity(
    artifacts: StagedJointRiggerArtifacts,
    path: Path,
    metadata: os.stat_result,
    *,
    expects_directory: bool = False,
) -> None:
    """Bind one validated staging payload through its retained parent."""

    reservation, owner_payload = _find_staging_cleanup_reservation(artifacts, path)
    if reservation.binding_revoked or reservation.closed:
        raise RuntimeError("Staging cleanup ownership binding has been revoked")
    if owner_payload:
        reservation.payload_binding_attempted = True
        parent_descriptor = reservation.owned_descriptor
        parent_metadata = os.fstat(parent_descriptor)
        if (
            reservation.owned_identity is None
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or (parent_metadata.st_dev, parent_metadata.st_ino)
            != reservation.owned_identity
        ):
            raise RuntimeError(
                f"Staging owner changed before payload binding: {path.parent}"
            )
        entry_name = path.name
        owned_identity = reservation.payload_identity
        owned_descriptor = reservation.payload_descriptor
        promotion_state = reservation.payload_promotion_state
        source_parent_identity = reservation.owned_identity
    else:
        parent_descriptor = reservation.parent.descriptor
        entry_name = reservation.name
        owned_identity = reservation.owned_identity
        owned_descriptor = reservation.owned_descriptor
        promotion_state = reservation.promotion_state
        source_parent_identity = reservation.parent.identity

    expected_type = stat.S_ISDIR if expects_directory else stat.S_ISREG
    if owned_identity is not None or owned_descriptor >= 0:
        if owned_identity is None or owned_descriptor < 0:
            raise RuntimeError(
                "Staged artifact has incomplete descriptor-bound cleanup ownership"
            )
        opened = os.fstat(owned_descriptor)
        held_identity = (opened.st_dev, opened.st_ino)
        if (
            not expected_type(opened.st_mode)
            or held_identity != owned_identity
            or held_identity != (metadata.st_dev, metadata.st_ino)
            or _optional_descriptor_entry_identity(parent_descriptor, entry_name)
            != held_identity
        ):
            raise RuntimeError(
                "Staged artifact changed inode after cleanup ownership was "
                f"bound: {path}"
            )
        _require_staging_owner_unchanged(artifacts, path)
        source_tree_sha256 = None
        source_mount_id = None
        if expects_directory:
            source_tree_sha256, source_mount_id = _seal_and_hash_staging_directory(
                owned_descriptor,
                parent_descriptor=parent_descriptor,
                entry_name=entry_name,
                expected_identity=held_identity,
                label=str(path),
            )
        _bind_staging_promotion_source(
            promotion_state,
            source_identity=held_identity,
            source_parent_identity=source_parent_identity,
            source_descriptor=owned_descriptor,
            source_is_directory=expects_directory,
            source_tree_sha256=source_tree_sha256,
            source_mount_id=source_mount_id,
            path=path,
        )
        return

    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    if expects_directory:
        flags |= os.O_DIRECTORY
    descriptor = -1
    binding_error: BaseException | None = None
    try:
        descriptor = os.open(
            entry_name,
            flags,
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        descriptor_identity = (opened.st_dev, opened.st_ino)
        if (
            not expected_type(opened.st_mode)
            or descriptor_identity != (metadata.st_dev, metadata.st_ino)
            or _optional_descriptor_entry_identity(parent_descriptor, entry_name)
            != descriptor_identity
        ):
            raise RuntimeError(
                "Staged artifact changed inode while cleanup ownership was "
                f"bound: {path}"
            )
        _require_staging_owner_unchanged(artifacts, path)
        source_tree_sha256 = None
        source_mount_id = None
        if expects_directory:
            source_tree_sha256, source_mount_id = _seal_and_hash_staging_directory(
                descriptor,
                parent_descriptor=parent_descriptor,
                entry_name=entry_name,
                expected_identity=descriptor_identity,
                label=str(path),
            )
        _bind_staging_promotion_source(
            promotion_state,
            source_identity=descriptor_identity,
            source_parent_identity=source_parent_identity,
            source_descriptor=descriptor,
            source_is_directory=expects_directory,
            source_tree_sha256=source_tree_sha256,
            source_mount_id=source_mount_id,
            path=path,
        )
        if owner_payload:
            reservation.payload_identity = descriptor_identity
            reservation.payload_descriptor = descriptor
            reservation.payload_is_directory = expects_directory
        else:
            reservation.owned_identity = descriptor_identity
            reservation.owned_descriptor = descriptor
        descriptor = -1
    except BaseException as error:
        binding_error = error
        raise
    finally:
        if descriptor >= 0:
            owned_descriptor = descriptor
            descriptor = -1
            _run_cleanup_steps(
                [
                    (
                        f"Staged artifact binding descriptor cleanup failed for {path}",
                        partial(os.close, owned_descriptor),
                    )
                ],
                primary_error=binding_error,
                label="Staged artifact binding descriptor cleanup failed",
            )


def _seal_and_hash_staging_directory(
    descriptor: int,
    *,
    parent_descriptor: int,
    entry_name: str,
    expected_identity: tuple[int, int],
    label: str,
) -> tuple[str, int]:
    """Reject mounted staged content before sealing and hashing its exact tree."""

    expected_mount_id = _descriptor_mount_id(parent_descriptor)
    _require_descriptor_entry_mount_id(
        parent_descriptor,
        entry_name,
        expected_identity=expected_identity,
        expected_mount_id=expected_mount_id,
        label=f"Staged {label}",
    )
    if _descriptor_mount_id(descriptor) != expected_mount_id:
        raise ValueError(f"Staged {label} root is a mount point")
    _require_directory_tree_mount_id(
        descriptor,
        expected_mount_id=expected_mount_id,
        label=f"Staged {label}",
    )
    _require_sidecar_tree_entries(
        descriptor,
        expected_mount_id=expected_mount_id,
        label="Staged composition sidecar",
    )
    _seal_staging_directory_descriptor_tree(
        descriptor,
        expected_mount_id=expected_mount_id,
        label=label,
    )
    return (
        _directory_descriptor_tree_sha256(
            descriptor,
            expected_mount_id=expected_mount_id,
            label=label,
            require_no_write_bits=True,
        ),
        expected_mount_id,
    )


def _bind_staging_promotion_source(
    state: _StagingPromotionState | None,
    *,
    source_identity: tuple[int, int],
    source_parent_identity: tuple[int, int] | None,
    source_descriptor: int,
    source_is_directory: bool,
    source_tree_sha256: str | None,
    source_mount_id: int | None,
    path: Path,
) -> None:
    """Bind move authority to the exact validated payload and owner inodes."""

    if state is None or source_parent_identity is None:
        raise RuntimeError(f"Missing promotion state for staged artifact: {path}")
    if (
        state.source_identity not in (None, source_identity)
        or (state.source_parent_identity not in (None, source_parent_identity))
        or state.source_descriptor not in (-1, source_descriptor)
        or (
            state.source_identity is not None
            and state.source_is_directory != source_is_directory
        )
        or state.source_tree_sha256 not in (None, source_tree_sha256)
        or state.source_mount_id not in (None, source_mount_id)
    ):
        raise RuntimeError(
            f"Staged artifact promotion authority changed identity: {path}"
        )
    state.source_identity = source_identity
    state.source_parent_identity = source_parent_identity
    state.source_descriptor = source_descriptor
    state.source_is_directory = source_is_directory
    state.source_tree_sha256 = source_tree_sha256
    state.source_mount_id = source_mount_id


def _find_staging_cleanup_reservation(
    artifacts: StagedJointRiggerArtifacts,
    path: Path,
) -> tuple[_StagingCleanupReservation, bool]:
    """Find the reservation that owns one sibling or owner-child payload."""

    expected_parent = Path(os.path.abspath(path.parent.expanduser()))
    for reservation in artifacts._cleanup_reservations:
        reservation_parent = Path(
            os.path.abspath(reservation.parent.locator_path.expanduser())
        )
        if (
            not reservation.is_owner_directory
            and reservation_parent == expected_parent
            and reservation.name == path.name
        ):
            return reservation, False
        reservation_owner = Path(
            os.path.abspath(
                (reservation.parent.locator_path / reservation.name).expanduser()
            )
        )
        if (
            reservation.is_owner_directory
            and reservation_owner == expected_parent
            and reservation.payload_name == path.name
        ):
            return reservation, True
    raise RuntimeError(f"Missing cleanup reservation for staged artifact: {path}")


def _staging_promotion_state(
    artifacts: StagedJointRiggerArtifacts,
    path: Path,
) -> _StagingPromotionState:
    """Return identity-bound move evidence for one staged payload."""

    reservation, owner_payload = _find_staging_cleanup_reservation(artifacts, path)
    state = (
        reservation.payload_promotion_state
        if owner_payload
        else reservation.promotion_state
    )
    if state is None:
        raise RuntimeError(f"Missing promotion state for staged artifact: {path}")
    return state


def sidecar_dependency_bundle_sha256(sidecar_path: str | Path) -> str:
    """Hash one sidecar tree as a deterministic regular-file manifest.

    Relative POSIX paths, byte sizes, and content SHA-256 values are sorted by
    path and serialized canonically. Empty directories do not affect identity.
    Symlinks, multiply linked regular files, and special files are rejected so
    the digest never depends on resources outside the staged sidecar or on
    platform-specific file types.
    """
    root = Path(sidecar_path)
    try:
        root_mode = root.lstat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Composition sidecar does not exist: {root}") from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ValueError(f"Composition sidecar must be a non-symlink directory: {root}")

    tree_entries: list[dict[str, str | int]] = []
    with _bound_sidecar_tree(root, label="Composition sidecar") as (
        descriptor,
        mount_id,
    ):
        _collect_directory_tree_entries(
            descriptor,
            relative_directory=".",
            label="composition sidecar",
            require_no_write_bits=False,
            expected_mount_id=mount_id,
            entries=tree_entries,
        )
    entries = [
        {
            "path": entry["path"],
            "size": entry["size"],
            "sha256": entry["sha256"],
        }
        for entry in tree_entries
        if entry["type"] == "file"
    ]

    payload = json.dumps(
        {
            "schema_version": _SIDECAR_BUNDLE_SCHEMA_VERSION,
            "files": entries,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sidecar_tree_entries(
    descriptor: int,
    *,
    expected_mount_id: int,
    label: str,
    relative_directory: str = ".",
    traversal_budget: _ArtifactTreeTraversalBudget | None = None,
    depth: int = 0,
) -> None:
    """Reject mounted, linked, or special sidecar entries without reading bytes."""

    if traversal_budget is None:
        traversal_budget = _ArtifactTreeTraversalBudget(label=label)
    traversal_budget.consume(
        relative_path=relative_directory,
        depth=depth,
    )
    if _descriptor_mount_id(descriptor) != expected_mount_id:
        raise ValueError(f"{label} contains a mount point at {relative_directory}")
    names = traversal_budget.sorted_child_names(
        descriptor,
        relative_path=relative_directory,
    )
    for name in names:
        relative_path = (
            name if relative_directory == "." else f"{relative_directory}/{name}"
        )
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            traversal_budget.require_depth(
                relative_path=relative_path,
                depth=depth + 1,
            )
        else:
            traversal_budget.consume(
                relative_path=relative_path,
                depth=depth + 1,
                byte_count=metadata.st_size,
            )
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink: {relative_path}")
        if stat.S_ISDIR(metadata.st_mode):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            child_descriptor = -1
            operation_error: BaseException | None = None
            try:
                child_descriptor = os.open(name, flags, dir_fd=descriptor)
                opened = os.fstat(child_descriptor)
                if (opened.st_dev, opened.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise RuntimeError(f"{label} changed inode: {relative_path}")
                if _descriptor_mount_id(child_descriptor) != expected_mount_id:
                    raise ValueError(
                        f"{label} contains a mount point at {relative_path}"
                    )
                _require_sidecar_tree_entries(
                    child_descriptor,
                    expected_mount_id=expected_mount_id,
                    label=label,
                    relative_directory=relative_path,
                    traversal_budget=traversal_budget,
                    depth=depth + 1,
                )
            except BaseException as error:
                operation_error = error
                raise
            finally:
                if child_descriptor >= 0:
                    _run_cleanup_steps(
                        [
                            (
                                f"{label} directory descriptor close failed for {relative_path}",
                                partial(os.close, child_descriptor),
                            )
                        ],
                        primary_error=operation_error,
                        label=f"{label} descriptor cleanup failed",
                    )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} contains a special file: {relative_path}")
        if metadata.st_nlink != 1:
            raise ValueError(
                f"{label} regular file must have exactly one hard link: {relative_path}"
            )
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        file_descriptor = -1
        operation_error = None
        try:
            file_descriptor = os.open(name, flags, dir_fd=descriptor)
            opened = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
                or opened.st_nlink != 1
            ):
                raise ValueError(f"{label} regular file changed: {relative_path}")
            if _descriptor_mount_id(file_descriptor) != expected_mount_id:
                raise ValueError(f"{label} contains a mount point at {relative_path}")
        except BaseException as error:
            operation_error = error
            raise
        finally:
            if file_descriptor >= 0:
                _run_cleanup_steps(
                    [
                        (
                            f"{label} file descriptor close failed for {relative_path}",
                            partial(os.close, file_descriptor),
                        )
                    ],
                    primary_error=operation_error,
                    label=f"{label} descriptor cleanup failed",
                )


@contextmanager
def _bound_sidecar_tree(
    sidecar_path: Path,
    *,
    label: str,
) -> Iterator[tuple[int, int]]:
    """Hold one sidecar root and parent while rejecting every mount boundary."""

    root = _absolute_lexical_path(sidecar_path)
    parent = _open_bound_directory(root.parent)
    descriptor = -1
    operation_error: BaseException | None = None
    try:
        metadata = os.stat(root.name, dir_fd=parent.descriptor, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} must be a non-symlink directory: {root}")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(root.name, flags, dir_fd=parent.descriptor)
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or identity != (metadata.st_dev, metadata.st_ino)
            or _optional_descriptor_entry_identity(parent.descriptor, root.name)
            != identity
        ):
            raise RuntimeError(f"{label} changed inode while it was opened: {root}")
        mount_id = _descriptor_mount_id(parent.descriptor)
        if _descriptor_mount_id(descriptor) != mount_id:
            raise ValueError(f"{label} root is a mount point: {root}")
        _require_directory_tree_mount_id(
            descriptor,
            expected_mount_id=mount_id,
            label=label,
        )
        _require_sidecar_tree_entries(
            descriptor,
            expected_mount_id=mount_id,
            label=label,
        )
        yield descriptor, mount_id
        _require_descriptor_entry_mount_id(
            parent.descriptor,
            root.name,
            expected_identity=identity,
            expected_mount_id=mount_id,
            label=label,
        )
        _require_directory_tree_mount_id(
            descriptor,
            expected_mount_id=mount_id,
            label=label,
        )
        _require_sidecar_tree_entries(
            descriptor,
            expected_mount_id=mount_id,
            label=label,
        )
    except BaseException as error:
        operation_error = error
        raise
    finally:
        cleanup_steps: list[tuple[str, Callable[[], None]]] = []
        if descriptor >= 0:
            cleanup_steps.append(
                (
                    f"{label} root descriptor cleanup failed",
                    partial(os.close, descriptor),
                )
            )
        cleanup_steps.append(
            (
                f"{label} parent descriptor cleanup failed",
                partial(os.close, parent.descriptor),
            )
        )
        _run_cleanup_steps(
            cleanup_steps,
            primary_error=operation_error,
            label=f"{label} descriptor cleanup failed",
        )


def copy_sidecar_directory(
    source_path: str | Path,
    target_descriptor: int,
    *,
    label: str,
    preserve_modes: bool = True,
) -> None:
    """Copy one path-bound sidecar without traversing a root or nested mount."""

    if (
        isinstance(target_descriptor, bool)
        or not isinstance(target_descriptor, int)
        or target_descriptor < 0
        or not stat.S_ISDIR(os.fstat(target_descriptor).st_mode)
    ):
        raise ValueError("target_descriptor must identify a directory")
    with _bound_sidecar_tree(Path(source_path), label=label) as (
        source_descriptor,
        mount_id,
    ):
        _copy_directory_descriptor_tree(
            source_descriptor,
            target_descriptor,
            label=label,
            preserve_modes=preserve_modes,
            expected_source_mount_id=mount_id,
            require_no_write_bits=False,
        )


def _seal_staging_directory_descriptor_tree(
    descriptor: int,
    *,
    expected_mount_id: int,
    label: str,
    relative_directory: str = ".",
    traversal_budget: _ArtifactTreeTraversalBudget | None = None,
    depth: int = 0,
) -> None:
    """Seal one direct-move sidecar tree through retained no-follow fds."""

    root_traversal = traversal_budget is None
    if root_traversal:
        traversal_budget = _ArtifactTreeTraversalBudget(label=f"Staged {label}")
    traversal_budget = _require_present_invariant(
        traversal_budget,
        label="staging sidecar seal traversal budget",
    )
    traversal_budget.consume(
        relative_path=relative_directory,
        depth=depth,
    )
    if root_traversal:
        _require_directory_tree_mount_id(
            descriptor,
            expected_mount_id=expected_mount_id,
            label=f"Staged {label}",
            relative_path=relative_directory,
        )
    root_metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError(f"Staged {label} payload is not a directory")
    names = traversal_budget.sorted_child_names(
        descriptor,
        relative_path=relative_directory,
    )
    os.fchmod(descriptor, stat.S_IMODE(root_metadata.st_mode) & ~0o222)
    for name in names:
        relative_path = (
            name if relative_directory == "." else f"{relative_directory}/{name}"
        )
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            traversal_budget.require_depth(
                relative_path=relative_path,
                depth=depth + 1,
            )
        else:
            traversal_budget.consume(
                relative_path=relative_path,
                depth=depth + 1,
                byte_count=metadata.st_size,
            )
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(
                f"Staged {label} tree contains a symlink: {relative_path}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            child_descriptor = -1
            operation_error: BaseException | None = None
            try:
                child_descriptor = os.open(name, flags, dir_fd=descriptor)
                opened = os.fstat(child_descriptor)
                if (opened.st_dev, opened.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise RuntimeError(
                        f"Staged {label} tree changed inode: {relative_path}"
                    )
                if _descriptor_mount_id(child_descriptor) != expected_mount_id:
                    raise ValueError(
                        f"Staged {label} contains a mount point at {relative_path}"
                    )
                _seal_staging_directory_descriptor_tree(
                    child_descriptor,
                    expected_mount_id=expected_mount_id,
                    label=label,
                    relative_directory=relative_path,
                    traversal_budget=traversal_budget,
                    depth=depth + 1,
                )
            except BaseException as error:
                operation_error = error
                raise
            finally:
                if child_descriptor >= 0:
                    owned_child_descriptor = child_descriptor
                    child_descriptor = -1
                    _run_cleanup_steps(
                        [
                            (
                                "Staging sidecar directory descriptor cleanup "
                                f"failed for {relative_path}",
                                partial(os.close, owned_child_descriptor),
                            )
                        ],
                        primary_error=operation_error,
                        label="Staging sidecar descriptor cleanup failed",
                    )
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(
                f"Staged {label} tree has an invalid file: {relative_path}"
            )
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        file_descriptor = -1
        operation_error = None
        try:
            file_descriptor = os.open(name, flags, dir_fd=descriptor)
            opened = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
                or opened.st_nlink != 1
            ):
                raise RuntimeError(
                    f"Staged {label} tree changed inode: {relative_path}"
                )
            if _descriptor_mount_id(file_descriptor) != expected_mount_id:
                raise ValueError(
                    f"Staged {label} contains a mount point at {relative_path}"
                )
            os.fchmod(file_descriptor, stat.S_IMODE(opened.st_mode) & ~0o222)
        except BaseException as error:
            operation_error = error
            raise
        finally:
            if file_descriptor >= 0:
                owned_file_descriptor = file_descriptor
                file_descriptor = -1
                _run_cleanup_steps(
                    [
                        (
                            "Staging sidecar file descriptor cleanup failed for "
                            f"{relative_path}",
                            partial(os.close, owned_file_descriptor),
                        )
                    ],
                    primary_error=operation_error,
                    label="Staging sidecar descriptor cleanup failed",
                )


def directory_tree_sha256(directory_path: str | Path) -> str:
    """Hash an exact non-symlink directory tree, including empty directories."""

    path = Path(directory_path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    operation_error: BaseException | None = None
    try:
        descriptor = os.open(path, flags)
        return _directory_descriptor_tree_sha256(
            descriptor,
            label=str(path),
            require_no_write_bits=False,
        )
    except BaseException as error:
        operation_error = error
        raise
    finally:
        if descriptor >= 0:
            owned_descriptor = descriptor
            descriptor = -1
            _run_cleanup_steps(
                [
                    (
                        "Directory tree descriptor close failed",
                        partial(os.close, owned_descriptor),
                    )
                ],
                primary_error=operation_error,
                label="Directory tree descriptor cleanup failed",
            )


def directory_descriptor_tree_sha256(descriptor: int) -> str:
    """Hash an exact tree through one caller-owned read-only directory fd."""

    if (
        isinstance(descriptor, bool)
        or not isinstance(descriptor, int)
        or descriptor < 0
    ):
        raise ValueError("descriptor must be a non-negative directory descriptor")
    return _directory_descriptor_tree_sha256(
        descriptor,
        label="directory descriptor",
        require_no_write_bits=False,
    )


def copy_directory_descriptor_tree(
    source_descriptor: int,
    target_descriptor: int,
    *,
    label: str,
    preserve_modes: bool = True,
    expected_source_mount_id: int | None = None,
    require_no_write_bits: bool = True,
) -> None:
    """Copy one stable tree between caller-owned directory descriptors."""

    for descriptor, descriptor_label in (
        (source_descriptor, "source_descriptor"),
        (target_descriptor, "target_descriptor"),
    ):
        if (
            isinstance(descriptor, bool)
            or not isinstance(descriptor, int)
            or descriptor < 0
        ):
            raise ValueError(f"{descriptor_label} must be a non-negative descriptor")
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"{descriptor_label} must identify a directory")
    _copy_directory_descriptor_tree(
        source_descriptor,
        target_descriptor,
        label=label,
        preserve_modes=preserve_modes,
        expected_source_mount_id=expected_source_mount_id,
        require_no_write_bits=require_no_write_bits,
    )


def promote_staged_artifacts(
    artifacts: list[StagedArtifact],
    *,
    prebackup_validator: Callable[[], None] | None = None,
    precommit_validator: Callable[[], None] | None = None,
) -> None:
    """Promote one logical artifact set and roll back every partial move.

    A nonblocking advisory lock is acquired on every distinct physical target
    parent before any existing artifact is moved. Concurrent transactions under
    one parent therefore fail before publication instead of interleaving, while
    disjoint physical parents remain concurrent. Locks are attached directly to
    held directory descriptors and released on close or process termination;
    publication creates no filesystem lock entry. Once staged artifacts are
    bound and their parents validated, ``prebackup_validator`` is invoked
    exactly once under those locks, before any target backup or staged-content
    move. Generated staging bundles also require every locked target to retain
    the parent and absent-or-present entry identity captured before backend
    authoring. Bound parents and captured targets are revalidated after the hook
    returns, and the hook is not invoked when lock acquisition or staged binding
    fails. If execution reaches the commit gate, ``precommit_validator`` is
    invoked exactly once under those locks, after descriptor-backed evidence and
    the private root copy have been validated and before the final root rename.
    It is never invoked after that rename. Successful regular-file publication
    consumes its staged name; descriptor-backed directory sources remain
    caller-owned after success.
    """
    seen_targets: list[tuple[Path, Path]] = []
    for artifact in artifacts:
        target = artifact.target_path.expanduser().resolve(strict=False)
        for seen_target, seen_original in seen_targets:
            if target == seen_target:
                raise ValueError(
                    f"Duplicate transaction target: {artifact.target_path}"
                )
            if _is_relative_to(target, seen_target) or _is_relative_to(
                seen_target, target
            ):
                raise ValueError(
                    "Nested transaction targets are not supported: "
                    f"{artifact.target_path} overlaps {seen_original}"
                )
        seen_targets.append((target, artifact.target_path))
        if not artifact.staged_path.exists() and not artifact.staged_path.is_symlink():
            raise FileNotFoundError(
                f"Staged {artifact.label} is missing: {artifact.staged_path}"
            )
    for artifact in artifacts:
        artifact.target_path.parent.mkdir(parents=True, exist_ok=True)

    cleanup_state = _PublicationCleanupState()
    try:
        with _publication_target_locks(
            (artifact.target_path for artifact in artifacts),
            cleanup_state=cleanup_state,
        ) as locked_targets:
            with _bound_staged_artifacts(
                artifacts,
                locked_targets,
                cleanup_state=cleanup_state,
            ) as bound_artifacts:
                _promote_staged_artifacts_locked(
                    bound_artifacts,
                    cleanup_state,
                    prebackup_validator=prebackup_validator,
                    precommit_validator=precommit_validator,
                )
    except Exception as exc:
        if not cleanup_state.committed:
            raise
        cleanup_state.errors.append(exc)
    if cleanup_state.errors:
        raise CommittedArtifactPublicationCleanupError(cleanup_state.errors)


def _promote_staged_artifacts_locked(
    artifacts: list[_BoundArtifact],
    cleanup_state: _PublicationCleanupState,
    *,
    prebackup_validator: Callable[[], None] | None,
    precommit_validator: Callable[[], None] | None,
) -> None:
    """Run one replacement transaction while every final target is locked."""
    backups: list[_ArtifactBackup] = []
    promoted: list[_BoundArtifact] = []
    promoted_identities: dict[int, tuple[int, int]] = {}
    detached_targets: dict[int, _DetachedTarget] = {}

    def require_precommit_state() -> None:
        _require_artifact_backups_unchanged(backups)
        _require_promoted_target_identities(promoted, promoted_identities)
        _require_precommit_descriptor_artifacts(
            promoted,
            detached_targets,
        )
        if precommit_validator is not None:
            precommit_validator()
            _require_promoted_target_identities(promoted, promoted_identities)
            _require_precommit_descriptor_artifacts(
                promoted,
                detached_targets,
            )
            _require_artifact_backups_unchanged(backups)
        _require_bound_artifact_parents_unchanged(artifacts)

    def record_completed_promotion(
        bound_artifact: _BoundArtifact,
        identity: tuple[int, int],
        *,
        is_commit_point: bool,
    ) -> None:
        if (
            bound_artifact.artifact._promotion_state is not None
            and bound_artifact.descriptor_source is None
        ):
            bound_artifact.artifact._promotion_state.committed_identity = identity
        if is_commit_point:
            cleanup_state.committed = True
            return
        promoted_identities[id(bound_artifact)] = identity
        promoted.append(bound_artifact)

    def replace_and_record_promotion(
        source: _BoundEntry,
        bound_artifact: _BoundArtifact,
        identity: tuple[int, int],
        *,
        is_commit_point: bool,
    ) -> None:
        """Track a rename even when interruption arrives after the syscall."""

        target = bound_artifact.target_entry
        try:
            _replace_entry(source, target)
            source_after = _optional_bound_entry_identity(source)
            target_after = _optional_bound_entry_identity(target)
            if source_after is not None or target_after != identity:
                raise RuntimeError(
                    "Artifact promotion changed inode after rename; refusing "
                    f"commit evidence for {bound_artifact.artifact.label}: "
                    f"source={source_after}, target={target_after}, expected={identity}"
                )
            _require_retained_staging_promotion_source(bound_artifact.artifact)
            if is_commit_point:
                _require_artifact_backups_unchanged(backups)
            record_completed_promotion(
                bound_artifact,
                identity,
                is_commit_point=is_commit_point,
            )
        except BaseException:
            moved = (
                _optional_bound_entry_identity(source) is None
                and _optional_bound_entry_identity(target) == identity
            )
            if moved:
                try:
                    _require_retained_staging_promotion_source(bound_artifact.artifact)
                except BaseException:
                    promoted_identities[id(bound_artifact)] = identity
                    if not any(item is bound_artifact for item in promoted):
                        promoted.append(bound_artifact)
                    raise
                if is_commit_point:
                    try:
                        _require_artifact_backups_unchanged(backups)
                    except BaseException:
                        promoted_identities[id(bound_artifact)] = identity
                        if not any(item is bound_artifact for item in promoted):
                            promoted.append(bound_artifact)
                        raise
                    record_completed_promotion(
                        bound_artifact,
                        identity,
                        is_commit_point=True,
                    )
                else:
                    record_completed_promotion(
                        bound_artifact,
                        identity,
                        is_commit_point=False,
                    )
            raise

    active_error: BaseException | None = None
    try:
        try:
            if _RENAMEAT2 is None:
                raise RuntimeError(
                    "Atomic no-replace artifact publication requires Linux renameat2"
                )
            _require_bound_artifact_parents_unchanged(artifacts)
            _require_initial_target_states(artifacts)
            if prebackup_validator is not None:
                prebackup_validator()
                _require_bound_artifact_parents_unchanged(artifacts)
                _require_initial_target_states(artifacts)
            _require_existing_target_mount_boundaries(artifacts)
            # Invalidate an existing root commit point before moving its evidence
            # away. Reversing promotion order also makes rollback restore the root
            # last, after its reports and optional sidecar.
            for bound_artifact in reversed(artifacts):
                target = bound_artifact.target_entry
                target_identity = _require_initial_target_state(bound_artifact)
                if target_identity is None:
                    continue
                if not bound_artifact.artifact.replace_existing:
                    raise FileExistsError(
                        f"Artifact target was expected to remain absent: {target.path}"
                    )
                backup = _create_artifact_backup(
                    bound_artifact,
                    artifact_identity=target_identity,
                )
                try:
                    _replace_entry_with_directory_mode_guard(
                        target,
                        backup.artifact_entry,
                        expected_identity=target_identity,
                        label=(
                            "published artifact backup move for "
                            f"{bound_artifact.artifact.label}"
                        ),
                    )
                except BaseException as move_error:
                    if any(item is backup for item in backups):
                        raise
                    backup_identity = _optional_bound_entry_identity(
                        backup.artifact_entry
                    )
                    target_after_move = _optional_bound_entry_identity(target)
                    if backup_identity == target_identity and target_after_move is None:
                        backups.append(backup)
                        raise
                    if not (
                        backup_identity is None and target_after_move == target_identity
                    ):
                        backups.append(backup)
                        raise RuntimeError(
                            "Artifact backup move was interrupted in an ambiguous state: "
                            f"{target.path}"
                        ) from move_error
                    _run_cleanup_steps(
                        (
                            (
                                "Unmoved artifact backup cleanup also failed",
                                partial(_remove_backup_directory, backup),
                            ),
                            (
                                "Unmoved artifact backup descriptor close also failed",
                                partial(os.close, backup.directory.descriptor),
                            ),
                        ),
                        primary_error=move_error,
                        label="Unmoved artifact backup cleanup failed",
                    )
                    raise
                else:
                    # A successful return is not enough: an untrusted sibling can
                    # exchange either name immediately around renameat2.  Record
                    # the backup first so any mismatch is preserved for recovery,
                    # then require the exact target inode to be under the held
                    # rollback-directory descriptor and the source name absent.
                    backups.append(backup)
                    _require_artifact_backup_ready(
                        backup,
                        operation="after target-to-backup move",
                    )
                    _capture_artifact_backup_state(backup)

            for artifact_index, bound_artifact in enumerate(artifacts):
                is_commit_point = artifact_index == len(artifacts) - 1
                source = bound_artifact.descriptor_source
                if source is None:
                    promoted_identity = _require_staging_promotion_source(
                        bound_artifact.artifact,
                        bound_artifact.staged_entry,
                    )
                    if is_commit_point:
                        require_precommit_state()
                        promoted_identity = _require_staging_promotion_source(
                            bound_artifact.artifact,
                            bound_artifact.staged_entry,
                        )
                    replace_and_record_promotion(
                        bound_artifact.staged_entry,
                        bound_artifact,
                        promoted_identity,
                        is_commit_point=is_commit_point,
                    )
                    if is_commit_point:
                        _require_bound_artifact_parents_unchanged(artifacts)
                elif source.is_directory:
                    _require_descriptor_source(
                        bound_artifact.artifact,
                        bound_artifact.staged_entry,
                        source,
                        require_staged_name=True,
                    )
                    private_entry, target_descriptor, detached_target = (
                        _create_private_detached_directory(bound_artifact)
                    )
                    private_operation_error: BaseException | None = None
                    try:
                        _copy_directory_descriptor_source_to_target(
                            bound_artifact,
                            target_descriptor,
                        )
                        _require_detached_entry(
                            bound_artifact,
                            private_entry,
                            detached_target,
                        )
                        if _bound_entry_exists(bound_artifact.target_entry):
                            raise RuntimeError(
                                "Descriptor-backed publication target was recreated "
                                f"before atomic commit: {bound_artifact.target_entry.path}"
                            )
                        if is_commit_point:
                            _require_descriptor_source(
                                bound_artifact.artifact,
                                bound_artifact.staged_entry,
                                source,
                                require_staged_name=True,
                            )
                            require_precommit_state()
                            _require_detached_entry(
                                bound_artifact,
                                private_entry,
                                detached_target,
                            )
                            _require_descriptor_source(
                                bound_artifact.artifact,
                                bound_artifact.staged_entry,
                                source,
                                require_staged_name=True,
                            )
                            _require_bound_artifact_parents_unchanged(artifacts)
                        replace_and_record_promotion(
                            private_entry,
                            bound_artifact,
                            detached_target.identity,
                            is_commit_point=is_commit_point,
                        )
                        if is_commit_point:
                            _require_bound_artifact_parents_unchanged(artifacts)
                    except BaseException as exc:
                        private_operation_error = exc
                        raise
                    finally:
                        _run_cleanup_steps(
                            (
                                (
                                    "Private detached-directory cleanup also failed",
                                    partial(
                                        _remove_bound_entry,
                                        private_entry,
                                        expected_identity=detached_target.identity,
                                        source_descriptor=target_descriptor,
                                    ),
                                ),
                                (
                                    "Private detached-directory descriptor close "
                                    "also failed",
                                    partial(os.close, target_descriptor),
                                ),
                            ),
                            primary_error=private_operation_error,
                            label="Private detached-directory cleanup failed",
                        )
                    detached_targets[id(bound_artifact)] = detached_target
                    if not is_commit_point:
                        _require_descriptor_target(bound_artifact, detached_target)
                        _require_descriptor_source(
                            bound_artifact.artifact,
                            bound_artifact.staged_entry,
                            source,
                            require_staged_name=True,
                        )
                else:
                    _require_descriptor_source(
                        bound_artifact.artifact,
                        bound_artifact.staged_entry,
                        source,
                        require_staged_name=True,
                    )
                    private_entry, target_descriptor, detached_target = (
                        _create_private_detached_target(bound_artifact)
                    )
                    private_operation_error = None
                    try:
                        _copy_descriptor_source_to_target(
                            bound_artifact,
                            target_descriptor,
                        )
                        _require_detached_entry(
                            bound_artifact,
                            private_entry,
                            detached_target,
                        )
                        if _bound_entry_exists(bound_artifact.target_entry):
                            raise RuntimeError(
                                "Descriptor-backed publication target was recreated "
                                f"before atomic commit: {bound_artifact.target_entry.path}"
                            )
                        if is_commit_point:
                            _require_descriptor_source(
                                bound_artifact.artifact,
                                bound_artifact.staged_entry,
                                source,
                                require_staged_name=True,
                            )
                            require_precommit_state()
                            _require_detached_entry(
                                bound_artifact,
                                private_entry,
                                detached_target,
                            )
                            _require_descriptor_source(
                                bound_artifact.artifact,
                                bound_artifact.staged_entry,
                                source,
                                require_staged_name=True,
                            )
                            _require_bound_artifact_parents_unchanged(artifacts)
                        replace_and_record_promotion(
                            private_entry,
                            bound_artifact,
                            detached_target.identity,
                            is_commit_point=is_commit_point,
                        )
                        if is_commit_point:
                            _require_bound_artifact_parents_unchanged(artifacts)
                    except BaseException as exc:
                        private_operation_error = exc
                        raise
                    finally:
                        _run_cleanup_steps(
                            (
                                (
                                    "Private detached-file cleanup also failed",
                                    partial(
                                        _remove_bound_entry,
                                        private_entry,
                                        expected_identity=detached_target.identity,
                                        source_descriptor=target_descriptor,
                                    ),
                                ),
                                (
                                    "Private detached-file descriptor close also failed",
                                    partial(os.close, target_descriptor),
                                ),
                            ),
                            primary_error=private_operation_error,
                            label="Private detached-file cleanup failed",
                        )
                    detached_targets[id(bound_artifact)] = detached_target
                    if not is_commit_point:
                        _require_descriptor_target(bound_artifact, detached_target)
                        _require_descriptor_source(
                            bound_artifact.artifact,
                            bound_artifact.staged_entry,
                            source,
                            require_staged_name=True,
                        )

            for bound_artifact in artifacts:
                source = bound_artifact.descriptor_source
                if source is None or source.is_directory:
                    continue
                _unlink_descriptor_source_name(bound_artifact)
        except BaseException as promotion_error:
            if cleanup_state.committed:
                _run_cleanup_steps(
                    (
                        (
                            "Committed artifact backup cleanup also failed for "
                            f"{backup.bound_artifact.artifact.label}",
                            partial(_remove_committed_backup_if_unchanged, backup),
                        )
                        for backup in backups
                    ),
                    primary_error=promotion_error,
                    label="Committed artifact backup cleanup failed",
                )
                raise

            def remove_promoted_target(bound_artifact: _BoundArtifact) -> None:
                target = bound_artifact.target_entry
                expected_identity = promoted_identities[id(bound_artifact)]
                current_identity = _optional_bound_entry_identity(target)
                if current_identity is None:
                    return
                if current_identity != expected_identity:
                    raise RuntimeError(
                        "promoted target changed inode before rollback; refusing "
                        f"deletion: {target.path}"
                    )
                _remove_bound_entry(
                    target,
                    expected_identity=expected_identity,
                )

            rollback_failures = _collect_cleanup_failures(
                (
                    *(
                        (
                            "Promoted target rollback removal also failed for "
                            f"{bound_artifact.artifact.label}",
                            partial(remove_promoted_target, bound_artifact),
                        )
                        for bound_artifact in reversed(promoted)
                    ),
                    *(
                        (
                            "Artifact backup restore also failed for "
                            f"{backup.bound_artifact.artifact.label}",
                            partial(_restore_artifact_backup, backup),
                        )
                        for backup in reversed(backups)
                    ),
                )
            )
            if rollback_failures:
                if backups:
                    backup_locations = ", ".join(
                        str(backup.directory.opened_path) for backup in backups
                    )
                    rollback_note = (
                        "Artifact rollback was incomplete; backups remain under "
                        f"{backup_locations}"
                    )
                else:
                    rollback_note = "Artifact rollback was incomplete"
                BaseException.add_note(promotion_error, rollback_note)
                _route_cleanup_failures(
                    rollback_failures,
                    primary_error=promotion_error,
                    label="Artifact rollback failed",
                )
                raise

            _run_cleanup_steps(
                (
                    (
                        "Rolled-back artifact backup cleanup also failed for "
                        f"{backup.bound_artifact.artifact.label}",
                        partial(_remove_backup_directory, backup),
                    )
                    for backup in backups
                ),
                primary_error=promotion_error,
                label="Rolled-back artifact backup cleanup failed",
            )
            raise
        else:
            _run_cleanup_steps(
                (
                    (
                        "Committed artifact backup cleanup also failed for "
                        f"{backup.bound_artifact.artifact.label}",
                        partial(_remove_committed_backup_if_unchanged, backup),
                    )
                    for backup in backups
                ),
                cleanup_state=cleanup_state,
                label="Committed artifact backup cleanup failed",
            )
    except BaseException as error:
        active_error = error
        raise
    finally:
        _run_cleanup_steps(
            (
                (
                    "Artifact backup descriptor cleanup also failed for "
                    f"{backup.bound_artifact.artifact.label}",
                    partial(os.close, backup.directory.descriptor),
                )
                for backup in backups
            ),
            primary_error=active_error,
            cleanup_state=cleanup_state,
            label="Artifact backup descriptor cleanup failed",
        )


def _require_precommit_descriptor_artifacts(
    promoted: list[_BoundArtifact],
    detached_targets: dict[int, _DetachedTarget],
) -> None:
    """Revalidate promoted evidence immediately before the commit-point item."""

    for bound_artifact in promoted:
        source = bound_artifact.descriptor_source
        if source is None:
            continue
        _require_descriptor_source(
            bound_artifact.artifact,
            bound_artifact.staged_entry,
            source,
            require_staged_name=True,
        )
        _require_descriptor_target(
            bound_artifact,
            detached_targets[id(bound_artifact)],
        )


def _require_promoted_target_identities(
    promoted: list[_BoundArtifact],
    promoted_identities: dict[int, tuple[int, int]],
) -> None:
    """Require every pre-commit target to remain the inode just promoted."""

    for bound_artifact in promoted:
        target = bound_artifact.target_entry
        expected_identity = promoted_identities[id(bound_artifact)]
        if _optional_bound_entry_identity(target) != expected_identity:
            raise RuntimeError(
                f"Promoted artifact target changed inode before commit: {target.path}"
            )
        _require_retained_staging_promotion_source(bound_artifact.artifact)


def _require_initial_target_states(artifacts: list[_BoundArtifact]) -> None:
    """Reject any target drift captured before backend authoring."""

    for bound_artifact in artifacts:
        _require_initial_target_state(bound_artifact)


def _require_initial_target_state(
    bound_artifact: _BoundArtifact,
) -> tuple[int, int] | None:
    """Return current target identity after checking captured content and state."""

    target = bound_artifact.target_entry
    observed_identity = _optional_bound_entry_identity(target)
    initial_state = bound_artifact.artifact._initial_target_state
    if initial_state is None:
        return observed_identity
    if target.parent.identity != initial_state.parent_identity:
        raise RuntimeError(
            "Artifact target parent changed after staged targets were created: "
            f"{initial_state.requested_path}"
        )
    observed_state = _optional_target_entry_state(
        target.parent.descriptor,
        target.name,
    )
    if observed_state != initial_state.entry_state:
        raise RuntimeError(
            "Artifact target changed after staged targets were created: "
            f"{initial_state.requested_path}; expected "
            f"{initial_state.entry_state}, found {observed_state}"
        )
    if initial_state.entry_state is None:
        return None
    _require_captured_target_descriptor_state(
        initial_state,
        parent_descriptor=target.parent.descriptor,
        entry_name=target.name,
        expected_state=initial_state.entry_state,
        expected_content_sha256=initial_state.content_sha256,
        label=f"initial publication target {initial_state.requested_path}",
    )
    return initial_state.entry_identity


def _require_captured_target_descriptor_state(
    state: _CapturedTargetState,
    *,
    parent_descriptor: int,
    entry_name: str,
    expected_state: _CapturedStatState,
    expected_content_sha256: str | None,
    label: str,
) -> None:
    """Revalidate one retained target inode and its exact current name."""

    handle = state.entry_handle
    if handle is None or handle.closed or handle.descriptor < 0:
        raise RuntimeError(f"{label} capture descriptor is unavailable")
    descriptor_state = _captured_stat_state(os.fstat(handle.descriptor))
    if descriptor_state != expected_state:
        raise RuntimeError(
            f"{label} changed through its retained descriptor; expected "
            f"{expected_state}, found {descriptor_state}"
        )
    mount_id = _descriptor_mount_id(handle.descriptor)
    if state.entry_mount_id is None or mount_id != state.entry_mount_id:
        raise RuntimeError(f"{label} changed mount identity")
    observed_content = _captured_target_content_sha256(
        handle.descriptor,
        entry_state=expected_state,
        expected_mount_id=mount_id,
        label=label,
    )
    if observed_content != expected_content_sha256:
        raise RuntimeError(
            f"{label} content changed; expected {expected_content_sha256}, "
            f"found {observed_content}"
        )
    if _captured_stat_state(os.fstat(handle.descriptor)) != expected_state:
        raise RuntimeError(f"{label} changed while it was revalidated")


def _capture_artifact_backup_state(backup: _ArtifactBackup) -> None:
    """Establish a post-rename baseline for one exact rollback artifact."""

    initial_state = backup.bound_artifact.artifact._initial_target_state
    if initial_state is None or initial_state.entry_state is None:
        return
    observed_state = _optional_target_entry_state(
        backup.artifact_entry.parent.descriptor,
        backup.artifact_entry.name,
    )
    if observed_state is None or observed_state[:2] != backup.artifact_identity:
        raise RuntimeError(
            "Artifact backup changed before its post-move state was captured: "
            f"{backup.artifact_entry.path}"
        )
    if observed_state[:-1] != initial_state.entry_state[:-1]:
        raise RuntimeError(
            "Artifact backup metadata changed during target-to-backup move: "
            f"{backup.artifact_entry.path}"
        )
    handle = initial_state.entry_handle
    if handle is None or handle.closed:
        raise RuntimeError("Artifact backup capture descriptor is unavailable")
    if _captured_stat_state(os.fstat(handle.descriptor)) != observed_state:
        raise RuntimeError(
            "Artifact backup descriptor differs from its post-move entry: "
            f"{backup.artifact_entry.path}"
        )
    mount_id = _descriptor_mount_id(handle.descriptor)
    if mount_id != initial_state.entry_mount_id:
        raise RuntimeError("Artifact backup changed mount identity after move")
    post_move_content_sha256 = _captured_target_content_sha256(
        handle.descriptor,
        entry_state=observed_state,
        expected_mount_id=mount_id,
        label=f"artifact backup {backup.artifact_entry.path}",
    )
    if post_move_content_sha256 != initial_state.content_sha256:
        raise RuntimeError(
            "Artifact backup content changed during target-to-backup move: "
            f"{backup.artifact_entry.path}"
        )
    if _captured_stat_state(os.fstat(handle.descriptor)) != observed_state:
        raise RuntimeError("Artifact backup changed while its baseline was captured")
    backup.post_move_state = observed_state
    backup.post_move_content_sha256 = post_move_content_sha256


def _require_artifact_backup_unchanged(backup: _ArtifactBackup) -> None:
    """Require one rollback artifact to match its post-move baseline."""

    initial_state = backup.bound_artifact.artifact._initial_target_state
    if initial_state is None or initial_state.entry_state is None:
        return
    expected_state = backup.post_move_state
    if expected_state is None:
        raise RuntimeError("Artifact backup has no post-move state baseline")
    observed_state = _optional_target_entry_state(
        backup.artifact_entry.parent.descriptor,
        backup.artifact_entry.name,
    )
    if observed_state != expected_state:
        raise RuntimeError(
            "Artifact backup changed after target replacement began: "
            f"{backup.artifact_entry.path}"
        )
    _require_captured_target_descriptor_state(
        initial_state,
        parent_descriptor=backup.artifact_entry.parent.descriptor,
        entry_name=backup.artifact_entry.name,
        expected_state=expected_state,
        expected_content_sha256=backup.post_move_content_sha256,
        label=f"artifact backup {backup.artifact_entry.path}",
    )


def _require_artifact_backups_unchanged(
    backups: list[_ArtifactBackup],
) -> None:
    """Require every rollback artifact to retain its captured post-move state."""

    for backup in backups:
        _require_artifact_backup_unchanged(backup)


def _remove_committed_backup_if_unchanged(backup: _ArtifactBackup) -> None:
    """Delete committed rollback evidence only while its state remains exact."""

    _require_artifact_backup_unchanged(backup)
    _remove_backup_directory(backup)


@contextmanager
def _bound_staged_artifacts(
    artifacts: list[StagedArtifact],
    locked_targets: list[_LockedTarget],
    *,
    cleanup_state: _PublicationCleanupState | None = None,
) -> Iterator[list[_BoundArtifact]]:
    """Bind staged entries to held parents and pair them with locked targets."""

    targets_by_path = {target.requested_path: target for target in locked_targets}
    bound_artifacts: list[_BoundArtifact] = []
    staged_directories: list[_BoundDirectory] = []
    staged_identities: set[bytes] = set()
    active_error: BaseException | None = None
    try:
        for artifact in artifacts:
            requested_target = _absolute_lexical_path(artifact.target_path)
            locked_target = targets_by_path[requested_target]
            staged_path = _absolute_lexical_path(artifact.staged_path)
            staged_parent = _open_bound_directory(staged_path.parent)
            staged_directories.append(staged_parent)
            staged_entry = _BoundEntry(
                parent=staged_parent,
                name=staged_path.name,
            )
            if not _bound_entry_exists(staged_entry):
                raise FileNotFoundError(
                    f"Staged {artifact.label} is missing: {staged_path}"
                )
            staged_identity = _physical_entry_identity(
                staged_parent.identity,
                entry_name=staged_path.name,
            )
            _require_staging_promotion_source(
                artifact,
                staged_entry,
            )
            if staged_identity in staged_identities:
                raise ValueError(
                    f"Duplicate physical staged artifact: {artifact.staged_path}"
                )
            staged_identities.add(staged_identity)
            if staged_identity == locked_target.identity:
                raise ValueError(
                    "Staged artifact must not alias its transaction target: "
                    f"{artifact.staged_path}"
                )
            descriptor_source = _bind_descriptor_source(artifact, staged_entry)
            bound_artifacts.append(
                _BoundArtifact(
                    artifact=artifact,
                    staged_entry=staged_entry,
                    target_entry=locked_target.entry,
                    descriptor_source=descriptor_source,
                )
            )
        _require_bound_artifact_parents_unchanged(bound_artifacts)
        yield bound_artifacts
    except BaseException as error:
        active_error = error
        raise
    finally:
        _run_cleanup_steps(
            (
                (
                    f"Staged parent descriptor cleanup failed for "
                    f"{directory.locator_path}",
                    partial(os.close, directory.descriptor),
                )
                for directory in staged_directories
            ),
            primary_error=active_error,
            cleanup_state=cleanup_state,
            label="Staged parent descriptor cleanup failed",
        )


def _require_staging_promotion_source(
    artifact: StagedArtifact,
    staged_entry: _BoundEntry,
) -> tuple[int, int]:
    """Require a direct move to retain its validated payload and owner inodes."""

    identity = _bound_entry_identity(staged_entry)
    _require_retained_staging_promotion_source(artifact)
    state = artifact._promotion_state
    if state is None:
        return identity
    source_identity = _require_present_invariant(
        state.source_identity,
        label=f"staged {artifact.label} source identity",
    )
    source_parent_identity = _require_present_invariant(
        state.source_parent_identity,
        label=f"staged {artifact.label} source parent identity",
    )
    if identity != source_identity:
        raise RuntimeError(
            f"Staged {artifact.label} changed inode after validation: "
            f"{staged_entry.path}"
        )
    if staged_entry.parent.identity != source_parent_identity:
        raise RuntimeError(
            f"Staged {artifact.label} owner changed inode after validation: "
            f"{staged_entry.parent.opened_path}"
        )
    return identity


def _require_retained_staging_promotion_source(artifact: StagedArtifact) -> None:
    """Revalidate retained direct-move authority without reopening its path."""

    state = artifact._promotion_state
    if state is None:
        return
    if (
        state.source_identity is None
        or state.source_parent_identity is None
        or state.source_descriptor < 0
    ):
        raise RuntimeError(
            f"Staged {artifact.label} promotion authority is no longer active"
        )
    retained = os.fstat(state.source_descriptor)
    expected_type = stat.S_ISDIR if state.source_is_directory else stat.S_ISREG
    if (
        not expected_type(retained.st_mode)
        or (
            retained.st_dev,
            retained.st_ino,
        )
        != state.source_identity
    ):
        raise RuntimeError(f"Staged {artifact.label} retained descriptor changed inode")
    if not state.source_is_directory and retained.st_nlink != 1:
        raise RuntimeError(
            f"Staged {artifact.label} gained additional links after validation"
        )
    if state.source_is_directory:
        if state.source_tree_sha256 is None or state.source_mount_id is None:
            raise RuntimeError(
                f"Staged {artifact.label} has no retained tree or mount identity"
            )
        if _descriptor_mount_id(state.source_descriptor) != state.source_mount_id:
            raise ValueError(f"Staged {artifact.label} root crossed a mount point")
        _require_directory_tree_mount_id(
            state.source_descriptor,
            expected_mount_id=state.source_mount_id,
            label=f"Staged {artifact.label}",
        )
        observed_tree_sha256 = _directory_descriptor_tree_sha256(
            state.source_descriptor,
            label=artifact.label,
            require_no_write_bits=True,
            expected_mount_id=state.source_mount_id,
        )
        if observed_tree_sha256 != state.source_tree_sha256:
            raise RuntimeError(f"Staged {artifact.label} tree changed after validation")


def _bind_descriptor_source(
    artifact: StagedArtifact,
    staged_entry: _BoundEntry,
) -> _BoundDescriptorSource | None:
    """Bind and validate one optional descriptor-backed file or directory."""

    if artifact.source_descriptor is None:
        return None
    assert artifact.source_sha256 is not None
    try:
        metadata = os.fstat(artifact.source_descriptor)
    except OSError as exc:
        raise RuntimeError(
            f"Descriptor-backed staged {artifact.label} is not open"
        ) from exc
    if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
        raise ValueError(
            f"Descriptor-backed staged {artifact.label} must be a regular file "
            "or directory"
        )
    try:
        descriptor_flags = fcntl.fcntl(artifact.source_descriptor, fcntl.F_GETFL)
    except OSError as exc:
        raise RuntimeError(
            f"Descriptor-backed staged {artifact.label} flags are unavailable"
        ) from exc
    if (descriptor_flags & os.O_ACCMODE) != os.O_RDONLY:
        raise ValueError(
            f"Descriptor-backed staged {artifact.label} must be opened read-only"
        )
    source_mount_id = None
    if stat.S_ISDIR(metadata.st_mode):
        parent_mount_id = _descriptor_mount_id(staged_entry.parent.descriptor)
        state_mount_id = (
            None
            if artifact._promotion_state is None
            else artifact._promotion_state.source_mount_id
        )
        if state_mount_id is not None and state_mount_id != parent_mount_id:
            raise ValueError(
                f"Descriptor-backed staged {artifact.label} owner crossed a mount point"
            )
        source_mount_id = parent_mount_id
        if _descriptor_mount_id(artifact.source_descriptor) != source_mount_id:
            raise ValueError(
                f"Descriptor-backed staged {artifact.label} root is a mount point"
            )
        _require_descriptor_entry_mount_id(
            staged_entry.parent.descriptor,
            staged_entry.name,
            expected_identity=(metadata.st_dev, metadata.st_ino),
            expected_mount_id=source_mount_id,
            label=f"Descriptor-backed staged {artifact.label}",
        )
        _require_directory_tree_mount_id(
            artifact.source_descriptor,
            expected_mount_id=source_mount_id,
            label=f"Descriptor-backed staged {artifact.label}",
        )
    source = _BoundDescriptorSource(
        descriptor=artifact.source_descriptor,
        identity=(metadata.st_dev, metadata.st_ino),
        sha256=artifact.source_sha256,
        mode=stat.S_IMODE(metadata.st_mode),
        is_directory=stat.S_ISDIR(metadata.st_mode),
        mount_id=source_mount_id,
    )
    _require_descriptor_source(
        artifact,
        staged_entry,
        source,
        require_staged_name=True,
    )
    return source


def _require_descriptor_source(
    artifact: StagedArtifact,
    staged_entry: _BoundEntry,
    source: _BoundDescriptorSource,
    *,
    require_staged_name: bool,
) -> None:
    """Require a stable expected inode and exact file or tree content."""

    try:
        metadata = os.fstat(source.descriptor)
    except OSError as exc:
        raise RuntimeError(
            f"Descriptor-backed staged {artifact.label} was closed"
        ) from exc
    expected_type = stat.S_ISDIR if source.is_directory else stat.S_ISREG
    if not expected_type(metadata.st_mode):
        raise RuntimeError(
            f"Descriptor-backed staged {artifact.label} changed entry type"
        )
    if (metadata.st_dev, metadata.st_ino) != source.identity:
        raise RuntimeError(f"Descriptor-backed staged {artifact.label} changed inode")
    if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError(
            f"Descriptor-backed staged {artifact.label} must have no write permissions"
        )
    source_mount_id = source.mount_id
    if source.is_directory:
        source_mount_id = _require_present_invariant(
            source_mount_id,
            label=f"descriptor-backed staged {artifact.label} source mount ID",
        )
    if require_staged_name:
        try:
            staged_metadata = os.stat(
                staged_entry.name,
                dir_fd=staged_entry.parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Descriptor-backed staged {artifact.label} entry disappeared"
            ) from exc
        if (
            not expected_type(staged_metadata.st_mode)
            or (
                staged_metadata.st_dev,
                staged_metadata.st_ino,
            )
            != source.identity
        ):
            raise RuntimeError(
                f"Descriptor-backed staged {artifact.label} entry changed inode"
            )
        if source.is_directory:
            source_mount_id = _require_present_invariant(
                source_mount_id,
                label=(f"descriptor-backed staged {artifact.label} source mount ID"),
            )
            if _descriptor_mount_id(staged_entry.parent.descriptor) != source_mount_id:
                raise ValueError(
                    f"Descriptor-backed staged {artifact.label} owner crossed a mount point"
                )
            _require_descriptor_entry_mount_id(
                staged_entry.parent.descriptor,
                staged_entry.name,
                expected_identity=source.identity,
                expected_mount_id=source_mount_id,
                label=f"Descriptor-backed staged {artifact.label}",
            )
        if not source.is_directory and staged_metadata.st_nlink != 1:
            raise RuntimeError(
                f"Descriptor-backed staged {artifact.label} entry must have exactly "
                f"1 link, found {staged_metadata.st_nlink}"
            )
    if not source.is_directory and metadata.st_nlink != 1:
        raise RuntimeError(
            f"Descriptor-backed staged {artifact.label} must have exactly 1 link, "
            f"found {metadata.st_nlink}"
        )
    if source.is_directory:
        source_mount_id = _require_present_invariant(
            source_mount_id,
            label=f"descriptor-backed staged {artifact.label} source mount ID",
        )
        if _descriptor_mount_id(source.descriptor) != source_mount_id:
            raise ValueError(
                f"Descriptor-backed staged {artifact.label} root crossed a mount point"
            )
        _require_directory_tree_mount_id(
            source.descriptor,
            expected_mount_id=source_mount_id,
            label=f"Descriptor-backed staged {artifact.label}",
        )
        actual_sha256 = _directory_descriptor_tree_sha256(
            source.descriptor,
            label=artifact.label,
            expected_mount_id=source_mount_id,
        )
    else:
        actual_sha256 = _descriptor_sha256(source.descriptor, label=artifact.label)
    if actual_sha256 != source.sha256:
        raise RuntimeError(
            f"Descriptor-backed staged {artifact.label} SHA-256 mismatch: "
            f"expected {source.sha256}, got {actual_sha256}"
        )


def _descriptor_sha256(descriptor: int, *, label: str) -> str:
    """Hash exact descriptor bytes with positional reads and stability checks."""

    return _descriptor_sha256_from_state(
        descriptor,
        expected_state=os.fstat(descriptor),
        label=label,
    )


def _descriptor_sha256_from_state(
    descriptor: int,
    *,
    expected_state: os.stat_result,
    label: str,
) -> str:
    """Hash only the bytes in one already-accounted descriptor state."""

    before = expected_state
    before_state = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    current = os.fstat(descriptor)
    current_state = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )
    if current_state != before_state:
        raise RuntimeError(
            f"Descriptor-backed staged {label} changed while it was hashed"
        )
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        try:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Descriptor-backed staged {label} is not readable"
            ) from exc
        if not chunk:
            raise RuntimeError(
                f"Descriptor-backed staged {label} changed while it was hashed"
            )
        digest.update(chunk)
        offset += len(chunk)
    try:
        grew = bool(os.pread(descriptor, 1, offset))
    except OSError as exc:
        raise RuntimeError(f"Descriptor-backed staged {label} is not readable") from exc
    after = os.fstat(descriptor)
    after_state = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if grew or after_state != before_state:
        raise RuntimeError(
            f"Descriptor-backed staged {label} changed while it was hashed"
        )
    return digest.hexdigest()


def _directory_descriptor_tree_sha256(
    descriptor: int,
    *,
    label: str,
    require_no_write_bits: bool = True,
    expected_mount_id: int | None = None,
) -> str:
    """Hash an exact directory tree through fd-relative, nofollow traversal."""

    mount_id = (
        _descriptor_mount_id(descriptor)
        if expected_mount_id is None
        else expected_mount_id
    )
    _require_directory_tree_mount_id(
        descriptor,
        expected_mount_id=mount_id,
        label=f"Descriptor-backed staged {label}",
    )
    entries: list[dict[str, str | int]] = []
    _collect_directory_tree_entries(
        descriptor,
        relative_directory=".",
        label=label,
        require_no_write_bits=require_no_write_bits,
        expected_mount_id=mount_id,
        entries=entries,
    )
    payload = json.dumps(
        {
            "schema_version": _DIRECTORY_TREE_SCHEMA_VERSION,
            "entries": entries,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _collect_directory_tree_entries(
    descriptor: int,
    *,
    relative_directory: str,
    label: str,
    require_no_write_bits: bool,
    expected_mount_id: int,
    entries: list[dict[str, str | int]],
    traversal_budget: _ArtifactTreeTraversalBudget | None = None,
    depth: int = 0,
) -> None:
    """Collect one stable directory subtree through held descriptors."""

    if traversal_budget is None:
        traversal_budget = _ArtifactTreeTraversalBudget(
            label=f"Descriptor-backed staged {label} tree"
        )
    traversal_budget.consume(
        relative_path=relative_directory,
        depth=depth,
    )
    if _descriptor_mount_id(descriptor) != expected_mount_id:
        raise ValueError(
            "Descriptor-backed staged "
            f"{label} contains a mount point at {relative_directory}"
        )
    before = os.fstat(descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(f"Descriptor-backed staged {label} changed entry type")
    if require_no_write_bits and before.st_mode & 0o222:
        raise RuntimeError(
            f"Descriptor-backed staged {label} tree has writable directory: "
            f"{relative_directory}"
        )
    entries.append(
        {
            "path": relative_directory,
            "type": "directory",
            "mode": stat.S_IMODE(before.st_mode),
        }
    )
    names = traversal_budget.sorted_child_names(
        descriptor,
        relative_path=relative_directory,
    )
    for name in names:
        relative_path = (
            name if relative_directory == "." else f"{relative_directory}/{name}"
        )
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            traversal_budget.require_depth(
                relative_path=relative_path,
                depth=depth + 1,
            )
        elif not stat.S_ISREG(metadata.st_mode):
            traversal_budget.consume(
                relative_path=relative_path,
                depth=depth + 1,
                byte_count=metadata.st_size,
            )
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(
                f"Descriptor-backed staged {label} tree contains a symlink: "
                f"{relative_path}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            child_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            child_flags |= getattr(os, "O_CLOEXEC", 0)
            child_descriptor = -1
            operation_error = None
            try:
                child_descriptor = os.open(name, child_flags, dir_fd=descriptor)
                child_metadata = os.fstat(child_descriptor)
                if (child_metadata.st_dev, child_metadata.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise RuntimeError(
                        f"Descriptor-backed staged {label} tree changed inode: "
                        f"{relative_path}"
                    )
                if _descriptor_mount_id(child_descriptor) != expected_mount_id:
                    raise ValueError(
                        "Descriptor-backed staged "
                        f"{label} contains a mount point at {relative_path}"
                    )
                _collect_directory_tree_entries(
                    child_descriptor,
                    relative_directory=relative_path,
                    label=label,
                    require_no_write_bits=require_no_write_bits,
                    expected_mount_id=expected_mount_id,
                    entries=entries,
                    traversal_budget=traversal_budget,
                    depth=depth + 1,
                )
            except BaseException as error:
                operation_error = error
                raise
            finally:
                if child_descriptor >= 0:
                    owned_child_descriptor = child_descriptor
                    child_descriptor = -1
                    _run_cleanup_steps(
                        [
                            (
                                f"Directory tree child descriptor close failed for {relative_path}",
                                partial(os.close, owned_child_descriptor),
                            )
                        ],
                        primary_error=operation_error,
                        label="Directory tree child descriptor cleanup failed",
                    )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"Descriptor-backed staged {label} tree contains a special file: "
                f"{relative_path}"
            )
        if metadata.st_nlink != 1:
            raise RuntimeError(
                f"Descriptor-backed staged {label} tree file must have exactly 1 "
                f"link: {relative_path}"
            )
        if require_no_write_bits and metadata.st_mode & 0o222:
            raise RuntimeError(
                f"Descriptor-backed staged {label} tree has writable file: "
                f"{relative_path}"
            )
        file_flags = os.O_RDONLY | os.O_NOFOLLOW
        file_flags |= getattr(os, "O_NONBLOCK", 0)
        file_flags |= getattr(os, "O_CLOEXEC", 0)
        file_descriptor = -1
        operation_error = None
        try:
            file_descriptor = os.open(name, file_flags, dir_fd=descriptor)
            opened_metadata = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(opened_metadata.st_mode)
                or _captured_stat_state(opened_metadata)
                != _captured_stat_state(metadata)
                or opened_metadata.st_nlink != 1
            ):
                raise RuntimeError(
                    f"Descriptor-backed staged {label} tree changed inode: "
                    f"{relative_path}"
                )
            if _descriptor_mount_id(file_descriptor) != expected_mount_id:
                raise ValueError(
                    "Descriptor-backed staged "
                    f"{label} contains a mount point at {relative_path}"
                )
            traversal_budget.consume(
                relative_path=relative_path,
                depth=depth + 1,
                byte_count=opened_metadata.st_size,
            )
            content_sha256 = _descriptor_sha256_from_state(
                file_descriptor,
                expected_state=opened_metadata,
                label=f"{label} tree file {relative_path}",
            )
            after_file = os.fstat(file_descriptor)
            if (
                (after_file.st_dev, after_file.st_ino)
                != (metadata.st_dev, metadata.st_ino)
                or after_file.st_mode != metadata.st_mode
                or after_file.st_nlink != metadata.st_nlink
                or after_file.st_size != metadata.st_size
            ):
                raise RuntimeError(
                    f"Descriptor-backed staged {label} tree changed while hashed: "
                    f"{relative_path}"
                )
        except BaseException as error:
            operation_error = error
            raise
        finally:
            if file_descriptor >= 0:
                owned_file_descriptor = file_descriptor
                file_descriptor = -1
                _run_cleanup_steps(
                    [
                        (
                            f"Directory tree file descriptor close failed for {relative_path}",
                            partial(os.close, owned_file_descriptor),
                        )
                    ],
                    primary_error=operation_error,
                    label="Directory tree file descriptor cleanup failed",
                )
        entries.append(
            {
                "path": relative_path,
                "type": "file",
                "mode": stat.S_IMODE(metadata.st_mode),
                "size": metadata.st_size,
                "sha256": content_sha256,
            }
        )
    after = os.fstat(descriptor)
    after_names = _bounded_sorted_directory_names(
        descriptor,
        maximum_names=len(names),
        overflow_message=(
            "Descriptor-backed staged "
            f"{label} tree changed while hashed: {relative_directory}"
        ),
    )
    before_state = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_state = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if after_names != names or after_state != before_state:
        raise RuntimeError(
            f"Descriptor-backed staged {label} tree changed while hashed: "
            f"{relative_directory}"
        )


def _open_bound_directory(path: Path) -> _BoundDirectory:
    """Open one directory without following its final canonical component."""

    locator_path = _absolute_lexical_path(path)
    opened_path = locator_path.resolve(strict=True)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(opened_path, flags)
    try:
        identity = _physical_directory_identity(
            opened_path,
            descriptor=descriptor,
        )
        if (
            _physical_directory_identity(locator_path) != identity
            or _physical_directory_identity(opened_path) != identity
        ):
            raise RuntimeError(
                f"Publication parent changed while it was opened: {locator_path}"
            )
        return _BoundDirectory(
            locator_path=locator_path,
            opened_path=opened_path,
            descriptor=descriptor,
            identity=identity,
        )
    except BaseException as error:
        _run_cleanup_steps(
            [
                (
                    f"Bound directory descriptor cleanup failed for {locator_path}",
                    lambda: os.close(descriptor),
                )
            ],
            primary_error=error,
            label="Bound directory descriptor cleanup failed",
        )
        raise


def _open_bound_child_directory(parent: _BoundDirectory, name: str) -> _BoundDirectory:
    """Open a newly created child relative to a held parent descriptor."""

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=parent.descriptor)
    try:
        opened_path = parent.opened_path / name
        identity = _physical_directory_identity(
            opened_path,
            descriptor=descriptor,
        )
        return _BoundDirectory(
            locator_path=parent.locator_path / name,
            opened_path=opened_path,
            descriptor=descriptor,
            identity=identity,
        )
    except BaseException as error:
        _run_cleanup_steps(
            [
                (
                    f"Bound child directory descriptor cleanup failed for "
                    f"{parent.locator_path / name}",
                    lambda: os.close(descriptor),
                )
            ],
            primary_error=error,
            label="Bound child directory descriptor cleanup failed",
        )
        raise


def _require_bound_directory_unchanged(directory: _BoundDirectory) -> None:
    """Require both original parent locators to still reach the held inode."""

    try:
        locator_identity = _physical_directory_identity(directory.locator_path)
        opened_identity = _physical_directory_identity(directory.opened_path)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise RuntimeError(
            f"Publication parent changed during transaction: {directory.locator_path}"
        ) from exc
    if locator_identity != directory.identity or opened_identity != directory.identity:
        raise RuntimeError(
            f"Publication parent changed during transaction: {directory.locator_path}"
        )


def _require_bound_artifact_parents_unchanged(
    artifacts: list[_BoundArtifact],
) -> None:
    """Revalidate every staged and target parent locator once per descriptor."""

    directories: dict[int, _BoundDirectory] = {}
    for artifact in artifacts:
        directories[artifact.staged_entry.parent.descriptor] = (
            artifact.staged_entry.parent
        )
        directories[artifact.target_entry.parent.descriptor] = (
            artifact.target_entry.parent
        )
    for directory in directories.values():
        _require_bound_directory_unchanged(directory)


def _bound_entry_exists(entry: _BoundEntry) -> bool:
    """Return whether an fd-relative entry exists without following symlinks."""

    try:
        os.stat(
            entry.name,
            dir_fd=entry.parent.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    return True


def _optional_bound_entry_identity(entry: _BoundEntry) -> tuple[int, int] | None:
    """Return one no-follow entry identity, or ``None`` when it is absent."""

    try:
        metadata = os.stat(
            entry.name,
            dir_fd=entry.parent.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    return metadata.st_dev, metadata.st_ino


def _bound_entry_identity(entry: _BoundEntry) -> tuple[int, int]:
    """Return one required no-follow entry identity."""

    identity = _optional_bound_entry_identity(entry)
    if identity is None:
        raise FileNotFoundError(f"Artifact entry is missing: {entry.path}")
    return identity


def _optional_descriptor_entry_identity(
    parent_descriptor: int,
    entry_name: str,
) -> tuple[int, int] | None:
    """Return one fd-relative no-follow identity, or ``None`` when absent."""

    try:
        metadata = os.stat(
            entry_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    return metadata.st_dev, metadata.st_ino


def _open_child_directory_descriptor(parent_descriptor: int, name: str) -> int:
    """Open one child directory without binding identity from its lexical name."""

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    return os.open(name, flags, dir_fd=parent_descriptor)


def _note_ambiguous_directory_creation(
    error: BaseException,
    parent: _BoundDirectory,
    name: str,
    *,
    label: str,
) -> None:
    """Record a mkdir failure without adopting or deleting its lexical result."""

    path = parent.opened_path / name
    try:
        observed_identity = _optional_descriptor_entry_identity(
            parent.descriptor,
            name,
        )
    except BaseException as evidence_error:
        error.add_note(
            f"{label} mkdir outcome is ambiguous; no cleanup ownership was bound "
            f"for unpredictable private name {path}; preservation check failed: "
            f"{type(evidence_error).__name__}: {evidence_error}"
        )
        return
    if observed_identity is None:
        error.add_note(
            f"{label} mkdir outcome is ambiguous; no lexical entry was observed "
            f"and no cleanup deletion was attempted for {path}"
        )
        return
    error.add_note(
        f"{label} mkdir outcome is ambiguous; unpredictable private name preserved "
        f"without cleanup ownership at {path}"
    )


def _note_ambiguous_file_creation(
    error: BaseException,
    parent: _BoundDirectory,
    name: str,
    *,
    label: str,
) -> None:
    """Record an O_CREAT failure without adopting or deleting its lexical result."""

    path = parent.opened_path / name
    try:
        observed_identity = _optional_descriptor_entry_identity(
            parent.descriptor,
            name,
        )
    except BaseException as evidence_error:
        error.add_note(
            f"{label} O_CREAT outcome is ambiguous; no cleanup ownership was "
            f"bound for unpredictable private name {path}; preservation check "
            f"failed: {type(evidence_error).__name__}: {evidence_error}"
        )
        return
    if observed_identity is None:
        error.add_note(
            f"{label} O_CREAT outcome is ambiguous; no lexical entry was observed "
            f"and no cleanup deletion was attempted for {path}"
        )
        return
    error.add_note(
        f"{label} O_CREAT outcome is ambiguous; unpredictable private name "
        f"preserved without cleanup ownership at {path}"
    )


def _remove_created_directory_if_bound(
    parent: _BoundDirectory,
    name: str,
    *,
    descriptor: int,
    identity: tuple[int, int] | None,
    label: str,
) -> None:
    """Remove a created name only when it still denotes its fd-bound inode."""

    original_identity = identity
    if original_identity is None:
        if descriptor < 0:
            raise RuntimeError(
                f"{label} could not bind the created inode; unpredictable private "
                f"name preserved at {parent.opened_path / name}"
            )
        created = os.fstat(descriptor)
        original_identity = (created.st_dev, created.st_ino)

    lexical_identity = _optional_descriptor_entry_identity(
        parent.descriptor,
        name,
    )
    if lexical_identity is None:
        if descriptor < 0:
            raise RuntimeError(
                f"{label} disappeared from its created name, but its owned inode "
                f"could not be verified; preservation is required"
            )
        retained = os.fstat(descriptor)
        retained_identity = (retained.st_dev, retained.st_ino)
        if not stat.S_ISDIR(retained.st_mode) or retained_identity != original_identity:
            raise RuntimeError(
                f"{label} disappeared from its created name and its retained "
                "descriptor no longer identifies the owned directory"
            )
        if retained.st_nlink == 0:
            return
        raise RuntimeError(
            f"{label} disappeared from its created name; descriptor-owned inode "
            "remains linked elsewhere and was preserved"
        )
    if lexical_identity != original_identity:
        raise RuntimeError(
            f"{label} changed inode; replacement preserved at "
            f"{parent.opened_path / name}"
        )
    _remove_descriptor_entry(
        parent.descriptor,
        name,
        expected_identity=original_identity,
        source_descriptor=descriptor,
        label=label,
    )


def _rename_descriptor_entry_noreplace(
    source_parent_descriptor: int,
    source_name: str,
    target_parent_descriptor: int,
    target_name: str,
    *,
    label: str,
) -> None:
    """Atomically move one fd-relative name without replacing another entry."""

    if _RENAMEAT2 is None:
        raise RuntimeError(f"Atomic no-replace {label} requires Linux renameat2")
    result = _RENAMEAT2(
        source_parent_descriptor,
        os.fsencode(source_name),
        target_parent_descriptor,
        os.fsencode(target_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {
        errno.ENOSYS,
        errno.EOPNOTSUPP,
        errno.EINVAL,
    }:
        raise RuntimeError(f"Filesystem does not support atomic no-replace {label}")
    raise OSError(error_number, os.strerror(error_number), target_name)


def _best_effort_restore_quarantined_entry(
    parent_descriptor: int,
    quarantine_name: str,
    original_name: str,
    *,
    preserved_identity: tuple[int, int],
    label: str,
) -> str:
    """Restore one quarantined inode without overwriting any replacement."""

    quarantine_identity = _optional_descriptor_entry_identity(
        parent_descriptor,
        quarantine_name,
    )
    if quarantine_identity is None:
        return f"{label} quarantine entry disappeared before restoration"
    if quarantine_identity != preserved_identity:
        return (
            f"{label} quarantine changed inode; unrelated entry preserved at "
            f"{quarantine_name}"
        )
    original_identity = _optional_descriptor_entry_identity(
        parent_descriptor,
        original_name,
    )
    if original_identity is not None:
        return (
            f"{label} original name is occupied; quarantined inode preserved at "
            f"{quarantine_name}"
        )
    try:
        _rename_descriptor_entry_noreplace(
            parent_descriptor,
            quarantine_name,
            parent_descriptor,
            original_name,
            label=f"{label} restoration",
        )
    except BaseException as restore_error:
        quarantine_after = _optional_descriptor_entry_identity(
            parent_descriptor,
            quarantine_name,
        )
        original_after = _optional_descriptor_entry_identity(
            parent_descriptor,
            original_name,
        )
        if quarantine_after is None and original_after == preserved_identity:
            return (
                f"{label} inode was restored despite {type(restore_error).__name__}: "
                f"{restore_error}"
            )
        return (
            f"{label} restoration failed ({type(restore_error).__name__}: "
            f"{restore_error}); quarantined inode remains at {quarantine_name}"
        )
    return f"{label} inode restored to {original_name}"


def _quarantine_recovery_note(
    parent_descriptor: int,
    quarantine_name: str,
    original_name: str,
    *,
    preserved_identity: tuple[int, int],
    label: str,
) -> str:
    """Attempt quarantine recovery without ever replacing a primary error."""

    try:
        return _best_effort_restore_quarantined_entry(
            parent_descriptor,
            quarantine_name,
            original_name,
            preserved_identity=preserved_identity,
            label=label,
        )
    except BaseException as recovery_error:
        return (
            f"{label} quarantine recovery itself failed "
            f"({type(recovery_error).__name__}: {recovery_error}); entries preserved"
        )


def _quarantine_descriptor_entry(
    parent_descriptor: int,
    entry_name: str,
    *,
    expected_identity: tuple[int, int],
    label: str,
) -> _QuarantinedDescriptorEntry:
    """Atomically hide and then identity-bind one cleanup target."""

    for _ in range(128):
        quarantine_name = f".joint-rigger.cleanup-{secrets.token_hex(16)}"
        try:
            _rename_descriptor_entry_noreplace(
                parent_descriptor,
                entry_name,
                parent_descriptor,
                quarantine_name,
                label=f"{label} quarantine",
            )
        except BaseException as rename_error:
            try:
                source_identity = _optional_descriptor_entry_identity(
                    parent_descriptor,
                    entry_name,
                )
                quarantine_identity = _optional_descriptor_entry_identity(
                    parent_descriptor,
                    quarantine_name,
                )
            except BaseException as reconciliation_error:
                rename_error.add_note(
                    "Atomic cleanup quarantine state reconciliation failed: "
                    f"{type(reconciliation_error).__name__}: {reconciliation_error}"
                )
                raise rename_error from reconciliation_error
            if (
                isinstance(rename_error, FileExistsError)
                and source_identity == expected_identity
                and quarantine_identity not in {None, expected_identity}
            ):
                continue
            if (
                quarantine_identity == expected_identity
                and source_identity != expected_identity
            ):
                restoration = _quarantine_recovery_note(
                    parent_descriptor,
                    quarantine_name,
                    entry_name,
                    preserved_identity=quarantine_identity,
                    label=label,
                )
            elif quarantine_identity is not None:
                restoration = (
                    f"{label} unrelated quarantine entry preserved at {quarantine_name}"
                )
            else:
                restoration = (
                    f"{label} source remains at {entry_name}; quarantine was not "
                    "adopted"
                )
            rename_error.add_note(
                "Atomic cleanup quarantine failed after state reconciliation; "
                f"{restoration}"
            )
            raise

        try:
            quarantine_identity = _optional_descriptor_entry_identity(
                parent_descriptor,
                quarantine_name,
            )
        except BaseException as identity_error:
            restoration = _quarantine_recovery_note(
                parent_descriptor,
                quarantine_name,
                entry_name,
                preserved_identity=expected_identity,
                label=label,
            )
            identity_error.add_note(
                "Atomic cleanup quarantine completed before identity verification; "
                f"{restoration}"
            )
            raise
        if quarantine_identity == expected_identity:
            return _QuarantinedDescriptorEntry(
                original_name=entry_name,
                quarantine_name=quarantine_name,
                identity=expected_identity,
            )
        if quarantine_identity is None:
            restoration = f"{label} quarantine entry disappeared"
        else:
            restoration = _quarantine_recovery_note(
                parent_descriptor,
                quarantine_name,
                entry_name,
                preserved_identity=quarantine_identity,
                label=label,
            )
        raise RuntimeError(
            f"{label} changed inode during atomic quarantine; refusing deletion; "
            f"{restoration}"
        )
    raise RuntimeError(f"Could not allocate a private cleanup quarantine for {label}")


def _replace_entry(source: _BoundEntry, target: _BoundEntry) -> None:
    """Atomically move one entry while refusing an unexpected destination."""

    if _RENAMEAT2 is None:
        raise RuntimeError(
            "Atomic no-replace artifact publication requires Linux renameat2"
        )
    result = _RENAMEAT2(
        source.parent.descriptor,
        os.fsencode(source.name),
        target.parent.descriptor,
        os.fsencode(target.name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {
            errno.ENOSYS,
            errno.EOPNOTSUPP,
            errno.EINVAL,
        }:
            raise RuntimeError(
                "Filesystem does not support atomic no-replace artifact "
                f"publication: {target.path}"
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(target.path),
        )


def _restore_directory_mode(
    descriptor: int,
    *,
    expected_identity: tuple[int, int],
    expected_mode: int,
    label: str,
) -> None:
    """Restore and verify one exact directory through its retained descriptor."""

    before = os.fstat(descriptor)
    before_identity = (before.st_dev, before.st_ino)
    if not stat.S_ISDIR(before.st_mode) or before_identity != expected_identity:
        raise RuntimeError(
            f"{label} changed inode before mode restoration: expected "
            f"{expected_identity}, found {before_identity}"
        )
    os.fchmod(descriptor, expected_mode)
    after = os.fstat(descriptor)
    after_identity = (after.st_dev, after.st_ino)
    after_mode = stat.S_IMODE(after.st_mode)
    if (
        not stat.S_ISDIR(after.st_mode)
        or after_identity != expected_identity
        or after_mode != expected_mode
    ):
        raise RuntimeError(
            f"{label} mode restoration could not be verified: expected "
            f"identity={expected_identity}, mode={oct(expected_mode)}; found "
            f"identity={after_identity}, mode={oct(after_mode)}"
        )


@contextmanager
def _temporarily_owner_writable_directory(
    entry: _BoundEntry,
    *,
    expected_identity: tuple[int, int],
    label: str,
) -> Iterator[None]:
    """Permit one cross-parent directory rename, then restore its exact mode.

    Linux may require write permission on a directory whose ``..`` entry changes
    during a cross-parent rename. Published sidecars are deliberately sealed, so
    the exact held directory receives owner-write only for the rename window.
    This creates a narrow same-UID mutation window; retained-fd identity checks
    prevent path substitution, but a same-parent backup layout would be needed
    to eliminate that permission tradeoff entirely.
    """

    descriptor = -1
    original_mode: int | None = None
    active_error: BaseException | None = None
    try:
        descriptor = _open_child_directory_descriptor(
            entry.parent.descriptor,
            entry.name,
        )
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        if not stat.S_ISDIR(metadata.st_mode) or identity != expected_identity:
            raise RuntimeError(
                f"{label} changed inode before temporary mode change: expected "
                f"{expected_identity}, found {identity}"
            )
        original_mode = stat.S_IMODE(metadata.st_mode)
        temporary_mode = original_mode | stat.S_IWUSR
        if temporary_mode != original_mode:
            os.fchmod(descriptor, temporary_mode)
            changed = os.fstat(descriptor)
            changed_identity = (changed.st_dev, changed.st_ino)
            if (
                not stat.S_ISDIR(changed.st_mode)
                or changed_identity != expected_identity
                or stat.S_IMODE(changed.st_mode) != temporary_mode
            ):
                raise RuntimeError(
                    f"{label} temporary mode change could not be verified"
                )
        yield
    except BaseException as error:
        active_error = error
        raise
    finally:
        cleanup_steps: list[tuple[str, Callable[[], None]]] = []
        if descriptor >= 0 and original_mode is not None:
            cleanup_steps.append(
                (
                    f"{label} exact mode restoration also failed",
                    partial(
                        _restore_directory_mode,
                        descriptor,
                        expected_identity=expected_identity,
                        expected_mode=original_mode,
                        label=label,
                    ),
                )
            )
        if descriptor >= 0:
            cleanup_steps.append(
                (
                    f"{label} mode-guard descriptor close also failed",
                    partial(os.close, descriptor),
                )
            )
        _run_cleanup_steps(
            cleanup_steps,
            primary_error=active_error,
            label=f"{label} mode-guard cleanup failed",
        )


def _replace_entry_with_directory_mode_guard(
    source: _BoundEntry,
    target: _BoundEntry,
    *,
    expected_identity: tuple[int, int],
    label: str,
) -> None:
    """Rename an exact entry, temporarily unsealing it only when a directory."""

    metadata = os.stat(
        source.name,
        dir_fd=source.parent.descriptor,
        follow_symlinks=False,
    )
    identity = (metadata.st_dev, metadata.st_ino)
    if identity != expected_identity:
        raise RuntimeError(
            f"{label} changed inode before rename: expected {expected_identity}, "
            f"found {identity}"
        )
    if not stat.S_ISDIR(metadata.st_mode):
        _replace_entry(source, target)
        return
    with _temporarily_owner_writable_directory(
        source,
        expected_identity=expected_identity,
        label=label,
    ):
        _replace_entry(source, target)


def _create_private_detached_target(
    bound_artifact: _BoundArtifact,
) -> tuple[_BoundEntry, int, _DetachedTarget]:
    """Create one cryptographically private mode-000 sibling copy target."""

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    parent = bound_artifact.target_entry.parent
    for _ in range(128):
        private_entry = _BoundEntry(
            parent=parent,
            name=f".joint-rigger-copy-{secrets.token_hex(16)}",
        )
        try:
            descriptor = os.open(
                private_entry.name,
                flags,
                0o000,
                dir_fd=parent.descriptor,
            )
        except FileExistsError:  # pragma: no cover - cryptographic collision
            continue
        except BaseException as acquisition_error:
            _note_ambiguous_file_creation(
                acquisition_error,
                parent,
                private_entry.name,
                label="Private detached-copy target",
            )
            raise
        created_identity: tuple[int, int] | None = None
        try:
            metadata = os.fstat(descriptor)
            created_identity = (metadata.st_dev, metadata.st_ino)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0
            ):
                raise RuntimeError(
                    "Promoter-owned copy must start as a singly linked mode-000 "
                    f"regular file: {private_entry.path}"
                )
            source = bound_artifact.descriptor_source
            assert source is not None
            return (
                private_entry,
                descriptor,
                _DetachedTarget(
                    identity=created_identity,
                    sha256=source.sha256,
                    mode=0o444,
                    is_directory=False,
                ),
            )
        except BaseException as creation_error:
            if created_identity is None:
                try:
                    metadata = os.fstat(descriptor)
                    created_identity = (metadata.st_dev, metadata.st_ino)
                except BaseException as identity_error:
                    creation_error.add_note(
                        "Private detached-copy cleanup could not bind the created "
                        f"inode: {identity_error}"
                    )
            if created_identity is not None:
                try:
                    _remove_bound_entry(
                        private_entry,
                        expected_identity=created_identity,
                        source_descriptor=descriptor,
                    )
                except BaseException as cleanup_error:
                    creation_error.add_note(
                        f"Private detached-copy cleanup also failed: {cleanup_error}"
                    )
            try:
                os.close(descriptor)
            except BaseException as close_error:
                creation_error.add_note(
                    f"Private detached-copy descriptor close also failed: {close_error}"
                )
            raise
    raise RuntimeError("Could not allocate a private detached-copy target")


def _create_private_detached_directory(
    bound_artifact: _BoundArtifact,
) -> tuple[_BoundEntry, int, _DetachedTarget]:
    """Create one cryptographically private target-parent directory copy."""

    parent = bound_artifact.target_entry.parent
    source = bound_artifact.descriptor_source
    assert source is not None and source.is_directory
    for _ in range(128):
        private_entry = _BoundEntry(
            parent=parent,
            name=f".joint-rigger-tree-copy-{secrets.token_hex(16)}",
        )
        try:
            os.mkdir(private_entry.name, mode=0o700, dir_fd=parent.descriptor)
        except FileExistsError:  # pragma: no cover - cryptographic collision
            continue
        except BaseException as acquisition_error:
            _note_ambiguous_directory_creation(
                acquisition_error,
                parent,
                private_entry.name,
                label="Private detached-tree target",
            )
            raise
        descriptor = -1
        created_identity: tuple[int, int] | None = None
        try:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(
                private_entry.name,
                flags,
                dir_fd=parent.descriptor,
            )
            metadata = os.fstat(descriptor)
            created_identity = (metadata.st_dev, metadata.st_ino)
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(
                    f"Promoter-owned tree copy is not a directory: {private_entry.path}"
                )
            return (
                private_entry,
                descriptor,
                _DetachedTarget(
                    identity=created_identity,
                    sha256=source.sha256,
                    mode=source.mode,
                    is_directory=True,
                ),
            )
        except BaseException as creation_error:
            if descriptor >= 0:
                if created_identity is None:
                    try:
                        metadata = os.fstat(descriptor)
                        created_identity = (metadata.st_dev, metadata.st_ino)
                    except BaseException as identity_error:
                        creation_error.add_note(
                            "Private detached-tree cleanup could not bind the "
                            f"created inode: {identity_error}"
                        )
                if created_identity is not None:
                    try:
                        _remove_bound_entry(
                            private_entry,
                            expected_identity=created_identity,
                            source_descriptor=descriptor,
                        )
                    except BaseException as cleanup_error:
                        creation_error.add_note(
                            "Private detached-tree cleanup also failed: "
                            f"{cleanup_error}"
                        )
                owned_descriptor = descriptor
                descriptor = -1
                try:
                    os.close(owned_descriptor)
                except BaseException as close_error:
                    creation_error.add_note(
                        "Private detached-tree descriptor close also failed: "
                        f"{close_error}"
                    )
            else:
                creation_error.add_note(
                    "Private detached-tree creation failed before its inode could "
                    "be bound; the unpredictable private name was preserved"
                )
            raise
    raise RuntimeError("Could not allocate a private detached-tree target")


def _copy_directory_descriptor_source_to_target(
    bound_artifact: _BoundArtifact,
    target_descriptor: int,
) -> None:
    """Copy one stable descriptor-bound tree into a private target directory."""

    source = bound_artifact.descriptor_source
    assert source is not None and source.is_directory
    source_mount_id = source.mount_id
    source_mount_id = _require_present_invariant(
        source_mount_id,
        label=(
            f"descriptor-backed staged {bound_artifact.artifact.label} source mount ID"
        ),
    )
    _require_descriptor_source(
        bound_artifact.artifact,
        bound_artifact.staged_entry,
        source,
        require_staged_name=True,
    )
    _copy_directory_descriptor_tree(
        source.descriptor,
        target_descriptor,
        label=bound_artifact.artifact.label,
        expected_source_mount_id=source_mount_id,
    )
    os.fchmod(target_descriptor, source.mode)
    os.fsync(target_descriptor)
    source_after_sha256 = _directory_descriptor_tree_sha256(
        source.descriptor,
        label=bound_artifact.artifact.label,
        expected_mount_id=source_mount_id,
    )
    target_sha256 = _directory_descriptor_tree_sha256(
        target_descriptor,
        label=f"detached {bound_artifact.artifact.label}",
    )
    if source_after_sha256 != source.sha256 or target_sha256 != source.sha256:
        raise RuntimeError(
            f"Descriptor-backed staged {bound_artifact.artifact.label} tree "
            "changed during detached copy"
        )


def _copy_directory_descriptor_tree(
    source_descriptor: int,
    target_descriptor: int,
    *,
    label: str,
    preserve_modes: bool = True,
    expected_source_mount_id: int | None = None,
    require_no_write_bits: bool = True,
    traversal_budget: _ArtifactTreeTraversalBudget | None = None,
    relative_directory: str = ".",
    depth: int = 0,
    _current_entry_accounted: bool = False,
) -> None:
    """Recursively copy a nofollow directory tree through held descriptors."""

    root_traversal = traversal_budget is None
    if root_traversal:
        traversal_budget = _ArtifactTreeTraversalBudget(
            label=f"Descriptor-backed staged {label} tree copy"
        )
    traversal_budget = _require_present_invariant(
        traversal_budget,
        label="artifact tree copy traversal budget",
    )
    if not _current_entry_accounted:
        traversal_budget.consume(
            relative_path=relative_directory,
            depth=depth,
        )
    source_mount_id = (
        _descriptor_mount_id(source_descriptor)
        if expected_source_mount_id is None
        else expected_source_mount_id
    )
    if root_traversal:
        _require_directory_tree_mount_id(
            source_descriptor,
            expected_mount_id=source_mount_id,
            label=f"Descriptor-backed staged {label}",
        )
    source_before = os.fstat(source_descriptor)
    if not stat.S_ISDIR(source_before.st_mode) or (
        require_no_write_bits and source_before.st_mode & 0o222
    ):
        raise RuntimeError(f"Descriptor-backed staged {label} tree is not sealed")
    names = traversal_budget.sorted_child_names(
        source_descriptor,
        relative_path=relative_directory,
    )
    _bounded_sorted_directory_names(
        target_descriptor,
        maximum_names=0,
        overflow_message=f"Detached target for {label} must start empty",
    )
    for name in names:
        relative_path = (
            name if relative_directory == "." else f"{relative_directory}/{name}"
        )
        source_metadata = os.stat(
            name,
            dir_fd=source_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISDIR(source_metadata.st_mode):
            traversal_budget.consume(
                relative_path=relative_path,
                depth=depth + 1,
            )
            if require_no_write_bits and source_metadata.st_mode & 0o222:
                raise RuntimeError(
                    f"Descriptor-backed staged {label} has writable directory: {name}"
                )
            os.mkdir(name, mode=0o700, dir_fd=target_descriptor)
            source_child = -1
            target_child = -1
            operation_error: BaseException | None = None
            try:
                directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                directory_flags |= getattr(os, "O_CLOEXEC", 0)
                source_child = os.open(
                    name,
                    directory_flags,
                    dir_fd=source_descriptor,
                )
                target_child = os.open(
                    name,
                    directory_flags,
                    dir_fd=target_descriptor,
                )
                opened_metadata = os.fstat(source_child)
                if (opened_metadata.st_dev, opened_metadata.st_ino) != (
                    source_metadata.st_dev,
                    source_metadata.st_ino,
                ):
                    raise RuntimeError(
                        f"Descriptor-backed staged {label} tree changed inode: {name}"
                    )
                if _descriptor_mount_id(source_child) != source_mount_id:
                    raise ValueError(
                        f"Descriptor-backed staged {label} contains a mount point at {name}"
                    )
                _copy_directory_descriptor_tree(
                    source_child,
                    target_child,
                    label=f"{label}/{name}",
                    preserve_modes=preserve_modes,
                    expected_source_mount_id=source_mount_id,
                    require_no_write_bits=require_no_write_bits,
                    traversal_budget=traversal_budget,
                    relative_directory=relative_path,
                    depth=depth + 1,
                    _current_entry_accounted=True,
                )
                if preserve_modes:
                    os.fchmod(target_child, stat.S_IMODE(source_metadata.st_mode))
                os.fsync(target_child)
            except BaseException as error:
                operation_error = error
                raise
            finally:
                close_steps: list[tuple[str, Callable[[], None]]] = []
                for descriptor in (target_child, source_child):
                    if descriptor < 0:
                        continue
                    owned_descriptor = descriptor
                    close_steps.append(
                        (
                            f"Directory-copy descriptor close failed for {label}/{name}",
                            partial(os.close, owned_descriptor),
                        )
                    )
                _run_cleanup_steps(
                    close_steps,
                    primary_error=operation_error,
                    label=(
                        f"Directory-copy descriptor cleanup failed for {label}/{name}"
                    ),
                )
            continue
        if not stat.S_ISREG(source_metadata.st_mode):
            traversal_budget.consume(
                relative_path=relative_path,
                depth=depth + 1,
                byte_count=source_metadata.st_size,
            )
            raise RuntimeError(
                f"Descriptor-backed staged {label} contains a special entry: {name}"
            )
        if source_metadata.st_nlink != 1 or (
            require_no_write_bits and source_metadata.st_mode & 0o222
        ):
            raise RuntimeError(
                f"Descriptor-backed staged {label} file is not sealed: {name}"
            )
        source_file = -1
        target_file = -1
        operation_error = None
        try:
            source_flags = os.O_RDONLY | os.O_NOFOLLOW
            source_flags |= getattr(os, "O_NONBLOCK", 0)
            source_flags |= getattr(os, "O_CLOEXEC", 0)
            source_file = os.open(name, source_flags, dir_fd=source_descriptor)
            opened_metadata = os.fstat(source_file)
            if (
                not stat.S_ISREG(opened_metadata.st_mode)
                or _captured_stat_state(opened_metadata)
                != _captured_stat_state(source_metadata)
                or opened_metadata.st_nlink != 1
            ):
                raise RuntimeError(
                    f"Descriptor-backed staged {label} tree changed inode: {name}"
                )
            if _descriptor_mount_id(source_file) != source_mount_id:
                raise ValueError(
                    f"Descriptor-backed staged {label} contains a mount point at {name}"
                )
            traversal_budget.consume(
                relative_path=relative_path,
                depth=depth + 1,
                byte_count=opened_metadata.st_size,
            )
            source_sha256 = _descriptor_sha256_from_state(
                source_file,
                expected_state=opened_metadata,
                label=f"{label}/{name}",
            )
            target_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
            target_flags |= getattr(os, "O_CLOEXEC", 0)
            target_file = os.open(
                name,
                target_flags,
                0o000,
                dir_fd=target_descriptor,
            )
            _copy_stable_descriptor(
                source_file,
                expected_identity=(opened_metadata.st_dev, opened_metadata.st_ino),
                expected_sha256=source_sha256,
                expected_mode=stat.S_IMODE(opened_metadata.st_mode),
                target_descriptor=target_file,
                target_mode=(
                    stat.S_IMODE(opened_metadata.st_mode) if preserve_modes else 0o600
                ),
                label=f"{label}/{name}",
                expected_source_state=opened_metadata,
            )
        except BaseException as error:
            operation_error = error
            raise
        finally:
            close_steps = []
            for descriptor in (target_file, source_file):
                if descriptor < 0:
                    continue
                owned_descriptor = descriptor
                close_steps.append(
                    (
                        f"File-copy descriptor close failed for {label}/{name}",
                        partial(os.close, owned_descriptor),
                    )
                )
            _run_cleanup_steps(
                close_steps,
                primary_error=operation_error,
                label=f"File-copy descriptor cleanup failed for {label}/{name}",
            )
    source_after = os.fstat(source_descriptor)
    after_names = _bounded_sorted_directory_names(
        source_descriptor,
        maximum_names=len(names),
        overflow_message=(f"Descriptor-backed staged {label} changed during copy"),
    )
    if after_names != names or (
        source_after.st_dev,
        source_after.st_ino,
        source_after.st_mode,
        source_after.st_mtime_ns,
        source_after.st_ctime_ns,
    ) != (
        source_before.st_dev,
        source_before.st_ino,
        source_before.st_mode,
        source_before.st_mtime_ns,
        source_before.st_ctime_ns,
    ):
        raise RuntimeError(f"Descriptor-backed staged {label} changed during copy")


def _copy_descriptor_source_to_target(
    bound_artifact: _BoundArtifact,
    target_descriptor: int,
) -> None:
    """Copy stable expected descriptor bytes and seal the detached target."""

    source = bound_artifact.descriptor_source
    assert source is not None
    _copy_stable_descriptor(
        source.descriptor,
        expected_identity=source.identity,
        expected_sha256=source.sha256,
        expected_mode=source.mode,
        target_descriptor=target_descriptor,
        target_mode=0o444,
        label=bound_artifact.artifact.label,
    )


def _copy_stable_descriptor(
    source_descriptor: int,
    *,
    expected_identity: tuple[int, int],
    expected_sha256: str,
    expected_mode: int,
    target_descriptor: int,
    target_mode: int,
    label: str,
    expected_source_state: os.stat_result | None = None,
) -> None:
    """Copy one stable regular-file descriptor via positional reads."""

    before = expected_source_state or os.fstat(source_descriptor)
    observed_before = os.fstat(source_descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or (before.st_dev, before.st_ino) != expected_identity
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
        or _captured_stat_state(observed_before) != _captured_stat_state(before)
    ):
        raise RuntimeError(f"Descriptor-backed staged {label} changed before copy")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        try:
            chunk = os.pread(
                source_descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Descriptor-backed staged {label} is not readable"
            ) from exc
        if not chunk:
            raise RuntimeError(
                f"Descriptor-backed staged {label} changed while it was copied"
            )
        digest.update(chunk)
        _write_all(target_descriptor, chunk, label=label)
        offset += len(chunk)
    try:
        grew = bool(os.pread(source_descriptor, 1, offset))
    except OSError as exc:
        raise RuntimeError(f"Descriptor-backed staged {label} is not readable") from exc
    after = os.fstat(source_descriptor)
    before_state = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mode,
        before.st_nlink,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_state = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mode,
        after.st_nlink,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if grew or after_state != before_state:
        raise RuntimeError(
            f"Descriptor-backed staged {label} changed while it was copied"
        )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Descriptor-backed staged {label} SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    os.fsync(target_descriptor)
    os.fchmod(target_descriptor, target_mode)
    os.fsync(target_descriptor)
    target_metadata = os.fstat(target_descriptor)
    if (
        not stat.S_ISREG(target_metadata.st_mode)
        or target_metadata.st_nlink != 1
        or target_metadata.st_size != before.st_size
        or stat.S_IMODE(target_metadata.st_mode) != target_mode
    ):
        raise RuntimeError(f"Detached target for {label} changed during copy")


def _write_all(descriptor: int, content: bytes, *, label: str) -> None:
    """Write every copied byte or fail the surrounding transaction."""

    remaining = memoryview(content)
    while remaining:
        try:
            written = os.write(descriptor, remaining)
        except InterruptedError:
            continue
        except OSError as exc:
            raise RuntimeError(f"Could not write detached target for {label}") from exc
        if written <= 0:
            raise RuntimeError(f"Could not write detached target for {label}")
        remaining = remaining[written:]


def _require_descriptor_target(
    bound_artifact: _BoundArtifact,
    detached_target: _DetachedTarget,
) -> None:
    """Reopen and validate one immutable promoter-owned target copy."""

    _require_detached_entry(
        bound_artifact,
        bound_artifact.target_entry,
        detached_target,
    )


def _require_detached_entry(
    bound_artifact: _BoundArtifact,
    entry: _BoundEntry,
    detached_target: _DetachedTarget,
) -> None:
    """Reopen and validate one exact detached-copy file or directory."""

    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    if detached_target.is_directory:
        flags |= os.O_DIRECTORY
    descriptor = -1
    operation_error: BaseException | None = None
    try:
        descriptor = os.open(
            entry.name,
            flags,
            dir_fd=entry.parent.descriptor,
        )
        metadata = os.fstat(descriptor)
        expected_type = stat.S_ISDIR if detached_target.is_directory else stat.S_ISREG
        if (
            not expected_type(metadata.st_mode)
            or (
                metadata.st_dev,
                metadata.st_ino,
            )
            != detached_target.identity
            or stat.S_IMODE(metadata.st_mode) != detached_target.mode
            or (not detached_target.is_directory and metadata.st_nlink != 1)
        ):
            raise RuntimeError(
                f"Descriptor-backed target changed after detached copy: {entry.path}"
            )
        actual_sha256 = (
            _directory_descriptor_tree_sha256(
                descriptor,
                label=bound_artifact.artifact.label,
            )
            if detached_target.is_directory
            else _descriptor_sha256(
                descriptor,
                label=bound_artifact.artifact.label,
            )
        )
        if actual_sha256 != detached_target.sha256:
            raise RuntimeError(
                f"Descriptor-backed target for {bound_artifact.artifact.label} "
                "failed SHA-256 verification"
            )
    except BaseException as error:
        operation_error = error
        raise
    finally:
        if descriptor >= 0:
            owned_descriptor = descriptor
            descriptor = -1
            _run_cleanup_steps(
                [
                    (
                        "Detached target verification descriptor close failed",
                        partial(os.close, owned_descriptor),
                    )
                ],
                primary_error=operation_error,
                label="Detached target verification descriptor cleanup failed",
            )


def _restore_descriptor_source_name(
    bound_artifact: _BoundArtifact,
    detached_target: _DetachedTarget,
) -> None:
    """Restore one removed staged source from the immutable detached target."""

    source = bound_artifact.descriptor_source
    assert source is not None
    if _bound_entry_exists(bound_artifact.staged_entry):
        raise RuntimeError(
            "Cannot restore descriptor-backed staged source because its name is "
            f"occupied: {bound_artifact.staged_entry.path}"
        )
    source_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    source_flags |= getattr(os, "O_NONBLOCK", 0)
    source_descriptor = -1
    target_descriptor = -1
    target_identity: tuple[int, int] | None = None
    operation_error: BaseException | None = None
    try:
        source_descriptor = os.open(
            bound_artifact.target_entry.name,
            source_flags,
            dir_fd=bound_artifact.target_entry.parent.descriptor,
        )
        target_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
        target_flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            target_descriptor = os.open(
                bound_artifact.staged_entry.name,
                target_flags,
                0o000,
                dir_fd=bound_artifact.staged_entry.parent.descriptor,
            )
        except FileExistsError:
            raise
        except BaseException as acquisition_error:
            _note_ambiguous_file_creation(
                acquisition_error,
                bound_artifact.staged_entry.parent,
                bound_artifact.staged_entry.name,
                label="Descriptor-source restoration target",
            )
            raise
        target_metadata = os.fstat(target_descriptor)
        target_identity = (target_metadata.st_dev, target_metadata.st_ino)
        _copy_stable_descriptor(
            source_descriptor,
            expected_identity=detached_target.identity,
            expected_sha256=detached_target.sha256,
            expected_mode=0o444,
            target_descriptor=target_descriptor,
            target_mode=source.mode,
            label=bound_artifact.artifact.label,
        )
    except BaseException as restore_error:
        operation_error = restore_error
        if target_identity is None and target_descriptor >= 0:
            try:
                target_metadata = os.fstat(target_descriptor)
                target_identity = (
                    target_metadata.st_dev,
                    target_metadata.st_ino,
                )
            except BaseException as identity_error:
                restore_error.add_note(
                    "Descriptor-source restoration could not bind the partial "
                    f"staged name for cleanup: {identity_error}"
                )
        if target_identity is not None:
            try:
                _remove_bound_entry(
                    bound_artifact.staged_entry,
                    expected_identity=target_identity,
                    source_descriptor=target_descriptor,
                )
            except BaseException as cleanup_error:
                restore_error.add_note(
                    "Descriptor-source restoration cleanup also failed: "
                    f"{cleanup_error}"
                )
    finally:
        close_steps: list[tuple[str, Callable[[], None]]] = []
        if target_descriptor >= 0:
            owned_target_descriptor = target_descriptor
            target_descriptor = -1
            close_steps.append(
                (
                    "Descriptor-source restoration target close failed",
                    partial(os.close, owned_target_descriptor),
                )
            )
        if source_descriptor >= 0:
            owned_source_descriptor = source_descriptor
            source_descriptor = -1
            close_steps.append(
                (
                    "Descriptor-source restoration source close failed",
                    partial(os.close, owned_source_descriptor),
                )
            )
        _run_cleanup_steps(
            close_steps,
            primary_error=operation_error,
            label="Descriptor-source restoration cleanup failed",
        )
    if operation_error is not None:
        raise operation_error


def _unlink_descriptor_source_name(bound_artifact: _BoundArtifact) -> None:
    """Remove one validated staged source through atomic identity quarantine."""

    source = bound_artifact.descriptor_source
    assert source is not None
    _remove_descriptor_entry(
        bound_artifact.staged_entry.parent.descriptor,
        bound_artifact.staged_entry.name,
        expected_identity=source.identity,
        source_descriptor=source.descriptor,
        label=f"descriptor source {bound_artifact.staged_entry.path}",
    )


def _remove_bound_entry(
    entry: _BoundEntry,
    *,
    expected_identity: tuple[int, int] | None = None,
    source_descriptor: int = -1,
) -> None:
    """Remove one fd-relative entry only after atomic identity quarantine."""

    try:
        metadata = os.stat(
            entry.name,
            dir_fd=entry.parent.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    cleanup_identity = (
        (metadata.st_dev, metadata.st_ino)
        if expected_identity is None
        else expected_identity
    )
    _remove_descriptor_entry(
        entry.parent.descriptor,
        entry.name,
        expected_identity=cleanup_identity,
        source_descriptor=source_descriptor,
        label=f"artifact cleanup entry {entry.path}",
    )


def _descriptor_mount_id(descriptor: int) -> int:
    """Read one Linux mount ID for a held descriptor or fail closed."""

    try:
        lines = (
            (_PROC_SELF_FDINFO / str(descriptor))
            .read_text(encoding="utf-8")
            .splitlines()
        )
    except OSError as exc:
        raise RuntimeError(
            "Safe recursive artifact replacement requires Linux /proc/self/fdinfo"
        ) from exc
    values: list[int] = []
    for line in lines:
        key, separator, value = line.partition(":")
        if key != "mnt_id":
            continue
        if not separator:
            raise RuntimeError(f"Malformed mount ID for descriptor {descriptor}")
        try:
            mount_id = int(value.strip())
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed mount ID for descriptor {descriptor}"
            ) from exc
        if mount_id <= 0:
            raise RuntimeError(f"Malformed mount ID for descriptor {descriptor}")
        values.append(mount_id)
    if len(values) != 1:
        raise RuntimeError(f"Expected exactly one mount ID for descriptor {descriptor}")
    return values[0]


def _require_descriptor_entry_mount_id(
    parent_descriptor: int,
    entry_name: str,
    *,
    expected_identity: tuple[int, int],
    expected_mount_id: int,
    label: str,
) -> None:
    """Reopen one named directory and reject identity or mount substitutions."""

    metadata = os.stat(entry_name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        raise RuntimeError(f"{label} changed inode before mount validation")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    operation_error: BaseException | None = None
    try:
        descriptor = os.open(entry_name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != expected_identity
        ):
            raise RuntimeError(f"{label} changed inode during mount validation")
        if _descriptor_mount_id(descriptor) != expected_mount_id:
            raise ValueError(f"{label} is or crossed a mount point")
    except BaseException as error:
        operation_error = error
        raise
    finally:
        if descriptor >= 0:
            _run_cleanup_steps(
                [
                    (
                        f"{label} mount descriptor close failed",
                        partial(os.close, descriptor),
                    )
                ],
                primary_error=operation_error,
                label=f"{label} mount descriptor cleanup failed",
            )


def _require_existing_target_mount_boundaries(
    artifacts: list[_BoundArtifact],
) -> None:
    """Reject root or nested mounts before any existing target is moved."""

    for bound_artifact in artifacts:
        entry = bound_artifact.target_entry
        try:
            metadata = os.stat(
                entry.name,
                dir_fd=entry.parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            continue
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = -1
        operation_error: BaseException | None = None
        try:
            descriptor = os.open(entry.name, flags, dir_fd=entry.parent.descriptor)
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise RuntimeError(
                    f"Existing {bound_artifact.artifact.label} changed inode "
                    "during mount validation"
                )
            parent_mount_id = _descriptor_mount_id(entry.parent.descriptor)
            root_mount_id = _descriptor_mount_id(descriptor)
            if root_mount_id != parent_mount_id:
                raise ValueError(
                    f"Existing {bound_artifact.artifact.label} root is a mount point: "
                    f"{entry.path}"
                )
            _require_directory_tree_mount_id(
                descriptor,
                expected_mount_id=root_mount_id,
                label=f"Existing {bound_artifact.artifact.label}",
            )
        except BaseException as error:
            operation_error = error
            raise
        finally:
            if descriptor >= 0:
                owned_descriptor = descriptor
                descriptor = -1
                _run_cleanup_steps(
                    [
                        (
                            "Existing target mount descriptor close failed",
                            partial(os.close, owned_descriptor),
                        )
                    ],
                    primary_error=operation_error,
                    label="Existing target mount descriptor cleanup failed",
                )


def _require_directory_tree_mount_id(
    descriptor: int,
    *,
    expected_mount_id: int,
    label: str,
    relative_path: str = ".",
    traversal_budget: _ArtifactTreeTraversalBudget | None = None,
    depth: int = 0,
) -> None:
    """Reject nested file or directory mounts through fd-bound traversal."""

    if traversal_budget is None:
        traversal_budget = _ArtifactTreeTraversalBudget(label=label)
    traversal_budget.consume(
        relative_path=relative_path,
        depth=depth,
    )
    if _descriptor_mount_id(descriptor) != expected_mount_id:
        raise ValueError(f"{label} contains a mount point at {relative_path}")
    names = traversal_budget.sorted_child_names(
        descriptor,
        relative_path=relative_path,
    )
    for name in names:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        child_relative = name if relative_path == "." else f"{relative_path}/{name}"
        if stat.S_ISDIR(metadata.st_mode):
            traversal_budget.require_depth(
                relative_path=child_relative,
                depth=depth + 1,
            )
        else:
            traversal_budget.consume(
                relative_path=child_relative,
                depth=depth + 1,
                byte_count=metadata.st_size,
            )
        if stat.S_ISREG(metadata.st_mode):
            flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NONBLOCK", 0)
        elif stat.S_ISDIR(metadata.st_mode):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
        else:
            o_path = getattr(os, "O_PATH", None)
            if o_path is None:  # pragma: no cover - official targets are Linux
                raise RuntimeError("Safe artifact traversal requires Linux O_PATH")
            flags = o_path | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        child = -1
        operation_error: BaseException | None = None
        try:
            child = os.open(name, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            if (opened.st_dev, opened.st_ino) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise RuntimeError(f"{label} changed inode at {relative_path}/{name}")
            if _descriptor_mount_id(child) != expected_mount_id:
                raise ValueError(f"{label} contains a mount point at {child_relative}")
            if stat.S_ISDIR(opened.st_mode):
                _require_directory_tree_mount_id(
                    child,
                    expected_mount_id=expected_mount_id,
                    label=label,
                    relative_path=child_relative,
                    traversal_budget=traversal_budget,
                    depth=depth + 1,
                )
        except BaseException as error:
            operation_error = error
            raise
        finally:
            if child >= 0:
                owned_child = child
                child = -1
                _run_cleanup_steps(
                    [
                        (
                            f"Mount traversal descriptor close failed at {child_relative}",
                            partial(os.close, owned_child),
                        )
                    ],
                    primary_error=operation_error,
                    label="Mount traversal descriptor cleanup failed",
                )


def _remove_descriptor_entry(
    parent_descriptor: int,
    entry_name: str,
    *,
    expected_identity: tuple[int, int],
    label: str,
    source_descriptor: int = -1,
    expected_mount_id: int | None = None,
    _deletion_budget: _ArtifactTreeTraversalBudget | None = None,
    _depth: int = 0,
) -> None:
    """Atomically quarantine and remove one exact fd-relative inode."""

    metadata = os.stat(
        entry_name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (metadata.st_dev, metadata.st_ino) != expected_identity:
        raise RuntimeError(
            f"{label} changed inode; refusing deletion; replacement preserved"
        )

    descriptor = source_descriptor
    close_descriptor = False
    if descriptor < 0:
        if stat.S_ISDIR(metadata.st_mode):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        else:
            o_path = getattr(os, "O_PATH", None)
            if o_path is None:  # pragma: no cover - official targets are Linux
                raise RuntimeError("Safe artifact cleanup requires Linux O_PATH")
            flags = o_path | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(entry_name, flags, dir_fd=parent_descriptor)
        close_descriptor = True

    operation_error: BaseException | None = None
    quarantine: _QuarantinedDescriptorEntry | None = None
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != expected_identity:
            raise RuntimeError(f"{label} changed inode while opening cleanup target")
        mount_id = (
            _descriptor_mount_id(parent_descriptor)
            if expected_mount_id is None
            else expected_mount_id
        )
        if _descriptor_mount_id(descriptor) != mount_id:
            raise ValueError(f"{label} is or crossed a mount point")
        if stat.S_ISDIR(opened.st_mode):
            try:
                _require_directory_tree_mount_id(
                    descriptor,
                    expected_mount_id=mount_id,
                    label=label,
                )
            except ValueError as exc:
                raise ValueError(f"{label} cleanup crossed a mount: {exc}") from exc

        quarantine = _quarantine_descriptor_entry(
            parent_descriptor,
            entry_name,
            expected_identity=expected_identity,
            label=label,
        )
        if stat.S_ISDIR(opened.st_mode):
            deletion_budget = _deletion_budget or _ArtifactTreeTraversalBudget(
                label=f"Artifact cleanup {label}"
            )
            _remove_directory_descriptor_contents(
                descriptor,
                expected_mount_id=mount_id,
                label=label,
                traversal_budget=deletion_budget,
                depth=_depth,
            )
        quarantine_identity = _optional_descriptor_entry_identity(
            parent_descriptor,
            quarantine.quarantine_name,
        )
        if quarantine_identity != expected_identity:
            raise RuntimeError(
                f"{label} quarantine changed inode before final deletion"
            )
        if stat.S_ISDIR(opened.st_mode):
            os.rmdir(quarantine.quarantine_name, dir_fd=parent_descriptor)
        else:
            os.unlink(quarantine.quarantine_name, dir_fd=parent_descriptor)
        if (
            _optional_descriptor_entry_identity(
                parent_descriptor,
                quarantine.quarantine_name,
            )
            is not None
        ):
            raise RuntimeError(f"{label} quarantine still exists after deletion")
        if (
            _optional_descriptor_entry_identity(parent_descriptor, entry_name)
            is not None
        ):
            raise RuntimeError(
                f"{label} original name became occupied; replacement preserved"
            )
    except BaseException as exc:
        operation_error = exc
        if quarantine is not None:
            restoration = _quarantine_recovery_note(
                parent_descriptor,
                quarantine.quarantine_name,
                quarantine.original_name,
                preserved_identity=quarantine.identity,
                label=label,
            )
            exc.add_note(f"Cleanup quarantine recovery: {restoration}")

    if close_descriptor:
        try:
            os.close(descriptor)
        except BaseException as close_error:
            if operation_error is None:
                operation_error = close_error
            else:
                operation_error.add_note(
                    f"Cleanup descriptor close also failed: {close_error}"
                )
    if operation_error is not None:
        raise operation_error


def _remove_directory_descriptor_contents(
    descriptor: int,
    *,
    expected_mount_id: int,
    label: str,
    relative_path: str = ".",
    traversal_budget: _ArtifactTreeTraversalBudget | None = None,
    depth: int = 0,
) -> None:
    """Delete one held tree while checking every entry's mount and inode."""

    if traversal_budget is None:
        traversal_budget = _ArtifactTreeTraversalBudget(
            label=f"Artifact cleanup {label}"
        )
    try:
        _require_directory_tree_mount_id(
            descriptor,
            expected_mount_id=expected_mount_id,
            label=f"Artifact cleanup {label}",
            relative_path=relative_path,
        )
    except ValueError as exc:
        raise ValueError(
            f"Artifact cleanup crossed a mount at {label}:{relative_path}: {exc}"
        ) from exc
    traversal_budget.consume(
        relative_path=relative_path,
        depth=depth,
    )
    names = traversal_budget.sorted_child_names(
        descriptor,
        relative_path=relative_path,
    )
    os.fchmod(descriptor, 0o700)
    steps: list[tuple[str, Callable[[], None]]] = []
    for name in names:
        child_relative = name if relative_path == "." else f"{relative_path}/{name}"
        steps.append(
            (
                f"Artifact cleanup failed at {label}:{child_relative}",
                partial(
                    _remove_directory_descriptor_child,
                    descriptor,
                    name,
                    expected_mount_id=expected_mount_id,
                    label=label,
                    child_relative=child_relative,
                    traversal_budget=traversal_budget,
                    child_depth=depth + 1,
                ),
            )
        )
    _run_cleanup_steps(
        steps,
        label=f"Artifact directory cleanup failed for {label}:{relative_path}",
    )


def _remove_directory_descriptor_child(
    descriptor: int,
    name: str,
    *,
    expected_mount_id: int,
    label: str,
    child_relative: str,
    traversal_budget: _ArtifactTreeTraversalBudget,
    child_depth: int,
) -> None:
    """Remove one child while retaining its exact operation error through close."""

    metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    if stat.S_ISDIR(metadata.st_mode):
        traversal_budget.require_depth(
            relative_path=child_relative,
            depth=child_depth,
        )
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    else:
        traversal_budget.consume(
            relative_path=child_relative,
            depth=child_depth,
            byte_count=metadata.st_size,
        )
        o_path = getattr(os, "O_PATH", None)
        if o_path is None:  # pragma: no cover - official targets are Linux
            raise RuntimeError("Safe artifact cleanup requires Linux O_PATH")
        flags = o_path | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    child = -1
    operation_error: BaseException | None = None
    try:
        child = os.open(name, flags, dir_fd=descriptor)
        opened = os.fstat(child)
        child_identity = (opened.st_dev, opened.st_ino)
        if child_identity != (metadata.st_dev, metadata.st_ino):
            raise RuntimeError(
                f"Artifact cleanup entry changed inode at {label}:{child_relative}"
            )
        if _descriptor_mount_id(child) != expected_mount_id:
            raise ValueError(
                f"Artifact cleanup crossed a mount at {label}:{child_relative}"
            )
        _remove_descriptor_entry(
            descriptor,
            name,
            expected_identity=child_identity,
            expected_mount_id=expected_mount_id,
            source_descriptor=child,
            label=f"artifact cleanup entry {label}:{child_relative}",
            _deletion_budget=traversal_budget,
            _depth=child_depth,
        )
    except BaseException as error:
        operation_error = error
        raise
    finally:
        if child >= 0:
            owned_child = child
            child = -1
            _run_cleanup_steps(
                [
                    (
                        f"Artifact cleanup descriptor close failed at {label}:{child_relative}",
                        partial(os.close, owned_child),
                    )
                ],
                primary_error=operation_error,
                label="Artifact cleanup descriptor cleanup failed",
            )


def _create_artifact_backup(
    bound_artifact: _BoundArtifact,
    *,
    artifact_identity: tuple[int, int],
) -> _ArtifactBackup:
    """Create and bind one private rollback directory beside its target."""

    parent = bound_artifact.target_entry.parent
    for _ in range(128):
        directory_name = f".joint-rigger.rollback-{secrets.token_hex(12)}"
        try:
            os.mkdir(directory_name, mode=0o700, dir_fd=parent.descriptor)
        except FileExistsError:  # pragma: no cover - cryptographic collision
            continue
        except BaseException as mkdir_error:
            _note_ambiguous_directory_creation(
                mkdir_error,
                parent,
                directory_name,
                label="Artifact rollback directory",
            )
            raise
        descriptor = -1
        created_identity: tuple[int, int] | None = None
        try:
            descriptor = _open_child_directory_descriptor(
                parent.descriptor,
                directory_name,
            )
            created = os.fstat(descriptor)
            created_identity = (created.st_dev, created.st_ino)
            lexical_identity = _optional_descriptor_entry_identity(
                parent.descriptor,
                directory_name,
            )
            if lexical_identity != created_identity:
                raise RuntimeError(
                    "Artifact rollback directory changed inode while it was bound: "
                    f"{parent.opened_path / directory_name}"
                )
            directory = _BoundDirectory(
                locator_path=parent.locator_path / directory_name,
                opened_path=parent.opened_path / directory_name,
                descriptor=descriptor,
                identity=created_identity,
            )
            return _ArtifactBackup(
                bound_artifact=bound_artifact,
                directory=directory,
                directory_name=directory_name,
                artifact_entry=_BoundEntry(parent=directory, name="artifact"),
                artifact_identity=artifact_identity,
            )
        except BaseException as creation_error:
            cleanup_descriptor = descriptor
            descriptor = -1
            cleanup_steps: list[tuple[str, Callable[[], None]]] = [
                (
                    "Created artifact rollback directory cleanup also failed",
                    partial(
                        _remove_created_directory_if_bound,
                        parent,
                        directory_name,
                        descriptor=cleanup_descriptor,
                        identity=created_identity,
                        label=(
                            "uncommitted artifact rollback directory "
                            f"{parent.opened_path / directory_name}"
                        ),
                    ),
                )
            ]
            if cleanup_descriptor >= 0:
                cleanup_steps.append(
                    (
                        "Created artifact rollback directory descriptor close also "
                        "failed",
                        partial(os.close, cleanup_descriptor),
                    )
                )
            _run_cleanup_steps(
                cleanup_steps,
                primary_error=creation_error,
                label="Created artifact rollback directory cleanup failed",
            )
            raise
    raise RuntimeError("Could not allocate a unique artifact rollback directory")


def _require_artifact_backup_ready(
    backup: _ArtifactBackup,
    *,
    operation: str,
) -> None:
    """Require the exact saved inode and a vacant restoration destination."""

    payload_identity = _optional_bound_entry_identity(backup.artifact_entry)
    target_identity = _optional_bound_entry_identity(backup.bound_artifact.target_entry)
    if payload_identity != backup.artifact_identity or target_identity is not None:
        raise RuntimeError(
            f"Artifact backup state changed {operation}; refusing rollback restore "
            f"for {backup.bound_artifact.artifact.label}: expected payload "
            f"{backup.artifact_identity}, found {payload_identity}; target identity "
            f"is {target_identity}. Backup entries were preserved"
        )


def _restore_artifact_backup(backup: _ArtifactBackup) -> None:
    """Restore one exact rollback payload with pre/post move verification."""

    _require_artifact_backup_ready(backup, operation="before rollback restore")
    try:
        _replace_entry_with_directory_mode_guard(
            backup.artifact_entry,
            backup.bound_artifact.target_entry,
            expected_identity=backup.artifact_identity,
            label=(
                f"artifact backup restore for {backup.bound_artifact.artifact.label}"
            ),
        )
    except BaseException as move_error:
        payload_after = _optional_bound_entry_identity(backup.artifact_entry)
        target_after = _optional_bound_entry_identity(
            backup.bound_artifact.target_entry
        )
        if payload_after is None and target_after == backup.artifact_identity:
            raise
        raise RuntimeError(
            "Artifact backup restore was interrupted in an ambiguous state for "
            f"{backup.bound_artifact.artifact.label}; expected payload "
            f"{backup.artifact_identity}, found backup={payload_after}, "
            f"target={target_after}. Entries were preserved"
        ) from move_error

    payload_after = _optional_bound_entry_identity(backup.artifact_entry)
    target_after = _optional_bound_entry_identity(backup.bound_artifact.target_entry)
    if payload_after is not None or target_after != backup.artifact_identity:
        recovery = "mismatched entries were preserved"
        if payload_after is None and target_after not in {
            None,
            backup.artifact_identity,
        }:
            # The move returned successfully but carried a substituted source.
            # Put that exact payload back under the held backup directory when
            # the now-vacant source name still permits a no-replace recovery.
            displaced_identity = target_after
            try:
                _rename_descriptor_entry_noreplace(
                    backup.bound_artifact.target_entry.parent.descriptor,
                    backup.bound_artifact.target_entry.name,
                    backup.artifact_entry.parent.descriptor,
                    backup.artifact_entry.name,
                    label="mismatched artifact backup restore recovery",
                )
            except BaseException as recovery_error:
                recovery = (
                    "mismatched restore recovery failed; entries were preserved "
                    f"({type(recovery_error).__name__}: {recovery_error})"
                )
            else:
                recovered_payload = _optional_bound_entry_identity(
                    backup.artifact_entry
                )
                recovered_target = _optional_bound_entry_identity(
                    backup.bound_artifact.target_entry
                )
                if recovered_payload == displaced_identity and recovered_target is None:
                    recovery = "mismatched payload was preserved in the backup"
                else:
                    recovery = (
                        "mismatched restore recovery ended ambiguously; entries "
                        "were preserved"
                    )
        raise RuntimeError(
            "Artifact backup state changed after rollback restore for "
            f"{backup.bound_artifact.artifact.label}; expected target "
            f"{backup.artifact_identity}, found backup={payload_after}, "
            f"target={target_after}; {recovery}"
        )


def _remove_backup_directory(
    backup: _ArtifactBackup,
) -> None:
    """Remove one rollback payload and its directory through held descriptors."""

    _remove_bound_entry(
        backup.artifact_entry,
        expected_identity=backup.artifact_identity,
    )
    _remove_descriptor_entry(
        backup.bound_artifact.target_entry.parent.descriptor,
        backup.directory_name,
        expected_identity=backup.directory.identity,
        source_descriptor=backup.directory.descriptor,
        label=f"artifact rollback directory {backup.directory.opened_path}",
    )


@contextmanager
def _publication_target_locks(
    targets: Iterable[Path],
    *,
    cleanup_state: _PublicationCleanupState | None = None,
) -> Iterator[list[_LockedTarget]]:
    """Lock every physical target parent or reject a concurrent publication.

    Held directory descriptors are deduplicated by device/inode identity and
    locked in deterministic identity order. All final entries in one physical
    parent therefore serialize, including paths reached through aliases, while
    transactions in disjoint physical parents remain concurrent.
    """

    with _bound_publication_targets(
        targets,
        cleanup_state=cleanup_state,
    ) as bound_targets:
        parents_by_identity: dict[tuple[int, int], _BoundDirectory] = {}
        for target in bound_targets:
            parents_by_identity.setdefault(
                target.entry.parent.identity,
                target.entry.parent,
            )
        ordered_parents = [
            parents_by_identity[identity] for identity in sorted(parents_by_identity)
        ]
        locked_parents: list[_BoundDirectory] = []
        active_error: BaseException | None = None
        try:
            for parent in ordered_parents:
                try:
                    fcntl.flock(
                        parent.descriptor,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError as exc:
                    raise ConcurrentArtifactPublicationError(
                        "Another artifact publication is already targeting parent "
                        f"{parent.locator_path}"
                    ) from exc
                locked_parents.append(parent)
            yield bound_targets
        except BaseException as error:
            active_error = error
            raise
        finally:
            _run_cleanup_steps(
                (
                    (
                        f"Publication lock cleanup failed for {parent.locator_path}",
                        partial(fcntl.flock, parent.descriptor, fcntl.LOCK_UN),
                    )
                    for parent in reversed(locked_parents)
                ),
                primary_error=active_error,
                cleanup_state=cleanup_state,
                label="Publication lock cleanup failed",
            )


@contextmanager
def _bound_publication_targets(
    targets: Iterable[Path],
    *,
    cleanup_state: _PublicationCleanupState | None = None,
) -> Iterator[list[_LockedTarget]]:
    """Open all target parents, reject physical duplicates, and sort locks."""

    bound_targets: list[_LockedTarget] = []
    bound_parents: list[_BoundDirectory] = []
    active_error: BaseException | None = None
    try:
        for raw_target in targets:
            requested_path = _absolute_lexical_path(raw_target)
            parent = _open_bound_directory(requested_path.parent)
            bound_parents.append(parent)
            identity = _physical_entry_identity(
                parent.identity,
                entry_name=requested_path.name,
            )
            bound_targets.append(
                _LockedTarget(
                    requested_path=requested_path,
                    entry=_BoundEntry(parent=parent, name=requested_path.name),
                    identity=identity,
                )
            )

        seen: dict[bytes, Path] = {}
        for target in bound_targets:
            previous = seen.get(target.identity)
            if previous is not None:
                raise ValueError(
                    "Duplicate physical transaction target: "
                    f"{target.requested_path} aliases {previous}"
                )
            seen[target.identity] = target.requested_path
        yield sorted(bound_targets, key=lambda target: target.identity)
    except BaseException as error:
        active_error = error
        raise
    finally:
        _run_cleanup_steps(
            (
                (
                    f"Target parent descriptor cleanup failed for "
                    f"{parent.locator_path}",
                    partial(os.close, parent.descriptor),
                )
                for parent in bound_parents
            ),
            primary_error=active_error,
            cleanup_state=cleanup_state,
            label="Target parent descriptor cleanup failed",
        )


def _physical_directory_identity(
    parent: Path,
    *,
    descriptor: int | None = None,
) -> tuple[int, int]:
    """Return one directory's stable filesystem device/inode identity."""

    metadata = os.fstat(descriptor) if descriptor is not None else parent.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(
            f"Publication target parent is not a directory: {parent}"
        )
    return metadata.st_dev, metadata.st_ino


def _physical_entry_identity(
    parent_identity: tuple[int, int],
    *,
    entry_name: str,
) -> bytes:
    """Encode a physical parent identity and exact lexical entry name."""

    device, inode = parent_identity
    return (
        str(device).encode("ascii")
        + b"\0"
        + str(inode).encode("ascii")
        + b"\0"
        + os.fsencode(entry_name)
    )


def _remove_bound_entry_if_identity(
    entry: _BoundEntry,
    identity: tuple[int, int],
) -> None:
    """Remove an fd-relative cleanup entry only while it remains owned."""

    if _optional_bound_entry_identity(entry) != identity:
        return
    try:
        _remove_descriptor_entry(
            entry.parent.descriptor,
            entry.name,
            expected_identity=identity,
            label=f"owned cleanup entry {entry.path}",
        )
    except FileNotFoundError:
        pass


def _remove_bound_entry_if_owned(
    entry: _BoundEntry,
    identity: tuple[int, int],
) -> None:
    """Remove one file or tree only while its root inode remains owned."""

    if _optional_bound_entry_identity(entry) != identity:
        return
    _remove_descriptor_entry(
        entry.parent.descriptor,
        entry.name,
        expected_identity=identity,
        label=f"owned cleanup entry {entry.path}",
    )


def remove_artifact(path: Path) -> None:
    """Remove one file, symlink, or directory through a bound parent inode."""

    requested_path = _absolute_lexical_path(path)
    if requested_path.parent == requested_path:
        raise ValueError(f"Artifact path must name a directory entry: {path}")
    try:
        parent = _open_bound_directory(requested_path.parent)
    except (FileNotFoundError, NotADirectoryError):
        return

    active_error: BaseException | None = None
    try:
        entry = _BoundEntry(parent=parent, name=requested_path.name)
        identity = _optional_bound_entry_identity(entry)
        if identity is None:
            return
        _remove_descriptor_entry(
            parent.descriptor,
            entry.name,
            expected_identity=identity,
            label=f"artifact cleanup entry {entry.path}",
        )
    except BaseException as error:
        active_error = error
        raise
    finally:
        _run_cleanup_steps(
            [
                (
                    f"Artifact cleanup parent descriptor close failed for "
                    f"{parent.locator_path}",
                    partial(os.close, parent.descriptor),
                )
            ],
            primary_error=active_error,
            label="Artifact cleanup parent descriptor cleanup failed",
        )


def _target_items(
    targets: JointRiggerArtifactTargets,
) -> list[tuple[str, Path, bool]]:
    items = [
        ("output_path", targets.output_path, False),
        ("diagnostics_path", targets.diagnostics_path, False),
        ("result_path", targets.result_path, False),
    ]
    if targets.sidecar_path is not None:
        items.append(("sidecar_path", targets.sidecar_path, True))
    return items


def _validate_sidecar_parent(targets: JointRiggerArtifactTargets) -> None:
    if targets.sidecar_path is None:
        return
    if targets.sidecar_path.parent.expanduser().resolve(
        strict=False
    ) != targets.output_path.parent.expanduser().resolve(strict=False):
        raise ValueError("sidecar_path must share output_path's parent directory")


def _validate_caller_publication_layout(
    targets: JointRiggerArtifactTargets,
) -> None:
    """Keep final caller targets distinct from internal staging metadata."""

    if targets.publication_output_path != targets.output_path:
        raise ValueError(
            "Caller-facing publication_output_path must equal output_path; "
            "only internally staged targets may use a different physical path"
        )
    if targets.publication_diagnostics_path != targets.diagnostics_path:
        raise ValueError(
            "Caller-facing publication_diagnostics_path must equal "
            "diagnostics_path; only internally staged targets may use a different "
            "physical path"
        )
    if targets.publication_result_path != targets.result_path:
        raise ValueError(
            "Caller-facing publication_result_path must equal result_path; "
            "only internally staged targets may use a different physical path"
        )
    if targets.publication_sidecar_path != targets.sidecar_path:
        raise ValueError(
            "Caller-facing publication_sidecar_path must equal sidecar_path; "
            "only internally staged targets may use a different physical path"
        )


def _target_paths(targets: JointRiggerArtifactTargets) -> list[Path]:
    return [path for _, path, _ in _target_items(targets)]


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise RuntimeError(
            f"Staged {label} is missing or not a regular file: {path}"
        ) from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"Staged {label} is missing or not a regular file: {path}")
    if metadata.st_nlink != 1:
        raise RuntimeError(f"Staged {label} must have exactly one hard link: {path}")
    return metadata


def _reserve_backend_staging_name(
    target: Path,
    *,
    descriptor_owned: bool = True,
) -> tuple[Path, _StagingCleanupReservation]:
    """Create an absent backend path with the strongest compatible ownership."""

    if not descriptor_owned:
        return _reserve_unbound_backend_staging_name(target)
    owner_path, reservation = _create_sidecar_owner_reservation(
        target.parent,
        target_name=target.name,
    )
    return owner_path / target.name, reservation


def _reserve_unbound_backend_staging_name(
    target: Path,
) -> tuple[Path, _StagingCleanupReservation]:
    """Reserve an absent sibling root without adopting its future inode."""

    target.parent.mkdir(parents=True, exist_ok=True)
    parent = _open_bound_directory(target.parent)
    try:
        for _ in range(128):
            name = f".{target.stem}.stage-{secrets.token_hex(12)}{target.suffix}"
            flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            try:
                descriptor = os.open(name, flags, 0o600, dir_fd=parent.descriptor)
            except FileExistsError:  # pragma: no cover - cryptographic collision
                continue
            except BaseException as acquisition_error:
                _note_ambiguous_file_creation(
                    acquisition_error,
                    parent,
                    name,
                    label="Backend staging placeholder",
                )
                raise
            entry = _BoundEntry(parent=parent, name=name)
            identity: tuple[int, int] | None = None
            placeholder_removed = False
            try:
                metadata = os.fstat(descriptor)
                identity = (metadata.st_dev, metadata.st_ino)
                _remove_descriptor_entry(
                    parent.descriptor,
                    name,
                    expected_identity=identity,
                    source_descriptor=descriptor,
                    label=f"backend staging placeholder {entry.path}",
                )
                placeholder_removed = True
                owned_descriptor = descriptor
                # Never retry a failed close; Linux may already have reused the fd.
                descriptor = -1
                os.close(owned_descriptor)
                if _bound_entry_exists(entry):
                    raise RuntimeError(
                        "Backend staging placeholder name was reoccupied after "
                        f"descriptor-bound removal; replacement preserved: {entry.path}"
                    )
                _require_bound_directory_unchanged(parent)
                promotion_state = _StagingPromotionState()
                return (
                    parent.locator_path / name,
                    _StagingCleanupReservation(
                        parent=parent,
                        name=name,
                        publication_name=target.name,
                        promotion_state=promotion_state,
                    ),
                )
            except BaseException as reservation_error:

                def remove_placeholder(
                    entry: _BoundEntry = entry,
                    descriptor: int = descriptor,
                    placeholder_was_removed: bool = placeholder_removed,
                ) -> None:
                    nonlocal identity
                    if placeholder_was_removed:
                        if _bound_entry_exists(entry):
                            raise RuntimeError(
                                "Backend staging placeholder name was reoccupied; "
                                f"replacement preserved: {entry.path}"
                            )
                        return
                    if identity is None:
                        if descriptor < 0:
                            raise RuntimeError(
                                "Backend staging placeholder cleanup could not bind "
                                f"the created inode: {entry.path}"
                            )
                        created = os.fstat(descriptor)
                        identity = (created.st_dev, created.st_ino)
                    lexical_identity = _optional_bound_entry_identity(entry)
                    if lexical_identity is None:
                        if descriptor < 0:
                            raise RuntimeError(
                                "Backend staging placeholder disappeared from its "
                                "reserved name, but its owned inode could not be "
                                f"verified; preservation is required: {entry.path}"
                            )
                        retained = os.fstat(descriptor)
                        retained_identity = (retained.st_dev, retained.st_ino)
                        if (
                            not stat.S_ISREG(retained.st_mode)
                            or retained_identity != identity
                        ):
                            raise RuntimeError(
                                "Backend staging placeholder disappeared from its "
                                "reserved name and its retained descriptor no longer "
                                f"identifies the owned file: {entry.path}"
                            )
                        if retained.st_nlink == 0:
                            return
                        raise RuntimeError(
                            "Backend staging placeholder disappeared from its reserved "
                            "name; descriptor-owned inode remains linked elsewhere and "
                            f"was preserved: {entry.path}"
                        )
                    if lexical_identity != identity:
                        raise RuntimeError(
                            "Backend staging placeholder changed inode; replacement "
                            f"preserved: {entry.path}"
                        )
                    _remove_descriptor_entry(
                        entry.parent.descriptor,
                        entry.name,
                        expected_identity=identity,
                        source_descriptor=descriptor,
                        label=f"backend staging placeholder {entry.path}",
                    )

                cleanup_steps: list[tuple[str, Callable[[], None]]] = [
                    (
                        f"Backend staging placeholder cleanup failed for {entry.path}",
                        remove_placeholder,
                    )
                ]
                if descriptor >= 0:
                    owned_descriptor = descriptor
                    descriptor = -1
                    cleanup_steps.append(
                        (
                            f"Backend staging placeholder descriptor cleanup failed "
                            f"for {entry.path}",
                            partial(os.close, owned_descriptor),
                        )
                    )
                _run_cleanup_steps(
                    cleanup_steps,
                    primary_error=reservation_error,
                    label="Backend staging reservation cleanup failed",
                )
                raise
        raise RuntimeError("Could not allocate a unique backend staging name")
    except BaseException as reservation_error:
        _run_cleanup_steps(
            [
                (
                    f"Backend staging parent descriptor cleanup failed for "
                    f"{parent.locator_path}",
                    lambda: os.close(parent.descriptor),
                )
            ],
            primary_error=reservation_error,
            label="Backend staging parent descriptor cleanup failed",
        )
        raise


def _create_sidecar_owner_reservation(
    parent_path: Path,
    *,
    target_name: str,
) -> tuple[Path, _StagingCleanupReservation]:
    """Create a private staging owner under a retained parent directory."""

    parent_path.mkdir(parents=True, exist_ok=True)
    parent = _open_bound_directory(parent_path)
    try:
        for _ in range(128):
            name = f".{target_name}.stage-{secrets.token_hex(12)}"
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent.descriptor)
            except FileExistsError:  # pragma: no cover - cryptographic collision
                continue
            except BaseException as mkdir_error:
                _note_ambiguous_directory_creation(
                    mkdir_error,
                    parent,
                    name,
                    label="Staging owner",
                )
                raise
            entry = _BoundEntry(parent=parent, name=name)
            descriptor = -1
            identity: tuple[int, int] | None = None
            try:
                descriptor = _open_child_directory_descriptor(
                    parent.descriptor,
                    name,
                )
                metadata = os.fstat(descriptor)
                identity = (metadata.st_dev, metadata.st_ino)
                lexical_identity = _optional_descriptor_entry_identity(
                    parent.descriptor,
                    name,
                )
                if lexical_identity != identity:
                    raise RuntimeError(
                        f"Staging owner changed inode while it was bound: {entry.path}"
                    )
                _require_bound_directory_unchanged(parent)
                owned_descriptor = descriptor
                # Transfer the descriptor into the reservation.  Keeping the
                # inode open prevents dev/inode reuse from forging ownership.
                descriptor = -1
            except BaseException as reservation_error:
                cleanup_descriptor = descriptor
                descriptor = -1
                cleanup_steps: list[tuple[str, Callable[[], None]]] = [
                    (
                        f"Staging owner cleanup failed for {entry.path}",
                        partial(
                            _remove_created_directory_if_bound,
                            parent,
                            name,
                            descriptor=cleanup_descriptor,
                            identity=identity,
                            label=f"uncommitted staging owner {entry.path}",
                        ),
                    )
                ]
                if cleanup_descriptor >= 0:
                    cleanup_steps.append(
                        (
                            f"Staging owner descriptor cleanup failed for {entry.path}",
                            partial(os.close, cleanup_descriptor),
                        )
                    )
                _run_cleanup_steps(
                    cleanup_steps,
                    primary_error=reservation_error,
                    label="Staging owner cleanup failed",
                )
                raise
            return (
                parent.locator_path / name,
                _StagingCleanupReservation(
                    parent=parent,
                    name=name,
                    owned_identity=identity,
                    owned_descriptor=owned_descriptor,
                    is_owner_directory=True,
                    payload_name=target_name,
                    payload_promotion_state=_StagingPromotionState(),
                ),
            )
        raise RuntimeError("Could not allocate a unique staging owner")
    except BaseException as reservation_error:
        _run_cleanup_steps(
            [
                (
                    f"Sidecar staging parent descriptor cleanup failed for "
                    f"{parent.locator_path}",
                    lambda: os.close(parent.descriptor),
                )
            ],
            primary_error=reservation_error,
            label="Sidecar staging parent descriptor cleanup failed",
        )
        raise


def _preserve_unbound_staging_reservation(
    reservation: _StagingCleanupReservation,
    parent: _BoundDirectory,
) -> None:
    """Report a backend-known name that never gained descriptor-bound ownership."""

    entry = _BoundEntry(parent=parent, name=reservation.name)
    if _optional_bound_entry_identity(entry) is None:
        return
    raise RuntimeError(
        "Staged artifact has no descriptor-bound cleanup identity; "
        f"unpredictable backend-known name preserved at {entry.path}"
    )


def _cleanup_staging_owner_payload(
    reservation: _StagingCleanupReservation,
) -> None:
    """Remove or account for the exact payload retained inside one owner."""

    if not reservation.payload_binding_attempted:
        return
    identity = reservation.payload_identity
    descriptor = reservation.payload_descriptor
    payload_name = reservation.payload_name
    if identity is None or descriptor < 0 or payload_name is None:
        if (
            payload_name is not None
            and reservation.owned_descriptor >= 0
            and _optional_descriptor_entry_identity(
                reservation.owned_descriptor,
                payload_name,
            )
            is None
        ):
            return
        raise RuntimeError(
            "Staging payload cleanup ownership could not be bound; descriptor-owned "
            "owner preserved"
        )

    opened = os.fstat(descriptor)
    expected_type = stat.S_ISDIR if reservation.payload_is_directory else stat.S_ISREG
    if (
        not expected_type(opened.st_mode)
        or (
            opened.st_dev,
            opened.st_ino,
        )
        != identity
    ):
        raise RuntimeError("Staging payload cleanup descriptor changed inode")

    current_identity = _optional_descriptor_entry_identity(
        reservation.owned_descriptor,
        payload_name,
    )
    if current_identity is not None and current_identity != identity:
        raise RuntimeError(
            "Staging payload changed inode; replacement and descriptor-owned "
            f"payload preserved at {reservation.parent.opened_path / reservation.name}"
        )
    if current_identity == identity:
        if not reservation.payload_is_directory and opened.st_nlink != 1:
            raise RuntimeError(
                "Staging payload gained additional links; linked names preserved "
                f"for manual recovery: {reservation.parent.opened_path / reservation.name / payload_name}"
            )
        _remove_descriptor_entry(
            reservation.owned_descriptor,
            payload_name,
            expected_identity=identity,
            source_descriptor=descriptor,
            label=(
                "owned staging payload "
                f"{reservation.parent.opened_path / reservation.name / payload_name}"
            ),
        )
        after_removal = os.fstat(descriptor)
        if after_removal.st_nlink != 0:
            raise RuntimeError(
                "Staging payload retained an unauthorized linked name after "
                f"cleanup: {reservation.parent.opened_path / reservation.name / payload_name}"
            )
        return

    if opened.st_nlink == 0:
        return
    publication_identity = _optional_descriptor_entry_identity(
        reservation.parent.descriptor,
        payload_name,
    )
    promotion_identity = (
        None
        if reservation.payload_promotion_state is None
        else reservation.payload_promotion_state.committed_identity
    )
    exact_direct_promotion = (
        promotion_identity == identity and publication_identity == identity
    )
    if exact_direct_promotion:
        if not reservation.payload_is_directory and opened.st_nlink != 1:
            raise RuntimeError(
                "Promoted staging payload gained additional links; linked names "
                f"preserved for manual recovery: {reservation.parent.opened_path / payload_name}"
            )
        return
    raise RuntimeError(
        "Descriptor-owned staging payload disappeared without an exact recorded "
        "promotion; linked inode was preserved elsewhere: "
        f"{reservation.parent.opened_path / reservation.name / payload_name}"
    )


def _remove_staging_reservation_entry(
    reservation: _StagingCleanupReservation,
    parent: _BoundDirectory,
    *,
    require_owned_name: bool = False,
    require_present: bool = False,
    require_published_absence: bool = False,
) -> None:
    """Remove one reserved name through one held parent when identity-bound."""

    identity = reservation.owned_identity
    if identity is None:
        return
    descriptor = reservation.owned_descriptor
    if descriptor < 0:
        raise RuntimeError("Staging cleanup ownership descriptor is unavailable")
    opened = os.fstat(descriptor)
    expected_type = stat.S_ISDIR if reservation.is_owner_directory else stat.S_ISREG
    if not expected_type(opened.st_mode) or (opened.st_dev, opened.st_ino) != identity:
        raise RuntimeError("Staging cleanup ownership descriptor changed inode")
    entry = _BoundEntry(parent=parent, name=reservation.name)
    current_identity = _optional_bound_entry_identity(entry)
    if current_identity is None and require_published_absence:
        publication_name = reservation.publication_name
        publication_identity = (
            None
            if publication_name is None
            else _optional_descriptor_entry_identity(
                parent.descriptor,
                publication_name,
            )
        )
        promotion_identity = (
            None
            if reservation.promotion_state is None
            else reservation.promotion_state.committed_identity
        )
        promotion_committed = (
            reservation.promotion_state is not None and promotion_identity == identity
        )
        if opened.st_nlink != 0 and not (
            promotion_committed
            and publication_identity == identity
            and opened.st_nlink == 1
        ):
            raise RuntimeError(
                "Bound staged artifact disappeared without reaching its exact "
                "publication target; held inode preserved elsewhere: "
                f"{entry.path}"
            )
        return
    if require_present and current_identity is None:
        raise RuntimeError(
            "Staging owner disappeared from its reserved name; "
            f"descriptor-bound owner may remain preserved elsewhere: {entry.path}"
        )
    if require_owned_name and current_identity not in (None, identity):
        raise RuntimeError(
            f"Staging owner changed inode; replacement preserved at {entry.path}"
        )
    if current_identity == identity:
        if not reservation.is_owner_directory and opened.st_nlink != 1:
            raise RuntimeError(
                "Bound staged artifact gained additional links; linked names "
                f"preserved for manual recovery: {entry.path}"
            )
        if reservation.is_owner_directory and parent is reservation.parent:
            _cleanup_staging_owner_payload(reservation)
        _remove_descriptor_entry(
            parent.descriptor,
            reservation.name,
            expected_identity=identity,
            label=f"owned cleanup entry {entry.path}",
            source_descriptor=descriptor,
        )
        if not reservation.is_owner_directory:
            after_removal = os.fstat(descriptor)
            if after_removal.st_nlink != 0:
                raise RuntimeError(
                    "Bound staged artifact retained an unauthorized linked "
                    f"name after cleanup: {entry.path}"
                )


def _cleanup_staging_reservations(
    reservations: tuple[_StagingCleanupReservation, ...],
    *,
    primary_error: BaseException | None = None,
) -> None:
    """Remove reserved names through original and current lexical parents."""

    failures: list[tuple[str, BaseException]] = []
    for reservation in reservations:
        if reservation.closed:
            continue
        reservation.binding_revoked = True
        if reservation.owned_identity is None:
            failures.extend(
                _collect_cleanup_failures(
                    [
                        (
                            f"Unbound staged artifact preservation required for "
                            f"{reservation.parent.locator_path / reservation.name}",
                            partial(
                                _preserve_unbound_staging_reservation,
                                reservation,
                                reservation.parent,
                            ),
                        )
                    ]
                )
            )
        if reservation.owned_identity is not None:
            failures.extend(
                _collect_cleanup_failures(
                    [
                        (
                            f"Original staged artifact cleanup failed for "
                            f"{reservation.parent.locator_path / reservation.name}",
                            partial(
                                _remove_staging_reservation_entry,
                                reservation,
                                reservation.parent,
                                require_owned_name=True,
                                require_present=reservation.is_owner_directory,
                                require_published_absence=(
                                    not reservation.is_owner_directory
                                ),
                            ),
                        )
                    ]
                )
            )

        current_parent: _BoundDirectory | None = None
        try:
            current_parent = _open_bound_directory(reservation.parent.locator_path)
        except (FileNotFoundError, NotADirectoryError):
            pass
        except BaseException as error:
            failures.append(
                (
                    f"Current staged parent open failed for "
                    f"{reservation.parent.locator_path}",
                    error,
                )
            )
        if current_parent is not None:
            if (
                reservation.owned_identity is None
                and current_parent.identity != reservation.parent.identity
            ):
                failures.extend(
                    _collect_cleanup_failures(
                        [
                            (
                                f"Unbound staged artifact preservation required for "
                                f"{reservation.parent.locator_path / reservation.name}",
                                partial(
                                    _preserve_unbound_staging_reservation,
                                    reservation,
                                    current_parent,
                                ),
                            )
                        ]
                    )
                )
            elif (
                reservation.owned_identity is not None
                and current_parent.identity != reservation.parent.identity
            ):
                failures.extend(
                    _collect_cleanup_failures(
                        [
                            (
                                f"Current staged artifact cleanup failed for "
                                f"{reservation.parent.locator_path / reservation.name}",
                                partial(
                                    _remove_staging_reservation_entry,
                                    reservation,
                                    current_parent,
                                ),
                            )
                        ]
                    )
                )
            _close_cleanup_descriptor(
                current_parent.descriptor,
                failures,
                context=(
                    "Current staged parent descriptor cleanup failed for "
                    f"{current_parent.locator_path}"
                ),
            )

        _invalidate_staging_promotion_state(reservation.payload_promotion_state)
        if reservation.payload_descriptor >= 0:
            payload_descriptor = reservation.payload_descriptor
            reservation.payload_descriptor = -1
            _close_cleanup_descriptor(
                payload_descriptor,
                failures,
                context=(
                    "Staged payload ownership descriptor cleanup failed for "
                    f"{reservation.parent.locator_path / reservation.name}"
                ),
            )
        _invalidate_staging_promotion_state(reservation.promotion_state)
        if reservation.owned_descriptor >= 0:
            owned_descriptor = reservation.owned_descriptor
            reservation.owned_descriptor = -1
            _close_cleanup_descriptor(
                owned_descriptor,
                failures,
                context=(
                    "Staged ownership descriptor cleanup failed for "
                    f"{reservation.parent.locator_path / reservation.name}"
                ),
            )
        reservation.closed = _close_cleanup_descriptor(
            reservation.parent.descriptor,
            failures,
            context=(
                "Original staged parent descriptor cleanup failed for "
                f"{reservation.parent.locator_path}"
            ),
        )
        reservation.owned_identity = None
        reservation.payload_identity = None

    _route_cleanup_failures(
        failures,
        primary_error=primary_error,
        label="Staged artifact cleanup failed",
    )


def _invalidate_staging_promotion_state(
    state: _StagingPromotionState | None,
) -> None:
    """Retire move authority before its retained source descriptor is closed."""

    if state is None:
        return
    state.source_identity = None
    state.source_parent_identity = None
    state.source_descriptor = -1
    state.source_is_directory = False
    state.source_tree_sha256 = None
    state.source_mount_id = None
    state.committed_identity = None


def _close_cleanup_descriptor(
    descriptor: int,
    failures: list[tuple[str, BaseException]],
    *,
    context: str,
) -> bool:
    """Relinquish descriptor ownership after exactly one close attempt."""

    try:
        os.close(descriptor)
        return True
    except OSError as exc:
        if exc.errno in {errno.EBADF, errno.EINTR}:
            return True
        failures.append((context, exc))
    except BaseException as exc:
        failures.append((context, exc))
    # Linux may release and immediately reuse the fd even when close reports an
    # error. Ownership is therefore always relinquished after the first attempt.
    return True


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = [
    "CommittedArtifactPublicationCleanupError",
    "ConcurrentArtifactPublicationError",
    "JointRiggerArtifactTargets",
    "StagedArtifact",
    "StagedJointRiggerArtifacts",
    "copy_sidecar_directory",
    "create_staged_artifact_targets",
    "directory_tree_sha256",
    "directory_descriptor_tree_sha256",
    "invalidate_artifact_targets",
    "promote_staged_artifacts",
    "remove_artifact",
    "sidecar_dependency_bundle_sha256",
    "staged_promotion_artifacts",
    "validate_artifact_targets",
]
