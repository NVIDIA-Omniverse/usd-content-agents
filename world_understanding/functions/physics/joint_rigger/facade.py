# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""App-agnostic execution seam for structured USD joint authoring."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, ValidationError

from world_understanding.functions.physics.joint_rigger.artifacts import (
    CommittedArtifactPublicationCleanupError,
    JointRiggerArtifactTargets,
    StagedArtifact,
    StagedJointRiggerArtifacts,
    _descriptor_mount_id,
    _remove_descriptor_entry,
    _require_directory_tree_mount_id,
    _require_present_invariant,
    copy_directory_descriptor_tree,
    copy_sidecar_directory,
    create_staged_artifact_targets,
    directory_descriptor_tree_sha256,
    promote_staged_artifacts,
    sidecar_dependency_bundle_sha256,
    staged_promotion_artifacts,
    validate_artifact_targets,
)
from world_understanding.functions.physics.joint_rigger.opaque_dependencies import (
    OPAQUE_DEPENDENCY_EXTENSIONS as _OPAQUE_DEPENDENCY_EXTENSIONS,
)
from world_understanding.functions.physics.joint_rigger.opaque_dependencies import (
    OpaqueDependencyError,
    materialx_local_references,
    mdl_local_references,
    resolve_local_reference,
    strip_mdl_comments,
)

if TYPE_CHECKING:  # pragma: no cover - runtime imports stay lazy by design
    from world_understanding.functions.physics.joint_rigger.models import (
        ArtifactIdentityV1,
        JointRiggerInputV1,
        JointRiggerResultV1,
    )
    from world_understanding.functions.physics.joint_rigger.reference import (
        _CapturedDependencyIdentityRecord,
    )

_MAX_REPORT_BYTES = 64 * 1024 * 1024
_MAX_OPAQUE_DOCUMENT_BYTES = 16 * 1024 * 1024
_MAX_OPAQUE_DEPENDENCY_FILES = 256
_MAX_OPAQUE_DEPENDENCY_REFERENCES = 4096


class _DuplicateJsonObjectKeyError(ValueError):
    """A backend report contains an ambiguous JSON object."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonObjectKeyError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _validate_model_report_payload(payload: bytes, model_type: Any) -> Any:
    json.loads(payload, object_pairs_hook=_object_without_duplicate_keys)
    return model_type.model_validate_json(payload)


class JointRiggerFacadeError(RuntimeError):
    """Base error for the shared execution seam."""


class JointRiggerBackendUnavailableError(JointRiggerFacadeError):
    """The selected backend or one of its runtime dependencies is unavailable."""


class JointRiggerBackendIncompatibleError(JointRiggerFacadeError):
    """The selected backend does not implement the required facade contract."""


class JointRiggerArtifactError(JointRiggerFacadeError):
    """A backend did not produce one complete, contract-valid artifact set."""


def _attach_cleanup_failure(
    primary_error: BaseException,
    cleanup_error: BaseException,
    *,
    context: str,
) -> None:
    """Attach one cleanup failure and its nested notes to an active primary."""

    summary = f"{context}: {type(cleanup_error).__name__}"
    if str(cleanup_error):
        summary += f": {cleanup_error}"
    primary_error.add_note(summary)
    for note in getattr(cleanup_error, "__notes__", ()):
        primary_error.add_note(f"{context} detail: {note}")


def _run_cleanup_steps(
    steps: Iterable[tuple[str, Callable[[], None]]],
    *,
    primary_error: BaseException | None = None,
) -> None:
    """Run every cleanup while preserving an active operation primary."""

    failures: list[tuple[str, BaseException]] = []
    for context, cleanup in steps:
        try:
            cleanup()
        except BaseException as cleanup_error:
            failures.append((context, cleanup_error))
    if not failures:
        return
    if primary_error is not None:
        for context, failure in failures:
            _attach_cleanup_failure(
                primary_error,
                failure,
                context=context,
            )
        return
    primary_index = next(
        (
            index
            for index, (_, error) in enumerate(failures)
            if not isinstance(error, Exception)
        ),
        0,
    )
    primary_error = failures[primary_index][1]
    for index, (context, failure) in enumerate(failures):
        if index == primary_index:
            continue
        _attach_cleanup_failure(
            primary_error,
            failure,
            context=context,
        )
    raise primary_error


class JointRiggerPostCommitCleanupError(JointRiggerFacadeError):
    """Publication committed successfully, but temporary-state cleanup failed."""

    committed: bool = True

    def __init__(
        self,
        committed_result: JointRiggerResultV1,
        cleanup_error: Exception,
    ) -> None:
        self.committed_result = committed_result
        self.result = committed_result
        self.cleanup_error = cleanup_error
        super().__init__(
            "Joint Rigger artifact publication committed successfully "
            f"(committed=True), but post-commit cleanup failed: {cleanup_error}"
        )


@dataclass(frozen=True)
class _LocalInputSnapshot:
    """Pre-probe identity and dependency paths for one local request artifact."""

    label: str
    artifact: ArtifactIdentityV1
    path: Path
    dependency_paths: tuple[Path, ...]
    actual_dependency_bundle_sha256: str | None


@dataclass(frozen=True)
class _ParsedModelReport:
    """One bounded report payload and the contract model parsed from it."""

    model: Any
    payload: bytes


@dataclass(frozen=True)
class _ValidatedReports:
    """The exact backend report bytes accepted for one publication."""

    result: _ParsedModelReport
    diagnostics: _ParsedModelReport


@dataclass
class _PrivateDirectoryOwner:
    """One unpredictable temporary tree bound through retained descriptors."""

    path: Path
    entry_name: str
    parent_descriptor: int
    parent_identity: tuple[int, int]
    source_descriptor: int
    source_identity: tuple[int, int]

    def cleanup(self, *, primary_error: BaseException | None = None) -> None:
        parent_descriptor = self.parent_descriptor
        source_descriptor = self.source_descriptor
        self.parent_descriptor = -1
        self.source_descriptor = -1

        def remove_owned_tree() -> None:
            parent_metadata = os.fstat(parent_descriptor)
            if (
                not stat.S_ISDIR(parent_metadata.st_mode)
                or (parent_metadata.st_dev, parent_metadata.st_ino)
                != self.parent_identity
            ):
                raise RuntimeError("private temporary parent descriptor changed inode")
            source_metadata = os.fstat(source_descriptor)
            if (
                not stat.S_ISDIR(source_metadata.st_mode)
                or (source_metadata.st_dev, source_metadata.st_ino)
                != self.source_identity
            ):
                raise RuntimeError(
                    "private temporary directory descriptor changed inode"
                )
            _remove_descriptor_entry(
                parent_descriptor,
                self.entry_name,
                expected_identity=self.source_identity,
                source_descriptor=source_descriptor,
                label=f"private temporary directory {self.path}",
            )

        _run_cleanup_steps(
            (
                ("Private temporary directory cleanup failed", remove_owned_tree),
                (
                    "Private temporary directory descriptor cleanup failed",
                    partial(os.close, source_descriptor),
                ),
                (
                    "Private temporary parent descriptor cleanup failed",
                    partial(os.close, parent_descriptor),
                ),
            ),
            primary_error=primary_error,
        )


@dataclass
class _SealedReportSnapshot:
    """One facade-private report inode and its fd-relative cleanup state."""

    path: Path
    entry_name: str
    parent_descriptor: int
    parent_identity: tuple[int, int]
    source_descriptor: int
    source_identity: tuple[int, int]
    source_sha256: str

    def cleanup(self) -> None:
        parent_descriptor = self.parent_descriptor
        source_descriptor = self.source_descriptor
        self.parent_descriptor = -1
        self.source_descriptor = -1
        errors = _cleanup_private_snapshot_resources(
            entry_name=self.entry_name,
            parent_descriptor=parent_descriptor,
            parent_identity=self.parent_identity,
            source_descriptor=source_descriptor,
            source_identity=self.source_identity,
        )
        if errors:
            raise JointRiggerArtifactError(
                f"Could not clean private report snapshot {self.path}: "
                + "; ".join(errors)
            )


@dataclass(frozen=True)
class _SealedReportSnapshots:
    """Facade-private report inodes that a backend was never given."""

    result: _SealedReportSnapshot
    diagnostics: _SealedReportSnapshot

    def cleanup(self) -> None:
        _run_cleanup_steps(
            (
                ("Private result snapshot cleanup also failed", self.result.cleanup),
                (
                    "Private diagnostics snapshot cleanup also failed",
                    self.diagnostics.cleanup,
                ),
            )
        )


@dataclass
class _SealedGeneratedRoot:
    """The exact validated generated-root inode retained through promotion."""

    path: Path
    source_descriptor: int
    source_identity: tuple[int, int]
    source_sha256: str
    source_mode: int

    def cleanup(self) -> None:
        descriptor = self.source_descriptor
        self.source_descriptor = -1
        if descriptor < 0:
            return
        try:
            os.close(descriptor)
        except OSError as exc:
            raise JointRiggerArtifactError(
                f"Could not close sealed generated-root descriptor for {self.path}: "
                f"{exc}"
            ) from exc


@dataclass
class _SealedSidecarSnapshot:
    """A facade-owned deep copy of one validated composition sidecar."""

    path: Path
    entry_name: str
    parent_descriptor: int
    parent_identity: tuple[int, int]
    source_descriptor: int
    source_identity: tuple[int, int]
    source_tree_sha256: str
    source_mode: int
    dependency_bundle_sha256: str

    def cleanup(self) -> None:
        parent_descriptor = self.parent_descriptor
        source_descriptor = self.source_descriptor
        self.parent_descriptor = -1
        self.source_descriptor = -1
        errors = _cleanup_private_snapshot_resources(
            entry_name=self.entry_name,
            parent_descriptor=parent_descriptor,
            parent_identity=self.parent_identity,
            source_descriptor=source_descriptor,
            source_identity=self.source_identity,
        )
        if errors:
            raise JointRiggerArtifactError(
                f"Could not clean private sidecar snapshot {self.path}: "
                + "; ".join(errors)
            )


@dataclass
class _SealedDependencySnapshot:
    """One no-sidecar dependency inode retained through the commit point."""

    path: Path
    source_descriptor: int
    source_identity: tuple[int, int]
    source_state: tuple[int, int, int, int, int, int, int]
    source_sha256: str

    def cleanup(self) -> None:
        descriptor = self.source_descriptor
        self.source_descriptor = -1
        if descriptor < 0:
            return
        try:
            os.close(descriptor)
        except OSError as exc:
            raise JointRiggerArtifactError(
                f"Could not close sealed dependency descriptor for {self.path}: {exc}"
            ) from exc


@dataclass
class _SealedGeneratedArtifacts:
    """Generated root and optional sidecar sealed after identity validation."""

    root: _SealedGeneratedRoot
    sidecar: _SealedSidecarSnapshot | None
    dependencies: tuple[_SealedDependencySnapshot, ...] = ()
    dependency_records: tuple[_CapturedDependencyIdentityRecord, ...] = ()
    package_identity: ArtifactIdentityV1 | None = None

    def cleanup(self) -> None:
        steps: list[tuple[str, Callable[[], None]]] = []
        if self.sidecar is not None:
            steps.append(
                ("Private sidecar snapshot cleanup also failed", self.sidecar.cleanup)
            )
        steps.extend(
            (
                f"Sealed dependency {dependency.path} cleanup also failed",
                dependency.cleanup,
            )
            for dependency in self.dependencies
        )
        steps.append(("Sealed generated-root cleanup also failed", self.root.cleanup))
        _run_cleanup_steps(steps)


@runtime_checkable
class JointRiggerBackend(Protocol):
    """Concrete authorer supplied by a consuming app or integration package.

    Backends write only to the physical staging paths they receive, while USD
    references and reported locations must derive from the immutable
    ``publication_*`` metadata on those targets. ``probe`` must perform
    dependency, version, and API-shape checks without writing artifacts.
    Missing and incompatible implementations should raise the typed
    exceptions above. A backend whose ``author`` method repeats every probe
    check in the same protected resource lifecycle may declare the exact class
    marker ``author_runs_probe_checks = True``. The facade then validates both
    methods but avoids running a redundant standalone probe immediately before
    ``author``.
    """

    def probe(self, request: JointRiggerInputV1) -> None:
        """Fail before authoring when the backend cannot consume ``request``."""
        ...  # pragma: no cover - Protocol declaration only

    def author(
        self,
        request: JointRiggerInputV1,
        artifact_targets: JointRiggerArtifactTargets,
    ) -> JointRiggerResultV1:
        """Write a complete staged artifact set and return its typed result."""
        ...  # pragma: no cover - Protocol declaration only


def author_joint_rig(
    request: JointRiggerInputV1,
    backend: JointRiggerBackend,
    artifact_targets: JointRiggerArtifactTargets,
) -> JointRiggerResultV1:
    """Run one backend and publish its complete artifact set atomically.

    Final targets are validated before backend probing, then left untouched
    while the backend writes same-filesystem staging targets. Only a
    complete, validated staged bundle enters the rollback-capable replacement
    transaction. This facade publishes only ``status=succeeded`` results; all
    other statuses are rejected without replacing an existing complete bundle.
    """
    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerInputV1,
    )

    if not isinstance(request, JointRiggerInputV1):
        raise TypeError("request must be a JointRiggerInputV1")
    return author_joint_rig_from_factory(
        lambda: (request, backend),
        artifact_targets,
    )


def author_joint_rig_from_factory(
    request_backend_factory: Callable[
        [],
        tuple[JointRiggerInputV1, JointRiggerBackend],
    ],
    artifact_targets: JointRiggerArtifactTargets,
) -> JointRiggerResultV1:
    """Capture final targets, then build the request/backend and publish once.

    Request factories are for wrappers whose request construction reads local
    source artifacts. The one authoritative staging reservation is created
    first, so those reads cannot move an input inode onto an absent final target
    and have a later transaction legitimize it.
    """
    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerInputV1,
    )

    if not callable(request_backend_factory):
        raise TypeError("request_backend_factory must be callable")
    if not isinstance(artifact_targets, JointRiggerArtifactTargets):
        raise TypeError("artifact_targets must be JointRiggerArtifactTargets")

    # Validate only caller-controlled target shape first. Local input reads are
    # deliberately deferred until every final target has been descriptor-bound;
    # otherwise an input inode moved onto an initially absent target during
    # preflight could be legitimized as the target's initial state.
    validate_artifact_targets(artifact_targets)
    # Reserve staging and pin every final target before invoking backend code.
    # A probe is non-writing by contract, but it still runs outside our control;
    # target state captured after it returned could legitimize a read inode that
    # the probe moved onto a publication path and replaced byte-for-byte.
    staged = create_staged_artifact_targets(artifact_targets)
    try:
        produced = request_backend_factory()
        if not isinstance(produced, tuple) or len(produced) != 2:
            raise TypeError(
                "request_backend_factory must return (JointRiggerInputV1, backend)"
            )
        request, backend = produced
        if not isinstance(request, JointRiggerInputV1):
            raise TypeError(
                "request_backend_factory must return a JointRiggerInputV1 first"
            )
    except BaseException as factory_error:
        try:
            staged.cleanup()
        except BaseException as cleanup_error:
            _attach_cleanup_failure(
                factory_error,
                cleanup_error,
                context="Deferred Joint Rigger request cleanup also failed",
            )
        raise
    return _author_joint_rig_with_staged_targets(
        request,
        backend,
        artifact_targets,
        staged,
    )


def _author_joint_rig_with_staged_targets(
    request: JointRiggerInputV1,
    backend: JointRiggerBackend,
    artifact_targets: JointRiggerArtifactTargets,
    staged: StagedJointRiggerArtifacts,
) -> JointRiggerResultV1:
    """Execute one request against an already captured final-target state."""
    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerContractError,
        JointRiggerResultV1,
    )

    sealed_reports: _SealedReportSnapshots | None = None
    sealed_generated: _SealedGeneratedArtifacts | None = None
    primary_error: BaseException | None = None
    committed_result: JointRiggerResultV1 | None = None
    try:
        input_snapshots, read_paths = _preflight_request_inputs(request)
        validate_artifact_targets(artifact_targets, read_paths=read_paths)
        _probe_backend(backend, request)
        try:
            author = backend.author
            result = author(request, staged.staged_targets)
        except (JointRiggerFacadeError, JointRiggerContractError):
            raise
        except ImportError as exc:
            raise JointRiggerBackendUnavailableError(
                f"{_backend_label(backend)} dependency failed during authoring: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        except (AttributeError, TypeError) as exc:
            raise JointRiggerBackendIncompatibleError(
                f"{_backend_label(backend)} API failed during authoring: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        except Exception as exc:
            raise JointRiggerArtifactError(
                f"{_backend_label(backend)} failed during authoring: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(result, JointRiggerResultV1):
            raise JointRiggerBackendIncompatibleError(
                f"{_backend_label(backend)} returned {type(result).__name__}; "
                "expected JointRiggerResultV1"
            )
        _validate_result_identity(request, result, staged)
        _validate_diagnostic_decisions(request, result.diagnostics)
        _verify_request_inputs_unchanged(input_snapshots)

        validated_reports = _validate_reports(result, staged.staged_targets)
        sealed_reports = _seal_validated_reports(
            validated_reports,
            staged.staged_targets,
        )
        sealed_generated = _seal_generated_artifacts(
            result,
            staged.staged_targets,
        )
        sealed_generated.dependencies = _validate_sealed_generated_composition(
            result,
            staged,
            sealed_generated,
            capture_dependencies=True,
        )
        try:
            promotion = staged_promotion_artifacts(staged)
            promotion = _substitute_sealed_report_snapshots(
                promotion,
                staged_targets=staged.staged_targets,
                sealed_reports=sealed_reports,
            )
            promotion = _substitute_sealed_generated_artifacts(
                promotion,
                staged_targets=staged.staged_targets,
                sealed_generated=sealed_generated,
            )
            persisted_result = _revalidate_sealed_reports(
                validated_reports,
                sealed_reports,
            )
            _revalidate_sealed_generated_artifacts(sealed_generated)
            _validate_sealed_generated_composition(
                result,
                staged,
                sealed_generated,
                capture_dependencies=False,
            )

            def validate_precommit_state() -> None:
                _verify_request_inputs_unchanged(input_snapshots)
                _revalidate_sealed_reports(validated_reports, sealed_reports)
                _revalidate_sealed_generated_artifacts(sealed_generated)
                _validate_sealed_generated_composition(
                    result,
                    staged,
                    sealed_generated,
                    capture_dependencies=False,
                )

            promote_staged_artifacts(
                promotion,
                precommit_validator=validate_precommit_state,
            )
            committed_result = persisted_result
        except CommittedArtifactPublicationCleanupError as exc:
            raise JointRiggerPostCommitCleanupError(
                persisted_result,
                exc,
            ) from exc
        except JointRiggerArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise JointRiggerArtifactError(
                f"Could not publish the Joint Rigger artifact set: {exc}"
            ) from exc
        return persisted_result
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            _cleanup_authoring_state(sealed_reports, sealed_generated, staged)
        except BaseException as cleanup_error:
            if primary_error is None:
                if committed_result is not None and isinstance(
                    cleanup_error, Exception
                ):
                    raise JointRiggerPostCommitCleanupError(
                        committed_result,
                        cleanup_error,
                    ) from cleanup_error
                raise
            _attach_cleanup_failure(
                primary_error,
                cleanup_error,
                context=(
                    "Joint Rigger cleanup also failed without replacing the "
                    "primary error"
                ),
            )


def _probe_backend(backend: JointRiggerBackend, request: JointRiggerInputV1) -> None:
    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerInputV2,
    )

    probe = getattr(backend, "probe", None)
    if not callable(probe):
        raise JointRiggerBackendIncompatibleError(
            f"{_backend_label(backend)} does not expose probe(...)"
        )
    if not callable(getattr(backend, "author", None)):
        raise JointRiggerBackendIncompatibleError(
            f"{_backend_label(backend)} does not expose author(...)"
        )
    if isinstance(request, JointRiggerInputV2) and (
        getattr(type(backend), "supports_joint_rigger_input_v2", False) is not True
    ):
        raise JointRiggerBackendIncompatibleError(
            f"{_backend_label(backend)} does not support JointRiggerInputV2"
        )
    if _request_requires_aggregate_authoring(request) and (
        getattr(type(backend), "supports_aggregate_rigid_links", False) is not True
    ):
        raise JointRiggerBackendIncompatibleError(
            f"{_backend_label(backend)} does not support V2 aggregate rigid-link "
            "authoring"
        )
    if vars(type(backend)).get("author_runs_probe_checks") is True:
        return
    try:
        probe(request)
    except (JointRiggerBackendUnavailableError, JointRiggerBackendIncompatibleError):
        raise
    except (ImportError, OSError) as exc:
        raise JointRiggerBackendUnavailableError(
            f"{_backend_label(backend)} dependency probe failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    except (AttributeError, TypeError) as exc:
        raise JointRiggerBackendIncompatibleError(
            f"{_backend_label(backend)} API probe failed: {type(exc).__name__}: {exc}"
        ) from exc


def _request_requires_aggregate_authoring(request: JointRiggerInputV1) -> bool:
    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerInputV2,
    )

    return isinstance(request, JointRiggerInputV2) and any(
        link.body_authoring == "aggregate" for link in request.rigid_links
    )


def _validate_reports(
    result: JointRiggerResultV1,
    staged_targets: JointRiggerArtifactTargets,
) -> _ValidatedReports:
    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerDiagnosticsV1,
        JointRiggerResultV1,
    )

    persisted_result = _load_model_report(
        staged_targets.result_path,
        JointRiggerResultV1,
        "result",
    )
    persisted_diagnostics = _load_model_report(
        staged_targets.diagnostics_path,
        JointRiggerDiagnosticsV1,
        "diagnostics",
    )
    if _canonical_model_payload(persisted_result.model) != _canonical_model_payload(
        result
    ):
        raise JointRiggerArtifactError(
            "Persisted Joint Rigger result does not match the returned result"
        )

    returned_diagnostics = getattr(result, "diagnostics", None)
    if returned_diagnostics is None:
        raise JointRiggerBackendIncompatibleError(
            "JointRiggerResultV1 does not expose diagnostics"
        )
    if _canonical_model_payload(
        persisted_diagnostics.model
    ) != _canonical_model_payload(returned_diagnostics):
        raise JointRiggerArtifactError(
            "Persisted Joint Rigger diagnostics do not match the returned result"
        )
    return _ValidatedReports(
        result=persisted_result,
        diagnostics=persisted_diagnostics,
    )


def _seal_validated_reports(
    reports: _ValidatedReports,
    staged_targets: JointRiggerArtifactTargets,
) -> _SealedReportSnapshots:
    """Detach accepted report bytes from every backend-owned inode."""

    diagnostics = _create_private_report_snapshot(
        staged_targets.diagnostics_path,
        reports.diagnostics.payload,
        label="diagnostics",
    )
    try:
        result = _create_private_report_snapshot(
            staged_targets.result_path,
            reports.result.payload,
            label="result",
        )
    except BaseException as error:
        try:
            diagnostics.cleanup()
        except BaseException as cleanup_error:
            _attach_cleanup_failure(
                error,
                cleanup_error,
                context=(
                    "Joint Rigger snapshot cleanup also failed without replacing "
                    "the primary error"
                ),
            )
        raise
    return _SealedReportSnapshots(
        result=result,
        diagnostics=diagnostics,
    )


def _create_private_report_snapshot(
    path: Path,
    payload: bytes,
    *,
    label: str,
) -> _SealedReportSnapshot:
    """Write accepted bytes to a fresh fd-bound entry unknown to the backend."""

    if len(payload) > _MAX_REPORT_BYTES:  # pragma: no cover - loader invariant
        raise JointRiggerArtifactError(
            f"Validated {label} report exceeds the {_MAX_REPORT_BYTES}-byte limit"
        )
    parent_descriptor = -1
    source_descriptor = -1
    writer_descriptor = -1
    entry_name: str | None = None
    parent_identity: tuple[int, int] | None = None
    source_identity: tuple[int, int] | None = None
    parent_path = path.parent.expanduser()
    try:
        parent_path = parent_path.resolve(strict=True)
        parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent_flags |= getattr(os, "O_CLOEXEC", 0)
        parent_descriptor = os.open(parent_path, parent_flags)
        parent_metadata = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(parent_metadata.st_mode):  # pragma: no cover - OS guard
            raise NotADirectoryError(parent_path)
        parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
        observed_parent = os.stat(parent_path, follow_symlinks=False)
        if (observed_parent.st_dev, observed_parent.st_ino) != parent_identity:
            raise RuntimeError(
                f"Report snapshot parent changed while it was opened: {parent_path}"
            )

        writer_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        writer_flags |= getattr(os, "O_CLOEXEC", 0)
        for _ in range(128):
            candidate_name = f".{path.name}.sealed-{secrets.token_hex(12)}"
            entry_name = candidate_name
            try:
                writer_descriptor = os.open(
                    entry_name,
                    writer_flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:  # pragma: no cover - cryptographic collision
                entry_name = None
                continue
            writer_metadata = os.fstat(writer_descriptor)
            source_identity = (writer_metadata.st_dev, writer_metadata.st_ino)
            break
        else:  # pragma: no cover - cryptographic collision exhaustion
            raise JointRiggerArtifactError(
                f"Could not allocate a private {label} report snapshot"
            )

        remaining = memoryview(payload)
        while remaining:
            written = os.write(writer_descriptor, remaining)
            if written <= 0:  # pragma: no cover - regular-file OS invariant
                raise OSError("short write while sealing report")
            remaining = remaining[written:]
        os.fchmod(writer_descriptor, 0o444)
        os.fsync(writer_descriptor)

        source_flags = os.O_RDONLY | os.O_NOFOLLOW
        for optional_flag in ("O_CLOEXEC", "O_NONBLOCK"):
            source_flags |= getattr(os, optional_flag, 0)
        source_descriptor = os.open(
            entry_name,
            source_flags,
            dir_fd=parent_descriptor,
        )
        writer_metadata = os.fstat(writer_descriptor)
        source_metadata = os.fstat(source_descriptor)
        entry_metadata = os.stat(
            entry_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        opened_source_identity = (source_metadata.st_dev, source_metadata.st_ino)
        if (
            not stat.S_ISREG(source_metadata.st_mode)
            or source_metadata.st_nlink != 1
            or source_metadata.st_size != len(payload)
            or source_metadata.st_mode & 0o222
            or (writer_metadata.st_dev, writer_metadata.st_ino)
            != opened_source_identity
            or (entry_metadata.st_dev, entry_metadata.st_ino) != opened_source_identity
            or source_identity != opened_source_identity
        ):
            raise JointRiggerArtifactError(
                f"Could not create a private regular {label} report snapshot"
            )
        owned_writer_descriptor = writer_descriptor
        writer_descriptor = -1
        os.close(owned_writer_descriptor)
        parent_identity = _require_present_invariant(
            parent_identity,
            label=f"private {label} report snapshot parent identity",
        )
        snapshot = _SealedReportSnapshot(
            path=parent_path / entry_name,
            entry_name=entry_name,
            parent_descriptor=parent_descriptor,
            parent_identity=parent_identity,
            source_descriptor=source_descriptor,
            source_identity=opened_source_identity,
            source_sha256=hashlib.sha256(payload).hexdigest(),
        )
        parent_descriptor = -1
        source_descriptor = -1
        entry_name = None
        return snapshot
    except JointRiggerArtifactError as artifact_error:
        cleanup_errors = _cleanup_private_snapshot_resources(
            entry_name=entry_name,
            parent_descriptor=parent_descriptor,
            parent_identity=parent_identity,
            source_descriptor=source_descriptor,
            source_identity=source_identity,
            writer_descriptor=writer_descriptor,
            defer_fatal_error=True,
        )
        if cleanup_errors:
            artifact_error.add_note(
                "Snapshot cleanup also failed: " + "; ".join(cleanup_errors)
            )
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        wrapped_error = JointRiggerArtifactError(
            f"Could not seal the validated {label} report at {path}: {exc}"
        )
        cleanup_errors = _cleanup_private_snapshot_resources(
            entry_name=entry_name,
            parent_descriptor=parent_descriptor,
            parent_identity=parent_identity,
            source_descriptor=source_descriptor,
            source_identity=source_identity,
            writer_descriptor=writer_descriptor,
            defer_fatal_error=True,
        )
        if cleanup_errors:
            wrapped_error.add_note(
                "Snapshot cleanup also failed: " + "; ".join(cleanup_errors)
            )
        raise wrapped_error from exc
    except BaseException as fatal_error:
        cleanup_errors = _cleanup_private_snapshot_resources(
            entry_name=entry_name,
            parent_descriptor=parent_descriptor,
            parent_identity=parent_identity,
            source_descriptor=source_descriptor,
            source_identity=source_identity,
            writer_descriptor=writer_descriptor,
            defer_fatal_error=True,
        )
        if cleanup_errors:
            fatal_error.add_note(
                "Snapshot cleanup also failed: " + "; ".join(cleanup_errors)
            )
        raise


def _cleanup_private_snapshot_resources(
    *,
    entry_name: str | None,
    parent_descriptor: int,
    parent_identity: tuple[int, int] | None,
    source_descriptor: int,
    source_identity: tuple[int, int] | None,
    writer_descriptor: int = -1,
    defer_fatal_error: bool = False,
) -> list[str]:
    """Remove one reserved entry and close every fd before propagating failure."""

    errors: list[str] = []
    fatal_error: BaseException | None = None
    fatal_error_context = ""
    secondary_fatal_errors: list[str] = []

    def capture_fatal_error(context: str, error: BaseException) -> None:
        nonlocal fatal_error, fatal_error_context
        if fatal_error is None:
            fatal_error = error
            fatal_error_context = context
            return
        secondary_fatal_errors.append(f"{context}: {type(error).__name__}: {error}")

    try:
        if parent_descriptor >= 0:
            try:
                parent_metadata = os.fstat(parent_descriptor)
                observed_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
                if (
                    not stat.S_ISDIR(parent_metadata.st_mode)
                    or parent_identity is None
                    or observed_identity != parent_identity
                ):
                    raise RuntimeError(
                        "private snapshot parent descriptor changed inode"
                    )
                if entry_name is not None:
                    descriptor = (
                        source_descriptor
                        if source_descriptor >= 0
                        else writer_descriptor
                    )
                    if source_identity is None:
                        if descriptor < 0:
                            raise RuntimeError(
                                "private snapshot source identity could not be "
                                "bound through an owned descriptor"
                            )
                        created_metadata = os.fstat(descriptor)
                        source_identity = (
                            created_metadata.st_dev,
                            created_metadata.st_ino,
                        )
                    if descriptor >= 0:
                        descriptor_metadata = os.fstat(descriptor)
                        if (
                            descriptor_metadata.st_dev,
                            descriptor_metadata.st_ino,
                        ) != source_identity:
                            raise RuntimeError(
                                "private snapshot source descriptor changed inode"
                            )
                    _remove_private_snapshot_entry(
                        parent_descriptor,
                        entry_name,
                        expected_identity=source_identity,
                        source_descriptor=descriptor,
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"remove reserved entry: {exc}")
            except BaseException as exc:
                capture_fatal_error("remove reserved entry", exc)
    finally:
        closed_descriptors: set[int] = set()
        for descriptor, descriptor_label in (
            (writer_descriptor, "writer"),
            (source_descriptor, "source"),
            (parent_descriptor, "parent"),
        ):
            if descriptor < 0 or descriptor in closed_descriptors:
                continue
            closed_descriptors.add(descriptor)
            try:
                os.close(descriptor)
            except OSError as exc:
                errors.append(f"close {descriptor_label} descriptor: {exc}")
            except BaseException as exc:
                capture_fatal_error(f"close {descriptor_label} descriptor", exc)

    if fatal_error is not None:
        additional_errors = [*errors, *secondary_fatal_errors]
        if additional_errors:
            fatal_error.add_note(
                "Private snapshot resource cleanup also failed: "
                + "; ".join(additional_errors)
            )
        if not defer_fatal_error:
            raise fatal_error
        fatal_summary = f"{fatal_error_context}: {type(fatal_error).__name__}"
        if str(fatal_error):
            fatal_summary += f": {fatal_error}"
        return [fatal_summary, *additional_errors]
    return errors


def _create_private_directory_owner(
    parent_path: Path,
    *,
    prefix: str,
) -> _PrivateDirectoryOwner:
    """Create and immediately bind one private child directory."""

    parent_descriptor = -1
    source_descriptor = -1
    parent_identity: tuple[int, int] | None = None
    source_identity: tuple[int, int] | None = None
    entry_name: str | None = None
    resolved_parent = parent_path.expanduser()
    try:
        resolved_parent = resolved_parent.resolve(strict=True)
        parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent_flags |= getattr(os, "O_CLOEXEC", 0)
        parent_descriptor = os.open(resolved_parent, parent_flags)
        parent_metadata = os.fstat(parent_descriptor)
        parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
        observed_parent = os.stat(resolved_parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or (observed_parent.st_dev, observed_parent.st_ino) != parent_identity
        ):
            raise RuntimeError(
                f"Private temporary parent changed while opened: {resolved_parent}"
            )

        source_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        source_flags |= getattr(os, "O_CLOEXEC", 0)
        for _ in range(128):
            candidate_name = f"{prefix}{secrets.token_hex(12)}"
            entry_name = candidate_name
            try:
                os.mkdir(candidate_name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:  # pragma: no cover - cryptographic collision
                entry_name = None
                continue
            source_descriptor = os.open(
                entry_name,
                source_flags,
                dir_fd=parent_descriptor,
            )
            source_metadata = os.fstat(source_descriptor)
            source_identity = (source_metadata.st_dev, source_metadata.st_ino)
            observed_source = os.stat(
                entry_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(source_metadata.st_mode)
                or (observed_source.st_dev, observed_source.st_ino) != source_identity
            ):
                raise RuntimeError(
                    "Private temporary directory changed while it was bound"
                )
            parent_identity = _require_present_invariant(
                parent_identity,
                label="private temporary directory parent identity",
            )
            owner = _PrivateDirectoryOwner(
                path=resolved_parent / entry_name,
                entry_name=entry_name,
                parent_descriptor=parent_descriptor,
                parent_identity=parent_identity,
                source_descriptor=source_descriptor,
                source_identity=source_identity,
            )
            parent_descriptor = -1
            source_descriptor = -1
            entry_name = None
            return owner
        raise RuntimeError("Could not allocate a private temporary directory")
    except BaseException as creation_error:
        cleanup_steps: list[tuple[str, Callable[[], None]]] = []
        cleanup_parent_descriptor = parent_descriptor
        cleanup_source_descriptor = source_descriptor
        if (
            entry_name is not None
            and cleanup_parent_descriptor >= 0
            and cleanup_source_descriptor >= 0
        ):

            def remove_created_directory() -> None:
                nonlocal source_identity
                if source_identity is None:
                    source_metadata = os.fstat(cleanup_source_descriptor)
                    source_identity = (
                        source_metadata.st_dev,
                        source_metadata.st_ino,
                    )
                _remove_descriptor_entry(
                    cleanup_parent_descriptor,
                    entry_name,
                    expected_identity=source_identity,
                    source_descriptor=cleanup_source_descriptor,
                    label=(
                        "uncommitted private temporary directory "
                        f"{resolved_parent / entry_name}"
                    ),
                )

            cleanup_steps.append(
                (
                    "Private temporary directory creation cleanup failed",
                    remove_created_directory,
                )
            )
        elif entry_name is not None:

            def report_preserved_directory() -> None:
                raise RuntimeError(
                    "Private temporary directory could not be bound through an "
                    "owned descriptor; its unpredictable name was preserved at "
                    f"{resolved_parent / entry_name}"
                )

            cleanup_steps.append(
                (
                    "Private temporary directory creation cleanup failed",
                    report_preserved_directory,
                )
            )
        if cleanup_source_descriptor >= 0:
            source_descriptor = -1
            cleanup_steps.append(
                (
                    "Private temporary directory descriptor cleanup failed",
                    partial(os.close, cleanup_source_descriptor),
                )
            )
        if cleanup_parent_descriptor >= 0:
            parent_descriptor = -1
            cleanup_steps.append(
                (
                    "Private temporary parent descriptor cleanup failed",
                    partial(os.close, cleanup_parent_descriptor),
                )
            )
        _run_cleanup_steps(cleanup_steps, primary_error=creation_error)
        raise


@contextmanager
def _private_directory_owner(parent_path: Path, *, prefix: str) -> Iterator[Path]:
    """Yield one fd-bound private tree and clean only its exact inode."""

    owner = _create_private_directory_owner(parent_path, prefix=prefix)
    primary_error: BaseException | None = None
    try:
        yield owner.path
    except BaseException as error:
        primary_error = error
        raise
    finally:
        owner.cleanup(primary_error=primary_error)


def _seal_generated_artifacts(
    result: JointRiggerResultV1,
    staged_targets: JointRiggerArtifactTargets,
) -> _SealedGeneratedArtifacts:
    """Bind the validated root and deep-copy its optional sidecar."""

    output_artifact = result.output_artifact
    if output_artifact is None:  # pragma: no cover - validated result invariant
        raise JointRiggerArtifactError(
            "A succeeded Joint Rigger result must identify its generated root"
        )
    expected_bundle_sha256 = output_artifact.dependency_bundle_sha256
    if expected_bundle_sha256 is None:  # pragma: no cover - validated invariant
        raise JointRiggerArtifactError(
            "A succeeded Joint Rigger result must identify its dependency bundle"
        )
    root = _seal_generated_root(
        staged_targets.output_path,
        expected_sha256=output_artifact.root_sha256,
    )
    sidecar: _SealedSidecarSnapshot | None = None
    try:
        if staged_targets.sidecar_path is not None:
            sidecar = _create_private_sidecar_snapshot(
                staged_targets.sidecar_path,
                private_parent=staged_targets.output_path.parent,
                expected_sha256=expected_bundle_sha256,
            )
    except BaseException as error:
        try:
            root.cleanup()
        except BaseException as cleanup_error:
            _attach_cleanup_failure(
                error,
                cleanup_error,
                context=(
                    "Generated-root cleanup also failed without replacing the "
                    "primary error"
                ),
            )
        raise
    return _SealedGeneratedArtifacts(root=root, sidecar=sidecar)


def _seal_generated_root(
    path: Path,
    *,
    expected_sha256: str,
) -> _SealedGeneratedRoot:
    """Retain a read-only descriptor for the exact validated root inode."""

    descriptor = -1
    operation_error: BaseException | None = None
    flags = os.O_RDONLY | os.O_NOFOLLOW
    for optional_flag in ("O_CLOEXEC", "O_NONBLOCK"):
        flags |= getattr(os, optional_flag, 0)
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        source_identity = (metadata.st_dev, metadata.st_ino)
        path_metadata = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (path_metadata.st_dev, path_metadata.st_ino) != source_identity
        ):
            raise JointRiggerArtifactError(
                "The generated root must be a regular file with exactly one hard "
                f"link: {path}"
            )
        source_mode = stat.S_IMODE(metadata.st_mode) & ~0o222
        os.fchmod(descriptor, source_mode)
        observed_sha256 = _stable_descriptor_sha256(
            descriptor,
            label="generated root",
        )
        sealed_metadata = os.fstat(descriptor)
        if (
            (sealed_metadata.st_dev, sealed_metadata.st_ino) != source_identity
            or sealed_metadata.st_nlink != 1
            or stat.S_IMODE(sealed_metadata.st_mode) != source_mode
            or sealed_metadata.st_mode & 0o222
        ):
            raise JointRiggerArtifactError(
                f"Generated root changed while it was sealed: {path}"
            )
        if observed_sha256 != expected_sha256:
            raise JointRiggerArtifactError(
                "Joint Rigger output_artifact root_sha256 changed while the "
                "generated root was sealed"
            )
        snapshot = _SealedGeneratedRoot(
            path=path,
            source_descriptor=descriptor,
            source_identity=source_identity,
            source_sha256=expected_sha256,
            source_mode=source_mode,
        )
        descriptor = -1
        return snapshot
    except JointRiggerArtifactError as error:
        operation_error = error
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        wrapped_error = JointRiggerArtifactError(
            f"Could not seal the validated generated root at {path}: {exc}"
        )
        operation_error = wrapped_error
        raise wrapped_error from exc
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
                        "Generated-root seal descriptor cleanup failed",
                        partial(os.close, owned_descriptor),
                    )
                ],
                primary_error=operation_error,
            )


def _create_private_sidecar_snapshot(
    source: Path,
    *,
    private_parent: Path,
    expected_sha256: str,
) -> _SealedSidecarSnapshot:
    """Deep-copy one validated sidecar into a facade-owned private directory."""

    parent_descriptor = -1
    source_descriptor = -1
    parent_identity: tuple[int, int] | None = None
    source_identity: tuple[int, int] | None = None
    entry_name: str | None = None
    parent_path = private_parent.expanduser()
    try:
        source_before_sha256 = sidecar_dependency_bundle_sha256(source)
        if source_before_sha256 != expected_sha256:
            raise JointRiggerArtifactError(
                "Composition sidecar changed before its private copy was created"
            )

        parent_path = parent_path.resolve(strict=True)
        parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent_flags |= getattr(os, "O_CLOEXEC", 0)
        parent_descriptor = os.open(parent_path, parent_flags)
        parent_metadata = os.fstat(parent_descriptor)
        parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
        observed_parent = os.stat(parent_path, follow_symlinks=False)
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or (observed_parent.st_dev, observed_parent.st_ino) != parent_identity
        ):
            raise RuntimeError(
                f"Sidecar snapshot parent changed while it was opened: {parent_path}"
            )

        source_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        source_flags |= getattr(os, "O_CLOEXEC", 0)
        for _ in range(128):
            candidate_name = f".{source.name}.sealed-{secrets.token_hex(12)}"
            entry_name = candidate_name
            try:
                os.mkdir(candidate_name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:  # pragma: no cover - cryptographic collision
                entry_name = None
                continue
            source_descriptor = os.open(
                entry_name,
                source_flags,
                dir_fd=parent_descriptor,
            )
            source_metadata = os.fstat(source_descriptor)
            source_identity = (source_metadata.st_dev, source_metadata.st_ino)
            entry_metadata = os.stat(
                entry_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(source_metadata.st_mode)
                or (entry_metadata.st_dev, entry_metadata.st_ino) != source_identity
            ):
                raise JointRiggerArtifactError(
                    "Private composition sidecar changed inode during creation"
                )
            break
        else:  # pragma: no cover - cryptographic collision exhaustion
            raise JointRiggerArtifactError(
                "Could not allocate a private composition sidecar snapshot"
            )

        snapshot_path = parent_path / entry_name
        copy_sidecar_directory(
            source,
            source_descriptor,
            label="composition sidecar snapshot",
        )
        source_after_sha256 = sidecar_dependency_bundle_sha256(source)
        snapshot_sha256 = sidecar_dependency_bundle_sha256(snapshot_path)
        if source_after_sha256 != expected_sha256:
            raise JointRiggerArtifactError(
                "Composition sidecar changed while its private copy was created"
            )
        if snapshot_sha256 != expected_sha256:
            raise JointRiggerArtifactError(
                "Private composition sidecar copy does not match the validated bundle"
            )
        source_metadata = os.fstat(source_descriptor)
        opened_source_identity = (source_metadata.st_dev, source_metadata.st_ino)
        entry_metadata = os.stat(
            entry_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(source_metadata.st_mode)
            or (entry_metadata.st_dev, entry_metadata.st_ino) != opened_source_identity
            or source_identity != opened_source_identity
        ):
            raise JointRiggerArtifactError(
                "Private composition sidecar changed inode before sealing"
            )
        _seal_directory_descriptor_tree(
            source_descriptor,
            expected_mount_id=_descriptor_mount_id(parent_descriptor),
        )
        source_mode = stat.S_IMODE(os.fstat(source_descriptor).st_mode)
        source_tree_sha256 = directory_descriptor_tree_sha256(source_descriptor)
        parent_identity = _require_present_invariant(
            parent_identity,
            label="private composition sidecar parent identity",
        )
        snapshot = _SealedSidecarSnapshot(
            path=snapshot_path,
            entry_name=entry_name,
            parent_descriptor=parent_descriptor,
            parent_identity=parent_identity,
            source_descriptor=source_descriptor,
            source_identity=opened_source_identity,
            source_tree_sha256=source_tree_sha256,
            source_mode=source_mode,
            dependency_bundle_sha256=expected_sha256,
        )
        parent_descriptor = -1
        source_descriptor = -1
        entry_name = None
        return snapshot
    except JointRiggerArtifactError as artifact_error:
        cleanup_errors = _cleanup_private_snapshot_resources(
            entry_name=entry_name,
            parent_descriptor=parent_descriptor,
            parent_identity=parent_identity,
            source_descriptor=source_descriptor,
            source_identity=source_identity,
            defer_fatal_error=True,
        )
        if cleanup_errors:
            artifact_error.add_note(
                "Sidecar snapshot cleanup also failed: " + "; ".join(cleanup_errors)
            )
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        wrapped_error = JointRiggerArtifactError(
            f"Could not seal the validated composition sidecar at {source}: {exc}"
        )
        cleanup_errors = _cleanup_private_snapshot_resources(
            entry_name=entry_name,
            parent_descriptor=parent_descriptor,
            parent_identity=parent_identity,
            source_descriptor=source_descriptor,
            source_identity=source_identity,
            defer_fatal_error=True,
        )
        if cleanup_errors:
            wrapped_error.add_note(
                "Sidecar snapshot cleanup also failed: " + "; ".join(cleanup_errors)
            )
        raise wrapped_error from exc
    except BaseException as fatal_error:
        cleanup_errors = _cleanup_private_snapshot_resources(
            entry_name=entry_name,
            parent_descriptor=parent_descriptor,
            parent_identity=parent_identity,
            source_descriptor=source_descriptor,
            source_identity=source_identity,
            defer_fatal_error=True,
        )
        if cleanup_errors:
            fatal_error.add_note(
                "Sidecar snapshot cleanup also failed: " + "; ".join(cleanup_errors)
            )
        raise


def _seal_directory_descriptor_tree(
    descriptor: int,
    *,
    expected_mount_id: int | None = None,
) -> None:
    """Remove write permissions from every fd-relative sidecar entry."""

    try:
        mount_id = (
            _descriptor_mount_id(descriptor)
            if expected_mount_id is None
            else expected_mount_id
        )
        _require_directory_tree_mount_id(
            descriptor,
            expected_mount_id=mount_id,
            label="Private composition sidecar",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointRiggerArtifactError(
            f"Private composition sidecar mount validation failed: {exc}"
        ) from exc
    root_metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise JointRiggerArtifactError(
            "Private composition sidecar descriptor is not a directory"
        )
    for name in os.listdir(descriptor):
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            child_descriptor = os.open(name, flags, dir_fd=descriptor)
            child_error: BaseException | None = None
            try:
                child_metadata = os.fstat(child_descriptor)
                if (child_metadata.st_dev, child_metadata.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise JointRiggerArtifactError(
                        f"Private composition sidecar changed inode: {name}"
                    )
                if _descriptor_mount_id(child_descriptor) != mount_id:
                    raise JointRiggerArtifactError(
                        f"Private composition sidecar contains a mount point: {name}"
                    )
                _seal_directory_descriptor_tree(
                    child_descriptor,
                    expected_mount_id=mount_id,
                )
            except BaseException as error:
                child_error = error
                raise
            finally:
                owned_child_descriptor = child_descriptor
                child_descriptor = -1
                _run_cleanup_steps(
                    [
                        (
                            f"Private sidecar child descriptor cleanup failed for {name}",
                            partial(os.close, owned_child_descriptor),
                        )
                    ],
                    primary_error=child_error,
                )
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise JointRiggerArtifactError(
                f"Private composition sidecar contains invalid entry: {name}"
            )
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        file_descriptor = os.open(name, flags, dir_fd=descriptor)
        file_error: BaseException | None = None
        try:
            opened_metadata = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(opened_metadata.st_mode)
                or (opened_metadata.st_dev, opened_metadata.st_ino)
                != (metadata.st_dev, metadata.st_ino)
                or opened_metadata.st_nlink != 1
            ):
                raise JointRiggerArtifactError(
                    f"Private composition sidecar changed inode: {name}"
                )
            if _descriptor_mount_id(file_descriptor) != mount_id:
                raise JointRiggerArtifactError(
                    f"Private composition sidecar contains a mount point: {name}"
                )
            os.fchmod(file_descriptor, stat.S_IMODE(metadata.st_mode) & ~0o222)
        except BaseException as error:
            file_error = error
            raise
        finally:
            owned_file_descriptor = file_descriptor
            file_descriptor = -1
            _run_cleanup_steps(
                [
                    (
                        f"Private sidecar file descriptor cleanup failed for {name}",
                        partial(os.close, owned_file_descriptor),
                    )
                ],
                primary_error=file_error,
            )
    os.fchmod(descriptor, stat.S_IMODE(root_metadata.st_mode) & ~0o222)


def _stable_descriptor_sha256(descriptor: int, *, label: str) -> str:
    """Hash one regular-file descriptor with positional stability checks."""

    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"{label} descriptor must identify a regular file")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(
            descriptor,
            min(1024 * 1024, before.st_size - offset),
            offset,
        )
        if not chunk:
            raise RuntimeError(f"{label} changed while it was hashed")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, offset):
        raise RuntimeError(f"{label} grew while it was hashed")
    after = os.fstat(descriptor)
    before_state = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_state = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_state != after_state:
        raise RuntimeError(f"{label} changed while it was hashed")
    return digest.hexdigest()


def _regular_descriptor_state(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    """Return the identity and mutation-sensitive state of one regular inode."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_sealed_dependency_snapshot(path: Path) -> _SealedDependencySnapshot:
    """Bind one no-sidecar dependency path to a stable non-symlink inode."""

    normalized = _normalize_local_path_without_symlinks(
        path,
        symlink_error="Generated dependency path must not contain symlinks",
    )
    expected = os.stat(normalized, follow_symlinks=False)
    if not stat.S_ISREG(expected.st_mode):
        raise JointRiggerArtifactError(
            f"Generated dependency must be a regular file: {normalized}"
        )
    flags = os.O_RDONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    operation_error: BaseException | None = None
    try:
        descriptor = os.open(normalized, flags)
        opened = os.fstat(descriptor)
        expected_state = _regular_descriptor_state(expected)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _regular_descriptor_state(opened) != expected_state
        ):
            raise JointRiggerArtifactError(
                f"Generated dependency changed before it was sealed: {normalized}"
            )
        sha256 = _stable_descriptor_sha256(
            descriptor,
            label=f"generated dependency {normalized}",
        )
        after = os.fstat(descriptor)
        observed_path = os.stat(normalized, follow_symlinks=False)
        if (
            _regular_descriptor_state(after) != expected_state
            or _regular_descriptor_state(observed_path) != expected_state
        ):
            raise JointRiggerArtifactError(
                f"Generated dependency changed while it was sealed: {normalized}"
            )
        snapshot = _SealedDependencySnapshot(
            path=normalized,
            source_descriptor=descriptor,
            source_identity=(opened.st_dev, opened.st_ino),
            source_state=expected_state,
            source_sha256=sha256,
        )
        descriptor = -1
        return snapshot
    except JointRiggerArtifactError as error:
        operation_error = error
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        wrapped_error = JointRiggerArtifactError(
            f"Could not seal generated dependency {normalized}: {exc}"
        )
        operation_error = wrapped_error
        raise wrapped_error from exc
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
                        "Sealed dependency descriptor cleanup failed",
                        partial(os.close, owned_descriptor),
                    )
                ],
                primary_error=operation_error,
            )


def _require_sealed_dependency_snapshot(
    snapshot: _SealedDependencySnapshot,
) -> None:
    """Require one retained dependency fd and locator to remain exact."""

    normalized = _normalize_local_path_without_symlinks(
        snapshot.path,
        symlink_error="Generated dependency path must not contain symlinks",
    )
    descriptor_metadata = os.fstat(snapshot.source_descriptor)
    path_metadata = os.stat(normalized, follow_symlinks=False)
    if (
        normalized != snapshot.path
        or not stat.S_ISREG(descriptor_metadata.st_mode)
        or not stat.S_ISREG(path_metadata.st_mode)
        or (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
        != snapshot.source_identity
        or _regular_descriptor_state(descriptor_metadata) != snapshot.source_state
        or _regular_descriptor_state(path_metadata) != snapshot.source_state
    ):
        raise JointRiggerArtifactError(
            f"Sealed generated dependency changed inode or metadata: {snapshot.path}"
        )
    sha256 = _stable_descriptor_sha256(
        snapshot.source_descriptor,
        label=f"sealed generated dependency {snapshot.path}",
    )
    if sha256 != snapshot.source_sha256:
        raise JointRiggerArtifactError(
            f"Sealed generated dependency changed content: {snapshot.path}"
        )


def _require_stable_copy_parent(
    parent_descriptor: int,
    destination: Path,
) -> tuple[int, int]:
    """Bind a copy parent fd to the current nofollow lexical parent."""

    parent_path = destination.parent.expanduser().absolute()
    descriptor_metadata = os.fstat(parent_descriptor)
    lexical_metadata = os.stat(parent_path, follow_symlinks=False)
    descriptor_identity = (
        descriptor_metadata.st_dev,
        descriptor_metadata.st_ino,
    )
    if (
        not stat.S_ISDIR(descriptor_metadata.st_mode)
        or not stat.S_ISDIR(lexical_metadata.st_mode)
        or (lexical_metadata.st_dev, lexical_metadata.st_ino) != descriptor_identity
    ):
        raise RuntimeError(
            f"Stable-copy parent changed while opened: {destination.parent}"
        )
    return descriptor_identity


def _open_stable_copy_parent(destination: Path) -> int:
    """Open the exact parent used for one fd-relative stable-copy target."""

    parent = destination.parent.expanduser().absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(parent, flags)
    try:
        _require_stable_copy_parent(descriptor, destination)
    except BaseException as error:
        _run_cleanup_steps(
            [
                (
                    f"Stable-copy parent descriptor cleanup failed for {destination}",
                    partial(os.close, descriptor),
                )
            ],
            primary_error=error,
        )
        raise
    return descriptor


def _remove_stable_copy_target_from_rebound_parent(
    destination: Path,
    *,
    expected_parent_identity: tuple[int, int],
    expected_target_identity: tuple[int, int],
) -> None:
    """Remove a copy target after an owned parent close reported failure."""

    parent_descriptor = _open_stable_copy_parent(destination)
    operation_error: BaseException | None = None
    try:
        observed_parent_identity = _require_stable_copy_parent(
            parent_descriptor,
            destination,
        )
        if observed_parent_identity != expected_parent_identity:
            raise RuntimeError(
                f"Stable-copy parent changed inode for {destination}; target preserved"
            )
        _remove_descriptor_entry(
            parent_descriptor,
            destination.name,
            expected_identity=expected_target_identity,
            label=f"stable-copy target {destination}",
        )
    except BaseException as error:
        operation_error = error
        raise
    finally:
        _run_cleanup_steps(
            [
                (
                    f"Rebound stable-copy parent cleanup failed for {destination}",
                    partial(os.close, parent_descriptor),
                )
            ],
            primary_error=operation_error,
        )


def _copy_stable_regular_descriptor(
    source_descriptor: int,
    destination: Path,
    *,
    expected_identity: tuple[int, int],
    expected_sha256: str,
    expected_mode: int,
    expected_nlink: int,
    label: str,
    destination_parent_descriptor: int | None = None,
) -> tuple[int, int]:
    """Copy exact stable descriptor bytes into one new nofollow target."""

    parent_descriptor = (
        -1 if destination_parent_descriptor is None else destination_parent_descriptor
    )
    close_parent_descriptor = destination_parent_descriptor is None
    parent_identity: tuple[int, int] | None = None
    target_descriptor = -1
    target_identity: tuple[int, int] | None = None
    unbound_target_name: str | None = None
    primary_error: BaseException | None = None
    try:
        before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != expected_identity
            or stat.S_IMODE(before.st_mode) != expected_mode
            or before.st_nlink != expected_nlink
        ):
            raise RuntimeError(f"{label} changed before descriptor copy")
        before_state = _regular_descriptor_state(before)
        if parent_descriptor < 0:
            parent_descriptor = _open_stable_copy_parent(destination)
        parent_identity = _require_stable_copy_parent(
            parent_descriptor,
            destination,
        )
        target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        target_flags |= getattr(os, "O_CLOEXEC", 0)
        unbound_target_name = destination.name
        try:
            target_descriptor = os.open(
                unbound_target_name,
                target_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            unbound_target_name = None
            raise
        unbound_target_name = None
        target_metadata = os.fstat(target_descriptor)
        target_identity = (target_metadata.st_dev, target_metadata.st_ino)
        if not stat.S_ISREG(target_metadata.st_mode) or target_metadata.st_nlink != 1:
            raise RuntimeError(f"Detached target for {label} is not a regular file")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                source_descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise RuntimeError(f"{label} changed during descriptor copy")
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(target_descriptor, remaining)
                if written <= 0:  # pragma: no cover - regular-file OS invariant
                    raise OSError(f"Short write while copying {label}")
                remaining = remaining[written:]
            offset += len(chunk)
        if os.pread(source_descriptor, 1, offset):
            raise RuntimeError(f"{label} grew during descriptor copy")
        after = os.fstat(source_descriptor)
        if _regular_descriptor_state(after) != before_state:
            raise RuntimeError(f"{label} changed during descriptor copy")
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"{label} content changed before descriptor copy")
        os.fchmod(target_descriptor, expected_mode)
        os.fsync(target_descriptor)
    except BaseException as exc:
        primary_error = exc
    finally:
        cleanup_steps: list[tuple[str, Callable[[], None]]] = []
        cleanup_parent_descriptor = parent_descriptor
        owned_target_descriptor = -1
        if target_descriptor >= 0:
            owned_target_descriptor = target_descriptor
            target_descriptor = -1

        if primary_error is None and owned_target_descriptor >= 0:
            closing_target_descriptor = owned_target_descriptor
            owned_target_descriptor = -1
            try:
                os.close(closing_target_descriptor)
            except BaseException as error:
                # POSIX leaves a failed close's fd state ambiguous. Never retry it;
                # roll back through the still-held parent and captured inode instead.
                primary_error = error

        if primary_error is not None and target_identity is not None:
            expected_target_identity = target_identity

            def remove_created_target() -> None:
                _remove_descriptor_entry(
                    cleanup_parent_descriptor,
                    destination.name,
                    expected_identity=expected_target_identity,
                    source_descriptor=owned_target_descriptor,
                    label=f"stable-copy target {destination}",
                )

            cleanup_steps.append(
                (
                    f"Stable-copy target cleanup failed for {destination}",
                    remove_created_target,
                )
            )
        elif primary_error is not None and owned_target_descriptor >= 0:

            def bind_and_remove_created_target() -> None:
                nonlocal target_identity
                target_metadata = os.fstat(owned_target_descriptor)
                bound_target_identity = (
                    target_metadata.st_dev,
                    target_metadata.st_ino,
                )
                target_identity = bound_target_identity
                _remove_descriptor_entry(
                    cleanup_parent_descriptor,
                    destination.name,
                    expected_identity=bound_target_identity,
                    source_descriptor=owned_target_descriptor,
                    label=f"stable-copy target {destination}",
                )

            cleanup_steps.append(
                (
                    f"Stable-copy target cleanup failed for {destination}",
                    bind_and_remove_created_target,
                )
            )
        elif primary_error is not None and unbound_target_name is not None:
            preserved_target_name = unbound_target_name

            def report_preserved_unbound_target() -> None:
                raise RuntimeError(
                    "Stable-copy target creation could not be bound through an "
                    "owned descriptor; the candidate name was preserved without "
                    f"deletion at {destination.parent / preserved_target_name}"
                )

            cleanup_steps.append(
                (
                    f"Stable-copy target cleanup failed for {destination}",
                    report_preserved_unbound_target,
                )
            )
        if owned_target_descriptor >= 0:
            cleanup_steps.append(
                (
                    f"Stable-copy target descriptor cleanup failed for {destination}",
                    partial(os.close, owned_target_descriptor),
                )
            )
        if close_parent_descriptor and cleanup_parent_descriptor >= 0:
            if primary_error is None:
                closing_parent_descriptor = cleanup_parent_descriptor
                parent_descriptor = -1
                try:
                    os.close(closing_parent_descriptor)
                except BaseException as error:
                    primary_error = error
                    parent_identity = _require_present_invariant(
                        parent_identity,
                        label="stable-copy rollback parent identity",
                    )
                    target_identity = _require_present_invariant(
                        target_identity,
                        label="stable-copy rollback target identity",
                    )
                    cleanup_steps.append(
                        (
                            f"Stable-copy rollback failed for {destination}",
                            partial(
                                _remove_stable_copy_target_from_rebound_parent,
                                destination,
                                expected_parent_identity=parent_identity,
                                expected_target_identity=target_identity,
                            ),
                        )
                    )
            else:
                owned_parent_descriptor = cleanup_parent_descriptor
                parent_descriptor = -1
                cleanup_steps.append(
                    (
                        f"Stable-copy parent descriptor cleanup failed for {destination}",
                        partial(os.close, owned_parent_descriptor),
                    )
                )
        _run_cleanup_steps(
            cleanup_steps,
            primary_error=primary_error,
        )
    if primary_error is not None:
        raise primary_error
    target_identity = _require_present_invariant(
        target_identity,
        label="stable-copy target identity",
    )
    return target_identity


def _remove_private_snapshot_entry(
    parent_descriptor: int,
    entry_name: str,
    *,
    expected_identity: tuple[int, int],
    source_descriptor: int = -1,
) -> None:
    """Remove only the retained private inode, without crossing a mount."""

    try:
        os.stat(
            entry_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        if source_descriptor >= 0:
            source_metadata = os.fstat(source_descriptor)
            if (source_metadata.st_dev, source_metadata.st_ino) != expected_identity:
                raise RuntimeError(
                    "private snapshot source descriptor changed inode"
                ) from None
            if source_metadata.st_nlink != 0:
                raise RuntimeError(
                    "private snapshot entry disappeared while its retained inode "
                    "remains linked"
                ) from None
        return
    _remove_descriptor_entry(
        parent_descriptor,
        entry_name,
        expected_identity=expected_identity,
        source_descriptor=source_descriptor,
        label="private snapshot entry",
    )


def _cleanup_authoring_state(
    sealed_reports: _SealedReportSnapshots | None,
    sealed_generated: _SealedGeneratedArtifacts | None,
    staged: StagedJointRiggerArtifacts,
) -> None:
    """Clean facade-owned and backend staging state without skipping either."""

    steps: list[tuple[str, Callable[[], None]]] = []
    if sealed_reports is not None:
        steps.append(("Sealed report cleanup also failed", sealed_reports.cleanup))
    if sealed_generated is not None:
        steps.append(
            ("Sealed generated-artifact cleanup also failed", sealed_generated.cleanup)
        )
    steps.append(("Backend staging cleanup also failed", staged.cleanup))
    _run_cleanup_steps(steps)


def _substitute_sealed_report_snapshots(
    promotion: list[StagedArtifact],
    *,
    staged_targets: JointRiggerArtifactTargets,
    sealed_reports: _SealedReportSnapshots,
) -> list[StagedArtifact]:
    """Promote fd-bound report snapshots instead of backend-known paths."""

    replacements = {
        staged_targets.diagnostics_path: sealed_reports.diagnostics,
        staged_targets.result_path: sealed_reports.result,
    }
    substituted: list[StagedArtifact] = []
    for artifact in promotion:
        snapshot = replacements.get(artifact.staged_path)
        if snapshot is None:
            substituted.append(artifact)
            continue
        substituted.append(
            StagedArtifact(
                staged_path=snapshot.path,
                target_path=artifact.target_path,
                label=artifact.label,
                source_descriptor=snapshot.source_descriptor,
                source_sha256=snapshot.source_sha256,
                _initial_target_state=artifact._initial_target_state,
            )
        )
    return substituted


def _substitute_sealed_generated_artifacts(
    promotion: list[StagedArtifact],
    *,
    staged_targets: JointRiggerArtifactTargets,
    sealed_generated: _SealedGeneratedArtifacts,
) -> list[StagedArtifact]:
    """Promote the bound root and facade-owned sidecar instead of backend paths."""

    substituted: list[StagedArtifact] = []
    for artifact in promotion:
        if artifact.staged_path == staged_targets.output_path:
            substituted.append(
                StagedArtifact(
                    staged_path=sealed_generated.root.path,
                    target_path=artifact.target_path,
                    label=artifact.label,
                    source_descriptor=sealed_generated.root.source_descriptor,
                    source_sha256=sealed_generated.root.source_sha256,
                    _initial_target_state=artifact._initial_target_state,
                    # The staging reservation's commit evidence belongs to the
                    # backend inode.  This descriptor source is a distinct,
                    # facade-sealed inode, so transferring that state would let
                    # cleanup mistake a later move of the backend inode for the
                    # committed publication.
                )
            )
            continue
        if (
            sealed_generated.sidecar is not None
            and staged_targets.sidecar_path is not None
            and artifact.staged_path == staged_targets.sidecar_path
        ):
            substituted.append(
                StagedArtifact(
                    staged_path=sealed_generated.sidecar.path,
                    target_path=artifact.target_path,
                    label=artifact.label,
                    source_descriptor=sealed_generated.sidecar.source_descriptor,
                    source_sha256=sealed_generated.sidecar.source_tree_sha256,
                    _initial_target_state=artifact._initial_target_state,
                )
            )
            continue
        substituted.append(artifact)
    return substituted


def _revalidate_sealed_reports(
    expected: _ValidatedReports,
    sealed_reports: _SealedReportSnapshots,
) -> JointRiggerResultV1:
    """Bind promotion to the exact bounded bytes accepted before sealing."""

    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerDiagnosticsV1,
        JointRiggerResultV1,
    )

    observed_result = _load_sealed_model_report(
        sealed_reports.result,
        JointRiggerResultV1,
        "sealed result",
    )
    observed_diagnostics = _load_sealed_model_report(
        sealed_reports.diagnostics,
        JointRiggerDiagnosticsV1,
        "sealed diagnostics",
    )
    for label, observed, accepted in (
        ("result", observed_result, expected.result),
        ("diagnostics", observed_diagnostics, expected.diagnostics),
    ):
        if observed.payload != accepted.payload or (
            _canonical_model_payload(observed.model)
            != _canonical_model_payload(accepted.model)
        ):
            raise JointRiggerArtifactError(
                f"Sealed Joint Rigger {label} report changed after validation"
            )
    if _canonical_model_payload(
        observed_result.model.diagnostics
    ) != _canonical_model_payload(observed_diagnostics.model):
        raise JointRiggerArtifactError(
            "Sealed Joint Rigger result and diagnostics reports are inconsistent"
        )
    return cast("JointRiggerResultV1", observed_result.model)


def _revalidate_sealed_generated_artifacts(
    sealed_generated: _SealedGeneratedArtifacts,
) -> None:
    """Revalidate sealed generated artifacts immediately before promotion."""

    root = sealed_generated.root
    try:
        metadata = os.fstat(root.source_descriptor)
        path_metadata = os.stat(root.path, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != root.source_identity
            or (path_metadata.st_dev, path_metadata.st_ino) != root.source_identity
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != root.source_mode
            or metadata.st_mode & 0o222
        ):
            raise JointRiggerArtifactError(
                "Sealed generated root changed inode, mode, or link count"
            )
        observed_sha256 = _stable_descriptor_sha256(
            root.source_descriptor,
            label="sealed generated root",
        )
    except JointRiggerArtifactError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointRiggerArtifactError(
            f"Could not revalidate the sealed generated root: {exc}"
        ) from exc
    if observed_sha256 != root.source_sha256:
        raise JointRiggerArtifactError(
            "Sealed generated root content changed after validation"
        )

    sidecar = sealed_generated.sidecar
    if sidecar is not None:
        try:
            sidecar_metadata = os.fstat(sidecar.source_descriptor)
            path_metadata = os.stat(sidecar.path, follow_symlinks=False)
            if (
                not stat.S_ISDIR(sidecar_metadata.st_mode)
                or (sidecar_metadata.st_dev, sidecar_metadata.st_ino)
                != sidecar.source_identity
                or (path_metadata.st_dev, path_metadata.st_ino)
                != sidecar.source_identity
                or stat.S_IMODE(sidecar_metadata.st_mode) != sidecar.source_mode
                or sidecar_metadata.st_mode & 0o222
            ):
                raise JointRiggerArtifactError(
                    "Private composition sidecar changed inode or mode"
                )
            observed_tree_sha256 = directory_descriptor_tree_sha256(
                sidecar.source_descriptor
            )
            observed_bundle_sha256 = sidecar_dependency_bundle_sha256(sidecar.path)
        except JointRiggerArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise JointRiggerArtifactError(
                f"Could not revalidate the private composition sidecar: {exc}"
            ) from exc
        if observed_bundle_sha256 != sidecar.dependency_bundle_sha256:
            raise JointRiggerArtifactError(
                "Private composition sidecar changed after validation"
            )
        if observed_tree_sha256 != sidecar.source_tree_sha256:
            raise JointRiggerArtifactError(
                "Private composition sidecar tree changed after validation"
            )

    for dependency in sealed_generated.dependencies:
        try:
            _require_sealed_dependency_snapshot(dependency)
        except JointRiggerArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise JointRiggerArtifactError(
                f"Could not revalidate sealed dependency {dependency.path}: {exc}"
            ) from exc


def _validate_sealed_generated_composition(
    result: JointRiggerResultV1,
    staged_artifacts: StagedJointRiggerArtifacts,
    sealed_generated: _SealedGeneratedArtifacts,
    *,
    capture_dependencies: bool,
) -> tuple[_SealedDependencySnapshot, ...]:
    """Prove composition from retained descriptors at publication coordinates."""

    output_artifact = result.output_artifact
    if output_artifact is None:  # pragma: no cover - validated result invariant
        raise JointRiggerArtifactError(
            "A succeeded Joint Rigger result must identify its generated root"
        )
    expected_bundle_sha256 = output_artifact.dependency_bundle_sha256
    if expected_bundle_sha256 is None:  # pragma: no cover - validated invariant
        raise JointRiggerArtifactError(
            "A succeeded Joint Rigger result must identify its dependency bundle"
        )
    if sealed_generated.root.path.suffix.lower() == ".usdz":
        if sealed_generated.sidecar is not None:
            raise JointRiggerArtifactError(
                "A self-contained USDZ output must not declare a composition sidecar"
            )
        if sealed_generated.dependencies or sealed_generated.dependency_records:
            raise JointRiggerArtifactError(
                "A self-contained USDZ output must not retain external dependencies"
            )
        if capture_dependencies:
            if sealed_generated.package_identity is not None:
                raise JointRiggerArtifactError(
                    "The sealed USDZ package was already composition-validated"
                )
            sealed_generated.package_identity = _validate_sealed_usdz_composition(
                sealed_generated.root,
                staged_artifacts.staged_targets,
                uri=output_artifact.uri,
            )
        package_identity = sealed_generated.package_identity
        if package_identity is None:
            raise JointRiggerArtifactError(
                "The sealed USDZ package has no retained composition identity"
            )
        if package_identity != output_artifact:
            raise JointRiggerArtifactError(
                "The sealed USDZ package identity does not match the backend result"
            )
        return ()
    if sealed_generated.sidecar is not None:
        if sealed_generated.dependencies or sealed_generated.dependency_records:
            raise JointRiggerArtifactError(
                "A sidecar-backed output must not retain external dependencies"
            )
        _validate_sealed_sidecar_composition(
            sealed_generated,
            staged_artifacts.staged_targets,
        )
        return ()

    dependencies = sealed_generated.dependencies
    captured_here = False
    try:
        if capture_dependencies:
            if dependencies:
                raise JointRiggerArtifactError(
                    "No-sidecar dependencies were already captured"
                )
            dependencies, dependency_records = _capture_sealed_no_sidecar_dependencies(
                sealed_generated.root,
                staged_artifacts,
            )
            sealed_generated.dependency_records = dependency_records
            captured_here = True
        _validate_sealed_no_sidecar_composition(
            sealed_generated.root,
            dependencies,
            sealed_generated.dependency_records,
            staged_artifacts,
            uri=output_artifact.uri,
            expected_bundle_sha256=expected_bundle_sha256,
        )
        return dependencies
    except BaseException as error:
        if captured_here:
            try:
                _run_cleanup_steps(
                    (
                        f"Sealed dependency {dependency.path} cleanup also failed",
                        dependency.cleanup,
                    )
                    for dependency in dependencies
                )
            except BaseException as cleanup_error:
                _attach_cleanup_failure(
                    error,
                    cleanup_error,
                    context="Sealed dependency cleanup also failed",
                )
            sealed_generated.dependency_records = ()
        raise


def _validate_sealed_sidecar_composition(
    sealed_generated: _SealedGeneratedArtifacts,
    staged_targets: JointRiggerArtifactTargets,
) -> None:
    """Project a retained root and sidecar under their exact publication names."""

    sidecar = sealed_generated.sidecar
    assert sidecar is not None
    publication_output = staged_targets.publication_output_path
    publication_sidecar = staged_targets.publication_sidecar_path
    if publication_output is None or publication_sidecar is None:
        raise JointRiggerArtifactError(
            "A composition sidecar requires complete publication path metadata"
        )
    try:
        with _private_directory_owner(
            publication_output.parent,
            prefix=f".{publication_output.name}.sealed-validate-",
        ) as owner:
            projected_output = owner / publication_output.name
            projected_sidecar = owner / publication_sidecar.name
            _copy_stable_regular_descriptor(
                sealed_generated.root.source_descriptor,
                projected_output,
                expected_identity=sealed_generated.root.source_identity,
                expected_sha256=sealed_generated.root.source_sha256,
                expected_mode=sealed_generated.root.source_mode,
                expected_nlink=1,
                label="sealed generated root",
            )
            projected_sidecar.mkdir(mode=0o700)
            target_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            target_flags |= getattr(os, "O_CLOEXEC", 0)
            target_descriptor = os.open(projected_sidecar, target_flags)
            target_error: BaseException | None = None
            try:
                source_before_sha256 = directory_descriptor_tree_sha256(
                    sidecar.source_descriptor
                )
                if source_before_sha256 != sidecar.source_tree_sha256:
                    raise JointRiggerArtifactError(
                        "Private composition sidecar changed before projection"
                    )
                copy_directory_descriptor_tree(
                    sidecar.source_descriptor,
                    target_descriptor,
                    label="sealed composition sidecar",
                    preserve_modes=False,
                )
                source_after_sha256 = directory_descriptor_tree_sha256(
                    sidecar.source_descriptor
                )
                if source_after_sha256 != sidecar.source_tree_sha256:
                    raise JointRiggerArtifactError(
                        "Private composition sidecar changed during projection"
                    )
            except BaseException as error:
                target_error = error
                raise
            finally:
                owned_target_descriptor = target_descriptor
                target_descriptor = -1
                _run_cleanup_steps(
                    [
                        (
                            "Projected sidecar descriptor cleanup failed",
                            partial(os.close, owned_target_descriptor),
                        )
                    ],
                    primary_error=target_error,
                )
            if (
                sidecar_dependency_bundle_sha256(projected_sidecar)
                != sidecar.dependency_bundle_sha256
            ):
                raise JointRiggerArtifactError(
                    "Projected sealed composition sidecar changed bundle identity"
                )
            dependencies = _local_usd_dependency_paths(
                projected_output,
                label="sealed generated root publication projection",
            )
            _reject_uri_usd_dependencies(projected_output)
            normalized_sidecar = projected_sidecar.resolve(strict=True)
            sidecar_dependencies = []
            for dependency in dependencies:
                normalized = dependency.resolve(strict=True)
                if not _is_relative_to(normalized, normalized_sidecar):
                    raise JointRiggerArtifactError(
                        "The sealed generated root has a dependency outside its "
                        f"composition sidecar: {dependency}"
                    )
                sidecar_dependencies.append(normalized)
            _validate_opaque_dependency_closure(
                sidecar_dependencies,
                allowed_root=normalized_sidecar,
            )
            if not sidecar_dependencies and any(projected_sidecar.iterdir()):
                raise JointRiggerArtifactError(
                    "The sealed generated root does not reference its non-empty "
                    "composition sidecar"
                )
    except JointRiggerFacadeError:
        raise
    except ImportError as exc:
        raise JointRiggerBackendUnavailableError(
            "Could not validate sealed sidecar composition because a required "
            f"runtime dependency is unavailable: {exc}"
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointRiggerArtifactError(
            f"Could not validate sealed sidecar composition: {exc}"
        ) from exc


def _validate_sealed_usdz_composition(
    root: _SealedGeneratedRoot,
    staged_targets: JointRiggerArtifactTargets,
    *,
    uri: str,
) -> ArtifactIdentityV1:
    """Validate one package-internal closure at its publication coordinates."""

    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerContractError,
    )
    from world_understanding.functions.physics.joint_rigger.reference import (
        identify_usd_artifact,
        local_usd_dependency_paths,
    )
    from world_understanding.functions.physics.joint_rigger.source_binding import (
        _validate_bound_projection_dependencies,
    )

    publication_output = staged_targets.publication_output_path
    if publication_output is None:  # pragma: no cover - artifact invariant
        raise JointRiggerArtifactError(
            "Staged targets do not declare publication_output_path"
        )
    publication_output = _lexical_absolute_path(publication_output)
    try:
        with _private_directory_owner(
            publication_output.parent,
            prefix=f".{publication_output.name}.sealed-usdz-validate-",
        ) as owner:
            projected_output = owner / publication_output.name
            _copy_stable_regular_descriptor(
                root.source_descriptor,
                projected_output,
                expected_identity=root.source_identity,
                expected_sha256=root.source_sha256,
                expected_mode=root.source_mode,
                expected_nlink=1,
                label="sealed generated USDZ root",
            )
            _validate_bound_projection_dependencies(
                projected_output,
                projection_root=owner,
                materialized_paths=frozenset({projected_output}),
                layer_paths=frozenset(),
                restore_paths={},
            )
            _reject_uri_usd_dependencies(projected_output)
            _reject_symlink_usd_dependencies(projected_output)
            local_dependencies = {
                dependency.expanduser().resolve(strict=True)
                for dependency in local_usd_dependency_paths(projected_output)
            }
            expected_local_dependencies = {projected_output.resolve(strict=True)}
            if local_dependencies != expected_local_dependencies:
                external = sorted(local_dependencies - expected_local_dependencies)
                raise JointRiggerArtifactError(
                    "The sealed USDZ package has dependencies outside its archive: "
                    f"{external}"
                )
            return identify_usd_artifact(projected_output, uri=uri)
    except JointRiggerFacadeError:
        raise
    except ImportError as exc:
        raise JointRiggerBackendUnavailableError(
            "Could not validate the sealed USDZ package because a required "
            f"runtime dependency is unavailable: {exc}"
        ) from exc
    except JointRiggerContractError as exc:
        raise JointRiggerArtifactError(
            f"Could not verify sealed USDZ package identity: {exc}"
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointRiggerArtifactError(
            f"Could not validate sealed USDZ package composition: {exc}"
        ) from exc


def _capture_sealed_no_sidecar_dependencies(
    root: _SealedGeneratedRoot,
    staged_artifacts: StagedJointRiggerArtifacts,
) -> tuple[
    tuple[_SealedDependencySnapshot, ...],
    tuple[_CapturedDependencyIdentityRecord, ...],
]:
    """Discover from sealed root bytes and retain every external dependency fd."""

    from world_understanding.functions.physics.joint_rigger.reference import (
        _capture_dependency_structure,
        _CapturedDependencyIdentityRecord,
    )

    staged_targets = staged_artifacts.staged_targets
    final_targets = staged_artifacts.final_targets
    publication_output = staged_targets.publication_output_path
    if publication_output is None:  # pragma: no cover - artifact invariant
        raise JointRiggerArtifactError(
            "Staged targets do not declare publication_output_path"
        )
    discovery_descriptor, discovery_value = tempfile.mkstemp(
        dir=publication_output.parent,
        prefix=f".{publication_output.name}.sealed-discovery-",
        suffix=publication_output.suffix,
    )
    discovery_path = Path(discovery_value)
    discovery_parent_descriptor = -1
    placeholder_identity: tuple[int, int] | None = None
    acquisition_error: BaseException | None = None
    try:
        discovery_parent_descriptor = _open_stable_copy_parent(discovery_path)
        placeholder_metadata = os.fstat(discovery_descriptor)
        placeholder_identity = (
            placeholder_metadata.st_dev,
            placeholder_metadata.st_ino,
        )
    except BaseException as error:
        acquisition_error = error
    finally:
        placeholder_cleanup_steps: list[tuple[str, Callable[[], None]]] = []
        cleanup_parent_descriptor = discovery_parent_descriptor
        owned_discovery_descriptor = discovery_descriptor
        discovery_descriptor = -1
        if cleanup_parent_descriptor >= 0:

            def remove_discovery_placeholder() -> None:
                nonlocal placeholder_identity
                if placeholder_identity is None:
                    placeholder_metadata = os.fstat(owned_discovery_descriptor)
                    placeholder_identity = (
                        placeholder_metadata.st_dev,
                        placeholder_metadata.st_ino,
                    )
                _remove_descriptor_entry(
                    cleanup_parent_descriptor,
                    discovery_path.name,
                    expected_identity=placeholder_identity,
                    source_descriptor=owned_discovery_descriptor,
                    label=f"sealed dependency discovery placeholder {discovery_path}",
                )

            placeholder_cleanup_steps.append(
                (
                    "Sealed dependency discovery placeholder cleanup failed",
                    remove_discovery_placeholder,
                )
            )
        elif acquisition_error is not None:

            def report_preserved_placeholder() -> None:
                raise RuntimeError(
                    "Sealed dependency discovery placeholder could not be bound; "
                    f"the private name was preserved at {discovery_path}"
                )

            placeholder_cleanup_steps.append(
                (
                    "Sealed dependency discovery placeholder cleanup failed",
                    report_preserved_placeholder,
                )
            )
        placeholder_cleanup_steps.append(
            (
                "Sealed dependency discovery descriptor close failed",
                partial(os.close, owned_discovery_descriptor),
            )
        )
        if cleanup_parent_descriptor >= 0:
            discovery_parent_descriptor = -1
            placeholder_cleanup_steps.append(
                (
                    "Sealed dependency discovery placeholder parent cleanup failed",
                    partial(os.close, cleanup_parent_descriptor),
                )
            )
        _run_cleanup_steps(
            placeholder_cleanup_steps,
            primary_error=acquisition_error,
        )
    if acquisition_error is not None:
        raise acquisition_error
    snapshots: list[_SealedDependencySnapshot] = []
    discovery_identity: tuple[int, int] | None = None
    captured_result: tuple[
        tuple[_SealedDependencySnapshot, ...],
        tuple[_CapturedDependencyIdentityRecord, ...],
    ]
    try:
        discovery_parent_descriptor = _open_stable_copy_parent(discovery_path)
        discovery_identity = _copy_stable_regular_descriptor(
            root.source_descriptor,
            discovery_path,
            expected_identity=root.source_identity,
            expected_sha256=root.source_sha256,
            expected_mode=root.source_mode,
            expected_nlink=1,
            label="sealed generated root",
            destination_parent_descriptor=discovery_parent_descriptor,
        )
        _reject_uri_usd_dependencies(discovery_path)
        _reject_symlink_usd_dependencies(discovery_path)
        dependency_paths = _local_usd_dependency_paths(
            discovery_path,
            label="sealed generated root dependency discovery",
        )
        normalized_dependencies: set[Path] = set()
        for dependency in dependency_paths:
            normalized = _normalize_local_path_without_symlinks(
                dependency,
                symlink_error="Generated dependency path must not contain symlinks",
            )
            normalized_dependencies.add(normalized)
        publication_output = _lexical_absolute_path(publication_output)
        publication_current = publication_output.resolve(strict=False)
        captured_publication = sorted(
            dependency
            for dependency in normalized_dependencies
            if dependency in {publication_output, publication_current}
        )
        if captured_publication:
            raise JointRiggerArtifactError(
                "The sealed generated root captures the existing publication root: "
                f"{captured_publication}"
            )
        if root.path in normalized_dependencies:
            raise JointRiggerArtifactError(
                "The sealed generated root depends on its physical staging path: "
                f"{root.path}"
            )
        ordered_dependencies = tuple(
            sorted(normalized_dependencies, key=lambda path: path.as_posix())
        )
        first_structure = _capture_dependency_structure(
            discovery_path,
            logical_artifact_path=publication_output,
        )
        package_dependencies = sorted(
            record.package_inner_locator
            for record in first_structure
            if record.package_inner_locator is not None
        )
        if package_dependencies:
            raise JointRiggerArtifactError(
                "No-sidecar publication does not support package-relative USD "
                f"dependencies: {package_dependencies}"
            )
        record_backings = {
            record.backing_path
            for record in first_structure
            if record.backing_path is not None
        }
        if record_backings != set(ordered_dependencies):
            raise JointRiggerArtifactError(
                "The sealed dependency inventory and identity records disagree: "
                f"inventory={sorted(ordered_dependencies)}, "
                f"records={sorted(record_backings)}"
            )
        _reject_transaction_target_dependencies(
            ordered_dependencies,
            staged_targets=staged_targets,
            final_targets=final_targets,
        )
        for dependency in ordered_dependencies:
            snapshots.append(_open_sealed_dependency_snapshot(dependency))
        snapshots_by_path = {snapshot.path: snapshot for snapshot in snapshots}
        second_structure = _capture_dependency_structure(
            discovery_path,
            logical_artifact_path=publication_output,
        )
        if second_structure != first_structure:
            raise JointRiggerArtifactError(
                "The generated dependency structure changed while its backing "
                "files were retained"
            )
        identity_records: list[_CapturedDependencyIdentityRecord] = []
        for record in first_structure:
            if record.backing_path is None:
                if record.kind != "stage_root_layer" or record.locator != "$artifact":
                    raise JointRiggerArtifactError(
                        "The sealed root dependency structure is inconsistent"
                    )
                record_sha256 = root.source_sha256
            else:
                snapshot = snapshots_by_path.get(record.backing_path)
                if snapshot is None:  # pragma: no cover - inventory invariant
                    raise JointRiggerArtifactError(
                        "A captured dependency has no retained backing file: "
                        f"{record.backing_path}"
                    )
                record_sha256 = snapshot.source_sha256
            identity_records.append(
                _CapturedDependencyIdentityRecord(
                    kind=record.kind,
                    locator=record.locator,
                    sha256=record_sha256,
                    backing_path=record.backing_path,
                )
            )
        if not identity_records:
            raise JointRiggerArtifactError(
                "The generated dependency structure does not contain a root"
            )
        captured_result = (tuple(snapshots), tuple(identity_records))
    except BaseException as error:
        cleanup_steps: list[tuple[str, Callable[[], None]]] = [
            (
                f"Sealed dependency {snapshot.path} cleanup also failed",
                snapshot.cleanup,
            )
            for snapshot in snapshots
        ]
        if discovery_identity is not None and discovery_parent_descriptor >= 0:
            cleanup_steps.append(
                (
                    "Sealed dependency discovery cleanup also failed",
                    partial(
                        _remove_descriptor_entry,
                        discovery_parent_descriptor,
                        discovery_path.name,
                        expected_identity=discovery_identity,
                        label=f"sealed dependency discovery {discovery_path}",
                    ),
                )
            )
        if discovery_parent_descriptor >= 0:
            owned_parent_descriptor = discovery_parent_descriptor
            discovery_parent_descriptor = -1
            cleanup_steps.append(
                (
                    "Sealed dependency discovery parent cleanup also failed",
                    partial(os.close, owned_parent_descriptor),
                )
            )
        _run_cleanup_steps(
            cleanup_steps,
            primary_error=error,
        )
        raise

    discovery_cleanup_steps: list[tuple[str, Callable[[], None]]] = []
    assert discovery_identity is not None
    assert discovery_parent_descriptor >= 0
    discovery_cleanup_steps.append(
        (
            "Sealed dependency discovery cleanup failed",
            partial(
                _remove_descriptor_entry,
                discovery_parent_descriptor,
                discovery_path.name,
                expected_identity=discovery_identity,
                label=f"sealed dependency discovery {discovery_path}",
            ),
        )
    )
    owned_parent_descriptor = discovery_parent_descriptor
    discovery_parent_descriptor = -1
    discovery_cleanup_steps.append(
        (
            "Sealed dependency discovery parent cleanup failed",
            partial(os.close, owned_parent_descriptor),
        )
    )
    try:
        _run_cleanup_steps(discovery_cleanup_steps)
    except BaseException as error:
        _run_cleanup_steps(
            (
                (
                    f"Sealed dependency {snapshot.path} cleanup also failed",
                    snapshot.cleanup,
                )
                for snapshot in snapshots
            ),
            primary_error=error,
        )
        if not isinstance(error, Exception):
            raise
        raise JointRiggerArtifactError(
            f"Could not clean the sealed dependency discovery artifact: {error}"
        ) from error
    return captured_result


def _validate_sealed_no_sidecar_composition(
    root: _SealedGeneratedRoot,
    dependencies: tuple[_SealedDependencySnapshot, ...],
    dependency_records: tuple[_CapturedDependencyIdentityRecord, ...],
    staged_artifacts: StagedJointRiggerArtifacts,
    *,
    uri: str,
    expected_bundle_sha256: str,
) -> None:
    """Project retained no-sidecar descriptors and verify the final closure."""

    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerContractError,
    )
    from world_understanding.functions.physics.joint_rigger.reference import (
        _artifact_identity_from_captured_records,
    )

    staged_targets = staged_artifacts.staged_targets
    publication_output = staged_targets.publication_output_path
    if publication_output is None:  # pragma: no cover - artifact invariant
        raise JointRiggerArtifactError(
            "Staged targets do not declare publication_output_path"
        )
    publication_output = _lexical_absolute_path(publication_output)
    try:
        snapshots_by_path = {dependency.path: dependency for dependency in dependencies}
        record_backings = {
            record.backing_path
            for record in dependency_records
            if record.backing_path is not None
        }
        if record_backings != set(snapshots_by_path):
            raise JointRiggerArtifactError(
                "Frozen dependency records do not match retained dependencies"
            )
        root_records = [
            record for record in dependency_records if record.backing_path is None
        ]
        if (
            len(root_records) != 1
            or root_records[0].kind != "stage_root_layer"
            or root_records[0].locator != "$artifact"
            or root_records[0].sha256 != root.source_sha256
        ):
            raise JointRiggerArtifactError(
                "Frozen dependency records do not bind the sealed root"
            )
        for record in dependency_records:
            if record.backing_path is None:
                continue
            snapshot = snapshots_by_path[record.backing_path]
            if record.sha256 != snapshot.source_sha256:
                raise JointRiggerArtifactError(
                    "Frozen dependency record does not bind retained bytes: "
                    f"{record.backing_path}"
                )
        identity = _artifact_identity_from_captured_records(
            logical_artifact_path=publication_output,
            uri=uri,
            root_sha256=root.source_sha256,
            records=dependency_records,
        )
        if identity.dependency_bundle_sha256 != expected_bundle_sha256:
            raise JointRiggerArtifactError(
                "The sealed generated root records changed dependency identity"
            )
        for dependency in dependencies:
            _require_sealed_dependency_snapshot(dependency)
        with _private_directory_owner(
            publication_output.parent,
            prefix=f".{publication_output.name}.sealed-validate-",
        ) as owner:
            projection_root = _lexical_absolute_path(owner / "filesystem")

            def projected_path(path: Path) -> Path:
                absolute = _lexical_absolute_path(path)
                anchor = Path(absolute.anchor)
                projected = _lexical_absolute_path(
                    projection_root / absolute.relative_to(anchor)
                )
                if not _is_relative_to(projected, projection_root):
                    raise JointRiggerArtifactError(
                        "The sealed publication projection escapes its isolated "
                        f"filesystem tree: {path}"
                    )
                return projected

            projected_to_published_dependency: dict[Path, Path] = {}
            projected_output = projected_path(publication_output)
            for dependency in dependencies:
                projected_dependency = projected_path(dependency.path)
                if projected_dependency == projected_output:
                    raise JointRiggerArtifactError(
                        "The sealed generated root captures its publication path: "
                        f"{publication_output}"
                    )
                projected_dependency.parent.mkdir(parents=True, exist_ok=True)
                _copy_stable_regular_descriptor(
                    dependency.source_descriptor,
                    projected_dependency,
                    expected_identity=dependency.source_identity,
                    expected_sha256=dependency.source_sha256,
                    expected_mode=stat.S_IMODE(dependency.source_state[2]),
                    expected_nlink=dependency.source_state[3],
                    label=f"sealed generated dependency {dependency.path}",
                )
                projected_to_published_dependency[
                    projected_dependency.resolve(strict=True)
                ] = dependency.path

            projected_output.parent.mkdir(parents=True, exist_ok=True)
            _copy_stable_regular_descriptor(
                root.source_descriptor,
                projected_output,
                expected_identity=root.source_identity,
                expected_sha256=root.source_sha256,
                expected_mode=root.source_mode,
                expected_nlink=1,
                label="sealed generated root",
            )
            projected_dependencies = _local_usd_dependency_paths(
                projected_output,
                label="sealed generated root publication projection",
            )
            normalized_projected_dependencies = {
                dependency.expanduser().resolve(strict=True)
                for dependency in projected_dependencies
            }
            _validate_opaque_dependency_closure(
                normalized_projected_dependencies,
                allowed_files=normalized_projected_dependencies,
            )
            published_dependencies = {
                projected_to_published_dependency.get(dependency, dependency)
                for dependency in normalized_projected_dependencies
            }
            expected_dependencies = {dependency.path for dependency in dependencies}
            if published_dependencies != expected_dependencies:
                missing = sorted(expected_dependencies - published_dependencies)
                added = sorted(published_dependencies - expected_dependencies)
                raise JointRiggerArtifactError(
                    "The sealed generated root dependency closure changes at its "
                    f"publication name; missing={missing}, added={added}"
                )
        for dependency in dependencies:
            _require_sealed_dependency_snapshot(dependency)
    except JointRiggerFacadeError:
        raise
    except ImportError as exc:
        raise JointRiggerBackendUnavailableError(
            "Could not verify sealed publication composition because a required "
            f"runtime dependency is unavailable: {exc}"
        ) from exc
    except JointRiggerContractError as exc:
        raise JointRiggerArtifactError(
            f"Could not verify sealed publication composition: {exc}"
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointRiggerArtifactError(
            f"Could not validate sealed publication composition: {exc}"
        ) from exc


def _validate_result_identity(
    request: JointRiggerInputV1,
    result: JointRiggerResultV1,
    staged_artifacts: StagedJointRiggerArtifacts,
) -> None:
    from world_understanding.functions.physics.joint_rigger.models import (
        canonical_sha256,
    )

    staged_targets = staged_artifacts.staged_targets
    if result.status != "succeeded":
        raise JointRiggerArtifactError(
            "A backend may publish a generated root only with status=succeeded; "
            f"got {result.status}"
        )
    expected_input_sha256 = canonical_sha256(request)
    if result.input_sha256 != expected_input_sha256:
        raise JointRiggerArtifactError(
            "Joint Rigger result input_sha256 does not match the canonical request"
        )
    expected_plan_sha256 = canonical_sha256(request.plan)
    if result.plan_sha256 != expected_plan_sha256:
        raise JointRiggerArtifactError(
            "Joint Rigger result plan_sha256 does not match the canonical plan"
        )
    output_artifact = result.output_artifact
    if output_artifact is None:
        raise JointRiggerArtifactError(
            "A succeeded Joint Rigger result must identify its generated root"
        )
    _validate_output_artifact_uri(output_artifact.uri, staged_targets)
    staged_output_path = staged_targets.output_path
    if staged_output_path.is_symlink() or not staged_output_path.is_file():
        raise JointRiggerArtifactError(
            f"Backend did not write a regular generated root: {staged_output_path}"
        )
    actual_output_sha256 = _file_sha256(staged_output_path)
    if output_artifact.root_sha256 != actual_output_sha256:
        raise JointRiggerArtifactError(
            "Joint Rigger output_artifact root_sha256 does not match the generated root"
        )
    expected_bundle_sha256 = output_artifact.dependency_bundle_sha256
    if expected_bundle_sha256 is None:
        raise JointRiggerArtifactError(
            "A succeeded Joint Rigger result must claim output_artifact "
            "dependency_bundle_sha256"
        )
    if staged_targets.sidecar_path is not None:
        try:
            actual_bundle_sha256 = sidecar_dependency_bundle_sha256(
                staged_targets.sidecar_path
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise JointRiggerArtifactError(
                f"Backend wrote an invalid composition sidecar: {exc}"
            ) from exc
        if expected_bundle_sha256 != actual_bundle_sha256:
            raise JointRiggerArtifactError(
                "Joint Rigger output_artifact dependency_bundle_sha256 does not "
                "match the staged composition sidecar"
            )
        _validate_staged_sidecar_composition(staged_targets)
    else:
        actual_output_artifact = _project_no_sidecar_output_identity(
            staged_artifacts,
            uri=output_artifact.uri,
        )
        if expected_bundle_sha256 != actual_output_artifact.dependency_bundle_sha256:
            raise JointRiggerArtifactError(
                "Joint Rigger output_artifact dependency_bundle_sha256 does not "
                "match the projected publication USD dependency closure"
            )


def _validate_output_artifact_uri(
    uri: str,
    staged_targets: JointRiggerArtifactTargets,
) -> None:
    """Bind local output identities to the immutable publication location."""

    parsed = urlparse(uri)
    local_path: Path | None
    if parsed.scheme == "file":
        local_path = _canonical_file_uri_path(uri, label="output_artifact.uri")
    else:
        local_path = _local_artifact_path(uri)
    if local_path is None:
        return
    publication_path = staged_targets.publication_output_path
    if publication_path is None:  # pragma: no cover - artifact model invariant
        raise JointRiggerArtifactError(
            "Staged targets do not declare publication_output_path"
        )
    # Compare the immutable lexical publication location, not the path's
    # current filesystem referent.  A caller may intentionally replace an
    # existing output symlink during commit; resolving that symlink here would
    # let a backend bind its result to the old referent, which stops aliasing
    # the publication path as soon as the generated root is promoted.
    normalized_uri_path = _lexical_absolute_path(local_path)
    normalized_publication_path = _lexical_absolute_path(publication_path)
    if normalized_uri_path != normalized_publication_path:
        raise JointRiggerArtifactError(
            "A local output_artifact.uri must identify "
            f"publication_output_path={publication_path}; got {uri}"
        )


def _validate_staged_sidecar_composition(
    staged_targets: JointRiggerArtifactTargets,
) -> None:
    """Prove a staged root is self-contained in its declared sidecar layout."""

    staged_sidecar = staged_targets.sidecar_path
    publication_output = staged_targets.publication_output_path
    publication_sidecar = staged_targets.publication_sidecar_path
    if (
        staged_sidecar is None
        or publication_output is None
        or publication_sidecar is None
    ):
        raise JointRiggerArtifactError(
            "A composition sidecar requires complete publication path metadata"
        )

    try:
        with _private_directory_owner(
            publication_output.parent,
            prefix=f".{publication_output.name}.validate-",
        ) as owner:
            projected_output = owner / publication_output.name
            projected_sidecar = owner / publication_sidecar.name
            _copy_stable_regular_file(staged_targets.output_path, projected_output)
            projected_sidecar.mkdir(mode=0o700)
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            projected_descriptor = os.open(projected_sidecar, flags)
            copy_error: BaseException | None = None
            try:
                copy_sidecar_directory(
                    staged_sidecar,
                    projected_descriptor,
                    label="staged composition sidecar projection",
                )
            except BaseException as error:
                copy_error = error
                raise
            finally:
                _run_cleanup_steps(
                    [
                        (
                            "Projected staged sidecar descriptor cleanup failed",
                            partial(os.close, projected_descriptor),
                        )
                    ],
                    primary_error=copy_error,
                )
            dependencies = _local_usd_dependency_paths(
                projected_output,
                label="generated root publication projection",
            )
            _reject_uri_usd_dependencies(projected_output)
            normalized_sidecar = projected_sidecar.resolve(strict=True)
            sidecar_dependencies = []
            for dependency in dependencies:
                normalized = dependency.resolve(strict=True)
                if not _is_relative_to(normalized, normalized_sidecar):
                    raise JointRiggerArtifactError(
                        "The staged generated root has a dependency outside its "
                        "declared publication sidecar: "
                        f"{dependency}"
                    )
                sidecar_dependencies.append(normalized)
            _validate_opaque_dependency_closure(
                sidecar_dependencies,
                allowed_root=normalized_sidecar,
            )
            if not sidecar_dependencies and any(projected_sidecar.iterdir()):
                raise JointRiggerArtifactError(
                    "The staged generated root does not reference its non-empty "
                    "declared publication sidecar"
                )
    except JointRiggerFacadeError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointRiggerArtifactError(
            f"Could not validate the staged publication sidecar layout: {exc}"
        ) from exc


def _project_no_sidecar_output_identity(
    staged_artifacts: StagedJointRiggerArtifacts,
    *,
    uri: str,
) -> ArtifactIdentityV1:
    """Identify a no-sidecar root in an isolated publication-path projection."""

    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerContractError,
    )
    from world_understanding.functions.physics.joint_rigger.reference import (
        identify_usd_artifact,
    )

    staged_targets = staged_artifacts.staged_targets
    final_targets = staged_artifacts.final_targets
    staged_output = staged_targets.output_path.expanduser().resolve(strict=True)
    publication_output = staged_targets.publication_output_path
    if publication_output is None:  # pragma: no cover - artifact model invariant
        raise JointRiggerArtifactError(
            "Staged targets do not declare publication_output_path"
        )
    publication_output = _lexical_absolute_path(publication_output)
    publication_current = publication_output.resolve(strict=False)

    try:
        dependencies = _local_usd_dependency_paths(
            staged_output,
            label="generated root staging closure",
        )
        _reject_symlink_usd_dependencies(staged_output)
        normalized_dependencies = tuple(
            dependency.expanduser().resolve(strict=True) for dependency in dependencies
        )
        _validate_opaque_dependency_closure(
            normalized_dependencies,
            allowed_files=set(normalized_dependencies),
        )
        captured_publication = sorted(
            dependency
            for dependency in normalized_dependencies
            if dependency in {publication_output, publication_current}
        )
        if captured_publication:
            raise JointRiggerArtifactError(
                "The staged generated root captures the existing publication root: "
                f"{captured_publication}"
            )
        _reject_transaction_target_dependencies(
            normalized_dependencies,
            staged_targets=staged_targets,
            final_targets=final_targets,
        )

        with _private_directory_owner(
            publication_output.parent,
            prefix=f".{publication_output.name}.validate-",
        ) as owner:
            projection_root = _lexical_absolute_path(owner / "filesystem")
            projected_to_published_dependency: dict[Path, Path] = {}

            def projected_path(path: Path) -> Path:
                absolute = _lexical_absolute_path(path)
                anchor = Path(absolute.anchor)
                projected = _lexical_absolute_path(
                    projection_root / absolute.relative_to(anchor)
                )
                if not _is_relative_to(projected, projection_root):
                    raise JointRiggerArtifactError(
                        "The publication projection path escapes its isolated "
                        f"filesystem tree: {path}"
                    )
                return projected

            projected_output = projected_path(publication_output)
            for dependency in normalized_dependencies:
                if dependency == staged_output:
                    raise JointRiggerArtifactError(
                        "The staged generated root depends on its physical staging "
                        f"path: {staged_output}"
                    )
                projected_dependency = projected_path(dependency)
                if projected_dependency == projected_output:
                    raise JointRiggerArtifactError(
                        "The staged generated root captures its publication path: "
                        f"{publication_output}"
                    )
                projected_dependency.parent.mkdir(parents=True, exist_ok=True)
                _copy_stable_regular_file(dependency, projected_dependency)
                projected_to_published_dependency[
                    projected_dependency.resolve(strict=True)
                ] = dependency

            projected_output.parent.mkdir(parents=True, exist_ok=True)
            _copy_stable_regular_file(staged_output, projected_output)
            projected_dependencies = _local_usd_dependency_paths(
                projected_output,
                label="generated root publication projection",
            )
            normalized_projected_dependencies = {
                dependency.expanduser().resolve(strict=True)
                for dependency in projected_dependencies
            }
            _validate_opaque_dependency_closure(
                normalized_projected_dependencies,
                allowed_files=normalized_projected_dependencies,
            )
            if staged_output in normalized_projected_dependencies:
                raise JointRiggerArtifactError(
                    "The projected publication depends on the physical staging path: "
                    f"{staged_output}"
                )
            _reject_transaction_target_dependencies(
                normalized_projected_dependencies,
                staged_targets=staged_targets,
                final_targets=final_targets,
            )
            published_dependencies = {
                projected_to_published_dependency.get(dependency, dependency)
                for dependency in normalized_projected_dependencies
            }
            expected_dependencies = set(normalized_dependencies)
            if published_dependencies != expected_dependencies:
                missing = sorted(expected_dependencies - published_dependencies)
                added = sorted(published_dependencies - expected_dependencies)
                raise JointRiggerArtifactError(
                    "The generated root dependency closure changes at its "
                    "publication name; "
                    f"missing={missing}, added={added}"
                )
            # Opening the projection proves the closure resolves with the final
            # basename, while the equality above proves it resolves to the same
            # files. The staged root remains beside the publication root, so its
            # canonical dependency locators are exactly the published locators.
            identify_usd_artifact(projected_output, uri=uri)
            return identify_usd_artifact(staged_output, uri=uri)
    except JointRiggerFacadeError:
        raise
    except ImportError as exc:
        raise JointRiggerBackendUnavailableError(
            "Could not verify the projected publication USD dependency closure "
            f"because a required runtime dependency is unavailable: {exc}"
        ) from exc
    except JointRiggerContractError as exc:
        raise JointRiggerArtifactError(
            f"Could not verify the projected publication USD dependency closure: {exc}"
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointRiggerArtifactError(
            f"Could not validate the projected publication USD dependency closure: {exc}"
        ) from exc


def _validate_opaque_dependency_closure(
    dependencies: Iterable[Path],
    *,
    allowed_root: Path | None = None,
    allowed_files: set[Path] | None = None,
) -> None:
    """Fail unless every local MDL/MaterialX dependency is bounded and present."""

    normalized_root = (
        _lexical_absolute_path(allowed_root) if allowed_root is not None else None
    )
    normalized_files = (
        {_lexical_absolute_path(path) for path in allowed_files}
        if allowed_files is not None
        else None
    )
    pending = sorted(
        {
            _lexical_absolute_path(path)
            for path in dependencies
            if path.suffix.lower() in _OPAQUE_DEPENDENCY_EXTENSIONS
        },
        key=lambda path: path.as_posix(),
    )
    visited: set[Path] = set()
    reference_count = 0
    while pending:
        document = pending.pop(0)
        if document in visited:  # pragma: no cover - queue insertion deduplicates
            continue
        if len(visited) >= _MAX_OPAQUE_DEPENDENCY_FILES:
            raise JointRiggerArtifactError(
                "Opaque material dependency closure exceeds the "
                f"{_MAX_OPAQUE_DEPENDENCY_FILES}-file limit"
            )
        _require_opaque_dependency_file(
            document,
            allowed_root=normalized_root,
            allowed_files=normalized_files,
        )
        visited.add(document)
        text = _read_bounded_opaque_document(document)
        if document.suffix.lower() == ".mdl":
            references = _mdl_local_references(text, document=document)
        else:
            references = _materialx_local_references(text, document=document)
        reference_count += len(references)
        if reference_count > _MAX_OPAQUE_DEPENDENCY_REFERENCES:
            raise JointRiggerArtifactError(
                "Opaque material dependency closure exceeds the "
                f"{_MAX_OPAQUE_DEPENDENCY_REFERENCES}-reference limit"
            )
        for value in references:
            target = _resolve_opaque_dependency_reference(
                document,
                value,
                allowed_root=normalized_root,
                allowed_files=normalized_files,
            )
            if (
                target.suffix.lower() in _OPAQUE_DEPENDENCY_EXTENSIONS
                and target not in visited
                and target not in pending
            ):
                pending.append(target)
        pending.sort(key=lambda path: path.as_posix())


def _read_bounded_opaque_document(path: Path) -> str:
    """Read one stable opaque material document with no-follow bounded I/O."""

    flags = os.O_RDONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    operation_error: BaseException | None = None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise JointRiggerArtifactError(
                f"Opaque material dependency is not a regular file: {path}"
            )
        if before.st_size > _MAX_OPAQUE_DOCUMENT_BYTES:
            raise JointRiggerArtifactError(
                "Opaque material dependency exceeds the "
                f"{_MAX_OPAQUE_DOCUMENT_BYTES}-byte limit: {path}"
            )
        payload = bytearray()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise JointRiggerArtifactError(
                    f"Opaque material dependency changed while read: {path}"
                )
            payload.extend(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, offset):
            raise JointRiggerArtifactError(
                f"Opaque material dependency grew while read: {path}"
            )
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if _regular_descriptor_state(after) != _regular_descriptor_state(
            before
        ) or _regular_descriptor_state(current) != _regular_descriptor_state(before):
            raise JointRiggerArtifactError(
                f"Opaque material dependency changed while read: {path}"
            )
    except BaseException as error:
        operation_error = error
        raise
    finally:
        _run_cleanup_steps(
            [
                (
                    f"Opaque material descriptor cleanup failed for {path}",
                    partial(os.close, descriptor),
                )
            ],
            primary_error=operation_error,
        )
    try:
        return bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JointRiggerArtifactError(
            f"Opaque material dependency is not UTF-8 text: {path}"
        ) from exc


def _strip_mdl_comments(text: str, *, document: Path) -> str:
    """Remove C-style comments without treating comment markers in strings as code."""

    try:
        return strip_mdl_comments(text, document=document)
    except OpaqueDependencyError as exc:
        raise JointRiggerArtifactError(str(exc)) from exc


def _mdl_local_references(text: str, *, document: Path) -> tuple[str, ...]:
    """Extract provable local imports and resource paths from one MDL module."""

    try:
        return mdl_local_references(text, document=document)
    except OpaqueDependencyError as exc:
        raise JointRiggerArtifactError(str(exc)) from exc


def _materialx_local_references(text: str, *, document: Path) -> tuple[str, ...]:
    """Extract every supported path-bearing attribute from one MaterialX XML file."""

    try:
        return materialx_local_references(text, document=document)
    except OpaqueDependencyError as exc:
        raise JointRiggerArtifactError(str(exc)) from exc


def _require_opaque_dependency_file(
    path: Path,
    *,
    allowed_root: Path | None,
    allowed_files: set[Path] | None,
) -> None:
    if allowed_root is not None and not _is_relative_to(path, allowed_root):
        raise JointRiggerArtifactError(
            f"Opaque material dependency escapes its sidecar: {path}"
        )
    if allowed_files is not None and path not in allowed_files:
        raise JointRiggerArtifactError(
            "Opaque material dependency is not represented in the generated "
            f"artifact identity: {path}"
        )
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise JointRiggerArtifactError(
            f"Opaque material dependency is missing: {path}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise JointRiggerArtifactError(
            f"Opaque material dependency is not a regular file: {path}"
        )


def _resolve_opaque_dependency_reference(
    document: Path,
    value: str,
    *,
    allowed_root: Path | None,
    allowed_files: set[Path] | None,
) -> Path:
    try:
        target = resolve_local_reference(
            document,
            value,
            allowed_root=allowed_root,
            allowed_files=allowed_files,
        )
    except OpaqueDependencyError as exc:
        raise JointRiggerArtifactError(str(exc)) from exc
    _require_opaque_dependency_file(
        target,
        allowed_root=allowed_root,
        allowed_files=allowed_files,
    )
    return target


def _reject_transaction_target_dependencies(
    dependencies: tuple[Path, ...] | set[Path],
    *,
    staged_targets: JointRiggerArtifactTargets,
    final_targets: JointRiggerArtifactTargets,
) -> None:
    """Reject output dependencies replaced or removed by the transaction."""

    transaction_targets = (
        ("staged generated root", staged_targets.output_path),
        ("staged diagnostics report", staged_targets.diagnostics_path),
        ("staged result report", staged_targets.result_path),
        ("staged composition sidecar", staged_targets.sidecar_path),
        ("final generated root", final_targets.output_path),
        ("final diagnostics report", final_targets.diagnostics_path),
        ("final result report", final_targets.result_path),
        ("final composition sidecar", final_targets.sidecar_path),
    )
    for dependency in dependencies:
        normalized_dependency = dependency.expanduser().resolve(strict=True)
        for label, target in transaction_targets:
            if target is None:
                continue
            normalized_target = target.expanduser().resolve(strict=False)
            if normalized_dependency == normalized_target or _is_relative_to(
                normalized_dependency,
                normalized_target,
            ):
                raise JointRiggerArtifactError(
                    "The generated root dependency closure overlaps a transaction "
                    f"target ({label}): {dependency}"
                )


def _reject_symlink_usd_dependencies(path: Path) -> None:
    """Reject lexical symlinks in every local generated-root dependency path."""

    try:
        from pxr import UsdUtils
    except ImportError as exc:
        raise JointRiggerBackendUnavailableError(
            "Could not inspect generated root dependency locators because the USD "
            f"runtime is unavailable: {exc}"
        ) from exc
    try:
        layers, assets, _ = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:
        raise JointRiggerArtifactError(
            f"Could not inspect generated root dependency locators: {exc}"
        ) from exc

    identifiers: list[str] = []
    for layer in layers:
        identifiers.extend(
            _nonempty_usd_locator_fields(
                layer,
                ("identifier", "resolvedPath", "realPath"),
            )
        )
    for asset in assets:
        asset_identifiers = _nonempty_usd_locator_fields(
            asset,
            ("path", "resolvedPath", "identifier"),
        )
        identifiers.extend(asset_identifiers or (str(asset),))
    lexical_root = path.expanduser().absolute()
    for identifier in identifiers:
        outer = identifier.partition("[")[0]
        # USD locators are authored content, not shell/user input.  Preserve a
        # leading ``~`` as a literal path component relative to the layer.
        lexical_path = (path.parent / Path(outer)).absolute()
        if lexical_path == lexical_root:
            continue
        _normalize_local_path_without_symlinks(
            lexical_path,
            symlink_error="Generated root dependency path must not contain symlinks",
        )


def _reject_uri_usd_dependencies(path: Path) -> None:
    """Reject remotely resolved dependencies from a sidecar publication set."""

    try:
        from pxr import UsdUtils
    except ImportError as exc:
        raise JointRiggerBackendUnavailableError(
            "Could not inspect the staged publication sidecar because the USD "
            f"runtime is unavailable: {exc}"
        ) from exc
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:
        raise JointRiggerArtifactError(
            f"Could not enumerate staged publication dependencies: {exc}"
        ) from exc
    if unresolved:
        raise JointRiggerArtifactError(
            "The staged publication has unresolved dependencies: "
            f"{sorted(str(item) for item in unresolved)}"
        )
    identifiers: list[str] = []
    for layer in layers:
        identifiers.extend(
            _nonempty_usd_locator_fields(
                layer,
                ("identifier", "resolvedPath", "realPath"),
            )
        )
    for asset in assets:
        asset_identifiers = _nonempty_usd_locator_fields(
            asset,
            ("path", "resolvedPath", "identifier"),
        )
        identifiers.extend(asset_identifiers or (str(asset),))
    remote = sorted(
        {
            identifier
            for identifier in identifiers
            if _is_remote_dependency_identifier(identifier)
        }
    )
    if remote:
        raise JointRiggerArtifactError(
            f"The staged publication sidecar must not depend on external URIs: {remote}"
        )


def _nonempty_usd_locator_fields(
    dependency: Any,
    fields: tuple[str, ...],
) -> tuple[str, ...]:
    """Return every nonempty authored or resolved locator exposed by USD."""

    return tuple(
        identifier
        for field in fields
        if (identifier := str(getattr(dependency, field, "") or "").strip())
    )


def _is_remote_dependency_identifier(identifier: str) -> bool:
    outer = identifier.partition("[")[0]
    parsed = urlparse(outer)
    if not parsed.scheme or parsed.scheme == "file":
        return False
    if "://" in outer:
        return True
    # Only an actual rooted native Windows drive spelling is local. A one-letter
    # opaque resolver scheme such as ``s:asset`` is still a remote identifier.
    return re.match(r"^[A-Za-z]:[\\/]", outer) is None


def _validate_diagnostic_decisions(
    request: JointRiggerInputV1,
    diagnostics: Any,
) -> None:
    """Require an exact, provenance-bound disposition for every plan fact.

    Joint decisions are scoped by ``joint_id`` and use paths relative to that
    joint (for example, ``topology.body0`` and ``drive.stiffness``).  Body and
    articulation decisions live at the diagnostics top level because the v1
    diagnostics model has no body-specific container.  Their stable paths are
    ``rigid_bodies[<body path>].*`` and either ``articulation_root`` or
    ``articulation_roots[<root path>]``; collider paths are nested as
    ``colliders[<collider path>].*``.

    A succeeded result may describe absent optional inputs as ignored or as a
    documented backend default, but every fact actually present in the plan
    must be accepted with the exact provenance carried by that fact.  This
    prevents a backend from publishing success after silently dropping a
    structured field.
    """

    _require_unique_decision_fields(
        diagnostics.field_decisions,
        label="top-level",
    )
    _require_unique_joint_diagnostics(diagnostics.joint_diagnostics)
    _validate_legacy_component_decisions(request, diagnostics)

    expected_top_level, allowed_top_level = _planned_top_level_decisions(request)
    legacy_fields = _legacy_decision_fields(request)
    allowed_top_level.update(legacy_fields)
    if not request.plan.rigid_bodies:
        allowed_top_level["rigid_bodies"] = frozenset({"ignored"})
    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerPlanV1,
    )

    if isinstance(request.plan, JointRiggerPlanV1):
        if request.plan.articulation_root is None:
            allowed_top_level["articulation_root"] = frozenset({"ignored"})
    elif not request.plan.articulation_roots:
        allowed_top_level["articulation_roots"] = frozenset({"ignored"})
    _validate_decision_set(
        diagnostics.field_decisions,
        expected=expected_top_level,
        allowed=allowed_top_level,
        label="top-level",
    )

    expected_joints = {joint.topology.joint_id: joint for joint in request.plan.joints}
    actual_joints = {item.joint_id: item for item in diagnostics.joint_diagnostics}
    missing_joints = set(expected_joints) - set(actual_joints)
    if missing_joints:
        raise JointRiggerArtifactError(
            "Joint Rigger diagnostics are missing planned joint diagnostic(s): "
            f"{', '.join(sorted(missing_joints))}"
        )
    unexpected_joints = set(actual_joints) - set(expected_joints)
    if unexpected_joints:
        raise JointRiggerArtifactError(
            "Joint Rigger diagnostics contain unexpected joint diagnostic(s): "
            f"{', '.join(sorted(unexpected_joints))}"
        )

    for joint_id, joint in expected_joints.items():
        diagnostic = actual_joints[joint_id]
        _require_unique_decision_fields(
            diagnostic.field_decisions,
            label=f"joint {joint_id}",
        )
        expected, allowed, alias_groups, default_reason_codes = (
            _planned_joint_decisions(joint)
        )
        _validate_decision_set(
            diagnostic.field_decisions,
            expected=expected,
            allowed=allowed,
            label=f"joint {joint_id}",
            default_reason_codes=default_reason_codes,
        )
        actual_fields = {decision.field for decision in diagnostic.field_decisions}
        for aliases in alias_groups:
            present = actual_fields.intersection(aliases)
            if len(present) > 1:
                raise JointRiggerArtifactError(
                    f"Joint Rigger diagnostics joint {joint_id} contains multiple "
                    "decisions for one absent optional fact: "
                    f"{', '.join(sorted(present))}"
                )


def _planned_top_level_decisions(
    request: JointRiggerInputV1,
) -> tuple[dict[str, Any], dict[str, frozenset[str]]]:
    expected: dict[str, Any] = {}
    allowed: dict[str, frozenset[str]] = {}
    for body in request.plan.rigid_bodies:
        prefix = f"rigid_bodies[{body.prim_path}]"
        expected[f"{prefix}.rigid_body"] = body.provenance
        if body.mass is None:
            allowed[f"{prefix}.mass"] = frozenset({"ignored"})
        else:
            expected[f"{prefix}.mass.mass_kg"] = body.mass.provenance
            if body.mass.center_of_mass_m is None:
                allowed[f"{prefix}.mass.center_of_mass_m"] = frozenset({"ignored"})
            else:
                expected[f"{prefix}.mass.center_of_mass_m"] = body.mass.provenance
            expected[f"{prefix}.mass.diagonal_inertia_kg_m2"] = body.mass.provenance
            if body.mass.principal_axes is None:
                allowed[f"{prefix}.mass.principal_axes"] = frozenset({"ignored"})
            else:
                expected[f"{prefix}.mass.principal_axes"] = body.mass.provenance
        for collider in body.colliders:
            collider_prefix = f"{prefix}.colliders[{collider.prim_path}]"
            expected[f"{collider_prefix}.collision"] = collider.provenance
            if collider.has_mesh_collision_api:
                expected[f"{collider_prefix}.mesh_collision_api"] = collider.provenance
            else:
                allowed[f"{collider_prefix}.mesh_collision_api"] = frozenset(
                    {"ignored"}
                )
            if collider.mesh_approximation is None:
                allowed[f"{collider_prefix}.mesh_approximation"] = frozenset(
                    {"ignored"}
                )
            else:
                expected[f"{collider_prefix}.mesh_approximation"] = collider.provenance
    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerPlanV1,
    )

    if isinstance(request.plan, JointRiggerPlanV1):
        articulation_root = request.plan.articulation_root
        if articulation_root is not None:
            expected["articulation_root"] = articulation_root.provenance
    else:
        for articulation_root in request.plan.articulation_roots:
            expected[f"articulation_roots[{articulation_root.prim_path}]"] = (
                articulation_root.provenance
            )
    return expected, allowed


def _planned_joint_decisions(
    joint: Any,
) -> tuple[
    dict[str, Any],
    dict[str, frozenset[str]],
    tuple[frozenset[str], ...],
    dict[str, str],
]:
    expected: dict[str, Any] = {}
    allowed: dict[str, frozenset[str]] = {
        "usd.joint_prim_path": frozenset({"defaulted"}),
        "usd.local_frames": frozenset({"defaulted"}),
    }
    default_reason_codes = {
        "usd.joint_prim_path": "deterministic_joint_path",
        "usd.local_frames": "derived_from_stage_axis_and_anchor",
    }
    aliases: list[frozenset[str]] = []
    topology = joint.topology
    for field, provenance in topology.field_provenance.items():
        expected[f"topology.{field}"] = provenance
    if topology.axis_stage is None:
        allowed["topology.axis_stage"] = frozenset({"ignored"})

    limit = joint.limit
    if limit is None:
        limit_aliases = frozenset({"limit", "limit.lower", "limit.upper", "limit.unit"})
        for field in limit_aliases:
            allowed[field] = frozenset({"ignored"})
        # A backend may use either one aggregate decision or individual leaf
        # decisions, but never both representations for the same absent fact.
        aliases.append(frozenset({"limit", "limit.lower"}))
        aliases.append(frozenset({"limit", "limit.upper"}))
        aliases.append(frozenset({"limit", "limit.unit"}))
    else:
        expected["limit.unit"] = limit.provenance
        for field in ("lower", "upper"):
            if getattr(limit, field) is None:
                allowed[f"limit.{field}"] = frozenset({"ignored"})
            else:
                expected[f"limit.{field}"] = limit.provenance

    anchor = joint.anchor
    if anchor is None:
        allowed["anchor"] = frozenset({"ignored", "defaulted"})
        allowed["anchor.position_stage"] = frozenset({"ignored", "defaulted"})
        default_reason_codes["anchor"] = "inferred_body1_world_origin"
        default_reason_codes["anchor.position_stage"] = "inferred_body1_world_origin"
        aliases.append(frozenset({"anchor", "anchor.position_stage"}))
    else:
        expected["anchor.position_stage"] = anchor.provenance

    _add_optional_object_decisions(
        expected,
        allowed,
        aliases,
        prefix="joint_friction",
        value=joint.joint_friction,
        fields=("coefficient",),
    )
    _add_optional_object_decisions(
        expected,
        allowed,
        aliases,
        prefix="drive",
        value=joint.drive,
        fields=(
            "drive_type",
            "stiffness",
            "damping",
            "max_force",
            "target_position",
            "target_velocity",
            "max_joint_velocity",
        ),
        optional_fields=frozenset({"max_joint_velocity"}),
    )
    _add_optional_object_decisions(
        expected,
        allowed,
        aliases,
        prefix="state",
        value=joint.state,
        fields=("position", "velocity"),
    )
    _add_optional_object_decisions(
        expected,
        allowed,
        aliases,
        prefix="mimic",
        value=joint.mimic,
        fields=(
            "reference_joint_id",
            "gearing",
            "offset",
            "natural_frequency",
            "damping_ratio",
        ),
    )
    return expected, allowed, tuple(aliases), default_reason_codes


def _add_optional_object_decisions(
    expected: dict[str, Any],
    allowed: dict[str, frozenset[str]],
    aliases: list[frozenset[str]],
    *,
    prefix: str,
    value: Any,
    fields: tuple[str, ...],
    optional_fields: frozenset[str] = frozenset(),
) -> None:
    if value is None:
        allowed[prefix] = frozenset({"ignored"})
        for field in fields:
            allowed[f"{prefix}.{field}"] = frozenset({"ignored"})
            aliases.append(frozenset({prefix, f"{prefix}.{field}"}))
        return
    for field in fields:
        field_path = f"{prefix}.{field}"
        if field in optional_fields and getattr(value, field) is None:
            allowed[field_path] = frozenset({"ignored"})
        else:
            expected[field_path] = value.provenance


def _validate_decision_set(
    decisions: Any,
    *,
    expected: Mapping[str, Any],
    allowed: Mapping[str, frozenset[str]],
    label: str,
    default_reason_codes: Mapping[str, str] | None = None,
) -> None:
    by_field = {decision.field: decision for decision in decisions}
    missing = set(expected) - set(by_field)
    if missing:
        raise JointRiggerArtifactError(
            f"Joint Rigger diagnostics {label} are missing planned field "
            f"decision(s): {', '.join(sorted(missing))}"
        )
    unexpected = set(by_field) - set(expected) - set(allowed)
    if unexpected:
        raise JointRiggerArtifactError(
            f"Joint Rigger diagnostics {label} contain unexpected field "
            f"decision(s): {', '.join(sorted(unexpected))}"
        )
    for field, provenance in expected.items():
        decision = by_field[field]
        if decision.disposition != "accepted":
            raise JointRiggerArtifactError(
                f"Joint Rigger diagnostics {label} field {field} must be accepted; "
                f"got {decision.disposition}"
            )
        if decision.provenance != provenance:
            raise JointRiggerArtifactError(
                f"Joint Rigger diagnostics {label} field {field} provenance does "
                "not match the planned fact"
            )
    for field, dispositions in allowed.items():
        decision = by_field.get(field)
        if decision is None:
            continue
        if decision.disposition not in dispositions:
            raise JointRiggerArtifactError(
                f"Joint Rigger diagnostics {label} field {field} must be "
                f"{', '.join(sorted(dispositions))}; got {decision.disposition}"
            )
        if decision.disposition == "defaulted":
            expected_reason = (default_reason_codes or {}).get(field)
            if expected_reason is not None and decision.reason_code != expected_reason:
                raise JointRiggerArtifactError(
                    f"Joint Rigger diagnostics {label} field {field} must use "
                    f"reason_code={expected_reason} when defaulted; got "
                    f"{decision.reason_code}"
                )
        if decision.disposition != "accepted" and decision.provenance is not None:
            raise JointRiggerArtifactError(
                f"Joint Rigger diagnostics {label} field {field} must not claim "
                "provenance for an absent or backend-derived fact"
            )


def _require_unique_decision_fields(decisions: Any, *, label: str) -> None:
    fields = [decision.field for decision in decisions]
    duplicates = sorted(field for field, count in Counter(fields).items() if count > 1)
    if duplicates:
        raise JointRiggerArtifactError(
            f"Joint Rigger diagnostics {label} must contain exactly one decision "
            "per field; duplicate field(s): "
            f"{', '.join(duplicates)}"
        )


def _require_unique_joint_diagnostics(diagnostics: Any) -> None:
    identifiers = [item.joint_id for item in diagnostics]
    duplicates = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    if duplicates:
        raise JointRiggerArtifactError(
            "Joint Rigger diagnostics joint identifiers must be unique: "
            f"{', '.join(duplicates)}"
        )


def _legacy_decision_fields(request: JointRiggerInputV1) -> dict[str, frozenset[str]]:
    compatibility = request.legacy_component_names
    if compatibility is None:
        return {"legacy_component_names": frozenset({"ignored", "rejected"})}
    return {
        f"legacy_component_names[{assignment.prim_path}]": (
            frozenset({"defaulted"})
            if assignment.source_field == "role"
            else frozenset({"accepted"})
        )
        for assignment in compatibility.assignments
    }


def _validate_legacy_component_decisions(
    request: JointRiggerInputV1,
    diagnostics: Any,
) -> None:
    compatibility = request.legacy_component_names
    top_level_legacy_decisions = [
        decision
        for decision in diagnostics.field_decisions
        if decision.field == "legacy_component_names"
    ]
    if compatibility is None:
        if len(top_level_legacy_decisions) != 1:
            raise JointRiggerArtifactError(
                "Joint Rigger diagnostics must contain exactly one top-level "
                "legacy_component_names no-fallback decision"
            )
        no_fallback = top_level_legacy_decisions[0]
        if no_fallback.disposition not in {"ignored", "rejected"}:
            raise JointRiggerArtifactError(
                "Joint Rigger diagnostics top-level legacy_component_names decision "
                "must be ignored or rejected to prove no fallback was used"
            )
    decisions = {decision.field: decision for decision in diagnostics.field_decisions}
    assignment_decisions = {
        field: decision
        for field, decision in decisions.items()
        if field.startswith("legacy_component_names[")
    }
    assignments = () if compatibility is None else compatibility.assignments
    expected_fields = {
        f"legacy_component_names[{assignment.prim_path}]" for assignment in assignments
    }
    actual_fields = set(assignment_decisions)
    missing_fields = expected_fields - actual_fields
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise JointRiggerArtifactError(
            f"Joint Rigger diagnostics are missing field decision(s): {missing}"
        )
    unexpected_fields = actual_fields - expected_fields
    if unexpected_fields:
        unexpected = ", ".join(sorted(unexpected_fields))
        raise JointRiggerArtifactError(
            "Joint Rigger diagnostics contain unexpected legacy component field "
            f"decision(s): {unexpected}"
        )

    for assignment in assignments:
        field = f"legacy_component_names[{assignment.prim_path}]"
        decision = assignment_decisions[field]
        if assignment.source_field == "role":
            if (
                decision.disposition != "defaulted"
                or decision.reason_code != "legacy_component_name_compatibility"
            ):
                raise JointRiggerArtifactError(
                    f"Joint Rigger diagnostics field decision {field} must be "
                    "defaulted with reason_code=legacy_component_name_compatibility"
                )
        elif decision.disposition != "accepted" or decision.provenance is None:
            raise JointRiggerArtifactError(
                f"Joint Rigger diagnostics field decision {field} must be accepted "
                "with provenance"
            )


def _load_model_report(
    path: Path,
    model_type: Any,
    label: str,
) -> _ParsedModelReport:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise JointRiggerArtifactError(
            f"Backend did not write a regular {label} report: {path}"
        ) from exc
    operation_error: BaseException | None = None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise JointRiggerArtifactError(
                f"Backend did not write a regular {label} report: {path}"
            )
        if metadata.st_size > _MAX_REPORT_BYTES:
            raise JointRiggerArtifactError(
                f"Backend {label} report exceeds the {_MAX_REPORT_BYTES}-byte "
                f"limit: {path}"
            )
        payload = bytearray()
        while len(payload) <= _MAX_REPORT_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_REPORT_BYTES + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > _MAX_REPORT_BYTES:
            raise JointRiggerArtifactError(
                f"Backend {label} report exceeds the {_MAX_REPORT_BYTES}-byte "
                f"limit: {path}"
            )
    except JointRiggerArtifactError as error:
        operation_error = error
        raise
    except OSError as exc:
        wrapped_error = JointRiggerArtifactError(
            f"Could not read the backend {label} report at {path}: {exc}"
        )
        operation_error = wrapped_error
        raise wrapped_error from exc
    except BaseException as error:
        operation_error = error
        raise
    finally:
        _run_cleanup_steps(
            [
                (
                    f"Backend {label} report descriptor cleanup failed",
                    partial(os.close, descriptor),
                )
            ],
            primary_error=operation_error,
        )
    try:
        bounded_payload = bytes(payload)
        return _ParsedModelReport(
            model=_validate_model_report_payload(bounded_payload, model_type),
            payload=bounded_payload,
        )
    except (
        RecursionError,
        UnicodeError,
        ValidationError,
        json.JSONDecodeError,
        _DuplicateJsonObjectKeyError,
    ) as exc:
        raise JointRiggerArtifactError(
            f"Backend wrote an invalid {label} report at {path}: {exc}"
        ) from exc


def _load_sealed_model_report(
    snapshot: _SealedReportSnapshot,
    model_type: Any,
    label: str,
) -> _ParsedModelReport:
    """Parse bounded bytes from the exact held snapshot inode."""

    try:
        parent_metadata = os.fstat(snapshot.parent_descriptor)
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or (parent_metadata.st_dev, parent_metadata.st_ino)
            != snapshot.parent_identity
        ):
            raise JointRiggerArtifactError(
                f"Private {label} parent descriptor changed inode"
            )
        before = os.fstat(snapshot.source_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != snapshot.source_identity
            or before.st_nlink != 1
            or before.st_mode & 0o222
        ):
            raise JointRiggerArtifactError(
                f"Private {label} descriptor is no longer a sealed regular file"
            )
        if before.st_size > _MAX_REPORT_BYTES:
            raise JointRiggerArtifactError(
                f"Private {label} exceeds the {_MAX_REPORT_BYTES}-byte limit"
            )
        payload = bytearray()
        while len(payload) <= _MAX_REPORT_BYTES:
            chunk = os.pread(
                snapshot.source_descriptor,
                min(1024 * 1024, _MAX_REPORT_BYTES + 1 - len(payload)),
                len(payload),
            )
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > _MAX_REPORT_BYTES:
            raise JointRiggerArtifactError(
                f"Private {label} exceeds the {_MAX_REPORT_BYTES}-byte limit"
            )
        after = os.fstat(snapshot.source_descriptor)
        before_state = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_state = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_state != after_state:
            raise JointRiggerArtifactError(
                f"Private {label} changed while it was revalidated"
            )
    except JointRiggerArtifactError:
        raise
    except OSError as exc:
        raise JointRiggerArtifactError(
            f"Could not read the private {label} descriptor: {exc}"
        ) from exc

    bounded_payload = bytes(payload)
    observed_sha256 = hashlib.sha256(bounded_payload).hexdigest()
    if observed_sha256 != snapshot.source_sha256:
        raise JointRiggerArtifactError(
            f"Private {label} SHA-256 changed after validation"
        )
    try:
        return _ParsedModelReport(
            model=_validate_model_report_payload(bounded_payload, model_type),
            payload=bounded_payload,
        )
    except (
        RecursionError,
        UnicodeError,
        ValidationError,
        json.JSONDecodeError,
        _DuplicateJsonObjectKeyError,
    ) as exc:
        raise JointRiggerArtifactError(
            f"Private {label} contains invalid report JSON: {exc}"
        ) from exc


def _canonical_model_payload(value: Any) -> str:
    if not isinstance(value, BaseModel):
        raise JointRiggerBackendIncompatibleError(
            f"Expected a Pydantic contract model, got {type(value).__name__}"
        )
    # Keep semantic model comparisons on the same canonical encoding used by
    # contract hashes.  Import lazily so the public facade remains importable
    # without eagerly loading the contract-model module.
    from world_understanding.functions.physics.joint_rigger.models import (
        canonical_json,
    )

    return canonical_json(value)


def _copy_stable_regular_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> str:
    """Copy one path-bound regular file without following or blocking on races."""

    source_path = Path(source)
    destination_path = Path(destination)
    expected = os.stat(source_path, follow_symlinks=False)
    if not stat.S_ISREG(expected.st_mode):
        raise ValueError(f"Copy source must be a regular file: {source_path}")

    source_descriptor = -1
    destination_parent_descriptor = -1
    destination_identity: tuple[int, int] | None = None
    primary_error: BaseException | None = None
    try:
        source_flags = os.O_RDONLY | os.O_NOFOLLOW
        source_flags |= getattr(os, "O_CLOEXEC", 0)
        source_flags |= getattr(os, "O_NONBLOCK", 0)
        source_descriptor = os.open(source_path, source_flags)
        opened = os.fstat(source_descriptor)
        expected_state = _regular_descriptor_state(expected)
        opened_state = _regular_descriptor_state(opened)
        if not stat.S_ISREG(opened.st_mode) or opened_state != expected_state:
            raise RuntimeError(
                f"Copy source changed before it was opened: {source_path}"
            )
        source_sha256 = _stable_descriptor_sha256(
            source_descriptor,
            label=f"copy source {source_path}",
        )
        destination_parent_descriptor = _open_stable_copy_parent(destination_path)
        destination_identity = _copy_stable_regular_descriptor(
            source_descriptor,
            destination_path,
            expected_identity=(opened.st_dev, opened.st_ino),
            expected_sha256=source_sha256,
            expected_mode=stat.S_IMODE(opened.st_mode),
            expected_nlink=opened.st_nlink,
            label=f"copy source {source_path}",
            destination_parent_descriptor=destination_parent_descriptor,
        )
        after = os.fstat(source_descriptor)
        observed_path = os.stat(source_path, follow_symlinks=False)
        after_state = _regular_descriptor_state(after)
        path_state = _regular_descriptor_state(observed_path)
        if after_state != expected_state or path_state != expected_state:
            raise RuntimeError(f"Copy source changed while read: {source_path}")
    except BaseException as exc:
        primary_error = exc
    finally:
        cleanup_steps: list[tuple[str, Callable[[], None]]] = []
        if (
            primary_error is not None
            and destination_parent_descriptor >= 0
            and destination_identity is not None
        ):
            cleanup_steps.append(
                (
                    f"Stable-copy rollback failed for {destination_path}",
                    partial(
                        _remove_descriptor_entry,
                        destination_parent_descriptor,
                        destination_path.name,
                        expected_identity=destination_identity,
                        label=f"stable-copy target {destination_path}",
                    ),
                )
            )
        if source_descriptor >= 0:
            owned_source_descriptor = source_descriptor
            source_descriptor = -1
            cleanup_steps.append(
                (
                    f"Stable-copy source descriptor cleanup failed for {source_path}",
                    partial(os.close, owned_source_descriptor),
                )
            )
        if destination_parent_descriptor >= 0:
            owned_parent_descriptor = destination_parent_descriptor
            destination_parent_descriptor = -1
            cleanup_steps.append(
                (
                    f"Stable-copy parent descriptor cleanup failed for {destination_path}",
                    partial(os.close, owned_parent_descriptor),
                )
            )
        _run_cleanup_steps(
            cleanup_steps,
            primary_error=primary_error,
        )
    if primary_error is not None:
        raise primary_error
    return str(destination_path)


def _file_sha256(path: Path) -> str:
    """Hash one regular file without following or blocking on special files."""

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    digest = ""
    operation_error: BaseException | None = None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Artifact path must identify a regular file: {path}")
        digest = _stable_descriptor_sha256(
            descriptor,
            label=f"artifact {path}",
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
                        f"Artifact hash descriptor cleanup failed for {path}",
                        partial(os.close, owned_descriptor),
                    )
                ],
                primary_error=operation_error,
            )
    return digest


def _request_local_read_paths(
    request: JointRiggerInputV1,
) -> list[tuple[str, Path]]:
    """Return every unique local artifact input that publication must preserve."""
    _, read_paths = _preflight_request_inputs(request)
    return read_paths


def _preflight_request_inputs(
    request: JointRiggerInputV1,
) -> tuple[tuple[_LocalInputSnapshot, ...], list[tuple[str, Path]]]:
    """Bind all local request inputs and enumerate their complete read closure."""

    candidates = _request_artifact_identities(request)
    snapshots: list[_LocalInputSnapshot] = []
    read_paths: list[tuple[str, Path]] = []
    seen: dict[Path, _LocalInputSnapshot] = {}
    seen_reads: set[Path] = set()
    for label, artifact in candidates:
        local_path = _local_artifact_path(artifact.uri)
        if local_path is None:
            continue
        normalized = _normalize_local_input_path(local_path, label=label)
        previous = seen.get(normalized)
        if previous is not None:
            if previous.artifact != artifact:
                raise JointRiggerArtifactError(
                    "The request assigns conflicting identities to one local "
                    f"artifact: {normalized}"
                )
            continue
        snapshot = _inspect_local_input(label, artifact, normalized)
        seen[normalized] = snapshot
        snapshots.append(snapshot)
        for dependency_index, read_path in enumerate(
            (snapshot.path, *snapshot.dependency_paths)
        ):
            # Preserve authored dependency aliases as distinct destructive-read
            # locations.  Resolving here would erase a locator inside a sidecar
            # that publication later replaces recursively.
            normalized_read = _lexical_absolute_path(read_path)
            if normalized_read in seen_reads:
                continue
            seen_reads.add(normalized_read)
            read_label = label if dependency_index == 0 else f"{label} dependency"
            read_paths.append((read_label, normalized_read))
    return tuple(snapshots), read_paths


def _request_artifact_identities(
    request: JointRiggerInputV1,
) -> list[tuple[str, ArtifactIdentityV1]]:
    """Return source and provenance identities in deterministic request order."""

    from world_understanding.functions.physics.joint_rigger.models import (
        ArtifactIdentityV1,
    )

    candidates: list[tuple[str, ArtifactIdentityV1]] = []
    source_asset = getattr(request, "source_asset", None)
    if isinstance(source_asset, ArtifactIdentityV1):
        candidates.append(("source_asset", source_asset))
    plan = getattr(request, "plan", None)
    candidates.extend(
        ("plan provenance artifact", artifact)
        for artifact in _nested_artifact_identities(plan)
    )

    return candidates


def _inspect_local_input(
    label: str,
    artifact: ArtifactIdentityV1,
    path: Path,
) -> _LocalInputSnapshot:
    """Verify one local request identity and capture its dependency closure."""

    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerContractError,
    )
    from world_understanding.functions.physics.joint_rigger.reference import (
        identify_usd_artifact,
    )

    path = _normalize_local_input_path(path, label=label)
    try:
        root_sha256 = _file_sha256(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointRiggerArtifactError(
            f"Could not read {label} at {path}: {exc}"
        ) from exc
    if root_sha256 != artifact.root_sha256:
        raise JointRiggerArtifactError(
            f"{label} root_sha256 does not match the local artifact at {path}"
        )

    if not _is_usd_path(path):
        return _LocalInputSnapshot(label, artifact, path, (), None)

    if artifact.dependency_bundle_sha256 is None:
        raise JointRiggerArtifactError(
            f"{label} local USD identity must provide dependency_bundle_sha256: {path}"
        )
    dependency_paths = _local_usd_dependency_paths(path, label=label)
    try:
        actual_identity = identify_usd_artifact(path, uri=artifact.uri)
    except ImportError as exc:
        raise JointRiggerBackendUnavailableError(
            f"Could not verify {label} because a required USD runtime dependency "
            f"is unavailable: {exc}"
        ) from exc
    except JointRiggerContractError as exc:
        raise JointRiggerArtifactError(f"Could not verify {label}: {exc}") from exc
    if artifact.dependency_bundle_sha256 != actual_identity.dependency_bundle_sha256:
        raise JointRiggerArtifactError(
            f"{label} dependency_bundle_sha256 does not match the local USD "
            f"dependency closure at {path}"
        )
    return _LocalInputSnapshot(
        label,
        artifact,
        path,
        dependency_paths,
        actual_identity.dependency_bundle_sha256,
    )


def _normalize_local_input_path(path: Path, *, label: str) -> Path:
    """Return an absolute lexical path after rejecting every symlink component."""

    return _normalize_local_path_without_symlinks(
        path,
        symlink_error=f"{label} local input path must not contain symlinks",
    )


def _normalize_local_path_without_symlinks(
    path: Path,
    *,
    symlink_error: str,
) -> Path:
    """Normalize one lexical path while rejecting symlink leaves and ancestors."""

    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    components = absolute.parts[1:]
    if current.is_symlink():  # pragma: no cover - filesystem-root invariant
        raise JointRiggerArtifactError(f"{symlink_error}: {current}")
    for component in components:
        if component == "..":
            current = current.parent
            continue
        current /= component
        if current.is_symlink():
            raise JointRiggerArtifactError(f"{symlink_error}: {current}")
    return current


def _verify_request_inputs_unchanged(
    expected_snapshots: tuple[_LocalInputSnapshot, ...],
) -> None:
    """Fail before publication if a backend changed any bound request input."""

    for expected in expected_snapshots:
        current = _inspect_local_input(
            expected.label,
            expected.artifact,
            expected.path,
        )
        if current.dependency_paths != expected.dependency_paths:
            raise JointRiggerArtifactError(
                f"{expected.label} USD dependency paths changed during authoring"
            )
        if (
            current.actual_dependency_bundle_sha256
            != expected.actual_dependency_bundle_sha256
        ):
            raise JointRiggerArtifactError(
                f"{expected.label} USD dependency closure changed during authoring"
            )


def _local_usd_dependency_paths(path: Path, *, label: str) -> tuple[Path, ...]:
    """Enumerate one USD closure with facade-specific typed error mapping."""

    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerContractError,
    )
    from world_understanding.functions.physics.joint_rigger.reference import (
        local_usd_dependency_paths,
    )

    try:
        discovered = local_usd_dependency_paths(
            path,
            include_lexical_aliases=True,
        )
        normalized = {
            _lexical_absolute_path(Path(candidate)) for candidate in discovered
        }
    except ImportError as exc:
        raise JointRiggerBackendUnavailableError(
            f"Could not inspect {label} because a required USD runtime dependency "
            f"is unavailable: {exc}"
        ) from exc
    except JointRiggerContractError as exc:
        raise JointRiggerArtifactError(f"Could not inspect {label}: {exc}") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointRiggerArtifactError(f"Could not inspect {label}: {exc}") from exc
    root = path.expanduser().resolve(strict=False)
    normalized.discard(root)
    return tuple(sorted(normalized, key=lambda candidate: candidate.as_posix()))


def _is_usd_path(path: Path) -> bool:
    return path.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}


def _nested_artifact_identities(value: Any) -> Iterator[ArtifactIdentityV1]:
    """Yield artifact identities nested in immutable plan contract models."""
    from world_understanding.functions.physics.joint_rigger.models import (
        ArtifactIdentityV1,
    )

    if isinstance(value, ArtifactIdentityV1):
        yield value
    elif isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            yield from _nested_artifact_identities(getattr(value, field_name))
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _nested_artifact_identities(nested)
    elif isinstance(value, list | tuple):
        for nested in value:
            yield from _nested_artifact_identities(nested)


def _local_artifact_path(uri: str) -> Path | None:
    """Resolve plain and file-scheme artifact URIs without opening them."""
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.scheme == "file":
        return _canonical_file_uri_path(uri, label="artifact URI")
    local_value = uri
    return Path(local_value) if local_value else None


def _canonical_file_uri_path(uri: str, *, label: str) -> Path:
    """Decode one exact canonical absolute local file URI."""

    parsed = urlparse(uri)
    decoded_path = unquote(parsed.path)
    local_path = Path(decoded_path)
    try:
        canonical_uri = local_path.as_uri()
    except ValueError as exc:
        raise JointRiggerArtifactError(
            f"A file {label} must be an exact canonical absolute file URI; got {uri}"
        ) from exc
    has_dot_segment = any(segment in {".", ".."} for segment in decoded_path.split("/"))
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.params
        or parsed.query
        or parsed.fragment
        or has_dot_segment
        or uri != canonical_uri
    ):
        raise JointRiggerArtifactError(
            f"A file {label} must be an exact canonical absolute file URI; got {uri}"
        )
    return local_path


def _lexical_absolute_path(path: Path) -> Path:
    """Collapse dot segments without following symlinks."""

    return Path(os.path.abspath(path.expanduser()))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _backend_label(backend: object) -> str:
    for attribute in ("name", "backend_name"):
        value = getattr(backend, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return type(backend).__name__


__all__ = [
    "JointRiggerArtifactError",
    "JointRiggerBackend",
    "JointRiggerBackendIncompatibleError",
    "JointRiggerBackendUnavailableError",
    "JointRiggerFacadeError",
    "JointRiggerPostCommitCleanupError",
    "author_joint_rig",
]
