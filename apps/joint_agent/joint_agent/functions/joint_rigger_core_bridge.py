# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transitional Stage 2 adapter for the shared Joint Rigger facade.

The existing candidate-edge authorer remains the source of USD authoring
semantics.  This module translates its already-preflighted Stage 2 document
into the shared v1 request, runs the v0 authorer behind ``author_joint_rig``,
and publishes only shared v1 reports.  The v0 diagnostics and validation
reports are private implementation details created in a temporary directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import tempfile
import threading
import weakref
from _thread import LockType
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import replace as dataclass_replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, cast
from weakref import ReferenceType

from world_understanding.functions.physics.joint_rigger import (
    DIAGNOSTICS_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    ArtifactIdentityV1,
    FieldDecisionV1,
    FieldProvenanceV1,
    JointDiagnosticV1,
    JointLimitV1,
    JointPlanV1,
    JointRiggerArtifactError,
    JointRiggerArtifactTargets,
    JointRiggerBackendIncompatibleError,
    JointRiggerContractError,
    JointRiggerDiagnosticsV1,
    JointRiggerInputV1,
    JointRiggerInputV2,
    JointRiggerPlanV1,
    JointRiggerResultV1,
    JointTopologyV1,
    author_joint_rig_from_factory,
    author_joint_topology,
    canonical_json,
    canonical_sha256,
    identify_usd_artifact,
    local_usd_dependency_paths,
    sidecar_dependency_bundle_sha256,
    validate_authored_joint_topology,
)
from world_understanding.functions.physics.joint_rigger.artifacts import (
    StagedArtifact,
    _ArtifactTreeTraversalBudget,
    _BoundDirectory,
    _capture_target_state,
    _CapturedTargetState,
    _remove_descriptor_entry,
    directory_descriptor_tree_sha256,
    promote_staged_artifacts,
    validate_artifact_targets,
)
from world_understanding.functions.physics.joint_rigger.artifacts import (
    _require_directory_tree_mount_id as _shared_require_directory_tree_mount_id,
)
from world_understanding.functions.physics.joint_rigger.author import (
    _validate_v2_articulation_roots,
)
from world_understanding.functions.physics.joint_rigger.rigid_links import (
    validate_authored_rigid_links,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    BoundInputDirectory,
    _create_sealed_file_binding,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    SealedDependencyBinding as _SealedDependencyBinding,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    SealedSourceBinding as _SealedSourceBinding,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    bound_input_dependency_snapshots as _bound_input_dependency_snapshots,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    close_source_binding as _close_source_binding,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    create_sealed_source_binding as _create_sealed_source_binding,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    materialize_bound_input as _materialize_bound_input,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    remove_bound_input_directory as _remove_bound_input_directory,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    require_sealed_source_binding as _require_sealed_source_binding,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    restore_bound_projection_paths as _restore_bound_projection_paths,
)

from joint_agent.functions import candidate_edge_authoring
from joint_agent.functions.candidate_edge_authoring import (
    ADAPTER_NAME,
    AUTHORING_SCHEMA_VERSION,
    author_stage2_candidate_edges,
)

_RAW_USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc"})
_USD_EXTENSIONS = frozenset({*_RAW_USD_EXTENSIONS, ".usdz"})
_MAX_PROBE_BINDINGS = 256
_CONTRACT_TOPOLOGY_BACKEND_VERSION = "joint-agent-contract-topology-v1"
_CANDIDATE_READINESS_SHA256_FIELD = "articulation_candidates_sha256"


class NoReadyJointCandidatesError(ValueError):
    """Raised when a valid Stage 2 document contains no authorable candidates."""


class InitialNoReadyJointCandidatesError(NoReadyJointCandidatesError):
    """Raised only when the facade's initial validated preflight has no work."""


_MAX_BOUND_CANDIDATE_BYTES = 64 * 1024 * 1024


@dataclass
class _ProbeBinding:
    """One weak, exact-object proof produced by an expensive backend probe."""

    request_ref: ReferenceType[JointRiggerInputV1]
    request_sha256: str
    count: int = 1


@dataclass(frozen=True)
class _DirectoryTreeSnapshot:
    """One retained directory root with exact tree and mount identity."""

    descriptor: int
    identity: tuple[int, int]
    mount_id: int
    tree_sha256: str


@dataclass(frozen=True)
class _TargetEntrySnapshot:
    """One no-follow target and parent state captured before private authoring."""

    label: str
    path: Path
    parent_path: Path
    parent_descriptor: int
    parent_identity: tuple[int, int]
    parent_mount_id: int
    entry_state: tuple[int, int, int, int, int, int, int] | None
    initial_target_state: _CapturedTargetState
    directory_tree: _DirectoryTreeSnapshot | None = None

    @property
    def stable_path(self) -> Path:
        """Address the target through the retained original parent inode."""

        return Path(f"/proc/self/fd/{self.parent_descriptor}") / self.path.name


@dataclass(frozen=True)
class _PrivateStagingEntry:
    """One absent random name addressed through a retained parent descriptor."""

    parent_path: Path
    parent_descriptor: int
    parent_identity: tuple[int, int]
    parent_mount_id: int
    name: str

    @property
    def stable_path(self) -> Path:
        """Return the Linux fd-stable locator used for sealing and cleanup."""

        return Path(f"/proc/self/fd/{self.parent_descriptor}") / self.name


@dataclass(frozen=True)
class _PrivateBackendArtifacts:
    """Descriptor-parented physical targets for one backend call."""

    output: _PrivateStagingEntry
    diagnostics: _PrivateStagingEntry
    result: _PrivateStagingEntry
    targets: JointRiggerArtifactTargets


@dataclass(frozen=True)
class _SealedPrivateArtifact:
    """One descriptor-bound private file or directory ready for promotion."""

    label: str
    path: Path
    descriptor: int
    sha256: str
    mount_id: int | None = None


def build_stage2_candidate_edges_input(
    *,
    input_usd_path: str | Path,
    articulation_candidates_path: str | Path,
    expected_articulation_candidates_sha256: str | None = None,
) -> JointRiggerInputV1:
    """Build a shared v1 request from one fully preflighted Stage 2 document.

    Preflight deliberately delegates to the existing authorer so the bridge
    cannot acquire a subtly different readiness, graph, frame, or limit policy.
    Inferred body1-origin anchors remain absent from the source-backed plan and
    are reported later as backend defaults.
    """

    input_path = Path(input_usd_path)
    candidates_path = Path(articulation_candidates_path)
    source_identity = identify_usd_artifact(input_path, uri=str(input_path))
    candidate_binding: _SealedDependencyBinding | None = None
    candidate_uri: str | None = None
    source_binding: _SealedSourceBinding | None = None
    bound_input_dir: BoundInputDirectory | None = None
    primary_error: BaseException | None = None
    try:
        candidate_binding = _create_sealed_candidate_binding(
            candidates_path,
            expected_sha256=expected_articulation_candidates_sha256,
        )
        candidates_sha256 = candidate_binding.sha256
        candidate_uri = str(candidate_binding.path)
        with tempfile.TemporaryDirectory(
            prefix="joint-rigger-stage2-candidate-"
        ) as candidate_tmp:
            candidate_snapshot = Path(candidate_tmp) / "candidates.json"
            _write_sealed_candidate_snapshot(candidate_binding, candidate_snapshot)
            document = candidate_edge_authoring._load_and_validate_document(
                candidate_snapshot
            )
            source_binding = _create_sealed_source_binding(
                input_path,
                expected=source_identity,
            )
            bound_input_path, bound_input_dir, _ = (
                candidate_edge_authoring._materialize_bound_input(
                    descriptor=source_binding.descriptor,
                    expected_sha256=source_binding.sha256,
                    logical_input_path=input_path,
                    dependencies=tuple(
                        _bound_input_dependency_snapshots(source_binding)
                    ),
                )
            )
            stage, plans, _ = candidate_edge_authoring._preflight_stage_and_edges(
                bound_input_path,
                document,
            )
            # Release every OpenUSD handle before deleting either projection.
            del stage
            _require_sealed_candidate_binding(candidate_binding)
            _require_candidate_path_authority(candidates_path, candidate_binding)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[Exception] = []
        if bound_input_dir is not None:
            try:
                candidate_edge_authoring._remove_bound_input_directory(bound_input_dir)
            except Exception as exc:
                cleanup_errors.append(exc)
        if source_binding is not None:
            cleanup_errors.extend(_close_source_binding(source_binding))
        if candidate_binding is not None:
            cleanup_errors.extend(_close_sealed_candidate_binding(candidate_binding))
        if cleanup_errors:
            detail = "; ".join(str(error) for error in cleanup_errors)
            if primary_error is not None:
                primary_error.add_note("Bound source cleanup also failed: " + detail)
            elif len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            else:
                raise ExceptionGroup("Bound source cleanup failed", cleanup_errors)
    if not plans:
        raise NoReadyJointCandidatesError(
            "The transitional Stage 2 core bridge requires at least one "
            "ready_for_rigger_input candidate"
        )
    if identify_usd_artifact(input_path, uri=str(input_path)) != source_identity:
        raise RuntimeError("Input USD changed while the Stage 2 request was built")
    if (
        _candidate_file_sha256(
            candidates_path,
            label="Stage 2 candidate document",
        )
        != candidates_sha256
    ):
        raise RuntimeError(
            "Stage 2 candidate document changed while the request was built"
        )

    assert candidate_uri is not None  # successful binding invariant
    candidate_artifact = ArtifactIdentityV1(
        uri=candidate_uri,
        root_sha256=candidates_sha256,
    )
    joint_plans = tuple(
        _shared_joint_plan(plan, candidate_artifact=candidate_artifact)
        for plan in plans
    )
    return JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=source_identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=joint_plans,
        ),
        # This owned structured path never manufactures a wheel label or falls
        # back to the external component_name-only interface.
        legacy_component_names=None,
    )


@dataclass(frozen=True)
class Stage2CandidateEdgesBackend:
    """Facade backend that delegates physical writes to the v0 authorer.

    Published URIs and authored relative dependencies come exclusively from
    ``JointRiggerArtifactTargets.publication_*`` metadata.  The backend never
    infers the final layout from transaction staging paths.
    """

    input_usd_path: Path
    articulation_candidates_path: Path
    predictions_path: Path | None = None
    candidate_readiness: Mapping[str, Any] | None = None
    _probe_lock: LockType = dataclass_field(init=False, repr=False, compare=False)
    _probe_bindings: OrderedDict[int, _ProbeBinding] = dataclass_field(
        init=False,
        repr=False,
        compare=False,
    )

    name: ClassVar[str] = ADAPTER_NAME
    backend_name: ClassVar[str] = ADAPTER_NAME
    backend_version: ClassVar[str] = AUTHORING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize caller paths and freeze the optional policy report."""

        object.__setattr__(self, "input_usd_path", Path(self.input_usd_path))
        object.__setattr__(
            self,
            "articulation_candidates_path",
            Path(self.articulation_candidates_path),
        )
        if self.predictions_path is not None:
            raise ValueError(
                "predictions_path must be supplied through "
                "author_stage2_candidate_edges_via_core; the backend does not "
                "consume predictions and cannot validate them against every "
                "final facade target"
            )
        if self.candidate_readiness is not None:
            object.__setattr__(
                self,
                "candidate_readiness",
                MappingProxyType(dict(self.candidate_readiness)),
            )
        object.__setattr__(self, "_probe_lock", threading.Lock())
        object.__setattr__(self, "_probe_bindings", OrderedDict())

    def probe(self, request: JointRiggerInputV1) -> None:
        """Verify dependencies and prove the request still matches app inputs."""

        with self._probe_lock:
            self._purge_dead_probe_bindings()
            request_sha256 = self._validate_request_against_inputs(request)
            request_id = id(request)
            binding = self._probe_bindings.get(request_id)
            if binding is not None and binding.request_ref() is request:
                binding.count += 1
                self._probe_bindings.move_to_end(request_id)
            else:
                if binding is not None:
                    del self._probe_bindings[request_id]
                self._probe_bindings[request_id] = self._new_probe_binding(
                    request,
                    request_sha256=request_sha256,
                )
                while len(self._probe_bindings) > _MAX_PROBE_BINDINGS:
                    self._probe_bindings.popitem(last=False)

    def _new_probe_binding(
        self,
        request: JointRiggerInputV1,
        *,
        request_sha256: str,
    ) -> _ProbeBinding:
        """Create an exact-object proof that cannot retain an abandoned request."""

        return _ProbeBinding(
            request_ref=weakref.ref(request),
            request_sha256=request_sha256,
        )

    def _purge_dead_probe_bindings(self) -> None:
        """Drop collected requests while the caller owns ``_probe_lock``."""

        dead = [
            request_id
            for request_id, binding in self._probe_bindings.items()
            if binding.request_ref() is None
        ]
        for request_id in dead:
            del self._probe_bindings[request_id]

    def _validate_request_against_inputs(
        self,
        request: JointRiggerInputV1,
    ) -> str:
        """Run the expensive app preflight and return the exact request hash."""

        if (
            self.candidate_readiness is not None
            and self.candidate_readiness.get("status") == "blocked"
        ):
            raise JointRiggerBackendIncompatibleError(
                "The shared facade requires an authored root, but Stage 2 "
                "candidate readiness is blocked"
            )
        _validate_supported_request_shape(request)
        _request_candidate_sha256(
            request,
            expected_path=self.articulation_candidates_path,
        )
        try:
            expected = build_stage2_candidate_edges_input(
                input_usd_path=self.input_usd_path,
                articulation_candidates_path=self.articulation_candidates_path,
                expected_articulation_candidates_sha256=(
                    _candidate_readiness_sha256(self.candidate_readiness)
                ),
            )
        except NoReadyJointCandidatesError as exc:
            raise JointRiggerBackendIncompatibleError(
                "Stage 2 inputs no longer match the supplied JointRiggerInputV1: "
                "the candidate document no longer contains a "
                "ready_for_rigger_input edge"
            ) from exc
        actual_sha256 = canonical_sha256(request)
        if canonical_sha256(expected) != actual_sha256:
            raise JointRiggerBackendIncompatibleError(
                "Stage 2 inputs no longer match the supplied JointRiggerInputV1"
            )
        return actual_sha256

    def _consume_or_create_probe_binding(self, request: JointRiggerInputV1) -> str:
        """Consume one exact probe or safely perform the direct-author probe."""

        with self._probe_lock:
            self._purge_dead_probe_bindings()
            request_id = id(request)
            binding = self._probe_bindings.get(request_id)
            if binding is not None and binding.request_ref() is request:
                if binding.count == 1:
                    del self._probe_bindings[request_id]
                else:
                    binding.count -= 1
                    self._probe_bindings.move_to_end(request_id)
                return binding.request_sha256
            if binding is not None:
                del self._probe_bindings[request_id]
            # Direct backend users retain the old probe-before-author safety.
            # Holding the lock through preflight makes shared-instance reuse
            # deterministic: another thread cannot consume this proof first.
            return self._validate_request_against_inputs(request)

    def author(
        self,
        request: JointRiggerInputV1,
        artifact_targets: JointRiggerArtifactTargets,
    ) -> JointRiggerResultV1:
        """Run v0 authoring privately and emit the facade's v1 artifact set."""

        _validate_supported_request_shape(request)
        _validate_staged_sidecar_configuration(
            input_path=self.input_usd_path,
            artifact_targets=artifact_targets,
        )
        additional_read_paths = (
            [("predictions_path", self.predictions_path)]
            if self.predictions_path is not None
            else []
        )
        # Perform target-only shape checks before creating target parents. The
        # source/candidate/dependency read closure is intentionally deferred
        # until every target entry has been descriptor-bound below.
        _validate_backend_targets_against_reads(
            artifact_targets,
            read_paths=[],
            generated_sidecar_path=None,
        )
        publication_output_path = artifact_targets.publication_output_path
        if publication_output_path is None:  # pragma: no cover - model invariant
            raise JointRiggerBackendIncompatibleError(
                "Facade omitted the publication output path"
            )

        private_artifacts: _PrivateBackendArtifacts | None = None
        candidate_binding: _SealedDependencyBinding | None = None
        source_binding: _SealedSourceBinding | None = None
        captured_source_read_paths: list[tuple[str, Path]] = []
        sealed_artifacts: tuple[_SealedPrivateArtifact, ...] = ()
        sealed_output: _SealedPrivateArtifact | None = None
        sealed_sidecar: _SealedPrivateArtifact | None = None
        target_snapshots: tuple[_TargetEntrySnapshot, ...] = ()
        primary_error: BaseException | None = None
        try:
            _ensure_target_parent_directories(artifact_targets)
            target_snapshots = _capture_target_entry_snapshots(artifact_targets)
            _validate_backend_artifact_targets(
                input_path=self.input_usd_path,
                candidates_path=self.articulation_candidates_path,
                artifact_targets=artifact_targets,
                additional_read_paths=additional_read_paths,
            )
            request_sha256 = self._consume_or_create_probe_binding(request)
            candidate_sha256 = _request_candidate_sha256(
                request,
                expected_path=self.articulation_candidates_path,
            )
            _require_source_identity(self.input_usd_path, request.source_asset)
            candidate_binding = _create_sealed_candidate_binding(
                self.articulation_candidates_path,
                expected_sha256=candidate_sha256,
            )
            _require_candidate_path_authority(
                self.articulation_candidates_path,
                candidate_binding,
            )
            source_binding = _create_sealed_source_binding(
                self.input_usd_path,
                expected=request.source_asset,
            )
            captured_source_read_paths = [
                (
                    "captured articulation_candidates_path",
                    candidate_binding.path,
                ),
                *_sealed_source_read_paths(source_binding),
            ]
            _validate_backend_artifact_targets(
                input_path=self.input_usd_path,
                candidates_path=self.articulation_candidates_path,
                artifact_targets=artifact_targets,
                additional_read_paths=additional_read_paths,
                captured_read_paths=captured_source_read_paths,
            )
            private_artifacts = _create_private_backend_artifacts(
                artifact_targets,
                target_snapshots=target_snapshots,
            )
            private_targets = private_artifacts.targets
            sidecar_layout = _validate_staged_sidecar_configuration(
                input_path=self.input_usd_path,
                artifact_targets=private_targets,
            )
            _validate_backend_artifact_targets(
                input_path=self.input_usd_path,
                candidates_path=self.articulation_candidates_path,
                artifact_targets=private_targets,
                additional_read_paths=additional_read_paths,
                captured_read_paths=captured_source_read_paths,
                generated_sidecar_path=(
                    sidecar_layout[0] if sidecar_layout is not None else None
                ),
            )

            with tempfile.TemporaryDirectory(prefix="joint-rigger-stage2-v0-") as tmp:
                private_dir = Path(tmp)
                candidate_snapshot = private_dir / "candidates.json"
                _write_sealed_candidate_snapshot(
                    candidate_binding,
                    candidate_snapshot,
                )
                expected_v0_edges, meters_per_unit = _preflight_expected_v0_edges(
                    input_path=self.input_usd_path,
                    candidates_path=candidate_snapshot,
                    bound_input_descriptor=source_binding.descriptor,
                    bound_input_sha256=source_binding.sha256,
                    bound_input_dependencies=tuple(
                        _bound_input_dependency_snapshots(source_binding)
                    ),
                )
                private_diagnostics = private_dir / "diagnostics-v0.json"
                private_validation = private_dir / "validation-v0.json"
                v0_result = author_stage2_candidate_edges(
                    input_usd_path=self.input_usd_path,
                    articulation_candidates_path=candidate_snapshot,
                    output_usd_path=private_targets.output_path,
                    diagnostics_path=private_diagnostics,
                    validation_path=private_validation,
                    predictions_path=self.predictions_path,
                    candidate_readiness=self.candidate_readiness,
                    _skip_direct_artifact_clear=True,
                    _bound_input_descriptor=source_binding.descriptor,
                    _bound_input_sha256=source_binding.sha256,
                    _bound_input_dependencies=tuple(
                        _bound_input_dependency_snapshots(source_binding)
                    ),
                    _logical_output_parent=publication_output_path.parent,
                )
                if sidecar_layout is None:
                    sealed_output = _seal_private_regular_file(
                        "output_path",
                        private_artifacts.output.stable_path,
                    )
                    sealed_artifacts = (sealed_output,)
                _require_sealed_candidate_binding(candidate_binding)
                _require_sealed_source_binding(source_binding)
                sealed_sidecar = _seal_private_sidecar(private_artifacts)
                if sealed_sidecar is not None:
                    sealed_artifacts = (sealed_sidecar,)
                v0_diagnostics = _load_json_object(
                    private_diagnostics,
                    label="private v0 diagnostics",
                )
                v0_validation = _load_json_object(
                    private_validation,
                    label="private v0 validation",
                )
                _validate_private_v0_result(
                    request=request,
                    result=v0_result,
                    diagnostics=v0_diagnostics,
                    validation=v0_validation,
                    expected_edges=expected_v0_edges,
                    meters_per_unit=meters_per_unit,
                )
                _require_candidate_file_sha256(
                    candidate_snapshot,
                    candidate_sha256,
                    label="bound Stage 2 candidate snapshot",
                )

            if sidecar_layout is not None:
                direct_sidecar, staged_sidecar, publication_sidecar = sidecar_layout
                _rebase_staged_sidecar_paths(
                    output_path=private_targets.output_path,
                    staged_sidecar_name=direct_sidecar.name,
                    final_sidecar_name=publication_sidecar.name,
                )
                if direct_sidecar != staged_sidecar:
                    raise JointRiggerArtifactError(
                        "Private Stage 2 sidecar did not use its descriptor-parented "
                        "generated path"
                    )

            _require_source_identity(self.input_usd_path, request.source_asset)
            _require_candidate_path_authority(
                self.articulation_candidates_path,
                candidate_binding,
            )

            diagnostics = _shared_diagnostics(request, v0_diagnostics)
            if private_targets.sidecar_path is None:
                output_artifact = _identify_backend_artifact(
                    private_targets.output_path.resolve(strict=True),
                    uri=str(publication_output_path),
                    label="generated USD",
                )
            else:
                _require_sealed_private_sidecar(
                    private_artifacts,
                    sealed_artifacts,
                )
                dependency_bundle_sha256 = sidecar_dependency_bundle_sha256(
                    private_targets.sidecar_path
                )
                _require_sealed_private_sidecar(
                    private_artifacts,
                    sealed_artifacts,
                )
                output_artifact = ArtifactIdentityV1(
                    uri=str(publication_output_path),
                    root_sha256=_file_sha256(
                        private_targets.output_path,
                        label="generated USD",
                    ),
                    dependency_bundle_sha256=dependency_bundle_sha256,
                )
            result = JointRiggerResultV1(
                schema_version=RESULT_SCHEMA_VERSION,
                status="succeeded",
                input_sha256=request_sha256,
                plan_sha256=canonical_sha256(request.plan),
                output_artifact=output_artifact,
                diagnostics=diagnostics,
            )
            _write_contract_report(private_targets.diagnostics_path, diagnostics)
            _write_contract_report(private_targets.result_path, result)
            sealed_artifacts = _seal_private_backend_artifacts(
                private_artifacts,
                sealed_sidecar=sealed_sidecar,
                sealed_output=sealed_output,
            )

            # Preserve the bridge's domain-specific drift diagnostics before
            # the shared promoter performs its own descriptor/content checks
            # under publication locks. The shared state remains authoritative
            # for races after this check, including after the prebackup hook.
            _require_target_entry_snapshots(target_snapshots)
            promotion = _private_backend_promotion(
                sealed_artifacts,
                target_snapshots=target_snapshots,
            )

            def require_prebackup_state() -> None:
                _require_target_entry_snapshots(target_snapshots)
                _require_sealed_candidate_binding(candidate_binding)
                _require_candidate_path_authority(
                    self.articulation_candidates_path,
                    candidate_binding,
                )
                _validate_staged_sidecar_configuration(
                    input_path=self.input_usd_path,
                    artifact_targets=artifact_targets,
                )
                _validate_backend_artifact_targets(
                    input_path=self.input_usd_path,
                    candidates_path=self.articulation_candidates_path,
                    artifact_targets=artifact_targets,
                    additional_read_paths=additional_read_paths,
                    captured_read_paths=captured_source_read_paths,
                )
                _require_sealed_private_sidecar(
                    private_artifacts,
                    sealed_artifacts,
                )

            def require_precommit_state() -> None:
                _require_target_parent_snapshots(target_snapshots)
                _require_sealed_candidate_binding(candidate_binding)
                _require_sealed_source_binding(source_binding)
                _require_sealed_private_sidecar(
                    private_artifacts,
                    sealed_artifacts,
                )
                _require_source_identity(self.input_usd_path, request.source_asset)
                _require_candidate_path_authority(
                    self.articulation_candidates_path,
                    candidate_binding,
                )

            promote_staged_artifacts(
                promotion,
                prebackup_validator=require_prebackup_state,
                precommit_validator=require_precommit_state,
            )
            return result
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_errors = _cleanup_private_backend_resources(
                private_artifacts,
                sealed_artifacts=sealed_artifacts,
                target_snapshots=target_snapshots,
            )
            if source_binding is not None:
                cleanup_errors.extend(_close_source_binding(source_binding))
            if candidate_binding is not None:
                cleanup_errors.extend(
                    _close_sealed_candidate_binding(candidate_binding)
                )
            if cleanup_errors:
                detail = "; ".join(str(error) for error in cleanup_errors)
                if primary_error is not None:
                    primary_error.add_note(
                        "Private backend cleanup also failed: " + detail
                    )
                elif len(cleanup_errors) == 1:
                    raise cleanup_errors[0]
                else:
                    raise ExceptionGroup(
                        "Private backend cleanup failed",
                        cleanup_errors,
                    )


def author_stage2_candidate_edges_via_core(
    *,
    input_usd_path: str | Path,
    articulation_candidates_path: str | Path,
    artifact_targets: JointRiggerArtifactTargets,
    predictions_path: str | Path | None = None,
    candidate_readiness: Mapping[str, Any] | None = None,
) -> JointRiggerResultV1:
    """Translate Stage 2 inputs and author them through the shared facade.

    App-owned candidate and prediction reads are validated against the final
    facade targets before ``author_joint_rig`` probes or stages work. Existing
    artifacts remain intact until the facade atomically promotes a complete
    replacement. Existing pipeline and service schemas intentionally remain
    unchanged; callers must opt into this transitional helper directly.
    """

    input_path = Path(input_usd_path)
    candidates_path = Path(articulation_candidates_path)
    predictions = Path(predictions_path) if predictions_path is not None else None

    def build_request_and_backend() -> tuple[
        JointRiggerInputV1,
        Stage2CandidateEdgesBackend,
    ]:
        _validate_backend_artifact_targets(
            input_path=input_path,
            candidates_path=candidates_path,
            artifact_targets=artifact_targets,
            additional_read_paths=(
                [("predictions_path", predictions)] if predictions is not None else []
            ),
        )
        _validate_sidecar_configuration(
            input_path=input_path,
            output_path=artifact_targets.output_path,
            configured_sidecar=artifact_targets.sidecar_path,
        )
        try:
            request = build_stage2_candidate_edges_input(
                input_usd_path=input_path,
                articulation_candidates_path=candidates_path,
                expected_articulation_candidates_sha256=(
                    _candidate_readiness_sha256(candidate_readiness)
                ),
            )
        except NoReadyJointCandidatesError as exc:
            raise InitialNoReadyJointCandidatesError(str(exc)) from exc
        backend = Stage2CandidateEdgesBackend(
            input_usd_path=input_path,
            articulation_candidates_path=candidates_path,
            candidate_readiness=candidate_readiness,
        )
        return request, backend

    return author_joint_rig_from_factory(
        build_request_and_backend,
        artifact_targets,
    )


def _shared_joint_plan(
    plan: Any,
    *,
    candidate_artifact: ArtifactIdentityV1,
) -> JointPlanV1:
    candidate = plan.candidate
    topology_provenance = {
        "joint_type": _candidate_provenance(
            candidate_artifact,
            candidate=candidate,
            source_field="motion_type",
            prim_path=plan.body1,
        ),
        "body0": _candidate_provenance(
            candidate_artifact,
            candidate=candidate,
            source_field="fixed_parent_prim",
            prim_path=plan.body0,
        ),
        "body1": _candidate_provenance(
            candidate_artifact,
            candidate=candidate,
            source_field="moving_part_prims[0]",
            prim_path=plan.body1,
        ),
    }
    axis_stage = None
    if plan.joint_type in {"revolute", "prismatic"}:
        axis_stage = plan.motion_axis_world
        topology_provenance["axis_stage"] = _candidate_provenance(
            candidate_artifact,
            candidate=candidate,
            source_field="motion_axis_world",
            prim_path=plan.body1,
        )

    limit = None
    if candidate.limit_readiness == "source_backed":
        limit_provenance = FieldProvenanceV1(
            source="accepted_manifest",
            artifact=candidate_artifact,
            prim_path=plan.body1,
            properties=("limit_unit", "lower_limit", "upper_limit"),
            evidence=(
                f"Stage 2 candidate {candidate.candidate_id!r} accepted "
                f"source-backed {candidate.limit_source} limits from the "
                "bound candidate manifest."
            ),
        )
        limit = JointLimitV1(
            lower=candidate.lower_limit,
            upper=candidate.upper_limit,
            unit=candidate.limit_unit,
            provenance=limit_provenance,
        )

    return JointPlanV1(
        topology=JointTopologyV1(
            joint_id=plan.joint_path,
            joint_type=plan.joint_type,
            body0=plan.body0,
            body1=plan.body1,
            axis_stage=axis_stage,
            field_provenance=topology_provenance,
        ),
        limit=limit,
        # The v0 authorer derives a shared anchor at the body1 world origin.
        # It is not source evidence, so the v1 plan intentionally omits it.
        anchor=None,
    )


def _candidate_provenance(
    artifact: ArtifactIdentityV1,
    *,
    candidate: Any,
    source_field: str,
    prim_path: str,
) -> FieldProvenanceV1:
    raw_source = _candidate_field_source(candidate, source_field)
    return FieldProvenanceV1(
        source="accepted_manifest",
        artifact=artifact,
        prim_path=prim_path,
        properties=(source_field,),
        derivation="preflighted_stage2_candidate_edge",
        evidence=(
            f"Stage 2 candidate {candidate.candidate_id!r} accepted "
            f"{source_field} from {raw_source}."
        ),
    )


def _candidate_field_source(candidate: Any, source_field: str) -> str:
    if source_field == "moving_part_prims[0]":
        sources = sorted({item.source for item in candidate.connectivity_evidence})
        return ",".join(sources) or "accepted_manifest"
    value = candidate.field_sources.get(source_field)
    return str(value or "accepted_manifest")


def _validate_supported_request_shape(request: JointRiggerInputV1) -> None:
    if request.legacy_component_names is not None:
        raise JointRiggerBackendIncompatibleError(
            "stage2_candidate_edges does not consume legacy component_name inputs"
        )
    if request.plan.rigid_bodies or request.plan.articulation_root is not None:
        raise JointRiggerBackendIncompatibleError(
            "stage2_candidate_edges authors topology only, not body physics schemas"
        )
    for joint in request.plan.joints:
        if joint.topology.joint_type == "spherical":
            raise JointRiggerBackendIncompatibleError(
                "stage2_candidate_edges cannot safely author spherical topology: "
                "the v0 authorer emits an axis and local frames that are not "
                "represented by the shared v1 spherical contract"
            )
        if any(
            value is not None
            for value in (joint.anchor, joint.drive, joint.state, joint.mimic)
        ):
            raise JointRiggerBackendIncompatibleError(
                "stage2_candidate_edges accepts topology and source-backed limits "
                "only; anchors are an explicitly diagnosed backend default"
            )


def _request_candidate_sha256(
    request: JointRiggerInputV1,
    *,
    expected_path: Path,
) -> str:
    """Return the one candidate identity already frozen into the v1 request."""

    identities: set[tuple[str, str]] = set()
    for joint in request.plan.joints:
        provenances = list(joint.topology.field_provenance.values())
        if joint.limit is not None:
            provenances.append(joint.limit.provenance)
        for provenance in provenances:
            artifact = provenance.artifact
            if artifact is None:
                raise JointRiggerBackendIncompatibleError(
                    "Stage 2 request provenance must identify the candidate document"
                )
            identities.add((artifact.uri, artifact.root_sha256))
    if len(identities) != 1:
        raise JointRiggerBackendIncompatibleError(
            "Stage 2 request must bind exactly one candidate document identity"
        )
    candidate_uri, candidate_sha256 = identities.pop()
    if Path(candidate_uri).expanduser().resolve(strict=False) != (
        expected_path.expanduser().resolve(strict=False)
    ):
        raise JointRiggerBackendIncompatibleError(
            "Stage 2 request candidate identity does not match the configured read path"
        )
    return candidate_sha256


def _preflight_expected_v0_edges(
    *,
    input_path: Path,
    candidates_path: Path,
    bound_input_descriptor: int | None = None,
    bound_input_sha256: str | None = None,
    bound_input_dependencies: tuple[tuple[str, int, str, str, bool], ...] = (),
) -> tuple[dict[str, dict[str, Any]], float]:
    """Reuse v0 preflight to freeze the exact backend-authored frame facts."""

    from pxr import UsdGeom

    authoring_input_path = input_path
    bound_input_dir: BoundInputDirectory | None = None
    try:
        if bound_input_descriptor is not None:
            if bound_input_sha256 is None:
                raise ValueError("bound input SHA-256 is required")
            authoring_input_path, bound_input_dir, _ = (
                candidate_edge_authoring._materialize_bound_input(
                    descriptor=bound_input_descriptor,
                    expected_sha256=bound_input_sha256,
                    logical_input_path=input_path,
                    dependencies=bound_input_dependencies,
                )
            )
        document = candidate_edge_authoring._load_and_validate_document(candidates_path)
        stage, plans, _ = candidate_edge_authoring._preflight_stage_and_edges(
            authoring_input_path,
            document,
        )
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        del stage
        return (
            {
                plan.joint_path: candidate_edge_authoring._authored_edge_record(plan)
                for plan in plans
            },
            meters_per_unit,
        )
    finally:
        if bound_input_dir is not None:
            candidate_edge_authoring._remove_bound_input_directory(bound_input_dir)


def _validate_private_v0_result(
    *,
    request: JointRiggerInputV1,
    result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    validation: Mapping[str, Any],
    expected_edges: Mapping[str, Mapping[str, Any]] | None = None,
    meters_per_unit: float = 1.0,
) -> None:
    expected_count = len(request.plan.joints)
    if result.get("joint_rigger_status") != "authored":
        raise JointRiggerArtifactError(
            "Private Stage 2 authorer did not return joint_rigger_status=authored"
        )
    if result.get("authored_joint_count") != expected_count:
        raise JointRiggerArtifactError(
            "Private Stage 2 authorer returned an unexpected joint count"
        )
    if diagnostics.get("status") != "authored":
        raise JointRiggerArtifactError(
            "Private Stage 2 diagnostics did not report status=authored"
        )
    if diagnostics.get("predictions_consumed") is not False:
        raise JointRiggerArtifactError(
            "Private Stage 2 authorer unexpectedly consumed predictions"
        )
    if validation.get("status") != "passed":
        raise JointRiggerArtifactError(
            "Private Stage 2 validation did not report status=passed"
        )

    expected_graph = sorted(
        (
            joint.topology.joint_id,
            joint.topology.joint_type,
            joint.topology.body0,
            joint.topology.body1,
            joint.topology.axis_stage,
        )
        for joint in request.plan.joints
    )
    raw_edges = diagnostics.get("authored_edges")
    if not isinstance(raw_edges, list):
        raise JointRiggerArtifactError(
            "Private Stage 2 diagnostics omitted authored_edges"
        )
    observed_graph = sorted(
        (
            str(edge.get("joint_path")),
            str(edge.get("joint_type")),
            str(edge.get("body0")),
            str(edge.get("body1")),
            (
                tuple(float(value) for value in edge.get("motion_axis_world", ()))
                if edge.get("joint_type") in {"revolute", "prismatic"}
                else None
            ),
        )
        for edge in raw_edges
        if isinstance(edge, Mapping)
    )
    if observed_graph != expected_graph:
        raise JointRiggerArtifactError(
            "Private Stage 2 authored graph does not match the shared v1 plan"
        )
    observed_edges = {
        str(edge["joint_path"]): edge
        for edge in raw_edges
        if isinstance(edge, Mapping) and "joint_path" in edge
    }
    for joint in request.plan.joints:
        joint_path = joint.topology.joint_id
        observed_edge = observed_edges.get(joint_path)
        if observed_edge is None:
            raise JointRiggerArtifactError(
                f"Private Stage 2 diagnostics omitted planned edge {joint_path}"
            )
        expected_lower: float | None = None
        expected_upper: float | None = None
        expected_unit: str | None = None
        if joint.limit is not None:
            expected_lower = joint.limit.lower
            expected_upper = joint.limit.upper
            if joint.topology.joint_type == "prismatic":
                expected_lower = _optional_divide(expected_lower, meters_per_unit)
                expected_upper = _optional_divide(expected_upper, meters_per_unit)
                expected_unit = "stage_units"
            else:
                expected_unit = "degrees"
        observed_provenance = observed_edge.get("field_provenance")
        observed_limit_provenance = (
            observed_provenance.get("limits")
            if isinstance(observed_provenance, Mapping)
            else None
        )
        observed_unit = (
            observed_limit_provenance.get("authored_unit")
            if isinstance(observed_limit_provenance, Mapping)
            else None
        )
        if (
            observed_edge.get("lower_limit") != expected_lower
            or observed_edge.get("upper_limit") != expected_upper
            or observed_unit != expected_unit
        ):
            raise JointRiggerArtifactError(
                "Private Stage 2 authored limits do not match the v1 request for "
                f"{joint_path}"
            )
    if expected_edges is None:
        return

    if set(observed_edges) != set(expected_edges):
        raise JointRiggerArtifactError(
            "Private Stage 2 diagnostics do not identify every preflighted edge"
        )
    exact_fields = (
        "axis_token",
        "local_pos0",
        "local_pos1",
        "local_rot0",
        "local_rot1",
        "anchor_world",
        "lower_limit",
        "upper_limit",
    )
    for joint_path, expected_edge in expected_edges.items():
        observed_edge = observed_edges[joint_path]
        for field in exact_fields:
            if (
                field not in observed_edge
                or observed_edge[field] != expected_edge[field]
            ):
                raise JointRiggerArtifactError(
                    "Private Stage 2 diagnostics changed authored fact "
                    f"{joint_path}.{field}"
                )
        expected_limit_provenance = expected_edge["field_provenance"]["limits"]
        observed_provenance = observed_edge.get("field_provenance")
        if not isinstance(observed_provenance, Mapping) or (
            observed_provenance.get("limits") != expected_limit_provenance
        ):
            raise JointRiggerArtifactError(
                "Private Stage 2 diagnostics changed authored limit provenance "
                f"for {joint_path}"
            )


def _shared_diagnostics(
    request: JointRiggerInputV1,
    v0_diagnostics: Mapping[str, Any],
) -> JointRiggerDiagnosticsV1:
    edges = {
        str(edge.get("joint_path")): edge
        for edge in v0_diagnostics.get("authored_edges", [])
        if isinstance(edge, Mapping)
    }
    joint_diagnostics = []
    for joint in request.plan.joints:
        topology = joint.topology
        decisions = [
            FieldDecisionV1(
                field=f"topology.{field}",
                disposition="accepted",
                provenance=topology.field_provenance[field],
            )
            for field in ("joint_type", "body0", "body1")
        ]
        if topology.axis_stage is not None:
            decisions.append(
                FieldDecisionV1(
                    field="topology.axis_stage",
                    disposition="accepted",
                    provenance=topology.field_provenance["axis_stage"],
                )
            )
        if joint.limit is None:
            decisions.append(
                FieldDecisionV1(
                    field="limit",
                    disposition="ignored",
                    reason_code="not_provided",
                    detail="No source-backed scalar limit was accepted.",
                )
            )
        else:
            if joint.limit.lower is not None:
                decisions.append(
                    FieldDecisionV1(
                        field="limit.lower",
                        disposition="accepted",
                        provenance=joint.limit.provenance,
                    )
                )
            if joint.limit.upper is not None:
                decisions.append(
                    FieldDecisionV1(
                        field="limit.upper",
                        disposition="accepted",
                        provenance=joint.limit.provenance,
                    )
                )
            decisions.append(
                FieldDecisionV1(
                    field="limit.unit",
                    disposition="accepted",
                    provenance=joint.limit.provenance,
                )
            )

        edge = edges.get(topology.joint_id, {})
        anchor = edge.get("anchor_world")
        decisions.append(
            FieldDecisionV1(
                field="anchor",
                disposition="defaulted",
                reason_code="inferred_body1_world_origin",
                detail=(
                    "The existing Stage 2 authorer placed the shared anchor at "
                    f"body1's world origin: {anchor!r}."
                ),
            )
        )
        joint_diagnostics.append(
            JointDiagnosticV1(
                joint_id=topology.joint_id,
                field_decisions=tuple(decisions),
                reason_codes=("inferred_body1_world_origin",),
            )
        )

    warnings = tuple(str(item) for item in v0_diagnostics.get("warnings", ()))
    return JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name=ADAPTER_NAME,
        backend_version=AUTHORING_SCHEMA_VERSION,
        field_decisions=(
            FieldDecisionV1(
                field="legacy_component_names",
                disposition="ignored",
                reason_code="not_applicable_structured_stage2_input",
                detail=(
                    "This backend consumes explicit Stage 2 topology and never "
                    "falls back to wheel or component_name labels."
                ),
            ),
        ),
        joint_diagnostics=tuple(joint_diagnostics),
        warnings=warnings,
    )


def _descriptor_mount_id(descriptor: int) -> int:
    """Return one Linux mount ID from a retained descriptor, failing closed."""

    try:
        lines = (
            Path(f"/proc/self/fdinfo/{descriptor}")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    except OSError as exc:
        raise JointRiggerBackendIncompatibleError(
            "Safe Joint Rigger publication requires Linux /proc/self/fdinfo"
        ) from exc
    values: list[int] = []
    for line in lines:
        key, separator, value = line.partition(":")
        if key != "mnt_id":
            continue
        if not separator:
            raise JointRiggerBackendIncompatibleError(
                f"Malformed mount ID for descriptor {descriptor}"
            )
        try:
            mount_id = int(value.strip())
        except ValueError as exc:
            raise JointRiggerBackendIncompatibleError(
                f"Malformed mount ID for descriptor {descriptor}"
            ) from exc
        if mount_id <= 0:
            raise JointRiggerBackendIncompatibleError(
                f"Malformed mount ID for descriptor {descriptor}"
            )
        values.append(mount_id)
    if len(values) != 1:
        raise JointRiggerBackendIncompatibleError(
            f"Expected exactly one mount ID for descriptor {descriptor}"
        )
    return values[0]


def _require_directory_tree_mount_id(
    descriptor: int,
    *,
    expected_mount_id: int,
    label: str,
    relative_path: str = ".",
) -> None:
    """Apply the shared fixed traversal ceilings before sidecar operations."""

    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise JointRiggerArtifactError(f"{label} is not a directory")
    try:
        _shared_require_directory_tree_mount_id(
            descriptor,
            expected_mount_id=expected_mount_id,
            label=label,
            relative_path=relative_path,
        )
    except (RuntimeError, ValueError) as exc:
        raise JointRiggerArtifactError(str(exc)) from exc


def _snapshot_directory_tree(
    descriptor: int,
    *,
    parent_mount_id: int,
    label: str,
) -> _DirectoryTreeSnapshot:
    """Capture one exact fd-bound tree after validating every mount boundary."""

    metadata = os.fstat(descriptor)
    identity = (metadata.st_dev, metadata.st_ino)
    mount_id = _descriptor_mount_id(descriptor)
    if mount_id != parent_mount_id:
        raise JointRiggerArtifactError(f"{label} root is a mount point")
    _require_directory_tree_mount_id(
        descriptor,
        expected_mount_id=mount_id,
        label=label,
    )
    tree_sha256 = directory_descriptor_tree_sha256(descriptor)
    _require_directory_tree_mount_id(
        descriptor,
        expected_mount_id=mount_id,
        label=label,
    )
    after = os.fstat(descriptor)
    if (after.st_dev, after.st_ino) != identity:
        raise JointRiggerArtifactError(f"{label} root changed inode")
    return _DirectoryTreeSnapshot(
        descriptor=descriptor,
        identity=identity,
        mount_id=mount_id,
        tree_sha256=tree_sha256,
    )


def _close_descriptors(descriptors: list[int] | set[int]) -> list[Exception]:
    """Attempt every distinct close and return all failures."""

    errors: list[Exception] = []
    for descriptor in dict.fromkeys(descriptors):
        try:
            os.close(descriptor)
        except Exception as exc:
            errors.append(exc)
    return errors


def _close_captured_target_states(
    states: list[_CapturedTargetState] | tuple[_CapturedTargetState, ...],
) -> list[Exception]:
    """Close each shared target-capture handle exactly once."""

    errors: list[Exception] = []
    handles = {
        id(state.entry_handle): state.entry_handle
        for state in states
        if state.entry_handle is not None
    }
    for handle in handles.values():
        try:
            handle.close()
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
        primary_error.add_note(label + ": " + "; ".join(str(error) for error in errors))


def _directory_path_identity_and_mount(path: Path) -> tuple[int, int, int]:
    """Bind one directory path and return its inode and Linux mount identity."""

    resolved = path.expanduser().resolve(strict=True)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(resolved, flags)
    try:
        held = os.fstat(descriptor)
        observed = path.stat()
        identity = (held.st_dev, held.st_ino)
        if (observed.st_dev, observed.st_ino) != identity:
            raise JointRiggerArtifactError(
                f"Artifact directory changed while it was opened: {path}"
            )
        return held.st_dev, held.st_ino, _descriptor_mount_id(descriptor)
    finally:
        os.close(descriptor)


def _validate_physical_publication_root_parent(
    targets: JointRiggerArtifactTargets,
) -> None:
    """Require physical and publication roots to share one directory mount."""

    publication_output = targets.publication_output_path
    if publication_output is None:  # pragma: no cover - model invariant
        raise JointRiggerBackendIncompatibleError(
            "Facade omitted the publication output path"
        )
    physical_parent = targets.output_path.parent
    publication_parent = publication_output.parent
    if Path(os.path.abspath(physical_parent)) == Path(
        os.path.abspath(publication_parent)
    ):
        return
    try:
        physical_identity = _directory_path_identity_and_mount(physical_parent)
        publication_identity = _directory_path_identity_and_mount(publication_parent)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise JointRiggerBackendIncompatibleError(
            "Physical and publication output parents must both exist and match"
        ) from exc
    if physical_identity != publication_identity:
        raise JointRiggerBackendIncompatibleError(
            "Physical and publication output parents must identify the same "
            "directory and mount"
        )


def _ensure_target_parent_directories(
    targets: JointRiggerArtifactTargets,
) -> None:
    """Create target parents before binding their stable physical identities."""

    for _, path in _artifact_target_paths(targets):
        path.parent.mkdir(parents=True, exist_ok=True)


def _reserve_private_staging_entry(
    target_snapshot: _TargetEntrySnapshot,
) -> _PrivateStagingEntry:
    """Reserve one absent random name under a retained physical parent fd."""

    target_path = target_snapshot.path
    parent_path = target_snapshot.parent_path
    parent_descriptor = os.dup(target_snapshot.parent_descriptor)
    try:
        parent_metadata = os.fstat(parent_descriptor)
        parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
        parent_mount_id = _descriptor_mount_id(parent_descriptor)
        if (
            parent_identity != target_snapshot.parent_identity
            or parent_mount_id != target_snapshot.parent_mount_id
        ):
            raise JointRiggerArtifactError(
                f"Artifact target parent changed while it was duplicated: {parent_path}"
            )
        proc_parent = Path(f"/proc/self/fd/{parent_descriptor}")
        proc_metadata = proc_parent.stat()
        if (proc_metadata.st_dev, proc_metadata.st_ino) != parent_identity:
            raise JointRiggerBackendIncompatibleError(
                "Descriptor-stable private staging requires Linux /proc/self/fd"
            )
        for _ in range(128):
            name = (
                f".{target_path.stem}.backend-{secrets.token_hex(12)}"
                f"{target_path.suffix}"
            )
            create_flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW
            create_flags |= getattr(os, "O_CLOEXEC", 0)
            try:
                descriptor = os.open(
                    name,
                    create_flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:  # pragma: no cover - cryptographic collision
                continue
            try:
                os.unlink(name, dir_fd=parent_descriptor)
            except BaseException as unlink_error:
                # Like close(2), an unlink error can be reported after the
                # destructive operation took effect. Retrying by name could
                # delete an unrelated entry created in the meantime.
                cleanup_errors: list[Exception] = []
                try:
                    os.close(descriptor)
                except Exception as exc:
                    cleanup_errors.append(exc)
                _add_cleanup_error_note(
                    unlink_error,
                    label="Private staging placeholder descriptor cleanup also failed",
                    errors=cleanup_errors,
                )
                raise
            try:
                os.close(descriptor)
            except BaseException:
                # The private name is already absent. A close error leaves
                # ownership of the numeric descriptor indeterminate, so never
                # retry it after another thread could reuse the number.
                raise
            return _PrivateStagingEntry(
                parent_path=parent_path,
                parent_descriptor=parent_descriptor,
                parent_identity=parent_identity,
                parent_mount_id=parent_mount_id,
                name=name,
            )
        raise JointRiggerArtifactError("Could not reserve private backend staging name")
    except BaseException as reservation_error:
        close_errors = _close_descriptors([parent_descriptor])
        _add_cleanup_error_note(
            reservation_error,
            label="Private staging parent cleanup also failed",
            errors=close_errors,
        )
        raise


def _cleanup_private_staging_entries(
    entries: list[_PrivateStagingEntry] | tuple[_PrivateStagingEntry, ...],
    *,
    sealed_artifacts: tuple[_SealedPrivateArtifact, ...] = (),
) -> list[Exception]:
    """Remove only still-owned private names, then close every parent fd."""

    errors: list[Exception] = []
    sealed_by_path = {artifact.path: artifact for artifact in sealed_artifacts}
    for entry in entries:
        sealed = sealed_by_path.get(entry.stable_path)
        if sealed is None:
            # An absent cryptographically random name proves only that it was
            # free when reserved. If an entry appears before we retain its
            # descriptor during sealing, we cannot distinguish backend output
            # from a foreign occupant. Preserve it rather than treating a
            # cleanup-time open as retroactive ownership proof.
            try:
                os.stat(
                    entry.name,
                    dir_fd=entry.parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            except Exception as exc:
                errors.append(exc)
                continue
            errors.append(
                JointRiggerArtifactError(
                    "Unsealed private staging entry has no retained ownership "
                    f"proof; refusing deletion: {entry.stable_path}"
                )
            )
            continue
        try:
            current = os.stat(
                entry.name,
                dir_fd=entry.parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        except Exception as exc:
            errors.append(exc)
            continue
        try:
            owned = os.fstat(sealed.descriptor)
            if (current.st_dev, current.st_ino) != (owned.st_dev, owned.st_ino):
                raise JointRiggerArtifactError(
                    "Private staging entry changed inode before cleanup; "
                    f"refusing deletion: {entry.stable_path}"
                )
            _remove_descriptor_entry(
                entry.parent_descriptor,
                entry.name,
                expected_identity=(owned.st_dev, owned.st_ino),
                source_descriptor=sealed.descriptor,
                label=f"sealed private staging entry {entry.stable_path}",
            )
        except Exception as exc:
            errors.append(exc)
    errors.extend(_close_descriptors({entry.parent_descriptor for entry in entries}))
    return errors


def _create_private_backend_artifacts(
    publication_targets: JointRiggerArtifactTargets,
    *,
    target_snapshots: tuple[_TargetEntrySnapshot, ...],
) -> _PrivateBackendArtifacts:
    """Create fd-stable private paths adjacent to each physical file target."""

    snapshots = {snapshot.label: snapshot for snapshot in target_snapshots}
    entries: list[_PrivateStagingEntry] = []
    try:
        output = _reserve_private_staging_entry(snapshots["output_path"])
        entries.append(output)
        diagnostics = _reserve_private_staging_entry(snapshots["diagnostics_path"])
        entries.append(diagnostics)
        result = _reserve_private_staging_entry(snapshots["result_path"])
        entries.append(result)
        sidecar_path = None
        if publication_targets.sidecar_path is not None:
            sidecar_path = output.stable_path.with_name(
                f"{output.stable_path.stem}_assets"
            )
        targets = JointRiggerArtifactTargets(
            output_path=output.stable_path,
            diagnostics_path=diagnostics.stable_path,
            result_path=result.stable_path,
            sidecar_path=sidecar_path,
            publication_output_path=publication_targets.publication_output_path,
            publication_sidecar_path=publication_targets.publication_sidecar_path,
        )
        return _PrivateBackendArtifacts(
            output=output,
            diagnostics=diagnostics,
            result=result,
            targets=targets,
        )
    except BaseException as creation_error:
        cleanup_errors = _cleanup_private_staging_entries(entries)
        _add_cleanup_error_note(
            creation_error,
            label="Private backend artifact creation cleanup also failed",
            errors=cleanup_errors,
        )
        raise


def _capture_target_entry_snapshots(
    targets: JointRiggerArtifactTargets,
) -> tuple[_TargetEntrySnapshot, ...]:
    """Bind each target to an original physical parent and no-follow state."""

    snapshots: list[_TargetEntrySnapshot] = []
    captured_target_states: list[_CapturedTargetState] = []
    parents: dict[tuple[int, int, int], int] = {}
    open_parent_descriptors: set[int] = set()
    open_tree_descriptors: set[int] = set()
    try:
        for label, path in _artifact_target_paths(targets):
            parent_path = path.parent
            resolved_parent = parent_path.expanduser().resolve(strict=True)
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(resolved_parent, flags)
            open_parent_descriptors.add(descriptor)
            metadata = os.fstat(descriptor)
            identity = (metadata.st_dev, metadata.st_ino)
            mount_id = _descriptor_mount_id(descriptor)
            observed_parent = parent_path.stat()
            if (observed_parent.st_dev, observed_parent.st_ino) != identity:
                raise JointRiggerArtifactError(
                    f"Artifact target parent changed while it was bound: {parent_path}"
                )
            parent_key = (*identity, mount_id)
            existing_descriptor = parents.get(parent_key)
            if existing_descriptor is not None:
                open_parent_descriptors.remove(descriptor)
                os.close(descriptor)
                descriptor = existing_descriptor
            else:
                parents[parent_key] = descriptor
            captured_target_state = _capture_target_state(
                _BoundDirectory(
                    locator_path=Path(os.path.abspath(parent_path.expanduser())),
                    opened_path=resolved_parent,
                    descriptor=descriptor,
                    identity=identity,
                ),
                path,
            )
            captured_target_states.append(captured_target_state)
            stable_target_path = Path(
                os.path.abspath(
                    Path(f"/proc/self/fd/{descriptor}") / path.name,
                )
            )
            captured_target_state = dataclass_replace(
                captured_target_state,
                requested_path=stable_target_path,
            )
            entry_state = _target_entry_state(descriptor, path.name)
            directory_tree = None
            if label == "sidecar_path" and entry_state is not None:
                if not stat.S_ISDIR(entry_state[2]):
                    raise JointRiggerArtifactError(
                        f"Existing sidecar_path is not a directory: {path}"
                    )
                tree_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                tree_flags |= getattr(os, "O_CLOEXEC", 0)
                tree_descriptor = os.open(path.name, tree_flags, dir_fd=descriptor)
                open_tree_descriptors.add(tree_descriptor)
                opened_tree = os.fstat(tree_descriptor)
                if (opened_tree.st_dev, opened_tree.st_ino) != entry_state[:2]:
                    raise JointRiggerArtifactError(
                        f"Existing sidecar_path changed while it was bound: {path}"
                    )
                directory_tree = _snapshot_directory_tree(
                    tree_descriptor,
                    parent_mount_id=mount_id,
                    label="Existing sidecar_path",
                )
            snapshots.append(
                _TargetEntrySnapshot(
                    label=label,
                    path=path,
                    parent_path=parent_path,
                    parent_descriptor=descriptor,
                    parent_identity=identity,
                    parent_mount_id=mount_id,
                    entry_state=entry_state,
                    initial_target_state=captured_target_state,
                    directory_tree=directory_tree,
                )
            )
        _require_target_parent_snapshots(tuple(snapshots))
        return tuple(snapshots)
    except BaseException as capture_error:
        close_errors = _close_captured_target_states(captured_target_states)
        close_errors.extend(
            _close_descriptors(open_tree_descriptors | open_parent_descriptors)
        )
        _add_cleanup_error_note(
            capture_error,
            label="Target snapshot cleanup also failed",
            errors=close_errors,
        )
        raise


def _target_entry_state(
    parent_descriptor: int,
    name: str,
) -> tuple[int, int, int, int, int, int, int] | None:
    try:
        metadata = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_target_parent_snapshots(
    snapshots: tuple[_TargetEntrySnapshot, ...],
) -> None:
    """Require every lexical target parent to retain its bound physical inode."""

    for snapshot in snapshots:
        try:
            held = os.fstat(snapshot.parent_descriptor)
            observed = _directory_path_identity_and_mount(snapshot.parent_path)
        except (JointRiggerArtifactError, OSError) as exc:
            raise JointRiggerArtifactError(
                f"Artifact target parent became unavailable: {snapshot.parent_path}"
            ) from exc
        expected = (*snapshot.parent_identity, snapshot.parent_mount_id)
        held_state = (
            held.st_dev,
            held.st_ino,
            _descriptor_mount_id(snapshot.parent_descriptor),
        )
        if held_state != expected or observed != expected:
            raise JointRiggerArtifactError(
                f"Artifact target parent changed during authoring: {snapshot.parent_path}"
            )


def _require_target_entry_snapshots(
    snapshots: tuple[_TargetEntrySnapshot, ...],
) -> None:
    """Fail before backup if any no-follow target entry changed."""

    _require_target_parent_snapshots(snapshots)
    for snapshot in snapshots:
        current_state = _target_entry_state(
            snapshot.parent_descriptor,
            snapshot.path.name,
        )
        if current_state != snapshot.entry_state:
            raise JointRiggerArtifactError(
                f"Artifact target changed during private authoring: {snapshot.path}"
            )
        expected_tree = snapshot.directory_tree
        if expected_tree is None:
            continue
        held = os.fstat(expected_tree.descriptor)
        if (held.st_dev, held.st_ino) != expected_tree.identity:
            raise JointRiggerArtifactError(
                f"Existing sidecar_path changed during private authoring: {snapshot.path}"
            )
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        current_descriptor = os.open(
            snapshot.path.name,
            flags,
            dir_fd=snapshot.parent_descriptor,
        )
        try:
            current_tree = _snapshot_directory_tree(
                current_descriptor,
                parent_mount_id=snapshot.parent_mount_id,
                label="Existing sidecar_path",
            )
        finally:
            os.close(current_descriptor)
        if (
            current_tree.identity != expected_tree.identity
            or current_tree.mount_id != expected_tree.mount_id
            or current_tree.tree_sha256 != expected_tree.tree_sha256
        ):
            raise JointRiggerArtifactError(
                f"Existing sidecar_path changed during private authoring: {snapshot.path}"
            )


def _stable_descriptor_sha256(descriptor: int, *, label: str) -> str:
    """Hash one regular descriptor while detecting concurrent mutation."""

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

    def states(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if states(before) != states(after):
        raise JointRiggerArtifactError(f"Private {label} changed while hashing")
    return digest.hexdigest()


def _seal_private_regular_file(
    label: str,
    path: Path,
) -> _SealedPrivateArtifact:
    """Bind one private file descriptor and remove every write permission."""

    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise JointRiggerArtifactError(
                f"Private {label} must be a singly linked regular file"
            )
        os.fchmod(descriptor, stat.S_IMODE(metadata.st_mode) & ~0o222)
        observed = os.stat(path, follow_symlinks=False)
        sealed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) != (sealed.st_dev, sealed.st_ino):
            raise JointRiggerArtifactError(f"Private {label} changed inode")
        return _SealedPrivateArtifact(
            label=label,
            path=path,
            descriptor=descriptor,
            sha256=_stable_descriptor_sha256(descriptor, label=label),
        )
    except BaseException:
        os.close(descriptor)
        raise


def _require_sealed_private_regular_file(
    artifact: _SealedPrivateArtifact,
) -> None:
    """Revalidate one retained private regular file without reopening it."""

    opened = os.fstat(artifact.descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or stat.S_IMODE(opened.st_mode) & 0o222
    ):
        raise JointRiggerArtifactError(
            f"Private {artifact.label} lost its sealed regular-file state"
        )
    observed = os.stat(artifact.path, follow_symlinks=False)
    if (observed.st_dev, observed.st_ino) != (opened.st_dev, opened.st_ino):
        raise JointRiggerArtifactError(
            f"Private {artifact.label} changed inode after it was sealed"
        )
    if (
        _stable_descriptor_sha256(artifact.descriptor, label=artifact.label)
        != artifact.sha256
    ):
        raise JointRiggerArtifactError(
            f"Private {artifact.label} changed after it was sealed"
        )


def _seal_directory_descriptor_tree(
    descriptor: int,
    *,
    expected_mount_id: int,
    label: str,
    relative_path: str = ".",
    traversal_budget: _ArtifactTreeTraversalBudget | None = None,
    depth: int = 0,
    count_root: bool = True,
) -> None:
    """Remove write permissions from a private fd-relative directory tree."""

    if traversal_budget is None:
        traversal_budget = _ArtifactTreeTraversalBudget(
            label=f"Private sidecar seal {label}"
        )
    if count_root:
        traversal_budget.consume(
            relative_path=relative_path,
            depth=depth,
        )
    root_metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise JointRiggerArtifactError("Private sidecar is not a directory")
    if _descriptor_mount_id(descriptor) != expected_mount_id:
        raise JointRiggerArtifactError(
            f"{label} contains a mount point at {relative_path}"
        )
    entries: list[tuple[str, os.stat_result]] = []
    with os.scandir(descriptor) as iterator:
        for item in iterator:
            metadata = item.stat(follow_symlinks=False)
            child_relative = (
                item.name if relative_path == "." else f"{relative_path}/{item.name}"
            )
            traversal_budget.consume(
                relative_path=child_relative,
                depth=depth + 1,
                byte_count=0 if stat.S_ISDIR(metadata.st_mode) else metadata.st_size,
            )
            entries.append((item.name, metadata))
    for name, metadata in sorted(entries, key=lambda item: item[0]):
        if stat.S_ISDIR(metadata.st_mode):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            child = os.open(name, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                if (opened.st_dev, opened.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise JointRiggerArtifactError(
                        f"Private sidecar directory changed inode: {name}"
                    )
                child_relative = (
                    name if relative_path == "." else f"{relative_path}/{name}"
                )
                _seal_directory_descriptor_tree(
                    child,
                    expected_mount_id=expected_mount_id,
                    label=label,
                    relative_path=child_relative,
                    traversal_budget=traversal_budget,
                    depth=depth + 1,
                    count_root=False,
                )
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise JointRiggerArtifactError(
                f"Private sidecar contains an invalid entry: {name}"
            )
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        child = os.open(name, flags, dir_fd=descriptor)
        try:
            opened = os.fstat(child)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
            ):
                raise JointRiggerArtifactError(
                    f"Private sidecar file changed inode: {name}"
                )
            if _descriptor_mount_id(child) != expected_mount_id:
                raise JointRiggerArtifactError(
                    f"{label} contains a mount point at {relative_path}/{name}"
                )
            os.fchmod(child, stat.S_IMODE(opened.st_mode) & ~0o222)
        finally:
            os.close(child)
    os.fchmod(descriptor, stat.S_IMODE(root_metadata.st_mode) & ~0o222)


def _seal_private_directory(
    label: str,
    path: Path,
    *,
    parent_descriptor: int,
) -> _SealedPrivateArtifact:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        observed = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or (observed.st_dev, observed.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise JointRiggerArtifactError(f"Private {label} changed inode")
        parent_mount_id = _descriptor_mount_id(parent_descriptor)
        mount_id = _descriptor_mount_id(descriptor)
        if mount_id != parent_mount_id:
            raise JointRiggerArtifactError(f"Private {label} root is a mount point")
        _require_directory_tree_mount_id(
            descriptor,
            expected_mount_id=mount_id,
            label=f"Private {label}",
        )
        _seal_directory_descriptor_tree(
            descriptor,
            expected_mount_id=mount_id,
            label=f"Private {label}",
        )
        snapshot = _snapshot_directory_tree(
            descriptor,
            parent_mount_id=parent_mount_id,
            label=f"Private {label}",
        )
        return _SealedPrivateArtifact(
            label=label,
            path=path,
            descriptor=descriptor,
            sha256=snapshot.tree_sha256,
            mount_id=snapshot.mount_id,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _seal_private_sidecar(
    private_artifacts: _PrivateBackendArtifacts,
) -> _SealedPrivateArtifact | None:
    """Seal a successfully authored private sidecar before later validation."""

    sidecar_path = private_artifacts.targets.sidecar_path
    if sidecar_path is None:
        return None
    return _seal_private_directory(
        "sidecar_path",
        sidecar_path,
        parent_descriptor=private_artifacts.output.parent_descriptor,
    )


def _require_sealed_private_sidecar(
    private_artifacts: _PrivateBackendArtifacts | None,
    sealed_artifacts: tuple[_SealedPrivateArtifact, ...],
) -> None:
    """Revalidate the exact private sidecar mount and tree while retained."""

    if private_artifacts is None or private_artifacts.targets.sidecar_path is None:
        return
    sealed = next(
        (artifact for artifact in sealed_artifacts if artifact.label == "sidecar_path"),
        None,
    )
    if sealed is None or sealed.mount_id is None:
        raise JointRiggerArtifactError("Private sidecar is not sealed")
    current = _snapshot_directory_tree(
        sealed.descriptor,
        parent_mount_id=private_artifacts.output.parent_mount_id,
        label="Private sidecar_path",
    )
    if current.mount_id != sealed.mount_id or current.tree_sha256 != sealed.sha256:
        raise JointRiggerArtifactError("Private sidecar changed after it was sealed")


def _seal_private_backend_artifacts(
    private_artifacts: _PrivateBackendArtifacts,
    *,
    sealed_sidecar: _SealedPrivateArtifact | None,
    sealed_output: _SealedPrivateArtifact | None = None,
) -> tuple[_SealedPrivateArtifact, ...]:
    """Seal a complete private bundle in root-last promotion order."""

    targets = private_artifacts.targets
    sealed: list[_SealedPrivateArtifact] = []
    newly_sealed: list[_SealedPrivateArtifact] = []
    try:
        diagnostics = _seal_private_regular_file(
            "diagnostics_path",
            targets.diagnostics_path,
        )
        sealed.append(diagnostics)
        newly_sealed.append(diagnostics)
        result = _seal_private_regular_file("result_path", targets.result_path)
        sealed.append(result)
        newly_sealed.append(result)
        if targets.sidecar_path is not None:
            if sealed_sidecar is None:
                raise JointRiggerArtifactError("Private sidecar was not sealed")
            sealed.append(sealed_sidecar)
        if sealed_output is None:
            output = _seal_private_regular_file(
                "output_path",
                private_artifacts.output.stable_path,
            )
            newly_sealed.append(output)
        else:
            _require_sealed_private_regular_file(sealed_output)
            output = sealed_output
        sealed.append(output)
        return tuple(sealed)
    except BaseException as seal_error:
        close_errors = _close_descriptors(
            {artifact.descriptor for artifact in newly_sealed}
        )
        _add_cleanup_error_note(
            seal_error,
            label="Partial private bundle seal cleanup also failed",
            errors=close_errors,
        )
        raise


def _private_backend_promotion(
    sealed_artifacts: tuple[_SealedPrivateArtifact, ...],
    *,
    target_snapshots: tuple[_TargetEntrySnapshot, ...],
) -> list[StagedArtifact]:
    """Build one descriptor-backed root-last nested publication transaction."""

    snapshots = {snapshot.label: snapshot for snapshot in target_snapshots}
    display_labels = {
        "diagnostics_path": "diagnostics report",
        "result_path": "result report",
        "sidecar_path": "composition sidecar",
        "output_path": "generated root",
    }
    return [
        StagedArtifact(
            staged_path=artifact.path,
            target_path=snapshots[artifact.label].stable_path,
            label=display_labels[artifact.label],
            source_descriptor=artifact.descriptor,
            source_sha256=artifact.sha256,
            _initial_target_state=snapshots[artifact.label].initial_target_state,
        )
        for artifact in sealed_artifacts
    ]


def _cleanup_private_sidecar(
    private_artifacts: _PrivateBackendArtifacts,
    sealed_sidecar: _SealedPrivateArtifact | None,
) -> None:
    """Remove only the exact private sidecar entry bound under its held parent."""

    sidecar_path = private_artifacts.targets.sidecar_path
    if sidecar_path is None:
        return
    parent_descriptor = private_artifacts.output.parent_descriptor
    name = sidecar_path.name
    try:
        entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if sealed_sidecar is not None:
        opened = os.fstat(sealed_sidecar.descriptor)
        if not stat.S_ISDIR(entry.st_mode) or (opened.st_dev, opened.st_ino) != (
            entry.st_dev,
            entry.st_ino,
        ):
            raise JointRiggerArtifactError(
                "Private sidecar entry changed inode before cleanup; refusing "
                "recursive deletion"
            )
    else:
        raise JointRiggerArtifactError(
            "Private sidecar has no retained identity proof; refusing deletion"
        )
    assert sealed_sidecar is not None
    descriptor = sealed_sidecar.descriptor
    expected_mount_id = sealed_sidecar.mount_id
    if expected_mount_id is None:  # pragma: no cover - sealed sidecar invariant
        raise JointRiggerArtifactError("Private sidecar mount identity is missing")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    current_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        current = _snapshot_directory_tree(
            current_descriptor,
            parent_mount_id=_descriptor_mount_id(parent_descriptor),
            label="Private sidecar cleanup",
        )
        if (
            current.identity != (entry.st_dev, entry.st_ino)
            or current.mount_id != expected_mount_id
            or current.tree_sha256 != sealed_sidecar.sha256
        ):
            raise JointRiggerArtifactError(
                "Private sidecar entry changed inode before cleanup; refusing "
                "recursive deletion"
            )
    finally:
        os.close(current_descriptor)
    _remove_descriptor_entry(
        parent_descriptor,
        name,
        expected_identity=(entry.st_dev, entry.st_ino),
        source_descriptor=descriptor,
        expected_mount_id=expected_mount_id,
        label=f"sealed private sidecar {sidecar_path}",
    )


def _cleanup_private_backend_resources(
    private_artifacts: _PrivateBackendArtifacts | None,
    *,
    sealed_artifacts: tuple[_SealedPrivateArtifact, ...],
    target_snapshots: tuple[_TargetEntrySnapshot, ...],
) -> list[Exception]:
    """Close every held fd and remove private names through original parents."""

    errors: list[Exception] = []
    sealed_sidecar = next(
        (artifact for artifact in sealed_artifacts if artifact.label == "sidecar_path"),
        None,
    )
    if private_artifacts is not None:
        try:
            _cleanup_private_sidecar(private_artifacts, sealed_sidecar)
        except Exception as exc:
            errors.append(exc)

    if private_artifacts is not None:
        entries = (
            private_artifacts.output,
            private_artifacts.diagnostics,
            private_artifacts.result,
        )
        errors.extend(
            _cleanup_private_staging_entries(
                entries,
                sealed_artifacts=sealed_artifacts,
            )
        )

    for artifact in sealed_artifacts:
        try:
            os.close(artifact.descriptor)
        except Exception as exc:
            errors.append(exc)

    errors.extend(
        _close_captured_target_states(
            [snapshot.initial_target_state for snapshot in target_snapshots]
        )
    )
    snapshot_descriptors = {snapshot.parent_descriptor for snapshot in target_snapshots}
    snapshot_descriptors.update(
        snapshot.directory_tree.descriptor
        for snapshot in target_snapshots
        if snapshot.directory_tree is not None
    )
    errors.extend(_close_descriptors(snapshot_descriptors))
    return errors


def _validate_sidecar_configuration(
    *,
    input_path: Path,
    output_path: Path,
    configured_sidecar: Path | None,
) -> Path | None:
    is_usdz_to_raw = (
        input_path.suffix.lower() == ".usdz"
        and output_path.suffix.lower() in _RAW_USD_EXTENSIONS
    )
    if is_usdz_to_raw:
        expected_sidecar = output_path.parent / f"{output_path.stem}_assets"
        if configured_sidecar != expected_sidecar:
            raise JointRiggerBackendIncompatibleError(
                "USDZ-to-raw Stage 2 authoring requires sidecar_path exactly "
                f"{expected_sidecar}"
            )
        return expected_sidecar
    if configured_sidecar is not None:
        raise JointRiggerBackendIncompatibleError(
            "A facade sidecar target is supported only for USDZ-to-raw authoring"
        )
    return None


def _validate_staged_sidecar_configuration(
    *,
    input_path: Path,
    artifact_targets: JointRiggerArtifactTargets,
) -> tuple[Path, Path, Path] | None:
    """Validate physical writes against the declared publication layout."""

    publication_output_path = artifact_targets.publication_output_path
    if publication_output_path is None:  # pragma: no cover - model invariant
        raise JointRiggerBackendIncompatibleError(
            "Facade omitted the publication output path"
        )
    _validate_physical_publication_root_parent(artifact_targets)
    is_usdz_to_raw = (
        input_path.suffix.lower() == ".usdz"
        and publication_output_path.suffix.lower() in _RAW_USD_EXTENSIONS
    )
    if not is_usdz_to_raw:
        if (
            artifact_targets.sidecar_path is not None
            or artifact_targets.publication_sidecar_path is not None
        ):
            raise JointRiggerBackendIncompatibleError(
                "A staged sidecar is supported only for USDZ-to-raw authoring"
            )
        return None
    if artifact_targets.output_path.suffix.lower() not in _RAW_USD_EXTENSIONS:
        raise JointRiggerBackendIncompatibleError(
            "USDZ-to-raw Stage 2 authoring requires a raw physical output path"
        )
    direct_sidecar = artifact_targets.output_path.with_name(
        f"{artifact_targets.output_path.stem}_assets"
    )
    staged_sidecar = artifact_targets.sidecar_path
    if staged_sidecar is None:
        raise JointRiggerBackendIncompatibleError(
            "Facade did not provide a staged sidecar destination"
        )
    publication_sidecar = artifact_targets.publication_sidecar_path
    if publication_sidecar is None:
        raise JointRiggerBackendIncompatibleError(
            "USDZ-to-raw Stage 2 authoring requires a publication sidecar path"
        )
    expected_publication_sidecar = publication_output_path.parent / (
        f"{publication_output_path.stem}_assets"
    )
    if publication_sidecar != expected_publication_sidecar:
        raise JointRiggerBackendIncompatibleError(
            "Publication sidecar must exactly match the published output basename: "
            f"{expected_publication_sidecar}"
        )
    return direct_sidecar, staged_sidecar, publication_sidecar


def _rebase_staged_sidecar_paths(
    *,
    output_path: Path,
    staged_sidecar_name: str,
    final_sidecar_name: str,
) -> None:
    """Rewrite root-layer asset paths from a staged to final sidecar basename."""

    if staged_sidecar_name == final_sidecar_name:
        return
    from pxr import Sdf, UsdUtils

    layer = Sdf.Layer.FindOrOpen(str(output_path))
    if layer is None:
        raise JointRiggerArtifactError(
            f"Could not open staged USD layer for sidecar rebasing: {output_path}"
        )

    def rebase_asset_path(asset_path: str) -> str:
        return _rebase_sidecar_asset_path(
            asset_path,
            staged_sidecar_name=staged_sidecar_name,
            final_sidecar_name=final_sidecar_name,
        )

    UsdUtils.ModifyAssetPaths(
        layer,
        rebase_asset_path,
        keepEmptyPathsInArrays=True,
    )
    stale_dependencies: list[str] = []

    def collect_stale_asset_path(asset_path: str) -> str:
        if rebase_asset_path(asset_path) != asset_path:
            stale_dependencies.append(asset_path)
        return asset_path

    UsdUtils.ModifyAssetPaths(
        layer,
        collect_stale_asset_path,
        keepEmptyPathsInArrays=True,
    )
    if stale_dependencies:
        raise JointRiggerArtifactError(
            "Staged USD still references the temporary sidecar basename: "
            + ", ".join(sorted(stale_dependencies))
        )
    if not layer.Save():
        raise JointRiggerArtifactError(
            f"Could not save sidecar-rebased staged USD layer: {output_path}"
        )


def _rebase_sidecar_asset_path(
    asset_path: str,
    *,
    staged_sidecar_name: str,
    final_sidecar_name: str,
) -> str:
    for prefix in (staged_sidecar_name, f"./{staged_sidecar_name}"):
        if asset_path == prefix:
            return final_sidecar_name
        if asset_path.startswith(f"{prefix}/"):
            return f"{final_sidecar_name}/{asset_path[len(prefix) + 1 :]}"
    return asset_path


def _validate_distinct_read_paths(read_paths: list[tuple[str, Path]]) -> None:
    seen: dict[Path, str] = {}
    for label, path in read_paths:
        normalized = path.expanduser().resolve(strict=False)
        previous = seen.get(normalized)
        if previous is not None:
            raise ValueError(f"{label} must not alias {previous}: {path}")
        seen[normalized] = label


def _validate_backend_artifact_targets(
    *,
    input_path: Path,
    candidates_path: Path,
    artifact_targets: JointRiggerArtifactTargets,
    additional_read_paths: list[tuple[str, Path]] | None = None,
    captured_read_paths: list[tuple[str, Path]] | None = None,
    generated_sidecar_path: Path | None = None,
) -> None:
    """Apply one destructive-alias policy to facade and direct backend calls.

    The shared facade validates caller-facing targets, but a backend can also
    be used directly with physical staging paths. Projecting those physical
    paths into caller-shaped targets lets us reuse the shared validator without
    accepting caller-supplied publication metadata as a validation bypass.
    Existing inode checks additionally protect direct writes through hard-link
    or bind-mount aliases; facade promotion does not write through those inodes.
    """

    primary_read_paths = [
        ("input_usd_path", input_path),
        ("articulation_candidates_path", candidates_path),
    ]
    primary_read_paths.extend(additional_read_paths or [])
    _validate_distinct_read_paths(primary_read_paths)
    protected_read_paths = [
        *primary_read_paths,
        *(captured_read_paths or []),
    ]
    _validate_backend_targets_against_reads(
        artifact_targets,
        read_paths=protected_read_paths,
        generated_sidecar_path=generated_sidecar_path,
    )

    dependencies = _source_usd_dependency_read_paths(input_path)
    _validate_primary_reads_against_dependencies(primary_read_paths, dependencies)
    protected_read_paths = [
        *primary_read_paths,
        *dependencies,
        *(captured_read_paths or []),
    ]
    _validate_backend_targets_against_reads(
        artifact_targets,
        read_paths=protected_read_paths,
        generated_sidecar_path=generated_sidecar_path,
    )


def _sealed_source_read_paths(
    binding: _SealedSourceBinding,
) -> list[tuple[str, Path]]:
    """Return every root/dependency path captured with the sealed descriptors."""

    read_paths = [("captured input_usd_path", binding.path)]
    for index, dependency in enumerate(binding.dependencies):
        label = f"captured input USD dependency[{index}]"
        read_paths.append((label, dependency.path))
        read_paths.extend((label, path) for path in dependency.projection_paths)
    return read_paths


def _validate_backend_targets_against_reads(
    artifact_targets: JointRiggerArtifactTargets,
    *,
    read_paths: list[tuple[str, Path]],
    generated_sidecar_path: Path | None,
) -> None:
    """Validate physical write paths against one complete backend read set."""

    physical_targets = JointRiggerArtifactTargets(
        output_path=artifact_targets.output_path,
        diagnostics_path=artifact_targets.diagnostics_path,
        result_path=artifact_targets.result_path,
    )
    read_aliases = _read_path_aliases(read_paths)
    validate_artifact_targets(physical_targets, read_paths=read_aliases)

    target_paths = _artifact_target_paths(artifact_targets)
    if (
        generated_sidecar_path is not None
        and generated_sidecar_path != artifact_targets.sidecar_path
    ):
        target_paths.append(("generated_sidecar_path", generated_sidecar_path))

    _validate_backend_target_relationships(target_paths)
    for label, path in target_paths:
        if label not in {"sidecar_path", "generated_sidecar_path"}:
            continue
        _validate_backend_sidecar_target(
            label,
            path,
            read_paths=read_aliases,
        )
    _validate_existing_physical_aliases(target_paths, read_aliases)
    _validate_physical_sidecar_containment(target_paths, read_aliases)


def _validate_primary_reads_against_dependencies(
    primary_reads: list[tuple[str, Path]],
    dependencies: list[tuple[str, Path]],
) -> None:
    """Reject configured non-root reads that alias any source dependency."""

    dependency_identities = {
        path.expanduser().resolve(strict=False) for _, path in dependencies
    }
    for label, path in primary_reads:
        if label == "input_usd_path":
            continue
        if path.expanduser().resolve(strict=False) in dependency_identities:
            raise ValueError(f"{label} must not alias an input USD dependency: {path}")


def _artifact_target_paths(
    targets: JointRiggerArtifactTargets,
) -> list[tuple[str, Path]]:
    paths = [
        ("output_path", targets.output_path),
        ("diagnostics_path", targets.diagnostics_path),
        ("result_path", targets.result_path),
    ]
    if targets.sidecar_path is not None:
        paths.append(("sidecar_path", targets.sidecar_path))
    return paths


def _validate_backend_target_relationships(
    target_paths: list[tuple[str, Path]],
) -> None:
    """Reject aliases and nesting across every physical backend write path."""

    normalized: list[tuple[str, Path]] = []
    for label, path in target_paths:
        current = path.expanduser().resolve(strict=False)
        for previous_label, previous in normalized:
            if current == previous:
                raise ValueError(f"{label} must not alias {previous_label}: {path}")
            if current.is_relative_to(previous) or previous.is_relative_to(current):
                raise ValueError(
                    "Nested Joint Rigger artifact targets are not supported: "
                    f"{label}={path} overlaps {previous_label}"
                )
        normalized.append((label, current))


def _validate_backend_sidecar_target(
    label: str,
    sidecar_path: Path,
    *,
    read_paths: list[tuple[str, Path]],
) -> None:
    """Validate a recursively replaced sidecar without assuming its parent."""

    normalized = sidecar_path.expanduser().resolve(strict=False)
    if sidecar_path.exists() or sidecar_path.is_symlink():
        if sidecar_path.is_symlink() or not sidecar_path.is_dir():
            raise ValueError(
                f"Existing {label} must be a non-symlink directory: {sidecar_path}"
            )
        _reject_sidecar_mount_points(label, sidecar_path)
    for read_label, read_path in read_paths:
        if read_path == normalized:
            raise ValueError(f"{label} must not alias {read_label}: {sidecar_path}")
        if read_path.is_relative_to(normalized):
            raise ValueError(f"{read_label} must not be inside {label}: {read_path}")


def _reject_sidecar_mount_points(label: str, sidecar_path: Path) -> None:
    """Reject root and nested mounts through fdinfo-backed traversal."""

    parent = sidecar_path.parent.expanduser().resolve(strict=True)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    parent_descriptor = os.open(parent, flags)
    try:
        parent_metadata = os.fstat(parent_descriptor)
        observed_parent = sidecar_path.parent.stat()
        if (parent_metadata.st_dev, parent_metadata.st_ino) != (
            observed_parent.st_dev,
            observed_parent.st_ino,
        ):
            raise ValueError(f"Existing {label} parent changed: {sidecar_path.parent}")
        root_descriptor = os.open(
            sidecar_path.name,
            flags,
            dir_fd=parent_descriptor,
        )
        try:
            root_metadata = os.fstat(root_descriptor)
            observed_root = sidecar_path.stat()
            if (root_metadata.st_dev, root_metadata.st_ino) != (
                observed_root.st_dev,
                observed_root.st_ino,
            ):
                raise ValueError(f"Existing {label} changed: {sidecar_path}")
            parent_mount_id = _descriptor_mount_id(parent_descriptor)
            root_mount_id = _descriptor_mount_id(root_descriptor)
            if root_mount_id != parent_mount_id:
                raise ValueError(
                    f"Existing {label} root is a mount point: {sidecar_path}"
                )
            try:
                _require_directory_tree_mount_id(
                    root_descriptor,
                    expected_mount_id=root_mount_id,
                    label=f"Existing {label}",
                )
            except JointRiggerArtifactError as exc:
                raise ValueError(str(exc)) from exc
        finally:
            os.close(root_descriptor)
    finally:
        os.close(parent_descriptor)


def _physical_directory_identity(path: Path) -> tuple[int, int] | None:
    """Return one existing directory's physical identity through mount aliases."""

    try:
        metadata = path.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(metadata.st_mode):
        return None
    return metadata.st_dev, metadata.st_ino


def _physical_ancestor_directory_identities(path: Path) -> set[tuple[int, int]]:
    """Return physical directory identities from a locator up to its root."""

    absolute = Path(os.path.abspath(path.expanduser()))
    current = absolute if _physical_directory_identity(absolute) else absolute.parent
    identities: set[tuple[int, int]] = set()
    while True:
        identity = _physical_directory_identity(current)
        if identity is not None:
            identities.add(identity)
        if current == current.parent:
            break
        current = current.parent
    return identities


def _validate_physical_sidecar_containment(
    target_paths: list[tuple[str, Path]],
    read_paths: list[tuple[str, Path]],
) -> None:
    """Reject recursive sidecar overlap through bind-mount path aliases."""

    sidecars = [
        (label, path, _physical_directory_identity(path))
        for label, path in target_paths
        if label in {"sidecar_path", "generated_sidecar_path"}
    ]
    for sidecar_label, sidecar_path, sidecar_identity in sidecars:
        if sidecar_identity is None:
            continue
        for target_label, target_path in target_paths:
            if target_label == sidecar_label and target_path == sidecar_path:
                continue
            if sidecar_identity in _physical_ancestor_directory_identities(target_path):
                raise ValueError(
                    f"{target_label} is physically inside {sidecar_label}: "
                    f"{target_path}"
                )
        for read_label, read_path in read_paths:
            if sidecar_identity in _physical_ancestor_directory_identities(read_path):
                raise ValueError(
                    f"{read_label} is physically inside {sidecar_label}: {read_path}"
                )


def _validate_existing_physical_aliases(
    target_paths: list[tuple[str, Path]],
    read_paths: list[tuple[str, Path]],
) -> None:
    """Reject existing hard links and physical-parent aliases before writes."""

    target_identities: dict[tuple[int, int], tuple[str, Path]] = {}
    target_locators: dict[tuple[int, int, str], tuple[str, Path]] = {}
    for label, path in target_paths:
        identity = _existing_path_identity(path)
        if identity is not None:
            previous = target_identities.get(identity)
            if previous is not None:
                previous_label, _ = previous
                raise ValueError(f"{label} must not alias {previous_label}: {path}")
            target_identities[identity] = (label, path)
        locator = _physical_locator_identity(path)
        if locator is not None:
            previous = target_locators.get(locator)
            if previous is not None:
                previous_label, _ = previous
                raise ValueError(f"{label} must not alias {previous_label}: {path}")
            target_locators[locator] = (label, path)

    for read_label, read_path in read_paths:
        identity = _existing_path_identity(read_path)
        if identity is not None and identity in target_identities:
            target_label, target_path = target_identities[identity]
            raise ValueError(
                f"{target_label} must not alias {read_label}: {target_path}"
            )
        locator = _physical_locator_identity(read_path)
        if locator is not None and locator in target_locators:
            target_label, target_path = target_locators[locator]
            raise ValueError(
                f"{target_label} must not alias {read_label}: {target_path}"
            )


def _existing_path_identity(path: Path) -> tuple[int, int] | None:
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return None
    return metadata.st_dev, metadata.st_ino


def _physical_locator_identity(path: Path) -> tuple[int, int, str] | None:
    try:
        metadata = path.parent.stat()
    except FileNotFoundError:
        return None
    return metadata.st_dev, metadata.st_ino, path.name


def _read_path_aliases(
    read_paths: list[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    """Return every symlink-resolution hop for destructive-target checks."""

    aliases: list[tuple[str, Path]] = []
    for label, path in read_paths:
        for index, alias in enumerate(_read_path_alias_chain(path)):
            alias_label = label if index == 0 else f"{label} resolved path"
            aliases.append((alias_label, alias))
    return aliases


def _read_path_alias_chain(path: Path) -> tuple[Path, ...]:
    """Preserve one full locator at each symlink hop in ``path``.

    Tracking only the authored locator and its final referent misses a path that
    enters a recursively replaced sidecar through one symlink and leaves it
    through another.  The intermediate locator is destructive input overlap
    even though both endpoints live outside the sidecar.
    """

    absolute = Path(os.path.abspath(path.expanduser()))
    aliases = [absolute]
    current = Path(absolute.anchor)
    remaining = list(absolute.parts[1:])
    visited_symlinks: set[Path] = set()
    while remaining:
        current /= remaining.pop(0)
        if not current.is_symlink():
            continue
        if current in visited_symlinks:
            raise ValueError(f"App read path contains a symlink cycle: {current}")
        visited_symlinks.add(current)
        target = current.readlink()
        if not target.is_absolute():
            target = current.parent / target
        rewritten = Path(os.path.abspath(target.joinpath(*remaining)))
        aliases.append(rewritten)
        current = Path(rewritten.anchor)
        remaining = list(rewritten.parts[1:])
    return tuple(aliases)


def _source_usd_dependency_read_paths(input_path: Path) -> list[tuple[str, Path]]:
    """Return every local layer and authored asset read beyond the USD root."""

    try:
        discovered = local_usd_dependency_paths(
            input_path,
            include_lexical_aliases=True,
        )
    except JointRiggerContractError as exc:
        raise JointRiggerArtifactError(
            f"Could not inspect input USD dependency closure: {exc}"
        ) from exc
    root_aliases = set(_read_path_alias_chain(input_path))
    dependencies = [path for path in discovered if path not in root_aliases]
    return [
        (f"source_asset_dependency[{index}:{path.name}]", path)
        for index, path in enumerate(dependencies)
    ]


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        _, encoded = _read_stable_regular_file(
            path,
            label=label,
            capture_payload=True,
            max_bytes=_MAX_BOUND_CANDIDATE_BYTES,
        )
        assert encoded is not None
        payload = json.loads(encoded.decode("utf-8"))
    except (
        JointRiggerArtifactError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise JointRiggerArtifactError(f"Could not read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise JointRiggerArtifactError(f"{label} must be a JSON object")
    return cast(dict[str, Any], payload)


def _write_contract_report(path: Path, value: Any) -> None:
    """Create one private report without following or truncating an entry."""

    payload = canonical_json(value).encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - regular-file invariant
                raise OSError("Could not write private Joint Rigger report")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        owned_descriptor = descriptor
        descriptor = -1
        os.close(owned_descriptor)


def _create_sealed_candidate_binding(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> _SealedDependencyBinding:
    """Bind one bounded candidate document before hashing or parsing it."""

    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise FileNotFoundError(
            f"Stage 2 candidate document must be a regular file: {path}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise FileNotFoundError(
            f"Stage 2 candidate document must be a regular file: {path}"
        )
    if metadata.st_size > _MAX_BOUND_CANDIDATE_BYTES:
        raise JointRiggerArtifactError(
            "Stage 2 candidate document exceeds the "
            f"{_MAX_BOUND_CANDIDATE_BYTES}-byte snapshot limit"
        )

    binding: _SealedDependencyBinding | None = None
    try:
        binding = _create_sealed_file_binding(path)
        if os.fstat(binding.descriptor).st_size > _MAX_BOUND_CANDIDATE_BYTES:
            raise JointRiggerArtifactError(
                "Stage 2 candidate document exceeds the "
                f"{_MAX_BOUND_CANDIDATE_BYTES}-byte snapshot limit"
            )
        if expected_sha256 is not None and binding.sha256 != expected_sha256:
            raise JointRiggerArtifactError(
                "Stage 2 candidate document changed during authoring"
            )
        _require_sealed_candidate_binding(binding)
        return binding
    except BaseException as binding_error:
        if binding is not None:
            cleanup_errors = _close_sealed_candidate_binding(binding)
            _add_cleanup_error_note(
                binding_error,
                label="Bound candidate cleanup also failed",
                errors=cleanup_errors,
            )
        raise


def _require_sealed_candidate_binding(
    binding: _SealedDependencyBinding | None,
) -> None:
    """Revalidate one immutable candidate snapshot at each trust boundary."""

    if binding is None:
        raise JointRiggerArtifactError("Bound candidate snapshot is missing")
    _require_sealed_source_binding(
        _SealedSourceBinding(
            path=binding.path,
            descriptor=binding.descriptor,
            sha256=binding.sha256,
        )
    )


def _require_candidate_path_authority(
    configured_path: Path,
    binding: _SealedDependencyBinding,
) -> None:
    """Require the configured locator to retain the bound physical authority."""

    try:
        current = configured_path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise JointRiggerArtifactError(
            "Stage 2 candidate document path changed during authoring"
        ) from exc
    if current != binding.path:
        raise JointRiggerArtifactError(
            "Stage 2 candidate document path changed during authoring: "
            f"bound={binding.path}, current={current}"
        )
    _require_candidate_file_sha256(
        configured_path,
        binding.sha256,
        label="Stage 2 candidate document",
    )


def _close_sealed_candidate_binding(
    binding: _SealedDependencyBinding,
) -> list[Exception]:
    """Close one retained candidate descriptor exactly once."""

    return _close_descriptors([binding.descriptor])


def _write_sealed_candidate_snapshot(
    binding: _SealedDependencyBinding,
    snapshot_path: Path,
) -> None:
    """Copy exact sealed candidate bytes into a private parser input."""

    _require_sealed_candidate_binding(binding)
    before = os.fstat(binding.descriptor)
    if before.st_size > _MAX_BOUND_CANDIDATE_BYTES:
        raise JointRiggerArtifactError(
            "Stage 2 candidate document exceeds the "
            f"{_MAX_BOUND_CANDIDATE_BYTES}-byte snapshot limit"
        )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    output_descriptor = os.open(snapshot_path, flags, 0o600)
    digest = hashlib.sha256()
    try:
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                binding.descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise JointRiggerArtifactError(
                    "Bound Stage 2 candidate snapshot changed during copy"
                )
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(output_descriptor, view)
                if written <= 0:  # pragma: no cover - regular-file invariant
                    raise OSError("Could not write bound candidate snapshot")
                view = view[written:]
            offset += len(chunk)
        if os.pread(binding.descriptor, 1, offset):
            raise JointRiggerArtifactError(
                "Bound Stage 2 candidate snapshot grew during copy"
            )
        if digest.hexdigest() != binding.sha256:
            raise JointRiggerArtifactError(
                "Bound Stage 2 candidate snapshot changed during copy"
            )
        os.fsync(output_descriptor)
    finally:
        os.close(output_descriptor)
    _require_sealed_candidate_binding(binding)


def _write_bound_candidate_snapshot(
    *,
    source_path: Path,
    snapshot_path: Path,
    expected_sha256: str,
) -> None:
    observed_sha256, payload = _read_stable_regular_file(
        source_path,
        label="Stage 2 candidate document",
        capture_payload=True,
        max_bytes=_MAX_BOUND_CANDIDATE_BYTES,
    )
    assert payload is not None
    if observed_sha256 != expected_sha256:
        raise JointRiggerArtifactError(
            "Stage 2 candidate document changed before snapshotting"
        )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(snapshot_path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - regular-file invariant
                raise OSError("Could not write bound Stage 2 candidate snapshot")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _optional_divide(value: float | None, divisor: float) -> float | None:
    return None if value is None else value / divisor


def _file_sha256(path: Path, *, label: str) -> str:
    digest, _ = _read_stable_regular_file(
        path,
        label=label,
        capture_payload=False,
    )
    return digest


def _candidate_file_sha256(path: Path, *, label: str) -> str:
    """Hash one Stage 2 document only after enforcing its I/O bound."""

    digest, _ = _read_stable_regular_file(
        path,
        label=label,
        capture_payload=False,
        max_bytes=_MAX_BOUND_CANDIDATE_BYTES,
    )
    return digest


def _candidate_readiness_sha256(
    candidate_readiness: Mapping[str, Any] | None,
) -> str | None:
    """Return the exact candidate identity bound into an adapter readiness report."""

    if candidate_readiness is None:
        return None
    value = candidate_readiness.get(_CANDIDATE_READINESS_SHA256_FIELD)
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise JointRiggerBackendIncompatibleError(
            "Candidate readiness contains an invalid articulation candidate SHA-256"
        )
    return value


def _read_stable_regular_file(
    path: Path,
    *,
    label: str,
    capture_payload: bool,
    max_bytes: int | None = None,
) -> tuple[str, bytes | None]:
    """Read one unchanged regular inode without following or blocking on races."""

    def state(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    try:
        expected = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise FileNotFoundError(f"{label} must be a regular file: {path}") from exc
    if not stat.S_ISREG(expected.st_mode):
        raise FileNotFoundError(f"{label} must be a regular file: {path}")
    if max_bytes is not None and expected.st_size > max_bytes:
        raise JointRiggerArtifactError(
            f"{label} exceeds the {max_bytes}-byte snapshot limit"
        )
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FileNotFoundError(f"{label} must be a regular file: {path}") from exc
    digest = hashlib.sha256()
    payload = bytearray() if capture_payload else None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or state(opened) != state(expected):
            raise JointRiggerArtifactError(f"{label} changed before it was opened")
        offset = 0
        while offset < opened.st_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, opened.st_size - offset),
                offset,
            )
            if not chunk:
                raise JointRiggerArtifactError(f"{label} changed while it was read")
            digest.update(chunk)
            if payload is not None:
                payload.extend(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, offset):
            raise JointRiggerArtifactError(f"{label} grew while it was read")
        after = os.fstat(descriptor)
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise JointRiggerArtifactError(
                f"{label} changed while it was read"
            ) from exc
        if state(after) != state(opened) or state(current) != state(opened):
            raise JointRiggerArtifactError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), bytes(payload) if payload is not None else None


def _require_candidate_file_sha256(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> None:
    if _candidate_file_sha256(path, label=label) != expected_sha256:
        raise JointRiggerArtifactError(f"{label} changed during authoring")


def _require_source_identity(path: Path, expected: ArtifactIdentityV1) -> None:
    current = _identify_backend_artifact(
        path,
        uri=expected.uri,
        label="input USD dependency closure",
    )
    if current != expected:
        raise JointRiggerArtifactError("Input USD or its dependency closure changed")


def _identify_backend_artifact(
    path: Path,
    *,
    uri: str,
    label: str,
) -> ArtifactIdentityV1:
    try:
        return identify_usd_artifact(path, uri=uri)
    except JointRiggerContractError as exc:
        raise JointRiggerArtifactError(f"Could not identify {label}: {exc}") from exc


def build_stage2_articulation_contract_input(
    *,
    input_usd_path: str | Path,
    articulation_candidates_path: str | Path,
    predictions_path: str | Path,
    expected_articulation_candidates_sha256: str | None = None,
    allow_ready_subset: bool = False,
) -> JointRiggerInputV1 | JointRiggerInputV2:
    """Project exact Stage 1/Stage 2 evidence into the owned topology request."""

    from joint_agent.functions.articulation_contract_stage2 import (
        build_articulation_contract_from_stage2,
    )
    from joint_agent.functions.joint_rigger_contract_bridge import (
        build_canonical_joint_rigger_input_from_contract,
    )

    input_path = Path(input_usd_path)
    contract = build_articulation_contract_from_stage2(
        input_usd_path=input_path,
        articulation_candidates_path=articulation_candidates_path,
        predictions_path=predictions_path,
        expected_articulation_candidates_sha256=(
            expected_articulation_candidates_sha256
        ),
        allow_ready_subset=allow_ready_subset,
    )
    source_asset = identify_usd_artifact(input_path, uri=str(input_path))
    return cast(
        JointRiggerInputV1 | JointRiggerInputV2,
        build_canonical_joint_rigger_input_from_contract(
            contract,
            source_asset=source_asset,
        ),
    )


def _validate_final_articulation_contract_output(
    output_path: Path,
    *,
    request: JointRiggerInputV1,
    diagnostics: JointRiggerDiagnosticsV1,
) -> None:
    """Reopen and validate the exact staged graph against the outer request.

    The output is expected to have different bytes and a different URI from the
    source asset after authoring.  Graph validation therefore deliberately uses
    only the request's rigid-link, joint-topology, and articulation-root plans;
    it never compares ``request.source_asset`` with the authored output.
    """

    try:
        from pxr import Usd
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise JointRiggerContractError(
            "runtime_dependency_unavailable",
            f"OpenUSD pxr bindings are required for final graph validation: {exc}",
        ) from exc

    try:
        stage = Usd.Stage.Open(str(output_path))
    except Exception as exc:
        raise JointRiggerArtifactError(
            f"Could not reopen final staged Joint Rigger output: {output_path}"
        ) from exc
    if stage is None:
        raise JointRiggerArtifactError(
            f"Could not reopen final staged Joint Rigger output: {output_path}"
        )
    topology_plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=request.plan.joints,
    )
    try:
        try:
            if isinstance(request, JointRiggerInputV2):
                validate_authored_rigid_links(stage, request)
                _validate_v2_articulation_roots(stage, request)
            validate_authored_joint_topology(
                stage,
                topology_plan,
                diagnostics,
            )
        except JointRiggerContractError as exc:
            raise JointRiggerArtifactError(
                "Final staged Joint Rigger graph validation failed: "
                f"{exc.code}: {exc.detail}"
            ) from exc
    finally:
        del stage


@dataclass(frozen=True)
class Stage2ArticulationContractBackend:
    """Package adapter around exact contract-derived shared topology authoring."""

    input_usd_path: Path
    articulation_candidates_path: Path
    predictions_path: Path
    candidate_readiness: Mapping[str, Any] | None = None
    allow_ready_subset: bool = False

    name: ClassVar[str] = "owned_topology"
    backend_name: ClassVar[str] = "owned_topology"
    backend_version: ClassVar[str] = _CONTRACT_TOPOLOGY_BACKEND_VERSION
    supports_joint_rigger_input_v2: ClassVar[bool] = True
    supports_aggregate_rigid_links: ClassVar[bool] = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_usd_path", Path(self.input_usd_path))
        object.__setattr__(
            self,
            "articulation_candidates_path",
            Path(self.articulation_candidates_path),
        )
        object.__setattr__(self, "predictions_path", Path(self.predictions_path))
        if self.candidate_readiness is not None:
            object.__setattr__(
                self,
                "candidate_readiness",
                MappingProxyType(dict(self.candidate_readiness)),
            )

    def probe(self, request: JointRiggerInputV1) -> None:
        """Prove that the request still exactly projects the three live inputs."""

        self._validate_request_against_inputs(request)

    def _validate_request_against_inputs(
        self,
        request: JointRiggerInputV1,
    ) -> None:
        if (
            self.candidate_readiness is not None
            and self.candidate_readiness.get("status") == "blocked"
        ):
            raise JointRiggerBackendIncompatibleError(
                "The contract-derived owned topology request is readiness-blocked"
            )
        expected = build_stage2_articulation_contract_input(
            input_usd_path=self.input_usd_path,
            articulation_candidates_path=self.articulation_candidates_path,
            predictions_path=self.predictions_path,
            expected_articulation_candidates_sha256=(
                _candidate_readiness_sha256(self.candidate_readiness)
            ),
            allow_ready_subset=self.allow_ready_subset,
        )
        if canonical_sha256(expected) != canonical_sha256(request):
            raise JointRiggerBackendIncompatibleError(
                "Stage 1 predictions or Stage 2 candidates no longer match the "
                "contract-derived owned topology request"
            )

    def author(
        self,
        request: JointRiggerInputV1,
        artifact_targets: JointRiggerArtifactTargets,
    ) -> JointRiggerResultV1:
        """Author a private raw root, then publish raw USD or one USDZ package."""

        if not isinstance(request, JointRiggerInputV1):
            raise TypeError("request must be a JointRiggerInputV1")
        if (
            artifact_targets.sidecar_path is not None
            or artifact_targets.publication_sidecar_path is not None
        ):
            raise JointRiggerBackendIncompatibleError(
                "Contract-derived owned topology does not accept a publication sidecar"
            )
        publication_output = artifact_targets.publication_output_path
        if publication_output is None:  # pragma: no cover - artifact invariant
            raise JointRiggerBackendIncompatibleError(
                "Facade omitted publication_output_path"
            )
        if publication_output.suffix.lower() not in _USD_EXTENSIONS:
            raise JointRiggerBackendIncompatibleError(
                "Contract-derived owned topology output must be USD or USDZ"
            )
        if (
            self.input_usd_path.suffix.lower() == ".usdz"
            and publication_output.suffix.lower() != ".usdz"
        ):
            raise JointRiggerBackendIncompatibleError(
                "USDZ input requires USDZ output because raw output would need a "
                "separately published dependency sidecar"
            )
        _validate_backend_artifact_targets(
            input_path=self.input_usd_path,
            candidates_path=self.articulation_candidates_path,
            artifact_targets=artifact_targets,
            additional_read_paths=[("predictions_path", self.predictions_path)],
        )
        self._validate_request_against_inputs(request)

        source_binding: _SealedSourceBinding | None = None
        bound_input_dir: BoundInputDirectory | None = None
        validated_output_dir: BoundInputDirectory | None = None
        sealed_output: _SealedPrivateArtifact | None = None
        sealed_validated_output: _SealedPrivateArtifact | None = None
        primary_error: BaseException | None = None
        try:
            source_binding = _create_sealed_source_binding(
                self.input_usd_path,
                expected=request.source_asset,
            )
            bound_source, bound_input_dir, restore_paths = _materialize_bound_input(
                descriptor=source_binding.descriptor,
                expected_sha256=source_binding.sha256,
                logical_input_path=self.input_usd_path,
                dependencies=tuple(_bound_input_dependency_snapshots(source_binding)),
                editable_root=True,
            )
            _require_sealed_source_binding(source_binding)
            with tempfile.TemporaryDirectory(
                prefix="joint-rigger-contract-topology-"
            ) as temp_dir_value:
                temp_dir = Path(temp_dir_value)
                private_source = self._materialize_private_source(
                    temp_dir,
                    bound_source=bound_source,
                )
                rebound_request = request.model_copy(
                    update={
                        "source_asset": identify_usd_artifact(
                            private_source,
                            uri=str(private_source),
                        )
                    },
                    deep=True,
                )
                private_output = private_source.with_name(
                    f"{private_source.stem}.authored{private_source.suffix}"
                )
                inner_result = author_joint_topology(
                    rebound_request,
                    source_usd_path=private_source,
                    artifact_targets=JointRiggerArtifactTargets(
                        output_path=private_output,
                        diagnostics_path=temp_dir / "topology-diagnostics.json",
                        result_path=temp_dir / "topology-result.json",
                    ),
                )
                self._publish_generated_root(
                    private_output,
                    artifact_targets.output_path,
                    publication_output=publication_output,
                    projection_root=bound_input_dir.path / "filesystem",
                    restore_paths=restore_paths,
                )
                sealed_output = _seal_private_regular_file(
                    "contract-authored output",
                    artifact_targets.output_path,
                )
                if artifact_targets._created_file_binder is not None:
                    artifact_targets._bind_created_file(
                        artifact_targets.output_path,
                        os.fstat(sealed_output.descriptor),
                    )
                validated_output, validated_output_dir, _ = _materialize_bound_input(
                    descriptor=sealed_output.descriptor,
                    expected_sha256=sealed_output.sha256,
                    logical_input_path=publication_output,
                    dependencies=(
                        tuple(_bound_input_dependency_snapshots(source_binding))
                        if publication_output.suffix.lower() in _RAW_USD_EXTENSIONS
                        else ()
                    ),
                )
                sealed_validated_output = _seal_private_regular_file(
                    "contract-authored validation snapshot",
                    validated_output,
                )
                _validate_final_articulation_contract_output(
                    validated_output,
                    request=request,
                    diagnostics=inner_result.diagnostics,
                )
                _require_sealed_private_regular_file(sealed_validated_output)
                _require_sealed_private_regular_file(sealed_output)
                _require_sealed_source_binding(source_binding)
                self._validate_request_against_inputs(request)
                output_artifact = identify_usd_artifact(
                    validated_output,
                    uri=str(publication_output),
                )
                _require_sealed_private_regular_file(sealed_validated_output)
                _require_sealed_private_regular_file(sealed_output)
                if (
                    sealed_validated_output.sha256 != sealed_output.sha256
                    or output_artifact.root_sha256 != sealed_output.sha256
                ):
                    raise JointRiggerArtifactError(
                        "Validated Joint Rigger output identity does not match its "
                        "sealed staged bytes"
                    )
                _require_sealed_source_binding(source_binding)
                self._validate_request_against_inputs(request)
                result = JointRiggerResultV1(
                    schema_version=RESULT_SCHEMA_VERSION,
                    status="succeeded",
                    input_sha256=canonical_sha256(request),
                    plan_sha256=canonical_sha256(request.plan),
                    output_artifact=output_artifact,
                    diagnostics=inner_result.diagnostics,
                )
                _write_contract_report(
                    artifact_targets.diagnostics_path,
                    result.diagnostics,
                )
                _write_contract_report(artifact_targets.result_path, result)
                _require_sealed_private_regular_file(sealed_validated_output)
                _require_sealed_private_regular_file(sealed_output)
                return result
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_errors: list[Exception] = []
            if sealed_validated_output is not None:
                cleanup_errors.extend(
                    _close_descriptors([sealed_validated_output.descriptor])
                )
            if validated_output_dir is not None:
                try:
                    _remove_bound_input_directory(validated_output_dir)
                except Exception as exc:
                    cleanup_errors.append(exc)
            if bound_input_dir is not None:
                try:
                    _remove_bound_input_directory(bound_input_dir)
                except Exception as exc:
                    cleanup_errors.append(exc)
            if sealed_output is not None:
                cleanup_errors.extend(_close_descriptors([sealed_output.descriptor]))
            if source_binding is not None:
                cleanup_errors.extend(_close_source_binding(source_binding))
            if cleanup_errors:
                detail = "; ".join(str(error) for error in cleanup_errors)
                if primary_error is not None:
                    primary_error.add_note(
                        "Bound contract source cleanup also failed: " + detail
                    )
                elif len(cleanup_errors) == 1:
                    raise cleanup_errors[0]
                else:
                    raise ExceptionGroup(
                        "Bound contract source cleanup failed",
                        cleanup_errors,
                    )

    def _materialize_private_source(
        self,
        temp_dir: Path,
        *,
        bound_source: Path,
    ) -> Path:
        from joint_agent.functions.joint_rigger_adapter import (
            _extract_usdz_for_handoff,
        )

        suffix = self.input_usd_path.suffix.lower()
        if suffix == ".usdz":
            return cast(
                Path,
                _extract_usdz_for_handoff(bound_source, temp_dir),
            )
        if suffix not in _RAW_USD_EXTENSIONS:
            raise JointRiggerBackendIncompatibleError(
                "Contract-derived owned topology input must be USD or USDZ"
            )
        return bound_source

    @staticmethod
    def _publish_generated_root(
        private_output: Path,
        staged_output: Path,
        *,
        publication_output: Path,
        projection_root: Path,
        restore_paths: Mapping[Path, Path],
    ) -> None:
        from joint_agent.functions.joint_rigger_adapter import (
            _export_usd_for_handoff,
            _package_usdz_for_handoff,
        )

        if publication_output.suffix.lower() == ".usdz":
            _package_usdz_for_handoff(private_output, staged_output)
            return
        _export_usd_for_handoff(private_output, staged_output)
        _restore_bound_projection_paths(
            staged_output,
            projection_root=projection_root,
            logical_output_parent=publication_output.parent,
            restore_paths=restore_paths,
        )


def author_stage2_articulation_contract_via_core(
    *,
    input_usd_path: str | Path,
    articulation_candidates_path: str | Path,
    predictions_path: str | Path,
    artifact_targets: JointRiggerArtifactTargets,
    candidate_readiness: Mapping[str, Any] | None = None,
    allow_ready_subset: bool = False,
) -> JointRiggerResultV1:
    """Author the exact contract-derived V1/V2 request through shared semantics."""

    input_path = Path(input_usd_path)
    candidates_path = Path(articulation_candidates_path)
    predictions = Path(predictions_path)

    def build_request_and_backend() -> tuple[
        JointRiggerInputV1,
        Stage2ArticulationContractBackend,
    ]:
        _validate_backend_artifact_targets(
            input_path=input_path,
            candidates_path=candidates_path,
            artifact_targets=artifact_targets,
            additional_read_paths=[("predictions_path", predictions)],
        )
        try:
            request = build_stage2_articulation_contract_input(
                input_usd_path=input_path,
                articulation_candidates_path=candidates_path,
                predictions_path=predictions,
                expected_articulation_candidates_sha256=(
                    _candidate_readiness_sha256(candidate_readiness)
                ),
                allow_ready_subset=allow_ready_subset,
            )
        except NoReadyJointCandidatesError as exc:
            raise InitialNoReadyJointCandidatesError(str(exc)) from exc
        except JointRiggerContractError as exc:
            if exc.code in {"stage2_candidates_empty", "stage2_candidates_no_ready"}:
                raise InitialNoReadyJointCandidatesError(exc.detail) from exc
            raise
        return request, Stage2ArticulationContractBackend(
            input_usd_path=input_path,
            articulation_candidates_path=candidates_path,
            predictions_path=predictions,
            candidate_readiness=candidate_readiness,
            allow_ready_subset=allow_ready_subset,
        )

    return author_joint_rig_from_factory(
        build_request_and_backend,
        artifact_targets,
    )


__all__ = [
    "InitialNoReadyJointCandidatesError",
    "NoReadyJointCandidatesError",
    "Stage2ArticulationContractBackend",
    "Stage2CandidateEdgesBackend",
    "author_stage2_articulation_contract_via_core",
    "author_stage2_candidate_edges_via_core",
    "build_stage2_articulation_contract_input",
    "build_stage2_candidate_edges_input",
]
