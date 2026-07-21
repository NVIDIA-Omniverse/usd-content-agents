# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic owned topology and physics-schema Joint Rigger authoring."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from world_understanding.functions.physics.joint_rigger.artifacts import (
    JointRiggerArtifactTargets,
    _route_cleanup_failures,
)
from world_understanding.functions.physics.joint_rigger.author import (
    TOPOLOGY_AUTHOR_VERSION,
    _author_topology_stage,
    _bound_source_projection,
    _build_diagnostics,
    _open_stage,
    _require_set,
    _require_source_identity,
    _validate_dependency_relocation,
    _validate_format_preserving_output,
    _validate_publication_targets_against_binding,
    _validate_raw_usd_path,
    _validate_supported_request,
)
from world_understanding.functions.physics.joint_rigger.facade import (
    JointRiggerArtifactError,
    JointRiggerBackendIncompatibleError,
    author_joint_rig_from_factory,
)
from world_understanding.functions.physics.joint_rigger.models import (
    DIAGNOSTICS_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION_V2,
    RESULT_SCHEMA_VERSION,
    ArtifactIdentityV1,
    JointDiagnosticV1,
    JointMimicV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerDiagnosticsV1,
    JointRiggerInputV1,
    JointRiggerInputV2,
    JointRiggerPlanV1,
    JointRiggerPlanV2,
    JointRiggerResultV1,
    JointTopologyV1,
    canonical_json,
    canonical_sha256,
)
from world_understanding.functions.physics.joint_rigger.reference import (
    identify_usd_artifact,
)
from world_understanding.functions.physics.joint_rigger.rigid_links import (
    author_aggregate_rigid_links,
    request_has_aggregate_links,
    validate_authored_rigid_links,
)
from world_understanding.functions.physics.joint_rigger.schemas import (
    _preflight as _preflight_physics_schemas,
)
from world_understanding.functions.physics.joint_rigger.schemas import (
    _r3_raw_authorship_contract,
    author_physics_schemas,
    validate_authored_physics_schemas,
    validate_physics_plan_evidence,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    BoundInputDirectory,
    SealedSourceBinding,
    bound_input_dependency_snapshots,
    close_source_binding,
    copy_regular_file_to_new_path,
    create_sealed_source_binding,
    freeze_bound_projection_root,
    materialize_bound_input,
    remove_bound_input_directory,
    require_sealed_source_binding,
    restore_bound_projection_paths,
    write_new_text_file,
)
from world_understanding.functions.physics.joint_rigger.validation import (
    JointRiggerPhysicsSchemaSnapshot,
    JointRiggerStageSnapshot,
    _preflight_topology_authoring,
    _validate_authored_preflight,
    capture_joint_rigger_physics_schema_snapshot,
    capture_joint_rigger_stage_snapshot,
    validate_joint_rigger_stage_preservation,
)

COMBINED_AUTHOR_NAME = "owned_topology_and_physics"
COMBINED_AUTHOR_VERSION = "world-understanding-joint-rig-author-v1"
type JointRiggerCombinedPlan = JointRiggerPlanV1 | JointRiggerPlanV2


@dataclass(frozen=True)
class OwnedTopologyAndPhysicsBackend:
    """Facade backend that commits topology and owned schemas as one artifact."""

    source_usd_path: Path

    name: ClassVar[str] = COMBINED_AUTHOR_NAME
    backend_name: ClassVar[str] = COMBINED_AUTHOR_NAME
    backend_version: ClassVar[str] = COMBINED_AUTHOR_VERSION
    # ``author`` repeats this read-only evidence/topology probe inside the same
    # sealed projection, then runs R3 stage checks after deterministic joint
    # prims exist. The facade may therefore skip the standalone probe.
    author_runs_probe_checks: ClassVar[bool] = True
    supports_joint_rigger_input_v2: ClassVar[bool] = True
    supports_aggregate_rigid_links: ClassVar[bool] = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_usd_path", Path(self.source_usd_path))

    def probe(self, request: JointRiggerInputV1) -> None:
        """Check plan evidence and pre-R2 topology without writing artifacts.

        R3 stage checks require the deterministic joint prims created inside
        :meth:`author`, so the standalone probe intentionally does not claim
        complete post-topology schema compatibility.
        """

        _validate_supported_request(request)
        validate_physics_plan_evidence(request.plan)
        _validate_raw_usd_path(self.source_usd_path, label="source USD")
        topology_plan = _topology_phase_plan(request.plan)
        has_aggregate_links = request_has_aggregate_links(request)
        with _bound_source_projection(
            self.source_usd_path,
            request,
            editable_root=has_aggregate_links,
        ) as (binding, bound_source, _, _):
            stage = _open_stage(bound_source, label="bound source USD")
            try:
                if isinstance(request, JointRiggerInputV2):
                    if has_aggregate_links:
                        author_aggregate_rigid_links(stage, request)
                    else:
                        validate_authored_rigid_links(stage, request)
                _preflight_topology_authoring(stage, topology_plan)
            finally:
                del stage
            require_sealed_source_binding(binding)

    def author(
        self,
        request: JointRiggerInputV1,
        artifact_targets: JointRiggerArtifactTargets,
    ) -> JointRiggerResultV1:
        """Run R2 and R3 in one facade-owned staging transaction."""

        _validate_supported_request(request)
        validate_physics_plan_evidence(request.plan)
        _validate_raw_usd_path(self.source_usd_path, label="source USD")
        publication_output = artifact_targets.publication_output_path
        if publication_output is None:  # pragma: no cover - artifact invariant
            raise JointRiggerBackendIncompatibleError(
                "Facade omitted publication_output_path"
            )
        if (
            artifact_targets.sidecar_path is not None
            or artifact_targets.publication_sidecar_path is not None
        ):
            raise JointRiggerBackendIncompatibleError(
                "Combined owned authoring currently supports raw USD roots without "
                "a publication sidecar"
            )
        _validate_raw_usd_path(
            publication_output,
            label="publication output USD",
            require_file=False,
        )
        _validate_raw_usd_path(
            artifact_targets.output_path,
            label="physical output USD",
            require_file=False,
        )
        _validate_format_preserving_output(
            source_path=self.source_usd_path,
            publication_path=publication_output,
            physical_path=artifact_targets.output_path,
        )
        topology_plan = _topology_phase_plan(request.plan)
        with _bound_source_projection(
            self.source_usd_path,
            request,
            editable_root=True,
        ) as (binding, bound_source, bound_directory, restore_paths):
            _validate_publication_targets_against_binding(
                artifact_targets,
                source_path=self.source_usd_path,
                binding=binding,
            )
            _validate_dependency_relocation(
                source_path=self.source_usd_path,
                publication_path=publication_output,
                has_external_dependencies=bool(binding.dependencies),
            )
            stage = _open_stage(bound_source, label="bound editable source USD")
            try:
                if isinstance(request, JointRiggerInputV2):
                    author_aggregate_rigid_links(stage, request)
                _preflight_topology_authoring(stage, topology_plan)
                try:
                    topology_diagnostics = _author_topology_stage(
                        stage,
                        topology_plan,
                    )
                    topology_snapshot = capture_joint_rigger_stage_snapshot(stage)
                    resolved_plan = _resolve_physics_plan(
                        request.plan,
                        topology_diagnostics,
                    )
                    physics_diagnostics = author_physics_schemas(
                        stage,
                        resolved_plan,
                        backend_name=COMBINED_AUTHOR_NAME,
                        backend_version=COMBINED_AUTHOR_VERSION,
                    )
                    combined_diagnostics = _combine_diagnostics(
                        request.plan,
                        topology_diagnostics,
                        physics_diagnostics,
                    )
                    _bind_combined_diagnostics(
                        stage,
                        request.plan,
                        combined_diagnostics,
                    )
                    authored_snapshot = capture_joint_rigger_stage_snapshot(stage)
                    validate_joint_rigger_stage_preservation(
                        topology_snapshot,
                        authored_snapshot,
                    )
                    schema_snapshot = capture_joint_rigger_physics_schema_snapshot(
                        stage
                    )
                    if not stage.GetRootLayer().Save():
                        raise JointRiggerArtifactError(
                            "OpenUSD could not save the combined authored root layer"
                        )
                except JointRiggerContractError as exc:
                    raise JointRiggerArtifactError(
                        "Combined authoring failed post-preflight validation: "
                        f"{exc.code}: {exc.detail}"
                    ) from exc
                except JointRiggerArtifactError as exc:
                    raise JointRiggerArtifactError(
                        f"Combined authoring failed post-preflight validation: {exc}"
                    ) from exc
            finally:
                del stage

            def validate_final_projection(validation_path: Path) -> None:
                verification_stage = _open_stage(
                    validation_path,
                    label="descriptor-pinned saved combined output USD",
                )
                try:
                    try:
                        _validate_saved_combined_stage(
                            verification_stage,
                            request,
                            expected_diagnostics=combined_diagnostics,
                            topology_snapshot=topology_snapshot,
                            schema_snapshot=schema_snapshot,
                        )
                    except JointRiggerContractError as exc:
                        raise JointRiggerArtifactError(
                            "Final descriptor-pinned combined validation failed: "
                            f"{exc.code}: {exc.detail}"
                        ) from exc
                    except JointRiggerArtifactError as exc:
                        raise JointRiggerArtifactError(
                            f"Final descriptor-pinned combined validation failed: {exc}"
                        ) from exc
                finally:
                    del verification_stage

            def restore_final_projection_paths(output_descriptor: int) -> None:
                restore_bound_projection_paths(
                    bound_source,
                    projection_root=bound_directory.path / "filesystem",
                    logical_output_parent=publication_output.parent,
                    restore_paths=restore_paths,
                    output_descriptor=output_descriptor,
                )

            with freeze_bound_projection_root(
                bound_source,
                validate_frozen_projection=validate_final_projection,
                prepare_before_freeze=restore_final_projection_paths,
            ) as frozen_source:
                require_sealed_source_binding(binding)
                output_artifact = identify_usd_artifact(
                    bound_source,
                    uri=str(publication_output),
                )
                _require_source_identity(self.source_usd_path, request)
                result = JointRiggerResultV1(
                    schema_version=RESULT_SCHEMA_VERSION,
                    status="succeeded",
                    input_sha256=canonical_sha256(request),
                    plan_sha256=canonical_sha256(request.plan),
                    output_artifact=output_artifact,
                    diagnostics=combined_diagnostics,
                )
                try:
                    validate_authored_joint_rig_with_physics(
                        request,
                        result,
                        output_usd_path=bound_source,
                    )
                except JointRiggerContractError as exc:
                    raise JointRiggerArtifactError(
                        "Saved combined output failed post-author validation: "
                        f"{exc.code}: {exc.detail}"
                    ) from exc
                except JointRiggerArtifactError as exc:
                    raise JointRiggerArtifactError(
                        f"Saved combined output failed post-author validation: {exc}"
                    ) from exc
                require_sealed_source_binding(binding)
                copy_regular_file_to_new_path(
                    bound_source,
                    artifact_targets.output_path,
                    label="combined authored root",
                    frozen_source=frozen_source,
                    bind_created_file=artifact_targets._bind_created_file,
                )

            write_new_text_file(
                artifact_targets.diagnostics_path,
                canonical_json(combined_diagnostics),
                label="Joint Rigger diagnostics",
                bind_created_file=artifact_targets._bind_created_file,
            )
            write_new_text_file(
                artifact_targets.result_path,
                canonical_json(result),
                label="Joint Rigger result",
                bind_created_file=artifact_targets._bind_created_file,
            )
            return result


def author_joint_rig_with_physics(
    request: JointRiggerInputV1,
    *,
    source_usd_path: str | Path,
    artifact_targets: JointRiggerArtifactTargets,
) -> JointRiggerResultV1:
    """Atomically author owned topology and evidence-backed physics schemas.

    The complete versioned plan enters one facade transaction. R2 topology is first
    authored into the staged root, semantic joint identifiers are resolved to
    those deterministic prim paths for R3 only, and the saved root is reopened
    and fully validated before its final identity and reports are derived.
    """

    if not isinstance(request, JointRiggerInputV1):
        raise TypeError("request must be a JointRiggerInputV1")
    if not isinstance(artifact_targets, JointRiggerArtifactTargets):
        raise TypeError("artifact_targets must be JointRiggerArtifactTargets")
    source_path = Path(source_usd_path)
    return author_joint_rig_from_factory(
        lambda: (request, OwnedTopologyAndPhysicsBackend(source_path)),
        artifact_targets,
    )


def _validate_saved_combined_stage(
    stage: Any,
    request: JointRiggerInputV1,
    *,
    expected_diagnostics: JointRiggerDiagnosticsV1,
    topology_snapshot: JointRiggerStageSnapshot,
    schema_snapshot: JointRiggerPhysicsSchemaSnapshot,
) -> None:
    """Validate the exact rebound staging inode before descriptor copying."""

    if isinstance(request, JointRiggerInputV2):
        validate_authored_rigid_links(stage, request)
    saved_snapshot = capture_joint_rigger_stage_snapshot(stage)
    validate_joint_rigger_stage_preservation(topology_snapshot, saved_snapshot)
    saved_schema_snapshot = capture_joint_rigger_physics_schema_snapshot(stage)
    if saved_schema_snapshot != schema_snapshot:
        raise JointRiggerArtifactError(
            "Saved combined output physics-schema snapshot does not match the "
            "authored staging stage"
        )

    topology_plan = _topology_phase_plan(request.plan)
    topology_preflight = _preflight_topology_authoring(
        stage,
        topology_plan,
        allow_existing_joint_paths=True,
    )
    topology_diagnostics = _build_diagnostics(
        topology_plan,
        topology_preflight,
    )
    resolved_plan = _resolve_physics_plan(
        request.plan,
        topology_diagnostics,
    )
    physics_diagnostics = validate_authored_physics_schemas(
        stage,
        resolved_plan,
        backend_name=COMBINED_AUTHOR_NAME,
        backend_version=COMBINED_AUTHOR_VERSION,
    )
    observed_diagnostics = _combine_diagnostics(
        request.plan,
        topology_diagnostics,
        physics_diagnostics,
    )
    if canonical_json(observed_diagnostics) != canonical_json(expected_diagnostics):
        raise JointRiggerArtifactError(
            "Saved combined output diagnostics do not match the authored staging stage"
        )
    readback_contract = _r3_raw_authorship_contract(
        stage,
        resolved_plan,
        _preflight_physics_schemas(stage, resolved_plan),
    )
    _validate_authored_preflight(
        stage,
        topology_preflight,
        diagnostics=observed_diagnostics,
        additional_allowed_applied_schemas={
            path: contract.schema_tokens for path, contract in readback_contract.items()
        },
        additional_expected_applied_schema_order={
            path: contract.schema_order for path, contract in readback_contract.items()
        },
        additional_allowed_authored_properties={
            path: contract.authored_properties
            for path, contract in readback_contract.items()
        },
        additional_expected_attribute_specs={
            path: contract.attribute_specs
            for path, contract in readback_contract.items()
        },
        additional_expected_relationship_targets=(
            {
                path: contract.relationship_targets
                for path, contract in readback_contract.items()
            }
        ),
        plan_sha256_override=canonical_sha256(request.plan),
    )
    _validate_combined_diagnostics(
        stage,
        request.plan,
        observed_diagnostics,
    )


def validate_authored_joint_rig_with_physics(
    request: JointRiggerInputV1,
    result: JointRiggerResultV1,
    *,
    output_usd_path: str | Path,
) -> None:
    """Validate one published combined result against its current USD bytes.

    The validator is read-only. It snapshots the complete root/dependency closure
    into sealed descriptors, validates their descriptor-pinned private projection,
    and independently binds the expected topology, physics schemas, and combined
    diagnostics to ``request`` and ``result``.
    """

    if not isinstance(request, JointRiggerInputV1):
        raise TypeError("request must be a JointRiggerInputV1")
    if not isinstance(result, JointRiggerResultV1):
        raise TypeError("result must be a JointRiggerResultV1")
    output_path = Path(output_usd_path)
    _validate_raw_usd_path(output_path, label="authored output USD")
    if result.status != "succeeded" or result.output_artifact is None:
        raise JointRiggerArtifactError(
            "Combined validation requires a succeeded result with output identity"
        )
    if result.input_sha256 != canonical_sha256(request):
        raise JointRiggerArtifactError(
            "Combined result input identity does not match the request"
        )
    if result.plan_sha256 != canonical_sha256(request.plan):
        raise JointRiggerArtifactError(
            "Combined result plan identity does not match the request plan"
        )

    expected_output = result.output_artifact
    with _bound_combined_validation_projection(
        output_path,
        expected=expected_output,
    ) as (binding, bound_output):

        def validate_frozen_projection(validation_path: Path) -> None:
            stage = _open_stage(
                validation_path,
                label="descriptor-pinned authored combined output USD",
            )
            try:
                if isinstance(request, JointRiggerInputV2):
                    validate_authored_rigid_links(stage, request)
                topology_plan = _topology_phase_plan(request.plan)
                topology_preflight = _preflight_topology_authoring(
                    stage,
                    topology_plan,
                    allow_existing_joint_paths=True,
                )
                topology_diagnostics = _build_diagnostics(
                    topology_plan,
                    topology_preflight,
                )
                resolved_plan = _resolve_physics_plan(
                    request.plan,
                    topology_diagnostics,
                )
                # Prove the exact R3-owned subset before deriving its applied-schema
                # and authored-property allowlists for the topology-only validator.
                physics_diagnostics = validate_authored_physics_schemas(
                    stage,
                    resolved_plan,
                    backend_name=COMBINED_AUTHOR_NAME,
                    backend_version=COMBINED_AUTHOR_VERSION,
                )
                expected_diagnostics = _combine_diagnostics(
                    request.plan,
                    topology_diagnostics,
                    physics_diagnostics,
                )
                if canonical_json(result.diagnostics) != canonical_json(
                    expected_diagnostics
                ):
                    raise JointRiggerArtifactError(
                        "Combined result diagnostics do not match the authored USD"
                    )
                readback_contract = _r3_raw_authorship_contract(
                    stage,
                    resolved_plan,
                    _preflight_physics_schemas(stage, resolved_plan),
                )
                _validate_authored_preflight(
                    stage,
                    topology_preflight,
                    diagnostics=expected_diagnostics,
                    additional_allowed_applied_schemas={
                        path: contract.schema_tokens
                        for path, contract in readback_contract.items()
                    },
                    additional_expected_applied_schema_order={
                        path: contract.schema_order
                        for path, contract in readback_contract.items()
                    },
                    additional_allowed_authored_properties={
                        path: contract.authored_properties
                        for path, contract in readback_contract.items()
                    },
                    additional_expected_attribute_specs={
                        path: contract.attribute_specs
                        for path, contract in readback_contract.items()
                    },
                    additional_expected_relationship_targets={
                        path: contract.relationship_targets
                        for path, contract in readback_contract.items()
                    },
                    plan_sha256_override=canonical_sha256(request.plan),
                )
                _validate_combined_diagnostics(
                    stage,
                    request.plan,
                    expected_diagnostics,
                )
            finally:
                del stage

        with freeze_bound_projection_root(
            bound_output,
            validate_frozen_projection=validate_frozen_projection,
        ):
            require_sealed_source_binding(binding)


@contextmanager
def _bound_combined_validation_projection(
    output_path: Path,
    *,
    expected: ArtifactIdentityV1,
) -> Iterator[tuple[SealedSourceBinding, Path]]:
    """Bind identity and readback to one descriptor-sealed USD closure."""

    binding: SealedSourceBinding | None = None
    bound_directory: BoundInputDirectory | None = None
    primary_error: BaseException | None = None
    try:
        binding = create_sealed_source_binding(output_path, expected=expected)
        bound_output, bound_directory, _ = materialize_bound_input(
            descriptor=binding.descriptor,
            expected_sha256=binding.sha256,
            logical_input_path=output_path,
            dependencies=bound_input_dependency_snapshots(binding),
            # The root is never opened while writable. The caller immediately
            # freezes it and validates through a retained-descriptor alias.
            editable_root=True,
        )
        require_sealed_source_binding(binding)
        yield binding, bound_output
        require_sealed_source_binding(binding)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[tuple[str, BaseException]] = []
        if bound_directory is not None:
            try:
                remove_bound_input_directory(bound_directory)
            except BaseException as exc:
                cleanup_errors.append(("Bound validation cleanup failed", exc))
        if binding is not None:
            try:
                cleanup_errors.extend(
                    ("Bound validation descriptor cleanup failed", error)
                    for error in close_source_binding(binding)
                )
            except BaseException as exc:
                cleanup_errors.append(
                    ("Bound validation descriptor cleanup failed", exc)
                )
        _route_cleanup_failures(
            cleanup_errors,
            primary_error=primary_error,
            label="Bound combined-validation cleanup failed",
        )


def _topology_phase_plan(plan: JointRiggerCombinedPlan) -> JointRiggerPlanV1:
    """Project the full plan onto fields owned by the topology phase."""

    return JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=tuple(
            JointPlanV1(
                topology=joint.topology,
                limit=joint.limit,
                anchor=joint.anchor,
            )
            for joint in plan.joints
        ),
    )


def _authored_path_by_joint_id(
    plan: JointRiggerCombinedPlan,
    diagnostics: JointRiggerDiagnosticsV1,
) -> dict[str, str]:
    expected = {joint.topology.joint_id for joint in plan.joints}
    observed = {item.joint_id for item in diagnostics.joint_diagnostics}
    if observed != expected:
        raise JointRiggerArtifactError(
            "Topology diagnostics do not exactly cover the combined plan: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    mapping: dict[str, str] = {}
    for diagnostic in diagnostics.joint_diagnostics:
        path = diagnostic.authored_prim_path
        if path is None:
            raise JointRiggerArtifactError(
                "Topology diagnostics omitted the authored prim path for "
                f"{diagnostic.joint_id!r}"
            )
        mapping[diagnostic.joint_id] = path
    if len(set(mapping.values())) != len(mapping):
        raise JointRiggerArtifactError(
            "Topology diagnostics mapped multiple joint ids to one authored prim"
        )
    return mapping


def _resolve_physics_plan(
    plan: JointRiggerCombinedPlan,
    topology_diagnostics: JointRiggerDiagnosticsV1,
) -> JointRiggerCombinedPlan:
    """Resolve semantic joint ids to exact R2-authored prim paths for R3."""

    authored_paths = _authored_path_by_joint_id(plan, topology_diagnostics)
    resolved_joints: list[JointPlanV1] = []
    for joint in plan.joints:
        joint_id = joint.topology.joint_id
        topology = JointTopologyV1(
            joint_id=authored_paths[joint_id],
            joint_type=joint.topology.joint_type,
            body0=joint.topology.body0,
            body1=joint.topology.body1,
            axis_stage=joint.topology.axis_stage,
            field_provenance=joint.topology.field_provenance,
        )
        mimic = joint.mimic
        resolved_mimic = None
        if mimic is not None:
            reference_path = authored_paths.get(mimic.reference_joint_id)
            if reference_path is None:  # pragma: no cover - model invariant
                raise JointRiggerArtifactError(
                    "Mimic reference is absent from the topology diagnostics: "
                    f"{mimic.reference_joint_id!r}"
                )
            resolved_mimic = JointMimicV1(
                reference_joint_id=reference_path,
                gearing=mimic.gearing,
                offset=mimic.offset,
                natural_frequency=mimic.natural_frequency,
                damping_ratio=mimic.damping_ratio,
                provenance=mimic.provenance,
            )
        resolved_joints.append(
            JointPlanV1(
                topology=topology,
                limit=joint.limit,
                anchor=joint.anchor,
                joint_friction=joint.joint_friction,
                drive=joint.drive,
                state=joint.state,
                mimic=resolved_mimic,
            )
        )
    if isinstance(plan, JointRiggerPlanV2):
        return JointRiggerPlanV2(
            schema_version=PLAN_SCHEMA_VERSION_V2,
            joints=tuple(resolved_joints),
            rigid_bodies=plan.rigid_bodies,
            articulation_roots=plan.articulation_roots,
        )
    return JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=tuple(resolved_joints),
        rigid_bodies=plan.rigid_bodies,
        articulation_root=plan.articulation_root,
    )


def _combine_diagnostics(
    plan: JointRiggerCombinedPlan,
    topology_diagnostics: JointRiggerDiagnosticsV1,
    physics_diagnostics: JointRiggerDiagnosticsV1,
) -> JointRiggerDiagnosticsV1:
    """Bind both phase diagnostics back to the original semantic plan ids."""

    authored_paths = _authored_path_by_joint_id(plan, topology_diagnostics)
    topology_by_id = {
        item.joint_id: item for item in topology_diagnostics.joint_diagnostics
    }
    physics_by_path = {
        item.joint_id: item for item in physics_diagnostics.joint_diagnostics
    }
    expected_paths = set(authored_paths.values())
    if set(physics_by_path) != expected_paths:
        raise JointRiggerArtifactError(
            "Physics diagnostics do not exactly cover the authored topology: "
            f"missing={sorted(expected_paths - set(physics_by_path))}, "
            f"extra={sorted(set(physics_by_path) - expected_paths)}"
        )

    joint_diagnostics: list[JointDiagnosticV1] = []
    for joint in plan.joints:
        joint_id = joint.topology.joint_id
        authored_path = authored_paths[joint_id]
        topology_item = topology_by_id[joint_id]
        physics_item = physics_by_path[authored_path]
        decisions = {
            item.field: item
            for item in topology_item.field_decisions
            if not _is_physics_phase_joint_field(item.field)
        }
        decisions.update(
            {
                item.field: item
                for item in physics_item.field_decisions
                if _is_physics_phase_joint_field(item.field)
            }
        )
        merged_decisions = tuple(decisions.values())
        joint_diagnostics.append(
            JointDiagnosticV1(
                joint_id=joint_id,
                field_decisions=merged_decisions,
                reason_codes=tuple(
                    sorted(
                        {
                            item.reason_code
                            for item in merged_decisions
                            if item.reason_code is not None
                        }
                    )
                ),
            )
        )

    top_level = {
        item.field: item
        for item in topology_diagnostics.field_decisions
        if item.field == "legacy_component_names"
    }
    top_level.update({item.field: item for item in physics_diagnostics.field_decisions})
    return JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name=COMBINED_AUTHOR_NAME,
        backend_version=COMBINED_AUTHOR_VERSION,
        field_decisions=tuple(top_level.values()),
        joint_diagnostics=tuple(joint_diagnostics),
        errors=tuple(
            sorted(set(topology_diagnostics.errors) | set(physics_diagnostics.errors))
        ),
        warnings=tuple(
            sorted(
                set(topology_diagnostics.warnings) | set(physics_diagnostics.warnings)
            )
        ),
    )


def _is_physics_phase_joint_field(field: str) -> bool:
    """Return whether R3, rather than topology authoring, owns a decision."""

    return any(
        field == prefix or field.startswith(f"{prefix}.")
        for prefix in ("state", "joint_friction", "drive", "mimic")
    )


def _field_decisions_payload(diagnostic: JointDiagnosticV1) -> str:
    return json.dumps(
        [
            item.model_dump(mode="json", exclude_none=True)
            for item in diagnostic.field_decisions
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _bind_combined_diagnostics(
    stage: Any,
    plan: JointRiggerCombinedPlan,
    diagnostics: JointRiggerDiagnosticsV1,
) -> None:
    """Bind semantic diagnostics and the full-plan identity to authored joints.

    Topology authoring initially records the topology-only plan hash. A combined
    artifact replaces it with the original full-plan hash, so topology readback
    must pass that same identity through ``plan_sha256_override``.
    """

    diagnostics_by_id = {item.joint_id: item for item in diagnostics.joint_diagnostics}
    expected_ids = {joint.topology.joint_id for joint in plan.joints}
    if set(diagnostics_by_id) != expected_ids:
        raise JointRiggerArtifactError(
            "Combined diagnostics do not exactly cover the original plan"
        )
    plan_sha256 = canonical_sha256(plan)
    for joint_id in sorted(expected_ids):
        diagnostic = diagnostics_by_id[joint_id]
        path = diagnostic.authored_prim_path
        if path is None:  # pragma: no cover - combined diagnostics invariant
            raise JointRiggerArtifactError(
                f"Combined diagnostics omitted the authored path for {joint_id!r}"
            )
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise JointRiggerArtifactError(
                f"Combined diagnostics identify a missing authored joint: {path}"
            )
        _require_set(
            prim.SetCustomDataByKey("jointRigger:jointId", joint_id),
            f"{path} combined joint id customData",
        )
        _require_set(
            prim.SetCustomDataByKey("jointRigger:planSha256", plan_sha256),
            f"{path} combined plan identity customData",
        )
        _require_set(
            prim.SetCustomDataByKey(
                "jointRigger:fieldDecisions",
                _field_decisions_payload(diagnostic),
            ),
            f"{path} combined field decisions customData",
        )


def _validate_combined_diagnostics(
    stage: Any,
    plan: JointRiggerCombinedPlan,
    diagnostics: JointRiggerDiagnosticsV1,
) -> None:
    if (
        diagnostics.backend_name != COMBINED_AUTHOR_NAME
        or diagnostics.backend_version != COMBINED_AUTHOR_VERSION
    ):
        raise JointRiggerArtifactError(
            "Combined diagnostics carry the wrong backend identity"
        )
    diagnostics_by_id = {item.joint_id: item for item in diagnostics.joint_diagnostics}
    expected_ids = {joint.topology.joint_id for joint in plan.joints}
    if set(diagnostics_by_id) != expected_ids:
        raise JointRiggerArtifactError(
            "Combined diagnostics do not exactly cover the saved plan"
        )
    plan_sha256 = canonical_sha256(plan)
    for joint_id in sorted(expected_ids):
        diagnostic = diagnostics_by_id[joint_id]
        path = diagnostic.authored_prim_path
        if path is None:
            raise JointRiggerArtifactError(
                f"Combined diagnostics omitted the saved path for {joint_id!r}"
            )
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise JointRiggerArtifactError(
                f"Combined diagnostics identify a missing saved joint: {path}"
            )
        if prim.GetCustomDataByKey("jointRigger:jointId") != joint_id:
            raise JointRiggerArtifactError(
                f"Saved combined joint id does not match diagnostics at {path}"
            )
        if prim.GetCustomDataByKey("jointRigger:planSha256") != plan_sha256:
            raise JointRiggerArtifactError(
                f"Saved combined plan identity does not match at {path}"
            )
        if (
            prim.GetCustomDataByKey("jointRigger:authoringVersion")
            != TOPOLOGY_AUTHOR_VERSION
        ):
            raise JointRiggerArtifactError(
                f"Saved combined topology version does not match at {path}"
            )
        if prim.GetCustomDataByKey(
            "jointRigger:fieldDecisions"
        ) != _field_decisions_payload(diagnostic):
            raise JointRiggerArtifactError(
                f"Saved combined field decisions do not match at {path}"
            )


__all__ = [
    "COMBINED_AUTHOR_NAME",
    "COMBINED_AUTHOR_VERSION",
    "OwnedTopologyAndPhysicsBackend",
    "author_joint_rig_with_physics",
    "validate_authored_joint_rig_with_physics",
]
