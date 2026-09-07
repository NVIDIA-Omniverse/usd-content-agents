# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint Agent adapter boundary plus deterministic articulation-v1 mock."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import tempfile
import zipfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol, cast
from urllib.parse import unquote, urlparse

import yaml
from pydantic import BaseModel, ConfigDict

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
)

from .models import (
    ArticulationAuthoringRequest,
    ArticulationAuthoringResult,
    ArticulationInferenceResult,
    ArticulationTerminalStatus,
    ArticulationValidationResult,
    ArticulationWorkflowRequest,
    MembershipDispositionDocument,
    Stage2CandidateDocument,
)

CancelChecker = Callable[[], bool]
_MAX_SAVED_GRAPH_PRIM_VISITS = 1_000_000
_MAX_SAVED_GRAPH_PATHS = 16_384
_PHYSICS_PROPERTY_NAMESPACE = "physics:"
_PHYSICS_API_PATH_KEYS = (
    "rigid_body_api_paths",
    "articulation_root_api_paths",
    "mass_api_paths",
    "collision_api_paths",
)
_PHYSICS_PROPERTY_STATE_KEY = "physics_property_state"


@dataclass(frozen=True)
class _OwnedCoreDiagnosticBinding:
    """Exact candidate-to-saved-joint binding recovered from owned-core evidence."""

    candidate_id: str
    joint_id: str
    prim_path: str
    body0_prim_path: str
    body1_prim_path: str
    plan_sha256: str
    authoring_version: str
    field_decisions_json: str


class _ArticulationEvidenceBindingError(ValueError):
    """Raised when adapter inputs no longer match durable evidence bindings."""


class JointAgentInferenceTerminalError(RuntimeError):
    """Typed required-reconciliation failure returned by Joint Agent."""

    def __init__(self, status: Mapping[str, Any]) -> None:
        ArticulationTerminalStatus.model_validate(status)
        self.status = copy.deepcopy(dict(status))
        super().__init__("Joint Agent required reconciliation failed terminally")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_non_articulated_structure(
    candidate_document: Stage2CandidateDocument,
    structure_metadata: Mapping[str, Any],
) -> bool:
    """Recognize a coherent provider-backed result with no articulation work."""

    if structure_metadata.get("structure_outcome") != "not_articulated":
        return False
    reasoning = structure_metadata.get("reasoning")
    evidence = structure_metadata.get("evidence")
    if candidate_document.candidates:
        raise RuntimeError(
            "Joint Agent not_articulated result conflicts with inferred candidates"
        )
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise RuntimeError("Joint Agent not_articulated result is missing reasoning")
    if not isinstance(evidence, Mapping) or evidence.get("accepted") is not True:
        raise RuntimeError(
            "Joint Agent not_articulated result is missing accepted evidence"
        )
    source_prims = evidence.get("source_prim_inventory")
    if (
        evidence.get("dof") != 0
        or evidence.get("segment_names") != []
        or not isinstance(source_prims, list)
        or not source_prims
        or any(
            not isinstance(path, str) or not path.startswith("/")
            for path in source_prims
        )
    ):
        raise RuntimeError(
            "Joint Agent not_articulated result is not a coherent zero-DOF finding"
        )
    return True


def _empty_membership_disposition_document() -> MembershipDispositionDocument:
    return MembershipDispositionDocument.model_validate(
        {
            "schema_version": "joint-agent-membership-disposition-v1",
            "summary": {
                "disposition_count": 0,
                "disposition_counts": {},
                "review_required_count": 0,
                "pending_downstream_count": 0,
            },
            "dispositions": [],
        }
    )


def _source_identity(source_path: Path) -> tuple[str, str]:
    from world_understanding.functions.physics.joint_rigger import (
        identify_usd_artifact,
    )

    try:
        identity = identify_usd_artifact(
            source_path,
            uri=source_path.resolve().as_uri(),
        )
    except Exception as exc:
        raise _ArticulationEvidenceBindingError(
            f"cannot establish composed authoring source identity: {exc}"
        ) from exc
    dependency_sha256 = identity.dependency_bundle_sha256
    if dependency_sha256 is None:
        raise _ArticulationEvidenceBindingError(
            "authoring source identity lacks a dependency bundle digest"
        )
    return identity.root_sha256, dependency_sha256


def _read_bound_bytes(
    path_value: str | Path,
    expected_sha256: str | None,
    *,
    label: str,
) -> bytes:
    if expected_sha256 is None:
        raise _ArticulationEvidenceBindingError(f"{label} digest binding is missing")
    path = Path(path_value).expanduser().resolve()
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise _ArticulationEvidenceBindingError(
            f"cannot read bound {label} at {path}: {exc}"
        ) from exc
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise _ArticulationEvidenceBindingError(f"{label} digest changed before use")
    return payload


def _require_bound_file(
    path_value: str | Path,
    expected_sha256: str | None,
    *,
    label: str,
) -> Path:
    if expected_sha256 is None:
        raise _ArticulationEvidenceBindingError(f"{label} digest binding is missing")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise _ArticulationEvidenceBindingError(f"bound {label} is missing: {path}")
    try:
        observed_sha256 = file_sha256(path)
    except OSError as exc:
        raise _ArticulationEvidenceBindingError(
            f"cannot read bound {label} at {path}: {exc}"
        ) from exc
    if observed_sha256 != expected_sha256:
        raise _ArticulationEvidenceBindingError(f"{label} digest changed before use")
    return path


def _validate_authoring_request_bindings(
    request: ArticulationAuthoringRequest,
) -> Stage2CandidateDocument:
    source_path = Path(request.source_asset).expanduser().resolve()
    candidate_path = Path(request.candidate_document_path).expanduser().resolve()
    source_sha256, dependency_sha256 = _source_identity(source_path)
    if source_sha256 != request.source_sha256:
        raise _ArticulationEvidenceBindingError(
            "authoring source digest changed before invocation"
        )
    if dependency_sha256 != request.source_dependency_bundle_sha256:
        raise _ArticulationEvidenceBindingError(
            "authoring source dependency bundle changed before invocation"
        )
    _validate_authoring_prediction_binding(request)
    candidate_document = Stage2CandidateDocument.model_validate_json(
        _read_bound_bytes(
            candidate_path,
            request.candidate_document_sha256,
            label="authoring candidate document",
        )
    )
    if candidate_document.candidate_ids != request.accepted_candidate_ids:
        raise _ArticulationEvidenceBindingError(
            "authoring candidate document does not exactly match accepted_candidate_ids"
        )
    if not all(
        candidate.is_articulation_v1_authorable
        for candidate in candidate_document.candidates
    ):
        raise _ArticulationEvidenceBindingError(
            "authoring candidate document contains non-authorable Stage 2 evidence"
        )
    _load_bound_accepted_authoring_plan(request)
    return candidate_document


def _load_bound_accepted_authoring_plan(request: ArticulationAuthoringRequest) -> Any:
    """Load one exact outer-accepted V2 body-membership plan, when selected."""

    if request.accepted_authoring_plan_path is None:
        return None
    from world_understanding.functions.physics.joint_rigger import JointRiggerInputV2

    plan = JointRiggerInputV2.model_validate_json(
        _read_bound_bytes(
            request.accepted_authoring_plan_path,
            request.accepted_authoring_plan_sha256,
            label="accepted rigid-link authoring plan",
        )
    )
    if (
        plan.source_asset.root_sha256 != request.source_sha256
        or plan.source_asset.dependency_bundle_sha256
        != request.source_dependency_bundle_sha256
        or tuple(sorted(item.topology.joint_id for item in plan.plan.joints))
        != tuple(sorted(request.accepted_candidate_ids))
    ):
        raise _ArticulationEvidenceBindingError(
            "accepted rigid-link authoring plan differs from the exact workflow request"
        )
    return plan


def _accepted_membership_artifact_paths(
    request: ArticulationAuthoringRequest,
) -> tuple[Path, Path]:
    source_suffix = Path(request.source_asset).suffix.lower()
    if source_suffix not in {".usd", ".usda", ".usdc"}:
        raise _ArticulationEvidenceBindingError(
            "accepted rigid-link authoring currently requires a raw USD source"
        )
    root = request.output_dir.resolve() / "joint_rigger"
    return (
        root / f"accepted_membership_source{source_suffix}",
        root / "accepted_membership_operation_receipt.json",
    )


def _accepted_membership_operation_paths(accepted_plan: Any) -> tuple[str, ...]:
    marker = "operation:apply_rigid_body_membership"
    paths = tuple(
        body.prim_path
        for body in accepted_plan.plan.rigid_bodies
        if marker in body.provenance.properties
    )
    if not paths or len(paths) != len(set(paths)):
        raise _ArticulationEvidenceBindingError(
            "accepted rigid-link plan lacks unique body-membership operations"
        )
    return paths


def _require_new_membership_operation_coverage(
    source_path: Path,
    accepted_plan: Any,
    operation_paths: tuple[str, ...],
) -> None:
    from pxr import Usd, UsdPhysics

    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise _ArticulationEvidenceBindingError(
            "accepted body-membership source could not be opened"
        )
    planned_body_paths = {body.prim_path for body in accepted_plan.plan.rigid_bodies}
    existing_body_paths = {
        path
        for path in planned_body_paths
        if source_stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI)
    }
    if existing_body_paths | set(operation_paths) != planned_body_paths or (
        existing_body_paths & set(operation_paths)
    ):
        raise _ArticulationEvidenceBindingError(
            "accepted body-membership operations do not exactly cover new owners"
        )


def _apply_accepted_membership_operations(
    *,
    source_path: Path,
    output_path: Path,
    operation_paths: tuple[str, ...],
) -> dict[str, Any]:
    from world_understanding.functions.physics.physics_topology import (
        apply_physics_topology_plan,
        inspect_physics_topology,
    )

    topology = inspect_physics_topology(source_path)
    return apply_physics_topology_plan(
        input_usd_path=source_path,
        output_usd_path=output_path,
        expected_source_digest=topology["source_digest"],
        mobility_intent="preserve",
        operations=[
            {"op": "ensure_rigid_body_api", "prim_path": path}
            for path in operation_paths
        ],
        invariants={
            "enabled_collider_count": topology["enabled_collider_count"],
            "reject_articulation_changes": True,
        },
    )


def _flattened_usd_sha256(path: Path) -> str:
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise _ArticulationEvidenceBindingError(
            "accepted body-membership derivative could not be opened"
        )
    flattened = stage.Flatten()
    if flattened is None:  # pragma: no cover - OpenUSD allocation guard
        raise _ArticulationEvidenceBindingError(
            "accepted body-membership derivative could not be flattened"
        )
    # Stage.Flatten appends the physical root-layer path to documentation.
    # Normalize only those generated locator lines so two otherwise identical
    # projections at private paths retain the same semantic digest.
    flattened.documentation = "\n".join(
        (
            "Generated from Composed Stage of root layer <normalized>"
            if line.startswith("Generated from Composed Stage of root layer ")
            else line
        )
        for line in flattened.documentation.splitlines()
    )
    payload = flattened.ExportToString().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_accepted_membership_derivative(
    request: ArticulationAuthoringRequest,
    accepted_plan: Any,
) -> Path:
    prepared_path, receipt_path = _accepted_membership_artifact_paths(request)
    if (
        prepared_path.is_symlink()
        or receipt_path.is_symlink()
        or not prepared_path.is_file()
        or not receipt_path.is_file()
    ):
        raise _ArticulationEvidenceBindingError(
            "accepted body-membership derivative or operation receipt is missing"
        )
    receipt = load_json(receipt_path)
    operation_paths = _accepted_membership_operation_paths(accepted_plan)
    expected = {
        "schema_version": (
            "content-agent-workflows.accepted-membership-operation-receipt.v1"
        ),
        "source_sha256": request.source_sha256,
        "source_dependency_bundle_sha256": (request.source_dependency_bundle_sha256),
        "accepted_authoring_plan_sha256": (request.accepted_authoring_plan_sha256),
        "operation": "physics.apply_topology_plan",
        "operation_paths": list(operation_paths),
        "source_mutated": False,
    }
    if not isinstance(receipt, Mapping) or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise _ArticulationEvidenceBindingError(
            "accepted body-membership operation receipt differs from the bound plan"
        )
    output = receipt.get("output")
    if (
        not isinstance(output, Mapping)
        or output.get("path") != str(prepared_path.resolve())
        or output.get("sha256") != file_sha256(prepared_path)
    ):
        raise _ArticulationEvidenceBindingError(
            "accepted body-membership derivative changed after publication"
        )
    source_sha256, dependency_sha256 = _source_identity(
        Path(request.source_asset).expanduser().resolve()
    )
    if (
        source_sha256 != request.source_sha256
        or dependency_sha256 != request.source_dependency_bundle_sha256
    ):
        raise _ArticulationEvidenceBindingError(
            "authoring source changed after accepted body-membership preparation"
        )
    source_path = Path(request.source_asset).expanduser().resolve()
    _require_new_membership_operation_coverage(
        source_path,
        accepted_plan,
        operation_paths,
    )
    with tempfile.TemporaryDirectory(
        prefix="content-agent-membership-validation-"
    ) as temporary_directory:
        trusted_path = Path(temporary_directory) / (
            f"accepted_membership_source{prepared_path.suffix.lower()}"
        )
        _apply_accepted_membership_operations(
            source_path=source_path,
            output_path=trusted_path,
            operation_paths=operation_paths,
        )
        if _flattened_usd_sha256(prepared_path) != _flattened_usd_sha256(trusted_path):
            raise _ArticulationEvidenceBindingError(
                "accepted body-membership derivative differs from the exact "
                "trusted source-plus-operation projection"
            )
    return prepared_path


def _reset_incomplete_accepted_membership_prefix(
    prepared_path: Path,
    receipt_path: Path,
) -> bool:
    """Return whether the run-owned derivative must be deterministically rebuilt."""

    for path in (prepared_path, receipt_path):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise _ArticulationEvidenceBindingError(
                "accepted body-membership artifact path is unsafe"
            )
    if prepared_path.is_file() and receipt_path.is_file():
        return False
    if prepared_path.exists() or receipt_path.exists():
        prepared_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
    return True


def _prepare_accepted_membership_derivative(
    request: ArticulationAuthoringRequest,
    accepted_plan: Any,
) -> Path:
    """Apply only accepted rigid-body membership to a source-bound derivative."""

    prepared_path, receipt_path = _accepted_membership_artifact_paths(request)
    rebuild_required = _reset_incomplete_accepted_membership_prefix(
        prepared_path,
        receipt_path,
    )
    if not rebuild_required:
        return _validate_accepted_membership_derivative(request, accepted_plan)
    # A crash can leave either write-once artifact without its paired custody
    # record. Both paths are run-owned derivatives; rebuild only that incomplete
    # prefix from the still-bound source and accepted plan.
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(request.source_asset).expanduser().resolve()
    operation_paths = _accepted_membership_operation_paths(accepted_plan)
    _require_new_membership_operation_coverage(
        source_path,
        accepted_plan,
        operation_paths,
    )
    source_bytes_sha256 = file_sha256(source_path)
    report = _apply_accepted_membership_operations(
        source_path=source_path,
        output_path=prepared_path,
        operation_paths=operation_paths,
    )
    if file_sha256(source_path) != source_bytes_sha256:
        raise _ArticulationEvidenceBindingError(
            "accepted body-membership operation mutated the source asset"
        )
    receipt = {
        "schema_version": (
            "content-agent-workflows.accepted-membership-operation-receipt.v1"
        ),
        "source_sha256": request.source_sha256,
        "source_dependency_bundle_sha256": (request.source_dependency_bundle_sha256),
        "accepted_authoring_plan_sha256": (request.accepted_authoring_plan_sha256),
        "operation": "physics.apply_topology_plan",
        "operation_paths": list(operation_paths),
        "source_mutated": False,
        "output": {
            "path": str(prepared_path.resolve()),
            "sha256": file_sha256(prepared_path),
        },
        "topology_report": report,
    }
    atomic_write_json(receipt_path, receipt)
    return _validate_accepted_membership_derivative(request, accepted_plan)


def _accepted_execution_request(
    request: ArticulationAuthoringRequest,
    accepted_plan: Any,
) -> Any:
    from world_understanding.functions.physics.joint_rigger import (
        identify_usd_artifact,
    )

    prepared_path = _validate_accepted_membership_derivative(request, accepted_plan)
    prepared_identity = identify_usd_artifact(
        prepared_path,
        uri=prepared_path.as_uri(),
    )
    # Retain the validated V2 body/link coverage in the serialized request.
    # The owned authoring facade projects V2 onto its topology-only plan and
    # verifies the already-authored rigid-link membership on this derivative;
    # it does not reapply masses, colliders, drives, or rigid-body opinions.
    return accepted_plan.model_copy(update={"source_asset": prepared_identity})


def _validate_authoring_prediction_binding(
    request: ArticulationAuthoringRequest,
) -> None:
    if request.predictions_path is not None:
        _require_bound_file(
            request.predictions_path,
            request.predictions_sha256,
            label="authoring predictions",
        )


def _require_result_matches_authoring_request(
    result: ArticulationAuthoringResult,
    request: ArticulationAuthoringRequest,
) -> None:
    _validate_authoring_prediction_binding(request)
    if result.idempotency_key != request.idempotency_key:
        raise _ArticulationEvidenceBindingError(
            "recovered authoring idempotency key does not match"
        )
    if result.source_sha256 != request.source_sha256:
        raise _ArticulationEvidenceBindingError(
            "recovered authoring source digest does not match"
        )
    if (
        result.source_dependency_bundle_sha256
        != request.source_dependency_bundle_sha256
    ):
        raise _ArticulationEvidenceBindingError(
            "recovered authoring source dependency bundle digest does not match"
        )
    if (
        result.candidate_document_path != request.candidate_document_path
        or result.candidate_document_sha256 != request.candidate_document_sha256
    ):
        raise _ArticulationEvidenceBindingError(
            "recovered authoring candidate binding does not match"
        )
    if result.authored_candidate_ids != request.accepted_candidate_ids:
        raise _ArticulationEvidenceBindingError(
            "recovered authored candidate IDs do not match"
        )
    _require_bound_file(
        result.output_asset_path,
        result.output_asset_sha256,
        label="recovered authored output",
    )
    for path_value, expected_sha256, label in (
        (
            result.diagnostics_path,
            result.diagnostics_sha256,
            "recovered diagnostics",
        ),
        (
            result.joint_rigger_result_path,
            result.joint_rigger_result_sha256,
            "recovered Joint Rigger result",
        ),
        (
            result.membership_operation_receipt_path,
            result.membership_operation_receipt_sha256,
            "recovered membership operation receipt",
        ),
    ):
        if path_value is not None:
            _read_bound_bytes(path_value, expected_sha256, label=label)


def _build_owned_core_authoring_result(
    request: ArticulationAuthoringRequest,
    *,
    output_path: Path,
    diagnostics_path: Path,
    result_path: Path,
    recovered: bool,
) -> ArticulationAuthoringResult:
    from world_understanding.functions.physics.joint_rigger import (
        JointRiggerDiagnosticsV1,
        JointRiggerResultV1,
    )

    candidate_document = _validate_authoring_request_bindings(request)
    joint_rigger_result = JointRiggerResultV1.model_validate_json(
        result_path.read_text(encoding="utf-8")
    )
    diagnostics = JointRiggerDiagnosticsV1.model_validate_json(
        diagnostics_path.read_text(encoding="utf-8")
    )
    if joint_rigger_result.status != "succeeded":
        raise ValueError(
            "owned_core result does not claim succeeded output: "
            f"{joint_rigger_result.status}"
        )
    if joint_rigger_result.diagnostics != diagnostics:
        raise ValueError("owned_core result and diagnostics artifacts disagree")
    if joint_rigger_result.output_artifact is None:
        raise ValueError("owned_core result is missing output identity")
    output_sha256 = file_sha256(output_path)
    if joint_rigger_result.output_artifact.root_sha256 != output_sha256:
        raise ValueError("owned_core output digest differs from result identity")
    if len(diagnostics.joint_diagnostics) != len(request.accepted_candidate_ids):
        raise ValueError(
            "owned_core authored joint count differs from accepted candidates"
        )
    bound_joint_request = _build_bound_owned_core_request(request)
    from world_understanding.functions.physics.joint_rigger import (
        canonical_sha256,
    )

    expected_input_sha256 = canonical_sha256(bound_joint_request)
    expected_plan_sha256 = canonical_sha256(bound_joint_request.plan)
    if joint_rigger_result.input_sha256 != expected_input_sha256:
        raise ValueError(
            "owned_core result input identity differs from the exact bound request"
        )
    if joint_rigger_result.plan_sha256 != expected_plan_sha256:
        raise ValueError(
            "owned_core result plan identity differs from the exact bound request"
        )
    expected_source_path = Path(request.source_asset)
    if request.accepted_authoring_plan_path is not None:
        accepted_plan = _load_bound_accepted_authoring_plan(request)
        expected_source_path = _validate_accepted_membership_derivative(
            request,
            accepted_plan,
        )
    expected_physics_state = _expected_saved_physics_state(
        expected_source_path,
        bound_joint_request,
    )
    saved_contract_failures = list(
        _validate_saved_owned_core_contract(
            output_path,
            joint_request=bound_joint_request,
            diagnostics=diagnostics,
            expected_physics_state=expected_physics_state,
        )
    )
    if diagnostics.backend_name == "stage2_candidate_edges":
        if request.predictions_path is not None:
            saved_contract_failures.append(
                "stage2_candidate_edges results cannot carry a predictions binding."
            )
        else:
            (
                diagnostic_bindings,
                diagnostic_joint_paths,
                diagnostic_binding_failures,
            ) = _resolve_owned_core_diagnostic_bindings(
                candidate_document,
                joint_diagnostics=diagnostics.joint_diagnostics,
                plan_sha256=expected_plan_sha256,
                expected_topology_plan=bound_joint_request.plan,
                backend_name=diagnostics.backend_name,
                backend_version=diagnostics.backend_version,
            )
            saved_contract_failures.extend(diagnostic_binding_failures)
            _, graph_failures, _ = _validate_saved_owned_core_graph(
                output_path,
                candidate_document,
                diagnostic_joint_paths=diagnostic_joint_paths,
                diagnostic_bindings=diagnostic_bindings,
                require_stage2_candidate_custom_data=True,
            )
            saved_contract_failures.extend(graph_failures)
    if saved_contract_failures:
        raise ValueError("; ".join(saved_contract_failures))
    physics_api_inventory_payload = _physics_state_payload(expected_physics_state)
    bound_request_metadata: dict[str, Any] = {
        "joint_rigger_input": bound_joint_request.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "joint_rigger_input_sha256": expected_input_sha256,
        "joint_rigger_plan_sha256": expected_plan_sha256,
        "physics_api_inventory": physics_api_inventory_payload,
        "physics_api_inventory_sha256": _canonical_sha256(
            physics_api_inventory_payload
        ),
    }
    result = ArticulationAuthoringResult(
        status="succeeded",
        idempotency_key=request.idempotency_key,
        source_sha256=request.source_sha256,
        source_dependency_bundle_sha256=(request.source_dependency_bundle_sha256),
        candidate_document_path=request.candidate_document_path,
        candidate_document_sha256=request.candidate_document_sha256,
        output_asset_path=str(output_path.resolve()),
        output_asset_sha256=output_sha256,
        authored_candidate_ids=request.accepted_candidate_ids,
        authored_joint_count=len(diagnostics.joint_diagnostics),
        diagnostics_path=str(diagnostics_path.resolve()),
        diagnostics_sha256=file_sha256(diagnostics_path),
        joint_rigger_result_path=str(result_path.resolve()),
        joint_rigger_result_sha256=file_sha256(result_path),
        membership_operation_receipt_path=(
            str(_accepted_membership_artifact_paths(request)[1].resolve())
            if request.accepted_authoring_plan_path is not None
            else None
        ),
        membership_operation_receipt_sha256=(
            file_sha256(_accepted_membership_artifact_paths(request)[1])
            if request.accepted_authoring_plan_path is not None
            else None
        ),
        backend_call_id=request.idempotency_key,
        metadata={
            "backend": "joint-agent-owned-core",
            "live_backend_invoked": True,
            "joint_rigger_status": "authored",
            "recovered_from_published_artifacts": recovered,
            "apply_masses": False,
            "apply_collision": False,
            "accepted_authoring_plan_sha256": (request.accepted_authoring_plan_sha256),
            **bound_request_metadata,
        },
    )
    _require_result_matches_authoring_request(result, request)
    return result


def _owned_core_authoring_result(
    request: ArticulationAuthoringRequest,
    *,
    output_path: Path,
    diagnostics_path: Path,
    result_path: Path,
    recovered: bool,
) -> ArticulationAuthoringResult:
    try:
        return _build_owned_core_authoring_result(
            request,
            output_path=output_path,
            diagnostics_path=diagnostics_path,
            result_path=result_path,
            recovered=recovered,
        )
    except _ArticulationEvidenceBindingError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise _ArticulationEvidenceBindingError(
            f"owned_core published authoring evidence is invalid: {exc}"
        ) from exc


def _vectors_close(
    actual: tuple[float, float, float],
    expected: tuple[float, float, float],
) -> bool:
    return all(
        math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-5)
        for left, right in zip(actual, expected, strict=True)
    )


def _anchor_positions_close(
    actual: tuple[float, float, float],
    expected: tuple[float, float, float],
) -> bool:
    for left, right in zip(actual, expected, strict=True):
        if not math.isfinite(left) or not math.isfinite(right):
            return False
        magnitude = max(abs(left), abs(right))
        tolerance = max(1e-5, 8.0 * math.ulp(magnitude))
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
            return False
    return True


def _vector3(value: Any) -> tuple[float, float, float]:
    return float(value[0]), float(value[1]), float(value[2])


def _normalized_vector(value: Any) -> tuple[float, float, float]:
    vector = (float(value[0]), float(value[1]), float(value[2]))
    length = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError("axis vector is not finite and nonzero")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def _dependency_locators(layers: Any, assets: Any) -> tuple[str, ...]:
    locators: list[str] = []
    for dependency in (*layers, *assets):
        values = [
            str(value)
            for field in ("identifier", "resolvedPath", "realPath", "path")
            if (value := getattr(dependency, field, None))
        ]
        locators.extend(values or (str(dependency),))
    return tuple(sorted(set(filter(None, locators))))


def _is_remote_dependency_locator(locator: str) -> bool:
    outer = locator.partition("[")[0]
    parsed = urlparse(outer)
    if not parsed.scheme or parsed.scheme == "file":
        return False
    return not (
        len(parsed.scheme) == 1
        and len(outer) >= 3
        and outer[1] == ":"
        and outer[2] in {"/", "\\"}
    )


def _local_dependency_path(locator: str) -> Path | None:
    outer = locator.partition("[")[0]
    parsed = urlparse(outer)
    if parsed.scheme == "file":
        path_value = unquote(parsed.path)
    elif parsed.scheme:
        return None
    else:
        path_value = outer
    path = Path(path_value)
    if not path.is_absolute():
        return None
    return path.expanduser().resolve()


def _validate_sealed_usdz_identity(
    output_path: Path,
    *,
    expected_root_sha256: str,
    expected_dependency_bundle_sha256: str | None,
) -> tuple[bool, str | None, tuple[str, ...]]:
    """Verify package-local dependency closure and its observed identity."""

    failures: list[str] = []
    observed_dependency_bundle_sha256: str | None = None
    if output_path.suffix.lower() != ".usdz" or not zipfile.is_zipfile(output_path):
        return (
            False,
            None,
            ("Published articulation output is not a valid USDZ package.",),
        )

    try:
        from pxr import UsdUtils

        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(output_path))
        if unresolved:
            failures.append(
                "Published USDZ has unresolved dependencies: "
                + ", ".join(sorted(str(item) for item in unresolved))
            )
        locators = _dependency_locators(layers, assets)
        remote = tuple(
            locator for locator in locators if _is_remote_dependency_locator(locator)
        )
        if remote:
            failures.append(
                "Published USDZ resolves external URI dependencies: "
                + ", ".join(remote)
            )
        resolved_output = output_path.expanduser().resolve()
        external_paths = tuple(
            sorted(
                {
                    str(path)
                    for locator in locators
                    if (path := _local_dependency_path(locator)) is not None
                    and path != resolved_output
                }
            )
        )
        if external_paths:
            failures.append(
                "Published USDZ dependency closure resolves outside the sealed "
                "archive: " + ", ".join(external_paths)
            )
    except Exception as exc:
        failures.append(f"Could not enumerate published USDZ dependencies: {exc}")

    try:
        from world_understanding.functions.physics.joint_rigger import (
            identify_usd_artifact,
            local_usd_dependency_paths,
        )

        resolved_output = output_path.expanduser().resolve(strict=True)
        local_dependencies = {
            dependency.expanduser().resolve(strict=True)
            for dependency in local_usd_dependency_paths(resolved_output)
        }
        external = tuple(
            sorted(
                str(dependency) for dependency in local_dependencies - {resolved_output}
            )
        )
        if external:
            failures.append(
                "Published USDZ dependency closure resolves outside the sealed "
                "archive: " + ", ".join(external)
            )
        identity = identify_usd_artifact(
            resolved_output,
            uri=resolved_output.as_uri(),
        )
        observed_dependency_bundle_sha256 = identity.dependency_bundle_sha256
        if identity.root_sha256 != expected_root_sha256:
            failures.append(
                "Joint Rigger root identity differs from the observed USDZ."
            )
        if (
            expected_dependency_bundle_sha256 is None
            or observed_dependency_bundle_sha256 != expected_dependency_bundle_sha256
        ):
            failures.append(
                "Joint Rigger dependency bundle identity differs from the "
                "observed USDZ closure."
            )
    except Exception as exc:
        failures.append(f"Could not establish published USDZ identity: {exc}")

    return not failures, observed_dependency_bundle_sha256, tuple(failures)


def _validate_self_contained_raw_usd_identity(
    output_path: Path,
    *,
    expected_root_sha256: str,
    expected_dependency_bundle_sha256: str | None,
) -> tuple[bool, str | None, tuple[str, ...]]:
    """Verify one raw USD output has exact identity and no external closure."""

    failures: list[str] = []
    observed_dependency_bundle_sha256: str | None = None
    if output_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        return False, None, ("Published output is not a raw USD layer.",)
    try:
        from world_understanding.functions.physics.joint_rigger import (
            identify_usd_artifact,
            local_usd_dependency_paths,
        )
        from world_understanding.functions.physics.joint_rigger.reference import (
            usd_dependency_inventory,
        )

        resolved_output = output_path.expanduser().resolve(strict=True)
        remote_dependencies = tuple(
            sorted(
                {
                    locator
                    for dependency in usd_dependency_inventory(resolved_output)
                    for locator in (
                        dependency.package_outer_identifier,
                        dependency.asset_identifier,
                        dependency.identifier,
                    )
                    if locator and _is_remote_dependency_locator(locator)
                }
            )
        )
        if remote_dependencies:
            failures.append(
                "Published raw USD resolves external URI dependencies: "
                + ", ".join(remote_dependencies)
            )
        external = tuple(
            sorted(
                str(path.expanduser().resolve(strict=True))
                for path in local_usd_dependency_paths(resolved_output)
                if path.expanduser().resolve(strict=True) != resolved_output
            )
        )
        if external:
            failures.append(
                "Published raw USD dependency closure is not self-contained: "
                + ", ".join(external)
            )
        identity = identify_usd_artifact(
            resolved_output,
            uri=resolved_output.as_uri(),
        )
        observed_dependency_bundle_sha256 = identity.dependency_bundle_sha256
        if identity.root_sha256 != expected_root_sha256:
            failures.append(
                "Joint Rigger root identity differs from the observed raw USD."
            )
        if (
            expected_dependency_bundle_sha256 is None
            or observed_dependency_bundle_sha256 != expected_dependency_bundle_sha256
        ):
            failures.append(
                "Joint Rigger dependency bundle identity differs from the "
                "observed raw USD closure."
            )
    except Exception as exc:
        failures.append(f"Could not establish published raw USD identity: {exc}")
    return not failures, observed_dependency_bundle_sha256, tuple(failures)


class _PhysicsStageState(NamedTuple):
    """Exact physics API path inventory plus authored physics property values."""

    api_paths: dict[str, tuple[str, ...]]
    property_state: dict[str, list[Any]]


def _canonical_physics_value(value: Any) -> Any:
    """Return a deterministic JSON-safe projection of one authored USD value."""

    from pxr import Gf, Sdf

    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        # repr round-trips exactly, so 12.0 and 99.0 stay distinguishable.
        return repr(value)
    if isinstance(value, Sdf.AssetPath):
        return ["assetPath", value.path]
    if isinstance(value, Sdf.Path):
        return ["path", str(value)]
    if isinstance(value, Gf.Quatd | Gf.Quatf | Gf.Quath):
        return [
            _canonical_physics_value(value.GetReal()),
            _canonical_physics_value(tuple(value.GetImaginary())),
        ]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    try:
        items = list(value)
    except TypeError:
        return ["repr", repr(value)]
    return [_canonical_physics_value(item) for item in items]


def _prim_physics_property_state(prim: Any) -> list[Any]:
    """Capture every authored physics-namespace property fact on one prim."""

    entries: list[Any] = []
    for attribute in prim.GetAuthoredAttributes():
        name = attribute.GetName()
        if not name.startswith(_PHYSICS_PROPERTY_NAMESPACE):
            continue
        time_samples = [
            [
                _canonical_physics_value(float(sample)),
                _canonical_physics_value(attribute.Get(sample)),
            ]
            for sample in sorted(float(time) for time in attribute.GetTimeSamples())
        ]
        entries.append(
            [
                "attribute",
                name,
                str(attribute.GetTypeName()),
                _canonical_physics_value(
                    attribute.Get() if attribute.HasAuthoredValueOpinion() else None
                ),
                time_samples,
            ]
        )
    for relationship in prim.GetAuthoredRelationships():
        name = relationship.GetName()
        if not name.startswith(_PHYSICS_PROPERTY_NAMESPACE):
            continue
        entries.append(
            [
                "relationship",
                name,
                [str(target) for target in relationship.GetTargets()],
            ]
        )
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    return entries


def _stage_physics_state(stage: Any) -> _PhysicsStageState:
    """Capture exact physics API paths plus authored physics property values."""

    from pxr import Usd, UsdPhysics

    paths_by_api: dict[str, set[str]] = {key: set() for key in _PHYSICS_API_PATH_KEYS}
    property_state: dict[str, list[Any]] = {}
    prim_visits = 0

    def inspect(prims: Any) -> None:
        nonlocal prim_visits
        for prim in prims:
            prim_visits += 1
            if prim_visits > _MAX_SAVED_GRAPH_PRIM_VISITS:
                raise ValueError(
                    "physics API inventory exceeds the fixed "
                    f"{_MAX_SAVED_GRAPH_PRIM_VISITS}-prim visit limit"
                )
            prim_path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                paths_by_api["rigid_body_api_paths"].add(prim_path)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                paths_by_api["articulation_root_api_paths"].add(prim_path)
            if prim.HasAPI(UsdPhysics.MassAPI):
                paths_by_api["mass_api_paths"].add(prim_path)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                paths_by_api["collision_api_paths"].add(prim_path)
            # Joint prims are authored by the owned core and are already validated
            # exactly against the bound topology plan, so they carry no
            # source-projected physics expectation.
            if prim.IsA(UsdPhysics.Joint):
                continue
            entries = _prim_physics_property_state(prim)
            if entries:
                if len(property_state) >= _MAX_SAVED_GRAPH_PATHS:
                    raise ValueError(
                        "authored physics property state exceeds the fixed "
                        f"{_MAX_SAVED_GRAPH_PATHS}-prim retention limit"
                    )
                property_state[prim_path] = entries

    inspect(
        Usd.PrimRange.Stage(
            stage,
            Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
        )
    )
    for prototype in stage.GetPrototypes():
        inspect(Usd.PrimRange.AllPrims(prototype))
    return _PhysicsStageState(
        api_paths={key: tuple(sorted(paths)) for key, paths in paths_by_api.items()},
        property_state=property_state,
    )


def _expected_saved_physics_state(
    source_path: Path,
    joint_request: Any,
) -> _PhysicsStageState:
    """Project source physics API paths and values through V2 member moves."""

    from pxr import Usd
    from world_understanding.functions.physics.joint_rigger import JointRiggerInputV2

    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise ValueError(f"Could not open authoring source stage: {source_path}")
    source_state = _stage_physics_state(source_stage)
    if not isinstance(joint_request, JointRiggerInputV2):
        return source_state

    member_mappings = sorted(
        (
            (member.source_prim_path, member.authored_prim_path)
            for link in joint_request.rigid_links
            for member in link.members
            if member.source_prim_path != member.authored_prim_path
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    def projected_path(source_prim_path: str) -> str:
        for source_root, authored_root in member_mappings:
            if source_prim_path == source_root:
                return authored_root
            if source_prim_path.startswith(f"{source_root}/"):
                return authored_root + source_prim_path[len(source_root) :]
        return source_prim_path

    projected_api_paths = {
        key: tuple(sorted(projected_path(path) for path in paths))
        for key, paths in source_state.api_paths.items()
    }
    projected_api_paths["articulation_root_api_paths"] = tuple(
        sorted(
            set(projected_api_paths["articulation_root_api_paths"])
            | {root.prim_path for root in joint_request.plan.articulation_roots}
        )
    )
    projected_property_state: dict[str, list[Any]] = {}
    for source_prim_path, entries in source_state.property_state.items():
        authored_prim_path = projected_path(source_prim_path)
        if authored_prim_path in projected_property_state:
            raise ValueError(
                "physics property projection collides on authored prim path "
                f"{authored_prim_path}"
            )
        projected_property_state[authored_prim_path] = entries
    return _PhysicsStageState(
        api_paths=projected_api_paths,
        property_state=projected_property_state,
    )


def _physics_state_payload(state: _PhysicsStageState) -> dict[str, Any]:
    """Serialize a physics stage state into canonical bound-request metadata."""

    payload: dict[str, Any] = {
        key: list(state.api_paths[key]) for key in _PHYSICS_API_PATH_KEYS
    }
    payload[_PHYSICS_PROPERTY_STATE_KEY] = dict(state.property_state)
    return payload


def _parse_physics_state_payload(payload: Any) -> _PhysicsStageState:
    """Recover an exact physics stage state from bound-request metadata."""

    expected_keys = {*_PHYSICS_API_PATH_KEYS, _PHYSICS_PROPERTY_STATE_KEY}
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError(
            "physics_api_inventory must contain exact rigid-body, "
            "articulation-root, mass, and collision path lists plus the "
            "authored physics property state"
        )
    api_paths: dict[str, tuple[str, ...]] = {}
    for key in _PHYSICS_API_PATH_KEYS:
        raw_paths = payload[key]
        if (
            not isinstance(raw_paths, list)
            or not all(isinstance(path, str) for path in raw_paths)
            or raw_paths != sorted(set(raw_paths))
        ):
            raise ValueError(
                f"physics_api_inventory {key} must be a sorted unique string list"
            )
        api_paths[key] = tuple(raw_paths)
    raw_property_state = payload[_PHYSICS_PROPERTY_STATE_KEY]
    if not isinstance(raw_property_state, Mapping) or not all(
        isinstance(prim_path, str) and isinstance(entries, list)
        for prim_path, entries in raw_property_state.items()
    ):
        raise ValueError(
            f"physics_api_inventory {_PHYSICS_PROPERTY_STATE_KEY} must map prim "
            "paths to authored physics property entry lists"
        )
    property_state = {
        prim_path: list(entries) for prim_path, entries in raw_property_state.items()
    }
    return _PhysicsStageState(api_paths=api_paths, property_state=property_state)


def _physics_property_state_failures(
    expected_property_state: Mapping[str, Any],
    observed_property_state: Mapping[str, Any],
) -> tuple[str, ...]:
    """Compare exact source-projected authored physics values to the saved stage."""

    failures: list[str] = []
    for prim_path in sorted(
        set(expected_property_state) | set(observed_property_state)
    ):
        expected_entries = expected_property_state.get(prim_path)
        observed_entries = observed_property_state.get(prim_path)
        if expected_entries == observed_entries:
            continue
        if expected_entries is None:
            failures.append(
                "Saved physics property state differs from the source-bound "
                f"expectation: {prim_path} gained unbound authored physics "
                f"properties {json.dumps(observed_entries, sort_keys=True)}."
            )
        elif observed_entries is None:
            failures.append(
                "Saved physics property state differs from the source-bound "
                f"expectation: {prim_path} lost authored physics properties "
                f"{json.dumps(expected_entries, sort_keys=True)}."
            )
        else:
            failures.append(
                "Saved physics property state differs from the source-bound "
                f"expectation at {prim_path}: "
                f"expected={json.dumps(expected_entries, sort_keys=True)}, "
                f"observed={json.dumps(observed_entries, sort_keys=True)}."
            )
    return tuple(failures)


def _validate_saved_owned_core_contract_stage(
    stage: Any,
    *,
    joint_request: Any,
    diagnostics: Any,
    expected_physics_state: _PhysicsStageState,
) -> tuple[str, ...]:
    """Run shared exact topology checks plus wrapper-owned API inventory checks."""

    from world_understanding.functions.physics.joint_rigger import (
        JointRiggerContractError,
        JointRiggerInputV2,
        validate_authored_joint_topology,
        validate_authored_rigid_links,
        validate_v2_articulation_roots,
    )

    failures = list(
        _owned_core_diagnostic_failures(
            diagnostics,
            joint_request,
        )
    )
    topology_plan = _project_owned_core_topology_plan(joint_request)
    try:
        uses_external_author_metadata = (
            diagnostics.backend_name == "stage2_candidate_edges"
        )
        authored_joint_paths_by_id = None
        if uses_external_author_metadata:
            # The transitional backend records its exact authored prim path as
            # ``joint_id``. Adapt only path resolution; the shared validator
            # still enforces every schema, property, scope, and frame invariant
            # against the saved stage. Its author-owned metadata is validated
            # separately by ``_validate_saved_owned_core_graph``.
            authored_joint_paths_by_id = {
                joint.topology.joint_id: joint.topology.joint_id
                for joint in topology_plan.joints
            }
        validate_authored_joint_topology(
            stage,
            topology_plan,
            None if uses_external_author_metadata else diagnostics,
            authored_joint_paths_by_id=authored_joint_paths_by_id,
            validate_joint_rigger_metadata=not uses_external_author_metadata,
        )
        if isinstance(joint_request, JointRiggerInputV2):
            validate_authored_rigid_links(stage, joint_request)
            validate_v2_articulation_roots(stage, joint_request)
    except (JointRiggerContractError, TypeError) as exc:
        failures.append(f"Saved Joint Rigger contract validation failed: {exc}")

    try:
        observed_state = _stage_physics_state(stage)
    except ValueError as exc:
        failures.append(f"Saved physics API inventory failed: {exc}")
        return tuple(failures)
    for api_name, label in (
        ("rigid_body_api_paths", "RigidBodyAPI"),
        ("articulation_root_api_paths", "ArticulationRootAPI"),
        ("mass_api_paths", "MassAPI"),
        ("collision_api_paths", "CollisionAPI"),
    ):
        expected_paths = tuple(expected_physics_state.api_paths.get(api_name, ()))
        observed_paths = observed_state.api_paths[api_name]
        if observed_paths != expected_paths:
            failures.append(
                f"Saved {label} inventory differs from the source-bound "
                f"expectation: expected={list(expected_paths)}, "
                f"observed={list(observed_paths)}."
            )
    failures.extend(
        _physics_property_state_failures(
            expected_physics_state.property_state,
            observed_state.property_state,
        )
    )
    return tuple(failures)


def _validate_saved_owned_core_contract(
    output_path: Path,
    *,
    joint_request: Any,
    diagnostics: Any,
    expected_physics_state: _PhysicsStageState,
) -> tuple[str, ...]:
    """Open one saved stage and run exact owned-core contract validation."""

    from pxr import Usd

    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        return ("Published USDZ could not be opened as a USD stage.",)
    return _validate_saved_owned_core_contract_stage(
        stage,
        joint_request=joint_request,
        diagnostics=diagnostics,
        expected_physics_state=expected_physics_state,
    )


def _expected_stage2_candidate_custom_data(candidate: Any) -> dict[str, Any]:
    """Project the exact customData authored by Stage 2 candidate edges."""

    from joint_agent.functions.candidate_edge_authoring import (
        stage2_candidate_custom_data,
    )

    authored_limit_unit = None
    if candidate.limit_readiness == "source_backed":
        authored_limit_unit = (
            "degrees" if candidate.motion_type == "revolute" else "stage_units"
        )
    return cast(
        dict[str, Any],
        stage2_candidate_custom_data(
            candidate,
            authored_limit_unit=authored_limit_unit,
        ),
    )


def _validate_saved_owned_core_graph(
    output_path: Path,
    candidate_document: Stage2CandidateDocument,
    *,
    diagnostic_joint_paths: tuple[str, ...],
    diagnostic_bindings: tuple[_OwnedCoreDiagnosticBinding, ...] = (),
    expected_joint_request: Any | None = None,
    expected_diagnostics: Any | None = None,
    expected_physics_state: _PhysicsStageState | None = None,
    require_stage2_candidate_custom_data: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Open the published stage and compare exact managed joints to Stage 2."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    failures: list[str] = []
    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        return (), ("Published USDZ could not be opened as a USD stage.",), ()
    if (
        expected_joint_request is not None
        and expected_diagnostics is not None
        and expected_physics_state is not None
    ):
        failures.extend(
            _validate_saved_owned_core_contract_stage(
                stage,
                joint_request=expected_joint_request,
                diagnostics=expected_diagnostics,
                expected_physics_state=expected_physics_state,
            )
        )

    managed_prims: dict[str, Any] = {}
    managed_paths: set[str] = set()
    physics_joint_paths: set[str] = set()
    retained_paths: set[str] = set()
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in candidate_document.candidates
    }
    bindings_by_path = {binding.prim_path: binding for binding in diagnostic_bindings}
    bindings_by_candidate_id = {
        binding.candidate_id: binding for binding in diagnostic_bindings
    }
    prim_visits = 0
    scan_stopped = False

    def inspect_prims(prims: Any, *, phase: str) -> None:
        nonlocal prim_visits, scan_stopped
        for prim in prims:
            prim_visits += 1
            if prim_visits > _MAX_SAVED_GRAPH_PRIM_VISITS:
                failures.append(
                    "Saved joint graph inspection exceeds the fixed "
                    f"{_MAX_SAVED_GRAPH_PRIM_VISITS}-prim visit limit during "
                    f"{phase}."
                )
                scan_stopped = True
                return

            prim_path = str(prim.GetPath())
            binding = bindings_by_path.get(prim_path)
            candidate_id = prim.GetCustomDataByKey("jointAgent:candidateId")
            joint_rigger_id = prim.GetCustomDataByKey("jointRigger:jointId")
            is_physics_joint = prim.IsA(UsdPhysics.Joint)
            has_managed_marker = (
                candidate_id is not None
                or joint_rigger_id is not None
                or binding is not None
            )
            if not has_managed_marker and not is_physics_joint:
                continue

            if prim_path not in retained_paths:
                if len(retained_paths) >= _MAX_SAVED_GRAPH_PATHS:
                    failures.append(
                        "Saved joint graph inspection exceeds the fixed "
                        f"{_MAX_SAVED_GRAPH_PATHS}-path retention limit."
                    )
                    scan_stopped = True
                    return
                retained_paths.add(prim_path)
            if is_physics_joint:
                physics_joint_paths.add(prim_path)
            if has_managed_marker:
                managed_paths.add(prim_path)

            if binding is not None:
                if candidate_id is not None and candidate_id != binding.candidate_id:
                    failures.append(
                        f"Owned joint {prim_path} has a conflicting candidate ID "
                        "marker."
                    )
                candidate_id = binding.candidate_id
                if joint_rigger_id != binding.joint_id:
                    failures.append(
                        f"Owned joint {prim_path} has the wrong Joint Rigger ID."
                    )
                if (
                    prim.GetCustomDataByKey("jointRigger:planSha256")
                    != binding.plan_sha256
                ):
                    failures.append(
                        f"Owned joint {prim_path} has the wrong Joint Rigger plan "
                        "identity."
                    )
                if (
                    prim.GetCustomDataByKey("jointRigger:authoringVersion")
                    != binding.authoring_version
                ):
                    failures.append(
                        f"Owned joint {prim_path} has the wrong Joint Rigger "
                        "authoring version."
                    )
                if (
                    prim.GetCustomDataByKey("jointRigger:fieldDecisions")
                    != binding.field_decisions_json
                ):
                    failures.append(
                        f"Owned joint {prim_path} has Joint Rigger field decisions "
                        "that differ from bound diagnostics."
                    )
            elif joint_rigger_id is not None:
                failures.append(
                    f"Joint Rigger marker at {prim_path} is not bound by diagnostics."
                )

            if candidate_id is None:
                continue

            if not isinstance(candidate_id, str) or not candidate_id.strip():
                failures.append(
                    f"Managed joint {prim_path} has an invalid candidate ID marker."
                )
                continue
            candidate = candidates_by_id.get(candidate_id)
            # GetCustomData() includes schema fallbacks such as userDocBrief;
            # the Stage 2 contract governs only opinions authored on the prim.
            authored_custom_data = {
                key: value
                for key, value in prim.GetCustomData().items()
                if prim.HasAuthoredCustomDataKey(key)
            }
            if (
                binding is None
                and candidate is not None
                and require_stage2_candidate_custom_data
                and authored_custom_data
                != _expected_stage2_candidate_custom_data(candidate)
            ):
                failures.append(
                    f"Managed joint {prim_path} has transitional customData that "
                    "differs from the approved candidate evidence."
                )
            if candidate_id in managed_prims:
                failures.append(
                    f"Candidate {candidate_id} is marked on more than one joint prim."
                )
                continue
            managed_prims[candidate_id] = prim
            if binding is None and (
                prim.GetCustomDataByKey("jointAgent:sourceSchemaVersion")
                != "joint-agent-stage2-v0"
            ):
                failures.append(
                    f"Managed joint {prim_path} has the wrong source schema marker."
                )
            if not (
                prim.IsA(UsdPhysics.RevoluteJoint)
                or prim.IsA(UsdPhysics.PrismaticJoint)
            ):
                failures.append(
                    f"Managed prim {prim_path} is not a supported USD physics joint."
                )

    inspect_prims(
        Usd.PrimRange.Stage(
            stage,
            Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
        ),
        phase="composed-stage scan",
    )
    if not scan_stopped:
        for prototype in stage.GetPrototypes():
            inspect_prims(
                Usd.PrimRange.AllPrims(prototype),
                phase="prototype scan",
            )
            if scan_stopped:
                break
    unexpected_joint_paths = sorted(physics_joint_paths - managed_paths)
    marker_non_joint_paths = sorted(managed_paths - physics_joint_paths)
    if unexpected_joint_paths or marker_non_joint_paths:
        details: list[str] = []
        if unexpected_joint_paths:
            details.append(
                "unapproved physics joints: " + ", ".join(unexpected_joint_paths)
            )
        if marker_non_joint_paths:
            details.append(
                "marker-bearing non-joints: " + ", ".join(marker_non_joint_paths)
            )
        failures.append(
            "Saved physics joint inventory differs from approved managed joints; "
            + "; ".join(details)
            + "."
        )

    expected_ids = candidate_document.candidate_ids
    if set(managed_prims) != set(expected_ids) or len(managed_prims) != len(
        expected_ids
    ):
        failures.append(
            "Saved managed candidate markers differ from the approved candidate IDs."
        )
    if set(diagnostic_joint_paths) != physics_joint_paths or len(
        diagnostic_joint_paths
    ) != len(physics_joint_paths):
        failures.append(
            "Joint Rigger diagnostic joint paths differ from all saved physics joints."
        )

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        failures.append("Published stage has invalid metersPerUnit.")
        meters_per_unit = 1.0
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    validated_ids: list[str] = []
    axis_vectors = {
        "X": Gf.Vec3d(1.0, 0.0, 0.0),
        "Y": Gf.Vec3d(0.0, 1.0, 0.0),
        "Z": Gf.Vec3d(0.0, 0.0, 1.0),
    }

    for candidate in candidate_document.candidates:
        prim = managed_prims.get(candidate.candidate_id)
        if prim is None:
            continue
        binding = bindings_by_candidate_id.get(candidate.candidate_id)
        candidate_failures: list[str] = []
        expected_schema = (
            UsdPhysics.RevoluteJoint
            if candidate.motion_type == "revolute"
            else UsdPhysics.PrismaticJoint
        )
        if not prim.IsA(expected_schema):
            candidate_failures.append("joint type")
        joint = expected_schema(prim)

        expected_body0 = Sdf.Path(
            binding.body0_prim_path
            if binding is not None
            else cast(str, candidate.fixed_parent_prim)
        )
        expected_body1 = Sdf.Path(
            binding.body1_prim_path
            if binding is not None
            else candidate.moving_part_prims[0]
        )
        if list(joint.GetBody0Rel().GetTargets()) != [expected_body0]:
            candidate_failures.append("body0")
        if list(joint.GetBody1Rel().GetTargets()) != [expected_body1]:
            candidate_failures.append("body1")

        body0_prim = stage.GetPrimAtPath(expected_body0)
        body1_prim = stage.GetPrimAtPath(expected_body1)
        local_pos0_attr = joint.GetLocalPos0Attr()
        local_pos1_attr = joint.GetLocalPos1Attr()
        anchor_attributes_valid = True
        for label, attribute in (
            ("localPos0 anchor", local_pos0_attr),
            ("localPos1 anchor", local_pos1_attr),
        ):
            if (
                not attribute.HasAuthoredValueOpinion()
                or attribute.GetNumTimeSamples() != 0
                or attribute.Get() is None
            ):
                candidate_failures.append(label)
                anchor_attributes_valid = False
        if (
            not body0_prim
            or not body0_prim.IsValid()
            or not body1_prim
            or not body1_prim.IsValid()
        ):
            candidate_failures.append("shared anchor bodies")
            anchor_attributes_valid = False
        if anchor_attributes_valid:
            try:
                body0_transform = xform_cache.GetLocalToWorldTransform(body0_prim)
                body1_transform = xform_cache.GetLocalToWorldTransform(body1_prim)
                expected_anchor_value = body1_transform.Transform(
                    Gf.Vec3d(0.0, 0.0, 0.0)
                )
                expected_anchor = _vector3(expected_anchor_value)
                local_pos0 = local_pos0_attr.Get()
                local_pos1 = local_pos1_attr.Get()
                anchor0_value = body0_transform.Transform(
                    Gf.Vec3d(*(float(value) for value in local_pos0))
                )
                anchor1_value = body1_transform.Transform(
                    Gf.Vec3d(*(float(value) for value in local_pos1))
                )
                anchor0 = _vector3(anchor0_value)
                anchor1 = _vector3(anchor1_value)
                if not _anchor_positions_close(anchor0, expected_anchor):
                    candidate_failures.append("localPos0 anchor")
                if not _anchor_positions_close(anchor1, expected_anchor):
                    candidate_failures.append("localPos1 anchor")
                if not _anchor_positions_close(anchor0, anchor1):
                    candidate_failures.append("shared anchor")
            except (TypeError, ValueError):
                candidate_failures.append("shared anchor")

        axis_attr = joint.GetAxisAttr()
        local_rot0_attr = joint.GetLocalRot0Attr()
        local_rot1_attr = joint.GetLocalRot1Attr()
        expected_axis_token = candidate.axis_hint[-1].upper()
        for label, attribute in (
            ("axis", axis_attr),
            ("localRot0", local_rot0_attr),
            ("localRot1", local_rot1_attr),
        ):
            if (
                not attribute.HasAuthoredValueOpinion()
                or attribute.GetNumTimeSamples() != 0
            ):
                candidate_failures.append(label)
        authored_axis_token = str(axis_attr.Get())
        if authored_axis_token != expected_axis_token:
            candidate_failures.append("axis token")
        else:
            base_axis = axis_vectors[authored_axis_token]
            expected_world_axis = cast(
                tuple[float, float, float],
                candidate.motion_axis_world,
            )
            for label, body_path, local_rotation in (
                ("body0 axis", expected_body0, local_rot0_attr.Get()),
                ("body1 axis", expected_body1, local_rot1_attr.Get()),
            ):
                body_prim = stage.GetPrimAtPath(body_path)
                if not body_prim or not body_prim.IsValid():
                    candidate_failures.append(label)
                    continue
                try:
                    local_axis = Gf.Rotation(local_rotation).TransformDir(base_axis)
                    world_axis = xform_cache.GetLocalToWorldTransform(
                        body_prim
                    ).TransformDir(local_axis)
                    if not _vectors_close(
                        _normalized_vector(world_axis),
                        expected_world_axis,
                    ):
                        candidate_failures.append(label)
                except (TypeError, ValueError):
                    candidate_failures.append(label)

        for label, attribute, candidate_value in (
            ("lower limit", joint.GetLowerLimitAttr(), candidate.lower_limit),
            ("upper limit", joint.GetUpperLimitAttr(), candidate.upper_limit),
        ):
            if attribute.GetNumTimeSamples():
                candidate_failures.append(f"{label} time samples")
            if candidate.limit_readiness != "source_backed" or candidate_value is None:
                if attribute.HasAuthoredValueOpinion():
                    candidate_failures.append(label)
                continue
            expected_limit = candidate_value
            if candidate.motion_type == "prismatic":
                expected_limit /= meters_per_unit
            authored_limit = attribute.Get()
            if (
                not attribute.HasAuthoredValueOpinion()
                or authored_limit is None
                or not math.isclose(
                    float(authored_limit),
                    expected_limit,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            ):
                candidate_failures.append(label)

        if candidate_failures:
            failures.append(
                f"Candidate {candidate.candidate_id} saved readback differs in: "
                + ", ".join(sorted(set(candidate_failures)))
                + "."
            )
        else:
            validated_ids.append(candidate.candidate_id)

    return tuple(validated_ids), tuple(failures), tuple(sorted(managed_paths))


def _build_bound_owned_core_request(
    request: ArticulationAuthoringRequest,
) -> Any:
    """Rebuild the exact owned-core request from the bound workflow inputs."""

    from joint_agent.functions.joint_rigger_core_bridge import (
        build_stage2_articulation_contract_input,
        build_stage2_candidate_edges_input,
    )

    accepted_plan = _load_bound_accepted_authoring_plan(request)
    if accepted_plan is not None:
        return _accepted_execution_request(request, accepted_plan)
    if request.predictions_path is None:
        return build_stage2_candidate_edges_input(
            input_usd_path=request.source_asset,
            articulation_candidates_path=request.candidate_document_path,
            expected_articulation_candidates_sha256=(request.candidate_document_sha256),
        )
    return build_stage2_articulation_contract_input(
        input_usd_path=request.source_asset,
        articulation_candidates_path=request.candidate_document_path,
        predictions_path=request.predictions_path,
        expected_articulation_candidates_sha256=request.candidate_document_sha256,
    )


def _project_owned_core_topology_plan(joint_request: Any) -> Any:
    """Project a full V2 request onto the exact joint-authored plan identity."""

    from world_understanding.functions.physics.joint_rigger import JointRiggerPlanV1

    if isinstance(joint_request.plan, JointRiggerPlanV1):
        return joint_request.plan
    return JointRiggerPlanV1(
        schema_version="world-understanding-joint-rigger-plan-v1",
        joints=joint_request.plan.joints,
    )


def _owned_core_diagnostic_failures(
    diagnostics: Any,
    joint_request: Any,
) -> tuple[str, ...]:
    """Run the core facade's exact request-versus-diagnostics validator."""

    from world_understanding.functions.physics.joint_rigger import (
        JointRiggerFacadeError,
        validate_diagnostic_decisions,
    )

    try:
        validate_diagnostic_decisions(joint_request, diagnostics)
    except (JointRiggerFacadeError, TypeError, ValueError) as exc:
        return (f"Saved Joint Rigger diagnostic contract validation failed: {exc}",)
    return ()


def _resolve_owned_core_diagnostic_bindings(
    candidate_document: Stage2CandidateDocument,
    *,
    joint_diagnostics: tuple[Any, ...],
    plan_sha256: str,
    expected_topology_plan: Any | None = None,
    backend_name: str,
    backend_version: str | None,
) -> tuple[
    tuple[_OwnedCoreDiagnosticBinding, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Resolve owned-core diagnostics through exact typed endpoint provenance."""

    logical_joint_ids = tuple(diagnostic.joint_id for diagnostic in joint_diagnostics)
    authored_paths = tuple(
        diagnostic.authored_prim_path for diagnostic in joint_diagnostics
    )
    if not any(path is not None for path in authored_paths):
        if backend_name == "stage2_candidate_edges":
            return (), logical_joint_ids, ()
        return (
            (),
            logical_joint_ids,
            ("Owned Joint Rigger diagnostics omit every authored joint prim path.",),
        )

    failures: list[str] = []
    if any(path is None for path in authored_paths):
        failures.append(
            "Joint Rigger diagnostics mix authored and missing joint prim paths."
        )
        return (
            (),
            tuple(path for path in authored_paths if path is not None),
            tuple(failures),
        )
    if backend_version is None:
        failures.append(
            "Joint Rigger diagnostics with authored paths omit backend_version."
        )
        return (), cast(tuple[str, ...], authored_paths), tuple(failures)

    expected_joints_by_id: dict[str, Any] = {}
    if expected_topology_plan is not None:
        expected_joints_by_id = {
            joint.topology.joint_id: joint for joint in expected_topology_plan.joints
        }
        if set(expected_joints_by_id) != set(logical_joint_ids):
            failures.append(
                "Joint Rigger diagnostics differ from the exact bound topology plan."
            )
            return (), cast(tuple[str, ...], authored_paths), tuple(failures)

    candidates_by_endpoints: dict[tuple[str, str], list[Any]] = {}
    candidates_by_body1_link: dict[str, list[Any]] = {}
    for candidate in candidate_document.candidates:
        if candidate.fixed_parent_prim is None or len(candidate.moving_part_prims) != 1:
            failures.append(
                f"Candidate {candidate.candidate_id} lacks exact body endpoints."
            )
            continue
        key = (candidate.fixed_parent_prim, candidate.moving_part_prims[0])
        candidates_by_endpoints.setdefault(key, []).append(candidate)
        candidates_by_body1_link.setdefault(candidate.moving_part_prims[0], []).append(
            candidate
        )

    bindings: list[_OwnedCoreDiagnosticBinding] = []
    bound_candidate_ids: set[str] = set()
    for diagnostic, authored_path in zip(
        joint_diagnostics,
        authored_paths,
        strict=True,
    ):
        authored_path = cast(str, authored_path)
        decisions = {
            decision.field: decision for decision in diagnostic.field_decisions
        }
        body0_decision = decisions.get("topology.body0")
        body1_decision = decisions.get("topology.body1")
        if (
            body0_decision is None
            or body1_decision is None
            or body0_decision.disposition != "accepted"
            or body1_decision.disposition != "accepted"
            or body0_decision.provenance is None
            or body1_decision.provenance is None
            or body0_decision.provenance.prim_path is None
            or body1_decision.provenance.prim_path is None
        ):
            failures.append(
                f"Joint Rigger diagnostic {diagnostic.joint_id} lacks accepted "
                "body endpoint provenance."
            )
            continue

        body0_provenance = body0_decision.provenance
        body1_provenance = body1_decision.provenance
        body0_prim_path = cast(str, body0_provenance.prim_path)
        body1_prim_path = cast(str, body1_provenance.prim_path)
        expected_joint = expected_joints_by_id.get(diagnostic.joint_id)
        if expected_joint is not None and (
            body0_prim_path != expected_joint.topology.body0
            or body1_prim_path != expected_joint.topology.body1
        ):
            failures.append(
                f"Joint Rigger diagnostic {diagnostic.joint_id} body endpoints "
                "differ from the exact bound topology plan."
            )
            continue
        body0_joint_property = f"joint:{diagnostic.joint_id}.body0_link"
        body1_joint_property = f"joint:{diagnostic.joint_id}.body1_link"

        def endpoint_link_id(
            properties: tuple[str, ...],
            *,
            joint_property: str,
        ) -> tuple[str | None, bool]:
            if properties == (joint_property,):
                return None, True
            prefix = "link:"
            suffix = ".body_prim_path"
            link_properties = tuple(
                item
                for item in properties
                if item.startswith(prefix) and item.endswith(suffix)
            )
            if len(link_properties) != 1 or set(properties) != {
                joint_property,
                link_properties[0],
            }:
                return None, False
            link_id = link_properties[0][len(prefix) : -len(suffix)]
            if not link_id:
                return None, False
            return link_id, True

        body0_link_id, body0_properties_valid = endpoint_link_id(
            body0_provenance.properties,
            joint_property=body0_joint_property,
        )
        body1_link_id, body1_properties_valid = endpoint_link_id(
            body1_provenance.properties,
            joint_property=body1_joint_property,
        )
        linked_endpoint_provenance = (
            body0_link_id is not None and body1_link_id is not None
        )
        legacy_endpoint_provenance = body0_link_id is None and body1_link_id is None
        if (
            not body0_properties_valid
            or not body1_properties_valid
            or not (linked_endpoint_provenance or legacy_endpoint_provenance)
        ):
            failures.append(
                f"Joint Rigger diagnostic {diagnostic.joint_id} has malformed "
                "body endpoint link provenance."
            )
            continue

        if linked_endpoint_provenance:
            body0_link_id = cast(str, body0_link_id)
            body1_link_id = cast(str, body1_link_id)
            matches = candidates_by_body1_link.get(body1_link_id, [])
        else:
            endpoint_key = (
                body0_prim_path,
                body1_prim_path,
            )
            matches = candidates_by_endpoints.get(endpoint_key, [])
        if len(matches) != 1:
            failures.append(
                f"Joint Rigger diagnostic {diagnostic.joint_id} does not resolve "
                "to exactly one approved candidate by source/link provenance."
            )
            continue
        candidate = matches[0]
        if linked_endpoint_provenance:
            body0_link_id = cast(str, body0_link_id)
            body1_link_id = cast(str, body1_link_id)
            fixed_parent = cast(str, candidate.fixed_parent_prim)
            body0_link_parent = body0_link_id.rsplit("/", 1)[0] or "/"
            fixed_parent_parent = fixed_parent.rsplit("/", 1)[0] or "/"
            if not (
                body0_link_id == fixed_parent
                or body0_link_parent == fixed_parent
                or fixed_parent_parent == body0_link_id
            ):
                failures.append(
                    f"Joint Rigger diagnostic {diagnostic.joint_id} has body0 "
                    "link provenance incompatible with the approved candidate."
                )
                continue
            # V2 may relocate body0 onto an aggregate prim. The owned Joint
            # contract is the authority for that endpoint; do not duplicate
            # its aggregate naming or parenting algorithm in this wrapper.
            relocated_body0_is_bound = (
                expected_joint is not None
                and body0_prim_path == expected_joint.topology.body0
            )
            if body0_prim_path != body0_link_id and not relocated_body0_is_bound:
                failures.append(
                    f"Joint Rigger diagnostic {diagnostic.joint_id} has an "
                    "unsupported authored body0 endpoint."
                )
                continue
            if (
                body1_link_id != candidate.moving_part_prims[0]
                or body1_prim_path != body1_link_id
            ):
                failures.append(
                    f"Joint Rigger diagnostic {diagnostic.joint_id} has an "
                    "unsupported authored body1 endpoint."
                )
                continue
        if candidate.candidate_id in bound_candidate_ids:
            failures.append(
                f"Candidate {candidate.candidate_id} is claimed by more than one "
                "Joint Rigger diagnostic."
            )
            continue

        expected_provenance = (
            (
                "topology.body0",
                body0_prim_path,
                (
                    (body0_joint_property,)
                    if body0_link_id is None
                    else (
                        body0_joint_property,
                        f"link:{body0_link_id}.body_prim_path",
                    )
                ),
            ),
            (
                "topology.body1",
                body1_prim_path,
                (
                    (body1_joint_property,)
                    if body1_link_id is None
                    else (
                        body1_joint_property,
                        f"link:{body1_link_id}.body_prim_path",
                    )
                ),
            ),
            (
                "topology.joint_type",
                body1_prim_path,
                (f"joint:{diagnostic.joint_id}.motion_type",),
            ),
            (
                "topology.axis_stage",
                body1_prim_path,
                (f"joint:{diagnostic.joint_id}.axis_stage",),
            ),
        )
        provenance_valid = True
        contract_artifact = body0_provenance.artifact
        for field, expected_prim_path, expected_properties in expected_provenance:
            decision = decisions.get(field)
            provenance = None if decision is None else decision.provenance
            plan_field = field.removeprefix("topology.")
            expected_plan_provenance = (
                None
                if expected_joint is None
                else expected_joint.topology.field_provenance.get(plan_field)
            )
            if (
                decision is None
                or decision.disposition != "accepted"
                or provenance is None
                or provenance.source != "accepted_manifest"
                or provenance.artifact is None
                or provenance.artifact != contract_artifact
                or provenance.prim_path != expected_prim_path
                or set(provenance.properties) != set(expected_properties)
                or provenance.derivation
                != "articulation_contract_v1_to_joint_rigger_input_v1"
                or (
                    expected_joint is not None
                    and provenance != expected_plan_provenance
                )
            ):
                failures.append(
                    f"Joint Rigger diagnostic {diagnostic.joint_id} has invalid "
                    f"{field} provenance for candidate {candidate.candidate_id}."
                )
                provenance_valid = False
        if not provenance_valid:
            continue

        bound_candidate_ids.add(candidate.candidate_id)
        bindings.append(
            _OwnedCoreDiagnosticBinding(
                candidate_id=candidate.candidate_id,
                joint_id=diagnostic.joint_id,
                prim_path=authored_path,
                body0_prim_path=body0_prim_path,
                body1_prim_path=body1_prim_path,
                plan_sha256=plan_sha256,
                authoring_version=backend_version,
                field_decisions_json=json.dumps(
                    [
                        decision.model_dump(mode="json", exclude_none=True)
                        for decision in diagnostic.field_decisions
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            )
        )

    if bound_candidate_ids != set(candidate_document.candidate_ids):
        failures.append(
            "Joint Rigger diagnostics do not bind exactly the approved candidate set."
        )
    return (
        tuple(bindings),
        cast(tuple[str, ...], authored_paths),
        tuple(failures),
    )


class ArticulationAuthoringClient(Protocol):
    """Provider-neutral seam for deterministic graph authoring and readback."""

    def author(
        self,
        request: ArticulationAuthoringRequest,
        *,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationAuthoringResult:
        """Author only the exact outer-accepted canonical graph."""

    def validate(
        self,
        authoring: ArticulationAuthoringResult,
        *,
        expected_candidate_ids: tuple[str, ...],
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationValidationResult:
        """Reopen the saved output and validate the exact authored joint graph."""


class ArticulationWorkflowClient(ArticulationAuthoringClient, Protocol):
    """Stable seam between workflow durability and inference capabilities."""

    def configuration_sha256(
        self,
        request: ArticulationWorkflowRequest,
    ) -> str:
        """Bind inference configuration before any resumable work starts."""

    def infer(
        self,
        request: ArticulationWorkflowRequest,
        *,
        resume: bool,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationInferenceResult:
        """Run prediction and source-bound Stage 2 candidate inference."""

    def author(
        self,
        request: ArticulationAuthoringRequest,
        *,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationAuthoringResult:
        """Author only the exact accepted, native-ready candidate subset."""

    def validate(
        self,
        authoring: ArticulationAuthoringResult,
        *,
        expected_candidate_ids: tuple[str, ...],
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationValidationResult:
        """Reopen the saved output and validate the exact authored joint graph."""


class MockArticulationCall(BaseModel):
    """One recorded mock adapter invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: str
    candidate_ids: tuple[str, ...] = ()
    resume: bool = False


class MockArticulationWorkflowClient:
    """Credential-free, file-backed Joint workflow adapter for focused tests."""

    def __init__(
        self,
        candidate_document: Stage2CandidateDocument,
        *,
        membership_disposition_document: MembershipDispositionDocument | None = None,
        validated_candidate_ids: tuple[str, ...] | None = None,
        exact_graph_match: bool = True,
        self_contained: bool = True,
    ) -> None:
        self.candidate_document = candidate_document
        self.membership_disposition_document = membership_disposition_document
        self._validated_candidate_ids = validated_candidate_ids
        self._exact_graph_match = exact_graph_match
        self._self_contained = self_contained
        self.calls: list[MockArticulationCall] = []

    def configuration_sha256(
        self,
        request: ArticulationWorkflowRequest,
    ) -> str:
        return _canonical_sha256(
            {
                "adapter": "mock",
                "candidate_document": self.candidate_document.model_dump(mode="json"),
                "membership_disposition_document": (
                    self.membership_disposition_document.model_dump(mode="json")
                    if self.membership_disposition_document is not None
                    else None
                ),
                "source_asset": request.source_asset,
            }
        )

    def infer(
        self,
        request: ArticulationWorkflowRequest,
        *,
        resume: bool,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationInferenceResult:
        del cancel_checker
        self.calls.append(MockArticulationCall(operation="infer", resume=resume))
        return ArticulationInferenceResult(
            candidate_document=self.candidate_document,
            membership_disposition_document=self.membership_disposition_document,
            backend_configuration_sha256=self.configuration_sha256(request),
            backend_run_id="mock-articulation-v1",
            metadata={"backend": "mock", "live_backend_invoked": False},
        )

    def author(
        self,
        request: ArticulationAuthoringRequest,
        *,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationAuthoringResult:
        if cancel_checker is not None and cancel_checker():
            raise asyncio.CancelledError
        _validate_authoring_request_bindings(request)

        output_dir = request.output_dir.resolve() / "joint_rigger"
        output_dir.mkdir(parents=True, exist_ok=True)
        attempt_path = output_dir / "authoring_attempt.json"
        recovery_path = output_dir / "workflow_authoring_result.json"
        output_path = output_dir / "rigged.usdz"
        diagnostics_path = output_dir / "diagnostics.json"
        result_path = output_dir / "result.json"
        attempt = {
            "schema_version": "content-agent-workflows.authoring-attempt.v1",
            "idempotency_key": request.idempotency_key,
            "source_sha256": request.source_sha256,
            "source_dependency_bundle_sha256": (
                request.source_dependency_bundle_sha256
            ),
            "candidate_document_sha256": request.candidate_document_sha256,
            "accepted_candidate_ids": list(request.accepted_candidate_ids),
            "predictions_path": request.predictions_path,
            "predictions_sha256": request.predictions_sha256,
        }
        if attempt_path.exists():
            if load_json(attempt_path) != attempt:
                raise ValueError("mock authoring attempt conflicts with prior request")
        else:
            atomic_write_json(attempt_path, attempt)

        if recovery_path.exists():
            recovered = ArticulationAuthoringResult.model_validate(
                load_json(recovery_path)
            )
            _require_result_matches_authoring_request(recovered, request)
            self.calls.append(
                MockArticulationCall(
                    operation="author_recover",
                    candidate_ids=request.accepted_candidate_ids,
                )
            )
            return recovered
        if any(path.exists() for path in (output_path, diagnostics_path, result_path)):
            raise ValueError(
                "mock authoring artifacts exist without a recoverable result"
            )

        self.calls.append(
            MockArticulationCall(
                operation="author",
                candidate_ids=request.accepted_candidate_ids,
            )
        )
        with zipfile.ZipFile(output_path, mode="w") as archive:
            archive.writestr(
                "rigged.usda",
                '#usda 1.0\n\ndef Xform "MockArticulationOutput" {\n}\n',
            )
        diagnostics = {
            "schema_version": "content-agent-workflows.mock-rigger-diagnostics.v1",
            "authored_edges": [
                {
                    "candidate_id": candidate_id,
                    "joint_path": f"/Mock/Joints/{candidate_id}",
                }
                for candidate_id in request.accepted_candidate_ids
            ],
        }
        atomic_write_json(diagnostics_path, diagnostics)
        from world_understanding.functions.physics.joint_rigger import (
            ArtifactIdentityV1,
            JointDiagnosticV1,
            JointRiggerDiagnosticsV1,
            JointRiggerResultV1,
        )

        output_sha256 = file_sha256(output_path)
        joint_rigger_result = JointRiggerResultV1(
            schema_version="world-understanding-joint-rigger-result-v1",
            status="succeeded",
            input_sha256=_canonical_sha256(
                {"idempotency_key": request.idempotency_key, "kind": "input"}
            ),
            plan_sha256=_canonical_sha256(
                {"idempotency_key": request.idempotency_key, "kind": "plan"}
            ),
            output_artifact=ArtifactIdentityV1(
                uri=output_path.resolve().as_uri(),
                root_sha256=output_sha256,
                dependency_bundle_sha256=output_sha256,
            ),
            diagnostics=JointRiggerDiagnosticsV1(
                schema_version="world-understanding-joint-rigger-diagnostics-v1",
                backend_name="mock-articulation-v1",
                joint_diagnostics=tuple(
                    JointDiagnosticV1(joint_id=f"/Mock/Joints/{candidate_id}")
                    for candidate_id in request.accepted_candidate_ids
                ),
            ),
        )
        atomic_write_json(result_path, joint_rigger_result)
        authored = ArticulationAuthoringResult(
            status="succeeded",
            idempotency_key=request.idempotency_key,
            source_sha256=request.source_sha256,
            source_dependency_bundle_sha256=(request.source_dependency_bundle_sha256),
            candidate_document_path=request.candidate_document_path,
            candidate_document_sha256=request.candidate_document_sha256,
            output_asset_path=str(output_path.resolve()),
            output_asset_sha256=output_sha256,
            authored_candidate_ids=request.accepted_candidate_ids,
            authored_joint_count=len(request.accepted_candidate_ids),
            diagnostics_path=str(diagnostics_path.resolve()),
            diagnostics_sha256=file_sha256(diagnostics_path),
            joint_rigger_result_path=str(result_path.resolve()),
            joint_rigger_result_sha256=file_sha256(result_path),
            backend_call_id=f"mock-author-{len(self.calls)}",
            metadata={"backend": "mock", "live_backend_invoked": False},
        )
        atomic_write_json(recovery_path, authored)
        return authored

    def validate(
        self,
        authoring: ArticulationAuthoringResult,
        *,
        expected_candidate_ids: tuple[str, ...],
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationValidationResult:
        if cancel_checker is not None and cancel_checker():
            raise asyncio.CancelledError
        self.calls.append(
            MockArticulationCall(
                operation="validate",
                candidate_ids=expected_candidate_ids,
            )
        )
        validated_ids = (
            self._validated_candidate_ids
            if self._validated_candidate_ids is not None
            else expected_candidate_ids
        )
        failures: list[str] = []
        if not self._exact_graph_match:
            failures.append("Saved joint graph differs from the accepted candidates.")
        if not self._self_contained:
            failures.append("Saved USDZ is not self-contained.")
        if validated_ids != expected_candidate_ids:
            failures.append("Saved joint IDs differ from accepted candidate IDs.")
        observed_output_sha256 = file_sha256(authoring.output_asset_path)
        if observed_output_sha256 != authoring.output_asset_sha256:
            failures.append("Saved USDZ digest differs from the authored output.")
        passed = not failures
        return ArticulationValidationResult(
            status="pass" if passed else "fail",
            output_asset_path=authoring.output_asset_path,
            expected_output_asset_sha256=authoring.output_asset_sha256,
            observed_output_asset_sha256=observed_output_sha256,
            expected_candidate_ids=expected_candidate_ids,
            validated_candidate_ids=validated_ids,
            exact_graph_match=self._exact_graph_match,
            self_contained=self._self_contained,
            failures=tuple(failures),
            evidence_paths=tuple(
                path
                for path in (
                    authoring.diagnostics_path,
                    authoring.joint_rigger_result_path,
                )
                if path is not None
            ),
            metadata={"backend": "mock", "live_backend_invoked": False},
        )


class _JointAgentClientImplementation:
    """Dynamic local adapter for the existing Joint Agent and ``owned_core``.

    The dynamic import keeps the articulation workflow package independent of a
    particular Joint Agent 0.6 implementation. The Joint Agent must still be
    installed in the active environment when this adapter is used.
    """

    def __init__(
        self,
        config: str | Path | Mapping[str, Any],
        *,
        source_config_path: str | Path | None = None,
        session_id: str | None = None,
        verbose: bool = False,
    ) -> None:
        self._config = config
        self._source_config_path = (
            Path(source_config_path).expanduser().resolve()
            if source_config_path is not None
            else None
        )
        self._session_id = session_id
        self._verbose = verbose

    def _load_config(self) -> tuple[dict[str, Any], Path | None]:
        if isinstance(self._config, Mapping):
            return copy.deepcopy(dict(self._config)), self._source_config_path
        config_path = Path(self._config).expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Joint Agent config must be a mapping: {config_path}")
        return cast(dict[str, Any], payload), config_path

    def _resolved_config(
        self,
        request: ArticulationWorkflowRequest,
    ) -> tuple[dict[str, Any], Path | None]:
        config, source_config_path = self._load_config()
        config.setdefault("project", {})
        config.setdefault("input", {})
        config.setdefault("steps", {})
        if not isinstance(config["project"], dict):
            raise ValueError("Joint Agent config project must be a mapping")
        if not isinstance(config["input"], dict):
            raise ValueError("Joint Agent config input must be a mapping")
        if not isinstance(config["steps"], dict):
            raise ValueError("Joint Agent config steps must be a mapping")

        config["project"]["working_dir"] = str(
            (request.output_dir.resolve() / "joint_agent_session").resolve()
        )
        config["input"]["usd_path"] = str(Path(request.source_asset).resolve())
        for step_name in ("apply_joint_rigger", "author_physics_schemas"):
            step = config["steps"].setdefault(step_name, {})
            if not isinstance(step, dict):
                raise ValueError(f"Joint Agent step {step_name} must be a mapping")
            step["enabled"] = False
        inference_step = config["steps"].setdefault("infer_articulation_candidates", {})
        if not isinstance(inference_step, dict):
            raise ValueError(
                "Joint Agent infer_articulation_candidates must be a mapping"
            )
        inference_step["enabled"] = True
        inference_step["candidate_joint_types"] = list(request.allowed_motion_types)
        prepare_step = config["steps"].setdefault(
            "build_dataset_prepare_dataset",
            {},
        )
        if not isinstance(prepare_step, dict):
            raise ValueError(
                "Joint Agent build_dataset_prepare_dataset must be a mapping"
            )
        prompts = prepare_step.setdefault("prompts", {})
        if not isinstance(prompts, dict):
            raise ValueError(
                "Joint Agent build_dataset_prepare_dataset prompts must be a mapping"
            )
        configured_user_prompt = prompts.get("user")
        if configured_user_prompt is not None and not isinstance(
            configured_user_prompt,
            str,
        ):
            raise ValueError("Joint Agent user prompt must be a string")
        prompts["user"] = (
            f"{configured_user_prompt.rstrip()}\n\nWorkflow intent:\n{request.intent}"
            if configured_user_prompt
            else request.intent
        )
        return config, source_config_path

    def configuration_sha256(
        self,
        request: ArticulationWorkflowRequest,
    ) -> str:
        config, source_config_path = self._resolved_config(request)
        return _canonical_sha256(
            {
                "config": config,
                "source_config_path": (
                    str(source_config_path.resolve())
                    if source_config_path is not None
                    else None
                ),
                "relative_path_anchor": str(
                    source_config_path.resolve().parent
                    if source_config_path is not None
                    else Path.cwd().resolve()
                ),
                "session_id": self._session_id,
            }
        )

    def infer(
        self,
        request: ArticulationWorkflowRequest,
        *,
        resume: bool,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationInferenceResult:
        try:
            from joint_agent.api import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Joint graph authoring requires the joint-agent package in the "
                "active Python environment"
            ) from exc

        config, source_config_path = self._resolved_config(request)
        result = pipeline(
            config,
            skip_steps=["apply_joint_rigger", "author_physics_schemas"],
            session_id=self._session_id,
            resume=resume,
            clean=False,
            verbose=self._verbose,
            cancel_checker=cancel_checker,
            source_config_path=source_config_path,
        )
        if not result.success:
            terminal_status = getattr(result, "terminal_status", None)
            if terminal_status is not None:
                raise JointAgentInferenceTerminalError(terminal_status)
            raise RuntimeError(result.error or "Joint Agent inference failed")
        candidate_step = result.step_results.get("infer_articulation_candidates", {})
        candidate_path_value = candidate_step.get("articulation_candidates_path")
        if not candidate_path_value:
            raise RuntimeError(
                "Joint Agent completed without an articulation candidate artifact"
            )
        candidate_path = Path(str(candidate_path_value)).expanduser().resolve()
        candidate_document = Stage2CandidateDocument.model_validate(
            load_json(candidate_path)
        )

        prediction_value = result.step_results.get("consistency_pass", {}).get(
            "consistent_predictions_path"
        ) or result.step_results.get("predict", {}).get("predictions_path")
        report_value = candidate_step.get("articulation_report_path")
        predictions_path = (
            Path(str(prediction_value)).expanduser().resolve()
            if prediction_value
            else None
        )
        if predictions_path is None:
            raise RuntimeError(
                "Joint Agent completed without predictions required for typed "
                "membership disposition"
            )

        structure_metadata = result.step_results.get("analyze_structure", {}).get(
            "structure_metadata"
        )
        if not isinstance(structure_metadata, dict):
            structure_metadata = {}

        def prediction_rows() -> Iterator[dict[str, Any]]:
            with predictions_path.open(encoding="utf-8") as stream:
                for raw_line in stream:
                    line = raw_line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise RuntimeError(
                            "Joint Agent membership disposition requires "
                            "JSON-object rows"
                        )
                    yield row

        predictions_sha256 = file_sha256(predictions_path)
        non_articulated = _verified_non_articulated_structure(
            candidate_document,
            structure_metadata,
        )
        if non_articulated:
            membership_disposition_document = _empty_membership_disposition_document()
        else:
            try:
                from joint_agent.functions.membership_disposition import (
                    infer_membership_dispositions,
                )

                membership_disposition_document = (
                    MembershipDispositionDocument.model_validate(
                        infer_membership_dispositions(
                            prediction_rows(),
                            candidate_document,
                            candidate_joint_types=request.allowed_motion_types,
                        ).model_dump(mode="json")
                    )
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "Joint Agent membership disposition could not read "
                    f"predictions: {exc}"
                ) from exc
            except (ImportError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Joint Agent membership disposition failed closed: {exc}"
                ) from exc
        report_path = (
            Path(str(report_value)).expanduser().resolve() if report_value else None
        )
        provider_response_evidence: dict[str, Any] = {}
        for step_name, path_key, sha_key, metadata_prefix in (
            (
                "analyze_structure",
                "structure_provider_response_diagnostics_path",
                "structure_provider_response_diagnostics_sha256",
                "structure_provider_response_diagnostics",
            ),
            (
                "predict",
                "provider_response_diagnostics_path",
                "provider_response_diagnostics_sha256",
                "stage1_provider_response_diagnostics",
            ),
        ):
            step_evidence = result.step_results.get(step_name, {})
            evidence_path_value = step_evidence.get(path_key)
            evidence_sha256 = step_evidence.get(sha_key)
            if not evidence_path_value and not evidence_sha256:
                continue
            if not evidence_path_value or not evidence_sha256:
                raise RuntimeError(
                    "Joint Agent provider-response evidence is missing its "
                    f"{step_name} path or digest binding"
                )
            evidence_path = _require_bound_file(
                str(evidence_path_value),
                str(evidence_sha256),
                label=f"{step_name} provider-response diagnostics",
            )
            provider_response_evidence[f"{metadata_prefix}_path"] = str(evidence_path)
            provider_response_evidence[f"{metadata_prefix}_sha256"] = str(
                evidence_sha256
            )
        return ArticulationInferenceResult(
            candidate_document=candidate_document,
            membership_disposition_document=membership_disposition_document,
            backend_configuration_sha256=self.configuration_sha256(request),
            predictions_path=(
                str(predictions_path) if predictions_path is not None else None
            ),
            predictions_sha256=predictions_sha256,
            report_path=str(report_path) if report_path is not None else None,
            report_sha256=(
                file_sha256(report_path) if report_path is not None else None
            ),
            backend_run_id=result.session_id,
            metadata={
                "backend": "joint-agent-local",
                "membership_disposition_schema_version": (
                    membership_disposition_document.schema_version
                ),
                "membership_disposition_required": not non_articulated,
                "membership_disposition_classic_fallback_used": False,
                "membership_disposition_suppressed_by_structure": non_articulated,
                "completed_steps": result.completed_steps,
                "live_backend_invoked": True,
                "working_dir": str(result.working_dir)
                if result.working_dir is not None
                else None,
                "structure_analysis_outcome": structure_metadata.get(
                    "structure_outcome"
                ),
                "structure_analysis_reasoning": structure_metadata.get("reasoning"),
                "structure_analysis_evidence": structure_metadata.get("evidence"),
                **provider_response_evidence,
            },
        )

    def author(
        self,
        request: ArticulationAuthoringRequest,
        *,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationAuthoringResult:
        if cancel_checker is not None and cancel_checker():
            raise asyncio.CancelledError
        _validate_authoring_request_bindings(request)

        accepted_plan = _load_bound_accepted_authoring_plan(request)
        source_suffix = Path(request.source_asset).suffix.lower()
        if accepted_plan is not None and source_suffix not in {
            ".usd",
            ".usda",
            ".usdc",
        }:
            raise _ArticulationEvidenceBindingError(
                "accepted rigid-link authoring currently requires a raw USD source"
            )
        output_dir = request.output_dir.resolve() / "joint_rigger"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f"rigged{source_suffix}" if accepted_plan is not None else "rigged.usdz"
        )
        diagnostics_path = output_dir / "diagnostics.json"
        result_path = output_dir / "result.json"
        attempt_path = output_dir / "authoring_attempt.json"
        attempt = {
            "schema_version": "content-agent-workflows.authoring-attempt.v1",
            "idempotency_key": request.idempotency_key,
            "source_sha256": request.source_sha256,
            "source_dependency_bundle_sha256": (
                request.source_dependency_bundle_sha256
            ),
            "candidate_document_sha256": request.candidate_document_sha256,
            "accepted_candidate_ids": list(request.accepted_candidate_ids),
            "predictions_path": request.predictions_path,
            "predictions_sha256": request.predictions_sha256,
            "accepted_authoring_plan_path": request.accepted_authoring_plan_path,
            "accepted_authoring_plan_sha256": (request.accepted_authoring_plan_sha256),
        }
        attempt_preexisted = attempt_path.exists()
        if attempt_preexisted:
            if load_json(attempt_path) != attempt:
                raise RuntimeError(
                    "owned_core authoring attempt conflicts with prior request"
                )

        published_paths = (output_path, diagnostics_path, result_path)
        present_paths = tuple(
            path for path in published_paths if path.exists() or path.is_symlink()
        )
        if present_paths and not attempt_preexisted:
            raise RuntimeError(
                "owned_core artifacts exist without a matching prior authoring "
                "attempt; refusing to relabel stale output"
            )
        invalid_paths = tuple(
            path for path in present_paths if path.is_symlink() or not path.is_file()
        )
        if invalid_paths:
            raise _ArticulationEvidenceBindingError(
                "owned_core publication contains invalid artifact entries: "
                + ", ".join(path.name for path in invalid_paths)
            )
        if not attempt_preexisted:
            atomic_write_json(attempt_path, attempt)
        output_published = output_path.is_file()
        diagnostics_published = diagnostics_path.is_file()
        result_published = result_path.is_file()
        if output_published and diagnostics_published and result_published:
            return _owned_core_authoring_result(
                request,
                output_path=output_path,
                diagnostics_path=diagnostics_path,
                result_path=result_path,
                recovered=True,
            )
        # The shared transaction promotes diagnostics, then result, then the
        # root as its commit point. Either report-only prefix is safe to retry.
        report_only_prefix = not output_published and diagnostics_published
        incomplete_publication = any(
            (output_published, diagnostics_published, result_published)
        )
        if incomplete_publication and not report_only_prefix:
            raise _ArticulationEvidenceBindingError(
                "owned_core publication state is not a recoverable root-last "
                "report prefix"
            )

        _validate_authoring_prediction_binding(request)
        if accepted_plan is not None:
            from world_understanding.functions.physics.joint_rigger import (
                JointRiggerArtifactTargets,
                author_joint_topology,
            )

            prepared_source = _prepare_accepted_membership_derivative(
                request,
                accepted_plan,
            )
            execution_request = _accepted_execution_request(request, accepted_plan)
            author_joint_topology(
                execution_request,
                source_usd_path=prepared_source,
                artifact_targets=JointRiggerArtifactTargets(
                    output_path=output_path,
                    diagnostics_path=diagnostics_path,
                    result_path=result_path,
                ),
            )
        else:
            try:
                from joint_agent.functions.joint_rigger_adapter import (
                    apply_joint_rigger,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Joint graph authoring requires the joint-agent package in the "
                    "active Python environment"
                ) from exc

            result = apply_joint_rigger(
                input_usd_path=request.source_asset,
                predictions_path=request.predictions_path,
                output_usd_path=output_path,
                diagnostics_path=diagnostics_path,
                validation_path=result_path,
                articulation_candidates_path=request.candidate_document_path,
                adapter="owned_core",
                on_missing_dependency="block",
                on_unready_candidates="block",
                apply_masses=False,
                apply_collision=False,
            )
            if result.get("joint_rigger_status") != "authored":
                raise RuntimeError(
                    "owned_core did not publish the accepted articulation candidates: "
                    f"{result.get('joint_rigger_status')}"
                )
        return _owned_core_authoring_result(
            request,
            output_path=output_path,
            diagnostics_path=diagnostics_path,
            result_path=result_path,
            recovered=False,
        )

    def validate(
        self,
        authoring: ArticulationAuthoringResult,
        *,
        expected_candidate_ids: tuple[str, ...],
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationValidationResult:
        if cancel_checker is not None and cancel_checker():
            raise asyncio.CancelledError
        if authoring.joint_rigger_result_path is None:
            raise RuntimeError("owned_core result artifact is required for readback")

        from world_understanding.functions.physics.joint_rigger import (
            JointRiggerDiagnosticsV1,
            JointRiggerResultV1,
        )

        result_bytes = _read_bound_bytes(
            authoring.joint_rigger_result_path,
            authoring.joint_rigger_result_sha256,
            label="Joint Rigger result",
        )
        diagnostics_bytes = None
        if authoring.diagnostics_path is not None:
            diagnostics_bytes = _read_bound_bytes(
                authoring.diagnostics_path,
                authoring.diagnostics_sha256,
                label="Joint Rigger diagnostics",
            )
        result = JointRiggerResultV1.model_validate_json(result_bytes)
        failures: list[str] = []
        if diagnostics_bytes is not None:
            bound_diagnostics = JointRiggerDiagnosticsV1.model_validate_json(
                diagnostics_bytes
            )
            if result.diagnostics != bound_diagnostics:
                failures.append(
                    "Joint Rigger result and bound diagnostics artifact disagree."
                )
        observed_output_sha256 = file_sha256(authoring.output_asset_path)
        if observed_output_sha256 != authoring.output_asset_sha256:
            failures.append("Published USDZ digest changed before validation.")
        candidate_path = Path(authoring.candidate_document_path)
        candidate_document = Stage2CandidateDocument.model_validate_json(
            _read_bound_bytes(
                candidate_path,
                authoring.candidate_document_sha256,
                label="approved candidate document",
            )
        )
        if candidate_document.candidate_ids != expected_candidate_ids:
            failures.append(
                "Approved candidate document differs from expected candidate IDs."
            )
        observed_dependency_bundle_sha256: str | None = None
        diagnostic_joint_ids = tuple(
            diagnostic.joint_id for diagnostic in result.diagnostics.joint_diagnostics
        )
        has_bound_joint_request = "joint_rigger_input" in authoring.metadata
        expected_joint_request: Any | None = None
        expected_topology_plan = None
        # Metadata-less artifacts remain readable for legacy compatibility, but
        # their result-owned plan hash is not source-bound. Surface that weaker
        # validation mode in the returned metadata.
        expected_topology_plan_sha256 = result.plan_sha256
        expected_physics_state: _PhysicsStageState | None = None
        request_identity_failures: list[str] = []
        if has_bound_joint_request:
            from world_understanding.functions.physics.joint_rigger import (
                JointRiggerInputV1,
                JointRiggerInputV2,
                canonical_sha256,
            )

            try:
                joint_request_payload = authoring.metadata.get("joint_rigger_input")
                if not isinstance(joint_request_payload, Mapping):
                    raise ValueError("joint_rigger_input must be a mapping")
                input_schema_version = joint_request_payload.get("schema_version")
                joint_request_json = json.dumps(
                    joint_request_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if input_schema_version == "world-understanding-joint-rigger-input-v2":
                    expected_joint_request = JointRiggerInputV2.model_validate_json(
                        joint_request_json
                    )
                elif (
                    input_schema_version == "world-understanding-joint-rigger-input-v1"
                ):
                    expected_joint_request = JointRiggerInputV1.model_validate_json(
                        joint_request_json
                    )
                else:
                    raise ValueError(
                        "joint_rigger_input has an unsupported schema_version"
                    )
                observed_input_sha256 = canonical_sha256(expected_joint_request)
                observed_plan_sha256 = canonical_sha256(expected_joint_request.plan)
                if (
                    authoring.metadata.get("joint_rigger_input_sha256")
                    != observed_input_sha256
                ):
                    request_identity_failures.append(
                        "Bound Joint Rigger input digest differs from its payload."
                    )
                if (
                    authoring.metadata.get("joint_rigger_plan_sha256")
                    != observed_plan_sha256
                ):
                    request_identity_failures.append(
                        "Bound Joint Rigger plan digest differs from its payload."
                    )
                if result.input_sha256 != observed_input_sha256:
                    request_identity_failures.append(
                        "Joint Rigger result input identity differs from the exact "
                        "bound request."
                    )
                if result.plan_sha256 != observed_plan_sha256:
                    request_identity_failures.append(
                        "Joint Rigger result plan identity differs from the exact "
                        "bound request."
                    )
                expected_topology_plan = _project_owned_core_topology_plan(
                    expected_joint_request
                )
                expected_topology_plan_sha256 = canonical_sha256(expected_topology_plan)
                expected_physics_state = _parse_physics_state_payload(
                    authoring.metadata.get("physics_api_inventory")
                )
                if authoring.metadata.get(
                    "physics_api_inventory_sha256"
                ) != _canonical_sha256(_physics_state_payload(expected_physics_state)):
                    request_identity_failures.append(
                        "Bound physics API inventory digest differs from its payload."
                    )
            except (TypeError, ValueError) as exc:
                request_identity_failures.append(
                    f"Bound Joint Rigger request metadata is invalid: {exc}"
                )
                expected_joint_request = None
                expected_topology_plan = None
                expected_physics_state = None
        failures.extend(request_identity_failures)
        if result.status != "succeeded" or result.output_artifact is None:
            failures.append("Joint Rigger result does not claim succeeded output.")
            diagnostic_joint_paths: tuple[str, ...] = ()
            diagnostic_bindings: tuple[_OwnedCoreDiagnosticBinding, ...] = ()
            diagnostic_binding_failures: tuple[str, ...] = ()
            self_contained = False
        else:
            (
                diagnostic_bindings,
                diagnostic_joint_paths,
                diagnostic_binding_failures,
            ) = _resolve_owned_core_diagnostic_bindings(
                candidate_document,
                joint_diagnostics=result.diagnostics.joint_diagnostics,
                plan_sha256=expected_topology_plan_sha256,
                expected_topology_plan=expected_topology_plan,
                backend_name=result.diagnostics.backend_name,
                backend_version=result.diagnostics.backend_version,
            )
            failures.extend(diagnostic_binding_failures)
            identity_validator = (
                _validate_self_contained_raw_usd_identity
                if authoring.membership_operation_receipt_path is not None
                else _validate_sealed_usdz_identity
            )
            (
                self_contained,
                observed_dependency_bundle_sha256,
                package_failures,
            ) = identity_validator(
                Path(authoring.output_asset_path),
                expected_root_sha256=result.output_artifact.root_sha256,
                expected_dependency_bundle_sha256=(
                    result.output_artifact.dependency_bundle_sha256
                ),
            )
            failures.extend(package_failures)

        validated_ids, graph_failures, managed_joint_paths = (
            _validate_saved_owned_core_graph(
                Path(authoring.output_asset_path),
                candidate_document,
                diagnostic_joint_paths=diagnostic_joint_paths,
                diagnostic_bindings=diagnostic_bindings,
                expected_joint_request=expected_joint_request,
                expected_diagnostics=result.diagnostics,
                expected_physics_state=expected_physics_state,
                require_stage2_candidate_custom_data=(
                    result.diagnostics.backend_name == "stage2_candidate_edges"
                ),
            )
        )
        failures.extend(graph_failures)
        exact_graph_match = bool(
            not graph_failures
            and not diagnostic_binding_failures
            and not request_identity_failures
            and validated_ids == expected_candidate_ids
            and candidate_document.candidate_ids == expected_candidate_ids
        )
        passed = not failures
        return ArticulationValidationResult(
            status="pass" if passed else "fail",
            output_asset_path=authoring.output_asset_path,
            expected_output_asset_sha256=authoring.output_asset_sha256,
            observed_output_asset_sha256=observed_output_sha256,
            expected_candidate_ids=expected_candidate_ids,
            validated_candidate_ids=validated_ids,
            exact_graph_match=exact_graph_match,
            self_contained=self_contained,
            failures=tuple(failures),
            evidence_paths=tuple(
                path
                for path in (
                    authoring.candidate_document_path,
                    authoring.diagnostics_path,
                    authoring.joint_rigger_result_path,
                    authoring.membership_operation_receipt_path,
                )
                if path is not None
            ),
            metadata={
                "backend": "joint-agent-owned-core",
                "joint_rigger_input_sha256": result.input_sha256,
                "joint_rigger_plan_sha256": result.plan_sha256,
                "expected_dependency_bundle_sha256": (
                    result.output_artifact.dependency_bundle_sha256
                    if result.output_artifact is not None
                    else None
                ),
                "observed_dependency_bundle_sha256": (
                    observed_dependency_bundle_sha256
                ),
                "joint_diagnostic_ids": diagnostic_joint_ids,
                "joint_diagnostic_paths": diagnostic_joint_paths,
                "saved_managed_joint_paths": managed_joint_paths,
                "live_backend_invoked": True,
                "request_identity_source_bound": has_bound_joint_request,
            },
        )


class JointAgentGraphAuthoringClient(_JointAgentClientImplementation):
    """Deterministic graph-only adapter with no inference configuration or call."""

    def __init__(self) -> None:
        # ``author`` and ``validate`` use only their exact typed requests.  The
        # private base retains the shared implementation without constructing a
        # JointAgentLocalClient or loading a model/service configuration.
        super().__init__({})

    def configuration_sha256(
        self,
        request: ArticulationWorkflowRequest,
    ) -> str:
        del request
        raise RuntimeError("Graph-only Articulation authoring has no inference config")

    def infer(
        self,
        request: ArticulationWorkflowRequest,
        *,
        resume: bool,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationInferenceResult:
        del request, resume, cancel_checker
        raise RuntimeError("Graph-only Articulation authoring cannot run inference")


class JointAgentLocalClient(_JointAgentClientImplementation):
    """Configured Joint Agent inference plus deterministic owned-core authoring."""


__all__ = [
    "ArticulationAuthoringClient",
    "ArticulationWorkflowClient",
    "CancelChecker",
    "JointAgentGraphAuthoringClient",
    "JointAgentLocalClient",
    "JointAgentInferenceTerminalError",
    "MockArticulationCall",
    "MockArticulationWorkflowClient",
]
