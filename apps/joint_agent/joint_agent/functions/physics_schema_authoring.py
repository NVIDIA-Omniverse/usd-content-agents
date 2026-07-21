# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, evidence-backed post-rigger physics schema authoring.

This module is deliberately downstream of ``stage2_candidate_edges``. It reads
that adapter's diagnostics as an identity manifest, verifies the authored joint
graph in the input USD, and only then adds explicitly planned physics schemas.
It never reads articulation candidates and never translates or reshapes their
graph.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    ValidationError,
    field_validator,
    model_validator,
)

PLAN_SCHEMA_VERSION = "joint-agent-physics-authoring-plan-v0"
DIAGNOSTICS_SCHEMA_VERSION = "joint-agent-physics-authoring-diagnostics-v0"
VALIDATION_SCHEMA_VERSION = "joint-agent-physics-authoring-validation-v0"
STAGE2_DIAGNOSTICS_SCHEMA_VERSION = "joint-agent-rigger-diagnostics-v0"
STAGE2_AUTHORING_SCHEMA_VERSION = "joint-agent-stage2-candidate-edge-authoring-v0"
STAGE2_SCHEMA_VERSION = "joint-agent-stage2-v0"
STAGE2_ADAPTER = "stage2_candidate_edges"

_RAW_USD_EXTENSIONS = {".usd", ".usda", ".usdc"}
_SUPPORTED_USD_EXTENSIONS = _RAW_USD_EXTENSIONS | {".usdz"}
_SHA256_LENGTH = 64
_MESH_APPROXIMATIONS = {"convexHull", "convexDecomposition", "sdf"}
_MIMIC_AXES = {"rotX", "rotY", "rotZ"}
_EVIDENCE_SOURCES = {
    "accepted_manifest",
    "authored_reference",
    "source_metadata",
    "owner_approved_plan",
    "stage2_graph_root",
    "gate2_rest_pose",
    "profile_requirement",
}


class PhysicsAuthoringError(ValueError):
    """A fail-closed authoring error with a machine-readable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.detail = message


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class _EvidenceModel(_StrictModel):
    source: str = Field(min_length=1)
    evidence: str = Field(min_length=1)

    @field_validator("source", "evidence")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("source")
    @classmethod
    def _approved_source(cls, value: str) -> str:
        if value not in _EVIDENCE_SOURCES:
            raise ValueError(
                "must use an approved provenance source: "
                + ", ".join(sorted(_EVIDENCE_SOURCES))
            )
        return value


class MassPlan(_EvidenceModel):
    mode: Literal["preserve", "author"]
    mass_kg: StrictFloat | None = None
    diagonal_inertia_kg_m2: tuple[StrictFloat, StrictFloat, StrictFloat] | None = None
    principal_axes: (
        tuple[
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
        ]
        | None
    ) = None
    replace_invalid_existing: StrictBool = False

    @model_validator(mode="after")
    def _validate_mode(self) -> MassPlan:
        if self.mode == "author":
            if self.mass_kg is None or self.diagonal_inertia_kg_m2 is None:
                raise ValueError(
                    "author mode requires mass_kg and diagonal_inertia_kg_m2"
                )
            _require_positive_finite(self.mass_kg, "mass_kg")
            for index, value in enumerate(self.diagonal_inertia_kg_m2):
                _require_positive_finite(
                    value,
                    f"diagonal_inertia_kg_m2[{index}]",
                )
            _require_inertia_triangle(
                self.diagonal_inertia_kg_m2,
                "diagonal_inertia_kg_m2",
            )
            if self.principal_axes is not None:
                _require_normalized_quaternion(
                    self.principal_axes,
                    "principal_axes",
                )
            if self.replace_invalid_existing and self.source not in {
                "owner_approved_plan",
                "accepted_manifest",
            }:
                raise ValueError(
                    "replace_invalid_existing requires owner_approved_plan or "
                    "accepted_manifest provenance"
                )
        elif any(
            value is not None
            for value in (
                self.mass_kg,
                self.diagonal_inertia_kg_m2,
                self.principal_axes,
            )
        ):
            raise ValueError("preserve mode cannot carry replacement mass opinions")
        elif self.replace_invalid_existing:
            raise ValueError("replace_invalid_existing is only valid for author mode")
        return self


class ColliderPlan(_EvidenceModel):
    prim_path: str = Field(min_length=1)
    mode: Literal["preserve", "author"]
    mesh_approximation: Literal["convexHull", "convexDecomposition", "sdf"] | None = (
        None
    )

    @model_validator(mode="after")
    def _validate_mode(self) -> ColliderPlan:
        if self.mode == "preserve" and self.mesh_approximation is not None:
            raise ValueError(
                "preserve mode cannot carry a replacement mesh_approximation"
            )
        return self


class BodyPlan(_StrictModel):
    prim_path: str = Field(min_length=1)
    mass: MassPlan
    colliders: list[ColliderPlan] = Field(min_length=1)
    nested_body_transform: Literal["preserve", "preserve_world_reset"] = "preserve"

    @model_validator(mode="after")
    def _unique_colliders(self) -> BodyPlan:
        paths = [collider.prim_path for collider in self.colliders]
        if len(paths) != len(set(paths)):
            raise ValueError("collider prim_path values must be unique per body")
        return self


class JointStatePlan(_EvidenceModel):
    mode: Literal["preserve", "rest_zero", "not_applicable"]


class JointControlPlan(_EvidenceModel):
    mode: Literal["preserve", "passive", "drive", "mimic"]

    drive_type: Literal["force", "acceleration"] | None = Field(
        default=None,
        alias="type",
    )
    stiffness: StrictFloat | None = None
    damping: StrictFloat | None = None
    max_force: StrictFloat | None = None
    target_position: StrictFloat | None = None
    target_velocity: StrictFloat | None = None
    max_joint_velocity: StrictFloat | None = None

    axis: Literal["rotX", "rotY", "rotZ"] | None = None
    reference_candidate_id: str | None = None
    reference_axis: Literal["rotX", "rotY", "rotZ"] | None = None
    gearing: StrictFloat | None = None
    offset: StrictFloat | None = None
    natural_frequency: StrictFloat | None = None
    damping_ratio: StrictFloat | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> JointControlPlan:
        drive_values = (
            self.drive_type,
            self.stiffness,
            self.damping,
            self.max_force,
            self.target_position,
            self.target_velocity,
            self.max_joint_velocity,
        )
        mimic_values = (
            self.axis,
            self.reference_candidate_id,
            self.reference_axis,
            self.gearing,
            self.offset,
            self.natural_frequency,
            self.damping_ratio,
        )
        if self.mode == "drive":
            required = drive_values[:6]
            if any(value is None for value in required):
                raise ValueError(
                    "drive mode requires type, stiffness, damping, max_force, "
                    "target_position, and target_velocity"
                )
            assert self.stiffness is not None
            assert self.damping is not None
            assert self.max_force is not None
            assert self.target_position is not None
            assert self.target_velocity is not None
            _require_positive_finite(self.stiffness, "stiffness")
            _require_nonnegative_finite(self.damping, "damping")
            _require_positive_finite(self.max_force, "max_force")
            if not math.isclose(self.target_position, 0.0, abs_tol=0.0):
                raise ValueError("drive target_position must be exactly 0 in v0")
            if not math.isclose(self.target_velocity, 0.0, abs_tol=0.0):
                raise ValueError("drive target_velocity must be exactly 0 in v0")
            if self.max_joint_velocity is not None:
                _require_positive_finite(
                    self.max_joint_velocity,
                    "max_joint_velocity",
                )
            if any(value is not None for value in mimic_values):
                raise ValueError("drive mode cannot carry mimic fields")
        elif self.mode == "mimic":
            if any(value is not None for value in drive_values):
                raise ValueError("mimic mode cannot carry drive fields")
            if any(value is None for value in mimic_values):
                raise ValueError(
                    "mimic mode requires axis, reference_candidate_id, "
                    "reference_axis, gearing, offset, natural_frequency, and "
                    "damping_ratio"
                )
            assert self.reference_candidate_id is not None
            assert self.gearing is not None
            assert self.offset is not None
            assert self.natural_frequency is not None
            assert self.damping_ratio is not None
            if not self.reference_candidate_id.strip():
                raise ValueError("reference_candidate_id must not be blank")
            _require_finite(self.gearing, "gearing")
            if math.isclose(self.gearing, 0.0, abs_tol=0.0):
                raise ValueError("gearing must be nonzero")
            _require_finite(self.offset, "offset")
            _require_positive_finite(
                self.natural_frequency,
                "natural_frequency",
            )
            _require_nonnegative_finite(self.damping_ratio, "damping_ratio")
        elif any(value is not None for value in (*drive_values, *mimic_values)):
            raise ValueError(
                f"{self.mode} mode cannot carry drive or mimic value opinions"
            )
        return self


class JointPlan(_EvidenceModel):
    candidate_id: str = Field(min_length=1)
    state: JointStatePlan
    control: JointControlPlan


class ArticulationRootPlan(_EvidenceModel):
    prim_path: str = Field(min_length=1)


class PhysicsAuthoringPlan(_StrictModel):
    schema_version: Literal["joint-agent-physics-authoring-plan-v0"]
    expected_input_sha256: str | None = None
    expected_stage2_diagnostics_sha256: str | None = None
    expected_stage2_validation_sha256: str | None = None
    articulation_root: ArticulationRootPlan
    bodies: list[BodyPlan] = Field(min_length=1)
    joints: list[JointPlan] = Field(min_length=1)
    conflict_policy: Literal["error"] = "error"

    @field_validator(
        "expected_input_sha256",
        "expected_stage2_diagnostics_sha256",
        "expected_stage2_validation_sha256",
    )
    @classmethod
    def _valid_sha256(cls, value: str | None) -> str | None:
        if value is not None:
            _require_sha256(value)
        return value

    @model_validator(mode="after")
    def _unique_targets(self) -> PhysicsAuthoringPlan:
        body_paths = [body.prim_path for body in self.bodies]
        if len(body_paths) != len(set(body_paths)):
            raise ValueError("body prim_path values must be unique")
        candidate_ids = [joint.candidate_id for joint in self.joints]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("joint candidate_id values must be unique")
        collider_paths = [
            collider.prim_path for body in self.bodies for collider in body.colliders
        ]
        if len(collider_paths) != len(set(collider_paths)):
            raise ValueError("a collider prim_path may belong to only one body")
        return self


@dataclass(frozen=True)
class _Preflight:
    stage2_diagnostics: dict[str, Any]
    stage2_diagnostics_path: Path
    stage2_diagnostics_sha256: str
    stage2_validation: dict[str, Any]
    stage2_validation_path: Path
    stage2_validation_sha256: str
    edges: tuple[dict[str, Any], ...]
    edge_by_candidate: dict[str, dict[str, Any]]
    graph_root: str
    graph_snapshot: dict[str, Any]
    default_prim_path: str
    prim_paths: tuple[str, ...]
    stage_metadata: dict[str, Any]
    before_schema_snapshot: dict[str, Any]
    before_rule_snapshot: dict[str, Any]
    before_world_matrices: dict[str, list[float]]
    allowed_schema_additions: dict[str, set[str]]


def _author_physics_schemas_impl(
    *,
    input_usd_path: str | Path,
    stage2_diagnostics_path: str | Path,
    stage2_validation_path: str | Path,
    authoring_plan_path: str | Path,
    output_usd_path: str | Path,
    diagnostics_path: str | Path,
    validation_path: str | Path,
) -> dict[str, Any]:
    """Apply an explicit post-rigger physics authoring plan atomically."""

    input_path = Path(input_usd_path)
    stage2_diagnostics = Path(stage2_diagnostics_path)
    stage2_validation = Path(stage2_validation_path)
    plan_path = Path(authoring_plan_path)
    output_path = Path(output_usd_path)
    diagnostics = Path(diagnostics_path)
    validation = Path(validation_path)
    _validate_artifact_paths(
        input_path=input_path,
        stage2_diagnostics_path=stage2_diagnostics,
        stage2_validation_path=stage2_validation,
        plan_path=plan_path,
        output_path=output_path,
        diagnostics_path=diagnostics,
        validation_path=validation,
    )

    raw_plan, plan = _load_plan(plan_path)
    plan_sha256 = _canonical_json_sha256(raw_plan)
    plan_file_sha256 = _file_sha256(plan_path)
    input_sha256 = _file_sha256(input_path)
    input_dependency_manifest = _dependency_manifest(input_path)
    input_hash_mismatch = (
        plan.expected_input_sha256 is not None
        and plan.expected_input_sha256 != input_sha256
    )

    preflight = _preflight_input(
        input_path=input_path,
        stage2_diagnostics_path=stage2_diagnostics,
        stage2_validation_path=stage2_validation,
        plan=plan,
        plan_sha256=plan_sha256,
    )
    if input_hash_mismatch and not _is_owned_derivative(
        input_path,
        plan_sha256=plan_sha256,
        stage2_diagnostics_sha256=preflight.stage2_diagnostics_sha256,
        stage2_validation_sha256=preflight.stage2_validation_sha256,
    ):
        raise PhysicsAuthoringError(
            "artifact_identity_mismatch",
            "input USD SHA256 does not match expected_input_sha256",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = _temporary_output_path(output_path)
    preparation_dir: Path | None = None
    try:
        editable_path, preparation_dir = _prepare_editable_output(
            input_path,
            temp_output,
        )
        stage = _open_stage(editable_path, label="editable physics output USD")
        operation_records = _apply_plan(
            stage,
            plan=plan,
            preflight=preflight,
            plan_sha256=plan_sha256,
        )
        if not stage.GetRootLayer().Save():
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                f"could not save authored USD layer: {editable_path}",
            )
        del stage

        if output_path.suffix.lower() == ".usdz":
            try:
                _package_usdz(editable_path, temp_output)
            except Exception as exc:
                raise PhysicsAuthoringError("package_error", str(exc)) from exc

        validation_checks = _validate_authored_output(
            temp_output,
            plan=plan,
            preflight=preflight,
        )
        output_sha256 = _file_sha256(temp_output)
        output_dependency_manifest = _dependency_manifest(
            temp_output,
            logical_root_path=output_path,
        )
        validation_checks["dependency_identity_preserved"] = (
            _validate_dependency_identity_preserved(
                input_dependency_manifest,
                output_dependency_manifest,
            )
        )
        semantic_output_sha256 = _canonical_json_sha256(
            validation_checks["after_schema_snapshot"]
        )
    except Exception:
        if temp_output.exists():
            _unlink_temporary(temp_output)
        raise
    finally:
        if preparation_dir is not None:
            shutil.rmtree(preparation_dir, ignore_errors=True)

    identity = {
        "input_usd_path": str(input_path),
        "input_sha256": input_sha256,
        "input_hash_scope": (
            "package" if input_path.suffix.lower() == ".usdz" else "root_layer"
        ),
        "input_dependency_manifest": input_dependency_manifest,
        "input_dependency_manifest_sha256": _canonical_json_sha256(
            input_dependency_manifest
        ),
        "authoring_plan_path": str(plan_path),
        "authoring_plan_canonical_sha256": plan_sha256,
        "authoring_plan_file_sha256": plan_file_sha256,
        "stage2_diagnostics_path": str(preflight.stage2_diagnostics_path),
        "stage2_diagnostics_sha256": preflight.stage2_diagnostics_sha256,
        "stage2_validation_path": str(preflight.stage2_validation_path),
        "stage2_validation_sha256": preflight.stage2_validation_sha256,
        "output_usd_path": str(output_path),
        "output_sha256": output_sha256,
        "output_hash_scope": (
            "package" if output_path.suffix.lower() == ".usdz" else "root_layer"
        ),
        "output_dependency_manifest": output_dependency_manifest,
        "output_dependency_manifest_sha256": _canonical_json_sha256(
            output_dependency_manifest
        ),
        "semantic_output_sha256": semantic_output_sha256,
        "default_prim_path": preflight.default_prim_path,
        "stage2_adapter": STAGE2_ADAPTER,
        "stage2_authoring_schema_version": STAGE2_AUTHORING_SCHEMA_VERSION,
    }
    diagnostics_payload = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "status": "authored",
        "identity": identity,
        "graph_root": preflight.graph_root,
        "operation_records": operation_records,
        "schema_deltas": validation_checks["schema_deltas"],
        "structural_rule_deltas": validation_checks["structural_rule_deltas"],
        "expected_residuals": validation_checks["expected_residuals"],
        "semantic_change_count": validation_checks["semantic_change_count"],
        "warnings": validation_checks["warnings"],
        "errors": [],
    }
    validation_payload = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "status": "passed",
        "identity": identity,
        "checks": {
            key: value
            for key, value in validation_checks.items()
            if key
            not in {
                "after_schema_snapshot",
                "schema_deltas",
                "structural_rule_deltas",
                "expected_residuals",
                "semantic_change_count",
                "warnings",
            }
        },
        "schema_deltas": validation_checks["schema_deltas"],
        "structural_rule_deltas": validation_checks["structural_rule_deltas"],
        "expected_residuals": validation_checks["expected_residuals"],
        "warnings": validation_checks["warnings"],
        "errors": [],
    }
    staged_diagnostics: Path | None = None
    staged_validation: Path | None = None
    try:
        staged_diagnostics = _stage_json_file(diagnostics, diagnostics_payload)
        staged_validation = _stage_json_file(validation, validation_payload)
        _commit_staged_artifacts(
            (
                (temp_output, output_path),
                (staged_diagnostics, diagnostics),
                (staged_validation, validation),
            )
        )
    finally:
        for temporary in (temp_output, staged_diagnostics, staged_validation):
            if temporary is not None and temporary.exists():
                _unlink_temporary(temporary)

    return {
        "physics_authoring_status": "authored",
        "physics_ready_usd_path": str(output_path),
        "physics_authoring_diagnostics_path": str(diagnostics),
        "physics_authoring_validation_path": str(validation),
        "physics_authoring_identity": identity,
        "physics_authoring_schema_deltas": validation_checks["schema_deltas"],
        "physics_authoring_rule_deltas": validation_checks["structural_rule_deltas"],
        "physics_authoring_expected_residuals": validation_checks["expected_residuals"],
    }


def author_physics_schemas(
    *,
    input_usd_path: str | Path,
    stage2_diagnostics_path: str | Path,
    stage2_validation_path: str | Path,
    authoring_plan_path: str | Path,
    output_usd_path: str | Path,
    diagnostics_path: str | Path,
    validation_path: str | Path,
) -> dict[str, Any]:
    """Apply a plan and persist its fail-closed reason taxonomy on failure."""

    output_path = Path(output_usd_path)
    try:
        return _author_physics_schemas_impl(
            input_usd_path=input_usd_path,
            stage2_diagnostics_path=stage2_diagnostics_path,
            stage2_validation_path=stage2_validation_path,
            authoring_plan_path=authoring_plan_path,
            output_usd_path=output_usd_path,
            diagnostics_path=diagnostics_path,
            validation_path=validation_path,
        )
    except PhysicsAuthoringError as exc:
        error = exc
    except FileNotFoundError as exc:
        missing_path = Path(exc.filename) if exc.filename else None
        if missing_path in {
            Path(stage2_diagnostics_path),
            Path(stage2_validation_path),
        }:
            code = "upstream_gate2_not_passed"
        elif missing_path == Path(authoring_plan_path):
            code = "authoring_plan_invalid"
        else:
            code = "artifact_identity_mismatch"
        error = PhysicsAuthoringError(code, str(exc))
    except Exception as exc:
        error = PhysicsAuthoringError("authoring_failed", str(exc))

    _persist_failure_artifacts(
        error,
        input_path=Path(input_usd_path),
        stage2_diagnostics_path=Path(stage2_diagnostics_path),
        stage2_validation_path=Path(stage2_validation_path),
        plan_path=Path(authoring_plan_path),
        output_path=output_path,
        diagnostics_path=Path(diagnostics_path),
        validation_path=Path(validation_path),
    )
    raise error


def _validate_existing_output_artifact(output_path: Path) -> None:
    if output_path.exists() or output_path.is_symlink():
        if output_path.is_symlink():
            raise PhysicsAuthoringError(
                "authoring_plan_invalid",
                f"refusing to replace output USD symlink: {output_path}",
            )
        if output_path.is_dir() and not output_path.is_symlink():
            raise PhysicsAuthoringError(
                "authoring_plan_invalid",
                f"output USD path is a directory: {output_path}",
            )
        if not _is_owned_output_artifact(output_path):
            raise PhysicsAuthoringError(
                "artifact_identity_mismatch",
                f"refusing to replace unowned output USD: {output_path}",
            )


def _is_owned_output_artifact(path: Path) -> bool:
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(str(path))
        if stage is None:
            return False
        default_prim = stage.GetDefaultPrim()
        if not default_prim:
            return False
        stamps = (
            default_prim.GetCustomDataByKey("jointAgent:physicsAuthoringPlanSha256"),
            default_prim.GetCustomDataByKey(
                "jointAgent:physicsStage2DiagnosticsSha256"
            ),
            default_prim.GetCustomDataByKey("jointAgent:physicsStage2ValidationSha256"),
        )
        return all(
            isinstance(value, str)
            and len(value) == _SHA256_LENGTH
            and all(character in "0123456789abcdef" for character in value)
            for value in stamps
        )
    except Exception:
        return False


def _persist_failure_artifacts(
    error: PhysicsAuthoringError,
    *,
    input_path: Path,
    stage2_diagnostics_path: Path,
    stage2_validation_path: Path,
    plan_path: Path,
    output_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
) -> None:
    if error.code == "artifact_write_failed":
        return
    if output_path.exists() or output_path.is_symlink():
        return
    resolved_outputs = {
        diagnostics_path.expanduser().resolve(),
        validation_path.expanduser().resolve(),
    }
    protected = {
        input_path.expanduser().resolve(),
        stage2_diagnostics_path.expanduser().resolve(),
        stage2_validation_path.expanduser().resolve(),
        plan_path.expanduser().resolve(),
        output_path.expanduser().resolve(),
    }
    if len(resolved_outputs) != 2 or resolved_outputs & protected:
        return
    for path, schema_version in (
        (diagnostics_path, DIAGNOSTICS_SCHEMA_VERSION),
        (validation_path, VALIDATION_SCHEMA_VERSION),
    ):
        if path.exists() and not _is_owned_report_artifact(
            path,
            schema_version=schema_version,
            output_path=output_path,
        ):
            return
    status = (
        "failed"
        if error.code
        in {
            "package_error",
            "postwrite_validation_failed",
            "authoring_failed",
            "artifact_write_failed",
        }
        else "blocked"
    )

    def identity(path: Path) -> str | None:
        try:
            return _file_sha256(path) if path.is_file() else None
        except OSError:
            return None

    evidence_identity = {
        "input_usd_path": str(input_path),
        "input_sha256": identity(input_path),
        "input_hash_scope": (
            "package" if input_path.suffix.lower() == ".usdz" else "root_layer"
        ),
        "authoring_plan_path": str(plan_path),
        "authoring_plan_canonical_sha256": _optional_canonical_json_sha256(plan_path),
        "authoring_plan_file_sha256": identity(plan_path),
        "stage2_diagnostics_path": str(stage2_diagnostics_path),
        "stage2_diagnostics_sha256": identity(stage2_diagnostics_path),
        "stage2_validation_path": str(stage2_validation_path),
        "stage2_validation_sha256": identity(stage2_validation_path),
        "output_usd_path": str(output_path),
    }
    shared = {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "status": status,
        "reason_code": error.code,
        "identity": evidence_identity,
        "warnings": [],
        "errors": [error.detail],
    }
    payloads = (
        (
            diagnostics_path,
            {"schema_version": DIAGNOSTICS_SCHEMA_VERSION, **shared},
        ),
        (
            validation_path,
            {"schema_version": VALIDATION_SCHEMA_VERSION, **shared},
        ),
    )
    staged: list[tuple[Path, Path]] = []
    try:
        for path, payload in payloads:
            staged.append((_stage_json_file(path, payload), path))
        _commit_staged_artifacts(tuple(staged))
    except (OSError, PhysicsAuthoringError):
        pass
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                _unlink_temporary(temporary)


def _load_plan(path: Path) -> tuple[dict[str, Any], PhysicsAuthoringPlan]:
    raw = _load_json_object(path, label="physics authoring plan")
    try:
        plan = PhysicsAuthoringPlan.model_validate(raw)
    except ValidationError as exc:
        raise PhysicsAuthoringError("authoring_plan_invalid", str(exc)) from exc
    return raw, plan


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        code = (
            "authoring_plan_invalid"
            if label == "physics authoring plan"
            else "upstream_gate2_not_passed"
        )
        raise PhysicsAuthoringError(code, f"{label} not found: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_object_without_duplicates)
    except PhysicsAuthoringError as exc:
        if label == "physics authoring plan":
            raise
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"invalid duplicate key in {label}: {exc.detail}",
        ) from exc
    except json.JSONDecodeError as exc:
        code = (
            "authoring_plan_invalid"
            if label == "physics authoring plan"
            else "upstream_gate2_not_passed"
        )
        raise PhysicsAuthoringError(
            code,
            f"invalid JSON in {label} {path}: {exc.msg}",
        ) from exc
    if not isinstance(value, dict):
        code = (
            "authoring_plan_invalid"
            if label == "physics authoring plan"
            else "upstream_gate2_not_passed"
        )
        raise PhysicsAuthoringError(
            code,
            f"{label} must be a JSON object",
        )
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhysicsAuthoringError(
                "authoring_plan_invalid",
                f"duplicate JSON key: {key}",
            )
        result[key] = value
    return result


def _preflight_input(
    *,
    input_path: Path,
    stage2_diagnostics_path: Path,
    stage2_validation_path: Path,
    plan: PhysicsAuthoringPlan,
    plan_sha256: str,
) -> _Preflight:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage2_path = stage2_diagnostics_path
    if not stage2_path.is_file() or not stage2_validation_path.is_file():
        missing = [
            str(path)
            for path in (stage2_path, stage2_validation_path)
            if not path.is_file()
        ]
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"required Stage 2 evidence artifact is missing: {missing}",
        )
    stage2_sha256 = _file_sha256(stage2_path)
    if (
        plan.expected_stage2_diagnostics_sha256 is not None
        and plan.expected_stage2_diagnostics_sha256 != stage2_sha256
    ):
        raise PhysicsAuthoringError(
            "artifact_identity_mismatch",
            "Stage 2 diagnostics SHA256 does not match the plan",
        )
    stage2 = _load_json_object(stage2_path, label="Stage 2 diagnostics")
    _validate_stage2_diagnostics_header(stage2)
    stage2_validation_sha256 = _file_sha256(stage2_validation_path)
    if (
        plan.expected_stage2_validation_sha256 is not None
        and plan.expected_stage2_validation_sha256 != stage2_validation_sha256
    ):
        raise PhysicsAuthoringError(
            "artifact_identity_mismatch",
            "Stage 2 validation SHA256 does not match the plan",
        )
    stage2_validation = _load_json_object(
        stage2_validation_path,
        label="Stage 2 validation",
    )
    _validate_stage2_validation(
        stage2_validation,
        diagnostics=stage2,
        input_path=input_path,
        stage2_diagnostics_sha256=stage2_sha256,
        stage2_validation_sha256=stage2_validation_sha256,
        plan_sha256=plan_sha256,
    )
    edges_value = stage2.get("authored_edges")
    if not isinstance(edges_value, list) or not edges_value:
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            "Stage 2 diagnostics must contain at least one authored edge",
        )
    if stage2.get("authored_joint_count") != len(edges_value):
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            "Stage 2 authored_joint_count does not match authored_edges",
        )
    edges: list[dict[str, Any]] = []
    for index, value in enumerate(edges_value):
        if not isinstance(value, dict):
            raise PhysicsAuthoringError(
                "upstream_gate2_not_passed",
                f"Stage 2 authored_edges[{index}] is not an object",
            )
        edges.append(value)

    stage = _open_stage(input_path, label="input USD")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            "input USD has no valid default prim",
        )
    existing_plan_hash = default_prim.GetCustomDataByKey(
        "jointAgent:physicsAuthoringPlanSha256"
    )
    if existing_plan_hash not in (None, plan_sha256):
        raise PhysicsAuthoringError(
            "schema_conflict",
            "input USD was authored by a different physics authoring plan",
        )

    edge_by_candidate: dict[str, dict[str, Any]] = {}
    for index, edge in enumerate(edges):
        _validate_edge_identity(stage, edge, index=index)
        candidate_id = str(edge["candidate_id"])
        if candidate_id in edge_by_candidate:
            raise PhysicsAuthoringError(
                "upstream_gate2_not_passed",
                f"duplicate Stage 2 candidate_id: {candidate_id}",
            )
        edge_by_candidate[candidate_id] = edge

    plan_candidates = {joint.candidate_id for joint in plan.joints}
    if plan_candidates != set(edge_by_candidate):
        missing = sorted(set(edge_by_candidate) - plan_candidates)
        extra = sorted(plan_candidates - set(edge_by_candidate))
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"joint plan must exactly cover Stage 2 edges; missing={missing}, "
            f"extra={extra}",
        )

    endpoint_paths = {
        str(edge[field]) for edge in edges for field in ("body0", "body1")
    }
    body_paths = {body.prim_path for body in plan.bodies}
    if body_paths != endpoint_paths:
        missing = sorted(endpoint_paths - body_paths)
        extra = sorted(body_paths - endpoint_paths)
        raise PhysicsAuthoringError(
            "body_plan_incomplete",
            f"body plan must exactly cover Stage 2 endpoints; missing={missing}, "
            f"extra={extra}",
        )

    relevant_paths = _relevant_paths(plan, edges)
    _reject_time_sampled_physics_attributes(stage, relevant_paths)
    _reject_unmanaged_joints(
        stage,
        managed_joint_paths={str(edge["joint_path"]) for edge in edges},
        managed_body_paths=body_paths,
    )
    _reject_unmanaged_rigid_bodies(stage, managed_body_paths=body_paths)

    graph_root = _validate_articulation_graph(edges)
    _validate_articulation_root(stage, plan, graph_root=graph_root)
    _validate_body_plans(stage, plan)
    _validate_joint_plans(stage, plan, edge_by_candidate=edge_by_candidate)

    before_world_matrices = {
        body.prim_path: _matrix_values(
            UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                stage.GetPrimAtPath(body.prim_path)
            )
        )
        for body in plan.bodies
    }
    graph_snapshot = _joint_graph_snapshot(stage, edges)
    before_schema_snapshot = _schema_snapshot(stage, relevant_paths)
    before_rule_snapshot = _structural_rule_snapshot(
        stage,
        plan=plan,
        edge_by_candidate=edge_by_candidate,
    )
    allowed_schema_additions = _allowed_schema_additions(
        plan,
        edge_by_candidate=edge_by_candidate,
    )
    stage_metadata = {
        "default_prim_path": str(default_prim.GetPath()),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "kilograms_per_unit": float(UsdPhysics.GetStageKilogramsPerUnit(stage)),
    }
    return _Preflight(
        stage2_diagnostics=stage2,
        stage2_diagnostics_path=stage2_path,
        stage2_diagnostics_sha256=stage2_sha256,
        stage2_validation=stage2_validation,
        stage2_validation_path=stage2_validation_path,
        stage2_validation_sha256=stage2_validation_sha256,
        edges=tuple(edges),
        edge_by_candidate=edge_by_candidate,
        graph_root=graph_root,
        graph_snapshot=graph_snapshot,
        default_prim_path=str(default_prim.GetPath()),
        prim_paths=tuple(str(prim.GetPath()) for prim in stage.Traverse()),
        stage_metadata=stage_metadata,
        before_schema_snapshot=before_schema_snapshot,
        before_rule_snapshot=before_rule_snapshot,
        before_world_matrices=before_world_matrices,
        allowed_schema_additions=allowed_schema_additions,
    )


def _validate_stage2_diagnostics_header(stage2: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": STAGE2_DIAGNOSTICS_SCHEMA_VERSION,
        "authoring_schema_version": STAGE2_AUTHORING_SCHEMA_VERSION,
        "adapter": STAGE2_ADAPTER,
        "status": "authored",
    }
    mismatches = {
        key: {"expected": value, "actual": stage2.get(key)}
        for key, value in expected.items()
        if stage2.get(key) != value
    }
    if mismatches:
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"Stage 2 diagnostics identity mismatch: {mismatches}",
        )


def _validate_stage2_validation(
    validation: Mapping[str, Any],
    *,
    diagnostics: Mapping[str, Any],
    input_path: Path,
    stage2_diagnostics_sha256: str,
    stage2_validation_sha256: str,
    plan_sha256: str,
) -> None:
    expected = {
        "schema_version": "joint-agent-rigger-validation-v0",
        "authoring_schema_version": STAGE2_AUTHORING_SCHEMA_VERSION,
        "adapter": STAGE2_ADAPTER,
        "status": "passed",
        "validation_skipped": False,
    }
    mismatches = {
        key: {"expected": value, "actual": validation.get(key)}
        for key, value in expected.items()
        if validation.get(key) != value
    }
    if mismatches:
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"Stage 2 validation status mismatch: {mismatches}",
        )
    if validation.get("authored_joint_count") != diagnostics.get(
        "authored_joint_count"
    ):
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            "Stage 2 validation joint count does not match diagnostics",
        )
    diagnostics_output = diagnostics.get("output_usd_path")
    validation_output = validation.get("output_usd_path")
    if not isinstance(diagnostics_output, str) or not isinstance(
        validation_output, str
    ):
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            "Stage 2 artifacts must identify their output USD path",
        )
    evidence_paths = {
        Path(diagnostics_output).expanduser().resolve(),
        Path(validation_output).expanduser().resolve(),
    }
    if len(evidence_paths) != 1:
        raise PhysicsAuthoringError(
            "artifact_identity_mismatch",
            "Stage 2 diagnostics and validation identify different artifacts",
        )
    if input_path.expanduser().resolve() not in evidence_paths:
        derived_stage = _open_stage(input_path, label="derived physics input USD")
        default_prim = derived_stage.GetDefaultPrim()
        identities = {
            "plan": default_prim.GetCustomDataByKey(
                "jointAgent:physicsAuthoringPlanSha256"
            ),
            "diagnostics": default_prim.GetCustomDataByKey(
                "jointAgent:physicsStage2DiagnosticsSha256"
            ),
            "validation": default_prim.GetCustomDataByKey(
                "jointAgent:physicsStage2ValidationSha256"
            ),
        }
        expected_identities = {
            "plan": plan_sha256,
            "diagnostics": stage2_diagnostics_sha256,
            "validation": stage2_validation_sha256,
        }
        if identities != expected_identities:
            raise PhysicsAuthoringError(
                "artifact_identity_mismatch",
                "input is neither the Gate 2 output nor a provenance-stamped "
                "derivative of its diagnostics and validation",
            )
    if validation.get("candidate_readiness", {}) != diagnostics.get(
        "candidate_readiness", {}
    ):
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            "Stage 2 candidate readiness differs between diagnostics and validation",
        )
    checks = validation.get("checks")
    if not isinstance(checks, dict):
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            "Stage 2 validation has no checks object",
        )
    expected_checks = {
        "joint_graph_fidelity": "pass",
        "source_prim_paths_preserved": True,
        "applied_physics_schemas_unchanged": True,
        "forbidden_schema_authoring": False,
    }
    failed_checks = {
        key: {"expected": value, "actual": checks.get(key)}
        for key, value in expected_checks.items()
        if checks.get(key) != value
    }
    if failed_checks:
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"Stage 2 validation checks failed: {failed_checks}",
        )
    per_joint = checks.get("per_joint")
    if not isinstance(per_joint, list):
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            "Stage 2 validation has no per_joint records",
        )
    expected_pairs = sorted(
        (str(edge.get("candidate_id")), str(edge.get("joint_path")))
        for edge in diagnostics.get("authored_edges", [])
        if isinstance(edge, dict)
    )
    actual_pairs = sorted(
        (str(record.get("candidate_id")), str(record.get("joint_path")))
        for record in per_joint
        if isinstance(record, dict) and record.get("status") == "passed"
    )
    if actual_pairs != expected_pairs:
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            "Stage 2 validation per-joint identity differs from diagnostics",
        )
    expected_per_joint_checks = {
        "joint_type": "pass",
        "body0": "pass",
        "body1": "pass",
        "signed_world_axis": "pass",
        "shared_anchor": "pass",
        "source_backed_limits": "pass",
    }
    for record in per_joint:
        if not isinstance(record, dict):
            raise PhysicsAuthoringError(
                "upstream_gate2_not_passed",
                "Stage 2 per-joint validation record is not an object",
            )
        record_checks = record.get("checks")
        if record_checks != expected_per_joint_checks:
            raise PhysicsAuthoringError(
                "upstream_gate2_not_passed",
                "Stage 2 per-joint checks are incomplete for "
                f"{record.get('candidate_id')}: {record_checks}",
            )


def _validate_edge_identity(stage: Any, edge: Mapping[str, Any], *, index: int) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    required = {
        "candidate_id",
        "joint_path",
        "joint_type",
        "body0",
        "body1",
        "axis_token",
        "motion_axis_world",
        "local_pos0",
        "local_pos1",
        "local_rot0",
        "local_rot1",
        "anchor_world",
        "lower_limit",
        "upper_limit",
    }
    missing = sorted(required - set(edge))
    if missing:
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"Stage 2 authored_edges[{index}] is missing fields: {missing}",
        )
    joint_path = str(edge["joint_path"])
    prim = stage.GetPrimAtPath(joint_path)
    if not prim or not prim.IsValid() or not prim.IsActive():
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"Stage 2 joint path does not resolve: {joint_path}",
        )
    expected_type = str(edge["joint_type"])
    actual_type = _joint_type(prim)
    if actual_type != expected_type:
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"joint type mismatch at {joint_path}: {actual_type} != {expected_type}",
        )
    joint = UsdPhysics.Joint(prim)
    _require_relationship_targets(
        joint.GetBody0Rel(),
        [str(edge["body0"])],
        label=f"{joint_path} body0",
        code="upstream_gate2_not_passed",
    )
    _require_relationship_targets(
        joint.GetBody1Rel(),
        [str(edge["body1"])],
        label=f"{joint_path} body1",
        code="upstream_gate2_not_passed",
    )
    axis = _authored_attribute_value(prim.GetAttribute("physics:axis"))
    if axis != str(edge["axis_token"]):
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"joint axis mismatch at {joint_path}: {axis} != {edge['axis_token']}",
        )
    for attribute_name, edge_name in (
        ("physics:localPos0", "local_pos0"),
        ("physics:localPos1", "local_pos1"),
        ("physics:localRot0", "local_rot0"),
        ("physics:localRot1", "local_rot1"),
        ("physics:lowerLimit", "lower_limit"),
        ("physics:upperLimit", "upper_limit"),
    ):
        actual = _authored_attribute_value(prim.GetAttribute(attribute_name))
        expected_value = _json_value(edge[edge_name])
        if not _values_equal(actual, expected_value):
            raise PhysicsAuthoringError(
                "upstream_gate2_not_passed",
                f"{edge_name} mismatch at {joint_path}: {actual!r} != "
                f"{expected_value!r}",
            )

    body0 = stage.GetPrimAtPath(str(edge["body0"]))
    body1 = stage.GetPrimAtPath(str(edge["body1"]))
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    body0_xform = xform_cache.GetLocalToWorldTransform(body0)
    body1_xform = xform_cache.GetLocalToWorldTransform(body1)
    local_pos0 = prim.GetAttribute("physics:localPos0").Get()
    local_pos1 = prim.GetAttribute("physics:localPos1").Get()
    anchor0 = body0_xform.Transform(Gf.Vec3d(*local_pos0))
    anchor1 = body1_xform.Transform(Gf.Vec3d(*local_pos1))
    expected_anchor = [float(value) for value in edge["anchor_world"]]
    if not _numeric_sequence_close(
        [float(value) for value in anchor0],
        expected_anchor,
        tolerance=1e-5,
    ) or not _numeric_sequence_close(
        [float(value) for value in anchor1],
        expected_anchor,
        tolerance=1e-5,
    ):
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"shared anchor mismatch at {joint_path}",
        )

    axis_token = str(edge["axis_token"]).lower()
    base_axes = {
        "x": Gf.Vec3d(1.0, 0.0, 0.0),
        "y": Gf.Vec3d(0.0, 1.0, 0.0),
        "z": Gf.Vec3d(0.0, 0.0, 1.0),
    }
    base_axis = base_axes.get(axis_token)
    if base_axis is None:
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"unsupported axis token at {joint_path}: {axis_token}",
        )
    local_axis0 = Gf.Rotation(
        prim.GetAttribute("physics:localRot0").Get()
    ).TransformDir(base_axis)
    local_axis1 = Gf.Rotation(
        prim.GetAttribute("physics:localRot1").Get()
    ).TransformDir(base_axis)
    world_axis0 = body0_xform.TransformDir(local_axis0).GetNormalized()
    world_axis1 = body1_xform.TransformDir(local_axis1).GetNormalized()
    expected_axis = [float(value) for value in edge["motion_axis_world"]]
    if not _numeric_sequence_close(
        [float(value) for value in world_axis0],
        expected_axis,
        tolerance=1e-5,
    ) or not _numeric_sequence_close(
        [float(value) for value in world_axis1],
        expected_axis,
        tolerance=1e-5,
    ):
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"signed world axis mismatch at {joint_path}",
        )
    if prim.GetCustomDataByKey("jointAgent:candidateId") != edge["candidate_id"]:
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"candidate identity mismatch at {joint_path}",
        )
    if (
        prim.GetCustomDataByKey("jointAgent:sourceSchemaVersion")
        != STAGE2_SCHEMA_VERSION
    ):
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"source schema identity mismatch at {joint_path}",
        )


def _validate_articulation_graph(edges: list[dict[str, Any]]) -> str:
    outgoing: dict[str, list[str]] = {}
    incoming_count: dict[str, int] = {}
    nodes: set[str] = set()
    for edge in edges:
        body0 = str(edge["body0"])
        body1 = str(edge["body1"])
        if body0 == body1:
            raise PhysicsAuthoringError(
                "articulation_graph_cycle",
                f"self-edge at {body0}",
            )
        nodes.update((body0, body1))
        outgoing.setdefault(body0, []).append(body1)
        incoming_count[body1] = incoming_count.get(body1, 0) + 1
        incoming_count.setdefault(body0, incoming_count.get(body0, 0))
    multiple_parents = sorted(
        body for body, count in incoming_count.items() if count > 1
    )
    if multiple_parents:
        raise PhysicsAuthoringError(
            "articulation_graph_ambiguous_root",
            f"body endpoints have multiple parents: {multiple_parents}",
        )
    roots = sorted(node for node in nodes if incoming_count.get(node, 0) == 0)
    if len(roots) != 1:
        raise PhysicsAuthoringError(
            "articulation_graph_ambiguous_root",
            f"expected one graph root, found {roots}",
        )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise PhysicsAuthoringError(
                "articulation_graph_cycle",
                f"cycle includes {node}",
            )
        # Multiple incoming edges are rejected above, so a completed node cannot
        # be reached twice. Keep the guard as a defensive DFS invariant.
        if node in visited:  # pragma: no cover - precluded by incoming_count
            return
        visiting.add(node)
        for child in outgoing.get(node, []):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    visit(roots[0])
    if visited != nodes:
        raise PhysicsAuthoringError(
            "articulation_graph_ambiguous_root",
            f"graph root does not reach all bodies: {sorted(nodes - visited)}",
        )
    return roots[0]


def _reject_unmanaged_joints(
    stage: Any,
    *,
    managed_joint_paths: set[str],
    managed_body_paths: set[str],
) -> None:
    from pxr import UsdPhysics

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path in managed_joint_paths or not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        targets = {
            str(target)
            for relationship in (joint.GetBody0Rel(), joint.GetBody1Rel())
            for target in relationship.GetTargets()
        }
        overlap = sorted(targets & managed_body_paths)
        if overlap:
            raise PhysicsAuthoringError(
                "unmanaged_joint_conflict",
                f"unmanaged joint {prim_path} touches managed bodies {overlap}",
            )


def _reject_unmanaged_rigid_bodies(
    stage: Any,
    *,
    managed_body_paths: set[str],
) -> None:
    from pxr import Sdf

    managed = {Sdf.Path(path) for path in managed_body_paths}
    for prim in stage.Traverse():
        prim_path = prim.GetPath()
        if str(prim_path) in managed_body_paths:
            continue
        if not _has_api_schema(prim, "PhysicsRigidBodyAPI"):
            continue
        if any(
            prim_path.HasPrefix(body_path) or body_path.HasPrefix(prim_path)
            for body_path in managed
        ):
            raise PhysicsAuthoringError(
                "unmanaged_rigid_body_conflict",
                f"enabled unmanaged rigid body overlaps managed topology: {prim_path}",
            )


def _validate_articulation_root(
    stage: Any,
    plan: PhysicsAuthoringPlan,
    *,
    graph_root: str,
) -> None:
    from pxr import Sdf

    root_path = _absolute_prim_path(
        plan.articulation_root.prim_path,
        label="articulation_root.prim_path",
        sdf=Sdf,
    )
    graph_root_path = Sdf.Path(graph_root)
    if root_path != graph_root_path:
        raise PhysicsAuthoringError(
            "articulation_graph_ambiguous_root",
            f"articulation root must equal graph-root body {graph_root_path}, "
            f"got {root_path}",
        )
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid() or not root_prim.IsActive():
        raise PhysicsAuthoringError(
            "articulation_graph_ambiguous_root",
            f"articulation root path does not resolve: {root_path}",
        )
    graph_root_prim = stage.GetPrimAtPath(graph_root_path)
    kinematic = graph_root_prim.GetAttribute("physics:kinematicEnabled")
    if kinematic and kinematic.HasAuthoredValueOpinion() and bool(kinematic.Get()):
        raise PhysicsAuthoringError(
            "articulation_graph_ambiguous_root",
            f"graph root is kinematic: {graph_root}",
        )

    managed_body_paths = {body.prim_path for body in plan.bodies}
    for prim in stage.Traverse():
        prim_path = prim.GetPath()
        if not _has_api_schema(prim, "PhysicsArticulationRootAPI"):
            continue
        if prim_path == root_path:
            continue
        if (
            str(prim_path) in managed_body_paths
            or prim_path.HasPrefix(root_path)
            or root_path.HasPrefix(prim_path)
        ):
            raise PhysicsAuthoringError(
                "articulation_graph_ambiguous_root",
                "planned articulation root is nested with an existing root at "
                f"{prim_path}",
            )


def _validate_authored_articulation_root(
    stage: Any,
    plan: PhysicsAuthoringPlan,
) -> None:
    root_prim = stage.GetPrimAtPath(plan.articulation_root.prim_path)
    if not _has_api_schema(root_prim, "PhysicsArticulationRootAPI"):
        raise PhysicsAuthoringError(
            "postwrite_validation_failed",
            "authored articulation root lacks ArticulationRootAPI: "
            f"{plan.articulation_root.prim_path}",
        )


def _validate_body_plans(stage: Any, plan: PhysicsAuthoringPlan) -> None:
    from pxr import Sdf, UsdGeom

    body_paths = [
        _absolute_prim_path(body.prim_path, label="body.prim_path", sdf=Sdf)
        for body in plan.bodies
    ]
    body_path_strings = {str(path) for path in body_paths}
    for body in plan.bodies:
        body_path = Sdf.Path(body.prim_path)
        prim = stage.GetPrimAtPath(body_path)
        if not prim or not prim.IsValid() or not prim.IsActive():
            raise PhysicsAuthoringError(
                "body_plan_incomplete",
                f"body path does not resolve: {body.prim_path}",
            )
        if prim.IsInstanceProxy() or prim.IsInstanceable():
            raise PhysicsAuthoringError(
                "body_plan_incomplete",
                f"body cannot be authored without reshaping an instance: "
                f"{body.prim_path}",
            )
        if not UsdGeom.Xformable(prim):
            raise PhysicsAuthoringError(
                "body_plan_incomplete",
                f"body is not transformable: {body.prim_path}",
            )
        _preflight_rigid_body(prim)
        _preflight_mass(prim, body.mass)

        parent_body_paths = [
            other
            for other in body_paths
            if other != body_path and body_path.HasPrefix(other)
        ]
        reset_stack = bool(UsdGeom.Xformable(prim).GetResetXformStack())
        if (
            parent_body_paths
            and not reset_stack
            and body.nested_body_transform != "preserve_world_reset"
        ):
            raise PhysicsAuthoringError(
                "nested_body_reset_required",
                f"nested body {body.prim_path} requires preserve_world_reset",
            )
        if (
            parent_body_paths
            and not reset_stack
            and body.nested_body_transform == "preserve_world_reset"
            and _has_time_varying_transform_chain(prim)
        ):
            raise PhysicsAuthoringError(
                "nested_body_reset_required",
                f"time-varying nested body cannot be safely baked at default "
                f"time: {body.prim_path}",
            )

        for collider in body.colliders:
            collider_path = _absolute_prim_path(
                collider.prim_path,
                label=f"{body.prim_path} collider.prim_path",
                sdf=Sdf,
            )
            collider_prim = stage.GetPrimAtPath(collider_path)
            if (
                not collider_prim
                or not collider_prim.IsValid()
                or not collider_prim.IsActive()
            ):
                raise PhysicsAuthoringError(
                    "collider_not_gprim",
                    f"collider path does not resolve: {collider.prim_path}",
                )
            if not collider_path.HasPrefix(body_path):
                raise PhysicsAuthoringError(
                    "collider_outside_body",
                    f"collider {collider_path} is not the body GPrim or a "
                    f"descendant of {body_path}",
                )
            owning_body = _deepest_body_owner(
                collider_path,
                body_path_strings,
            )
            if owning_body != body.prim_path:
                raise PhysicsAuthoringError(
                    "collider_outside_body",
                    f"collider {collider_path} belongs to nested body {owning_body}, "
                    f"not {body.prim_path}",
                )
            if not UsdGeom.Gprim(collider_prim):
                raise PhysicsAuthoringError(
                    "collider_not_gprim",
                    f"collider must be a GPrim, not {collider_prim.GetTypeName()}: "
                    f"{collider_path}",
                )
            is_mesh = collider_prim.IsA(UsdGeom.Mesh)
            if collider.mode == "author":
                if is_mesh and collider.mesh_approximation is None:
                    raise PhysicsAuthoringError(
                        "collider_not_gprim",
                        f"Mesh collider requires an explicit mesh_approximation: "
                        f"{collider_path}",
                    )
                if not is_mesh and collider.mesh_approximation is not None:
                    raise PhysicsAuthoringError(
                        "collider_not_gprim",
                        "mesh_approximation is only valid for Mesh colliders: "
                        f"{collider_path}",
                    )
                _preflight_bool_attr(
                    collider_prim,
                    "physics:collisionEnabled",
                    True,
                    label=str(collider_path),
                )
                if is_mesh:
                    _preflight_attr_value(
                        collider_prim,
                        "physics:approximation",
                        collider.mesh_approximation,
                        label=str(collider_path),
                        expected_type="token",
                    )
            else:
                if not _has_api_schema(collider_prim, "PhysicsCollisionAPI"):
                    raise PhysicsAuthoringError(
                        "collider_not_gprim",
                        f"preserved collider lacks CollisionAPI: {collider_path}",
                    )
                collision_enabled = collider_prim.GetAttribute(
                    "physics:collisionEnabled"
                )
                _require_attribute_type(
                    collider_prim,
                    "physics:collisionEnabled",
                    "bool",
                )
                if (
                    collision_enabled
                    and collision_enabled.HasAuthoredValueOpinion()
                    and not bool(collision_enabled.Get())
                ):
                    raise PhysicsAuthoringError(
                        "collider_not_gprim",
                        f"preserved collider is disabled: {collider_path}",
                    )
                if is_mesh:
                    if not _has_api_schema(
                        collider_prim,
                        "PhysicsMeshCollisionAPI",
                    ):
                        raise PhysicsAuthoringError(
                            "collider_not_gprim",
                            f"preserved Mesh lacks MeshCollisionAPI: {collider_path}",
                        )
                    approximation = _authored_attribute_value(
                        collider_prim.GetAttribute("physics:approximation")
                    )
                    _require_attribute_type(
                        collider_prim,
                        "physics:approximation",
                        "token",
                    )
                    if approximation not in _MESH_APPROXIMATIONS:
                        raise PhysicsAuthoringError(
                            "collider_not_gprim",
                            "preserved Mesh has no supported authored approximation: "
                            f"{collider_path}",
                        )


def _validate_authored_body_schemas(stage: Any, plan: PhysicsAuthoringPlan) -> None:
    from pxr import UsdGeom

    for body in plan.bodies:
        prim = stage.GetPrimAtPath(body.prim_path)
        if not _has_api_schema(prim, "PhysicsRigidBodyAPI"):
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                f"authored body lacks RigidBodyAPI: {body.prim_path}",
            )
        if not _has_api_schema(prim, "PhysicsMassAPI"):
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                f"authored body lacks MassAPI: {body.prim_path}",
            )
        _require_authored_bool_attr(
            prim,
            "physics:rigidBodyEnabled",
            True,
        )
        _require_authored_bool_attr(
            prim,
            "physics:kinematicEnabled",
            False,
        )
        for collider in body.colliders:
            collider_prim = stage.GetPrimAtPath(collider.prim_path)
            if not _has_api_schema(collider_prim, "PhysicsCollisionAPI"):
                raise PhysicsAuthoringError(
                    "postwrite_validation_failed",
                    f"authored collider lacks CollisionAPI: {collider.prim_path}",
                )
            if collider_prim.IsA(UsdGeom.Mesh) and not _has_api_schema(
                collider_prim,
                "PhysicsMeshCollisionAPI",
            ):
                raise PhysicsAuthoringError(
                    "postwrite_validation_failed",
                    "authored Mesh collider lacks MeshCollisionAPI: "
                    f"{collider.prim_path}",
                )
            if collider.mode == "author":
                _require_authored_bool_attr(
                    collider_prim,
                    "physics:collisionEnabled",
                    True,
                )


def _require_authored_bool_attr(prim: Any, name: str, expected: bool) -> None:
    _require_attribute_type(prim, name, "bool")
    attribute = prim.GetAttribute(name)
    if (
        not attribute
        or not attribute.HasAuthoredValueOpinion()
        or bool(attribute.Get()) is not expected
    ):
        raise PhysicsAuthoringError(
            "postwrite_validation_failed",
            f"{name} must be explicitly authored as {expected} at {prim.GetPath()}",
        )


def _preflight_rigid_body(prim: Any) -> None:
    _preflight_bool_attr(
        prim,
        "physics:rigidBodyEnabled",
        True,
        label=str(prim.GetPath()),
    )
    _preflight_bool_attr(
        prim,
        "physics:kinematicEnabled",
        False,
        label=str(prim.GetPath()),
    )


def _preflight_mass(prim: Any, mass: MassPlan) -> None:
    _reject_unsupported_mass_opinions(prim)
    if mass.mode == "preserve":
        if not _has_api_schema(prim, "PhysicsMassAPI"):
            raise PhysicsAuthoringError(
                "mass_missing",
                f"preserved body lacks MassAPI: {prim.GetPath()}",
            )
        _validate_authored_mass(prim)
        return

    assert mass.mass_kg is not None
    assert mass.diagonal_inertia_kg_m2 is not None
    mass_stage_units, inertia_stage_units = _mass_stage_values(prim, mass)
    _preflight_mass_replacement_value(
        prim,
        "physics:mass",
        mass_stage_units,
        mass=mass,
        invalid_reason=_invalid_positive_scalar_reason,
        label=str(prim.GetPath()),
        expected_type="float",
    )
    _preflight_mass_replacement_value(
        prim,
        "physics:diagonalInertia",
        list(inertia_stage_units),
        mass=mass,
        invalid_reason=_invalid_inertia_reason,
        label=str(prim.GetPath()),
        expected_type="float3",
    )
    if mass.principal_axes is not None:
        _preflight_attr_value(
            prim,
            "physics:principalAxes",
            list(mass.principal_axes),
            label=str(prim.GetPath()),
            expected_type="quatf",
        )
    else:
        _validate_existing_principal_axes(prim)


def _reject_unsupported_mass_opinions(prim: Any) -> None:
    unsupported = []
    for name in ("physics:centerOfMass", "physics:density"):
        attribute = prim.GetAttribute(name)
        if attribute and attribute.HasAuthoredValueOpinion():
            unsupported.append(name)
    if unsupported:
        raise PhysicsAuthoringError(
            "mass_conflict",
            "v0 mass plans do not own authored centerOfMass or density at "
            f"{prim.GetPath()}: {unsupported}",
        )


def _mass_stage_values(
    prim: Any,
    mass: MassPlan,
) -> tuple[float, tuple[float, float, float]]:
    from pxr import UsdGeom, UsdPhysics

    assert mass.mass_kg is not None
    assert mass.diagonal_inertia_kg_m2 is not None
    stage = prim.GetStage()
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    try:
        _require_positive_finite(meters_per_unit, "metersPerUnit")
        _require_positive_finite(kilograms_per_unit, "kilogramsPerUnit")
    except ValueError as exc:
        raise PhysicsAuthoringError(
            "mass_conflict",
            f"invalid stage unit metadata: {exc}",
        ) from exc
    mass_stage_units = float(mass.mass_kg) / kilograms_per_unit
    inertia_divisor = kilograms_per_unit * meters_per_unit**2
    inertia_values = tuple(
        float(value) / inertia_divisor for value in mass.diagonal_inertia_kg_m2
    )
    inertia_stage_units = (
        inertia_values[0],
        inertia_values[1],
        inertia_values[2],
    )
    return mass_stage_units, inertia_stage_units


def _preflight_mass_replacement_value(
    prim: Any,
    name: str,
    expected: Any,
    *,
    mass: MassPlan,
    invalid_reason: Any,
    label: str,
    expected_type: str,
) -> None:
    _require_attribute_type(prim, name, expected_type)
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.HasAuthoredValueOpinion():
        return
    actual = _json_value(attribute.Get())
    if _values_equal(actual, _json_value(expected)):
        return
    reason = invalid_reason(actual)
    if mass.replace_invalid_existing and reason is not None:
        return
    raise PhysicsAuthoringError(
        "mass_conflict",
        f"conflicting {name} at {label}: {actual!r} != "
        f"{_json_value(expected)!r}; existing_reason={reason or 'valid_different'}",
    )


def _invalid_positive_scalar_reason(value: Any) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not_numeric"
    if not math.isfinite(number):
        return "non_finite"
    if number <= 0:
        return "non_positive"
    return None


def _invalid_inertia_reason(value: Any) -> str | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return "not_three_components"
    reasons = [_invalid_positive_scalar_reason(component) for component in value]
    if any(reason is not None for reason in reasons):
        return "invalid_components:" + ",".join(reason or "valid" for reason in reasons)
    try:
        _require_inertia_triangle(value, "physics:diagonalInertia")
    except ValueError:
        return "inertia_triangle_violation"
    return None


def _mass_replacement_records(prim: Any, mass: MassPlan) -> list[dict[str, Any]]:
    if mass.mode != "author" or not mass.replace_invalid_existing:
        return []
    mass_stage_units, inertia_stage_units = _mass_stage_values(prim, mass)
    specs = (
        (
            "physics:mass",
            mass_stage_units,
            _invalid_positive_scalar_reason,
        ),
        (
            "physics:diagonalInertia",
            list(inertia_stage_units),
            _invalid_inertia_reason,
        ),
    )
    records = []
    for name, expected, invalid_reason in specs:
        attribute = prim.GetAttribute(name)
        if not attribute or not attribute.HasAuthoredValueOpinion():
            continue
        before = _json_value(attribute.Get())
        reason = invalid_reason(before)
        if reason is not None and not _values_equal(before, _json_value(expected)):
            records.append(
                {
                    "attribute": name,
                    "before": before,
                    "after": _json_value(expected),
                    "reason": reason,
                }
            )
    return records


def _validate_authored_mass(prim: Any) -> None:
    mass_attr = prim.GetAttribute("physics:mass")
    inertia_attr = prim.GetAttribute("physics:diagonalInertia")
    if not mass_attr or not mass_attr.HasAuthoredValueOpinion():
        raise PhysicsAuthoringError(
            "mass_missing",
            f"MassAPI has no authored mass: {prim.GetPath()}",
        )
    if not inertia_attr or not inertia_attr.HasAuthoredValueOpinion():
        raise PhysicsAuthoringError(
            "mass_missing",
            f"MassAPI has no authored diagonal inertia: {prim.GetPath()}",
        )
    _require_attribute_type(prim, "physics:mass", "float")
    _require_attribute_type(prim, "physics:diagonalInertia", "float3")
    try:
        _require_positive_finite(float(mass_attr.Get()), "physics:mass")
    except (TypeError, ValueError) as exc:
        raise PhysicsAuthoringError(
            "mass_missing",
            f"invalid authored mass at {prim.GetPath()}",
        ) from exc
    inertia = inertia_attr.Get()
    if inertia is None or len(inertia) != 3:
        raise PhysicsAuthoringError(
            "mass_missing",
            f"invalid diagonal inertia at {prim.GetPath()}",
        )
    try:
        for index, value in enumerate(inertia):
            _require_positive_finite(float(value), f"diagonal inertia[{index}]")
        _require_inertia_triangle(inertia, "physics:diagonalInertia")
    except (TypeError, ValueError) as exc:
        raise PhysicsAuthoringError(
            "mass_missing",
            f"invalid authored diagonal inertia at {prim.GetPath()}",
        ) from exc
    _validate_existing_principal_axes(prim)


def _validate_existing_principal_axes(prim: Any) -> None:
    axes_attr = prim.GetAttribute("physics:principalAxes")
    if not axes_attr or not axes_attr.HasAuthoredValueOpinion():
        return
    _require_attribute_type(prim, "physics:principalAxes", "quatf")
    axes = axes_attr.Get()
    try:
        values = [
            float(axes.GetReal()),
            *[float(value) for value in axes.GetImaginary()],
        ]
        _require_normalized_quaternion(values, "physics:principalAxes")
    except (AttributeError, TypeError, ValueError) as exc:
        raise PhysicsAuthoringError(
            "mass_conflict",
            f"invalid authored principal axes at {prim.GetPath()}",
        ) from exc


def _validate_joint_plans(
    stage: Any,
    plan: PhysicsAuthoringPlan,
    *,
    edge_by_candidate: Mapping[str, dict[str, Any]],
    require_authored_schemas: bool = False,
) -> None:
    joint_plans = {joint.candidate_id: joint for joint in plan.joints}
    for candidate_id, joint_plan in joint_plans.items():
        edge = edge_by_candidate[candidate_id]
        prim = stage.GetPrimAtPath(str(edge["joint_path"]))
        joint_type = str(edge["joint_type"])
        motion = _motion_instance(joint_type)
        _validate_joint_instance_tokens(
            prim,
            motion=motion,
            state_mode=joint_plan.state.mode,
            control=joint_plan.control,
            require_authored_schemas=require_authored_schemas,
        )
        if joint_type == "spherical":
            if joint_plan.state.mode != "not_applicable":
                raise PhysicsAuthoringError(
                    "mimic_unsupported",
                    f"spherical joint state must be not_applicable: {candidate_id}",
                )
        elif joint_plan.state.mode == "not_applicable":
            raise PhysicsAuthoringError(
                "rest_state_outside_limits",
                f"{joint_type} joint state cannot be not_applicable: {candidate_id}",
            )
        if joint_plan.state.mode == "rest_zero":
            _require_zero_inside_limits(prim, candidate_id=candidate_id)
            assert motion is not None
            _preflight_attr_value(
                prim,
                f"state:{motion}:physics:position",
                0.0,
                label=str(prim.GetPath()),
                expected_type="float",
            )
            _preflight_attr_value(
                prim,
                f"state:{motion}:physics:velocity",
                0.0,
                label=str(prim.GetPath()),
                expected_type="float",
            )
        elif joint_plan.state.mode == "preserve":
            if motion is None or not _has_api_schema(
                prim,
                f"PhysicsJointStateAPI:{motion}",
            ):
                raise PhysicsAuthoringError(
                    "rest_state_outside_limits",
                    f"preserved joint has no matching JointStateAPI: {candidate_id}",
                )
            _validate_finite_authored_attr(
                prim,
                f"state:{motion}:physics:position",
                expected_type="float",
            )
            _validate_finite_authored_attr(
                prim,
                f"state:{motion}:physics:velocity",
                expected_type="float",
            )
            _require_position_inside_limits(
                prim,
                position=float(
                    prim.GetAttribute(f"state:{motion}:physics:position").Get()
                ),
                candidate_id=candidate_id,
            )

        control = joint_plan.control
        drive_token = f"PhysicsDriveAPI:{motion}" if motion else None
        mimic_tokens = {
            token
            for token in _api_schema_tokens(prim)
            if token.startswith("PhysxMimicJointAPI:")
        }
        if control.mode == "passive":
            if (drive_token and _has_api_schema(prim, drive_token)) or mimic_tokens:
                raise PhysicsAuthoringError(
                    "schema_conflict",
                    f"passive joint already has drive or mimic schemas: {candidate_id}",
                )
        elif control.mode == "preserve":
            has_drive = bool(drive_token and _has_api_schema(prim, drive_token))
            if has_drive and mimic_tokens:
                raise PhysicsAuthoringError(
                    "schema_conflict",
                    f"preserved joint has both drive and mimic schemas: {candidate_id}",
                )
            if mimic_tokens:
                raise PhysicsAuthoringError(
                    "mimic_unsupported",
                    f"preserved mimic requires an explicit mimic plan: {candidate_id}",
                )
            if not has_drive:
                raise PhysicsAuthoringError(
                    "drive_plan_incomplete",
                    f"preserved joint has neither drive nor mimic: {candidate_id}",
                )
            assert motion is not None
            _validate_preserved_drive(prim, motion=motion)
        elif control.mode == "drive":
            if motion is None:
                raise PhysicsAuthoringError(
                    "drive_plan_incomplete",
                    f"drive is unsupported for {joint_type}: {candidate_id}",
                )
            if mimic_tokens:
                raise PhysicsAuthoringError(
                    "schema_conflict",
                    f"drive plan conflicts with existing mimic: {candidate_id}",
                )
            _preflight_drive(prim, motion=motion, control=control)
        else:
            _preflight_mimic(
                stage,
                candidate_id=candidate_id,
                joint_plan=joint_plan,
                edge=edge,
                edge_by_candidate=edge_by_candidate,
                joint_plans=joint_plans,
            )
    _validate_mimic_cycles(joint_plans)


def _validate_joint_instance_tokens(
    prim: Any,
    *,
    motion: str | None,
    state_mode: str,
    control: JointControlPlan,
    require_authored_schemas: bool,
) -> None:
    tokens = _api_schema_tokens(prim)
    state_tokens = {
        token for token in tokens if token.startswith("PhysicsJointStateAPI:")
    }
    drive_tokens = {token for token in tokens if token.startswith("PhysicsDriveAPI:")}
    mimic_tokens = {
        token for token in tokens if token.startswith("PhysxMimicJointAPI:")
    }
    expected_state = f"PhysicsJointStateAPI:{motion}" if motion else None
    allowed_state = {expected_state} if expected_state is not None else set()
    extra_states = state_tokens - allowed_state
    if extra_states:
        raise PhysicsAuthoringError(
            "schema_conflict",
            f"joint has incompatible JointStateAPI instances at {prim.GetPath()}: "
            f"{sorted(extra_states)}",
        )
    state_required = state_mode == "preserve" or (
        require_authored_schemas and state_mode == "rest_zero"
    )
    if state_required and expected_state not in state_tokens:
        raise PhysicsAuthoringError(
            "rest_state_outside_limits",
            f"joint lacks required {expected_state} at {prim.GetPath()}",
        )
    if state_mode == "not_applicable" and state_tokens:
        raise PhysicsAuthoringError(
            "schema_conflict",
            f"not_applicable state conflicts with {sorted(state_tokens)} at "
            f"{prim.GetPath()}",
        )

    expected_drive = f"PhysicsDriveAPI:{motion}" if motion else None
    allowed_drives = (
        {expected_drive}
        if expected_drive is not None and control.mode in {"drive", "preserve"}
        else set()
    )
    extra_drives = drive_tokens - allowed_drives
    if extra_drives:
        raise PhysicsAuthoringError(
            "schema_conflict",
            f"joint has incompatible PhysicsDriveAPI instances at {prim.GetPath()}: "
            f"{sorted(extra_drives)}",
        )
    drive_required = control.mode == "preserve" or (
        require_authored_schemas and control.mode == "drive"
    )
    if drive_required and expected_drive not in drive_tokens:
        raise PhysicsAuthoringError(
            "drive_plan_incomplete",
            f"joint lacks required {expected_drive} at {prim.GetPath()}",
        )

    expected_mimic = (
        f"PhysxMimicJointAPI:{control.axis}"
        if control.mode == "mimic" and control.axis is not None
        else None
    )
    allowed_mimics = {expected_mimic} if expected_mimic is not None else set()
    extra_mimics = mimic_tokens - allowed_mimics
    if extra_mimics:
        raise PhysicsAuthoringError(
            "schema_conflict",
            f"joint has incompatible mimic instances at {prim.GetPath()}: "
            f"{sorted(extra_mimics)}",
        )
    if (
        require_authored_schemas
        and control.mode == "mimic"
        and expected_mimic not in mimic_tokens
    ):
        raise PhysicsAuthoringError(
            "mimic_unsupported",
            f"joint lacks required {expected_mimic} at {prim.GetPath()}",
        )
    if (
        require_authored_schemas
        and control.mode == "drive"
        and control.max_joint_velocity is not None
        and "PhysxJointAPI" not in tokens
    ):
        raise PhysicsAuthoringError(
            "drive_plan_incomplete",
            f"joint lacks required PhysxJointAPI at {prim.GetPath()}",
        )


def _preflight_drive(
    prim: Any,
    *,
    motion: str,
    control: JointControlPlan,
) -> None:
    assert control.drive_type is not None
    assert control.stiffness is not None
    assert control.damping is not None
    assert control.max_force is not None
    assert control.target_position is not None
    assert control.target_velocity is not None
    values = {
        f"drive:{motion}:physics:type": control.drive_type,
        f"drive:{motion}:physics:stiffness": control.stiffness,
        f"drive:{motion}:physics:damping": control.damping,
        f"drive:{motion}:physics:maxForce": control.max_force,
        f"drive:{motion}:physics:targetPosition": control.target_position,
        f"drive:{motion}:physics:targetVelocity": control.target_velocity,
    }
    for name, value in values.items():
        _preflight_attr_value(
            prim,
            name,
            value,
            label=str(prim.GetPath()),
            expected_type=("token" if name.endswith(":type") else "float"),
        )
    if control.max_joint_velocity is not None:
        _preflight_attr_value(
            prim,
            "physxJoint:maxJointVelocity",
            control.max_joint_velocity,
            label=str(prim.GetPath()),
            expected_type="float",
        )


def _validate_preserved_drive(prim: Any, *, motion: str) -> None:
    names = {
        "type": f"drive:{motion}:physics:type",
        "stiffness": f"drive:{motion}:physics:stiffness",
        "damping": f"drive:{motion}:physics:damping",
        "max_force": f"drive:{motion}:physics:maxForce",
        "target_position": f"drive:{motion}:physics:targetPosition",
        "target_velocity": f"drive:{motion}:physics:targetVelocity",
    }
    values: dict[str, Any] = {}
    for key, name in names.items():
        _require_attribute_type(
            prim,
            name,
            "token" if key == "type" else "float",
        )
        attribute = prim.GetAttribute(name)
        if not attribute or not attribute.HasAuthoredValueOpinion():
            raise PhysicsAuthoringError(
                "drive_plan_incomplete",
                f"preserved drive is missing authored {name} at {prim.GetPath()}",
            )
        values[key] = _json_value(attribute.Get())
    if values["type"] not in {"force", "acceleration"}:
        raise PhysicsAuthoringError(
            "drive_plan_incomplete",
            f"preserved drive type is invalid at {prim.GetPath()}",
        )
    validators = (
        ("stiffness", _require_positive_finite),
        ("damping", _require_nonnegative_finite),
        ("max_force", _require_positive_finite),
        ("target_position", _require_finite),
        ("target_velocity", _require_finite),
    )
    try:
        for key, validator in validators:
            validator(float(values[key]), key)
    except (TypeError, ValueError) as exc:
        raise PhysicsAuthoringError(
            "drive_plan_incomplete",
            f"invalid preserved drive at {prim.GetPath()}: {exc}",
        ) from exc
    max_velocity = prim.GetAttribute("physxJoint:maxJointVelocity")
    if max_velocity and max_velocity.HasAuthoredValueOpinion():
        _require_attribute_type(
            prim,
            "physxJoint:maxJointVelocity",
            "float",
        )
        if not _has_api_schema(prim, "PhysxJointAPI"):
            raise PhysicsAuthoringError(
                "drive_plan_incomplete",
                f"preserved max joint velocity lacks PhysxJointAPI at {prim.GetPath()}",
            )
        try:
            _require_positive_finite(
                float(max_velocity.Get()),
                "physxJoint:maxJointVelocity",
            )
        except (TypeError, ValueError) as exc:
            raise PhysicsAuthoringError(
                "drive_plan_incomplete",
                f"invalid preserved max joint velocity at {prim.GetPath()}",
            ) from exc


def _preflight_mimic(
    stage: Any,
    *,
    candidate_id: str,
    joint_plan: JointPlan,
    edge: Mapping[str, Any],
    edge_by_candidate: Mapping[str, dict[str, Any]],
    joint_plans: Mapping[str, JointPlan],
) -> None:
    control = joint_plan.control
    if str(edge["joint_type"]) != "revolute":
        raise PhysicsAuthoringError(
            "mimic_unsupported",
            f"v0 mimic supports only revolute joints: {candidate_id}",
        )
    assert control.axis is not None
    assert control.reference_candidate_id is not None
    assert control.reference_axis is not None
    reference_id = control.reference_candidate_id
    if reference_id == candidate_id or reference_id not in edge_by_candidate:
        raise PhysicsAuthoringError(
            "mimic_unsupported",
            f"invalid mimic reference {reference_id!r} for {candidate_id}",
        )
    reference_edge = edge_by_candidate[reference_id]
    if str(reference_edge["joint_type"]) != "revolute":
        raise PhysicsAuthoringError(
            "mimic_unsupported",
            f"mimic reference must be revolute: {reference_id}",
        )
    if joint_plans[reference_id].control.mode == "mimic":
        raise PhysicsAuthoringError(
            "mimic_unsupported",
            f"mimic reference cannot itself be a mimic: {reference_id}",
        )
    expected_axis = f"rot{edge['axis_token']}"
    expected_reference_axis = f"rot{reference_edge['axis_token']}"
    if (
        control.axis != expected_axis
        or control.reference_axis != expected_reference_axis
    ):
        raise PhysicsAuthoringError(
            "mimic_unsupported",
            f"mimic axes must match Gate 2 axes for {candidate_id}",
        )
    prim = stage.GetPrimAtPath(str(edge["joint_path"]))
    reference_prim = stage.GetPrimAtPath(str(reference_edge["joint_path"]))
    _require_authored_limits(prim, candidate_id=candidate_id)
    _require_authored_limits(reference_prim, candidate_id=reference_id)
    if _has_api_schema(prim, "PhysicsDriveAPI:angular"):
        raise PhysicsAuthoringError(
            "schema_conflict",
            f"mimic plan conflicts with existing drive: {candidate_id}",
        )
    namespace = f"physxMimicJoint:{control.axis}"
    assert control.gearing is not None
    assert control.offset is not None
    assert control.natural_frequency is not None
    assert control.damping_ratio is not None
    values = {
        f"{namespace}:referenceJointAxis": control.reference_axis,
        f"{namespace}:gearing": control.gearing,
        f"{namespace}:offset": control.offset,
        f"{namespace}:naturalFrequency": control.natural_frequency,
        f"{namespace}:dampingRatio": control.damping_ratio,
    }
    for name, value in values.items():
        _preflight_attr_value(
            prim,
            name,
            value,
            label=str(prim.GetPath()),
            expected_type=(
                "token" if name.endswith(":referenceJointAxis") else "float"
            ),
        )
    relationship_name = f"{namespace}:referenceJoint"
    if prim.GetAttribute(relationship_name):
        raise PhysicsAuthoringError(
            "schema_conflict",
            f"mimic reference must be a relationship at {prim.GetPath()}",
        )
    relationship = prim.GetRelationship(relationship_name)
    if relationship and relationship.HasAuthoredTargets():
        _require_relationship_targets(
            relationship,
            [str(reference_edge["joint_path"])],
            label=f"{candidate_id} mimic reference",
        )


def _validate_mimic_cycles(joint_plans: Mapping[str, JointPlan]) -> None:
    references = {
        candidate_id: plan.control.reference_candidate_id
        for candidate_id, plan in joint_plans.items()
        if plan.control.mode == "mimic"
    }
    for start in references:
        seen: set[str] = set()
        current: str | None = start
        while current in references:
            if current in seen:
                raise PhysicsAuthoringError(
                    "mimic_cycle",
                    f"mimic cycle includes {current}",
                )
            seen.add(current)
            current = references[current]


def _require_zero_inside_limits(prim: Any, *, candidate_id: str) -> None:
    _require_position_inside_limits(
        prim,
        position=0.0,
        candidate_id=candidate_id,
    )


def _require_position_inside_limits(
    prim: Any,
    *,
    position: float,
    candidate_id: str,
) -> None:
    lower_attr = prim.GetAttribute("physics:lowerLimit")
    upper_attr = prim.GetAttribute("physics:upperLimit")
    lower = (
        float(lower_attr.Get())
        if lower_attr and lower_attr.HasAuthoredValueOpinion()
        else None
    )
    upper = (
        float(upper_attr.Get())
        if upper_attr and upper_attr.HasAuthoredValueOpinion()
        else None
    )
    if (lower is not None and position < lower) or (
        upper is not None and position > upper
    ):
        raise PhysicsAuthoringError(
            "rest_state_outside_limits",
            f"state position {position} is outside authored limits for "
            f"{candidate_id}: [{lower}, {upper}]",
        )


def _require_authored_limits(prim: Any, *, candidate_id: str) -> None:
    lower = prim.GetAttribute("physics:lowerLimit")
    upper = prim.GetAttribute("physics:upperLimit")
    if (
        not lower
        or not lower.HasAuthoredValueOpinion()
        or not upper
        or not upper.HasAuthoredValueOpinion()
    ):
        raise PhysicsAuthoringError(
            "mimic_unsupported",
            f"mimic joint requires authored limits: {candidate_id}",
        )
    lower_value = float(lower.Get())
    upper_value = float(upper.Get())
    if not math.isfinite(lower_value) or not math.isfinite(upper_value):
        raise PhysicsAuthoringError(
            "mimic_unsupported",
            f"mimic limits must be finite: {candidate_id}",
        )
    if lower_value >= upper_value:
        raise PhysicsAuthoringError(
            "mimic_unsupported",
            f"mimic limits are invalid: {candidate_id}",
        )
    _require_zero_inside_limits(prim, candidate_id=candidate_id)


def _apply_plan(
    stage: Any,
    *,
    plan: PhysicsAuthoringPlan,
    preflight: _Preflight,
    plan_sha256: str,
) -> list[dict[str, Any]]:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    records: list[dict[str, Any]] = []
    default_prim = stage.GetDefaultPrim()
    _set_owned_custom_data(
        default_prim,
        "jointAgent:physicsAuthoringPlanSha256",
        plan_sha256,
    )
    _set_owned_custom_data(
        default_prim,
        "jointAgent:physicsStage2DiagnosticsSha256",
        preflight.stage2_diagnostics_sha256,
    )
    _set_owned_custom_data(
        default_prim,
        "jointAgent:physicsStage2ValidationSha256",
        preflight.stage2_validation_sha256,
    )

    body_paths = {body.prim_path for body in plan.bodies}
    for body in plan.bodies:
        prim = stage.GetPrimAtPath(body.prim_path)
        before_tokens = _api_schema_tokens(prim)
        mass_replacements = _mass_replacement_records(prim, body.mass)
        if _is_nested_body(body.prim_path, body_paths):
            xformable = UsdGeom.Xformable(prim)
            if (
                body.nested_body_transform == "preserve_world_reset"
                and not xformable.GetResetXformStack()
            ):
                world = Gf.Matrix4d(*preflight.before_world_matrices[body.prim_path])
                matrix_op = xformable.MakeMatrixXform()
                matrix_op.Set(world)
                xformable.SetResetXformStack(True)

        rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
        _require_schema_application(rigid_body, "RigidBodyAPI", prim)
        rigid_body.CreateRigidBodyEnabledAttr(True)
        rigid_body.CreateKinematicEnabledAttr(False)

        if body.mass.mode == "author":
            assert body.mass.mass_kg is not None
            assert body.mass.diagonal_inertia_kg_m2 is not None
            mass_stage_units, inertia_stage_units = _mass_stage_values(
                prim,
                body.mass,
            )
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            _require_schema_application(mass_api, "MassAPI", prim)
            mass_api.CreateMassAttr(mass_stage_units)
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia_stage_units))
            if body.mass.principal_axes is not None:
                real, x, y, z = body.mass.principal_axes
                mass_api.CreatePrincipalAxesAttr(
                    Gf.Quatf(float(real), Gf.Vec3f(float(x), float(y), float(z)))
                )
        _write_prim_provenance(
            prim,
            plan_sha256=plan_sha256,
            fields={
                "rigid_body": {
                    "source": "author_physics_schemas_v0",
                    "evidence": "body is an exact Stage 2 joint endpoint",
                },
                "mass": {
                    "mode": body.mass.mode,
                    "mass_kg": body.mass.mass_kg,
                    "diagonal_inertia_kg_m2": body.mass.diagonal_inertia_kg_m2,
                    "replace_invalid_existing": (body.mass.replace_invalid_existing),
                    "replacements": mass_replacements,
                    "source": body.mass.source,
                    "evidence": body.mass.evidence,
                },
                "nested_body_transform": body.nested_body_transform,
            },
        )
        records.append(
            {
                "kind": "body",
                "prim_path": body.prim_path,
                "schemas_before": sorted(before_tokens),
                "schemas_after": sorted(_api_schema_tokens(prim)),
                "mass_mode": body.mass.mode,
                "mass_replacements": mass_replacements,
                "nested_body_transform": body.nested_body_transform,
            }
        )

        for collider in body.colliders:
            collider_prim = stage.GetPrimAtPath(collider.prim_path)
            before_collider_tokens = _api_schema_tokens(collider_prim)
            if collider.mode == "author":
                collision = UsdPhysics.CollisionAPI.Apply(collider_prim)
                _require_schema_application(
                    collision,
                    "CollisionAPI",
                    collider_prim,
                )
                collision.CreateCollisionEnabledAttr(True)
                if collider_prim.IsA(UsdGeom.Mesh):
                    assert collider.mesh_approximation is not None
                    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collider_prim)
                    _require_schema_application(
                        mesh_collision,
                        "MeshCollisionAPI",
                        collider_prim,
                    )
                    mesh_collision.CreateApproximationAttr(collider.mesh_approximation)
            _write_prim_provenance(
                collider_prim,
                plan_sha256=plan_sha256,
                fields={
                    "collider": {
                        "body_prim_path": body.prim_path,
                        "mode": collider.mode,
                        "mesh_approximation": collider.mesh_approximation,
                        "source": collider.source,
                        "evidence": collider.evidence,
                    }
                },
            )
            records.append(
                {
                    "kind": "collider",
                    "prim_path": collider.prim_path,
                    "body_prim_path": body.prim_path,
                    "schemas_before": sorted(before_collider_tokens),
                    "schemas_after": sorted(_api_schema_tokens(collider_prim)),
                    "mode": collider.mode,
                    "mesh_approximation": collider.mesh_approximation,
                }
            )

    root_prim = stage.GetPrimAtPath(plan.articulation_root.prim_path)
    before_root_tokens = _api_schema_tokens(root_prim)
    articulation_root = UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    _require_schema_application(
        articulation_root,
        "ArticulationRootAPI",
        root_prim,
    )
    _write_prim_provenance(
        root_prim,
        plan_sha256=plan_sha256,
        fields={
            "articulation_root": {
                "source": plan.articulation_root.source,
                "evidence": plan.articulation_root.evidence,
                "graph_root": preflight.graph_root,
            }
        },
    )
    records.append(
        {
            "kind": "articulation_root",
            "prim_path": plan.articulation_root.prim_path,
            "schemas_before": sorted(before_root_tokens),
            "schemas_after": sorted(_api_schema_tokens(root_prim)),
        }
    )

    for joint_plan in plan.joints:
        edge = preflight.edge_by_candidate[joint_plan.candidate_id]
        prim = stage.GetPrimAtPath(str(edge["joint_path"]))
        before_joint_tokens = _api_schema_tokens(prim)
        motion = _motion_instance(str(edge["joint_type"]))
        if joint_plan.state.mode == "rest_zero":
            assert motion is not None
            _require_schema_application(
                prim.AddAppliedSchema(f"PhysicsJointStateAPI:{motion}"),
                f"PhysicsJointStateAPI:{motion}",
                prim,
            )
            prim.CreateAttribute(
                f"state:{motion}:physics:position",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(0.0)
            prim.CreateAttribute(
                f"state:{motion}:physics:velocity",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(0.0)

        control = joint_plan.control
        if control.mode == "drive":
            assert motion is not None
            assert control.drive_type is not None
            assert control.stiffness is not None
            assert control.damping is not None
            assert control.max_force is not None
            assert control.target_position is not None
            assert control.target_velocity is not None
            drive = UsdPhysics.DriveAPI.Apply(prim, motion)
            _require_schema_application(drive, f"DriveAPI:{motion}", prim)
            drive.CreateTypeAttr(control.drive_type)
            drive.CreateStiffnessAttr(float(control.stiffness))
            drive.CreateDampingAttr(float(control.damping))
            drive.CreateMaxForceAttr(float(control.max_force))
            drive.CreateTargetPositionAttr(float(control.target_position))
            drive.CreateTargetVelocityAttr(float(control.target_velocity))
            if control.max_joint_velocity is not None:
                _require_schema_application(
                    prim.AddAppliedSchema("PhysxJointAPI"),
                    "PhysxJointAPI",
                    prim,
                )
                prim.CreateAttribute(
                    "physxJoint:maxJointVelocity",
                    Sdf.ValueTypeNames.Float,
                    custom=False,
                ).Set(float(control.max_joint_velocity))
        elif control.mode == "mimic":
            _author_mimic(
                prim,
                control=control,
                reference_joint_path=str(
                    preflight.edge_by_candidate[str(control.reference_candidate_id)][
                        "joint_path"
                    ]
                ),
                sdf=Sdf,
            )

        _write_prim_provenance(
            prim,
            plan_sha256=plan_sha256,
            fields={
                "candidate_id": joint_plan.candidate_id,
                "joint_state": {
                    "mode": joint_plan.state.mode,
                    "source": joint_plan.state.source,
                    "evidence": joint_plan.state.evidence,
                },
                "control": {
                    "mode": control.mode,
                    "source": control.source,
                    "evidence": control.evidence,
                },
            },
        )
        records.append(
            {
                "kind": "joint",
                "candidate_id": joint_plan.candidate_id,
                "prim_path": str(edge["joint_path"]),
                "joint_type": edge["joint_type"],
                "schemas_before": sorted(before_joint_tokens),
                "schemas_after": sorted(_api_schema_tokens(prim)),
                "state_mode": joint_plan.state.mode,
                "control_mode": control.mode,
            }
        )
    return records


def _author_mimic(
    prim: Any,
    *,
    control: JointControlPlan,
    reference_joint_path: str,
    sdf: Any,
) -> None:
    assert control.axis is not None
    assert control.reference_axis is not None
    assert control.gearing is not None
    assert control.offset is not None
    assert control.natural_frequency is not None
    assert control.damping_ratio is not None
    _require_schema_application(
        prim.AddAppliedSchema(f"PhysxMimicJointAPI:{control.axis}"),
        f"PhysxMimicJointAPI:{control.axis}",
        prim,
    )
    namespace = f"physxMimicJoint:{control.axis}"
    attrs = (
        ("referenceJointAxis", sdf.ValueTypeNames.Token, control.reference_axis),
        ("gearing", sdf.ValueTypeNames.Float, float(control.gearing)),
        ("offset", sdf.ValueTypeNames.Float, float(control.offset)),
        (
            "naturalFrequency",
            sdf.ValueTypeNames.Float,
            float(control.natural_frequency),
        ),
        ("dampingRatio", sdf.ValueTypeNames.Float, float(control.damping_ratio)),
    )
    for suffix, value_type, value in attrs:
        prim.CreateAttribute(
            f"{namespace}:{suffix}",
            value_type,
            custom=False,
        ).Set(value)
    prim.CreateRelationship(
        f"{namespace}:referenceJoint",
        custom=False,
    ).SetTargets([sdf.Path(reference_joint_path)])


def _require_schema_application(applied: Any, schema: str, prim: Any) -> None:
    if not applied:
        raise PhysicsAuthoringError(
            "postwrite_validation_failed",
            f"could not apply {schema} at {prim.GetPath()}",
        )


def _write_prim_provenance(
    prim: Any,
    *,
    plan_sha256: str,
    fields: Mapping[str, Any],
) -> None:
    existing_raw = prim.GetCustomDataByKey("jointAgent:physicsFieldProvenance")
    merged_fields: dict[str, Any] = {}
    if existing_raw is not None:
        if not isinstance(existing_raw, str):
            raise PhysicsAuthoringError(
                "schema_conflict",
                f"physicsFieldProvenance is not a JSON string at {prim.GetPath()}",
            )
        try:
            existing = json.loads(existing_raw)
        except json.JSONDecodeError as exc:
            raise PhysicsAuthoringError(
                "schema_conflict",
                f"invalid physicsFieldProvenance at {prim.GetPath()}",
            ) from exc
        if (
            not isinstance(existing, dict)
            or existing.get("plan_sha256") != plan_sha256
            or not isinstance(existing.get("fields"), dict)
        ):
            raise PhysicsAuthoringError(
                "schema_conflict",
                f"physicsFieldProvenance belongs to another plan at {prim.GetPath()}",
            )
        merged_fields.update(existing["fields"])
    for key, value in fields.items():
        canonical_value = json.loads(
            json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
        existing_value = merged_fields.get(key)
        if existing_value is not None and existing_value != canonical_value:
            raise PhysicsAuthoringError(
                "schema_conflict",
                f"conflicting provenance field {key} at {prim.GetPath()}",
            )
        merged_fields[key] = canonical_value
    payload = {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "fields": merged_fields,
    }
    prim.SetCustomDataByKey(
        "jointAgent:physicsFieldProvenance",
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )


def _set_owned_custom_data(prim: Any, key: str, value: Any) -> None:
    existing = prim.GetCustomDataByKey(key)
    if existing not in (None, value):
        raise PhysicsAuthoringError(
            "schema_conflict",
            f"conflicting owned customData {key} at {prim.GetPath()}",
        )
    prim.SetCustomDataByKey(key, value)


def _validate_authored_output(
    path: Path,
    *,
    plan: PhysicsAuthoringPlan,
    preflight: _Preflight,
) -> dict[str, Any]:
    from pxr import UsdGeom, UsdPhysics

    try:
        stage = _open_stage(path, label="authored physics output USD")
        default_prim = stage.GetDefaultPrim()
        if str(default_prim.GetPath()) != preflight.default_prim_path:
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                "authoring changed the default prim",
            )
        prim_paths = tuple(str(prim.GetPath()) for prim in stage.Traverse())
        if prim_paths != preflight.prim_paths:
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                "authoring changed the prim path set or traversal order",
            )
        stage_metadata = {
            "default_prim_path": str(default_prim.GetPath()),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "kilograms_per_unit": float(UsdPhysics.GetStageKilogramsPerUnit(stage)),
        }
        if stage_metadata != preflight.stage_metadata:
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                f"authoring changed stage metadata: {stage_metadata}",
            )
        if (
            _joint_graph_snapshot(stage, list(preflight.edges))
            != preflight.graph_snapshot
        ):
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                "authoring changed the Gate 2 joint graph",
            )

        _validate_articulation_root(stage, plan, graph_root=preflight.graph_root)
        _validate_authored_articulation_root(stage, plan)
        _validate_body_plans(stage, plan)
        _validate_authored_body_schemas(stage, plan)
        _validate_joint_plans(
            stage,
            plan,
            edge_by_candidate=preflight.edge_by_candidate,
            require_authored_schemas=True,
        )
        if (
            default_prim.GetCustomDataByKey("jointAgent:physicsStage2DiagnosticsSha256")
            != preflight.stage2_diagnostics_sha256
        ):
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                "output lost the Stage 2 diagnostics identity stamp",
            )
        if (
            default_prim.GetCustomDataByKey("jointAgent:physicsStage2ValidationSha256")
            != preflight.stage2_validation_sha256
        ):
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                "output lost the Stage 2 validation identity stamp",
            )

        world_transform_checks: dict[str, bool] = {}
        xform_cache = UsdGeom.XformCache()
        for body_path, expected in preflight.before_world_matrices.items():
            actual = _matrix_values(
                xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path))
            )
            matches = _numeric_sequence_close(actual, expected, tolerance=1e-6)
            world_transform_checks[body_path] = matches
            if not matches:
                raise PhysicsAuthoringError(
                    "postwrite_validation_failed",
                    f"body world transform changed: {body_path}",
                )

        relevant_paths = _relevant_paths(plan, list(preflight.edges))
        after_snapshot = _schema_snapshot(stage, relevant_paths)
        schema_deltas = _schema_deltas(
            preflight.before_schema_snapshot,
            after_snapshot,
        )
        _validate_schema_deltas(
            schema_deltas,
            allowed_additions=preflight.allowed_schema_additions,
        )
        after_rules = _structural_rule_snapshot(
            stage,
            plan=plan,
            edge_by_candidate=preflight.edge_by_candidate,
        )
        structural_rule_deltas = _structural_rule_deltas(
            preflight.before_rule_snapshot,
            after_rules,
        )
        expected_residuals = _expected_residuals(
            plan,
            preflight=preflight,
            after_rules=after_rules,
        )
        semantic_change_count = sum(
            len(delta["added_api_schemas"])
            + len(delta["changed_attributes"])
            + len(delta["changed_attribute_time_samples"])
            + len(delta["changed_relationships"])
            for delta in schema_deltas.values()
        )
        warnings = [
            residual["message"]
            for residual in expected_residuals
            if residual.get("state") == "expected_residual"
        ]
        return {
            "gate2_joint_graph_unchanged": True,
            "prim_paths_unchanged": True,
            "stage_metadata_unchanged": True,
            "world_transforms_unchanged": world_transform_checks,
            "planned_schemas_and_values_verified": True,
            "no_unplanned_schema_removals": True,
            "after_schema_snapshot": after_snapshot,
            "schema_deltas": schema_deltas,
            "structural_rule_deltas": structural_rule_deltas,
            "expected_residuals": expected_residuals,
            "semantic_change_count": semantic_change_count,
            "warnings": warnings,
        }
    except PhysicsAuthoringError:
        raise
    except Exception as exc:
        raise PhysicsAuthoringError(
            "postwrite_validation_failed",
            str(exc),
        ) from exc


def _relevant_paths(
    plan: PhysicsAuthoringPlan,
    edges: list[dict[str, Any]],
) -> list[str]:
    return sorted(
        {
            plan.articulation_root.prim_path,
            *(body.prim_path for body in plan.bodies),
            *(
                collider.prim_path
                for body in plan.bodies
                for collider in body.colliders
            ),
            *(str(edge["joint_path"]) for edge in edges),
        }
    )


def _reject_time_sampled_physics_attributes(
    stage: Any,
    prim_paths: list[str],
) -> None:
    prefixes = (
        "physics:",
        "drive:",
        "state:",
        "physxJoint:",
        "physxMimicJoint:",
    )
    sampled: dict[str, dict[str, list[float]]] = {}
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if not name.startswith(prefixes):
                continue
            times = [float(value) for value in attribute.GetTimeSamples()]
            if times:
                sampled.setdefault(prim_path, {})[name] = times
    if sampled:
        raise PhysicsAuthoringError(
            "time_sampled_physics_unsupported",
            f"v0 requires default-time-only physics opinions: {sampled}",
        )


def _schema_snapshot(stage: Any, prim_paths: list[str]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    prefixes = (
        "physics:",
        "drive:",
        "state:",
        "physxJoint:",
        "physxMimicJoint:",
    )
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        attributes = {}
        attribute_time_samples = {}
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if not name.startswith(prefixes) or not attribute.HasAuthoredValueOpinion():
                continue
            attributes[name] = _json_value(attribute.Get())
            time_samples = attribute.GetTimeSamples()
            if time_samples:
                attribute_time_samples[name] = {
                    str(float(time)): _json_value(attribute.Get(time))
                    for time in time_samples
                }
        relationships = {}
        for relationship in prim.GetRelationships():
            name = relationship.GetName()
            if not name.startswith(prefixes) or not relationship.HasAuthoredTargets():
                continue
            relationships[name] = [str(target) for target in relationship.GetTargets()]
        snapshot[prim_path] = {
            "api_schemas": sorted(_api_schema_tokens(prim)),
            "attributes": dict(sorted(attributes.items())),
            "attribute_time_samples": dict(sorted(attribute_time_samples.items())),
            "relationships": dict(sorted(relationships.items())),
        }
    return snapshot


def _schema_deltas(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prim_path in sorted(set(before) | set(after)):
        old = before.get(
            prim_path,
            {
                "api_schemas": [],
                "attributes": {},
                "attribute_time_samples": {},
                "relationships": {},
            },
        )
        new = after.get(
            prim_path,
            {
                "api_schemas": [],
                "attributes": {},
                "attribute_time_samples": {},
                "relationships": {},
            },
        )
        old_schemas = set(old["api_schemas"])
        new_schemas = set(new["api_schemas"])
        changed_attributes = _mapping_delta(
            old["attributes"],
            new["attributes"],
        )
        changed_attribute_time_samples = _mapping_delta(
            old.get("attribute_time_samples", {}),
            new.get("attribute_time_samples", {}),
        )
        changed_relationships = _mapping_delta(
            old["relationships"],
            new["relationships"],
        )
        added = sorted(new_schemas - old_schemas)
        removed = sorted(old_schemas - new_schemas)
        if (
            added
            or removed
            or changed_attributes
            or changed_attribute_time_samples
            or changed_relationships
        ):
            result[prim_path] = {
                "added_api_schemas": added,
                "removed_api_schemas": removed,
                "changed_attributes": changed_attributes,
                "changed_attribute_time_samples": (changed_attribute_time_samples),
                "changed_relationships": changed_relationships,
            }
    return result


def _mapping_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }


def _allowed_schema_additions(
    plan: PhysicsAuthoringPlan,
    *,
    edge_by_candidate: Mapping[str, dict[str, Any]],
) -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = {}

    def add(path: str, *tokens: str) -> None:
        allowed.setdefault(path, set()).update(tokens)

    for body in plan.bodies:
        add(body.prim_path, "PhysicsRigidBodyAPI")
        if body.mass.mode == "author":
            add(body.prim_path, "PhysicsMassAPI")
        for collider in body.colliders:
            if collider.mode != "author":
                continue
            add(collider.prim_path, "PhysicsCollisionAPI")
            if collider.mesh_approximation is not None:
                add(collider.prim_path, "PhysicsMeshCollisionAPI")
    add(plan.articulation_root.prim_path, "PhysicsArticulationRootAPI")
    for joint in plan.joints:
        edge = edge_by_candidate[joint.candidate_id]
        path = str(edge["joint_path"])
        motion = _motion_instance(str(edge["joint_type"]))
        if joint.state.mode == "rest_zero":
            assert motion is not None
            add(path, f"PhysicsJointStateAPI:{motion}")
        if joint.control.mode == "drive":
            assert motion is not None
            add(path, f"PhysicsDriveAPI:{motion}")
            if joint.control.max_joint_velocity is not None:
                add(path, "PhysxJointAPI")
        elif joint.control.mode == "mimic":
            assert joint.control.axis is not None
            add(path, f"PhysxMimicJointAPI:{joint.control.axis}")
    return allowed


def _validate_schema_deltas(
    deltas: Mapping[str, Any],
    *,
    allowed_additions: Mapping[str, set[str]],
) -> None:
    for prim_path, delta in deltas.items():
        if delta["changed_attribute_time_samples"]:
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                f"authoring changed time-sampled physics opinions at {prim_path}: "
                f"{delta['changed_attribute_time_samples']}",
            )
        if delta["removed_api_schemas"]:
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                f"authoring removed API schemas at {prim_path}: "
                f"{delta['removed_api_schemas']}",
            )
        unexpected = set(delta["added_api_schemas"]) - allowed_additions.get(
            prim_path,
            set(),
        )
        if unexpected:
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                f"authoring added unplanned API schemas at {prim_path}: "
                f"{sorted(unexpected)}",
            )


def _structural_rule_snapshot(
    stage: Any,
    *,
    plan: PhysicsAuthoringPlan,
    edge_by_candidate: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    from pxr import UsdGeom

    rigid_body_findings: list[str] = []
    mass_findings: list[str] = []
    collider_findings: list[str] = []
    for body in plan.bodies:
        prim = stage.GetPrimAtPath(body.prim_path)
        if not _has_api_schema(prim, "PhysicsRigidBodyAPI"):
            rigid_body_findings.append(body.prim_path)
        try:
            if not _has_api_schema(prim, "PhysicsMassAPI"):
                raise PhysicsAuthoringError("mass_missing", "MassAPI missing")
            _validate_authored_mass(prim)
        except (PhysicsAuthoringError, TypeError, ValueError):
            mass_findings.append(body.prim_path)
        valid_collider = False
        for collider in body.colliders:
            collider_prim = stage.GetPrimAtPath(collider.prim_path)
            if not _has_api_schema(collider_prim, "PhysicsCollisionAPI"):
                continue
            enabled = collider_prim.GetAttribute("physics:collisionEnabled")
            if (
                enabled
                and enabled.HasAuthoredValueOpinion()
                and not bool(enabled.Get())
            ):
                continue
            if collider_prim.IsA(UsdGeom.Mesh):
                if not _has_api_schema(
                    collider_prim,
                    "PhysicsMeshCollisionAPI",
                ):
                    continue
                approximation = _authored_attribute_value(
                    collider_prim.GetAttribute("physics:approximation")
                )
                if approximation not in _MESH_APPROXIMATIONS:
                    continue
            valid_collider = True
        if not valid_collider:
            collider_findings.append(body.prim_path)

    state_findings: list[str] = []
    control_findings: list[str] = []
    for joint in plan.joints:
        edge = edge_by_candidate[joint.candidate_id]
        motion = _motion_instance(str(edge["joint_type"]))
        if motion is None:
            continue
        prim = stage.GetPrimAtPath(str(edge["joint_path"]))
        if not _has_api_schema(prim, f"PhysicsJointStateAPI:{motion}"):
            state_findings.append(joint.candidate_id)
        has_drive = _has_api_schema(prim, f"PhysicsDriveAPI:{motion}")
        has_mimic = any(
            token.startswith("PhysxMimicJointAPI:")
            for token in _api_schema_tokens(prim)
        )
        if not has_drive and not has_mimic:
            control_findings.append(joint.candidate_id)

    root_prim = stage.GetPrimAtPath(plan.articulation_root.prim_path)
    root_findings = (
        []
        if _has_api_schema(root_prim, "PhysicsArticulationRootAPI")
        else [plan.articulation_root.prim_path]
    )
    return {
        "RigidBodyHasAPI": sorted(rigid_body_findings),
        "RigidBodyHasMassAPI": sorted(mass_findings),
        "RigidBodyHasCollider": sorted(collider_findings),
        "JointHasJointStateAPI": sorted(state_findings),
        "PhysicsJointHasDriveOrMimicAPI": sorted(control_findings),
        "HasArticulationRoot": sorted(root_findings),
    }


def _structural_rule_deltas(
    before: Mapping[str, list[str]],
    after: Mapping[str, list[str]],
) -> dict[str, Any]:
    return {
        "semantics": (
            "after minus before for local structural checks; rerun Isaac Gate 3A "
            "for validator rule deltas"
        ),
        "rules": {
            rule: {
                "before_nonconforming_count": len(before.get(rule, [])),
                "after_nonconforming_count": len(after.get(rule, [])),
                "nonconforming_count_delta": (
                    len(after.get(rule, [])) - len(before.get(rule, []))
                ),
                "before_findings": before.get(rule, []),
                "after_findings": after.get(rule, []),
            }
            for rule in sorted(set(before) | set(after))
        },
    }


def _expected_residuals(
    plan: PhysicsAuthoringPlan,
    *,
    preflight: _Preflight,
    after_rules: Mapping[str, list[str]],
) -> list[dict[str, Any]]:
    residuals: list[dict[str, Any]] = []
    for candidate_id in after_rules.get("PhysicsJointHasDriveOrMimicAPI", []):
        mode = next(
            joint.control.mode
            for joint in plan.joints
            if joint.candidate_id == candidate_id
        )
        residuals.append(
            {
                "lane": "Gate 3A",
                "rule": "PhysicsJointHasDriveOrMimicAPI",
                "candidate_id": candidate_id,
                "state": "expected_residual",
                "message": (
                    f"{candidate_id} remains {mode}; no drive or mimic values "
                    "were invented"
                ),
            }
        )
    residuals.extend(
        [
            {
                "lane": "Gate 3B",
                "rule": "PMT.001",
                "state": "not_evaluated",
                "message": "physics material authoring is outside this v0 owner boundary",
            },
            {
                "lane": "Gate 3A",
                "rule": "NonAdjacentCollisionMeshesDoNotClash",
                "state": "not_evaluated",
                "message": (
                    "exact plan-backed collider authoring does not certify geometric "
                    "non-overlap; use official Isaac validator evidence"
                ),
            },
            {
                "lane": "Gate 3B",
                "rule": "AA.001",
                "state": "not_evaluated",
                "message": (
                    "anchored dependency and composition packaging is outside "
                    "post-rigger physics schema authoring"
                ),
            },
            {
                "lane": "Gate 3B",
                "rule": "GSP.001",
                "state": "not_evaluated",
                "message": "grasp vectors are outside post-rigger physics schema authoring",
            },
            {
                "lane": "Gate 3B",
                "rule": "ISA.001",
                "state": "not_evaluated",
                "message": "Foundation composition restructuring is intentionally excluded",
            },
        ]
    )
    if not math.isclose(
        float(preflight.stage_metadata["meters_per_unit"]),
        1.0,
        abs_tol=1e-12,
    ):
        residuals.append(
            {
                "lane": "Gate 3B",
                "rule": "UN.007",
                "state": "expected_residual",
                "message": "stage units were preserved; the step does not rescale assets",
            }
        )
    return residuals


def _joint_graph_snapshot(stage: Any, edges: list[dict[str, Any]]) -> dict[str, Any]:
    from pxr import UsdPhysics

    snapshot: dict[str, Any] = {}
    for edge in edges:
        prim = stage.GetPrimAtPath(str(edge["joint_path"]))
        joint = UsdPhysics.Joint(prim)
        snapshot[str(edge["candidate_id"])] = {
            "joint_path": str(prim.GetPath()),
            "joint_type": _joint_type(prim),
            "body0": [str(target) for target in joint.GetBody0Rel().GetTargets()],
            "body1": [str(target) for target in joint.GetBody1Rel().GetTargets()],
            "axis": _authored_attribute_value(prim.GetAttribute("physics:axis")),
            "local_pos0": _authored_attribute_value(
                prim.GetAttribute("physics:localPos0")
            ),
            "local_pos1": _authored_attribute_value(
                prim.GetAttribute("physics:localPos1")
            ),
            "local_rot0": _authored_attribute_value(
                prim.GetAttribute("physics:localRot0")
            ),
            "local_rot1": _authored_attribute_value(
                prim.GetAttribute("physics:localRot1")
            ),
            "lower_limit": _authored_attribute_value(
                prim.GetAttribute("physics:lowerLimit")
            ),
            "upper_limit": _authored_attribute_value(
                prim.GetAttribute("physics:upperLimit")
            ),
            "candidate_id": prim.GetCustomDataByKey("jointAgent:candidateId"),
            "source_schema_version": prim.GetCustomDataByKey(
                "jointAgent:sourceSchemaVersion"
            ),
        }
    return snapshot


def _joint_type(prim: Any) -> str | None:
    from pxr import UsdPhysics

    schemas = {
        "revolute": UsdPhysics.RevoluteJoint,
        "prismatic": UsdPhysics.PrismaticJoint,
        "spherical": UsdPhysics.SphericalJoint,
    }
    for name, schema in schemas.items():
        if prim.IsA(schema):
            return name
    return None


def _motion_instance(joint_type: str) -> str | None:
    if joint_type == "revolute":
        return "angular"
    if joint_type == "prismatic":
        return "linear"
    return None


def _api_schema_tokens(prim: Any) -> set[str]:
    tokens = {str(token) for token in prim.GetAppliedSchemas()}
    metadata = prim.GetMetadata("apiSchemas")
    if metadata is not None:
        try:
            tokens.update(str(token) for token in metadata.GetAppliedItems())
        except AttributeError:
            if isinstance(metadata, list | tuple):
                tokens.update(str(token) for token in metadata)
    return tokens


def _has_api_schema(prim: Any, token: str) -> bool:
    return token in _api_schema_tokens(prim)


def _preflight_bool_attr(
    prim: Any,
    name: str,
    expected: bool,
    *,
    label: str,
) -> None:
    _preflight_attr_value(
        prim,
        name,
        expected,
        label=label,
        expected_type="bool",
    )


def _preflight_attr_value(
    prim: Any,
    name: str,
    expected: Any,
    *,
    label: str,
    expected_type: str,
) -> None:
    _require_attribute_type(prim, name, expected_type)
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.HasAuthoredValueOpinion():
        return
    actual = _json_value(attribute.Get())
    if not _values_equal(actual, _json_value(expected)):
        raise PhysicsAuthoringError(
            "schema_conflict",
            f"conflicting {name} at {label}: {actual!r} != {_json_value(expected)!r}",
        )


def _require_attribute_type(prim: Any, name: str, expected_type: str) -> None:
    attribute = prim.GetAttribute(name)
    if not attribute:
        return
    composed_type = str(attribute.GetTypeName())
    authored_types = {
        str(spec.typeName)
        for spec in attribute.GetPropertyStack()
        if str(spec.typeName)
    }
    if composed_type != expected_type or any(
        value != expected_type for value in authored_types
    ):
        raise PhysicsAuthoringError(
            "schema_conflict",
            f"{name} at {prim.GetPath()} has composed type {composed_type} and "
            f"authored types {sorted(authored_types)}, expected {expected_type}",
        )


def _validate_finite_authored_attr(
    prim: Any,
    name: str,
    *,
    expected_type: str,
) -> None:
    _require_attribute_type(prim, name, expected_type)
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.HasAuthoredValueOpinion():
        raise PhysicsAuthoringError(
            "rest_state_outside_limits",
            f"missing authored {name} at {prim.GetPath()}",
        )
    value = float(attribute.Get())
    if not math.isfinite(value):
        raise PhysicsAuthoringError(
            "rest_state_outside_limits",
            f"non-finite authored {name} at {prim.GetPath()}",
        )


def _authored_attribute_value(attribute: Any) -> Any:
    if not attribute or not attribute.HasAuthoredValueOpinion():
        return None
    return _json_value(attribute.Get())


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return [
            float(value.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ]
    if hasattr(value, "path") and type(value).__name__ == "AssetPath":
        return str(value.path)
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        try:
            return [_json_value(value[index]) for index in range(len(value))]
        except (TypeError, IndexError):
            pass
    return str(value)


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, int | float) and isinstance(right, int | float):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-7)
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require_relationship_targets(
    relationship: Any,
    expected: list[str],
    *,
    label: str,
    code: str = "schema_conflict",
) -> None:
    actual = [str(target) for target in relationship.GetTargets()]
    if actual != expected:
        raise PhysicsAuthoringError(
            code,
            f"{label} targets mismatch: {actual} != {expected}",
        )


def _absolute_prim_path(value: str, *, label: str, sdf: Any) -> Any:
    path = sdf.Path(value)
    if not path.IsAbsoluteRootOrPrimPath() or path.IsAbsoluteRootPath():
        raise PhysicsAuthoringError(
            "authoring_plan_invalid",
            f"{label} must be an exact absolute prim path: {value}",
        )
    return path


def _deepest_body_owner(collider_path: Any, body_paths: set[str]) -> str | None:
    from pxr import Sdf

    candidates = [
        body_path
        for body_path in body_paths
        if collider_path.HasPrefix(Sdf.Path(body_path))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: Sdf.Path(path).pathElementCount)


def _is_nested_body(body_path: str, body_paths: set[str]) -> bool:
    from pxr import Sdf

    path = Sdf.Path(body_path)
    return any(
        other != body_path and path.HasPrefix(Sdf.Path(other)) for other in body_paths
    )


def _has_time_varying_transform_chain(prim: Any) -> bool:
    from pxr import UsdGeom

    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        xformable = UsdGeom.Xformable(current)
        if xformable:
            if xformable.TransformMightBeTimeVarying():
                return True
            if any(
                operation.GetAttr().GetTimeSamples()
                for operation in xformable.GetOrderedXformOps()
            ):
                return True
        current = current.GetParent()
    return False


def _matrix_values(matrix: Any) -> list[float]:
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _numeric_sequence_close(
    actual: list[float],
    expected: list[float],
    *,
    tolerance: float,
) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)
        for left, right in zip(actual, expected, strict=True)
    )


def _validate_artifact_paths(
    *,
    input_path: Path,
    stage2_diagnostics_path: Path,
    stage2_validation_path: Path,
    plan_path: Path,
    output_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
) -> None:
    paths = {
        "input_usd_path": input_path,
        "stage2_diagnostics_path": stage2_diagnostics_path,
        "stage2_validation_path": stage2_validation_path,
        "authoring_plan_path": plan_path,
        "output_usd_path": output_path,
        "diagnostics_path": diagnostics_path,
        "validation_path": validation_path,
    }
    seen: dict[Path, str] = {}
    for label, path in paths.items():
        resolved = path.expanduser().resolve()
        previous = seen.get(resolved)
        if previous is not None:
            raise PhysicsAuthoringError(
                "authoring_plan_invalid",
                f"{label} collides with {previous}: {path}",
            )
        seen[resolved] = label
    stranded_backups = sorted(
        str(backup)
        for target in (output_path, diagnostics_path, validation_path)
        for backup in target.parent.glob(f".{target.name}.*.backup")
    )
    if stranded_backups:
        raise PhysicsAuthoringError(
            "artifact_write_failed",
            "stranded transaction backups require recovery before rerun: "
            + ", ".join(stranded_backups),
        )
    input_suffix = input_path.suffix.lower()
    output_suffix = output_path.suffix.lower()
    if input_suffix not in _SUPPORTED_USD_EXTENSIONS:
        raise PhysicsAuthoringError(
            "authoring_plan_invalid",
            f"unsupported input USD suffix: {input_path}",
        )
    if output_suffix != input_suffix:
        raise PhysicsAuthoringError(
            "authoring_plan_invalid",
            "author_physics_schemas must preserve the input container suffix: "
            f"{input_suffix} != {output_suffix}",
        )
    for label, path in {
        "diagnostics_path": diagnostics_path,
        "validation_path": validation_path,
    }.items():
        if path.is_symlink() or path.is_dir():
            raise PhysicsAuthoringError(
                "authoring_plan_invalid",
                f"{label} must be a regular file path: {path}",
            )
    for label, path, schema_version in (
        ("diagnostics_path", diagnostics_path, DIAGNOSTICS_SCHEMA_VERSION),
        ("validation_path", validation_path, VALIDATION_SCHEMA_VERSION),
    ):
        if path.exists() and not _is_owned_report_artifact(
            path,
            schema_version=schema_version,
            output_path=output_path,
        ):
            raise PhysicsAuthoringError(
                "artifact_identity_mismatch",
                f"refusing to replace unowned {label}: {path}",
            )
    _validate_existing_output_artifact(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(2, "input USD not found", str(input_path))


def _is_owned_report_artifact(
    path: Path,
    *,
    schema_version: str,
    output_path: Path,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False
        identity = payload.get("identity")
        if not isinstance(identity, dict):
            return False
        recorded_output = identity.get("output_usd_path")
        return (
            payload.get("schema_version") == schema_version
            and payload.get("plan_schema_version") == PLAN_SCHEMA_VERSION
            and isinstance(recorded_output, str)
            and Path(recorded_output).expanduser().resolve()
            == output_path.expanduser().resolve()
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def _temporary_output_path(output_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=output_path.suffix.lower(),
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    temporary.unlink()
    return temporary


def _prepare_editable_output(
    input_path: Path,
    temp_output: Path,
) -> tuple[Path, Path | None]:
    from joint_agent.functions.joint_rigger_adapter import _prepare_usd_for_handoff

    editable_path, preparation_dir = _prepare_usd_for_handoff(
        input_path,
        temp_output,
    )
    return (
        Path(editable_path),
        Path(preparation_dir) if preparation_dir is not None else None,
    )


def _package_usdz(editable_path: Path, temp_output: Path) -> None:
    from joint_agent.functions.joint_rigger_adapter import _package_usdz_for_handoff

    _package_usdz_for_handoff(editable_path, temp_output)


def _open_stage(path: Path, *, label: str) -> Any:
    from pxr import Usd

    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise PhysicsAuthoringError(
            "upstream_gate2_not_passed",
            f"could not open {label}: {path}",
        )
    return stage


def _is_owned_derivative(
    path: Path,
    *,
    plan_sha256: str,
    stage2_diagnostics_sha256: str,
    stage2_validation_sha256: str,
) -> bool:
    try:
        stage = _open_stage(path, label="owned derivative USD")
    except (FileNotFoundError, PhysicsAuthoringError):
        return False
    default_prim = stage.GetDefaultPrim()
    return bool(default_prim) and {
        "plan": default_prim.GetCustomDataByKey(
            "jointAgent:physicsAuthoringPlanSha256"
        ),
        "diagnostics": default_prim.GetCustomDataByKey(
            "jointAgent:physicsStage2DiagnosticsSha256"
        ),
        "validation": default_prim.GetCustomDataByKey(
            "jointAgent:physicsStage2ValidationSha256"
        ),
    } == {
        "plan": plan_sha256,
        "diagnostics": stage2_diagnostics_sha256,
        "validation": stage2_validation_sha256,
    }


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_manifest(
    path: Path,
    *,
    logical_root_path: Path | None = None,
) -> dict[str, Any]:
    if path.suffix.lower() == ".usdz":
        entries: list[dict[str, Any]] = []
        with zipfile.ZipFile(path) as archive:
            archive_entries = archive.infolist()
            if not archive_entries:
                raise PhysicsAuthoringError(
                    "artifact_identity_mismatch",
                    f"USDZ package is empty: {path}",
                )
            names = [info.filename for info in archive_entries]
            if len(names) != len(set(names)):
                raise PhysicsAuthoringError(
                    "artifact_identity_mismatch",
                    f"USDZ package contains duplicate entry names: {path}",
                )
            root_info = archive_entries[0]
            if root_info.is_dir() or Path(root_info.filename).suffix.lower() not in (
                _RAW_USD_EXTENSIONS
            ):
                raise PhysicsAuthoringError(
                    "artifact_identity_mismatch",
                    "USDZ first entry must be a USD root layer: "
                    f"{root_info.filename!r} in {path}",
                )
            root_entry = root_info.filename
            for info in archive_entries:
                _validate_usdz_entry(info, path=path)
                digest = hashlib.sha256()
                if not info.is_dir():
                    with archive.open(info) as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                entries.append(
                    {
                        "path": info.filename,
                        "is_root": info.filename == root_entry,
                        "size": info.file_size,
                        "sha256": digest.hexdigest(),
                    }
                )
        return {
            "container": "usdz",
            "package_sha256": _file_sha256(path),
            "root_entry": root_entry,
            "entries": sorted(entries, key=lambda record: record["path"]),
        }
    from pxr import UsdUtils

    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:
        raise PhysicsAuthoringError(
            "artifact_identity_mismatch",
            f"could not enumerate USD dependencies for {path}: {exc}",
        ) from exc

    root_path = path.expanduser().resolve()
    logical_root = (logical_root_path or path).expanduser().resolve()

    def file_record(value: Any, *, kind: str) -> dict[str, Any]:
        real_path = getattr(value, "realPath", None)
        identifier = str(getattr(value, "identifier", value))
        asset_path = getattr(value, "path", None)
        resolved_asset_path = getattr(value, "resolvedPath", None)
        candidate = Path(
            real_path or resolved_asset_path or asset_path or identifier
        ).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        candidate = candidate.resolve()
        is_root = kind == "layer" and candidate == root_path
        manifest_path = logical_root if is_root else candidate
        return {
            "kind": kind,
            "identifier": str(logical_root) if is_root else identifier,
            "is_root": is_root,
            "resolved_path": str(manifest_path) if candidate.is_file() else None,
            "sha256": _file_sha256(candidate) if candidate.is_file() else None,
        }

    layer_records = sorted(
        (file_record(layer, kind="layer") for layer in layers),
        key=lambda item: item["identifier"],
    )
    asset_records = sorted(
        (file_record(asset, kind="asset") for asset in assets),
        key=lambda item: item["identifier"],
    )
    return {
        "container": "raw_usd",
        "layers": layer_records,
        "assets": asset_records,
        "unresolved": sorted(str(value) for value in unresolved),
    }


def _validate_usdz_entry(info: zipfile.ZipInfo, *, path: Path) -> None:
    name = info.filename
    pure_path = PurePosixPath(name)
    if not name or "\\" in name or pure_path.is_absolute() or ".." in pure_path.parts:
        raise PhysicsAuthoringError(
            "artifact_identity_mismatch",
            f"USDZ package contains unsafe entry path {name!r}: {path}",
        )
    if info.flag_bits & 0x1:
        raise PhysicsAuthoringError(
            "artifact_identity_mismatch",
            f"USDZ package contains encrypted entry {name!r}: {path}",
        )
    unix_mode = (info.external_attr >> 16) & 0o170000
    if unix_mode == 0o120000:
        raise PhysicsAuthoringError(
            "artifact_identity_mismatch",
            f"USDZ package contains symbolic-link entry {name!r}: {path}",
        )


def _validate_dependency_identity_preserved(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> bool:
    if before.get("container") != after.get("container"):
        raise PhysicsAuthoringError(
            "postwrite_validation_failed",
            "authoring changed the USD container kind",
        )
    if before.get("container") == "usdz":

        def package_dependencies(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
            return sorted(
                (
                    {
                        "path": record.get("path"),
                        "size": record.get("size"),
                        "sha256": record.get("sha256"),
                    }
                    for record in manifest.get("entries", [])
                    if not record.get("is_root", False)
                ),
                key=lambda record: str(record["path"]),
            )

        before_entries = package_dependencies(before)
        after_entries = package_dependencies(after)
        if before_entries != after_entries:
            raise PhysicsAuthoringError(
                "postwrite_validation_failed",
                "authoring changed non-root USDZ package entries: "
                f"before={before_entries}, after={after_entries}",
            )
        return True

    def external_records(
        manifest: Mapping[str, Any],
        key: str,
    ) -> list[dict[str, Any]]:
        return sorted(
            (
                {
                    "kind": record.get("kind"),
                    "resolved_path": record.get("resolved_path"),
                    "sha256": record.get("sha256"),
                }
                for record in manifest.get(key, [])
                if not record.get("is_root", False)
            ),
            key=lambda record: (
                str(record["kind"]),
                str(record["resolved_path"]),
                str(record["sha256"]),
            ),
        )

    fields = {
        "layers": (
            external_records(before, "layers"),
            external_records(after, "layers"),
        ),
        "assets": (
            external_records(before, "assets"),
            external_records(after, "assets"),
        ),
        "unresolved": (
            sorted(str(value) for value in before.get("unresolved", [])),
            sorted(str(value) for value in after.get("unresolved", [])),
        ),
    }
    changed = {
        key: {"before": old, "after": new}
        for key, (old, new) in fields.items()
        if old != new
    }
    if changed:
        raise PhysicsAuthoringError(
            "postwrite_validation_failed",
            f"authoring changed external dependency identity: {changed}",
        )
    return True


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _optional_canonical_json_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return _canonical_json_sha256(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = _stage_json_file(path, payload)
    try:
        _commit_staged_artifacts(((temporary, path),))
    finally:
        if temporary.exists():
            _unlink_temporary(temporary)


def _stage_json_file(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        return temporary
    except Exception:
        if temporary is not None and temporary.exists():
            _unlink_temporary(temporary)
        raise


def _commit_staged_artifacts(
    artifacts: tuple[tuple[Path, Path], ...],
) -> None:
    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    committed = False
    try:
        for _, target in artifacts:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                continue
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".backup",
                delete=False,
            ) as stream:
                backup = Path(stream.name)
            backup.unlink()
            _replace_path(target, backup)
            backups[target] = backup
        for staged, target in artifacts:
            _replace_path(staged, target)
            promoted.append(target)
        committed = True
    except Exception as commit_error:
        rollback_errors: list[str] = []
        for target in reversed(promoted):
            try:
                if target.exists() or target.is_symlink():
                    target.unlink()
            except OSError as exc:
                rollback_errors.append(f"could not remove promoted {target}: {exc}")
        for target, backup in backups.items():
            if backup.exists():
                try:
                    _replace_path(backup, target)
                except OSError as exc:
                    rollback_errors.append(
                        f"could not restore {target} from {backup}: {exc}"
                    )
        recovery_paths = [str(backup) for backup in backups.values() if backup.exists()]
        detail = f"artifact promotion failed: {commit_error}"
        if rollback_errors:
            detail += "; rollback errors: " + "; ".join(rollback_errors)
        if recovery_paths:
            detail += "; preserved recovery backups: " + ", ".join(recovery_paths)
        raise PhysicsAuthoringError("artifact_write_failed", detail) from commit_error
    finally:
        if committed:
            for backup in backups.values():
                if backup.exists():
                    _unlink_temporary(backup)


def _replace_path(source: Path, target: Path) -> None:
    source.replace(target)


def _unlink_temporary(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _require_sha256(value: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("must be a lowercase hexadecimal SHA256")


def _require_finite(value: float, label: str) -> None:
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")


def _require_positive_finite(value: float, label: str) -> None:
    _require_finite(value, label)
    if float(value) <= 0:
        raise ValueError(f"{label} must be positive")


def _require_nonnegative_finite(value: float, label: str) -> None:
    _require_finite(value, label)
    if float(value) < 0:
        raise ValueError(f"{label} must be non-negative")


def _require_normalized_quaternion(values: Any, label: str) -> None:
    if len(values) != 4:
        raise ValueError(f"{label} must contain four values")
    for index, value in enumerate(values):
        _require_finite(float(value), f"{label}[{index}]")
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"{label} must be normalized")


def _require_inertia_triangle(values: Any, label: str) -> None:
    components = [float(value) for value in values]
    for index, component in enumerate(components):
        other_sum = sum(components) - component
        tolerance = max(abs(component), abs(other_sum), 1.0) * 1e-9
        if component > other_sum + tolerance:
            raise ValueError(
                f"{label}[{index}] violates the rigid-body inertia triangle"
            )
