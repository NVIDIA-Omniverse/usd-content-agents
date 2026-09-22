# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic physics authoring workflow implementation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SkipValidation,
    field_validator,
    model_validator,
)

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    prepare_writable_directory,
    prepare_writable_file_path,
    read_contained_artifact,
)
from content_agent_workflows.common.run_record import WorkflowRunRecorder
from content_agent_workflows.common.usd_cli_session import WorkflowUsdCliSession
from content_agent_workflows.common.validation_evidence import (
    EvidenceArtifact,
    ValidationCheck,
    ValidationEvidence,
    physics_validation_evidence,
)
from content_agent_workflows.physics.policy import infer_material_profile

from . import scene_ops, usd_cli_ops
from .vomp import (
    PhysicsVompMassConfig,
    PhysicsVompMassResult,
    resolve_vomp_target_prim,
    run_agentic_vomp_mass_authoring,
    verify_vomp_mass_properties,
)

PhysicsSimulationEngine = Literal["ovphysx", "fake", "none"]
PhysicsSceneBackend = Literal["usd-cli"]
_VOMP_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})

PHYSICS_ASSIGNMENTS_SCHEMA_VERSION = "content-agent-workflows.physics-assignments.v1"
PHYSICS_DECISION_PATCH_SCHEMA_VERSION = (
    "content-agent-workflows.physics-decision-patch.v2"
)
LEGACY_PHYSICS_DECISION_PATCH_SCHEMA_VERSION = (
    "content-agent-workflows.physics-decision-patch.v1"
)
PHYSICS_BEHAVIOR_ASSESSMENT_SCHEMA_VERSION = (
    "content-agent-workflows.physics-behavior-assessment.v1"
)
MAX_PHYSICS_INPUT_JSON_BYTES = 16 * 1024 * 1024
MAX_PHYSICS_USD_BYTES = 512 * 1024 * 1024


class PhysicsCandidate(BaseModel):
    """A V1 mesh candidate retained for compatibility callers."""

    model_config = ConfigDict(extra="forbid")

    prim_path: str = Field(min_length=1)
    prim_name: str = Field(min_length=1)
    type_name: str = Field(min_length=1)
    material_path: str | None = None
    material_name: str | None = None
    bbox_min_m: list[float]
    bbox_max_m: list[float]
    bbox_size_m: list[float]
    bbox_volume_m3: float
    existing_physics_schemas: list[str] = Field(default_factory=list)
    path_space: str = "source"


class PhysicsDecision(BaseModel):
    """One V1 mesh-target decision retained for compatibility callers."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(min_length=1)
    prim_paths: list[str] = Field(min_length=1)
    component_label: str = Field(min_length=1)
    inferred_material_family: str = Field(min_length=1)
    inferred_material_name: str | None = None
    collision_approximation: str = Field(min_length=1)
    physical_properties: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class PhysicsComponent(BaseModel):
    """A logical physics unit with evidence and authoring roles separated."""

    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(min_length=1)
    component_role: Literal["body", "unowned_static"] = "body"
    path_space: str = "source"
    body_root_path: str = Field(min_length=1)
    visual_evidence_paths: list[str] = Field(default_factory=list)
    collider_paths: list[str] = Field(default_factory=list)
    helper_paths: list[str] = Field(default_factory=list)
    rigid_body_paths: list[str] = Field(default_factory=list)
    joint_paths: list[str] = Field(default_factory=list)
    material_evidence: list[dict[str, str]] = Field(default_factory=list)
    bounds_m: dict[str, Any] = Field(default_factory=dict)
    topology_findings: list[str] = Field(default_factory=list)


def _validate_quality_warnings(
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Require warning records that downstream authoring policy can evaluate."""

    for index, warning in enumerate(warnings):
        code = warning.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"quality_warnings[{index}].code must be non-empty")
        severity = warning.get("severity", "warning")
        if severity not in {"info", "warning", "error"}:
            raise ValueError(
                f"quality_warnings[{index}].severity must be info, warning, or error"
            )
        if code == "mass_scale_suspicious" and severity != "warning":
            raise ValueError(
                "mass_scale_suspicious quality warnings must use severity='warning'"
            )
    return warnings


_CANONICAL_PHYSICAL_PROPERTY_KEYS = (
    "density",
    "estimated_mass_kg",
    "static_friction",
    "dynamic_friction",
    "restitution",
)


def _validated_physical_properties(
    properties: dict[str, float],
) -> dict[str, float]:
    """Shared guard for both V2 decision models' ``physical_properties``.

    The child-facing patch-schema example fixes the JSON types with 0.0
    placeholders. A copied placeholder vector is type-valid but physically
    meaningless -- the apply path would skip the nonpositive density and mass
    while authoring a frictionless, rebound-free material -- so fail it loudly
    instead of authoring it silently. Non-finite values are rejected for the
    same reason: they parse as floats but poison every downstream mass and
    material computation.
    """

    for key, value in properties.items():
        if not math.isfinite(value):
            raise ValueError(
                f"physical_properties[{key!r}] must be a finite number, got {value!r}."
            )
    # The placeholder signature compares the canonical five keys directly, so
    # a nonzero auxiliary key (e.g. "fill_fraction") beside the copied zeros
    # cannot smuggle the placeholder past the guard -- downstream authoring
    # ignores auxiliary keys and would still act on the zeros. An
    # unowned_static fixture legitimately authors zero density and mass
    # without the friction/restitution keys, so it does not match.
    canonical_values = [
        properties[key]
        for key in _CANONICAL_PHYSICAL_PROPERTY_KEYS
        if key in properties
    ]
    zeroed_friction_keys = {
        "static_friction",
        "dynamic_friction",
        "restitution",
    } & set(properties)
    if (
        canonical_values
        and all(value == 0.0 for value in canonical_values)
        and zeroed_friction_keys
    ):
        raise ValueError(
            "physical_properties is the copied schema placeholder (every "
            "physical value is 0.0); author real values derived from the "
            "inferred material family."
        )
    return properties


class PhysicsMassProperties(BaseModel):
    """Explicit body-local MassAPI values in stage distance/mass units.

    Inertia is about the center of mass; principal_axes is (w, x, y, z).
    These are caller-authored estimates, not inferred or measured properties.
    """

    model_config = ConfigDict(extra="forbid")

    center_of_mass: tuple[float, float, float]
    diagonal_inertia: tuple[float, float, float]
    principal_axes: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @field_validator("center_of_mass", "diagonal_inertia", "principal_axes", mode="before")
    @classmethod
    def _plain_numeric_vectors(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
        ):
            raise ValueError("mass-property vectors must contain plain numbers")
        return value

    @model_validator(mode="after")
    def _physical_values(self) -> "PhysicsMassProperties":
        import struct

        values = (*self.center_of_mass, *self.diagonal_inertia, *self.principal_axes)
        try:
            stored = [struct.unpack("f", struct.pack("f", value))[0] for value in values]
        except (OverflowError, struct.error) as exc:
            raise ValueError("mass properties must be representable as finite USD floats") from exc
        if not all(math.isfinite(value) for value in stored):
            raise ValueError("mass properties must be finite")
        inertia = stored[3:6]
        if min(inertia) <= 0.0:
            raise ValueError("diagonal inertia must be positive")
        if 2 * max(inertia) > sum(inertia) + 1e-6 * max(inertia):
            raise ValueError("diagonal inertia violates the principal-moment triangle inequality")
        if abs(sum(value * value for value in stored[6:]) - 1.0) > 1e-5:
            raise ValueError("principal_axes must be a normalized (w,x,y,z) quaternion")
        return self


class PhysicsConvexDecompositionOptions(BaseModel):
    """Explicit PhysX cooking options; bounds limit workflow cooking resources.

    Defaults match the OvPhysX 0.4.13 USD schema. Omitting the entire record
    preserves existing authoring behavior; providing it authors all five values.
    These are cooking controls, not a geometry-fidelity guarantee.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    shrink_wrap: bool = False
    error_percentage: float = Field(default=10.0, ge=0.0, le=100.0, allow_inf_nan=False)
    hull_vertex_limit: int = Field(default=64, ge=4, le=255)
    max_convex_hulls: int = Field(default=32, ge=1, le=256)
    voxel_resolution: int = Field(default=500_000, ge=10_000, le=4_000_000)


class PhysicsComponentDecision(BaseModel):
    """One accepted V2 component-level physics authoring decision."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    component_role: Literal["body", "unowned_static"] = "body"
    body_root_path: str = Field(min_length=1)
    visual_evidence_paths: list[str] = Field(default_factory=list)
    collider_paths: list[str] = Field(min_length=1)
    collision_mode: Literal["preserve_existing", "author_on_targets"]
    mass_authoring_path: str = Field(min_length=1)
    inferred_material_family: str = Field(min_length=1)
    inferred_material_name: str | None = None
    collision_approximation: str = Field(min_length=1)
    physical_properties: dict[str, float]
    mass_properties: PhysicsMassProperties | None = None
    convex_decomposition: PhysicsConvexDecompositionOptions | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    rigid_body_grouping: str | None = None
    quality_warnings: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cooking_approximation(self) -> "PhysicsComponentDecision":
        if self.convex_decomposition is not None and self.collision_approximation != "convexDecomposition":
            raise ValueError("convex_decomposition requires convexDecomposition approximation")
        return self

    @field_validator("quality_warnings")
    @classmethod
    def _quality_warning_records(
        cls, warnings: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return _validate_quality_warnings(warnings)

    @field_validator("physical_properties")
    @classmethod
    def _physical_property_records(
        cls, properties: dict[str, float]
    ) -> dict[str, float]:
        return _validated_physical_properties(properties)

    @model_validator(mode="before")
    @classmethod
    def _fill_author_on_targets_colliders(cls, data: Any) -> Any:
        """When authoring colliders on visible geometry, fall back to the
        identified ``visual_evidence_paths`` if the agent left
        ``collider_paths`` empty. Mirrors the auto-inference path
        (``component.collider_paths or component.visual_evidence_paths``) so an
        ``author_on_targets`` decision that already names its target geometry is
        not rejected purely for not duplicating it into ``collider_paths``.
        """
        if isinstance(data, dict) and data.get("collision_mode") == "author_on_targets":
            if not data.get("collider_paths"):
                targets = data.get("visual_evidence_paths") or []
                if targets:
                    data = {**data, "collider_paths": list(targets)}
        return data


class PhysicsComponentTargetDecision(BaseModel):
    """Canonical V2 decision that selects inspected targets by stable ID."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    collider_target_ids: list[str] = Field(min_length=1)
    collision_mode: Literal["preserve_existing", "author_on_targets"]
    inferred_material_family: str = Field(min_length=1)
    inferred_material_name: str | None = None
    collision_approximation: str = Field(min_length=1)
    physical_properties: dict[str, float]
    mass_properties: PhysicsMassProperties | None = None
    convex_decomposition: PhysicsConvexDecompositionOptions | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    rigid_body_grouping: str | None = None
    quality_warnings: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cooking_approximation(self) -> "PhysicsComponentTargetDecision":
        if self.convex_decomposition is not None and self.collision_approximation != "convexDecomposition":
            raise ValueError("convex_decomposition requires convexDecomposition approximation")
        return self

    @field_validator("quality_warnings")
    @classmethod
    def _quality_warning_records(
        cls, warnings: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return _validate_quality_warnings(warnings)

    @field_validator("physical_properties")
    @classmethod
    def _physical_property_records(
        cls, properties: dict[str, float]
    ) -> dict[str, float]:
        return _validated_physical_properties(properties)


class PhysicsApplyWorkflowInput(BaseModel):
    """Input for the agentic physics apply workflow."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    usd_path: Path
    inspection_asset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_usd_path: Path | None = None
    source_asset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    path_space: Literal["source", "inspection"] = "source"
    source_path_expansions: dict[str, list[str]] = Field(default_factory=dict)
    output_dir: Path
    output_usd_path: Path | None = None
    decision_patch_path: Path | None = None
    topology_plan_path: Path | None = None
    collision_approximation: str = "convexHull"
    run_simulation: bool = True
    simulation_engine: PhysicsSimulationEngine = "ovphysx"
    simulation_duration_s: float = 3.0
    simulation_dt: float = 1.0 / 240.0
    simulation_sample_fps: int = 30
    drop_height_m: float | None = None
    runtime_placement_mode: Literal["drop", "mounted"] = "drop"
    vomp_mass: PhysicsVompMassConfig | None = None
    vomp_artifact_namespace: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
    )
    # Acceptance override for the runtime ground-clearance gate. Without this the
    # apply phase enforces the runtime default (scale-relative on exact
    # collider geometry, 0.005 m on the conservative bbox fallback), so a run
    # configured for a deeper legitimate rest records a hard failure that the
    # refinement loop is
    # simultaneously told not to fix. Bounded to mirror the runtime request
    # schema (PhysicsRuntimeAcceptance caps at 1.0 m) so an out-of-range value
    # fails here instead of as a remote 422 mid-finalization.
    max_ground_penetration_m: float | None = Field(default=None, gt=0, le=1.0)
    fail_on_validation_error: bool = False
    scene_backend: PhysicsSceneBackend = "usd-cli"
    scene_tool_timeout_seconds: float = Field(default=300.0, gt=0.0)
    usd_cli_session: WorkflowUsdCliSession | None = Field(default=None, exclude=True)
    # Wrapper-owned in-memory state. It is deliberately excluded from request
    # artifacts: a child-writable JSON file must never control validation
    # support reuse on a later finalization pass.
    ground_clearance_support_cache: SkipValidation[dict[str, dict[str, Any]]] | None = (
        Field(
            default=None,
            exclude=True,
        )
    )
    resume: bool = False
    workflow_run_record_subdir: Path | None = None

    @field_validator("ground_clearance_support_cache", mode="before")
    @classmethod
    def _preserve_parent_owned_support_cache(cls, value: Any) -> Any:
        """Validate the cache container without copying its object identity."""

        if value is not None and not isinstance(value, dict):
            raise ValueError("ground_clearance_support_cache must be a dict")
        return value


class PhysicsApplyWorkflowResult(BaseModel):
    """Result and canonical artifacts from a physics apply workflow."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    asset: str
    output_dir: str
    physics_usd_path: str | None = None
    assignments_path: str | None = None
    decision_patch_path: str | None = None
    components_path: str | None = None
    candidate_prims_path: str | None = None
    predictions_path: str | None = None
    apply_report_path: str | None = None
    topology_report_path: str | None = None
    validation_evidence_path: str | None = None
    simulation_report_path: str | None = None
    behavior_assessment_path: str | None = None
    scene_operation_record_path: str | None = None
    vomp_result_path: str | None = None
    vomp_provenance_path: str | None = None
    validation_status: str = "not_evaluated"
    workflow_run_manifest_path: str | None = None
    error: str | None = None


class PhysicsBehaviorAssessment(BaseModel):
    """Agent-authored review of rendered physics runtime behavior."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PHYSICS_BEHAVIOR_ASSESSMENT_SCHEMA_VERSION
    status: Literal["pass", "fixed", "unresolved_issues"]
    checked_views: list[str] = Field(default_factory=list)
    runtime_report: str | None = None
    rendered_frames: list[str] = Field(default_factory=list)
    issues_found: list[Any] = Field(default_factory=list)
    issues_fixed: list[Any] = Field(default_factory=list)
    unresolved_issues: list[Any] = Field(default_factory=list)
    assessment_notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_shapes(cls, data: Any) -> Any:
        """Coerce the richer shapes the visual-review LLM reliably emits into the
        strict contract: ``runtime_report`` as an object, ``rendered_frames`` /
        ``checked_views`` as objects or lists of per-view objects, and
        ``assessment_notes`` as a list. Also drops unknown keys (which
        ``extra="forbid"`` would otherwise reject) so an assessment that is valid
        apart from LLM formatting is not rejected — a rejection here aborts
        finalize and gates out the downstream tune phase.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)

        # runtime_report: contract wants a path string; the LLM often nests a metrics object.
        report = data.get("runtime_report")
        if isinstance(report, dict):
            report_path = (
                report.get("path")
                or report.get("report_path")
                or report.get("runtime_report_path")
            )
            data["runtime_report"] = str(report_path) if report_path else None
        elif isinstance(report, list | tuple):
            data["runtime_report"] = None

        # rendered_frames / checked_views: contract wants list[str]; the LLM may
        # emit a summary object, a bare string, or a list of per-view/-frame
        # objects — coerce all of those to a flat list of strings.
        def _as_str_list(value: Any, *, preserve_object_details: bool) -> Any:
            if isinstance(value, dict):
                nested = value.get("frames") or value.get("paths") or value.get("files")
                value = (
                    nested
                    if isinstance(nested, list)
                    else (
                        [json.dumps(value, sort_keys=True)]
                        if preserve_object_details
                        else []
                    )
                )
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                items: list[str] = []
                for item in value:
                    if isinstance(item, str):
                        items.append(item)
                        continue
                    if isinstance(item, dict):
                        if preserve_object_details:
                            items.append(json.dumps(item, sort_keys=True))
                            continue
                        path = (
                            item.get("path")
                            or item.get("frame_path")
                            or item.get("image_path")
                        )
                        if path:
                            items.append(str(path))
                        continue
                    items.append(str(item))
                return items
            return value  # leave other types for the model to reject

        if "rendered_frames" in data:
            data["rendered_frames"] = _as_str_list(
                data["rendered_frames"],
                preserve_object_details=False,
            )
        if "checked_views" in data:
            data["checked_views"] = _as_str_list(
                data["checked_views"],
                preserve_object_details=True,
            )

        # assessment_notes: contract wants a single string; the LLM often emits a list/object.
        notes = data.get("assessment_notes")
        if isinstance(notes, list):
            data["assessment_notes"] = "\n".join(str(item) for item in notes)
        elif isinstance(notes, dict):
            data["assessment_notes"] = json.dumps(notes, sort_keys=True)
        elif notes is not None and not isinstance(notes, str):
            data["assessment_notes"] = str(notes)

        return {key: value for key, value in data.items() if key in cls.model_fields}


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    within: Path | None = None,
) -> Path:
    """Atomically write JSON without following a child-authored path component."""

    return atomic_write_json(path, _json_safe(payload), within=within)


def _read_json_object(
    path: Path | str,
    *,
    within: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Read one bounded regular JSON object through no-follow descriptors."""

    candidate = Path(path).expanduser()
    if within is not None:
        root = within.expanduser().resolve(strict=True)
        absolute = Path(os.path.abspath(candidate))
        artifact = read_contained_artifact(
            root,
            absolute,
            max_bytes=MAX_PHYSICS_INPUT_JSON_BYTES,
            parse_json=True,
        )
    else:
        absolute = Path(os.path.abspath(candidate))
        parent = absolute.parent.resolve(strict=True)
        artifact = read_contained_artifact(
            parent,
            parent / absolute.name,
            max_bytes=MAX_PHYSICS_INPUT_JSON_BYTES,
            parse_json=True,
        )
    if artifact.json_object is None:
        raise RuntimeError(f"Expected a JSON object in {artifact.path}")
    return artifact.path, artifact.json_object


def _contained_input_root(path: Path | str, output_dir: Path) -> Path | None:
    """Select strict run containment for a lexically run-local input."""

    root = output_dir.expanduser().resolve(strict=True)
    candidate = Path(path).expanduser()
    absolute = Path(
        os.path.abspath(candidate if candidate.is_absolute() else candidate)
    )
    try:
        absolute.relative_to(root)
    except ValueError:
        return None
    return root


def _read_regular_bytes(
    path: Path | str,
    *,
    within: Path | None = None,
) -> bytes:
    """Capture one bounded regular file through a held no-follow descriptor."""

    candidate = Path(path).expanduser()
    if within is not None:
        root = within.expanduser().resolve(strict=True)
        absolute = Path(os.path.abspath(candidate))
        artifact = read_contained_artifact(
            root,
            absolute,
            max_bytes=MAX_PHYSICS_USD_BYTES,
            capture_bytes=True,
        )
    else:
        absolute = Path(os.path.abspath(candidate))
        parent = absolute.parent.resolve(strict=True)
        artifact = read_contained_artifact(
            parent,
            parent / absolute.name,
            max_bytes=MAX_PHYSICS_USD_BYTES,
            capture_bytes=True,
        )
    assert artifact.data is not None
    return artifact.data


def _bind_validation_evidence_asset(
    evidence: ValidationEvidence,
    asset: Path,
) -> None:
    """Bind native validation claims to the exact authored file bytes."""

    resolved = asset.resolve(strict=True)
    evidence.asset = str(resolved)
    evidence.metadata = {
        **evidence.metadata,
        "asset_sha256": file_sha256(resolved),
    }


def _apply_vomp_mass_if_requested(
    *,
    params: PhysicsApplyWorkflowInput,
    decisions: Iterable[object],
    mobility_intent: str,
    authored_usd: Path,
    output_usd: Path,
    output_dir: Path,
    raw_dir: Path,
) -> tuple[Path, PhysicsVompMassResult | None, Path | None]:
    config = params.vomp_mass
    if config is None:
        return authored_usd, None, None
    accepted_decisions = list(decisions)
    if not accepted_decisions:
        raise RuntimeError(
            "VoMP mass authoring requires at least one accepted body decision."
        )
    if mobility_intent == "static":
        raise RuntimeError(
            "VoMP mass authoring requires an enabled rigid body and cannot run "
            "with static mobility intent."
        )
    _validate_vomp_output_suffix(output_usd)
    if not authored_usd.is_file():
        raise RuntimeError(
            "VoMP requires the schema-authored USD to be readable on the agentic "
            f"wrapper host: {authored_usd}"
        )
    source = authored_usd.resolve(strict=True)
    output = output_usd.resolve()
    original_input = Path(params.usd_path).resolve()
    if original_input == output:
        raise RuntimeError(
            "VoMP canonical output USD must differ from the original input USD"
        )
    if source == output:
        raise RuntimeError("VoMP input and canonical output USD paths must differ")

    artifact_namespace = params.vomp_artifact_namespace
    if artifact_namespace is not None:
        immutable_root = raw_dir / "vomp" / artifact_namespace
        source = _copy_remote_usd_artifact(
            source,
            immutable_root / f"physics_pre_vomp{source.suffix.lower()}",
        )

    target_prim_path = resolve_vomp_target_prim(config, accepted_decisions)
    canonical_result = run_agentic_vomp_mass_authoring(
        input_usd_path=source,
        output_usd_path=output,
        output_dir=output_dir,
        target_prim_path=target_prim_path,
        config=config,
        artifact_namespace=artifact_namespace,
    )
    authored = Path(canonical_result.output_usd_path).resolve(strict=True)
    result = canonical_result
    result_path = raw_dir / "physics_vomp_result.json"
    if artifact_namespace is not None:
        immutable_output = _copy_remote_usd_artifact(
            authored,
            immutable_root / f"physics_vomp_output{authored.suffix.lower()}",
        )
        result = canonical_result.model_copy(
            update={
                "output_usd_path": str(immutable_output),
                "output_usd_sha256": file_sha256(immutable_output),
            }
        )
        verify_vomp_mass_properties(immutable_output, result)
        result_path = immutable_root / "physics_vomp_result.json"
        _write_json(result_path, result.model_dump(mode="json"))
        canonical_provenance = raw_dir / "physics_vomp_mass_properties.json"
        shutil.copy2(result.provenance_path, canonical_provenance)
        _write_json(
            raw_dir / "physics_vomp_result.json",
            canonical_result.model_dump(mode="json"),
        )
    else:
        _write_json(result_path, result.model_dump(mode="json"))
    return authored, result, result_path


def _accepted_vomp_body_decisions(
    decisions: Iterable[PhysicsDecision | PhysicsComponentDecision],
    components: Iterable[PhysicsComponent],
) -> list[PhysicsDecision | PhysicsComponentDecision]:
    """Exclude inspected static-only components from VoMP target selection."""

    component_roles = {
        component.component_id: component.component_role for component in components
    }
    return [
        decision
        for decision in decisions
        if not isinstance(decision, PhysicsComponentDecision)
        or component_roles.get(decision.component_id) == "body"
    ]


def _validate_vomp_output_suffix(output_usd: Path) -> None:
    if output_usd.suffix.lower() not in _VOMP_USD_SUFFIXES:
        raise RuntimeError("Agentic VoMP output must be .usd, .usda, or .usdc")


def _resolved_usd_asset_dependency(asset: object, *, root: Path) -> Path | None:
    resolved_path = str(getattr(asset, "resolvedPath", "") or "")
    if resolved_path:
        candidate = Path(resolved_path)
        if candidate.is_file():
            return candidate.resolve()
    authored_path = str(getattr(asset, "path", "") or str(asset))
    candidate = Path(authored_path)
    if not candidate.is_absolute():
        candidate = root.parent / candidate
    return candidate.resolve() if candidate.is_file() else None


def _remote_usd_dependency_files(root: Path) -> set[Path]:
    """Resolve the filesystem closure of a portable USD export."""

    from pxr import UsdUtils
    from world_understanding.functions.graphics.so_export import (
        is_runtime_resolved_asset_path,
    )

    source = root.resolve(strict=True)
    dependencies = {source}
    if source.suffix.lower() == ".usdz":
        return dependencies

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(source))
    rejected_unresolved = [
        item for item in unresolved if not is_runtime_resolved_asset_path(item)
    ]
    if rejected_unresolved:
        sample = ", ".join(str(item) for item in rejected_unresolved[:5])
        suffix = "" if len(rejected_unresolved) <= 5 else ", ..."
        raise RuntimeError(
            f"Remote USD export contains unresolved dependencies: {sample}{suffix}"
        )
    for layer in layers:
        real_path = str(getattr(layer, "realPath", "") or "")
        if real_path and Path(real_path).is_file():
            dependencies.add(Path(real_path).resolve())
    for asset in assets:
        authored_path = getattr(asset, "path", None) or asset
        if is_runtime_resolved_asset_path(authored_path):
            continue
        dependency = _resolved_usd_asset_dependency(asset, root=source)
        if dependency is not None:
            dependencies.add(dependency)

    export_root = source.parent
    escaped = sorted(
        dependency
        for dependency in dependencies
        if dependency != source and export_root not in dependency.parents
    )
    if escaped:
        raise RuntimeError(
            "Remote USD export is not portable; dependencies escape "
            f"{export_root}: {', '.join(str(path) for path in escaped)}"
        )
    return dependencies


def _portable_sidecar_marker_dependency(
    root: Path,
    dependencies: set[Path],
) -> Path | None:
    """Return a validated ownership marker for a referenced portable sidecar."""

    from world_understanding.functions.graphics.so_export import (
        PORTABLE_SIDECAR_MARKER_NAME,
        _require_owned_sidecar,
        portable_sidecar_name,
    )

    sidecar = root.parent / portable_sidecar_name(root)
    resolved_sidecar = sidecar.resolve()
    if not any(
        dependency != root and dependency.is_relative_to(resolved_sidecar)
        for dependency in dependencies
    ):
        return None

    marker = sidecar / PORTABLE_SIDECAR_MARKER_NAME
    if not marker.is_file():
        return None
    _require_owned_sidecar(sidecar)
    return marker.resolve(strict=True)


def _copy_remote_usd_artifact(source: Path, target: Path) -> Path:
    """Copy a remote root and its complete resolved portable dependency bundle."""

    remote_root = source.resolve(strict=True)
    canonical_root = target.resolve()
    if remote_root == canonical_root:
        return canonical_root

    dependencies = _remote_usd_dependency_files(remote_root)
    sidecar_marker = _portable_sidecar_marker_dependency(remote_root, dependencies)
    existing_destination_marker: Path | None = None
    if sidecar_marker is not None:
        dependencies.add(sidecar_marker)
        destination_sidecar = canonical_root.parent / sidecar_marker.parent.relative_to(
            remote_root.parent
        )
        if destination_sidecar.is_symlink() or destination_sidecar.exists():
            from world_understanding.functions.graphics.so_export import (
                _require_owned_sidecar,
            )

            _require_owned_sidecar(destination_sidecar)
            existing_destination_marker = destination_sidecar / sidecar_marker.name

    copy_plan: list[tuple[Path, Path]] = []
    for dependency in dependencies:
        destination = (
            canonical_root
            if dependency == remote_root
            else canonical_root.parent / dependency.relative_to(remote_root.parent)
        )
        copy_plan.append((dependency, destination))

    # Publish dependencies before the root layer so a partially failed copy does
    # not expose a new root whose relative assets are still absent.
    copy_plan.sort(key=lambda item: (item[0] == remote_root, str(item[0])))
    for dependency, destination in copy_plan:
        if dependency == destination or destination == existing_destination_marker:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dependency, destination)
    return canonical_root.resolve(strict=True)


def _vomp_evidence_artifacts(
    result: PhysicsVompMassResult | None,
    result_path: Path | None,
) -> list[EvidenceArtifact]:
    if result is None or result_path is None:
        return []
    return [
        EvidenceArtifact(
            kind="vomp_mass_result",
            path=str(result_path),
            description="Agentic VoMP mass-property result and canonical artifacts.",
        ),
        EvidenceArtifact(
            kind="vomp_mass_provenance",
            path=result.provenance_path,
            description="Attested OVRTX and VoMP mass-property provenance.",
        ),
        EvidenceArtifact(
            kind="vomp_evidence_manifest",
            path=result.evidence_manifest_path,
            description="Calibrated OVRTX evidence manifest consumed by VoMP.",
        ),
    ]


def _bind_vomp_validation_metadata(
    evidence: ValidationEvidence,
    result: PhysicsVompMassResult | None,
    result_path: Path | None,
) -> None:
    if result is None:
        return
    evidence.evidence_artifacts.extend(_vomp_evidence_artifacts(result, result_path))
    evidence.metadata = {
        **evidence.metadata,
        "mass_property_provider": "vomp",
        "vomp_target_prim_path": result.target_prim_path,
        "vomp_sample_count": result.sample_count,
        "vomp_mass_kg": result.mass_kg,
        "vomp_provenance_sha256": result.provenance_sha256,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _as_json(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _ground_clearance_support_context_key(
    *,
    source_digest: str,
    decisions: list[
        PhysicsDecision | PhysicsComponentDecision | PhysicsComponentTargetDecision
    ],
) -> str:
    """Bind support reuse to geometry plus collider-authoring structure.

    Numeric physics properties are intentionally excluded: tuning changes
    mass, friction, and restitution without changing pose-local collider
    support. The immutable inspected-source digest binds geometry, extents,
    transforms, and stage metadata.
    """

    collider_contract: list[dict[str, Any]] = []
    for decision in decisions:
        payload = decision.model_dump(mode="json")
        collider_contract.append(
            {
                "decision_id": payload.get("decision_id"),
                "component_id": payload.get("component_id"),
                "body_root_path": payload.get("body_root_path"),
                "targets": payload.get("collider_target_ids")
                or payload.get("collider_paths")
                or payload.get("prim_paths")
                or [],
                "collision_mode": payload.get("collision_mode"),
                "collision_approximation": payload.get("collision_approximation"),
            }
        )
    encoded = json.dumps(
        {
            "source_digest": source_digest,
            "collider_contract": sorted(
                collider_contract,
                key=lambda item: str(item.get("decision_id") or ""),
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _round_float(value: float, digits: int = 6) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(float(value), digits)


def inspect_mesh_prims(usd_path: Path | str) -> list[PhysicsCandidate]:
    """Inspect raw mesh candidates through the V1 compatibility contract."""

    result = scene_ops.inspect_mesh_candidates(usd_path)
    raw_candidates = result.get("candidates")
    if not isinstance(raw_candidates, list):
        raise RuntimeError("Physics inspection returned no candidates list.")
    return [PhysicsCandidate.model_validate(candidate) for candidate in raw_candidates]


def inspect_physics_components(usd_path: Path | str) -> list[PhysicsComponent]:
    """Inspect logical physics components through the normative V2 contract."""

    result = scene_ops.inspect_components(usd_path)
    raw_components = result.get("components")
    if not isinstance(raw_components, list):
        raise RuntimeError("Physics inspection returned no components list.")
    return [PhysicsComponent.model_validate(item) for item in raw_components]


def _decision_id(prim_path: str) -> str:
    return (
        prim_path.strip("/").replace("/", "__").replace(":", "_").replace(".", "_")
        or "root"
    )


def infer_physics_decisions(
    candidates: list[PhysicsCandidate],
    *,
    collision_approximation: str = "convexHull",
) -> list[PhysicsDecision]:
    """Infer baseline physics decisions from inspected mesh candidates."""

    decisions: list[PhysicsDecision] = []
    for candidate in candidates:
        profile = infer_material_profile(
            candidate.prim_name,
            candidate.material_name,
            candidate.material_path,
        )
        mass = candidate.bbox_volume_m3 * profile.density * profile.volume_fraction
        mass = max(mass, 1e-6) if candidate.bbox_volume_m3 > 0 else 0.0
        properties = {
            "density": _round_float(profile.density, digits=3),
            "estimated_mass_kg": _round_float(mass, digits=6),
            "static_friction": _round_float(profile.static_friction, digits=3),
            "dynamic_friction": _round_float(profile.dynamic_friction, digits=3),
            "restitution": _round_float(profile.restitution, digits=3),
        }
        label_bits = [
            candidate.prim_name,
            profile.family,
            "collider",
        ]
        rationale = (
            f"{profile.rationale} Evidence: prim={candidate.prim_name!r}, "
            f"material={candidate.material_name or candidate.material_path or 'unbound'!r}, "
            f"bbox_volume_m3={candidate.bbox_volume_m3:.12g}."
        )
        decisions.append(
            PhysicsDecision(
                decision_id=_decision_id(candidate.prim_path),
                prim_paths=[candidate.prim_path],
                component_label=" ".join(label_bits),
                inferred_material_family=profile.family,
                inferred_material_name=candidate.material_name,
                collision_approximation=collision_approximation,
                physical_properties=properties,
                confidence=0.72 if profile.family != "generic" else 0.45,
                rationale=rationale,
            )
        )
    return decisions


def infer_component_decisions(
    components: list[PhysicsComponent],
    *,
    collision_approximation: str = "convexHull",
) -> list[PhysicsComponentDecision]:
    """Infer baseline authoring decisions once per logical component."""

    decisions: list[PhysicsComponentDecision] = []
    for component in components:
        material = component.material_evidence[0] if component.material_evidence else {}
        evidence_path = (
            component.visual_evidence_paths[0]
            if component.visual_evidence_paths
            else component.body_root_path
        )
        prim_name = evidence_path.rsplit("/", 1)[-1]
        material_name = material.get("material_name")
        material_path = material.get("material_path")
        profile = infer_material_profile(prim_name, material_name, material_path)
        volume = float(component.bounds_m.get("volume_m3") or 0.0)
        if component.component_role == "unowned_static":
            density = 0.0
            mass = 0.0
        else:
            density = profile.density
            mass = volume * profile.density * profile.volume_fraction
            mass = max(mass, 1e-6) if volume > 0 else 0.0
        authoring_paths = component.collider_paths or component.visual_evidence_paths
        if not authoring_paths:
            raise RuntimeError(
                f"Physics component {component.component_id} has no collider targets "
                "or visual geometry suitable for explicit collider authoring."
            )
        collision_mode: Literal["preserve_existing", "author_on_targets"] = (
            "preserve_existing" if component.collider_paths else "author_on_targets"
        )
        findings = ", ".join(component.topology_findings) or "none"
        decisions.append(
            PhysicsComponentDecision(
                decision_id=component.component_id,
                component_id=component.component_id,
                component_role=component.component_role,
                body_root_path=component.body_root_path,
                visual_evidence_paths=component.visual_evidence_paths,
                collider_paths=authoring_paths,
                collision_mode=collision_mode,
                mass_authoring_path=component.body_root_path,
                inferred_material_family=profile.family,
                inferred_material_name=material_name,
                collision_approximation=collision_approximation,
                physical_properties={
                    "density": _round_float(density, digits=3),
                    "estimated_mass_kg": _round_float(mass, digits=6),
                    "static_friction": _round_float(profile.static_friction, digits=3),
                    "dynamic_friction": _round_float(
                        profile.dynamic_friction, digits=3
                    ),
                    "restitution": _round_float(profile.restitution, digits=3),
                },
                confidence=0.72 if profile.family != "generic" else 0.45,
                rationale=(
                    f"{profile.rationale} Component evidence: visual_paths="
                    f"{len(component.visual_evidence_paths)}, existing_colliders="
                    f"{len(component.collider_paths)}, bounds_volume_m3={volume:.12g}, "
                    f"component_role={component.component_role}, "
                    f"topology_findings={findings}."
                ),
            )
        )
    return decisions


def _write_predictions_jsonl(
    path: Path,
    decisions: list[PhysicsDecision] | list[PhysicsComponentDecision],
    *,
    within: Path | None = None,
) -> Path:
    lines: list[str] = []
    for decision in decisions:
        physical_properties = dict(decision.physical_properties)
        estimated_mass = physical_properties.get("estimated_mass_kg")
        prim_paths = (
            decision.collider_paths
            if isinstance(decision, PhysicsComponentDecision)
            else decision.prim_paths
        )
        if estimated_mass is not None and len(prim_paths) > 1:
            physical_properties["estimated_mass_kg"] = estimated_mass / len(prim_paths)
        for prim_path in prim_paths:
            classification: dict[str, Any] = {
                "component": (
                    decision.component_id
                    if isinstance(decision, PhysicsComponentDecision)
                    else decision.component_label
                ),
                "material": decision.inferred_material_family,
                "physical_properties": physical_properties,
                "collision_approximation": decision.collision_approximation,
                "confidence": decision.confidence,
                "reasoning": decision.rationale,
            }
            if isinstance(decision, PhysicsComponentDecision):
                classification.update(
                    {
                        "decision_id": decision.decision_id,
                        "component_id": decision.component_id,
                        "collision_mode": decision.collision_mode,
                        "mass_authoring_path": decision.mass_authoring_path,
                        "component_estimated_mass_kg": estimated_mass,
                    }
                )
                if decision.rigid_body_grouping:
                    classification["rigid_body_grouping"] = decision.rigid_body_grouping
                if decision.convex_decomposition is not None:
                    classification["convex_decomposition"] = decision.convex_decomposition.model_dump(mode="json")
                if decision.quality_warnings:
                    classification["quality_warnings"] = deepcopy(
                        decision.quality_warnings
                    )
            record = {
                "id": prim_path,
                "classification": classification,
                "source": "content_agent_workflows.physics.inspect_components",
            }
            if (
                isinstance(decision, PhysicsComponentDecision)
                and decision.quality_warnings
            ):
                record["quality_warnings"] = deepcopy(decision.quality_warnings)
            lines.append(
                json.dumps(_json_safe(record), allow_nan=False, sort_keys=True)
            )
    return atomic_write_text(
        path,
        "".join(f"{line}\n" for line in lines),
        within=within,
    )


def load_physics_decision_patch(
    path: Path | str,
    *,
    within: Path | None = None,
) -> (
    list[PhysicsDecision]
    | list[PhysicsComponentDecision]
    | list[PhysicsComponentTargetDecision]
):
    """Load accepted physics decisions from an agent-authored decision patch."""

    patch_path, payload = _read_json_object(path, within=within)
    return parse_physics_decision_patch(payload, label=str(patch_path))


def parse_physics_decision_patch(
    payload: dict[str, Any],
    *,
    label: str = "payload",
) -> (
    list[PhysicsDecision]
    | list[PhysicsComponentDecision]
    | list[PhysicsComponentTargetDecision]
):
    """Parse an already-read physics patch without reopening its path."""

    schema_version = payload.get("schema_version")
    if schema_version not in {
        PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        LEGACY_PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
    }:
        raise RuntimeError(
            f"Unsupported physics decision patch schema_version {schema_version!r}."
        )
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise RuntimeError(f"Physics decision patch has no decisions: {label}")
    if schema_version == PHYSICS_DECISION_PATCH_SCHEMA_VERSION:
        uses_target_ids = _validate_v2_decision_representation(raw_decisions)
        if uses_target_ids:
            return [
                PhysicsComponentTargetDecision.model_validate(item)
                for item in raw_decisions
            ]
        return [PhysicsComponentDecision.model_validate(item) for item in raw_decisions]
    return [PhysicsDecision.model_validate(item) for item in raw_decisions]


def _validate_v2_decision_representation(raw_decisions: list[Any]) -> bool:
    """Require one canonical target representation across a V2 patch."""

    uses_target_ids = [
        isinstance(item, dict) and "collider_target_ids" in item
        for item in raw_decisions
    ]
    if any(uses_target_ids) and not all(uses_target_ids):
        raise RuntimeError(
            "Physics V2 decision patch cannot mix target-ID and resolved-path "
            "decisions."
        )
    return bool(uses_target_ids and all(uses_target_ids))


def _load_decision_patch_payload(
    path: Path | str,
    *,
    within: Path | None = None,
) -> dict[str, Any]:
    _patch_path, payload = _read_json_object(path, within=within)
    return payload


def _v2_decisions_from_payload(
    payload: dict[str, Any],
) -> list[PhysicsComponentDecision]:
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise RuntimeError("Physics V2 decision patch must include a decisions list.")
    return [PhysicsComponentDecision.model_validate(item) for item in raw_decisions]


def _unresolved_components_from_payload(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    unresolved = payload.get("unresolved_components") or []
    if not isinstance(unresolved, list):
        raise RuntimeError("Physics V2 unresolved_components must be a list.")
    items: list[dict[str, Any]] = []
    for index, item in enumerate(unresolved):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"Physics V2 unresolved component at index {index} must be an object."
            )
        items.append(dict(item))
    return items


def _component_authoring_targets(
    component: PhysicsComponent,
) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for role, paths in (
        ("visual", component.visual_evidence_paths),
        ("collider", component.collider_paths),
    ):
        for prim_path in sorted(set(paths)):
            path_digest = hashlib.sha256(prim_path.encode("utf-8")).hexdigest()
            targets.append(
                {
                    "target_id": f"{component.component_id}:{role}:{path_digest}",
                    "role": role,
                    "prim_path": prim_path,
                }
            )
    return targets


_PHYSICS_COMPONENT_CATALOG_ADDITIVE_FIELDS = frozenset(
    {"authoring_targets", "source_path_expansions"}
)


def parse_physics_component_catalog_entry(
    raw_component: Any,
) -> PhysicsComponent:
    """Parse one component while allowing only documented catalog enrichments."""

    if not isinstance(raw_component, dict):
        raise RuntimeError("Physics component inspection entry must be an object.")
    model_fields = set(PhysicsComponent.model_fields)
    unsupported_fields = sorted(
        set(raw_component) - model_fields - _PHYSICS_COMPONENT_CATALOG_ADDITIVE_FIELDS
    )
    if unsupported_fields:
        raise RuntimeError(
            "Physics component inspection contains unsupported fields: "
            f"{unsupported_fields}."
        )
    return PhysicsComponent.model_validate(
        {key: value for key, value in raw_component.items() if key in model_fields}
    )


def add_physics_component_target_catalog(
    inspection_payload: dict[str, Any],
    *,
    source_path_expansions: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Attach stable IDs that let an agent select targets without copying paths."""

    result = deepcopy(inspection_payload)
    schema_version = result.get("schema_version")
    if schema_version not in {
        None,
        "content-agent-workflows.physics-components.v2",
        "usd-cli.physics-components.v2",
    }:
        raise RuntimeError(
            "Unsupported physics component inspection schema_version "
            f"{schema_version!r}."
        )
    normalized_expansions: dict[str, list[str]] = {}
    for path, source_paths in (source_path_expansions or {}).items():
        if (
            not isinstance(path, str)
            or not isinstance(source_paths, list)
            or not all(isinstance(source_path, str) for source_path in source_paths)
        ):
            raise RuntimeError("Physics source_path_expansions has an invalid shape.")
        normalized_expansions[path] = list(dict.fromkeys(source_paths))
    if normalized_expansions:
        result["source_path_expansions"] = normalized_expansions
    raw_components = result.get("components")
    if not isinstance(raw_components, list):
        return result
    for raw_component in raw_components:
        component = parse_physics_component_catalog_entry(raw_component)
        raw_component["authoring_targets"] = _component_authoring_targets(component)
        component_paths = {
            component.body_root_path,
            *component.visual_evidence_paths,
            *component.collider_paths,
        }
        component_expansions = {
            path: normalized_expansions[path]
            for path in sorted(component_paths)
            if path in normalized_expansions
        }
        if component_expansions:
            raw_component["source_path_expansions"] = component_expansions
    return result


def _validate_raw_v2_component_coverage(
    payload: dict[str, Any],
    components: list[PhysicsComponent],
) -> None:
    """Report component-ID coverage before target binding or model validation."""

    raw_decisions = payload.get("decisions")
    raw_unresolved = payload.get("unresolved_components") or []
    if not isinstance(raw_decisions, list) or not isinstance(raw_unresolved, list):
        return
    decision_ids: list[str] = []
    unresolved_ids: list[str] = []
    for label, records, destination in (
        ("decision", raw_decisions, decision_ids),
        ("unresolved component", raw_unresolved, unresolved_ids),
    ):
        for index, item in enumerate(records):
            component_id = item.get("component_id") if isinstance(item, dict) else None
            if not isinstance(component_id, str) or not component_id.strip():
                raise RuntimeError(
                    f"Physics V2 {label} at index {index} requires a non-empty "
                    "component_id for coverage validation."
                )
            destination.append(component_id)
    supplied_ids = [*decision_ids, *unresolved_ids]
    expected_ids = {component.component_id for component in components}
    missing = sorted(expected_ids - set(supplied_ids))
    unknown = sorted(set(supplied_ids) - expected_ids)
    if missing or unknown:
        raise RuntimeError(
            "Physics V2 decision coverage mismatch: "
            f"missing={missing or 'none'}, unknown={unknown or 'none'}."
        )


def _bind_patch_targets_in_place(
    payload: dict[str, Any],
    components: list[PhysicsComponent],
) -> None:
    """Resolve stable target IDs against authoritative inspected components."""

    components_by_id = {component.component_id: component for component in components}
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        return
    for raw_decision in raw_decisions:
        if not isinstance(raw_decision, dict):
            continue
        component_id = raw_decision.get("component_id")
        if not isinstance(component_id, str):
            continue
        component = components_by_id.get(component_id)
        if component is None:
            continue

        target_ids = raw_decision.pop("collider_target_ids", None)
        if target_ids is not None:
            if not isinstance(target_ids, list) or not target_ids:
                raise RuntimeError(
                    f"Physics decision for {component_id} must select at least one "
                    "collider_target_id."
                )
            if not all(isinstance(target_id, str) for target_id in target_ids):
                raise RuntimeError(
                    f"Physics decision for {component_id} contains a non-string "
                    "collider_target_id."
                )
            catalog = {
                target["target_id"]: target
                for target in _component_authoring_targets(component)
            }
            unknown_ids = sorted(set(target_ids) - set(catalog))
            if unknown_ids:
                raise RuntimeError(
                    f"Physics decision for {component_id} contains unknown collider "
                    f"target IDs: {unknown_ids}."
                )
            expected_role = (
                "collider"
                if raw_decision.get("collision_mode") == "preserve_existing"
                else "visual"
            )
            wrong_role_ids = sorted(
                target_id
                for target_id in set(target_ids)
                if catalog[target_id]["role"] != expected_role
            )
            if wrong_role_ids:
                raise RuntimeError(
                    f"Physics decision for {component_id} selected target IDs with "
                    f"the wrong role for {raw_decision.get('collision_mode')}: "
                    f"{wrong_role_ids}."
                )
            raw_decision["collider_paths"] = [
                catalog[target_id]["prim_path"]
                for target_id in dict.fromkeys(target_ids)
            ]
            raw_decision["visual_evidence_paths"] = list(
                component.visual_evidence_paths
            )
            raw_decision["body_root_path"] = component.body_root_path
            raw_decision["mass_authoring_path"] = component.body_root_path
        # Mobility comes from the authoritative inspection, never a child hint.
        supplied_role = raw_decision.get("component_role")
        if supplied_role is not None and supplied_role != component.component_role:
            raise RuntimeError(f"Physics decision for {component_id} changes component_role.")
        raw_decision["component_role"] = component.component_role


def _validate_component_decisions(
    components: list[PhysicsComponent],
    decisions: list[PhysicsComponentDecision],
    unresolved_components: list[dict[str, Any]],
) -> None:
    components_by_id = {component.component_id: component for component in components}
    decision_ids = [decision.component_id for decision in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise RuntimeError(
            "Physics V2 decision patch covers a component more than once."
        )
    unresolved_ids: list[str] = []
    for item in unresolved_components:
        component_id = item.get("component_id")
        reason = str(item.get("reason") or "").strip()
        if not isinstance(component_id, str) or not reason:
            raise RuntimeError(
                "Each unresolved physics component requires component_id and reason."
            )
        unresolved_ids.append(component_id)
    all_ids = [*decision_ids, *unresolved_ids]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("A physics component is covered more than once.")
    missing = sorted(set(components_by_id) - set(all_ids))
    extra = sorted(set(all_ids) - set(components_by_id))
    if missing or extra:
        raise RuntimeError(
            "Physics V2 decision coverage mismatch: "
            f"missing={missing or 'none'}, unknown={extra or 'none'}."
        )
    for decision in decisions:
        component = components_by_id[decision.component_id]
        if decision.component_role != component.component_role:
            raise RuntimeError(f"Physics decision {decision.decision_id} changes component_role.")
        if component.component_role == "unowned_static" and decision.mass_properties is not None:
            raise RuntimeError("Static components cannot request rigid-body mass properties.")
        targets = set(decision.collider_paths)
        helpers = set(component.helper_paths)
        if targets & helpers:
            raise RuntimeError(
                f"Physics decision {decision.decision_id} targets helper geometry."
            )
        if decision.collision_mode == "author_on_targets" and component.collider_paths:
            raise RuntimeError(
                f"Physics decision {decision.decision_id} must preserve existing "
                "colliders instead of authoring new collider targets."
            )
        allowed = (
            set(component.collider_paths)
            if decision.collision_mode == "preserve_existing"
            else set(component.visual_evidence_paths)
        )
        if not targets <= allowed:
            known_component_paths = {
                component.body_root_path,
                *component.visual_evidence_paths,
                *component.collider_paths,
                *component.helper_paths,
                *component.rigid_body_paths,
                *component.joint_paths,
            }
            unknown_paths = sorted(targets - known_component_paths)
            if unknown_paths:
                raise RuntimeError(
                    f"Physics decision {decision.decision_id} contains collider "
                    "paths that are not present in the inspected component: "
                    f"{unknown_paths}. Select targets by collider_target_id from "
                    "physics_components.json."
                )
            wrong_role_paths = sorted(targets - allowed)
            raise RuntimeError(
                f"Physics decision {decision.decision_id} contains collider paths "
                f"with a role incompatible with {decision.collision_mode}: "
                f"{wrong_role_paths}."
            )
        if decision.body_root_path != component.body_root_path:
            raise RuntimeError(
                f"Physics decision {decision.decision_id} changes body_root_path."
            )
        if decision.mass_authoring_path != component.body_root_path:
            raise RuntimeError(
                f"Physics decision {decision.decision_id} must author mass on the "
                "inspected component body root."
            )


def _snap_patch_body_roots_in_place(
    payload: dict[str, Any],
    components: list[PhysicsComponent],
) -> None:
    """Snap each decision's ``body_root_path`` / ``mass_authoring_path`` to the
    inspected component's ``body_root_path`` when the agent nominated a path
    *inside* the component (e.g. the visual mesh ``/World/Object`` instead of the
    body root ``/World``). The inspected topology is authoritative — a
    physics-property decision must not relocate the rigid body. Mutates ``payload``
    in place so the written apply patch (re-validated by the schema-apply step)
    carries the corrected paths too. A no-op for already-correct decisions; a
    genuinely foreign path is left untouched so the validator still rejects it.
    """
    roots = {
        component.component_id: component.body_root_path for component in components
    }
    for decision in payload.get("decisions", []):
        if not isinstance(decision, dict):
            continue
        root = roots.get(decision.get("component_id"))
        if not root:
            continue
        prefix = root.rstrip("/") + "/"
        for key in ("body_root_path", "mass_authoring_path"):
            value = decision.get(key)
            if isinstance(value, str) and value != root and value.startswith(prefix):
                decision[key] = root


def _validate_v2_patch_payload_against_components(
    payload: dict[str, Any],
    *,
    components: list[PhysicsComponent],
    source_digest: str,
) -> tuple[list[PhysicsComponentDecision], list[dict[str, Any]]]:
    expected_digest = payload.get("source_digest")
    if expected_digest != source_digest:
        raise RuntimeError(
            "Physics V2 decision patch source_digest does not match the inspected "
            "asset."
        )
    raw_decisions = payload.get("decisions")
    if isinstance(raw_decisions, list):
        _validate_v2_decision_representation(raw_decisions)
    _validate_raw_v2_component_coverage(payload, components)
    _bind_patch_targets_in_place(payload, components)
    _snap_patch_body_roots_in_place(payload, components)
    decisions = _v2_decisions_from_payload(payload)
    unresolved_components = _unresolved_components_from_payload(payload)
    _validate_component_decisions(components, decisions, unresolved_components)
    payload["decisions"] = [_as_json(decision) for decision in decisions]
    return decisions, unresolved_components


def resolve_physics_v2_patch_targets(
    payload: dict[str, Any],
    *,
    components: list[PhysicsComponent],
    source_digest: str,
) -> tuple[dict[str, Any], list[PhysicsComponentDecision], list[dict[str, Any]]]:
    """Resolve target-ID decisions against one authoritative component catalog."""

    if payload.get("schema_version") != PHYSICS_DECISION_PATCH_SCHEMA_VERSION:
        raise RuntimeError(
            "Only Physics V2 decision patches support target resolution."
        )
    resolved_payload = deepcopy(payload)
    decisions, unresolved_components = _validate_v2_patch_payload_against_components(
        resolved_payload,
        components=components,
        source_digest=source_digest,
    )
    return resolved_payload, decisions, unresolved_components


def _allowed_decision_targets(
    component: PhysicsComponent,
    decision: PhysicsComponentDecision,
) -> set[str]:
    if decision.collision_mode == "preserve_existing":
        return set(component.collider_paths)
    return set(component.visual_evidence_paths)


def _safe_heuristic_rebase_match(
    component: PhysicsComponent,
    decision: PhysicsComponentDecision,
) -> bool:
    if component.component_role != "unowned_static":
        return True
    if decision.body_root_path != component.body_root_path:
        return False
    if decision.collision_mode != "preserve_existing":
        return False
    density = float(decision.physical_properties.get("density") or 0.0)
    mass = float(decision.physical_properties.get("estimated_mass_kg") or 0.0)
    return density <= 0.0 and mass <= 0.0


def _single_decision_value(
    decisions: list[PhysicsComponentDecision],
    field_name: str,
) -> Any:
    values = {getattr(decision, field_name) for decision in decisions}
    if len(values) != 1:
        raise RuntimeError(
            "Topology repair coalesced physics components with incompatible "
            f"{field_name} values. Re-author the V2 decision patch against the "
            "prepared topology derivative."
        )
    return next(iter(values))


def _merge_physical_properties(
    decisions: list[PhysicsComponentDecision],
) -> dict[str, float]:
    property_keys = {
        key for decision in decisions for key in decision.physical_properties
    }
    merged: dict[str, float] = {}
    for key in sorted(property_keys):
        values = [
            float(decision.physical_properties.get(key, 0.0)) for decision in decisions
        ]
        if key == "estimated_mass_kg":
            merged[key] = sum(values)
            continue
        first_value = values[0]
        if any(abs(value - first_value) > 1e-9 for value in values[1:]):
            raise RuntimeError(
                "Topology repair coalesced physics components with incompatible "
                f"physical property {key!r}. Re-author the V2 decision patch "
                "against the prepared topology derivative."
            )
        merged[key] = first_value
    return merged


def _merge_quality_warnings(
    decisions: list[PhysicsComponentDecision],
) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for decision in decisions:
        for warning in decision.quality_warnings:
            normalized = _json_safe(warning)
            key = json.dumps(normalized, allow_nan=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
    return merged


def _merge_rigid_body_grouping(
    decisions: list[PhysicsComponentDecision],
) -> str | None:
    values = sorted(
        {
            decision.rigid_body_grouping.strip()
            for decision in decisions
            if decision.rigid_body_grouping and decision.rigid_body_grouping.strip()
        }
    )
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return "Merged pre-topology grouping notes: " + " | ".join(values)


def _merge_rebased_component_decisions(
    component: PhysicsComponent,
    decisions: list[PhysicsComponentDecision],
) -> PhysicsComponentDecision:
    if not decisions:
        raise RuntimeError(
            f"Prepared physics component {component.component_id} has no matching "
            "pre-topology decision. Re-author the V2 decision patch against the "
            "prepared topology derivative."
        )
    ordered = sorted(decisions, key=lambda decision: decision.decision_id)
    base = ordered[0]
    if any(decision.mass_properties is not None for decision in ordered) and (
        len(ordered) != 1 or base.body_root_path != component.body_root_path
    ):
        raise RuntimeError(
            "Topology repair changes the body frame or mass grouping for explicit "
            "mass_properties. Re-author them against the prepared derivative."
        )
    collider_paths = sorted(
        {path for decision in ordered for path in decision.collider_paths}
    )
    rationale = base.rationale
    if len(ordered) > 1:
        rationale = (
            "Merged compatible pre-topology decisions after topology repair: "
            + ", ".join(decision.decision_id for decision in ordered)
            + "."
        )
    return base.model_copy(
        update={
            "decision_id": component.component_id,
            "component_id": component.component_id,
            "component_role": component.component_role,
            "body_root_path": component.body_root_path,
            "visual_evidence_paths": component.visual_evidence_paths,
            "collider_paths": collider_paths,
            "collision_mode": _single_decision_value(ordered, "collision_mode"),
            "mass_authoring_path": component.body_root_path,
            "inferred_material_family": _single_decision_value(
                ordered, "inferred_material_family"
            ),
            "inferred_material_name": _single_decision_value(
                ordered, "inferred_material_name"
            ),
            "collision_approximation": _single_decision_value(
                ordered, "collision_approximation"
            ),
            "convex_decomposition": _single_decision_value(
                ordered, "convex_decomposition"
            ),
            "physical_properties": _merge_physical_properties(ordered),
            "rigid_body_grouping": _merge_rigid_body_grouping(ordered),
            "quality_warnings": _merge_quality_warnings(ordered),
            "confidence": min(decision.confidence for decision in ordered),
            "rationale": rationale,
        }
    )


def rebase_physics_v2_patch_to_components(
    payload: dict[str, Any],
    *,
    components: list[PhysicsComponent],
    source_digest: str,
    asset: Path,
) -> tuple[dict[str, Any], list[PhysicsComponentDecision], list[dict[str, Any]]]:
    """Rewrite a pre-topology V2 patch to a prepared topology derivative."""

    rebased_payload = dict(payload)
    rebased_payload["asset"] = str(asset)
    rebased_payload["source_digest"] = source_digest
    decisions = _v2_decisions_from_payload(rebased_payload)
    unresolved_components = _unresolved_components_from_payload(rebased_payload)
    exact_error_message = ""
    try:
        _validate_component_decisions(components, decisions, unresolved_components)
    except RuntimeError as exact_error:
        original_decisions = decisions
        exact_error_message = str(exact_error)
    else:
        rebased_payload["decisions"] = [_as_json(decision) for decision in decisions]
        if unresolved_components:
            rebased_payload["unresolved_components"] = unresolved_components
        return rebased_payload, decisions, unresolved_components

    if unresolved_components:
        raise RuntimeError(
            "Topology repair changed component identity for a V2 decision patch "
            "with unresolved_components. Re-author the patch against the prepared "
            f"topology derivative. Original validation error: {exact_error_message}"
        )

    component_matches: dict[int, list[int]] = {
        component_index: [] for component_index in range(len(components))
    }
    for decision_index, decision in enumerate(original_decisions):
        candidate_components: list[int] = []
        for component_index, component in enumerate(components):
            targets = set(decision.collider_paths)
            allowed = _allowed_decision_targets(component, decision)
            if (
                targets
                and targets <= allowed
                and _safe_heuristic_rebase_match(component, decision)
            ):
                candidate_components.append(component_index)
        if len(candidate_components) != 1:
            raise RuntimeError(
                "Topology repair changed physics component identity. Re-author the "
                "V2 decision patch against the prepared topology derivative. "
                f"Original validation error: {exact_error_message}"
            )
        component_matches[candidate_components[0]].append(decision_index)

    rebased_decisions: list[PhysicsComponentDecision] = []
    for component_index, component in enumerate(components):
        rebased_decisions.append(
            _merge_rebased_component_decisions(
                component,
                [
                    original_decisions[index]
                    for index in component_matches[component_index]
                ],
            )
        )
    _validate_component_decisions(components, rebased_decisions, [])
    rebased_payload["decisions"] = [
        _as_json(decision) for decision in rebased_decisions
    ]
    rebased_payload.pop("unresolved_components", None)
    return rebased_payload, rebased_decisions, []


def load_physics_behavior_assessment(
    path: Path | str,
    *,
    within: Path | None = None,
) -> PhysicsBehaviorAssessment:
    """Load an agent-authored visual behavior assessment artifact.

    ``PhysicsBehaviorAssessment`` coerces the richer shapes the review LLM emits
    (see its ``_coerce_llm_shapes`` validator), so a plain ``model_validate`` is
    enough here.
    """

    _assessment_path, payload = _read_json_object(path, within=within)
    return PhysicsBehaviorAssessment.model_validate(payload)


def default_physics_behavior_assessment(
    *,
    runtime_report: Path | str | None,
    rendered_frames: list[str] | None = None,
    unresolved_issue: str,
) -> PhysicsBehaviorAssessment:
    """Build a conservative assessment when visual review is unavailable."""

    frames = rendered_frames or []
    return PhysicsBehaviorAssessment(
        status="unresolved_issues",
        checked_views=frames,
        runtime_report=str(runtime_report) if runtime_report is not None else None,
        rendered_frames=frames,
        issues_found=[unresolved_issue],
        unresolved_issues=[unresolved_issue],
        assessment_notes=unresolved_issue,
    )


def merge_physics_behavior_assessment(
    evidence: ValidationEvidence,
    assessment: PhysicsBehaviorAssessment,
    *,
    assessment_path: Path | str | None = None,
    render_receipt_path: Path | str | None = None,
) -> ValidationEvidence:
    """Merge agent-authored visual behavior review into runtime evidence."""

    incoming_status = evidence.sim_ready_status
    review_artifacts = (
        [
            EvidenceArtifact(
                kind="physics_behavior_assessment",
                path=str(assessment_path),
                description="Agent-authored visual review of rendered simulation frames.",
            )
        ]
        if assessment_path is not None
        else []
    )
    # checked_views may contain human-readable view summaries rather than file
    # paths. Only renderer-produced frame paths are valid evidence artifacts.
    for frame_path in assessment.rendered_frames:
        review_artifacts.append(
            EvidenceArtifact(
                kind="simulation_frame",
                path=str(frame_path),
                description="Rendered frame used for physics behavior review.",
            )
        )
    if assessment.runtime_report:
        review_artifacts.append(
            EvidenceArtifact(
                kind="runtime_report",
                path=assessment.runtime_report,
                description="Runtime metrics reviewed alongside rendered frames.",
            )
        )
    if render_receipt_path is not None:
        resolved_receipt = Path(render_receipt_path).resolve(strict=True)
        review_artifacts.append(
            EvidenceArtifact(
                kind="simulation_frame_receipt",
                path=str(resolved_receipt),
                description=(
                    "Digest-bound mapping from rendered frames to their source USD, "
                    "camera data, and OVRTX settings."
                ),
                metadata={"sha256": file_sha256(resolved_receipt)},
            )
        )

    unresolved = [str(item) for item in assessment.unresolved_issues]
    if assessment.status == "unresolved_issues" and not unresolved:
        unresolved = [
            "Physics behavior assessment reported unresolved issues without details."
        ]
    failures = list(evidence.failures)
    warnings = list(evidence.warnings)
    if unresolved:
        warnings.extend(unresolved)

    check_status: Literal["warning", "pass"] = "warning" if unresolved else "pass"
    visual_check = ValidationCheck(
        name="simulation_visual_review",
        status=check_status,
        summary="Rendered simulation behavior was reviewed by the agent.",
        evidence_artifacts=review_artifacts,
        warnings=unresolved,
        repair_hints=[
            "Refine body grouping, collider approximation, mass, friction, or restitution; rerun runtime validation and visual review."
        ]
        if unresolved
        else [],
        metadata={
            "assessment_status": assessment.status,
            "issues_found_count": len(assessment.issues_found),
            "issues_fixed_count": len(assessment.issues_fixed),
        },
    )

    checks = [check for check in evidence.checks if check.name != visual_check.name]
    checks.append(visual_check)
    evidence.checks = checks
    evidence.evidence_artifacts = [
        *evidence.evidence_artifacts,
        *review_artifacts,
    ]
    evidence.warnings = _dedupe_strings(warnings)
    evidence.unresolved_issues = _dedupe_strings(
        [*evidence.unresolved_issues, *unresolved]
    )
    if unresolved:
        evidence.repair_hints = _dedupe_strings(
            [
                *evidence.repair_hints,
                "Use the rendered simulation review to target the next physics decision patch.",
            ]
        )

    if failures or evidence.sim_ready_status == "fail":
        evidence.sim_ready_status = "fail"
    elif unresolved:
        evidence.sim_ready_status = "conditional"
    elif incoming_status == "conditional" or evidence.warnings:
        evidence.sim_ready_status = "conditional"
    elif evidence.sim_ready_status == "not_evaluated":
        evidence.sim_ready_status = "conditional"
    else:
        evidence.sim_ready_status = "pass"
    return evidence


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _status_from_authored_report(
    report: dict[str, Any],
    *,
    mobility_intent: str = "preserve",
) -> tuple[str, list[str]]:
    failures: list[str] = []
    require_rigid_body = mobility_intent != "static"
    if require_rigid_body and int(report.get("rigid_body_count") or 0) < 1:
        failures.append("No UsdPhysics.RigidBodyAPI prims were authored.")
    if int(report.get("collision_count") or 0) < 1:
        failures.append("No UsdPhysics.CollisionAPI prims were authored.")
    if (
        not report.get("physics_scene_paths")
        and int(report.get("physics_scene_count") or 0) < 1
    ):
        failures.append("No UsdPhysics.Scene prim was authored.")
    return ("fail" if failures else "pass", failures)


def _runtime_acceptance_from_authored_report(
    report: dict[str, Any],
    *,
    mobility_intent: str = "preserve",
    drop_height_m: float | None = None,
) -> dict[str, Any] | None:
    if mobility_intent == "static":
        return None
    authored_body_count = _authored_rigid_body_count(report)
    if authored_body_count is None or authored_body_count <= 0:
        return None
    if authored_body_count > 1:
        return None
    acceptance: dict[str, Any] = {}
    if authored_body_count == 1:
        acceptance["expected_body_count"] = 1
    if drop_height_m is not None and float(drop_height_m) <= 0.0:
        acceptance["require_gravity_response"] = False
    return acceptance


def _authored_rigid_body_count(report: dict[str, Any]) -> int | None:
    body_count = report.get("enabled_rigid_body_count", report.get("rigid_body_count"))
    if body_count is not None:
        try:
            return int(body_count)
        except (TypeError, ValueError):
            return None
    body_paths = report.get("rigid_body_paths")
    if isinstance(body_paths, list):
        return len(body_paths)
    return None


MAX_MULTI_BODY_RUNTIME_BODIES = 12


def _multi_body_runtime_skip_message(body_count: int) -> str:
    return (
        "Per-body runtime validation is capped at "
        f"{MAX_MULTI_BODY_RUNTIME_BODIES} enabled rigid bodies; this asset "
        f"has {body_count}, so whole-asset runtime validation is marked not "
        "evaluated."
    )


def _multi_body_runtime_paths_unavailable_message() -> str:
    return (
        "Per-body runtime validation requires the enabled rigid-body prim "
        "paths, and they could not be determined from the authored asset, so "
        "whole-asset runtime validation is marked not evaluated."
    )


def _enabled_rigid_body_paths_for_runtime(
    report: dict[str, Any],
    physics_usd: Path,
) -> list[str]:
    """Return the enabled rigid-body prim paths for per-body validation.

    Prefers the authored report (the pxr inspection path records the paths
    directly); the usd-cli structural report only carries counts, so fall
    back to one in-process inspection of the authored USD. Returns an empty
    list when the paths cannot be determined.
    """

    def _valid_paths(value: Any) -> list[str]:
        if not isinstance(value, list) or not value:
            return []
        paths = [item for item in value if isinstance(item, str)]
        if len(paths) != len(value) or any(not path.startswith("/") for path in paths):
            return []
        return paths

    body_paths = _valid_paths(report.get("enabled_rigid_body_paths"))
    if body_paths:
        return body_paths
    try:
        inspection = scene_ops.inspect_authored_physics(physics_usd)
    except Exception:
        return []
    return _valid_paths(inspection.get("enabled_rigid_body_paths"))


def _common_prim_ancestor(paths: Iterable[str]) -> str | None:
    """Return the deepest prim path that prefixes every given prim path."""

    segment_lists = [
        [segment for segment in path.split("/") if segment] for path in paths
    ]
    if not segment_lists:
        return None
    common: list[str] = []
    for segments in zip(*segment_lists):
        if any(segment != segments[0] for segment in segments):
            break
        common.append(segments[0])
    if not common:
        return None
    return "/" + "/".join(common)


def physics_decision_assignment_payload(
    decision: PhysicsComponentDecision,
    *,
    path_space: str,
    source_path_expansions: dict[str, list[str]],
) -> dict[str, Any]:
    """Add deterministic runtime/source path provenance to one decision."""

    if path_space not in {"source", "inspection"}:
        raise ValueError("Physics assignment path_space must be source or inspection.")
    if not isinstance(source_path_expansions, dict) or any(
        not isinstance(path, str)
        or not isinstance(source_paths, list)
        or not all(isinstance(source_path, str) for source_path in source_paths)
        for path, source_paths in source_path_expansions.items()
    ):
        raise ValueError("Physics assignment source_path_expansions is invalid.")
    payload = _as_json(decision)
    runtime_paths = _dedupe_strings(
        [
            decision.body_root_path,
            decision.mass_authoring_path,
            *decision.visual_evidence_paths,
            *decision.collider_paths,
        ]
    )
    expansions: dict[str, list[str]] = {}
    source_paths: list[str] = []
    for runtime_path in runtime_paths:
        expanded = source_path_expansions.get(runtime_path)
        if expanded is None:
            expanded = [runtime_path] if path_space == "source" else []
        expansions[runtime_path] = _dedupe_strings(expanded)
        source_paths.extend(expansions[runtime_path])
    payload.update(
        {
            "path_space": path_space,
            "runtime_prim_paths": runtime_paths,
            "source_prim_paths": _dedupe_strings(source_paths),
            "source_path_expansions": expansions,
            "unmapped_runtime_prim_paths": [
                runtime_path
                for runtime_path in runtime_paths
                if not expansions[runtime_path]
            ],
        }
    )
    return payload


def _optional_file_sha256(path: Path) -> str | None:
    return file_sha256(path) if path.is_file() else None


def _inspect_workflow_components(
    params: PhysicsApplyWorkflowInput,
    usd_path: Path,
) -> tuple[list[PhysicsComponent], str]:
    component_response = scene_ops.inspect_components(
        usd_path,
        path_space=params.path_space,
    )
    components = [
        parse_physics_component_catalog_entry(component)
        for component in component_response.get("components") or []
    ]
    source_digest = str(component_response.get("source_digest") or "")
    if not components:
        raise RuntimeError(f"No physics components found in {usd_path}")
    return components, source_digest


def _runtime_result_to_evidence(
    *,
    result: dict[str, Any],
    physics_usd_path: Path,
    engine: PhysicsSimulationEngine,
    duration_s: float,
    sample_fps: int,
    physics_properties_status: Literal["pass", "fail"] = "pass",
    report_path_override: Path | None = None,
) -> tuple[ValidationEvidence, Path]:
    failures = [str(item) for item in result.get("failures") or []]
    warnings = [str(item) for item in result.get("warnings") or []]
    report_path_value = result.get("runtime_report")
    if not isinstance(report_path_value, str) or not report_path_value:
        raise RuntimeError("Runtime validation returned no report path.")
    report_path = report_path_override or Path(report_path_value)
    artifacts = [
        EvidenceArtifact(
            kind=str(artifact.get("kind") or "runtime_artifact"),
            path=str(artifact.get("path") or ""),
            description=str(artifact.get("description") or ""),
        )
        for artifact in result.get("evidence_artifacts") or []
        if isinstance(artifact, dict) and artifact.get("path")
    ]
    metadata = {
        "engine": engine,
        "duration_s": duration_s,
        "sample_fps": sample_fps,
        "settle_distance": result.get("settle_distance"),
        "trajectory_summary": result.get("summary"),
    }
    diagnostics = [str(item) for item in result.get("diagnostics") or []]
    if diagnostics:
        metadata["diagnostics"] = diagnostics
    phase_timings = result.get("phase_timings_seconds")
    if isinstance(phase_timings, dict):
        metadata["phase_timings_seconds"] = deepcopy(phase_timings)
    scene_info = result.get("scene_info")
    if isinstance(scene_info, dict):
        support_decision = scene_info.get("ground_clearance_support_decision")
        if isinstance(support_decision, dict):
            metadata["ground_clearance_support_decision"] = deepcopy(support_decision)
    runtime_asset_provenance = result.get("runtime_asset_provenance")
    if isinstance(runtime_asset_provenance, dict):
        metadata["runtime_asset_provenance"] = deepcopy(runtime_asset_provenance)
    evidence = physics_validation_evidence(
        asset=str(physics_usd_path),
        target_runtime=engine,
        physics_properties_status=physics_properties_status,
        runtime_loadability_status="fail" if failures else "pass",
        no_explosions_status="fail" if failures else "pass",
        validation_tier="T2_simulation_match",
        evidence_artifacts=artifacts,
        failures=failures,
        warnings=warnings,
        metadata=metadata,
    )
    return evidence, report_path


def validate_physics_runtime(
    *,
    physics_usd: Path | str,
    output_dir: Path | str,
    engine: PhysicsSimulationEngine = "ovphysx",
    duration_s: float = 3.0,
    dt: float = 1.0 / 240.0,
    sample_fps: int = 30,
    drop_height_m: float | None = None,
    placement_mode: Literal["drop", "mounted"] = "drop",
    acceptance: dict[str, Any] | None = None,
    physics_properties_status: Literal["pass", "fail"] = "pass",
    usd_cli_session: WorkflowUsdCliSession | None = None,
    scene_tool_timeout_seconds: float = 1800.0,
    ground_clearance_support_cache: dict[str, dict[str, Any]] | None = None,
    ground_clearance_support_cache_key: str | None = None,
    approved_dependency_roots: Iterable[Path | str] | None = None,
) -> tuple[ValidationEvidence, Path | None]:
    """Run simulation-backed validation and return evidence plus report path.

    ``approved_dependency_roots`` bounds which filesystem dependencies the
    portable export may follow; defaults to ``scene_ops.validate_runtime``'s
    own default (the physics USD's parent directory) when omitted. Callers
    validating a USD whose texture sidecar sits outside that immediate parent
    (e.g. a tuning broker's materialized candidate, whose ``<name>_assets``
    sidecar is a sibling of the candidate's own ``candidates/`` directory, not
    a child of it) must pass the wider root explicitly or the export raises
    "dependency is outside approved roots" for a legitimate reference.
    """

    physics_usd_path = Path(physics_usd).resolve()
    validation_dir = Path(output_dir).resolve()
    validation_dir.mkdir(parents=True, exist_ok=True)

    if engine == "none":
        evidence = physics_validation_evidence(
            asset=str(physics_usd_path),
            target_runtime="none",
            physics_properties_status="not_evaluated",
            runtime_loadability_status="not_evaluated",
            no_explosions_status="not_evaluated",
            unresolved_issues=["Runtime simulation was disabled."],
        )
        return evidence, None

    try:
        result = scene_ops.validate_runtime(
            physics_usd=physics_usd_path,
            output_dir=validation_dir,
            engine=engine,
            duration_s=duration_s,
            dt=dt,
            sample_fps=sample_fps,
            drop_height_m=drop_height_m,
            placement_mode=placement_mode,
            acceptance=acceptance,
            usd_cli_session=usd_cli_session,
            scene_tool_timeout_seconds=scene_tool_timeout_seconds,
            ground_clearance_support_cache=ground_clearance_support_cache,
            ground_clearance_support_cache_key=(ground_clearance_support_cache_key),
            approved_dependency_roots=approved_dependency_roots,
        )
        return _runtime_result_to_evidence(
            result=result,
            physics_usd_path=physics_usd_path,
            engine=engine,
            duration_s=duration_s,
            sample_fps=sample_fps,
            physics_properties_status=physics_properties_status,
        )
    except Exception as exc:
        report = {
            "engine": engine,
            "physics_usd": str(physics_usd_path),
            "error": str(exc),
        }
        report_path = _write_json(
            validation_dir / "runtime_validation_report.json",
            report,
            within=validation_dir,
        )
        evidence = physics_validation_evidence(
            asset=str(physics_usd_path),
            target_runtime=engine,
            physics_properties_status=physics_properties_status,
            runtime_loadability_status="fail",
            no_explosions_status="not_evaluated",
            validation_tier="T2_simulation_match",
            evidence_artifacts=[
                EvidenceArtifact(
                    kind="runtime_report",
                    path=str(report_path),
                    description="Runtime validation failure report.",
                )
            ],
            failures=[str(exc)],
            unresolved_issues=[
                "Runtime validation failed before producing a usable trajectory."
            ],
            metadata={"engine": engine},
        )
        return evidence, report_path


def _xformable_placement_root(
    physics_usd_path: Path, body_prim_paths: list[str]
) -> str | None:
    """Deepest common body ancestor resolved to the nearest Xformable prim.

    USD silently ignores xform ops on prims that are not UsdGeomXformable
    (Scope or typeless grouping prims), which would discard the drop
    placement while every downstream gate assumes it happened. Walk up from
    the lexical common ancestor to the nearest Xformable prim — an ancestor
    still contains every body, so relative poses and joint frames are
    preserved. Returns None (fail closed) when no Xformable ancestor exists.
    """
    from pxr import Usd, UsdGeom

    common = _common_prim_ancestor(body_prim_paths)
    if common is None:
        return None
    try:
        stage = Usd.Stage.Open(str(physics_usd_path))
    except Exception:  # noqa: BLE001 — unreadable stage fails closed below
        return None
    if stage is None:
        return None
    path = common
    while path and path != "/":
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and prim.IsA(UsdGeom.Xformable):
            return path
        path = path.rsplit("/", 1)[0] or "/"
    return None


def _kinematic_body_paths(
    physics_usd_path: Path, body_prim_paths: list[str]
) -> set[str]:
    """Bodies authored with physics:kinematicEnabled=true.

    A kinematic body correctly does NOT fall under gravity (SimReady
    authoring deliberately anchors bodies this way), so the drop gates must
    not require a gravity response from it. Unreadable stages return the
    empty set — the body then faces the ordinary gates, which is the
    pre-existing behavior.
    """
    from pxr import Usd, UsdPhysics

    kinematic: set[str] = set()
    try:
        stage = Usd.Stage.Open(str(physics_usd_path))
    except Exception:  # noqa: BLE001 — fall back to the ordinary gates
        return kinematic
    if stage is None:
        return kinematic
    for body_path in body_prim_paths:
        prim = stage.GetPrimAtPath(body_path)
        if not prim or not prim.IsValid():
            continue
        api = UsdPhysics.RigidBodyAPI(prim)
        attr = api.GetKinematicEnabledAttr() if api else None
        if attr and attr.Get():
            kinematic.add(body_path)
    return kinematic


def _author_multi_body_recording(
    per_body_results: list[dict[str, Any]],
) -> str | None:
    """One recording that animates EVERY tracked body.

    A per-body recording time-samples only its own tracked body; the other
    bodies sit frozen at the elevated drop pose, which reads to the visual
    reviewer as "parts separating" / "no visible motion under gravity".
    Build an aggregate layer beside the first body's recording that
    sublayers it and adds each remaining body's pose samples from that
    body's own trajectory (every run simulates the complete scene under the
    same placement, so the timelines align). Returns None when any piece is
    missing or unreadable — the review gate then sees no recording instead
    of frames that misrepresent the drop.
    """
    from pxr import Gf, Usd, UsdGeom

    base = next((item for item in per_body_results if item.get("recording_usda")), None)
    if base is None:
        return None
    base_path = Path(str(base["recording_usda"]))
    if not base_path.is_file():
        return None
    aggregate_path = base_path.with_name("aggregate_recording.usda")
    try:
        aggregate_path.unlink(missing_ok=True)
        stage = Usd.Stage.CreateNew(str(aggregate_path))
        stage.GetRootLayer().subLayerPaths.append(base_path.name)
        end_code = float(stage.GetEndTimeCode())
        # Simulator poses are WORLD-space, but the samples are authored on
        # each body's LOCAL pose ops. With a translated placement ancestor
        # (the multi-body drop root) the composition would re-apply that
        # ancestor transform on top of the world pose, so every authored
        # sample is converted through the body's parent-to-world inverse.
        # The base body's own sublayered samples carry the same defect, so
        # its samples are re-authored (the root layer's samples shadow the
        # sublayer's) rather than skipped.
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for item in per_body_results:
            trajectory_value = item.get("trajectory_jsonl")
            if not trajectory_value:
                return None
            rows = []
            for line in (
                Path(str(trajectory_value)).read_text(encoding="utf-8").splitlines()
            ):
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append([float(v) for v in row["pose"]])
            if not rows:
                return None
            prim = stage.GetPrimAtPath(str(item["body_prim_path"]))
            if not prim or not prim.IsValid():
                return None
            xf = UsdGeom.Xformable(prim)
            if not xf:
                return None
            # Mirror author_trajectory_usda's pose-op handling: reuse the
            # composed translate/orient ops, drop the FULL pose-op set from
            # the tail (per-axis translates and every rotate variant would
            # re-apply an authored local rotation on top of the simulated
            # orientation), and clear stale samples before writing.
            XformOp = UsdGeom.XformOp
            pose_op_types = {
                op_type
                for op_type in (
                    XformOp.TypeTranslate,
                    XformOp.TypeOrient,
                    XformOp.TypeTransform,
                    getattr(XformOp, "TypeTranslateX", None),
                    getattr(XformOp, "TypeTranslateY", None),
                    getattr(XformOp, "TypeTranslateZ", None),
                    XformOp.TypeRotateX,
                    XformOp.TypeRotateY,
                    XformOp.TypeRotateZ,
                    XformOp.TypeRotateXYZ,
                    XformOp.TypeRotateXZY,
                    XformOp.TypeRotateYXZ,
                    XformOp.TypeRotateYZX,
                    XformOp.TypeRotateZXY,
                    XformOp.TypeRotateZYX,
                )
                if op_type is not None
            }
            t_op = o_op = None
            preserved = []
            for op in xf.GetOrderedXformOps():
                if op.GetOpName() == "xformOp:translate" and t_op is None:
                    t_op = op
                elif op.GetOpName() == "xformOp:orient" and o_op is None:
                    o_op = op
                elif op.GetOpType() not in pose_op_types:
                    preserved.append(op)
            if t_op is None:
                t_op = xf.AddTranslateOp()
            if o_op is None:
                o_op = xf.AddOrientOp()
            xf.SetXformOpOrder([t_op, o_op, *preserved])
            for attr in (t_op.GetAttr(), o_op.GetAttr()):
                if attr.GetTimeSamples():
                    attr.Clear()
            parent_to_world = xform_cache.GetParentToWorldTransform(prim)
            inverse_parent = parent_to_world.GetInverse()
            for index, pose in enumerate(rows):
                px, py, pz, qx, qy, qz, qw = pose
                world = Gf.Matrix4d()
                world.SetTransform(
                    Gf.Rotation(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz))),
                    Gf.Vec3d(px, py, pz),
                )
                local = world * inverse_parent
                local_translation = local.ExtractTranslation()
                local_orientation = local.ExtractRotationQuat()
                code = Usd.TimeCode(float(index))
                t_op.Set(local_translation, time=code)
                o_op.Set(
                    Gf.Quatf(
                        local_orientation.GetReal(),
                        Gf.Vec3f(*local_orientation.GetImaginary()),
                    ),
                    time=code,
                )
            end_code = max(end_code, float(len(rows) - 1))
        stage.SetStartTimeCode(0.0)
        stage.SetEndTimeCode(end_code)
        stage.GetRootLayer().Save()
    except Exception:  # noqa: BLE001 — a partial aggregate must not present as evidence
        aggregate_path.unlink(missing_ok=True)
        return None
    return str(aggregate_path)


def _all_numbers_finite(value: Any) -> bool:
    """True when every numeric leaf in the structure is finite."""
    if isinstance(value, bool):
        return True
    if isinstance(value, int | float):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    return True


def _append_aggregate_trajectory_rows(
    rows: list[dict[str, Any]],
    trajectory_jsonl: Any,
    body_prim_path: str,
) -> str | None:
    """Copy one body's trajectory rows into the aggregate, tagging each row."""

    if not isinstance(trajectory_jsonl, str) or not trajectory_jsonl:
        return f"{body_prim_path}: runtime run produced no trajectory artifact."
    trajectory_path = Path(trajectory_jsonl)
    if not trajectory_path.is_file():
        return f"{body_prim_path}: runtime trajectory artifact is missing."
    try:
        lines = trajectory_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return f"{body_prim_path}: runtime trajectory artifact is unreadable."
    appended = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return f"{body_prim_path}: runtime trajectory artifact is invalid."
        if not isinstance(row, dict):
            return f"{body_prim_path}: runtime trajectory artifact is invalid."
        # Consumers expect numeric t/pose/vel samples; a NaN or infinity
        # would be nulled by the JSON-safe write and silently corrupt the
        # aggregate, so fail the body instead.
        if not _all_numbers_finite(row):
            return f"{body_prim_path}: runtime trajectory contains non-finite values."
        row["body_prim_path"] = body_prim_path
        rows.append(row)
        appended += 1
    if appended == 0:
        # An empty/blank-only artifact is no evidence: a body must not pass
        # with zero sampled rows in the aggregate.
        return f"{body_prim_path}: runtime trajectory artifact is empty."
    return None


def validate_physics_runtime_multi_body(
    *,
    physics_usd: Path | str,
    output_dir: Path | str,
    body_prim_paths: list[str],
    engine: PhysicsSimulationEngine = "ovphysx",
    duration_s: float = 3.0,
    dt: float = 1.0 / 240.0,
    sample_fps: int = 30,
    drop_height_m: float | None = None,
    placement_mode: Literal["drop", "mounted"] = "drop",
    acceptance: dict[str, Any] | None = None,
    physics_properties_status: Literal["pass", "fail"] = "pass",
    usd_cli_session: WorkflowUsdCliSession | None = None,
    scene_tool_timeout_seconds: float = 1800.0,
    ground_clearance_support_cache: dict[str, dict[str, Any]] | None = None,
    ground_clearance_support_cache_key: str | None = None,
) -> tuple[ValidationEvidence, Path | None]:
    """Validate each enabled rigid body with one whole-scene simulation.

    The available engines record one tracked trajectory per simulation (the
    ovphysx daemon and the version-checked remote executor both emit a
    single-body trajectory stream), so per-body coverage runs the existing
    whole-scene drop-settle validation once per enabled rigid body. Every run
    simulates the complete scene; ``body_prim_path_hint`` selects the tracked
    body and the deepest common ancestor of all enabled bodies is translated
    as one placement root, preserving relative body poses and joint frames.
    The per-body acceptance checks reuse the single-body gates; the asset
    passes only when every body passes, and per-body failures are listed in
    the aggregate runtime report.

    Explicit mounted placement leaves every authored transform and anchor
    unchanged, so disconnected roots do not require a common placement root.
    """

    if not body_prim_paths:
        raise ValueError("body_prim_paths must name at least one rigid body")
    physics_usd_path = Path(physics_usd).resolve()
    validation_dir = Path(output_dir).resolve()
    validation_dir.mkdir(parents=True, exist_ok=True)

    # Same disabled-run contract as validate_physics_runtime: a caller that
    # reaches this public helper with the engine disabled must get fail-safe
    # not_evaluated evidence, not a fabricated per-body runtime failure.
    if engine == "none":
        evidence = physics_validation_evidence(
            asset=str(physics_usd_path),
            target_runtime="none",
            physics_properties_status="not_evaluated",
            runtime_loadability_status="not_evaluated",
            no_explosions_status="not_evaluated",
            unresolved_issues=["Runtime simulation was disabled."],
        )
        report_path = _write_json(
            validation_dir / "runtime_validation_report.json",
            {
                "engine": engine,
                "physics_usd": str(physics_usd_path),
                "mode": "multi_body",
                "not_evaluated": True,
                "skip_reason": "engine_none",
                "body_prim_paths": list(body_prim_paths),
                "warnings": ["Runtime simulation was disabled."],
            },
            within=validation_dir,
        )
        return evidence, report_path

    placement_prim_path = _xformable_placement_root(physics_usd_path, body_prim_paths)
    if placement_prim_path is None and placement_mode == "drop":
        # No shared Xformable placement root: translating each tracked body
        # alone would break the relative poses and joint frames this
        # validation promises to preserve, and USD silently ignores xform
        # ops on a non-Xformable root — so fail safe instead of simulating
        # a scene whose drop placement never happened.
        evidence = physics_validation_evidence(
            asset=str(physics_usd_path),
            target_runtime=engine,
            physics_properties_status="not_evaluated",
            runtime_loadability_status="not_evaluated",
            no_explosions_status="not_evaluated",
            unresolved_issues=[
                "Enabled rigid bodies share no common Xformable ancestor "
                "prim; per-body drop placement cannot preserve relative "
                "body poses and joint frames, so runtime validation was "
                "skipped."
            ],
        )
        # A skip must still leave a runtime report in the receipt: returning
        # no path would silently drop the runtime artifact from evidence for
        # a supported (multi-root) asset shape.
        report_path = _write_json(
            validation_dir / "runtime_validation_report.json",
            {
                "engine": engine,
                "physics_usd": str(physics_usd_path),
                "mode": "multi_body",
                "not_evaluated": True,
                "skip_reason": "no_common_xformable_ancestor",
                "body_prim_paths": list(body_prim_paths),
                "warnings": [
                    "Enabled rigid bodies share no common Xformable "
                    "ancestor prim; runtime validation was skipped."
                ],
            },
            within=validation_dir,
        )
        return evidence, report_path

    per_body_results: list[dict[str, Any]] = []
    failures: list[str] = []
    warnings: list[str] = []
    diagnostics: list[str] = []
    aggregate_rows: list[dict[str, Any]] = []
    evidence_artifacts: list[EvidenceArtifact] = []
    settle_distances: list[float] = []
    phase_timings_seconds: dict[str, float] = {}
    support_decisions: list[dict[str, Any]] = []

    kinematic_paths = _kinematic_body_paths(physics_usd_path, body_prim_paths)
    for index, body_prim_path in enumerate(body_prim_paths):
        body_dir = validation_dir / f"body_{index:03d}"
        per_body_acceptance = dict(acceptance or {})
        if body_prim_path in kinematic_paths:
            # An anchored/kinematic body correctly does not fall; requiring
            # a gravity response would deterministically fail a valid asset.
            per_body_acceptance["require_gravity_response"] = False
        try:
            result = scene_ops.validate_runtime(
                physics_usd=physics_usd_path,
                output_dir=body_dir,
                engine=engine,
                duration_s=duration_s,
                dt=dt,
                sample_fps=sample_fps,
                drop_height_m=drop_height_m,
                placement_mode=placement_mode,
                acceptance=per_body_acceptance,
                body_prim_path_hint=body_prim_path,
                placement_prim_path_hint=placement_prim_path,
                usd_cli_session=usd_cli_session,
                scene_tool_timeout_seconds=scene_tool_timeout_seconds,
                ground_clearance_support_cache=ground_clearance_support_cache,
                ground_clearance_support_cache_key=(ground_clearance_support_cache_key),
            )
        except Exception as exc:
            failures.append(f"{body_prim_path}: {exc}")
            per_body_results.append(
                {
                    "body_prim_path": body_prim_path,
                    "status": "fail",
                    "error": str(exc),
                    "failures": [str(exc)],
                    "warnings": [],
                }
            )
            continue
        body_failures = [str(item) for item in result.get("failures") or []]
        body_warnings = [str(item) for item in result.get("warnings") or []]
        body_diagnostics = [str(item) for item in result.get("diagnostics") or []]
        failures.extend(f"{body_prim_path}: {item}" for item in body_failures)
        warnings.extend(f"{body_prim_path}: {item}" for item in body_warnings)
        diagnostics.extend(f"{body_prim_path}: {item}" for item in body_diagnostics)
        trajectory_error = _append_aggregate_trajectory_rows(
            aggregate_rows,
            result.get("trajectory_jsonl"),
            body_prim_path,
        )
        if trajectory_error is not None:
            # A missing/unreadable trajectory fails the body itself, so
            # per_body_results and metadata.per_body_statuses cannot report a
            # pass that contradicts the aggregate failure.
            body_failures.append(trajectory_error)
            if trajectory_error not in failures:
                failures.append(trajectory_error)
        settle_distance = result.get("settle_distance")
        if isinstance(settle_distance, int | float) and math.isfinite(
            float(settle_distance)
        ):
            settle_distances.append(float(settle_distance))
        result_phase_timings = result.get("phase_timings_seconds")
        if isinstance(result_phase_timings, dict):
            for phase, raw_duration in result_phase_timings.items():
                if (
                    isinstance(raw_duration, int | float)
                    and not isinstance(raw_duration, bool)
                    and math.isfinite(float(raw_duration))
                    and float(raw_duration) >= 0.0
                ):
                    phase_timings_seconds[str(phase)] = phase_timings_seconds.get(
                        str(phase), 0.0
                    ) + float(raw_duration)
        result_scene_info = result.get("scene_info")
        support_decision = (
            result_scene_info.get("ground_clearance_support_decision")
            if isinstance(result_scene_info, dict)
            else None
        )
        if isinstance(support_decision, dict):
            support_decisions.append(deepcopy(support_decision))
        per_body_results.append(
            {
                "body_prim_path": body_prim_path,
                "status": "fail" if body_failures else "pass",
                "kinematic": body_prim_path in kinematic_paths,
                "runtime_report": result.get("runtime_report"),
                "scene_usd": result.get("scene_usd"),
                "trajectory_jsonl": result.get("trajectory_jsonl"),
                "recording_usda": result.get("recording_usda"),
                "settle_distance": settle_distance,
                "summary": result.get("summary"),
                "acceptance": result.get("acceptance"),
                "failures": body_failures,
                "warnings": body_warnings,
                "diagnostics": body_diagnostics,
                "ground_clearance_support_decision": deepcopy(support_decision)
                if isinstance(support_decision, dict)
                else None,
                "phase_timings_seconds": deepcopy(result_phase_timings)
                if isinstance(result_phase_timings, dict)
                else {},
            }
        )
        for artifact in result.get("evidence_artifacts") or []:
            if not isinstance(artifact, dict) or not artifact.get("path"):
                continue
            evidence_artifacts.append(
                EvidenceArtifact(
                    kind=str(artifact.get("kind") or "runtime_artifact"),
                    path=str(artifact["path"]),
                    description=(
                        f"{body_prim_path}: "
                        f"{artifact.get('description') or 'Runtime artifact.'}"
                    ),
                )
            )

    aggregate_trajectory_path = validation_dir / "trajectory.jsonl"
    # Same JSON discipline as _write_json: simulated rows can carry
    # non-finite floats, and bare NaN/Infinity tokens break strict parsers.
    atomic_write_text(
        aggregate_trajectory_path,
        "".join(
            json.dumps(_json_safe(row), allow_nan=False) + "\n"
            for row in aggregate_rows
        ),
        within=validation_dir,
    )
    # Surface a renderable recording at the top level: the visual-review
    # frame renderer reads report["recording_usda"], and without it the
    # review silently proceeds with zero rendered frames — a pass with no
    # OVRTX render provenance. The aggregate recording animates EVERY
    # tracked body (a single body's recording would show the others frozen
    # mid-air); when it cannot be authored, no recording ships rather than
    # frames that misrepresent the drop.
    aggregate_recording = _author_multi_body_recording(per_body_results)
    if aggregate_recording is None and per_body_results:
        warnings.append(
            "Aggregate multi-body recording unavailable; visual review has "
            "no whole-asset rollout."
        )
    report = {
        "engine": engine,
        "physics_usd": str(physics_usd_path),
        # Bind the report to the exact validated bytes, matching the digest
        # discipline of the other evidence artifacts in this module.
        "physics_usd_sha256": _optional_file_sha256(physics_usd_path),
        "recording_usda": aggregate_recording,
        "mode": "multi_body",
        "enabled_rigid_body_count": len(body_prim_paths),
        "placement_mode": placement_mode,
        "body_prim_paths": list(body_prim_paths),
        "placement_prim_path": placement_prim_path,
        "trajectory_jsonl": str(aggregate_trajectory_path),
        "settle_distance": max(settle_distances) if settle_distances else None,
        "per_body_results": per_body_results,
        "failures": failures,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "acceptance": dict(acceptance or {}),
        "ground_clearance_support_decisions": support_decisions,
        "phase_timings_seconds": phase_timings_seconds,
    }
    report_path = _write_json(
        validation_dir / "runtime_validation_report.json",
        report,
        within=validation_dir,
    )
    evidence_artifacts = [
        EvidenceArtifact(
            kind="runtime_report",
            path=str(report_path),
            description="Aggregate per-body runtime validation report.",
        ),
        EvidenceArtifact(
            kind="trajectory_jsonl",
            path=str(aggregate_trajectory_path),
            description="Aggregate per-body simulated pose and velocity.",
        ),
        *evidence_artifacts,
    ]
    evidence = physics_validation_evidence(
        asset=str(physics_usd_path),
        target_runtime=engine,
        physics_properties_status=physics_properties_status,
        runtime_loadability_status="fail" if failures else "pass",
        no_explosions_status="fail" if failures else "pass",
        validation_tier="T2_simulation_match",
        evidence_artifacts=evidence_artifacts,
        failures=failures,
        warnings=warnings,
        metadata={
            "engine": engine,
            "duration_s": duration_s,
            "sample_fps": sample_fps,
            "runtime_validation_mode": "multi_body",
            "enabled_rigid_body_count": len(body_prim_paths),
            "placement_prim_path": placement_prim_path,
            "settle_distance": report["settle_distance"],
            "per_body_statuses": {
                str(item["body_prim_path"]): str(item["status"])
                for item in per_body_results
            },
            "ground_clearance_support_decisions": support_decisions,
            "phase_timings_seconds": phase_timings_seconds,
        },
    )
    return evidence, report_path


def run_physics_apply_workflow(
    params: PhysicsApplyWorkflowInput,
) -> PhysicsApplyWorkflowResult:
    """Inspect, infer, author physics schema, and optionally simulate validate."""

    # Preserve and inspect the caller's lexical output paths before creating a
    # run directory or handing a destination to an external authoring backend.
    # A prior resolve() would turn a symlinked run directory into its target and
    # defeat the no-follow boundary below.
    output_dir_candidate = Path(os.path.abspath(params.output_dir.expanduser()))
    output_usd_value = params.output_usd_path or (output_dir_candidate / "physics.usdc")
    output_usd_candidate = Path(os.path.abspath(output_usd_value.expanduser()))
    for label, candidate in (
        ("Physics output directory", output_dir_candidate),
        ("Physics output USD", output_usd_candidate),
    ):
        try:
            resolved_candidate = candidate.resolve(strict=False)
        except OSError as exc:
            raise ValueError(
                f"{label} cannot be inspected safely: {candidate}"
            ) from exc
        if resolved_candidate != candidate:
            raise ValueError(
                f"{label} path must resolve without traversing symlinks: {candidate}"
            )

    source_path = Path(params.usd_path).expanduser().resolve()
    if output_usd_candidate == source_path or (
        output_usd_candidate.exists()
        and source_path.exists()
        and os.path.samefile(output_usd_candidate, source_path)
    ):
        if params.vomp_mass is not None:
            # The VoMP contract reports this as a failed result before any
            # inspection instead of raising, so callers retain the artifact.
            return PhysicsApplyWorkflowResult(
                success=False,
                asset=str(source_path),
                output_dir=str(output_dir_candidate),
                validation_status="fail",
                error=(
                    "VoMP canonical output USD must differ from the original input USD"
                ),
            )
        raise ValueError("Physics output USD must not overwrite the source USD.")
    output_dir = prepare_writable_directory(output_dir_candidate)
    output_usd = prepare_writable_file_path(output_usd_candidate)
    raw_dir = output_dir / "raw"
    source_usd = Path(params.source_usd_path or params.usd_path).resolve()
    request = params.model_dump(mode="json")
    request.pop("resume", None)
    request.pop("workflow_run_record_subdir", None)
    recorder = WorkflowRunRecorder.start(
        output_dir,
        workflow="physics_authoring",
        request=request,
        source_path=source_path if source_path.is_file() else None,
        backend={
            "scene_tool": "usd-cli",
            "transport": "usd-cli-tel",
            "simulation_engine": params.simulation_engine,
            "renderer": {
                "required": "ovrtx",
                "probe_timing": "before_scene_mutation",
            },
            "low_level_operations": ["physics.apply", "physics.validate", "save"],
            "workflow_owned_helpers": [
                "component_grouping",
                "topology_plan_guarded_apply",
                "authored_physics_summary",
                "runtime_acceptance_and_evidence",
            ],
        },
        policy={
            "collision_approximation": params.collision_approximation,
            "run_simulation": params.run_simulation,
            "fail_on_validation_error": params.fail_on_validation_error,
        },
        required_artifacts=[
            "assignments",
            "decision_patch",
            "components",
            "predictions",
            "apply_report",
            "validation_evidence",
        ],
        resume=params.resume,
        record_subdir=params.workflow_run_record_subdir,
    )
    # A destination may predate this attempt. Only attribute an output to a
    # failed run after an authoring backend has returned that artifact; merely
    # finding bytes at the requested path is not proof this attempt produced it.
    failure_physics_usd_path: Path | None = None

    def finish(result: PhysicsApplyWorkflowResult) -> PhysicsApplyWorkflowResult:
        artifact_fields = (
            ("assignments", result.assignments_path, "physics_assignments", True),
            ("decision_patch", result.decision_patch_path, "decision_patch", True),
            ("components", result.components_path, "physics_components", True),
            ("predictions", result.predictions_path, "predictions", True),
            ("apply_report", result.apply_report_path, "scene_operation_report", True),
            (
                "validation_evidence",
                result.validation_evidence_path,
                "validation",
                True,
            ),
            ("topology_report", result.topology_report_path, "topology_report", False),
            (
                "simulation_report",
                result.simulation_report_path,
                "simulation_report",
                False,
            ),
            (
                "scene_operation_record",
                result.scene_operation_record_path,
                "scene_operation_record",
                False,
            ),
            ("vomp_result", result.vomp_result_path, "vomp_result", False),
            (
                "vomp_provenance",
                result.vomp_provenance_path,
                "vomp_provenance",
                False,
            ),
            ("physics_usd", result.physics_usd_path, "usd", False),
        )
        recorded: list[str] = []
        for logical_name, path_text, kind, required in artifact_fields:
            if path_text is None:
                continue
            try:
                artifact = recorder.record_artifact(
                    logical_name,
                    Path(path_text),
                    kind=kind,
                    required=required,
                )
            except ValueError:
                if required:
                    raise
                continue
            recorded.append(artifact.logical_name)
        if recorded:
            recorder.checkpoint(
                "validated" if result.success else "failed",
                recorded,
            )
        manifest = recorder.finalize(
            "pass" if result.success else "fail",
            failure={
                "error": result.error,
                "validation_status": result.validation_status,
            }
            if not result.success
            else None,
        )
        updates: dict[str, Any] = {
            "workflow_run_manifest_path": str(recorder.path),
        }
        if result.success and manifest.status != "pass":
            updates.update(
                {
                    "success": False,
                    "validation_status": "fail",
                    "error": (
                        f"Physics workflow artifact contract failed: {manifest.failure}"
                    ),
                }
            )
        return result.model_copy(update=updates)

    try:
        source_asset_sha256 = _optional_file_sha256(source_usd)
        if params.source_asset_sha256 is not None:
            if source_asset_sha256 is None:
                raise RuntimeError(
                    f"Source USD recorded during physics preflight is missing: {source_usd}."
                )
            if params.source_asset_sha256 != source_asset_sha256:
                raise RuntimeError(
                    "Source USD changed after physics preflight: expected sha256 "
                    f"{params.source_asset_sha256}, observed {source_asset_sha256}."
                )

        working_usd = Path(params.usd_path).resolve()
        inspection_asset_sha256 = _optional_file_sha256(working_usd)
        if params.inspection_asset_sha256 is not None:
            if inspection_asset_sha256 is None:
                raise RuntimeError(
                    f"Inspection USD recorded during physics preflight is missing: "
                    f"{working_usd}."
                )
            if params.inspection_asset_sha256 != inspection_asset_sha256:
                raise RuntimeError(
                    "Inspection USD changed after physics preflight: expected sha256 "
                    f"{params.inspection_asset_sha256}, observed "
                    f"{inspection_asset_sha256}."
                )
        if params.vomp_mass is not None:
            _validate_vomp_output_suffix(output_usd)
            if output_usd == working_usd:
                raise RuntimeError(
                    "VoMP canonical output USD must differ from the original input USD"
                )
        supplied_patch_payload: dict[str, Any] | None = None
        prevalidated_patch_payload: dict[str, Any] | None = None
        supplied_v2_patch_prevalidated = False
        supplied_v2_patch_post_topology_prevalidated = False
        if params.decision_patch_path is not None:
            supplied_patch_payload = _load_decision_patch_payload(
                params.decision_patch_path,
                within=_contained_input_root(
                    params.decision_patch_path,
                    output_dir,
                ),
            )
            if (
                supplied_patch_payload.get("schema_version")
                == LEGACY_PHYSICS_DECISION_PATCH_SCHEMA_VERSION
            ):
                raise RuntimeError(
                    "Legacy V1 physics decision patches are not supported by the "
                    "canonical usd-cli scene backend. Regenerate a V2 component "
                    "decision patch; no scene mutation was attempted."
                )
        topology_report_path: Path | None = None
        mobility_intent = "preserve"
        if params.topology_plan_path is not None:
            if supplied_patch_payload is not None:
                if (
                    supplied_patch_payload.get("schema_version")
                    != PHYSICS_DECISION_PATCH_SCHEMA_VERSION
                ):
                    raise RuntimeError(
                        "Legacy physics decision patches cannot be combined with "
                        "topology plans; regenerate a V2 component decision patch."
                    )
                pre_topology_components, pre_topology_digest = (
                    _inspect_workflow_components(params, working_usd)
                )
                if supplied_patch_payload.get("source_digest") == pre_topology_digest:
                    prevalidated_patch_payload = deepcopy(supplied_patch_payload)
                    _validate_v2_patch_payload_against_components(
                        prevalidated_patch_payload,
                        components=pre_topology_components,
                        source_digest=pre_topology_digest,
                    )
                    supplied_v2_patch_prevalidated = True
                else:
                    prior_catalog_path = raw_dir / "physics_components.json"
                    prior_catalog = (
                        json.loads(prior_catalog_path.read_text(encoding="utf-8"))
                        if prior_catalog_path.is_file()
                        else None
                    )
                    prior_digest = (
                        prior_catalog.get("source_digest")
                        if isinstance(prior_catalog, dict)
                        else None
                    )
                    raw_prior_components = (
                        prior_catalog.get("components")
                        if isinstance(prior_catalog, dict)
                        else None
                    )
                    if supplied_patch_payload.get(
                        "source_digest"
                    ) != prior_digest or not isinstance(raw_prior_components, list):
                        raise RuntimeError(
                            "Physics V2 decision patch source_digest does not match "
                            "the inspected asset or prior prepared topology."
                        )
                    prior_components = [
                        parse_physics_component_catalog_entry(component)
                        for component in raw_prior_components
                    ]
                    prior_patch_payload = deepcopy(supplied_patch_payload)
                    _validate_v2_patch_payload_against_components(
                        prior_patch_payload,
                        components=prior_components,
                        source_digest=str(prior_digest),
                    )
                    supplied_v2_patch_post_topology_prevalidated = True
            _topology_plan_path, topology_plan = _read_json_object(
                params.topology_plan_path,
                within=_contained_input_root(
                    params.topology_plan_path,
                    output_dir,
                ),
            )
            mobility_intent = str(topology_plan.get("mobility_intent") or "preserve")
            topology_report = scene_ops.apply_topology_plan(
                input_usd_path=working_usd,
                output_usd_path=output_dir / "prepared.usda",
                expected_source_digest=str(
                    topology_plan.get("expected_source_digest") or ""
                ),
                mobility_intent=str(topology_plan.get("mobility_intent") or "preserve"),
                operations=list(topology_plan.get("operations") or []),
                invariants=dict(topology_plan.get("invariants") or {}),
                joint_endpoint_owner_promotions=list(
                    topology_plan.get("joint_endpoint_owner_promotions") or []
                ),
            )
            topology_report["scene_operation_backend"] = "workflow-helper"
            topology_report["backend_boundary"] = (
                "The workflow helper applies the already-approved topology plan; "
                "usd-cli remains the low-level scene tool."
            )
            working_usd = Path(str(topology_report["output_usd_path"])).resolve()
            topology_report_path = _write_json(
                raw_dir / "physics_topology_report.json",
                topology_report,
                within=output_dir,
            )

        components, source_digest = _inspect_workflow_components(params, working_usd)
        unresolved_components: list[dict[str, Any]] = []
        decision_patch_payload: dict[str, Any] | None = None
        if supplied_patch_payload is not None:
            decision_patch_payload = deepcopy(
                prevalidated_patch_payload or supplied_patch_payload
            )
            if decision_patch_payload.get("schema_version") == (
                PHYSICS_DECISION_PATCH_SCHEMA_VERSION
            ):
                if params.topology_plan_path is not None:
                    if supplied_v2_patch_prevalidated:
                        (
                            decision_patch_payload,
                            decisions,
                            unresolved_components,
                        ) = rebase_physics_v2_patch_to_components(
                            decision_patch_payload,
                            components=components,
                            source_digest=source_digest,
                            asset=working_usd,
                        )
                    elif supplied_v2_patch_post_topology_prevalidated:
                        decision_patch_payload["asset"] = str(working_usd)
                        decisions, unresolved_components = (
                            _validate_v2_patch_payload_against_components(
                                decision_patch_payload,
                                components=components,
                                source_digest=source_digest,
                            )
                        )
                    else:
                        raise RuntimeError(
                            "Physics V2 decision patch was not validated against "
                            "pre- or post-topology components."
                        )
                else:
                    decision_patch_payload["asset"] = str(working_usd)
                    _validate_v2_patch_payload_against_components(
                        decision_patch_payload,
                        components=components,
                        source_digest=source_digest,
                    )
                    decisions = _v2_decisions_from_payload(decision_patch_payload)
                    unresolved_components = _unresolved_components_from_payload(
                        decision_patch_payload
                    )
            else:
                decisions = load_physics_decision_patch(
                    params.decision_patch_path,
                    within=_contained_input_root(
                        params.decision_patch_path,
                        output_dir,
                    ),
                )
        else:
            decisions = infer_component_decisions(
                components,
                collision_approximation=params.collision_approximation,
            )
        if (
            decision_patch_payload is not None
            and decision_patch_payload.get("schema_version")
            == PHYSICS_DECISION_PATCH_SCHEMA_VERSION
        ):
            _validate_component_decisions(
                components,
                cast(list[PhysicsComponentDecision], decisions),
                unresolved_components,
            )
        elif decisions and isinstance(decisions[0], PhysicsComponentDecision):
            _validate_component_decisions(
                components,
                cast(list[PhysicsComponentDecision], decisions),
                unresolved_components,
            )
        vomp_decisions = _accepted_vomp_body_decisions(decisions, components)
        ground_clearance_support_cache = params.ground_clearance_support_cache
        ground_clearance_support_cache_key = _ground_clearance_support_context_key(
            source_digest=source_digest,
            decisions=cast(
                list[
                    PhysicsDecision
                    | PhysicsComponentDecision
                    | PhysicsComponentTargetDecision
                ],
                decisions,
            ),
        )

        candidate_prims_path = _write_json(
            raw_dir / "physics_components.json",
            add_physics_component_target_catalog(
                {
                    "schema_version": "content-agent-workflows.physics-components.v2",
                    "asset": str(working_usd),
                    "source_digest": source_digest,
                    "path_space": params.path_space,
                    "component_count": len(components),
                    "inspection_backend": "workflow-helper",
                    "components": [_as_json(component) for component in components],
                },
                source_path_expansions=params.source_path_expansions,
            ),
            within=output_dir,
        )
        if decision_patch_payload is not None:
            decision_patch_path = raw_dir / "physics_decision_patch.json"
            canonical_decision_patch_payload = (
                supplied_patch_payload
                if supplied_patch_payload is not None
                else decision_patch_payload
            )
            _write_json(
                decision_patch_path,
                canonical_decision_patch_payload,
                within=output_dir,
            )
            if decision_patch_payload != canonical_decision_patch_payload:
                apply_decision_patch_path = _write_json(
                    raw_dir / "physics_decision_patch_apply.json",
                    decision_patch_payload,
                    within=output_dir,
                )
            else:
                apply_decision_patch_path = decision_patch_path
        else:
            decision_patch_path = _write_json(
                raw_dir / "physics_decision_patch.json",
                {
                    "schema_version": PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
                    "asset": str(working_usd),
                    "source_digest": source_digest,
                    "decisions": [_as_json(decision) for decision in decisions],
                },
                within=output_dir,
            )
            apply_decision_patch_path = decision_patch_path
        predictions_path = _write_predictions_jsonl(
            raw_dir / "physics_predictions.jsonl",
            decisions,
            within=output_dir,
        )
        scene_operation_record_path: Path | None = None

        if not decisions:
            authored = working_usd
            authored, vomp_result, vomp_result_path = _apply_vomp_mass_if_requested(
                params=params,
                decisions=vomp_decisions,
                mobility_intent=mobility_intent,
                authored_usd=authored,
                output_usd=output_usd,
                output_dir=output_dir,
                raw_dir=raw_dir,
            )
            authored_report = scene_ops.inspect_authored_physics(authored)
            physics_failures: list[str] = []
            if (
                mobility_intent == "static"
                and _authored_rigid_body_count(authored_report) != 0
            ):
                physics_failures.append(
                    "Static mobility intent requires zero enabled rigid bodies."
                )
            authored_report.update(
                {
                    "operation": "physics.apply_schema",
                    "authoring_skipped": True,
                    "skip_reason": "No accepted physics decisions were provided.",
                    "scene_operation_backend": params.scene_backend,
                    "decision_patch_path": str(apply_decision_patch_path),
                    "predictions_jsonl": str(predictions_path),
                    "vomp_mass": vomp_result.model_dump(mode="json")
                    if vomp_result is not None
                    else None,
                }
            )
            apply_report_path = _write_json(
                raw_dir / "physics_apply_report.json",
                authored_report,
                within=output_dir,
            )
            unresolved_messages = [
                f"{item['component_id']}: {item['reason']}"
                for item in unresolved_components
            ] or ["No accepted physics decisions were provided."]
            validation_evidence = physics_validation_evidence(
                asset=str(authored),
                target_runtime=params.simulation_engine,
                physics_properties_status="fail"
                if physics_failures
                else "not_evaluated",
                runtime_loadability_status="not_evaluated",
                no_explosions_status="not_evaluated",
                evidence_artifacts=[
                    EvidenceArtifact(
                        kind="physics_decision_patch",
                        path=str(decision_patch_path),
                        description="Physics decision coverage patch.",
                    ),
                    EvidenceArtifact(
                        kind="physics_apply_report",
                        path=str(apply_report_path),
                        description="Physics authoring skip report.",
                    ),
                ],
                failures=physics_failures,
                warnings=unresolved_messages,
                unresolved_issues=unresolved_messages,
            )
            if validation_evidence.sim_ready_status != "fail":
                validation_evidence.sim_ready_status = "conditional"
            _bind_vomp_validation_metadata(
                validation_evidence,
                vomp_result,
                vomp_result_path,
            )
            _bind_validation_evidence_asset(validation_evidence, authored)
            validation_evidence_path = _write_json(
                output_dir / "validation_evidence.json",
                validation_evidence.model_dump(mode="json"),
                within=output_dir,
            )
            assignments_path = _write_json(
                output_dir / "physics_assignments.json",
                {
                    "schema_version": PHYSICS_ASSIGNMENTS_SCHEMA_VERSION,
                    "scene_backend": params.scene_backend,
                    "asset": str(source_usd),
                    "source_asset_sha256": source_asset_sha256,
                    "prepared_asset": str(working_usd),
                    "prepared_asset_sha256": _optional_file_sha256(working_usd),
                    "physics_usd": str(authored),
                    "path_space": params.path_space,
                    "source_path_expansions": params.source_path_expansions,
                    "candidate_count": len(components),
                    "component_count": len(components),
                    "decision_count": 0,
                    "decision_patch": str(decision_patch_path),
                    "apply_decision_patch": str(apply_decision_patch_path),
                    "decisions": [],
                    "unresolved_components": unresolved_components,
                    "mobility_intent": mobility_intent,
                    "vomp_mass": vomp_result.model_dump(mode="json")
                    if vomp_result is not None
                    else None,
                    "apply_report": authored_report,
                    "validation_evidence": str(validation_evidence_path),
                    "simulation_report": None,
                },
                within=output_dir,
            )

            return finish(
                PhysicsApplyWorkflowResult(
                    success=validation_evidence.sim_ready_status != "fail",
                    asset=str(source_usd),
                    output_dir=str(output_dir),
                    physics_usd_path=str(authored),
                    assignments_path=str(assignments_path),
                    decision_patch_path=str(decision_patch_path),
                    components_path=str(candidate_prims_path),
                    candidate_prims_path=str(candidate_prims_path),
                    predictions_path=str(predictions_path),
                    apply_report_path=str(apply_report_path),
                    topology_report_path=str(topology_report_path)
                    if topology_report_path
                    else None,
                    validation_evidence_path=str(validation_evidence_path),
                    simulation_report_path=None,
                    scene_operation_record_path=None,
                    vomp_result_path=str(vomp_result_path)
                    if vomp_result_path
                    else None,
                    vomp_provenance_path=vomp_result.provenance_path
                    if vomp_result is not None
                    else None,
                    validation_status=validation_evidence.sim_ready_status,
                )
            )

        if params.vomp_mass is not None:
            _validate_vomp_output_suffix(output_usd)
            schema_output_usd = raw_dir / f"physics_pre_vomp{output_usd.suffix.lower()}"
        else:
            schema_output_usd = output_usd
        if decisions and isinstance(decisions[0], PhysicsComponentDecision):
            physics_scene_path = scene_ops.select_physics_scene_path(working_usd)
            usd_cli_result = usd_cli_ops.apply_physics_patch(
                source_usd=working_usd,
                output_usd=schema_output_usd,
                raw_dir=raw_dir,
                workflow_decisions=[
                    _as_json(decision)
                    for decision in cast(list[PhysicsComponentDecision], decisions)
                ],
                author_rigid_body=mobility_intent != "static",
                physics_scene_path=physics_scene_path,
                usd_cli_session=params.usd_cli_session,
                timeout_seconds=params.scene_tool_timeout_seconds,
            )
            authored = Path(str(usd_cli_result["physics_usd"])).resolve()
            failure_physics_usd_path = authored
            authored_report = usd_cli_ops.authored_physics_report(usd_cli_result)
            authored_report.update(usd_cli_result)
            authored_report.update(
                {
                    "operation": "physics.apply_schema",
                    "scene_operation_backend": "usd-cli",
                    "collision_approximation": params.collision_approximation,
                    "author_rigid_body": mobility_intent != "static",
                    "decision_patch_path": str(apply_decision_patch_path),
                    "predictions_jsonl": str(predictions_path),
                }
            )
            scene_operation_record_path = Path(
                str(usd_cli_result["command_record_path"])
            ).resolve()
        else:
            raise RuntimeError(
                "Physics authoring requires a V2 component decision patch."
            )
        authored = Path(str(authored_report["physics_usd"])).resolve()
        failure_physics_usd_path = authored

        pre_vomp_authored = authored
        authored, vomp_result, vomp_result_path = _apply_vomp_mass_if_requested(
            params=params,
            decisions=vomp_decisions,
            mobility_intent=mobility_intent,
            authored_usd=authored,
            output_usd=output_usd,
            output_dir=output_dir,
            raw_dir=raw_dir,
        )
        failure_physics_usd_path = authored
        if vomp_result is not None:
            post_vomp_report = scene_ops.inspect_authored_physics(authored)
            authored_report = {
                **authored_report,
                **post_vomp_report,
            }
            authored_report.update(
                {
                    "pre_vomp_physics_usd": str(pre_vomp_authored),
                    "physics_usd": str(authored),
                    "vomp_mass": vomp_result.model_dump(mode="json"),
                }
            )
        physics_status, physics_failures = _status_from_authored_report(
            authored_report,
            mobility_intent=mobility_intent,
        )
        structural_validation = authored_report.get("structural_validation")
        if isinstance(structural_validation, dict) and not bool(
            structural_validation.get("ok")
        ):
            validation_data = structural_validation.get("data")
            raw_issues = (
                validation_data.get("issues")
                if isinstance(validation_data, dict)
                else structural_validation.get("issues")
            )
            issues = (
                [str(issue) for issue in raw_issues]
                if isinstance(raw_issues, list)
                else ["usd-cli structural physics validation failed."]
            )
            physics_failures.extend(issues)
            physics_status = "fail"
        authored_rigid_body_count = _authored_rigid_body_count(authored_report)
        if mobility_intent == "static" and authored_rigid_body_count != 0:
            physics_failures.append(
                "Static mobility intent requires zero enabled rigid bodies."
            )
            physics_status = "fail"
        apply_report_path = _write_json(
            raw_dir / "physics_apply_report.json",
            authored_report,
            within=output_dir,
        )

        validation_evidence: ValidationEvidence
        simulation_report_path: Path | None = None
        if params.run_simulation and mobility_intent == "static":
            simulation_report_path = _write_json(
                output_dir / "runtime" / "runtime_validation_report.json",
                {
                    "engine": params.simulation_engine,
                    "physics_usd": str(authored),
                    "mobility_intent": mobility_intent,
                    "not_evaluated": True,
                    "failures": physics_failures,
                    "warnings": [
                        "Runtime drop simulation was skipped for static mobility intent."
                    ],
                },
                within=output_dir,
            )
            validation_evidence = physics_validation_evidence(
                asset=str(authored),
                target_runtime=params.simulation_engine,
                physics_properties_status=cast(Any, physics_status),
                runtime_loadability_status="not_evaluated",
                no_explosions_status="not_evaluated",
                validation_tier="T2_simulation_match",
                evidence_artifacts=[
                    EvidenceArtifact(
                        kind="runtime_report",
                        path=str(simulation_report_path),
                        description="Static mobility runtime validation skip report.",
                    )
                ],
                failures=physics_failures,
                warnings=[
                    "Runtime drop simulation was skipped for static mobility intent."
                ],
            )
        elif (
            params.run_simulation
            and params.simulation_engine != "none"
            and authored_rigid_body_count is not None
            and authored_rigid_body_count > 1
        ):
            runtime_body_paths = _enabled_rigid_body_paths_for_runtime(
                authored_report,
                authored,
            )
            runtime_skip_message: str | None = None
            if not runtime_body_paths:
                runtime_skip_message = _multi_body_runtime_paths_unavailable_message()
            elif len(runtime_body_paths) > MAX_MULTI_BODY_RUNTIME_BODIES:
                runtime_skip_message = _multi_body_runtime_skip_message(
                    len(runtime_body_paths)
                )
            if runtime_skip_message is not None:
                simulation_report_path = _write_json(
                    output_dir / "runtime" / "runtime_validation_report.json",
                    {
                        "engine": params.simulation_engine,
                        "physics_usd": str(authored),
                        "enabled_rigid_body_count": authored_rigid_body_count,
                        "not_evaluated": True,
                        "failures": physics_failures,
                        "warnings": [runtime_skip_message],
                    },
                    within=output_dir,
                )
                validation_evidence = physics_validation_evidence(
                    asset=str(authored),
                    target_runtime=params.simulation_engine,
                    physics_properties_status=cast(Any, physics_status),
                    runtime_loadability_status="not_evaluated",
                    no_explosions_status="not_evaluated",
                    validation_tier="T2_simulation_match",
                    evidence_artifacts=[
                        EvidenceArtifact(
                            kind="runtime_report",
                            path=str(simulation_report_path),
                            description="Multi-body runtime validation skip report.",
                        )
                    ],
                    failures=physics_failures,
                    warnings=[runtime_skip_message],
                    unresolved_issues=[runtime_skip_message],
                    metadata={"enabled_rigid_body_count": authored_rigid_body_count},
                )
            else:
                multi_body_acceptance: dict[str, Any] = {}
                if (
                    params.drop_height_m is not None
                    and float(params.drop_height_m) <= 0.0
                ):
                    multi_body_acceptance["require_gravity_response"] = False
                if params.max_ground_penetration_m is not None:
                    multi_body_acceptance["max_ground_penetration_m"] = (
                        params.max_ground_penetration_m
                    )
                validation_evidence, simulation_report_path = (
                    validate_physics_runtime_multi_body(
                        physics_usd=authored,
                        output_dir=output_dir / "runtime",
                        body_prim_paths=runtime_body_paths,
                        engine=params.simulation_engine,
                        duration_s=params.simulation_duration_s,
                        dt=params.simulation_dt,
                        sample_fps=params.simulation_sample_fps,
                        drop_height_m=params.drop_height_m,
                        placement_mode=params.runtime_placement_mode,
                        acceptance=multi_body_acceptance,
                        physics_properties_status=cast(Any, physics_status),
                        usd_cli_session=params.usd_cli_session,
                        scene_tool_timeout_seconds=(params.scene_tool_timeout_seconds),
                        ground_clearance_support_cache=(ground_clearance_support_cache),
                        ground_clearance_support_cache_key=(
                            ground_clearance_support_cache_key
                        ),
                    )
                )
                if physics_status == "fail":
                    validation_evidence.failures.extend(physics_failures)
                    validation_evidence.sim_ready_status = "fail"
        elif params.run_simulation:
            acceptance = _runtime_acceptance_from_authored_report(
                authored_report,
                mobility_intent=mobility_intent,
                drop_height_m=params.drop_height_m,
            )
            # Applied after the authored-report acceptance so the configured
            # limit reaches the gate even on the paths that derive no other
            # acceptance terms; otherwise the runtime check silently falls back
            # to its default despite the explicitly selected workflow limit.
            if params.max_ground_penetration_m is not None:
                acceptance = dict(acceptance or {})
                acceptance["max_ground_penetration_m"] = params.max_ground_penetration_m
            validation_evidence, simulation_report_path = validate_physics_runtime(
                physics_usd=authored,
                output_dir=output_dir / "runtime",
                engine=params.simulation_engine,
                duration_s=params.simulation_duration_s,
                dt=params.simulation_dt,
                sample_fps=params.simulation_sample_fps,
                drop_height_m=params.drop_height_m,
                placement_mode=params.runtime_placement_mode,
                acceptance=acceptance,
                physics_properties_status=cast(Any, physics_status),
                usd_cli_session=params.usd_cli_session,
                scene_tool_timeout_seconds=params.scene_tool_timeout_seconds,
                ground_clearance_support_cache=ground_clearance_support_cache,
                ground_clearance_support_cache_key=(ground_clearance_support_cache_key),
            )
            if physics_status == "fail":
                validation_evidence.failures.extend(physics_failures)
                validation_evidence.sim_ready_status = "fail"
        else:
            validation_evidence = physics_validation_evidence(
                asset=str(authored),
                target_runtime=params.simulation_engine,
                physics_properties_status=cast(Any, physics_status),
                runtime_loadability_status="not_evaluated",
                no_explosions_status="not_evaluated",
                failures=physics_failures,
                unresolved_issues=["Runtime simulation was not requested."],
            )

        validation_evidence.metadata.update(
            {
                "scene_backend": params.scene_backend,
                "scene_operation_backend": authored_report.get(
                    "scene_operation_backend",
                    "workflow-helper",
                ),
                "runtime_execution_backend": (
                    "workflow-helper" if params.run_simulation else "not_requested"
                ),
            }
        )
        if unresolved_components:
            unresolved_messages = [
                f"{item['component_id']}: {item['reason']}"
                for item in unresolved_components
            ]
            validation_evidence.unresolved_issues.extend(unresolved_messages)
            validation_evidence.warnings.extend(unresolved_messages)
            if validation_evidence.sim_ready_status != "fail":
                validation_evidence.sim_ready_status = "conditional"

        _bind_vomp_validation_metadata(
            validation_evidence,
            vomp_result,
            vomp_result_path,
        )
        _bind_validation_evidence_asset(validation_evidence, authored)
        validation_evidence_path = _write_json(
            output_dir / "validation_evidence.json",
            validation_evidence.model_dump(mode="json"),
            within=output_dir,
        )

        assignments_path = _write_json(
            output_dir / "physics_assignments.json",
            {
                "schema_version": PHYSICS_ASSIGNMENTS_SCHEMA_VERSION,
                "scene_backend": params.scene_backend,
                "asset": str(source_usd),
                "source_asset_sha256": source_asset_sha256,
                "prepared_asset": str(working_usd),
                "prepared_asset_sha256": _optional_file_sha256(working_usd),
                "physics_usd": str(authored),
                "path_space": params.path_space,
                "source_path_expansions": params.source_path_expansions,
                "candidate_count": len(components),
                "component_count": len(components),
                "decision_count": len(decisions),
                "decision_patch": str(decision_patch_path),
                "apply_decision_patch": str(apply_decision_patch_path),
                "decisions": [
                    physics_decision_assignment_payload(
                        decision,
                        path_space=params.path_space,
                        source_path_expansions=params.source_path_expansions,
                    )
                    if isinstance(decision, PhysicsComponentDecision)
                    else _as_json(decision)
                    for decision in decisions
                ],
                "unresolved_components": unresolved_components,
                "mobility_intent": mobility_intent,
                "vomp_mass": vomp_result.model_dump(mode="json")
                if vomp_result is not None
                else None,
                "apply_report": authored_report,
                "validation_evidence": str(validation_evidence_path),
                "simulation_report": str(simulation_report_path)
                if simulation_report_path
                else None,
                "scene_operation_record": str(scene_operation_record_path)
                if scene_operation_record_path
                else None,
            },
            within=output_dir,
        )

        success = validation_evidence.sim_ready_status != "fail"
        error_message: str | None = None
        if params.fail_on_validation_error and not success:
            error_message = "Physics workflow validation failed: " + "; ".join(
                validation_evidence.failures or ["unknown validation failure"]
            )

        return finish(
            PhysicsApplyWorkflowResult(
                success=success,
                asset=str(source_usd),
                output_dir=str(output_dir),
                physics_usd_path=str(authored),
                assignments_path=str(assignments_path),
                decision_patch_path=str(decision_patch_path),
                components_path=str(candidate_prims_path),
                candidate_prims_path=str(candidate_prims_path),
                predictions_path=str(predictions_path),
                apply_report_path=str(apply_report_path),
                topology_report_path=str(topology_report_path)
                if topology_report_path
                else None,
                validation_evidence_path=str(validation_evidence_path),
                simulation_report_path=str(simulation_report_path)
                if simulation_report_path
                else None,
                scene_operation_record_path=str(scene_operation_record_path)
                if scene_operation_record_path
                else None,
                vomp_result_path=str(vomp_result_path) if vomp_result_path else None,
                vomp_provenance_path=vomp_result.provenance_path
                if vomp_result is not None
                else None,
                validation_status=validation_evidence.sim_ready_status,
                error=error_message,
            )
        )
    except Exception as exc:
        return finish(
            PhysicsApplyWorkflowResult(
                success=False,
                asset=str(source_usd),
                output_dir=str(output_dir),
                physics_usd_path=(
                    str(failure_physics_usd_path)
                    if failure_physics_usd_path is not None
                    else None
                ),
                validation_status="fail",
                error=str(exc),
            )
        )
