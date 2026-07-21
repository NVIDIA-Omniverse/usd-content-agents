# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic physics authoring workflow implementation."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from content_agent_workflows.common.validation_evidence import (
    EvidenceArtifact,
    ValidationCheck,
    ValidationEvidence,
    physics_validation_evidence,
)
from content_agent_workflows.physics.policy import infer_material_profile

from . import workbench_ops

PhysicsSimulationEngine = Literal["ovphysx", "fake", "none"]

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


class PhysicsComponentDecision(BaseModel):
    """One accepted V2 component-level physics authoring decision."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    body_root_path: str = Field(min_length=1)
    visual_evidence_paths: list[str] = Field(default_factory=list)
    collider_paths: list[str] = Field(min_length=1)
    collision_mode: Literal["preserve_existing", "author_on_targets"]
    mass_authoring_path: str = Field(min_length=1)
    inferred_material_family: str = Field(min_length=1)
    inferred_material_name: str | None = None
    collision_approximation: str = Field(min_length=1)
    physical_properties: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class PhysicsApplyWorkflowInput(BaseModel):
    """Input for the agentic physics apply workflow."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    usd_path: Path
    output_dir: Path
    output_usd_path: Path | None = None
    decision_patch_path: Path | None = None
    topology_plan_path: Path | None = None
    collision_approximation: str = "convexHull"
    run_simulation: bool = True
    simulation_engine: PhysicsSimulationEngine = "ovphysx"
    simulation_duration_s: float = 1.0
    simulation_dt: float = 1.0 / 240.0
    simulation_sample_fps: int = 30
    drop_height_m: float | None = None
    fail_on_validation_error: bool = False
    workbench_url: str | None = None
    workbench_session_id: str | None = None
    workbench_timeout_s: float = 300.0


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
    validation_status: str = "not_evaluated"
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


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


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


def _round_float(value: float, digits: int = 6) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(float(value), digits)


def inspect_mesh_prims(usd_path: Path | str) -> list[PhysicsCandidate]:
    """Inspect raw mesh candidates through the V1 compatibility contract."""

    result = workbench_ops.inspect_mesh_candidates(usd_path)
    raw_candidates = result.get("candidates")
    if not isinstance(raw_candidates, list):
        raise RuntimeError("Workbench physics inspection returned no candidates list.")
    return [PhysicsCandidate.model_validate(candidate) for candidate in raw_candidates]


def inspect_physics_components(usd_path: Path | str) -> list[PhysicsComponent]:
    """Inspect logical physics components through the normative V2 contract."""

    result = workbench_ops.inspect_components(usd_path)
    raw_components = result.get("components")
    if not isinstance(raw_components, list):
        raise RuntimeError("Workbench physics inspection returned no components list.")
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
    """Infer conservative physics decisions from inspected mesh candidates."""

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
    """Infer conservative authoring decisions once per logical component."""

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
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for decision in decisions:
            physical_properties = dict(decision.physical_properties)
            estimated_mass = physical_properties.get("estimated_mass_kg")
            prim_paths = (
                decision.collider_paths
                if isinstance(decision, PhysicsComponentDecision)
                else decision.prim_paths
            )
            if estimated_mass is not None and len(prim_paths) > 1:
                physical_properties["estimated_mass_kg"] = estimated_mass / len(
                    prim_paths
                )
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
                record = {
                    "id": prim_path,
                    "classification": classification,
                    "source": "content_agent_workflows.physics.inspect_components",
                }
                f.write(
                    json.dumps(_json_safe(record), allow_nan=False, sort_keys=True)
                    + "\n"
                )
    return path


def load_physics_decision_patch(
    path: Path | str,
) -> list[PhysicsDecision] | list[PhysicsComponentDecision]:
    """Load accepted physics decisions from an agent-authored decision patch."""

    patch_path = Path(path).resolve()
    payload = json.loads(patch_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Physics decision patch must be a JSON object: {patch_path}"
        )
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
        raise RuntimeError(f"Physics decision patch has no decisions: {patch_path}")
    if schema_version == PHYSICS_DECISION_PATCH_SCHEMA_VERSION:
        return [PhysicsComponentDecision.model_validate(item) for item in raw_decisions]
    return [PhysicsDecision.model_validate(item) for item in raw_decisions]


def _load_decision_patch_payload(path: Path | str) -> dict[str, Any]:
    patch_path = Path(path).resolve()
    payload = json.loads(patch_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Physics decision patch must be a JSON object: {patch_path}"
        )
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
            raise RuntimeError(
                f"Physics decision {decision.decision_id} contains targets outside "
                f"the component's {decision.collision_mode} role set."
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
    decisions = _v2_decisions_from_payload(payload)
    unresolved_components = _unresolved_components_from_payload(payload)
    _validate_component_decisions(components, decisions, unresolved_components)
    return decisions, unresolved_components


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
            "physical_properties": _merge_physical_properties(ordered),
            "confidence": min(decision.confidence for decision in ordered),
            "rationale": rationale,
        }
    )


def _rebase_v2_patch_payload_to_components(
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


def load_physics_behavior_assessment(path: Path | str) -> PhysicsBehaviorAssessment:
    """Load an agent-authored visual behavior assessment artifact."""

    assessment_path = Path(path).resolve()
    payload = json.loads(assessment_path.read_text(encoding="utf-8"))
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
    for frame_path in assessment.rendered_frames or assessment.checked_views:
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
    if not report.get("physics_scene_paths"):
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


def _multi_body_runtime_skip_message(body_count: int) -> str:
    return (
        "Runtime simulation currently validates a single bound rigid body; "
        f"this asset has {body_count} enabled rigid bodies, so whole-asset "
        "runtime validation is marked not evaluated until per-body validation "
        "is supported."
    )


def _remote_workbench_enabled(params: PhysicsApplyWorkflowInput) -> bool:
    return bool(params.workbench_url and params.workbench_session_id)


def _inspect_workflow_components(
    params: PhysicsApplyWorkflowInput,
    usd_path: Path,
) -> tuple[list[PhysicsComponent], str]:
    if _remote_workbench_enabled(params):
        from content_workbench_agent_client import (
            inspect_physics_components as workbench_inspect_physics_components,
        )

        assert params.workbench_url is not None
        assert params.workbench_session_id is not None
        inspect_response = workbench_inspect_physics_components(
            params.workbench_url,
            params.workbench_session_id,
            {
                "usd_path": str(usd_path),
                "path_space": "source",
            },
            timeout=params.workbench_timeout_s,
        )
        raw_components = inspect_response.get("components")
        if not isinstance(raw_components, list):
            raise RuntimeError(
                "Workbench physics inspection returned no components list."
            )
        components = [
            PhysicsComponent.model_validate(component) for component in raw_components
        ]
        source_digest = str(inspect_response.get("source_digest") or "")
    else:
        component_response = workbench_ops.inspect_components(usd_path)
        components = [
            PhysicsComponent.model_validate(component)
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
) -> tuple[ValidationEvidence, Path]:
    failures = [str(item) for item in result.get("failures") or []]
    warnings = [str(item) for item in result.get("warnings") or []]
    report_path_value = result.get("runtime_report")
    if not isinstance(report_path_value, str) or not report_path_value:
        raise RuntimeError("Workbench runtime validation returned no report path.")
    report_path = Path(report_path_value)
    artifacts = [
        EvidenceArtifact(
            kind=str(artifact.get("kind") or "runtime_artifact"),
            path=str(artifact.get("path") or ""),
            description=str(artifact.get("description") or ""),
        )
        for artifact in result.get("evidence_artifacts") or []
        if isinstance(artifact, dict) and artifact.get("path")
    ]
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
        metadata={
            "engine": engine,
            "duration_s": duration_s,
            "sample_fps": sample_fps,
            "settle_distance": result.get("settle_distance"),
            "trajectory_summary": result.get("summary"),
        },
    )
    return evidence, report_path


def validate_physics_runtime(
    *,
    physics_usd: Path | str,
    output_dir: Path | str,
    engine: PhysicsSimulationEngine = "ovphysx",
    duration_s: float = 1.0,
    dt: float = 1.0 / 240.0,
    sample_fps: int = 30,
    drop_height_m: float | None = None,
    acceptance: dict[str, Any] | None = None,
    physics_properties_status: Literal["pass", "fail"] = "pass",
) -> tuple[ValidationEvidence, Path | None]:
    """Run simulation-backed validation and return evidence plus report path."""

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
        result = workbench_ops.validate_runtime(
            physics_usd=physics_usd_path,
            output_dir=validation_dir,
            engine=engine,
            duration_s=duration_s,
            dt=dt,
            sample_fps=sample_fps,
            drop_height_m=drop_height_m,
            acceptance=acceptance,
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


def run_physics_apply_workflow(
    params: PhysicsApplyWorkflowInput,
) -> PhysicsApplyWorkflowResult:
    """Inspect, infer, author physics schema, and optionally simulate validate."""

    output_dir = params.output_dir.resolve()
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_usd = (params.output_usd_path or (output_dir / "physics.usda")).resolve()

    try:
        if bool(params.workbench_url) != bool(params.workbench_session_id):
            raise RuntimeError(
                "workbench_url and workbench_session_id must be provided together."
            )

        working_usd = Path(params.usd_path).resolve()
        supplied_patch_payload: dict[str, Any] | None = None
        supplied_v2_patch_prevalidated = False
        if params.decision_patch_path is not None:
            supplied_patch_payload = _load_decision_patch_payload(
                params.decision_patch_path
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
                _validate_v2_patch_payload_against_components(
                    supplied_patch_payload,
                    components=pre_topology_components,
                    source_digest=pre_topology_digest,
                )
                supplied_v2_patch_prevalidated = True
            topology_plan = json.loads(
                Path(params.topology_plan_path).resolve().read_text(encoding="utf-8")
            )
            if not isinstance(topology_plan, dict):
                raise RuntimeError("Physics topology plan must be a JSON object.")
            mobility_intent = str(topology_plan.get("mobility_intent") or "preserve")
            if _remote_workbench_enabled(params):
                from content_workbench_agent_client import (
                    apply_physics_topology_plan as workbench_apply_topology_plan,
                )

                assert params.workbench_url is not None
                assert params.workbench_session_id is not None
                topology_request: dict[str, Any] = {
                    "input_usd_path": str(working_usd),
                    "output_usd_path": None,
                    "expected_source_digest": str(
                        topology_plan.get("expected_source_digest") or ""
                    ),
                    "mobility_intent": str(
                        topology_plan.get("mobility_intent") or "preserve"
                    ),
                    "operations": list(topology_plan.get("operations") or []),
                    "invariants": dict(topology_plan.get("invariants") or {}),
                }
                if "schema_version" in topology_plan:
                    topology_request["schema_version"] = topology_plan["schema_version"]
                topology_report = workbench_apply_topology_plan(
                    params.workbench_url,
                    params.workbench_session_id,
                    topology_request,
                    timeout=params.workbench_timeout_s,
                )
            else:
                topology_report = workbench_ops.apply_topology_plan(
                    input_usd_path=working_usd,
                    output_usd_path=output_dir / "prepared.usda",
                    expected_source_digest=str(
                        topology_plan.get("expected_source_digest") or ""
                    ),
                    mobility_intent=str(
                        topology_plan.get("mobility_intent") or "preserve"
                    ),
                    operations=list(topology_plan.get("operations") or []),
                    invariants=dict(topology_plan.get("invariants") or {}),
                )
            working_usd = Path(str(topology_report["output_usd_path"])).resolve()
            topology_report_path = _write_json(
                raw_dir / "physics_topology_report.json", topology_report
            )

        components, source_digest = _inspect_workflow_components(params, working_usd)
        unresolved_components: list[dict[str, Any]] = []
        decision_patch_payload: dict[str, Any] | None = None
        if supplied_patch_payload is not None:
            decision_patch_payload = dict(supplied_patch_payload)
            if decision_patch_payload.get("schema_version") == (
                PHYSICS_DECISION_PATCH_SCHEMA_VERSION
            ):
                if params.topology_plan_path is not None:
                    if not supplied_v2_patch_prevalidated:
                        raise RuntimeError(
                            "Physics V2 decision patch was not validated before "
                            "topology repair."
                        )
                    (
                        decision_patch_payload,
                        decisions,
                        unresolved_components,
                    ) = _rebase_v2_patch_payload_to_components(
                        decision_patch_payload,
                        components=components,
                        source_digest=source_digest,
                        asset=working_usd,
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
                decisions = load_physics_decision_patch(params.decision_patch_path)
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

        candidate_prims_path = _write_json(
            raw_dir / "physics_components.json",
            {
                "asset": str(working_usd),
                "source_digest": source_digest,
                "component_count": len(components),
                "components": [_as_json(component) for component in components],
            },
        )
        if decision_patch_payload is not None:
            decision_patch_path = raw_dir / "physics_decision_patch.json"
            canonical_decision_patch_payload = (
                supplied_patch_payload
                if supplied_patch_payload is not None
                else decision_patch_payload
            )
            _write_json(decision_patch_path, canonical_decision_patch_payload)
            if decision_patch_payload != canonical_decision_patch_payload:
                apply_decision_patch_path = _write_json(
                    raw_dir / "physics_decision_patch_apply.json",
                    decision_patch_payload,
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
            )
            apply_decision_patch_path = decision_patch_path
        predictions_path = _write_predictions_jsonl(
            raw_dir / "physics_predictions.jsonl",
            decisions,
        )

        if not decisions:
            authored = working_usd
            authored_report = workbench_ops.inspect_authored_physics(authored)
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
                    "decision_patch_path": str(apply_decision_patch_path),
                    "predictions_jsonl": str(predictions_path),
                }
            )
            apply_report_path = _write_json(
                raw_dir / "physics_apply_report.json",
                authored_report,
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
            validation_evidence_path = _write_json(
                output_dir / "validation_evidence.json",
                validation_evidence.model_dump(mode="json"),
            )
            assignments_path = _write_json(
                output_dir / "physics_assignments.json",
                {
                    "schema_version": PHYSICS_ASSIGNMENTS_SCHEMA_VERSION,
                    "asset": str(Path(params.usd_path).resolve()),
                    "prepared_asset": str(working_usd),
                    "physics_usd": str(authored),
                    "candidate_count": len(components),
                    "component_count": len(components),
                    "decision_count": 0,
                    "decision_patch": str(decision_patch_path),
                    "apply_decision_patch": str(apply_decision_patch_path),
                    "decisions": [],
                    "unresolved_components": unresolved_components,
                    "mobility_intent": mobility_intent,
                    "apply_report": authored_report,
                    "validation_evidence": str(validation_evidence_path),
                    "simulation_report": None,
                },
            )

            return PhysicsApplyWorkflowResult(
                success=validation_evidence.sim_ready_status != "fail",
                asset=str(Path(params.usd_path).resolve()),
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
                validation_status=validation_evidence.sim_ready_status,
            )

        if _remote_workbench_enabled(params):
            from content_workbench_agent_client import (
                apply_physics_schema as workbench_apply_physics_schema,
            )

            assert params.workbench_url is not None
            assert params.workbench_session_id is not None
            authored_report = workbench_apply_physics_schema(
                params.workbench_url,
                params.workbench_session_id,
                {
                    "usd_path": str(working_usd),
                    "decision_patch_path": str(apply_decision_patch_path),
                    "predictions_jsonl_path": str(predictions_path),
                    "collision_approximation": params.collision_approximation,
                    "output_key": "classification",
                    "author_rigid_body": mobility_intent != "static",
                },
                timeout=params.workbench_timeout_s,
            )
        else:
            authored_report = workbench_ops.apply_schema(
                usd_path=working_usd,
                decision_patch_path=apply_decision_patch_path,
                predictions_jsonl_path=predictions_path,
                output_usd_path=output_usd,
                collision_approximation=params.collision_approximation,
                output_key="classification",
                author_rigid_body=mobility_intent != "static",
            )
        authored = Path(str(authored_report["physics_usd"])).resolve()
        if (
            _remote_workbench_enabled(params)
            and authored != output_usd
            and authored.exists()
        ):
            output_usd.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(authored, output_usd)
            authored = output_usd
            authored_report["physics_usd"] = str(authored)
        physics_status, physics_failures = _status_from_authored_report(
            authored_report,
            mobility_intent=mobility_intent,
        )
        authored_rigid_body_count = _authored_rigid_body_count(authored_report)
        if mobility_intent == "static" and authored_rigid_body_count != 0:
            physics_failures.append(
                "Static mobility intent requires zero enabled rigid bodies."
            )
            physics_status = "fail"
        apply_report_path = _write_json(
            raw_dir / "physics_apply_report.json",
            authored_report,
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
            and authored_rigid_body_count is not None
            and authored_rigid_body_count > 1
        ):
            runtime_skip_message = _multi_body_runtime_skip_message(
                authored_rigid_body_count
            )
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
        elif params.run_simulation:
            acceptance = _runtime_acceptance_from_authored_report(
                authored_report,
                mobility_intent=mobility_intent,
                drop_height_m=params.drop_height_m,
            )
            if _remote_workbench_enabled(params):
                from content_workbench_agent_client import (
                    validate_physics_runtime as workbench_validate_physics_runtime,
                )

                assert params.workbench_url is not None
                assert params.workbench_session_id is not None
                runtime_payload: dict[str, Any] = {
                    "physics_usd_path": str(authored),
                    "engine": params.simulation_engine,
                    "duration_s": params.simulation_duration_s,
                    "dt": params.simulation_dt,
                    "sample_fps": params.simulation_sample_fps,
                    "drop_height_m": params.drop_height_m,
                }
                if acceptance is not None:
                    runtime_payload["acceptance"] = acceptance
                runtime_result = workbench_validate_physics_runtime(
                    params.workbench_url,
                    params.workbench_session_id,
                    runtime_payload,
                    timeout=params.workbench_timeout_s,
                )
                validation_evidence, simulation_report_path = (
                    _runtime_result_to_evidence(
                        result=runtime_result,
                        physics_usd_path=authored,
                        engine=params.simulation_engine,
                        duration_s=params.simulation_duration_s,
                        sample_fps=params.simulation_sample_fps,
                        physics_properties_status=cast(Any, physics_status),
                    )
                )
            else:
                validation_evidence, simulation_report_path = validate_physics_runtime(
                    physics_usd=authored,
                    output_dir=output_dir / "runtime",
                    engine=params.simulation_engine,
                    duration_s=params.simulation_duration_s,
                    dt=params.simulation_dt,
                    sample_fps=params.simulation_sample_fps,
                    drop_height_m=params.drop_height_m,
                    acceptance=acceptance,
                    physics_properties_status=cast(Any, physics_status),
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

        if unresolved_components:
            unresolved_messages = [
                f"{item['component_id']}: {item['reason']}"
                for item in unresolved_components
            ]
            validation_evidence.unresolved_issues.extend(unresolved_messages)
            validation_evidence.warnings.extend(unresolved_messages)
            if validation_evidence.sim_ready_status != "fail":
                validation_evidence.sim_ready_status = "conditional"

        validation_evidence_path = _write_json(
            output_dir / "validation_evidence.json",
            validation_evidence.model_dump(mode="json"),
        )

        assignments_path = _write_json(
            output_dir / "physics_assignments.json",
            {
                "schema_version": PHYSICS_ASSIGNMENTS_SCHEMA_VERSION,
                "asset": str(Path(params.usd_path).resolve()),
                "prepared_asset": str(working_usd),
                "physics_usd": str(authored),
                "candidate_count": len(components),
                "component_count": len(components),
                "decision_count": len(decisions),
                "decision_patch": str(decision_patch_path),
                "apply_decision_patch": str(apply_decision_patch_path),
                "decisions": [_as_json(decision) for decision in decisions],
                "unresolved_components": unresolved_components,
                "mobility_intent": mobility_intent,
                "apply_report": authored_report,
                "validation_evidence": str(validation_evidence_path),
                "simulation_report": str(simulation_report_path)
                if simulation_report_path
                else None,
            },
        )

        success = validation_evidence.sim_ready_status != "fail"
        error_message: str | None = None
        if params.fail_on_validation_error and not success:
            error_message = "Physics workflow validation failed: " + "; ".join(
                validation_evidence.failures or ["unknown validation failure"]
            )

        return PhysicsApplyWorkflowResult(
            success=success,
            asset=str(Path(params.usd_path).resolve()),
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
            validation_status=validation_evidence.sim_ready_status,
            error=error_message,
        )
    except Exception as exc:
        return PhysicsApplyWorkflowResult(
            success=False,
            asset=str(Path(params.usd_path).resolve()),
            output_dir=str(output_dir),
            physics_usd_path=str(output_usd) if output_usd.exists() else None,
            validation_status="fail",
            error=str(exc),
        )
