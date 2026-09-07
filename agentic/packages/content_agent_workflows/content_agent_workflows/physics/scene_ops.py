# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physics-domain scene adapters and workflow-owned policy implementation.

The ``usd-cli`` boundary exposes only explicit schema patches, structural
validation, explicit simulation, and raw topology inspection.  This module owns
the domain interpretation layered over those primitives: component grouping,
legacy patch compatibility, topology-plan validation, and runtime-evidence
normalization.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from content_agent_workflows.common.usd_cli_session import WorkflowUsdCliSession

PhysicsSimulationEngine = Literal["newton", "ovphysx", "fake", "none"]
VALID_SIMULATION_ENGINES = {"newton", "ovphysx", "fake", "none"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def _round_float(value: float, digits: int = 6) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(float(value), digits)


def _mesh_material(prim: Any) -> tuple[str | None, str | None]:
    from pxr import UsdShade

    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not material:
        return None, None
    material_prim = material.GetPrim()
    if not material_prim:
        return None, None
    return str(material_prim.GetPath()), material_prim.GetName()


def inspect_mesh_candidates(
    usd_path: Path | str,
    *,
    root_prim_path: str | None = None,
    include_existing_schema: bool = True,
    path_space: str = "source",
) -> dict[str, Any]:
    """Inspect a USD asset and return mesh prim physics candidates."""

    from pxr import Usd, UsdGeom

    path = Path(usd_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input USD not found: {path}")

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {path}")

    root_prim = stage.GetPrimAtPath(root_prim_path) if root_prim_path else None
    if root_prim_path and (not root_prim or not root_prim.IsValid()):
        raise RuntimeError(f"Root prim not found: {root_prim_path}")

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )

    candidates: list[dict[str, Any]] = []
    traversal = (
        Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies())
        if root_prim
        else Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    )
    for prim in traversal:
        if not prim.IsA(UsdGeom.Mesh):
            continue
        bbox = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        size_stage = bbox.GetSize()
        bbox_min = [float(v) * meters_per_unit for v in bbox.GetMin()]
        bbox_max = [float(v) * meters_per_unit for v in bbox.GetMax()]
        bbox_size = [max(float(v) * meters_per_unit, 0.0) for v in size_stage]
        bbox_volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
        material_path, material_name = _mesh_material(prim)
        candidate = {
            "prim_path": str(prim.GetPath()),
            "prim_name": prim.GetName(),
            "type_name": prim.GetTypeName(),
            "material_path": material_path,
            "material_name": material_name,
            "bbox_min_m": [_round_float(v) for v in bbox_min],
            "bbox_max_m": [_round_float(v) for v in bbox_max],
            "bbox_size_m": [_round_float(v) for v in bbox_size],
            "bbox_volume_m3": _round_float(bbox_volume, digits=12),
            "path_space": path_space,
        }
        if include_existing_schema:
            candidate["existing_physics_schemas"] = list(prim.GetAppliedSchemas())
        candidates.append(candidate)

    return {
        "asset": str(path),
        "path_space": path_space,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def inspect_components(
    usd_path: Path | str,
    *,
    root_prim_path: str | None = None,
    path_space: str = "source",
) -> dict[str, Any]:
    """Inspect logical physics components using shared USD role analysis."""

    from world_understanding.functions.physics.physics_topology import (
        inspect_physics_components,
    )

    return inspect_physics_components(
        usd_path,
        root_prim_path=root_prim_path,
        path_space=path_space,
    )


def inspect_topology(
    usd_path: Path | str,
    *,
    root_prim_path: str | None = None,
    path_space: str = "source",
) -> dict[str, Any]:
    """Inspect authored rigid-body, collider, joint, and articulation facts."""

    from world_understanding.functions.physics.physics_topology import (
        inspect_physics_topology,
    )

    return inspect_physics_topology(
        usd_path,
        root_prim_path=root_prim_path,
        path_space=path_space,
    )


def apply_topology_plan(
    *,
    input_usd_path: Path | str,
    output_usd_path: Path | str,
    expected_source_digest: str,
    mobility_intent: str,
    operations: list[dict[str, Any]],
    invariants: dict[str, Any],
    joint_endpoint_owner_promotions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply and audit an explicit topology plan on a derivative USD."""

    from world_understanding.functions.physics.physics_topology import (
        PhysicsTopologyPlanError,
        apply_physics_topology_plan,
        sha256_file,
    )
    from world_understanding.utils.usd.package import extract_usdz_package_for_edit

    source = Path(input_usd_path).resolve()
    output = Path(output_usd_path).resolve()
    packaged_source: dict[str, str] | None = None
    if source.suffix.lower() == ".usdz":
        # The coordinator's topology plan is bound to the composed package it
        # inspected. Verify that binding before changing transport into an
        # editable layer; the extracted layer necessarily has a different
        # dependency digest because its resolved identifiers have changed.
        packaged_digest = sha256_file(source)
        if packaged_digest != expected_source_digest:
            raise PhysicsTopologyPlanError(
                "Source digest mismatch; inspect the current asset before applying "
                "a plan"
            )
        if output.suffix.lower() == ".usdz":
            raise PhysicsTopologyPlanError(
                "Topology plans must write a USD layer output, not a USDZ package"
            )
        extract_dir = output.parent / f"{output.stem}_source"
        extracted_root = extract_usdz_package_for_edit(source, extract_dir)
        prepared_output = extract_dir / output.name
        if prepared_output == extracted_root:
            prepared_output = extract_dir / (
                f"{output.stem}_topology{output.suffix or '.usda'}"
            )
        extracted_digest = sha256_file(extracted_root)
        packaged_source = {
            "input_usdz_path": str(source),
            "source_digest": packaged_digest,
            "extracted_root_path": str(extracted_root),
            "extracted_source_digest": extracted_digest,
        }
        source = extracted_root
        output = prepared_output
        expected_source_digest = extracted_digest

    report = apply_physics_topology_plan(
        input_usd_path=source,
        output_usd_path=output,
        expected_source_digest=expected_source_digest,
        mobility_intent=mobility_intent,
        operations=operations,
        invariants=invariants,
        joint_endpoint_owner_promotions=joint_endpoint_owner_promotions,
    )
    if packaged_source is not None:
        report["prepared_input_usd_path"] = report["input_usd_path"]
        report["prepared_source_digest"] = report["source_digest"]
        report["input_usd_path"] = packaged_source["input_usdz_path"]
        report["source_digest"] = packaged_source["source_digest"]
        report["package_extraction"] = packaged_source
    report_path = output.with_name(f"{output.stem}_topology_report.json")
    report["topology_report"] = str(report_path)
    _write_json(report_path, report)
    return report


def inspect_authored_physics(physics_usd: Path | str) -> dict[str, Any]:
    """Summarize authored USD physics schemas."""

    from pxr import Usd, UsdGeom, UsdPhysics
    from world_understanding.utils.physics_units import (
        STANDARD_GRAVITY_M_PER_S2,
        acceleration_stage_units_to_m_per_s2,
        looks_like_unscaled_standard_gravity,
        validate_meters_per_unit,
    )

    physics_usd_path = Path(physics_usd).resolve()
    stage = Usd.Stage.Open(str(physics_usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open authored physics USD: {physics_usd_path}")

    default_prim = stage.GetDefaultPrim()
    default_path = str(default_prim.GetPath()) if default_prim else None
    meters_per_unit = validate_meters_per_unit(
        float(UsdGeom.GetStageMetersPerUnit(stage))
    )
    meters_per_unit_authored = bool(UsdGeom.StageHasAuthoredMetersPerUnit(stage))
    rigid_body_paths: list[str] = []
    enabled_rigid_body_paths: list[str] = []
    disabled_rigid_body_paths: list[str] = []
    collision_paths: list[str] = []
    physics_material_paths: list[str] = []
    scene_paths: list[str] = []
    scene_details: list[dict[str, Any]] = []
    physics_warnings: list[str] = []
    if not meters_per_unit_authored:
        physics_warnings.append(
            "metersPerUnit is not authored; OpenUSD fallback 0.01 is in effect"
        )
    physics_traversal = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for prim in Usd.PrimRange.Stage(stage, physics_traversal):
        path = str(prim.GetPath())
        if prim.IsA(UsdPhysics.Scene):
            scene_paths.append(path)
            scene = UsdPhysics.Scene(prim)
            gravity_attr = scene.GetGravityMagnitudeAttr()
            gravity_authored = bool(gravity_attr.HasAuthoredValueOpinion())
            gravity_value = gravity_attr.Get()
            gravity_stage_units: float | None = None
            gravity_m_per_s2: float | None = None
            gravity_source = "invalid"
            if gravity_value is not None:
                raw_gravity = float(gravity_value)
                if raw_gravity < 0.0:
                    gravity_source = "usd_earth_gravity_default"
                    gravity_m_per_s2 = STANDARD_GRAVITY_M_PER_S2
                elif math.isfinite(raw_gravity):
                    gravity_source = "authored"
                    gravity_stage_units = raw_gravity
                    gravity_m_per_s2 = acceleration_stage_units_to_m_per_s2(
                        raw_gravity,
                        meters_per_unit,
                    )
                    if gravity_authored and looks_like_unscaled_standard_gravity(
                        raw_gravity,
                        meters_per_unit,
                    ):
                        physics_warnings.append(
                            f"{path} may contain legacy unscaled gravity: "
                            f"{raw_gravity:.6g} stage units/s^2 at "
                            f"metersPerUnit={meters_per_unit:.6g}"
                        )
                else:
                    physics_warnings.append(
                        f"{path} has invalid non-finite gravity magnitude: "
                        f"{raw_gravity!r}"
                    )
            direction = scene.GetGravityDirectionAttr().Get()
            scene_details.append(
                {
                    "path": path,
                    "gravity_authored": gravity_authored,
                    "gravity_source": gravity_source,
                    "gravity_magnitude_stage_units": gravity_stage_units,
                    "gravity_magnitude_m_per_s2": gravity_m_per_s2,
                    "gravity_direction": (
                        [float(value) for value in direction]
                        if direction is not None
                        else None
                    ),
                }
            )
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_paths.append(path)
            enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
            if enabled is False:
                disabled_rigid_body_paths.append(path)
            else:
                enabled_rigid_body_paths.append(path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_paths.append(path)
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            physics_material_paths.append(path)
    return {
        "physics_usd": str(physics_usd_path),
        "default_prim": default_path,
        "meters_per_unit": meters_per_unit,
        "meters_per_unit_authored": meters_per_unit_authored,
        "physics_scene_paths": scene_paths,
        "physics_scene_details": scene_details,
        "physics_warnings": physics_warnings,
        "rigid_body_paths": rigid_body_paths,
        "enabled_rigid_body_paths": enabled_rigid_body_paths,
        "disabled_rigid_body_paths": disabled_rigid_body_paths,
        "collision_paths": collision_paths,
        "physics_material_paths": physics_material_paths,
        "rigid_body_count": len(rigid_body_paths),
        "enabled_rigid_body_count": len(enabled_rigid_body_paths),
        "disabled_rigid_body_count": len(disabled_rigid_body_paths),
        "collision_count": len(collision_paths),
        "physics_material_count": len(physics_material_paths),
    }


def select_physics_scene_path(physics_usd: Path | str) -> str:
    """Select the workflow-owned target for physics-scene authoring.

    Existing ``UsdPhysics.Scene`` prims are reused deterministically.  A scene is
    created only when none exists, and its path is always a free child of the
    stage default prim so the workflow never invents a competing pseudo-root
    ``/PhysicsScene``.
    """

    from pxr import Usd, UsdPhysics

    path = Path(physics_usd).resolve()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage for physics-scene policy: {path}")

    existing_paths = sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange.Stage(
            stage,
            Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate),
        )
        if prim.IsA(UsdPhysics.Scene)
    )
    if existing_paths:
        return existing_paths[0]

    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError(
            "Physics authoring requires a stage default prim when no existing "
            "UsdPhysics.Scene is available."
        )

    base_path = default_prim.GetPath().AppendChild("PhysicsScene")
    candidate = base_path
    suffix = 2
    while stage.GetPrimAtPath(candidate).IsValid():
        candidate = default_prim.GetPath().AppendChild(f"PhysicsScene_{suffix}")
        suffix += 1
    return str(candidate)


def _resolve_collider_targets(decision: dict[str, Any]) -> list[str]:
    """Collider authoring targets for a V2 decision.

    For ``author_on_targets`` with an empty ``collider_paths``, fall back to the
    decision's identified ``visual_evidence_paths`` (the visible geometry to
    author colliders on). Mirrors the workflow's auto-inference path
    (``component.collider_paths or component.visual_evidence_paths``) so a bare
    single-mesh asset whose agent named its target geometry only under
    ``visual_evidence_paths`` is not rejected purely for not duplicating it.
    """
    targets = decision.get("collider_paths") or []
    if not targets and decision.get("collision_mode") == "author_on_targets":
        targets = decision.get("visual_evidence_paths") or []
    return list(targets)


def _prediction_records_from_decision_patch(
    decision_patch_path: Path | str,
) -> list[dict[str, Any]]:
    patch_path = Path(decision_patch_path).resolve()
    payload = json.loads(patch_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("physics decision patch must be a JSON object")
    schema_version = payload.get("schema_version")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("physics decision patch must include a decisions list")

    records: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ValueError(f"physics decision at index {index} must be an object")
        prim_paths = (
            _resolve_collider_targets(decision)
            if schema_version == "content-agent-workflows.physics-decision-patch.v2"
            else decision.get("prim_paths")
        )
        if not isinstance(prim_paths, list) or not prim_paths:
            raise ValueError(
                f"physics decision at index {index} must include authoring targets"
            )
        if (
            schema_version == "content-agent-workflows.physics-decision-patch.v2"
            and decision.get("collision_mode")
            not in {"preserve_existing", "author_on_targets"}
        ):
            raise ValueError(
                f"physics decision at index {index} has an invalid collision_mode"
            )
        physical_properties = decision.get("physical_properties")
        if not isinstance(physical_properties, dict):
            raise ValueError(
                f"physics decision at index {index} must include physical_properties"
            )
        per_prim_properties = dict(physical_properties)
        estimated_mass = per_prim_properties.get("estimated_mass_kg")
        if estimated_mass is not None and len(prim_paths) > 1:
            per_prim_properties["estimated_mass_kg"] = float(estimated_mass) / len(
                prim_paths
            )
        quality_warnings = decision.get("quality_warnings") or []
        if not isinstance(quality_warnings, list):
            raise ValueError(
                f"physics decision at index {index} has invalid quality_warnings"
            )
        for warning_index, warning in enumerate(quality_warnings):
            if (
                not isinstance(warning, dict)
                or not isinstance(warning.get("code"), str)
                or not warning["code"].strip()
            ):
                raise ValueError(
                    "physics decision at index "
                    f"{index} has invalid quality_warnings[{warning_index}]"
                )
            severity = warning.get("severity", "warning")
            if severity not in {"info", "warning", "error"} or (
                warning["code"] == "mass_scale_suspicious" and severity != "warning"
            ):
                raise ValueError(
                    "physics decision at index "
                    f"{index} has invalid quality_warnings[{warning_index}]"
                )
        for prim_path in prim_paths:
            if not isinstance(prim_path, str) or not prim_path:
                raise ValueError(
                    f"physics decision at index {index} contains an invalid prim path"
                )
            classification = {
                "decision_id": decision.get("decision_id"),
                "component_id": decision.get("component_id"),
                "component": decision.get("component_label")
                or decision.get("component_id"),
                "material": decision.get("inferred_material_family"),
                "physical_properties": per_prim_properties,
                "collision_mode": decision.get("collision_mode"),
                "collision_approximation": decision.get("collision_approximation"),
                "mass_authoring_path": decision.get("mass_authoring_path"),
                "component_estimated_mass_kg": estimated_mass,
                "confidence": decision.get("confidence"),
                "reasoning": decision.get("rationale"),
            }
            rigid_body_grouping = decision.get("rigid_body_grouping")
            if rigid_body_grouping:
                classification["rigid_body_grouping"] = rigid_body_grouping
            if quality_warnings:
                classification["quality_warnings"] = quality_warnings
            record = {
                "id": prim_path,
                "classification": classification,
                "source": "content_agent_workflows.physics.scene_ops.decision_patch",
            }
            if quality_warnings:
                record["quality_warnings"] = quality_warnings
            records.append(record)
    return records


def _validate_v2_decision_patch(
    usd_path: Path | str,
    decision_patch_path: Path | str,
) -> None:
    payload = json.loads(Path(decision_patch_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("physics decision patch must be a JSON object")
    if payload.get("schema_version") != (
        "content-agent-workflows.physics-decision-patch.v2"
    ):
        return
    inspection = inspect_components(usd_path)
    expected_digest = payload.get("source_digest")
    if expected_digest != inspection["source_digest"]:
        raise ValueError(
            "Physics V2 decision patch source_digest does not match the input USD"
        )
    components = {
        component["component_id"]: component for component in inspection["components"]
    }
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Physics V2 decision patch must include a decisions list")
    decision_ids: list[str] = []
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ValueError(f"Physics V2 decision at index {index} must be an object")
        component_id = decision.get("component_id")
        if not isinstance(component_id, str):
            raise ValueError(
                f"Physics V2 decision at index {index} requires component_id"
            )
        decision_ids.append(component_id)
    unresolved = payload.get("unresolved_components") or []
    if not isinstance(unresolved, list):
        raise ValueError("Physics V2 unresolved_components must be a list")
    unresolved_ids: list[str] = []
    for item in unresolved:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("component_id"), str)
            or not str(item.get("reason") or "").strip()
        ):
            raise ValueError(
                "Each unresolved physics component requires component_id and reason"
            )
        unresolved_ids.append(item["component_id"])
    all_ids = [*decision_ids, *unresolved_ids]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(components):
        raise ValueError(
            "Physics V2 patch must cover each component exactly once as decided or unresolved"
        )
    for decision in decisions:
        component = components[decision["component_id"]]
        raw_targets = decision.get("collider_paths")
        if not isinstance(raw_targets, list) or not all(
            isinstance(target, str) for target in raw_targets
        ):
            raise ValueError("Physics V2 decisions require collider_paths strings")
        targets = set(_resolve_collider_targets(decision))
        if targets & set(component["helper_paths"]):
            raise ValueError("Physics V2 decisions may not target helper geometry")
        mode = decision.get("collision_mode")
        if mode == "author_on_targets" and component["collider_paths"]:
            raise ValueError(
                "Physics V2 decisions must preserve existing colliders when present"
            )
        allowed = set(
            component[
                "collider_paths"
                if mode == "preserve_existing"
                else "visual_evidence_paths"
            ]
        )
        if not targets or not targets <= allowed:
            raise ValueError(
                f"Physics V2 {mode!r} targets do not match inspected component roles"
            )
        if decision.get("body_root_path") != component["body_root_path"]:
            raise ValueError(
                "Physics V2 decision body_root_path changed after inspection"
            )
        if decision.get("mass_authoring_path") != component["body_root_path"]:
            raise ValueError(
                "Physics V2 decision mass_authoring_path must be the component body root"
            )


def _write_predictions_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_json_safe(record), allow_nan=False, sort_keys=True))
            f.write("\n")
    return path


def _prediction_target_paths(path: Path) -> list[str]:
    """Return valid top-level prim IDs from a predictions JSONL artifact."""

    targets: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid predictions JSONL at line {line_number}: {exc.msg}"
                ) from exc
            prim_path = record.get("id") if isinstance(record, dict) else None
            if isinstance(prim_path, str) and prim_path:
                targets.append(prim_path)
    return list(dict.fromkeys(targets))


def _instance_proxy_prediction_paths(
    usd_path: Path,
    predictions_path: Path,
) -> list[str]:
    """Identify selected collider targets that require de-instancing."""

    from pxr import Usd

    if not usd_path.is_file():
        return []
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        return []
    return [
        prim_path
        for prim_path in _prediction_target_paths(predictions_path)
        if (prim := stage.GetPrimAtPath(prim_path)).IsValid() and prim.IsInstanceProxy()
    ]


def _deinstance_prediction_targets(
    usd_path: Path,
    output_path: Path,
    proxy_paths: list[str],
    *,
    approved_dependency_roots: Iterable[Path],
) -> Path:
    """Create an authorable derivative for selected instance-proxy targets."""

    from pxr import Usd
    from world_understanding.functions.graphics.so_export import export_stage_portably

    source_stage = Usd.Stage.Open(str(usd_path))
    if source_stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    export_stage_portably(
        source_stage,
        output_path,
        approved_dependency_roots=approved_dependency_roots,
        export_layer=source_stage.GetRootLayer(),
    )
    editable = Usd.Stage.Open(str(output_path))
    if editable is None:
        raise RuntimeError(
            f"Failed to open portable USD stage for physics: {output_path}"
        )

    for prim_path in proxy_paths:
        prim = editable.GetPrimAtPath(prim_path)
        deinstanced_roots: set[str] = set()
        while prim.IsValid() and prim.IsInstanceProxy():
            instance_root = prim
            while (
                instance_root.IsValid()
                and not instance_root.IsPseudoRoot()
                and (not instance_root.IsInstance() or instance_root.IsInstanceProxy())
            ):
                instance_root = instance_root.GetParent()
            if (
                not instance_root.IsValid()
                or instance_root.IsPseudoRoot()
                or not instance_root.IsInstance()
                or instance_root.IsInstanceProxy()
            ):
                raise RuntimeError(
                    "Physics collider target has no editable instance root: "
                    f"{prim_path}"
                )
            instance_root_path = str(instance_root.GetPath())
            if instance_root_path in deinstanced_roots:
                raise RuntimeError(
                    "Physics collider target remained an instance proxy after "
                    f"de-instancing {instance_root_path}: {prim_path}"
                )
            deinstanced_roots.add(instance_root_path)
            instance_root.SetInstanceable(False)
            prim = editable.GetPrimAtPath(prim_path)
        if not prim.IsValid() or prim.IsInstanceProxy():
            raise RuntimeError(
                f"Physics collider target is not authorable after de-instancing: "
                f"{prim_path}"
            )

    if not editable.GetRootLayer().Save():
        raise RuntimeError(f"Failed to save de-instanced physics stage: {output_path}")
    return output_path.resolve(strict=True)


def apply_schema(
    *,
    usd_path: Path | str,
    decision_patch_path: Path | str | None = None,
    predictions_jsonl_path: Path | str,
    output_usd_path: Path | str,
    collision_approximation: str = "convexHull",
    output_key: str = "classification",
    author_rigid_body: bool = True,
    approved_dependency_roots: Iterable[Path | str] | None = None,
) -> dict[str, Any]:
    """Author USD physics schema from accepted physics predictions.

    ``approved_dependency_roots`` bounds the filesystem dependencies that may
    be copied into a portable authored-output sidecar. Direct callers default
    to the input USD's parent. usd-cli service callers should also pass the
    trusted source-session input root when authoring a generated derivative.
    Post-author inspection validates resolved ``metersPerUnit`` and raises on
    invalid stage units even when the authored USD was already written.
    """

    from physics_agent.functions.apply_physics import apply_physics
    from world_understanding.functions.graphics.so_export import (
        _normalize_dependency_roots,
    )

    source_path = Path(usd_path).resolve()
    output_path = Path(output_usd_path).resolve()
    dependency_roots = _normalize_dependency_roots(
        tuple(approved_dependency_roots)
        if approved_dependency_roots is not None
        else (source_path.parent,)
    )
    resolved_predictions_path = Path(predictions_jsonl_path).resolve()
    generated_predictions: list[dict[str, Any]] | None = None
    with ExitStack() as cleanup:
        if decision_patch_path is not None:
            _validate_v2_decision_patch(usd_path, decision_patch_path)
            generated_predictions = _prediction_records_from_decision_patch(
                decision_patch_path
            )
            temporary_predictions = Path(
                cleanup.enter_context(
                    tempfile.TemporaryDirectory(prefix="usd-cli-physics-predictions-")
                )
            )
            resolved_predictions_path = _write_predictions_jsonl(
                temporary_predictions / "predictions.jsonl",
                generated_predictions,
            )

        authoring_source = source_path
        proxy_paths = _instance_proxy_prediction_paths(
            source_path,
            resolved_predictions_path,
        )
        if proxy_paths:
            temporary_authoring_source = Path(
                cleanup.enter_context(
                    tempfile.TemporaryDirectory(prefix="usd-cli-physics-deinstance-")
                )
            )
            authoring_source = _deinstance_prediction_targets(
                source_path,
                temporary_authoring_source / f"{source_path.stem}.usdc",
                proxy_paths,
                approved_dependency_roots=dependency_roots,
            )
            dependency_roots = (*dependency_roots, authoring_source.parent)

        authored = Path(
            apply_physics(
                str(authoring_source),
                str(resolved_predictions_path),
                str(output_path),
                collision_approx=collision_approximation,
                output_key=output_key,
                author_rigid_body=author_rigid_body,
                approved_dependency_roots=dependency_roots,
            )
        ).resolve()
        if generated_predictions is not None:
            resolved_predictions_path = _write_predictions_jsonl(
                output_path.with_name(f"{output_path.stem}_predictions.jsonl"),
                generated_predictions,
            )
    report = inspect_authored_physics(authored)
    report["operation"] = "physics.apply_schema"
    report["collision_approximation"] = collision_approximation
    report["author_rigid_body"] = author_rigid_body
    if decision_patch_path is not None:
        report["decision_patch_path"] = str(Path(decision_patch_path).resolve())
    report["predictions_jsonl"] = str(resolved_predictions_path)
    if decision_patch_path is not None:
        report["source_predictions_jsonl"] = str(Path(predictions_jsonl_path).resolve())
    return report


def _fake_trajectory(
    *,
    rest_position: list[float],
    world_up: list[float],
    duration_s: float,
    sample_fps: int,
    drop_height_m: float,
) -> list[tuple[float, list[float], list[float]]]:
    sample_count = max(2, int(round(duration_s * sample_fps)) + 1)
    up_idx = max(range(3), key=lambda idx: abs(float(world_up[idx])))
    trajectory: list[tuple[float, list[float], list[float]]] = []
    for i in range(sample_count):
        t = duration_s * i / (sample_count - 1)
        alpha = i / (sample_count - 1)
        height_offset = max(drop_height_m * (1.0 - alpha) ** 2, 0.0)
        pose = [float(v) for v in rest_position] + [0.0, 0.0, 0.0, 1.0]
        pose[up_idx] += height_offset
        velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        velocity[up_idx] = (
            -2.0
            * drop_height_m
            * (1.0 - alpha)
            / max(
                duration_s,
                1e-6,
            )
        )
        trajectory.append((float(t), pose, velocity))
    return trajectory


def _write_trajectory_response(path: Path, response: dict[str, Any]) -> Path:
    safe_response = dict(response)
    trajectory = safe_response.pop("trajectory", None)
    if trajectory is not None:
        safe_response["trajectory_sample_count"] = len(trajectory)
    _write_json(path, safe_response)
    return path


def _read_usd_cli_trajectory(
    path: Path,
) -> list[tuple[float, list[float], list[float]]]:
    trajectory: list[tuple[float, list[float], list[float]]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                time_s = float(row["t"])
                pose = [float(value) for value in row["pose"]]
                velocity = [float(value) for value in row["vel"]]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"usd-cli trajectory row {line_number} is invalid: {path}"
                ) from exc
            trajectory.append((time_s, pose, velocity))
    return trajectory


def _meters_per_stage_unit(scene_info: dict[str, Any]) -> float:
    explicit_mpu = scene_info.get("meters_per_unit")
    if explicit_mpu is not None:
        try:
            value = float(explicit_mpu)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(value) and value > 0.0:
                return value

    bbox_size_m = scene_info.get("bbox_size_m") or []
    bbox_min = scene_info.get("bbox_min_local_stage") or []
    bbox_max = scene_info.get("bbox_max_local_stage") or []
    ratios = [
        float(size_m) / abs(float(high) - float(low))
        for size_m, low, high in zip(
            bbox_size_m,
            bbox_min,
            bbox_max,
            strict=False,
        )
        if abs(float(high) - float(low)) > 1e-12 and float(size_m) > 0
    ]
    if not ratios:
        return 1.0
    ratios.sort()
    return ratios[len(ratios) // 2]


def _gravity_magnitude_m_per_s2(scene_info: dict[str, Any]) -> float:
    for key in (
        "gravity_magnitude_m_per_s2",
        "gravity_m_per_s2",
        "gravity",
    ):
        value = scene_info.get(key)
        if value is None:
            continue
        try:
            gravity = abs(float(value))
        except (TypeError, ValueError):
            continue
        if math.isfinite(gravity):
            return gravity
    return 9.81


def _rotate_vector_by_quaternion(
    vector: list[float], quaternion_xyzw: list[float]
) -> list[float]:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _runtime_acceptance_metrics(
    *,
    trajectory: list[tuple[float, list[float], list[float]]],
    scene_info: dict[str, Any],
    loaded_body_count: int,
    acceptance: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Calculate solver-backed continuity metrics and hard failures."""

    world_up = [float(value) for value in scene_info.get("world_up") or [0, 0, 1]]
    meters_per_unit = _meters_per_stage_unit(scene_info)
    gravity_magnitude = _gravity_magnitude_m_per_s2(scene_info)
    first_step_displacement_m: float | None = None
    expected_ballistic_displacement_m: float | None = None
    initial_pose_discontinuity = False
    if len(trajectory) >= 2:
        first_time, first_pose, _first_velocity = trajectory[0]
        second_time, second_pose, _second_velocity = trajectory[1]
        delta_t = max(float(second_time) - float(first_time), 0.0)
        first_step_displacement_m = (
            math.sqrt(
                sum(
                    (float(second_pose[index]) - float(first_pose[index])) ** 2
                    for index in range(3)
                )
            )
            * meters_per_unit
        )
        expected_ballistic_displacement_m = 0.5 * gravity_magnitude * delta_t * delta_t
        threshold = acceptance.get("max_initial_pose_displacement_m")
        if threshold is None:
            threshold = expected_ballistic_displacement_m * float(
                acceptance.get("ballistic_displacement_multiplier", 3.0)
            ) + float(acceptance.get("initial_pose_tolerance_m", 0.002))
        initial_pose_discontinuity = first_step_displacement_m > float(threshold)

    gravity_response_observed = False
    if len(trajectory) >= 2:
        origin = trajectory[0][1]
        for _time, pose, velocity in trajectory[1 : min(len(trajectory), 5)]:
            position_delta = sum(
                (float(pose[index]) - float(origin[index])) * world_up[index]
                for index in range(3)
            )
            projected_velocity = sum(
                float(velocity[index]) * world_up[index] for index in range(3)
            )
            if position_delta < -1e-7 or projected_velocity < -1e-7:
                gravity_response_observed = True
                break

    maximum_ground_penetration_m: float | None = None
    resting_ground_penetration_m: float | None = None
    ground_clearance_geometry = "unavailable"
    bbox_min = scene_info.get("bbox_min_local_stage")
    bbox_max = scene_info.get("bbox_max_local_stage")
    geometry_points = _filtered_geometry_points(
        scene_info.get("geometry_points_local_stage")
    )

    if trajectory and (geometry_points or (bbox_min and bbox_max)):
        rotate_bounds = scene_info.get("bbox_local_stage_space") == "pose_local"
        bbox_local_scale = scene_info.get("bbox_local_stage_scale") or [1.0, 1.0, 1.0]
        try:
            scale = [float(bbox_local_scale[index]) for index in range(3)]
        except (IndexError, TypeError, ValueError):
            scale = [1.0, 1.0, 1.0]
        if not all(math.isfinite(value) for value in scale):
            scale = [1.0, 1.0, 1.0]
        if geometry_points:
            local_support_points = geometry_points
            ground_clearance_geometry = "collider_mesh_vertices"
        else:
            local_support_points = [
                [x, y, z]
                for x in (float(bbox_min[0]), float(bbox_max[0]))
                for y in (float(bbox_min[1]), float(bbox_max[1]))
                for z in (float(bbox_min[2]), float(bbox_max[2]))
            ]
            ground_clearance_geometry = "conservative_bbox_corners"
        maximum_ground_penetration_m = 0.0
        for _time, pose, _velocity in trajectory:
            projections: list[float] = []
            for local_support_point in local_support_points:
                local_point = list(local_support_point)
                if rotate_bounds:
                    local_point = [
                        local_point[index] * scale[index] for index in range(3)
                    ]
                    offset = _rotate_vector_by_quaternion(
                        local_point, [float(value) for value in pose[3:7]]
                    )
                else:
                    offset = local_point
                world_point = [offset[index] + float(pose[index]) for index in range(3)]
                projections.append(
                    sum(world_point[index] * world_up[index] for index in range(3))
                )
            sample_penetration_m = max(0.0, -min(projections) * meters_per_unit)
            maximum_ground_penetration_m = max(
                maximum_ground_penetration_m,
                sample_penetration_m,
            )
            final_sample_penetration_m = sample_penetration_m

        # The final sample's penetration is recorded separately: the gated
        # maximum is dominated by the drop impact, while the resting value is
        # what stays visible in the settled scene. Recorded for calibration;
        # not gated (yet) — the motivating soft-contact artifact on large
        # assets persists at rest, so a rest gate needs its own
        # suite-measured factor first. Rest requires SUSTAINED stillness,
        # not one quiet sample: a window that ends at a bounce apex has zero
        # instantaneous speed but is not at rest, and recording its
        # mid-bounce depth under this name would bias any factor calibrated
        # from it. Every sample in the trailing 0.25 s (at least two) must
        # stay under both thresholds — linear AND angular: a body whose
        # center of mass has stopped translating but is still tumbling has
        # changing contact support, not a rest pose.
        def _sample_is_still(velocity: list[float]) -> bool:
            linear_speed_m_per_s = (
                math.sqrt(sum(float(velocity[index]) ** 2 for index in range(3)))
                * meters_per_unit
            )
            angular_speed_rad_per_s = (
                math.sqrt(sum(float(velocity[index]) ** 2 for index in range(3, 6)))
                if len(velocity) >= 6
                else 0.0
            )
            return linear_speed_m_per_s <= 0.05 and angular_speed_rad_per_s <= 0.1

        trailing_window_start_s = float(trajectory[-1][0]) - 0.25
        trailing_samples_settled = [
            _sample_is_still(velocity)
            for sample_time, _pose, velocity in trajectory
            if float(sample_time) >= trailing_window_start_s
        ]
        if len(trailing_samples_settled) >= 2 and all(trailing_samples_settled):
            resting_ground_penetration_m = final_sample_penetration_m

    failures: list[str] = []
    if (
        acceptance.get("detect_initial_pose_discontinuity", True)
        and initial_pose_discontinuity
    ):
        failures.append(
            "Simulation initial pose was discontinuous: first-step displacement "
            f"was {first_step_displacement_m:.6f} m."
        )
    penetration_limit_raw = acceptance.get("max_ground_penetration_m", 0.005)
    if (
        penetration_limit_raw is not None
        and maximum_ground_penetration_m is not None
        and maximum_ground_penetration_m > float(penetration_limit_raw)
    ):
        failures.append(
            "Simulation body penetrated the ground by "
            f"{maximum_ground_penetration_m:.6f} m "
            f"(limit {float(penetration_limit_raw):.6f} m)."
        )
    if (
        acceptance.get("require_gravity_response", True)
        and not gravity_response_observed
    ):
        failures.append("Simulation did not exhibit the expected gravity response.")
    expected_body_count = acceptance.get("expected_body_count")
    if expected_body_count is not None and loaded_body_count != int(
        expected_body_count
    ):
        failures.append(
            "Simulation loaded body count did not match the expectation: "
            f"expected {int(expected_body_count)}, got {loaded_body_count}."
        )

    return (
        {
            "loaded_body_count": loaded_body_count,
            "first_step_displacement_m": first_step_displacement_m,
            "expected_ballistic_displacement_m": expected_ballistic_displacement_m,
            "gravity_magnitude_m_per_s2": gravity_magnitude,
            "initial_pose_discontinuity": initial_pose_discontinuity,
            "maximum_ground_penetration_m": maximum_ground_penetration_m,
            "resting_ground_penetration_m": resting_ground_penetration_m,
            "ground_clearance_geometry": ground_clearance_geometry,
            "gravity_response_observed": gravity_response_observed,
        },
        failures,
    )


_RUNTIME_ACCEPTANCE_KEY_ALIASES = {
    "discontinuity": "detect_initial_pose_discontinuity",
    "detect_discontinuity": "detect_initial_pose_discontinuity",
}


def _filtered_geometry_points(geometry_points_raw: Any) -> list[list[float]]:
    """Return the usable exact support points from a scene-info value.

    This single filter decides both the measurement geometry in
    _runtime_acceptance_metrics and the exactness flag in
    _default_ground_penetration_limit_m, so the two can never diverge on a
    malformed list.
    """
    geometry_points: list[list[float]] = []
    if isinstance(geometry_points_raw, list):
        for raw_point in geometry_points_raw:
            if not isinstance(raw_point, list | tuple) or len(raw_point) < 3:
                continue
            try:
                point = [float(raw_point[index]) for index in range(3)]
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in point):
                geometry_points.append(point)
    return geometry_points


def _conservative_penetration_warning(
    support_decision: dict[str, Any] | None = None,
) -> str:
    reason_code = (
        str(support_decision.get("reason_code") or "")
        if isinstance(support_decision, dict)
        else ""
    )
    if reason_code in {
        "raw_point_estimate_exceeded",
        "unique_point_bound_exceeded",
    }:
        count = (
            support_decision.get("observed_unique_point_count")
            if reason_code == "unique_point_bound_exceeded"
            else support_decision.get("raw_point_count_estimate")
        )
        bound = support_decision.get("processing_bound")
        return (
            "Ground penetration was measured from conservative bbox corners "
            f"because validation support geometry exceeded its processing "
            f"bound (estimated points {count}, bound {bound}). The authored "
            "collider was not changed. If runtime and visual validation pass, "
            "this support fallback is accepted and is not a reason to revise "
            "the collider."
        )
    return (
        "Ground penetration was measured from bbox corners because exact "
        "collider support geometry was unavailable (analytic shape "
        "colliders, a boundingSphere collision approximation, or "
        "convex-hull reduction was unavailable or exceeded its point "
        "limits). This measurement can mis-report: a rotated resting pose "
        "over-reports for mesh or box shapes, and a bounding sphere dips "
        "below the bbox corners between corner directions, so real "
        "penetration can be under-reported. The authored collider was not "
        "changed; revise it only in response to a specific validation failure, "
        "or pass an explicit max_ground_penetration_m to size the gate "
        "deliberately."
    )


def _default_ground_penetration_limit_m(
    scene_info: dict[str, Any],
) -> tuple[float, bool]:
    """Return the default runtime penetration limit and measurement exactness.

    With exact support geometry (collider mesh vertices, hull-reduced when
    large) a reported penetration is real solver softness — measured
    0 - 2.45% of the bbox diagonal across the 16-asset PhysX-Mobility suite
    and bit-reproducible across repeat runs — so the default scales at 2.5%
    of the diagonal. The scaled limit is capped at half the smallest bbox
    extent, so a long thin body cannot legally rest with its whole
    thickness below ground, and at the 1.0 m ceiling the request model
    enforces for explicit inputs. The 0.005 m floor applies after the
    extent cap: the enforced limit never drops below 5 mm, even for
    geometry thinner than 10 mm.

    Without exact support geometry the only available measurement is bbox
    corners, whose rotation artifact reaches (sqrt(3)-1)/(2*sqrt(3)) ~= 21%
    of the diagonal on a sphere at rest — a scaled limit loose enough to
    absorb that cannot catch real sinking. The default then stays at the
    absolute 0.005 m floor, matching the pre-scale-relative behavior for
    analytic-shape colliders, and the caller records a warning that the
    measurement is conservative.
    """
    # Exactness must be decided from the same filtered view of the point list
    # that _runtime_acceptance_metrics measures with: a list whose entries all
    # fail the shape/finiteness filter would otherwise be declared exact here
    # (loose scale-relative limit, no warning) while the measurement silently
    # falls back to conservative bbox corners.
    exact_support_geometry = bool(
        _filtered_geometry_points(scene_info.get("geometry_points_local_stage"))
    )
    bbox_min = scene_info.get("bbox_min_local_stage")
    bbox_max = scene_info.get("bbox_max_local_stage")
    if not exact_support_geometry or not (bbox_min and bbox_max):
        return 0.005, exact_support_geometry
    meters_per_unit = _meters_per_stage_unit(scene_info)
    # Pose-local bounds exclude the body's authored scale (it travels
    # separately so recorded rigid poses aren't double-applied); fold it
    # back in here or a scaled asset gets a limit sized to its unscaled
    # mesh. Other bound spaces already include the scale —
    # _runtime_acceptance_metrics applies the same space-conditional rule
    # to the measurement, so limit and measurement stay in one space.
    scale = [1.0, 1.0, 1.0]
    if scene_info.get("bbox_local_stage_space") == "pose_local":
        scale_raw = scene_info.get("bbox_local_stage_scale") or [1.0, 1.0, 1.0]
        try:
            scale = [float(scale_raw[index]) for index in range(3)]
        except (IndexError, TypeError, ValueError):
            scale = [1.0, 1.0, 1.0]
        # Mirrored assets carry negative scale components; the measurement
        # in _runtime_acceptance_metrics applies them signed, and extent
        # magnitudes are mirror-invariant, so a negative component must not
        # force the neutral fallback (that would size the limit for the
        # unscaled mesh while the measurement uses the real scale). Only
        # zero/non-finite components are degenerate.
        if not all(math.isfinite(value) and value != 0.0 for value in scale):
            scale = [1.0, 1.0, 1.0]
    extents_m = [
        abs(float(bbox_max[index]) - float(bbox_min[index]))
        * abs(scale[index])
        * meters_per_unit
        for index in range(3)
    ]
    diagonal_m = math.sqrt(sum(extent**2 for extent in extents_m))
    limit = max(0.005, min(0.025 * diagonal_m, 0.5 * min(extents_m)))
    return min(limit, 1.0), True


def _normalize_runtime_acceptance(acceptance: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(acceptance)
    for alias, canonical in _RUNTIME_ACCEPTANCE_KEY_ALIASES.items():
        if alias in normalized:
            normalized.setdefault(canonical, normalized[alias])
            normalized.pop(alias, None)
    return normalized


def validate_runtime(
    *,
    physics_usd: Path | str,
    output_dir: Path | str,
    engine: PhysicsSimulationEngine = "ovphysx",
    duration_s: float = 3.0,
    dt: float = 1.0 / 240.0,
    sample_fps: int = 30,
    drop_height_m: float | None = None,
    acceptance: dict[str, Any] | None = None,
    body_prim_path_hint: str | None = None,
    body_pattern_hint: str | None = None,
    placement_prim_path_hint: str | None = None,
    approved_dependency_roots: Iterable[Path | str] | None = None,
    relax_ovphysx_address_space_limit: bool = False,
    usd_cli_session: WorkflowUsdCliSession | None = None,
    scene_tool_timeout_seconds: float = 1800.0,
    ground_clearance_support_cache: dict[str, dict[str, Any]] | None = None,
    ground_clearance_support_cache_key: str | None = None,
) -> dict[str, Any]:
    """Run simulation-backed validation and write runtime evidence artifacts.

    ``approved_dependency_roots`` bounds which filesystem dependencies may be
    copied into the portable runtime-scene sidecar. Direct callers default to
    the authored Physics USD's parent. usd-cli service callers should also
    pass the trusted source-session input root when validating a generated
    derivative whose asset paths still resolve into that input tree.
    """

    if engine not in VALID_SIMULATION_ENGINES:
        raise ValueError(
            f"engine must be one of {sorted(VALID_SIMULATION_ENGINES)}, got {engine!r}"
        )

    acceptance_config = None
    normalized_acceptance: dict[str, Any] = {}
    if acceptance is not None:
        normalized_acceptance = _normalize_runtime_acceptance(acceptance)
        if normalized_acceptance.get("require_settle", True) is not True:
            raise ValueError(
                "acceptance.require_settle cannot be disabled for SimReady validation"
            )
        acceptance_config = {
            "detect_initial_pose_discontinuity": True,
            "max_initial_pose_displacement_m": None,
            "ballistic_displacement_multiplier": 3.0,
            "initial_pose_tolerance_m": 0.002,
            "max_ground_penetration_m": 0.005,
            "require_gravity_response": True,
            "require_settle": True,
            "expected_body_count": None,
            **normalized_acceptance,
        }
        if (
            engine == "fake"
            and "detect_initial_pose_discontinuity" not in normalized_acceptance
        ):
            acceptance_config["detect_initial_pose_discontinuity"] = False
        if engine == "fake" and "max_ground_penetration_m" not in normalized_acceptance:
            acceptance_config["max_ground_penetration_m"] = None
    physics_usd_path = Path(physics_usd).resolve()
    validation_dir = Path(output_dir).resolve()

    if engine == "none":
        validation_dir.mkdir(parents=True, exist_ok=True)
        report_path = _write_json(
            validation_dir / "runtime_validation_report.json",
            {
                "engine": engine,
                "physics_usd": str(physics_usd_path),
                "failures": [],
                "warnings": ["Runtime simulation was disabled."],
                "not_evaluated": True,
            },
        )
        return {
            "engine": engine,
            "physics_usd": str(physics_usd_path),
            "runtime_report": str(report_path),
            "failures": [],
            "warnings": ["Runtime simulation was disabled."],
            "not_evaluated": True,
            "evidence_artifacts": [
                {
                    "kind": "runtime_report",
                    "path": str(report_path),
                    "description": "Runtime validation disabled report.",
                }
            ],
        }

    from world_understanding.functions.graphics.so_export import (
        _normalize_dependency_roots,
    )

    dependency_roots = _normalize_dependency_roots(
        tuple(approved_dependency_roots)
        if approved_dependency_roots is not None
        else (physics_usd_path.parent,)
    )

    from physics_agent.recording import (
        author_trajectory_jsonl,
        author_trajectory_usda,
    )
    from physics_agent.tuning.scenarios._scene_builder import (
        build_drop_settle_scene,
    )
    from world_understanding.functions.physics.trajectory import (
        settle_distance,
        trajectory_summary,
    )

    scene_path = validation_dir / "drop_settle_scene.usda"
    scene_build_started = time.perf_counter()
    scene_info = build_drop_settle_scene(
        physics_usd_path,
        scene_path,
        drop_height_m=drop_height_m,
        gravity=-9.81,
        ground_friction=0.6,
        cameras=["+x+y+z"],
        body_prim_path_hint=body_prim_path_hint,
        body_pattern_hint=body_pattern_hint,
        placement_prim_path_hint=placement_prim_path_hint,
        approved_dependency_roots=dependency_roots,
        ground_clearance_support_cache=ground_clearance_support_cache,
        ground_clearance_support_cache_key=ground_clearance_support_cache_key,
    )
    scene_build_wall_seconds = time.perf_counter() - scene_build_started
    raw_scene_build_seconds = scene_info.get("scene_build_duration_seconds")
    scene_build_seconds = (
        float(raw_scene_build_seconds)
        if isinstance(raw_scene_build_seconds, int | float)
        and not isinstance(raw_scene_build_seconds, bool)
        and math.isfinite(float(raw_scene_build_seconds))
        and float(raw_scene_build_seconds) >= 0.0
        else scene_build_wall_seconds
    )
    support_decision = scene_info.get("ground_clearance_support_decision")
    support_selection_seconds = (
        float(support_decision.get("selection_duration_seconds"))
        if isinstance(support_decision, dict)
        and isinstance(support_decision.get("selection_duration_seconds"), int | float)
        else None
    )
    resolved_drop_height = float(
        drop_height_m
        if drop_height_m is not None
        else scene_info.get("drop_height_m_resolved", 0.05)
    )
    if (
        acceptance_config is not None
        and resolved_drop_height <= 0.0
        and "require_gravity_response" not in normalized_acceptance
    ):
        acceptance_config["require_gravity_response"] = False
    # When the caller did not explicitly set a penetration limit — and the
    # engine-specific bypass above did not already disable the numeric gate —
    # replace the 5 mm hard default with a scale-relative threshold where the
    # measurement is exact, so solver contact softness on large assets does
    # not produce a false failure. Writing the computed value back into
    # acceptance_config means the returned report reflects the enforced
    # limit, so the child agent can interpret the measurement correctly.
    # Without exact support geometry the absolute 5 mm default and the
    # conservative bbox-corner measurement stay in force, recorded as a
    # non-blocking diagnostic rather than a failure or warning:
    # analytic-shape colliders that passed before keep passing, and callers
    # can still set an explicit limit.
    conservative_penetration_measurement = False
    if acceptance_config is not None:
        default_limit, _exact_support_geometry = _default_ground_penetration_limit_m(
            scene_info
        )
        if (
            "max_ground_penetration_m" not in normalized_acceptance
            and acceptance_config["max_ground_penetration_m"] is not None
        ):
            acceptance_config["max_ground_penetration_m"] = default_limit
        # Measurement exactness is a property of the scene, not of how the
        # limit was chosen: an explicitly-configured limit is still compared
        # against a rotation-sensitive bbox-corner measurement when exact
        # support geometry is missing, and the report has to say so.
        conservative_penetration_measurement = (
            acceptance_config["max_ground_penetration_m"] is not None
            and not _exact_support_geometry
            and bool(scene_info.get("bbox_min_local_stage"))
            and bool(scene_info.get("bbox_max_local_stage"))
        )
    response: dict[str, Any]
    trajectory: list[tuple[float, list[float], list[float]]]
    usd_cli_simulation: dict[str, Any] | None = None
    simulation_started = time.perf_counter()
    if engine == "fake":
        trajectory = _fake_trajectory(
            rest_position=list(scene_info["rest_position"]),
            world_up=list(scene_info.get("world_up") or [0.0, 0.0, 1.0]),
            duration_s=duration_s,
            sample_fps=sample_fps,
            drop_height_m=resolved_drop_height,
        )
        response = {
            "status": "ok",
            "engine": "fake",
            "trajectory": trajectory,
            "final_pose": trajectory[-1][1] if trajectory else None,
            "final_velocity": trajectory[-1][2] if trajectory else None,
            "n_bodies": 1,
            "duration_s": duration_s,
        }
    elif engine == "newton":
        # NVIDIA Newton (Warp + MuJoCo) runs in-process — no daemon subprocess,
        # so it sidesteps the ovphysx daemon-lifecycle failures seen inside the
        # long-running usd-cli server, and its stiffer MuJoCo contact model
        # keeps drop-impact ground penetration far smaller than ovphysx's soft
        # PhysX contact. MuJoCo ignores UsdPhysics restitution/static_friction,
        # which is fine for runtime validation (finite/bounded, penetration,
        # gravity response, settle — none require restitution).
        from physics_agent.tuning.newton_simulator import NewtonSimulator

        response = NewtonSimulator().evaluate(
            scene_usd=scene_path,
            body_pattern=str(scene_info["body_pattern"]),
            duration_s=duration_s,
            dt=dt,
            sample_fps=sample_fps,
        )
        trajectory = [
            (float(t), [float(v) for v in pose], [float(v) for v in vel])
            for t, pose, vel in response.get("trajectory", [])
        ]
    else:
        from content_agent_workflows.physics.usd_cli_ops import (
            simulate_physics_scene,
        )

        del relax_ovphysx_address_space_limit
        usd_cli_simulation = simulate_physics_scene(
            scene_usd=scene_path,
            output_dir=validation_dir / "usd_cli_simulation",
            body_path=str(scene_info["body_prim_path"]),
            body_pattern=str(scene_info["body_pattern"]),
            rest_position=[float(value) for value in scene_info["rest_position"]],
            world_up=[
                float(value) for value in scene_info.get("world_up") or [0.0, 0.0, 1.0]
            ],
            duration_s=duration_s,
            dt=dt,
            sample_fps=sample_fps,
            usd_cli_session=usd_cli_session,
            timeout_seconds=scene_tool_timeout_seconds,
        )
        trajectory_path = Path(str(usd_cli_simulation.get("trajectory_jsonl") or ""))
        if not trajectory_path.is_file():
            raise RuntimeError(
                "usd-cli physics simulate returned no readable trajectory artifact"
            )
        trajectory = _read_usd_cli_trajectory(trajectory_path)
        facts = usd_cli_simulation.get("simulation_facts")
        response = {
            "status": "ok",
            "engine": "ovphysx",
            "trajectory": trajectory,
            "n_bodies": usd_cli_simulation.get("n_bodies"),
            "n_steps": facts.get("reported_step_count")
            if isinstance(facts, dict)
            else None,
            "usd_cli_report_path": usd_cli_simulation.get("report_path"),
        }
    simulation_seconds = time.perf_counter() - simulation_started
    acceptance_started = time.perf_counter()

    response_path = _write_trajectory_response(
        validation_dir / "simulation_response.json",
        response,
    )
    finite = all(
        math.isfinite(float(v)) for _t, pose, vel in trajectory for v in [*pose, *vel]
    )
    if not trajectory or not finite:
        trajectory_jsonl = validation_dir / "trajectory.jsonl"
        trajectory_jsonl.write_text("", encoding="utf-8")
        early_failures: list[str] = []
        if not trajectory:
            early_failures.append("Simulation produced no trajectory samples.")
        if not finite:
            early_failures.append("Simulation trajectory contained non-finite values.")
        if isinstance(support_decision, dict):
            support_decision["validation_outcome"] = "fail"
            support_decision["fallback_accepted"] = False
        phase_timings_seconds = {
            "support_selection": support_selection_seconds,
            "scene_build": scene_build_seconds,
            "simulation": simulation_seconds,
            "acceptance": time.perf_counter() - acceptance_started,
        }
        report = {
            "engine": engine,
            "physics_usd": str(physics_usd_path),
            "scene_usd": str(scene_path),
            "trajectory_jsonl": str(trajectory_jsonl),
            "recording_usda": None,
            "response_path": str(response_path),
            "scene_info": scene_info,
            "summary": {},
            "settle_distance": None,
            "max_abs_position": 0.0,
            "failures": early_failures,
            "warnings": [],
            "acceptance": acceptance_config,
            "phase_timings_seconds": phase_timings_seconds,
        }
        report_path = _write_json(
            validation_dir / "runtime_validation_report.json",
            report,
        )
        return {
            "engine": engine,
            "physics_usd": str(physics_usd_path),
            "scene_usd": str(scene_path),
            "trajectory_jsonl": str(trajectory_jsonl),
            "recording_usda": None,
            "response_path": str(response_path),
            "runtime_report": str(report_path),
            "scene_info": scene_info,
            "summary": {},
            "settle_distance": None,
            "max_abs_position": 0.0,
            "failures": early_failures,
            "warnings": [],
            "acceptance": acceptance_config,
            "phase_timings_seconds": phase_timings_seconds,
            "evidence_artifacts": [
                {
                    "kind": "simulation_scene",
                    "path": str(scene_path),
                    "description": "Drop-settle validation scene.",
                },
                {
                    "kind": "trajectory_jsonl",
                    "path": str(trajectory_jsonl),
                    "description": "Empty trajectory output from failed simulation.",
                },
                {
                    "kind": "runtime_report",
                    "path": str(report_path),
                    "description": "Runtime validation failure metrics and artifact index.",
                },
            ],
        }
    if usd_cli_simulation is None:
        trajectory_jsonl = author_trajectory_jsonl(
            trajectory,
            validation_dir / "trajectory.jsonl",
            fps=sample_fps,
            max_duration_s=duration_s,
        )
        recording_usda = author_trajectory_usda(
            scene_path,
            trajectory,
            str(scene_info["body_prim_path"]),
            validation_dir / "recording.usda",
            fps=sample_fps,
            max_duration_s=duration_s,
        )
    else:
        trajectory_jsonl = Path(str(usd_cli_simulation["trajectory_jsonl"]))
        recording_value = usd_cli_simulation.get("recording_usda")
        if not isinstance(recording_value, str) or not Path(recording_value).is_file():
            raise RuntimeError(
                "usd-cli physics simulate returned no readable recording artifact"
            )
        recording_usda = Path(recording_value)

    world_up = scene_info.get("world_up") or [0.0, 0.0, 1.0]
    summary = trajectory_summary(trajectory, world_up=world_up)
    distance = settle_distance(trajectory, rest_position=scene_info["rest_position"])
    max_abs_position = max(
        (abs(float(v)) for _t, pose, _vel in trajectory for v in pose[:3]),
        default=0.0,
    )
    failures: list[str] = []
    warnings: list[str] = []
    # Diagnostics are informational notes that must NOT ride the warnings
    # channel: a non-empty warnings list downgrades sim_ready_status to
    # "conditional" downstream (validation_evidence.py), and the
    # asset-composition physics handoff hard-rejects anything that is not
    # "pass" — a purely informational note would block a passing asset.
    diagnostics: list[str] = []
    if max_abs_position > 100.0:
        failures.append(
            f"Simulation trajectory moved out of bounded range: {max_abs_position:.3f}."
        )
    if int(response.get("n_bodies") or 0) < 1:
        failures.append("Simulation did not load any rigid bodies.")
    if acceptance_config is not None:
        acceptance_metrics, acceptance_failures = _runtime_acceptance_metrics(
            trajectory=trajectory,
            scene_info=scene_info,
            loaded_body_count=int(response.get("n_bodies") or 0),
            acceptance=acceptance_config,
        )
        summary.update(acceptance_metrics)
        failures.extend(acceptance_failures)
    if conservative_penetration_measurement:
        diagnostics.append(_conservative_penetration_warning(support_decision))
    # The scale-relative default is sized on trajectory maxima (impact
    # transients included), so a settled pose can legally sit well below
    # the old absolute floor. Surface that in the report instead of leaving
    # it as a silent summary field: a deep rest is a visible artifact even
    # when the gate passes, and these notes are the calibration trail for a
    # future rest-specific factor. Only the defaulted limit is second-
    # guessed — a caller that explicitly authorized a deeper rest with
    # max_ground_penetration_m made that call deliberately.
    _resting_penetration = summary.get("resting_ground_penetration_m")
    _enforced_limit = (
        acceptance_config.get("max_ground_penetration_m")
        if acceptance_config is not None
        else None
    )
    _limit_was_defaulted = (
        acceptance_config is not None
        and "max_ground_penetration_m" not in normalized_acceptance
    )
    if (
        _limit_was_defaulted
        and _resting_penetration is not None
        and _enforced_limit is not None
        and float(_resting_penetration) > 0.005
    ):
        diagnostics.append(
            f"Body rests {float(_resting_penetration):.6f} m below ground "
            f"(enforced limit {float(_enforced_limit):.6f} m). The default "
            "limit is sized on trajectory maxima; a rest pose this deep is "
            "a visible artifact — recorded for rest-gate calibration."
        )
    if summary.get("settle_time_s") is None:
        warnings.append(
            "Body did not reach the default settle threshold during the validation window."
        )

    if isinstance(support_decision, dict):
        support_decision["validation_outcome"] = "fail" if failures else "pass"
        support_decision["fallback_accepted"] = bool(
            not failures
            and support_decision.get("selected_support_type")
            == "conservative_bbox_corners"
        )
    phase_timings_seconds = {
        "support_selection": support_selection_seconds,
        "scene_build": scene_build_seconds,
        "simulation": simulation_seconds,
        "acceptance": time.perf_counter() - acceptance_started,
    }
    report = {
        "engine": engine,
        "physics_usd": str(physics_usd_path),
        "scene_usd": str(scene_path),
        "trajectory_jsonl": str(trajectory_jsonl),
        "recording_usda": str(recording_usda),
        "response_path": str(response_path),
        "scene_info": scene_info,
        "summary": summary,
        "settle_distance": distance,
        "max_abs_position": max_abs_position,
        "failures": failures,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "acceptance": acceptance_config,
        "phase_timings_seconds": phase_timings_seconds,
    }
    report_path = _write_json(
        validation_dir / "runtime_validation_report.json",
        report,
    )

    return {
        "engine": engine,
        "physics_usd": str(physics_usd_path),
        "scene_usd": str(scene_path),
        "trajectory_jsonl": str(trajectory_jsonl),
        "recording_usda": str(recording_usda),
        "response_path": str(response_path),
        "runtime_report": str(report_path),
        "scene_info": scene_info,
        "summary": summary,
        "settle_distance": distance,
        "max_abs_position": max_abs_position,
        "failures": failures,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "acceptance": acceptance_config,
        "phase_timings_seconds": phase_timings_seconds,
        "evidence_artifacts": [
            {
                "kind": "simulation_scene",
                "path": str(scene_path),
                "description": "Drop-settle validation scene.",
            },
            {
                "kind": "trajectory_jsonl",
                "path": str(trajectory_jsonl),
                "description": "Per-frame simulated pose and velocity.",
            },
            {
                "kind": "recording_usda",
                "path": str(recording_usda),
                "description": "Time-sampled USD recording of the validation run.",
            },
            {
                "kind": "runtime_report",
                "path": str(report_path),
                "description": "Runtime validation metrics and artifact index.",
            },
        ],
    }
