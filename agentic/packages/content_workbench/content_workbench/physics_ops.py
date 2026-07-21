# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workbench-owned physics operations used by agentic physics workflows.

Keep agent policy out of this module: it should inspect, author, simulate, and
report, but not decide what the physics properties ought to be.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

PhysicsSimulationEngine = Literal["ovphysx", "fake", "none"]
VALID_SIMULATION_ENGINES = {"ovphysx", "fake", "none"}


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
    traversal = Usd.PrimRange(root_prim) if root_prim else stage.Traverse()
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
) -> dict[str, Any]:
    """Apply and audit an explicit topology plan on a derivative USD."""

    from world_understanding.functions.physics.physics_topology import (
        apply_physics_topology_plan,
    )

    report = apply_physics_topology_plan(
        input_usd_path=input_usd_path,
        output_usd_path=output_usd_path,
        expected_source_digest=expected_source_digest,
        mobility_intent=mobility_intent,
        operations=operations,
        invariants=invariants,
    )
    report_path = (
        Path(output_usd_path)
        .resolve()
        .with_name(f"{Path(output_usd_path).stem}_topology_report.json")
    )
    report["topology_report"] = str(report_path)
    _write_json(report_path, report)
    return report


def inspect_authored_physics(physics_usd: Path | str) -> dict[str, Any]:
    """Summarize authored USD physics schemas."""

    from pxr import Usd, UsdPhysics

    physics_usd_path = Path(physics_usd).resolve()
    stage = Usd.Stage.Open(str(physics_usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open authored physics USD: {physics_usd_path}")

    default_prim = stage.GetDefaultPrim()
    default_path = str(default_prim.GetPath()) if default_prim else None
    rigid_body_paths: list[str] = []
    enabled_rigid_body_paths: list[str] = []
    disabled_rigid_body_paths: list[str] = []
    collision_paths: list[str] = []
    physics_material_paths: list[str] = []
    scene_paths: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.IsA(UsdPhysics.Scene):
            scene_paths.append(path)
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
        "physics_scene_paths": scene_paths,
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
            decision.get("collider_paths")
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
        for prim_path in prim_paths:
            if not isinstance(prim_path, str) or not prim_path:
                raise ValueError(
                    f"physics decision at index {index} contains an invalid prim path"
                )
            records.append(
                {
                    "id": prim_path,
                    "classification": {
                        "decision_id": decision.get("decision_id"),
                        "component_id": decision.get("component_id"),
                        "component": decision.get("component_label")
                        or decision.get("component_id"),
                        "material": decision.get("inferred_material_family"),
                        "physical_properties": per_prim_properties,
                        "collision_mode": decision.get("collision_mode"),
                        "collision_approximation": decision.get(
                            "collision_approximation"
                        ),
                        "mass_authoring_path": decision.get("mass_authoring_path"),
                        "component_estimated_mass_kg": estimated_mass,
                        "confidence": decision.get("confidence"),
                        "reasoning": decision.get("rationale"),
                    },
                    "source": "content_workbench.physics_ops.decision_patch",
                }
            )
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
        targets = set(raw_targets)
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


def apply_schema(
    *,
    usd_path: Path | str,
    decision_patch_path: Path | str | None = None,
    predictions_jsonl_path: Path | str,
    output_usd_path: Path | str,
    collision_approximation: str = "convexHull",
    output_key: str = "classification",
    author_rigid_body: bool = True,
) -> dict[str, Any]:
    """Author USD physics schema from accepted physics predictions."""

    from physics_agent.functions.apply_physics import apply_physics

    output_path = Path(output_usd_path).resolve()
    resolved_predictions_path = Path(predictions_jsonl_path).resolve()
    if decision_patch_path is not None:
        _validate_v2_decision_patch(usd_path, decision_patch_path)
        resolved_predictions_path = _write_predictions_jsonl(
            output_path.with_name(f"{output_path.stem}_predictions.jsonl"),
            _prediction_records_from_decision_patch(decision_patch_path),
        )

    authored = Path(
        apply_physics(
            str(Path(usd_path).resolve()),
            str(resolved_predictions_path),
            str(output_path),
            collision_approx=collision_approximation,
            output_key=output_key,
            author_rigid_body=author_rigid_body,
        )
    ).resolve()
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
    bbox_min = scene_info.get("bbox_min_local_stage")
    bbox_max = scene_info.get("bbox_max_local_stage")
    if trajectory and bbox_min and bbox_max:
        rotate_bounds = scene_info.get("bbox_local_stage_space") == "pose_local"
        bbox_local_scale = scene_info.get("bbox_local_stage_scale") or [1.0, 1.0, 1.0]
        try:
            scale = [float(bbox_local_scale[index]) for index in range(3)]
        except (IndexError, TypeError, ValueError):
            scale = [1.0, 1.0, 1.0]
        if not all(math.isfinite(value) for value in scale):
            scale = [1.0, 1.0, 1.0]
        maximum_ground_penetration_m = 0.0
        for _time, pose, _velocity in trajectory:
            projections: list[float] = []
            for x in (float(bbox_min[0]), float(bbox_max[0])):
                for y in (float(bbox_min[1]), float(bbox_max[1])):
                    for z in (float(bbox_min[2]), float(bbox_max[2])):
                        local_corner = [x, y, z]
                        if rotate_bounds:
                            local_corner = [
                                local_corner[index] * scale[index] for index in range(3)
                            ]
                        if rotate_bounds:
                            offset = _rotate_vector_by_quaternion(
                                local_corner, [float(value) for value in pose[3:7]]
                            )
                        else:
                            offset = local_corner
                        world_point = [
                            offset[index] + float(pose[index]) for index in range(3)
                        ]
                        projections.append(
                            sum(
                                world_point[index] * world_up[index]
                                for index in range(3)
                            )
                        )
            maximum_ground_penetration_m = max(
                maximum_ground_penetration_m,
                max(0.0, -min(projections) * meters_per_unit),
            )

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
            "gravity_response_observed": gravity_response_observed,
        },
        failures,
    )


_RUNTIME_ACCEPTANCE_KEY_ALIASES = {
    "discontinuity": "detect_initial_pose_discontinuity",
    "detect_discontinuity": "detect_initial_pose_discontinuity",
}


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
    duration_s: float = 1.0,
    dt: float = 1.0 / 240.0,
    sample_fps: int = 30,
    drop_height_m: float | None = None,
    acceptance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run simulation-backed validation and write runtime evidence artifacts."""

    if engine not in VALID_SIMULATION_ENGINES:
        raise ValueError(
            f"engine must be one of {sorted(VALID_SIMULATION_ENGINES)}, got {engine!r}"
        )

    acceptance_config = None
    normalized_acceptance: dict[str, Any] = {}
    if acceptance is not None:
        normalized_acceptance = _normalize_runtime_acceptance(acceptance)
        acceptance_config = {
            "detect_initial_pose_discontinuity": True,
            "max_initial_pose_displacement_m": None,
            "ballistic_displacement_multiplier": 3.0,
            "initial_pose_tolerance_m": 0.002,
            "max_ground_penetration_m": 0.005,
            "require_gravity_response": True,
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
    validation_dir.mkdir(parents=True, exist_ok=True)

    if engine == "none":
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
    scene_info = build_drop_settle_scene(
        physics_usd_path,
        scene_path,
        drop_height_m=drop_height_m,
        gravity=-9.81,
        ground_friction=0.6,
        cameras=["+x+y+z"],
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
    response: dict[str, Any]
    trajectory: list[tuple[float, list[float], list[float]]]
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
    else:
        from world_understanding.functions.physics.ovphysx_daemon import (
            _OvPhysXDaemon,
        )

        daemon = _OvPhysXDaemon()
        try:
            response = daemon.evaluate(
                scene_usd=scene_path,
                body_pattern=str(scene_info["body_pattern"]),
                duration_s=duration_s,
                dt=dt,
                sample_fps=sample_fps,
            )
        finally:
            daemon.shutdown()
        trajectory = [
            (float(t), [float(v) for v in pose], [float(v) for v in vel])
            for t, pose, vel in response.get("trajectory", [])
        ]

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

    world_up = scene_info.get("world_up") or [0.0, 0.0, 1.0]
    summary = trajectory_summary(trajectory, world_up=world_up)
    distance = settle_distance(trajectory, rest_position=scene_info["rest_position"])
    max_abs_position = max(
        (abs(float(v)) for _t, pose, _vel in trajectory for v in pose[:3]),
        default=0.0,
    )
    failures: list[str] = []
    warnings: list[str] = []
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
    if summary.get("settle_time_s") is None:
        warnings.append(
            "Body did not reach the default settle threshold during the validation window."
        )

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
        "acceptance": acceptance_config,
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
        "acceptance": acceptance_config,
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
