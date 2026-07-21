# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic USD physics component inspection and constrained topology repair.

USD imports stay inside public functions so importing the physics package does not
load usd-core into processes that may later start the isolated ovphysx runtime.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

TOPOLOGY_PLAN_SCHEMA = "content-workflows.physics-topology-plan.v1"
SUPPORTED_TOPOLOGY_OPERATIONS = {
    "ensure_rigid_body_api",
    "remove_rigid_body_api",
    "remove_fixed_joint",
}
SUPPORTED_MOBILITY_INTENTS = {"preserve", "movable", "static"}
_UNOWNED_STATIC_GROUP_PREFIX = "__unowned_static__:"
_USD_LAYER_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}


class PhysicsTopologyPlanError(ValueError):
    """Raised when a topology plan is unsafe, stale, or unsupported."""


def sha256_file(path: Path | str) -> str:
    """Return a prefixed SHA-256 digest for a file or composed USD stage."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Input USD not found: {resolved}")
    if resolved.suffix.lower() in _USD_LAYER_EXTENSIONS:
        from pxr import Usd

        stage = Usd.Stage.Open(str(resolved))
        assert stage is not None
        digest = hashlib.sha256()
        digest.update(b"content-workflows-usd-stage-dependency-digest-v1\0")
        for layer in sorted(stage.GetUsedLayers(), key=lambda layer: layer.identifier):
            if layer.anonymous:
                continue
            identifier = str(layer.resolvedPath or layer.realPath or layer.identifier)
            digest.update(identifier.encode("utf-8"))
            digest.update(b"\0")
            layer_path = Path(identifier)
            payload = (
                layer_path.read_bytes()
                if layer_path.is_file()
                else layer.ExportToString().encode("utf-8", errors="replace")
            )
            digest.update(payload)
            digest.update(b"\0")
        return f"sha256:{digest.hexdigest()}"

    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _api_enabled(attr: Any) -> bool:
    value = attr.Get() if attr else None
    return value is not False


def _enabled_rigid_body(prim: Any, UsdPhysics: Any) -> bool:
    return prim.HasAPI(UsdPhysics.RigidBodyAPI) and _api_enabled(
        UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr()
    )


def _enabled_collider(prim: Any, UsdPhysics: Any) -> bool:
    return prim.HasAPI(UsdPhysics.CollisionAPI) and _api_enabled(
        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
    )


def _nearest_path(path: Any, candidates: set[str]) -> str | None:
    current = path
    while current and not current.IsAbsoluteRootPath():
        text = str(current)
        if text in candidates:
            return text
        current = current.GetParentPath()
    return None


def _ancestor_enabled_rigid_body_paths(
    stage: Any,
    root_prim_path: str | None,
    UsdPhysics: Any,
) -> set[str]:
    if not root_prim_path:
        return set()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        return set()
    paths: set[str] = set()
    parent = root.GetParent()
    while parent and parent.IsValid() and not parent.IsPseudoRoot():
        if _enabled_rigid_body(parent, UsdPhysics):
            paths.add(str(parent.GetPath()))
        parent = parent.GetParent()
    return paths


def _relationship_targets(joint: Any) -> tuple[list[str], list[str]]:
    return (
        [str(path) for path in joint.GetBody0Rel().GetTargets()],
        [str(path) for path in joint.GetBody1Rel().GetTargets()],
    )


def _path_in_scope(path: str, root_prim_path: str | None) -> bool:
    if not root_prim_path:
        return True
    root = root_prim_path.rstrip("/") or "/"
    if root == "/":
        return path.startswith("/")
    return path == root or path.startswith(f"{root}/")


def _path_is_or_under(path: str, root: str) -> bool:
    root = root.rstrip("/") or "/"
    if root == "/":
        return path.startswith("/")
    return path == root or path.startswith(f"{root}/")


def _path_overlaps_any_root(path: str, roots: list[str]) -> bool:
    return any(
        _path_is_or_under(path, root) or _path_is_or_under(root, path) for root in roots
    )


def _non_fixed_joint_endpoint_paths(topology: dict[str, Any]) -> list[str]:
    endpoints: set[str] = set()
    for joint in topology.get("joints", []):
        if joint.get("enabled") is False or joint.get("is_fixed_joint"):
            continue
        endpoints.update(joint.get("body0_rigid_body_paths") or [])
        endpoints.update(joint.get("body1_rigid_body_paths") or [])
    return sorted(endpoints)


def _non_fixed_joint_endpoint_signature(
    topology: dict[str, Any],
) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    records: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for joint in topology.get("joints", []):
        if joint.get("enabled") is False or joint.get("is_fixed_joint"):
            continue
        records.append(
            (
                str(joint.get("prim_path") or ""),
                tuple(sorted(joint.get("body0_rigid_body_paths") or [])),
                tuple(sorted(joint.get("body1_rigid_body_paths") or [])),
            )
        )
    return tuple(sorted(records))


def _unowned_static_group_key(prim_path: str) -> str:
    return f"{_UNOWNED_STATIC_GROUP_PREFIX}{prim_path}"


def _is_unowned_static_group(group_key: str) -> bool:
    return group_key.startswith(_UNOWNED_STATIC_GROUP_PREFIX)


def _unowned_static_body_root(group_key: str) -> str:
    return group_key.removeprefix(_UNOWNED_STATIC_GROUP_PREFIX)


def inspect_physics_topology(
    usd_path: Path | str,
    *,
    root_prim_path: str | None = None,
    path_space: str = "source",
) -> dict[str, Any]:
    """Return authored rigid-body, collider, joint, and articulation facts."""

    from pxr import Usd, UsdPhysics

    path = Path(usd_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input USD not found: {path}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {path}")
    if root_prim_path:
        root = stage.GetPrimAtPath(root_prim_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"Root prim not found: {root_prim_path}")

    prims = [
        prim
        for prim in stage.Traverse()
        if _path_in_scope(str(prim.GetPath()), root_prim_path)
    ]
    rigid_body_paths = sorted(
        str(prim.GetPath()) for prim in prims if _enabled_rigid_body(prim, UsdPhysics)
    )
    rigid_body_set = set(rigid_body_paths)
    owner_rigid_body_set = rigid_body_set | _ancestor_enabled_rigid_body_paths(
        stage,
        root_prim_path,
        UsdPhysics,
    )
    articulation_root_paths = sorted(
        str(prim.GetPath())
        for prim in prims
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )

    colliders: list[dict[str, Any]] = []
    for prim in prims:
        if not _enabled_collider(prim, UsdPhysics):
            continue
        prim_path = str(prim.GetPath())
        colliders.append(
            {
                "prim_path": prim_path,
                "owner_rigid_body_path": _nearest_path(
                    prim.GetPath(),
                    owner_rigid_body_set,
                ),
                "type_name": prim.GetTypeName(),
            }
        )

    joints: list[dict[str, Any]] = []
    fixed_to_world: list[dict[str, Any]] = []
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0_targets, body1_targets = _relationship_targets(joint)
        body0_owners = sorted(
            {
                owner
                for target in body0_targets
                if (
                    owner := _nearest_path(
                        stage.GetPrimAtPath(target).GetPath(),
                        owner_rigid_body_set,
                    )
                )
            }
        )
        body1_owners = sorted(
            {
                owner
                for target in body1_targets
                if (
                    owner := _nearest_path(
                        stage.GetPrimAtPath(target).GetPath(),
                        owner_rigid_body_set,
                    )
                )
            }
        )
        record = {
            "prim_path": str(prim.GetPath()),
            "joint_type": prim.GetTypeName(),
            "is_fixed_joint": prim.IsA(UsdPhysics.FixedJoint),
            "enabled": _api_enabled(joint.GetJointEnabledAttr()),
            "body0_targets": body0_targets,
            "body1_targets": body1_targets,
            "body0_rigid_body_paths": body0_owners,
            "body1_rigid_body_paths": body1_owners,
        }
        joints.append(record)
        if (
            record["enabled"]
            and prim.IsA(UsdPhysics.FixedJoint)
            and (
                (not body0_targets and body1_targets)
                or (body0_targets and not body1_targets)
                or (not body0_owners and bool(body1_owners))
                or (bool(body0_owners) and not body1_owners)
            )
        ):
            fixed_to_world.append(record)

    findings: list[dict[str, Any]] = []
    for body_path in rigid_body_paths:
        parent_body = _nearest_path(
            stage.GetPrimAtPath(body_path).GetPath().GetParentPath(),
            owner_rigid_body_set,
        )
        if parent_body:
            findings.append(
                {
                    "code": "nested_enabled_rigid_body",
                    "prim_path": body_path,
                    "related_paths": [parent_body],
                }
            )
        if not any(
            collider["owner_rigid_body_path"] == body_path for collider in colliders
        ):
            findings.append(
                {
                    "code": "rigid_body_without_collider",
                    "prim_path": body_path,
                    "related_paths": [],
                }
            )
    for joint in fixed_to_world:
        findings.append(
            {
                "code": "fixed_joint_to_non_rigid_root",
                "prim_path": joint["prim_path"],
                "related_paths": sorted(
                    set(joint["body0_targets"] + joint["body1_targets"])
                ),
            }
        )
    for joint in joints:
        resolved_bodies = set(
            joint["body0_rigid_body_paths"] + joint["body1_rigid_body_paths"]
        )
        if (
            joint["enabled"]
            and joint["is_fixed_joint"]
            and joint["body0_targets"]
            and joint["body1_targets"]
            and len(resolved_bodies) == 1
        ):
            findings.append(
                {
                    "code": "fixed_joint_same_rigid_body",
                    "prim_path": joint["prim_path"],
                    "related_paths": sorted(resolved_bodies),
                }
            )

    return {
        "schema_version": "content-workbench.physics-topology.v1",
        "asset": str(path),
        "source_digest": sha256_file(path),
        "path_space": path_space,
        "root_prim_path": root_prim_path,
        "rigid_body_paths": rigid_body_paths,
        "enabled_rigid_body_count": len(rigid_body_paths),
        "colliders": sorted(colliders, key=lambda item: item["prim_path"]),
        "enabled_collider_count": len(colliders),
        "joints": sorted(joints, key=lambda item: item["prim_path"]),
        "articulation_root_paths": articulation_root_paths,
        "fixed_to_world_joints": sorted(
            fixed_to_world, key=lambda item: item["prim_path"]
        ),
        "findings": sorted(
            findings, key=lambda item: (item["code"], item["prim_path"])
        ),
    }


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)


def _display_opacity(prim: Any, UsdGeom: Any) -> float | None:
    if not prim.IsA(UsdGeom.Gprim):
        return None
    values = UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Get()
    if not values:
        return None
    return max(float(value) for value in values)


def _material_evidence(prim: Any, UsdShade: Any) -> dict[str, str] | None:
    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not material or not material.GetPrim():
        return None
    material_prim = material.GetPrim()
    return {
        "prim_path": str(prim.GetPath()),
        "material_path": str(material_prim.GetPath()),
        "material_name": material_prim.GetName(),
    }


def _is_helper_prim(prim: Any, *, opacity: float | None, UsdGeom: Any) -> bool:
    name = prim.GetName().lower()
    purpose = None
    visibility = None
    if prim.IsA(UsdGeom.Imageable):
        imageable = UsdGeom.Imageable(prim)
        purpose = imageable.ComputePurpose()
        visibility = imageable.ComputeVisibility()
    helper_name = any(
        token in name
        for token in ("bbox", "bounding", "bounds", "helper", "guide", "debug")
    )
    return bool(
        helper_name
        or purpose == UsdGeom.Tokens.guide
        or visibility == UsdGeom.Tokens.invisible
        or opacity == 0.0
    )


def _component_bounds(
    stage: Any, paths: list[str], Usd: Any, UsdGeom: Any
) -> dict[str, Any]:
    if not paths:
        return {}
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    minimum: list[float] | None = None
    maximum: list[float] | None = None
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Boundable):
            continue
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        current_min = [float(value) * meters_per_unit for value in aligned.GetMin()]
        current_max = [float(value) * meters_per_unit for value in aligned.GetMax()]
        minimum = (
            current_min
            if minimum is None
            else [min(a, b) for a, b in zip(minimum, current_min, strict=False)]
        )
        maximum = (
            current_max
            if maximum is None
            else [max(a, b) for a, b in zip(maximum, current_max, strict=False)]
        )
    if minimum is None or maximum is None:
        return {}
    size = [max(high - low, 0.0) for low, high in zip(minimum, maximum, strict=False)]
    return {
        "min_m": [round(value, 6) for value in minimum],
        "max_m": [round(value, 6) for value in maximum],
        "size_m": [round(value, 6) for value in size],
        "volume_m3": round(size[0] * size[1] * size[2], 12),
    }


def _finding_signature(finding: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    return (
        str(finding.get("code") or ""),
        str(finding.get("prim_path") or ""),
        tuple(sorted(str(path) for path in finding.get("related_paths") or [])),
    )


def inspect_physics_components(
    usd_path: Path | str,
    *,
    root_prim_path: str | None = None,
    path_space: str = "source",
) -> dict[str, Any]:
    """Group visual evidence, colliders, helpers, bodies, and joints by component."""

    from pxr import Usd, UsdGeom, UsdShade

    topology = inspect_physics_topology(
        usd_path, root_prim_path=root_prim_path, path_space=path_space
    )
    path = Path(usd_path).resolve()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {path}")

    bodies = sorted(
        set(topology["rigid_body_paths"])
        | {
            str(collider["owner_rigid_body_path"])
            for collider in topology["colliders"]
            if collider.get("owner_rigid_body_path")
        }
        | {
            str(owner)
            for joint in topology["joints"]
            for owner in (
                joint.get("body0_rigid_body_paths", [])
                + joint.get("body1_rigid_body_paths", [])
            )
        }
    )
    body_set = set(bodies)
    groups = _DisjointSet(bodies)
    for body_path in bodies:
        parent = _nearest_path(
            stage.GetPrimAtPath(body_path).GetPath().GetParentPath(), body_set
        )
        if parent:
            groups.union(body_path, parent)

    grouped_bodies: dict[str, list[str]] = {}
    for body_path in bodies:
        grouped_bodies.setdefault(groups.find(body_path), []).append(body_path)
    default_prim = stage.GetDefaultPrim()
    fallback_root = root_prim_path or (
        str(default_prim.GetPath()) if default_prim and default_prim.IsValid() else "/"
    )
    has_body_groups = bool(grouped_bodies)
    group_keys = sorted(grouped_bodies) or [fallback_root]

    owner_to_group = {
        body_path: group_key
        for group_key, group_bodies in grouped_bodies.items()
        for body_path in group_bodies
    }
    collider_to_group: dict[str, str] = {}
    collider_paths: set[str] = set()
    for collider in topology["colliders"]:
        collider_paths.add(collider["prim_path"])
        owner = collider["owner_rigid_body_path"]
        if owner in owner_to_group:
            group_key = owner_to_group[owner]
        elif has_body_groups:
            group_key = _unowned_static_group_key(collider["prim_path"])
            group_keys.append(group_key)
        else:
            group_key = group_keys[0]
        collider_to_group[collider["prim_path"]] = group_key
    group_keys = sorted(set(group_keys))

    joint_to_group_keys: dict[str, list[str]] = {}
    for joint in topology["joints"]:
        owners = sorted(
            set(joint["body0_rigid_body_paths"] + joint["body1_rigid_body_paths"])
        )
        keys = sorted(
            {owner_to_group[owner] for owner in owners if owner in owner_to_group}
        )
        if keys:
            joint_to_group_keys[joint["prim_path"]] = keys

    role_records: dict[str, dict[str, Any]] = {
        key: {
            "component_role": "unowned_static"
            if _is_unowned_static_group(key)
            else "body",
            "visual_evidence_paths": [],
            "collider_paths": [],
            "helper_paths": [],
            "material_evidence": [],
        }
        for key in group_keys
    }
    for collider_path, group_key in collider_to_group.items():
        role_records[group_key]["collider_paths"].append(collider_path)

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not _path_in_scope(prim_path, root_prim_path):
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        owner = _nearest_path(prim.GetPath(), body_set)
        if owner:
            group_key = owner_to_group.get(owner, group_keys[0])
        elif has_body_groups:
            nearest_collider = _nearest_path(prim.GetPath(), collider_paths)
            group_key = collider_to_group.get(
                nearest_collider or prim_path,
                _unowned_static_group_key(prim_path),
            )
            role_records.setdefault(
                group_key,
                {
                    "component_role": "unowned_static",
                    "visual_evidence_paths": [],
                    "collider_paths": [],
                    "helper_paths": [],
                    "material_evidence": [],
                },
            )
            if group_key not in group_keys:
                group_keys.append(group_key)
        else:
            group_key = group_keys[0]
        opacity = _display_opacity(prim, UsdGeom)
        material = _material_evidence(prim, UsdShade)
        is_collider = prim_path in collider_to_group
        is_helper = _is_helper_prim(prim, opacity=opacity, UsdGeom=UsdGeom)
        if is_helper and not is_collider:
            role_records[group_key]["helper_paths"].append(prim_path)
        elif not is_helper and (
            not is_collider or material is not None or opacity is None or opacity > 0.0
        ):
            role_records[group_key]["visual_evidence_paths"].append(prim_path)
        if material is not None and not is_helper:
            role_records[group_key]["material_evidence"].append(material)

    findings_by_body: dict[str, list[str]] = {}
    for finding in topology["findings"]:
        path_owner = _nearest_path(
            stage.GetPrimAtPath(finding["prim_path"]).GetPath(), body_set
        )
        keys = (
            [owner_to_group.get(path_owner, group_keys[0])]
            if path_owner
            else joint_to_group_keys.get(finding["prim_path"], [group_keys[0]])
        )
        for key in keys:
            findings_by_body.setdefault(key, []).append(finding["code"])

    components: list[dict[str, Any]] = []
    for index, group_key in enumerate(sorted(set(group_keys)), start=1):
        group_bodies = sorted(grouped_bodies.get(group_key, []))
        relevant_joints = sorted(
            joint["prim_path"]
            for joint in topology["joints"]
            if set(joint["body0_rigid_body_paths"] + joint["body1_rigid_body_paths"])
            & set(group_bodies)
        )
        roles = role_records[group_key]
        visual_paths = sorted(set(roles["visual_evidence_paths"]))
        collider_paths = sorted(set(roles["collider_paths"]))
        helper_paths = sorted(set(roles["helper_paths"]))
        body_root = (
            min(group_bodies, key=lambda item: (item.count("/"), item))
            if group_bodies
            else (
                _unowned_static_body_root(group_key)
                if _is_unowned_static_group(group_key)
                else fallback_root
            )
        )
        components.append(
            {
                "component_id": f"component_{index:03d}",
                "component_role": roles["component_role"],
                "path_space": path_space,
                "body_root_path": body_root,
                "visual_evidence_paths": visual_paths,
                "collider_paths": collider_paths,
                "helper_paths": helper_paths,
                "rigid_body_paths": group_bodies,
                "joint_paths": relevant_joints,
                "material_evidence": sorted(
                    roles["material_evidence"],
                    key=lambda item: (item["prim_path"], item["material_path"]),
                ),
                "bounds_m": _component_bounds(
                    stage,
                    visual_paths or collider_paths,
                    Usd,
                    UsdGeom,
                ),
                "topology_findings": sorted(set(findings_by_body.get(group_key, []))),
            }
        )

    return {
        "schema_version": "content-workbench.physics-components.v2",
        "asset": str(path),
        "source_digest": topology["source_digest"],
        "path_space": path_space,
        "topology_summary": {
            "enabled_rigid_body_count": topology["enabled_rigid_body_count"],
            "enabled_collider_count": topology["enabled_collider_count"],
            "joint_count": len(topology["joints"]),
            "articulation_root_count": len(topology["articulation_root_paths"]),
            "findings": topology["findings"],
        },
        "component_count": len(components),
        "components": components,
    }


def apply_physics_topology_plan(
    *,
    input_usd_path: Path | str,
    output_usd_path: Path | str,
    expected_source_digest: str,
    mobility_intent: str = "preserve",
    operations: list[dict[str, Any]],
    invariants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply an allowlisted topology plan to a flattened derivative USD."""

    from pxr import Usd, UsdGeom, UsdPhysics

    source = Path(input_usd_path).resolve()
    output = Path(output_usd_path).resolve()
    if source == output:
        raise PhysicsTopologyPlanError("Topology plans must write a derivative output")
    if source.suffix.lower() == ".usdz":
        raise PhysicsTopologyPlanError(
            "Topology plans do not support USDZ package inputs; unpack the package "
            "to a USD layer with copied dependencies before topology repair"
        )
    if output.suffix.lower() == ".usdz":
        raise PhysicsTopologyPlanError(
            "Topology plans must write a USD layer output, not a USDZ package"
        )
    if mobility_intent not in SUPPORTED_MOBILITY_INTENTS:
        raise PhysicsTopologyPlanError(
            f"mobility_intent must be one of {sorted(SUPPORTED_MOBILITY_INTENTS)}"
        )
    actual_digest = sha256_file(source)
    if expected_source_digest != actual_digest:
        raise PhysicsTopologyPlanError(
            "Source digest mismatch; inspect the current asset before applying a plan"
        )
    if not isinstance(operations, list):
        raise PhysicsTopologyPlanError("operations must be a list")
    invariants = dict(invariants or {})
    if "enabled_collider_count" not in invariants:
        raise PhysicsTopologyPlanError(
            "Topology plans require an enabled_collider_count invariant"
        )
    if invariants.get("reject_articulation_changes") is not True:
        raise PhysicsTopologyPlanError(
            "Topology plans require reject_articulation_changes=true"
        )
    before = inspect_physics_topology(source)
    before_components = inspect_physics_components(source)
    expected_colliders = invariants.get("enabled_collider_count")
    if expected_colliders is not None and int(expected_colliders) != int(
        before["enabled_collider_count"]
    ):
        raise PhysicsTopologyPlanError(
            "enabled_collider_count invariant does not match the source asset"
        )

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {source}")
    articulation_roots = list(before["articulation_root_paths"])
    non_fixed_joint_endpoints = _non_fixed_joint_endpoint_paths(before)
    normalized: list[tuple[str, str]] = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise PhysicsTopologyPlanError(f"operation {index} must be an object")
        op = operation.get("op")
        prim_path = operation.get("prim_path")
        if op not in SUPPORTED_TOPOLOGY_OPERATIONS:
            raise PhysicsTopologyPlanError(f"Unsupported topology operation: {op!r}")
        if not isinstance(prim_path, str) or not prim_path.startswith("/"):
            raise PhysicsTopologyPlanError(
                f"operation {index} has an invalid prim_path"
            )
        if mobility_intent == "preserve" and op in {
            "remove_rigid_body_api",
            "remove_fixed_joint",
        }:
            raise PhysicsTopologyPlanError(f"mobility_intent='preserve' forbids {op}")
        if mobility_intent == "static" and op == "ensure_rigid_body_api":
            raise PhysicsTopologyPlanError(
                "mobility_intent='static' forbids ensure_rigid_body_api"
            )
        if op == "remove_fixed_joint" and mobility_intent != "movable":
            raise PhysicsTopologyPlanError(
                "Removing a fixed joint requires mobility_intent='movable'"
            )
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise PhysicsTopologyPlanError(
                f"Topology target does not exist: {prim_path}"
            )
        if op == "remove_fixed_joint" and not prim.IsA(UsdPhysics.FixedJoint):
            raise PhysicsTopologyPlanError(
                f"Only UsdPhysics.FixedJoint prims may be removed: {prim_path}"
            )
        if op == "remove_fixed_joint" and prim.GetChildren():
            raise PhysicsTopologyPlanError(
                "remove_fixed_joint refuses to delete a joint prim with children "
                f"because RemovePrim would delete the child subtree: {prim_path}"
            )
        if op == "remove_rigid_body_api" and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise PhysicsTopologyPlanError(
                "Only prims with UsdPhysics.RigidBodyAPI may remove "
                f"RigidBodyAPI: {prim_path}"
            )
        if op == "ensure_rigid_body_api" and not prim.IsA(UsdGeom.Xformable):
            raise PhysicsTopologyPlanError(
                f"Only Xformable prims may receive UsdPhysics.RigidBodyAPI: {prim_path}"
            )
        if _path_overlaps_any_root(prim_path, articulation_roots):
            raise PhysicsTopologyPlanError(
                "Topology operation would violate "
                f"reject_articulation_changes=true: {prim_path}"
            )
        if op == "remove_fixed_joint":
            fixed_joint = UsdPhysics.FixedJoint(prim)
            body0_targets, body1_targets = _relationship_targets(fixed_joint)
            for target_path in [*body0_targets, *body1_targets]:
                if _path_overlaps_any_root(target_path, articulation_roots):
                    raise PhysicsTopologyPlanError(
                        "Topology operation would violate "
                        "reject_articulation_changes=true by editing a joint "
                        f"endpoint under an articulation root: {target_path}"
                    )
        if op == "remove_rigid_body_api" and _path_overlaps_any_root(
            prim_path,
            non_fixed_joint_endpoints,
        ):
            raise PhysicsTopologyPlanError(
                "Topology operation would remove a rigid body referenced by a "
                f"non-fixed joint endpoint: {prim_path}"
            )
        normalized.append((op, prim_path))

    flattened = stage.Flatten()
    editable = Usd.Stage.Open(flattened)
    if editable is None:
        raise RuntimeError(f"Failed to create editable flattened stage: {source}")
    applied: list[dict[str, str]] = []
    for op, prim_path in normalized:
        prim = editable.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise PhysicsTopologyPlanError(
                f"Topology target disappeared after flattening: {prim_path}"
            )
        if prim.IsInstanceProxy():
            raise PhysicsTopologyPlanError(
                f"Topology target is an instance proxy and cannot be edited: {prim_path}"
            )
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
        if op == "ensure_rigid_body_api":
            api = UsdPhysics.RigidBodyAPI.Apply(prim)
            api.CreateRigidBodyEnabledAttr(True)
        elif op == "remove_rigid_body_api":
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        else:
            editable.RemovePrim(prim_path)
        applied.append({"op": op, "prim_path": prim_path})

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.stem}.{uuid.uuid4().hex}{output.suffix}")
    try:
        if not editable.GetRootLayer().Export(str(temp)):
            raise RuntimeError(f"Failed to export topology derivative: {temp}")
        temp_after = inspect_physics_topology(temp)
        if temp_after["enabled_collider_count"] != before["enabled_collider_count"]:
            raise PhysicsTopologyPlanError(
                "Topology plan changed the enabled collider count"
            )
        before_nested = {
            _finding_signature(finding)
            for finding in before.get("findings", [])
            if finding.get("code") == "nested_enabled_rigid_body"
        }
        after_nested = {
            _finding_signature(finding)
            for finding in temp_after.get("findings", [])
            if finding.get("code") == "nested_enabled_rigid_body"
        }
        if after_nested - before_nested:
            raise PhysicsTopologyPlanError(
                "Topology plan created nested enabled rigid bodies"
            )
        if temp_after["articulation_root_paths"] != before["articulation_root_paths"]:
            raise PhysicsTopologyPlanError(
                "Topology plan changed articulation roots despite "
                "reject_articulation_changes=true"
            )
        if _non_fixed_joint_endpoint_signature(temp_after) != (
            _non_fixed_joint_endpoint_signature(before)
        ):
            raise PhysicsTopologyPlanError(
                "Topology plan changed non-fixed joint endpoint ownership"
            )
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    after = inspect_physics_topology(output)
    after_components = inspect_physics_components(output)
    return {
        "operation": "physics.apply_topology_plan",
        "schema_version": TOPOLOGY_PLAN_SCHEMA,
        "input_usd_path": str(source),
        "output_usd_path": str(output),
        "source_digest": actual_digest,
        "output_digest": sha256_file(output),
        "mobility_intent": mobility_intent,
        "applied_operations": applied,
        "rejected_operations": [],
        "warnings": [],
        "invariants": {
            "enabled_collider_count": before["enabled_collider_count"],
            "reject_articulation_changes": True,
        },
        "invariant_results": {
            "enabled_collider_count_preserved": True,
            "articulation_changes_rejected": True,
        },
        "before": before,
        "after": after,
        "before_components": before_components,
        "after_components": after_components,
    }
