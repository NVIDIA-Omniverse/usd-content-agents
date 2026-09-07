# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw authored-physics topology inspection for the usd-cli service.

This module deliberately reports composed USD facts only.  Component grouping,
repair plans, and any accept/reject policy belong to domain workflows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _source_digest(stage: Any) -> str:
    """Return a digest of the exact live layers used by an inspection."""

    digest = hashlib.sha256()
    digest.update(b"usd-cli-stage-dependency-digest-v1\0")
    session_identifier = stage.GetSessionLayer().identifier
    layers: list[tuple[str, Any]] = []
    for layer in stage.GetUsedLayers():
        if layer.identifier == session_identifier:
            # Anonymous identifiers are process-specific. Hash the live session
            # layer under a stable label so unsaved tool edits affect evidence
            # without making equal content nondeterministic across processes.
            layers.append(("<session-layer>", layer))
        elif layer.anonymous:
            continue
        else:
            layers.append(
                (str(layer.resolvedPath or layer.realPath or layer.identifier), layer)
            )
    for identifier, layer in sorted(layers, key=lambda item: item[0]):
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\0")
        # Export the loaded layer instead of rereading its backing file. A usd-cli
        # session may have unsaved edits, and the topology and digest must describe
        # the same in-memory stage contents.
        payload = layer.ExportToString().encode("utf-8", errors="replace")
        digest.update(payload)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def source_digest(usd_path: Path | str) -> str:
    """Return the exact composed-layer digest used by topology inspection."""

    from pxr import Usd

    path = Path(usd_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input USD not found: {path}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {path}")
    return _source_digest(stage)


def _enabled(prim: Any, api: Any, attr_name: str) -> bool:
    if not prim.HasAPI(api):
        return False
    value = getattr(api(prim), attr_name)().Get()
    return value is not False


def _nearest_owner(path: Any, owners: set[str]) -> str | None:
    current = path
    while current and not current.IsAbsoluteRootPath():
        candidate = str(current)
        if candidate in owners:
            return candidate
        current = current.GetParentPath()
    return None


def _traverse(stage: Any, Usd: Any, root_prim_path: str | None) -> list[Any]:
    predicate = Usd.TraverseInstanceProxies()
    if root_prim_path:
        root = stage.GetPrimAtPath(root_prim_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"Root prim not found: {root_prim_path}")
        return list(Usd.PrimRange(root, predicate))
    return list(Usd.PrimRange.Stage(stage, predicate))


def inspect_topology(
    usd_path: Path | str,
    *,
    root_prim_path: str | None = None,
    path_space: str = "source",
    stage: Any | None = None,
) -> dict[str, Any]:
    """Inspect authored rigid-body, collider, joint, and articulation facts."""

    from pxr import Usd, UsdGeom, UsdPhysics

    path = Path(usd_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input USD not found: {path}")
    if stage is None:
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise RuntimeError(f"Failed to open USD stage: {path}")
    else:
        stage_path = stage.GetRootLayer().realPath
        if not stage_path or Path(stage_path).resolve() != path:
            raise RuntimeError("Live topology stage does not match the requested asset")
    prims = _traverse(stage, Usd, root_prim_path)
    rigid_body_paths = sorted(
        str(prim.GetPath())
        for prim in prims
        if _enabled(prim, UsdPhysics.RigidBodyAPI, "GetRigidBodyEnabledAttr")
    )
    owners = set(rigid_body_paths)
    if root_prim_path:
        ancestor = stage.GetPrimAtPath(root_prim_path).GetParent()
        while ancestor and ancestor.IsValid() and not ancestor.IsPseudoRoot():
            if _enabled(ancestor, UsdPhysics.RigidBodyAPI, "GetRigidBodyEnabledAttr"):
                owners.add(str(ancestor.GetPath()))
            ancestor = ancestor.GetParent()

    colliders: list[dict[str, Any]] = []
    for prim in prims:
        if _enabled(prim, UsdPhysics.CollisionAPI, "GetCollisionEnabledAttr"):
            colliders.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "owner_rigid_body_path": _nearest_owner(prim.GetPath(), owners),
                    "type_name": prim.GetTypeName(),
                }
            )

    joints: list[dict[str, Any]] = []
    fixed_to_world: list[dict[str, Any]] = []
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0_targets = [str(item) for item in joint.GetBody0Rel().GetTargets()]
        body1_targets = [str(item) for item in joint.GetBody1Rel().GetTargets()]
        body0_owners = sorted(
            {
                owner
                for target in body0_targets
                if (owner := _nearest_owner(stage.GetPrimAtPath(target).GetPath(), owners))
            }
        )
        body1_owners = sorted(
            {
                owner
                for target in body1_targets
                if (owner := _nearest_owner(stage.GetPrimAtPath(target).GetPath(), owners))
            }
        )
        record = {
            "prim_path": str(prim.GetPath()),
            "joint_type": prim.GetTypeName(),
            "is_fixed_joint": prim.IsA(UsdPhysics.FixedJoint),
            "enabled": joint.GetJointEnabledAttr().Get() is not False,
            "body0_targets": body0_targets,
            "body1_targets": body1_targets,
            "body0_rigid_body_paths": body0_owners,
            "body1_rigid_body_paths": body1_owners,
        }
        joints.append(record)
        if record["enabled"] and record["is_fixed_joint"] and (
            (not body0_targets and body1_targets)
            or (body0_targets and not body1_targets)
            or (not body0_owners and bool(body1_owners))
            or (bool(body0_owners) and not body1_owners)
        ):
            fixed_to_world.append(record)

    findings: list[dict[str, Any]] = []
    for body_path in rigid_body_paths:
        body = stage.GetPrimAtPath(body_path)
        parent_body = _nearest_owner(body.GetPath().GetParentPath(), owners)
        if parent_body and not UsdGeom.Xformable(body).GetResetXformStack():
            findings.append(
                {
                    "code": "nested_enabled_rigid_body",
                    "prim_path": body_path,
                    "related_paths": [parent_body],
                }
            )
        if not any(item["owner_rigid_body_path"] == body_path for item in colliders):
            findings.append(
                {
                    "code": "rigid_body_without_collider",
                    "prim_path": body_path,
                    "related_paths": [],
                }
            )
    for record in fixed_to_world:
        findings.append(
            {
                "code": "fixed_joint_to_non_rigid_root",
                "prim_path": record["prim_path"],
                "related_paths": sorted(set(record["body0_targets"] + record["body1_targets"])),
            }
        )
    for record in joints:
        resolved = set(record["body0_rigid_body_paths"] + record["body1_rigid_body_paths"])
        if (
            record["enabled"]
            and record["is_fixed_joint"]
            and record["body0_targets"]
            and record["body1_targets"]
            and len(resolved) == 1
        ):
            findings.append(
                {
                    "code": "fixed_joint_same_rigid_body",
                    "prim_path": record["prim_path"],
                    "related_paths": sorted(resolved),
                }
            )
    return {
        "schema_version": "usd-cli.physics-topology.v1",
        "asset": str(path),
        "source_digest": _source_digest(stage),
        "path_space": path_space,
        "root_prim_path": root_prim_path,
        "rigid_body_paths": rigid_body_paths,
        "enabled_rigid_body_count": len(rigid_body_paths),
        "colliders": sorted(colliders, key=lambda item: item["prim_path"]),
        "enabled_collider_count": len(colliders),
        "joints": sorted(joints, key=lambda item: item["prim_path"]),
        "articulation_root_paths": sorted(
            str(prim.GetPath())
            for prim in prims
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ),
        "fixed_to_world_joints": sorted(fixed_to_world, key=lambda item: item["prim_path"]),
        "findings": sorted(findings, key=lambda item: (item["code"], item["prim_path"])),
    }
