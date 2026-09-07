# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract authoritative articulated-geometry intent from authored USD physics."""

from __future__ import annotations

import hashlib
import math
import re
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from .advanced_profiles import (
    AdjacentLinkExclusion,
    AdvancedProfileRequest,
    JointSweepInput,
    LinkGeometryMapping,
    SemanticLinkHypothesis,
    SourceEvidence,
)
from .artifacts import atomic_write_json, file_sha256
from .models import StrictModel

USD_ARTICULATION_EXTRACTION_SCHEMA_VERSION = "geometry-repair.usd-articulation-extraction.v1"


class UsdArticulationExtraction(StrictModel):
    """Source-bound request or explicit blockers when source intent is incomplete."""

    schema_version: Literal["geometry-repair.usd-articulation-extraction.v1"] = (
        USD_ARTICULATION_EXTRACTION_SCHEMA_VERSION
    )
    status: Literal["ready", "blocked"]
    source_path: str
    source_sha256: str
    default_prim_path: str | None
    up_axis: str
    meters_per_unit: float = Field(gt=0.0)
    request: AdvancedProfileRequest | None = None
    joint_records: list[dict[str, Any]] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class NestedRigidBodyNormalization(StrictModel):
    """Evidence for transform-preserving nested-rigid-body normalization."""

    schema_version: Literal["geometry-repair.nested-rigid-body-normalization.v1"] = (
        "geometry-repair.nested-rigid-body-normalization.v1"
    )
    source_path: str
    source_sha256: str
    output_path: str
    output_sha256: str
    normalized_prim_paths: list[str]
    maximum_world_transform_drift: float = Field(ge=0.0)
    status: Literal["pass", "fail"]
    failures: list[str] = Field(default_factory=list)


def _link_id(path: str) -> str:
    basename = path.rsplit("/", 1)[-1] or "link"
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", basename).strip("_") or "link"
    suffix = hashlib.sha256(path.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{suffix}"


def _source_evidence(source_sha256: str, path: str, summary: str) -> SourceEvidence:
    return SourceEvidence(
        fact_kind="source_fact",
        source_ref=f"sha256:{source_sha256}#{path}",
        summary=summary,
        confidence=1.0,
    )


def _canonical_vector(
    value: tuple[float, float, float],
    *,
    up_axis: str,
) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=np.float64)
    if up_axis.upper() == "Y":
        vector = np.asarray((vector[0], -vector[2], vector[1]), dtype=np.float64)
    elif up_axis.upper() == "X":
        vector = np.asarray((-vector[2], vector[1], vector[0]), dtype=np.float64)
    return tuple(float(item) for item in vector)


def _nearest_rigid_body(prim: Any) -> Any | None:
    from pxr import UsdPhysics

    current = prim.GetParent()
    while current and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def _owned_geometry(stage: Any, body_path: str) -> tuple[list[str], list[str]]:
    from pxr import Usd, UsdGeom, UsdPhysics

    body = stage.GetPrimAtPath(body_path)
    render_paths: list[str] = []
    collision_paths: list[str] = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if prim == body or not prim.IsA(UsdGeom.Gprim):
            continue
        owner = _nearest_rigid_body(prim)
        if not owner or str(owner.GetPath()) != body_path:
            continue
        imageable = UsdGeom.Imageable(prim)
        purpose = str(imageable.ComputePurpose())
        visibility = str(imageable.ComputeVisibility())
        opacity = UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Get() or [1.0]
        visible = (
            purpose not in {str(UsdGeom.Tokens.guide), str(UsdGeom.Tokens.proxy)}
            and visibility != str(UsdGeom.Tokens.invisible)
            and any(float(item) > 0.0 for item in opacity)
        )
        path_text = str(prim.GetPath())
        if visible:
            render_paths.append(path_text)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is not False:
                collision_paths.append(path_text)
    return sorted(render_paths), sorted(collision_paths)


def _descendant_links(
    child_path: str,
    children_by_parent: dict[str, set[str]],
) -> list[str]:
    pending = [child_path]
    descendants: set[str] = set()
    while pending:
        current = pending.pop()
        if current in descendants:
            continue
        descendants.add(current)
        pending.extend(sorted(children_by_parent.get(current, ())))
    return sorted(descendants)


def extract_usd_articulation_request(
    source_path: str | Path,
    *,
    sample_count: int = 9,
    max_candidate_pairs: int = 250_000,
    output_path: str | Path | None = None,
) -> UsdArticulationExtraction:
    """Read joints and per-link geometry without inferring missing source intent."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    source = Path(source_path).expanduser().resolve()
    source_sha256 = file_sha256(source)
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open returned None for {source}")
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    transform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    motion_joints = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
    ]
    blockers: list[str] = []
    joint_sources: list[dict[str, Any]] = []
    body_paths: set[str] = set()
    children_by_parent: dict[str, set[str]] = {}
    for prim in motion_joints:
        joint = UsdPhysics.Joint(prim)
        body0_targets = [str(item) for item in joint.GetBody0Rel().GetTargets()]
        body1_targets = [str(item) for item in joint.GetBody1Rel().GetTargets()]
        path_text = str(prim.GetPath())
        if len(body0_targets) != 1 or len(body1_targets) != 1:
            blockers.append(
                f"{path_text}: motion joint requires exactly one body0 and one body1 target"
            )
            continue
        parent_path, child_path = body0_targets[0], body1_targets[0]
        if not stage.GetPrimAtPath(parent_path).HasAPI(UsdPhysics.RigidBodyAPI):
            blockers.append(f"{path_text}: body0 {parent_path!r} is not a USD rigid body")
            continue
        if not stage.GetPrimAtPath(child_path).HasAPI(UsdPhysics.RigidBodyAPI):
            blockers.append(f"{path_text}: body1 {child_path!r} is not a USD rigid body")
            continue
        body_paths.update((parent_path, child_path))
        children_by_parent.setdefault(parent_path, set()).add(child_path)
        joint_sources.append(
            {
                "prim": prim,
                "path": path_text,
                "parent_path": parent_path,
                "child_path": child_path,
            }
        )
    if not motion_joints:
        blockers.append("source USD has no revolute or prismatic motion joints")

    path_to_link = {path: _link_id(path) for path in sorted(body_paths)}
    semantic_links: list[SemanticLinkHypothesis] = []
    mappings: list[LinkGeometryMapping] = []
    for body_path in sorted(body_paths):
        render_paths, collision_paths = _owned_geometry(stage, body_path)
        if not render_paths:
            blockers.append(f"{body_path}: source-authored visible geometry is missing")
        if not collision_paths:
            blockers.append(f"{body_path}: source-authored enabled collision geometry is missing")
        if not render_paths or not collision_paths:
            continue
        evidence = [
            _source_evidence(
                source_sha256,
                body_path,
                "USD RigidBodyAPI ownership and descendant Gprim roles were read from source.",
            )
        ]
        link_id = path_to_link[body_path]
        semantic_links.append(
            SemanticLinkHypothesis(
                link_id=link_id,
                semantic_label=body_path.rsplit("/", 1)[-1],
                source_part_paths=[body_path],
                confidence=1.0,
                evidence=evidence,
            )
        )
        mappings.append(
            LinkGeometryMapping(
                link_id=link_id,
                render_paths=render_paths,
                collision_paths=collision_paths,
                evidence=evidence,
            )
        )

    joint_inputs: list[JointSweepInput] = []
    exclusions: list[AdjacentLinkExclusion] = []
    joint_records: list[dict[str, Any]] = []
    axis_vectors = {
        "X": Gf.Vec3d(1.0, 0.0, 0.0),
        "Y": Gf.Vec3d(0.0, 1.0, 0.0),
        "Z": Gf.Vec3d(0.0, 0.0, 1.0),
    }
    for source_joint in joint_sources:
        prim = source_joint["prim"]
        path_text = source_joint["path"]
        parent_path = source_joint["parent_path"]
        child_path = source_joint["child_path"]
        joint = UsdPhysics.Joint(prim)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            schema: Any = UsdPhysics.RevoluteJoint(prim)
            joint_type = "revolute"
            units = "radians"
            limit_scale = math.pi / 180.0
            source_limit_units = "degrees"
        else:
            schema = UsdPhysics.PrismaticJoint(prim)
            joint_type = "prismatic"
            units = "meters"
            limit_scale = meters_per_unit
            source_limit_units = "stage_units"
        axis_token = str(schema.GetAxisAttr().Get() or "")
        local_axis = axis_vectors.get(axis_token.upper())
        lower_raw = schema.GetLowerLimitAttr().Get()
        upper_raw = schema.GetUpperLimitAttr().Get()
        local_position = joint.GetLocalPos0Attr().Get()
        local_rotation = joint.GetLocalRot0Attr().Get()
        missing = [
            name
            for name, value in (
                ("axis", local_axis),
                ("lowerLimit", lower_raw),
                ("upperLimit", upper_raw),
                ("localPos0", local_position),
                ("localRot0", local_rotation),
            )
            if value is None
        ]
        if missing:
            blockers.append(f"{path_text}: missing authored fields {', '.join(missing)}")
            continue
        parent_matrix = transform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path))
        origin_stage = parent_matrix.Transform(Gf.Vec3d(local_position))
        rotated_axis = Gf.Rotation(local_rotation).TransformDir(local_axis)
        world_axis = parent_matrix.TransformDir(rotated_axis)
        axis_array = np.asarray(tuple(float(item) for item in world_axis), dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis_array))
        if axis_norm <= 1e-12:
            blockers.append(f"{path_text}: resolved joint axis has zero length")
            continue
        axis_array /= axis_norm
        axis_world = _canonical_vector(tuple(axis_array), up_axis=up_axis)
        origin_world = _canonical_vector(
            tuple(float(item) * meters_per_unit for item in origin_stage),
            up_axis=up_axis,
        )
        lower = float(lower_raw) * limit_scale
        upper = float(upper_raw) * limit_scale
        reference = 0.0 if lower <= 0.0 <= upper else lower
        evidence = [
            _source_evidence(
                source_sha256,
                path_text,
                "Joint type, bodies, frame, axis, and limits were read from USD Physics schemas.",
            )
        ]
        moving_paths = _descendant_links(child_path, children_by_parent)
        joint_inputs.append(
            JointSweepInput(
                joint_id=path_text,
                parent_link_id=path_to_link[parent_path],
                child_link_id=path_to_link[child_path],
                moving_link_ids=[path_to_link[path] for path in moving_paths],
                joint_type=joint_type,
                axis_world=axis_world,
                origin_world_m=origin_world,
                lower_limit=lower,
                upper_limit=upper,
                reference_value=reference,
                units=units,
                sample_count=sample_count,
                evidence=evidence,
            )
        )
        collision_enabled = joint.GetCollisionEnabledAttr().Get()
        if collision_enabled is not True:
            exclusions.append(
                AdjacentLinkExclusion(
                    link_a=path_to_link[parent_path],
                    link_b=path_to_link[child_path],
                    reason="USD Physics Joint collisionEnabled resolves false for connected bodies.",
                    evidence=evidence,
                )
            )
        joint_records.append(
            {
                "joint_path": path_text,
                "joint_type": joint_type,
                "parent_body_path": parent_path,
                "child_body_path": child_path,
                "axis_token": axis_token,
                "axis_world": list(axis_world),
                "origin_world_m": list(origin_world),
                "source_lower_limit": float(lower_raw),
                "source_upper_limit": float(upper_raw),
                "source_limit_units": source_limit_units,
                "canonical_lower_limit": lower,
                "canonical_upper_limit": upper,
                "canonical_units": units,
                "connected_body_collision_enabled": collision_enabled is True,
            }
        )

    request: AdvancedProfileRequest | None = None
    if not blockers:
        request = AdvancedProfileRequest(
            profile="articulated_rigid",
            semantic_links=semantic_links,
            link_mappings=mappings,
            joints=joint_inputs,
            adjacent_link_exclusions=exclusions,
            max_candidate_pairs=max_candidate_pairs,
        )
    result = UsdArticulationExtraction(
        status="ready" if request is not None else "blocked",
        source_path=str(source),
        source_sha256=source_sha256,
        default_prim_path=(
            str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None
        ),
        up_axis=up_axis,
        meters_per_unit=meters_per_unit,
        request=request,
        joint_records=joint_records,
        blockers=sorted(set(blockers)),
    )
    if output_path is not None:
        atomic_write_json(Path(output_path), result.model_dump(mode="json"))
    return result


def normalize_nested_rigid_body_xforms(
    source_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    tolerance: float = 1e-10,
) -> NestedRigidBodyNormalization:
    """Create a runtime copy with world-preserving reset stacks on nested bodies."""

    from pxr import Usd, UsdGeom, UsdPhysics

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if source == output:
        raise ValueError("nested rigid-body normalization must not overwrite the source")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open returned None for {output}")
    before_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    nested: list[Any] = []
    original_world: dict[str, Any] = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        ancestor = prim.GetParent()
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                path_text = str(prim.GetPath())
                nested.append(prim)
                original_world[path_text] = before_cache.GetLocalToWorldTransform(prim)
                break
            ancestor = ancestor.GetParent()

    for prim in nested:
        path_text = str(prim.GetPath())
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp(
            opSuffix="geometryRepairWorld",
            precision=UsdGeom.XformOp.PrecisionDouble,
        ).Set(original_world[path_text])
        xformable.SetResetXformStack(True)
        prim.SetCustomDataByKey(
            "geometryRepair:nestedRigidBodyNormalization",
            "world_transform_baked_reset_xform_stack",
        )
    stage.GetRootLayer().Save()

    reloaded = Usd.Stage.Open(str(output))
    if reloaded is None:
        raise RuntimeError(f"normalized stage could not be reopened: {output}")
    after_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    failures: list[str] = []
    maximum_drift = 0.0
    for path_text, expected in original_world.items():
        actual = after_cache.GetLocalToWorldTransform(reloaded.GetPrimAtPath(path_text))
        drift = max(
            abs(float(actual[row][column]) - float(expected[row][column]))
            for row in range(4)
            for column in range(4)
        )
        maximum_drift = max(maximum_drift, drift)
        if drift > tolerance:
            failures.append(
                f"{path_text}: world-transform drift {drift:.6g} exceeds {tolerance:.6g}"
            )
    report = NestedRigidBodyNormalization(
        source_path=str(source),
        source_sha256=file_sha256(source),
        output_path=str(output),
        output_sha256=file_sha256(output),
        normalized_prim_paths=sorted(original_world),
        maximum_world_transform_drift=maximum_drift,
        status="fail" if failures else "pass",
        failures=failures,
    )
    if report_path is not None:
        atomic_write_json(Path(report_path), report.model_dump(mode="json"))
    return report
