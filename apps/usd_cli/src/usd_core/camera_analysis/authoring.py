# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic-rig authoring primitives (transaction ownership remains in Session)."""

from __future__ import annotations

import json

import numpy as np

from usd_core.camera import author_camera, camera_lens_mm_to_raw
from usd_core.camera_analysis.contracts import (
    CameraPose,
    SceneAnalysisIR,
    SceneAnalysisPolicy,
)
from usd_core.camera_analysis.scene import (
    scene_analysis_policy_from_dict,
    transform_points,
)

RIG_SCHEMA = "usd-cli.camera-rig.v1"


def _validated_rig_path(path: str) -> str:
    from pxr import Sdf

    if not isinstance(path, str) or not Sdf.Path.IsValidPathString(path):
        raise ValueError("author-under must be a valid USD prim path")
    parsed = Sdf.Path(path)
    if (
        not parsed.IsAbsolutePath()
        or not parsed.IsPrimPath()
        or parsed.IsAbsoluteRootPath()
    ):
        raise ValueError("author-under must be a non-root absolute USD prim path")
    if parsed.ContainsPrimVariantSelection():
        raise ValueError("author-under cannot contain a variant selection")
    return parsed.pathString


def _available_append_path(stage, requested: str) -> str:
    if not stage.GetPrimAtPath(requested).IsValid():
        return requested
    for index in range(2, 10_000):
        candidate = f"{requested}_{index}"
        if not stage.GetPrimAtPath(candidate).IsValid():
            return candidate
    raise RuntimeError(f"could not allocate an append path below {requested}")


def camera_rig_destination(
    stage,
    author_under: str,
    on_existing: str = "error",
) -> str:
    """Resolve the exact rig path without mutating the stage."""

    requested = _validated_rig_path(author_under)
    if on_existing not in {"error", "replace", "append"}:
        raise ValueError("on-existing must be one of: error, replace, append")
    existing = stage.GetPrimAtPath(requested)
    if not existing.IsValid():
        return requested
    if existing.IsInstanceProxy():
        raise ValueError(
            f"cannot author a camera rig at {requested}: the prim is an "
            "instance proxy and is read-only"
        )
    if on_existing == "error":
        raise ValueError(
            f"camera rig target already exists: {requested} "
            "(use --on-existing replace|append)"
        )
    if on_existing == "append":
        return _available_append_path(stage, requested)
    return requested


def _prepare_rig_ancestors(stage, path: str) -> None:
    """Make regular instance ancestors editable and reject read-only proxies."""

    from pxr import Sdf

    current = Sdf.Path(path)
    while not current.IsAbsoluteRootPath():
        prim = stage.GetPrimAtPath(current)
        if prim.IsValid():
            if prim.IsInstanceProxy():
                raise ValueError(
                    f"cannot author a camera rig at {path}: {current} is an "
                    "instance proxy and is read-only"
                )
            if prim.IsInstance():
                if prim.SetInstanceable(False) is False:
                    raise RuntimeError(
                        f"failed to disable instancing at {current} before camera rig authoring"
                    )
                if stage.GetPrimAtPath(current).IsInstance():
                    raise RuntimeError(
                        f"could not make instance ancestor {current} editable"
                    )
        current = current.GetParentPath()


def author_camera_rig(
    stage,
    scene: SceneAnalysisIR,
    poses: tuple[CameraPose, ...],
    report: dict,
    *,
    author_under: str,
    on_existing: str = "error",
) -> dict:
    """Author an already accepted placement result as one deterministic rig subtree.

    The caller snapshots and restores the edit layer around this function; this helper
    deliberately performs no partial-error recovery of its own.
    """

    from pxr import Sdf, UsdGeom

    if not poses:
        raise ValueError("cannot author an empty camera rig")
    requested = _validated_rig_path(author_under)
    path = camera_rig_destination(stage, requested, on_existing)
    existing = stage.GetPrimAtPath(requested)
    if existing.IsValid() and on_existing == "replace":
        if not stage.RemovePrim(requested):
            raise RuntimeError(f"failed to remove existing camera rig at {requested}")
        if stage.GetPrimAtPath(requested).IsValid():
            raise ValueError(
                f"cannot replace {requested}: a weaker composed prim remains after removing "
                "the current edit-target spec"
            )

    _prepare_rig_ancestors(stage, path)
    rig = UsdGeom.Xform.Define(stage, Sdf.Path(path))
    # Placement poses are world-space facts.  A rig may be requested below an
    # already transformed organizational scope; resetting the rig stack keeps
    # those poses invariant instead of accidentally applying the ancestor
    # transform a second time.
    rig.SetResetXformStack(True)
    rig_prim = rig.GetPrim()
    rig_prim.SetCustomDataByKey("usdCameraRigSchema", RIG_SCHEMA)
    rig_prim.SetCustomDataByKey("usdCameraRigMethod", str(report["method"]))
    rig_prim.SetCustomDataByKey(
        "usdCameraRigSourceDigest", str(report["source_digest"])
    )
    normalized_policy = scene.policy.as_dict()
    policy_digest = scene.policy.digest
    if report.get("analysis_policy") not in (None, normalized_policy):
        raise ValueError("placement report analysis policy differs from its scene")
    if report.get("analysis_policy_digest") not in (None, policy_digest):
        raise ValueError(
            "placement report analysis policy digest differs from its scene"
        )
    rig_prim.SetCustomDataByKey(
        "usdCameraRigAnalysisPolicyJson",
        json.dumps(normalized_policy, sort_keys=True, separators=(",", ":")),
    )
    rig_prim.SetCustomDataByKey("usdCameraRigAnalysisPolicyDigest", policy_digest)
    rig_prim.SetCustomDataByKey("usdCameraRigSeed", int(report.get("seed", 0)))
    if report.get("scope_path"):
        rig_prim.SetCustomDataByKey("usdCameraRigScopePath", str(report["scope_path"]))
    if report.get("target_path"):
        rig_prim.SetCustomDataByKey(
            "usdCameraRigTargetPath", str(report["target_path"])
        )
    metadata = {
        key: report.get(key)
        for key in (
            "method",
            "scope_path",
            "target_path",
            "seed",
            "target_coverage",
            "per_cell",
            "occlusion_threshold",
            "config",
        )
        if report.get(key) is not None
    }
    metadata["analysis_policy"] = normalized_policy
    metadata["analysis_policy_digest"] = policy_digest
    rig_prim.SetCustomDataByKey(
        "usdCameraRigConfigJson",
        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    )

    records = report.get("cameras") or []
    if len(records) != len(poses):
        raise ValueError("placement result camera records and poses disagree")
    camera_paths: list[str] = []
    canonical_to_stage = scene.canonical_to_stage
    for index, (pose, record) in enumerate(zip(poses, records, strict=True), start=1):
        camera_path = f"{path}/Camera_{index:03d}"
        position = transform_points(
            np.asarray([pose.position_m], dtype=np.float64), canonical_to_stage
        )[0]
        look_at = np.asarray(record["look_at"], dtype=np.float64)
        camera = author_camera(
            stage,
            camera_path,
            position,
            look_at,
            focal=camera_lens_mm_to_raw(stage, pose.focal_length_mm),
            h_ap=camera_lens_mm_to_raw(stage, pose.horizontal_aperture_mm),
            v_ap=camera_lens_mm_to_raw(stage, pose.vertical_aperture_mm),
            near=max(0.001, pose.clipping_range_m[0] / scene.meters_per_unit),
            far=max(1.0, pose.clipping_range_m[1] / scene.meters_per_unit),
        )
        camera.CreateHorizontalApertureOffsetAttr(
            camera_lens_mm_to_raw(stage, pose.horizontal_aperture_offset_mm)
        )
        camera.CreateVerticalApertureOffsetAttr(
            camera_lens_mm_to_raw(stage, pose.vertical_aperture_offset_mm)
        )
        prim = camera.GetPrim()
        stable_id = str(record.get("id") or f"camera-{index:03d}")
        prim.SetCustomDataByKey("usdCameraStableId", stable_id)
        prim.SetCustomDataByKey("usdCameraRigMethod", str(report["method"]))
        prim.SetCustomDataByKey("usdCameraRigPath", path)
        camera_paths.append(camera_path)

    invalid = [
        camera_path
        for camera_path in camera_paths
        if not stage.GetPrimAtPath(camera_path).IsA(UsdGeom.Camera)
    ]
    if invalid:
        raise RuntimeError(f"camera rig verification failed for {invalid[0]}")
    return {
        "schema": RIG_SCHEMA,
        "rig_path": path,
        "requested_path": requested,
        "on_existing": on_existing,
        "camera_paths": camera_paths,
        "camera_count": len(camera_paths),
    }


def read_rig_metadata(prim) -> dict:
    raw = prim.GetCustomDataByKey("usdCameraRigConfigJson")
    if raw is None:
        return {}
    if not isinstance(raw, str) or not raw:
        raise ValueError("camera rig config metadata is malformed")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("camera rig config metadata is malformed") from exc
    if not isinstance(value, dict):
        raise ValueError("camera rig config metadata is malformed")
    return value


def read_rig_analysis_policy(prim) -> SceneAnalysisPolicy:
    """Read and cross-check the normalized policy persisted on an authored rig."""

    metadata = read_rig_metadata(prim)
    raw = prim.GetCustomDataByKey("usdCameraRigAnalysisPolicyJson")
    serialized = None
    if raw:
        if not isinstance(raw, str):
            raise ValueError("camera rig analysis policy metadata is malformed")
        try:
            serialized = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "camera rig analysis policy metadata is malformed"
            ) from exc
    elif "analysis_policy" in metadata:
        serialized = metadata["analysis_policy"]
    if serialized is None:
        policy = SceneAnalysisPolicy()
    else:
        policy = scene_analysis_policy_from_dict(serialized)

    metadata_policy = metadata.get("analysis_policy")
    if metadata_policy is not None:
        parsed_metadata = scene_analysis_policy_from_dict(metadata_policy)
        if parsed_metadata != policy:
            raise ValueError("camera rig contains conflicting analysis policies")
    declared_digests = {
        str(value)
        for value in (
            prim.GetCustomDataByKey("usdCameraRigAnalysisPolicyDigest"),
            metadata.get("analysis_policy_digest"),
        )
        if value
    }
    if declared_digests and declared_digests != {policy.digest}:
        raise ValueError("camera rig analysis policy digest does not match its policy")
    return policy
