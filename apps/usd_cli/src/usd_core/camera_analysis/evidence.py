# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Digest-bound, versioned OVRTX evidence for accepted camera rigs.

The local OVRTX 0.4 worker supplies GPU-reduced metric, normal, position, semantic,
albedo, and beauty observations.  The schema retains a narrow RGB compatibility path
for a future remote protocol that reports exact executing-worker identity.  The
bundled remote client and service do not currently expose that identity, so remote
verification fails closed rather than treating controller metadata as evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

from usd_core.camera_analysis.cancellation import check_cancelled, defer_cancellation
from usd_core.camera_analysis.contracts import SceneAnalysisIR, SceneAnalysisPolicy
from usd_core.camera_analysis.identifiers import camera_stable_identity
from usd_core.camera_analysis.scene import (
    is_path_at_or_below,
    normalize_scene_analysis_policy,
)
from usd_core.render.capabilities import (
    CAMERA_OBSERVATION_CAPABILITY,
    WARP_DLPACK_REDUCTION_CAPABILITY,
)

VERIFICATION_REQUEST_SCHEMA = "usd-cli.camera-verification-request.v1"
VERIFICATION_EVIDENCE_SCHEMA = "usd-cli.camera-verification-evidence.v1"
NEWTON_OVRTX_PARITY_SCHEMA = "usd-cli.newton-ovrtx-parity.v1"
PARITY_V1_TOLERANCE_POLICY = "raster_edges_allow_backend_specific_sampling_v1"
PARITY_V1_MINIMUM_MASK_IOU = 0.80
_PARITY_V1_MEAN_ABS_TOLERANCE_FLOOR_M = 0.01
_PARITY_V1_MEAN_ABS_MEDIAN_FRACTION = 0.025
_PARITY_V1_PERCENTILE_95_TOLERANCE_FLOOR_M = 0.025
_PARITY_V1_PERCENTILE_95_MEDIAN_FRACTION = 0.05
CANONICAL_JSON_DIGEST = "sha256-canonical-json-v1"
RGB_RENDER_CAPABILITY = "rgb_render_v1"
OVSTAGE_ATTACHED_CAPABILITY = "ovstage_attached_v1"
SEMANTIC_OVERLAY_CAPABILITY = "semantic_overlay_v1"
OVSTAGE_TRANSPORT = "ovstage_attached_ordinals"
WORKER_PROTOCOL_VERSION = 3
# SceneAnalysisIR, camera definitions, and Newton observations are all sampled at
# Usd.TimeCode.Default().  Bind that choice explicitly so numeric time code 0 can
# never be substituted silently by a renderer or by a later evidence consumer.
USD_DEFAULT_ANALYSIS_TIME = {"kind": "usd_default"}

OBSERVATION_AOVS = [
    "LdrColor",
    "DistanceToCameraSD",
    "DistanceToImagePlaneSD",
    "Camera3dPositionSD",
    "NormalSD",
    "SemanticSegmentation",
    "SemanticIdMap",
    "DiffuseAlbedoSD",
]
OBSERVATION_IMAGE_CHANNELS = {
    "LdrColor": 4,
    "DistanceToCameraSD": 1,
    "DistanceToImagePlaneSD": 1,
    "Camera3dPositionSD": 4,
    "NormalSD": 4,
    "SemanticSegmentation": 1,
    "DiffuseAlbedoSD": 4,
}

QUALIFIED_RUNTIME_VERSIONS = {
    "ovrtx": "0.4.1.364340",
    "ovstage": "0.1.1.355824",
    "warp": "1.16.0",
}

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the deterministic byte representation used for evidence digests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def parity_v1_depth_tolerances(median_distance_m: float) -> tuple[float, float]:
    """Return the fixed v1 mean and 95th-percentile depth tolerances."""

    median = float(median_distance_m)
    if not math.isfinite(median) or median <= 0.0:
        raise ValueError("parity median distance must be finite and positive")
    return (
        max(
            _PARITY_V1_MEAN_ABS_TOLERANCE_FLOOR_M,
            _PARITY_V1_MEAN_ABS_MEDIAN_FRACTION * median,
        ),
        max(
            _PARITY_V1_PERCENTILE_95_TOLERANCE_FLOOR_M,
            _PARITY_V1_PERCENTILE_95_MEDIAN_FRACTION * median,
        ),
    )


def sha256_file(path: str | Path) -> tuple[str, int]:
    resolved = Path(path)
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return "sha256:" + digest.hexdigest(), size


def _matrix(value) -> list[list[float]]:
    result = [[float(value[row][column]) for column in range(4)] for row in range(4)]
    if not all(math.isfinite(item) for row in result for item in row):
        raise ValueError("camera transform contains NaN or infinity")
    return result


def _optional_float(attribute, *, default: float | None = None) -> float | None:
    value = attribute.Get()
    if value is None:
        value = default
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("camera definition contains NaN or infinity")
    return result


def _camera_definition(stage, camera_path: str) -> dict:
    """Exact imaging-relevant camera definition sent to the renderer.

    This is deliberately independent from the calibration export.  Its digest binds
    evidence to the authored USD camera, including properties that affect RGB but are
    not part of a pinhole calibration (focus, shutter, and exposure).
    """

    from pxr import Sdf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(Sdf.Path(camera_path))
    if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise ValueError(f"not a camera: {camera_path}")
    camera = UsdGeom.Camera(prim)
    time = Usd.TimeCode.Default()
    transform = UsdGeom.XformCache(time).GetLocalToWorldTransform(prim)
    clipping = camera.GetClippingRangeAttr().Get(time) or (0.1, 1.0e6)
    clipping_planes = camera.GetClippingPlanesAttr().Get(time) or ()
    stable_id, _ = camera_stable_identity(prim, camera_path)
    definition = {
        "path": camera_path,
        "stable_id": stable_id,
        "world_from_camera": _matrix(transform),
        "matrix_layout": "row_major",
        "vector_convention": "row_vector_postmultiply",
        "projection": str(
            camera.GetProjectionAttr().Get(time) or UsdGeom.Tokens.perspective
        ),
        "focal_length_raw": _optional_float(camera.GetFocalLengthAttr(), default=50.0),
        "horizontal_aperture_raw": _optional_float(
            camera.GetHorizontalApertureAttr(), default=20.955
        ),
        "vertical_aperture_raw": _optional_float(
            camera.GetVerticalApertureAttr(), default=15.2908
        ),
        "horizontal_aperture_offset_raw": _optional_float(
            camera.GetHorizontalApertureOffsetAttr(), default=0.0
        ),
        "vertical_aperture_offset_raw": _optional_float(
            camera.GetVerticalApertureOffsetAttr(), default=0.0
        ),
        "clipping_range_stage_units": [float(clipping[0]), float(clipping[1])],
        "clipping_planes": [
            [float(plane[index]) for index in range(4)] for plane in clipping_planes
        ],
        "f_stop": _optional_float(camera.GetFStopAttr(), default=0.0),
        "focus_distance_stage_units": _optional_float(
            camera.GetFocusDistanceAttr(), default=0.0
        ),
        "shutter_open": _optional_float(camera.GetShutterOpenAttr(), default=0.0),
        "shutter_close": _optional_float(camera.GetShutterCloseAttr(), default=0.0),
        "exposure": _optional_float(camera.GetExposureAttr(), default=0.0),
    }
    # Validate nested values and make the digest's scope explicit.
    canonical_json_bytes(definition)
    definition["definition_digest"] = canonical_json_digest(definition)
    return definition


def _runtime_identity(backend) -> dict:
    reporter = getattr(backend, "runtime_identity", None)
    if not callable(reporter):
        raise RuntimeError(
            "OVRTX verification requires an executing backend that reports its worker "
            "runtime identity; this backend/protocol does not provide it"
        )
    identity = reporter()
    if not isinstance(identity, dict):
        raise RuntimeError("OVRTX worker returned a malformed runtime identity")
    # Round-trip through canonical JSON so evidence can never acquire backend-owned,
    # mutable, or non-JSON values.
    return json.loads(canonical_json_bytes(identity))


def _qualified_runtime(identity: dict, *, require_observation: bool = False) -> dict:
    reported = identity.get("runtime_versions")
    if not isinstance(reported, dict):
        reported = {}
    normalized = {
        "ovrtx": str(reported.get("ovrtx", "unreported")),
        "ovstage": str(reported.get("ovstage", "unreported")),
        "warp": str(reported.get("warp", reported.get("warp-lang", "unreported"))),
    }
    capabilities = identity.get("capabilities") or []
    if not isinstance(capabilities, list):
        capabilities = []
    checks = {
        key: normalized[key] == expected
        for key, expected in QUALIFIED_RUNTIME_VERSIONS.items()
    }
    checks[RGB_RENDER_CAPABILITY] = RGB_RENDER_CAPABILITY in capabilities
    checks[OVSTAGE_ATTACHED_CAPABILITY] = OVSTAGE_ATTACHED_CAPABILITY in capabilities
    if require_observation:
        checks[CAMERA_OBSERVATION_CAPABILITY] = (
            CAMERA_OBSERVATION_CAPABILITY in capabilities
        )
        checks[WARP_DLPACK_REDUCTION_CAPABILITY] = (
            WARP_DLPACK_REDUCTION_CAPABILITY in capabilities
        )
        checks[SEMANTIC_OVERLAY_CAPABILITY] = (
            SEMANTIC_OVERLAY_CAPABILITY in capabilities
        )
    checks["worker_protocol_version"] = (
        identity.get("worker_protocol_version") == WORKER_PROTOCOL_VERSION
    )
    checks["stage_transport"] = identity.get("stage_transport") == OVSTAGE_TRANSPORT
    qualification = {
        "required_versions": dict(QUALIFIED_RUNTIME_VERSIONS),
        "reported_versions": normalized,
        "required_worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "reported_worker_protocol_version": identity.get("worker_protocol_version"),
        "required_stage_transport": OVSTAGE_TRANSPORT,
        "reported_stage_transport": identity.get("stage_transport"),
        "required_capabilities": [RGB_RENDER_CAPABILITY, OVSTAGE_ATTACHED_CAPABILITY]
        + (
            [
                CAMERA_OBSERVATION_CAPABILITY,
                WARP_DLPACK_REDUCTION_CAPABILITY,
                SEMANTIC_OVERLAY_CAPABILITY,
            ]
            if require_observation
            else []
        ),
        "reported_capabilities": sorted(str(item) for item in capabilities),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not qualification["passed"]:
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "OVRTX rig verification requires the reviewed OVRTX 0.4.1.364340 + "
            "ovstage 0.1.1.355824 + Warp 1.16.0 worker; failed qualification: "
            + ", ".join(failures)
        )
    return qualification


def _relative_artifact(path: Path, directory: Path, artifact_base: Path) -> str:
    try:
        path.relative_to(directory)
    except ValueError as exc:
        raise RuntimeError(
            f"OVRTX verification wrote outside its evidence directory: {path}"
        ) from exc
    try:
        return path.relative_to(artifact_base).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"OVRTX verification artifact is outside its declared path base: {path}"
        ) from exc


def semantic_label(path: str, role: str) -> str:
    """Stable opaque label; the evidence separately retains the human USD path."""

    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
    safe_role = re.sub(r"[^a-z0-9_-]+", "_", role.strip().lower()).strip("_")
    if not safe_role:
        raise ValueError(f"semantic role for {path} is empty")
    return f"usd-cli:{safe_role}:path-sha256:{digest}"


def verification_semantic_roles(
    scene: SceneAnalysisIR,
    *,
    target_path: str | None,
    floor_paths: list[str],
) -> dict[str, str]:
    """Resolve measured floor and configured target identities without shadowing.

    USD semantic labels are inherited, so a label on a measured floor descendant
    overrides a target label on its ancestor.  Mark that descendant as carrying both
    roles.  Retain the ancestor label only when it still covers admitted target shapes
    outside the measured floor descendants; per-label parity then treats descendant
    labels as part of the ancestor's semantic mask.
    """

    roles = {str(path): "floor" for path in floor_paths}
    target = str(target_path or "")
    if not target:
        return dict(sorted(roles.items()))

    nested_floors = [
        path for path in roles if is_path_at_or_below(path, target)
    ]
    for path in nested_floors:
        roles[path] = "target_and_floor"

    target_shapes = [
        shape.prim_path
        for shape in scene.shapes
        if shape.analysis_role != "helper"
        and is_path_at_or_below(shape.prim_path, target)
    ]
    target_is_fully_covered = bool(target_shapes) and all(
        any(is_path_at_or_below(shape_path, floor_path) for floor_path in nested_floors)
        for shape_path in target_shapes
    )
    if not target_is_fully_covered:
        roles[target] = "target_and_floor" if target in roles else "target"
    return dict(sorted(roles.items()))


def _validated_semantic_paths(
    stage, semantic_roles: dict[str, str] | None
) -> list[str]:
    from pxr import Sdf

    paths: list[str] = []
    for path in semantic_roles or {}:
        if not isinstance(path, str) or not Sdf.Path.IsValidPathString(path):
            raise ValueError(f"invalid OVRTX verification semantic path: {path!r}")
        parsed = Sdf.Path(path)
        if not parsed.IsAbsolutePath() or not parsed.IsPrimPath():
            raise ValueError(
                f"OVRTX verification semantic path must be absolute: {path}"
            )
        if not stage.GetPrimAtPath(parsed).IsValid():
            raise ValueError(f"OVRTX verification semantic prim does not exist: {path}")
        paths.append(parsed.pathString)
    return sorted(paths)


def _make_clone_prim_authorable(
    stage,
    target_path: str,
    expanded_roots: list[str],
    expanded_root_set: set[str],
    *,
    purpose: str,
) -> None:
    """Expand containing native instances until one clone prim is authorable."""

    while True:
        target = stage.GetPrimAtPath(target_path)
        if not target.IsValid():
            raise RuntimeError(
                f"OVRTX verification {purpose} prim disappeared while expanding "
                f"instances: {target_path}"
            )
        if not target.IsInstanceProxy():
            return

        containing_instance = target
        while containing_instance.IsValid() and not (
            containing_instance.IsInstance()
            and not containing_instance.IsInstanceProxy()
        ):
            containing_instance = containing_instance.GetParent()
        if not containing_instance.IsValid():
            raise RuntimeError(
                "OVRTX verification could not find an authorable instance root "
                f"for {purpose} proxy {target_path}"
            )
        root_path = containing_instance.GetPath().pathString
        if root_path in expanded_root_set:
            raise RuntimeError(
                "OVRTX verification instance expansion made no progress for "
                f"{purpose} proxy {target_path} at {root_path}"
            )
        if not containing_instance.SetInstanceable(False):
            raise RuntimeError(
                "OVRTX verification could not expand instance root "
                f"{root_path} for {purpose} proxy {target_path}"
            )
        expanded_roots.append(root_path)
        expanded_root_set.add(root_path)


def _expand_semantic_instance_proxies(
    stage, semantic_roles: dict[str, str] | None
) -> list[str]:
    """Make semantic targets authorable on the disposable verification clone.

    Native USD instance-proxy prims cannot receive the semantic overlay used by the
    OVRTX worker.  Disable instanceability on the closest authorable containing
    instance, then resolve the target again.  Repeating is required when the target
    is below nested native instances: the inner instance root is itself a proxy until
    its outer instance has been expanded.
    """

    expanded_roots: list[str] = []
    expanded_root_set: set[str] = set()
    for target_path in _validated_semantic_paths(stage, semantic_roles):
        _make_clone_prim_authorable(
            stage,
            target_path,
            expanded_roots,
            expanded_root_set,
            purpose="semantic",
        )
    return expanded_roots


def _verification_policy(
    analysis_policy: SceneAnalysisPolicy | None,
    analysis_scene,
    *,
    source_digest: str,
) -> SceneAnalysisPolicy:
    """Resolve one policy and reject parity against a differently filtered world."""

    explicit = (
        normalize_scene_analysis_policy(analysis_policy)
        if analysis_policy is not None
        else None
    )
    if analysis_scene is not None:
        if analysis_scene.source_digest != source_digest:
            raise ValueError(
                "OVRTX verification SceneAnalysisIR is bound to another source stage"
            )
        scene_policy = normalize_scene_analysis_policy(analysis_scene.policy)
        if explicit is not None and explicit != scene_policy:
            raise ValueError(
                "OVRTX verification policy differs from its SceneAnalysisIR"
            )
        return scene_policy
    policy = explicit or SceneAnalysisPolicy()
    if policy != SceneAnalysisPolicy():
        raise ValueError(
            "non-default OVRTX verification policy requires the exact "
            "SceneAnalysisIR used by Newton"
        )
    return policy


def analysis_overlay_actions(stage, analysis_scene) -> list[dict[str, str]]:
    """Return the exact clone actions required to mirror one analytic world."""

    from pxr import Sdf, Usd, UsdGeom

    actions: list[dict[str, str]] = []
    if analysis_scene is None:
        return actions
    allowed_paths = {
        shape.prim_path
        for shape in analysis_scene.shapes
        if shape.analysis_role != "helper"
    }
    helper_paths = {
        shape.prim_path
        for shape in analysis_scene.shapes
        if shape.analysis_role == "helper"
    }
    helper_roots = tuple(Sdf.Path(path) for path in analysis_scene.policy.helper_paths)
    clone_paths = sorted(
        prim.GetPath().pathString
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        # PointInstancer is renderable Boundable geometry but not a Gprim.  It
        # must receive the same clone-only visibility action as excluded/helper
        # Gprims or OVRTX would render instances that Newton intentionally omitted.
        if prim.IsA(UsdGeom.Boundable) and not prim.IsA(UsdGeom.Camera)
    )
    for path in clone_paths:
        if path in allowed_paths:
            continue
        parsed = Sdf.Path(path)
        if any(
            allowed != path and Sdf.Path(allowed).HasPrefix(parsed)
            for allowed in allowed_paths
        ):
            raise ValueError(
                "OVRTX verification cannot safely hide an excluded ancestor "
                f"Gprim while retaining its admitted descendant: {path}"
            )
        actions.append(
            {
                "path": path,
                "operation": "set_visibility_invisible",
                "reason": (
                    "helper_geometry"
                    if path in helper_paths
                    or any(parsed.HasPrefix(root) for root in helper_roots)
                    else "absent_from_analysis_world"
                ),
            }
        )
    actions.extend(
        {
            "path": shape.prim_path,
            "operation": "set_double_sided_true",
            "reason": "analytic_meshes_are_two_sided",
        }
        for shape in analysis_scene.shapes
        if shape.analysis_role != "helper" and shape.kind == "mesh"
    )
    actions.sort(key=lambda item: (item["path"], item["operation"]))
    return actions


def _apply_analysis_visibility_overlay(stage, analysis_scene, policy) -> dict:
    """Apply exact analytic-world parity actions to a disposable render clone."""

    from pxr import UsdGeom

    actions = analysis_overlay_actions(stage, analysis_scene)
    expanded_roots: list[str] = []
    expanded_root_set: set[str] = set()
    for action in actions:
        path = action["path"]
        purpose = (
            "visibility-overlay"
            if action["operation"] == "set_visibility_invisible"
            else "mesh-sidedness-overlay"
        )
        _make_clone_prim_authorable(
            stage,
            path,
            expanded_roots,
            expanded_root_set,
            purpose=purpose,
        )
        if action["operation"] == "set_visibility_invisible":
            prim = stage.GetPrimAtPath(path)
            imageable = UsdGeom.Imageable(prim)
            if not imageable or not imageable.CreateVisibilityAttr().Set(
                UsdGeom.Tokens.invisible
            ):
                raise RuntimeError(
                    "OVRTX verification could not hide geometry excluded from "
                    f"Newton analysis: {path}"
                )
        else:
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
            if not mesh or not mesh.CreateDoubleSidedAttr().Set(True):
                raise RuntimeError(
                    "OVRTX verification could not align mesh sidedness with "
                    f"Newton analysis: {path}"
                )
    return {
        "scope": "verification_clone_only",
        "policy_digest": policy.digest,
        "actions": actions,
        "expanded_instance_roots": expanded_roots,
    }


def _semantic_assignments(
    stage, semantic_roles: dict[str, str] | None
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for path in _validated_semantic_paths(stage, semantic_roles):
        if stage.GetPrimAtPath(path).IsInstanceProxy():
            raise RuntimeError(
                "OVRTX verification semantic prim remains an instance proxy after "
                f"clone expansion: {path}"
            )
        labels[path] = semantic_label(path, str((semantic_roles or {})[path]))
    return labels


def _result_value(result, name: str, default=None):
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def has_exact_aov_element_count(record) -> bool:
    """Return whether an AOV summary covers every element in its declared shape."""

    if not isinstance(record, dict):
        return False
    shape = record.get("shape")
    statistics = record.get("statistics")
    if not isinstance(shape, list) or not shape or not isinstance(statistics, dict):
        return False
    if any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1
        for dimension in shape
    ):
        return False
    element_count = statistics.get("element_count")
    return (
        isinstance(element_count, int)
        and not isinstance(element_count, bool)
        and element_count == math.prod(shape)
    )


def has_positive_metric_distance_samples(statistics, *, max_pixels: int) -> bool:
    """Validate that a metric-distance reduction contains a positive hit sample."""

    if not isinstance(statistics, dict) or max_pixels < 1:
        return False
    valid_count = statistics.get("valid_count")
    nonzero_count = statistics.get("nonzero_count")
    minimum = statistics.get("minimum")
    maximum = statistics.get("maximum")
    counts_are_valid = (
        isinstance(valid_count, int)
        and not isinstance(valid_count, bool)
        and 1 <= valid_count <= max_pixels
        and isinstance(nonzero_count, int)
        and not isinstance(nonzero_count, bool)
        and 1 <= nonzero_count <= valid_count
    )
    extrema_are_valid = (
        isinstance(minimum, int | float)
        and not isinstance(minimum, bool)
        and math.isfinite(float(minimum))
        and float(minimum) >= 0.0
        and isinstance(maximum, int | float)
        and not isinstance(maximum, bool)
        and math.isfinite(float(maximum))
        and 0.0 < float(maximum) < 1.0e30
        and float(minimum) <= float(maximum)
    )
    return bool(counts_are_valid and extrema_are_valid)


def positive_requested_semantic_id_map(
    rendered_labels, expected_labels: set[str], *, max_pixels: int
) -> dict[str, int] | None:
    """Validate per-label raster counts and map each expected label to its ID."""

    if not expected_labels:
        return {}
    if not isinstance(rendered_labels, list) or max_pixels < 1:
        return None
    matched: dict[str, tuple[int, int]] = {}
    semantic_ids: set[int] = set()
    for item in rendered_labels:
        if not isinstance(item, dict) or item.get("label") not in expected_labels:
            continue
        label = item["label"]
        semantic_id = item.get("id")
        pixels = item.get("pixels")
        if (
            label in matched
            or not isinstance(semantic_id, int)
            or isinstance(semantic_id, bool)
            # OVRTX reserves zero for background/unlabelled pixels. Treating it
            # as a requested object ID would turn every background pixel into a
            # false-positive semantic match.
            or semantic_id <= 0
            or semantic_id in semantic_ids
            or not isinstance(pixels, int)
            or isinstance(pixels, bool)
            or pixels < 1
            or pixels > max_pixels
        ):
            return None
        matched[label] = (semantic_id, pixels)
        semantic_ids.add(semantic_id)
    if set(matched) != expected_labels:
        return None
    if sum(pixels for _, pixels in matched.values()) > max_pixels:
        return None
    return {label: semantic_id for label, (semantic_id, _) in matched.items()}


def positive_requested_semantic_ids(
    rendered_labels, expected_labels: set[str], *, max_pixels: int
) -> set[int] | None:
    """Validate per-label raster counts and return their unique semantic IDs."""

    semantic_id_map = positive_requested_semantic_id_map(
        rendered_labels,
        expected_labels,
        max_pixels=max_pixels,
    )
    return None if semantic_id_map is None else set(semantic_id_map.values())


def _newton_ovrtx_parity(
    camera_path: str,
    *,
    analytic_observation,
    scene,
    aovs: dict,
    artifact_map: dict[str, str],
    semantic_labels: dict[str, str],
) -> dict:
    """Compare exact independent-camera rasters with declared edge tolerances."""

    import numpy as np

    depth_path = artifact_map.get("DistanceToCameraSD")
    semantic_path = artifact_map.get("SemanticSegmentation")
    if not depth_path or not semantic_path:
        raise RuntimeError("OVRTX parity requires metric and semantic raster artifacts")
    ovrtx_depth = np.load(depth_path, allow_pickle=False)
    ovrtx_semantic = np.load(semantic_path, allow_pickle=False)
    if ovrtx_depth.ndim != 3 or ovrtx_depth.shape[-1] != 1:
        raise RuntimeError(f"OVRTX depth artifact for {camera_path} has invalid shape")
    if ovrtx_semantic.ndim != 3 or ovrtx_semantic.shape[-1] != 1:
        raise RuntimeError(
            f"OVRTX semantic artifact for {camera_path} has invalid shape"
        )
    ovrtx_depth = ovrtx_depth[..., 0].astype(np.float64, copy=False)
    ovrtx_semantic = ovrtx_semantic[..., 0].astype(np.uint32, copy=False)
    if ovrtx_semantic.shape != ovrtx_depth.shape:
        raise RuntimeError(
            f"OVRTX depth and semantic raster shapes differ for {camera_path}"
        )
    analytic_depth = getattr(analytic_observation, "depth_m", None)
    analytic_shape_ids = getattr(analytic_observation, "shape_ids", None)
    if analytic_depth is None or analytic_shape_ids is None:
        raise RuntimeError("Newton parity observation omitted depth or shape IDs")
    analytic_depth = np.asarray(analytic_depth)
    analytic_shape_ids = np.asarray(analytic_shape_ids)
    if (
        analytic_depth.shape[:1] != (1,)
        or analytic_shape_ids.shape != analytic_depth.shape
    ):
        raise RuntimeError("Newton parity observation has invalid camera dimensions")
    analytic_depth = analytic_depth[0].astype(np.float64, copy=False)
    analytic_shape_ids = analytic_shape_ids[0].astype(np.int64, copy=False)
    if analytic_depth.shape != ovrtx_depth.shape:
        raise RuntimeError(
            f"Newton/OVRTX parity raster shape differs for {camera_path}: "
            f"{analytic_depth.shape} vs {ovrtx_depth.shape}"
        )

    ovrtx_valid = (
        np.isfinite(ovrtx_depth) & (ovrtx_depth > 0.0) & (ovrtx_depth < 1.0e30)
    )
    analytic_valid = (
        np.isfinite(analytic_depth) & (analytic_depth > 0.0) & (analytic_shape_ids >= 0)
    )
    intersection = ovrtx_valid & analytic_valid
    union = ovrtx_valid | analytic_valid
    if not np.any(intersection):
        raise RuntimeError(
            f"Newton and OVRTX have no common finite depth samples for {camera_path}"
        )
    mask_iou = float(np.count_nonzero(intersection) / np.count_nonzero(union))
    errors = np.abs(ovrtx_depth[intersection] - analytic_depth[intersection])
    median_distance = float(np.median(ovrtx_depth[intersection]))
    mean_tolerance, percentile_tolerance = parity_v1_depth_tolerances(median_distance)
    mean_abs = float(np.mean(errors))
    rms = float(math.sqrt(float(np.mean(errors * errors))))
    percentile_95 = float(np.percentile(errors, 95.0))
    depth_passed = bool(
        mask_iou >= PARITY_V1_MINIMUM_MASK_IOU
        and mean_abs <= mean_tolerance
        and percentile_95 <= percentile_tolerance
    )

    rendered_labels = aovs["SemanticSegmentation"]["statistics"].get(
        "semantic_labels", []
    )
    expected_rendered = {f"usd_cli: {label};" for label in semantic_labels.values()}
    semantic_requested = bool(semantic_labels)
    semantic_iou: float | None = None
    semantic_passed = not semantic_requested
    ovrtx_target_pixels = 0
    newton_target_pixels = 0
    intersection_pixels = 0
    union_pixels = 0
    per_label_records: list[dict] = []
    if semantic_requested:
        semantic_id_map = positive_requested_semantic_id_map(
            rendered_labels,
            expected_rendered,
            max_pixels=int(ovrtx_semantic.size),
        )
        if semantic_id_map is None:
            semantic_passed = False
        else:
            semantic_ids = set(semantic_id_map.values())
            ovrtx_mask = np.isin(
                ovrtx_semantic, np.asarray(sorted(semantic_ids), dtype=np.uint32)
            )
            relevant_shape_ids = {
                shape_id
                for shape_id, path in scene.shape_path_by_id.items()
                if any(
                    is_path_at_or_below(path, semantic_path)
                    for semantic_path in semantic_labels
                )
            }
            analytic_mask = np.isin(
                analytic_shape_ids,
                np.asarray(sorted(relevant_shape_ids), dtype=np.int64),
            )
            semantic_union = ovrtx_mask | analytic_mask
            semantic_intersection = ovrtx_mask & analytic_mask
            ovrtx_target_pixels = int(np.count_nonzero(ovrtx_mask))
            newton_target_pixels = int(np.count_nonzero(analytic_mask))
            intersection_pixels = int(np.count_nonzero(semantic_intersection))
            union_pixels = int(np.count_nonzero(semantic_union))
            semantic_iou = (
                float(intersection_pixels / union_pixels) if union_pixels else 0.0
            )
            for requested_path, label in sorted(semantic_labels.items()):
                rendered_label = f"usd_cli: {label};"
                semantic_id = semantic_id_map[rendered_label]
                # Descendant assignments override inherited USD semantics.  Their
                # pixels still belong to an ancestor target's logical mask, so union
                # their IDs when validating that ancestor while keeping exact-path
                # masks for every descendant assignment.
                included_semantic_ids = {
                    semantic_id_map[f"usd_cli: {candidate_label};"]
                    for candidate_path, candidate_label in semantic_labels.items()
                    if is_path_at_or_below(candidate_path, requested_path)
                }
                label_ovrtx_mask = np.isin(
                    ovrtx_semantic,
                    np.asarray(sorted(included_semantic_ids), dtype=np.uint32),
                )
                label_shape_ids = {
                    shape_id
                    for shape_id, path in scene.shape_path_by_id.items()
                    if is_path_at_or_below(path, requested_path)
                }
                label_analytic_mask = np.isin(
                    analytic_shape_ids,
                    np.asarray(sorted(label_shape_ids), dtype=np.int64),
                )
                label_union = label_ovrtx_mask | label_analytic_mask
                label_intersection = label_ovrtx_mask & label_analytic_mask
                label_ovrtx_pixels = int(np.count_nonzero(label_ovrtx_mask))
                label_newton_pixels = int(np.count_nonzero(label_analytic_mask))
                label_intersection_pixels = int(np.count_nonzero(label_intersection))
                label_union_pixels = int(np.count_nonzero(label_union))
                label_iou = (
                    float(label_intersection_pixels / label_union_pixels)
                    if label_union_pixels
                    else 0.0
                )
                label_passed = bool(
                    label_ovrtx_pixels > 0
                    and label_newton_pixels > 0
                    and label_union_pixels > 0
                    and label_iou >= PARITY_V1_MINIMUM_MASK_IOU
                )
                per_label_records.append(
                    {
                        "path": requested_path,
                        "label": label,
                        "semantic_id": semantic_id,
                        "mask_iou": label_iou,
                        "minimum_mask_iou": PARITY_V1_MINIMUM_MASK_IOU,
                        "ovrtx_target_pixels": label_ovrtx_pixels,
                        "newton_target_pixels": label_newton_pixels,
                        "intersection_pixels": label_intersection_pixels,
                        "union_pixels": label_union_pixels,
                        "passed": label_passed,
                    }
                )
            semantic_passed = bool(
                per_label_records
                and all(record["passed"] for record in per_label_records)
            )

    return {
        "schema": NEWTON_OVRTX_PARITY_SCHEMA,
        "camera": camera_path,
        "raster_shape": [int(value) for value in ovrtx_depth.shape],
        "depth": {
            "ovrtx_valid_pixels": int(np.count_nonzero(ovrtx_valid)),
            "newton_valid_pixels": int(np.count_nonzero(analytic_valid)),
            "valid_mask_iou": mask_iou,
            "median_distance_m": median_distance,
            "mean_abs_error_m": mean_abs,
            "rms_error_m": rms,
            "percentile_95_error_m": percentile_95,
            "max_error_m": float(np.max(errors)),
            "mean_abs_tolerance_m": mean_tolerance,
            "percentile_95_tolerance_m": percentile_tolerance,
            "minimum_valid_mask_iou": PARITY_V1_MINIMUM_MASK_IOU,
            "passed": depth_passed,
        },
        "semantics": {
            "requested": semantic_requested,
            "mask_iou": semantic_iou,
            "minimum_mask_iou": PARITY_V1_MINIMUM_MASK_IOU,
            "ovrtx_target_pixels": ovrtx_target_pixels,
            "newton_target_pixels": newton_target_pixels,
            "intersection_pixels": intersection_pixels,
            "union_pixels": union_pixels,
            "labels": per_label_records,
            "passed": semantic_passed,
        },
        "passed": depth_passed and semantic_passed,
        "tolerance_policy": PARITY_V1_TOLERANCE_POLICY,
    }


def _observation_artifacts_and_records(
    results,
    *,
    requested: list[str],
    camera_definitions: list[dict],
    width: int,
    height: int,
    directory: Path,
    artifact_base: Path,
    semantic_labels: dict[str, str],
    artifact_aovs: tuple[str, ...],
    analytic_observations: dict[str, object] | None = None,
    scene=None,
) -> tuple[list[dict], list[dict], list[str], bool, bool, list[dict], bool]:
    """Validate exact observation output and bind compact records/artifacts."""

    if len(results) != len(requested):
        raise RuntimeError(
            f"OVRTX verification returned {len(results)} observation(s) for "
            f"{len(requested)} cameras"
        )
    by_camera: dict[str, object] = {}
    for result in results:
        camera = str(_result_value(result, "camera", ""))
        if camera not in requested or camera in by_camera:
            raise RuntimeError(
                f"OVRTX verification returned an unexpected or duplicate camera: {camera!r}"
            )
        by_camera[camera] = result
    missing = [path for path in requested if path not in by_camera]
    if missing:
        raise RuntimeError(f"OVRTX verification omitted camera {missing[0]}")

    definition_by_path = {item["path"]: item for item in camera_definitions}
    expected_assignments = [
        {"path": path, "label": semantic_labels[path]}
        for path in sorted(semantic_labels)
    ]
    expected_rendered_labels = {
        f"usd_cli: {label};" for label in semantic_labels.values()
    }
    artifacts: list[dict] = []
    observations: list[dict] = []
    paths: list[str] = []
    relative_artifact_paths: set[str] = set()
    render_products: set[str] = set()
    parity_records: list[dict] = []
    metric_verified = True
    semantic_verified = bool(semantic_labels)
    parity_verified = analytic_observations is not None
    from usd_core.imaging import is_blank_suspect

    for camera_path in requested:
        result = by_camera[camera_path]
        aovs = _result_value(result, "aovs")
        if not isinstance(aovs, dict) or set(aovs) != set(OBSERVATION_AOVS):
            raise RuntimeError(
                f"OVRTX verification returned an incomplete AOV set for {camera_path}"
            )
        # Detach the backend-owned records before retaining them in evidence and fail
        # before artifact publication if a summary is not finite JSON.
        aovs = json.loads(canonical_json_bytes(aovs))
        assignments = list(_result_value(result, "semantic_assignments", ()))
        if assignments != expected_assignments:
            raise RuntimeError(
                f"OVRTX verification semantic assignments changed for {camera_path}"
            )
        for name, record in aovs.items():
            if not isinstance(record, dict) or not isinstance(
                record.get("statistics"), dict
            ):
                raise RuntimeError(
                    f"OVRTX verification returned a malformed {name} record"
                )
            expected_reduction = (
                "cpu_semantic_metadata_decode_v1"
                if name == "SemanticIdMap"
                else "warp_cuda_dlpack_v1"
            )
            if record["statistics"].get("reduction") != expected_reduction:
                raise RuntimeError(
                    f"OVRTX verification {name} used an unqualified reduction"
                )
            shape = record.get("shape")
            if name == "SemanticIdMap":
                if not isinstance(shape, list) or not shape:
                    raise RuntimeError(
                        f"OVRTX verification {name} returned an invalid shape"
                    )
            elif shape != [height, width, OBSERVATION_IMAGE_CHANNELS[name]]:
                raise RuntimeError(
                    f"OVRTX verification {name} returned an invalid shape"
                )
            if not isinstance(record.get("dtype"), str) or not record["dtype"]:
                raise RuntimeError(
                    f"OVRTX verification {name} returned an invalid dtype"
                )
            if not has_exact_aov_element_count(record):
                raise RuntimeError(
                    f"OVRTX verification {name} returned an invalid element count"
                )
        distance = aovs["DistanceToCameraSD"]["statistics"]
        metric_verified = metric_verified and bool(
            has_positive_metric_distance_samples(
                distance,
                max_pixels=width * height,
            )
        )
        rendered_semantics = aovs["SemanticSegmentation"]["statistics"].get(
            "semantic_labels"
        )
        if semantic_labels:
            semantic_verified = bool(
                semantic_verified
                and positive_requested_semantic_ids(
                    rendered_semantics,
                    expected_rendered_labels,
                    max_pixels=width * height,
                )
                is not None
            )

        artifact_map = _result_value(result, "artifacts")
        if not isinstance(artifact_map, dict) or set(artifact_map) != set(
            artifact_aovs
        ):
            raise RuntimeError(
                f"OVRTX verification returned an unexpected artifact set for {camera_path}"
            )
        render_mode = _result_value(result, "ovrtx_render_mode")
        sensor_updates = _result_value(result, "ovrtx_num_sensor_updates")
        if not isinstance(render_mode, str) or not render_mode:
            raise RuntimeError("OVRTX verification result omitted its render mode")
        if (
            not isinstance(sensor_updates, int)
            or isinstance(sensor_updates, bool)
            or sensor_updates < 1
        ):
            raise RuntimeError(
                "OVRTX verification result omitted its sensor-update count"
            )
        for artifact_aov in artifact_aovs:
            path = Path(artifact_map[artifact_aov]).expanduser().resolve()
            if not path.is_file():
                raise RuntimeError(f"OVRTX verification artifact is missing: {path}")
            digest, size = sha256_file(path)
            if size <= 0:
                raise RuntimeError(f"OVRTX verification artifact is empty: {path}")
            blank = artifact_aov == "LdrColor" and is_blank_suspect(str(path))
            if blank:
                raise RuntimeError(
                    "OVRTX verification produced a blank-suspect camera view: "
                    f"{camera_path}"
                )
            relative_path = _relative_artifact(path, directory, artifact_base)
            if relative_path in relative_artifact_paths:
                raise RuntimeError(
                    f"OVRTX verification reused artifact path {relative_path!r}"
                )
            relative_artifact_paths.add(relative_path)
            artifacts.append(
                {
                    "camera": camera_path,
                    "camera_definition_digest": definition_by_path[camera_path][
                        "definition_digest"
                    ],
                    "aov": artifact_aov,
                    "relative_path": relative_path,
                    "sha256": digest,
                    "size_bytes": size,
                    "blank_suspect": False,
                    "render_settings": {
                        "ovrtx_render_mode": render_mode,
                        "ovrtx_num_sensor_updates": sensor_updates,
                        "active_aov": artifact_aov,
                        "requested_aovs": list(OBSERVATION_AOVS),
                        "observation_reduction": (
                            "cpu_semantic_metadata_decode_v1"
                            if artifact_aov == "SemanticIdMap"
                            else "warp_cuda_dlpack_v1"
                        ),
                    },
                }
            )
            paths.append(str(path))
        if analytic_observations is not None:
            if scene is None or camera_path not in analytic_observations:
                raise RuntimeError(
                    f"Newton parity observation is missing for {camera_path}"
                )
            parity = _newton_ovrtx_parity(
                camera_path,
                analytic_observation=analytic_observations[camera_path],
                scene=scene,
                aovs=aovs,
                artifact_map=artifact_map,
                semantic_labels=semantic_labels,
            )
            parity_records.append(parity)
            parity_verified = parity_verified and parity["passed"]
        render_product = str(_result_value(result, "render_product", ""))
        parsed_render_product = PurePosixPath(render_product)
        if (
            not parsed_render_product.is_absolute()
            or parsed_render_product.as_posix() != render_product
            or "\\" in render_product
            or render_product in render_products
        ):
            raise RuntimeError(
                f"OVRTX verification returned an invalid or duplicate render product: "
                f"{render_product!r}"
            )
        render_products.add(render_product)
        observations.append(
            {
                "camera": camera_path,
                "camera_definition_digest": definition_by_path[camera_path][
                    "definition_digest"
                ],
                "render_product": render_product,
                "aovs": aovs,
                "semantic_assignments": expected_assignments,
            }
        )
    if not metric_verified:
        raise RuntimeError(
            "OVRTX verification produced no positive finite DistanceToCameraSD samples for "
            "one or more cameras"
        )
    if semantic_labels and not semantic_verified:
        raise RuntimeError(
            "OVRTX verification could not bind one or more requested USD semantic labels"
        )
    if analytic_observations is not None and not parity_verified:
        failed = next(
            record["camera"] for record in parity_records if not record["passed"]
        )
        raise RuntimeError(
            f"Newton/OVRTX depth or semantic parity did not meet tolerance for {failed}"
        )
    return (
        artifacts,
        observations,
        paths,
        metric_verified,
        semantic_verified,
        parity_records,
        parity_verified,
    )


def verify_rig_with_ovrtx(
    stage,
    camera_paths: list[str],
    *,
    backend,
    resolution: tuple[int, int],
    output_dir: str | Path,
    source_digest: str,
    artifact_base_dir: str | Path | None = None,
    semantic_roles: dict[str, str] | None = None,
    configuration: dict | None = None,
    analytic_backend=None,
    analysis_scene=None,
    analysis_policy: SceneAnalysisPolicy | None = None,
) -> tuple[dict, list[str]]:
    """Collect qualified, digest-bound OVRTX evidence for every rig camera.

    A flattened in-memory clone prevents renderer overlays from touching the live
    stage. Local OVRTX uses the full typed observation protocol. A remote backend may
    use the narrow RGB compatibility scope only after its protocol supplies exact
    executing-worker identity; the bundled remote backend does not yet do so and is
    intentionally rejected by :func:`_runtime_identity`. When ``artifact_base_dir``
    is supplied it must be the future rig JSON's directory; artifact paths are then
    recorded relative to that directory so moving the JSON and evidence folder
    together preserves every reference.
    """

    from pxr import Usd

    check_cancelled()
    if getattr(backend, "name", None) not in {"ovrtx", "remote"}:
        raise RuntimeError("rig verification requires a local or remote OVRTX backend")
    if not isinstance(source_digest, str) or not _SHA256_RE.fullmatch(source_digest):
        raise ValueError("source_digest must be a prefixed SHA-256 digest")
    if not camera_paths:
        raise ValueError("rig verification requires at least one camera")
    requested = [str(path) for path in camera_paths]
    if any(not path.startswith("/") for path in requested):
        raise ValueError("rig verification camera paths must be absolute USD paths")
    if len(set(requested)) != len(requested):
        raise ValueError("rig verification camera paths must be unique")
    width, height = (int(resolution[0]), int(resolution[1]))
    if width < 1 or height < 1:
        raise ValueError("rig verification resolution must be positive")

    runtime_identity = _runtime_identity(backend)
    observer = getattr(backend, "observe", None)
    use_observation = getattr(backend, "name", None) == "ovrtx" and callable(observer)
    runtime_qualification = _qualified_runtime(
        runtime_identity, require_observation=use_observation
    )

    policy = _verification_policy(
        analysis_policy, analysis_scene, source_digest=source_digest
    )
    normalized_policy = policy.as_dict()
    analysis_time = dict(USD_DEFAULT_ANALYSIS_TIME)

    clone = Usd.Stage.Open(stage.Flatten())
    if clone is None:
        raise RuntimeError("could not clone the composed stage for OVRTX verification")
    clone_visibility_overlay = _apply_analysis_visibility_overlay(
        clone, analysis_scene, policy
    )
    camera_definitions = [_camera_definition(clone, path) for path in requested]
    stable_ids = [definition["stable_id"] for definition in camera_definitions]
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("rig verification camera stable IDs must be unique")
    detached_configuration = json.loads(canonical_json_bytes(configuration or {}))
    expanded_instance_roots: list[str] = []
    semantic_labels: dict[str, str] = {}
    if use_observation:
        expanded_instance_roots = _expand_semantic_instance_proxies(
            clone, semantic_roles
        )
        semantic_labels = _semantic_assignments(clone, semantic_roles)

    directory = Path(output_dir).expanduser().resolve()
    if artifact_base_dir is None:
        artifact_base = directory
        artifact_path_base = "verification_output_directory"
    else:
        artifact_base = Path(artifact_base_dir).expanduser().resolve()
        artifact_path_base = "rig_document_directory"
        try:
            directory.relative_to(artifact_base)
        except ValueError as exc:
            raise ValueError(
                "OVRTX evidence directory must be within the rig document directory"
            ) from exc
    directory.mkdir(parents=True, exist_ok=True)
    names = [
        f"camera_{index:03d}__{width}x{height}__verify"
        for index in range(1, len(requested) + 1)
    ]

    if use_observation:
        artifact_aovs = (
            ("LdrColor", "DistanceToCameraSD", "SemanticSegmentation")
            if analytic_backend is not None
            else ("LdrColor",)
        )
        analytic_observations: dict[str, object] | None = None
        if analytic_backend is not None:
            if analysis_scene is None or analysis_scene.source_digest != source_digest:
                raise ValueError(
                    "Newton/OVRTX parity requires the exact source SceneAnalysisIR"
                )
            from usd_core.camera_analysis.contracts import CameraBatchRequest
            from usd_core.camera_analysis.scene import camera_pose

            cameras_by_path = {
                camera.prim_path: camera for camera in analysis_scene.cameras
            }
            analytic_observations = {}
            for path in requested:
                camera = cameras_by_path.get(path)
                if camera is None:
                    raise ValueError(
                        f"Newton/OVRTX parity camera is absent from SceneAnalysisIR: {path}"
                    )
                analytic_observations[path] = analytic_backend.render_cameras(
                    CameraBatchRequest(
                        cameras=(camera_pose(camera),),
                        width=width,
                        height=height,
                        include_depth=True,
                        include_shape_ids=True,
                    )
                )
        assignments = [
            {
                "path": path,
                "role": str((semantic_roles or {}).get(path, "semantic")),
                "label": semantic_labels[path],
            }
            for path in sorted(semantic_labels)
        ]
        clone_instance_expansion = {
            "scope": "verification_clone_only",
            "operation": "set_instanceable_false",
            "expanded_roots": expanded_instance_roots,
        }
        request = {
            "schema": VERIFICATION_REQUEST_SCHEMA,
            "schema_version": 2,
            "source_digest": source_digest,
            "analysis_time": analysis_time,
            "resolution": {"width": width, "height": height},
            "camera_definitions": camera_definitions,
            "configuration": detached_configuration,
            "settings": {
                "mode": "quality",
                "requested_aovs": list(OBSERVATION_AOVS),
                "artifact_aovs": list(artifact_aovs),
                "scope": "ovrtx_metric_semantic_aov_evidence",
                "semantic_assignments": assignments,
                "camera_mapping": "one_render_product_per_camera",
                "clone_instance_expansion": clone_instance_expansion,
                "analysis_policy": normalized_policy,
                "analysis_policy_digest": policy.digest,
                "clone_visibility_overlay": clone_visibility_overlay,
            },
        }
        request_digest = canonical_json_digest(request)
        check_cancelled()
        with defer_cancellation():
            results = observer(
                clone,
                requested,
                width,
                height,
                directory,
                mode="quality",
                aovs=tuple(OBSERVATION_AOVS),
                artifact_aovs=artifact_aovs,
                semantic_labels=semantic_labels,
                names=names,
            )
        (
            artifacts,
            observations,
            paths,
            metric_verified,
            semantic_verified,
            parity_records,
            parity_verified,
        ) = _observation_artifacts_and_records(
            results,
            requested=requested,
            camera_definitions=camera_definitions,
            width=width,
            height=height,
            directory=directory,
            artifact_base=artifact_base,
            semantic_labels=semantic_labels,
            artifact_aovs=artifact_aovs,
            analytic_observations=analytic_observations,
            scene=analysis_scene,
        )
        evidence = {
            "schema": VERIFICATION_EVIDENCE_SCHEMA,
            "schema_version": 2,
            "status": "passed",
            "scope": "ovrtx_metric_semantic_aov_evidence",
            "source_digest": source_digest,
            "analysis_time": analysis_time,
            "request_digest": request_digest,
            "digest_algorithm": CANONICAL_JSON_DIGEST,
            "artifact_path_base": artifact_path_base,
            "resolution": {"width": width, "height": height},
            "camera_definitions": camera_definitions,
            "configuration": detached_configuration,
            "analysis_policy": normalized_policy,
            "analysis_policy_digest": policy.digest,
            "runtime_identity": runtime_identity,
            "runtime_qualification": runtime_qualification,
            "assertions": {
                "rendered_each_camera": True,
                "artifact_integrity_bound": True,
                "nonblank_views": True,
                "metric_distance_verified": metric_verified,
                "semantic_identity_verified": semantic_verified,
                "newton_ovrtx_parity_verified": parity_verified,
                "independent_camera_mapping_verified": True,
                "warp_dlpack_reduction_verified": True,
                "analysis_world_alignment_verified": True,
            },
            "limitations": (
                []
                if parity_verified
                else [
                    "Newton-to-OVRTX per-pixel parity was not requested",
                ]
            )
            + ["OVRTX tiled camera rendering is not enabled"],
            "clone_instance_expansion": clone_instance_expansion,
            "clone_visibility_overlay": clone_visibility_overlay,
            "semantic_assignments": assignments,
            "artifact_aovs": list(artifact_aovs),
            "observations": observations,
            "newton_ovrtx_parity": parity_records,
            "artifacts": artifacts,
        }
        evidence["evidence_digest"] = canonical_json_digest(evidence)
        return evidence, paths

    request = {
        "schema": VERIFICATION_REQUEST_SCHEMA,
        "schema_version": 1,
        "source_digest": source_digest,
        "analysis_time": analysis_time,
        "resolution": {"width": width, "height": height},
        "camera_definitions": camera_definitions,
        "settings": {
            "mode": "quality",
            "requested_aovs": ["LdrColor"],
            "scope": "ovrtx_rgb_artifact_grounding",
            "analysis_policy": normalized_policy,
            "analysis_policy_digest": policy.digest,
            "clone_visibility_overlay": clone_visibility_overlay,
        },
    }
    if detached_configuration:
        request["configuration"] = detached_configuration
    request_digest = canonical_json_digest(request)

    check_cancelled()
    with defer_cancellation():
        results = backend.render(
            clone,
            requested,
            width,
            height,
            directory,
            mode="quality",
            names=names,
        )
    if len(results) != len(requested):
        raise RuntimeError(
            f"OVRTX verification returned {len(results)} image(s) for "
            f"{len(requested)} cameras"
        )
    by_camera: dict[str, object] = {}
    for result in results:
        camera = str(getattr(result, "camera", ""))
        if camera not in set(requested):
            raise RuntimeError(
                f"OVRTX verification returned an unexpected camera: {camera!r}"
            )
        if camera in by_camera:
            raise RuntimeError(
                f"OVRTX verification returned camera {camera} more than once"
            )
        by_camera[camera] = result
    missing = [path for path in requested if path not in by_camera]
    if missing:
        raise RuntimeError(f"OVRTX verification omitted camera {missing[0]}")

    definition_by_path = {item["path"]: item for item in camera_definitions}
    artifacts: list[dict] = []
    paths: list[str] = []
    relative_artifact_paths: set[str] = set()
    for camera_path in requested:
        result = by_camera[camera_path]
        path = Path(getattr(result, "path", "")).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"OVRTX verification artifact is missing: {path}")
        digest, size = sha256_file(path)
        if size <= 0:
            raise RuntimeError(f"OVRTX verification artifact is empty: {path}")
        render_mode = getattr(result, "ovrtx_render_mode", None)
        sensor_updates = getattr(result, "ovrtx_num_sensor_updates", None)
        active_aov = getattr(result, "active_aov", None)
        if not isinstance(render_mode, str) or not render_mode:
            raise RuntimeError("OVRTX verification result omitted its render mode")
        if (
            not isinstance(sensor_updates, int)
            or isinstance(sensor_updates, bool)
            or sensor_updates < 1
        ):
            raise RuntimeError(
                "OVRTX verification result omitted its sensor-update count"
            )
        if active_aov != "LdrColor":
            raise RuntimeError(
                f"OVRTX RGB verification expected LdrColor, received {active_aov!r}"
            )
        blank = bool(getattr(result, "blank_suspect", False))
        relative_path = _relative_artifact(path, directory, artifact_base)
        if relative_path in relative_artifact_paths:
            raise RuntimeError(
                f"OVRTX verification reused artifact path {relative_path!r}"
            )
        relative_artifact_paths.add(relative_path)
        artifacts.append(
            {
                "camera": camera_path,
                "camera_definition_digest": definition_by_path[camera_path][
                    "definition_digest"
                ],
                "relative_path": relative_path,
                "sha256": digest,
                "size_bytes": size,
                "blank_suspect": blank,
                "render_settings": {
                    "ovrtx_render_mode": render_mode,
                    "ovrtx_num_sensor_updates": sensor_updates,
                    "active_aov": active_aov,
                },
                **(
                    {"renderer_identity": result.renderer_identity}
                    if getattr(result, "renderer_identity", None) is not None
                    else {}
                ),
            }
        )
        paths.append(str(path))
    if any(item["blank_suspect"] for item in artifacts):
        raise RuntimeError(
            "OVRTX verification produced one or more blank-suspect camera views"
        )

    evidence = {
        "schema": VERIFICATION_EVIDENCE_SCHEMA,
        "schema_version": 1,
        "status": "passed",
        "scope": "ovrtx_rgb_artifact_grounding",
        "source_digest": source_digest,
        "analysis_time": analysis_time,
        "request_digest": request_digest,
        "digest_algorithm": CANONICAL_JSON_DIGEST,
        "artifact_path_base": artifact_path_base,
        "resolution": {"width": width, "height": height},
        "camera_definitions": camera_definitions,
        "analysis_policy": normalized_policy,
        "analysis_policy_digest": policy.digest,
        **({"configuration": detached_configuration} if detached_configuration else {}),
        "runtime_identity": runtime_identity,
        "runtime_qualification": runtime_qualification,
        "assertions": {
            "rendered_each_camera": True,
            "artifact_integrity_bound": True,
            "nonblank_views": True,
            "metric_distance_verified": False,
            "semantic_identity_verified": False,
            "newton_ovrtx_parity_verified": False,
            "analysis_world_alignment_verified": True,
        },
        "limitations": [
            "RGB grounding does not establish metric-distance agreement",
            "RGB grounding does not establish semantic-mask agreement",
            "OVRTX-to-Warp metric reductions require the OVRTX 0.4 observation worker",
        ],
        "clone_visibility_overlay": clone_visibility_overlay,
        "artifacts": artifacts,
    }
    evidence["evidence_digest"] = canonical_json_digest(evidence)
    return evidence, paths
