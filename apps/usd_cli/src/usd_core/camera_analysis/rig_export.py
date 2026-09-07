# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned, deterministic camera-rig interchange."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from usd_core.camera_analysis.authoring import RIG_SCHEMA, read_rig_metadata
from usd_core.camera_analysis.calibration import camera_calibration
from usd_core.camera_analysis.contracts import SceneAnalysisIR, SceneAnalysisPolicy
from usd_core.camera_analysis.evidence import (
    CAMERA_OBSERVATION_CAPABILITY,
    CANONICAL_JSON_DIGEST,
    NEWTON_OVRTX_PARITY_SCHEMA,
    OBSERVATION_AOVS,
    OBSERVATION_IMAGE_CHANNELS,
    OVSTAGE_ATTACHED_CAPABILITY,
    OVSTAGE_TRANSPORT,
    PARITY_V1_MINIMUM_MASK_IOU,
    PARITY_V1_TOLERANCE_POLICY,
    QUALIFIED_RUNTIME_VERSIONS,
    RGB_RENDER_CAPABILITY,
    SEMANTIC_OVERLAY_CAPABILITY,
    USD_DEFAULT_ANALYSIS_TIME,
    VERIFICATION_EVIDENCE_SCHEMA,
    VERIFICATION_REQUEST_SCHEMA,
    WARP_DLPACK_REDUCTION_CAPABILITY,
    WORKER_PROTOCOL_VERSION,
    analysis_overlay_actions,
    canonical_json_bytes,
    canonical_json_digest,
    has_exact_aov_element_count,
    has_positive_metric_distance_samples,
    parity_v1_depth_tolerances,
    positive_requested_semantic_id_map,
    semantic_label,
    sha256_file,
)
from usd_core.camera_analysis.scene import scene_analysis_policy_from_dict

RIG_DOCUMENT_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _matrix(value) -> list[list[float]]:
    return [[float(item) for item in row] for row in value]


def _derived_id(kind: str, path: str) -> str:
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
    return f"{kind}-path-sha256:{digest}"


def _visibility_by_camera(coverage_report: dict | None) -> dict[str, dict | None]:
    result: dict[str, dict | None] = {}
    for item in (coverage_report or {}).get("cameras", []):
        if not isinstance(item, dict) or not isinstance(item.get("camera"), str):
            raise ValueError("coverage report contains a malformed camera record")
        path = item["camera"]
        if path in result:
            raise ValueError(f"coverage report contains duplicate camera {path}")
        visibility = item.get("visibility")
        if visibility is not None and not isinstance(visibility, dict):
            raise ValueError(f"coverage visibility for {path} must be an object")
        result[path] = visibility
    return result


def _finite_number(
    value, *, minimum: float = 0.0, maximum: float | None = None
) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    number = float(value)
    return (
        math.isfinite(number)
        and number >= minimum
        and (maximum is None or number <= maximum)
    )


def _matches_parity_v1_minimum_mask_iou(value) -> bool:
    return bool(
        _finite_number(value, maximum=1.0)
        and math.isclose(
            float(value),
            PARITY_V1_MINIMUM_MASK_IOU,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    )


def _passing_semantic_mask_metrics(record: dict, raster_pixels: int) -> bool:
    """Recompute one declared semantic-mask IoU from its integer counts."""

    semantic_iou = record.get("mask_iou")
    semantic_minimum = record.get("minimum_mask_iou")
    ovrtx_pixels = record.get("ovrtx_target_pixels")
    newton_pixels = record.get("newton_target_pixels")
    intersection_pixels = record.get("intersection_pixels")
    union_pixels = record.get("union_pixels")
    pixel_counts = (
        ovrtx_pixels,
        newton_pixels,
        intersection_pixels,
        union_pixels,
    )
    counts_valid = all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= raster_pixels
        for value in pixel_counts
    )
    if counts_valid:
        counts_valid = bool(
            intersection_pixels <= ovrtx_pixels
            and intersection_pixels <= newton_pixels
            and intersection_pixels <= union_pixels
            and union_pixels == ovrtx_pixels + newton_pixels - intersection_pixels
        )
    return bool(
        _finite_number(semantic_iou, maximum=1.0)
        and _matches_parity_v1_minimum_mask_iou(semantic_minimum)
        and record.get("passed") is True
        and counts_valid
        and ovrtx_pixels > 0
        and newton_pixels > 0
        and union_pixels > 0
        and math.isclose(
            float(semantic_iou),
            intersection_pixels / union_pixels,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        and float(semantic_iou) >= float(semantic_minimum)
    )


def _passing_parity_record(
    parity: dict,
    camera_path: str,
    semantic_assignments: list[dict],
    rendered_semantic_ids: dict[str, int],
) -> bool:
    """Check that a passing claim is supported by its declared tolerance policy."""

    if (
        parity.get("schema") != NEWTON_OVRTX_PARITY_SCHEMA
        or parity.get("camera") != camera_path
        or parity.get("passed") is not True
        or parity.get("tolerance_policy") != PARITY_V1_TOLERANCE_POLICY
        or not isinstance(parity.get("raster_shape"), list)
        or len(parity["raster_shape"]) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in parity["raster_shape"]
        )
    ):
        return False
    depth = parity.get("depth")
    semantics = parity.get("semantics")
    if not isinstance(depth, dict) or not isinstance(semantics, dict):
        return False
    valid_mask_iou = depth.get("valid_mask_iou")
    minimum_iou = depth.get("minimum_valid_mask_iou")
    median_distance = depth.get("median_distance_m")
    mean_abs = depth.get("mean_abs_error_m")
    mean_tolerance = depth.get("mean_abs_tolerance_m")
    percentile = depth.get("percentile_95_error_m")
    percentile_tolerance = depth.get("percentile_95_tolerance_m")
    median_valid = bool(
        _finite_number(median_distance) and float(median_distance) > 0.0
    )
    if median_valid:
        expected_mean_tolerance, expected_percentile_tolerance = (
            parity_v1_depth_tolerances(float(median_distance))
        )
    else:
        expected_mean_tolerance = math.nan
        expected_percentile_tolerance = math.nan
    depth_passes = (
        _finite_number(valid_mask_iou, maximum=1.0)
        and _matches_parity_v1_minimum_mask_iou(minimum_iou)
        and median_valid
        and _finite_number(mean_abs)
        and _finite_number(mean_tolerance)
        and _finite_number(percentile)
        and _finite_number(percentile_tolerance)
        and math.isclose(
            float(mean_tolerance),
            expected_mean_tolerance,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(percentile_tolerance),
            expected_percentile_tolerance,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        and float(valid_mask_iou) >= float(minimum_iou)
        and float(mean_abs) <= float(mean_tolerance)
        and float(percentile) <= float(percentile_tolerance)
        and depth.get("passed") is True
    )
    semantic_requested = semantics.get("requested")
    semantic_minimum = semantics.get("minimum_mask_iou")
    raster_pixels = math.prod(parity["raster_shape"])
    labels = semantics.get("labels")
    if semantic_requested:
        expected_labels = [
            {"path": item["path"], "label": item["label"]}
            for item in semantic_assignments
        ]
        expected_rendered_labels = {
            f"usd_cli: {item['label']};" for item in semantic_assignments
        }
        semantic_ids: set[int] = set()
        labels_pass = bool(
            isinstance(labels, list)
            and len(labels) == len(expected_labels)
            and set(rendered_semantic_ids) == expected_rendered_labels
        )
        if labels_pass:
            for label_record, expected in zip(labels, expected_labels, strict=True):
                semantic_id = (
                    label_record.get("semantic_id")
                    if isinstance(label_record, dict)
                    else None
                )
                expected_semantic_id = rendered_semantic_ids.get(
                    f"usd_cli: {expected['label']};"
                )
                if (
                    not isinstance(label_record, dict)
                    or label_record.get("path") != expected["path"]
                    or label_record.get("label") != expected["label"]
                    or not isinstance(semantic_id, int)
                    or isinstance(semantic_id, bool)
                    or semantic_id <= 0
                    or semantic_id in semantic_ids
                    or semantic_id != expected_semantic_id
                    or not _matches_parity_v1_minimum_mask_iou(
                        label_record.get("minimum_mask_iou")
                    )
                    or not _matches_parity_v1_minimum_mask_iou(semantic_minimum)
                    or not math.isclose(
                        float(label_record["minimum_mask_iou"]),
                        float(semantic_minimum),
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                    or not _passing_semantic_mask_metrics(label_record, raster_pixels)
                ):
                    labels_pass = False
                    break
                semantic_ids.add(semantic_id)
        semantic_passes = bool(
            isinstance(semantic_requested, bool)
            and semantic_requested is bool(semantic_assignments)
            and _matches_parity_v1_minimum_mask_iou(semantic_minimum)
            and _passing_semantic_mask_metrics(semantics, raster_pixels)
            and labels_pass
        )
    else:
        aggregate_counts = (
            semantics.get("ovrtx_target_pixels"),
            semantics.get("newton_target_pixels"),
            semantics.get("intersection_pixels"),
            semantics.get("union_pixels"),
        )
        semantic_passes = bool(
            isinstance(semantic_requested, bool)
            and semantic_requested is bool(semantic_assignments)
            and not rendered_semantic_ids
            and _matches_parity_v1_minimum_mask_iou(semantic_minimum)
            and semantics.get("mask_iou") is None
            and all(value == 0 for value in aggregate_counts)
            and labels == []
            and semantics.get("passed") is True
        )
    return bool(depth_passes and semantic_passes)


def _validated_policy_binding(
    value,
    digest,
    *,
    expected: SceneAnalysisPolicy | None = None,
    context: str,
) -> SceneAnalysisPolicy:
    try:
        policy = scene_analysis_policy_from_dict(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has a malformed analysis policy") from exc
    if policy.as_dict() != value:
        raise ValueError(f"{context} analysis policy is not canonical")
    if digest != policy.digest:
        raise ValueError(f"{context} analysis policy digest does not match")
    if expected is not None and policy != expected:
        raise ValueError(f"{context} analysis policy differs from the exported rig")
    return policy


def _validated_visibility_overlay(value, *, policy_digest: str) -> dict:
    from pxr import Sdf

    if (
        not isinstance(value, dict)
        or set(value)
        != {"scope", "policy_digest", "actions", "expanded_instance_roots"}
        or value.get("scope") != "verification_clone_only"
        or value.get("policy_digest") != policy_digest
        or not isinstance(value.get("actions"), list)
        or not isinstance(value.get("expanded_instance_roots"), list)
    ):
        raise ValueError("verification evidence has an invalid visibility overlay")

    action_paths: list[Sdf.Path] = []
    for action in value["actions"]:
        path = action.get("path") if isinstance(action, dict) else None
        if (
            not isinstance(action, dict)
            or set(action) != {"path", "operation", "reason"}
            or not isinstance(path, str)
            or not Sdf.Path.IsValidPathString(path)
        ):
            raise ValueError("verification evidence has an invalid visibility action")
        parsed = Sdf.Path(path)
        operation_reason = (action.get("operation"), action.get("reason"))
        if (
            not parsed.IsAbsolutePath()
            or not parsed.IsPrimPath()
            or parsed.pathString != path
            or operation_reason
            not in {
                ("set_visibility_invisible", "helper_geometry"),
                ("set_visibility_invisible", "absent_from_analysis_world"),
                ("set_double_sided_true", "analytic_meshes_are_two_sided"),
            }
        ):
            raise ValueError("verification evidence has an invalid visibility action")
        action_paths.append(parsed)
    serialized_actions = [
        (action["path"], action["operation"]) for action in value["actions"]
    ]
    if serialized_actions != sorted(set(serialized_actions)):
        raise ValueError("verification visibility actions are duplicated or unordered")

    expanded_roots: list[Sdf.Path] = []
    for value_path in value["expanded_instance_roots"]:
        if not isinstance(value_path, str) or not Sdf.Path.IsValidPathString(
            value_path
        ):
            raise ValueError(
                "verification visibility overlay has invalid instance roots"
            )
        parsed = Sdf.Path(value_path)
        if (
            not parsed.IsAbsolutePath()
            or not parsed.IsPrimPath()
            or parsed.pathString != value_path
            or not any(
                path != parsed and path.HasPrefix(parsed) for path in action_paths
            )
        ):
            raise ValueError(
                "verification visibility overlay has invalid instance roots"
            )
        expanded_roots.append(parsed)
    if len({path.pathString for path in expanded_roots}) != len(expanded_roots):
        raise ValueError("verification visibility overlay has duplicate instance roots")
    return value


def _verification_record(
    verification: dict | None,
    *,
    source_digest: str,
    resolution: tuple[int, int],
    camera_paths: list[str],
    camera_ids: list[str],
    analysis_policy: SceneAnalysisPolicy,
    expected_overlay_actions: list[dict],
) -> dict:
    if verification is None:
        return {
            "schema": VERIFICATION_EVIDENCE_SCHEMA,
            "schema_version": 1,
            "status": "not_requested",
        }
    if not isinstance(verification, dict):
        raise ValueError("verification evidence must be an object")
    # Work on a detached JSON value so callers cannot mutate the published document.
    record = json.loads(canonical_json_bytes(verification))
    if record.get("schema") != VERIFICATION_EVIDENCE_SCHEMA:
        raise ValueError("verification evidence has an unsupported schema")
    evidence_version = record.get("schema_version")
    if (
        not isinstance(evidence_version, int)
        or isinstance(evidence_version, bool)
        or evidence_version not in {1, 2}
    ):
        raise ValueError("verification evidence has an unsupported schema version")
    if record.get("status") != "passed":
        raise ValueError("verification evidence is not a passing record")
    expected_scope = (
        "ovrtx_metric_semantic_aov_evidence"
        if evidence_version == 2
        else "ovrtx_rgb_artifact_grounding"
    )
    if record.get("scope") != expected_scope:
        raise ValueError("verification evidence has an unsupported assertion scope")
    if record.get("digest_algorithm") != CANONICAL_JSON_DIGEST:
        raise ValueError("verification evidence uses an unsupported digest algorithm")
    if record.get("artifact_path_base") != "rig_document_directory":
        raise ValueError(
            "verification artifact paths are not relative to the rig document"
        )
    if record.get("source_digest") != source_digest:
        raise ValueError("verification evidence is bound to a different source stage")
    analysis_time = record.get("analysis_time")
    if analysis_time != USD_DEFAULT_ANALYSIS_TIME:
        raise ValueError(
            "verification evidence is not bound to USD Default analysis time"
        )
    evidence_policy = _validated_policy_binding(
        record.get("analysis_policy"),
        record.get("analysis_policy_digest"),
        expected=analysis_policy,
        context="verification evidence",
    )
    clone_visibility_overlay = _validated_visibility_overlay(
        record.get("clone_visibility_overlay"),
        policy_digest=evidence_policy.digest,
    )
    if clone_visibility_overlay["actions"] != expected_overlay_actions:
        raise ValueError(
            "verification clone actions differ from the exported analysis world"
        )
    expected_resolution = {"width": int(resolution[0]), "height": int(resolution[1])}
    if record.get("resolution") != expected_resolution:
        raise ValueError("verification evidence resolution differs from the rig export")
    definitions = record.get("camera_definitions")
    if (
        not isinstance(definitions, list)
        or len(definitions) != len(camera_paths)
        or not all(isinstance(item, dict) for item in definitions)
    ):
        raise ValueError("verification evidence omits its camera definitions")
    verified_paths = [item.get("path") for item in definitions]
    if verified_paths != camera_paths:
        raise ValueError("verification evidence cameras differ from the exported rig")
    stable_ids = [item.get("stable_id") for item in definitions]
    if not all(isinstance(item, str) and item for item in stable_ids) or len(
        set(stable_ids)
    ) != len(stable_ids):
        raise ValueError("verification evidence camera stable IDs are invalid")
    if stable_ids != camera_ids:
        raise ValueError(
            "verification evidence camera stable IDs differ from the exported rig"
        )
    definition_digests: dict[str, str] = {}
    for definition in definitions:
        payload = dict(definition)
        definition_digest = payload.pop("definition_digest", None)
        if (
            not isinstance(definition_digest, str)
            or not _SHA256_RE.fullmatch(definition_digest)
            or definition_digest != canonical_json_digest(payload)
        ):
            raise ValueError(
                "verification evidence contains a changed camera definition"
            )
        definition_digests[definition["path"]] = definition_digest

    has_configuration = "configuration" in record
    configuration = record.get("configuration", {})
    if not isinstance(configuration, dict):
        raise ValueError("verification evidence configuration must be an object")
    artifact_aovs = ["LdrColor"]
    semantic_assignments: list[dict] = []
    clone_instance_expansion: dict = {}
    if evidence_version == 2:
        from pxr import Sdf

        artifact_aovs = record.get("artifact_aovs")
        if (
            not isinstance(artifact_aovs, list)
            or not artifact_aovs
            or artifact_aovs[0] != "LdrColor"
            or not all(isinstance(item, str) for item in artifact_aovs)
            or len(set(artifact_aovs)) != len(artifact_aovs)
            or not set(artifact_aovs).issubset(set(OBSERVATION_AOVS))
        ):
            raise ValueError("observation evidence has an invalid artifact AOV set")
        semantic_assignments = record.get("semantic_assignments")
        if not isinstance(semantic_assignments, list):
            raise ValueError("observation evidence omits semantic assignments")
        assignment_paths: list[str] = []
        parsed_assignments: list[Sdf.Path] = []
        for assignment in semantic_assignments:
            if not isinstance(assignment, dict):
                raise ValueError(
                    "observation evidence has a malformed semantic assignment"
                )
            path = assignment.get("path")
            role = assignment.get("role")
            label = assignment.get("label")
            if (
                not isinstance(path, str)
                or not Sdf.Path.IsValidPathString(path)
                or not Sdf.Path(path).IsAbsolutePath()
                or not Sdf.Path(path).IsPrimPath()
                or Sdf.Path(path).pathString != path
                or not isinstance(role, str)
                or not isinstance(label, str)
                or label != semantic_label(path, role)
            ):
                raise ValueError(
                    "observation evidence has a malformed semantic assignment"
                )
            assignment_paths.append(path)
            parsed_assignments.append(Sdf.Path(path))
        if assignment_paths != sorted(set(assignment_paths)):
            raise ValueError(
                "observation evidence semantic assignments are duplicated or unordered"
            )
        clone_instance_expansion = record.get("clone_instance_expansion")
        if (
            not isinstance(clone_instance_expansion, dict)
            or set(clone_instance_expansion) != {"scope", "operation", "expanded_roots"}
            or clone_instance_expansion.get("scope") != "verification_clone_only"
            or clone_instance_expansion.get("operation") != "set_instanceable_false"
        ):
            raise ValueError(
                "observation evidence has invalid clone instance expansion settings"
            )
        expanded_roots = clone_instance_expansion.get("expanded_roots")
        if not isinstance(expanded_roots, list):
            raise ValueError("observation evidence has invalid expanded instance roots")
        parsed_roots: list[Sdf.Path] = []
        for root in expanded_roots:
            if not isinstance(root, str) or not Sdf.Path.IsValidPathString(root):
                raise ValueError(
                    "observation evidence has invalid expanded instance roots"
                )
            parsed = Sdf.Path(root)
            if (
                not parsed.IsAbsolutePath()
                or not parsed.IsPrimPath()
                or parsed.pathString != root
                or not any(
                    path != parsed and path.HasPrefix(parsed)
                    for path in parsed_assignments
                )
            ):
                raise ValueError(
                    "observation evidence has invalid expanded instance roots"
                )
            parsed_roots.append(parsed)
        if len({path.pathString for path in parsed_roots}) != len(parsed_roots):
            raise ValueError(
                "observation evidence has duplicate expanded instance roots"
            )
        if any(
            current.HasPrefix(later)
            for index, current in enumerate(parsed_roots)
            for later in parsed_roots[index + 1 :]
        ):
            raise ValueError(
                "observation evidence expanded nested instance roots out of order"
            )
    request = {
        "schema": VERIFICATION_REQUEST_SCHEMA,
        "schema_version": evidence_version,
        "source_digest": source_digest,
        "analysis_time": analysis_time,
        "resolution": expected_resolution,
        "camera_definitions": definitions,
        "settings": (
            {
                "mode": "quality",
                "requested_aovs": list(OBSERVATION_AOVS),
                "artifact_aovs": artifact_aovs,
                "scope": expected_scope,
                "semantic_assignments": semantic_assignments,
                "camera_mapping": "one_render_product_per_camera",
                "clone_instance_expansion": clone_instance_expansion,
                "analysis_policy": evidence_policy.as_dict(),
                "analysis_policy_digest": evidence_policy.digest,
                "clone_visibility_overlay": clone_visibility_overlay,
            }
            if evidence_version == 2
            else {
                "mode": "quality",
                "requested_aovs": ["LdrColor"],
                "scope": expected_scope,
                "analysis_policy": evidence_policy.as_dict(),
                "analysis_policy_digest": evidence_policy.digest,
                "clone_visibility_overlay": clone_visibility_overlay,
            }
        ),
    }
    if has_configuration:
        request["configuration"] = configuration
    if record.get("request_digest") != canonical_json_digest(request):
        raise ValueError("verification request digest does not match its contents")

    identity = record.get("runtime_identity")
    if not isinstance(identity, dict):
        raise ValueError("verification evidence omits its worker runtime identity")
    reported_versions = identity.get("runtime_versions")
    capabilities = identity.get("capabilities")
    if (
        not isinstance(reported_versions, dict)
        or not isinstance(capabilities, list)
        or not all(isinstance(item, str) for item in capabilities)
    ):
        raise ValueError("verification evidence contains a malformed worker identity")
    normalized_versions = {
        "ovrtx": str(reported_versions.get("ovrtx", "unreported")),
        "ovstage": str(reported_versions.get("ovstage", "unreported")),
        "warp": str(
            reported_versions.get(
                "warp", reported_versions.get("warp-lang", "unreported")
            )
        ),
    }
    required_capabilities = {
        RGB_RENDER_CAPABILITY,
        OVSTAGE_ATTACHED_CAPABILITY,
    }
    if evidence_version == 2:
        required_capabilities.update(
            {
                CAMERA_OBSERVATION_CAPABILITY,
                WARP_DLPACK_REDUCTION_CAPABILITY,
                SEMANTIC_OVERLAY_CAPABILITY,
            }
        )
    if (
        normalized_versions != QUALIFIED_RUNTIME_VERSIONS
        or identity.get("worker_protocol_version") != WORKER_PROTOCOL_VERSION
        or identity.get("stage_transport") != OVSTAGE_TRANSPORT
        or not required_capabilities.issubset(set(capabilities))
    ):
        raise ValueError(
            "verification evidence worker is not a qualified OVRTX runtime"
        )
    qualification = record.get("runtime_qualification")
    qualification_capabilities = (
        qualification.get("required_capabilities")
        if isinstance(qualification, dict)
        else None
    )
    expected_checks = {
        *QUALIFIED_RUNTIME_VERSIONS,
        *required_capabilities,
        "worker_protocol_version",
        "stage_transport",
    }
    if (
        not isinstance(qualification, dict)
        or qualification.get("passed") is not True
        or not isinstance(qualification.get("checks"), dict)
        or set(qualification["checks"]) != expected_checks
        or not all(value is True for value in qualification["checks"].values())
        or qualification.get("required_versions") != QUALIFIED_RUNTIME_VERSIONS
        or qualification.get("reported_versions") != normalized_versions
        or qualification.get("required_worker_protocol_version")
        != WORKER_PROTOCOL_VERSION
        or qualification.get("reported_worker_protocol_version")
        != WORKER_PROTOCOL_VERSION
        or qualification.get("required_stage_transport") != OVSTAGE_TRANSPORT
        or qualification.get("reported_stage_transport") != OVSTAGE_TRANSPORT
        or not isinstance(qualification_capabilities, list)
        or not all(isinstance(item, str) for item in qualification_capabilities)
        or set(qualification_capabilities) != required_capabilities
        or qualification.get("reported_capabilities")
        != sorted(str(item) for item in capabilities)
    ):
        raise ValueError("verification evidence runtime qualification did not pass")

    artifacts = record.get("artifacts")
    expected_artifacts = [
        (camera, aov) for camera in camera_paths for aov in artifact_aovs
    ]
    if not isinstance(artifacts, list) or len(artifacts) != len(expected_artifacts):
        raise ValueError("verification evidence has an incomplete artifact set")
    artifact_paths: set[str] = set()
    for (expected_camera, expected_aov), artifact in zip(
        expected_artifacts, artifacts, strict=True
    ):
        if not isinstance(artifact, dict) or artifact.get("camera") != expected_camera:
            raise ValueError(
                "verification evidence artifacts differ from the exported rig"
            )
        if (
            artifact.get("camera_definition_digest")
            != definition_digests[expected_camera]
        ):
            raise ValueError(
                "verification artifact is bound to a different camera definition"
            )
        if evidence_version == 2 and artifact.get("aov") != expected_aov:
            raise ValueError("verification artifact AOV ordering is invalid")
        relative_path = artifact.get("relative_path")
        if not isinstance(relative_path, str):
            raise ValueError("verification artifact omits its relative path")
        parsed_path = PurePosixPath(relative_path)
        if (
            not relative_path
            or "\\" in relative_path
            or parsed_path.is_absolute()
            or parsed_path.as_posix() != relative_path
            or (parsed_path.parts and parsed_path.parts[0].endswith(":"))
            or relative_path in artifact_paths
            or any(part in {"", ".", ".."} for part in parsed_path.parts)
        ):
            raise ValueError(
                "verification artifact has an unsafe or duplicate relative path"
            )
        artifact_paths.add(relative_path)
        if not _SHA256_RE.fullmatch(str(artifact.get("sha256", ""))):
            raise ValueError("verification artifact omits its SHA-256 digest")
        if (
            not isinstance(artifact.get("size_bytes"), int)
            or isinstance(artifact.get("size_bytes"), bool)
            or artifact["size_bytes"] < 1
        ):
            raise ValueError("verification artifact has an invalid byte count")
        if artifact.get("blank_suspect") is not False:
            raise ValueError("verification artifact is blank-suspect")
        settings = artifact.get("render_settings")
        if (
            not isinstance(settings, dict)
            or settings.get("active_aov") != expected_aov
            or not isinstance(settings.get("ovrtx_render_mode"), str)
            or not settings["ovrtx_render_mode"]
            or not isinstance(settings.get("ovrtx_num_sensor_updates"), int)
            or isinstance(settings.get("ovrtx_num_sensor_updates"), bool)
            or settings["ovrtx_num_sensor_updates"] < 1
        ):
            raise ValueError("verification artifact has incomplete render settings")
        if evidence_version == 2 and (
            settings.get("requested_aovs") != list(OBSERVATION_AOVS)
            or settings.get("observation_reduction")
            != (
                "cpu_semantic_metadata_decode_v1"
                if expected_aov == "SemanticIdMap"
                else "warp_cuda_dlpack_v1"
            )
        ):
            raise ValueError(
                "verification artifact has incomplete observation settings"
            )

    assertions = record.get("assertions")
    if not isinstance(assertions, dict) or any(
        assertions.get(name) is not True
        for name in (
            "rendered_each_camera",
            "artifact_integrity_bound",
            "nonblank_views",
            "analysis_world_alignment_verified",
        )
    ):
        raise ValueError("verification evidence omits its proven RGB assertions")
    if evidence_version == 1:
        if any(
            assertions.get(name) is not False
            for name in (
                "metric_distance_verified",
                "semantic_identity_verified",
                "newton_ovrtx_parity_verified",
            )
        ):
            raise ValueError(
                "RGB evidence must not claim metric or semantic verification"
            )
    else:
        parity_records = record.get("newton_ovrtx_parity")
        parity_claimed = assertions.get("newton_ovrtx_parity_verified")
        if not isinstance(parity_claimed, bool):
            raise ValueError("observation evidence has an invalid parity assertion")
        if (
            not isinstance(parity_records, list)
            or (parity_claimed and len(parity_records) != len(camera_paths))
            or (not parity_claimed and parity_records)
        ):
            raise ValueError(
                "observation evidence parity records disagree with its claim"
            )
        if parity_claimed:
            if not {
                "LdrColor",
                "DistanceToCameraSD",
                "SemanticSegmentation",
            }.issubset(set(artifact_aovs)):
                raise ValueError(
                    "observation parity evidence omits its metric or semantic artifact"
                )
        if (
            assertions.get("metric_distance_verified") is not True
            or assertions.get("semantic_identity_verified")
            is not bool(semantic_assignments)
            or assertions.get("independent_camera_mapping_verified") is not True
            or assertions.get("warp_dlpack_reduction_verified") is not True
        ):
            raise ValueError("observation evidence contains unsupported assertions")
        observations = record.get("observations")
        if not isinstance(observations, list) or len(observations) != len(camera_paths):
            raise ValueError("observation evidence has an incomplete camera set")
        render_products: set[str] = set()
        expected_rendered_labels = {
            f"usd_cli: {item['label']};" for item in semantic_assignments
        }
        rendered_semantic_ids_by_camera: dict[str, dict[str, int]] = {}
        for camera_path, observation in zip(camera_paths, observations, strict=True):
            if (
                not isinstance(observation, dict)
                or observation.get("camera") != camera_path
                or observation.get("camera_definition_digest")
                != definition_digests[camera_path]
                or observation.get("semantic_assignments")
                != [
                    {"path": item["path"], "label": item["label"]}
                    for item in semantic_assignments
                ]
            ):
                raise ValueError(
                    "observation evidence camera binding differs from the rig"
                )
            render_product = observation.get("render_product")
            parsed_render_product = (
                PurePosixPath(render_product)
                if isinstance(render_product, str)
                else None
            )
            if (
                parsed_render_product is None
                or not parsed_render_product.is_absolute()
                or parsed_render_product.as_posix() != render_product
                or "\\" in render_product
                or render_product in render_products
            ):
                raise ValueError(
                    "observation evidence has an invalid or duplicate render product"
                )
            render_products.add(render_product)
            aovs = observation.get("aovs")
            if not isinstance(aovs, dict) or set(aovs) != set(OBSERVATION_AOVS):
                raise ValueError("observation evidence has an incomplete AOV set")
            for name, aov in aovs.items():
                statistics = aov.get("statistics") if isinstance(aov, dict) else None
                expected_reduction = (
                    "cpu_semantic_metadata_decode_v1"
                    if name == "SemanticIdMap"
                    else "warp_cuda_dlpack_v1"
                )
                if (
                    not isinstance(statistics, dict)
                    or statistics.get("reduction") != expected_reduction
                    or not has_exact_aov_element_count(aov)
                ):
                    raise ValueError(
                        "observation evidence contains an unqualified AOV reduction"
                    )
                shape = aov.get("shape")
                if name == "SemanticIdMap":
                    if not isinstance(shape, list) or not shape:
                        raise ValueError(
                            "observation evidence contains an invalid AOV shape"
                        )
                elif shape != [
                    expected_resolution["height"],
                    expected_resolution["width"],
                    OBSERVATION_IMAGE_CHANNELS[name],
                ]:
                    raise ValueError(
                        "observation evidence contains an invalid AOV shape"
                    )
                if not isinstance(aov.get("dtype"), str) or not aov["dtype"]:
                    raise ValueError(
                        "observation evidence contains an invalid AOV dtype"
                    )
            distance_statistics = aovs["DistanceToCameraSD"]["statistics"]
            if not has_positive_metric_distance_samples(
                distance_statistics,
                max_pixels=(
                    expected_resolution["width"] * expected_resolution["height"]
                ),
            ):
                raise ValueError(
                    "observation evidence does not support its metric assertion"
                )
            if semantic_assignments:
                rendered_labels = aovs["SemanticSegmentation"]["statistics"].get(
                    "semantic_labels"
                )
                rendered_semantic_ids = positive_requested_semantic_id_map(
                    rendered_labels,
                    expected_rendered_labels,
                    max_pixels=(
                        expected_resolution["width"] * expected_resolution["height"]
                    ),
                )
                if rendered_semantic_ids is None:
                    raise ValueError(
                        "observation evidence does not support its semantic assertion"
                    )
            else:
                rendered_semantic_ids = {}
            rendered_semantic_ids_by_camera[camera_path] = rendered_semantic_ids
        if parity_claimed:
            for camera_path, parity in zip(camera_paths, parity_records, strict=True):
                if (
                    not isinstance(parity, dict)
                    or not isinstance(parity.get("semantics"), dict)
                    or parity["semantics"].get("requested")
                    is not bool(semantic_assignments)
                    or not _passing_parity_record(
                        parity,
                        camera_path,
                        semantic_assignments,
                        rendered_semantic_ids_by_camera[camera_path],
                    )
                ):
                    raise ValueError("observation evidence parity record did not pass")
    expected_digest = record.pop("evidence_digest", None)
    if expected_digest != canonical_json_digest(record):
        raise ValueError("verification evidence digest does not match its contents")
    record["evidence_digest"] = expected_digest
    return record


def _camera_definition_payload(camera: dict) -> dict:
    """Calibration fields that define camera interchange, excluding derived evidence."""

    return {
        key: camera[key]
        for key in (
            "schema",
            "id",
            "id_source",
            "path",
            "resolution",
            "projection",
            "distortion",
            "intrinsics",
            "extrinsics",
            "image_axes",
            "clipping_range_m",
        )
    }


def rig_camera_paths(stage, rig_path: str) -> list[str]:
    from pxr import Sdf, Usd, UsdGeom

    from usd_core.camera_analysis.scene import imageable_analysis_policy

    prim = stage.GetPrimAtPath(Sdf.Path(rig_path))
    if not prim.IsValid():
        raise ValueError(f"camera rig does not exist: {rig_path}")
    camera_count = 0
    paths: list[str] = []
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not child.IsA(UsdGeom.Camera):
            continue
        camera_count += 1
        allowed, _ = imageable_analysis_policy(child)
        if allowed:
            paths.append(child.GetPath().pathString)
    paths.sort()
    if not paths:
        if camera_count:
            raise ValueError(
                "camera rig contains no cameras eligible under composed visibility "
                f"and purpose policy: {rig_path}"
            )
        raise ValueError(f"camera rig contains no cameras: {rig_path}")
    return paths


def prepare_verification_generation(
    verification: dict,
    *,
    staging_dir: str | Path,
    artifact_base_dir: str | Path,
    evidence_root_name: str,
    staged_artifact_base_dir: str | Path | None = None,
) -> tuple[dict, Path, list[str]]:
    """Retarget staged evidence to one immutable content-addressed generation."""

    if not isinstance(verification, dict):
        raise ValueError("verification evidence must be an object")
    staging = Path(staging_dir).expanduser().resolve()
    # ``artifact_base_dir`` is the already-validated publication parent. Keep its
    # lexical absolute name for the final report paths; resolving it again after a
    # long render would follow a parent symlink substituted by another process.
    artifact_base = Path(os.path.abspath(Path(artifact_base_dir).expanduser()))
    staged_artifact_base = (
        Path(staged_artifact_base_dir).expanduser().resolve()
        if staged_artifact_base_dir is not None
        else artifact_base
    )
    if not staging.is_dir():
        raise ValueError("verification evidence staging directory is missing")
    try:
        staging.relative_to(staged_artifact_base)
    except ValueError as exc:
        raise ValueError(
            "verification evidence staging directory escapes its private base"
        ) from exc
    root_component = PurePosixPath(evidence_root_name)
    if (
        not evidence_root_name
        or root_component.is_absolute()
        or len(root_component.parts) != 1
        or root_component.parts[0] in {".", ".."}
        or "\\" in evidence_root_name
    ):
        raise ValueError("verification evidence root name is unsafe")

    record = json.loads(canonical_json_bytes(verification))
    artifacts = record.get("artifacts")
    request_digest = record.get("request_digest")
    if (
        record.get("artifact_path_base") != "rig_document_directory"
        or not isinstance(artifacts, list)
        or not artifacts
        or not _SHA256_RE.fullmatch(str(request_digest or ""))
    ):
        raise ValueError("verification evidence is not generation-ready")

    relative_within_staging: list[PurePosixPath] = []
    content_binding: list[dict] = []
    seen: set[str] = set()
    for artifact in artifacts:
        relative = artifact.get("relative_path") if isinstance(artifact, dict) else None
        if not isinstance(relative, str):
            raise ValueError("verification artifact omits its staged path")
        parsed = PurePosixPath(relative)
        if (
            not relative
            or parsed.is_absolute()
            or parsed.as_posix() != relative
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise ValueError("verification artifact has an unsafe staged path")
        source = staged_artifact_base.joinpath(*parsed.parts).resolve()
        try:
            within = source.relative_to(staging)
        except ValueError as exc:
            raise ValueError(
                "verification artifact is outside its unique staging directory"
            ) from exc
        within_posix = PurePosixPath(within.as_posix())
        if within_posix.as_posix() in seen or not source.is_file():
            raise ValueError("verification artifact staging set is invalid")
        seen.add(within_posix.as_posix())
        digest, size = sha256_file(source)
        if digest != artifact.get("sha256") or size != artifact.get("size_bytes"):
            raise ValueError(
                "verification staged artifact content differs from evidence"
            )
        relative_within_staging.append(within_posix)
        content_binding.append(
            {
                "camera": artifact.get("camera"),
                "aov": artifact.get("aov", "LdrColor"),
                "relative_path": within_posix.as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
        )

    artifact_set_digest = canonical_json_digest(content_binding)
    generation = (
        f"request-{request_digest.removeprefix('sha256:')}--"
        f"artifacts-{artifact_set_digest.removeprefix('sha256:')}"
    )
    generation_dir = artifact_base / evidence_root_name / generation
    final_paths: list[str] = []
    for artifact, relative in zip(artifacts, relative_within_staging, strict=True):
        final_relative = PurePosixPath(evidence_root_name, generation, *relative.parts)
        artifact["relative_path"] = final_relative.as_posix()
        final_paths.append(str(artifact_base.joinpath(*final_relative.parts)))
    record["artifact_path_base"] = "rig_document_directory"
    record.pop("evidence_digest", None)
    record["evidence_digest"] = canonical_json_digest(record)
    return record, generation_dir, final_paths


def create_owned_private_directory(
    parent_dir_fd: int,
    name: str,
) -> tuple[int, tuple[int, int]]:
    """Create and securely open one current-user 0700 directory and its identity.

    Directory creation honors the process umask.  An exact-0700 inode needs no metadata
    mutation.  A mode-mismatched inode is hardened through its ``O_PATH`` descriptor only
    when the held parent prevents cross-user entry swaps.  Every named/held identity is
    checked, and an ambiguous replacement is never chmodded or cleanup-owned.
    """

    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or not hasattr(os, "O_PATH")
    ):
        raise ValueError("private directory name or platform is unsafe")
    parent_metadata = os.fstat(parent_dir_fd)
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("private directory parent descriptor is not a directory")
    parent_prevents_cross_user_swaps = not (
        stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ) or (
        parent_metadata.st_uid in {0, os.geteuid()}
        and bool(parent_metadata.st_mode & stat.S_ISVTX)
    )
    os.mkdir(name, mode=0o700, dir_fd=parent_dir_fd)
    creation_identity: tuple[int, int] | None = None
    path_descriptor = -1
    directory_descriptor = -1
    try:
        path_descriptor = os.open(
            name,
            os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_dir_fd,
        )
        held_path = os.fstat(path_descriptor)
        if (
            not stat.S_ISDIR(held_path.st_mode)
            or held_path.st_uid != os.geteuid()
            or stat.S_IMODE(held_path.st_mode) & 0o077
        ):
            raise ValueError("private directory changed during path open")
        candidate_identity = (held_path.st_dev, held_path.st_ino)
        named_at_creation = os.stat(
            name,
            dir_fd=parent_dir_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named_at_creation.st_mode)
            or named_at_creation.st_uid != os.geteuid()
            or (named_at_creation.st_dev, named_at_creation.st_ino)
            != candidate_identity
        ):
            raise ValueError("private directory changed during creation")
        needs_permission_hardening = (
            stat.S_IMODE(held_path.st_mode) & 0o777 != 0o700
            or stat.S_IMODE(named_at_creation.st_mode) & 0o777 != 0o700
        )
        if needs_permission_hardening and not parent_prevents_cross_user_swaps:
            raise ValueError(
                "private directory permissions are ambiguous in a shared parent"
            )
        creation_identity = candidate_identity
        if needs_permission_hardening:
            proc_descriptor_path = f"/proc/self/fd/{path_descriptor}"
            special_bits = stat.S_IMODE(held_path.st_mode) & 0o7000
            os.chmod(proc_descriptor_path, special_bits | 0o700)
        secured_path = os.fstat(path_descriptor)
        if (
            secured_path.st_uid != os.geteuid()
            or stat.S_IMODE(secured_path.st_mode) & 0o777 != 0o700
            or (secured_path.st_dev, secured_path.st_ino) != creation_identity
        ):
            raise ValueError("private directory permissions could not be secured")
        directory_descriptor = os.open(
            ".",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=path_descriptor,
        )
        opened = os.fstat(directory_descriptor)
        named = os.stat(name, dir_fd=parent_dir_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or opened.st_uid != os.geteuid()
            or named.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) & 0o777 != 0o700
            or stat.S_IMODE(named.st_mode) & 0o777 != 0o700
            or (opened.st_dev, opened.st_ino) != creation_identity
            or (named.st_dev, named.st_ino) != creation_identity
        ):
            raise ValueError("private directory changed while it was secured")
        if creation_identity is None:
            raise RuntimeError("private directory identity was not established")
        result = (directory_descriptor, creation_identity)
        directory_descriptor = -1
        return result
    except Exception:
        if creation_identity is None:
            recovery_descriptor = -1
            close_recovery_descriptor = False
            try:
                if path_descriptor >= 0:
                    recovery_descriptor = path_descriptor
                else:
                    recovery_descriptor = os.open(
                        name,
                        os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=parent_dir_fd,
                    )
                    close_recovery_descriptor = True
                recovered = os.fstat(recovery_descriptor)
                recovered_identity = (recovered.st_dev, recovered.st_ino)
                named_recovery = os.stat(
                    name,
                    dir_fd=parent_dir_fd,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISDIR(recovered.st_mode)
                    and stat.S_ISDIR(named_recovery.st_mode)
                    and recovered.st_uid == os.geteuid()
                    and named_recovery.st_uid == os.geteuid()
                    and not (stat.S_IMODE(recovered.st_mode) & 0o077)
                    and not (stat.S_IMODE(named_recovery.st_mode) & 0o077)
                    and (
                        parent_prevents_cross_user_swaps
                        or (
                            stat.S_IMODE(recovered.st_mode) & 0o777 == 0o700
                            and stat.S_IMODE(named_recovery.st_mode) & 0o777
                            == 0o700
                        )
                    )
                    and (named_recovery.st_dev, named_recovery.st_ino)
                    == recovered_identity
                ):
                    creation_identity = recovered_identity
            except OSError:
                pass
            finally:
                if close_recovery_descriptor and recovery_descriptor >= 0:
                    try:
                        os.close(recovery_descriptor)
                    except OSError:
                        pass
        if creation_identity is not None:
            try:
                current = os.stat(
                    name,
                    dir_fd=parent_dir_fd,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISDIR(current.st_mode)
                    and current.st_uid == os.geteuid()
                    and not (stat.S_IMODE(current.st_mode) & 0o077)
                    and (
                        parent_prevents_cross_user_swaps
                        or stat.S_IMODE(current.st_mode) & 0o777 == 0o700
                    )
                    and (current.st_dev, current.st_ino) == creation_identity
                ):
                    os.rmdir(name, dir_fd=parent_dir_fd)
            except OSError:
                pass
        raise
    finally:
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        if path_descriptor >= 0:
            os.close(path_descriptor)


class PreparedVerificationStaging(os.PathLike[str]):
    """A private system-temp directory retained by descriptor until cleanup."""

    def __init__(self, prefix: str) -> None:
        if not prefix or "/" in prefix or "\\" in prefix:
            raise ValueError("verification private staging prefix is unsafe")
        temp_root = Path(tempfile.gettempdir()).resolve()
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        named_temp_root = os.stat(temp_root, follow_symlinks=False)
        parent_fd = os.open(temp_root, directory_flags)
        directory_fd = -1
        directory_identity: tuple[int, int] | None = None
        created: Path | None = None
        try:
            opened_temp_root = os.fstat(parent_fd)
            if (
                not stat.S_ISDIR(named_temp_root.st_mode)
                or not stat.S_ISDIR(opened_temp_root.st_mode)
                or (named_temp_root.st_dev, named_temp_root.st_ino)
                != (opened_temp_root.st_dev, opened_temp_root.st_ino)
            ):
                raise ValueError(
                    "verification system temporary root changed while it was opened"
                )
            trusted_owners = {0, os.geteuid()}
            if (
                named_temp_root.st_uid not in trusted_owners
                or opened_temp_root.st_uid not in trusted_owners
                or (
                    stat.S_IMODE(named_temp_root.st_mode) & 0o022
                    and not (named_temp_root.st_mode & stat.S_ISVTX)
                )
                or (
                    stat.S_IMODE(opened_temp_root.st_mode) & 0o022
                    and not (opened_temp_root.st_mode & stat.S_ISVTX)
                )
            ):
                raise ValueError(
                    "verification system temporary root is not a trusted private or "
                    "sticky directory"
                )
            for _attempt in range(64):
                candidate_name = f"{prefix}{os.urandom(8).hex()}"
                try:
                    directory_fd, directory_identity = (
                        create_owned_private_directory(parent_fd, candidate_name)
                    )
                except FileExistsError:
                    continue
                created = temp_root / candidate_name
                break
            else:
                raise FileExistsError(
                    "verification private staging name collisions exceeded retry limit"
                )
            if created is None or directory_identity is None:
                raise RuntimeError("verification private staging was not established")
            named = os.stat(
                created.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            opened = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(named.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise ValueError(
                    "verification private staging changed while it was opened"
                )
        except Exception:
            if directory_identity is not None and created is not None:
                try:
                    current = os.stat(
                        created.name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    if (
                        stat.S_ISDIR(current.st_mode)
                        and current.st_uid == os.geteuid()
                        and stat.S_IMODE(current.st_mode) & 0o777 == 0o700
                        and (current.st_dev, current.st_ino) == directory_identity
                    ):
                        os.rmdir(created.name, dir_fd=parent_fd)
                except OSError:
                    pass
            if directory_fd >= 0:
                os.close(directory_fd)
            os.close(parent_fd)
            raise
        self.path = created
        self.parent_descriptor = parent_fd
        self.directory_descriptor = directory_fd
        self._identity = directory_identity
        self._closed = False

    def __fspath__(self) -> str:
        return str(self.path)

    def validate_named_identity(self) -> None:
        try:
            named = os.stat(
                self.path.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ValueError(
                "verification private staging path changed after creation"
            ) from exc
        opened = os.fstat(self.directory_descriptor)
        if (
            not stat.S_ISDIR(named.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != self._identity
            or (named.st_dev, named.st_ino) != self._identity
        ):
            raise ValueError(
                "verification private staging path changed after creation"
            )

    @staticmethod
    def _clear_directory(directory_fd: int) -> None:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        for name in sorted(os.listdir(directory_fd)):
            named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(named.st_mode):
                child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                try:
                    opened = os.fstat(child_fd)
                    if (opened.st_dev, opened.st_ino) != (
                        named.st_dev,
                        named.st_ino,
                    ):
                        raise ValueError(
                            "verification private staging changed during cleanup"
                        )
                    PreparedVerificationStaging._clear_directory(child_fd)
                finally:
                    os.close(child_fd)
                current = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (current.st_dev, current.st_ino) != (
                    named.st_dev,
                    named.st_ino,
                ):
                    raise ValueError(
                        "verification private staging changed during cleanup"
                    )
                os.rmdir(name, dir_fd=directory_fd)
            else:
                os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)

    def cleanup(self) -> None:
        if self._closed:
            return
        try:
            self._clear_directory(self.directory_descriptor)
            try:
                named = os.stat(
                    self.path.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                named = None
            if named is not None and (named.st_dev, named.st_ino) == self._identity:
                os.rmdir(self.path.name, dir_fd=self.parent_descriptor)
                os.fsync(self.parent_descriptor)
        finally:
            os.close(self.directory_descriptor)
            os.close(self.parent_descriptor)
            self._closed = True


def discard_unpublished_verification_generation(
    *,
    evidence_root_dir_fd: int,
    generation_dir_fd: int,
    generation_name: str,
    generation_identity: tuple[int, int],
) -> None:
    """Clear one unpublished generation without touching a substituted name.

    The contents are reached only through the descriptor retained from creation.  The
    directory name is removed only while it still denotes that same opened inode; a
    renamed generation is emptied but safely orphaned, and a replacement at the old
    name is left untouched.
    """

    if (
        not generation_name
        or generation_name in {".", ".."}
        or "/" in generation_name
        or "\\" in generation_name
    ):
        raise ValueError("unpublished verification generation name is unsafe")
    opened = os.fstat(generation_dir_fd)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != generation_identity
    ):
        raise ValueError("unpublished verification generation identity changed")
    PreparedVerificationStaging._clear_directory(generation_dir_fd)
    try:
        named = os.stat(
            generation_name,
            dir_fd=evidence_root_dir_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if stat.S_ISDIR(named.st_mode) and (
        named.st_dev,
        named.st_ino,
    ) == generation_identity:
        os.rmdir(generation_name, dir_fd=evidence_root_dir_fd)
        os.fsync(evidence_root_dir_fd)


def rename_noreplace(
    source_name: str,
    destination_name: str,
    *,
    source_dir_fd: int,
    destination_dir_fd: int,
) -> None:
    """Atomically rename a directory entry only when the destination is absent."""

    for name in (source_name, destination_name):
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("atomic no-replace rename requires safe entry names")
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication requires renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_dir_fd,
        os.fsencode(source_name),
        destination_dir_fd,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise RuntimeError(
            "filesystem does not support atomic no-replace publication"
        )
    raise OSError(error, os.strerror(error), destination_name)


def _owned_without_shared_write(metadata: os.stat_result) -> bool:
    return metadata.st_uid == os.geteuid() and not (
        stat.S_IMODE(metadata.st_mode) & 0o022
    )


def copy_verification_staging(
    *,
    source_dir_fd: int,
    destination_dir_fd: int,
    artifact_relative_paths: list[str],
) -> None:
    """Copy only declared artifacts into an opened unpublished generation."""

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parsed_paths = [PurePosixPath(path) for path in artifact_relative_paths]
    if not parsed_paths or any(
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        for path in parsed_paths
    ):
        raise ValueError("verification staging has an unsafe artifact inventory")
    if len({path.as_posix() for path in parsed_paths}) != len(parsed_paths):
        raise ValueError("verification staging has duplicate artifact paths")

    for relative_path in sorted(parsed_paths, key=lambda path: path.as_posix()):
        current_source_fd = -1
        current_destination_fd = -1
        source_file_fd = -1
        destination_file_fd = -1
        pending_source_fd = -1
        pending_destination_fd = -1
        pending_destination_identity: tuple[int, int] | None = None
        pending_destination_created = False
        try:
            current_source_fd = os.dup(source_dir_fd)
            current_destination_fd = os.dup(destination_dir_fd)
            for component in relative_path.parts[:-1]:
                named_source = os.stat(
                    component,
                    dir_fd=current_source_fd,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(named_source.st_mode):
                    raise ValueError(
                        "verification artifact parent is not a real directory"
                    )
                pending_source_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=current_source_fd,
                )
                opened_source = os.fstat(pending_source_fd)
                if (opened_source.st_dev, opened_source.st_ino) != (
                    named_source.st_dev,
                    named_source.st_ino,
                ):
                    raise ValueError(
                        "verification artifact parent changed during copy"
                )
                try:
                    (
                        pending_destination_fd,
                        pending_destination_identity,
                    ) = create_owned_private_directory(
                        current_destination_fd, component
                    )
                    pending_destination_created = True
                    named_destination = os.stat(
                        component,
                        dir_fd=current_destination_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    named_destination = os.stat(
                        component,
                        dir_fd=current_destination_fd,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISDIR(named_destination.st_mode)
                        or not _owned_without_shared_write(named_destination)
                    ):
                        raise ValueError(
                            "verification destination parent is not privately writable"
                        )
                    pending_destination_identity = (
                        named_destination.st_dev,
                        named_destination.st_ino,
                    )
                    pending_destination_fd = os.open(
                        component,
                        directory_flags,
                        dir_fd=current_destination_fd,
                    )
                opened_destination = os.fstat(pending_destination_fd)
                if (
                    not stat.S_ISDIR(named_destination.st_mode)
                    or not stat.S_ISDIR(opened_destination.st_mode)
                    or not _owned_without_shared_write(named_destination)
                    or not _owned_without_shared_write(opened_destination)
                    or pending_destination_identity is None
                    or (opened_destination.st_dev, opened_destination.st_ino)
                    != (named_destination.st_dev, named_destination.st_ino)
                    or (opened_destination.st_dev, opened_destination.st_ino)
                    != pending_destination_identity
                ):
                    raise ValueError(
                        "verification destination parent changed during copy"
                    )
                os.close(current_source_fd)
                os.close(current_destination_fd)
                current_source_fd = pending_source_fd
                current_destination_fd = pending_destination_fd
                pending_source_fd = -1
                pending_destination_fd = -1
                pending_destination_identity = None
                pending_destination_created = False

            filename = relative_path.parts[-1]
            named_source = os.stat(
                filename,
                dir_fd=current_source_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(named_source.st_mode) or named_source.st_nlink != 1:
                raise ValueError(
                    "verification staging contains a non-regular or linked artifact"
                )
            source_file_fd = os.open(
                filename,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                dir_fd=current_source_fd,
            )
            opened_source = os.fstat(source_file_fd)
            if (opened_source.st_dev, opened_source.st_ino) != (
                named_source.st_dev,
                named_source.st_ino,
            ):
                raise ValueError("verification staging artifact changed during copy")
            destination_file_fd = os.open(
                filename,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=current_destination_fd,
            )
            os.fchmod(destination_file_fd, 0o600)
            while chunk := os.read(source_file_fd, 1024 * 1024):
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_file_fd, view)
                    if written <= 0:
                        raise OSError(
                            "short write while copying verification artifact"
                        )
                    view = view[written:]
            os.fsync(destination_file_fd)
            opened_destination = os.fstat(destination_file_fd)
            named_destination = os.stat(
                filename,
                dir_fd=current_destination_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(opened_destination.st_mode)
                or opened_destination.st_nlink != 1
                or (opened_destination.st_dev, opened_destination.st_ino)
                != (named_destination.st_dev, named_destination.st_ino)
            ):
                raise ValueError(
                    "verification destination artifact changed during copy"
                )
            os.fsync(current_destination_fd)
        except OSError as exc:
            raise ValueError(
                "verification staging could not be copied without following links"
            ) from exc
        finally:
            if destination_file_fd >= 0:
                os.close(destination_file_fd)
            if source_file_fd >= 0:
                os.close(source_file_fd)
            if pending_destination_fd >= 0:
                os.close(pending_destination_fd)
                pending_destination_fd = -1
            if (
                pending_destination_created
                and pending_destination_identity is not None
                and current_destination_fd >= 0
            ):
                try:
                    current_pending_destination = os.stat(
                        component,
                        dir_fd=current_destination_fd,
                        follow_symlinks=False,
                    )
                    if (
                        stat.S_ISDIR(current_pending_destination.st_mode)
                        and current_pending_destination.st_uid == os.geteuid()
                        and stat.S_IMODE(current_pending_destination.st_mode) & 0o777
                        == 0o700
                        and (
                            current_pending_destination.st_dev,
                            current_pending_destination.st_ino,
                        )
                        == pending_destination_identity
                    ):
                        os.rmdir(component, dir_fd=current_destination_fd)
                except OSError:
                    pass
            if pending_source_fd >= 0:
                os.close(pending_source_fd)
            if current_destination_fd >= 0:
                os.close(current_destination_fd)
            if current_source_fd >= 0:
                os.close(current_source_fd)
    os.fsync(destination_dir_fd)
def _sha256_file_at(
    directory_fd: int,
    relative_path: PurePosixPath,
) -> tuple[str, int, tuple[int, int, int, int, int]]:
    """Hash one regular file beneath an open directory without following symlinks."""

    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError("verification generation has an unsafe artifact path")
    current_fd = os.dup(directory_fd)
    file_fd: int | None = None
    try:
        root_metadata = os.fstat(current_fd)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or not _owned_without_shared_write(root_metadata)
        ):
            raise ValueError(
                "verification generation directory is not privately writable"
            )
        for part in relative_path.parts[:-1]:
            named = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(named.st_mode)
                or not _owned_without_shared_write(named)
            ):
                raise ValueError(
                    "verification generation artifact parent is not a private directory"
                )
            opened = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current_fd,
            )
            opened_stat = os.fstat(opened)
            if (
                opened_stat.st_dev != named.st_dev
                or opened_stat.st_ino != named.st_ino
                or not _owned_without_shared_write(opened_stat)
            ):
                os.close(opened)
                raise ValueError(
                    "verification generation artifact parent changed during validation"
                )
            os.close(current_fd)
            current_fd = opened

        filename = relative_path.parts[-1]
        named_file = os.stat(filename, dir_fd=current_fd, follow_symlinks=False)
        if not stat.S_ISREG(named_file.st_mode):
            raise ValueError(
                "verification generation artifact must be a real regular file"
            )
        file_fd = os.open(
            filename,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=current_fd,
        )
        opened_file = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened_file.st_mode)
            or opened_file.st_nlink != 1
            or not _owned_without_shared_write(named_file)
            or not _owned_without_shared_write(opened_file)
            or opened_file.st_dev != named_file.st_dev
            or opened_file.st_ino != named_file.st_ino
        ):
            raise ValueError(
                "verification generation artifact changed during validation"
            )
        opened_signature = (
            opened_file.st_dev,
            opened_file.st_ino,
            opened_file.st_size,
            opened_file.st_mtime_ns,
            opened_file.st_ctime_ns,
        )
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        final_opened = os.fstat(file_fd)
        final_named = os.stat(
            filename,
            dir_fd=current_fd,
            follow_symlinks=False,
        )
        final_signature = (
            final_opened.st_dev,
            final_opened.st_ino,
            final_opened.st_size,
            final_opened.st_mtime_ns,
            final_opened.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(final_opened.st_mode)
            or final_opened.st_nlink != 1
            or not _owned_without_shared_write(final_opened)
            or not _owned_without_shared_write(final_named)
            or final_signature != opened_signature
            or (final_named.st_dev, final_named.st_ino)
            != (final_opened.st_dev, final_opened.st_ino)
        ):
            raise ValueError(
                "verification generation artifact changed while it was hashed"
            )
        return "sha256:" + digest.hexdigest(), size, final_signature
    except OSError as exc:
        raise ValueError(
            "verification generation cannot be opened without following links"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(current_fd)


def _regular_file_signature_at(
    directory_fd: int,
    relative_path: PurePosixPath,
) -> tuple[int, int, int, int, int]:
    """Re-stat one declared artifact through a no-follow descriptor walk."""

    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError("verification generation has an unsafe artifact path")
    current_fd = os.dup(directory_fd)
    file_fd: int | None = None
    try:
        root_metadata = os.fstat(current_fd)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or not _owned_without_shared_write(root_metadata)
        ):
            raise ValueError(
                "verification generation directory is not privately writable"
            )
        for part in relative_path.parts[:-1]:
            named = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(named.st_mode)
                or not _owned_without_shared_write(named)
            ):
                raise ValueError(
                    "verification generation artifact parent is not a private directory"
                )
            opened = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current_fd,
            )
            opened_stat = os.fstat(opened)
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                named.st_dev,
                named.st_ino,
            ) or not _owned_without_shared_write(opened_stat):
                os.close(opened)
                raise ValueError(
                    "verification generation artifact parent changed after hashing"
                )
            os.close(current_fd)
            current_fd = opened

        filename = relative_path.parts[-1]
        named_file = os.stat(filename, dir_fd=current_fd, follow_symlinks=False)
        file_fd = os.open(
            filename,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=current_fd,
        )
        opened_file = os.fstat(file_fd)
        if (
            not stat.S_ISREG(named_file.st_mode)
            or not stat.S_ISREG(opened_file.st_mode)
            or opened_file.st_nlink != 1
            or not _owned_without_shared_write(named_file)
            or not _owned_without_shared_write(opened_file)
            or (opened_file.st_dev, opened_file.st_ino)
            != (named_file.st_dev, named_file.st_ino)
        ):
            raise ValueError(
                "verification generation artifact changed after it was hashed"
            )
        return (
            opened_file.st_dev,
            opened_file.st_ino,
            opened_file.st_size,
            opened_file.st_mtime_ns,
            opened_file.st_ctime_ns,
        )
    except OSError as exc:
        raise ValueError(
            "verification generation artifact changed after it was hashed"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(current_fd)


def _verification_generation_inventory(
    directory_fd: int,
    prefix: PurePosixPath = PurePosixPath(),
) -> tuple[set[PurePosixPath], set[PurePosixPath]]:
    """Return exact regular-file/directory inventory without following links."""

    files: set[PurePosixPath] = set()
    directories: set[PurePosixPath] = set()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    before = os.fstat(directory_fd)
    before_signature = (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if (
        not stat.S_ISDIR(before.st_mode)
        or not _owned_without_shared_write(before)
    ):
        raise ValueError("verification generation directory is not privately writable")
    try:
        for name in sorted(os.listdir(directory_fd)):
            if name in {"", ".", ".."} or "/" in name or "\\" in name:
                raise ValueError("verification generation contains an unsafe name")
            relative = prefix / name
            named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(named.st_mode):
                if not _owned_without_shared_write(named):
                    raise ValueError(
                        "verification generation directory is not privately writable"
                    )
                child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                try:
                    opened = os.fstat(child_fd)
                    if (opened.st_dev, opened.st_ino) != (
                        named.st_dev,
                        named.st_ino,
                    ) or not _owned_without_shared_write(opened):
                        raise ValueError(
                            "verification generation directory changed during inventory"
                        )
                    directories.add(relative)
                    child_files, child_directories = (
                        _verification_generation_inventory(child_fd, relative)
                    )
                    files.update(child_files)
                    directories.update(child_directories)
                finally:
                    os.close(child_fd)
            elif (
                stat.S_ISREG(named.st_mode)
                and named.st_nlink == 1
                and _owned_without_shared_write(named)
            ):
                files.add(relative)
            else:
                raise ValueError(
                    "verification generation contains an unbound or linked entry"
                )
    except OSError as exc:
        raise ValueError(
            "verification generation inventory cannot be read without following links"
        ) from exc
    after = os.fstat(directory_fd)
    after_signature = (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        not _owned_without_shared_write(after)
        or after_signature != before_signature
    ):
        raise ValueError("verification generation changed during inventory")
    return files, directories


def validate_verification_generation(
    verification: dict,
    *,
    artifact_base_dir: str | Path,
    generation_dir_fd: int | None = None,
    generation_relative_path: str | None = None,
) -> None:
    """Prove an existing immutable generation exactly matches its bound artifacts."""

    artifact_base = Path(artifact_base_dir).expanduser().resolve()
    if (generation_dir_fd is None) != (generation_relative_path is None):
        raise ValueError(
            "verification generation descriptor and relative path must be supplied together"
        )
    generation_prefix = (
        PurePosixPath(generation_relative_path)
        if generation_relative_path is not None
        else None
    )
    if generation_prefix is not None and (
        generation_prefix.is_absolute()
        or not generation_prefix.parts
        or any(part in {"", ".", ".."} for part in generation_prefix.parts)
    ):
        raise ValueError("verification generation relative path is unsafe")
    artifact_base_fd: int | None = None
    within_generation_paths: list[PurePosixPath] = []
    artifact_signatures: list[
        tuple[PurePosixPath, tuple[int, int, int, int, int]]
    ] = []
    try:
        if generation_prefix is None:
            artifact_base_fd = os.open(
                artifact_base,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        for artifact in verification.get("artifacts", []):
            relative = PurePosixPath(artifact["relative_path"])
            if generation_prefix is not None:
                prefix_parts = generation_prefix.parts
                if (
                    len(relative.parts) <= len(prefix_parts)
                    or relative.parts[: len(prefix_parts)] != prefix_parts
                ):
                    raise ValueError(
                        "verification artifact is outside its opened generation"
                    )
                within_generation = PurePosixPath(
                    *relative.parts[len(prefix_parts) :]
                )
                within_generation_paths.append(within_generation)
                assert generation_dir_fd is not None
                digest, size, signature = _sha256_file_at(
                    generation_dir_fd, within_generation
                )
                artifact_signatures.append((within_generation, signature))
            else:
                assert artifact_base_fd is not None
                digest, size, signature = _sha256_file_at(artifact_base_fd, relative)
                artifact_signatures.append((relative, signature))
            if digest != artifact.get("sha256") or size != artifact.get("size_bytes"):
                raise ValueError("verification generation differs from staged evidence")
        if generation_prefix is not None:
            assert generation_dir_fd is not None
            actual_files, actual_directories = _verification_generation_inventory(
                generation_dir_fd
            )
            expected_files = set(within_generation_paths)
            expected_directories = {
                PurePosixPath(*path.parts[:index])
                for path in within_generation_paths
                for index in range(1, len(path.parts))
            }
            if (
                actual_files != expected_files
                or actual_directories != expected_directories
            ):
                raise ValueError(
                    "verification generation inventory differs from declared artifacts"
                )
        signature_base_fd = (
            generation_dir_fd
            if generation_prefix is not None
            else artifact_base_fd
        )
        assert signature_base_fd is not None
        for relative_path, expected_signature in artifact_signatures:
            if (
                _regular_file_signature_at(signature_base_fd, relative_path)
                != expected_signature
            ):
                raise ValueError(
                    "verification generation artifact changed after it was hashed"
                )
    except OSError as exc:
        raise ValueError(
            "verification generation base cannot be opened without following links"
        ) from exc
    finally:
        if artifact_base_fd is not None:
            os.close(artifact_base_fd)


def build_rig_document(
    stage,
    scene: SceneAnalysisIR,
    rig_path: str,
    *,
    resolution: tuple[int, int],
    floor_bounds_m=None,
    coverage_report: dict | None = None,
    verification: dict | None = None,
) -> dict:
    prim = stage.GetPrimAtPath(rig_path)
    paths = rig_camera_paths(stage, rig_path)
    if coverage_report is not None:
        if not isinstance(coverage_report, dict):
            raise ValueError("coverage report must be an object")
        # Detach caller-owned evidence and reject values JSON cannot represent before
        # any of its nested visibility records are embedded in camera calibrations.
        coverage_report = json.loads(canonical_json_bytes(coverage_report))
        if coverage_report.get("schema") != "usd-cli.camera-coverage.v1":
            raise ValueError("coverage report has an unsupported schema")
        if coverage_report.get("source_digest") != scene.source_digest:
            raise ValueError("coverage report is bound to a different source stage")
        _validated_policy_binding(
            coverage_report.get("analysis_policy"),
            coverage_report.get("analysis_policy_digest"),
            expected=scene.policy,
            context="coverage report",
        )
    visibility_by_camera = _visibility_by_camera(coverage_report)
    if coverage_report is not None and set(visibility_by_camera) != set(paths):
        raise ValueError("coverage report cameras differ from the exported rig")
    cameras = [
        camera_calibration(
            stage,
            scene,
            path,
            resolution=resolution,
            floor_bounds_m=floor_bounds_m,
            floor_surface=(coverage_report or {}).get("grid"),
            visibility=visibility_by_camera.get(path),
        )
        for path in paths
    ]
    stable_ids = [camera["id"] for camera in cameras]
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("camera rig contains duplicate stable camera IDs")
    for camera in cameras:
        camera["definition_digest"] = canonical_json_digest(
            _camera_definition_payload(camera)
        )

    authored_rig_id = prim.GetCustomDataByKey("usdCameraRigStableId")
    rig_id = str(authored_rig_id) if authored_rig_id else _derived_id("rig", rig_path)
    expected_overlay_actions = (
        analysis_overlay_actions(stage, scene) if verification is not None else []
    )
    verification_record = _verification_record(
        verification,
        source_digest=scene.source_digest,
        resolution=resolution,
        camera_paths=paths,
        camera_ids=stable_ids,
        analysis_policy=scene.policy,
        expected_overlay_actions=expected_overlay_actions,
    )
    document = {
        "schema": RIG_SCHEMA,
        "schema_version": RIG_DOCUMENT_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST,
        "conventions": {
            "canonical_coordinate_space": "right_handed_meter_z_up",
            "matrix_layout": "row_major",
            "vector_convention": "row_vector_postmultiply",
            "camera_axes": "+X right, +Y up, -Z forward",
            "image_axes": "+u right, +v down, origin top-left",
            "pixel_coordinates": (
                "continuous edge-origin coordinates; integer values denote pixel "
                "edges and pixel (column, row) has center "
                "(column + 0.5, row + 0.5)"
            ),
            "verification_artifact_paths": "relative to the directory containing this rig JSON",
            "floating_point": "IEEE-754 binary64 serialized as JSON numbers",
            "non_finite_values": "forbidden",
        },
        "rig": {
            "id": rig_id,
            "id_source": "authored_custom_data"
            if authored_rig_id
            else "derived_prim_path",
            "path": rig_path,
            "method": prim.GetCustomDataByKey("usdCameraRigMethod"),
            "metadata": read_rig_metadata(prim),
        },
        "source": {
            "digest": scene.source_digest,
            "meters_per_unit": scene.meters_per_unit,
            "source_up_axis": scene.source_up_axis,
            "canonical_up_axis": scene.canonical_up_axis,
            "stage_to_canonical": _matrix(scene.stage_to_canonical),
            "canonical_to_stage": _matrix(scene.canonical_to_stage),
            "coordinate_handedness": "right_handed",
            "digest_scope": "flattened composed stage including live session opinions",
            "analysis_policy": scene.policy.as_dict(),
            "analysis_policy_digest": scene.policy.digest,
            "verification_clone_actions": expected_overlay_actions,
            "verification_clone_actions_digest": canonical_json_digest(
                expected_overlay_actions
            ),
        },
        "cameras": cameras,
        "camera_set_digest": canonical_json_digest(
            [_camera_definition_payload(camera) for camera in cameras]
        ),
        "verification": verification_record,
    }
    if coverage_report is not None:
        document["coverage"] = {
            key: coverage_report[key]
            for key in (
                "schema",
                "source_digest",
                "scope_path",
                "target_coverage",
                "per_cell",
                "coverage_fraction",
                "passed",
                "accessible_cells",
                "covered_cells",
                "uncovered_cells",
                "accessible_area_m2",
                "covered_area_m2",
                "overlap_histogram",
                "grid",
                "cameras",
                "backend",
                "analysis_policy",
                "analysis_policy_digest",
            )
            if key in coverage_report
        }
    # Fail here rather than after creating a destination temp file.  This also proves
    # that visibility polygons remain representable without NaN/Infinity sentinels.
    canonical_json_bytes(document)
    return document


class PreparedJsonPublication(os.PathLike[str]):
    """An output name bound to the directory identity validated for publication.

    Camera analysis can run for long enough that re-resolving its output pathname at
    the final rename is unsafe: an intermediate directory may have been replaced by a
    symlink.  Keep the no-follow parent descriptor open from admission through the
    atomic replace, and use descriptor-relative operations for every publication
    mutation.
    """

    def __init__(self, destination: Path, directory_descriptor: int) -> None:
        self.destination = destination
        self.parent = destination.parent
        self.name = destination.name
        self.directory_descriptor = directory_descriptor
        metadata = os.fstat(directory_descriptor)
        self._parent_identity = (metadata.st_dev, metadata.st_ino)
        self._closed = False

    def __fspath__(self) -> str:
        return str(self.destination)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                os.close(self.directory_descriptor)
            except OSError:
                pass

    def validate_parent_identity(self) -> None:
        if self._closed:
            raise ValueError("camera publication directory descriptor is closed")
        held = os.fstat(self.directory_descriptor)
        try:
            named = os.stat(self.parent, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(
                "camera publication directory changed after validation"
            ) from exc
        if (
            not stat.S_ISDIR(held.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or (held.st_dev, held.st_ino) != self._parent_identity
            or (named.st_dev, named.st_ino) != self._parent_identity
        ):
            raise ValueError(
                "camera publication directory changed after validation"
            )

    @staticmethod
    def _temporary_name(destination_name: str, purpose: str) -> str:
        entropy = os.urandom(8).hex()
        return f".{destination_name}.{purpose}.{entropy}.tmp"


class BoundVerificationGeneration:
    """Descriptor capability binding one exact evidence generation to a rig JSON."""

    def __init__(
        self,
        *,
        artifact_base_dir: str | Path,
        generation_relative_path: str,
        publication_parent_dir_fd: int,
        evidence_parent_dir_fd: int,
        evidence_root_name: str,
        evidence_root_dir_fd: int,
        generation_name: str,
        generation_dir_fd: int,
    ) -> None:
        expected_relative = PurePosixPath(evidence_root_name, generation_name)
        if (
            any(
                not name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                for name in (evidence_root_name, generation_name)
            )
            or PurePosixPath(generation_relative_path) != expected_relative
        ):
            raise ValueError("verification generation descriptor binding is unsafe")
        self.artifact_base_dir = Path(artifact_base_dir)
        self.generation_relative_path = expected_relative.as_posix()
        self.publication_parent_dir_fd = publication_parent_dir_fd
        self.evidence_parent_dir_fd = evidence_parent_dir_fd
        self.evidence_root_name = evidence_root_name
        self.evidence_root_dir_fd = evidence_root_dir_fd
        self.generation_name = generation_name
        self.generation_dir_fd = generation_dir_fd
        evidence_root = os.fstat(evidence_root_dir_fd)
        generation = os.fstat(generation_dir_fd)
        self._evidence_root_identity = (evidence_root.st_dev, evidence_root.st_ino)
        self._generation_identity = (generation.st_dev, generation.st_ino)

    def _validate_named_identities(self) -> None:
        publication_parent = os.fstat(self.publication_parent_dir_fd)
        evidence_parent = os.fstat(self.evidence_parent_dir_fd)
        opened_root = os.fstat(self.evidence_root_dir_fd)
        named_root = os.stat(
            self.evidence_root_name,
            dir_fd=self.evidence_parent_dir_fd,
            follow_symlinks=False,
        )
        opened_generation = os.fstat(self.generation_dir_fd)
        named_generation = os.stat(
            self.generation_name,
            dir_fd=self.evidence_root_dir_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(publication_parent.st_mode)
            or not stat.S_ISDIR(evidence_parent.st_mode)
            or (publication_parent.st_dev, publication_parent.st_ino)
            != (evidence_parent.st_dev, evidence_parent.st_ino)
            or not stat.S_ISDIR(opened_root.st_mode)
            or not stat.S_ISDIR(named_root.st_mode)
            or not _owned_without_shared_write(opened_root)
            or not _owned_without_shared_write(named_root)
            or (opened_root.st_dev, opened_root.st_ino)
            != self._evidence_root_identity
            or (named_root.st_dev, named_root.st_ino)
            != self._evidence_root_identity
            or not stat.S_ISDIR(opened_generation.st_mode)
            or not stat.S_ISDIR(named_generation.st_mode)
            or not _owned_without_shared_write(opened_generation)
            or not _owned_without_shared_write(named_generation)
            or (opened_generation.st_dev, opened_generation.st_ino)
            != self._generation_identity
            or (named_generation.st_dev, named_generation.st_ino)
            != self._generation_identity
        ):
            raise ValueError(
                "camera verification generation changed during rig publication"
            )

    def validate(self, verification: dict) -> None:
        """Hash and inventory the held generation between identity checks."""

        before = os.fstat(self.generation_dir_fd)
        if not _owned_without_shared_write(before):
            raise ValueError(
                "camera verification generation is not privately writable"
            )
        before_signature = (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        self._validate_named_identities()
        validate_verification_generation(
            verification,
            artifact_base_dir=self.artifact_base_dir,
            generation_dir_fd=self.generation_dir_fd,
            generation_relative_path=self.generation_relative_path,
        )
        self._validate_named_identities()
        after = os.fstat(self.generation_dir_fd)
        after_signature = (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            not _owned_without_shared_write(after)
            or after_signature != before_signature
        ):
            raise ValueError(
                "camera verification generation changed during rig publication"
            )


def prepare_json_publication(
    output: str | Path,
    *,
    base_dir: str | Path | None = None,
    allowed_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> PreparedJsonPublication:
    """Resolve one output once, then create/open its parent without following links."""

    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(
            "identity-bound camera publication requires O_DIRECTORY and O_NOFOLLOW"
        )
    raw_destination = Path(output).expanduser()
    if not raw_destination.is_absolute():
        raw_destination = Path(base_dir or Path.cwd()).expanduser() / raw_destination
    if allowed_roots is not None:
        destination = Path(os.path.abspath(raw_destination))
    else:
        # Preserve unrestricted/direct-call resolution of parent symlinks without
        # resolving the final entry itself: an existing output symlink must be visible
        # to the publication preflight and rejected rather than followed.
        destination = raw_destination.parent.resolve() / raw_destination.name
    if not destination.name:
        raise ValueError("camera publication output must name a file")
    if allowed_roots is not None:
        canonical_roots = tuple(
            Path(os.path.abspath(Path(root).expanduser())) for root in allowed_roots
        )
        if not canonical_roots or not any(
            root in destination.parents for root in canonical_roots
        ):
            raise ValueError("output path is outside server.allowed_write_roots")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    anchor = Path(destination.anchor)
    descriptor = os.open(anchor, directory_flags)
    try:
        opened_anchor = os.fstat(descriptor)
        named_anchor = os.stat(anchor, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened_anchor.st_mode)
            or (opened_anchor.st_dev, opened_anchor.st_ino)
            != (named_anchor.st_dev, named_anchor.st_ino)
        ):
            raise ValueError("camera publication filesystem anchor changed")
        for component in destination.parent.parts[1:]:
            try:
                named_component = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                os.mkdir(component, mode=0o777, dir_fd=descriptor)
                named_component = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            if not stat.S_ISDIR(named_component.st_mode):
                raise ValueError("camera publication parent is not a real directory")
            opened_component = os.open(
                component,
                directory_flags,
                dir_fd=descriptor,
            )
            opened_metadata = os.fstat(opened_component)
            if (
                not stat.S_ISDIR(opened_metadata.st_mode)
                or (opened_metadata.st_dev, opened_metadata.st_ino)
                != (named_component.st_dev, named_component.st_ino)
            ):
                os.close(opened_component)
                raise ValueError(
                    "camera publication directory changed while it was opened"
                )
            os.close(descriptor)
            descriptor = opened_component

        opened = os.fstat(descriptor)
        named_parent = os.stat(destination.parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(named_parent.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (named_parent.st_dev, named_parent.st_ino)
        ):
            raise ValueError(
                "camera publication directory changed while it was opened"
            )
        publication = PreparedJsonPublication(destination, descriptor)
        publication.validate_parent_identity()
        return publication
    except Exception:
        os.close(descriptor)
        raise


def publish_json_document(
    document: dict,
    output: str | Path | PreparedJsonPublication,
    *,
    expected_schema: str | None = None,
    precommit: Callable[[], None] | None = None,
) -> tuple[str, str]:
    """Atomically publish one finite JSON object and return its content digest."""

    if not isinstance(document, dict):
        raise ValueError("published camera-analysis document must be an object")
    schema = document.get("schema")
    if not isinstance(schema, str) or not schema:
        raise ValueError("published camera-analysis document must declare a schema")
    if expected_schema is not None and schema != expected_schema:
        raise ValueError(
            f"camera-analysis document schema {schema!r} is not {expected_schema!r}"
        )
    # Validate the canonical representation first, then publish a human-readable form
    # whose key ordering and whitespace are also stable.
    canonical_json_bytes(document)
    payload = (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary_name = f"document-{os.urandom(16).hex()}.json"
    publication = (
        output
        if isinstance(output, PreparedJsonPublication)
        else prepare_json_publication(output)
    )
    owns_publication = publication is not output
    result = (
        str(publication.destination),
        "sha256:" + hashlib.sha256(payload).hexdigest(),
    )
    private_directory_descriptor = -1
    private_directory_name: str | None = None
    private_directory_identity: tuple[int, int] | None = None
    temporary_descriptor = -1
    temporary_identity: tuple[int, int] | None = None
    temporary_created = False
    temporary_source_bound = False
    try:
        publication.validate_parent_identity()
        try:
            existing_destination = os.stat(
                publication.name,
                dir_fd=publication.directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            if (
                not stat.S_ISREG(existing_destination.st_mode)
                or existing_destination.st_nlink != 1
            ):
                raise ValueError(
                    "camera publication destination must be absent or a single-link "
                    "regular file"
                )

        # Keep unpublished payload bytes out of the caller-controlled output parent.
        # A held 0700 child directory gives the temp name a private same-filesystem
        # namespace while preserving an atomic cross-directory descriptor replace.
        private_directory_name = publication._temporary_name(
            publication.name, "publish"
        )
        (
            private_directory_descriptor,
            private_directory_identity,
        ) = create_owned_private_directory(
            publication.directory_descriptor, private_directory_name
        )
        opened_private_directory = os.fstat(private_directory_descriptor)
        named_private_directory = os.stat(
            private_directory_name,
            dir_fd=publication.directory_descriptor,
            follow_symlinks=False,
        )
        opened_private_identity = (
            opened_private_directory.st_dev,
            opened_private_directory.st_ino,
        )
        if (
            not stat.S_ISDIR(opened_private_directory.st_mode)
            or not stat.S_ISDIR(named_private_directory.st_mode)
            or opened_private_directory.st_uid != os.geteuid()
            or named_private_directory.st_uid != os.geteuid()
            or stat.S_IMODE(opened_private_directory.st_mode) & 0o777 != 0o700
            or stat.S_IMODE(named_private_directory.st_mode) & 0o777 != 0o700
            or opened_private_identity != private_directory_identity
            or opened_private_identity
            != (named_private_directory.st_dev, named_private_directory.st_ino)
        ):
            raise ValueError(
                "camera publication private directory changed during open"
            )
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=private_directory_descriptor,
        )
        temporary_created = True
        opened_temporary = os.fstat(temporary_descriptor)
        temporary_identity = (
            opened_temporary.st_dev,
            opened_temporary.st_ino,
        )
        os.fchmod(temporary_descriptor, 0o600)
        opened_temporary = os.fstat(temporary_descriptor)
        named_temporary = os.stat(
            temporary_name,
            dir_fd=private_directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened_temporary.st_mode)
            or opened_temporary.st_nlink != 1
            or opened_temporary.st_uid != os.geteuid()
            or stat.S_IMODE(opened_temporary.st_mode) != 0o600
            or temporary_identity
            != (named_temporary.st_dev, named_temporary.st_ino)
        ):
            raise ValueError("camera publication temporary file changed during open")
        with os.fdopen(os.dup(temporary_descriptor), "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_metadata = os.fstat(temporary_descriptor)
        named_temporary = os.stat(
            temporary_name,
            dir_fd=private_directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(temporary_metadata.st_mode)
            or temporary_metadata.st_nlink != 1
            or (temporary_metadata.st_dev, temporary_metadata.st_ino)
            != temporary_identity
            or temporary_identity != (named_temporary.st_dev, named_temporary.st_ino)
        ):
            raise ValueError("camera publication temporary file changed during write")
        temporary_source_bound = True
        named_private_directory = os.stat(
            private_directory_name,
            dir_fd=publication.directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named_private_directory.st_mode)
            or (named_private_directory.st_dev, named_private_directory.st_ino)
            != private_directory_identity
        ):
            raise ValueError(
                "camera publication private directory changed during write"
            )
        publication.validate_parent_identity()
        if precommit is not None:
            precommit()
        try:
            os.replace(
                temporary_name,
                publication.name,
                src_dir_fd=private_directory_descriptor,
                dst_dir_fd=publication.directory_descriptor,
            )
        except OSError:
            # Some remote filesystems can report an indeterminate rename result. A
            # delegate-then-raise wrapper has the same shape. If the held temp inode is
            # already the destination, the commit happened and must be reported as
            # success; otherwise truncate only while its source name remains bound.
            try:
                installed = os.stat(
                    publication.name,
                    dir_fd=publication.directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError:
                installed = None
            if installed is None or (
                installed.st_dev,
                installed.st_ino,
            ) != temporary_identity:
                try:
                    remaining_source = os.stat(
                        temporary_name,
                        dir_fd=private_directory_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    # The source lived in our held 0700 namespace. If it was consumed,
                    # the rename committed even when an external last-writer replaced or
                    # removed the destination before the filesystem reported its result.
                    temporary_source_bound = False
                except OSError:
                    temporary_source_bound = False
                    raise
                else:
                    temporary_source_bound = (
                        remaining_source.st_dev,
                        remaining_source.st_ino,
                    ) == temporary_identity
                    raise
        temporary_source_bound = False
        # The descriptor-relative replace is the commit point.  All validation is
        # complete, and external writers intentionally follow ordinary atomic
        # last-writer-wins semantics.  Durability sync is best-effort so a committed
        # document can never be reported as a failed transaction.
        try:
            os.close(temporary_descriptor)
        except OSError:
            pass
        temporary_descriptor = -1
        try:
            current_private_directory = os.stat(
                private_directory_name,
                dir_fd=publication.directory_descriptor,
                follow_symlinks=False,
            )
            if (
                stat.S_ISDIR(current_private_directory.st_mode)
                and (
                    current_private_directory.st_dev,
                    current_private_directory.st_ino,
                )
                == private_directory_identity
            ):
                os.rmdir(
                    private_directory_name,
                    dir_fd=publication.directory_descriptor,
                )
        except OSError:
            pass
        try:
            os.fsync(publication.directory_descriptor)
        except OSError:
            pass
        return result
    except Exception:
        if temporary_descriptor >= 0:
            if temporary_created and temporary_identity is None:
                try:
                    recovered_temporary = os.fstat(temporary_descriptor)
                    if (
                        stat.S_ISREG(recovered_temporary.st_mode)
                        and recovered_temporary.st_nlink == 1
                        and recovered_temporary.st_uid == os.geteuid()
                    ):
                        temporary_identity = (
                            recovered_temporary.st_dev,
                            recovered_temporary.st_ino,
                        )
                except OSError:
                    pass
            if temporary_source_bound:
                try:
                    os.ftruncate(temporary_descriptor, 0)
                except OSError:
                    pass
            try:
                os.close(temporary_descriptor)
            except OSError:
                pass
            temporary_descriptor = -1
        if private_directory_descriptor >= 0:
            if temporary_created and temporary_identity is not None:
                try:
                    current_temporary = os.stat(
                        temporary_name,
                        dir_fd=private_directory_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        stat.S_ISREG(current_temporary.st_mode)
                        and current_temporary.st_nlink == 1
                        and (current_temporary.st_dev, current_temporary.st_ino)
                        == temporary_identity
                    ):
                        os.unlink(
                            temporary_name,
                            dir_fd=private_directory_descriptor,
                        )
                except OSError:
                    pass
            if (
                private_directory_identity is not None
                and private_directory_name is not None
            ):
                try:
                    current_private_directory = os.stat(
                        private_directory_name,
                        dir_fd=publication.directory_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        stat.S_ISDIR(current_private_directory.st_mode)
                        and (
                            current_private_directory.st_dev,
                            current_private_directory.st_ino,
                        )
                        == private_directory_identity
                    ):
                        os.rmdir(
                            private_directory_name,
                            dir_fd=publication.directory_descriptor,
                        )
                except OSError:
                    pass
        raise
    finally:
        if private_directory_descriptor >= 0:
            try:
                os.close(private_directory_descriptor)
            except OSError:
                pass
        if owns_publication:
            publication.close()


def publish_rig_document(
    document: dict,
    output: str | Path | PreparedJsonPublication,
    *,
    verification_generation: BoundVerificationGeneration | None = None,
) -> tuple[str, str]:
    if not isinstance(document, dict) or document.get("schema") != RIG_SCHEMA:
        raise ValueError("camera rig document has an unsupported schema")
    schema_version = document.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != RIG_DOCUMENT_SCHEMA_VERSION
    ):
        raise ValueError("camera rig document has an unsupported schema version")
    if document.get("digest_algorithm") != CANONICAL_JSON_DIGEST:
        raise ValueError("camera rig document uses an unsupported digest algorithm")
    source = document.get("source")
    cameras = document.get("cameras")
    if (
        not isinstance(source, dict)
        or not _SHA256_RE.fullmatch(str(source.get("digest", "")))
        or not isinstance(cameras, list)
        or not cameras
        or not all(isinstance(camera, dict) for camera in cameras)
    ):
        raise ValueError("camera rig document has malformed source or camera records")
    analysis_policy = _validated_policy_binding(
        source.get("analysis_policy"),
        source.get("analysis_policy_digest"),
        context="camera rig document source",
    )
    expected_overlay_actions = source.get("verification_clone_actions")
    if not isinstance(expected_overlay_actions, list) or source.get(
        "verification_clone_actions_digest"
    ) != canonical_json_digest(expected_overlay_actions):
        raise ValueError("camera rig document clone-action manifest is invalid")
    _validated_visibility_overlay(
        {
            "scope": "verification_clone_only",
            "policy_digest": analysis_policy.digest,
            "actions": expected_overlay_actions,
            "expanded_instance_roots": [],
        },
        policy_digest=analysis_policy.digest,
    )
    coverage = document.get("coverage")
    if coverage is not None:
        if (
            not isinstance(coverage, dict)
            or coverage.get("schema") != "usd-cli.camera-coverage.v1"
            or coverage.get("source_digest") != source["digest"]
        ):
            raise ValueError(
                "camera rig document coverage is bound to a different source"
            )
        _validated_policy_binding(
            coverage.get("analysis_policy"),
            coverage.get("analysis_policy_digest"),
            expected=analysis_policy,
            context="camera rig document coverage",
        )
    camera_paths = [camera.get("path") for camera in cameras]
    camera_ids = [camera.get("id") for camera in cameras]
    if (
        not all(isinstance(path, str) and path.startswith("/") for path in camera_paths)
        or len(set(camera_paths)) != len(camera_paths)
        or not all(isinstance(stable_id, str) and stable_id for stable_id in camera_ids)
        or len(set(camera_ids)) != len(camera_ids)
    ):
        raise ValueError("camera rig document has invalid camera paths or stable IDs")
    resolutions = [camera.get("resolution") for camera in cameras]
    if not all(
        isinstance(item, dict) and item == resolutions[0] for item in resolutions
    ):
        raise ValueError("camera rig document cameras use inconsistent resolutions")
    resolution = resolutions[0]
    width = resolution.get("width")
    height = resolution.get("height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width < 1
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height < 1
    ):
        raise ValueError("camera rig document has an invalid resolution")
    try:
        camera_payloads = [_camera_definition_payload(camera) for camera in cameras]
    except KeyError as exc:
        raise ValueError("camera rig document omits a camera definition field") from exc
    for camera, payload in zip(cameras, camera_payloads, strict=True):
        definition_digest = camera.get("definition_digest")
        if definition_digest != canonical_json_digest(payload):
            raise ValueError("camera rig document has a changed camera definition")
    if document.get("camera_set_digest") != canonical_json_digest(camera_payloads):
        raise ValueError("camera rig document camera-set digest does not match")
    verification = document.get("verification")
    not_requested = {
        "schema": VERIFICATION_EVIDENCE_SCHEMA,
        "schema_version": 1,
        "status": "not_requested",
    }
    precommit: Callable[[], None] | None = None
    if verification != not_requested:
        validated = _verification_record(
            verification,
            source_digest=source["digest"],
            resolution=(width, height),
            camera_paths=camera_paths,
            camera_ids=camera_ids,
            analysis_policy=analysis_policy,
            expected_overlay_actions=expected_overlay_actions,
        )
        if validated != verification:
            raise ValueError(
                "camera rig document verification changed during validation"
            )
        if verification_generation is None:
            raise ValueError(
                "verified camera rig publication requires a descriptor-bound "
                "verification generation"
            )
        precommit = lambda: verification_generation.validate(verification)
    result = publish_json_document(
        document,
        output,
        expected_schema=RIG_SCHEMA,
        precommit=precommit,
    )
    return result
