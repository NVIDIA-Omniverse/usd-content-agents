# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation and integrity metadata for external-runtime image evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .types import EvidenceSettings


class EvidenceValidationError(ValueError):
    """Required BYOR image evidence is missing or invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_current_usd_layer(path: Path, sdf: Any) -> None:
    """Fail closed when OpenUSD's process-global layer cache is stale."""

    try:
        cached = sdf.Layer.FindOrOpen(str(path))
        fresh = sdf.Layer.OpenAsAnonymous(str(path))
    except Exception as exc:  # noqa: BLE001 - normalize OpenUSD read failures
        raise EvidenceValidationError(
            f"could not establish a fresh read of rollout recording: {path}"
        ) from exc
    if cached is None or fresh is None:
        raise EvidenceValidationError(
            f"could not establish a fresh read of rollout recording: {path}"
        )
    if bool(getattr(cached, "dirty", False)):
        raise EvidenceValidationError(
            "rollout recording has unsaved edits in OpenUSD's layer cache"
        )
    try:
        cached_text = cached.ExportToString()
        fresh_text = fresh.ExportToString()
    except Exception as exc:  # noqa: BLE001 - normalize OpenUSD export failures
        raise EvidenceValidationError(
            f"could not compare rollout recording with a fresh read: {path}"
        ) from exc
    if bool(getattr(cached, "dirty", False)):
        raise EvidenceValidationError(
            "rollout recording changed while validating OpenUSD's layer cache"
        )
    if cached_text != fresh_text:
        raise EvidenceValidationError(
            "rollout recording differs from OpenUSD's cached layer; "
            "release cached stages or retry in a fresh process"
        )


def inspect_file_artifact(path: Path, *, relative_path: str) -> dict[str, Any]:
    """Return integrity metadata for a validated adapter output file."""

    return {
        "path": relative_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def inspect_frame_evidence(
    path: Path,
    settings: EvidenceSettings,
    adapter_metadata: Mapping[str, Any],
    *,
    relative_path: str,
) -> dict[str, Any]:
    """Validate a PNG-frame manifest and bind every referenced image byte."""

    evidence_metadata = adapter_metadata.get("evidence")
    if not isinstance(evidence_metadata, Mapping):
        raise EvidenceValidationError("external trial metadata.evidence is required")
    if evidence_metadata.get("renderer") != settings.renderer:
        raise EvidenceValidationError(
            "external trial evidence renderer changed: "
            f"expected {settings.renderer!r}, got {evidence_metadata.get('renderer')!r}"
        )
    for name, expected in (("width", settings.width), ("height", settings.height)):
        if evidence_metadata.get(name) != expected:
            raise EvidenceValidationError(
                f"external trial evidence {name} must be {expected}"
            )
    reported_fps = evidence_metadata.get("fps")
    if (
        isinstance(reported_fps, bool)
        or not isinstance(reported_fps, int | float)
        or not math.isclose(
            float(reported_fps), settings.fps, rel_tol=0.02, abs_tol=0.05
        )
    ):
        raise EvidenceValidationError(
            f"external trial evidence fps must be {settings.fps:g}"
        )

    if path.suffix.lower() != ".json" or not path.is_file():
        raise EvidenceValidationError(
            f"frame evidence must be a JSON manifest file: {path}"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError(
            f"could not read frame evidence manifest: {path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise EvidenceValidationError("frame evidence manifest must be an object")
    if set(manifest) != {
        "schema_version",
        "renderer",
        "width",
        "height",
        "fps",
        "frames",
    }:
        raise EvidenceValidationError(
            "frame evidence manifest fields do not match the public contract"
        )
    if manifest.get("schema_version") != "physics-agent.qualification-frames.v1":
        raise EvidenceValidationError("unsupported frame evidence manifest schema")
    if manifest.get("renderer") != settings.renderer:
        raise EvidenceValidationError("frame evidence renderer changed")
    if (
        manifest.get("width") != settings.width
        or manifest.get("height") != settings.height
    ):
        raise EvidenceValidationError("frame evidence dimensions changed")
    manifest_fps = manifest.get("fps")
    if (
        isinstance(manifest_fps, bool)
        or not isinstance(manifest_fps, int | float)
        or not math.isfinite(float(manifest_fps))
        or not math.isclose(
            float(manifest_fps), settings.fps, rel_tol=0.02, abs_tol=0.05
        )
    ):
        raise EvidenceValidationError(
            f"frame evidence fps changed: expected {settings.fps:g}"
        )
    manifest_frames = manifest.get("frames")
    if not isinstance(manifest_frames, list):
        raise EvidenceValidationError("frame evidence manifest frames must be a list")
    if len(manifest_frames) < settings.min_frames:
        raise EvidenceValidationError(
            "frame evidence has too few PNG frames: "
            f"{len(manifest_frames)} < {settings.min_frames}"
        )

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - core dependency invariant
        raise EvidenceValidationError(
            "Pillow and NumPy are required to validate external frame evidence"
        ) from exc

    root = path.parent.resolve()
    relative_manifest_parent = Path(relative_path).parent
    frame_descriptors: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    max_frame_stddev = 0.0
    max_motion_score = 0.0
    previous: Any | None = None
    for index, entry in enumerate(manifest_frames):
        if not isinstance(entry, dict) or set(entry) != {"path", "timestamp_seconds"}:
            raise EvidenceValidationError(
                f"frame evidence entry {index} must contain path and timestamp_seconds"
            )
        raw_path = entry.get("path")
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or "\\" in raw_path
            or Path(raw_path).is_absolute()
            or Path(raw_path).suffix.lower() != ".png"
            or ".." in Path(raw_path).parts
            or raw_path in seen_paths
        ):
            raise EvidenceValidationError(
                f"frame evidence entry {index} has an invalid PNG path"
            )
        seen_paths.add(raw_path)
        timestamp = entry.get("timestamp_seconds")
        expected_timestamp = index / float(manifest_fps)
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int | float)
            or not math.isfinite(float(timestamp))
            or not math.isclose(
                float(timestamp), expected_timestamp, rel_tol=1.0e-6, abs_tol=1.0e-6
            )
        ):
            raise EvidenceValidationError(
                f"frame evidence entry {index} has an invalid timestamp"
            )

        candidate = path.parent / Path(raw_path)
        current = path.parent
        for part in Path(raw_path).parts:
            current = current / part
            if current.is_symlink():
                raise EvidenceValidationError(
                    f"frame evidence entry {index} traverses a symbolic link"
                )
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise EvidenceValidationError(
                f"frame evidence entry {index} escaped the artifact directory"
            ) from None
        if not resolved.is_file():
            raise EvidenceValidationError(
                f"frame evidence entry {index} is missing: {raw_path}"
            )
        try:
            with Image.open(resolved) as image:
                image.verify()
            with Image.open(resolved) as image:
                if image.format != "PNG" or image.size != (
                    settings.width,
                    settings.height,
                ):
                    raise EvidenceValidationError(
                        f"frame evidence entry {index} is not the required PNG size"
                    )
                frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except EvidenceValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize Pillow failures
            raise EvidenceValidationError(
                f"could not decode PNG frame evidence entry {index}: {raw_path}"
            ) from exc
        max_frame_stddev = max(max_frame_stddev, float(frame.std()))
        if previous is not None:
            difference = np.abs(frame.astype(np.int16) - previous.astype(np.int16))
            max_motion_score = max(max_motion_score, float(difference.mean()))
        previous = frame
        frame_descriptors.append(
            {
                "path": (relative_manifest_parent / Path(raw_path)).as_posix(),
                "sha256": _sha256(resolved),
                "size_bytes": resolved.stat().st_size,
                "media_type": "image/png",
                "width": settings.width,
                "height": settings.height,
                "timestamp_seconds": float(timestamp),
            }
        )

    if max_frame_stddev < settings.min_frame_stddev:
        raise EvidenceValidationError(
            "frame evidence is blank or visually uniform: "
            f"frame stddev {max_frame_stddev:.6g} < {settings.min_frame_stddev:.6g}"
        )
    if settings.require_motion and max_motion_score < settings.min_motion_score:
        raise EvidenceValidationError(
            "frame evidence contains no detectable motion: "
            f"score {max_motion_score:.6g} < {settings.min_motion_score:.6g}"
        )

    return {
        "path": relative_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "media_type": settings.media_type,
        "renderer": settings.renderer,
        "width": settings.width,
        "height": settings.height,
        "fps": float(manifest_fps),
        "frame_count": len(frame_descriptors),
        "max_frame_stddev": max_frame_stddev,
        "max_motion_score": max_motion_score,
        "frames": frame_descriptors,
    }


def inspect_usd_recording(
    path: Path,
    settings: EvidenceSettings,
    *,
    relative_path: str,
) -> dict[str, Any]:
    """Validate a time-sampled USD produced by one evaluated rollout."""

    if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise EvidenceValidationError(
            f"rollout recording must be USD, got {path.suffix or '<no suffix>'!r}"
        )
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdUtils
    except ImportError as exc:  # pragma: no cover - Physics Agent dependency invariant
        raise EvidenceValidationError(
            "OpenUSD is required to validate external rollout recordings"
        ) from exc
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as exc:  # noqa: BLE001 - normalize OpenUSD parser failures
        raise EvidenceValidationError(
            f"could not open rollout recording: {path}"
        ) from exc
    if stage is None:
        raise EvidenceValidationError(f"could not open rollout recording: {path}")

    resolved_path = path.resolve()
    _require_current_usd_layer(resolved_path, Sdf)
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(resolved_path))
    except Exception as exc:  # noqa: BLE001 - normalize OpenUSD inspection failures
        raise EvidenceValidationError(
            f"could not inspect rollout recording dependencies: {path}"
        ) from exc
    _require_current_usd_layer(resolved_path, Sdf)
    recording_path = resolved_path
    external_layers: list[str] = []
    for layer in layers:
        identifier = str(layer.realPath or layer.identifier)
        try:
            is_recording_layer = Path(identifier).resolve() == recording_path
        except (OSError, RuntimeError, ValueError):
            is_recording_layer = False
        if not is_recording_layer:
            external_layers.append(identifier)
    dependency_failures: list[str] = []
    if external_layers:
        dependency_failures.append(
            "external layers: " + ", ".join(sorted(external_layers))
        )
    if assets:
        dependency_failures.append(
            "external assets: " + ", ".join(sorted(str(asset) for asset in assets))
        )
    if unresolved:
        dependency_failures.append(
            "unresolved paths: " + ", ".join(sorted(str(asset) for asset in unresolved))
        )
    if dependency_failures:
        raise EvidenceValidationError(
            "rollout recording must be a self-contained single-file USD; "
            + "; ".join(dependency_failures)
        )

    fps = float(stage.GetTimeCodesPerSecond() or 0.0)
    if not math.isfinite(fps) or not math.isclose(
        fps, settings.fps, rel_tol=0.02, abs_tol=0.05
    ):
        raise EvidenceValidationError(
            f"rollout recording fps must be {settings.fps:g}, got {fps:g}"
        )
    frames_per_second = float(stage.GetFramesPerSecond() or 0.0)
    if not math.isfinite(frames_per_second) or not math.isclose(
        frames_per_second, settings.fps, rel_tol=0.02, abs_tol=0.05
    ):
        raise EvidenceValidationError(
            "rollout recording framesPerSecond must be "
            f"{settings.fps:g}, got {frames_per_second:g}"
        )
    if not stage.HasAuthoredTimeCodeRange():
        raise EvidenceValidationError(
            "rollout recording must author startTimeCode and endTimeCode"
        )
    start_time_code = float(stage.GetStartTimeCode())
    end_time_code = float(stage.GetEndTimeCode())
    if (
        not math.isfinite(start_time_code)
        or not math.isfinite(end_time_code)
        or end_time_code < start_time_code
        or not start_time_code.is_integer()
        or not end_time_code.is_integer()
    ):
        raise EvidenceValidationError(
            "rollout recording time-code range must be finite, integral, and ordered"
        )
    frame_range_count = int(end_time_code - start_time_code) + 1
    max_frames = max(1, int(settings.max_duration_seconds * fps)) + 1
    if frame_range_count > max_frames:
        raise EvidenceValidationError(
            "rollout recording frame range exceeds playback cap: "
            f"{frame_range_count} > {max_frames}"
        )

    time_samples: set[float] = set()
    pose_prim_paths: list[str] = []
    camera_prim_paths: list[str] = []
    rotation_types = {
        UsdGeom.XformOp.TypeOrient,
        UsdGeom.XformOp.TypeRotateX,
        UsdGeom.XformOp.TypeRotateY,
        UsdGeom.XformOp.TypeRotateZ,
        UsdGeom.XformOp.TypeRotateXYZ,
        UsdGeom.XformOp.TypeRotateXZY,
        UsdGeom.XformOp.TypeRotateYXZ,
        UsdGeom.XformOp.TypeRotateYZX,
        UsdGeom.XformOp.TypeRotateZXY,
        UsdGeom.XformOp.TypeRotateZYX,
    }
    for prim in stage.TraverseAll():
        if prim.IsInstanceProxy():
            continue
        if prim.IsA(UsdGeom.Camera):
            camera_prim_paths.append(str(prim.GetPath()))
        for attribute in prim.GetAttributes():
            time_samples.update(float(value) for value in attribute.GetTimeSamples())
        if not prim.IsA(UsdGeom.Xformable):
            continue
        translate_samples: set[float] = set()
        rotation_samples: set[float] = set()
        transform_samples: set[float] = set()
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            samples = {float(value) for value in op.GetAttr().GetTimeSamples()}
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_samples.update(samples)
            elif op.GetOpType() in rotation_types:
                rotation_samples.update(samples)
            elif op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                transform_samples.update(samples)
        has_matrix_poses = len(transform_samples) >= settings.min_frames
        has_split_poses = (
            len(translate_samples & rotation_samples) >= settings.min_frames
        )
        if has_matrix_poses or has_split_poses:
            pose_prim_paths.append(str(prim.GetPath()))
    if len(time_samples) < settings.min_frames:
        raise EvidenceValidationError(
            "rollout recording has too few authored time samples: "
            f"{len(time_samples)} < {settings.min_frames}"
        )
    if not pose_prim_paths:
        raise EvidenceValidationError(
            "rollout recording must contain time-sampled position and orientation"
        )
    if not camera_prim_paths:
        raise EvidenceValidationError(
            "rollout recording must contain a camera for deterministic playback"
        )
    if any(
        not math.isfinite(value)
        or not value.is_integer()
        or value < start_time_code
        or value > end_time_code
        for value in time_samples
    ):
        raise EvidenceValidationError(
            "rollout recording time samples must be integral and inside its frame range"
        )

    ordered = sorted(time_samples)
    return {
        "path": relative_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "media_type": "model/vnd.usd",
        "fps": fps,
        "frame_count": len(ordered),
        "start_time_code": start_time_code,
        "end_time_code": end_time_code,
        "pose_prim_paths": pose_prim_paths,
        "camera_prim_paths": camera_prim_paths,
    }


__all__ = [
    "EvidenceValidationError",
    "inspect_file_artifact",
    "inspect_frame_evidence",
    "inspect_usd_recording",
]
