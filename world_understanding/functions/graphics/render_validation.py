# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract-neutral render evidence validation utilities."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
from PIL import Image as PILImage

type ImageInput = str | Path | PILImage.Image
type FrameId = str | int | float

RENDER_MISSING_OUTPUT = "render.missing_output"
RENDER_MALFORMED_RESPONSE = "render.malformed_response"
RENDER_IMAGE_DECODE_FAILED = "render.image_decode_failed"
RENDER_BLANK_IMAGE = "render.blank_image"
RENDER_LOW_CONTRAST = "render.low_contrast"
RENDER_SUSPECTED_ERROR_MATERIAL = "render.suspected_error_material"
RENDER_IDENTICAL_ANIMATION_FRAMES = "render.identical_animation_frames"

OVRTX_MISSING_FRAME = "ovrtx.missing_frame"
OVRTX_CAMERA_RESPONSE_MISSING = "ovrtx.camera_response_missing"
OVRTX_TIME_SAMPLE_VISIBILITY_FAILED = "ovrtx.time_sample_visibility_failed"
OVRTX_RENDER_ARTIFACT_DETECTED = "ovrtx.render_artifact_detected"


@dataclass(frozen=True)
class RenderValidationIssue:
    """Simple issue payload that can be mapped into future validation models."""

    code: str
    message: str
    subject: str | None = None
    severity: str = "error"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable issue dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "subject": self.subject,
            "severity": self.severity,
            "details": self.details,
        }


@dataclass(frozen=True)
class ImageValidationResult:
    """Validation result for one image artifact."""

    source: str | None
    readable: bool
    width: int | None
    height: int | None
    mode: str | None
    metrics: dict[str, Any] = field(default_factory=dict)
    issues: list[RenderValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether this image has no validation issues."""
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""
        return {
            "source": self.source,
            "readable": self.readable,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "metrics": self.metrics,
            "issues": [issue.to_dict() for issue in self.issues],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class DuplicateFramePair:
    """Pair of identical or near-identical frames."""

    first_index: int
    second_index: int
    first_frame: FrameId
    second_frame: FrameId
    mean_absolute_delta: float
    identical: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable frame pair dictionary."""
        return {
            "first_index": self.first_index,
            "second_index": self.second_index,
            "first_frame": self.first_frame,
            "second_frame": self.second_frame,
            "mean_absolute_delta": self.mean_absolute_delta,
            "identical": self.identical,
        }


@dataclass(frozen=True)
class DuplicateFrameValidationResult:
    """Validation result for duplicate animation evidence frames."""

    pairs: list[DuplicateFramePair] = field(default_factory=list)
    issues: list[RenderValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether no duplicate or near-duplicate frame pairs were found."""
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""
        return {
            "pairs": [pair.to_dict() for pair in self.pairs],
            "issues": [issue.to_dict() for issue in self.issues],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RenderResponseMetadata:
    """Render response metadata extracted without binding to a final contract."""

    backend: str | None = None
    cameras: list[str] = field(default_factory=list)
    frames: list[FrameId] = field(default_factory=list)
    sensors: list[str] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    malformed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata dictionary."""
        return {
            "backend": self.backend,
            "cameras": self.cameras,
            "frames": self.frames,
            "sensors": self.sensors,
            "output_paths": self.output_paths,
            "failures": self.failures,
            "malformed": self.malformed,
        }


@dataclass(frozen=True)
class RenderResponseValidationResult:
    """Validation result for a renderer response shape."""

    metadata: RenderResponseMetadata
    issues: list[RenderValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the render response has no validation issues."""
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "passed": self.passed,
        }


def validate_image_artifact(
    image: ImageInput,
    *,
    backend: str | None = None,
    min_width: int = 1,
    min_height: int = 1,
    max_analysis_pixels: int = 1_000_000,
    black_pixel_threshold: int = 3,
    white_pixel_threshold: int = 252,
    nonblack_pixel_threshold: float = 8.0,
    solid_ratio_threshold: float = 0.995,
    blank_std_threshold: float = 1.0,
    blank_dynamic_range_threshold: float = 2.0,
    low_contrast_std_threshold: float = 6.0,
    low_contrast_percentile_range_threshold: float = 16.0,
    detect_error_material_artifacts: bool = False,
    error_material_min_nonblack_ratio: float = 0.02,
    error_material_red_nonblack_ratio_threshold: float = 0.90,
    error_material_red_pixel_threshold: float = 240.0,
    error_material_red_dominance_ratio: float = 8.0,
) -> ImageValidationResult:
    """Check that an image can be decoded and is useful render evidence.

    The output uses simple dataclasses and generic issue codes so callers can
    map it into future Validation Agent result models without depending on
    those contracts. When opt-in error-material detection is enabled, saturated
    red fallback signatures are reported before generic blank/low-contrast
    classification so renderer material failures keep a specific issue code.
    ``nonblack_pixel_threshold`` controls the per-channel pixel value used to
    decide whether a pixel contributes to foreground/nonblack analysis.
    """
    source = _image_source(image)
    issues: list[RenderValidationIssue] = []

    try:
        loaded_image = _load_rgb_image(image)
    except FileNotFoundError as exc:
        issues.append(
            RenderValidationIssue(
                code=RENDER_MISSING_OUTPUT,
                message=str(exc),
                subject=source,
            )
        )
        return ImageValidationResult(source, False, None, None, None, {}, issues)
    except Exception as exc:
        issues.append(
            RenderValidationIssue(
                code=RENDER_IMAGE_DECODE_FAILED,
                message=f"Image could not be decoded: {exc}",
                subject=source,
                details={"error_type": type(exc).__name__},
            )
        )
        _append_ovrtx_artifact_issue(issues, backend, source)
        return ImageValidationResult(source, False, None, None, None, {}, issues)

    width, height = loaded_image.size
    if width < min_width or height < min_height:
        issues.append(
            RenderValidationIssue(
                code=RENDER_MISSING_OUTPUT,
                message=(
                    "Image dimensions are below the expected minimum: "
                    f"{width}x{height} < {min_width}x{min_height}"
                ),
                subject=source,
                details={
                    "width": width,
                    "height": height,
                    "min_width": min_width,
                    "min_height": min_height,
                },
            )
        )

    pixels = _sample_rgb_pixels(loaded_image, max_analysis_pixels)
    luma = _rgb_to_luma(pixels)
    metrics = _image_metrics(
        pixels,
        luma,
        width,
        height,
        nonblack_pixel_threshold=nonblack_pixel_threshold,
        red_pixel_threshold=error_material_red_pixel_threshold,
        red_dominance_ratio=error_material_red_dominance_ratio,
    )

    blank_kind = _classify_blank_image(
        pixels,
        luma,
        black_pixel_threshold=black_pixel_threshold,
        white_pixel_threshold=white_pixel_threshold,
        solid_ratio_threshold=solid_ratio_threshold,
        blank_std_threshold=blank_std_threshold,
        blank_dynamic_range_threshold=blank_dynamic_range_threshold,
    )
    suspected_error_material = (
        detect_error_material_artifacts
        and _is_suspected_error_material_render(
            metrics,
            min_nonblack_ratio=error_material_min_nonblack_ratio,
            red_nonblack_ratio_threshold=error_material_red_nonblack_ratio_threshold,
        )
    )
    if suspected_error_material:
        issues.append(
            RenderValidationIssue(
                code=RENDER_SUSPECTED_ERROR_MATERIAL,
                message=(
                    "Image is dominated by near-pure saturated red foreground "
                    "pixels, which is a common renderer error-material/fallback "
                    "signature."
                ),
                subject=source,
                details={
                    "nonblack_pixel_ratio": metrics["nonblack_pixel_ratio"],
                    "red_dominant_pixel_ratio": metrics["red_dominant_pixel_ratio"],
                    "red_dominant_nonblack_ratio": metrics[
                        "red_dominant_nonblack_ratio"
                    ],
                    "error_material_min_nonblack_ratio": (
                        error_material_min_nonblack_ratio
                    ),
                    "error_material_red_nonblack_ratio_threshold": (
                        error_material_red_nonblack_ratio_threshold
                    ),
                    "error_material_red_pixel_threshold": (
                        error_material_red_pixel_threshold
                    ),
                    "error_material_red_dominance_ratio": (
                        error_material_red_dominance_ratio
                    ),
                },
            )
        )
    elif blank_kind is not None:
        issues.append(
            RenderValidationIssue(
                code=RENDER_BLANK_IMAGE,
                message=f"Image is {blank_kind.replace('_', ' ')}.",
                subject=source,
                details={"kind": blank_kind},
            )
        )
    elif _is_low_contrast(
        luma,
        low_contrast_std_threshold=low_contrast_std_threshold,
        low_contrast_percentile_range_threshold=low_contrast_percentile_range_threshold,
    ):
        issues.append(
            RenderValidationIssue(
                code=RENDER_LOW_CONTRAST,
                message="Image has low luminance contrast.",
                subject=source,
                details={
                    "luma_std": metrics["luma_std"],
                    "luma_p95_minus_p05": metrics["luma_p95_minus_p05"],
                    "low_contrast_std_threshold": low_contrast_std_threshold,
                    "low_contrast_percentile_range_threshold": (
                        low_contrast_percentile_range_threshold
                    ),
                },
            )
        )

    _append_ovrtx_artifact_issue(issues, backend, source)

    return ImageValidationResult(
        source=source,
        readable=True,
        width=width,
        height=height,
        mode=loaded_image.mode,
        metrics=metrics,
        issues=issues,
    )


def detect_duplicate_frames(
    frames: Sequence[ImageInput],
    *,
    frame_ids: Sequence[FrameId] | None = None,
    backend: str | None = None,
    mean_absolute_delta_threshold: float = 0.01,
    comparison_size: tuple[int, int] = (32, 32),
) -> DuplicateFrameValidationResult:
    """Detect exact and near-duplicate frames in animation/render evidence."""
    if frame_ids is not None and len(frame_ids) != len(frames):
        raise ValueError("frame_ids must have the same length as frames")

    if len(frames) < 2:
        return DuplicateFrameValidationResult()

    ids: list[FrameId] = (
        list(frame_ids) if frame_ids is not None else list(range(len(frames)))
    )
    issues: list[RenderValidationIssue] = []
    pairs: list[DuplicateFramePair] = []
    fingerprints: list[np.ndarray] = []

    for index, frame in enumerate(frames):
        try:
            fingerprints.append(_frame_fingerprint(frame, comparison_size))
        except Exception as exc:
            issues.append(
                RenderValidationIssue(
                    code=RENDER_IMAGE_DECODE_FAILED,
                    message=f"Frame {ids[index]!r} could not be decoded: {exc}",
                    subject=_image_source(frame),
                    details={"frame": ids[index], "error_type": type(exc).__name__},
                )
            )

    if issues:
        _append_ovrtx_artifact_issue(issues, backend, None)
        return DuplicateFrameValidationResult(pairs, issues)

    for first_index in range(len(fingerprints)):
        for second_index in range(first_index + 1, len(fingerprints)):
            mean_delta = float(
                np.mean(np.abs(fingerprints[first_index] - fingerprints[second_index]))
            )
            if mean_delta <= mean_absolute_delta_threshold:
                pairs.append(
                    DuplicateFramePair(
                        first_index=first_index,
                        second_index=second_index,
                        first_frame=ids[first_index],
                        second_frame=ids[second_index],
                        mean_absolute_delta=mean_delta,
                        identical=mean_delta == 0.0,
                    )
                )

    if pairs:
        issues.append(
            RenderValidationIssue(
                code=RENDER_IDENTICAL_ANIMATION_FRAMES,
                message="Animation evidence contains identical or near-identical frames.",
                details={
                    "pairs": [pair.to_dict() for pair in pairs],
                    "mean_absolute_delta_threshold": mean_absolute_delta_threshold,
                },
            )
        )
        _append_ovrtx_artifact_issue(issues, backend, None)

    return DuplicateFrameValidationResult(pairs, issues)


def extract_render_response_metadata(
    response: Any,
    *,
    backend: str | None = None,
    image_keys: Sequence[str] = (
        "images",
        "image_files",
        "output_paths",
        "outputs",
    ),
) -> RenderResponseMetadata:
    """Extract best-effort render metadata from common response shapes."""
    raw_results, malformed = _extract_result_entries(response)
    entries = [entry for entry in raw_results if isinstance(entry, Mapping)]

    cameras: list[str] = []
    frames: list[FrameId] = []
    sensors: list[str] = []
    output_paths: list[str] = []
    failures: list[str] = []

    if isinstance(response, Mapping):
        cameras.extend(_coerce_strings(response.get("cameras")))
        frames.extend(_coerce_frame_ids(response.get("frames")))
        sensors.extend(_coerce_strings(response.get("sensors")))
        failures.extend(_coerce_failures(response.get("failures")))
        failures.extend(_coerce_failures(response.get("errors")))

    for entry in entries:
        camera = _entry_camera(entry)
        if camera is not None:
            cameras.append(camera)

        frames.extend(_coerce_frame_ids(_first_present(entry, "frames", "frame_ids")))
        sensors.extend(_entry_sensors(entry))
        failures.extend(_coerce_failures(entry.get("failures")))
        failure = entry.get("error")
        if failure is not None:
            failures.append(str(failure))

        image_entries = _entry_image_entries(entry, image_keys)
        if image_entries is not None:
            output_paths.extend(_image_entry_paths(image_entries))

    return RenderResponseMetadata(
        backend=backend,
        cameras=_dedupe_preserve_order(cameras),
        frames=_dedupe_preserve_order(frames),
        sensors=_dedupe_preserve_order(sensors),
        output_paths=_dedupe_preserve_order(output_paths),
        failures=_dedupe_preserve_order(failures),
        malformed=malformed,
    )


def validate_render_response(
    response: Any,
    *,
    expected_cameras: Sequence[str] | None = None,
    expected_frames: Sequence[FrameId] | None = None,
    backend: str | None = None,
    image_keys: Sequence[str] = (
        "images",
        "image_files",
        "output_paths",
        "outputs",
    ),
) -> RenderResponseValidationResult:
    """Validate that a mocked or real renderer response has expected outputs."""
    issues: list[RenderValidationIssue] = []
    raw_results, malformed = _extract_result_entries(response)
    if malformed:
        issues.append(
            RenderValidationIssue(
                code=RENDER_MALFORMED_RESPONSE,
                message="Render response is not a mapping or a sequence of mappings.",
            )
        )

    entries: list[Mapping[str, Any]] = []
    for index, raw_result in enumerate(raw_results):
        if not isinstance(raw_result, Mapping):
            issues.append(
                RenderValidationIssue(
                    code=RENDER_MALFORMED_RESPONSE,
                    message=f"Render response entry {index} is not a mapping.",
                    subject=f"results[{index}]",
                )
            )
            continue
        entries.append(raw_result)

    if not entries and not malformed:
        issues.append(
            RenderValidationIssue(
                code=RENDER_MALFORMED_RESPONSE,
                message="Render response does not contain any camera result entries.",
            )
        )

    entries_by_camera = _entries_by_camera(entries, issues)
    camera_entries = _select_camera_entries(
        entries, expected_cameras, entries_by_camera
    )
    _validate_expected_cameras(
        expected_cameras,
        entries_by_camera,
        issues,
        backend=backend,
    )

    for camera, entry in camera_entries:
        _validate_image_entries(
            camera,
            entry,
            image_keys,
            expected_frames,
            issues,
            backend=backend,
        )

    metadata = extract_render_response_metadata(
        response,
        backend=backend,
        image_keys=image_keys,
    )
    issues.extend(_ovrtx_failure_issues(metadata))

    return RenderResponseValidationResult(metadata=metadata, issues=issues)


def _image_source(image: ImageInput) -> str | None:
    if isinstance(image, str | Path):
        return str(image)
    return None


def _load_rgb_image(image: ImageInput) -> PILImage.Image:
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image output does not exist: {path}")
        with PILImage.open(path) as opened:
            opened.load()
            return opened.convert("RGB")

    if isinstance(image, PILImage.Image):
        copied = image.copy()
        copied.load()
        return copied.convert("RGB")

    raise TypeError(f"Unsupported image input type: {type(image)}")


def _sample_rgb_pixels(image: PILImage.Image, max_pixels: int) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    pixels = array.reshape(-1, 3)
    if len(pixels) <= max_pixels:
        return pixels

    stride = max(1, math.ceil(len(pixels) / max_pixels))
    return pixels[::stride]


def _rgb_to_luma(pixels: np.ndarray) -> np.ndarray:
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return pixels @ weights


def _image_metrics(
    pixels: np.ndarray,
    luma: np.ndarray,
    width: int,
    height: int,
    *,
    nonblack_pixel_threshold: float = 8.0,
    red_pixel_threshold: float = 240.0,
    red_dominance_ratio: float = 8.0,
) -> dict[str, Any]:
    luma_p05 = float(np.percentile(luma, 5))
    luma_p95 = float(np.percentile(luma, 95))
    unique_colors = int(np.unique(pixels.astype(np.uint8), axis=0).shape[0])
    black_ratio = float(np.mean(np.all(pixels <= 3, axis=1)))
    white_ratio = float(np.mean(np.all(pixels >= 252, axis=1)))
    nonblack_mask = np.any(pixels > nonblack_pixel_threshold, axis=1)
    nonblack_ratio = float(np.mean(nonblack_mask))
    red_dominant_mask = (
        (pixels[:, 0] >= red_pixel_threshold)
        & (pixels[:, 0] > pixels[:, 1] * red_dominance_ratio)
        & (pixels[:, 0] > pixels[:, 2] * red_dominance_ratio)
    )
    red_dominant_ratio = float(np.mean(red_dominant_mask))
    red_dominant_nonblack_ratio = (
        float(np.mean(red_dominant_mask[nonblack_mask]))
        if bool(np.any(nonblack_mask))
        else 0.0
    )

    return {
        "width": width,
        "height": height,
        "sampled_pixels": int(len(pixels)),
        "unique_colors": unique_colors,
        "rgb_min": float(np.min(pixels)),
        "rgb_max": float(np.max(pixels)),
        "luma_min": float(np.min(luma)),
        "luma_max": float(np.max(luma)),
        "luma_mean": float(np.mean(luma)),
        "luma_std": float(np.std(luma)),
        "luma_dynamic_range": float(np.max(luma) - np.min(luma)),
        "luma_p05": luma_p05,
        "luma_p95": luma_p95,
        "luma_p95_minus_p05": luma_p95 - luma_p05,
        "black_pixel_ratio": black_ratio,
        "white_pixel_ratio": white_ratio,
        "nonblack_pixel_ratio": nonblack_ratio,
        "red_dominant_pixel_ratio": red_dominant_ratio,
        "red_dominant_nonblack_ratio": red_dominant_nonblack_ratio,
    }


def _classify_blank_image(
    pixels: np.ndarray,
    luma: np.ndarray,
    *,
    black_pixel_threshold: int,
    white_pixel_threshold: int,
    solid_ratio_threshold: float,
    blank_std_threshold: float,
    blank_dynamic_range_threshold: float,
) -> str | None:
    black_ratio = float(np.mean(np.all(pixels <= black_pixel_threshold, axis=1)))
    if black_ratio >= solid_ratio_threshold:
        return "all_black"

    white_ratio = float(np.mean(np.all(pixels >= white_pixel_threshold, axis=1)))
    if white_ratio >= solid_ratio_threshold:
        return "all_white"

    luma_std = float(np.std(luma))
    luma_dynamic_range = float(np.max(luma) - np.min(luma))
    if (
        luma_std <= blank_std_threshold
        or luma_dynamic_range <= blank_dynamic_range_threshold
    ):
        return "near_blank"

    return None


def _is_low_contrast(
    luma: np.ndarray,
    *,
    low_contrast_std_threshold: float,
    low_contrast_percentile_range_threshold: float,
) -> bool:
    luma_std = float(np.std(luma))
    percentile_range = float(np.percentile(luma, 95) - np.percentile(luma, 5))
    return (
        luma_std <= low_contrast_std_threshold
        and percentile_range <= low_contrast_percentile_range_threshold
    )


def _is_suspected_error_material_render(
    metrics: Mapping[str, Any],
    *,
    min_nonblack_ratio: float,
    red_nonblack_ratio_threshold: float,
) -> bool:
    nonblack_ratio = float(metrics.get("nonblack_pixel_ratio", 0.0))
    red_nonblack_ratio = float(metrics.get("red_dominant_nonblack_ratio", 0.0))
    return (
        nonblack_ratio >= min_nonblack_ratio
        and red_nonblack_ratio >= red_nonblack_ratio_threshold
    )


def _frame_fingerprint(
    frame: ImageInput,
    comparison_size: tuple[int, int],
) -> np.ndarray:
    image = _load_rgb_image(frame)
    resampling = getattr(PILImage, "Resampling", PILImage).BILINEAR
    resized = image.convert("L").resize(comparison_size, resampling)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _append_ovrtx_artifact_issue(
    issues: list[RenderValidationIssue],
    backend: str | None,
    subject: str | None,
) -> None:
    if not _is_ovrtx_backend(backend):
        return

    related_codes = [
        issue.code
        for issue in issues
        if issue.code
        in {
            RENDER_IMAGE_DECODE_FAILED,
            RENDER_BLANK_IMAGE,
            RENDER_LOW_CONTRAST,
            RENDER_SUSPECTED_ERROR_MATERIAL,
            RENDER_IDENTICAL_ANIMATION_FRAMES,
        }
    ]
    if not related_codes:
        return

    issues.append(
        RenderValidationIssue(
            code=OVRTX_RENDER_ARTIFACT_DETECTED,
            message="OVRTX render output appears invalid or artifacted.",
            subject=subject,
            details={"related_codes": related_codes},
        )
    )


def _extract_result_entries(response: Any) -> tuple[list[Any], bool]:
    if isinstance(response, Mapping):
        results = response.get("results")
        if results is None and _looks_like_camera_entry(response):
            return [response], False
        if _is_non_string_sequence(results):
            return list(results), False
        return [], True

    if _is_non_string_sequence(response):
        return list(response), False

    return [], True


def _looks_like_camera_entry(response: Mapping[str, Any]) -> bool:
    known_keys = {"camera", "camera_path", "images", "image_files", "output_paths"}
    return bool(known_keys.intersection(response.keys()))


def _entries_by_camera(
    entries: Sequence[Mapping[str, Any]],
    issues: list[RenderValidationIssue],
) -> dict[str, Mapping[str, Any]]:
    entries_by_camera: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(entries):
        camera = _entry_camera(entry)
        if camera is None:
            issues.append(
                RenderValidationIssue(
                    code=RENDER_MALFORMED_RESPONSE,
                    message=f"Render response entry {index} is missing a camera.",
                    subject=f"results[{index}]",
                )
            )
            continue
        if camera in entries_by_camera:
            issues.append(
                RenderValidationIssue(
                    code=RENDER_MALFORMED_RESPONSE,
                    message=f"Render response contains duplicate camera {camera!r}.",
                    subject=camera,
                )
            )
            continue
        entries_by_camera[camera] = entry
    return entries_by_camera


def _select_camera_entries(
    entries: Sequence[Mapping[str, Any]],
    expected_cameras: Sequence[str] | None,
    entries_by_camera: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    if expected_cameras is not None:
        return [
            (camera, entries_by_camera[camera])
            for camera in expected_cameras
            if camera in entries_by_camera
        ]

    selected: list[tuple[str, Mapping[str, Any]]] = []
    for index, entry in enumerate(entries):
        camera = _entry_camera(entry) or f"results[{index}]"
        selected.append((camera, entry))
    return selected


def _validate_expected_cameras(
    expected_cameras: Sequence[str] | None,
    entries_by_camera: Mapping[str, Mapping[str, Any]],
    issues: list[RenderValidationIssue],
    *,
    backend: str | None,
) -> None:
    if expected_cameras is None:
        return

    for camera in expected_cameras:
        if camera in entries_by_camera:
            continue
        issues.append(
            RenderValidationIssue(
                code=RENDER_MISSING_OUTPUT,
                message=f"Render response is missing expected camera {camera!r}.",
                subject=camera,
                details={"camera": camera},
            )
        )
        if _is_ovrtx_backend(backend):
            issues.append(
                RenderValidationIssue(
                    code=OVRTX_CAMERA_RESPONSE_MISSING,
                    message=f"OVRTX response is missing camera {camera!r}.",
                    subject=camera,
                    details={"camera": camera},
                )
            )


def _validate_image_entries(
    camera: str,
    entry: Mapping[str, Any],
    image_keys: Sequence[str],
    expected_frames: Sequence[FrameId] | None,
    issues: list[RenderValidationIssue],
    *,
    backend: str | None,
) -> None:
    image_entries = _entry_image_entries(entry, image_keys)
    if image_entries is None:
        issues.append(
            RenderValidationIssue(
                code=RENDER_MISSING_OUTPUT,
                message=f"Render response for camera {camera!r} has no image entries.",
                subject=camera,
                details={"camera": camera, "image_keys": list(image_keys)},
            )
        )
        return

    if not _is_non_string_sequence(image_entries):
        issues.append(
            RenderValidationIssue(
                code=RENDER_MALFORMED_RESPONSE,
                message=f"Image entries for camera {camera!r} are not a sequence.",
                subject=camera,
                details={"camera": camera},
            )
        )
        return

    image_list = list(image_entries)
    if not image_list:
        issues.append(
            RenderValidationIssue(
                code=RENDER_MISSING_OUTPUT,
                message=f"Render response for camera {camera!r} has no images.",
                subject=camera,
                details={"camera": camera},
            )
        )

    has_missing_image_entry = False
    for image_index, image_entry in enumerate(image_list):
        if _is_missing_image_entry(image_entry):
            has_missing_image_entry = True
            issues.append(
                RenderValidationIssue(
                    code=RENDER_MISSING_OUTPUT,
                    message=(
                        f"Render response for camera {camera!r} has an empty "
                        f"image entry at index {image_index}."
                    ),
                    subject=camera,
                    details={"camera": camera, "image_index": image_index},
                )
            )

    frame_image_count = (
        len(image_list)
        if has_missing_image_entry
        else sum(
            1 for image_entry in image_list if not _is_missing_image_entry(image_entry)
        )
    )
    _validate_expected_frames(
        camera,
        entry,
        image_count=frame_image_count,
        expected_frames=expected_frames,
        issues=issues,
        backend=backend,
    )


def _validate_expected_frames(
    camera: str,
    entry: Mapping[str, Any],
    *,
    image_count: int,
    expected_frames: Sequence[FrameId] | None,
    issues: list[RenderValidationIssue],
    backend: str | None,
) -> None:
    rendered_frames = _coerce_frame_ids(_first_present(entry, "frames", "frame_ids"))
    if expected_frames is None:
        if rendered_frames and image_count < len(rendered_frames):
            missing_frames = rendered_frames[image_count:]
        else:
            return
    else:
        expected_frame_list = list(expected_frames)
        missing_frames = []
        if rendered_frames:
            rendered_frame_set = set(rendered_frames)
            missing_frames.extend(
                frame
                for frame in expected_frame_list
                if frame not in rendered_frame_set
            )
            if image_count < len(rendered_frames):
                missing_frames.extend(rendered_frames[image_count:])
        elif image_count < len(expected_frame_list):
            missing_frames.extend(expected_frame_list[image_count:])

        missing_frames = _dedupe_preserve_order(missing_frames)

    if not missing_frames:
        return

    details: dict[str, Any] = {
        "camera": camera,
        "missing_frames": missing_frames,
        "image_count": image_count,
    }
    if expected_frames is not None:
        details["expected_frame_count"] = len(expected_frames)
    if rendered_frames:
        details["rendered_frames"] = rendered_frames

    issues.append(
        RenderValidationIssue(
            code=RENDER_MISSING_OUTPUT,
            message=(
                f"Render response for camera {camera!r} is missing frames "
                "or image outputs."
            ),
            subject=camera,
            details=details,
        )
    )
    if _is_ovrtx_backend(backend):
        issues.append(
            RenderValidationIssue(
                code=OVRTX_MISSING_FRAME,
                message=(
                    f"OVRTX response for camera {camera!r} is missing frames "
                    "or image outputs."
                ),
                subject=camera,
                details=details,
            )
        )


def _entry_camera(entry: Mapping[str, Any]) -> str | None:
    camera = _first_present(entry, "camera", "camera_path", "camera_name")
    if camera is None:
        return None
    return str(camera)


def _entry_image_entries(
    entry: Mapping[str, Any],
    image_keys: Sequence[str],
) -> Any | None:
    saw_empty_sequence = False
    empty_entries: list[Any] = []
    for key in image_keys:
        if key not in entry:
            continue
        raw_entries = entry[key]
        if raw_entries is None:
            continue
        if not _is_non_string_sequence(raw_entries):
            return raw_entries
        image_entries = list(raw_entries)
        if image_entries:
            return image_entries
        saw_empty_sequence = True
    if saw_empty_sequence:
        return empty_entries
    return None


def _entry_sensors(entry: Mapping[str, Any]) -> list[str]:
    sensors = _coerce_strings(entry.get("sensors"))
    sensor_files = entry.get("sensor_files")
    if isinstance(sensor_files, Mapping):
        sensors.extend(str(sensor_name) for sensor_name in sensor_files.keys())
    return sensors


def _image_entry_paths(image_entries: Any) -> list[str]:
    if not _is_non_string_sequence(image_entries):
        return []
    paths: list[str] = []
    for entry in image_entries:
        if isinstance(entry, str | Path) and str(entry):
            paths.append(str(entry))
    return paths


def _is_missing_image_entry(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str | Path) and not str(value):
        return True
    return False


def _ovrtx_failure_issues(
    metadata: RenderResponseMetadata,
) -> list[RenderValidationIssue]:
    if not _is_ovrtx_backend(metadata.backend):
        return []

    issues: list[RenderValidationIssue] = []
    for failure in metadata.failures:
        normalized = failure.lower()
        if "visibility" in normalized and (
            "time" in normalized or "sample" in normalized
        ):
            issues.append(
                RenderValidationIssue(
                    code=OVRTX_TIME_SAMPLE_VISIBILITY_FAILED,
                    message="OVRTX reported a time-sampled visibility failure.",
                    details={"failure": failure},
                )
            )
    return issues


def _coerce_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if _is_non_string_sequence(value):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_frame_ids(value: Any) -> list[FrameId]:
    if value is None:
        return []
    if isinstance(value, str | int | float):
        return [value]
    if _is_non_string_sequence(value):
        frame_ids: list[FrameId] = []
        for item in value:
            if isinstance(item, str | int | float):
                frame_ids.append(item)
            else:
                frame_ids.append(str(item))
        return frame_ids
    return [str(value)]


def _coerce_failures(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [f"{key}: {failure}" for key, failure in value.items()]
    if _is_non_string_sequence(value):
        return [str(failure) for failure in value]
    return [str(value)]


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _dedupe_preserve_order[T: Hashable](items: Sequence[T]) -> list[T]:
    seen: set[T] = set()
    result: list[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _is_non_string_sequence(value: Any) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _is_ovrtx_backend(backend: str | None) -> bool:
    return backend is not None and backend.strip().lower() == "ovrtx"
