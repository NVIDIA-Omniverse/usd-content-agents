# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, code-owned diagnostics for known renderer failures."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, TypeGuard

from PIL import Image as PILImage
from PIL import ImageStat

from world_understanding.utils.durable_diagnostics import (
    DIAGNOSTIC_SCHEMA,
    FailurePhase,
)

PIPELINE_FAILURE_DIAGNOSTIC_CONTEXT_KEY = "pipeline_failure_diagnostic"
BLANK_DATASET_RENDERS_CODE = "blank_dataset_renders"
BLANK_DATASET_RENDERS_STEP = "build_dataset_usd"
RENDER_FAILURE_REPORT_SCHEMA = "world-understanding-render-failure-report-v1"

_MAX_DIAGNOSTIC_SAMPLES = 4
_MAX_REPORT_IMAGE_SIZE = 256
_MAX_EVIDENCE_SOURCE_PIXELS = 4096 * 4096
_MAX_RENDER_COUNT = 1_000_000
_SAMPLE_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_SAMPLE_NAME_PATTERN = re.compile(r"^[0-9a-f]{16}\.png$")
_CANONICAL_RENDER_MODES = frozenset(
    {
        "composition",
        "prim_with_stage",
        "prim_only",
        "linear_depth",
        "depth",
        "instance_id_segmentation",
        "custom",
    }
)
_CANONICAL_RENDER_BACKENDS = frozenset({"warp", "ovrtx", "remote", "mock", "unknown"})
_BLANK_REASONS = frozenset(
    {
        "transparent",
        "solid_color",
        "too_few_unique_colors",
        "dominant_color",
        "near_uniform_luminance",
        "analysis_error",
        "remote_blank_render",
        "unknown",
    }
)
_STAT_FIELDS = (
    "width",
    "height",
    "sampled_pixels",
    "unique_colors",
    "dominant_color_ratio",
    "luma_std",
    "luma_dynamic_range",
    "rgb_dynamic_range",
    "strong_minority_pixel_ratio",
    "alpha_visible_ratio",
)


@dataclass(frozen=True)
class PipelineFailureDiagnostic:
    """Trusted marker for one known pipeline failure.

    The object itself is carried only across in-process workflow boundaries.
    Public callers receive :meth:`to_dict`, which contains no raw paths,
    exception text, prompts, request values, or provider payloads.
    """

    renderer_backend: str
    checked_count: int
    blank_count: int
    threshold: float
    render_modes: tuple[str, ...]
    samples: tuple[dict[str, Any], ...]
    evidence_report: str | None = None
    evidence_samples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject hand-built markers that do not satisfy the safe contract."""
        if (
            type(self.renderer_backend) is not str
            or self.renderer_backend not in _CANONICAL_RENDER_BACKENDS
        ):
            raise ValueError("Unsupported renderer diagnostic backend")
        if not _valid_count(self.checked_count, minimum=1):
            raise ValueError("Invalid checked render count")
        if (
            not _valid_count(self.blank_count, minimum=1)
            or self.blank_count > self.checked_count
        ):
            raise ValueError("Invalid blank render count")
        if (
            type(self.threshold) not in {int, float}
            or not math.isfinite(float(self.threshold))
            or not 0.0 <= float(self.threshold) <= 1.0
        ):
            raise ValueError("Invalid blank render threshold")
        if (
            type(self.render_modes) is not tuple
            or not 1 <= len(self.render_modes) <= 6
            or any(
                type(mode) is not str or mode not in _CANONICAL_RENDER_MODES
                for mode in self.render_modes
            )
        ):
            raise ValueError("Invalid canonical render modes")
        if (
            type(self.samples) is not tuple
            or len(self.samples) > _MAX_DIAGNOSTIC_SAMPLES
            or any(_normalize_public_sample(sample) is None for sample in self.samples)
        ):
            raise ValueError("Invalid renderer diagnostic samples")
        if type(self.evidence_samples) is not tuple:
            raise ValueError("Invalid renderer evidence sample collection")
        if self.evidence_report is None:
            if self.evidence_samples:
                raise ValueError("Evidence samples require a report")
            return
        evidence = _normalize_evidence_names(
            {
                "report": self.evidence_report,
                "samples": list(self.evidence_samples),
            },
            self.samples,
        )
        if evidence is None:
            raise ValueError("Invalid renderer evidence names")

    @property
    def code(self) -> str:
        return BLANK_DATASET_RENDERS_CODE

    @property
    def failed_step(self) -> str:
        return BLANK_DATASET_RENDERS_STEP

    def to_dict(self) -> dict[str, Any]:
        """Return the bounded JSON-safe public diagnostic."""
        public_samples = [
            normalized
            for sample in self.samples
            if (normalized := _normalize_public_sample(sample)) is not None
        ]
        result: dict[str, Any] = {
            "schema": DIAGNOSTIC_SCHEMA,
            "code": self.code,
            "phase": FailurePhase.RENDERING.value,
            "retryable": False,
            "failed_step": self.failed_step,
            "renderer_backend": self.renderer_backend,
            "checked_count": self.checked_count,
            "blank_count": self.blank_count,
            "blank_ratio": round(self.blank_count / self.checked_count, 6),
            "threshold": self.threshold,
            "render_modes": list(self.render_modes),
            "samples": public_samples,
        }
        if self.evidence_report is not None:
            result["evidence"] = {
                "report": self.evidence_report,
                "samples": list(self.evidence_samples),
            }
        return result


def canonical_renderer_backend(rendering_backend: Any) -> str:
    """Map a runtime renderer object to a fixed public backend identifier."""
    backend_by_type = {
        "WarpRenderingBackend": "warp",
        "OvRTXRenderingBackend": "ovrtx",
        "RemoteRenderingBackend": "remote",
        "MockRenderingBackend": "mock",
    }
    for backend_type in type(rendering_backend).__mro__:
        canonical = backend_by_type.get(backend_type.__name__)
        if canonical is not None:
            return canonical
    return "unknown"


def canonical_render_modes(
    render_modes: Sequence[Any],
    base_mode_map: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return fixed mode categories without exposing caller-defined aliases."""
    canonical: list[str] = []
    for render_mode in render_modes:
        base_mode = (
            base_mode_map.get(render_mode)
            if type(render_mode) is str and base_mode_map is not None
            else None
        )
        mode = (
            render_mode
            if type(render_mode) is str and render_mode in _CANONICAL_RENDER_MODES
            else (
                base_mode
                if type(base_mode) is str and base_mode in _CANONICAL_RENDER_MODES
                else "custom"
            )
        )
        if mode not in canonical:
            canonical.append(mode)
        if len(canonical) == 6:
            break
    return tuple(canonical or ["custom"])


def create_blank_render_failure_diagnostic(
    *,
    blank_renders: Sequence[Mapping[str, Any]],
    output_dir: Path,
    rendering_backend: Any,
    checked_count: int,
    blank_count: int,
    threshold: float,
    render_modes: Sequence[Any],
    render_mode_base_map: Mapping[str, Any] | None = None,
) -> PipelineFailureDiagnostic:
    """Build a trusted marker and retain a small renderer-evidence bundle."""
    checked_count = _bounded_count(checked_count, minimum=1)
    blank_count = _bounded_count(blank_count, minimum=1)
    blank_count = min(blank_count, checked_count)
    threshold = _bounded_ratio(threshold)
    backend = canonical_renderer_backend(rendering_backend)
    modes = canonical_render_modes(render_modes, render_mode_base_map)

    evidence_dir = output_dir / "failure_evidence"
    samples: list[dict[str, Any]] = []
    evidence_samples: list[str] = []
    used_ids: set[str] = set()
    try:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        if not evidence_dir.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError("Failure evidence directory escaped the render output")
        for candidate in blank_renders:
            if len(samples) >= _MAX_DIAGNOSTIC_SAMPLES:
                break
            sample_id = _sample_identifier(candidate)
            if sample_id in used_ids:
                continue
            used_ids.add(sample_id)
            sample = _safe_sample(candidate, sample_id, modes)
            evidence_name = f"{sample_id}.png"
            pixel_intensity = _write_bounded_sample(
                candidate,
                output_dir=output_dir,
                destination=evidence_dir / evidence_name,
            )
            if pixel_intensity is not None:
                sample["pixel_intensity"] = pixel_intensity
                evidence_samples.append(evidence_name)
            samples.append(sample)

        diagnostic = PipelineFailureDiagnostic(
            renderer_backend=backend,
            checked_count=checked_count,
            blank_count=blank_count,
            threshold=threshold,
            render_modes=modes,
            samples=tuple(samples),
            evidence_report="report.json",
            evidence_samples=tuple(evidence_samples),
        )
        _write_report(evidence_dir / "report.json", diagnostic)
        return diagnostic
    except Exception:
        # Evidence is diagnostic-only. Never replace the known renderer failure
        # with an image-decoder, filesystem, or report-publication exception.
        return PipelineFailureDiagnostic(
            renderer_backend=backend,
            checked_count=checked_count,
            blank_count=blank_count,
            threshold=threshold,
            render_modes=modes,
            samples=tuple(samples),
        )


def trusted_pipeline_failure_diagnostic(value: Any) -> PipelineFailureDiagnostic | None:
    """Accept only the exact code-owned marker type from a child workflow."""
    return value if type(value) is PipelineFailureDiagnostic else None


def normalize_pipeline_failure_diagnostic(value: Any) -> dict[str, Any] | None:
    """Validate a detached public diagnostic before durable service storage."""
    if type(value) is not dict:
        return None
    if (
        value.get("schema") != DIAGNOSTIC_SCHEMA
        or value.get("code") != BLANK_DATASET_RENDERS_CODE
        or value.get("phase") != FailurePhase.RENDERING.value
        or value.get("retryable") is not False
        or value.get("failed_step") != BLANK_DATASET_RENDERS_STEP
    ):
        return None

    backend = value.get("renderer_backend")
    checked_count = value.get("checked_count")
    blank_count = value.get("blank_count")
    threshold = value.get("threshold")
    if type(backend) is not str or backend not in _CANONICAL_RENDER_BACKENDS:
        return None
    if not _valid_count(checked_count, minimum=1):
        return None
    if not _valid_count(blank_count, minimum=1) or blank_count > checked_count:
        return None
    if not _valid_finite_number(threshold):
        return None
    if not 0.0 <= float(threshold) <= 1.0:
        return None

    raw_modes = value.get("render_modes")
    if not isinstance(raw_modes, list) or not 1 <= len(raw_modes) <= 6:
        return None
    if any(
        type(mode) is not str or mode not in _CANONICAL_RENDER_MODES
        for mode in raw_modes
    ):
        return None

    raw_samples = value.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) > _MAX_DIAGNOSTIC_SAMPLES:
        return None
    samples: list[dict[str, Any]] = []
    for raw_sample in raw_samples:
        sample = _normalize_public_sample(raw_sample)
        if sample is None:
            return None
        samples.append(sample)

    diagnostic = PipelineFailureDiagnostic(
        renderer_backend=backend,
        checked_count=checked_count,
        blank_count=blank_count,
        threshold=round(float(threshold), 6),
        render_modes=tuple(raw_modes),
        samples=tuple(samples),
    )
    raw_evidence = value.get("evidence")
    if raw_evidence is not None:
        evidence = _normalize_evidence_names(raw_evidence, samples)
        if evidence is None:
            return None
        diagnostic = PipelineFailureDiagnostic(
            renderer_backend=diagnostic.renderer_backend,
            checked_count=diagnostic.checked_count,
            blank_count=diagnostic.blank_count,
            threshold=diagnostic.threshold,
            render_modes=diagnostic.render_modes,
            samples=diagnostic.samples,
            evidence_report=evidence[0],
            evidence_samples=evidence[1],
        )
    return diagnostic.to_dict()


def _sample_identifier(candidate: Mapping[str, Any]) -> str:
    identity = []
    for key in ("prim_path", "render_mode", "view", "camera", "frame"):
        value = candidate.get(key)
        identity.append(value if type(value) in {str, int} else None)
    encoded = json.dumps(identity, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _safe_sample(
    candidate: Mapping[str, Any],
    sample_id: str,
    fallback_modes: tuple[str, ...],
) -> dict[str, Any]:
    raw_mode = candidate.get("render_mode")
    mode = (
        raw_mode
        if type(raw_mode) is str and raw_mode in _CANONICAL_RENDER_MODES
        else fallback_modes[0]
    )
    raw_stats = candidate.get("stats")
    stats = raw_stats if isinstance(raw_stats, Mapping) else {}
    raw_reason = stats.get("reason")
    reason = (
        raw_reason
        if type(raw_reason) is str and raw_reason in _BLANK_REASONS
        else "unknown"
    )
    safe_stats: dict[str, int | float | None] = {}
    for field in _STAT_FIELDS:
        raw_value = stats.get(field)
        if raw_value is None:
            if field == "alpha_visible_ratio":
                safe_stats[field] = None
            continue
        if type(raw_value) is int and 0 <= raw_value <= _MAX_RENDER_COUNT:
            safe_stats[field] = raw_value
        elif type(raw_value) is float and math.isfinite(raw_value):
            safe_stats[field] = round(raw_value, 6)
    sample: dict[str, Any] = {
        "id": sample_id,
        "render_mode": mode,
        "reason": reason,
    }
    if safe_stats:
        sample["statistics"] = safe_stats
    return sample


def _write_bounded_sample(
    candidate: Mapping[str, Any],
    *,
    output_dir: Path,
    destination: Path,
) -> dict[str, float] | None:
    raw_path = candidate.get("path")
    if type(raw_path) is str:
        source = Path(raw_path)
    elif isinstance(raw_path, PurePath):
        source = Path(raw_path)
    else:
        return None
    if not source.is_absolute():
        source = output_dir / source
    try:
        if not source.resolve(strict=True).is_relative_to(output_dir.resolve()):
            return None
        with PILImage.open(source) as opened:
            if (
                opened.width <= 0
                or opened.height <= 0
                or opened.width * opened.height > _MAX_EVIDENCE_SOURCE_PIXELS
            ):
                return None
            opened.load()
            sample = opened.convert("RGB")
        sample.thumbnail((_MAX_REPORT_IMAGE_SIZE, _MAX_REPORT_IMAGE_SIZE))
        intensity = ImageStat.Stat(sample.convert("L"))
        extrema = intensity.extrema[0]
        temporary = destination.with_suffix(".tmp")
        sample.save(temporary, format="PNG", optimize=True)
        temporary.replace(destination)
    except (OSError, TypeError, ValueError):
        return None
    return {
        "minimum": round(float(extrema[0]), 6),
        "maximum": round(float(extrema[1]), 6),
        "mean": round(float(intensity.mean[0]), 6),
    }


def _write_report(path: Path, diagnostic: PipelineFailureDiagnostic) -> None:
    report = {
        "schema": RENDER_FAILURE_REPORT_SCHEMA,
        "diagnostic": diagnostic.to_dict(),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _bounded_count(value: Any, *, minimum: int) -> int:
    if type(value) is not int:
        return minimum
    return max(minimum, min(value, _MAX_RENDER_COUNT))


def _bounded_ratio(value: Any) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        return 0.5
    return round(max(0.0, min(float(value), 1.0)), 6)


def _valid_count(value: Any, *, minimum: int) -> TypeGuard[int]:
    return type(value) is int and minimum <= value <= _MAX_RENDER_COUNT


def _valid_finite_number(value: Any) -> TypeGuard[int | float]:
    return type(value) in {int, float} and math.isfinite(float(value))


def _normalize_public_sample(value: Any) -> dict[str, Any] | None:
    if type(value) is not dict:
        return None
    sample_id = value.get("id")
    render_mode = value.get("render_mode")
    reason = value.get("reason")
    if (
        type(sample_id) is not str
        or _SAMPLE_ID_PATTERN.fullmatch(sample_id) is None
        or type(render_mode) is not str
        or render_mode not in _CANONICAL_RENDER_MODES
        or type(reason) is not str
        or reason not in _BLANK_REASONS
    ):
        return None
    sample: dict[str, Any] = {
        "id": sample_id,
        "render_mode": render_mode,
        "reason": reason,
    }
    statistics = value.get("statistics")
    if statistics is not None:
        if type(statistics) is not dict:
            return None
        safe_statistics: dict[str, int | float | None] = {}
        for field in _STAT_FIELDS:
            raw_value = statistics.get(field)
            if raw_value is None:
                if field in statistics:
                    safe_statistics[field] = None
                continue
            if type(raw_value) is int and 0 <= raw_value <= _MAX_RENDER_COUNT:
                safe_statistics[field] = raw_value
            elif type(raw_value) is float and math.isfinite(raw_value):
                safe_statistics[field] = round(raw_value, 6)
            else:
                return None
        sample["statistics"] = safe_statistics
    pixel_intensity = value.get("pixel_intensity")
    if pixel_intensity is not None:
        if type(pixel_intensity) is not dict:
            return None
        safe_intensity: dict[str, float] = {}
        for field in ("minimum", "maximum", "mean"):
            raw_value = pixel_intensity.get(field)
            if not _valid_finite_number(raw_value):
                return None
            numeric = float(raw_value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 255.0:
                return None
            safe_intensity[field] = round(numeric, 6)
        sample["pixel_intensity"] = safe_intensity
    return sample


def _normalize_evidence_names(
    value: Any,
    samples: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]] | None:
    if type(value) is not dict or value.get("report") != "report.json":
        return None
    raw_sample_names = value.get("samples")
    if not isinstance(raw_sample_names, list) or len(raw_sample_names) > len(samples):
        return None
    known_names = {f"{sample['id']}.png" for sample in samples}
    if any(
        type(name) is not str
        or _SAMPLE_NAME_PATTERN.fullmatch(name) is None
        or name not in known_names
        for name in raw_sample_names
    ):
        return None
    if len(set(raw_sample_names)) != len(raw_sample_names):
        return None
    return "report.json", tuple(raw_sample_names)
