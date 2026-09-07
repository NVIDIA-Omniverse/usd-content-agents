# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quality checks for physics mass predictions against geometric scale."""

from __future__ import annotations

import math
import re
import sys
from typing import Any

LARGE_COMPONENT_MAX_DIMENSION_M = 5.0
LARGE_COMPONENT_VOLUME_M3 = 5.0
HIGH_MASS_KG = 500.0
MAX_IMPLIED_FILL_FACTOR = 1.5
MASS_SCALE_SUSPICIOUS_CODE = "mass_scale_suspicious"
VALID_MASS_SCALE_POLICIES = frozenset({"warn", "skip_mass", "fail"})

# Keep these prompt fallbacks aligned with geometric context emitted by
# PrepareDatasetTask.
_DIMENSIONS_RE = re.compile(
    r"Dimensions \(meters\):\s*"
    r"width=(?P<width>[0-9.+\-eE]+)m,\s*"
    r"height=(?P<height>[0-9.+\-eE]+)m,\s*"
    r"depth=(?P<depth>[0-9.+\-eE]+)m"
)
_BBOX_VOLUME_RE = re.compile(
    r"Bounding box volume:\s*(?P<volume>[0-9.+\-eE]+)\s*m(?:\^3|3|\u00b3)"
)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _bbox_size_from_mapping(data: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(data, dict):
        return None
    size = data.get("size")
    if not isinstance(size, list | tuple) or len(size) != 3:
        return None

    parsed = [_as_float(value) for value in size]
    if any(value is None for value in parsed):
        return None
    size_m = [float(value) for value in parsed if value is not None]
    if any(value < 0 for value in size_m):
        return None
    return size_m


def _bbox_size_from_prompt(text: str) -> list[float] | None:
    match = _DIMENSIONS_RE.search(text)
    if not match:
        return None
    values = [
        _as_float(match.group("width")),
        _as_float(match.group("height")),
        _as_float(match.group("depth")),
    ]
    if any(value is None for value in values):
        return None
    size_m = [float(value) for value in values if value is not None]
    if any(value < 0 for value in size_m):
        return None
    return size_m


def _bbox_volume_from_prompt(text: str) -> float | None:
    match = _BBOX_VOLUME_RE.search(text)
    if not match:
        return None
    volume = _as_float(match.group("volume"))
    if volume is None or volume < 0:
        return None
    return volume


def extract_bbox_metrics_meters(
    dataset_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract bbox size/volume in meters from a prepared dataset entry."""

    if not isinstance(dataset_entry, dict):
        return {}

    metadata = dataset_entry.get("metadata")
    bbox_meters = (
        metadata.get("world_bbox_meters") if isinstance(metadata, dict) else None
    )
    size_m = _bbox_size_from_mapping(bbox_meters)

    prompt_text = (
        dataset_entry.get("user_prompt")
        or dataset_entry.get("text")
        or dataset_entry.get("prompt")
        or ""
    )
    if size_m is None and isinstance(prompt_text, str):
        size_m = _bbox_size_from_prompt(prompt_text)

    volume_m3: float | None = None
    if isinstance(bbox_meters, dict):
        volume_m3 = _as_float(bbox_meters.get("volume"))
    if volume_m3 is None and size_m:
        volume_m3 = size_m[0] * size_m[1] * size_m[2]
    if volume_m3 is None and isinstance(prompt_text, str):
        volume_m3 = _bbox_volume_from_prompt(prompt_text)

    metrics: dict[str, Any] = {}
    if size_m:
        metrics["size_m"] = size_m
        metrics["max_dimension_m"] = max(size_m)
    if volume_m3 is not None:
        metrics["volume_m3"] = volume_m3
    return metrics


def get_physical_properties(
    prediction: dict[str, Any],
    output_key: str = "classification",
) -> dict[str, Any]:
    """Return the physical_properties dict from a prediction record."""

    classification = prediction.get(output_key)
    if not isinstance(classification, dict):
        return {}
    props = classification.get("physical_properties")
    return props if isinstance(props, dict) else {}


def build_mass_scale_quality_warnings(
    prediction: dict[str, Any],
    dataset_entry: dict[str, Any] | None,
    output_key: str = "classification",
) -> list[dict[str, Any]]:
    """Detect mass predictions inconsistent with geometric scale or density.

    These checks intentionally warn instead of modifying mass. The existing
    large-component check catches likely unit/scale problems, while the implied
    fill-factor check catches impossible mass/density/volume combinations at any
    scale. Users should verify the prediction and source USD units before treating
    the authored MassAPI value as simulation-ready.
    """

    bbox = extract_bbox_metrics_meters(dataset_entry)
    props = get_physical_properties(prediction, output_key)
    mass_kg = _as_float(props.get("estimated_mass_kg"))
    density = _as_float(props.get("density"))

    warnings: list[dict[str, Any]] = []
    max_dimension_m = bbox.get("max_dimension_m")
    volume_m3 = bbox.get("volume_m3")

    if max_dimension_m is None and volume_m3 is None:
        return warnings

    details: dict[str, Any] = {}
    if max_dimension_m is not None:
        details["max_dimension_m"] = max_dimension_m
    if volume_m3 is not None:
        details["bbox_volume_m3"] = volume_m3
    if mass_kg is not None:
        details["estimated_mass_kg"] = mass_kg
    if density is not None:
        details["density_kg_m3"] = density

    implied_fill_factor: float | None = None
    if (
        mass_kg is not None
        and mass_kg >= 0
        and density is not None
        and density > 0
        and volume_m3 is not None
        and volume_m3 > 0
    ):
        capacity_kg = density * volume_m3
        if capacity_kg == 0:
            implied_fill_factor = 0.0 if mass_kg == 0 else sys.float_info.max
        else:
            implied_fill_factor = mass_kg / capacity_kg
            if not math.isfinite(implied_fill_factor):
                implied_fill_factor = sys.float_info.max
        details["implied_fill_factor"] = implied_fill_factor
    details["thresholds"] = {
        "large_component_max_dimension_m": LARGE_COMPONENT_MAX_DIMENSION_M,
        "large_component_volume_m3": LARGE_COMPONENT_VOLUME_M3,
        "high_mass_kg": HIGH_MASS_KG,
        "max_implied_fill_factor": MAX_IMPLIED_FILL_FACTOR,
    }

    scale_is_large = (
        max_dimension_m is not None
        and max_dimension_m >= LARGE_COMPONENT_MAX_DIMENSION_M
    ) or (volume_m3 is not None and volume_m3 >= LARGE_COMPONENT_VOLUME_M3)
    mass_is_high = mass_kg is not None and mass_kg >= HIGH_MASS_KG
    # Sparse, hollow, or rotated geometry can occupy an arbitrarily small share
    # of its axis-aligned bbox, so only the physically impossible upper side is
    # guarded here.
    fill_factor_is_too_high = (
        implied_fill_factor is not None
        and implied_fill_factor > MAX_IMPLIED_FILL_FACTOR
    )

    if scale_is_large and mass_is_high:
        warnings.append(
            {
                "code": MASS_SCALE_SUSPICIOUS_CODE,
                "severity": "warning",
                "message": (
                    "Predicted mass is very high and the component bounding box is "
                    "many meters across. Verify source USD units/scale before using "
                    "this mass in simulation."
                ),
                "details": details,
            }
        )
    elif fill_factor_is_too_high:
        warnings.append(
            {
                "code": MASS_SCALE_SUSPICIOUS_CODE,
                "severity": "warning",
                "message": (
                    "Predicted mass requires more material than fits inside the "
                    "component bounding box at the predicted density. Verify the "
                    "mass, density, and source USD units/scale before using this "
                    "mass in simulation."
                ),
                "details": details,
            }
        )
    elif scale_is_large:
        warnings.append(
            {
                "code": "large_component_scale",
                "severity": "info",
                "message": (
                    "Component bounding box is unusually large. Verify source USD "
                    "units/scale if this is expected to be a small or medium asset."
                ),
                "details": details,
            }
        )

    return warnings


def has_mass_scale_suspicious_warning(prediction: dict[str, Any]) -> bool:
    """Return whether a prediction carries a high-severity mass/scale warning."""

    warnings = prediction.get("quality_warnings", [])
    if not isinstance(warnings, list):
        return False
    return any(
        isinstance(warning, dict)
        and warning.get("code") == MASS_SCALE_SUSPICIOUS_CODE
        and warning.get("severity", "warning") == "warning"
        for warning in warnings
    )


def merge_quality_warnings(
    existing: list[dict[str, Any]] | None,
    generated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge generated warnings with existing prediction warnings by code."""

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for warning in [*(existing or []), *generated]:
        if not isinstance(warning, dict):
            continue
        code = str(warning.get("code", ""))
        if code in seen:
            continue
        seen.add(code)
        merged.append(warning)
    return merged
