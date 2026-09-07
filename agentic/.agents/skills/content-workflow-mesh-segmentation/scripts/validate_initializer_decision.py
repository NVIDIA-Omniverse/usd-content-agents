#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a recorded per-part initializer route decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mesh_geometry import schema_matches, sha256_file, write_json

SCHEMA_VERSION = "mesh-segmentation-initializer-decision.v1"
VALIDATION_SCHEMA_VERSION = "mesh-segmentation-initializer-decision-validation.v1"
DECISIONS = {"image_mask_seed", "direct_agentic_selection"}
PROBE_STATUSES = {"completed", "skipped_image_generation_unavailable"}
WARNING_SCHEMA_VERSION = "mesh-segmentation-image-generation-warning.v1"
NEXT_STEPS = {
    "image_mask_seed": "eight_view_registered_mask_seed",
    "direct_agentic_selection": "direct_signed_fragment_selection",
}
ASSESSMENT_VALUES = {
    "semantic_quality": {"good", "mixed", "poor", "unavailable"},
    "mask_scale": {"sufficient", "marginal", "too_small", "unavailable"},
    "eroded_interior_support": {"sufficient", "marginal", "empty", "unavailable"},
    "cross_view_consistency": {"consistent", "mixed", "inconsistent", "unavailable"},
    "mesh_projection_quality": {"clean", "mixed", "leaky", "unavailable"},
    "confuser_leakage": {"none", "minor", "major", "unavailable"},
}
EVIDENCE_LIST_FIELDS = {
    "registration_manifests": 1,
    "registered_masks": 3,
    "chosen_pixel_overlays": 3,
    "fragment_projections": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _resolve_artifact(run_dir: Path, decision_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Evidence paths must be nonempty strings")
    raw = Path(value)
    path = raw if raw.is_absolute() else decision_dir / raw
    if path.is_symlink():
        raise ValueError(f"Evidence artifact must not be a symlink: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"Evidence path escapes the run directory: {value}") from exc
    if not resolved.is_file():
        raise ValueError(f"Evidence artifact is missing: {resolved}")
    return resolved


def _require_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    decision_path = args.decision.resolve()
    output_path = args.output.resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    try:
        decision_path.relative_to(run_dir)
        output_path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("Decision and output paths must stay under --run-dir") from exc
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")

    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Initializer decision must be a JSON object")
    if not schema_matches(payload.get("schema_version"), SCHEMA_VERSION):
        raise ValueError("Unsupported initializer decision schema")
    if payload.get("status") != "accepted":
        raise ValueError("Initializer decision status must be 'accepted'")
    semantic_part = _require_string(payload, "semantic_part")
    segment_id = payload.get("segment_id")
    if (
        not isinstance(segment_id, int)
        or isinstance(segment_id, bool)
        or segment_id <= 0
    ):
        raise ValueError("segment_id must be a positive integer")
    decision = payload.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
    if payload.get("next_step") != NEXT_STEPS[decision]:
        raise ValueError(f"next_step must be {NEXT_STEPS[decision]!r} for {decision!r}")

    probe_status = payload.get("probe_status", "completed")
    if probe_status not in PROBE_STATUSES:
        raise ValueError(f"probe_status must be one of {sorted(PROBE_STATUSES)}")
    view_ids = payload.get("probe_view_ids")
    if probe_status == "completed":
        if (
            not isinstance(view_ids, list)
            or len(view_ids) != 3
            or any(not isinstance(value, str) or not value for value in view_ids)
            or len(set(view_ids)) != 3
        ):
            raise ValueError("probe_view_ids must contain exactly three unique strings")
    elif view_ids != []:
        raise ValueError(
            "probe_view_ids must be empty when image generation is unavailable"
        )

    assessment = payload.get("assessment")
    if not isinstance(assessment, dict):
        raise ValueError("assessment must be a JSON object")
    for field, allowed in ASSESSMENT_VALUES.items():
        if assessment.get(field) not in allowed:
            raise ValueError(f"assessment.{field} must be one of {sorted(allowed)}")
    _require_string(assessment, "rationale")

    if probe_status == "skipped_image_generation_unavailable":
        if decision != "direct_agentic_selection":
            raise ValueError(
                "An unavailable image-generation probe requires "
                "direct_agentic_selection"
            )
        non_unavailable = [
            field
            for field in ASSESSMENT_VALUES
            if assessment.get(field) != "unavailable"
        ]
        if non_unavailable:
            raise ValueError(
                "Skipped probe assessment fields must be 'unavailable': "
                + ", ".join(non_unavailable)
            )

    if decision == "image_mask_seed":
        disqualifying = {
            "semantic_quality": "poor",
            "mask_scale": "too_small",
            "eroded_interior_support": "empty",
            "cross_view_consistency": "inconsistent",
            "mesh_projection_quality": "leaky",
            "confuser_leakage": "major",
        }
        failures = [
            f"{field}={value}"
            for field, value in disqualifying.items()
            if assessment.get(field) == value
        ]
        if failures:
            raise ValueError(
                "image_mask_seed is incompatible with the recorded assessment: "
                + ", ".join(failures)
            )

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be a JSON object")
    artifacts: list[dict[str, str]] = []
    if probe_status == "completed":
        for field, minimum_count in sorted(EVIDENCE_LIST_FIELDS.items()):
            values = evidence.get(field)
            if not isinstance(values, list) or len(values) < minimum_count:
                raise ValueError(
                    f"evidence.{field} must contain at least {minimum_count} paths"
                )
            for value in values:
                path = _resolve_artifact(run_dir, decision_path.parent, value)
                artifacts.append(
                    {
                        "role": field,
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                )
        projection_manifest = _resolve_artifact(
            run_dir,
            decision_path.parent,
            evidence.get("projection_manifest"),
        )
        artifacts.append(
            {
                "role": "projection_manifest",
                "path": str(projection_manifest),
                "sha256": sha256_file(projection_manifest),
            }
        )
    else:
        warning_path = _resolve_artifact(
            run_dir,
            decision_path.parent,
            evidence.get("image_generation_warning"),
        )
        warning = json.loads(warning_path.read_text(encoding="utf-8"))
        if not isinstance(warning, dict):
            raise ValueError("Image-generation warning must be a JSON object")
        expected_warning = {
            "schema_version": WARNING_SCHEMA_VERSION,
            "severity": "warning",
            "code": "image_generation_unavailable",
            "semantic_part": semantic_part,
            "fallback": "direct_agentic_selection",
        }
        for field, expected in expected_warning.items():
            if warning.get(field) != expected:
                raise ValueError(
                    f"Image-generation warning {field} must be {expected!r}"
                )
        _require_string(warning, "reason")
        artifacts.append(
            {
                "role": "image_generation_warning",
                "path": str(warning_path),
                "sha256": sha256_file(warning_path),
            }
        )

    result = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "passed",
        "semantic_part": semantic_part,
        "segment_id": segment_id,
        "decision": decision,
        "next_step": NEXT_STEPS[decision],
        "probe_status": probe_status,
        "probe_view_ids": view_ids,
        "decision_path": str(decision_path),
        "decision_sha256": sha256_file(decision_path),
        "artifacts": artifacts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
