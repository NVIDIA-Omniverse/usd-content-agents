# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow-owned deterministic visual-grounding helpers.

Pixel samples and candidate evidence are policy artifacts.  A scene adapter may
optionally provide a low-level pixel-to-prim primitive, but this module never
requires or encodes a scene service protocol.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from content_agent_workflows.common.artifacts import (
    ContainedArtifactRead,
    atomic_write_json,
    read_contained_artifact,
)

_MAX_GROUNDING_IMAGE_BYTES = 64 * 1024 * 1024
_MAX_SEGMENTATION_LEGEND_BYTES = 4 * 1024 * 1024
_MAX_GROUNDING_RECORD_BYTES = 16 * 1024 * 1024


def sample_grounding_pixels(
    image_path: Path,
    *,
    issue_text: str,
    max_points: int = 6,
) -> list[dict[str, Any]]:
    """Return deterministic salient dark/bright/saturated pixel samples."""

    artifact = read_contained_artifact(
        image_path.parent,
        image_path,
        max_bytes=_MAX_GROUNDING_IMAGE_BYTES,
        image=True,
        capture_bytes=True,
    )
    assert artifact.data is not None
    return _sample_grounding_pixels(
        artifact.data, issue_text=issue_text, max_points=max_points
    )


def _sample_grounding_pixels(
    image_bytes: bytes,
    *,
    issue_text: str,
    max_points: int,
) -> list[dict[str, Any]]:
    with Image.open(BytesIO(image_bytes)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if rgb.size == 0 or max_points <= 0:
        return []
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    text = issue_text.lower()
    if any(word in text for word in ("dark", "black", "shadow")):
        score = 1.0 - luminance
        mode = "dark"
    elif any(word in text for word in ("bright", "white", "washed")):
        score = luminance
        mode = "bright"
    else:
        score = rgb.max(axis=2) - rgb.min(axis=2)
        mode = "saturated"
    height, width = score.shape
    chosen: list[dict[str, Any]] = []
    # Non-max suppression keeps points useful for independent low-level picks.
    blocked = np.zeros_like(score, dtype=bool)
    for _ in range(max_points):
        masked = np.where(blocked, -np.inf, score)
        flat = int(np.argmax(masked))
        if not np.isfinite(masked.flat[flat]):
            break
        y, x = np.unravel_index(flat, score.shape)
        chosen.append(
            {
                "x": int(x),
                "y": int(y),
                "score": round(float(score[y, x]), 6),
                "mode": mode,
            }
        )
        radius = max(4, min(width, height) // 12)
        blocked[
            max(0, y - radius) : min(height, y + radius + 1),
            max(0, x - radius) : min(width, x + radius + 1),
        ] = True
    return chosen


def candidate_grounding_evidence(
    candidates: dict[str, Any], *, issue_text: str, limit: int = 8
) -> list[dict[str, Any]]:
    """Rank source candidates as an auditable fallback when no pixel-pick exists."""

    words = {
        word for word in issue_text.lower().replace("-", " ").split() if len(word) > 2
    }
    scored = []
    for candidate in candidates.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        tokens = {str(value).lower() for value in candidate.get("path_tokens", [])}
        semantic = (
            str(candidate.get("semantic_hint") or "").lower().replace("_", " ").split()
        )
        score = len(words & (tokens | set(semantic)))
        if score:
            scored.append((score, str(candidate.get("source_path") or ""), candidate))
    return [
        {"source_path": path, "score": score, "candidate": candidate}
        for score, path, candidate in sorted(
            scored, key=lambda item: (-item[0], item[1])
        )[:limit]
    ]


def decode_segmentation_picks(
    *,
    segmentation_path: Path,
    legend_path: Path,
    sample_points: list[dict[str, Any]],
    label_paths: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Decode visible-pixel prim paths from usd-cli's analytic segmentation AOV."""

    try:
        segmentation = read_contained_artifact(
            segmentation_path.parent,
            segmentation_path,
            max_bytes=_MAX_GROUNDING_IMAGE_BYTES,
            image=True,
            capture_bytes=True,
        )
        legend = read_contained_artifact(
            legend_path.parent,
            legend_path,
            max_bytes=_MAX_SEGMENTATION_LEGEND_BYTES,
            capture_bytes=True,
        )
    except ValueError:
        return []
    return _decode_segmentation_picks(
        segmentation,
        legend,
        sample_points=sample_points,
        label_paths=label_paths,
    )


def _decode_segmentation_picks(
    segmentation: ContainedArtifactRead,
    legend: ContainedArtifactRead,
    *,
    sample_points: list[dict[str, Any]],
    label_paths: dict[str, str] | None,
) -> list[dict[str, Any]]:
    assert segmentation.data is not None
    assert legend.data is not None
    colors = _segmentation_legend(legend.data)
    if not colors:
        return []
    with Image.open(BytesIO(segmentation.data)) as image:
        pixels = image.convert("RGB")
        width, height = pixels.size
        picks = []
        for point in sample_points:
            x, y = int(point["x"]), int(point["y"])
            if not (0 <= x < width and 0 <= y < height):
                continue
            color = tuple(pixels.getpixel((x, y)))
            label = colors.get(color)
            path = label_paths.get(label) if label and label_paths else label
            if path and not path.startswith("/"):
                path = None
            picks.append(
                {
                    "x": x,
                    "y": y,
                    "rgb": list(color),
                    "prim_paths": [path] if path else [],
                }
            )
    return picks


def write_material_grounding_diagnostics(
    *,
    run_dir: Path,
    validation_iteration: int,
    unresolved_issues: list[str],
) -> dict[str, str] | None:
    """Persist deterministic segmentation-backed grounding for unresolved VQA issues."""

    if not unresolved_issues:
        return None
    raw = run_dir / "raw"
    candidates_path = raw / "visible_candidate_prims.json"
    final_records_path = raw / "final_render_records.json"
    try:
        candidates = _read_grounding_json(run_dir, candidates_path)
        records = _read_grounding_json(run_dir, final_records_path)
    except (OSError, ValueError):
        candidates, records = {}, {}
    if not isinstance(candidates, dict):
        candidates = {}
    render_records = (
        records.get("renders", []) if isinstance(records, dict) else records
    )
    if not isinstance(render_records, list):
        render_records = []
    issues = []
    for issue in unresolved_issues[:4]:
        views = []
        for record in render_records[:2]:
            if not isinstance(record, dict):
                continue
            image = Path(str(record.get("image_path") or ""))
            segmentation = Path(str(record.get("segmentation_path") or ""))
            legend = Path(str(record.get("segmentation_legend_path") or ""))
            image_artifact = _read_grounding_artifact(
                run_dir,
                image,
                max_bytes=_MAX_GROUNDING_IMAGE_BYTES,
                image=True,
            )
            segmentation_artifact = _read_grounding_artifact(
                run_dir,
                segmentation,
                max_bytes=_MAX_GROUNDING_IMAGE_BYTES,
                image=True,
            )
            legend_artifact = _read_grounding_artifact(
                run_dir,
                legend,
                max_bytes=_MAX_SEGMENTATION_LEGEND_BYTES,
            )
            samples = (
                _sample_grounding_pixels(
                    image_artifact.data,
                    issue_text=issue,
                    max_points=6,
                )
                if image_artifact is not None and image_artifact.data is not None
                else []
            )
            response = record.get("segmentation_response")
            response_data = response.get("data") if isinstance(response, dict) else None
            raw_label_paths = (
                response_data.get("segmentation_legend_paths")
                if isinstance(response_data, dict)
                else None
            )
            label_paths = (
                {
                    str(label): str(path)
                    for label, path in raw_label_paths.items()
                    if isinstance(label, str)
                    and isinstance(path, str)
                    and path.startswith("/")
                }
                if isinstance(raw_label_paths, dict)
                else None
            )
            picks = (
                _decode_segmentation_picks(
                    segmentation_artifact,
                    legend_artifact,
                    sample_points=samples,
                    label_paths=label_paths,
                )
                if segmentation_artifact is not None and legend_artifact is not None
                else []
            )
            picked = sorted(
                {path for pick in picks for path in pick["prim_paths"] if path}
            )
            views.append(
                {
                    "render_name": record.get("name"),
                    "image_path": (
                        str(image_artifact.path) if image_artifact is not None else None
                    ),
                    "sample_points": samples,
                    "pick_results": picks,
                    "picked_inspection_paths": picked,
                    "picked_source_paths": picked,
                    "segmentation_path": (
                        str(segmentation_artifact.path)
                        if segmentation_artifact is not None
                        else None
                    ),
                    "segmentation_legend_path": (
                        str(legend_artifact.path)
                        if legend_artifact is not None
                        else None
                    ),
                    "skip_reason": None if picks else "missing_segmentation_evidence",
                }
            )
        issues.append(
            {
                "issue_text": issue,
                "candidate_evidence": candidate_grounding_evidence(
                    candidates, issue_text=issue
                ),
                "views": views,
            }
        )
    record = {
        "schema_version": "content-agents.material-grounding-diagnostics.v1",
        "validation_iteration": validation_iteration,
        "status": "completed",
        "issues": issues,
        "operation_counts": {
            "pick_calls": sum(
                len(view["pick_results"]) for item in issues for view in item["views"]
            ),
            "render_calls": 0,
            "render_artifact_downloads": 0,
        },
    }
    iteration = (
        raw / f"material_grounding_diagnostics_repair_{validation_iteration}.json"
    )
    aggregate = raw / "material_grounding_diagnostics.json"
    atomic_write_json(iteration, record, within=run_dir)
    try:
        prior = _read_grounding_json(run_dir, aggregate)
    except (OSError, ValueError):
        prior = {}
    runs = (
        prior.get("runs", [])
        if isinstance(prior, dict)
        and prior.get("schema_version") == record["schema_version"]
        else []
    )
    if not isinstance(runs, list):
        runs = []
    retained_runs = [
        run
        for run in runs
        if not (
            isinstance(run, dict)
            and run.get("validation_iteration") == validation_iteration
        )
    ]
    retained_runs.append(record)
    atomic_write_json(
        aggregate,
        {
            "schema_version": record["schema_version"],
            "runs": retained_runs,
            "latest": record,
        },
        within=run_dir,
    )
    return {"iteration": str(iteration), "aggregate": str(aggregate)}


def _read_grounding_json(run_dir: Path, path: Path) -> Any:
    artifact = read_contained_artifact(
        run_dir,
        path,
        max_bytes=_MAX_GROUNDING_RECORD_BYTES,
        capture_bytes=True,
    )
    assert artifact.data is not None
    try:
        return json.loads(artifact.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Grounding record is not valid UTF-8 JSON: {path}") from exc


def _read_grounding_artifact(
    run_dir: Path,
    path: Path,
    *,
    max_bytes: int,
    image: bool = False,
) -> ContainedArtifactRead | None:
    if not str(path):
        return None
    try:
        return read_contained_artifact(
            run_dir,
            path,
            max_bytes=max_bytes,
            image=image,
            capture_bytes=True,
        )
    except ValueError:
        return None


def _segmentation_legend(data: bytes) -> dict[tuple[int, int, int], str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {}
    result: dict[tuple[int, int, int], str] = {}
    for line in text.splitlines():
        label, separator, color = line.partition("\t")
        if not label or not separator:
            continue
        values = color.removeprefix("rgb(").removesuffix(")").split(",")
        try:
            result[tuple(int(value) for value in values)] = label  # type: ignore[assignment]
        except ValueError:
            continue
    return result
