# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ground post-apply material defects to exact Workbench prims."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw

from content_workflow_cli.trace import append_jsonl, utc_now

from .material_run_packet import _download_to_file, _post_json

GROUNDING_SCHEMA_VERSION = "content-agents.material-grounding-diagnostics.v1"

_PREFERRED_RENDER_NAMES = (
    "final_oblique",
    "final_front_py",
    "final_side_px",
    "final_top",
)

_VIEW_KEYWORDS = {
    "top": "final_top",
    "bottom": "final_bottom",
    "front": "final_front_py",
    "side": "final_side_px",
    "oblique": "final_oblique",
}

_DARK_TERMS = (
    "black",
    "dark",
    "too dark",
    "shadow",
)
_BRIGHT_TERMS = (
    "white",
    "light",
    "too bright",
    "washed out",
)
_SATURATED_TERMS = (
    "blue",
    "cyan",
    "red",
    "green",
    "yellow",
    "amber",
    "screen",
    "display",
    "lens",
)


@dataclass(frozen=True)
class MaterialGroundingConfig:
    workbench_url: str
    run_dir: Path
    session_id: str
    validation_iteration: int = 0
    max_issues: int = 4
    max_views_per_issue: int = 2
    max_picks_per_view: int = 6
    render_width: int = 768
    render_height: int = 576


def run_material_grounding_diagnostics(
    config: MaterialGroundingConfig,
) -> dict[str, str] | None:
    """Use Workbench picking/selection outlines to ground unresolved VQA issues."""

    run_dir = config.run_dir
    raw_dir = run_dir / "raw"
    grounding_dir = run_dir / "grounding_renders"
    raw_dir.mkdir(parents=True, exist_ok=True)
    grounding_dir.mkdir(parents=True, exist_ok=True)

    visual_quality = _load_json(run_dir / "visual_quality_assessment.json", default={})
    issues = _grounding_issues(visual_quality)[: config.max_issues]
    if not issues:
        return None

    final_render_records = _load_json(raw_dir / "final_render_records.json", default=[])
    if not isinstance(final_render_records, list):
        final_render_records = []

    assignments = _load_json(run_dir / "assignments.json", default={})
    if not isinstance(assignments, dict):
        assignments = {}
    visible_candidates = _load_json(
        raw_dir / "visible_candidate_prims.json", default={}
    )
    if not isinstance(visible_candidates, dict):
        visible_candidates = {}
    authoring_context = _load_json(
        raw_dir / "material_authoring_context.json", default={}
    )
    if not isinstance(authoring_context, dict):
        authoring_context = {}

    assignment_by_path = _assignment_by_source_path(assignments)
    candidate_by_path = _candidate_by_source_path(visible_candidates)
    authoring_group_by_path = _authoring_group_by_source_path(authoring_context)
    size_rank_by_path = _candidate_size_ranks(candidate_by_path)

    iteration_record: dict[str, Any] = {
        "schema_version": GROUNDING_SCHEMA_VERSION,
        "validation_iteration": config.validation_iteration,
        "status": "completed",
        "issues": [],
        "operation_counts": {
            "pick_calls": 0,
            "render_calls": 0,
            "render_artifact_downloads": 0,
            "workbench_api_calls": 0,
        },
    }

    for issue_index, issue_text in enumerate(issues, start=1):
        issue_record = _ground_issue(
            config=config,
            issue_index=issue_index,
            issue_text=issue_text,
            final_render_records=final_render_records,
            grounding_dir=grounding_dir,
            assignment_by_path=assignment_by_path,
            candidate_by_path=candidate_by_path,
            authoring_group_by_path=authoring_group_by_path,
            size_rank_by_path=size_rank_by_path,
        )
        _merge_counts(
            iteration_record["operation_counts"], issue_record["operation_counts"]
        )
        del issue_record["operation_counts"]
        iteration_record["issues"].append(issue_record)

    if not any(issue.get("views") for issue in iteration_record["issues"]):
        iteration_record["status"] = "skipped_no_groundable_views"

    iteration_path = (
        raw_dir
        / f"material_grounding_diagnostics_repair_{config.validation_iteration}.json"
    )
    iteration_path.write_text(
        json.dumps(iteration_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    aggregate_path = raw_dir / "material_grounding_diagnostics.json"
    aggregate = _load_json(aggregate_path, default={})
    if (
        not isinstance(aggregate, dict)
        or aggregate.get("schema_version") != GROUNDING_SCHEMA_VERSION
    ):
        aggregate = {"schema_version": GROUNDING_SCHEMA_VERSION, "runs": []}
    runs = aggregate.setdefault("runs", [])
    if not isinstance(runs, list):
        runs = []
        aggregate["runs"] = runs
    runs = [
        run
        for run in runs
        if not (
            isinstance(run, dict)
            and run.get("validation_iteration") == config.validation_iteration
        )
    ]
    runs.append(iteration_record)
    aggregate["runs"] = runs
    aggregate["latest"] = iteration_record
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _update_operation_counts(run_dir, iteration_record["operation_counts"])
    _append_grounding_trace(run_dir, iteration_path, aggregate_path, iteration_record)
    return {
        "iteration": str(iteration_path),
        "aggregate": str(aggregate_path),
    }


def sample_grounding_pixels(
    image_path: Path,
    *,
    issue_text: str,
    max_points: int = 6,
) -> list[dict[str, Any]]:
    """Return deterministic pixel samples for a visual defect region."""

    mode = _issue_sample_mode(issue_text)
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    height, width = arr.shape[:2]
    background = _estimate_background(arr)
    distance = np.linalg.norm(arr - background, axis=2)
    luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    object_mask = distance > 18.0
    if int(object_mask.sum()) < 64:
        object_mask = luminance < 245.0

    if mode == "dark":
        target_mask = object_mask & (luminance <= 90.0)
    elif mode == "bright":
        target_mask = object_mask & (luminance >= 160.0)
    elif mode == "saturated":
        max_channel = arr.max(axis=2)
        min_channel = arr.min(axis=2)
        saturation = (max_channel - min_channel) / np.maximum(max_channel, 1.0)
        target_mask = object_mask & (saturation >= 0.22)
    else:
        target_mask = object_mask

    if int(target_mask.sum()) < 16:
        target_mask = object_mask
    if int(target_mask.sum()) < 16:
        return []

    return _grid_sample_mask(
        target_mask, mode=mode, max_points=max_points, width=width, height=height
    )


def _ground_issue(
    *,
    config: MaterialGroundingConfig,
    issue_index: int,
    issue_text: str,
    final_render_records: list[Any],
    grounding_dir: Path,
    assignment_by_path: dict[str, dict[str, Any]],
    candidate_by_path: dict[str, dict[str, Any]],
    authoring_group_by_path: dict[str, dict[str, Any]],
    size_rank_by_path: dict[str, str],
) -> dict[str, Any]:
    selected_views = _select_render_records(
        issue_text,
        final_render_records,
        max_views=config.max_views_per_issue,
    )
    issue_record: dict[str, Any] = {
        "issue_text": issue_text,
        "sample_mode": _issue_sample_mode(issue_text),
        "views": [],
        "grounded_source_paths": [],
        "operation_counts": {
            "pick_calls": 0,
            "render_calls": 0,
            "render_artifact_downloads": 0,
            "workbench_api_calls": 0,
        },
    }
    grounded_source_paths: list[str] = []
    for view_index, record in enumerate(selected_views, start=1):
        try:
            view_record = _ground_view(
                config=config,
                issue_index=issue_index,
                view_index=view_index,
                issue_text=issue_text,
                render_record=record,
                grounding_dir=grounding_dir,
                assignment_by_path=assignment_by_path,
                candidate_by_path=candidate_by_path,
                authoring_group_by_path=authoring_group_by_path,
                size_rank_by_path=size_rank_by_path,
            )
        except Exception as exc:
            view_record = _failed_grounding_view_record(record, exc)
        _merge_counts(issue_record["operation_counts"], view_record["operation_counts"])
        del view_record["operation_counts"]
        issue_record["views"].append(view_record)
        grounded_source_paths.extend(
            _string_list(view_record.get("picked_source_paths"))
        )
    issue_record["grounded_source_paths"] = _dedupe_strings(grounded_source_paths)
    return issue_record


def _failed_grounding_view_record(render_record: Any, exc: Exception) -> dict[str, Any]:
    record = render_record if isinstance(render_record, dict) else {}
    return {
        "render_name": str(record.get("name") or "unknown"),
        "image_path": str(record.get("image_path") or "") or None,
        "camera_json_path": str(record.get("camera_json_path") or "") or None,
        "sample_points": [],
        "pick_results": [],
        "picked_inspection_paths": [],
        "picked_source_paths": [],
        "source_path_evidence": [],
        "outline_render_path": None,
        "skip_reason": "grounding_view_failed",
        "error": str(exc),
        "operation_counts": {
            "pick_calls": 0,
            "render_calls": 0,
            "render_artifact_downloads": 0,
            "workbench_api_calls": 0,
        },
    }


def _ground_view(
    *,
    config: MaterialGroundingConfig,
    issue_index: int,
    view_index: int,
    issue_text: str,
    render_record: dict[str, Any],
    grounding_dir: Path,
    assignment_by_path: dict[str, dict[str, Any]],
    candidate_by_path: dict[str, dict[str, Any]],
    authoring_group_by_path: dict[str, dict[str, Any]],
    size_rank_by_path: dict[str, str],
) -> dict[str, Any]:
    raw_dir = config.run_dir / "raw"
    view_name = str(render_record.get("name") or f"view_{view_index}")
    image_path = Path(str(render_record.get("image_path") or ""))
    camera_path = Path(str(render_record.get("camera_json_path") or ""))
    view_record: dict[str, Any] = {
        "render_name": view_name,
        "image_path": str(image_path) if image_path else None,
        "camera_json_path": str(camera_path) if camera_path else None,
        "sample_points": [],
        "pick_results": [],
        "picked_inspection_paths": [],
        "picked_source_paths": [],
        "source_path_evidence": [],
        "outline_render_path": None,
        "operation_counts": {
            "pick_calls": 0,
            "render_calls": 0,
            "render_artifact_downloads": 0,
            "workbench_api_calls": 0,
        },
    }
    if not image_path.is_file() or not camera_path.is_file():
        view_record["skip_reason"] = "missing_image_or_camera_artifact"
        return view_record

    camera_payload = _load_json(camera_path, default={})
    camera_state = (
        camera_payload.get("camera_state") if isinstance(camera_payload, dict) else None
    )
    if not isinstance(camera_state, dict):
        view_record["skip_reason"] = "missing_camera_state"
        return view_record

    width, height = _image_dimensions(
        image_path,
        fallback=(
            _positive_int(camera_payload.get("image_width"), config.render_width),
            _positive_int(camera_payload.get("image_height"), config.render_height),
        ),
    )
    sample_points = sample_grounding_pixels(
        image_path,
        issue_text=issue_text,
        max_points=config.max_picks_per_view,
    )
    overlay_path = grounding_dir / (
        f"grounding_repair_{config.validation_iteration}_issue_{issue_index}_"
        f"{view_name}_samples.png"
    )
    if sample_points:
        _write_sample_overlay(image_path, overlay_path, sample_points)
        view_record["sample_overlay_path"] = str(overlay_path)

    encoded_session_id = quote(config.session_id, safe="")
    workbench_url = config.workbench_url.rstrip("/")
    set_camera_body = {"command": "set_camera", "payload": {"camera": camera_state}}
    set_camera_response = _post_json(
        f"{workbench_url}/sessions/{encoded_session_id}/commands",
        set_camera_body,
    )
    set_camera_path = raw_dir / (
        f"grounding_repair_{config.validation_iteration}_issue_{issue_index}_"
        f"{view_name}_set_camera_response.json"
    )
    set_camera_path.write_text(
        json.dumps(set_camera_response, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    view_record["operation_counts"]["workbench_api_calls"] += 1
    view_record["sample_points"] = sample_points

    picked_inspection_paths: list[str] = []
    for sample_index, point in enumerate(sample_points, start=1):
        pick_body = {
            "x": int(point["x"]),
            "y": int(point["y"]),
            "width": width,
            "height": height,
            "update_selection": False,
            "ovrtx_render_mode": "rt2",
            "ovrtx_num_sensor_updates": 1,
        }
        pick_response = _post_json(
            f"{workbench_url}/sessions/{encoded_session_id}/pick",
            pick_body,
        )
        pick_path = raw_dir / (
            f"grounding_pick_repair_{config.validation_iteration}_issue_{issue_index}_"
            f"{view_name}_{sample_index}.json"
        )
        pick_path.write_text(
            json.dumps(
                {"request": pick_body, "response": pick_response},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        prim_paths = _string_list(pick_response.get("prim_paths"))
        picked_inspection_paths.extend(prim_paths)
        view_record["pick_results"].append(
            {
                "x": pick_body["x"],
                "y": pick_body["y"],
                "prim_paths": prim_paths,
                "response_path": str(pick_path),
            }
        )
        view_record["operation_counts"]["pick_calls"] += 1
        view_record["operation_counts"]["workbench_api_calls"] += 1

    picked_inspection_paths = _dedupe_strings(picked_inspection_paths)
    view_record["picked_inspection_paths"] = picked_inspection_paths
    if not picked_inspection_paths:
        return view_record

    translation_response = _translate_picked_paths(
        workbench_url=workbench_url,
        session_id=encoded_session_id,
        inspection_paths=picked_inspection_paths,
    )
    translate_path = raw_dir / (
        f"grounding_translate_repair_{config.validation_iteration}_issue_{issue_index}_"
        f"{view_name}.json"
    )
    translate_path.write_text(
        json.dumps(translation_response, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    view_record["translation_response_path"] = str(translate_path)
    view_record["operation_counts"]["workbench_api_calls"] += 1
    picked_source_paths = _source_paths_from_translation(translation_response)
    view_record["picked_source_paths"] = picked_source_paths
    view_record["source_path_evidence"] = [
        _source_path_evidence(
            source_path=source_path,
            assignment_by_path=assignment_by_path,
            candidate_by_path=candidate_by_path,
            authoring_group_by_path=authoring_group_by_path,
            size_rank_by_path=size_rank_by_path,
        )
        for source_path in picked_source_paths
    ]

    binding_response = _post_json(
        f"{workbench_url}/sessions/{encoded_session_id}/material-binding:batch",
        {"prim_paths": picked_inspection_paths},
    )
    binding_path = raw_dir / (
        f"grounding_material_binding_repair_{config.validation_iteration}_"
        f"issue_{issue_index}_{view_name}.json"
    )
    binding_path.write_text(
        json.dumps(binding_response, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    view_record["material_binding_response_path"] = str(binding_path)
    view_record["operation_counts"]["workbench_api_calls"] += 1

    outline = _render_outline(
        workbench_url=workbench_url,
        encoded_session_id=encoded_session_id,
        output_dir=grounding_dir,
        name=(
            f"grounding_repair_{config.validation_iteration}_issue_{issue_index}_"
            f"{view_name}_outline"
        ),
        width=width,
        height=height,
        inspection_paths=picked_inspection_paths,
    )
    if outline:
        view_record["outline_render_path"] = outline.get("image_path")
        view_record["outline_response_path"] = outline.get("response_path")
        view_record["operation_counts"]["render_calls"] += 1
        view_record["operation_counts"]["render_artifact_downloads"] += int(
            outline.get("artifact_download_count") or 0
        )
        view_record["operation_counts"]["workbench_api_calls"] += 3 + int(
            outline.get("artifact_download_count") or 0
        )

    return view_record


def _render_outline(
    *,
    workbench_url: str,
    encoded_session_id: str,
    output_dir: Path,
    name: str,
    width: int,
    height: int,
    inspection_paths: list[str],
) -> dict[str, Any] | None:
    select_body = {"command": "select", "payload": {"paths": inspection_paths}}
    _post_json(f"{workbench_url}/sessions/{encoded_session_id}/commands", select_body)
    try:
        render_body = {
            "width": width,
            "height": height,
            "use_session_camera": True,
            "render_quality": "inspection",
            "ovrtx_render_mode": "rt2",
            "ovrtx_num_sensor_updates": 32,
            "save_camera_json": True,
        }
        response = _post_json(
            f"{workbench_url}/sessions/{encoded_session_id}/render",
            render_body,
        )
        image_url = response.get("image_url")
        if not isinstance(image_url, str) or not image_url:
            return None
        image_path = output_dir / f"{name}.png"
        response_path = output_dir / f"{name}_response.json"
        camera_path = output_dir / f"{name}_camera.json"
        artifact_download_count = 1
        _download_to_file(f"{workbench_url}{image_url}", image_path)
        if response.get("camera_json_url"):
            _download_to_file(
                f"{workbench_url}{response['camera_json_url']}", camera_path
            )
            artifact_download_count += 1
        response_path.write_text(
            json.dumps(response, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "image_path": str(image_path),
            "response_path": str(response_path),
            "camera_json_path": str(camera_path) if camera_path.exists() else None,
            "artifact_download_count": artifact_download_count,
        }
    finally:
        _post_json(
            f"{workbench_url}/sessions/{encoded_session_id}/commands",
            {"command": "select", "payload": {"paths": []}},
        )


def _translate_picked_paths(
    *, workbench_url: str, session_id: str, inspection_paths: list[str]
) -> dict[str, Any]:
    return _post_json(
        f"{workbench_url}/sessions/{session_id}/paths/translate:batch",
        {
            "requests": [
                {
                    "prim_path": path,
                    "source_space": "inspection",
                    "target_space": "source",
                }
                for path in inspection_paths
            ]
        },
    )


def _grounding_issues(visual_quality: Any) -> list[str]:
    if not isinstance(visual_quality, dict):
        return []
    issues = _string_list(visual_quality.get("unresolved_issues"))
    if not issues and visual_quality.get("status") == "unresolved_issues":
        issues = _string_list(visual_quality.get("issues_found"))
    return _dedupe_strings(issues)


def _select_render_records(
    issue_text: str,
    final_render_records: list[Any],
    *,
    max_views: int,
) -> list[dict[str, Any]]:
    records = [record for record in final_render_records if isinstance(record, dict)]
    by_name = {
        str(record.get("name")): record for record in records if record.get("name")
    }
    issue_lower = issue_text.lower()
    selected: list[dict[str, Any]] = []
    for keyword, view_name in _VIEW_KEYWORDS.items():
        if keyword in issue_lower and view_name in by_name:
            selected.append(by_name[view_name])
    for view_name in _PREFERRED_RENDER_NAMES:
        if len(selected) >= max_views:
            break
        record = by_name.get(view_name)
        if record is not None and record not in selected:
            selected.append(record)
    return selected[:max_views]


def _issue_sample_mode(issue_text: str) -> str:
    lower = issue_text.lower()
    if any(term in lower for term in _DARK_TERMS):
        return "dark"
    if any(term in lower for term in _SATURATED_TERMS):
        return "saturated"
    if any(term in lower for term in _BRIGHT_TERMS):
        return "bright"
    return "object"


def _estimate_background(arr: np.ndarray) -> np.ndarray:
    edge_pixels = np.concatenate(
        [
            arr[0, :, :],
            arr[-1, :, :],
            arr[:, 0, :],
            arr[:, -1, :],
        ],
        axis=0,
    )
    return np.median(edge_pixels, axis=0)


def _grid_sample_mask(
    mask: np.ndarray,
    *,
    mode: str,
    max_points: int,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    cell_size = max(24, min(width, height) // 12)
    cells: list[tuple[int, int, int, int, int]] = []
    for y0 in range(0, height, cell_size):
        y1 = min(height, y0 + cell_size)
        for x0 in range(0, width, cell_size):
            x1 = min(width, x0 + cell_size)
            count = int(mask[y0:y1, x0:x1].sum())
            if count >= max(8, int((x1 - x0) * (y1 - y0) * 0.04)):
                cells.append((count, x0, y0, x1, y1))
    cells.sort(reverse=True)
    points: list[dict[str, Any]] = []
    min_distance = max(18.0, min(width, height) * 0.06)
    for count, x0, y0, x1, y1 in cells:
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        if xs.size == 0:
            continue
        x = int(round(float(xs.mean()) + x0))
        y = int(round(float(ys.mean()) + y0))
        if any(math.hypot(x - p["x"], y - p["y"]) < min_distance for p in points):
            continue
        points.append({"x": x, "y": y, "mode": mode, "cell_score": count})
        if len(points) >= max_points:
            break
    return points


def _write_sample_overlay(
    image_path: Path,
    output_path: Path,
    sample_points: list[dict[str, Any]],
) -> None:
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, point in enumerate(sample_points, start=1):
        x = int(point["x"])
        y = int(point["y"])
        radius = 7
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(255, 184, 0),
            width=3,
        )
        draw.text((x + radius + 2, y - radius - 2), str(index), fill=(255, 184, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _source_paths_from_translation(response: dict[str, Any]) -> list[str]:
    source_paths: list[str] = []
    results = response.get("results")
    if not isinstance(results, list):
        return []
    for result in results:
        if isinstance(result, dict):
            source_paths.extend(_string_list(result.get("source_paths")))
    return _dedupe_strings(source_paths)


def _source_path_evidence(
    *,
    source_path: str,
    assignment_by_path: dict[str, dict[str, Any]],
    candidate_by_path: dict[str, dict[str, Any]],
    authoring_group_by_path: dict[str, dict[str, Any]],
    size_rank_by_path: dict[str, str],
) -> dict[str, Any]:
    assignment = assignment_by_path.get(source_path, {})
    candidate = candidate_by_path.get(source_path, {})
    authoring_group = authoring_group_by_path.get(source_path, {})
    return {
        "source_path": source_path,
        "current_assignment": _compact_assignment(assignment),
        "candidate": _compact_candidate(candidate, size_rank_by_path.get(source_path)),
        "authoring_group": _compact_authoring_group(authoring_group),
        "diagnosis_hints": _diagnosis_hints(
            source_path=source_path,
            assignment=assignment,
            candidate=candidate,
            authoring_group=authoring_group,
            size_rank=size_rank_by_path.get(source_path),
        ),
    }


def _compact_assignment(group: dict[str, Any]) -> dict[str, Any] | None:
    if not group:
        return None
    return {
        "family": group.get("family"),
        "coverage_status": group.get("coverage_status"),
        "material_name": group.get("material_name"),
        "material_path": group.get("material_path"),
        "rationale": group.get("rationale"),
    }


def _compact_candidate(
    candidate: dict[str, Any], size_rank: str | None
) -> dict[str, Any] | None:
    if not candidate:
        return None
    bounds = candidate.get("bounds_samples")
    return {
        "type_name": candidate.get("type_name"),
        "shape_hint": candidate.get("shape_hint"),
        "translation_ambiguous": candidate.get("translation_ambiguous"),
        "inspection_paths": _string_list(candidate.get("inspection_paths")),
        "bounds_samples": bounds if isinstance(bounds, list) else [],
        "relative_size": size_rank,
    }


def _compact_authoring_group(group: dict[str, Any]) -> dict[str, Any] | None:
    if not group:
        return None
    return {
        "authoring_family": group.get("authoring_family"),
        "size_hints": group.get("size_hints"),
        "shape_hints": group.get("shape_hints"),
        "semantic_hints": group.get("semantic_hints"),
        "recommended_coverage_status": group.get("recommended_coverage_status"),
    }


def _diagnosis_hints(
    *,
    source_path: str,
    assignment: dict[str, Any],
    candidate: dict[str, Any],
    authoring_group: dict[str, Any],
    size_rank: str | None,
) -> list[str]:
    hints = []
    material_name = str(assignment.get("material_name") or "")
    family = str(assignment.get("family") or "")
    if material_name:
        hints.append(
            f"Picked visible pixel resolves to {source_path}, currently assigned {material_name!r} in family {family!r}."
        )
    if size_rank in {"large", "very_large"}:
        hints.append(
            "Picked prim is relatively large in the visible candidate set; do not treat it as a small fastener without outline evidence."
        )
    size_hints = authoring_group.get("size_hints")
    if isinstance(size_hints, dict) and size_hints:
        hints.append(f"Authoring context size hints: {size_hints}.")
    shape_hints = authoring_group.get("shape_hints")
    if isinstance(shape_hints, dict) and shape_hints:
        hints.append(f"Authoring context shape hints: {shape_hints}.")
    if not hints:
        hints.append(f"Picked visible pixel resolves to {source_path}.")
    return hints


def _assignment_by_source_path(
    assignments: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    groups = assignments.get("assignments")
    if not isinstance(groups, list):
        return result
    for group in groups:
        if not isinstance(group, dict):
            continue
        paths = _string_list(group.get("source_prim_paths")) or _string_list(
            group.get("prim_paths")
        )
        for path in paths:
            result[path] = group
    return result


def _candidate_by_source_path(candidates: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for candidate in candidates.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        source_paths = _string_list(candidate.get("source_paths"))
        source_path = candidate.get("source_path")
        if isinstance(source_path, str) and source_path:
            source_paths.append(source_path)
        for path in _dedupe_strings(source_paths):
            result[path] = candidate
    return result


def _authoring_group_by_source_path(
    context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    groups = context.get("candidate_groups")
    if not isinstance(groups, list):
        return result
    for group in groups:
        if not isinstance(group, dict):
            continue
        for path in _string_list(group.get("source_paths")):
            result[path] = group
    return result


def _candidate_size_ranks(
    candidate_by_path: dict[str, dict[str, Any]],
) -> dict[str, str]:
    diagonals: list[tuple[str, float]] = []
    for path, candidate in candidate_by_path.items():
        diagonal = _candidate_diagonal(candidate)
        if diagonal is not None:
            diagonals.append((path, diagonal))
    if not diagonals:
        return {}
    values = np.asarray([diagonal for _path, diagonal in diagonals], dtype=np.float32)
    q25 = float(np.percentile(values, 25))
    q75 = float(np.percentile(values, 75))
    q90 = float(np.percentile(values, 90))
    ranks: dict[str, str] = {}
    for path, diagonal in diagonals:
        if diagonal >= q90:
            ranks[path] = "very_large"
        elif diagonal >= q75:
            ranks[path] = "large"
        elif diagonal <= q25:
            ranks[path] = "small"
        else:
            ranks[path] = "medium"
    return ranks


def _candidate_diagonal(candidate: dict[str, Any]) -> float | None:
    samples = candidate.get("bounds_samples")
    if not isinstance(samples, list):
        return None
    diagonals = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        size = sample.get("size")
        if not isinstance(size, list) or len(size) != 3:
            continue
        try:
            diagonals.append(
                math.sqrt(sum(float(component) ** 2 for component in size))
            )
        except (TypeError, ValueError):
            continue
    return max(diagonals) if diagonals else None


def _update_operation_counts(run_dir: Path, counts: dict[str, Any]) -> None:
    counts_path = run_dir / "api_operation_counts.json"
    existing = _load_json(counts_path, default={})
    if not isinstance(existing, dict):
        existing = {}
    pick_calls = int(counts.get("pick_calls") or 0)
    render_calls = int(counts.get("render_calls") or 0)
    render_downloads = int(counts.get("render_artifact_downloads") or 0)
    api_calls = int(counts.get("workbench_api_calls") or 0)
    existing["pick_calls"] = int(existing.get("pick_calls") or 0) + pick_calls
    existing["render_count_total"] = (
        int(existing.get("render_count_total") or 0) + render_calls
    )
    existing["render_artifact_downloads"] = (
        int(existing.get("render_artifact_downloads") or 0) + render_downloads
    )
    existing["api_operation_count_total"] = (
        int(existing.get("api_operation_count_total") or 0) + api_calls
    )
    existing["grounding_diagnostic_runs"] = (
        int(existing.get("grounding_diagnostic_runs") or 0) + 1
    )
    existing["grounding_pick_calls"] = (
        int(existing.get("grounding_pick_calls") or 0) + pick_calls
    )
    existing["grounding_render_calls"] = (
        int(existing.get("grounding_render_calls") or 0) + render_calls
    )
    counts_path.write_text(
        json.dumps(existing, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def grounding_operation_counts(run_dir: Path) -> dict[str, int]:
    aggregate = _load_json(
        run_dir / "raw" / "material_grounding_diagnostics.json",
        default={},
    )
    if not isinstance(aggregate, dict):
        return {
            "pick_calls": 0,
            "render_calls": 0,
            "render_artifact_downloads": 0,
            "workbench_api_calls": 0,
            "runs": 0,
        }
    counts = {
        "pick_calls": 0,
        "render_calls": 0,
        "render_artifact_downloads": 0,
        "workbench_api_calls": 0,
        "runs": 0,
    }
    runs = aggregate.get("runs")
    if not isinstance(runs, list):
        return counts
    for run in runs:
        if not isinstance(run, dict):
            continue
        operation_counts = run.get("operation_counts")
        if not isinstance(operation_counts, dict):
            continue
        counts["runs"] += 1
        counts["pick_calls"] += int(operation_counts.get("pick_calls") or 0)
        counts["render_calls"] += int(operation_counts.get("render_calls") or 0)
        counts["render_artifact_downloads"] += int(
            operation_counts.get("render_artifact_downloads") or 0
        )
        counts["workbench_api_calls"] += int(
            operation_counts.get("workbench_api_calls") or 0
        )
    return counts


def _append_grounding_trace(
    run_dir: Path,
    iteration_path: Path,
    aggregate_path: Path,
    iteration_record: dict[str, Any],
) -> None:
    operation_counts = iteration_record.get("operation_counts")
    if not isinstance(operation_counts, dict):
        operation_counts = {}
    append_jsonl(
        run_dir / "trace" / "events.jsonl",
        {
            "schema_version": "content-agents.trace.v1",
            "time": utc_now(),
            "event_type": "api",
            "phase": "grounding",
            "summary": (
                "Grounded unresolved material VQA issues to Workbench picked prims "
                "and selection-outline evidence."
            ),
            "artifacts": [str(iteration_path), str(aggregate_path)],
            "data": {
                "validation_iteration": iteration_record.get("validation_iteration"),
                "status": iteration_record.get("status"),
                "issue_count": len(iteration_record.get("issues", [])),
                "operation_counts": operation_counts,
                "api_calls": [
                    "POST /sessions/{session_id}/commands set_camera",
                    "POST /sessions/{session_id}/pick",
                    "POST /sessions/{session_id}/paths/translate:batch",
                    "POST /sessions/{session_id}/material-binding:batch",
                    "POST /sessions/{session_id}/commands select",
                    "POST /sessions/{session_id}/render",
                ],
            },
        },
    )


def _merge_counts(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "pick_calls",
        "render_calls",
        "render_artifact_downloads",
        "workbench_api_calls",
    ):
        target[key] = int(target.get(key) or 0) + int(source.get(key) or 0)


def _load_json(path: Path, *, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _image_dimensions(
    image_path: Path, *, fallback: tuple[int, int]
) -> tuple[int, int]:
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return fallback


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _dedupe_strings(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
