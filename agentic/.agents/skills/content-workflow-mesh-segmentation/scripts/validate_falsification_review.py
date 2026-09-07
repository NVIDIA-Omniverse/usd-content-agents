#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a signed-evidence falsification review against its candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from mesh_geometry import load_usd, schema_matches, sha256_file, write_json
from PIL import Image

REVIEW_SCHEMA_VERSION = "mesh-segmentation-falsification-review.v1"
VALIDATION_SCHEMA_VERSION = "mesh-segmentation-falsification-validation.v1"
EVIDENCE_SCHEMA_VERSION = "mesh-segmentation-fragment-evidence.v1"
COMPARISON_SCHEMA_VERSION = "mesh-segmentation-fragment-label-comparison.v1"
PLAN_SCHEMA_VERSION = "mesh-segmentation-falsification-plan.v1"
EXPORT_SCHEMA_VERSION = "mesh-segmentation-fragment-export.v1"
SELECTED_ONLY_SCHEMA_VERSION = "mesh-segmentation-selected-only-stage.v1"
REVISION_CONSISTENCY_SCHEMA_VERSION = "mesh-segmentation-revision-consistency-review.v1"
CONSISTENCY_SHEET_SCHEMA_VERSION = "mesh-segmentation-consistency-sheet.v1"
REQUIRED_POSITIVE_ROLES = frozenset(
    {
        "target_interior",
        "target_extent",
        "opposing_or_occluded_surface",
    }
)
MAX_BOUNDARY_PAIR_DISTANCE_FRACTION = 0.04
REQUIRED_COMPLETENESS_SURFACE_ROLES = frozenset(
    {
        "primary_surface",
        "opposing_or_occluded_surface",
    }
)
ALLOWED_BOUNDARY_CLASSIFICATIONS = frozenset(
    {
        "intended_semantic_boundary",
        "silhouette_or_occlusion",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _require_string_list(
    payload: dict[str, Any],
    field: str,
    *,
    minimum: int = 1,
) -> list[str]:
    values = payload.get(field)
    if (
        not isinstance(values, list)
        or len(values) < minimum
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise ValueError(f"{field} must contain at least {minimum} nonempty strings")
    return [value.strip() for value in values]


def _resolve_artifact(run_dir: Path, review_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Artifact paths must be nonempty strings")
    raw = Path(value)
    path = raw if raw.is_absolute() else review_dir / raw
    if path.is_symlink():
        raise ValueError(f"Artifact must not be a symlink: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes the run directory: {value}") from exc
    if not resolved.is_file():
        raise ValueError(f"Artifact is missing: {resolved}")
    return resolved


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _require_named_objects(
    payload: dict[str, Any],
    field: str,
    *,
    name_field: str,
) -> list[str]:
    values = payload.get(field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be a nonempty object array")
    names: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"{field} item {index} must be an object")
        names.append(_require_string(value, name_field))
    return names


def _accepted_signed_events(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    raw_events = evidence.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("signed evidence must contain an events array")
    accepted: list[dict[str, Any]] = []
    for index, event in enumerate(raw_events):
        if not isinstance(event, dict):
            raise ValueError(f"signed evidence event {index} is not an object")
        if event.get("polarity") not in {"positive", "negative"}:
            raise ValueError(f"signed evidence event {index} has invalid polarity")
        face_id = event.get("face_id")
        if not isinstance(face_id, int) or isinstance(face_id, bool) or face_id < 0:
            continue
        if event.get("probe_passed") is False or event.get("rejection_reason"):
            continue
        accepted.append(event)
    return accepted


def _event_string(event: dict[str, Any], field: str, index: int) -> str:
    value = event.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"accepted signed evidence event {index} lacks {field}")
    return value.strip()


def _camera_size(
    event: dict[str, Any],
    *,
    event_index: int,
    run_dir: Path,
) -> tuple[int, int]:
    raw_camera = _event_string(event, "camera", event_index)
    raw_path = Path(raw_camera)
    camera_path = raw_path if raw_path.is_absolute() else run_dir / raw_path
    camera_path = camera_path.resolve()
    try:
        camera_path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(
            f"accepted signed evidence event {event_index} camera escapes run directory"
        ) from exc
    camera = _load_object(camera_path, "signed evidence camera")
    width = camera.get("image_width")
    height = camera.get("image_height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height <= 0
    ):
        raise ValueError(
            f"accepted signed evidence event {event_index} camera lacks image size"
        )
    return width, height


def _validate_challenge_coverage(
    *,
    accepted_events: list[dict[str, Any]],
    expected_instances: list[str],
    confusers: list[str],
    held_out_view_ids: list[str],
    run_dir: Path,
) -> dict[str, Any]:
    positive_by_instance: dict[str, list[dict[str, Any]]] = {
        instance_id: [] for instance_id in expected_instances
    }
    negative_confusers: set[str] = set()
    polarities_by_view: dict[str, set[str]] = {}
    boundary_pairs: dict[str, list[tuple[int, dict[str, Any]]]] = {}

    for index, event in enumerate(accepted_events):
        view_id = _event_string(event, "view_id", index)
        instance_id = _event_string(event, "instance_id", index)
        polarity = str(event["polarity"])
        polarities_by_view.setdefault(view_id, set()).add(polarity)
        pair_id = event.get("boundary_pair_id")
        if pair_id is not None:
            if not isinstance(pair_id, str) or not pair_id.strip():
                raise ValueError(
                    f"accepted signed evidence event {index} has invalid boundary_pair_id"
                )
            boundary_pairs.setdefault(pair_id.strip(), []).append((index, event))

        if polarity == "positive":
            if instance_id not in positive_by_instance:
                raise ValueError(
                    "positive signed evidence references an undeclared expected "
                    f"instance: {instance_id}"
                )
            _event_string(event, "coverage_role", index)
            positive_by_instance[instance_id].append(event)
        else:
            confuser_id = _event_string(event, "confuser_id", index)
            if confuser_id not in confusers:
                raise ValueError(
                    "negative signed evidence references an undeclared confuser: "
                    f"{confuser_id}"
                )
            negative_confusers.add(confuser_id)

    for instance_id, events in positive_by_instance.items():
        roles = {str(event["coverage_role"]).strip() for event in events}
        missing_roles = sorted(REQUIRED_POSITIVE_ROLES - roles)
        if missing_roles:
            raise ValueError(
                f"expected instance {instance_id!r} lacks positive coverage roles: "
                f"{missing_roles}"
            )
        views = {str(event["view_id"]).strip() for event in events}
        if len(views) < 2:
            raise ValueError(
                f"expected instance {instance_id!r} needs positive evidence from "
                "at least two views"
            )

    missing_confusers = sorted(set(confusers) - negative_confusers)
    if missing_confusers:
        raise ValueError(
            f"signed evidence lacks negative checks for confusers: {missing_confusers}"
        )

    for view_id in held_out_view_ids:
        missing_polarities = {"positive", "negative"} - polarities_by_view.get(
            view_id,
            set(),
        )
        if missing_polarities:
            raise ValueError(
                f"held-out view {view_id!r} lacks accepted signed evidence "
                f"polarities: {sorted(missing_polarities)}"
            )

    paired_instances: set[str] = set()
    pair_distances: dict[str, float] = {}
    for pair_id, indexed_events in boundary_pairs.items():
        if len(indexed_events) != 2:
            raise ValueError(
                f"boundary pair {pair_id!r} must contain exactly two accepted events"
            )
        first_index, first = indexed_events[0]
        second_index, second = indexed_events[1]
        if {first["polarity"], second["polarity"]} != {"positive", "negative"}:
            raise ValueError(
                f"boundary pair {pair_id!r} must contain one positive and one negative"
            )
        first_view = _event_string(first, "view_id", first_index)
        second_view = _event_string(second, "view_id", second_index)
        first_instance = _event_string(first, "instance_id", first_index)
        second_instance = _event_string(second, "instance_id", second_index)
        if first_view != second_view or first_instance != second_instance:
            raise ValueError(
                f"boundary pair {pair_id!r} must share one view and expected instance"
            )
        if first_instance not in positive_by_instance:
            raise ValueError(
                f"boundary pair {pair_id!r} references undeclared instance "
                f"{first_instance!r}"
            )
        first_size = _camera_size(first, event_index=first_index, run_dir=run_dir)
        second_size = _camera_size(second, event_index=second_index, run_dir=run_dir)
        if first_size != second_size:
            raise ValueError(f"boundary pair {pair_id!r} camera sizes do not match")
        first_pixel = first.get("pixel")
        second_pixel = second.get("pixel")
        if (
            not isinstance(first_pixel, list)
            or len(first_pixel) != 2
            or not isinstance(second_pixel, list)
            or len(second_pixel) != 2
        ):
            raise ValueError(f"boundary pair {pair_id!r} lacks valid pixels")
        distance = math.dist(
            [float(value) for value in first_pixel],
            [float(value) for value in second_pixel],
        )
        distance_fraction = distance / max(first_size)
        if distance_fraction > MAX_BOUNDARY_PAIR_DISTANCE_FRACTION:
            raise ValueError(
                f"boundary pair {pair_id!r} is too far apart "
                f"({distance_fraction:.6f} > "
                f"{MAX_BOUNDARY_PAIR_DISTANCE_FRACTION:.6f})"
            )
        paired_instances.add(first_instance)
        pair_distances[pair_id] = distance_fraction

    missing_pairs = sorted(set(expected_instances) - paired_instances)
    if missing_pairs:
        raise ValueError(
            f"expected instances lack a close signed boundary pair: {missing_pairs}"
        )
    return {
        "required_positive_roles": sorted(REQUIRED_POSITIVE_ROLES),
        "positive_evidence_count_by_instance": {
            instance_id: len(events)
            for instance_id, events in positive_by_instance.items()
        },
        "held_out_views_with_both_polarities": held_out_view_ids,
        "checked_confusers": sorted(negative_confusers),
        "boundary_pair_count": len(boundary_pairs),
        "boundary_pair_distance_fractions": pair_distances,
    }


def _manifest_render_images(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    run_dir: Path,
    label: str,
) -> dict[str, Path]:
    raw_renders = manifest.get("renders")
    if not isinstance(raw_renders, list) or not raw_renders:
        raise ValueError(f"{label} lacks renders")
    images: dict[str, Path] = {}
    for index, record in enumerate(raw_renders):
        if not isinstance(record, dict):
            raise ValueError(f"{label} render {index} must be an object")
        view_id = _require_string(record, "name")
        if view_id in images:
            raise ValueError(f"{label} has duplicate render name: {view_id}")
        images[view_id] = _resolve_artifact(
            run_dir,
            manifest_path.parent,
            record.get("image"),
        )
    return images


def _validate_revision_consistency(
    *,
    review: dict[str, Any],
    semantic_part: str,
    segment_id: int,
    revision: str,
    candidate_path: Path,
    expected_instances: list[str],
    run_dir: Path,
    review_dir: Path,
    flat_label_manifest_path: Path,
    flat_label_images: dict[str, Path],
    selected_only_manifest_path: Path,
    selected_only_images: dict[str, Path],
) -> tuple[dict[str, Any], list[Path]]:
    consistency_path = _resolve_artifact(
        run_dir,
        review_dir,
        review.get("revision_consistency_review"),
    )
    consistency = _load_object(
        consistency_path,
        "revision consistency review",
    )
    if not schema_matches(
        consistency.get("schema_version"), REVISION_CONSISTENCY_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported revision consistency review schema")
    if consistency.get("status") != "passed":
        raise ValueError("revision consistency review status must be 'passed'")
    if _require_string(consistency, "semantic_part") != semantic_part:
        raise ValueError("revision consistency semantic_part does not match review")
    if consistency.get("segment_id") != segment_id:
        raise ValueError("revision consistency segment_id does not match review")
    if _require_string(consistency, "revision") != revision:
        raise ValueError("revision consistency revision does not match review")

    revision_dir = candidate_path.parent
    try:
        consistency_path.relative_to(revision_dir)
    except ValueError as exc:
        raise ValueError(
            "revision consistency review must be stored under the final revision"
        ) from exc
    consistency_candidate_path = _resolve_artifact(
        run_dir,
        consistency_path.parent,
        consistency.get("candidate_labels"),
    )
    if consistency_candidate_path != candidate_path:
        raise ValueError(
            "revision consistency review does not use the final candidate labels"
        )
    candidate_sha256 = sha256_file(candidate_path)
    if consistency.get("candidate_labels_sha256") != candidate_sha256:
        raise ValueError("revision consistency candidate digest is stale")

    consistency_flat_manifest = _resolve_artifact(
        run_dir,
        consistency_path.parent,
        consistency.get("flat_label_render_manifest"),
    )
    if consistency_flat_manifest != flat_label_manifest_path:
        raise ValueError(
            "revision consistency flat-label manifest is not the final manifest"
        )
    if consistency.get("flat_label_render_manifest_sha256") != sha256_file(
        flat_label_manifest_path
    ):
        raise ValueError("revision consistency flat-label manifest digest is stale")

    consistency_selected_manifest = _resolve_artifact(
        run_dir,
        consistency_path.parent,
        consistency.get("selected_only_render_manifest"),
    )
    if consistency_selected_manifest != selected_only_manifest_path:
        raise ValueError(
            "revision consistency selected-only manifest is not the final manifest"
        )
    if consistency.get("selected_only_render_manifest_sha256") != sha256_file(
        selected_only_manifest_path
    ):
        raise ValueError("revision consistency selected-only manifest digest is stale")

    comparison_sheet = _resolve_artifact(
        run_dir,
        consistency_path.parent,
        consistency.get("comparison_sheet"),
    )
    try:
        comparison_sheet.relative_to(revision_dir)
    except ValueError as exc:
        raise ValueError(
            "revision consistency comparison sheet must be stored under the "
            "final revision"
        ) from exc
    if consistency.get("comparison_sheet_sha256") != sha256_file(comparison_sheet):
        raise ValueError("revision consistency comparison sheet digest is stale")

    sheet_manifest_path = _resolve_artifact(
        run_dir,
        consistency_path.parent,
        consistency.get("consistency_sheet_manifest"),
    )
    sheet_manifest = _load_object(
        sheet_manifest_path,
        "revision consistency sheet manifest",
    )
    if not schema_matches(
        sheet_manifest.get("schema_version"), CONSISTENCY_SHEET_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported revision consistency sheet schema")
    if sheet_manifest.get("status") != "passed":
        raise ValueError("revision consistency sheet manifest did not pass")
    consistency_neutral_manifest = _resolve_artifact(
        run_dir,
        consistency_path.parent,
        consistency.get("neutral_render_manifest"),
    )
    if consistency.get("neutral_render_manifest_sha256") != sha256_file(
        consistency_neutral_manifest
    ):
        raise ValueError("revision consistency neutral manifest digest is stale")
    sheet_neutral_manifest = _resolve_artifact(
        run_dir,
        sheet_manifest_path.parent,
        sheet_manifest.get("neutral_render_manifest"),
    )
    if sheet_neutral_manifest != consistency_neutral_manifest:
        raise ValueError(
            "revision consistency context pages do not use the reviewed "
            "neutral manifest"
        )
    if sheet_manifest.get("neutral_render_manifest_sha256") != sha256_file(
        consistency_neutral_manifest
    ):
        raise ValueError("revision consistency neutral context digest is stale")
    neutral_manifest = _load_object(
        consistency_neutral_manifest,
        "revision consistency neutral manifest",
    )
    neutral_images = _manifest_render_images(
        manifest=neutral_manifest,
        manifest_path=consistency_neutral_manifest,
        run_dir=run_dir,
        label="revision consistency neutral manifest",
    )
    sheet_flat_manifest = _resolve_artifact(
        run_dir,
        sheet_manifest_path.parent,
        sheet_manifest.get("flat_label_render_manifest"),
    )
    if sheet_flat_manifest != flat_label_manifest_path:
        raise ValueError(
            "revision consistency crops do not use the final flat-label manifest"
        )
    if sheet_manifest.get("flat_label_render_manifest_sha256") != sha256_file(
        flat_label_manifest_path
    ):
        raise ValueError("revision consistency crop flat-label digest is stale")
    sheet_selected_manifest = _resolve_artifact(
        run_dir,
        sheet_manifest_path.parent,
        sheet_manifest.get("selected_only_render_manifest"),
    )
    if sheet_selected_manifest != selected_only_manifest_path:
        raise ValueError(
            "revision consistency crops do not use the final selected-only manifest"
        )
    if sheet_manifest.get("selected_only_render_manifest_sha256") != sha256_file(
        selected_only_manifest_path
    ):
        raise ValueError("revision consistency crop selected-only digest is stale")
    sheet_manifest_image = _resolve_artifact(
        run_dir,
        sheet_manifest_path.parent,
        sheet_manifest.get("comparison_sheet"),
    )
    if sheet_manifest_image != comparison_sheet:
        raise ValueError(
            "revision consistency review does not use the standardized crop sheet"
        )
    if sheet_manifest.get("comparison_sheet_sha256") != sha256_file(comparison_sheet):
        raise ValueError("revision consistency crop sheet digest is stale")
    crop_size = sheet_manifest.get("crop_size")
    if (
        not isinstance(crop_size, list)
        or len(crop_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 192
            for value in crop_size
        )
        or crop_size[0] != crop_size[1]
    ):
        raise ValueError(
            "revision consistency crops must use one square size of at least 192 px"
        )
    layout = sheet_manifest.get("layout")
    if not isinstance(layout, dict) or layout.get("mode") != "tiled_peer_grid":
        raise ValueError(
            "revision consistency sheet must use the tiled peer grid layout"
        )
    layout_columns = layout.get("columns")
    layout_rows = layout.get("rows")
    layout_sheet_size = layout.get("sheet_size")
    if (
        not isinstance(layout_columns, int)
        or isinstance(layout_columns, bool)
        or layout_columns < 1
        or not isinstance(layout_rows, int)
        or isinstance(layout_rows, bool)
        or layout_rows < 1
        or not isinstance(layout_sheet_size, list)
        or len(layout_sheet_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in layout_sheet_size
        )
    ):
        raise ValueError("revision consistency sheet has invalid tiled layout")
    with Image.open(comparison_sheet) as image:
        actual_sheet_size = list(image.size)
    if layout_sheet_size != actual_sheet_size:
        raise ValueError(
            "revision consistency tiled layout dimensions do not match its image"
        )

    channels = _require_string_list(
        consistency,
        "comparison_channels",
        minimum=2,
    )
    if channels != ["flat_label", "selected_only"]:
        raise ValueError(
            "revision consistency comparison_channels must be exactly "
            "['flat_label', 'selected_only']"
        )
    if (
        _require_string_list(
            consistency,
            "expected_instances",
        )
        != expected_instances
    ):
        raise ValueError(
            "revision consistency expected_instances do not match the review"
        )
    views_compared = _require_string_list(
        consistency,
        "views_compared",
        minimum=2,
    )
    for view_id in views_compared:
        if (
            view_id not in neutral_images
            or view_id not in flat_label_images
            or view_id not in selected_only_images
        ):
            raise ValueError(
                f"revision consistency view {view_id!r} is absent from matching "
                "neutral, flat-label, or selected-only renders"
            )

    context_channels = _require_string_list(
        consistency,
        "context_comparison_channels",
        minimum=3,
    )
    if context_channels != ["neutral", "flat_label", "selected_only"]:
        raise ValueError(
            "revision consistency context_comparison_channels must be exactly "
            "['neutral', 'flat_label', 'selected_only']"
        )
    if consistency.get("context_outlier_regions") != []:
        raise ValueError(
            "passing revision consistency review has full-view context outliers"
        )
    context_layout = sheet_manifest.get("context_layout")
    if (
        not isinstance(context_layout, dict)
        or context_layout.get("mode") != "full_view_triptych_pages"
        or context_layout.get("channels") != context_channels
        or context_layout.get("view_order") != views_compared
        or context_layout.get("views_per_page") != 2
    ):
        raise ValueError(
            "revision consistency context must use two-view neutral/flat/"
            "selected triptych pages in review order"
        )
    context_image_size = context_layout.get("image_size")
    if (
        not isinstance(context_image_size, list)
        or len(context_image_size) != 2
        or context_image_size[0] != context_image_size[1]
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 256
            for value in context_image_size
        )
    ):
        raise ValueError("revision consistency context images are too small")
    raw_context_pages = sheet_manifest.get("context_pages")
    raw_review_context_pages = consistency.get("context_comparison_sheets")
    if (
        not isinstance(raw_context_pages, list)
        or not raw_context_pages
        or not isinstance(raw_review_context_pages, list)
        or len(raw_review_context_pages) != len(raw_context_pages)
    ):
        raise ValueError(
            "revision consistency review must enumerate every context page"
        )
    context_artifacts: list[Path] = []
    context_views: list[str] = []
    for index, (page, reviewed_page) in enumerate(
        zip(raw_context_pages, raw_review_context_pages, strict=True)
    ):
        if not isinstance(page, dict) or not isinstance(reviewed_page, dict):
            raise ValueError(
                f"revision consistency context page {index} must be an object"
            )
        view_ids = _require_string_list(page, "view_ids")
        reviewed_view_ids = _require_string_list(reviewed_page, "view_ids")
        if view_ids != reviewed_view_ids or len(view_ids) > 2:
            raise ValueError(
                f"revision consistency context page {index} has invalid views"
            )
        context_views.extend(view_ids)
        page_image = _resolve_artifact(
            run_dir,
            sheet_manifest_path.parent,
            page.get("image"),
        )
        reviewed_image = _resolve_artifact(
            run_dir,
            consistency_path.parent,
            reviewed_page.get("image"),
        )
        page_digest = sha256_file(page_image)
        if (
            reviewed_image != page_image
            or page.get("image_sha256") != page_digest
            or reviewed_page.get("image_sha256") != page_digest
        ):
            raise ValueError(
                f"revision consistency context page {index} digest is stale"
            )
        expected_size = [
            context_image_size[0] * 3,
            len(view_ids) * (context_image_size[1] + 24),
        ]
        with Image.open(page_image) as image:
            actual_size = list(image.size)
        if page.get("size") != expected_size or actual_size != expected_size:
            raise ValueError(
                f"revision consistency context page {index} has invalid size"
            )
        context_artifacts.append(page_image)
    if context_views != views_compared:
        raise ValueError(
            "revision consistency context pages do not cover every reviewed view"
        )

    raw_crop_instances = sheet_manifest.get("instances")
    if not isinstance(raw_crop_instances, list) or not raw_crop_instances:
        raise ValueError("revision consistency sheet lacks instance crops")
    crop_instances: list[str] = []
    crop_artifacts: list[Path] = [sheet_manifest_path]
    for index, raw_crop_instance in enumerate(raw_crop_instances):
        if not isinstance(raw_crop_instance, dict):
            raise ValueError(
                f"revision consistency crop instance {index} must be an object"
            )
        instance_id = _require_string(raw_crop_instance, "instance_id")
        crop_instances.append(instance_id)
        raw_crops = raw_crop_instance.get("evidence_crops")
        if not isinstance(raw_crops, list) or len(raw_crops) < 2:
            raise ValueError(
                f"revision consistency instance {instance_id!r} needs at least "
                "two standardized crop pairs"
            )
        seen_crop_views: set[str] = set()
        crop_surface_roles: set[str] = set()
        for crop_index, raw_crop in enumerate(raw_crops):
            if not isinstance(raw_crop, dict):
                raise ValueError(
                    f"revision consistency instance {instance_id!r} crop "
                    f"{crop_index} must be an object"
                )
            view_id = _require_string(raw_crop, "view_id")
            if view_id in seen_crop_views:
                raise ValueError(
                    f"revision consistency instance {instance_id!r} repeats "
                    f"crop view {view_id!r}"
                )
            seen_crop_views.add(view_id)
            if view_id not in views_compared:
                raise ValueError(
                    f"revision consistency crop view {view_id!r} was not compared"
                )
            surface_role = _require_string(raw_crop, "surface_role")
            if surface_role not in {
                "primary_surface",
                "opposing_or_occlusion_revealing",
            }:
                raise ValueError(
                    f"revision consistency instance {instance_id!r} crop has "
                    f"unsupported surface role {surface_role!r}"
                )
            crop_surface_roles.add(surface_role)
            flat_crop = _resolve_artifact(
                run_dir,
                sheet_manifest_path.parent,
                raw_crop.get("flat_label_crop"),
            )
            selected_crop = _resolve_artifact(
                run_dir,
                sheet_manifest_path.parent,
                raw_crop.get("selected_only_crop"),
            )
            if raw_crop.get("flat_label_crop_sha256") != sha256_file(
                flat_crop
            ) or raw_crop.get("selected_only_crop_sha256") != sha256_file(
                selected_crop
            ):
                raise ValueError(
                    f"revision consistency instance {instance_id!r} crop "
                    "digest is stale"
                )
            if raw_crop.get("crop_size") != crop_size:
                raise ValueError(
                    f"revision consistency instance {instance_id!r} crop size "
                    "does not match the sheet"
                )
            with Image.open(flat_crop) as image:
                flat_size = list(image.size)
            with Image.open(selected_crop) as image:
                selected_size = list(image.size)
            if flat_size != crop_size or selected_size != crop_size:
                raise ValueError(
                    f"revision consistency instance {instance_id!r} crops are "
                    "not standardized"
                )
            crop_artifacts.extend([flat_crop, selected_crop])
        if crop_surface_roles != {
            "primary_surface",
            "opposing_or_occlusion_revealing",
        }:
            raise ValueError(
                f"revision consistency instance {instance_id!r} must include "
                "primary and opposing-or-occlusion-revealing crop evidence"
            )
    if crop_instances != expected_instances:
        raise ValueError(
            "revision consistency crop instances must match expected instances "
            "in plan order"
        )
    if layout.get("instance_order") != expected_instances:
        raise ValueError(
            "revision consistency tiled layout must preserve expected instance order"
        )
    if layout.get("channels_per_view") != ["flat_label", "selected_only"]:
        raise ValueError(
            "revision consistency tiled layout must pair flat-label and "
            "selected-only channels"
        )
    if layout.get("surface_role_order") != [
        "primary_surface",
        "opposing_or_occlusion_revealing",
    ]:
        raise ValueError(
            "revision consistency tiled layout must present primary then "
            "opposing surface roles"
        )
    expected_columns = min(4, len(expected_instances))
    expected_rows = (len(expected_instances) + expected_columns - 1) // (
        expected_columns
    )
    if layout_columns != expected_columns or layout_rows != expected_rows:
        raise ValueError(
            "revision consistency tiled layout must use up to four peer "
            "instances per row"
        )

    repeated = len(expected_instances) > 1
    expected_mode = "repeated_instance_outlier" if repeated else "cross_view_outlier"
    if consistency.get("comparison_mode") != expected_mode:
        raise ValueError(
            f"revision consistency comparison_mode must be {expected_mode!r}"
        )
    if consistency.get("outlier_regions") != []:
        raise ValueError("passing revision consistency review has outlier regions")

    raw_comparisons = consistency.get("instance_comparisons")
    if not isinstance(raw_comparisons, list) or not raw_comparisons:
        raise ValueError(
            "revision consistency instance_comparisons must be a nonempty array"
        )
    comparisons_by_instance: dict[str, dict[str, Any]] = {}
    for index, comparison in enumerate(raw_comparisons):
        if not isinstance(comparison, dict):
            raise ValueError(
                f"revision consistency instance comparison {index} must be an object"
            )
        instance_id = _require_string(comparison, "instance_id")
        if instance_id in comparisons_by_instance:
            raise ValueError(f"duplicate revision consistency instance: {instance_id}")
        comparisons_by_instance[instance_id] = comparison
        if comparison.get("status") != "consistent":
            raise ValueError(
                f"revision consistency instance {instance_id!r} is not consistent"
            )
        raw_peers = comparison.get("peer_instance_ids")
        if not isinstance(raw_peers, list) or any(
            not isinstance(value, str) or not value.strip() for value in raw_peers
        ):
            raise ValueError(
                f"revision consistency instance {instance_id!r} has invalid peers"
            )
        peers = [value.strip() for value in raw_peers]
        expected_peers = (
            [value for value in expected_instances if value != instance_id]
            if repeated
            else []
        )
        if peers != expected_peers:
            raise ValueError(
                f"revision consistency instance {instance_id!r} must compare "
                "against every peer instance in plan order"
            )
        for field in (
            "unique_gap_regions",
            "unique_protrusion_regions",
            "unique_boundary_regions",
        ):
            if comparison.get(field) != []:
                raise ValueError(
                    f"revision consistency instance {instance_id!r} has "
                    f"unresolved {field}"
                )
        _require_string(comparison, "assessment")

    if list(comparisons_by_instance) != expected_instances:
        raise ValueError(
            "revision consistency instance_comparisons must match expected "
            "instances in plan order"
        )

    return (
        {
            "comparison_mode": expected_mode,
            "instance_count": len(comparisons_by_instance),
            "views_compared": views_compared,
            "comparison_sheet": str(comparison_sheet),
            "comparison_sheet_sha256": sha256_file(comparison_sheet),
            "context_comparison_sheets": [str(path) for path in context_artifacts],
        },
        [
            consistency_path,
            comparison_sheet,
            consistency_neutral_manifest,
            flat_label_manifest_path,
            selected_only_manifest_path,
            *context_artifacts,
            *crop_artifacts,
        ],
    )


def _recompute_unselected_frontier(
    run_dir: Path,
    candidate_path: Path,
    segment_id: int,
) -> set[int]:
    """Derive the unselected frontier from the frozen fragment map.

    The audit and its ids file are both written by the child, so corroborating
    one against the other proves nothing: truncating the file and setting the
    count to zero satisfies both. The frozen fragment map and the
    digest-verified candidate labels are the only inputs the child cannot
    restate, so the frontier is recomputed from them.
    """

    labels = np.fromfile(candidate_path, dtype="<u4")
    fragment_ids_path = run_dir / "fragments" / "fragment_ids.u32le"
    # Bind the fragment map to its manifest before trusting it, so the map
    # cannot simply be swapped for one that makes the frontier empty.
    manifest = _load_object(
        run_dir / "fragments" / "fragment_manifest.json", "fragment manifest"
    )
    declared = manifest.get("fragment_ids_sha256")
    if not isinstance(declared, str) or not declared:
        raise ValueError("fragment manifest does not record fragment_ids_sha256")
    if sha256_file(fragment_ids_path) != declared:
        raise ValueError("frozen fragment map does not match its manifest digest")
    fragment_ids = np.fromfile(fragment_ids_path, dtype="<u4")
    if labels.size != fragment_ids.size:
        raise ValueError(
            "candidate labels and the frozen fragment map disagree on face count"
        )
    # Derive fragment adjacency from the prepared mesh rather than reading
    # `fragments/fragment_adjacency.npy`, which carries no digest: emptying
    # that file would make the frontier empty and reinstate the waiver this
    # recomputation exists to deny.
    data = load_usd(run_dir / "prepare" / "neutral.usdc")
    if data.face_count != fragment_ids.size:
        raise ValueError(
            "prepared mesh and the frozen fragment map disagree on face count"
        )
    adjacency = np.column_stack(
        (
            fragment_ids[data.face_adjacency[:, 0]],
            fragment_ids[data.face_adjacency[:, 1]],
        )
    )
    adjacency = adjacency[adjacency[:, 0] != adjacency[:, 1]]
    selected_faces = labels == segment_id
    if not np.any(selected_faces):
        raise ValueError("the active segment contains no faces")
    selected_fragments = set(np.unique(fragment_ids[selected_faces]).tolist())
    frontier: set[int] = set()
    for first, second in np.asarray(adjacency).reshape(-1, 2).tolist():
        first_selected = int(first) in selected_fragments
        second_selected = int(second) in selected_fragments
        if first_selected == second_selected:
            continue
        frontier.add(int(second) if first_selected else int(first))
    return frontier


def _validate_instance_completeness(
    *,
    review: dict[str, Any],
    expected_instances: list[str],
    run_dir: Path,
    review_dir: Path,
    flat_label_images: dict[str, Path],
    selected_only_images: dict[str, Path],
    frontier: dict[str, Any],
    frontier_path: Path,
    candidate_path: Path,
    segment_id: int,
) -> tuple[dict[str, Any], list[Path]]:
    raw_reviews = review.get("instance_completeness_reviews")
    if not isinstance(raw_reviews, list) or not raw_reviews:
        raise ValueError("instance_completeness_reviews must be a nonempty array")

    frontier_ids_path = _resolve_artifact(
        run_dir,
        frontier_path.parent,
        frontier.get("unselected_frontier_fragment_ids"),
    )
    unselected_frontier_ids = {
        int(value) for value in np.fromfile(frontier_ids_path, dtype="<u4").tolist()
    }
    # Both the audit and its ids file are written by the child, so checking one
    # against the other proves nothing -- truncating the file and setting the
    # count to zero satisfies both, waiving the whole adjacent-frontier
    # contract for a part that genuinely under-selects. Recompute the frontier
    # from the frozen fragment map and the digest-verified candidate labels,
    # which the child cannot restate, and require the audit to match it.
    recomputed = _recompute_unselected_frontier(run_dir, candidate_path, segment_id)
    if unselected_frontier_ids != recomputed:
        missing = sorted(recomputed - unselected_frontier_ids)[:20]
        extra = sorted(unselected_frontier_ids - recomputed)[:20]
        raise ValueError(
            "frontier audit does not match the frontier recomputed from the "
            f"frozen fragment map (missing {missing}, unexpected {extra})"
        )
    declared_count = frontier.get("unselected_frontier_fragment_count")
    if (
        isinstance(declared_count, int)
        and not isinstance(declared_count, bool)
        and declared_count != len(recomputed)
    ):
        raise ValueError(
            "frontier audit unselected_frontier_fragment_count "
            f"({declared_count}) does not match the recomputed frontier "
            f"({len(recomputed)})"
        )
    # A part built from complete disconnected source components has no adjacent
    # unselected fragment to challenge. Now that the empty frontier is a
    # recomputed geometric fact rather than a claim, the challenge is vacuously
    # satisfied, not skipped.
    frontier_challenge_required = bool(recomputed)

    records_by_instance: dict[str, dict[str, Any]] = {}
    artifacts: list[Path] = [frontier_ids_path]
    checked_frontier_ids: set[int] = set()
    view_counts: dict[str, int] = {}

    for index, instance_review in enumerate(raw_reviews):
        if not isinstance(instance_review, dict):
            raise ValueError(
                f"instance_completeness_reviews item {index} must be an object"
            )
        instance_id = _require_string(instance_review, "instance_id")
        if instance_id in records_by_instance:
            raise ValueError(f"duplicate instance completeness review: {instance_id}")
        records_by_instance[instance_id] = instance_review
        if instance_review.get("status") != "passed":
            raise ValueError(
                f"instance completeness review {instance_id!r} did not pass"
            )

        raw_views = instance_review.get("views")
        if not isinstance(raw_views, list) or len(raw_views) < 2:
            raise ValueError(
                f"instance completeness review {instance_id!r} needs at least "
                "two visible selected-only views"
            )
        seen_views: set[str] = set()
        surface_roles: set[str] = set()
        for view_index, view in enumerate(raw_views):
            if not isinstance(view, dict):
                raise ValueError(
                    f"instance {instance_id!r} view {view_index} must be an object"
                )
            view_id = _require_string(view, "view_id")
            if view_id in seen_views:
                raise ValueError(
                    f"instance {instance_id!r} repeats completeness view {view_id!r}"
                )
            seen_views.add(view_id)
            view_counts[view_id] = view_counts.get(view_id, 0) + 1

            surface_role = _require_string(view, "surface_role")
            if surface_role not in REQUIRED_COMPLETENESS_SURFACE_ROLES:
                raise ValueError(
                    f"instance {instance_id!r} view {view_id!r} has invalid "
                    f"surface_role: {surface_role}"
                )
            surface_roles.add(surface_role)

            if view_id not in selected_only_images:
                raise ValueError(
                    f"instance {instance_id!r} completeness view {view_id!r} "
                    "is absent from selected-only renders"
                )
            if view_id not in flat_label_images:
                raise ValueError(
                    f"instance {instance_id!r} completeness view {view_id!r} "
                    "is absent from flat-label renders"
                )
            selected_image = _resolve_artifact(
                run_dir,
                review_dir,
                view.get("selected_only_render"),
            )
            flat_label_image = _resolve_artifact(
                run_dir,
                review_dir,
                view.get("flat_label_render"),
            )
            if selected_image != selected_only_images[view_id]:
                raise ValueError(
                    f"instance {instance_id!r} view {view_id!r} selected-only "
                    "evidence does not match its manifest"
                )
            if flat_label_image != flat_label_images[view_id]:
                raise ValueError(
                    f"instance {instance_id!r} view {view_id!r} flat-label "
                    "evidence does not match its manifest"
                )
            artifacts.extend([selected_image, flat_label_image])

            for field in (
                "coherent_semantic_shell",
                "outer_contour_continuous",
                "semantic_openings_plausible",
            ):
                if view.get(field) is not True:
                    raise ValueError(
                        f"instance {instance_id!r} view {view_id!r} requires "
                        f"{field}: true"
                    )
            if view.get("unexplained_boundaries") != []:
                raise ValueError(
                    f"instance {instance_id!r} view {view_id!r} has unresolved "
                    "selected-only boundaries"
                )

            raw_boundaries = view.get("boundary_classifications")
            if not isinstance(raw_boundaries, list) or not raw_boundaries:
                raise ValueError(
                    f"instance {instance_id!r} view {view_id!r} must classify "
                    "its visible selected-only boundaries"
                )
            boundary_ids: set[str] = set()
            for boundary_index, boundary in enumerate(raw_boundaries):
                if not isinstance(boundary, dict):
                    raise ValueError(
                        f"instance {instance_id!r} view {view_id!r} boundary "
                        f"{boundary_index} must be an object"
                    )
                boundary_id = _require_string(boundary, "region_id")
                if boundary_id in boundary_ids:
                    raise ValueError(
                        f"instance {instance_id!r} view {view_id!r} repeats "
                        f"boundary region {boundary_id!r}"
                    )
                boundary_ids.add(boundary_id)
                classification = _require_string(boundary, "classification")
                if classification not in ALLOWED_BOUNDARY_CLASSIFICATIONS:
                    raise ValueError(
                        f"instance {instance_id!r} view {view_id!r} boundary "
                        f"{boundary_id!r} is not resolved"
                    )
                _require_string(boundary, "rationale")

            raw_fragment_ids = (
                view.get("adjacent_unselected_frontier_fragment_ids_checked") or []
            )
            if not isinstance(raw_fragment_ids, list) or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in raw_fragment_ids
            ):
                raise ValueError(
                    f"instance {instance_id!r} view {view_id!r} has an invalid "
                    "adjacent unselected frontier fragment list"
                )
            fragment_ids = {int(value) for value in raw_fragment_ids}
            if frontier_challenge_required and not raw_fragment_ids:
                raise ValueError(
                    f"instance {instance_id!r} view {view_id!r} must check at "
                    "least one adjacent unselected frontier fragment"
                )
            invalid_fragment_ids = sorted(fragment_ids - unselected_frontier_ids)
            if invalid_fragment_ids:
                raise ValueError(
                    f"instance {instance_id!r} view {view_id!r} claims "
                    "non-frontier fragment IDs: "
                    f"{invalid_fragment_ids[:20]}"
                )
            expected_decisions = (
                {"confirmed_non_target"}
                if frontier_challenge_required
                else {
                    "no_adjacent_unselected_frontier",
                    "not_applicable_complete_disconnected_component",
                }
            )
            if view.get("frontier_decision") not in expected_decisions:
                raise ValueError(
                    f"instance {instance_id!r} view {view_id!r} must record "
                    "frontier_decision as one of "
                    f"{sorted(expected_decisions)!r}"
                )
            _require_string(view, "assessment")
            checked_frontier_ids.update(fragment_ids)

        missing_roles = sorted(REQUIRED_COMPLETENESS_SURFACE_ROLES - surface_roles)
        if missing_roles:
            raise ValueError(
                f"instance completeness review {instance_id!r} lacks surface "
                f"roles: {missing_roles}"
            )

    if set(records_by_instance) != set(expected_instances):
        missing = sorted(set(expected_instances) - set(records_by_instance))
        unexpected = sorted(set(records_by_instance) - set(expected_instances))
        raise ValueError(
            "instance_completeness_reviews do not match expected instances; "
            f"missing={missing}, unexpected={unexpected}"
        )

    return (
        {
            "required_surface_roles": sorted(REQUIRED_COMPLETENESS_SURFACE_ROLES),
            "instance_count": len(records_by_instance),
            "views_per_instance": {
                instance_id: len(instance_review["views"])
                for instance_id, instance_review in records_by_instance.items()
            },
            "reviewed_view_ids": sorted(view_counts),
            "frontier_check_mode": (
                "checked_unselected_frontier"
                if frontier_challenge_required
                else "no_adjacent_unselected_frontier"
            ),
            "checked_unselected_frontier_fragment_count": len(checked_frontier_ids),
            "unselected_frontier_fragment_count": len(unselected_frontier_ids),
            "frontier_challenge_required": frontier_challenge_required,
        },
        artifacts,
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    review_path = args.review.resolve()
    output_path = args.output.resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    try:
        review_path.relative_to(run_dir)
        output_path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("Review and output paths must stay under --run-dir") from exc
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")

    review = _load_object(review_path, "falsification review")
    if not schema_matches(review.get("schema_version"), REVIEW_SCHEMA_VERSION):
        raise ValueError("Unsupported falsification review schema")
    if review.get("status") != "accepted":
        raise ValueError("falsification review status must be 'accepted'")
    semantic_part = _require_string(review, "semantic_part")
    segment_id = review.get("segment_id")
    if (
        not isinstance(segment_id, int)
        or isinstance(segment_id, bool)
        or segment_id <= 0
    ):
        raise ValueError("segment_id must be a positive integer")
    revision = _require_string(review, "revision")
    if Path(revision).name != revision or not revision.startswith("rev-"):
        raise ValueError("revision must be a safe rev-* directory name")

    review_dir = review_path.parent
    plan_path = _resolve_artifact(
        run_dir,
        review_dir,
        review.get("falsification_plan"),
    )
    plan = _load_object(plan_path, "falsification plan")
    if not schema_matches(plan.get("schema_version"), PLAN_SCHEMA_VERSION):
        raise ValueError("Unsupported falsification plan schema")
    if _require_string(plan, "semantic_part") != semantic_part:
        raise ValueError("falsification plan semantic_part does not match review")
    if plan.get("segment_id") != segment_id:
        raise ValueError("falsification plan segment_id does not match review")

    construction_view_ids = _require_string_list(
        review,
        "construction_view_ids",
    )
    held_out_view_ids = _require_string_list(review, "held_out_view_ids")
    if set(construction_view_ids) & set(held_out_view_ids):
        raise ValueError("held_out_view_ids must be disjoint from construction views")
    expected_instances = _require_string_list(
        review,
        "expected_instances_checked",
    )
    confusers = _require_string_list(review, "confusers_checked")
    if construction_view_ids != _require_string_list(
        plan,
        "construction_view_ids",
    ):
        raise ValueError("review construction views do not match falsification plan")
    if held_out_view_ids != _require_string_list(plan, "held_out_view_ids"):
        raise ValueError("review held-out views do not match falsification plan")
    if expected_instances != _require_string_list(plan, "expected_instances"):
        raise ValueError("review expected instances do not match falsification plan")
    plan_confusers = _require_named_objects(
        plan,
        "confusers",
        name_field="name",
    )
    if confusers != plan_confusers:
        raise ValueError("review confusers do not match falsification plan")
    if review.get("false_positive_review") != "passed":
        raise ValueError("false_positive_review must be 'passed'")
    if review.get("false_negative_review") != "passed":
        raise ValueError("false_negative_review must be 'passed'")
    if review.get("actionable_issues") != []:
        raise ValueError("accepted review must have no actionable_issues")
    if review.get("selected_only_review") != "passed":
        raise ValueError("selected_only_review must be 'passed'")
    if (
        _require_string_list(
            review,
            "repeated_instance_consistency_checked",
        )
        != expected_instances
    ):
        raise ValueError(
            "repeated_instance_consistency_checked must match expected instances"
        )
    uncertainties = review.get("unresolved_uncertainties")
    if not isinstance(uncertainties, list) or any(
        not isinstance(value, str) for value in uncertainties
    ):
        raise ValueError("unresolved_uncertainties must be a string array")

    candidate_path = _resolve_artifact(
        run_dir,
        review_dir,
        review.get("candidate_labels"),
    )
    if candidate_path.parent.name != revision:
        raise ValueError("candidate_labels does not belong to the recorded revision")
    candidate = np.fromfile(candidate_path, dtype="<u4")
    if not len(candidate):
        raise ValueError("candidate_labels is empty")

    evidence_path = _resolve_artifact(
        run_dir,
        review_dir,
        review.get("signed_evidence"),
    )
    evidence = _load_object(evidence_path, "signed evidence")
    if not schema_matches(evidence.get("schema_version"), EVIDENCE_SCHEMA_VERSION):
        raise ValueError("Unsupported signed evidence schema")
    accepted_events = _accepted_signed_events(evidence)
    positive_face_ids = sorted(
        {
            int(event["face_id"])
            for event in accepted_events
            if event["polarity"] == "positive"
        }
    )
    negative_face_ids = sorted(
        {
            int(event["face_id"])
            for event in accepted_events
            if event["polarity"] == "negative"
        }
    )
    if not positive_face_ids or not negative_face_ids:
        raise ValueError(
            "signed evidence must contain accepted positive and negative faces"
        )
    if max([*positive_face_ids, *negative_face_ids]) >= len(candidate):
        raise ValueError("signed evidence references a face outside candidate_labels")
    unsatisfied_positive = [
        face_id
        for face_id in positive_face_ids
        if int(candidate[face_id]) != segment_id
    ]
    violated_negative = [
        face_id
        for face_id in negative_face_ids
        if int(candidate[face_id]) == segment_id
    ]
    if unsatisfied_positive:
        raise ValueError(
            f"candidate omits positive evidence faces: {unsatisfied_positive[:20]}"
        )
    if violated_negative:
        raise ValueError(
            f"candidate includes negative evidence faces: {violated_negative[:20]}"
        )
    challenge_coverage = _validate_challenge_coverage(
        accepted_events=accepted_events,
        expected_instances=expected_instances,
        confusers=confusers,
        held_out_view_ids=held_out_view_ids,
        run_dir=run_dir,
    )

    comparison_path = _resolve_artifact(
        run_dir,
        review_dir,
        review.get("label_comparison"),
    )
    comparison = _load_object(comparison_path, "label comparison")
    if not schema_matches(comparison.get("schema_version"), COMPARISON_SCHEMA_VERSION):
        raise ValueError("Unsupported label comparison schema")
    if comparison.get("status") != "passed" or comparison.get("failures") != []:
        raise ValueError("label comparison did not pass")
    if comparison.get("active_segment_id") != segment_id:
        raise ValueError("label comparison active segment does not match review")
    if comparison.get("candidate_labels_sha256") != sha256_file(candidate_path):
        raise ValueError("label comparison candidate digest is stale")
    if comparison.get("unsatisfied_positive_face_ids") != []:
        raise ValueError("label comparison reports unsatisfied positive evidence")
    if comparison.get("violated_negative_face_ids") != []:
        raise ValueError("label comparison reports violated negative evidence")
    if comparison.get("locked_face_change_count") != 0:
        raise ValueError("label comparison reports changes to locked faces")

    frontier_path = _resolve_artifact(
        run_dir,
        review_dir,
        review.get("frontier_audit"),
    )
    frontier = _load_object(frontier_path, "frontier audit")
    if frontier.get("active_segment_id") != segment_id:
        raise ValueError("frontier audit active segment does not match review")
    if frontier.get("face_labels_sha256") != sha256_file(candidate_path):
        raise ValueError("frontier audit candidate digest is stale")

    render_manifest_path = _resolve_artifact(
        run_dir,
        review_dir,
        review.get("latest_render_manifest"),
    )
    render_manifest = _load_object(
        render_manifest_path,
        "latest render manifest",
    )
    flat_label_images = _manifest_render_images(
        manifest=render_manifest,
        manifest_path=render_manifest_path,
        run_dir=run_dir,
        label="latest render manifest",
    )
    id_buffer_values = review.get("id_buffer_manifests")
    if not isinstance(id_buffer_values, list) or not id_buffer_values:
        raise ValueError("id_buffer_manifests must contain at least one path")
    id_buffer_manifests = [
        _resolve_artifact(run_dir, review_dir, value) for value in id_buffer_values
    ]
    selected_export_path = _resolve_artifact(
        run_dir,
        review_dir,
        review.get("selected_only_export_manifest"),
    )
    selected_export = _load_object(
        selected_export_path,
        "selected-only export manifest",
    )
    if not schema_matches(selected_export.get("schema_version"), EXPORT_SCHEMA_VERSION):
        raise ValueError("Unsupported selected-only export manifest schema")
    if selected_export.get("face_labels_sha256") != sha256_file(candidate_path):
        raise ValueError("selected-only export is not bound to candidate labels")
    selected_export_usd = _resolve_artifact(
        run_dir,
        selected_export_path.parent,
        selected_export.get("output_usd"),
    )
    if selected_export.get("output_usd_sha256") != sha256_file(selected_export_usd):
        raise ValueError("selected-only export USD digest is stale")

    selected_stage_manifest_path = _resolve_artifact(
        run_dir,
        review_dir,
        review.get("selected_only_stage_manifest"),
    )
    selected_stage_manifest = _load_object(
        selected_stage_manifest_path,
        "selected-only stage manifest",
    )
    if not schema_matches(
        selected_stage_manifest.get("schema_version"), SELECTED_ONLY_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported selected-only stage manifest schema")
    if selected_stage_manifest.get("status") != "passed":
        raise ValueError("selected-only stage manifest did not pass")
    selected_stage_source = _resolve_artifact(
        run_dir,
        selected_stage_manifest_path.parent,
        selected_stage_manifest.get("source_usd"),
    )
    if selected_stage_source != selected_export_usd:
        raise ValueError("selected-only stage does not use the candidate export")
    if selected_stage_manifest.get("source_usd_sha256") != sha256_file(
        selected_stage_source
    ):
        raise ValueError("selected-only stage source digest is stale")
    selected_only_usd = _resolve_artifact(
        run_dir,
        selected_stage_manifest_path.parent,
        selected_stage_manifest.get("output_usd"),
    )
    if selected_stage_manifest.get("output_usd_sha256") != sha256_file(
        selected_only_usd
    ):
        raise ValueError("selected-only stage output digest is stale")

    selected_render_manifest_path = _resolve_artifact(
        run_dir,
        review_dir,
        review.get("selected_only_render_manifest"),
    )
    selected_render_manifest = _load_object(
        selected_render_manifest_path,
        "selected-only render manifest",
    )
    selected_render_scene = _resolve_artifact(
        run_dir,
        selected_render_manifest_path.parent,
        selected_render_manifest.get("scene"),
    )
    if selected_render_scene != selected_only_usd:
        raise ValueError("selected-only renders do not use the selected-only stage")
    if selected_render_manifest.get("focus") != selected_stage_manifest.get(
        "target_prim"
    ):
        raise ValueError("selected-only render focus does not match target segment")
    selected_only_images = _manifest_render_images(
        manifest=selected_render_manifest,
        manifest_path=selected_render_manifest_path,
        run_dir=run_dir,
        label="selected-only render manifest",
    )
    selected_render_names = set(selected_only_images)
    if not set(held_out_view_ids).issubset(selected_render_names):
        raise ValueError("selected-only renders do not cover every held-out view")
    instance_completeness_coverage, completeness_artifacts = (
        _validate_instance_completeness(
            review=review,
            expected_instances=expected_instances,
            run_dir=run_dir,
            review_dir=review_dir,
            flat_label_images=flat_label_images,
            selected_only_images=selected_only_images,
            frontier=frontier,
            frontier_path=frontier_path,
            candidate_path=candidate_path,
            segment_id=segment_id,
        )
    )
    revision_consistency, consistency_artifacts = _validate_revision_consistency(
        review=review,
        semantic_part=semantic_part,
        segment_id=segment_id,
        revision=revision,
        candidate_path=candidate_path,
        expected_instances=expected_instances,
        run_dir=run_dir,
        review_dir=review_dir,
        flat_label_manifest_path=render_manifest_path,
        flat_label_images=flat_label_images,
        selected_only_manifest_path=selected_render_manifest_path,
        selected_only_images=selected_only_images,
    )

    falsifiers = review.get("falsifiers")
    if not isinstance(falsifiers, list) or not falsifiers:
        raise ValueError("falsifiers must contain at least one result")
    falsifier_artifacts: list[Path] = []
    falsifier_ids: set[str] = set()
    for index, falsifier in enumerate(falsifiers):
        if not isinstance(falsifier, dict):
            raise ValueError(f"falsifier {index} must be an object")
        falsifier_id = _require_string(falsifier, "id")
        if falsifier_id in falsifier_ids:
            raise ValueError(f"duplicate falsifier id: {falsifier_id}")
        falsifier_ids.add(falsifier_id)
        if falsifier.get("status") != "passed":
            raise ValueError(f"falsifier {falsifier_id} did not pass")
        _require_string(falsifier, "rationale")
        evidence_values = falsifier.get("evidence")
        if not isinstance(evidence_values, list) or not evidence_values:
            raise ValueError(f"falsifier {falsifier_id} lacks evidence")
        falsifier_artifacts.extend(
            _resolve_artifact(run_dir, review_dir, value) for value in evidence_values
        )
    plan_falsifier_ids = set(
        _require_named_objects(
            plan,
            "falsifiers",
            name_field="id",
        )
    )
    if falsifier_ids != plan_falsifier_ids:
        raise ValueError("review falsifiers do not match falsification plan")

    artifacts = [
        plan_path,
        candidate_path,
        evidence_path,
        comparison_path,
        frontier_path,
        render_manifest_path,
        *id_buffer_manifests,
        selected_export_path,
        selected_export_usd,
        selected_stage_manifest_path,
        selected_only_usd,
        selected_render_manifest_path,
        *completeness_artifacts,
        *consistency_artifacts,
        *falsifier_artifacts,
    ]
    result = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "passed",
        "semantic_part": semantic_part,
        "segment_id": segment_id,
        "revision": revision,
        "review_path": str(review_path),
        "review_sha256": sha256_file(review_path),
        "candidate_labels": str(candidate_path),
        "candidate_labels_sha256": sha256_file(candidate_path),
        "positive_face_count": len(positive_face_ids),
        "negative_face_count": len(negative_face_ids),
        "construction_view_ids": construction_view_ids,
        "held_out_view_ids": held_out_view_ids,
        "expected_instances_checked": expected_instances,
        "confusers_checked": confusers,
        "falsifier_ids": sorted(falsifier_ids),
        "challenge_coverage": challenge_coverage,
        "instance_completeness_coverage": instance_completeness_coverage,
        "revision_consistency": revision_consistency,
        "artifacts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in dict.fromkeys(artifacts)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
