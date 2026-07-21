# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply material decision patches and write standard workflow artifacts."""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from content_agent_workflows.common import (
    EvidenceArtifact,
    material_assignment_validation_evidence,
)

from content_workflow_cli.material_policy import (
    MATERIAL_ASSIGNMENT_TARGET_PRIM_SOFT_CAP,
    PAINTED_OR_SATURATED_MATERIAL_TAGS,
    structured_finalizer_rejection,
)
from content_workflow_cli.trace import append_jsonl, utc_now

from .material_grounding import grounding_operation_counts
from .material_run_packet import _download_to_file, _post_json, _render_view

logger = logging.getLogger(__name__)

_MISSING = object()

FINAL_RENDER_VIEWS = (
    {"name": "final_top", "direction": "+z"},
    {"name": "final_oblique", "direction": "+x-y+z"},
    {"name": "final_side_px", "direction": "+x"},
    {"name": "final_front_py", "direction": "+y"},
)
TURNTABLE_RENDER_NAME = "final_turntable"
TURNTABLE_FRAME_COUNT = 24
TURNTABLE_FPS = 8
TURNTABLE_Z_WEIGHT = 0.62
TURNTABLE_GIF_ENCODE_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class MaterialFinalizeConfig:
    workbench_url: str
    run_dir: Path
    session_id: str
    source_usd: Path
    materials_usd: Path
    reference_images: list[Path]
    decision_patch: dict[str, Any]
    reference_files: list[Path] | None = None
    target_runtime: str = "unspecified"
    width: int = 768
    height: int = 576
    render_quality: str = "final"


def finalize_material_decisions(config: MaterialFinalizeConfig) -> dict[str, str]:
    """Apply model decisions and write assignments/counts/VQA/summary artifacts."""

    run_dir = config.run_dir
    raw_dir = run_dir / "raw"
    final_dir = run_dir / "final_renders"
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    packet = _load_json(raw_dir / "material_run_packet.json", default={})
    seed = _load_json(raw_dir / "material_assignment_seed.json")
    palette = _load_json(raw_dir / "material_palette.json")
    candidates = _load_json(raw_dir / "visible_candidate_prims.json")
    decision_patch_path = raw_dir / "material_decision_patch.json"
    decision_patch_path.write_text(
        json.dumps(config.decision_patch, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    respect_existing_material_bindings = bool(
        packet.get("respect_existing_material_bindings")
    )
    alias_target_paths = _candidate_alias_target_path_map(candidates)
    runtime_alias_target_paths = _candidate_runtime_alias_target_path_map(candidates)
    alias_to_target_paths = _candidate_alias_to_target_path_map(
        alias_target_paths=alias_target_paths,
    )
    ambiguous_alias_paths = _candidate_ambiguous_alias_paths(
        alias_target_paths=alias_target_paths,
    )
    candidate_paths = _candidate_path_set(candidates)
    candidates_by_path = _candidate_by_path(candidates)
    material_groups, rejected_groups = _normalize_material_assignment_groups(
        config.decision_patch,
        palette,
        candidates,
        runtime_alias_target_paths=runtime_alias_target_paths,
        alias_to_target_paths=alias_to_target_paths,
        ambiguous_alias_paths=ambiguous_alias_paths,
        candidate_paths=candidate_paths,
    )
    reviewed_groups, rejected_reviewed_groups = _normalize_reviewed_no_override_groups(
        config.decision_patch,
        candidates,
        allow_preserved_existing=respect_existing_material_bindings,
        alias_to_target_paths=alias_to_target_paths,
        ambiguous_alias_paths=ambiguous_alias_paths,
        candidate_paths=candidate_paths,
    )
    rejected_groups.extend(rejected_reviewed_groups)
    rejected_payload = json.dumps(rejected_groups, indent=2, sort_keys=True)
    (raw_dir / "rejected_material_assignments.json").write_text(
        rejected_payload,
        encoding="utf-8",
    )
    applied_records = []
    rejected_apply_failures: list[dict[str, Any]] = []
    for group_index, group in enumerate(material_groups, start=1):
        group_records = _apply_material_assignment_group(
            workbench_url=config.workbench_url.rstrip("/"),
            session_id=config.session_id,
            raw_dir=raw_dir,
            group=group,
            group_index=group_index,
            materials_usd=config.materials_usd,
            runtime_alias_target_paths=runtime_alias_target_paths,
            alias_to_target_paths=alias_to_target_paths,
            ambiguous_alias_paths=ambiguous_alias_paths,
            candidate_paths=candidate_paths,
            candidates_by_path=candidates_by_path,
        )
        applied_records.extend(group_records)
        failed_canonical_paths = _dedupe_strings(
            [
                str(path)
                for record in group_records
                if record.get("success") is False
                for path in record["canonical_prim_paths"]
            ]
        )
        if failed_canonical_paths:
            rejected_apply_failures.append(
                _shared_runtime_alias_rejection_group(
                    group,
                    failed_canonical_paths,
                    candidates,
                    path_space=str(group.get("path_space") or "source"),
                    reason=(
                        "Workbench rejected the material_override command for "
                        "these canonical candidates (see the matching "
                        "material_assignment_*_response.json / command log for "
                        "the underlying error). Coverage was not counted as "
                        "assigned since the live override could not be applied."
                    ),
                )
            )
            # Reuse `_material_group_with_paths` (not a bare `prim_paths`
            # mutation) so `source_prim_paths`/`runtime_prim_paths` are
            # recomputed alongside it; both are copied into assignments.json
            # via `_build_assignments` and would otherwise still list the
            # failed candidate even though `prim_paths` no longer does.
            material_groups[group_index - 1] = _material_group_with_paths(
                group,
                [
                    path
                    for path in _string_list(group.get("prim_paths"))
                    if path not in set(failed_canonical_paths)
                ],
                candidates,
                path_space=str(group.get("path_space") or "source"),
            )

    if rejected_apply_failures:
        rejected_groups.extend(rejected_apply_failures)
        rejected_payload = json.dumps(rejected_groups, indent=2, sort_keys=True)
        (raw_dir / "rejected_material_assignments.json").write_text(
            rejected_payload,
            encoding="utf-8",
        )

    final_renders = [
        _render_view(
            workbench_url=config.workbench_url.rstrip("/"),
            session_id=config.session_id,
            output_dir=final_dir,
            name=str(view["name"]),
            direction=str(view["direction"]),
            width=config.width,
            height=config.height,
            render_quality=config.render_quality,
        )
        for view in FINAL_RENDER_VIEWS
    ]
    final_renders.append(
        _render_turntable_view(
            workbench_url=config.workbench_url.rstrip("/"),
            session_id=config.session_id,
            output_dir=final_dir,
            width=config.width,
            height=config.height,
            render_quality=config.render_quality,
        )
    )
    (raw_dir / "final_render_records.json").write_text(
        json.dumps(final_renders, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    assignments = _build_assignments(
        seed=seed,
        packet=packet,
        material_groups=material_groups,
        reviewed_groups=reviewed_groups,
        decision_patch=config.decision_patch,
        rejected_groups=rejected_groups,
        session_id=config.session_id,
        source_usd=config.source_usd,
        materials_usd=config.materials_usd,
        final_renders=final_renders,
        reference_images=config.reference_images,
        reference_files=config.reference_files or [],
    )
    counts = _build_counts(
        run_dir=run_dir,
        packet=packet,
        material_groups=material_groups,
        applied_records=applied_records,
        final_renders=final_renders,
        assignments=assignments,
    )
    summary = _build_summary(
        assignments=assignments,
        counts=counts,
        final_renders=final_renders,
        decision_patch=config.decision_patch,
    )
    validation_evidence = _build_validation_evidence(
        assignments=assignments,
        final_renders=final_renders,
        source_usd=config.source_usd,
        target_runtime=config.target_runtime,
    )

    paths = {
        "assignments": run_dir / "assignments.json",
        "api_operation_counts": run_dir / "api_operation_counts.json",
        "visual_quality_assessment": run_dir / "visual_quality_assessment.json",
        "validation_evidence": run_dir / "validation_evidence.json",
        "final_summary": run_dir / "final_summary.md",
        "decision_patch": decision_patch_path,
    }
    paths["assignments"].write_text(
        json.dumps(assignments, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["api_operation_counts"].write_text(
        json.dumps(counts, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["visual_quality_assessment"].write_text(
        json.dumps(assignments["visual_quality_assessment"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["validation_evidence"].write_text(
        json.dumps(validation_evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["final_summary"].write_text(summary, encoding="utf-8")
    _append_finalize_trace(run_dir, paths, material_groups, final_renders)
    return {name: str(path) for name, path in paths.items()}


def _turntable_directions() -> list[str]:
    directions: list[str] = []
    for index in range(TURNTABLE_FRAME_COUNT):
        angle = 2.0 * math.pi * index / TURNTABLE_FRAME_COUNT
        directions.append(
            f"{math.cos(angle):+.4f}x{math.sin(angle):+.4f}y{TURNTABLE_Z_WEIGHT:+.4f}z"
        )
    return directions


def _encode_turntable_gif(
    *, frames_dir: Path, output_path: Path, width: int
) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    ffmpeg_error: str | None = None
    ffmpeg_skip_reason = "ffmpeg_not_found" if not ffmpeg else None
    if ffmpeg:
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-framerate",
                    str(TURNTABLE_FPS),
                    "-i",
                    str(frames_dir / "frame_%03d.png"),
                    "-vf",
                    (
                        f"fps={TURNTABLE_FPS},scale={width}:-1:flags=lanczos,"
                        "split[s0][s1];[s0]palettegen=max_colors=128[p];"
                        "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
                    ),
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=TURNTABLE_GIF_ENCODE_TIMEOUT_SECONDS,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            ffmpeg_skip_reason = "ffmpeg_failed"
            ffmpeg_error = (exc.stderr or exc.stdout or str(exc)).strip()
        except subprocess.TimeoutExpired as exc:
            ffmpeg_skip_reason = "ffmpeg_timeout"
            ffmpeg_error = str(exc)
        else:
            return {
                "encoded": True,
                "encoder": "ffmpeg",
                "image_path": str(output_path),
            }

    pillow_result = _encode_turntable_gif_with_pillow(
        frames_dir=frames_dir,
        output_path=output_path,
        width=width,
    )
    if pillow_result.get("encoded"):
        if not ffmpeg:
            pillow_result["ffmpeg_skip_reason"] = "ffmpeg_not_found"
        elif ffmpeg_error:
            pillow_result["ffmpeg_skip_reason"] = ffmpeg_skip_reason
            pillow_result["ffmpeg_error"] = ffmpeg_error
        return pillow_result
    pillow_skip_reason = pillow_result.get("skip_reason")
    if pillow_skip_reason:
        pillow_result["pillow_skip_reason"] = pillow_skip_reason
    if ffmpeg_skip_reason:
        pillow_result["skip_reason"] = ffmpeg_skip_reason
    if ffmpeg_error:
        pillow_result["ffmpeg_error"] = ffmpeg_error
    if output_path.exists():
        try:
            output_path.unlink()
        except OSError as exc:
            pillow_result["cleanup_error"] = str(exc)
    return pillow_result


def _encode_turntable_gif_with_pillow(
    *, frames_dir: Path, output_path: Path, width: int
) -> dict[str, Any]:
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        return {"encoded": False, "skip_reason": "pillow_no_frames"}
    try:
        from PIL import Image
    except ImportError as exc:
        return {
            "encoded": False,
            "skip_reason": "pillow_not_available",
            "error": str(exc),
        }

    frames = []
    try:
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                frame = image.convert("RGBA")
                if width > 0 and frame.width != width:
                    height = max(1, round(frame.height * width / frame.width))
                    frame = frame.resize((width, height), Image.Resampling.LANCZOS)
                frames.append(frame)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=max(1, round(1000 / TURNTABLE_FPS)),
            loop=0,
            optimize=True,
        )
    except Exception as exc:  # noqa: BLE001 - report best-effort fallback failure
        return {
            "encoded": False,
            "skip_reason": "pillow_failed",
            "error": str(exc),
        }
    return {
        "encoded": True,
        "encoder": "pillow",
        "image_path": str(output_path),
    }


def _render_turntable_view(
    *,
    workbench_url: str,
    session_id: str,
    output_dir: Path,
    width: int,
    height: int,
    render_quality: str,
) -> dict[str, Any]:
    frames_dir = output_dir / TURNTABLE_RENDER_NAME / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_records: list[dict[str, Any]] = []
    batched_render_used = False
    batched_render_error: str | None = None
    encoded_session_id = quote(session_id, safe="")
    try:
        response = _post_json(
            f"{workbench_url}/sessions/{encoded_session_id}/render-frames",
            {
                "width": width,
                "height": height,
                "frames": f"0:{TURNTABLE_FRAME_COUNT - 1}",
                "directions": _turntable_directions(),
                "use_session_camera": False,
                "margin": 1.25,
                "render_quality": render_quality,
                "save_camera_json": True,
            },
        )
        frame_urls = response.get("frame_urls")
        if not isinstance(frame_urls, list) or not frame_urls:
            raise RuntimeError("Workbench frame render response missing frame_urls")
        directions = _turntable_directions()
        if len(frame_urls) > len(directions):
            raise RuntimeError(
                "Workbench frame render response returned more frames than requested"
            )
        batched_render_used = True
        camera_urls = response.get("camera_json_urls")
        for index, frame_url in enumerate(frame_urls):
            if not isinstance(frame_url, str) or not frame_url:
                raise RuntimeError("Workbench frame render response has invalid URL")
            image_path = frames_dir / f"frame_{index:03d}.png"
            camera_path = frames_dir / f"frame_{index:03d}_camera.json"
            _download_to_file(f"{workbench_url}{frame_url}", image_path)
            artifact_download_count = 1
            if (
                isinstance(camera_urls, list)
                and index < len(camera_urls)
                and isinstance(camera_urls[index], str)
                and camera_urls[index]
            ):
                _download_to_file(f"{workbench_url}{camera_urls[index]}", camera_path)
                artifact_download_count += 1
            frame_records.append(
                {
                    "name": f"frame_{index:03d}",
                    "direction": directions[index],
                    "width": width,
                    "height": height,
                    "render_quality": render_quality,
                    "image_path": str(image_path),
                    "camera_json_path": str(camera_path)
                    if camera_path.exists()
                    else None,
                    "artifact_download_count": artifact_download_count,
                    "elapsed_seconds": response.get("elapsed_seconds"),
                    "ovrtx_render_mode": response.get("ovrtx_render_mode"),
                    "ovrtx_num_sensor_updates": response.get(
                        "ovrtx_num_sensor_updates"
                    ),
                }
            )
        (output_dir / f"{TURNTABLE_RENDER_NAME}_response.json").write_text(
            json.dumps(response, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 - fall back while preserving diagnostics
        batched_render_error = str(exc)
        frame_records = [
            _render_view(
                workbench_url=workbench_url,
                session_id=session_id,
                output_dir=frames_dir,
                name=f"frame_{index:03d}",
                direction=direction,
                width=width,
                height=height,
                render_quality=render_quality,
            )
            for index, direction in enumerate(_turntable_directions())
        ]
    gif_path = output_dir / f"{TURNTABLE_RENDER_NAME}.gif"
    gif_record = _encode_turntable_gif(
        frames_dir=frames_dir,
        output_path=gif_path,
        width=width,
    )
    gif_encoded = bool(gif_record.get("encoded"))
    result = {
        "name": TURNTABLE_RENDER_NAME,
        "direction": "turntable_up_axis",
        "width": width,
        "height": height,
        "render_quality": render_quality,
        "image_path": str(gif_path) if gif_encoded else None,
        "gif_encoded": gif_encoded,
        "gif_skip_reason": None if gif_encoded else gif_record.get("skip_reason"),
        "gif_error": None
        if gif_encoded
        else gif_record.get("error") or gif_record.get("ffmpeg_error"),
        "frame_count": len(frame_records),
        "render_call_count": 1 if batched_render_used else len(frame_records),
        "artifact_download_count": sum(
            _render_artifact_download_count(record) for record in frame_records
        )
        + (1 if gif_encoded else 0),
        "frame_records": frame_records,
    }
    if gif_encoded and gif_record.get("ffmpeg_skip_reason"):
        result["gif_fallback_reason"] = gif_record.get("ffmpeg_skip_reason")
    if gif_encoded and gif_record.get("ffmpeg_error"):
        result["gif_fallback_error"] = gif_record.get("ffmpeg_error")
    if batched_render_error:
        result["batched_render_error"] = batched_render_error
    return result


def apply_post_apply_visual_quality(
    *,
    run_dir: Path,
    visual_quality: dict[str, Any],
    validator_artifact: Path | None = None,
) -> dict[str, str]:
    """Merge independent post-apply VQA into standard material artifacts."""

    paths = {
        "assignments": run_dir / "assignments.json",
        "api_operation_counts": run_dir / "api_operation_counts.json",
        "visual_quality_assessment": run_dir / "visual_quality_assessment.json",
        "validation_evidence": run_dir / "validation_evidence.json",
        "final_summary": run_dir / "final_summary.md",
    }
    assignments = _load_json(paths["assignments"])
    if not isinstance(assignments, dict):
        raise RuntimeError("assignments.json must contain a JSON object")
    counts = _load_json(paths["api_operation_counts"], default={})
    if not isinstance(counts, dict):
        counts = {}
    existing_vqa = assignments.get("visual_quality_assessment")
    if not isinstance(existing_vqa, dict):
        existing_vqa = _load_json(paths["visual_quality_assessment"], default={})
    if not isinstance(existing_vqa, dict):
        existing_vqa = {}

    merged_vqa = _merge_post_apply_visual_quality(
        existing_vqa=existing_vqa,
        post_apply_vqa=visual_quality,
        validator_artifact=validator_artifact,
    )
    assignments["visual_quality_assessment"] = merged_vqa
    post_vqa = _normalize_visual_quality(visual_quality)

    final_review = assignments.get("final_review")
    if isinstance(final_review, dict):
        if isinstance(final_review.get("issues_found"), list):
            final_review["issues_found"] = _dedupe_strings(
                _string_list(final_review.get("issues_found"))
                + _string_list(post_vqa.get("issues_found"))
            )
        if isinstance(final_review.get("issues_fixed"), list):
            final_review["issues_fixed"] = _dedupe_strings(
                _string_list(final_review.get("issues_fixed"))
                + _string_list(post_vqa.get("issues_fixed"))
            )
        final_review["unresolved_issues"] = merged_vqa["unresolved_issues"]
        if merged_vqa["unresolved_issues"]:
            notes = str(final_review.get("review_notes") or "").strip()
            suffix = "Post-apply visual validation reported unresolved issues."
            final_review["review_notes"] = f"{notes} {suffix}".strip()
        assignments["final_review"] = final_review

    counts["visual_quality_issues_found"] = _count_items(merged_vqa.get("issues_found"))
    counts["visual_quality_issues_fixed"] = _count_items(merged_vqa.get("issues_fixed"))
    if isinstance(final_review, dict):
        counts["final_review_issues_found"] = _count_items(
            final_review.get("issues_found")
        )
        counts["final_review_issues_fixed"] = _count_items(
            final_review.get("issues_fixed")
        )
    counts["post_apply_vqa_issues_found"] = _count_items(post_vqa.get("issues_found"))
    counts["post_apply_vqa_unresolved_issues"] = _count_items(
        post_vqa.get("unresolved_issues")
    )

    final_renders = _load_json(
        run_dir / "raw" / "final_render_records.json",
        default=[],
    )
    if not isinstance(final_renders, list):
        final_renders = []
    decision_patch = _load_json(
        run_dir / "raw" / "material_decision_patch.json",
        default={},
    )
    if not isinstance(decision_patch, dict):
        decision_patch = {}

    paths["assignments"].write_text(
        json.dumps(assignments, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["api_operation_counts"].write_text(
        json.dumps(counts, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["visual_quality_assessment"].write_text(
        json.dumps(merged_vqa, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["validation_evidence"].write_text(
        json.dumps(
            _build_validation_evidence(
                assignments=assignments,
                final_renders=final_renders,
                source_usd=Path(str(assignments.get("source_usd") or "unknown")),
                target_runtime=_existing_validation_target_runtime(run_dir),
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths["final_summary"].write_text(
        _build_summary(
            assignments=assignments,
            counts=counts,
            final_renders=final_renders,
            decision_patch=decision_patch,
        ),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


def _build_validation_evidence(
    *,
    assignments: dict[str, Any],
    final_renders: list[dict[str, Any]],
    source_usd: Path,
    target_runtime: str,
) -> dict[str, Any]:
    visual_quality = assignments.get("visual_quality_assessment")
    if not isinstance(visual_quality, dict):
        visual_quality = {}
    unresolved_issues = _string_list(visual_quality.get("unresolved_issues"))
    coverage = assignments.get("coverage")
    if not isinstance(coverage, dict):
        coverage = {}
    failures = []
    missing_count = int(coverage.get("missing_assignment_prim_count") or 0)
    rejected_count = int(coverage.get("rejected_assignment_prim_count") or 0)
    if missing_count:
        failures.append(
            f"{missing_count} visible candidate prim(s) have no material assignment."
        )
    if rejected_count:
        failures.append(
            f"{rejected_count} visible candidate prim(s) had rejected assignments."
        )
    artifacts = [
        EvidenceArtifact(
            kind="render",
            path=str(record["image_path"]),
            description=f"Material assignment verification render: {record.get('name')}",
        )
        for record in final_renders
        if isinstance(record, dict) and isinstance(record.get("image_path"), str)
    ]
    visual_status = "fail" if failures else "warning" if unresolved_issues else "pass"
    return material_assignment_validation_evidence(
        asset=str(source_usd),
        target_runtime=target_runtime,
        visual_materials_status=visual_status,
        evidence_artifacts=artifacts,
        failures=failures,
        warnings=unresolved_issues,
        unresolved_issues=unresolved_issues,
    ).model_dump()


def _existing_validation_target_runtime(run_dir: Path) -> str:
    evidence = _load_json(run_dir / "validation_evidence.json", default={})
    if isinstance(evidence, dict) and isinstance(evidence.get("target_runtime"), str):
        return str(evidence["target_runtime"])
    return "unspecified"


def _normalize_material_assignment_groups(
    decision_patch: dict[str, Any],
    palette: dict[str, Any],
    candidates: dict[str, Any],
    *,
    runtime_alias_target_paths: dict[str, list[str]],
    alias_to_target_paths: dict[str, list[str]],
    ambiguous_alias_paths: set[str],
    candidate_paths: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    materials_by_name = {
        str(material.get("name")): material
        for material in palette.get("materials", [])
        if isinstance(material, dict) and material.get("name")
    }
    shape_by_path = _shape_hints_by_path(candidates)
    path_space = _candidate_path_space(candidates)
    groups = []
    rejected = []
    claimed_paths: dict[str, str] = {}
    for index, raw_group in enumerate(_material_assignment_items(decision_patch)):
        if not isinstance(raw_group, dict):
            continue
        prim_paths = _group_target_paths(
            raw_group,
            candidates,
            alias_to_target_paths=alias_to_target_paths,
            candidate_paths=candidate_paths,
        )
        family = str(raw_group.get("family") or f"Material assignment {index + 1}")
        material_name = str(raw_group.get("material_name") or "").strip()
        if not prim_paths or not material_name:
            continue
        valid_paths = [
            path
            for path in prim_paths
            if not candidate_paths or path in candidate_paths
        ]
        invalid_paths = [
            path
            for path in prim_paths
            if candidate_paths and path not in candidate_paths
        ]
        ambiguous_invalid_paths = [
            path for path in invalid_paths if path in ambiguous_alias_paths
        ]
        unknown_invalid_paths = [
            path for path in invalid_paths if path not in ambiguous_alias_paths
        ]
        if ambiguous_invalid_paths:
            rejected.append(
                {
                    "family": family,
                    "coverage_status": "material_assignment",
                    "prim_paths": ambiguous_invalid_paths,
                    "rejection_reason": (
                        "Rejected material assignment for ambiguous aliases that "
                        "map to multiple visible material candidates. Use exact "
                        f"{path_space}-space candidate target paths instead."
                    ),
                }
            )
        if unknown_invalid_paths:
            rejected.append(
                {
                    "family": family,
                    "coverage_status": "material_assignment",
                    "prim_paths": unknown_invalid_paths,
                    "rejection_reason": (
                        f"Rejected material assignment for {path_space}-space paths "
                        "that were not visible material candidates."
                    ),
                }
            )
        if not valid_paths:
            continue
        material = materials_by_name.get(material_name)
        if material is None:
            rejected.append(
                {
                    "family": family,
                    "coverage_status": "material_assignment",
                    "material_name": material_name,
                    "prim_paths": valid_paths,
                    "runtime_prim_paths": valid_paths
                    if path_space == "inspection"
                    else [],
                    "source_prim_paths": _source_paths_for_targets(
                        valid_paths,
                        candidates,
                    ),
                    "rationale": str(raw_group.get("rationale") or "").strip(),
                    "rejection_reason": (
                        "Rejected material assignment with material_name not found "
                        "in material_palette.json."
                    ),
                }
            )
            continue
        palette_material_path = str(material.get("material_path") or "").strip()
        requested_material_path = str(raw_group.get("material_path") or "").strip()
        if (
            requested_material_path
            and palette_material_path
            and requested_material_path != palette_material_path
        ):
            rejected.append(
                {
                    "family": family,
                    "coverage_status": "material_assignment",
                    "material_name": material_name,
                    "material_path": requested_material_path,
                    "prim_paths": valid_paths,
                    "runtime_prim_paths": valid_paths
                    if path_space == "inspection"
                    else [],
                    "source_prim_paths": _source_paths_for_targets(
                        valid_paths,
                        candidates,
                    ),
                    "rationale": str(raw_group.get("rationale") or "").strip(),
                    "rejection_reason": (
                        "Rejected material assignment whose material_path does not "
                        "match material_palette.json."
                    ),
                }
            )
            continue
        material_path = requested_material_path or palette_material_path
        if not material_path:
            rejected.append(
                {
                    "family": family,
                    "coverage_status": "material_assignment",
                    "material_name": material_name,
                    "material_path": material_path,
                    "prim_paths": valid_paths,
                    "runtime_prim_paths": valid_paths
                    if path_space == "inspection"
                    else [],
                    "source_prim_paths": _source_paths_for_targets(
                        valid_paths,
                        candidates,
                    ),
                    "rationale": str(raw_group.get("rationale") or "").strip(),
                    "rejection_reason": (
                        "Rejected material assignment without a resolvable material_path."
                    ),
                }
            )
            continue
        duplicate_paths = [path for path in valid_paths if path in claimed_paths]
        claimable_paths = [path for path in valid_paths if path not in claimed_paths]
        if duplicate_paths:
            claimed_by = _dedupe_strings(
                [
                    claimed_paths[path]
                    for path in duplicate_paths
                    if path in claimed_paths
                ]
            )
            rejected.append(
                {
                    "family": family,
                    "coverage_status": "material_assignment",
                    "material_name": material_name,
                    "material_path": material_path,
                    "prim_paths": duplicate_paths,
                    "runtime_prim_paths": duplicate_paths
                    if path_space == "inspection"
                    else [],
                    "source_prim_paths": _source_paths_for_targets(
                        duplicate_paths,
                        candidates,
                    ),
                    "rationale": str(raw_group.get("rationale") or "").strip(),
                    "rejection_reason": (
                        "Rejected duplicate material assignment for canonical "
                        "candidate target path(s) already claimed by earlier "
                        f"assignment(s): {', '.join(claimed_by)}."
                    ),
                }
            )
        if not claimable_paths:
            continue
        valid_paths = claimable_paths
        group = {
            "family": family,
            "coverage_status": "material_assignment",
            "material_name": material_name,
            "material_path": material_path,
            "material_tags": material.get("tags") or [],
            "material_description": material.get("description") or "",
            "material_manifest_semantics": material.get("manifest_semantics") or {},
            "path_space": path_space,
            "runtime_space": "inspection" if path_space == "inspection" else "source",
            "runtime_prim_paths": valid_paths if path_space == "inspection" else [],
            "source_prim_paths": _source_paths_for_targets(valid_paths, candidates),
            "prim_paths": valid_paths,
            "rationale": str(raw_group.get("rationale") or "").strip(),
        }
        rejection_reason = _rejection_reason(group, shape_by_path)
        if rejection_reason:
            rejected.append({**group, "rejection_reason": rejection_reason})
        else:
            groups.append(group)
            for path in valid_paths:
                claimed_paths[path] = family
    groups, shared_runtime_rejected = _coalesce_shared_runtime_alias_groups(
        groups,
        candidates,
        runtime_alias_target_paths=runtime_alias_target_paths,
    )
    rejected.extend(shared_runtime_rejected)
    return groups, rejected


def _coalesce_shared_runtime_alias_groups(
    groups: list[dict[str, Any]],
    candidates: dict[str, Any],
    *,
    runtime_alias_target_paths: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make optimized shared-runtime material assignments atomic.

    A single optimized runtime prim can represent multiple canonical source
    candidates. Workbench material overrides are stored with translated source
    and inspection coverage; separate commands against source aliases that map
    to the same runtime alias therefore replace each other. If every represented
    canonical candidate is assigned the same material, merge those assignments
    so command planning can emit one runtime-space override for the shared alias.
    If the component is partial or has conflicting materials, reject it instead
    of writing assignment coverage that Workbench cannot faithfully represent.
    """

    if not groups:
        return groups, []
    path_space = _candidate_path_space(candidates)
    if path_space != "source":
        return groups, []
    components = _shared_runtime_alias_target_components(
        runtime_alias_target_paths=runtime_alias_target_paths,
    )
    if not components:
        return groups, []

    target_runtime_paths = _runtime_alias_paths_by_target(runtime_alias_target_paths)
    overlapping_component_paths = _overlapping_shared_runtime_alias_paths(components)
    candidates_by_path = _candidate_by_path(candidates)
    working_groups = [dict(group) for group in groups]
    merged_groups: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for runtime_path, component_paths in components:
        component_path_set = set(component_paths)
        # A canonical candidate can have runtime fragments beyond this shared
        # alias (split+dedup combined). Workbench's session-level material
        # override tracking narrows an existing override's coverage instead of
        # deleting it outright on overlap, so such a candidate no longer needs
        # to be rejected here: leave it out of this alias's merge and let the
        # normal per-candidate command-target logic bind it atomically via its
        # own dedicated command (see `_command_targets_for_group`'s
        # `len(all_runtime_paths) > 1` branch). Only the *other* members of
        # this shared alias, if any remain, still need coalescing below.
        extra_edge_paths = {
            path
            for path in component_paths
            if any(
                alias_path != runtime_path for alias_path in target_runtime_paths[path]
            )
        }
        mergeable_path_set = component_path_set - extra_edge_paths

        # An excluded extra-edge candidate still physically shares this
        # runtime alias with the rest of the component, even though it binds
        # atomically via its own dedicated command instead of being merged
        # below. If it disagrees on material with the rest of the component,
        # its command and the mergeable members' command still overlap on
        # the shared prim, so Workbench's last write silently wins while
        # both assignments are reported successful. Check material equality
        # across the whole component (mergeable and extra-edge alike) before
        # falling back to the mergeable-only checks that follow.
        component_group_paths: list[tuple[int, list[str]]] = []
        for index, group in enumerate(working_groups):
            paths = [
                path
                for path in _string_list(group.get("prim_paths"))
                if path in component_path_set
            ]
            if paths:
                component_group_paths.append((index, paths))
        component_material_keys = {
            (
                str(working_groups[index].get("material_name") or ""),
                str(working_groups[index].get("material_path") or ""),
            )
            for index, _paths in component_group_paths
        }
        component_covered_paths = {
            path for _index, paths in component_group_paths for path in paths
        }
        # An extra-edge candidate binds atomically via its own dedicated
        # command instead of merging here, but that command still physically
        # touches the same shared runtime prim. If a sibling in this
        # component has *no* material decision at all (not even a
        # reviewed-no-override -- that still leaves it exposed, since it
        # doesn't protect the shared mesh from another member's override),
        # applying any other member's command would silently change the
        # undecided sibling's rendered appearance too, with no coverage
        # crediting or safety check having run for it. Require every member
        # of the whole component to have a decision before any of them can
        # proceed.
        component_incomplete = component_covered_paths != component_path_set
        if len(component_material_keys) > 1 or component_incomplete:
            reason = (
                "Rejected conflicting material assignments for canonical "
                "candidates that share one optimized runtime alias. "
                "Optimized Workbench sessions can store only one material "
                "override for the shared runtime path."
                if len(component_material_keys) > 1
                else "Rejected material assignment for canonical candidates "
                "that share one optimized runtime alias with a sibling that "
                "has no material decision. Applying any of these commands "
                "would still change the undecided sibling's appearance on "
                "the shared runtime prim, so the whole component must be "
                "decided together or not at all."
            )
            for index, paths in component_group_paths:
                group = working_groups[index]
                rejected.append(
                    _shared_runtime_alias_rejection_group(
                        group,
                        paths,
                        candidates,
                        path_space=path_space,
                        reason=reason,
                    )
                )
                _replace_group_paths(
                    working_groups,
                    index,
                    [
                        path
                        for path in _string_list(group.get("prim_paths"))
                        if path not in component_path_set
                    ],
                    candidates,
                    path_space=path_space,
                )
            continue

        if len(mergeable_path_set) < 2:
            continue
        mergeable_paths = [
            path for path in component_paths if path in mergeable_path_set
        ]

        group_paths: list[tuple[int, list[str]]] = [
            (index, mergeable_paths_for_group)
            for index, paths in component_group_paths
            if (
                mergeable_paths_for_group := [
                    path for path in paths if path in mergeable_path_set
                ]
            )
        ]
        if not group_paths:
            continue

        assigned_paths = _dedupe_strings(
            [path for _index, paths in group_paths for path in paths]
        )
        material_keys = {
            (
                str(working_groups[index].get("material_name") or ""),
                str(working_groups[index].get("material_path") or ""),
            )
            for index, _paths in group_paths
        }
        is_complete_component = set(assigned_paths) == mergeable_path_set
        has_overlapping_command_target = bool(
            mergeable_path_set & overlapping_component_paths
        )
        has_instance_collapsed_inspection_alias = (
            _has_instance_collapsed_inspection_alias(
                mergeable_paths,
                candidates_by_path,
            )
        )
        if (
            has_overlapping_command_target
            or has_instance_collapsed_inspection_alias
            or not is_complete_component
            or len(material_keys) > 1
        ):
            reason = (
                "Rejected material assignment for canonical candidates that "
                "participate in overlapping optimized runtime aliases. No single "
                "runtime command target covers the whole component, so Workbench "
                "would remove an earlier override when applying the next one."
                if has_overlapping_command_target
                else (
                    "Rejected material assignment for instance-collapsed "
                    "canonical candidates that share one optimized runtime "
                    "alias. The shared inspection alias can resolve back to "
                    "instance proxy paths while coverage reports canonical "
                    "source/prototype paths, so Workbench cannot apply a "
                    "faithful atomic override."
                    if has_instance_collapsed_inspection_alias
                    else (
                        "Rejected partial material assignment for canonical "
                        "candidates that share one optimized runtime alias. "
                        "Assign every represented candidate to the same material "
                        "so Workbench can apply one atomic override for the "
                        "shared runtime path."
                        if len(material_keys) == 1
                        else "Rejected conflicting material assignments for "
                        "canonical candidates that share one optimized runtime "
                        "alias. Optimized Workbench sessions can store only one "
                        "material override for the shared runtime path."
                    )
                )
            )
            for index, paths in group_paths:
                group = working_groups[index]
                rejected.append(
                    _shared_runtime_alias_rejection_group(
                        group,
                        paths,
                        candidates,
                        path_space=path_space,
                        reason=reason,
                    )
                )
                _replace_group_paths(
                    working_groups,
                    index,
                    [
                        path
                        for path in _string_list(group.get("prim_paths"))
                        if path not in mergeable_path_set
                    ],
                    candidates,
                    path_space=path_space,
                )
            continue

        first_index = min(index for index, _paths in group_paths)
        source_groups = [working_groups[index] for index, _paths in group_paths]
        merged_group = dict(working_groups[first_index])
        families = _dedupe_strings(
            [str(group.get("family") or "") for group in source_groups]
        )
        rationales = _dedupe_strings(
            [
                str(group.get("rationale") or "").strip()
                for group in source_groups
                if str(group.get("rationale") or "").strip()
            ]
        )
        if families:
            merged_group["family"] = " / ".join(families)
        if rationales:
            merged_group["rationale"] = "\n".join(rationales)
        merged_groups.append(
            _material_group_with_paths(
                merged_group,
                mergeable_paths,
                candidates,
                path_space=path_space,
            )
        )
        for index, _paths in group_paths:
            group = working_groups[index]
            _replace_group_paths(
                working_groups,
                index,
                [
                    path
                    for path in _string_list(group.get("prim_paths"))
                    if path not in mergeable_path_set
                ],
                candidates,
                path_space=path_space,
            )

    remaining_groups = [
        group for group in working_groups if _string_list(group.get("prim_paths"))
    ]
    return remaining_groups + merged_groups, rejected


def _shared_runtime_alias_target_components(
    *,
    runtime_alias_target_paths: dict[str, list[str]],
) -> list[tuple[str, list[str]]]:
    return [
        (runtime_path, target_paths)
        for runtime_path, target_paths in runtime_alias_target_paths.items()
        if len(target_paths) > 1
    ]


def _runtime_alias_paths_by_target(
    runtime_alias_target_paths: dict[str, list[str]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for runtime_path, target_paths in runtime_alias_target_paths.items():
        for target_path in target_paths:
            result.setdefault(target_path, []).append(runtime_path)
    return result


def _has_instance_collapsed_inspection_alias(
    component_paths: list[str],
    candidates_by_path: dict[str, dict[str, Any]],
) -> bool:
    return any(
        bool(candidate.get("instance_collapsed"))
        and str(candidate.get("runtime_space") or "") == "inspection"
        for candidate in (
            candidates_by_path.get(component_path, {})
            for component_path in component_paths
        )
    )


def _overlapping_shared_runtime_alias_paths(
    components: list[tuple[str, list[str]]],
) -> set[str]:
    counts: dict[str, int] = {}
    for _runtime_path, component in components:
        for path in component:
            counts[path] = counts.get(path, 0) + 1
    return {path for path, count in counts.items() if count > 1}


def _replace_group_paths(
    groups: list[dict[str, Any]],
    index: int,
    paths: list[str],
    candidates: dict[str, Any],
    *,
    path_space: str,
) -> None:
    if not paths:
        groups[index] = {}
        return
    groups[index] = _material_group_with_paths(
        groups[index],
        paths,
        candidates,
        path_space=path_space,
    )


def _material_group_with_paths(
    group: dict[str, Any],
    paths: list[str],
    candidates: dict[str, Any],
    *,
    path_space: str,
) -> dict[str, Any]:
    updated = dict(group)
    deduped_paths = _dedupe_strings(paths)
    updated["prim_paths"] = deduped_paths
    updated["runtime_prim_paths"] = deduped_paths if path_space == "inspection" else []
    updated["source_prim_paths"] = _source_paths_for_targets(deduped_paths, candidates)
    return updated


def _shared_runtime_alias_rejection_group(
    group: dict[str, Any],
    paths: list[str],
    candidates: dict[str, Any],
    *,
    path_space: str,
    reason: str,
) -> dict[str, Any]:
    deduped_paths = _dedupe_strings(paths)
    return {
        "family": group.get("family"),
        "coverage_status": "material_assignment",
        "material_name": group.get("material_name"),
        "material_path": group.get("material_path"),
        "prim_paths": deduped_paths,
        "runtime_prim_paths": deduped_paths if path_space == "inspection" else [],
        "source_prim_paths": _source_paths_for_targets(deduped_paths, candidates),
        "rationale": str(group.get("rationale") or "").strip(),
        "rejection_reason": reason,
    }


def _material_assignment_items(decision_patch: dict[str, Any]) -> list[Any]:
    material_assignments = decision_patch.get("material_assignments")
    if isinstance(material_assignments, list):
        return material_assignments
    return []


def _normalize_reviewed_no_override_groups(
    decision_patch: dict[str, Any],
    candidates: dict[str, Any],
    *,
    allow_preserved_existing: bool,
    alias_to_target_paths: dict[str, list[str]],
    ambiguous_alias_paths: set[str],
    candidate_paths: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path_space = _candidate_path_space(candidates)
    groups = []
    rejected = []
    for index, raw_group in enumerate(decision_patch.get("reviewed_no_override", [])):
        if not isinstance(raw_group, dict):
            continue
        prim_paths = _group_target_paths(
            raw_group,
            candidates,
            alias_to_target_paths=alias_to_target_paths,
            candidate_paths=candidate_paths,
        )
        if not prim_paths:
            continue
        valid_paths = [
            path
            for path in prim_paths
            if not candidate_paths or path in candidate_paths
        ]
        invalid_paths = [
            path
            for path in prim_paths
            if candidate_paths and path not in candidate_paths
        ]
        ambiguous_invalid_paths = [
            path for path in invalid_paths if path in ambiguous_alias_paths
        ]
        unknown_invalid_paths = [
            path for path in invalid_paths if path not in ambiguous_alias_paths
        ]
        family = str(raw_group.get("family") or f"Reviewed no-override {index + 1}")
        if not allow_preserved_existing:
            rejected.append(
                {
                    "family": family,
                    "coverage_status": "preserved_existing",
                    "prim_paths": valid_paths or prim_paths,
                    "rejection_reason": (
                        "Rejected reviewed-no-override decision because this session "
                        "started from clean materials. Assign an explicit library "
                        "material to each visible candidate instead."
                    ),
                }
            )
            continue
        if ambiguous_invalid_paths:
            rejected.append(
                {
                    "family": family,
                    "coverage_status": "preserved_existing",
                    "prim_paths": ambiguous_invalid_paths,
                    "rejection_reason": (
                        "Rejected reviewed-no-override decision for ambiguous "
                        "aliases that map to multiple visible material candidates. "
                        f"Use exact {path_space}-space candidate target paths "
                        "instead."
                    ),
                }
            )
        if unknown_invalid_paths:
            rejected.append(
                {
                    "family": family,
                    "coverage_status": "preserved_existing",
                    "prim_paths": unknown_invalid_paths,
                    "rejection_reason": (
                        f"Rejected reviewed-no-override decision for {path_space}-"
                        "space paths that were not visible material candidates."
                    ),
                }
            )
        if not valid_paths:
            continue
        valid_paths = _dedupe_strings(valid_paths)
        groups.append(
            {
                "family": family,
                "coverage_status": "preserved_existing",
                "material_name": "Reviewed current material",
                "material_path": None,
                "path_space": path_space,
                "runtime_space": (
                    "inspection" if path_space == "inspection" else "source"
                ),
                "runtime_prim_paths": valid_paths if path_space == "inspection" else [],
                "source_prim_paths": _source_paths_for_targets(
                    valid_paths,
                    candidates,
                ),
                "prim_paths": valid_paths,
                "rationale": str(raw_group.get("rationale") or "").strip(),
            }
        )
    return groups, rejected


def _candidate_path_set(candidates: dict[str, Any]) -> set[str]:
    paths = set()
    path_space = _candidate_path_space(candidates)
    for candidate in candidates.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        paths.update(_candidate_target_paths(candidate, path_space=path_space))
    return paths


def _shape_hints_by_path(candidates: dict[str, Any]) -> dict[str, str]:
    result = {}
    path_space = _candidate_path_space(candidates)
    for candidate in candidates.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        shape_hint = candidate.get("shape_hint")
        if not isinstance(shape_hint, str):
            continue
        for path in _candidate_target_paths(candidate, path_space=path_space):
            result[path] = shape_hint
    return result


def _candidate_path_space(candidates: dict[str, Any]) -> str:
    return (
        "inspection"
        if str(candidates.get("path_space") or "source") == "inspection"
        else "source"
    )


def _candidate_target_paths(
    candidate: dict[str, Any],
    *,
    path_space: str,
) -> list[str]:
    """Return canonical assignment targets for the active candidate path space."""
    if path_space == "inspection":
        return _candidate_runtime_paths(candidate) or _dedupe_strings(
            _path_values(candidate.get("prim_path"))
            or _path_values(candidate.get("prim_paths"))
        )
    return _dedupe_strings(
        _path_values(candidate.get("source_path"))
        or _path_values(candidate.get("source_paths"))
        or _path_values(candidate.get("prim_path"))
        or _path_values(candidate.get("prim_paths"))
    )


def _candidate_runtime_paths(candidate: dict[str, Any]) -> list[str]:
    """Return live Workbench paths when a candidate exposes optimized targets."""
    return _dedupe_strings(
        _path_values(candidate.get("runtime_paths"))
        or _path_values(candidate.get("inspection_paths"))
        or _path_values(candidate.get("runtime_path"))
        or _path_values(candidate.get("inspection_path"))
    )


def _path_values(value: Any) -> list[str]:
    """Coerce a scalar path or list of paths into a clean string list."""
    if isinstance(value, str) and value:
        return [value]
    return _string_list(value)


def _candidate_alias_paths(candidate: dict[str, Any]) -> list[str]:
    """Return all path fields that may appear in a model-authored patch."""
    paths: list[str] = []
    for key in (
        "runtime_path",
        "runtime_paths",
        "runtime_prim_paths",
        "inspection_path",
        "inspection_paths",
        "inspection_prim_paths",
        "source_path",
        "source_paths",
        "source_prim_paths",
        "original_source_path",
        "original_source_paths",
        "prim_path",
        "prim_paths",
    ):
        paths.extend(_path_values(candidate.get(key)))
    return _dedupe_strings(paths)


def _candidate_alias_target_path_map(
    candidates: dict[str, Any],
) -> dict[str, list[str]]:
    """Map candidate aliases to every distinct canonical target they reference."""
    path_space = _candidate_path_space(candidates)
    result: dict[str, list[str]] = {}
    for candidate in candidates.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        target_paths = _candidate_target_paths(candidate, path_space=path_space)
        if not target_paths:
            continue
        for alias_path in _dedupe_strings(
            _candidate_alias_paths(candidate) + target_paths
        ):
            result.setdefault(alias_path, [])
            for target_path in target_paths:
                if target_path not in result[alias_path]:
                    result[alias_path].append(target_path)
    return result


def _candidate_runtime_alias_target_path_map(
    candidates: dict[str, Any],
) -> dict[str, list[str]]:
    """Map each optimized runtime alias to canonical targets it represents."""

    path_space = _candidate_path_space(candidates)
    result: dict[str, list[str]] = {}
    for candidate in candidates.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        target_paths = _candidate_target_paths(candidate, path_space=path_space)
        if not target_paths:
            continue
        for runtime_path in _candidate_runtime_paths(candidate):
            result.setdefault(runtime_path, [])
            for target_path in target_paths:
                if target_path not in result[runtime_path]:
                    result[runtime_path].append(target_path)
    return result


def _candidate_alias_to_target_path_map(
    *,
    alias_target_paths: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Return aliases that resolve to exactly one canonical target path."""
    return {
        alias_path: target_paths
        for alias_path, target_paths in alias_target_paths.items()
        if len(target_paths) == 1
    }


def _candidate_ambiguous_alias_paths(
    *,
    alias_target_paths: dict[str, list[str]],
) -> set[str]:
    return {
        alias_path
        for alias_path, target_paths in alias_target_paths.items()
        if len(target_paths) > 1
    }


def _source_paths_by_candidate_path(candidates: dict[str, Any]) -> dict[str, list[str]]:
    path_space = _candidate_path_space(candidates)
    result: dict[str, list[str]] = {}
    for candidate in candidates.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        candidate_paths = _candidate_target_paths(candidate, path_space=path_space)
        if not candidate_paths:
            continue
        source_paths = _path_values(candidate.get("source_paths"))
        if path_space == "inspection":
            source_paths = _dedupe_strings(
                _path_values(candidate.get("source_path"))
                + source_paths
                + _path_values(candidate.get("original_source_path"))
                + _path_values(candidate.get("original_source_paths"))
            )
        for candidate_path in candidate_paths:
            result[candidate_path] = _dedupe_strings(source_paths or [candidate_path])
    return result


def _candidate_by_path(candidates: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path_space = _candidate_path_space(candidates)
    result: dict[str, dict[str, Any]] = {}
    for candidate in candidates.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        for candidate_path in _candidate_target_paths(candidate, path_space=path_space):
            result[candidate_path] = candidate
    return result


def _group_target_paths(
    raw_group: dict[str, Any],
    candidates: dict[str, Any],
    *,
    alias_to_target_paths: dict[str, list[str]],
    candidate_paths: set[str],
) -> list[str]:
    path_space = _candidate_path_space(candidates)
    if path_space == "inspection":
        raw_paths = _dedupe_strings(
            _string_list(raw_group.get("runtime_prim_paths"))
            or _string_list(raw_group.get("inspection_prim_paths"))
            or _string_list(raw_group.get("prim_paths"))
            or _string_list(raw_group.get("source_prim_paths"))
            or _string_list(raw_group.get("source_paths"))
        )
    else:
        raw_paths = _dedupe_strings(
            _string_list(raw_group.get("prim_paths"))
            or _string_list(raw_group.get("source_prim_paths"))
            or _string_list(raw_group.get("source_paths"))
        )
    target_paths: list[str] = []
    for path in raw_paths:
        if path in candidate_paths:
            target_paths.append(path)
        else:
            target_paths.extend(alias_to_target_paths.get(path) or [path])
    return _dedupe_strings(target_paths)


def _source_paths_for_targets(
    paths: list[str],
    candidates: dict[str, Any],
) -> list[str]:
    source_by_path = _source_paths_by_candidate_path(candidates)
    source_paths: list[str] = []
    for path in paths:
        source_paths.extend(source_by_path.get(path) or [path])
    return _dedupe_strings(source_paths)


def _rejection_reason(
    group: dict[str, Any],
    shape_by_path: dict[str, str],
) -> str | None:
    paths = _string_list(group.get("prim_paths"))
    shape_hints = [shape_by_path.get(path, "unknown") for path in paths]
    shape_set = {shape for shape in shape_hints if shape != "unknown"}
    tags = set(_string_list(group.get("material_tags")))
    saturated_or_painted = bool(tags & PAINTED_OR_SATURATED_MATERIAL_TAGS)
    if saturated_or_painted and "slender_bar" in shape_set:
        return structured_finalizer_rejection("slender_bar_metal_family")
    if saturated_or_painted and len(paths) > 3 and len(shape_set) > 1:
        return structured_finalizer_rejection("split_broad_painted_mixed_groups")
    if len(paths) > MATERIAL_ASSIGNMENT_TARGET_PRIM_SOFT_CAP and len(shape_set) > 1:
        return structured_finalizer_rejection("split_large_mixed_groups")
    return None


def _command_targets_for_group(
    group: dict[str, Any],
    *,
    runtime_alias_target_paths: dict[str, list[str]],
    alias_to_target_paths: dict[str, list[str]],
    ambiguous_alias_paths: set[str],
    candidate_paths: set[str],
    candidates_by_path: dict[str, dict[str, Any]],
) -> list[tuple[str, str, list[str]]]:
    path_space = str(group.get("path_space") or "source")
    targets: list[tuple[str, str, str]] = []
    group_canonical_paths = set(_string_list(group.get("prim_paths")))
    if path_space == "inspection":
        command_paths = _string_list(group.get("runtime_prim_paths")) or _string_list(
            group.get("prim_paths")
        )
        for command_path in command_paths:
            if command_path in candidate_paths:
                resolved_paths = [command_path]
            else:
                resolved_paths = alias_to_target_paths.get(command_path) or [
                    command_path
                ]
            targets.extend(
                (resolved_path, "inspection", command_path)
                for resolved_path in resolved_paths
            )
        return _dedupe_command_targets(targets)

    for canonical_path in _string_list(group.get("prim_paths")):
        candidate = candidates_by_path.get(canonical_path, {})
        all_runtime_paths = _candidate_runtime_paths(candidate)
        runtime_space = str(candidate.get("runtime_space") or "")
        original_source_paths = [
            path
            for path in _string_list(candidate.get("original_source_paths"))
            if path != canonical_path
        ]
        if len(all_runtime_paths) > 1:
            runtime_alias_is_ambiguous = any(
                path in ambiguous_alias_paths for path in all_runtime_paths
            )
            if (
                bool(candidate.get("instance_collapsed"))
                and runtime_space == "inspection"
                and not runtime_alias_is_ambiguous
            ):
                # Canonical prototype candidates are not necessarily part of the
                # optimizer correspondence map. Their instance paths are live,
                # distinct inspection targets, so apply every instance while
                # retaining the single canonical candidate for coverage.
                targets.extend(
                    (path, "inspection", canonical_path) for path in all_runtime_paths
                )
                continue
            # A single source candidate can optimize into several runtime fragments.
            # Keep source-space binds atomic. Workbench stores inspection overrides
            # with translated source coverage, so issuing one inspection command per
            # sibling fragment can make later fragments replace earlier ones when
            # they map back to the same source prim. Flattened/prototype canonical
            # paths can be synthetic and absent from the live source session, so use
            # a single non-ambiguous authored source alias when it is the only
            # recorded source alias. If several source aliases collapse under one
            # canonical candidate, keep the canonical logical source bind so one
            # instance alias does not under-cover the collapsed siblings.
            needs_source_fanout = _needs_source_command_fanout(
                canonical_path,
                include_nested_prototypes=runtime_space == "inspection",
            )
            command_path = (
                _source_command_path_for_atomic_bind(
                    canonical_path,
                    original_source_paths,
                    ambiguous_alias_paths=ambiguous_alias_paths,
                    prefer_canonical_when_multiple_aliases=True,
                )
                if needs_source_fanout
                else canonical_path
            )
            targets.append((command_path, "source", canonical_path))
            continue
        needs_source_fanout = _needs_source_command_fanout(
            canonical_path,
            include_nested_prototypes=runtime_space == "inspection",
        )
        shared_runtime_paths = [
            path
            for path in all_runtime_paths
            if len(runtime_alias_target_paths.get(path) or []) > 1
            and set(runtime_alias_target_paths.get(path) or []).issubset(
                group_canonical_paths
            )
        ]
        if runtime_space == "inspection" and shared_runtime_paths:
            # When a shared optimized runtime alias represents exactly the
            # canonical source candidates covered by this material group, one
            # inspection-space command is the only atomic Workbench operation.
            # Issuing source-space commands for the individual aliases would
            # translate back to the same inspection path and make later commands
            # remove earlier ones.
            # Every canonical candidate covered by this shared runtime alias
            # gets its own target tuple, even though they resolve to the same
            # command. `_dedupe_command_targets` merges same-command tuples
            # into one issued command while preserving every canonical path
            # so a failed command can reject all of them, not just the first.
            for runtime_path in shared_runtime_paths:
                targets.append((runtime_path, "inspection", canonical_path))
            continue
        if needs_source_fanout and (
            original_source_paths
            or (runtime_space == "inspection" and all_runtime_paths)
        ):
            runtime_alias_is_ambiguous = any(
                path in ambiguous_alias_paths for path in all_runtime_paths
            )
            use_runtime_fanout = (
                runtime_space == "inspection"
                and bool(all_runtime_paths)
                and not runtime_alias_is_ambiguous
            )
            if use_runtime_fanout:
                fanout_paths = all_runtime_paths
                fanout_space = "inspection"
            else:
                fanout_paths = [
                    _source_command_path_for_atomic_bind(
                        canonical_path,
                        original_source_paths,
                        ambiguous_alias_paths=ambiguous_alias_paths,
                    )
                ]
                fanout_space = "source"
            # Optimized source-space candidates can expose a canonical
            # source/prototype coverage path while the live Workbench session only
            # contains a corresponding authored/original or runtime mesh path. Keep
            # canonical coverage in assignments.json, but issue the command against
            # the path Workbench can resolve. Prefer the single recorded runtime
            # path for optimized inspection-backed candidates because split-part
            # original_source_paths may be pre-optimization aliases that are not
            # live in the session, and send those runtime targets as inspection
            # commands. When that runtime path is an ambiguous alias for multiple
            # canonical candidates, keep the command in source space and prefer a
            # non-ambiguous recorded source alias so Workbench does not translate
            # one runtime override onto siblings.
            targets.extend(
                (path, fanout_space, canonical_path) for path in fanout_paths
            )
            continue
        targets.append((canonical_path, "source", canonical_path))
    return _dedupe_command_targets(targets)


def _source_command_path_for_atomic_bind(
    canonical_path: str,
    original_source_paths: list[str],
    *,
    ambiguous_alias_paths: set[str],
    prefer_canonical_when_multiple_aliases: bool = False,
) -> str:
    source_paths = _dedupe_strings(original_source_paths)
    if prefer_canonical_when_multiple_aliases and len(source_paths) > 1:
        return canonical_path
    for path in source_paths:
        if path not in ambiguous_alias_paths:
            return path
    return canonical_path


def _needs_source_command_fanout(
    path: str,
    *,
    include_nested_prototypes: bool = False,
) -> bool:
    stripped = path.strip("/")
    parts = [part for part in stripped.split("/") if part]
    root = parts[0] if parts else ""
    return (
        root.startswith("Flattened_Prototype")
        or root.startswith("__Prototype_")
        or (include_nested_prototypes and "Prototypes" in parts)
    )


def _dedupe_command_targets(
    targets: list[tuple[str, str, str]],
) -> list[tuple[str, str, list[str]]]:
    """Merge targets issuing the same command, preserving every canonical path.

    Several canonical candidates can resolve to one physical (prim_path,
    command_space) command (e.g. siblings sharing one optimized runtime
    alias). Collapsing to a single canonical path per command would let a
    failed command's rejection cover only the first candidate while the rest
    are silently counted as assigned. Keep the full list so a failure can be
    attributed to every candidate the command was actually responsible for.
    """
    order: list[tuple[str, str]] = []
    canonical_paths_by_key: dict[tuple[str, str], list[str]] = {}
    for prim_path, command_space, canonical_prim_path in targets:
        key = (prim_path, command_space)
        if key not in canonical_paths_by_key:
            order.append(key)
            canonical_paths_by_key[key] = []
        if canonical_prim_path not in canonical_paths_by_key[key]:
            canonical_paths_by_key[key].append(canonical_prim_path)
    return [
        (prim_path, command_space, canonical_paths_by_key[(prim_path, command_space)])
        for prim_path, command_space in order
    ]


def _apply_material_assignment_group(
    *,
    workbench_url: str,
    session_id: str,
    raw_dir: Path,
    group: dict[str, Any],
    group_index: int,
    materials_usd: Path,
    runtime_alias_target_paths: dict[str, list[str]],
    alias_to_target_paths: dict[str, list[str]],
    ambiguous_alias_paths: set[str],
    candidate_paths: set[str],
    candidates_by_path: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    slug = f"{group_index:03d}_{_slug(str(group['family']))}"
    command_targets = _command_targets_for_group(
        group,
        runtime_alias_target_paths=runtime_alias_target_paths,
        alias_to_target_paths=alias_to_target_paths,
        ambiguous_alias_paths=ambiguous_alias_paths,
        candidate_paths=candidate_paths,
        candidates_by_path=candidates_by_path,
    )
    for index, (prim_path, command_space, canonical_prim_paths) in enumerate(
        command_targets,
        start=1,
    ):
        body = {
            "command": "material_override",
            "payload": {
                "prim_path": prim_path,
                "space": command_space,
                "unbind_existing": True,
                "material": {
                    "source": "material_library",
                    "library_path": str(materials_usd),
                    "material_path": group["material_path"],
                    "material_name": group["material_name"],
                },
            },
        }
        body_path = raw_dir / f"material_assignment_{slug}_{index}_body.json"
        body_path.write_text(
            json.dumps(body, indent=2, sort_keys=True), encoding="utf-8"
        )
        # `canonical_prim_paths` may hold more than one entry when several
        # canonical candidates share this exact command (see
        # `_dedupe_command_targets`); a single success/failure applies to all
        # of them, so every candidate the command represents is tracked.
        record: dict[str, Any] = {
            "family": group["family"],
            "prim_path": prim_path,
            "canonical_prim_paths": canonical_prim_paths,
            "space": command_space,
            "material_name": group["material_name"],
            "material_path": group["material_path"],
            "body_path": str(body_path),
        }
        try:
            response = _post_json(
                f"{workbench_url}/sessions/{session_id}/commands",
                body,
            )
        except Exception as exc:  # noqa: BLE001 - one bad command must not sink the run
            record["success"] = False
            record["error"] = str(exc)
            logger.warning(
                "material_override command failed for %s (%s): %s",
                prim_path,
                ", ".join(canonical_prim_paths),
                exc,
            )
            records.append(record)
            continue
        response_path = raw_dir / f"material_assignment_{slug}_{index}_response.json"
        response_path.write_text(
            json.dumps(response, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        record["success"] = True
        record["response_path"] = str(response_path)
        record["response"] = response
        records.append(record)
    return records


def _build_assignments(
    *,
    seed: dict[str, Any],
    packet: dict[str, Any],
    material_groups: list[dict[str, Any]],
    reviewed_groups: list[dict[str, Any]],
    decision_patch: dict[str, Any],
    rejected_groups: list[dict[str, Any]],
    session_id: str,
    source_usd: Path,
    materials_usd: Path,
    final_renders: list[dict[str, Any]],
    reference_images: list[Path],
    reference_files: list[Path],
) -> dict[str, Any]:
    overridden_paths = {
        path
        for group in material_groups
        for path in _string_list(group.get("prim_paths"))
    }
    rejected_paths = {
        path
        for group in rejected_groups
        for path in _string_list(group.get("prim_paths"))
    }
    reviewed_paths = {
        path
        for group in reviewed_groups
        for path in _string_list(group.get("prim_paths"))
        if path not in overridden_paths
    }
    assignments = [dict(group) for group in material_groups]
    for reviewed_group in reviewed_groups:
        paths = [
            path
            for path in _string_list(reviewed_group.get("prim_paths"))
            if path not in overridden_paths
        ]
        if not paths:
            continue
        group = dict(reviewed_group)
        group["prim_paths"] = paths
        assignments.append(group)
    respect_existing_material_bindings = bool(
        packet.get("respect_existing_material_bindings")
    )
    for raw_group in seed.get("assignments", []):
        if not isinstance(raw_group, dict):
            continue
        remaining_paths = [
            path
            for path in _string_list(raw_group.get("prim_paths"))
            if path not in overridden_paths and path not in reviewed_paths
        ]
        if not remaining_paths:
            continue
        if not respect_existing_material_bindings:
            rejected_remaining_paths = [
                path for path in remaining_paths if path in rejected_paths
            ]
            missing_remaining_paths = [
                path for path in remaining_paths if path not in rejected_paths
            ]
            if rejected_remaining_paths:
                group = dict(raw_group)
                group["prim_paths"] = rejected_remaining_paths
                group["coverage_status"] = "rejected_material_assignment"
                group["material_name"] = None
                group["material_path"] = None
                group["rationale"] = (
                    "Visible mesh candidate had a proposed material assignment, "
                    "but the finalizer rejected that assignment. See "
                    "raw/rejected_material_assignments.json for the rejected "
                    "family, paths, and guardrail reason."
                )
                assignments.append(group)
            if missing_remaining_paths:
                group = dict(raw_group)
                group["prim_paths"] = missing_remaining_paths
                group["coverage_status"] = "missing_material_assignment"
                group["material_name"] = None
                group["material_path"] = None
                group["rationale"] = (
                    "Visible mesh candidate had no proposed material-library "
                    "assignment in a clean-slate session. The finalizer treats "
                    "this as unresolved workflow coverage, not as a completed "
                    "material decision."
                )
                assignments.append(group)
            continue
        group = dict(raw_group)
        group["prim_paths"] = remaining_paths
        assignments.append(group)

    coverage = _coverage(assignments, seed)
    visual_quality = _visual_quality(
        decision_patch=decision_patch,
        rejected_groups=rejected_groups,
        final_renders=final_renders,
        reference_images=reference_images,
        reference_files=reference_files,
        material_groups=material_groups,
        coverage=coverage,
        assignments=assignments,
        respect_existing_material_bindings=respect_existing_material_bindings,
    )
    final_review = {
        "issues_found": _string_list(decision_patch.get("final_review_issues_found")),
        "issues_fixed": _string_list(decision_patch.get("final_review_issues_fixed")),
        "unresolved_issues": visual_quality["unresolved_issues"],
        "review_notes": str(
            decision_patch.get("final_review_notes")
            or visual_quality["assessment_notes"]
        ),
    }
    return {
        "schema_version": "content-agents.assignments.v1",
        "session_id": session_id,
        "source_usd": str(source_usd),
        "inspection_usd": seed.get("inspection_usd"),
        "path_space": str(seed.get("path_space") or "source"),
        "library_path": str(materials_usd),
        "per_prim_material_assignment_count": sum(
            len(group["prim_paths"]) for group in material_groups
        ),
        "coverage": coverage,
        "assignments": assignments,
        "final_review": final_review,
        "visual_quality_assessment": visual_quality,
        "generated_by": "content-workflow-cli material finalizer",
    }


def _coverage(
    assignments: list[dict[str, Any]], seed: dict[str, Any]
) -> dict[str, Any]:
    status_counts: dict[str, int] = {
        "material_assignment": 0,
        "preserved_existing": 0,
        "ambiguous_unassigned": 0,
        "missing_material_assignment": 0,
        "rejected_material_assignment": 0,
        "unassigned_visible_candidate": 0,
    }
    decision_paths = []
    for group in assignments:
        status = str(group.get("coverage_status") or "")
        paths = _string_list(group.get("prim_paths"))
        if status in status_counts:
            status_counts[status] += len(paths)
        if status in {
            "material_assignment",
            "preserved_existing",
            "ambiguous_unassigned",
        }:
            decision_paths.extend(paths)
    candidate_count = int(
        seed.get("coverage", {}).get("candidate_visible_prim_count")
        or len(_dedupe_strings(decision_paths))
    )
    missing_count = (
        status_counts["missing_material_assignment"]
        + status_counts["unassigned_visible_candidate"]
    )
    rejected_count = status_counts["rejected_material_assignment"]
    return {
        "candidate_visible_prim_count": candidate_count,
        "material_decision_prim_count": len(_dedupe_strings(decision_paths)),
        "material_assignment_prim_count": status_counts["material_assignment"],
        "preserved_existing_prim_count": status_counts["preserved_existing"],
        "ambiguous_unassigned_prim_count": status_counts["ambiguous_unassigned"],
        "missing_assignment_prim_count": missing_count,
        "rejected_assignment_prim_count": rejected_count,
        "unassigned_visible_prim_count": missing_count + rejected_count,
        "coverage_notes": (
            "Coverage was produced by applying the structured material decision "
            "patch to the canonical material candidate universe. "
            "Missing assignments are visible candidates that received no proposed "
            "library material. Rejected assignments are visible candidates whose "
            "proposed material decisions were discarded by finalizer guardrails. "
            "Neither category is counted as a completed material decision."
        ),
    }


def _visual_quality(
    *,
    decision_patch: dict[str, Any],
    rejected_groups: list[dict[str, Any]],
    final_renders: list[dict[str, Any]],
    reference_images: list[Path],
    reference_files: list[Path],
    material_groups: list[dict[str, Any]],
    coverage: dict[str, Any],
    assignments: list[dict[str, Any]],
    respect_existing_material_bindings: bool,
) -> dict[str, Any]:
    raw_vqa = decision_patch.get("visual_quality_assessment")
    if not isinstance(raw_vqa, dict):
        raw_vqa = {}
    status = str(raw_vqa.get("status") or "fixed")
    if status not in {"pass", "fixed", "unresolved_issues"}:
        status = "fixed"
    unresolved_issues = _issue_description_list(raw_vqa.get("unresolved_issues"))
    if rejected_groups:
        unresolved_issues.extend(
            f"Rejected material decision for {group.get('family')}: {group.get('rejection_reason')}"
            for group in rejected_groups
        )
    ambiguous_count = int(coverage.get("ambiguous_unassigned_prim_count") or 0)
    missing_count = int(coverage.get("missing_assignment_prim_count") or 0)
    rejected_count = int(coverage.get("rejected_assignment_prim_count") or 0)
    if missing_count > 0:
        unresolved_issues.append(
            f"{missing_count} visible material candidate prim(s) have no proposed "
            "material-library assignment."
        )
    if rejected_count > 0:
        unresolved_issues.append(
            f"{rejected_count} visible material candidate prim(s) had proposed "
            "material-library assignments rejected by the finalizer."
        )
    if not respect_existing_material_bindings and ambiguous_count > 0:
        unresolved_issues.append(
            f"{ambiguous_count} visible material candidate prim(s) were left without "
            "explicit library material assignments in a clean-slate session."
        )
    if not respect_existing_material_bindings:
        unresolved_high_salience = _unresolved_high_salience_families(assignments)
        unresolved_issues.extend(
            f"High-salience authoring family remains ambiguous: {family}"
            for family in unresolved_high_salience
        )
    if unresolved_issues:
        status = "unresolved_issues"
    checked_views = [
        str(record.get("image_path"))
        for record in final_renders
        if isinstance(record.get("image_path"), str)
    ]
    return {
        "schema_version": "content-agents.visual-quality-assessment.v1",
        "status": status,
        "checked_views": checked_views,
        "reference_images": [str(path) for path in reference_images],
        "reference_files": [str(path) for path in reference_files],
        "issues_found": _issue_description_list(raw_vqa.get("issues_found")),
        "issues_fixed": _issue_description_list(raw_vqa.get("issues_fixed")),
        "unresolved_issues": unresolved_issues,
        "assessment_notes": str(
            raw_vqa.get("assessment_notes")
            or "Structured material decision patch was applied and verification renders were generated."
        ),
    }


def _merge_post_apply_visual_quality(
    *,
    existing_vqa: dict[str, Any],
    post_apply_vqa: dict[str, Any],
    validator_artifact: Path | None,
) -> dict[str, Any]:
    existing = _normalize_visual_quality(existing_vqa)
    post_apply = _normalize_visual_quality(post_apply_vqa)
    post_apply_unresolved = _string_list(post_apply.get("unresolved_issues"))
    if validator_artifact is not None and not post_apply_unresolved:
        unresolved_issues = []
    else:
        unresolved_issues = _dedupe_strings(
            _string_list(existing.get("unresolved_issues")) + post_apply_unresolved
        )
    issues_found = _dedupe_strings(
        _string_list(existing.get("issues_found"))
        + _string_list(post_apply.get("issues_found"))
    )
    issues_fixed = _dedupe_strings(
        _string_list(existing.get("issues_fixed"))
        + _string_list(post_apply.get("issues_fixed"))
    )
    status = str(post_apply.get("status") or existing.get("status") or "fixed")
    if unresolved_issues:
        status = "unresolved_issues"
    elif status not in {"pass", "fixed"}:
        status = "fixed"
    existing_notes = str(existing.get("assessment_notes") or "").strip()
    post_notes = str(post_apply.get("assessment_notes") or "").strip()
    notes = []
    if post_notes:
        notes.append(f"Post-apply validation: {post_notes}")
    if existing_notes and existing_notes != post_notes:
        notes.append(f"Planner/finalizer context: {existing_notes}")
    if validator_artifact is not None:
        notes.append(f"Validation artifact: {validator_artifact}")
    return {
        "schema_version": "content-agents.visual-quality-assessment.v1",
        "status": status,
        "checked_views": _dedupe_strings(
            _string_list(existing.get("checked_views"))
            + _string_list(post_apply.get("checked_views"))
        ),
        "reference_images": _dedupe_strings(
            _string_list(existing.get("reference_images"))
            + _string_list(post_apply.get("reference_images"))
        ),
        "reference_files": _dedupe_strings(
            _string_list(existing.get("reference_files"))
            + _string_list(post_apply.get("reference_files"))
        ),
        "issues_found": issues_found,
        "issues_fixed": issues_fixed,
        "unresolved_issues": unresolved_issues,
        "assessment_notes": " ".join(notes).strip()
        or "Post-apply visual validation completed.",
    }


def _normalize_visual_quality(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    status = str(value.get("status") or "fixed")
    if status not in {"pass", "fixed", "unresolved_issues"}:
        status = "fixed"
    unresolved_issues = _string_list(value.get("unresolved_issues"))
    if unresolved_issues:
        status = "unresolved_issues"
    return {
        "status": status,
        "checked_views": _string_list(value.get("checked_views")),
        "reference_images": _string_list(value.get("reference_images")),
        "reference_files": _string_list(value.get("reference_files")),
        "issues_found": _string_list(value.get("issues_found")),
        "issues_fixed": _string_list(value.get("issues_fixed")),
        "unresolved_issues": unresolved_issues,
        "assessment_notes": str(value.get("assessment_notes") or ""),
    }


def _unresolved_high_salience_families(assignments: list[dict[str, Any]]) -> list[str]:
    high_salience = {
        "hand_gripper",
        "foot_ankle",
        "head_shell",
        "waist_pelvis",
        "torso_shell",
        "logo_marking",
    }
    unresolved_statuses = {
        "ambiguous_unassigned",
        "missing_material_assignment",
        "rejected_material_assignment",
    }
    unresolved = []
    for group in assignments:
        if group.get("coverage_status") not in unresolved_statuses:
            continue
        semantic_hints = group.get("semantic_hints")
        if not isinstance(semantic_hints, dict):
            continue
        matches = sorted(str(hint) for hint in semantic_hints if hint in high_salience)
        if not matches:
            continue
        family = str(group.get("authoring_family") or group.get("family") or matches[0])
        unresolved.append(family)
    return _dedupe_strings(unresolved)


def _build_counts(
    *,
    run_dir: Path,
    packet: dict[str, Any],
    material_groups: list[dict[str, Any]],
    applied_records: list[dict[str, Any]],
    final_renders: list[dict[str, Any]],
    assignments: dict[str, Any],
) -> dict[str, Any]:
    preflight_counts = packet.get("operation_counts_so_far")
    if not isinstance(preflight_counts, dict):
        preflight_counts = {}
    preflight_render_count = int(preflight_counts.get("render_calls_total") or 0)
    preflight_render_downloads = int(
        preflight_counts.get("render_artifact_downloads") or 0
    )
    material_assignment_count = len(applied_records)
    final_render_artifact_count = len(final_renders)
    final_render_call_count = sum(
        _render_call_count(record) for record in final_renders
    )
    final_render_downloads = sum(
        _render_artifact_download_count(record) for record in final_renders
    )
    grounding_counts = grounding_operation_counts(run_dir)
    grounding_api_calls = int(grounding_counts.get("workbench_api_calls") or 0)
    grounding_render_calls = int(grounding_counts.get("render_calls") or 0)
    grounding_render_downloads = int(
        grounding_counts.get("render_artifact_downloads") or 0
    )
    grounding_pick_calls = int(grounding_counts.get("pick_calls") or 0)
    api_total = (
        int(preflight_counts.get("workbench_api_calls_total") or 0)
        + material_assignment_count
        + final_render_call_count
        + final_render_downloads
        + grounding_api_calls
    )
    coverage = assignments.get("coverage", {})
    visual_quality = assignments.get("visual_quality_assessment", {})
    final_review = assignments.get("final_review", {})
    return {
        "schema_version": "content-agents.api-operation-counts.v1",
        "api_operation_count_total": api_total,
        "count_basis": (
            "Includes preflight Workbench docs/session/snapshot/renders, material "
            "override commands, final render requests, final render artifact "
            "downloads, and post-VQA grounding diagnostics."
        ),
        "render_count_total": (
            preflight_render_count + final_render_call_count + grounding_render_calls
        ),
        "evidence_renders": preflight_render_count,
        "evidence_render_downloads": preflight_render_downloads,
        "pick_calls": grounding_pick_calls,
        "material_override_commands": material_assignment_count,
        "material_assignment_target_prims": sum(
            len(group["prim_paths"]) for group in material_groups
        ),
        "final_renders": final_render_artifact_count,
        "final_render_calls": final_render_call_count,
        "final_render_downloads": final_render_downloads,
        "render_artifact_downloads": (
            preflight_render_downloads
            + final_render_downloads
            + grounding_render_downloads
        ),
        "grounding_diagnostic_runs": int(grounding_counts.get("runs") or 0),
        "grounding_pick_calls": grounding_pick_calls,
        "grounding_render_calls": grounding_render_calls,
        "accepted_final_render_images": final_render_artifact_count,
        "coverage_candidate_visible_prims": coverage.get(
            "candidate_visible_prim_count"
        ),
        "coverage_material_decision_prims": coverage.get(
            "material_decision_prim_count"
        ),
        "coverage_unassigned_visible_prims": coverage.get(
            "unassigned_visible_prim_count"
        ),
        "coverage_missing_assignment_prims": coverage.get(
            "missing_assignment_prim_count"
        ),
        "coverage_rejected_assignment_prims": coverage.get(
            "rejected_assignment_prim_count"
        ),
        "final_review_issues_found": _count_items(final_review.get("issues_found")),
        "final_review_issues_fixed": _count_items(final_review.get("issues_fixed")),
        "visual_quality_issues_found": _count_items(visual_quality.get("issues_found")),
        "visual_quality_issues_fixed": _count_items(visual_quality.get("issues_fixed")),
    }


def _render_artifact_download_count(record: dict[str, Any]) -> int:
    value = record.get("artifact_download_count")
    if isinstance(value, int):
        return value
    count = 0
    if record.get("image_path"):
        count += 1
    if record.get("camera_json_path"):
        count += 1
    return count


def _render_call_count(record: dict[str, Any]) -> int:
    value = record.get("render_call_count")
    if isinstance(value, int):
        return value
    return 1


def _build_summary(
    *,
    assignments: dict[str, Any],
    counts: dict[str, Any],
    final_renders: list[dict[str, Any]],
    decision_patch: dict[str, Any],
) -> str:
    lines = [
        "# Material Assignment Summary",
        "",
        "Generated from a structured material decision patch and Workbench verification renders.",
        "",
        "## Coverage",
        "",
        f"- Candidate visible prims: `{assignments['coverage']['candidate_visible_prim_count']}`",
        f"- Material decisions: `{assignments['coverage']['material_decision_prim_count']}`",
        f"- Material-assigned prims: `{assignments['coverage']['material_assignment_prim_count']}`",
        f"- Preserved existing prims: `{assignments['coverage']['preserved_existing_prim_count']}`",
        f"- Ambiguous/unassigned prims: `{assignments['coverage']['ambiguous_unassigned_prim_count']}`",
        f"- Missing proposed assignments: `{assignments['coverage'].get('missing_assignment_prim_count', 0)}`",
        f"- Rejected proposed assignments: `{assignments['coverage'].get('rejected_assignment_prim_count', 0)}`",
        f"- Total uncovered visible prims: `{assignments['coverage'].get('unassigned_visible_prim_count', 0)}`",
        "",
        "## Material Map",
        "",
        "| Family | Status | Material | Prim Count | Rationale |",
        "|---|---|---|---:|---|",
    ]
    for group in assignments["assignments"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(str(group.get("family") or "")),
                    _md(str(group.get("coverage_status") or "")),
                    _md(str(group.get("material_name") or "n/a")),
                    str(len(_string_list(group.get("prim_paths")))),
                    _md(str(group.get("rationale") or "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Visual Quality",
            "",
            f"- Status: `{assignments['visual_quality_assessment']['status']}`",
            f"- Notes: {assignments['visual_quality_assessment']['assessment_notes']}",
            "",
            "## Final Renders",
            "",
        ]
    )
    for record in final_renders:
        lines.append(f"- `{record.get('image_path')}`")
    lines.extend(
        [
            "",
            "## Operation Counts",
            "",
            f"- API operations: `{counts['api_operation_count_total']}`",
            f"- Render calls: `{counts['render_count_total']}`",
            f"- Material override commands: `{counts['material_override_commands']}`",
            f"- Pick calls: `{counts['pick_calls']}`",
            "",
            "## Decision Notes",
            "",
            str(decision_patch.get("final_review_notes") or ""),
            "",
        ]
    )
    return "\n".join(lines)


def _append_finalize_trace(
    run_dir: Path,
    paths: dict[str, Path],
    material_groups: list[dict[str, Any]],
    final_renders: list[dict[str, Any]],
) -> None:
    append_jsonl(
        run_dir / "trace" / "events.jsonl",
        {
            "schema_version": "content-agents.trace.v1",
            "time": utc_now(),
            "event_type": "verification",
            "phase": "material_finalize",
            "summary": (
                "Applied structured material decisions, rendered final verification "
                "views, and wrote standard material assignment artifacts."
            ),
            "artifacts": [
                *(str(path) for path in paths.values()),
                *[
                    str(record.get("image_path"))
                    for record in final_renders
                    if isinstance(record, dict) and record.get("image_path")
                ],
            ],
            "data": {
                "material_assignment_groups": len(material_groups),
                "material_assignment_target_prims": sum(
                    len(group["prim_paths"]) for group in material_groups
                ),
                "final_renders": len(final_renders),
                "api_calls": [
                    "POST /sessions/{session_id}/commands material_override",
                    "POST /sessions/{session_id}/render",
                    "GET /sessions/{session_id}/renders/{filename}",
                ],
            },
        },
    )


def _load_json(path: Path, *, default: Any = _MISSING) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        if default is not _MISSING:
            return default
        raise RuntimeError(f"Required JSON artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        if default is not _MISSING:
            return default
        raise RuntimeError(f"Required JSON artifact is malformed: {path}") from exc


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _issue_description_list(value: Any) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if not isinstance(value, list):
        return []
    descriptions: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            descriptions.append(item)
        elif isinstance(item, dict):
            description = (
                item.get("description")
                or item.get("issue")
                or item.get("summary")
                or item.get("message")
            )
            if description:
                descriptions.append(str(description))
    return descriptions


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _count_items(value: Any) -> int:
    if isinstance(value, list | tuple | set):
        return len(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return 1 if value else 0
    return 0


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "material"


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
