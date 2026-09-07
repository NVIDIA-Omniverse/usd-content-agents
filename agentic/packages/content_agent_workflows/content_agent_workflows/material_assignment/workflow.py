# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain-owned execution and goal policy for Material assignment.

The launcher may supply model and scene adapters, but it must not decide how a
material generation is applied or when the VQA goal should continue.  This
module therefore owns the deterministic scene transaction and the bounded VQA
state transitions used by every Material-assignment entry point.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    prepare_writable_directory,
    read_contained_artifact,
)

MAX_MATERIAL_APPLY_ASSIGNMENT_GROUPS = 256
MAX_MATERIAL_APPLY_TARGET_PATHS = 4096
# Preserve the already-published artifact contract while moving its owner from
# the launcher into the domain package.
MATERIAL_EXECUTION_SCHEMA_VERSION = "content-agents.wrapper-material-execution.v1"


class MaterialSceneSession(Protocol):
    """Scene operations required by one deterministic material generation."""

    def open(self, scene: Path, *, force_reload: bool = False) -> dict[str, Any]: ...

    def run_json(
        self,
        args: list[str],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class MaterialGenerationRequest:
    """Complete immutable inputs for applying and rendering one generation."""

    run_dir: Path
    staged_source: Path
    staged_library: Path
    decision_patch_path: Path
    output_path: Path
    final_render_dir: Path
    decision_patch: Mapping[str, Any]
    iteration: int
    timeout_seconds: float
    respect_existing_material_bindings: bool
    source_focus_prim_path: str
    inspection_focus_prim_path: str | None = None
    inspection_usd_path: Path | None = None
    inspection_usd_sha256: str | None = None
    appearance_clear_report_schema_version: str = (
        "content-agents.appearance-clear-report.v1"
    )
    appearance_clear_capability: str = "appearance.clear.v1"
    stage_gprim_count: int = 0


@dataclass(frozen=True, slots=True)
class MaterialGenerationResult:
    """Digest-bound result of one deterministic material generation."""

    artifact_path: Path
    payload: dict[str, Any]


def _assignment_target_paths(
    group: Mapping[str, Any],
    *,
    path_key: str,
) -> list[str]:
    raw_paths = group.get(path_key)
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(f"Material assignment requires non-empty {path_key}")
    paths = [str(path) for path in raw_paths]
    if any(not path.startswith("/") or path == "/" for path in paths):
        raise ValueError(f"Material assignment contains an invalid {path_key}")
    return paths


def partition_material_assignments(
    assignment_groups: Sequence[Mapping[str, Any]],
    *,
    path_key: str,
    max_assignment_groups: int = MAX_MATERIAL_APPLY_ASSIGNMENT_GROUPS,
    max_target_paths: int = MAX_MATERIAL_APPLY_TARGET_PATHS,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Partition a complete plan by both daemon transaction limits.

    An individual semantic group may exceed the target-path limit, so the
    selected execution-path list is split without changing its material
    identity.  Batches preserve input order and exact target coverage.
    """

    if max_assignment_groups <= 0 or max_target_paths <= 0:
        raise ValueError("Material apply limits must be positive")
    batches: list[tuple[dict[str, Any], ...]] = []
    current: list[dict[str, Any]] = []
    current_target_count = 0

    def flush() -> None:
        nonlocal current, current_target_count
        if current:
            batches.append(tuple(current))
            current = []
            current_target_count = 0

    for raw_group in assignment_groups:
        if not isinstance(raw_group, Mapping):
            raise ValueError("Material assignment groups must be objects")
        paths = _assignment_target_paths(raw_group, path_key=path_key)
        offset = 0
        while offset < len(paths):
            if (
                len(current) >= max_assignment_groups
                or current_target_count >= max_target_paths
            ):
                flush()
            available_targets = max_target_paths - current_target_count
            path_slice = paths[offset : offset + available_targets]
            split_group = dict(raw_group)
            split_group[path_key] = path_slice
            current.append(split_group)
            current_target_count += len(path_slice)
            offset += len(path_slice)
    flush()
    return tuple(batches)


def _valid_appearance_clear_response(payload: Mapping[str, Any]) -> bool:
    """Validate the scene adapter's clean-slate receipt."""

    if (
        payload.get("schema_version") != "1"
        or payload.get("ok") is not True
        or payload.get("command") not in {"appearance-clear", "appearance.clear"}
    ):
        return False
    summary = payload.get("summary")
    data = payload.get("data")
    audit = data.get("audit") if isinstance(data, Mapping) else None
    counts = audit.get("counts") if isinstance(audit, Mapping) else None
    return bool(
        isinstance(summary, Mapping)
        and summary.get("clear") is True
        and isinstance(audit, Mapping)
        and audit.get("clear") is True
        and audit.get("overlay_active") is True
        and isinstance(counts, Mapping)
        and counts
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value == 0
            for value in counts.values()
        )
    )


def _render_verification(
    *,
    session: MaterialSceneSession,
    request: MaterialGenerationRequest,
    focus: str,
) -> dict[str, Any]:
    """Render complementary upper/lower beauty and segmentation pairs."""

    combined_results: list[dict[str, Any]] = []
    combined_segmentation_results: list[dict[str, Any]] = []
    combined_artifacts: list[dict[str, Any]] = []
    combined_summary: dict[str, Any] = {}
    for view_name, elevation in (("upper", "55"), ("lower", "-35")):
        response = session.run_json(
            [
                "render",
                "--photoreal",
                "--seg",
                "--focus",
                focus,
                "--orbit",
                "2",
                "--elevation",
                elevation,
                "--res",
                "640x480",
                "--mode",
                "quality",
                "--output",
                str(
                    request.final_render_dir
                    / f"verification_{request.iteration}"
                    / view_name
                ),
            ],
            timeout_seconds=request.timeout_seconds,
        )
        summary = response.get("summary")
        data = response.get("data")
        results = data.get("results") if isinstance(data, Mapping) else None
        artifacts = response.get("artifacts")
        segmentation_artifacts = (
            [
                item
                for item in artifacts
                if isinstance(item, dict)
                and str(item.get("label") or "").startswith("segmentation:")
                and isinstance(item.get("path"), str)
            ]
            if isinstance(artifacts, list)
            else []
        )
        if (
            response.get("command") != "render"
            or response.get("ok") is not True
            or not isinstance(summary, dict)
            or not isinstance(results, list)
            or len(results) != 2
            or any(not isinstance(item, dict) for item in results)
            or len(segmentation_artifacts) != 2
        ):
            raise ValueError(
                "Material verification render omitted a beauty/segmentation pair"
            )
        typed_results = [item for item in results if isinstance(item, dict)]
        combined_results.extend(typed_results)
        combined_artifacts.extend(item for item in artifacts if isinstance(item, dict))
        combined_segmentation_results.extend(
            {
                **result,
                "path": str(segmentation["path"]),
                "active_aov": "segmentation",
            }
            for result, segmentation in zip(
                typed_results,
                segmentation_artifacts,
                strict=True,
            )
        )
        if not combined_summary:
            combined_summary = dict(summary)
    combined_summary.update({"orbit": 4, "verification_elevations": [55, -35]})
    return {
        "schema_version": "1",
        "ok": True,
        "command": "render",
        "summary": combined_summary,
        "data": {
            "results": combined_results,
            "segmentation_results": combined_segmentation_results,
        },
        "artifacts": combined_artifacts,
    }


def execute_material_generation(
    *,
    session: MaterialSceneSession,
    request: MaterialGenerationRequest,
) -> MaterialGenerationResult:
    """Apply, persist, audit, and render one complete material generation.

    Model adapters never receive scene tools in this path.  Every generation
    starts from its digest-bound execution stage, and every successful call
    produces a reopened audit plus the complete VQA render set.
    """

    started = time.monotonic()
    run_dir = request.run_dir.resolve(strict=True)
    prepare_writable_directory(request.output_path.parent)
    prepare_writable_directory(request.final_render_dir)
    patch = dict(request.decision_patch)
    assignment_groups = patch.get("material_assignments")
    if not isinstance(assignment_groups, list):
        raise ValueError("Material decision patch omitted material_assignments")
    candidate_count = patch.get("candidate_count")
    if not isinstance(candidate_count, int) or isinstance(candidate_count, bool):
        raise ValueError("Material decision patch omitted candidate_count")
    if (
        candidate_count == 0
        and request.stage_gprim_count > 0
        and not request.respect_existing_material_bindings
    ):
        raise ValueError(
            "Material candidate filters excluded all stage geometry; refusing "
            "stage-wide appearance mutation"
        )

    use_inspection_stage = (
        request.inspection_usd_path is not None
        and not request.respect_existing_material_bindings
    )
    if use_inspection_stage:
        if (
            request.inspection_usd_sha256 is None
            or file_sha256(request.inspection_usd_path) != request.inspection_usd_sha256
        ):
            raise ValueError("Material generation requires the trusted inspection USD")
        execution_usd = request.inspection_usd_path
        path_key = "runtime_prim_paths"
        focus = request.inspection_focus_prim_path
    else:
        execution_usd = request.staged_source
        path_key = "prim_paths"
        focus = request.source_focus_prim_path
    if not focus or not focus.startswith("/"):
        raise ValueError("Material generation requires an absolute focus prim path")

    timings: dict[str, float] = {}

    def timed(name: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        operation_started = time.monotonic()
        response = operation()
        timings[name] = round(time.monotonic() - operation_started, 6)
        return response

    timed(
        "open_inspection" if use_inspection_stage else "open_source",
        lambda: session.open(execution_usd, force_reload=True),
    )
    clear_response: dict[str, Any] | None = None
    clear_audit: dict[str, Any] | None = None
    if not request.respect_existing_material_bindings:
        clear_response = timed(
            "appearance_clear",
            lambda: session.run_json(
                ["appearance", "clear"],
                timeout_seconds=request.timeout_seconds,
            ),
        )
        if not _valid_appearance_clear_response(clear_response):
            raise ValueError("Appearance clear returned invalid evidence")
        clear_audit = timed(
            "appearance_audit",
            lambda: session.run_json(
                ["appearance", "audit"],
                timeout_seconds=request.timeout_seconds,
            ),
        )
        source_sha256 = file_sha256(request.staged_source)
        atomic_write_json(
            run_dir / "raw" / "appearance_clear_report.json",
            {
                "schema_version": request.appearance_clear_report_schema_version,
                "capability": request.appearance_clear_capability,
                "status": "pass",
                "source_usd_path": str(request.staged_source),
                "source_sha256_before": source_sha256,
                "source_sha256_after": source_sha256,
                "source_unchanged": True,
                "cli_response": clear_response,
                "post_clear_audit": clear_audit,
                "derived_by_wrapper": True,
                "workflow_owner": "content_agent_workflows.material_assignment",
            },
            within=run_dir,
        )

    checkpoint_name = f"material-before-apply-{request.iteration}"
    timed(
        "checkpoint",
        lambda: session.run_json(
            ["checkpoint", "save", checkpoint_name, "--full"],
            timeout_seconds=request.timeout_seconds,
        ),
    )
    batches = partition_material_assignments(
        assignment_groups,
        path_key=path_key,
    )
    apply_response: dict[str, Any] | None = None
    if batches:
        batch_dir = prepare_writable_directory(
            run_dir / "raw" / "material_apply_batches"
        )
        apply_started = time.monotonic()
        apply_responses: list[dict[str, Any]] = []
        try:
            for batch_index, batch in enumerate(batches, start=1):
                batch_path = (
                    request.decision_patch_path
                    if len(batches) == 1
                    else batch_dir
                    / f"iteration_{request.iteration}_batch_{batch_index}.json"
                )
                if len(batches) != 1:
                    atomic_write_json(
                        batch_path,
                        {**patch, "material_assignments": list(batch)},
                        within=run_dir,
                    )
                response = timed(
                    (
                        "material_apply"
                        if len(batches) == 1
                        else f"material_apply_batch_{batch_index}"
                    ),
                    lambda batch_path=batch_path: session.run_json(
                        [
                            "material-apply",
                            str(batch_path),
                            "--library",
                            str(request.staged_library),
                            "--path-key",
                            path_key,
                        ],
                        timeout_seconds=request.timeout_seconds,
                    ),
                )
                if response.get("ok") is not True:
                    raise ValueError(f"Material apply batch {batch_index} failed")
                apply_responses.append(response)
        except Exception as apply_error:
            try:
                restore_response = session.run_json(
                    ["checkpoint", "load", checkpoint_name],
                    timeout_seconds=request.timeout_seconds,
                )
                if restore_response.get("ok") is not True:
                    raise ValueError("Material apply checkpoint restore failed")
            except Exception as restore_error:
                try:
                    session.open(execution_usd, force_reload=True)
                except Exception as reopen_error:
                    raise ExceptionGroup(
                        "Material apply failed and the session could not be restored",
                        [apply_error, restore_error, reopen_error],
                    ) from apply_error
            raise
        if len(apply_responses) == 1:
            apply_response = apply_responses[0]
        else:
            timings["material_apply"] = round(
                time.monotonic() - apply_started,
                6,
            )
            apply_response = {
                "schema_version": "1",
                "ok": True,
                "command": "material.apply.batched",
                "summary": {
                    "assignment_groups": len(assignment_groups),
                    "batch_count": len(apply_responses),
                },
                "data": {"batch_responses": apply_responses},
            }

    live_audit = timed(
        "live_material_audit",
        lambda: session.run_json(
            ["material", "audit", "--effective", "--include-subsets"],
            timeout_seconds=request.timeout_seconds,
        ),
    )
    save_response = timed(
        "save_output",
        lambda: session.run_json(
            ["save", str(request.output_path), "--flatten"],
            timeout_seconds=request.timeout_seconds,
        ),
    )
    timed(
        "reopen_output",
        lambda: session.open(request.output_path, force_reload=True),
    )
    reopened_audit = timed(
        "reopened_material_audit",
        lambda: session.run_json(
            ["material", "audit", "--effective", "--include-subsets"],
            timeout_seconds=request.timeout_seconds,
        ),
    )

    if candidate_count == 0:
        verification_response = turntable_response = None
    else:
        verification_response = timed(
            "verification_render",
            lambda: _render_verification(
                session=session,
                request=request,
                focus=focus,
            ),
        )
        turntable_response = timed(
            "turntable_render",
            lambda: session.run_json(
                [
                    "render",
                    "--photoreal",
                    "--focus",
                    focus,
                    "--orbit",
                    "24",
                    "--elevation",
                    "20",
                    "--res",
                    "640x480",
                    "--mode",
                    "quality",
                    "--output",
                    str(request.final_render_dir / "final_turntable"),
                ],
                timeout_seconds=request.timeout_seconds,
            ),
        )

    output_artifact = read_contained_artifact(run_dir, request.output_path)
    payload = {
        "schema_version": MATERIAL_EXECUTION_SCHEMA_VERSION,
        "iteration": request.iteration,
        "status": "pass",
        "candidate_count": candidate_count,
        "assignment_group_count": len(assignment_groups),
        "apply_batch_count": len(batches),
        "path_key": path_key,
        "execution_usd_path": str(execution_usd),
        "focus_prim_path": focus,
        "output_usd": {
            "path": str(request.output_path),
            "sha256": output_artifact.sha256,
            "size_bytes": output_artifact.size_bytes,
        },
        "responses": {
            "appearance_clear": clear_response,
            "appearance_audit": clear_audit,
            "material_apply": apply_response,
            "live_material_audit": live_audit,
            "save": save_response,
            "reopened_material_audit": reopened_audit,
            "verification_render": verification_response,
            "turntable_render": turntable_response,
        },
        "timings_seconds": timings,
        "duration_seconds": round(time.monotonic() - started, 6),
    }
    artifact_path = (
        run_dir / "raw" / f"wrapper_material_execution_{request.iteration}.json"
    )
    atomic_write_json(artifact_path, payload, within=run_dir)
    return MaterialGenerationResult(artifact_path=artifact_path, payload=payload)


MaterialGoalAction = Literal["continue", "succeed", "fail"]


@dataclass(frozen=True, slots=True)
class MaterialGoalDecision:
    """One domain-owned transition for the VQA/repair goal."""

    action: MaterialGoalAction
    status: str
    reason: str
    returncode: int


@dataclass(frozen=True, slots=True)
class MaterialVqaGoal:
    """Bounded quality goal shared by CLI, service, and composed workflows."""

    max_iterations: int
    delegated_review_required: bool
    verified_scene_session: bool

    @property
    def repair_iterations(self) -> range:
        """Return the only iterations authorized for surgical repair."""

        return range(2, self.max_iterations + 1)

    def evaluate_initial(
        self,
        *,
        candidate_count: int,
        satisfied: bool,
        systematic_unfixable: bool,
        reviewed_render_signature: str | None,
        current_render_signature: str | None,
    ) -> MaterialGoalDecision:
        """Decide whether initial evidence completes or enters the repair loop."""

        if candidate_count == 0:
            return MaterialGoalDecision(
                "succeed",
                "satisfied_no_candidates",
                "canonical_material_candidate_universe_is_empty",
                0,
            )
        if self.delegated_review_required and not self.verified_scene_session:
            return MaterialGoalDecision(
                "fail",
                (
                    "skipped_no_session"
                    if systematic_unfixable
                    else "delegated_vision_review_not_run"
                ),
                "The configured independent delegated VLM review was not run because "
                "the workflow has no verified preflight usd-cli session.",
                2,
            )
        delegated_review_matches = bool(
            self.delegated_review_required
            and reviewed_render_signature
            and reviewed_render_signature == current_render_signature
        )
        if satisfied and delegated_review_matches:
            return MaterialGoalDecision(
                "succeed",
                "delegated_vision_verified",
                "The initial delegated post-apply VQA review already covers the "
                "current final renders.",
                0,
            )
        if satisfied and not self.delegated_review_required:
            return MaterialGoalDecision(
                "succeed",
                "satisfied_initial",
                "initial_canonical_artifacts_satisfied_vqa_gate",
                0,
            )
        if systematic_unfixable and not self.delegated_review_required:
            return MaterialGoalDecision(
                "fail",
                "systematic_unfixable_initial",
                "VQA refinement skipped because all initial remaining active issues "
                "are recorded as material-library, scene inspection/picking, or "
                "prim-granularity limitations.",
                2,
            )
        if self.max_iterations <= 1:
            return MaterialGoalDecision(
                "fail",
                (
                    "delegated_vision_review_not_run"
                    if satisfied
                    else "max_iterations_reached"
                ),
                (
                    "The initial canonical VQA gate was satisfied, but the configured "
                    "independent delegated VLM review requires an additional iteration."
                    if satisfied
                    else "VQA issues remain after the initial review, but no additional "
                    "refinement iterations are configured."
                ),
                2,
            )
        return MaterialGoalDecision(
            "continue",
            ("delegated_vision_review_required" if satisfied else "repair_required"),
            "The current generation requires a bounded surgical review or repair.",
            0,
        )

    def evaluate_iteration(
        self,
        *,
        iteration: int,
        returncode: int,
        satisfied: bool,
        systematic_unfixable: bool,
        previous_signature: object,
        next_signature: object,
        delegated_review_covers_final_renders: bool,
    ) -> MaterialGoalDecision:
        """Decide the next state after one fully reviewed repair generation."""

        if returncode != 0:
            return MaterialGoalDecision(
                "fail",
                "child_failed",
                f"VQA refinement iteration {iteration} exited with return code "
                f"{returncode}.",
                returncode,
            )
        if not delegated_review_covers_final_renders:
            if iteration >= self.max_iterations:
                return MaterialGoalDecision(
                    "fail",
                    "delegated_post_finalize_review_not_run",
                    f"VQA refinement iteration {iteration} changed canonical final "
                    "renders, but no iteration remains for the delegated VLM to "
                    "review those pixels.",
                    2,
                )
            return MaterialGoalDecision(
                "continue",
                "delegated_post_finalize_review_required",
                "The next iteration must independently review the changed final "
                "render generation.",
                0,
            )
        if satisfied:
            return MaterialGoalDecision(
                "succeed",
                "satisfied",
                f"VQA refinement iteration {iteration} satisfied the VQA gate.",
                0,
            )
        if systematic_unfixable:
            return MaterialGoalDecision(
                "fail",
                "systematic_unfixable",
                f"VQA refinement stopped after iteration {iteration}; all remaining "
                "active issues are recorded as material-library, scene "
                "inspection/picking, or prim-granularity limitations.",
                2,
            )
        if next_signature == previous_signature:
            return MaterialGoalDecision(
                "fail",
                "converged_unresolved",
                f"VQA refinement converged after iteration {iteration}; canonical "
                "issue signature did not change.",
                2,
            )
        if iteration >= self.max_iterations:
            return MaterialGoalDecision(
                "fail",
                "max_iterations_reached",
                f"VQA refinement reached the configured maximum of "
                f"{self.max_iterations} iterations with unresolved issues still "
                "present.",
                2,
            )
        return MaterialGoalDecision(
            "continue",
            "repair_required",
            "The updated generation still has a repairable VQA issue.",
            0,
        )
