# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply-phase prompt contract for the collider pin and penetration limit.

The tuning loop is not the only place that rewrites the decision patch: the
visual-validation refinement loop does too, and it hits the same degenerate
optimum. Coarsening the collider makes the ground-penetration check pass by
removing the body's ability to roll.

Observed on a RoboCasa apple drop configured with
``--revalidation-max-penetration-m 0.05``: the apply phase authored
``convexHull``, then refinement rewrote it to ``convexDecomposition`` and then
``boundingSphere``, while the child reported fighting a ``0.005 m`` limit the
run had already raised to ``0.05``.
"""

import os
from pathlib import Path

from content_workflow_cli.prompts import (
    build_physics_apply_prompt,
    build_physics_post_tuning_assessment_prompt,
    build_physics_visual_refinement_prompt,
)


def _apply_prompt(tmp_path: Path, **overrides: object) -> str:
    kwargs: dict[str, object] = {
        "repo_root": tmp_path,
        "run_dir": tmp_path / "run",
        "usd_path": tmp_path / "asset.usd",
    }
    kwargs.update(overrides)
    return build_physics_apply_prompt(**kwargs)  # type: ignore[arg-type]


def _refinement_prompt(tmp_path: Path, **overrides: object) -> str:
    kwargs: dict[str, object] = {
        "repo_root": tmp_path,
        "run_dir": tmp_path / "run",
        "usd_path": tmp_path / "asset.usd",
        "iteration": 1,
        "max_iterations": 3,
        "decision_patch_path": tmp_path / "run" / "raw" / "patch.json",
        "validation_evidence_path": tmp_path / "run" / "validation_evidence.json",
        "runtime_report_path": tmp_path / "run" / "runtime.json",
        "rendered_frames": ["frame_0.png"],
    }
    kwargs.update(overrides)
    return build_physics_visual_refinement_prompt(**kwargs)  # type: ignore[arg-type]


def test_apply_prompt_states_the_configured_penetration_limit(tmp_path: Path) -> None:
    """The apply-phase child evaluates raw simulation facts in the workflow, so an unstated
    override leaves it fighting the 0.005 m default the run already raised."""

    prompt = _apply_prompt(tmp_path, validation_max_penetration_m=0.05)

    assert "max_ground_penetration_m" in prompt
    assert "0.05" in prompt


def test_apply_prompt_marks_the_collision_approximation_required(
    tmp_path: Path,
) -> None:
    """`collision_approximation_default` read as licence to coarsen the collider."""

    prompt = _apply_prompt(tmp_path, collision_approximation="convexHull")

    assert "collision_approximation_required" in prompt


def test_apply_prompt_selects_stable_target_ids_instead_of_paths(
    tmp_path: Path,
) -> None:
    prompt = _apply_prompt(tmp_path)

    assert '"collider_target_ids"' in prompt
    assert '"collider_paths"' not in prompt
    assert '"mass_authoring_path"' not in prompt
    assert "Never type or reconstruct USD prim paths" in prompt


def test_apply_prompt_provides_the_first_path_for_generated_json(
    tmp_path: Path,
) -> None:
    prompt = _apply_prompt(tmp_path)

    assert "do not try `apply_patch` for generated JSON" in prompt
    if os.name == "nt":
        assert "content-workflow-cli artifact write-json" in prompt
        assert "Run the writer as its own command" in prompt
        assert "rejects an existing destination" in prompt
        assert "Set-Content" not in prompt
        assert "Move-Item" not in prompt
        assert "ConvertFrom-Json | Out-Null" not in prompt
        assert "`jq -nS`" not in prompt
    else:
        assert "same-directory `.tmp` file" in prompt
        assert "`jq -nS`" in prompt
        assert "`jq -e . <path> >/dev/null`" in prompt


def test_apply_prompt_documents_structured_mass_scale_warning(tmp_path: Path) -> None:
    prompt = _apply_prompt(tmp_path)

    assert '"code": "mass_scale_suspicious"' in prompt
    assert '"severity":' in prompt
    assert '"warning"' in prompt
    assert "do not use free-form strings" in prompt


def test_refinement_prompt_pins_collider_and_penetration_limit(
    tmp_path: Path,
) -> None:
    """Refinement rewrites the patch, so it needs both constraints."""

    prompt = _refinement_prompt(
        tmp_path,
        collision_approximation="convexHull",
        validation_max_penetration_m=0.05,
    )

    assert "PRESERVE THE COLLISION SHAPE" in prompt
    assert "GROUND-PENETRATION LIMIT" in prompt
    assert "convexHull" in prompt
    assert "0.05" in prompt
    if os.name == "nt":
        assert "artifact write-json --replace --output" in prompt
        assert "raw/physics_decision_patch.json" in prompt


def test_refinement_prompt_preserves_target_ids(tmp_path: Path) -> None:
    prompt = _refinement_prompt(tmp_path)

    assert "Preserve `collider_target_ids`" in prompt
    assert "handwritten USD paths" in prompt


def test_refinement_prompt_requires_exact_checked_view_paths(tmp_path: Path) -> None:
    prompt = _refinement_prompt(tmp_path)

    assert "In `checked_views`, write only exact paths" in prompt
    assert "Never put phase names or prose labels there" in prompt


def test_refinement_prompt_defers_goal_matching_before_tuning(
    tmp_path: Path,
) -> None:
    prompt = _refinement_prompt(
        tmp_path,
        defer_behavior_goal_until_tuning=True,
    )

    assert '"validation_scope": "pre_tuning_baseline"' in prompt
    assert "Goal-level visual acceptance is deferred" in prompt
    assert "target motion such as a slide or bounce" in prompt


def test_refinement_prompt_keeps_final_behavior_scope_by_default(
    tmp_path: Path,
) -> None:
    prompt = _refinement_prompt(tmp_path)

    assert '"validation_scope": "final_behavior"' in prompt
    assert "Goal-level visual acceptance is deferred" not in prompt


def test_refinement_prompt_treats_passing_support_fallback_as_non_actionable(
    tmp_path: Path,
) -> None:
    prompt = _refinement_prompt(tmp_path)

    assert "ground_clearance_support_decision" in prompt
    assert "`ground_clearance_support_decisions`" in prompt
    assert "`per_body_results[].ground_clearance_support_decision`" in prompt
    assert "fallback_accepted: true" in prompt
    assert "measurement metadata" in prompt
    assert "never as the authored collider" in prompt


def test_refinement_prompt_provides_windows_controlled_json_writer(
    tmp_path: Path,
) -> None:
    prompt = _refinement_prompt(tmp_path)

    if os.name == "nt":
        assert "content-workflow-cli artifact write-json" in prompt
        assert "Run the writer as its own command" in prompt
        assert "Set-Content" not in prompt
        assert "Move-Item" not in prompt
        assert "ConvertFrom-Json | Out-Null" not in prompt
    else:
        assert "content-workflow-cli artifact write-json" not in prompt


def test_refinement_prompt_states_the_default_gate_when_unset(tmp_path: Path) -> None:
    """A default run still needs the gate contract spelled out: the child must
    run workflow runtime validation without `max_ground_penetration_m` so its evidence
    measures the same workflow default the run enforces, and must not send an
    explicit null (which disables the numeric gate instead of selecting the
    default). The collision-shape constraint stays conditional."""

    prompt = _refinement_prompt(tmp_path)

    assert "PRESERVE THE COLLISION SHAPE" not in prompt
    assert "GROUND-PENETRATION LIMIT" in prompt
    assert "WITHOUT `max_ground_penetration_m`" in prompt
    assert "disables the numeric gate" in prompt


def test_default_run_task_blocks_omit_the_penetration_key(tmp_path: Path) -> None:
    """A literal `"max_ground_penetration_m": null` in the apply or visual
    task JSON reads as a value to mirror, and mirroring it disables the
    numeric gate; the key must be absent when the run configures no
    override."""

    for prompt in (_apply_prompt(tmp_path), _refinement_prompt(tmp_path)):
        assert '"max_ground_penetration_m": null' not in prompt
        assert '"max_ground_penetration_m"' not in prompt


def test_overridden_run_task_blocks_carry_the_penetration_key(
    tmp_path: Path,
) -> None:
    apply_prompt = _apply_prompt(tmp_path, validation_max_penetration_m=0.05)
    refinement_prompt = _refinement_prompt(tmp_path, validation_max_penetration_m=0.05)

    assert '"max_ground_penetration_m": 0.05' in apply_prompt
    assert '"max_ground_penetration_m": 0.05' in refinement_prompt


def test_post_tuning_prompt_is_goal_level_and_assessment_only(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    prompt = build_physics_post_tuning_assessment_prompt(
        repo_root=tmp_path,
        run_dir=run_dir,
        promoted_usd=run_dir / "physics.usdc",
        promoted_usd_sha256="a" * 64,
        validation_evidence_path=run_dir / "tuning" / "validation.json",
        runtime_report_path=run_dir / "tuning" / "runtime.json",
        rendered_frames=[str(run_dir / "frame.png")],
        reference_frames=[{"label": "Reference Image 1", "path": "reference.png"}],
        assessment_path=run_dir / "raw" / "post_tuning_assessment.json",
        behavior_prompt="Match the reference bounce.",
        scenario_path=tmp_path / "scenario.yaml",
        additional_instructions="Keep the tire upright after the final bounce.",
    )

    assert '"validation_scope": "post_tuning_final_behavior"' in prompt
    assert "assessment-only turn" in prompt
    assert 'Never use `status: "fixed"`' in prompt
    assert "Do not write `physics_behavior_assessment.json`" in prompt
    assert '"additional_instructions": "Keep the tire upright' in prompt
    assert "complete user request" in prompt
