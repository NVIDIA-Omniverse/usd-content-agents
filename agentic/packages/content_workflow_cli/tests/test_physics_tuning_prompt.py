# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physics tuning session prompt contract.

The wrapper revalidates an accepted candidate with its own ground-penetration
acceptance override, while the agent must mirror it when evaluating raw usd-cli
simulation facts. When the prompt does not carry the override,
the agent reads a candidate the wrapper would promote as a ground-clearance
failure and stops instead of accepting it -- so the run can never promote.
"""

import os
from pathlib import Path

from content_workflow_cli.prompts import (
    build_physics_external_refine_prompt,
    build_physics_tuning_session_prompt,
)


def _prompt(tmp_path: Path, **overrides: object) -> str:
    kwargs: dict[str, object] = {
        "repo_root": tmp_path,
        "run_dir": tmp_path / "run",
        "physics_usd": tmp_path / "run" / "physics.usdc",
        "broker_url": "http://127.0.0.1:37597",
        "sweep_client_path": "content-workflow-physics-tune-sweep",
        "contract_path": tmp_path / "run" / "raw" / "physics_agentic_contract.json",
        "max_iterations": 3,
        "max_trials_per_sweep": 4,
        "sweep_deadline_seconds": 900.0,
        "engine": "ovphysx",
        "optimizer": "random",
    }
    kwargs.update(overrides)
    return build_physics_tuning_session_prompt(**kwargs)  # type: ignore[arg-type]


def test_prompt_carries_the_wrapper_ground_penetration_override(
    tmp_path: Path,
) -> None:
    """Without this the agent validates at 0.005 m and self-rejects a candidate
    the wrapper's relaxed gate would have promoted."""

    prompt = _prompt(tmp_path, revalidation_max_penetration_m=0.05)

    assert "0.05" in prompt
    assert "max_ground_penetration_m" in prompt
    # The agent must be told to apply it to its own workflow-owned evaluation,
    # not merely that the wrapper uses a different number.
    assert "workflow authors" in prompt
    assert "physics simulate" in prompt


def test_prompt_provides_windows_controlled_json_writer(tmp_path: Path) -> None:
    prompt = _prompt(tmp_path)

    if os.name == "nt":
        assert "content-workflow-cli artifact write-json" in prompt
        assert "Run the writer as its own command" in prompt
        assert "Set-Content" not in prompt
        assert "Move-Item" not in prompt
    else:
        assert "content-workflow-cli artifact write-json" not in prompt


def test_external_refine_prompt_provides_windows_controlled_json_writer(
    tmp_path: Path,
) -> None:
    prompt = build_physics_external_refine_prompt(
        run_dir=tmp_path / "run",
        broker_url="http://127.0.0.1:37597",
        sweep_client_path="content-workflow-physics-external-sweep",
        contract_path=tmp_path / "run" / "raw" / "external_contract.json",
        user_prompt="Tune the bounce",
        task_name="bounce",
        objective={"name": "bounce_error", "direction": "minimize"},
        parameter_catalog=[{"name": "restitution"}],
        initial_active_search={"restitution": {"min": 0.0, "max": 1.0}},
        nominal_params={"restitution": 0.5},
        max_iterations=2,
        max_trials_per_sweep=4,
        sweep_deadline_seconds=900.0,
    )

    if os.name == "nt":
        assert "content-workflow-cli artifact write-json" in prompt
        assert "Run the writer as its own command" in prompt
        assert "Set-Content" not in prompt
        assert "Move-Item" not in prompt
    else:
        assert "content-workflow-cli artifact write-json" not in prompt


def test_prompt_omits_the_override_when_unset(tmp_path: Path) -> None:
    """With no revalidation setup at all there is no gate to mirror."""

    prompt = _prompt(tmp_path)

    assert "PROMOTION GATE" not in prompt


def test_default_run_is_not_told_the_limit_is_non_default(tmp_path: Path) -> None:
    """The runner always supplies `duration_s`/`sample_fps` (never None), so the
    gate note renders on every real run -- including default ones, where the
    wrapper passes `acceptance={}` and the workflow applies its own default (the
    scale-relative limit on exact collider geometry, 0.005 m on the bbox
    fallback). Telling the agent the limit is non-default there inverts the
    very divergence this note exists to close, so the prompt must describe the
    actual default rule and tell the child to omit the key."""

    prompt = _prompt(
        tmp_path,
        revalidation_duration_s=1.0,
        revalidation_sample_fps=30,
        revalidation_dt=1.0 / 240.0,
    )

    assert "PROMOTION GATE" in prompt
    assert "NOT the runtime default" not in prompt
    assert "penetration limit is the runtime default" in prompt
    assert "scale-relative" in prompt
    assert "do not invent one" in prompt
    # No drop-height override was configured, so the gate cannot claim its drop
    # height differs from the scenario's.
    assert "drop height is NOT" not in prompt


def test_overridden_run_is_told_the_limit_is_non_default(tmp_path: Path) -> None:
    """The non-default claim must still render when the run really overrides."""

    prompt = _prompt(
        tmp_path,
        revalidation_duration_s=1.0,
        revalidation_sample_fps=30,
        revalidation_max_penetration_m=0.05,
        revalidation_drop_height_m=0.9,
    )

    assert "NOT the runtime default" in prompt
    assert "drop height is NOT" in prompt


def test_prompt_reports_the_override_in_the_task_block(tmp_path: Path) -> None:
    """The machine-readable task block is the durable contract the agent reads
    first; the gate value belongs there too, not only in prose."""

    prompt = _prompt(tmp_path, revalidation_max_penetration_m=0.05)

    assert "promotion_gate" in prompt


def test_prompt_carries_the_revalidation_drop_setup(tmp_path: Path) -> None:
    """Matching the threshold is not enough: the wrapper revalidates with its own
    drop height, so an agent validating from a shorter scenario drop under-reports
    penetration and accepts a candidate that then fails promotion."""

    prompt = _prompt(
        tmp_path,
        revalidation_max_penetration_m=0.05,
        revalidation_drop_height_m=0.9,
        revalidation_duration_s=2.0,
        revalidation_sample_fps=30.0,
        revalidation_dt=0.005,
    )

    assert "drop_height_m=0.9" in prompt
    assert "duration_s=2.0" in prompt
    assert "sample_fps=30.0" in prompt
    # `dt` reaches the JSON task block, but an agent that follows the prose
    # instruction alone would otherwise integrate at a different timestep than
    # the gate -- the same "different experiment" failure this note exists to
    # close for drop height, duration, and sample rate.
    assert "dt=0.005" in prompt


def test_prompt_pins_the_authored_collision_approximation(tmp_path: Path) -> None:
    """`revise_patch` may rewrite any patch field. Unconstrained, degrading the
    collider to a bounding primitive is the degenerate optimum: a shape that
    cannot roll trivially satisfies settle / no-backtracking / zero-final-angular
    velocity. Observed convexHull -> boundingSphere -> boundingCube on an apple."""

    prompt = _prompt(tmp_path, collision_approximation="convexHull")

    assert "PRESERVE THE COLLISION SHAPE" in prompt
    assert "convexHull" in prompt
    assert "authored_collision_approximation" in prompt


def test_prompt_omits_the_collider_pin_when_unset(tmp_path: Path) -> None:
    """No collision approximation configured means no constraint to state."""

    prompt = _prompt(tmp_path)

    assert "PRESERVE THE COLLISION SHAPE" not in prompt


def test_prompt_gate_survives_a_partial_revalidation_setup(tmp_path: Path) -> None:
    """`--drop-height-m` defaults to the asset bbox height, so drop height can be
    None while the penetration override is set. The gate note must still render."""

    prompt = _prompt(tmp_path, revalidation_max_penetration_m=0.05)

    assert "PROMOTION GATE" in prompt
    assert "drop_height_m=None" not in prompt


def test_prompt_protects_vomp_mass_properties_during_tuning(tmp_path: Path) -> None:
    prompt = _prompt(
        tmp_path,
        protected_parameters=["mass_scale"],
        allow_revise_patch=False,
    )

    assert '"protected_parameters": [' in prompt
    assert '"allow_revise_patch": false' in prompt
    assert "do not include `mass_scale`" in prompt
    assert "revise_patch: PROHIBITED" in prompt
    assert "broker rejects a sweep before reservation" in prompt


def test_default_run_task_block_omits_the_penetration_key(tmp_path: Path) -> None:
    """The prompt says to mirror `promotion_gate` exactly, and an explicit
    `"max_ground_penetration_m": null` disables the numeric gate at
    workflow runtime evaluator instead of selecting its default — so on a
    default run the key must be absent from the serialized task JSON, not
    null."""

    prompt = _prompt(
        tmp_path,
        revalidation_duration_s=1.0,
        revalidation_sample_fps=30,
        revalidation_dt=1.0 / 240.0,
    )

    assert '"max_ground_penetration_m": null' not in prompt
    assert '"max_ground_penetration_m"' not in prompt


def test_overridden_run_task_block_carries_the_penetration_key(
    tmp_path: Path,
) -> None:
    prompt = _prompt(
        tmp_path,
        revalidation_duration_s=1.0,
        revalidation_sample_fps=30,
        revalidation_max_penetration_m=0.05,
    )

    assert '"max_ground_penetration_m": 0.05' in prompt


def test_prompt_exports_candidates_for_ovrtx_review_before_acceptance(
    tmp_path: Path,
) -> None:
    prompt = _prompt(tmp_path, candidate_suffix=".usdc")
    normalized_prompt = " ".join(prompt.split())

    assert "--output-usd" in prompt
    assert "--output-recording" in prompt
    assert "trial_<n>.usdc" in prompt
    assert "trial_<n>_recording<recording-suffix>" in prompt
    assert "candidate's `recording` path" in prompt
    assert "materialize \\\n      --sweep-id" in prompt
    assert "--trial-index <n> \\\n      --output-usd" in prompt
    assert "exported_usd_path" in prompt
    assert "exported_recording_path" in prompt
    assert "exact scenario rollout" in normalized_prompt
    assert "Render that exported recording directly through OVRTX" in normalized_prompt
    assert "failed final visual assessment rejects and rolls back promotion" in prompt
