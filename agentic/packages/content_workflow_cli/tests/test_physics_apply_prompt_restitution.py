# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restitution guidance in ``content_workflow_cli.prompts`` apply prompts.

The apply prompt's patch schema carried one illustrative
``"restitution": 0.1``, and the child anchored on it: across 450 prims of a
16-asset property benchmark, every single authored restitution landed in
0.0-0.08 (mean 0.048) against a 0.35-0.65 ground-truth range (mean 0.528).
Friction, whose example values were already physically representative
(0.5/0.4), came out well calibrated over the same run (predicted mean 0.536 vs
0.494 expected) -- the difference between the two is the example value, not the
model's ability to reason about materials.

Near-zero restitution means a body that lands dead with no rebound, which is
wrong for ordinary rigid materials regardless of any benchmark.

Policy-side coverage lives in ``test_physics_policy_restitution.py``.
"""

from pathlib import Path

from content_workflow_cli.prompts import build_physics_apply_prompt


def _prompt(tmp_path: Path, **overrides: object) -> str:
    kwargs: dict[str, object] = {
        "repo_root": tmp_path,
        "run_dir": tmp_path / "run",
        "usd_path": tmp_path / "asset.usd",
    }
    kwargs.update(overrides)
    return build_physics_apply_prompt(**kwargs)  # type: ignore[arg-type]


def test_prompt_gives_per_family_restitution_references(tmp_path: Path) -> None:
    """One number in the schema becomes the answer for every material; the
    prompt has to carry the per-family range instead."""

    prompt = _prompt(tmp_path)

    assert "rubber 0.7-0.85" in prompt
    assert "metal 0.5-0.65" in prompt
    assert "wood 0.4-0.55" in prompt


def test_prompt_bounds_when_near_zero_restitution_is_legitimate(
    tmp_path: Path,
) -> None:
    """Near-zero is a real answer for foam or fabric -- it just must be the
    exception, argued from the material, not the default."""

    prompt = _prompt(tmp_path)

    assert "below 0.1" in prompt
    assert "lands dead with no rebound" in prompt


def test_prompt_disowns_the_example_values(tmp_path: Path) -> None:
    """The example exists to fix JSON types. Saying so is what stops it from
    being read as the recommended answer."""

    prompt = _prompt(tmp_path)

    assert "Do not copy them" in prompt
    assert "own inferred material family" in prompt


def test_example_carries_no_numeric_property_anchors(tmp_path: Path) -> None:
    """The example is the most concrete thing in the prompt, so any number in
    it is read as the recommended answer no matter what the prose says.

    These two tests previously asserted the opposite: keep the numbers
    physically representative and disown them in prose. That did not hold. Over
    a 16-asset property benchmark the example's plastic-range density was
    reproduced on components whose evidence pointed elsewhere, capping material
    accuracy at 0.58. Every value in `physical_properties` is a placeholder
    naming its source now, so there is no number left to copy.
    """

    prompt = _prompt(tmp_path)

    assert '"density": 1150.0' not in prompt
    assert '"static_friction": 0.45' not in prompt
    assert '"restitution": 0.55' not in prompt


def test_example_property_placeholders_are_type_correct_numbers(
    tmp_path: Path,
) -> None:
    """`physical_properties` is `dict[str, float]`, so a quoted-string
    placeholder copied from the example fails patch validation before
    finalization. The placeholders must be JSON numbers, with the prose beside
    the example naming where each real value comes from."""

    prompt = _prompt(tmp_path)

    assert '"density": 0.0' in prompt
    assert '"estimated_mass_kg": 0.0' in prompt
    assert '"static_friction": 0.0' in prompt
    assert '"dynamic_friction": 0.0' in prompt
    assert '"restitution": 0.0' in prompt
    # No string instances left in the example's physical_properties block.
    assert '"density": "<' not in prompt
    assert '"estimated_mass_kg": "<' not in prompt
    assert '"static_friction": "<' not in prompt
    assert '"dynamic_friction": "<' not in prompt
    assert '"restitution": "<' not in prompt
    # The prose still names the sources the placeholders no longer can.
    assert "density in kg/m^3 from that family" in prompt
    assert "friction and restitution from the reference band" in prompt
