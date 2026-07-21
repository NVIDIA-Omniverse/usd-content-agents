# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the freeform scenario's programmatic-score helper.

These lock the documented 0.3 / 0.3 weights for the ``settled`` and
``finite_position`` components and verify that ``fell_over`` remains a
diagnostic without becoming an automatic score. Ground-clearance tests
also lock the relative tolerance derived from the selected body's bbox.
"""

from __future__ import annotations

import math

import pytest

from physics_agent.tuning.scenarios.freeform import (
    _add_ground_clearance_to_summary,
    _ground_clearance_tolerance,
    _normalize_observations,
    _normalize_weights,
    _pose7_from_trajectory_sample,
    _score_programmatic_from_summary,
    _world_up_axis,
)


def _summary(
    *,
    fell_over: bool = False,
    settle_time_s: float | None = 0.5,
    duration_s: float = 1.0,
    final_position: tuple[float, float, float] = (0.0, 1.0, 0.0),
    n_samples: int = 30,
    min_ground_clearance: float | None = None,
    ground_clearance_tolerance: object | None = None,
) -> dict[str, object]:
    out = {
        "fell_over": fell_over,
        "settle_time_s": settle_time_s,
        "duration_s": duration_s,
        "final_position": list(final_position),
        "n_samples": n_samples,
    }
    if min_ground_clearance is not None:
        out["min_ground_clearance"] = min_ground_clearance
    if ground_clearance_tolerance is not None:
        out["ground_clearance_tolerance"] = ground_clearance_tolerance
    return out


def _bottle_ground_summary(*, clearance: float) -> dict[str, object]:
    """Build clearance metrics for the observed bottle bbox."""
    summary: dict[str, object] = {}
    _add_ground_clearance_to_summary(
        summary,
        [(0.0, [0.0, 0.0, clearance, 0.0, 0.0, 0.0, 1.0], [0.0] * 6)],
        {
            "world_up": [0.0, 0.0, 1.0],
            "bbox_min_local_stage": [-0.01666755, -0.0167747, 0.0],
            "bbox_max_local_stage": [0.01666755, 0.0167747, 0.13],
        },
    )
    return summary


# ---------------------------------------------------------------------------
# Base programmatic components.
# ---------------------------------------------------------------------------


def test_all_base_components_pass_returns_one() -> None:
    score, critique = _score_programmatic_from_summary(
        _summary(),
        observations=["should stay upright"],
    )
    assert score == pytest.approx(1.0)
    assert "upright" not in critique
    assert "settled=pass" in critique
    assert "finite_position=pass" in critique


@pytest.mark.parametrize("trigger", ["upright", "stable", "fall", "topple", "tip"])
@pytest.mark.parametrize("fell_over", [False, True])
def test_fell_over_remains_diagnostic_not_programmatic_score(
    trigger: str, fell_over: bool
) -> None:
    score, critique = _score_programmatic_from_summary(
        _summary(fell_over=fell_over, settle_time_s=None),
        observations=[f"the object should {trigger}"],
    )

    assert score == pytest.approx(0.5)
    assert critique == "settled=fail; finite_position=pass"
    assert "upright" not in critique


def test_only_settled_fails_costs_exactly_zero_point_three() -> None:
    score, _ = _score_programmatic_from_summary(
        _summary(settle_time_s=None),
        observations=["should stay upright"],
    )
    assert score == pytest.approx(0.5)


def test_only_finite_fails_costs_exactly_zero_point_three() -> None:
    score, _ = _score_programmatic_from_summary(
        _summary(final_position=(float("inf"), 0.0, 0.0)),
        observations=["should stay upright"],
    )
    assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Base component normalization.
# ---------------------------------------------------------------------------


def test_base_components_ignore_unrelated_observation_text() -> None:
    score, critique = _score_programmatic_from_summary(
        _summary(),
        observations=["bounce around the room"],
    )
    assert score == pytest.approx(1.0)
    assert "upright" not in critique
    assert "settled=pass" in critique
    assert "finite_position=pass" in critique


def test_only_settled_passes_normalizes_to_zero_point_five() -> None:
    score, _ = _score_programmatic_from_summary(
        _summary(final_position=(float("nan"), 0.0, 0.0)),
        observations=["just see what happens"],
    )
    assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_samples_returns_zero() -> None:
    """No trajectory data → score is 0 with a clear critique."""
    score, critique = _score_programmatic_from_summary(
        _summary(n_samples=0),
        observations=["upright"],
    )
    assert score == 0.0
    assert "no programmatic signal" in critique


def test_ground_clearance_tolerance_rejects_invalid_bbox() -> None:
    assert _ground_clearance_tolerance((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)) is None
    assert _ground_clearance_tolerance((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) is None


def test_ground_clearance_shallow_bottle_penetration_passes() -> None:
    """The worst observed 1.912 mm overlap fits within 2% of the diagonal."""
    clearance_summary = _bottle_ground_summary(clearance=-0.001912)
    tolerance = float(clearance_summary["ground_clearance_tolerance"])
    expected_tolerance = 0.02 * math.hypot(0.0333351, 0.0335494, 0.13)
    assert tolerance == pytest.approx(expected_tolerance)
    assert tolerance > 0.001912

    summary = _summary()
    summary.update(clearance_summary)
    score, critique = _score_programmatic_from_summary(
        summary,
        observations=["settle on the floor without sinking through it"],
    )
    assert score == pytest.approx(1.0)
    assert "ground_clearance=pass(clearance=-0.001912" in critique
    assert f"tolerance={tolerance:.6g}" in critique


def test_ground_clearance_materially_deep_penetration_fails() -> None:
    """A 20 mm overlap remains far outside the bottle-scale tolerance."""
    clearance_summary = _bottle_ground_summary(clearance=-0.02)
    tolerance = float(clearance_summary["ground_clearance_tolerance"])
    summary = _summary()
    summary.update(clearance_summary)

    score, critique = _score_programmatic_from_summary(
        summary,
        observations=["settle on the floor without sinking through it"],
    )
    assert score == pytest.approx(0.5)
    assert "ground_clearance=fail(clearance=-0.02" in critique
    assert f"tolerance={tolerance:.6g}" in critique


def test_ground_clearance_pass_keeps_floor_observation_at_one() -> None:
    score, critique = _score_programmatic_from_summary(
        _summary(min_ground_clearance=0.0),
        observations=["rest naturally on the ground surface"],
    )
    assert score == pytest.approx(1.0)
    assert "ground_clearance=pass" in critique


def test_ground_clearance_is_conditional_on_observation_text() -> None:
    """A negative clearance is only scored when the prompt/observations
    actually ask about floor/ground/surface contact."""
    score, critique = _score_programmatic_from_summary(
        _summary(min_ground_clearance=-0.02),
        observations=["spin in open air"],
    )
    assert score == pytest.approx(1.0)
    assert "ground_clearance" not in critique


def test_ground_clearance_ignores_unrelated_through_prompt() -> None:
    score, critique = _score_programmatic_from_summary(
        _summary(min_ground_clearance=-0.02),
        observations=["roll through the open air"],
    )

    assert score == pytest.approx(1.0)
    assert "ground_clearance" not in critique


def test_ground_clearance_summary_uses_world_up_and_local_bbox() -> None:
    """The runtime helper combines trajectory pose with the selected
    body's local bbox, which catches the RoboCasa nested-body failure
    where the origin settled at the floor while the mesh bottom was below it."""
    summary: dict[str, object] = {}
    trajectory = [
        (0.0, [0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
        (1.0, [0.0, 0.0, -0.03, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
    ]
    _add_ground_clearance_to_summary(
        summary,
        trajectory,
        {"world_up": [0.0, 0.0, 1.0], "bbox_min_local_stage": [0.0, 0.0, 0.0]},
    )
    assert summary["min_ground_clearance"] == pytest.approx(-0.03)


def test_ground_clearance_summary_uses_rotated_bbox_corners() -> None:
    """A tilted object can penetrate with a corner even when its origin
    remains above the floor; freeform must score the rotated bbox bottom."""
    summary: dict[str, object] = {}
    qy = math.sin(math.pi / 4.0)
    qw = math.cos(math.pi / 4.0)
    trajectory = [
        (0.0, [0.0, 0.0, 0.10, 0.0, qy, 0.0, qw], [0.0] * 6),
    ]
    _add_ground_clearance_to_summary(
        summary,
        trajectory,
        {
            "world_up": [0.0, 0.0, 1.0],
            "bbox_min_local_stage": [-0.20, -0.05, 0.0],
            "bbox_max_local_stage": [0.20, 0.05, 0.10],
        },
    )
    assert summary["min_ground_clearance"] == pytest.approx(-0.10)
    assert summary["ground_clearance_tolerance"] == pytest.approx(
        0.02 * math.hypot(0.40, 0.10, 0.10)
    )

    score_summary = _summary()
    score_summary.update(summary)
    score, critique = _score_programmatic_from_summary(
        score_summary,
        observations=["rest on the floor without penetrating it"],
    )
    assert score == pytest.approx(0.5)
    assert "ground_clearance=fail(clearance=-0.1" in critique


def test_freeform_trajectory_helpers_cover_malformed_samples() -> None:
    """Defensive parser branches should degrade to no pose, not crash."""
    assert _world_up_axis([0.0, 0.0, 0.0]) == 1

    assert _pose7_from_trajectory_sample({"position": [1.0, 2.0, 3.0]}) == (
        1.0,
        2.0,
        3.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert _pose7_from_trajectory_sample(object()) is None
    assert _pose7_from_trajectory_sample({"pose": ["bad", 0.0, 0.0]}) is None
    assert _pose7_from_trajectory_sample({"pose": [1.0, 2.0]}) is None


def test_ground_clearance_summary_skips_malformed_samples() -> None:
    summary: dict[str, object] = {}
    _add_ground_clearance_to_summary(
        summary,
        [object()],
        {"world_up": [0.0, 0.0, 1.0], "bbox_min_local_stage": [0.0, 0.0, 0.0]},
    )
    assert "min_ground_clearance" not in summary


def test_ground_clearance_summary_omits_tolerance_for_malformed_bbox() -> None:
    """An incomplete bbox keeps the legacy bottom estimate but earns no slop."""
    summary: dict[str, object] = {}
    _add_ground_clearance_to_summary(
        summary,
        [(0.0, [0.0, 0.0, -0.0017], [0.0] * 6)],
        {
            "world_up": [0.0, 0.0, 1.0],
            "bbox_min_local_stage": [0.0, 0.0, 0.0],
            "bbox_max_local_stage": [0.1, "bad", 0.155],
        },
    )

    assert summary["min_ground_clearance"] == pytest.approx(-0.0017)
    assert "ground_clearance_tolerance" not in summary


@pytest.mark.parametrize(
    "tolerance",
    ["not-a-number", -0.01, float("nan"), float("inf")],
)
def test_invalid_ground_clearance_tolerance_fails_ground_component(
    tolerance: object,
) -> None:
    score, critique = _score_programmatic_from_summary(
        _summary(
            min_ground_clearance=-0.001,
            ground_clearance_tolerance=tolerance,
        ),
        observations=["rest on the ground"],
    )

    assert score == pytest.approx(0.5)
    assert "ground_clearance=fail(clearance=-0.001, tolerance=invalid)" in critique


def test_invalid_ground_clearance_value_fails_ground_component() -> None:
    summary = _summary()
    summary["min_ground_clearance"] = "not-a-number"
    summary["ground_clearance_tolerance"] = 0.002

    score, critique = _score_programmatic_from_summary(
        summary,
        observations=["rest on the ground"],
    )

    assert score == pytest.approx(0.5)
    assert "ground_clearance=fail(clearance=invalid, tolerance=0.002)" in critique


# ---------------------------------------------------------------------------
# Weight validation (CodeRabbit R13 thread #6).
# ---------------------------------------------------------------------------


def test_normalize_weights_rejects_unknown_key() -> None:
    """Unknown keys must surface as a clear ValueError, not silently
    extend ``base`` and corrupt the optimizer signal."""
    with pytest.raises(ValueError, match="Unsupported freeform weight key"):
        _normalize_weights({"vision": 0.5}, vlm_available=True)


def test_normalize_weights_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        _normalize_weights({"programmatic": -0.1}, vlm_available=True)


def test_normalize_weights_rejects_nan_weight() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        _normalize_weights(
            {"programmatic": float("nan"), "vlm": 0.5}, vlm_available=True
        )


def test_normalize_weights_rejects_inf_weight() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        _normalize_weights(
            {"programmatic": float("inf"), "vlm": 0.5}, vlm_available=True
        )


def test_normalize_weights_rejects_all_zero_total_when_vlm_available() -> None:
    """Both weights zero with VLM enabled → must raise. Otherwise the
    optimizer would consume an arbitrary tiebreak from epsilon clamps."""
    with pytest.raises(ValueError, match="At least one of freeform weights"):
        _normalize_weights({"programmatic": 0.0, "vlm": 0.0}, vlm_available=True)


def test_normalize_weights_rejects_bool() -> None:
    """``True`` is a Python int subclass but accepting it would let
    ``programmatic: yes`` (YAML coerces to True) silently authorize 1.0."""
    with pytest.raises(ValueError, match="must be a number"):
        _normalize_weights({"programmatic": True}, vlm_available=True)


def test_normalize_weights_unchanged_path_remains_05_05() -> None:
    """No weights supplied → 0.5 / 0.5 default; sanity-check the happy
    path isn't broken by the new validation."""
    weights = _normalize_weights(None, vlm_available=True)
    assert weights["programmatic"] == pytest.approx(0.5)
    assert weights["vlm"] == pytest.approx(0.5)


def test_normalize_weights_no_vlm_returns_programmatic_one() -> None:
    """VLM unavailable → programmatic = 1.0 regardless of inputs."""
    weights = _normalize_weights({"programmatic": 0.3, "vlm": 0.7}, vlm_available=False)
    assert weights == {"programmatic": 1.0, "vlm": 0.0}


# ---------------------------------------------------------------------------
# Observations normalization (CodeRabbit R13 thread #7).
# ---------------------------------------------------------------------------


def test_normalize_observations_scalar_string_stays_single_observation() -> None:
    """YAML scalar ``observations: "steady"`` must become ``["steady"]``,
    NOT ``['s', 't', 'e', 'a', 'd', 'y']``."""
    assert _normalize_observations("steady") == ["steady"]


def test_normalize_observations_list_pass_through() -> None:
    assert _normalize_observations(["a", "b"]) == ["a", "b"]


def test_normalize_observations_tuple_pass_through() -> None:
    assert _normalize_observations(("a", "b")) == ["a", "b"]


def test_normalize_observations_none_becomes_empty_list() -> None:
    assert _normalize_observations(None) == []


def test_normalize_observations_coerces_non_string_items_to_strings() -> None:
    assert _normalize_observations([1, 2.5, "stay"]) == ["1", "2.5", "stay"]


def test_normalize_observations_unexpected_shape_falls_through_to_str() -> None:
    """An unexpected dict shape becomes a one-item list of its repr —
    keeps audit artifacts honest instead of silently dropping the value."""
    out = _normalize_observations({"text": "should stay upright"})
    assert len(out) == 1
    assert "upright" in out[0]
