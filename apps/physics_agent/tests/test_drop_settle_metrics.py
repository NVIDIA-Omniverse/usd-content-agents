# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the drop_settle metric registry.

The registry decouples ``scenario.metric`` from a single hard-coded
scalar so the refine loop can swap in metrics like
``max_bounce_height`` without touching ``evaluate()``. Tests run on
synthetic ``[(t, pose7, vel6), ...]`` trajectories — no daemon, no USD.
"""

from __future__ import annotations

import math
import pathlib

import pytest

from physics_agent.tuning.scenarios.drop_settle import (
    _METRICS,
    MetricContext,
    _infer_up_idx,
    _metric_max_bounce_height,
    _metric_settle_distance,
    _resolve_up_idx,
    _rotate_vector_by_pose_quat,
)
from physics_agent.tuning.types import Scenario, TunableParam


def _scenario(
    metric: str = "settle_distance",
    target: dict[str, object] | None = None,
) -> Scenario:
    scenario_target: dict[str, object] = {"drop_height_m": 0.5}
    if target:
        scenario_target.update(target)
    return Scenario(
        name="drop_settle",
        params=(TunableParam(name="restitution", min_value=0.0, max_value=1.0),),
        target=scenario_target,
        metric=metric,
    )


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[float]:
    """Pose7 [px, py, pz, qx, qy, qz, qw]."""
    return [x, y, z, 0.0, 0.0, 0.0, 1.0]


def _pose_x90(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[float]:
    """Pose7 with a 90-degree rotation about X."""
    angle = math.pi / 2.0
    return [x, y, z, math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0)]


def _vel(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    wx: float = 0.0,
    wy: float = 0.0,
    wz: float = 0.0,
) -> list[float]:
    """Vel6 [vx, vy, vz, wx, wy, wz]."""
    return [x, y, z, wx, wy, wz]


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_registry_exposes_known_metrics() -> None:
    assert "settle_distance" in _METRICS
    assert "max_bounce_height" in _METRICS
    for fn in _METRICS.values():
        assert callable(fn)


# ---------------------------------------------------------------------------
# Up-axis inference
# ---------------------------------------------------------------------------


def test_infer_up_idx_y_up() -> None:
    assert _infer_up_idx([0.0, 0.5, 0.0]) == 1


def test_infer_up_idx_z_up() -> None:
    assert _infer_up_idx([0.0, 0.0, 0.7]) == 2


def test_infer_up_idx_origin_falls_back_to_y() -> None:
    """Corner-origin assets (rest_position == origin) default to Y-up."""
    assert _infer_up_idx([0.0, 0.0, 0.0]) == 1


# ---------------------------------------------------------------------------
# _resolve_up_idx — prefers scene_info["world_up"] over inference
# ---------------------------------------------------------------------------


def test_resolve_up_idx_prefers_world_up_for_z_up_corner_origin() -> None:
    """Corner-origin Z-up asset: rest_position is the origin (would default
    to Y-up under inference), but scene_info["world_up"] = [0, 0, 1] resolves
    to Z-up so bounce metrics measure the correct axis."""
    scene_info = {"world_up": [0.0, 0.0, 1.0]}
    assert _resolve_up_idx(scene_info, [0.0, 0.0, 0.0]) == 2


def test_resolve_up_idx_prefers_world_up_y_up() -> None:
    scene_info = {"world_up": [0.0, 1.0, 0.0]}
    assert _resolve_up_idx(scene_info, [0.0, 0.0, 0.0]) == 1


def test_resolve_up_idx_falls_back_when_world_up_missing() -> None:
    """Older callers stub scene_info without world_up — fall through to
    inference from rest_position."""
    assert _resolve_up_idx({}, [0.0, 0.5, 0.0]) == 1
    assert _resolve_up_idx(None, [0.0, 0.0, 0.7]) == 2


def test_resolve_up_idx_handles_zero_world_up_as_fallback() -> None:
    """A degenerate world_up vector (all zeros) falls back to inference."""
    assert _resolve_up_idx({"world_up": [0.0, 0.0, 0.0]}, [0.0, 0.5, 0.0]) == 1


# ---------------------------------------------------------------------------
# settle_distance metric
# ---------------------------------------------------------------------------


def test_settle_distance_equals_zero_when_final_pose_at_rest() -> None:
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=2.0), _vel()),
        (0.5, _pose(y=1.5), _vel()),
        (1.0, _pose(y=0.5), _vel()),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario(),
    )
    assert _metric_settle_distance(ctx) == pytest.approx(0.0, abs=1e-6)


def test_settle_distance_grows_with_offset() -> None:
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=2.0), _vel()),
        (1.0, _pose(y=1.5), _vel()),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario(),
    )
    # final at y=1.5, rest at y=0.5 → 1.0
    assert _metric_settle_distance(ctx) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# max_bounce_height metric
# ---------------------------------------------------------------------------


def test_max_bounce_height_finds_peak_after_first_contact() -> None:
    """Drop, contact, rebound, then apex: score is bbox-bottom event height."""
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=2.0), _vel(y=-3.0)),
        (0.1, _pose(y=1.5), _vel(y=-2.0)),
        (0.2, _pose(y=0.5), _vel(y=0.0)),  # velocity-defined contact
        (0.3, _pose(y=1.0), _vel(y=1.0)),  # rebound rising
        (0.4, _pose(y=1.4), _vel(y=0.0)),  # velocity-defined apex
        (0.5, _pose(y=1.0), _vel(y=-1.0)),  # falling again
        (0.6, _pose(y=0.5), _vel(y=-0.2)),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=(-0.5, -0.5, -0.5),
        bbox_max_local=(0.5, 0.5, 0.5),
    )
    score = _metric_max_bounce_height(ctx)
    # Contact bottom = 0.0, apex bottom = 0.9.
    assert score == pytest.approx(-0.9, abs=1e-6)


def test_max_bounce_height_uses_velocity_apex_sample_not_prior_peak() -> None:
    """The apex event is the velocity crossing, not a geometric peak search."""
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=2.0), _vel(y=-3.0)),
        (0.1, _pose(y=0.5), _vel(y=0.0)),  # velocity-defined contact
        (0.2, _pose(y=1.5), _vel(y=1.0)),  # prior geometric peak
        (0.3, _pose(y=1.3), _vel(y=0.0)),  # velocity-defined apex
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=(-0.5, -0.5, -0.5),
        bbox_max_local=(0.5, 0.5, 0.5),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(-0.8, abs=1e-6)


def test_max_bounce_height_zero_crossing_survives_low_positive_velocity() -> None:
    """A low positive frame before apex must not trigger position fallback."""
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=2.0), _vel(y=-3.0)),
        (0.1, _pose(y=0.5), _vel(y=0.0)),  # velocity-defined contact
        (0.2, _pose(y=1.0), _vel(y=1.0)),  # rebound rising
        (0.3, _pose(y=0.95), _vel(y=0.01)),  # spin-like bbox dip
        (0.4, _pose(y=1.3), _vel(y=0.0)),  # velocity-defined apex
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=(-0.5, -0.5, -0.5),
        bbox_max_local=(0.5, 0.5, 0.5),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(-0.8, abs=1e-6)


def test_max_bounce_height_uses_sampled_contact_baseline_without_clamp() -> None:
    """Penetration is not hidden by clamping the measured contact sample."""
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=2.0), _vel(y=-3.0)),
        (0.1, _pose(y=0.3), _vel(y=0.0)),  # bbox bottom penetrates to -0.2
        (0.2, _pose(y=1.1), _vel(y=1.0)),
        (0.3, _pose(y=1.1), _vel(y=0.0)),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=(-0.5, -0.5, -0.5),
        bbox_max_local=(0.5, 0.5, 0.5),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(-0.8, abs=1e-6)


def test_max_bounce_height_higher_rebound_yields_lower_score() -> None:
    """A higher rebound (more bouncy) must produce a lower score so
    the optimizer drives toward larger bounce heights."""
    rest = (0.0, 0.5, 0.0)
    low_bounce = [
        (0.0, _pose(y=2.0), _vel(y=-3.0)),
        (0.1, _pose(y=0.5), _vel(y=0.0)),  # contact
        (0.2, _pose(y=0.7), _vel(y=0.8)),  # tiny bounce
        (0.3, _pose(y=0.7), _vel(y=0.0)),  # apex
    ]
    high_bounce = [
        (0.0, _pose(y=2.0), _vel(y=-3.0)),
        (0.1, _pose(y=0.5), _vel(y=0.0)),  # contact
        (0.2, _pose(y=1.8), _vel(y=1.0)),  # big bounce
        (0.3, _pose(y=1.8), _vel(y=0.0)),  # apex
    ]
    sc = _scenario("max_bounce_height")
    bbox_min = (-0.5, -0.5, -0.5)
    bbox_max = (0.5, 0.5, 0.5)
    low = _metric_max_bounce_height(
        MetricContext(
            low_bounce,
            rest,
            up_idx=1,
            scenario=sc,
            bbox_min_local=bbox_min,
            bbox_max_local=bbox_max,
        )
    )
    high = _metric_max_bounce_height(
        MetricContext(
            high_bounce,
            rest,
            up_idx=1,
            scenario=sc,
            bbox_min_local=bbox_min,
            bbox_max_local=bbox_max,
        )
    )
    assert high < low  # negative-of-bigger is smaller


def test_max_bounce_height_z_up_uses_z_axis() -> None:
    rest = (0.0, 0.0, 0.5)  # Z-up
    trajectory = [
        (0.0, _pose(z=2.0), _vel(z=-3.0)),
        (0.1, _pose(z=0.5), _vel(z=0.0)),  # contact
        (0.2, _pose(z=1.6), _vel(z=1.0)),  # rebound on Z
        (0.3, _pose(z=1.6), _vel(z=0.0)),  # apex
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=2,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=(-0.5, -0.5, -0.5),
        bbox_max_local=(0.5, 0.5, 0.5),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(-1.1, abs=1e-6)


def test_max_bounce_height_uses_bbox_bottom_not_origin_height() -> None:
    """Tire-like case: origin stays high while bbox bottom hits ground."""
    rest = (0.0, 1.0, 0.0)
    trajectory = [
        (0.0, _pose(y=2.2), _vel(y=-3.0)),
        (0.1, _pose(y=1.0), _vel(y=0.0)),  # origin high, bottom at ground
        (0.2, _pose(y=1.5), _vel(y=1.0)),
        (0.3, _pose(y=1.5), _vel(y=0.0)),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=(-0.2, -1.0, -0.2),
        bbox_max_local=(0.2, 1.0, 0.2),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(-0.5, abs=1e-6)


def test_max_bounce_height_without_bbox_uses_rest_origin_as_contact() -> None:
    """Fallback origin-height scoring still works for center-origin assets."""
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=1.4), _vel(y=-3.0)),
        (0.1, _pose(y=0.5), _vel(y=0.0)),  # origin at expected rest/contact
        (0.2, _pose(y=1.0), _vel(y=1.0)),
        (0.3, _pose(y=1.0), _vel(y=0.0)),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(-0.5, abs=1e-6)


def test_rotate_vector_by_pose_quat_handles_rotation_and_normalization() -> None:
    angle = math.pi / 2.0
    qx = math.sin(angle / 2.0) * 2.0
    qw = math.cos(angle / 2.0) * 2.0
    rotated = _rotate_vector_by_pose_quat(
        (0.0, 1.0, 0.0),
        [0.0, 0.0, 0.0, qx, 0.0, 0.0, qw],
    )
    assert rotated == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)


def test_rotate_vector_by_pose_quat_ignores_degenerate_quaternion() -> None:
    assert _rotate_vector_by_pose_quat(
        (1.0, 2.0, 3.0),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ) == (1.0, 2.0, 3.0)


def test_max_bounce_height_uses_rotated_bbox_bottom() -> None:
    """Metric-level coverage for rotated bbox corner selection."""
    rest = (0.0, 0.2, 0.0)
    trajectory = [
        (0.0, _pose_x90(y=1.2), _vel(y=-3.0)),
        (0.1, _pose_x90(y=0.2), _vel(y=0.0)),  # rotated bbox bottom at ground
        (0.2, _pose_x90(y=0.9), _vel(y=1.0)),
        (0.3, _pose_x90(y=0.9), _vel(y=0.0)),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=(-0.1, -1.0, -0.2),
        bbox_max_local=(0.1, 1.0, 0.2),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(-0.7, abs=1e-6)


def test_max_bounce_height_requires_velocity_defined_apex() -> None:
    """Position decreases must not stand in for the upward→non-upward event."""
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=2.0), _vel(y=-3.0)),
        (0.1, _pose(y=0.5), _vel(y=0.0)),
        (0.2, _pose(y=1.1), _vel(y=1.0)),  # first rebound peak
        (0.3, _pose(y=0.9), _vel(y=0.8)),  # position falls, velocity still up
        (0.4, _pose(y=1.7), _vel(y=0.7)),  # later/higher bounce ignored
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=(-0.5, -0.5, -0.5),
        bbox_max_local=(0.5, 0.5, 0.5),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(0.0, abs=1e-6)


def test_max_bounce_height_without_observed_apex_returns_zero() -> None:
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=2.0), _vel(y=-3.0)),
        (0.1, _pose(y=0.5), _vel(y=0.0)),
        (0.2, _pose(y=0.8), _vel(y=1.0)),
        (0.3, _pose(y=1.0), _vel(y=0.8)),
        (0.4, _pose(y=1.2), _vel(y=0.6)),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=(-0.5, -0.5, -0.5),
        bbox_max_local=(0.5, 0.5, 0.5),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(0.0, abs=1e-6)


def test_max_bounce_height_empty_trajectory_returns_inf() -> None:
    ctx = MetricContext(
        trajectory=[],
        rest_position=(0.0, 0.5, 0.0),
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
    )
    assert math.isinf(_metric_max_bounce_height(ctx))


def test_max_bounce_height_single_sample_trajectory_returns_inf() -> None:
    ctx = MetricContext(
        trajectory=[(0.0, _pose(y=1.0), _vel())],
        rest_position=(0.0, 0.5, 0.0),
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
    )
    assert math.isinf(_metric_max_bounce_height(ctx))


def test_max_bounce_height_misaligned_pose_velocity_arrays_returns_inf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.physics import (
        trajectory as trajectory_mod,
    )

    class _Arrayish(list[list[float]]):
        @property
        def size(self) -> int:
            return len(self)

    def _misaligned_arrays(_trajectory: object) -> tuple[object, object, object]:
        return (
            [0.0, 0.1],
            _Arrayish([_pose(y=1.0), _pose(y=1.1)]),
            _Arrayish([_vel(y=-1.0), _vel(y=0.0), _vel(y=1.0)]),
        )

    monkeypatch.setattr(trajectory_mod, "_trajectory_to_arrays", _misaligned_arrays)
    ctx = MetricContext(
        trajectory=[(0.0, _pose(y=1.0), _vel())],
        rest_position=(0.0, 0.5, 0.0),
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
    )
    assert math.isinf(_metric_max_bounce_height(ctx))


def test_max_bounce_height_no_rebound_returns_zero() -> None:
    """No upward velocity after impact means no bounce height."""
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=3.0), _vel(y=-3.0)),
        (0.1, _pose(y=0.5), _vel(y=-0.2)),
        (0.2, _pose(y=0.5), _vel(y=0.0)),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=(-0.5, -0.5, -0.5),
        bbox_max_local=(0.5, 0.5, 0.5),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(0.0, abs=1e-6)


def test_max_bounce_height_does_not_use_geometry_gate_for_contact() -> None:
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=3.0), _vel(y=-3.0)),
        (0.1, _pose(y=2.5), _vel(y=0.0)),  # velocity-defined contact
        (0.2, _pose(y=3.0), _vel(y=1.0)),
        (0.3, _pose(y=3.0), _vel(y=0.0)),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=(-0.5, -0.5, -0.5),
        bbox_max_local=(0.5, 0.5, 0.5),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(-0.5, abs=1e-6)


def test_max_bounce_height_velocity_threshold_knobs_affect_detection() -> None:
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=1.5), _vel(y=-3.0)),
        (0.1, _pose(y=0.5), _vel(y=0.0)),
        (0.2, _pose(y=1.08), _vel(y=1.0)),
        (0.3, _pose(y=1.08), _vel(y=0.0)),
    ]
    bbox_min = (-0.5, -0.5, -0.5)
    bbox_max = (0.5, 0.5, 0.5)

    default_ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=bbox_min,
        bbox_max_local=bbox_max,
    )
    assert _metric_max_bounce_height(default_ctx) == pytest.approx(-0.58, abs=1e-6)

    strict_velocity_ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario(
            "max_bounce_height",
            target={
                "bounce_min_upward_velocity": 1.5,
            },
        ),
        bbox_min_local=bbox_min,
        bbox_max_local=bbox_max,
    )
    assert _metric_max_bounce_height(strict_velocity_ctx) == pytest.approx(
        0.0,
        abs=1e-6,
    )

    strict_downward_ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario(
            "max_bounce_height",
            target={
                "bounce_min_downward_velocity": 5.0,
            },
        ),
        bbox_min_local=bbox_min,
        bbox_max_local=bbox_max,
    )
    assert _metric_max_bounce_height(strict_downward_ctx) == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_max_bounce_height_ignores_geometry_contact_knobs() -> None:
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=1.5), _vel(y=-3.0)),
        (0.1, _pose(y=0.5), _vel(y=0.0)),
        (0.2, _pose(y=1.0), _vel(y=1.0)),
        (0.3, _pose(y=1.0), _vel(y=0.0)),
    ]
    bbox_min = (-0.5, -0.5, -0.5)
    bbox_max = (0.5, 0.5, 0.5)

    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario(
            "max_bounce_height",
            target={
                "bounce_contact_window_samples": 999,
                "bounce_contact_tolerance": 0.0,
                "ground_height": 999.0,
            },
        ),
        bbox_min_local=bbox_min,
        bbox_max_local=bbox_max,
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(
        -0.5,
        abs=1e-6,
    )


def test_max_bounce_height_ignores_near_ground_geometry_gate() -> None:
    rest = (0.0, 0.5, 0.0)
    trajectory = [
        (0.0, _pose(y=3.0), _vel(y=-3.0)),
        (0.1, _pose(y=2.5), _vel(y=0.0)),
        (0.2, _pose(y=2.8), _vel(y=1.0)),
        (0.3, _pose(y=2.8), _vel(y=0.0)),
    ]
    ctx = MetricContext(
        trajectory=trajectory,
        rest_position=rest,
        up_idx=1,
        scenario=_scenario("max_bounce_height"),
        bbox_min_local=(-0.5, -0.5, -0.5),
        bbox_max_local=(0.5, 0.5, 0.5),
    )
    assert _metric_max_bounce_height(ctx) == pytest.approx(-0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# Unknown metric guard
# ---------------------------------------------------------------------------


def test_evaluate_threads_bbox_into_bounce_metric_and_outputs(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The evaluator must bridge scene bbox metadata into the metric context."""
    from world_understanding.functions.physics import (
        trajectory as trajectory_mod,
    )

    from physics_agent import recording as recording_pkg
    from physics_agent.tuning import usd_patch as usd_patch_mod
    from physics_agent.tuning.scenarios import _scene_builder as scene_builder_mod
    from physics_agent.tuning.scenarios.drop_settle import evaluate

    trajectory = [
        (0.0, _pose(y=1.5), _vel(y=-3.0)),
        (0.1, _pose(y=0.5), _vel(y=0.0)),
        (0.2, _pose(y=1.0), _vel(y=1.0)),
        (0.3, _pose(y=1.0), _vel(y=0.0)),
    ]

    def _fake_build(_src: object, dst: object, **_kwargs: object) -> dict[str, object]:
        pathlib.Path(dst).write_bytes(b"")
        return {
            "body_pattern": "/Body",
            "body_prim_path": "/Body",
            "rest_position": [0.0, 0.5, 0.0],
            "world_up": [0.0, 1.0, 0.0],
            "drop_height_m_resolved": 0.5,
            "bbox_size_m": [1.0, 1.0, 1.0],
            "bbox_min_local_stage": [-0.5, -0.5, -0.5],
            "bbox_max_local_stage": [0.5, 0.5, 0.5],
            "camera_paths": [],
        }

    monkeypatch.setattr(
        usd_patch_mod,
        "patch_physics_usd",
        lambda src, dst, params: pathlib.Path(dst).write_bytes(b""),
    )
    monkeypatch.setattr(scene_builder_mod, "build_drop_settle_scene", _fake_build)
    monkeypatch.setattr(
        recording_pkg,
        "author_trajectory_usda",
        lambda scene, traj, body, out, fps, **kwargs: pathlib.Path(out).write_bytes(
            b""
        ),
    )
    monkeypatch.setattr(
        recording_pkg,
        "author_trajectory_jsonl",
        lambda traj, out, fps, **kwargs: pathlib.Path(out).write_text(""),
    )
    monkeypatch.setattr(
        trajectory_mod, "settle_distance", lambda traj, rest_position: 0.0
    )

    class _FakeDaemon:
        def evaluate(self, **kwargs: object) -> dict[str, object]:
            return {
                "trajectory": trajectory,
                "final_pose": _pose(y=1.0),
            }

    physics_usd = tmp_path / "physics.usda"
    physics_usd.write_bytes(b"")

    result = evaluate(
        params={},
        scenario=_scenario(metric="max_bounce_height"),
        physics_usd=physics_usd,
        seed=0,
        simulator=_FakeDaemon(),  # type: ignore[arg-type]
        work_dir=tmp_path / "work",
    )

    assert result["score"] == pytest.approx(-0.5, abs=1e-6)
    assert result["max_bounce_height"] == pytest.approx(0.5, abs=1e-6)
    assert result["first_bounce_height"] == pytest.approx(0.5, abs=1e-6)
    assert result["bbox_min_local_stage"] == [-0.5, -0.5, -0.5]
    assert result["bbox_max_local_stage"] == [0.5, 0.5, 0.5]
    assert result["world_up"] == [0.0, 1.0, 0.0]


def test_evaluate_rejects_unsupported_metric(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """drop_settle.evaluate raises ValueError when scenario.metric isn't in
    _METRICS — silent fallback would make artifacts report a metric the run
    didn't actually optimize, masking LLM/typo misconfigurations."""
    from world_understanding.functions.physics import (
        trajectory as trajectory_mod,
    )

    from physics_agent import recording as recording_pkg
    from physics_agent.tuning import usd_patch as usd_patch_mod
    from physics_agent.tuning.scenarios import _scene_builder as scene_builder_mod
    from physics_agent.tuning.scenarios.drop_settle import evaluate

    # Stub side effects so we never touch real USD/daemon code.
    def _fake_build(_src: object, dst: object, **_kwargs: object) -> dict[str, object]:
        pathlib.Path(dst).write_bytes(b"")  # type: ignore[arg-type]
        return {
            "body_pattern": "/Body",
            "body_prim_path": "/Body",
            "rest_position": [0.0, 0.0, 0.0],
            "drop_height_m_resolved": 0.05,
            "bbox_size_m": 0.1,
            "camera_paths": [],
        }

    monkeypatch.setattr(
        usd_patch_mod,
        "patch_physics_usd",
        lambda src, dst, params: pathlib.Path(dst).write_bytes(b""),
    )
    monkeypatch.setattr(scene_builder_mod, "build_drop_settle_scene", _fake_build)
    monkeypatch.setattr(
        recording_pkg,
        "author_trajectory_usda",
        lambda scene, traj, body, out, fps, **kwargs: pathlib.Path(out).write_bytes(
            b""
        ),
    )
    monkeypatch.setattr(
        trajectory_mod, "settle_distance", lambda traj, rest_position: 0.0
    )

    class _FakeDaemon:
        def evaluate(self, **kwargs: object) -> dict[str, object]:
            return {
                "trajectory": [(0.0, [0.0] * 7, [0.0] * 6)],
                "final_pose": [0.0] * 7,
            }

    bad_scenario = _scenario(metric="not_a_real_metric")
    physics_usd = tmp_path / "physics.usda"
    physics_usd.write_bytes(b"")

    with pytest.raises(ValueError, match="Unsupported drop_settle metric"):
        evaluate(
            params={},
            scenario=bad_scenario,
            physics_usd=physics_usd,
            seed=0,
            simulator=_FakeDaemon(),  # type: ignore[arg-type]
            work_dir=tmp_path / "work",
        )
