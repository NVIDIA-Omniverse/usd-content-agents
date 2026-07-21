# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""drop_settle scenario evaluator.

Pipeline (per ``OvPhysXBackend.evaluate`` trial):

  1. ``patch_physics_usd`` → ``patched_physics.usd`` (existing utility,
     applies resolved USD-backed tuning parameters).
  2. ``build_drop_settle_scene`` → ``scene.usd``
     (patched body + ground plane + UsdPhysics.Scene + camera; body
     translated so its bottom is at ``drop_height_m`` above the
     ground; default == bbox_height).
  3. Daemon ``evaluate(scene, body_pattern, duration_s, dt, sample_fps)``
     returns trajectory JSON.
  4. ``recorder.author_trajectory_usda`` → ``recording.usd`` with
     ``Sdf.TimeCode`` time samples.
  5. Programmatic score = METRIC_REGISTRY[scenario.metric](context)
     (default: ``settle_distance(trajectory, rest_position)``).
  6. Optional ``final_state_judge`` per ``target.vlm_check``
     (``"off" | "end_of_tune" | "always"``). VLM is NEVER the
     drop_settle objective — the programmatic metric is authoritative;
     VLM only attaches a verdict to the artifact when enabled.

The result dict matches the ``backend.evaluate`` contract: ``score`` is
the metric output (lower is better — bouncy-style metrics like
``max_bounce_height`` are negated internally so the optimizer can keep
minimising); plus auxiliary fields for artifact persistence.

Adding a new metric is a one-liner:
    1. Author a ``def my_metric(ctx: MetricContext) -> float:`` that
       returns a scalar where lower-is-better.
    2. Register it in ``_METRICS`` below.
    3. Reference it in the scenario YAML's ``metric:`` key.

``ctx`` contains the trajectory, ``rest_position`` (stage units), the
stage up-axis index inferred from ``rest_position``, and the parsed
``Scenario`` (so metrics that want to pick up extra YAML knobs can read
them from ``ctx.scenario.target``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from physics_agent.tuning.simulator import Simulator
    from physics_agent.tuning.types import Scenario

from physics_agent.tuning.video_rendering import resolve_video_renderer

logger = logging.getLogger(__name__)


# Callback shape for the optional VLM "did this look right" check.
# Signature mirrors ``physics_agent.tuning.scenarios.freeform`` for
# consistency: (frames, user_prompt | None, observations) -> dict
# with ``score`` in [0, 1] and ``reasoning`` text.
FinalStateJudgeCallback = Callable[[list[Path], str | None, list[str]], dict[str, Any]]


# ---------------------------------------------------------------------------
# Metric registry — scenario.metric → callable(ctx) → float (lower is better)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricContext:
    """Inputs available to a drop_settle metric callable.

    Metrics return a scalar where **lower is better** (the optimizer
    minimises). Quantities that are physically "more is better" (peak
    bounce height, peak speed) should return their negation so the
    optimiser drives toward larger physical values.

    Attributes:
        trajectory: The simulator's trajectory shape ``[(t, pose7, vel6), ...]``.
        rest_position: Expected rest position in stage units, length 3.
            The non-zero axis identifies the stage's up-axis.
        up_idx: Inferred stage up-axis index (1=Y-up, 2=Z-up). Derived
            from ``rest_position`` so metrics don't reopen the stage.
        scenario: Parsed Scenario — gives metrics access to the YAML
            ``target`` block for any per-metric knobs.
        bbox_min_local: Optional body-space bbox minimum in stage units.
            When present, bounce metrics transform bbox corners by each
            sampled pose so contact/apex are measured from the body's bottom
            instead of the rigid-body origin.
        bbox_max_local: Optional body-space bbox maximum in stage units.
    """

    trajectory: Any
    rest_position: tuple[float, float, float]
    up_idx: int
    scenario: Scenario
    bbox_min_local: tuple[float, float, float] | None = None
    bbox_max_local: tuple[float, float, float] | None = None


def _metric_settle_distance(ctx: MetricContext) -> float:
    """Default metric — Euclidean distance from final pose to ``rest_position``.

    Lower is better. Reference implementation in
    ``world_understanding.functions.physics.trajectory.settle_distance``.
    """
    from world_understanding.functions.physics.trajectory import settle_distance

    return float(settle_distance(ctx.trajectory, rest_position=ctx.rest_position))


def _bbox_corners(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (x, y, z)
        for x in (bbox_min[0], bbox_max[0])
        for y in (bbox_min[1], bbox_max[1])
        for z in (bbox_min[2], bbox_max[2])
    )


def _rotate_vector_by_pose_quat(
    vector: tuple[float, float, float],
    pose: Any,
) -> tuple[float, float, float]:
    """Rotate ``vector`` by pose7's quaternion.

    Pose quaternions are ``[qx, qy, qz, qw]``. The formula is the standard
    unit-quaternion vector rotation ``v + qw*t + cross(q_xyz, t)`` with
    ``t = 2 * cross(q_xyz, v)``.
    """

    qx, qy, qz, qw = (float(pose[i]) for i in range(3, 7))
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0 or not math.isfinite(norm):
        return vector
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _bottom_positions_from_bbox(
    poses: Any,
    *,
    up_idx: int,
    bbox_min_local: tuple[float, float, float] | None,
    bbox_max_local: tuple[float, float, float] | None,
) -> list[float]:
    """Return bbox-bottom height for every pose, falling back to origin height."""

    if bbox_min_local is None or bbox_max_local is None:
        return [float(pose[up_idx]) for pose in poses]

    corners = _bbox_corners(bbox_min_local, bbox_max_local)
    bottoms: list[float] = []
    for pose in poses:
        position = (float(pose[0]), float(pose[1]), float(pose[2]))
        bottom = float("inf")
        for corner in corners:
            rx, ry, rz = _rotate_vector_by_pose_quat(corner, pose)
            world_corner = (
                position[0] + rx,
                position[1] + ry,
                position[2] + rz,
            )
            bottom = min(bottom, world_corner[up_idx])
        bottoms.append(float(bottom))
    return bottoms


def _metric_max_bounce_height(ctx: MetricContext) -> float:
    """First rebound height measured from bbox bottom and up-axis velocity.

    The old implementation used rigid-body origin height to decide "contact".
    That is wrong for tire-like assets: the origin can remain far above the
    ground while the tire bottom is already in contact. This metric instead:

    1. tracks velocity along the stage up-axis,
    2. treats the first downward→non-downward transition as first contact,
    3. measures bbox-bottom clearance at the first upward→non-upward apex.

    The returned score is ``-first_bounce_height`` so the minimizer drives
    toward higher first rebounds. No rebound returns ``0.0``: finite, but not
    better than any positive bounce.
    """
    from world_understanding.functions.physics.trajectory import (
        _trajectory_to_arrays,
    )

    _, poses, velocities = _trajectory_to_arrays(ctx.trajectory)
    if poses.size == 0:
        return float("inf")
    if len(poses) < 2 or len(velocities) < 2:
        return float("inf")
    if len(poses) != len(velocities):
        return float("inf")

    up_idx = int(ctx.up_idx)
    target = dict(ctx.scenario.target or {})
    min_downward_v = float(target.get("bounce_min_downward_velocity", 0.05))
    min_upward_v = float(target.get("bounce_min_upward_velocity", 0.02))

    bottom_positions = _bottom_positions_from_bbox(
        poses,
        up_idx=up_idx,
        bbox_min_local=ctx.bbox_min_local,
        bbox_max_local=ctx.bbox_max_local,
    )
    v_up = [float(v[up_idx]) for v in velocities]

    contact_idx: int | None = None
    last_down_idx: int | None = None
    for i, velocity in enumerate(v_up):
        if velocity < -min_downward_v:
            last_down_idx = i
            continue
        if velocity >= 0.0 and last_down_idx is not None and last_down_idx < i:
            contact_idx = i
            break

    if contact_idx is None:
        return 0.0

    apex_idx: int | None = None
    saw_upward = v_up[contact_idx] > min_upward_v
    for i in range(contact_idx + 1, len(v_up)):
        if v_up[i] > min_upward_v:
            saw_upward = True
            continue
        if saw_upward and v_up[i] <= 0.0:
            apex_idx = i
            break
    if apex_idx is None:
        return 0.0

    first_bounce_height = max(
        0.0,
        float(bottom_positions[apex_idx] - bottom_positions[contact_idx]),
    )
    return -first_bounce_height


# Single source of truth for drop_settle metrics. Add a new entry to
# extend; the dispatch in ``evaluate`` reads this map verbatim.
_METRICS: dict[str, Callable[[MetricContext], float]] = {
    "settle_distance": _metric_settle_distance,
    "max_bounce_height": _metric_max_bounce_height,
}


def _infer_up_idx(rest_position: list[float] | tuple[float, ...]) -> int:
    """Return the up-axis index from ``scene_info["rest_position"]``.

    The scene builder writes the body's expected rest position with
    only the up-axis component non-zero. When the asset's local origin
    coincides with bbox-min the stored rest_up is exactly 0 (a
    corner-origin asset like the SimReady ladder), in which case we
    fall back to the convention of Y-up (1) — drop_settle's metric
    bake on the in-memory stage forces a consistent metric so this
    fallback only matters for degenerate inputs that wouldn't simulate
    correctly anyway.
    """
    if len(rest_position) < 3:
        return 1  # pragma: no cover — defensive
    # Pick the axis with the largest absolute value.
    best_idx = 1
    best_val = -1.0
    for i, v in enumerate(rest_position[:3]):
        if abs(float(v)) > best_val:
            best_val = abs(float(v))
            best_idx = i
    if best_val == 0.0:
        return 1  # default Y-up when rest_position is the origin
    return best_idx


def _resolve_up_idx(
    scene_info: dict[str, Any] | None,
    rest_position: list[float] | tuple[float, ...],
) -> int:
    """Pick the up-axis index, preferring the scene builder's authoritative value.

    The scene builder stashes ``world_up`` (a one-hot unit vector along the
    stage up-axis) on the returned ``scene_info`` so corner-origin Z-up
    assets — whose ``rest_position`` collapses to the origin — still report
    the correct axis. ``_infer_up_idx`` is only consulted when ``world_up``
    is absent (older callers / tests that stub ``scene_info``).
    """
    if isinstance(scene_info, dict):
        wu = scene_info.get("world_up")
        if isinstance(wu, list | tuple) and len(wu) >= 3:
            best_idx = -1
            best_val = -1.0
            for i in range(3):
                try:
                    v = abs(float(wu[i]))
                except (TypeError, ValueError):
                    continue
                if v > best_val:
                    best_val = v
                    best_idx = i
            if best_idx >= 0 and best_val > 0.0:
                return best_idx
    return _infer_up_idx(rest_position)


def evaluate(
    params: dict[str, float],
    scenario: Scenario,
    physics_usd: Path,
    *,
    seed: int,
    simulator: Simulator,
    work_dir: Path | None = None,
    final_state_judge: FinalStateJudgeCallback | None = None,
) -> dict[str, Any]:
    """Run one drop_settle tune trial against a physics simulator.

    Args:
        params: Tuned parameter values (mass_scale, friction, ...).
        scenario: Parsed Scenario; reads ``target.gravity``,
            ``target.duration_s``, ``target.drop_height_m`` (the GAP
            between body bottom and ground), ``target.vlm_check``.
        physics_usd: Path to the apply_physics output USD.
        seed: Per-trial seed (used by the runner to derive the per-step
            wall-time inside the simulator).
        simulator: Engine-agnostic per-trial simulator (``_OvPhysXDaemon``
            held by ``OvPhysXBackend`` or ``NewtonSimulator`` held by
            ``NewtonBackend`` — both satisfy the
            :class:`physics_agent.tuning.simulator.Simulator` protocol).
        work_dir: Where to write the patched USD, scene USD, and
            recording. Defaults to ``physics_usd.parent / ".tune_scenes"``.
        final_state_judge: Optional VLM judge invoked when
            ``target.vlm_check`` is ``"end_of_tune"`` or ``"always"``.

    Returns:
        Dict shaped for ``backend.evaluate`` consumption: ``score``,
        ``settle_distance``, ``final_position``, ``rest_position``,
        ``trajectory``, ``scene_usd``, ``patched_usd``, ``recording_usd``,
        legacy ``recording_usda``, ``trajectory_jsonl``, optionally ``vlm_check``.
    """
    from physics_agent.recording import (
        author_trajectory_jsonl,
        author_trajectory_usda,
    )
    from physics_agent.tuning.scenario_resolution import get_resolved_bindings
    from physics_agent.tuning.scenarios._scene_builder import (
        build_drop_settle_scene,
    )
    from physics_agent.tuning.usd_patch import patch_physics_usd

    target = dict(scenario.target or {})
    vlm_check_mode = str(target.get("vlm_check", "off")).lower()
    record_video_mode = str(target.get("record_video", "off")).lower()
    needs_render = (vlm_check_mode in {"end_of_tune", "always"}) or (
        record_video_mode in {"end_of_tune", "always"}
    )
    video_renderer = resolve_video_renderer(target) if needs_render else None

    work = (
        Path(work_dir)
        if work_dir is not None
        else Path(physics_usd).parent / ".tune_scenes"
    )
    trial_dir = work / f"trial_seed_{int(seed)}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # 1. Patch params.
    patched_path = trial_dir / "patched_physics.usd"
    resolved_bindings = get_resolved_bindings(scenario)
    if resolved_bindings is None:
        patch_physics_usd(Path(physics_usd), patched_path, dict(params))
    else:
        patch_physics_usd(
            Path(physics_usd),
            patched_path,
            dict(params),
            bindings=resolved_bindings,
        )

    # 2. Build scene USD with ground + camera + body @ drop_height_m gap.
    gravity = float(target.get("gravity", -9.81))
    duration_s = float(target.get("duration_s", 2.0))

    scene_path = trial_dir / "scene.usd"
    scene_info = build_drop_settle_scene(
        patched_path,
        scene_path,
        # drop_height_m=None → defaults to bbox_height (the GAP equals
        # the body's own height per the maintainer's framing).
        drop_height_m=target.get("drop_height_m"),
        gravity=gravity,
        ground_friction=float(target.get("ground_friction", 0.5)),
        # Pass ``target.cameras`` through unchanged so the scene
        # builder's corner-view default (``["+x+y+z"]``) applies when
        # the YAML / NL-authored target omits it. Hardcoding ``["-z"]``
        # here would defeat that default and produce top-down renders
        # on Z-up scenes.
        cameras=list(target.get("cameras")) if target.get("cameras") else None,
        camera_ground_bias_fraction=target.get("camera_ground_bias_fraction"),
    )

    # 3. Simulator evaluate (drop_settle has no initial velocity).
    response = simulator.evaluate(
        scene_usd=scene_path,
        body_pattern=scene_info["body_pattern"],
        duration_s=duration_s,
        dt=float(target.get("dt", 1.0 / 240.0)),
        sample_fps=int(target.get("sample_fps", 30)),
    )
    trajectory = response["trajectory"]

    # 4. Author recording.usd (issue #50) and trajectory.jsonl (the
    #    judge-readable companion — mirrors material-agent's
    #    predictions.jsonl pattern). Both are best-effort; failure of
    #    either never aborts tune. The trajectory-derived settle_distance
    #    is authoritative for the score regardless.
    fps_recording = int(target.get("sample_fps", 30))
    recording_path = trial_dir / "recording.usd"
    trajectory_jsonl_path: Path | None = trial_dir / "trajectory.jsonl"
    try:
        author_trajectory_usda(
            scene_path,
            trajectory,
            scene_info["body_prim_path"],
            recording_path,
            fps=fps_recording,
            max_duration_s=duration_s,
        )
    except Exception as exc:
        logger.warning(
            "drop_settle: failed to author recording.usd for trial seed=%d: %s",
            int(seed),
            exc,
        )
        recording_path = None  # type: ignore[assignment]
    try:
        author_trajectory_jsonl(
            trajectory,
            trajectory_jsonl_path,
            fps=fps_recording,
            max_duration_s=duration_s,
        )
    except Exception as exc:
        logger.warning(
            "drop_settle: failed to author trajectory.jsonl for trial seed=%d: %s",
            int(seed),
            exc,
        )
        trajectory_jsonl_path = None

    # 5. Programmatic metric — dispatch on scenario.metric. Default
    # ``settle_distance`` keeps PR #43 behaviour byte-identical;
    # alternatives like ``max_bounce_height`` plug in via _METRICS.
    # An unrecognized metric name is a user/LLM mistake we surface
    # loudly: a silent fallback to settle_distance would have the
    # artifacts report metric=<bogus_name> while the score actually
    # reflects settle_distance, masking the misconfiguration.
    rest_position = scene_info["rest_position"]
    metric_name = str(scenario.metric or "settle_distance")
    if metric_name not in _METRICS:
        raise ValueError(
            f"Unsupported drop_settle metric {metric_name!r}; "
            f"choose from {sorted(_METRICS)} or omit ``metric`` "
            "to default to 'settle_distance'."
        )
    metric_fn = _METRICS[metric_name]
    metric_ctx = MetricContext(
        trajectory=trajectory,
        rest_position=tuple(float(v) for v in rest_position[:3]),  # type: ignore[arg-type]
        up_idx=_resolve_up_idx(scene_info, rest_position),
        scenario=scenario,
        bbox_min_local=(
            tuple(float(v) for v in scene_info["bbox_min_local_stage"][:3])
            if "bbox_min_local_stage" in scene_info
            else None
        ),
        bbox_max_local=(
            tuple(float(v) for v in scene_info["bbox_max_local_stage"][:3])
            if "bbox_max_local_stage" in scene_info
            else None
        ),
    )
    score_value = float(metric_fn(metric_ctx))
    # ``settle_distance`` is also surfaced separately for backward
    # compatibility — older artifacts and tests inspect it by name.
    distance_value = float(_metric_settle_distance(metric_ctx))

    # 6. Optional video rendering and VLM check.
    #
    # ``vlm_check`` and ``record_video`` share the same render output but
    # have independent triggers. ``record_video`` exists for users who
    # want PNG/mp4 evidence to eyeball without paying for a VLM call —
    # rendering is the expensive geometric step and VLM is the optional
    # interpretive step on top.
    #
    # Mode strings (both fields):
    #   - "off"          : never invoke this branch
    #   - "end_of_tune"  : orchestrator-gated. The per-trial evaluator
    #                      treats this identically to "always" — there
    #                      is no winning-trial replay step inside
    #                      ``run_tune`` itself. ``IterativePhysics\
    #                      RefinementTask`` (driven by ``physics-agent
    #                      refine``) gates by passing
    #                      ``force_record_video="off"`` so per-trial
    #                      rendering is suppressed and the orchestrator
    #                      replays the winning trial after the sweep.
    #                      Standalone ``run_tune`` callers that set this
    #                      mode will render once per trial (N videos for
    #                      an N-trial sweep, dominating runtime/cost);
    #                      a one-time WARNING fires per-trial to keep
    #                      the misuse loud. (Codex CX R14 P2#2.)
    #   - "always"       : run on every trial
    #
    # Both default to "off". VLM is NEVER the drop_settle objective —
    # settle_distance above is authoritative — so an unavailable renderer
    # just drops the optional verdict + video.
    if vlm_check_mode == "end_of_tune" or record_video_mode == "end_of_tune":
        logger.warning(
            "drop_settle: vlm_check=%r / record_video=%r set to "
            "'end_of_tune' but the per-trial evaluator runs without "
            "winning-trial gating — rendering will fire on every trial. "
            "Use physics-agent refine / IterativePhysicsRefinementTask "
            "for true end-of-tune rendering (sets force_record_video='off' "
            "and replays the winner once).",
            vlm_check_mode,
            record_video_mode,
        )
    vlm_block: dict[str, Any] | None = None
    video_block: dict[str, Any] | None = None
    if needs_render and recording_path is not None:
        # Rendering belongs to ``world_understanding.functions.graphics``;
        # imported lazily so drop_settle stays usable when the helper
        # isn't installed (it ships in PR #66).
        try:
            from world_understanding.functions.graphics import (
                render_time_sampled_usd,
            )
        except ImportError:
            render_time_sampled_usd = None  # type: ignore[assignment]
            skip_reason = (
                "render_time_sampled_usd is not installed (see issue #50 / PR #66)"
            )
            if vlm_check_mode in {"end_of_tune", "always"}:
                vlm_block = {
                    "mode": vlm_check_mode,
                    "status": "skipped",
                    "reason": skip_reason,
                }
            if record_video_mode in {"end_of_tune", "always"}:
                video_block = {
                    "mode": record_video_mode,
                    "status": "skipped",
                    "reason": skip_reason,
                }

        if render_time_sampled_usd is not None:
            assert video_renderer is not None
            try:
                frames = render_time_sampled_usd(
                    recording_path,
                    trial_dir / "render",
                    renderer=video_renderer,
                    cameras=scene_info.get("camera_paths"),
                    fps=int(target.get("sample_fps", 30)),
                    max_duration_seconds=duration_s or 2.0,
                    image_width=int(target.get("video_image_width", 512)),
                    image_height=int(target.get("video_image_height", 512)),
                    num_sensor_updates=int(target.get("video_sensor_updates", 32)),
                    render_mode=str(target.get("video_render_mode", "rt2")),
                )
                if record_video_mode in {"end_of_tune", "always"}:
                    video_block = {
                        "mode": record_video_mode,
                        "status": "ok" if frames else "no_frames",
                        "render_dir": str(trial_dir / "render"),
                        "frame_count": len(frames),
                    }
                if (
                    final_state_judge is not None
                    and vlm_check_mode in {"end_of_tune", "always"}
                    and frames
                ):
                    # Just the final frame — cheap, single-image VLM check.
                    vlm_block = final_state_judge(
                        [frames[-1]],
                        None,  # drop_settle doesn't carry a user_prompt
                        list(
                            target.get("observations")
                            or ["object came to rest cleanly"]
                        ),
                    )
                    vlm_block["mode"] = vlm_check_mode
            except Exception as exc:
                # Capture the exception message in addition to the type so
                # downstream consumers (history.jsonl, judge prompts) can
                # tell ``ValueError: Selected 61 frames, exceeding cap of 60``
                # apart from ``ValueError: bad camera path`` apart from
                # ``RuntimeError: Renderer returned 0 image(s)``. Logging at
                # exception level so the traceback also lands in any
                # configured log handler.
                logger.exception(
                    "drop_settle: render/vlm-check failed for trial seed=%d",
                    seed,
                )
                error_block = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if vlm_check_mode in {"end_of_tune", "always"}:
                    vlm_block = {"mode": vlm_check_mode, **error_block}
                if record_video_mode in {"end_of_tune", "always"}:
                    video_block = {"mode": record_video_mode, **error_block}

    final_pose = list(response["final_pose"])
    final_position = final_pose[0:3]

    # ``world_up`` is the authoritative stage up-axis the scene
    # builder used; downstream consumers (notably the judge's
    # ``_best_trial_summary``) prefer this over inferring from
    # ``rest_position`` because corner-origin assets yield a
    # zero rest_position whose inferred up degenerates to legacy
    # Y-up. Older test harnesses that stub ``scene_info`` may omit
    # the field — fall back to the inferred-from-rest path silently.
    scene_world_up = (
        scene_info.get("world_up") if isinstance(scene_info, dict) else None
    )

    out: dict[str, Any] = {
        "score": score_value,
        "settle_distance": distance_value,
        "final_position": final_position,
        "rest_position": list(rest_position),
        "trajectory": trajectory,
        "scene_usd": str(scene_path),
        "patched_usd": str(patched_path),
        "recording_usd": str(recording_path) if recording_path else None,
        "recording_usda": str(recording_path) if recording_path else None,
        "trajectory_jsonl": (
            str(trajectory_jsonl_path) if trajectory_jsonl_path else None
        ),
        "drop_height_m": scene_info["drop_height_m_resolved"],
        "bbox_size_m": scene_info["bbox_size_m"],
        "metric": metric_name,
    }
    if "bbox_min_local_stage" in scene_info:
        out["bbox_min_local_stage"] = list(scene_info["bbox_min_local_stage"])
    if "bbox_max_local_stage" in scene_info:
        out["bbox_max_local_stage"] = list(scene_info["bbox_max_local_stage"])
    if scene_world_up is not None:
        out["world_up"] = list(scene_world_up)
    # Surface the raw bounce height (not negated) when the metric is
    # ``max_bounce_height`` — useful for the refine loop's monotonicity
    # check and for the report. ``max_bounce_height`` is retained as the
    # backward-compatible field name; ``first_bounce_height`` is the precise
    # diagnostic name for the same value until a separate true-max metric lands.
    # ``score`` stays negated for the optimizer.
    if metric_name == "max_bounce_height":
        out["max_bounce_height"] = -score_value
        out["first_bounce_height"] = -score_value
    if vlm_block is not None:
        out["vlm_check"] = vlm_block
    if video_block is not None:
        out["record_video"] = video_block
    return out


__all__ = [
    "evaluate",
    "FinalStateJudgeCallback",
    "MetricContext",
    "_METRICS",
]
