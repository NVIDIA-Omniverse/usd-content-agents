# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""freeform scenario evaluator: hybrid programmatic + VLM judge.

Pipeline (per ``OvPhysXBackend.evaluate`` trial):

  1. ``patch_physics_usd`` (existing utility).
  2. ``build_freeform_scene`` reads ``target.{gravity, surface,
     initial_pose, cameras}`` and authors the simulation USD.
  3. Daemon ``evaluate`` with ``initial_linear_velocity`` and
     ``initial_angular_velocity`` from ``target``.
  4. ``recorder.author_trajectory_usda`` → ``recording.usd``.
  5. Programmatic score from ``trajectory_summary`` against
     ``target.observations`` (e.g. "settled within 1s").
  6. Optional VLM judge over rendered frames
     (``judge_callback(frames, user_prompt, observations)``).
  7. Combined score = ``weights["programmatic"] * programmatic_score +
     weights["vlm"] * vlm_score``. Default weights 0.5/0.5. Optimizer
     minimizes, so we return ``score = 1.0 - combined``.

Freeform is **NOT VLM-only**: programmatic trajectory metrics are
always computed, and they're the entire score when no VLM callback is
supplied (weights re-normalize). The VLM is one signal among several,
not the only driver.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from physics_agent.tuning.simulator import Simulator
    from physics_agent.tuning.types import Scenario

from physics_agent.tuning.video_rendering import resolve_video_renderer

logger = logging.getLogger(__name__)


# Contact solvers and collider/render-geometry mismatches can leave a small,
# scale-dependent overlap at rest. Keep the allowance relative to the selected
# body instead of applying one world-unit threshold to every asset scale.
_GROUND_CLEARANCE_BBOX_DIAGONAL_FRACTION = 0.02


# (frames, user_prompt | None, observations) -> {score: float in [0,1],
#                                                 reasoning: str}
JudgeCallback = Callable[[list[Path], str | None, list[str]], dict[str, Any]]


def _world_up_axis(world_up: Any) -> int:
    """Return the dominant up-axis index, defaulting to legacy Y-up."""
    try:
        values = [abs(float(v)) for v in list(world_up)[:3]]
    except (TypeError, ValueError):
        return 1
    if len(values) != 3 or not any(values):
        return 1
    return max(range(3), key=lambda idx: values[idx])


def _pose7_from_trajectory_sample(
    sample: Any,
) -> tuple[float, float, float, float, float, float, float] | None:
    """Extract pose7 from simulator trajectory samples.

    The normal simulator shape is ``(t, pose7, vel6)``, but a few tests
    and fakes use dict-ish samples. Keep this permissive so ground-clearance
    scoring can degrade to "not available" rather than failing the trial.
    """
    pose: Any
    if isinstance(sample, dict):
        pose = sample.get("pose") or sample.get("position")
    else:
        try:
            pose = sample[1]
        except (TypeError, IndexError):
            return None
    try:
        if len(pose) >= 7:
            return tuple(float(pose[i]) for i in range(7))  # type: ignore[return-value]
        if len(pose) >= 3:
            return (
                float(pose[0]),
                float(pose[1]),
                float(pose[2]),
                0.0,
                0.0,
                0.0,
                1.0,
            )
    except (TypeError, ValueError, IndexError):
        return None
    return None


def _as_vec3(raw: Any) -> tuple[float, float, float] | None:
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError, IndexError):
        return None


def _ground_clearance_tolerance(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
) -> float | None:
    """Return 2% of a valid bbox diagonal, in the bbox's own units.

    Clearance and the local bbox both come from the simulator in stage units.
    Keeping the dimensionless relative calculation in those units avoids
    mixing trajectory stage units with the user-facing ``bbox_size_m`` value.
    """
    extents = tuple(bbox_max[i] - bbox_min[i] for i in range(3))
    if any(not math.isfinite(extent) or extent < 0.0 for extent in extents):
        return None
    diagonal = math.hypot(*extents)
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        return None
    return _GROUND_CLEARANCE_BBOX_DIAGONAL_FRACTION * diagonal


def _add_ground_clearance_to_summary(
    summary: dict[str, Any],
    trajectory: Any,
    scene_info: dict[str, Any],
) -> None:
    """Annotate summary with clearance and its scale-aware tolerance.

    ``trajectory_summary`` only knows about the rigid-body origin, not the
    body's local bbox. For floor/ground objectives we need the bbox bottom:
    pose[up] + bbox_min_local_stage[up]. The scene builder authors the
    ground plane at up-coordinate 0. When both bbox corners are valid, the
    summary also records the 2%-of-diagonal tolerance in the same stage units.
    """
    bbox_min = _as_vec3(scene_info.get("bbox_min_local_stage"))
    if bbox_min is None:
        return
    up_idx = _world_up_axis(scene_info.get("world_up"))
    bbox_max = _as_vec3(scene_info.get("bbox_max_local_stage"))
    clearance_tolerance_stage = (
        _ground_clearance_tolerance(bbox_min, bbox_max)
        if bbox_max is not None
        else None
    )
    bottom_positions_from_bbox = None
    if bbox_max is not None:
        from physics_agent.tuning.scenarios.drop_settle import (
            _bottom_positions_from_bbox,
        )

        bottom_positions_from_bbox = _bottom_positions_from_bbox

    clearances: list[float] = []
    for sample in trajectory or []:
        pose7 = _pose7_from_trajectory_sample(sample)
        if pose7 is None:
            continue
        if bbox_max is None or bottom_positions_from_bbox is None:
            clearance = pose7[up_idx] + bbox_min[up_idx]
        else:
            clearance = bottom_positions_from_bbox(
                [pose7],
                up_idx=up_idx,
                bbox_min_local=bbox_min,
                bbox_max_local=bbox_max,
            )[0]
        if math.isfinite(clearance):
            clearances.append(float(clearance))
    if clearances:
        summary["min_ground_clearance"] = min(clearances)
        if clearance_tolerance_stage is not None:
            summary["ground_clearance_tolerance"] = clearance_tolerance_stage


def _normalize_observations(raw: Any) -> list[str]:
    """Normalize a YAML ``observations`` value into a list of strings.

    A YAML scalar like ``observations: "steady"`` is parsed as a Python
    ``str``; ``list(str)`` would explode it into its characters and
    silently feed garbage tokens to the VLM judge_callback. Treat a scalar
    str as a single
    observation, a list / tuple as multiple, ``None`` / missing as
    empty, and any other shape as a one-item list of its repr (so the
    user still sees *something* in artifacts rather than a silent drop).
    (CodeRabbit R13 thread #7.)
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list | tuple):
        return [str(item) for item in raw]
    return [str(raw)]


def _normalize_weights(
    weights: dict[str, float] | None, vlm_available: bool
) -> dict[str, float]:
    """Resolve weight defaults and re-normalize when VLM is unavailable.

    Default: ``{"programmatic": 0.5, "vlm": 0.5}``. When VLM is
    unavailable (no callback supplied or callback failed), we
    re-normalize to ``{"programmatic": 1.0, "vlm": 0.0}`` so the
    score is purely programmatic — no silent down-weighting.
    """
    base = {"programmatic": 0.5, "vlm": 0.5}
    if weights:
        # Validate up-front: unknown keys, negative or non-finite values
        # would silently distort the optimizer signal if we let the
        # later normalize/clamp absorb them. (CodeRabbit R13 thread #6.)
        unknown = set(weights) - set(base)
        if unknown:
            raise ValueError(
                "Unsupported freeform weight key(s) "
                f"{sorted(unknown)}; expected subset of {sorted(base)}."
            )
        for name, value in weights.items():
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(
                    f"Freeform weight {name!r} must be a number, "
                    f"got {type(value).__name__}: {value!r}"
                )
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(
                    f"Freeform weight {name!r} must be a finite non-negative "
                    f"number, got {value!r}"
                )
        base.update(weights)
    if not vlm_available:
        return {"programmatic": 1.0, "vlm": 0.0}
    total = float(base["programmatic"]) + float(base["vlm"])
    if total <= 0:
        raise ValueError(
            "At least one of freeform weights {'programmatic', 'vlm'} must be "
            f"positive; got programmatic={base['programmatic']!r}, "
            f"vlm={base['vlm']!r}."
        )
    return {
        "programmatic": float(base["programmatic"]) / total,
        "vlm": float(base["vlm"]) / total,
    }


def _score_programmatic_from_summary(
    summary: dict[str, Any], observations: list[str]
) -> tuple[float, str]:
    """Map ``trajectory_summary`` + ``observations`` to a score in [0,1]
    where 1.0 == fully-satisfies-prompt.

    v1 strategy is intentionally simple — it surfaces structural
    signals that almost any freeform observation cares about and
    leaves the nuance to the VLM. Each enabled component contributes
    its **weight** when it passes; the final score normalises by the
    sum of enabled weights:

      • body settled before the trajectory ended → weight 0.3
      • body did NOT escape the scene (no infinite / NaN positions) →
        weight 0.3
      • body did NOT penetrate the ground by more than 2% of its bbox
        diagonal when observations mention floor / ground / surface contact
        → conditional weight 0.6

    Each check is a hard yes/no. Returns ``(score, critique)`` so the
    surrounding evaluator can include the critique in its result dict
    for audit.
    """
    obs_text = " ".join(o.lower() for o in observations)
    final_pos = summary.get("final_position") or [0.0, 0.0, 0.0]
    settle_time_s = summary.get("settle_time_s")
    duration_s = float(summary.get("duration_s") or 0.0)
    n_samples = int(summary.get("n_samples") or 0)

    # (name, passed, weight) — weights match the documented contract.
    components: list[tuple[str, bool, float]] = []
    ground_audit: tuple[str, str] | None = None

    # Component 1: settled before trajectory ended
    settled = settle_time_s is not None and float(settle_time_s) <= duration_s
    components.append(("settled", bool(settled), 0.3))

    # Component 2: position is finite (no NaN/Inf escape).
    # ``math`` is imported at module level (see line 30).
    finite = all(math.isfinite(float(v)) for v in final_pos)
    components.append(("finite_position", bool(finite), 0.3))

    # Component 3: floor/ground contact must not materially penetrate the
    # authored ground plane. This is conditional so prompts about unusual
    # open-air motion don't inherit a floor assumption they never asked for.
    cares_about_ground = any(
        kw in obs_text
        for kw in (
            "floor",
            "ground",
            "surface",
            "table",
            "sink",
            "penetrat",
            "clip",
            "intersect",
        )
    )
    min_ground_clearance = summary.get("min_ground_clearance")
    if cares_about_ground and min_ground_clearance is not None:
        try:
            clearance = float(min_ground_clearance)
        except (TypeError, ValueError):
            clearance = float("-inf")
        try:
            # A zero fallback keeps legacy or partial summaries conservative:
            # only runtime summaries with a valid selected-body bbox receive
            # the scale-aware negative-clearance allowance.
            tolerance = float(summary.get("ground_clearance_tolerance", 0.0))
        except (TypeError, ValueError):
            tolerance = float("nan")
        tolerance_valid = math.isfinite(tolerance) and tolerance >= 0.0
        ground_ok = (
            math.isfinite(clearance) and tolerance_valid and clearance >= -tolerance
        )
        ground_audit = (
            f"{clearance:.6g}" if math.isfinite(clearance) else "invalid",
            f"{tolerance:.6g}" if tolerance_valid else "invalid",
        )
        components.append(("ground_clearance", ground_ok, 0.6))

    if not components or n_samples == 0:
        return 0.0, "no programmatic signal extracted"

    total_weight = sum(weight for _, _, weight in components)
    earned = sum(weight for _, ok, weight in components if ok)
    # ``total_weight`` is always > 0 here (we always append settled +
    # finite_position above), but guard divide-by-zero anyway.
    score = earned / total_weight if total_weight > 0 else 0.0
    critique_parts: list[str] = []
    for name, ok, _ in components:
        component_critique = f"{name}={'pass' if ok else 'fail'}"
        if name == "ground_clearance" and ground_audit is not None:
            clearance_text, tolerance_text = ground_audit
            component_critique += (
                f"(clearance={clearance_text}, tolerance={tolerance_text})"
            )
        critique_parts.append(component_critique)
    critique = "; ".join(critique_parts)
    return float(score), critique


def evaluate(
    params: dict[str, float],
    scenario: Scenario,
    physics_usd: Path,
    *,
    seed: int,
    simulator: Simulator,
    work_dir: Path | None = None,
    judge_callback: JudgeCallback | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run one freeform tune trial against a physics simulator.

    See module docstring for the pipeline. Returns the dict shape
    consumed by ``backend.evaluate``: ``score`` (lower is better),
    ``programmatic_score``, ``vlm_score``, ``reasoning``, ``frames``,
    ``trajectory``, ``scene_usd``, ``recording_usd``, legacy
    ``recording_usda``, ``weights_used``.
    """
    from world_understanding.functions.physics.trajectory import (
        trajectory_summary,
    )

    from physics_agent.recording import author_trajectory_usda
    from physics_agent.tuning.scenario_resolution import get_resolved_bindings
    from physics_agent.tuning.scenarios._scene_builder import (
        build_freeform_scene,
    )
    from physics_agent.tuning.usd_patch import patch_physics_usd

    target = dict(scenario.target or {})
    record_video_mode = str(target.get("record_video", "off")).lower()
    record_video_on = record_video_mode in {"end_of_tune", "always"}
    needs_render = (judge_callback is not None) or record_video_on
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

    # 2. Build scene USD from the target dict.
    duration_s = float(target.get("duration_s", 2.0))
    sample_fps = int(target.get("sample_fps", 30))

    scene_path = trial_dir / "scene.usd"
    scene_info = build_freeform_scene(
        patched_path,
        scene_path,
        target=target,
    )

    # 3. Simulator evaluate with the LLM-authored initial conditions.
    init_lin = target.get("initial_velocity")
    init_ang = target.get("initial_angular_velocity")
    response = simulator.evaluate(
        scene_usd=scene_path,
        body_pattern=scene_info["body_pattern"],
        duration_s=duration_s,
        dt=float(target.get("dt", 1.0 / 240.0)),
        sample_fps=sample_fps,
        initial_linear_velocity=tuple(init_lin) if init_lin else None,
        initial_angular_velocity=tuple(init_ang) if init_ang else None,
    )
    trajectory = response["trajectory"]

    # 4. Recording USD for VLM and audit.
    recording_path: Path | None = trial_dir / "recording.usd"
    try:
        author_trajectory_usda(
            scene_path,
            trajectory,
            scene_info["body_prim_path"],
            recording_path,
            fps=sample_fps,
            max_duration_s=duration_s,
        )
    except Exception as exc:
        logger.warning(
            "freeform: failed to author recording.usd (seed=%d): %s",
            int(seed),
            exc,
        )
        recording_path = None

    # 5. Programmatic score from trajectory summary + observations.
    # The simulator trajectory is already (t, pose7, vel6) tuples — pass
    # through unchanged; trajectory_summary reads velocity directly
    # from each sample.
    #
    # Round 12 (CX P2#4): pass the stage's actual up-axis through to
    # ``trajectory_summary``. ``trajectory_summary`` defaults to Y-up
    # when ``world_up`` is omitted, which mis-classifies a yaw spin on
    # a Z-up asset as ``fell_over=True`` in the diagnostic summary. The
    # scene builder records the actual axis under ``scene_info["world_up"]``;
    # if it's missing for any reason (programmatic-only callers building
    # scene_info dicts by hand) we fall back to the trajectory_summary default.
    observations = _normalize_observations(target.get("observations"))
    world_up = scene_info.get("world_up")
    summary = trajectory_summary(trajectory, world_up=world_up)
    _add_ground_clearance_to_summary(summary, trajectory, scene_info)
    programmatic_score, prog_critique = _score_programmatic_from_summary(
        summary, observations
    )

    # 6. Optional video rendering and VLM judge.
    #
    # Rendering produces inspection-ready PNG/mp4 evidence and is the
    # input to the optional VLM judge. The two are independent triggers:
    # ``record_video`` ("off" / "always", default "off") writes the
    # render artifacts unconditionally on every trial it fires; the VLM
    # judge runs only when ``judge_callback`` was passed and rendering
    # produced frames. Without a record_video opt-in, the VLM judge
    # itself implicitly forces a render so existing behavior is
    # preserved when callers wire up the callback.
    #
    # Rendering belongs to ``world_understanding.functions.graphics``;
    # imported lazily so freeform stays usable when the helper isn't
    # installed (PR #66). When rendering is unavailable the VLM step
    # is skipped cleanly and the score collapses to programmatic-only
    # via ``_normalize_weights(..., vlm_available=False)``.
    vlm_score: float | None = None
    vlm_reasoning: str = ""
    frames: list[Path] = []
    vlm_available = False
    video_block: dict[str, Any] | None = None
    if needs_render and recording_path is not None:
        try:
            from world_understanding.functions.graphics import (
                render_time_sampled_usd,
            )
        except ImportError:
            render_time_sampled_usd = None  # type: ignore[assignment]
            skip_reason = (
                "render_time_sampled_usd is not installed (see issue #50 / PR #66)"
            )
            if judge_callback is not None:
                vlm_reasoning = (
                    f"VLM unavailable: {skip_reason}; freeform falls back to "
                    "programmatic-only scoring."
                )
            if record_video_on:
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
                    fps=sample_fps,
                    max_duration_seconds=duration_s or 2.0,
                    image_width=int(target.get("video_image_width", 512)),
                    image_height=int(target.get("video_image_height", 512)),
                    num_sensor_updates=int(target.get("video_sensor_updates", 32)),
                    render_mode=str(target.get("video_render_mode", "rt2")),
                )
                if record_video_on:
                    video_block = {
                        "mode": record_video_mode,
                        "status": "ok" if frames else "no_frames",
                        "render_dir": str(trial_dir / "render"),
                        "frame_count": len(frames),
                    }
                if judge_callback is not None and frames:
                    verdict = judge_callback(
                        frames,
                        target.get("description"),
                        observations,
                    )
                    raw_score = verdict.get("score")
                    if isinstance(raw_score, int | float):
                        vlm_score = max(0.0, min(1.0, float(raw_score)))
                        vlm_reasoning = str(verdict.get("reasoning") or "")
                        vlm_available = True
            except Exception as exc:
                # Same intent as drop_settle: capture the exception
                # message in addition to the type, and log the
                # traceback, so silent render failures (frame-cap
                # mismatch, missing camera, ovrtx returning 0 images)
                # surface in history.jsonl + server logs.
                logger.exception(
                    "freeform: render/judge failed for trial seed=%d",
                    seed,
                )
                if judge_callback is not None:
                    vlm_reasoning = f"VLM unavailable: {type(exc).__name__}: {exc}"
                    vlm_available = False
                if record_video_on:
                    video_block = {
                        "mode": record_video_mode,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }

    # 7. Combine weighted, then map to "lower is better" optimizer
    # objective: score = 1.0 - combined.
    used_weights = _normalize_weights(weights, vlm_available)
    if vlm_available and vlm_score is not None:
        combined = (
            used_weights["programmatic"] * programmatic_score
            + used_weights["vlm"] * vlm_score
        )
    else:
        combined = programmatic_score
    combined_clamped = max(0.0, min(1.0, float(combined)))
    score = 1.0 - combined_clamped

    out: dict[str, Any] = {
        "score": float(score),
        "combined_score": float(combined_clamped),
        "programmatic_score": float(programmatic_score),
        "programmatic_critique": prog_critique,
        "vlm_score": float(vlm_score) if vlm_score is not None else None,
        "reasoning": vlm_reasoning,
        "frames": [str(p) for p in frames],
        "trajectory": trajectory,
        "trajectory_summary": summary,
        "scene_usd": str(scene_path),
        "patched_usd": str(patched_path),
        "recording_usd": str(recording_path) if recording_path else None,
        "recording_usda": str(recording_path) if recording_path else None,
        "weights_used": used_weights,
        "metric": str(scenario.metric),
    }
    if video_block is not None:
        out["record_video"] = video_block
    return out


__all__ = ["evaluate", "JudgeCallback"]
