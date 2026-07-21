# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author a time-sampled ``recording.usd`` from an ovphysx trajectory.

Per issue #50: the simulation loop emits a list of
``(t_seconds, pose7, vel6)`` tuples (raw simulator output — position
+ orientation + linear/angular velocity). This module turns them into
a USD that:

* Reads the simulation scene USD, then exports a copy to
  ``output_path`` with body transforms / velocities authored as time
  samples. The source scene_usd is NOT modified on disk.
* Authors per-frame ``xformOp:translate`` (Vec3d) +
  ``xformOp:orient`` (Quatf) on the body prim — split form so usdview
  shows raw position/orientation values directly. Frame indices are
  integer (``0, 1, 2, ...``) so renderers can iterate ``frames="0:N"``.
* Authors per-frame ``physics:velocity`` and ``physics:angularVelocity``
  (``UsdPhysics.RigidBodyAPI``) — raw simulator velocities, NOT
  finite-differenced.
* Sets ``timeCodesPerSecond`` and the stage's ``startTimeCode`` /
  ``endTimeCode`` so renderers know the playback rate.

The recording carries **only raw simulator output**. Derived metrics
(max_linear_speed, settle_time, fell_over, etc.) are computed on
demand by ``world_understanding.functions.physics.trajectory`` from
the daemon's in-flight dict OR by reading them back from the USD via
``read_pose_velocity_trajectory`` — same numbers either way. The VLM
judge verdict lives in ``tune_results.json``, not the USD.

Hard cap from issue #50: ``fps <= 60``. Recording duration defaults to
2.0 seconds for legacy callers, but scenario evaluators pass their
configured ``duration_s`` so judge evidence can cover the full run.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

# Hard caps/defaults mandated by issue #50 spec.
_MAX_FPS = 60
_DEFAULT_MAX_DURATION_S = 2.0


def _validate_and_truncate(
    trajectory: Sequence[tuple[float, Sequence[float], Sequence[float]]],
    *,
    max_duration_s: float = _DEFAULT_MAX_DURATION_S,
) -> list[tuple[float, list[float], list[float]]]:
    """Validate 3-tuple shape and truncate at the duration cap.

    Shared by ``author_trajectory_usda`` and ``author_trajectory_jsonl`` so
    the USD and JSONL artifacts always cover the same span and reject the
    same legacy 2-tuple shape consistently.
    """
    duration_cap = float(max_duration_s)
    if not math.isfinite(duration_cap) or duration_cap <= 0.0:
        raise ValueError(f"max_duration_s must be finite and > 0, got {max_duration_s}")

    truncated: list[tuple[float, list[float], list[float]]] = []
    for entry in trajectory:
        if len(entry) != 3:
            raise ValueError(
                "trajectory entries must be (t, pose7, vel6) 3-tuples; got "
                f"length-{len(entry)}. Daemon must emit velocity per sample."
            )
        t, pose, vel = entry
        t_f = float(t)
        if t_f > duration_cap:
            logger.warning(
                "recording trajectory truncated at %.2fs duration cap",
                duration_cap,
            )
            break
        truncated.append((t_f, [float(v) for v in pose], [float(v) for v in vel]))
    return truncated


def _quantize_to_fps(
    truncated: list[tuple[float, list[float], list[float]]],
    fps: int,
) -> list[tuple[int, float, list[float], list[float]]]:
    """Quantize daemon samples onto the recording's frame grid (last-wins).

    Returns ``[(frame_index, t_quantized_seconds, pose, vel), ...]`` sorted
    by frame index. Mirrors the USD authoring path so the JSONL artifact
    aligns frame-for-frame with the recording.usd.
    """
    samples_by_frame: dict[int, tuple[list[float], list[float]]] = {}
    for t, pose, vel in truncated:
        frame = int(round(t * fps))
        samples_by_frame[frame] = (pose, vel)
    return [
        (f, float(f) / fps, samples_by_frame[f][0], samples_by_frame[f][1])
        for f in sorted(samples_by_frame)
    ]


def _clamp_fps(fps: int) -> int:
    fps_clamped = min(int(fps), _MAX_FPS)
    if fps_clamped <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if fps_clamped != int(fps):
        logger.warning(
            "recording fps clamped from %d to %d (issue #50 max)",
            fps,
            fps_clamped,
        )
    return fps_clamped


def author_trajectory_usda(
    scene_usd: Path,
    trajectory: Sequence[tuple[float, Sequence[float], Sequence[float]]],
    body_prim_path: str,
    output_path: Path,
    *,
    fps: int = 30,
    max_duration_s: float = _DEFAULT_MAX_DURATION_S,
) -> Path:
    """Author ``recording.usd`` and return its path.

    Args:
        scene_usd: Path to the simulation scene USD (the one fed to
            ovphysx). Opened in-memory; not modified on disk.
        trajectory: Sequence of ``(t_seconds, pose7, vel6)`` 3-tuples
            where ``pose7 = [px,py,pz,qx,qy,qz,qw]`` and
            ``vel6 = [vx,vy,vz,wx,wy,wz]``. Legacy ``(t, pose7)``
            2-tuples are rejected — the daemon emits velocity per
            sample now.
        body_prim_path: USD path of the rigid body prim. Must exist in
            ``scene_usd``.
        output_path: Where to write the recording. Parent directory is
            created if missing.
        fps: Recording frame rate. Capped at 60.
        max_duration_s: Recording duration cap. Defaults to 2.0 seconds
            for legacy callers; scenario evaluators should pass their
            configured ``target.duration_s``.

    Returns:
        ``output_path``.
    """
    from pxr import Sdf, Usd  # type: ignore[import-untyped]
    from world_understanding.utils.usd.time_samples import (
        add_pose_velocity_trajectory,
    )

    fps_clamped = _clamp_fps(fps)
    truncated = _validate_and_truncate(trajectory, max_duration_s=max_duration_s)
    if not truncated:
        raise ValueError(
            "trajectory is empty after duration cap; cannot author recording"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open the scene USD into an in-memory stage; we add time samples
    # to the body and Export to ``output_path`` without touching the
    # source on disk.
    rec_stage = Usd.Stage.Open(str(scene_usd))
    if rec_stage is None:
        raise ValueError(f"could not open scene_usd {scene_usd}")

    # Time-codes-per-second drives renderers' frame iteration.
    rec_stage.SetTimeCodesPerSecond(float(fps_clamped))
    rec_stage.SetFramesPerSecond(float(fps_clamped))

    # Quantize daemon samples onto the recording's frame grid (last-wins
    # on a frame index — mirrors USD's last-set semantics).
    #
    # **Downsampling note:** when ``sample_fps`` (daemon) > ``fps``
    # (recording, capped at 60 by issue #50), each recording frame
    # absorbs roughly ``sample_fps / fps`` daemon samples and only the
    # last one survives. The metric module derives quantities from raw
    # daemon trajectories before this downsample, so settle_time /
    # max_linear_speed see every daemon sample. The recording is
    # purely the persisted view; high-res analysis stays in the
    # in-flight trajectory dict.
    quantized = _quantize_to_fps(truncated, fps_clamped)
    sorted_frames = [q[0] for q in quantized]

    body_prim = rec_stage.GetPrimAtPath(Sdf.Path(body_prim_path))
    if not body_prim:
        raise ValueError(
            f"body_prim_path {body_prim_path!r} not present in scene_usd {scene_usd}"
        )

    # Author the full trajectory (translate + orient + physics:velocity
    # + physics:angularVelocity time samples in one pass). Times here
    # are seconds (= frame / fps); the helper multiplies by the stage's
    # tcps to get integer frame timecodes back.
    rec_traj = [(t_q, pose, vel) for _, t_q, pose, vel in quantized]
    add_pose_velocity_trajectory(body_prim, rec_traj)

    # Stage timecode range — set in TIMECODES (frame indices), not seconds.
    rec_stage.SetStartTimeCode(float(sorted_frames[0]))
    rec_stage.SetEndTimeCode(float(sorted_frames[-1]))

    rec_stage.GetRootLayer().Export(str(output_path))
    return output_path


def author_trajectory_jsonl(
    trajectory: Sequence[tuple[float, Sequence[float], Sequence[float]]],
    output_path: Path,
    *,
    fps: int = 30,
    max_duration_s: float = _DEFAULT_MAX_DURATION_S,
) -> Path:
    """Author a per-frame ``trajectory.jsonl`` and return its path.

    One JSON line per frame, sharing the same fps quantization and
    duration cap as ``author_trajectory_usda`` so the JSONL and USD
    artifacts cover the same span frame-for-frame. This is the
    judge-readable companion to ``recording.usd`` — mirrors the
    material-agent pattern where ``predict`` writes ``predictions.jsonl``
    that the judge's programmatic side ingests.

    Each line:

        {"frame": int, "t": float,
         "pose": [px, py, pz, qx, qy, qz, qw],
         "vel":  [vx, vy, vz, wx, wy, wz]}

    Field names match the daemon's wire format
    (``world_understanding.functions.physics.trajectory._trajectory_to_arrays``)
    so reconstructing ``[(t, pose, vel), ...]`` for ``trajectory_summary``
    is a trivial list comprehension.

    Args:
        trajectory: Same shape as ``author_trajectory_usda`` —
            ``[(t_seconds, pose7, vel6), ...]`` 3-tuples. Legacy
            ``(t, pose7)`` 2-tuples are rejected.
        output_path: Where to write the JSONL. Parent directory is
            created if missing.
        fps: Frame rate. Capped at 60. Daemon samples are quantized onto
            this grid (last-wins per frame).
        max_duration_s: Recording duration cap. Defaults to 2.0 seconds
            for legacy callers; scenario evaluators should pass their
            configured ``target.duration_s``.

    Returns:
        ``output_path``.
    """
    fps_clamped = _clamp_fps(fps)
    truncated = _validate_and_truncate(trajectory, max_duration_s=max_duration_s)
    if not truncated:
        raise ValueError(
            "trajectory is empty after duration cap; cannot author trajectory.jsonl"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantized = _quantize_to_fps(truncated, fps_clamped)
    with output_path.open("w", encoding="utf-8") as f:
        for frame, t_q, pose, vel in quantized:
            f.write(
                json.dumps(
                    {
                        "frame": int(frame),
                        "t": float(t_q),
                        "pose": pose,
                        "vel": vel,
                    }
                )
                + "\n"
            )
    return output_path


__all__ = ["author_trajectory_jsonl", "author_trajectory_usda"]
