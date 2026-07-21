# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-agnostic per-trial simulator protocol.

Both :class:`~world_understanding.functions.physics.ovphysx_daemon._OvPhysXDaemon`
(daemon-isolated PhysX 5) and :class:`physics_agent.tuning.newton_simulator.NewtonSimulator`
(in-process Warp + MuJoCo) satisfy this protocol structurally. Scenario
evaluators (``drop_settle`` and ``freeform``) call ``simulator.evaluate(...)``
exactly once per trial, regardless of which backend the runner selected.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Simulator(Protocol):
    """Per-trial physics simulator contract.

    Implementations MUST return a dict with the following keys:

    * ``trajectory``: ``list[tuple[float, list[float], list[float]]]``
      i.e. ``[(t_s, [px,py,pz,qx,qy,qz,qw], [vx,vy,vz,wx,wy,wz]), ...]``.
      Velocity is in world-frame linear-then-angular order. Quaternion is
      ``(x, y, z, w)``.
    * ``final_pose``: ``[px,py,pz,qx,qy,qz,qw]`` of the last sample.
    * ``final_velocity``: ``[vx,vy,vz,wx,wy,wz]`` of the last sample.
    * ``n_bodies``: ``int`` — number of rigid bodies in the scene.
    * ``duration_s``: ``float`` — actual simulated duration.
    * ``n_steps``: ``int`` — number of solver steps taken.

    The trajectory shape matches what
    :func:`physics_agent.recording.recorder.author_trajectory_usda` expects
    so scenarios author ``recording.usd`` + ``trajectory.jsonl`` uniformly
    across engines. This matches the OvPhysX rigid-body velocity tensor and
    Newton's public ``body_qd`` convention.
    """

    def evaluate(
        self,
        *,
        scene_usd: Path,
        body_pattern: str,
        duration_s: float,
        dt: float = 1.0 / 240.0,
        sample_fps: int = 30,
        initial_linear_velocity: Sequence[float] | None = None,
        initial_angular_velocity: Sequence[float] | None = None,
    ) -> dict[str, Any]: ...
