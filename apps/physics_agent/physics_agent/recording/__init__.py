# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time-sampled USD recording of physics-tune trials (issue #50).

Each tune trial that runs ovphysx (or any backend that returns a
trajectory) writes a ``recording.usd`` with
``Sdf.TimeCode`` time samples on the body's ``xformOp:translate``,
``xformOp:orient``, ``physics:velocity``, and ``physics:angularVelocity``
attributes. The recording is **raw simulator output** — programmatic
metrics in :mod:`world_understanding.functions.physics.trajectory`
read these values back on demand.

This module is intentionally **data-only**. Rendering the recording
into PNGs / mp4s for the VLM judge or visual debugging is a downstream
concern owned by
:mod:`world_understanding.functions.graphics.render_time_sampled_usd`
(issue #50 follow-up); the scenario evaluators import that helper
lazily and degrade to programmatic-only scoring when it isn't
available yet.

Public:
    :func:`physics_agent.recording.recorder.author_trajectory_usda`
    :func:`physics_agent.recording.recorder.author_trajectory_jsonl`
"""

from physics_agent.recording.recorder import (
    author_trajectory_jsonl,
    author_trajectory_usda,
)

__all__ = ["author_trajectory_jsonl", "author_trajectory_usda"]
