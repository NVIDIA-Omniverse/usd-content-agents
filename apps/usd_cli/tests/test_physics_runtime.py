# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit physics simulation + time-sampled render-frames.

Simulation uses the real ovphysx GPU solver — there is no synthetic fallback — so
these tests run only on a supported host (Linux + NVIDIA GPU) and skip elsewhere, the same
way the ovrtx render tests do. Playback (render-frames) additionally needs a beauty backend.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from usd_core import physics_runtime


def _ovphysx_available() -> bool:
    """Accept native Linux or WSL2 through the production platform gate."""
    return sys.platform.startswith(
        "linux"
    ) and physics_runtime.ovphysx_platform_supported()


pytestmark = pytest.mark.skipif(
    not _ovphysx_available(),
    reason="runtime physics needs a supported local OvPhysX GPU platform",
)


def _have_beauty(project) -> bool:
    env = project.cli("render", "--res", "64x48", json=True).json()
    return env.get("ok") and any(a["label"].startswith("rgb") for a in env.get("artifacts", []))


def _preauthor_scenario(project) -> Path:
    source = project.root / "physics_source.usda"
    source.write_text(
        """#usda 1.0
(
    defaultPrim = "Body"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Cube "Body"
{
    double size = 0.2
}
""",
        encoding="utf-8",
    )
    project.cli("open", source, "--force-reload", json=True, expect_ok=True)
    project.cli("physics", "apply", "-f", _operations(project), json=True, expect_ok=True)
    scenario = project.root / "preauthored_physics.usda"
    project.cli("save", str(scenario), json=True, expect_ok=True)
    return scenario


def test_simulate_produces_recording(project):
    scenario = _preauthor_scenario(project)
    env = project.cli(
        "physics", "simulate", "--scene", scenario, "--body", "@n1",
        "--rest-position", "0,0,0", "--world-up", "0,0,1", "--engine", "ovphysx",
        "--duration", "1.0", "--fps", "24", json=True, expect_ok=True,
    ).json()
    assert env["ok"] is True
    assert env["data"]["recording_usda"] and Path(env["data"]["recording_usda"]).exists()
    assert env["data"]["metrics"]["n_samples"] >= 2
    # a recording artifact is surfaced
    assert any(a["label"] == "recording" for a in env["artifacts"])


def test_render_frames_plays_a_recording(project):
    if not _have_beauty(project):
        pytest.skip("no OVRTX beauty backend is configured on this host")
    scenario = _preauthor_scenario(project)
    rv = project.cli(
        "physics", "simulate", "--scene", scenario, "--body", "@n1",
        "--rest-position", "0,0,0", "--world-up", "0,0,1", "--engine", "ovphysx",
        "--fps", "24", json=True, expect_ok=True,
    ).json()
    rec = rv["data"]["recording_usda"]
    rf = project.cli("render-frames", "--scene", rec, "--frames", "0:12", "--res", "200x150",
                     "--fps", "12", json=True, expect_ok=True).json()
    paths = rf["data"]["frame_paths"]
    assert len(paths) == 13
    # the body falls, so frames must differ (not a static repeat)
    hashes = {hashlib.md5(Path(p).read_bytes()).hexdigest() for p in paths}
    assert len(hashes) > 1
    assert rf["data"]["animation"] and Path(rf["data"]["animation"]).exists()


def _operations(project) -> str:
    import json
    p = project.root / "rb.json"
    p.write_text(
        json.dumps(
            {
                "scene_paths": ["/PhysicsScenario"],
                "rigid_bodies": [{"path": "@n1", "density": 600.0}],
                "colliders": [{"path": "@n1"}],
            }
        )
    )
    return str(p)


def test_trajectory_metrics_reports_rebound():
    """Raw metrics surface bounce facts so a workflow can evaluate a
    'first rebound >= X% of drop' criterion is checkable without hard-coding it."""
    from usd_core.physics_runtime import trajectory_metrics

    # drop from z=1.0, impact at z=0.0, rebound to z=0.3, settle at z=0.05
    hs = [1.0, 0.5, 0.0, 0.2, 0.3, 0.15, 0.05, 0.05]
    traj = [(i * 0.1, [0.0, 0.0, z], [0, 0, 0, 0, 0, 0]) for i, z in enumerate(hs)]
    m = trajectory_metrics(traj, rest_position=[0, 0, 0.05], world_up=[0, 0, 1])
    assert m["drop_height"] == 1.0
    assert m["first_rebound_height"] == 0.3
    assert m["rebound_fraction"] == 0.3
