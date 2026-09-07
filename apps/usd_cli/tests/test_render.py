# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PR-3: single-modality render, parametrized over render backends.

The active backend is chosen at *daemon startup* from config, so each backend gets its
own daemon via the `render_project` fixture (parametrized in conftest). Backends not
available on the host are skipped, not failed. These are the slowest tests — keep
resolutions tiny.
"""

from __future__ import annotations

import os

import pytest

from conftest import ASSETS, SPRAY, png_dimensions


def _only_result(env: dict) -> dict:
    results = env["data"]["results"]
    assert len(results) == 1, f"expected one render, got {len(results)}"
    return results[0]


@pytest.mark.parametrize("asset", ASSETS, ids=[a.id for a in ASSETS])
def test_render_writes_a_real_png(render_project, asset):
    render_project.open(asset)
    env = render_project.cli(
        "render", "--res", "160x120", json=True, expect_ok=True
    ).json()
    backend = env["summary"]["backend"]
    assert env["summary"]["res"] == "160x120"

    path = _only_result(env)["path"]
    assert os.path.exists(path), f"{backend} reported a render that isn't on disk: {path}"
    assert os.path.getsize(path) > 0
    assert png_dimensions(path) == (160, 120)


def test_render_focus_on_one_object(render_project):
    render_project.open(SPRAY)
    env = render_project.cli(
        "render", "--focus", "@n4", "--res", "128x128", json=True, expect_ok=True
    ).json()
    assert os.path.exists(_only_result(env)["path"])


def test_render_orbit_produces_n_views(render_project):
    render_project.open(SPRAY)
    env = render_project.cli(
        "render", "--orbit", "3", "--res", "96x96", json=True, expect_ok=True
    ).json()
    results = env["data"]["results"]
    assert len(results) == 3
    assert env["summary"]["orbit"] == 3
    for r in results:
        assert os.path.exists(r["path"])
        assert r["camera"].startswith("/")
        assert len(r["camera_world_transform"]) == 4
        assert all(len(row) == 4 for row in r["camera_world_transform"])
        assert len(r["camera_pos"]) == 3
        assert len(r["camera_dir"]) == 3


@pytest.mark.parametrize("res", ["64x64", "200x120"], ids=["square", "wide"])
def test_render_honors_resolution(render_project, res):
    render_project.open(SPRAY)
    env = render_project.cli(
        "render", "--res", res, json=True, expect_ok=True
    ).json()
    w, h = (int(x) for x in res.split("x"))
    assert png_dimensions(_only_result(env)["path"]) == (w, h)
