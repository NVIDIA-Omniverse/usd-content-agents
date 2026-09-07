# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal OVRTX-render tests: analytic AOVs, overlays, diffs, and inline data.

AOV passes are CPU-computed from geometry + camera intrinsics, but are emitted only
alongside an OVRTX-backed RGB render.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from conftest import SPRAY


def _img(path, mode="RGB"):
    return np.asarray(Image.open(path).convert(mode))


def test_render_rejects_aov_only_request_without_ovrtx(project):
    """AOV flags never turn `render` into a CPU-only renderer."""
    project.open(SPRAY)
    env = project.cli("render", "--depth", "--normals", "--seg", "--wireframe",
                      "--res", "320x240", json=True, expect_ok=False).json()
    assert env["ok"] is False
    assert "ovrtx renderer" in env["issues"][0]["message"].lower()


def test_render_frames_requires_ovrtx(project):
    """Animation rendering has no CPU-only fallback either."""
    project.open(SPRAY)
    env = project.cli("render-frames", "--frames", "0", "--no-animate",
                      json=True, expect_ok=False).json()
    assert env["ok"] is False
    assert "ovrtx renderer" in env["issues"][0]["message"].lower()


def test_aovs_accompany_ovrtx_render(render_project):
    """AOVs are non-blank auxiliary artifacts of a successful OVRTX render."""
    project = render_project
    project.open(SPRAY)
    env = project.cli("render", "--depth", "--normals", "--seg", "--wireframe",
                      "--res", "320x240", json=True, expect_ok=True).json()
    labels = {a["label"] for a in env["artifacts"]}
    assert {"rgb", "depth", "normals", "segmentation", "wireframe"} <= {
        label.split(":", 1)[0] for label in labels
    }
    assert env["data"]["segmentation_legend"]  # ref -> color map present
    # each AOV file is a real, non-empty image of the requested size; the
    # segmentation legend ships as a kind="legend" TEXT artifact beside it
    for a in env["artifacts"]:
        if a["kind"] == "legend":
            continue
        im = Image.open(a["path"])
        assert im.size == (320, 240)
    legends = [a for a in env["artifacts"] if a["kind"] == "legend"]
    assert len(legends) == 1 and legends[0]["path"].endswith(".legend.txt")


def test_segmentation_has_distinct_object_colors(render_project):
    project = render_project
    project.open(SPRAY)
    env = project.cli("render", "--seg", "--res", "256x256", json=True, expect_ok=True).json()
    # legend is deterministic (one entry per mesh); image color count depends on per-pixel
    # coverage, so assert on the legend (SPRAY has 4 meshes) plus "more than just background".
    legend = env["data"]["segmentation_legend"]
    assert len(legend) == 4
    seg_path = next(a["path"] for a in env["artifacts"] if a["label"] == "segmentation")
    colors = np.unique(_img(seg_path).reshape(-1, 3), axis=0)
    assert len(colors) >= 2  # background + at least one object


def test_depth_has_range(render_project):
    project = render_project
    project.open(SPRAY)
    env = project.cli("render", "--depth", "--res", "256x256", json=True, expect_ok=True).json()
    depth_path = next(a["path"] for a in env["artifacts"] if a["label"] == "depth")
    linear_depth_path = next(
        a["path"] for a in env["artifacts"] if a["label"] == "linear_depth"
    )
    d = _img(depth_path, "L")
    linear_depth = np.load(linear_depth_path, allow_pickle=False)
    assert int(d.max()) > int(d.min())  # a real near→far gradient, not flat
    assert linear_depth.dtype == np.float32
    assert np.isfinite(linear_depth).any()
    assert np.all(linear_depth[np.isfinite(linear_depth)] > 0)
    assert env["data"]["linear_depth_unit"] == "meter"


def test_inline_base64_present(render_project):
    project = render_project
    project.open(SPRAY)
    env = project.cli("render", "--seg", "--inline", "--res", "128x128",
                      json=True, expect_ok=True).json()
    assert "segmentation" in env["data"]["inline"]
    assert len(env["data"]["inline"]["segmentation"]) > 100  # looks like real base64


def _have_beauty(project):
    """True if a default OVRTX beauty render works on this host."""
    env = project.cli("render", "--res", "64x48", json=True).json()
    return env.get("ok") and any(a["label"].startswith("rgb") for a in env.get("artifacts", []))


def test_geometry_diff_catches_moves(project):
    """`--diff both` produces a visual diff (beauty) AND a geometry diff (segmentation)."""
    project.open(SPRAY)
    if not _have_beauty(project):
        pytest.skip("no OVRTX beauty backend is configured on this host")
    base = project.cli("render", "--photoreal", "--seg", "--res", "200x150",
                       json=True, expect_ok=True).json()
    beauty = next(a["path"] for a in base["artifacts"] if a["label"].startswith("rgb"))
    project.cli("transform", "@n4", "--tx=+0.1", expect_ok=True)
    env = project.cli("render", "--photoreal", "--seg", "--against", beauty, "--diff", "both",
                      "--res", "200x150", json=True, expect_ok=True).json()
    labels = {a["label"] for a in env["artifacts"]}
    assert "diff" in labels and "diff:geometry" in labels
    assert env["data"]["diff"]["changed_fraction"] >= 0
    assert env["data"]["diff_geometry"]["changed_fraction"] > 0  # the footprint moved


def test_annotate_and_against(project):
    project.open(SPRAY)
    if not _have_beauty(project):
        pytest.skip("no OVRTX beauty backend is configured on this host")
    a = project.cli("render", "--photoreal", "--annotate", "--res", "200x150",
                     json=True, expect_ok=True).json()
    assert any(x["label"] == "annotated" for x in a["artifacts"])
    beauty = next(x["path"] for x in a["artifacts"] if x["label"].startswith("rgb"))
    project.cli("transform", "@n4", "--tx=+0.3", expect_ok=True)
    b = project.cli("render", "--photoreal", "--against", beauty, "--res", "200x150",
                     json=True, expect_ok=True).json()
    assert "diff" in b["data"] and b["data"]["diff"]["changed_fraction"] > 0
