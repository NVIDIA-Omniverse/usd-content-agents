# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PR-1/PR-2: the perceive loop — open, the @ref tree, filters, and resolve.

All assertions go through the `--json` envelope so they're robust to text formatting.
"""

from __future__ import annotations

import pytest

from conftest import ASSETS, SPRAY


@pytest.mark.parametrize("asset", ASSETS, ids=[a.id for a in ASSETS])
def test_open_reports_scene_facts(project, asset):
    env = project.open(asset).json()
    assert env["ok"] is True
    assert env["command"] == "open"
    assert env["summary"]["stage"].endswith(asset.relpath)
    assert env["summary"]["prims"] == asset.prims + asset.materials
    assert env["summary"]["up_axis"] == asset.up_axis


@pytest.mark.parametrize("asset", ASSETS, ids=[a.id for a in ASSETS])
def test_snapshot_lists_every_ref(project, asset):
    project.open(asset)
    env = project.cli("snapshot", json=True, expect_ok=True).json()
    assert env["summary"]["refs"] == asset.prims + asset.materials
    tree = env["data"]["text"]
    assert tree.startswith("@n1")
    for name in asset.prim_names:
        assert name in tree, f"{name!r} missing from snapshot tree:\n{tree}"


def test_snapshot_depth_filter_is_shallower(project):
    project.open(SPRAY)
    full = project.cli("snapshot", json=True, expect_ok=True).json()
    shallow = project.cli("snapshot", "-d", "1", json=True, expect_ok=True).json()
    assert shallow["summary"]["refs"] < full["summary"]["refs"]
    # depth 1 stops above the meshes nested under Geometry
    assert "bottle_body" not in shallow["data"]["text"]
    assert "spray_bottle" in shallow["data"]["text"]


def test_snapshot_type_filter_keeps_only_matches(project):
    project.open(SPRAY)
    env = project.cli("snapshot", "-t", "Mesh", json=True, expect_ok=True).json()
    tree = env["data"]["text"]
    assert tree, "type-filtered snapshot was empty"
    for line in tree.splitlines():
        assert "[Mesh]" in line, f"non-Mesh row survived -t Mesh:\n{line}"
    assert "[Scope]" not in tree


def test_snapshot_scope_to_subtree(project):
    project.open(SPRAY)
    env = project.cli("snapshot", "@n3", json=True, expect_ok=True).json()
    tree = env["data"]["text"]
    assert tree.startswith("@n3")
    assert "Geometry" in tree
    assert "Materials" not in tree  # a sibling of @n3, outside the scope


@pytest.mark.parametrize("asset", ASSETS, ids=[a.id for a in ASSETS])
def test_resolve_lists_full_paths(project, asset):
    project.open(asset)
    env = project.cli("resolve", json=True, expect_ok=True).json()
    text = env["data"]["text"]
    assert f"/{asset.root_name}" in text
    # every indexed ref should be revealed
    assert text.count("@n") == asset.prims


def test_resolve_single_ref(project):
    project.open(SPRAY)
    env = project.cli("resolve", "@n4", json=True, expect_ok=True).json()
    assert env["data"]["text"].strip().count("@n") == 1
    assert "/spray_bottle/Geometry/bottle_body" in env["data"]["text"]
