# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PR-2.5: the structural scene-tree diff — `snapshot -D`, `--structural`, `--since`.

Black-box, through the `--json` envelope like the rest of the suite. Each test opens a
fresh scene first (which resets the daemon's diff baseline), then drives edits and asserts
on the diff payload.
"""

from __future__ import annotations

from conftest import SPRAY


def _snapshot(project, *args):
    return project.cli("snapshot", *args, json=True, expect_ok=True).json()


def test_first_diff_records_a_baseline(project):
    project.open(SPRAY)
    env = _snapshot(project, "-D")
    # Nothing to diff against yet — the snapshot is banked as the baseline.
    assert env["summary"].get("diff") == "baseline"
    assert "note" in env["data"]["diff"]


def test_diff_reports_a_moved_prim(project):
    project.open(SPRAY)
    _snapshot(project)  # bank the pre-edit baseline
    project.cli("transform", "@n4", "--tx=5", json=True, expect_ok=True)
    env = _snapshot(project, "-D")
    assert env["summary"]["changed"] >= 1
    assert env["summary"]["added"] == 0 and env["summary"]["removed"] == 0
    assert env["summary"]["against"] == "previous"
    changed = {c["ref"]: c for c in env["data"]["diff"]["changed"]}
    assert "@n4" in changed
    assert "trs" in changed["@n4"]["fields"]
    assert "~ @n4" in env["data"]["text"] and "moved" in env["data"]["text"]


def test_structural_diff_shows_attribute_deltas(project):
    project.open(SPRAY)
    _snapshot(project)
    project.cli("transform", "@n4", "--tx=5", json=True, expect_ok=True)
    env = _snapshot(project, "-D", "--structural")
    text = env["data"]["text"]
    assert "~ @n4" in text
    assert "+t(" in text and "-t(" in text  # before/after translate vectors


def test_diff_reports_a_deleted_prim(project):
    project.open(SPRAY)
    _snapshot(project)  # baseline still has @n4
    project.cli("delete", "@n4", json=True, expect_ok=True)
    env = _snapshot(project, "-D")
    assert env["summary"]["removed"] >= 1
    removed = {r["ref"] for r in env["data"]["diff"]["removed"]}
    assert "@n4" in removed
    assert "- @n4" in env["data"]["text"]


def test_since_diffs_against_a_full_checkpoint(project):
    project.open(SPRAY)
    # --full flattens the stage so the checkpoint is self-contained (SPRAY uses a payload,
    # which a root-layer-only checkpoint would fail to resolve from .usd-cli/checkpoints/).
    project.cli("checkpoint", "save", "full_base", "--full", json=True, expect_ok=True)
    project.cli("transform", "@n4", "--tx=7", json=True, expect_ok=True)
    env = _snapshot(project, "--since", "full_base")
    assert env["summary"]["against"] == "full_base"
    assert env["summary"]["changed"] >= 1
    changed = {c["ref"] for c in env["data"]["diff"]["changed"]}
    assert "@n4" in changed


def test_since_warns_when_checkpoint_is_not_self_contained(project):
    project.open(SPRAY)
    # A non-full checkpoint of a referenced asset loses its payload when reopened elsewhere;
    # --since must warn rather than silently report every prim as "added".
    project.cli("checkpoint", "save", "thin", json=True, expect_ok=True)
    env = _snapshot(project, "--since", "thin")
    warns = [i["message"] for i in env["issues"] if i["severity"] == "warn"]
    assert any("checkpoint save --full" in w for w in warns)


def test_since_unknown_checkpoint_errors_cleanly(project):
    project.open(SPRAY)
    res = project.cli("snapshot", "--since", "nope", json=True, expect_ok=False)
    env = res.json()
    assert env["ok"] is False
    assert any("nope" in i["message"] for i in env["issues"])


def test_plain_snapshot_is_unchanged_by_diff_wiring(project):
    project.open(SPRAY)
    env = _snapshot(project)
    # No diff requested → the tree still lands in data.text, no diff payload.
    assert env["data"]["text"].startswith("@n1")
    assert "diff" not in env["data"]
    assert env["summary"]["refs"] == SPRAY.prims + SPRAY.materials
