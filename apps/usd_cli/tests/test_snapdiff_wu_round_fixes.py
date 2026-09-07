# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural-diff reliability fixes from the 2026-07-09 wu-examples benchmark (task-12).

Two failure modes seen in the field, both breaking the documented "no collateral
damage" check (`snapshot -D` / `--since <checkpoint>`):

A. A scoped/filtered snapshot (`snapshot /visuals -d 5`) banked as the `-D` baseline,
   so the next whole-scene `snapshot -D` diffed two different *views* of the stage and
   reported near-whole-tree churn (added: 148 | removed: 143) after a single bind.
B. `--since <checkpoint>` reported "(no changes)" after a full material pass because
   the diff capture (a) was rooted at the defaultPrim, missing sibling trees like
   /visuals and /Looks, (b) skipped Material/Shader prims (not Imageable), (c) never
   descended into native instances, and (d) after `checkpoint load`, `Usd.Stage.Open`
   on the checkpoint path reused the *live, edited* in-memory layer instead of the
   saved on-disk state.

Direct Session tests (need pxr), no daemon, no GPU.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pxr")


def _write_robot_scene(tmp_path):
    """A miniature of task-12's g1_zup.usdc layout: defaultPrim /Robot, prototype
    sources under a *sibling* /visuals tree, and instanceable internal references."""
    from pxr import Usd, UsdGeom

    path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(path))
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(robot.GetPrim())
    for part in ("pelvis", "torso", "head"):
        UsdGeom.Xform.Define(stage, f"/visuals/{part}")
        mesh = UsdGeom.Mesh.Define(stage, f"/visuals/{part}/mesh")
        mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
        UsdGeom.Xform.Define(stage, f"/Robot/{part}_link")
        inst = stage.DefinePrim(f"/Robot/{part}_link/visuals")
        inst.GetReferences().AddInternalReference(f"/visuals/{part}")
        inst.SetInstanceable(True)
    stage.Save()
    return str(path)


@pytest.fixture()
def session(tmp_path):
    from usd_core.config import Config
    from usd_core.session import Session

    return Session.open(_write_robot_scene(tmp_path), Config(project_dir=tmp_path))


def _counts(resp):
    assert resp.ok, resp.issues
    return resp.data["diff"]["counts"]


def _changed_paths(resp):
    return {c["path"] for c in resp.data["diff"]["changed"]}


def _added(resp):
    return {a["path"]: a for a in resp.data["diff"]["added"]}


# ── bug A: scoped snapshots must not poison the -D baseline ───────────────────────
def test_scoped_snapshot_then_whole_scene_diff_is_small(session):
    # a scoped + depth-limited snapshot banks the baseline…
    assert session.snapshot(scope="/visuals", depth=5).ok
    # …then ONE bind happens…
    r = session.material(ref="/visuals/pelvis/mesh", color=[1, 1, 1], name="White")
    assert r.ok, r.issues
    # …and the whole-scene diff must show a handful of deltas, not wholesale churn.
    resp = session.snapshot(diff=True, structural=True)
    counts = _counts(resp)
    assert counts["removed"] == 0
    assert counts["changed"] >= 1
    assert "/visuals/pelvis/mesh" in _changed_paths(resp)
    # added = the new Looks scope / Material / Shader — not half the tree
    assert counts["added"] <= 6
    assert counts["added"] + counts["changed"] <= 12


def test_diff_identity_is_keyed_by_path_not_ref(session):
    from usd_core import snapdiff

    before = snapdiff.capture_state(session._stage, "/", refs=session.refs)
    # simulate a re-index that renumbers every display ref: paths are the identity,
    # so a refs-only change must produce an empty diff
    after = {p: dict(e, ref=f"@n{999 + i}") for i, (p, e) in enumerate(before.items())}
    result = snapdiff.diff_states(before, after)
    assert result["counts"] == {"added": 0, "removed": 0, "changed": 0}


# ── bug B: --since must see material deltas anywhere in the tree ──────────────────
def test_since_full_checkpoint_reports_bind_outside_default_prim(session):
    assert session.checkpoint_save("base", full=True).ok
    r = session.material(ref="/visuals/pelvis/mesh", color=[1, 1, 1], name="White")
    assert r.ok, r.issues
    resp = session.snapshot(since="base")
    counts = _counts(resp)
    # the bound source mesh lives OUTSIDE the defaultPrim (/Robot) — it must show
    assert "/visuals/pelvis/mesh" in _changed_paths(resp)
    assert counts["changed"] >= 1
    # …and the new Material prim (not Imageable) must show as added
    added = _added(resp)
    assert any(e["type"] == "Material" for e in added.values()), added


def test_since_full_checkpoint_reports_new_material_prims(session):
    from pxr import UsdShade

    assert session.checkpoint_save("base", full=True).ok
    # author a material + shader directly under a root-level /Looks scope
    mat = UsdShade.Material.Define(session._stage, "/Looks/Steel")
    shader = UsdShade.Shader.Define(session._stage, "/Looks/Steel/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    resp = session.snapshot(since="base")
    added = _added(resp)
    assert "/Looks/Steel" in added and added["/Looks/Steel"]["type"] == "Material"
    assert "/Looks/Steel/Shader" in added and added["/Looks/Steel/Shader"]["type"] == "Shader"


def test_diff_sees_binding_change_inside_native_instances(session):
    assert session.snapshot().ok  # bank the pre-edit baseline
    r = session.material(ref="/visuals/pelvis/mesh", color=[0, 0, 1], name="Blue")
    assert r.ok, r.issues
    resp = session.snapshot(diff=True)
    # instances diff as units, but the bind every pelvis instance inherits from its
    # prototype must surface — as a materials_within change on the instance root
    changed = {c["path"]: c for c in resp.data["diff"]["changed"]}
    assert "/Robot/pelvis_link/visuals" in changed, changed
    assert "materials_within" in changed["/Robot/pelvis_link/visuals"]["fields"]


def test_since_after_checkpoint_load_diffs_the_on_disk_state(session):
    # `checkpoint load` makes the live root layer BE the checkpoint file; a later
    # `--since` must still diff against the *saved* bytes, not the edited layer.
    assert session.checkpoint_save("cp", full=True).ok
    assert session.checkpoint_load("cp").ok
    r = session.material(ref="/visuals/torso/mesh", color=[1, 0, 0], name="Red")
    assert r.ok, r.issues
    resp = session.snapshot(since="cp")
    counts = _counts(resp)
    assert counts["changed"] >= 1, resp.data["diff"]
    assert "/visuals/torso/mesh" in _changed_paths(resp)
    # …and the live stage keeps its unsaved edits (the fix must not reload the layer)
    from usd_core.materials import bound_material
    assert bound_material(session._stage, "/visuals/torso/mesh")["bound_material_path"]


def test_since_full_checkpoint_after_no_edits_is_clean(session):
    assert session.checkpoint_save("base", full=True).ok
    resp = session.snapshot(since="base")
    assert _counts(resp) == {"added": 0, "removed": 0, "changed": 0}


# ── diff filters stay symmetric ────────────────────────────────────────────────────
def test_scoped_diff_restricts_both_sides_to_the_scope(session):
    assert session.snapshot().ok  # whole-scene baseline
    r = session.material(ref="/visuals/pelvis/mesh", color=[1, 1, 1], name="White")
    assert r.ok, r.issues
    resp = session.snapshot(scope="/Robot", diff=True)
    # the bind on /visuals and the new /Looks material are outside the scope…
    assert all(p.startswith("/Robot") for p in _changed_paths(resp))
    assert all(p.startswith("/Robot") for p in _added(resp))
    # …but the instance that inherits the bind still reports it, inside the scope
    assert "/Robot/pelvis_link/visuals" in _changed_paths(resp)
