# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-isolation and honesty fixes from the v3 code review (items 1–4, 17, 18).

Covers, per review item:
  1  two Sessions opening the same file → a clear error instead of a shared (and
     silently clobbered) root SdfLayer; the registry frees on teardown / re-open
  2  per-Session Config deep copy + `--renderer` overrides that never mutate it
  3  session-namespaced filesystem defaults (renders / checkpoints / physics outputs),
     with the "default" session keeping the historical layout exactly
  4  failed-save temp cleanup by directory-snapshot OWNERSHIP, never name+age
  17 real undo for remove-api and material update-in-place (no fake
     set_active(True)→set_active(True) entries), non-undoable ops WARN, and
     properties whose weaker referenced opinions survive are reported as masked
  18 physics.apply pre-resolves explicit operation targets and rolls back on
     mid-apply failure — the stage stays byte-identical

All GPU-free: direct Session calls plus a stub render backend.
"""
from __future__ import annotations

import gc
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from usd_core.config import Config
from usd_core.render.base import RenderResult
from usd_core.session import Session


# ── helpers ─────────────────────────────────────────────────────────────────────


def _write_prop_scene(tmp_path, name="scene.usda"):
    """A cube with RigidBody+Mass physics, exported to disk."""
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/cube")
    prim = stage.GetPrimAtPath("/World/cube")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(5.0)
    path = tmp_path / name
    stage.GetRootLayer().Export(str(path))
    return path


def _write_mesh_scene(tmp_path, name="mesh.usda"):
    """A triangle-soup tetrahedron with deterministic welded topology."""
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/mesh")
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    points = [
        Gf.Vec3f(1, 1, 1), Gf.Vec3f(-1, -1, 1),
        Gf.Vec3f(-1, 1, -1), Gf.Vec3f(1, -1, -1),
    ]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    mesh.CreatePointsAttr([points[index] for face in faces for index in face])
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr(list(range(12)))
    path = tmp_path / name
    stage.GetRootLayer().Export(str(path))
    return path


def _session_on(tmp_path, path, name="default") -> Session:
    s = Session(Config(project_dir=tmp_path), name=name)
    s.open_stage(str(path))
    return s


class _StubBackend:
    """Minimal render backend that just writes the requested PNGs."""

    name = "stub"

    def __init__(self, seen_configs=None):
        self._seen = seen_configs

    def render(self, stage, cameras, width, height, out_dir, mode="fast",
               names=None, frame=None):
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i, cam in enumerate(cameras):
            p = out_dir / f"{(names[i] if names else f'cam{i}')}.png"
            p.write_bytes(b"png")
            results.append(RenderResult(path=str(p), camera=cam, width=width,
                                        height=height, backend=self.name))
        return results


# ── item 1: same file, two sessions ─────────────────────────────────────────────


def test_second_session_on_same_file_errors_clearly(tmp_path):
    path = _write_prop_scene(tmp_path)
    a = _session_on(tmp_path, path, name="agent-a")
    a.transform("/World/cube", translate=[1.0, 2.0, 3.0])  # unsaved edit at stake
    b = Session(Config(project_dir=tmp_path), name="agent-b")
    with pytest.raises(RuntimeError, match=r"open in session 'agent-a'.*layer cache"):
        b.open_stage(str(path))
    # session A's unsaved edit was NOT force-reloaded away by B's attempt
    trs = a._stage.GetPrimAtPath("/World/cube").GetAttribute("xformOp:translate").Get()
    assert list(trs) == [1.0, 2.0, 3.0]


def test_registry_frees_on_open_of_a_different_stage(tmp_path):
    path = _write_prop_scene(tmp_path)
    other = _write_prop_scene(tmp_path, "other.usda")
    a = _session_on(tmp_path, path, name="agent-a")
    a.open_stage(str(other))  # A moved on: the first file is free again
    b = Session(Config(project_dir=tmp_path), name="agent-b")
    assert b.open_stage(str(path)).ok
    # ... and `new` frees too
    b.new()
    c = Session(Config(project_dir=tmp_path), name="agent-c")
    assert c.open_stage(str(path)).ok


def test_registry_frees_on_session_teardown(tmp_path):
    path = _write_prop_scene(tmp_path)
    a = _session_on(tmp_path, path, name="agent-a")
    del a
    gc.collect()
    b = Session(Config(project_dir=tmp_path), name="agent-b")
    assert b.open_stage(str(path)).ok


def test_same_session_reopen_still_reflects_disk(tmp_path):
    """Re-running `open` in ONE session discards its own edits — but only when
    forced (round 6): an unforced same-file reopen with unsaved edits is refused
    with steering, because a second agent sharing the session used exactly that
    to observe/clobber mid-edit state past the cross-session guard."""
    import pytest

    path = _write_prop_scene(tmp_path)
    s = _session_on(tmp_path, path)
    s.transform("/World/cube", translate=[9.0, 9.0, 9.0])
    with pytest.raises(RuntimeError, match="force-reload"):
        s.open_stage(str(path))
    s.open_stage(str(path), force_reload=True)
    attr = s._stage.GetPrimAtPath("/World/cube").GetAttribute("xformOp:translate")
    assert attr.Get() is None or list(attr.Get()) != [9.0, 9.0, 9.0]


# ── item 2: config isolation + renderer overrides ───────────────────────────────


def test_sessions_own_a_private_config_copy(tmp_path):
    cfg = Config(project_dir=tmp_path)
    a = Session(cfg, name="a")
    b = Session(cfg, name="b")
    assert a.config is not cfg and a.config is not b.config
    a.config.render["renderer"] = "ovrtx"
    assert cfg.render["renderer"] == "auto"
    assert b.config.render["renderer"] == "auto"


def test_renderer_override_never_mutates_session_config(tmp_path, monkeypatch):
    import usd_core.render as render_pkg
    path = _write_prop_scene(tmp_path)
    s = _session_on(tmp_path, path)
    seen = []

    def factory(cfg):
        # the override must arrive as an explicit argument (an ephemeral config)...
        seen.append(cfg.render.get("renderer"))
        # ...while the session's own config stays untouched EVEN DURING the render
        assert s.config.render.get("renderer") == "auto"
        if cfg.render.get("renderer") != "auto":
            assert cfg is not s.config  # override rides a copy, never the session config
        return _StubBackend()

    monkeypatch.setattr(render_pkg, "make_backend", factory)
    assert s.render(res=[32, 32], renderer="ovrtx").ok
    assert seen == ["ovrtx"]
    assert s.config.render.get("renderer") == "auto"  # no mutate-and-restore either
    # no override → the session config object itself is passed through
    assert s.render(res=[32, 32]).ok
    assert seen[-1] == "auto"


# ── item 3: session-namespaced filesystem defaults ──────────────────────────────


def test_default_render_dir_is_namespaced_per_session(tmp_path, monkeypatch):
    import usd_core.render as render_pkg
    monkeypatch.setattr(render_pkg, "make_backend", lambda cfg: _StubBackend())
    path = _write_prop_scene(tmp_path)

    named = _session_on(tmp_path, path, name="agent-a")
    resp = named.render(res=[32, 32])
    out = resp.data["results"][0]["path"]
    assert str(tmp_path / ".usd-cli" / "renders" / "agent-a") in out

    named.new()  # free the file for the default session
    default = _session_on(tmp_path, path)
    resp = default.render(res=[32, 32])
    out = resp.data["results"][0]["path"]
    # back-compat: the default session keeps the historical un-namespaced layout
    assert str(tmp_path / ".usd-cli" / "renders") in out
    assert str(tmp_path / ".usd-cli" / "renders" / "agent-a") not in out


def test_checkpoints_are_namespaced_per_session(tmp_path):
    path = _write_prop_scene(tmp_path)
    named = _session_on(tmp_path, path, name="agent-a")
    assert named.checkpoint_save("cp").ok
    assert (tmp_path / ".usd-cli" / "checkpoints" / "agent-a" / "cp.usd").exists()
    assert named.checkpoint_list().data["checkpoints"] == ["cp"]

    named.new()
    default = _session_on(tmp_path, path)
    assert default.checkpoint_save("cp").ok  # same name, no collision
    assert (tmp_path / ".usd-cli" / "checkpoints" / "cp.usd").exists()  # exact old layout
    assert default.checkpoint_list().data["checkpoints"] == ["cp"]


def test_viewer_snapshot_publishes_verified_line_geometry(tmp_path):
    import hashlib
    import numpy as np

    path = _write_mesh_scene(tmp_path)
    session = _session_on(tmp_path, path, name="live-lines")
    destination = tmp_path / "live_view" / "scene-{revision}.usdc"
    geometry = tmp_path / "live_view" / "scene-{revision}.lines.npz"
    source_before = path.read_bytes()

    response = session.viewer_snapshot(str(destination), line_geometry=str(geometry))

    assert response.ok
    metadata = response.summary["line_geometry"]
    geometry_path = tmp_path / "live_view" / "scene-1.lines.npz"
    assert metadata["path"] == str(geometry_path)
    assert metadata["mesh_count"] == 1
    assert metadata["triangle_count"] == 4
    assert metadata["feature_edge_count"] == 6
    assert metadata["silhouette_edge_count"] == 6
    assert hashlib.sha256(geometry_path.read_bytes()).hexdigest() == metadata["sha256"]
    with np.load(geometry_path, allow_pickle=False) as archive:
        assert archive["vertices"].shape == (4, 3)
        assert archive["triangles"].shape == (4, 3)
        assert archive["feature_edges"].shape == (6, 2)
        assert archive["silhouette_normals"].shape == (6, 2, 3)
    assert path.read_bytes() == source_before


def test_viewer_line_geometry_reuses_material_topology_and_rebuilds_transform(tmp_path):
    path = _write_mesh_scene(tmp_path)
    session = _session_on(tmp_path, path, name="live-lines-cache")
    destination = tmp_path / "live_view" / "scene-{revision}.usdc"
    geometry = tmp_path / "live_view" / "scene-{revision}.lines.npz"
    first = session.viewer_snapshot(str(destination), line_geometry=str(geometry))
    first_lines = first.summary["line_geometry"]

    assert session.material(
        "/World/mesh", color=[0.2, 0.4, 0.8], name="Blue").ok
    material_snapshot = session.viewer_snapshot(
        str(destination), line_geometry=str(geometry))
    assert material_snapshot.summary["revision"] == 2
    assert material_snapshot.summary["line_geometry"] == first_lines

    unchanged = session.viewer_snapshot(
        str(destination), since_revision=2, line_geometry=str(geometry))
    assert unchanged.summary["changed"] is False
    assert session.transform("/World/mesh", translate=[2.0, 0.0, 0.0]).ok
    transformed = session.viewer_snapshot(
        str(destination), line_geometry=str(geometry))
    assert transformed.summary["revision"] == 3
    assert transformed.summary["line_geometry"]["path"].endswith(
        "scene-3.lines.npz")
    assert transformed.summary["line_geometry"]["sha256"] != first_lines["sha256"]


def test_viewer_line_geometry_cache_isolated_by_filename_pattern(tmp_path):
    path = _write_mesh_scene(tmp_path)
    session = _session_on(tmp_path, path, name="live-lines-patterns")
    destination = tmp_path / "live_view" / "scene-{revision}.usdc"
    first_pattern = tmp_path / "live_view" / "primary-{revision}.lines.npz"
    second_pattern = tmp_path / "live_view" / "alternate-{revision}.lines.npz"

    first = session.viewer_snapshot(
        str(destination),
        line_geometry=str(first_pattern),
    )
    assert session.material(
        "/World/mesh", color=[0.2, 0.4, 0.8], name="Blue"
    ).ok
    second = session.viewer_snapshot(
        str(destination),
        line_geometry=str(second_pattern),
    )

    assert first.summary["line_geometry"]["path"].endswith(
        "primary-1.lines.npz"
    )
    assert second.summary["line_geometry"]["path"].endswith(
        "alternate-2.lines.npz"
    )
    assert Path(second.summary["line_geometry"]["path"]).is_file()


def test_new_stage_advances_viewer_revision_via_default_camera(tmp_path):
    path = _write_mesh_scene(tmp_path)
    session = _session_on(tmp_path, path, name="live-new-stage")
    previous_revision = session._viewer_revision

    assert session.new().ok

    assert session._viewer_generation == 2
    assert session._viewer_revision > previous_revision


def test_physics_outputs_are_namespaced_per_session(tmp_path, monkeypatch):
    from usd_core import physics_runtime

    seen = []

    def fake_simulate_scene(scene_usd, output_dir, **kw):
        seen.append(output_dir)
        return {"ok": True, "failures": [], "warnings": [],
                "metrics": {}, "scene_usd": "", "recording_usda": "",
                "trajectory_jsonl": "", "report_path": "", "executor": "stub",
                "engine": kw.get("engine", "ovphysx"), "n_bodies": 1}

    monkeypatch.setattr(physics_runtime, "simulate_scene", fake_simulate_scene)
    path = _write_prop_scene(tmp_path)
    named = _session_on(tmp_path, path, name="agent-a")
    assert named.physics_simulate(
        scene=str(path),
        body="/World/cube",
        rest_position=[0.0, 0.0, 0.0],
        world_up=[0.0, 0.0, 1.0],
    ).ok
    assert Path(seen[-1]) == tmp_path / ".usd-cli" / "physics" / "agent-a"

    named.new()
    default = _session_on(tmp_path, path)
    assert default.physics_simulate(
        scene=str(path),
        body="/World/cube",
        rest_position=[0.0, 0.0, 0.0],
        world_up=[0.0, 0.0, 1.0],
    ).ok
    assert Path(seen[-1]) == tmp_path / ".usd-cli" / "physics"


def test_session_name_is_validated(tmp_path):
    with pytest.raises(ValueError, match="session name"):
        Session(Config(project_dir=tmp_path), name="../escape")
    with pytest.raises(ValueError, match="session name"):
        Session(Config(project_dir=tmp_path), name="")


# ── item 4: failed-save cleanup is ownership-based ──────────────────────────────


def test_failed_save_cleanup_spares_preexisting_lookalikes(tmp_path):
    path = _write_prop_scene(tmp_path)
    s = _session_on(tmp_path, path)
    out_dir = tmp_path / "deliver"
    out_dir.mkdir()
    lookalike = out_dir / "model.2024Q1"  # a user file matching the mkstemp shape
    lookalike.write_text("keep me")
    (out_dir / "model.usda").mkdir()  # a directory at the target path → save fails

    resp = s.save(str(out_dir / "model.usda"))
    assert resp.ok is False
    assert lookalike.exists()  # the old name+age heuristic deleted this
    assert lookalike.read_text() == "keep me"


def test_failed_save_cleanup_still_removes_new_writer_litter(tmp_path, monkeypatch):
    path = _write_prop_scene(tmp_path)
    s = _session_on(tmp_path, path)
    target = tmp_path / "out.usda"
    pre_existing = tmp_path / "out.Zz99Aa"  # present before the attempt: not ours
    pre_existing.write_text("x")
    litter = tmp_path / "out.VxRCU3"

    def boom(root_layer, dest):
        litter.write_text("half-written")
        raise RuntimeError("disk full")

    monkeypatch.setattr(Session, "_atomic_export", staticmethod(boom))
    resp = s.save(str(target))
    assert resp.ok is False
    assert not litter.exists()      # created after the snapshot → ours → removed
    assert pre_existing.exists()    # in the snapshot → never touched


# ── item 17: honest undo ────────────────────────────────────────────────────────


def test_remove_api_undo_restores_schema_and_values(tmp_path):
    path = _write_prop_scene(tmp_path)
    s = _session_on(tmp_path, path)
    prim = s._stage.GetPrimAtPath("/World/cube")

    resp = s.remove_api("/World/cube", "MassAPI")
    assert resp.ok
    assert resp.data["properties_removed"] == ["physics:mass"]
    assert not prim.HasAPI(UsdPhysics.MassAPI)

    undo = s.undo()
    assert undo.ok
    assert not undo.issues  # a real restore — no "not undoable" warning
    prim = s._stage.GetPrimAtPath("/World/cube")
    assert prim.HasAPI(UsdPhysics.MassAPI)
    assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == 5.0

    redo = s.redo()
    assert redo.ok
    assert not s._stage.GetPrimAtPath("/World/cube").HasAPI(UsdPhysics.MassAPI)


def test_non_undoable_ops_warn_instead_of_fake_success(tmp_path):
    from usd_core.history import Op
    path = _write_prop_scene(tmp_path)
    s = _session_on(tmp_path, path)
    s.history.record(Op(command="remove-api", label="remove-api X",
                        non_undoable="remove-api"))
    resp = s.undo()
    assert resp.ok
    assert resp.summary.get("not_undoable") == 1
    assert any("not undoable (remove-api)" in i.message for i in resp.issues)


def test_remove_api_reports_masked_referenced_opinions(tmp_path):
    base_path = _write_prop_scene(tmp_path, "base.usda")
    top = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(top, "/World")
    top.SetDefaultPrim(world.GetPrim())
    top.DefinePrim("/World/asset", "Xform").GetReferences().AddReference(
        str(base_path), "/World")
    top_path = tmp_path / "top.usda"
    top.GetRootLayer().Export(str(top_path))
    del top

    s = _session_on(tmp_path, top_path)
    assert s.set("/World/asset/cube", "physics:mass", "10").ok  # local override
    resp = s.remove_api("/World/asset/cube", "MassAPI")
    assert resp.ok
    masked = resp.data["properties_masked"]
    assert [m["name"] for m in masked] == ["physics:mass"]
    assert masked[0]["layer"].endswith("base.usda")
    assert "physics:mass" not in resp.data["properties_removed"]  # not claimed removed
    assert any("masked, opinion remains in" in i.message for i in resp.issues)
    # honesty check: the value indeed still resolves from the referenced layer
    attr = s._stage.GetPrimAtPath("/World/asset/cube").GetAttribute("physics:mass")
    assert attr.Get() == 5.0


def test_material_update_in_place_undo_restores_inputs(tmp_path):
    path = _write_prop_scene(tmp_path)
    s = _session_on(tmp_path, path)
    mat = UsdShade.Material.Define(s._stage, "/World/Looks/M")
    shader = UsdShade.Shader.Define(s._stage, "/World/Looks/M/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 0))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    resp = s.material("/World/Looks/M", color=[0, 0, 1], roughness=0.9)
    assert resp.ok
    assert shader.GetInput("diffuseColor").Get() == Gf.Vec3f(0, 0, 1)

    undo = s.undo()
    assert undo.ok and not undo.issues
    assert shader.GetInput("diffuseColor").Get() == Gf.Vec3f(1, 0, 0)  # value restored
    rough = shader.GetInput("roughness")
    assert not rough or not rough.GetAttr().HasAuthoredValue()  # new input rolled back

    redo = s.redo()
    assert redo.ok
    assert shader.GetInput("diffuseColor").Get() == Gf.Vec3f(0, 0, 1)
    assert shader.GetInput("roughness").Get() == pytest.approx(0.9)


# ── item 18: physics.apply is validated up front and transactional ──────────────


def test_physics_apply_bad_collision_ref_leaves_stage_byte_identical(tmp_path):
    path = _write_prop_scene(tmp_path)
    s = _session_on(tmp_path, path)
    before = s._stage.GetRootLayer().ExportToString()
    resp = s.physics_apply(
        operations={"colliders": [{"path": "@zz9", "approximation": "convexHull"}]}
    )
    assert resp.ok is False
    assert s._stage.GetRootLayer().ExportToString() == before

    resp = s.physics_apply(
        operations={
            "colliders": [{"path": "/World/nope", "approximation": "convexHull"}]
        }
    )
    assert resp.ok is False
    assert "missing prim" in resp.issues[0].message
    assert s._stage.GetRootLayer().ExportToString() == before


def test_physics_apply_mid_apply_failure_rolls_back(tmp_path):
    path = _write_prop_scene(tmp_path)
    s = _session_on(tmp_path, path)
    before = s._stage.GetRootLayer().ExportToString()
    # The invalid collider fails after the rigid body is authored; all changes roll back.
    resp = s.physics_apply(
        operations={
            "rigid_bodies": [{"path": "/World/cube", "mass": 1.0}],
            "colliders": [{"path": "/World/cube", "approximation": "bogus"}],
        }
    )
    assert resp.ok is False
    assert s._stage.GetRootLayer().ExportToString() == before

    # and the happy path still authors (the rollback plumbing isn't in the way)
    resp = s.physics_apply(
        operations={"rigid_bodies": [{"path": "/World/cube", "mass": 1.0}]}
    )
    assert resp.ok
    assert s._stage.GetPrimAtPath("/World/cube").HasAPI(UsdPhysics.RigidBodyAPI)
