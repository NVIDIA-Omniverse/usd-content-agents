# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-5 session fixes from the v4 agent-benchmark traces.

Covers, per finding:
  1  in-place `save` over the session's own open .usdc must never corrupt it —
     verified atomic publish (temp sibling + fsync + read-back verify + rename), and
     `open` of a truncated/corrupt crate fails with a clean, actionable error
  2  drop_none keeps zero-valued options (--elevation 0 / --metallic 0 …) and still
     drops literal False (unset CLI flags), None, and empty tuple/list
  3  `save --flatten` localizes external references (e.g. `material --library` arcs)
     into a self-contained file
  4  `open --read-only`: parallel readers alongside one writer; every mutating
     command hard-blocked in the reader session
  5  `convert in.usdc out.usda` writes the requested output for USD→USD instead of a
     silent passthrough no-op
  6  render/camera focus on a hidden prim says the geometry is HIDDEN (with a `show`
     hint) instead of claiming there is no geometry
  7  remove-api summary reads as a real removal when the API had no authored props
  8  the render section is re-read from the project config file per render, so
     mid-session `.usd-cli/config.toml` edits reach the next backend construction
  9  bare-number "refs" (agents strip the '@n') get a clean error with a
     did-you-mean hint instead of an 'Ill-formed SdfPath' log flood, and
     `render --exclude` fails loudly on unresolvable refs instead of silently
     rendering without the exclusion

Review-v5 follow-ups (numbering from .codex-review-v5-findings.txt) are appended
below the original items:
   1  a failed candidate `open` (corrupt file) leaves the OLD stage live and owned
   2  publish refuses another session's open file; concurrent same-destination
      publishes serialize on a per-destination lock
   3  a symlink destination is written through to its canonical target (the link
      is preserved) and aliases can't bypass the ownership check
   4  open/save verification force-reads authored values + time samples, catching
      value-corrupted crates whose prim indexes parse
   5  same-file save rebuilds the ref/camera caches; a post-commit reload failure
      is a warning with a reopen hint, never 'save failed'
   6  read-only sessions: camera look-at/orbit/fit/create/pan/zoom, validate --fix,
      render authoring, and export-over-source are all blocked
   8  `--exclude` resolves/validates/dedupes the whole list BEFORE hiding anything
  15  render hot-reload keeps env > project precedence and drops removed keys
  16  convert USD→USD goes through the verified-atomic publish; .usdz outputs are
      packaged via UsdUtils
  17  the destination directory is fsynced after the final rename
  23  property/variant-selection paths get a clear error from `_path_of`
  24  save --flatten is described as composition-flattening (external files stay
      referenced, not embedded)
  25  the hidden-focus hint requires hidden BOUNDABLE geometry, not a bare Xform

All GPU-free: direct Session calls plus a stub render backend.
"""
from __future__ import annotations

import gc
import os
import shutil
import threading
import time

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from usd_cli.parsing import drop_none
from usd_core.config import Config
from usd_core.render.base import RenderResult
from usd_core.session import Session


# ── helpers ─────────────────────────────────────────────────────────────────────


def _write_scene(tmp_path, name="scene.usda", n_cubes=1):
    """A /World with cube prim(s), exported to disk (format from the suffix)."""
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for i in range(n_cubes):
        suffix = "" if i == 0 else f"_{i}"
        UsdGeom.Cube.Define(stage, f"/World/cube{suffix}")
    path = tmp_path / name
    stage.GetRootLayer().Export(str(path))
    return path


def _write_material_library(tmp_path, name="library.usda"):
    """A standalone material library USD with one preview-surface material."""
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Scope.Define(stage, "/Materials")
    stage.SetDefaultPrim(root.GetPrim())
    mat = UsdShade.Material.Define(stage, "/Materials/RedMetal")
    shader = UsdShade.Shader.Define(stage, "/Materials/RedMetal/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.8, 0.05, 0.05))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    path = tmp_path / name
    stage.GetRootLayer().Export(str(path))
    return path


def _session_on(tmp_path, path, name="default", read_only=False) -> Session:
    s = Session(Config(project_dir=tmp_path), name=name)
    s.open_stage(str(path), read_only=read_only)
    return s


def _reopen_and_read(path) -> Usd.Stage:
    """Fully re-read `path` from its bytes (bypassing the process layer cache) and
    prove every prim + authored attribute value is readable."""
    cached = Sdf.Layer.Find(str(path))
    if cached is not None:
        cached.Reload(force=True)
    stage = Usd.Stage.Open(str(path))
    assert stage
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        for attr in prim.GetAttributes():
            attr.Get()  # a corrupt crate explodes exactly here (zero-copy ranges)
    return stage


class _StubBackend:
    """Minimal render backend that just writes the requested PNGs."""

    name = "stub"

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


def test_orbit_render_preserves_each_transient_camera_pose(tmp_path, monkeypatch):
    from usd_core import render as render_pkg

    monkeypatch.setattr(render_pkg, "make_backend", lambda _config: _StubBackend())
    session = _session_on(tmp_path, _write_scene(tmp_path))

    response = session.render(
        orbit=3,
        res="64x64",
        output=str(tmp_path / "orbit"),
    )

    assert response.ok, response.issues
    assert len(response.data["results"]) == 3
    assert len({tuple(item["camera_pos"]) for item in response.data["results"]}) == 3
    for item in response.data["results"]:
        assert item["camera"].startswith("/")
        assert len(item["camera_world_transform"]) == 4
        assert all(len(row) == 4 for row in item["camera_world_transform"])
        assert len(item["camera_dir"]) == 3


def test_plain_render_preserves_camera_pose_captured_at_dispatch(tmp_path, monkeypatch):
    from usd_core import render as render_pkg

    monkeypatch.setattr(render_pkg, "make_backend", lambda _config: _StubBackend())
    session = _session_on(tmp_path, _write_scene(tmp_path))
    assert session.camera_create(name="dispatch").ok
    dispatch_pose = session._camera_pose()
    assert dispatch_pose is not None
    moved_pose = {
        **dispatch_pose,
        "camera_pos": [999.0, 999.0, 999.0],
    }
    calls = 0

    def camera_pose(_camera_path=None):
        nonlocal calls
        calls += 1
        return dispatch_pose if calls == 1 else moved_pose

    monkeypatch.setattr(session, "_camera_pose", camera_pose)
    response = session.render(res="64x64", output=str(tmp_path / "plain.png"))

    assert response.ok, response.issues
    assert response.data["results"][0]["camera_pos"] == dispatch_pose["camera_pos"]


# ── item 1: corrupt-save protection ─────────────────────────────────────────────


def test_in_place_save_over_own_open_usdc_survives_reopen(tmp_path):
    """The v4 killer: `save` over the session's own open crate must read back clean."""
    path = _write_scene(tmp_path, "scene.usdc")
    s = _session_on(tmp_path, path)
    s.camera_fit(None)  # authors a managed ov_cam (exercises the strip-on-save path)
    assert s.transform("/World/cube", translate=[1.0, 2.0, 3.0]).ok
    resp = s.save()  # in-place, no path argument
    assert resp.ok, resp.issues
    assert resp.summary["path"] == str(path)
    assert resp.summary["verified_prims"] >= 2  # read-back verification really ran
    s.new()  # release the file so a fresh session can open it
    del s
    gc.collect()

    stage = _reopen_and_read(path)  # every prim + attribute value readable
    trs = stage.GetPrimAtPath("/World/cube").GetAttribute("xformOp:translate").Get()
    assert list(trs) == [1.0, 2.0, 3.0]
    # the managed render camera was stripped from the deliverable
    assert not [p for p in stage.Traverse() if p.GetTypeName() == "Camera"]
    # and the reopened file passes stage validation in a new session
    s2 = _session_on(tmp_path, path, name="verifier")
    assert s2.validate_stage().ok
    s2.new()


def test_explicit_save_to_own_open_path_survives_reopen(tmp_path):
    """`save <same file>` (the exact benchmark invocation) gets the same protection."""
    path = _write_scene(tmp_path, "explicit.usdc")
    s = _session_on(tmp_path, path)
    s.camera_fit(None)
    assert s.transform("/World/cube", translate=[4.0, 5.0, 6.0]).ok
    resp = s.save(str(path))
    assert resp.ok, resp.issues
    assert resp.summary["verified_prims"] >= 2
    # the live session keeps working after the in-place replace (layer reloaded)
    assert s.snapshot().ok
    s.new()
    del s
    gc.collect()

    stage = _reopen_and_read(path)
    trs = stage.GetPrimAtPath("/World/cube").GetAttribute("xformOp:translate").Get()
    assert list(trs) == [4.0, 5.0, 6.0]


def test_save_verification_failure_leaves_target_untouched(tmp_path, monkeypatch):
    path = _write_scene(tmp_path, "precious.usdc")
    original = path.read_bytes()
    s = _session_on(tmp_path, path)
    s.transform("/World/cube", translate=[9.0, 9.0, 9.0])

    def bad_verify(p):
        raise RuntimeError("verification of the written file failed (simulated)")

    monkeypatch.setattr(Session, "_verify_usd_file", staticmethod(bad_verify))
    resp = s.save()
    assert resp.ok is False
    assert any("verification" in i.message for i in resp.issues)
    assert path.read_bytes() == original  # target never replaced
    # no work/temp litter left beside the target
    leftovers = [p.name for p in tmp_path.iterdir()
                 if "dscsave" in p.name or "dsctmp" in p.name]
    assert leftovers == []
    s.new()


def test_open_truncated_crate_fails_with_clean_error(tmp_path):
    path = _write_scene(tmp_path, "broken.usdc", n_cubes=8)
    size = path.stat().st_size
    with path.open("r+b") as fh:
        fh.truncate(size // 2)
    s = Session(Config(project_dir=tmp_path), name="opener")
    with pytest.raises(RuntimeError, match=r"appears corrupt.*USD crate error.*"
                                           r"checkpoint|re-save"):
        s.open_stage(str(path))
    # the failed open holds no registry claim: another session may open other files
    assert s._layer_key is None and s._stage is None


def test_open_missing_file_still_reports_file_not_found(tmp_path):
    s = Session(Config(project_dir=tmp_path), name="opener")
    with pytest.raises(FileNotFoundError, match="failed to open USD file"):
        s.open_stage(str(tmp_path / "nope.usdc"))


# ── item 2: drop_none keeps zeros ───────────────────────────────────────────────


def test_drop_none_keeps_zero_valued_options():
    payload = {"elevation": 0, "metallic": 0.0, "roughness": 0.0, "opacity": 0.0,
               "orbit": 0, "factor": -0.0}
    assert drop_none(payload) == payload  # nothing zero-valued vanishes


def test_drop_none_still_drops_unset_values():
    assert drop_none({"a": None, "b": (), "c": [], "d": False}) == {}


def test_drop_none_keeps_true_strings_and_nonempty_lists():
    payload = {"flag": True, "name": "", "refs": ["@n1"], "vec": (0.0, 0.0, 0.0)}
    assert drop_none(payload) == payload


# ── item 3: save --flatten ──────────────────────────────────────────────────────


def test_save_flatten_localizes_library_reference(tmp_path):
    lib = _write_material_library(tmp_path)
    path = _write_scene(tmp_path, "asset.usda")
    s = _session_on(tmp_path, path)
    resp = s.material("/World/cube", library=str(lib), name="RedMetal")
    assert resp.ok, resp.issues
    out = tmp_path / "delivery" / "self_contained.usdc"
    out.parent.mkdir()
    resp = s.save(str(out), flatten=True)
    assert resp.ok, resp.issues
    assert resp.summary["flattened"] is True
    assert resp.summary["verified_prims"] >= 2
    s.new()
    del s
    gc.collect()

    # move the flattened file to a bare directory (no library.usda anywhere near it)
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    solo = isolated / out.name
    shutil.copy2(out, solo)
    stage = _reopen_and_read(solo)
    assert not stage.GetCompositionErrors()  # nothing external left to resolve
    mats = [p for p in stage.Traverse() if p.GetTypeName() == "Material"
            and p.GetName() == "RedMetal"]
    assert mats, "library material was not localized into the flattened save"
    shader = next(c for c in mats[0].GetChildren() if c.GetTypeName() == "Shader")
    color = shader.GetAttribute("inputs:diffuseColor").Get()
    assert color and abs(color[0] - 0.8) < 1e-5  # real library values, not a re-guess
    # the cube is still bound to it
    bound = UsdShade.MaterialBindingAPI(
        stage.GetPrimAtPath("/World/cube")).ComputeBoundMaterial()[0]
    assert bound and bound.GetPrim().GetName() == "RedMetal"


def test_save_flatten_without_path_on_memory_stage_errors(tmp_path):
    s = Session(Config(project_dir=tmp_path), name="mem")
    s.new()
    resp = s.save(flatten=True)
    assert resp.ok is False
    assert any("--flatten" in i.message for i in resp.issues)


# ── item 4: open --read-only ────────────────────────────────────────────────────


def test_two_readers_alongside_one_writer_see_live_edits(tmp_path):
    path = _write_scene(tmp_path)
    writer = _session_on(tmp_path, path, name="writer")
    assert writer.transform("/World/cube", translate=[1.0, 2.0, 3.0]).ok  # unsaved

    r1 = Session(Config(project_dir=tmp_path), name="reader-1")
    resp = r1.open_stage(str(path), read_only=True)
    assert resp.ok and resp.summary["read_only"] is True
    r2 = Session(Config(project_dir=tmp_path), name="reader-2")
    assert r2.open_stage(str(path), read_only=True).ok  # multiple readers coexist

    # readers share the writer's LIVE layer: the unsaved edit is visible...
    trs = r1._stage.GetPrimAtPath("/World/cube").GetAttribute("xformOp:translate").Get()
    assert list(trs) == [1.0, 2.0, 3.0]
    # ...and the writer's unsaved edit survived both read-only opens (no force-reload)
    trs_w = writer._stage.GetPrimAtPath("/World/cube") \
        .GetAttribute("xformOp:translate").Get()
    assert list(trs_w) == [1.0, 2.0, 3.0]
    # perception works in readers
    assert r1.snapshot().ok and r2.find(type="Cube").ok
    writer.new()
    r1.new()
    r2.new()


def test_read_only_session_blocks_every_mutating_command(tmp_path):
    path = _write_scene(tmp_path)
    reader = _session_on(tmp_path, path, name="ro", read_only=True)
    blocked = [
        reader.transform("/World/cube", translate=[1.0, 0.0, 0.0]),
        reader.create(type="xform", name="X"),
        reader.delete(["/World/cube"]),
        reader.material("/World/cube", color=[1.0, 0.0, 0.0]),
        reader.hide(["/World/cube"]),
        reader.show(["/World/cube"]),
        reader.rename("/World/cube", "kube"),
        reader.set("/World/cube", "size", "3.0"),
        reader.save(),
        reader.undo(),
        reader.sublayers(drop_dead=True),
    ]
    for resp in blocked:
        assert resp.ok is False, resp.command
        assert any("read-only (opened with --read-only)" in i.message
                   for i in resp.issues), (resp.command, resp.issues)
    # the shared layer really is untouched
    prim = reader._stage.GetPrimAtPath("/World/cube")
    assert prim.IsValid() and prim.IsActive()
    assert prim.GetAttribute("xformOp:translate").Get() is None
    reader.new()


def test_writer_open_is_allowed_while_readers_exist(tmp_path):
    path = _write_scene(tmp_path)
    reader = _session_on(tmp_path, path, name="early-reader", read_only=True)
    writer = Session(Config(project_dir=tmp_path), name="late-writer")
    assert writer.open_stage(str(path)).ok  # readers never block a writer
    assert writer.transform("/World/cube", translate=[7.0, 0.0, 0.0]).ok
    # the pre-existing reader observes the writer's live edit
    trs = reader._stage.GetPrimAtPath("/World/cube") \
        .GetAttribute("xformOp:translate").Get()
    assert list(trs) == [7.0, 0.0, 0.0]
    writer.new()
    reader.new()


def test_plain_reopen_clears_read_only(tmp_path):
    path = _write_scene(tmp_path)
    other = _write_scene(tmp_path, "other.usda")
    s = _session_on(tmp_path, path, name="flip", read_only=True)
    assert s.transform("/World/cube", translate=[1.0, 0.0, 0.0]).ok is False
    assert s.open_stage(str(other)).ok  # plain open → writable again
    assert s.read_only is False
    assert s.transform("/World/cube", translate=[1.0, 0.0, 0.0]).ok
    s.new()


# ── item 5: convert USD→USD honors an explicit output ───────────────────────────


def test_convert_usd_passthrough_writes_explicit_output(tmp_path):
    src = _write_scene(tmp_path, "in.usdc")
    out = tmp_path / "out.usda"
    s = Session(Config(project_dir=tmp_path), name="conv")
    resp = s.convert(str(src), output=str(out))
    assert resp.ok, resp.issues
    assert resp.summary["route"] == "usd-export"
    assert resp.summary["output"] == str(out)
    assert resp.data["output_usd_path"] == str(out)
    assert out.exists()
    stage = Usd.Stage.Open(str(out))
    assert stage.GetPrimAtPath("/World/cube").IsValid()


def test_convert_usd_output_format_writes_sibling(tmp_path):
    src = _write_scene(tmp_path, "in2.usdc")
    s = Session(Config(project_dir=tmp_path), name="conv")
    resp = s.convert(str(src), output_format="usda")
    assert resp.ok, resp.issues
    sibling = tmp_path / "in2.usda"
    assert sibling.exists()
    assert resp.data["output_usd_path"] == str(sibling)


def test_convert_usd_without_output_stays_passthrough(tmp_path):
    src = _write_scene(tmp_path, "in3.usda")
    before = {p.name for p in tmp_path.iterdir()}
    s = Session(Config(project_dir=tmp_path), name="conv")
    resp = s.convert(str(src))
    assert resp.ok and resp.summary["route"] == "passthrough"
    assert {p.name for p in tmp_path.iterdir()} == before  # nothing new written


def test_convert_usd_to_unsupported_output_errors_clearly(tmp_path):
    src = _write_scene(tmp_path, "in4.usda")
    s = Session(Config(project_dir=tmp_path), name="conv")
    resp = s.convert(str(src), output=str(tmp_path / "mesh.obj"))
    assert resp.ok is False
    assert any(".usd/.usda/.usdc/.usdz" in i.message for i in resp.issues)


# ── item 6: hidden focus target says so ─────────────────────────────────────────


def test_focus_on_prim_with_hidden_geometry_names_the_cause(tmp_path):
    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    assert s.hide(["/World/cube"]).ok
    # focusing an ancestor of hidden-only geometry used to claim "no geometry"
    with pytest.raises(RuntimeError, match=r"1 prim\(s\) under the focus target are "
                                           r"hidden — `show` them first"):
        s.camera_fit(["/World"])
    s.new()


def test_focus_on_truly_empty_prim_keeps_no_geometry_message(tmp_path):
    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    assert s.create(type="xform", name="Empty").ok
    with pytest.raises(RuntimeError) as exc:
        s.camera_fit(["/World/Empty"])
    assert "no geometry under those prims" in str(exc.value)
    assert "hidden" not in str(exc.value)
    s.new()


# ── item 7: remove-api summary on a property-less schema ────────────────────────


def test_remove_api_without_authored_properties_reads_as_removal(tmp_path):
    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    prim = s._stage.GetPrimAtPath("/World/cube")
    UsdPhysics.RigidBodyAPI.Apply(prim)  # applied, but no properties authored
    resp = s.remove_api("/World/cube", "RigidBodyAPI")
    assert resp.ok
    assert resp.summary["result"] == "api removed (no authored properties)"
    assert "properties_removed" not in resp.summary  # no misleading ': 0'
    assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    s.new()


def test_remove_api_with_authored_properties_keeps_the_count(tmp_path):
    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    prim = s._stage.GetPrimAtPath("/World/cube")
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(5.0)
    resp = s.remove_api("/World/cube", "MassAPI")
    assert resp.ok
    assert resp.summary["properties_removed"] == 1
    assert "result" not in resp.summary
    s.new()


# ── item 8: render config hot-reload ────────────────────────────────────────────


def _write_render_config(tmp_path, **render_keys):
    lines = ["[render]"] + [
        f'{k} = {v if not isinstance(v, str) else chr(34) + v + chr(34)}'
        for k, v in render_keys.items()
    ]
    cfg_dir = tmp_path / ".usd-cli"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config.toml").write_text("\n".join(lines) + "\n")


def test_render_config_rereads_project_file_each_call(tmp_path):
    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    _write_render_config(tmp_path, remote_max_upload_mb=7)
    assert s._render_config(None).render["remote_max_upload_mb"] == 7
    _write_render_config(tmp_path, remote_max_upload_mb=21)  # agent edits mid-session
    assert s._render_config(None).render["remote_max_upload_mb"] == 21
    # the session's own deep-copied config is untouched (everything-else contract)
    assert s.config.render["remote_max_upload_mb"] == 512
    s.new()


def test_workflow_confined_render_config_cannot_be_hot_reloaded(
    tmp_path,
    monkeypatch,
):
    from usd_core.config import DEFAULTS

    safe_url = "https://attested-renderer.example.test"
    config = Config(
        project_dir=tmp_path,
        render={
            **DEFAULTS["render"],
            "renderer": "remote",
            "remote_url": safe_url,
            "remote_max_upload_mb": 64,
        },
    )
    session = Session(config, name="confined")
    monkeypatch.setenv("USD_CLI_LOCK_RENDER_CONFIG", "1")
    _write_render_config(
        tmp_path,
        remote_url="https://child-controlled.example.test",
        remote_max_upload_mb=4096,
    )
    render = session._fresh_render_section()
    assert render["remote_url"] == safe_url
    assert render["remote_max_upload_mb"] == 64
    assert session.config.render["remote_url"] == safe_url


def test_render_path_constructs_backend_with_fresh_config(tmp_path, monkeypatch):
    import usd_core.render as render_pkg
    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    seen = []

    def factory(cfg):
        seen.append(cfg.render.get("remote_max_upload_mb"))
        return _StubBackend()

    monkeypatch.setattr(render_pkg, "make_backend", factory)
    _write_render_config(tmp_path, remote_max_upload_mb=64)
    assert s.render(res=[64, 64]).ok
    _write_render_config(tmp_path, remote_max_upload_mb=1024)  # edited between renders
    assert s.render(res=[64, 64]).ok
    assert seen == [64, 1024]
    s.new()


def test_render_defaults_to_configured_quality_mode(tmp_path, monkeypatch):
    import usd_core.render as render_pkg

    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    seen = []

    class CapturingBackend(_StubBackend):
        def render(self, stage, cameras, width, height, out_dir, mode="quality",
                   names=None, frame=None):
            seen.append(mode)
            return super().render(
                stage, cameras, width, height, out_dir,
                mode=mode, names=names, frame=frame,
            )

    monkeypatch.setattr(render_pkg, "make_backend", lambda cfg: CapturingBackend())

    assert s.render(res=[64, 64]).ok
    assert s.render_frames(frames="0", res=[64, 64], animate=False).ok
    assert seen == ["quality", "quality"]
    s.new()


def test_render_mode_can_be_overridden_explicitly(tmp_path, monkeypatch):
    import usd_core.render as render_pkg

    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    seen = []

    class CapturingBackend(_StubBackend):
        def render(self, stage, cameras, width, height, out_dir, mode="quality",
                   names=None, frame=None):
            seen.append(mode)
            return super().render(
                stage, cameras, width, height, out_dir,
                mode=mode, names=names, frame=frame,
            )

    monkeypatch.setattr(render_pkg, "make_backend", lambda cfg: CapturingBackend())

    assert s.render(res=[64, 64], mode="fast").ok
    assert seen == ["fast"]
    s.new()


def test_renderer_override_still_rides_an_ephemeral_copy(tmp_path):
    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    cfg = s._render_config("ovrtx")
    assert cfg.render["renderer"] == "ovrtx"
    assert cfg is not s.config
    assert s.config.render["renderer"] == "auto"
    s.new()


# ── item 9: ill-formed refs error cleanly; --exclude fails loudly ───────────────


def test_bare_number_ref_errors_with_did_you_mean_hint(tmp_path):
    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    resp = s.transform("6346", translate=[1.0, 0.0, 0.0])
    assert resp.ok is False
    assert any("did you mean '@n6346'?" in i.message for i in resp.issues), resp.issues
    resp = s.properties("cube")  # relative name: clean error, no pxr warning spam
    assert resp.ok is False
    assert any("not a ref or an absolute prim path" in i.message
               for i in resp.issues), resp.issues
    s.new()


def test_render_exclude_with_bogus_ref_fails_loudly(tmp_path, monkeypatch):
    import usd_core.render as render_pkg
    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    monkeypatch.setattr(render_pkg, "make_backend", lambda cfg: _StubBackend())
    with pytest.raises(ValueError, match=r"--exclude /World/nope: no prim"):
        s.render(res=[64, 64], exclude=["/World/nope"])
    with pytest.raises(ValueError, match=r"did you mean '@n6346'\?"):
        s.render(res=[64, 64], exclude=["6346"])
    # a valid exclusion still renders (and restores visibility afterwards)
    assert s.render(res=[64, 64], exclude=["/World/cube"]).ok
    vis = s._stage.GetPrimAtPath("/World/cube").GetAttribute("visibility")
    assert not vis.HasAuthoredValue()
    s.new()


# ════════════════════════════════════════════════════════════════════════════════
# review-v5 follow-ups (numbering from .codex-review-v5-findings.txt)
# ════════════════════════════════════════════════════════════════════════════════


def _corrupt_value_crate(tmp_path, name="valuecorrupt.usdc"):
    """A crate whose STRUCTURE (prim indexes) parses but whose value blocks are
    damaged: it opens cleanly and shallow-traverses cleanly, and only explodes
    when an attribute value is actually read (lz4 decompression failure)."""
    import random

    from pxr import Vt

    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/mesh")
    rnd = random.Random(7)
    mesh.CreateFaceVertexIndicesAttr().Set(
        Vt.IntArray([rnd.randrange(0, 1 << 30) for _ in range(200000)]))
    UsdGeom.Cube.Define(stage, "/World/cube")
    path = tmp_path / name
    stage.GetRootLayer().Export(str(path))
    size = path.stat().st_size
    data = bytearray(path.read_bytes())
    # keep the bootstrap header (start) and the structural sections + TOC (end)
    # intact; wipe the value-block region in between
    lo, hi = 512, size - 16384
    data[lo:hi] = bytes([0xA5]) * (hi - lo)
    path.write_bytes(bytes(data))
    return path


def _write_scene_with_camera(tmp_path, name="cam_scene.usda"):
    """A scene that already CONTAINS an authored camera (for read-only renders)."""
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/cube")
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    UsdGeom.XformCommonAPI(cam.GetPrim()).SetTranslate((0.0, 0.0, 10.0))
    path = tmp_path / name
    stage.GetRootLayer().Export(str(path))
    return path


# ── v5 item 1: a failed candidate open leaves the old stage owned ───────────────


def test_failed_candidate_open_keeps_old_stage_owned(tmp_path):
    good = _write_scene(tmp_path, "good.usda")
    s = _session_on(tmp_path, good, name="holder")
    assert s.transform("/World/cube", translate=[1.0, 0.0, 0.0]).ok  # unsaved work
    bad = _corrupt_value_crate(tmp_path, "bad_candidate.usdc")
    with pytest.raises(RuntimeError, match="appears corrupt"):
        s.open_stage(str(bad))
    # the OLD stage is still live, still registered, and its unsaved edit survived
    assert s._stage_path == str(good)
    assert s._layer_key == os.path.realpath(str(good))
    trs = s._stage.GetPrimAtPath("/World/cube").GetAttribute("xformOp:translate").Get()
    assert list(trs) == [1.0, 0.0, 0.0]
    # another session still cannot grab the file (ownership was never released)
    other = Session(Config(project_dir=tmp_path), name="intruder")
    with pytest.raises(RuntimeError, match="open in session 'holder'"):
        other.open_stage(str(good))
    assert s.snapshot().ok  # the session keeps working
    s.new()


# ── v5 item 2: cross-session publish refusal + per-destination serialization ────


def test_publish_refuses_another_sessions_open_file(tmp_path):
    theirs = _write_scene(tmp_path, "theirs.usdc")
    mine = _write_scene(tmp_path, "mine.usda")
    original = theirs.read_bytes()
    holder = _session_on(tmp_path, theirs, name="holder")
    s = _session_on(tmp_path, mine, name="writer2")
    for resp in (s.save(str(theirs)),
                 s.export("usdc", str(theirs)),
                 s.convert(str(mine), output=str(theirs))):
        assert resp.ok is False, resp.command
        assert any("open (read-write) in session 'holder'" in i.message
                   for i in resp.issues), (resp.command, resp.issues)
    assert theirs.read_bytes() == original  # the live file was never touched
    holder.new()
    s.new()


def test_concurrent_publishes_to_one_destination_serialize(tmp_path, monkeypatch):
    a = _session_on(tmp_path, _write_scene(tmp_path, "pub_a.usda"), name="pub-a")
    b = _session_on(tmp_path, _write_scene(tmp_path, "pub_b.usda"), name="pub-b")
    dest = tmp_path / "contested.usda"
    gate = threading.Semaphore(1)
    overlaps: list[str] = []
    real_verify = Session._verify_usd_file

    def slow_verify(path):
        if not gate.acquire(blocking=False):
            overlaps.append(str(path))  # a second publish entered the window
            return real_verify(path)
        try:
            time.sleep(0.05)
            return real_verify(path)
        finally:
            gate.release()

    monkeypatch.setattr(Session, "_verify_usd_file", staticmethod(slow_verify))
    barrier = threading.Barrier(2)
    results: dict[str, object] = {}

    def go(sess, key):
        barrier.wait()
        results[key] = sess.save(str(dest))

    threads = [threading.Thread(target=go, args=(a, "a")),
               threading.Thread(target=go, args=(b, "b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results["a"].ok and results["b"].ok
    assert overlaps == []  # export/verify/replace windows never interleaved
    assert Usd.Stage.Open(str(dest)).GetPrimAtPath("/World/cube").IsValid()
    a.new()
    b.new()


# ── v5 item 3: symlink destinations write through to the target ─────────────────


def test_save_through_symlink_preserves_the_link(tmp_path):
    target = _write_scene(tmp_path, "link_target.usda")
    link = tmp_path / "link.usda"
    link.symlink_to(target)
    s = _session_on(tmp_path, link)  # opened via the symlink
    assert s.transform("/World/cube", translate=[2.0, 0.0, 0.0]).ok
    resp = s.save()  # in place, through the link
    assert resp.ok, resp.issues
    assert resp.summary.get("destination") == os.path.realpath(str(target))
    assert link.is_symlink()  # the link itself survived the replace
    assert os.path.realpath(str(link)) == os.path.realpath(str(target))
    s.new()
    del s
    gc.collect()
    stage = _reopen_and_read(target)
    trs = stage.GetPrimAtPath("/World/cube").GetAttribute("xformOp:translate").Get()
    assert list(trs) == [2.0, 0.0, 0.0]


def test_publish_via_symlink_alias_of_open_file_is_refused(tmp_path):
    target = _write_scene(tmp_path, "owned.usda")
    alias = tmp_path / "alias.usda"
    alias.symlink_to(target)
    holder = _session_on(tmp_path, target, name="holder2")
    s = _session_on(tmp_path, _write_scene(tmp_path, "src2.usda"), name="w2")
    resp = s.save(str(alias))  # the alias resolves to holder2's live file
    assert resp.ok is False
    assert any("'holder2'" in i.message for i in resp.issues), resp.issues
    holder.new()
    s.new()


# ── v5 item 4: verification force-reads values / time samples ────────────────────


def test_open_value_corrupted_crate_fails_cleanly(tmp_path):
    bad = _corrupt_value_crate(tmp_path)
    # structure parses: a shallow prim-index traversal reads right past the damage
    probe = Usd.Stage.Open(str(bad))
    assert probe
    assert sum(1 for _ in probe.Traverse(Usd.TraverseInstanceProxies())) == 3
    del probe
    gc.collect()
    s = Session(Config(project_dir=tmp_path), name="opener")
    with pytest.raises(RuntimeError, match="appears corrupt"):
        s.open_stage(str(bad))
    assert s._layer_key is None and s._stage is None


def test_deep_read_covers_time_samples_and_relationships(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/World/cube")
    attr = cube.GetSizeAttr()
    attr.Set(1.0, 0.0)
    attr.Set(2.0, 24.0)
    cube.GetPrim().CreateRelationship("dsc:probe", custom=True).SetTargets(["/World"])
    n = Session._deep_read_stage(stage)
    assert n == 2  # both prims visited; sampled + relationship reads succeeded


# ── v5 item 5: same-file save cache rebuild + post-commit reload warnings ────────


def test_same_file_save_rebuilds_camera_cache(tmp_path):
    path = _write_scene(tmp_path, "camcache.usdc")
    s = _session_on(tmp_path, path)
    s.camera_fit(None)  # authors the managed ov_cam and makes it active
    cam_before = s._active_cam
    assert cam_before and s._stage.GetPrimAtPath(cam_before).IsValid()
    resp = s.save()
    assert resp.ok, resp.issues
    # the managed camera was stripped from the file, and the post-save reload
    # dropped it from the live stage — the cache must not point at the dead prim
    assert not s._stage.GetPrimAtPath(cam_before).IsValid()
    if s._active_cam is not None:
        assert s._stage.GetPrimAtPath(s._active_cam).IsValid()
    assert s.camera_fit(None).ok  # camera commands keep working immediately
    s.new()


def test_reload_failure_after_commit_warns_not_errors(tmp_path, monkeypatch):
    path = _write_scene(tmp_path, "commit.usdc")
    s = _session_on(tmp_path, path)
    assert s.transform("/World/cube", translate=[3.0, 3.0, 3.0]).ok

    def boom(layer):
        raise RuntimeError("reload exploded")

    monkeypatch.setattr(Session, "_force_reload_layer", staticmethod(boom))
    resp = s.save()
    assert resp.ok is True, resp.issues  # the replace committed — never 'save failed'
    warns = [i for i in resp.issues if i.severity == "warn"]
    assert warns and "committed to disk" in warns[0].message
    assert "open " in warns[0].message  # actionable: reopen the stage
    monkeypatch.undo()
    s.new()
    del s
    gc.collect()
    stage = _reopen_and_read(path)  # the bytes on disk ARE the verified save
    trs = stage.GetPrimAtPath("/World/cube").GetAttribute("xformOp:translate").Get()
    assert list(trs) == [3.0, 3.0, 3.0]


# ── v5 item 6: read-only enforcement gaps ────────────────────────────────────────


def test_read_only_blocks_camera_authoring_commands(tmp_path):
    path = _write_scene(tmp_path)
    reader = _session_on(tmp_path, path, name="ro-cam", read_only=True)
    blocked = [
        reader.camera_fit(None),
        reader.camera_orbit(az=10.0),
        reader.camera_look_at("/World/cube"),
        reader.camera_create(name="C"),
        reader.camera_pan([1.0, 0.0]),
        reader.camera_zoom(2.0),
    ]
    for resp in blocked:
        assert resp.ok is False, resp.command
        assert any("read-only (opened with --read-only)" in i.message
                   for i in resp.issues), (resp.command, resp.issues)
    # nothing was authored on the shared layer
    assert not [p for p in reader._stage.Traverse() if p.GetTypeName() == "Camera"]
    reader.new()


def test_read_only_validate_fix_blocked_but_plain_validate_works(tmp_path):
    path = _write_scene(tmp_path)
    reader = _session_on(tmp_path, path, name="ro-val", read_only=True)
    assert reader.validate_stage().ok  # perception stays available
    resp = reader.validate_stage(fix=True)
    assert resp.ok is False
    assert any("--fix" in i.message for i in resp.issues), resp.issues
    reader.new()


def test_read_only_render_requires_existing_camera(tmp_path, monkeypatch):
    import usd_core.render as render_pkg
    monkeypatch.setattr(render_pkg, "make_backend", lambda cfg: _StubBackend())
    bare = _write_scene(tmp_path, "bare_ro.usda")
    reader = _session_on(tmp_path, bare, name="ro-render", read_only=True)
    resp = reader.render(res=[64, 64])  # would have to author a ov_cam
    assert resp.ok is False
    assert any("read-only session cannot author render cameras" in i.message
               for i in resp.issues), resp.issues
    reader.new()

    with_cam = _write_scene_with_camera(tmp_path)
    reader2 = _session_on(tmp_path, with_cam, name="ro-render2", read_only=True)
    assert reader2.render(res=[64, 64]).ok  # adopted existing camera: no authoring
    assert reader2.render(res=[64, 64], camera="/World/cam").ok  # explicit --camera
    for opts in ({"focus": "/World/cube"}, {"exclude": ["/World/cube"]}, {"orbit": 2}):
        resp = reader2.render(res=[64, 64], **opts)
        assert resp.ok is False, opts
        assert any("read-only session cannot author" in i.message
                   for i in resp.issues), (opts, resp.issues)
    reader2.new()


def test_read_only_export_over_source_is_blocked(tmp_path):
    path = _write_scene(tmp_path, "ro_export.usda")
    writer = _session_on(tmp_path, path, name="rw-holder")
    reader = _session_on(tmp_path, path, name="ro-exporter", read_only=True)
    resp = reader.export("usda", str(path))
    assert resp.ok is False
    assert any("read-only" in i.message for i in resp.issues), resp.issues
    # exporting to a NEW path from a reader stays fine
    out = tmp_path / "reader_copy.usda"
    assert reader.export("usda", str(out)).ok
    assert out.exists()
    writer.new()
    reader.new()


# ── v5 item 8: --exclude resolves/validates/dedupes before hiding ────────────────


def test_exclude_failure_restores_already_hidden_prims(tmp_path, monkeypatch):
    import usd_core.render as render_pkg
    path = _write_scene(tmp_path, "excl.usda", n_cubes=2)
    s = _session_on(tmp_path, path)
    monkeypatch.setattr(render_pkg, "make_backend", lambda cfg: _StubBackend())
    with pytest.raises(ValueError, match="/World/nope"):
        s.render(res=[64, 64], exclude=["/World/cube", "/World/nope"])
    # the valid prim was never hidden (validation happens before any mutation)
    vis = s._stage.GetPrimAtPath("/World/cube").GetAttribute("visibility")
    assert not vis.HasAuthoredValue()
    s.new()


def test_exclude_duplicates_restore_correctly(tmp_path, monkeypatch):
    import usd_core.render as render_pkg
    path = _write_scene(tmp_path, "excl2.usda", n_cubes=2)
    s = _session_on(tmp_path, path)
    monkeypatch.setattr(render_pkg, "make_backend", lambda cfg: _StubBackend())
    # duplicated with no prior opinion: must NOT be left invisible afterwards
    assert s.render(res=[64, 64], exclude=["/World/cube", "/World/cube"]).ok
    attr = s._stage.GetPrimAtPath("/World/cube").GetAttribute("visibility")
    assert not attr.HasAuthoredValue()
    # duplicated with a pre-authored opinion: the opinion survives untouched
    img = UsdGeom.Imageable(s._stage.GetPrimAtPath("/World/cube_1"))
    img.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)  # explicit authored opinion
    assert s.render(res=[64, 64], exclude=["/World/cube_1", "/World/cube_1"]).ok
    attr = img.GetVisibilityAttr()
    assert attr.HasAuthoredValue() and attr.Get() == UsdGeom.Tokens.inherited
    s.new()


# ── v5 item 15: hot-reload precedence (env > project) + removed keys ─────────────


def test_env_override_beats_project_render_config(tmp_path, monkeypatch):
    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    _write_render_config(tmp_path, remote_max_upload_mb=7)
    monkeypatch.setenv("USD_CLI_RENDER_REMOTE_MAX_UPLOAD_MB", "99")
    assert s._render_config(None).render["remote_max_upload_mb"] == "99"  # env wins
    monkeypatch.delenv("USD_CLI_RENDER_REMOTE_MAX_UPLOAD_MB")
    assert s._render_config(None).render["remote_max_upload_mb"] == 7
    s.new()


def test_removed_project_render_keys_fall_back(tmp_path):
    from usd_core.config import DEFAULTS
    _write_render_config(tmp_path, remote_max_upload_mb=7)
    # the session config as load_config would have baked it at startup
    cfg = Config(project_dir=tmp_path,
                 render={**DEFAULTS["render"], "remote_max_upload_mb": 7})
    s = Session(cfg, name="hotreload")
    s.open_stage(str(_write_scene(tmp_path, "hot.usda")))
    assert s._render_config(None).render["remote_max_upload_mb"] == 7
    _write_render_config(tmp_path, remote_timeout=60)  # the key is REMOVED
    render = s._render_config(None).render
    assert render["remote_max_upload_mb"] == 512  # back to the built-in default
    assert render["remote_timeout"] == 60  # the new key is live
    s.new()


# ── v5 item 16: convert through the verified publish pipeline ────────────────────


def test_convert_usd_to_usdz_packages(tmp_path):
    src = _write_scene(tmp_path, "pkg_src.usda")
    out = tmp_path / "pkg.usdz"
    s = Session(Config(project_dir=tmp_path), name="conv-z")
    resp = s.convert(str(src), output=str(out))
    assert resp.ok, resp.issues
    assert resp.summary["route"] == "usd-export"
    assert resp.summary["verified_prims"] >= 2  # the read-back verify really ran
    stage = Usd.Stage.Open(str(out))
    assert stage and stage.GetPrimAtPath("/World/cube").IsValid()


def test_convert_verification_failure_leaves_destination_untouched(tmp_path, monkeypatch):
    src = _write_scene(tmp_path, "conv_src.usdc")
    dest = _write_scene(tmp_path, "conv_dest.usda")  # pre-existing, precious
    original = dest.read_bytes()

    def bad_verify(p):
        raise RuntimeError("verification of the written file failed (simulated)")

    monkeypatch.setattr(Session, "_verify_usd_file", staticmethod(bad_verify))
    s = Session(Config(project_dir=tmp_path), name="conv-v")
    resp = s.convert(str(src), output=str(dest))
    assert resp.ok is False
    assert dest.read_bytes() == original  # never replaced
    assert not [p for p in tmp_path.iterdir() if "dscsave" in p.name]  # no litter


def test_convert_symlink_alias_of_source_stays_passthrough(tmp_path):
    src = _write_scene(tmp_path, "alias_src.usda")
    alias = tmp_path / "alias_of_src.usda"
    alias.symlink_to(src)
    before = {p.name for p in tmp_path.iterdir()}
    s = Session(Config(project_dir=tmp_path), name="conv-a")
    resp = s.convert(str(src), output=str(alias))  # same file through the alias
    assert resp.ok and resp.summary["route"] == "passthrough"
    assert {p.name for p in tmp_path.iterdir()} == before  # nothing rewritten


# ── v5 item 17: destination directory fsync after the rename ─────────────────────


def test_publish_fsyncs_destination_directory(tmp_path, monkeypatch):
    path = _write_scene(tmp_path, "fsync.usda")
    s = _session_on(tmp_path, path)
    seen: list[str] = []
    monkeypatch.setattr(Session, "_fsync_dir", staticmethod(lambda p: seen.append(p)))
    assert s.save(str(tmp_path / "fs_out.usda")).ok
    assert seen == [os.path.dirname(os.path.realpath(str(tmp_path / "fs_out.usda")))]
    s.new()


# ── v5 item 23: property / variant-selection paths error clearly ─────────────────


def test_property_and_variant_paths_error_clearly(tmp_path):
    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    resp = s.properties("/World/cube.size")
    assert resp.ok is False
    assert any("property path" in i.message for i in resp.issues), resp.issues
    resp = s.transform("/World{look=red}cube", translate=[1.0, 0.0, 0.0])
    assert resp.ok is False
    assert any("variant selection" in i.message for i in resp.issues), resp.issues
    assert s.properties("/World/cube").ok  # plain prim paths still resolve
    s.new()


# ── v5 item 24: save --flatten wording notes referenced external files ───────────


def test_save_flatten_notes_external_files_stay_referenced(tmp_path):
    path = _write_scene(tmp_path, "flatnote.usda")
    s = _session_on(tmp_path, path)
    resp = s.save(str(tmp_path / "flat_out.usda"), flatten=True)
    assert resp.ok, resp.issues
    assert any(i.severity == "info" and "not embedded" in i.message
               for i in resp.issues), resp.issues
    s.new()


# ── v5 item 25: hidden-focus hint requires hidden BOUNDABLE geometry ─────────────


def test_hidden_focus_hint_requires_boundable_geometry(tmp_path):
    path = _write_scene(tmp_path)
    s = _session_on(tmp_path, path)
    assert s.create(type="xform", name="Group").ok
    assert s.create(type="xform", name="EmptyChild", parent="/World/Group").ok
    assert s.hide(["/World/Group/EmptyChild"]).ok
    with pytest.raises(RuntimeError) as exc:
        s.camera_fit(["/World/Group"])
    # a hidden bare Xform is not renderable — no misleading `show` hint
    assert "no geometry under those prims" in str(exc.value)
    assert "hidden" not in str(exc.value)
    s.new()
