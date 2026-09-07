# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-6 fixes (benchmark round 5 findings + the usd-cli rename compat layer).

Covers, unit-level:
1. Pool health persists across construction (daemon-restart amnesia).
2. Legacy `dsc_remote_render_*` staging dirs still parse for the reaper.
3. Managed-camera specs (protocol v2): extraction, strip, camera-independent
   fingerprints for the bundle cache.
4. Read-only mutating commands fast-fail via MUTATING_COMMANDS (no queueing).
5. Re-opening a file with unsaved edits is refused without --force-reload.
6. Published files keep the destination's permissions (not mkstemp 0600).
7. A failed render removes the output directory it created (litter).
8. `convert` reads the on-disk source, never a live dirty layer.
9. Transmissive library materials carry a usable UsdPreviewSurface fallback.
"""

from __future__ import annotations

import os
import stat as stat_mod
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from usd_core import camera as ov_camera  # noqa: E402
from usd_core.render import remote  # noqa: E402
from usd_core.session import Session  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _tiny_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    upAxis = "Z"
    defaultPrim = "World"
)
def Xform "World"
{
    def Cube "box"
    {
        double size = 1
    }
}
""")


# ── 1. pool-state persistence ────────────────────────────────────────────────

def test_pool_state_persists_failure_across_restart(tmp_path):
    persist = tmp_path / "pool.json"
    slots = [remote._BackendSlot("http://a:8000"), remote._BackendSlot("http://b:8000")]
    state = remote._PoolState(slots, persist_path=persist)
    state.record_upload(slots[0], 10_000_000, 2.0)
    state.record_failure(slots[1])
    assert persist.exists()

    fresh = [remote._BackendSlot("http://a:8000"), remote._BackendSlot("http://b:8000")]
    restored = remote._PoolState(fresh, persist_path=persist)
    assert fresh[0].recent_bps, "throughput samples must survive a restart"
    assert fresh[1].recent_failure, "an in-cooldown failure must survive a restart"
    assert restored is not None


def test_pool_state_expired_entries_ignored(tmp_path, monkeypatch):
    persist = tmp_path / "pool.json"
    slots = [remote._BackendSlot("http://a:8000"), remote._BackendSlot("http://b:8000")]
    state = remote._PoolState(slots, persist_path=persist)
    state.record_failure(slots[0])
    # age the file beyond the TTL
    import json as _json
    raw = _json.loads(persist.read_text())
    raw["saved_at"] -= remote._POOL_STATE_TTL_S + 10
    persist.write_text(_json.dumps(raw))
    fresh = [remote._BackendSlot("http://a:8000"), remote._BackendSlot("http://b:8000")]
    remote._PoolState(fresh, persist_path=persist)
    assert not fresh[0].recent_failure


# ── 2. legacy staging prefix ─────────────────────────────────────────────────

def test_staging_pid_parses_both_prefixes():
    assert remote._staging_pid("ov_remote_render_123_ab") == 123
    assert remote._staging_pid("dsc_remote_render_456_cd") == 456
    assert remote._staging_pid("unrelated_789_x") is None


# ── 3. managed cameras: defs, strip, fingerprints ────────────────────────────

def _stage_with_managed_cam(cam_name: str = "ov_cam"):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Cube.Define(stage, "/World/box")
    ov_camera.author_camera(stage, f"/{cam_name}", (5, 5, 5), (0, 0, 0))
    return stage


def test_author_camera_tags_orbit_names_too():
    stage = _stage_with_managed_cam("usd_cam_orbit_03")
    prim = stage.GetPrimAtPath("/usd_cam_orbit_03")
    assert prim.GetCustomData().get("usdManagedCamera") is True


def test_author_camera_tags_legacy_ov_names_as_managed():
    # a legacy-named ov_cam* camera (older sessions, saved stages) is still
    # the tool's — it gets the CURRENT managed key and strips identically
    stage = _stage_with_managed_cam("ov_cam_orbit_03")
    prim = stage.GetPrimAtPath("/ov_cam_orbit_03")
    assert prim.GetCustomData().get("usdManagedCamera") is True


def test_managed_camera_defs_extraction():
    stage = _stage_with_managed_cam()
    defs = remote._managed_camera_defs(stage, ["/ov_cam"])
    assert len(defs) == 1
    spec = defs[0]
    assert spec["path"] == "/ov_cam"
    assert len(spec["matrix"]) == 16
    assert spec.get("focal_length")
    # user cameras (untagged) are never extracted
    UsdGeom.Camera.Define(stage, "/World/user_cam")
    assert remote._managed_camera_defs(stage, ["/World/user_cam"]) == []


def test_managed_camera_defs_tolerates_stubs():
    assert remote._managed_camera_defs(None, ["/x"]) == []


def test_layer_fingerprint_ignores_camera_moves():
    a = _stage_with_managed_cam()
    b = _stage_with_managed_cam()
    ov_camera.author_camera(b, "/ov_cam", (9, 1, 2), (0, 0, 1))  # different view
    ta = remote._layer_text_without_managed_cams(a.GetRootLayer())
    tb = remote._layer_text_without_managed_cams(b.GetRootLayer())
    assert ta == tb, "a moved managed camera must not change the fingerprint"
    # but real scene changes must
    UsdGeom.Sphere.Define(b, "/World/extra")
    assert ta != remote._layer_text_without_managed_cams(b.GetRootLayer())


def test_strip_managed_cameras_spec():
    stage = _stage_with_managed_cam()
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.TransferContent(stage.GetRootLayer())
    assert remote._strip_managed_cameras_spec(layer) == 1
    assert layer.GetPrimAtPath("/ov_cam") is None
    assert layer.GetPrimAtPath("/World/box") is not None


def test_legacy_dsc_tag_still_strips():
    stage = Usd.Stage.CreateInMemory()
    cam = UsdGeom.Camera.Define(stage, "/dsc_cam")
    cam.GetPrim().SetCustomDataByKey("dscManagedCamera", True)
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.TransferContent(stage.GetRootLayer())
    assert remote._strip_managed_cameras_spec(layer) == 1


# ── 4. read-only fast-fail set ───────────────────────────────────────────────

def test_mutating_commands_registry():
    from usd_server.app import MUTATING_COMMANDS, _read_only_reject

    for wire in ("save", "transform", "material", "delete", "checkpoint.load"):
        assert wire in MUTATING_COMMANDS, wire
    for wire in ("snapshot", "find", "render", "open", "describe", "validate"):
        assert wire not in MUTATING_COMMANDS, wire

    class RO:
        read_only = True
        name = "aud"

    rej = _read_only_reject(RO(), "transform")
    assert rej is not None and not rej.ok
    assert "read-only" in rej.issues[0].message
    assert _read_only_reject(RO(), "snapshot") is None


# ── 5. unsaved-edit reopen guard ─────────────────────────────────────────────

def test_reopen_with_unsaved_edits_refused_then_forced(tmp_path):
    f = tmp_path / "scene.usda"
    _tiny_stage(f)
    s = Session.open(str(f))
    r = s.transform("/World/box", tz={"mode": "relative", "value": 0.5})
    assert r.ok, r.issues
    with pytest.raises(RuntimeError, match="force-reload"):
        s.open_stage(str(f))  # the daemon turns this into an error envelope
    forced = s.open_stage(str(f), force_reload=True)
    assert forced.ok
    # a clean session reopens without friction
    again = s.open_stage(str(f))
    assert again.ok


def test_reopen_after_save_is_friction_free(tmp_path):
    f = tmp_path / "scene.usda"
    _tiny_stage(f)
    s = Session.open(str(f))
    assert s.transform("/World/box", tz={"mode": "relative", "value": 0.25}).ok
    assert s.save().ok
    assert s.open_stage(str(f)).ok


# ── 6. published file permissions ────────────────────────────────────────────

def test_publish_preserves_destination_mode(tmp_path):
    f = tmp_path / "scene.usda"
    _tiny_stage(f)
    s = Session.open(str(f))
    out = tmp_path / "out.usda"
    assert s.save(str(out)).ok
    mode = stat_mod.S_IMODE(os.stat(out).st_mode)
    umask = os.umask(0)
    os.umask(umask)
    assert mode == (0o666 & ~umask), f"got {oct(mode)}"
    # overwriting keeps the destination's explicit mode
    os.chmod(out, 0o640)
    assert s.save(str(out)).ok
    assert stat_mod.S_IMODE(os.stat(out).st_mode) == 0o640


# ── 7. failed-render output-dir litter ───────────────────────────────────────

def test_failed_render_removes_created_output_dir(tmp_path):
    f = tmp_path / "scene.usda"
    _tiny_stage(f)
    s = Session.open(str(f))
    target = tmp_path / "_assignment_scene.png"
    with pytest.raises(RuntimeError):
        with s._remove_output_dir_on_failure(target):
            target.mkdir()
            raise RuntimeError("backend fell over")
    assert not target.exists()
    # pre-existing dirs are never removed
    target.mkdir()
    with pytest.raises(RuntimeError):
        with s._remove_output_dir_on_failure(target):
            raise RuntimeError("boom")
    assert target.exists()


# ── 8. convert reads disk, not live layers ──────────────────────────────────

def test_convert_reads_disk_state_not_live_edits(tmp_path):
    src = tmp_path / "src.usda"
    _tiny_stage(src)
    s = Session.open(str(src))
    assert s.transform("/World/box", tz={"mode": "relative", "value": 3.0}).ok  # live, unsaved
    out = tmp_path / "converted.usdc"
    r = s.convert(str(src), str(out))
    assert r.ok, r.issues
    clean = Usd.Stage.Open(str(out))
    box = UsdGeom.Xformable(clean.GetPrimAtPath("/World/box"))
    ops = box.GetOrderedXformOps()
    moved = any(op.Get() and getattr(op.Get(), "__getitem__", None)
                and abs(op.Get()[2] - 3.0) < 1e-6
                for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
    assert not moved, "convert must not see the live session's unsaved transform"
    # and no scratch copies left behind
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".ovconvert_")]


# ── 9. transmissive preview fallbacks ────────────────────────────────────────

def test_glass_preview_fallbacks_are_not_opaque_black():
    lib = REPO / "internal/example_tasks/shared/materials/materials_libs_v2.usd"
    if not lib.exists():
        pytest.skip("shared material library not present")
    stage = Usd.Stage.Open(str(lib))
    for name in ("Glass_Clear", "Glass_Frosted", "Glass_Clear_Tinted_Blue"):
        fb = stage.GetPrimAtPath(f"/World/Looks/{name}/PreviewSurfaceFallback")
        assert fb, name
        op = fb.GetAttribute("inputs:opacity").Get()
        assert op is not None and op < 0.5, f"{name}: opacity {op}"
        dc = fb.GetAttribute("inputs:diffuseColor").Get()
        assert dc is not None and max(dc) >= 0.6, f"{name}: diffuse {dc}"
