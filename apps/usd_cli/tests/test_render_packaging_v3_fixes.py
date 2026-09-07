# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render/packaging fixes, round 3 (unit tests — no GPU, no remote service).

Covers four issues found on the 2026-07-09 benchmark night:

1. Remote packaging loses per-instance material overrides / ancestor-inherited bindings
   on instanced prims *for the renderer* (task-03: Component_Blue on C617, PCB_Default_Dark
   over 269 instance roots came back unbound-gray). pxr composes these bindings fine in
   the packaged USDZ, but OVRTX-style renderers resolve materials per prototype and drop
   opinions authored outside it — so packaging now bakes them as direct gprim bindings.
2. `render.remote_max_upload_mb` defaulted to 0 (unlimited), letting a 1,377 MB bundle
   monopolize the render service; the config default is now 512, with 0 still unlimited.
3. Cancelled/killed remote renders leaked /tmp/ov_remote_render_* staging dirs (2.5 GB
   observed): failures clean up via try/finally, and a best-effort reaper sweeps
   orphaned dirs (older than 24h, or owned by a dead PID) when a new render starts.
4. Blank-frame sanity: featureless render outputs are flagged (`blank_suspect`) and
   warned about instead of being trusted as a successful view of the subject.

Plus the v3 review round of packaging/staging fixes (sections at the bottom):
purpose-preserving binding bakes (preview/full), nested-instance prototype-root
comparison, subset/collection-binding renderer-proofing, USDZ-rooted bake routing,
the two-tier upload estimate, the fd-2 capture lock, the hardened /tmp reaper +
private per-user staging root, unbounded-depth de-instancing, and backend
error-body sanitization.

Plus the v4 benchmark round (final sections): the pre-packaging size estimate
includes resolved external asset dependencies (the layer sum undercounted a real
1,377 MB bundle as 103.8 MB, so the >2× fail-fast fired only AFTER minutes of
packaging); a content-keyed packaged-bundle cache (an identical 1.38 GB bundle was
packaged 4× and its gzip uploaded 2× in 15 minutes) with an LRU bound; uniform,
logged gzip policy (a 350 MB bundle uploaded raw with no gzip line in the trace);
and packaging-warning summaries that separate usdUtils' own package-internal remap
noise (`@0/tex.png@`) from genuinely unresolved references.

Plus the v5 review round (final sections): the bundle-cache key covers external
asset dependencies (editing a texture in place used to serve the stale bundle)
and mixes a content sha into small clean-layer fingerprints (a byte change with
preserved size+mtime_ns collided); a failed dependency walk makes the stage
uncacheable; the cache trusts only the private 0700 staging root (shared-tmp
fallback disables it) and only regular, owned, non-symlink artifacts (a planted
symlink is rejected, never uploaded); and the size estimate no longer returns
(0, 0) for anonymous/in-memory root layers with big asset dependencies.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import pytest


# ── issue 1: instance-external material bindings survive packaging ────────────────


def _build_instanced_scene(root: Path):
    """A prototype referenced 3× (+1 for the proxy-material case), with every binding
    arrangement from the task-03 failure:

    * inst1 — binding authored on the mesh *inside* the prototype source (renderers see
      this one; it must stay instanced and untouched),
    * inst2 — per-instance override on the instance root (usd-cli's `material bind`
      redirect authors exactly this, as strongerThanDescendants),
    * Group/inst3 — ancestor-inherited binding over the instance root,
    * inst4 — instance-root binding targeting a material *inside* its own prototype.
    """
    from pxr import Usd, UsdGeom, UsdShade

    root.mkdir(parents=True, exist_ok=True)
    proto = Usd.Stage.CreateNew(str(root / "proto.usda"))
    UsdGeom.Xform.Define(proto, "/Proto")
    mesh = UsdGeom.Mesh.Define(proto, "/Proto/Geom")
    mat_p = UsdShade.Material.Define(proto, "/Proto/Looks/MatP")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat_p)
    proto.GetRootLayer().defaultPrim = "Proto"
    proto.Save()

    scene = Usd.Stage.CreateNew(str(root / "scene.usda"))
    UsdGeom.Xform.Define(scene, "/World")
    mat_b = UsdShade.Material.Define(scene, "/World/Looks/MatB")
    mat_c = UsdShade.Material.Define(scene, "/World/Looks/MatC")

    def add_instance(path: str):
        prim = scene.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference("./proto.usda")
        prim.SetInstanceable(True)
        return prim

    add_instance("/World/inst1")
    i2 = add_instance("/World/inst2")
    UsdShade.MaterialBindingAPI.Apply(i2).Bind(
        mat_b, UsdShade.Tokens.strongerThanDescendants)
    grp = scene.DefinePrim("/World/Group", "Xform")
    UsdShade.MaterialBindingAPI.Apply(grp).Bind(
        mat_c, UsdShade.Tokens.strongerThanDescendants)
    add_instance("/World/Group/inst3")
    i4 = add_instance("/World/inst4")
    proxy_mat = UsdShade.Material(scene.GetPrimAtPath("/World/inst4/Looks/MatP"))
    UsdShade.MaterialBindingAPI.Apply(i4).Bind(
        proxy_mat, UsdShade.Tokens.strongerThanDescendants)
    scene.Save()
    return scene


def _assert_bindings_renderer_proof(usdz_path: Path):
    """The packaged stage must not depend on instance-external binding resolution:
    every overridden gprim is a real (non-proxy) prim with a *direct* binding."""
    from pxr import Usd, UsdShade

    pkg = Usd.Stage.Open(str(usdz_path))
    assert pkg, f"could not open packaged stage {usdz_path}"

    def gprim(path):
        p = pkg.GetPrimAtPath(path)
        assert p, f"{path} missing from packaged stage"
        return p

    # inst1: prototype-internal binding — stays instanced, still composes MatP
    p1 = gprim("/World/inst1/Geom")
    assert p1.IsInstanceProxy(), "untouched instance must keep its prototype sharing"
    mat, _ = UsdShade.MaterialBindingAPI(p1).ComputeBoundMaterial()
    assert mat and mat.GetPath().name == "MatP"

    # inst2 / inst3 / inst4: baked — de-instanced, direct binding on the gprim itself
    for path, expected in [("/World/inst2/Geom", "/World/Looks/MatB"),
                           ("/World/Group/inst3/Geom", "/World/Looks/MatC"),
                           ("/World/inst4/Geom", "/World/inst4/Looks/MatP")]:
        p = gprim(path)
        assert not p.IsInstanceProxy(), f"{path} must be editable/real in the package"
        direct = UsdShade.MaterialBindingAPI(p).GetDirectBinding()
        assert direct.GetMaterial(), f"{path} lost its direct binding"
        assert str(direct.GetMaterialPath()) == expected
        mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
        assert mat and str(mat.GetPath()) == expected


def _assert_live_stage_untouched(scene):
    """Packaging must only rewrite its throwaway copy, never the caller's stage."""
    for path in ("/World/inst1", "/World/inst2", "/World/Group/inst3", "/World/inst4"):
        assert scene.GetPrimAtPath(path).IsInstance(), f"{path} was de-instanced live"


def test_collect_plan_finds_only_instance_external_bindings(tmp_path):
    from pxr import UsdShade

    from usd_core.render.remote import RemoteRenderBackend

    scene = _build_instanced_scene(tmp_path / "scene")
    plan = RemoteRenderBackend._collect_instance_binding_bakes(scene)
    # inst1's binding lives inside the prototype — renderers resolve it; not in the plan
    assert {g: m for g, m, _ in plan.bakes} == {
        "/World/inst2/Geom": "/World/Looks/MatB",
        "/World/Group/inst3/Geom": "/World/Looks/MatC",
        "/World/inst4/Geom": "/World/inst4/Looks/MatP"}
    # plain all-purpose bindings: every bake carries the all-purpose token
    assert {p for _, _, p in plan.bakes} == {str(UsdShade.Tokens.allPurpose)}
    assert plan.neutralize_rels == []


def test_package_usdz_bakes_instance_external_bindings(tmp_path):
    """Normal packaging path: instance-root overrides and ancestor-inherited bindings
    become direct gprim bindings in the bundle (repro of benchmark task-03)."""
    from usd_core.render.remote import RemoteRenderBackend

    scene = _build_instanced_scene(tmp_path / "scene")
    work = tmp_path / "work"
    work.mkdir()
    usdz = RemoteRenderBackend._package_usdz(scene, work)
    _assert_bindings_renderer_proof(usdz)
    _assert_live_stage_untouched(scene)


def test_package_flattened_bakes_instance_external_bindings(tmp_path):
    """The flatten fallback preserves instancing, so it needs the same bake."""
    from usd_core.render.remote import RemoteRenderBackend

    scene = _build_instanced_scene(tmp_path / "scene")
    work = tmp_path / "work"
    work.mkdir()
    usdz = work / "scene_bundle.usdz"
    RemoteRenderBackend._package_flattened(scene, work, usdz)
    _assert_bindings_renderer_proof(usdz)
    _assert_live_stage_untouched(scene)


def test_forced_flatten_fallback_still_bakes_bindings(tmp_path):
    """A broken asset path pushes packaging onto the fallback organically; the baked
    bindings must survive that route end-to-end too."""
    from pxr import Sdf, UsdShade
    from usd_core.render.remote import RemoteRenderBackend

    scene = _build_instanced_scene(tmp_path / "scene")
    sh = UsdShade.Shader.Define(scene, "/World/Looks/MatB/Tex")
    sh.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("/nonexistent/textures/tex.png"))
    scene.Save()

    work = tmp_path / "work"
    work.mkdir()
    usdz = RemoteRenderBackend._package_usdz(scene, work)
    _assert_bindings_renderer_proof(usdz)
    _assert_live_stage_untouched(scene)


def test_scene_without_instances_is_left_alone(tmp_path):
    from pxr import Usd, UsdGeom, UsdShade
    from usd_core.render.remote import RemoteRenderBackend

    scene = Usd.Stage.CreateNew(str(tmp_path / "plain.usda"))
    mesh = UsdGeom.Mesh.Define(scene, "/World/Mesh")
    mat = UsdShade.Material.Define(scene, "/World/Looks/Mat")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    scene.Save()
    plan = RemoteRenderBackend._collect_instance_binding_bakes(scene)
    assert not plan and plan.bakes == [] and plan.neutralize_rels == []


# ── issue 2: bounded upload default ────────────────────────────────────────────────


def test_config_default_upload_cap_is_512():
    from usd_core.config import DEFAULTS

    assert DEFAULTS["render"]["remote_max_upload_mb"] == 512


def test_default_render_profile_matches_workbench_inspection():
    from usd_core.config import DEFAULTS
    from usd_core.render.ovrtx import (
        MAX_OVRTX_SENSOR_UPDATES,
        _DAEMON_SCRIPT,
        OvRTXRenderBackend,
    )

    assert DEFAULTS["render"]["mode"] == "quality"
    assert DEFAULTS["render"]["ovrtx_num_sensor_updates"] == 64

    backend = OvRTXRenderBackend()
    assert backend._resolve_render_mode("quality") == "rt2"
    assert backend._resolve_updates("quality") == 64
    assert backend._resolve_render_mode("fast") == "rt1"
    assert 'req.get("num_sensor_updates", 64)' in _DAEMON_SCRIPT
    assert "MAX_SENSOR_UPDATES = 1024" in _DAEMON_SCRIPT
    with pytest.raises(ValueError, match="between 1 and"):
        OvRTXRenderBackend(num_sensor_updates=0)
    with pytest.raises(ValueError, match="between 1 and"):
        OvRTXRenderBackend(num_sensor_updates=MAX_OVRTX_SENSOR_UPDATES + 1)
    with pytest.raises(ValueError, match="must be an integer"):
        OvRTXRenderBackend(num_sensor_updates=True)


def test_invalid_default_hdri_fails_before_render_temp_files(tmp_path, monkeypatch):
    from pxr import Usd, UsdGeom
    from usd_core.render.ovrtx import OvRTXRenderBackend

    scene = Usd.Stage.CreateNew(str(tmp_path / "scene.usda"))
    UsdGeom.Xform.Define(scene, "/World")
    UsdGeom.Camera.Define(scene, "/World/Camera")
    scene.Save()
    output_dir = tmp_path / "renders"
    missing_hdri = tmp_path / "missing.exr"
    monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", str(missing_hdri))

    with pytest.raises(RuntimeError, match="default HDRI does not exist"):
        OvRTXRenderBackend().render(
            scene,
            ["/World/Camera"],
            64,
            64,
            output_dir,
        )

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_default_light_rig_matches_workbench_studio_hdri(monkeypatch):
    """Lightless scenes use the original Content Workbench studio rig exactly."""
    from usd_core.render.ovrtx import (
        _build_default_lights_usda,
        _default_hdri_intensity,
        _default_hdri_path,
    )

    monkeypatch.delenv("OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)
    monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)
    monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)

    hdri = _default_hdri_path()
    assert hdri.name == "studio.exr"
    assert hdri.is_file()
    assert hashlib.sha256(hdri.read_bytes()).hexdigest() == (
        "f0379ca1056f578b0081fc1d80b702d61e7a79d5c8000a030d50e9ada1cee539"
    )
    assert _default_hdri_intensity() == 600.0

    lights = _build_default_lights_usda(_default_hdri_intensity())
    assert 'def "OvRTXDefaultLights"' in lights
    assert 'def DomeLight "DomeLight"' in lights
    assert "float inputs:intensity = 600.0" in lights
    assert 'token inputs:texture:format = "latlong"' in lights
    assert f"asset inputs:texture:file = @{hdri}@" in lights
    assert "custom bool visibleInPrimaryRay = 0" in lights
    assert "DistantLight" not in lights
    assert "KeyLight" not in lights
    assert "FillLight" not in lights


def test_default_light_rig_honors_intensity_and_hdri_overrides(tmp_path, monkeypatch):
    from usd_core.render.ovrtx import (
        _build_default_lights_usda,
        _default_hdri_intensity,
        _default_hdri_path,
    )

    custom_hdri = tmp_path / "custom.exr"
    custom_hdri.write_bytes(b"custom")
    monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", str(custom_hdri))
    monkeypatch.setenv("OVRTX_DEFAULT_HDRI_INTENSITY", "725")

    assert _default_hdri_path() == custom_hdri.resolve()
    assert _default_hdri_intensity() == 725.0
    lights = _build_default_lights_usda(_default_hdri_intensity())
    assert f"asset inputs:texture:file = @{custom_hdri.resolve()}@" in lights
    assert "float inputs:intensity = 725.0" in lights


def test_custom_hdri_keeps_legacy_one_intensity_fallback(tmp_path, monkeypatch):
    from usd_core.render.ovrtx import _default_hdri_intensity

    custom_hdri = tmp_path / "custom.exr"
    custom_hdri.write_bytes(b"custom")
    monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", str(custom_hdri))
    monkeypatch.delenv("OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)
    monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

    assert _default_hdri_intensity() == 1.0


def test_malformed_hdri_intensity_warns_and_uses_effective_fallback(
    tmp_path, monkeypatch, caplog
):
    import logging

    from usd_core.render.ovrtx import _default_hdri_intensity

    custom_hdri = tmp_path / "custom.exr"
    custom_hdri.write_bytes(b"custom")
    monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", str(custom_hdri))
    monkeypatch.setenv("OVRTX_DEFAULT_HDRI_INTENSITY", "not-a-number")

    with caplog.at_level(logging.WARNING, logger="usd_core.render.ovrtx"):
        assert _default_hdri_intensity() == 1.0
    assert "Invalid OVRTX default HDRI intensity" in caplog.text


def test_factory_passes_the_512_default_to_the_backend(caplog):
    import logging

    from usd_core.config import Config
    from usd_core.render.factory import make_backend

    cfg = Config()
    cfg.render["renderer"] = "remote"
    cfg.render["remote_url"] = "http://gpu:8000"
    backend = make_backend(cfg)
    assert backend._max_upload_mb == 512

    # the estimate fail-fast is armed at the default (it was inert at 0):
    # over 2× the cap it is hopeless even after gzip — hard fail before packaging
    with pytest.raises(RuntimeError, match="remote_max_upload_mb"):
        backend._check_estimate(1100 * 1024 * 1024, 3, "scene")
    backend._check_estimate(100 * 1024 * 1024, 3, "scene")  # under the cap: no raise
    # between 1× and 2× the cap: raw layer bytes overstate the gzip wire bytes, so a
    # compressible scene must not be rejected before packaging — warn and proceed to
    # the exact post-gzip check instead
    with caplog.at_level(logging.WARNING, logger="usd_core.render.remote"):
        backend._check_estimate(600 * 1024 * 1024, 3, "scene")
    warned = [r for r in caplog.records if "proceeding anyway" in r.getMessage()]
    assert len(warned) == 1 and "600.0 MB" in warned[0].getMessage()


def test_zero_still_means_unlimited():
    from usd_core.config import Config
    from usd_core.render.factory import make_backend
    from usd_core.render.remote import RemoteRenderBackend

    cfg = Config()
    cfg.render["renderer"] = "remote"
    cfg.render["remote_url"] = "http://gpu:8000"
    cfg.render["remote_max_upload_mb"] = 0
    backend = make_backend(cfg)
    assert backend._max_upload_mb == 0
    backend._check_estimate(10**13, 5, "scene")  # unlimited: any estimate passes

    # direct construction keeps 0-as-unlimited semantics too
    RemoteRenderBackend("http://gpu:8000", max_upload_mb=0)._check_estimate(10**13, 5, "s")


# ── issue 3: staging dirs are cleaned on failure and reaped when orphaned ──────────


def test_execute_cleans_staging_dir_when_packaging_fails(monkeypatch):
    from usd_core.render.remote import RemoteRenderBackend

    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    seen: dict = {}

    def boom(cls, stage, work_dir):
        seen["work_dir"] = Path(work_dir)
        assert seen["work_dir"].is_dir()
        raise RuntimeError("packaging exploded mid-flight")

    monkeypatch.setattr(RemoteRenderBackend, "_package_usdz", classmethod(boom))

    class StageStub:  # _scene_stem/_estimate degrade gracefully on non-stage objects
        pass

    with pytest.raises(RuntimeError, match="packaging exploded"):
        backend._execute(StageStub(), {"cameras": ["/c"]})
    assert not seen["work_dir"].exists(), "staging dir leaked after a packaging failure"


def test_reaper_removes_stale_and_dead_owner_dirs(tmp_path):
    import subprocess

    from usd_core.render.remote import _reap_stale_staging

    # a dir older than 24h (any owner)
    old = tmp_path / f"ov_remote_render_{os.getpid()}9_old"
    old.mkdir()
    (old / "scene_bundle.usdz").write_bytes(b"x")
    stale = time.time() - 25 * 3600
    os.utime(old, (stale, stale))

    # a fresh dir owned by a PID that no longer exists
    proc = subprocess.Popen(["true"])
    proc.wait()
    dead = tmp_path / f"ov_remote_render_{proc.pid}_dead"
    dead.mkdir()

    # a fresh dir owned by a live process (this one) — must survive
    ours = tmp_path / f"ov_remote_render_{os.getpid()}_live"
    ours.mkdir()

    # a fresh legacy dir without a parseable PID — age-gated only, must survive
    legacy = tmp_path / "ov_remote_render_legacy"
    legacy.mkdir()

    # unrelated entries are never touched
    other = tmp_path / "somebody_elses_dir"
    other.mkdir()

    _reap_stale_staging(base=tmp_path)

    assert not old.exists(), "24h-old staging dir survived the reaper"
    assert not dead.exists(), "dead-owner staging dir survived the reaper"
    assert ours.exists(), "reaper must not remove an in-flight render's staging dir"
    assert legacy.exists(), "fresh legacy-named dir must be age-gated, not removed"
    assert other.exists()


def test_execute_reaps_on_start(monkeypatch, tmp_path):
    """A new render sweeps leftovers before doing its own staging."""
    import usd_core.render.remote as remote_mod
    from usd_core.render.remote import RemoteRenderBackend

    import subprocess
    _child = subprocess.Popen(["sleep", "0"]); _child.wait()
    leftover = tmp_path / f"ov_remote_render_{_child.pid}_leftover"
    leftover.mkdir()
    stale = time.time() - 25 * 3600
    os.utime(leftover, (stale, stale))
    monkeypatch.setattr(remote_mod.tempfile, "gettempdir", lambda: str(tmp_path))

    def boom(cls, stage, work_dir):
        raise RuntimeError("stop after the reap")

    monkeypatch.setattr(RemoteRenderBackend, "_package_usdz", classmethod(boom))
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    with pytest.raises(RuntimeError, match="stop after the reap"):
        backend._execute(object(), {"cameras": ["/c"]})
    assert not leftover.exists()


# ── issue 4: blank-frame sanity heuristic ──────────────────────────────────────────


def _write_png(path: Path, mode: str) -> Path:
    import numpy as np
    from PIL import Image

    if mode == "uniform":
        arr = np.full((64, 64, 3), 200, dtype=np.uint8)
    elif mode == "faint-gradient":  # featureless: stddev well under 2
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, :] = 180 + (np.linspace(0, 3, 64).astype(np.uint8))[None, :, None]
    else:  # "textured": a real subject — high-contrast structure
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)
    return path


def test_blank_suspect_heuristic(tmp_path):
    from usd_core.imaging import blank_suspects, is_blank_suspect

    uniform = _write_png(tmp_path / "uniform.png", "uniform")
    faint = _write_png(tmp_path / "faint.png", "faint-gradient")
    textured = _write_png(tmp_path / "textured.png", "textured")

    assert is_blank_suspect(str(uniform))
    assert is_blank_suspect(str(faint))
    assert not is_blank_suspect(str(textured))
    assert blank_suspects([str(uniform), str(textured), str(faint)]) == [
        str(uniform), str(faint)]

    # advisory: unreadable input reads as "not blank", never an exception
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    assert not is_blank_suspect(str(broken))
    assert not is_blank_suspect(str(tmp_path / "missing.png"))


def test_flag_blank_suspects_tags_results_and_warns(tmp_path, caplog):
    import logging

    from usd_core.render.base import RenderResult
    from usd_core.render.remote import _flag_blank_suspects

    blank = _write_png(tmp_path / "blank.png", "uniform")
    good = _write_png(tmp_path / "good.png", "textured")
    results = [
        RenderResult(path=str(blank), camera="/c1", width=64, height=64, backend="t"),
        RenderResult(path=str(good), camera="/c2", width=64, height=64, backend="t"),
    ]
    with caplog.at_level(logging.WARNING, logger="usd_core.render.remote"):
        suspects = _flag_blank_suspects(results)
    assert suspects == [str(blank)]
    assert results[0].blank_suspect is True
    assert results[1].blank_suspect is False
    warned = [r for r in caplog.records if "blank_suspect" in r.getMessage()]
    assert len(warned) == 1 and str(blank) in warned[0].getMessage()


def test_remote_render_flags_blank_output(tmp_path, monkeypatch, caplog):
    """End-to-end through RemoteRenderBackend.render: a featureless returned image is
    flagged on the results the caller receives."""
    import base64
    import io
    import logging

    import numpy as np
    from PIL import Image

    from usd_core.render.remote import RemoteRenderBackend

    buf = io.BytesIO()
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8), "RGB").save(
        buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    backend = RemoteRenderBackend("http://gpu:8000")
    monkeypatch.setattr(
        backend, "_execute",
        lambda stage, params: ([{"camera": "/World/cam", "image_base64": img_b64}], 0.1))
    with caplog.at_level(logging.WARNING, logger="usd_core.render.remote"):
        results = backend.render(None, ["/World/cam"], 32, 32, tmp_path / "out")
    assert len(results) == 1
    assert results[0].blank_suspect is True
    assert any("blank" in r.getMessage() for r in caplog.records)


# ── v3 review round ────────────────────────────────────────────────────────────────
# ── purpose-specific bindings (preview/full) are collected and baked ───────────────


def test_preview_purpose_binding_is_baked_with_its_purpose(tmp_path):
    """A preview-purpose binding override on an instance root used to produce an
    EMPTY bake plan (only allPurpose was queried) and render prototype materials."""
    from pxr import Usd, UsdGeom, UsdShade

    from usd_core.render.remote import RemoteRenderBackend

    root = tmp_path / "scene"
    root.mkdir()
    proto = Usd.Stage.CreateNew(str(root / "proto.usda"))
    UsdGeom.Xform.Define(proto, "/Proto")
    mesh = UsdGeom.Mesh.Define(proto, "/Proto/Geom")
    mat_p = UsdShade.Material.Define(proto, "/Proto/Looks/MatP")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat_p)
    proto.GetRootLayer().defaultPrim = "Proto"
    proto.Save()

    scene = Usd.Stage.CreateNew(str(root / "scene.usda"))
    UsdGeom.Xform.Define(scene, "/World")
    mat_prev = UsdShade.Material.Define(scene, "/World/Looks/MatPrev")
    inst = scene.DefinePrim("/World/inst", "Xform")
    inst.GetReferences().AddReference("./proto.usda")
    inst.SetInstanceable(True)
    UsdShade.MaterialBindingAPI.Apply(inst).Bind(
        mat_prev, UsdShade.Tokens.strongerThanDescendants,
        UsdShade.Tokens.preview)
    scene.Save()

    plan = RemoteRenderBackend._collect_instance_binding_bakes(scene)
    assert ("/World/inst/Geom", "/World/Looks/MatPrev",
            str(UsdShade.Tokens.preview)) in plan.bakes
    # the all-purpose binding still composes from inside the prototype — not planned
    assert all(p != str(UsdShade.Tokens.allPurpose) for _, _, p in plan.bakes)

    work = tmp_path / "work"
    work.mkdir()
    usdz = RemoteRenderBackend._package_usdz(scene, work)
    pkg = Usd.Stage.Open(str(usdz))
    g = pkg.GetPrimAtPath("/World/inst/Geom")
    assert g and not g.IsInstanceProxy()
    api = UsdShade.MaterialBindingAPI(g)
    direct = api.GetDirectBinding(materialPurpose=UsdShade.Tokens.preview)
    assert str(direct.GetMaterialPath()) == "/World/Looks/MatPrev"
    mat, _ = api.ComputeBoundMaterial(materialPurpose=UsdShade.Tokens.preview)
    assert mat and str(mat.GetPath()) == "/World/Looks/MatPrev"
    # the purpose was preserved: the all-purpose resolution still sees the prototype's
    mat_all, _ = api.ComputeBoundMaterial()
    assert mat_all and mat_all.GetPath().name == "MatP"


# ── nested instancing: prototype roots are compared, not proxy-ness ────────────────


def _build_nested_instance_scene(root: Path):
    """A binding authored on a *nested* instance root: its owner is an outer-prototype
    proxy (the old `IsInstanceProxy()` test skipped it) yet lies outside the inner
    mesh's prototype, so a prototype-resolving renderer drops it."""
    from pxr import Usd, UsdGeom, UsdShade

    root.mkdir(parents=True, exist_ok=True)
    inner = Usd.Stage.CreateNew(str(root / "inner.usda"))
    UsdGeom.Xform.Define(inner, "/Inner")
    UsdGeom.Mesh.Define(inner, "/Inner/Geom")
    inner.GetRootLayer().defaultPrim = "Inner"
    inner.Save()

    outer = Usd.Stage.CreateNew(str(root / "outer.usda"))
    UsdGeom.Xform.Define(outer, "/Outer")
    mat_o = UsdShade.Material.Define(outer, "/Outer/Looks/MatO")
    nested = outer.DefinePrim("/Outer/inner", "Xform")
    nested.GetReferences().AddReference("./inner.usda")
    nested.SetInstanceable(True)
    UsdShade.MaterialBindingAPI.Apply(nested).Bind(
        mat_o, UsdShade.Tokens.strongerThanDescendants)
    outer.GetRootLayer().defaultPrim = "Outer"
    outer.Save()

    scene = Usd.Stage.CreateNew(str(root / "scene.usda"))
    UsdGeom.Xform.Define(scene, "/World")
    o = scene.DefinePrim("/World/outer", "Xform")
    o.GetReferences().AddReference("./outer.usda")
    o.SetInstanceable(True)
    scene.Save()
    return scene


def test_binding_on_nested_instance_root_is_baked(tmp_path):
    from pxr import Usd, UsdShade

    from usd_core.render.remote import RemoteRenderBackend

    scene = _build_nested_instance_scene(tmp_path / "scene")
    g = scene.GetPrimAtPath("/World/outer/inner/Geom")
    _, rel = UsdShade.MaterialBindingAPI(g).ComputeBoundMaterial()
    assert rel.GetPrim().IsInstanceProxy()  # the exact shape the old check skipped

    plan = RemoteRenderBackend._collect_instance_binding_bakes(scene)
    assert {(g, m) for g, m, _ in plan.bakes} == {
        ("/World/outer/inner/Geom", "/World/outer/Looks/MatO")}

    work = tmp_path / "work"
    work.mkdir()
    usdz = RemoteRenderBackend._package_usdz(scene, work)
    pkg = Usd.Stage.Open(str(usdz))
    p = pkg.GetPrimAtPath("/World/outer/inner/Geom")
    assert p and not p.IsInstanceProxy()
    direct = UsdShade.MaterialBindingAPI(p).GetDirectBinding()
    assert str(direct.GetMaterialPath()) == "/World/outer/Looks/MatO"
    # the live stage stays instanced
    assert scene.GetPrimAtPath("/World/outer").IsInstance()


def test_deinstancing_handles_deeper_than_eight_levels(tmp_path):
    """De-instancing used to stop after a fixed 8 levels, leaving valid deeper
    nesting unbaked; it now loops until the prim is real, guarded by progress."""
    from pxr import Usd, UsdGeom, UsdShade

    from usd_core.render.remote import RemoteRenderBackend

    root = tmp_path / "deep"
    root.mkdir()
    depth = 10
    base = Usd.Stage.CreateNew(str(root / "level0.usda"))
    UsdGeom.Xform.Define(base, "/L")
    UsdGeom.Mesh.Define(base, "/L/Geom")
    base.GetRootLayer().defaultPrim = "L"
    base.Save()
    for k in range(1, depth):
        st = Usd.Stage.CreateNew(str(root / f"level{k}.usda"))
        UsdGeom.Xform.Define(st, "/L")
        nested = st.DefinePrim("/L/inst", "Xform")
        nested.GetReferences().AddReference(f"./level{k - 1}.usda")
        nested.SetInstanceable(True)
        st.GetRootLayer().defaultPrim = "L"
        st.Save()
    scene = Usd.Stage.CreateNew(str(root / "scene.usda"))
    UsdGeom.Xform.Define(scene, "/World")
    mat = UsdShade.Material.Define(scene, "/World/Looks/MatDeep")
    top = scene.DefinePrim("/World/top", "Xform")
    top.GetReferences().AddReference(f"./level{depth - 1}.usda")
    top.SetInstanceable(True)
    UsdShade.MaterialBindingAPI.Apply(top).Bind(
        mat, UsdShade.Tokens.strongerThanDescendants)
    scene.Save()

    geom_path = "/World/top" + "/inst" * (depth - 1) + "/Geom"
    assert scene.GetPrimAtPath(geom_path).IsInstanceProxy()
    plan = RemoteRenderBackend._collect_instance_binding_bakes(scene)
    assert {g for g, _, _ in plan.bakes} == {geom_path}

    work = tmp_path / "work"
    work.mkdir()
    usdz = RemoteRenderBackend._package_usdz(scene, work)
    pkg = Usd.Stage.Open(str(usdz))
    p = pkg.GetPrimAtPath(geom_path)
    assert p and not p.IsInstanceProxy(), "deep nesting was left unbaked"
    direct = UsdShade.MaterialBindingAPI(p).GetDirectBinding()
    assert str(direct.GetMaterialPath()) == "/World/Looks/MatDeep"


# ── GeomSubsets and prototype-local collection bindings ─────────────────────────────


def test_subset_with_parent_equal_material_is_still_baked(tmp_path):
    """A subset whose composed material equals its parent's was skipped — but its
    prototype-local subset binding resurfaces once the external binding is dropped
    by a prototype-resolving renderer. Every affected subset is baked now."""
    from pxr import Usd, UsdGeom, UsdShade

    from usd_core.render.remote import RemoteRenderBackend

    root = tmp_path / "scene"
    root.mkdir()
    proto = Usd.Stage.CreateNew(str(root / "proto.usda"))
    UsdGeom.Xform.Define(proto, "/Proto")
    UsdGeom.Mesh.Define(proto, "/Proto/Geom")
    sub = UsdGeom.Subset.Define(proto, "/Proto/Geom/Sub")
    sub.CreateElementTypeAttr(UsdGeom.Tokens.face)
    sub.CreateIndicesAttr([0])
    sub.CreateFamilyNameAttr("materialBind")
    mat_sub = UsdShade.Material.Define(proto, "/Proto/Looks/MatSub")
    UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mat_sub)
    proto.GetRootLayer().defaultPrim = "Proto"
    proto.Save()

    scene = Usd.Stage.CreateNew(str(root / "scene.usda"))
    UsdGeom.Xform.Define(scene, "/World")
    mat_b = UsdShade.Material.Define(scene, "/World/Looks/MatB")
    inst = scene.DefinePrim("/World/inst", "Xform")
    inst.GetReferences().AddReference("./proto.usda")
    inst.SetInstanceable(True)
    UsdShade.MaterialBindingAPI.Apply(inst).Bind(
        mat_b, UsdShade.Tokens.strongerThanDescendants)
    scene.Save()

    # sanity: the subset's composed material EQUALS its parent's (the skipped shape)
    smat, _ = UsdShade.MaterialBindingAPI(
        scene.GetPrimAtPath("/World/inst/Geom/Sub")).ComputeBoundMaterial()
    assert smat and str(smat.GetPath()) == "/World/Looks/MatB"

    plan = RemoteRenderBackend._collect_instance_binding_bakes(scene)
    assert {g: m for g, m, _ in plan.bakes} == {
        "/World/inst/Geom": "/World/Looks/MatB",
        "/World/inst/Geom/Sub": "/World/Looks/MatB"}  # the subset is baked too

    work = tmp_path / "work"
    work.mkdir()
    usdz = RemoteRenderBackend._package_usdz(scene, work)
    pkg = Usd.Stage.Open(str(usdz))
    for path in ("/World/inst/Geom", "/World/inst/Geom/Sub"):
        p = pkg.GetPrimAtPath(path)
        assert p and not p.IsInstanceProxy()
        direct = UsdShade.MaterialBindingAPI(p).GetDirectBinding()
        assert str(direct.GetMaterialPath()) == "/World/Looks/MatB", path
        mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
        assert mat and str(mat.GetPath()) == "/World/Looks/MatB", path


def _build_collection_scene(root: Path, collection_strength):
    """Prototype with GeomA/GeomB under /P/Geoms; a collection binding on /P/Geoms
    (inside the prototype) binds MatColl to GeomB; the scene instance root binds
    MatB externally."""
    from pxr import Usd, UsdGeom, UsdShade

    root.mkdir(parents=True, exist_ok=True)
    proto = Usd.Stage.CreateNew(str(root / "proto.usda"))
    UsdGeom.Xform.Define(proto, "/P")
    geoms = UsdGeom.Xform.Define(proto, "/P/Geoms").GetPrim()
    UsdGeom.Mesh.Define(proto, "/P/Geoms/GeomA")
    UsdGeom.Mesh.Define(proto, "/P/Geoms/GeomB")
    mat_coll = UsdShade.Material.Define(proto, "/P/Looks/MatColl")
    coll = Usd.CollectionAPI.Apply(geoms, "grp")
    coll.CreateIncludesRel().AddTarget("/P/Geoms/GeomB")
    UsdShade.MaterialBindingAPI.Apply(geoms).Bind(
        coll, mat_coll, "grp", collection_strength)
    proto.GetRootLayer().defaultPrim = "P"
    proto.Save()

    scene = Usd.Stage.CreateNew(str(root / "scene.usda"))
    UsdGeom.Xform.Define(scene, "/World")
    mat_b = UsdShade.Material.Define(scene, "/World/Looks/MatB")
    inst = scene.DefinePrim("/World/inst", "Xform")
    inst.GetReferences().AddReference("./proto.usda")
    inst.SetInstanceable(True)
    UsdShade.MaterialBindingAPI.Apply(inst).Bind(
        mat_b, UsdShade.Tokens.strongerThanDescendants)
    scene.Save()
    return scene


def test_prototype_local_collection_binding_is_neutralized(tmp_path):
    """A strongerThanDescendants collection binding inside the prototype outranks a
    baked direct binding once its subtree is de-instanced — the packaging copy must
    block it or GeomB silently renders MatColl instead of the composed MatB."""
    from pxr import Usd, UsdShade

    from usd_core.render.remote import RemoteRenderBackend

    scene = _build_collection_scene(tmp_path / "scene",
                                    UsdShade.Tokens.strongerThanDescendants)
    # composed winner is the external MatB for BOTH meshes (instance root is higher)
    for path in ("/World/inst/Geoms/GeomA", "/World/inst/Geoms/GeomB"):
        mat, _ = UsdShade.MaterialBindingAPI(
            scene.GetPrimAtPath(path)).ComputeBoundMaterial()
        assert mat and str(mat.GetPath()) == "/World/Looks/MatB"

    plan = RemoteRenderBackend._collect_instance_binding_bakes(scene)
    assert "/World/inst/Geoms.material:binding:collection:grp" in plan.neutralize_rels

    work = tmp_path / "work"
    work.mkdir()
    usdz = RemoteRenderBackend._package_usdz(scene, work)
    pkg = Usd.Stage.Open(str(usdz))
    for path in ("/World/inst/Geoms/GeomA", "/World/inst/Geoms/GeomB"):
        mat, _ = UsdShade.MaterialBindingAPI(
            pkg.GetPrimAtPath(path)).ComputeBoundMaterial()
        assert mat and str(mat.GetPath()) == "/World/Looks/MatB", (
            f"{path}: the prototype-local collection binding resurfaced post-bake")
    rel = pkg.GetPrimAtPath("/World/inst/Geoms").GetRelationship(
        "material:binding:collection:grp")
    assert rel and rel.GetTargets() == []  # blocked in the throwaway copy
    # the live stage keeps its collection binding untouched
    live = scene.GetPrimAtPath("/World/inst/Geoms")
    assert live.IsInstanceProxy()  # still instanced, nothing de-instanced live


def test_prim_bound_by_a_neutralized_collection_is_baked_too(tmp_path):
    """Blocking a collection rel must not strip the material of a prim for which that
    rel WAS the composed winner — the fixpoint bakes it with its composed material."""
    from pxr import Usd, UsdGeom, UsdShade

    from usd_core.render.remote import RemoteRenderBackend

    root = tmp_path / "scene"
    root.mkdir()
    proto = Usd.Stage.CreateNew(str(root / "proto.usda"))
    UsdGeom.Xform.Define(proto, "/P")
    geoms = UsdGeom.Xform.Define(proto, "/P/Geoms").GetPrim()
    UsdGeom.Mesh.Define(proto, "/P/Geoms/GeomA")
    UsdGeom.Mesh.Define(proto, "/P/Geoms/GeomB")
    mat_coll = UsdShade.Material.Define(proto, "/P/Looks/MatColl")
    coll = Usd.CollectionAPI.Apply(geoms, "grp")
    coll.CreateIncludesRel().AddTarget("/P/Geoms/GeomB")
    UsdShade.MaterialBindingAPI.Apply(geoms).Bind(coll, mat_coll, "grp")
    proto.GetRootLayer().defaultPrim = "P"
    proto.Save()

    scene = Usd.Stage.CreateNew(str(root / "scene.usda"))
    UsdGeom.Xform.Define(scene, "/World")
    mat_b = UsdShade.Material.Define(scene, "/World/Looks/MatB")
    inst = scene.DefinePrim("/World/inst", "Xform")
    inst.GetReferences().AddReference("./proto.usda")
    inst.SetInstanceable(True)
    UsdShade.MaterialBindingAPI.Apply(inst).Bind(mat_b)  # weaker than descendants
    scene.Save()

    # composed: GeomA gets the external MatB, GeomB keeps the collection's MatColl
    for path, want in (("/World/inst/Geoms/GeomA", "/World/Looks/MatB"),
                       ("/World/inst/Geoms/GeomB", "/World/inst/Looks/MatColl")):
        mat, _ = UsdShade.MaterialBindingAPI(
            scene.GetPrimAtPath(path)).ComputeBoundMaterial()
        assert mat and str(mat.GetPath()) == want

    plan = RemoteRenderBackend._collect_instance_binding_bakes(scene)
    assert {g: m for g, m, _ in plan.bakes} == {
        "/World/inst/Geoms/GeomA": "/World/Looks/MatB",
        "/World/inst/Geoms/GeomB": "/World/inst/Looks/MatColl"}  # fixpoint added B

    work = tmp_path / "work"
    work.mkdir()
    usdz = RemoteRenderBackend._package_usdz(scene, work)
    pkg = Usd.Stage.Open(str(usdz))
    for path, want in (("/World/inst/Geoms/GeomA", "/World/Looks/MatB"),
                       ("/World/inst/Geoms/GeomB", "/World/inst/Looks/MatColl")):
        p = pkg.GetPrimAtPath(path)
        mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
        assert mat and str(mat.GetPath()) == want, path
        direct = UsdShade.MaterialBindingAPI(p).GetDirectBinding()
        assert str(direct.GetMaterialPath()) == want, path


# ── USDZ-rooted stages with a nonempty bake plan ────────────────────────────────────


def _usdz_rooted_instanced_stage(tmp_path):
    from pxr import Usd, UsdUtils

    scene = _build_instanced_scene(tmp_path / "scene")
    del scene  # release the stage; package from the saved layers
    usdz_src = tmp_path / "scene" / "scene.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(tmp_path / "scene" / "scene.usda"),
                                         str(usdz_src))
    stage = Usd.Stage.Open(str(usdz_src))
    assert stage
    return stage


def test_usdz_rooted_stage_with_bake_plan_routes_through_flattened_copy(tmp_path):
    """A USDZ root layer is immutable: packaging used to warn and knowingly render
    wrong per-instance materials. It must bake via the writable flattened copy."""
    from usd_core.render.remote import RemoteRenderBackend

    stage = _usdz_rooted_instanced_stage(tmp_path)
    plan = RemoteRenderBackend._collect_instance_binding_bakes(stage)
    assert plan, "fixture must reproduce a nonempty plan on a USDZ-rooted stage"

    work = tmp_path / "work"
    work.mkdir()
    usdz = RemoteRenderBackend._package_usdz(stage, work)
    _assert_bindings_renderer_proof(usdz)


def test_usdz_rooted_stage_fails_render_when_the_bake_cannot_happen(tmp_path, monkeypatch):
    """When even the flattened-copy bake fails, the render must FAIL with a clear
    error instead of returning a knowingly-wrong bundle."""
    from usd_core.render.remote import RemoteRenderBackend

    stage = _usdz_rooted_instanced_stage(tmp_path)

    def boom(cls, stage, work_dir, usdz_path):
        raise RuntimeError("flatten exploded")

    monkeypatch.setattr(RemoteRenderBackend, "_package_flattened", classmethod(boom))
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(RuntimeError, match="refusing to render"):
        RemoteRenderBackend._package_usdz(stage, work)


# ── the fd-2 capture window is serialized process-wide ──────────────────────────────


def test_stderr_capture_holds_a_process_wide_lock():
    """Concurrent sessions packaging in parallel used to swap/restore fd 2 out of
    order, leaving daemon stderr pointed at an unlinked temp file. The whole capture
    window now holds a module-level lock."""
    from usd_core.render.remote import _STDERR_SWAP_LOCK, _StderrCapture

    entered = threading.Event()
    release = threading.Event()
    state: dict = {}

    def worker():
        with _StderrCapture() as cap:
            os.write(2, b"thread-A line\n")
            entered.set()
            release.wait(timeout=10)
        state["text"] = cap.text

    t = threading.Thread(target=worker)
    t.start()
    try:
        assert entered.wait(timeout=10)
        # the capture window holds the lock — a second capture would serialize here
        assert _STDERR_SWAP_LOCK.locked()
    finally:
        release.set()
        t.join(timeout=10)
    assert not _STDERR_SWAP_LOCK.locked()  # released on exit, even on the no-op path
    assert "thread-A line" in state["text"]


def test_concurrent_captures_leave_fd2_intact():
    from usd_core.render.remote import _StderrCapture

    before = os.fstat(2)

    def cycle():
        for _ in range(5):
            with _StderrCapture():
                os.write(2, b"x")

    threads = [threading.Thread(target=cycle) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    after = os.fstat(2)
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino), (
        "fd 2 was restored out of order and no longer points at the real stderr")


# ── hardened /tmp reaper + private per-user staging root ───────────────────────────


def test_reaper_survives_hostile_names_and_never_reaps_live_owners(tmp_path):
    from usd_core.render.remote import _reap_stale_staging

    # any local user can create this in a shared /tmp: os.kill(10**30, 0) raises
    # OverflowError, which used to break every subsequent remote render
    huge = tmp_path / f"ov_remote_render_{10**30}_x"
    huge.mkdir()
    zero = tmp_path / "ov_remote_render_0_x"  # pid 0 would signal our process group
    zero.mkdir()

    # a stale dir with a verifiably LIVE owner (our parent) must never be age-deleted
    live_stale = tmp_path / f"ov_remote_render_{os.getppid()}_live"
    live_stale.mkdir()
    stale = time.time() - 25 * 3600
    os.utime(live_stale, (stale, stale))

    _reap_stale_staging(base=tmp_path)  # must not raise

    assert not huge.exists()  # unrepresentable owner: cannot be alive — reaped
    assert not zero.exists()
    assert live_stale.exists(), "age-deleted a dir whose owner is verifiably alive"


def test_staging_root_is_private_per_user(tmp_path, monkeypatch):
    import stat

    import usd_core.render.remote as remote_mod

    monkeypatch.setattr(remote_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    root = remote_mod._staging_root()
    uid = os.getuid() if hasattr(os, "getuid") else None
    assert root == tmp_path / (f"dsc3-{uid}" if uid is not None else "dsc3")
    assert stat.S_IMODE(root.stat().st_mode) == 0o700

    # a squatted name (symlink) must never be staged into — fall back to shared tmp
    squat_base = tmp_path / "squat"
    squat_base.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (squat_base / root.name).symlink_to(elsewhere)
    monkeypatch.setattr(remote_mod.tempfile, "gettempdir", lambda: str(squat_base))
    assert remote_mod._staging_root() == squat_base


def test_execute_stages_inside_the_private_root_and_reaps_legacy(tmp_path, monkeypatch):
    import usd_core.render.remote as remote_mod
    from usd_core.render.remote import RemoteRenderBackend

    monkeypatch.setattr(remote_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    # a leftover in the LEGACY shared location is still swept defensively
    import subprocess
    _child = subprocess.Popen(["sleep", "0"]); _child.wait()
    legacy_leftover = tmp_path / f"ov_remote_render_{_child.pid}_leftover"
    legacy_leftover.mkdir()
    stale = time.time() - 25 * 3600
    os.utime(legacy_leftover, (stale, stale))

    seen: dict = {}

    def boom(cls, stage, work_dir):
        seen["work_dir"] = Path(work_dir)
        raise RuntimeError("stop after staging")

    monkeypatch.setattr(RemoteRenderBackend, "_package_usdz", classmethod(boom))
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    with pytest.raises(RuntimeError, match="stop after staging"):
        backend._execute(object(), {"cameras": ["/c"]})

    assert seen["work_dir"].parent == remote_mod._work_staging_root()
    assert seen["work_dir"].parent != tmp_path  # not the shared legacy location
    assert not legacy_leftover.exists()  # legacy sweep still happened


def test_execute_staging_survives_launcher_tmpdir_cleanup(tmp_path, monkeypatch):
    """A launcher cleanup must not delete an in-flight render bundle directory."""
    import shutil

    import usd_core.render.remote as remote_mod
    from usd_core.render.remote import RemoteRenderBackend

    launcher_dir = tmp_path / ".content-workflow-codex-launcher-repro"
    launcher_dir.mkdir(mode=0o700)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staging_root = run_dir.parent / ".usd-cli-remote-render-staging"
    monkeypatch.setenv(
        remote_mod.REMOTE_RENDER_STAGING_ROOT_ENV,
        str(staging_root),
    )
    monkeypatch.setattr(
        remote_mod.tempfile,
        "gettempdir",
        lambda: str(launcher_dir),
    )
    seen: dict[str, Path | bool] = {}

    def package_after_launcher_cleanup(cls, stage, work_dir):
        work_path = Path(work_dir)
        seen["work_dir"] = work_path
        shutil.rmtree(launcher_dir)
        (work_path / "scene_bundle.usdz").write_bytes(b"in-flight bundle")
        seen["write_succeeded"] = True
        raise RuntimeError("stop after deterministic overlap")

    monkeypatch.setattr(
        RemoteRenderBackend,
        "_package_usdz",
        classmethod(package_after_launcher_cleanup),
    )
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)

    with pytest.raises(RuntimeError, match="stop after deterministic overlap"):
        backend._execute(object(), {"cameras": ["/c"]})

    assert seen["write_succeeded"] is True
    work_dir = seen["work_dir"]
    assert isinstance(work_dir, Path)
    assert work_dir.parent == staging_root
    assert remote_mod.stat_mod.S_IMODE(staging_root.stat().st_mode) == 0o700
    assert not work_dir.exists(), "per-render cleanup must remove staging artifacts"


def test_explicit_staging_root_rejects_symlink(tmp_path, monkeypatch):
    import usd_core.render.remote as remote_mod

    real_root = tmp_path / "real"
    real_root.mkdir()
    staging_link = tmp_path / "staging-link"
    staging_link.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setenv(
        remote_mod.REMOTE_RENDER_STAGING_ROOT_ENV,
        str(staging_link),
    )

    with pytest.raises(RuntimeError, match="not a directory"):
        remote_mod._work_staging_root_info()


def test_explicit_staging_root_rejects_broad_directory_without_chmod(
    tmp_path, monkeypatch
):
    import stat

    import usd_core.render.remote as remote_mod

    broad_root = tmp_path / "shared-temp"
    broad_root.mkdir()
    broad_root.chmod(0o1777)
    monkeypatch.setenv(
        remote_mod.REMOTE_RENDER_STAGING_ROOT_ENV,
        str(broad_root),
    )

    with pytest.raises(RuntimeError, match="dedicated 0700 directory"):
        remote_mod._work_staging_root_info()

    assert stat.S_IMODE(broad_root.stat().st_mode) == 0o1777


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX mode contract")
def test_explicit_staging_root_accepts_setgid_private_directory(tmp_path, monkeypatch):
    import stat

    import usd_core.render.remote as remote_mod

    private_root = tmp_path / "setgid-private"
    private_root.mkdir(mode=0o700)
    private_root.chmod(0o2700)
    monkeypatch.setenv(
        remote_mod.REMOTE_RENDER_STAGING_ROOT_ENV,
        str(private_root),
    )

    root, is_private = remote_mod._work_staging_root_info()

    assert root == private_root
    assert is_private is True
    assert stat.S_IMODE(private_root.stat().st_mode) == 0o2700


def test_explicit_empty_staging_root_fails_closed(monkeypatch):
    import usd_core.render.remote as remote_mod

    monkeypatch.setenv(remote_mod.REMOTE_RENDER_STAGING_ROOT_ENV, "")

    with pytest.raises(RuntimeError, match="must be an absolute path"):
        remote_mod._work_staging_root_info()


def test_reaper_does_not_raise_for_invalid_explicit_staging_root(monkeypatch):
    import usd_core.render.remote as remote_mod

    monkeypatch.setenv(remote_mod.REMOTE_RENDER_STAGING_ROOT_ENV, "")

    remote_mod._reap_stale_staging()


# ── backend error bodies are terminal/log-safe ──────────────────────────────────────


def test_sanitize_text_escapes_c0_and_esc():
    from usd_core.render.remote import _sanitize_text

    assert _sanitize_text("plain ascii — ünïcode ok") == "plain ascii — ünïcode ok"
    assert _sanitize_text("\x1b[2J\x07a\r\nb\x00") == "\\x1b[2J\\x07a\\x0d\\x0ab\\x00"


def test_backend_error_bodies_are_terminal_safe(caplog):
    import logging

    import httpx

    from usd_core.render.remote import RemoteRenderBackend

    backend = RemoteRenderBackend("http://gpu:8000")
    evil = "boom \x1b[8;;http://evil\x07 injected\r\nFAKE LOG LINE"
    resp = httpx.Response(500, request=httpx.Request("POST", "http://gpu:8000/render"),
                          text=evil)
    with caplog.at_level(logging.ERROR, logger="usd_core.render.remote"):
        with pytest.raises(RuntimeError) as ei:
            backend._raise_for_status(resp, 1024, "scene")
    msg = str(ei.value)
    for raw in ("\x1b", "\x07", "\r", "\n"):
        assert raw not in msg, "raw control characters leaked into the error text"
    assert "\\x1b" in msg and "injected" in msg  # visible escape, content kept
    for record in caplog.records:
        rendered = record.getMessage()
        assert "\x1b" not in rendered and "\x07" not in rendered


# ── v4: the pre-packaging estimate includes resolved asset dependencies ─────────────


def _scene_with_big_asset(tmp_path, asset_bytes: int = 5_000_000):
    """A tiny layer whose shader references a big external asset payload — the
    shape that made the old layer-sum estimate report 103.8 MB for a 1,377 MB
    bundle (USDZ packaging pulls the asset; the layer sum ignored it)."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    root = tmp_path / "scene"
    root.mkdir()
    (root / "payload.bin").write_bytes(b"\0" * asset_bytes)
    stage = Usd.Stage.CreateNew(str(root / "estimate_scene.usda"))
    UsdGeom.Mesh.Define(stage, "/World/Mesh")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/M/Tex")
    sh.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./payload.bin"))
    stage.Save()
    return stage


def test_estimate_includes_resolved_asset_dependencies(tmp_path):
    from usd_core.render.remote import _estimate_stage_bytes

    stage = _scene_with_big_asset(tmp_path)
    est, n_files = _estimate_stage_bytes(stage)
    assert est >= 5_000_000, "the referenced asset payload must be counted"
    assert n_files >= 2  # the layer AND the asset


def test_estimate_falls_back_to_the_layer_sum_when_the_walk_fails(tmp_path, monkeypatch):
    from pxr import UsdUtils

    from usd_core.render.remote import _estimate_stage_bytes

    stage = _scene_with_big_asset(tmp_path)

    def boom(*a, **kw):
        raise RuntimeError("dependency walk exploded")

    monkeypatch.setattr(UsdUtils, "ComputeAllDependencies", boom)
    est, n_files = _estimate_stage_bytes(stage)
    assert 0 < est < 5_000_000  # layer bytes only — the estimate degrades, never fails
    assert n_files == 1


def test_asset_heavy_scene_fails_fast_before_packaging(tmp_path, monkeypatch):
    """A tiny layer + a huge asset used to sail past the fail-fast tier and burn
    minutes in packaging; the truthful estimate must reject it up front."""
    from usd_core.render.remote import RemoteRenderBackend

    stage = _scene_with_big_asset(tmp_path)  # ~5 MB of assets, ~KB of layers

    def no_packaging(cls, stage, work_dir):
        raise AssertionError("packaging must not start for an over-cap scene")

    monkeypatch.setattr(RemoteRenderBackend, "_package_usdz", classmethod(no_packaging))
    backend = RemoteRenderBackend("http://gpu:8000", max_upload_mb=2,
                                  verify_version=False, bundle_cache=False)
    with pytest.raises(RuntimeError, match="before packaging"):
        backend._package_payload(stage, tmp_path / "work", "estimate_scene")


# ── v4: packaged-bundle cache (skip re-packaging identical content) ─────────────────


@pytest.fixture
def private_staging(tmp_path, monkeypatch):
    """Point the staging root (and the bundle cache under it) at this test's tmp."""
    import usd_core.render.remote as remote_mod

    monkeypatch.delenv(remote_mod.REMOTE_RENDER_STAGING_ROOT_ENV, raising=False)
    monkeypatch.setattr(remote_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    return tmp_path


def _counting_packager(monkeypatch):
    """Count real _package_usdz invocations without changing its behavior."""
    from usd_core.render.remote import RemoteRenderBackend

    calls: list[int] = []
    orig = RemoteRenderBackend._package_usdz.__func__

    def counting(cls, stage, work_dir):
        calls.append(1)
        return orig(cls, stage, work_dir)

    monkeypatch.setattr(RemoteRenderBackend, "_package_usdz", classmethod(counting))
    return calls


def _small_saved_scene(root, name="cached_scene", salt=""):
    from pxr import Usd, UsdGeom

    root.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(root / f"{name}.usda"))
    UsdGeom.Mesh.Define(stage, "/World/Mesh")
    if salt:
        UsdGeom.Mesh.Define(stage, f"/World/{salt}")
    stage.Save()
    return stage


def _work(tmp_path, i):
    d = tmp_path / f"work{i}"
    d.mkdir()
    return d


def test_bundle_cache_hit_skips_packaging_and_reuses_the_gzip(
        private_staging, tmp_path, monkeypatch):
    import usd_core.render.remote as remote_mod
    from usd_core.render.remote import RemoteRenderBackend

    stage = _small_saved_scene(tmp_path / "scene")
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    calls = _counting_packager(monkeypatch)

    first_work = _work(tmp_path, 1)
    second_work = _work(tmp_path, 2)
    first = backend._package_payload(stage, first_work, "cached_scene")
    second = backend._package_payload(stage, second_work, "cached_scene")

    assert len(calls) == 1, "identical content must not be re-packaged"
    first_usdz, first_send, first_compression = first
    usdz_path, send_path, compression = second
    assert first_compression == compression
    assert first_usdz.parent == first_work
    assert usdz_path.parent == second_work
    assert first_usdz.read_bytes() == usdz_path.read_bytes()
    assert first_send.read_bytes() == send_path.read_bytes()
    assert usdz_path.is_file() and send_path.is_file()
    cache_root = remote_mod._bundle_cache_root()
    assert cache_root is not None
    assert any(path.name.startswith("cached_scene--") for path in cache_root.iterdir())
    if compression == "gzip":  # a compressible bundle reuses its gzip too
        assert send_path.name.endswith(".usdz.gz")


def test_bundle_cache_uses_the_pinned_launcher_independent_root(
        tmp_path, monkeypatch):
    """A cached bundle must survive deletion of the launcher's private TMPDIR."""
    import shutil
    import zipfile

    import usd_core.render.remote as remote_mod
    from usd_core.render.remote import RemoteRenderBackend

    launcher_dir = tmp_path / ".content-workflow-codex-launcher-repro"
    launcher_dir.mkdir(mode=0o700)
    work_root = tmp_path / "parent-owned-remote-render-staging"
    work_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        remote_mod.tempfile,
        "gettempdir",
        lambda: str(launcher_dir),
    )
    monkeypatch.setenv(
        remote_mod.REMOTE_RENDER_STAGING_ROOT_ENV,
        str(work_root),
    )
    stage = _small_saved_scene(tmp_path / "scene")
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    calls = _counting_packager(monkeypatch)

    first_work = _work(work_root, 1)
    second_work = _work(work_root, 2)
    backend._package_payload(stage, first_work, "cached_scene")
    second = backend._package_payload(stage, second_work, "cached_scene")

    assert len(calls) == 1, "the pinned root must preserve warm cache reuse"
    usdz_path, send_path, _compression = second
    shutil.rmtree(launcher_dir)

    assert usdz_path.is_file()
    assert send_path.is_file()
    assert usdz_path.parent == second_work
    cache_root = remote_mod._bundle_cache_root()
    assert cache_root is not None
    assert not usdz_path.is_relative_to(cache_root)
    with zipfile.ZipFile(usdz_path) as archive:
        assert archive.testzip() is None


def test_bundle_cache_hit_survives_concurrent_entry_eviction(
        private_staging, tmp_path, monkeypatch):
    """An in-flight upload owns work-dir paths, not evictable cache paths."""
    import shutil

    import usd_core.render.remote as remote_mod
    from usd_core.render.remote import RemoteRenderBackend

    stage = _small_saved_scene(tmp_path / "scene")
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    calls = _counting_packager(monkeypatch)

    backend._package_payload(stage, _work(tmp_path, 1), "cached_scene")
    work = _work(tmp_path, 2)
    usdz_path, send_path, _compression = backend._package_payload(
        stage, work, "cached_scene"
    )
    assert len(calls) == 1

    cache_root = remote_mod._bundle_cache_root()
    assert cache_root is not None
    for entry in cache_root.iterdir():
        shutil.rmtree(entry)

    assert usdz_path.parent == work and usdz_path.is_file()
    assert send_path.parent == work and send_path.is_file()
    assert usdz_path.read_bytes()
    assert send_path.read_bytes()


def test_bundle_cache_invalidates_when_the_scene_changes(
        private_staging, tmp_path, monkeypatch):
    from pxr import UsdGeom

    from usd_core.render.remote import RemoteRenderBackend

    stage = _small_saved_scene(tmp_path / "scene")
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    calls = _counting_packager(monkeypatch)

    backend._package_payload(stage, _work(tmp_path, 1), "cached_scene")
    UsdGeom.Mesh.Define(stage, "/World/Another")  # an in-memory edit (dirty layer)
    backend._package_payload(stage, _work(tmp_path, 2), "cached_scene")
    assert len(calls) == 2, "edited content must be re-packaged (content-hash key)"

    backend._package_payload(stage, _work(tmp_path, 3), "cached_scene")
    assert len(calls) == 2  # the edited state itself is now cached


def test_bundle_cache_can_be_disabled(private_staging, tmp_path, monkeypatch):
    from usd_core.config import Config
    from usd_core.render.factory import make_backend
    from usd_core.render.remote import RemoteRenderBackend

    stage = _small_saved_scene(tmp_path / "scene")
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False,
                                  bundle_cache=False)
    calls = _counting_packager(monkeypatch)
    backend._package_payload(stage, _work(tmp_path, 1), "cached_scene")
    backend._package_payload(stage, _work(tmp_path, 2), "cached_scene")
    assert len(calls) == 2  # every render packages

    cfg = Config()
    cfg.render["renderer"] = "remote"
    cfg.render["remote_url"] = "http://gpu:8000"
    cfg.render["remote_bundle_cache"] = "false"
    assert make_backend(cfg)._bundle_cache is False
    cfg.render["remote_bundle_cache"] = True
    assert make_backend(cfg)._bundle_cache is True  # and the default is on


def test_bundle_cache_lru_keeps_the_newest_four_per_scene(
        private_staging, tmp_path, monkeypatch):
    import usd_core.render.remote as remote_mod
    from usd_core.render.remote import RemoteRenderBackend

    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    _counting_packager(monkeypatch)
    for i in range(6):  # six distinct contents under the same scene stem
        stage = _small_saved_scene(tmp_path / f"v{i}", name="lru_scene",
                                   salt=f"Variant{i}")
        backend._package_payload(stage, _work(tmp_path, i), "lru_scene")

    entries = [d for d in remote_mod._bundle_cache_root().iterdir()
               if d.is_dir() and d.name.startswith("lru_scene--")]
    assert len(entries) == 4, "the cache must keep only the newest 4 bundles per scene"


def test_bundle_cache_key_covers_unfingerprintable_stages(private_staging):
    from usd_core.render.remote import RemoteRenderBackend

    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)

    class StageStub:  # no layers to fingerprint — such renders are simply uncached
        pass

    assert backend._bundle_cache_key(StageStub()) is None
    assert backend._bundle_cache_key(None) is None


# ── v4: uniform, logged compression policy ──────────────────────────────────────────


def test_prepare_upload_logs_the_raw_upload_decision(tmp_path, caplog):
    import logging

    from usd_core.render.remote import RemoteRenderBackend

    incompressible = tmp_path / "bundle.usdz"
    incompressible.write_bytes(os.urandom(512 * 1024))  # gzip cannot help

    backend = RemoteRenderBackend("http://gpu:8000")
    with caplog.at_level(logging.INFO, logger="usd_core.render.remote"):
        send, compression = backend._prepare_upload(incompressible)
    assert compression == "none" and send == incompressible
    assert any("uploading the raw" in r.getMessage() for r in caplog.records), (
        "a raw upload must be visibly a policy outcome, not a silently skipped gzip")

    caplog.clear()
    off = RemoteRenderBackend("http://gpu:8000", compress=False)
    with caplog.at_level(logging.INFO, logger="usd_core.render.remote"):
        send, compression = off._prepare_upload(incompressible)
    assert compression == "none"
    assert any("disabled by config" in r.getMessage() for r in caplog.records)


def test_prepare_upload_still_gzips_compressible_payloads(tmp_path, caplog):
    import logging

    from usd_core.render.remote import RemoteRenderBackend

    compressible = tmp_path / "bundle.usdz"
    compressible.write_bytes(b"a" * 512 * 1024)

    backend = RemoteRenderBackend("http://gpu:8000")
    with caplog.at_level(logging.INFO, logger="usd_core.render.remote"):
        send, compression = backend._prepare_upload(compressible)
    assert compression == "gzip" and send.name.endswith(".usdz.gz")
    assert any("compressed" in r.getMessage() for r in caplog.records)


# ── v4: packaging-warning summaries classify package-internal remap noise ───────────


def test_warning_summary_separates_package_internal_remap_noise(caplog):
    """usdUtils re-warns about its OWN numbered-archive remaps (`@0/tex.png@` —
    benchmark task-05's flood); those assets ARE in the bundle and must not be
    reported as unresolved references the agent then chases."""
    import logging

    from usd_core.render.remote import _log_packaging_warnings

    captured = "\n".join([
        "Warning: in _EnqueueDependency ... Failed to resolve reference "
        "@0/t_rubber_new_a01_tile_orm.png@ with computed asset path "
        "@0/t_rubber_new_a01_tile_orm.png@ found in layer @/tmp/x/_flattened.usdc@.",
        "Warning: in _EnqueueDependency ... Failed to resolve reference "
        "@1/t_rubber_new_a01_tile_alb.png@ with computed asset path "
        "@1/t_rubber_new_a01_tile_alb.png@ found in layer @/tmp/x/_flattened.usdc@.",
        "Warning: in _EnqueueDependency ... Failed to resolve reference "
        "@../missing/tex.png@ with computed asset path @/abs/missing/tex.png@ "
        "found in layer @/tmp/x/_flattened.usdc@.",
        "Warning: some other USD noise line",
    ])
    with caplog.at_level(logging.WARNING, logger="usd_core.render.remote"):
        _log_packaging_warnings(captured)
    warned = [r for r in caplog.records if "suppressed" in r.getMessage()]
    assert len(warned) == 1
    msg = warned[0].getMessage()
    assert "suppressed 4 USD warning line(s)" in msg
    assert "1 look like unresolved asset references" in msg
    assert "2 are usdUtils package-internal remap noise" in msg
    # the example shown first is a GENUINELY unresolved reference, not remap noise
    assert "@../missing/tex.png@" in msg


# ── v5: the bundle-cache key covers external assets + small-layer content ──────────


def test_bundle_cache_invalidates_when_an_external_asset_changes(
        private_staging, tmp_path, monkeypatch):
    """Review item 9: the key must include resolved external asset dependencies —
    editing a texture/payload in place (no layer touched) used to serve the stale
    cached bundle."""
    from usd_core.render.remote import RemoteRenderBackend

    stage = _scene_with_big_asset(tmp_path, asset_bytes=1024)
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    calls = _counting_packager(monkeypatch)

    backend._package_payload(stage, _work(tmp_path, 1), "estimate_scene")
    backend._package_payload(stage, _work(tmp_path, 2), "estimate_scene")
    assert len(calls) == 1  # unchanged asset: still a cache hit

    # retexture IN PLACE: no layer changes, only the referenced asset file
    (tmp_path / "scene" / "payload.bin").write_bytes(b"\1" * 2048)
    backend._package_payload(stage, _work(tmp_path, 3), "estimate_scene")
    assert len(calls) == 2, "an edited external asset must invalidate the bundle"


def test_bundle_cache_key_catches_same_stat_byte_changes_in_small_layers(
        private_staging, tmp_path):
    """Review item 9: clean on-disk layers under the 8 MB threshold mix a content
    sha into the key — different bytes with a preserved size+mtime_ns stat must
    not collide onto the same cached bundle."""
    from usd_core.render.remote import RemoteRenderBackend

    stage = _small_saved_scene(tmp_path / "scene")
    layer_file = tmp_path / "scene" / "cached_scene.usda"
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    key1 = backend._bundle_cache_key(stage)
    assert key1

    st = layer_file.stat()
    text = layer_file.read_text()
    assert '"Mesh"' in text
    layer_file.write_text(text.replace('"Mesh"', '"Mush"'))  # same byte length
    os.utime(layer_file, ns=(st.st_atime_ns, st.st_mtime_ns))  # forge the stat key
    after = layer_file.stat()
    assert (after.st_size, after.st_mtime_ns) == (st.st_size, st.st_mtime_ns)

    key2 = backend._bundle_cache_key(stage)
    assert key2 and key2 != key1, "a same-stat byte change collided the cache key"


def test_walk_failure_makes_the_stage_uncacheable(
        private_staging, tmp_path, monkeypatch):
    """Review item 9: when the dependency walk fails, the dependency set is
    unknown — the stage must be uncacheable rather than risk serving a bundle
    whose (unhashed) assets have changed."""
    import usd_core.render.remote as remote_mod
    from usd_core.render.remote import RemoteRenderBackend

    stage = _small_saved_scene(tmp_path / "scene")
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    assert backend._bundle_cache_key(stage)  # sanity: normally cacheable

    monkeypatch.setattr(remote_mod, "_walk_asset_dependencies", lambda s: None)
    assert backend._bundle_cache_key(stage) is None
    calls = _counting_packager(monkeypatch)
    backend._package_payload(stage, _work(tmp_path, 1), "cached_scene")
    backend._package_payload(stage, _work(tmp_path, 2), "cached_scene")
    assert len(calls) == 2  # every render re-packages; no stale-bundle risk


# ── v5: cache trust — private root only, regular owned artifacts only ──────────────


def test_squatted_staging_root_disables_the_bundle_cache(tmp_path, monkeypatch):
    """Review item 10 (security): when the private staging root is squatted, staging
    falls back to the shared temp dir — the bundle cache must turn OFF there
    instead of trusting a fixed path other local users control."""
    import usd_core.render.remote as remote_mod
    from usd_core.render.remote import RemoteRenderBackend

    monkeypatch.delenv(remote_mod.REMOTE_RENDER_STAGING_ROOT_ENV, raising=False)
    monkeypatch.setattr(remote_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    uid = os.getuid() if hasattr(os, "getuid") else None
    name = f"dsc3-{uid}" if uid is not None else "dsc3"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / name).symlink_to(elsewhere)  # the squat
    assert remote_mod._staging_root() == tmp_path  # shared-tmp fallback (existing)
    assert remote_mod._bundle_cache_root() is None  # cache disabled on the fallback

    stage = _small_saved_scene(tmp_path / "scene")
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    calls = _counting_packager(monkeypatch)
    backend._package_payload(stage, _work(tmp_path, 1), "cached_scene")
    backend._package_payload(stage, _work(tmp_path, 2), "cached_scene")
    assert len(calls) == 2  # no caching in a directory other users control
    assert not (tmp_path / "dsc3_bundle_cache").exists()  # and none was created there


def test_bundle_cache_reaper_never_raises_for_invalid_explicit_root(monkeypatch):
    import usd_core.render.remote as remote_mod

    monkeypatch.setenv(remote_mod.REMOTE_RENDER_STAGING_ROOT_ENV, "relative")

    assert remote_mod._bundle_cache_root() is None
    remote_mod._reap_bundle_cache()  # hygiene must not become a render gate


def test_symlinked_cache_root_disables_the_cache(private_staging, tmp_path):
    """Review item 10 (security): the cache root itself must be a real directory
    owned by us — a symlink squatting the name disables caching."""
    import usd_core.render.remote as remote_mod

    staging = remote_mod._staging_root()
    elsewhere = tmp_path / "cache-elsewhere"
    elsewhere.mkdir()
    (staging / remote_mod._BUNDLE_CACHE_DIRNAME).symlink_to(elsewhere)
    assert remote_mod._bundle_cache_root() is None


def test_symlinked_cache_artifact_is_rejected(private_staging, tmp_path, monkeypatch):
    """Review item 10 (security): a symlink planted as a cached artifact must
    neither be served nor uploaded on the victim's behalf — the hit is rejected
    and the scene re-packages."""
    import usd_core.render.remote as remote_mod
    from usd_core.render.remote import RemoteRenderBackend

    stage = _small_saved_scene(tmp_path / "scene")
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    key = backend._bundle_cache_key(stage)
    assert key
    entry = remote_mod._bundle_cache_entry_dir(key, "cached_scene")
    assert entry is not None
    entry.mkdir(parents=True)
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"another user's readable file")
    (entry / "scene_bundle.usdz").symlink_to(secret)

    assert remote_mod._bundle_cache_get(entry) is None  # the poisoned hit is rejected

    calls = _counting_packager(monkeypatch)
    usdz_path, send_path, _compression = backend._package_payload(
        stage, _work(tmp_path, 1), "cached_scene")
    assert len(calls) == 1  # re-packaged, not served from the poisoned entry
    assert Path(usdz_path).resolve() != secret.resolve()
    assert Path(send_path).resolve() != secret.resolve()


def test_poisoned_gzip_sibling_rejects_the_whole_entry(private_staging, tmp_path):
    """Review item 10 (security): a planted gzip sibling must not ride along with a
    valid usdz — the whole entry is distrusted (no silent raw-usdz fallback)."""
    import usd_core.render.remote as remote_mod

    root = remote_mod._bundle_cache_root()
    assert root is not None
    entry = root / "scene--deadbeef"
    entry.mkdir()
    (entry / "scene_bundle.usdz").write_bytes(b"usdz")
    secret = tmp_path / "gz-secret"
    secret.write_bytes(b"x")
    (entry / "scene_bundle.usdz.gz").symlink_to(secret)
    assert remote_mod._bundle_cache_get(entry) is None


# ── v5: anonymous/in-memory root layers still estimate their asset payloads ────────


def test_anonymous_stage_estimate_counts_asset_dependencies(tmp_path, monkeypatch):
    """Review item 18: an in-memory root layer over big asset dependencies used to
    estimate as (0, 0) — asset-valued attributes are now scanned directly, so the
    fail-fast tier can fire before minutes of packaging."""
    from pxr import Sdf, Usd, UsdShade

    from usd_core.render.remote import RemoteRenderBackend, _estimate_stage_bytes

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"\0" * 3_000_000)
    stage = Usd.Stage.CreateInMemory()
    sh = UsdShade.Shader.Define(stage, "/World/Looks/M/Tex")
    sh.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(payload)))

    est, n_files = _estimate_stage_bytes(stage)
    assert est >= 3_000_000, "the anonymous stage's asset payload must be counted"
    assert n_files >= 1

    # and the fail-fast tier actually fires for it, before any packaging
    def no_packaging(cls, stage, work_dir):
        raise AssertionError("packaging must not start for an over-cap scene")

    monkeypatch.setattr(RemoteRenderBackend, "_package_usdz", classmethod(no_packaging))
    backend = RemoteRenderBackend("http://gpu:8000", max_upload_mb=1,
                                  verify_version=False, bundle_cache=False)
    with pytest.raises(RuntimeError, match="before packaging"):
        backend._package_payload(stage, tmp_path / "work", "anon")
