# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-9 MED fixes from the round-8 field analyses.

  1  snapshot on LARGE stages defers the full-stage snapdiff capture until a
     diff is requested (119k prims: the capture was 8.4s of an 11.1s snapshot
     and starved sibling sessions into daemon-busy timeouts); `-D` banks a
     baseline and diffs exactly as before
"""
from __future__ import annotations

import pathlib

from pxr import Usd, UsdGeom

import usd_core.session as session_mod
from usd_core.config import Config
from usd_core.session import Session


def _scene(tmp_path, n=6):
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for i in range(n):
        UsdGeom.Cube.Define(stage, f"/World/cube_{i}")
    path = tmp_path / "scene.usda"
    stage.GetRootLayer().Export(str(path))
    return path


def test_large_stage_snapshot_defers_diff_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "_SNAPDIFF_CAPTURE_MAX_PRIMS", 3)
    s = Session(Config(project_dir=tmp_path), name="t")
    s.open_stage(str(_scene(tmp_path)))

    r = s.snapshot()  # "large" stage: no baseline banked, says so
    assert r.ok
    assert r.summary["diff_baseline"] == "not banked (large stage)"
    assert any("snapshot -D" in i.message for i in r.issues)
    assert s._prev_snapshot is None

    r = s.snapshot(diff=True)  # first -D banks the baseline
    assert r.summary["diff"] == "baseline"
    assert s._prev_snapshot is not None

    s._stage.DefinePrim("/World/added_later", "Xform")
    r = s.snapshot(diff=True)  # second -D diffs against it
    assert r.data["diff"]["counts"]["added"] >= 1


def test_small_stage_snapshot_still_banks_every_time(tmp_path):
    s = Session(Config(project_dir=tmp_path), name="t")
    s.open_stage(str(_scene(tmp_path)))
    r = s.snapshot()
    assert "diff_baseline" not in r.summary
    assert s._prev_snapshot is not None


#   2  selector-surface gaps: find --where, usd-cli bounds, usd-cli visibility, tolerant
#      attr: float equality, USD-encoded (_UXX_) name matching


def _selector_scene(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    a = UsdGeom.Cube.Define(stage, "/World/panel_a")
    a.CreateDisplayColorAttr([(0.0100001, 0.02, 0.03)])
    b = UsdGeom.Cube.Define(stage, "/World/panel_b")
    b.CreateDisplayColorAttr([(0.9, 0.8, 0.7)])
    # the real importer encoding, verified in the Siemens scene:
    # Male_U20_right_U20_hand == "Male right hand"
    UsdGeom.Cube.Define(stage, "/World/Steel_U20_Painted_U20_White")
    path = tmp_path / "sel.usda"
    stage.GetRootLayer().Export(str(path))
    return path


def test_find_where_attr_rule_with_float_tolerance(tmp_path):
    s = Session(Config(project_dir=tmp_path), name="t")
    s.open_stage(str(_selector_scene(tmp_path)))
    # formatting/bracket style and 1e-5 float noise must not matter
    r = s.find(where=["attr:primvars:displayColor==[(0.01, 0.02, 0.03)]"])
    assert r.ok, r.issues
    assert [row["name"] for row in r.data["results"]] == ["panel_a"]
    r = s.find(where=["attr:primvars:displayColor!=(0.01,0.02,0.03)"], type="Cube")
    names = {row["name"] for row in r.data["results"]}
    assert "panel_a" not in names and "panel_b" in names


def test_find_name_matches_usd_encoded_labels(tmp_path):
    s = Session(Config(project_dir=tmp_path), name="t")
    s.open_stage(str(_selector_scene(tmp_path)))
    r = s.find(name="Steel Painted*")
    assert [row["name"] for row in r.data["results"]] == \
        ["Steel_U20_Painted_U20_White"]


def test_bounds_one_liner(tmp_path):
    s = Session(Config(project_dir=tmp_path), name="t")
    s.open_stage(str(_selector_scene(tmp_path)))
    ref = s.refs.ref_for_path("/World/panel_a")
    r = s.bounds(ref)
    assert r.ok
    assert r.summary["size"] == [2.0, 2.0, 2.0]
    r = s.bounds()  # whole stage
    assert r.ok and r.summary["target"] == "/"


def test_visibility_reports_hiding_ancestor(tmp_path):
    s = Session(Config(project_dir=tmp_path), name="t")
    s.open_stage(str(_selector_scene(tmp_path)))
    world_ref = s.refs.ref_for_path("/World")
    cube_ref = s.refs.ref_for_path("/World/panel_a")
    r = s.visibility(cube_ref)
    assert r.summary["computed"] == "inherited"
    UsdGeom.Imageable(s._stage.GetPrimAtPath("/World")).GetVisibilityAttr() \
        .Set("invisible") if False else None
    from pxr import UsdGeom as _ug
    _ug.Imageable(s._stage.GetPrimAtPath("/World")).CreateVisibilityAttr() \
        .Set("invisible")
    r = s.visibility(cube_ref)
    assert r.summary["computed"] == "invisible"
    assert r.summary["hidden_by"] == world_ref
    assert "hidden by ancestor" in r.data["text"]


#   3  ovrtx auto-install streams its output — a 2.5 GB pip download used to
#      produce ZERO log lines for ~10 minutes (looked like a hang)


def test_run_logged_streams_and_heartbeats(caplog):
    import logging as _logging
    import sys as _sys

    from usd_core.render import ovrtx as ovrtx_mod

    with caplog.at_level(_logging.INFO, logger=ovrtx_mod.logger.name):
        ovrtx_mod._run_logged(
            [_sys.executable, "-c",
             "import time; print('line one'); time.sleep(0.25); print('line two')"],
            what="test-install", heartbeat_s=0.1)
    msgs = [r.message for r in caplog.records]
    assert any("test-install: line one" in m for m in msgs)
    assert any("test-install: line two" in m for m in msgs)
    assert any("still running" in m for m in msgs)  # heartbeat through silence


def test_run_logged_failure_carries_output_tail():
    import sys as _sys

    import pytest as _pytest

    from usd_core.render import ovrtx as ovrtx_mod

    with _pytest.raises(RuntimeError, match="boom-detail"):
        ovrtx_mod._run_logged(
            [_sys.executable, "-c", "print('boom-detail'); raise SystemExit(3)"],
            what="test-install")


def test_nonblocking_provision_fails_fast_and_reports_progress(tmp_path, monkeypatch):
    """Render-path provisioning must NOT block: first call kicks off the install
    and errors with 'STARTED', a second call reports elapsed progress, and after
    the worker finishes a call returns the venv python. The daemon thread doing
    a real 2.5 GB download used to hold the session lock for ~10 minutes."""
    import time as _time

    from usd_core.render import ovrtx as ovrtx_mod

    venv = tmp_path / "venv"
    release = ovrtx_mod.threading.Event()

    def _fake_provision(venv_dir):
        release.wait(5)
        (venv_dir / "bin").mkdir(parents=True)
        (venv_dir / "bin" / "python").write_text("")
        (venv_dir / ".usd-cli-ovrtx-ready").write_text(
            ovrtx_mod._readiness_marker_body(ovrtx_mod._ovrtx_runtime_lock())
        )

    monkeypatch.setattr(ovrtx_mod, "_provision_venv", _fake_provision)
    monkeypatch.setattr(ovrtx_mod, "_env_ovrtx_python", lambda: None)
    monkeypatch.setitem(ovrtx_mod._PROVISION, "thread", None)
    monkeypatch.setitem(ovrtx_mod._PROVISION, "error", None)

    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="STARTED in the background"):
        ovrtx_mod._ovrtx_python(venv, auto_install=True, block=False)
    with _pytest.raises(RuntimeError, match="in progress"):
        ovrtx_mod._ovrtx_python(venv, auto_install=True, block=False)
    release.set()
    for _ in range(50):
        if not ovrtx_mod._PROVISION["thread"].is_alive():
            break
        _time.sleep(0.05)
    assert ovrtx_mod._ovrtx_python(venv, auto_install=True, block=False) == \
        str(venv / "bin" / "python")


def test_nonblocking_provision_surfaces_worker_failure(tmp_path, monkeypatch):
    import time as _time

    from usd_core.render import ovrtx as ovrtx_mod

    def _boom(venv_dir):
        raise RuntimeError("index unreachable")

    monkeypatch.setattr(ovrtx_mod, "_provision_venv", _boom)
    monkeypatch.setattr(ovrtx_mod, "_env_ovrtx_python", lambda: None)
    monkeypatch.setitem(ovrtx_mod._PROVISION, "thread", None)
    monkeypatch.setitem(ovrtx_mod._PROVISION, "error", None)

    import pytest as _pytest
    venv = tmp_path / "venv2"
    with _pytest.raises(RuntimeError, match="STARTED"):
        ovrtx_mod._ovrtx_python(venv, auto_install=True, block=False)
    for _ in range(50):
        th = ovrtx_mod._PROVISION["thread"]
        if th is None or not th.is_alive():
            break
        _time.sleep(0.05)
    with _pytest.raises(RuntimeError, match="FAILED: index unreachable"):
        ovrtx_mod._ovrtx_python(venv, auto_install=True, block=False)


#   4  segmentation renders write their FULL legend to a .legend.txt beside the
#      PNG (the CLI shows 12 rows; a 2,800-entry legend was reachable only via
#      a giant --json dump)


def test_segmentation_render_writes_full_legend_file(tmp_path, monkeypatch):
    from usd_core import render as render_pkg
    from usd_core.render.base import RenderResult

    class OvrtxStub:
        name = "ovrtx"

        def render(self, _stage, cameras, width, height, out_dir, **_kwargs):
            path = pathlib.Path(out_dir) / "rgb.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")
            return [RenderResult(path=str(path), camera=cameras[0], width=width,
                                 height=height, backend=self.name)]

    monkeypatch.setattr(render_pkg, "make_backend", lambda _config: OvrtxStub())
    s = Session(Config(project_dir=tmp_path), name="t")
    s.open_stage(str(_selector_scene(tmp_path)))
    r = s.render(seg=True, res="160x120", output=str(tmp_path / "seg.png"))
    assert r.ok, r.issues
    legend_arts = [a for a in r.artifacts if a.kind == "legend"]
    assert legend_arts, [a.kind for a in r.artifacts]
    lf = pathlib.Path(legend_arts[0].path)
    assert lf.exists() and lf.name.endswith(".legend.txt")
    body = lf.read_text()
    legend = r.data["segmentation_legend"]
    legend_paths = r.data["segmentation_legend_paths"]
    assert len(body.strip().splitlines()) == len(legend)  # FULL, not capped
    ref, color = next(iter(legend.items()))
    assert f"{ref}\trgb({color[0]},{color[1]},{color[2]})" in body
    assert legend_paths[ref].startswith("/World/")
    assert r.data["segmentation_legend_file"] == str(lf)


def test_repeated_depth_renders_preserve_each_metric_artifact(tmp_path, monkeypatch):
    from usd_core import render as render_pkg
    from usd_core.render.base import RenderResult

    class OvrtxStub:
        name = "ovrtx"

        def render(self, _stage, cameras, width, height, out_dir, **_kwargs):
            path = pathlib.Path(out_dir) / "rgb.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")
            return [
                RenderResult(
                    path=str(path),
                    camera=cameras[0],
                    width=width,
                    height=height,
                    backend=self.name,
                )
            ]

    monkeypatch.setattr(render_pkg, "make_backend", lambda _config: OvrtxStub())
    session = Session(Config(project_dir=tmp_path), name="depth-uniqueness")
    session.open_stage(str(_selector_scene(tmp_path)))

    first = session.render(depth=True, res="160x120", output=str(tmp_path / "a.png"))
    second = session.render(depth=True, res="160x120", output=str(tmp_path / "b.png"))

    first_raw = pathlib.Path(
        next(artifact.path for artifact in first.artifacts if artifact.label == "linear_depth")
    )
    second_raw = pathlib.Path(
        next(artifact.path for artifact in second.artifacts if artifact.label == "linear_depth")
    )
    assert first.ok and second.ok
    assert first_raw != second_raw
    assert first_raw.is_file() and second_raw.is_file()


#   5  CLI parse errors are one clean line (a mistyped `--screen 512 512`
#      printed a 40-line rich traceback), and screen-ray hits print plain
#      floats (np.float64 reprs leaked into the hit line)


def test_vec_errors_name_the_argument_and_shape():
    import pytest as _pytest

    from usd_cli.parsing import vec

    assert vec("512,512", 2, "--screen") == [512.0, 512.0]
    with _pytest.raises(ValueError, match=r"--screen: expected 2 comma-separated"):
        vec("512", 2, "--screen")
    with _pytest.raises(ValueError, match=r"ORIGIN: .*e\.g\. 1,2,3"):
        vec("512", 3, "ORIGIN")
    with _pytest.raises(ValueError, match="no spaces between components"):
        vec("1", 3, "ORIGIN")


def test_cli_bad_vec_is_one_clean_error_line(tmp_path):
    import subprocess as _sp
    import sys as _sys

    out = _sp.run([_sys.executable, "-m", "usd_cli", "raycast",
                   "--screen", "512", "512"],
                  capture_output=True, text=True, cwd=tmp_path,
                  env={**__import__('os').environ, "USD_CLI_NO_DAEMON": "1"})
    assert out.returncode == 2
    assert "error: ORIGIN: expected 3 comma-separated components" in out.stderr
    assert "Traceback" not in out.stderr and "│" not in out.stderr


def test_raycast_screen_hit_prints_plain_floats(tmp_path):
    import numpy as np

    s = Session(Config(project_dir=tmp_path), name="t")
    s.open_stage(str(_selector_scene(tmp_path)))
    s.camera_fit()  # frames the scene; default render resolution is 1024x1024
    r = s.raycast(screen=[512.0, 512.0])  # image center -> guaranteed geometry
    assert r.ok and r.summary["hit"] is True, r.data
    assert "np.float64" not in r.data["text"]
    assert all(isinstance(v, float) and not isinstance(v, np.floating)
               for v in r.data["point"])
    assert isinstance(r.data["distance"], float)


#   6  local ovrtx child is SHARED across render-backend constructions — every
#      render used to spawn a fresh process and recompile shaders


def test_ovrtx_daemon_shared_across_backend_constructions(tmp_path, monkeypatch):
    from usd_core.render import ovrtx as ovrtx_mod

    spawned = []

    class FakeDaemon:
        def __init__(self, venv_dir, log_level="warn", auto_install=False,
                     block_install=True):
            spawned.append(self)
            self._proc = type("P", (), {"poll": staticmethod(lambda: None)})()

        def alive(self):
            return True

    monkeypatch.setattr(ovrtx_mod, "_OvRTXDaemon", FakeDaemon)
    monkeypatch.setattr(ovrtx_mod, "_DAEMON_REGISTRY", {})

    # two independent backends (= two renders, the daemon's per-command
    # construction pattern) share ONE child
    b1 = ovrtx_mod.OvRTXRenderBackend(venv_dir=tmp_path / "venv")
    b2 = ovrtx_mod.OvRTXRenderBackend(venv_dir=tmp_path / "venv")
    d1 = b1._ensure_daemon()
    d2 = b2._ensure_daemon()
    assert d1 is d2 and len(spawned) == 1

    # a different venv is a different child
    b3 = ovrtx_mod.OvRTXRenderBackend(venv_dir=tmp_path / "other")
    assert b3._ensure_daemon() is not d1 and len(spawned) == 2

    # restart evicts: the NEXT construction gets a fresh child, not the corpse
    b1.restart_daemon()
    d4 = ovrtx_mod.OvRTXRenderBackend(venv_dir=tmp_path / "venv")._ensure_daemon()
    assert d4 is not d1 and len(spawned) == 3


def test_ovrtx_dead_shared_daemon_is_replaced(tmp_path, monkeypatch):
    from usd_core.render import ovrtx as ovrtx_mod

    class FakeDaemon:
        dead = False

        def __init__(self, *a, **kw):
            self._proc = type("P", (), {"poll": staticmethod(lambda: None)})()

        def alive(self):
            return not self.dead

    monkeypatch.setattr(ovrtx_mod, "_OvRTXDaemon", FakeDaemon)
    monkeypatch.setattr(ovrtx_mod, "_DAEMON_REGISTRY", {})
    b = ovrtx_mod.OvRTXRenderBackend(venv_dir=tmp_path / "venv")
    d1 = b._ensure_daemon()
    d1.dead = True  # GPU child died — a new construction must NOT reuse it
    d2 = ovrtx_mod.OvRTXRenderBackend(venv_dir=tmp_path / "venv")._ensure_daemon()
    assert d2 is not d1


def test_ovrtx_daemon_converts_usd_time_codes_to_seconds():
    from usd_core.render import ovrtx as ovrtx_mod

    script = ovrtx_mod._DAEMON_SCRIPT
    assert "def _seconds_for_time_code" in script
    assert "return float(time_code) / tcps" in script
    assert script.count("_seconds_for_time_code(req,") == 2
    assert 'requested_time = req.get("usd_time")' in script
    assert "if requested_time is not None:" in script


def test_ovrtx_render_product_paths_do_not_alias_camera_paths():
    from usd_core.render import ovrtx as ovrtx_mod

    first = ovrtx_mod._product_path("/A_B/C")
    second = ovrtx_mod._product_path("/A/B_C")

    assert first != second
    assert first == ovrtx_mod._product_path("/A_B/C")


def test_ovrtx_worker_is_bound_to_project_daemon_lifetime(tmp_path, monkeypatch):
    import io
    import json
    import os

    from usd_core.render import ovrtx as ovrtx_mod

    captured = {}
    ready = {
        "status": "ready",
        "runtime_versions": dict(ovrtx_mod._QUALIFIED_WORKER_RUNTIME_VERSIONS),
    }

    class FakeProcess:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs["env"]
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(json.dumps(ready) + "\n")
            self._returncode = None

        def poll(self):
            return self._returncode

        def wait(self, timeout=None):
            self._returncode = 0
            return 0

        def kill(self):
            self._returncode = -9

    monkeypatch.setattr(ovrtx_mod, "_ovrtx_python", lambda *args, **kwargs: "python")
    monkeypatch.setattr(ovrtx_mod.subprocess, "Popen", FakeProcess)

    daemon = ovrtx_mod._OvRTXDaemon(tmp_path / "venv")
    try:
        assert captured["command"][:2] == ["python", "-I"]
        worker_script = pathlib.Path(captured["command"][2])
        assert worker_script.name.startswith("usd_cli_ovrtx_worker_")
        assert worker_script.suffix == ".py"
        assert worker_script.is_file()
        assert captured["env"]["USD_CLI_OVRTX_PARENT_PID"] == str(os.getpid())
        assert "PR_SET_PDEATHSIG" in ovrtx_mod._DAEMON_SCRIPT
        assert "libc.prctl(1, signal.SIGKILL" in ovrtx_mod._DAEMON_SCRIPT
        assert "os.getppid() != expected_parent" in ovrtx_mod._DAEMON_SCRIPT
    finally:
        daemon.close()


def test_ovrtx_daemon_rejects_unqualified_worker_runtime(tmp_path, monkeypatch):
    import io
    import json

    import pytest
    from usd_core.render import ovrtx as ovrtx_mod

    spawned = []
    runtime_versions_to_report = [None]

    class FakeProcess:
        def __init__(self, _command, **_kwargs):
            spawned.append(self)
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                json.dumps(
                    {
                        "status": "ready",
                        "runtime_versions": runtime_versions_to_report[0],
                    }
                )
                + "\n"
            )
            self._returncode = None

        def poll(self):
            return self._returncode

        def wait(self, timeout=None):
            return self._returncode

        def kill(self):
            self._returncode = -9

    monkeypatch.setattr(ovrtx_mod, "_ovrtx_python", lambda *args, **kwargs: "python")
    monkeypatch.setattr(ovrtx_mod.subprocess, "Popen", FakeProcess)

    unqualified_profiles = [
        None,
        {"ovrtx": "0.3.0.312915", "ovstage": "0.1.1.355824", "warp": "1.16.0"},
    ]
    for runtime_versions in unqualified_profiles:
        runtime_versions_to_report[0] = runtime_versions
        with pytest.raises(RuntimeError, match="runtime profile mismatch"):
            ovrtx_mod._OvRTXDaemon(tmp_path / "venv")

    assert len(spawned) == len(unqualified_profiles)
    assert all(process.poll() == -9 for process in spawned)


def test_close_shared_ovrtx_daemons_closes_each_worker_once(monkeypatch):
    from usd_core.render import ovrtx as ovrtx_mod

    closed = []

    class FakeDaemon:
        def close(self):
            closed.append(self)

    first = FakeDaemon()
    second = FakeDaemon()
    registry = {
        ("a", "warn"): first,
        ("alias", "warn"): first,
        ("b", "warn"): second,
    }
    monkeypatch.setattr(ovrtx_mod, "_DAEMON_REGISTRY", registry)

    ovrtx_mod.close_shared_daemons()

    assert closed == [first, second]
    assert registry == {}


def test_ovrtx_parent_binding_guards_platform_and_malformed_pid():
    from usd_core.render import ovrtx as ovrtx_mod

    script = ovrtx_mod._DAEMON_SCRIPT
    platform_guard = 'if not sys.platform.startswith("linux"):'
    parent_parse = 'expected_parent = int(os.environ.get("USD_CLI_OVRTX_PARENT_PID"'

    assert script.index(platform_guard) < script.index(parent_parse)
    assert "except ValueError:\n        expected_parent = 0" in script


#   7  round-9 field defects: inf-valued attrs 500'd JSON encoding (mislabeled
#      "unreachable"), and a configured remote GPU pool must remain selected.


def test_properties_inf_attr_is_json_safe(tmp_path):
    import json as _json
    import math

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/World/cube")
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/joint")
    joint.CreateBreakForceAttr(float("inf"))
    path = tmp_path / "phys.usda"
    stage.GetRootLayer().Export(str(path))

    s = Session(Config(project_dir=tmp_path), name="t")
    s.open_stage(str(path))
    ref = s.refs.ref_for_path("/World/joint")
    r = s.properties(ref, attr=["physics:breakForce"])
    assert r.ok
    # the whole envelope must survive a STRICT encoder (FastAPI uses allow_nan=False)
    _json.dumps(r.to_dict(), allow_nan=False)
    assert r.data["attrs"]["physics:breakForce"]["value"] == "inf"
    r = s.properties(ref)  # whole-prim path too
    _json.dumps(r.to_dict(), allow_nan=False)
