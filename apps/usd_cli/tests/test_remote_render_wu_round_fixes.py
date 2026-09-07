# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote-render failure behavior — fixes for the 2026-07-09 WU benchmark round.

GPU-free unit tests (no remote service, no RTX box needed) covering:

* backend error observability: non-2xx responses surface the response body in the
  raised error and log the request context (task-02);
* packaging observability + fail-fast: the pre-packaging size estimate rejects
  hopeless uploads before minutes of bundling, and the usdUtils unresolved-reference
  warning flood is summarized to one line (task-14);
* the ovrtx auto-install gate: a missing ovrtx venv errors immediately instead of
  pip-downloading a ~2.5 GB wheel mid-render (task-13);
* daemon responsiveness: /health takes no session lock and reports busy state, so a
  daemon mid-render is distinguishable from a dead one (tasks-03/12/14).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import pytest

from conftest import SPRAY, ovrtx_lock_this_interpreter_supports


def _open_spray():
    from pxr import Usd
    stage = Usd.Stage.Open(str(SPRAY.path))
    if not stage:
        pytest.skip("SPRAY sample asset not available")
    return stage


# ── 1. error observability: response body in errors + context in the log ─────────


def test_non_2xx_error_carries_response_body_and_logs_context(caplog):
    import httpx
    from usd_core.render.remote import RemoteRenderBackend

    backend = RemoteRenderBackend("http://gpu:8000")
    resp = httpx.Response(
        500, request=httpx.Request("POST", "http://gpu:8000/render/upload"),
        json={"detail": "OVRTX daemon crashed: LdrColor missing"})

    with caplog.at_level(logging.ERROR, logger="usd_core.render.remote"):
        with pytest.raises(RuntimeError) as ei:
            backend._raise_for_status(resp, 3 * 1024 * 1024, scene="spray_bottle")

    msg = str(ei.value)
    # the agent-visible error names the status, endpoint, scene, and the body
    assert "HTTP 500" in msg
    assert "http://gpu:8000/render/upload" in msg
    assert "spray_bottle" in msg
    assert "LdrColor missing" in msg
    # ... and the daemon log carries the request context (endpoint, size, scene, body)
    logged = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("render/upload" in m and "3.0 MB" in m and "spray_bottle" in m
               and "LdrColor missing" in m for m in logged)


def test_error_body_is_truncated_not_dumped():
    import httpx
    from usd_core.render.remote import RemoteRenderBackend

    backend = RemoteRenderBackend("http://gpu:8000")
    resp = httpx.Response(502, request=httpx.Request("POST", "http://gpu:8000/render"),
                          text="x" * 5000)
    with pytest.raises(RuntimeError, match=r"truncated") as ei:
        backend._raise_for_status(resp, 1024, scene="s")
    assert len(str(ei.value)) < 1000  # ~500 chars of body, not 5000


def test_2xx_and_413_paths():
    import httpx
    from usd_core.render.remote import RemoteRenderBackend

    backend = RemoteRenderBackend("http://gpu:8000")
    ok = httpx.Response(200, request=httpx.Request("POST", "http://gpu:8000/render"))
    backend._raise_for_status(ok, 1024, scene="s")  # no raise

    too_big = httpx.Response(413, request=httpx.Request("POST", "http://gpu:8000/render"),
                             json={"detail": "body exceeds OVRTX_MAX_BODY_BYTES"})
    with pytest.raises(RuntimeError, match="OVRTX_MAX_BODY_BYTES") as ei:
        backend._raise_for_status(too_big, 700 * 1024 * 1024, scene="s")
    assert "body exceeds" in str(ei.value)  # the service's own words survive


def test_render_surfaces_backend_500_body_end_to_end(tmp_path, monkeypatch):
    """The full render() path: an agent no longer sees only httpx's opaque
    "Server error '500 Internal Server Error'" (task-02)."""
    import httpx
    from usd_core.remote_protocol import PROTOCOL_VERSION
    from usd_core.render.remote import RemoteRenderBackend

    stage = _open_spray()

    def fake_get(self, url, **kw):  # version handshake: a compatible backend
        return httpx.Response(200, request=httpx.Request("GET", url),
                              json={"status": "alive",
                                    "protocol_version": PROTOCOL_VERSION})

    def fake_post(self, url, **kw):
        return httpx.Response(
            500, request=httpx.Request("POST", url),
            json={"detail": "render failed: usd_time 0.0 out of range"})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    monkeypatch.setattr(httpx.Client, "post", fake_post)

    backend = RemoteRenderBackend("http://gpu:8000")
    with pytest.raises(RuntimeError) as ei:
        backend.render(stage, ["/spray_bottle"], 64, 64, tmp_path / "out")
    msg = str(ei.value)
    assert "HTTP 500" in msg and "/render/upload" in msg
    assert "usd_time 0.0 out of range" in msg  # the backend's actual complaint
    assert "spray_bottle" in msg  # scene context


# ── 2. packaging: size-estimate fail-fast + warning-flood summarization ──────────


def _saved_scene_stage(tmp_path, pad_bytes: int = 0):
    """A real on-disk scene whose root layer is padded to a controllable size."""
    from pxr import Usd, UsdGeom

    scene = tmp_path / "big_scene.usda"
    stage = Usd.Stage.CreateNew(str(scene))
    UsdGeom.Mesh.Define(stage, "/World/Mesh")
    stage.Save()
    if pad_bytes:
        with scene.open("a") as fh:  # trailing comment: inflates size, still valid USDA
            fh.write("# " + "x" * pad_bytes + "\n")
    return stage


def test_estimate_counts_used_layer_bytes(tmp_path):
    from usd_core.render.remote import _estimate_stage_bytes

    stage = _saved_scene_stage(tmp_path, pad_bytes=2_000_000)
    est, n_layers = _estimate_stage_bytes(stage)
    assert n_layers >= 1
    assert est >= 2_000_000  # the padded root layer is counted


def test_estimate_fail_fast_skips_packaging(tmp_path, monkeypatch):
    """Hopelessly-over-cap scenes fail in milliseconds, before the minutes-long
    bundling step (task-14: a 1.3 GB stage packaged for ~8 minutes only to fail at
    upload). The hard fail-fast requires an estimate over 2× the cap — raw layer
    bytes overstate gzip wire bytes, so merely-over-cap scenes warn and proceed to
    the exact post-gzip check instead (see test_render_packaging_v3_fixes)."""
    import httpx
    from usd_core.remote_protocol import PROTOCOL_VERSION
    from usd_core.render.remote import RemoteRenderBackend

    stage = _saved_scene_stage(tmp_path, pad_bytes=3_000_000)  # > 2× the 1 MB cap

    def fake_get(self, url, **kw):
        return httpx.Response(200, request=httpx.Request("GET", url),
                              json={"protocol_version": PROTOCOL_VERSION})

    def no_packaging(self, stage, work_dir):
        raise AssertionError("packaging must not start for an over-cap scene")

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    monkeypatch.setattr(RemoteRenderBackend, "_package_usdz", no_packaging)

    backend = RemoteRenderBackend("http://gpu:8000", max_upload_mb=1)
    with pytest.raises(RuntimeError, match="remote_max_upload_mb") as ei:
        backend.render(stage, ["/World/Mesh"], 64, 64, tmp_path / "out")
    msg = str(ei.value)
    assert "before packaging" in msg and "big_scene" in msg
    assert "0 for unlimited" in msg  # the override is spelled out


def test_estimate_default_is_unlimited():
    from usd_core.render.remote import RemoteRenderBackend

    # default max_upload_mb=0: even a huge estimate passes (packaging proceeds)
    RemoteRenderBackend("http://gpu:8000")._check_estimate(10**12, 5, "scene")


def test_stderr_warning_flood_summarized_to_one_line(caplog):
    from usd_core.render import remote as remote_mod

    caplog.set_level(logging.DEBUG, logger="usd_core.render.remote")
    cap = remote_mod._StderrCapture()
    with cap:
        for i in range(300):
            os.write(2, f"Warning: unresolved reference @/nope/tex_{i}.jpg@\n".encode())
    assert cap.text.count("\n") == 300
    remote_mod._log_packaging_warnings(cap.text)

    warnings = [r for r in caplog.records
                if r.levelno == logging.WARNING and "suppressed" in r.getMessage()]
    assert len(warnings) == 1  # ONE summary line instead of a 300-line flood
    assert "300" in warnings[0].getMessage()
    assert "tex_0.jpg" in warnings[0].getMessage()  # a first example is kept
    # the full list survives at DEBUG for deep debugging
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("tex_299.jpg" in r.getMessage() for r in debugs)


def test_log_packaging_warnings_silent_when_clean(caplog):
    from usd_core.render.remote import _log_packaging_warnings

    with caplog.at_level(logging.DEBUG, logger="usd_core.render.remote"):
        _log_packaging_warnings("")
        _log_packaging_warnings("\n  \n")
    assert not caplog.records


def test_package_usdz_still_survives_broken_refs_with_capture(tmp_path):
    """The stderr capture must not change packaging behavior: the broken-reference
    fallback (task-08 regression) still produces a usable bundle."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    from usd_core.render.remote import RemoteRenderBackend

    scene = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene))
    UsdGeom.Mesh.Define(stage, "/World/Mesh")
    for i in range(3):
        UsdShade.Material.Define(stage, f"/World/Looks/Broken_{i}")
        sh = UsdShade.Shader.Define(stage, f"/World/Looks/Broken_{i}/Tex")
        sh.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(f"/nonexistent/textures/missing_{i}.jpg"))
    stage.Save()

    work = tmp_path / "work"
    work.mkdir()
    usdz = RemoteRenderBackend._package_usdz(stage, work)
    assert usdz.is_file() and usdz.stat().st_size > 0
    reopened = Usd.Stage.Open(str(usdz))
    assert reopened.GetPrimAtPath("/World/Mesh").IsValid()


# ── 3. ovrtx auto-install gate ────────────────────────────────────────────────────


def test_ovrtx_missing_venv_fails_fast_by_default(tmp_path, monkeypatch):
    """--renderer ovrtx without an installed venv must error immediately, not spend
    ~10 minutes pip-downloading a 2.5 GB wheel mid-render (task-13)."""
    from usd_core.render import ovrtx as ovrtx_mod

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    monkeypatch.setattr(ovrtx_mod, "_env_ovrtx_python", lambda: None)
    # This test covers the default auto-provision gate and its actionable
    # remediation, not the shipped lock's Python/platform breadth.  The
    # production guard deliberately withholds a destructive ``venv --clear``
    # recipe when the selected lock cannot run on this interpreter (currently
    # Python 3.11); that fail-closed contract has dedicated tests.
    monkeypatch.setattr(ovrtx_mod, "_uv_command", lambda: ["/fake/uv"])
    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    calls: list = []
    monkeypatch.setattr(ovrtx_mod.subprocess, "run",
                        lambda *a, **k: calls.append(a))

    with pytest.raises(RuntimeError) as ei:
        ovrtx_mod._ovrtx_python(tmp_path / "ovrtx_venv", auto_install=False)
    msg = str(ei.value)
    assert "2.5 GB" in msg
    assert "render.ovrtx_auto_install" in msg
    assert "--renderer remote" in msg
    assert "pip install" in msg  # ... and at the manual pre-install commands
    assert not calls  # nothing was downloaded


def test_ovrtx_env_install_detected(tmp_path, monkeypatch):
    """ovrtx importable in the current environment (uv sync --extra ovrtx / manual pip
    install) is used directly — no isolated venv, no 2.5 GB download, no error."""
    from usd_core.render import ovrtx as ovrtx_mod

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    monkeypatch.setattr(ovrtx_mod, "_env_ovrtx_python", lambda: "/env/bin/python")
    calls: list = []
    monkeypatch.setattr(ovrtx_mod.subprocess, "run",
                        lambda *a, **k: calls.append(a))

    py = ovrtx_mod._ovrtx_python(tmp_path / "ovrtx_venv", auto_install=False)
    assert py == "/env/bin/python"
    assert not calls  # no provisioning attempted


def test_ovrtx_provisioned_venv_wins_over_env_install(tmp_path, monkeypatch):
    """A pre-provisioned isolated venv (Docker/WU parity) keeps priority: the pinned,
    known-good install is used without even probing the current environment."""
    from usd_core.render import ovrtx as ovrtx_mod

    venv_dir = tmp_path / "ovrtx_venv"
    py = venv_dir / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.touch()
    (venv_dir / ".usd-cli-ovrtx-ready").write_text(
        ovrtx_mod._readiness_marker_body(ovrtx_mod._ovrtx_runtime_lock())
    )
    monkeypatch.setattr(
        ovrtx_mod, "_env_ovrtx_python",
        lambda: pytest.fail("env probe must not run when the venv is ready"))
    assert ovrtx_mod._ovrtx_python(venv_dir) == str(py)


def test_ovrtx_env_probe_survives_broken_subprocess(monkeypatch):
    """The probe treats any subprocess misbehavior as 'not available here'."""
    from usd_core.render import ovrtx as ovrtx_mod

    monkeypatch.setattr(ovrtx_mod.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    assert ovrtx_mod._env_ovrtx_python() is None


def _fake_provision(ovrtx_mod, monkeypatch, venv_dir):
    """Stub the streamed install runner so 'provisioning' just materializes the
    venv python (installs go through _run_logged since the silent-download fix).

    Also stubs uv discovery: these tests exercise the allow/deny GATING around
    provisioning, not the installer, and must behave identically on hosts
    without uv (the CI gate has none — a real `_uv_command()` would fail closed
    there before the gating under test runs). The uv-required fail-closed
    contract has its own tests in test_pr819_review_fixes.py."""
    calls: list[list[str]] = []

    def fake_run(cmd, *, what="", **kw):
        calls.append([str(c) for c in cmd])
        if not what.endswith("(venv)"):
            return
        staged_venv = Path(cmd[-1])
        py = staged_venv / "bin" / "python"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.touch()

    monkeypatch.setattr(ovrtx_mod, "_run_logged", fake_run)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx_mod, "_installed_ovrtx_version",
        lambda py: ovrtx_mod._pinned_ovrtx_version())
    monkeypatch.setattr(ovrtx_mod, "_uv_command", lambda: ["/fake/uv"])
    ovrtx_lock_this_interpreter_supports(monkeypatch, venv_dir.parent)
    return calls


def test_ovrtx_auto_install_flag_allows_provisioning(tmp_path, monkeypatch):
    from usd_core.render import ovrtx as ovrtx_mod

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    venv_dir = tmp_path / "ovrtx_venv"
    calls = _fake_provision(ovrtx_mod, monkeypatch, venv_dir)
    monkeypatch.setattr(ovrtx_mod, "_env_ovrtx_python", lambda: None)

    py = ovrtx_mod._ovrtx_python(venv_dir, auto_install=True)
    assert py == str(venv_dir / "bin" / "python")
    assert any("pip" in c for call in calls for c in call)  # install actually ran
    assert (venv_dir / ".usd-cli-ovrtx-ready").exists()


def test_ovrtx_env_var_overrides_both_ways(tmp_path, monkeypatch):
    from usd_core.render import ovrtx as ovrtx_mod

    # explicit env "1" allows provisioning even when the config gate is off
    venv_a = tmp_path / "venv_a"
    calls = _fake_provision(ovrtx_mod, monkeypatch, venv_a)
    monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "1")
    # assert the provisioned interpreter specifically. A bare truthiness
    # check passes for any resolution -- on a host whose project env
    # already carries the pinned ovrtx, _ovrtx_python returns
    # sys.executable and the branch under test never runs.
    monkeypatch.setattr(ovrtx_mod, "_env_ovrtx_python", lambda: None)
    assert ovrtx_mod._ovrtx_python(venv_a, auto_install=False) == str(
        venv_a / "bin" / "python")
    assert any("pip" in c for call in calls for c in call)

    # explicit env "0" refuses even when the config gate is on
    monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "0")
    with pytest.raises(RuntimeError, match="2.5 GB"):
        ovrtx_mod._ovrtx_python(tmp_path / "venv_b", auto_install=True)


def test_factory_wires_ovrtx_auto_install_from_config():
    from usd_core.config import Config
    from usd_core.render.factory import make_backend

    cfg = Config()
    cfg.render["renderer"] = "ovrtx"
    assert make_backend(cfg)._auto_install is False  # default: gated

    for truthy in (True, "true", "1"):
        cfg.render["ovrtx_auto_install"] = truthy
        assert make_backend(cfg)._auto_install is True, truthy


def test_ensure_ready_allows_install_render_path_does_not(monkeypatch):
    """Service warm-up (ensure_ready) is the sanctioned moment for the one-time
    download; the lazy render-time boot stays gated by config."""
    from usd_core.render import ovrtx as ovrtx_mod

    seen: list[bool] = []

    class FakeDaemon:
        def __init__(self, venv_dir, log_level="warn", auto_install=False,
                     block_install=True):
            seen.append((auto_install, block_install))

        def alive(self):
            return True

    monkeypatch.setattr(ovrtx_mod, "_OvRTXDaemon", FakeDaemon)
    monkeypatch.setattr(ovrtx_mod, "_DAEMON_REGISTRY", {})

    ovrtx_mod.OvRTXRenderBackend().ensure_ready()
    assert seen == [(True, True)]  # warm-up may provision, and BLOCKS

    # daemons are now SHARED per (venv, log level) — reset the registry so each
    # assertion observes a fresh construction's flags
    ovrtx_mod._DAEMON_REGISTRY.clear()
    ovrtx_mod.OvRTXRenderBackend()._ensure_daemon()
    assert seen[-1] == (False, False)  # render-time boot must not

    # config-allowed render-time install runs in the BACKGROUND — a synchronous
    # 2.5 GB download inside a command handler wedged the daemon for ~10 min
    ovrtx_mod._DAEMON_REGISTRY.clear()
    ovrtx_mod.OvRTXRenderBackend(auto_install=True)._ensure_daemon()
    assert seen[-1] == (True, False)


# ── 4. daemon responsiveness: /health is lock-free and reports busy ───────────────


def test_health_answers_and_reports_busy_during_a_long_command(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from usd_core.config import Config
    from usd_core.models import Response
    from usd_server import app as app_mod

    release = threading.Event()
    started = threading.Event()

    def slow_dispatch(session, command, payload):
        started.set()
        assert release.wait(timeout=30), "test never released the slow command"
        return Response(command=command, ok=True)

    monkeypatch.setattr(app_mod, "dispatch", slow_dispatch)

    cfg = Config(project_dir=tmp_path)
    app = app_mod.build_app(cfg, token="t", instance_id="i")
    headers = {"x-usd-cli-token": "t"}

    with TestClient(app) as client:
        worker = threading.Thread(
            target=lambda: client.post(
                "/cmd", json={"command": "info", "payload": {}}, headers=headers))
        worker.start()
        try:
            assert started.wait(timeout=10), "the /cmd request never reached dispatch"
            # the session lock is now held by the in-flight command, yet /health answers
            deadline = time.time() + 10
            health = None
            while time.time() < deadline:
                health = client.get("/health", headers=headers).json()
                if health.get("busy"):
                    break
                time.sleep(0.05)
            assert health and health["ok"]
            assert health["busy"] is True
            assert health["current_command"] == "info"
            assert health["busy_seconds"] >= 0.0
            assert app_mod._IN_FLIGHT[0] == 1  # what the idle watcher checks
        finally:
            release.set()
            worker.join(timeout=30)

        health = client.get("/health", headers=headers).json()
        assert health["busy"] is False and health["current_command"] is None
        assert app_mod._IN_FLIGHT[0] == 0


def test_in_flight_counter_resets_after_a_failing_command(tmp_path, monkeypatch):
    """A command that blows up must not leave the daemon looking busy forever
    (that would also disable the idle auto-shutdown)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from usd_core.config import Config
    from usd_server import app as app_mod

    def exploding_dispatch(session, command, payload):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_mod, "dispatch", exploding_dispatch)
    app = app_mod.build_app(Config(project_dir=tmp_path), token="t", instance_id="i")
    headers = {"x-usd-cli-token": "t"}

    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/cmd", json={"command": "info", "payload": {}}, headers=headers)
        assert app_mod._IN_FLIGHT[0] == 0
        health = client.get("/health", headers=headers).json()
        assert health["busy"] is False


def test_ovrtx_probes_run_isolated(tmp_path, monkeypatch):
    """Every readiness probe runs python -I with PYTHONPATH stripped: a module
    shadowing ovrtx/numpy/PIL in the CWD, user site-packages, or PYTHONPATH
    must not satisfy readiness and then fail where the daemon actually runs."""
    from usd_core.render import ovrtx as ovrtx_mod

    monkeypatch.setenv("PYTHONPATH", "/somewhere/with/ovrtx")
    seen: list = []

    def fake_run(cmd, **kwargs):
        seen.append((list(cmd), kwargs.get("env")))

        class _Proc:
            returncode = 0
            stdout = "0.0.0\n"

        return _Proc()

    monkeypatch.setattr(ovrtx_mod.subprocess, "run", fake_run)
    ovrtx_mod._env_ovrtx_python()
    ovrtx_mod._installed_ovrtx_version(tmp_path / "python")
    # The WU-managed shared-runtime probe must use the same isolation as the
    # daemon before it admits a pre-provisioned runtime.
    shared_dir = tmp_path / "wu_venv"
    (shared_dir / "bin").mkdir(parents=True)
    (shared_dir / "bin" / "python").touch()
    (shared_dir / ovrtx_mod._WU_MANAGED_MARKER_NAME).touch()
    ovrtx_mod._shared_runtime_matching_pin(shared_dir)
    assert len(seen) == 3
    for cmd, env in seen:
        assert "-I" in cmd
        assert env is not None and "PYTHONPATH" not in env
