# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wave-3 CLI/UX fixes from the 2026-07-09/10 benchmark round.

Covers: material-audit JSON payload capping on huge scenes, bulk-selection excluding
the --under anchor, render-stem overflow with hundreds of --exclude values, the
explicit physics.apply authored-APIs text, clear transport errors for slow commands
against an unresponsive daemon, and temp-litter cleanup after failed saves.

All GPU-free: direct Session/function calls, plus a raw-socket stub server that
accepts connections but never responds (for the client transport tests).
"""
from __future__ import annotations

import socket
import threading
import time

import pytest
from pxr import Usd, UsdGeom, UsdShade
from typer.testing import CliRunner

from usd_core.audit import _capped, material_audit
from usd_core.config import Config
from usd_core.models import Response
from usd_core.selector import select_prims
from usd_core.session import Session


def test_cli_session_env_default_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from usd_cli.main import main_callback
    from usd_cli.state import G

    previous = (G.json, G.quiet, G.server, G.session, G.timeout, G.timeout_explicit)
    monkeypatch.setenv("USD_CLI_SESSION", "workflow-materials-live")
    monkeypatch.setenv("CONTENT_WORKFLOW_USD_CLI_SESSION_ID", "attested-parent")
    try:
        main_callback(
            version=False,
            json_=False,
            quiet=False,
            server=None,
            session=None,
            timeout=-1.0,
        )
        assert G.session == "attested-parent"

        main_callback(
            version=False,
            json_=False,
            quiet=False,
            server=None,
            session="explicit-session",
            timeout=-1.0,
        )
        assert G.session == "explicit-session"
    finally:
        (G.json, G.quiet, G.server, G.session, G.timeout, G.timeout_explicit) = previous


def test_release_session_requires_and_dispatches_explicit_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from usd_cli import main

    calls: list[tuple[str, dict]] = []

    def fake_dispatch(command: str, payload: dict) -> Response:
        calls.append((command, payload))
        return Response(command=command)

    monkeypatch.setattr(main, "dispatch", fake_dispatch)
    missing = CliRunner().invoke(main.app, ["server", "release-session"])
    assert missing.exit_code != 0
    assert "--name" in missing.output
    assert calls == []

    released = CliRunner().invoke(
        main.app,
        [
            "--session",
            "ambient-wrong-session",
            "server",
            "release-session",
            "--name",
            "workflow-item-1",
        ],
    )
    assert released.exit_code == 0
    assert calls == [
        ("server.release-session", {"name": "workflow-item-1"})
    ]


# ── helpers ─────────────────────────────────────────────────────────────────────


def _big_unbound_stage(n: int = 250):
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for i in range(n):
        UsdGeom.Mesh.Define(stage, f"/World/mesh_{i:04d}")
    return stage


def _session_on(tmp_path, stage, name="scene.usda") -> Session:
    path = tmp_path / name
    stage.GetRootLayer().Export(str(path))
    s = Session(Config(project_dir=tmp_path))
    s.open_stage(str(path))
    return s


# ── #1 material audit JSON cap ──────────────────────────────────────────────────


def test_audit_capped_helper():
    small = [f"/p{i}" for i in range(200)]
    out, omitted = _capped(small)
    assert out == small and omitted == 0
    big = [f"/p{i}" for i in range(201)]
    out, omitted = _capped(big)
    assert omitted == 151
    assert len(out) == 50
    assert out == big[:50]


def test_audit_caps_per_prim_lists_on_large_scene():
    report = material_audit(_big_unbound_stage(250), effective=True)
    # aggregate counts stay exact
    assert report["counts"]["renderables"] == 250
    assert report["counts"]["unbound"] == 250
    # per-prim listings are sliced to 50; omitted counts are separate fields so
    # lists stay schema-homogeneous (review finding 22)
    assert report["truncated"] is True
    assert len(report["unbound_paths"]) == 50
    assert report["unbound_paths_omitted"] == 200
    assert all(isinstance(x, str) for x in report["unbound_paths"])
    assert len(report["renderables"]) == 50
    assert report["renderables_omitted"] == 200
    assert all(isinstance(x, dict) for x in report["renderables"])


def test_audit_small_scene_not_truncated():
    report = material_audit(_big_unbound_stage(5), effective=True)
    assert "truncated" not in report
    assert len(report["unbound_paths"]) == 5
    assert len(report["renderables"]) == 5


def test_session_audit_text_survives_capping(tmp_path):
    s = _session_on(tmp_path, _big_unbound_stage(250))
    resp = s.material_audit(effective=True)
    assert resp.ok
    assert resp.data["truncated"] is True
    assert "renderables=250" in resp.data["text"]  # text summary keeps exact counts
    assert resp.summary["renderables"] == 250


# ── #2 bulk selection excludes the --under anchor ───────────────────────────────


def _assembly_stage():
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    # anchor is itself a Gprim whose name matches the rule (the wu incident: bulk
    # `--under @n3 --where 'name~=P*'` bound the assembly root P5Z00040937 too)
    UsdGeom.Mesh.Define(stage, "/World/P5Z00040937")
    UsdGeom.Mesh.Define(stage, "/World/P5Z00040937/P5Z001")
    UsdGeom.Mesh.Define(stage, "/World/P5Z00040937/P5Z002")
    UsdGeom.Mesh.Define(stage, "/World/P5Z00040937/other")
    UsdGeom.Mesh.Define(stage, "/World/P9_outside")
    return stage


def test_select_prims_under_excludes_anchor():
    stage = _assembly_stage()
    got = {p.GetPath().pathString
           for p in select_prims(stage, where=["name~=P*"], under="/World/P5Z00040937")}
    assert got == {"/World/P5Z00040937/P5Z001", "/World/P5Z00040937/P5Z002"}


def test_select_prims_without_under_still_matches_everything():
    stage = _assembly_stage()
    got = {p.GetName() for p in select_prims(stage, where=["name~=P*"])}
    assert got == {"P5Z00040937", "P5Z001", "P5Z002", "P9_outside"}


# ── #3 render stem overflow with many --exclude values ──────────────────────────


def test_render_stem_short_excludes_stay_verbatim():
    s = Session(Config())
    stem = s._render_stem("/World/cam", 640, 480, "fast", exclude=["@n1", "@n2"])
    assert "excl-n1-n2" in stem


def test_render_stem_long_excludes_collapse_to_count_and_sha1():
    s = Session(Config())
    excludes = [f"/World/Assembly/P5Z0004{i:04d}" for i in range(300)]
    stem = s._render_stem("/World/cam", 640, 480, "fast", exclude=excludes)
    assert len(stem) < 100  # no more [Errno 74] filename overflow
    assert "excl-300x-" in stem
    digest = stem.split("excl-300x-")[1].split("__")[0]
    assert len(digest) == 8
    assert all(c in "0123456789abcdef" for c in digest)
    # deterministic for the same exclusion set
    assert stem == s._render_stem("/World/cam", 640, 480, "fast", exclude=excludes)


# ── #4 physics apply reports authored APIs in text output ───────────────────────


def test_physics_authored_line_shapes():
    rec = {"scene": ["/physicsScene"], "rigid_body": ["/Robot"],
           "collision": [f"/Robot/c{i}" for i in range(33)], "material": [],
           "authored_apis": {"/Robot": ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]}}
    line = Session._physics_authored_line(rec)
    assert line == "authored: /Robot (PhysicsRigidBodyAPI+PhysicsMassAPI), 33 colliders updated"
    assert Session._physics_authored_line(
        {"rigid_body": [], "collision": [], "material": [], "authored_apis": {}}) is None
    line = Session._physics_authored_line(
        {"rigid_body": [], "collision": ["/a"], "material": ["/Looks/wheel_mat"],
         "authored_apis": {}})
    assert line == "authored: 1 collider updated, 1 physics material(s) updated"


def test_physics_apply_text_reports_authored(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/Body")
    s = _session_on(tmp_path, stage, "prop.usda")
    resp = s.physics_apply(
        operations={"rigid_bodies": [{"path": "/World/Body", "mass": 2.0}]}
    )
    assert resp.ok
    assert resp.summary == {"operations": 1}
    text = resp.data["text"]
    assert text == "authored: /World/Body (PhysicsRigidBodyAPI+PhysicsMassAPI)"


# ── #5 slow commands fail loudly against an unresponsive daemon ─────────────────


@pytest.fixture()
def dead_server():
    """A server that accepts TCP connections but never sends a byte back."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    conns: list[socket.socket] = []
    stop = threading.Event()

    def loop():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                c, _ = srv.accept()
                conns.append(c)  # hold it open, never respond
            except TimeoutError:
                continue
            except OSError:
                break

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.getsockname()[1]}"
    stop.set()
    for c in conns:
        c.close()
    srv.close()
    t.join(timeout=2)


@pytest.fixture()
def cli_globals(dead_server):
    from usd_cli.state import G
    saved = (G.json, G.quiet, G.server, G.session, G.timeout, G.timeout_explicit)
    G.json, G.quiet, G.session = False, False, None
    G.server = dead_server
    G.timeout, G.timeout_explicit = 2.0, False
    yield G
    (G.json, G.quiet, G.server, G.session, G.timeout, G.timeout_explicit) = saved


@pytest.mark.parametrize("command", ["render", "save", "snapshot", "render-frames",
                                     "physics.simulate"])
def test_dispatch_reports_transport_error_for_every_command(cli_globals, command):
    from usd_cli.client import dispatch
    t0 = time.monotonic()
    resp = dispatch(command, {})
    elapsed = time.monotonic() - t0
    assert resp.ok is False
    assert resp.summary["error_type"] == "transport"
    assert any(i.severity == "error" and "unreachable" in i.message for i in resp.issues)
    # every command fails within its snappy budget — no silent 1800s hang for
    # render/save while snapshot errors (the wu benchmark's silent-failure split)
    assert elapsed < 15.0


def test_transport_error_exits_nonzero_with_stderr_message(cli_globals, capsys):
    from usd_cli.client import dispatch
    from usd_cli.output import EXIT_UNREACHABLE, emit
    code = emit(dispatch("render", {}))
    assert code == EXIT_UNREACHABLE
    err = capsys.readouterr().err
    assert "error:" in err and "unreachable" in err


def test_slow_command_probe_skips_reachability_check_when_daemon_answers(
        cli_globals, monkeypatch):
    # A daemon that answers /live (busy but alive) must NOT be reported unreachable
    # by the probe — the long read ceiling is then legitimate queueing.
    from usd_cli import client

    assert client._probe_alive(cli_globals.server, 1.0) is not None  # dead: probe fails

    monkeypatch.setattr(client, "_probe_alive", lambda base, timeout_s: None)  # alive
    monkeypatch.setattr(client, "_SLOW_TIMEOUT_S", 3.0)  # keep the long read short here
    resp = client.dispatch("render", {})
    # probe passed, so the failure comes from the actual POST read timeout —
    # still a loud transport error, never a silent empty success
    assert resp.ok is False
    assert resp.summary["error_type"] == "transport"


# ── #6 failed atomic saves clean their temp litter ──────────────────────────────


def test_cleanup_save_temps_removes_only_owned_mkstemp_litter(tmp_path):
    target = tmp_path / "recording.usda"
    target.write_text("#usda 1.0\n")
    keep_word = tmp_path / "recording.backup"          # 6 lowercase letters: a real file
    keep_word.write_text("x")
    keep_lookalike = tmp_path / "recording.2024Q1"     # user file that MATCHES the temp
    keep_lookalike.write_text("x")                     # pattern, but pre-dates the save
    keep_other_live = tmp_path / "recording.Zz99Aa"    # e.g. another session's live temp
    keep_other_live.write_text("x")

    before = Session._dir_snapshot(str(target))        # ← the save attempt starts here
    litter_stem = tmp_path / "recording.VxRCU3"        # stem-based TfSafeOutputFile temp
    litter_name = tmp_path / "recording.usda.Ab12Cd"   # full-name-based temp
    litter_stem.write_text("x")
    litter_name.write_text("x")
    keep_other = tmp_path / "other.Ab12Cd"             # new, but a different stem
    keep_other.write_text("x")

    Session._cleanup_save_temps(str(target), before)

    assert not litter_stem.exists()
    assert not litter_name.exists()
    assert target.exists()
    assert keep_word.exists()
    assert keep_other.exists()
    # ownership, not name+age: files present BEFORE the save attempt are never touched,
    # even when recent and pattern-matching (the old heuristic unlinked both of these)
    assert keep_lookalike.exists()
    assert keep_other_live.exists()


def test_cleanup_save_temps_tolerates_bad_targets(tmp_path):
    Session._cleanup_save_temps(None, None)  # no target: no-op
    missing = str(tmp_path / "missing" / "deep" / "file.usda")
    Session._cleanup_save_temps(missing, Session._dir_snapshot(missing))


def test_failed_save_triggers_cleanup(tmp_path, monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/cube")
    s = _session_on(tmp_path, stage, "prop.usda")

    target = tmp_path / "out.usda"
    litter = tmp_path / "out.VxRCU3"

    def boom(root_layer, dest):
        litter.write_text("half-written")  # what a failed TfSafeOutputFile leaves
        raise RuntimeError("disk full")

    monkeypatch.setattr(Session, "_atomic_export", staticmethod(boom))
    resp = s.save(str(target))
    assert resp.ok is False
    assert not litter.exists()


# ── regression: audit markers never break the text renderer ─────────────────────


def test_audit_invalid_binding_marker_renders_in_text(tmp_path):
    stage = _big_unbound_stage(5)
    # author >200 invalid direct bindings so the invalid_bindings list gets capped
    for i in range(201):
        prim = UsdGeom.Mesh.Define(stage, f"/World/bad_{i:04d}").GetPrim()
        rel = prim.CreateRelationship(UsdShade.Tokens.materialBinding)
        rel.SetTargets(["/Looks/DoesNotExist"])
    s = _session_on(tmp_path, stage, "invalid.usda")
    resp = s.material_audit()
    assert resp.ok is False  # invalid bindings fail the audit
    assert resp.data["truncated"] is True
    assert resp.data["invalid_bindings_omitted"] == 151
    # the text summary reflects the exact total even though the list is sliced
    assert "201" in resp.data["text"] or "invalid" in resp.data["text"].lower()
