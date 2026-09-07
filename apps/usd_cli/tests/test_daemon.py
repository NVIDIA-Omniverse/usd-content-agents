# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Daemon lifecycle (PR-9), the response envelope contract, and stub/error behavior.

These exercise the transport and CLI plumbing rather than the engine: auto-start,
`server status/stop`, the `--json` envelope shape, the no-daemon stub, and the clean
"not implemented" path for scaffolded commands.
"""

from __future__ import annotations

import pytest

from conftest import SPRAY, _make_project


@pytest.fixture
def fresh_project(tmp_path):
    """A daemon that isn't shared with other tests, so it's safe to stop mid-test."""
    proj = _make_project(tmp_path)
    yield proj
    proj.stop()


def test_help_lists_core_commands(project):
    res = project.cli("--help", expect_ok=True)
    for command in ("snapshot", "render", "camera", "resolve", "open"):
        assert command in res.out


def test_version_reports_backend(project):
    res = project.cli("--version", expect_ok=True)
    assert "usd-cli 0.0.1" in res.out
    assert "engine" in res.out


def test_json_envelope_has_stable_shape(project):
    project.open(SPRAY)
    env = project.cli("snapshot", json=True, expect_ok=True).json()
    # round 8: empty collections are DROPPED from the wire (compact envelope);
    # the stable core is command/ok/schema_version, everything else is optional
    assert set(env) >= {"command", "ok", "schema_version"}
    assert set(env) <= {"command", "ok", "schema_version", "summary", "data",
                        "artifacts", "issues"}
    assert env["schema_version"] == "1"
    assert env["command"] == "snapshot"
    assert isinstance(env.get("artifacts", []), list)
    assert isinstance(env.get("issues", []), list)


@pytest.mark.parametrize(
    "environment_key",
    ["USD_CLI_NO_DAEMON", "OV_NO_DAEMON", "3DSC_NO_DAEMON"],
)
def test_no_daemon_mode_fails_closed_after_echoing_parsed_request(
    project,
    environment_key,
):
    # Disabling the daemon may expose parsing diagnostics, but it must never look
    # like a successful low-level scene operation.
    env = project.cli(
        "snapshot", json=True, expect_ok=False, extra_env={environment_key: "1"}
    ).json()
    assert env["ok"] is False
    assert env["summary"]["stub"] is True
    assert env["summary"]["error_type"] == "daemon-disabled"
    assert env["data"]["would_send"]["command"] == "snapshot"


def test_daemon_lifecycle_start_status_stop(fresh_project):
    # No command sent yet → no daemon.
    assert "not running" in fresh_project.cli("server", "status").out

    # A real command auto-starts the per-project daemon.
    fresh_project.open(SPRAY)
    running = fresh_project.cli("server", "status")
    assert "running" in running.out
    assert SPRAY.relpath.split("/")[-1] in running.out  # live stage is reported

    assert "stopped" in fresh_project.cli("server", "stop").out
    assert "not running" in fresh_project.cli("server", "status").out


def test_externally_owned_daemon_refuses_implicit_autostart(
    tmp_path,
    monkeypatch,
):
    from usd_core.config import Config
    from usd_cli import daemon

    config = Config(project_dir=tmp_path)
    config.server["lifecycle_owner"] = "external"
    spawned = False

    def unexpected_spawn(_config):
        nonlocal spawned
        spawned = True
        raise AssertionError("externally owned daemon must not auto-start")

    monkeypatch.delenv(daemon.EXTERNAL_LIFECYCLE_BOOTSTRAP_ENV, raising=False)
    monkeypatch.setattr(daemon, "discover", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(daemon, "_spawn", unexpected_spawn)

    with pytest.raises(
        daemon.DaemonStopRefused,
        match="cannot auto-start a replacement",
    ):
        daemon.ensure_running(config)

    assert spawned is False


def test_reap_orphans_prunes_only_dead_records_and_cleans_project_litter(tmp_path):
    """A mutable ledger cannot authorize signals; only dead records are pruned."""
    import subprocess
    import sys

    from usd_core.config import Config
    from usd_cli import daemon

    state = tmp_path / ".usd-cli"
    state.mkdir()
    sleeper = [sys._base_executable, "-c", "import time;time.sleep(60)"]
    ours = subprocess.Popen(sleeper)
    theirs = subprocess.Popen(sleeper)  # a different project's daemon — must survive
    try:
        # New pid:start-token ledger format (round-4: pid reuse safety); the second
        # line is an already-dead pid whose recorded birth time can't match anything.
        tok = daemon._proc_start_token(ours.pid)
        (state / "daemon.pids").write_text(f"{ours.pid}:{tok}\n999999999:t12345\n")
        litter = tmp_path / ".fuse_hidden0001"
        litter.write_bytes(b"leftover")

        killed = daemon._reap_orphans(Config(project_dir=tmp_path), keep_pid=None)

        assert killed == 0
        assert ours.poll() is None  # ledger-only ownership never authorizes a signal
        assert theirs.poll() is None  # the unrelated daemon is untouched
        assert not litter.exists()  # .fuse_hidden litter cleaned
        assert (state / "daemon.pids").read_text() == f"{ours.pid}:{tok}\n"
    finally:
        ours.kill()
        theirs.kill()


def test_explicit_timeout_is_a_hard_ceiling_even_for_renders():
    """Default --timeout lets slow commands (render/physics) use the generous floor, but an
    explicit --timeout bounds total wall time for every command."""
    from usd_cli import client
    from usd_cli.state import G

    saved = (G.timeout, G.timeout_explicit)
    try:
        # default: render/physics get the slow floor, quick commands the snappy default
        G.timeout, G.timeout_explicit = 30.0, False
        assert client._effective_timeout("render") == client._SLOW_TIMEOUT_S
        assert client._effective_timeout("render-probe") == client._SLOW_TIMEOUT_S
        assert client._effective_timeout("info") == 30.0
        # explicit --timeout: a hard ceiling everywhere, including renders
        G.timeout, G.timeout_explicit = 45.0, True
        assert client._effective_timeout("render") == 45.0
        assert client._effective_timeout("info") == 45.0
    finally:
        G.timeout, G.timeout_explicit = saved


def test_eval_is_not_exposed(project):
    """`eval` (the code-as-action escape hatch) is not yet implemented, so it's commented
    out of the CLI — invoking it is an unknown command (usage error), not a crash."""
    project.open(SPRAY)
    res = project.cli("eval", "n1.tx += 1", expect_ok=False)
    assert res.code == 2  # typer: no such command


@pytest.mark.parametrize("argv", [
    ["transform", "@n5", "--tx=+3"],
    ["describe"],
    ["info"],
], ids=["transform", "describe", "info"])
def test_manipulation_and_query_commands_work(project, argv):
    """transform/describe/info are implemented now and return ok envelopes."""
    project.open(SPRAY)
    env = project.cli(*argv, json=True, expect_ok=True).json()
    assert env["ok"] is True


def test_save_writes_to_an_explicit_path(project):
    """`save <path>` exports the stage without touching the source asset."""
    project.open(SPRAY)
    out = project.root / "saved.usda"
    env = project.cli("save", str(out), json=True, expect_ok=True).json()
    assert env["ok"] is True
    assert out.exists() and out.stat().st_size > 0


def test_in_place_save_is_atomic_and_leaves_no_tmp(fresh_project):
    """In-place `save` (export to temp sibling + atomic replace) keeps a valid, re-openable
    crate and leaves no .dsctmp litter — the old direct crate rewrite could truncate."""
    fresh_project.open(SPRAY)
    work = fresh_project.root / "work.usdc"
    fresh_project.cli("save", str(work), json=True, expect_ok=True)
    fresh_project.cli("open", str(work), "--force-reload", json=True, expect_ok=True)

    env = fresh_project.cli("save", json=True, expect_ok=True).json()  # in-place
    assert env["ok"] is True
    assert work.exists() and work.stat().st_size > 0
    assert not list(fresh_project.root.glob("*.dsctmp*"))  # no leftover temp files
    # the saved crate re-opens cleanly (no 'Corrupt asset' truncation)
    fresh_project.cli("open", str(work), "--force-reload", json=True, expect_ok=True)


def test_sublayers_lists_and_drops_dead_arcs(fresh_project):
    """`sublayers` reports resolvability; `--drop-dead` strips arcs that don't resolve."""
    root = fresh_project.root / "layered.usda"
    root.write_text(
        '#usda 1.0\n'
        '(\n'
        '    subLayers = [\n'
        '        @./ghost_layer.usda@\n'
        '    ]\n'
        ')\n'
        'def Xform "World" {}\n'
    )
    fresh_project.cli("open", str(root), "--force-reload", json=True, expect_ok=True)

    env = fresh_project.cli("sublayers", json=True, expect_ok=True).json()
    assert env["summary"]["total"] == 1
    assert env["summary"]["dead"] == 1
    assert env["summary"]["dropped"] == 0  # listing only, nothing removed

    env = fresh_project.cli("sublayers", "--drop-dead", json=True, expect_ok=True).json()
    assert env["summary"]["dropped"] == 1
    assert env["data"]["dropped"] == ["./ghost_layer.usda"]

    # after dropping, none remain
    env = fresh_project.cli("sublayers", json=True, expect_ok=True).json()
    assert env["summary"]["total"] == 0
