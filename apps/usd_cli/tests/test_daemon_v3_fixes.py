# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the round-3 daemon fixes.

Multi-agent benchmark runs (several sub-agents hammering one project daemon) exposed
three failure modes: (1) `--session NAME` did not isolate anything — the daemon held ONE
global Session, so a sub-agent's `open` clobbered the main agent's stage; (2) cleanup
trusted mutable PID evidence without authenticating project and health identity; (3)
`server status` reported a
defunct (zombie) daemon as "busy … likely mid-operation" because `kill -0` succeeds on
zombies. These tests pin the fixes: a per-session registry with per-session locks in
`usd_server.app`, fail-closed lifecycle ownership in `usd_cli.daemon.stop`, and real
process state detection in `usd_cli.daemon._proc_state`.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import CUBE, SPRAY
from fastapi.testclient import TestClient
from usd_cli import daemon as cli_daemon
from usd_core.config import Config
from usd_core.models import Response
from usd_server import app as server_app

HEADERS = {"x-usd-cli-token": "t"}


def _app(tmp_path):
    return server_app.build_app(
        Config(
            project_dir=tmp_path,
            server={
                "allowed_roots": [
                    str(tmp_path),
                    str(CUBE.path.parent),
                    str(SPRAY.path.parent),
                ]
            },
        ),
        token="t",
        instance_id="i",
    )


def _cmd(
    http, command: str, payload: dict | None = None, session: str | None = None
) -> dict:
    body: dict = {"command": command, "payload": payload or {}}
    if session is not None:
        body["session"] = session
    res = http.post("/cmd", json=body, headers=HEADERS)
    assert res.status_code == 200, res.text
    return res.json()


# ── issue 1: per-session isolation ────────────────────────────────────────────────
def test_named_sessions_isolate_stages(tmp_path):
    """Two sub-agents on different --session names get INDEPENDENT stages — neither
    clobbers the other's, nor the un-sessioned (default) session's."""
    with TestClient(_app(tmp_path)) as http:
        assert (
            _cmd(http, "open", {"file": str(CUBE.path)}, session="main")["ok"] is True
        )
        assert (
            _cmd(http, "open", {"file": str(SPRAY.path)}, session="asset-audit")["ok"]
            is True
        )

        # Each named session still sees ITS stage after the other session's `open`.
        assert _cmd(http, "info", session="main")["data"]["stage"] == str(CUBE.path)
        assert _cmd(http, "info", session="asset-audit")["data"]["stage"] == str(
            SPRAY.path
        )

        # The default session never had a stage opened: no bleed-through to
        # un-sessioned commands (`--json info` used to return another agent's stage).
        assert _cmd(http, "info")["ok"] is False

        # /health reports every session and its stage; the legacy top-level `stage`
        # field mirrors the default session.
        health = http.get("/health", headers=HEADERS).json()
        assert set(health["sessions"]) >= {"default", "main", "asset-audit"}
        assert health["sessions"]["main"]["stage"] == str(CUBE.path)
        assert health["sessions"]["asset-audit"]["stage"] == str(SPRAY.path)
        assert health["sessions"]["default"]["stage"] is None
        assert health["stage"] is None


def test_unsessioned_requests_keep_the_single_session_contract(tmp_path):
    """Requests without a session name (and with a blank one) share one default session,
    preserving the original single-session behavior and /health `stage` field."""
    with TestClient(_app(tmp_path)) as http:
        assert _cmd(http, "open", {"file": str(CUBE.path)})["ok"] is True
        assert _cmd(http, "info")["data"]["stage"] == str(CUBE.path)
        assert _cmd(http, "info", session="")["data"]["stage"] == str(CUBE.path)
        assert http.get("/health", headers=HEADERS).json()["stage"] == str(CUBE.path)


def test_session_count_is_capped_with_a_clear_error(tmp_path):
    with TestClient(_app(tmp_path)) as http:
        for i in range(server_app.MAX_SESSIONS - 1):  # + the default = MAX_SESSIONS
            _cmd(http, "info", session=f"s{i}")  # creates the session (ok not required)
        over = _cmd(http, "info", session="one-too-many")
        assert over["ok"] is False
        assert (
            f"session limit reached ({server_app.MAX_SESSIONS})"
            in over["issues"][0]["message"]
        )
        # existing sessions keep working at the cap
        again = _cmd(http, "info", session="s0")
        assert "session limit" not in again["issues"][0]["message"]


def test_sequential_named_sessions_release_daemon_capacity(tmp_path):
    """More than one registry's worth of sequential work must not require restart."""

    with TestClient(_app(tmp_path)) as http:
        for index in range(server_app.MAX_SESSIONS * 2):
            name = f"item-{index}"
            _cmd(http, "info", session=name)
            released = _cmd(
                http,
                "server.release-session",
                {"name": name},
                session="ambient-wrong-session",
            )
            assert released["ok"] is True
            assert released["summary"] == {"session": name, "released": True}

        health = http.get("/health", headers=HEADERS).json()
        assert set(health["sessions"]) == {server_app.DEFAULT_SESSION}

        repeated = _cmd(
            http,
            "server.release-session",
            {"name": "item-0"},
        )
        assert repeated["ok"] is True
        assert repeated["summary"]["released"] is False

        missing = _cmd(http, "server.release-session")
        assert missing["ok"] is False
        assert "explicit session name is required" in missing["issues"][0]["message"]

        default = _cmd(http, "server.release-session", {"name": "default"})
        assert default["ok"] is False
        assert "default daemon session cannot be released" in default["issues"][0]["message"]


def test_busy_named_session_cannot_be_released(monkeypatch, tmp_path):
    started = threading.Event()
    finish = threading.Event()

    class FakeSession:
        def __init__(self, _config, name="default"):
            self.name = name
            self._stage_path = None

        def info(self):
            if self.name == "work":
                started.set()
                finish.wait(timeout=30)
            return Response(command="info")

    monkeypatch.setattr(server_app, "Session", FakeSession)
    with TestClient(_app(tmp_path)) as http:
        worker = threading.Thread(target=lambda: _cmd(http, "info", session="work"))
        worker.start()
        try:
            assert started.wait(timeout=10)
            rejected = _cmd(
                http,
                "server.release-session",
                {"name": "work"},
            )
            assert rejected["ok"] is False
            assert rejected["summary"]["busy"] is True
        finally:
            finish.set()
            worker.join(timeout=30)

        assert not worker.is_alive()
        released = _cmd(
            http,
            "server.release-session",
            {"name": "work"},
        )
        assert released["ok"] is True
        assert released["summary"]["released"] is True


@pytest.mark.parametrize("name", ["../escape", "a b", "-leading", "x" * 65])
def test_invalid_session_names_are_rejected(tmp_path, name):
    with TestClient(_app(tmp_path)) as http:
        env = _cmd(http, "info", session=name)
        assert env["ok"] is False
        assert "invalid session name" in env["issues"][0]["message"]


def test_concurrent_commands_in_different_sessions_overlap(monkeypatch, tmp_path):
    """Per-session locks: requests to DIFFERENT sessions proceed concurrently (the old
    global lock forced peak concurrency 1), and /health reports each busy session."""
    release = threading.Event()
    gate = threading.Lock()
    active, peak = [0], [0]

    class FakeSession:
        def __init__(self, _config):
            self._stage_path = None

        def info(self):
            with gate:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            try:
                release.wait(timeout=30)
            finally:
                with gate:
                    active[0] -= 1
            return Response(command="info")

    monkeypatch.setattr(server_app, "Session", FakeSession)
    with TestClient(_app(tmp_path)) as http:
        threads = [
            threading.Thread(
                target=lambda s=s: http.post(
                    "/cmd",
                    json={"command": "info", "payload": {}, "session": s},
                    headers=HEADERS,
                )
            )
            for s in ("alpha", "beta")
        ]
        for thread in threads:
            thread.start()
        try:
            deadline = time.time() + 10
            while peak[0] < 2 and time.time() < deadline:
                time.sleep(0.01)
            assert peak[0] == 2, "different sessions never ran concurrently"

            health = http.get("/health", headers=HEADERS).json()
            assert health["busy"] is True
            busy = {name for name, entry in health["sessions"].items() if entry["busy"]}
            assert busy == {"alpha", "beta"}
            assert health["sessions"]["alpha"]["current_command"] == "info"
            assert health["sessions"]["alpha"]["busy_seconds"] >= 0.0
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=30)

        health = http.get("/health", headers=HEADERS).json()
        assert health["busy"] is False
        assert not any(entry["busy"] for entry in health["sessions"].values())
        assert server_app._IN_FLIGHT[0] == 0  # what the idle watcher checks


def test_batch_requires_a_single_session(tmp_path):
    """A /batch stays atomic w.r.t. one session; mixing session names is a 400."""
    with TestClient(_app(tmp_path)) as http:
        mixed = [
            {"command": "info", "payload": {}, "session": "a"},
            {"command": "info", "payload": {}, "session": "b"},
        ]
        res = http.post("/batch", json=mixed, headers=HEADERS)
        assert res.status_code == 400
        assert "single session" in res.json()["detail"]

        same = [{"command": "info", "payload": {}, "session": "a"}] * 2
        res = http.post("/batch", json=same, headers=HEADERS)
        assert res.status_code == 200
        assert len(res.json()) == 2


# ── issue 2: mutable state alone must never authorize a process signal ────────────
def test_stop_refuses_alive_unresponsive_recorded_daemon(monkeypatch, tmp_path):
    """A copied ledger is not authorization: live stop requires matching health."""
    cfg = Config(project_dir=tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "pid": 4242,
        "host": "127.0.0.1",
        "port": 65535,
        "token": "tok",
        "instance_id": "inst",
        "project_id": cli_daemon._project_id(cfg),
        "process_start_token": "t-owned",
    }
    cfg.server_state_path.write_text(json.dumps(state))
    (cfg.state_dir / "daemon.pids").write_text("4242:t-owned\n")
    monkeypatch.setattr(cli_daemon, "_proc_start_token", lambda _pid: "t-owned")
    monkeypatch.setattr(cli_daemon, "_proc_state", lambda _pid: "alive")
    monkeypatch.setattr(cli_daemon, "_health", lambda *_a, **_k: None)
    signaled: list[int] = []
    monkeypatch.setattr(
        cli_daemon,
        "_terminate",
        lambda pid, **_kwargs: signaled.append(pid) or True,
    )

    with pytest.raises(cli_daemon.DaemonStopRefused, match="unauthenticated"):
        cli_daemon.stop(cfg)

    assert not signaled
    assert json.loads(cfg.server_state_path.read_text()) == state
    assert (cfg.state_dir / "daemon.pids").read_text() == "4242:t-owned\n"


def test_stop_still_never_signals_an_unrecorded_unverified_pid(monkeypatch, tmp_path):
    """Malformed/unrecorded state is preserved and never signaled."""
    cfg = Config(project_dir=tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.server_state_path.write_text(
        json.dumps(
            {
                "pid": 4242,
                "host": "127.0.0.1",
                "port": 65535,
                "token": "tok",
                "instance_id": "inst",
                "project_id": cli_daemon._project_id(cfg),
            }
        )
    )
    monkeypatch.setattr(cli_daemon, "_health", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_daemon, "_proc_state", lambda _pid: "alive")
    signaled = []
    monkeypatch.setattr(
        cli_daemon, "_terminate", lambda pid, **_k: signaled.append(pid)
    )

    with pytest.raises(cli_daemon.DaemonStopRefused, match="malformed"):
        cli_daemon.stop(cfg)
    assert signaled == []
    assert cfg.server_state_path.exists()


def test_externally_owned_daemon_refuses_direct_stop_without_env(tmp_path):
    """Persistent state, not a cooperative child environment marker, owns stop."""

    cfg = Config(
        project_dir=tmp_path,
        server={"lifecycle_owner": "external"},
    )
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.server_state_path.write_text(
        json.dumps(
            {
                "pid": 4242,
                "process_start_token": "t-owned",
                "lifecycle_owner": "external",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(cli_daemon.DaemonStopRefused, match="externally owned"):
        cli_daemon.stop(cfg)


def test_externally_owned_daemon_start_requires_one_time_bootstrap(
    tmp_path,
    monkeypatch,
):
    cfg = Config(
        project_dir=tmp_path,
        server={"lifecycle_owner": "external"},
    )

    with pytest.raises(cli_daemon.DaemonStopRefused, match="externally owned"):
        cli_daemon.start(cfg)

    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.server_state_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("USD_CLI_EXTERNAL_LIFECYCLE_BOOTSTRAP", "1")
    with pytest.raises(cli_daemon.DaemonStopRefused, match="externally owned"):
        cli_daemon.start(cfg)


def test_daemon_discovery_rejects_lifecycle_owner_mismatch(
    tmp_path,
    monkeypatch,
):
    cfg = Config(
        project_dir=tmp_path,
        server={"lifecycle_owner": "external"},
    )
    state = {
        "pid": 4242,
        "process_start_token": "t-owned",
        "project_id": "project-id",
        "instance_id": "instance-id",
        "lifecycle_owner": "external",
    }
    health = {
        "ok": True,
        **state,
        "lifecycle_owner": "cli",
    }
    monkeypatch.setattr(
        cli_daemon,
        "_owned_process_identity",
        lambda _config, _state: (4242, "t-owned"),
    )
    monkeypatch.setattr(cli_daemon, "_project_id", lambda _config: "project-id")

    assert cli_daemon._health_matches_state(cfg, state, health) is False


# ── issue 3: zombie daemons must be reported as died, not busy ─────────────────────
def _nonreaping_state(pid: int) -> str:
    """Process state via /proc or `ps`, WITHOUT waitpid — the test must not reap the
    zombie before the code under test sees it."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except OSError:
        out = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
        return out or "X"


def _spawn_zombie() -> subprocess.Popen:
    """A child that exits immediately and sits unreaped (defunct) until wait()ed."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.time() + 10
    while not _nonreaping_state(proc.pid).startswith("Z") and time.time() < deadline:
        time.sleep(0.05)
    assert _nonreaping_state(proc.pid).startswith("Z"), "child never became a zombie"
    return proc


def test_status_reports_a_zombie_daemon_as_died(monkeypatch, tmp_path):
    """`kill -0` succeeds on a zombie, so `server status` used to say 'busy — pid N is
    alive but not answering health checks (likely mid-operation)' about a corpse."""
    cfg = Config(project_dir=tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "pid": 4242,
        "host": "127.0.0.1",
        "port": 65535,
        "token": "tok",
        "instance_id": "inst",
        "project_id": cli_daemon._project_id(cfg),
        "process_start_token": "t-dead",
    }
    cfg.server_state_path.write_text(json.dumps(state))
    (cfg.state_dir / "daemon.pids").write_text("4242:t-dead\n")
    monkeypatch.setattr(cli_daemon, "_proc_start_token", lambda _pid: None)
    monkeypatch.setattr(cli_daemon, "_proc_state", lambda _pid: "zombie")
    monkeypatch.setattr(cli_daemon, "_health", lambda *_a, **_k: None)

    msg = cli_daemon.status(cfg)

    assert "died (zombie)" in msg
    assert "server restart" in msg
    assert "busy" not in msg


def test_stop_on_a_zombie_daemon_reports_death_without_signaling(monkeypatch, tmp_path):
    cfg = Config(project_dir=tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "pid": 4242,
        "host": "127.0.0.1",
        "port": 65535,
        "token": "tok",
        "instance_id": "inst",
        "project_id": cli_daemon._project_id(cfg),
        "process_start_token": "t-dead",
    }
    cfg.server_state_path.write_text(json.dumps(state))
    (cfg.state_dir / "daemon.pids").write_text("4242:t-dead\n")
    monkeypatch.setattr(cli_daemon, "_proc_start_token", lambda _pid: None)
    monkeypatch.setattr(cli_daemon, "_proc_state", lambda _pid: "zombie")
    signaled = []
    monkeypatch.setattr(
        cli_daemon, "_terminate", lambda pid, **_k: signaled.append(pid)
    )

    msg = cli_daemon.stop(cfg)

    assert "already exited" in msg
    assert signaled == []
    assert not cfg.server_state_path.exists()
    assert not (cfg.state_dir / "daemon.pids").exists()


# ── round 4, item 14: the PID ledger must carry process identity, not bare pids ────
def _sigterm_immune_child() -> subprocess.Popen:
    """A child that ignores SIGTERM (forces the SIGKILL escalation path)."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(600)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "ready"
    proc.stdout.close()
    return proc


def test_recorded_pids_rejects_an_entire_malformed_or_ambiguous_ledger(tmp_path):
    """A mixed or ambiguous ledger is never partially accepted as ownership proof."""
    cfg = Config(project_dir=tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    me = os.getpid()
    good = "t-current"
    (cfg.state_dir / "daemon.pids").write_text(
        f"{me}:{good}\n"  # matching identity → returned
        f"{me}:x-bogus-start\n"  # simulated pid reuse (wrong birth time) → excluded
        f"0:{good}\n"  # pid 0 (process group) → rejected
        f"-1:{good}\n"  # pid -1 (every permitted process!) → rejected
        f"1:{good}\n"  # init → rejected
        f"{me}\n"
    )  # old bare-pid format → unverifiable, excluded
    assert cli_daemon._recorded_pids(cfg) == set()


def test_reap_orphans_preserves_a_malformed_ledger_without_signaling(
    monkeypatch, tmp_path
):
    """Malformed ownership evidence is preserved for diagnosis and never signaled."""
    cfg = Config(project_dir=tmp_path)
    state = cfg.state_dir
    state.mkdir(parents=True, exist_ok=True)
    sleeper = [sys._base_executable, "-c", "import time;time.sleep(60)"]
    reused = subprocess.Popen(sleeper)  # plays the innocent pid-reuse victim
    bare = subprocess.Popen(sleeper)  # recorded in the OLD bare-pid format
    signaled = []
    monkeypatch.setattr(
        cli_daemon, "_terminate", lambda pid, **_k: signaled.append(pid) or True
    )
    try:
        ledger_text = f"{reused.pid}:x-bogus-reuse\n{bare.pid}\n"
        (state / "daemon.pids").write_text(ledger_text)

        killed = cli_daemon._reap_orphans(cfg, keep_pid=None)

        assert killed == 0
        assert signaled == []  # neither was ever signaled
        assert (state / "daemon.pids").read_text() == ledger_text
        assert reused.poll() is None and bare.poll() is None
    finally:
        reused.kill()
        bare.kill()


def test_reap_orphans_keeps_entries_not_confirmed_gone(monkeypatch, tmp_path):
    """An identity-verified orphan that survives TERM+KILL stays in the ledger — the
    old code forgot every entry even when the process was still alive."""
    cfg = Config(project_dir=tmp_path)
    state = cfg.state_dir
    state.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys._base_executable, "-c", "import time;time.sleep(60)"]
    )
    try:
        tok = cli_daemon._proc_start_token(proc.pid)
        (state / "daemon.pids").write_text(f"{proc.pid}:{tok}\n")
        monkeypatch.setattr(cli_daemon, "_terminate", lambda pid, **_k: False)

        killed = cli_daemon._reap_orphans(cfg, keep_pid=None)

        assert killed == 0
        assert (state / "daemon.pids").read_text() == f"{proc.pid}:{tok}\n"
    finally:
        proc.kill()


def test_terminate_rejects_pid_group_and_init_targets(monkeypatch):
    """kill(0)/kill(-1) signal whole process groups; pid 1 is init. _terminate must
    refuse them before any signal is sent."""
    sent = []
    monkeypatch.setattr(cli_daemon.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert cli_daemon._terminate(0, expected_token="t-safe") is False
    assert cli_daemon._terminate(-1, expected_token="t-safe") is False
    assert cli_daemon._terminate(1, expected_token="t-safe") is False
    assert sent == []


def test_terminate_revalidates_identity_before_sigkill(monkeypatch):
    """The pid can be recycled DURING the TERM grace window: _terminate re-checks the
    birth-time token immediately before SIGKILL and never force-kills a stranger."""
    sent = []
    monkeypatch.setattr(cli_daemon.os, "kill", lambda _pid, sig: sent.append(sig))
    monkeypatch.setattr(cli_daemon, "_proc_state", lambda _pid: "alive")
    tokens = iter(["t-owned"])
    monkeypatch.setattr(
        cli_daemon,
        "_proc_start_token",
        lambda _pid: next(tokens, "t-recycled-stranger"),
    )

    ok = cli_daemon._terminate(
        4242, term_wait_s=0.0, kill_wait_s=0.0, expected_token="t-owned"
    )

    assert ok is True
    assert sent == [signal.SIGTERM]


# ── round 4, item 21: ledger-only stop must fail closed ───────────────────────────
def test_stop_without_state_never_signals_from_ledger_only(monkeypatch, tmp_path):
    """A copied ledger without current-project state cannot authorize a signal."""
    cfg = Config(project_dir=tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    ledger = cfg.state_dir / "daemon.pids"
    ledger.write_text("4242:t-owned\n")
    monkeypatch.setattr(cli_daemon, "_proc_start_token", lambda _pid: "t-owned")
    monkeypatch.setattr(cli_daemon, "_proc_state", lambda _pid: "alive")
    signaled: list[int] = []
    monkeypatch.setattr(
        cli_daemon,
        "_terminate",
        lambda pid, **_kwargs: signaled.append(pid) or True,
    )

    assert cli_daemon.stop(cfg) == "no daemon for this project"
    assert not signaled
    assert ledger.read_text() == "4242:t-owned\n"


def test_stop_without_state_and_without_orphans_is_a_noop(tmp_path):
    cfg = Config(project_dir=tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    assert cli_daemon.stop(cfg) == "no daemon for this project"


# ── round 4, item 15: .fuse_hidden cleanup must stay inside the project ────────────
def test_fuse_cleanup_never_follows_symlinks_out_of_the_project(tmp_path):
    """The old recursive glob followed directory symlinks and deleted by NAME anywhere
    they pointed. Cleanup must skip symlinked dirs, skip symlink files, and only
    unlink regular files that resolve inside the project."""
    project = tmp_path / "proj"
    (project / ".usd-cli").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / ".fuse_hidden_victim"
    victim.write_bytes(b"unrelated file that merely matches the name pattern")
    try:
        (project / "escape").symlink_to(outside)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    link = project / ".fuse_hidden_link"
    link.symlink_to(victim)  # symlink whose NAME matches
    sub = project / "sub"
    sub.mkdir()
    litter = sub / ".fuse_hidden0001"
    litter.write_bytes(b"real litter")
    keeper = sub / "model.usda"
    keeper.write_text("#usda 1.0\n")

    cli_daemon._clean_fuse_litter(project)

    assert not litter.exists()  # genuine litter inside the project is removed
    assert victim.exists()  # never reached through the symlinked dir
    assert link.is_symlink()  # symlinks are never unlinked, even by name
    assert keeper.exists()  # non-matching files untouched


# ── round 4, item 20: idle-watcher must not race a finishing command ────────────────
def test_idle_exit_decision_is_atomic_with_exit_busy():
    """_exit_busy updates the in-flight count AND the activity stamp under ONE lock
    hold, and the watcher (_should_idle_exit) reads both under the same lock — so it
    can never observe 'nothing in flight' paired with a stale timestamp (the window
    that let the daemon exit between a long command finishing and its response
    being returned)."""
    saved = (server_app._IN_FLIGHT[0], server_app._LAST_ACTIVITY[0])
    busy_sessions: dict[str, dict] = {}
    try:
        server_app._IN_FLIGHT[0] = 0
        server_app._enter_busy(busy_sessions, "s", "render")
        # Simulate a command that has ALREADY outlived idle_timeout…
        server_app._LAST_ACTIVITY[0] = time.monotonic() - 10_000
        assert server_app._should_idle_exit(60) is False  # in flight → never idle

        # Play the watcher: hold the busy lock and prove _exit_busy publishes nothing
        # until it can make BOTH updates — no half-finished state is ever visible.
        assert server_app._BUSY_LOCK.acquire(timeout=5)
        finisher = threading.Thread(
            target=server_app._exit_busy, args=(busy_sessions, "s")
        )
        try:
            finisher.start()
            finisher.join(timeout=0.3)
            assert finisher.is_alive()  # blocked on the lock
            assert server_app._IN_FLIGHT[0] == 1  # decrement not yet visible
            assert time.monotonic() - server_app._LAST_ACTIVITY[0] > 60  # stamp stale
        finally:
            server_app._BUSY_LOCK.release()
        finisher.join(timeout=10)
        assert not finisher.is_alive()

        # Once _exit_busy ran, the count is 0 AND the activity stamp is fresh — the
        # watcher sees an atomically-consistent 'just finished', not an idle daemon.
        assert server_app._IN_FLIGHT[0] == 0
        assert server_app._should_idle_exit(60) is False

        # A genuinely idle daemon (past the timeout, nothing in flight) still exits.
        server_app._LAST_ACTIVITY[0] = time.monotonic() - 10_000
        assert server_app._should_idle_exit(60) is True
    finally:
        server_app._IN_FLIGHT[0], server_app._LAST_ACTIVITY[0] = saved


# ── round 5, item 7: per-canonical-root RW lock ─────────────────────────────────────
# `open --read-only` sessions share the writer's live SdfLayer but had independent
# session locks, so a reader could traverse WHILE the writer ran a multi-step edit.
# Dispatch now also takes a per-root RW lock: SHARED for read-only sessions, EXCLUSIVE
# for writers, keyed by the session's canonical root (session.root_key, with
# non-invasive fallbacks). Sessions on different roots stay concurrent.
def _root_fake(root_for, body):
    """A Session fake whose read_only flag and root key derive from the session name
    (reader* => read-only), running the test-scripted `body(session)` inside `info`."""

    class FakeSession:
        def __init__(self, _config, name="default"):
            self._stage_path = None
            self.name = name
            self.read_only = name.startswith("reader")
            self.root_key = root_for(name)

        def info(self):
            body(self)
            return Response(command="info")

    return FakeSession


def _post_info(http, session):
    return threading.Thread(
        target=lambda: http.post(
            "/cmd",
            json={"command": "info", "payload": {}, "session": session},
            headers=HEADERS,
        )
    )


def test_reader_is_blocked_out_during_a_writers_command_on_the_same_root(
    monkeypatch, tmp_path
):
    release = threading.Event()
    writer_running = threading.Event()
    reader_ran = threading.Event()

    def body(session):
        if session.read_only:
            reader_ran.set()
        else:
            writer_running.set()
            release.wait(timeout=30)

    monkeypatch.setattr(
        server_app, "Session", _root_fake(lambda _name: "/canonical/scene.usda", body)
    )
    with TestClient(_app(tmp_path)) as http:
        writer = _post_info(http, "writer")
        writer.start()
        assert writer_running.wait(10), "writer command never started"
        reader = _post_info(http, "reader-1")
        reader.start()
        try:
            assert not reader_ran.wait(0.5), (
                "reader traversed WHILE the writer's multi-step edit was in flight"
            )
        finally:
            release.set()
        assert reader_ran.wait(10), "reader never ran after the writer finished"
        writer.join(timeout=30)
        reader.join(timeout=30)


def test_writer_waits_for_an_in_flight_reader_on_the_same_root(monkeypatch, tmp_path):
    release = threading.Event()
    reader_running = threading.Event()
    writer_ran = threading.Event()

    def body(session):
        if session.read_only:
            reader_running.set()
            release.wait(timeout=30)
        else:
            writer_ran.set()

    monkeypatch.setattr(
        server_app, "Session", _root_fake(lambda _name: "/canonical/scene.usda", body)
    )
    with TestClient(_app(tmp_path)) as http:
        reader = _post_info(http, "reader-1")
        reader.start()
        assert reader_running.wait(10), "reader command never started"
        writer = _post_info(http, "writer")
        writer.start()
        try:
            assert not writer_ran.wait(0.5), (
                "writer mutated the shared layer WHILE a reader was mid-traversal"
            )
        finally:
            release.set()
        assert writer_ran.wait(10), "writer never ran after the reader finished"
        reader.join(timeout=30)
        writer.join(timeout=30)


def test_readers_on_one_root_traverse_concurrently(monkeypatch, tmp_path):
    release = threading.Event()
    gate = threading.Lock()
    active, peak = [0], [0]

    def body(_session):
        with gate:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        try:
            release.wait(timeout=30)
        finally:
            with gate:
                active[0] -= 1

    monkeypatch.setattr(
        server_app, "Session", _root_fake(lambda _name: "/canonical/scene.usda", body)
    )
    with TestClient(_app(tmp_path)) as http:
        threads = [_post_info(http, name) for name in ("reader-a", "reader-b")]
        for thread in threads:
            thread.start()
        try:
            deadline = time.time() + 10
            while peak[0] < 2 and time.time() < deadline:
                time.sleep(0.01)
            assert peak[0] == 2, (
                "two READ-ONLY sessions never overlapped (shared "
                "root-lock mode is broken)"
            )
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=30)


def test_writers_on_different_roots_stay_concurrent(monkeypatch, tmp_path):
    release = threading.Event()
    gate = threading.Lock()
    active, peak = [0], [0]

    def body(_session):
        with gate:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        try:
            release.wait(timeout=30)
        finally:
            with gate:
                active[0] -= 1

    monkeypatch.setattr(
        server_app,
        "Session",  # distinct root per session name
        _root_fake(lambda name: f"/canonical/{name}.usda", body),
    )
    with TestClient(_app(tmp_path)) as http:
        threads = [_post_info(http, name) for name in ("writer-a", "writer-b")]
        for thread in threads:
            thread.start()
        try:
            deadline = time.time() + 10
            while peak[0] < 2 and time.time() < deadline:
                time.sleep(0.01)
            assert peak[0] == 2, "writers on DIFFERENT roots were serialized"
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=30)


def test_writer_open_waits_for_readers_of_the_target_root(monkeypatch, tmp_path):
    """`open` moves to a DIFFERENT root and force-reloads its cached layer — that
    reload must exclude readers currently traversing the target, so the root lock is
    planned from the open's target path, not the session's (old) root."""
    target = str(tmp_path / "scene.usda")
    release = threading.Event()
    reader_running = threading.Event()
    opened = threading.Event()

    class FakeSession:
        def __init__(self, _config, name="default"):
            self._stage_path = None
            self.read_only = name.startswith("reader")
            self.root_key = os.path.realpath(target) if self.read_only else None

        def info(self):
            reader_running.set()
            release.wait(timeout=30)
            return Response(command="info")

        def open_stage(self, file, read_only=False):
            opened.set()
            return Response(command="open")

    monkeypatch.setattr(server_app, "Session", FakeSession)
    with TestClient(_app(tmp_path)) as http:
        reader = _post_info(http, "reader-1")
        reader.start()
        assert reader_running.wait(10)
        writer = threading.Thread(
            target=lambda: http.post(
                "/cmd",
                json={
                    "command": "open",
                    "payload": {"file": target},
                    "session": "writer",
                },
                headers=HEADERS,
            )
        )
        writer.start()
        try:
            assert not opened.wait(0.5), (
                "writer `open` reloaded the target layer under a live reader"
            )
        finally:
            release.set()
        assert opened.wait(10)
        reader.join(timeout=30)
        writer.join(timeout=30)


def test_session_root_key_prefers_root_key_and_falls_back_non_invasively(tmp_path):
    """The dispatcher reads the canonical root via session.root_key (landing
    separately) but must keep working against a Session that predates it: the
    ownership/readers registry keys, then the stage path, all realpath-canonical."""

    class WithRootKey:
        root_key = "/canonical/a"
        _layer_key = "/stale/should-not-win"

    class WriterLegacy:
        _layer_key = "/canonical/w"

    class ReaderLegacy:
        _reader_key = "/canonical/r"

    class PathOnly:
        _stage_path = str(tmp_path / "s.usda")

    class Anonymous:
        _stage_path = None

    assert server_app._session_root_key(WithRootKey()) == "/canonical/a"
    assert server_app._session_root_key(WriterLegacy()) == "/canonical/w"
    assert server_app._session_root_key(ReaderLegacy()) == "/canonical/r"
    assert server_app._session_root_key(PathOnly()) == os.path.realpath(
        str(tmp_path / "s.usda")
    )
    assert server_app._session_root_key(Anonymous()) is None
    assert server_app._session_root_key(object()) is None


# ── round 5, item 13: /health must answer while the worker threadpool is starved ────
def test_health_and_live_answer_while_the_worker_threadpool_is_saturated(
    monkeypatch, tmp_path
):
    """FastAPI SYNC endpoints run on the AnyIO threadpool; same-session waiters park
    there holding a worker token each, so a saturated pool starved the (previously
    sync) /health — and its auth dependency — past the client's 2s busy probe.
    /health and /live are async now and must answer with every worker occupied."""
    release = threading.Event()

    class FakeSession:
        def __init__(self, _config, name="default"):
            self._stage_path = None

        def info(self):
            release.wait(timeout=60)
            return Response(command="info")

    monkeypatch.setattr(server_app, "Session", FakeSession)
    inner = server_app.build_app(
        Config(project_dir=tmp_path), token="t", instance_id="i"
    )
    tokens = 4

    async def limited(scope, receive, send):
        # shrink the loop's worker pool so 4 stuck sync requests saturate it
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = tokens
        await inner(scope, receive, send)

    with TestClient(limited) as http:
        posters = [
            threading.Thread(
                target=lambda: http.post(
                    "/cmd",
                    json={"command": "info", "payload": {}, "session": "work"},
                    headers=HEADERS,
                ),
                daemon=True,
            )
            for _ in range(tokens)
        ]
        for thread in posters:
            thread.start()
        try:
            # all 4 handlers inside worker threads (1 executing + 3 parked on the
            # session lock) == every worker token consumed
            deadline = time.time() + 10
            while server_app._IN_FLIGHT[0] < tokens and time.time() < deadline:
                time.sleep(0.01)
            assert server_app._IN_FLIGHT[0] == tokens, "threadpool never saturated"

            result: dict = {}

            def probe():
                t0 = time.monotonic()
                health = http.get("/health", headers=HEADERS)
                live = http.get("/live")
                result["elapsed"] = time.monotonic() - t0
                result["health"] = health.json()
                result["live"] = live.json()

            prober = threading.Thread(target=probe, daemon=True)
            prober.start()
            prober.join(timeout=10)
            assert result, "/health starved behind the saturated worker threadpool"
            assert result["elapsed"] < 2.0, (
                f"/health took {result['elapsed']:.1f}s — the 2s busy probe would fail"
            )
            assert result["health"]["ok"] is True
            assert result["health"]["sessions"]["work"]["busy"] is True
            assert result["health"]["sessions"]["work"]["queued"] == tokens - 1
            assert result["live"] == {"ok": True}
        finally:
            release.set()
            for thread in posters:
                thread.join(timeout=30)


# ── round 5, item 22: active metadata must update at lock handoff ────────────────────
def test_health_reflects_the_executing_command_after_lock_handoff(
    monkeypatch, tmp_path
):
    """After a queued waiter acquires the session lock, /health used to keep showing
    the completed predecessor's command and start time. The metadata is restamped at
    handoff: the follower's own command name and a fresh `since`."""
    first_release, second_release = threading.Event(), threading.Event()
    first_started = threading.Event()

    class FakeSession:
        def __init__(self, _config, name="default"):
            self._stage_path = None

        def info(self):
            first_started.set()
            first_release.wait(timeout=30)
            return Response(command="info")

        def describe(self):
            second_release.wait(timeout=30)
            return Response(command="describe")

    monkeypatch.setattr(server_app, "Session", FakeSession)
    with TestClient(_app(tmp_path)) as http:
        first = threading.Thread(
            target=lambda: http.post(
                "/cmd",
                json={"command": "info", "payload": {}, "session": "work"},
                headers=HEADERS,
            )
        )
        first.start()
        assert first_started.wait(10)
        second = threading.Thread(
            target=lambda: http.post(
                "/cmd",
                json={"command": "describe", "payload": {}, "session": "work"},
                headers=HEADERS,
            )
        )
        second.start()
        try:
            deadline = time.time() + 10
            health: dict = {}
            while time.time() < deadline:
                health = http.get("/health", headers=HEADERS).json()
                if health["sessions"].get("work", {}).get("queued") == 1:
                    break
                time.sleep(0.02)
            work = health["sessions"]["work"]
            assert work["queued"] == 1, f"queued never became visible: {health}"
            # while `describe` waits, the RUNNING command is still `info`
            assert work["current_command"] == "info"

            time.sleep(2.0)  # let `info` accumulate visible runtime before handoff
            first_release.set()
            deadline = time.time() + 10
            while time.time() < deadline:
                health = http.get("/health", headers=HEADERS).json()
                work = health["sessions"].get("work", {})
                if work.get("current_command") == "describe":
                    break
                time.sleep(0.02)
            assert work["current_command"] == "describe", (
                f"handoff kept the predecessor's metadata: {health}"
            )
            assert work["busy"] is True
            assert work["queued"] == 0
            # `since` was restamped at handoff — not inherited from `info`, which by
            # now had been running for well over 2s
            assert work["busy_seconds"] < 1.5, (
                f"busy_seconds carried over from the predecessor: {work}"
            )
        finally:
            first_release.set()
            second_release.set()
            first.join(timeout=30)
            second.join(timeout=30)


# ── session-name wiring (coordinated with the Session(name=...) change) ────────────
def test_session_name_is_passed_through_when_the_constructor_accepts_it(
    monkeypatch, tmp_path
):
    """When Session grows a `name` parameter, the daemon passes the registry key in;
    a Session (or fake) without it still works via the TypeError fallback — the
    fallback path is already pinned by the FakeSession concurrency test above."""
    names = []

    class NamedSession:
        def __init__(self, _config, name):
            names.append(name)
            self._stage_path = None

    monkeypatch.setattr(server_app, "Session", NamedSession)
    with TestClient(_app(tmp_path)) as http:
        _cmd(http, "info", session="alpha")
    assert names == [server_app.DEFAULT_SESSION, "alpha"]
