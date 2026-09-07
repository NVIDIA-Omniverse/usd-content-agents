# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Black-box tests for usd-cli-tel, the transparent telemetry wrapper.

The wrapper is driven as a subprocess (`python -m usd_telemetry`) against a
fake usd-cli, mirroring how the rest of the suite drives the real CLI. No GPU,
daemon, or USD runtime needed.
"""

from __future__ import annotations

import http.server
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

sys.path.insert(0, str(SRC))

from usd_telemetry.span import parse_command, redact_args, sniff_scenes  # noqa: E402
from usd_telemetry import main as telemetry_main  # noqa: E402
from usd_cli.main import app  # noqa: E402

FAKE_USD_CLI = """\
#!/usr/bin/env python3
import os, signal, sys, time
sys.stdout.write("payload:" + " ".join(sys.argv[1:]) + "\\n")
sys.stderr.write("progress line\\n")
if "--pid-file" in sys.argv:
    path = sys.argv[sys.argv.index("--pid-file") + 1]
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
if "--stderr-bytes" in sys.argv:
    sys.stderr.write("x" * (2 * 1024 * 1024))
if "--ignore-term" in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if "--sleep-forever" in sys.argv:
    while True:
        time.sleep(1)
if "--sleep" in sys.argv:
    time.sleep(0.15)
if "--fail" in sys.argv:
    sys.stderr.write("boom: stage not found\\n")
    sys.exit(3)
"""


@pytest.mark.skipif(os.name != "posix", reason="/proc token parsing is POSIX-only")
def test_process_start_token_handles_spaces_and_parentheses_in_comm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields_after_comm = ["S", *(str(value) for value in range(4, 22)), "98765"]
    raw = f"123 (worker name) with spaces) {' '.join(fields_after_comm)}\n"
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _self, **_kwargs: raw,
    )

    assert telemetry_main._process_start_token(123) == "98765"


def test_action_lease_degrades_without_process_start_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lock = tmp_path / "agent-action.lock.json"
    monkeypatch.setattr(
        telemetry_main,
        "_action_owner",
        lambda: {"pid": 7, "start_token": None, "scope": "parent_process"},
    )

    telemetry_main._acquire_action_lease(lock, timeout_seconds=0.1)

    assert not lock.exists()
    assert "action serialization unavailable" in capsys.readouterr().err


def test_action_lease_publishes_a_complete_owner_record_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock = tmp_path / "agent-action.lock.json"
    owner = {"pid": 7, "start_token": "start", "scope": "parent_process"}
    observed: list[object] = []
    original_link = os.link
    monkeypatch.setattr(telemetry_main, "_action_owner", lambda: owner)

    def inspect_link(source, destination, *, follow_symlinks=True):
        observed.append(json.loads(Path(source).read_text(encoding="utf-8")))
        return original_link(
            source,
            destination,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(os, "link", inspect_link)

    telemetry_main._acquire_action_lease(lock, timeout_seconds=0.1)

    assert observed == [owner]
    assert json.loads(lock.read_text(encoding="utf-8")) == owner


@pytest.mark.skipif(os.name != "posix", reason="fcntl contention is POSIX-only")
def test_action_lease_guard_contention_honors_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fcntl

    lock = tmp_path / "agent-action.lock.json"
    guard = tmp_path / ".agent-action.lock.json.guard"
    owner = {"pid": 7, "start_token": "start", "scope": "parent_process"}
    monkeypatch.setattr(telemetry_main, "_action_owner", lambda: owner)

    descriptor = os.open(guard, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        started = time.monotonic()

        with pytest.raises(TimeoutError, match="timed out waiting"):
            telemetry_main._acquire_action_lease(lock, timeout_seconds=0.05)

        assert time.monotonic() - started < 1
        assert not lock.exists()
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@pytest.fixture
def fake_cli(tmp_path: Path) -> Path:
    path = tmp_path / "fake-usd-cli"
    path.write_text(FAKE_USD_CLI)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def run_wrapper(
    args: list[str],
    fake_cli: Path,
    log: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "USD_CLI_TEL_TARGET": str(fake_cli),
        "USD_CLI_TEL_FILE": str(log),
        **(extra_env or {}),
    }
    return subprocess.run(
        [sys.executable, "-m", "usd_telemetry", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def read_records(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


# ------------------------------------------------------------- passthrough


def test_passthrough_stdout_stderr_and_exit_code(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    res = run_wrapper(["snapshot", "--flag", "value"], fake_cli, log)
    assert res.returncode == 0
    assert res.stdout == "payload:snapshot --flag value\n"
    assert "progress line" in res.stderr
    assert "usd-cli-tel" not in res.stderr  # no wrapper noise


def test_failure_exit_code_mirrored(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    res = run_wrapper(["render", "--fail"], fake_cli, log)
    assert res.returncode == 3
    assert "boom: stage not found" in res.stderr


def test_large_stderr_is_drained_without_blocking_child(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"

    res = run_wrapper(["render", "--stderr-bytes"], fake_cli, log)

    assert res.returncode == 0
    assert len(res.stderr) > 2 * 1024 * 1024
    assert len(read_records(log)) == 1


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="requires procfs")
def test_sigterm_escalates_when_child_ignores_shutdown(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    pid_file = tmp_path / "child.pid"
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "USD_CLI_TEL_TARGET": str(fake_cli),
        "USD_CLI_TEL_FILE": str(log),
    }
    wrapper = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "usd_telemetry",
            "render",
            "--pid-file",
            str(pid_file),
            "--ignore-term",
            "--sleep-forever",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        deadline = time.monotonic() + 5
        while not pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.is_file()

        started = time.monotonic()
        wrapper.terminate()
        _stdout, stderr = wrapper.communicate(timeout=6)
        elapsed = time.monotonic() - started

        assert wrapper.returncode == 137, stderr
        assert elapsed < 5
        assert read_records(log)[0]["attributes"]["process.exit_code"] == 137
        child_pid = int(pid_file.read_text())
        child_deadline = time.monotonic() + 2
        while (
            Path(f"/proc/{child_pid}").exists()
            and time.monotonic() < child_deadline
        ):
            time.sleep(0.01)
        assert not Path(f"/proc/{child_pid}").exists()
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.communicate()


def test_missing_target_exits_127(tmp_path):
    log = tmp_path / "tel.jsonl"
    res = run_wrapper(
        ["snapshot"], tmp_path / "does-not-exist", log
    )
    assert res.returncode == 127
    assert "not found" in res.stderr
    assert not log.exists()


# ------------------------------------------------------------------ record


def test_record_shape_ok(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    run_wrapper(["render", "--camera", "/World/Cam", "--sleep"], fake_cli, log)
    (rec,) = read_records(log)

    assert rec["name"] == "usd-cli.render"
    assert len(rec["trace_id"]) == 32 and len(rec["span_id"]) == 16
    assert rec["status"] == {"code": "OK"}
    assert rec["end_time_unix_nano"] > rec["start_time_unix_nano"]
    assert rec["duration_ms"] >= 150  # --sleep floor

    attrs = rec["attributes"]
    assert attrs["usd.command"] == "render"
    assert attrs["process.exit_code"] == 0
    assert attrs["process.command_args"] == ["render", "--camera", "/World/Cam", "--sleep"]
    assert attrs["telemetry.sdk.name"] == "usd-cli-tel"


def test_record_error_status_and_stderr_tail(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    run_wrapper(["render", "--fail"], fake_cli, log)
    (rec,) = read_records(log)
    assert rec["status"]["code"] == "ERROR"
    assert "boom: stage not found" in rec["status"]["message"]
    assert rec["attributes"]["error.type"] == "exit_code:3"


def test_scene_sniffing(fake_cli, tmp_path):
    scene = tmp_path / "pcb.usda"
    scene.write_text("#usda 1.0\n")
    log = tmp_path / "tel.jsonl"
    run_wrapper(["snapshot", str(scene)], fake_cli, log)
    (rec,) = read_records(log)
    attrs = rec["attributes"]
    assert attrs["usd.scene.path"] == str(scene)
    assert attrs["usd.scene.size_bytes"] == scene.stat().st_size


def test_session_and_server_lifted(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    run_wrapper(
        ["--session", "demo", "--server=http://ovx15:8000", "snapshot"], fake_cli, log
    )
    (rec,) = read_records(log)
    assert rec["attributes"]["usd.command"] == "snapshot"
    assert rec["attributes"]["usd.session"] == "demo"
    assert rec["attributes"]["usd.server"] == "http://ovx15:8000"


def test_trace_id_env_correlates_invocations(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    tid = "ab" * 16
    env = {"USD_CLI_TEL_TRACE_ID": tid}
    run_wrapper(["snapshot"], fake_cli, log, env)
    run_wrapper(["render"], fake_cli, log, env)
    recs = read_records(log)
    assert [r["trace_id"] for r in recs] == [tid, tid]
    assert recs[0]["span_id"] != recs[1]["span_id"]


def test_disabled_writes_nothing_but_passes_through(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    res = run_wrapper(["snapshot"], fake_cli, log, {"USD_CLI_TEL_DISABLED": "1"})
    assert res.returncode == 0
    assert res.stdout == "payload:snapshot\n"
    assert not log.exists()


@pytest.mark.parametrize("action", ["start", "stop", "restart"])
def test_parent_managed_child_cannot_run_server_lifecycle(
    action, fake_cli, tmp_path
):
    log = tmp_path / "tel.jsonl"
    res = run_wrapper(
        ["--server", "http://127.0.0.1:8123", "server", action],
        fake_cli,
        log,
        {
            "USD_CLI_LIFECYCLE_EXTERNALLY_OWNED": "1",
            "USD_CLI_TEL_DISABLED": "1",
        },
    )
    assert res.returncode == 2
    assert f"forbid `server {action}`" in res.stderr
    assert "payload:" not in res.stdout
    assert not log.exists()


def test_parent_managed_child_can_run_non_lifecycle_command(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    res = run_wrapper(
        ["--server", "http://127.0.0.1:8123", "snapshot"],
        fake_cli,
        log,
        {"USD_CLI_LIFECYCLE_EXTERNALLY_OWNED": "1"},
    )
    assert res.returncode == 0
    assert "payload:--server http://127.0.0.1:8123 snapshot" in res.stdout


@pytest.mark.parametrize("action", ["start", "stop", "restart"])
def test_bare_cli_parent_managed_child_cannot_run_server_lifecycle(
    action, monkeypatch
):
    monkeypatch.setenv("USD_CLI_LIFECYCLE_EXTERNALLY_OWNED", "1")

    result = CliRunner().invoke(app, ["server", action])

    assert result.exit_code != 0
    assert f"forbid child `server {action}`" in result.stderr


def test_parent_managed_parser_does_not_block_server_stop_values(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    res = run_wrapper(
        ["set", "/Prim", "server", "stop"],
        fake_cli,
        log,
        {"USD_CLI_LIFECYCLE_EXTERNALLY_OWNED": "1"},
    )

    assert res.returncode == 0
    assert "payload:set /Prim server stop" in res.stdout


def test_file_rotation(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    env = {"USD_CLI_TEL_FILE_MAX_BYTES": "1"}
    run_wrapper(["snapshot"], fake_cli, log, env)
    run_wrapper(["render"], fake_cli, log, env)
    assert log.exists() and log.with_name("tel.jsonl.1").exists()


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="requires procfs")
def test_action_lock_serializes_distinct_parent_shells(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    lock = tmp_path / "agent-action.lock.json"
    environment = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "USD_CLI_TEL_TARGET": str(fake_cli),
        "USD_CLI_TEL_FILE": str(log),
        "USD_CLI_TEL_ACTION_LOCK": str(lock),
        "USD_CLI_TEL_ACTION_LOCK_TIMEOUT": "5",
    }
    first = subprocess.Popen(
        [
            "/bin/bash",
            "-c",
            f"{sys.executable} -m usd_telemetry snapshot --sleep; sleep 0.4",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    deadline = time.monotonic() + 5
    while not lock.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert lock.exists()

    started = time.monotonic()
    second = subprocess.run(
        ["/bin/bash", "-c", f"{sys.executable} -m usd_telemetry render"],
        capture_output=True,
        text=True,
        env=environment,
        timeout=5,
    )
    elapsed = time.monotonic() - started
    first_stdout, first_stderr = first.communicate(timeout=5)

    assert first.returncode == 0, first_stderr
    assert second.returncode == 0, second.stderr
    assert first_stdout == "payload:snapshot --sleep\n"
    assert second.stdout == "payload:render\n"
    assert elapsed >= 0.35


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="requires procfs")
def test_action_lock_recovers_stale_owner(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    lock = tmp_path / "agent-action.lock.json"
    lock.write_text('{"pid": 2147483647, "start_token": "stale"}\n')

    result = run_wrapper(
        ["snapshot"],
        fake_cli,
        log,
        {
            "USD_CLI_TEL_ACTION_LOCK": str(lock),
            "USD_CLI_TEL_ACTION_LOCK_TIMEOUT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    replacement = json.loads(lock.read_text())
    assert replacement["pid"] != 2147483647
    assert replacement["start_token"] != "stale"


# -------------------------------------------------------------------- otlp


class _CaptureHandler(http.server.BaseHTTPRequestHandler):
    captured: list[dict] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).captured.append(
            {"path": self.path, "headers": dict(self.headers), "body": json.loads(body)}
        )
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_otlp_backend_posts_spec_shaped_payload(fake_cli, tmp_path):
    _CaptureHandler.captured = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        log = tmp_path / "tel.jsonl"
        res = run_wrapper(
            ["render", "--fail"],
            fake_cli,
            log,
            {
                "USD_CLI_TEL_BACKENDS": "file,otlp",
                "USD_CLI_TEL_OTLP_ENDPOINT": f"http://127.0.0.1:{server.server_port}/v1/traces",
                "USD_CLI_TEL_OTLP_HEADERS": "Authorization=Bearer sekret",
            },
        )
        assert res.returncode == 3
        (posted,) = _CaptureHandler.captured
        assert posted["path"] == "/v1/traces"
        assert posted["headers"]["Authorization"] == "Bearer sekret"
        (span,) = posted["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert span["name"] == "usd-cli.render"
        assert span["status"]["code"] == 2  # STATUS_CODE_ERROR
        assert span["kind"] == 3  # SPAN_KIND_CLIENT
        keys = {a["key"] for a in span["attributes"]}
        assert {"usd.command", "process.exit_code", "process.command_args"} <= keys
        # file backend still wrote alongside
        assert len(read_records(log)) == 1
    finally:
        server.shutdown()


def test_otlp_endpoint_down_never_breaks_the_call(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    res = run_wrapper(
        ["snapshot"],
        fake_cli,
        log,
        {
            "USD_CLI_TEL_BACKENDS": "otlp,file",
            "USD_CLI_TEL_OTLP_ENDPOINT": "http://127.0.0.1:1/v1/traces",
            "USD_CLI_TEL_OTLP_TIMEOUT": "0.2",
        },
    )
    assert res.returncode == 0
    assert res.stdout == "payload:snapshot\n"
    assert len(read_records(log)) == 1  # other backends unaffected


# -------------------------------------------------------------- unit-level


def test_parse_command_skips_global_flags():
    cmd, lifted = parse_command(["--json", "--timeout", "60", "material", "set"])
    assert cmd == "material"
    assert lifted == {"--timeout": "60"}


def test_redact_args_masks_secret_values():
    argv = ["render", "--api-key", "s3cr3t", "--token=abc", "--camera", "/World/Cam"]
    assert redact_args(argv) == [
        "render", "--api-key", "<redacted>", "--token=<redacted>", "--camera", "/World/Cam",
    ]


def test_sniff_scenes_marks_missing_files(tmp_path):
    real = tmp_path / "a.usd"
    real.write_text("x")
    scenes = sniff_scenes([str(real), "ghost.usdz", str(real)])
    assert len(scenes) == 2
    assert scenes[0]["path"] == str(real) and scenes[0]["size_bytes"] == 1
    assert scenes[1] == {"path": "ghost.usdz", "exists": False}


def test_outer_pid_namespace_process_group_uses_host_visible_group(monkeypatch):
    from usd_telemetry import main

    monkeypatch.setattr(
        main.Path,
        "read_text",
        lambda *_args, **_kwargs: (
            "Name:\tpython\n"
            "NSpid:\t3514151\t2\n"
            "NSpgid:\t3514150\t1\n"
        ),
    )

    assert main._outer_pid_namespace_process_group() == 3514150


def test_outer_pid_namespace_process_group_ignores_host_namespace(monkeypatch):
    from usd_telemetry import main

    monkeypatch.setattr(
        main.Path,
        "read_text",
        lambda *_args, **_kwargs: "NSpid:\t123\nNSpgid:\t100\n",
    )

    assert main._outer_pid_namespace_process_group() is None


# ------------------------------------------------- distributed trace context

ENV_DUMP_CLI = """\
#!/usr/bin/env python3
import json, os, sys
with open(sys.argv[sys.argv.index("--dump") + 1], "w") as fh:
    json.dump({k: v for k, v in os.environ.items() if k == "TRACEPARENT"}, fh)
"""


@pytest.fixture
def env_dump_cli(tmp_path: Path) -> Path:
    path = tmp_path / "fake-usd-cli-envdump"
    path.write_text(ENV_DUMP_CLI)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_traceparent_env_joins_parent_trace(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    tid, parent = "cd" * 16, "12" * 8
    run_wrapper(["snapshot"], fake_cli, log, {"TRACEPARENT": f"00-{tid}-{parent}-01"})
    (rec,) = read_records(log)
    assert rec["trace_id"] == tid
    assert rec["parent_span_id"] == parent
    assert rec["span_id"] not in (parent, "")


def test_traceparent_wins_over_trace_id_env(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    tid, parent = "cd" * 16, "12" * 8
    run_wrapper(
        ["snapshot"],
        fake_cli,
        log,
        {"TRACEPARENT": f"00-{tid}-{parent}-01", "USD_CLI_TEL_TRACE_ID": "ab" * 16},
    )
    (rec,) = read_records(log)
    assert rec["trace_id"] == tid


def test_malformed_traceparent_ignored(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    for bad in ("garbage", "00-short-1212121212121212-01", "ff-" + "cd" * 16 + "-" + "12" * 8 + "-01", "00-" + "0" * 32 + "-" + "12" * 8 + "-01"):
        run_wrapper(["snapshot"], fake_cli, log, {"TRACEPARENT": bad})
    recs = read_records(log)
    assert all("parent_span_id" not in r for r in recs)
    assert len({r["trace_id"] for r in recs}) == len(recs)  # random per call


def test_tracestate_recorded_with_valid_parent(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    tid, parent = "cd" * 16, "12" * 8
    run_wrapper(
        ["snapshot"],
        fake_cli,
        log,
        {"TRACEPARENT": f"00-{tid}-{parent}-01", "TRACESTATE": "wu=run1"},
    )
    (rec,) = read_records(log)
    assert rec["trace_state"] == "wu=run1"


def test_child_receives_traceparent_naming_wrapper_span(env_dump_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    dump = tmp_path / "childenv.json"
    tid, parent = "cd" * 16, "12" * 8
    res = subprocess.run(
        [sys.executable, "-m", "usd_telemetry", "--dump", str(dump)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(SRC),
            "USD_CLI_TEL_TARGET": str(env_dump_cli),
            "USD_CLI_TEL_FILE": str(log),
            "TRACEPARENT": f"00-{tid}-{parent}-01",
        },
        timeout=30,
    )
    assert res.returncode == 0
    (rec,) = read_records(log)
    child_tp = json.loads(dump.read_text())["TRACEPARENT"]
    # Child's parent is the wrapper span, same trace.
    assert child_tp == f"00-{tid}-{rec['span_id']}-01"


def test_extra_attrs_and_parent_agent_session_recorded(fake_cli, tmp_path):
    log = tmp_path / "tel.jsonl"
    run_wrapper(
        ["snapshot"],
        fake_cli,
        log,
        {
            "USD_CLI_TEL_ATTRS": "wu.run_id=asset-1, wu.step =initial",
            "CLAUDE_CODE_SESSION_ID": "sess-42",
        },
    )
    (rec,) = read_records(log)
    assert rec["attributes"]["wu.run_id"] == "asset-1"
    assert rec["attributes"]["wu.step"] == "initial"
    assert rec["attributes"]["parent.claude_code.session_id"] == "sess-42"


def test_otlp_payload_carries_parent_span_id():
    from usd_telemetry.backends import to_otlp_payload
    from usd_telemetry.span import build_record

    record = build_record(
        argv=["render"],
        executable="/bin/usd-cli",
        exit_code=0,
        start_unix_nano=1,
        end_unix_nano=2,
        stderr_tail="",
        trace_id="cd" * 16,
        span_id="34" * 8,
        parent_span_id="12" * 8,
        trace_state="wu=run1",
    )
    span = to_otlp_payload(record)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["parentSpanId"] == "12" * 8
    assert span["traceState"] == "wu=run1"


def test_resolve_target_skips_self_shim(fake_cli, tmp_path, monkeypatch):
    """A `usd-cli` PATH shim pointing at the wrapper must not recurse."""
    from usd_telemetry.config import Config
    from usd_telemetry.main import _resolve_target

    wrapper_self = tmp_path / "wrapper-self"
    wrapper_self.write_text("#!/bin/sh\n")
    wrapper_self.chmod(0o755)
    shim_dir = tmp_path / "shims"
    shim_dir.mkdir()
    (shim_dir / "usd-cli").symlink_to(wrapper_self)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_cli = real_dir / "usd-cli"
    real_cli.write_text(FAKE_USD_CLI)
    real_cli.chmod(0o755)

    monkeypatch.setattr(sys, "argv", [str(wrapper_self)])
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{real_dir}")
    assert _resolve_target(Config()) == str(real_cli)

    monkeypatch.setenv("PATH", str(shim_dir))
    assert _resolve_target(Config()) is None


@pytest.mark.skipif(os.name != "nt", reason="Windows PATHEXT regression")
def test_resolve_target_applies_windows_pathext(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from usd_telemetry.config import Config
    from usd_telemetry.main import _resolve_target

    real_cli = tmp_path / "usd-cli.exe"
    real_cli.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "usd-cli-tel.exe")])
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("PATHEXT", ".EXE")

    resolved = _resolve_target(Config())

    assert resolved is not None
    assert os.path.samefile(resolved, real_cli)
