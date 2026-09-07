# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-5 fixes from the v4 benchmark traces: busy-vs-dead daemon reporting,
queued-command visibility in /health, and --server token handling.

The traces showed three client/daemon failure modes: (1) a /cmd read timeout against a
daemon whose session lock was held through a multi-minute packaging/upload surfaced as
"daemon unreachable: timed out" — agents read that as a dead daemon and ran `server
restart`, killing in-flight renders; (2) /health reported only in-flight commands, so a
pile-up of commands QUEUED behind one long operation was invisible; (3) `--server URL`
dropped the stored daemon token and died with an unexplained 401. These tests pin the
fixes: the busy probe + error_type "busy" + exit code 4 in `usd_cli.client`/`output`,
per-session queued counters in `usd_server.app`, and state-file/USD_CLI_TOKEN reuse for an
explicit --server.

The v5 review tightened all three: "busy" requires a VERIFIED /health answer (200 +
{"ok": true} — a 401/404/bad body could be an unrelated listener), only a ReadTimeout
is potentially-busy (write/pool timeouts may never have reached the daemon), the busy
message reports the outcome as UNKNOWN (the timed-out request is not cancelled and may
still execute — blind retries can run a mutation twice), the structured busy summary
carries the queue depth, and --server inherits the state file's token only on an EXACT
scheme+host+port match (no ::1/localhost/127.0.0.1 folding).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import httpx
from fastapi.testclient import TestClient

from usd_cli import client, daemon
from usd_cli.output import EXIT_BUSY, EXIT_UNREACHABLE, emit
from usd_cli.state import G
from usd_core.config import Config
from usd_core.models import Response
from usd_server import app as server_app

HEADERS = {"x-usd-cli-token": "t"}

#: What a daemon mid-packaging reports on its lock-free /health: two busy sessions —
#: the client must name the BUSIEST one (longest-running), including its queue depth.
BUSY_HEALTH = {
    "ok": True,
    "busy": True,
    "current_command": "render",
    "busy_seconds": 34.0,
    "queued": 2,
    "sessions": {
        "default": {
            "stage": None,
            "busy": True,
            "current_command": "render",
            "busy_seconds": 34.0,
            "queued": 0,
        },
        "packager": {
            "stage": "/tmp/scene.usda",
            "busy": True,
            "current_command": "save",
            "busy_seconds": 512.3,
            "queued": 2,
        },
    },
}


# ── stub daemons ─────────────────────────────────────────────────────────────────
class _StubHandler(BaseHTTPRequestHandler):
    """Scriptable daemon stand-in: /health answers per `health_payload` (None → 401),
    /cmd hangs past the client timeout when `hang_cmd`, else checks `expected_token`."""

    def log_message(self, *_args):  # keep pytest output clean
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (http.server API)
        srv = self.server
        srv.requests.append(("GET", self.path, self.headers.get("x-usd-cli-token")))
        if self.path == "/health":
            if srv.health_payload is None:
                self._json(401, {"detail": "invalid daemon token"})
            else:
                self._json(200, srv.health_payload)
        elif self.path == "/live":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self):  # noqa: N802 (http.server API)
        srv = self.server
        token = self.headers.get("x-usd-cli-token")
        srv.requests.append(("POST", self.path, token))
        self.rfile.read(int(self.headers.get("content-length") or 0))
        if srv.hang_cmd:
            srv.stop.wait(30)  # hold the request well past the client's timeout
            return
        if srv.expected_token is not None and token != srv.expected_token:
            self._json(401, {"detail": "invalid daemon token"})
            return
        self._json(
            200,
            Response(command="snapshot", ok=True, summary={"served": True}).to_dict(),
        )


@pytest.fixture()
def stub_server():
    """Factory for stub daemons; returns (server, base_url) and tears them all down."""
    live: list[ThreadingHTTPServer] = []

    def make(
        *,
        health_payload: dict | None = None,
        hang_cmd: bool = False,
        expected_token: str | None = None,
    ):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        srv.daemon_threads = True
        srv.health_payload = health_payload
        srv.hang_cmd = hang_cmd
        srv.expected_token = expected_token
        srv.stop = threading.Event()
        srv.requests = []
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        live.append(srv)
        return srv, f"http://127.0.0.1:{srv.server_address[1]}"

    yield make
    for srv in live:
        srv.stop.set()
        srv.shutdown()
        srv.server_close()


@pytest.fixture()
def dead_socket():
    """A server that accepts TCP connections but never sends a byte back — /cmd AND
    /health both time out, which is the one case that IS 'unreachable'."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    conns: list[socket.socket] = []
    stop = threading.Event()

    def loop():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
                conns.append(conn)  # hold it open, never respond
            except TimeoutError:
                continue
            except OSError:
                break

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.getsockname()[1]}"
    stop.set()
    for conn in conns:
        conn.close()
    srv.close()
    thread.join(timeout=2)


@pytest.fixture()
def cli_g(monkeypatch, tmp_path):
    """Isolated CLI globals + config: short explicit timeout, no token env bleed, and
    load_config pinned to a throwaway project dir (no real .usd-cli discovery)."""
    saved = (G.json, G.quiet, G.server, G.session, G.timeout, G.timeout_explicit)
    G.json, G.quiet, G.session = False, False, None
    G.server, G.timeout, G.timeout_explicit = None, 1.0, True
    monkeypatch.delenv("USD_CLI_TOKEN", raising=False)
    monkeypatch.delenv("USD_CLI_SERVER_TOKEN", raising=False)
    monkeypatch.delenv(client.PARENT_USD_CLI_TOKEN_ENV, raising=False)
    monkeypatch.delenv(client.PARENT_USD_CLI_AUTH_PROXY_ENV, raising=False)
    monkeypatch.delenv(client.PARENT_USD_CLI_SERVER_URL_ENV, raising=False)
    monkeypatch.setattr(client, "load_config", lambda: Config(project_dir=tmp_path))
    yield G
    (G.json, G.quiet, G.server, G.session, G.timeout, G.timeout_explicit) = saved


def _write_state(tmp_path, *, port: int, token: str) -> None:
    state_dir = tmp_path / ".usd-cli"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "server.json").write_text(
        json.dumps(
            {
                "pid": 12345,
                "host": "127.0.0.1",
                "port": port,
                "token": token,
                "instance_id": "i",
                "project_id": "p",
            }
        )
    )


def test_loopback_command_live_and_health_requests_bypass_proxy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[bool] = []
    requests: list[tuple[str, str, dict | None]] = []

    class FakeResponse:
        status_code = 200

        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.payload

    class FakeClient:
        def __init__(self, *, trust_env: bool) -> None:
            clients.append(trust_env)

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_args) -> None:  # noqa: ANN002
            return None

        def get(self, url: str, **kwargs):  # noqa: ANN003,ANN201
            requests.append(("GET", url, kwargs.get("headers")))
            return FakeResponse({"ok": True})

        def post(self, url: str, **kwargs):  # noqa: ANN003,ANN201
            requests.append(("POST", url, kwargs.get("headers")))
            return FakeResponse(Response(command="snapshot", ok=True).to_dict())

    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setattr(httpx, "Client", FakeClient)

    base = "http://127.0.0.1:4567"
    assert client._probe_alive(base, 1.0) is None
    assert client._fetch_health(base, "secret", 1.0) == {"ok": True}
    assert client._post(f"{base}/cmd", "secret", {"command": "snapshot"}, 1.0).ok

    assert clients == [False, False, False]
    assert requests[0] == ("GET", f"{base}/live", None)
    for method, url, headers in requests[1:]:
        assert method in {"GET", "POST"}
        assert url.startswith(base)
        assert headers == {
            "x-usd-cli-token": "secret",
            "x-ov-token": "secret",
            "x-3dsc-token": "secret",
        }
    assert client._trust_proxy_environment("http://localhost.:4567") is False
    assert client._trust_proxy_environment("http://[::1]:4567") is False
    assert client._trust_proxy_environment("https://render.example.test") is True


@pytest.mark.parametrize(
    "sdk_proxy",
    (
        "http://srt:opaque-sdk-token@localhost:32123",
        "http://srt.run-context:opaque-cli-token@localhost:32123",
    ),
)
def test_attached_parent_requests_use_only_validated_codex_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    sdk_proxy: str,
) -> None:
    attached_project = tmp_path / "parent" / "run"
    attached_project.mkdir(parents=True)
    discovered: list[tuple[object, str]] = []

    def discover_attached(config, expected_base: str):  # noqa: ANN001,ANN202
        discovered.append((config, expected_base))
        if ":4567" not in expected_base:
            return None
        return "http://127.0.0.1:4567", "parent-token"

    monkeypatch.setattr(daemon, "discover_attached", discover_attached)
    monkeypatch.setenv(daemon.PARENT_USD_CLI_PROXY_ALLOWED_ENV, "1")
    monkeypatch.setenv(
        daemon.ATTACHED_PROJECT_DIR_ENV,
        str(attached_project.resolve()),
    )
    monkeypatch.setenv("HTTP_PROXY", sdk_proxy)

    assert (
        client._attached_parent_http_proxy_for_url("http://127.0.0.1:4567/cmd")
        == sdk_proxy
    )
    assert (
        client._attached_parent_http_proxy_for_url("http://127.0.0.1:4568/cmd")
        is None
    )
    assert client._trust_proxy_environment("https://render.example.test") is False
    assert (
        client._attached_parent_http_proxy_for_url("https://render.example.test/cmd")
        is None
    )
    assert discovered[0][0].project_dir == attached_project.resolve()
    assert daemon.attached_parent_http_proxy() == sdk_proxy

    for invalid_proxy in (
        "http://proxy.example:32123",
        "http://user:secret@127.0.0.1:32123",
        "http://srt:@localhost:32123",
        "http://srt.:opaque-token@localhost:32123",
        "http://srt.run-context@localhost:32123",
        "http://srt:secret@localhost.:32123",
        "http://127.0.0.1:32123/redirect",
        "socks5://127.0.0.1:32123",
    ):
        monkeypatch.setenv("HTTP_PROXY", invalid_proxy)
        assert daemon.attached_parent_http_proxy() is None
        assert (
            client._attached_parent_http_proxy_for_url(
                "http://127.0.0.1:4567/cmd"
            )
            is None
        )

    monkeypatch.setenv("HTTP_PROXY", sdk_proxy)
    monkeypatch.delenv(daemon.PARENT_USD_CLI_PROXY_ALLOWED_ENV)
    assert daemon.attached_parent_http_proxy() is None


def test_managed_parent_auth_proxy_is_exact_endpoint_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = "http://127.0.0.1:32123"
    parent = "http://127.0.0.1:4567"
    monkeypatch.setenv(client.PARENT_USD_CLI_AUTH_PROXY_ENV, proxy)
    monkeypatch.setenv(client.PARENT_USD_CLI_SERVER_URL_ENV, parent)
    monkeypatch.setenv(client.PARENT_USD_CLI_MANAGED_ENV, "1")
    monkeypatch.setenv(client.USD_CLI_EXTERNAL_LIFECYCLE_ENV, "1")
    monkeypatch.setenv("USD_CLI_NO_DAEMON", "1")

    assert client._attached_parent_http_proxy_for_url(f"{parent}/cmd") == proxy
    assert client._attached_parent_http_proxy_for_url(f"{parent}/live") == proxy
    assert (
        client._attached_parent_http_proxy_for_url("http://127.0.0.1:4568/cmd")
        is None
    )

    monkeypatch.setenv(
        client.PARENT_USD_CLI_AUTH_PROXY_ENV,
        "http://user:secret@127.0.0.1:32123",
    )
    assert client._attached_parent_http_proxy_for_url(f"{parent}/cmd") is None

    monkeypatch.setenv(client.PARENT_USD_CLI_AUTH_PROXY_ENV, proxy)
    monkeypatch.delenv(client.PARENT_USD_CLI_MANAGED_ENV)
    assert client._attached_parent_http_proxy_for_url(f"{parent}/cmd") is None


@pytest.mark.parametrize(
    "proxy",
    (
        "http://127.0.0.1:32123",
        "http://[::1]:32123",
        "http://localhost:32123",
    ),
)
def test_attached_parent_proxy_accepts_only_loopback_endpoint_variants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    proxy: str,
) -> None:
    attached_project = tmp_path / "parent" / "run"
    attached_project.mkdir(parents=True)
    monkeypatch.setenv(daemon.PARENT_USD_CLI_PROXY_ALLOWED_ENV, "1")
    monkeypatch.setenv(daemon.ATTACHED_PROJECT_DIR_ENV, str(attached_project))
    monkeypatch.setenv("HTTP_PROXY", proxy)

    assert daemon.attached_parent_http_proxy() == proxy


# ── #1 busy daemon must not be reported "unreachable" ────────────────────────────
def test_timeout_against_busy_daemon_reports_busy_with_exit_4(
    cli_g, stub_server, capsys
):
    _srv, base = stub_server(health_payload=BUSY_HEALTH, hang_cmd=True)
    cli_g.server = base
    resp = client.dispatch("snapshot", {})
    assert resp.ok is False
    assert resp.summary["error_type"] == "busy"
    # names the BUSIEST session (packager @ 512s, not default @ 34s) + queue depth,
    # in the message AND the structured summary
    assert resp.summary["busy_session"] == "packager"
    assert resp.summary["queued"] == 2
    assert resp.summary["outcome"] == "unknown"
    message = resp.issues[0].message
    assert "daemon busy: save running for 512s in session packager" in message
    assert "+2 queued" in message
    # v5 review #12: the timed-out request is NOT cancelled — it may still execute on
    # the daemon, so the outcome is UNKNOWN and a blind retry of a mutating command
    # could run it twice. No "retry later" advice, ever.
    assert "retry later" not in message
    assert "outcome UNKNOWN" in message
    assert "may still complete on the daemon" in message
    assert (
        "verify state (snapshot/history) before retrying mutating commands" in message
    )
    assert "--session" in message and "--timeout" in message
    assert "unreachable" not in message
    assert emit(resp) == EXIT_BUSY == 4
    capsys.readouterr()  # swallow emit's output


def test_timeout_with_unverified_401_health_is_not_busy(cli_g, stub_server, capsys):
    # v5 review #21: /health answering 401 proves nothing about OUR daemon — the
    # listener could be an unrelated process on the same port (see the exact-endpoint
    # tests below). "busy" (exit 4) requires a verified 200 {"ok": true}; anything
    # else is an outcome-unknown transport failure (exit 3), never a busy claim.
    _srv, base = stub_server(health_payload=None, hang_cmd=True)
    cli_g.server = base
    resp = client.dispatch("snapshot", {})
    assert resp.ok is False
    assert resp.summary["error_type"] == "transport"
    assert resp.summary["outcome"] == "unknown"
    message = resp.issues[0].message
    assert "busy" not in message
    assert "outcome UNKNOWN" in message
    assert "verify state (snapshot/history) before retrying" in message
    assert emit(resp) == EXIT_UNREACHABLE
    capsys.readouterr()


def test_timeout_with_health_lacking_ok_true_is_not_busy(cli_g, stub_server, capsys):
    # A 200 whose body does not claim {"ok": true} (wrong service, error page JSON)
    # is equally unverified: not busy.
    _srv, base = stub_server(health_payload={"status": "up"}, hang_cmd=True)
    cli_g.server = base
    resp = client.dispatch("snapshot", {})
    assert resp.ok is False
    assert resp.summary["error_type"] == "transport"
    assert resp.summary["outcome"] == "unknown"
    assert emit(resp) == EXIT_UNREACHABLE
    capsys.readouterr()


def test_timeout_against_dead_socket_is_unreachable_with_exit_3(cli_g, dead_socket):
    cli_g.server = dead_socket
    resp = client.dispatch("snapshot", {})
    assert resp.ok is False
    assert resp.summary["error_type"] == "transport"
    message = resp.issues[0].message
    assert "unreachable" in message
    # honest even here: the read timed out AFTER the request was sent, so the
    # outcome is unknown — the daemon may be wedged, not dead
    assert resp.summary["outcome"] == "unknown"
    assert emit(resp) == EXIT_UNREACHABLE == 3


def test_only_read_timeouts_are_classified_potentially_busy(cli_g, stub_server, capsys):
    # v5 review #21: a Write/PoolTimeout may never have delivered the request — it is
    # an outcome-unknown transport error (exit 3), and the /health busy probe must
    # not even run. ConnectTimeout never sent anything: plain unreachable.
    import httpx

    srv, base = stub_server(health_payload=BUSY_HEALTH)
    for exc in (httpx.WriteTimeout("w"), httpx.PoolTimeout("p")):
        resp = client._transport_error("scatter", base, None, exc, 1.0)
        assert resp.summary["error_type"] == "transport"
        assert resp.summary["outcome"] == "unknown"
        assert "outcome" in resp.issues[0].message
        assert "busy" not in resp.issues[0].message
        assert emit(resp) == EXIT_UNREACHABLE
    resp = client._transport_error(
        "scatter", base, None, httpx.ConnectTimeout("c"), 1.0
    )
    assert resp.summary["error_type"] == "transport"
    assert emit(resp) == EXIT_UNREACHABLE
    assert srv.requests == []  # no /health probe for any non-read timeout

    # ONLY a ReadTimeout (request sent, daemon never answered) probes /health, and a
    # verified answer yields the busy classification.
    resp = client._transport_error("scatter", base, None, httpx.ReadTimeout("r"), 1.0)
    assert resp.summary["error_type"] == "busy"
    assert [r[:2] for r in srv.requests] == [("GET", "/health")]
    assert emit(resp) == EXIT_BUSY
    capsys.readouterr()


def test_connection_refused_is_unreachable(cli_g):
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()  # nothing listens here anymore
    cli_g.server = f"http://127.0.0.1:{port}"
    resp = client.dispatch("snapshot", {})
    assert resp.summary["error_type"] == "transport"
    assert emit(resp) == EXIT_UNREACHABLE


# ── #2 /health exposes commands QUEUED per session ───────────────────────────────
def test_health_reports_queued_commands_per_session(tmp_path, monkeypatch):
    release = threading.Event()
    started = threading.Event()

    class FakeSession:
        def __init__(self, _config):
            self._stage_path = None

        def info(self):
            started.set()
            release.wait(timeout=30)
            return Response(command="info")

    monkeypatch.setattr(server_app, "Session", FakeSession)
    app = server_app.build_app(Config(project_dir=tmp_path), token="t", instance_id="i")
    with TestClient(app) as http:

        def post():
            http.post(
                "/cmd",
                headers=HEADERS,
                json={"command": "info", "payload": {}, "session": "work"},
            )

        first = threading.Thread(target=post)
        first.start()
        assert started.wait(10), "first command never started executing"
        second = threading.Thread(target=post)  # queues behind the held session lock
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
            assert work["busy"] is True
            assert work["current_command"] == "info"
            assert health["queued"] == 1  # top-level total across sessions
        finally:
            release.set()
            first.join(timeout=30)
            second.join(timeout=30)

        health = http.get("/health", headers=HEADERS).json()
        assert health["queued"] == 0
        assert health["sessions"]["work"]["queued"] == 0
        assert health["sessions"]["work"]["busy"] is False
        assert server_app._IN_FLIGHT[0] == 0  # what the idle watcher checks


def test_health_uncontended_command_never_counts_as_queued(tmp_path, monkeypatch):
    class FakeSession:
        def __init__(self, _config):
            self._stage_path = None

        def info(self):
            return Response(command="info")

    monkeypatch.setattr(server_app, "Session", FakeSession)
    app = server_app.build_app(Config(project_dir=tmp_path), token="t", instance_id="i")
    with TestClient(app) as http:
        assert http.post(
            "/cmd", headers=HEADERS, json={"command": "info", "payload": {}}
        ).json()["ok"]
        health = http.get("/health", headers=HEADERS).json()
        assert health["queued"] == 0
        assert all(entry["queued"] == 0 for entry in health["sessions"].values())


# ── #3 --server token handling ───────────────────────────────────────────────────
def test_explicit_server_reuses_the_state_files_token_on_exact_match(
    cli_g, stub_server, tmp_path, monkeypatch
):
    srv, base = stub_server(expected_token="secret-tok")
    monkeypatch.setattr(
        daemon,
        "discover",
        lambda _config: (base, "secret-tok"),
    )
    # --server names EXACTLY the state file's endpoint (http://127.0.0.1:<port>)
    cli_g.server = base
    resp = client.dispatch("snapshot", {})
    assert resp.ok is True, resp.issues
    assert srv.requests[-1] == ("POST", "/cmd", "secret-tok")


def test_loopback_aliases_are_not_folded_for_token_reuse(
    cli_g, stub_server, tmp_path, monkeypatch
):
    # v5 review #14 (security): localhost / ::1 / 127.0.0.1 on one port can be
    # DIFFERENT listeners (separate processes bind IPv4 vs IPv6), so the stored token
    # is only sent on an exact scheme+host+port match with the state file's URL.
    srv, base = stub_server(expected_token="secret-tok")
    monkeypatch.setattr(
        daemon,
        "discover",
        lambda _config: (base, "secret-tok"),
    )
    cli_g.server = f"http://localhost:{base.rsplit(':', 1)[1]}"
    resp = client.dispatch("snapshot", {})
    assert resp.ok is False
    assert resp.summary["error_type"] == "authentication"
    assert srv.requests[-1][2] is None  # the stored token was never sent


def test_server_token_never_inherits_copied_foreign_project_state(
    tmp_path, monkeypatch
):
    project_a = Config(project_dir=tmp_path / "project-a")
    project_b = Config(project_dir=tmp_path / "project-b")
    project_b.state_dir.mkdir(parents=True)
    state = {
        "pid": 4242,
        "process_start_token": "t-owned",
        "host": "127.0.0.1",
        "port": 4567,
        "token": "must-not-leak",
        "instance_id": "instance-a",
        "project_id": daemon._project_id(project_a),
    }
    project_b.server_state_path.write_text(json.dumps(state), encoding="utf-8")
    (project_b.state_dir / daemon.DAEMON_LEDGER_NAME).write_text(
        "4242:t-owned\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: "t-owned")
    monkeypatch.setattr(
        daemon,
        "_health",
        lambda *_args, **_kwargs: pytest.fail("foreign state must not be probed"),
    )

    assert client._server_token(project_b, "http://127.0.0.1:4567") is None


def test_same_endpoint_requires_exact_scheme_host_port():
    same = client._same_endpoint
    assert same("http://127.0.0.1:8000", "http://127.0.0.1:8000") is True
    assert same("127.0.0.1:8000", "http://127.0.0.1:8000") is True  # scheme defaults
    # no loopback-alias folding, in any direction
    assert same("http://localhost:8000", "http://127.0.0.1:8000") is False
    assert same("http://[::1]:8000", "http://127.0.0.1:8000") is False
    assert same("http://[::1]:8000", "http://localhost:8000") is False
    # scheme and port are part of the identity too
    assert same("https://127.0.0.1:8000", "http://127.0.0.1:8000") is False
    assert same("http://127.0.0.1:8000", "http://127.0.0.1:8001") is False


def _write_attached_state(project_dir, *, port: int = 4567, token: str = "attached"):
    config = Config(project_dir=project_dir)
    config.state_dir.mkdir(parents=True)
    state = {
        "pid": 4242,
        "process_start_token": "t-parent",
        "host": "127.0.0.1",
        "port": port,
        "token": token,
        "instance_id": "parent-instance",
        "project_id": daemon._project_id(config),
        "lifecycle_owner": "external",
    }
    config.server_state_path.write_text(json.dumps(state), encoding="utf-8")
    return config, state


def test_attached_discovery_authenticates_state_without_host_pid_visibility(
    tmp_path, monkeypatch
):
    config, state = _write_attached_state(tmp_path)
    base = "http://127.0.0.1:4567"
    observed: list[tuple[str, str | None, str | None]] = []

    def health(url, token, timeout=2.0, *, proxy=None):
        observed.append((url, token, proxy))
        return {
            "ok": True,
            **{
                field: state[field]
                for field in (
                    "pid",
                    "process_start_token",
                    "project_id",
                    "instance_id",
                    "lifecycle_owner",
                )
            },
        }

    monkeypatch.setattr(daemon, "_health", health)
    monkeypatch.setattr(
        daemon,
        "_read_ledger",
        lambda _config: pytest.fail("attached discovery must not inspect host PIDs"),
    )

    assert daemon.discover_attached(config, base) == (base, "attached")
    assert observed == [(base, "attached", None)]


def test_attached_discovery_routes_health_through_codex_proxy(
    tmp_path, monkeypatch
):
    config, state = _write_attached_state(tmp_path)
    base = "http://127.0.0.1:4567"
    observed: list[str | None] = []

    monkeypatch.setenv(daemon.PARENT_USD_CLI_PROXY_ALLOWED_ENV, "1")
    monkeypatch.setenv(daemon.ATTACHED_PROJECT_DIR_ENV, str(tmp_path.resolve()))
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:32123")

    def health(_url, _token, timeout=2.0, *, proxy=None):
        observed.append(proxy)
        return {
            "ok": True,
            **{
                field: state[field]
                for field in (
                    "pid",
                    "process_start_token",
                    "project_id",
                    "instance_id",
                    "lifecycle_owner",
                )
            },
        }

    monkeypatch.setattr(daemon, "_health", health)

    assert daemon.discover_attached(config, base) == (base, "attached")
    assert observed == ["http://127.0.0.1:32123"]


def test_status_uses_attached_discovery_without_host_pid_visibility(
    tmp_path, monkeypatch
):
    attached = tmp_path / "parent-run"
    attached.mkdir()
    _config, state = _write_attached_state(attached)
    unrelated = tmp_path / "child-cwd"
    unrelated.mkdir()
    base = "http://127.0.0.1:4567"

    monkeypatch.setenv(daemon.ATTACHED_PROJECT_DIR_ENV, str(attached.resolve()))
    monkeypatch.setattr(
        daemon,
        "_proc_start_token",
        lambda _pid: pytest.fail("attached status must not inspect host PIDs"),
    )
    monkeypatch.setattr(
        daemon,
        "_health",
        lambda _url, _token, timeout=2.0, *, proxy=None: {
            "ok": True,
            **{
                field: state[field]
                for field in (
                    "pid",
                    "process_start_token",
                    "project_id",
                    "instance_id",
                    "lifecycle_owner",
                )
            },
        },
    )

    assert daemon.status(Config(project_dir=unrelated)) == (
        f"running (attached parent) — pid 4242, {base}, lifecycle external"
    )


def test_status_fails_closed_for_unverified_attached_parent(tmp_path, monkeypatch):
    attached = tmp_path / "parent-run"
    attached.mkdir()
    _write_attached_state(attached)
    unrelated = tmp_path / "child-cwd"
    unrelated.mkdir()

    monkeypatch.setenv(daemon.ATTACHED_PROJECT_DIR_ENV, str(attached.resolve()))
    monkeypatch.setattr(daemon, "_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        daemon,
        "_proc_start_token",
        lambda _pid: pytest.fail("unverified attachment must not inspect host PIDs"),
    )

    assert daemon.status(Config(project_dir=unrelated)) == (
        "unverified attached parent daemon state for pid 4242 (no signal sent)"
    )


def test_attached_discovery_never_sends_token_to_a_different_endpoint(
    tmp_path, monkeypatch
):
    config, _state = _write_attached_state(tmp_path)
    monkeypatch.setattr(
        daemon,
        "_health",
        lambda *_args, **_kwargs: pytest.fail("mismatched endpoint must not be probed"),
    )

    assert (
        daemon.discover_attached(config, "http://127.0.0.1:4568") is None
    )
    assert daemon.discover_attached(config, "http://localhost:4567") is None


def test_attached_discovery_rejects_health_identity_change(tmp_path, monkeypatch):
    config, state = _write_attached_state(tmp_path)
    monkeypatch.setattr(
        daemon,
        "_health",
        lambda *_args, **_kwargs: {
            "ok": True,
            "pid": state["pid"],
            "process_start_token": state["process_start_token"],
            "project_id": state["project_id"],
            "instance_id": "replacement-instance",
            "lifecycle_owner": state["lifecycle_owner"],
        },
    )

    assert (
        daemon.discover_attached(config, "http://127.0.0.1:4567") is None
    )


def test_server_token_uses_canonical_attached_project_hint(tmp_path, monkeypatch):
    attached = tmp_path / "parent-run"
    attached.mkdir()
    base = "http://127.0.0.1:4567"
    observed: list[tuple[Config, str]] = []
    monkeypatch.setattr(daemon, "discover", lambda _config: None)

    def discover_attached(config, expected_base):
        observed.append((config, expected_base))
        return base, "attached-token"

    monkeypatch.setattr(daemon, "discover_attached", discover_attached)
    monkeypatch.setenv(client.ATTACHED_PROJECT_DIR_ENV, str(attached.resolve()))

    assert client._server_token(Config(project_dir=tmp_path), base) == "attached-token"
    assert observed[0][0].project_dir == attached.resolve()
    assert observed[0][1] == base


def test_server_token_rejects_noncanonical_attached_project_hint(
    tmp_path, monkeypatch
):
    attached = tmp_path / "parent-run"
    attached.mkdir()
    monkeypatch.setattr(daemon, "discover", lambda _config: None)
    monkeypatch.setattr(
        daemon,
        "discover_attached",
        lambda *_args, **_kwargs: pytest.fail("noncanonical hint must not be used"),
    )
    monkeypatch.setenv(
        client.ATTACHED_PROJECT_DIR_ENV,
        str(attached / ".." / attached.name),
    )

    assert (
        client._server_token(
            Config(project_dir=tmp_path), "http://127.0.0.1:4567"
        )
        is None
    )


def test_ov_token_env_overrides_the_state_file(
    cli_g, stub_server, tmp_path, monkeypatch
):
    srv, base = stub_server(expected_token="env-tok")
    _write_state(tmp_path, port=int(base.rsplit(":", 1)[1]), token="stale-tok")
    monkeypatch.setenv("USD_CLI_TOKEN", "env-tok")
    cli_g.server = base
    resp = client.dispatch("snapshot", {})
    assert resp.ok is True, resp.issues
    assert srv.requests[-1][2] == "env-tok"


def test_scoped_parent_token_authenticates_repeated_scene_reads(
    cli_g, stub_server, tmp_path, monkeypatch
):
    srv, base = stub_server(expected_token="scoped-parent-token")
    monkeypatch.setattr(daemon, "discover", lambda _config: None)
    monkeypatch.setenv(client.PARENT_USD_CLI_TOKEN_ENV, "scoped-parent-token")
    # The launcher and explicit --server route may differ only by a trailing slash.
    monkeypatch.setenv(client.PARENT_USD_CLI_SERVER_URL_ENV, f"{base}/")
    monkeypatch.setenv(client.PARENT_USD_CLI_MANAGED_ENV, "1")
    monkeypatch.setenv(client.USD_CLI_EXTERNAL_LIFECYCLE_ENV, "1")
    monkeypatch.setenv("USD_CLI_NO_DAEMON", "1")
    cli_g.server = base

    first = client.dispatch("snapshot", {})
    second = client.dispatch("snapshot", {})

    assert first.ok is True, first.issues
    assert second.ok is True, second.issues
    assert [request[2] for request in srv.requests[-2:]] == [
        "scoped-parent-token",
        "scoped-parent-token",
    ]


@pytest.mark.parametrize("failure", ["wrong-token", "wrong-endpoint"])
def test_scoped_parent_token_rejects_unauthorized_child(
    cli_g, stub_server, tmp_path, monkeypatch, failure
):
    srv, base = stub_server(expected_token="right-parent-token")
    monkeypatch.setattr(daemon, "discover", lambda _config: None)
    monkeypatch.setenv(
        client.PARENT_USD_CLI_TOKEN_ENV,
        "wrong-parent-token" if failure == "wrong-token" else "right-parent-token",
    )
    expected_base = (
        base
        if failure == "wrong-token"
        else f"http://127.0.0.1:{int(base.rsplit(':', 1)[1]) + 1}"
    )
    monkeypatch.setenv(client.PARENT_USD_CLI_SERVER_URL_ENV, expected_base)
    monkeypatch.setenv(client.PARENT_USD_CLI_MANAGED_ENV, "1")
    monkeypatch.setenv(client.USD_CLI_EXTERNAL_LIFECYCLE_ENV, "1")
    monkeypatch.setenv("USD_CLI_NO_DAEMON", "1")
    cli_g.server = base

    response = client.dispatch("snapshot", {})

    assert response.ok is False
    assert response.summary["error_type"] == "authentication"
    assert srv.requests[-1][2] == (
        "wrong-parent-token" if failure == "wrong-token" else None
    )


def test_state_token_is_not_reused_for_a_different_daemon(cli_g, stub_server, tmp_path):
    srv, base = stub_server(expected_token="secret-tok")
    _write_state(tmp_path, port=59999, token="secret-tok")  # some OTHER daemon's state
    cli_g.server = base
    resp = client.dispatch("snapshot", {})
    assert resp.ok is False  # no token sent → the stub 401s
    assert resp.summary["error_type"] == "authentication"
    assert srv.requests[-1][2] is None


def test_401_explains_the_token_recourse(cli_g, stub_server):
    _srv, base = stub_server(expected_token="right")
    cli_g.server = base  # no state file, no env → request goes out tokenless
    resp = client.dispatch("snapshot", {})
    assert resp.ok is False
    assert resp.summary["error_type"] == "authentication"
    message = resp.issues[0].message
    assert "401" in message
    assert "USD_CLI_TOKEN" in message
    assert "omit --server to auto-discover" in message
    assert emit(resp) == EXIT_UNREACHABLE
