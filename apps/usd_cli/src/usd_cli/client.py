# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discovery + dispatch: find (or auto-start) the daemon, POST the command.

By default the CLI auto-starts a per-project daemon on first command (PR-9.4). Set
`USD_CLI_NO_DAEMON=1` suppresses auto-start and fails the request closed. `--server
<url>` targets a daemon directly and never auto-starts.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit

from usd_core.config import Config, load_config
from usd_core.models import Issue, Response
from usd_cli.state import G


# Commands that can legitimately take much longer than a snapshot — a GPU render (esp.
# OVRTX cold start + path tracing), a multi-frame animation, or a physics simulation
# routinely exceed the snappy default timeout.
_SLOW_COMMANDS = {
    "render",
    "render-frames",
    "render-probe",
    "physics.simulate",
    "camera.coverage",
    "camera.place",
    "camera.rig-export",
    "wait",
}
_SLOW_TIMEOUT_S = 1800.0
# Heavy-but-bounded commands: opening/saving/checkpointing a multi-10s-of-MB
# stage legitimately exceeds a snappy default (round 7: a 53 MB open trickled
# past 30s and returned outcome-UNKNOWN noise).
_MEDIUM_COMMANDS = {"open", "save", "export", "convert", "checkpoint.save",
                    # Empty-space detection: the first query in a daemon's life pays
                    # Warp's JIT compile of the heightfield/extract kernels (seconds,
                    # once), then ingests and rasterizes the scope. Bounded, but well
                    # past a snapshot's budget.
                    "space.free", "space.support"}
_MEDIUM_TIMEOUT_S = 300.0
# Fast-fail ceilings so the generous slow-command floor never hides a dead daemon: TCP
# connect is always quick, and before committing to a long read we prove the daemon can
# answer *anything* at all (its lock-free liveness route) within a snappy window.
_CONNECT_TIMEOUT_S = 10.0
_PROBE_TIMEOUT_S = 5.0
# After a /cmd READ timeout, one quick look at the lock-free /health tells busy from
# dead. Reporting a busy daemon as "unreachable" made agents restart the server and
# kill in-flight renders (v4 benchmark) — but "busy" requires a VERIFIED answer
# (200 + {"ok": true}); any other response could be an unrelated listener.
_BUSY_PROBE_TIMEOUT_S = 2.0
ATTACHED_PROJECT_DIR_ENV = "USD_CLI_ATTACHED_PROJECT_DIR"
PARENT_USD_CLI_MANAGED_ENV = "CONTENT_WORKFLOW_PARENT_USD_CLI_MANAGED"
PARENT_USD_CLI_SERVER_URL_ENV = "CONTENT_WORKFLOW_USD_CLI_SERVER_URL"
PARENT_USD_CLI_TOKEN_ENV = "CONTENT_WORKFLOW_PARENT_USD_CLI_TOKEN"
PARENT_USD_CLI_AUTH_PROXY_ENV = "CONTENT_WORKFLOW_PARENT_USD_CLI_AUTH_PROXY"
USD_CLI_EXTERNAL_LIFECYCLE_ENV = "USD_CLI_LIFECYCLE_EXTERNALLY_OWNED"


def _trust_proxy_environment(url: str) -> bool:
    """Keep ordinary remote proxy behavior outside parent-attached mode."""

    if os.environ.get(ATTACHED_PROJECT_DIR_ENV):
        return False

    try:
        host = urlsplit(url if "://" in url else f"http://{url}").hostname
    except ValueError:
        return True
    if host is None:
        return True
    if host.rstrip(".").lower() == "localhost":
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


def _attached_parent_http_proxy_for_url(url: str) -> str | None:
    """Select the only proxy route authorized for parent daemon attachment."""

    auth_proxy = _managed_parent_auth_proxy_for_url(url)
    if auth_proxy is not None:
        return auth_proxy

    trust_env = _trust_proxy_environment(url)
    if not trust_env:
        from usd_cli import daemon

        proxy = daemon.attached_parent_http_proxy()
        attached_path = _attached_parent_project_path()
        if not isinstance(proxy, str) or attached_path is None:
            return None
        try:
            attached = daemon.discover_attached(
                Config(project_dir=attached_path),
                url,
            )
        except (OSError, RuntimeError, ValueError):
            return None
        if attached is not None:
            return proxy
    return None


def _managed_parent_auth_proxy_for_url(url: str) -> str | None:
    """Return the launcher broker only for its exact managed parent endpoint."""

    proxy = os.environ.get(PARENT_USD_CLI_AUTH_PROXY_ENV)
    expected_base = os.environ.get(PARENT_USD_CLI_SERVER_URL_ENV)
    if (
        not proxy
        or not expected_base
        or os.environ.get(PARENT_USD_CLI_MANAGED_ENV) != "1"
        or os.environ.get(USD_CLI_EXTERNAL_LIFECYCLE_ENV) != "1"
        or os.environ.get("USD_CLI_NO_DAEMON") != "1"
    ):
        return None
    try:
        parsed_proxy = urlsplit(proxy)
        proxy_port = parsed_proxy.port
        parsed_expected = urlsplit(expected_base)
        expected_port = parsed_expected.port
        parsed_target = urlsplit(url)
        target_port = parsed_target.port
    except ValueError:
        return None
    if (
        parsed_proxy.scheme != "http"
        or parsed_proxy.hostname != "127.0.0.1"
        or proxy_port is None
        or not 1 <= proxy_port <= 65535
        or parsed_proxy.username is not None
        or parsed_proxy.password is not None
        or parsed_proxy.path not in {"", "/"}
        or parsed_proxy.query
        or parsed_proxy.fragment
        or parsed_expected.scheme != "http"
        or parsed_expected.hostname != "127.0.0.1"
        or expected_port is None
        or parsed_expected.username is not None
        or parsed_expected.password is not None
        or parsed_expected.path not in {"", "/"}
        or parsed_expected.query
        or parsed_expected.fragment
        or parsed_target.scheme != parsed_expected.scheme
        or parsed_target.hostname != parsed_expected.hostname
        or target_port != expected_port
    ):
        return None
    return proxy


def _attached_parent_project_path() -> Path | None:
    """Return the canonical parent project explicitly named by the launcher."""

    attached_project = os.environ.get(ATTACHED_PROJECT_DIR_ENV)
    if not attached_project:
        return None
    try:
        attached_path = Path(attached_project)
        if not attached_path.is_absolute():
            return None
        attached_path = attached_path.resolve(strict=True)
    except OSError:
        return None
    if str(attached_path) != attached_project or not attached_path.is_dir():
        return None
    return attached_path


def _effective_timeout(command: str | None) -> float:
    """The wall-time ceiling for one command.

    An explicit --timeout is a hard ceiling for every command. Only when the user left it at
    the default do slow commands (GPU render, physics) get the generous floor so a snappy 30s
    default doesn't abort a legitimate render.
    """
    if G.timeout_explicit:
        return G.timeout
    if command in _SLOW_COMMANDS:
        return max(G.timeout, _SLOW_TIMEOUT_S)
    if command in _MEDIUM_COMMANDS:
        return max(G.timeout, _MEDIUM_TIMEOUT_S)
    return G.timeout


def _post(url: str, token: str | None, body: dict, timeout_s: float) -> Response:
    import httpx  # lazy: keeps `--help` fast

    # all header generations: a pre-rename daemon only reads x-ov-token/x-3dsc-token
    headers = (
        {"x-usd-cli-token": token, "x-ov-token": token, "x-3dsc-token": token}
        if token
        else {}
    )
    # The full timeout budget applies to the read (a long render legitimately blocks
    # there); connecting must never take that long, so a dead daemon fails fast.
    proxy = _attached_parent_http_proxy_for_url(url)
    http_client = (
        httpx.Client(proxy=proxy, trust_env=False)
        if proxy is not None
        else httpx.Client(trust_env=_trust_proxy_environment(url))
    )
    with http_client as client:
        resp = client.post(
            url,
            json=body,
            headers=headers,
            timeout=httpx.Timeout(
                timeout_s,
                connect=min(timeout_s, _CONNECT_TIMEOUT_S),
            ),
        )
    resp.raise_for_status()
    return Response.from_dict(resp.json())


def _probe_alive(base: str, timeout_s: float) -> str | None:
    """None when the daemon answers its liveness route quickly, else the failure reason.

    Any HTTP response (even a 404 from an older daemon without /live) proves the server
    is alive and answering — only transport-level failures raise.
    """
    import httpx  # lazy: keeps `--help` fast

    try:
        url = f"{base}/live"
        proxy = _attached_parent_http_proxy_for_url(url)
        http_client = (
            httpx.Client(proxy=proxy, trust_env=False)
            if proxy is not None
            else httpx.Client(trust_env=_trust_proxy_environment(url))
        )
        with http_client as client:
            client.get(url, timeout=timeout_s)
        return None
    except Exception as exc:  # noqa: BLE001 — any transport failure means "not answering"
        return str(exc) or type(exc).__name__


def _same_endpoint(a: str, b: str) -> bool:
    """True only for an EXACT scheme+host+port match — no loopback-alias folding.

    `::1`, `localhost`, and `127.0.0.1` are NOT one origin: separate processes can
    bind IPv4 and IPv6 on the same port, so folding aliases could send the stored
    daemon token to an unrelated listener. `--server` must match the state file's
    URL exactly to inherit its token; otherwise pass the token via USD_CLI_TOKEN."""
    from urllib.parse import urlsplit

    def parts(url: str) -> tuple[str, str, int | None]:
        s = urlsplit(url if "://" in url else f"http://{url}")
        return (s.scheme or "http", s.hostname or "", s.port)

    return parts(a) == parts(b)


def _scoped_parent_token(base: str) -> str | None:
    """Return a workflow-provisioned token only for its exact parent endpoint."""

    token = os.environ.get(PARENT_USD_CLI_TOKEN_ENV)
    expected_base = os.environ.get(PARENT_USD_CLI_SERVER_URL_ENV)
    if (
        not token
        or os.environ.get(PARENT_USD_CLI_MANAGED_ENV) != "1"
        or os.environ.get(USD_CLI_EXTERNAL_LIFECYCLE_ENV) != "1"
        or os.environ.get("USD_CLI_NO_DAEMON") != "1"
        or not expected_base
    ):
        return None
    try:
        parsed = urlsplit(expected_base)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or base.rstrip("/") != expected_base.rstrip("/")
    ):
        return None
    return token


def _server_token(config: Config, base: str) -> str | None:
    """The token to use with an explicit --server URL.

    Precedence: USD_CLI_TOKEN (explicit override, documented in the 401 message) >
    USD_CLI_SERVER_TOKEN (legacy) > a launcher-provisioned parent token bound to its
    exact attested endpoint > ordinary project discovery > an externally owned parent
    project named by USD_CLI_ATTACHED_PROJECT_DIR. State-file credentials are reused
    only when --server is EXACTLY the state file's endpoint (scheme+host+port; loopback
    aliases like localhost vs 127.0.0.1 vs ::1 do NOT fold — see _same_endpoint). The
    attached path additionally authenticates health identity so sandbox PID namespaces
    do not require visibility into the host's ``/proc``.
    """
    # OV_*/3DSC_* are the pre-rename names — honored so existing environments keep working.
    token = (
        os.environ.get("USD_CLI_TOKEN")
        or os.environ.get("USD_CLI_SERVER_TOKEN")
        or os.environ.get("OV_TOKEN")
        or os.environ.get("OV_SERVER_TOKEN")
        or os.environ.get("3DSC_TOKEN")
        or os.environ.get("3DSC_SERVER_TOKEN")
    )
    if token:
        return token
    if (parent_token := _scoped_parent_token(base)) is not None:
        return parent_token
    from usd_cli import daemon

    discovered = daemon.discover(config)
    if discovered is not None and _same_endpoint(discovered[0], base):
        return discovered[1]

    # Parent-owned workflow daemons live under a run directory that is normally
    # different from the coding agent's cwd.  The child may also have a different
    # PID namespace, so ordinary discovery cannot authenticate the host PID.  A
    # canonical, non-secret project hint enables the daemon's attached-discovery
    # proof without placing its token in the child environment.
    attached_path = _attached_parent_project_path()
    if attached_path is None:
        return None
    attached = daemon.discover_attached(
        Config(project_dir=attached_path),
        base,
    )
    if attached is not None and _same_endpoint(attached[0], base):
        return attached[1]
    return None


def _fetch_health(base: str, token: str | None, timeout_s: float) -> dict | None:
    """GET the daemon's lock-free /health; the parsed body only for a VERIFIED answer
    (HTTP 200 whose JSON object claims {"ok": true}), else None. A 401/404 or an
    unparsable body proves nothing about OUR daemon — it may be an unrelated listener
    on the same port (see _same_endpoint) — so it must never be treated as 'busy'."""
    import httpx  # lazy: keeps `--help` fast

    try:
        url = f"{base}/health"
        proxy = _attached_parent_http_proxy_for_url(url)
        http_client = (
            httpx.Client(proxy=proxy, trust_env=False)
            if proxy is not None
            else httpx.Client(trust_env=_trust_proxy_environment(url))
        )
        with http_client as client:
            resp = client.get(
                url,
                headers=(
                    {
                        "x-usd-cli-token": token,
                        "x-ov-token": token,
                        "x-3dsc-token": token,
                    }
                    if token
                    else {}
                ),
                timeout=timeout_s,
            )
    except Exception:  # noqa: BLE001 — transport failure means "no verified answer"
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict) or data.get("ok") is not True:
        return None
    return data


def _busiest_session(health: dict) -> tuple[str, str | None, float, int] | None:
    """(session, command, busy_seconds, queued) of the longest-running in-flight
    command, from /health's per-session map (falling back to the legacy top-level
    fields for an older daemon); None when nothing is reported busy."""
    sessions = health.get("sessions") or {}
    busy = [
        (
            name,
            entry.get("current_command"),
            float(entry.get("busy_seconds") or 0.0),
            int(entry.get("queued") or 0),
        )
        for name, entry in sessions.items()
        if isinstance(entry, dict) and entry.get("busy")
    ]
    if busy:
        return max(busy, key=lambda row: row[2])
    if health.get("busy"):
        return (
            "default",
            health.get("current_command"),
            float(health.get("busy_seconds") or 0.0),
            0,
        )
    return None


def _refusal_detail(response: object) -> str | None:
    """FastAPI's {"detail": ...} from a refusal, or None if the body isn't one.

    The daemon returns the actual reason here (which allowed_roots the path fell
    outside of, which bound was exceeded); httpx's stringified HTTPStatusError
    carries only the status line."""
    try:
        payload = response.json()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — a non-JSON error body proves nothing
        return None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()[:500]
    return None


def _transport_error(
    command: str, base: str, token: str | None, exc: Exception, timeout_s: float
) -> Response:
    """Classify a failed POST honestly: auth, refusal, busy-but-alive, or transport.

    A 4xx the daemon itself emits is a REFUSAL, not a transport failure: it proves
    the daemon is reachable and answering, and its body carries the reason. That
    is error_type "policy" (exit code 1 — the command failed), never "transport"
    (exit code 3 — which reads as "the daemon is down, restart it").

    A /cmd READ timeout means the request reached the daemon and — since the daemon
    does not cancel work on disconnect — may STILL complete there. The outcome is
    UNKNOWN, so the error must say to verify state instead of recommending a blind
    retry: retrying `scatter`, a relative `transform`, or any other mutation could
    execute it twice. (Daemon-side request-id idempotency is deferred; until then
    "verify before retrying a mutation" is the contract.)

    'Busy' (error_type "busy", exit code 4) is claimed only on a VERIFIED /health
    answer (200 + {"ok": true}); a 401/404/bad-JSON answer could be an unrelated
    listener and proves nothing. Only a ReadTimeout is potentially-busy: connect/
    write/pool timeouts may never have delivered the request at all and surface as
    outcome-honest transport errors (exit code 3) without probing /health."""
    import httpx  # lazy: keeps `--help` fast

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status in {401, 403}:
        return Response(
            command=command,
            ok=False,
            summary={"error_type": "authentication"},
            issues=[
                Issue(
                    "error",
                    f"daemon at {base} rejected the request ({status} "
                    "unauthorized) — pass the daemon token via USD_CLI_TOKEN or "
                    "omit --server to auto-discover",
                )
            ],
        )
    if status in {400, 413, 422}:
        # The daemon answered and REFUSED — request policy (paths outside
        # server.allowed_roots, resource bounds, oversized body, malformed
        # batch). Reporting that as "unreachable" hid the only detail that
        # explains it and pushed agents into restarting a healthy daemon.
        # Deliberately NOT every 4xx: a 404/405 is more likely an unrelated
        # listener on the port than our daemon refusing, and that IS a
        # transport-level problem.
        return Response(
            command=command,
            ok=False,
            summary={"error_type": "policy", "status": status},
            issues=[
                Issue(
                    "error",
                    f"daemon at {base} refused '{command}' ({status}): "
                    f"{_refusal_detail(response) or exc}",
                )
            ],
        )
    if isinstance(exc, httpx.ReadTimeout):
        # The request was sent; the daemon just never answered in time. Only a
        # verified /health answer distinguishes busy from dead.
        health = _fetch_health(base, token, _BUSY_PROBE_TIMEOUT_S)
        if health is not None:  # verified 200 {"ok": true} — OUR daemon is alive
            busiest = _busiest_session(health)
            summary: dict = {"error_type": "busy", "outcome": "unknown"}
            if busiest:
                name, current, seconds, queued = busiest
                extra = f" (+{queued} queued)" if queued else ""
                message = (
                    f"daemon busy: {current or 'a command'} running for "
                    f"{seconds:.0f}s in session {name}{extra} — '{command}' "
                    "timed out client-side but may still complete on the "
                    "daemon (outcome UNKNOWN); verify state (snapshot/"
                    "history) before retrying mutating commands, use another "
                    "--session, or raise --timeout"
                )
                summary.update(
                    {
                        "busy_session": name,
                        "current_command": current,
                        "busy_seconds": seconds,
                        "queued": queued,
                    }
                )
            else:
                message = (
                    f"daemon at {base} is alive but '{command}' did not "
                    f"finish within {timeout_s:.0f}s — outcome UNKNOWN: the "
                    "timed-out command may still complete on the daemon; "
                    "verify state (snapshot/history) before retrying "
                    "mutating commands, use another --session, or raise "
                    "--timeout"
                )
            return Response(
                command=command,
                ok=False,
                summary=summary,
                issues=[Issue("error", message)],
            )
        # /health silent — but if the daemon PID is verifiably alive, this is a
        # GIL-starved daemon mid-heavy-operation (packaging/flattening a big
        # stage blocks the whole process), NOT a dead one. Round 7: agents read
        # "unreachable" here and killed healthy daemons mid-render. Classify as
        # busy (exit 4, outcome UNKNOWN) with wait guidance instead.
        try:
            from usd_cli import daemon as _daemon

            config = load_config()
            st = _daemon._read_state(config)
            identity = _daemon._owned_process_identity(config, st) if st else None
            pid_alive = bool(
                identity
                and _daemon._proc_state(identity[0]) == "alive"
                and _same_endpoint(_daemon._base(st), base)
            )
        except Exception:  # noqa: BLE001 — no state file / not our daemon
            pid_alive = False
        if pid_alive:
            return Response(
                command=command,
                ok=False,
                summary={"error_type": "busy", "outcome": "unknown"},
                issues=[
                    Issue(
                        "error",
                        f"daemon busy: its process is alive but wholly "
                        f"occupied by a heavy operation (health starved) — "
                        f"'{command}' timed out client-side and may still "
                        "complete (outcome UNKNOWN). Wait and verify state "
                        "(snapshot/history) before retrying mutating "
                        "commands; do NOT restart the daemon mid-operation",
                    )
                ],
            )
        return Response(
            command=command,
            ok=False,
            summary={"error_type": "transport", "outcome": "unknown"},
            issues=[
                Issue(
                    "error",
                    f"daemon at {base} unreachable: '{command}' timed out and "
                    "/health gave no verified answer — outcome UNKNOWN: the "
                    "request may still complete if the daemon is alive; "
                    "verify state (snapshot/history) before retrying "
                    "mutating commands",
                )
            ],
        )
    if isinstance(exc, httpx.TimeoutException) and not isinstance(
        exc, httpx.ConnectTimeout
    ):
        # Write/pool timeout: the request body may never have reached the daemon —
        # but that is not proven, so the outcome stays UNKNOWN (and this is not
        # evidence of "busy", so /health is not probed).
        return Response(
            command=command,
            ok=False,
            summary={"error_type": "transport", "outcome": "unknown"},
            issues=[
                Issue(
                    "error",
                    f"daemon at {base} unreachable: sending '{command}' timed "
                    f"out before completing ({type(exc).__name__}) — the "
                    "request may not have reached the daemon, but the outcome "
                    "is UNKNOWN; verify state (snapshot/history) before "
                    "retrying mutating commands",
                )
            ],
        )
    if isinstance(status, int) and status >= 500:
        # The daemon ANSWERED — a 5xx is a server-side bug, not a dead daemon.
        # Round 9: inf-valued attrs 500'd JSON encoding and the "unreachable"
        # label sent the agent into connectivity triage instead of a bug report.
        return Response(
            command=command,
            ok=False,
            summary={"error_type": "server"},
            issues=[
                Issue(
                    "error",
                    f"daemon at {base} hit an internal error ({status}) "
                    f"running '{command}' — the daemon is alive; this is a "
                    "usd-cli bug worth reporting (do NOT restart the "
                    "daemon or retry mutations blindly; see "
                    ".usd-cli/daemon.log for the traceback)",
                )
            ],
        )
    return Response(
        command=command,
        ok=False,
        summary={"error_type": "transport"},
        issues=[Issue("error", f"daemon at {base} unreachable: {exc}")],
    )


def _stub(config: Config, command: str, body: dict, hint: str) -> Response:
    return Response(
        command=command,
        ok=False,
        summary={
            "stub": True,
            "error_type": "daemon-disabled",
            "engine": config.backend.get("engine"),
            "renderer": config.render.get("renderer"),
        },
        data={"would_send": body},
        issues=[
            Issue(
                "error",
                f"request was not executed; showing parsed request only — {hint}",
            )
        ],
    )


def dispatch(command: str, payload: dict) -> Response:
    """Send one command to the daemon, auto-starting it if needed."""
    config = load_config()
    body = {"command": command, "payload": payload, "session": G.session}
    token: str | None = None

    if G.server:
        base = G.server.rstrip("/")
        token = _server_token(config, base)
    elif (
        os.environ.get("USD_CLI_NO_DAEMON")
        or os.environ.get("OV_NO_DAEMON")
        or os.environ.get("3DSC_NO_DAEMON")
    ):
        return _stub(config, command, body, "USD_CLI_NO_DAEMON set (no auto-start)")
    else:
        from usd_cli import daemon

        try:
            base, token = daemon.ensure_running(config)
        except Exception as exc:  # noqa: BLE001 — startup failure must fail closed
            return Response(
                command=command,
                ok=False,
                summary={"error_type": "startup"},
                issues=[Issue("error", f"daemon startup failed: {exc}")],
            )

    timeout_s = _effective_timeout(command)
    if timeout_s > G.timeout:
        # Slow command riding the generous default floor: an unresponsive daemon would
        # otherwise leave the CLI silent for up to 30 minutes (render/save against a
        # busy daemon "succeeded" with empty output once the caller gave up waiting,
        # while snapshot's snappy timeout reported the transport error). Prove the
        # daemon answers at all before committing to the long read; a daemon that is
        # merely busy still answers /live (it takes no session lock).
        reason = _probe_alive(base, min(G.timeout, _PROBE_TIMEOUT_S))
        if reason is not None:
            return Response(
                command=command,
                ok=False,
                summary={"error_type": "transport"},
                issues=[
                    Issue(
                        "error",
                        f"daemon at {base} unreachable "
                        f"(liveness probe before '{command}'): {reason}",
                    )
                ],
            )
    try:
        return _post(f"{base}/cmd", token, body, timeout_s)
    except Exception as exc:  # noqa: BLE001 - surface transport errors cleanly
        return _transport_error(command, base, token, exc, timeout_s)
