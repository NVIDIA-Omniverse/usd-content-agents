# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Daemon lifecycle from the CLI side (PR-9.4/.5): discover, auto-start, stop, status.

Follows the standard ensure-server pattern: read the state file, health-check, take an
exclusive lock to avoid double-spawn races, spawn a detached daemon, poll until healthy.
Each project (the dir holding `.usd-cli/`) gets its own daemon on its own free port, so
agents in different directories don't collide.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import errno
import os
import secrets
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from usd_core.config import Config

START_TIMEOUT_S = 60.0  # readiness after spawn (raised again: large-scene warm-up
# under qemu emulation exceeded 20s and produced a false
# "did not come up" error while the daemon was in fact starting)
BUSY_WAIT_S = 180.0  # how long to wait out a busy (but alive) daemon before erroring
MAX_DAEMON_LEDGER_BYTES = 64 * 1024
MAX_DAEMON_STATE_BYTES = 64 * 1024


def _is_windows_breakaway_denied(error: OSError) -> bool:
    """Return whether Windows denied only the optional breakaway flag."""

    return error.winerror == 5


def _popen_windows_daemon(
    command: list[str], *, creationflags: int, **kwargs: Any
) -> subprocess.Popen[Any]:
    """Spawn a daemon without weakening its required breakaway containment."""

    try:
        return subprocess.Popen(command, creationflags=creationflags, **kwargs)
    except OSError as error:
        if not _is_windows_breakaway_denied(error):
            raise
        raise RuntimeError(
            "cannot start persistent usd-cli daemon: Windows denied job breakaway"
        ) from error
MAX_DAEMON_REGISTRATION_BYTES = 1024
MAX_SAFE_PID = 2_147_483_647
DAEMON_REGISTRATION_TIMEOUT_S = 15.0
DAEMON_LEDGER_NAME = "daemon.pids"
DAEMON_LEDGER_LOCK_NAME = "daemon.pids.lock"
STARTUP_RELEASE_FD_ENV = "USD_CLI_STARTUP_RELEASE_FD"
STARTUP_ACK_FD_ENV = "USD_CLI_STARTUP_ACK_FD"
STARTUP_NONCE_ENV = "USD_CLI_STARTUP_NONCE"
STARTUP_PROJECT_FD_ENV = "USD_CLI_STARTUP_PROJECT_FD"
STARTUP_STATE_FD_ENV = "USD_CLI_STARTUP_STATE_FD"
STARTUP_PROJECT_PATH_ENV = "USD_CLI_STARTUP_PROJECT_PATH"
STARTUP_STATE_PATH_ENV = "USD_CLI_STARTUP_STATE_PATH"
STARTUP_RENDER_CREDENTIALS_FD_ENV = "USD_CLI_STARTUP_RENDER_CREDENTIALS_FD"
EXTERNAL_LIFECYCLE_BOOTSTRAP_ENV = "USD_CLI_EXTERNAL_LIFECYCLE_BOOTSTRAP"
ATTACHED_PROJECT_DIR_ENV = "USD_CLI_ATTACHED_PROJECT_DIR"
PARENT_USD_CLI_PROXY_ALLOWED_ENV = "CONTENT_WORKFLOW_PARENT_USD_CLI_PROXY_ALLOWED"
MAX_STARTUP_RENDER_CREDENTIALS_BYTES = 64 * 1024
_LIFECYCLE_DIRECTORY_LOCKS = threading.local()


class DaemonStopRefused(RuntimeError):
    """The available evidence cannot authorize signaling a live daemon."""


@dataclass(frozen=True)
class DaemonStartupIdentity:
    """Parent-authenticated child identity plus inherited directory anchors."""

    pid: int
    process_start_token: str
    project_descriptor: int
    state_descriptor: int
    project_path: Path
    state_path: Path


@dataclass(frozen=True)
class _LifecycleAnchor:
    project_descriptor: int
    state_descriptor: int
    project_path: Path
    state_path: Path


def _project_id(config: Config) -> str:
    """Stable identity of the project whose daemon state is being inspected."""

    root = str((config.project_dir or Path.cwd()).resolve())
    return hashlib.sha256(root.encode("utf-8")).hexdigest()[:24]


def _read_state(config: Config) -> dict | None:
    if os.name == "nt":
        from usd_core.windows_files import read_confined_regular_file

        try:
            payload = read_confined_regular_file(
                config.state_dir,
                "server.json",
                max_bytes=MAX_DAEMON_STATE_BYTES,
            )
            state = json.loads(payload)
            return state if isinstance(state, dict) else None
        except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError):
            return None
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        return None
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        directory_descriptor = os.open(config.state_dir, directory_flags)
    except OSError:
        return None
    try:
        payload = _read_regular_file_at(
            directory_descriptor,
            "server.json",
            max_bytes=MAX_DAEMON_STATE_BYTES,
        )
        state = json.loads(payload)
        return state if isinstance(state, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError):
        return None
    finally:
        os.close(directory_descriptor)


def _base(state: dict) -> str:
    host = state.get("host")
    port = state.get("port")
    if not isinstance(host, str):
        raise RuntimeError("daemon state host is not a numeric loopback address")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise RuntimeError(
            "daemon state host is not a numeric loopback address"
        ) from exc
    if not address.is_loopback:
        raise RuntimeError("daemon state host is not loopback")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 < port <= 65535:
        raise RuntimeError("daemon state port is invalid")
    display_host = f"[{host}]" if address.version == 6 else str(host)
    return f"http://{display_host}:{port}"


def attached_parent_http_proxy() -> str | None:
    """Return the Codex-owned proxy only for an explicit parent attachment.

    Ordinary loopback discovery stays proxy-free so ambient proxy variables can
    never receive a project token. The workflow launcher opts in only for the
    parent session handed to a network-isolated child, and Codex injects a
    loopback HTTP proxy inside that child's namespace.
    """

    attached_project = os.environ.get(ATTACHED_PROJECT_DIR_ENV)
    if (
        os.environ.get(PARENT_USD_CLI_PROXY_ALLOWED_ENV) != "1"
        or not attached_project
    ):
        return None
    # Do not let a relative, stale, or symlink-spelling project path turn an
    # otherwise ambient proxy into an attached-session capability.  The caller
    # subsequently binds this exact directory to the authenticated state file.
    try:
        resolved_project = Path(attached_project).resolve(strict=True)
    except OSError:
        return None
    if str(resolved_project) != attached_project or not resolved_project.is_dir():
        return None
    raw_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if not raw_proxy:
        return None
    try:
        parsed = urlsplit(raw_proxy)
        host = parsed.hostname
        port = parsed.port
        address = (
            None
            if host == "localhost"
            else ipaddress.ip_address(host) if host is not None else None
        )
    except ValueError:
        return None
    if (
        parsed.scheme != "http"
        or (
            host != "localhost"
            and (address is None or not address.is_loopback)
        )
        or port is None
        or not 1 <= port <= 65535
        # Claude's SDK loopback conduit authenticates as
        # srt:<opaque-token>. Newer Claude CLI sandboxes scope that username as
        # srt.<opaque-context>:<opaque-token>. Credentials remain forbidden for
        # every other proxy namespace and must always include a password.
        or (
            (parsed.username is not None or parsed.password is not None)
            and (
                not parsed.password
                or (
                    parsed.username != "srt"
                    and not (
                        parsed.username is not None
                        and parsed.username.startswith("srt.")
                        and len(parsed.username) > len("srt.")
                    )
                )
            )
        )
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    return raw_proxy


def _health(
    base: str,
    token: str | None,
    timeout: float = 2.0,
    *,
    proxy: str | None = None,
) -> dict | None:
    """GET /health -> dict (live engine/renderer/stage), or None if unreachable."""
    import httpx

    try:
        # all header generations: a pre-rename daemon only reads x-ov-token/x-3dsc-token
        headers = (
            {"x-usd-cli-token": token, "x-ov-token": token, "x-3dsc-token": token}
            if token
            else {}
        )
        # Daemon discovery is loopback-only. Do not let caller-controlled proxy
        # variables receive the project token or impersonate local health. The
        # one explicit proxy is validated above for an attached parent session.
        http_client = (
            httpx.Client(proxy=proxy, trust_env=False)
            if proxy is not None
            else httpx.Client(trust_env=False)
        )
        with http_client as client:
            r = client.get(f"{base}/health", headers=headers, timeout=timeout)
        payload = r.json() if r.status_code == 200 else None
        if isinstance(payload, dict) and payload.get("ok") is True:
            return payload
    except Exception:  # noqa: BLE001
        return None
    return None


def is_healthy(base: str, token: str | None = None, timeout: float = 2.0) -> bool:
    return _health(base, token, timeout) is not None


def _alive(pid: int) -> bool:
    if not 1 < pid <= MAX_SAFE_PID:
        # 0/-1 address process groups and oversized values can overflow pid_t.
        return False
    if os.name == "nt":
        from usd_core.windows_files import windows_process_state

        return windows_process_state(pid) != "dead"
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM proves that a process exists but is not signalable by this user.
        return True
    except OSError:
        # Only ESRCH proves absence.  Unknown probe failures must retain lifecycle
        # evidence and must never broaden signal authority.
        return True


def _system_ps_path() -> str | None:
    """Return a trusted system ps binary; never resolve it through caller PATH."""

    return next(
        (
            candidate
            for candidate in ("/bin/ps", "/usr/bin/ps")
            if Path(candidate).is_file()
        ),
        None,
    )


def _proc_state(pid: int) -> str:
    """'alive' | 'zombie' | 'dead'. `os.kill(pid, 0)` alone cannot tell a working daemon
    from a defunct one — a zombie still "exists" — which made `server status` claim a
    long-dead daemon was busy mid-operation and made `ensure_running` wait it out."""
    if os.name == "nt":
        from usd_core.windows_files import windows_process_state

        state = windows_process_state(pid)
        return "dead" if state == "dead" else "alive"
    # Never call waitpid here: pid comes from mutable state/ledger input and could
    # name an unrelated child whose exit status belongs to the embedding process.
    # Exact Popen-owned startup paths poll/wait their own child explicitly.
    if not _alive(pid):
        return "dead"
    try:  # Linux (the containers this usually runs in): the kernel's own state field
        stat = Path(f"/proc/{pid}/stat").read_text()
        return "zombie" if stat.rpartition(")")[2].split()[0] == "Z" else "alive"
    except OSError:
        pass
    ps_path = _system_ps_path()
    if ps_path is None:
        return "alive"
    try:  # macOS / other POSIX: `ps` state code ("Z…" = zombie)
        completed = subprocess.run(
            [ps_path, "-o", "state=", "-p", str(pid)],
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            capture_output=True,
            text=True,
            timeout=5,
        )
        out = completed.stdout.strip()
        if completed.returncode != 0 or not out:
            # kill(0) above proved existence or returned an unclassified error.
            # An empty/failing ps result cannot upgrade that to proof of death.
            return "alive"
        return "zombie" if out.startswith("Z") else "alive"
    except (OSError, subprocess.SubprocessError):
        return "alive"  # cannot prove otherwise; treat as alive (the safe default)


def _proc_start_token(pid: int) -> str | None:
    """Opaque process-identity token that changes when a PID number is reused: the
    process birth time. Linux: `starttime` (field 22 of /proc/<pid>/stat, clock ticks
    since boot). macOS/other POSIX: a hash of `ps -o lstart=` (full start date-time).
    None = identity cannot be determined (no such process, or no platform source) —
    callers MUST treat None as unverifiable and never signal."""
    if not 1 < pid <= MAX_SAFE_PID:
        return None
    if os.name == "nt":
        from usd_core.windows_files import windows_process_start_token

        return windows_process_start_token(pid)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rpartition(")")[2].split()
        return f"t{fields[19]}"  # starttime is field 22; index 19 after (pid, comm)
    except (OSError, IndexError):
        pass
    ps_path = _system_ps_path()
    if ps_path is None:
        return None
    try:
        completed = subprocess.run(
            [ps_path, "-o", "lstart=", "-p", str(pid)],
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            capture_output=True,
            text=True,
            timeout=5,
        )
        out = completed.stdout.strip()
        if completed.returncode == 0 and out:
            import hashlib

            return "h" + hashlib.sha256(out.encode()).hexdigest()[:16]
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _terminate(
    pid: int,
    *,
    expected_token: str,
    term_wait_s: float = 5.0,
    kill_wait_s: float = 2.0,
) -> bool:
    """Terminate, wait, then force-stop an authenticated daemon. Returns True
    once the process is gone (or reduced to a reaped/zombie corpse). The escalation
    matters: `server restart` on a wedged daemon used to leave the old process alive
    (900 MB RSS, 18 minutes later) because nothing ever signaled it.

    pid <= 1 is always rejected: 0 and negative values signal whole process groups
    (kill(2)), and pid 1 is init. `expected_token` (a _proc_start_token) re-verifies
    process identity immediately before EACH signal — the pid can be recycled between
    our ownership proof and SIGTERM, or during the TERM grace window before escalation,
    and a bare pid number must never take out an unrelated process."""
    if not 1 < pid <= MAX_SAFE_PID or not _valid_spawn_token(expected_token):
        return False

    def _gone(deadline: float) -> bool:
        while time.time() < deadline:
            if _proc_state(pid) != "alive":
                return True
            time.sleep(0.05)
        return _proc_state(pid) != "alive"

    def _identity_status() -> str:
        current_token = _proc_start_token(pid)
        if current_token == expected_token:
            return "owned"
        if current_token is not None or _proc_state(pid) != "alive":
            return "gone"
        return "unverifiable"

    identity_status = _identity_status()
    if identity_status != "owned":
        return identity_status == "gone"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    if _gone(time.time() + term_wait_s):
        return True
    # Still alive after the grace period (wedged mid-render or ignoring SIGTERM): force
    # — but only if the pid still belongs to OUR process.
    identity_status = _identity_status()
    if identity_status != "owned":
        return identity_status == "gone"
    try:
        # Windows os.kill uses TerminateProcess for SIGTERM and exposes no
        # SIGKILL constant. The first signal is already a forced termination
        # there; retry it only after revalidating the process identity.
        force_signal = signal.SIGTERM if os.name == "nt" else signal.SIGKILL
        os.kill(pid, force_signal)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    return _gone(time.time() + kill_wait_s)


def _read_ledger(config: Config) -> list[tuple[int, str]]:
    """Read a strictly validated daemon ledger under its lifecycle lock."""

    try:
        with _locked_ledger(config, create_state=False) as directory_descriptor:
            return [
                (pid, token) for pid, token in _read_ledger_locked(directory_descriptor)
            ]
    except FileNotFoundError:
        return []


def _recorded_pids(config: Config) -> set[int]:
    """PIDs this project's CLI spawned whose recorded birth time still matches the
    process CURRENTLY holding that pid. Bare-pid (old-format) and recycled-pid
    entries are excluded: a stale pid number is not process identity, and signaling
    it kills unrelated processes."""
    try:
        entries = _read_ledger(config)
    except (OSError, RuntimeError):
        return set()
    grouped = _group_ledger_entries(entries)
    return {
        pid
        for pid, tokens in grouped.items()
        if len(tokens) == 1 and _proc_start_token(pid) == tokens[0]
    }


def _recorded_state_identity(config: Config, state: dict) -> tuple[int, str] | None:
    """Return this project's unambiguous state+ledger identity, live or dead."""

    pid = state.get("pid")
    process_start_token = state.get("process_start_token")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or not 1 < pid <= MAX_SAFE_PID
        or not isinstance(process_start_token, str)
        or not _valid_spawn_token(process_start_token)
        or state.get("project_id") != _project_id(config)
        or not isinstance(state.get("instance_id"), str)
        or not state.get("instance_id")
        or not isinstance(state.get("token"), str)
        or not state.get("token")
    ):
        return None
    try:
        recorded_identities = _group_ledger_entries(_read_ledger(config))
    except (OSError, RuntimeError):
        return None
    if recorded_identities != {pid: [process_start_token]}:
        return None
    return pid, process_start_token


def _owned_process_identity(config: Config, state: dict) -> tuple[int, str] | None:
    """Return an OS-live identity only for this config's unambiguous state+ledger."""

    identity = _recorded_state_identity(config, state)
    if identity is None:
        return None
    pid, process_start_token = identity
    if _proc_start_token(pid) != process_start_token:
        return None
    return pid, process_start_token


def _health_matches_state(config: Config, state: dict, health: dict | None) -> bool:
    """Require authenticated health to echo every current-project daemon identity."""

    identity = _owned_process_identity(config, state)
    if identity is None or not isinstance(health, dict) or health.get("ok") is not True:
        return False
    pid, process_start_token = identity
    return bool(
        health.get("pid") == pid
        and health.get("process_start_token") == process_start_token
        and health.get("project_id") == _project_id(config)
        and health.get("instance_id") == state.get("instance_id")
        and health.get("lifecycle_owner") == state.get("lifecycle_owner")
    )


def discover(
    config: Config, health_timeout: float = 2.0
) -> tuple[str, str | None] | None:
    """Return (base_url, token) for a healthy daemon for this project, else None."""
    st = _read_state(config)
    if not st:
        return None
    identity = _owned_process_identity(config, st)
    if identity is None:
        return None
    try:
        base = _base(st)
    except RuntimeError:
        return None
    health = _health(base, st.get("token"), timeout=health_timeout)
    if _health_matches_state(config, st, health):
        return base, st.get("token")
    return None


def discover_attached(
    config: Config,
    expected_base: str,
    health_timeout: float = 2.0,
) -> tuple[str, str] | None:
    """Discover a parent-owned daemon from inside a PID namespace.

    A sandboxed child cannot prove the host daemon's PID through its local
    ``/proc`` view.  Attachment therefore uses a narrower proof: the caller must
    name a canonical parent project, its state must describe an externally owned
    loopback daemon at the *exact* requested endpoint, and an authenticated health
    response must echo every durable daemon identity field.  This grants command
    access only; lifecycle operations retain the stronger OS identity checks used
    by :func:`discover` and the stop/restart paths.
    """

    st = _read_state(config)
    if not st:
        return None
    try:
        base = _base(st)
    except RuntimeError:
        return None

    # Keep endpoint comparison deliberately literal.  Different loopback aliases
    # can be served by different processes, so a token recorded for 127.0.0.1 must
    # never be sent to localhost or ::1 merely because all are loopback names.
    from urllib.parse import urlsplit

    def endpoint(url: str) -> tuple[str, str, int | None] | None:
        try:
            parsed = urlsplit(url if "://" in url else f"http://{url}")
            return (parsed.scheme or "http", parsed.hostname or "", parsed.port)
        except ValueError:
            return None

    if endpoint(base) != endpoint(expected_base):
        return None

    pid = st.get("pid")
    process_start_token = st.get("process_start_token")
    instance_id = st.get("instance_id")
    token = st.get("token")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or not 1 < pid <= MAX_SAFE_PID
        or not isinstance(process_start_token, str)
        or not _valid_spawn_token(process_start_token)
        or st.get("project_id") != _project_id(config)
        or not isinstance(instance_id, str)
        or not instance_id
        or not isinstance(token, str)
        or not token
        or st.get("lifecycle_owner") != "external"
    ):
        return None

    health = _health(
        base,
        token,
        timeout=health_timeout,
        proxy=attached_parent_http_proxy(),
    )
    if not isinstance(health, dict) or health.get("ok") is not True:
        return None
    if any(
        health.get(field) != st.get(field)
        for field in (
            "pid",
            "process_start_token",
            "project_id",
            "instance_id",
            "lifecycle_owner",
        )
    ):
        return None
    return base, token


def _managed_bind_host(config: Config) -> str:
    """Return the numeric loopback host supported by managed discovery."""

    configured = config.server.get("host", "127.0.0.1")
    if not isinstance(configured, str):
        raise RuntimeError("managed daemon host must be a numeric IP address")
    normalized = "127.0.0.1" if configured == "localhost" else configured
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError as exc:
        raise RuntimeError("managed daemon host must be a numeric IP address") from exc
    if not address.is_loopback:
        raise RuntimeError(
            "usd-cli daemon binding is loopback-only; configure remote rendering "
            "through the authenticated OVRTX service"
        )
    return normalized


def _free_port(host: str = "127.0.0.1") -> int:
    family = (
        socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET
    )
    s = socket.socket(family)
    s.bind((host, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _open_daemon_log(state_dir: Path):  # noqa: ANN202
    if os.name == "nt":
        from usd_core.windows_files import (
            open_confined_directory,
            open_confined_regular_file_at,
        )

        anchor = getattr(_LIFECYCLE_DIRECTORY_LOCKS, "anchor", None)
        requested_state_path = Path(os.path.abspath(state_dir))
        if anchor is not None and requested_state_path != anchor.state_path:
            raise RuntimeError("daemon log state-directory identity changed")
        descriptor = -1
        if anchor is not None:
            _require_directory_path_identity(
                anchor.state_descriptor,
                anchor.state_path,
                label="state-directory",
            )
            descriptor = open_confined_regular_file_at(
                anchor.state_descriptor,
                "daemon.log",
                writable=True,
                append=True,
                create=True,
            )
        else:
            with open_confined_directory(state_dir) as directory_descriptor:
                descriptor = open_confined_regular_file_at(
                    directory_descriptor,
                    "daemon.log",
                    writable=True,
                    append=True,
                    create=True,
                )
        try:
            stream = os.fdopen(descriptor, "ab")
        except BaseException:
            os.close(descriptor)
            raise
        return stream
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe daemon logging requires POSIX no-follow flags")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    anchor = getattr(_LIFECYCLE_DIRECTORY_LOCKS, "anchor", None)
    requested_state_path = Path(os.path.abspath(state_dir))
    if anchor is not None and requested_state_path != anchor.state_path:
        raise RuntimeError("daemon log state-directory identity changed")
    if anchor is not None:
        _require_directory_path_identity(
            anchor.state_descriptor,
            anchor.state_path,
            label="state-directory",
        )
    directory_descriptor = (
        os.dup(anchor.state_descriptor)
        if anchor is not None
        else os.open(state_dir, directory_flags)
    )
    log_descriptor = -1
    try:
        log_flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
        log_flags |= getattr(os, "O_CLOEXEC", 0)
        log_descriptor = os.open(
            "daemon.log",
            log_flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        metadata = os.fstat(log_descriptor)
        path_metadata = os.stat(
            "daemon.log",
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise RuntimeError("daemon log is not a single-link regular file")
        os.fchmod(log_descriptor, 0o600)
        stream = os.fdopen(log_descriptor, "ab")
        log_descriptor = -1
        return stream
    finally:
        if log_descriptor >= 0:
            os.close(log_descriptor)
        os.close(directory_descriptor)


def _write_all(file_descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(file_descriptor, view)
        if written <= 0:
            raise OSError("daemon spawn-ledger write made no forward progress")
        view = view[written:]


def _startup_descriptor_environment_value(descriptor: int) -> str:
    """Encode an inherited startup capability for the daemon child.

    POSIX inherits CRT descriptors directly.  Windows ``Popen`` can inherit an
    explicit kernel-handle allowlist, but not ``pass_fds``; prefixing the handle
    also prevents a child from accidentally treating an unrelated CRT descriptor
    number as the capability.
    """

    if os.name != "nt":
        return str(descriptor)
    import msvcrt

    return f"handle:{msvcrt.get_osfhandle(descriptor)}"


def _startup_descriptor_from_value(raw_value: str, *, writable: bool) -> int:
    if os.name != "nt":
        return int(raw_value)
    prefix, separator, handle_text = raw_value.partition(":")
    if prefix != "handle" or not separator:
        raise ValueError("Windows startup capability is not an inherited handle")
    handle = int(handle_text)
    if handle <= 0:
        raise ValueError("Windows startup capability handle is invalid")
    import msvcrt

    flags = os.O_WRONLY if writable else os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    return msvcrt.open_osfhandle(handle, flags)


def _startup_descriptor_from_environment(name: str, *, writable: bool) -> int:
    return _startup_descriptor_from_value(os.environ[name], writable=writable)


def _startup_pipe_readable(descriptor: int, *, timeout: float) -> bool:
    """Wait for registration bytes or EOF on one inherited anonymous pipe."""

    if os.name != "nt":
        ready, _, _ = select.select([descriptor], [], [], timeout)
        return bool(ready)
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    peek_named_pipe = kernel32.PeekNamedPipe
    peek_named_pipe.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    peek_named_pipe.restype = wintypes.BOOL
    handle = msvcrt.get_osfhandle(descriptor)
    deadline = time.monotonic() + max(0.0, timeout)
    eof_errors = {109, 232, 233}  # broken, no data, or disconnected pipe
    while True:
        available = wintypes.DWORD()
        if peek_named_pipe(handle, None, 0, None, ctypes.byref(available), None):
            if available.value > 0:
                return True
        else:
            error = ctypes.get_last_error()
            if error in eof_errors:
                return True
            raise ctypes.WinError(error)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.01, remaining))


def _read_regular_file_at(
    parent_descriptor: int,
    name: str,
    *,
    max_bytes: int,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > max_bytes
        ):
            raise RuntimeError(
                f"daemon spawn ledger must be a bounded single-link regular file: {name}"
            )
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes - size + 1))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise RuntimeError(
                    f"daemon spawn ledger exceeds {max_bytes} bytes: {name}"
                )
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        path_after = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            identity_before != identity_after
            or size != after.st_size
            or not stat.S_ISREG(path_after.st_mode)
            or path_after.st_nlink != 1
            or (path_after.st_dev, path_after.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise RuntimeError(f"daemon spawn ledger changed while read: {name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _valid_spawn_token(token: str) -> bool:
    return bool(
        token
        and len(token) <= 256
        and ":" not in token
        and not any(character.isspace() for character in token)
    )


def _parse_ledger(payload: bytes) -> list[tuple[int, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("daemon spawn ledger is not valid UTF-8") from exc
    entries: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        pid_text, separator, token = line.partition(":")
        if not separator or not pid_text.isascii() or not pid_text.isdecimal():
            raise RuntimeError(
                f"daemon spawn ledger has an invalid entry on line {line_number}"
            )
        pid = int(pid_text)
        if not 1 < pid <= MAX_SAFE_PID or not _valid_spawn_token(token):
            raise RuntimeError(
                f"daemon spawn ledger has an unsafe entry on line {line_number}"
            )
        entries.append((pid, token))
    return entries


def _group_ledger_entries(
    entries: list[tuple[int, str | None]],
) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}
    for pid, token in entries:
        if token is None:
            continue
        tokens = grouped.setdefault(pid, [])
        if token not in tokens:
            tokens.append(token)
    return grouped


def _directory_identity(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("daemon lifecycle anchor is not a directory")
    return metadata.st_dev, metadata.st_ino


def _require_directory_path_identity(
    descriptor: int,
    path: Path,
    *,
    label: str,
) -> None:
    try:
        held_identity = _directory_identity(descriptor)
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"daemon {label} identity changed") from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or held_identity != (current.st_dev, current.st_ino)
    ):
        raise RuntimeError(f"daemon {label} identity changed")


def _held_lifecycle_anchor(config: Config) -> _LifecycleAnchor | None:
    anchor = getattr(_LIFECYCLE_DIRECTORY_LOCKS, "anchor", None)
    if anchor is None:
        return None
    project_path = (config.project_dir or Path.cwd()).resolve()
    if project_path != anchor.project_path:
        raise RuntimeError("daemon project-directory identity changed")
    return anchor


def _held_state_directory_descriptor(config: Config) -> int | None:
    anchor = _held_lifecycle_anchor(config)
    return anchor.state_descriptor if anchor is not None else None


def _require_current_lifecycle_state_directory(config: Config) -> None:
    """Fail if either frozen lifecycle inode moved from its original path."""

    from usd_core.config import state_dir_for

    anchor = _held_lifecycle_anchor(config)
    if anchor is None:
        return
    _require_directory_path_identity(
        anchor.project_descriptor,
        anchor.project_path,
        label="project-directory",
    )
    if Path(os.path.abspath(state_dir_for(anchor.project_path))) != anchor.state_path:
        raise RuntimeError("daemon state-directory identity changed")
    _require_directory_path_identity(
        anchor.state_descriptor,
        anchor.state_path,
        label="state-directory",
    )


@contextmanager
def _locked_ledger(
    config: Config,
    *,
    create_state: bool,
) -> Iterator[int]:
    """Yield the no-follow state-dir descriptor under the ledger lifecycle lock."""

    from usd_core.config import state_dir_for

    held_anchor = _held_lifecycle_anchor(config)
    if held_anchor is not None:
        # The start lifecycle already holds the stable directory inode exclusively
        # in this thread. Re-entering flock through a separately opened fd can
        # self-deadlock, and the outer directory lock already excludes every
        # cooperating ledger writer.
        _require_current_lifecycle_state_directory(config)
        yield held_anchor.state_descriptor
        _require_current_lifecycle_state_directory(config)
        return
    if os.name == "nt":
        from usd_core.windows_files import (
            advisory_file_lock,
            open_confined_directory,
            open_confined_regular_file_at,
            require_confined_regular_file,
        )

        project_dir = (config.project_dir or Path.cwd()).resolve()
        with open_confined_directory(project_dir) as project_descriptor:
            _require_directory_path_identity(
                project_descriptor,
                project_dir,
                label="project-directory",
            )
            state_dir = Path(os.path.abspath(state_dir_for(project_dir)))
            if create_state:
                state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            with open_confined_directory(state_dir) as directory_descriptor:
                _require_directory_path_identity(
                    project_descriptor,
                    project_dir,
                    label="project-directory",
                )
                _require_directory_path_identity(
                    directory_descriptor,
                    state_dir,
                    label="state-directory",
                )
                lock_descriptor = open_confined_regular_file_at(
                    directory_descriptor,
                    DAEMON_LEDGER_LOCK_NAME,
                    readable=True,
                    writable=True,
                    create=True,
                )
                try:
                    with advisory_file_lock(lock_descriptor):
                        require_confined_regular_file(lock_descriptor)
                        yield directory_descriptor
                        _require_directory_path_identity(
                            project_descriptor,
                            project_dir,
                            label="project-directory",
                        )
                        if Path(os.path.abspath(state_dir_for(project_dir))) != state_dir:
                            raise RuntimeError("daemon state-directory identity changed")
                        _require_directory_path_identity(
                            directory_descriptor,
                            state_dir,
                            label="state-directory",
                        )
                        require_confined_regular_file(lock_descriptor)
                finally:
                    os.close(lock_descriptor)
        return
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe daemon ledger access requires POSIX no-follow flags")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    project_dir = (config.project_dir or Path.cwd()).resolve()
    project_descriptor = os.open(project_dir, directory_flags)
    directory_descriptor = -1
    lock_descriptor = -1
    try:
        import fcntl

        # The resolved project-directory inode survives replacement of the state
        # directory name, so all cooperating lifecycle operations serialize here
        # before selecting/opening a state inode.
        fcntl.flock(project_descriptor, fcntl.LOCK_EX)
        _require_directory_path_identity(
            project_descriptor,
            project_dir,
            label="project-directory",
        )
        state_dir = Path(os.path.abspath(state_dir_for(project_dir)))
        if create_state:
            state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory_descriptor = os.open(state_dir, directory_flags)
        _require_directory_path_identity(
            project_descriptor,
            project_dir,
            label="project-directory",
        )
        _require_directory_path_identity(
            directory_descriptor,
            state_dir,
            label="state-directory",
        )
        os.fchmod(directory_descriptor, 0o700)
        lock_flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
        lock_flags |= getattr(os, "O_CLOEXEC", 0)
        lock_descriptor = os.open(
            DAEMON_LEDGER_LOCK_NAME,
            lock_flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        lock_metadata = os.fstat(lock_descriptor)
        lock_path_metadata = os.stat(
            DAEMON_LEDGER_LOCK_NAME,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(lock_metadata.st_mode)
            or lock_metadata.st_nlink != 1
            or not stat.S_ISREG(lock_path_metadata.st_mode)
            or lock_path_metadata.st_nlink != 1
            or (lock_metadata.st_dev, lock_metadata.st_ino)
            != (lock_path_metadata.st_dev, lock_path_metadata.st_ino)
        ):
            raise RuntimeError("daemon ledger lock is not a single-link regular file")
        os.fchmod(lock_descriptor, 0o600)
        # Lock the stable state-directory inode as well as the conventional lock
        # file. A same-user process can unlink and recreate daemon.pids.lock while
        # another writer holds it; every cooperating writer still opens and locks
        # this directory inode, so replacing the named file cannot split the lock.
        fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        locked_path_metadata = os.stat(
            DAEMON_LEDGER_LOCK_NAME,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(locked_path_metadata.st_mode)
            or locked_path_metadata.st_nlink != 1
            or (lock_metadata.st_dev, lock_metadata.st_ino)
            != (locked_path_metadata.st_dev, locked_path_metadata.st_ino)
        ):
            raise RuntimeError("daemon ledger lock identity changed")
        yield directory_descriptor
        _require_directory_path_identity(
            project_descriptor,
            project_dir,
            label="project-directory",
        )
        if Path(os.path.abspath(state_dir_for(project_dir))) != state_dir:
            raise RuntimeError("daemon state-directory identity changed")
        _require_directory_path_identity(
            directory_descriptor,
            state_dir,
            label="state-directory",
        )
    finally:
        if lock_descriptor >= 0:
            try:
                import fcntl

                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                fcntl.flock(directory_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(lock_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        try:
            import fcntl

            fcntl.flock(project_descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(project_descriptor)


@contextmanager
def _locked_start_lifecycle(config: Config) -> Iterator[None]:
    """Serialize discovery/spawn/readiness through a no-follow advisory lock."""

    from usd_core.config import state_dir_for

    if os.name == "nt":
        from usd_core.windows_files import (
            advisory_file_lock,
            open_confined_directory,
            open_confined_regular_file_at,
            require_confined_regular_file,
        )

        if getattr(_LIFECYCLE_DIRECTORY_LOCKS, "anchor", None) is not None:
            raise RuntimeError("daemon lifecycle lock is not reentrant")
        project_dir = (config.project_dir or Path.cwd()).resolve()
        with open_confined_directory(project_dir) as project_descriptor:
            _require_directory_path_identity(
                project_descriptor,
                project_dir,
                label="project-directory",
            )
            state_dir = Path(os.path.abspath(state_dir_for(project_dir)))
            state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            with open_confined_directory(state_dir) as directory_descriptor:
                _require_directory_path_identity(
                    project_descriptor,
                    project_dir,
                    label="project-directory",
                )
                _require_directory_path_identity(
                    directory_descriptor,
                    state_dir,
                    label="state-directory",
                )
                lock_descriptor = open_confined_regular_file_at(
                    directory_descriptor,
                    "server.json.lock",
                    readable=True,
                    writable=True,
                    create=True,
                )
                ledger_lock_descriptor = -1
                try:
                    # POSIX locks the stable state-directory inode, which also
                    # excludes standalone ledger writers for the entire start.
                    # Windows denies directory replacement through the held
                    # handle and takes the ledger's portable advisory lock to
                    # preserve that cross-operation serialization.
                    ledger_lock_descriptor = open_confined_regular_file_at(
                        directory_descriptor,
                        DAEMON_LEDGER_LOCK_NAME,
                        readable=True,
                        writable=True,
                        create=True,
                    )
                    with advisory_file_lock(lock_descriptor):
                        with advisory_file_lock(ledger_lock_descriptor):
                            require_confined_regular_file(lock_descriptor)
                            require_confined_regular_file(ledger_lock_descriptor)
                            _LIFECYCLE_DIRECTORY_LOCKS.anchor = _LifecycleAnchor(
                                project_descriptor=project_descriptor,
                                state_descriptor=directory_descriptor,
                                project_path=project_dir,
                                state_path=state_dir,
                            )
                            try:
                                _require_current_lifecycle_state_directory(config)
                                yield
                                _require_current_lifecycle_state_directory(config)
                                require_confined_regular_file(lock_descriptor)
                                require_confined_regular_file(ledger_lock_descriptor)
                            finally:
                                _LIFECYCLE_DIRECTORY_LOCKS.anchor = None
                finally:
                    if ledger_lock_descriptor >= 0:
                        os.close(ledger_lock_descriptor)
                    os.close(lock_descriptor)
        return
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe daemon lifecycle locking requires POSIX flags")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    project_dir = (config.project_dir or Path.cwd()).resolve()
    project_descriptor = os.open(project_dir, directory_flags)
    directory_descriptor = -1
    lock_descriptor = -1
    try:
        import fcntl

        fcntl.flock(project_descriptor, fcntl.LOCK_EX)
        _require_directory_path_identity(
            project_descriptor,
            project_dir,
            label="project-directory",
        )
        state_dir = Path(os.path.abspath(state_dir_for(project_dir)))
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory_descriptor = os.open(state_dir, directory_flags)
        _require_directory_path_identity(
            project_descriptor,
            project_dir,
            label="project-directory",
        )
        _require_directory_path_identity(
            directory_descriptor,
            state_dir,
            label="state-directory",
        )
        os.fchmod(directory_descriptor, 0o700)
        lock_flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
        lock_flags |= getattr(os, "O_CLOEXEC", 0)
        lock_descriptor = os.open(
            "server.json.lock",
            lock_flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        metadata = os.fstat(lock_descriptor)
        path_metadata = os.stat(
            "server.json.lock",
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise RuntimeError(
                "daemon lifecycle lock is not a single-link regular file"
            )
        os.fchmod(lock_descriptor, 0o600)
        # The directory inode is the non-replaceable cooperating lock.  The named
        # file remains for compatibility, but unlinking/recreating it cannot split
        # two starters because both must first serialize on this directory.
        fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        locked_metadata = os.stat(
            "server.json.lock",
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(locked_metadata.st_mode)
            or locked_metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino)
            != (locked_metadata.st_dev, locked_metadata.st_ino)
        ):
            raise RuntimeError("daemon lifecycle lock identity changed")
        if getattr(_LIFECYCLE_DIRECTORY_LOCKS, "anchor", None) is not None:
            raise RuntimeError("daemon lifecycle lock is not reentrant")
        _LIFECYCLE_DIRECTORY_LOCKS.anchor = _LifecycleAnchor(
            project_descriptor=project_descriptor,
            state_descriptor=directory_descriptor,
            project_path=project_dir,
            state_path=state_dir,
        )
        _require_current_lifecycle_state_directory(config)
        yield
        _require_current_lifecycle_state_directory(config)
    finally:
        if lock_descriptor >= 0:
            try:
                import fcntl

                _LIFECYCLE_DIRECTORY_LOCKS.anchor = None
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                fcntl.flock(directory_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(lock_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        try:
            import fcntl

            fcntl.flock(project_descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(project_descriptor)


def _read_ledger_locked(directory_descriptor: int) -> list[tuple[int, str]]:
    if os.name == "nt":
        from usd_core.windows_files import read_confined_regular_file_at

        try:
            payload = read_confined_regular_file_at(
                directory_descriptor,
                DAEMON_LEDGER_NAME,
                max_bytes=MAX_DAEMON_LEDGER_BYTES,
            )
        except FileNotFoundError:
            return []
        return _parse_ledger(payload)
    try:
        payload = _read_regular_file_at(
            directory_descriptor,
            DAEMON_LEDGER_NAME,
            max_bytes=MAX_DAEMON_LEDGER_BYTES,
        )
    except FileNotFoundError:
        return []
    return _parse_ledger(payload)


def _write_ledger_locked(
    directory_descriptor: int,
    entries: list[tuple[int, str]],
) -> None:
    """Atomically publish normalized ledger entries and fsync the rename."""

    payload = "".join(f"{pid}:{token}\n" for pid, token in entries).encode("utf-8")
    if len(payload) > MAX_DAEMON_LEDGER_BYTES:
        raise RuntimeError(
            f"daemon spawn ledger would exceed {MAX_DAEMON_LEDGER_BYTES} bytes"
        )
    if os.name == "nt":
        from usd_core.windows_files import replace_confined_regular_file_at

        replace_confined_regular_file_at(
            directory_descriptor,
            DAEMON_LEDGER_NAME,
            payload,
            max_bytes=MAX_DAEMON_LEDGER_BYTES,
            remove_if_empty=True,
        )
        return
    temporary_name = f".{DAEMON_LEDGER_NAME}.{secrets.token_hex(8)}.tmp"
    temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    temporary_flags |= getattr(os, "O_CLOEXEC", 0)
    temporary_descriptor = os.open(
        temporary_name,
        temporary_flags,
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        _write_all(temporary_descriptor, payload)
        os.fsync(temporary_descriptor)
        temporary_metadata = os.fstat(temporary_descriptor)
        staged_metadata = os.stat(
            temporary_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        temporary_identity = (temporary_metadata.st_dev, temporary_metadata.st_ino)
        if (
            not stat.S_ISREG(temporary_metadata.st_mode)
            or temporary_metadata.st_nlink != 1
            or not stat.S_ISREG(staged_metadata.st_mode)
            or staged_metadata.st_nlink != 1
            or (staged_metadata.st_dev, staged_metadata.st_ino) != temporary_identity
        ):
            raise RuntimeError("daemon spawn-ledger temporary identity changed")
        os.replace(
            temporary_name,
            DAEMON_LEDGER_NAME,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        temporary_name = ""
        installed_metadata = os.stat(
            DAEMON_LEDGER_NAME,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(installed_metadata.st_mode)
            or installed_metadata.st_nlink != 1
            or (installed_metadata.st_dev, installed_metadata.st_ino)
            != temporary_identity
        ):
            raise RuntimeError("daemon spawn-ledger installed identity changed")
        os.fsync(directory_descriptor)
        if not payload:
            # Entries are removed only after publishing and verifying our own empty
            # regular file under the lifecycle lock.  This preserves the historical
            # absent-file representation without unlinking an untrusted object.
            os.unlink(DAEMON_LEDGER_NAME, dir_fd=directory_descriptor)
            os.fsync(directory_descriptor)
    finally:
        os.close(temporary_descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass


def _normalized_entries_for_record(
    entries: list[tuple[int, str]],
    *,
    new_pid: int,
    new_token: str,
) -> list[tuple[int, str]]:
    """Prune stale/duplicate identities before recording the new child."""

    normalized: list[tuple[int, str]] = []
    for pid, tokens in _group_ledger_entries(entries).items():
        if len(tokens) != 1:
            normalized.extend((pid, token) for token in tokens)
            continue
        if pid == new_pid:
            continue
        current_token = _proc_start_token(pid)
        if current_token in tokens:
            normalized.append((pid, current_token))
        elif current_token is None and _proc_state(pid) == "alive":
            normalized.extend((pid, token) for token in tokens)
    if (new_pid, new_token) not in normalized:
        normalized.append((new_pid, new_token))
    return normalized


def _record_spawn_identity(config: Config, pid: int, token: str) -> None:
    """Atomically and durably record one daemon before it can serve."""

    if not 1 < pid <= MAX_SAFE_PID or not _valid_spawn_token(token):
        raise RuntimeError("daemon spawn identity is not safely recordable")
    with _locked_ledger(config, create_state=True) as directory_descriptor:
        entries = _read_ledger_locked(directory_descriptor)
        normalized = _normalized_entries_for_record(
            entries,
            new_pid=pid,
            new_token=token,
        )
        _write_ledger_locked(directory_descriptor, normalized)


def _terminate_unrecorded_spawn(
    proc: subprocess.Popen,
    identity: tuple[int, str] | None,
) -> bool:
    """Terminate the exact Popen when durable ownership recording failed."""

    if identity is not None:
        pid, token = identity
        identity_gone = _terminate(pid, expected_token=token)
        # ``_terminate`` deliberately treats an unverifiable/recycled pid as the
        # owned identity already gone.  On Windows a venv launcher is the direct
        # Popen child and the authenticated interpreter is its child; wait for the
        # launcher to observe that interpreter's exit before falling back to killing
        # the exact launcher process.
        if identity_gone:
            try:
                proc.wait(timeout=2)
                return True
            except subprocess.TimeoutExpired:
                pass
    elif proc.poll() is not None:
        return True
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    except OSError:
        pass
    if identity is None:
        return proc.poll() is not None
    pid, token = identity
    return proc.poll() is not None and (
        _proc_start_token(pid) != token or _proc_state(pid) != "alive"
    )


def _cleanup_failed_spawn(config: Config, identity: tuple[int, str]) -> bool:
    """Reap only the daemon identity created by the failed start attempt."""

    pid, token = identity
    if not _terminate(pid, expected_token=token):
        # Keep durable ownership evidence for an unverifiable/still-live child.
        return False
    state = _read_state(config)
    if (
        isinstance(state, dict)
        and state.get("pid") == pid
        and state.get("process_start_token") == token
        and state.get("project_id") == _project_id(config)
    ):
        try:
            config.server_state_path.unlink(missing_ok=True)
        except OSError:
            pass
    _discard_spawn_identity(config, identity)
    return True


def _discard_spawn_identity(config: Config, identity: tuple[int, str]) -> None:
    """Atomically forget only one exact in-memory spawn identity."""

    try:
        with _locked_ledger(config, create_state=False) as directory_descriptor:
            entries = _read_ledger_locked(directory_descriptor)
            survivors = [entry for entry in entries if entry != identity]
            if survivors != entries:
                _write_ledger_locked(directory_descriptor, survivors)
    except (FileNotFoundError, OSError, RuntimeError):
        # Unsafe evidence is preserved. Failure to rewrite must never broaden the
        # set of processes a later command is allowed to signal.
        pass


def validate_child_startup_identity(
    config: Config,
    identity: DaemonStartupIdentity,
) -> None:
    """Revalidate inherited project/state inodes against frozen parent paths."""

    from usd_core.config import state_dir_for

    if (config.project_dir or Path.cwd()).resolve() != identity.project_path:
        raise RuntimeError("daemon child project-directory identity changed")
    if Path(os.path.abspath(state_dir_for(identity.project_path))) != identity.state_path:
        raise RuntimeError("daemon child state-directory identity changed")
    _require_directory_path_identity(
        identity.project_descriptor,
        identity.project_path,
        label="child project-directory",
    )
    _require_directory_path_identity(
        identity.state_descriptor,
        identity.state_path,
        label="child state-directory",
    )


def child_startup_handshake(config: Config) -> DaemonStartupIdentity:
    """Report this child's birth identity, then await the parent's serve release.

    The parent durably records the acknowledged identity while holding the stable
    project lifecycle lock. The child cannot import the server, create state, or bind
    until the parent sends one release byte; parent death closes the pipe and makes
    the child exit instead, including before durable registration completes.
    """

    try:
        release_descriptor = _startup_descriptor_from_environment(
            STARTUP_RELEASE_FD_ENV,
            writable=False,
        )
        acknowledgement_descriptor = _startup_descriptor_from_environment(
            STARTUP_ACK_FD_ENV,
            writable=True,
        )
        project_descriptor = _startup_descriptor_from_environment(
            STARTUP_PROJECT_FD_ENV,
            writable=False,
        )
        state_descriptor = _startup_descriptor_from_environment(
            STARTUP_STATE_FD_ENV,
            writable=False,
        )
        project_path = Path(os.environ[STARTUP_PROJECT_PATH_ENV])
        state_path = Path(os.environ[STARTUP_STATE_PATH_ENV])
        nonce = os.environ[STARTUP_NONCE_ENV]
    except (KeyError, OSError, ValueError) as exc:
        raise RuntimeError(
            "daemon child is missing its startup registration gate"
        ) from exc
    descriptors = {
        release_descriptor,
        acknowledgement_descriptor,
        project_descriptor,
        state_descriptor,
    }
    if (
        any(descriptor <= 2 for descriptor in descriptors)
        or len(descriptors) != 4
        or not project_path.is_absolute()
        or not state_path.is_absolute()
        or not nonce
    ):
        raise RuntimeError("daemon child received an unsafe startup registration gate")

    pid = os.getpid()
    identity: DaemonStartupIdentity | None = None
    try:
        identity = DaemonStartupIdentity(
            pid=pid,
            process_start_token="",
            project_descriptor=project_descriptor,
            state_descriptor=state_descriptor,
            project_path=project_path,
            state_path=state_path,
        )
        validate_child_startup_identity(config, identity)
        token = _proc_start_token(pid)
        if token is None:
            raise RuntimeError("daemon child cannot capture its process birth token")
        identity = DaemonStartupIdentity(
            pid=pid,
            process_start_token=token,
            project_descriptor=project_descriptor,
            state_descriptor=state_descriptor,
            project_path=project_path,
            state_path=state_path,
        )
        acknowledgement = (
            json.dumps(
                {"pid": pid, "process_start_token": token, "nonce": nonce},
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        if len(acknowledgement) > MAX_DAEMON_REGISTRATION_BYTES:
            raise RuntimeError("daemon child registration acknowledgement is oversized")
        _write_all(acknowledgement_descriptor, acknowledgement)
        try:
            os.close(acknowledgement_descriptor)
        except OSError:
            pass
        acknowledgement_descriptor = -1
        release = os.read(release_descriptor, 1)
        if release != b"1":
            raise RuntimeError(
                "daemon starter exited before releasing the registered child"
            )
        validate_child_startup_identity(config, identity)
        return identity
    except BaseException:
        for descriptor in (project_descriptor, state_descriptor):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    finally:
        for descriptor in (acknowledgement_descriptor, release_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _await_child_registration(
    config: Config,
    proc: subprocess.Popen,
    acknowledgement_descriptor: int,
    nonce: str,
) -> tuple[int, str]:
    """Wait for and authenticate the child's OS birth-identity acknowledgement."""

    del config

    deadline = time.monotonic() + DAEMON_REGISTRATION_TIMEOUT_S
    payload = bytearray()
    while b"\n" not in payload:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("daemon child registration timed out")
        if not _startup_pipe_readable(
            acknowledgement_descriptor,
            timeout=min(remaining, 0.25),
        ):
            if proc.poll() is not None:
                raise RuntimeError("daemon child exited before durable registration")
            continue
        chunk = os.read(
            acknowledgement_descriptor,
            MAX_DAEMON_REGISTRATION_BYTES - len(payload) + 1,
        )
        if not chunk:
            raise RuntimeError("daemon child closed its registration acknowledgement")
        payload.extend(chunk)
        if len(payload) > MAX_DAEMON_REGISTRATION_BYTES:
            raise RuntimeError("daemon child registration acknowledgement is oversized")
    line, separator, remainder = bytes(payload).partition(b"\n")
    if not separator or remainder:
        raise RuntimeError("daemon child registration acknowledgement is malformed")
    try:
        registration = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "daemon child registration acknowledgement is malformed"
        ) from exc
    if not isinstance(registration, dict):
        raise RuntimeError("daemon child registration acknowledgement is malformed")
    pid = registration.get("pid")
    token = registration.get("process_start_token")
    invalid_fields: list[str] = []
    if not isinstance(pid, int) or isinstance(pid, bool):
        invalid_fields.append("pid")
    elif not 1 < pid <= MAX_SAFE_PID:
        invalid_fields.append("pid-range")
    elif pid != proc.pid:
        if os.name != "nt":
            invalid_fields.append("pid")
        else:
            from usd_core.windows_files import windows_process_parent_pid

            if windows_process_parent_pid(pid) != proc.pid:
                invalid_fields.append("pid-parent")
    if not isinstance(token, str) or not _valid_spawn_token(token):
        invalid_fields.append("process-token")
    if registration.get("nonce") != nonce:
        invalid_fields.append("nonce")
    if invalid_fields:
        raise RuntimeError(
            "daemon child registration identity does not match its spawn: "
            + ", ".join(invalid_fields)
        )
    assert isinstance(pid, int) and not isinstance(pid, bool)
    assert isinstance(token, str)
    if _proc_start_token(pid) != token:
        raise RuntimeError("daemon child registration process identity is not live")
    return pid, token


def _startup_render_credentials_payload(config: Config) -> bytes:
    remote_api_key = str(config.render.get("remote_api_key") or "")
    backend_api_keys: dict[str, str] = {}
    backends = config.render.get("backends")
    if isinstance(backends, list):
        for entry in backends:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url") or "").rstrip("/")
            api_key = str(entry.get("api_key") or "")
            if url and api_key and url not in backend_api_keys:
                backend_api_keys[url] = api_key
    payload = json.dumps(
        {
            "remote_api_key": remote_api_key,
            "backend_api_keys": backend_api_keys,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(payload) > MAX_STARTUP_RENDER_CREDENTIALS_BYTES:
        raise RuntimeError("daemon render credential payload is oversized")
    return payload


def apply_startup_render_credentials(config: Config) -> None:
    """Read parent-only render credentials from the inherited startup pipe."""

    raw_descriptor = os.environ.pop(STARTUP_RENDER_CREDENTIALS_FD_ENV, None)
    if raw_descriptor is None:
        return
    try:
        descriptor = _startup_descriptor_from_value(
            raw_descriptor,
            writable=False,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError("daemon render credential descriptor is invalid") from exc
    payload = bytearray()
    try:
        while chunk := os.read(
            descriptor,
            MAX_STARTUP_RENDER_CREDENTIALS_BYTES - len(payload) + 1,
        ):
            payload.extend(chunk)
            if len(payload) > MAX_STARTUP_RENDER_CREDENTIALS_BYTES:
                raise RuntimeError("daemon render credential payload is oversized")
    finally:
        os.close(descriptor)
    try:
        credentials = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("daemon render credential payload is malformed") from exc
    if not isinstance(credentials, dict):
        raise RuntimeError("daemon render credential payload is malformed")
    remote_api_key = credentials.get("remote_api_key")
    backend_api_keys = credentials.get("backend_api_keys")
    if not isinstance(remote_api_key, str) or not isinstance(backend_api_keys, dict):
        raise RuntimeError("daemon render credential payload is malformed")
    if any(
        not isinstance(url, str) or not isinstance(api_key, str)
        for url, api_key in backend_api_keys.items()
    ):
        raise RuntimeError("daemon render credential payload is malformed")
    config.render["remote_api_key"] = remote_api_key
    backends = config.render.get("backends")
    if isinstance(backends, list):
        config.render["backends"] = [
            (
                {
                    **entry,
                    "api_key": backend_api_keys[str(entry.get("url", "")).rstrip("/")],
                }
                if isinstance(entry, dict)
                and str(entry.get("url", "")).rstrip("/") in backend_api_keys
                else entry
            )
            for entry in backends
        ]


def _spawn(config: Config) -> tuple[int, str]:
    anchor = _held_lifecycle_anchor(config)
    if anchor is None:
        raise RuntimeError("daemon spawn requires the project lifecycle lock")
    base_dir = anchor.project_path
    state_dir = anchor.state_path
    _require_current_lifecycle_state_directory(config)
    env = dict(os.environ)
    for key in tuple(env):
        if (
            key.startswith("PYTHON")
            or key.startswith("DYLD_")
            or key
            in {
                "LD_AUDIT",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "PXR_PLUGINPATH_NAME",
                "USD_PLUGIN_PATH",
            }
        ):
            env.pop(key, None)
    for credential_env in (
        "OVRTX_API_KEY",
        "3DSC_RENDER_REMOTE_API_KEY",
        "OV_RENDER_REMOTE_API_KEY",
        "USD_CLI_RENDER_REMOTE_API_KEY",
        "3DSC_RENDER_BACKEND_API_KEYS_JSON",
        "OV_RENDER_BACKEND_API_KEYS_JSON",
        "USD_CLI_RENDER_BACKEND_API_KEYS_JSON",
    ):
        env.pop(credential_env, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONSAFEPATH"] = "1"
    managed_host = _managed_bind_host(config)
    env["USD_CLI_SERVER_HOST"] = managed_host
    env["USD_CLI_SERVER_PORT"] = str(_free_port(managed_host))
    credentials_payload = _startup_render_credentials_payload(config)
    release_read, release_write = os.pipe()
    acknowledgement_read, acknowledgement_write = os.pipe()
    credentials_read, credentials_write = os.pipe()
    nonce = secrets.token_hex(16)
    env[STARTUP_RELEASE_FD_ENV] = _startup_descriptor_environment_value(release_read)
    env[STARTUP_ACK_FD_ENV] = _startup_descriptor_environment_value(
        acknowledgement_write
    )
    env[STARTUP_NONCE_ENV] = nonce
    env[STARTUP_PROJECT_FD_ENV] = _startup_descriptor_environment_value(
        anchor.project_descriptor
    )
    env[STARTUP_STATE_FD_ENV] = _startup_descriptor_environment_value(
        anchor.state_descriptor
    )
    env[STARTUP_PROJECT_PATH_ENV] = str(anchor.project_path)
    env[STARTUP_STATE_PATH_ENV] = str(anchor.state_path)
    env[STARTUP_RENDER_CREDENTIALS_FD_ENV] = _startup_descriptor_environment_value(
        credentials_read
    )
    try:
        log = _open_daemon_log(state_dir)
    except BaseException:
        for descriptor in (
            release_read,
            release_write,
            acknowledgement_read,
            acknowledgement_write,
            credentials_read,
            credentials_write,
        ):
            os.close(descriptor)
        raise
    try:
        try:
            bootstrap = (
                "from usd_core.config import load_config;"
                "from usd_cli.daemon import apply_startup_render_credentials,child_startup_handshake;"
                "cfg=load_config();apply_startup_render_credentials(cfg);"
                "identity=child_startup_handshake(cfg);"
                "from usd_server.app import serve;"
                "serve(cfg,startup_identity=identity)"
            )
            inherited_descriptors = (
                release_read,
                acknowledgement_write,
                credentials_read,
                anchor.project_descriptor,
                anchor.state_descriptor,
            )
            popen_kwargs: dict[str, Any] = {}
            inherited_handles: list[tuple[int, bool]] = []
            try:
                if os.name == "nt":
                    import msvcrt

                    startupinfo = subprocess.STARTUPINFO()
                    handle_list: list[int] = []
                    for descriptor in inherited_descriptors:
                        handle = msvcrt.get_osfhandle(descriptor)
                        inherited_handles.append(
                            (handle, os.get_handle_inheritable(handle))
                        )
                        os.set_handle_inheritable(handle, True)
                        handle_list.append(handle)
                    startupinfo.lpAttributeList = {"handle_list": handle_list}
                    popen_kwargs["startupinfo"] = startupinfo
                    popen_kwargs["creationflags"] = (
                        subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.CREATE_BREAKAWAY_FROM_JOB
                    )
                else:
                    popen_kwargs["pass_fds"] = inherited_descriptors
                    popen_kwargs["start_new_session"] = True
                spawn_command = [sys.executable, "-c", bootstrap]
                if os.name == "nt":
                    proc = _popen_windows_daemon(
                        spawn_command,
                        creationflags=popen_kwargs.pop("creationflags"),
                        cwd=str(base_dir),
                        env=env,
                        stdout=log,
                        stderr=log,
                        stdin=subprocess.DEVNULL,
                        close_fds=True,
                        **popen_kwargs,
                    )
                else:
                    proc = subprocess.Popen(
                        spawn_command,
                        cwd=str(base_dir),
                        env=env,
                        stdout=log,
                        stderr=log,
                        stdin=subprocess.DEVNULL,
                        close_fds=True,
                        **popen_kwargs,
                    )
            finally:
                for handle, was_inheritable in inherited_handles:
                    os.set_handle_inheritable(handle, was_inheritable)
        except BaseException:
            os.close(acknowledgement_read)
            os.close(release_write)
            os.close(credentials_write)
            raise
        finally:
            os.close(release_read)
            os.close(acknowledgement_write)
            os.close(credentials_read)
    finally:
        log.close()

    identity: tuple[int, str] | None = None
    try:
        _write_all(credentials_write, credentials_payload)
        os.close(credentials_write)
        credentials_write = -1
        registration = _await_child_registration(
            config,
            proc,
            acknowledgement_read,
            nonce,
        )
        if isinstance(registration, tuple):
            registered_pid, token = registration
        else:
            # Preserve compatibility with focused tests and older in-tree callers
            # that monkeypatch the registration helper to return only the token.
            registered_pid, token = proc.pid, registration
        identity = (registered_pid, token)
        # The outer start lifecycle holds the stable state-directory inode. Record
        # the exact acknowledged Popen identity before the child can import/bind the
        # server. If this parent dies first, the release pipe closes and the child
        # exits without ever serving.
        _record_spawn_identity(config, registered_pid, token)
        recorded_identities = _group_ledger_entries(_read_ledger(config))
        if recorded_identities != {registered_pid: [token]}:
            raise RuntimeError("daemon child registration is not durably unambiguous")
        _require_current_lifecycle_state_directory(config)
        _write_all(release_write, b"1")
    except BaseException as exc:
        terminated = _terminate_unrecorded_spawn(proc, identity)
        if identity is not None and terminated:
            _discard_spawn_identity(config, identity)
        else:
            _reap_orphans(config, keep_pid=None)
        if not isinstance(exc, Exception):
            raise
        raise RuntimeError("cannot complete daemon child registration") from exc
    finally:
        os.close(acknowledgement_read)
        os.close(release_write)
        if credentials_write >= 0:
            os.close(credentials_write)
    assert identity is not None
    return identity


def ensure_running(config: Config) -> tuple[str, str | None]:
    """Return (base_url, token), auto-starting a daemon if none is healthy (PR-9.4)."""
    found = discover(config)
    if found:
        return found

    # The daemon serves /health from a threadpool, but heavy C++ USD work (flattening /
    # packaging a large stage) can starve the GIL long enough that a 2s probe times out.
    # Spawning a *second* daemon then hijacks the state file: the old one becomes an
    # orphan and every later command sees a fresh, empty session ("no stage open").
    # If the recorded pid is alive, be patient before declaring the daemon dead.
    # A daemon busy with a long render/sim can miss the quick /health probe. WAIT it out
    # (up to BUSY_WAIT_S) with a longer probe rather than respawning (which orphans it and
    # loses the stage) or hard-failing (which drove agents to kill -9 loops). A live pid
    # means it will answer once the current operation finishes.
    # A zombie pid still passes `os.kill(pid, 0)`, so use the real process state here:
    # waiting 180s for a defunct daemon to "finish" would just stall every command.
    st = _read_state(config)
    owned_identity = _owned_process_identity(config, st) if st else None
    if owned_identity and _proc_state(owned_identity[0]) == "alive":
        pid = owned_identity[0]
        deadline = time.time() + BUSY_WAIT_S
        while time.time() < deadline:
            found = discover(config, health_timeout=6.0)
            if found:
                return found
            if _proc_state(pid) != "alive":
                break  # genuinely died (or went defunct) — fall through and spawn
            time.sleep(1.0)
        if _proc_state(pid) == "alive":
            raise RuntimeError(
                f"daemon (pid {pid}) is still busy after {BUSY_WAIT_S:.0f}s "
                "(a long render/sim on a large scene). Retry the command shortly — it will "
                "run once the current operation finishes. Do NOT `server stop`/kill unless "
                "it is truly wedged, or you'll lose the open stage."
            )

    if (
        config.server.get("lifecycle_owner") == "external"
        and os.environ.get(EXTERNAL_LIFECYCLE_BOOTSTRAP_ENV) != "1"
    ):
        raise DaemonStopRefused(
            "externally owned daemon lifecycle cannot auto-start a replacement"
        )

    spawned_identity: tuple[int, str] | None = None
    try:
        with _locked_start_lifecycle(config):
            found = discover(config)
            if found:
                return found
            spawned_identity = _spawn(config)

            deadline = time.time() + START_TIMEOUT_S
            while time.time() < deadline:
                found = discover(config)
                if found:
                    return found
                time.sleep(0.2)
            # Final grace probe with a generous per-request timeout: a warming
            # daemon can be alive but too slow for discover()'s 2s health probes.
            found = discover(config, health_timeout=10.0)
            if found:
                return found
            raise RuntimeError(
                f"daemon did not come up in {START_TIMEOUT_S:.0f}s "
                f"(see {config.state_dir / 'daemon.log'}; it may still be "
                f"starting — check 'usd-cli server status' before restarting)"
            )
    except BaseException:
        # Keep this handler OUTSIDE the lifecycle context. Its exit validation can
        # itself fail (for example, if the project/state directory name was replaced
        # after the daemon became ready); that failure must still reap the exact
        # in-memory child identity created by this attempt.
        if spawned_identity is not None:
            _cleanup_failed_spawn(config, spawned_identity)
        raise


# ── server start/stop/status (the `usd-cli server …` subcommands) ──────────────────
def start(config: Config) -> str:
    if config.server.get("lifecycle_owner") == "external":
        if (
            os.environ.get(EXTERNAL_LIFECYCLE_BOOTSTRAP_ENV) != "1"
            or _read_state(config) is not None
        ):
            raise DaemonStopRefused(
                "externally owned daemon lifecycle cannot be started by this process"
            )
    found = discover(config)
    if found:
        st = _read_state(config)
        return f"already running (pid {st['pid']}, port {st['port']})"
    base, _ = ensure_running(config)
    st = _read_state(config)
    return f"started (pid {st['pid']}, {base}, renderer {st.get('renderer')})"


def stop(config: Config) -> str:
    st = _read_state(config)
    if not st:
        # A pid:start-token ledger is durable crash evidence, not a project-secret
        # capability. Without current-project state and authenticated health it can
        # have been copied or forged, so generic stop may prune dead records but must
        # never signal a live process from the ledger alone.
        _reap_orphans(config, keep_pid=None)
        return "no daemon for this project"
    if st.get("lifecycle_owner") == "external":
        raise DaemonStopRefused(
            "externally owned daemon lifecycle cannot be stopped by this process"
        )
    identity = _recorded_state_identity(config, st)
    expected_project = _project_id(config)
    if identity is None:
        raise DaemonStopRefused(
            "refused to use malformed or foreign-project daemon state"
        )
    pid, process_start_token = identity
    current_process_start_token = _proc_start_token(pid)
    process_state = _proc_state(pid)
    if current_process_start_token != process_start_token:
        if process_state == "alive" and current_process_start_token is None:
            raise DaemonStopRefused(
                f"refused to signal unverifiable live daemon pid {pid}; "
                "preserved state and ledger"
            )
        current = _read_state(config)
        if current == st:
            try:
                config.server_state_path.unlink(missing_ok=True)
            except OSError:
                pass
        _discard_spawn_identity(config, (pid, process_start_token))
        return f"daemon (pid {pid}) had already exited; removed its state"
    try:
        health = _health(_base(st), st.get("token"))
    except RuntimeError:
        health = None
    verified = _health_matches_state(config, st, health)
    if (
        verified
        and process_state == "alive"
        and _proc_start_token(pid) == process_start_token
    ):
        ok = _terminate(pid, expected_token=process_start_token)
        if not ok:
            raise DaemonStopRefused(
                f"failed to stop authenticated daemon pid {pid}; preserved lifecycle "
                "evidence and refused restart"
            )
        current = _read_state(config)
        if (
            current
            and current.get("pid") == pid
            and current.get("process_start_token") == process_start_token
            and current.get("project_id") == expected_project
        ):
            try:
                config.server_state_path.unlink(missing_ok=True)
            except OSError:
                pass
        _discard_spawn_identity(config, (pid, process_start_token))
        _reap_orphans(config, keep_pid=None)
        return f"stopped (pid {pid})"
    if process_state != "alive":
        current = _read_state(config)
        if current == st:
            try:
                config.server_state_path.unlink(missing_ok=True)
            except OSError:
                pass
        _discard_spawn_identity(config, (pid, process_start_token))
        return f"daemon (pid {pid}) had already exited; removed its state"
    raise DaemonStopRefused(
        f"refused to signal unauthenticated live daemon pid {pid}; "
        "preserved state and ledger"
    )


def _reap_orphans(config: Config, keep_pid: int | None) -> int:
    """Prune proven-dead ledger records without signaling a live process.

    A mutable pid:start-token file is not sufficient authorization to signal. Exact
    spawn-failure paths retain an in-memory Popen/(pid, token), and normal stop also
    requires current-project state plus authenticated health before calling _terminate.
    """
    base_dir = config.project_dir or Path.cwd()
    killed = 0
    try:
        with _locked_ledger(config, create_state=False) as directory_descriptor:
            entries = _read_ledger_locked(directory_descriptor)
            if _read_ledger_locked(directory_descriptor) != entries:
                raise RuntimeError("daemon spawn ledger changed before reap")
            survivors: list[tuple[int, str]] = []
            for pid, tokens in _group_ledger_entries(entries).items():
                if keep_pid and pid == keep_pid:
                    survivors.extend((pid, token) for token in tokens)
                    continue
                # Conflicting externally supplied identities are not ownership proof.
                # Signal none and preserve every token for a future safe inspection.
                if len(tokens) != 1:
                    survivors.extend((pid, token) for token in tokens)
                    continue
                token = tokens[0]
                current_token = _proc_start_token(pid)
                if current_token != token:
                    # A different readable token proves pid reuse and is safe to
                    # forget. An unavailable token does not: retain the durable
                    # identity while the process is still alive.
                    if current_token is None and _proc_state(pid) == "alive":
                        survivors.append((pid, token))
                    continue
                survivors.append((pid, token))
            _write_ledger_locked(directory_descriptor, survivors)
    except (FileNotFoundError, OSError, RuntimeError):
        # Unsafe/malformed ledger input must remain untouched and signal nobody.
        pass
    _clean_fuse_litter(base_dir)
    return killed


def _clean_fuse_litter(base_dir: Path) -> None:
    """Delete `.fuse_hidden*` litter under the project. Never follows directory
    symlinks, never unlinks anything but a regular non-symlink file, and skips any
    match whose resolved location escapes the resolved project dir — the old
    recursive glob followed symlinked dirs and could delete matching files anywhere
    they pointed."""
    try:
        project_root = base_dir.resolve()
    except OSError:
        return
    for root, _dirs, files in os.walk(base_dir, followlinks=False):
        for name in files:
            if not name.startswith(".fuse_hidden"):
                continue
            path = Path(root) / name
            try:
                if path.is_symlink():
                    continue
                resolved = path.resolve()
                if project_root != resolved and project_root not in resolved.parents:
                    continue
                if not resolved.is_file():
                    continue
                os.unlink(path)
            except OSError:
                continue


def _attached_parent_status() -> str | None:
    """Report an authenticated parent daemon without requiring host PID visibility.

    Coding-agent sandboxes may place the child in a separate PID namespace.  The
    launcher names the canonical parent project explicitly, and
    :func:`discover_attached` authenticates that project's exact loopback endpoint
    and durable daemon identity.  Keep this path read-only: lifecycle operations
    retain the stronger host-process proof used by ``discover`` and ``stop``.
    """

    raw_project = os.environ.get(ATTACHED_PROJECT_DIR_ENV)
    if raw_project is None:
        return None
    try:
        project_dir = Path(raw_project).resolve(strict=True)
    except OSError:
        return "unverified attached parent daemon state (invalid project hint)"
    if str(project_dir) != raw_project or not project_dir.is_dir():
        return "unverified attached parent daemon state (invalid project hint)"

    attached_config = Config(project_dir=project_dir)
    state = _read_state(attached_config)
    if not state:
        return "attached parent daemon state not found"
    try:
        base = _base(state)
    except RuntimeError:
        return "unverified attached parent daemon endpoint (no signal sent)"
    if discover_attached(attached_config, base) is None:
        return (
            "unverified attached parent daemon state for pid "
            f"{state.get('pid')} (no signal sent)"
        )
    return (
        f"running (attached parent) — pid {state['pid']}, {base}, "
        "lifecycle external"
    )


def status(config: Config) -> str:
    attached_status = _attached_parent_status()
    if attached_status is not None:
        return attached_status
    st = _read_state(config)
    if st:
        identity = _recorded_state_identity(config, st)
        if identity is None:
            return f"unverified daemon state for pid {st.get('pid')} (no signal sent)"
        pid, process_start_token = identity
        current_process_start_token = _proc_start_token(pid)
        state = _proc_state(pid)
        if current_process_start_token != process_start_token:
            if state == "zombie":
                return (
                    f"died (zombie) — pid {st['pid']} is defunct and will never answer; "
                    "restart with `usd-cli server restart`"
                )
            if state != "alive" or current_process_start_token is not None:
                return (
                    "state file present but daemon identity is gone "
                    f"(stale pid {st.get('pid')})"
                )
            return (
                f"unverified live daemon state for pid {st.get('pid')} (no signal sent)"
            )
        # A freshly-restarted or busy daemon can miss one 2s probe; a second look with a
        # longer timeout avoids reporting a healthy process as "stale".
        try:
            base = _base(st)
        except RuntimeError:
            base = ""
        live = (base and _health(base, st.get("token"))) or (
            base and _health(base, st.get("token"), timeout=8.0)
        )
        if _health_matches_state(config, st, live):
            msg = (
                f"running — pid {st['pid']}, {base}, engine {live.get('engine')}, "
                f"renderer {live.get('renderer')}, stage {live.get('stage') or '(none)'}"
            )
            named = live.get("sessions") or {}
            if len(named) > 1:
                msg += f", sessions {len(named)} ({', '.join(sorted(named))})"
            return msg
        if live:
            return (
                f"unverified daemon health identity for pid {st.get('pid')} "
                "(no signal sent)"
            )
        if state == "zombie":
            return (
                f"died (zombie) — pid {st['pid']} is defunct and will never answer; "
                "restart with `usd-cli server restart`"
            )
        if state == "alive":
            return (
                f"busy — pid {st['pid']} is alive but not answering health checks "
                "(likely mid-operation)"
            )
        return f"state file present but daemon not healthy (stale pid {st.get('pid')})"
    return "not running"
