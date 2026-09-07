# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""usd-cli-tel entry point: exec usd-cli transparently, record one span.

Transparency contract:
  * every argv token after `usd-cli-tel` goes to usd-cli byte-for-byte
    (configuration is env-only, so no flag can ever collide),
  * stdin and stdout are inherited untouched (pipes, TTYs, binary output all
    behave exactly as with bare usd-cli),
  * stderr is teed through a pipe so error tails can be recorded — output
    still reaches the terminal unmodified,
  * the child's exit code is mirrored (128+N for signal deaths),
  * telemetry failures are swallowed; USD_CLI_TEL_DISABLED=1 downgrades to a
    plain os.exec of the target.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .backends import emit_all
from .config import Config
from .span import (
    build_record,
    format_traceparent,
    new_span_id,
    new_trace_id,
    parse_traceparent,
)

_STDERR_TAIL_BYTES = 4096
_EXTERNAL_LIFECYCLE_ENV = "USD_CLI_LIFECYCLE_EXTERNALLY_OWNED"
_PARENT_MANAGED_SERVER_ACTIONS = frozenset({"start", "stop", "restart"})
_GLOBAL_FLAGS = frozenset({"--json", "-q", "--quiet", "--version"})
_GLOBAL_OPTIONS = frozenset({"--server", "--session", "--timeout"})


def _parent_managed_lifecycle_action(argv: list[str]) -> str | None:
    """Return a forbidden explicit lifecycle action from one wrapper argv."""

    if not os.environ.get(_EXTERNAL_LIFECYCLE_ENV):
        return None
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            index += 1
            break
        if argument in _GLOBAL_FLAGS:
            index += 1
            continue
        if argument in _GLOBAL_OPTIONS:
            index += 2
            continue
        if any(argument.startswith(f"{option}=") for option in _GLOBAL_OPTIONS):
            index += 1
            continue
        break
    if (
        index + 1 < len(argv)
        and argv[index] == "server"
        and argv[index + 1] in _PARENT_MANAGED_SERVER_ACTIONS
    ):
        return argv[index + 1]
    return None


_ACTION_LOCK_POLL_SECONDS = 0.05
_CHILD_TERMINATION_GRACE_SECONDS = 2.0


def _process_start_token(pid: int) -> str | None:
    """Return a platform process-start token used to reject PID reuse."""

    if os.name == "nt":
        from usd_core.windows_files import windows_process_start_token

        return windows_process_start_token(pid)

    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    # Field 2 (comm) is parenthesized but otherwise unescaped and may contain
    # spaces or parentheses. Split after its final closing parenthesis so the
    # documented positions of state and starttime remain stable.
    _prefix, separator, remainder = raw.rpartition(")")
    if not separator:
        return None
    fields = remainder.split()
    if not fields or fields[0] == "Z":
        return None
    return fields[19] if len(fields) > 19 else None


def _action_owner() -> dict[str, object]:
    """Identify the shell process that owns one agent command execution."""

    pid = _outer_pid_namespace_process_group()
    scope = "outer_process_group"
    if pid is None:
        pid = os.getppid()
        scope = "parent_process"
    return {
        "pid": pid,
        "start_token": _process_start_token(pid),
        "scope": scope,
    }


def _outer_pid_namespace_process_group() -> int | None:
    """Resolve the host-visible action group from a nested PID namespace.

    Codex gives each sandboxed shell tool call a fresh PID namespace. Inside
    that namespace the wrapper's parent is PID 1 for every action, while the
    first ``NSpgid`` value remains the distinct host process-group leader for
    the outer sandbox. Use it only when ``NSpid`` proves that nesting exists;
    ordinary local shells retain the parent-process fallback.
    """

    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    fields: dict[str, list[str]] = {}
    for line in status.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"NSpid", "NSpgid"}:
            fields[key] = value.split()
    namespace_pids = fields.get("NSpid", [])
    process_groups = fields.get("NSpgid", [])
    if len(namespace_pids) <= 1 or not process_groups:
        return None
    try:
        group = int(process_groups[0])
    except ValueError:
        return None
    return group if group > 1 else None


def _live_action_owner(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    pid = value.get("pid")
    token = value.get("start_token")
    if not isinstance(pid, int) or pid <= 1 or not isinstance(token, str):
        return False
    return _process_start_token(pid) == token


def _same_action_owner(left: object, right: object) -> bool:
    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and left.get("pid") == right.get("pid")
        and left.get("start_token") == right.get("start_token")
    )


def _read_action_owner(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _publish_action_owner(path: Path, encoded: bytes) -> bool:
    """Atomically install a fully written owner record without replacement."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("failed to write usd-cli action lease owner")
            offset += written
        os.fsync(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            return False
        return True
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _acquire_action_lease(path: Path, *, timeout_seconds: float) -> None:
    """Serialize commands at the agent shell-action boundary.

    usd-cli's daemon serializes individual commands within a session. An agent
    can still launch two shell tool calls concurrently, though, allowing a
    many-command edit loop to interleave with save or render. This lightweight
    lease remains owned by the wrapper's parent shell after each individual
    wrapper process exits. A competing shell waits until that owner exits,
    while the same shell can issue any number of consecutive usd-cli calls.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    owner = _action_owner()
    if owner["start_token"] is None:
        print(
            "usd-cli-tel: action serialization unavailable "
            "(process start token unavailable)",
            file=sys.stderr,
        )
        return
    deadline = time.monotonic() + timeout_seconds
    encoded = (json.dumps(owner, sort_keys=True) + "\n").encode("utf-8")
    # A kernel-held guard makes stale-owner inspection and removal one atomic
    # recovery transaction. It is distinct from the durable shell-owner lease,
    # which intentionally remains after this short acquisition function exits.
    guard_path = path.with_name(f".{path.name}.guard")
    guard_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        guard_flags |= os.O_NOFOLLOW
    guard_descriptor = os.open(guard_path, guard_flags, 0o600)
    try:
        guard_stat = os.fstat(guard_descriptor)
        guard_path_stat = guard_path.lstat()
    except Exception:
        os.close(guard_descriptor)
        raise
    if (
        not os.path.samestat(guard_stat, guard_path_stat)
        or not stat.S_ISREG(guard_stat.st_mode)
        or guard_stat.st_nlink != 1
        or (
            os.name == "posix"
            and (
                guard_stat.st_uid != os.geteuid()
                or stat.S_IMODE(guard_stat.st_mode) != 0o600
            )
        )
        or bool(
            getattr(guard_path_stat, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    ):
        os.close(guard_descriptor)
        raise RuntimeError("usd-cli action lease guard is unsafe")
    try:
        if os.name == "nt" and guard_stat.st_size == 0:
            os.write(guard_descriptor, b"\0")
            os.fsync(guard_descriptor)

        def lock_guard() -> bool:
            os.lseek(guard_descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(guard_descriptor, msvcrt.LK_NBLCK, 1)
                except OSError:
                    return False
                return True
            import fcntl

            try:
                fcntl.flock(guard_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True

        def unlock_guard() -> None:
            os.lseek(guard_descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(guard_descriptor, msvcrt.LK_UNLCK, 1)
                return
            import fcntl

            fcntl.flock(guard_descriptor, fcntl.LOCK_UN)

        while True:
            guard_locked = False
            try:
                guard_locked = lock_guard()
                if guard_locked:
                    lease_exists = os.path.lexists(path)
                    existing = _read_action_owner(path) if lease_exists else None
                    if lease_exists:
                        if _same_action_owner(existing, owner):
                            return
                        if not _live_action_owner(existing):
                            path.unlink(missing_ok=True)
                            lease_exists = False
                    if not lease_exists and _publish_action_owner(path, encoded):
                        return
            finally:
                if guard_locked:
                    unlock_guard()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out waiting for another agent shell action to release "
                    f"the usd-cli lease: {path}"
                )
            time.sleep(_ACTION_LOCK_POLL_SECONDS)
    finally:
        os.close(guard_descriptor)


def _is_self(path: str) -> bool:
    try:
        return os.path.realpath(path) == os.path.realpath(sys.argv[0])
    except OSError:
        return False


def _resolve_target(cfg: Config) -> str | None:
    # Explicit path target: resolve directly (still refuse to wrap ourselves —
    # USD_CLI_TEL_TARGET=usd-cli-tel would recurse).
    if os.sep in cfg.target:
        target = shutil.which(cfg.target)
        if target is None or _is_self(target):
            return None
        return target
    # PATH scan that skips self, so a `usd-cli` → usd-cli-tel shim earlier on
    # PATH transparently wraps the next real usd-cli instead of recursing.
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        candidate = shutil.which(os.path.join(directory, cfg.target))
        if candidate is None:
            continue
        if _is_self(candidate):
            continue
        return candidate
    return None


def _pump_stderr(
    pipe,
    sink,
    tail: bytearray,
    lock: threading.Lock,
    failed: threading.Event,
) -> None:
    try:
        while True:
            chunk = pipe.read1(65536)
            if not chunk:
                break
            try:
                sink.write(chunk)
                sink.flush()
            except (BrokenPipeError, ValueError):
                pass
            with lock:
                tail.extend(chunk)
                if len(tail) > _STDERR_TAIL_BYTES:
                    del tail[: len(tail) - _STDERR_TAIL_BYTES]
    except Exception:
        # Closing the only read end prevents a verbose child from blocking on a
        # full orphaned pipe. The parent loop observes this and terminates the
        # child, keeping telemetry machinery from hanging the wrapped command.
        failed.set()
    finally:
        pipe.close()


def run() -> None:
    cfg = Config.from_env()
    argv = sys.argv[1:]
    lifecycle_action = _parent_managed_lifecycle_action(argv)
    if lifecycle_action is not None:
        print(
            "usd-cli-tel: externally owned usd-cli sessions forbid "
            f"`server {lifecycle_action}` in child processes",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if cfg.disabled:
        # Resolve through the same self-skipping scan as the active path so a
        # `usd-cli` → usd-cli-tel PATH shim + USD_CLI_TEL_DISABLED=1 can't
        # re-exec this wrapper forever (bare shutil.which would return the shim).
        target = _resolve_target(cfg)
        if target is None:
            print(f"usd-cli-tel: {cfg.target!r} not found on PATH", file=sys.stderr)
            raise SystemExit(127)
        os.execv(target, [cfg.target, *argv])  # never returns

    target = _resolve_target(cfg)
    if target is None:
        print(f"usd-cli-tel: {cfg.target!r} not found on PATH", file=sys.stderr)
        raise SystemExit(127)

    if cfg.action_lock_path is not None:
        try:
            _acquire_action_lease(
                cfg.action_lock_path,
                timeout_seconds=cfg.action_lock_timeout,
            )
        except Exception as exc:
            print(f"usd-cli-tel: action serialization failed: {exc}", file=sys.stderr)
            raise SystemExit(75) from exc

    # Distributed trace context: a valid W3C TRACEPARENT from the parent
    # (agent harness, workflow wrapper, CI) wins over USD_CLI_TEL_TRACE_ID.
    # The wrapped child gets TRACEPARENT rewritten to name this wrapper span
    # as its parent, so future daemon-emitted spans slot under it.
    parent_ctx = parse_traceparent(os.environ.get("TRACEPARENT"))
    if parent_ctx is not None:
        trace_id, parent_span_id = parent_ctx
    else:
        trace_id, parent_span_id = (cfg.trace_id or new_trace_id()), None
    span_id = new_span_id()
    child_env = {**os.environ, "TRACEPARENT": format_traceparent(trace_id, span_id)}

    tail = bytearray()
    tail_lock = threading.Lock()
    start_unix_nano = time.time_ns()
    start_perf = time.perf_counter_ns()

    proc = subprocess.Popen(
        [cfg.target, *argv], executable=target, stderr=subprocess.PIPE, env=child_env
    )
    pump_failed = threading.Event()
    pump = threading.Thread(
        target=_pump_stderr,
        args=(proc.stderr, sys.stderr.buffer, tail, tail_lock, pump_failed),
        daemon=True,
    )
    pump.start()

    # SIGINT already reaches the child through the foreground process group.
    # Forward SIGTERM explicitly, then bound shutdown even when the child
    # ignores it so a supervisor cannot leave the wrapper waiting forever.
    termination_deadline: float | None = None
    child_killed = False

    def request_child_shutdown(signum: int) -> None:
        nonlocal termination_deadline
        if proc.poll() is not None:
            return
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            return
        if termination_deadline is None:
            termination_deadline = (
                time.monotonic() + _CHILD_TERMINATION_GRACE_SECONDS
            )

    signal.signal(signal.SIGTERM, lambda signum, _frame: request_child_shutdown(signum))
    while proc.poll() is None:
        if pump_failed.is_set() and termination_deadline is None:
            request_child_shutdown(signal.SIGTERM)
        if (
            termination_deadline is not None
            and time.monotonic() >= termination_deadline
            and not child_killed
        ):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            child_killed = True
        try:
            proc.wait(timeout=0.05)
        except subprocess.TimeoutExpired:
            continue
        except KeyboardInterrupt:
            request_child_shutdown(signal.SIGINT)
            continue
    pump.join(timeout=2.0)

    end_unix_nano = start_unix_nano + (time.perf_counter_ns() - start_perf)
    rc = proc.returncode
    exit_code = 128 - rc if rc < 0 else rc  # -SIGINT → 130 etc.

    try:
        with tail_lock:
            stderr_tail = tail.decode("utf-8", errors="replace").strip()
        record = build_record(
            argv=argv,
            executable=target,
            exit_code=exit_code,
            start_unix_nano=start_unix_nano,
            end_unix_nano=end_unix_nano,
            stderr_tail=stderr_tail if exit_code != 0 else "",
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            trace_state=os.environ.get("TRACESTATE") if parent_ctx else None,
            extra_attributes=cfg.extra_attrs,
        )
        emit_all(record, cfg)
    except Exception as exc:  # telemetry must never change the outcome
        if cfg.debug:
            print(f"usd-cli-tel: telemetry failed: {exc}", file=sys.stderr)

    raise SystemExit(exit_code)


if __name__ == "__main__":
    run()
