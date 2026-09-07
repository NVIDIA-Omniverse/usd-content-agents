# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one command and supervise descendant cleanup before returning.

This module is intentionally standard-library-only because the runner invokes it
with an isolated Python interpreter. It does not create or replace the model
provider's sandbox; it only owns the provider process tree for its lifetime. The
guarded child installs a parent-death signal and a control-plane seccomp filter
before ``exec``. The supervisor becomes a Linux subreaper and, after the provider
exits, repeatedly kills and polls direct or adopted children until two scans
separated by a scheduler delay both find the process tree empty. Reaping uses
nonblocking polls under a fixed deadline; if any descendant remains, the
supervisor fails closed so the caller cannot treat cleanup as complete. Missing
Linux primitives fail closed before the provider starts. After installing its
signal handlers and required supervisor state, the supervisor signals readiness
and waits for the runner to release provider startup.
"""

from __future__ import annotations

import ctypes
import errno
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_PR_SET_CHILD_SUBREAPER = 36
_PR_SET_DUMPABLE = 4
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_PDEATHSIG = 1
_SUPERVISOR_ERROR = 125
# Keep these one-byte pipe tokens in sync with runner.py. The runner does not
# release provider startup until it has received READY and written START.
_SUPERVISOR_READY_TOKEN = b"R"
_SUPERVISOR_START_TOKEN = b"S"
_DESCENDANT_REAP_TIMEOUT_SECONDS = 2.0
_DESCENDANT_REAP_POLL_SECONDS = 0.01
_DESCENDANT_PID_REPORT_LIMIT = 16
_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO = 0x00050000
_SCMP_CMP_MASKED_EQ = 7
_PID_T_MASK = (1 << 32) - 1
_CONTROL_SYSCALL_RULES = {
    "kill": (0,),
    "kcmp": (0, 1),
    "move_pages": (0,),
    "pidfd_getfd": (),
    "pidfd_open": (0,),
    "pidfd_send_signal": (),
    "process_madvise": (),
    "process_mrelease": (),
    "process_vm_readv": (),
    "process_vm_writev": (),
    "ptrace": (),
    "prlimit64": (0,),
    "rt_sigqueueinfo": (0,),
    "rt_tgsigqueueinfo": (0,),
    "sched_setaffinity": (0,),
    "sched_setattr": (0,),
    "sched_setparam": (0,),
    "sched_setscheduler": (0,),
    "setpriority": (1,),
    "tgkill": (0,),
    # tkill accepts only a TID, so exact TGID rules cannot protect every
    # control-plane thread. Modern runtimes use tgkill for self-signals.
    "tkill": (),
}
_CONTROL_COMMAND_RULES = {
    "fcntl": (1, {8, 10, 15}),  # F_SETOWN, F_SETSIG, F_SETOWN_EX
    "fcntl64": (1, {8, 10, 15}),
    "ioctl": (1, {0x8901, 0x8902}),  # FIOSETOWN, SIOCSPGRP
}


class _ScmpArgCmp(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def _uint32_argument_comparison(argument_index: int, value: int) -> _ScmpArgCmp:
    """Match the low 32 bits consumed by a kernel integer argument."""

    return _ScmpArgCmp(
        argument_index,
        _SCMP_CMP_MASKED_EQ,
        _PID_T_MASK,
        ctypes.c_uint32(value).value,
    )


def _load_control_plane_filter(
    *,
    protected_pids: set[int],
    protected_process_groups: set[int],
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))

    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_ScmpArgCmp),
    ]
    seccomp.seccomp_rule_add_array.restype = ctypes.c_int

    context = seccomp.seccomp_init(_SCMP_ACT_ALLOW)
    if not context:
        raise RuntimeError("libseccomp could not create a filter context")
    deny_action = _SCMP_ACT_ERRNO | errno.EPERM
    try:
        for syscall_name, argument_indexes in _CONTROL_SYSCALL_RULES.items():
            syscall_number = seccomp.seccomp_syscall_resolve_name(
                syscall_name.encode("ascii")
            )
            if syscall_number < 0:
                continue
            if not argument_indexes:
                result = seccomp.seccomp_rule_add_array(
                    context,
                    deny_action,
                    syscall_number,
                    0,
                    None,
                )
                if result != 0:
                    raise OSError(-result, os.strerror(-result))
                continue
            for argument_index in argument_indexes:
                for protected_pid in protected_pids:
                    comparison = _uint32_argument_comparison(
                        argument_index, protected_pid
                    )
                    result = seccomp.seccomp_rule_add_array(
                        context,
                        deny_action,
                        syscall_number,
                        1,
                        ctypes.pointer(comparison),
                    )
                    if result != 0:
                        raise OSError(-result, os.strerror(-result))
                if syscall_name == "kill" and argument_index == 0:
                    denied_targets = {-1, *(-pgid for pgid in protected_process_groups)}
                    for denied_target in denied_targets:
                        comparison = _uint32_argument_comparison(
                            argument_index, denied_target
                        )
                        result = seccomp.seccomp_rule_add_array(
                            context,
                            deny_action,
                            syscall_number,
                            1,
                            ctypes.pointer(comparison),
                        )
                        if result != 0:
                            raise OSError(-result, os.strerror(-result))
        for syscall_name, (
            argument_index,
            denied_commands,
        ) in _CONTROL_COMMAND_RULES.items():
            syscall_number = seccomp.seccomp_syscall_resolve_name(
                syscall_name.encode("ascii")
            )
            if syscall_number < 0:
                continue
            for denied_command in denied_commands:
                comparison = _uint32_argument_comparison(
                    argument_index,
                    denied_command,
                )
                result = seccomp.seccomp_rule_add_array(
                    context,
                    deny_action,
                    syscall_number,
                    1,
                    ctypes.pointer(comparison),
                )
                if result != 0:
                    raise OSError(-result, os.strerror(-result))
        result = seccomp.seccomp_load(context)
        if result != 0:
            raise OSError(-result, os.strerror(-result))
    finally:
        seccomp.seccomp_release(context)


def _exec_guarded_child(arguments: list[str]) -> int:
    if len(arguments) < 4 or arguments[2] != "--":
        print("descendant-reaper: invalid guarded-child arguments", file=sys.stderr)
        return _SUPERVISOR_ERROR
    supervisor_pid = int(arguments[0])
    runner_pid = int(arguments[1])
    command = arguments[3:]
    if not command:
        print("descendant-reaper: guarded child missing command", file=sys.stderr)
        return _SUPERVISOR_ERROR
    if os.getppid() != supervisor_pid:
        print(
            "descendant-reaper: supervisor exited before child guard", file=sys.stderr
        )
        return _SUPERVISOR_ERROR
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        print(
            f"descendant-reaper: could not set child parent-death signal: "
            f"{os.strerror(error_number)}",
            file=sys.stderr,
        )
        return _SUPERVISOR_ERROR
    if os.getppid() != supervisor_pid:
        return _SUPERVISOR_ERROR
    try:
        _load_control_plane_filter(
            protected_pids={supervisor_pid, runner_pid},
            protected_process_groups={supervisor_pid, os.getpgid(runner_pid)},
        )
        os.execvpe(command[0], command, os.environ)
    except Exception as exc:  # noqa: BLE001 - isolated process boundary
        print(f"descendant-reaper: guarded child setup failed: {exc}", file=sys.stderr)
        return _SUPERVISOR_ERROR


def _configure_supervisor(parent_pid: int) -> None:
    if sys.platform != "linux":
        raise RuntimeError("descendant reaping requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != parent_pid:
        raise RuntimeError("runner exited while descendant reaper was starting")
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    _direct_child_pids()


def _direct_child_pids() -> set[int]:
    children_path = Path(f"/proc/self/task/{os.getpid()}/children")
    try:
        payload = children_path.read_text(encoding="ascii").strip()
    except FileNotFoundError as exc:
        raise RuntimeError("Linux /proc child tracking is unavailable") from exc
    if not payload:
        return set()
    return {int(value) for value in payload.split()}


def _reap_exited_children(*, protected_pids: set[int] | None = None) -> None:
    if protected_pids:
        for child_pid in sorted(_direct_child_pids() - protected_pids):
            while True:
                try:
                    os.waitpid(child_pid, os.WNOHANG)
                    break
                except ChildProcessError:
                    break
                except InterruptedError:
                    continue
        return
    while True:
        try:
            waited_pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        except InterruptedError:
            continue
        if waited_pid == 0:
            return


def _kill_and_reap_descendants(
    *,
    timeout_seconds: float = _DESCENDANT_REAP_TIMEOUT_SECONDS,
    poll_interval_seconds: float = _DESCENDANT_REAP_POLL_SECONDS,
) -> set[int]:
    if timeout_seconds < 0:
        raise ValueError("descendant reap timeout must be non-negative")
    if poll_interval_seconds <= 0:
        raise ValueError("descendant reap poll interval must be positive")

    deadline = time.monotonic() + timeout_seconds
    attempted_kills: set[int] = set()
    saw_empty_scan = False
    while True:
        _reap_exited_children()
        children = _direct_child_pids()
        attempted_kills.intersection_update(children)
        if not children:
            if saw_empty_scan:
                return set()
            # Give a just-exiting intermediary one scheduling turn to reparent
            # its children to this subreaper before declaring the tree empty.
            saw_empty_scan = True
        else:
            saw_empty_scan = False
        for child_pid in children - attempted_kills:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                print(
                    f"descendant-reaper: could not SIGKILL descendant "
                    f"{child_pid}: {exc}",
                    file=sys.stderr,
                )
            attempted_kills.add(child_pid)

        now = time.monotonic()
        if children and now >= deadline:
            _reap_exited_children()
            return _direct_child_pids()
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - now)))


def _unreaped_descendants_message(
    child_pids: set[int],
    *,
    timeout_seconds: float = _DESCENDANT_REAP_TIMEOUT_SECONDS,
) -> str:
    ordered_pids = sorted(child_pids)
    displayed = ", ".join(
        str(pid) for pid in ordered_pids[:_DESCENDANT_PID_REPORT_LIMIT]
    )
    omitted_count = len(ordered_pids) - _DESCENDANT_PID_REPORT_LIMIT
    if omitted_count > 0:
        displayed = f"{displayed}, ... ({omitted_count} more)"
    return (
        f"{len(ordered_pids)} descendant(s) remained after "
        f"{timeout_seconds:.1f}s: {displayed}"
    )


def _print_cleanup_timeout(
    child_pids: set[int],
    *,
    after: BaseException | None = None,
    timeout_seconds: float = _DESCENDANT_REAP_TIMEOUT_SECONDS,
) -> None:
    context = f" after {after}" if after is not None else ""
    print(
        f"descendant-reaper: cleanup timed out{context}: "
        + _unreaped_descendants_message(
            child_pids,
            timeout_seconds=timeout_seconds,
        ),
        file=sys.stderr,
    )


def _forward_signal(target: subprocess.Popen[bytes] | None, signum: int) -> None:
    if target is None:
        return
    if target.poll() is not None:
        return
    try:
        os.killpg(target.pid, signum)
    except ProcessLookupError:
        return


def _target_exit_code(returncode: int) -> int:
    if returncode < 0:
        return 128 + abs(returncode)
    return returncode


def _supervisor_command(arguments: list[str]) -> tuple[int, int, list[str]]:
    if (
        len(arguments) < 5
        or arguments[0] != "--ready-fd"
        or arguments[2] != "--start-fd"
        or arguments[4] != "--"
    ):
        raise ValueError("missing or invalid readiness protocol")
    ready_fd = int(arguments[1])
    start_fd = int(arguments[3])
    command = arguments[5:]
    if ready_fd < 0 or start_fd < 0 or ready_fd == start_fd:
        raise ValueError("invalid readiness file descriptors")
    if not command:
        raise ValueError("missing command")
    return ready_fd, start_fd, command


def _close_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _signal_supervisor_ready(ready_fd: int) -> None:
    if os.write(ready_fd, _SUPERVISOR_READY_TOKEN) != len(_SUPERVISOR_READY_TOKEN):
        raise RuntimeError("short readiness write")


def _await_runner_start(start_fd: int) -> None:
    token = os.read(start_fd, len(_SUPERVISOR_START_TOKEN))
    if token != _SUPERVISOR_START_TOKEN:
        raise RuntimeError("runner did not confirm supervisor readiness")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--guarded-child"]:
        return _exec_guarded_child(arguments[1:])
    try:
        ready_fd, start_fd, arguments = _supervisor_command(arguments)
    except (TypeError, ValueError) as exc:
        print(f"descendant-reaper: {exc}", file=sys.stderr)
        return _SUPERVISOR_ERROR

    target: subprocess.Popen[bytes] | None = None
    requested_signal: int | None = None
    force_kill_at: float | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal requested_signal, force_kill_at
        requested_signal = signum
        force_kill_at = time.monotonic() + 1.0
        _forward_signal(target, signum)

    for candidate in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(candidate, handle_signal)

    try:
        parent_pid = os.getppid()
        _configure_supervisor(parent_pid)
        _signal_supervisor_ready(ready_fd)
        _close_fd(ready_fd)
        ready_fd = -1
        # Do not spawn the provider until the runner has positively observed
        # readiness. Closing the start pipe makes every pre-readiness failure
        # exit without creating descendants.
        _await_runner_start(start_fd)
        _close_fd(start_fd)
        start_fd = -1
        target = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                str(Path(__file__).resolve()),
                "--guarded-child",
                str(os.getpid()),
                str(parent_pid),
                "--",
                *arguments,
            ],
            start_new_session=True,
        )
        if requested_signal is not None:
            _forward_signal(target, requested_signal)
    except Exception as exc:  # noqa: BLE001 - isolated process boundary
        _close_fd(ready_fd)
        _close_fd(start_fd)
        print(f"descendant-reaper: failed to start target: {exc}", file=sys.stderr)
        return _SUPERVISOR_ERROR

    try:
        assert target is not None
        while target.poll() is None:
            # As a subreaper, this supervisor adopts short-lived sandbox
            # descendants while the provider is still running. Reap those
            # promptly so a long session cannot exhaust the process table, but
            # leave the provider itself for subprocess to collect with its
            # exact return status.
            _reap_exited_children(protected_pids={target.pid})
            if force_kill_at is not None and time.monotonic() >= force_kill_at:
                _forward_signal(target, signal.SIGKILL)
                force_kill_at = None
            time.sleep(0.02)
        returncode = int(target.wait())
        unreaped_descendants = _kill_and_reap_descendants()
        if unreaped_descendants:
            _print_cleanup_timeout(unreaped_descendants)
            return _SUPERVISOR_ERROR
    except Exception as exc:  # noqa: BLE001 - isolated process boundary
        _forward_signal(target, signal.SIGKILL)
        try:
            target.wait(timeout=2)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass
        try:
            unreaped_descendants = _kill_and_reap_descendants()
        except Exception as cleanup_exc:  # noqa: BLE001 - preserve both failures
            print(
                f"descendant-reaper: cleanup failed after {exc}: {cleanup_exc}",
                file=sys.stderr,
            )
        else:
            if unreaped_descendants:
                _print_cleanup_timeout(unreaped_descendants, after=exc)
            else:
                print(f"descendant-reaper: cleanup failed: {exc}", file=sys.stderr)
        return _SUPERVISOR_ERROR

    if requested_signal is not None and returncode == 0:
        returncode = 128 + requested_signal
    return _target_exit_code(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
