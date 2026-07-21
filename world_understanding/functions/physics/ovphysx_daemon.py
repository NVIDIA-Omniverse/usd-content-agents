# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent ovphysx subprocess client.

This is the parent-side companion to
:mod:`world_understanding.functions.physics._ovphysx_daemon_script`. It
mirrors the well-trodden ``_OvRTXDaemon`` pattern in
``world_understanding/functions/graphics/render_ovrtx.py`` —
JSON-line-over-stdio, lazy-start, lock-serialized, crash-restart — and
shifts ovphysx into its own venv so the parent's ``usd-core`` stays
out of ovphysx's process.

Why a daemon at all:

* ovphysx initialization (``PhysX(...)``) takes a few seconds. Spawning
  it per tune trial would dominate the optimizer budget.
* ovphysx ships its own OpenUSD 25.11 and refuses to coexist with
  ``usd-core`` in the same Python process. The daemon process never
  imports ``pxr``; the parent uses ``pxr`` for scene authoring; the two
  never meet.

Determinism contract: ``daemon.evaluate(scene, ...)`` called N times
in a row MUST return the same trajectory each time. The daemon-side
script enforces this by tearing down the previous trial's USD +
tensor bindings before each new ``evaluate`` (see
:mod:`._ovphysx_daemon_script`); the integration test
``test_ovphysx_determinism_across_resets`` is the gate.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import platform
import queue
import selectors
import subprocess
import threading
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Default location for the daemon's ovphysx venv. Mirrors the
# ``~/.cache/wu/ovrtx_venv/`` precedent from render_ovrtx.py.
_DEFAULT_VENV_DIR = Path(
    os.environ.get("WU_OVPHYSX_VENV_DIR", str(Path.home() / ".cache/wu/ovphysx_venv"))
).expanduser()


_THIS_DIR = Path(__file__).resolve().parent
_DAEMON_SCRIPT_PATH = _THIS_DIR / "_ovphysx_daemon_script.py"
_OVPHYSX_RUNTIME_LOCK = Path("apps/physics_agent/runtime/pylock.ovphysx-runtime.toml")
_OVPHYSX_RUNTIME_LOCK_AARCH64 = Path(
    "apps/physics_agent/runtime/pylock.ovphysx-runtime.aarch64.toml"
)
_OVPHYSX_RUNTIME_READY_MARKER = ".wu-ovphysx-runtime-ready"
_STDERR_TAIL_LINES = 40
_STDERR_TAIL_CHARS = 4000


def _ovphysx_runtime_lock(machine: str | None = None) -> Path:
    """Return the reviewed daemon lock for the current machine architecture."""
    machine = (machine or platform.machine()).lower()
    if machine in {"aarch64", "arm64"}:
        return _OVPHYSX_RUNTIME_LOCK_AARCH64
    if machine in {"x86_64", "amd64"}:
        return _OVPHYSX_RUNTIME_LOCK
    raise ValueError(f"Unsupported architecture for OvPhysX runtime: {machine}")


def _ovphysx_venv_python_path(venv_dir: Path) -> Path:
    """Return the platform-specific Python executable path for a venv."""

    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ovphysx_venv_python_candidates(venv_dir: Path) -> tuple[Path, ...]:
    """Return preferred and compatibility daemon Python locations."""

    preferred = _ovphysx_venv_python_path(venv_dir)
    fallback = (
        venv_dir / "bin" / "python"
        if os.name == "nt"
        else venv_dir / "Scripts" / "python.exe"
    )
    return (preferred, fallback)


def ovphysx_runtime_available(venv_dir: Path | None = None) -> bool:
    """Return whether a successfully probed OvPhysX runtime is provisioned."""

    if venv_dir is None:
        venv_dir = Path(
            os.environ.get("WU_OVPHYSX_VENV_DIR", str(_DEFAULT_VENV_DIR))
        ).expanduser()
    python_available = any(
        candidate.is_file() for candidate in _ovphysx_venv_python_candidates(venv_dir)
    )
    return python_available and (venv_dir / _OVPHYSX_RUNTIME_READY_MARKER).is_file()


def _ovphysx_unavailable_message(venv_dir: Path = _DEFAULT_VENV_DIR) -> str:
    python_path = _ovphysx_venv_python_path(venv_dir)
    runtime_lock = _ovphysx_runtime_lock()
    return (
        "ovphysx daemon is not available. The daemon venv at "
        f"{venv_dir} either does not exist or does not have ovphysx "
        "installed. From a source checkout's repository root, bootstrap the exact "
        "runtime with:\n"
        f"  uv venv --python 3.12 {venv_dir}\n"
        f"  uv pip install --python {python_path} --require-hashes --no-deps "
        f"-r {runtime_lock} --no-config --no-sources\n"
        f'  env -u PYTHONPATH {python_path} -c "from ovphysx import PhysX; '
        "physics = PhysX(device='cpu'); physics.release()\" && "
        f"touch {venv_dir / _OVPHYSX_RUNTIME_READY_MARKER}\n"
        "Or override the venv path with WU_OVPHYSX_VENV_DIR=/some/path."
    )


class OvPhysXDaemonError(RuntimeError):
    """Raised when an ovphysx daemon call fails after start-up.

    Wraps the ``error`` field of a daemon JSON error response or any
    parent-side IO/protocol failure that doesn't fall under
    :class:`OvPhysXDaemonUnavailableError` (which is reserved for the
    "daemon could not start at all" case).
    """


class OvPhysXDaemonUnavailableError(OvPhysXDaemonError):
    """The daemon venv / python / startup handshake failed.

    The error message is the actionable install hint mandated by the
    issue body — callers (CLI, REST, runner) surface it verbatim.
    """

    DEFAULT_MESSAGE = "ovphysx daemon is not available."

    def __init__(self, message: str | None = None) -> None:
        default_message = _ovphysx_unavailable_message() if message is None else message
        super().__init__(default_message)


class _OvPhysXDaemon:
    """Persistent ovphysx subprocess wrapping ``ovphysx.PhysX``.

    Lifecycle:

    * Lazy-start: the subprocess is spawned on the first
      :meth:`evaluate` (or :meth:`reset_only`, or :meth:`ensure_running`)
      call. Subsequent calls reuse the same process.
    * Lock-serialized: ``threading.Lock`` around every command so two
      threads cannot interleave JSON requests on the shared pipe.
    * Crash-restart: a broken pipe or daemon exit triggers a clean
      restart on the next call (the previous process is reaped).
    * Atexit shutdown: the registered ``atexit`` hook sends a
      ``shutdown`` command and reaps the subprocess so a parent
      shutdown does not leave a zombie ovphysx process holding GPU
      memory.

    Configuration:

    * ``venv_dir``: ovphysx's own venv (default
      ``~/.cache/wu/ovphysx_venv``). Override via
      ``WU_OVPHYSX_VENV_DIR``.
    * ``device``: passed through to the daemon's ``PhysX(device=...)``.
      Only relevant on the FIRST trial of a daemon's life — ovphysx
      locks the device per-process. Default ``"auto"``.
    * Timeouts via env: ``WU_OVPHYSX_DAEMON_START_TIMEOUT`` (seconds,
      default 300), ``WU_OVPHYSX_DAEMON_EVALUATE_TIMEOUT`` (default
      1800).
    """

    def __init__(
        self,
        *,
        venv_dir: Path | None = None,
        device: str = "auto",
    ) -> None:
        self._venv_dir = Path(venv_dir) if venv_dir is not None else _DEFAULT_VENV_DIR
        self._device = device
        self._process: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._stderr_lock = threading.Lock()
        self._stdout_buffer = b""
        self._lock = threading.Lock()
        self._start_timeout_s = float(
            os.environ.get("WU_OVPHYSX_DAEMON_START_TIMEOUT", "300")
        )
        self._evaluate_timeout_s = float(
            os.environ.get("WU_OVPHYSX_DAEMON_EVALUATE_TIMEOUT", "1800")
        )
        atexit.register(self._atexit_shutdown)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_running(self) -> None:
        """Start the daemon if it is not already running. Used by tests."""
        with self._lock:
            if not self._is_running():
                self._start()

    def _resolve_python(self) -> Path:
        """Return the daemon python path; raise if the venv isn't usable."""
        for python_path in _ovphysx_venv_python_candidates(self._venv_dir):
            if python_path.exists():
                return python_path
        raise OvPhysXDaemonUnavailableError(
            _ovphysx_unavailable_message(self._venv_dir)
        )

    def _start(self) -> None:
        """Launch the daemon subprocess and wait for the ``ready`` line."""
        python = self._resolve_python()

        env = os.environ.copy()
        # Strip PYTHONPATH so the parent's ``usd-core`` cannot leak into
        # the daemon process. ovphysx's own bundled USD must win.
        env.pop("PYTHONPATH", None)
        # Honor the configured device on first start.
        env["WU_OVPHYSX_DEVICE"] = self._device

        if not _DAEMON_SCRIPT_PATH.exists():
            raise OvPhysXDaemonUnavailableError(
                f"daemon script missing at {_DAEMON_SCRIPT_PATH}"
            )

        logger.info("Starting ovphysx daemon subprocess (%s)", python)
        self._stdout_buffer = b""
        with self._stderr_lock:
            self._stderr_tail.clear()
        # Wrap ``Popen`` in the daemon-unavailable error class so spawn-time
        # failures (interpreter missing despite the venv check, fork EAGAIN,
        # missing libs, EPERM, …) surface through the same CLI/REST
        # install-hint path as the rest of the start-up failures, instead of
        # leaking a raw ``OSError`` through the public surface (CodeRabbit
        # Round 11 thread #16).
        try:
            self._process = subprocess.Popen(
                [str(python), str(_DAEMON_SCRIPT_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except OSError as exc:
            self._process = None
            raise OvPhysXDaemonUnavailableError(
                f"failed to spawn ovphysx daemon ({python}): {exc}"
            ) from exc

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process,),
            daemon=True,
        )
        self._stderr_thread.start()

        # ``_read_stdout_line`` raises plain ``OvPhysXDaemonError`` on
        # timeout, which loses the actionable "daemon unavailable"
        # classification CLI/REST callers branch on for bootstrap
        # failures. Catch that here so every non-success start-up
        # outcome surfaces as ``OvPhysXDaemonUnavailableError``.
        try:
            ready_line = self._read_stdout_line(self._start_timeout_s, "startup")
        except OvPhysXDaemonError as exc:
            # ``_read_stdout_line`` already kills the subprocess on
            # timeout; re-raise as the more specific class.
            raise OvPhysXDaemonUnavailableError(
                self._with_stderr_tail(str(exc))
            ) from exc
        if not ready_line:
            rc = self._process.wait(timeout=10)
            message = self._with_stderr_tail(
                f"ovphysx daemon exited during start-up (exit code {rc})"
            )
            self._process = None
            raise OvPhysXDaemonUnavailableError(message)
        try:
            msg = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            self._kill_process()
            raise OvPhysXDaemonUnavailableError(
                f"daemon ready line was not JSON: {ready_line!r} ({exc})"
            ) from exc
        if msg.get("status") == "error":
            err = str(msg.get("error", "unknown daemon start-up error"))
            self._kill_process()
            raise OvPhysXDaemonUnavailableError(self._with_stderr_tail(err))
        if msg.get("status") != "ready":
            self._kill_process()
            raise OvPhysXDaemonUnavailableError(
                f"daemon emitted unexpected start-up message: {msg!r}"
            )
        logger.info("ovphysx daemon ready (pid %d)", self._process.pid)

    def _drain_stderr(self, proc: subprocess.Popen[str]) -> None:
        if proc.stderr is None:  # pragma: no cover
            return
        for line in proc.stderr:
            stripped = line.rstrip()
            if stripped:
                with self._stderr_lock:
                    self._stderr_tail.append(stripped)
                logger.debug("[ovphysx-daemon] %s", stripped)

    def _with_stderr_tail(self, message: str) -> str:
        thread = self._stderr_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1)
        with self._stderr_lock:
            stderr_tail = "\n".join(self._stderr_tail)
        if not stderr_tail:
            return message
        stderr_tail = stderr_tail[-_STDERR_TAIL_CHARS:]
        return f"{message}\novphysx stderr (tail):\n{stderr_tail}"

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        scene_usd: Path,
        body_pattern: str,
        duration_s: float,
        dt: float = 1.0 / 240.0,
        sample_fps: int = 30,
        initial_linear_velocity: Sequence[float] | None = None,
        initial_angular_velocity: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Run one trial in the daemon. Returns the parsed JSON response.

        Response keys:

        * ``trajectory``: ``list[[t_s, pose7, vel6]]`` where
          ``pose7 = [px,py,pz,qx,qy,qz,qw]`` and
          ``vel6 = [vx,vy,vz,wx,wy,wz]``. Velocity is read directly
          from the daemon's ``RIGID_BODY_VELOCITY`` tensor binding —
          the parent does NOT need to finite-difference positions.
        * ``final_pose``: ``pose7`` of the last sample.
        * ``final_velocity``: ``vel6`` of the last sample.
        * ``n_bodies``, ``duration_s``, ``n_steps``: bookkeeping.

        Raises:
            OvPhysXDaemonUnavailableError: daemon could not start.
            OvPhysXDaemonError: daemon returned ``status=error`` or the
                pipe broke mid-call.
        """
        request: dict[str, Any] = {
            "command": "evaluate",
            "scene_usd": str(scene_usd),
            "body_pattern": body_pattern,
            "duration_s": float(duration_s),
            "dt": float(dt),
            "sample_fps": int(sample_fps),
            "initial_linear_velocity": (
                list(initial_linear_velocity)
                if initial_linear_velocity is not None
                else None
            ),
            "initial_angular_velocity": (
                list(initial_angular_velocity)
                if initial_angular_velocity is not None
                else None
            ),
        }
        return self._send_command(request, op_label="evaluate")

    def reset_only(self) -> dict[str, Any]:
        """Tear down the previous trial's state without running a new
        trial. Used by reset-correctness tests."""
        return self._send_command({"command": "reset_only"}, op_label="reset_only")

    def shutdown(self) -> None:
        """Send a graceful shutdown and reap the subprocess."""
        with self._lock:
            self._shutdown_locked()

    def _atexit_shutdown(self) -> None:
        try:
            self.shutdown()
        except Exception:  # pragma: no cover — best-effort
            pass

    def _shutdown_locked(self) -> None:
        if not self._is_running():
            self._process = None
            return
        proc = self._process
        assert proc is not None
        try:
            assert proc.stdin is not None
            proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._kill_process()
        finally:
            self._process = None
            self._stdout_buffer = b""

    # ------------------------------------------------------------------
    # Send / receive plumbing
    # ------------------------------------------------------------------

    def _send_command(
        self, request: dict[str, Any], *, op_label: str
    ) -> dict[str, Any]:
        with self._lock:
            if not self._is_running():
                logger.warning("ovphysx daemon not running — restarting")
                self._start()
            assert self._process is not None
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                rc = self._process.poll()
                # Reap the previous child before dropping the handle
                # (CodeRabbit Round 11 thread #17). If the daemon already
                # exited, ``_kill_process`` is a no-op past the
                # ``poll() is None`` check; if it's wedged but still alive,
                # ``_kill_process`` SIGKILL+wait()s it so the next call
                # doesn't start a second daemon while the first one still
                # owns GPU memory.
                self._kill_process()
                raise OvPhysXDaemonError(
                    f"ovphysx daemon pipe broke before {op_label} response "
                    f"(exit code {rc})"
                ) from exc
            response_line = self._read_stdout_line(self._evaluate_timeout_s, op_label)
            if not response_line:
                rc = self._process.poll() if self._process is not None else None
                # Reap before clearing — see comment above.
                self._kill_process()
                raise OvPhysXDaemonError(
                    f"ovphysx daemon died during {op_label} (exit code {rc})"
                )
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as exc:
                raise OvPhysXDaemonError(
                    f"ovphysx daemon emitted non-JSON response: {response_line!r}"
                ) from exc

        if response.get("status") == "error":
            raise OvPhysXDaemonError(
                f"ovphysx daemon {op_label} error: {response.get('error', 'unknown')}"
            )
        if response.get("status") != "ok":
            raise OvPhysXDaemonError(
                f"ovphysx daemon {op_label} unexpected response: {response!r}"
            )
        return response

    def _read_stdout_line(self, timeout_s: float, phase: str) -> str:
        """Read one daemon stdout line with a timeout.

        Mirrors the OvRTX-daemon implementation in
        ``render_ovrtx.py:1019`` line-for-line: ``readline()`` blocks
        unconditionally if the daemon stops writing without closing
        stdout, and partial lines make a single ``select()`` insufficient.
        Read raw bytes through ``select`` until newline or deadline.
        """
        assert self._process is not None
        assert self._process.stdout is not None

        buffered_line = self._pop_stdout_line()
        if buffered_line is not None:
            return buffered_line

        if timeout_s <= 0:
            if self._stdout_buffer:
                prefix = self._stdout_buffer.decode(errors="replace")
                self._stdout_buffer = b""
                return prefix + self._process.stdout.readline()
            return self._process.stdout.readline()

        if os.name == "nt":
            return self._read_stdout_line_threaded(timeout_s, phase)

        fd = self._process.stdout.fileno()
        deadline = time.monotonic() + timeout_s
        selector = selectors.DefaultSelector()
        try:
            selector.register(fd, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                events = selector.select(remaining)
                if not events:
                    break
                chunk = os.read(fd, 4096)
                if not chunk:
                    line = self._stdout_buffer.decode(errors="replace")
                    self._stdout_buffer = b""
                    return line
                self._stdout_buffer += chunk
                buffered_line = self._pop_stdout_line()
                if buffered_line is not None:
                    return buffered_line
        finally:
            selector.close()

        logger.error(
            "ovphysx daemon %s timed out after %.1fs; killing subprocess",
            phase,
            timeout_s,
        )
        self._kill_process()
        raise OvPhysXDaemonError(
            f"ovphysx daemon {phase} timed out after {timeout_s:.1f}s"
        )

    def _read_stdout_line_threaded(self, timeout_s: float, phase: str) -> str:
        """Read one stdout line with a timeout on Windows subprocess pipes."""
        assert self._process is not None
        assert self._process.stdout is not None

        prefix = self._stdout_buffer.decode(errors="replace")
        self._stdout_buffer = b""
        results: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)
        stdout = self._process.stdout

        def _readline() -> None:
            try:
                results.put(prefix + stdout.readline())
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                results.put(exc)

        reader = threading.Thread(target=_readline, daemon=True)
        reader.start()
        try:
            item = results.get(timeout=timeout_s)
        except queue.Empty as exc:
            logger.error(
                "ovphysx daemon %s timed out after %.1fs; killing subprocess",
                phase,
                timeout_s,
            )
            # Killing the daemon closes stdout, which unblocks the reader thread.
            self._kill_process()
            raise OvPhysXDaemonError(
                f"ovphysx daemon {phase} timed out after {timeout_s:.1f}s"
            ) from exc
        if isinstance(item, BaseException):
            raise OvPhysXDaemonError(
                f"ovphysx daemon {phase} stdout read failed: {item}"
            ) from item
        return item

    def _pop_stdout_line(self) -> str | None:
        if b"\n" not in self._stdout_buffer:
            return None
        line, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
        return (line + b"\n").decode(errors="replace")

    def _kill_process(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        except Exception:
            logger.exception("Failed to kill ovphysx daemon subprocess")
        finally:
            self._process = None
            self._stdout_buffer = b""


__all__ = [
    "_OvPhysXDaemon",
    "OvPhysXDaemonError",
    "OvPhysXDaemonUnavailableError",
    "ovphysx_runtime_available",
]
