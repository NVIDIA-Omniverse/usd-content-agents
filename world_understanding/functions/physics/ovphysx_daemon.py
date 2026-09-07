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
import errno
import hashlib
import json
import logging
import math
import os
import platform
import queue
import selectors
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from world_understanding.utils.file_locking import (
    blocking_exclusive_descriptor_lock,
)

from .ovphysx_process_limits import OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV

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
_OVPHYSX_RUNTIME_LOCK_WINDOWS = Path(
    "apps/physics_agent/runtime/pylock.ovphysx-runtime-windows.toml"
)
# Shared with usd_cli's preflight so both supported callers can safely use one
# explicitly configured runtime venv.
_OVPHYSX_RUNTIME_READY_MARKER = ".usd-cli-ovphysx-ready"
_OVPHYSX_RUNTIME_READY_MARKER_SCHEMA_VERSION = "usd-cli.ovphysx-runtime-ready.v2"
_OVPHYSX_RUNTIME_LOCK_ENV = "WU_OVPHYSX_RUNTIME_LOCK"
_OVPHYSX_AUTO_PROVISION_ENV = "WU_OVPHYSX_AUTO_PROVISION"
_OVPHYSX_PROVISION_TIMEOUT_ENV = "WU_OVPHYSX_PROVISION_TIMEOUT"
_OVPHYSX_PROVISION_TIMEOUT_S = 900.0
_OVPHYSX_RUNTIME_PROBE_TIMEOUT_S = 60.0
_OVPHYSX_RUNTIME_PROBE = (
    "from ovphysx import PhysX; physics = PhysX(device='cpu'); physics.release()"
)
_STDERR_TAIL_LINES = 40
_STDERR_TAIL_CHARS = 4000


@dataclass(frozen=True)
class OvPhysXRuntimeSpec:
    """Resolved paths for the isolated, reviewed OvPhysX runtime."""

    venv_dir: Path
    python_path: Path
    lock_path: Path
    ready_marker_path: Path


def _ovphysx_runtime_lock(
    machine: str | None = None,
    system: str | None = None,
) -> Path:
    """Return the reviewed daemon lock for the current OS and architecture."""
    machine = (machine or platform.machine()).lower()
    system = (system or platform.system()).lower()
    if system == "windows":
        if machine in {"x86_64", "amd64"}:
            return _OVPHYSX_RUNTIME_LOCK_WINDOWS
        raise ValueError(
            f"Unsupported architecture for Windows OvPhysX runtime: {machine}"
        )
    if system != "linux":
        raise ValueError(f"Unsupported operating system for OvPhysX runtime: {system}")
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
    spec = resolve_ovphysx_runtime_spec(_source_checkout_root(), venv_dir=venv_dir)
    return _ovphysx_runtime_ready(spec)


def _ovphysx_runtime_ready(spec: OvPhysXRuntimeSpec) -> bool:
    """Return whether the lock-bound interpreter still passes its runtime probe."""

    if not spec.python_path.is_file():
        return False
    try:
        marker = json.loads(spec.ready_marker_path.read_text(encoding="utf-8"))
        lock_sha256 = hashlib.sha256(spec.lock_path.read_bytes()).hexdigest()
    except (OSError, json.JSONDecodeError):
        return False
    marker_valid = bool(
        isinstance(marker, dict)
        and marker.get("schema_version") == _OVPHYSX_RUNTIME_READY_MARKER_SCHEMA_VERSION
        and marker.get("runtime_lock_sha256") == lock_sha256
        and marker.get("python_path") == str(spec.python_path)
    )
    return marker_valid and _ovphysx_runtime_probe(spec.python_path)


def _ovphysx_runtime_probe(python_path: Path) -> bool:
    """Return whether the isolated interpreter can initialize OvPhysX."""

    environment = _ovphysx_provision_environment()
    try:
        subprocess.run(
            [str(python_path), "-c", _OVPHYSX_RUNTIME_PROBE],
            check=True,
            env=environment,
            timeout=_OVPHYSX_RUNTIME_PROBE_TIMEOUT_S,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def resolve_ovphysx_runtime_spec(
    repo_root: Path | str,
    *,
    venv_dir: Path | str | None = None,
) -> OvPhysXRuntimeSpec:
    """Resolve the exact runtime paths used by setup and readiness checks."""

    resolved_repo = (
        (repo_root if isinstance(repo_root, Path) else Path(repo_root))
        .expanduser()
        .resolve()
    )
    resolved_venv = (
        (
            venv_dir
            if isinstance(venv_dir, Path)
            else Path(
                venv_dir
                if venv_dir is not None
                else os.environ.get("WU_OVPHYSX_VENV_DIR", str(_DEFAULT_VENV_DIR))
            )
        )
        .expanduser()
        .resolve()
    )
    configured_lock = os.environ.get(_OVPHYSX_RUNTIME_LOCK_ENV)
    if configured_lock:
        lock_path = Path(configured_lock).expanduser().resolve()
    else:
        reviewed_lock = _ovphysx_runtime_lock()
        if platform.system().lower() == "linux":
            # The Linux wheel and source tree both own the reviewed copy beside
            # this module. Keep it authoritative even when it is absent.
            lock_path = (_THIS_DIR / reviewed_lock.name).resolve()
        else:
            # Native-Windows fixed-pipeline execution is not a packaged-wheel
            # surface. Preserve the pre-existing checkout-only lock lookup.
            lock_path = (resolved_repo / reviewed_lock).resolve()
    return OvPhysXRuntimeSpec(
        venv_dir=resolved_venv,
        python_path=_ovphysx_venv_python_path(resolved_venv),
        lock_path=lock_path,
        ready_marker_path=resolved_venv / _OVPHYSX_RUNTIME_READY_MARKER,
    )


def ovphysx_runtime_install_commands(
    repo_root: Path | str,
    *,
    venv_dir: Path | str | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Return copy/pasteable commands for the reviewed isolated runtime."""

    spec = resolve_ovphysx_runtime_spec(repo_root, venv_dir=venv_dir)
    create = [
        "uv",
        "venv",
        "--python",
        "3.12",
        "--allow-existing",
        str(spec.venv_dir),
    ]
    return (
        tuple(create),
        (
            "uv",
            "pip",
            "install",
            "--python",
            str(spec.python_path),
            "--require-hashes",
            "--no-deps",
            "-r",
            str(spec.lock_path),
            "--no-config",
            "--no-sources",
        ),
        (
            str(spec.python_path),
            "-c",
            "from ovphysx import PhysX; physics = PhysX(device='cpu'); physics.release()",
        ),
    )


def _ovphysx_auto_provision_enabled() -> bool:
    """Return whether first-use daemon runtime setup is permitted."""

    # The fixed Physics Agent pipeline supports Windows through WSL2, which
    # reports Linux here. Native-Windows provisioning remains out of scope
    # until the platform plan required by the repository root is approved.
    return (
        platform.system().lower() == "linux"
        and os.environ.get(_OVPHYSX_AUTO_PROVISION_ENV, "1") == "1"
    )


def _source_checkout_root() -> Path:
    """Return the checkout root that owns the reviewed runtime lock."""

    return _THIS_DIR.parents[2]


def _ovphysx_provision_timeout_s() -> float:
    """Return the bounded timeout for one setup command."""

    value = float(
        os.environ.get(_OVPHYSX_PROVISION_TIMEOUT_ENV, _OVPHYSX_PROVISION_TIMEOUT_S)
    )
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{_OVPHYSX_PROVISION_TIMEOUT_ENV} must be finite and positive"
        )
    return value


def _ovphysx_provision_lock_path(venv_dir: Path) -> Path:
    """Return the cross-process lock that owns setup of ``venv_dir``."""

    return venv_dir.parent / f"{venv_dir.name}.provision.lock"


@contextmanager
def _ovphysx_provision_lock(venv_dir: Path) -> Iterator[None]:
    """Hold usd-cli's persistent lock inode without unlinking it on release."""

    lock_path = _ovphysx_provision_lock_path(venv_dir.expanduser().resolve())
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        if os.name == "nt" and opened.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
        with blocking_exclusive_descriptor_lock(descriptor):
            yield
    finally:
        os.close(descriptor)


def _ovphysx_provision_environment() -> dict[str, str]:
    """Return an environment that cannot leak parent OpenUSD imports."""

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return environment


def _provision_ovphysx_runtime_lock_held(spec: OvPhysXRuntimeSpec) -> Path:
    """Create, install, and probe the runtime while its shared lock is held.

    The marker is deliberately written only after the import probe succeeds, so
    a failed or interrupted setup is retried on the next daemon start.
    """

    active_prefix = Path(sys.prefix).expanduser().resolve()
    active_python = Path(sys.executable).expanduser().resolve()
    if spec.venv_dir == active_prefix or spec.python_path == active_python:
        raise OvPhysXDaemonUnavailableError(
            "Refusing to provision OvPhysX into the active parent environment; "
            "choose a separate WU_OVPHYSX_VENV_DIR."
        )

    uv = shutil.which("uv")
    if uv is None:
        raise OvPhysXDaemonUnavailableError(
            "ovphysx daemon auto-provisioning requires the uv executable. "
            + _ovphysx_unavailable_message(spec.venv_dir)
        )
    if not spec.lock_path.is_file():
        raise OvPhysXDaemonUnavailableError(
            f"OvPhysX runtime lock is missing: {spec.lock_path}. "
            + _ovphysx_unavailable_message(spec.venv_dir)
        )

    try:
        environment = _ovphysx_provision_environment()
        timeout_s = _ovphysx_provision_timeout_s()
        # A concurrent session may have completed provisioning while this one
        # waited for the lock. This return also preserves its current marker:
        # marker removal below is reachable only for an already-invalid runtime.
        if _ovphysx_runtime_ready(spec):
            return spec.python_path
        # Do not let another daemon accept a previously valid marker while this
        # process replaces the runtime in place.
        spec.ready_marker_path.unlink(missing_ok=True)
        for command in ovphysx_runtime_install_commands(
            _source_checkout_root(), venv_dir=spec.venv_dir
        )[:2]:
            subprocess.run(
                (uv, *command[1:]),
                check=True,
                env=environment,
                timeout=timeout_s,
            )
        subprocess.run(
            [
                str(spec.python_path),
                "-c",
                _OVPHYSX_RUNTIME_PROBE,
            ],
            check=True,
            env=environment,
            timeout=timeout_s,
        )
        if not spec.python_path.is_file():
            raise OvPhysXDaemonUnavailableError(
                f"OvPhysX provisioning did not create Python at {spec.python_path}. "
                + _ovphysx_unavailable_message(spec.venv_dir)
            )
        spec.ready_marker_path.write_text(
            json.dumps(
                {
                    "schema_version": _OVPHYSX_RUNTIME_READY_MARKER_SCHEMA_VERSION,
                    "runtime_lock_sha256": hashlib.sha256(
                        spec.lock_path.read_bytes()
                    ).hexdigest(),
                    "python_path": str(spec.python_path),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return spec.python_path
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise OvPhysXDaemonUnavailableError(
            f"Failed to auto-provision OvPhysX runtime at {spec.venv_dir}: {exc}. "
            + _ovphysx_unavailable_message(spec.venv_dir)
        ) from exc


def _provision_ovphysx_runtime(venv_dir: Path) -> Path:
    """Provision a runtime while acquiring its shared cross-tool lock."""

    spec = resolve_ovphysx_runtime_spec(_source_checkout_root(), venv_dir=venv_dir)
    try:
        with _ovphysx_provision_lock(spec.venv_dir):
            return _provision_ovphysx_runtime_lock_held(spec)
    except OSError as exc:
        raise OvPhysXDaemonUnavailableError(
            f"Unable to lock OvPhysX runtime at {spec.venv_dir}: {exc}. "
            + _ovphysx_unavailable_message(spec.venv_dir)
        ) from exc


def _ovphysx_unavailable_message(venv_dir: Path = _DEFAULT_VENV_DIR) -> str:
    python_path = _ovphysx_venv_python_path(venv_dir)
    runtime_lock = resolve_ovphysx_runtime_spec(
        _source_checkout_root(), venv_dir=venv_dir
    ).lock_path
    return (
        "ovphysx daemon is not available. If the daemon venv at "
        f"{venv_dir} does not exist or does not have ovphysx installed, "
        "bootstrap the exact reviewed runtime with:\n"
        f"  uv venv --python 3.12 --allow-existing {venv_dir}\n"
        f"  uv pip install --python {python_path} --require-hashes --no-deps "
        f"-r {runtime_lock} --no-config --no-sources\n"
        f'  env -u PYTHONPATH {python_path} -c "from ovphysx import PhysX; '
        "physics = PhysX(device='cpu'); physics.release()\"\n"
        "  # Then run the repository's OvPhysX preflight to record the lock-bound "
        f"readiness marker at {venv_dir / _OVPHYSX_RUNTIME_READY_MARKER}.\n"
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

    @classmethod
    def safe_remediation_message(cls) -> str:
        """Return the reviewed product-authored recovery instructions."""

        return _ovphysx_unavailable_message()


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
    * ``relax_address_space_limit``: allow only the daemon process to raise an
      inherited soft RLIMIT_AS to its inherited hard ceiling before loading
      NumPy or CUDA-facing libraries. Default ``False``.
    * Timeouts via env: ``WU_OVPHYSX_DAEMON_START_TIMEOUT`` (seconds,
      default 300), ``WU_OVPHYSX_DAEMON_EVALUATE_TIMEOUT`` (default
      1800).
    """

    def __init__(
        self,
        *,
        venv_dir: Path | None = None,
        device: str = "auto",
        relax_address_space_limit: bool = False,
    ) -> None:
        self._venv_dir = Path(venv_dir) if venv_dir is not None else _DEFAULT_VENV_DIR
        self._device = device
        self._relax_address_space_limit = relax_address_space_limit
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
        """Return the daemon Python, provisioning its missing runtime by default."""
        spec = resolve_ovphysx_runtime_spec(
            _source_checkout_root(), venv_dir=self._venv_dir
        )
        # A baked runtime may be intentionally read-only to the daemon user.
        # Its digest-bound marker is sufficient for reuse; only mutation needs
        # the sibling provisioning lock.
        if _ovphysx_runtime_ready(spec):
            return spec.python_path
        if _ovphysx_auto_provision_enabled():
            return _provision_ovphysx_runtime(self._venv_dir)
        try:
            with _ovphysx_provision_lock(spec.venv_dir):
                return self._resolve_python_lock_held(spec)
        except (OSError, ValueError) as exc:
            raise OvPhysXDaemonUnavailableError(
                f"Unable to verify OvPhysX runtime at {self._venv_dir}: {exc}. "
                + _ovphysx_unavailable_message(self._venv_dir)
            ) from exc

    def _resolve_python_lock_held(self, spec: OvPhysXRuntimeSpec) -> Path:
        """Resolve or provision Python without reacquiring the shared lock."""

        if _ovphysx_runtime_ready(spec):
            return spec.python_path
        if _ovphysx_auto_provision_enabled():
            return _provision_ovphysx_runtime_lock_held(spec)
        raise OvPhysXDaemonUnavailableError(
            _ovphysx_unavailable_message(self._venv_dir)
        )

    @staticmethod
    def _read_only_baked_runtime_python(spec: OvPhysXRuntimeSpec) -> Path | None:
        """Bind a verified baked interpreter when its runtime cannot be mutated."""

        if os.access(spec.venv_dir, os.W_OK) or not _ovphysx_runtime_ready(spec):
            return None
        return spec.python_path

    def _start(self) -> None:
        """Launch the daemon subprocess and wait for the ``ready`` line."""
        env = os.environ.copy()
        # Strip PYTHONPATH so the parent's ``usd-core`` cannot leak into
        # the daemon process. ovphysx's own bundled USD must win.
        env.pop("PYTHONPATH", None)
        # Honor the configured device on first start.
        env["WU_OVPHYSX_DEVICE"] = self._device
        # Never honor an ambient marker. Only an explicit constructor opt-in
        # grants this one daemon launch permission to relax its soft limit.
        env.pop(OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV, None)
        if self._relax_address_space_limit:
            env[OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV] = "1"

        if not _DAEMON_SCRIPT_PATH.exists():
            raise OvPhysXDaemonUnavailableError(
                f"daemon script missing at {_DAEMON_SCRIPT_PATH}"
            )

        self._stdout_buffer = b""
        with self._stderr_lock:
            self._stderr_tail.clear()
        # Wrap ``Popen`` in the daemon-unavailable error class so spawn-time
        # failures (interpreter missing despite the venv check, fork EAGAIN,
        # missing libs, EPERM, …) surface through the same CLI/REST
        # install-hint path as the rest of the start-up failures, instead of
        # leaking a raw ``OSError`` through the public surface (CodeRabbit
        # Round 11 thread #16).
        spec = resolve_ovphysx_runtime_spec(
            _source_checkout_root(), venv_dir=self._venv_dir
        )
        python: Path | None = None
        spawn_lock = _ovphysx_provision_lock(spec.venv_dir)
        spawn_lock_held = False
        try:
            try:
                spawn_lock.__enter__()
            except (OSError, ValueError) as exc:
                if isinstance(exc, PermissionError) and exc.errno == errno.EACCES:
                    python = self._read_only_baked_runtime_python(spec)
                if python is None:
                    raise OvPhysXDaemonUnavailableError(
                        f"Unable to lock OvPhysX runtime at {self._venv_dir}: {exc}. "
                        + _ovphysx_unavailable_message(self._venv_dir)
                    ) from exc
            else:
                spawn_lock_held = True
                # Validate while holding the mutation lock, then retain it
                # through spawn so a shared runtime cannot change mid-start.
                python = self._resolve_python_lock_held(spec)
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

            # Keep the mutation lock through the child's import and readiness
            # handshake. Popen alone does not prove the shared native runtime
            # has finished loading, so releasing here would let another tool
            # replace its packages while this process is still importing them.
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
        finally:
            if spawn_lock_held:
                spawn_lock.__exit__(None, None, None)
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
    "OvPhysXRuntimeSpec",
    "ovphysx_runtime_install_commands",
    "ovphysx_runtime_available",
    "resolve_ovphysx_runtime_spec",
]
