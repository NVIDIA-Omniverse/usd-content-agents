# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Daemon-side script for ``_OvPhysXDaemon``.

This module runs inside the isolated ovphysx venv (default
``~/.cache/wu/ovphysx_venv``; ``bin/python`` on POSIX and
``Scripts/python.exe`` on Windows). It MUST NOT import ``pxr`` or
``usd-core`` — ovphysx bundles its own OpenUSD 25.11 and any other USD in
the process triggers a fatal version-mismatch error at ovphysx bootstrap.

Protocol (JSON line over stdin / stdout):

* startup: emits ``{"status": "ready", "version": "...", "device": "..."}``
* ``{"command": "evaluate", "scene_usd", "body_pattern",
   "duration_s", "dt", "sample_fps",
   "initial_linear_velocity"|null, "initial_angular_velocity"|null}`` →
  ``{"status": "ok", "trajectory": [[t_s, [px,py,pz,qx,qy,qz,qw]], ...],
     "final_pose": [...], "n_bodies", "duration_s", "n_steps"}``
* ``{"command": "reset_only"}`` → ``{"status": "ok"}`` — clears the
  previous trial's USD + bindings without running a new trial. Used by
  tests to verify the reset path in isolation.
* ``{"command": "shutdown"}`` → emits ``{"status": "ok"}`` then exits 0.

Per-trial reset (HARD REQUIREMENT — silent state leakage between trials
is the single most common simulator bug):

  1. Destroy every tensor binding from the previous trial.
  2. ``remove_usd(previous_usd_handle)`` to evict the previous scene's
     prims and invalidate solver warm-starts that referenced them.
  3. Add the new scene; create fresh bindings; write fresh initial
     velocities; step from ``current_time = 0.0``.

Determinism is the test gate: two consecutive ``evaluate`` calls on the
same (scene, params, seed) must produce byte-equivalent trajectories.
``test_ovphysx_daemon_protocol.py`` asserts this.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Stdout/stderr split: JSON goes to a saved-stdout fd; all other writes —
# including C-level carb / PhysX warnings that bypass sys.stdout — go to
# stderr, where the parent's stderr-drain thread logs them at DEBUG.
#
# Why fd-level (os.dup / os.dup2) instead of contextlib.redirect_stdout:
# ovphysx is built on Carbonite (carb), which prints via C fprintf
# straight to fd 1. Python-level sys.stdout swaps don't see those writes.
# We dup fd 1 to ``_JSON_OUT_FD`` then redirect fd 1 to fd 2 so C noise
# goes to stderr while our JSON emitter writes directly to the saved fd.
# ---------------------------------------------------------------------------
_JSON_OUT_FD = os.dup(sys.stdout.fileno())
# Make sure Python's stdout buffer is empty before the fd redirection so
# any python-side stdout writes (none expected) are flushed to the
# original fd, not the new stderr-merged one.
sys.stdout.flush()
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())


def _emit(payload: dict[str, Any]) -> None:
    """Write one JSON line to the saved stdout fd; never raises.

    ``os.write`` on a pipe can return a partial count (POSIX semantics:
    the call may return fewer bytes than requested, especially for
    payloads larger than ``PIPE_BUF`` or when interrupted by a signal).
    A trajectory frame for a long sim trajectory is well above the
    typical 4 KiB ``PIPE_BUF`` threshold, so a single ``os.write`` could
    truncate the frame and break the parent's line-oriented JSON parser.
    Loop until the entire encoded line has been written, retrying
    automatically on EINTR.
    """
    try:
        data = (json.dumps(payload) + "\n").encode("utf-8")
        view = memoryview(data)
        while view:
            try:
                written = os.write(_JSON_OUT_FD, view)
            except InterruptedError:  # signal mid-write; resume.
                continue
            if written <= 0:
                # Defensive — POSIX write should never return 0 on a
                # blocking fd unless the peer closed; bail rather than
                # spin.
                break
            view = view[written:]
    except Exception:  # pragma: no cover — final-resort sentinel
        pass


def _emit_error(message: str, **extra: Any) -> None:
    payload: dict[str, Any] = {"status": "error", "error": message}
    payload.update(extra)
    _emit(payload)


def _import_ovphysx() -> tuple[Any, Any, Any]:
    """Import ovphysx and return (module, PhysX class, TensorType enum).

    Done inside the script so a missing daemon venv produces a clean
    error on the *parent's* startup-handshake read instead of an opaque
    subprocess crash.
    """
    try:
        import ovphysx
        from ovphysx import PhysX, TensorType
    except ImportError as exc:  # pragma: no cover — caught by parent
        _emit_error(
            "ovphysx import failed; is the daemon venv populated? "
            "Recreate it from the matching checked-in PEP 751 profile at "
            "apps/physics_agent/runtime/pylock.ovphysx-runtime.toml or "
            "apps/physics_agent/runtime/pylock.ovphysx-runtime.aarch64.toml; the "
            "parent error reports the exact hash-enforced command. Detail: " + str(exc)
        )
        raise
    return ovphysx, PhysX, TensorType


def _read_command() -> dict[str, Any] | None:
    """Read one JSON line from stdin. Return None on EOF."""
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        _emit_error(f"daemon could not parse stdin JSON: {exc}", line=line[:200])
        return {}


def _new_pose_buffer(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


class _DaemonState:
    """One PhysX instance reused across many trials; per-trial state
    explicitly torn down between calls."""

    def __init__(self, ovphysx_mod: Any, PhysX_cls: Any, TensorType_cls: Any) -> None:
        self._ovphysx = ovphysx_mod
        self._TensorType = TensorType_cls
        # Pin to CPU. ovphysx tensor bindings carry a device and must
        # match the numpy/torch buffer the parent allocates on the
        # other side; numpy is CPU, so a GPU binding would fail with
        # ``device mismatch``. CPU is also the deterministic default
        # for tests and adequate for drop_settle / freeform's
        # single-rigid-body workload. GPU is a future config knob —
        # the device choice is process-global once ovphysx
        # initializes, so we cannot mix in one daemon.
        self._physx = PhysX_cls(device="cpu")
        self._previous_usd_handle: int | None = None
        self._previous_bindings: list[Any] = []

    def reset_state(self) -> None:
        """Tear down the previous trial's USD + bindings.

        Called at the start of every ``evaluate`` and from the
        ``reset_only`` command. Also called from ``shutdown``. Idempotent
        — safe to invoke when no trial has run.
        """
        for binding in self._previous_bindings:
            try:
                binding.destroy()
            except Exception:
                pass
        self._previous_bindings = []
        if self._previous_usd_handle is not None:
            try:
                self._physx.remove_usd(self._previous_usd_handle)
            except Exception:
                pass
            self._previous_usd_handle = None

    def evaluate(self, req: dict[str, Any]) -> dict[str, Any]:
        """Run one trial: reset-then-load-then-step-then-read.

        Reads BOTH ``RIGID_BODY_POSE`` and ``RIGID_BODY_VELOCITY``
        bindings on every sample so the trajectory carries raw
        simulator velocity (not finite-differenced from positions). The
        recording.usda authoring on the parent side then time-samples
        ``physics:velocity`` and ``physics:angularVelocity`` directly
        from the daemon's output.
        """
        # ------------------------------------------------------------------
        # 1. Reset previous trial's state
        # ------------------------------------------------------------------
        self.reset_state()

        # ------------------------------------------------------------------
        # 2. Load this trial's scene
        # ------------------------------------------------------------------
        scene_usd = str(req["scene_usd"])
        body_pattern = str(req["body_pattern"])
        usd_handle, _ = self._physx.add_usd(scene_usd)
        self._previous_usd_handle = usd_handle

        # ------------------------------------------------------------------
        # 3. Bind tensors for pose AND velocity (always — velocity is
        #    persisted into recording.usda regardless of whether the
        #    caller wrote initial velocities).
        # ------------------------------------------------------------------
        pose_binding = self._physx.create_tensor_binding(
            pattern=body_pattern,
            tensor_type=self._TensorType.RIGID_BODY_POSE,
        )
        # Track immediately so any subsequent error path (e.g. zero-body
        # match, vel-binding allocation failure) destroys the binding on
        # the next ``reset_state`` instead of leaking it across trials.
        self._previous_bindings.append(pose_binding)

        n_bodies = int(pose_binding.shape[0]) if pose_binding.shape else 0
        if n_bodies == 0:
            raise RuntimeError(
                f"no rigid bodies matched pattern {body_pattern!r} in "
                f"{scene_usd!r}; check that the scene authored a body "
                "with UsdPhysics.RigidBodyAPI applied at the expected "
                "prim path"
            )

        vel_binding = self._physx.create_tensor_binding(
            pattern=body_pattern,
            tensor_type=self._TensorType.RIGID_BODY_VELOCITY,
        )
        self._previous_bindings.append(vel_binding)

        # Always write the velocity tensor — even when the request
        # supplies no initial velocities — so any ``physics:velocity`` /
        # ``physics:angularVelocity`` values that an upstream USD pipeline
        # authored on the body can't silently leak in as a starting
        # condition. drop_settle in particular requires "no initial
        # velocity, fall under gravity only"; without this overwrite a
        # user-supplied physics_usd that happens to carry stale
        # velocities would invalidate the trial.
        init_lin = req.get("initial_linear_velocity")
        init_ang = req.get("initial_angular_velocity")
        vel = np.zeros(vel_binding.shape, dtype=np.float32)
        if init_lin is not None:
            vel[:, 0:3] = list(init_lin)
        if init_ang is not None:
            vel[:, 3:6] = list(init_ang)
        vel_binding.write(vel)

        # ------------------------------------------------------------------
        # 4. Plan stepping: chunk between samples so the read cadence is
        #    sample_fps without sub-stepping correctness pain.
        # ------------------------------------------------------------------
        duration_s = float(req["duration_s"])
        dt = float(req["dt"])
        sample_fps = int(req.get("sample_fps", 30))
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if sample_fps <= 0:
            raise ValueError(f"sample_fps must be > 0, got {sample_fps}")
        if duration_s <= 0:
            raise ValueError(f"duration_s must be > 0, got {duration_s}")

        sample_dt = 1.0 / float(sample_fps)
        steps_per_sample = max(1, int(round(sample_dt / dt)))
        total_steps = max(1, int(round(duration_s / dt)))

        pose_buf = _new_pose_buffer(pose_binding.shape)
        vel_buf = np.zeros(vel_binding.shape, dtype=np.float32)

        # Sample 0: initial pose + velocity (BEFORE any step). The
        # initial velocity readout reflects whatever the caller wrote
        # via ``vel_binding.write`` above (or zeros when no initial
        # was supplied).
        pose_binding.read(pose_buf)
        vel_binding.read(vel_buf)
        trajectory: list[list[Any]] = [[0.0, pose_buf[0].tolist(), vel_buf[0].tolist()]]

        current_time = 0.0
        steps_done = 0
        while steps_done < total_steps:
            chunk = min(steps_per_sample, total_steps - steps_done)
            self._physx.step_n_sync(chunk, dt, current_time)
            steps_done += chunk
            current_time = steps_done * dt
            pose_binding.read(pose_buf)
            vel_binding.read(vel_buf)
            trajectory.append(
                [float(current_time), pose_buf[0].tolist(), vel_buf[0].tolist()]
            )

        return {
            "status": "ok",
            "trajectory": trajectory,
            "final_pose": pose_buf[0].tolist(),
            "final_velocity": vel_buf[0].tolist(),
            "n_bodies": n_bodies,
            "duration_s": duration_s,
            "n_steps": int(steps_done),
        }

    def shutdown(self) -> None:
        self.reset_state()
        try:
            self._physx.release()
        except Exception:
            pass


def _build_state() -> _DaemonState | None:
    try:
        ovphysx, PhysX, TensorType = _import_ovphysx()
    except ImportError:
        return None
    try:
        return _DaemonState(ovphysx, PhysX, TensorType)
    except Exception as exc:
        _emit_error(
            f"ovphysx initialization failed: {exc}",
            traceback=traceback.format_exc(),
        )
        return None


def main() -> int:
    state = _build_state()
    if state is None:
        return 1

    _emit({"status": "ready", "device": "auto"})

    while True:
        cmd = _read_command()
        if cmd is None:
            # EOF on stdin — caller closed the pipe; exit cleanly.
            break
        if not cmd:
            continue
        command = cmd.get("command")
        try:
            if command == "evaluate":
                _emit(state.evaluate(cmd))
            elif command == "reset_only":
                state.reset_state()
                _emit({"status": "ok"})
            elif command == "shutdown":
                state.shutdown()
                _emit({"status": "ok"})
                break
            else:
                _emit_error(f"unknown command: {command!r}")
        except Exception as exc:
            _emit_error(str(exc), traceback=traceback.format_exc())
            # Non-fatal — keep serving subsequent commands. The parent
            # decides whether to abandon the daemon based on the
            # error_type.
    return 0


if __name__ == "__main__":
    sys.exit(main())
