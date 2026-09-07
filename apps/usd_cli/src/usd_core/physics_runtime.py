# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-level OVRTX physics simulation → trajectory → recording → metrics.

Ports world-understanding's runtime-validation pipeline (physics_ops.validate_runtime +
physics_agent scenario/recording + the ovphysx daemon) onto usd-cli.

The simulation is the real NVIDIA **ovphysx** solver in an isolated venv subprocess,
mirroring the ovrtx daemon pattern (reader thread, timeouts, file-locked provisioning). It
runs locally on a supported Linux or Windows host; other hosts use the configured remote
OVRTX service's `POST /physics/simulate` endpoint. With neither backend available, the command
raises rather than fabricating a trajectory — a physics *validation* must reflect a real
solve, never a synthetic stand-in.

The output `recording.usda` carries time-sampled translate/orient on the body, so it plays
back through `render-frames --scene recording.usda` for behavior review.
"""

from __future__ import annotations

import hashlib
import errno
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

DEFAULT_VENV = Path(
    os.environ.get("WU_OVPHYSX_VENV_DIR", str(Path.home() / ".cache" / "usd-cli" / "ovphysx_venv"))
).expanduser()
# ovphysx provisioning is reproducible or it does not happen: the runtime venv
# installs only from a hash-pinned pylock via ``uv`` (never bare ``pip`` and
# never an unpinned index resolution). WU_OVPHYSX_RUNTIME_LOCK overrides the
# lock explicitly; otherwise the containing world-understanding checkout's
# canonical physics-agent runtime lock is used.
_OVPHYSX_RUNTIME_LOCK_ENV = "WU_OVPHYSX_RUNTIME_LOCK"
_OVPHYSX_READY_MARKER = ".usd-cli-ovphysx-ready"
_OVPHYSX_READY_MARKER_SCHEMA_VERSION = "usd-cli.ovphysx-runtime-ready.v2"
_OVPHYSX_RUNTIME_PROBE = (
    "from ovphysx import PhysX; physics = PhysX(device='cpu'); physics.release()"
)
_OVPHYSX_RUNTIME_PROBE_TIMEOUT_S = 60.0
REMOTE_PHYSICS_FEATURE = "physics-simulate"
_UV_EXECUTABLE_ENV = "USD_CLI_UV_EXECUTABLE"
_WU_RUNTIME_LOCK_RELATIVE = {
    ("linux", "aarch64"): Path(
        "apps/physics_agent/runtime/pylock.ovphysx-runtime.aarch64.toml"
    ),
    ("linux", "arm64"): Path(
        "apps/physics_agent/runtime/pylock.ovphysx-runtime.aarch64.toml"
    ),
    ("windows", "amd64"): Path(
        "apps/physics_agent/runtime/pylock.ovphysx-runtime-windows.toml"
    ),
    ("windows", "x86_64"): Path(
        "apps/physics_agent/runtime/pylock.ovphysx-runtime-windows.toml"
    ),
}
_WU_RUNTIME_LOCK_DEFAULT = Path("apps/physics_agent/runtime/pylock.ovphysx-runtime.toml")
_PACKAGED_RUNTIME_LOCKS = {
    ("linux", "amd64", (3, 11)): Path(__file__).with_name(
        "pylock.ovphysx-runtime.py311.toml"
    ),
    ("linux", "x86_64", (3, 11)): Path(__file__).with_name(
        "pylock.ovphysx-runtime.py311.toml"
    ),
    ("linux", "aarch64", (3, 11)): Path(__file__).with_name(
        "pylock.ovphysx-runtime.py311.aarch64.toml"
    ),
    ("linux", "arm64", (3, 11)): Path(__file__).with_name(
        "pylock.ovphysx-runtime.py311.aarch64.toml"
    ),
    ("linux", "amd64", (3, 12)): Path(__file__).with_name(
        "pylock.ovphysx-runtime.toml"
    ),
    ("linux", "x86_64", (3, 12)): Path(__file__).with_name(
        "pylock.ovphysx-runtime.toml"
    ),
    ("linux", "aarch64", (3, 12)): Path(__file__).with_name(
        "pylock.ovphysx-runtime.aarch64.toml"
    ),
    ("linux", "arm64", (3, 12)): Path(__file__).with_name(
        "pylock.ovphysx-runtime.aarch64.toml"
    ),
    ("windows", "amd64", (3, 12)): Path(__file__).with_name(
        "pylock.ovphysx-runtime-windows.toml"
    ),
    ("windows", "x86_64", (3, 12)): Path(__file__).with_name(
        "pylock.ovphysx-runtime-windows.toml"
    ),
}
_START_TIMEOUT_S = float(os.environ.get("OVPHYSX_DAEMON_START_TIMEOUT", "300"))
_EVAL_TIMEOUT_S = float(os.environ.get("OVPHYSX_DAEMON_EVALUATE_TIMEOUT", "1800"))
#: Ceiling on `duration_s / dt`, the step count the solver schedules
#: (`total = max(1, int(round(dur/dt)))` in the daemon script). `dt` alone is only
#: bounded by > 0, so `--duration 60 --dt 1e-6` would queue 60,000,000 steps and
#: hang the command until the long evaluate timeout. Same knob and default as the
#: service model's guard (apps/ovrtx_rendering_api/service/models.py) so the local
#: path and the remote service reject the same requests. The default admits a full
#: 60 s run at the default 1/240 s timestep (14,400 steps) with room to spare.
MAX_PHYSICS_STEPS = int(os.environ.get("OVRTX_MAX_PHYSICS_STEPS", str(120_000)))
_OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV = (
    "_USD_CLI_OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE"
)


def _checked_total_steps(duration_s: float, dt: float) -> int:
    """The step total `duration_s / dt` schedules; raises before the solver runs
    when it exceeds MAX_PHYSICS_STEPS."""
    steps = max(1, int(round(float(duration_s) / float(dt))))
    if steps > MAX_PHYSICS_STEPS:
        raise ValueError(
            f"duration_s / dt schedules {steps} simulation steps, over the "
            f"limit of {MAX_PHYSICS_STEPS}; raise dt or shorten duration_s")
    return steps


# ── trajectory recording ──────────────────────────────────────────────────────────
def author_trajectory_jsonl(trajectory, out_path: str) -> str:
    with open(out_path, "w") as fh:
        for i, (t, pose, vel) in enumerate(trajectory):
            fh.write(json.dumps({"frame": i, "t": float(t),
                                 "pose": [float(v) for v in pose],
                                 "vel": [float(v) for v in vel]}) + "\n")
    return out_path


def author_trajectory_usda(scene_usd: str, trajectory, body_path: str, out_path: str,
                           fps: float = 30.0) -> str:
    """Write a recording: time-sampled translate/orient on the body, fps as timeCodesPerSecond."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils
    stage = Usd.Stage.Open(str(scene_usd))
    stage.SetTimeCodesPerSecond(float(fps)); stage.SetFramesPerSecond(float(fps))
    body = stage.GetPrimAtPath(Sdf.Path(body_path))
    xf = UsdGeom.Xformable(body)
    # Replace only the POSE ops: the simulator's pose is a full world-space
    # translate+orient, so pre-existing pose ops (translate/rotate/orient/matrix)
    # would stack on top of it — but ops like `xformOp:scale` carry intrinsic body
    # geometry the simulator doesn't model, and clearing the whole op order
    # (the old behavior) played the recording back at unscaled dimensions.
    # Mirrors world-understanding utils/usd/time_samples.py: a pose_op_types set,
    # reused translate/orient ops, preserved tail ops, standard T·R·S order.
    XformOp = UsdGeom.XformOp
    pose_op_types = {t for t in (
        XformOp.TypeTranslate, XformOp.TypeOrient, XformOp.TypeTransform,
        # per-axis translate ops (newer USD) would double the time-sampled offset
        getattr(XformOp, "TypeTranslateX", None), getattr(XformOp, "TypeTranslateY", None),
        getattr(XformOp, "TypeTranslateZ", None),
        XformOp.TypeRotateX, XformOp.TypeRotateY, XformOp.TypeRotateZ,
        XformOp.TypeRotateXYZ, XformOp.TypeRotateXZY, XformOp.TypeRotateYXZ,
        XformOp.TypeRotateYZX, XformOp.TypeRotateZXY, XformOp.TypeRotateZYX,
    ) if t is not None}
    t_op = o_op = None
    preserved = []
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate" and t_op is None:
            t_op = op  # reuse — suffixed variants (e.g. a pivot pair) stay pose ops
        elif op.GetOpName() == "xformOp:orient" and o_op is None:
            o_op = op
        elif op.GetOpType() not in pose_op_types:
            preserved.append(op)
    if t_op is None:
        t_op = xf.AddTranslateOp()
    if o_op is None:
        o_op = xf.AddOrientOp()  # Quatf
    xf.SetXformOpOrder([t_op, o_op, *preserved])
    # A reused op may carry stale samples (re-recording over a recording) that
    # would interleave with the new frames and corrupt playback.
    for attr in (t_op.GetAttr(), o_op.GetAttr()):
        if attr.GetTimeSamples():
            attr.Clear()
    for i, (_t, pose, _vel) in enumerate(trajectory):
        px, py, pz, qx, qy, qz, qw = [float(v) for v in pose]
        tc = Usd.TimeCode(float(i))
        t_op.Set(Gf.Vec3d(px, py, pz), time=tc)
        o_op.Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)), time=tc)  # USD: real first
    stage.SetStartTimeCode(0.0); stage.SetEndTimeCode(float(max(0, len(trajectory) - 1)))
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output_path))

    # Export preserves authored asset paths verbatim. If the recording is in a
    # different directory from the simulated scene, those relative paths would
    # resolve from the wrong anchor and the visual-review renderer would lose its
    # textures or payloads. Rebase only dependencies that resolved at the source.
    source_anchor = Path(scene_usd).expanduser().resolve(strict=True).parent
    output_anchor = output_path.expanduser().resolve(strict=True).parent
    if source_anchor != output_anchor:
        recording_layer = Sdf.Layer.FindOrOpen(str(output_path))
        if recording_layer is None:
            raise RuntimeError(f"Unable to reopen physics recording: {output_path}")
        rebased_count = 0

        def rebase_asset_path(asset_path: str) -> str:
            nonlocal rebased_count
            if not asset_path or "://" in asset_path or os.path.isabs(asset_path):
                return asset_path
            resolved = (source_anchor / asset_path).resolve()
            if not resolved.exists():
                return asset_path
            rebased_count += 1
            return os.path.relpath(resolved, output_anchor).replace(os.sep, "/")

        UsdUtils.ModifyAssetPaths(recording_layer, rebase_asset_path)
        if rebased_count:
            recording_layer.Save()
    return out_path


# ── metrics ────────────────────────────────────────────────────────────────────────
def trajectory_metrics(trajectory, rest_position, world_up) -> dict:
    import numpy as np
    if not trajectory:
        return {"n_samples": 0}
    times = np.array([t for t, _, _ in trajectory])
    poses = np.array([p[:3] for _, p, _ in trajectory], dtype=float)
    vels = np.array([v for _, _, v in trajectory], dtype=float)
    lin = np.linalg.norm(vels[:, 0:3], axis=1) if vels.shape[1] >= 3 else np.zeros(len(vels))
    ang = np.linalg.norm(vels[:, 3:6], axis=1) if vels.shape[1] >= 6 else np.zeros(len(vels))
    settle = float(np.linalg.norm(poses[-1] - np.asarray(rest_position[:3])))
    # Bounce metrics along world-up: drop distance to first impact (up-position minimum),
    # then the rebound peak after it. Surfaced so behavioral criteria like "first rebound
    # >= 10% of drop height" are checkable from the report without the tool hard-coding a
    # task-specific threshold into `ok`.
    up = np.asarray(world_up[:3], dtype=float)
    nrm = np.linalg.norm(up)
    up = up / nrm if nrm else np.array([0.0, 0.0, 1.0])
    h = poses @ up
    impact_i = int(np.argmin(h))
    drop_height = float(h[0] - h[impact_i])
    rebound_height = 0.0
    if impact_i < len(h) - 1:
        rebound_height = float(max(0.0, h[impact_i:].max() - h[impact_i]))
    rebound_fraction = round(rebound_height / drop_height, 4) if drop_height > 1e-6 else 0.0
    return {"n_samples": len(trajectory),
            "duration_s": float(times[-1]),
            "final_position": poses[-1].tolist(),
            "max_linear_speed": float(lin.max()),
            "final_linear_speed": float(lin[-1]),
            "min_linear_speed": float(lin.min()),
            "max_angular_speed": float(ang.max()),
            "settle_distance": round(settle, 4),
            "drop_height": round(drop_height, 4),
            "first_rebound_height": round(rebound_height, 4),
            "rebound_fraction": rebound_fraction,
            "max_abs_position": float(np.abs(poses).max())}


# ── platform gate ───────────────────────────────────────────────────────────────────
def ovphysx_platform_supported() -> bool:
    """Return whether this host has a reviewed local OvPhysX runtime.

    Import-light (no pxr/ovphysx import). Windows uses the upstream x86_64 wheel
    and the runtime smoke test is authoritative. Linux requires a native NVIDIA
    device node or WSL2's ``/dev/dxg`` bridge. This is only a platform-routing
    gate; the locked OvPhysX runtime probe remains authoritative for readiness.
    """
    import glob
    import platform

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        return machine in {"amd64", "x86_64"} and sys.version_info[:2] == (3, 12)
    if system != "Linux":
        return False
    return (
        Path("/proc/driver/nvidia/version").exists()
        or Path("/dev/dxg").exists()
        or bool(glob.glob("/dev/nvidia[0-9]*"))
    )


def resolve_remote_physics_backend(
    render_config: dict,
    *,
    verify: bool = True,
) -> dict | None:
    """Resolve a configured remote that explicitly supports physics transport.

    Backend pools are authoritative over the legacy single ``remote_url`` field,
    matching remote rendering. Verification rejects a stale render-only service
    before launching a child workflow.
    """

    from usd_core.config import resolve_render_backends

    backends = resolve_render_backends(render_config)
    if not backends:
        return None
    timeout = float(render_config.get("remote_timeout", 300))
    verify_version = str(
        render_config.get("remote_verify_version", "true")
    ).lower() not in ("false", "0", "no")
    errors: list[str] = []
    for backend in backends:
        url = str(backend.get("url") or "").rstrip("/")
        if not url:
            continue
        if verify and verify_version:
            from usd_core.remote_protocol import check_remote_protocol

            try:
                check_remote_protocol(
                    url,
                    timeout=min(timeout, 10.0),
                    required_engine="ovrtx",
                    required_features=(REMOTE_PHYSICS_FEATURE,),
                    api_key=str(backend.get("api_key") or "") or None,
                )
            except RuntimeError:
                try:
                    parsed = urlsplit(url)
                    hostname = parsed.hostname or ""
                    port_value = parsed.port
                except ValueError:
                    diagnostic_url = "<redacted-invalid-url>"
                else:
                    if ":" in hostname and not hostname.startswith("["):
                        hostname = f"[{hostname}]"
                    port = f":{port_value}" if port_value is not None else ""
                    diagnostic_url = urlunsplit(
                        (parsed.scheme, f"{hostname}{port}", "", "", "")
                    )
                # Protocol-client exceptions may echo a normalized, quoted, or
                # partially transformed credential-bearing URL.  Exact string
                # replacement cannot prove that every secret spelling was
                # removed, so expose only the sanitized origin and a fixed
                # failure class.
                errors.append(
                    f"{diagnostic_url}: physics protocol readiness check failed"
                )
                continue
        return {
            "base_url": url,
            "api_key": str(backend.get("api_key") or "") or None,
            "timeout": timeout,
            "verify_version": verify_version,
        }
    if errors:
        raise RuntimeError(
            "no configured remote OVRTX backend passed physics readiness: "
            + "; ".join(errors)
        )
    return None


# ── ovphysx isolated daemon ─────────────────────────────────────────────────────────
def _ovphysx_runtime_python_minor(venv_dir: Path | None) -> tuple[int, int]:
    """Return an existing daemon venv's Python minor, else the parent minor."""

    if venv_dir is not None:
        config = venv_dir.expanduser().resolve() / "pyvenv.cfg"
        try:
            lines = config.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            key, separator, value = line.partition("=")
            if separator and key.strip().lower() in {"version", "version_info"}:
                pieces = value.strip().split(".", maxsplit=2)
                try:
                    return int(pieces[0]), int(pieces[1])
                except (IndexError, ValueError):
                    continue
    return sys.version_info[:2]


def _ovphysx_runtime_lock(venv_dir: Path | None = None) -> Path:
    """Return the hash-pinned ovphysx runtime lock, failing closed when absent."""
    import platform

    override = os.environ.get(_OVPHYSX_RUNTIME_LOCK_ENV)
    if override:
        lock = Path(override).expanduser()
        if not lock.is_file():
            raise RuntimeError(
                f"{_OVPHYSX_RUNTIME_LOCK_ENV} does not point to a readable "
                f"ovphysx runtime lock: {lock}"
            )
        return lock
    system = platform.system().lower()
    machine = platform.machine().lower()
    python_minor = _ovphysx_runtime_python_minor(venv_dir)
    packaged_lock = _PACKAGED_RUNTIME_LOCKS.get(
        (system, machine, python_minor)
    )
    if packaged_lock is not None and packaged_lock.is_file():
        return packaged_lock
    if system == "linux" and machine in {"x86_64", "amd64"}:
        relative = _WU_RUNTIME_LOCK_DEFAULT
    else:
        relative = _WU_RUNTIME_LOCK_RELATIVE.get((system, machine))
    if relative is None:
        raise RuntimeError(
            f"No reviewed OvPhysX runtime is available for {system}/{machine}."
        )
    source_root = Path(__file__).resolve()
    for candidate_root in source_root.parents:
        candidate = candidate_root / relative
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "No hash-pinned ovphysx runtime lock was found. Auto-provisioning "
        "installs only exact, reproducible solver versions; set "
        f"{_OVPHYSX_RUNTIME_LOCK_ENV} to a pylock file (for a "
        f"world-understanding checkout: {_WU_RUNTIME_LOCK_DEFAULT}) or "
        "pre-provision the venv and mark it ready."
    )


def _ovphysx_readiness_marker_body(runtime_lock: Path, python_path: Path) -> str:
    """Return the canonical marker for one reviewed runtime and bound Python."""

    return (
        json.dumps(
            {
                "schema_version": _OVPHYSX_READY_MARKER_SCHEMA_VERSION,
                "runtime_lock_sha256": hashlib.sha256(
                    runtime_lock.read_bytes()
                ).hexdigest(),
                "python_path": str(python_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _uv_command() -> list[str]:
    """Return an invocation of ``uv`` without trusting the ambient PATH alone."""
    import importlib.util
    import shutil

    pinned_executable = os.environ.get(_UV_EXECUTABLE_ENV)
    if pinned_executable:
        executable_path = Path(pinned_executable).expanduser()
        if (
            not executable_path.is_absolute()
            or not executable_path.is_file()
            or not os.access(executable_path, os.X_OK)
        ):
            raise RuntimeError(
                f"{_UV_EXECUTABLE_ENV} does not identify an executable file: "
                f"{pinned_executable}"
            )
        return [str(executable_path.resolve(strict=True))]
    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv"]
    executable = shutil.which("uv")
    if executable:
        return [executable]
    raise RuntimeError(
        "uv is required to provision the pinned ovphysx runtime venv "
        "(dependencies are never installed with bare pip). Install uv or "
        "pre-provision the venv and mark it ready."
    )


def _ovphysx_venv_python_path(venv_dir: Path) -> Path:
    """Return the platform-specific Python executable path for a venv."""

    venv_dir = venv_dir.expanduser().resolve()
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


@contextmanager
def ovphysx_provision_lock(venv_dir: Path | None = None) -> Iterator[None]:
    """Serialize all provisioning and readiness checks for one runtime venv."""

    from usd_core.render.ovrtx import _file_lock

    selected_venv = (venv_dir or DEFAULT_VENV).expanduser().resolve()
    selected_venv.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(
        selected_venv.parent / f"{selected_venv.name}.provision.lock"
    ):
        yield


def _ovphysx_ready_python(venv_dir: Path) -> str | None:
    """Return the lock-bound interpreter only when it still passes its probe."""

    venv_dir = venv_dir.expanduser().resolve()
    py = _ovphysx_venv_python_path(venv_dir)
    marker = venv_dir / _OVPHYSX_READY_MARKER
    runtime_lock = _ovphysx_runtime_lock(venv_dir)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        lock_sha256 = hashlib.sha256(runtime_lock.read_bytes()).hexdigest()
    except (OSError, json.JSONDecodeError):
        return None
    if (
        py.is_file()
        and isinstance(payload, dict)
        and payload.get("schema_version") == _OVPHYSX_READY_MARKER_SCHEMA_VERSION
        and payload.get("runtime_lock_sha256") == lock_sha256
        and payload.get("python_path") == str(py)
    ):
        if _ovphysx_runtime_probe(py):
            return str(py)
    return None


def _ovphysx_runtime_probe(python_path: Path) -> bool:
    """Return whether the isolated interpreter can initialize OvPhysX."""

    probe_environment = os.environ.copy()
    probe_environment.pop("PYTHONPATH", None)
    try:
        subprocess.run(
            [str(python_path), "-c", _OVPHYSX_RUNTIME_PROBE],
            check=True,
            env=probe_environment,
            timeout=_OVPHYSX_RUNTIME_PROBE_TIMEOUT_S,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _ovphysx_python_lock_held(venv_dir: Path) -> str:
    """Resolve or provision one runtime while its mutation lock is held."""

    venv_dir = venv_dir.expanduser().resolve()
    py = _ovphysx_venv_python_path(venv_dir)
    marker = venv_dir / _OVPHYSX_READY_MARKER
    runtime_lock = _ovphysx_runtime_lock(venv_dir)
    ready_python = _ovphysx_ready_python(venv_dir)
    if ready_python is not None:
        return ready_python
    if os.environ.get("WU_OVPHYSX_AUTO_PROVISION", "1") != "1":
        raise RuntimeError(f"ovphysx venv missing at {venv_dir} and auto-provision disabled")
    # Resolve the lock and uv before creating anything so a missing pin fails
    # closed without leaving a half-provisioned venv behind.
    uv = _uv_command()
    if not py.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    # The pylock carries exact artifact URLs and hashes for every runtime
    # dependency (ovphysx, numpy, ...), so identical runs provision the
    # identical native solver and no index resolution can substitute a
    # colliding package name.
    subprocess.run(
        [
            *uv,
            "pip",
            "install",
            "--python",
            str(py),
            "--require-hashes",
            "--no-deps",
            "--no-config",
            "--no-sources",
            "-r",
            str(runtime_lock),
        ],
        check=True,
    )
    probe_environment = os.environ.copy()
    probe_environment.pop("PYTHONPATH", None)
    subprocess.run(
        [
            str(py),
            "-c",
            _OVPHYSX_RUNTIME_PROBE,
        ],
        check=True,
        env=probe_environment,
    )
    marker.write_text(
        _ovphysx_readiness_marker_body(runtime_lock, py), encoding="utf-8"
    )
    return str(py)


def _ovphysx_python(venv_dir: Path) -> str:
    """Resolve or provision one runtime while acquiring its mutation lock."""

    venv_dir = venv_dir.expanduser().resolve()
    ready_python = _ovphysx_ready_python(venv_dir)
    if ready_python is not None:
        return ready_python
    with ovphysx_provision_lock(venv_dir):
        return _ovphysx_python_lock_held(venv_dir)


_DAEMON_ADDRESS_SPACE_LIMIT_PRELUDE = r'''
import os as _limit_os
try:
    import resource as _limit_resource
except ImportError:
    _limit_resource = None
if (
    _limit_os.environ.pop("_USD_CLI_OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE", None) == "1"
    and _limit_resource is not None
):
    _limit_soft, _limit_hard = _limit_resource.getrlimit(_limit_resource.RLIMIT_AS)
    _limit_resource.setrlimit(
        _limit_resource.RLIMIT_AS,
        (_limit_hard, _limit_hard),
    )
del _limit_os, _limit_resource
'''


_DAEMON_SCRIPT = _DAEMON_ADDRESS_SPACE_LIMIT_PRELUDE + r'''
import json, sys
import numpy as np
import ovphysx
from ovphysx import PhysX, TensorType

def _emit(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()

def main():
    physx = PhysX(device="cpu")
    _emit({"status": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        req = json.loads(line)
        if req.get("command") == "shutdown": break
        if req.get("command") != "evaluate":
            _emit({"status": "error", "error": "unknown command"}); continue
        try:
            dt = float(req.get("dt", 1.0/240.0)); fps = int(req.get("sample_fps", 30))
            dur = float(req.get("duration_s", 1.0))
            handle, _ = physx.add_usd(req["scene_usd"])
            pose_b = physx.create_tensor_binding(pattern=req["body_pattern"], tensor_type=TensorType.RIGID_BODY_POSE)
            vel_b = physx.create_tensor_binding(pattern=req["body_pattern"], tensor_type=TensorType.RIGID_BODY_VELOCITY)
            n_bodies = int(pose_b.shape[0]) if pose_b.shape else 0
            steps_per = max(1, int(round((1.0/fps) / dt))); total = max(1, int(round(dur/dt)))
            pbuf = np.zeros(pose_b.shape, dtype=np.float32); vbuf = np.zeros(vel_b.shape, dtype=np.float32)
            pose_b.read(pbuf); vel_b.read(vbuf)
            traj = [[0.0, pbuf[0].tolist(), vbuf[0].tolist()]]
            done = 0; t = 0.0
            while done < total:
                chunk = min(steps_per, total - done)
                physx.step_n_sync(chunk, dt, t)
                done += chunk; t = done * dt
                pose_b.read(pbuf); vel_b.read(vbuf)
                traj.append([float(t), pbuf[0].tolist(), vbuf[0].tolist()])
            physx.remove_usd(handle)
            _emit({"status": "ok", "trajectory": traj, "n_bodies": n_bodies, "n_steps": done})
        except Exception as exc:
            _emit({"status": "error", "error": repr(exc)})

main()
'''


def _ovphysx_daemon_environment(
    *,
    relax_address_space_limit: bool,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    # Never trust an ambient marker. Only an explicit call-site opt-in grants
    # the dedicated OvPhysX child permission to relax its inherited soft cap.
    environment.pop(_OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV, None)
    if relax_address_space_limit:
        environment[_OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV] = "1"
    return environment


class _OvPhysXDaemon:
    def __init__(
        self,
        venv_dir: Path | None = None,
        *,
        relax_address_space_limit: bool = False,
    ):
        selected_venv = (Path(venv_dir) if venv_dir else DEFAULT_VENV).expanduser().resolve()
        env = _ovphysx_daemon_environment(
            relax_address_space_limit=relax_address_space_limit,
        )
        self._errfile = tempfile.NamedTemporaryFile(prefix="ovphysx_err_", suffix=".log", delete=False)
        self._lock = threading.Lock()
        self._lines: queue.Queue[str | None] = queue.Queue()
        provision_lock = ovphysx_provision_lock(selected_venv)
        provision_lock_held = False
        try:
            try:
                provision_lock.__enter__()
            except PermissionError as exc:
                if exc.errno != errno.EACCES or os.access(selected_venv, os.W_OK):
                    raise
                # A root-owned container runtime cannot create its sibling lock.
                # Reuse it only when the packaged lock digest and bound Python
                # already validate; an incomplete baked runtime still fails closed.
                ready_python = _ovphysx_ready_python(selected_venv)
                if ready_python is None:
                    raise
                self._py = ready_python
            else:
                provision_lock_held = True
                self._py = _ovphysx_python_lock_held(selected_venv)
            self._proc = subprocess.Popen(
                [self._py, "-c", _DAEMON_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._errfile,
                text=True,
                env=env,
            )
            threading.Thread(target=self._pump, daemon=True).start()
            if self._read_status(_START_TIMEOUT_S).get("status") != "ready":
                raise RuntimeError("ovphysx daemon failed to start")
        finally:
            if provision_lock_held:
                provision_lock.__exit__(None, None, None)

    def _pump(self):
        for line in self._proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _read_status(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                self._kill(); raise RuntimeError(f"ovphysx daemon timed out after {timeout_s:.0f}s")
            try:
                line = self._lines.get(timeout=rem)
            except queue.Empty:
                self._kill(); raise RuntimeError("ovphysx daemon timed out")
            if line is None:
                err = Path(self._errfile.name).read_text()[-2000:]
                raise RuntimeError(f"ovphysx daemon died: {err.strip()}")
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "status" in obj:
                return obj

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def evaluate(self, *, scene_usd, body_pattern, duration_s, dt, sample_fps) -> dict:
        _checked_total_steps(duration_s, dt)  # refuse runaway totals before the solver
        with self._lock:
            req = {"command": "evaluate", "scene_usd": str(scene_usd), "body_pattern": body_pattern,
                   "duration_s": duration_s, "dt": dt, "sample_fps": sample_fps}
            self._proc.stdin.write(json.dumps(req) + "\n"); self._proc.stdin.flush()
            resp = self._read_status(_EVAL_TIMEOUT_S)
            if resp.get("status") == "error":
                raise RuntimeError(f"ovphysx error: {resp.get('error')}")
            return resp

    def _kill(self):
        try:
            self._proc.kill(); self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        try:
            self._proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n"); self._proc.stdin.flush()
            self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self._kill()


# ── remote executor ─────────────────────────────────────────────────────────────────
def evaluate_remote(scene_usd, *, body_pattern: str, duration_s: float, dt: float,
                    sample_fps: int, base_url: str, api_key: str | None = None,
                    timeout: float = 1800.0, verify_version: bool = True) -> dict:
    """Run a caller-authored simulation through the managed OVRTX adapter.

    Same contract as `_OvPhysXDaemon.evaluate`: the scene travels as the raw (gzipped when
    that helps) flattened USD, the response carries {trajectory, n_bodies, n_steps}. The
    recording and metrics are still authored locally by the caller.
    """
    import gzip

    import httpx  # lazy

    if verify_version:
        from usd_core.remote_protocol import check_remote_protocol
        check_remote_protocol(
            base_url,
            required_engine="ovrtx",
            required_features=(REMOTE_PHYSICS_FEATURE,),
            api_key=api_key,
        )

    scene_path = Path(scene_usd)
    data = scene_path.read_bytes()
    compression = "none"
    gz = gzip.compress(data, compresslevel=6)
    if len(gz) < len(data) * 0.95:
        data, compression = gz, "gzip"
    params = {"body_pattern": body_pattern, "duration_s": duration_s, "dt": dt,
              "sample_fps": sample_fps, "compression": compression}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = f"{base_url.rstrip('/')}/physics/simulate"
    # The service's ovphysx daemon boots lazily on the first call, so a cold service can answer
    # 503 (starting) or 429 (busy). Wait it out with a few probes rather than failing fast —
    # but bound the total so a permanently-down service still errors clearly.
    deadline = time.monotonic() + max(timeout, 60.0)
    attempt = 0
    with httpx.Client() as client:
        while True:
            attempt += 1
            resp = client.post(url,
                               files={"file": (scene_path.name, data, "application/octet-stream")},
                               data={"params": json.dumps(params)},
                               headers=headers, timeout=timeout)
            if resp.status_code in (404, 405):
                raise RuntimeError(
                    "the managed OVRTX adapter has no /physics/simulate endpoint — rebuild it "
                    "from the current apps/ovrtx_rendering_api image to enable remote physics")
            if resp.status_code in (429, 503) and time.monotonic() < deadline:
                time.sleep(min(5.0, 1.0 * attempt))  # physics daemon still warming up
                continue
            break
    if resp.status_code in (429, 503):
        raise RuntimeError(
            f"remote physics unavailable after {attempt} attempt(s) (HTTP {resp.status_code}); "
            "the OVRTX service's physics daemon never became ready")
    if resp.status_code >= 400:
        # Mirror the render client: surface the service's response body instead
        # of a bare "500 Internal Server Error" (wu round: undiagnosable 500s).
        # Strip C0/ESC controls — the body is untrusted and goes to a terminal.
        body = "".join(c for c in (resp.text or "").strip()[:500]
                       if c == "\n" or (ord(c) >= 32 and c != "\x7f"))
        raise RuntimeError(
            f"remote physics failed: HTTP {resp.status_code} from {url} "
            f"(scene {scene_path.name})" + (f" — {body}" if body else ""))
    return resp.json()


def _evaluate_remote_composed_scene(
    scene_path: Path,
    *,
    staging_dir: Path,
    body_pattern: str,
    duration_s: float,
    dt: float,
    sample_fps: int,
    remote: dict,
) -> dict:
    """Flatten the local composition before sending one self-contained USD.

    The remote service opens an uploaded file in an otherwise empty temporary
    directory. Sending only a caller's root layer would therefore lose sublayers,
    references, and payloads. Physics does not need texture assets, so a binary
    flattened layer is the smallest complete transport for the composed geometry
    and schemas that the solver consumes.
    """

    from pxr import Usd

    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise ValueError(f"could not open physics scene: {scene_path}")
    with tempfile.TemporaryDirectory(
        prefix=".usd-cli_remote_physics_",
        dir=str(staging_dir),
    ) as temporary_dir:
        flattened_path = Path(temporary_dir) / "scene.usdc"
        if not stage.Flatten().Export(str(flattened_path)):
            raise RuntimeError(
                f"could not flatten physics scene for remote simulation: {scene_path}"
            )
        return evaluate_remote(
            flattened_path,
            body_pattern=body_pattern,
            duration_s=duration_s,
            dt=dt,
            sample_fps=sample_fps,
            **remote,
        )


# ── orchestration ───────────────────────────────────────────────────────────────────
def simulate_scene(scene_usd: str | Path, output_dir: str, *, body_path: str,
                   rest_position: list[float], world_up: list[float],
                   body_pattern: str | None = None,
                   engine: str = "ovphysx", duration_s: float = 1.0,
                   dt: float = 1.0 / 240.0, sample_fps: int = 30,
                   remote: dict | None = None,
                   relax_address_space_limit: bool = False) -> dict:
    """Simulate one explicit, pre-authored USD scene with ovphysx.

    The caller supplies the scenario USD, simulated body, and metric coordinate
    system. Constructing a drop test, selecting bodies, and evaluating results
    are workflow responsibilities.
    Only the real ovphysx solver is supported. It runs locally on a supported Linux or Windows host;
    elsewhere the simulation is sent to the managed OVRTX adapter described by `remote`
    ({base_url, api_key, timeout} — the render backend's connection). With neither
    available this raises rather than fabricating a trajectory.
    """
    if engine != "ovphysx":
        raise ValueError(f"unknown physics engine '{engine}' (only 'ovphysx' is supported)")
    # Reject a runaway duration/dt combination up front — before authoring the
    # scene, provisioning a daemon, or uploading anything to the remote service.
    _checked_total_steps(duration_s, dt)
    local_ok = ovphysx_platform_supported()
    if not local_ok and not (remote and remote.get("base_url")):
        raise RuntimeError(
            "runtime physics requires a supported local OvPhysX runtime or a compatible remote "
            "OVRTX physics backend; neither is available — set one up with "
            "`OVRTX_API_KEY=<key> usd-cli remote "
            "configure <url>` to run the simulation on the OVRTX service instead (pass the "
            "key through the environment; remote configure never accepts credentials on the "
            "command line)")

    # `-o recording.usda` means "put the recording THERE", not "make a directory
    # named recording.usda" (that produced recording.usda/recording.usda). An
    # EXISTING directory whose name merely ends in .usda is honored as a
    # directory; and a custom recording name prefixes the sidecar artifacts so
    # `-o run1.usda` and `-o run2.usda` in one dir don't clobber each other.
    scene_path = Path(scene_usd).expanduser().resolve(strict=True)
    from pxr import Sdf

    if (
        not isinstance(body_path, str)
        or not Sdf.Path.IsValidPathString(body_path)
        or not Sdf.Path(body_path).IsAbsolutePath()
        or not Sdf.Path(body_path).IsPrimPath()
        or body_path == "/"
    ):
        raise ValueError("body_path must be an exact absolute prim path")
    solver_body_pattern = body_pattern or body_path
    if not isinstance(solver_body_pattern, str) or not solver_body_pattern.startswith("/"):
        raise ValueError("body_pattern must be an absolute solver binding pattern")
    if len(rest_position) != 3 or len(world_up) != 3:
        raise ValueError("rest_position and world_up must each contain three values")
    out = Path(output_dir)
    recording_name = "recording.usda"
    sidecar_prefix = ""
    if out.suffix.lower() in {".usd", ".usda", ".usdc"} and not out.is_dir():
        recording_name = out.name
        if out.stem != "recording":
            sidecar_prefix = f"{out.stem}_"
        out = out.parent if str(out.parent) not in ("", ".") else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    if local_ok:
        executor = "local"
        daemon = _OvPhysXDaemon(
            relax_address_space_limit=relax_address_space_limit,
        )
        try:
            resp = daemon.evaluate(scene_usd=scene_path, body_pattern=solver_body_pattern,
                                   duration_s=duration_s, dt=dt, sample_fps=sample_fps)
        finally:
            daemon.close()
    else:
        executor = "remote"
        resp = _evaluate_remote_composed_scene(
            scene_path,
            staging_dir=out,
            body_pattern=solver_body_pattern,
            duration_s=duration_s,
            dt=dt,
            sample_fps=sample_fps,
            remote=remote,
        )
    trajectory = [(float(t), [float(v) for v in p], [float(v) for v in vv])
                  for t, p, vv in resp.get("trajectory", [])]
    n_bodies = int(resp.get("n_bodies") or 0)

    finite = all(math.isfinite(v) for _t, p, vv in trajectory for v in (*p, *vv))
    # The CLI records what the solver returned; deciding whether a trajectory is
    # sufficiently complete, bounded, or settled is workflow policy.  In
    # particular, do not turn a heuristic threshold into a CLI verdict here.
    jsonl = author_trajectory_jsonl(
        trajectory, str(out / f"{sidecar_prefix}trajectory.jsonl")
    )
    recording = None
    metrics = {}
    if trajectory and finite:
        recording = author_trajectory_usda(str(scene_path), trajectory, body_path,
                                           str(out / recording_name), fps=sample_fps)
        metrics = trajectory_metrics(trajectory, rest_position, world_up)
    report = {"engine": engine, "executor": executor, "n_bodies": n_bodies,
              "scene_usd": str(scene_path), "recording_usda": recording,
              "trajectory_jsonl": jsonl,
              "metrics": metrics,
              "simulation_facts": {
                  "trajectory_sample_count": len(trajectory),
                  "trajectory_finite": finite,
                  "reported_body_count": n_bodies,
                  "reported_step_count": resp.get("n_steps"),
                  "modeled_body_path": body_path,
                  "solver_body_pattern": solver_body_pattern,
              },
              "scenario": {
                  "scene_usd": str(scene_path), "body_path": body_path,
                  "body_pattern": solver_body_pattern,
                  "rest_position": [float(value) for value in rest_position],
                  "world_up": [float(value) for value in world_up],
              }}
    (out / f"{sidecar_prefix}runtime_validation_report.json").write_text(json.dumps(report, indent=2))
    report["report_path"] = str(out / f"{sidecar_prefix}runtime_validation_report.json")
    return report
