# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics simulator wrapper: uploaded drop-settle scene → usd_core ovphysx daemon → trajectory.

Runs in the service (main) process; the actual solve happens in the isolated ovphysx
daemon subprocess that usd_core.physics_runtime spawns (same pattern as the render side).
The daemon is provisioned lazily on first use and kept alive across requests; a dead
daemon is replaced on the next request.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class Simulator:
    def __init__(self) -> None:
        self._daemon = None
        self._lock = threading.Lock()

    @property
    def daemon_running(self) -> bool:
        return self._daemon is not None and self._daemon.alive

    def _get_daemon(self):
        from usd_core.physics_runtime import _OvPhysXDaemon

        with self._lock:
            if self._daemon is not None and not self._daemon.alive:
                logger.warning("ovphysx daemon died; restarting")
                self._daemon = None
            if self._daemon is None:
                self._daemon = _OvPhysXDaemon()
            return self._daemon

    def simulate_scene_bytes(self, data: bytes, *, filename: str, body_pattern: str,
                             duration_s: float, dt: float, sample_fps: int) -> dict:
        """Simulate an uploaded scene (flattened USD/USDA/USDZ bytes); return the daemon's
        {trajectory, n_bodies, n_steps} response."""
        suffix = Path(filename).suffix.lower()
        if suffix not in (".usd", ".usda", ".usdc", ".usdz"):
            suffix = ".usda"
        with tempfile.TemporaryDirectory(prefix="ovphysx_api_") as tmp:
            scene_path = Path(tmp) / f"scene{suffix}"
            scene_path.write_bytes(data)
            # reject unopenable scenes here (400) rather than deep inside the solver (500)
            from pxr import Tf, Usd
            try:
                stage = Usd.Stage.Open(str(scene_path))
            except Tf.ErrorException as exc:
                raise ValueError(f"could not open the uploaded scene as USD: {exc}") from exc
            if not stage:
                raise ValueError("could not open the uploaded scene as USD")
            del stage
            daemon = self._get_daemon()
            resp = daemon.evaluate(scene_usd=scene_path, body_pattern=body_pattern,
                                   duration_s=duration_s, dt=dt, sample_fps=sample_fps)
        return {"trajectory": resp.get("trajectory", []),
                "n_bodies": int(resp.get("n_bodies") or 0),
                "n_steps": int(resp.get("n_steps") or 0)}

    def close(self) -> None:
        with self._lock:
            if self._daemon is not None:
                self._daemon.close()
                self._daemon = None
