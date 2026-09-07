# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select the local or remote OVRTX render backend."""

from __future__ import annotations

from pathlib import Path

from usd_core.config import Config
from usd_core.render.base import RenderBackend


def _ovrtx_available() -> bool:
    """True only where a real local OVRTX render could plausibly run.

    Kept import-light (no pxr/ovrtx import) — just probes for an NVIDIA driver so
    ``auto`` does not pick OVRTX on an unsupported host and fail at render time.
    """
    import glob
    import os
    import platform

    system = platform.system()
    if system == "Windows":
        system_root = os.environ.get("SystemRoot")
        return bool(
            system_root and (Path(system_root) / "System32" / "nvcuda.dll").is_file()
        )
    if system != "Linux":
        return False
    return (
        Path("/proc/driver/nvidia/version").exists()
        or bool(glob.glob("/dev/nvidia[0-9]*"))
    )


def _resolve_auto(config: Config) -> str:
    """Resolve ``auto`` to an available local or remote OVRTX runtime."""
    if _ovrtx_available():
        return "ovrtx"
    from usd_core.config import resolve_render_backends

    if resolve_render_backends(config.render):
        # No local renderer, but the managed OVRTX adapter is configured (remote_url or a
        # [[render.backends]] pool) — use it. Without this fallback `--renderer auto` on
        # a CPU box claimed nothing was available even when render.remote_url was set,
        # and agents wrongly gave up on rendering.
        return "remote"
    raise RuntimeError(
        "no OVRTX renderer is available: provision a local NVIDIA GPU runtime or "
        "configure a remote OVRTX backend"
    )


def resolved_renderer(config: Config) -> str:
    """The concrete renderer that will actually run (resolve 'auto'). For display."""
    r = (config.render.get("renderer") or "auto").lower()
    if r != "auto":
        return r
    try:
        return _resolve_auto(config)
    except Exception:  # noqa: BLE001 — nothing available; report it rather than raise
        return "none"


def make_backend(config: Config) -> RenderBackend:
    renderer = (config.render.get("renderer") or "auto").lower()
    if renderer == "auto":
        renderer = _resolve_auto(config)

    if renderer == "ovrtx":
        from usd_core.render.ovrtx import OvRTXRenderBackend
        return OvRTXRenderBackend(
            num_sensor_updates=int(config.render.get("ovrtx_num_sensor_updates", 64)),
            render_mode=str(config.render.get("ovrtx_render_mode", "") or ""),
            log_level=str(config.render.get("ovrtx_log_level", "warn")),
            auto_install=str(config.render.get("ovrtx_auto_install", False)).lower()
            in ("true", "1", "yes"),
        )
    if renderer == "remote":
        from usd_core.config import resolve_render_backends
        from usd_core.render.remote import RemoteRenderBackend
        return RemoteRenderBackend(
            base_url=config.render.get("remote_url", ""),
            api_key=config.render.get("remote_api_key") or None,
            # [[render.backends]] pool (falls back to the single fields above when
            # empty) — dedup and normalization happen in resolve_render_backends
            backends=resolve_render_backends(config.render),
            timeout=float(config.render.get("remote_timeout", 300)),
            compress=str(config.render.get("remote_compress", "true")).lower()
            not in ("false", "0", "no"),
            max_upload_mb=float(config.render.get("remote_max_upload_mb", 0) or 0),
            verify_version=str(config.render.get("remote_verify_version", "true")).lower()
            not in ("false", "0", "no"),
            bundle_cache=str(config.render.get("remote_bundle_cache", "true")).lower()
            not in ("false", "0", "no"),
            cas=str(config.render.get("remote_cas", "true")).lower()
            not in ("false", "0", "no"),
        )

    raise ValueError(
        f"unknown renderer '{renderer}' (expected auto|ovrtx|remote; "
        "OVRTX is the only supported rendering engine)"
    )
