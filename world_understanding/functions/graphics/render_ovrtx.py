# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD rendering functions using OvRTX local RTX renderer.

This module provides rendering functions that use the ovrtx library for
local, in-process RTX rendering. It combines the quality of RTX rendering
with the low latency of local execution (no cloud overhead).

Because ovrtx bundles its own USD C libraries which conflict with external
``pxr``/OpenUSD bindings at the shared-library level, all ovrtx work runs in
an isolated subprocess using a separate virtual environment that has ovrtx
installed without another ``pxr`` provider. The main process exports the stage
to a temp file and the subprocess does the actual rendering.

Requires: ovrtx == 0.3.0.312915
"""

import atexit
import hashlib
import json
import logging
import os
import re
import selectors
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources as importlib_resources
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import numpy as np
from filelock import FileLock, Timeout
from PIL import Image

from world_understanding.functions.graphics.material_targets import (
    normalize_render_material_target,
    preview_fallbacks_enabled_for_material_target,
)
from world_understanding.utils.image_blankness import analyze_image_blankness

if TYPE_CHECKING:  # pragma: no cover
    from pxr import Usd

logger = logging.getLogger(__name__)
_REMOTE_ASSET_SCHEMES = frozenset({"http", "https"})
# Keep per-frame render-loop analysis bounded; downstream dataset checks can
# run deeper image analysis when deciding whether to fail a pipeline.
_BLANKNESS_MAX_ANALYSIS_PIXELS = 65_536
_NATIVE_VISIBILITY_PROBE_ENV = "WU_OVRTX_EXPERIMENTAL_NATIVE_VISIBILITY"
_NATIVE_DISPLAYCOLOR_PROBE_ENV = "WU_OVRTX_EXPERIMENTAL_NATIVE_DISPLAYCOLOR"
_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

# Mapping from WU sensor names to ovrtx render variable names
_SENSOR_TO_RENDER_VAR: dict[str, str] = {
    "depth": "Depth",
    "normal": "Normal",
    "albedo": "Albedo",
}


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    """Return True when ``path`` is contained by ``parent``."""
    try:
        return path.is_relative_to(parent)
    except ValueError:
        return False


def _looks_like_windows_drive_path(value: str) -> bool:
    """Return True for Windows drive-qualified paths parsed as URL schemes."""
    return (
        len(value) >= 2
        and value[1] == ":"
        and value[0].isascii()
        and value[0].isalpha()
    )


def _is_remote_asset_path(value: str) -> bool:
    """Return True for URL-like asset paths, excluding Windows drive paths."""
    scheme = urlparse(value).scheme.lower()
    return (
        bool(scheme) and scheme != "file" and not _looks_like_windows_drive_path(value)
    )


def _is_local_asset_path(value: str) -> bool:
    """Return True for filesystem paths that should exist before binding."""
    return not _is_remote_asset_path(value)


def _local_asset_path(value: str) -> Path:
    """Return a filesystem path for local asset syntax, including file:// URIs."""
    parsed = urlparse(value)
    if parsed.scheme.lower() != "file":
        return Path(value)

    path = unquote(parsed.path)
    if parsed.netloc and _looks_like_windows_drive_path(parsed.netloc):
        return Path(f"{parsed.netloc}{path}")
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        return Path(f"//{parsed.netloc}{path}")
    if len(path) >= 4 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


def _sanitize_ovrtx_path_env(path_value: str) -> str:
    """Drop PATH entries that OVRTX cannot stat while scanning DLL locations."""
    entries: list[str] = []
    for entry in path_value.split(os.pathsep):
        if not entry:
            entries.append(entry)
            continue
        expanded = os.path.expandvars(entry).strip('"')
        try:
            path = Path(expanded)
            if path.exists() and not path.is_dir():
                continue
        except OSError:
            logger.debug("Skipping inaccessible OVRTX PATH entry: %s", entry)
            continue
        entries.append(entry)
    return os.pathsep.join(entries)


def _ovrtx_subprocess_env() -> dict[str, str]:
    """Return an environment safe for OVRTX import probes and workers."""
    env = os.environ.copy()
    for key in ("PATH", "Path"):
        if key in env:
            env[key] = _sanitize_ovrtx_path_env(env[key])
    return env


def _native_visibility_probe_enabled() -> bool:
    """Return True for the OVRTX 0.3 native-visibility validation probe.

    Default production behavior keeps the 0.2-era visibility overlay
    workaround. This opt-in exists only so GPU validation can exercise the
    real render path with time-sampled visibility left in the exported USD.
    """
    return (
        os.environ.get(_NATIVE_VISIBILITY_PROBE_ENV, "").strip().lower()
        in _TRUE_ENV_VALUES
    )


def _native_displaycolor_probe_enabled() -> bool:
    """Return True for the OVRTX 0.3 native-displayColor validation probe.

    Default production behavior keeps the displayColor frame overlay
    workaround. This opt-in exists only so GPU validation can exercise native
    time-sampled ``primvars:displayColor`` without removing the safe path.
    """
    return (
        os.environ.get(_NATIVE_DISPLAYCOLOR_PROBE_ENV, "").strip().lower()
        in _TRUE_ENV_VALUES
    )


# Default location for the auto-provisioned ovrtx venv.
# Honour WU_OVRTX_VENV_DIR env var so Docker images can ship a pre-built venv.
_DEFAULT_OVRTX_VENV_DIR = Path.home() / ".cache" / "wu" / "ovrtx_venv"
_OVRTX_VENV_DIR = Path(
    os.environ.get("WU_OVRTX_VENV_DIR", str(_DEFAULT_OVRTX_VENV_DIR))
).expanduser()
_OVRTX_MANAGED_MARKER = ".wu-managed-ovrtx-venv"
_OVRTX_PROVISIONING_MARKER = ".wu-managed-ovrtx-venv.provisioning"
_OVRTX_PROVISION_LOCK_TIMEOUT_S = 600
_OVRTX_PROVISION_LOCK_TIMEOUT_SECONDS = _OVRTX_PROVISION_LOCK_TIMEOUT_S

# Cached path to the ovrtx venv Python executable
_ovrtx_python: str | None = None
_ovrtx_python_cache: dict[Path, str] = {}
_verified_ovrtx_python_cache: set[tuple[Path, str]] = set()
_verified_managed_ovrtx_python_cache: set[Path] = set()

# Number of ``renderer.step(delta_time=0)`` iterations per frame. OVRtx's
# path tracer accumulates samples across successive step() calls when
# ``delta_time`` is zero. In 0.2.0 testing this was the only quality knob
# that actually had effect; keep using it until 0.3 GPU validation proves
# the schema sample attributes are honored. Empirically (see the
# ovrtx_kit_parity.py convergence-cap sweep on the kit-gen-ai-service golden
# scene), PT mode plateaued at ~500 steps / ~39.7 dB PSNR vs the Kit reference.
#
# The field is named ``num_sensor_updates`` on the wire for historical reasons
# (and Kit's rendering-api uses ``num_sensor_updates`` for the same
# concept — outer update loop) but is semantically *iteration count*,
# not samples-per-pixel. The schema-level ``omni:rtx:pt:samplesPerPixel``
# attribute was silently ignored in 0.2.0 and remains guarded pending 0.3
# validation.
DEFAULT_NUM_SENSOR_UPDATES = 32

# Default RTX render mode. ``pt`` maps to ``PathTracing`` — Kit's
# ground-truth mode. rt2 is available as an override for callers that
# want real-time-path-tracing speed, but pt is the quality-parity target.
DEFAULT_RENDER_MODE = "rt2"

# Bound the lifetime of the native renderer process. ``reset_stage()`` clears
# USD scene state, but it does not guarantee that OVRTX/Vulkan allocations are
# returned to the operating system. A long-lived service therefore recycles
# the isolated process before the next render after either guard trips. The
# count guard is deterministic and portable; the RSS guard catches a single
# unusually expensive scene on Linux. Set either environment variable to 0 to
# disable that guard.
OVRTX_DAEMON_MAX_RENDERS_ENV = "OVRTX_DAEMON_MAX_RENDERS"
OVRTX_DAEMON_MAX_RSS_BYTES_ENV = "OVRTX_DAEMON_MAX_RSS_BYTES"
DEFAULT_OVRTX_DAEMON_MAX_RENDERS = 64
DEFAULT_OVRTX_DAEMON_MAX_RSS_BYTES = 24 * 1024 * 1024 * 1024


def _parse_nonnegative_int_env(name: str, default: int) -> int:
    """Return a non-negative integer environment setting."""
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _linux_process_rss_bytes(
    pid: int, *, proc_root: Path = Path("/proc")
) -> int | None:
    """Return Linux resident bytes for ``pid``, or ``None`` when unavailable."""
    try:
        fields = (proc_root / str(pid) / "statm").read_text(encoding="utf-8").split()
        resident_pages = int(fields[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, ValueError):
        return None
    if resident_pages < 0 or page_size <= 0:
        return None
    return resident_pages * page_size


# Map the short mode tokens accepted on the wire (kit-gen-ai-service's
# ``RenderMode`` enum values) to the ``omni:rtx:rendermode`` token the
# RTX engine expects.
_RENDER_MODE_TOKENS: dict[str, str] = {
    "rt1": "RaytracedLighting",
    "rt2": "RealTimePathTracing",
    "pt": "PathTracing",
}

# RTX USD api schemas. Applying these on the RenderProduct is required
# so OVRtx's ``omni:rtx:rendermode`` attribute is honored — without the
# schema prepend, the RenderProduct is treated as a bare RenderProduct
# and the engine falls back to its internal default (RaytracedLighting,
# which isn't even a supported token at runtime, and gets remapped to
# RealTimePathTracing).
#
# The schema names come from ovrtx's bundled rtx_settings plugin —
# ``.ovrtx_venv/.../ovrtx/bin/usd_plugins/rtx_settings/generatedSchema.usda``.
_RTX_RENDER_PRODUCT_API_SCHEMAS = (
    "OmniRtxSettingsCommonAdvancedAPI_1",
    "OmniRtxSettingsRtAdvancedAPI_1",
    "OmniRtxSettingsPtAdvancedAPI_1",
)
_RTX_PT_SAMPLES_ATTR = "omni:rtx:pt:samplesPerPixel"
_RTX_RT_ACCUMULATION_ATTR = "omni:rtx:rt:accumulationLimit"


def _parse_frames(frames: str) -> list[int]:
    """Parse a frames string into a list of integer frame numbers.

    Supports three formats:
    - Single frame: "0", "42"
    - Frame range: "0:10" (inclusive, produces [0, 1, ..., 10])
    - Comma-separated: "0,5,10"

    Args:
        frames: Frame specification string.

    Returns:
        Sorted list of integer frame numbers.

    Raises:
        ValueError: If the frames string cannot be parsed.

    Examples:
        >>> _parse_frames("0")
        [0]
        >>> _parse_frames("0:3")
        [0, 1, 2, 3]
        >>> _parse_frames("0,5,10")
        [0, 5, 10]
    """
    frames = frames.strip()

    if ":" in frames:
        parts = frames.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid frame range: '{frames}'. Expected 'start:end'.")
        start = int(parts[0])
        end = int(parts[1])
        return list(range(start, end + 1))
    elif "," in frames:
        return sorted(int(f.strip()) for f in frames.split(",") if f.strip())
    else:
        return [int(frames)]


def _build_visibility_frame_updates(
    visibility_schedule: dict[str, dict[str, str]],
    frames: list[int],
) -> dict[str, dict[str, str]]:
    """Collapse full-frame visibility samples to per-frame deltas.

    ``render_all_cameras`` now handles time-sampled visibility through
    per-frame static overlay layers because OVRTX 0.2.0 crashed on both
    authored visibility samples and visibility ``write_attribute`` updates.
    This helper remains for the legacy worker/daemon parameter path until
    native 0.3 visibility handling is validated on GPU.
    """
    if not visibility_schedule:
        return {}

    current_visibility: dict[str, str] = {}
    frame_updates: dict[str, dict[str, str]] = {}

    for frame_num in frames:
        frame_key = str(float(frame_num))
        vis_map = visibility_schedule.get(frame_key)
        if not vis_map:
            continue

        changes: dict[str, str] = {}
        for prim_path, vis_value in vis_map.items():
            token = "inherited" if vis_value == "inherited" else "invisible"
            if current_visibility.get(prim_path, "inherited") == token:
                continue
            current_visibility[prim_path] = token
            changes[prim_path] = token

        if changes:
            frame_updates[frame_key] = changes

    return frame_updates


def _write_frame_overlay(
    overlay_path: str,
    visibility_values: dict[str, str],
    display_color_values: dict[str, Any],
) -> None:
    """Write a USDA layer with static per-frame opinions."""
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateNew(overlay_path)
    for prim_path, vis_value in visibility_values.items():
        token = "inherited" if vis_value == "inherited" else "invisible"
        prim = stage.OverridePrim(prim_path)
        prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set(token)
    for prim_path, display_color in display_color_values.items():
        prim = stage.OverridePrim(prim_path)
        prim.CreateAttribute(
            "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
        ).Set(display_color)
    stage.GetRootLayer().Save()


def _build_render_products_usda(
    cameras: list[str],
    image_width: int,
    image_height: int,
    sensors: list[str] | None = None,
    render_mode: str = DEFAULT_RENDER_MODE,
    *,
    pt_samples_per_pixel: int | None = None,
    rt_accumulation_limit: int | None = None,
) -> tuple[str, list[str]]:
    """Generate USDA content defining RenderProduct prims for each camera.

    Each RenderProduct references a camera, specifies the render resolution,
    and carries the ``omni:rtx:rendermode`` USD attribute so OVRtx's
    ``step()`` picks the right RTX mode. The api schema prepend is the
    tripwire that unlocks the attribute on the RenderProduct.

    Historical ovrtx 0.2.0 limitation: the schema-level attributes
    ``omni:rtx:pt:samplesPerPixel`` and ``omni:rtx:rt:accumulationLimit``
    are defined in the bundled rtx_settings plugin but are silently
    ignored by the render path (verified by the step-cost + noise-proxy
    timing test in /tmp/ovrtx_verify.py — extreme values produce
    bitwise-identical timings and noise). Keep this guarded on 0.3 until
    GPU validation proves those attributes affect output. The quality knob
    we currently rely on is the number of ``renderer.step(delta_time=0)``
    iterations in the daemon. We therefore do not emit those attributes
    here — keeping the USDA to the two things already validated to matter:
    api schemas + rendermode.

    Args:
        cameras: List of camera prim paths (e.g., ["/Cameras/Camera1"]).
        image_width: Render width in pixels.
        image_height: Render height in pixels.
        sensors: Optional list of sensor names to include (e.g., ["depth"]).
        render_mode: ``rt1``/``rt2``/``pt`` short token — translated to
            ``omni:rtx:rendermode`` (RaytracedLighting / RealTimePathTracing /
            PathTracing) on the RenderProduct.
        pt_samples_per_pixel: Probe-only value for
            ``omni:rtx:pt:samplesPerPixel``. Production render products leave
            this unset until OVRTX 0.3 GPU evidence proves it is honored.
        rt_accumulation_limit: Probe-only value for
            ``omni:rtx:rt:accumulationLimit``. Production render products leave
            this unset until OVRTX 0.3 GPU evidence proves it is honored.

    Returns:
        Tuple of (usda_string, product_paths):
            - usda_string: USDA layer content to sublayer into the stage
            - product_paths: List of RenderProduct prim paths for step()

    Raises:
        ValueError: If ``render_mode`` is not one of ``rt1``/``rt2``/``pt``.
    """
    if render_mode not in _RENDER_MODE_TOKENS:
        raise ValueError(
            f"Unknown render_mode: {render_mode!r}. "
            f"Expected one of {sorted(_RENDER_MODE_TOKENS)}."
        )
    if pt_samples_per_pixel is not None and pt_samples_per_pixel < 1:
        raise ValueError("pt_samples_per_pixel must be a positive integer")
    if rt_accumulation_limit is not None and rt_accumulation_limit < 1:
        raise ValueError("rt_accumulation_limit must be a positive integer")
    rendermode_token = _RENDER_MODE_TOKENS[render_mode]
    api_schema_list = ", ".join(f'"{s}"' for s in _RTX_RENDER_PRODUCT_API_SCHEMAS)
    product_paths = []
    render_var_defs = []
    render_var_refs = []

    # Always include LdrColor (matches ovrtx reference format)
    render_var_defs.append(
        '        def RenderVar "LdrColor"\n'
        "        {\n"
        '            uniform string sourceName = "LdrColor"\n'
        "        }\n"
    )
    render_var_refs.append("</Render/Vars/LdrColor>")

    # Add sensor render vars
    for sensor_name in sensors or []:
        render_var_name = _map_sensor_to_render_var(sensor_name)
        if render_var_name is None:
            logger.warning(
                "Unknown sensor '%s', skipping render var generation", sensor_name
            )
            continue

        render_var_defs.append(
            f'        def RenderVar "{render_var_name}"\n'
            f"        {{\n"
            f'            uniform string sourceName = "{render_var_name}"\n'
            f"        }}\n"
        )
        render_var_refs.append(f"</Render/Vars/{render_var_name}>")

    # Build render var references string
    ordered_vars = ", ".join(render_var_refs)

    # Build product definitions. In 0.2.0 validation only the api schema
    # prepend and ``omni:rtx:rendermode`` token influenced rendering; the
    # bundled sample/accumulation-limit attributes were ignored. Keep that
    # guarded behavior until 0.3 GPU validation proves otherwise.
    sample_attr_lines = []
    if pt_samples_per_pixel is not None:
        sample_attr_lines.append(
            f"        uint {_RTX_PT_SAMPLES_ATTR} = {pt_samples_per_pixel}\n"
        )
    if rt_accumulation_limit is not None:
        sample_attr_lines.append(
            f"        int {_RTX_RT_ACCUMULATION_ATTR} = {rt_accumulation_limit}\n"
        )
    sample_attrs = "".join(sample_attr_lines)

    product_defs = []
    for camera_path in cameras:
        # Sanitize camera path for use as prim name
        safe_name = camera_path.strip("/").replace("/", "_")
        product_prim_name = f"Product_{safe_name}"
        product_path = f"/Render/{product_prim_name}"
        product_paths.append(product_path)

        product_defs.append(
            f'    def RenderProduct "{product_prim_name}" (\n'
            f"        prepend apiSchemas = [{api_schema_list}]\n"
            f"    )\n"
            f"    {{\n"
            f"        rel camera = <{camera_path}>\n"
            f"        rel orderedVars = [{ordered_vars}]\n"
            f"        uniform int2 resolution = ({image_width}, {image_height})\n"
            f'        token omni:rtx:rendermode = "{rendermode_token}"\n'
            f"{sample_attrs}"
            f"    }}\n"
        )

    # Assemble full USDA (matching ovrtx reference structure)
    products_block = "\n".join(product_defs)
    vars_block = "\n".join(render_var_defs)

    usda = (
        "#usda 1.0\n"
        "(\n"
        ")\n"
        "\n"
        'def Scope "Render"\n'
        "{\n"
        f"{products_block}\n"
        "\n"
        '    def Scope "Vars"\n'
        "    {\n"
        f"{vars_block}"
        "    }\n"
        "}\n"
    )

    return usda, product_paths


def _map_sensor_to_render_var(sensor_name: str) -> str | None:
    """Map a WU sensor name to an ovrtx render variable name.

    Args:
        sensor_name: WU sensor name (e.g., "depth", "normal").

    Returns:
        OvRTX render variable name, or None if no mapping exists.

    Examples:
        >>> _map_sensor_to_render_var("depth")
        'Depth'
        >>> _map_sensor_to_render_var("unknown")
    """
    return _SENSOR_TO_RENDER_VAR.get(sensor_name)


# ---------------------------------------------------------------------------
# Isolated ovrtx venv management
# ---------------------------------------------------------------------------

# Pre-built ovrtx from NVIDIA PyPI (includes native libovrtx-dynamic.so).
_OVRTX_VERSION = "0.3.0.312915"
# This PEP 751 lock is the only install source for the isolated worker. Its
# exact wheel URLs and SHA-256 digests keep provisioning independent of package
# index state and of the main application's OpenUSD environment.
_OVRTX_RUNTIME_LOCK_FILE = Path(__file__).with_name("pylock.ovrtx-runtime.toml")
_OVRTX_BUNDLED_PYTHON_LIBRARY_GLOB = "libpython*.so*"
_OVRTX_PROBE_PREFIX = "WU_OVRTX_VERSION="


def _ovrtx_venv_python_path(venv_dir: Path) -> Path:
    """Return the platform-specific Python executable path for a venv."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ovrtx_venv_dir_from_python_path(
    python_path: str, fallback_venv_dir: Path | None = None
) -> Path:
    """Infer the ovrtx runtime root from its Python executable path."""
    path_type = type(fallback_venv_dir or _OVRTX_VENV_DIR)
    executable = path_type(python_path)
    if fallback_venv_dir is not None and _path_is_relative_to(
        executable, fallback_venv_dir
    ):
        return fallback_venv_dir
    if executable.parent.name.lower() in {"bin", "scripts"}:
        return executable.parent.parent
    return fallback_venv_dir or _OVRTX_VENV_DIR


def _is_standard_ovrtx_python_path(python_path: str) -> bool:
    """True when a Python path has a venv executable layout we created."""
    executable = type(_OVRTX_VENV_DIR)(python_path)
    if executable.parent.name.lower() not in {"bin", "scripts"}:
        return False
    return (executable.parent.parent / "pyvenv.cfg").exists()


def _ovrtx_runtime_cache_key(venv_dir: Path) -> Path:
    """Return a stable cache key for an ovrtx runtime directory."""
    return venv_dir.expanduser().resolve(strict=False)


def _ovrtx_provision_lock_path(venv_dir: Path) -> Path:
    """Return the cross-process provisioning lock path for a venv."""
    resolved = venv_dir.expanduser().resolve(strict=False)
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    lock_root_value = os.environ.get("WU_OVRTX_LOCK_DIR")
    lock_name = f"{venv_dir.name}-{digest}.lock"
    if lock_root_value:
        return type(venv_dir)(lock_root_value) / lock_name
    return venv_dir.parent / f".{lock_name}"


def _ovrtx_runtime_lock_path(venv_dir: Path) -> Path:
    """Return the process lock file path for an ovrtx runtime directory."""
    return _ovrtx_provision_lock_path(venv_dir)


def _ovrtx_target_fallback_site_dir(venv_dir: Path) -> Path:
    """Return the exact pip --target fallback dir, not a normal venv site-dir."""
    return venv_dir / "lib" / "python" / "site-packages"


def _ovrtx_runtime_lock_args(lock_file: Path | None = None) -> list[str]:
    """Return fail-closed ``uv pip install`` args for the reviewed lock."""
    lock_file = lock_file or _OVRTX_RUNTIME_LOCK_FILE
    return [
        "--require-hashes",
        "--no-deps",
        "-r",
        str(lock_file),
        "--no-config",
        "--no-sources",
    ]


def _ovrtx_runtime_lock_digest() -> str:
    """Return the SHA-256 identity of the complete managed runtime lock."""
    return hashlib.sha256(_OVRTX_RUNTIME_LOCK_FILE.read_bytes()).hexdigest()


def _ovrtx_auto_provision_enabled() -> bool:
    """Return whether runtime OVRTX venv creation/recreation is allowed."""
    value = os.environ.get("WU_OVRTX_AUTO_PROVISION", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _clear_ovrtx_runtime_state() -> None:
    """Clear cached OVRTX subprocess state after a rejected install."""
    global _ovrtx_python
    _ovrtx_python = None


def _cached_ovrtx_python_ready(python_path: str, venv_dir: Path) -> bool:
    """Return True when a cached runtime matches the complete reviewed lock."""
    if not os.path.exists(python_path):
        return False
    cache_key = _ovrtx_runtime_cache_key(venv_dir)
    verified_key = (cache_key, _ovrtx_runtime_lock_digest())
    is_verified = verified_key in _verified_ovrtx_python_cache
    was_managed = cache_key in _verified_managed_ovrtx_python_cache
    is_managed = _is_managed_ovrtx_runtime_dir(venv_dir)
    if is_managed:
        # Once ownership is observed, keep it stable across marker races so a
        # managed runtime cannot fall back to the unmanaged trust path.
        _verified_managed_ovrtx_python_cache.add(cache_key)
    if is_verified and not was_managed and not is_managed:
        return True
    if not _ovrtx_managed_marker_matches_runtime_lock(venv_dir):
        return False
    if is_verified:
        _verified_managed_ovrtx_python_cache.add(cache_key)
        return True
    return not _ovrtx_bundled_python_libraries(venv_dir)


def _read_ovrtx_managed_marker(marker_path: Path) -> dict[str, str]:
    """Return line-based key/value fields from a managed runtime marker."""
    try:
        marker = marker_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    fields: dict[str, str] = {}
    for line in marker.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() and value.strip():
            fields[key.strip()] = value.strip()
    return fields


def _read_ovrtx_managed_marker_version(marker_path: Path) -> str | None:
    """Return the version recorded in a managed runtime marker, if present."""
    return _read_ovrtx_managed_marker(marker_path).get("ovrtx_version")


def _ovrtx_managed_marker_matches_runtime_lock(
    venv_dir: Path, runtime_lock_digest: str | None = None
) -> bool:
    """Return whether a managed marker identifies the complete current lock."""
    expanded_venv_dir = venv_dir.expanduser()
    if (expanded_venv_dir / _OVRTX_PROVISIONING_MARKER).exists():
        return False
    if runtime_lock_digest is None:
        runtime_lock_digest = _ovrtx_runtime_lock_digest()
    marker = _read_ovrtx_managed_marker(expanded_venv_dir / _OVRTX_MANAGED_MARKER)
    return (
        marker.get("ovrtx_version") == _OVRTX_VERSION
        and marker.get("runtime_lock_sha256") == runtime_lock_digest
    )


def _remember_verified_ovrtx_python(
    cache_key: Path,
    python_path: str,
    runtime_lock_digest: str | None = None,
) -> None:
    """Record a runtime that was validated or completed by this process."""
    _ovrtx_python_cache[cache_key] = python_path
    verified_key = (
        cache_key,
        runtime_lock_digest or _ovrtx_runtime_lock_digest(),
    )
    _verified_ovrtx_python_cache.add(verified_key)
    if _is_managed_ovrtx_runtime_dir(cache_key):
        _verified_managed_ovrtx_python_cache.add(cache_key)
    else:
        _verified_managed_ovrtx_python_cache.discard(cache_key)


def _write_ovrtx_provisioning_marker(venv_dir: Path) -> None:
    """Mark a runtime as safe to recreate if provisioning is interrupted."""
    (venv_dir / _OVRTX_PROVISIONING_MARKER).write_text(
        "Provisioning by world_understanding.functions.graphics.render_ovrtx\n"
    )


def _write_ovrtx_managed_marker(
    venv_dir: Path, runtime_lock_digest: str | None = None
) -> None:
    """Atomically mark the managed runtime ready for cache fast-path reads."""
    runtime_lock_digest = runtime_lock_digest or _ovrtx_runtime_lock_digest()
    marker_path = venv_dir / _OVRTX_MANAGED_MARKER
    marker_tmp = venv_dir / f"{_OVRTX_MANAGED_MARKER}.tmp.{os.getpid()}"
    marker_tmp.write_text(
        "Created by world_understanding.functions.graphics.render_ovrtx\n"
        f"ovrtx_version={_OVRTX_VERSION}\n"
        f"runtime_lock_sha256={runtime_lock_digest}\n",
        encoding="utf-8",
    )
    os.replace(marker_tmp, marker_path)
    try:
        (venv_dir / _OVRTX_PROVISIONING_MARKER).unlink()
    except FileNotFoundError:
        pass


def _try_refresh_ovrtx_managed_marker(venv_dir: Path, runtime_lock_digest: str) -> None:
    """Best-effort marker refresh for already verified managed runtimes."""
    try:
        _write_ovrtx_managed_marker(venv_dir, runtime_lock_digest)
    except OSError as exc:
        logger.info(
            "Could not refresh OVRTX managed marker at %s after version match; "
            "continuing with verified runtime: %s",
            venv_dir / _OVRTX_MANAGED_MARKER,
            exc,
        )


def _is_managed_ovrtx_runtime_dir(venv_dir: Path) -> bool:
    """Return True when it is safe for this module to recreate ``venv_dir``."""
    expanded_venv_dir = venv_dir.expanduser()
    resolved = _ovrtx_runtime_cache_key(expanded_venv_dir)
    default_resolved = _ovrtx_runtime_cache_key(_DEFAULT_OVRTX_VENV_DIR)
    return (
        resolved == default_resolved
        or resolved in _verified_managed_ovrtx_python_cache
        or (expanded_venv_dir / _OVRTX_MANAGED_MARKER).exists()
        or (expanded_venv_dir / _OVRTX_PROVISIONING_MARKER).exists()
    )


def _probe_existing_ovrtx_python_before_lock(
    python_path: Path, venv_dir: Path, cache_key: Path
) -> str | None:
    """Return an already-ready runtime without taking the provisioning lock."""
    python_str = str(python_path)
    if _cached_ovrtx_python_ready(python_str, venv_dir):
        _remember_verified_ovrtx_python(cache_key, python_str)
        return python_str
    if not python_path.exists() or _is_managed_ovrtx_runtime_dir(venv_dir):
        return None
    try:
        if _ovrtx_import_probe_succeeds(python_path, venv_dir):
            _remember_verified_ovrtx_python(cache_key, python_str)
            return python_str
    except subprocess.TimeoutExpired:
        logger.warning("ovrtx import probe timed out for existing unmanaged runtime")
    except OSError as exc:
        logger.warning("ovrtx import probe could not launch: %s", exc)
    return None


def _remove_ovrtx_venv(venv_dir: Path) -> None:
    """Clear cached state and remove a rejected or partial OVRTX environment."""
    _clear_ovrtx_runtime_state()
    cache_key = _ovrtx_runtime_cache_key(venv_dir)
    was_managed = _is_managed_ovrtx_runtime_dir(venv_dir)
    _ovrtx_python_cache.pop(cache_key, None)
    _verified_ovrtx_python_cache.difference_update(
        {key for key in _verified_ovrtx_python_cache if key[0] == cache_key}
    )
    shutil.rmtree(venv_dir, ignore_errors=True)
    if venv_dir.exists() or venv_dir.is_symlink():
        if was_managed:
            _verified_managed_ovrtx_python_cache.add(cache_key)
            try:
                _write_ovrtx_provisioning_marker(venv_dir)
            except OSError as exc:
                logger.warning(
                    "Could not persist the failed-removal marker at %s: %s",
                    venv_dir,
                    exc,
                )
        raise RuntimeError(
            f"OVRTX runtime at {venv_dir} could not be completely removed; "
            "refusing to provision into a partial environment"
        )
    _verified_managed_ovrtx_python_cache.discard(cache_key)


def _parse_ovrtx_probe_stdout(stdout: str) -> str | None:
    """Extract the version line from the isolated import probe."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(_OVRTX_PROBE_PREFIX):
            return line[len(_OVRTX_PROBE_PREFIX) :].strip() or None
    return None


def _probe_ovrtx_version(python_path: Path, venv_dir: Path) -> str | None:
    """Return the installed ovrtx distribution version, if importable."""
    env = _ovrtx_subprocess_env()
    env.pop("PYTHONPATH", None)

    with tempfile.NamedTemporaryFile(
        prefix="wu_ovrtx_probe_", suffix=".txt", delete=False
    ) as version_file:
        version_path = Path(version_file.name)

    try:
        probe = subprocess.run(
            [
                str(python_path),
                "-c",
                (
                    "import sys\n"
                    "from importlib import metadata\n"
                    "import ovrtx\n"
                    "_version = metadata.version('ovrtx')\n"
                    "with open(sys.argv[1], 'w', encoding='utf-8') as _f:\n"
                    f"    _f.write({_OVRTX_PROBE_PREFIX!r} + _version + '\\n')\n"
                ),
                str(version_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
        if probe.returncode != 0:
            logger.warning(
                "ovrtx import probe failed for %s in %s (exit %s): %s",
                python_path,
                venv_dir,
                probe.returncode,
                probe.stderr[-500:],
            )
            return None

        if version_path.exists():
            version = _parse_ovrtx_probe_stdout(
                version_path.read_text(encoding="utf-8")
            )
            if version:
                return version
        return _parse_ovrtx_probe_stdout(probe.stdout) or _OVRTX_VERSION
    finally:
        version_path.unlink(missing_ok=True)


def _cached_ovrtx_python_matches(python_path: Path, venv_dir: Path) -> bool:
    """Return whether cached OVRTX state still points at the target build."""
    global _ovrtx_python
    if not python_path.exists():
        return False
    try:
        if _probe_ovrtx_version(python_path, venv_dir) == _OVRTX_VERSION:
            _ovrtx_python = str(python_path)
            return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    _clear_ovrtx_runtime_state()
    return False


def _get_ovrtx_python(venv_dir: Path | None = None) -> str:
    """Return the ovrtx Python path, serializing runtime provisioning."""
    global _ovrtx_python
    venv_dir = (venv_dir or _OVRTX_VENV_DIR).expanduser()
    cache_key = _ovrtx_runtime_cache_key(venv_dir)
    python_path = _ovrtx_venv_python_path(venv_dir)
    python_str = str(python_path)

    cached_python = _ovrtx_python_cache.get(cache_key)
    if cached_python and _cached_ovrtx_python_ready(cached_python, venv_dir):
        return cached_python

    if _cached_ovrtx_python_ready(python_str, venv_dir):
        _remember_verified_ovrtx_python(cache_key, python_str)
        return python_str

    if _ovrtx_python is not None:
        cached_venv_dir = _ovrtx_venv_dir_from_python_path(_ovrtx_python)
        if cached_venv_dir == venv_dir and _cached_ovrtx_python_ready(
            _ovrtx_python, venv_dir
        ):
            _remember_verified_ovrtx_python(cache_key, _ovrtx_python)
            return _ovrtx_python

    if not _ovrtx_auto_provision_enabled():
        return _get_ovrtx_python_unlocked(venv_dir)

    lock_path = _ovrtx_provision_lock_path(venv_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock_path), timeout=_OVRTX_PROVISION_LOCK_TIMEOUT_S):
            return _get_ovrtx_python_unlocked(venv_dir)
    except Timeout as exc:
        raise RuntimeError(
            f"Timed out waiting for OVRTX runtime provisioning lock: {lock_path}"
        ) from exc


def _ovrtx_import_probe_succeeds(python_path: Path, venv_dir: Path) -> bool:
    """Return True when ``python_path`` can import ovrtx."""
    probe = subprocess.run(
        [str(python_path), "-c", "import ovrtx"],
        capture_output=True,
        timeout=30,
        check=False,
        env=_ovrtx_subprocess_env(),
    )
    if probe.returncode == 0:
        return True

    try:
        site_dir = _ovrtx_site_packages_dir(venv_dir)
    except RuntimeError as exc:
        logger.debug("No local ovrtx site-package fallback for import probe: %s", exc)
        return False

    probe_code = f"import sys; sys.path.insert(0, {str(site_dir)!r}); import ovrtx"
    probe = subprocess.run(
        [str(python_path), "-c", probe_code],
        capture_output=True,
        timeout=30,
        check=False,
        env=_ovrtx_subprocess_env(),
    )
    return probe.returncode == 0


def _get_ovrtx_python_unlocked(venv_dir: Path | None = None) -> str:
    """Return the path to the Python executable in the ovrtx venv.

    If the venv does not exist, it is created and ovrtx + dependencies
    are installed into it. The venv intentionally does NOT have another
    ``pxr`` provider to avoid native library conflicts.

    Args:
        venv_dir: Override directory for the venv. Defaults to
            ``~/.cache/wu/ovrtx_venv``.

    Returns:
        Absolute path to the venv's python executable.

    Raises:
        RuntimeError: If venv creation or package installation fails.
    """
    global _ovrtx_python
    venv_dir = (venv_dir or _OVRTX_VENV_DIR).expanduser()
    cache_key = _ovrtx_runtime_cache_key(venv_dir)
    cached_python = _ovrtx_python_cache.get(cache_key)
    if cached_python and _cached_ovrtx_python_ready(cached_python, venv_dir):
        _remember_verified_ovrtx_python(cache_key, cached_python)
        return cached_python

    python_path = _ovrtx_venv_python_path(venv_dir)
    python_str = str(python_path)

    if _cached_ovrtx_python_ready(python_str, venv_dir):
        _ovrtx_python = python_str
        _remember_verified_ovrtx_python(cache_key, python_str)
        return python_str

    if _ovrtx_python is not None and (
        _is_standard_ovrtx_python_path(_ovrtx_python)
        or type(venv_dir)(_ovrtx_python) == _ovrtx_venv_python_path(venv_dir)
    ):
        cached_venv_dir = _ovrtx_venv_dir_from_python_path(_ovrtx_python)
        if _path_is_relative_to(type(venv_dir)(_ovrtx_python), venv_dir):
            cached_venv_dir = venv_dir
        if cached_venv_dir == venv_dir:
            if _cached_ovrtx_python_ready(_ovrtx_python, venv_dir):
                _remember_verified_ovrtx_python(cache_key, _ovrtx_python)
                return _ovrtx_python

    if python_path.exists():
        managed_runtime = _is_managed_ovrtx_runtime_dir(venv_dir)
        if managed_runtime:
            # Preserve ownership even if the marker is removed while this
            # validation pass is in flight. Removal and later validation must
            # not reinterpret a runtime we already recognized as managed.
            _verified_managed_ovrtx_python_cache.add(cache_key)
        validated_runtime_lock_digest = (
            _ovrtx_runtime_lock_digest() if managed_runtime else None
        )
        managed_marker_matches = bool(
            managed_runtime
            and validated_runtime_lock_digest is not None
            and _ovrtx_managed_marker_matches_runtime_lock(
                venv_dir, validated_runtime_lock_digest
            )
        )
        if managed_runtime and not managed_marker_matches:
            if not _ovrtx_auto_provision_enabled():
                _clear_ovrtx_runtime_state()
                raise RuntimeError(
                    "Existing managed ovrtx venv at "
                    f"{venv_dir} does not match the current runtime lock, and "
                    "WU_OVRTX_AUTO_PROVISION is disabled"
                )
            logger.warning(
                "Existing managed ovrtx venv does not match the current runtime "
                "lock; recreating: %s",
                venv_dir,
            )
        else:
            # Unmanaged runtimes cannot carry our lock marker, so retain the
            # historical exact-OVRTX-version probe without deleting them solely
            # because their dependency set is externally managed.
            try:
                version = _probe_ovrtx_version(python_path, venv_dir)
                if version == _OVRTX_VERSION:
                    if not managed_runtime:
                        _ovrtx_python = str(python_path)
                        _remember_verified_ovrtx_python(cache_key, _ovrtx_python)
                        return _ovrtx_python

                    assert validated_runtime_lock_digest is not None
                    runtime_identity_changed = (
                        _ovrtx_runtime_lock_digest() != validated_runtime_lock_digest
                        or not _ovrtx_managed_marker_matches_runtime_lock(
                            venv_dir, validated_runtime_lock_digest
                        )
                    )
                    if not runtime_identity_changed:
                        _remove_ovrtx_bundled_python_libraries(venv_dir)
                        runtime_identity_changed = (
                            _ovrtx_runtime_lock_digest()
                            != validated_runtime_lock_digest
                            or not _ovrtx_managed_marker_matches_runtime_lock(
                                venv_dir, validated_runtime_lock_digest
                            )
                        )
                    if not runtime_identity_changed:
                        _try_refresh_ovrtx_managed_marker(
                            venv_dir, validated_runtime_lock_digest
                        )
                        runtime_identity_changed = (
                            _ovrtx_runtime_lock_digest()
                            != validated_runtime_lock_digest
                            or not _ovrtx_managed_marker_matches_runtime_lock(
                                venv_dir, validated_runtime_lock_digest
                            )
                        )

                    if runtime_identity_changed:
                        if not _ovrtx_auto_provision_enabled():
                            _clear_ovrtx_runtime_state()
                            raise RuntimeError(
                                "Existing managed ovrtx venv runtime identity "
                                f"changed while validating {venv_dir}, and "
                                "WU_OVRTX_AUTO_PROVISION is disabled"
                            )
                        logger.warning(
                            "OVRTX runtime identity changed while validating an "
                            "existing managed venv; recreating: %s",
                            venv_dir,
                        )
                    else:
                        _ovrtx_python = str(python_path)
                        _remember_verified_ovrtx_python(
                            cache_key,
                            _ovrtx_python,
                            validated_runtime_lock_digest,
                        )
                        return _ovrtx_python
                else:
                    if not _ovrtx_auto_provision_enabled():
                        _clear_ovrtx_runtime_state()
                        raise RuntimeError(
                            "Existing ovrtx venv at "
                            f"{venv_dir} has version {version!r}, expected "
                            f"{_OVRTX_VERSION!r}, and "
                            "WU_OVRTX_AUTO_PROVISION is disabled"
                        )
                    logger.warning(
                        "Existing ovrtx venv has version %r, expected %s; "
                        "recreating: %s",
                        version,
                        _OVRTX_VERSION,
                        venv_dir,
                    )
            except subprocess.TimeoutExpired as exc:
                _clear_ovrtx_runtime_state()
                if not managed_runtime:
                    raise RuntimeError(
                        "Existing unmanaged ovrtx runtime import probe timed out "
                        f"at {venv_dir}; refusing to replace it"
                    ) from exc
                if not _ovrtx_auto_provision_enabled():
                    raise RuntimeError(
                        "Existing ovrtx venv import probe timed out at "
                        f"{venv_dir} and WU_OVRTX_AUTO_PROVISION is disabled"
                    ) from exc
                logger.warning("ovrtx import probe timed out, recreating venv")
            except OSError as exc:
                _clear_ovrtx_runtime_state()
                if not managed_runtime:
                    raise RuntimeError(
                        "Existing unmanaged ovrtx runtime import probe could not "
                        f"launch at {venv_dir}; refusing to replace it"
                    ) from exc
                if not _ovrtx_auto_provision_enabled():
                    raise RuntimeError(
                        "Existing managed ovrtx runtime import probe could not "
                        "launch and WU_OVRTX_AUTO_PROVISION is disabled"
                    ) from exc
                logger.warning("ovrtx import probe could not launch: %s", exc)

        _remove_ovrtx_venv(venv_dir)
    elif python_path.is_symlink():
        if not _is_managed_ovrtx_runtime_dir(venv_dir):
            raise RuntimeError(
                "Existing OVRTX Python path is a broken symlink, but its "
                "runtime is not managed by world-understanding and will not "
                f"be deleted: {python_path}. Fix the symlink target or use a "
                "managed WU_OVRTX_VENV_DIR cache path."
            )
        logger.warning(
            "Existing ovrtx Python symlink is broken, recreating: %s", python_path
        )
        _remove_ovrtx_venv(venv_dir)
    elif (
        venv_dir.exists()
        and _is_managed_ovrtx_runtime_dir(venv_dir)
        and _ovrtx_auto_provision_enabled()
    ):
        logger.warning(
            "Existing managed ovrtx runtime is incomplete; recreating: %s",
            venv_dir,
        )
        _remove_ovrtx_venv(venv_dir)

    if not _ovrtx_auto_provision_enabled():
        _clear_ovrtx_runtime_state()
        raise RuntimeError(
            f"OvRTX venv not found at {venv_dir} and "
            "WU_OVRTX_AUTO_PROVISION is disabled"
        )

    # Create the venv
    logger.info("Creating isolated ovrtx venv at %s", venv_dir)
    venv_dir.mkdir(parents=True, exist_ok=True)
    _write_ovrtx_provisioning_marker(venv_dir)

    # Try uv first, fall back to stdlib venv.
    # shutil.which may miss uv inside venvs, so also check next to sys.executable.
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        for _name in ("uv.exe", "uv"):
            _candidate = os.path.join(os.path.dirname(sys.executable), _name)
            if os.path.exists(_candidate):
                uv_bin = _candidate
                break
    if uv_bin is None:
        _remove_ovrtx_venv(venv_dir)
        raise RuntimeError(
            "OVRTX auto-provisioning requires the uv executable so the "
            f"hash-locked PEP 751 profile can be enforced: {_OVRTX_RUNTIME_LOCK_FILE}"
        )

    lock_snapshot_path: Path | None = None
    provisioned_lock_digest = ""
    try:
        runtime_lock = _OVRTX_RUNTIME_LOCK_FILE.read_bytes()
        provisioned_lock_digest = hashlib.sha256(runtime_lock).hexdigest()
        with tempfile.NamedTemporaryFile(
            prefix="pylock.wu-ovrtx-runtime-",
            suffix=".toml",
            delete=False,
        ) as lock_snapshot:
            lock_snapshot.write(runtime_lock)
            lock_snapshot_path = type(_OVRTX_RUNTIME_LOCK_FILE)(lock_snapshot.name)
        _run_checked(
            [
                uv_bin,
                "venv",
                str(venv_dir),
                "--allow-existing",
                "--python",
                sys.executable,
            ],
            "uv venv creation",
        )
        _run_checked(
            [
                uv_bin,
                "pip",
                "install",
                "--python",
                str(python_path),
                *_ovrtx_runtime_lock_args(lock_snapshot_path),
            ],
            "locked OVRTX runtime install",
        )
    except Exception:
        _remove_ovrtx_venv(venv_dir)
        raise
    finally:
        if lock_snapshot_path is not None:
            lock_snapshot_path.unlink(missing_ok=True)

    if not python_path.exists():
        raise RuntimeError(f"Failed to create ovrtx venv at {venv_dir}")
    _remove_ovrtx_bundled_python_libraries(venv_dir)

    # Symlink MaterialX standard data libraries so MaterialX shaders
    # (ND_tiledimage, OpenPBR, etc.) resolve correctly.  ovrtx ships
    # them under ovrtx/bin/library/ but looks for them at library/.
    for d in venv_dir.rglob("ovrtx/bin/library"):
        expected = d.parent.parent.parent / "library"
        if not expected.exists():
            try:
                expected.symlink_to(d, target_is_directory=True)
                logger.info("Created MaterialX library symlink: %s -> %s", expected, d)
            except OSError:  # pragma: no cover - Windows symlink fallback
                if os.name != "nt":
                    raise
                shutil.copytree(d, expected, dirs_exist_ok=True)
                logger.info("Copied MaterialX library: %s -> %s", d, expected)
        break

    try:
        version = _probe_ovrtx_version(python_path, venv_dir)
    except subprocess.TimeoutExpired as exc:
        _remove_ovrtx_venv(venv_dir)
        raise RuntimeError("Installed ovrtx import probe timed out") from exc
    if version != _OVRTX_VERSION:
        _clear_ovrtx_runtime_state()
        if version is None:
            raise RuntimeError(
                "Installed ovrtx import probe failed; retained environment at "
                f"{venv_dir} for debugging"
            )
        _remove_ovrtx_venv(venv_dir)
        raise RuntimeError(
            "Installed ovrtx version "
            f"{version!r} does not match expected {_OVRTX_VERSION!r}"
        )

    if _ovrtx_runtime_lock_digest() != provisioned_lock_digest:
        _remove_ovrtx_venv(venv_dir)
        raise RuntimeError(
            "OVRTX runtime lock changed during provisioning; retry with the "
            "current lock"
        )

    _write_ovrtx_managed_marker(venv_dir, provisioned_lock_digest)
    if _ovrtx_runtime_lock_digest() != provisioned_lock_digest:
        _remove_ovrtx_venv(venv_dir)
        raise RuntimeError(
            "OVRTX runtime lock changed while marking the provisioned runtime; "
            "retry with the current lock"
        )

    _ovrtx_python = str(python_path)
    _remember_verified_ovrtx_python(cache_key, _ovrtx_python, provisioned_lock_digest)
    logger.info("OvRTX venv ready: %s", _ovrtx_python)
    return _ovrtx_python


def _run_checked(cmd: list[str], label: str) -> None:
    """Run a command and raise RuntimeError on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}): {result.stderr[-500:]}"
        )


def _copy_exported_relative_assets(
    stage: "Usd.Stage",
    export_dir: Path,
    base_dir: str | Path | None = None,
    exported_stage_path: str | Path | None = None,
) -> int:
    """Copy local texture assets next to an exported render stage.

    ``render_all_cameras`` exports the caller's stage into an OVRTX IPC temp
    directory before the isolated daemon opens it. Texture paths in that
    exported layer are interpreted from the new temp directory, not the
    original stage directory, so extracted ZIP/USDZ bundles and prepared stages
    with host-absolute texture refs lose their textures unless we mirror those
    files here and rewrite the exported layer to the mirrored paths.
    """
    from pxr import Sdf

    from world_understanding.utils.usd.material import get_local_texture_file_assets

    if base_dir is not None:
        resolved_base_dir = Path(base_dir)
    else:
        root_layer = stage.GetRootLayer()
        if root_layer.realPath:
            resolved_base_dir = Path(root_layer.realPath).parent
        else:
            resolved_base_dir = Path.cwd()

    relative_source_by_path: dict[str, Path] = {}

    def digest_relative_path(source: Path) -> Path:
        digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
        return Path("textures") / f"{source.stem}_{digest}{source.suffix}"

    def localized_relative_path(asset_path: str, source: Path) -> Path:
        resolved_source = source.resolve()
        authored_path = _local_asset_path(asset_path)
        if not authored_path.is_absolute():
            clean_parts = [
                part for part in authored_path.parts if part not in ("", ".")
            ]
            if clean_parts and ".." not in clean_parts:
                candidate = Path(*clean_parts)
                candidate_key = candidate.as_posix()
                existing_source = relative_source_by_path.get(candidate_key)
                if existing_source is None or existing_source == resolved_source:
                    relative_source_by_path[candidate_key] = resolved_source
                    return candidate

        candidate = digest_relative_path(source)
        relative_source_by_path[candidate.as_posix()] = resolved_source
        return candidate

    def resolved_texture_path(asset_path: str) -> str | None:
        if not asset_path or _is_remote_asset_path(asset_path):  # pragma: no cover
            return None
        path = _local_asset_path(asset_path)
        if not path.is_absolute():
            path = resolved_base_dir / path
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            return None
        return str(resolved) if resolved.is_file() else None

    copied = 0
    localized_by_resolved: dict[str, str] = {}
    localized_by_attr: dict[tuple[str, str], str] = {}
    for asset in get_local_texture_file_assets(stage, base_dir=resolved_base_dir):
        if not asset.get("is_local") or not asset.get("resolved_path"):
            continue

        asset_path = str(asset.get("file_path", ""))
        if not asset_path or _is_remote_asset_path(asset_path):
            continue

        source = Path(str(asset["resolved_path"]))
        if not source.is_file():
            continue

        relative_path = localized_relative_path(asset_path, source)
        destination = export_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != source.resolve():
            shutil.copy2(source, destination)
            copied += 1

        rel_path_string = relative_path.as_posix()
        resolved_source = str(source.resolve())
        localized_by_resolved[resolved_source] = rel_path_string
        localized_by_attr[(str(asset["prim_path"]), str(asset["attr_name"]))] = (
            rel_path_string
        )

    if exported_stage_path is None or not localized_by_resolved:
        return copied

    layer = Sdf.Layer.FindOrOpen(str(exported_stage_path))
    if layer is None:
        logger.warning(
            "Could not reopen exported OVRTX stage for texture path localization: %s",
            exported_stage_path,
        )
        return copied

    updated = 0

    def update_prim_spec(prim_spec: Any) -> None:
        nonlocal updated
        prim_path = str(prim_spec.path)
        for attr_name in list(prim_spec.attributes.keys()):
            attr_spec = prim_spec.attributes[attr_name]
            value = attr_spec.default
            if not isinstance(value, Sdf.AssetPath):
                continue
            asset_path = value.path if hasattr(value, "path") else str(value)
            if not asset_path or _is_remote_asset_path(asset_path):
                continue

            new_path = localized_by_attr.get((prim_path, attr_name))
            if new_path is None:
                resolved = resolved_texture_path(asset_path)
                new_path = (
                    localized_by_resolved.get(resolved)
                    if resolved is not None
                    else None
                )
            if new_path is None or asset_path == new_path:
                continue

            attr_spec.default = Sdf.AssetPath(new_path)
            updated += 1

        for child in prim_spec.nameChildren:
            update_prim_spec(child)

    for prim in layer.rootPrims:
        update_prim_spec(prim)

    if updated:
        layer.Save()
        logger.info(
            "Localized %d exported OVRTX texture asset path(s) to render temp paths",
            updated,
        )

    return copied


# ---------------------------------------------------------------------------
# Subprocess worker script (executed in the ovrtx venv without pxr)
#
# This follows the same pattern as the ovrtx reference example
# (render_to_disk.py):
#   1. ovrtx.Renderer() — plain constructor, no config
#   2. renderer.open_usd(path) — load scene with render products sublayered
#   3. Multiple step() calls for path-tracer convergence
#   4. np.from_dlpack(var) to extract pixels
#   5. Image.fromarray(pixels) to save
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = r'''
"""OvRTX subprocess worker — runs in the isolated ovrtx venv."""
import json, os, sys

# Compat: ovrtx >=110.1.0.273788 changed map(device=) from str to enum.
def _cpu_device():
    import ovrtx
    return getattr(ovrtx, "Device", None) and ovrtx.Device.CPU or "cpu"


def main():
    params = json.loads(sys.argv[1])
    usd_path = params["usd_path"]
    fps = params.get("fps", 24.0)
    cameras = params["cameras"]
    frames = params["frames"]
    sensors = params["sensors"]
    output_dir = params["output_dir"]
    product_paths = params["product_paths"]
    # ``num_sensor_updates`` drives the progressive
    # ``renderer.step(delta_time=0)`` accumulation loop below. This is the
    # quality knob validated on 0.2.0; keep using it until 0.3 GPU validation
    # proves the ``samplesPerPixel`` / ``accumulationLimit`` schema attributes
    # affect convergence.
    num_sensor_updates = params.get("num_sensor_updates", 1)
    visibility_schedule = params.get("visibility_schedule", {})
    visibility_updates = params.get("visibility_updates")
    frame_usd_paths = params.get("frame_usd_paths", {})
    material_target = params.get("material_target", "auto")
    log_level = params.get("log_level", "warn")

    import logging
    _LOG_MAP = {"error": logging.ERROR, "warn": logging.WARNING,
                "info": logging.INFO, "debug": logging.DEBUG}
    logging.basicConfig(level=_LOG_MAP.get(log_level, logging.WARNING))
    logging.debug("Render material target: %s", material_target)

    import ovrtx
    import numpy as np
    from PIL import Image

    SENSOR_MAP = {"depth": "Depth", "normal": "Normal", "albedo": "Albedo"}

    all_product_paths = set(product_paths)

    cam_data = [
        {"images": [], "sensor_files": {s: {} for s in sensors}}
        for _ in cameras
    ]

    renderer = ovrtx.Renderer()
    if not frame_usd_paths:
        renderer.open_usd(usd_path)

    for frame_num in frames:
        frame_usd_path = frame_usd_paths.get(str(frame_num))
        if frame_usd_path:
            renderer.reset_stage()
            renderer.open_usd(frame_usd_path)

        renderer.update_from_usd_time(float(frame_num) / fps)
        # Reset the progressive accumulator after selecting frame time but
        # before writing per-frame overrides. Visibility writes must be the
        # final scene-state change before step().
        renderer.reset()

        # Legacy fallback for direct worker callers that pass visibility
        # updates without frame-specific overlay USDs.
        frame_key = str(float(frame_num))
        vis_map = None
        if frame_usd_path:
            vis_map = None
        elif visibility_updates is not None:
            vis_map = visibility_updates.get(frame_key)
        elif frame_key in visibility_schedule:
            vis_map = visibility_schedule[frame_key]
        if vis_map:
            for prim_path, vis_value in vis_map.items():
                token = "inherited" if vis_value == "inherited" else "invisible"
                renderer.write_attribute([prim_path], "visibility", [token])

        # Progressive path-tracer accumulation: step() with delta_time=0
        # keeps simulation time fixed so OVRtx's accumulator layers more
        # samples onto the same frame. Convergence plateaus near
        # ~500 iterations on the kit golden scene (PSNR climbs ~12 dB
        # over 1→100, another ~1.3 dB over 100→500, flat past there).
        # See the convergence sweep in /tmp/ovrtx_cap.py.
        all_products = None
        for update_idx in range(num_sensor_updates):
            # Historical 0.2.0 guard: visibility write_attribute changes could
            # crash when followed only by dt=0 accumulation steps. A single
            # nonzero step lets the renderer consume the scene-state change;
            # subsequent steps keep accumulating the selected USD frame. Keep
            # this fallback guarded until native 0.3 visibility is validated.
            delta_time = (1.0 / 60.0) if vis_map and update_idx == 0 else 0.0
            all_products = renderer.step(
                render_products=all_product_paths,
                delta_time=delta_time,
            )

        if all_products:
            for cam_idx, product_path in enumerate(product_paths):
                if product_path not in all_products:
                    continue
                product = all_products[product_path]
                for frame in product.frames:
                    if "LdrColor" in frame.render_vars:
                        with frame.render_vars["LdrColor"].map(device=_cpu_device()) as var:
                            pixels = np.from_dlpack(var).copy()
                        fname = f"cam{cam_idx}_f{frame_num}.png"
                        fpath = os.path.join(output_dir, fname)
                        Image.fromarray(pixels).save(fpath)
                        cam_data[cam_idx]["images"].append(fname)

                    for sname in sensors:
                        rv = SENSOR_MAP.get(sname)
                        if rv and rv in frame.render_vars:
                            with frame.render_vars[rv].map(device=_cpu_device()) as var:
                                sarr = np.from_dlpack(var).copy()
                            sfname = f"cam{cam_idx}_f{frame_num}_{sname}.npy"
                            np.save(os.path.join(output_dir, sfname), sarr)
                            cam_data[cam_idx]["sensor_files"][sname][frame_num] = sfname

    del renderer

    results = []
    for cam_idx, camera in enumerate(cameras):
        results.append({
            "camera": camera,
            "image_files": cam_data[cam_idx]["images"],
            "sensor_files": cam_data[cam_idx]["sensor_files"],
            "frame_count": len(cam_data[cam_idx]["images"]),
        })

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Persistent daemon script (executed in the ovrtx venv without pxr)
#
# Unlike _WORKER_SCRIPT which processes a single batch and exits, this
# daemon creates ovrtx.Renderer() once and loops on stdin reading JSON
# commands.  This amortises the ~5.5s GPU init across all render calls.
# ---------------------------------------------------------------------------

_DAEMON_SCRIPT = r'''
"""OvRTX persistent daemon — runs in the isolated ovrtx venv.

Creates Renderer() once, then loops on stdin for JSON-line commands.
Protocol:
  startup  → stdout: {"status": "ready"}
  request  → stdin:  {"command": "render", ...}
           ← stdout: {"status": "ok", "manifest": [...]}
  shutdown → stdin:  {"command": "shutdown"}
  error    ← stdout: {"status": "error", "error": "msg"}
"""
import json
import os
import sys
import traceback

# Compat: ovrtx >=110.1.0.273788 changed map(device=) from str to enum.
def _cpu_device():
    import ovrtx
    return getattr(ovrtx, "Device", None) and ovrtx.Device.CPU or "cpu"


def _redirect_native_stdout():
    """Redirect C-level fd 1 (stdout) to fd 2 (stderr).

    ovrtx / Vulkan may print init messages via the C runtime which bypass
    Python's sys.stdout and would corrupt our JSON protocol on the pipe.
    Returns the saved fd so it can be restored later.
    """
    saved_fd = os.dup(1)
    os.dup2(2, 1)  # fd 1 now points to stderr
    return saved_fd


def _restore_native_stdout(saved_fd):
    """Restore the original C-level stdout from a previously saved fd."""
    sys.stdout.flush()
    os.dup2(saved_fd, 1)
    os.close(saved_fd)


def main():
    # Configure Python logging from env var set by parent process
    import logging
    _LOG_MAP = {"error": logging.ERROR, "warn": logging.WARNING,
                "info": logging.INFO, "debug": logging.DEBUG}
    _lvl = os.environ.get("OVRTX_LOG_LEVEL", "warn")
    logging.basicConfig(level=_LOG_MAP.get(_lvl, logging.WARNING))

    # Redirect native stdout → stderr while importing ovrtx and creating
    # the Renderer, so Vulkan/driver messages don't corrupt our JSON pipe.
    saved_fd = _redirect_native_stdout()
    import ovrtx
    import numpy as np
    from PIL import Image

    SENSOR_MAP = {"depth": "Depth", "normal": "Normal", "albedo": "Albedo"}

    renderer = ovrtx.Renderer()
    _restore_native_stdout(saved_fd)

    # Signal readiness to parent
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(
                json.dumps({"status": "error", "error": f"bad JSON: {exc}"}) + "\n"
            )
            sys.stdout.flush()
            continue

        command = request.get("command")

        if command == "shutdown":
            break

        if command != "render":
            sys.stdout.write(
                json.dumps({"status": "error", "error": f"unknown command: {command}"}) + "\n"
            )
            sys.stdout.flush()
            continue

        try:
            usd_path = request["usd_path"]
            fps = request.get("fps", 24.0)
            cameras = request["cameras"]
            frames = request["frames"]
            sensors = request["sensors"]
            output_dir = request["output_dir"]
            product_paths = request["product_paths"]
            # num_sensor_updates drives the progressive
            # renderer.step(delta_time=0) accumulation loop below. This is the
            # quality knob validated on 0.2.0; keep it until 0.3 validation
            # proves RenderProduct SPP/accum USD attributes affect output.
            num_sensor_updates = request.get("num_sensor_updates", 1)
            visibility_schedule = request.get("visibility_schedule", {})
            visibility_updates = request.get("visibility_updates")
            frame_usd_paths = request.get("frame_usd_paths", {})
            material_target = request.get("material_target", "auto")
            logging.debug("Render material target: %s", material_target)

            all_product_paths = set(product_paths)

            cam_data = [
                {"images": [], "sensor_files": {s: {} for s in sensors}}
                for _ in cameras
            ]

            # Redirect native stdout during ovrtx calls so Vulkan log
            # messages don't corrupt the JSON protocol on the pipe.
            saved_fd = _redirect_native_stdout()
            try:
                if not frame_usd_paths:
                    renderer.reset_stage()
                    renderer.open_usd(usd_path)

                for frame_num in frames:
                    frame_usd_path = frame_usd_paths.get(str(frame_num))
                    if frame_usd_path:
                        renderer.reset_stage()
                        renderer.open_usd(frame_usd_path)

                    renderer.update_from_usd_time(float(frame_num) / fps)
                    # Reset the progressive accumulator after selecting
                    # frame time but before writing per-frame overrides.
                    # Visibility writes must be the final scene-state change
                    # before step().
                    renderer.reset()

                    # Legacy fallback for direct daemon callers that pass
                    # visibility updates without frame-specific overlay USDs.
                    frame_key = str(float(frame_num))
                    vis_map = None
                    if frame_usd_path:
                        vis_map = None
                    elif visibility_updates is not None:
                        vis_map = visibility_updates.get(frame_key)
                    elif frame_key in visibility_schedule:
                        vis_map = visibility_schedule[frame_key]
                    if vis_map:
                        for prim_path, vis_value in vis_map.items():
                            token = "inherited" if vis_value == "inherited" else "invisible"
                            renderer.write_attribute([prim_path], "visibility", [token])

                    # Progressive accumulation via dt=0 loop (see the
                    # one-shot worker block above for rationale).
                    all_products = None
                    for update_idx in range(num_sensor_updates):
                        # Historical 0.2.0 guard: visibility write_attribute
                        # changes could crash when followed only by dt=0
                        # accumulation steps. A single nonzero step lets the
                        # renderer consume the scene-state change; subsequent
                        # steps keep accumulating the selected USD frame. Keep
                        # this fallback guarded until native 0.3 visibility is
                        # validated.
                        delta_time = (1.0 / 60.0) if vis_map and update_idx == 0 else 0.0
                        all_products = renderer.step(
                            render_products=all_product_paths,
                            delta_time=delta_time,
                        )

                    if all_products:
                        for cam_idx, product_path in enumerate(product_paths):
                            if product_path not in all_products:
                                continue
                            product = all_products[product_path]
                            for frame in product.frames:
                                if "LdrColor" in frame.render_vars:
                                    with frame.render_vars["LdrColor"].map(device=_cpu_device()) as var:
                                        pixels = np.from_dlpack(var).copy()
                                    fname = f"cam{cam_idx}_f{frame_num}.png"
                                    fpath = os.path.join(output_dir, fname)
                                    Image.fromarray(pixels).save(fpath)
                                    cam_data[cam_idx]["images"].append(fname)

                                for sname in sensors:
                                    rv = SENSOR_MAP.get(sname)
                                    if rv and rv in frame.render_vars:
                                        with frame.render_vars[rv].map(device=_cpu_device()) as var:
                                            sarr = np.from_dlpack(var).copy()
                                        sfname = f"cam{cam_idx}_f{frame_num}_{sname}.npy"
                                        np.save(os.path.join(output_dir, sfname), sarr)
                                        cam_data[cam_idx]["sensor_files"][sname][frame_num] = sfname
            finally:
                _restore_native_stdout(saved_fd)

            manifest = []
            for cam_idx, camera in enumerate(cameras):
                manifest.append({
                    "camera": camera,
                    "image_files": cam_data[cam_idx]["images"],
                    "sensor_files": cam_data[cam_idx]["sensor_files"],
                    "frame_count": len(cam_data[cam_idx]["images"]),
                })

            sys.stdout.write(
                json.dumps({"status": "ok", "manifest": manifest}) + "\n"
            )
            sys.stdout.flush()

        except Exception:
            sys.stdout.write(
                json.dumps({"status": "error", "error": traceback.format_exc()[-2000:]}) + "\n"
            )
            sys.stdout.flush()
        finally:
            # Drop the final native result/mapping wrappers before the next
            # stage reset. Process recycling remains the hard memory bound,
            # but retaining the last result while idle delays normal OVRTX
            # cleanup unnecessarily.
            all_products = None
            product = None
            frame = None
            var = None
            pixels = None
            sarr = None

    # Clean shutdown
    del renderer


if __name__ == "__main__":
    main()
'''


class _OvRTXDaemon:
    """Manages a persistent OvRTX renderer subprocess.

    The daemon creates ``ovrtx.Renderer()`` once at startup and loops on
    stdin for JSON-line render commands, avoiding the ~5.5 s GPU init cost
    on every call.  If the daemon crashes, the next ``render()`` call
    restarts it transparently.
    """

    def __init__(
        self,
        ovrtx_python: str,
        daemon_script_path: str,
        log_level: str = "warn",
        ovrtx_venv_dir: Path | None = None,
    ) -> None:
        self._ovrtx_python = ovrtx_python
        self._ovrtx_venv_dir = ovrtx_venv_dir
        self._daemon_script_path = daemon_script_path
        self._log_level = log_level
        self._process: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stdout_buffer = b""
        self._lock = threading.Lock()
        self._start_timeout_s = float(
            os.environ.get("OVRTX_DAEMON_START_TIMEOUT", "600")
        )
        self._render_timeout_s = float(
            os.environ.get("OVRTX_DAEMON_RENDER_TIMEOUT", "1800")
        )
        self._max_completed_renders = _parse_nonnegative_int_env(
            OVRTX_DAEMON_MAX_RENDERS_ENV,
            DEFAULT_OVRTX_DAEMON_MAX_RENDERS,
        )
        self._max_rss_bytes = _parse_nonnegative_int_env(
            OVRTX_DAEMON_MAX_RSS_BYTES_ENV,
            DEFAULT_OVRTX_DAEMON_MAX_RSS_BYTES,
        )
        self._completed_renders = 0
        self._last_rss_bytes: int | None = None
        self._recycle_count = 0
        self._last_recycle_reason: str | None = None
        self._pending_recycle_reason: str | None = None
        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_running(self) -> None:
        """Start the daemon if it is not already running."""
        with self._lock:
            if self._is_running():
                return
            self._start()
            # A render-limit recycle may have stopped the old process and then
            # failed while starting its replacement. Production callers invoke
            # ensure_running() before render(), so acknowledge that pending
            # recycle here once the later replacement reaches ready.
            self._record_successful_pending_recycle()

    def lifecycle_snapshot(self) -> dict[str, int | str | None]:
        """Return non-blocking process lifecycle diagnostics for service health."""
        process = self._process
        pid = process.pid if process is not None and process.poll() is None else None
        rss_bytes = _linux_process_rss_bytes(pid) if pid is not None else None
        if rss_bytes is None:
            rss_bytes = self._last_rss_bytes
        return {
            "daemon_pid": pid,
            "daemon_completed_renders": self._completed_renders,
            "daemon_rss_bytes": rss_bytes,
            "daemon_recycle_count": self._recycle_count,
            "daemon_last_recycle_reason": self._last_recycle_reason,
            "daemon_pending_recycle_reason": self._recycle_reason(rss_bytes),
        }

    def _start(self) -> None:
        """Launch the daemon subprocess and wait for its *ready* signal."""
        env = _ovrtx_subprocess_env()
        # Remove PYTHONPATH so the isolated ovrtx venv doesn't pick up
        # the app's pxr/OpenUSD bindings (which conflict with ovrtx's bundled USD).
        env.pop("PYTHONPATH", None)
        if not env.get("DISPLAY"):
            env["DISPLAY"] = ":0"
        env["OVRTX_LOG_LEVEL"] = self._log_level
        env.pop("_WU_OVRTX_SITE_DIR", None)
        site_dir_env = _ovrtx_site_dir_env_for_python(
            self._ovrtx_python, self._ovrtx_venv_dir
        )
        if site_dir_env is not None:
            env["_WU_OVRTX_SITE_DIR"] = site_dir_env

        logger.info("Starting OvRTX daemon subprocess …")
        self._stdout_buffer = b""
        self._process = subprocess.Popen(
            [self._ovrtx_python, self._daemon_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        # Background thread to drain stderr so the pipe never fills up
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        # Wait for the "ready" JSON line. Do not let a wedged daemon pin
        # service startup forever; kill it so the next request can retry cleanly.
        ready_line = self._read_stdout_line(self._start_timeout_s, "startup")
        if not ready_line:
            rc = self._process.wait(timeout=10)
            self._process = None
            self._stdout_buffer = b""
            raise RuntimeError(f"OvRTX daemon exited during init (exit code {rc})")
        try:
            msg = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            self._kill_process()
            raise RuntimeError("OvRTX daemon returned invalid startup JSON") from exc
        if msg.get("status") != "ready":
            self._kill_process()
            raise RuntimeError(f"OvRTX daemon unexpected init msg: {msg}")
        # Do not clear lifecycle state until the replacement process has
        # actually reached the ready protocol state.
        self._completed_renders = 0
        self._last_rss_bytes = _linux_process_rss_bytes(self._process.pid)
        logger.info("OvRTX daemon ready (pid %d)", self._process.pid)

    def _drain_stderr(self) -> None:
        """Read stderr lines and send them to the logger."""
        proc = self._process
        assert proc is not None and proc.stderr is not None
        for line in proc.stderr:
            logger.debug("[ovrtx-daemon] %s", line.rstrip())

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _recycle_reason(self, rss_bytes: int | None = None) -> str | None:
        # Once a guard has tripped, retain its reason until a replacement has
        # reached ready. This prevents a later RSS sample from rewriting the
        # reason associated with the recycle.
        if self._pending_recycle_reason is not None:
            return self._pending_recycle_reason
        if (
            self._max_rss_bytes > 0
            and rss_bytes is not None
            and rss_bytes >= self._max_rss_bytes
        ):
            return "rss_limit"
        if (
            self._max_completed_renders > 0
            and self._completed_renders >= self._max_completed_renders
        ):
            return "completed_render_limit"
        return None

    def _record_successful_pending_recycle(self) -> None:
        reason = self._pending_recycle_reason
        if reason is None:
            return
        self._recycle_count += 1
        self._last_recycle_reason = reason
        self._pending_recycle_reason = None

    def _recycle_before_render(self, reason: str) -> None:
        logger.warning(
            "Recycling OvRTX daemon before render "
            "(reason=%s, completed_renders=%d, rss_bytes=%s)",
            reason,
            self._completed_renders,
            self._last_rss_bytes,
        )
        self._pending_recycle_reason = reason
        self._shutdown_locked()
        self._start()
        self._record_successful_pending_recycle()

    def render(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Send a render request and return the manifest.

        If the daemon is not running (or has crashed), it is (re)started
        automatically.
        """
        with self._lock:
            if not self._is_running():
                logger.warning("OvRTX daemon not running — restarting")
                self._start()
                self._record_successful_pending_recycle()

            # Sample the child immediately before dispatch while the daemon
            # lock is held. The post-render sample normally trips the guard,
            # while this sample also catches allocations that settle or grow
            # after the previous response was written.
            assert self._process is not None
            current_rss_bytes = _linux_process_rss_bytes(self._process.pid)
            if current_rss_bytes is not None:
                self._last_rss_bytes = current_rss_bytes
            recycle_reason = self._recycle_reason(self._last_rss_bytes)
            if recycle_reason is not None:
                self._recycle_before_render(recycle_reason)

            assert self._process is not None
            assert self._process.stdin is not None
            assert self._process.stdout is not None

            request = {"command": "render", **params}
            try:
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                rc = self._process.poll()
                self._kill_process()
                raise RuntimeError(
                    f"OvRTX daemon pipe failed before render response (exit code {rc})"
                ) from exc

            response_line = self._read_stdout_line(self._render_timeout_s, "render")
            if not response_line:
                rc = self._process.poll() if self._process is not None else None
                self._kill_process()
                raise RuntimeError(f"OvRTX daemon died during render (exit code {rc})")
            response = json.loads(response_line)
            self._completed_renders += 1
            self._last_rss_bytes = _linux_process_rss_bytes(self._process.pid)
            if self._pending_recycle_reason is None:
                self._pending_recycle_reason = self._recycle_reason(
                    self._last_rss_bytes
                )

        if response.get("status") == "error":
            raise RuntimeError(f"OvRTX daemon render error: {response.get('error')}")
        if response.get("status") != "ok":
            raise RuntimeError(f"OvRTX daemon unexpected response: {response}")
        manifest: list[dict[str, Any]] = response["manifest"]
        return manifest

    def _read_stdout_line(self, timeout_s: float, phase: str) -> str:
        """Read one daemon stdout line with a timeout.

        ``readline()`` on a subprocess pipe blocks indefinitely if the daemon
        stays alive but stops writing. A single ``select()`` before
        ``readline()`` is not enough because a partial line makes the fd
        readable and ``readline()`` can then block waiting for ``\n``. Read the
        pipe bytes directly until newline, EOF, or the real deadline.
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
            "OvRTX daemon %s timed out after %.1fs; killing subprocess",
            phase,
            timeout_s,
        )
        self._kill_process()
        raise TimeoutError(f"OvRTX daemon {phase} timed out after {timeout_s:.1f}s")

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
            logger.exception("Failed to kill OvRTX daemon subprocess")
        finally:
            self._process = None
            self._stdout_buffer = b""

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown_locked(self) -> None:
        """Stop the current daemon while the caller owns ``self._lock``."""
        if not self._is_running():
            self._process = None
            self._stdout_buffer = b""
            return
        assert self._process is not None
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
            self._process.stdin.flush()
            self._process.wait(timeout=10)
        except (BrokenPipeError, OSError):
            self._kill_process()
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        finally:
            logger.info("OvRTX daemon shut down")
            self._process = None
            self._stdout_buffer = b""

    def shutdown(self) -> None:
        """Gracefully shut down the daemon subprocess."""
        with self._lock:
            self._shutdown_locked()


def _world_understanding_resource_path(*parts: str) -> Path:
    """Return a filesystem path for a bundled world_understanding resource."""
    resource = importlib_resources.files("world_understanding").joinpath(*parts)
    try:
        return Path(resource)
    except TypeError:
        return Path(str(resource))


# Default latlong HDRI bundled with world-understanding for OVRTX light
# injection. Keep this repo-owned rather than reaching into the isolated OVRTX
# runtime so the default rig is stable across renderer package layouts.
_BUNDLED_DEFAULT_HDRI_FILENAME = "studio.exr"
_BUNDLED_DEFAULT_HDRI_PATH = _world_understanding_resource_path(
    "data",
    "env_maps",
    _BUNDLED_DEFAULT_HDRI_FILENAME,
)

# OVRTX-packaged StinsonBeach HDRI path helpers are retained for compatibility
# with explicit operator overrides and tests that classify packaged OVRTX assets.
_OVRTX_DEFAULT_HDRI_FILENAME = "StinsonBeach.hdr"
_OVRTX_DEFAULT_HDRI_RELATIVE_PATH = (
    Path("ovrtx")
    / "bin"
    / "plugins"
    / "usd"
    / "hdx"
    / "resources"
    / "textures"
    / _OVRTX_DEFAULT_HDRI_FILENAME
)
_BUNDLED_LEGACY_HDRI_PATH = _world_understanding_resource_path(
    "data",
    "env_maps",
    "SmartMaterials_Environment_with_Lights.exr",
)
_OVRTX_MOVED_HDRI_NEGATIVE_CACHE_SECONDS = 60.0
_OVRTX_MOVED_HDRI_CACHE: dict[Path, tuple[float, tuple[Path, ...]]] = {}
_OVRTX_MOVED_HDRI_RELATIVE_CANDIDATES = (
    Path("resources") / "textures" / _OVRTX_DEFAULT_HDRI_FILENAME,
    Path("resources") / _OVRTX_DEFAULT_HDRI_FILENAME,
    Path("textures") / _OVRTX_DEFAULT_HDRI_FILENAME,
)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Return paths in order without duplicates."""
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _ovrtx_site_packages_candidates(venv_dir: Path) -> list[Path]:
    """Return possible site-packages dirs for the isolated ovrtx runtime."""
    candidates: list[Path] = []

    if os.name == "nt":
        candidates.append(venv_dir / "Lib" / "site-packages")

    lib_dirs = [venv_dir / "lib", venv_dir / "lib64"]
    discovered_sites = [
        site_dir
        for lib_dir in lib_dirs
        for site_dir in sorted(lib_dir.glob("python*/site-packages"))
    ]
    target_sites = [lib_dir / "python" / "site-packages" for lib_dir in lib_dirs]

    candidates.extend(discovered_sites)
    candidates.extend(target_sites)

    return _unique_paths(candidates)


def _default_ovrtx_hdri_candidates(
    site_dir: Path, *, include_moved_fallbacks: bool = False
) -> list[Path]:
    """Return candidate StinsonBeach HDRI paths under one site-packages dir."""
    primary_path = site_dir / _OVRTX_DEFAULT_HDRI_RELATIVE_PATH
    if primary_path.exists() or not include_moved_fallbacks:
        return [primary_path]

    ovrtx_package_dir = site_dir / "ovrtx"
    moved_paths = _moved_ovrtx_hdri_candidates(ovrtx_package_dir)
    return _unique_paths([primary_path, *moved_paths])


def _moved_ovrtx_hdri_candidates(ovrtx_package_dir: Path) -> tuple[Path, ...]:
    """Return cached bounded fallback candidates for moved StinsonBeach.hdr."""
    if not ovrtx_package_dir.exists():
        return ()
    now = time.monotonic()
    cached = _OVRTX_MOVED_HDRI_CACHE.get(ovrtx_package_dir)
    if cached is not None:
        expires_at, moved_paths = cached
        if moved_paths and all(path.exists() for path in moved_paths):
            return moved_paths
        if not moved_paths and now < expires_at:
            return moved_paths

    moved_paths = tuple(
        path
        for path in (
            ovrtx_package_dir / relative_path
            for relative_path in _OVRTX_MOVED_HDRI_RELATIVE_CANDIDATES
        )
        if path.exists()
    )
    expires_at = (
        float("inf") if moved_paths else now + _OVRTX_MOVED_HDRI_NEGATIVE_CACHE_SECONDS
    )
    _OVRTX_MOVED_HDRI_CACHE[ovrtx_package_dir] = (expires_at, moved_paths)
    return moved_paths


def _ovrtx_site_packages_dir(venv_dir: Path) -> Path:
    """Return the site-packages dir that contains the ovrtx package."""
    candidates = _ovrtx_site_packages_candidates(venv_dir)
    if not candidates:
        raise RuntimeError(
            "No candidate site-packages directories were found for the isolated "
            f"ovrtx runtime at {venv_dir}."
        )
    for candidate in candidates:
        if (candidate / "ovrtx").is_dir():
            return candidate
    searched = ", ".join(str(candidate / "ovrtx") for candidate in candidates)
    raise RuntimeError(
        "The ovrtx package directory was not found in the isolated runtime. "
        f"Searched: {searched}."
    )


def _ovrtx_bundled_python_libraries(venv_dir: Path) -> list[Path]:
    """Return OVRTX-bundled Python shared libraries that trigger image scans."""
    libraries: list[Path] = []
    for site_dir in _ovrtx_site_packages_candidates(venv_dir):
        ovrtx_dir = site_dir / "ovrtx"
        if ovrtx_dir.is_dir():
            libraries.extend(
                sorted(ovrtx_dir.rglob(_OVRTX_BUNDLED_PYTHON_LIBRARY_GLOB))
            )
    return _unique_paths(libraries)


def _remove_ovrtx_bundled_python_libraries(venv_dir: Path) -> list[Path]:
    """Remove unused OVRTX-bundled Python runtimes before image scanning."""
    removed: list[Path] = []
    for library in _ovrtx_bundled_python_libraries(venv_dir):
        try:
            library.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(
                "Failed to remove OVRTX-bundled Python runtime library "
                f"{library}; this file is not used by the isolated OVRTX daemon "
                "but triggers release image scans."
            ) from exc
        removed.append(library)

    if removed:
        logger.info(
            "Removed %d OVRTX-bundled Python runtime libraries before image scan: %s",
            len(removed),
            ", ".join(str(path) for path in removed),
        )
    remaining = _ovrtx_bundled_python_libraries(venv_dir)
    if remaining:
        raise RuntimeError(
            "OVRTX-bundled Python runtime libraries remain after cleanup: "
            + ", ".join(str(path) for path in remaining)
        )
    return removed


def _default_ovrtx_hdri_path(
    venv_dir: Path | None = None, *, require_exists: bool = False
) -> str:
    """Return the default ovrtx-packaged HDRI path without importing ovrtx."""
    candidates = _ovrtx_site_packages_candidates(venv_dir or _OVRTX_VENV_DIR)
    if not candidates:
        raise RuntimeError(
            "No candidate site-packages directories were found for the isolated "
            f"ovrtx runtime at {venv_dir or _OVRTX_VENV_DIR}."
        )
    for site_dir in candidates:
        for hdri_path in _default_ovrtx_hdri_candidates(
            site_dir, include_moved_fallbacks=require_exists
        ):
            if hdri_path.exists():
                return str(hdri_path)

    fallback_path = candidates[0] / _OVRTX_DEFAULT_HDRI_RELATIVE_PATH
    if require_exists:
        searched = ", ".join(
            str(site_dir / _OVRTX_DEFAULT_HDRI_RELATIVE_PATH) for site_dir in candidates
        )
        raise RuntimeError(
            "Default OVRTX HDRI StinsonBeach.hdr was not found in the isolated "
            f"ovrtx runtime. Searched: {searched}. Set WU_OVRTX_DEFAULT_HDRI "
            "to an explicit HDRI path or rebuild the ovrtx runtime."
        )
    return str(fallback_path)


def _default_bundled_hdri_path(*, require_exists: bool = False) -> str:
    """Return the repo-bundled default HDRI path.

    Non-strict calls are import-time compatibility hints only. Render paths use
    ``require_exists=True`` so explicit ``WU_OVRTX_DEFAULT_HDRI`` overrides can
    still rescue stripped redistributions that omit the bundled EXR.
    """
    if require_exists and not _BUNDLED_DEFAULT_HDRI_PATH.is_file():
        raise RuntimeError(
            "Default OVRTX HDRI studio.exr was not found in "
            f"{_BUNDLED_DEFAULT_HDRI_PATH.parent}. Set WU_OVRTX_DEFAULT_HDRI "
            "to an explicit HDRI path or restore the bundled env map asset."
        )
    return str(_BUNDLED_DEFAULT_HDRI_PATH)


# Import-time hint for compatibility with existing introspection callers. Keep
# non-strict; render-time helpers validate the file before authoring it.
_DEFAULT_HDRI_PATH = _default_bundled_hdri_path()


def _resolve_default_hdri(*, require_exists: bool = False) -> str:
    """Return the HDRI asset path/URL to use for default DomeLight binding."""
    override = os.environ.get("WU_OVRTX_DEFAULT_HDRI", "").strip()
    if override:
        if _has_usda_asset_delimiter_chars(override):
            raise RuntimeError(
                "WU_OVRTX_DEFAULT_HDRI contains characters that cannot be safely "
                "embedded in a USDA asset path. Avoid newlines, carriage returns, "
                "and '@' characters."
            )
        override_path = _local_asset_path(override)
        absolute_local_override = override_path.is_absolute() or (
            _looks_like_windows_drive_path(str(override_path))
        )
        if (
            require_exists
            and _is_local_asset_path(override)
            and absolute_local_override
            and not override_path.exists()
        ):
            raise RuntimeError(
                "WU_OVRTX_DEFAULT_HDRI points to a missing local HDRI file: "
                f"{override}. Set it to an existing file path or a remote asset URL."
            )
        return override
    return _default_bundled_hdri_path(require_exists=require_exists)


def _ovrtx_site_dir_env_for_python(
    python_path: str, venv_dir: Path | None = None
) -> str | None:
    """Return a site-packages override only when tied to the active runtime."""
    try:
        if venv_dir is not None:
            return str(_ovrtx_site_packages_dir(venv_dir))
        if not _is_standard_ovrtx_python_path(python_path):
            return None
        return str(
            _ovrtx_site_packages_dir(_ovrtx_venv_dir_from_python_path(python_path))
        )
    except RuntimeError as exc:
        logger.debug("No local ovrtx site-package env override: %s", exc)
        return None


# DomeLight ``intensity`` multiplier applied to the HDRI texture. OVRTX needs
# the studio HDRI amplified to light first-run, lightless assets reliably.
_DEFAULT_HDRI_INTENSITY = 600.0
_CUSTOM_HDRI_DEFAULT_INTENSITY = 1.0


def _stage_has_lights(stage: "Usd.Stage") -> bool:
    """True if ``stage`` already contains at least one UsdLux light prim."""
    from pxr import UsdLux

    return any(
        p.IsA(UsdLux.BoundableLightBase) or p.IsA(UsdLux.NonboundableLightBase)
        for p in stage.Traverse()
    )


def _is_bundled_legacy_hdri(hdri_asset: str | None) -> bool:
    """True when ``hdri_asset`` is the bundled legacy EXR."""
    if not hdri_asset or not _is_local_asset_path(hdri_asset):
        return False
    return _paths_equal(
        _local_asset_path(hdri_asset).expanduser(),
        _BUNDLED_LEGACY_HDRI_PATH.expanduser(),
    )


def _is_bundled_default_hdri(hdri_asset: str | None) -> bool:
    """True when ``hdri_asset`` is the repo-bundled active default HDRI."""
    if not hdri_asset or not _is_local_asset_path(hdri_asset):
        return False
    return _paths_equal(
        _local_asset_path(hdri_asset).expanduser(),
        _BUNDLED_DEFAULT_HDRI_PATH.expanduser(),
    )


def _paths_equal(left: Path, right: Path) -> bool:
    """Return True when two local paths identify the same location."""
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
            os.path.abspath(right)
        )


def _is_ovrtx_packaged_default_hdri(
    hdri_asset: str | None, venv_dir: Path | None = None
) -> bool:
    """True when ``hdri_asset`` is the ovrtx-packaged default HDRI path."""
    if not hdri_asset or not _is_local_asset_path(hdri_asset):
        return False
    hdri_path = _local_asset_path(hdri_asset).expanduser()
    for site_dir in _ovrtx_site_packages_candidates(venv_dir or _OVRTX_VENV_DIR):
        for candidate in _default_ovrtx_hdri_candidates(
            site_dir, include_moved_fallbacks=True
        ):
            if _paths_equal(hdri_path, candidate.expanduser()):
                return True
    return False


def _has_usda_asset_delimiter_chars(asset_path: str) -> bool:
    """True when ``asset_path`` cannot be embedded in ``@asset@`` syntax."""
    return any(char in asset_path for char in ("\n", "\r", "@"))


def _resolve_default_hdri_intensity(
    hdri_asset: str | None = None, venv_dir: Path | None = None
) -> float:
    """Return the ``DomeLight.intensity`` to use for the default HDRI dome.

    Reads ``WU_OVRTX_DEFAULT_HDRI_INTENSITY`` env var (float); falls back
    to 600.0 for the bundled studio HDRI and compatibility OVRTX-packaged
    StinsonBeach HDRI paths. Explicit HDRI overrides, even when they share
    the same basename, and explicitly selected bundled legacy EXR assets
    keep the historical 1.0 fallback unless the operator sets
    ``WU_OVRTX_DEFAULT_HDRI_INTENSITY``.
    """
    val = os.environ.get("WU_OVRTX_DEFAULT_HDRI_INTENSITY", "").strip()
    hdri_override = os.environ.get("WU_OVRTX_DEFAULT_HDRI", "").strip()
    effective_hdri_asset = hdri_asset if hdri_asset is not None else hdri_override
    if _is_bundled_default_hdri(
        effective_hdri_asset
    ) or _is_ovrtx_packaged_default_hdri(effective_hdri_asset, venv_dir):
        fallback_intensity = _DEFAULT_HDRI_INTENSITY
    elif _is_bundled_legacy_hdri(effective_hdri_asset):
        fallback_intensity = _CUSTOM_HDRI_DEFAULT_INTENSITY
    elif hdri_override:
        fallback_intensity = _CUSTOM_HDRI_DEFAULT_INTENSITY
    else:
        fallback_intensity = _DEFAULT_HDRI_INTENSITY
    if not val:
        return fallback_intensity
    try:
        return float(val)
    except ValueError:
        logger.warning(
            "Invalid WU_OVRTX_DEFAULT_HDRI_INTENSITY=%r, using default %g",
            val,
            fallback_intensity,
        )
        return fallback_intensity


def _portable_stage_hdri_asset(stage: "Usd.Stage", hdri_asset: str) -> str:
    """Return a stage-portable HDRI asset path for direct stage mutation."""
    if _is_remote_asset_path(hdri_asset):
        return hdri_asset

    source = _local_asset_path(hdri_asset).expanduser()
    if not source.is_absolute():
        return hdri_asset

    root_layer = stage.GetRootLayer()
    if not root_layer.realPath or not source.is_file():
        return hdri_asset

    destination = Path(root_layer.realPath).parent / source.name
    if destination.resolve(strict=False) != source.resolve(strict=False):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return source.name


def _build_default_lights_usda(hdri_asset: str, intensity: float) -> str:
    """Serialize a default HDRI DomeLight as a standalone USDA overlay.

    This is the *sublayer* form of the lights rig, used by
    ``render_all_cameras``. Mutating the loaded scene stage in-place and
    then ``Export()``-ing it sometimes rewrites ``@<url>@`` asset paths
    to resolved local paths that OvRTX can't resolve, producing a black
    render. Writing the dome to a separate sublayer file preserves the
    asset URL verbatim, matching the robot-ovrtx/planet-system reference
    scene structure and what the ovrtx examples render cleanly.
    """
    hdri_asset = hdri_asset.replace("\\", "/")
    if _has_usda_asset_delimiter_chars(hdri_asset):
        raise ValueError(
            "Default OVRTX HDRI asset cannot be embedded in USDA asset syntax "
            "because it contains a newline, carriage return, or '@' character."
        )
    return f"""#usda 1.0
(
)

def "OvRTXDefaultLights" (
    hide_in_stage_window = true
    no_delete = true
)
{{
    def DomeLight "DomeLight"
    {{
        float inputs:intensity = {intensity}
        token inputs:texture:format = "latlong"
        asset inputs:texture:file = @{hdri_asset}@
        custom bool visibleInPrimaryRay = 0
    }}
}}
"""


def build_default_hdri_lights_usda(
    *,
    intensity: float | None = None,
    require_exists: bool = True,
    venv_dir: Path | None = None,
) -> str:
    """Return the default HDRI DomeLight USDA overlay used by OvRTX rendering."""
    hdri = _resolve_default_hdri(require_exists=require_exists)
    hdri_venv_dir = venv_dir.expanduser() if venv_dir is not None else None
    resolved_intensity = (
        _resolve_default_hdri_intensity(hdri, hdri_venv_dir)
        if intensity is None
        else float(intensity)
    )
    return _build_default_lights_usda(hdri, resolved_intensity)


def _ensure_lights(stage: "Usd.Stage", venv_dir: Path | None = None) -> None:
    """Add a default HDRI DomeLight to the stage if none are present.

    Kept for stand-alone callers that mutate a ``Usd.Stage`` directly
    (tests, CLIs). The daemon pipeline in ``render_all_cameras`` uses
    ``_build_default_lights_usda`` as a sublayer overlay instead -
    see the helper's docstring for why. Direct stage mutation validates the
    default HDRI before authoring it.

    Args:
        stage: USD stage to check/modify (in-place).
        venv_dir: OVRTX runtime root used only for compatibility intensity
            classification of explicit packaged-HDRI paths; it does not choose
            the active bundled default HDRI.
    """
    from pxr import Sdf, UsdLux

    if _stage_has_lights(stage):
        return

    hdri_venv_dir = venv_dir.expanduser() if venv_dir is not None else None
    hdri = _resolve_default_hdri(require_exists=True)
    intensity = _resolve_default_hdri_intensity(hdri, hdri_venv_dir)
    hdri = _portable_stage_hdri_asset(stage, hdri)
    logger.info("No lights in stage - adding HDRI DomeLight (%s)", hdri)

    dome = UsdLux.DomeLight.Define(stage, "/OvRTXDefaultLights/DomeLight")
    dome.CreateIntensityAttr(intensity)
    dome.GetPrim().CreateAttribute(
        "inputs:texture:format", Sdf.ValueTypeNames.Token
    ).Set("latlong")
    dome.GetPrim().CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set(
        hdri
    )
    dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(
        False
    )


def render_all_cameras(
    stage: "Usd.Stage",
    image_width: int = 512,
    image_height: int = 512,
    cameras: list[str] | None = None,
    frames: str = "0",
    sensors: list[str] | None = None,
    ovrtx_renderer: Any = None,
    log_level: str = "warn",
    ovrtx_venv_dir: Path | str | None = None,
    num_sensor_updates: int = DEFAULT_NUM_SENSOR_UPDATES,
    render_mode: str = DEFAULT_RENDER_MODE,
    daemon: _OvRTXDaemon | None = None,
    base_dir: str | Path | None = None,
    *,
    rtx_pt_samples_per_pixel: int | None = None,
    rtx_rt_accumulation_limit: int | None = None,
    material_target: str | None = "auto",
) -> dict[str, Any]:
    """Render multiple cameras from a USD stage using OvRTX.

    This function exports the stage to a temp file, then launches an isolated
    subprocess using a separate ovrtx-only venv (without another pxr provider) that
    renders all cameras and saves images to a temp directory, which are then
    loaded back in the main process. ``frames`` are USD frame numbers; the
    subprocess converts each frame to seconds using the stage's
    ``timeCodesPerSecond`` before calling ``renderer.update_from_usd_time``.
    Authored time-sampled USD state is preserved in the exported stage except
    for time-sampled visibility, which is replayed through static per-frame
    visibility overlay layers to avoid historical OvRTX 0.2.0 crashes until
    native 0.3 visibility handling is validated on GPU. Set
    ``WU_OVRTX_EXPERIMENTAL_NATIVE_VISIBILITY=1`` only for that validation
    probe to leave authored visibility samples in the exported USD and skip
    visibility-generated frame overlays/write-attribute updates. Time-sampled
    ``primvars:displayColor`` is preserved in the exported stage and, by
    default, replayed through static per-frame displayColor overlays until
    OVRTX 0.3 native sampling is validated. Set
    ``WU_OVRTX_EXPERIMENTAL_NATIVE_DISPLAYCOLOR=1`` only for that validation
    probe to skip displayColor-generated frame overlays while keeping the
    production workaround otherwise unchanged.

    Args:
        stage: USD stage to render.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        cameras: List of camera prim paths. If None, uses ["/Camera"].
        frames: Frame specification (e.g., "0", "0:10", "0,5,10").
        sensors: Optional sensor names (e.g., ["depth"]).
        ovrtx_renderer: Ignored (kept for API compatibility). Subprocess
            always creates its own renderer.
        log_level: OvRTX log level ("error", "warn", "info", "debug").
        ovrtx_venv_dir: Override directory for the isolated ovrtx venv.
            Defaults to ``~/.cache/wu/ovrtx_venv``.
        num_sensor_updates: Number of progressive
            ``renderer.step(delta_time=0)`` iterations to run per frame.
            Drives quality: more iterations = less path-trace noise, at
            linear wall-clock cost. Default ``32`` is the fast-iteration
            value paired with the ``rt2`` default mode (sufficient
            convergence for real-time path tracing). Raise toward
            ``500`` to reach the ``pt`` convergence plateau (within
            ~2 dB PSNR of Kit's reference render) at proportional
            wall-clock cost. Note: OVRtx 0.2.0 validation showed the
            ``omni:rtx:pt:samplesPerPixel`` /
            ``omni:rtx:rt:accumulationLimit`` USD attributes were silently
            ignored, so this guarded step loop remains the quality knob until
            0.3 GPU validation proves otherwise.
        render_mode: ``rt1`` | ``rt2`` | ``pt``. Translates to
            ``omni:rtx:rendermode`` on the RenderProduct. Default
            ``rt2`` (``RealTimePathTracing``, Kit's default) is the
            fast-iteration choice; it caps at ~27 dB PSNR vs the Kit
            reference regardless of step count. Use ``pt``
            (PathTracing) when Kit-parity quality is required — only
            ``pt`` reaches the Kit reference, at proportionally higher
            wall-clock cost. See ``docs/developer/OVRTX_LIMITATIONS.md``
            §5.
        daemon: Optional persistent daemon. When provided, the daemon's
            already-running ``ovrtx.Renderer()`` is reused, avoiding the
            ~5.5 s GPU init on every call.  When ``None`` (the default),
            falls back to a one-shot ``subprocess.run()`` worker.
        base_dir: Base directory for resolving relative texture assets when the
            stage root layer is anonymous or has been exported elsewhere.
        rtx_pt_samples_per_pixel: Probe-only value to emit as
            ``omni:rtx:pt:samplesPerPixel`` on generated RenderProducts.
            Leave as ``None`` for production renders until OVRTX 0.3 evidence
            proves the native attribute affects convergence. Unsupported with
            the persistent daemon to avoid silently misleading production
            callers.
        rtx_rt_accumulation_limit: Probe-only value to emit as
            ``omni:rtx:rt:accumulationLimit`` on generated RenderProducts.
            Leave as ``None`` for production renders until OVRTX 0.3 evidence
            proves the native attribute affects convergence. Unsupported with
            the persistent daemon to avoid silently misleading production
            callers.
        material_target: Explicit material target. ``auto`` preserves
            authored/native material outputs, ``preview_surface`` requests the
            OVRTX PreviewSurface fallback overlay explicitly, and
            ``openpbr_materialx`` preserves native OpenPBR/MaterialX output.

    Returns:
        Dict matching RenderingBackend.render() contract with keys:
            total_cameras, successful_cameras, failed_cameras,
            total_render_time, results (list of per-camera dicts).
    """
    if cameras is None or len(cameras) == 0:
        cameras = ["/Camera"]
    if daemon is not None and (
        rtx_pt_samples_per_pixel is not None or rtx_rt_accumulation_limit is not None
    ):
        raise ValueError(
            "rtx_pt_samples_per_pixel and rtx_rt_accumulation_limit are "
            "probe-only and are unsupported with the persistent OvRTX daemon; "
            "pass daemon=None when running the sample-attribute probe."
        )

    frame_list = _parse_frames(frames)
    native_displaycolor_probe = _native_displaycolor_probe_enabled()
    total_start_time = time.time()

    # Resolve the ovrtx venv Python (auto-provisions on first call)
    venv_path = Path(ovrtx_venv_dir) if ovrtx_venv_dir else None
    ovrtx_python = _get_ovrtx_python(venv_dir=venv_path)
    active_venv_path = _ovrtx_venv_dir_from_python_path(ovrtx_python, venv_path)

    # Create temp directory for IPC (exported USD + rendered images)
    tmp_dir = tempfile.mkdtemp(prefix="ovrtx_render_")
    tmp_usd_path = os.path.join(tmp_dir, "stage.usdc")

    # Imports and state used by both the try and finally blocks. Keeping
    # them out here means an early exception inside the try doesn't trigger
    # a NameError in the finally cleanup that would mask the original error.
    from pxr import Usd
    from pxr import UsdGeom as _UsdGeom

    # Keys are stringified floats (e.g. "1.0", "1.5") so subframe time
    # samples survive both the dict lookup and the JSON serialization
    # round-trip into the renderer subprocess. Integer-frame consumers
    # build their lookup with `str(float(frame_num))` to match.
    visibility_schedule: dict[str, dict[str, str]] = {}
    visibility_frame_values: dict[str, dict[str, str]] = {}
    display_color_frame_values: dict[str, dict[str, Any]] = {}
    stripped_prim_paths: list[str] = []
    deinstanced_prim_paths: list[str] = []
    deinstanced_prim_path_set: set[str] = set()
    # Captured default visibility per prim path. Restoring time samples
    # without restoring the default would silently flip prims that were
    # `invisible` by default to `inherited`. Populated at strip-time and
    # consumed in the finally block.
    stripped_default_vis: dict[str, str] = {}
    native_visibility_probe = _native_visibility_probe_enabled()
    had_default_lights = True  # safe default — finally only removes if False

    try:
        # Pre-build the USDA and product paths (pure string operations, no ovrtx)
        usda_content, product_paths = _build_render_products_usda(
            cameras=cameras,
            image_width=image_width,
            image_height=image_height,
            sensors=sensors,
            render_mode=render_mode,
            pt_samples_per_pixel=rtx_pt_samples_per_pixel,
            rt_accumulation_limit=rtx_rt_accumulation_limit,
        )

        # OvRTX requires explicit lights — scenes without any get a
        # default HDRI DomeLight. We emit it as a *sublayer* overlay
        # rather than mutating the live stage + re-exporting, because
        # Export() sometimes rewrites asset URL paths in ways OvRTX
        # can't resolve (resulting in a black render). The sublayer
        # path preserves the URL verbatim. See _build_default_lights_usda.
        root_layer = stage.GetRootLayer()
        had_default_lights = bool(stage.GetPrimAtPath("/OvRTXDefaultLights"))
        default_lights_layer_path: str | None = None
        if not _stage_has_lights(stage):
            default_lights_layer_path = os.path.join(tmp_dir, "default_lights.usda")
            default_hdri = _resolve_default_hdri(require_exists=True)
            with open(default_lights_layer_path, "w", encoding="utf-8") as f:
                f.write(
                    _build_default_lights_usda(
                        default_hdri,
                        _resolve_default_hdri_intensity(default_hdri, active_venv_path),
                    )
                )
            logger.info(
                "Scene has no lights — overlaying default HDRI DomeLight sublayer (%s)",
                default_lights_layer_path,
            )

        # Default: extract time-sampled visibility schedule and strip it from
        # the stage. OvRTX 0.2.0 segfaulted when USD contained time-sampled
        # visibility attributes. Until 0.3 GPU validation proves native
        # visibility safe, strip those samples from the base export and build
        # per-frame static visibility overlays below. The opt-in probe leaves
        # samples authored so the exact production path can be validated with
        # visibility overlays disabled.
        if native_visibility_probe:
            logger.warning(
                "%s=1: leaving time-sampled visibility authored for OVRTX "
                "native-visibility validation; keep disabled in production "
                "until GPU evidence proves parity",
                _NATIVE_VISIBILITY_PROBE_ENV,
            )
        visibility_prims: Iterable[Any] = (
            () if native_visibility_probe else stage.Traverse()
        )
        for prim in visibility_prims:
            if (
                prim.IsInstanceProxy()
            ):  # pragma: no cover - stage.Traverse skips proxies
                proxy_path = str(prim.GetPath())
                instance_root = prim
                while instance_root and not instance_root.IsInstance():
                    instance_root = instance_root.GetParent()
                if not instance_root or not instance_root.IsInstance():
                    continue
                instance_root_path = str(instance_root.GetPath())
                if instance_root_path not in deinstanced_prim_path_set:
                    instance_root.SetInstanceable(False)
                    deinstanced_prim_paths.append(instance_root_path)
                    deinstanced_prim_path_set.add(instance_root_path)
                prim = stage.GetPrimAtPath(proxy_path)
                if not prim or prim.IsInstanceProxy():  # pragma: no cover
                    continue
            vis_attr = _UsdGeom.Imageable(prim).GetVisibilityAttr()
            if not vis_attr or vis_attr.GetNumTimeSamples() == 0:
                continue
            prim_path_str = str(prim.GetPath())
            if prim.IsInstance():
                prim.SetInstanceable(False)
                if prim_path_str not in deinstanced_prim_path_set:
                    deinstanced_prim_paths.append(prim_path_str)
                    deinstanced_prim_path_set.add(prim_path_str)
                prim = stage.GetPrimAtPath(prim_path_str)
                vis_attr = _UsdGeom.Imageable(prim).GetVisibilityAttr()
            # Capture the prim's *default* visibility opinion before we
            # blow it away — Clear() drops both default and time samples,
            # and the restore path needs to put the default back so prims
            # that were invisible-by-default stay that way.
            default_val = vis_attr.Get(Usd.TimeCode.Default())
            if default_val is not None:
                stripped_default_vis[prim_path_str] = str(default_val)
            # Record schedule per time code (preserving subframes)
            for tc in vis_attr.GetTimeSamples():
                frame_key = str(float(tc))
                if frame_key not in visibility_schedule:
                    visibility_schedule[frame_key] = {}
                val = vis_attr.Get(Usd.TimeCode(tc))
                visibility_schedule[frame_key][prim_path_str] = str(val)
            for frame_num in frame_list:
                frame_key = str(float(frame_num))
                val = vis_attr.Get(Usd.TimeCode(frame_num))
                visibility_frame_values.setdefault(frame_key, {})[prim_path_str] = (
                    str(val) if val is not None else "inherited"
                )
            # Clear time samples and set default to inherited (visible)
            vis_attr.Clear()
            vis_attr.Set(_UsdGeom.Tokens.inherited)
            stripped_prim_paths.append(prim_path_str)

        if native_visibility_probe and (
            stripped_prim_paths or visibility_schedule or visibility_frame_values
        ):
            raise RuntimeError(
                "Native visibility probe unexpectedly stripped or scheduled "
                "visibility; disable the probe and inspect the export path"
            )

        # Capture time-sampled displayColor independently of visibility.
        # Stages with animated displayColor but static visibility used to
        # fall through here unhandled and rely on OvRTX's native per-time
        # evaluation, which was unverified in the 0.2.0 work. Keep the
        # per-frame overlay until 0.3 GPU validation confirms native
        # displayColor sampling.
        if native_displaycolor_probe:
            logger.warning(
                "%s=1: skipping displayColor frame overlays for OVRTX "
                "native-displayColor validation; keep disabled in production "
                "until GPU evidence proves parity",
                _NATIVE_DISPLAYCOLOR_PROBE_ENV,
            )
        display_color_prims: Iterable[Any] = (
            () if native_displaycolor_probe else stage.Traverse()
        )
        for prim in display_color_prims:
            color_attr = prim.GetAttribute("primvars:displayColor")
            if not color_attr or color_attr.GetNumTimeSamples() == 0:
                continue
            prim_path_str = str(prim.GetPath())
            for frame_num in frame_list:
                frame_key = str(float(frame_num))
                val = color_attr.Get(Usd.TimeCode(frame_num))
                if val is not None:
                    display_color_frame_values.setdefault(frame_key, {})[
                        prim_path_str
                    ] = val

        if stripped_prim_paths:
            logger.info(
                "Extracted visibility schedule for %d prims across %d frames "
                "(ovrtx per-frame overlay workaround)",
                len(stripped_prim_paths),
                len(visibility_schedule),
            )
        visibility_updates = _build_visibility_frame_updates(
            visibility_schedule, frame_list
        )
        if visibility_updates:
            update_count = sum(len(v) for v in visibility_updates.values())
            logger.info(
                "Prepared %d visibility write(s) across %d rendered frame(s) "
                "from %d scheduled frame(s)",
                update_count,
                len(visibility_updates),
                len(visibility_schedule),
            )

        # Export the stage to a temp file (without render products).
        t_export = time.time()
        if not root_layer.Export(tmp_usd_path):
            raise RuntimeError("Failed to export USD stage to temp file")
        material_fallback_layer_path = os.path.join(
            tmp_dir,
            "ovrtx_material_fallbacks.usda",
        )
        preview_fallbacks = 0
        effective_material_target = normalize_render_material_target(material_target)
        if preview_fallbacks_enabled_for_material_target(effective_material_target):
            from world_understanding.utils.usd.material import (
                write_ovrtx_preview_fallback_overlay_for_materialx_openpbr,
            )

            preview_fallbacks = (
                write_ovrtx_preview_fallback_overlay_for_materialx_openpbr(
                    stage,
                    material_fallback_layer_path,
                )
            )
            if preview_fallbacks:
                logger.info(
                    "Updated %d OpenPBR material fallback(s) for OVRTX export",
                    preview_fallbacks,
                )
        # This synchronous localization must finish before the daemon render call
        # below; the daemon only opens the exported stage inside ``render``.
        copied_assets = _copy_exported_relative_assets(
            stage,
            Path(tmp_dir),
            base_dir=base_dir,
            exported_stage_path=tmp_usd_path,
        )
        if copied_assets:
            logger.info(
                "Copied %d local texture asset(s) next to exported OVRTX stage",
                copied_assets,
            )

        # Write render products + lights as a separate USDA overlay
        render_products_layer_path = os.path.join(tmp_dir, "render_products.usda")
        with open(render_products_layer_path, "w", encoding="utf-8") as f:
            f.write(usda_content)

        # Create a combined USDA that sublayers the scene + render products.
        # This matches the contents-claw pattern: ovrtx loads the combined
        # file and resolves both layers correctly (vs flattening, which broke
        # render product discovery in the 0.2.0 validation work and remains
        # guarded pending 0.3 validation).
        from pxr import Sdf

        scene_layer_path = tmp_usd_path
        combined_path = os.path.join(tmp_dir, "combined.usda")
        combined = Sdf.Layer.CreateNew(combined_path)
        sublayers = []
        if preview_fallbacks:
            sublayers.append(material_fallback_layer_path)
        sublayers.extend([scene_layer_path, render_products_layer_path])
        if default_lights_layer_path is not None:
            sublayers.append(default_lights_layer_path)
        combined.subLayerPaths = sublayers
        combined.Save()

        frame_usd_paths: dict[str, str] = {}
        if visibility_frame_values or display_color_frame_values:
            for frame_num in frame_list:
                frame_key = str(float(frame_num))
                frame_token = str(frame_num).replace("-", "neg_")
                overlay_path = os.path.join(
                    tmp_dir, f"visibility_frame_{frame_token}.usda"
                )
                _write_frame_overlay(
                    overlay_path,
                    visibility_frame_values.get(frame_key, {}),
                    display_color_frame_values.get(frame_key, {}),
                )

                frame_combined_path = os.path.join(
                    tmp_dir, f"combined_frame_{frame_token}.usda"
                )
                frame_combined = Sdf.Layer.CreateNew(frame_combined_path)
                frame_sublayers = [overlay_path]
                if preview_fallbacks:
                    frame_sublayers.append(material_fallback_layer_path)
                frame_sublayers.extend([scene_layer_path, render_products_layer_path])
                if default_lights_layer_path is not None:
                    frame_sublayers.append(default_lights_layer_path)
                frame_combined.subLayerPaths = frame_sublayers
                frame_combined.Save()
                frame_usd_paths[str(frame_num)] = frame_combined_path

        tmp_usd_path = combined_path
        logger.debug("Exported USD stage in %.2fs", time.time() - t_export)

        # Read timeCodesPerSecond so the subprocess converts frame
        # numbers to seconds for update_from_usd_time().
        fps = stage.GetTimeCodesPerSecond()  # defaults to 24.0

        # Build subprocess parameters
        params = {
            "usd_path": tmp_usd_path,
            "fps": fps,
            "cameras": cameras,
            "image_width": image_width,
            "image_height": image_height,
            "frames": frame_list,
            "sensors": sensors or [],
            "output_dir": tmp_dir,
            "product_paths": product_paths,
            "num_sensor_updates": num_sensor_updates,
            "visibility_updates": visibility_updates,
            "frame_usd_paths": frame_usd_paths,
            "material_target": effective_material_target,
            "log_level": log_level,
        }

        if daemon is not None:
            # ---- Persistent daemon path (reuses Renderer across calls) ----
            logger.info(
                "Sending render request to OvRTX daemon: %d camera(s), %d frame(s)",
                len(cameras),
                len(frame_list),
            )
            daemon.ensure_running()
            t_render = time.time()
            manifest = daemon.render(params)
            logger.debug("Daemon render completed in %.2fs", time.time() - t_render)
        else:
            # ---- One-shot subprocess path (backward compatible) ----
            # Write worker script to temp file
            worker_path = os.path.join(tmp_dir, "_ovrtx_worker.py")
            with open(worker_path, "w", encoding="utf-8") as f:
                f.write(_WORKER_SCRIPT)

            # Launch subprocess using the isolated ovrtx venv Python.
            # Remove PYTHONPATH so the venv doesn't pick up the app's pxr/OpenUSD bindings.
            # OvRTX requires Vulkan GPU access which needs a display server.
            env = _ovrtx_subprocess_env()
            env.pop("PYTHONPATH", None)
            if not env.get("DISPLAY"):
                env["DISPLAY"] = ":0"
            env["OVRTX_LOG_LEVEL"] = log_level
            env.pop("_WU_OVRTX_SITE_DIR", None)
            site_dir_env = _ovrtx_site_dir_env_for_python(ovrtx_python, venv_path)
            if site_dir_env is not None:
                env["_WU_OVRTX_SITE_DIR"] = site_dir_env

            logger.info(
                "Launching OvRTX subprocess for %d camera(s), %d frame(s)",
                len(cameras),
                len(frame_list),
            )
            proc = subprocess.run(
                [ovrtx_python, worker_path, json.dumps(params)],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

            if proc.returncode != 0:
                error_msg = f"OvRTX subprocess failed (exit code {proc.returncode})"
                if proc.stdout:
                    error_msg += f"\n--- stdout (last 1000) ---\n{proc.stdout[-1000:]}"
                if proc.stderr:
                    error_msg += f"\n--- stderr (last 2000) ---\n{proc.stderr[-2000:]}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Read manifest from subprocess output
            manifest_path = os.path.join(tmp_dir, "manifest.json")
            if not os.path.exists(manifest_path):
                raise RuntimeError(
                    "OvRTX subprocess did not produce manifest. "
                    f"stdout: {proc.stdout[-500:]}, "
                    f"stderr: {proc.stderr[-500:]}"
                )

            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)

        # Load images and sensor data back into memory
        t_load = time.time()
        results = []
        successful_cameras = 0
        failed_cameras = 0
        render_warnings: list[str] = []
        blank_render_frames: list[dict[str, Any]] = []

        for cam_result in manifest:
            camera_images: list[Image.Image] = []
            camera_sensors: dict[str, dict[int, np.ndarray]] = {
                s: {} for s in (sensors or [])
            }
            camera_blank_frames: list[dict[str, Any]] = []
            camera_image_frames: list[int] = []

            # Load images
            image_files = cam_result["image_files"]
            for image_index, img_fname in enumerate(image_files):
                img_path = os.path.join(tmp_dir, img_fname)
                if os.path.exists(img_path):
                    frame = _frame_from_image_filename(
                        img_fname,
                        image_index=image_index,
                        frame_list=frame_list,
                        image_file_count=len(image_files),
                    )
                    image = Image.open(img_path).copy()
                    camera_images.append(image)
                    camera_image_frames.append(frame)
                    stats = analyze_image_blankness(
                        image,
                        max_analysis_pixels=_BLANKNESS_MAX_ANALYSIS_PIXELS,
                    )
                    if stats.blank:
                        blank_frame = {
                            "frame": frame,
                            "camera": cam_result["camera"],
                            "image_file": img_fname,
                            "stats": stats.to_dict(),
                        }
                        # Keep both per-camera and top-level lists so old and new
                        # clients can consume blank-frame metadata without merging.
                        camera_blank_frames.append(blank_frame)
                        blank_render_frames.append(blank_frame)
                        render_warnings.append(_blank_frame_warning(blank_frame))

            # Load sensor data
            for sensor_name, frame_files in cam_result["sensor_files"].items():
                for frame_num_str, npy_fname in frame_files.items():
                    npy_path = os.path.join(tmp_dir, npy_fname)
                    if os.path.exists(npy_path):
                        camera_sensors[sensor_name][int(frame_num_str)] = np.load(
                            npy_path
                        )

            if camera_images:
                successful_cameras += 1
                result: dict[str, Any] = {
                    "camera": cam_result["camera"],
                    "images": camera_images,
                    "sensors": camera_sensors,
                    "render_time": time.time() - total_start_time,
                    "frame_count": len(camera_images),
                    "image_frames": camera_image_frames,
                }
                if camera_blank_frames:
                    result["warnings"] = [
                        _blank_frame_warning(frame) for frame in camera_blank_frames
                    ]
                    result["blank_render_frames"] = camera_blank_frames
            else:
                failed_cameras += 1
                result = {
                    "camera": cam_result["camera"],
                    "images": [],
                    "sensors": {},
                    "render_time": time.time() - total_start_time,
                    "frame_count": 0,
                    "error": "No images produced",
                }

            results.append(result)

        logger.debug(
            "Loaded %d images from disk in %.2fs",
            sum(len(r.get("images", [])) for r in results),
            time.time() - t_load,
        )

    finally:
        # Revert stage mutations so the caller's stage is unchanged
        # Restore visibility time samples and the prim's original default
        # opinion (the strip path captured the default before clearing).
        if stripped_prim_paths:
            for prim_path in stripped_prim_paths:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim:
                    continue
                if prim.IsInstanceProxy():  # pragma: no cover
                    continue
                if prim.IsInstance():
                    prim.SetInstanceable(False)
                    prim = stage.GetPrimAtPath(prim_path)
                vis_attr = _UsdGeom.Imageable(prim).GetVisibilityAttr()
                vis_attr.Clear()
                # Restore the original default opinion BEFORE writing time
                # samples so prims that were invisible-by-default keep that
                # default after the render call returns.
                orig_default = stripped_default_vis.get(prim_path)
                if orig_default is not None:
                    default_token = (
                        _UsdGeom.Tokens.inherited
                        if orig_default == "inherited"
                        else _UsdGeom.Tokens.invisible
                    )
                    vis_attr.Set(default_token)
                for frame_key, vis_map in visibility_schedule.items():
                    if prim_path in vis_map:
                        val = vis_map[prim_path]
                        token = (
                            _UsdGeom.Tokens.inherited
                            if val == "inherited"
                            else _UsdGeom.Tokens.invisible
                        )
                        # frame_key is a stringified float ("1.0", "1.5", ...)
                        # so use float() to preserve subframe samples on
                        # round-trip back to the stage.
                        vis_attr.Set(token, time=Usd.TimeCode(float(frame_key)))

        for prim_path in reversed(deinstanced_prim_paths):
            prim = stage.GetPrimAtPath(prim_path)
            if prim and not prim.IsInstanceProxy():
                prim.SetInstanceable(True)

        if not had_default_lights:
            default_lights = stage.GetPrimAtPath("/OvRTXDefaultLights")
            if default_lights:
                stage.RemovePrim("/OvRTXDefaultLights")
        # Capture-and-replay debug hook: when set, the entire render
        # tmp_dir (combined.usda + scene export + render_products +
        # default_lights sublayer) is copied to this directory and the
        # combined.usda gets its sublayer paths rewritten to basenames
        # so the whole bundle is portable. Lets us diff the daemon's
        # output against a known-good pure-ovrtx scene to root-cause
        # rendering issues without rebuilding the container.
        dump_path = os.environ.get("WU_OVRTX_DUMP_COMBINED", "").strip()
        if dump_path:
            try:
                dump_dir = Path(dump_path).parent
                dump_dir.mkdir(parents=True, exist_ok=True)
                # Copy every file under tmp_dir to dump_dir (flat).
                for p in Path(tmp_dir).rglob("*"):
                    if p.is_file():
                        shutil.copy(p, dump_dir / p.name)
                # Rewrite the combined.usda sublayer paths to basenames.
                combined_src = dump_dir / Path(tmp_usd_path).name
                combined_dst = Path(dump_path)
                text = combined_src.read_text()
                text = text.replace(str(tmp_dir) + "/", "")
                combined_dst.write_text(text)
                if combined_dst != combined_src:
                    combined_src.unlink(missing_ok=True)
                logger.info(
                    "WU_OVRTX_DUMP_COMBINED: wrote %s (+ sublayers in %s)",
                    combined_dst,
                    dump_dir,
                )
            except Exception as _e:
                logger.warning("WU_OVRTX_DUMP_COMBINED copy failed: %s", _e)
        # Clean up temp directory
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

    total_render_time = time.time() - total_start_time

    return {
        "total_cameras": len(cameras),
        "successful_cameras": successful_cameras,
        "failed_cameras": failed_cameras,
        "total_render_time": total_render_time,
        "results": results,
        "warnings": render_warnings,
        "blank_render_frames": blank_render_frames,
    }


def _frame_from_image_filename(
    image_filename: str,
    *,
    image_index: int,
    frame_list: list[int],
    image_file_count: int,
) -> int:
    match = re.search(r"_f(\d+)(?:\.|_|$)", image_filename)
    if match:
        return int(match.group(1))
    if image_file_count == len(frame_list) and image_index < len(frame_list):
        return frame_list[image_index]
    return image_index


def _blank_frame_warning(blank_frame: dict[str, Any]) -> str:
    stats = blank_frame["stats"]
    return (
        "Blank or near-blank OVRTX render detected "
        f"for frame {blank_frame['frame']} camera {blank_frame['camera']}: "
        f"{stats['reason']} "
        f"(unique_colors={stats['unique_colors']}, "
        f"dominant_color_ratio={stats['dominant_color_ratio']:.3f}, "
        f"luma_std={stats['luma_std']:.3f})"
    )


def _make_sample_attribute_probe_stage() -> "Usd.Stage":
    """Build a small lit scene for OVRTX sample-attribute probing."""
    from pxr import Gf, Usd, UsdGeom, UsdLux, Vt

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(0.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    cube = UsdGeom.Cube.Define(stage, "/World/ProbeCube")
    cube.GetSizeAttr().Set(1.0)
    cube.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.85, 0.18, 0.08)]))

    light = UsdLux.SphereLight.Define(stage, "/World/KeyLight")
    light.CreateIntensityAttr(25000.0)
    light.CreateRadiusAttr(0.4)
    UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(1.5, 2.5, 3.0))

    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.GetFocalLengthAttr().Set(45.0)
    camera.GetHorizontalApertureAttr().Set(36.0)
    UsdGeom.Xformable(camera.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 4.0))
    return stage


def _probe_image_metrics(image: Image.Image) -> dict[str, Any]:
    """Return compact image metrics for low/high sample comparisons."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width, _ = rgb.shape
    center = rgb[height // 4 : (height * 3) // 4, width // 4 : (width * 3) // 4]
    center_f = center.astype(np.float32)
    luma = (
        center_f[..., 0] * 0.2126
        + center_f[..., 1] * 0.7152
        + center_f[..., 2] * 0.0722
    )
    return {
        "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "mean_rgb": [float(v) for v in rgb.mean(axis=(0, 1))],
        "center_luma_std": float(luma.std()),
        "unique_colors": int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0]),
    }


def _probe_mean_abs_rgb_diff(left: Image.Image, right: Image.Image) -> float:
    """Return mean absolute RGB delta between two probe renders."""
    left_rgb = np.asarray(left.convert("RGB"), dtype=np.int16)
    right_rgb = np.asarray(right.convert("RGB"), dtype=np.int16)
    return float(np.abs(left_rgb - right_rgb).mean())


def _probe_gpu_summary() -> str:
    """Return a short best-effort GPU/driver summary for probe evidence."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"nvidia-smi unavailable: {exc}"
    if result.returncode != 0:
        return f"nvidia-smi failed: {result.stderr.strip()[-300:]}"
    return result.stdout.strip()


@dataclass(frozen=True)
class _SampleAttributeProbeVariant:
    name: str
    render_mode: str
    num_sensor_updates: int
    rtx_pt_samples_per_pixel: int | None = None
    rtx_rt_accumulation_limit: int | None = None

    def to_result_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "render_mode": self.render_mode,
            "num_sensor_updates": self.num_sensor_updates,
        }
        if self.rtx_pt_samples_per_pixel is not None:
            result["rtx_pt_samples_per_pixel"] = self.rtx_pt_samples_per_pixel
        if self.rtx_rt_accumulation_limit is not None:
            result["rtx_rt_accumulation_limit"] = self.rtx_rt_accumulation_limit
        return result


def _run_sample_attribute_probe(
    *,
    image_size: int = 96,
    low_value: int = 1,
    high_value: int = 512,
    baseline_updates: int = 8,
    log_level: str = "warn",
    ovrtx_venv_dir: Path | None = None,
) -> dict[str, Any]:
    """Render low/high RTX sample-attribute variants and return metrics.

    This is a GPU maintenance probe, not production rendering behavior. It
    keeps ``num_sensor_updates`` at 1 for native-attribute comparisons so any
    measurable low/high delta comes from OVRTX honoring the authored USD
    attributes, not from our progressive step loop.
    """
    if image_size < 8:
        raise ValueError("image_size must be at least 8 pixels")
    if low_value < 1 or high_value < 1:
        raise ValueError("low_value and high_value must be positive integers")
    if baseline_updates < 2:
        raise ValueError("baseline_updates must be at least 2")

    ovrtx_python = _get_ovrtx_python(venv_dir=ovrtx_venv_dir)
    active_venv_dir = _ovrtx_venv_dir_from_python_path(
        ovrtx_python,
        ovrtx_venv_dir,
    )
    ovrtx_version = _probe_ovrtx_version(Path(ovrtx_python), active_venv_dir)
    ovrtx_version_warning = None
    if ovrtx_version != _OVRTX_VERSION:
        if ovrtx_version is None:
            ovrtx_version_warning = (
                "OVRTX sample-attribute probe expected "
                f"{_OVRTX_VERSION}, but could not confirm the installed "
                f"version from {ovrtx_python} in {active_venv_dir}. "
                "Continuing best-effort because this maintenance probe is "
                "used to evaluate patch/runtime drift."
            )
        else:
            ovrtx_version_warning = (
                "OVRTX sample-attribute probe expected "
                f"{_OVRTX_VERSION}; found {ovrtx_version!r} at "
                f"{active_venv_dir}. Continuing best-effort because this "
                "maintenance probe is used to evaluate patch/runtime drift."
            )
        logger.warning("%s", ovrtx_version_warning)

    variants = [
        _SampleAttributeProbeVariant(
            name="baseline_steps_1",
            render_mode="pt",
            num_sensor_updates=1,
        ),
        _SampleAttributeProbeVariant(
            name=f"baseline_steps_{baseline_updates}",
            render_mode="pt",
            num_sensor_updates=baseline_updates,
        ),
        _SampleAttributeProbeVariant(
            name=f"pt_spp_{low_value}",
            render_mode="pt",
            num_sensor_updates=1,
            rtx_pt_samples_per_pixel=low_value,
        ),
        _SampleAttributeProbeVariant(
            name=f"pt_spp_{high_value}",
            render_mode="pt",
            num_sensor_updates=1,
            rtx_pt_samples_per_pixel=high_value,
        ),
        _SampleAttributeProbeVariant(
            name=f"rt_accum_{low_value}",
            render_mode="rt2",
            num_sensor_updates=1,
            rtx_rt_accumulation_limit=low_value,
        ),
        _SampleAttributeProbeVariant(
            name=f"rt_accum_{high_value}",
            render_mode="rt2",
            num_sensor_updates=1,
            rtx_rt_accumulation_limit=high_value,
        ),
    ]

    images: dict[str, Image.Image] = {}
    rendered_variants: list[dict[str, Any]] = []
    for variant in variants:
        pt_samples = variant.rtx_pt_samples_per_pixel
        rt_accumulation = variant.rtx_rt_accumulation_limit
        start = time.perf_counter()
        result = render_all_cameras(
            stage=_make_sample_attribute_probe_stage(),
            image_width=image_size,
            image_height=image_size,
            cameras=["/Camera"],
            frames="0",
            log_level=log_level,
            ovrtx_venv_dir=active_venv_dir,
            render_mode=variant.render_mode,
            num_sensor_updates=variant.num_sensor_updates,
            rtx_pt_samples_per_pixel=pt_samples,
            rtx_rt_accumulation_limit=rt_accumulation,
        )
        elapsed = time.perf_counter() - start
        if result["successful_cameras"] != 1:
            raise RuntimeError(
                f"Probe variant {variant.name} did not render successfully: {result}"
            )
        image = result["results"][0]["images"][0]
        images[variant.name] = image
        rendered_variants.append(
            {
                **variant.to_result_dict(),
                "elapsed_s": elapsed,
                "metrics": _probe_image_metrics(image),
            }
        )

    baseline_high_name = f"baseline_steps_{baseline_updates}"
    comparisons = {
        "num_sensor_updates_baseline": {
            "left": "baseline_steps_1",
            "right": baseline_high_name,
            "mean_abs_rgb_diff": _probe_mean_abs_rgb_diff(
                images["baseline_steps_1"],
                images[baseline_high_name],
            ),
        },
        "pt_samples_per_pixel_low_vs_high": {
            "left": f"pt_spp_{low_value}",
            "right": f"pt_spp_{high_value}",
            "mean_abs_rgb_diff": _probe_mean_abs_rgb_diff(
                images[f"pt_spp_{low_value}"],
                images[f"pt_spp_{high_value}"],
            ),
        },
        "rt_accumulation_limit_low_vs_high": {
            "left": f"rt_accum_{low_value}",
            "right": f"rt_accum_{high_value}",
            "mean_abs_rgb_diff": _probe_mean_abs_rgb_diff(
                images[f"rt_accum_{low_value}"],
                images[f"rt_accum_{high_value}"],
            ),
        },
    }

    return {
        "probe": "ovrtx_sample_attributes",
        "ovrtx_version": ovrtx_version,
        "ovrtx_python": ovrtx_python,
        "platform": sys.platform,
        "gpu": _probe_gpu_summary(),
        "ovrtx_version_warning": ovrtx_version_warning,
        "scene": "in-memory lit cube, camera /Camera, frame 0",
        "image_size": image_size,
        "variants": rendered_variants,
        "comparisons": comparisons,
        "decision_hint": (
            "Retain num_sensor_updates unless the sample-attribute low/high "
            "comparisons show a measurable effect on OVRTX 0.3 while the "
            "num_sensor_updates baseline confirms this scene responds to "
            "accumulation."
        ),
    }


def _main(argv: list[str] | None = None) -> int:
    """Run small maintenance commands for the isolated OVRTX runtime."""
    import argparse

    parser = argparse.ArgumentParser(description="OVRTX renderer maintenance")
    parser.add_argument(
        "--provision-only",
        action="store_true",
        help="create or validate the isolated OVRTX Python environment and exit",
    )
    parser.add_argument(
        "--probe-sample-attributes",
        action="store_true",
        help=("render low/high RTX sample-attribute variants and emit JSON metrics"),
    )
    parser.add_argument(
        "--ovrtx-venv-dir",
        type=Path,
        default=None,
        help="override the isolated OVRTX runtime directory",
    )
    parser.add_argument(
        "--probe-image-size",
        type=int,
        default=96,
        help="square image size for --probe-sample-attributes",
    )
    parser.add_argument(
        "--probe-low-value",
        type=int,
        default=1,
        help="low samplesPerPixel/accumulationLimit value for the probe",
    )
    parser.add_argument(
        "--probe-high-value",
        type=int,
        default=512,
        help="high samplesPerPixel/accumulationLimit value for the probe",
    )
    parser.add_argument(
        "--probe-baseline-updates",
        type=int,
        default=8,
        help="high num_sensor_updates value for the accumulation baseline",
    )
    parser.add_argument(
        "--log-level",
        default="warn",
        choices=["error", "warn", "info", "debug"],
        help="OVRTX log level for maintenance commands",
    )
    args = parser.parse_args(argv)

    if args.provision_only:
        ovrtx_python = _get_ovrtx_python(args.ovrtx_venv_dir)
        sys.stdout.write(f"OvRTX Python ready: {ovrtx_python}\n")
        return 0

    if args.probe_sample_attributes:
        result = _run_sample_attribute_probe(
            image_size=args.probe_image_size,
            low_value=args.probe_low_value,
            high_value=args.probe_high_value,
            baseline_updates=args.probe_baseline_updates,
            log_level=args.log_level,
            ovrtx_venv_dir=args.ovrtx_venv_dir,
        )
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        return 0

    parser.error("no action requested")
    return 2  # pragma: no cover - argparse.error exits before returning


if __name__ == "__main__":
    raise SystemExit(_main())
