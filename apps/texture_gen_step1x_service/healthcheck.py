#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Healthcheck helper for the Step1X texture-generation service."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import sys
from pathlib import Path

_PREFLIGHT_MARKER = Path(
    os.environ.get(
        "TEXTURE_STEP1X_PREFLIGHT_MARKER",
        "/var/texture-agent/sessions/step1x_runtime_preflight.ok",
    )
)
_VALID_RUNTIME_PROFILE_SETS = (
    ("texture-step1x-core",),
    ("texture-step1x-core", "texture-material-anything"),
    ("texture-step1x-core", "texture-swin2sr"),
    ("texture-full-pbr-upscale",),
)


class _RuntimeMarkerError(ValueError):
    """A present runtime marker is malformed or incomplete."""


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _configured_runtime_python() -> str | None:
    configured = os.environ.get("TEXTURE_STEP1X_PYTHON", "").strip()
    if configured:
        return configured
    runtime_dir = Path(
        os.environ.get("TEXTURE_STEP1X_RUNTIME_DIR", "/opt/texture-editing")
    )
    for relative_path in (".venv_gen/bin/python", ".venv/bin/python"):
        candidate = runtime_dir / relative_path
        if candidate.exists():
            return str(candidate)
    return None


def _runtime_marker_profiles() -> tuple[str, ...] | None:
    runtime_dir = Path(
        os.environ.get("TEXTURE_STEP1X_RUNTIME_DIR", "/opt/texture-editing")
    )
    marker_path = runtime_dir / ".texture-agent-runtime.json"
    try:
        marker_text = marker_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _RuntimeMarkerError(f"STEP1X_RUNTIME_MARKER_UNREADABLE: {exc}") from exc
    try:
        marker = json.loads(marker_text)
    except json.JSONDecodeError as exc:
        raise _RuntimeMarkerError(
            f"STEP1X_RUNTIME_MARKER_INVALID: invalid JSON: {exc}"
        ) from exc
    if not isinstance(marker, dict):
        raise _RuntimeMarkerError(
            "STEP1X_RUNTIME_MARKER_INVALID: expected a JSON object"
        )
    profiles = marker.get("runtime_profiles")
    if "runtime_profiles" in marker:
        if not (
            isinstance(profiles, list)
            and profiles
            and all(
                isinstance(profile, str) and profile.strip() for profile in profiles
            )
        ):
            raise _RuntimeMarkerError(
                "STEP1X_RUNTIME_PROFILES_INVALID_IN_MARKER: runtime_profiles must "
                "be a non-empty list of non-empty strings"
            )
        return tuple(profile.strip() for profile in profiles)
    if marker.get("runtime_source") == "compose_managed":
        raise _RuntimeMarkerError(
            "STEP1X_RUNTIME_PROFILES_MISSING_FROM_MARKER: compose-managed "
            "marker has no runtime_profiles"
        )
    profile = marker.get("runtime_profile")
    return (profile,) if isinstance(profile, str) and profile else None


def _runtime_probe_pythonpath(profiles: tuple[str, ...]) -> str:
    runtime_dir = Path(
        os.environ.get("TEXTURE_STEP1X_RUNTIME_DIR", "/opt/texture-editing")
    )
    paths = [
        runtime_dir / "src",
        runtime_dir / "third_party" / "Step1X-3D",
    ]
    if (
        "texture-material-anything" in profiles
        or "texture-full-pbr-upscale" in profiles
    ):
        paths.append(runtime_dir / "third_party" / "MaterialAnything")
    configured = os.environ.get("PYTHONPATH", "")
    if configured:
        paths.extend(Path(value) for value in configured.split(os.pathsep) if value)
    return os.pathsep.join(str(path) for path in paths)


def _selected_runtime_profiles() -> tuple[tuple[str, ...], str]:
    # Validate any present marker even when the operator also supplied an
    # explicit profile. A stale/corrupt success marker must never be bypassed.
    marker_profiles = _runtime_marker_profiles()
    configured = os.environ.get("TEXTURE_STEP1X_RUNTIME_PROFILES", "").strip()
    if configured:
        return tuple(profile.strip() for profile in configured.split(",")), "explicit"
    legacy_configured = os.environ.get("TEXTURE_STEP1X_RUNTIME_PROFILE", "").strip()
    if legacy_configured:
        return (legacy_configured,), "explicit"
    if marker_profiles:
        return marker_profiles, "runtime_marker"
    skip_ma_value = os.environ.get("TEXTURE_STEP1X_SKIP_MA")
    skip_ma = True if skip_ma_value is None else _truthy(skip_ma_value)
    require_upscaler = _truthy(os.environ.get("TEXTURE_STEP1X_REQUIRE_UPSCALER"))
    if not skip_ma and require_upscaler:
        return ("texture-full-pbr-upscale",), "legacy_flags"
    if not skip_ma:
        return (
            "texture-step1x-core",
            "texture-material-anything",
        ), "legacy_flags"
    if require_upscaler:
        return ("texture-step1x-core", "texture-swin2sr"), "legacy_flags"
    return ("texture-step1x-core",), "legacy_flags"


def _runtime_fingerprint() -> str | None:
    python_executable = _configured_runtime_python()
    if not python_executable:
        return None

    profiles, profile_source = _selected_runtime_profiles()
    return "\n".join(
        [
            python_executable,
            os.environ.get("LD_LIBRARY_PATH", ""),
            _runtime_probe_pythonpath(profiles),
            os.environ.get("HF_HOME", ""),
            os.environ.get("HF_HUB_CACHE", ""),
            os.environ.get("TEXTURE_STEP1X_HF_REVISION", ""),
            os.environ.get("TEXTURE_SDXL_BASE_REVISION", ""),
            os.environ.get("TEXTURE_SDXL_VAE_REVISION", ""),
            os.environ.get("TEXTURE_STEP1X_SKIP_MA", ""),
            os.environ.get("TEXTURE_STEP1X_REQUIRE_UPSCALER", ""),
            ",".join(profiles),
            profile_source,
        ]
    )


def _runtime_preflight_marker_matches() -> bool:
    try:
        fingerprint = _runtime_fingerprint()
    except _RuntimeMarkerError:
        return False
    if fingerprint is None:
        return False
    try:
        return _PREFLIGHT_MARKER.read_text(encoding="utf-8") == fingerprint
    except OSError:
        return False


def _runtime_import_preflight() -> bool:
    python_executable = _configured_runtime_python()
    if not python_executable:
        print(
            "Step1X runtime import preflight failed: TEXTURE_STEP1X_PYTHON "
            "is unset and no mounted .venv_gen/.venv Python was found.",
            file=sys.stderr,
        )
        return False

    try:
        profiles, _profile_source = _selected_runtime_profiles()
    except _RuntimeMarkerError as exc:
        print(f"Step1X runtime import preflight failed: {exc}", file=sys.stderr)
        return False
    if profiles not in _VALID_RUNTIME_PROFILE_SETS:
        print(
            "Step1X runtime import preflight failed: unsupported "
            f"TEXTURE_STEP1X_RUNTIME_PROFILES={','.join(profiles)!r}.",
            file=sys.stderr,
        )
        return False

    fingerprint = _runtime_fingerprint()
    if fingerprint is None:
        return False
    if _runtime_preflight_marker_matches():
        return True

    script = """
import importlib
import importlib.util
import os
import sys

import cupy as cp
import torch
import torchvision

profiles = set(os.environ["TEXTURE_STEP1X_RUNTIME_PROFILES"].split(","))
modules = [
    "diffusers",
    "imageio",
    "pxr",
    "pytorch3d",
    "step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline",
    "transformers",
    "trimesh",
    "xatlas",
]
if "texture-material-anything" in profiles or "texture-full-pbr-upscale" in profiles:
    modules.extend([
        "accelerate",
        "cv2",
        "einops",
        "kaolin",
        "scipy",
        "scripts.generate_texture_pbr_3d",
        "skimage",
    ])
if "texture-swin2sr" in profiles or "texture-full-pbr-upscale" in profiles:
    modules.append("texture_edit.upscaler")

for module_name in modules:
    importlib.import_module(module_name)

forbidden = {
    "GPUtil",
    "antlr4",
    "crc32c",
    "deepspeed",
    "easydict",
    "iopath",
    "mathutils",
    "oci",
    "paramiko",
    "plyfile",
    "pymeshlab",
    "streaming",
}
unexpected = sorted(
    module_name
    for module_name in forbidden
    if importlib.util.find_spec(module_name) is not None
)
if unexpected:
    raise SystemExit("pruned modules installed: " + ", ".join(unexpected))
if any(
    name == "step1x3d_geometry" or name.startswith("step1x3d_geometry.")
    for name in sys.modules
):
    raise SystemExit("Step1X texture preflight imported step1x3d_geometry")

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")

cp.cuda.runtime.runtimeGetVersion()
cp.cuda.nvrtc.getVersion()
values = cp.asarray([1, 2, 3])
if int(cp.sum(values).get()) != 6:
    raise SystemExit("CuPy kernel check returned an unexpected result")
"""
    try:
        timeout = int(os.environ.get("TEXTURE_STEP1X_PREFLIGHT_TIMEOUT_SEC", "180"))
    except ValueError:
        timeout = 180
    result = subprocess.run(
        [python_executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            **os.environ,
            "TEXTURE_STEP1X_RUNTIME_PROFILES": ",".join(profiles),
            "PYTHONPATH": _runtime_probe_pythonpath(profiles),
        },
    )
    if result.returncode != 0:
        print(
            "Step1X runtime import preflight failed. "
            "Check TEXTURE_STEP1X_PYTHON and LD_LIBRARY_PATH.",
            file=sys.stderr,
        )
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return False

    try:
        _PREFLIGHT_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _PREFLIGHT_MARKER.write_text(fingerprint, encoding="utf-8")
    except OSError:
        pass
    return True


def _http_timeout(require_ready: bool) -> int:
    specific_name = (
        "TEXTURE_STEP1X_READINESS_HTTP_TIMEOUT_SEC"
        if require_ready
        else "TEXTURE_STEP1X_LIVENESS_HTTP_TIMEOUT_SEC"
    )
    fallback = os.environ.get("TEXTURE_STEP1X_HEALTHCHECK_HTTP_TIMEOUT_SEC")
    default = "180" if require_ready else "5"
    raw = os.environ.get(specific_name) or fallback or default
    try:
        timeout = int(raw)
    except ValueError:
        timeout = int(default)
    return max(1, timeout)


def _http_health(path: str, *, require_ready: bool) -> tuple[int, dict[str, object]]:
    conn = http.client.HTTPConnection(
        "localhost",
        8000,
        timeout=_http_timeout(require_ready=require_ready),
    )
    conn.request("GET", path)
    response = conn.getresponse()
    payload = json.loads(response.read() or b"{}")
    return response.status, payload


def _healthy(require_ready: bool) -> bool:
    runtime_imports = _truthy(
        os.environ.get("TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS")
    )
    status, payload = _http_health(
        "/health" if require_ready else "/livez",
        require_ready=require_ready,
    )
    if not 200 <= status < 300:
        return False
    if not require_ready:
        return True
    if payload.get("ready") is not True:
        return False
    try:
        selected_profiles, _selection_source = _selected_runtime_profiles()
    except _RuntimeMarkerError:
        return False
    capabilities = payload.get("capabilities")
    external_runtime = (
        capabilities.get("external_runtime") if isinstance(capabilities, dict) else None
    )
    runtime_profiles = (
        external_runtime.get("runtime_profiles")
        if isinstance(external_runtime, dict)
        else None
    )
    reported_profiles = (
        runtime_profiles.get("selected") if isinstance(runtime_profiles, dict) else None
    )
    if reported_profiles != list(selected_profiles):
        return False
    if runtime_imports:
        return _runtime_import_preflight()
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--liveness",
        action="store_true",
        help="Only require a 2xx health response; ignore the ready field.",
    )
    args = parser.parse_args()

    try:
        return 0 if _healthy(require_ready=not args.liveness) else 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
