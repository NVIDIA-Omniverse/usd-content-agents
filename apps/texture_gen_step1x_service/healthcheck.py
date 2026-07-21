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


def _runtime_fingerprint() -> str | None:
    python_executable = _configured_runtime_python()
    if not python_executable:
        return None

    return "\n".join(
        [
            python_executable,
            os.environ.get("LD_LIBRARY_PATH", ""),
            os.environ.get("PYTHONPATH", ""),
            os.environ.get("TEXTURE_STEP1X_SKIP_MA", ""),
        ]
    )


def _runtime_preflight_marker_matches() -> bool:
    fingerprint = _runtime_fingerprint()
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

    fingerprint = _runtime_fingerprint()
    if fingerprint is None:
        return False
    if _runtime_preflight_marker_matches():
        return True

    script = """
import os

import cupy as cp
import torch

def _truthy(value):
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}

if not _truthy(os.environ.get("TEXTURE_STEP1X_SKIP_MA")):
    import pymeshlab

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
