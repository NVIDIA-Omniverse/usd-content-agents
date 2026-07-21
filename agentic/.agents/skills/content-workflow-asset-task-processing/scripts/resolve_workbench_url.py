#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the immutable Workbench URL from a scene workflow request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


class WorkbenchURLResolutionError(ValueError):
    """Raised when a frozen scene request cannot provide a safe root URL."""


def resolve_workbench_url(request_path: Path) -> str:
    """Return ``runtime.workbench_url`` exactly as frozen in ``request.json``."""
    if not request_path.is_file():
        raise WorkbenchURLResolutionError(
            f"Scene run request does not exist: {request_path}. Point RUN at a "
            "prepared scene workflow directory containing request.json."
        )

    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkbenchURLResolutionError(
            f"Scene run request is not readable valid JSON: {request_path}: {exc}"
        ) from exc

    if not isinstance(request, dict):
        raise WorkbenchURLResolutionError(
            f"Scene run request must contain a JSON object: {request_path}"
        )
    runtime = request.get("runtime")
    if not isinstance(runtime, dict):
        raise WorkbenchURLResolutionError(
            f"Scene run request is missing the runtime object: {request_path}. "
            "Recreate the run with content-workflow-cli scene run."
        )
    workbench_url = runtime.get("workbench_url")
    if not isinstance(workbench_url, str) or not workbench_url:
        raise WorkbenchURLResolutionError(
            "Scene run request runtime.workbench_url must be a non-empty string: "
            f"{request_path}. Recreate the run with content-workflow-cli scene run."
        )

    parsed = urlparse(workbench_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise WorkbenchURLResolutionError(
            "Scene run request runtime.workbench_url must be an absolute http(s) "
            f"URL: {request_path}"
        )
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise WorkbenchURLResolutionError(
            "Scene run request runtime.workbench_url must be a root URL without "
            f"path, params, query, or fragment: {request_path}"
        )
    return workbench_url


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print runtime.workbench_url from a frozen scene request."
    )
    parser.add_argument("request", type=Path, help="Path to RUN/request.json.")
    args = parser.parse_args()
    try:
        workbench_url = resolve_workbench_url(args.request)
    except WorkbenchURLResolutionError as exc:
        parser.error(str(exc))
    print(workbench_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
