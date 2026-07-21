# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test a running Step1X Texture Variation API service."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=600)
    parser.add_argument("--allow-not-ready", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--require-orm",
        action="store_true",
        help="Require full PBR output with generated_textures.orm and maps.orm.",
    )
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    health = _get_json(f"{endpoint}/health")
    print(json.dumps({"health": health}, indent=2))
    if not health.get("ready") and not args.allow_not_ready:
        print("Service is not ready; set --allow-not-ready to only check health.")
        return 2
    if not health.get("ready"):
        return 0
    if args.require_gpu and health.get("gpu_available") is not True:
        print("Service did not report gpu_available=true.")
        return 7

    body = json.loads(args.request.read_text(encoding="utf-8"))
    created = _request_json(
        f"{endpoint}/v1/texture-variations",
        method="POST",
        body=body,
    )
    job_id = created["job_id"]
    print(json.dumps({"created": created}, indent=2))

    deadline = time.monotonic() + args.timeout_sec
    status = created
    while status.get("status") in {"queued", "processing"}:
        if time.monotonic() > deadline:
            print(f"Timed out waiting for {job_id}.")
            return 3
        time.sleep(2)
        status = _get_json(f"{endpoint}/v1/texture-variations/{job_id}")
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "status": status.get("status"),
                    "progress": status.get("progress"),
                    "message": status.get("message"),
                },
                indent=2,
            )
        )

    print(json.dumps({"final": status}, indent=2))
    if status.get("status") != "completed":
        return 4

    result = status.get("result") or {}
    maps = result.get("maps") or {}
    albedo = maps.get("albedo") or {}
    uri = albedo.get("uri")
    if not uri:
        print("Completed response did not include maps.albedo.uri.")
        return 5
    if not _artifact_visible(uri):
        print(f"maps.albedo.uri is not visible from this process: {uri}")
        return 6
    if not albedo.get("width") or not albedo.get("height"):
        print("maps.albedo is missing dimensions.")
        return 8
    if albedo.get("colorspace") != "srgb":
        print(f"maps.albedo colorspace should be srgb: {albedo.get('colorspace')}")
        return 9

    generated = result.get("generated_textures") or {}
    diagnostics = result.get("diagnostics") or []
    degraded = set((result.get("metadata") or {}).get("degraded_channels") or [])
    if not generated.get("normal") and "normal" not in degraded:
        print("normal is absent but metadata.degraded_channels does not include it.")
        return 10
    if not generated.get("orm") and "orm" not in degraded:
        print("orm is absent but metadata.degraded_channels does not include it.")
        return 11
    if degraded and not any(
        item.get("code") == "STEP1X_MAPS_DEGRADED" for item in diagnostics
    ):
        print("degraded output is missing STEP1X_MAPS_DEGRADED diagnostic.")
        return 12
    if args.require_orm:
        orm = maps.get("orm") or {}
        if not generated.get("orm"):
            print("--require-orm set but generated_textures.orm is missing.")
            return 13
        if not orm.get("uri"):
            print("--require-orm set but maps.orm.uri is missing.")
            return 14
        if not _artifact_visible(orm["uri"]):
            print(f"maps.orm.uri is not visible from this process: {orm['uri']}")
            return 15
        if orm.get("packing") != "occlusion_roughness_metallic":
            print(f"maps.orm.packing is invalid: {orm.get('packing')}")
            return 16
        if "orm" in degraded:
            print("--require-orm set but metadata.degraded_channels includes orm.")
            return 17
    return 0


def _get_json(url: str) -> dict[str, Any]:
    return _request_json(url, method="GET")


def _request_json(
    url: str,
    *,
    method: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {url} failed with HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc
    return json.loads(payload)


def _artifact_visible(uri: str) -> bool:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path).is_file() and Path(parsed.path).stat().st_size > 0
    if parsed.scheme in {"http", "https"}:
        try:
            with urlopen(uri, timeout=30) as response:
                return response.status < 400
        except (HTTPError, URLError):
            return False
    return False


if __name__ == "__main__":
    sys.exit(main())
