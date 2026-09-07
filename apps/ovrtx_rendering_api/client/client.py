# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal client for the OVRTX Rendering API — used by CI smoke tests.

Encodes a local USD file as a ``data:`` URI and POSTs it to ``/render``, then
verifies a non-empty ``images`` map comes back. It can also package the same
scene as USDZ and exercise the usd-cli protocol-v3 ``/render/upload`` route.
Works identically against a local docker container and an NVCF function URL.

Usage:
    python apps/ovrtx_rendering_api/client/client.py \\
        --base-url http://localhost:8000 \\
        --usd apps/ovrtx_rendering_api/tests/renders/smoke_cube.usda
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import sys
import zipfile
from typing import Any

import requests

RETRYABLE_EXIT_CODE = 75


def _request_headers(token: str | None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if version_id := os.getenv("NVCF_INVOKE_VERSION_ID"):
        headers["Function-Version-Id"] = version_id
    return headers


def _raise_for_status(response: requests.Response) -> None:
    if 500 <= response.status_code < 600:
        try:
            body = response.json()
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        retryable_body = body.get("retryable") is True
        retryable_header = response.headers.get("Retry-After") is not None
        renderer_initializing = (
            response.status_code == 503
            and body.get("detail") == "renderer is not ready"
        )
        gateway_timeout = response.status_code == 504
        if (
            retryable_body
            or retryable_header
            or renderer_initializing
            or gateway_timeout
        ):
            reason = (
                body.get("error")
                or body.get("detail")
                or body.get("error_code")
                or "service is temporarily unavailable"
            )
            print(f"✗ retryable service response: {reason}", file=sys.stderr)
            raise SystemExit(RETRYABLE_EXIT_CODE)
    response.raise_for_status()


def _encode_usd_as_data_uri(path: pathlib.Path) -> str:
    """Base64-encode a USD file into a data: URI the renderer understands."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:application/octet-stream;base64,{encoded}"


def _build_request(usd_data_uri: str) -> dict[str, Any]:
    return {
        "url": usd_data_uri,
        "force_render": True,
        "render_settings": {
            "camera_paths": ["/World/Camera"],
            "frame_range": {"start": 0, "end": 0},
            "camera_parameters": {"width": 256, "height": 256},
            "sensors": None,
            "apply_background_mask": False,
        },
    }


def _validate_png(image: bytes, expected_size: tuple[int, int]) -> None:
    """Validate the decoded image format, dimensions, and file integrity."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(image)) as rendered:
            if rendered.format != "PNG":
                raise SystemExit("✗ /render/upload returned a non-PNG image")
            if rendered.size != expected_size:
                raise SystemExit(
                    "✗ /render/upload returned PNG dimensions "
                    f"{rendered.width}x{rendered.height}; expected "
                    f"{expected_size[0]}x{expected_size[1]}"
                )
            rendered.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise SystemExit("✗ /render/upload returned invalid PNG image data") from exc


def health_check(base_url: str, token: str | None, timeout: float) -> None:
    headers = _request_headers(token)
    r = requests.get(f"{base_url.rstrip('/')}/health", headers=headers, timeout=timeout)
    _raise_for_status(r)
    body = r.json()
    print(f"health: {body}")
    if not body.get("gpu_initialized", False):
        print(
            "✗ /health returned but gpu_initialized=false — renderer did not start",
            file=sys.stderr,
        )
        raise SystemExit(RETRYABLE_EXIT_CODE)
    print("✓ /health passed, gpu_initialized=true")


def render_smoke(
    base_url: str, usd_path: pathlib.Path, token: str | None, timeout: float
) -> None:
    data_uri = _encode_usd_as_data_uri(usd_path)
    payload = _build_request(data_uri)
    headers = _request_headers(token)
    headers["Content-Type"] = "application/json"

    print(f"POST /render with {usd_path.name} ({usd_path.stat().st_size} bytes)")
    r = requests.post(
        f"{base_url.rstrip('/')}/render",
        data=json.dumps(payload),
        headers=headers,
        timeout=timeout,
    )
    _raise_for_status(r)
    body = r.json()
    status = body.get("status", "unknown")
    error = body.get("error")
    images = body.get("images", {})

    if status != "success":
        raise SystemExit(f"✗ /render status={status} error={error}")
    if not images:
        raise SystemExit("✗ /render returned empty images map")

    # Structure: images[frame][camera][sensor] = base64
    total_images = sum(
        1 for frame in images.values() for cam in frame.values() for _ in cam.values()
    )
    print(f"✓ /render produced {total_images} image(s)")


def protocol_v3_render_smoke(
    base_url: str, usd_path: pathlib.Path, token: str | None, timeout: float
) -> None:
    """Package one USD layer and verify the multipart protocol-v3 render path."""
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(usd_path.name, usd_path.read_bytes())
    payload = bundle.getvalue()
    params = {
        "cameras": ["/World/Camera"],
        "image_width": 64,
        "image_height": 64,
        "mode": "fast",
        "compression": "none",
        "frames": [0.0],
    }

    print(f"POST /render/upload with {usd_path.name} ({len(payload)} bytes USDZ)")
    response = requests.post(
        f"{base_url.rstrip('/')}/render/upload",
        files={"file": ("smoke_scene.usdz", payload, "application/octet-stream")},
        data={"params": json.dumps(params)},
        headers=_request_headers(token),
        timeout=timeout,
    )
    _raise_for_status(response)
    results = response.json().get("results")
    if not isinstance(results, list) or not results:
        raise SystemExit("✗ /render/upload returned no protocol-v3 results")

    item = results[0]
    if item.get("camera") != "/World/Camera":
        raise SystemExit("✗ /render/upload returned the wrong camera")
    encoded = item.get("image_base64")
    try:
        image = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise SystemExit("✗ /render/upload returned invalid image_base64") from exc
    if not image:
        raise SystemExit("✗ /render/upload returned an empty image")
    _validate_png(image, (params["image_width"], params["image_height"]))
    if item.get("ovrtx_render_mode") not in {"rt1", "rt2", "pt"}:
        raise SystemExit("✗ /render/upload omitted OVRTX render mode evidence")
    if not isinstance(item.get("ovrtx_num_sensor_updates"), int):
        raise SystemExit("✗ /render/upload omitted sensor-update evidence")
    if not item.get("active_aov"):
        raise SystemExit("✗ /render/upload omitted active-AOV evidence")

    print("✓ /render/upload produced a protocol-v3 image with OVRTX evidence")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True, help="Service base URL")
    p.add_argument(
        "--usd",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent
        / "tests"
        / "renders"
        / "smoke_cube.usda",
        help="Path to a small USD file to render",
    )
    p.add_argument("--token", default=None, help="Bearer token (for NVCF)")
    p.add_argument("--skip-health", action="store_true", help="Skip /health probe")
    p.add_argument(
        "--require-protocol-v3",
        action="store_true",
        help="Also require a successful usd-cli protocol-v3 multipart render",
    )
    p.add_argument(
        "--timeout", type=float, default=300.0, help="Per-request HTTP timeout"
    )
    args = p.parse_args()

    if not args.usd.exists():
        print(f"✗ USD fixture not found: {args.usd}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_health:
        health_check(args.base_url, args.token, args.timeout)

    render_smoke(args.base_url, args.usd, args.token, args.timeout)
    if args.require_protocol_v3:
        protocol_v3_render_smoke(args.base_url, args.usd, args.token, args.timeout)
    print("✓ all checks passed")


if __name__ == "__main__":
    main()
