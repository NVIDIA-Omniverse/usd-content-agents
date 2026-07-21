# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from PIL import Image

DEFAULT_SOURCE_ASSET_URI = "file:///work/ladder/prepared_input.usd"
DEFAULT_PRIM_PATH = "/RootNode/SM_Ladder_A/SM_Ladder_A_Aluminum_0"
DEFAULT_MATERIAL_NAME = "Aluminum_Matte"
DEFAULT_MATERIAL_PATH = "/RootNode/Looks/Aluminum_Matte"
DEFAULT_PROMPT = "deterministic scuffed aluminum projection"
DEFAULT_SEED = 11631
DEFAULT_TEXTURE_SIZE = 16

FULL_PBR_VARIANTS = {"success", "success_full_pbr"}
ALBEDO_ONLY_VARIANTS = {
    "albedo_only",
    "albedo_only_degraded",
    "degraded_albedo_only",
}
LOW_COVERAGE_VARIANTS = {"low_coverage", "low_coverage_warning"}
DEGRADED_LOW_COVERAGE_VARIANTS = {"degraded_low_coverage"}
GEOMETRY_RETURN_VARIANTS = {"geometry_return", "geometry_return_ignored"}
MISSING_ALBEDO_VARIANTS = {"missing_albedo", "failure_missing_albedo"}
BAD_URI_VARIANTS = {"bad_uri", "failure_bad_uri"}


class _ProjectionHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def build_projection_request(
    variant: str = "success_full_pbr",
    *,
    texture_size: int = DEFAULT_TEXTURE_SIZE,
    seed: int = DEFAULT_SEED,
    source_asset_uri: str = DEFAULT_SOURCE_ASSET_URI,
    text_prompt: str = DEFAULT_PROMPT,
    material_name: str = DEFAULT_MATERIAL_NAME,
    material_path: str = DEFAULT_MATERIAL_PATH,
    prim_paths: list[str] | None = None,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized projection request for client/service smoke tests."""
    return {
        "source_asset_uri": source_asset_uri,
        "target": {
            "material_name": material_name,
            "material_path": material_path,
            "prim_paths": prim_paths or [DEFAULT_PRIM_PATH],
            "mode": "per_material",
            "strict_scope": True,
        },
        "conditioning": {
            "text_prompt": text_prompt,
            "reference_image_uris": [],
            "turntable_video_uri": None,
            "multiview_image_uris": [],
        },
        "configuration": {
            "strength": 0.8,
            "seed": seed,
            "variant_name": material_name,
            "engine": "fake_projection",
            "texture_size": texture_size,
            "custom_parameters": {"variant": variant},
        },
        "capabilities": capabilities or _capabilities(),
    }


def submit_projection_request(
    endpoint_url: str,
    request: dict[str, Any],
    *,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Submit a projection request to a running fake backend."""
    response = httpx.post(
        f"{endpoint_url.rstrip('/')}/v1/texture-variations",
        json=request,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    assert isinstance(data, dict)
    return data


@pytest.fixture
def fake_projection_backend(tmp_path: Path) -> Iterator[FakeProjectionBackend]:
    """Run the deterministic fake projection backend for one pytest test."""
    with FakeProjectionBackend(tmp_path / "fake_projection_backend") as backend:
        yield backend


@pytest.fixture
def projection_request_factory() -> Callable[..., dict[str, Any]]:
    """Return the reusable request builder without forcing HTTP startup."""
    return build_projection_request


@pytest.fixture
def projection_submitter() -> Callable[..., dict[str, Any]]:
    """Return the reusable HTTP submit helper for client/service smoke tests."""
    return submit_projection_request


class FakeProjectionBackend:
    """HTTP fake implementing the issue #116 normalized projection contract."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self._state_lock = threading.Lock()
        self._server: _ProjectionHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def endpoint_url(self) -> str:
        if self._server is None:
            raise RuntimeError("FakeProjectionBackend is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def start(self) -> FakeProjectionBackend:
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/v1/texture-variations":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(body or "{}")
                    if not isinstance(payload, dict):
                        raise ValueError("Request body must be a JSON object")
                except Exception as exc:
                    self.send_error(400, explain=str(exc))
                    return
                self._send_json(backend._create_job(payload))

            def do_GET(self) -> None:  # noqa: N802
                prefix = "/v1/texture-variations/"
                if not self.path.startswith(prefix):
                    self.send_error(404)
                    return
                job_id = self.path.removeprefix(prefix)
                with backend._state_lock:
                    response = backend.jobs.get(job_id)
                if response is None:
                    self.send_error(404)
                    return
                self._send_json(response)

            def do_DELETE(self) -> None:  # noqa: N802
                prefix = "/v1/texture-variations/"
                if not self.path.startswith(prefix):
                    self.send_error(404)
                    return
                job_id = self.path.removeprefix(prefix)
                with backend._state_lock:
                    backend.jobs.pop(job_id, None)
                self.send_response(204)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, response: dict[str, Any]) -> None:
                body = json.dumps(response, sort_keys=True).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = _ProjectionHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> FakeProjectionBackend:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def _create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = payload.get("configuration") or {}
        custom = config.get("custom_parameters") or {}
        variant = str(custom.get("variant") or "success_full_pbr")
        with self._state_lock:
            self.requests.append(payload)
            job_id = f"fake-{len(self.jobs) + 1:04d}-{variant}"
            self.jobs[job_id] = {}
        seed = _int_or_default(config.get("seed"), DEFAULT_SEED)
        size = _positive_int_or_default(
            config.get("texture_size"), DEFAULT_TEXTURE_SIZE
        )
        source_asset_uri = str(payload.get("source_asset_uri") or "")

        if variant in BAD_URI_VARIANTS or not _valid_source_uri(source_asset_uri):
            response = self._bad_uri_failure(
                job_id=job_id,
                variant=variant,
                source_asset_uri=source_asset_uri,
                seed=seed,
                size=size,
            )
        elif variant in MISSING_ALBEDO_VARIANTS:
            response = self._missing_albedo_failure(
                job_id=job_id,
                variant=variant,
                seed=seed,
                size=size,
                source_asset_uri=source_asset_uri,
            )
        else:
            response = self._success(
                job_id=job_id,
                variant=variant,
                seed=seed,
                size=size,
                source_asset_uri=source_asset_uri,
            )

        with self._state_lock:
            self.jobs[job_id] = response
        return response

    def _success(
        self,
        *,
        job_id: str,
        variant: str,
        seed: int,
        size: int,
        source_asset_uri: str,
    ) -> dict[str, Any]:
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        albedo = _write_pattern(
            job_dir / "albedo.png", size, channel="albedo", seed=seed
        )
        maps: dict[str, dict[str, Any]] = {
            "albedo": _map(albedo, size=size, colorspace="srgb")
        }
        generated_textures: dict[str, str | None] = {
            "albedo": albedo.as_uri(),
            "normal": None,
            "orm": None,
        }
        diagnostics: list[dict[str, Any]] = []
        degraded_channels: list[str] = []
        geometry: list[dict[str, str]] = []
        capabilities = _capabilities()
        coverage = {"target_coverage": 0.97, "uv_island_coverage": 0.94}

        full_pbr = variant in FULL_PBR_VARIANTS
        albedo_only = variant in ALBEDO_ONLY_VARIANTS
        low_coverage = variant in LOW_COVERAGE_VARIANTS
        degraded_low_coverage = variant in DEGRADED_LOW_COVERAGE_VARIANTS
        geometry_return = variant in GEOMETRY_RETURN_VARIANTS

        if not (
            full_pbr
            or albedo_only
            or low_coverage
            or degraded_low_coverage
            or geometry_return
        ):
            full_pbr = True

        if full_pbr or low_coverage or geometry_return:
            normal = _write_pattern(
                job_dir / "normal.png", size, channel="normal", seed=seed
            )
            orm = _write_pattern(job_dir / "orm.png", size, channel="orm", seed=seed)
            maps["normal"] = _map(
                normal, size=size, colorspace="raw", packing="tangent_space_rgb"
            )
            maps["orm"] = _map(
                orm,
                size=size,
                colorspace="raw",
                packing="r=occlusion,g=roughness,b=metalness",
            )
            generated_textures["normal"] = normal.as_uri()
            generated_textures["orm"] = orm.as_uri()

        if albedo_only or degraded_low_coverage:
            capabilities["normal_map"] = False
            capabilities["orm"] = False
            degraded_channels = ["normal", "orm"]
            diagnostics.append(
                _diagnostic(
                    "BACKEND_MAP_MISSING",
                    severity="warning",
                    message="Backend returned albedo only; normal and ORM are missing.",
                    recommended_action=(
                        "Continue with albedo-only output or choose a PBR-capable backend."
                    ),
                    details={"missing_maps": ["normal", "orm"]},
                )
            )

        if low_coverage or degraded_low_coverage:
            coverage = {"target_coverage": 0.41, "uv_island_coverage": 0.38}
            diagnostics.append(
                _diagnostic(
                    "BACKEND_LOW_COVERAGE",
                    severity="warning",
                    message="Backend reported low target coverage for selected scope.",
                    recommended_action=(
                        "Inspect backend coverage artifacts and retry with a clearer target."
                    ),
                    details={"target_coverage": 0.41, "threshold": 0.75},
                )
            )

        if geometry_return:
            replacement = job_dir / "replacement.usd"
            replacement.write_text("#usda 1.0\n", encoding="utf-8")
            capabilities["geometry_output"] = "replacement"
            geometry.append(
                {
                    "label": "backend_replacement_geometry",
                    "uri": replacement.as_uri(),
                    "mime_type": "model/vnd.usd",
                }
            )
            diagnostics.append(
                _diagnostic(
                    "BACKEND_GEOMETRY_IGNORED",
                    severity="warning",
                    message=(
                        "Backend returned replacement geometry; Texture Agent "
                        "preserved source geometry."
                    ),
                    recommended_action=(
                        "Review auxiliary geometry artifact manually if needed."
                    ),
                    details={"geometry_uri": replacement.as_uri()},
                )
            )

        mask = _write_pattern(
            job_dir / "coverage_mask.png", size, channel="mask", seed=seed
        )
        preview = _write_pattern(
            job_dir / "projection_preview.png", size, channel="albedo", seed=seed + 1
        )
        return _status_response(
            job_id=job_id,
            status="completed",
            variant=variant,
            variant_asset_uri=source_asset_uri,
            generated_textures=generated_textures,
            maps=maps,
            auxiliary_artifacts={
                "masks": [
                    {
                        "label": "target_coverage",
                        "uri": mask.as_uri(),
                        "mime_type": "image/png",
                    }
                ],
                "debug_previews": [
                    {
                        "label": "projection_preview",
                        "uri": preview.as_uri(),
                        "mime_type": "image/png",
                    }
                ],
                "geometry": geometry,
            },
            metadata=_metadata(
                variant=variant,
                seed=seed,
                size=size,
                capabilities=capabilities,
                coverage=coverage,
                degraded_channels=degraded_channels,
            ),
            diagnostics=diagnostics,
        )

    def _missing_albedo_failure(
        self,
        *,
        job_id: str,
        variant: str,
        seed: int,
        size: int,
        source_asset_uri: str,
    ) -> dict[str, Any]:
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        normal = _write_pattern(
            job_dir / "normal.png", size, channel="normal", seed=seed
        )
        return _status_response(
            job_id=job_id,
            status="failed",
            variant=variant,
            variant_asset_uri=source_asset_uri,
            error_message="Backend did not return required albedo map.",
            generated_textures={
                "albedo": None,
                "normal": normal.as_uri(),
                "orm": None,
            },
            maps={
                "normal": _map(
                    normal,
                    size=size,
                    colorspace="raw",
                    packing="tangent_space_rgb",
                )
            },
            auxiliary_artifacts={"masks": [], "debug_previews": [], "geometry": []},
            metadata=_metadata(
                variant=variant,
                seed=seed,
                size=size,
                capabilities=_capabilities(orm=False),
                coverage={},
                degraded_channels=["albedo", "orm"],
            ),
            diagnostics=[
                _diagnostic(
                    "BACKEND_MAP_MISSING",
                    severity="error",
                    message="Backend did not return required albedo map.",
                    recommended_action=(
                        "Treat this job as failed and retry or choose another backend."
                    ),
                    details={"missing_maps": ["albedo"]},
                )
            ],
        )

    def _bad_uri_failure(
        self,
        *,
        job_id: str,
        variant: str,
        source_asset_uri: str,
        seed: int,
        size: int,
    ) -> dict[str, Any]:
        return _status_response(
            job_id=job_id,
            status="failed",
            variant=variant,
            variant_asset_uri=source_asset_uri,
            error_message="Backend could not read source URI.",
            generated_textures={"albedo": None, "normal": None, "orm": None},
            maps={},
            auxiliary_artifacts={"masks": [], "debug_previews": [], "geometry": []},
            metadata=_metadata(
                variant=variant,
                seed=seed,
                size=size,
                capabilities=_capabilities(),
                coverage={},
                degraded_channels=["albedo", "normal", "orm"],
            ),
            diagnostics=[
                _diagnostic(
                    "BACKEND_PARTIAL_FAILURE",
                    severity="error",
                    message="Backend could not read source URI.",
                    recommended_action="Provide a readable file, s3, omni, http, or https URI.",
                    details={"source_asset_uri": source_asset_uri},
                )
            ],
        )


def _status_response(
    *,
    job_id: str,
    status: str,
    variant: str,
    variant_asset_uri: str,
    generated_textures: dict[str, str | None],
    maps: dict[str, dict[str, Any]],
    auxiliary_artifacts: dict[str, Any],
    metadata: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    error_message: str | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "progress": 100,
        "result": {
            "variant_asset_uri": variant_asset_uri,
            "variant_name": variant,
            "generated_textures": generated_textures,
            "maps": maps,
            "auxiliary_artifacts": auxiliary_artifacts,
            "metadata": metadata,
            "diagnostics": diagnostics,
        },
    }
    if error_message:
        response["error_message"] = error_message
    return response


def _write_pattern(path: Path, size: int, *, channel: str, seed: int) -> Path:
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    if pixels is None:
        raise RuntimeError("PIL did not return a pixel access object")
    channel_bias = sum(channel.encode("utf-8")) % 256
    for y in range(size):
        for x in range(size):
            value = (x * 17 + y * 31 + seed + channel_bias) % 256
            if channel == "albedo":
                pixels[x, y] = (value, (value + 43) % 256, (value + 91) % 256)
            elif channel == "normal":
                pixels[x, y] = ((value // 3) % 256, (value // 2) % 256, 255)
            elif channel == "orm":
                pixels[x, y] = (255, value, 200)
            else:
                pixels[x, y] = (255 if value > 32 else 0, value, 255 - value)
    image.save(path, format="PNG")
    return path


def _map(
    path: Path,
    *,
    size: int,
    colorspace: str,
    packing: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "uri": path.as_uri(),
        "width": size,
        "height": size,
        "mime_type": "image/png",
        "colorspace": colorspace,
    }
    if packing:
        data["packing"] = packing
    return data


def _diagnostic(
    code: str,
    *,
    severity: str,
    message: str,
    recommended_action: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": code,
        "severity": severity,
        "stage": "generate_textures",
        "prim_path": DEFAULT_PRIM_PATH,
        "material_name": DEFAULT_MATERIAL_NAME,
        "message": message,
        "recommended_action": recommended_action,
        "details": details or {},
    }


def _metadata(
    *,
    variant: str,
    seed: int,
    size: int,
    capabilities: dict[str, Any],
    coverage: dict[str, float],
    degraded_channels: list[str],
) -> dict[str, Any]:
    return {
        "backend_name": "fake_projection_backend",
        "model": "fake-projection-v1",
        "endpoint_type": "fake_http",
        "seed": seed,
        "texture_size": size,
        "timings_ms": {"total": 1},
        "custom_parameter_summary": {"variant": variant},
        "capabilities": capabilities,
        "coverage": coverage,
        "degraded_channels": degraded_channels,
    }


def _capabilities(
    *,
    image_conditioning: bool = True,
    multiview: bool = False,
    normal_map: bool = True,
    orm: bool = True,
    masks: bool = True,
    coverage: bool = True,
    geometry_output: str = "none",
) -> dict[str, Any]:
    return {
        "image_conditioning": image_conditioning,
        "multiview": multiview,
        "normal_map": normal_map,
        "orm": orm,
        "masks": masks,
        "coverage": coverage,
        "geometry_output": geometry_output,
    }


def _valid_source_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    if parsed.scheme not in {"file", "s3", "omni", "http", "https"}:
        return False
    return bool(parsed.netloc or parsed.path)


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _positive_int_or_default(value: Any, default: int) -> int:
    parsed = _int_or_default(value, default)
    return parsed if parsed > 0 else default
