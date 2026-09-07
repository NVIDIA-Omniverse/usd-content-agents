# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured render-engine readiness probe."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from usd_core.config import Config, resolve_render_backends
from usd_core.render.factory import make_backend, resolved_renderer

_PROBE_SCHEMA_VERSION = "usd-cli.render-probe.v1"


def probe_render_engine(
    config: Config,
    *,
    required_engine: str = "ovrtx",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve and exercise the configured renderer with one tiny real render.

    The result reports the configured OVRTX backend and its readiness details.
    """

    renderer = resolved_renderer(config)
    transport = "remote" if renderer == "remote" else "local"
    result: dict[str, Any] = {
        "schema_version": _PROBE_SCHEMA_VERSION,
        "capabilities": ["appearance.clear.v1"],
        "required_engine": required_engine,
        "resolved_renderer": renderer,
        "engine": "ovrtx" if renderer in {"ovrtx", "remote"} else renderer,
        "transport": transport,
        "ready": False,
        "render": None,
    }
    if renderer == "remote":
        pool = resolve_render_backends(config.render)
        if not pool:
            result["error"] = "remote renderer has no configured backend endpoints"
            return result
        from usd_core.remote_protocol import check_remote_protocol

        profiles = []
        try:
            for backend in pool:
                url = str(backend.get("url") or "").rstrip("/")
                if not url:
                    raise RuntimeError("remote renderer contains an empty backend URL")
                info = check_remote_protocol(
                    url,
                    required_engine=required_engine,
                    api_key=str(backend.get("api_key") or "") or None,
                )
                profiles.append(
                    {
                        "url": url,
                        "engine": info.get("engine"),
                        "protocol_version": info.get("protocol_version"),
                        "status": info.get("status"),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - structured readiness failure
            result["error"] = (
                "remote backend identity/protocol probe failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return result
        result["backends"] = profiles
    if result["engine"] != required_engine:
        result["error"] = (
            f"required render engine {required_engine!r}, but configuration "
            f"resolved to {renderer!r}"
        )
        return result

    try:
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Cube.Define(stage, "/World/probe")
        camera = UsdGeom.Camera.Define(stage, "/World/cam")
        camera.AddTransformOp().Set(
            Gf.Matrix4d(
                0.707,
                -0.408,
                0.577,
                0,
                0,
                0.816,
                0.577,
                0,
                -0.707,
                -0.408,
                0.577,
                0,
                3,
                3,
                3,
                1,
            )
        )

        backend = make_backend(config)
        if backend.name not in {"ovrtx", "remote"}:
            result["error"] = (
                "renderer factory did not produce an OVRTX-backed renderer: "
                f"{backend.name}"
            )
            return result
        owns_dir = output_dir is None
        temporary = (
            tempfile.TemporaryDirectory(prefix="usd-cli-render-probe-")
            if owns_dir
            else None
        )
        target_dir = Path(temporary.name) if temporary is not None else Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        renders = backend.render(
            stage,
            ["/World/cam"],
            64,
            64,
            target_dir,
            mode="fast",
            names=["ovrtx_readiness_probe"],
        )
        elapsed = time.perf_counter() - started
        if len(renders) != 1:
            raise RuntimeError(
                f"readiness probe expected one render, received {len(renders)}"
            )
        render_path = Path(renders[0].path)
        if not render_path.is_file() or render_path.stat().st_size <= 0:
            raise RuntimeError("readiness probe did not write a non-empty image")
        with Image.open(render_path) as image:
            image.verify()
            dimensions = [image.width, image.height]
        if dimensions != [64, 64]:
            raise RuntimeError(
                f"readiness probe returned unexpected dimensions: {dimensions}"
            )
        result["ready"] = True
        result["render"] = {
            "path": str(render_path) if not owns_dir else None,
            "width": dimensions[0],
            "height": dimensions[1],
            "size_bytes": render_path.stat().st_size,
            "duration_s": round(elapsed, 6),
            "backend": renders[0].backend,
        }
        if temporary is not None:
            temporary.cleanup()
        return result
    except Exception as exc:  # noqa: BLE001 - structured probe failure
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
