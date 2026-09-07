# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderer wrapper: parse request USD → usd_core OvRTXRenderBackend → base64 PNGs.

Runs in the service (main) process, which has `pxr`; the actual RTX render happens in the
isolated ovrtx daemon the backend spawns. We do not duplicate any daemon logic here.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class Renderer:
    def __init__(self, log_level: str = "warn", num_sensor_updates: int = 64,
                 render_mode: str = ""):
        from usd_core.render.ovrtx import OvRTXRenderBackend

        self._backend = OvRTXRenderBackend(
            num_sensor_updates=num_sensor_updates,
            render_mode=render_mode,
            log_level=log_level,
        )
        self._initialized = False  # construction succeeded (backend object exists)
        self._gpu_ready = False  # daemon booted at least once
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def is_ready(self) -> bool:
        return self._gpu_ready

    @property
    def daemon_running(self) -> bool:
        return self._backend.daemon_running

    def warm_up(self) -> bool:
        """Boot the OVRTX daemon (provisioning the venv + GPU init) off the request path.
        Returns True on success; logs and returns False otherwise."""
        try:
            self._backend.ensure_ready()
            self._gpu_ready = True
            return True
        except Exception:  # noqa: BLE001
            logger.exception("OVRTX warm-up failed")
            return False

    def render(self, cameras: list[str], width: int, height: int, mode: str = "quality",
               usdz_base64: str | None = None, usd: str | None = None,
               frames: list[float] | None = None, camera_defs=None) -> list[dict]:
        """Render the request scene and return [{camera, image_base64, frame?}, ...].

        Prefers a USDZ bundle (`usdz_base64`) so textures resolve; falls back to flat USDA
        text (`usd`). Exactly one is expected (enforced by the request model). When `frames`
        is given, the scene is ingested once and each frame rendered at its USD time code.
        """
        from pxr import Sdf, Tf, Usd

        with tempfile.TemporaryDirectory(prefix="ovrtx_api_") as tmp:
            out_dir = Path(tmp)
            if usdz_base64:
                # A USDZ is a zip — write it to disk and open it so USD resolves the bundled
                # texture assets from inside the package.
                usdz_path = out_dir / "scene_bundle.usdz"
                try:
                    usdz_path.write_bytes(base64.b64decode(usdz_base64, validate=True))
                except ValueError as exc:
                    raise ValueError("usdz_base64 is not valid base64") from exc
                stage = Usd.Stage.Open(str(usdz_path))
                if not stage:
                    raise ValueError("could not open the request USDZ bundle")
            else:
                layer = Sdf.Layer.CreateAnonymous(".usda")
                # Malformed USD raises Tf.ErrorException (it does not return False); translate
                # it to ValueError so the API answers 400 (bad request) rather than 500.
                try:
                    ok = layer.ImportFromString(usd)
                except Tf.ErrorException as exc:
                    raise ValueError(f"could not parse request USD: {exc}") from exc
                if not ok:
                    raise ValueError("could not parse request USD as USDA text")
                stage = Usd.Stage.Open(layer)

            return self._render_stage(stage, cameras, width, height, mode, out_dir,
                                      frames, camera_defs)

    def render_usdz_bytes(self, cameras: list[str], width: int, height: int, mode: str,
                          usdz_bytes: bytes, frames: list[float] | None = None,
                          camera_defs=None) -> list[dict]:
        """Render a raw binary USDZ bundle (the multipart `/render/upload` transport)."""
        from pxr import Usd

        with tempfile.TemporaryDirectory(prefix="ovrtx_api_") as tmp:
            out_dir = Path(tmp)
            usdz_path = out_dir / "scene_bundle.usdz"
            usdz_path.write_bytes(usdz_bytes)
            stage = Usd.Stage.Open(str(usdz_path))
            if not stage:
                raise ValueError("could not open the uploaded USDZ bundle")
            return self._render_stage(stage, cameras, width, height, mode, out_dir,
                                      frames, camera_defs)

    def render_scene_path(self, cameras: list[str], width: int, height: int, mode: str,
                          scene_path: str, frames: list[float] | None = None,
                          camera_defs=None) -> list[dict]:
        """Render a scene already staged on local disk — the CAS manifest transport's
        materialized bundle tree. The root layer is a regular (non-package) layer, so
        protocol-v2 camera specs are authored as plain in-memory edits; the staged
        files are hardlinks into the blob store and are never written."""
        from pxr import Usd

        with tempfile.TemporaryDirectory(prefix="ovrtx_api_") as tmp:
            stage = Usd.Stage.Open(str(scene_path))
            if not stage:
                raise ValueError("could not open the staged scene's root layer")
            return self._render_stage(stage, cameras, width, height, mode, Path(tmp),
                                      frames, camera_defs)

    @staticmethod
    def _apply_camera_defs(stage, camera_defs, out_dir: Path):
        """Author protocol-v2 camera specs into the request stage; returns the stage
        to render (possibly a new wrapper stage).

        The client strips its tool-authored cameras from the bundle (so the bundle
        is viewpoint-independent and cache-friendly) and sends their specs here.
        A USDZ root layer is immutable, so for bundles the cameras go into a tiny
        writable wrapper layer that sublayers the package — stage metadata (up
        axis, units, time codes) is copied over so composition and framing match
        the original. Invalid specs raise ValueError (→ HTTP 400)."""
        if not camera_defs:
            return stage
        from pxr import Gf, Sdf, Usd, UsdGeom

        root = stage.GetRootLayer()
        real = getattr(root, "realPath", "") or ""
        if real.endswith(".usdz"):
            wrapper = Sdf.Layer.CreateNew(str(out_dir / "scene_with_cameras.usda"))
            wrapper.subLayerPaths = [Path(real).name]  # sits next to the bundle
            for field in ("upAxis", "metersPerUnit", "kilogramsPerUnit",
                          "startTimeCode", "endTimeCode", "timeCodesPerSecond",
                          "framesPerSecond", "defaultPrim"):
                if root.pseudoRoot.HasInfo(field):
                    wrapper.pseudoRoot.SetInfo(field, root.pseudoRoot.GetInfo(field))
            stage = Usd.Stage.Open(wrapper)
            if not stage:
                raise ValueError("could not compose the camera wrapper stage")
        for cd in camera_defs:
            spec = cd.model_dump() if hasattr(cd, "model_dump") else dict(cd)
            path = str(spec.get("path", ""))
            if (not Sdf.Path.IsValidPathString(path)
                    or not Sdf.Path(path).IsAbsolutePath()
                    or not Sdf.Path(path).IsPrimPath()):
                raise ValueError(f"camera_defs: invalid camera prim path {path!r}")
            cam = UsdGeom.Camera.Define(stage, Sdf.Path(path))
            if spec.get("focal_length") is not None:
                cam.CreateFocalLengthAttr(float(spec["focal_length"]))
            if spec.get("horizontal_aperture") is not None:
                cam.CreateHorizontalApertureAttr(float(spec["horizontal_aperture"]))
            if spec.get("vertical_aperture") is not None:
                cam.CreateVerticalApertureAttr(float(spec["vertical_aperture"]))
            clip = spec.get("clipping_range")
            if clip is not None:
                cam.CreateClippingRangeAttr(Gf.Vec2f(float(clip[0]), float(clip[1])))
            if spec.get("projection"):
                cam.CreateProjectionAttr(str(spec["projection"]))
            m = [float(v) for v in spec["matrix"]]
            if not all(map(math.isfinite, m)):
                raise ValueError(f"camera_defs: non-finite matrix for {path!r}")
            xf = UsdGeom.Xformable(cam.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d(*m))
            # The client sends the LOCAL-TO-WORLD matrix; the camera path may sit
            # under transformed ancestors (e.g. /World/ov_cam under a scaled
            # /World), which would apply the parent transform twice. Resetting
            # the xform stack makes the authored matrix the world transform.
            xf.SetResetXformStack(True)
        return stage

    def _render_stage(self, stage, cameras, width, height, mode, out_dir: Path,
                      frames: list[float] | None = None,
                      camera_defs=None) -> list[dict]:
        stage = self._apply_camera_defs(stage, camera_defs, out_dir)
        # frames=None → one render at the default time code (backward compatible). Otherwise
        # the stage is already ingested/open here; each frame reuses it and the warm daemon,
        # so N frames cost one upload + one stage-open instead of N of each.
        if not frames:
            results = self._backend.render(stage, cameras, width, height, out_dir, mode=mode)
            self._gpu_ready = True
            return [{"camera": r.camera,
                     "image_base64": base64.b64encode(Path(r.path).read_bytes()).decode("ascii"),
                     "ovrtx_render_mode": r.ovrtx_render_mode,
                     "ovrtx_num_sensor_updates": r.ovrtx_num_sensor_updates,
                     "active_aov": r.active_aov}
                    for r in results]
        items: list[dict] = []
        for f in frames:
            results = self._backend.render(stage, cameras, width, height, out_dir, mode=mode,
                                           names=[f"{c.strip('/').replace('/', '_')}__f{int(round(f)):04d}"
                                                  for c in cameras],
                                           frame=float(f))
            self._gpu_ready = True
            for r in results:
                items.append({
                    "camera": r.camera,
                    "frame": float(f),
                    "image_base64": base64.b64encode(Path(r.path).read_bytes()).decode("ascii"),
                    "ovrtx_render_mode": r.ovrtx_render_mode,
                    "ovrtx_num_sensor_updates": r.ovrtx_num_sensor_updates,
                    "active_aov": r.active_aov,
                })
        return items

    def close(self) -> None:
        daemon = getattr(self._backend, "_daemon", None)
        if daemon is not None:
            daemon.close()
