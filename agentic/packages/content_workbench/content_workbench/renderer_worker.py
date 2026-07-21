# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained OvRTX renderer worker for the inspector service."""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing
import os
import re
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

RENDER_PRODUCT_PATH = "/Session/Render/Viewport"
CAMERA_PATH = "/Session/Cameras/Main"
DEFAULT_HDRI_LIGHT_INTENSITY = 600.0
FALLBACK_DOME_LIGHT_INTENSITY = 350.0
FALLBACK_DISTANT_LIGHT_INTENSITY = 1200.0
_AOV_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
logger = logging.getLogger(__name__)
_OVRTX_ENV_LOCK = threading.Lock()
_OVRTX_ENV_CONFIGURED = False
_OVRTX_SPAWN_LOCK = threading.Lock()


@dataclass
class PickRenderResult:
    """Native OvRTX pick result."""

    prim_paths: list[str]
    elapsed_seconds: float


class _MissingActiveAOVError(RuntimeError):
    """Raised when OvRTX produces frames without the requested active AOV."""


def _managed_ovrtx_runtime() -> tuple[str, str]:
    """Return the managed interpreter and site directory for locked OvRTX."""
    from world_understanding.functions.graphics.render_ovrtx import (
        _get_ovrtx_python,
        _ovrtx_site_dir_env_for_python,
    )

    ovrtx_python = _get_ovrtx_python()
    site_dir = _ovrtx_site_dir_env_for_python(ovrtx_python)
    if site_dir is None:
        raise RuntimeError(
            "The managed OvRTX runtime did not expose a site-packages directory"
        )
    return ovrtx_python, str(Path(site_dir).resolve())


def _activate_managed_ovrtx_runtime() -> None:
    """Expose the hash-locked isolated OvRTX runtime to this worker process."""
    _ovrtx_python, resolved_site_dir = _managed_ovrtx_runtime()
    sys.path[:] = [
        resolved_site_dir,
        *(entry for entry in sys.path if entry != resolved_site_dir),
    ]
    importlib.invalidate_caches()


@contextmanager
def _managed_ovrtx_spawn_context() -> Iterator[multiprocessing.context.BaseContext]:
    """Prepare one spawned child to import the complete locked runtime set."""
    from multiprocessing import spawn

    with _OVRTX_SPAWN_LOCK:
        ovrtx_python, site_dir = _managed_ovrtx_runtime()
        context = multiprocessing.get_context("spawn")
        previous_executable = spawn.get_executable()
        previous_sys_path = list(sys.path)
        try:
            context.set_executable(ovrtx_python)
            sys.path[:] = [
                site_dir,
                *(entry for entry in sys.path if entry != site_dir),
            ]
            importlib.invalidate_caches()
            yield context
        finally:
            context.set_executable(previous_executable)
            sys.path[:] = previous_sys_path
            importlib.invalidate_caches()


def _runtime_process_identity() -> dict[str, str]:
    """Return child-runtime identities for isolation diagnostics and tests."""
    ovrtx_spec = importlib.util.find_spec("ovrtx")
    if ovrtx_spec is None or ovrtx_spec.origin is None:
        raise RuntimeError("ovrtx is not importable in the renderer child")
    return {
        "python": sys.executable,
        "ovrtx_version": metadata.version("ovrtx"),
        "numpy_version": metadata.version("numpy"),
        "pillow_version": metadata.version("pillow"),
        "ovrtx_origin": ovrtx_spec.origin,
        "numpy_origin": str(Path(np.__file__).resolve()),
        "pillow_origin": str(Path(Image.__file__).resolve()),
    }


def _configure_ovrtx_env(render_mode: str | None = None) -> None:
    """Set runtime variables before importing or constructing OvRTX."""
    global _OVRTX_ENV_CONFIGURED
    with _OVRTX_ENV_LOCK:
        os.environ["OVRTX_SKIP_USD_CHECK"] = "1"
        if render_mode:
            os.environ["OVRTX_RENDER_MODE"] = render_mode
        if _OVRTX_ENV_CONFIGURED:
            return
        _activate_managed_ovrtx_runtime()
        spec = importlib.util.find_spec("ovrtx")
        if spec is None or spec.origin is None:
            raise RuntimeError("ovrtx is not importable from the managed runtime")
        ovrtx_bin = Path(spec.origin).resolve().parent / "bin"
        os.environ["OVRTX_BIN_PATH"] = str(ovrtx_bin)
        _prepend_unique_env_path("LD_LIBRARY_PATH", ovrtx_bin / "plugins")
        _OVRTX_ENV_CONFIGURED = True


def _prepend_unique_env_path(name: str, path: Path) -> None:
    path_value = str(path)
    raw_existing = os.environ.get(name, "")
    existing = [
        item for item in raw_existing.split(os.pathsep) if item and item != path_value
    ]
    if name == "LD_LIBRARY_PATH":
        conflicts = [
            item for item in existing if "ovrtx" in item.lower() and item != path_value
        ]
        if conflicts:
            logger.warning(
                "Prepending OvRTX plugin path %s ahead of existing OvRTX-looking "
                "LD_LIBRARY_PATH entries: %s",
                path_value,
                os.pathsep.join(conflicts),
            )
    os.environ[name] = os.pathsep.join([path_value, *existing])


def _usd_path_literal(path: Path) -> str:
    value = str(path)
    if any(ord(character) < 32 for character in value):
        raise ValueError("USD asset paths must not contain control characters")
    return value.replace("\\", "\\\\").replace("@", "\\@")


def _usd_string_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _usd_prim_path_reference(value: str) -> str:
    path = str(value or "").strip()
    if not path.startswith("/") or any(
        character in path for character in ("<", ">", "\r", "\n", "\t")
    ):
        raise ValueError(f"Invalid USD prim path reference: {value!r}")
    return path


def validate_aov_name(active_aov: str) -> str:
    value = str(active_aov or "LdrColor")
    if _AOV_NAME_RE.fullmatch(value) is None:
        raise ValueError(
            "active AOV name must contain only ASCII letters, digits, or underscores"
        )
    return value


_validate_aov_name = validate_aov_name


def _release_render_products(products: object) -> None:
    """Release a renderer product holder if it exposes context-manager cleanup."""
    enter_method = getattr(products, "__enter__", None)
    exit_method = getattr(products, "__exit__", None)
    if enter_method is None or exit_method is None:
        return
    enter_method()
    exit_method(None, None, None)


def _matrix_literal(rows: list[list[float]]) -> str:
    return ",\n                    ".join(
        "(" + ", ".join(f"{value:.17g}" for value in row) + ")" for row in rows
    )


def _matrix_time_samples_literal(
    frame_numbers: list[int],
    camera_transforms: list[list[list[float]]],
) -> str:
    samples = []
    for frame, matrix in zip(frame_numbers, camera_transforms, strict=True):
        samples.append(
            f"                {frame}: (\n"
            f"                    {_matrix_literal(matrix)}\n"
            "                )"
        )
    return ",\n".join(samples)


def _default_hdri_lights_usda(intensity: float) -> str:
    from world_understanding.functions.graphics.render_ovrtx import (
        build_default_hdri_lights_usda,
    )

    usda = build_default_hdri_lights_usda(intensity=float(intensity))
    marker = 'def "OvRTXDefaultLights"'
    try:
        return usda[usda.index(marker) :]
    except ValueError as exc:
        raise RuntimeError(
            "Default OvRTX lights USDA did not contain lights prim"
        ) from exc


def _source_light_paths(scene_path: Path) -> list[str]:
    """Return authored scene lights that can be overridden in a viewer layer."""
    from pxr import Usd, UsdLux

    stage = Usd.Stage.Open(str(scene_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(
            f"Unable to open USD scene for light inspection: {scene_path}"
        )
    return [
        str(prim.GetPath())
        for prim in Usd.PrimRange(
            stage.GetPseudoRoot(),
            Usd.TraverseInstanceProxies(),
        )
        if not prim.IsPseudoRoot()
        and prim.HasAPI(UsdLux.LightAPI)
        and not prim.IsInstanceProxy()
    ]


def _source_light_overrides_usda(light_paths: list[str]) -> str:
    """Build non-destructive active=false opinions for source-authored lights."""
    if not light_paths:
        return ""

    tree: dict[str, dict[str, Any]] = {}
    for light_path in light_paths:
        path = _usd_prim_path_reference(light_path)
        components = [component for component in path.split("/") if component]
        if not components:
            raise ValueError("A source light path must identify a prim")
        children = tree
        entry: dict[str, Any] | None = None
        for component in components:
            entry = children.setdefault(
                component,
                {"active": False, "children": {}},
            )
            children = entry["children"]
        if entry is not None:
            entry["active"] = True

    def emit(nodes: dict[str, dict[str, Any]], indent: int = 0) -> list[str]:
        lines: list[str] = []
        prefix = " " * indent
        for name, entry in nodes.items():
            metadata = " (\n" + prefix + "    active = false\n" + prefix + ")"
            if not entry["active"]:
                metadata = ""
            lines.append(f'{prefix}over "{_usd_string_literal(name)}"{metadata}')
            lines.append(f"{prefix}{{")
            lines.extend(emit(entry["children"], indent + 4))
            lines.append(f"{prefix}}}")
        return lines

    return "\n".join(emit(tree))


def _viewer_stage_usda(
    *,
    scene_path: Path,
    width: int,
    height: int,
    camera_transform: list[list[float]],
    camera_focal_length: float,
    camera_horizontal_aperture: float,
    hdri_light: float | None,
    dome_light: float | None,
    distant_light: float | None,
    active_aov: str = "LdrColor",
    source_light_paths: list[str] | None = None,
) -> str:
    vertical_aperture = camera_horizontal_aperture * float(height) / float(width)
    active_aov = validate_aov_name(active_aov)
    active_var_name = active_aov
    active_source_name = _usd_string_literal(active_aov)
    default_hdri_lights = ""
    if hdri_light is not None:
        try:
            default_hdri_lights = f"\n{_default_hdri_lights_usda(float(hdri_light))}\n"
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Falling back to synthetic Workbench lights because default HDRI "
                "lights could not be generated: %s",
                exc,
            )
            if dome_light is None:
                dome_light = FALLBACK_DOME_LIGHT_INTENSITY
            if distant_light is None:
                distant_light = FALLBACK_DISTANT_LIGHT_INTENSITY
    dome = ""
    if dome_light is not None:
        dome = f"""
        def DomeLight "Fill"
        {{
            float inputs:intensity = {float(dome_light):.6g}
        }}
"""
    distant = ""
    if distant_light is not None:
        distant = f"""
        def DistantLight "Key"
        {{
            float inputs:intensity = {float(distant_light):.6g}
            float inputs:angle = 0.55
            double3 xformOp:rotateXYZ = (-35, 35, 0)
            uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
        }}
"""
    session_lights = ""
    if dome or distant:
        session_lights = f"""
    def Scope "Lights"
    {{{dome}{distant}
    }}
"""
    source_light_overrides = _source_light_overrides_usda(source_light_paths or [])
    if source_light_overrides:
        source_light_overrides = f"\n{source_light_overrides}\n"

    return f"""#usda 1.0
(
    defaultPrim = "Session"
    subLayers = [
        @{_usd_path_literal(scene_path)}@
    ]
)
{source_light_overrides}
{default_hdri_lights}
def Scope "Session"
{{
    def Scope "Cameras"
    {{
        def Camera "Main"
        {{
            float2 clippingRange = (0.01, 10000000)
            float focalLength = {float(camera_focal_length):.6g}
            float horizontalAperture = {float(camera_horizontal_aperture):.6g}
            float verticalAperture = {vertical_aperture:.6g}
            token projection = "perspective"
            matrix4d xformOp:transform = (
                    {_matrix_literal(camera_transform)}
            )
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}
    }}
{session_lights}

    def Scope "Render"
    {{
        def RenderProduct "Viewport"
        {{
            rel camera = <{CAMERA_PATH}>
            rel orderedVars = [
                </Session/Render/Vars/{active_var_name}>,
                </Session/Render/Vars/PickHit>
            ]
            uniform int2 resolution = ({int(width)}, {int(height)})
        }}

        def Scope "Vars"
        {{
            def RenderVar "{active_var_name}"
            {{
                uniform string sourceName = "{active_source_name}"
            }}

            def RenderVar "PickHit"
            {{
                uniform string sourceName = "ovrtx_pick_hit"
            }}
        }}

        def RenderSettings "Settings"
        {{
            rel products = [<{RENDER_PRODUCT_PATH}>]
        }}
    }}
}}
"""


def _viewer_stage_usda_for_frames(
    *,
    scene_path: Path,
    width: int,
    height: int,
    frame_numbers: list[int],
    camera_transforms: list[list[list[float]]],
    camera_path: str | None,
    camera_focal_length: float,
    camera_horizontal_aperture: float,
    fps: float,
    hdri_light: float | None,
    dome_light: float | None,
    distant_light: float | None,
    active_aov: str = "LdrColor",
    source_light_paths: list[str] | None = None,
) -> str:
    vertical_aperture = camera_horizontal_aperture * float(height) / float(width)
    active_aov = validate_aov_name(active_aov)
    active_var_name = active_aov
    active_source_name = _usd_string_literal(active_aov)
    default_hdri_lights = ""
    if hdri_light is not None:
        try:
            default_hdri_lights = f"\n{_default_hdri_lights_usda(float(hdri_light))}\n"
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Falling back to synthetic Workbench lights because default HDRI "
                "lights could not be generated: %s",
                exc,
            )
            if dome_light is None:
                dome_light = FALLBACK_DOME_LIGHT_INTENSITY
            if distant_light is None:
                distant_light = FALLBACK_DISTANT_LIGHT_INTENSITY
    dome = ""
    if dome_light is not None:
        dome = f"""
        def DomeLight "Fill"
        {{
            float inputs:intensity = {float(dome_light):.6g}
        }}
"""
    distant = ""
    if distant_light is not None:
        distant = f"""
        def DistantLight "Key"
        {{
            float inputs:intensity = {float(distant_light):.6g}
            float inputs:angle = 0.55
            double3 xformOp:rotateXYZ = (-35, 35, 0)
            uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
        }}
"""
    session_lights = ""
    if dome or distant:
        session_lights = f"""
    def Scope "Lights"
    {{{dome}{distant}
    }}
"""
    source_light_overrides = _source_light_overrides_usda(source_light_paths or [])
    if source_light_overrides:
        source_light_overrides = f"\n{source_light_overrides}\n"

    if not frame_numbers:
        raise ValueError("At least one frame number is required")
    if camera_path is None and len(camera_transforms) != len(frame_numbers):
        raise ValueError(
            "camera_transforms and frame_numbers must have the same length"
        )
    if camera_path is None:
        camera_scope = f"""
    def Scope "Cameras"
    {{
        def Camera "Main"
        {{
            float2 clippingRange = (0.01, 10000000)
            float focalLength = {float(camera_focal_length):.6g}
            float horizontalAperture = {float(camera_horizontal_aperture):.6g}
            float verticalAperture = {vertical_aperture:.6g}
            token projection = "perspective"
            matrix4d xformOp:transform.timeSamples = {{
{_matrix_time_samples_literal(frame_numbers, camera_transforms)}
            }}
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}
    }}
"""
        render_camera_path = CAMERA_PATH
    else:
        camera_scope = ""
        render_camera_path = _usd_prim_path_reference(camera_path)

    start_time_code = min(frame_numbers)
    end_time_code = max(frame_numbers)
    return f"""#usda 1.0
(
    defaultPrim = "Session"
    startTimeCode = {start_time_code}
    endTimeCode = {end_time_code}
    timeCodesPerSecond = {float(fps):.6g}
    subLayers = [
        @{_usd_path_literal(scene_path)}@
    ]
)
{source_light_overrides}
{default_hdri_lights}
def Scope "Session"
{{
{camera_scope}
{session_lights}

    def Scope "Render"
    {{
        def RenderProduct "Viewport"
        {{
            rel camera = <{render_camera_path}>
            rel orderedVars = [
                </Session/Render/Vars/{active_var_name}>,
                </Session/Render/Vars/PickHit>
            ]
            uniform int2 resolution = ({int(width)}, {int(height)})
        }}

        def Scope "Vars"
        {{
            def RenderVar "{active_var_name}"
            {{
                uniform string sourceName = "{active_source_name}"
            }}

            def RenderVar "PickHit"
            {{
                uniform string sourceName = "ovrtx_pick_hit"
            }}
        }}

        def RenderSettings "Settings"
        {{
            rel products = [<{RENDER_PRODUCT_PATH}>]
        }}
    }}
}}
"""


class OvRTXRendererWorker:
    """Persistent OvRTX renderer owned by the service process."""

    def __init__(self, *, log_file_path: str | None = None) -> None:
        self._lock = threading.RLock()
        self._renderer = None
        self._device = None
        self._render_mode: str | None = None
        self._log_file_path = log_file_path

    def render(
        self,
        *,
        scene_path: Path,
        output_path: Path,
        width: int,
        height: int,
        camera_transform: list[list[float]],
        camera_focal_length: float = 50.0,
        camera_horizontal_aperture: float = 36.0,
        num_updates: int = 64,
        render_mode: str = "rt2",
        hdri_light: float | None = DEFAULT_HDRI_LIGHT_INTENSITY,
        dome_light: float | None = None,
        distant_light: float | None = None,
        selected_prim_paths: list[str] | None = None,
        active_aov: str = "LdrColor",
        lock_timeout_seconds: float | None = None,
        source_light_paths: list[str] | None = None,
    ) -> float:
        """Render a PNG from a viewer-composed stage."""
        self._acquire_lock(timeout_seconds=lock_timeout_seconds)
        render_start = time.monotonic()
        try:
            active_aov = validate_aov_name(active_aov)
            try:
                return self._render_locked(
                    scene_path=scene_path,
                    output_path=output_path,
                    width=width,
                    height=height,
                    camera_transform=camera_transform,
                    camera_focal_length=camera_focal_length,
                    camera_horizontal_aperture=camera_horizontal_aperture,
                    num_updates=num_updates,
                    render_mode=render_mode,
                    hdri_light=hdri_light,
                    dome_light=dome_light,
                    distant_light=distant_light,
                    selected_prim_paths=selected_prim_paths,
                    active_aov=active_aov,
                    source_light_paths=source_light_paths,
                )
            except _MissingActiveAOVError:
                logger.warning(
                    "OvRTX render returned no %s; restarting renderer and "
                    "retrying once",
                    active_aov,
                )
                self._shutdown_renderer_locked()
                self._render_locked(
                    scene_path=scene_path,
                    output_path=output_path,
                    width=width,
                    height=height,
                    camera_transform=camera_transform,
                    camera_focal_length=camera_focal_length,
                    camera_horizontal_aperture=camera_horizontal_aperture,
                    num_updates=num_updates,
                    render_mode=render_mode,
                    hdri_light=hdri_light,
                    dome_light=dome_light,
                    distant_light=distant_light,
                    selected_prim_paths=selected_prim_paths,
                    active_aov=active_aov,
                    source_light_paths=source_light_paths,
                )
                return time.monotonic() - render_start
        finally:
            try:
                self._release_scene_locked()
            finally:
                self._lock.release()

    def pick(
        self,
        *,
        scene_path: Path,
        x: int,
        y: int,
        width: int,
        height: int,
        camera_transform: list[list[float]],
        camera_focal_length: float = 50.0,
        camera_horizontal_aperture: float = 36.0,
        num_updates: int = 1,
        render_mode: str = "rt2",
        hdri_light: float | None = DEFAULT_HDRI_LIGHT_INTENSITY,
        dome_light: float | None = None,
        distant_light: float | None = None,
        selected_prim_paths: list[str] | None = None,
        lock_timeout_seconds: float | None = None,
        source_light_paths: list[str] | None = None,
    ) -> PickRenderResult:
        """Run a native OvRTX pick query for one viewport pixel."""
        self._acquire_lock(timeout_seconds=lock_timeout_seconds)
        try:
            return self._pick_locked(
                scene_path=scene_path,
                x=x,
                y=y,
                width=width,
                height=height,
                camera_transform=camera_transform,
                camera_focal_length=camera_focal_length,
                camera_horizontal_aperture=camera_horizontal_aperture,
                num_updates=num_updates,
                render_mode=render_mode,
                hdri_light=hdri_light,
                dome_light=dome_light,
                distant_light=distant_light,
                selected_prim_paths=selected_prim_paths,
                source_light_paths=source_light_paths,
            )
        finally:
            try:
                self._release_scene_locked()
            finally:
                self._lock.release()

    def _pick_locked(
        self,
        *,
        scene_path: Path,
        x: int,
        y: int,
        width: int,
        height: int,
        camera_transform: list[list[float]],
        camera_focal_length: float,
        camera_horizontal_aperture: float,
        num_updates: int,
        render_mode: str,
        hdri_light: float | None,
        dome_light: float | None,
        distant_light: float | None,
        selected_prim_paths: list[str] | None,
        source_light_paths: list[str] | None,
    ) -> PickRenderResult:
        if source_light_paths is None:
            source_light_paths = _source_light_paths(scene_path)
        renderer, _device = self._renderer_and_device(render_mode=render_mode)
        start = time.monotonic()
        self._open_viewer_stage(
            scene_path=scene_path,
            width=width,
            height=height,
            camera_transform=camera_transform,
            camera_focal_length=camera_focal_length,
            camera_horizontal_aperture=camera_horizontal_aperture,
            hdri_light=hdri_light,
            dome_light=dome_light,
            distant_light=distant_light,
            render_mode=render_mode,
            source_light_paths=source_light_paths,
        )
        self._write_selection_groups(selected_prim_paths or [])
        for _index in range(max(0, int(num_updates) - 1)):
            _release_render_products(self._step_renderer())
        left = max(0, min(int(width) - 1, int(x)))
        top = max(0, min(int(height) - 1, int(y)))
        renderer.enqueue_pick_query(
            render_product_path=RENDER_PRODUCT_PATH,
            left=left,
            top=top,
            right=left + 1,
            bottom=top + 1,
        )
        products = self._step_renderer()
        paths = []
        ctx_mgr = products if hasattr(products, "__enter__") else None
        ctx = ctx_mgr.__enter__() if ctx_mgr else products
        try:
            product = ctx[RENDER_PRODUCT_PATH]
            for frame in product.frames:
                paths = _decode_pick_paths(renderer, frame)
                if paths:
                    break
        finally:
            if ctx_mgr:
                ctx_mgr.__exit__(None, None, None)
        return PickRenderResult(
            prim_paths=paths,
            elapsed_seconds=time.monotonic() - start,
        )

    def render_frames(
        self,
        *,
        scene_path: Path,
        output_paths: list[Path],
        width: int,
        height: int,
        frame_numbers: list[int],
        camera_transforms: list[list[list[float]]],
        camera_path: str | None = None,
        camera_focal_length: float = 50.0,
        camera_horizontal_aperture: float = 36.0,
        fps: float = 24.0,
        num_updates: int = 64,
        render_mode: str = "rt2",
        hdri_light: float | None = DEFAULT_HDRI_LIGHT_INTENSITY,
        dome_light: float | None = None,
        distant_light: float | None = None,
        selected_prim_paths: list[str] | None = None,
        active_aov: str = "LdrColor",
        lock_timeout_seconds: float | None = None,
        source_light_paths: list[str] | None = None,
    ) -> float:
        """Render a frame sequence from one time-sampled viewer stage."""
        if len(output_paths) != len(frame_numbers):
            raise ValueError("output_paths and frame_numbers must have the same length")
        if camera_path is None and len(output_paths) != len(camera_transforms):
            raise ValueError(
                "output_paths and camera_transforms must have the same length"
            )
        if not output_paths:
            raise ValueError("At least one frame output path is required")
        self._acquire_lock(timeout_seconds=lock_timeout_seconds)
        render_start = time.monotonic()
        try:
            active_aov = validate_aov_name(active_aov)
            try:
                self._render_frames_locked(
                    scene_path=scene_path,
                    output_paths=output_paths,
                    width=width,
                    height=height,
                    frame_numbers=frame_numbers,
                    camera_transforms=camera_transforms,
                    camera_path=camera_path,
                    camera_focal_length=camera_focal_length,
                    camera_horizontal_aperture=camera_horizontal_aperture,
                    fps=fps,
                    num_updates=num_updates,
                    render_mode=render_mode,
                    hdri_light=hdri_light,
                    dome_light=dome_light,
                    distant_light=distant_light,
                    selected_prim_paths=selected_prim_paths,
                    active_aov=active_aov,
                    source_light_paths=source_light_paths,
                )
            except _MissingActiveAOVError:
                logger.warning(
                    "OvRTX render_frames returned no %s; restarting renderer and "
                    "retrying once",
                    active_aov,
                )
                self._cleanup_render_outputs(output_paths)
                self._shutdown_renderer_locked()
                try:
                    self._render_frames_locked(
                        scene_path=scene_path,
                        output_paths=output_paths,
                        width=width,
                        height=height,
                        frame_numbers=frame_numbers,
                        camera_transforms=camera_transforms,
                        camera_path=camera_path,
                        camera_focal_length=camera_focal_length,
                        camera_horizontal_aperture=camera_horizontal_aperture,
                        fps=fps,
                        num_updates=num_updates,
                        render_mode=render_mode,
                        hdri_light=hdri_light,
                        dome_light=dome_light,
                        distant_light=distant_light,
                        selected_prim_paths=selected_prim_paths,
                        active_aov=active_aov,
                        source_light_paths=source_light_paths,
                    )
                except Exception:
                    self._cleanup_render_outputs(output_paths)
                    raise
            except Exception:
                self._cleanup_render_outputs(output_paths)
                raise
            return time.monotonic() - render_start
        finally:
            try:
                self._release_scene_locked()
            finally:
                self._lock.release()

    def release_scene(self, *, timeout_seconds: float | None = None) -> None:
        """Release scene-owned renderer state while keeping OvRTX warm."""
        self._acquire_lock(timeout_seconds=timeout_seconds)
        try:
            self._release_scene_locked()
        finally:
            self._lock.release()

    def shutdown(self, *, timeout_seconds: float | None = None) -> None:
        """Release the OvRTX renderer if it has been created."""
        self._acquire_lock(timeout_seconds=timeout_seconds)
        try:
            self._shutdown_renderer_locked()
        finally:
            self._lock.release()

    def _render_locked(
        self,
        *,
        scene_path: Path,
        output_path: Path,
        width: int,
        height: int,
        camera_transform: list[list[float]],
        camera_focal_length: float,
        camera_horizontal_aperture: float,
        num_updates: int,
        render_mode: str,
        hdri_light: float | None,
        dome_light: float | None,
        distant_light: float | None,
        selected_prim_paths: list[str] | None,
        active_aov: str,
        source_light_paths: list[str] | None,
    ) -> float:
        if source_light_paths is None:
            source_light_paths = _source_light_paths(scene_path)
        renderer, device = self._renderer_and_device(render_mode=render_mode)
        start = time.monotonic()
        active_var_name = active_aov
        self._open_viewer_stage(
            scene_path=scene_path,
            width=width,
            height=height,
            camera_transform=camera_transform,
            camera_focal_length=camera_focal_length,
            camera_horizontal_aperture=camera_horizontal_aperture,
            hdri_light=hdri_light,
            dome_light=dome_light,
            distant_light=distant_light,
            render_mode=render_mode,
            active_aov=active_aov,
            source_light_paths=source_light_paths,
        )
        self._write_selection_groups(selected_prim_paths or [])
        rgba = None
        step_count = max(1, int(num_updates))
        for _index in range(step_count):
            products = renderer.step(
                render_products={RENDER_PRODUCT_PATH},
                delta_time=1.0 / 30.0,
            )
            ctx_mgr = products if hasattr(products, "__enter__") else None
            ctx = ctx_mgr.__enter__() if ctx_mgr else products
            try:
                product = ctx[RENDER_PRODUCT_PATH]
                for frame in product.frames:
                    render_var = frame.render_vars.get(active_var_name)
                    if render_var is None:
                        continue
                    with render_var.map(device=device.CPU) as rv:
                        rgba = np.from_dlpack(rv).copy()
                    break
            finally:
                if ctx_mgr:
                    ctx_mgr.__exit__(None, None, None)

        if rgba is None:
            raise _MissingActiveAOVError(f"OvRTX did not return {active_aov}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(output_path)
        return time.monotonic() - start

    def _render_frames_locked(
        self,
        *,
        scene_path: Path,
        output_paths: list[Path],
        width: int,
        height: int,
        frame_numbers: list[int],
        camera_transforms: list[list[list[float]]],
        camera_path: str | None,
        camera_focal_length: float,
        camera_horizontal_aperture: float,
        fps: float,
        num_updates: int,
        render_mode: str,
        hdri_light: float | None,
        dome_light: float | None,
        distant_light: float | None,
        selected_prim_paths: list[str] | None,
        active_aov: str,
        source_light_paths: list[str] | None,
    ) -> None:
        if source_light_paths is None:
            source_light_paths = _source_light_paths(scene_path)
        renderer, device = self._renderer_and_device(render_mode=render_mode)
        renderer.open_usd_from_string(
            _viewer_stage_usda_for_frames(
                scene_path=scene_path,
                width=width,
                height=height,
                frame_numbers=frame_numbers,
                camera_transforms=camera_transforms,
                camera_path=camera_path,
                camera_focal_length=camera_focal_length,
                camera_horizontal_aperture=camera_horizontal_aperture,
                fps=fps,
                hdri_light=hdri_light,
                dome_light=dome_light,
                distant_light=distant_light,
                active_aov=active_aov,
                source_light_paths=source_light_paths,
            )
        )
        self._write_selection_groups(selected_prim_paths or [])
        step_count = max(1, int(num_updates))
        for frame_number, output_path in zip(frame_numbers, output_paths, strict=True):
            renderer.update_from_usd_time(float(frame_number) / float(fps))
            renderer.reset()
            rgba = None
            for _update_index in range(step_count):
                products = renderer.step(
                    render_products={RENDER_PRODUCT_PATH},
                    delta_time=0.0,
                )
                ctx_mgr = products if hasattr(products, "__enter__") else None
                ctx = ctx_mgr.__enter__() if ctx_mgr else products
                try:
                    product = ctx[RENDER_PRODUCT_PATH]
                    for frame in product.frames:
                        render_var = frame.render_vars.get(active_aov)
                        if render_var is None:
                            continue
                        with render_var.map(device=device.CPU) as rv:
                            rgba = np.from_dlpack(rv).copy()
                        break
                finally:
                    if ctx_mgr:
                        ctx_mgr.__exit__(None, None, None)
            if rgba is None:
                raise _MissingActiveAOVError(f"OvRTX did not return {active_aov}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba, mode="RGBA").save(output_path)

    def _cleanup_render_outputs(self, output_paths: list[Path]) -> None:
        for output_path in output_paths:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to remove partial render output %s", output_path)

    def _shutdown_renderer_locked(self) -> None:
        renderer = self._renderer
        self._renderer = None
        self._device = None
        self._render_mode = None
        if renderer is None:
            return
        shutdown = getattr(renderer, "shutdown", None)
        if shutdown is not None:
            shutdown()
            return
        reset_stage = getattr(renderer, "reset_stage", None)
        if reset_stage is not None:
            reset_stage()

    def _release_scene_locked(self) -> None:
        renderer = self._renderer
        if renderer is None:
            return
        reset_stage = getattr(renderer, "reset_stage", None)
        if reset_stage is None:
            return
        try:
            reset_stage()
        except Exception:
            logger.exception(
                "Unable to reset the OvRTX stage; discarding the renderer instance"
            )
            self._renderer = None
            self._device = None
            self._render_mode = None

    def _acquire_lock(self, *, timeout_seconds: float | None = None) -> None:
        if timeout_seconds is None:
            self._lock.acquire()
            return
        if self._lock.acquire(timeout=max(0.0, float(timeout_seconds))):
            return
        raise TimeoutError("Timed out waiting for the OvRTX renderer")

    def _renderer_and_device(self, *, render_mode: str = "rt2"):
        render_mode = _normalize_render_mode(render_mode)
        if self._renderer is not None and self._render_mode != render_mode:
            self.shutdown()
        if self._renderer is None or self._device is None:
            _configure_ovrtx_env(render_mode=render_mode)
            from ovrtx import (
                Device,
                Renderer,
                RendererConfig,
                SelectionFillMode,
                SelectionGroupStyle,
            )

            self._renderer = Renderer(
                config=RendererConfig(
                    sync_mode=True,
                    active_cuda_gpus="0",
                    keep_system_alive=False,
                    log_file_path=self._log_file_path,
                    selection_outline_enabled=True,
                    selection_outline_width=4,
                    selection_fill_mode=SelectionFillMode.GROUP_FILL_COLOR,
                )
            )
            self._renderer.set_selection_group_styles(
                {
                    1: SelectionGroupStyle(
                        outline_color=(1.0, 0.6, 0.0, 1.0),
                        fill_color=(0.0, 0.0, 0.0, 0.0),
                    ),
                    2: SelectionGroupStyle(
                        outline_color=(0.1, 0.55, 1.0, 1.0),
                        fill_color=(0.1, 0.55, 1.0, 0.16),
                    ),
                }
            )
            self._device = Device
            self._render_mode = render_mode
        return self._renderer, self._device

    def _open_viewer_stage(
        self,
        *,
        scene_path: Path,
        width: int,
        height: int,
        camera_transform: list[list[float]],
        camera_focal_length: float,
        camera_horizontal_aperture: float,
        hdri_light: float | None,
        dome_light: float | None,
        distant_light: float | None,
        render_mode: str,
        active_aov: str = "LdrColor",
        source_light_paths: list[str] | None = None,
    ) -> None:
        if source_light_paths is None:
            source_light_paths = _source_light_paths(scene_path)
        renderer, _device = self._renderer_and_device(render_mode=render_mode)
        renderer.open_usd_from_string(
            _viewer_stage_usda(
                scene_path=scene_path,
                width=width,
                height=height,
                camera_transform=camera_transform,
                camera_focal_length=camera_focal_length,
                camera_horizontal_aperture=camera_horizontal_aperture,
                hdri_light=hdri_light,
                dome_light=dome_light,
                distant_light=distant_light,
                active_aov=active_aov,
                source_light_paths=source_light_paths,
            )
        )

    def _step_renderer(self):
        renderer, _device = self._renderer_and_device(
            render_mode=self._render_mode or "rt2"
        )
        return renderer.step(
            render_products={RENDER_PRODUCT_PATH},
            delta_time=1.0 / 30.0,
        )

    def _write_selection_groups(self, selected_prim_paths: list[str]) -> None:
        if not selected_prim_paths:
            return
        renderer, _device = self._renderer_and_device(
            render_mode=self._render_mode or "rt2"
        )
        from ovrtx import OVRTX_ATTR_NAME_SELECTION_OUTLINE_GROUP, PrimMode

        paths = list(dict.fromkeys(selected_prim_paths))
        groups = np.ones((len(paths),), dtype=np.uint8)
        renderer.write_attribute(
            prim_paths=paths,
            attribute_name=OVRTX_ATTR_NAME_SELECTION_OUTLINE_GROUP,
            tensor=groups,
            prim_mode=PrimMode.CREATE_NEW,
        )


class IsolatedOvRTXRendererWorker(OvRTXRendererWorker):
    """Run each OvRTX operation in a child process with bounded lifetime."""

    def __init__(
        self,
        *,
        log_file_path: str | None = None,
        operation_timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(log_file_path=log_file_path)
        self._operation_timeout_seconds = operation_timeout_seconds

    def _run_isolated(self, operation: str, kwargs: dict[str, Any]) -> Any:
        for attempt in range(2):
            try:
                return self._run_isolated_once(operation, kwargs)
            except BrokenProcessPool:
                if attempt == 1:
                    raise
                logger.warning(
                    "OvRTX isolated %s process crashed; retrying once in a fresh process",
                    operation,
                )
        raise AssertionError("unreachable")

    def _run_isolated_once(self, operation: str, kwargs: dict[str, Any]) -> Any:
        executor: ProcessPoolExecutor | None = None
        try:
            with _managed_ovrtx_spawn_context() as context:
                executor = ProcessPoolExecutor(max_workers=1, mp_context=context)
                future = executor.submit(
                    _run_isolated_renderer_operation,
                    operation,
                    kwargs,
                    self._log_file_path,
                )
            timeout_seconds = self._operation_timeout_seconds
            if timeout_seconds is not None and timeout_seconds > 0:
                try:
                    return future.result(timeout=timeout_seconds)
                except FutureTimeoutError as exc:
                    _terminate_executor_children(executor)
                    raise TimeoutError(
                        f"OvRTX isolated {operation} exceeded "
                        f"{timeout_seconds:g} seconds"
                    ) from exc
            return future.result()
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

    def _render_locked(self, **kwargs: Any) -> float:
        if kwargs.get("source_light_paths") is None:
            kwargs["source_light_paths"] = _source_light_paths(kwargs["scene_path"])
        return cast(float, self._run_isolated("render", kwargs))

    def _pick_locked(self, **kwargs: Any) -> PickRenderResult:
        if kwargs.get("source_light_paths") is None:
            kwargs["source_light_paths"] = _source_light_paths(kwargs["scene_path"])
        return cast(PickRenderResult, self._run_isolated("pick", kwargs))

    def _render_frames_locked(self, **kwargs: Any) -> None:
        if kwargs.get("source_light_paths") is None:
            kwargs["source_light_paths"] = _source_light_paths(kwargs["scene_path"])
        self._run_isolated("render_frames", kwargs)


def _run_isolated_renderer_operation(
    operation: str,
    kwargs: dict[str, Any],
    log_file_path: str | None,
) -> Any:
    worker = OvRTXRendererWorker(log_file_path=log_file_path)
    try:
        method = getattr(worker, operation)
        return method(**kwargs)
    finally:
        worker.shutdown()


def _terminate_executor_children(executor: ProcessPoolExecutor) -> None:
    processes = list((getattr(executor, "_processes", {}) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=1.0)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)


def _normalize_render_mode(render_mode: str) -> str:
    value = str(render_mode or "rt2").strip().lower()
    if value not in {"rt2", "pt"}:
        raise ValueError(f"Unsupported OvRTX render mode: {render_mode}")
    return value


def _decode_pick_paths(renderer, frame) -> list[str]:
    import ovrtx

    if ovrtx.OVRTX_RENDER_VAR_PICK_HIT not in frame.render_vars:
        return []
    pick_var = frame.render_vars[ovrtx.OVRTX_RENDER_VAR_PICK_HIT]

    mapped = pick_var.map(device=ovrtx.Device.CPU)
    ctx_mgr = mapped if hasattr(mapped, "__enter__") else None
    rv = ctx_mgr.__enter__() if ctx_mgr else mapped
    try:
        magic = int(np.from_dlpack(rv.params["magic"]).reshape(-1)[0])
        version = int(np.from_dlpack(rv.params["version"]).reshape(-1)[0])
        hit_count = int(np.from_dlpack(rv.params["hitCount"]).reshape(-1)[0])
        prim_path_ids = np.from_dlpack(rv["primPath"]).copy().reshape(-1)
    finally:
        if ctx_mgr:
            ctx_mgr.__exit__(None, None, None)
        elif hasattr(mapped, "unmap"):
            mapped.unmap()

    if magic != ovrtx.OVRTX_PICK_HIT_MAGIC or version != ovrtx.OVRTX_PICK_HIT_VERSION:
        raise RuntimeError("Unexpected OvRTX pick-hit schema")

    paths = []
    seen = set()
    for prim_path_id in prim_path_ids[:hit_count]:
        path = renderer.resolve_prim_path_id(int(prim_path_id))
        if path and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths
