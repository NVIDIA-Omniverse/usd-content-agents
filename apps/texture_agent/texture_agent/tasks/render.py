# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Render the final textured USD."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task
from world_understanding.rendering_backend_contract import (
    RemoteRenderingSlotTimeoutError,
)

from texture_agent.config.rendering_backends import (
    DEFAULT_TEXTURE_RENDERING_BACKEND,
    has_production_visual_evidence,
    validate_texture_rendering_backend,
)
from texture_agent.tasks.render_results import render_result_items

logger = logging.getLogger(__name__)

_DIAGNOSTIC_SCHEMA_VERSION = "texture-agent-diagnostic.v1"
_DEFAULT_RENDER_SLOT_TIMEOUT_SEC = 300.0


def _render_result_items(results: Any) -> list[dict[str, Any]]:
    """Normalize renderer results using the final-render diagnostic contract."""
    return render_result_items(results, producer="render_all_cameras")


def _diagnostic(
    code: str,
    message: str,
    *,
    severity: str = "error",
    usd_path: str | None = None,
    camera_path: str | None = None,
    recommended_action: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {
        "schema_version": _DIAGNOSTIC_SCHEMA_VERSION,
        "code": code,
        "severity": severity,
        "stage": "render",
        "message": message,
        "recommended_action": recommended_action,
        "details": details or {},
    }
    if usd_path:
        diagnostic["usd_path"] = usd_path
    if camera_path:
        diagnostic["camera_path"] = camera_path
    return diagnostic


def _as_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value if item]
    return []


def _configured_camera_paths(config: dict[str, Any]) -> list[str]:
    for key in ("camera_paths", "cameras", "camera_path"):
        camera_paths = _as_path_list(config.get(key))
        if camera_paths:
            return camera_paths
    return []


def _config_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _config_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0.0 else default


def _render_slot_timeout_seconds(config: dict[str, Any]) -> float:
    return _config_float(
        config.get(
            "render_slot_timeout_sec",
            config.get("global_render_slot_timeout_sec"),
        ),
        default=_DEFAULT_RENDER_SLOT_TIMEOUT_SEC,
    )


def _render_request_timeout_seconds(config: dict[str, Any]) -> int:
    return int(
        _config_float(
            config.get(
                "timeout_sec",
                config.get(
                    "render_timeout_sec",
                    config.get("request_timeout_sec", config.get("timeout")),
                ),
            ),
            default=3600.0,
        )
    )


def _normalize_render_image_for_save(img: Any, config: dict[str, Any]) -> Any:
    """Return the final render image to persist as evidence.

    OVRTX can return RGBA images with a constant non-opaque alpha channel even
    for ordinary beauty renders. Persisting that alpha makes screenshots look
    dark or transparent depending on the viewer background, so final Texture
    Agent evidence defaults to opaque RGB output. Callers can opt back into raw
    alpha with ``steps.render.preserve_alpha: true``.
    """
    if _config_bool(
        config.get("preserve_alpha", config.get("preserve_render_alpha")),
        default=False,
    ):
        return img

    if getattr(img, "mode", None) in {"RGBA", "LA"} or "transparency" in getattr(
        img,
        "info",
        {},
    ):
        return img.convert("RGB")

    return img


def _focus_cameras_enabled(config: dict[str, Any]) -> bool:
    if "focus_cameras" in config:
        return _config_bool(config.get("focus_cameras"), default=True)
    return _config_bool(config.get("render_focus_cameras"), default=True)


def _stage_camera_paths(stage: Any) -> list[str]:
    from pxr import UsdGeom

    return [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)
    ]


def _stage_has_lights(stage: Any) -> bool:
    from pxr import UsdLux

    return any(prim.HasAPI(UsdLux.LightAPI) for prim in stage.Traverse())


def _add_default_lights(stage: Any, config: dict[str, Any]) -> bool:
    if not _config_bool(config.get("add_default_lights"), default=True):
        return False
    if _stage_has_lights(stage):
        return False

    from pxr import Gf, UsdGeom, UsdLux

    dome_intensity = float(config.get("dome_light_intensity", 500.0))
    distant_intensity = float(config.get("distant_light_intensity", 3000.0))

    dome = UsdLux.DomeLight.Define(stage, "/TextureAgentRenderLights/DomeLight")
    dome.GetIntensityAttr().Set(dome_intensity)

    distant = UsdLux.DistantLight.Define(
        stage,
        "/TextureAgentRenderLights/DistantLight",
    )
    distant.GetIntensityAttr().Set(distant_intensity)
    UsdGeom.Xformable(distant).AddRotateXYZOp().Set(Gf.Vec3f(315, 45, 0))
    return True


def _selected_prim_paths(context: dict[str, Any], config: dict[str, Any]) -> list[str]:
    explicit = _as_path_list(
        config.get("focus_prim_paths") or config.get("focus_prim_path")
    )
    if explicit:
        return explicit

    selected: list[str] = []
    units = context.get("prim_texture_units") or []
    for unit in units:
        unit_path = getattr(unit, "prim_path", None)
        if unit_path:
            selected.append(str(unit_path))
            continue
        material = getattr(unit, "material_info", None)
        selected.extend(str(path) for path in getattr(material, "bound_prim_paths", []))

    if selected:
        return list(dict.fromkeys(selected))

    material_textures = context.get("material_textures") or {}
    selected_materials = set(material_textures)
    if not selected_materials:
        return []

    for material in context.get("discovered_materials", []) or []:
        if getattr(material, "name", None) in selected_materials:
            selected.extend(
                str(path) for path in getattr(material, "bound_prim_paths", [])
            )

    return list(dict.fromkeys(selected))


def _add_default_camera(stage: Any, config: dict[str, Any]) -> str:
    from world_understanding.utils.usd.camera import add_corner_view_camera

    camera_path = str(config.get("camera_path") or "/Cameras/TextureAgentFinal")
    add_corner_view_camera(
        stage,
        camera_path=camera_path,
        direction=str(config.get("camera_direction", "+x+y+z")),
        margin=float(config.get("camera_margin", 1.25)),
        focal_length=float(config.get("camera_focal_length", 60.0)),
    )
    return camera_path


def _add_focus_cameras(
    stage: Any,
    context: dict[str, Any],
    config: dict[str, Any],
    output_index: int,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    if not _focus_cameras_enabled(config):
        return [], [], []

    from world_understanding.utils.usd.camera import add_focused_corner_view_camera

    camera_paths: list[str] = []
    focus_stats: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    focus_prim_paths = _selected_prim_paths(context, config)
    max_cameras = max(0, int(config.get("max_focus_cameras", 1)))
    if max_cameras == 0:
        return [], [], []

    threshold = float(config.get("target_frame_coverage_threshold", 0.2))
    margin = float(config.get("focus_camera_margin", 1.15))

    for focus_index, prim_path in enumerate(focus_prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            diagnostics.append(
                _diagnostic(
                    "RENDER_FOCUS_PRIM_MISSING",
                    f"Focused render prim does not exist: {prim_path}",
                    severity="warning",
                    recommended_action=(
                        "Use steps.render.focus_prim_paths with geometry prim "
                        "paths present in the output USD."
                    ),
                    details={"prim_path": prim_path},
                )
            )
            continue

        camera_path = f"/Cameras/TextureAgentFocus_{output_index}_{focus_index}"
        try:
            add_focused_corner_view_camera(
                prim,
                camera_path=camera_path,
                direction=str(
                    config.get(
                        "focus_camera_direction",
                        config.get("camera_direction", "+x+y+z"),
                    )
                ),
                margin=margin,
                focal_length=float(
                    config.get(
                        "focus_camera_focal_length",
                        config.get("camera_focal_length", 60.0),
                    )
                ),
            )
        except Exception as exc:
            diagnostics.append(
                _diagnostic(
                    "RENDER_FOCUS_CAMERA_FAILED",
                    f"Failed to add focused render camera for {prim_path}: {exc}",
                    severity="warning",
                    camera_path=camera_path,
                    recommended_action=(
                        "Use steps.render.focus_prim_paths with imageable "
                        "geometry prims that have valid bounds."
                    ),
                    details={
                        "prim_path": prim_path,
                        "exception_type": type(exc).__name__,
                    },
                )
            )
            logger.warning(
                "Failed to add focused render camera for %s", prim_path, exc_info=True
            )
            continue

        camera_paths.append(camera_path)
        coverage_estimate = min(1.0, 1.0 / max(margin * margin, 1e-6))
        focus_stat = {
            "prim_path": prim_path,
            "camera_path": camera_path,
            "target_frame_coverage_threshold": threshold,
            "target_frame_coverage_heuristic": coverage_estimate,
            "coverage_metric_source": "focus_camera_bbox_margin_heuristic",
            "coverage_is_estimate": True,
            "meets_target_frame_coverage": coverage_estimate >= threshold,
        }
        focus_stats.append(focus_stat)
        if not focus_stat["meets_target_frame_coverage"]:
            diagnostics.append(
                _diagnostic(
                    "RENDER_FRAME_TOO_WIDE",
                    (
                        "Focused render framing estimate is below the target "
                        f"coverage threshold for {prim_path}."
                    ),
                    severity="warning",
                    camera_path=camera_path,
                    recommended_action=(
                        "Reduce steps.render.focus_camera_margin or lower "
                        "steps.render.target_frame_coverage_threshold."
                    ),
                    details=focus_stat,
                )
            )
        if len(camera_paths) >= max_cameras:
            break

    return camera_paths, focus_stats, diagnostics


def _status_is_failure(result: dict[str, Any]) -> bool:
    if not result.get("images"):
        return True
    status = result.get("status")
    if status is None:
        return False
    return str(status) != "success"


def _result_camera_path(
    result: dict[str, Any],
    result_index: int,
    requested_camera_paths: list[str],
) -> str:
    if result.get("camera"):
        return str(result["camera"])
    if result_index < len(requested_camera_paths):
        return requested_camera_paths[result_index]
    return f"camera_{result_index}"


def _extend_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


class RenderOutputTask(Task):
    """Render the final textured USD for visual verification.

    Context keys read:
        output_usd_paths (list[str]): From ApplyTexturesTask.
        render_config (dict): backend, image_width, image_height, etc.
        working_dir (str): Working directory.

    Context keys written:
        rendered_image_paths (list[str]): Paths to rendered images.
        render_diagnostics (list[dict]): Structured render warnings and errors.
        render_errors (list[dict]): Error-severity render diagnostics.
        render_stats (dict): Render availability, cameras, focus cameras, and count.
    """

    def __init__(self) -> None:
        self.name = "RenderOutput"
        self.description = "Render the final textured USD"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        output_usd_paths: list[str] = context.get("output_usd_paths", [])
        config: dict[str, Any] = context.get("render_config", {})
        working_dir = Path(context["working_dir"])
        backend_type = validate_texture_rendering_backend(
            config.get("backend", DEFAULT_TEXTURE_RENDERING_BACKEND),
            step_name="render",
        )
        diagnostics: list[dict[str, Any]] = []
        evidence_classification = (
            "mock_placeholder" if backend_type == "mock" else "renderer_output"
        )
        render_stats: dict[str, Any] = {
            "backend": backend_type,
            "evidence_classification": evidence_classification,
            "production_visual_evidence": False,
            "camera_paths": [],
            "focus_cameras": [],
            "renders_count": 0,
            "render_available": False,
            "texture_detail_display_color_bakes": 0,
            "texture_detail_package_texture_localizations": 0,
            "texture_detail_uv_texture_fallbacks": 0,
            "textured_preview_fallbacks": 0,
        }

        if backend_type == "mock":
            diagnostics.append(
                _diagnostic(
                    "RENDER_MOCK_PLACEHOLDER",
                    (
                        "Mock rendering produces deterministic placeholder images, "
                        "not production visual evidence."
                    ),
                    severity="warning",
                    recommended_action=(
                        "Use the remote or ovrtx backend for production visual "
                        "inspection of generated textures."
                    ),
                    details={"backend": backend_type},
                )
            )

        if not output_usd_paths:
            logger.info("No output USDs to render")
            context["rendered_image_paths"] = []
            context["render_diagnostics"] = diagnostics
            context["render_errors"] = []
            context["render_stats"] = render_stats
            return context

        image_width = config.get("image_width", 1024)
        image_height = config.get("image_height", image_width)

        from world_understanding.functions.graphics.rendering_backend_factory import (
            create_rendering_backend,
        )

        backend_config = {
            **config,
            "timeout": _render_request_timeout_seconds(config),
        }
        rendering_backend = create_rendering_backend(backend_type, backend_config)
        logger.info("Using %s rendering backend for final output", backend_type)

        out_dir = working_dir / "renders"
        out_dir.mkdir(parents=True, exist_ok=True)

        from pxr import Usd
        from world_understanding.utils.usd.material import (
            add_ovrtx_preview_fallbacks_for_texture_file_materials,
            bake_texture_file_materials_to_display_color_for_render,
            convert_custom_mdl_to_builtin,
            localize_package_texture_assets_for_render,
        )

        rendered: list[str] = []

        for output_index, usd_path in enumerate(output_usd_paths):
            logger.info("Rendering %s", usd_path)
            try:
                try:
                    stage = Usd.Stage.Open(str(usd_path))
                except Exception as exc:
                    diagnostics.append(
                        _diagnostic(
                            "RENDER_OUTPUT_USD_OPEN_FAILED",
                            f"Failed to open output USD for rendering: {usd_path}",
                            usd_path=str(usd_path),
                            recommended_action=(
                                "Check that apply_textures produced a valid USD "
                                "path before the render step runs."
                            ),
                            details={
                                "exception_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        )
                    )
                    logger.exception("Failed to open stage: %s", usd_path)
                    continue

                if not stage:
                    diagnostics.append(
                        _diagnostic(
                            "RENDER_OUTPUT_USD_OPEN_FAILED",
                            f"Failed to open output USD for rendering: {usd_path}",
                            usd_path=str(usd_path),
                            recommended_action=(
                                "Check that apply_textures produced a valid USD "
                                "path before the render step runs."
                            ),
                            details={"reason": "stage_returned_none"},
                        )
                    )
                    logger.warning("Failed to open stage: %s", usd_path)
                    continue

                # Flatten so every backend receives resolved composition arcs.
                flat_layer = stage.Flatten()
                flat_stage = Usd.Stage.Open(flat_layer)
                convert_custom_mdl_to_builtin(flat_stage)
                localized_package_textures = localize_package_texture_assets_for_render(
                    flat_stage,
                    working_dir / "render_assets" / f"output_{output_index}",
                )
                render_stats["texture_detail_package_texture_localizations"] += (
                    localized_package_textures
                )
                if localized_package_textures:
                    logger.info(
                        "Localized %d USDZ package texture reference(s) for OVRTX",
                        localized_package_textures,
                    )
                preserve_mdl_surface = _config_bool(
                    config.get("preserve_mdl_surface"),
                    default=True,
                )
                display_color_bakes = 0
                if not preserve_mdl_surface:
                    display_color_bakes = (
                        bake_texture_file_materials_to_display_color_for_render(
                            flat_stage,
                        )
                    )
                render_stats["texture_detail_display_color_bakes"] += (
                    display_color_bakes
                )
                if display_color_bakes:
                    logger.info(
                        "Baked %d textured mesh(es) to displayColor for OVRTX",
                        display_color_bakes,
                    )

                textured_fallbacks = (
                    add_ovrtx_preview_fallbacks_for_texture_file_materials(
                        flat_stage,
                        override_existing_surface=True,
                        connect_diffuse_texture=preserve_mdl_surface,
                        diffuse_color_primvar=(
                            "displayColor"
                            if display_color_bakes and not preserve_mdl_surface
                            else None
                        ),
                        skip_connected_mdl_surface=preserve_mdl_surface,
                    )
                )
                render_stats["textured_preview_fallbacks"] += textured_fallbacks
                if preserve_mdl_surface:
                    render_stats["texture_detail_uv_texture_fallbacks"] += (
                        textured_fallbacks
                    )
                    fallback_label = "UsdUVTexture"
                elif display_color_bakes:
                    fallback_label = "displayColor"
                else:
                    fallback_label = "solid-color"
                if textured_fallbacks:
                    logger.info(
                        "Added %d %s UsdPreviewSurface fallback(s) for OVRTX",
                        textured_fallbacks,
                        fallback_label,
                    )

                if _add_default_lights(flat_stage, config):
                    logger.info("Added default Texture Agent final render lights")

                camera_paths = _configured_camera_paths(config)
                if not camera_paths:
                    camera_paths = _stage_camera_paths(flat_stage)

                if not camera_paths:
                    camera_path = _add_default_camera(flat_stage, config)
                    camera_paths = [camera_path]
                    diagnostics.append(
                        _diagnostic(
                            "RENDER_NO_CAMERA",
                            (
                                "Output USD has no authored camera; added a "
                                f"default final render camera at {camera_path}."
                            ),
                            severity="warning",
                            usd_path=str(usd_path),
                            camera_path=camera_path,
                            recommended_action=(
                                "Author a camera in the source USD or set "
                                "steps.render.camera_paths for deterministic "
                                "final renders."
                            ),
                        )
                    )

                focus_cameras, focus_stats, focus_diagnostics = _add_focus_cameras(
                    flat_stage, context, config, output_index
                )
                for camera_path in focus_cameras:
                    if camera_path not in camera_paths:
                        camera_paths.append(camera_path)
                render_stats["focus_cameras"].extend(focus_stats)
                diagnostics.extend(focus_diagnostics)
                _extend_unique(render_stats["camera_paths"], camera_paths)

                render_slot_timeout = _render_slot_timeout_seconds(config)
                try:
                    results = rendering_backend.render(
                        stage=flat_stage,
                        image_width=image_width,
                        image_height=image_height,
                        cameras=camera_paths,
                        base_dir=Path(usd_path).parent,
                        render_slot_timeout_sec=render_slot_timeout,
                    )
                except RemoteRenderingSlotTimeoutError as exc:
                    diagnostics.append(
                        _diagnostic(
                            "RENDER_GLOBAL_SLOT_TIMEOUT",
                            f"Timed out waiting for global render slot: {exc}",
                            usd_path=str(usd_path),
                            recommended_action=(
                                "Increase render_slot_timeout_sec, reduce "
                                "concurrent render jobs, or raise the global "
                                "remote render concurrency limit."
                            ),
                            details={"timeout_seconds": render_slot_timeout},
                        )
                    )
                    logger.error("Timed out waiting for render slot for %s", usd_path)
                    continue
                except TimeoutError as exc:
                    diagnostics.append(
                        _diagnostic(
                            "RENDER_BACKEND_TIMEOUT",
                            f"{backend_type} rendering timed out: {exc}",
                            usd_path=str(usd_path),
                            recommended_action=(
                                "Inspect the selected renderer logs and increase "
                                "its startup, request, or render deadline when "
                                "the workload is expected to take longer."
                            ),
                            details={
                                "backend": backend_type,
                                "exception_type": type(exc).__name__,
                            },
                        )
                    )
                    logger.error(
                        "%s rendering timed out for %s", backend_type, usd_path
                    )
                    continue

                try:
                    render_items = _render_result_items(results)
                except (TypeError, ValueError) as exc:
                    diagnostics.append(
                        _diagnostic(
                            "RENDER_RESULT_PARSE_ERROR",
                            f"Renderer returned an unsupported result shape: {exc}",
                            usd_path=str(usd_path),
                            recommended_action=(
                                "Update the render task contract or renderer "
                                "mock to return {'results': [...]}."
                            ),
                            details={"exception_type": type(exc).__name__},
                        )
                    )
                    logger.exception("Failed to parse render results for %s", usd_path)
                    continue

                for i, result in enumerate(render_items):
                    images = result.get("images", [])
                    if _status_is_failure(result):
                        camera_path = _result_camera_path(result, i, camera_paths)
                        status = result.get("status")
                        failed_status = status is not None and str(status) != "success"
                        code = (
                            "RENDER_PER_CAMERA_FAILURE"
                            if failed_status
                            else "RENDER_EMPTY_RESULT"
                        )
                        if code == "RENDER_EMPTY_RESULT":
                            message = str(
                                result.get("error") or "Renderer returned no images"
                            )
                        else:
                            message = str(
                                result.get("error")
                                or status
                                or "Renderer returned no images"
                            )
                        diagnostics.append(
                            _diagnostic(
                                code,
                                f"Render failed for {camera_path}: {message}",
                                usd_path=str(usd_path),
                                camera_path=camera_path,
                                recommended_action=(
                                    "Check render service logs, camera paths, "
                                    "and output USD asset dependencies."
                                ),
                                details={
                                    "status": status,
                                    "images_count": len(images),
                                },
                            )
                        )
                        logger.warning("Render failed for %s: %s", camera_path, message)
                        if failed_status:
                            continue

                    for j, img in enumerate(images):
                        out_name = f"render_{output_index}_{i}_{j}.png"
                        out_path = out_dir / out_name
                        _normalize_render_image_for_save(img, config).save(
                            str(out_path)
                        )
                        rendered.append(str(out_path))
                        logger.info("  Saved render: %s", out_path)

            except Exception as exc:
                message = f"Failed to render {usd_path}: {exc}"
                diagnostics.append(
                    _diagnostic(
                        "RENDER_UNEXPECTED_ERROR",
                        message,
                        usd_path=str(usd_path),
                        recommended_action=(
                            "Check renderer result shape, output USD validity, "
                            "camera paths, and render service connectivity."
                        ),
                        details={"exception_type": type(exc).__name__},
                    )
                )
                logger.exception("Failed to render %s", usd_path)

        context["rendered_image_paths"] = rendered
        render_stats["renders_count"] = len(rendered)
        render_stats["render_available"] = bool(rendered)
        render_stats["production_visual_evidence"] = has_production_visual_evidence(
            backend_type,
            render_count=len(rendered),
        )
        context["render_stats"] = render_stats
        context["render_diagnostics"] = diagnostics
        context["render_errors"] = [
            item for item in diagnostics if item.get("severity") == "error"
        ]
        logger.info("Rendered %d images", len(rendered))
        return context
