# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD render evidence helpers for Validation Agent visual templates."""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Mapping, Sequence
from hashlib import blake2s
from pathlib import Path
from typing import Any, Literal

from PIL import Image as PILImage

from world_understanding.functions.graphics.render_valid_adapter import (
    RENDER_RESPONSE_IMAGE_KEYS,
)
from world_understanding.functions.graphics.render_validation import (
    RENDER_MISSING_OUTPUT,
)
from world_understanding.rendering_backend_contract import (
    RENDERING_BACKEND_NAMES,
    UnknownRenderingBackendError,
    UnsupportedRenderingBackendError,
    validate_rendering_backend_for_surface,
)
from world_understanding.utils.nvcf_utils import get_base_url
from world_understanding.utils.usd.stage import prepare_stage_for_render
from world_understanding.validation.json_normalization import (
    StructuredJsonNormalizer,
)
from world_understanding.validation.rendering_backend_contract import (
    SUPPORTED_RENDER_BACKENDS,
    VALIDATION_RENDERING_BACKEND_NAMES,
    normalize_validation_rendering_backend,
)

RuntimeRenderStatus = Literal["completed", "failed", "unavailable", "skipped"]

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_RENDER_VIEWS: tuple[str, ...] = ("+x+y+z",)
_VALIDATION_RENDERING_SURFACE = "Validation Agent in-run USD rendering"
_SIDE_VIEW_DIRECTIONS: frozenset[str] = frozenset({"+x", "-x", "+y", "-y", "+z", "-z"})
_ASSET_KEY_STEM_CHARS = 16
_ASSET_KEY_DIGEST_CHARS = 8
_INVALID_RENDER_BACKEND_LABEL = "invalid"
_JSON_NORMALIZER = StructuredJsonNormalizer(unsupported_value_policy="stringify")
_json_value = _JSON_NORMALIZER.value
_VIEW_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "fixed_6": ("+x", "-x", "+y", "-y", "+z", "-z"),
}
_VIEW_DIRECTION_ALIASES: dict[str, str] = {
    "front": "+y",
    "back": "-y",
    "right": "+x",
    "left": "-x",
    "top": "+z",
    "bottom": "-z",
    "corner": "+x+y+z",
    "hero": "+x+y+z",
}


def render_usd_visual_evidence(
    *,
    usd_paths: Sequence[str | Path],
    working_dir: str | Path,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Render fixed whole-asset views for Validation Agent visual evidence.

    This helper intentionally reuses repository rendering backends instead of
    introducing a Validation Agent-specific rendering backend. Supported
    ``render_backend`` policy values are listed in
    ``SUPPORTED_RENDER_BACKENDS``.

    Returns:
        Runtime rendering evidence including ``status``, selected ``backend``,
        ``image_paths``, normalized backend response, output directory, issues,
        and metadata such as view and image counts.
    """

    backend_selection = _validation_render_backend_selection(
        _normalized_render_backend(policy)
    )
    if isinstance(backend_selection, dict):
        return backend_selection
    backend_name = backend_selection

    width = _optional_int(policy, "render_image_width", 1024)
    height = _optional_int(policy, "render_image_height", width)
    frames = (
        _optional_string(policy, "runtime_render_frames")
        or _optional_string(policy, "render_frames")
        or "0"
    )
    views = _runtime_render_views(policy)
    output_root = Path(working_dir) / "renders"
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        from pxr import Usd
    except ImportError as exc:
        return _unavailable_result(
            backend=backend_name,
            message=(
                "USD Python bindings are required for Validation Agent "
                "in-run rendering."
            ),
            details={"exception_type": type(exc).__name__},
        )
    backend_result = _create_render_backend(backend_name, policy)
    if isinstance(backend_result, dict):
        return backend_result
    backend = backend_result
    base_url_configured = backend_name == "remote"
    image_paths: list[str] = []
    response_entries: list[dict[str, Any]] = []
    response_cameras: list[str] = []
    render_issues: list[dict[str, Any]] = []
    stage_preparation: list[dict[str, Any]] = []
    scope_response_cameras = len(usd_paths) > 1

    for usd_index, usd_path_value in enumerate(usd_paths):
        usd_path = Path(usd_path_value)
        try:
            stage = Usd.Stage.Open(str(usd_path))
            if not stage:
                render_issues.append(
                    _runtime_issue(
                        code="render.usd_open_failed",
                        severity="fail",
                        message=f"Failed to open USD file for rendering: {usd_path}",
                        subject=str(usd_path),
                    )
                )
                continue

            stage, preparation_metadata = _prepare_stage_for_render(
                stage,
                backend_name=backend_name,
                policy=policy,
            )
            stage_preparation.append(
                {
                    "usd_path": str(usd_path),
                    **preparation_metadata,
                }
            )
            render_asset_base_dir = preparation_metadata.get("asset_base_dir")
            camera_specs = tuple(_view_spec(view) for view in views)
            camera_paths = [
                _add_view_camera(stage, label=label, direction=direction)
                for label, direction in camera_specs
            ]
            # OvRTXRenderingBackend accepts Usd.Stage here; its implementation
            # exports the stage to a temp USD before isolated subprocess render.
            render_result = backend.render(
                stage,
                cameras=camera_paths,
                image_width=width,
                image_height=height,
                frames=frames,
                base_dir=render_asset_base_dir,
            )
        except Exception as exc:
            render_issues.append(
                _runtime_issue(
                    code="render.runtime_render_failed",
                    severity="fail",
                    message=f"Validation Agent renderer failed for {usd_path}: {exc}",
                    subject=str(usd_path),
                    details={"exception_type": type(exc).__name__},
                )
            )
            continue

        asset_key = _asset_path_component(usd_path, usd_index)
        usd_output_dir, output_dir_issue = _reset_asset_output_dir(
            output_root,
            asset_key=asset_key,
            usd_path=usd_path,
        )
        if output_dir_issue is not None:
            render_issues.append(output_dir_issue)
            continue

        raw_results_value = render_result.get("results", ())
        if isinstance(raw_results_value, Sequence) and not isinstance(
            raw_results_value,
            str | bytes | bytearray,
        ):
            raw_results = raw_results_value
        else:
            raw_results = ()

        for index, (label, _) in enumerate(camera_specs):
            raw_entry = raw_results[index] if index < len(raw_results) else None
            if not isinstance(raw_entry, Mapping):
                render_issues.append(
                    _runtime_issue(
                        code="render.missing_view_evidence",
                        severity="fail",
                        message=(
                            f"Renderer returned no result for requested view {label!r}."
                        ),
                        subject=str(usd_path),
                        details={
                            "view": label,
                            "view_index": index,
                            "view_count": len(views),
                            "usd_path_count": len(usd_paths),
                        },
                    )
                )
                continue
            entry_paths, entry_issues = _save_entry_images(
                raw_entry,
                output_dir=usd_output_dir,
                asset_key=asset_key,
                view_label=label,
            )
            render_issues.extend(entry_issues)
            if not entry_paths and not entry_issues:
                render_issues.append(
                    _runtime_issue(
                        code="render.missing_view_evidence",
                        severity="fail",
                        message=(
                            "Renderer returned no image evidence for requested "
                            f"view {label!r}."
                        ),
                        subject=str(usd_path),
                        details={
                            "view": label,
                            "view_index": index,
                            "view_count": len(views),
                            "usd_path_count": len(usd_paths),
                        },
                    )
                )
            image_paths.extend(str(path) for path in entry_paths)
            response_camera = (
                _response_camera_key(asset_key, label)
                if scope_response_cameras
                else label
            )
            response_cameras.append(response_camera)
            response_entries.append(
                _json_render_response_entry(
                    raw_entry,
                    camera_label=label,
                    response_camera=response_camera,
                    camera_path=(
                        camera_paths[index] if index < len(camera_paths) else None
                    ),
                    image_paths=entry_paths,
                    usd_path=usd_path,
                )
            )

    if not response_entries and render_issues:
        return {
            "status": "failed",
            "backend": backend_name,
            "image_paths": [],
            "render_response": {
                "backend": backend_name,
                "cameras": [],
                "results": [],
                "status": "failed",
                "issues": render_issues,
            },
            "render_output_dir": str(output_root),
            "issues": render_issues,
            "metadata": {
                "backend": backend_name,
                "base_url_configured": base_url_configured,
                "image_count": 0,
                "view_count": len(views),
                "views": list(views),
                "usd_path_count": len(usd_paths),
                "frames": frames,
                "image_width": width,
                "image_height": height,
                "stage_preparation": stage_preparation,
            },
        }

    if not image_paths:
        render_issues.append(
            _runtime_issue(
                code="render.no_images_generated",
                severity="fail",
                message="The renderer completed but produced no image evidence.",
                details={"view_count": len(views), "usd_path_count": len(usd_paths)},
            )
        )

    status: RuntimeRenderStatus = (
        "completed" if image_paths and not render_issues else "failed"
    )
    return {
        "status": status,
        "backend": backend_name,
        "image_paths": image_paths,
        "render_response": {
            "backend": backend_name,
            "cameras": response_cameras,
            "results": response_entries,
            "status": status,
            "issues": render_issues,
        },
        "render_output_dir": str(output_root),
        "issues": render_issues,
        "metadata": {
            "backend": backend_name,
            "base_url_configured": base_url_configured,
            "image_count": len(image_paths),
            "response_cameras": response_cameras,
            "view_count": len(views),
            "views": list(views),
            "usd_path_count": len(usd_paths),
            "frames": frames,
            "image_width": width,
            "image_height": height,
            "stage_preparation": stage_preparation,
        },
    }


def _prepare_stage_for_render(
    stage: Any,
    *,
    backend_name: str,
    policy: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Prepare a composed USD stage for renderer backends.

    REST-style renderers receive a single exported root layer from the shared
    rendering backend. Flattening here resolves payloads, references, and
    sublayers before Validation Agent cameras are authored onto that prepared
    root layer.
    """

    flatten_before_render = _optional_bool(
        policy,
        "render_flatten_before_render",
        True,
    )
    normalize_materials = _optional_bool(
        policy,
        "render_normalize_materials",
        True,
    )
    prepared_stage, metadata = prepare_stage_for_render(
        stage,
        flatten=flatten_before_render,
        normalize_materials=normalize_materials,
    )
    metadata = {"backend": backend_name, **metadata}
    return prepared_stage, metadata


def _create_render_backend(
    backend_name: str,
    policy: Mapping[str, Any],
) -> Any | dict[str, Any]:
    # The primary path validates before opening the stage; keep this guard because
    # direct helper callers must receive the same structured failure contract.
    backend_selection = _validation_render_backend_selection(backend_name)
    if isinstance(backend_selection, dict):
        return backend_selection
    backend_name = backend_selection

    backend_config: dict[str, Any] = {}
    if backend_name == "remote":
        base_url = _optional_string(policy, "render_base_url")
        try:
            resolved_base_url = get_base_url(
                base_url,
                "RENDER_ENDPOINT",
                "NVCF_RENDER_FUNCTION_ID",
            )
        except ValueError as exc:
            return _unavailable_result(
                backend=backend_name,
                message=str(exc),
                details={
                    "render_backend": backend_name,
                    "required_env": ["RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID"],
                },
            )
        backend_config["base_url"] = resolved_base_url

    if backend_name == "ovrtx":
        backend_config.update(
            {
                "log_level": _optional_stripped_string(
                    policy,
                    "render_ovrtx_log_level",
                    "warn",
                ),
                "num_sensor_updates": _optional_int(
                    policy,
                    "render_ovrtx_num_sensor_updates",
                    32,
                ),
                "render_mode": _optional_stripped_string(
                    policy,
                    "render_ovrtx_mode",
                    "rt2",
                ),
            }
        )

    backend_label = "Remote" if backend_name == "remote" else "OVRTX"
    try:
        return _create_rendering_backend_from_factory(backend_name, backend_config)
    except (ImportError, AttributeError) as exc:
        return _unavailable_result(
            backend=backend_name,
            message=(
                f"{backend_label} rendering backend dependencies are required for "
                "Validation Agent in-run rendering."
            ),
            details={"exception_type": type(exc).__name__},
        )
    except Exception as exc:
        logger.warning(
            "%s rendering backend is unavailable: %s",
            backend_label,
            exc,
            exc_info=True,
        )
        return _unavailable_result(
            backend=backend_name,
            message=f"{backend_label} rendering backend is unavailable: {exc}",
            details={"exception_type": type(exc).__name__},
        )


def _create_rendering_backend_from_factory(
    backend_name: str,
    config: Mapping[str, Any],
) -> Any:
    """Load the renderer factory only when runtime USD rendering is attempted."""
    from world_understanding.functions.graphics.rendering_backend_factory import (
        create_rendering_backend,
    )

    return create_rendering_backend(backend_name, config)


def _validation_render_backend_selection(
    backend_value: Any,
) -> str | dict[str, Any]:
    try:
        return validate_rendering_backend_for_surface(
            backend_value,
            SUPPORTED_RENDER_BACKENDS,
            surface=_VALIDATION_RENDERING_SURFACE,
        )
    except UnknownRenderingBackendError as exc:
        message = (
            str(exc)
            if isinstance(backend_value, str)
            else "Invalid rendering backend selector; expected a string value."
        )
        return _failed_result(
            backend=_render_backend_label(backend_value),
            code="render.backend_unknown",
            message=message,
            details={
                "reason": "unknown_backend",
                "render_backend": _json_value(backend_value),
                "render_backend_type": type(backend_value).__name__,
                "canonical_render_backends": list(RENDERING_BACKEND_NAMES),
                "supported_render_backends": list(VALIDATION_RENDERING_BACKEND_NAMES),
            },
        )
    except UnsupportedRenderingBackendError as exc:
        return _unavailable_result(
            backend=_render_backend_label(backend_value),
            message=str(exc),
            details={
                "reason": "unsupported_by_validation",
                "render_backend": _json_value(backend_value),
                "canonical_render_backends": list(RENDERING_BACKEND_NAMES),
                "supported_render_backends": list(VALIDATION_RENDERING_BACKEND_NAMES),
            },
        )


def _render_backend_label(backend_value: Any) -> str | None:
    if backend_value is None:
        return None
    return (
        backend_value
        if isinstance(backend_value, str)
        else _INVALID_RENDER_BACKEND_LABEL
    )


def _unavailable_result(
    *,
    backend: str | None,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    issue = _runtime_issue(
        code="render.renderer_unavailable",
        severity="warn",
        message=message,
        details=dict(details or {}),
    )
    return {
        "status": "unavailable",
        "backend": backend,
        "image_paths": [],
        "render_response": None,
        "render_output_dir": None,
        "issues": [issue],
        "metadata": {
            "backend": backend,
            "base_url_configured": False,
        },
    }


def _failed_result(
    *,
    backend: str | None,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    issue = _runtime_issue(
        code=code,
        severity="fail",
        message=message,
        details=dict(details or {}),
    )
    return {
        "status": "failed",
        "backend": backend,
        "image_paths": [],
        "render_response": None,
        "render_output_dir": None,
        "issues": [issue],
        "metadata": {
            "backend": backend,
            "base_url_configured": False,
        },
    }


def expand_runtime_render_views(raw_views: Any) -> tuple[str, ...]:
    """Expand Validation Agent render view shorthand into concrete view labels."""

    if raw_views is None:
        return DEFAULT_RUNTIME_RENDER_VIEWS
    if isinstance(raw_views, str):
        views = _expand_runtime_render_view(raw_views)
        return views or DEFAULT_RUNTIME_RENDER_VIEWS
    if isinstance(raw_views, Sequence) and not isinstance(
        raw_views,
        bytes | bytearray,
    ):
        views = tuple(
            expanded_view
            for view in raw_views
            for expanded_view in _expand_runtime_render_view(str(view))
        )
        return views or DEFAULT_RUNTIME_RENDER_VIEWS
    return DEFAULT_RUNTIME_RENDER_VIEWS


def _runtime_render_views(policy: Mapping[str, Any]) -> tuple[str, ...]:
    raw_views = (
        policy.get("runtime_render_views")
        or policy.get("render_view_directions")
        or policy.get("expected_cameras")
    )
    return expand_runtime_render_views(raw_views)


def _expand_runtime_render_view(view: str) -> tuple[str, ...]:
    normalized = view.strip()
    if not normalized:
        return ()
    return _VIEW_GROUP_ALIASES.get(normalized.lower(), (normalized,))


def _view_spec(view: str) -> tuple[str, str]:
    view = view.strip()
    if not view:
        return ("corner", "+x+y+z")
    alias = _VIEW_DIRECTION_ALIASES.get(view.lower())
    if alias is not None:
        return (view, alias)
    if _looks_like_direction(view):
        return (view, view.lower())
    return (view, "+x+y+z")


def _looks_like_direction(value: str) -> bool:
    return bool(re.fullmatch(r"([+-](?:\d+(?:\.\d+)?)?[xyz])+", value.lower()))


def _add_view_camera(stage: Any, *, label: str, direction: str) -> str:
    from world_understanding.utils.usd.camera import (
        add_corner_view_camera,
        add_side_view_camera,
    )

    camera_path = f"/ValidationAgentCameras/{_safe_path_component(label)}"
    add_camera = (
        add_side_view_camera
        if direction in _SIDE_VIEW_DIRECTIONS
        else add_corner_view_camera
    )
    add_camera(
        stage,
        camera_path=camera_path,
        direction=_camera_helper_direction(direction),
        margin=1.2,
        focal_length=50.0,
        horizontal_aperture=36.0,
        vertical_aperture=36.0,
    )
    return camera_path


def _camera_helper_direction(direction: str) -> str:
    # The shared side-camera helper names vertical directions by viewing vector.
    # Validation view labels use asset-side labels, so +z means top evidence.
    if direction == "+z":
        return "-z"
    if direction == "-z":
        return "+z"
    return direction


def _save_entry_images(
    entry: Mapping[str, Any],
    *,
    output_dir: Path,
    asset_key: str,
    view_label: str,
) -> tuple[tuple[Path, ...], tuple[dict[str, Any], ...]]:
    saved_paths: list[Path] = []
    issues: list[dict[str, Any]] = []
    for frame_index, image in enumerate(_entry_images(entry)):
        filename = (
            f"{asset_key}_{_safe_path_component(view_label)}_{frame_index:04d}.png"
        )
        output_path = output_dir / filename
        if isinstance(image, PILImage.Image):
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path)
            except OSError as exc:
                issues.append(
                    _image_artifact_issue(
                        code=RENDER_MISSING_OUTPUT,
                        message=(
                            "Renderer returned image evidence, but Validation "
                            f"Agent could not write {output_path}: {exc}"
                        ),
                        output_path=output_path,
                        view_label=view_label,
                        frame_index=frame_index,
                        exception=exc,
                    )
                )
                continue
        elif isinstance(image, str | Path):
            source_path, source_was_relative = _entry_image_path(image, output_dir)
            if source_path is None:
                issues.append(
                    _image_artifact_issue(
                        code=RENDER_MISSING_OUTPUT,
                        message="Renderer reported a blank image artifact path.",
                        output_path=output_path,
                        view_label=view_label,
                        frame_index=frame_index,
                        source_value=str(image),
                    )
                )
                continue
            if not source_path.is_file():
                issues.append(
                    _image_artifact_issue(
                        code=RENDER_MISSING_OUTPUT,
                        message=(
                            "Renderer reported an image artifact, but the file "
                            f"does not exist: {source_path}"
                        ),
                        output_path=output_path,
                        view_label=view_label,
                        frame_index=frame_index,
                        source_path=source_path,
                        source_value=str(image),
                        source_was_relative=source_was_relative,
                        source_path_base=output_dir,
                    )
                )
                continue
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not _same_file(source_path, output_path):
                    shutil.copy2(source_path, output_path)
            except shutil.SameFileError:
                pass
            except OSError as exc:
                issues.append(
                    _image_artifact_issue(
                        code=RENDER_MISSING_OUTPUT,
                        message=(
                            "Renderer reported an image artifact, but Validation "
                            f"Agent could not copy it to {output_path}: {exc}"
                        ),
                        output_path=output_path,
                        view_label=view_label,
                        frame_index=frame_index,
                        source_path=source_path,
                        source_value=str(image),
                        source_was_relative=source_was_relative,
                        source_path_base=output_dir,
                        exception=exc,
                    )
                )
                continue
        else:
            continue

        # Defensive check for storage layers or image writers that return
        # without making the expected artifact visible immediately.
        if not output_path.is_file():
            issues.append(
                _image_artifact_issue(
                    code=RENDER_MISSING_OUTPUT,
                    message=(
                        "Renderer image evidence was processed, but the expected "
                        f"artifact is missing: {output_path}"
                    ),
                    output_path=output_path,
                    view_label=view_label,
                    frame_index=frame_index,
                )
            )
            continue
        saved_paths.append(output_path)
    return tuple(saved_paths), tuple(issues)


def _entry_image_path(image: str | Path, output_dir: Path) -> tuple[Path | None, bool]:
    if isinstance(image, str):
        value = image.strip()
        if not value:
            return None, False
        path = Path(value)
    else:
        path = Path(image)
    if path.is_absolute():
        return path, False
    return output_dir / path, True


def _same_file(source_path: Path, output_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        return source_path.samefile(output_path)
    except OSError:
        return False


def _entry_images(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    for key in RENDER_RESPONSE_IMAGE_KEYS:
        if key not in entry:
            continue
        images = entry.get(key)
        if images is None:
            continue
        # An explicit empty list means the renderer reported this evidence slot
        # but produced no images; preserve that instead of falling through.
        if isinstance(images, Sequence) and not isinstance(
            images,
            str | bytes | bytearray,
        ):
            return tuple(images)
        return (images,)
    return ()


def _image_artifact_issue(
    *,
    code: str,
    message: str,
    output_path: Path,
    view_label: str,
    frame_index: int,
    source_path: Path | None = None,
    source_value: str | None = None,
    source_was_relative: bool = False,
    source_path_base: Path | None = None,
    exception: OSError | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "view": view_label,
        "frame_index": frame_index,
        "expected_output_path": str(output_path),
    }
    if source_path is not None:
        details["source_path"] = str(source_path)
    if source_value is not None:
        details["reported_source_path"] = source_value
    if source_was_relative:
        details["source_path_was_relative"] = True
        if source_path_base is not None:
            details["source_path_base"] = str(source_path_base)
    if exception is not None:
        details["exception_type"] = type(exception).__name__
    return _runtime_issue(
        code=code,
        severity="fail",
        message=message,
        subject=str(source_path or output_path),
        details=details,
    )


def _json_render_response_entry(
    entry: Mapping[str, Any],
    *,
    camera_label: str,
    response_camera: str,
    camera_path: str | None,
    image_paths: Sequence[Path],
    usd_path: Path,
) -> dict[str, Any]:
    status = entry.get("status")
    return {
        "camera": response_camera,
        "camera_label": camera_label,
        "camera_path": camera_path,
        "usd_path": str(usd_path),
        "images": [str(path) for path in image_paths],
        "frame_count": _json_scalar(entry.get("frame_count", len(image_paths))),
        "render_time": _json_scalar(entry.get("render_time")),
        "status": str(status) if status is not None else None,
        "error": _json_scalar(entry.get("error")),
    }


def _runtime_issue(
    *,
    code: str,
    severity: Literal["info", "warn", "fail"],
    message: str,
    subject: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
        "details": dict(details or {}),
    }
    if subject is not None:
        issue["subject"] = subject
    return issue


def _safe_path_component(value: str) -> str:
    value = value.strip("/").replace("+", "plus_").replace("-", "minus_")
    safe = re.sub(r"[^A-Za-z0-9_.]+", "_", value)
    return safe.strip("._") or "view"


def _asset_path_component(usd_path: Path, usd_index: int) -> str:
    digest_source = f"{usd_index}:{usd_path.resolve().as_posix()}"
    digest = blake2s(
        digest_source.encode("utf-8"),
        digest_size=_ASSET_KEY_DIGEST_CHARS // 2,
    ).hexdigest()
    stem = _safe_path_component(usd_path.stem)[:_ASSET_KEY_STEM_CHARS]
    return f"{usd_index:03d}_{stem}_{digest}"


def _reset_asset_output_dir(
    output_root: Path,
    *,
    asset_key: str,
    usd_path: Path,
) -> tuple[Path, dict[str, Any] | None]:
    output_dir = output_root / asset_key
    try:
        output_root_resolved = output_root.resolve()
        output_dir_resolved = output_dir.resolve()
        output_dir_resolved.relative_to(output_root_resolved)
        if output_dir.is_dir() and not output_dir.is_symlink():
            shutil.rmtree(output_dir)
        elif output_dir.exists():
            output_dir.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as exc:
        return output_dir, _runtime_issue(
            code="render.output_dir_prepare_failed",
            severity="fail",
            message=(
                "Validation Agent could not prepare a clean render output "
                f"directory for {usd_path}: {exc}"
            ),
            subject=str(output_dir),
            details={
                "asset_key": asset_key,
                "usd_path": str(usd_path),
                "exception_type": type(exc).__name__,
            },
        )
    return output_dir, None


def _response_camera_key(asset_key: str, camera_label: str) -> str:
    return f"{asset_key}:{_safe_path_component(camera_label)}"


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _optional_string(policy: Mapping[str, Any], key: str) -> str | None:
    value = policy.get(key)
    return value if isinstance(value, str) else None


def _optional_bool(policy: Mapping[str, Any], key: str, default: bool) -> bool:
    value = policy.get(key)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean policy value {key}={value!r}")
    raise ValueError(f"Invalid boolean policy value {key}={value!r}")


def _optional_stripped_string(
    policy: Mapping[str, Any],
    key: str,
    default: str,
) -> str:
    value = _optional_string(policy, key)
    return value.strip() if value and value.strip() else default


def _normalized_render_backend(policy: Mapping[str, Any]) -> Any:
    raw_backend_name = policy.get("render_backend")
    backend_name = normalize_validation_rendering_backend(raw_backend_name)
    if isinstance(raw_backend_name, str) and not raw_backend_name.strip():
        logger.info("Blank render_backend value defaults to remote")
    if backend_name != raw_backend_name:
        logger.debug(
            "Normalized render_backend value from %r to %r",
            raw_backend_name,
            backend_name,
        )
    return backend_name


def _optional_int(
    policy: Mapping[str, Any],
    key: str,
    default: int,
) -> int:
    value = policy.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, str):
        try:
            parsed_value = int(value.strip())
        except ValueError:
            logger.warning("Ignoring invalid integer policy value %s=%r", key, value)
        else:
            if parsed_value > 0:
                return parsed_value
            logger.warning(
                "Ignoring non-positive integer policy value %s=%r", key, value
            )
    return default
