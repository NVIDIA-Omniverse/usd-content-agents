# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD rendering functions for REST API-based render services.

The module was originally named ``render_nvcf`` because the only supported
REST renderer was NVIDIA Cloud Functions. The same client path now also targets
local or external OVRTX rendering services through ``RENDER_ENDPOINT``/
``base_url``. In this codebase, ``remote`` means a REST API renderer, not an
in-process local backend like Warp.
"""

import io
import json
import logging
import os
import random
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import numpy as np
import requests
from PIL import Image
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

from world_understanding.config.s3 import WU_S3_BUCKET, WU_S3_PROFILE, WU_S3_REGION
from world_understanding.functions.graphics.material_targets import (
    preview_fallbacks_enabled_for_material_target,
)
from world_understanding.rendering_backend_contract import (
    validate_remote_render_max_workers,
)

if TYPE_CHECKING:  # pragma: no cover
    from pxr import Usd

from world_understanding.utils.data_uri import should_use_data_uri
from world_understanding.utils.image_utils import (
    base64_to_image,
    base64_to_numpy,
    process_depth_map,
)
from world_understanding.utils.nvcf_utils import (
    create_nvcf_headers,
    get_base_url,
    get_nvcf_api_key,
    s3_uri_to_https_url,
)
from world_understanding.utils.s3_utils import delete_s3_path, upload_file_to_s3
from world_understanding.utils.usd.material import (
    add_ovrtx_preview_fallbacks_for_texture_file_materials,
    add_ovrtx_preview_fallbacks_to_stage_file,
    bake_texture_file_materials_to_display_color_for_render,
    get_local_mdl_assets,
    get_local_texture_file_assets,
)
from world_understanding.utils.usd.stage import (
    create_data_uri_from_file,
    has_uri_scheme,
    is_windows_drive_path,
)

logger = logging.getLogger(__name__)


class RenderingStatus(StrEnum):
    """Status codes returned by a REST rendering service."""

    blank_render = "blank_render"
    empty_response = "empty_response"
    load_error = "load_error"
    exception = "exception"
    success = "success"


def _http_error_payload(response: requests.Response) -> dict[str, Any]:
    """Return a structured renderer error payload for an HTTP response."""
    try:
        payload = response.json()
    except ValueError:
        return {"error": response.text[:500] or str(response.reason)}

    detail = payload.get("detail") if isinstance(payload, dict) else payload
    if isinstance(detail, dict):
        return detail
    if detail is not None:
        return {"error": str(detail)[:500]}
    return {"error": json.dumps(payload, sort_keys=True)[:500]}


def _http_error_detail(response: requests.Response) -> str:
    """Return the useful renderer error message for an HTTP error response."""
    payload = _http_error_payload(response)
    if payload.get("error"):
        return str(payload["error"])
    return json.dumps(payload, sort_keys=True)[:500]


# Note: decode_base64_to_image and decode_base64_to_numpy have been moved to
# world_understanding.utils.image_utils for reusability

# Sensor name mapping from V2 to V1 format
_V2_SENSOR_TO_V1 = {
    "rgb": "images",
    "distance_to_image_plane": "linear_depth",
    "distance_to_camera": "linear_depth",
    "instance_segmentation": "instance_id_segmentation",
}


def _is_v2_response(result: dict[str, Any]) -> bool:
    """Check if a render response uses the V2 format."""
    return "rendered_data" in result and "total_cameras" in result


def _convert_v2_sensor(sensor_obj: dict[str, Any]) -> str | np.ndarray:
    """Convert a V2 sensor object {type, data, shape, dtype} to an array or passthrough string.

    Returns:
        np.ndarray if shape/dtype metadata is present (decoded raw array data).
        str (the original base64 string) if shape is missing or data is empty.
    """
    import base64

    data_b64 = sensor_obj.get("data", "")
    shape = sensor_obj.get("shape")
    dtype_str = sensor_obj.get("dtype", "uint8")

    if not data_b64 or not shape:
        return data_b64

    # Decode raw array data
    raw_bytes = base64.b64decode(data_b64)
    arr = np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape)

    return arr


def _convert_v2_to_v1(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a V2 render response to V1 format for backward compatibility.

    V2 structure: rendered_data[camera][frame] = {rgb: {type, data, shape, dtype}}
    V1 structure: images[frame][camera] = {images: base64_png, sensor: base64_raw}
    """
    import base64 as b64mod

    rendered_data = result.get("rendered_data", {})
    v1_images: dict[str, dict[str, dict[str, Any]]] = {}

    for camera_key, frames in rendered_data.items():
        for frame_str, sensors_dict in frames.items():
            if frame_str not in v1_images:
                v1_images[frame_str] = {}

            camera_data: dict[str, Any] = {}
            for sensor_name, sensor_obj in sensors_dict.items():
                v1_name = _V2_SENSOR_TO_V1.get(sensor_name, sensor_name)

                if not isinstance(sensor_obj, dict) or "data" not in sensor_obj:
                    camera_data[v1_name] = sensor_obj
                    continue

                arr = _convert_v2_sensor(sensor_obj)
                if isinstance(arr, np.ndarray) and v1_name == "images":
                    # Convert to PNG base64 for the main image
                    img = Image.fromarray(arr.astype(np.uint8))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    camera_data[v1_name] = b64mod.b64encode(buf.getvalue()).decode()
                elif isinstance(arr, np.ndarray):
                    # Raw sensor data as base64 numpy
                    camera_data[v1_name] = b64mod.b64encode(arr.tobytes()).decode()
                else:
                    camera_data[v1_name] = arr

            v1_images[frame_str][camera_key] = camera_data

    logger.info(
        "Converted V2 response to V1 format: %d cameras, %d frames",
        result.get("total_cameras", 0),
        result.get("total_frames", 0),
    )

    return {
        "images": v1_images,
        "status": RenderingStatus.success,
        "error": None,
    }


def _export_stage_and_get_url(
    stage_path: str,
    use_data_uri: bool,
    s3_bucket: str,
    s3_profile: str | None,
    s3_region: str,
) -> tuple[str, str | None]:
    """Export USD stage and return URL and optional S3 URI for cleanup.

    Args:
        stage_path: Path to the USD stage
        use_data_uri: If True, use data URI encoding instead of S3 upload.
        s3_bucket: S3 bucket for stage upload (ignored if use_data_uri=True).
        s3_profile: AWS profile for S3 upload (ignored if use_data_uri=True).
        s3_region: AWS region where the S3 bucket is located (ignored if use_data_uri=True).

    Returns:
        Tuple containing the asset URL and optional S3 URI for cleanup.
    """
    if use_data_uri:
        asset_url = create_data_uri_from_file(stage_path)
        logger.info("Created data URI for stage")
        return asset_url, None
    else:
        unique_id = uuid.uuid4().hex
        # Preserve original file extension so renderers can detect format
        ext = Path(stage_path).suffix or ".usd"
        s3_key = f"nvcf-renders/{unique_id}/stage{ext}"
        s3_uri = upload_file_to_s3(
            file_path=stage_path,
            s3_path=f"s3://{s3_bucket}/{s3_key}",
            profile_name=s3_profile,
        )
        asset_url = s3_uri_to_https_url(s3_uri, s3_region)
        logger.info("Uploaded stage to S3: %s", asset_url)
        return asset_url, s3_uri


def _prefer_preview_surface_for_remote_export(stage_path: Path) -> int:
    """Remove MDL material outputs from exported stages that have preview output.

    The remote OVRTX/NVCF renderer can select the MDL render-context output when
    a Material has both ``outputs:surface`` and ``outputs:mdl:surface``. For
    assets where the MDL module is unavailable or unsupported, that produces the
    renderer's red error material even though a valid UsdPreviewSurface fallback
    exists. This only mutates the temporary export sent to the renderer.
    """
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        return 0

    def has_usd_preview_surface(material: UsdShade.Material) -> bool:
        try:
            surface_source = material.ComputeSurfaceSource()
        except Exception:
            return False

        if not surface_source:
            return False

        source_shader = surface_source[0]
        if not source_shader:
            return False

        shader_id_attr = source_shader.GetIdAttr()
        return bool(shader_id_attr and shader_id_attr.Get() == "UsdPreviewSurface")

    removed_count = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Material":
            continue

        material = UsdShade.Material(prim)
        if not has_usd_preview_surface(material):
            continue

        for prop in list(prim.GetProperties()):
            prop_name = prop.GetName()
            if prop_name.startswith("outputs:mdl:"):
                prim.RemoveProperty(prop_name)
                removed_count += 1

    if removed_count:
        stage.GetRootLayer().Save()

    return removed_count


def _add_texture_file_fallbacks_for_remote_export(
    stage_path: Path,
    *,
    connect_diffuse_texture: bool = False,
    preserve_mdl_surface: bool = False,
) -> tuple[int, int]:
    """Bake local texture-file materials into a renderer-safe preview fallback.

    OVRTX service builds can load geometry and UsdPreviewSurface materials even
    when authored SimReady/MDL-style texture inputs are not evaluated directly.
    For self-contained bundles, ``connect_diffuse_texture`` keeps the generated
    albedo as a simple ``UsdUVTexture`` graph. USD-only fallbacks still bake
    sampled albedo to ``primvars:displayColor`` because the texture files are
    not uploaded with that export path. ``preserve_mdl_surface`` skips fallback
    authoring for materials that already have a connected MDL surface, which
    preserves richer OmniPBR/MDL texture response when bundled assets resolve.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        return 0, 0

    display_color_bakes = 0
    if not connect_diffuse_texture:
        display_color_bakes = bake_texture_file_materials_to_display_color_for_render(
            stage,
        )
    textured_fallbacks = add_ovrtx_preview_fallbacks_for_texture_file_materials(
        stage,
        override_existing_surface=True,
        connect_diffuse_texture=connect_diffuse_texture,
        diffuse_color_primvar=(
            "displayColor"
            if display_color_bakes and not connect_diffuse_texture
            else None
        ),
        skip_connected_mdl_surface=preserve_mdl_surface,
    )
    if display_color_bakes or textured_fallbacks:
        stage.GetRootLayer().Save()
    return display_color_bakes, textured_fallbacks


def _bundle_stage_with_local_assets(
    stage: "Usd.Stage",
    temp_dir: Path,
    base_dir: str | Path | None = None,
    has_local_composition_arcs: bool | None = None,
    add_preview_fallbacks: bool | None = None,
) -> tuple[Path | None, bool]:
    """Bundle USD stage with local MDL and texture assets into a ZIP archive.

    This function checks if the stage references any local MDL files or texture
    files (PNG, JPG, EXR, etc. from UsdPreviewSurface shaders). If so, it:
    1. Creates a directory structure with the USD, MDL, and texture files
    2. Updates asset paths in the USD to be relative
    3. Creates a ZIP archive containing everything

    Args:
        stage: USD stage to bundle
        temp_dir: Temporary directory for creating the bundle
        base_dir: Base directory for resolving relative texture paths. If None,
                 uses the stage's root layer directory.
        has_local_composition_arcs: Precomputed composition-arc guard result.
                 If None, the guard is evaluated here.
        add_preview_fallbacks: If True, author render-export UsdPreviewSurface
                 bridges for MaterialX OpenPBR materials. If None or False, no
                 PreviewSurface fallbacks are authored by this helper; higher
                 level callers derive this from ``material_target``.

    Returns:
        Tuple of (zip_path, was_bundled):
            - zip_path: Path to the created ZIP file, or None if no bundling needed
            - was_bundled: True if bundling occurred, False otherwise
    """
    import shutil

    from pxr import Sdf

    if has_local_composition_arcs is None:
        has_local_composition_arcs = _stage_has_local_composition_arcs(stage)
    if has_local_composition_arcs:
        raise RuntimeError(
            "Remote REST rendering cannot bundle non-flattened USD stages with "
            "local sublayers, references, or payloads. Flatten the stage before "
            "remote rendering, for example with prepare_stage_for_render(..., "
            "flatten=True) or by setting flatten_before_render=True."
        )

    # Get all MDL assets from the stage
    mdl_assets = get_local_mdl_assets(stage, base_dir=base_dir)

    # Filter to only local, existing files
    local_assets = [a for a in mdl_assets if a["is_local"] and a["resolved_path"]]

    # Get all texture file assets from the stage
    texture_assets = get_local_texture_file_assets(stage, base_dir=base_dir)
    local_textures = [a for a in texture_assets if a["is_local"] and a["resolved_path"]]

    if not local_assets and not local_textures:
        logger.info("No local MDL or texture assets found, skipping bundling")
        return None, False

    logger.info(
        f"Found {len(local_assets)} local MDL assets and "
        f"{len(local_textures)} local texture files to bundle"
    )

    # Create bundle directory structure
    bundle_dir = temp_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Track copied directories to avoid duplicates (for MDL)
    copied_dirs: dict[str, str] = {}  # original_dir -> relative_path_in_bundle
    copied_mdl_files: dict[str, str] = {}  # resolved MDL file -> relative bundle path

    # ---- Copy MDL files and their directories ----
    if local_assets:
        mdl_dir = bundle_dir / "mdl_materials"
        mdl_dir.mkdir(parents=True, exist_ok=True)

        for asset in local_assets:
            mdl_file = Path(asset["resolved_path"])
            mdl_parent = mdl_file.parent

            # Use directory name as unique identifier
            dir_name = mdl_parent.name

            if str(mdl_parent) not in copied_dirs:
                # Copy entire directory to preserve textures
                dest_dir = mdl_dir / dir_name

                # Handle duplicate directory names
                counter = 1
                while dest_dir.exists():
                    dest_dir = mdl_dir / f"{dir_name}_{counter}"
                    counter += 1

                try:
                    shutil.copytree(mdl_parent, dest_dir)
                    copied_dirs[str(mdl_parent)] = dest_dir.relative_to(
                        bundle_dir
                    ).as_posix()
                    logger.debug(f"Copied MDL directory: {mdl_parent} -> {dest_dir}")
                except Exception as e:
                    logger.warning(f"Failed to copy MDL directory {mdl_parent}: {e}")
                    continue

            rel_dir = copied_dirs.get(str(mdl_parent))
            if rel_dir is not None:
                copied_mdl_files[str(mdl_file.resolve())] = f"{rel_dir}/{mdl_file.name}"

    # ---- Copy texture files ----
    # Track: resolved_path -> relative_path_in_bundle
    copied_textures: dict[str, str] = {}

    if local_textures:
        textures_dir = bundle_dir / "textures"
        textures_dir.mkdir(parents=True, exist_ok=True)

        seen_filenames: dict[str, int] = {}  # filename -> counter for collisions

        for tex in local_textures:
            resolved = str(Path(tex["resolved_path"]).resolve())
            if resolved in copied_textures:
                continue

            src_path = Path(resolved)
            filename = src_path.name

            # Handle filename collisions
            if filename in seen_filenames:
                seen_filenames[filename] += 1
                stem = src_path.stem
                suffix = src_path.suffix
                filename = f"{stem}_{seen_filenames[filename]}{suffix}"
            else:
                seen_filenames[filename] = 0

            dest_path = textures_dir / filename
            try:
                shutil.copy2(str(src_path), str(dest_path))
                rel_path = dest_path.relative_to(bundle_dir).as_posix()
                copied_textures[resolved] = rel_path
                logger.debug(f"Copied texture: {src_path} -> {dest_path}")
            except Exception as e:
                logger.warning(f"Failed to copy texture {src_path}: {e}")

    if not copied_dirs and not copied_textures:
        logger.warning("No assets were copied, skipping bundling")
        return None, False

    copied_texture_attrs: dict[tuple[str, str], str] = {}
    for tex in local_textures:
        resolved = tex.get("resolved_path")
        if not resolved:  # pragma: no cover - local_textures is already prefiltered
            continue
        rel_path = copied_textures.get(str(Path(resolved).resolve()))
        if rel_path is not None:
            copied_texture_attrs[(tex["prim_path"], tex["attr_name"])] = rel_path

    # Export the stage and update paths
    root_layer = stage.GetRootLayer()
    if base_dir is None:
        asset_base_dir = (
            Path(root_layer.realPath).parent if root_layer.realPath else Path.cwd()
        )
    else:
        asset_base_dir = Path(base_dir)

    # Create a copy of the layer to modify
    temp_usda = bundle_dir / "stage.usda"
    root_layer.Export(str(temp_usda))

    if add_preview_fallbacks:
        preview_fallbacks = add_ovrtx_preview_fallbacks_to_stage_file(temp_usda)
        if preview_fallbacks:
            logger.info(
                "Updated %d OpenPBR material fallback(s) for remote render export",
                preview_fallbacks,
            )

    # Reopen the exported layer to update paths
    exported_layer = Sdf.Layer.FindOrOpen(str(temp_usda))
    if not exported_layer:
        logger.error("Failed to open exported layer for path updates")
        return None, False

    # Update asset paths in the exported layer
    def update_asset_paths_in_layer(layer: Sdf.Layer) -> int:
        """Recursively update MDL and texture asset paths in a layer."""
        updated_count = 0

        def process_prim_spec(prim_spec):
            nonlocal updated_count
            prim_path = str(prim_spec.path)

            for attr_name in list(prim_spec.attributes.keys()):
                attr_spec = prim_spec.attributes[attr_name]
                value = attr_spec.default

                if value is None:
                    continue

                # Only process Sdf.AssetPath values
                if not isinstance(value, Sdf.AssetPath):
                    continue

                try:
                    asset_path = value.path if hasattr(value, "path") else str(value)
                except Exception:  # pragma: no cover - defensive Sdf.AssetPath access
                    continue

                if not asset_path:
                    continue

                # --- MDL path rewriting ---
                if attr_name == "info:mdl:sourceAsset":
                    resolved_mdl_path = _resolve_export_asset_path(
                        asset_path,
                        asset_base_dir,
                    )

                    new_path = copied_mdl_files.get(resolved_mdl_path)
                    if new_path is None:
                        for orig_dir, rel_bundle_path in copied_dirs.items():
                            if asset_path.startswith(orig_dir) or (
                                os.path.isabs(asset_path)
                                and str(Path(asset_path).parent) == orig_dir
                            ):
                                mdl_filename = Path(asset_path).name
                                new_path = f"{rel_bundle_path}/{mdl_filename}"
                                break

                    if new_path is not None:
                        attr_spec.default = Sdf.AssetPath(new_path)
                        updated_count += 1
                        logger.debug(f"Updated MDL path: {asset_path} -> {new_path}")
                    continue

                # --- Texture path rewriting ---
                if not copied_textures:
                    continue

                new_path = copied_texture_attrs.get((prim_path, attr_name))
                if new_path is None:
                    resolved_texture_path = _resolve_export_asset_path(
                        asset_path,
                        asset_base_dir,
                    )
                    new_path = copied_textures.get(resolved_texture_path)

                if new_path is not None:
                    attr_spec.default = Sdf.AssetPath(new_path)
                    updated_count += 1
                    logger.debug(f"Updated texture path: {asset_path} -> {new_path}")

            # Process child prims
            for child in prim_spec.nameChildren:
                process_prim_spec(child)

        for prim in layer.rootPrims:
            process_prim_spec(prim)

        return updated_count

    updated_count = update_asset_paths_in_layer(exported_layer)
    logger.info(f"Updated {updated_count} asset paths to relative paths")

    # Save the modified layer
    exported_layer.Save()

    if add_preview_fallbacks:
        display_color_bakes, textured_fallbacks = (
            _add_texture_file_fallbacks_for_remote_export(
                temp_usda,
                connect_diffuse_texture=True,
                preserve_mdl_surface=True,
            )
        )
        if display_color_bakes:
            logger.info(
                "Baked %d textured mesh(es) to displayColor for remote render export",
                display_color_bakes,
            )
        if textured_fallbacks:
            logger.info(
                "Updated %d texture-file material fallback(s) for remote render export",
                textured_fallbacks,
            )

        removed_mdl_outputs = _prefer_preview_surface_for_remote_export(temp_usda)
        if removed_mdl_outputs:
            logger.info(
                "Removed %d MDL material outputs from remote render export",
                removed_mdl_outputs,
            )
        exported_layer = Sdf.Layer.FindOrOpen(str(temp_usda))
        if exported_layer:
            post_fallback_updates = update_asset_paths_in_layer(exported_layer)
            if post_fallback_updates:
                exported_layer.Save()
                logger.info(
                    "Updated %d fallback-authored asset path(s) to relative paths",
                    post_fallback_updates,
                )

    # Create ZIP archive
    zip_path = temp_dir / "bundle.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in bundle_dir.rglob("*"):
            if file_path.is_file():
                arc_name = file_path.relative_to(bundle_dir).as_posix()
                zf.write(file_path, arc_name)
                logger.debug(f"Added to ZIP: {arc_name}")

    zip_size = zip_path.stat().st_size
    logger.info(f"Created asset bundle: {zip_path} ({zip_size / 1024:.1f} KB)")

    return zip_path, True


def _resolve_export_asset_path(asset_path: str, asset_base_dir: Path) -> str:
    parsed = urlparse(asset_path)
    if parsed.scheme.lower() == "file":
        normalized_path = unquote(parsed.path)
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            normalized_path = f"//{parsed.netloc}{normalized_path}"
        if (
            len(normalized_path) >= 4
            and normalized_path[0] == "/"
            and normalized_path[2] == ":"
        ):
            normalized_path = normalized_path[1:]
        candidate_path = Path(normalized_path)
    else:
        candidate_path = Path(asset_path)
    if not candidate_path.is_absolute() and not is_windows_drive_path(
        str(candidate_path),
    ):
        candidate_path = asset_base_dir / candidate_path
    try:
        return str(candidate_path.resolve(strict=False))
    except (OSError, RuntimeError) as exc:
        logger.warning(
            "Failed to resolve asset path %s relative to %s: %s",
            asset_path,
            asset_base_dir,
            exc,
        )
        return str(candidate_path)


def _stage_has_local_composition_arcs(stage: "Usd.Stage") -> bool:
    """Return true when exporting only the root layer would drop local arcs."""
    root_layer = stage.GetRootLayer()
    if any(_is_local_composition_asset_path(path) for path in root_layer.subLayerPaths):
        return True

    from pxr import Sdf

    prim_paths: list[Any] = []
    root_layer.Traverse("/", prim_paths.append)
    for prim_path in prim_paths:
        prim_spec = root_layer.GetPrimAtPath(prim_path)
        if not isinstance(prim_spec, Sdf.PrimSpec):
            continue
        if _list_editor_has_local_composition_arc(prim_spec.referenceList):
            return True
        if _list_editor_has_local_composition_arc(prim_spec.payloadList):
            return True
    return False


def _list_editor_has_local_composition_arc(list_editor: Any) -> bool:
    # deletedItems are list-op removals, not dependencies to resolve or upload.
    for field in (
        "prependedItems",
        "appendedItems",
        "addedItems",
        "explicitItems",
    ):
        for item in getattr(list_editor, field):
            if _is_local_composition_asset_path(getattr(item, "assetPath", "")):
                return True
    return False


def _is_local_composition_asset_path(asset_path: str) -> bool:
    if not asset_path:
        return False
    if asset_path.startswith("anon:"):
        return True
    if is_windows_drive_path(asset_path):
        return True
    if asset_path.startswith("file:"):
        return True
    return not has_uri_scheme(asset_path)


def export_stage_to_s3(
    stage: "Usd.Stage",
    s3_bucket: str = WU_S3_BUCKET,
    s3_region: str = WU_S3_REGION,
    s3_profile: str = WU_S3_PROFILE,
    use_data_uri: bool | None = None,
    bundle_mdl_assets: bool = True,
    base_dir: str | Path | None = None,
    add_preview_fallbacks: bool | None = None,
    material_target: str | None = None,
) -> tuple[str, str | None]:
    """Export a USD stage for REST rendering, returning URL and optional S3 URI.

    This function is useful for batch rendering workflows where the same stage
    needs to be rendered multiple times. Instead of uploading the stage repeatedly,
    upload it once and reuse the URL.

    Data URI transfer is the default because it works with both local REST
    renderers and cloud endpoints without requiring S3 credentials. Set
    ``use_data_uri=False`` explicitly to use S3 upload mode.

    If ``bundle_mdl_assets`` is True, the function will bundle the USD with all
    referenced local MDL and texture files into a ZIP archive when such assets
    are found. Data URI mode encodes that ZIP directly; S3 mode uploads it.

    Args:
        stage: A Usd.Stage object from pxr package
        s3_bucket: S3 bucket for stage upload (ignored if use_data_uri=True).
                  Default: WU_S3_BUCKET env var (required for S3 mode).
        s3_region: AWS region (ignored if use_data_uri=True).
                  Default: WU_S3_REGION env var or "us-east-2"
        s3_profile: AWS profile for S3 upload (ignored if use_data_uri=True).
                   Default: WU_S3_PROFILE env var (required for S3 mode).
        use_data_uri: If True, use data URI encoding instead of S3 upload.
                     This embeds the USD file as base64 in the request.
                     If None, reads MA_RENDERING_USE_DATA_URI and defaults to
                     True.
        bundle_mdl_assets: If True, attempt to bundle local MDL and texture assets
                          with the USD file into a ZIP archive. If no local assets
                          are found, falls back to sending just the USD.
                          Default: True
        base_dir: Base directory for resolving relative texture paths. If None,
                 uses the stage's root layer directory.
        add_preview_fallbacks: Legacy compatibility flag. If explicitly True,
                     author render-export UsdPreviewSurface bridges for
                     MaterialX OpenPBR materials. If None, follows
                     ``material_target`` policy.
        material_target: Explicit material target for render export. ``auto``
                     preserves authored/native material outputs. Use
                     ``preview_surface`` to request preview fallback authoring.

    Returns:
        Tuple containing:
            - asset_url: The URL to access the stage (HTTPS URL or data URI)
            - s3_uri: The S3 URI for cleanup (None if use_data_uri=True)

    Example:
        >>> from pxr import Usd
        >>> stage = Usd.Stage.CreateInMemory()
        >>> # ... build scene ...
        >>> # Upload once
        >>> url, s3_uri = export_stage_to_s3(stage, s3_bucket="my-bucket")
        >>> # Use the URL for multiple render calls
        >>> result1 = render_single_camera_from_url(url, camera="/Camera1")
        >>> result2 = render_single_camera_from_url(url, camera="/Camera2")
        >>> # Clean up when done
        >>> if s3_uri:
        ...     delete_s3_path(s3_uri, profile_name="your-aws-profile")
    """
    use_data_uri = should_use_data_uri(use_data_uri)
    effective_add_preview_fallbacks = preview_fallbacks_enabled_for_material_target(
        material_target,
        legacy_add_preview_fallbacks=add_preview_fallbacks,
    )

    # Try bundling MDL assets if requested
    temp_dir = None
    zip_path = None
    was_bundled = False

    has_local_composition_arcs = _stage_has_local_composition_arcs(stage)
    if has_local_composition_arcs:
        raise RuntimeError(
            "Remote REST rendering requires a flattened stage when local "
            "sublayers, references, or payloads are present. Set "
            "flatten_before_render=True or call prepare_stage_for_render(..., "
            "flatten=True) before exporting to a remote renderer."
        )

    if bundle_mdl_assets:
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="remote_bundle_"))
            zip_path, was_bundled = _bundle_stage_with_local_assets(
                stage,
                temp_dir,
                base_dir=base_dir,
                has_local_composition_arcs=has_local_composition_arcs,
                add_preview_fallbacks=effective_add_preview_fallbacks,
            )
        except Exception as e:
            logger.warning(f"MDL bundling failed, falling back to USD-only: {e}")
            was_bundled = False

    if was_bundled and zip_path:
        try:
            if use_data_uri:
                asset_url = create_data_uri_from_file(
                    zip_path,
                    mime_type="application/zip",
                )
                logger.info("Created data URI for MDL/texture asset bundle")
                return asset_url, None

            # Upload the ZIP bundle
            unique_id = uuid.uuid4().hex
            s3_key = f"nvcf-renders/{unique_id}/bundle.zip"
            s3_uri = upload_file_to_s3(
                file_path=str(zip_path),
                s3_path=f"s3://{s3_bucket}/{s3_key}",
                profile_name=s3_profile,
            )
            asset_url = s3_uri_to_https_url(s3_uri, s3_region)
            logger.info("Uploaded MDL bundle to S3: %s", asset_url)
            return asset_url, s3_uri
        finally:
            # Clean up temp directory
            if temp_dir and temp_dir.exists():
                try:
                    import shutil

                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
    else:
        # Clean up temp directory if bundling was attempted but didn't produce a bundle
        if temp_dir and temp_dir.exists():
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except Exception:
                pass

    # Fall back to original behavior: export USD only
    with tempfile.NamedTemporaryFile(suffix=".usdc", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not stage.GetRootLayer().Export(tmp_path):
            raise RuntimeError("Failed to export USD stage")

        if effective_add_preview_fallbacks:
            preview_fallbacks = add_ovrtx_preview_fallbacks_to_stage_file(tmp_path)
            if preview_fallbacks:
                logger.info(
                    "Updated %d OpenPBR material fallback(s) for remote render export",
                    preview_fallbacks,
                )

        if effective_add_preview_fallbacks:
            display_color_bakes, textured_fallbacks = (
                _add_texture_file_fallbacks_for_remote_export(Path(tmp_path))
            )
            if display_color_bakes:
                logger.info(
                    "Baked %d textured mesh(es) to displayColor for remote render export",
                    display_color_bakes,
                )
            if textured_fallbacks:
                logger.info(
                    "Updated %d texture-file material fallback(s) for remote render export",
                    textured_fallbacks,
                )

            removed_mdl_outputs = _prefer_preview_surface_for_remote_export(
                Path(tmp_path),
            )
            if removed_mdl_outputs:
                logger.info(
                    "Removed %d MDL material outputs from remote render export",
                    removed_mdl_outputs,
                )

        asset_url, s3_uri = _export_stage_and_get_url(
            stage_path=tmp_path,
            use_data_uri=use_data_uri,
            s3_bucket=s3_bucket,
            s3_profile=s3_profile,
            s3_region=s3_region,
        )
        return asset_url, s3_uri
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _parse_zip_response(zip_content: bytes) -> dict | None:
    """
    Parse a ZIP response from the rendering server.

    The server sometimes returns a ZIP file containing a single JSON file ({uuid}.response)
    with the complete rendering result, including base64-encoded image data.
    This function extracts and parses that JSON file.

    Args:
        zip_content: Raw ZIP file content as bytes

    Returns:
        dict | None: Parsed result with structure:
            {
                "images": {frame_num: {camera_path: {sensor_name: base64_data}}},
                "status": "success",
                "error": null
            }
            Returns None if parsing fails.
    """
    # Open ZIP from memory
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
        # List all files in the ZIP
        file_list = zip_file.namelist()
        logger.info("ZIP contains files: %s", file_list)

        # Find the .response file (contains JSON with base64-encoded data)
        response_file = None
        for filename in file_list:
            if filename.endswith(".response"):
                response_file = filename
                break

        if response_file is None:
            logger.error("No .response file found in ZIP. Files: %s", file_list)
            raise ValueError("No .response file found in ZIP")

        # Extract and parse the JSON response
        logger.info("Extracting response file: %s", response_file)
        with zip_file.open(response_file) as f:
            response_text = f.read().decode("utf-8")
            result = json.loads(response_text)

        # The result is already in the correct format:
        # {"images": {frame: {camera_path: {sensor: base64_data}}}, "status": "success", "error": null}
        logger.info(
            "Successfully parsed ZIP response: status=%s, frames=%d",
            result.get("status", "unknown"),
            len(result.get("images", {})),
        )
        return result


def render_single_camera(
    stage: "Usd.Stage",
    camera: str,
    image_width: int = 1024,
    image_height: int = 1024,
    frames: str = "0",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 3600,
    sensors: list[str] | None = None,
    apply_background_mask: bool = False,
    use_data_uri: bool | None = None,
    s3_bucket: str = WU_S3_BUCKET,
    s3_region: str = WU_S3_REGION,
    s3_profile: str = WU_S3_PROFILE,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_jitter: float = 0.1,
    add_preview_fallbacks: bool | None = None,
    material_target: str | None = None,
) -> dict[str, Any]:
    """
    Render a single camera view from an in-memory USD Stage using a REST renderer.

    This function exports the USD stage and uses a REST rendering service to
    render images. It supports two transfer modes: data URI encoding (default)
    or explicit S3 upload.
    It returns PIL Image objects and optionally sensor data like depth and normals.

    Args:
        stage: A Usd.Stage object from pxr package
        camera: Camera path to use for rendering (e.g., "/Camera", "/World/Camera")
        image_width: Image width in pixels. Default: 1024
        image_height: Image height in pixels. Default: 1024
        frames: Frame(s) to render. Can be:
               - Single frame: "0", "42"
               - Frame range: "0:10", "5:15"
               Default: "0" (first frame)
        api_key: API key for the REST renderer. If None, uses NGC_API_KEY env var.
        base_url: REST renderer base URL. If None, uses RENDER_ENDPOINT env var
            (or NVCF_RENDER_FUNCTION_ID fallback).
        timeout: Request timeout in seconds. Default: 3600
        sensors: Additional sensors to render (e.g., ["linear_depth", "instance_id_segmentation"])
        apply_background_mask: If True, apply background masking during rendering. Default: False
        use_data_uri: If True, use data URI encoding instead of S3 upload.
                     This embeds the USD file as base64 in the request. If None,
                     reads MA_RENDERING_USE_DATA_URI and defaults to True.
        s3_bucket: S3 bucket for stage upload (ignored if use_data_uri=True).
                  Default: WU_S3_BUCKET env var (required for S3 mode).
        s3_region: AWS region (ignored if use_data_uri=True).
                  Default: WU_S3_REGION env var or "us-east-2"
        s3_profile: AWS profile for S3 upload (ignored if use_data_uri=True).
                   Default: WU_S3_PROFILE env var (required for S3 mode).
        max_retries: Maximum number of retry attempts. Default: 3
        retry_delay: Initial delay between retries in seconds. Default: 1.0
        retry_backoff_factor: Factor to multiply delay by after each retry. Default: 2.0
        retry_jitter: Random jitter factor (0-1) to add to delays. Default: 0.1
        add_preview_fallbacks: Accepted for API compatibility with export-based
            remote rendering, but this helper does not apply PreviewSurface
            fallbacks to its temporary export.
        material_target: Explicit material target sent in the REST request so
            the renderer can choose material handling. This helper does not
            rewrite the temporary stage export before sending that request.

    Returns:
        Dict containing:
            - camera: Camera path used for rendering
            - images: List of PIL Image objects (ordered by frame)
            - sensors: Dict of sensor_name -> frame_num -> numpy arrays (if sensors requested)
            - render_time: Total rendering time in seconds
            - frame_count: Number of frames rendered
            - status: Rendering status (success, load_error, etc.)
            - error: Error message if rendering failed (optional)

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If rendering fails

    Example:
        >>> from pxr import Usd, UsdGeom
        >>> stage = Usd.Stage.CreateInMemory()
        >>> # ... build your USD scene ...
        >>> # Using data URI (default, no S3 needed)
        >>> result = render_single_camera(
        ...     stage=stage,
        ...     camera="/Camera",
        ...     image_width=1920,
        ...     image_height=1080
        ... )
        >>> # Using S3 explicitly
        >>> result = render_single_camera(
        ...     stage=stage,
        ...     camera="/Camera",
        ...     use_data_uri=False,
        ...     s3_bucket="my-bucket",
        ...     s3_region="us-west-2"
        ... )
        >>> for i, img in enumerate(result['images']):
        ...     img.save(f"frame_{i}.png")
    """
    try:
        from pxr import Usd
    except ImportError as e:
        raise RuntimeError(
            "USD Python bindings are required for this function. Install a supported "
            "provider (Linux ARM64 + Python 3.12: `uv pip install usd-exchange`; "
            "Linux ARM64 + Python 3.13 is currently unsupported; other supported "
            "platforms: `uv pip install usd-core`)."
        ) from e

    if not isinstance(stage, Usd.Stage):
        raise ValueError(f"stage must be a Usd.Stage object. Got: {type(stage)}")

    use_data_uri = should_use_data_uri(use_data_uri)

    # Export stage to temporary file (using binary format for smaller disk footprint)
    with tempfile.NamedTemporaryFile(suffix=".usdc", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not stage.GetRootLayer().Export(tmp_path):
            raise RuntimeError("Failed to export USD stage")

        asset_url, s3_uri = _export_stage_and_get_url(
            stage_path=tmp_path,
            use_data_uri=use_data_uri,
            s3_bucket=s3_bucket,
            s3_profile=s3_profile,
            s3_region=s3_region,
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Render using the URL and clean up S3 file afterwards
    try:
        result = render_single_camera_from_url(
            usd_url=asset_url,
            camera=camera,
            image_width=image_width,
            image_height=image_height,
            frames=frames,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            sensors=sensors,
            apply_background_mask=apply_background_mask,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff_factor=retry_backoff_factor,
            retry_jitter=retry_jitter,
            material_target=material_target,
        )
        return result
    finally:
        # Clean up S3 file if it was used
        if s3_uri:
            try:
                delete_s3_path(s3_uri, profile_name=s3_profile)
                logger.info("Cleaned up S3 file: %s", s3_uri)
            except Exception as e:
                logger.warning("Failed to clean up S3 file %s: %s", s3_uri, e)


def render_single_camera_from_url(
    usd_url: str,
    camera: str,
    image_width: int = 1024,
    image_height: int = 1024,
    frames: str = "0",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 3600,
    sensors: list[str] | None = None,
    apply_background_mask: bool = False,
    force_render: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_jitter: float = 0.1,
    material_target: str | None = None,
) -> dict[str, Any]:
    """
    Render a single camera view from a USD file URL using a REST renderer.

    This function uses the REST rendering service to render images from USD scenes
    accessible via URL (HTTP/HTTPS or S3). It supports single frames or frame
    ranges and can render additional sensor data.

    Args:
        usd_url: URL to the USD file (HTTP/HTTPS or S3 URL)
        camera: Camera path to use for rendering (e.g., "/Camera", "/World/Camera")
        image_width: Image width in pixels. Default: 1024
        image_height: Image height in pixels. Default: 1024
        frames: Frame(s) to render. Can be:
               - Single frame: "0", "42"
               - Frame range: "0:10", "5:15"
               Default: "0" (first frame)
        api_key: REST renderer API key. If None, uses NGC_API_KEY env var
        base_url: REST renderer base URL. If None, uses RENDER_ENDPOINT env var (or NVCF_RENDER_FUNCTION_ID fallback)
        timeout: Request timeout in seconds. Default: 3600
        sensors: Additional sensors to render (e.g., ["linear_depth", "instance_id_segmentation"])
        apply_background_mask: If True, apply background masking during rendering. Default: False
        force_render: Force re-rendering even if cached. Default: True
        max_retries: Maximum number of retry attempts. Default: 3
        retry_delay: Initial delay between retries in seconds. Default: 1.0
        retry_backoff_factor: Factor to multiply delay by after each retry. Default: 2.0
        retry_jitter: Random jitter factor (0-1) to add to delays. Default: 0.1
        material_target: Explicit material target forwarded to the REST renderer.
            Use ``openpbr_materialx`` to render native OpenPBR/MaterialX without
            requesting render-export PreviewSurface fallbacks.

    Returns:
        Dict containing:
            - camera: Camera path used for rendering
            - images: List of PIL Image objects (ordered by frame)
            - sensors: Dict of sensor_name -> frame_num -> numpy arrays (if sensors requested)
            - render_time: Total rendering time in seconds
            - frame_count: Number of frames rendered
            - status: Rendering status (success, load_error, etc.)
            - error: Error message if rendering failed (optional)

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If rendering fails

    Example:
        >>> result = render_single_camera_from_url(
        ...     usd_url="https://example.com/scene.usd",
        ...     camera="/Camera",
        ...     image_width=1920,
        ...     image_height=1080,
        ...     frames="0:10",
        ...     sensors=["linear_depth", "instance_id_segmentation"]
        ... )
        >>> print(f"Rendered {result['frame_count']} frames")
        >>> # Access sensor data
        >>> depth_data = result['sensors']['linear_depth'][0]  # Frame 0 depth
        >>> seg_data = result['sensors']['instance_id_segmentation'][0]  # Frame 0 segmentation
    """
    # Get API key and base URL using common utilities
    api_key = get_nvcf_api_key(api_key)
    base_url = get_base_url(base_url, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")

    # Construct full URL with render endpoint
    full_url = f"{base_url.rstrip('/')}/render"

    # Parse frames parameter
    if ":" in frames:
        # Frame range
        start_str, end_str = frames.split(":")
        frame_start = int(start_str)
        frame_end = int(end_str)
    else:
        # Single frame
        frame_num = int(frames)
        frame_start = frame_num
        frame_end = frame_num

    # Build request parameters
    params = {
        "url": usd_url,
        "force_render": force_render,
        "render_settings": {
            "camera_paths": [camera],
            "frame_range": {"start": frame_start, "end": frame_end},
            "camera_parameters": {
                "width": image_width,
                "height": image_height,
            },
            "sensors": sensors,
            "apply_background_mask": apply_background_mask,
        },
    }
    if material_target is not None:
        params["render_settings"]["material_target"] = material_target

    # Create headers using common utility
    headers = create_nvcf_headers(api_key, timeout)

    # Truncate URL for logging (data URIs can be very long)
    logger.info(
        "Rendering camera %s with frames %s from %s", camera, frames, usd_url[:100]
    )
    start_time = time.time()

    # Retry logic for Remote render request
    last_error = None
    current_delay = retry_delay

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                # Add jitter to prevent thundering herd
                jittered_delay = current_delay * (
                    1 + random.uniform(-retry_jitter, retry_jitter)
                )
                logger.info(
                    "Retrying Remote render request (attempt %d/%d) after %.2fs delay",
                    attempt + 1,
                    max_retries + 1,
                    jittered_delay,
                )
                time.sleep(jittered_delay)
                current_delay *= retry_backoff_factor

            response = requests.post(
                full_url,
                headers=headers,
                json=params,
                timeout=timeout + 10,
                allow_redirects=True,
            )
            response.raise_for_status()

            # Check content type to handle both JSON and ZIP responses
            content_type = response.headers.get("Content-Type", "")

            if "application/json" in content_type:
                result = response.json()
            elif "application/zip" in content_type:
                logger.info(
                    "Received ZIP response for %s. Processing directly...",
                    usd_url[:100],
                )
                # Read the ZIP content
                zip_content = response.content

                # Parse ZIP and convert to expected result format
                result = _parse_zip_response(zip_content)
                if result is None:
                    logger.error("Failed to parse ZIP response for %s", usd_url[:100])
                    raise ValueError("Failed to parse ZIP response")
            else:
                logger.error(
                    "Unexpected content type '%s' for %s",
                    content_type,
                    usd_url[:100],
                )
                raise ValueError("Unexpected content type")

            # Success - break out of retry loop
            break

        except (ConnectionError, Timeout) as e:
            # Network errors - retry
            last_error = e
            logger.warning(
                "Remote render request attempt %d failed with network error: %s",
                attempt + 1,
                str(e),
            )
            if attempt == max_retries:
                error_msg = f"Remote render request failed after {max_retries + 1} attempts: {str(last_error)}"
                logger.error(error_msg)
                return {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": time.time() - start_time,
                    "frame_count": 0,
                    "status": RenderingStatus.exception,
                    "error": error_msg,
                }

        except HTTPError as e:
            response = e.response
            if response is None:
                error_msg = f"Remote render request failed with HTTP error: {e}"
                logger.error(error_msg)
                return {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": time.time() - start_time,
                    "frame_count": 0,
                    "status": RenderingStatus.exception,
                    "error": error_msg,
                }

            # HTTP errors - check if we should retry
            # Retryable status codes:
            # - 408: Request Timeout (client timeout, can retry)
            # - 429: Too Many Requests (rate limiting, should retry with backoff)
            # - 502, 503, 504: Gateway/service errors (server issues)
            retryable_codes = [408, 429, 502, 503, 504]

            if response.status_code in retryable_codes:
                last_error = e
                logger.warning(
                    "Remote render request attempt %d failed with HTTP %d: %s",
                    attempt + 1,
                    response.status_code,
                    str(e),
                )
                if attempt == max_retries:
                    error_msg = f"Remote render request failed after {max_retries + 1} attempts: HTTP {response.status_code}"
                    logger.error(error_msg)
                    return {
                        "camera": camera,
                        "images": [],
                        "sensors": {},
                        "render_time": time.time() - start_time,
                        "frame_count": 0,
                        "status": RenderingStatus.exception,
                        "error": error_msg,
                    }
            else:
                # Non-retryable HTTP error (400, 401, 403, 404, etc.)
                error_payload = _http_error_payload(response)
                error_detail = _http_error_detail(response)
                renderer_status = error_payload.get("status")
                status = (
                    RenderingStatus.blank_render
                    if renderer_status == RenderingStatus.blank_render
                    else RenderingStatus.exception
                )
                error_msg = (
                    "Remote render request failed with non-retryable HTTP "
                    f"{response.status_code}: {error_detail}"
                )
                logger.error(
                    "Non-retryable error: HTTP %d. Will not retry. Error: %s",
                    response.status_code,
                    error_detail,
                )
                return {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": time.time() - start_time,
                    "frame_count": 0,
                    "status": status,
                    "error": error_msg,
                    "warnings": error_payload.get("warnings", []),
                    "blank_render_frames": error_payload.get("blank_render_frames", []),
                }

        except RequestException as e:
            # Other request exceptions - don't retry
            error_msg = (
                f"Remote render request failed with non-retryable error: {str(e)}"
            )
            logger.error(
                "Non-retryable request exception. Will not retry. Error: %s", str(e)
            )
            return {
                "camera": camera,
                "images": [],
                "sensors": {},
                "render_time": time.time() - start_time,
                "frame_count": 0,
                "status": RenderingStatus.exception,
                "error": error_msg,
            }

    render_time = time.time() - start_time
    logger.info("Remote render request completed in %.2fs", render_time)

    # Convert V2 response to V1 format if needed
    if _is_v2_response(result):
        result = _convert_v2_to_v1(result)

    # Check status
    status = result.get("status", RenderingStatus.exception)
    if status != RenderingStatus.success:
        result_error = result.get("error")
        error_msg = f"Rendering failed with status: {status}"
        if result_error:
            error_msg = f"{error_msg}: {result_error}"
        logger.error(error_msg)
        return {
            "camera": camera,
            "images": [],
            "sensors": {},
            "render_time": render_time,
            "frame_count": 0,
            "status": status,
            "error": error_msg,
            "warnings": result.get("warnings", []),
            "blank_render_frames": result.get("blank_render_frames", []),
        }

    # Process results - convert dict to list for compatibility
    images = []
    sensor_data = {sensor: {} for sensor in (sensors or [])}
    warnings = result.get("warnings", [])
    blank_render_frames = result.get("blank_render_frames", [])

    # Sort frames by frame number to maintain order
    frame_items = sorted(result.get("images", {}).items(), key=lambda x: int(x[0]))
    for frame_num, frame_data in frame_items:
        frame_num_int = int(frame_num)

        # Get camera data (should only be one camera)
        for _camera_path, camera_data in frame_data.items():
            # Process main image
            if "images" in camera_data:
                try:
                    img = base64_to_image(camera_data["images"])
                    images.append(img)
                except Exception as e:
                    logger.warning(
                        "Failed to decode image for frame %s: %s", frame_num, e
                    )

            # Process sensor data
            for sensor_name in sensors or []:
                if sensor_name in camera_data:
                    try:
                        # Determine dtype based on sensor type
                        if sensor_name == "instance_id_segmentation":
                            # Segmentation uses uint32 for instance IDs (not uint8!)
                            # Using uint8 causes 4x data size and stride issues
                            dtype = np.uint32
                        else:
                            dtype = np.float32

                        data = base64_to_numpy(camera_data[sensor_name], dtype=dtype)
                        sensor_data[sensor_name][frame_num_int] = data
                    except Exception as e:
                        logger.warning(
                            "Failed to decode %s for frame %s: %s",
                            sensor_name,
                            frame_num,
                            e,
                        )

    frame_count = len(images)
    logger.info("Successfully rendered %s frames for camera %s", frame_count, camera)

    return {
        "camera": camera,
        "images": images,
        "sensors": sensor_data,
        "render_time": render_time,
        "frame_count": frame_count,
        "status": status,
        "warnings": warnings,
        "blank_render_frames": blank_render_frames,
    }


def render_all_cameras(
    stage: "Usd.Stage",
    image_width: int = 1024,
    image_height: int = 1024,
    cameras: list[str] | None = None,
    frames: str = "0",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 3600,
    sensors: list[str] | None = None,
    apply_background_mask: bool = False,
    use_data_uri: bool | None = None,
    s3_bucket: str = WU_S3_BUCKET,
    s3_region: str = WU_S3_REGION,
    s3_profile: str = WU_S3_PROFILE,
    max_workers: int = 8,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_jitter: float = 0.1,
    bundle_mdl_assets: bool = True,
    base_dir: str | Path | None = None,
    add_preview_fallbacks: bool | None = None,
    material_target: str | None = None,
    use_global_render_slots: bool = False,
    render_slot_timeout_sec: float | None = None,
) -> dict[str, Any]:
    """
    Render multiple cameras from an in-memory USD Stage using a REST renderer.

    This function renders multiple camera views from a USD Stage object.
    It supports two transfer modes: data URI encoding (default) or explicit S3
    upload.
    If no cameras are specified, it uses a default camera named "/Camera".
    Multiple cameras can be rendered in parallel using threading.

    Args:
        stage: A Usd.Stage object from pxr package
        image_width: Image width in pixels. Default: 1024
        image_height: Image height in pixels. Default: 1024
        cameras: List of camera paths to render. If None, uses ["/Camera"]
        frames: Frame(s) to render. Default: "0"
        api_key: API key for the REST renderer. If None, uses NGC_API_KEY env var.
        base_url: REST renderer base URL. If None, uses RENDER_ENDPOINT env var
            (or NVCF_RENDER_FUNCTION_ID fallback).
        timeout: Request timeout in seconds. Default: 3600
        sensors: Additional sensors to render (e.g., ["linear_depth", "instance_id_segmentation"])
        apply_background_mask: If True, apply background masking during rendering. Default: False
        use_data_uri: If True, use data URI encoding instead of S3 upload.
                     This embeds the USD file as base64 in the request. If None,
                     reads MA_RENDERING_USE_DATA_URI and defaults to True.
        s3_bucket: S3 bucket for stage upload (ignored if use_data_uri=True).
                  Default: WU_S3_BUCKET env var (required for S3 mode).
        s3_region: AWS region (ignored if use_data_uri=True).
                  Default: WU_S3_REGION env var or "us-east-2"
        s3_profile: AWS profile for S3 upload (ignored if use_data_uri=True).
                   Default: WU_S3_PROFILE env var (required for S3 mode).
        max_workers: Maximum number of parallel render threads, from 1 through
                     32. Default: 8
        max_retries: Maximum number of retry attempts. Default: 3
        retry_delay: Initial delay between retries in seconds. Default: 1.0
        retry_backoff_factor: Factor to multiply delay by after each retry. Default: 2.0
        retry_jitter: Random jitter factor (0-1) to add to delays. Default: 0.1
        bundle_mdl_assets: If True, attempt to bundle local MDL and texture assets
                          with the USD file into a ZIP archive. If no local assets
                          are found, falls back to uploading just the USD.
                          Default: True
        base_dir: Base directory for resolving relative MDL and texture asset paths.
                  If None, uses the stage root layer directory.
        add_preview_fallbacks: Legacy compatibility flag. If explicitly True,
                  author render-export UsdPreviewSurface bridges for MaterialX
                  OpenPBR materials. If None, follows ``material_target`` policy.
        material_target: Explicit material target for render export. ``auto``
                  preserves authored/native material outputs. Use
                  ``preview_surface`` to request preview fallback authoring.
        use_global_render_slots: Acquire one process-wide remote-render slot
                  around each camera request. Use this when ``max_workers`` is
                  greater than one so the shared request cap remains exact.
        render_slot_timeout_sec: Optional timeout for each process-wide slot
                  acquisition when ``use_global_render_slots`` is enabled.

    Returns:
        Dict containing:
            - total_cameras: Number of cameras to render
            - successful_cameras: Number of successfully rendered cameras
            - failed_cameras: Number of failed camera renders
            - total_render_time: Total time for all renders in seconds
            - results: List of individual camera render results

    Example:
        >>> from pxr import Usd
        >>> stage = Usd.Stage.CreateInMemory()
        >>> # ... build scene ...
        >>> # Using data URI (default, no S3 needed)
        >>> result = render_all_cameras(
        ...     stage=stage,
        ...     image_width=1920,
        ...     image_height=1080,
        ...     cameras=["/Camera1", "/Camera2"]
        ... )
        >>> # Using S3 explicitly
        >>> result = render_all_cameras(
        ...     stage=stage,
        ...     cameras=["/Camera1", "/Camera2"],
        ...     use_data_uri=False,
        ...     s3_bucket="my-bucket",
        ...     s3_region="us-west-2"
        ... )
        >>> print(f"Rendered {result['successful_cameras']} cameras")
    """
    max_workers = validate_remote_render_max_workers(max_workers)

    # Resolve the slot provider before starting worker threads. The call-time
    # import avoids the render_remote_async -> render_remote module cycle.
    global_render_slot = None
    if use_global_render_slots:
        from world_understanding.functions.graphics.render_remote_async import (
            global_remote_render_slot,
        )

        global_render_slot = global_remote_render_slot

    # Default camera if none specified
    if cameras is None or len(cameras) == 0:
        cameras = ["/Camera"]

    results = []
    successful_cameras = 0
    failed_cameras = 0
    total_start_time = time.time()

    # Export and upload stage once for all cameras
    # If bundle_mdl_assets is True, will attempt to bundle local MDL files
    try:
        asset_url, s3_uri = export_stage_to_s3(
            stage=stage,
            s3_bucket=s3_bucket,
            s3_region=s3_region,
            s3_profile=s3_profile,
            use_data_uri=use_data_uri,
            bundle_mdl_assets=bundle_mdl_assets,
            base_dir=base_dir,
            add_preview_fallbacks=add_preview_fallbacks,
            material_target=material_target,
        )
    except Exception as exc:
        error_msg = f"Failed to export stage for remote rendering: {exc}"
        logger.exception("Failed to export stage for remote rendering")
        return {
            "total_cameras": len(cameras),
            "successful_cameras": 0,
            "failed_cameras": len(cameras),
            "total_render_time": time.time() - total_start_time,
            "results": [
                {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": 0.0,
                    "frame_count": 0,
                    "status": RenderingStatus.exception,
                    "error": error_msg,
                    "error_type": type(exc).__name__,
                }
                for camera in cameras
            ],
        }

    # Render cameras and clean up S3 file afterwards
    try:

        def render_camera(camera: str) -> dict[str, Any]:
            def issue_request() -> dict[str, Any]:
                return render_single_camera_from_url(
                    usd_url=asset_url,
                    camera=camera,
                    image_width=image_width,
                    image_height=image_height,
                    frames=frames,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    sensors=sensors,
                    apply_background_mask=apply_background_mask,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    retry_backoff_factor=retry_backoff_factor,
                    retry_jitter=retry_jitter,
                    material_target=material_target,
                )

            if global_render_slot is None:
                return issue_request()

            with global_render_slot(
                timeout_seconds=render_slot_timeout_sec,
            ) as queue_wait:
                if queue_wait > 0.05:
                    logger.info(
                        "Remote camera %s waited %.2fs for global render slot",
                        camera,
                        queue_wait,
                    )
                return issue_request()

        # Render cameras in parallel if max_workers > 1
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(render_camera, camera): camera for camera in cameras
                }

                for future in as_completed(futures):
                    camera = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result.get("status") == RenderingStatus.success:
                            successful_cameras += 1
                        else:
                            failed_cameras += 1
                    except Exception as e:
                        failed_cameras += 1
                        error_result = {
                            "camera": camera,
                            "images": [],
                            "sensors": {},
                            "render_time": 0.0,
                            "frame_count": 0,
                            "status": RenderingStatus.exception,
                            "error": str(e),
                        }
                        results.append(error_result)
                        logger.exception("Failed to render camera %s", camera)
        else:
            # Sequential rendering
            for camera in cameras:
                try:
                    result = render_camera(camera)
                    results.append(result)
                    if result.get("status") == RenderingStatus.success:
                        successful_cameras += 1
                    else:
                        failed_cameras += 1
                except Exception as e:
                    failed_cameras += 1
                    error_result = {
                        "camera": camera,
                        "images": [],
                        "sensors": {},
                        "render_time": 0.0,
                        "frame_count": 0,
                        "status": RenderingStatus.exception,
                        "error": str(e),
                    }
                    results.append(error_result)
                    logger.exception("Failed to render camera %s", camera)

        total_render_time = time.time() - total_start_time

        return {
            "total_cameras": len(cameras),
            "successful_cameras": successful_cameras,
            "failed_cameras": failed_cameras,
            "total_render_time": total_render_time,
            "results": results,
        }
    finally:
        # Clean up S3 file if it was used
        if s3_uri:
            try:
                delete_s3_path(s3_uri, profile_name=s3_profile)
                logger.info("Cleaned up S3 file: %s", s3_uri)
            except Exception as e:
                logger.warning("Failed to clean up S3 file %s: %s", s3_uri, e)


def render_all_cameras_from_url(
    usd_url: str,
    image_width: int = 1024,
    image_height: int = 1024,
    cameras: list[str] | None = None,
    frames: str = "0",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 3600,
    sensors: list[str] | None = None,
    apply_background_mask: bool = False,
    max_workers: int = 1,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_jitter: float = 0.1,
    material_target: str | None = None,
) -> dict[str, Any]:
    """
    Render multiple cameras from a USD file URL using a REST renderer.

    This function renders multiple camera views from a USD file accessible
    via URL. Multiple cameras can be rendered in parallel using threading.

    Args:
        usd_url: URL to the USD file (HTTP/HTTPS or S3 URL)
        image_width: Image width in pixels. Default: 1024
        image_height: Image height in pixels. Default: 1024
        cameras: List of camera paths to render. If None, uses ["/Camera"]
        frames: Frame(s) to render. Default: "0"
        api_key: REST renderer API key. If None, uses NGC_API_KEY env var
        base_url: REST renderer base URL. If None, uses RENDER_ENDPOINT env var (or NVCF_RENDER_FUNCTION_ID fallback)
        timeout: Request timeout in seconds. Default: 3600
        sensors: Additional sensors to render (e.g., ["linear_depth", "instance_id_segmentation"])
        apply_background_mask: If True, apply background masking during rendering. Default: False
        max_workers: Maximum number of parallel render threads. Default: 1
        max_retries: Maximum number of retry attempts per camera. Default: 3
        retry_delay: Initial delay between retries in seconds. Default: 1.0
        retry_backoff_factor: Factor to multiply delay by after each retry. Default: 2.0
        retry_jitter: Random jitter factor (0-1) to add to delays. Default: 0.1
        material_target: Explicit material target forwarded to the REST renderer.

    Returns:
        Dict containing:
            - total_cameras: Number of cameras to render
            - successful_cameras: Number of successfully rendered cameras
            - failed_cameras: Number of failed camera renders
            - total_render_time: Total time for all renders in seconds
            - results: List of individual camera render results

    Example:
        >>> result = render_all_cameras_from_url(
        ...     usd_url="https://example.com/scene.usd",
        ...     image_width=1920,
        ...     image_height=1080,
        ...     cameras=["/Camera1", "/Camera2"],
        ...     sensors=["depth"],
        ...     max_workers=2
        ... )
        >>> for cam_result in result['results']:
        ...     if cam_result['status'] == 'success':
        ...         print(f"Camera {cam_result['camera']}: {cam_result['frame_count']} frames")
    """
    # Default camera if none specified
    if cameras is None or len(cameras) == 0:
        cameras = ["/Camera"]

    results = []
    successful_cameras = 0
    failed_cameras = 0
    total_start_time = time.time()

    # Render cameras in parallel if max_workers > 1
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    render_single_camera_from_url,
                    usd_url=usd_url,
                    camera=camera,
                    image_width=image_width,
                    image_height=image_height,
                    frames=frames,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    sensors=sensors,
                    apply_background_mask=apply_background_mask,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    retry_backoff_factor=retry_backoff_factor,
                    retry_jitter=retry_jitter,
                    material_target=material_target,
                ): camera
                for camera in cameras
            }

            for future in as_completed(futures):
                camera = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result.get("status") == RenderingStatus.success:
                        successful_cameras += 1
                    else:
                        failed_cameras += 1
                except Exception as e:
                    failed_cameras += 1
                    error_result = {
                        "camera": camera,
                        "images": [],
                        "sensors": {},
                        "render_time": 0.0,
                        "frame_count": 0,
                        "status": RenderingStatus.exception,
                        "error": str(e),
                    }
                    results.append(error_result)
                    logger.exception("Failed to render camera %s", camera)
    else:
        # Sequential rendering
        for camera in cameras:
            try:
                result = render_single_camera_from_url(
                    usd_url=usd_url,
                    camera=camera,
                    image_width=image_width,
                    image_height=image_height,
                    frames=frames,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    sensors=sensors,
                    apply_background_mask=apply_background_mask,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    retry_backoff_factor=retry_backoff_factor,
                    retry_jitter=retry_jitter,
                    material_target=material_target,
                )
                results.append(result)
                if result.get("status") == RenderingStatus.success:
                    successful_cameras += 1
                else:
                    failed_cameras += 1
            except Exception as e:
                failed_cameras += 1
                error_result = {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": 0.0,
                    "frame_count": 0,
                    "status": RenderingStatus.exception,
                    "error": str(e),
                }
                results.append(error_result)
                logger.exception("Failed to render camera %s", camera)

    total_render_time = time.time() - total_start_time

    return {
        "total_cameras": len(cameras),
        "successful_cameras": successful_cameras,
        "failed_cameras": failed_cameras,
        "total_render_time": total_render_time,
        "results": results,
    }


def batch_render_assets(
    asset_urls: list[str],
    cameras: list[str] | None = None,
    image_width: int = 1024,
    image_height: int = 1024,
    frames: str = "0",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 3600,
    sensors: list[str] | None = None,
    apply_background_mask: bool = False,
    max_workers: int = 32,
    material_target: str | None = None,
) -> dict[str, Any]:
    """
    Batch render multiple USD assets with specified cameras using a REST renderer.

    This function efficiently renders multiple USD files, each with multiple
    cameras, using parallel processing. It's optimized for high-throughput
    rendering of many assets.

    Args:
        asset_urls: List of USD file URLs to render
        cameras: List of camera paths to render for each asset
        image_width: Image width in pixels. Default: 1024
        image_height: Image height in pixels. Default: 1024
        frames: Frame(s) to render. Default: "0"
        api_key: REST renderer API key. If None, uses NGC_API_KEY env var
        base_url: REST renderer base URL. If None, uses RENDER_ENDPOINT env var (or NVCF_RENDER_FUNCTION_ID fallback)
        timeout: Request timeout in seconds. Default: 3600
        sensors: Additional sensors to render (e.g., ["linear_depth", "instance_id_segmentation"])
        apply_background_mask: If True, apply background masking during rendering. Default: False
        max_workers: Maximum number of parallel render threads. Default: 32
        material_target: Explicit material target forwarded to the REST renderer.

    Returns:
        Dict containing:
            - total_assets: Number of assets processed
            - successful_assets: Number of successfully rendered assets
            - failed_assets: Number of failed asset renders
            - total_render_time: Total time for all renders in seconds
            - asset_results: Dict mapping asset URL to render results

    Example:
        >>> asset_urls = [
        ...     "https://example.com/asset1.usd",
        ...     "https://example.com/asset2.usd",
        ...     "https://example.com/asset3.usd"
        ... ]
        >>> result = batch_render_assets(
        ...     asset_urls=asset_urls,
        ...     cameras=["/Camera"],
        ...     image_width=1920,
        ...     image_height=1080,
        ...     max_workers=8
        ... )
        >>> print(f"Rendered {result['successful_assets']}/{result['total_assets']} assets")
    """
    if not asset_urls:
        raise ValueError("No asset URLs provided")

    # Default camera if none specified
    if cameras is None or len(cameras) == 0:
        cameras = ["/Camera"]

    asset_results = {}
    successful_assets = 0
    failed_assets = 0
    total_start_time = time.time()

    logger.info(
        "Starting batch render of %s assets with %s workers",
        len(asset_urls),
        max_workers,
    )

    # Create tasks for all asset-camera combinations
    render_tasks = []
    for asset_url in asset_urls:
        for camera in cameras:
            render_tasks.append((asset_url, camera))

    # Process all tasks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                render_single_camera_from_url,
                usd_url=asset_url,
                camera=camera,
                image_width=image_width,
                image_height=image_height,
                frames=frames,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                sensors=sensors,
                apply_background_mask=apply_background_mask,
                material_target=material_target,
            ): (asset_url, camera)
            for asset_url, camera in render_tasks
        }

        # Collect results
        for future in as_completed(futures):
            asset_url, camera = futures[future]

            # Initialize asset results if needed
            if asset_url not in asset_results:
                asset_results[asset_url] = {
                    "cameras": {},
                    "successful_cameras": 0,
                    "failed_cameras": 0,
                }

            try:
                result = future.result()
                asset_results[asset_url]["cameras"][camera] = result

                if result.get("status") == RenderingStatus.success:
                    asset_results[asset_url]["successful_cameras"] += 1
                else:
                    asset_results[asset_url]["failed_cameras"] += 1

            except Exception as e:
                asset_results[asset_url]["cameras"][camera] = {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": 0.0,
                    "frame_count": 0,
                    "status": RenderingStatus.exception,
                    "error": str(e),
                }
                asset_results[asset_url]["failed_cameras"] += 1
                # Truncate URL for logging (data URIs can be very long)
                logger.exception(
                    "Failed to render %s camera %s", asset_url[:100], camera
                )

    # Count successful assets (all cameras rendered successfully)
    for _asset_url, results in asset_results.items():
        if results["failed_cameras"] == 0:
            successful_assets += 1
        else:
            failed_assets += 1

    total_render_time = time.time() - total_start_time

    logger.info(
        "Batch render completed in %.2fs: %s/%s assets successful",
        total_render_time,
        successful_assets,
        len(asset_urls),
    )

    return {
        "total_assets": len(asset_urls),
        "successful_assets": successful_assets,
        "failed_assets": failed_assets,
        "total_render_time": total_render_time,
        "asset_results": asset_results,
    }


def save_render_results(
    result: dict[str, Any],
    output_dir: Path | str,
    file_name: str = "render",
    image_width: int = 1024,
    image_height: int = 1024,
    save_npy: bool = False,
) -> dict[str, int]:
    """Save render results to disk with proper processing for different sensor types.

    This function saves rendered images and sensor data to disk, handling different
    sensor types appropriately:
    - images: saved as PNG
    - instance_id_segmentation: saved as raw NPY when requested and PNG visualization
    - depth/linear_depth: saved as NPY (float32) and processed PNG
    - other sensors: saved as NPY (float32)

    Args:
        result: Render result dictionary from remote render functions
        output_dir: Directory to save the output files
        file_name: Base name for output files
        image_width: Width of the rendered images
        image_height: Height of the rendered images
        save_npy: If True, save sensor data as NPY files. Default: False
    Returns:
        dict: Dictionary with counts:
            - total_count: Total number of files saved
            - success_count: Number of successfully saved files
            - error_count: Number of failed saves

    Example:
        >>> result = render_single_camera(
        ...     stage=stage,
        ...     camera="/Camera",
        ...     frames="0:2",
        ...     sensors=["depth", "instance_id_segmentation"]
        ... )
        >>> stats = save_render_results(
        ...     result=result,
        ...     output_dir="output",
        ...     file_name="scene",
        ...     image_width=1024,
        ...     image_height=1024
        ... )
        >>> print(f"Saved {stats['success_count']}/{stats['total_count']} files")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_count = 0
    success_count = 0
    error_count = 0

    # Save main images
    if "images" in result and result["images"]:
        for frame_idx, img in enumerate(result["images"]):
            total_count += 1
            try:
                output_path = output_dir / f"{file_name}_f{frame_idx:04d}_images.png"
                img.save(output_path)
                success_count += 1
            except Exception as e:
                logger.warning("Failed to save image for frame %d: %s", frame_idx, e)
                error_count += 1

    # Save sensor data
    if "sensors" in result and result["sensors"]:
        for sensor_name, frame_data in result["sensors"].items():
            for frame_num, data in frame_data.items():
                total_count += 1
                try:
                    # Reshape data to proper dimensions
                    data = data.reshape(image_height, image_width, -1)

                    # Save as NPY
                    npy_path = (
                        output_dir / f"{file_name}_f{frame_num:04d}_{sensor_name}.npy"
                    )
                    if save_npy:
                        np.save(npy_path, data)

                    # Apply depth processing for depth sensors
                    # Note: linear_depth (distance_to_image_plane) is the standard Z-depth for computer vision
                    # depth (distance_to_camera) is radial distance from camera center
                    png_path = npy_path.with_suffix(".png")
                    if sensor_name in ("depth", "linear_depth"):
                        depth_map = process_depth_map(data)
                        depth_map = Image.fromarray(
                            (depth_map[..., 0] * 255.0).astype(np.uint8)
                        )
                        depth_map.save(png_path)
                    elif sensor_name == "instance_id_segmentation":
                        ids = data.squeeze().astype(np.uint32, copy=False)
                        vis_data = np.stack(
                            (
                                (ids * 37) % 256,
                                (ids * 17 + 13) % 256,
                                (ids * 97 + 29) % 256,
                            ),
                            axis=-1,
                        ).astype(np.uint8)
                        vis_data[ids == 0] = 0
                        Image.fromarray(vis_data).save(png_path)

                    success_count += 1
                except Exception as e:
                    logger.warning(
                        "Failed to save %s for frame %d: %s",
                        sensor_name,
                        frame_num,
                        e,
                    )
                    error_count += 1

    logger.info(
        "Saved %d/%d files to %s (%d errors)",
        success_count,
        total_count,
        output_dir,
        error_count,
    )

    return {
        "total_count": total_count,
        "success_count": success_count,
        "error_count": error_count,
    }
