# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD prim traversal and rendering task."""

import asyncio
import hashlib
import json
import logging
import math
import os
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pxr import Tf, Usd, UsdGeom, UsdPhysics, UsdShade

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.rendering import (
    RemoteRenderingBackend,
    format_direction_for_filename,
    prepare_prims_with_composition,
    prepare_render_prims,
    render_from_prepared_composition,
    render_from_prepared_prims,
)
from world_understanding.functions.graphics.usd_model import USDModel
from world_understanding.utils.image_blankness import analyze_image_blankness
from world_understanding.utils.nvcf_utils import (
    NVCF_INVOCATION_HOST,
    resolve_endpoint_or_function_id,
)
from world_understanding.utils.object_store import ObjectStore
from world_understanding.utils.s3_utils import delete_s3_path
from world_understanding.utils.usd.stage import (
    MAX_FILENAME_STEM_LEN,
    MAX_PATH_COMPONENT_LEN,
    duplicate_stage,
    shorten_for_filesystem,
)

logger = logging.getLogger(__name__)


_MAX_SEGMENT_LEN = 80
"""Max characters per filesystem path segment before truncation."""

_PURPOSE_NOT_ALLOWED = "purpose_not_allowed"
_FULLY_TRANSPARENT = "fully_transparent"
_UNUSABLE_BBOX = "unusable_bbox"
_DISPLAY_OPACITY_EPSILON = 1e-6

_path_mapping_lock = threading.Lock()


def _allowed_purposes_from_filters(
    filters: dict[str, Any],
) -> tuple[str, ...] | None:
    """Normalize the optional render-purpose filter contract."""
    allowed_purposes = filters.get("allowed_purposes")
    if allowed_purposes is None:
        return None
    if isinstance(allowed_purposes, str):
        return (allowed_purposes,)
    return tuple(str(value) for value in allowed_purposes)


def _rigid_body_purpose_fallbacks_from_filters(
    filters: dict[str, Any],
) -> tuple[str, ...]:
    """Normalize purposes allowed only for guide-only rigid-body owners."""
    fallback_purposes = filters.get("rigid_body_purpose_fallbacks")
    if fallback_purposes is None:
        return ()
    if isinstance(fallback_purposes, str):
        return (fallback_purposes,)
    return tuple(str(value) for value in fallback_purposes)


def _bbox_purposes_from_filters(filters: dict[str, Any]) -> tuple[str, ...]:
    """Return the explicit bbox/framing purpose contract for prim filters."""
    allowed_purposes = _allowed_purposes_from_filters(filters)
    if allowed_purposes is None:
        return (str(UsdGeom.Tokens.default_),)
    return allowed_purposes


def _nearest_enabled_rigid_body_owner(prim: "Usd.Prim") -> str | None:
    """Return the nearest enabled authored rigid-body owner of ``prim``.

    USD Physics treats an applied ``RigidBodyAPI`` with no authored
    ``rigidBodyEnabled`` value as enabled through the schema fallback.
    """
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            enabled_attr = UsdPhysics.RigidBodyAPI(current).GetRigidBodyEnabledAttr()
            if not enabled_attr or enabled_attr.Get() is not False:
                return str(current.GetPath())
        current = current.GetParent()
    return None


def _explicit_members_world_bbox(
    stage: "Usd.Stage",
    member_paths: tuple[str, ...],
    base_purposes: tuple[str, ...],
    *,
    time_code: "Usd.TimeCode | None" = None,
    use_extents_hint: bool | None = None,
) -> dict[str, Any] | None:
    """Return the world-space union bound of explicit assembly members only."""
    member_bboxes: list[dict[str, Any]] = []
    for member_path in member_paths:
        member = stage.GetPrimAtPath(member_path)
        if not member or not member.IsValid():
            raise ValueError(f"Explicit bbox member does not exist: {member_path}")
        imageable = UsdGeom.Imageable(member)
        if not imageable:
            raise ValueError(
                "Explicit bbox member is not UsdGeom.Imageable: "
                f"{member_path} ({member.GetTypeName() or '<untyped>'})"
            )
        purpose = str(imageable.ComputePurpose())
        member_bbox = get_world_bbox_from_prim(
            member,
            tuple(dict.fromkeys((*base_purposes, purpose))),
            time_code=time_code,
            use_extents_hint=use_extents_hint,
        )
        if member_bbox is not None:
            member_bboxes.append(member_bbox)
    if not member_bboxes:
        return None

    bbox_min = [
        min(float(bbox["min"][axis]) for bbox in member_bboxes) for axis in range(3)
    ]
    bbox_max = [
        max(float(bbox["max"][axis]) for bbox in member_bboxes) for axis in range(3)
    ]
    return {
        "min": bbox_min,
        "max": bbox_max,
        "center": [(bbox_min[axis] + bbox_max[axis]) / 2.0 for axis in range(3)],
        "size": [bbox_max[axis] - bbox_min[axis] for axis in range(3)],
    }


def _effective_display_opacities(prim: "Usd.Prim") -> list[float]:
    """Return flattened display opacity at the render reference time."""
    primvar = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance("displayOpacity")
    if not primvar:
        return []

    values = None
    flatten_failure_types: list[str] = []
    for time_code in (Usd.TimeCode.Default(), Usd.TimeCode(0)):
        try:
            values = primvar.ComputeFlattened(time_code)
        except Exception as exc:  # pragma: no cover - malformed provider primvar
            flatten_failure_types.append(type(exc).__name__)
            values = None
        if values is not None:
            break
    if values is None:
        logger.warning(
            "Could not flatten displayOpacity for %s at the default or frame-0 "
            "reference time (%s); retaining the prim",
            prim.GetPath(),
            ", ".join(sorted(set(flatten_failure_types))) or "no value",
        )
        return []
    if flatten_failure_types:
        logger.debug(
            "displayOpacity fallback succeeded for %s after %s",
            prim.GetPath(),
            ", ".join(sorted(set(flatten_failure_types))),
        )

    try:
        raw_values = list(values)
    except TypeError:
        raw_values = [values]

    opacities: list[float] = []
    malformed_value_types: set[str] = set()
    nonfinite_count = 0
    for value in raw_values:
        try:
            opacity = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            malformed_value_types.add(type(exc).__name__)
            continue
        if math.isfinite(opacity):
            opacities.append(opacity)
        else:
            nonfinite_count += 1
    if malformed_value_types or nonfinite_count:
        logger.warning(
            "Ignored %d malformed and %d non-finite displayOpacity value(s) "
            "for %s (%s)",
            len(raw_values) - len(opacities) - nonfinite_count,
            nonfinite_count,
            prim.GetPath(),
            ", ".join(sorted(malformed_value_types)) or "no conversion error",
        )
    return opacities


def _has_usable_world_bbox(
    prim: "Usd.Prim",
    bbox_cache: Any | None = None,
) -> bool:
    """Return whether a Gprim can be framed by a focused render camera."""
    try:
        if bbox_cache is None:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode(0),
                [UsdGeom.Tokens.default_],
                useExtentsHint=False,
            )
        bbox_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bbox_range.IsEmpty():
            return False
        bbox_min = bbox_range.GetMin()
        bbox_max = bbox_range.GetMax()
        extents = [float(bbox_max[i] - bbox_min[i]) for i in range(3)]
        return all(math.isfinite(extent) and extent >= 0.0 for extent in extents) and (
            max(extents) > 0.0
        )
    except Exception:  # pragma: no cover - malformed provider bounds
        return False


def _render_evidence_skip_diagnostic(
    prim: "Usd.Prim",
    filters: dict[str, Any],
    bbox_cache: Any | None = None,
) -> dict[str, Any] | None:
    """Return a stable diagnostic when a selected Gprim is not render evidence."""
    allowed_purposes = _allowed_purposes_from_filters(filters)
    if (
        allowed_purposes is None
        and not filters.get("skip_fully_transparent", False)
        and not filters.get("skip_unusable_bbox", False)
    ):
        return None
    if not prim.IsA(UsdGeom.Gprim):
        return None

    prim_path = str(prim.GetPath())
    purpose = str(UsdGeom.Imageable(prim).ComputePurpose())
    if allowed_purposes is not None:
        allowed = sorted(set(allowed_purposes))
        if purpose not in allowed:
            return {
                "prim_path": prim_path,
                "reason": _PURPOSE_NOT_ALLOWED,
                "purpose": purpose,
                "allowed_purposes": allowed,
            }

    if filters.get("skip_fully_transparent", False):
        opacities = _effective_display_opacities(prim)
        if opacities and max(opacities) <= _DISPLAY_OPACITY_EPSILON:
            return {
                "prim_path": prim_path,
                "reason": _FULLY_TRANSPARENT,
                "max_display_opacity": max(opacities),
            }

    if filters.get("skip_unusable_bbox", False):
        if bbox_cache is None:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode(0),
                list(_bbox_purposes_from_filters(filters)),
                useExtentsHint=False,
            )
        if not _has_usable_world_bbox(prim, bbox_cache):
            return {
                "prim_path": prim_path,
                "reason": _UNUSABLE_BBOX,
            }

    return None


def _summarize_prim_filter_diagnostics(
    skipped_prims: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for item in skipped_prims:
        reason = str(item["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "skipped_prims": skipped_prims,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _sanitize_render_endpoint_for_diagnostic(endpoint: str) -> str:
    """Return a renderer endpoint that is useful without exposing URL secrets."""
    endpoint = endpoint.strip()
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        return endpoint
    if parsed.scheme and parsed.netloc:
        scheme = parsed.scheme
    else:
        try:
            parsed = urlsplit(
                endpoint if endpoint.startswith("//") else f"//{endpoint}"
            )
        except ValueError:
            return endpoint
        if not parsed.netloc:
            return endpoint
        scheme = "http"

    if not parsed.netloc:  # pragma: no cover - guarded by the branch above
        return endpoint

    netloc = parsed.netloc.rsplit("@", maxsplit=1)[-1]

    return urlunsplit((scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _resolve_render_endpoint_for_diagnostic(
    rendering_backend: Any,
) -> tuple[str | None, str | None, bool]:
    """Resolve enough renderer endpoint context to format failure diagnostics."""
    if not isinstance(rendering_backend, RemoteRenderingBackend):
        return None, None, False

    source: str | None = None
    raw_endpoint = getattr(rendering_backend, "base_url", None)
    if raw_endpoint:
        source = "base_url"
    else:
        raw_endpoint = os.environ.get("RENDER_ENDPOINT")
        if raw_endpoint:
            source = "RENDER_ENDPOINT"
        else:
            raw_endpoint = os.environ.get("NVCF_RENDER_FUNCTION_ID")
            if raw_endpoint:
                source = "NVCF_RENDER_FUNCTION_ID"

    if raw_endpoint is None:
        return None, None, False

    endpoint = str(raw_endpoint).strip()
    if not endpoint:
        return None, source, False

    endpoint = resolve_endpoint_or_function_id(endpoint)

    sanitized = _sanitize_render_endpoint_for_diagnostic(endpoint)
    is_nvcf = NVCF_INVOCATION_HOST in endpoint.lower()
    return sanitized, source, is_nvcf


def _zero_image_render_error_message(rendering_backend: Any) -> str:
    """Build a zero-render failure message without assuming REST means NVCF."""
    endpoint, source, is_nvcf = _resolve_render_endpoint_for_diagnostic(
        rendering_backend
    )
    endpoint_detail = (
        f" Resolved render endpoint from {source}: {endpoint}."
        if endpoint and source
        else ""
    )

    if is_nvcf:
        return (
            "Rendering produced 0 images. Check NVCF render function availability "
            f"and logs above.{endpoint_detail}"
        )
    if isinstance(rendering_backend, RemoteRenderingBackend):
        return (
            "Rendering produced 0 images. Check the configured render endpoint "
            f"and renderer service logs.{endpoint_detail}"
        )
    return "Rendering produced 0 images. Check renderer availability and logs above."


def _validate_positive_int_config(name: str, value: Any) -> int:
    """Parse a positive integer task config value."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _truncate_segment(name: str) -> str:
    """Truncate a path segment that exceeds *_MAX_SEGMENT_LEN*.

    If the sanitized name fits, return it unchanged.  Otherwise keep the first
    ``_MAX_SEGMENT_LEN - 9`` characters, append ``_`` and an 8-char hash of the
    full original name so the result is unique and still recognisable.
    """
    if len(name) <= _MAX_SEGMENT_LEN:
        return name
    short_hash = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"{name[: _MAX_SEGMENT_LEN - 9]}_{short_hash}"


def _record_path_mapping(base_dir: Path, original: str, truncated: str) -> None:
    """Append a truncated-path mapping entry to ``path_mapping.json``."""
    mapping_file = base_dir / "path_mapping.json"
    with _path_mapping_lock:
        if mapping_file.exists():
            try:
                mappings = json.loads(mapping_file.read_text())
            except (json.JSONDecodeError, OSError):
                mappings = {}
        else:
            mappings = {}
        if truncated not in mappings:
            mappings[truncated] = original
            try:
                mapping_file.write_text(json.dumps(mappings, indent=2))
            except OSError:
                pass


def _blank_dataset_render_message(blank_count: int, total_count: int) -> str:
    return (
        f"{blank_count}/{total_count} dataset renders are blank or near-blank. "
        "The VLM cannot produce meaningful predictions from these. Check the "
        "rendering endpoint logs and any HDRI / dome-light configuration "
        "(WU_OVRTX_DEFAULT_HDRI, WU_OVRTX_DEFAULT_HDRI_INTENSITY)."
    )


def _is_blank_render_status(status: Any) -> bool:
    return isinstance(status, str) and status == "blank_render"


def _is_config_enabled(value: Any, default: bool) -> bool:
    """Interpret permissive boolean task config values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def prim_path_to_directory_structure(
    prim_path: str, base_dir: Path, filename: str
) -> Path:
    """Convert a USD prim path to a directory structure.

    Long path segments (>80 chars) are truncated and suffixed with an 8-char
    hash for uniqueness.  A ``path_mapping.json`` file is written to
    *base_dir* whenever truncation occurs so the original names can be
    recovered.

    Args:
        prim_path: USD prim path (e.g., "/World/A/B")
        base_dir: Base output directory
        filename: Filename to use (e.g., "B_front.png")

    Returns:
        Full file path with directory structure

    Example:
        prim_path="/World/A/B", base_dir="output", filename="B_front.png"
        -> Path("output/World/A/B_front.png")
    """
    # Remove leading slash and split path
    path_parts = prim_path.strip("/").split("/")

    if not path_parts:  # pragma: no cover - str.split always returns at least one part
        return base_dir / filename

    # Use all but the last part as directory structure
    # The last part becomes part of the filename
    if len(path_parts) > 1:
        dir_parts = path_parts[:-1]
    else:
        # Single level path
        dir_parts = []

    # Sanitize and bound each directory component to avoid Errno 36.
    sanitized_dirs = [
        shorten_for_filesystem(part, max_len=MAX_PATH_COMPONENT_LEN)
        for part in dir_parts
    ]

    # Truncate filename as well (preserve extension)
    fname_stem = Path(filename).stem
    fname_ext = Path(filename).suffix
    trunc_stem = _truncate_segment(fname_stem)
    if trunc_stem != fname_stem:
        _record_path_mapping(base_dir, fname_stem, trunc_stem)
        filename = f"{trunc_stem}{fname_ext}"

    # Build the full directory path
    full_dir = base_dir
    for dir_part in sanitized_dirs:
        full_dir = full_dir / dir_part

    # Create the directory if it doesn't exist
    full_dir.mkdir(parents=True, exist_ok=True)

    # Return the full file path
    return full_dir / filename


def get_world_bbox_from_prim(
    prim: "Usd.Prim",
    included_purposes: Iterable[str] | None = None,
    *,
    time_code: "Usd.TimeCode | None" = None,
    use_extents_hint: bool | None = None,
) -> dict[str, Any] | None:
    """Get the world-space bounding box for a prim.

    Args:
        prim: USD prim
        included_purposes: Optional UsdGeom purposes to include. Defaults to
            ``default`` only.
        time_code: Optional evaluation time. Defaults to ``Usd.TimeCode.Default()``.
        use_extents_hint: Optional BBoxCache extents-hint policy. ``None``
            preserves the USD constructor default for existing callers.

    Returns:
        Dictionary with bbox information:
        - min: [x, y, z] minimum corner
        - max: [x, y, z] maximum corner
        - center: [x, y, z] center point
        - size: [width, height, depth]
    """
    try:
        # Create a BBoxCache object to compute the bounding box
        bbox_time = time_code if time_code is not None else Usd.TimeCode.Default()
        purposes = (
            list(included_purposes)
            if included_purposes is not None
            else [UsdGeom.Tokens.default_]
        )
        if use_extents_hint is None:
            bbox_cache = UsdGeom.BBoxCache(bbox_time, purposes)
        else:
            bbox_cache = UsdGeom.BBoxCache(
                bbox_time,
                purposes,
                useExtentsHint=use_extents_hint,
            )

        # Compute the world-space bounding box for the prim
        bbox = bbox_cache.ComputeWorldBound(prim)

        # Use ComputeAlignedRange() to get the axis-aligned bbox in world
        # space. GetRange() returns the untransformed local extent, which
        # ignores xformOp:scale and other prim transforms.
        bbox_range = bbox.ComputeAlignedRange()
        if bbox_range.IsEmpty():
            return None
        bbox_min = bbox_range.GetMin()
        bbox_max = bbox_range.GetMax()

        # Calculate center and size
        center = [
            (bbox_min[0] + bbox_max[0]) / 2.0,
            (bbox_min[1] + bbox_max[1]) / 2.0,
            (bbox_min[2] + bbox_max[2]) / 2.0,
        ]

        size = [
            bbox_max[0] - bbox_min[0],
            bbox_max[1] - bbox_min[1],
            bbox_max[2] - bbox_min[2],
        ]

        return {
            "min": [bbox_min[0], bbox_min[1], bbox_min[2]],
            "max": [bbox_max[0], bbox_max[1], bbox_max[2]],
            "center": center,
            "size": size,
        }
    except Exception as e:
        logger.debug(f"Could not compute world bbox for {prim.GetPath()}: {e}")
        return None


def get_stage_world_bbox(
    stage: "Usd.Stage",
    included_purposes: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """Get the world-space bounding box for the entire stage.

    Args:
        stage: USD stage
        included_purposes: Optional UsdGeom purposes to include. Defaults to
            ``default`` only.

    Returns:
        Dictionary with bbox information:
        - min: [x, y, z] minimum corner
        - max: [x, y, z] maximum corner
        - center: [x, y, z] center point
        - size: [width, height, depth]
    """
    try:
        # Create a BBoxCache object to compute the bounding box
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            (
                list(included_purposes)
                if included_purposes is not None
                else [UsdGeom.Tokens.default_]
            ),
        )

        # Compute the world-space bounding box for the entire stage
        root_prim = stage.GetPseudoRoot()
        bbox = bbox_cache.ComputeWorldBound(root_prim)

        # Use ComputeAlignedRange() to respect prim transforms
        bbox_range = bbox.ComputeAlignedRange()
        if bbox_range.IsEmpty():
            return None
        bbox_min = bbox_range.GetMin()
        bbox_max = bbox_range.GetMax()

        # Calculate center and size
        center = [
            (bbox_min[0] + bbox_max[0]) / 2.0,
            (bbox_min[1] + bbox_max[1]) / 2.0,
            (bbox_min[2] + bbox_max[2]) / 2.0,
        ]

        size = [
            bbox_max[0] - bbox_min[0],
            bbox_max[1] - bbox_min[1],
            bbox_max[2] - bbox_min[2],
        ]

        return {
            "min": [bbox_min[0], bbox_min[1], bbox_min[2]],
            "max": [bbox_max[0], bbox_max[1], bbox_max[2]],
            "center": center,
            "size": size,
        }
    except Exception as e:
        logger.debug(f"Could not compute stage world bbox: {e}")
        return None


def scale_bbox_by_mpu(bbox: dict[str, Any], mpu: float) -> dict[str, Any]:
    """Scale a bounding box by meters per unit to get values in meters.

    Args:
        bbox: Bounding box dictionary with min, max, center, size
        mpu: Meters per unit scale factor

    Returns:
        Scaled bounding box in meters
    """
    if not bbox or not mpu:
        return None

    return {
        "min": [v * mpu for v in bbox["min"]],
        "max": [v * mpu for v in bbox["max"]],
        "center": [v * mpu for v in bbox["center"]],
        "size": [v * mpu for v in bbox["size"]],
    }


def compute_relative_metrics(
    prim_bbox: dict[str, Any], stage_bbox: dict[str, Any]
) -> dict[str, Any]:
    """Compute relative size and position metrics for a prim compared to stage.

    Args:
        prim_bbox: Prim's bounding box
        stage_bbox: Stage's bounding box

    Returns:
        Dictionary with:
        - relative_size: [width_ratio, height_ratio, depth_ratio]
        - relative_volume: Volume ratio (prim_volume / stage_volume)
        - relative_center: [x_offset, y_offset, z_offset] from stage center
        - normalized_center: Center position normalized to [-1, 1] within stage bounds
    """
    if not prim_bbox or not stage_bbox:
        return None

    # Compute relative size (ratio of dimensions)
    relative_size = [
        (
            prim_bbox["size"][0] / stage_bbox["size"][0]
            if stage_bbox["size"][0] > 0
            else 0
        ),
        (
            prim_bbox["size"][1] / stage_bbox["size"][1]
            if stage_bbox["size"][1] > 0
            else 0
        ),
        (
            prim_bbox["size"][2] / stage_bbox["size"][2]
            if stage_bbox["size"][2] > 0
            else 0
        ),
    ]

    # Compute relative volume
    prim_volume = prim_bbox["size"][0] * prim_bbox["size"][1] * prim_bbox["size"][2]
    stage_volume = stage_bbox["size"][0] * stage_bbox["size"][1] * stage_bbox["size"][2]
    relative_volume = prim_volume / stage_volume if stage_volume > 0 else 0

    # Compute relative center (offset from stage center)
    relative_center = [
        prim_bbox["center"][0] - stage_bbox["center"][0],
        prim_bbox["center"][1] - stage_bbox["center"][1],
        prim_bbox["center"][2] - stage_bbox["center"][2],
    ]

    # Compute normalized center position within stage bounds [-1, 1]
    normalized_center = [
        (
            2.0 * relative_center[0] / stage_bbox["size"][0]
            if stage_bbox["size"][0] > 0
            else 0
        ),
        (
            2.0 * relative_center[1] / stage_bbox["size"][1]
            if stage_bbox["size"][1] > 0
            else 0
        ),
        (
            2.0 * relative_center[2] / stage_bbox["size"][2]
            if stage_bbox["size"][2] > 0
            else 0
        ),
    ]

    return {
        "relative_size": relative_size,
        "relative_volume": relative_volume,
        "relative_center": relative_center,
        "normalized_center": normalized_center,
    }


class USDPrimTraversalAndRenderingTask(Task):
    """Traverse USD prims and render views for each."""

    def __init__(self):
        self.name = "USDPrimTraversalAndRendering"
        self.description = "Traverse USD prims and render configured views"

    @staticmethod
    def _propagate_root_prim(prim_filters, rendering_config):
        """Propagate root_prim from prim_filters into rendering config."""
        root_prim_filter = prim_filters.get("root_prim")
        if root_prim_filter and not rendering_config.root_prim_path:
            return replace(rendering_config, root_prim_path=root_prim_filter)
        return rendering_config

    @staticmethod
    def _propagate_bbox_purposes(prim_filters, rendering_config):
        """Keep traversal bounds and camera framing on one explicit contract."""
        return replace(
            rendering_config,
            bbox_purposes=_bbox_purposes_from_filters(prim_filters),
        )

    def run(self, context: dict[str, Any], object_store: ObjectStore) -> dict[str, Any]:
        """Traverse USD prims and render views.

        Expected context inputs:
            - prim_filters: Filters for prim selection
            - render_output_dir: Directory for rendered images
            - extract_metadata: Whether to extract prim metadata
            - extract_display_color: Whether to extract display color attribute (default: False)
            - extract_material_bindings: Whether to extract material bindings (default: True)
            - extract_hierarchy: Whether to extract hierarchy info (default: True)
            - skip_existing: Skip rendering if output files already exist
            - batch_size: Number of prims to render in each batch (default: 10)

        Expected from object_store:
            - usd_stage: The loaded USD stage
            - rendering_backend: The rendering backend
            - rendering_config: The rendering configuration
            - usd_model: Optional USDModel for hierarchy queries

        Updates context with:
            - rendered_prims: List of rendered prim paths
            - prim_data: List of dicts with prim info and image paths
            - total_images_rendered: Total number of images rendered
            - prim_filter_diagnostics: Stable skip records and reason counts
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Get inputs
        stage = object_store.get("usd_stage")
        if not stage:
            raise ValueError("USD stage not found in object store")

        rendering_backend = object_store.get("rendering_backend")
        rendering_config = object_store.get("rendering_config")
        if not rendering_backend:
            raise ValueError("Rendering backend not found in object store")
        if not rendering_config:
            raise ValueError("Rendering config not found in object store")

        prim_filters = context.get("prim_filters", {})
        render_output_dir = Path(context.get("render_output_dir", "output/renders"))
        skip_existing = context.get("skip_existing", False)
        skip_existing_materials = context.get("skip_existing_materials", False)
        batch_size = context.get("batch_size", 10)  # Default batch size
        num_workers = context.get("num_workers", 1)  # Default to sequential processing
        rendering_modes = context.get(
            "rendering_modes", ["prim_with_stage", "prim_only"]
        )

        # Get output_dir for relative path computation
        # Use render_output_dir.parent if not provided
        output_dir = context.get("output_dir")
        if output_dir is None:
            output_dir = render_output_dir.parent
            listener.info(
                f"Using render_output_dir.parent as base "
                f"for relative paths: {output_dir}"
            )
        else:
            output_dir = Path(output_dir)

        rendering_config = self._propagate_root_prim(prim_filters, rendering_config)
        rendering_config = self._propagate_bbox_purposes(
            prim_filters,
            rendering_config,
        )

        listener.info("Starting USD prim traversal and rendering")
        listener.info(f"  Filters: {prim_filters}")
        listener.info(f"  Camera type: {rendering_config.camera_view_type.value}")
        listener.info(f"  Number of views: {len(rendering_config.camera_ordering)}")
        listener.info(f"  Skip existing: {skip_existing}")
        listener.info(f"  Skip existing materials: {skip_existing_materials}")
        listener.info(f"  Batch size: {batch_size}")
        listener.info(f"  Number of workers: {num_workers}")
        listener.info(f"  Rendering modes: {rendering_modes}")

        # Extract stage-level metrics
        stage_up_axis = str(UsdGeom.GetStageUpAxis(stage))
        listener.info(f"Stage up axis: {stage_up_axis}")
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        listener.info(f"Stage meters per unit: {meters_per_unit}")

        # Compute stage world bounding box
        stage_world_bbox = get_stage_world_bbox(
            stage,
            rendering_config.bbox_purposes,
        )
        stage_world_bbox_meters = (
            scale_bbox_by_mpu(stage_world_bbox, meters_per_unit)
            if stage_world_bbox
            else None
        )

        if stage_world_bbox:
            listener.info(f"Stage world bbox size: {stage_world_bbox['size']}")
            if stage_world_bbox_meters:
                listener.info(
                    f"Stage world bbox size (meters): {stage_world_bbox_meters['size']}"
                )

        # Store stage metrics in object store for other tasks
        object_store.set("stage_up_axis", stage_up_axis)
        object_store.set("meters_per_unit", meters_per_unit)
        object_store.set("stage_world_bbox", stage_world_bbox)
        object_store.set("stage_world_bbox_meters", stage_world_bbox_meters)

        # Collect prims to render based on filters
        prim_filter_skips: list[dict[str, Any]] = []
        assembly_target_members: dict[str, list[str]] = {}
        recovered_guide_gprim_targets: set[str] = set()
        all_prims = self._collect_prims(
            stage,
            prim_filters,
            listener,
            diagnostics=prim_filter_skips,
            assembly_target_members=assembly_target_members,
            recovered_guide_gprim_targets=recovered_guide_gprim_targets,
        )
        rendering_config = replace(
            rendering_config,
            assembly_target_members={
                target: tuple(member_paths)
                for target, member_paths in assembly_target_members.items()
            },
            recovered_guide_gprim_targets=tuple(sorted(recovered_guide_gprim_targets)),
        )
        context["assembly_render_targets"] = {
            target: member_paths.copy()
            for target, member_paths in assembly_target_members.items()
        }
        context["recovered_guide_gprim_targets"] = sorted(recovered_guide_gprim_targets)
        context["prim_filter_diagnostics"] = _summarize_prim_filter_diagnostics(
            prim_filter_skips
        )
        listener.info(f"Found {len(all_prims)} USD prims total")
        self._require_matching_prims(
            stage,
            all_prims,
            prim_filters,
            context["prim_filter_diagnostics"],
        )

        prims_to_render, prim_data, total_images = self._collect_and_filter_prims(
            stage,
            all_prims,
            prim_filters,
            rendering_config,
            rendering_modes,
            context,
            object_store,
            render_output_dir,
            output_dir,
            listener,
        )

        # ========== PASS 1: PREPARE STAGES (NO BATCHING) ==========
        rgb_modes = context.get("rgb_rendering_modes", rendering_modes)
        sensor_modes = context.get("sensor_rendering_modes", [])
        image_height = context.get("image_height", 512)

        prepared_stages = self._prepare_stages(
            stage,
            prims_to_render,
            rgb_modes,
            rendering_config,
            context,
            listener,
        )

        # ========== PASS 1.5: UPLOAD STAGES TO S3 (NVCF ONLY) ==========
        s3_cleanup_uris = self._upload_stages_to_s3(
            prepared_stages,
            rendering_backend,
            context,
            listener,
        )

        # ========== PASS 2: RENDER IN BATCHES ==========
        # Now render the prepared stages in batches for efficiency
        num_batches = (len(prims_to_render) + batch_size - 1) // batch_size
        listener.info(
            f"Pass 2: Rendering {len(prims_to_render)} prims in {num_batches} batches"
        )

        # Convert prim_data from list to dict for easier lookup in parallel processing
        prim_data_dict = {p["prim_path"]: p for p in prim_data}

        # Create all (batch, mode) combinations for RGB modes only
        # Sensor modes will be passed to the rendering backend
        batch_mode_tasks = []
        for batch_start in range(0, len(prims_to_render), batch_size):
            batch_end = min(batch_start + batch_size, len(prims_to_render))
            for render_mode in rgb_modes:
                batch_mode_tasks.append((batch_start, batch_end, render_mode))

        num_total_tasks = len(batch_mode_tasks)

        # Count skipped prims for final reporting
        skipped_count = len(prim_data) - len(prims_to_render)

        try:
            # Use parallel processing if we have multiple workers and multiple tasks
            if num_workers > 1 and num_total_tasks > 1:
                # Parallel processing
                listener.info(
                    f"Using {num_workers} workers for parallel processing of "
                    f"{num_batches} batches × {len(rgb_modes)} RGB modes = "
                    f"{num_total_tasks} total tasks"
                )
                if sensor_modes:
                    listener.info(
                        f"  Sensor modes {sensor_modes} will be rendered with each RGB mode"
                    )

                # Process all (batch, mode) combinations in parallel
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all (batch, mode) tasks
                    future_to_task = {
                        executor.submit(
                            self._process_batch_with_retry_split,
                            batch_start,
                            batch_end,
                            prims_to_render,
                            prim_data_dict,
                            prepared_stages,
                            rendering_backend,
                            render_mode,
                            render_output_dir,
                            output_dir,
                            num_total_tasks,
                            batch_size,
                            listener,
                            sensor_modes,
                            image_height,
                        ): (batch_start, batch_end, render_mode)
                        for batch_start, batch_end, render_mode in batch_mode_tasks
                    }

                    # Collect results as they complete
                    all_prim_images_data = {}  # Collect all image data from threads
                    completed_tasks = 0
                    for future in as_completed(future_to_task):
                        batch_start, batch_end, render_mode = future_to_task[future]
                        try:
                            batch_images, batch_failures, batch_prim_images = (
                                future.result()
                            )
                            total_images += batch_images
                            completed_tasks += 1

                            # Emit progress event with completed batch info
                            try:
                                listener.event(
                                    "rendering.progress",
                                    {
                                        "task_name": "USDPrimTraversalAndRendering",
                                        "current": completed_tasks,
                                        "total": num_total_tasks,
                                        "percent": int(
                                            (completed_tasks / num_total_tasks) * 100
                                        ),
                                        "message": f"Rendered batch {completed_tasks}/{num_total_tasks} ({batch_images} images, mode: {render_mode})",
                                        "images_rendered": total_images,
                                        "render_mode": render_mode,
                                    },
                                )
                            except Exception as e:
                                logger.warning(f"Failed to emit progress event: {e}")

                            # Collect the image data from this (batch, mode) task
                            for prim_path, images in batch_prim_images.items():
                                if prim_path not in all_prim_images_data:
                                    all_prim_images_data[prim_path] = []
                                all_prim_images_data[prim_path].extend(images)

                            # Track failed batches
                            if batch_failures:
                                if "failed_batches" not in context:
                                    context["failed_batches"] = []
                                context["failed_batches"].extend(batch_failures)

                        except Exception as e:
                            listener.error(
                                f"Failed to process batch {batch_start}-{batch_end}, mode '{render_mode}': {e}",
                                exc_info=True,
                            )
                            if "failed_batches" not in context:
                                context["failed_batches"] = []
                            context["failed_batches"].append(
                                {
                                    "batch_start": batch_start,
                                    "batch_end": batch_end,
                                    "render_mode": render_mode,
                                    "error": str(e),
                                }
                            )

                    # After all threads complete, merge the image data back into prim_data_dict
                    for prim_path, images in all_prim_images_data.items():
                        if prim_path in prim_data_dict:
                            prim_data_dict[prim_path]["images"].extend(images)

            else:
                # Sequential processing
                listener.info(
                    f"Using sequential processing for "
                    f"{num_batches} batches × {len(rgb_modes)} RGB modes = "
                    f"{num_total_tasks} total tasks"
                )
                if sensor_modes:
                    listener.info(
                        f"  Sensor modes {sensor_modes} will be rendered with each RGB mode"
                    )

                completed_tasks = 0
                for batch_start, batch_end, render_mode in batch_mode_tasks:
                    batch_images, batch_failures, batch_prim_images = (
                        self._process_batch_with_retry_split(
                            batch_start,
                            batch_end,
                            prims_to_render,
                            prim_data_dict,
                            prepared_stages,
                            rendering_backend,
                            render_mode,
                            render_output_dir,
                            output_dir,
                            num_total_tasks,
                            batch_size,
                            listener,
                            sensor_modes,
                            image_height,
                        )
                    )

                    total_images += batch_images
                    completed_tasks += 1

                    # Emit progress event with completed batch info
                    try:
                        listener.event(
                            "rendering.progress",
                            {
                                "task_name": "USDPrimTraversalAndRendering",
                                "current": completed_tasks,
                                "total": num_total_tasks,
                                "percent": int(
                                    (completed_tasks / num_total_tasks) * 100
                                ),
                                "message": f"Rendered batch {completed_tasks}/{num_total_tasks} ({batch_images} images, mode: {render_mode})",
                                "images_rendered": total_images,
                                "render_mode": render_mode,
                            },
                        )
                    except Exception as e:
                        logger.warning(f"Failed to emit progress event: {e}")

                    # Merge the image data from this (batch, mode) task into prim_data_dict
                    for prim_path, images in batch_prim_images.items():
                        if prim_path in prim_data_dict:
                            prim_data_dict[prim_path]["images"].extend(images)

                    # Track failed batches
                    if batch_failures:
                        if "failed_batches" not in context:
                            context["failed_batches"] = []
                        context["failed_batches"].extend(batch_failures)

        finally:
            # ========== POST-PROCESSING: CLEANUP S3 FILES ==========
            self._cleanup_s3(s3_cleanup_uris, listener)

        # Convert prim_data_dict back to list for context
        prim_data = list(prim_data_dict.values())

        self._check_blank_dataset_renders(
            prim_data,
            output_dir,
            rgb_modes=rgb_modes,
            sensor_modes=sensor_modes,
            listener=listener,
            context=context,
        )
        # Fail early if rendering produced no images and no blank-render guardrail
        # already explained the failure.
        if total_images == 0:
            raise RuntimeError(_zero_image_render_error_message(rendering_backend))
        missing_image_prims = [
            prim_path
            for prim_path in prims_to_render
            if not prim_data_dict.get(prim_path, {}).get("images")
        ]
        if missing_image_prims:
            context["missing_image_prims"] = missing_image_prims
            sample = ", ".join(missing_image_prims[:5])
            message = (
                f"Rendering produced no images for {len(missing_image_prims)} of "
                f"{len(prims_to_render)} selected prims. Sample: {sample}"
            )
            if _is_config_enabled(context.get("fail_on_missing_prim_images"), True):
                raise RuntimeError(message)
            listener.warning(message)

        # Update context with results
        context["rendered_prims"] = prims_to_render
        context["prim_data"] = prim_data
        context["total_images_rendered"] = total_images
        context["output_dir"] = str(output_dir)  # Store the resolved output_dir

        # Store prim data in object store for next task
        object_store.set("prim_data", prim_data)

        listener.info("USD rendering complete:")
        listener.info(f"  Prims rendered: {len(prims_to_render)}")
        listener.info(f"  Total images: {total_images}")

        # Emit overall rendering completion event
        try:
            listener.event(
                "rendering.all_completed",
                {
                    "total_prims": len(prims_to_render),
                    "total_images": total_images,
                    "skipped_prims": skipped_count,
                    "rendering_modes": rendering_modes,
                    "num_views": len(rendering_config.camera_ordering),
                    "failed_batches": len(context.get("failed_batches", [])),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to emit overall rendering event: {e}")

        return context

    def _check_blank_dataset_renders(
        self,
        prim_data: list[dict[str, Any]],
        output_dir: Path,
        *,
        rgb_modes: list[str],
        sensor_modes: list[str],
        listener,
        context: dict[str, Any],
    ) -> None:
        """Fail when too many material-dataset renders are blank."""
        candidates = self._dataset_render_candidates(
            prim_data,
            output_dir,
            rgb_modes=rgb_modes,
            sensor_modes=sensor_modes,
        )
        overlapping_blank_failures: list[dict[str, Any]] = []
        if not candidates:
            blank_failures = self._blank_render_failure_candidates(
                context,
                rgb_modes=rgb_modes,
                sensor_modes=sensor_modes,
            )
            if not blank_failures:
                return
            blank_failures = self._dedupe_blank_render_failures(blank_failures)
            candidates = []
        else:
            selected_rgb_modes = sorted(
                {
                    str(candidate.get("render_mode", ""))
                    for candidate in candidates
                    if candidate.get("render_mode")
                }
            )
            blank_failures = self._blank_render_failure_candidates(
                context,
                rgb_modes=selected_rgb_modes or rgb_modes,
                sensor_modes=sensor_modes,
            )
            candidate_keys = {
                self._blank_render_candidate_key(candidate) for candidate in candidates
            }
            overlapping_blank_failures = self._dedupe_blank_render_failures(
                [
                    failure
                    for failure in blank_failures
                    if self._blank_render_candidate_key(failure) in candidate_keys
                ],
            )
            blank_failures = self._dedupe_blank_render_failures(
                blank_failures,
                existing_keys=candidate_keys,
            )

        blank_renders: list[dict[str, Any]] = []
        for candidate in candidates:
            renderer_stats = candidate.get("stats")
            if candidate.get("blank_render") and isinstance(renderer_stats, dict):
                if renderer_stats.get("blank", True):
                    blank_renders.append(
                        {
                            **candidate,
                            "stats": renderer_stats,
                        }
                    )
                continue

            image_path = Path(candidate["path"])
            try:
                stats = analyze_image_blankness(image_path)
            except Exception as exc:
                listener.warning(
                    f"Could not inspect render output for blankness "
                    f"({image_path}): {exc}"
                )
                blank_renders.append(
                    {
                        **candidate,
                        "analysis_error": str(exc),
                        "stats": {
                            "blank": True,
                            "reason": "analysis_error",
                        },
                    }
                )
                continue
            if stats.blank:
                blank_renders.append(
                    {
                        **candidate,
                        "stats": stats.to_dict(),
                    }
                )

        blank_render_keys = {
            self._blank_render_candidate_key(blank_render)
            for blank_render in blank_renders
        }
        for failure in overlapping_blank_failures:
            key = self._blank_render_candidate_key(failure)
            if key in blank_render_keys:
                continue
            blank_render_keys.add(key)
            blank_renders.append(failure)

        blank_renders.extend(blank_failures)
        if not blank_renders:
            return

        checked_count = len(candidates) + len(blank_failures)
        blank_count = len(blank_renders)
        context["blank_renders"] = blank_renders
        context["blank_render_checked_count"] = checked_count

        threshold = float(context.get("blank_render_failure_threshold", 0.5))
        ratio = blank_count / checked_count
        message = _blank_dataset_render_message(blank_count, checked_count)
        fail_on_blank_renders = _is_config_enabled(
            context.get("fail_on_blank_dataset_renders"), True
        )
        if ratio > threshold and fail_on_blank_renders:
            raise RuntimeError(message)

        listener.warning(message)

    @staticmethod
    def _blank_render_failure_candidates(
        context: dict[str, Any],
        *,
        rgb_modes: list[str],
        sensor_modes: list[str],
    ) -> list[dict[str, Any]]:
        rgb_mode_set = set(rgb_modes)
        sensor_mode_set = set(sensor_modes)
        blank_failures: list[dict[str, Any]] = []
        for failure in context.get("failed_batches", []):
            if not isinstance(failure, dict) or not failure.get("blank_render"):
                continue
            render_mode = str(failure.get("render_mode", ""))
            if render_mode in sensor_mode_set:
                continue
            if rgb_mode_set and render_mode and render_mode not in rgb_mode_set:
                continue
            blank_failures.append(
                {
                    "prim_path": failure.get("prim_path"),
                    "path": failure.get("path"),
                    "render_mode": render_mode,
                    "view": failure.get("view"),
                    "camera": failure.get("camera"),
                    "stats": failure.get(
                        "stats",
                        {"blank": True, "reason": "remote_blank_render"},
                    ),
                    "error": failure.get("error"),
                    "blank_render": True,
                }
            )
        return blank_failures

    @staticmethod
    def _dedupe_blank_render_failures(
        blank_failures: list[dict[str, Any]],
        *,
        existing_keys: set[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        seen = set(existing_keys or set())
        deduped: list[dict[str, Any]] = []
        for failure in blank_failures:
            key = USDPrimTraversalAndRenderingTask._blank_render_candidate_key(failure)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(failure)
        return deduped

    @staticmethod
    def _blank_render_candidate_key(candidate: dict[str, Any]) -> tuple[str, str]:
        prim_path = candidate.get("prim_path")
        if prim_path is None:
            prim_path = (
                f"unknown:{candidate.get('camera')}:{candidate.get('frame')}:"
                f"{candidate.get('path')}:{candidate.get('view')}"
            )
        return (str(prim_path), str(candidate.get("render_mode", "")))

    @staticmethod
    def _blank_render_failures_from_results(
        result: dict[str, Any],
        *,
        batch_start: int,
        batch_prims: list[str],
        render_mode: str,
    ) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for camera_result in result.get("results", []):
            status = camera_result.get("status")
            blank_frames = camera_result.get("blank_render_frames", [])
            if not _is_blank_render_status(status):
                continue

            camera_name = camera_result.get("camera", "default")
            if blank_frames:
                for frame_info in blank_frames:
                    frame = frame_info.get("frame")
                    prim_path = None
                    if isinstance(frame, int):
                        prim_index = frame - batch_start
                        if 0 <= prim_index < len(batch_prims):
                            prim_path = batch_prims[prim_index]
                    failures.append(
                        {
                            "batch_start": batch_start,
                            "batch_prims": batch_prims,
                            "render_mode": render_mode,
                            "camera": camera_name,
                            "prim_path": prim_path,
                            "frame": frame,
                            "blank_render": True,
                            "stats": frame_info.get(
                                "stats",
                                {"blank": True, "reason": "remote_blank_render"},
                            ),
                            "error": camera_result.get("error"),
                        }
                    )
                continue

            for frame_offset, prim_path in enumerate(batch_prims):
                failures.append(
                    {
                        "batch_start": batch_start,
                        "batch_prims": batch_prims,
                        "render_mode": render_mode,
                        "camera": camera_name,
                        "prim_path": prim_path,
                        "frame": batch_start + frame_offset,
                        "blank_render": True,
                        "stats": {"blank": True, "reason": "remote_blank_render"},
                        "error": camera_result.get("error"),
                    }
                )
        return failures

    @staticmethod
    def _blank_render_frame_stats_by_prim(
        camera_result: dict[str, Any],
        *,
        batch_start: int,
        batch_prims: list[str],
    ) -> dict[str, dict[str, Any]]:
        stats_by_prim: dict[str, dict[str, Any]] = {}
        blank_frames = camera_result.get("blank_render_frames", [])
        if not isinstance(blank_frames, list):
            return stats_by_prim

        for frame_info in blank_frames:
            if not isinstance(frame_info, dict):
                continue
            frame = frame_info.get("frame")
            if not isinstance(frame, int):
                continue
            prim_index = frame - batch_start
            if not 0 <= prim_index < len(batch_prims):
                continue
            stats = frame_info.get("stats")
            if not isinstance(stats, dict):
                stats = {"blank": True, "reason": "remote_blank_render"}
            stats_by_prim[batch_prims[prim_index]] = stats
        return stats_by_prim

    @staticmethod
    def _dataset_render_candidates(
        prim_data: list[dict[str, Any]],
        output_dir: Path,
        *,
        rgb_modes: list[str],
        sensor_modes: list[str],
    ) -> list[dict[str, Any]]:
        """Return composition renders when present, otherwise all RGB renders."""
        rgb_mode_set = set(rgb_modes)
        sensor_mode_set = set(sensor_modes)
        candidates: list[dict[str, Any]] = []

        for prim_info in prim_data:
            prim_path = prim_info.get("prim_path")
            for image_info in prim_info.get("images", []):
                if not isinstance(image_info, dict):
                    continue
                image_path_value = image_info.get("path")
                if not image_path_value:
                    continue
                render_mode = str(image_info.get("render_mode", ""))
                if render_mode in sensor_mode_set:
                    continue
                if rgb_mode_set and render_mode and render_mode not in rgb_mode_set:
                    continue

                image_path = Path(str(image_path_value))
                if not image_path.is_absolute():
                    image_path = output_dir / image_path
                candidate = {
                    "prim_path": prim_path,
                    "path": str(image_path),
                    "render_mode": render_mode,
                    "view": image_info.get("view"),
                    "camera": image_info.get("camera"),
                }
                if image_info.get("blank_render"):
                    candidate["blank_render"] = True
                if isinstance(image_info.get("stats"), dict):
                    candidate["stats"] = image_info["stats"]
                candidates.append(candidate)

        composition_candidates = [
            candidate
            for candidate in candidates
            if candidate["render_mode"] == "composition"
            or Path(candidate["path"]).name.endswith("_composition.png")
        ]
        return composition_candidates or candidates

    def _collect_and_filter_prims(
        self,
        stage,
        all_prims: list[str],
        prim_filters: dict,
        rendering_config,
        rendering_modes: list[str],
        context: dict[str, Any],
        object_store,
        render_output_dir: Path,
        output_dir: Path,
        listener,
    ) -> tuple[list[str], list[dict[str, Any]], int]:
        """Collect and filter prims, returning (prims_to_render, prim_data, total_images)."""
        extract_metadata = context.get("extract_metadata", False)
        extract_display_color = context.get("extract_display_color", False)
        extract_material_bindings = context.get("extract_material_bindings", True)
        extract_hierarchy = context.get("extract_hierarchy", True)
        skip_existing = context.get("skip_existing", False)
        skip_existing_materials = context.get("skip_existing_materials", False)

        # Get optional USDModel for efficient hierarchy queries
        usd_model = object_store.get("usd_model")

        # Get stage metrics for relative bbox computation
        meters_per_unit = object_store.get("meters_per_unit")
        stage_world_bbox = object_store.get("stage_world_bbox")

        prim_data: list[dict[str, Any]] = []
        total_images = 0
        prims_to_render: list[str] = []
        skipped_prims: list[tuple[str, dict[str, Any]]] = []
        skipped_materials_count = 0

        for prim_path in all_prims:
            prim_info: dict[str, Any] = {
                "prim_path": prim_path,
                "images": [],
                "metadata": {},
            }

            prim = stage.GetPrimAtPath(prim_path)

            # Filter prims with direct material bindings if requested
            if skip_existing_materials:
                if prim.HasAPI(UsdShade.MaterialBindingAPI):
                    binding_api = UsdShade.MaterialBindingAPI(prim)
                    direct_binding = binding_api.GetDirectBinding()
                    direct_mat_path = (
                        direct_binding.GetMaterialPath() if direct_binding else None
                    )
                    if direct_mat_path:
                        listener.debug(
                            f"Skipping {prim_path} (has direct material binding)"
                        )
                        skipped_materials_count += 1
                        continue

            # Extract metadata if requested
            if extract_metadata:
                prim_info["metadata"] = self._extract_prim_metadata(prim)

            # Extract display color if requested
            if extract_display_color:
                display_color = self._extract_display_color(prim, listener)
                if display_color is not None:
                    prim_info["display_color"] = display_color
                    listener.debug(
                        f"Extracted display color for {prim_path}: {display_color}"
                    )
                else:
                    listener.debug(f"No display color found for {prim_path}")

            # Extract material bindings if requested
            if extract_material_bindings:
                material_bindings = self._extract_material_bindings(
                    prim, stage, listener
                )
                if material_bindings:
                    prim_info["material_bindings"] = material_bindings

            # Extract hierarchy info if requested
            if extract_hierarchy:
                hierarchy_info = self._extract_hierarchy_info(prim, usd_model)
                if hierarchy_info:
                    prim_info["hierarchy"] = hierarchy_info

            # Extract world-space bounding box
            assembly_members = rendering_config.assembly_target_members.get(prim_path)
            if assembly_members is not None:
                prim_world_bbox = _explicit_members_world_bbox(
                    stage,
                    assembly_members,
                    rendering_config.bbox_purposes,
                    time_code=Usd.TimeCode(0),
                    use_extents_hint=False,
                )
            elif prim_path in rendering_config.recovered_guide_gprim_targets:
                purpose = str(UsdGeom.Imageable(prim).ComputePurpose())
                prim_world_bbox = get_world_bbox_from_prim(
                    prim,
                    tuple(dict.fromkeys((*rendering_config.bbox_purposes, purpose))),
                    time_code=Usd.TimeCode(0),
                    use_extents_hint=False,
                )
            else:
                prim_world_bbox = get_world_bbox_from_prim(
                    prim,
                    rendering_config.bbox_purposes,
                )
            if prim_world_bbox:
                prim_info["world_bbox"] = prim_world_bbox

                # Add MPU-scaled world bbox (in meters)
                if meters_per_unit:
                    prim_world_bbox_meters = scale_bbox_by_mpu(
                        prim_world_bbox, meters_per_unit
                    )
                    if prim_world_bbox_meters:
                        prim_info["world_bbox_meters"] = prim_world_bbox_meters

                # Add relative metrics compared to stage bbox
                if stage_world_bbox:
                    relative_metrics = compute_relative_metrics(
                        prim_world_bbox, stage_world_bbox
                    )
                    if relative_metrics:
                        prim_info["relative_metrics"] = relative_metrics

            # Check if we should skip this prim based on existing files
            if skip_existing:
                rgb_modes_for_skip = context.get("rgb_rendering_modes", rendering_modes)
                sensor_modes_for_skip = context.get("sensor_rendering_modes", [])

                if self._check_prim_files_exist(
                    prim_path,
                    render_output_dir,
                    rendering_config,
                    rgb_modes_for_skip,
                    sensor_modes_for_skip,
                ):
                    self._record_existing_files(
                        prim_info,
                        prim_path,
                        render_output_dir,
                        rendering_config,
                        output_dir,
                        rgb_modes_for_skip,
                        sensor_modes_for_skip,
                    )
                    total_images += len(prim_info["images"])
                    skipped_prims.append((prim_path, prim_info))
                    continue

            prims_to_render.append(prim_path)
            prim_data.append(prim_info)

        # Add skipped prims to prim_data
        for _, prim_info in skipped_prims:
            prim_data.append(prim_info)

        # Log summary
        if skip_existing_materials and skipped_materials_count > 0:
            listener.info(
                f"Rendering {len(prims_to_render)} prims"
                f" (excluded {skipped_materials_count} with direct materials)"
            )
        else:
            listener.info(
                f"Rendering {len(prims_to_render)} prims"
                f" (skipped {len(skipped_prims)} existing)"
            )

        return prims_to_render, prim_data, total_images

    def _prepare_stages(
        self,
        stage,
        prims_to_render: list[str],
        rgb_modes: list[str],
        rendering_config,
        context: dict[str, Any],
        listener,
    ) -> dict[str, dict[str, Any]]:
        """Pass 1: Prepare stages for each rendering mode.

        Returns:
            prepared_stages dict mapping render_mode -> stage info
        """
        sensor_modes = context.get("sensor_rendering_modes", [])

        listener.info(
            f"Pass 1: Preparing stages for {len(prims_to_render)} prims "
            f"with {len(rgb_modes)} RGB modes and {len(sensor_modes)} sensor modes"
        )

        prepared_stages: dict[str, dict[str, Any]] = {}

        for render_mode in rgb_modes:
            base_mode = rendering_config.get_base_mode(render_mode)
            listener.info(
                f"  Preparing stages for mode: {render_mode}"
                + (f" (base: {base_mode})" if base_mode != render_mode else "")
            )
            mode_config = rendering_config

            # Apply original materials override if enabled for this mode
            if rendering_config.should_use_original_materials_for_mode(render_mode):
                mode_config = replace(
                    mode_config,
                    should_assign_random_colors=False,
                    should_reset_materials=False,
                )

            start_time = time.time()

            try:
                if base_mode == "composition":
                    listener.info(
                        f"    Duplicating stages and preparing for {len(prims_to_render)} prims..."
                    )
                    prepared_data = prepare_prims_with_composition(
                        stage,
                        prims_to_render,
                        config=mode_config,
                        render_mode=render_mode,
                    )
                    prepared_stages[render_mode] = {
                        "type": "composition",
                        "data": prepared_data,
                        "config": mode_config,
                    }
                elif base_mode == "prim_only":
                    listener.info(
                        f"    Duplicating stage for {len(prims_to_render)} prims..."
                    )
                    prim_only_config = replace(
                        mode_config,
                        should_render_prim_only=True,
                        should_highlight_prim=False,
                        enable_contour=False,
                        enable_bbox=False,
                    )
                    prim_only_stage = duplicate_stage(stage)
                    listener.info("    Stage duplicated, preparing render prims...")
                    prepared_data = prepare_render_prims(
                        prim_only_stage,
                        prims_to_render,
                        config=prim_only_config,
                        render_mode=render_mode,
                    )
                    prepared_stages[render_mode] = {
                        "type": "prim_only",
                        "data": prepared_data,
                        "config": prim_only_config,
                    }
                elif base_mode == "prim_with_stage":
                    listener.info(
                        f"    Duplicating stage for {len(prims_to_render)} prims..."
                    )
                    prim_with_stage_config = replace(
                        mode_config,
                        should_render_prim_only=False,
                        should_highlight_prim=True,
                        enable_contour=False,
                        enable_bbox=False,
                        camera_prim_focus_margin=mode_config.camera_prim_with_stage_margin,
                    )
                    prim_with_stage_stage = duplicate_stage(stage)
                    listener.info("    Stage duplicated, preparing render prims...")
                    prepared_data = prepare_render_prims(
                        prim_with_stage_stage,
                        prims_to_render,
                        config=prim_with_stage_config,
                        render_mode=render_mode,
                    )
                    prepared_stages[render_mode] = {
                        "type": "prim_with_stage",
                        "data": prepared_data,
                        "config": prim_with_stage_config,
                    }
                elapsed = time.time() - start_time
                listener.info(
                    f"    ✓ Prepared stage for {render_mode} in {elapsed:.1f}s"
                )
            except Exception as e:
                listener.error(f"  Failed to prepare {render_mode}: {e}", exc_info=True)
                # Continue with other modes

        listener.info(
            f"Pass 1 complete: Prepared {len(prepared_stages)} stage configurations"
        )

        return prepared_stages

    def _upload_stages_to_s3(
        self,
        prepared_stages: dict[str, dict[str, Any]],
        rendering_backend,
        context: dict[str, Any],
        listener,
    ) -> list[tuple[str, str]]:
        """Pass 1.5: Prepare reusable stage URLs for REST rendering.

        Mutates prepared_stages to add URLs (stage_url, highlight_url, plain_url).
        Depending on renderer transfer config, those URLs may be data URIs or
        S3-backed HTTPS URLs.

        Returns:
            s3_cleanup_uris: List of (s3_uri, s3_profile) tuples for cleanup
        """
        s3_cleanup_uris: list[tuple[str, str]] = []
        if not isinstance(rendering_backend, RemoteRenderingBackend):
            listener.info(
                "Non-remote backend detected, skipping reusable URL optimization"
            )
            return s3_cleanup_uris

        from world_understanding.functions.graphics.render_remote import (
            export_stage_to_s3,
        )

        listener.info(
            "Preparing reusable stage URLs for remote batch rendering "
            f"(bundle_mdl_assets={rendering_backend.bundle_mdl_assets})"
        )

        # Derive base_dir from usd_path so the bundler can
        # resolve relative texture references
        usd_path_val = context.get("usd_path")
        stage_base_dir = Path(str(usd_path_val)).parent if usd_path_val else None

        for render_mode, stage_info in prepared_stages.items():
            try:
                if stage_info["type"] == "composition":
                    # Upload both highlight and plain stages
                    (
                        (highlight_stage, _, _),
                        (plain_stage, _, _),
                    ) = stage_info["data"]

                    listener.info(f"  Uploading {render_mode} highlight stage...")
                    highlight_url, highlight_s3_uri = export_stage_to_s3(
                        highlight_stage,
                        s3_bucket=rendering_backend.s3_bucket,
                        s3_region=rendering_backend.s3_region,
                        s3_profile=rendering_backend.s3_profile,
                        base_dir=stage_base_dir,
                        use_data_uri=rendering_backend.use_data_uri,
                        bundle_mdl_assets=rendering_backend.bundle_mdl_assets,
                    )
                    stage_info["highlight_url"] = highlight_url
                    if highlight_s3_uri:
                        s3_cleanup_uris.append(
                            (highlight_s3_uri, rendering_backend.s3_profile)
                        )

                    listener.info(f"  Uploading {render_mode} plain stage...")
                    plain_url, plain_s3_uri = export_stage_to_s3(
                        plain_stage,
                        s3_bucket=rendering_backend.s3_bucket,
                        s3_region=rendering_backend.s3_region,
                        s3_profile=rendering_backend.s3_profile,
                        base_dir=stage_base_dir,
                        use_data_uri=rendering_backend.use_data_uri,
                        bundle_mdl_assets=rendering_backend.bundle_mdl_assets,
                    )
                    stage_info["plain_url"] = plain_url
                    if plain_s3_uri:
                        s3_cleanup_uris.append(
                            (plain_s3_uri, rendering_backend.s3_profile)
                        )

                    listener.info(
                        f"  ✓ Uploaded {render_mode} stages (highlight + plain)"
                    )

                else:
                    # Upload single stage (prim_only, prim_with_stage)
                    prepared_stage, _, _ = stage_info["data"]

                    listener.info(f"  Uploading {render_mode} stage...")
                    stage_url, stage_s3_uri = export_stage_to_s3(
                        prepared_stage,
                        s3_bucket=rendering_backend.s3_bucket,
                        s3_region=rendering_backend.s3_region,
                        s3_profile=rendering_backend.s3_profile,
                        base_dir=stage_base_dir,
                        use_data_uri=rendering_backend.use_data_uri,
                        bundle_mdl_assets=rendering_backend.bundle_mdl_assets,
                    )
                    stage_info["stage_url"] = stage_url
                    if stage_s3_uri:
                        s3_cleanup_uris.append(
                            (stage_s3_uri, rendering_backend.s3_profile)
                        )

                    listener.info(f"  ✓ Uploaded {render_mode} stage")

            except Exception as e:
                listener.error(
                    f"  Failed to upload {render_mode} to S3: {e}", exc_info=True
                )
                # Continue with other modes

        listener.info(
            "Remote stage URL preparation complete: "
            f"{len(s3_cleanup_uris)} S3 object(s) require cleanup"
        )

        return s3_cleanup_uris

    def _cleanup_s3(
        self,
        s3_cleanup_uris: list[tuple[str, str]],
        listener,
    ) -> None:
        """Clean up S3 files uploaded for rendering."""
        if not s3_cleanup_uris:
            return
        listener.info(
            f"Cleaning up {len(s3_cleanup_uris)} S3 file(s) uploaded for rendering"
        )
        for s3_uri, s3_profile in s3_cleanup_uris:
            try:
                delete_s3_path(s3_uri, profile_name=s3_profile)
                listener.debug(f"  ✓ Deleted {s3_uri}")
            except Exception as e:
                listener.warning(f"  Failed to clean up S3 file {s3_uri}: {e}")
        listener.info("S3 cleanup complete")

    # ------------------------------------------------------------------
    # Async methods
    # ------------------------------------------------------------------

    async def arun(
        self, context: dict[str, Any], object_store: ObjectStore
    ) -> dict[str, Any]:
        """Async version of run() with true async rendering for NVCF backends.

        Uses asyncio.as_completed() for concurrent batch rendering and
        asyncio.to_thread() for CPU-bound operations (stage prep, image processing).
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Get inputs
        stage = object_store.get("usd_stage")
        if not stage:
            raise ValueError("USD stage not found in object store")

        rendering_backend = object_store.get("rendering_backend")
        rendering_config = object_store.get("rendering_config")
        if not rendering_backend:
            raise ValueError("Rendering backend not found in object store")
        if not rendering_config:
            raise ValueError("Rendering config not found in object store")

        prim_filters = context.get("prim_filters", {})
        render_output_dir = Path(context.get("render_output_dir", "output/renders"))
        batch_size = context.get("batch_size", 64)
        rendering_modes = context.get(
            "rendering_modes", ["prim_with_stage", "prim_only"]
        )

        output_dir = context.get("output_dir")
        if output_dir is None:
            output_dir = render_output_dir.parent
            listener.info(
                f"Using render_output_dir.parent as base "
                f"for relative paths: {output_dir}"
            )
        else:
            output_dir = Path(output_dir)

        max_concurrent = _validate_positive_int_config(
            "max_concurrent_requests",
            context.get("max_concurrent_requests", 128),
        )

        rendering_config = self._propagate_root_prim(prim_filters, rendering_config)
        rendering_config = self._propagate_bbox_purposes(
            prim_filters,
            rendering_config,
        )

        listener.info("Starting USD prim traversal and rendering (async)")
        listener.info(f"  Filters: {prim_filters}")
        listener.info(f"  Camera type: {rendering_config.camera_view_type.value}")
        listener.info(f"  Number of views: {len(rendering_config.camera_ordering)}")
        listener.info(f"  Max batch size: {batch_size} (adaptive)")
        listener.info(f"  Max concurrent requests: {max_concurrent}")
        listener.info(f"  Rendering modes: {rendering_modes}")

        # Extract stage-level metrics
        stage_up_axis = str(UsdGeom.GetStageUpAxis(stage))
        listener.info(f"Stage up axis: {stage_up_axis}")
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        listener.info(f"Stage meters per unit: {meters_per_unit}")

        stage_world_bbox = get_stage_world_bbox(
            stage,
            rendering_config.bbox_purposes,
        )
        stage_world_bbox_meters = (
            scale_bbox_by_mpu(stage_world_bbox, meters_per_unit)
            if stage_world_bbox
            else None
        )

        if stage_world_bbox:
            listener.info(f"Stage world bbox size: {stage_world_bbox['size']}")
            if stage_world_bbox_meters:
                listener.info(
                    f"Stage world bbox size (meters): {stage_world_bbox_meters['size']}"
                )

        object_store.set("stage_up_axis", stage_up_axis)
        object_store.set("meters_per_unit", meters_per_unit)
        object_store.set("stage_world_bbox", stage_world_bbox)
        object_store.set("stage_world_bbox_meters", stage_world_bbox_meters)

        # Collect and filter prims (CPU-bound, run in thread)
        prim_filter_skips: list[dict[str, Any]] = []
        assembly_target_members: dict[str, list[str]] = {}
        recovered_guide_gprim_targets: set[str] = set()
        all_prims = await asyncio.to_thread(
            self._collect_prims,
            stage,
            prim_filters,
            listener,
            prim_filter_skips,
            assembly_target_members,
            recovered_guide_gprim_targets,
        )
        rendering_config = replace(
            rendering_config,
            assembly_target_members={
                target: tuple(member_paths)
                for target, member_paths in assembly_target_members.items()
            },
            recovered_guide_gprim_targets=tuple(sorted(recovered_guide_gprim_targets)),
        )
        context["assembly_render_targets"] = {
            target: member_paths.copy()
            for target, member_paths in assembly_target_members.items()
        }
        context["recovered_guide_gprim_targets"] = sorted(recovered_guide_gprim_targets)
        context["prim_filter_diagnostics"] = _summarize_prim_filter_diagnostics(
            prim_filter_skips
        )
        listener.info(f"Found {len(all_prims)} USD prims total")
        self._require_matching_prims(
            stage,
            all_prims,
            prim_filters,
            context["prim_filter_diagnostics"],
        )

        prims_to_render, prim_data, total_images = await asyncio.to_thread(
            self._collect_and_filter_prims,
            stage,
            all_prims,
            prim_filters,
            rendering_config,
            rendering_modes,
            context,
            object_store,
            render_output_dir,
            output_dir,
            listener,
        )

        # Pass 1: Prepare stages (CPU-bound)
        rgb_modes = context.get("rgb_rendering_modes", rendering_modes)
        sensor_modes = context.get("sensor_rendering_modes", [])
        image_height = context.get("image_height", 512)

        prepared_stages = await asyncio.to_thread(
            self._prepare_stages,
            stage,
            prims_to_render,
            rgb_modes,
            rendering_config,
            context,
            listener,
        )

        # Pass 1.5: Upload stages to S3 (async for NVCF)
        s3_cleanup_uris = await self._upload_stages_to_s3_async(
            prepared_stages,
            rendering_backend,
            context,
            listener,
        )

        # Pass 2: Render in batches (async)
        # Adaptive batch_size: for NVCF (cloud API), reduce batch size to create
        # enough concurrent tasks to saturate parallel API instances.
        # For local GPU backends (warp/ovrtx), use the configured batch_size
        # as-is since they render sequentially and benefit from large batches
        # (fewer USD exports + BVH rebuilds).
        max_batch_size = batch_size
        num_prims = len(prims_to_render)
        num_modes = len(rgb_modes)
        if isinstance(rendering_backend, RemoteRenderingBackend):
            min_tasks = min(max_concurrent, 64)
            if num_prims > 0 and num_modes > 0:
                adaptive = max(4, (num_prims * num_modes) // min_tasks)
                batch_size = min(max_batch_size, adaptive)

        num_batches = (num_prims + batch_size - 1) // batch_size
        listener.info(
            f"Pass 2 (async): Rendering {num_prims} prims in "
            f"{num_batches} batches (batch_size={batch_size}, "
            f"max={max_batch_size}, adaptive)"
        )

        prim_data_dict = {p["prim_path"]: p for p in prim_data}

        batch_mode_tasks = []
        for batch_start in range(0, len(prims_to_render), batch_size):
            batch_end = min(batch_start + batch_size, len(prims_to_render))
            for render_mode in rgb_modes:
                batch_mode_tasks.append((batch_start, batch_end, render_mode))

        num_total_tasks = len(batch_mode_tasks)
        skipped_count = len(prim_data) - len(prims_to_render)

        # Local GPU backends (warp/ovrtx) must serialise rendering — their
        # CUDA contexts are not thread-safe and concurrent asyncio.to_thread
        # calls deadlock or segfault. Remote REST services can benefit from
        # concurrent requests.
        is_remote = isinstance(rendering_backend, RemoteRenderingBackend)
        if is_remote:
            from world_understanding.functions.graphics.render_remote_async import (
                get_global_remote_render_limit,
            )

            global_limit = get_global_remote_render_limit()
            if global_limit is not None:
                listener.info(
                    f"  Global remote render request cap: {global_limit} (process-wide)"
                )
        effective_concurrent = max_concurrent if is_remote else 1
        if effective_concurrent < 1:
            raise ValueError("max_concurrent_requests must be a positive integer")
        semaphore = asyncio.Semaphore(effective_concurrent)

        try:
            batch_coros = [
                self._process_batch_async_with_retry_split(
                    batch_start,
                    batch_end,
                    prims_to_render,
                    prim_data_dict,
                    prepared_stages,
                    rendering_backend,
                    render_mode,
                    render_output_dir,
                    output_dir,
                    num_total_tasks,
                    batch_size,
                    listener,
                    sensor_modes,
                    image_height,
                    semaphore,
                )
                for batch_start, batch_end, render_mode in batch_mode_tasks
            ]

            all_prim_images_data: dict[str, list[dict[str, Any]]] = {}
            completed_tasks = 0
            for coro in asyncio.as_completed(batch_coros):
                (
                    batch_start,
                    batch_end,
                    render_mode,
                    batch_images,
                    batch_failures,
                    batch_prim_images,
                ) = await coro
                total_images += batch_images
                completed_tasks += 1

                try:
                    listener.event(
                        "rendering.progress",
                        {
                            "task_name": "USDPrimTraversalAndRendering",
                            "current": completed_tasks,
                            "total": num_total_tasks,
                            "percent": int((completed_tasks / num_total_tasks) * 100),
                            "message": (
                                f"Rendered batch {completed_tasks}/{num_total_tasks} "
                                f"({batch_images} images, mode: {render_mode})"
                            ),
                            "images_rendered": total_images,
                            "render_mode": render_mode,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Failed to emit progress event: {e}")

                for prim_path, images in batch_prim_images.items():
                    if prim_path not in all_prim_images_data:
                        all_prim_images_data[prim_path] = []
                    all_prim_images_data[prim_path].extend(images)

                if batch_failures:
                    if "failed_batches" not in context:
                        context["failed_batches"] = []
                    context["failed_batches"].extend(batch_failures)

            # Merge image data back into prim_data_dict
            for prim_path, images in all_prim_images_data.items():
                if prim_path in prim_data_dict:
                    prim_data_dict[prim_path]["images"].extend(images)

        finally:
            self._cleanup_s3(s3_cleanup_uris, listener)

        # Post-processing
        prim_data = list(prim_data_dict.values())

        self._check_blank_dataset_renders(
            prim_data,
            output_dir,
            rgb_modes=rgb_modes,
            sensor_modes=sensor_modes,
            listener=listener,
            context=context,
        )
        # Fail early if rendering produced no images, after blank-render
        # post-processing has had a chance to populate diagnostic context.
        if total_images == 0:
            raise RuntimeError(_zero_image_render_error_message(rendering_backend))
        missing_image_prims = [
            prim_path
            for prim_path in prims_to_render
            if not prim_data_dict.get(prim_path, {}).get("images")
        ]
        if missing_image_prims:
            context["missing_image_prims"] = missing_image_prims
            sample = ", ".join(missing_image_prims[:5])
            message = (
                f"Rendering produced no images for {len(missing_image_prims)} of "
                f"{len(prims_to_render)} selected prims. Sample: {sample}"
            )
            if _is_config_enabled(context.get("fail_on_missing_prim_images"), True):
                raise RuntimeError(message)
            listener.warning(message)

        context["rendered_prims"] = prims_to_render
        context["prim_data"] = prim_data
        context["total_images_rendered"] = total_images
        context["output_dir"] = str(output_dir)

        object_store.set("prim_data", prim_data)

        listener.info("USD rendering complete:")
        listener.info(f"  Prims rendered: {len(prims_to_render)}")
        listener.info(f"  Total images: {total_images}")

        try:
            listener.event(
                "rendering.all_completed",
                {
                    "total_prims": len(prims_to_render),
                    "total_images": total_images,
                    "skipped_prims": skipped_count,
                    "rendering_modes": rendering_modes,
                    "num_views": len(rendering_config.camera_ordering),
                    "failed_batches": len(context.get("failed_batches", [])),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to emit overall rendering event: {e}")

        return context

    async def _upload_stages_to_s3_async(
        self,
        prepared_stages: dict[str, dict[str, Any]],
        rendering_backend,
        context: dict[str, Any],
        listener,
    ) -> list[tuple[str, str]]:
        """Pass 1.5 async: Upload prepared stages to S3 for NVCF in parallel.

        Wraps each S3 upload in asyncio.to_thread() and uses asyncio.gather()
        for parallel uploads.

        Returns:
            s3_cleanup_uris: List of (s3_uri, s3_profile) tuples for cleanup
        """
        if not isinstance(rendering_backend, RemoteRenderingBackend):
            listener.info(
                "Non-remote backend detected, skipping reusable URL optimization"
            )
            return []

        from world_understanding.functions.graphics.render_remote import (
            export_stage_to_s3,
        )

        listener.info(
            "Preparing reusable stage URLs for remote batch rendering (async) "
            f"(bundle_mdl_assets={rendering_backend.bundle_mdl_assets})"
        )

        usd_path_val = context.get("usd_path")
        stage_base_dir = Path(str(usd_path_val)).parent if usd_path_val else None

        s3_cleanup_uris: list[tuple[str, str]] = []

        async def _upload_single_stage(
            render_mode: str,
            stage_to_upload: Any,
            label: str,
        ) -> tuple[str, str | None]:
            """Upload a single stage and return (url, s3_uri)."""
            listener.info(f"  Uploading {render_mode} {label} stage...")
            url, s3_uri = await asyncio.to_thread(
                export_stage_to_s3,
                stage_to_upload,
                s3_bucket=rendering_backend.s3_bucket,
                s3_region=rendering_backend.s3_region,
                s3_profile=rendering_backend.s3_profile,
                base_dir=stage_base_dir,
                use_data_uri=rendering_backend.use_data_uri,
                bundle_mdl_assets=rendering_backend.bundle_mdl_assets,
            )
            listener.info(f"  ✓ Uploaded {render_mode} {label} stage")
            return url, s3_uri

        upload_tasks = []
        # Build list of (coroutine, render_mode, key_to_set, label)
        upload_meta: list[tuple[str, str, str]] = []

        for render_mode, stage_info in prepared_stages.items():
            try:
                if stage_info["type"] == "composition":
                    (
                        (highlight_stage, _, _),
                        (plain_stage, _, _),
                    ) = stage_info["data"]

                    upload_tasks.append(
                        _upload_single_stage(render_mode, highlight_stage, "highlight")
                    )
                    upload_meta.append((render_mode, "highlight_url", "highlight"))

                    upload_tasks.append(
                        _upload_single_stage(render_mode, plain_stage, "plain")
                    )
                    upload_meta.append((render_mode, "plain_url", "plain"))
                else:
                    prepared_stage, _, _ = stage_info["data"]
                    upload_tasks.append(
                        _upload_single_stage(render_mode, prepared_stage, "")
                    )
                    upload_meta.append((render_mode, "stage_url", ""))
            except Exception as e:
                listener.error(
                    f"  Failed to prepare upload for {render_mode}: {e}",
                    exc_info=True,
                )

        # Run all uploads in parallel
        results = await asyncio.gather(*upload_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            render_mode, url_key, label = upload_meta[i]
            if isinstance(result, BaseException):
                listener.error(
                    f"  Failed to upload {render_mode} {label} to S3: {result}",
                    exc_info=True,
                )
                continue
            url, s3_uri = result  # type: ignore[misc]
            prepared_stages[render_mode][url_key] = url
            if s3_uri:
                s3_cleanup_uris.append((s3_uri, rendering_backend.s3_profile))

        listener.info(
            "Remote stage URL preparation complete: "
            f"{len(s3_cleanup_uris)} S3 object(s) require cleanup"
        )
        return s3_cleanup_uris

    async def _process_batch_async(
        self,
        batch_start: int,
        batch_end: int,
        prims_to_render: list[str],
        prim_data: dict[str, dict[str, Any]],
        prepared_stages: dict[str, dict[str, Any]],
        rendering_backend,
        render_mode: str,
        render_output_dir: Path,
        output_dir: Path,
        num_total_tasks: int,
        batch_size: int,
        listener,
        sensor_modes: list[str] | None = None,
        image_height: int = 512,
        semaphore: asyncio.Semaphore | None = None,
    ) -> tuple[
        int, int, str, int, list[dict[str, Any]], dict[str, list[dict[str, Any]]]
    ]:
        """Async version of _process_batch.

        For remote REST backends with precomputed URLs, uses async rendering
        functions. For non-remote backends, falls back to
        asyncio.to_thread(self._process_batch).

        Returns:
            (batch_start, batch_end, render_mode, total_images, failed_batches, prim_images_data)
        """
        # Check if this is a remote REST backend with precomputed URLs.
        is_remote = isinstance(rendering_backend, RemoteRenderingBackend)

        if not is_remote:
            # Fall back to sync _process_batch in a thread.
            # Use the semaphore to serialise access for GPU backends (warp/ovrtx)
            # whose CUDA contexts are not thread-safe — concurrent
            # asyncio.to_thread calls would otherwise deadlock or segfault.
            if semaphore is None:
                semaphore = asyncio.Semaphore(1)
            async with semaphore:
                sync_images, sync_failures, sync_prim_images = await asyncio.to_thread(
                    self._process_batch,
                    batch_start,
                    batch_end,
                    prims_to_render,
                    prim_data,
                    prepared_stages,
                    rendering_backend,
                    render_mode,
                    render_output_dir,
                    output_dir,
                    num_total_tasks,
                    batch_size,
                    listener,
                    sensor_modes,
                    image_height,
                )
            return (
                batch_start,
                batch_end,
                render_mode,
                sync_images,
                sync_failures,
                sync_prim_images,
            )

        # Remote REST async path.
        from world_understanding.functions.graphics.render_remote_async import (
            render_cameras_from_url,
            render_composition_from_url,
            save_images_parallel,
        )
        from world_understanding.functions.graphics.rendering import (
            paste_on_background,
        )
        from world_understanding.utils.image_utils import is_prim_visible_in_image

        batch_prims = prims_to_render[batch_start:batch_end]
        total_images = 0
        failed_batches: list[dict[str, Any]] = []
        prim_images_data: dict[str, list[dict[str, Any]]] = {}

        batch_num = batch_start // batch_size + 1
        listener.info(
            f"Processing batch {batch_num}, mode '{render_mode}' (async): "
            f"prims {batch_start + 1}-{batch_end}"
        )
        if semaphore is None:
            semaphore = asyncio.Semaphore(1)

        try:
            if render_mode not in prepared_stages:
                listener.warning(f"No prepared stage for mode {render_mode}, skipping")
                return (
                    batch_start,
                    batch_end,
                    render_mode,
                    total_images,
                    failed_batches,
                    prim_images_data,
                )

            prepared_info = prepared_stages[render_mode]
            config = prepared_info["config"]

            # Calculate batch frame range
            frame_range = (batch_start, batch_end - 1)
            start_frame, end_frame = frame_range
            frames_str = (
                f"{start_frame}:{end_frame}"
                if end_frame > start_frame
                else str(start_frame)
            )

            if prepared_info["type"] == "composition":
                # Composition mode - two stages
                (
                    (_, highlight_cameras, _),
                    (_, _, _),
                ) = prepared_info["data"]

                highlight_url = prepared_info.get("highlight_url")
                plain_url = prepared_info.get("plain_url")

                if not highlight_url or not plain_url:
                    # No URLs available, fall back to sync
                    async with semaphore:
                        (
                            total_images,
                            failed_batches,
                            prim_images_data,
                        ) = await asyncio.to_thread(
                            self._process_batch,
                            batch_start,
                            batch_end,
                            prims_to_render,
                            prim_data,
                            prepared_stages,
                            rendering_backend,
                            render_mode,
                            render_output_dir,
                            output_dir,
                            num_total_tasks,
                            batch_size,
                            listener,
                            sensor_modes,
                            image_height,
                        )
                    return (
                        batch_start,
                        batch_end,
                        render_mode,
                        total_images,
                        failed_batches,
                        prim_images_data,
                    )

                # Render both stages concurrently
                highlight_result, plain_result = await render_composition_from_url(
                    highlight_url=highlight_url,
                    plain_url=plain_url,
                    cameras=highlight_cameras,
                    image_width=config.image_width,
                    image_height=image_height,
                    frames=frames_str,
                    api_key=rendering_backend.api_key,
                    base_url=rendering_backend.base_url,
                    timeout=rendering_backend.timeout,
                    max_retries=rendering_backend.max_retries,
                    retry_delay=rendering_backend.retry_delay,
                    retry_backoff_factor=rendering_backend.retry_backoff_factor,
                    retry_jitter=rendering_backend.retry_jitter,
                    sensors=sensor_modes if sensor_modes else None,
                    apply_background_mask=config.use_background_color,
                    semaphore=semaphore,
                    material_target=getattr(rendering_backend, "material_target", None),
                )

                # Post-process composition results (CPU-bound, run in thread)
                def _postprocess_composition():
                    from world_understanding.utils.image_utils import (
                        draw_bounding_box_on_red,
                        extract_non_black_outline,
                        extract_red_outline,
                        paste_outline_to_image,
                    )

                    # Process highlight results
                    for i, result in enumerate(highlight_result["results"]):
                        prim_to_images = {}
                        for j, image in enumerate(result["images"]):
                            prim_idx = batch_start + j
                            if prim_idx < len(prims_to_render):
                                prim_path = prims_to_render[prim_idx]
                                if config.use_background_color:
                                    image = paste_on_background(
                                        image, config.background_color
                                    )
                                    result["images"][j] = image
                                prim_to_images[prim_path] = image
                        highlight_result["results"][i]["prim_to_images"] = (
                            prim_to_images
                        )

                    # Process plain results
                    for i, result in enumerate(plain_result["results"]):
                        prim_to_images = {}
                        for j, image in enumerate(result["images"]):
                            prim_idx = batch_start + j
                            if prim_idx < len(prims_to_render):
                                prim_path = prims_to_render[prim_idx]
                                if config.use_background_color:
                                    image = paste_on_background(
                                        image, config.background_color
                                    )
                                    result["images"][j] = image
                                prim_to_images[prim_path] = image
                        plain_result["results"][i]["prim_to_images"] = prim_to_images

                    # Compose images
                    def compose_image(highlight_img, plain_img, cfg):
                        final_image = plain_img.copy()
                        if cfg.enable_contour:
                            if cfg.contour_method == "non_black":
                                outline_img = extract_non_black_outline(
                                    highlight_img,
                                    black_threshold=cfg.contour_black_threshold,
                                    thickness=3,
                                )
                            else:
                                outline_img = extract_red_outline(
                                    highlight_img, thickness=3
                                )
                            contour_color_255 = tuple(
                                int(c * 255) for c in cfg.contour_color
                            )
                            final_image = paste_outline_to_image(
                                final_image, outline_img, contour_color_255
                            )
                        if cfg.enable_bbox:
                            bbox_img = draw_bounding_box_on_red(
                                highlight_img, box_width=3
                            )
                            bbox_color_255 = tuple(int(c * 255) for c in cfg.bbox_color)
                            final_image = paste_outline_to_image(
                                final_image, bbox_img, bbox_color_255
                            )
                        return final_image

                    highlight_cameras_dict = {
                        r["camera"]: r for r in highlight_result["results"]
                    }
                    plain_cameras_dict = {
                        r["camera"]: r for r in plain_result["results"]
                    }

                    for camera_name in highlight_cameras_dict:
                        results_with_highlight = highlight_cameras_dict[camera_name]
                        results_plain = plain_cameras_dict.get(camera_name)
                        if not results_plain:
                            continue

                        if "prim_occlusion" not in results_with_highlight:
                            results_with_highlight["prim_occlusion"] = {}

                        for prim_path, image_with_highlight in results_with_highlight[
                            "prim_to_images"
                        ].items():
                            image_plain = results_plain["prim_to_images"].get(prim_path)
                            if image_plain is None:
                                continue

                            is_occluded = False
                            if config.skip_occluded_images:
                                is_visible = is_prim_visible_in_image(
                                    image_with_highlight,
                                    contour_method=config.contour_method,
                                    pixel_threshold=config.occlusion_pixel_threshold,
                                    black_threshold=config.contour_black_threshold,
                                )
                                is_occluded = not is_visible
                                results_with_highlight["prim_occlusion"][prim_path] = (
                                    is_occluded
                                )

                            if not is_occluded or not config.skip_occluded_images:
                                image_composition = compose_image(
                                    image_with_highlight, image_plain, config
                                )
                                results_with_highlight["prim_to_images"][prim_path] = (
                                    image_composition
                                )
                            else:
                                results_with_highlight["prim_to_images"][prim_path] = (
                                    None
                                )

                    return highlight_result

                result = await asyncio.to_thread(_postprocess_composition)

            elif prepared_info["type"] in ("prim_only", "prim_with_stage"):
                # Single-stage mode
                _, camera_paths, _ = prepared_info["data"]

                stage_url = prepared_info.get("stage_url")

                if not stage_url:
                    # No URL available, fall back to sync
                    async with semaphore:
                        (
                            total_images,
                            failed_batches,
                            prim_images_data,
                        ) = await asyncio.to_thread(
                            self._process_batch,
                            batch_start,
                            batch_end,
                            prims_to_render,
                            prim_data,
                            prepared_stages,
                            rendering_backend,
                            render_mode,
                            render_output_dir,
                            output_dir,
                            num_total_tasks,
                            batch_size,
                            listener,
                            sensor_modes,
                            image_height,
                        )
                    return (
                        batch_start,
                        batch_end,
                        render_mode,
                        total_images,
                        failed_batches,
                        prim_images_data,
                    )

                # Async NVCF rendering
                render_results = await render_cameras_from_url(
                    usd_url=stage_url,
                    cameras=camera_paths,
                    image_width=config.image_width,
                    image_height=image_height,
                    frames=frames_str,
                    api_key=rendering_backend.api_key,
                    base_url=rendering_backend.base_url,
                    timeout=rendering_backend.timeout,
                    max_retries=rendering_backend.max_retries,
                    retry_delay=rendering_backend.retry_delay,
                    retry_backoff_factor=rendering_backend.retry_backoff_factor,
                    retry_jitter=rendering_backend.retry_jitter,
                    sensors=sensor_modes if sensor_modes else None,
                    apply_background_mask=config.use_background_color,
                    semaphore=semaphore,
                    material_target=getattr(rendering_backend, "material_target", None),
                )

                # Post-process (CPU-bound, run in thread)
                def _postprocess_prims():
                    skip_occluded = (
                        config.should_skip_occluded_for_mode(render_mode)
                        if render_mode
                        else config.skip_occluded_images
                    )

                    for i, cam_result in enumerate(render_results["results"]):
                        prim_to_images = {}
                        prim_occlusion = {}

                        for j, image in enumerate(cam_result["images"]):
                            prim_idx = batch_start + j
                            if prim_idx >= len(prims_to_render):
                                continue
                            prim_path = prims_to_render[prim_idx]

                            if config.use_background_color:
                                image = paste_on_background(
                                    image, config.background_color
                                )
                                cam_result["images"][j] = image

                            is_occluded = False
                            if (
                                skip_occluded
                                and config.should_highlight_prim
                                and not config.should_render_prim_only
                            ):
                                is_visible = is_prim_visible_in_image(
                                    image,
                                    contour_method="red",
                                    pixel_threshold=config.occlusion_pixel_threshold,
                                )
                                is_occluded = not is_visible
                                prim_occlusion[prim_path] = is_occluded

                                if is_occluded:
                                    prim_to_images[prim_path] = None
                                else:
                                    prim_to_images[prim_path] = image
                            else:
                                prim_to_images[prim_path] = image

                        render_results["results"][i]["prim_to_images"] = prim_to_images
                        if prim_occlusion:
                            render_results["results"][i]["prim_occlusion"] = (
                                prim_occlusion
                            )

                    return render_results

                result = await asyncio.to_thread(_postprocess_prims)

            else:
                listener.warning(
                    f"Unknown prepared stage type: {prepared_info['type']}"
                )
                return (
                    batch_start,
                    batch_end,
                    render_mode,
                    total_images,
                    failed_batches,
                    prim_images_data,
                )

            failed_batches.extend(
                self._blank_render_failures_from_results(
                    result,
                    batch_start=batch_start,
                    batch_prims=batch_prims,
                    render_mode=render_mode,
                )
            )

            # Save images and collect metadata
            save_tasks: list[tuple[Any, Path]] = []

            for camera_result in result.get("results", []):
                camera_name = camera_result.get("camera", "default")
                prim_images = camera_result.get("prim_to_images", {})
                prim_occlusion = camera_result.get("prim_occlusion", {})
                blank_stats_by_prim = self._blank_render_frame_stats_by_prim(
                    camera_result,
                    batch_start=batch_start,
                    batch_prims=batch_prims,
                )

                for prim_path, image in prim_images.items():
                    if prim_path not in prim_data:
                        continue

                    if image is None:
                        is_occluded = prim_occlusion.get(prim_path, False)
                        if is_occluded:
                            listener.debug(
                                f"  Skipping occluded image for {prim_path} in view {camera_name}"
                            )
                        continue

                    prim_parts = prim_path.strip("/").split("/")
                    prim_name = prim_parts[-1] if prim_parts else "unnamed"
                    safe_prim_name = shorten_for_filesystem(
                        prim_name, max_len=MAX_FILENAME_STEM_LEN
                    )

                    if "_" in camera_name:
                        view_name = camera_name.split("_", 1)[-1]
                    else:
                        view_name = camera_name

                    if render_mode == "prim_only":
                        filename = f"{safe_prim_name}_{view_name}_prim_only.png"
                    elif render_mode == "prim_with_stage":
                        filename = f"{safe_prim_name}_{view_name}_prim_with_stage.png"
                    elif render_mode == "composition":
                        filename = f"{safe_prim_name}_{view_name}_composition.png"
                    else:
                        filename = f"{safe_prim_name}_{view_name}_{render_mode}.png"

                    filepath = prim_path_to_directory_structure(
                        prim_path, render_output_dir, filename
                    )

                    save_tasks.append((image, filepath))

                    try:
                        listener.event(
                            "rendering.completed",
                            {
                                "prim_path": prim_path,
                                "camera_view": view_name,
                                "render_mode": render_mode,
                                "output_path": str(filepath),
                                "camera": camera_name,
                            },
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to emit rendering event for {prim_path}: {e}"
                        )

                    if prim_path not in prim_images_data:
                        prim_images_data[prim_path] = []

                    image_info: dict[str, Any] = {
                        "view": view_name,
                        "path": str(filepath.relative_to(output_dir)),
                        "camera": camera_name,
                        "render_mode": render_mode,
                    }
                    if prim_path in blank_stats_by_prim:
                        image_info["blank_render"] = True
                        image_info["stats"] = blank_stats_by_prim[prim_path]
                    prim_images_data[prim_path].append(image_info)
                    total_images += 1

                # Save sensor data if present
                if "sensors" in camera_result and camera_result["sensors"]:
                    sensors_data = camera_result["sensors"]
                    for sensor_name, frame_data in sensors_data.items():
                        for frame_num, sensor_array in frame_data.items():
                            frame_idx = int(frame_num) - batch_start
                            if frame_idx < 0 or frame_idx >= len(batch_prims):
                                continue

                            prim_path = batch_prims[frame_idx]
                            if prim_path not in prim_data:
                                continue

                            prim_parts = prim_path.strip("/").split("/")
                            prim_name = prim_parts[-1] if prim_parts else "unnamed"
                            safe_prim_name = shorten_for_filesystem(
                                prim_name, max_len=MAX_FILENAME_STEM_LEN
                            )

                            if "_" in camera_name:
                                view_name = camera_name.split("_", 1)[-1]
                            else:
                                view_name = camera_name

                            sensor_filename = (
                                f"{safe_prim_name}_{view_name}_{sensor_name}.png"
                            )
                            sensor_filepath = prim_path_to_directory_structure(
                                prim_path, render_output_dir, sensor_filename
                            )

                            # Process sensor data into PIL image
                            try:
                                sensor_img = self._process_sensor_array(
                                    sensor_array,
                                    sensor_name,
                                    config.image_width,
                                    image_height,
                                    prim_path,
                                    listener,
                                )
                                if sensor_img is None:
                                    continue

                                save_tasks.append((sensor_img, sensor_filepath))

                                if prim_path not in prim_images_data:
                                    prim_images_data[prim_path] = []

                                prim_images_data[prim_path].append(
                                    {
                                        "view": view_name,
                                        "path": str(
                                            sensor_filepath.relative_to(output_dir)
                                        ),
                                        "camera": camera_name,
                                        "render_mode": sensor_name,
                                    }
                                )
                                total_images += 1
                            except Exception as e:
                                listener.warning(
                                    f"Failed to process sensor {sensor_name} for {prim_path}: {e}"
                                )

            # Save all images in parallel
            if save_tasks:
                await asyncio.to_thread(save_images_parallel, save_tasks)

        except Exception as e:
            listener.error(
                f"Failed to render batch starting at {batch_start}: {e}",
                exc_info=True,
            )
            failed_batches.append(
                {
                    "batch_start": batch_start,
                    "batch_prims": batch_prims,
                    "error": str(e),
                }
            )

        return (
            batch_start,
            batch_end,
            render_mode,
            total_images,
            failed_batches,
            prim_images_data,
        )

    async def _process_batch_async_with_retry_split(
        self,
        batch_start: int,
        batch_end: int,
        prims_to_render: list[str],
        prim_data: dict[str, dict[str, Any]],
        prepared_stages: dict[str, dict[str, Any]],
        rendering_backend,
        render_mode: str,
        render_output_dir: Path,
        output_dir: Path,
        num_total_tasks: int,
        batch_size: int,
        listener,
        sensor_modes: list[str] | None = None,
        image_height: int = 512,
        semaphore: asyncio.Semaphore | None = None,
    ) -> tuple[
        int, int, str, int, list[dict[str, Any]], dict[str, list[dict[str, Any]]]
    ]:
        """Render a batch, splitting zero-image remote failures into smaller ranges."""
        result = await self._process_batch_async(
            batch_start,
            batch_end,
            prims_to_render,
            prim_data,
            prepared_stages,
            rendering_backend,
            render_mode,
            render_output_dir,
            output_dir,
            num_total_tasks,
            batch_size,
            listener,
            sensor_modes,
            image_height,
            semaphore,
        )

        _, _, _, total_images, failed_batches, prim_images_data = result
        if (
            total_images > 0
            or batch_end - batch_start <= 1
            or not isinstance(rendering_backend, RemoteRenderingBackend)
        ):
            return result

        mid = batch_start + (batch_end - batch_start) // 2
        listener.warning(
            f"Batch {batch_start}-{batch_end} for mode '{render_mode}' "
            f"produced 0 images; retrying as {batch_start}-{mid} "
            f"and {mid}-{batch_end}"
        )

        left = await self._process_batch_async_with_retry_split(
            batch_start,
            mid,
            prims_to_render,
            prim_data,
            prepared_stages,
            rendering_backend,
            render_mode,
            render_output_dir,
            output_dir,
            num_total_tasks,
            batch_size,
            listener,
            sensor_modes,
            image_height,
            semaphore,
        )
        right = await self._process_batch_async_with_retry_split(
            mid,
            batch_end,
            prims_to_render,
            prim_data,
            prepared_stages,
            rendering_backend,
            render_mode,
            render_output_dir,
            output_dir,
            num_total_tasks,
            batch_size,
            listener,
            sensor_modes,
            image_height,
            semaphore,
        )

        combined_images = left[3] + right[3]
        combined_failures = left[4] + right[4]
        combined_prim_images: dict[str, list[dict[str, Any]]] = {}
        for batch_prim_images in (left[5], right[5]):
            for prim_path, images in batch_prim_images.items():
                combined_prim_images.setdefault(prim_path, []).extend(images)

        return (
            batch_start,
            batch_end,
            render_mode,
            combined_images,
            combined_failures,
            combined_prim_images,
        )

    def _process_sensor_array(
        self,
        sensor_array,
        sensor_name: str,
        image_width: int,
        image_height: int,
        prim_path: str,
        listener,
    ):
        """Process a raw sensor array into a PIL Image.

        Returns:
            PIL Image or None if processing fails.
        """
        import numpy as np
        from PIL import Image as PILImage

        try:
            if sensor_array.ndim == 1:
                array_size = sensor_array.size
                reshaped = False

                for num_channels in [1, 3, 4]:
                    pixels_with_channels = array_size // num_channels
                    sqrt_size = int(np.sqrt(pixels_with_channels))

                    if sqrt_size * sqrt_size * num_channels == array_size:
                        actual_height = actual_width = sqrt_size
                        if num_channels == 1:
                            sensor_array = sensor_array.reshape(
                                actual_height, actual_width
                            )
                        else:
                            sensor_array = sensor_array.reshape(
                                actual_height, actual_width, num_channels
                            )
                        reshaped = True
                        break

                if not reshaped:
                    expected_size = image_height * image_width
                    if array_size == expected_size:
                        sensor_array = sensor_array.reshape(image_height, image_width)
                        reshaped = True

                if not reshaped:
                    listener.warning(
                        f"Sensor {sensor_name} size {array_size} doesn't match any expected format. Skipping."
                    )
                    return None

            elif sensor_array.ndim >= 2:
                sensor_array = sensor_array.squeeze()
                if sensor_array.ndim not in [2, 3]:
                    listener.warning(
                        f"Sensor {sensor_name} has unexpected shape {sensor_array.shape}. Skipping."
                    )
                    return None
        except Exception as reshape_err:
            listener.warning(f"Failed to reshape sensor {sensor_name}: {reshape_err}")
            return None

        if sensor_name in ("depth", "linear_depth"):
            finite_mask = np.isfinite(sensor_array)
            if np.any(finite_mask):
                finite_values = sensor_array[finite_mask]
                data_min = finite_values.min()
                data_max = finite_values.max()

                sensor_array_clipped = sensor_array.copy()
                sensor_array_clipped[~finite_mask] = data_max * 1.1

                if data_max > data_min:
                    normalized = (
                        (sensor_array_clipped - data_min)
                        / (data_max - data_min)
                        * 255.0
                    )
                    normalized = np.clip(normalized, 0, 255)
                else:
                    normalized = np.zeros_like(sensor_array)
            else:
                normalized = np.zeros_like(sensor_array)

            return PILImage.fromarray(normalized.astype(np.uint8), mode="L")

        elif sensor_name == "instance_id_segmentation":
            if sensor_array.ndim == 2:
                unique_ids = np.unique(sensor_array)
                rgb_array = np.zeros((*sensor_array.shape, 3), dtype=np.uint8)
                for instance_id in unique_ids:
                    if instance_id == 0:
                        continue
                    mask = sensor_array == instance_id
                    r = int((instance_id * 67) % 256)
                    g = int((instance_id * 131) % 256)
                    b = int((instance_id * 197) % 256)
                    rgb_array[mask] = [r, g, b]
                return PILImage.fromarray(rgb_array, mode="RGB")
            elif sensor_array.ndim == 3:
                num_channels = sensor_array.shape[2]
                if num_channels == 3:
                    return PILImage.fromarray(sensor_array.astype(np.uint8), mode="RGB")
                elif num_channels == 4:
                    seg_img = PILImage.fromarray(
                        sensor_array.astype(np.uint8), mode="RGBA"
                    )
                    return seg_img.convert("RGB")
                else:
                    listener.warning(
                        f"Segmentation has {num_channels} channels, expected 1, 3, or 4. Skipping."
                    )
                    return None
            else:
                listener.warning(
                    f"Segmentation has unexpected shape {sensor_array.shape}. Skipping."
                )
                return None
        else:
            try:
                return PILImage.fromarray(sensor_array.squeeze().astype(np.uint8))
            except Exception as sensor_err:
                listener.warning(
                    f"Failed to save sensor {sensor_name} for {prim_path}: {sensor_err}"
                )
                return None

    def _process_batch(
        self,
        batch_start: int,
        batch_end: int,
        prims_to_render: list[str],
        prim_data: dict[str, dict[str, Any]],
        prepared_stages: dict[str, dict[str, Any]],
        rendering_backend,
        render_mode: str,
        render_output_dir: Path,
        output_dir: Path,
        num_total_tasks: int,
        batch_size: int,
        listener,
        sensor_modes: list[str] | None = None,
        image_height: int = 512,
    ) -> tuple[int, list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """Process a single batch of prims for rendering with a specific render mode.

        Args:
            batch_start: Starting index for the batch
            batch_end: Ending index for the batch (exclusive)
            prims_to_render: List of all prim paths to render
            prim_data: Dictionary to look up prim information (read-only)
            prepared_stages: Prepared stage configurations
            rendering_backend: Rendering backend instance
            render_mode: Single rendering mode to apply
            render_output_dir: Path for saving rendered images
            output_dir: Base output directory for relative paths
            num_total_tasks: Total number of (batch, mode) tasks
            batch_size: Size of each batch
            listener: Event listener for progress reporting
            sensor_modes: Optional list of sensor modes to render alongside RGB mode

        Returns:
            Tuple of (total_images_rendered, failed_batch_info_list, prim_images_data)
            where prim_images_data maps prim paths to lists of image info dicts
        """
        batch_prims = prims_to_render[batch_start:batch_end]
        total_images = 0
        failed_batches = []
        # Collect image data to return instead of modifying shared state
        prim_images_data = {}

        batch_num = batch_start // batch_size + 1
        listener.info(
            f"Processing batch {batch_num}, mode '{render_mode}': "
            f"prims {batch_start + 1}-{batch_end}"
        )

        try:
            if render_mode not in prepared_stages:
                listener.warning(f"No prepared stage for mode {render_mode}, skipping")
                return total_images, failed_batches, prim_images_data

            prepared_info = prepared_stages[render_mode]
            prepared_data = prepared_info["data"]
            config = prepared_info["config"]

            # Duplicate stages for thread safety
            # USD stages are not thread-safe, so each thread needs its own copy
            from world_understanding.utils.usd.stage import duplicate_stage

            # Render using the prepared stage data
            if prepared_info["type"] == "composition":
                # Unpack prepared composition data
                (
                    (highlight_stage_template, highlight_cameras, frames),
                    (
                        plain_stage_template,
                        plain_cameras,
                        _,
                    ),  # frames are always identical
                ) = prepared_data

                # Calculate batch frame range
                frame_range = (batch_start, batch_end - 1)

                # Get pre-uploaded URLs if available
                highlight_url = prepared_info.get("highlight_url")
                plain_url = prepared_info.get("plain_url")

                # Only duplicate if we DON'T have URLs
                if (
                    highlight_url
                    and plain_url
                    and isinstance(rendering_backend, RemoteRenderingBackend)
                ):
                    # Use URLs directly - no duplication needed!
                    highlight_stage = None
                    plain_stage = None
                else:
                    # Duplicate (for thread-safety) only when we need to upload stages
                    highlight_stage = duplicate_stage(highlight_stage_template)
                    plain_stage = duplicate_stage(plain_stage_template)

                # Use helper function to render and compose
                result = render_from_prepared_composition(
                    rendering_backend,
                    highlight_stage,
                    highlight_cameras,
                    plain_stage,
                    plain_cameras,
                    frames,
                    batch_prims,
                    config,
                    frame_range=frame_range,
                    sensors=sensor_modes,
                    image_height=image_height,
                    highlight_url=highlight_url,
                    plain_url=plain_url,
                )

                # Debug: Check if sensors are in result for composition
                if sensor_modes:
                    for cam_res in result.get("results", []):
                        if "sensors" in cam_res:
                            listener.debug(
                                f"Composition result has sensors: {list(cam_res['sensors'].keys())}"
                            )
                        else:
                            listener.warning(
                                f"Composition result missing 'sensors' key for camera {cam_res.get('camera')}!"
                            )

            elif prepared_info["type"] in (
                "prim_only",
                "prim_with_stage",
            ):
                # Unpack prepared prim-only or prim-with-stage data
                prepared_stage_template, camera_paths, num_frames = prepared_data

                # Calculate batch frame range
                frame_range = (batch_start, batch_end - 1)

                # Get pre-uploaded URL if available
                stage_url = prepared_info.get("stage_url")

                if stage_url and isinstance(rendering_backend, RemoteRenderingBackend):
                    # Use URL directly - no duplication needed!
                    prepared_stage = None
                else:
                    # Duplicate stage for this thread
                    prepared_stage = duplicate_stage(prepared_stage_template)

                # Use helper function to render
                result = render_from_prepared_prims(
                    rendering_backend,
                    prepared_stage,
                    camera_paths,
                    num_frames,
                    batch_prims,
                    config,
                    frame_range=frame_range,
                    sensors=sensor_modes,
                    image_height=image_height,
                    stage_url=stage_url,
                    render_mode=render_mode,
                )

            # Debug: Check if sensors are in result
            if sensor_modes:
                for cam_res in result.get("results", []):
                    if "sensors" in cam_res:
                        listener.debug(
                            f"Render result has sensors: {list(cam_res['sensors'].keys())}"
                        )
                    else:
                        listener.warning(
                            f"Render result missing 'sensors' key for camera {cam_res.get('camera')}!"
                        )

            failed_batches.extend(
                self._blank_render_failures_from_results(
                    result,
                    batch_start=batch_start,
                    batch_prims=batch_prims,
                    render_mode=render_mode,
                )
            )

            # Save all rendered images
            for camera_result in result.get("results", []):
                camera_name = camera_result.get("camera", "default")
                prim_images = camera_result.get("prim_to_images", {})
                prim_occlusion = camera_result.get("prim_occlusion", {})
                blank_stats_by_prim = self._blank_render_frame_stats_by_prim(
                    camera_result,
                    batch_start=batch_start,
                    batch_prims=batch_prims,
                )

                # Save images for each prim in this camera view
                for prim_path, image in prim_images.items():
                    # Check if this prim exists in our data (validation only)
                    if prim_path not in prim_data:
                        continue

                    # Skip if image is None (occluded and skip_occluded_images is enabled)
                    if image is None:
                        is_occluded = prim_occlusion.get(prim_path, False)
                        if is_occluded:
                            listener.debug(
                                f"  Skipping occluded image for {prim_path} in view {camera_name}"
                            )
                        continue

                    # Generate filename with render mode suffix
                    # Get the last component of the prim path for the filename
                    prim_parts = prim_path.strip("/").split("/")
                    prim_name = prim_parts[-1] if prim_parts else "unnamed"
                    safe_prim_name = shorten_for_filesystem(
                        prim_name, max_len=MAX_FILENAME_STEM_LEN
                    )

                    # Extract direction from camera name for view name
                    if "_" in camera_name:
                        view_name = camera_name.split("_", 1)[-1]
                    else:
                        view_name = camera_name

                    # Include render mode in filename - all modes get their own suffix
                    if render_mode == "prim_only":
                        filename = f"{safe_prim_name}_{view_name}_prim_only.png"
                    elif render_mode == "prim_with_stage":
                        filename = f"{safe_prim_name}_{view_name}_prim_with_stage.png"
                    elif render_mode == "composition":
                        filename = f"{safe_prim_name}_{view_name}_composition.png"
                    else:
                        # For any other future modes, use the mode name as suffix
                        filename = f"{safe_prim_name}_{view_name}_{render_mode}.png"

                    # Use directory structure based on prim path
                    filepath = prim_path_to_directory_structure(
                        prim_path, render_output_dir, filename
                    )

                    # Save image
                    image.save(filepath)
                    listener.debug(f"  Saved: {filepath}")

                    # Emit per-prim rendering event
                    try:
                        listener.event(
                            "rendering.completed",
                            {
                                "prim_path": prim_path,
                                "camera_view": view_name,
                                "render_mode": render_mode,
                                "output_path": str(filepath),
                                "camera": camera_name,
                            },
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to emit rendering event for {prim_path}: {e}"
                        )

                    # Collect image info to return (instead of modifying shared state)
                    if prim_path not in prim_images_data:
                        prim_images_data[prim_path] = []

                    image_info: dict[str, Any] = {
                        "view": view_name,
                        "path": str(filepath.relative_to(output_dir)),
                        "camera": camera_name,
                        "render_mode": render_mode,
                    }
                    if prim_path in blank_stats_by_prim:
                        image_info["blank_render"] = True
                        image_info["stats"] = blank_stats_by_prim[prim_path]
                    prim_images_data[prim_path].append(image_info)
                    total_images += 1

                # Save sensor data if present
                if "sensors" in camera_result and camera_result["sensors"]:
                    sensors_data = camera_result["sensors"]
                    listener.debug(
                        f"Processing sensors for camera {camera_name}: "
                        f"{list(sensors_data.keys())}"
                    )

                    for sensor_name, frame_data in sensors_data.items():
                        for frame_num, sensor_array in frame_data.items():
                            frame_idx = int(frame_num) - batch_start
                            if frame_idx < 0 or frame_idx >= len(batch_prims):
                                continue

                            prim_path = batch_prims[frame_idx]
                            if prim_path not in prim_data:
                                continue

                            prim_parts = prim_path.strip("/").split("/")
                            prim_name = prim_parts[-1] if prim_parts else "unnamed"
                            safe_prim_name = shorten_for_filesystem(
                                prim_name, max_len=MAX_FILENAME_STEM_LEN
                            )

                            if "_" in camera_name:
                                view_name = camera_name.split("_", 1)[-1]
                            else:
                                view_name = camera_name

                            sensor_filename = (
                                f"{safe_prim_name}_{view_name}_{sensor_name}.png"
                            )
                            sensor_filepath = prim_path_to_directory_structure(
                                prim_path, render_output_dir, sensor_filename
                            )

                            sensor_img = self._process_sensor_array(
                                sensor_array,
                                sensor_name,
                                config.image_width,
                                image_height,
                                prim_path,
                                listener,
                            )
                            if sensor_img is None:
                                continue

                            sensor_img.save(sensor_filepath)

                            listener.debug(
                                f"  Saved sensor {sensor_name}: {sensor_filepath}"
                            )

                            # Add sensor data to prim images data
                            if prim_path not in prim_images_data:
                                prim_images_data[prim_path] = []

                            prim_images_data[prim_path].append(
                                {
                                    "view": view_name,
                                    "path": str(
                                        sensor_filepath.relative_to(output_dir)
                                    ),
                                    "camera": camera_name,
                                    "render_mode": sensor_name,  # Use sensor name as render_mode
                                }
                            )
                            total_images += 1

        except Exception as e:
            listener.error(
                f"Failed to render batch starting at {batch_start}: {e}",
                exc_info=True,
            )
            # Track failed batches for reporting
            failed_batches.append(
                {
                    "batch_start": batch_start,
                    "batch_prims": batch_prims,
                    "error": str(e),
                }
            )

        return total_images, failed_batches, prim_images_data

    def _process_batch_with_retry_split(
        self,
        batch_start: int,
        batch_end: int,
        prims_to_render: list[str],
        prim_data: dict[str, dict[str, Any]],
        prepared_stages: dict[str, dict[str, Any]],
        rendering_backend,
        render_mode: str,
        render_output_dir: Path,
        output_dir: Path,
        num_total_tasks: int,
        batch_size: int,
        listener,
        sensor_modes: list[str] | None = None,
        image_height: int = 512,
    ) -> tuple[int, list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """Render a batch, splitting zero-image failures into smaller ranges."""
        total_images, failed_batches, prim_images_data = self._process_batch(
            batch_start,
            batch_end,
            prims_to_render,
            prim_data,
            prepared_stages,
            rendering_backend,
            render_mode,
            render_output_dir,
            output_dir,
            num_total_tasks,
            batch_size,
            listener,
            sensor_modes,
            image_height,
        )

        if (
            total_images > 0
            or batch_end - batch_start <= 1
            or not isinstance(rendering_backend, RemoteRenderingBackend)
        ):
            return total_images, failed_batches, prim_images_data

        mid = batch_start + (batch_end - batch_start) // 2
        listener.warning(
            f"Batch {batch_start}-{batch_end} for mode '{render_mode}' "
            f"produced 0 images; retrying as {batch_start}-{mid} "
            f"and {mid}-{batch_end}"
        )

        left_images, left_failures, left_prim_images = (
            self._process_batch_with_retry_split(
                batch_start,
                mid,
                prims_to_render,
                prim_data,
                prepared_stages,
                rendering_backend,
                render_mode,
                render_output_dir,
                output_dir,
                num_total_tasks,
                batch_size,
                listener,
                sensor_modes,
                image_height,
            )
        )
        right_images, right_failures, right_prim_images = (
            self._process_batch_with_retry_split(
                mid,
                batch_end,
                prims_to_render,
                prim_data,
                prepared_stages,
                rendering_backend,
                render_mode,
                render_output_dir,
                output_dir,
                num_total_tasks,
                batch_size,
                listener,
                sensor_modes,
                image_height,
            )
        )

        combined_prim_images: dict[str, list[dict[str, Any]]] = {}
        for batch_prim_images in (left_prim_images, right_prim_images):
            for prim_path, images in batch_prim_images.items():
                combined_prim_images.setdefault(prim_path, []).extend(images)

        return (
            left_images + right_images,
            left_failures + right_failures,
            combined_prim_images,
        )

    def _get_prim_type_from_string(self, type_string: str, listener) -> type | None:
        """Convert a string type name to the actual USD type class.

        Args:
            type_string: String name of the type
                (e.g., "UsdGeom.Mesh", "UsdGeom.Xform")
            listener: Event listener for logging
            diagnostics: Optional sink for render-evidence rejection details.
            assembly_target_members: Optional sink mapping recovered Xform
                owners to the explicit descendant Gprims represented by one row.
            recovered_guide_gprim_targets: Optional sink for recovered Gprim
                owners that remain one leaf row and need render-copy promotion.

        Returns:
            The USD type class or None if not found
        """
        # Try to resolve the type from the string
        try:
            # Split the type string to get module and class name
            if "." in type_string:
                module_name, class_name = type_string.rsplit(".", 1)
                # Handle common USD type patterns
                if module_name == "UsdGeom":
                    return getattr(UsdGeom, class_name, None)
                elif module_name == "UsdShade":
                    from pxr import UsdShade

                    return getattr(UsdShade, class_name, None)
                elif module_name == "UsdLux":
                    from pxr import UsdLux

                    return getattr(UsdLux, class_name, None)
                elif module_name == "UsdSkel":
                    from pxr import UsdSkel

                    return getattr(UsdSkel, class_name, None)
                elif module_name == "UsdVol":
                    try:
                        from pxr import UsdVol
                    except ImportError:
                        listener.warning(
                            "pxr.UsdVol is not available from the active "
                            "OpenUSD provider; falling back to exact typeName "
                            f"matching for '{type_string}'."
                        )
                        return None

                    schema_class = getattr(UsdVol, class_name, None)
                    if schema_class is None:
                        listener.warning(
                            f"UsdVol schema class '{class_name}' was not found; "
                            f"falling back to exact typeName matching for "
                            f"'{type_string}'."
                        )
                    return schema_class
                else:
                    # Try using Tf.Type for dynamic type resolution
                    tf_type = Tf.Type.FindByName(type_string)
                    if tf_type:
                        return tf_type.pythonClass
            else:
                # Handle simple class names (assume UsdGeom)
                return getattr(UsdGeom, type_string, None)
        except Exception as e:
            listener.warning(f"Could not resolve type '{type_string}': {e}")
        return None

    def _matches_type_name_fallback(self, prim: "Usd.Prim", type_string: str) -> bool:
        """Match concrete typed schemas when provider Python classes are missing."""
        if "." not in type_string:
            return False

        module_name, class_name = type_string.rsplit(".", 1)
        if module_name != "UsdVol":
            return False

        # usd-exchange's Linux ARM64 wheel provides core pxr bindings but
        # currently omits pxr.UsdVol. Exact typeName matching keeps concrete
        # filters such as UsdVol.Volume useful without claiming full schema
        # inheritance support from the missing Python module.
        return str(prim.GetTypeName()) == class_name

    @staticmethod
    def _stage_type_histogram(stage: "Usd.Stage") -> dict[str, int]:
        """Return concrete prim type counts for filter-mismatch diagnostics."""

        histogram: dict[str, int] = {}
        prim_range = Usd.PrimRange(
            stage.GetPseudoRoot(),
            Usd.TraverseInstanceProxies(),
        )
        for prim in prim_range:
            if prim.IsPseudoRoot():
                continue
            type_name = str(prim.GetTypeName()) or "<untyped>"
            histogram[type_name] = histogram.get(type_name, 0) + 1
        return dict(sorted(histogram.items()))

    def _require_matching_prims(
        self,
        stage: "Usd.Stage",
        prim_paths: list[str],
        filters: dict[str, Any],
        filter_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        """Reject an empty selection before stage preparation or upload."""

        if prim_paths:
            return
        filters_json = json.dumps(filters, default=str, sort_keys=True)
        histogram_json = json.dumps(
            self._stage_type_histogram(stage),
            sort_keys=True,
        )
        diagnostic_detail = (
            "; prim_filter_diagnostics="
            + json.dumps(filter_diagnostics, default=str, sort_keys=True)
            if filter_diagnostics and filter_diagnostics.get("skipped_prims")
            else ""
        )
        raise RuntimeError(
            "No USD prims matched prim_filters before render preparation; "
            f"prim_filters={filters_json}; "
            f"stage_type_histogram={histogram_json}"
            f"{diagnostic_detail}"
        )

    def _collect_prims(
        self,
        stage: "Usd.Stage",
        filters: dict[str, Any],
        listener,
        diagnostics: list[dict[str, Any]] | None = None,
        assembly_target_members: dict[str, list[str]] | None = None,
        recovered_guide_gprim_targets: set[str] | None = None,
    ) -> list[str]:
        """Collect prims based on filters.

        Args:
            stage: USD stage
            filters: Dict with optional keys:
                - types: List of prim type names (e.g., ["UsdGeom.Mesh"])
                - paths: List of specific prim paths
                - exclude_paths: List of paths to exclude
                - skip_instances: Skip instance prims and instance proxies (default: True)
                - skip_prototypes: Skip prims inside prototype hierarchies (default: False)
                - skip_invisible: Skip prims with computed invisible visibility (default: False)
                - allowed_purposes: Effective purposes eligible for rendering (default: all)
                - rigid_body_purpose_fallbacks: Otherwise excluded purposes that may
                  be recovered when an enabled rigid-body owner has no selected
                  descendant (default: none)
                - skip_fully_transparent: Skip opacity-zero Gprims (default: False)
                - skip_unusable_bbox: Skip Gprims that cannot be camera-framed (default: False)
                - root_prim: Root prim path to start traversal from (default: None, traverse entire stage)
            listener: Event listener for logging

        Returns:
            List of prim paths matching filters
        """
        prims_to_render = []

        # Get filter parameters
        prim_types = filters.get("types", ["UsdGeom.Mesh"])
        specific_paths = filters.get("paths", [])
        exclude_paths = filters.get("exclude_paths", [])
        skip_instances = filters.get("skip_instances", True)
        skip_prototypes = filters.get("skip_prototypes", False)
        skip_invisible = filters.get("skip_invisible", False)
        root_prim_path = filters.get("root_prim", None)

        # Track statistics
        skipped_instances = 0
        skipped_prototypes = 0
        skipped_invisible = 0
        render_filter_skips: list[dict[str, Any]] = []
        selected_rigid_body_owners: set[str] = set()
        fallback_candidates_by_owner: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        fallback_purposes = set(_rigid_body_purpose_fallbacks_from_filters(filters))
        bbox_cache = None
        if filters.get("skip_unusable_bbox", False):
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode(0),
                list(_bbox_purposes_from_filters(filters)),
                useExtentsHint=False,
            )

        def _is_invisible(prim: "Usd.Prim") -> bool:
            imageable = UsdGeom.Imageable(prim)
            return (
                bool(imageable)
                and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
            )

        def _record_render_filter_skip(prim: "Usd.Prim") -> bool:
            diagnostic = _render_evidence_skip_diagnostic(
                prim,
                filters,
                bbox_cache,
            )
            if diagnostic is None:
                return False
            if (
                diagnostic["reason"] == _PURPOSE_NOT_ALLOWED
                and diagnostic.get("purpose") in fallback_purposes
            ):
                owner_path = _nearest_enabled_rigid_body_owner(prim)
                # Re-evaluate every non-purpose evidence guard with the explicit
                # fallback purpose admitted. Source transparency is allowed only
                # because accepted members are promoted on duplicated render
                # stages; an unframeable guide still cannot recover its owner.
                fallback_filters = {
                    **filters,
                    "allowed_purposes": [
                        *(_allowed_purposes_from_filters(filters) or ()),
                        str(diagnostic["purpose"]),
                    ],
                    # The rejected guide leaves are not prediction rows. They
                    # supply a bound for one owner target, whose normal render
                    # preparation blocks authored display opacity together with
                    # other source material state. Keep the finite bbox guard.
                    "skip_fully_transparent": False,
                }
                if owner_path is not None and (
                    _render_evidence_skip_diagnostic(
                        prim,
                        fallback_filters,
                    )
                    is None
                ):
                    render_filter_skips.append(diagnostic)
                    if diagnostics is not None:
                        diagnostics.append(diagnostic)
                    fallback_candidates_by_owner.setdefault(owner_path, []).append(
                        (str(prim.GetPath()), diagnostic)
                    )
                    return True
            render_filter_skips.append(diagnostic)
            if diagnostics is not None:
                diagnostics.append(diagnostic)
            listener.debug(
                "Skipping render-ineligible prim: "
                f"{diagnostic['prim_path']} (reason={diagnostic['reason']})"
            )
            return True

        # If specific paths provided, use those
        if specific_paths:
            for path in specific_paths:
                prim = stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    # Skip instances and instance proxies if configured
                    if skip_instances and (prim.IsInstance() or prim.IsInstanceProxy()):
                        listener.debug(f"Skipping instance/proxy: {path}")
                        skipped_instances += 1
                        continue
                    # Skip prims inside prototypes if configured
                    if skip_prototypes and prim.IsInPrototype():
                        listener.debug(f"Skipping prototype prim: {path}")
                        skipped_prototypes += 1
                        continue
                    if skip_invisible and _is_invisible(prim):
                        listener.debug(f"Skipping invisible prim: {path}")
                        skipped_invisible += 1
                        continue
                    if _record_render_filter_skip(prim):
                        continue
                    prims_to_render.append(path)
                    if fallback_purposes:
                        owner_path = _nearest_enabled_rigid_body_owner(prim)
                        if owner_path is not None:
                            selected_rigid_body_owners.add(owner_path)
        else:
            # Determine the root for traversal
            if root_prim_path:
                root_prim = stage.GetPrimAtPath(root_prim_path)
                if not root_prim or not root_prim.IsValid():
                    listener.warning(
                        f"Root prim '{root_prim_path}' not found, falling back to stage traversal"
                    )
                    prim_iterator = stage.TraverseAll()
                else:
                    listener.info(f"Traversing from root prim: {root_prim_path}")
                    # Use Traverse() on the root prim to get all descendants
                    prim_iterator = Usd.PrimRange(
                        root_prim, Usd.TraverseInstanceProxies()
                    )
            else:
                # Traverse all prims and filter by type
                # Use TraverseAll() to include prims inside class hierarchies (abstract prims)
                prim_iterator = Usd.PrimRange(
                    stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()
                )

            resolved_prim_types = [
                (type_string, self._get_prim_type_from_string(type_string, listener))
                for type_string in prim_types
            ]

            for prim in prim_iterator:
                prim_path = str(prim.GetPath())

                # Skip excluded paths
                if any(prim_path.startswith(exclude) for exclude in exclude_paths):
                    continue

                # Skip instances and instance proxies if configured
                if skip_instances and (prim.IsInstance() or prim.IsInstanceProxy()):
                    listener.debug(f"Skipping instance/proxy: {prim_path}")
                    skipped_instances += 1
                    continue

                # Skip prims inside prototypes if configured
                if skip_prototypes and prim.IsInPrototype():
                    listener.debug(f"Skipping prototype prim: {prim_path}")
                    skipped_prototypes += 1
                    continue

                # Check prim type dynamically
                for type_string, prim_type in resolved_prim_types:
                    if prim_type:
                        is_match = prim.IsA(prim_type)
                    else:
                        is_match = self._matches_type_name_fallback(prim, type_string)

                    if is_match:
                        if skip_invisible and _is_invisible(prim):
                            listener.debug(f"Skipping invisible prim: {prim_path}")
                            skipped_invisible += 1
                            break
                        if _record_render_filter_skip(prim):
                            break
                        prims_to_render.append(prim_path)
                        if fallback_purposes:
                            owner_path = _nearest_enabled_rigid_body_owner(prim)
                            if owner_path is not None:
                                selected_rigid_body_owners.add(owner_path)
                        break  # Stop checking once we find a match

        for owner_path, fallback_candidates in fallback_candidates_by_owner.items():
            # The rejected guide leaves remain diagnostics even when their
            # bounded owner is recovered. They are evidence for one assembly,
            # not independent prediction rows.
            if owner_path in selected_rigid_body_owners:
                continue
            if owner_path in prims_to_render:
                continue
            owner = stage.GetPrimAtPath(owner_path)
            owner_is_gprim = owner.IsA(UsdGeom.Gprim)
            owner_is_xform = owner.IsA(UsdGeom.Xform)
            if not owner_is_gprim and not owner_is_xform:
                listener.warning(
                    "Cannot recover guide-only rigid-body owner "
                    f"{owner_path}: expected UsdGeom.Gprim or UsdGeom.Xform, "
                    f"found {owner.GetTypeName() or '<untyped>'}"
                )
                continue
            fallback_member_paths = [path for path, _ in fallback_candidates]
            if owner_is_gprim and owner_path not in fallback_member_paths:
                listener.info(
                    "Did not recover rigid-body Gprim owner "
                    f"{owner_path}: only descendant guide candidates passed "
                    "the render-evidence filters"
                )
                continue
            prims_to_render.append(owner_path)
            if owner_is_xform and assembly_target_members is not None:
                assembly_target_members[owner_path] = fallback_member_paths
            if owner_is_gprim and recovered_guide_gprim_targets is not None:
                recovered_guide_gprim_targets.add(owner_path)
            if owner_is_gprim:
                ignored_descendant_count = len(fallback_member_paths) - 1
                listener.info(
                    "Recovered guide-only rigid-body owner "
                    f"{owner_path} as one bounded leaf target from its accepted "
                    f"self candidate; {ignored_descendant_count} descendant guide "
                    "candidate(s) remained diagnostic-only"
                )
            else:
                listener.info(
                    "Recovered guide-only rigid-body owner "
                    f"{owner_path} as one bounded assembly target from "
                    f"{len(fallback_candidates)} purpose-filtered descendant(s)"
                )

        if skipped_instances > 0:
            listener.info(
                f"Skipped {skipped_instances} instance/proxy prims (skip_instances=True)"
            )

        if skipped_prototypes > 0:
            listener.info(
                f"Skipped {skipped_prototypes} prototype prims (skip_prototypes=True)"
            )

        if skipped_invisible > 0:
            listener.info(
                f"Skipped {skipped_invisible} invisible prims (skip_invisible=True)"
            )

        render_filter_summary = _summarize_prim_filter_diagnostics(render_filter_skips)
        for reason, count in render_filter_summary["reason_counts"].items():
            listener.info(f"Skipped {count} render-ineligible prims (reason={reason})")
        return prims_to_render

    def _check_prim_files_exist(
        self,
        prim_path: str,
        render_output_dir: Path,
        rendering_config: Any,
        rendering_modes: list[str] | None = None,
        sensor_modes: list[str] | None = None,
    ) -> bool:
        """Check if expected output files exist for a prim.

        Args:
            prim_path: Path to the prim
            render_output_dir: Directory for rendered images
            rendering_config: Rendering configuration
            rendering_modes: List of RGB rendering modes to check
            sensor_modes: List of sensor rendering modes to check

        Returns:
            True if all expected files exist
        """
        if rendering_modes is None:
            rendering_modes = ["prim_with_stage", "prim_only"]
        if sensor_modes is None:
            sensor_modes = []

        # Get the last component of the prim path for the filename
        prim_parts = prim_path.strip("/").split("/")
        prim_name = prim_parts[-1] if prim_parts else "unnamed"
        safe_prim_name = shorten_for_filesystem(
            prim_name, max_len=MAX_FILENAME_STEM_LEN
        )

        # Check RGB mode files exist
        for render_mode in rendering_modes:
            # Get cameras for this specific mode (supports per-mode camera configuration)
            camera_specs = rendering_config.get_cameras_for_mode(render_mode)

            for camera_spec in camera_specs:
                # Build expected camera name using the same logic as rendering.py
                dir_suffix = format_direction_for_filename(camera_spec.direction)
                camera_name = f"{rendering_config.camera_name_prefix}_{dir_suffix}"

                # Extract view name from camera name
                if "_" in camera_name:
                    view_name = camera_name.split("_", 1)[-1]
                else:  # pragma: no cover - camera_name is always built as prefix_suffix
                    view_name = camera_name

                # Include render mode in filename - all modes get their own suffix
                if render_mode == "prim_only":
                    filename = f"{safe_prim_name}_{view_name}_prim_only.png"
                elif render_mode == "prim_with_stage":
                    filename = f"{safe_prim_name}_{view_name}_prim_with_stage.png"
                elif render_mode == "composition":
                    filename = f"{safe_prim_name}_{view_name}_composition.png"
                else:
                    # For any other future modes, use the mode name as suffix
                    filename = f"{safe_prim_name}_{view_name}_{render_mode}.png"

                # Use directory structure based on prim path
                filepath = prim_path_to_directory_structure(
                    prim_path, render_output_dir, filename
                )

                if not filepath.exists():
                    # If any file is missing, return False
                    return False

        # Check sensor mode files exist
        for sensor_mode in sensor_modes:
            # Get cameras for this specific sensor mode (supports per-mode camera configuration)
            camera_specs = rendering_config.get_cameras_for_mode(sensor_mode)

            for camera_spec in camera_specs:
                # Build expected camera name using the same logic as rendering.py
                dir_suffix = format_direction_for_filename(camera_spec.direction)
                camera_name = f"{rendering_config.camera_name_prefix}_{dir_suffix}"

                # Extract view name from camera name
                if "_" in camera_name:
                    view_name = camera_name.split("_", 1)[-1]
                else:  # pragma: no cover - camera_name is always built as prefix_suffix
                    view_name = camera_name

                # Sensor files are named as: {prim}_{view}_{sensor}.png
                sensor_filename = f"{safe_prim_name}_{view_name}_{sensor_mode}.png"

                # Use directory structure based on prim path
                sensor_filepath = prim_path_to_directory_structure(
                    prim_path, render_output_dir, sensor_filename
                )

                if not sensor_filepath.exists():
                    # If any sensor file is missing, return False
                    return False

        # All files exist
        return True

    def _record_existing_files(
        self,
        prim_info: dict[str, Any],
        prim_path: str,
        render_output_dir: Path,
        rendering_config: Any,
        output_dir: Path,
        rendering_modes: list[str] | None = None,
        sensor_modes: list[str] | None = None,
    ) -> None:
        """Record information about existing rendered files.

        Args:
            prim_info: Prim info dict to update
            prim_path: Path to the prim
            render_output_dir: Directory for rendered images
            rendering_config: Rendering configuration
            output_dir: Base output directory for relative paths
            rendering_modes: List of RGB rendering modes to check
            sensor_modes: List of sensor rendering modes to check
        """
        if rendering_modes is None:
            rendering_modes = ["prim_with_stage", "prim_only"]
        if sensor_modes is None:
            sensor_modes = []

        # Get the last component of the prim path for the filename
        prim_parts = prim_path.strip("/").split("/")
        prim_name = prim_parts[-1] if prim_parts else "unnamed"
        safe_prim_name = shorten_for_filesystem(
            prim_name, max_len=MAX_FILENAME_STEM_LEN
        )

        # Record RGB mode files
        for render_mode in rendering_modes:
            # Get cameras for this specific mode (supports per-mode camera configuration)
            camera_specs = rendering_config.get_cameras_for_mode(render_mode)

            for camera_spec in camera_specs:
                # Build expected camera name using the same logic as rendering.py
                dir_suffix = format_direction_for_filename(camera_spec.direction)
                camera_name = f"{rendering_config.camera_name_prefix}_{dir_suffix}"

                if "_" in camera_name:
                    view_name = camera_name.split("_", 1)[-1]
                else:  # pragma: no cover - camera_name is always built as prefix_suffix
                    view_name = camera_name

                # Include render mode in filename - all modes get their own suffix
                if render_mode == "prim_only":
                    filename = f"{safe_prim_name}_{view_name}_prim_only.png"
                elif render_mode == "prim_with_stage":
                    filename = f"{safe_prim_name}_{view_name}_prim_with_stage.png"
                elif render_mode == "composition":
                    filename = f"{safe_prim_name}_{view_name}_composition.png"
                else:
                    # For any other future modes, use the mode name as suffix
                    filename = f"{safe_prim_name}_{view_name}_{render_mode}.png"

                # Use directory structure based on prim path
                filepath = prim_path_to_directory_structure(
                    prim_path, render_output_dir, filename
                )

                if filepath.exists():
                    prim_info["images"].append(
                        {
                            "view": view_name,
                            "path": str(filepath.relative_to(output_dir)),
                            "camera": camera_name,
                            "render_mode": render_mode,
                            "skipped": True,
                        }
                    )

        # Record sensor mode files
        for sensor_mode in sensor_modes:
            # Get cameras for this specific sensor mode (supports per-mode camera configuration)
            camera_specs = rendering_config.get_cameras_for_mode(sensor_mode)

            for camera_spec in camera_specs:
                # Build expected camera name using the same logic as rendering.py
                dir_suffix = format_direction_for_filename(camera_spec.direction)
                camera_name = f"{rendering_config.camera_name_prefix}_{dir_suffix}"

                if "_" in camera_name:
                    view_name = camera_name.split("_", 1)[-1]
                else:  # pragma: no cover - camera_name is always built as prefix_suffix
                    view_name = camera_name

                # Sensor files are named as: {prim}_{view}_{sensor}.png
                sensor_filename = f"{safe_prim_name}_{view_name}_{sensor_mode}.png"

                # Use directory structure based on prim path
                sensor_filepath = prim_path_to_directory_structure(
                    prim_path, render_output_dir, sensor_filename
                )

                if sensor_filepath.exists():
                    prim_info["images"].append(
                        {
                            "view": view_name,
                            "path": str(sensor_filepath.relative_to(output_dir)),
                            "camera": camera_name,
                            "render_mode": sensor_mode,
                            "skipped": True,
                        }
                    )

    def _traverse_to_root(self, prim: "Usd.Prim"):
        """Generator that yields prims from current prim up to root.

        Args:
            prim: Starting USD prim

        Yields:
            Prim objects from current to root
        """
        current = prim
        while current and current.IsValid():
            yield current
            current = current.GetParent()

    def _extract_prim_metadata(self, prim: "Usd.Prim") -> dict[str, Any]:
        """Extract metadata from a USD prim.

        Args:
            prim: USD prim

        Returns:
            Dictionary of metadata
        """
        metadata = {
            "type": prim.GetTypeName(),
            "path": str(prim.GetPath()),
            "active": prim.IsActive(),
        }

        # Get bounding box if it's a boundable prim
        if prim.IsA(UsdGeom.Boundable):
            boundable = UsdGeom.Boundable(prim)
            extent = boundable.GetExtentAttr()
            if extent and extent.HasValue():
                try:
                    metadata["extent"] = extent.Get()
                except Exception:
                    # Handle cases where extent exists but can't be retrieved
                    pass

        # Get transform if it's an xformable
        if prim.IsA(UsdGeom.Xformable):
            xformable = UsdGeom.Xformable(prim)
            xform_ops = xformable.GetOrderedXformOps()
            if xform_ops:
                metadata["has_transform"] = True
                # Could extract actual transform matrix if needed

        # Get material binding if exists (legacy field for backward compatibility)
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            binding_api = UsdShade.MaterialBindingAPI(prim)
            try:
                direct_binding = binding_api.GetDirectBinding()
                if direct_binding:
                    material = direct_binding.GetMaterial()
                    if material:
                        metadata["material"] = str(material.GetPath())
            except Exception:
                # Handle invalid material bindings
                pass

        # Extract custom data, references, and hoops metadata by searching up hierarchy
        custom_data = {}
        references = []
        hoops_metadata = {}
        annotation = None

        for ancestor in self._traverse_to_root(prim):
            # Extract annotation (stop at first found)
            if annotation is None and ancestor.HasCustomData():
                annotation = ancestor.GetCustomDataByKey("annotation")
                if annotation is not None:
                    custom_data["annotation"] = annotation

            # Extract references from prim stack
            prim_stack = ancestor.GetPrimStack()
            if prim_stack:
                for prim_spec in prim_stack:
                    for ref in prim_spec.referenceList.GetAddedOrExplicitItems():
                        if ref.primPath:
                            ref_path = str(ref.primPath)
                            if ref_path not in references:
                                references.append(ref_path)

            # Extract omni:hoops:metadata:* attributes
            for attr in ancestor.GetAttributes():
                attr_name = attr.GetName()
                if attr_name.startswith("omni:hoops:metadata:"):
                    try:
                        value = attr.Get()
                        if value is not None and str(value).strip():
                            short_name = attr_name.split(":")[-1]
                            if short_name not in hoops_metadata:
                                hoops_metadata[short_name] = value
                    except Exception:
                        pass

        # Add to metadata if found
        if custom_data:
            metadata["custom_data"] = custom_data
        if references:
            metadata["references"] = references
        if hoops_metadata:
            metadata["hoops_metadata"] = hoops_metadata

        return metadata

    def _extract_display_color(self, prim: "Usd.Prim", listener) -> list[float] | None:
        """Extract display color from a USD prim.

        The display color is typically stored in the primvars:displayColor attribute.
        This is commonly used for visualization purposes.

        Args:
            prim: USD prim
            listener: Event listener for logging

        Returns:
            RGB color as a list of floats [r, g, b], or None if not found
        """
        try:
            # Check if the prim has primvars:displayColor attribute
            if prim.HasAttribute("primvars:displayColor"):
                display_color_attr = prim.GetAttribute("primvars:displayColor")
                if display_color_attr and display_color_attr.HasValue():
                    color_value = display_color_attr.Get()
                    # Color might be an array of colors (one per vertex/face)
                    # Return the first color if it's an array
                    if color_value:
                        if hasattr(color_value, "__len__") and len(color_value) > 0:
                            # Get first color and convert to list
                            first_color = color_value[0]
                            if hasattr(first_color, "__len__"):
                                return list(first_color)
                            return [float(first_color)]
                        return None
        except Exception as e:
            listener.debug(
                f"Could not extract display color from {prim.GetPath()}: {e}"
            )

        return None

    def _extract_material_bindings(
        self, prim: "Usd.Prim", stage: "Usd.Stage", listener
    ) -> dict[str, Any]:
        """Extract detailed material binding information from a USD prim.

        Args:
            prim: USD prim
            stage: USD stage
            listener: Event listener for logging

        Returns:
            Dictionary with material binding details
        """
        bindings = {}

        try:
            # Check if prim has material binding API
            if not prim.HasAPI(UsdShade.MaterialBindingAPI):
                return bindings

            binding_api = UsdShade.MaterialBindingAPI(prim)

            # Prefer ComputeBoundMaterial when available; fall back to relationship targets
            bound_mat = None
            try:
                # Compute bound material for any purpose
                result = binding_api.ComputeBoundMaterial()
                if isinstance(result, tuple):
                    bound_mat = result[0]
                    binding_rel = result[1] if len(result) > 1 else None
                else:
                    bound_mat = result
                    binding_rel = None
            except Exception:
                rel = binding_api.GetDirectBindingRel()
                targets = rel.GetTargets() if rel else []
                if len(targets) > 0:
                    mat_prim = prim.GetStage().GetPrimAtPath(str(targets[0]))
                    if mat_prim and mat_prim.IsA(UsdShade.Material):
                        bound_mat = UsdShade.Material(mat_prim)
                binding_rel = rel

            if bound_mat and isinstance(bound_mat, UsdShade.Material):
                bindings["resolved"] = str(bound_mat.GetPath())

                # Extract MDL file paths and metadata from the material
                listener.info(
                    f"Extracting MDL paths from material {bound_mat.GetPath()}"
                )
                mdl_info = self._extract_mdl_paths_from_material(bound_mat, listener)
                if mdl_info:
                    # Merge all MDL information into bindings
                    bindings.update(mdl_info)
                else:
                    listener.warning(
                        f"No MDL paths found for material {bound_mat.GetPath()}"
                    )

                # Determine where the binding is defined
                if binding_rel and binding_rel.IsValid():
                    binding_prim = binding_rel.GetPrim()
                    bindings["bound_at"] = str(binding_prim.GetPath())
                    bindings["inherited"] = str(binding_prim.GetPath()) != str(
                        prim.GetPath()
                    )

            # Check for subset/face-set material assignments
            # These are material bindings on child prims with GeomSubset type
            subassignments = {}
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Subset):
                    UsdGeom.Subset(child)
                    if child.HasAPI(UsdShade.MaterialBindingAPI):
                        subset_binding_api = UsdShade.MaterialBindingAPI(child)
                        subset_binding = subset_binding_api.GetDirectBinding()
                        if subset_binding:
                            subset_material = subset_binding.GetMaterial()
                            if subset_material and subset_material.GetPrim().IsValid():
                                subset_name = child.GetName()
                                subassignments[f"subset:{subset_name}"] = str(
                                    subset_material.GetPath()
                                )

            if subassignments:
                bindings["subassignments"] = subassignments

        except Exception as e:
            listener.debug(
                f"Error extracting material bindings for {prim.GetPath()}: {e}"
            )

        return bindings

    def _extract_mdl_paths_from_material(
        self, material: "UsdShade.Material", listener
    ) -> dict[str, Any]:
        """Extract MDL file paths and metadata from a USD material.

        Args:
            material: USD material
            listener: Event listener for logging

        Returns:
            Dictionary with MDL paths and metadata
        """
        mdl_info = {}

        try:
            # Get MDL-specific surface output
            surf_out = material.GetSurfaceOutput("mdl")
            if surf_out and surf_out.HasConnectedSource():
                source = surf_out.GetConnectedSource()
                shader = source[0]
                if shader:
                    shader_prim = UsdShade.Shader(shader.GetPrim())
                    shader_path = str(shader_prim.GetPath())
                    mdl_info["shader_path"] = shader_path

                    # Get MDL source asset path
                    mdl_attr = shader_prim.GetPrim().GetAttribute(
                        "info:mdl:sourceAsset"
                    )
                    if mdl_attr and mdl_attr.IsValid():
                        asset_val = mdl_attr.Get()
                        if asset_val is not None:
                            try:
                                mdl_asset_path = asset_val.GetAssetPath()
                            except Exception:
                                try:
                                    mdl_asset_path = asset_val.path
                                except Exception:
                                    mdl_asset_path = str(asset_val)
                            mdl_info["mdl_path"] = mdl_asset_path

                    # Get MDL sub-identifier
                    sub_id_attr = shader_prim.GetPrim().GetAttribute(
                        "info:mdl:sourceAsset:subIdentifier"
                    )
                    if sub_id_attr and sub_id_attr.IsValid():
                        sub_identifier = sub_id_attr.Get()
                        mdl_info["mdl_sub_identifier"] = sub_identifier

        except Exception as e:
            listener.debug(
                f"Error extracting MDL paths from material {material.GetPath()}: {e}"
            )

        return mdl_info

    def _extract_hierarchy_info(
        self, prim: "Usd.Prim", usd_model: USDModel | None
    ) -> dict[str, Any]:
        """Extract hierarchy information for a USD prim.

        Args:
            prim: USD prim
            usd_model: Optional USDModel for efficient hierarchy queries

        Returns:
            Dictionary with hierarchy information
        """
        prim_path = str(prim.GetPath())
        hierarchy = {
            "type_name": str(prim.GetTypeName()) if prim.GetTypeName() else None,
            "is_xform": prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Xformable),
            "is_instance": prim.IsInstance(),
        }

        if usd_model:
            # Use USDModel for efficient queries
            prim_node = usd_model.get_prim(prim_path)
            if prim_node:
                hierarchy["parent_path"] = prim_node.parent_path
                hierarchy["children_paths"] = prim_node.children_paths.copy()

                # Get ancestors
                ancestors = usd_model.get_ancestors(prim_path, include_self=False)
                hierarchy["ancestors"] = [a.path for a in ancestors]

                # Get collections containing this prim
                collections = usd_model.get_collections_containing_prim(prim_path)
                hierarchy["collections"] = [
                    {"name": c.name, "prim_path": c.prim_path} for c in collections
                ]
        else:
            # Fall back to direct USD API queries
            parent = prim.GetParent()
            if parent and parent.IsValid() and not parent.IsPseudoRoot():
                hierarchy["parent_path"] = str(parent.GetPath())
            else:
                hierarchy["parent_path"] = None

            # Get children paths
            children = prim.GetChildren()
            hierarchy["children_paths"] = [str(c.GetPath()) for c in children]

            # Get ancestors
            ancestors = []
            current = prim.GetParent()
            while current and current.IsValid() and not current.IsPseudoRoot():
                ancestors.append(str(current.GetPath()))
                current = current.GetParent()
            hierarchy["ancestors"] = ancestors

            # Collections would require more complex traversal without USDModel
            hierarchy["collections"] = []

        return hierarchy
