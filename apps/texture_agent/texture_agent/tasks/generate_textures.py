# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Generate PBR texture sets from prompts.

Supports two backend types (configured via texture_config.backend_type):
- "simple_image_gen" (default): Uses ImageGenEngine (Gemini image gen) in-process
- "service": Calls a remote Texture Variation API service via REST endpoint

Iterates over PrimTextureUnit list (from DiscoverMaterialsTask), which
handles both per-material and per-prim modes transparently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse

from apps.texture_gen_service_common.weathering_intent import (
    prompt_requests_weathering,
)
from world_understanding.agentic.tasks import Task
from world_understanding.utils.archive import ArchiveSizeLimitExceeded
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    copy_open_file_to_confined,
    open_confined_directory,
    open_regular_file_no_follow,
)
from world_understanding.utils.usd.package import (
    extract_usdz_member_to_dir,
    resolve_local_package_path,
    split_package_member_asset_path,
)

from texture_agent.functions.detail_policy import (
    DETAIL_POLICY_DEFAULT,
    DETAIL_POLICY_SURFACE_ONLY,
    SURFACE_ONLY_FORBIDDEN_DETAILS,
)
from texture_agent.functions.external_authoring import (
    EXTERNAL_AUTHORING_FEASIBILITY_SCHEMA_VERSION,
    ExternalAuthoringCapabilityReceipt,
    ExternalAuthoringSpec,
    evaluate_external_authoring_feasibility,
    unavailable_external_authoring_feasibility,
    validate_external_authoring_result,
)
from texture_agent.functions.material_discovery import (
    PrimTextureUnit,
    resolve_material_texture_spec,
)
from texture_agent.functions.texture_generation import (
    BackendCapabilities,
    Conditioning,
    GeneratedTextures,
    ImageGenEngine,
    MapArtifact,
    TextureTarget,
    TextureVariationClient,
    TextureVariationConfig,
    WeatheringControls,
)
from texture_agent.tasks.thresholds import (
    raise_if_failure_threshold_exceeded,
    validate_failure_threshold,
)

logger = logging.getLogger(__name__)

_HTTP_STATUS_RE = re.compile(r"HTTP\s*(?:Error\s*)?(\d{3})", re.IGNORECASE)
_WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_SUPPORTED_REFERENCE_URI_SCHEMES = {
    "",
    "file",
    "http",
    "https",
    "s3",
    "omni",
    "omniverse",
}
_DEFAULT_SERVICE_JOB_TIMEOUT_SEC = 3600
_MAX_PACKAGE_TEXTURE_BYTES = 512 * 1024 * 1024
_REFERENCE_TEXTURE_EXTENSIONS = frozenset(
    {".bmp", ".exr", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
)
_REBAKE_TEXTURE_INPUTS = {
    "albedo": {
        "albedo",
        "albedo_texture",
        "basecolor",
        "base_color",
        "base_color_texture",
        "base_color_texture_file",
        "diffuse",
        "diffusecolor",
        "diffuse_color",
        "diffuse_texture",
    },
    "normal": {
        "coat_normal_texture_file",
        "detail_normalmap_texture",
        "geometry_normal_texture_file",
        "normal",
        "normal_map_texture",
        "normal_texture",
        "normalmap",
        "normalmap_texture",
    },
    "orm": {
        "ao_roughness_metallic",
        "arm",
        "metallicroughness",
        "metallic_roughness",
        "occlusionroughnessmetallic",
        "orm",
        "orm_texture",
        "ormtexture",
    },
}
_REBAKE_AUTHOR_TEXTURE_INPUTS = {
    "albedo": ("inputs:base_color_texture_file",),
    "normal": ("inputs:normalmap_texture", "inputs:geometry_normal_texture_file"),
    "orm": ("inputs:ORM_texture",),
}
_STEP1X_OVERLAY_TARGET_TOKENS = (
    "decal",
    "decals",
    "label",
    "labels",
    "overlay",
    "overlays",
    "sticker",
    "stickers",
)
_SIMPLE_IMAGE_GEN_ENGINES = {"image_gen", "simple", "simple_image_gen"}


def _is_windows_drive_path(value: str) -> bool:
    return bool(_WINDOWS_DRIVE_PATH_RE.match(value))


def _file_uri_path(parsed: Any) -> Path:
    """Return a local Path from tolerant file URI parsing.

    Python's urlparse handles standard file URIs well, but test and service
    call sites may pass Windows paths as ``file://C:/...`` or
    ``file://C:\\...``. Treat those drive-letter hosts as local paths.
    """
    host = unquote(parsed.netloc)
    path = unquote(parsed.path)
    if host in {"", "localhost"}:
        if re.match(r"^/[A-Za-z]:[\\/]", path):
            path = path[1:]
        return Path(path)
    if _is_windows_drive_path(host):
        return Path(f"{host}{path}")
    if re.fullmatch(r"[A-Za-z]:", host):
        return Path(f"{host}{path}")
    raise RuntimeError(f"Only local file URIs are supported: {parsed.geturl()}")


def _classify_unit_failure(unit_key: str, exc: BaseException) -> dict[str, Any]:
    """Build a structured per-unit error record for SSE/status surfacing.

    Best-effort HTTP status extraction, in order of preference:
      1. ``httpx.HTTPStatusError.response.status_code`` -- raised by the
         service backend's ``RestTextureVariationClient`` polling path.
         The default ``httpx`` message format ("Client error '403 Forbidden'
         for url ...") does NOT contain a literal ``HTTP <NNN>`` substring,
         so the regex fallback below would miss it.
      2. ``urllib.error.HTTPError.code`` -- raised by the simple-image-gen
         backend through stdlib urllib.
      3. ``HTTP <NNN>`` regex scrape of the message -- catches strings
         raised by ``image_generation_models.py`` and the per-unit
         ``RuntimeError`` wrappers in this module.
    """
    try:
        import httpx as _httpx
    except ImportError:  # pragma: no cover -- httpx is a hard dep here.
        _httpx = None  # type: ignore[assignment]

    cause: BaseException | None = exc
    while cause is not None:
        if _httpx is not None and isinstance(cause, _httpx.HTTPStatusError):
            return {
                "material": unit_key,
                "type": "HTTPStatusError",
                "status": cause.response.status_code,
                "message": str(exc),
            }
        if isinstance(cause, HTTPError):
            return {
                "material": unit_key,
                "type": "HTTPError",
                "status": cause.code,
                "message": str(exc),
            }
        cause = cause.__cause__ or cause.__context__

    message = str(exc)
    status: int | None = None
    match = _HTTP_STATUS_RE.search(message)
    if match:
        status = int(match.group(1))

    return {
        "material": unit_key,
        "type": type(exc).__name__,
        "status": status,
        "message": message,
    }


class _BackendResultError(RuntimeError):
    """Service response normalization failed after producing diagnostics."""

    def __init__(self, message: str, backend_record: dict[str, Any]) -> None:
        super().__init__(message)
        self.backend_record = backend_record


def _backend_diagnostic(
    code: str,
    *,
    severity: str,
    unit: PrimTextureUnit | None,
    message: str,
    stage: str = "generate_textures",
    recommended_action: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a texture-agent-diagnostic.v1 payload for backend parsing."""
    prim_path = None
    material_name = None
    if unit is not None:
        prim_path = unit.prim_path or (
            unit.material_info.bound_prim_paths[0]
            if unit.material_info.bound_prim_paths
            else None
        )
        material_name = unit.material_info.name
    return {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": code,
        "severity": severity,
        "stage": stage,
        "prim_path": prim_path,
        "material_name": material_name,
        "message": message,
        "recommended_action": recommended_action,
        "details": details or {},
    }


def _diagnostic_key(diagnostic: dict[str, Any]) -> tuple[Any, ...]:
    details = diagnostic.get("details") or {}
    return (
        diagnostic.get("code"),
        diagnostic.get("severity"),
        diagnostic.get("prim_path"),
        diagnostic.get("material_name"),
        diagnostic.get("message"),
        tuple(sorted((str(k), str(v)) for k, v in details.items())),
    )


def _append_diagnostic_once(
    diagnostics: list[dict[str, Any]],
    diagnostic: dict[str, Any],
) -> None:
    key = _diagnostic_key(diagnostic)
    if any(_diagnostic_key(item) == key for item in diagnostics):
        return
    diagnostics.append(diagnostic)


def _has_diagnostic_code(diagnostics: list[dict[str, Any]], code: str) -> bool:
    return any(item.get("code") == code for item in diagnostics)


def _is_step1x_service_backend(texture_config: dict[str, Any]) -> bool:
    engine = str(texture_config.get("engine") or "").strip().lower()
    endpoint = str(texture_config.get("endpoint") or "").strip().lower()
    return engine == "step1x" or "step1x" in endpoint


def _normalized_backend_name(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _simple_image_gen_conditioning_capabilities(
    texture_config: dict[str, Any],
) -> dict[str, bool] | None:
    """Return supported media conditioning for a selected simple route."""
    backend = _normalized_backend_name(
        texture_config.get("backend", "simple_image_gen")
    )
    engine = _normalized_backend_name(texture_config.get("engine"))
    if backend == "service":
        if engine not in _SIMPLE_IMAGE_GEN_ENGINES:
            return None
        return {"image_conditioning": False, "multiview": False}
    if backend != "simple_image_gen":
        return None
    image_gen_config = texture_config.get("image_gen") or {}
    if not isinstance(image_gen_config, dict):
        return None
    provider = _normalized_backend_name(image_gen_config.get("backend") or "nim")
    return {
        "image_conditioning": provider != "nim",
        "multiview": False,
    }


def _allows_step1x_overlay_targets(texture_config: dict[str, Any]) -> bool:
    custom_parameters = texture_config.get("custom_parameters") or {}
    if not isinstance(custom_parameters, dict):
        return False
    return bool(custom_parameters.get("allow_step1x_overlay_targets"))


def _step1x_overlay_target_token(unit: PrimTextureUnit) -> str | None:
    identifiers = [
        unit.key,
        unit.prim_path,
        unit.material_info.name,
        unit.material_info.prim_path,
        *unit.material_info.bound_prim_paths,
    ]
    for identifier in identifiers:
        normalized = re.split(r"[^a-z0-9]+", str(identifier).lower())
        for token in _STEP1X_OVERLAY_TARGET_TOKENS:
            if token in normalized:
                return token
    return None


def _step1x_unsupported_overlay_record(
    unit: PrimTextureUnit,
    texture_config: dict[str, Any],
    *,
    matched_token: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = _service_target_for_unit(unit, texture_config)
    message = (
        "Step1X currently rejects decal/label/overlay targets before backend "
        "launch because the external PyTorch3D runtime can hit a CUDA "
        "device-side assert on this geometry class."
    )
    diagnostic = _backend_diagnostic(
        "STEP1X_UNSUPPORTED_OVERLAY_TARGET",
        severity="error",
        unit=unit,
        message=message,
        recommended_action=(
            "Exclude this material from Step1X requests or route the decal/overlay "
            "through a backend that supports tiny transparent overlay geometry."
        ),
        details={
            "matched_token": matched_token,
            "backend": "step1x",
            "material_path": unit.material_info.prim_path,
            "prim_paths": target.prim_paths,
            "override": "texture.custom_parameters.allow_step1x_overlay_targets=true",
        },
    )
    error = {
        "material": unit.key,
        "type": "UnsupportedStep1XTarget",
        "status": None,
        "message": message,
    }
    metadata = {
        "backend_name": "step1x",
        "target": {
            "material_name": target.material_name,
            "material_path": target.material_path,
            "prim_paths": target.prim_paths,
            "mode": target.mode,
            "strict_scope": target.strict_scope,
        },
        "skipped_before_backend_launch": True,
        "diagnostics": [diagnostic],
    }
    backend_record = {"maps": {}, "metadata": metadata, **metadata}
    return error, backend_record


def _preflight_step1x_targets(
    units: list[PrimTextureUnit],
    texture_config: dict[str, Any],
) -> tuple[
    list[PrimTextureUnit],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    if not _is_step1x_service_backend(texture_config) or _allows_step1x_overlay_targets(
        texture_config
    ):
        return units, [], {}, []

    supported: list[PrimTextureUnit] = []
    errors: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for unit in units:
        matched_token = _step1x_overlay_target_token(unit)
        if not matched_token:
            supported.append(unit)
            continue
        error, backend_record = _step1x_unsupported_overlay_record(
            unit,
            texture_config,
            matched_token=matched_token,
        )
        errors.append(error)
        metadata[unit.key] = backend_record
        diagnostics.extend(backend_record["diagnostics"])
    return supported, errors, metadata, diagnostics


def _external_authoring_spec_from_config(
    texture_config: dict[str, Any],
) -> ExternalAuthoringSpec | None:
    raw = texture_config.get("external_authoring")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("texture.external_authoring must be an object")
    if raw.get("enabled", True) is False:
        return None
    return ExternalAuthoringSpec.from_config(raw)


def _external_authoring_no_go_records(
    units: list[PrimTextureUnit],
    feasibility: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    diagnostics = [
        item for item in feasibility.get("diagnostics", []) if isinstance(item, dict)
    ]
    message = (
        diagnostics[0].get("message")
        if diagnostics
        else "External authoring feasibility gate returned no-go."
    )
    errors = [
        {
            "material": unit.key,
            "type": "ExternalAuthoringNoGo",
            "status": None,
            "message": message,
        }
        for unit in units
    ]
    metadata = {
        unit.key: {
            "maps": {},
            "auxiliary_artifacts": {},
            "metadata": {"external_authoring_preflight": feasibility},
            "capabilities": {},
            "degraded_channels": [],
            "diagnostics": diagnostics,
            "variant_asset_uri": "",
            "variant_name": unit.key,
            "endpoint": "",
        }
        for unit in units
    }
    return errors, metadata, diagnostics


def _preflight_external_authoring(
    units: list[PrimTextureUnit],
    texture_config: dict[str, Any],
) -> tuple[
    list[PrimTextureUnit],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    ExternalAuthoringSpec | None,
    ExternalAuthoringCapabilityReceipt | None,
    dict[str, Any] | None,
]:
    """Require a sanitized go receipt before any external DCC job launches."""
    try:
        spec = _external_authoring_spec_from_config(texture_config)
    except ValueError as exc:
        diagnostic = _backend_diagnostic(
            "EXTERNAL_AUTHORING_CONTRACT_INVALID",
            severity="error",
            unit=None,
            message="External authoring configuration is invalid.",
            recommended_action="Fix the versioned external_authoring contract.",
            details={"error": str(exc)},
        )
        feasibility = {
            "schema_version": EXTERNAL_AUTHORING_FEASIBILITY_SCHEMA_VERSION,
            "verdict": "no_go",
            "spec": None,
            "capabilities": None,
            "diagnostics": [diagnostic],
        }
        errors, metadata, diagnostics = _external_authoring_no_go_records(
            units, feasibility
        )
        return [], errors, metadata, diagnostics, None, None, feasibility
    if spec is None:
        return units, [], {}, [], None, None, None

    if texture_config.get("backend", "simple_image_gen") != "service":
        feasibility = unavailable_external_authoring_feasibility(
            spec,
            "texture.backend must be 'service' for external authoring",
        )
        errors, metadata, diagnostics = _external_authoring_no_go_records(
            units, feasibility
        )
        return [], errors, metadata, diagnostics, spec, None, feasibility
    endpoint = str(texture_config.get("endpoint") or "").strip()
    if not endpoint:
        feasibility = unavailable_external_authoring_feasibility(
            spec,
            "texture.endpoint is required for external authoring",
        )
        errors, metadata, diagnostics = _external_authoring_no_go_records(
            units, feasibility
        )
        return [], errors, metadata, diagnostics, spec, None, feasibility

    from texture_agent.functions.rest_client import RestTextureVariationClient

    client = RestTextureVariationClient(endpoint, timeout=60.0)
    try:
        receipt = client.preflight_external_authoring(spec)
    except RuntimeError as exc:
        feasibility = unavailable_external_authoring_feasibility(spec, str(exc))
        errors, metadata, diagnostics = _external_authoring_no_go_records(
            units, feasibility
        )
        return [], errors, metadata, diagnostics, spec, None, feasibility

    feasibility = evaluate_external_authoring_feasibility(spec, receipt)
    unseeded_units = [
        unit.key
        for unit in units
        if unit.seed is None and texture_config.get("seed") is None
    ]
    if feasibility["verdict"] == "go" and unseeded_units:
        feasibility["verdict"] = "no_go"
        feasibility["diagnostics"].append(
            _backend_diagnostic(
                "EXTERNAL_AUTHORING_REPRODUCIBILITY_UNSUPPORTED",
                severity="error",
                unit=None,
                message="External authoring requests require an explicit seed.",
                recommended_action=(
                    "Set texture.seed or use deterministic per-unit plan seeds."
                ),
                details={"unseeded_unit_ids": unseeded_units},
            )
        )
    if feasibility["verdict"] != "go":
        errors, metadata, diagnostics = _external_authoring_no_go_records(
            units, feasibility
        )
        return [], errors, metadata, diagnostics, spec, receipt, feasibility
    return units, [], {}, [], spec, receipt, feasibility


def _write_external_authoring_feasibility(
    working_dir: Path,
    feasibility: dict[str, Any],
) -> Path:
    """Publish the preflight verdict before a later pipeline failure can occur."""
    working_dir.mkdir(parents=True, exist_ok=True)
    output_path = working_dir / "external_authoring_feasibility.json"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=working_dir,
            prefix=".external_authoring_feasibility.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(feasibility, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return output_path


def _preflight_simple_image_gen_conditioning(
    units: list[PrimTextureUnit],
    context: dict[str, Any],
    texture_config: dict[str, Any],
) -> tuple[
    list[PrimTextureUnit],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Reject conditioning that the selected simple route cannot consume."""
    capabilities = _simple_image_gen_conditioning_capabilities(texture_config)
    if capabilities is None:
        return units, [], {}, []

    conditioning_by_key = _conditioning_by_unit_key(
        units,
        context,
        texture_config,
        validate_uris=False,
    )
    supported: list[PrimTextureUnit] = []
    errors: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for unit in units:
        conditioning = conditioning_by_key[unit.key]
        unsupported_fields: list[str] = []
        unsupported_reference_uri_schemes: list[str] = []
        if conditioning.reference_image_uris and not capabilities["image_conditioning"]:
            unsupported_fields.append("reference_image_uris")
        elif conditioning.reference_image_uris:
            # Direct providers receive Pillow images, not remote URI strings.
            # Keep remote references valid for projection services that can
            # materialize them, but fail this in-process route before loading a
            # model or starting any per-material generation work.
            for uri in conditioning.reference_image_uris:
                try:
                    _path_from_local_path_or_uri(uri)
                except RuntimeError:
                    scheme = urlparse(uri).scheme.lower() or "unknown"
                    unsupported_reference_uri_schemes.append(scheme)
            if unsupported_reference_uri_schemes:
                unsupported_fields.append("reference_image_uris")
        if conditioning.turntable_video_uri and not capabilities["multiview"]:
            unsupported_fields.append("turntable_video_uri")
        if conditioning.multiview_image_uris and not capabilities["multiview"]:
            unsupported_fields.append("multiview_image_uris")
        if _weathering_controls_for_unit(
            unit, context, texture_config
        ) is not None or prompt_requests_weathering(conditioning.text_prompt):
            unsupported_fields.append("weathering")
        if not unsupported_fields:
            supported.append(unit)
            continue

        target = _service_target_for_unit(unit, texture_config)
        if unsupported_reference_uri_schemes:
            message = (
                "Direct simple_image_gen providers require reference_image_uris "
                "to be local paths or file URIs; remote references are not "
                "materialized by this route."
            )
            recommended_action = (
                "Download the reference images to local storage and pass local "
                "paths or file URIs, or select a service backend that can "
                "materialize remote references."
            )
        elif "weathering" in unsupported_fields:
            message = (
                "simple_image_gen cannot enforce prompt-requested weathering "
                "masks or "
                "material-specific PBR correlation."
            )
            recommended_action = (
                "Select a service backend that advertises weathering and mask "
                "enforcement capabilities."
            )
        elif not capabilities["image_conditioning"]:
            message = (
                "simple_image_gen is a text-only backend and cannot use the "
                "requested reference, turntable, or multiview conditioning."
            )
            recommended_action = (
                "Remove the unsupported conditioning fields or select a backend "
                "that advertises the required capability."
            )
        else:
            message = (
                "simple_image_gen cannot use the requested turntable or "
                "multiview conditioning."
            )
            recommended_action = (
                "Remove the unsupported conditioning fields or select a backend "
                "that advertises the required capability."
            )
        diagnostic_details: dict[str, Any] = {
            "unsupported_fields": unsupported_fields,
            "backend": "simple_image_gen",
        }
        if unsupported_reference_uri_schemes:
            diagnostic_details["unsupported_reference_uri_schemes"] = sorted(
                set(unsupported_reference_uri_schemes)
            )
        diagnostic = _backend_diagnostic(
            "BACKEND_CONDITIONING_UNSUPPORTED",
            severity="error",
            unit=unit,
            message=message,
            recommended_action=recommended_action,
            details=diagnostic_details,
        )
        error = {
            "material": unit.key,
            "type": "UnsupportedBackendConditioning",
            "status": None,
            "message": message,
        }
        record_metadata = {
            "backend_name": "simple_image_gen",
            "target": {
                "material_name": target.material_name,
                "material_path": target.material_path,
                "prim_paths": target.prim_paths,
                "mode": target.mode,
                "strict_scope": target.strict_scope,
            },
            "capabilities": capabilities,
            "skipped_before_backend_launch": True,
            "diagnostics": [diagnostic],
        }
        errors.append(error)
        metadata[unit.key] = {
            "maps": {},
            "metadata": record_metadata,
            **record_metadata,
        }
        diagnostics.append(diagnostic)
    return supported, errors, metadata, diagnostics


def _service_target_for_unit(
    unit: PrimTextureUnit,
    texture_config: dict[str, Any],
) -> TextureTarget:
    """Build the normalized target scope for one texture generation unit."""
    strict_scope = bool(texture_config.get("strict_scope", True))
    mode = "per_prim" if unit.prim_path else "per_material"
    prim_paths = (
        [unit.prim_path] if unit.prim_path else unit.material_info.bound_prim_paths
    )
    return TextureTarget(
        material_name=unit.material_info.name,
        material_path=unit.material_info.prim_path,
        prim_paths=prim_paths,
        mode=mode,
        strict_scope=strict_scope,
    )


def _capabilities_from_config(texture_config: dict[str, Any]) -> BackendCapabilities:
    """Read optional backend capability hints from texture config."""
    raw = texture_config.get("capabilities") or {}
    if not isinstance(raw, dict):
        raw = {}
    return BackendCapabilities(
        image_conditioning=raw.get("image_conditioning"),
        multiview=raw.get("multiview"),
        normal_map=raw.get("normal_map"),
        orm=raw.get("orm"),
        masks=raw.get("masks"),
        coverage=raw.get("coverage"),
        weathering=raw.get("weathering"),
        geometry_output=raw.get("geometry_output"),
    )


def _custom_parameters_for_unit(
    texture_config: dict[str, Any],
    unit: PrimTextureUnit,
) -> dict[str, Any]:
    """Return backend custom parameters with typed policy metadata.

    ``detail_policy`` and ``forbidden_details`` are reserved here so generic
    backend parameters cannot contradict the typed material policy.
    """
    raw = texture_config.get("custom_parameters") or {}
    custom_parameters = dict(raw) if isinstance(raw, dict) else {}
    custom_parameters.pop("detail_policy", None)
    custom_parameters.pop("forbidden_details", None)
    if unit.detail_policy != DETAIL_POLICY_DEFAULT:
        custom_parameters["detail_policy"] = unit.detail_policy
    if unit.detail_policy == DETAIL_POLICY_SURFACE_ONLY:
        custom_parameters["forbidden_details"] = list(SURFACE_ONLY_FORBIDDEN_DETAILS)
    return custom_parameters


def _as_string_list(value: Any) -> list[str]:
    """Normalize a config value into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list | tuple):
        values = list(value)
    else:
        raise ValueError(
            "Reference/conditioning URI fields must be a string or list of strings"
        )
    normalized: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError(
                "Reference/conditioning URI fields must contain only strings"
            )
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    return normalized


def _validate_conditioning_uri(uri: str, *, field_name: str) -> None:
    if _is_windows_drive_path(unquote(uri)):
        if not Path(unquote(uri)).exists():
            raise FileNotFoundError(f"{field_name} local file does not exist: {uri}")
        return

    parsed = urlparse(uri)
    if parsed.scheme not in _SUPPORTED_REFERENCE_URI_SCHEMES:
        raise ValueError(
            f"Unsupported URI scheme '{parsed.scheme}' in {field_name}: {uri}"
        )
    if parsed.scheme in {"", "file"}:
        try:
            path = _path_from_local_path_or_uri(uri)
        except RuntimeError as exc:
            raise ValueError(
                f"Unsupported file URI host '{parsed.netloc}' in {field_name}: {uri}"
            ) from exc
        if not path.exists():
            raise FileNotFoundError(f"{field_name} local file does not exist: {uri}")


def _validate_conditioning_uris(uris: list[str], *, field_name: str) -> None:
    for uri in uris:
        _validate_conditioning_uri(uri, field_name=field_name)


def _material_texture_spec(context: dict[str, Any], unit: PrimTextureUnit) -> dict:
    raw_specs = context.get("material_textures") or {}
    resolved = resolve_material_texture_spec(unit.material_info, raw_specs)
    if resolved is not None:
        return resolved[1]
    # Retain the pre-plan runtime-key fallback for callers that supplied
    # specs under a legacy unit key rather than a material identity.
    spec = raw_specs.get(unit.key)
    return spec if isinstance(spec, dict) else {}


def _weathering_controls_for_unit(
    unit: PrimTextureUnit,
    context: dict[str, Any],
    texture_config: dict[str, Any],
) -> WeatheringControls | None:
    """Return optional precision masks for a prompt-first weathering request."""
    material_spec = _material_texture_spec(context, unit)
    raw = material_spec.get("weathering", texture_config.get("weathering"))
    if raw is None:
        return None
    try:
        return WeatheringControls.model_validate(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid weathering controls for {unit.key}: {exc}") from exc


def _conditioning_for_unit(
    unit: PrimTextureUnit,
    context: dict[str, Any],
    texture_config: dict[str, Any],
    *,
    validate_uris: bool = True,
) -> Conditioning:
    """Merge global and material-specific conditioning fields for a unit."""
    spec = _material_texture_spec(context, unit)
    reference_uris = [
        *_as_string_list(texture_config.get("reference_image_uris")),
        *_as_string_list(spec.get("reference_image_uris")),
    ]
    multiview_uris = [
        *_as_string_list(texture_config.get("multiview_image_uris")),
        *_as_string_list(spec.get("multiview_image_uris")),
    ]
    turntable_video_uri = spec.get("turntable_video_uri") or texture_config.get(
        "turntable_video_uri"
    )
    turntable_video_uri = (
        str(turntable_video_uri).strip() if turntable_video_uri else None
    )

    if validate_uris:
        _validate_conditioning_uris(
            reference_uris,
            field_name="reference_image_uris",
        )
        _validate_conditioning_uris(
            multiview_uris,
            field_name="multiview_image_uris",
        )
    if validate_uris and turntable_video_uri:
        _validate_conditioning_uri(
            turntable_video_uri, field_name="turntable_video_uri"
        )

    return Conditioning(
        text_prompt=unit.prompt,
        reference_image_uris=reference_uris,
        turntable_video_uri=turntable_video_uri,
        multiview_image_uris=multiview_uris,
    )


def _conditioning_by_unit_key(
    units: list[PrimTextureUnit],
    context: dict[str, Any],
    texture_config: dict[str, Any],
    *,
    validate_uris: bool = True,
) -> dict[str, Conditioning]:
    """Build conditioning for all units before any backend request is sent."""
    return {
        unit.key: _conditioning_for_unit(
            unit,
            context,
            texture_config,
            validate_uris=validate_uris,
        )
        for unit in units
    }


def _prepared_usd_from_uv_report(context: dict[str, Any]) -> str | None:
    """Read the prepared USD written by prepare_uvs, if the report is present."""
    uv_preparation = context.get("uv_preparation")
    if not isinstance(uv_preparation, dict):
        return None

    report_path = uv_preparation.get("uv_report_path")
    if not isinstance(report_path, str) or not report_path.strip():
        return None
    report_path = report_path.strip()

    if _is_windows_drive_path(report_path):
        path = Path(report_path)
    else:
        parsed = urlparse(report_path)
        if parsed.scheme and parsed.scheme != "file":
            return None
        path = Path(unquote(parsed.path) if parsed.scheme == "file" else report_path)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.debug("Unable to read UV preparation report: %s", path, exc_info=True)
        return None

    prepared_usd = payload.get("prepared_usd")
    if isinstance(prepared_usd, str) and prepared_usd.strip():
        return prepared_usd.strip()
    return None


def _path_or_uri_to_uri(raw: str) -> str:
    if not raw:
        return ""
    if _is_windows_drive_path(raw):
        return PureWindowsPath(raw).as_uri()
    if urlparse(raw).scheme:
        return raw
    return Path(raw).resolve().as_uri()


def _service_source_asset_uri(context: dict[str, Any]) -> str:
    """Return the prepared USD as a URI for projection backend requests."""
    for candidate in (
        _prepared_usd_from_uv_report(context),
        context.get("prepared_usd_path"),
        context.get("prepared_usd"),
        context.get("usd_path"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return _path_or_uri_to_uri(candidate.strip())
    return ""


def _service_source_asset_uri_for_unit(
    context: dict[str, Any],
    unit: PrimTextureUnit,
    texture_config: dict[str, Any],
    out_dir: Path,
) -> str:
    source_asset_uri = _service_source_asset_uri(context)
    if not _should_rebake_source_textures(context, unit, texture_config):
        return source_asset_uri

    rebaked_usd = _rebake_unit_source_textures(context, unit, texture_config, out_dir)
    return rebaked_usd.resolve().as_uri()


def _should_rebake_source_textures(
    context: dict[str, Any],
    unit: PrimTextureUnit,
    texture_config: dict[str, Any],
) -> bool:
    if not bool(texture_config.get("uv_rebake_source_albedo", False)):
        return False

    uv_preparation = context.get("uv_preparation")
    if not isinstance(uv_preparation, dict):
        return False
    if uv_preparation.get("uv_scope") != "target_prims":
        return False
    if not unit.material_info.base_color_texture:
        return False

    changed_targets = _normalized_path_set(uv_preparation.get("target_prim_paths"))
    if not changed_targets:
        return False
    unit_targets = _normalized_path_set(
        _service_target_for_unit(unit, texture_config).prim_paths
    )
    return any(
        _paths_overlap(unit_path, changed_path)
        for unit_path in unit_targets
        for changed_path in changed_targets
    )


def _normalized_path_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list | tuple | set)):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    return {path.strip().rstrip("/") for path in values if path.strip()}


def _paths_overlap(a: str, b: str) -> bool:
    return a == b or a.startswith(f"{b}/") or b.startswith(f"{a}/")


# Packed-ORM components restored from the asset's authored map. Roughness (G) is
# deliberately NOT preserved: it is the channel a weathering edit should legitimately
# move, and the only one measured to correlate with the requested effect. Occlusion
# and metallic are restored because neither generator derives them from evidence --
# ``simple_image_gen`` hardcodes R=255/B=0, and Material Anything receives no ORM
# input at all. See issue #950.
_PRESERVED_ORM_COMPONENTS = ("occlusion", "metallic")

# Channel index within the packed ORM (R=occlusion, G=roughness, B=metallic).
_ORM_COMPONENT_CHANNELS = {"occlusion": 0, "roughness": 1, "metallic": 2}

# Components whose authored band must be scaled with nearest-neighbour rather than
# LANCZOS. Metallic is typically an authored near-binary mask, and LANCZOS overshoots
# and rings at its edges, inventing intermediate metalness that never existed.
# Occlusion is a smooth field and interpolates cleanly. Named rather than holding
# PIL enums because PIL is imported lazily inside the functions below.
_ORM_COMPONENTS_NEEDING_NEAREST = frozenset({"metallic"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WEATHERING_RECEIPT_DIGESTS = (
    "source_albedo_sha256",
    "candidate_albedo_sha256",
    "candidate_orm_sha256",
    "final_albedo_sha256",
    "final_orm_sha256",
    "weathering_mask_sha256",
    "source_usd_sha256",
    "request_sha256",
)


def _local_artifact_sha256(uri_or_path: Any) -> str | None:
    if not isinstance(uri_or_path, str) or not uri_or_path:
        return None
    try:
        path = _path_from_local_path_or_uri(uri_or_path)
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, RuntimeError, ValueError):
        return None


def _passing_rust_receipt_matches_result(
    context: dict[str, Any],
    unit_key: str,
    record: dict[str, Any],
    evidence: dict[str, Any],
) -> bool:
    if (
        evidence.get("schema_version") != "texture-weathering-evidence.v1"
        or evidence.get("effect") != "rust"
        or evidence.get("status") != "pass"
        or evidence.get("failures") != []
        or not isinstance(evidence.get("metrics"), dict)
        or not isinstance(evidence.get("thresholds"), dict)
    ):
        return False
    internal_plan = evidence.get("internal_plan")
    if not isinstance(internal_plan, dict) or internal_plan.get("effect") != "rust":
        return False
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    if any(
        not isinstance(artifacts.get(key), str)
        or _SHA256_RE.fullmatch(artifacts[key]) is None
        for key in _WEATHERING_RECEIPT_DIGESTS
    ):
        return False
    if _local_artifact_sha256(record.get("source_asset_uri")) != artifacts.get(
        "source_usd_sha256"
    ):
        return False

    generated = context.get("generated_textures")
    textures = generated.get(unit_key) if isinstance(generated, dict) else None
    if textures is None:
        return False
    if _local_artifact_sha256(getattr(textures, "albedo", None)) != artifacts.get(
        "final_albedo_sha256"
    ):
        return False
    if _local_artifact_sha256(getattr(textures, "orm", None)) != artifacts.get(
        "final_orm_sha256"
    ):
        return False

    auxiliary = record.get("auxiliary_artifacts")
    masks = auxiliary.get("masks") if isinstance(auxiliary, dict) else None
    weathering_mask = masks.get("weathering") if isinstance(masks, dict) else None
    if not isinstance(weathering_mask, dict):
        return False
    mask_digest = artifacts.get("weathering_mask_sha256")
    if weathering_mask.get("sha256") != mask_digest:
        return False
    return _local_artifact_sha256(weathering_mask.get("uri")) == mask_digest


def _preserved_orm_components_for_result(
    context: dict[str, Any],
    unit_key: str,
) -> tuple[str, ...]:
    """Allow localized rust metallic only when its quality receipt passed."""
    records = context.get("projection_backend_results")
    if not isinstance(records, dict):
        return _PRESERVED_ORM_COMPONENTS
    record = records.get(unit_key)
    if not isinstance(record, dict):
        return _PRESERVED_ORM_COMPONENTS
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return _PRESERVED_ORM_COMPONENTS
    evidence = metadata.get("weathering")
    if isinstance(evidence, dict) and _passing_rust_receipt_matches_result(
        context,
        unit_key,
        record,
        evidence,
    ):
        return ("occlusion",)
    return _PRESERVED_ORM_COMPONENTS


# Counter fields prepare_uvs reports when it touched UVs. These must stay in sync
# with the action dicts built in tasks/prepare_uvs.py -- BOTH the python fallback
# and the scene_optimizer path emit exactly these four names. A field missing from
# this tuple means the guard below silently never fires for that mutation and
# authored texels get composited onto a moved atlas, so
# ``test_guard_covers_every_prepare_uvs_mutation_field`` pins the list.
_UV_MUTATION_COUNT_FIELDS = (
    "generated",
    "fixed_interpolation",
    "degenerate_repaired",
    "normalized",
)


def _uv_layout_changed_for_unit(
    context: dict[str, Any],
    unit: PrimTextureUnit,
) -> bool:
    """True when prepare_uvs may have moved this unit's UV layout.

    An authored ORM is only meaningful in the atlas it was authored against, so
    preservation must be skipped whenever UVs were generated or normalized for the
    unit's prims -- otherwise authored texels land on unrelated surface. Conservative
    by design: an unattributable mutation counts as changed.
    """
    uv_preparation = context.get("uv_preparation")
    if not isinstance(uv_preparation, dict):
        return False

    def _count(field: str) -> int:
        # uv_preparation is untyped upstream context, not a validated model. A
        # bare int() would raise on a string or list and abort the whole step.
        value = uv_preparation.get(field)
        return int(value) if isinstance(value, int | float) else 0

    mutated = any(_count(field) > 0 for field in _UV_MUTATION_COUNT_FIELDS)
    if not mutated:
        return False

    # Stage-wide scope means the mutation cannot be attributed, so assume it hit.
    if uv_preparation.get("uv_scope") != "target_prims":
        return True

    changed_targets = _normalized_path_set(uv_preparation.get("target_prim_paths"))
    if not changed_targets:
        return True

    # Subsets matter: a material bound through a `materialBind` GeomSubset has an
    # EMPTY bound_prim_paths, and in per_material mode an empty prim_path too. Without
    # the subset paths, unit_targets would be empty, `any()` over nothing would be
    # False, and the guard would report "UVs did not move" for a mesh whose UVs were
    # fully regenerated -- failing open on exactly the case it exists to catch.
    unit_targets = _normalized_path_set(
        [
            unit.prim_path,
            *unit.material_info.bound_prim_paths,
            *unit.material_info.bound_subset_paths,
        ]
    )
    if not unit_targets:
        # Nothing to attribute the mutation against; stay conservative.
        return True
    return any(
        _paths_overlap(unit_path, changed_path)
        for unit_path in unit_targets
        for changed_path in changed_targets
    )


def _preserve_authored_orm(
    *,
    new_generated: dict[str, Any],
    units: list[PrimTextureUnit],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Restore authored occlusion/metallic over each freshly generated ORM.

    Runs once for all backends rather than inside each, so ``service`` and
    ``simple_image_gen`` are covered by a single call site. Idempotent: applying it
    to an already-composited map is a no-op.
    """
    from PIL import Image

    diagnostics: list[dict[str, Any]] = []
    units_by_key = {unit.key: unit for unit in units}
    usd_path = context.get("usd_path")
    if not usd_path:
        return diagnostics
    base_usd_path = Path(str(usd_path))

    for key, textures in new_generated.items():
        generated_orm = getattr(textures, "orm", None)
        unit = units_by_key.get(key)
        if unit is None or not generated_orm:
            continue
        authored_ref = unit.material_info.orm_texture
        if not authored_ref:
            continue
        if _uv_layout_changed_for_unit(context, unit):
            logger.info(
                "Skipping authored ORM preservation for %s: UV layout was regenerated",
                key,
            )
            _append_diagnostic_once(
                diagnostics,
                _backend_diagnostic(
                    "AUTHORED_ORM_NOT_PRESERVED",
                    severity="warning",
                    unit=unit,
                    stage="blend_textures",
                    message=(
                        "Authored ORM was not preserved: prepare_uvs changed this "
                        "unit's UV layout, so authored texels no longer correspond "
                        "to the atlas."
                    ),
                    recommended_action=(
                        "Re-author the ORM against the prepared UVs, or run with a "
                        "UV policy that leaves the authored layout intact."
                    ),
                    details={"reason": "uv_layout_regenerated"},
                ),
            )
            continue

        authored_path = _resolve_texture_path(authored_ref, base_usd_path=base_usd_path)
        if authored_path is None or not authored_path.exists():
            logger.warning(
                "Authored ORM for %s did not resolve to a readable file: %r",
                key,
                authored_ref,
            )
            _append_diagnostic_once(
                diagnostics,
                _backend_diagnostic(
                    "AUTHORED_ORM_NOT_PRESERVED",
                    severity="warning",
                    unit=unit,
                    stage="blend_textures",
                    message=(
                        "Authored ORM was not preserved: the authored reference did "
                        "not resolve to a readable file."
                    ),
                    recommended_action=(
                        "Check that the ORM texture ships with the asset and is "
                        "reachable from the stage."
                    ),
                    details={
                        "reason": "authored_orm_unresolvable",
                        "authored_ref": str(authored_ref),
                    },
                ),
            )
            continue

        try:
            preserved_components = _preserved_orm_components_for_result(context, key)
            generated_path = _path_from_local_path_or_uri(str(generated_orm))
            with Image.open(generated_path) as gen_img:
                generated_rgb = gen_img.convert("RGB")
                generated_size = generated_rgb.size
            with Image.open(authored_path) as authored_img:
                authored_rgb = authored_img.convert("RGB")
                authored_size = authored_rgb.size

            # Resolve to the LARGER of the two so neither input is ever
            # downsampled. Downsampling the authored map would smooth away the
            # very occlusion/metallic detail this is protecting, while
            # upsampling is information-preserving. Equal sizes resample
            # nothing.
            #
            # Pick whichever map has more pixels and adopt ITS size rather than a
            # per-axis max: with differing aspect ratios (4096x1024 authored vs
            # 1024x4096 generated) a per-axis max would be 4096x4096 and stretch
            # both, corrupting the UV correspondence for every texel.
            target_size = max(generated_size, authored_size, key=lambda s: s[0] * s[1])
            if generated_size != target_size:
                generated_rgb = generated_rgb.resize(
                    target_size, Image.Resampling.LANCZOS
                )

            bands = list(generated_rgb.split())
            authored_bands = list(authored_rgb.split())
            for component in preserved_components:
                channel = _ORM_COMPONENT_CHANNELS[component]
                band = authored_bands[channel]
                if authored_size != target_size:
                    band = band.resize(
                        target_size,
                        Image.Resampling.NEAREST
                        if component in _ORM_COMPONENTS_NEEDING_NEAREST
                        else Image.Resampling.LANCZOS,
                    )
                bands[channel] = band
            Image.merge("RGB", bands).save(generated_path)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("Could not preserve authored ORM for %s: %s", key, exc)
            continue

        _append_diagnostic_once(
            diagnostics,
            _backend_diagnostic(
                "AUTHORED_ORM_PRESERVED",
                severity="info",
                unit=unit,
                stage="blend_textures",
                message=(
                    "Restored authored packed-ORM components over the generated "
                    "ORM; verified localized rust metallic is retained only when "
                    "its weathering receipt passed."
                ),
                details={
                    "preserved_components": list(preserved_components),
                    "localized_components": [
                        component
                        for component in _PRESERVED_ORM_COMPONENTS
                        if component not in preserved_components
                    ],
                    "source": "preserved_input",
                    "authored_orm": str(authored_path),
                    "authored_size": list(authored_size),
                    "generated_size": list(generated_size),
                    "output_size": list(target_size),
                },
            ),
        )

    return diagnostics


def _rebake_unit_source_textures(
    context: dict[str, Any],
    unit: PrimTextureUnit,
    texture_config: dict[str, Any],
    out_dir: Path,
) -> Path:
    uv_report = _read_uv_report_payload(context)
    if uv_report is None:
        raise RuntimeError(
            "texture.uv_rebake_source_albedo requires prepare_uvs to write a "
            "readable uv_report.json"
        )

    source_usd = _path_from_local_path_or_uri(str(uv_report.get("input_usd") or ""))
    prepared_usd = _path_from_local_path_or_uri(
        str(uv_report.get("prepared_usd") or "")
    )
    source_texture_refs = _find_unit_source_texture_refs(source_usd, unit)
    source_textures = {
        channel: _resolve_texture_path(texture_ref, base_usd_path=source_usd)
        for channel, texture_ref in source_texture_refs.items()
    }
    albedo_texture = source_textures.get("albedo")
    if albedo_texture is None or not albedo_texture.exists():
        raise RuntimeError(
            "texture.uv_rebake_source_albedo could not resolve source albedo "
            f"for {unit.key}: {unit.material_info.base_color_texture!r}"
        )

    rebake_dir = out_dir / "rebaked_source_textures"
    rebake_dir.mkdir(parents=True, exist_ok=True)
    rebaked_textures: dict[str, Path] = {}
    target_prim_paths = _service_target_for_unit(unit, texture_config).prim_paths
    for channel in ("albedo", "normal", "orm"):
        source_texture = source_textures.get(channel)
        if source_texture is None or not source_texture.exists():
            continue
        rebaked_texture = rebake_dir / f"{unit.key}_source_{channel}_rebaked.png"
        _rebake_texture_between_uv_sets(
            source_usd=source_usd,
            prepared_usd=prepared_usd,
            prim_paths=target_prim_paths,
            source_texture=source_texture,
            output_path=rebaked_texture,
            output_size=_positive_int(texture_config.get("uv_rebake_size")),
        )
        rebaked_textures[channel] = rebaked_texture

    source_dir = out_dir / "rebaked_source_assets"
    source_dir.mkdir(parents=True, exist_ok=True)
    rebaked_usd = source_dir / f"{unit.key}_source_asset.usda"
    _author_unit_source_usd(
        prepared_usd=prepared_usd,
        output_usd=rebaked_usd,
        material_path=unit.material_info.prim_path,
        texture_paths=rebaked_textures,
        required_prim_paths=target_prim_paths,
    )
    logger.info(
        "Rebaked %s source texture map(s) for %s after scoped UV preparation",
        len(rebaked_textures),
        unit.key,
    )
    return rebaked_usd


def _read_uv_report_payload(context: dict[str, Any]) -> dict[str, Any] | None:
    uv_preparation = context.get("uv_preparation")
    if not isinstance(uv_preparation, dict):
        return None
    report_path = uv_preparation.get("uv_report_path")
    if not isinstance(report_path, str) or not report_path.strip():
        return None
    path = _path_from_local_path_or_uri(report_path.strip())
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _path_from_local_path_or_uri(raw: str) -> Path:
    if not raw:
        return Path()
    parsed = urlparse(raw)
    if parsed.scheme == "file":
        return _file_uri_path(parsed)
    if parsed.scheme and not _is_windows_drive_path(raw):
        raise RuntimeError(f"Only local file paths are supported: {raw}")
    path = Path(raw)
    if path.exists():
        return path
    decoded = unquote(raw)
    if decoded != raw:
        decoded_path = Path(decoded)
        if decoded_path.exists():
            return decoded_path
    return path


def _resolve_texture_path(
    texture_ref: str | None, *, base_usd_path: Path
) -> Path | None:
    if not texture_ref:
        return None
    package_member = _extract_package_member_texture_ref(
        texture_ref,
        base_usd_path=base_usd_path,
    )
    if package_member is not None:
        return package_member

    try:
        path = _path_from_local_path_or_uri(texture_ref)
    except RuntimeError:
        return None
    if not path.is_absolute() and base_usd_path.suffix.lower() == ".usdz":
        member = texture_ref.replace("\\", "/").lstrip("./")
        package_texture = _extract_usdz_member(base_usd_path, member)
        if package_texture is not None:
            return package_texture
    if not path.is_absolute():
        path = base_usd_path.parent / path
    return path.expanduser().resolve() if path.exists() else path.expanduser()


def _extract_package_member_texture_ref(
    texture_ref: str,
    *,
    base_usd_path: Path,
) -> Path | None:
    package_member = split_package_member_asset_path(texture_ref)
    if package_member is None:
        return None
    package_ref, member_name = package_member
    package_path = _resolve_package_path(package_ref, base_usd_path.parent)
    return _extract_usdz_member(package_path, member_name)


def _resolve_package_path(package_ref: str, base_dir: Path) -> Path:
    return resolve_local_package_path(package_ref, base_dir)


def _extract_usdz_member(package_path: Path, member_name: str) -> Path | None:
    extract_root = (
        package_path.parent / ".texture_agent_usdz_assets" / package_path.stem
    )
    try:
        return extract_usdz_member_to_dir(
            package_path,
            member_name,
            extract_root,
            allowed_suffixes=_REFERENCE_TEXTURE_EXTENSIONS,
            max_bytes=_MAX_PACKAGE_TEXTURE_BYTES,
        )
    except ArchiveSizeLimitExceeded:
        logger.warning(
            "Skipped USDZ texture member %s[%s] because it exceeds %d bytes",
            package_path,
            member_name,
            _MAX_PACKAGE_TEXTURE_BYTES,
        )
        return None
    except OSError:
        return None


def _find_unit_source_texture_refs(
    source_usd: Path,
    unit: PrimTextureUnit,
) -> dict[str, str]:
    refs: dict[str, str] = {}
    if unit.material_info.base_color_texture:
        refs["albedo"] = unit.material_info.base_color_texture

    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        return refs
    material_prim = stage.GetPrimAtPath(unit.material_info.prim_path)
    if not material_prim:
        return refs

    for prim in _iter_rebake_material_subtree(material_prim):
        shader = UsdShade.Shader(prim) if prim.IsA(UsdShade.Shader) else None
        for attr in prim.GetAttributes():
            texture_ref = _coerce_rebake_texture_ref(attr.Get())
            if texture_ref is None:
                continue
            channel = _rebake_channel_from_name(attr.GetName())
            if channel is None and shader is not None:
                base_name = attr.GetBaseName()
                if base_name.lower() in {"file", "filename"}:
                    channel = _rebake_channel_from_name(prim.GetName())
            if channel is not None:
                refs.setdefault(channel, texture_ref)
    return refs


def _iter_rebake_material_subtree(prim: Any) -> list[Any]:
    prims = [prim]
    for child in prim.GetChildren():
        prims.extend(_iter_rebake_material_subtree(child))
    return prims


def _coerce_rebake_texture_ref(value: object) -> str | None:
    if value is None:
        return None
    if hasattr(value, "path"):
        resolved_path = str(getattr(value, "resolvedPath", "") or "")
        texture_ref = resolved_path or str(value.path)
    elif isinstance(value, str):
        texture_ref = value
    else:
        return None
    if not texture_ref or texture_ref == "@@":
        return None
    suffix_source = texture_ref
    package_member = split_package_member_asset_path(texture_ref)
    if package_member is not None:
        suffix_source = package_member[1]
    suffix = Path(urlparse(suffix_source).path).suffix.lower()
    if suffix not in {".bmp", ".exr", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        return None
    return texture_ref


def _rebake_channel_from_name(name: str) -> str | None:
    normalized = name.rsplit(":", 1)[-1].replace("-", "_").lower()
    compact = normalized.replace("_", "")
    for channel, names in _REBAKE_TEXTURE_INPUTS.items():
        compact_names = {item.replace("_", "") for item in names}
        if normalized in names or compact in compact_names:
            return channel
    return None


def _rebake_texture_between_uv_sets(
    *,
    source_usd: Path,
    prepared_usd: Path,
    prim_paths: list[str],
    source_texture: Path,
    output_path: Path,
    output_size: int | None,
) -> None:
    from PIL import Image
    from pxr import Usd, UsdGeom

    source_stage = Usd.Stage.Open(str(source_usd))
    prepared_stage = Usd.Stage.Open(str(prepared_usd))
    if not source_stage or not prepared_stage:
        raise RuntimeError(
            "Failed to open source/prepared USD for source texture rebake"
        )

    with Image.open(source_texture) as image:
        source_image = image.convert("RGB")
    size = output_size or max(source_image.size)
    fill = tuple(
        int(channel) for channel in source_image.resize((1, 1)).getpixel((0, 0))
    )
    dest_image = Image.new("RGB", (size, size), fill)

    for prim_path in prim_paths:
        source_prim = source_stage.GetPrimAtPath(prim_path)
        prepared_prim = prepared_stage.GetPrimAtPath(prim_path)
        if not source_prim or not prepared_prim:
            continue
        source_mesh = UsdGeom.Mesh(source_prim)
        prepared_mesh = UsdGeom.Mesh(prepared_prim)
        face_counts = list(source_mesh.GetFaceVertexCountsAttr().Get() or [])
        if face_counts != list(prepared_mesh.GetFaceVertexCountsAttr().Get() or []):
            raise RuntimeError(
                f"Cannot rebake source albedo for {prim_path}: topology changed"
            )
        old_uvs = _face_varying_uvs(source_mesh)
        new_uvs = _face_varying_uvs(prepared_mesh)
        if len(old_uvs) != len(new_uvs):
            raise RuntimeError(
                f"Cannot rebake source albedo for {prim_path}: UV counts differ"
            )
        offset = 0
        for count in face_counts:
            if count < 3:
                offset += count
                continue
            for i in range(1, count - 1):
                old_tri = [
                    old_uvs[offset],
                    old_uvs[offset + i],
                    old_uvs[offset + i + 1],
                ]
                new_tri = [
                    new_uvs[offset],
                    new_uvs[offset + i],
                    new_uvs[offset + i + 1],
                ]
                _paste_rebaked_triangle(source_image, dest_image, old_tri, new_tri)
            offset += count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dest_image.save(output_path)


def _face_varying_uvs(mesh: Any) -> list[Any]:
    from pxr import UsdGeom

    prim = mesh.GetPrim()
    st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    if not st or not st.HasValue():
        raise RuntimeError(f"Mesh has no st primvar: {prim.GetPath()}")
    values = list(st.Get() or [])
    if not values:
        raise RuntimeError(f"Mesh has empty st primvar: {prim.GetPath()}")
    face_vertex_indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    indices = list(st.GetIndices() or [])
    if indices:
        if len(indices) != len(face_vertex_indices):
            raise RuntimeError(f"Indexed st count mismatch: {prim.GetPath()}")
        return [values[int(index)] for index in indices]
    if len(values) == len(face_vertex_indices):
        return values
    points = list(mesh.GetPointsAttr().Get() or [])
    if len(values) == len(points):
        return [values[int(index)] for index in face_vertex_indices]
    raise RuntimeError(f"Unsupported st interpolation/count on {prim.GetPath()}")


def _paste_rebaked_triangle(
    source_image: Any,
    dest_image: Any,
    old_uvs: list[Any],
    new_uvs: list[Any],
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    dst_size = dest_image.size[0]
    src_w, src_h = source_image.size
    src = np.array([_uv_to_pixel(uv, src_w, src_h) for uv in old_uvs], dtype=float)
    dst = np.array(
        [_uv_to_pixel(uv, dst_size, dst_size) for uv in new_uvs], dtype=float
    )

    x_min = max(0, int(np.floor(dst[:, 0].min())))
    y_min = max(0, int(np.floor(dst[:, 1].min())))
    x_max = min(dst_size - 1, int(np.ceil(dst[:, 0].max())))
    y_max = min(dst_size - 1, int(np.ceil(dst[:, 1].max())))
    if x_max <= x_min or y_max <= y_min:
        return

    matrix = np.array(
        [
            [dst[0, 0], dst[0, 1], 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, dst[0, 0], dst[0, 1], 1.0],
            [dst[1, 0], dst[1, 1], 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, dst[1, 0], dst[1, 1], 1.0],
            [dst[2, 0], dst[2, 1], 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, dst[2, 0], dst[2, 1], 1.0],
        ],
        dtype=float,
    )
    rhs = np.array(
        [src[0, 0], src[0, 1], src[1, 0], src[1, 1], src[2, 0], src[2, 1]],
        dtype=float,
    )
    try:
        a, b, c, d, e, f = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    transform = (
        a,
        b,
        c + a * x_min + b * y_min,
        d,
        e,
        f + d * x_min + e * y_min,
    )
    affine_mode = getattr(Image, "Transform", Image).AFFINE
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    patch = source_image.transform((width, height), affine_mode, transform, resampling)
    mask = Image.new("L", (width, height), 0)
    localized = [(float(x - x_min), float(y - y_min)) for x, y in dst]
    ImageDraw.Draw(mask).polygon(localized, fill=255)
    dest_image.paste(patch, (x_min, y_min), mask)


def _uv_to_pixel(uv: Any, width: int, height: int) -> tuple[float, float]:
    u = max(0.0, min(1.0, float(uv[0])))
    v = max(0.0, min(1.0, float(uv[1])))
    return u * float(width - 1), (1.0 - v) * float(height - 1)


def _author_unit_source_usd(
    *,
    prepared_usd: Path,
    output_usd: Path,
    material_path: str,
    texture_paths: dict[str, Path],
    required_prim_paths: Sequence[str] = (),
) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.Open(str(prepared_usd))
    if stage is None:
        raise RuntimeError(f"Failed to open prepared USD: {prepared_usd}")
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_usd.stem}.",
        dir=output_usd.parent,
    ) as temporary_dir:
        # Let USD create this file so it receives normal umask-derived
        # permissions. NamedTemporaryFile would publish its private 0600 mode
        # when the inode is atomically moved to output_usd.
        temporary_output = Path(temporary_dir) / output_usd.name
        # Export the composed stage, not the root layer. A layered or packaged
        # asset reaches its geometry through relative arcs, and exporting only
        # the root layer into a different directory leaves those arcs dangling.
        # USD drops an unresolved sublayer with a warning rather than an error,
        # so the result can open cleanly and contain nothing. See issue #957.
        flattened = stage.Flatten(addSourceFileComment=False)
        if not flattened.Export(str(temporary_output)):
            raise RuntimeError(f"Failed to export rebaked source USD: {output_usd}")

        stage = Usd.Stage.Open(str(temporary_output))
        if stage is None:
            raise RuntimeError(f"Failed to open rebaked source USD: {output_usd}")
        missing_prim_paths = [
            prim_path
            for prim_path in required_prim_paths
            if not stage.GetPrimAtPath(prim_path)
        ]
        if missing_prim_paths:
            raise RuntimeError(
                "Rebaked source USD lost required target prim path(s): "
                + ", ".join(missing_prim_paths)
            )

        material_prim = stage.GetPrimAtPath(material_path)
        if not material_prim:
            raise RuntimeError(f"Material path not found for rebake: {material_path}")
        if material_prim.IsInstanceProxy():
            raise RuntimeError(
                "Material path is an instance proxy and cannot be modified: "
                f"{material_path}"
            )
        if material_prim.IsInstance() or material_prim.IsInstanceable():
            material_prim.SetInstanceable(False)

        resolved_refs = {
            channel: str(path.resolve()) for channel, path in texture_paths.items()
        }
        for channel, attr_names in _REBAKE_AUTHOR_TEXTURE_INPUTS.items():
            texture_ref = resolved_refs.get(channel)
            if texture_ref is None:
                continue
            for attr_name in attr_names:
                material_prim.CreateAttribute(
                    attr_name,
                    Sdf.ValueTypeNames.Asset,
                ).Set(Sdf.AssetPath(texture_ref))

        for prim in stage.TraverseAll():
            if not str(prim.GetPath()).startswith(material_path.rstrip("/") + "/"):
                continue
            if prim.IsInstanceProxy():
                continue
            if prim.IsInstance() or prim.IsInstanceable():
                prim.SetInstanceable(False)
            if not prim.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(prim)
            shader_id = shader.GetIdAttr().Get()
            shader_channel = _rebake_channel_from_name(prim.GetName())
            if (
                str(shader_id).lower() == "usduvtexture"
                and shader_channel in resolved_refs
            ):
                file_input = shader.GetInput("file")
                texture_ref = resolved_refs[shader_channel]
                if file_input:
                    file_input.Set(Sdf.AssetPath(texture_ref))
                else:
                    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                        Sdf.AssetPath(texture_ref)
                    )
            for shader_input in shader.GetInputs():
                channel = _rebake_channel_from_name(shader_input.GetBaseName())
                texture_ref = resolved_refs.get(channel or "")
                if texture_ref is None:
                    continue
                if _coerce_rebake_texture_ref(shader_input.Get()) is None:
                    continue
                shader_input.Set(Sdf.AssetPath(texture_ref))
        stage.GetRootLayer().Save()
        temporary_output.replace(output_usd)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0 and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            if parsed > 0:
                return parsed
    return None


def _service_job_timeout_sec(texture_config: dict[str, Any]) -> int:
    """Return the per-material service wait budget.

    Step1X jobs can legitimately run longer than the old 600s client default,
    especially on full-material real assets. Keep several aliases so YAML,
    service-generated configs, and ad-hoc validation scripts can all set the
    same behavior without schema churn.
    """
    for key in ("job_timeout_sec", "service_timeout_sec"):
        parsed = _positive_int(texture_config.get(key))
        if parsed is not None:
            return parsed
    return _DEFAULT_SERVICE_JOB_TIMEOUT_SEC


def _expected_size_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, int) and value > 0:
        return (value, value)
    if isinstance(value, list | tuple) and len(value) == 2:
        width, height = value
        if (
            isinstance(width, int)
            and isinstance(height, int)
            and width > 0
            and height > 0
        ):
            return (width, height)
    return None


def _validate_textures_or_raise(
    unit_key: str,
    textures: GeneratedTextures,
    *,
    expected_size: Any = None,
) -> None:
    """Reject completed-but-unusable texture results.

    A `JobStatus(status="completed")` only tells us the upstream call returned
    without an error code. The texture set itself can still be unusable -- a
    schema-skewed service may parse to ``GeneratedTextures(albedo="", ...)``,
    or ``_localize_textures`` may have failed to download a remote file and
    left a non-local URI in place. Such results must not silently flow into
    blend/apply, which would skip them and exit 0.

    Downstream ``BlendTexturesTask`` calls ``Path(...)`` on raw strings;
    ``Path`` does not parse URI schemes. So ANY URI -- including ``file://`` --
    would silently skip downstream even though the underlying bytes might be
    reachable. The texture-agent's own ``_localize_textures`` already strips
    ``file://`` from accessible service URIs and writes bare local paths, so by
    the time this validator runs the only forms a correctly-behaving caller
    passes in are bare local paths. Anything else (any ``://``) is treated as a
    per-unit failure here -- failing loud beats silently re-creating the very
    bug this task is meant to fix. Relax this when ``BlendTexturesTask`` learns
    to resolve URIs.
    """
    expected_dimensions = _expected_size_tuple(expected_size)
    for texture_name, texture_path in (
        ("albedo", textures.albedo),
        ("normal", textures.normal),
        ("orm", textures.orm),
    ):
        if not texture_path:
            raise RuntimeError(
                f"Generation reported success for {unit_key} but produced "
                f"no {texture_name} path (got empty string)"
            )
        if "://" in texture_path:
            raise RuntimeError(
                f"Generation reported success for {unit_key} but produced an "
                f"unsupported {texture_name} URI: {texture_path!r}. The "
                f"texture-agent pipeline currently only consumes local file "
                f"paths (BlendTexturesTask uses Path(...) which does not "
                f"parse URIs); the service backend's _localize_textures "
                f"strips file:// from accessible files before reaching here."
            )
        if not Path(texture_path).exists():
            raise RuntimeError(
                f"Generation reported success for {unit_key} but the "
                f"{texture_name} path does not exist on disk: {texture_path!r}"
            )
        try:
            from PIL import Image

            with Image.open(texture_path) as image:
                image.load()
                if image.format != "PNG":
                    raise RuntimeError(
                        f"Generation reported success for {unit_key} but the "
                        f"{texture_name} map is not a PNG image: {texture_path!r}"
                    )
                if expected_dimensions and (
                    image.size[0] < expected_dimensions[0]
                    or image.size[1] < expected_dimensions[1]
                ):
                    raise RuntimeError(
                        f"Generation reported success for {unit_key} but the "
                        f"{texture_name} map has size {image.size}; expected "
                        f"at least {expected_dimensions}"
                    )
                if image.getbbox() is None:
                    raise RuntimeError(
                        f"Generation reported success for {unit_key} but the "
                        f"{texture_name} map is blank: {texture_path!r}"
                    )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Generation reported success for {unit_key} but the "
                f"{texture_name} map is not a readable PNG image: {texture_path!r}"
            ) from exc


def _raise_if_above_threshold(
    attempted: list[PrimTextureUnit],
    fresh_generated: dict[str, GeneratedTextures],
    errors: list[dict[str, Any]],
    *,
    backend_label: str,
    failure_threshold: float,
) -> None:
    """Raise when the per-unit failure rate hits ``failure_threshold``.

    ``attempted`` is the slice of units actually submitted this run (after
    skip-existing filtering); ``fresh_generated`` is the result map for
    THIS run only -- cached entries from a previous run are deliberately
    excluded. A resumed run where every fresh request failed (e.g. expired
    NIM key returning HTTP 403 on every unit) raises even if cache from an
    earlier successful run partially populates the merged output. The
    customer's environment is broken; don't paper over it with stale cache.

    ``failure_threshold`` is a fraction in [0.0, 1.0]:
      - 1.0 (default): raise only when 100% of fresh attempts failed
        (preserves the original "all must fail" gate).
      - 0.5: raise when at least half of fresh attempts failed.
      - 0.0: raise on any failure.

    Sub-threshold failures are logged as a warning and allowed to continue;
    downstream steps can still apply whatever textures did succeed. Per-unit
    error records are surfaced separately via ``context`` regardless.
    """
    if not attempted:
        return
    if not errors:
        return

    raise_if_failure_threshold_exceeded(
        attempted_count=len(attempted),
        errors=errors,
        backend_label=backend_label,
        failure_threshold=failure_threshold,
    )
    logger.warning(
        "Texture generation completed with %d/%d failures via %s "
        "(below threshold %.0f%%)",
        len(errors),
        len(attempted),
        backend_label,
        failure_threshold * 100,
    )


def _cached_texture_set(
    out_dir: Path,
    key: str,
    *,
    expected_size: Any = None,
) -> GeneratedTextures | None:
    """Return a valid cached texture set from flat or per-variant layout.

    Candidates are tested in order, using ``albedo.exists()`` as a quick
    pre-filter before validating the full PBR set with
    ``_validate_textures_or_raise``. Partial or stale candidates, such as
    albedo-only outputs from failed generations, log a warning and fall through
    to the next layout. Returns ``None`` when no candidate passes validation.
    """
    candidates = [
        (
            out_dir / f"{key}_albedo.png",
            out_dir / f"{key}_normal.png",
            out_dir / f"{key}_orm.png",
        ),
        (
            out_dir / key / f"{key}_albedo.png",
            out_dir / key / f"{key}_normal.png",
            out_dir / key / f"{key}_orm.png",
        ),
    ]
    for albedo, normal, orm in candidates:
        if albedo.exists():
            textures = GeneratedTextures(
                albedo=str(albedo),
                normal=str(normal),
                orm=str(orm),
            )
            try:
                _validate_textures_or_raise(
                    key,
                    textures,
                    expected_size=expected_size,
                )
            except RuntimeError as exc:
                logger.warning("Skipping invalid cached textures for %s: %s", key, exc)
                continue
            return textures
    return None


class GenerateTexturesTask(Task):
    """Generate PBR texture sets (albedo, normal, ORM) from text prompts.

    Iterates over prim_texture_units (from DiscoverMaterialsTask).

    Backend types:
        simple_image_gen: Local image generation (Gemini via NVIDIA Inference).
            No external service needed. Generates albedo + normal + roughness
            using AI image gen model with tailored prompts.
        service: Remote Texture Variation API service (e.g., Step1X-3D).
            Calls POST /v1/texture-variations on the configured endpoint URL.
            The service handles texture extraction, generation, and write-back.

    Context keys read:
        prim_texture_units (list[PrimTextureUnit]): From DiscoverMaterialsTask.
        texture_config (dict): Configuration including:
            backend_type: "simple_image_gen" or "service"
            endpoint: REST endpoint URL (required for "service")
            backend: Image gen backend name (for "simple_image_gen")
            model: Model override
            size: Texture resolution
            workers: Number of parallel workers (default 4)
            skip_existing: Skip if texture already exists (default True)
        working_dir (str): Working directory.
        usd_path (str): Input USD path.

    Context keys written:
        generated_textures (dict[str, GeneratedTextures]):
            Unit key -> GeneratedTextures (albedo, normal, orm paths).
    """

    def __init__(self) -> None:
        self.name = "GenerateTextures"
        self.description = "Generate PBR texture sets from prompts"

    def _run_simple_image_gen(
        self,
        units: list[PrimTextureUnit],
        context: dict[str, Any],
        out_dir: Path,
        texture_config: dict,
    ) -> tuple[dict[str, GeneratedTextures], list[dict[str, Any]], str]:
        """Generate textures using local ImageGenEngine.

        Returns ``(generated, errors, backend_label)``. Each ``errors`` entry
        is a structured dict (``{material, type, status, message}``) so the
        service layer can surface per-unit failures in SSE/status payloads
        instead of forcing customers to grep container logs.
        """
        image_gen_config = texture_config.get("image_gen", {})
        backend = image_gen_config.get("backend", "nim")
        model = image_gen_config.get("model")
        base_url = image_gen_config.get("base_url")
        api_key = image_gen_config.get("api_key")
        api_key_env = image_gen_config.get("api_key_env")
        workers = texture_config.get("workers", 4)

        engine = ImageGenEngine(
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
        )
        engine._ensure_model()
        client = TextureVariationClient(engine=engine, output_dir=out_dir)
        conditioning_by_key = _conditioning_by_unit_key(
            units,
            context,
            texture_config,
        )

        logger.info(
            "Generating %d PBR texture sets with %s (simple_image_gen, workers=%d)",
            len(units),
            engine.name,
            workers,
        )

        generated: dict[str, GeneratedTextures] = {}
        errors: list[dict[str, Any]] = []

        def _gen(unit: PrimTextureUnit) -> tuple[str, GeneratedTextures]:
            status = client.generate(
                source_asset_uri=context.get("usd_path", ""),
                conditioning=conditioning_by_key[unit.key],
                config=TextureVariationConfig(
                    strength=unit.opacity,
                    variant_name=unit.key,
                    seed=unit.seed,
                    texture_size=texture_config.get("size"),
                ),
            )
            if status.status != "completed" or not status.result:
                raise RuntimeError(
                    f"Generation failed for {unit.key}: {status.error_message}"
                )
            textures = status.result.generated_textures
            _validate_textures_or_raise(
                unit.key,
                textures,
                expected_size=texture_config.get("size"),
            )
            return unit.key, textures

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_gen, unit): unit.key for unit in units}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    k, textures = future.result()
                    generated[k] = textures
                except Exception as exc:
                    logger.exception("Failed to generate textures for %s", key)
                    errors.append(_classify_unit_failure(key, exc))

        return generated, errors, engine.name

    def _run_service(
        self,
        units: list[PrimTextureUnit],
        context: dict[str, Any],
        out_dir: Path,
        texture_config: dict,
        external_authoring_spec: ExternalAuthoringSpec | None = None,
        external_authoring_receipt: ExternalAuthoringCapabilityReceipt | None = None,
    ) -> tuple[dict[str, GeneratedTextures], list[dict[str, Any]], str]:
        """Generate textures using a remote REST service.

        Returns ``(generated, errors, backend_label)``. Each ``errors`` entry
        is a structured dict (``{material, type, status, message}``) so the
        service layer can surface per-unit failures in SSE/status payloads
        instead of forcing customers to grep container logs.
        """
        from texture_agent.functions.rest_client import RestTextureVariationClient

        endpoint = texture_config.get("endpoint")
        if not endpoint:
            raise ValueError(
                "texture_config.endpoint is required for backend_type='service'"
            )
        workers = texture_config.get("workers", 4)
        job_timeout_sec = _service_job_timeout_sec(texture_config)

        conditioning_by_key = _conditioning_by_unit_key(
            units,
            context,
            texture_config,
        )
        client = RestTextureVariationClient(
            endpoint,
            timeout=max(1200, job_timeout_sec + 60),
            submit_retry_timeout_sec=job_timeout_sec,
        )

        logger.info(
            "Generating %d PBR texture sets via service "
            "(%s, workers=%d, job_timeout_sec=%d)",
            len(units),
            endpoint,
            workers,
            job_timeout_sec,
        )

        generated: dict[str, GeneratedTextures] = {}
        errors: list[dict[str, Any]] = []
        response_metadata: dict[str, Any] = {}
        response_diagnostics: list[dict[str, Any]] = []

        def _gen_one(
            unit: PrimTextureUnit,
        ) -> tuple[str, GeneratedTextures, dict[str, Any]]:
            capabilities = _capabilities_from_config(texture_config)
            conditioning = conditioning_by_key[unit.key]
            source_asset_uri = _service_source_asset_uri_for_unit(
                context,
                unit,
                texture_config,
                out_dir,
            )
            resolved_seed = (
                unit.seed if unit.seed is not None else texture_config.get("seed")
            )
            status = client.generate(
                source_asset_uri=source_asset_uri,
                target=_service_target_for_unit(unit, texture_config),
                conditioning=conditioning,
                config=TextureVariationConfig(
                    strength=texture_config.get("strength", unit.opacity),
                    variant_name=unit.key,
                    seed=resolved_seed,
                    engine=texture_config.get("engine"),
                    texture_size=texture_config.get("size"),
                    weathering=_weathering_controls_for_unit(
                        unit,
                        context,
                        texture_config,
                    ),
                    custom_parameters=_custom_parameters_for_unit(
                        texture_config,
                        unit,
                    ),
                ),
                capabilities=capabilities,
                external_authoring=external_authoring_spec,
                wait=True,
                timeout_sec=job_timeout_sec,
            )

            if not status.result:
                raise RuntimeError(
                    f"Service failed for {unit.key}: "
                    f"{status.error_message[:200] if status.error_message else 'unknown'}"
                )

            if status.status != "completed":
                diagnostics = [
                    item
                    for item in (status.result.diagnostics or [])
                    if isinstance(item, dict)
                ]
                self._append_response_diagnostics(
                    diagnostics=diagnostics,
                    result=status.result,
                    unit=unit,
                    conditioning=conditioning,
                )
                raise _BackendResultError(
                    f"Service failed for {unit.key}: "
                    f"{status.error_message[:200] if status.error_message else 'unknown'}",
                    self._backend_record(
                        status.result,
                        endpoint=endpoint,
                        maps=status.result.maps or {},
                        diagnostics=diagnostics,
                    ),
                )

            local_textures, backend_record, backend_map_paths = (
                self._materialize_service_result(
                    status.result,
                    unit=unit,
                    conditioning=conditioning,
                    out_dir=out_dir,
                    endpoint=endpoint,
                    expected_size=texture_config.get("size"),
                )
            )
            backend_record["source_asset_uri"] = source_asset_uri
            if external_authoring_spec is not None:
                if external_authoring_receipt is None:
                    raise AssertionError(
                        "External authoring dispatch requires a go preflight receipt"
                    )
                authoring_diagnostics = validate_external_authoring_result(
                    spec=external_authoring_spec,
                    receipt=external_authoring_receipt,
                    metadata=status.result.metadata or {},
                    auxiliary_artifacts=status.result.auxiliary_artifacts or {},
                    source_asset_uri=source_asset_uri,
                    seed=resolved_seed,
                    map_paths=backend_map_paths,
                )
                for diagnostic in authoring_diagnostics:
                    _append_diagnostic_once(backend_record["diagnostics"], diagnostic)
                if any(
                    item.get("severity") == "error" for item in authoring_diagnostics
                ):
                    raise _BackendResultError(
                        "External authoring result failed provenance validation",
                        backend_record,
                    )
            try:
                _validate_textures_or_raise(
                    unit.key,
                    local_textures,
                    expected_size=texture_config.get("size"),
                )
            except RuntimeError as exc:
                _append_diagnostic_once(
                    backend_record["diagnostics"],
                    _backend_diagnostic(
                        "BACKEND_MAP_VALIDATION_FAILED",
                        severity="error",
                        unit=unit,
                        message="Generated texture maps failed validation.",
                        recommended_action=(
                            "Retry this target or choose a backend that returns "
                            "readable nonblank PNG maps at the requested size."
                        ),
                        details={"error": str(exc)},
                    ),
                )
                raise _BackendResultError(str(exc), backend_record) from exc
            return unit.key, local_textures, backend_record

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_gen_one, unit): unit.key for unit in units}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    k, textures, backend_record = future.result()
                    generated[k] = textures
                    response_metadata[k] = backend_record
                    response_diagnostics.extend(backend_record.get("diagnostics", []))
                    logger.info("[%s] Complete", k)
                except Exception as exc:
                    logger.exception("Failed to generate textures for %s", key)
                    error = _classify_unit_failure(key, exc)
                    errors.append(error)
                    unit = next(unit for unit in units if unit.key == key)
                    specific_diagnostics: list[dict[str, Any]] = []
                    has_specific_error = False
                    if isinstance(exc, _BackendResultError):
                        response_metadata[key] = exc.backend_record
                        specific_diagnostics = [
                            item
                            for item in exc.backend_record.get("diagnostics", [])
                            if isinstance(item, dict)
                        ]
                        has_specific_error = any(
                            item.get("severity") == "error"
                            for item in specific_diagnostics
                        )
                        response_diagnostics.extend(specific_diagnostics)
                    if not has_specific_error:
                        _append_diagnostic_once(
                            response_diagnostics,
                            _backend_diagnostic(
                                "BACKEND_PARTIAL_FAILURE",
                                severity="error",
                                unit=unit,
                                message="Texture generation failed for this target.",
                                recommended_action=(
                                    "Inspect backend error details and retry this "
                                    "target."
                                ),
                                details=error,
                            ),
                        )

        context["projection_backend_results"] = response_metadata
        context["generate_textures_diagnostics"] = response_diagnostics
        return generated, errors, f"service ({endpoint})"

    @staticmethod
    def _localize_artifact_uri(uri: str, key: str, suffix: str, out_dir: Path) -> str:
        """Copy accessible service file artifacts into the generated output dir."""
        if not uri:
            return ""
        try:
            remote_path = _path_from_local_path_or_uri(uri)
        except RuntimeError:
            return uri

        if remote_path.exists():
            trusted_root = out_dir.parent.resolve(strict=True)
            lexical_source = Path(os.path.abspath(remote_path.expanduser()))
            if not lexical_source.is_relative_to(trusted_root):
                raise RuntimeError(
                    "Texture backend returned a local artifact outside the current "
                    f"task workspace: {remote_path}"
                )
            local_path = out_dir / f"{key}_{suffix}.png"
            try:
                with open_regular_file_no_follow(lexical_source) as (
                    source,
                    source_metadata,
                ):
                    with open_confined_directory(out_dir) as output_descriptor:
                        copy_open_file_to_confined(
                            output_descriptor,
                            local_path.name,
                            source,
                            source_metadata,
                            overwrite=True,
                        )
            except (ArtifactPathError, FileNotFoundError, OSError) as exc:
                raise RuntimeError(
                    "Texture backend returned a local artifact outside the current "
                    f"task workspace: {remote_path}"
                ) from exc
            return str(local_path)
        return uri

    @classmethod
    def _localize_map(
        cls,
        maps: dict[str, MapArtifact],
        channel: str,
        key: str,
        out_dir: Path,
    ) -> str:
        artifact = maps.get(channel)
        if not artifact:
            return ""
        return cls._localize_artifact_uri(artifact.uri, key, channel, out_dir)

    @staticmethod
    def _image_size_and_blank(path: str) -> tuple[tuple[int, int], bool]:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            return image.size, image.getbbox() is None

    @staticmethod
    def _write_neutral_normal(path: Path, size: tuple[int, int]) -> None:
        from PIL import Image

        Image.new("RGB", size, (128, 128, 255)).save(path)

    @staticmethod
    def _constant_channel(size: tuple[int, int], value: int) -> Any:
        import numpy as np

        return np.full((size[1], size[0]), value, dtype=np.uint8)

    @classmethod
    def _channel_from_map_or_constant(
        cls,
        path: str,
        size: tuple[int, int],
        fallback: int,
    ) -> Any:
        import numpy as np
        from PIL import Image

        if not path:
            return cls._constant_channel(size, fallback)
        with Image.open(path) as image:
            if image.size != size:
                image = image.resize(size, Image.Resampling.LANCZOS)
            return np.array(image.convert("L"), dtype=np.uint8)

    @classmethod
    def _write_orm_from_channels(
        cls,
        *,
        output_path: Path,
        size: tuple[int, int],
        occlusion_path: str,
        roughness_path: str,
        metalness_path: str,
        roughness_fallback: int,
        metalness_fallback: int,
    ) -> None:
        import numpy as np
        from PIL import Image

        orm_arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        orm_arr[:, :, 0] = cls._channel_from_map_or_constant(occlusion_path, size, 255)
        orm_arr[:, :, 1] = cls._channel_from_map_or_constant(
            roughness_path, size, roughness_fallback
        )
        orm_arr[:, :, 2] = cls._channel_from_map_or_constant(
            metalness_path, size, metalness_fallback
        )
        Image.fromarray(orm_arr).save(output_path)

    @staticmethod
    def _constant_to_byte(value: float | None, default: int) -> int:
        if value is None:
            return default
        return max(0, min(255, round(value * 255)))

    @staticmethod
    def _map_artifacts_as_dict(maps: dict[str, MapArtifact]) -> dict[str, Any]:
        return {
            channel: {
                "uri": artifact.uri,
                "width": artifact.width,
                "height": artifact.height,
                "mime_type": artifact.mime_type,
                "colorspace": artifact.colorspace,
                "packing": artifact.packing,
            }
            for channel, artifact in maps.items()
        }

    @staticmethod
    def _backend_record(
        result: Any,
        *,
        endpoint: str,
        maps: dict[str, MapArtifact],
        diagnostics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        metadata = result.metadata or {}
        return {
            "maps": GenerateTexturesTask._map_artifacts_as_dict(maps),
            "auxiliary_artifacts": result.auxiliary_artifacts or {},
            "metadata": metadata,
            "capabilities": metadata.get("capabilities", {}),
            "degraded_channels": metadata.get("degraded_channels", []),
            "diagnostics": diagnostics,
            "variant_asset_uri": result.variant_asset_uri,
            "variant_name": result.variant_name,
            "endpoint": endpoint,
        }

    @classmethod
    def _append_response_diagnostics(
        cls,
        *,
        diagnostics: list[dict[str, Any]],
        result: Any,
        unit: PrimTextureUnit,
        conditioning: Conditioning,
    ) -> None:
        metadata = result.metadata or {}
        capabilities = metadata.get("capabilities") or {}
        unsupported_fields: list[str] = []
        if (
            conditioning.reference_image_uris
            and capabilities.get("image_conditioning") is False
        ):
            unsupported_fields.append("reference_image_uris")
        if conditioning.turntable_video_uri and capabilities.get("multiview") is False:
            unsupported_fields.append("turntable_video_uri")
        if conditioning.multiview_image_uris and capabilities.get("multiview") is False:
            unsupported_fields.append("multiview_image_uris")
        if unsupported_fields and not _has_diagnostic_code(
            diagnostics, "BACKEND_CONDITIONING_UNSUPPORTED"
        ):
            _append_diagnostic_once(
                diagnostics,
                _backend_diagnostic(
                    "BACKEND_CONDITIONING_UNSUPPORTED",
                    severity="warning",
                    unit=unit,
                    message="Backend reported unsupported conditioning inputs.",
                    recommended_action=(
                        "Retry with a backend that supports the requested "
                        "conditioning inputs."
                    ),
                    details={"unsupported_fields": unsupported_fields},
                ),
            )

        coverage = metadata.get("coverage") or {}
        target_coverage = coverage.get("target_coverage")
        if (
            isinstance(target_coverage, int | float)
            and target_coverage < 0.75
            and not _has_diagnostic_code(diagnostics, "BACKEND_LOW_COVERAGE")
        ):
            _append_diagnostic_once(
                diagnostics,
                _backend_diagnostic(
                    "BACKEND_LOW_COVERAGE",
                    severity="warning",
                    unit=unit,
                    message="Backend reported low target coverage for selected scope.",
                    recommended_action=(
                        "Inspect backend coverage artifacts and retry with a "
                        "clearer target if needed."
                    ),
                    details={"target_coverage": target_coverage, "threshold": 0.75},
                ),
            )

        auxiliary = result.auxiliary_artifacts or {}
        raw_geometry = auxiliary.get("geometry") or []
        geometry = raw_geometry if isinstance(raw_geometry, list) else [raw_geometry]
        if geometry and not _has_diagnostic_code(
            diagnostics, "BACKEND_GEOMETRY_IGNORED"
        ):
            first = geometry[0] if isinstance(geometry[0], dict) else {}
            _append_diagnostic_once(
                diagnostics,
                _backend_diagnostic(
                    "BACKEND_GEOMETRY_IGNORED",
                    severity="warning",
                    unit=unit,
                    message=(
                        "Backend returned replacement geometry; Texture Agent "
                        "preserved source geometry."
                    ),
                    recommended_action=(
                        "Review auxiliary geometry artifacts manually if needed."
                    ),
                    details={
                        "geometry_uri": first.get("uri"),
                        "geometry_count": len(geometry),
                    },
                ),
            )

    @classmethod
    def _reject_blank_required_map(
        cls,
        *,
        path: str,
        channel: str,
        unit: PrimTextureUnit,
        diagnostics: list[dict[str, Any]],
        result: Any,
        endpoint: str,
        maps: dict[str, MapArtifact],
    ) -> tuple[int, int]:
        try:
            size, blank = cls._image_size_and_blank(path)
        except Exception as exc:
            blank = True
            size = (0, 0)
            details: dict[str, Any] = {
                "map": channel,
                "path": path,
                "reason": type(exc).__name__,
            }
        else:
            details = {"map": channel, "path": path}

        if blank:
            _append_diagnostic_once(
                diagnostics,
                _backend_diagnostic(
                    "BACKEND_TEXTURE_BLANK",
                    severity="error",
                    unit=unit,
                    message=f"Backend returned a blank required {channel} map.",
                    recommended_action="Treat this job as failed and retry.",
                    details=details,
                ),
            )
            raise _BackendResultError(
                f"Service completed for {unit.key} but returned a blank {channel} map",
                cls._backend_record(
                    result,
                    endpoint=endpoint,
                    maps=maps,
                    diagnostics=diagnostics,
                ),
            )
        return size

    @classmethod
    def _drop_blank_optional_map(
        cls,
        *,
        path: str,
        channel: str,
        unit: PrimTextureUnit,
        diagnostics: list[dict[str, Any]],
    ) -> str:
        if not path:
            return ""
        if "://" in path or not Path(path).exists():
            return ""
        try:
            _size, blank = cls._image_size_and_blank(path)
        except Exception:
            return path
        if not blank:
            return path
        _append_diagnostic_once(
            diagnostics,
            _backend_diagnostic(
                "BACKEND_TEXTURE_BLANK",
                severity="warning",
                unit=unit,
                message=f"Backend returned a blank optional {channel} map.",
                recommended_action=(
                    "Texture Agent substituted a fallback map for downstream "
                    "compatibility."
                ),
                details={"map": channel, "path": path},
            ),
        )
        return ""

    @classmethod
    def _materialize_service_result(
        cls,
        result: Any,
        *,
        unit: PrimTextureUnit,
        conditioning: Conditioning,
        out_dir: Path,
        endpoint: str,
        expected_size: Any = None,
    ) -> tuple[GeneratedTextures, dict[str, Any], dict[str, str]]:
        """Normalize service maps and retain paths actually returned by the backend."""
        diagnostics = [
            item for item in (result.diagnostics or []) if isinstance(item, dict)
        ]
        maps: dict[str, MapArtifact] = result.maps or {}
        generated = result.generated_textures

        cls._append_response_diagnostics(
            diagnostics=diagnostics,
            result=result,
            unit=unit,
            conditioning=conditioning,
        )

        local_albedo = cls._localize_map(maps, "albedo", unit.key, out_dir)
        if not local_albedo:
            local_albedo = cls._localize_artifact_uri(
                generated.albedo, unit.key, "albedo", out_dir
            )
        if not local_albedo:
            if not _has_diagnostic_code(diagnostics, "BACKEND_MAP_MISSING"):
                _append_diagnostic_once(
                    diagnostics,
                    _backend_diagnostic(
                        "BACKEND_MAP_MISSING",
                        severity="error",
                        unit=unit,
                        message="Backend did not return required albedo map.",
                        recommended_action=(
                            "Treat this job as failed and retry or choose another "
                            "backend."
                        ),
                        details={"missing_maps": ["albedo"]},
                    ),
                )
            raise _BackendResultError(
                f"Service completed for {unit.key} but returned no albedo map",
                cls._backend_record(
                    result,
                    endpoint=endpoint,
                    maps=maps,
                    diagnostics=diagnostics,
                ),
            )

        size = cls._reject_blank_required_map(
            path=local_albedo,
            channel="albedo",
            unit=unit,
            diagnostics=diagnostics,
            result=result,
            endpoint=endpoint,
            maps=maps,
        )
        expected_dimensions = _expected_size_tuple(expected_size)
        if expected_dimensions:
            # The backend may publish an upscaled albedo while omitting an
            # optional PBR map. Synthetic fallbacks must follow that published
            # atlas instead of shrinking back to the configured request size.
            size = (
                max(size[0], expected_dimensions[0]),
                max(size[1], expected_dimensions[1]),
            )

        local_normal = cls._localize_map(maps, "normal", unit.key, out_dir)
        if not local_normal:
            local_normal = cls._localize_artifact_uri(
                generated.normal, unit.key, "normal", out_dir
            )
        local_normal = cls._drop_blank_optional_map(
            path=local_normal,
            channel="normal",
            unit=unit,
            diagnostics=diagnostics,
        )
        backend_map_paths = {
            "albedo": local_albedo,
            "normal": local_normal,
        }
        if not local_normal:
            normal_path = out_dir / f"{unit.key}_normal.png"
            cls._write_neutral_normal(normal_path, size)
            local_normal = str(normal_path)
            _append_diagnostic_once(
                diagnostics,
                _backend_diagnostic(
                    "BACKEND_MAP_MISSING",
                    severity="warning",
                    unit=unit,
                    message=(
                        "Backend returned no normal map; Texture Agent created "
                        "a neutral tangent-space normal map."
                    ),
                    recommended_action=(
                        "Use a backend with normal-map support for full PBR output."
                    ),
                    details={"missing_maps": ["normal"]},
                ),
            )

        local_orm = cls._localize_map(maps, "orm", unit.key, out_dir)
        if not local_orm:
            local_orm = cls._localize_artifact_uri(
                generated.orm, unit.key, "orm", out_dir
            )
        local_orm = cls._drop_blank_optional_map(
            path=local_orm,
            channel="orm",
            unit=unit,
            diagnostics=diagnostics,
        )
        backend_map_paths["orm"] = local_orm
        if not local_orm:
            roughness_path = cls._localize_map(maps, "roughness", unit.key, out_dir)
            metalness_path = cls._localize_map(maps, "metalness", unit.key, out_dir)
            occlusion_path = cls._localize_map(maps, "occlusion", unit.key, out_dir)
            roughness_path = cls._drop_blank_optional_map(
                path=roughness_path,
                channel="roughness",
                unit=unit,
                diagnostics=diagnostics,
            )
            metalness_path = cls._drop_blank_optional_map(
                path=metalness_path,
                channel="metalness",
                unit=unit,
                diagnostics=diagnostics,
            )
            occlusion_path = cls._drop_blank_optional_map(
                path=occlusion_path,
                channel="occlusion",
                unit=unit,
                diagnostics=diagnostics,
            )
            orm_path = out_dir / f"{unit.key}_orm.png"
            cls._write_orm_from_channels(
                output_path=orm_path,
                size=size,
                occlusion_path=occlusion_path,
                roughness_path=roughness_path,
                metalness_path=metalness_path,
                roughness_fallback=cls._constant_to_byte(
                    unit.material_info.specular_roughness, 255
                ),
                metalness_fallback=cls._constant_to_byte(
                    unit.material_info.base_metalness, 0
                ),
            )
            local_orm = str(orm_path)
            missing = [
                channel
                for channel, path in (
                    ("roughness", roughness_path),
                    ("metalness", metalness_path),
                )
                if not path
            ]
            _append_diagnostic_once(
                diagnostics,
                _backend_diagnostic(
                    "BACKEND_MAP_MISSING",
                    severity="warning",
                    unit=unit,
                    message=(
                        "Backend returned no ORM map; Texture Agent packed ORM "
                        "from available channels and material constants."
                    ),
                    recommended_action=(
                        "Use a backend with ORM support for model-authored PBR maps."
                    ),
                    details={"missing_maps": ["orm", *missing]},
                ),
            )

        textures = GeneratedTextures(
            albedo=local_albedo,
            normal=local_normal,
            orm=local_orm,
        )
        backend_record = cls._backend_record(
            result,
            endpoint=endpoint,
            maps=maps,
            diagnostics=diagnostics,
        )
        auxiliary = backend_record.get("auxiliary_artifacts")
        masks = auxiliary.get("masks") if isinstance(auxiliary, dict) else None
        weathering_mask = masks.get("weathering") if isinstance(masks, dict) else None
        if isinstance(weathering_mask, dict):
            localized_mask = cls._localize_artifact_uri(
                str(weathering_mask.get("uri") or ""),
                unit.key,
                "weathering_mask",
                out_dir,
            )
            if localized_mask:
                weathering_mask["uri"] = localized_mask
        if unit.detail_policy != DETAIL_POLICY_DEFAULT:
            backend_record["metadata"] = {
                **backend_record.get("metadata", {}),
                "detail_policy": unit.detail_policy,
            }
        return textures, backend_record, backend_map_paths

    @staticmethod
    def _localize_textures(
        textures: GeneratedTextures,
        key: str,
        out_dir: Path,
        endpoint: str,
    ) -> GeneratedTextures:
        """Ensure texture paths are local files.

        If the service returns file:// URIs (local to the service host),
        download them via HTTP. If the service returns http:// URLs or
        local paths, use them directly.
        """

        def _download_if_needed(uri: str | None, suffix: str) -> str:
            if not uri:
                return ""
            # Already a local path
            if not uri.startswith("file://") and Path(uri).exists():
                return uri
            # For file:// URIs, try to download via a /files endpoint
            # or just use the path if it's accessible
            remote_path = uri.replace("file://", "").replace("file:", "")
            local_path = out_dir / f"{key}_{suffix}.png"
            if Path(remote_path).exists():
                import shutil

                shutil.copy2(remote_path, str(local_path))
                return str(local_path)
            # Path not accessible locally — leave as-is
            return uri

        return GeneratedTextures(
            albedo=_download_if_needed(textures.albedo, "albedo"),
            normal=_download_if_needed(textures.normal, "normal"),
            orm=_download_if_needed(textures.orm, "orm"),
        )

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        units: list[PrimTextureUnit] = context.get("prim_texture_units", [])
        if not units:
            logger.warning(
                "No prim_texture_units in context — was generate_prompts step skipped?"
            )
            context.setdefault("generated_textures", {})
            return context

        texture_config: dict = context.get("texture_config", {})
        working_dir = Path(context["working_dir"])
        skip_existing = bool(context.get("resume")) or texture_config.get(
            "skip_existing", True
        )
        external_authoring_config = texture_config.get("external_authoring")
        external_authoring_requested = external_authoring_config is not None and not (
            isinstance(external_authoring_config, dict)
            and external_authoring_config.get("enabled", True) is False
        )
        if external_authoring_requested and skip_existing:
            # Existing flat texture caches do not carry the source/spec/tool
            # receipt needed to prove they came from this preflight environment.
            # Regenerate until cache entries have their own digest-bound receipt.
            logger.info(
                "External authoring disables unbound generated-texture cache reuse"
            )
            skip_existing = False

        # Validate the threshold BEFORE any backend dispatch so a typo
        # (``failure_threshold: "nan"`` / ``1.1``) fails fast instead of
        # racking up 8x network round-trips and only THEN raising a config
        # error.
        failure_threshold = validate_failure_threshold(
            texture_config.get("failure_threshold", 1.0),
            config_key="texture_config.failure_threshold",
        )

        out_dir = working_dir / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Filter to units that need generation
        to_generate: list[PrimTextureUnit] = []
        generated: dict[str, GeneratedTextures] = {}

        for unit in units:
            cached_textures = (
                _cached_texture_set(
                    out_dir,
                    unit.key,
                    expected_size=texture_config.get("size"),
                )
                if skip_existing
                else None
            )
            if cached_textures:
                logger.info("Skipping %s (already generated)", unit.key)
                generated[unit.key] = cached_textures
                continue
            to_generate.append(unit)

        if not to_generate:
            logger.info("No textures to generate")
            context["generated_textures"] = generated
            return context

        # Choose backend
        backend = texture_config.get("backend", "simple_image_gen")
        attempted_units = list(to_generate)
        preflight_errors: list[dict[str, Any]] = []
        preflight_metadata: dict[str, dict[str, Any]] = {}
        preflight_diagnostics: list[dict[str, Any]] = []
        dispatch_units = list(to_generate)

        (
            dispatch_units,
            external_preflight_errors,
            external_preflight_metadata,
            external_preflight_diagnostics,
            external_authoring_spec,
            external_authoring_receipt,
            external_authoring_feasibility,
        ) = _preflight_external_authoring(dispatch_units, texture_config)
        preflight_errors.extend(external_preflight_errors)
        preflight_metadata.update(external_preflight_metadata)
        preflight_diagnostics.extend(external_preflight_diagnostics)
        if external_authoring_feasibility is not None:
            context["external_authoring_preflight"] = external_authoring_feasibility
            context["external_authoring_preflight_path"] = str(
                _write_external_authoring_feasibility(
                    working_dir,
                    external_authoring_feasibility,
                )
            )

        if backend == "service":
            (
                dispatch_units,
                step1x_preflight_errors,
                step1x_preflight_metadata,
                step1x_preflight_diagnostics,
            ) = _preflight_step1x_targets(dispatch_units, texture_config)
            preflight_errors.extend(step1x_preflight_errors)
            preflight_metadata.update(step1x_preflight_metadata)
            preflight_diagnostics.extend(step1x_preflight_diagnostics)

        (
            dispatch_units,
            simple_preflight_errors,
            simple_preflight_metadata,
            simple_preflight_diagnostics,
        ) = _preflight_simple_image_gen_conditioning(
            dispatch_units,
            context,
            texture_config,
        )
        preflight_errors.extend(simple_preflight_errors)
        preflight_metadata.update(simple_preflight_metadata)
        preflight_diagnostics.extend(simple_preflight_diagnostics)
        if preflight_errors and failure_threshold <= 0.0:
            # Service-created pipelines use failure_threshold=0 so any
            # unsupported target/conditioning rejects the whole request. Apply
            # the same fail-before-launch behavior to direct text-only NIM.
            dispatch_units = []

        if backend == "service":
            if dispatch_units:
                new_generated, errors, backend_label = self._run_service(
                    dispatch_units,
                    context,
                    out_dir,
                    texture_config,
                    external_authoring_spec,
                    external_authoring_receipt,
                )
            else:
                new_generated = {}
                errors = []
                endpoint = texture_config.get("endpoint", "")
                backend_label = f"service ({endpoint})"
        elif backend == "simple_image_gen":
            if dispatch_units:
                new_generated, errors, backend_label = self._run_simple_image_gen(
                    dispatch_units,
                    context,
                    out_dir,
                    texture_config,
                )
            else:
                new_generated = {}
                errors = []
                backend_label = "simple_image_gen (preflight rejected)"
        else:
            raise ValueError(
                f"Unknown texture backend: {backend}. "
                "Use 'simple_image_gen' or 'service'."
            )

        # Merge fresh successes onto cached hits and publish to context
        # BEFORE the threshold raise. The executor's per-step except block
        # extracts step stats from context; without this write a partial-
        # failure-above-threshold raise would report ``textures_generated:
        # 0`` even when some materials succeeded and were written to disk.
        generated.update(new_generated)
        if preflight_metadata:
            context["projection_backend_results"] = {
                **preflight_metadata,
                **(context.get("projection_backend_results") or {}),
            }
        if preflight_diagnostics:
            context["generate_textures_diagnostics"] = [
                *preflight_diagnostics,
                *(context.get("generate_textures_diagnostics") or []),
            ]
        errors = [*preflight_errors, *errors]
        context["generated_textures"] = generated
        context["generate_textures_errors"] = errors
        context["generate_textures_failed_count"] = len(errors)
        context["generate_textures_attempted_count"] = len(attempted_units)

        # Threshold decision uses FRESH attempts only -- cached entries
        # from prior runs must not mask a totally-broken backend (e.g.
        # expired NIM key returning HTTP 403 on every fresh request).
        _raise_if_above_threshold(
            attempted_units,
            new_generated,
            errors,
            backend_label=backend_label,
            failure_threshold=failure_threshold,
        )
        logger.info("Generated %d PBR texture sets", len(generated))
        return context
