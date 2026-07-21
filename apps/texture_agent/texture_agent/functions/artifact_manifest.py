# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact manifest and portability helpers for Texture Agent runs."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from pxr import Sdf, Usd, UsdShade

ARTIFACTS_MANIFEST_SCHEMA_VERSION = "texture-agent-artifacts.v1"
DIAGNOSTIC_SCHEMA_VERSION = "texture-agent-diagnostic.v1"
PIPELINE_FAILURE_CODE = "TEXTURE_PIPELINE_STEP_FAILED"
PIPELINE_FAILURE_MESSAGE = "Texture Agent pipeline step failed."

_URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_SENSITIVE_QUERY_RE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|authorization|credential|token|password|secret)=([^&\s]+)"
)
_BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")
_TOKEN_VALUE_RE = re.compile(r"\b(?:sk|nvapi|ghp|github_pat)-[A-Za-z0-9._~+/=-]{8,}")
_SENSITIVE_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_ENDPOINT_KEYS = {"endpoint", "endpoint_url", "base_url"}
_KNOWN_PROJECTION_CHANNELS = (
    "albedo",
    "normal",
    "orm",
    "roughness",
    "metalness",
    "occlusion",
    "mask",
    "uv_island_mask",
)


def make_diagnostic(
    code: str,
    *,
    severity: str,
    stage: str,
    message: str,
    recommended_action: str = "",
    prim_path: str | None = None,
    material_name: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a versioned diagnostic payload."""
    payload: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "code": code,
        "severity": severity,
        "stage": stage,
        "prim_path": prim_path,
        "material_name": material_name,
        "message": message,
        "recommended_action": recommended_action,
        "details": details or {},
    }
    return payload


def _working_root(context: dict[str, Any]) -> Path:
    working_dir = Path(context["working_dir"]).resolve()
    return working_dir.parent


def _display_path(path: str | Path | None, root: Path) -> str | None:
    if path is None:
        return None
    raw = str(path)
    if not raw:
        return raw
    if _URI_SCHEME_RE.match(raw):
        return _redact_string(raw)
    try:
        return _redact_string(
            Path(os.path.relpath(Path(raw).resolve(), root)).as_posix()
        )
    except (OSError, ValueError):
        return _redact_string(raw)


def _path_entry(path: str | Path | None, root: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    raw = str(path)
    entry: dict[str, Any] = {"path": _display_path(raw, root)}
    if not raw or _URI_SCHEME_RE.match(raw):
        entry["exists"] = False
        return entry
    try:
        local = Path(raw)
        entry["exists"] = local.exists()
        if local.is_file():
            entry["size_bytes"] = local.stat().st_size
    except (OSError, ValueError):
        entry["exists"] = False
    return entry


def _image_info(path: str | Path | None, root: Path) -> dict[str, Any] | None:
    entry = _path_entry(path, root)
    if entry is None:
        return None
    raw = str(path)
    if not raw or _URI_SCHEME_RE.match(raw):
        return entry
    try:
        from PIL import Image

        with Image.open(raw) as img:
            entry["width"] = img.width
            entry["height"] = img.height
            entry["mode"] = img.mode
            entry["nonblank"] = img.getbbox() is not None
    except Exception as err:
        entry["open_error"] = str(err)
    return entry


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in value]
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return str(value)


def _redact_string(value: str) -> str:
    redacted = _SENSITIVE_QUERY_RE.sub(r"\1=<redacted>", value)
    redacted = _BEARER_RE.sub("Bearer <redacted>", redacted)
    return _TOKEN_VALUE_RE.sub("<redacted>", redacted)


def _redact_sensitive(value: Any) -> Any:
    if is_dataclass(value):
        return _redact_sensitive(asdict(value))
    if isinstance(value, Path):
        return _redact_string(str(value))
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            key_lower = key_str.lower()
            if any(token in key_lower for token in _SENSITIVE_KEYS):
                redacted[key_str] = "<redacted>"
            elif key_lower in _ENDPOINT_KEYS:
                redacted[key_str] = "<configured>" if item else item
            else:
                redacted[key_str] = _redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return _jsonable(value)


def redact_sensitive(value: Any) -> Any:
    """Return a JSON-safe copy with secret-bearing values redacted."""
    return _redact_sensitive(value)


def _config_summary(context: dict[str, Any]) -> dict[str, Any]:
    config = context.get("config") or {}
    project = config.get("project") or {}
    texture = context.get("texture_config") or {}
    auto_prompt = context.get("auto_prompt_config") or {}
    return {
        "project_name": project.get("name"),
        "session_id": project.get("session_id"),
        "texture": _redact_sensitive(texture),
        "auto_prompt": _redact_sensitive(auto_prompt),
        "steps": _redact_sensitive(config.get("steps") or {}),
    }


def _read_uv_report(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _material_entries(context: dict[str, Any]) -> list[dict[str, Any]]:
    materials = context.get("discovered_materials") or []
    return [_jsonable(material) for material in materials]


def _discover_partial_artifacts(context: dict[str, Any]) -> list[dict[str, Any]]:
    """List files already published below the run directory without reading them."""
    working_dir = Path(context["working_dir"])
    try:
        resolved_working_dir = working_dir.resolve()
        candidates = sorted(working_dir.rglob("*"))
    except (OSError, ValueError):
        return []

    artifacts: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.name == "artifacts_manifest.json" or candidate.name.startswith(
            ".artifacts_manifest.json."
        ):
            continue
        try:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            candidate.resolve().relative_to(resolved_working_dir)
            artifacts.append(
                {
                    "path": candidate.relative_to(working_dir).as_posix(),
                    "size_bytes": candidate.stat().st_size,
                }
            )
        except (OSError, ValueError):
            continue
    return artifacts


def _planning_section(context: dict[str, Any], root: Path) -> dict[str, Any]:
    plan = context.get("texture_plan")
    model_dump = getattr(plan, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="json")
    elif isinstance(plan, dict):
        payload = _jsonable(plan)
    else:
        payload = None
    if payload is None:
        path_value = context.get("texture_plan_path")
        if isinstance(path_value, str) and path_value.strip():
            try:
                loaded = json.loads(Path(path_value).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                payload = _jsonable(loaded)
    return {
        "plan_available": payload is not None,
        "texture_plan": payload,
        "texture_plan_artifact": _path_entry(
            context.get("texture_plan_path"),
            root,
        ),
        "decision_state": (payload or {}).get("decision", {}).get("state"),
        "execution_allowed": (payload or {})
        .get("decision", {})
        .get("execution_allowed"),
        "counts": (payload or {}).get("counts", {}),
        "limits": (payload or {}).get("limits", {}),
    }


def _selected_materials(context: dict[str, Any]) -> list[dict[str, Any]]:
    units = context.get("prim_texture_units") or []
    selected: list[dict[str, Any]] = []
    for unit in units:
        material = getattr(unit, "material_info", None)
        selected.append(
            {
                "key": getattr(unit, "key", ""),
                "material_name": getattr(material, "name", ""),
                "material_path": getattr(material, "prim_path", ""),
                "prim_path": getattr(unit, "prim_path", ""),
                "prompt": getattr(unit, "prompt", ""),
                "opacity": getattr(unit, "opacity", None),
                "detail_policy": _manifest_detail_policy(unit),
                "seed": getattr(unit, "seed", None),
            }
        )
    return selected


def _manifest_detail_policy(unit: Any) -> str:
    return getattr(unit, "detail_policy", None) or "default"


def _texture_set_entries(
    textures: dict[str, Any],
    root: Path,
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for key, bundle in textures.items():
        entries[key] = {
            "albedo": _image_info(getattr(bundle, "albedo", None), root),
            "normal": _image_info(getattr(bundle, "normal", None), root),
            "orm": _image_info(getattr(bundle, "orm", None), root),
        }
    return entries


def _artifact_map_entry(value: Any, root: Path) -> dict[str, Any]:
    artifact = _redact_sensitive(value)
    if not isinstance(artifact, dict):
        artifact = {"uri": str(artifact)}

    uri = str(artifact.get("uri") or "")
    entry: dict[str, Any] = {
        "uri": _display_path(uri, root),
        "path": _path_entry(uri, root),
    }
    for key in ("width", "height", "mime_type", "colorspace", "packing"):
        if artifact.get(key) is not None:
            entry[key] = artifact[key]
    return entry


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _projection_channel_state(
    record: dict[str, Any],
    generated_bundle: Any,
) -> dict[str, str]:
    maps = _dict_or_empty(record.get("maps"))
    diagnostics = _list_or_empty(record.get("diagnostics"))
    degraded = {
        str(channel) for channel in _list_or_empty(record.get("degraded_channels"))
    }
    metadata = _dict_or_empty(record.get("metadata"))
    degraded.update(
        str(channel) for channel in _list_or_empty(metadata.get("degraded_channels"))
    )

    generated_channels = {
        "albedo": getattr(generated_bundle, "albedo", ""),
        "normal": getattr(generated_bundle, "normal", ""),
        "orm": getattr(generated_bundle, "orm", ""),
    }
    missing_channels: set[str] = set()
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        details = _dict_or_empty(diagnostic.get("details"))
        for channel in _list_or_empty(details.get("missing_maps")):
            missing_channels.add(str(channel))

    state: dict[str, str] = {}
    state["albedo"] = (
        "present" if maps.get("albedo") or generated_channels["albedo"] else "missing"
    )
    state["normal"] = (
        "present"
        if maps.get("normal")
        else "synthesized_neutral"
        if generated_channels["normal"] and "normal" in missing_channels
        else "present"
        if generated_channels["normal"]
        else "absent"
    )
    state["orm"] = (
        "present"
        if maps.get("orm")
        else "packed_from_channels_or_constants"
        if generated_channels["orm"]
        and (
            "orm" in missing_channels
            or any(
                channel in maps for channel in ("roughness", "metalness", "occlusion")
            )
        )
        else "present"
        if generated_channels["orm"]
        else "absent"
    )

    for channel in _KNOWN_PROJECTION_CHANNELS:
        if channel in state:
            continue
        if maps.get(channel):
            state[channel] = "present"
        elif channel in degraded or channel in missing_channels:
            state[channel] = "absent"
        else:
            state[channel] = "absent"
    return state


def _projection_artifact_groups(auxiliary_artifacts: Any) -> dict[str, Any]:
    auxiliary = auxiliary_artifacts if isinstance(auxiliary_artifacts, dict) else {}
    return {
        "masks": _redact_sensitive(
            auxiliary.get("masks")
            or auxiliary.get("mask")
            or auxiliary.get("uv_island_mask")
            or {}
        ),
        "coverage": _redact_sensitive(
            auxiliary.get("coverage")
            or auxiliary.get("coverage_mask")
            or auxiliary.get("coverage_artifacts")
            or {}
        ),
        "debug": _redact_sensitive(
            auxiliary.get("debug") or auxiliary.get("debug_artifacts") or {}
        ),
        "geometry": _redact_sensitive(auxiliary.get("geometry") or []),
    }


def _projection_warning_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = _list_or_empty(record.get("diagnostics"))
    warnings: list[dict[str, Any]] = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        if diagnostic.get("severity") == "warning":
            warnings.append(_redact_sensitive(diagnostic))
    return warnings


def _projection_backend_entries(
    context: dict[str, Any],
    root: Path,
) -> dict[str, dict[str, Any]]:
    records = _dict_or_empty(context.get("projection_backend_results"))
    generated = _dict_or_empty(context.get("generated_textures"))
    entries: dict[str, dict[str, Any]] = {}
    for key, raw_record in records.items():
        if not isinstance(raw_record, dict):
            continue

        maps = _dict_or_empty(raw_record.get("maps"))
        map_entries = {
            str(channel): _artifact_map_entry(artifact, root)
            for channel, artifact in maps.items()
        }
        raw_metadata = _dict_or_empty(raw_record.get("metadata"))
        metadata = _redact_sensitive(raw_metadata)
        raw_capabilities = raw_record.get("capabilities") or raw_metadata.get(
            "capabilities"
        )
        capabilities = _redact_sensitive(raw_capabilities or {})
        degraded_channels = _redact_sensitive(
            raw_record.get("degraded_channels")
            or raw_metadata.get("degraded_channels")
            or []
        )
        auxiliary_artifacts = _dict_or_empty(raw_record.get("auxiliary_artifacts"))
        diagnostics = _redact_sensitive(_list_or_empty(raw_record.get("diagnostics")))
        entries[str(key)] = {
            "maps": map_entries,
            "map_count": len(map_entries),
            "channel_state": _projection_channel_state(
                raw_record,
                generated.get(key),
            ),
            "auxiliary_artifacts": _redact_sensitive(auxiliary_artifacts),
            "artifacts": _projection_artifact_groups(auxiliary_artifacts),
            "metadata": metadata,
            "capabilities": capabilities,
            "projection": _redact_sensitive(
                raw_metadata.get("projection")
                or raw_metadata.get("projection_metadata")
                or {}
            ),
            "editing": _redact_sensitive(
                raw_metadata.get("editing")
                or raw_metadata.get("editing_metadata")
                or {}
            ),
            "coverage": _redact_sensitive(raw_metadata.get("coverage") or {}),
            "degraded_channels": degraded_channels,
            "warnings": _projection_warning_entries(raw_record),
            "diagnostics": diagnostics,
            "variant_asset": _path_entry(raw_record.get("variant_asset_uri"), root),
            "variant_name": raw_record.get("variant_name"),
            "endpoint": "<configured>"
            if raw_record.get("endpoint")
            else raw_record.get("endpoint"),
        }
    return entries


def _projection_backend_summary(context: dict[str, Any]) -> dict[str, Any]:
    records = _dict_or_empty(context.get("projection_backend_results"))
    metadata_by_unit: dict[str, Any] = {}
    capabilities_by_unit: dict[str, Any] = {}
    degraded_by_unit: dict[str, Any] = {}
    coverage_by_unit: dict[str, Any] = {}
    diagnostics: list[Any] = []

    for key, raw_record in records.items():
        if not isinstance(raw_record, dict):
            continue
        metadata = _dict_or_empty(raw_record.get("metadata"))
        metadata_by_unit[str(key)] = _redact_sensitive(metadata)
        capabilities_by_unit[str(key)] = _redact_sensitive(
            raw_record.get("capabilities") or metadata.get("capabilities") or {}
        )
        degraded_by_unit[str(key)] = _redact_sensitive(
            raw_record.get("degraded_channels")
            or metadata.get("degraded_channels")
            or []
        )
        coverage_by_unit[str(key)] = _redact_sensitive(metadata.get("coverage") or {})
        diagnostics.extend(_list_or_empty(raw_record.get("diagnostics")))

    return {
        "unit_count": len(records),
        "metadata": metadata_by_unit,
        "capabilities": capabilities_by_unit,
        "degraded_channels": degraded_by_unit,
        "coverage": coverage_by_unit,
        "diagnostics": _redact_sensitive(diagnostics),
    }


def _output_texture_references(output_usd: Path) -> list[dict[str, Any]] | None:
    try:
        stage = Usd.Stage.Open(str(output_usd))
    except Exception:
        return None
    if not stage:
        return None

    refs: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        is_shader = prim.IsA(UsdShade.Shader)
        for attr in prim.GetAttributes():
            val = attr.Get()
            path: str | None = None
            value_type = ""
            if isinstance(val, Sdf.AssetPath) and val.path:
                path = val.path
                value_type = "asset"
            elif isinstance(val, str) and val and is_shader:
                attr_name = attr.GetName()
                if attr_name.startswith("inputs:") and attr_name.endswith("_texture"):
                    path = val
                    value_type = "string"

            if path and path.lower().endswith(".png"):
                refs.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "attribute": attr.GetName(),
                        "value_type": value_type,
                        "path": path,
                    }
                )
    return refs


def validate_output_texture_portability(
    output_usd_path: str | Path,
    *,
    bundle_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate texture refs in an output USD are relative and resolvable."""
    output_usd = Path(output_usd_path)
    diagnostics: list[dict[str, Any]] = []

    if not output_usd.is_file():
        diagnostics.append(
            make_diagnostic(
                "PACKAGE_MISSING_ARTIFACT",
                severity="error",
                stage="package",
                message="Output USD is not present.",
                recommended_action=(
                    "Inspect the apply_textures step and download available "
                    "individual artifacts instead of the package."
                ),
                details={"path": str(output_usd)},
            )
        )
        return {
            "portable": False,
            "texture_reference_count": 0,
            "non_relative_texture_paths": [],
            "missing_texture_paths": [str(output_usd)],
            "diagnostics": diagnostics,
        }

    references = _output_texture_references(output_usd)
    if references is None:
        diagnostics.append(
            make_diagnostic(
                "PACKAGE_MISSING_ARTIFACT",
                severity="error",
                stage="package",
                message="Output USD could not be opened for portability validation.",
                recommended_action=(
                    "Inspect the output USD and download individual artifacts "
                    "instead of the package."
                ),
                details={"path": str(output_usd)},
            )
        )
        return {
            "portable": False,
            "texture_reference_count": 0,
            "non_relative_texture_paths": [],
            "missing_texture_paths": [str(output_usd)],
            "diagnostics": diagnostics,
        }

    missing: list[str] = []
    non_relative: list[str] = []
    resolved_bundle_root = (
        Path(bundle_root).resolve()
        if bundle_root
        else output_usd.parent.parent.resolve()
    )

    for ref in references:
        path = str(ref["path"])
        details = {
            "attribute": ref["attribute"],
            "path": path,
            "value_type": ref["value_type"],
        }
        if _URI_SCHEME_RE.match(path) or Path(path).is_absolute():
            non_relative.append(path)
            diagnostics.append(
                make_diagnostic(
                    "PACKAGE_ABSOLUTE_TEXTURE_PATH",
                    severity="error",
                    stage="package",
                    prim_path=ref["prim_path"],
                    message="Output USD texture reference is not sibling-relative.",
                    recommended_action=(
                        "Rewrite texture references under the output directory "
                        "before packaging or download."
                    ),
                    details=details,
                )
            )
            continue

        resolved = (output_usd.parent / path).resolve()
        try:
            resolved.relative_to(resolved_bundle_root)
        except ValueError:
            non_relative.append(path)
            diagnostics.append(
                make_diagnostic(
                    "PACKAGE_ABSOLUTE_TEXTURE_PATH",
                    severity="error",
                    stage="package",
                    prim_path=ref["prim_path"],
                    message=(
                        "Output USD texture reference leaves the result directory."
                    ),
                    recommended_action=(
                        "Copy textures into the run textures directory and author "
                        "paths relative to the output USD."
                    ),
                    details={
                        **details,
                        "resolved_path": str(resolved),
                        "bundle_root": str(resolved_bundle_root),
                    },
                )
            )
            continue

        if not resolved.is_file():
            missing.append(path)
            diagnostics.append(
                make_diagnostic(
                    "PACKAGE_MISSING_ARTIFACT",
                    severity="error",
                    stage="package",
                    prim_path=ref["prim_path"],
                    message="Output USD references a texture file that is not present.",
                    recommended_action=(
                        "Ensure generated/localized textures are copied next to "
                        "the output USD before packaging."
                    ),
                    details={**details, "resolved_path": str(resolved)},
                )
            )

    return {
        "portable": not diagnostics,
        "texture_reference_count": len(references),
        "non_relative_texture_paths": sorted(set(non_relative)),
        "missing_texture_paths": sorted(set(missing)),
        "diagnostics": diagnostics,
    }


def validate_artifacts_manifest_schema(manifest: dict[str, Any]) -> list[str]:
    """Return schema-contract errors for a texture-agent artifact manifest."""
    errors: list[str] = []
    if manifest.get("schema_version") != ARTIFACTS_MANIFEST_SCHEMA_VERSION:
        errors.append("schema_version must be texture-agent-artifacts.v1")

    required_sections = (
        "input",
        "prepared",
        "materials",
        "prompts",
        "textures",
        "outputs",
        "renders",
        "backend",
        "status",
    )
    for section in required_sections:
        if not isinstance(manifest.get(section), dict):
            errors.append(f"{section} section must be present")

    if "planning" in manifest:
        planning = manifest.get("planning")
        if not isinstance(planning, dict):
            errors.append("planning section must be an object")
        else:
            for key in (
                "plan_available",
                "texture_plan",
                "texture_plan_artifact",
                "decision_state",
                "execution_allowed",
                "counts",
                "limits",
            ):
                if key not in planning:
                    errors.append(f"planning.{key} is required")

    outputs = manifest.get("outputs") or {}
    portability = outputs.get("portability") or {}
    for key in ("portable", "texture_reference_count", "diagnostics"):
        if key not in portability:
            errors.append(f"outputs.portability.{key} is required")

    textures = manifest.get("textures") or {}
    for key in ("generated", "blended", "projection_backend"):
        if not isinstance(textures.get(key), dict):
            errors.append(f"textures.{key} must be present")

    projection_backend = textures.get("projection_backend") or {}
    for unit_key, entry in projection_backend.items():
        if not isinstance(entry, dict):
            errors.append(f"textures.projection_backend.{unit_key} must be an object")
            continue
        for key in (
            "maps",
            "map_count",
            "channel_state",
            "auxiliary_artifacts",
            "artifacts",
            "metadata",
            "capabilities",
            "coverage",
            "degraded_channels",
            "diagnostics",
            "warnings",
        ):
            if key not in entry:
                errors.append(
                    f"textures.projection_backend.{unit_key}.{key} is required"
                )
        channel_state = entry.get("channel_state") or {}
        for channel in _KNOWN_PROJECTION_CHANNELS:
            if channel not in channel_state:
                errors.append(
                    "textures.projection_backend."
                    f"{unit_key}.channel_state.{channel} is required"
                )

    backend = manifest.get("backend") or {}
    projection = backend.get("projection") or {}
    for key in (
        "unit_count",
        "metadata",
        "capabilities",
        "degraded_channels",
        "coverage",
        "diagnostics",
    ):
        if key not in projection:
            errors.append(f"backend.projection.{key} is required")

    status = manifest.get("status") or {}
    for key in ("state", "warnings", "errors", "diagnostics", "service_urls"):
        if key not in status:
            errors.append(f"status.{key} is required")

    for i, diagnostic in enumerate(status.get("diagnostics") or []):
        if not isinstance(diagnostic, dict):
            errors.append(f"status.diagnostics[{i}] must be an object")
            continue
        for key in (
            "schema_version",
            "code",
            "severity",
            "stage",
            "prim_path",
            "material_name",
            "message",
            "recommended_action",
            "details",
        ):
            if key not in diagnostic:
                errors.append(f"status.diagnostics[{i}].{key} is required")
        if diagnostic.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
            errors.append(
                f"status.diagnostics[{i}].schema_version must be "
                "texture-agent-diagnostic.v1"
            )

    return errors


def _dedupe_diagnostics(diagnostics: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    for diagnostic in diagnostics:
        key = json.dumps(_jsonable(diagnostic), sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(diagnostic)
    return deduped


def _backend_section(context: dict[str, Any]) -> dict[str, Any]:
    texture_config = context.get("texture_config") or {}
    image_gen = texture_config.get("image_gen") or {}
    return {
        "backend": texture_config.get("backend", "simple_image_gen"),
        "model": image_gen.get("model") or texture_config.get("model"),
        "endpoint_type": "service"
        if texture_config.get("backend") == "service"
        else image_gen.get("backend", "nim"),
        "endpoint": "<configured>" if texture_config.get("endpoint") else None,
        "conditioning_support": texture_config.get("conditioning_support"),
        "seed": (context.get("variations_config") or {}).get("seed"),
        "texture_size": texture_config.get("size"),
        "custom_parameters": _redact_sensitive(
            texture_config.get("custom_parameters", {})
        ),
        "projection": _projection_backend_summary(context),
    }


def build_artifacts_manifest(
    context: dict[str, Any],
    *,
    status: str,
    service_urls: dict[str, str] | None = None,
    duration_seconds: int | None = None,
) -> dict[str, Any]:
    """Build the schema-versioned artifacts manifest payload."""
    root = _working_root(context)
    completed_steps = [
        str(step) for step in context.get("pipeline_completed_steps", [])
    ]
    failed_step = context.get("pipeline_failed_step")
    orchestration_failure = status == "failed" and failed_step is not None
    uv_prep = context.get("uv_preparation") or {}
    uv_report_path = uv_prep.get("uv_report_path")
    uv_report = _read_uv_report(uv_report_path)

    output_paths = [str(p) for p in context.get("output_usd_paths", [])]
    output_portability = context.get("output_portability")
    if output_portability is None and output_paths:
        output_portability = validate_output_texture_portability(output_paths[0])

    package_diagnostics = context.get("package_diagnostics") or []
    generation_diagnostics = context.get("generate_textures_diagnostics") or []
    render_diagnostics = context.get("render_diagnostics") or []
    all_diagnostics = [*_redact_sensitive(package_diagnostics)]
    all_diagnostics.extend(_redact_sensitive(generation_diagnostics))
    all_diagnostics.extend(_redact_sensitive(render_diagnostics))
    if output_portability:
        all_diagnostics.extend(
            _redact_sensitive(output_portability.get("diagnostics", []))
        )
    all_diagnostics = _dedupe_diagnostics(all_diagnostics)

    generated = context.get("generated_textures") or {}
    blended = context.get("blended_textures") or {}
    render_paths = [str(p) for p in context.get("rendered_image_paths", [])]
    projection_entries = _projection_backend_entries(context, root)

    return {
        "schema_version": ARTIFACTS_MANIFEST_SCHEMA_VERSION,
        "input": {
            "usd": _path_entry(
                (context.get("config") or {}).get("input", {}).get("usd_path")
                or context.get("usd_path"),
                root,
            ),
            "current_usd": _path_entry(context.get("usd_path"), root),
            "config": _config_summary(context),
            "requested_material_scope": sorted(
                (context.get("material_textures") or {}).keys()
            ),
            "requested_prim_scope": context.get("prim_paths") or [],
        },
        "prepared": {
            "prepared_usd": _path_entry(
                uv_report.get("prepared_usd") if uv_report else context.get("usd_path"),
                root,
            ),
            "uv_report": _path_entry(uv_report_path, root),
            "uv_summary": (uv_report or {}).get("summary", {}),
            "uv_actions": _jsonable(uv_prep),
        },
        "planning": _planning_section(context, root),
        "materials": {
            "discovered": _material_entries(context),
            "selected": _selected_materials(context),
            "auto_prompt_additions": _jsonable(
                context.get("auto_prompt_additions", [])
            ),
        },
        "prompts": {
            "units": _selected_materials(context),
            "prompt_source": "auto_prompt"
            if context.get("auto_prompt_additions")
            else "material_textures",
        },
        "textures": {
            "generated": _texture_set_entries(generated, root),
            "blended": _texture_set_entries(blended, root),
            "projection_backend": projection_entries,
            "generated_count": len(generated),
            "blended_count": len(blended),
            "generation_errors": _redact_sensitive(
                context.get("generate_textures_errors", [])
            ),
            "blend_errors": _redact_sensitive(context.get("blend_textures_errors", [])),
        },
        "outputs": {
            "output_usd": [_path_entry(path, root) for path in output_paths],
            "output_usdz": _path_entry(context.get("output_usdz_path"), root),
            "portability": output_portability
            or {
                "portable": False,
                "texture_reference_count": 0,
                "diagnostics": [],
            },
        },
        "renders": {
            "render_available": bool(render_paths),
            "final": [_path_entry(path, root) for path in render_paths],
            "diagnostics": _redact_sensitive(context.get("render_diagnostics", [])),
        },
        "backend": _backend_section(context),
        "status": {
            "state": status,
            "completed_steps": completed_steps,
            "failed_step": str(failed_step) if orchestration_failure else None,
            "error_code": PIPELINE_FAILURE_CODE if orchestration_failure else None,
            "error": PIPELINE_FAILURE_MESSAGE if orchestration_failure else None,
            "partial_artifacts": (
                _discover_partial_artifacts(context) if orchestration_failure else []
            ),
            "duration_seconds": duration_seconds,
            "timings": _redact_sensitive(context.get("timings", {})),
            "warnings": _redact_sensitive(context.get("warnings", [])),
            "errors": {
                "generate_textures": _redact_sensitive(
                    context.get("generate_textures_errors", [])
                ),
                "blend_textures": _redact_sensitive(
                    context.get("blend_textures_errors", [])
                ),
                "package": _redact_sensitive(package_diagnostics),
            },
            "diagnostics": all_diagnostics,
            "service_urls": service_urls or {},
        },
    }


def build_failed_artifacts_manifest(
    context: dict[str, Any],
    *,
    failed_step: str,
    completed_steps: list[str],
) -> dict[str, Any]:
    """Build a failure manifest without copying provider or exception diagnostics."""
    safe_context = dict(context)
    safe_context.update(
        {
            "pipeline_completed_steps": list(completed_steps),
            "pipeline_failed_step": failed_step,
            "warnings": [],
            "generate_textures_errors": [],
            "blend_textures_errors": [],
            "package_diagnostics": [],
            "generate_textures_diagnostics": [],
            "render_diagnostics": [],
            "render_errors": [],
            "projection_backend_results": {},
            "output_portability": {},
        }
    )
    return build_artifacts_manifest(safe_context, status="failed")


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Serialize JSON to a sibling temporary file and atomically replace ``path``."""
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("x", encoding="utf-8") as manifest_file:
            json.dump(payload, manifest_file, indent=2, sort_keys=True, default=str)
            manifest_file.write("\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def write_artifacts_manifest(
    context: dict[str, Any],
    *,
    status: str = "completed",
    service_urls: dict[str, str] | None = None,
    duration_seconds: int | None = None,
    payload: dict[str, Any] | None = None,
) -> Path:
    """Write ``artifacts_manifest.json`` into the run working directory."""
    working_dir = Path(context["working_dir"])
    working_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = working_dir / "artifacts_manifest.json"
    manifest = payload or build_artifacts_manifest(
        context,
        status=status,
        service_urls=service_urls,
        duration_seconds=duration_seconds,
    )
    _write_json_atomically(manifest_path, manifest)
    context["artifacts_manifest_path"] = str(manifest_path)
    return manifest_path


def write_failed_artifacts_manifest(
    context: dict[str, Any],
    *,
    failed_step: str,
    completed_steps: list[str],
) -> Path:
    """Best-effort caller API for publishing a failure-safe manifest."""
    payload = build_failed_artifacts_manifest(
        context,
        failed_step=failed_step,
        completed_steps=completed_steps,
    )
    return write_artifacts_manifest(context, status="failed", payload=payload)
