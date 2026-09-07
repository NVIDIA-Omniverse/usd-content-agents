# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative envelope checks for connector geometry artifacts."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from .errors import InvalidProviderResponseError, UnsupportedCapabilityError

GEOMETRY_MEDIA_TYPES: dict[str, tuple[str, str]] = {
    ".step": ("cad_geometry", "model/step"),
    ".stp": ("cad_geometry", "model/step"),
    ".stl": ("mesh_geometry", "model/stl"),
    ".obj": ("mesh_geometry", "model/obj"),
    ".ply": ("mesh_geometry", "application/ply"),
    ".glb": ("render_geometry", "model/gltf-binary"),
    ".gltf": ("render_geometry", "model/gltf+json"),
    ".3mf": ("mesh_geometry", "model/3mf"),
    ".usd": ("render_geometry", "model/vnd.usd"),
    ".usda": ("render_geometry", "model/vnd.usda"),
    ".usdc": ("render_geometry", "model/vnd.usdc"),
}
SUPPORTING_ASSET_MEDIA_TYPES: dict[str, str] = {
    ".bin": "application/octet-stream",
    ".json": "application/json",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".mtl": "model/mtl",
    ".png": "image/png",
    ".webp": "image/webp",
}
_MAX_JSON_SUPPORTING_ASSET_BYTES = 4 * 1024 * 1024
_MAX_JSON_SUPPORTING_ASSET_DEPTH = 32
_MAX_JSON_SUPPORTING_ASSET_VALUES = 100_000


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _validate_json_supporting_asset(content: bytes) -> None:
    if len(content) > _MAX_JSON_SUPPORTING_ASSET_BYTES:
        raise InvalidProviderResponseError("JSON supporting asset exceeds the 4-MiB limit")
    try:
        document = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise InvalidProviderResponseError(
            "JSON supporting asset must contain standard UTF-8 JSON"
        ) from exc
    if not isinstance(document, dict | list):
        raise InvalidProviderResponseError("JSON supporting asset must contain an object or array")
    stack: list[tuple[object, int]] = [(document, 0)]
    value_count = 0
    while stack:
        value, depth = stack.pop()
        value_count += 1
        if value_count > _MAX_JSON_SUPPORTING_ASSET_VALUES:
            raise InvalidProviderResponseError(
                "JSON supporting asset exceeds the value-count limit"
            )
        if depth > _MAX_JSON_SUPPORTING_ASSET_DEPTH:
            raise InvalidProviderResponseError(
                "JSON supporting asset exceeds the nesting-depth limit"
            )
        if isinstance(value, dict):
            if any(len(key) > 1_024 for key in value):
                raise InvalidProviderResponseError(
                    "JSON supporting asset contains an oversized object key"
                )
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise InvalidProviderResponseError("JSON supporting asset contains a non-finite number")


def _validate_step(content: bytes) -> None:
    normalized = content.strip().upper()
    if not normalized.startswith(b"ISO-10303-21;") or not normalized.endswith(b"END-ISO-10303-21;"):
        raise InvalidProviderResponseError("STEP export is missing its required envelope")


def _validate_stl(content: bytes) -> None:
    if len(content) >= 84:
        triangle_count = struct.unpack("<I", content[80:84])[0]
        if len(content) == 84 + 50 * triangle_count:
            return
    lowered_lines = content.strip().lower().splitlines()
    if (
        not lowered_lines
        or not lowered_lines[0].lstrip().startswith(b"solid")
        or not any(line.lstrip().startswith(b"facet normal") for line in lowered_lines)
        or not lowered_lines[-1].lstrip().startswith(b"endsolid")
    ):
        raise InvalidProviderResponseError(
            "STL export is neither valid binary nor bounded ASCII STL"
        )


def _validate_obj(content: bytes) -> None:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise InvalidProviderResponseError("OBJ export must be UTF-8 text") from exc
    stripped = [line.lstrip() for line in lines]
    if not any(line.startswith("v ") for line in stripped) or not any(
        line.startswith("f ") for line in stripped
    ):
        raise InvalidProviderResponseError("OBJ export requires vertices and faces")


def _validate_gltf(content: bytes) -> None:
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidProviderResponseError("glTF export must be valid UTF-8 JSON") from exc
    if not isinstance(document, dict) or not isinstance(document.get("asset"), dict):
        raise InvalidProviderResponseError("glTF export is missing asset metadata")
    if not isinstance(document["asset"].get("version"), str):
        raise InvalidProviderResponseError("glTF export is missing an asset version")


def validate_geometry_bytes(filename: str, content: bytes) -> tuple[str, str]:
    """Return the canonical role/media type after a format envelope check."""

    extension = Path(filename).suffix.lower()
    definition = GEOMETRY_MEDIA_TYPES.get(extension)
    if definition is None:
        raise UnsupportedCapabilityError(
            f"unsupported connector geometry format: {extension or '<none>'}"
        )
    if not content:
        raise InvalidProviderResponseError("connector geometry artifact is empty")
    if extension in {".step", ".stp"}:
        _validate_step(content)
    elif extension == ".stl":
        _validate_stl(content)
    elif extension == ".obj":
        _validate_obj(content)
    elif extension == ".ply" and not content.startswith((b"ply\n", b"ply\r\n")):
        raise InvalidProviderResponseError("PLY export has an invalid header")
    elif extension == ".glb":
        if len(content) < 12 or content[:4] != b"glTF":
            raise InvalidProviderResponseError("GLB export has an invalid header")
        declared_length = struct.unpack("<I", content[8:12])[0]
        if declared_length != len(content):
            raise InvalidProviderResponseError("GLB export has an invalid declared length")
    elif extension == ".gltf":
        _validate_gltf(content)
    elif extension == ".3mf" and content[:4] != b"PK\x03\x04":
        raise InvalidProviderResponseError("3MF export is not a ZIP-based container")
    elif extension in {".usd", ".usdc"} and not (
        content.startswith(b"PXR-USDC") or content.startswith(b"#usda")
    ):
        raise InvalidProviderResponseError("USD export has an invalid crate or USDA header")
    elif extension == ".usda" and not content.startswith(b"#usda"):
        raise InvalidProviderResponseError("USDA export has an invalid header")
    return definition


def validate_supporting_asset_bytes(filename: str, media_type: str, content: bytes) -> None:
    """Validate one non-runnable geometry sidecar against a narrow format allowlist."""

    extension = Path(filename).suffix.lower()
    expected_media_type = SUPPORTING_ASSET_MEDIA_TYPES.get(extension)
    if expected_media_type is None or media_type != expected_media_type:
        raise InvalidProviderResponseError(
            f"unsupported connector supporting-asset format: {extension or '<none>'}"
        )
    if not content:
        raise InvalidProviderResponseError("connector supporting asset is empty")
    if extension == ".json":
        _validate_json_supporting_asset(content)
    elif extension == ".mtl":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidProviderResponseError(
                "material-library supporting asset must be UTF-8 text"
            ) from exc
        if "\x00" in text:
            raise InvalidProviderResponseError("material-library supporting asset contains NUL")
    elif extension == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise InvalidProviderResponseError("PNG supporting asset has an invalid header")
    elif extension in {".jpeg", ".jpg"} and not (
        len(content) >= 4 and content.startswith(b"\xff\xd8") and content.endswith(b"\xff\xd9")
    ):
        raise InvalidProviderResponseError("JPEG supporting asset has an invalid envelope")
    elif extension == ".webp" and not (
        len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP"
    ):
        raise InvalidProviderResponseError("WebP supporting asset has an invalid header")
