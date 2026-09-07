# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted JSON adapter for deterministic material package authoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from content_agent_workflows.common.artifacts import atomic_write_json

if TYPE_CHECKING:
    from material_agent.material_library_generation import (
        SourceMaterialReference,
        TextureMapSet,
    )
MATERIAL_AUTHOR_TOOL_REQUEST_SCHEMA_VERSION = (
    "content-workflow-material-author-tool-request.v1"
)
MATERIAL_AUTHOR_TOOL_RESULT_SCHEMA_VERSION = (
    "content-workflow-material-author-tool-result.v1"
)


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("material author tool request must be valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("material author tool request must contain a JSON object")
    return data


def _resolve_input_path(value: object, *, base_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    candidate = Path(value).expanduser()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (base_dir / candidate).resolve()
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _source_reference(
    value: object,
    *,
    base_dir: Path,
    label: str,
) -> SourceMaterialReference | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object or null")
    digest = value.get("usd_sha256")
    if not isinstance(digest, str) or not digest:
        raise ValueError(f"{label}.usd_sha256 is required")
    from material_agent.material_library_generation import SourceMaterialReference

    return SourceMaterialReference(
        usd_path=_resolve_input_path(
            value.get("usd_path"),
            base_dir=base_dir,
            label=f"{label}.usd_path",
        ),
        material_prim_path=str(value.get("material_prim_path", "")),
        usd_sha256=digest,
    )


def _textures(value: object, *, base_dir: Path) -> TextureMapSet | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"albedo", "normal", "orm"}:
        raise ValueError("textures must contain exactly albedo, normal, and orm")
    from material_agent.material_library_generation import TextureMapSet

    return TextureMapSet(
        albedo=_resolve_input_path(
            value["albedo"], base_dir=base_dir, label="textures.albedo"
        ),
        normal=_resolve_input_path(
            value["normal"], base_dir=base_dir, label="textures.normal"
        ),
        orm=_resolve_input_path(value["orm"], base_dir=base_dir, label="textures.orm"),
    )


def author_from_json(request_path: Path, output_dir: Path) -> dict[str, Any]:
    """Validate one data-only request and publish it through the fixed API."""

    try:
        from material_agent.material_library_generation import (
            MaterialAuthoringRequest,
            MaterialRecipe,
            author_material_package,
        )
    except ImportError as exc:
        raise RuntimeError(
            "material authoring requires content-workflow-cli[materials]"
        ) from exc

    resolved_request = request_path.expanduser().resolve(strict=True)
    data = _load_mapping(resolved_request)
    if data.get("schema_version") != MATERIAL_AUTHOR_TOOL_REQUEST_SCHEMA_VERSION:
        raise ValueError("material author tool request has the wrong schema_version")
    recipe_data = data.get("recipe")
    if not isinstance(recipe_data, dict):
        raise ValueError("material author tool request requires a recipe object")
    target_prim_paths = data.get("target_prim_paths", ())
    if not isinstance(target_prim_paths, list | tuple):
        raise ValueError("target_prim_paths must be an array")
    base_dir = resolved_request.parent
    request = MaterialAuthoringRequest(
        operation=str(data.get("operation", "")),
        recipe=MaterialRecipe.from_dict(recipe_data, base_dir=base_dir),
        source=_source_reference(data.get("source"), base_dir=base_dir, label="source"),
        provenance_source=_source_reference(
            data.get("provenance_source"),
            base_dir=base_dir,
            label="provenance_source",
        ),
        textures=_textures(data.get("textures"), base_dir=base_dir),
        target_prim_paths=tuple(str(path) for path in target_prim_paths),
        material_profile=str(data.get("material_profile", "auto")),
        recipe_semantics=str(data.get("recipe_semantics", "literal_shader_values")),
        representation_policy=str(data.get("representation_policy", "preserve")),
    )
    package = author_material_package(
        request,
        output_dir.expanduser().resolve(),
    )
    return {
        "schema_version": MATERIAL_AUTHOR_TOOL_RESULT_SCHEMA_VERSION,
        "request_id": package.request_id,
        "material_package": package.to_dict(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish one material package from a typed JSON recipe without "
            "executing model-authored code."
        )
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--result", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = author_from_json(args.request, args.output_dir)
    if args.result is not None:
        result_path = args.result.expanduser().resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(result_path, result, within=result_path.parent)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
