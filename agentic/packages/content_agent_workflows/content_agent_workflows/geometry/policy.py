# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing policy for the provider-neutral geometry workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from content_agent_workflows.convert_to_usd.workflow import (
    USD_CONVERT_CAD_EXTENSIONS,
    converter_reference_for_source_extension,
)

GeometryRoute = Literal[
    "provided_cad_repair",
    "provided_mesh_repair",
    "text_to_cad_generate",
    "image_to_cad_generate",
    "class_backed_asset_gen",
]

GeometrySourceCategory = Literal[
    "source_bundle",
    "existing_usd",
    "generated_usd",
    "convertible_external",
    "mesh_recovery_candidate",
    "unsupported_external",
    "text_generation",
    "image_generation",
    "class_backed_generation",
]

USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
MESH_RECOVERY_SUFFIXES = {".obj", ".stl"}


class GeometryRouteDecision(BaseModel):
    """One routing decision for a geometry request."""

    model_config = ConfigDict(extra="forbid")

    route: GeometryRoute
    source_category: GeometrySourceCategory
    input_modality: Literal[
        "provided_cad", "provided_mesh", "text", "image", "text_image"
    ]
    source_path: str | None = None
    prompt: str | None = None
    requires_generation: bool
    requires_scene_optimization: bool = True
    optional_bridge: str | None = None
    source_manifest_path: str | None = None
    representation_role: str | None = None
    program_digest: str | None = None
    program_schema_version: str | None = None
    rationale: str = Field(min_length=1)


def _suffix(path: str | Path | None) -> str:
    if path is None:
        return ""
    return Path(path).suffix.lower()


def _looks_class_backed(prompt: str | None) -> bool:
    if not prompt:
        return False
    lowered = prompt.lower()
    return any(
        token in lowered
        for token in (
            "robocasa",
            "ycb",
            "mcmaster",
            "mcmaster-carr",
            "part number",
            "catalog",
            "known class",
        )
    )


def route_geometry_request(
    *,
    source_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
    source_representation_role: str | None = None,
    generated_usd_path: str | Path | None = None,
    prompt: str | None = None,
    image_path: str | Path | None = None,
) -> GeometryRouteDecision:
    """Route geometry work using declared roles before extension fallbacks."""

    source_suffix = _suffix(source_path)
    generated_suffix = _suffix(generated_usd_path)
    if source_manifest_path is not None:
        return GeometryRouteDecision(
            route="provided_cad_repair",
            source_category="source_bundle",
            input_modality="provided_cad",
            source_path=str(source_path) if source_path is not None else None,
            source_manifest_path=str(source_manifest_path),
            representation_role=source_representation_role,
            prompt=prompt,
            requires_generation=False,
            rationale=(
                "An immutable geometry.source.v1 bundle selects a digest-bound "
                "exported representation for shared validation and handoff."
            ),
        )
    if generated_suffix in USD_SUFFIXES:
        return GeometryRouteDecision(
            route="provided_cad_repair",
            source_category="generated_usd",
            input_modality="provided_cad",
            source_path=str(generated_usd_path),
            prompt=prompt,
            requires_generation=False,
            rationale="Generated USD/CAD handoff is already available and should be optimized, validated, and packaged.",
        )
    if generated_usd_path is not None:
        return GeometryRouteDecision(
            route="provided_cad_repair",
            source_category="unsupported_external",
            input_modality="provided_cad",
            source_path=str(generated_usd_path),
            prompt=prompt,
            requires_generation=False,
            rationale=(
                "generated_usd_path was provided but is not a supported USD "
                "artifact; source preparation must report the blocker."
            ),
        )
    if source_suffix in USD_SUFFIXES:
        return GeometryRouteDecision(
            route="provided_cad_repair",
            source_category="existing_usd",
            input_modality="provided_cad",
            source_path=str(source_path),
            prompt=prompt,
            requires_generation=False,
            rationale="Existing CAD/USD/URDF input should be repaired, optimized, converted, and validated.",
        )
    if source_suffix in MESH_RECOVERY_SUFFIXES:
        return GeometryRouteDecision(
            route="provided_mesh_repair",
            source_category="mesh_recovery_candidate",
            input_modality="provided_mesh",
            source_path=str(source_path),
            prompt=prompt,
            requires_generation=False,
            rationale=(
                "Mesh input is convertible through the shared workflow and is "
                "eligible for explicit opt-in parametric recovery."
            ),
        )
    if source_path is not None and (
        source_suffix in USD_CONVERT_CAD_EXTENSIONS
        or converter_reference_for_source_extension(source_path) is not None
    ):
        return GeometryRouteDecision(
            route="provided_cad_repair",
            source_category="convertible_external",
            input_modality="provided_cad",
            source_path=str(source_path),
            prompt=prompt,
            requires_generation=False,
            rationale=(
                "External source should be converted by the shared "
                "convert-to-USD workflow before Geometry validation."
            ),
        )
    if source_path is not None:
        return GeometryRouteDecision(
            route="provided_cad_repair",
            source_category="unsupported_external",
            input_modality="provided_cad",
            source_path=str(source_path),
            prompt=prompt,
            requires_generation=False,
            rationale=(
                "A source artifact was provided, but no native or shared "
                "conversion route recognizes its format. Authoring providers "
                "must export a supported immutable representation first."
            ),
        )
    if image_path is not None and prompt:
        return GeometryRouteDecision(
            route="image_to_cad_generate",
            source_category="image_generation",
            input_modality="text_image",
            source_path=str(image_path),
            prompt=prompt,
            requires_generation=True,
            optional_bridge="product-image-decomposer",
            rationale="Text plus image input needs image decomposition or reconstruction before CAD enhancement.",
        )
    if image_path is not None:
        return GeometryRouteDecision(
            route="image_to_cad_generate",
            source_category="image_generation",
            input_modality="image",
            source_path=str(image_path),
            requires_generation=True,
            optional_bridge="product-image-decomposer",
            rationale="Image-only input needs reconstruction before CAD enhancement.",
        )
    if _looks_class_backed(prompt):
        return GeometryRouteDecision(
            route="class_backed_asset_gen",
            source_category="class_backed_generation",
            input_modality="text",
            prompt=prompt,
            requires_generation=True,
            optional_bridge="asset_gen",
            rationale="Prompt appears to reference a catalog or known object class; use a class-backed generator when configured.",
        )
    return GeometryRouteDecision(
        route="text_to_cad_generate",
        source_category="text_generation",
        input_modality="text",
        prompt=prompt,
        requires_generation=True,
        rationale=(
            "Novel text input requires an explicitly configured geometry authoring "
            "provider before shared USD enhancement."
        ),
    )
