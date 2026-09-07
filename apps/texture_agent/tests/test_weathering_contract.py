# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from apps.texture_gen_service_common import WeatheringControls
from pydantic import ValidationError

from texture_agent.functions.material_discovery import MaterialInfo, PrimTextureUnit
from texture_agent.functions.rest_client import RestTextureVariationClient
from texture_agent.functions.texture_generation import (
    Conditioning,
    TextureVariationClient,
    TextureVariationConfig,
)
from texture_agent.tasks.generate_textures import (
    _preflight_simple_image_gen_conditioning,
)


@pytest.mark.parametrize(
    "field_name",
    [
        "effect",
        "density",
        "scale",
        "directionality",
        "direction_degrees",
        "severity",
        "repeat",
    ],
)
def test_public_weathering_contract_rejects_internal_planning_fields(
    field_name: str,
) -> None:
    with pytest.raises(ValidationError):
        WeatheringControls.model_validate({field_name: "not-public"})


def test_rest_request_serializes_only_optional_weathering_masks() -> None:
    body = RestTextureVariationClient._build_request_body(
        source_asset_uri="file:///asset.usda",
        conditioning=Conditioning(text_prompt="localized rust around joints"),
        config=TextureVariationConfig(
            strength=0.8,
            seed=1045,
            weathering=WeatheringControls(
                protected_mask_uri="file:///protected.png",
            ),
        ),
    )

    weathering = body["configuration"]["weathering"]
    assert weathering["protected_mask_uri"] == "file:///protected.png"
    assert weathering["editable_mask_uri"] is None
    assert set(weathering) == {"editable_mask_uri", "protected_mask_uri"}
    assert body["configuration"]["seed"] == 1045
    assert body["configuration"]["strength"] == 0.8


def test_legacy_rest_request_omits_weathering_field() -> None:
    body = RestTextureVariationClient._build_request_body(
        source_asset_uri="file:///asset.usda",
        conditioning=Conditioning(text_prompt="clean material"),
        config=TextureVariationConfig(),
    )

    assert "weathering" not in body["configuration"]


def test_rest_request_validates_mapping_weathering_input() -> None:
    body = RestTextureVariationClient._build_request_body(
        source_asset_uri="file:///asset.usda",
        conditioning=Conditioning(text_prompt="dust"),
        config=TextureVariationConfig(
            weathering={"editable_mask_uri": "file:///editable.png"}  # type: ignore[arg-type]
        ),
    )

    assert (
        body["configuration"]["weathering"]["editable_mask_uri"]
        == "file:///editable.png"
    )


def test_simple_image_backend_rejects_prompted_weathering_before_launch() -> None:
    unit = PrimTextureUnit(
        prim_path="/World/Mesh",
        material_info=MaterialInfo(
            prim_path="/World/Looks/Steel",
            name="Steel",
            bound_prim_paths=["/World/Mesh"],
        ),
        key="Steel",
        prompt="localized rust",
        opacity=1.0,
    )
    context = {
        "material_textures": {
            "Steel": {
                "prompt": "localized rust",
                "weathering": {
                    "protected_mask_uri": "file:///protected.png",
                },
            }
        }
    }

    supported, errors, metadata, diagnostics = _preflight_simple_image_gen_conditioning(
        [unit],
        context,
        {"backend": "simple_image_gen"},
    )

    assert supported == []
    assert errors[0]["type"] == "UnsupportedBackendConditioning"
    assert metadata["Steel"]["skipped_before_backend_launch"] is True
    assert diagnostics[0]["details"]["unsupported_fields"] == ["weathering"]


def test_direct_local_client_rejects_prompted_weathering_before_writes(
    tmp_path,
) -> None:
    output_dir = tmp_path / "outputs"
    status = TextureVariationClient(output_dir=output_dir).generate(
        source_asset_uri="file:///asset.usda",
        conditioning=Conditioning(
            text_prompt="rust the lower metal joints but keep the rubber clean"
        ),
    )

    assert status.status == "failed"
    assert "cannot enforce prompt-requested weathering" in (status.error_message or "")
    assert not output_dir.exists()
