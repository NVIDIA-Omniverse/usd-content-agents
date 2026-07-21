# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request model validation tests for the texture-agent service."""

import pytest
from pydantic import ValidationError

from ...service.models.requests import (
    MaterialTextures,
    PrimTextureOverride,
    RegenerateRequest,
    TextureDetailPolicy,
    TexturePipelineStep,
)


def test_material_textures_strips_nested_per_prim_prompt() -> None:
    payload = MaterialTextures(
        root={
            "Steel": {
                "prompt": " weathered steel ",
                "per_prim": {
                    "/World/Rung_01": {"prompt": " scrape marks "},
                },
            }
        }
    )

    assert payload.as_config() == {
        "Steel": {
            "prompt": "weathered steel",
            "per_prim": {
                "/World/Rung_01": {"prompt": "scrape marks"},
            },
        }
    }


def test_material_textures_allows_projection_conditioning_fields() -> None:
    payload = MaterialTextures(
        root={
            "Aluminum_Matte": {
                "prompt": "scuffed aluminum",
                "reference_image_uris": [" file:///ref.png "],
                "turntable_video_uri": " file:///turntable.mp4 ",
                "multiview_image_uris": [" file:///view0.png "],
            }
        }
    )

    assert payload.as_config()["Aluminum_Matte"] == {
        "prompt": "scuffed aluminum",
        "reference_image_uris": ["file:///ref.png"],
        "turntable_video_uri": "file:///turntable.mp4",
        "multiview_image_uris": ["file:///view0.png"],
    }


def test_material_textures_allows_target_fields() -> None:
    payload = MaterialTextures(
        root={
            "Toolbox_Body": {
                "prompt": "rusty dark plastic",
                "material_path": " /Root/Looks/Toolbox_Body ",
                "prim_paths": [" /Root/Geometry/body "],
            }
        }
    )

    assert payload.as_config()["Toolbox_Body"] == {
        "prompt": "rusty dark plastic",
        "material_path": "/Root/Looks/Toolbox_Body",
        "prim_paths": ["/Root/Geometry/body"],
    }


def test_material_textures_allows_detail_policy_fields() -> None:
    payload = MaterialTextures(
        root={
            "Plastic_Green": {
                "prompt": "matte green solder mask",
                "detail_policy": TextureDetailPolicy.SURFACE_ONLY,
                "per_prim": {
                    "/World/PCB": {"detail_policy": "default"},
                },
            }
        }
    )

    assert payload.as_config()["Plastic_Green"] == {
        "prompt": "matte green solder mask",
        "detail_policy": "surface_only",
        "per_prim": {
            "/World/PCB": {"detail_policy": "default"},
        },
    }


def test_material_textures_rejects_unknown_detail_policy() -> None:
    with pytest.raises(ValidationError, match="surface_detail"):
        MaterialTextures(
            root={
                "Plastic_Green": {
                    "prompt": "matte green solder mask",
                    "detail_policy": "surface_detail",
                }
            }
        )


def test_material_textures_rejects_empty_per_prim_override() -> None:
    with pytest.raises(
        ValidationError,
        match="Per-prim override must include prompt, opacity, or detail_policy",
    ):
        MaterialTextures(
            root={
                "Steel": {
                    "prompt": "weathered steel",
                    "per_prim": {"/World/Rung_01": {}},
                }
            }
        )


def test_optional_request_model_fields_accept_none_and_strip_empty_values() -> None:
    prim_override = PrimTextureOverride(opacity=0.5, prompt=None)
    assert prim_override.prompt is None

    payload = MaterialTextures(
        root={
            "Steel": {
                "prompt": "brushed steel",
                "prim_paths": None,
                "reference_image_uris": None,
                "multiview_image_uris": None,
                "material_path": None,
                "prim_path": None,
                "turntable_video_uri": None,
                "per_prim": None,
            }
        }
    )
    assert payload.as_config() == {"Steel": {"prompt": "brushed steel"}}


def test_material_and_regenerate_overrides_reject_blank_keys() -> None:
    with pytest.raises(ValidationError, match="Material override keys"):
        MaterialTextures(root={" ": {"prompt": "brushed steel"}})

    with pytest.raises(ValidationError, match="Per-prim override keys"):
        MaterialTextures(
            root={
                "Steel": {
                    "prompt": "brushed steel",
                    "per_prim": {" ": {"prompt": "scratched"}},
                }
            }
        )

    request = RegenerateRequest(
        steps=[TexturePipelineStep.GENERATE_TEXTURES],
        material_textures=None,
    )
    assert request.material_textures is None


def test_regenerate_request_without_material_textures_has_no_override() -> None:
    request = RegenerateRequest(steps=[TexturePipelineStep.GENERATE_TEXTURES])

    assert request.material_textures_config() is None


def test_regenerate_request_accepts_only_canonical_target_unit_ids() -> None:
    unit_id = "tu_0123456789abcdefabcd"
    assert (
        RegenerateRequest(
            steps=[TexturePipelineStep.GENERATE_TEXTURES],
            texture_unit_ids=None,
        ).texture_unit_ids
        is None
    )
    request = RegenerateRequest(
        steps=[TexturePipelineStep.GENERATE_TEXTURES],
        texture_unit_ids=[unit_id],
    )

    assert request.texture_unit_ids == [unit_id]

    with pytest.raises(ValidationError, match="canonical"):
        RegenerateRequest(
            steps=[TexturePipelineStep.GENERATE_TEXTURES],
            texture_unit_ids=["material-name"],
        )
    with pytest.raises(ValidationError, match="canonical"):
        RegenerateRequest(
            steps=[TexturePipelineStep.GENERATE_TEXTURES],
            texture_unit_ids=["tu_0123456789ABCDEFabcd"],
        )
    with pytest.raises(ValidationError, match="requires the generate_textures"):
        RegenerateRequest(
            steps=[TexturePipelineStep.APPLY_TEXTURES],
            texture_unit_ids=[unit_id],
        )
