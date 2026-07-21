# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from apps.texture_gen_service_common.artifacts import local_path_from_file_uri
from PIL import Image

from texture_agent.functions.rest_client import RestTextureVariationClient
from texture_agent.functions.texture_generation import (
    BackendCapabilities,
    Conditioning,
    TextureTarget,
    TextureVariationConfig,
)

pytest_plugins = ("fake_projection_backend",)


def _file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    assert parsed.scheme == "file"
    path = local_path_from_file_uri(uri)
    assert path is not None
    return path


def _png_bytes(uri: str) -> bytes:
    return _file_uri_path(uri).read_bytes()


def _assert_png_map(map_info: dict[str, Any], *, expected_size: int) -> None:
    path = _file_uri_path(map_info["uri"])
    assert path.is_file()
    assert map_info["width"] == expected_size
    assert map_info["height"] == expected_size
    assert map_info["mime_type"] == "image/png"
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.size == (expected_size, expected_size)
        assert image.getbbox() is not None


def test_fake_projection_backend_fixture_records_requests(
    fake_projection_backend,
    projection_request_factory,
    projection_submitter,
) -> None:
    request = projection_request_factory("success_full_pbr", texture_size=8)

    response = projection_submitter(fake_projection_backend.endpoint_url, request)

    assert response["status"] == "completed"
    assert len(fake_projection_backend.requests) == 1
    assert fake_projection_backend.requests[0]["target"]["material_name"] == (
        "Aluminum_Matte"
    )


def test_fake_projection_backend_success_outputs_are_deterministic(
    fake_projection_backend,
    projection_request_factory,
    projection_submitter,
) -> None:
    request = projection_request_factory("success_full_pbr", texture_size=8, seed=11631)
    first = projection_submitter(fake_projection_backend.endpoint_url, request)
    second = projection_submitter(fake_projection_backend.endpoint_url, request)

    first_maps = first["result"]["maps"]
    second_maps = second["result"]["maps"]
    assert set(first_maps) == {"albedo", "normal", "orm"}
    assert first["result"]["metadata"]["texture_size"] == 8

    for channel in ("albedo", "normal", "orm"):
        _assert_png_map(first_maps[channel], expected_size=8)
        _assert_png_map(second_maps[channel], expected_size=8)
        assert _png_bytes(first_maps[channel]["uri"]) == _png_bytes(
            second_maps[channel]["uri"]
        )


@pytest.mark.parametrize(
    ("variant", "expected_maps", "expected_codes", "expected_coverage"),
    [
        ("success_full_pbr", {"albedo", "normal", "orm"}, set(), 0.97),
        (
            "albedo_only_degraded",
            {"albedo"},
            {"BACKEND_MAP_MISSING"},
            0.97,
        ),
        (
            "low_coverage_warning",
            {"albedo", "normal", "orm"},
            {"BACKEND_LOW_COVERAGE"},
            0.41,
        ),
        (
            "geometry_return_ignored",
            {"albedo", "normal", "orm"},
            {"BACKEND_GEOMETRY_IGNORED"},
            0.97,
        ),
    ],
)
def test_fake_projection_backend_completed_response_variants(
    fake_projection_backend,
    projection_request_factory,
    projection_submitter,
    variant: str,
    expected_maps: set[str],
    expected_codes: set[str],
    expected_coverage: float,
) -> None:
    request = projection_request_factory(variant, texture_size=12)
    response = projection_submitter(fake_projection_backend.endpoint_url, request)

    result = response["result"]
    maps = result["maps"]
    diagnostics = result["diagnostics"]
    metadata = result["metadata"]

    assert response["status"] == "completed"
    assert result["variant_asset_uri"] == request["source_asset_uri"]
    assert set(maps) == expected_maps
    assert {item["code"] for item in diagnostics} == expected_codes
    assert metadata["backend_name"] == "fake_projection_backend"
    assert metadata["texture_size"] == 12
    assert metadata["coverage"]["target_coverage"] == expected_coverage

    for map_info in maps.values():
        _assert_png_map(map_info, expected_size=12)

    if variant == "albedo_only_degraded":
        assert result["generated_textures"]["normal"] is None
        assert result["generated_textures"]["orm"] is None
        assert metadata["degraded_channels"] == ["normal", "orm"]
        assert metadata["capabilities"]["normal_map"] is False
        assert metadata["capabilities"]["orm"] is False

    if variant == "geometry_return_ignored":
        geometry = result["auxiliary_artifacts"]["geometry"]
        assert geometry
        assert _file_uri_path(geometry[0]["uri"]).is_file()


@pytest.mark.parametrize(
    ("variant", "source_asset_uri", "expected_maps", "expected_code"),
    [
        (
            "missing_albedo",
            "file:///work/ladder/prepared_input.usd",
            {"normal"},
            "BACKEND_MAP_MISSING",
        ),
        ("bad_uri", "not-a-uri", set(), "BACKEND_PARTIAL_FAILURE"),
    ],
)
def test_fake_projection_backend_failed_response_variants(
    fake_projection_backend,
    projection_request_factory,
    projection_submitter,
    variant: str,
    source_asset_uri: str,
    expected_maps: set[str],
    expected_code: str,
) -> None:
    request = projection_request_factory(
        variant,
        texture_size=10,
        source_asset_uri=source_asset_uri,
    )
    response = projection_submitter(fake_projection_backend.endpoint_url, request)

    result = response["result"]
    assert response["status"] == "failed"
    assert response["error_message"]
    assert set(result["maps"]) == expected_maps
    assert result["generated_textures"]["albedo"] is None
    assert result["diagnostics"][0]["severity"] == "error"
    assert result["diagnostics"][0]["code"] == expected_code

    for map_info in result["maps"].values():
        _assert_png_map(map_info, expected_size=10)


def test_fake_projection_backend_round_trips_through_rest_client(
    fake_projection_backend,
) -> None:
    status = RestTextureVariationClient(
        fake_projection_backend.endpoint_url, timeout=5
    ).generate(
        source_asset_uri="file:///work/ladder/prepared_input.usd",
        conditioning=Conditioning(text_prompt="matte aluminum"),
        config=TextureVariationConfig(
            seed=11631,
            variant_name="Aluminum_Matte",
            engine="fake_projection",
            texture_size=8,
            custom_parameters={"variant": "low_coverage_warning"},
        ),
        target=TextureTarget(
            material_name="Aluminum_Matte",
            material_path="/RootNode/Looks/Aluminum_Matte",
            prim_paths=["/RootNode/SM_Ladder_A/SM_Ladder_A_Aluminum_0"],
        ),
        capabilities=BackendCapabilities(coverage=True, geometry_output="none"),
    )

    assert len(fake_projection_backend.requests) == 1
    assert fake_projection_backend.requests[0]["capabilities"]["coverage"] is True

    assert status.status == "completed"
    assert status.result is not None
    assert set(status.result.maps) == {"albedo", "normal", "orm"}
    assert status.result.metadata["coverage"]["target_coverage"] == 0.41
    assert status.result.diagnostics[0]["code"] == "BACKEND_LOW_COVERAGE"
