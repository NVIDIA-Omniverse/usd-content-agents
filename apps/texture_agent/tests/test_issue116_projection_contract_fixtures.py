# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "issue116_projection_backend"

REQUIRED_CAPABILITIES = {
    "image_conditioning",
    "multiview",
    "normal_map",
    "orm",
    "masks",
    "coverage",
    "geometry_output",
}

RESERVED_DIAGNOSTIC_CODES = {
    "BACKEND_CAPABILITY_MISSING",
    "BACKEND_CONDITIONING_UNSUPPORTED",
    "BACKEND_MAP_MISSING",
    "BACKEND_MAP_VALIDATION_FAILED",
    "BACKEND_TEXTURE_BLANK",
    "BACKEND_PARTIAL_FAILURE",
    "BACKEND_LOW_COVERAGE",
    "BACKEND_GEOMETRY_IGNORED",
}


def _load_fixture(name: str) -> dict[str, Any]:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    return data


def test_issue116_ladder_request_fixture_contract() -> None:
    request = _load_fixture("request_ladder_aluminum_matte.json")

    assert request["source_asset_uri"].endswith("ladder_uv_ready.usd")
    assert request["target"] == {
        "material_name": "Aluminum_Matte",
        "material_path": "/RootNode/Looks/Aluminum_Matte",
        "prim_paths": ["/RootNode/SM_Ladder_A/SM_Ladder_A_Aluminum_0"],
        "mode": "per_material",
        "strict_scope": True,
    }
    assert request["conditioning"]["text_prompt"]
    assert request["conditioning"]["reference_image_uris"]
    assert request["configuration"]["seed"] == 11631
    assert request["configuration"]["texture_size"] == 1024
    assert set(request["capabilities"]) == REQUIRED_CAPABILITIES


@pytest.mark.parametrize(
    "name",
    [
        "response_success_full_pbr.json",
        "response_degraded_low_coverage.json",
        "response_failure_missing_albedo.json",
    ],
)
def test_issue116_response_fixture_contract(name: str) -> None:
    response = _load_fixture(name)

    assert response["status"] in {"completed", "failed"}
    assert isinstance(response["job_id"], str)

    result = response["result"]
    assert set(result["generated_textures"]) == {"albedo", "normal", "orm"}
    assert isinstance(result["maps"], dict)
    assert isinstance(result["auxiliary_artifacts"], dict)
    assert isinstance(result["diagnostics"], list)

    metadata = result["metadata"]
    for key in (
        "backend_name",
        "model",
        "endpoint_type",
        "seed",
        "texture_size",
        "timings_ms",
        "custom_parameter_summary",
        "capabilities",
        "coverage",
        "degraded_channels",
    ):
        assert key in metadata
    assert set(metadata["capabilities"]) == REQUIRED_CAPABILITIES

    for channel, map_info in result["maps"].items():
        assert channel in {
            "albedo",
            "normal",
            "orm",
            "roughness",
            "metalness",
            "occlusion",
        }
        assert map_info["uri"].startswith("file://")
        assert map_info["width"] == 1024
        assert map_info["height"] == 1024
        assert map_info["mime_type"] == "image/png"
        assert "colorspace" in map_info

    for diagnostic in result["diagnostics"]:
        assert diagnostic["schema_version"] == "texture-agent-diagnostic.v1"
        assert diagnostic["code"] in RESERVED_DIAGNOSTIC_CODES
        assert diagnostic["severity"] in {"info", "warning", "error"}
        assert diagnostic["stage"] == "generate_textures"
        assert diagnostic["material_name"] == "Aluminum_Matte"
        assert diagnostic["prim_path"] == "/RootNode/SM_Ladder_A/SM_Ladder_A_Aluminum_0"
        assert isinstance(diagnostic["details"], dict)


def test_issue116_degraded_fixture_records_low_coverage_and_ignored_geometry() -> None:
    response = _load_fixture("response_degraded_low_coverage.json")
    result = response["result"]

    assert result["metadata"]["degraded_channels"] == ["normal", "orm"]
    assert result["metadata"]["coverage"]["target_coverage"] == 0.41
    assert result["auxiliary_artifacts"]["geometry"]

    diagnostic_codes = {item["code"] for item in result["diagnostics"]}
    assert "BACKEND_LOW_COVERAGE" in diagnostic_codes
    assert "BACKEND_GEOMETRY_IGNORED" in diagnostic_codes
