# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from apps.texture_gen_service_common import CreateJobRequest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "step1x_service_requests"
REPO_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_CAPABILITIES = {
    "image_conditioning",
    "multiview",
    "normal_map",
    "orm",
    "masks",
    "coverage",
    "geometry_output",
}


def _load_fixture(name: str) -> dict[str, Any]:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    return data


@pytest.mark.parametrize(
    ("fixture_name", "expected_seed", "expected_material"),
    [
        (
            "request_cleaning_bucket_opaque_metal.json",
            42,
            "opaque__metal__cleaning_bucket_a",
        ),
        (
            "request_steel_rolling_scaffold_steel_a.json",
            4202,
            "Steel_A",
        ),
        (
            "request_steel_rolling_scaffold_metalpainted_yellow.json",
            4202,
            "MetalPainted_Yellow_Glossy_A",
        ),
    ],
)
def test_step1x_request_fixtures_match_texture_variation_contract(
    fixture_name: str,
    expected_seed: int,
    expected_material: str,
) -> None:
    request = _load_fixture(fixture_name)
    CreateJobRequest.model_validate(request)

    assert set(request) == {
        "source_asset_uri",
        "conditioning",
        "configuration",
        "target",
        "capabilities",
    }
    assert request["source_asset_uri"].startswith("file:///work/texture_step1x/")
    assert request["conditioning"]["text_prompt"]
    assert request["conditioning"]["reference_image_uris"] == []
    assert request["configuration"]["engine"] == "step1x"
    assert request["configuration"]["seed"] == expected_seed
    assert request["configuration"]["texture_size"] == 1024
    assert request["target"]["material_name"] == expected_material
    assert request["target"]["strict_scope"] is True
    assert set(request["capabilities"]) == REQUIRED_CAPABILITIES
    assert request["capabilities"]["geometry_output"] == "none"


def test_step1x_material_anything_request_fixture_is_opt_in() -> None:
    request = _load_fixture("request_cleaning_bucket_opaque_metal_ma.json")
    CreateJobRequest.model_validate(request)

    custom = request["configuration"]["custom_parameters"]
    assert request["configuration"]["variant_name"].endswith("_ma")
    assert request["target"]["material_name"] == "opaque__metal__cleaning_bucket_a"
    assert custom["skip_material_anything"] is False
    assert custom["ma_steps"] == 10
    assert custom["upscale"] is False


@pytest.mark.parametrize(
    "compose_path",
    (
        "apps/texture_agent_service/docker-compose.yml",
        "apps/texture_agent_service/docker-compose.step1x.yml",
        "apps/texture_gen_step1x_service/docker-compose.yml",
    ),
)
def test_public_texture_compose_ports_bind_loopback_by_default(
    compose_path: str,
) -> None:
    compose = yaml.safe_load((REPO_ROOT / compose_path).read_text(encoding="utf-8"))

    published_ports = [
        port
        for service in compose["services"].values()
        for port in service.get("ports", ())
    ]

    assert published_ports
    assert all(
        str(port).startswith("${WU_COMPOSE_BIND_HOST:-127.0.0.1}:")
        for port in published_ports
    )
