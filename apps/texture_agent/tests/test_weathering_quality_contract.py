# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apps.texture_gen_service_common.weathering_intent import (
    prompt_requests_weathering,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def test_weathering_benchmark_uses_pinned_public_fixture_identities() -> None:
    benchmark = _load(FIXTURES / "weathering_quality" / "contract.json")
    benchmark_sources = {
        fixture["id"]: fixture["source"] for fixture in benchmark["fixtures"]
    }
    expected_sources = {
        "cleaning-bucket-public": {
            "kind": "huggingface_dataset_file",
            "repository": "nvidia/PhysicalAI-SimReady-Warehouse-01",
            "revision": "c7fe115cb79c7ddbd0532630d7768b5736b0ecc4",
            "path": (
                "Props/general/HandManipulation/cleaning_bucket_a/"
                "sm_cleaning_bucket_iron_a01_simready_01.usd"
            ),
            "sha256": (
                "963b78395bed492237df76eda5f650659c777411f6b3f8cada364f658d0a7d11"
            ),
        },
        "steel-rolling-scaffold-uv-sensitive": {
            "kind": "huggingface_dataset_file",
            "repository": "nvidia/PhysicalAI-SimReady-Warehouse-01",
            "revision": "c7fe115cb79c7ddbd0532630d7768b5736b0ecc4",
            "path": (
                "Props/general/SM_SteelRollingScaffold_A01_01/"
                "SM_SteelRollingScaffold_A01_01.usd"
            ),
            "sha256": (
                "9dd9bd2a30c423efb8b42dd67fe3237d53a8f2a92304eb487a25538894b25a75"
            ),
        },
        "ladder-checked-in-multi-material": {
            "kind": "repository_file",
            "path": "apps/texture_agent/data/examples/ladder/sources/usd/ladder.usd",
            "sha256": (
                "be8ff3bf74539c422b9a5051c7adf42e139717f946acf8ca796346eecb1960b2"
            ),
        },
    }

    assert benchmark_sources == expected_sources


def test_weathering_benchmark_covers_metal_rust_and_dielectric_dust() -> None:
    benchmark = _load(FIXTURES / "weathering_quality" / "contract.json")
    effects = {
        case["expected_material_behavior"]["weathering_class"]
        for case in benchmark["cases"]
    }

    assert effects >= {"rust", "dust"}
    assert all(len(case["seeds"]) >= 3 for case in benchmark["cases"])
    assert all(case["prompt"] for case in benchmark["cases"])
    assert all(case["control_prompt"] for case in benchmark["cases"])
    assert benchmark["internal_quality_gates"]["strict_material_scope"] is True


def test_weathering_benchmark_keeps_normalized_controls_out_of_public_requests() -> (
    None
):
    benchmark = _load(FIXTURES / "weathering_quality" / "contract.json")
    internal_fields = {
        "effect",
        "density",
        "scale",
        "directionality",
        "direction_degrees",
        "severity",
    }

    for case in benchmark["cases"]:
        public_controls = case["optional_public_controls"]
        assert not internal_fields.intersection(public_controls)
        assert set(public_controls) <= {
            "strength",
            "editable_mask_uri",
            "protected_mask_uri",
        }
        assert "configuration.weathering" not in case["control_prompt"]
        assert prompt_requests_weathering(case["control_prompt"]) is False
    assert benchmark["fail_closed"]["non_ovrtx_render_is_final_evidence"] is False


def test_lightwheel_assets_and_signoff_remain_explicitly_pending() -> None:
    benchmark = _load(FIXTURES / "weathering_quality" / "contract.json")
    partner = benchmark["partner_validation"]

    assert partner["owner"] == "Lightwheel"
    assert partner["status"] == "pending"
    assert "partner-provided representative assets" in partner["required_later"]
    assert "partner sign-off on production suitability" in partner["required_later"]
