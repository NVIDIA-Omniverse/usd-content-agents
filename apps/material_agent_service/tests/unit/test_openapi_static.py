# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import yaml


def _load_openapi() -> dict:
    openapi_path = Path(__file__).parents[2] / "openapi.yaml"
    with open(openapi_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_static_openapi_documents_regenerate_and_event_log() -> None:
    spec = _load_openapi()

    regenerate = spec["paths"]["/pipeline/{session_id}/regenerate"]["post"]
    assert regenerate["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RegenerateRequest"
    }
    assert regenerate["responses"]["202"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SessionCreated"
    }

    event_log = spec["paths"]["/pipeline/{session_id}/event-log"]["get"]
    assert (
        event_log["responses"]["200"]["content"]["application/json"]["schema"][
            "properties"
        ]["events"]["type"]
        == "array"
    )

    regenerate_schema = spec["components"]["schemas"]["RegenerateRequest"]
    assert regenerate_schema["required"] == ["steps"]
    assert regenerate_schema["properties"]["steps"]["items"] == {
        "$ref": "#/components/schemas/PipelineStep"
    }

    step_values = spec["components"]["schemas"]["PipelineStep"]["enum"]
    assert {"predict", "apply", "render"}.issubset(step_values)


def test_static_openapi_documents_material_generation_fields() -> None:
    spec = _load_openapi()

    pipeline = spec["paths"]["/pipeline"]["post"]
    properties = pipeline["requestBody"]["content"]["multipart/form-data"]["schema"][
        "properties"
    ]

    assert properties["enable_material_generation"]["default"] == "false"
    assert properties["material_generation_texture_size"]["minimum"] == 64
    assert properties["material_generation_texture_size"]["maximum"] == 4096
    assert "material_generation_guidance" in properties
    assert "503" in pipeline["responses"]


def test_static_openapi_documents_initial_layer_only_field() -> None:
    spec = _load_openapi()

    pipeline = spec["paths"]["/pipeline"]["post"]
    properties = pipeline["requestBody"]["content"]["multipart/form-data"]["schema"][
        "properties"
    ]

    assert properties["layer_only"]["type"] == "string"
    assert properties["layer_only"]["default"] == "false"


def test_static_openapi_documents_material_coverage_contract() -> None:
    spec = _load_openapi()

    pipeline = spec["paths"]["/pipeline"]["post"]
    properties = pipeline["requestBody"]["content"]["multipart/form-data"]["schema"][
        "properties"
    ]
    assert properties["coverage_policy"]["enum"] == ["strict", "allow_partial"]

    coverage = spec["components"]["schemas"]["MaterialCoverage"]
    assert coverage["properties"]["readiness_grade"]["enum"] == [
        "complete",
        "complete_with_fallback",
        "partial",
        "not_evaluated",
    ]
    assert "missing_prediction_prim_ids" in coverage["properties"]
    assert "unbound_prim_ids" in coverage["properties"]
    assert set(coverage["required"]) == set(coverage["properties"])

    pipeline_error = spec["components"]["schemas"]["PipelineError"]
    assert pipeline_error["properties"]["completed_steps"]["items"] == {
        "type": "string"
    }
    failure_artifacts = pipeline_error["properties"]["download_urls"]["properties"]
    scene_artifacts = {
        "scene_manifest",
        "scene_validation_report",
        "scene_predictions",
        "final_render",
    }
    assert {"output_usd", "predictions", "report", *scene_artifacts}.issubset(
        failure_artifacts
    )
    result_artifacts = spec["components"]["schemas"]["PipelineResults"]["properties"][
        "download_urls"
    ]["properties"]
    assert scene_artifacts.issubset(result_artifacts)
    partial = pipeline_error["properties"]["partial_results"]
    assert set(partial["properties"]) == {"stats", "coverage"}
