# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import re
from pathlib import Path
from types import ModuleType

import yaml


def _load_openapi() -> dict:
    openapi_path = Path(__file__).parents[2] / "openapi.yaml"
    with open(openapi_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_response_models() -> ModuleType:
    models_path = Path(__file__).parents[2] / "service" / "models" / "responses.py"
    spec = importlib.util.spec_from_file_location(
        "material_agent_service_response_models", models_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _documented_session_created_fields() -> tuple[set[str], str]:
    docs_path = Path(__file__).parents[2] / "docs" / "api.md"
    docs = docs_path.read_text(encoding="utf-8")
    section = docs.split("### `SessionCreated`", maxsplit=1)[1].split(
        "\n### ", maxsplit=1
    )[0]
    fields = set(re.findall(r"^\| `([^`]+)` \|", section, flags=re.MULTILINE))
    return fields, section


def test_session_created_documentation_matches_runtime_contract() -> None:
    spec = _load_openapi()
    openapi_fields = set(spec["components"]["schemas"]["SessionCreated"]["properties"])
    model_fields = set(_load_response_models().SessionCreated.model_fields)
    documented_fields, section = _documented_session_created_fields()
    normalized_section = " ".join(section.split())

    assert documented_fields == model_fields == openapi_fields
    assert (
        "Initial status is `pending` for `POST /pipeline` and "
        "`POST /pipeline/{session_id}/regenerate`"
    ) in normalized_section
    assert (
        "and `ready` for `POST /pipeline/upload-usd` and `POST /pipeline/open-usd`"
    ) in normalized_section
    assert "`queued`" not in normalized_section
    assert "| `created_at` |" not in section


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


def test_static_openapi_documents_s3_pipeline_intake() -> None:
    spec = _load_openapi()

    pipeline = spec["paths"]["/pipeline"]["post"]
    schema = pipeline["requestBody"]["content"]["multipart/form-data"]["schema"]
    properties = schema["properties"]

    assert "s3_uri" in properties
    assert "MA_S3_ALLOWED_BUCKETS" in properties["s3_uri"]["description"]
    assert schema["anyOf"] == [
        {"required": ["session_id"]},
        {"required": ["s3_uri"]},
        {"required": ["usd_file"]},
    ]
    assert {"400", "403", "404", "413", "502"}.issubset(pipeline["responses"])


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


def test_static_openapi_documents_failure_diagnostics_and_evidence() -> None:
    spec = _load_openapi()
    schemas = spec["components"]["schemas"]

    status = schemas["PipelineStatus"]["properties"]
    assert status["error_diagnostic"]["oneOf"][0] == {
        "$ref": "#/components/schemas/PipelineErrorDiagnostic"
    }
    assert status["failure_evidence"]["oneOf"][0] == {
        "$ref": "#/components/schemas/FailureEvidence"
    }
    diagnostic = schemas["PipelineErrorDiagnostic"]["properties"]
    assert diagnostic["renderer_backend"]["enum"] == [
        "warp",
        "ovrtx",
        "remote",
        "mock",
        "unknown",
    ]
    assert diagnostic["samples"]["maxItems"] == 4
    assert schemas["FailureEvidence"]["properties"]["retention"]["const"] == (
        "until_session_expiry_or_deletion"
    )
    assert spec["paths"]["/assets/{session_id}/failure-evidence"]["get"]["responses"][
        "200"
    ]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/FailureEvidence"
    }
