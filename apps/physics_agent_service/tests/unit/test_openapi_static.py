# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import yaml
from world_understanding.functions.graphics.rendering_backend_factory import (
    RENDERING_BACKEND_NAMES,
)


def _load_openapi() -> dict:
    openapi_path = Path(__file__).parents[2] / "openapi.yaml"
    with open(openapi_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _multipart_schema(spec: dict, *, path: str, method: str) -> dict:
    body_schema = spec["paths"][path][method]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    schema_name = body_schema["$ref"].removeprefix("#/components/schemas/")
    return spec["components"]["schemas"][schema_name]


def _required_variants(schema: dict) -> set[tuple[str, ...]]:
    return {tuple(item["required"]) for item in schema["oneOf"]}


def _normalize_description(description: str) -> str:
    """Normalize whitespace and equivalent Markdown code-span syntax."""
    return " ".join(description.replace("``", "`").split())


def test_render_backend_enums_match_live_contract() -> None:
    from ...service.main import app

    expected = [*RENDERING_BACKEND_NAMES, ""]
    static_openapi = _load_openapi()
    live_openapi = app.openapi()

    for path in ("/pipeline", "/predict"):
        static_schema = _multipart_schema(static_openapi, path=path, method="post")
        live_schema = _multipart_schema(live_openapi, path=path, method="post")

        assert static_schema["properties"]["render_backend"]["enum"] == expected
        assert live_schema["properties"]["render_backend"]["enum"] == expected


def test_static_openapi_documents_client_s3_authorization() -> None:
    spec = _load_openapi()

    default_description = (
        "Client S3 URI rejected by the configured bucket allowlist, "
        "or S3 access denied."
    )
    expected_descriptions = {
        "/pipeline/upload-usd": default_description,
        "/pipeline": default_description,
        "/predict": (
            "dataset_path resolves outside allowed roots; client S3 URI rejected "
            "by the configured bucket allowlist; or S3 access denied."
        ),
        "/tune": default_description,
        "/refine": default_description,
    }
    for path, expected_description in expected_descriptions.items():
        response = spec["paths"][path]["post"]["responses"]["403"]
        assert response["description"] == expected_description
        assert response["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ErrorDetail"
        }

    assert spec["components"]["schemas"]["ErrorDetail"] == {
        "properties": {
            "detail": {
                "type": "string",
                "title": "Detail",
                "description": "Human-readable error detail",
            }
        },
        "type": "object",
        "required": ["detail"],
        "title": "ErrorDetail",
        "description": (
            "Standard FastAPI error payload returned by authorization failures."
        ),
    }


def test_live_openapi_reuses_error_detail_for_client_s3_operations(app) -> None:
    original_schema = app.openapi_schema
    app.openapi_schema = None
    try:
        spec = app.openapi()
    finally:
        app.openapi_schema = original_schema

    default_description = (
        "Client S3 URI rejected by the configured bucket allowlist, "
        "or S3 access denied."
    )
    expected_descriptions = {
        "/pipeline/upload-usd": default_description,
        "/pipeline": default_description,
        "/predict": (
            "dataset_path resolves outside allowed roots; client S3 URI rejected "
            "by the configured bucket allowlist; or S3 access denied."
        ),
        "/tune": default_description,
        "/refine": default_description,
    }
    for path, expected_description in expected_descriptions.items():
        response = spec["paths"][path]["post"]["responses"]["403"]
        assert response["description"] == expected_description
        assert response["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ErrorDetail"
        }

    error_schema = spec["components"]["schemas"]["ErrorDetail"]
    assert error_schema["required"] == ["detail"]
    assert error_schema["properties"]["detail"]["type"] == "string"


def test_static_and_live_openapi_share_pipeline_source_precedence(app) -> None:
    static_spec = _load_openapi()
    original_schema = app.openapi_schema
    app.openapi_schema = None
    try:
        live_spec = app.openapi()
    finally:
        app.openapi_schema = original_schema

    static_operation = static_spec["paths"]["/pipeline"]["post"]
    live_operation = live_spec["paths"]["/pipeline"]["post"]
    static_description = _normalize_description(static_operation["description"])
    live_description = _normalize_description(live_operation["description"])
    assert static_description == live_description

    static_schema = _multipart_schema(static_spec, path="/pipeline", method="post")
    live_schema = _multipart_schema(live_spec, path="/pipeline", method="post")
    assert "oneOf" not in static_schema
    assert "oneOf" not in live_schema

    for field_name in ("session_id", "s3_uri", "usd_file"):
        static_field = static_schema["properties"][field_name]
        live_field = live_schema["properties"][field_name]
        assert _normalize_description(
            static_field["description"]
        ) == _normalize_description(live_field["description"])

    assert "At least one input source is required" in static_description
    assert "`session_id` first" in static_description
    assert "then `s3_uri`" in static_description
    assert "then `usd_file`" in static_description
    assert "session_id, then s3_uri, then usd_file" in static_schema["description"]


def test_static_openapi_documents_tune_route_family() -> None:
    spec = _load_openapi()

    expected_routes = {
        "/tune": "post",
        "/tune/{session_id}/status": "get",
        "/tune/{session_id}/results": "get",
        "/tune/{session_id}/events": "get",
        "/tune/{session_id}/cancel": "post",
        "/tune/{session_id}/artifacts/{name}": "get",
    }
    for path, method in expected_routes.items():
        assert path in spec["paths"]
        assert method in spec["paths"][path]
        assert spec["paths"][path][method]["tags"] == ["tune"]

    create_tune = spec["paths"]["/tune"]["post"]
    assert create_tune["responses"]["202"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SessionCreated"
    }

    status = spec["paths"]["/tune/{session_id}/status"]["get"]
    assert status["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/TuneStatus"
    }

    results = spec["paths"]["/tune/{session_id}/results"]["get"]
    result_refs = results["responses"]["200"]["content"]["application/json"]["schema"][
        "anyOf"
    ]
    assert {"$ref": "#/components/schemas/TuneResults"} in result_refs
    assert "202" in results["responses"]

    schema = _multipart_schema(spec, path="/tune", method="post")
    assert _required_variants(schema) == {
        ("physics_usd",),
        ("s3_uri",),
        ("source_session_id",),
    }
    properties = schema["properties"]
    for field in (
        "scenario_yaml",
        "user_prompt",
        "optimizer",
        "engine",
        "max_trials",
        "seed",
        "enable_judge",
        "judge_max_iterations",
        "reference_video_frames",
        "judge_reference_frames",
        "judge_generated_frames",
    ):
        assert field in properties
    assert properties["optimizer"]["default"] == "auto"
    assert "botorch" in properties["optimizer"]["description"]
    assert properties["seed"]["default"] == 42
    assert properties["enable_judge"]["default"] is True
    assert properties["reference_video_frames"]["default"] == 8
    assert properties["judge_reference_frames"]["default"] == 8
    assert properties["judge_generated_frames"]["default"] == 16


def test_static_openapi_documents_refine_route_family() -> None:
    spec = _load_openapi()

    expected_routes = {
        "/refine": "post",
        "/refine/{session_id}/status": "get",
        "/refine/{session_id}/results": "get",
        "/refine/{session_id}/events": "get",
        "/refine/{session_id}/cancel": "post",
        "/refine/{session_id}/artifacts/{name}": "get",
    }
    for path, method in expected_routes.items():
        assert path in spec["paths"]
        assert method in spec["paths"][path]
        assert spec["paths"][path][method]["tags"] == ["refine"]

    create_refine = spec["paths"]["/refine"]["post"]
    assert create_refine["responses"]["202"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/SessionCreated"}

    status = spec["paths"]["/refine/{session_id}/status"]["get"]
    assert status["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RefineStatus"
    }

    results = spec["paths"]["/refine/{session_id}/results"]["get"]
    result_refs = results["responses"]["200"]["content"]["application/json"]["schema"][
        "anyOf"
    ]
    assert {"$ref": "#/components/schemas/RefineResults"} in result_refs
    assert "202" in results["responses"]

    schema = _multipart_schema(spec, path="/refine", method="post")
    assert set(schema["required"]) == {"scenario_yaml", "user_prompt"}
    assert _required_variants(schema) == {
        ("physics_usd",),
        ("s3_uri",),
        ("source_session_id",),
    }
    properties = schema["properties"]
    for field in (
        "scenario_yaml",
        "user_prompt",
        "optimizer",
        "engine",
        "max_trials",
        "max_iterations",
        "score_threshold",
        "seed",
        "reference_video_frames",
        "judge_reference_frames",
        "judge_generated_frames",
        "visual_evidence_enabled",
        "llm_timeout_seconds",
    ):
        assert field in properties
    assert properties["optimizer"]["default"] == "botorch"
    assert properties["max_iterations"]["maximum"] == 12
    assert properties["score_threshold"]["default"] == 0.9
    assert properties["seed"]["default"] == 42
    assert properties["reference_video_frames"]["default"] == 8
    assert properties["judge_reference_frames"]["default"] == 8
    assert properties["judge_generated_frames"]["default"] == 16
    assert properties["visual_evidence_enabled"]["default"] is True
