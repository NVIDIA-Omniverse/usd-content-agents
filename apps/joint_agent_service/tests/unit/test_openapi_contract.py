# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static contract checks for the checked-in Joint Agent OpenAPI document."""

from __future__ import annotations

from pathlib import Path

import yaml
from world_understanding.rendering_backend_contract import RENDERING_BACKEND_NAMES

REPO_ROOT = Path(__file__).resolve().parents[4]


def _multipart_properties(openapi: dict, path: str) -> dict:
    schema = openapi["paths"][path]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    if "$ref" in schema:
        schema = openapi["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
    return schema["properties"]


def test_live_openapi_types_cancellation_response() -> None:
    from ...service.main import app

    live_openapi = app.openapi()
    pipeline_body = live_openapi["paths"]["/pipeline"]["post"]["requestBody"][
        "content"
    ]["multipart/form-data"]["schema"]
    pipeline_schema_name = pipeline_body["$ref"].removeprefix("#/components/schemas/")
    render_backend_schema = live_openapi["components"]["schemas"][pipeline_schema_name][
        "properties"
    ]["render_backend"]
    cancel_operation = live_openapi["paths"]["/pipeline/{session_id}/cancel"]["post"]

    assert render_backend_schema["enum"] == [*RENDERING_BACKEND_NAMES, ""]
    assert (
        cancel_operation["responses"]["200"]["content"]["application/json"]["schema"][
            "$ref"
        ]
        == "#/components/schemas/PipelineCancellationAccepted"
    )
    assert (
        cancel_operation["responses"]["422"]["content"]["application/json"]["schema"][
            "$ref"
        ]
        == "#/components/schemas/HTTPValidationError"
    )
    for path in ("/pipeline/upload-usd", "/pipeline"):
        responses = live_openapi["paths"][path]["post"]["responses"]
        for status in ("400", "403"):
            assert responses[status]["content"]["application/json"]["schema"] == {
                "$ref": "#/components/schemas/ErrorDetail"
            }
        assert responses["400"]["description"] == (
            "Missing, conflicting, or invalid input source, including an "
            "unsupported USD extension."
        )

    assert live_openapi["components"]["schemas"]["ErrorDetail"] == {
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
        "description": "Standard FastAPI error payload returned by input failures.",
    }


def test_live_openapi_documents_source_precedence_and_fail_closed_s3() -> None:
    from ...service.main import app

    live_openapi = app.openapi()
    upload_operation = live_openapi["paths"]["/pipeline/upload-usd"]["post"]
    pipeline_operation = live_openapi["paths"]["/pipeline"]["post"]
    upload_properties = _multipart_properties(live_openapi, "/pipeline/upload-usd")
    pipeline_properties = _multipart_properties(live_openapi, "/pipeline")

    assert "exactly one" in upload_operation["description"]
    assert "exact bucket name" in upload_operation["description"]
    assert "empty allowlist" in upload_operation["description"]
    assert "exactly one" in upload_properties["usd_file"]["description"]
    assert "JA_S3_ALLOWED_BUCKETS" in upload_properties["s3_uri"]["description"]
    assert (
        "Missing, conflicting, or invalid input source"
        in upload_operation["responses"]["400"]["description"]
    )

    assert (
        "``session_id`` > ``s3_uri`` > ``usd_file``"
        in pipeline_operation["description"]
    )
    assert "lower-priority fields are" in pipeline_operation["description"]
    assert "empty allowlist" in pipeline_operation["description"]
    assert "Highest-priority" in pipeline_properties["session_id"]["description"]
    assert "Second-priority" in pipeline_properties["s3_uri"]["description"]
    assert "Lowest-priority" in pipeline_properties["usd_file"]["description"]
    assert (
        "before session-manager or S3 I/O"
        in pipeline_properties["s3_uri"]["description"]
    )


def test_openapi_documents_joint_rigger_service_surface() -> None:
    openapi = yaml.safe_load(
        (REPO_ROOT / "apps/joint_agent_service/openapi.yaml").read_text(
            encoding="utf-8"
        )
    )

    repository_version = (REPO_ROOT / "VERSION.md").read_text(encoding="utf-8").strip()
    assert openapi["info"]["version"] == repository_version == "0.5.0"

    form_properties = openapi["paths"]["/pipeline"]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]["properties"]
    assert "s3_uri" in form_properties
    upload_properties = openapi["paths"]["/pipeline/upload-usd"]["post"]["requestBody"][
        "content"
    ]["multipart/form-data"]["schema"]["properties"]
    assert "s3_uri" in upload_properties
    for path in ("/pipeline/upload-usd", "/pipeline"):
        responses = openapi["paths"][path]["post"]["responses"]
        for status in ("400", "403"):
            assert responses[status]["content"]["application/json"]["schema"] == {
                "$ref": "#/components/schemas/ErrorDetail"
            }
        assert responses["400"]["description"] == (
            "Missing, conflicting, or invalid input source, including an "
            "unsupported USD extension."
        )

    assert openapi["components"]["schemas"]["ErrorDetail"] == {
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
        "description": "Standard FastAPI error payload returned by input failures.",
    }

    upload_operation = openapi["paths"]["/pipeline/upload-usd"]["post"]
    pipeline_operation = openapi["paths"]["/pipeline"]["post"]
    assert "exactly one source" in upload_operation["description"]
    assert "exact bucket name" in upload_operation["description"]
    assert "empty allowlist" in upload_operation["description"]
    assert (
        "session_id first, otherwise s3_uri, otherwise usd_file"
        in (pipeline_operation["description"])
    )
    assert "lower-priority fields are ignored" in pipeline_operation["description"]
    assert "empty allowlist" in pipeline_operation["description"]
    assert "Highest-priority" in form_properties["session_id"]["description"]
    assert "Second-priority" in form_properties["s3_uri"]["description"]
    assert "Lowest-priority" in form_properties["usd_file"]["description"]
    assert (
        "before session-manager or S3 I/O" in form_properties["s3_uri"]["description"]
    )
    assert "exactly one" in upload_properties["usd_file"]["description"]
    assert "JA_S3_ALLOWED_BUCKETS" in upload_properties["s3_uri"]["description"]
    assert (
        "Missing, conflicting, or invalid input source"
        in upload_operation["responses"]["400"]["description"]
    )
    for field_name in (
        "apply_joint_rigger",
        "joint_rigger_adapter",
        "joint_rigger_on_missing_dependency",
        "joint_rigger_on_unready_candidates",
        "joint_rigger_template",
        "joint_rigger_apply_masses",
        "joint_rigger_apply_collision",
    ):
        assert field_name in form_properties

    adapter_schema = form_properties["joint_rigger_adapter"]
    assert adapter_schema["enum"] == [
        "owned_core",
        "mock",
        "usd_joint_rigger",
        "",
    ]
    assert "Omission selects" in adapter_schema["description"]
    assert (
        "requires false" in form_properties["joint_rigger_apply_masses"]["description"]
    )
    assert (
        "requires false"
        in form_properties["joint_rigger_apply_collision"]["description"]
    )
    render_backend_schema = form_properties["render_backend"]
    assert render_backend_schema["default"] == ""
    assert render_backend_schema["enum"] == [*RENDERING_BACKEND_NAMES, ""]
    assert (
        "Omission uses the configured service default"
        in render_backend_schema["description"]
    )
    assert "warp' (default" not in render_backend_schema["description"]

    create_response = openapi["paths"]["/pipeline"]["post"]["responses"]["202"]
    assert create_response["content"]["application/json"]["schema"]["$ref"] == (
        "#/components/schemas/PipelineRunCreated"
    )
    cancel_operation = openapi["paths"]["/pipeline/{session_id}/cancel"]["post"]
    run_id_parameter = next(
        parameter
        for parameter in cancel_operation["parameters"]
        if parameter.get("name") == "run_id"
    )
    assert run_id_parameter["required"] is True
    assert run_id_parameter["schema"]["pattern"] == "^[0-9a-f]{32}$"
    assert (
        cancel_operation["responses"]["200"]["content"]["application/json"]["schema"][
            "$ref"
        ]
        == "#/components/schemas/PipelineCancellationAccepted"
    )
    assert "409" in cancel_operation["responses"]
    assert (
        cancel_operation["responses"]["422"]["content"]["application/json"]["schema"][
            "$ref"
        ]
        == "#/components/schemas/HTTPValidationError"
    )
    regenerate_response = openapi["paths"]["/pipeline/{session_id}/regenerate"]["post"][
        "responses"
    ]["202"]
    assert (
        regenerate_response["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/PipelineRunCreated"
    )
    assert (
        "409"
        in openapi["paths"]["/pipeline/{session_id}/regenerate"]["post"]["responses"]
    )

    assert (
        openapi["components"]["schemas"]["HealthStatus"]["properties"]["capabilities"][
            "properties"
        ]["joint_rigger"]["$ref"]
        == "#/components/schemas/JointRiggerCapabilities"
    )
    capabilities = openapi["components"]["schemas"]["JointRiggerCapabilities"]
    assert capabilities["required"] == [
        "owned_core_available",
        "usd_joint_rigger_available",
    ]
    assert "owned_core_import_error_type" in capabilities["properties"]
    assert (
        "external"
        in capabilities["properties"]["usd_joint_rigger_available"]["description"]
    )

    output_response = openapi["paths"]["/artifacts/{session_id}/joint-rigger-output"][
        "get"
    ]["responses"]["200"]
    assert (
        output_response["headers"]["Content-Disposition"]["schema"]["example"]
        == 'attachment; filename="rigged.usdz"'
    )

    for artifact_path in (
        "/artifacts/{session_id}/joint-rigger-output",
        "/artifacts/{session_id}/joint-rigger-diagnostics",
        "/artifacts/{session_id}/joint-rigger-validation",
    ):
        assert artifact_path in openapi["paths"]

    result_schema = openapi["components"]["schemas"]["PipelineResults"]
    assert (
        "joint_rigger_authored_joints"
        in result_schema["properties"]["stats"]["properties"]
    )
    assert (
        "joint_rigger_artifact_keys"
        in result_schema["properties"]["stats"]["properties"]
    )
    assert (
        "joint_rigger_publication_id"
        in result_schema["properties"]["stats"]["properties"]
    )
    assert (
        "joint_rigger_output"
        in result_schema["properties"]["download_urls"]["properties"]
    )


def test_service_docs_match_health_and_renderer_contracts() -> None:
    api_docs = (REPO_ROOT / "apps/joint_agent_service/docs/api.md").read_text(
        encoding="utf-8"
    )
    health_section = api_docs.split("### `GET /health`", 1)[1].split("---", 1)[0]
    assert '"owned_core_available": true' in health_section
    assert '"owned_core_available": false' not in health_section
    assert '"owned_core_import_error_type": "ModuleNotFoundError"' not in (
        health_section
    )
    assert "present only when the built-in core cannot be" in health_section
    assert "Omission uses the configured service default" in api_docs
    assert "| `JA_RENDER_BACKEND` | `remote` |" in api_docs
    assert "Remote rendering resolves through `RENDER_ENDPOINT`" in api_docs

    service_readme = (
        REPO_ROOT / "apps/joint_agent_service/service/README.md"
    ).read_text(encoding="utf-8")
    assert "* `warp`" in service_readme
    assert "* `ovrtx`" in service_readme
    assert "* `remote`" in service_readme
    assert "`nvcf`" not in service_readme
    assert "Omission uses the configured service default" in service_readme
