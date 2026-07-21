# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAPI contract checks for client-controlled Texture Agent S3 inputs."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]


def _multipart_properties(openapi: dict, path: str) -> dict:
    schema = openapi["paths"][path]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    if "$ref" in schema:
        schema = openapi["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
    return schema["properties"]


def _assert_s3_source_contract(openapi: dict) -> None:
    upload_operation = openapi["paths"]["/pipeline/upload-usd"]["post"]
    pipeline_operation = openapi["paths"]["/pipeline"]["post"]
    upload_properties = _multipart_properties(openapi, "/pipeline/upload-usd")
    pipeline_properties = _multipart_properties(openapi, "/pipeline")

    assert "exactly one" in upload_operation["description"]
    assert "exact bucket name" in upload_operation["description"]
    assert "empty allowlist" in upload_operation["description"]
    assert "exactly one" in upload_properties["usd_file"]["description"]
    assert "TA_S3_ALLOWED_BUCKETS" in upload_properties["s3_uri"]["description"]
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
    for operation in (upload_operation, pipeline_operation):
        bad_request_response = operation["responses"]["400"]
        forbidden_response = operation["responses"]["403"]
        expected_schema = {"$ref": "#/components/schemas/ErrorDetail"}
        assert bad_request_response["content"]["application/json"]["schema"] == (
            expected_schema
        )
        assert forbidden_response["content"]["application/json"]["schema"] == (
            expected_schema
        )
        assert "configured bucket allowlist" in forbidden_response["description"]
        assert "S3 access denied" in forbidden_response["description"]

    error_schema = openapi["components"]["schemas"]["ErrorDetail"]
    assert error_schema["required"] == ["detail"]
    assert error_schema["properties"]["detail"]["type"] == "string"


def test_live_openapi_documents_source_precedence_and_fail_closed_s3() -> None:
    from ...service.main import app

    _assert_s3_source_contract(app.openapi())


def test_static_openapi_documents_source_precedence_and_fail_closed_s3() -> None:
    openapi = yaml.safe_load(
        (REPO_ROOT / "apps/texture_agent_service/openapi.yaml").read_text(
            encoding="utf-8"
        )
    )

    _assert_s3_source_contract(openapi)
