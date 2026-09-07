# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from content_agent_workflows.geometry.audit import audit_geometry_asset


def _source_bundle(*, assertion_status: str = "passed") -> dict[str, object]:
    return {
        "schema_version": "geometry.source.v1",
        "bundle_id": "b" * 64,
        "producer": {
            "provider_id": "fixture-provider",
            "provider_version": "1.0",
        },
        "source_revision": "revision-1",
        "selected_representation": {
            "representation_id": "render-usd",
            "role": "render_geometry",
            "format": "usda",
        },
        "parts": [
            {
                "part_id": "body",
                "name": "Body",
                "representation_ids": ["render-usd"],
            }
        ],
        "parameters": [
            {
                "name": "width",
                "value": 20.0,
                "unit": "mm",
                "minimum": 10.0,
                "maximum": 30.0,
            }
        ],
        "verification_assertions": [
            {
                "assertion_id": "provider-topology",
                "status": assertion_status,
                "summary": "Provider topology check.",
                "metrics": {},
            }
        ],
        "manifest_path": "/workflow/geometry.source.json",
    }


def test_audit_projects_only_provider_neutral_source_metadata() -> None:
    report = audit_geometry_asset(
        source_path=Path("/workflow/render.usda"),
        context={
            "prepared_source_metadata": {
                "source_bundle": _source_bundle(),
            }
        },
    )

    contract = report.authoring_contract
    assert contract.source_bundle_id == "b" * 64
    assert contract.source_provider_id == "fixture-provider"
    assert contract.source_revision == "revision-1"
    assert contract.source_representation_id == "render-usd"
    assert contract.semantic_part_count == 1
    assert contract.parameter_count == 1
    assert contract.verification_assertion_count == 1


def test_failed_provider_assertion_blocks_without_optional_quality_checks() -> None:
    report = audit_geometry_asset(
        context={
            "prepared_source_metadata": {
                "source_bundle": _source_bundle(assertion_status="failed"),
            }
        }
    )

    assert not report.passed
    assert [signal.code for signal in report.signals] == [
        "authoring.provider_assertion_failed"
    ]
    assert report.signals[0].blocking


def test_provider_warning_is_visible_but_nonblocking() -> None:
    report = audit_geometry_asset(
        context={
            "prepared_source_metadata": {
                "source_bundle": _source_bundle(assertion_status="warning"),
            }
        }
    )

    assert report.passed
    assert [signal.code for signal in report.signals] == [
        "authoring.provider_assertion_warning"
    ]
    assert not report.signals[0].blocking
