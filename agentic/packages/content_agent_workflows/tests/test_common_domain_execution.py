# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared standalone and embedded domain execution identity."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from content_agent_workflows.common.domain_execution import (
    DOMAIN_EXECUTION_CONTEXT_METADATA_KEY,
    DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION,
    DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2,
    DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3,
    DomainExecutionContext,
    EmbeddedStageBinding,
    ExecutionArtifactBinding,
    domain_execution_context_from_metadata,
    metadata_with_domain_execution_context,
)


def _artifact(path: str, fill: str) -> ExecutionArtifactBinding:
    """Build a binding whose digest repeats one hexadecimal character."""

    return ExecutionArtifactBinding(path=path, sha256=fill * 64, size_bytes=1)


def _embedded_context() -> DomainExecutionContext:
    return DomainExecutionContext(
        domain="texture",
        mode="embedded",
        reasoning_loop_owner="asset_coordinator",
        embedded_stage=EmbeddedStageBinding(
            outer_run_id="asset-run",
            outer_request=_artifact("/run/request.json", "1"),
            stage="texture",
            stage_attempt=2,
            coordinator_plan=_artifact("/run/coordinator/plan-002.json", "2"),
            input_asset=_artifact("/run/material.usdz", "3"),
            domain_run_root="/run/stages/03-texture/domain-run",
        ),
    )


def test_embedded_execution_context_round_trips_through_reserved_metadata() -> None:
    context = _embedded_context()

    metadata = metadata_with_domain_execution_context({"provider": "test"}, context)

    assert metadata["provider"] == "test"
    assert (
        domain_execution_context_from_metadata(metadata, expected_domain="texture")
        == context
    )
    assert context.schema_version == DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION


def test_validation_uses_additive_v2_without_changing_v1_domains() -> None:
    stage = EmbeddedStageBinding(
        outer_run_id="asset-run",
        outer_request=_artifact("/run/request.json", "1"),
        stage="validation",
        stage_attempt=1,
        coordinator_plan=_artifact("/run/coordinator/plan-001.json", "2"),
        input_asset=_artifact("/run/final.usdz", "3"),
        domain_run_root="/run/stages/05-validation/domain-run",
    )
    with pytest.raises(ValidationError, match="requires domain execution context v2"):
        DomainExecutionContext(
            domain="validation",
            mode="embedded",
            reasoning_loop_owner="asset_coordinator",
            embedded_stage=stage,
        )

    texture_v2 = _embedded_context().model_dump(mode="python")
    texture_v2["schema_version"] = DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2
    with pytest.raises(ValidationError, match="requires domain execution context v1"):
        DomainExecutionContext.model_validate(texture_v2)

    context = DomainExecutionContext(
        schema_version=DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2,
        domain="validation",
        mode="embedded",
        reasoning_loop_owner="asset_coordinator",
        embedded_stage=stage,
    )

    metadata = metadata_with_domain_execution_context({}, context)
    assert (
        domain_execution_context_from_metadata(
            metadata,
            expected_domain="validation",
        )
        == context
    )


def test_embedded_execution_context_rejects_wrong_owner_or_stage() -> None:
    with pytest.raises(ValidationError, match="asset_coordinator"):
        DomainExecutionContext(
            domain="texture",
            mode="embedded",
            reasoning_loop_owner="domain_child_agent",
            embedded_stage=_embedded_context().embedded_stage,
        )

    payload = _embedded_context().model_dump(mode="json")
    payload["embedded_stage"]["stage"] = "articulation"
    with pytest.raises(ValidationError, match="stage must match"):
        DomainExecutionContext.model_validate(payload)


def test_standalone_execution_context_rejects_coordinator_owner_or_stage() -> None:
    with pytest.raises(ValidationError, match="cannot claim asset_coordinator"):
        DomainExecutionContext(
            domain="texture",
            mode="standalone",
            reasoning_loop_owner="asset_coordinator",
        )

    with pytest.raises(ValidationError, match="cannot carry an outer stage binding"):
        DomainExecutionContext(
            domain="texture",
            mode="standalone",
            reasoning_loop_owner="domain_child_agent",
            embedded_stage=_embedded_context().embedded_stage,
        )


def test_articulation_v3_binds_standalone_asset_coordinator_ownership() -> None:
    context = DomainExecutionContext(
        schema_version=DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3,
        domain="articulation",
        mode="standalone",
        reasoning_loop_owner="asset_coordinator",
    )

    assert context.embedded_stage is None
    with pytest.raises(ValidationError, match="requires domain execution context v1"):
        DomainExecutionContext(
            schema_version=DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3,
            domain="texture",
            mode="standalone",
            reasoning_loop_owner="asset_coordinator",
        )
    with pytest.raises(ValidationError, match="requires standalone asset_coordinator"):
        DomainExecutionContext(
            schema_version=DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3,
            domain="articulation",
            mode="standalone",
            reasoning_loop_owner="domain_child_agent",
        )
    with pytest.raises(ValidationError, match="requires standalone asset_coordinator"):
        DomainExecutionContext(
            schema_version=DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3,
            domain="articulation",
            mode="embedded",
            reasoning_loop_owner="asset_coordinator",
            embedded_stage=EmbeddedStageBinding(
                outer_run_id="asset-run",
                outer_request=_artifact("/run/request.json", "1"),
                stage="articulation",
                stage_attempt=1,
                coordinator_plan=_artifact("/run/coordinator/plan-001.json", "2"),
                input_asset=_artifact("/run/input.usdz", "3"),
                domain_run_root="/run/stages/04-articulation/domain-run",
            ),
        )


def test_execution_context_metadata_fails_closed_on_conflict_or_wrong_domain() -> None:
    context = _embedded_context()
    metadata = metadata_with_domain_execution_context({}, context)

    with pytest.raises(ValueError, match="does not match"):
        domain_execution_context_from_metadata(
            metadata,
            expected_domain="articulation",
        )

    conflicting = dict(metadata)
    conflicting[DOMAIN_EXECUTION_CONTEXT_METADATA_KEY] = {
        **context.model_dump(mode="json"),
        "reasoning_loop_owner": "domain_child_agent",
    }
    with pytest.raises(ValueError, match="already differs"):
        metadata_with_domain_execution_context(conflicting, context)


def test_legacy_metadata_remains_unmodified_and_unbound() -> None:
    metadata = {"provider": {"backend": "test"}}

    assert (
        domain_execution_context_from_metadata(metadata, expected_domain="texture")
        is None
    )
    assert metadata == {"provider": {"backend": "test"}}


def test_execution_context_metadata_rejects_json_encoded_string() -> None:
    metadata = {
        DOMAIN_EXECUTION_CONTEXT_METADATA_KEY: _embedded_context().model_dump_json()
    }

    with pytest.raises(ValueError, match="must be an object"):
        domain_execution_context_from_metadata(metadata, expected_domain="texture")
