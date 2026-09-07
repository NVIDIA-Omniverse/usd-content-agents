# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from content_agent_workflows.articulation import (
    ARTICULATION_AUTHOR_LEAF_ID,
    ARTICULATION_EVIDENCE_LEAF_ID,
    ARTICULATION_PREPARATION_LEAF_ID,
    ARTICULATION_PROPOSAL_LEAF_ID,
    ARTICULATION_PUBLISH_LEAF_ID,
    ARTICULATION_REVIEW_LEAF_ID,
    ArticulationFocusedLeafInvocation,
    ArticulationPreparationLeafInvocation,
    ArticulationProposalLeafInvocation,
    articulation_asset_leaf_catalog,
    articulation_asset_leaf_descriptors,
    articulation_asset_leaf_runtime_bindings,
    articulation_asset_leaf_runtime_bundle,
)
from content_agent_workflows.asset_composition import (
    AssetExecutionGraph,
    AssetExecutionNode,
    AssetLeafCatalog,
    compose_asset_leaf_runtime_bundles,
)
from content_agent_workflows.asset_composition.catalog_adapters import (
    CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
    shared_asset_leaf_runtime_bundle,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.texture.asset_leaf_adapter import (
    texture_asset_leaf_runtime_bundle,
)

_DIGEST = "1" * 64


def _binding(path: Path) -> ExecutionArtifactBinding:
    return ExecutionArtifactBinding(path=str(path), sha256=_DIGEST, size_bytes=1)


def test_domain_catalog_binds_frozen_selected_leaf_contract() -> None:
    descriptors = {item.leaf_id: item for item in articulation_asset_leaf_descriptors()}
    catalog = articulation_asset_leaf_catalog()

    assert AssetLeafCatalog.model_validate_json(catalog.model_dump_json()) == catalog
    assert tuple(
        item.leaf_id
        for item in catalog.descriptors
        if item.leaf_id.startswith("articulation.")
    ) == (
        ARTICULATION_AUTHOR_LEAF_ID,
        ARTICULATION_EVIDENCE_LEAF_ID,
        ARTICULATION_PREPARATION_LEAF_ID,
        ARTICULATION_PROPOSAL_LEAF_ID,
        ARTICULATION_PUBLISH_LEAF_ID,
        ARTICULATION_REVIEW_LEAF_ID,
    )
    assert descriptors[ARTICULATION_PREPARATION_LEAF_ID].required_dependencies == []
    assert descriptors[ARTICULATION_PROPOSAL_LEAF_ID].required_dependencies == [
        ARTICULATION_PREPARATION_LEAF_ID
    ]
    assert descriptors[ARTICULATION_AUTHOR_LEAF_ID].required_dependencies == [
        ARTICULATION_PREPARATION_LEAF_ID
    ]
    assert descriptors[ARTICULATION_AUTHOR_LEAF_ID].required_dependents == [
        CANONICAL_OVRTX_EVIDENCE_LEAF_ID
    ]
    assert descriptors[ARTICULATION_EVIDENCE_LEAF_ID].required_dependencies == [
        ARTICULATION_AUTHOR_LEAF_ID,
        CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
    ]
    assert descriptors[ARTICULATION_REVIEW_LEAF_ID].required_dependencies == [
        ARTICULATION_EVIDENCE_LEAF_ID
    ]
    assert descriptors[ARTICULATION_PUBLISH_LEAF_ID].required_dependencies == [
        ARTICULATION_REVIEW_LEAF_ID
    ]

    composed = compose_asset_leaf_runtime_bundles(
        (shared_asset_leaf_runtime_bundle(), articulation_asset_leaf_runtime_bundle())
    )
    composed_descriptors = {
        leaf_id: binding.descriptor for leaf_id, binding in composed.bindings.items()
    }
    selected = (
        ARTICULATION_PREPARATION_LEAF_ID,
        ARTICULATION_AUTHOR_LEAF_ID,
        CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
        ARTICULATION_EVIDENCE_LEAF_ID,
        ARTICULATION_REVIEW_LEAF_ID,
        ARTICULATION_PUBLISH_LEAF_ID,
    )

    graph = AssetExecutionGraph.create(
        sole_coordinator_identity_digest="3" * 64,
        prompt_digest="4" * 64,
        source_digest="5" * 64,
        configuration_digest="6" * 64,
        reference_digest="7" * 64,
        leaf_catalog_digest=composed.catalog.catalog_digest,
        nodes=[
            AssetExecutionNode(
                leaf_id=leaf_id,
                depends_on=(
                    []
                    if leaf_id == ARTICULATION_PREPARATION_LEAF_ID
                    else composed_descriptors[leaf_id].required_dependencies
                    + (
                        [ARTICULATION_AUTHOR_LEAF_ID]
                        if leaf_id == CANONICAL_OVRTX_EVIDENCE_LEAF_ID
                        else []
                    )
                ),
                requirement="optional",
                descriptor_digest=composed_descriptors[leaf_id].descriptor_digest,
                terminal_output=leaf_id == ARTICULATION_PUBLISH_LEAF_ID,
            )
            for leaf_id in selected
        ],
        omitted_leaf_ids=sorted(set(composed.bindings).difference(selected)),
    )
    assert set(graph.selected_leaf_ids) == set(selected)
    assert ARTICULATION_PROPOSAL_LEAF_ID in graph.omitted_leaf_ids


def test_catalog_construction_is_deterministic_and_does_not_select() -> None:
    first = articulation_asset_leaf_catalog()
    second = articulation_asset_leaf_catalog()

    assert first == second
    assert first.catalog_digest == second.catalog_digest
    assert not hasattr(first, "selected_leaf_ids")


def test_catalog_rejects_schema_drift_without_versioned_leaf_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import asset_leaf_adapter

    bindings = articulation_asset_leaf_runtime_bindings()
    monkeypatch.setattr(
        asset_leaf_adapter.ArticulationPreparationLeafInvocation,
        "model_json_schema",
        lambda **_kwargs: {"unexpected": "schema drift"},
    )
    preparation = next(
        item
        for item in bindings
        if item.descriptor.leaf_id == ARTICULATION_PREPARATION_LEAF_ID
    )
    with pytest.raises(ValueError, match="invocation_schema_digest drifted"):
        preparation.validate_identity()


def test_joint_distribution_registers_installed_articulation_bundle() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    metadata = tomllib.loads(
        (repository_root / "apps" / "joint_agent" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert metadata["project"]["entry-points"][
        "content_agent_workflows.asset_leaf_runtime_bundles"
    ] == {
        "articulation": (
            "content_agent_workflows.articulation.asset_leaf_adapter:"
            "articulation_asset_leaf_runtime_bundle"
        )
    }


def test_released_bundle_set_composes_joint_texture_and_validation() -> None:
    composed = compose_asset_leaf_runtime_bundles(
        (
            shared_asset_leaf_runtime_bundle(),
            texture_asset_leaf_runtime_bundle(),
            articulation_asset_leaf_runtime_bundle(),
        )
    )

    assert len(composed.bindings) == 15
    assert {
        leaf_id for leaf_id in composed.bindings if leaf_id.startswith("articulation.")
    } == {
        ARTICULATION_PREPARATION_LEAF_ID,
        ARTICULATION_PROPOSAL_LEAF_ID,
        ARTICULATION_AUTHOR_LEAF_ID,
        ARTICULATION_EVIDENCE_LEAF_ID,
        ARTICULATION_REVIEW_LEAF_ID,
        ARTICULATION_PUBLISH_LEAF_ID,
    }


def test_focused_invocation_accepts_only_outer_owned_author_inputs(
    tmp_path: Path,
) -> None:
    required = {
        "leaf_id": ARTICULATION_AUTHOR_LEAF_ID,
        "attempt_root": str(tmp_path),
        "source": _binding(tmp_path / "source.usda"),
        "preparation_publication": _binding(tmp_path / "preparation.json"),
        "decision_patch_path": str(tmp_path / "articulation_decision_patch.json"),
        "intent": "author only the accepted cabinet drawer graph",
    }
    invocation = ArticulationFocusedLeafInvocation(**required)

    assert invocation.reasoning_loop_owner == "asset_coordinator"
    assert invocation.provider_invoked is False
    assert invocation.nested_coordinator_invoked is False
    with pytest.raises(ValidationError):
        ArticulationFocusedLeafInvocation(
            **required,
            nested_coordinator_invoked=True,
        )


def test_preparation_invocation_requires_readback_inside_retained_root(
    tmp_path: Path,
) -> None:
    retained = tmp_path / "retained"
    with pytest.raises(ValidationError, match="inside the retained root"):
        ArticulationPreparationLeafInvocation(
            readback=_binding(tmp_path / "other" / "readback.json"),
            retained_root=str(retained),
        )


def test_proposal_invocation_binds_replacement_as_exact_pair(tmp_path: Path) -> None:
    kwargs = {
        "preparation": _binding(tmp_path / "preparation.json"),
        "intent": "inspect the selected mechanism",
        "provider": {
            "adapter": "artifact-json",
            "provider_id": "fixture-provider",
            "capability_id": "joint-proposal-v1",
            "provider_payload": _binding(tmp_path / "provider-payload.json"),
        },
    }
    with pytest.raises(ValidationError, match="provided together"):
        ArticulationProposalLeafInvocation(
            **kwargs,
            replacement_reason="retry after a typed failure",
        )

    invocation = ArticulationProposalLeafInvocation(
        **kwargs,
        replaces_terminal_receipt=_binding(tmp_path / "terminal.json"),
        replacement_reason="retry after a typed failure",
    )
    assert invocation.replacement_reason == "retry after a typed failure"


def test_selected_leaf_invocations_reject_relative_binding_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="canonical and absolute"):
        ArticulationPreparationLeafInvocation(
            readback=ExecutionArtifactBinding(
                path="readback.json",
                sha256=_DIGEST,
                size_bytes=1,
            ),
            retained_root=str(tmp_path / "retained"),
        )


def test_selected_leaf_invocation_rejects_model_authored_http_configuration(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError):
        ArticulationProposalLeafInvocation(
            preparation=_binding(tmp_path / "preparation.json"),
            intent="inspect the selected mechanism",
            provider={
                "adapter": "http-json",
                "provider_id": "fixture-provider",
                "capability_id": "joint-proposal-v1",
                "endpoint_alias": "fixture-endpoint",
                "endpoint_url": "https://proposal.example.test/v1",
                "bearer_token_env": "AWS_SECRET_ACCESS_KEY",
            },
        )


def test_selected_leaf_invocations_reject_model_authored_output_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="output_dir"):
        ArticulationProposalLeafInvocation.model_validate(
            {
                "preparation": _binding(tmp_path / "preparation.json"),
                "output_dir": str(tmp_path / "arbitrary-output"),
                "intent": "Do not accept a model-authored filesystem destination.",
                "provider": {
                    "adapter": "artifact-json",
                    "provider_id": "fixture-provider",
                    "capability_id": "joint-proposal-v1",
                    "provider_payload": _binding(tmp_path / "provider-payload.json"),
                },
            },
        )
