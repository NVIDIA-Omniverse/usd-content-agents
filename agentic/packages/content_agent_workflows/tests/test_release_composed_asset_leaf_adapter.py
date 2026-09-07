# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest

from content_agent_workflows.articulation.asset_leaf_adapter import (
    articulation_asset_leaf_runtime_bundle,
)
from content_agent_workflows.asset_composition.catalog import (
    compose_asset_leaf_runtime_bundles,
)
from content_agent_workflows.asset_composition.catalog_adapters import (
    CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
    shared_asset_leaf_runtime_bundle,
)
from content_agent_workflows.asset_composition.models import (
    AssetExecutionGraph,
    AssetExecutionNode,
)
from content_agent_workflows.asset_composition.release_leaf_adapters import (
    ARTICULATION_PUBLISH_LEAF_ID,
    COMBINED_RESULT_LEAF_ID,
    FINAL_OVRTX_EVIDENCE_LEAF_ID,
    MATERIAL_ASSIGNMENT_LEAF_ID,
    PHYSICS_APPLY_LEAF_ID,
    PHYSICS_INSPECTION_LEAF_ID,
    PORTABLE_PACKAGE_LEAF_ID,
    RELEASE_COMPOSED_LEAF_IDS,
    SIMREADY_CONFORMANCE_LEAF_ID,
    SIMREADY_VALIDATION_LEAF_ID,
    TEXTURE_PUBLISH_LEAF_ID,
    ComposedAssetLeafResult,
    MaterialApplicationReviewReceipt,
    MaterialAssignmentLeafInvocation,
    MaterialCoordinatorTerminalReceipt,
    MaterialSessionReleaseReceipt,
    PhysicsInspectionLeafInvocation,
    artifact_binding,
    release_composed_asset_leaf_runtime_bundle,
    write_canonical_json,
)
from content_agent_workflows.asset_composition.state import _validate_graph_catalog
from content_agent_workflows.texture.asset_leaf_adapter import (
    texture_asset_leaf_runtime_bundle,
)


def _runtime_binding(leaf_id: str):  # type: ignore[no-untyped-def]
    return next(
        binding
        for binding in release_composed_asset_leaf_runtime_bundle().bindings
        if binding.descriptor.leaf_id == leaf_id
    )


def test_artifact_binding_rejects_a_path_swapped_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_bytes(b"original")
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    real_fstat = os.fstat
    fstat_calls = 0
    swapped = False

    def swap_after_validated_read(descriptor: int) -> os.stat_result:
        nonlocal fstat_calls, swapped
        metadata = real_fstat(descriptor)
        fstat_calls += 1
        if fstat_calls == 2 and not swapped:
            swapped = True
            candidate.unlink()
            candidate.symlink_to(outside)
        return metadata

    monkeypatch.setattr(os, "fstat", swap_after_validated_read)

    with pytest.raises(ValueError, match="cannot be opened safely"):
        artifact_binding(candidate)


def test_canonical_writer_does_not_follow_a_late_parent_symlink(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    source = tmp_path / "source.usdz"
    source.write_bytes(b"source")
    destination = attempt / "future" / "component_catalog.json"
    invocation = PhysicsInspectionLeafInvocation(
        attempt_root=str(attempt),
        source=artifact_binding(source),
        component_catalog_path=str(destination),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    destination.parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="destination is unsafe"):
        write_canonical_json(
            invocation.component_catalog_path,
            MaterialSessionReleaseReceipt(
                status="released",
                session_id="test-session",
            ),
        )

    assert not (outside / destination.name).exists()


def test_release_bundle_exposes_complete_domain_and_terminal_coverage() -> None:
    bundle = release_composed_asset_leaf_runtime_bundle()
    descriptors = {
        binding.descriptor.leaf_id: binding.descriptor for binding in bundle.bindings
    }

    assert bundle.bundle_id == "release-composed"
    assert tuple(sorted(descriptors)) == tuple(sorted(RELEASE_COMPOSED_LEAF_IDS))
    assert set(descriptors) == {
        MATERIAL_ASSIGNMENT_LEAF_ID,
        PHYSICS_INSPECTION_LEAF_ID,
        PHYSICS_APPLY_LEAF_ID,
        SIMREADY_CONFORMANCE_LEAF_ID,
        SIMREADY_VALIDATION_LEAF_ID,
        PORTABLE_PACKAGE_LEAF_ID,
        COMBINED_RESULT_LEAF_ID,
        FINAL_OVRTX_EVIDENCE_LEAF_ID,
    }
    assert descriptors[PHYSICS_APPLY_LEAF_ID].required_dependencies == [
        PHYSICS_INSPECTION_LEAF_ID
    ]
    assert descriptors[SIMREADY_VALIDATION_LEAF_ID].required_dependencies == [
        PORTABLE_PACKAGE_LEAF_ID,
        SIMREADY_CONFORMANCE_LEAF_ID,
    ]
    assert descriptors[COMBINED_RESULT_LEAF_ID].required_dependencies == sorted(
        [
            ARTICULATION_PUBLISH_LEAF_ID,
            PORTABLE_PACKAGE_LEAF_ID,
            MATERIAL_ASSIGNMENT_LEAF_ID,
            PHYSICS_APPLY_LEAF_ID,
            PHYSICS_INSPECTION_LEAF_ID,
            SIMREADY_CONFORMANCE_LEAF_ID,
            SIMREADY_VALIDATION_LEAF_ID,
            TEXTURE_PUBLISH_LEAF_ID,
            FINAL_OVRTX_EVIDENCE_LEAF_ID,
        ]
    )
    assert all(
        descriptor.entrypoint.startswith("content-workflow-composed-leaf ")
        for leaf_id, descriptor in descriptors.items()
        if leaf_id != FINAL_OVRTX_EVIDENCE_LEAF_ID
    )
    assert (
        descriptors[FINAL_OVRTX_EVIDENCE_LEAF_ID].entrypoint
        == "content_agent_workflows.validation.produce_canonical_visual_evidence"
    )
    catalog = compose_asset_leaf_runtime_bundles(
        (
            shared_asset_leaf_runtime_bundle(),
            articulation_asset_leaf_runtime_bundle(),
            texture_asset_leaf_runtime_bundle(),
            bundle,
        )
    ).catalog
    assert set(RELEASE_COMPOSED_LEAF_IDS).issubset(
        {descriptor.leaf_id for descriptor in catalog.descriptors}
    )


def test_progress_result_cannot_be_projected_as_terminal(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    invocation_path = attempt / "invocation.json"
    invocation_path.write_text("{}\n", encoding="utf-8")
    invocation_binding = artifact_binding(invocation_path)
    progress_path = attempt / "progress.json"
    progress_path.write_text("{}\n", encoding="utf-8")
    progress_binding = artifact_binding(progress_path)
    progress = ComposedAssetLeafResult(
        leaf_id=MATERIAL_ASSIGNMENT_LEAF_ID,
        status="awaiting_decision",
        native_status="awaiting_decision",
        invocation=invocation_binding,
        native_terminal_receipt=progress_binding,
        evidence=(progress_binding,),
        saved_stage_readbacks=(progress_binding,),
        resource_claims=("usd-cli:fixture",),
        summary="Awaiting the outer Material decision.",
    )
    binding = _runtime_binding(MATERIAL_ASSIGNMENT_LEAF_ID)

    with pytest.raises(ValueError, match="progress cannot be projected"):
        binding.project(
            binding.invocation_model.model_construct(),
            progress,
            invocation_artifact=invocation_binding,
            result_artifact=progress_binding,
        )


def test_material_projector_requires_native_post_apply_evidence(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "leaves" / "001-material.assignment.v1" / "attempts" / "01"
    native = attempt / "native"
    raw = native / "raw"
    raw.mkdir(parents=True)
    source = tmp_path / "source.usdz"
    source.write_bytes(b"source")
    materials_yaml = tmp_path / "materials.yaml"
    materials_yaml.write_text("materials: []\n", encoding="utf-8")
    materials_usd = tmp_path / "materials.usda"
    materials_usd.write_text("#usda 1.0\n", encoding="utf-8")
    staged_output = native / "output" / "materialized.usda"
    staged_output.parent.mkdir()
    staged_output.write_bytes(b"material-staging-output")
    staged_output_binding = artifact_binding(staged_output)
    output = native / "material.usdz"
    output.write_bytes(b"material-output")
    output_binding = artifact_binding(output)

    request = raw / "coordinator_request.json"
    request.write_text("{}\n", encoding="utf-8")
    decision = raw / "material_applied_decision_patch.json"
    decision.write_text("{}\n", encoding="utf-8")
    policy = raw / "material_finalization_policy.json"
    policy.write_text("{}\n", encoding="utf-8")
    checkpoint = raw / "receipt_checkpoint.json"
    checkpoint.write_text("{}\n", encoding="utf-8")
    render = native / "final_renders" / "canonical.png"
    render.parent.mkdir()
    render.write_bytes(b"ovrtx-render")
    render_binding = artifact_binding(render)
    request_binding = artifact_binding(request)
    decision_binding = artifact_binding(decision)

    application_binding = write_canonical_json(
        raw / "material_application_receipt.json",
        MaterialApplicationReviewReceipt(
            request=request_binding,
            decision_patch=decision_binding,
            policy=artifact_binding(policy),
            materialized_usd=staged_output_binding,
            receipt_checkpoint_binding=artifact_binding(checkpoint),
            final_render_bindings=(render_binding,),
            evidence=(staged_output_binding, render_binding),
            unresolved_issues=("Outer OVRTX review required.",),
        ),
    )
    native_binding = write_canonical_json(
        native / "coordinator_result.json",
        MaterialCoordinatorTerminalReceipt(
            status="pass",
            output_usd_path=str(output),
            output_usd_sha256=output_binding.sha256,
            request=request_binding,
            decision_patch=decision_binding,
            evidence=(staged_output_binding, render_binding),
        ),
    )
    release_binding = write_canonical_json(
        raw / "material_session_release.json",
        MaterialSessionReleaseReceipt(
            status="released",
            session_id="material-session",
        ),
    )
    invocation = MaterialAssignmentLeafInvocation(
        attempt_root=str(attempt),
        source=artifact_binding(source),
        repository_root=str(tmp_path),
        materials_yaml=artifact_binding(materials_yaml),
        materials_usd=artifact_binding(materials_usd),
        output_asset_path=str(output),
        decision_patch_path=str(raw / "material_decision_patch.json"),
        review_patch_path=str(raw / "material_post_apply_review.json"),
    )
    invocation_binding = write_canonical_json(attempt / "invocation.json", invocation)
    result = ComposedAssetLeafResult(
        leaf_id=MATERIAL_ASSIGNMENT_LEAF_ID,
        status="passed",
        native_status="pass",
        invocation=invocation_binding,
        native_terminal_receipt=native_binding,
        evidence=(
            render_binding,
            native_binding,
            application_binding,
            release_binding,
            output_binding,
            staged_output_binding,
            request_binding,
            decision_binding,
        ),
        saved_stage_readbacks=(
            native_binding,
            application_binding,
            staged_output_binding,
            output_binding,
        ),
        resource_claims=("usd-cli:material-session",),
        resource_release_receipts=(release_binding,),
        output_asset=output_binding,
        summary="Material passed.",
    )
    result_binding = write_canonical_json(attempt / "result.json", result)
    binding = _runtime_binding(MATERIAL_ASSIGNMENT_LEAF_ID)

    projection = binding.project(
        invocation,
        result,
        invocation_artifact=invocation_binding,
        result_artifact=result_binding,
    )
    assert projection.payload.native_disposition == "passed"

    omitted = result.model_copy(
        update={
            "evidence": tuple(
                item for item in result.evidence if item != render_binding
            )
        }
    )
    omitted_binding = write_canonical_json(attempt / "omitted-result.json", omitted)
    with pytest.raises(ValueError, match="omits native evidence"):
        binding.project(
            invocation,
            omitted,
            invocation_artifact=invocation_binding,
            result_artifact=omitted_binding,
        )


def test_full_composed_graph_covers_every_domain_with_one_terminal() -> None:
    runtime = compose_asset_leaf_runtime_bundles(
        (
            shared_asset_leaf_runtime_bundle(),
            articulation_asset_leaf_runtime_bundle(),
            texture_asset_leaf_runtime_bundle(),
            release_composed_asset_leaf_runtime_bundle(),
        )
    )
    descriptors = {
        descriptor.leaf_id: descriptor for descriptor in runtime.catalog.descriptors
    }
    selected = {
        *RELEASE_COMPOSED_LEAF_IDS,
        *(
            binding.descriptor.leaf_id
            for binding in articulation_asset_leaf_runtime_bundle().bindings
        ),
        *(
            binding.descriptor.leaf_id
            for binding in texture_asset_leaf_runtime_bundle().bindings
        ),
        CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
    }
    dependencies = {
        leaf_id: set(descriptors[leaf_id].required_dependencies) for leaf_id in selected
    }
    for leaf_id in selected:
        for dependent in descriptors[leaf_id].required_dependents:
            if dependent in selected:
                dependencies[dependent].add(leaf_id)

    dependencies[MATERIAL_ASSIGNMENT_LEAF_ID].add(ARTICULATION_PUBLISH_LEAF_ID)
    dependencies["articulation.author.v1"].add("articulation.proposal-provider.v1")
    dependencies["texture.uv-prepare.v1"].add(MATERIAL_ASSIGNMENT_LEAF_ID)
    dependencies[PHYSICS_INSPECTION_LEAF_ID].add("texture.publish.v1")
    dependencies[SIMREADY_CONFORMANCE_LEAF_ID].add(PHYSICS_APPLY_LEAF_ID)
    dependencies[FINAL_OVRTX_EVIDENCE_LEAF_ID].add(PORTABLE_PACKAGE_LEAF_ID)
    dependencies[COMBINED_RESULT_LEAF_ID].update(
        {ARTICULATION_PUBLISH_LEAF_ID, TEXTURE_PUBLISH_LEAF_ID}
    )
    required_dependency_contract = {
        FINAL_OVRTX_EVIDENCE_LEAF_ID: [PORTABLE_PACKAGE_LEAF_ID],
        COMBINED_RESULT_LEAF_ID: [
            ARTICULATION_PUBLISH_LEAF_ID,
            MATERIAL_ASSIGNMENT_LEAF_ID,
            PHYSICS_APPLY_LEAF_ID,
            PORTABLE_PACKAGE_LEAF_ID,
            SIMREADY_VALIDATION_LEAF_ID,
            TEXTURE_PUBLISH_LEAF_ID,
            FINAL_OVRTX_EVIDENCE_LEAF_ID,
        ],
        MATERIAL_ASSIGNMENT_LEAF_ID: [ARTICULATION_PUBLISH_LEAF_ID],
        PHYSICS_INSPECTION_LEAF_ID: [TEXTURE_PUBLISH_LEAF_ID],
        SIMREADY_CONFORMANCE_LEAF_ID: [PHYSICS_APPLY_LEAF_ID],
        "texture.uv-prepare.v1": [MATERIAL_ASSIGNMENT_LEAF_ID],
    }
    required_dependency_contract = {
        leaf_id: sorted(required_edges)
        for leaf_id, required_edges in sorted(required_dependency_contract.items())
    }

    graph = AssetExecutionGraph.create(
        sole_coordinator_identity_digest="1" * 64,
        prompt_digest="2" * 64,
        source_digest="3" * 64,
        configuration_digest="4" * 64,
        reference_digest="5" * 64,
        leaf_catalog_digest=runtime.catalog.catalog_digest,
        nodes=[
            AssetExecutionNode(
                leaf_id=leaf_id,
                depends_on=sorted(dependencies[leaf_id]),
                requirement="required",
                descriptor_digest=descriptors[leaf_id].descriptor_digest,
                terminal_output=leaf_id == COMBINED_RESULT_LEAF_ID,
            )
            for leaf_id in selected
        ],
        omitted_leaf_ids=sorted(set(descriptors).difference(selected)),
    )

    _validate_graph_catalog(
        graph,
        runtime.catalog,
        required_leaf_ids=sorted(selected),
        required_terminal_leaf_ids=[COMBINED_RESULT_LEAF_ID],
        required_leaf_dependencies=required_dependency_contract,
    )
    assert [node.leaf_id for node in graph.nodes if node.terminal_output] == [
        COMBINED_RESULT_LEAF_ID
    ]
