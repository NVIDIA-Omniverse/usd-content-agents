# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for durable three-phase large-scene orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from content_agent_workflows.asset_task_processing import (
    AgentPlanPointer,
    AssetTaskInventory,
    AssetTaskResult,
    AssetTaskResultsIndex,
    AssetTaskRunState,
    AssetTaskWorkItem,
    AssetTaskWorkItemState,
    DecisionLedgerEntry,
    ProcessingPhaseResult,
    ResultIndexEntry,
    TaskCatalog,
    TaskSpec,
)
from content_agent_workflows.common.artifacts import (
    artifact_set_digest,
    atomic_write_json,
    file_sha256,
    load_json,
    phase_result_digest,
    seal_phase_result,
)
from content_agent_workflows.large_scene import (
    begin_phase,
    complete_phase,
    create_run,
    invalidate_from,
    load_run_state,
    revise_additional_instructions,
    validate_phase_handoff,
)
from content_agent_workflows.large_scene import cli as large_scene_cli
from content_agent_workflows.large_scene.state import LargeSceneStateError
from content_agent_workflows.scene_collection import (
    CollectionPhaseResult,
    DomainCollectionResult,
)
from content_agent_workflows.scene_decomposition import (
    DecomposedAsset,
    DecompositionPhaseResult,
    ManifestCatalog,
    ManifestCatalogEntry,
    SceneDecompositionManifest,
)


def test_processing_phase_result_rejects_impossible_completion_counts() -> None:
    with pytest.raises(ValueError, match="completed_required_count"):
        ProcessingPhaseResult(
            success=True,
            input_digest="input",
            task_catalog_path="task_catalog.json",
            manifest_catalog_path="manifest_catalog.json",
            asset_task_inventory_path="inventory.json",
            work_item_state_path="state.json",
            agent_plan_pointer_path="plan_pointer.json",
            decision_ledger_path="decision_ledger.jsonl",
            results_index_path="results_index.json",
            required_work_item_count=1,
            completed_required_count=2,
        )

    with pytest.raises(ValueError, match="completed_optional_count"):
        ProcessingPhaseResult(
            success=True,
            input_digest="input",
            task_catalog_path="task_catalog.json",
            manifest_catalog_path="manifest_catalog.json",
            asset_task_inventory_path="inventory.json",
            work_item_state_path="state.json",
            agent_plan_pointer_path="plan_pointer.json",
            decision_ledger_path="decision_ledger.jsonl",
            results_index_path="results_index.json",
            required_work_item_count=0,
            completed_required_count=0,
            optional_work_item_count=1,
            completed_optional_count=2,
        )


def _write_usda(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#usda 1.0\n", encoding="utf-8")
    return path.resolve()


def test_source_input_digest_tracks_usd_sublayer_dependencies(
    tmp_path: Path,
) -> None:
    sublayer = tmp_path / "dependency.usda"
    _write_usda(sublayer)
    source = tmp_path / "source.usda"
    source.write_text(
        "#usda 1.0\n(\n    subLayers = [@dependency.usda@]\n)\n",
        encoding="utf-8",
    )
    run_state = tmp_path / "run.json"
    create_run(
        run_state,
        run_id="scene",
        source_scene=source,
        requested_tasks=["material"],
    )

    sublayer.write_text("#usda 1.0\n# dependency changed\n", encoding="utf-8")

    with pytest.raises(LargeSceneStateError, match="Source scene"):
        begin_phase(run_state, "decomposition")


def _complete_decomposition(run_state: Path, run_dir: Path, source: Path) -> None:
    run = begin_phase(run_state, "decomposition")
    input_digest = run.phases["decomposition"].input_digest
    assert input_digest

    extracted = _write_usda(run_dir / "01-decomposition" / "asset.usda")
    manifest_path = run_dir / "01-decomposition" / "scene_manifest.json"
    manifest = SceneDecompositionManifest(
        scene_id="pcb",
        original_usd_path=str(source),
        assets=[
            DecomposedAsset(
                asset_id="asset_001",
                label="Asset 001",
                original_root_path="/World/Asset",
                working_usd_path=str(extracted),
                working_root_path="/World/Asset",
            )
        ],
    )
    atomic_write_json(manifest_path, manifest)
    manifest_digest = file_sha256(manifest_path)
    source_identity_digest = artifact_set_digest([source])
    catalog_path = run_dir / "01-decomposition" / "manifest_catalog.json"
    catalog = ManifestCatalog(
        original_usd_path=str(source),
        source_identity_digest=source_identity_digest,
        structural_analysis_id=manifest_digest,
        manifests=[
            ManifestCatalogEntry(
                manifest_id="default",
                intent="material_processing",
                path=str(manifest_path),
                manifest_digest=manifest_digest,
            )
        ],
    )
    atomic_write_json(catalog_path, catalog)

    result_path = run_dir / "01-decomposition" / "decomposition_result.json"
    seal_phase_result(
        DecompositionPhaseResult(
            success=True,
            input_digest=input_digest,
            source_scene=str(source),
            source_identity_digest=source_identity_digest,
            manifest_catalog_path=str(catalog_path),
            manifest_paths=[str(manifest_path)],
            extracted_asset_paths=[str(extracted)],
            artifact_paths=[str(catalog_path), str(extracted), str(manifest_path)],
            completion_policy_satisfied=True,
        ),
        result_path,
    )
    report = validate_phase_handoff(run_state, "decomposition", result_path)
    assert report.valid, report.errors
    complete_phase(run_state, "decomposition", result_path)


def _complete_processing(run_state: Path, run_dir: Path) -> None:
    run = begin_phase(run_state, "asset_task_processing")
    input_digest = run.phases["asset_task_processing"].input_digest
    assert input_digest
    phase_dir = run_dir / "02-asset-tasks"
    manifest_catalog_path = run_dir / "01-decomposition" / "manifest_catalog.json"
    manifest_path = run_dir / "01-decomposition" / "scene_manifest.json"

    request_path = phase_dir / "material_request.json"
    request_payload = {"task": "material"}
    if run.additional_instructions:
        request_payload["additional_instructions"] = run.additional_instructions
    atomic_write_json(request_path, request_payload)
    request_digest = file_sha256(request_path)
    task_catalog_path = phase_dir / "task_catalog.json"
    atomic_write_json(
        task_catalog_path,
        TaskCatalog(
            tasks=[
                TaskSpec(
                    task_id="material",
                    domain="material",
                    skill="content-workflow-material",
                    manifest_id="default",
                    request_path=str(request_path),
                    validator="material",
                    collector="material",
                )
            ]
        ),
    )
    work_item_id = "material:default:asset_001"
    inventory_path = phase_dir / "asset_task_inventory.json"
    atomic_write_json(
        inventory_path,
        AssetTaskInventory(
            input_digest=input_digest,
            task_request_digests={"material": request_digest},
            work_items=[
                AssetTaskWorkItem(
                    work_item_id=work_item_id,
                    manifest_id="default",
                    asset_id="asset_001",
                    task_id="material",
                    original_root_path="/World/Asset",
                    working_usd_path=str(run_dir / "01-decomposition" / "asset.usda"),
                )
            ],
        ),
    )
    state_path = phase_dir / "asset_task_run_state.json"
    atomic_write_json(
        state_path,
        AssetTaskRunState(
            input_digest=input_digest,
            inventory_path=str(inventory_path),
            task_catalog_path=str(task_catalog_path),
            manifest_catalog_path=str(manifest_catalog_path),
            task_request_digests={"material": request_digest},
            work_items=[
                AssetTaskWorkItemState(
                    work_item_id=work_item_id,
                    status="completed",
                    attempt_count=1,
                )
            ],
        ),
    )
    plan_path = phase_dir / "agent_plan" / "revision-0001.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        "Process the representative material task.\n", encoding="utf-8"
    )
    plan_pointer_path = phase_dir / "agent_plan" / "current.json"
    atomic_write_json(
        plan_pointer_path,
        AgentPlanPointer(
            current_revision=1,
            current_plan_path=str(plan_path),
            revision_paths=[str(plan_path)],
        ),
    )

    decisions_path = phase_dir / "assets" / "asset_001" / "decisions.json"
    atomic_write_json(decisions_path, {"bindings": []})
    validation_path = phase_dir / "assets" / "asset_001" / "validation.json"
    atomic_write_json(validation_path, {"passed": True})
    work_result_path = phase_dir / "assets" / "asset_001" / "result.json"
    atomic_write_json(
        work_result_path,
        AssetTaskResult(
            task_id="material",
            domain="material",
            manifest_id="default",
            asset_id="asset_001",
            original_root_path="/World/Asset",
            working_usd_path=str(run_dir / "01-decomposition" / "asset.usda"),
            domain_outputs={"decisions_path": str(decisions_path)},
            provenance={
                "agent_plan_revision": 1,
                "task_request_digest": request_digest,
            },
        ),
    )
    index_path = phase_dir / "asset_task_results_index.json"
    atomic_write_json(
        index_path,
        AssetTaskResultsIndex(
            entries=[
                ResultIndexEntry(
                    work_item_id=work_item_id,
                    status="completed",
                    result_path=str(work_result_path),
                    validation_path=str(validation_path),
                )
            ]
        ),
    )
    ledger_path = phase_dir / "decision_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_entry = DecisionLedgerEntry(
        work_item_id=work_item_id,
        domain="material",
        task_id="material",
        evidence_summary="Representative asset inspected.",
        confidence=0.9,
        rationale="The result is explicit and validated.",
        validation_status="passed",
        agent_plan_revision=1,
        task_request_digest=request_digest,
    )
    ledger_path.write_text(ledger_entry.model_dump_json() + "\n", encoding="utf-8")

    result_path = phase_dir / "processing_result.json"
    seal_phase_result(
        ProcessingPhaseResult(
            success=True,
            input_digest=input_digest,
            task_catalog_path=str(task_catalog_path),
            manifest_catalog_path=str(manifest_catalog_path),
            asset_task_inventory_path=str(inventory_path),
            work_item_state_path=str(state_path),
            agent_plan_pointer_path=str(plan_pointer_path),
            decision_ledger_path=str(ledger_path),
            results_index_path=str(index_path),
            task_request_digests={"material": request_digest},
            required_work_item_count=1,
            completed_required_count=1,
            artifact_paths=[
                str(decisions_path),
                str(index_path),
                str(inventory_path),
                str(ledger_path),
                str(manifest_catalog_path),
                str(manifest_path),
                str(plan_path),
                str(plan_pointer_path),
                str(request_path),
                str(state_path),
                str(task_catalog_path),
                str(validation_path),
                str(work_result_path),
            ],
            completion_policy_satisfied=True,
        ),
        result_path,
    )
    report = validate_phase_handoff(run_state, "asset_task_processing", result_path)
    assert report.valid, report.errors
    complete_phase(run_state, "asset_task_processing", result_path)


def _complete_collection(run_state: Path, run_dir: Path) -> None:
    run = begin_phase(run_state, "collection")
    input_digest = run.phases["collection"].input_digest
    assert input_digest
    phase_dir = run_dir / "03-collection"
    processing_result = run_dir / "02-asset-tasks" / "processing_result.json"
    task_catalog = run_dir / "02-asset-tasks" / "task_catalog.json"
    task_request = run_dir / "02-asset-tasks" / "material_request.json"
    output_layer = _write_usda(phase_dir / "domains" / "material" / "output.usda")
    collection_report = phase_dir / "domains" / "material" / "collection.json"
    harmonization_report = phase_dir / "domains" / "material" / "harmonization.json"
    validation_report = phase_dir / "domains" / "material" / "validation.json"
    atomic_write_json(collection_report, {"collected": 1})
    atomic_write_json(harmonization_report, {"conflicts": []})
    atomic_write_json(validation_report, {"passed": True})

    domain_result_path = phase_dir / "domains" / "material" / "result.json"
    domain_artifacts = [
        str(collection_report),
        str(harmonization_report),
        str(output_layer),
        str(validation_report),
    ]
    seal_phase_result(
        DomainCollectionResult(
            domain="material",
            status="completed",
            output_paths=[str(output_layer)],
            collection_report_path=str(collection_report),
            harmonization_report_path=str(harmonization_report),
            validation_report_path=str(validation_report),
            validation_passed=True,
            artifact_paths=domain_artifacts,
        ),
        domain_result_path,
    )
    topology_report = phase_dir / "composition" / "topology.json"
    atomic_write_json(topology_report, {"passed": True})
    result_path = phase_dir / "collection_result.json"
    seal_phase_result(
        CollectionPhaseResult(
            success=True,
            input_digest=input_digest,
            domain_result_paths=[str(domain_result_path)],
            required_domain_count=1,
            completed_required_domain_count=1,
            topology_report_path=str(topology_report),
            final_output_paths=[str(output_layer)],
            artifact_paths=[
                str(processing_result),
                str(task_catalog),
                str(task_request),
                *domain_artifacts,
                str(domain_result_path),
                str(topology_report),
            ],
            completion_policy_satisfied=True,
        ),
        result_path,
    )
    report = validate_phase_handoff(run_state, "collection", result_path)
    assert report.valid, report.errors
    complete_phase(run_state, "collection", result_path)


def test_three_phase_run_advances_and_invalidates_backward(tmp_path: Path) -> None:
    source = _write_usda(tmp_path / "pcb.usda")
    run_dir = tmp_path / "run"
    run_state = run_dir / "large_scene_run.json"
    created = create_run(
        run_state,
        run_id="pcb-material",
        source_scene=source,
        requested_tasks=["material"],
        additional_instructions=(
            "Keep frames white and reserve yellow for isolated safety accents."
        ),
    )
    assert created.phases["decomposition"].status == "ready"
    assert created.additional_instructions is not None

    _complete_decomposition(run_state, run_dir, source)
    assert load_run_state(run_state).phases["asset_task_processing"].status == "ready"
    _complete_processing(run_state, run_dir)
    assert load_run_state(run_state).phases["collection"].status == "ready"
    _complete_collection(run_state, run_dir)

    completed = load_run_state(run_state)
    assert completed.current_phase is None
    assert all(state.status == "completed" for state in completed.phases.values())

    repaired = invalidate_from(
        run_state,
        "asset_task_processing",
        reason="Collector requested a revised task result.",
    )
    assert repaired.phases["decomposition"].status == "completed"
    assert repaired.phases["asset_task_processing"].status == "ready"
    assert repaired.phases["collection"].status == "invalidated"
    assert repaired.phases["collection"].output_digest is None


def test_collection_handoff_rejects_missing_requested_domain(tmp_path: Path) -> None:
    source = _write_usda(tmp_path / "pcb.usda")
    run_dir = tmp_path / "run"
    run_state = run_dir / "large_scene_run.json"
    create_run(
        run_state,
        run_id="pcb-material",
        source_scene=source,
        requested_tasks=["material"],
    )
    _complete_decomposition(run_state, run_dir, source)
    _complete_processing(run_state, run_dir)
    _complete_collection(run_state, run_dir)

    completed = load_run_state(run_state)
    completed.requested_tasks.append("physics")
    atomic_write_json(run_state, completed)
    result_path = run_dir / "03-collection" / "collection_result.json"

    report = validate_phase_handoff(run_state, "collection", result_path)

    assert not report.valid
    assert any(
        "missing requested domains: ['physics']" in error for error in report.errors
    )


def test_revise_instructions_preserves_completed_decomposition(tmp_path: Path) -> None:
    source = _write_usda(tmp_path / "pcb.usda")
    run_dir = tmp_path / "run"
    run_state = run_dir / "large_scene_run.json"
    create_run(
        run_state,
        run_id="pcb-material",
        source_scene=source,
        requested_tasks=["material"],
        additional_instructions="Use reference colors.",
    )
    _complete_decomposition(run_state, run_dir, source)
    before = load_run_state(run_state)

    revised = revise_additional_instructions(
        run_state,
        additional_instructions="Use display color only beneath /World/Robot.",
        reason="Authorize scoped display-color evidence for the robot.",
    )

    assert revised.additional_instructions == (
        "Use display color only beneath /World/Robot."
    )
    assert revised.source_input_digest != before.source_input_digest
    assert revised.phases["decomposition"] == before.phases["decomposition"]
    assert revised.phases["asset_task_processing"].status == "ready"
    assert (
        revised.phases["asset_task_processing"].input_digest
        == before.phases["decomposition"].output_digest
    )
    assert revised.phases["collection"].status == "invalidated"
    assert revised.phases["collection"].input_digest is None
    assert revised.current_phase == "asset_task_processing"


def test_begin_phase_rechecks_source_inputs_after_decomposition(tmp_path: Path) -> None:
    source = _write_usda(tmp_path / "pcb.usda")
    run_dir = tmp_path / "run"
    run_state = run_dir / "large_scene_run.json"
    create_run(
        run_state,
        run_id="pcb-material",
        source_scene=source,
        requested_tasks=["material"],
    )
    _complete_decomposition(run_state, run_dir, source)
    source.write_text("#usda 1.0\n# changed\n", encoding="utf-8")

    with pytest.raises(LargeSceneStateError, match="changed"):
        begin_phase(run_state, "asset_task_processing")


def test_begin_phase_rejects_resealed_predecessor_result(tmp_path: Path) -> None:
    source = _write_usda(tmp_path / "pcb.usda")
    run_dir = tmp_path / "run"
    run_state = run_dir / "large_scene_run.json"
    create_run(
        run_state,
        run_id="pcb-material",
        source_scene=source,
        requested_tasks=["material"],
    )
    _complete_decomposition(run_state, run_dir, source)

    result_path = run_dir / "01-decomposition" / "decomposition_result.json"
    result = DecompositionPhaseResult.model_validate(load_json(result_path))
    resealed = result.model_copy(
        update={"diagnostics": [{"message": "resealed replacement"}]}
    )
    seal_phase_result(resealed, result_path)

    with pytest.raises(LargeSceneStateError, match="output digest changed"):
        begin_phase(run_state, "asset_task_processing")


def test_phase_result_digest_is_stable_after_json_roundtrip(tmp_path: Path) -> None:
    source = _write_usda(tmp_path / "pcb.usda")
    artifact = _write_usda(tmp_path / "artifact.usda")
    result_path = tmp_path / "decomposition_result.json"
    result = DecompositionPhaseResult(
        success=True,
        input_digest="input",
        source_scene=str(source),
        source_identity_digest=artifact_set_digest([source]),
        manifest_catalog_path=str(artifact),
        manifest_paths=[str(artifact)],
        artifact_paths=[str(artifact)],
        completion_policy_satisfied=True,
    )
    sealed = seal_phase_result(result, result_path)
    loaded = DecompositionPhaseResult.model_validate(load_json(result_path))

    assert phase_result_digest(loaded, result_path=result_path) == sealed.output_digest


def test_handoff_detects_tampered_decomposition_artifact(tmp_path: Path) -> None:
    source = _write_usda(tmp_path / "pcb.usda")
    run_dir = tmp_path / "run"
    run_state = run_dir / "large_scene_run.json"
    create_run(
        run_state,
        run_id="pcb-material",
        source_scene=source,
        requested_tasks=["material"],
    )
    _complete_decomposition(run_state, run_dir, source)

    manifest_path = run_dir / "01-decomposition" / "scene_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    completed_result = run_dir / "01-decomposition" / "decomposition_result.json"
    report = validate_phase_handoff(run_state, "decomposition", completed_result)
    assert not report.valid
    assert any("digest mismatch" in error for error in report.errors)


def test_processing_handoff_rejects_ledger_task_domain_mismatch(
    tmp_path: Path,
) -> None:
    source = _write_usda(tmp_path / "pcb.usda")
    run_dir = tmp_path / "run"
    run_state = run_dir / "large_scene_run.json"
    create_run(
        run_state,
        run_id="pcb-material",
        source_scene=source,
        requested_tasks=["material"],
    )
    _complete_decomposition(run_state, run_dir, source)
    _complete_processing(run_state, run_dir)

    work_item_id = "material:default:asset_001"
    ledger_path = run_dir / "02-asset-tasks" / "decision_ledger.jsonl"
    ledger_entry = DecisionLedgerEntry(
        work_item_id=work_item_id,
        domain="physics",
        task_id="material",
        evidence_summary="Representative asset inspected.",
        confidence=0.9,
        rationale="The result is explicit and validated.",
        validation_status="passed",
        agent_plan_revision=1,
    )
    ledger_path.write_text(ledger_entry.model_dump_json() + "\n", encoding="utf-8")

    result_path = run_dir / "02-asset-tasks" / "processing_result.json"
    result_payload = load_json(result_path)
    result_payload["output_digest"] = None
    seal_phase_result(ProcessingPhaseResult.model_validate(result_payload), result_path)

    report = validate_phase_handoff(run_state, "asset_task_processing", result_path)

    assert not report.valid
    assert any(
        "decision ledger task/domain mismatch" in error for error in report.errors
    )


def test_large_scene_cli_creates_and_reads_run_state(tmp_path: Path, capsys) -> None:
    source = _write_usda(tmp_path / "pcb.usda")
    run_state = tmp_path / "run" / "large_scene_run.json"
    guidance_path = tmp_path / "material-guidance.md"
    guidance_path.write_text(
        "White structural frames.\nDo not spread yellow across machines.\n",
        encoding="utf-8",
    )
    rc = large_scene_cli.main(
        [
            "create",
            "--run-state",
            str(run_state),
            "--run-id",
            "pcb-material",
            "--source-scene",
            str(source),
            "--task",
            "material",
            "--additional-instructions-file",
            str(guidance_path),
        ]
    )
    assert rc == 0
    created = json.loads(capsys.readouterr().out)
    assert created["phases"]["decomposition"]["status"] == "ready"
    assert created["additional_instructions"] == (
        "White structural frames.\nDo not spread yellow across machines."
    )

    rc = large_scene_cli.main(["status", "--run-state", str(run_state)])
    assert rc == 0
    status = json.loads(capsys.readouterr().out)
    assert status["run_id"] == "pcb-material"
    assert status["additional_instructions"] == created["additional_instructions"]


def test_large_scene_cli_revise_instructions_rejects_blank_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = large_scene_cli.main(
        [
            "revise-instructions",
            "--run-state",
            str(tmp_path / "run.json"),
            "--additional-instructions",
            "   ",
            "--reason",
            "Blank guidance should be rejected.",
        ]
    )

    assert rc == 2
    assert "requires non-empty instructions" in capsys.readouterr().err
