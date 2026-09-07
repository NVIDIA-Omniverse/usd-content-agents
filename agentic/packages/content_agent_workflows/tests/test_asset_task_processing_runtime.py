# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Workflow 2 runtime state and the material adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from content_agent_workflows.asset_task_processing import (
    AssetTaskResult,
    DecisionLedgerEntry,
    material_appearance,
)
from content_agent_workflows.asset_task_processing.contracts import (
    AssetTaskInventory,
    AssetTaskResultsIndex,
    AssetTaskRunState,
    ResultIndexEntry,
    TaskCatalog,
    TaskSpec,
)
from content_agent_workflows.asset_task_processing.material_appearance import (
    MaterialAppearanceEntry,
    MaterialAppearanceIndex,
    RenderedAppearance,
    rank_display_color_candidates,
    representative_srgb,
    srgb_to_lab,
)
from content_agent_workflows.asset_task_processing.material_task import (
    WINDOWS_OPENUSD_REPLACE_SUFFIX_CODE_UNITS,
    MaterialAssignmentDecision,
    MaterialBatchItem,
    MaterialBatchPlan,
    MaterialDecisionPatch,
    _require_ovrtx_probe,
    _safe_path_component,
    _validate_saved_material_bindings,
    match_work_item_display_colors,
    run_material_batch,
    run_material_work_item,
    survey_material_inventory,
    survey_usd_material_candidates,
)
from content_agent_workflows.asset_task_processing.material_task import (
    main as material_task_main,
)
from content_agent_workflows.asset_task_processing.runtime import (
    AssetTaskRuntimeError,
    begin_work_item,
    commit_work_item,
    finalize_processing_run,
    prepare_processing_run,
    processing_status,
    record_plan,
    waive_work_item,
)
from content_agent_workflows.common.artifacts import (
    artifact_set_digest,
    atomic_write_json,
    file_sha256,
    load_json,
)
from content_agent_workflows.material_assignment import load_material_manifest
from content_agent_workflows.scene_decomposition import (
    DecomposedAsset,
    ManifestCatalog,
    ManifestCatalogEntry,
    SceneDecompositionManifest,
)


def _write_mesh(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Mesh "Mesh"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    }
}
""",
        encoding="utf-8",
    )
    return path.resolve()


def _write_material_library(
    path: Path,
    *,
    material_paths: tuple[str, ...] = ("/World/Looks/Plastic_Green",),
) -> Path:
    from pxr import Usd, UsdShade

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    for material_path in material_paths:
        UsdShade.Material.Define(stage, material_path)
    stage.GetRootLayer().Save()
    return path.resolve()


def _write_materialized_mesh(
    path: Path,
    *,
    material_path: str = "/World/Looks/Plastic_Green",
) -> Path:
    from pxr import Usd, UsdShade

    _write_mesh(path)
    stage = Usd.Stage.Open(str(path))
    material = UsdShade.Material.Define(stage, material_path)
    mesh = stage.GetPrimAtPath("/World/Mesh")
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
    stage.GetRootLayer().Save()
    return path.resolve()


def _write_conflicting_materialized_mesh(path: Path) -> Path:
    """Write a red local material at the path used by the green test library."""

    from pxr import Gf, Sdf, Usd, UsdShade

    _write_mesh(path)
    stage = Usd.Stage.Open(str(path))
    material = UsdShade.Material.Define(stage, "/World/Looks/Plastic_Green")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Plastic_Green/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.0, 0.0)
    )
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    mesh = stage.GetPrimAtPath("/World/Mesh")
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
    stage.GetRootLayer().Save()
    return path.resolve()


def _write_display_color_mesh(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Mesh "RobotShell"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        color3f[] primvars:displayColor = [(0.898, 0.447, 0.102)] (
            interpolation = "constant"
        )
    }
}
""",
        encoding="utf-8",
    )
    return path.resolve()


def _write_bound_material_mesh(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Scope "Looks"
    {
        def Material "DarkCADHint"
        {
            token outputs:surface.connect = </World/Looks/DarkCADHint/Preview.outputs:surface>

            def Shader "Preview"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.2, 0.2, 0.2)
                float inputs:metallic = 0.5
                float inputs:roughness = 0.2
            }
        }
    }

    def Mesh "Mesh"
    {
        rel material:binding = </World/Looks/DarkCADHint>
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    }
}
""",
        encoding="utf-8",
    )
    return path.resolve()


def _write_visible_and_hidden_meshes(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Mesh "VisibleMesh"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    }

    def Xform "HiddenGroup"
    {
        token visibility = "invisible"

        def Mesh "HiddenMesh"
        {
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        }
    }
}
""",
        encoding="utf-8",
    )
    return path.resolve()


USER_GUIDANCE = "Keep structural frames white and reserve yellow for safety accents."


@pytest.mark.parametrize("value", ["", ".", "..", "a/b", "a\\b"])
def test_safe_path_component_rejects_path_escape(value: str) -> None:
    with pytest.raises(AssetTaskRuntimeError, match="Unsafe asset_id path component"):
        _safe_path_component(value, "asset_id")


def test_task_catalog_rejects_dependency_cycles() -> None:
    with pytest.raises(ValueError, match="acyclic"):
        TaskCatalog(
            tasks=[
                TaskSpec(
                    task_id="material",
                    domain="material",
                    skill="content-workflow-material",
                    manifest_id="manifest",
                    request_path="material_request.json",
                    validator="material",
                    collector="material",
                    depends_on=["physics"],
                ),
                TaskSpec(
                    task_id="physics",
                    domain="physics",
                    skill="content-workflow-physics",
                    manifest_id="manifest",
                    request_path="physics_request.json",
                    validator="physics",
                    collector="physics",
                    depends_on=["material"],
                ),
            ]
        )


def _prepare_run(
    tmp_path: Path,
    *,
    material_request_overrides: dict[str, object] | None = None,
    asset_count: int = 1,
    source_with_conflicting_material: bool = False,
) -> tuple[Path, Path, str, str]:
    source_writer = (
        _write_conflicting_materialized_mesh
        if source_with_conflicting_material
        else _write_mesh
    )
    source = source_writer(tmp_path / "source.usda")
    manifest_path = tmp_path / "decomposition" / "scene_manifest.json"
    atomic_write_json(
        manifest_path,
        SceneDecompositionManifest(
            scene_id="source",
            original_usd_path=str(source),
            assets=[
                DecomposedAsset(
                    asset_id=f"asset_{index:03d}",
                    label=f"Asset {index:03d}",
                    original_root_path=(
                        "/World" if index == 1 else f"/World/Asset_{index:03d}"
                    ),
                    working_usd_path=str(source),
                    working_root_path="/World",
                )
                for index in range(1, asset_count + 1)
            ],
        ),
    )
    manifest_catalog_path = tmp_path / "decomposition" / "manifest_catalog.json"
    atomic_write_json(
        manifest_catalog_path,
        ManifestCatalog(
            original_usd_path=str(source),
            source_identity_digest=artifact_set_digest([source]),
            structural_analysis_id=file_sha256(manifest_path),
            manifests=[
                ManifestCatalogEntry(
                    manifest_id="default",
                    intent="material_processing",
                    path=str(manifest_path),
                    manifest_digest=file_sha256(manifest_path),
                )
            ],
        ),
    )
    request_path = tmp_path / "processing" / "material_request.json"
    request_payload = {
        "schema_version": "content-agent-workflows.material-task-request.v2",
        "reference_images": [],
        "additional_instructions": USER_GUIDANCE,
    }
    request_payload.update(material_request_overrides or {})
    atomic_write_json(request_path, request_payload)
    request_digest = file_sha256(request_path)
    task_catalog_path = tmp_path / "processing" / "task_catalog.json"
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
    input_digest = "workflow-2-input"
    run_dir = tmp_path / "processing"
    prepare_processing_run(
        manifest_catalog_path=manifest_catalog_path,
        task_catalog_path=task_catalog_path,
        output_dir=run_dir,
        input_digest=input_digest,
    )
    plan_draft = tmp_path / "plan.md"
    plan_draft.write_text(
        "Process each material representative independently.\n", encoding="utf-8"
    )
    record_plan(run_dir, plan_draft)
    return run_dir, source, input_digest, request_digest


def _prepare_dependency_run(tmp_path: Path) -> Path:
    source = _write_mesh(tmp_path / "source.usda")
    manifest_path = tmp_path / "decomposition" / "scene_manifest.json"
    atomic_write_json(
        manifest_path,
        SceneDecompositionManifest(
            scene_id="source",
            original_usd_path=str(source),
            assets=[
                DecomposedAsset(
                    asset_id="asset_001",
                    label="Asset 001",
                    original_root_path="/World/A",
                    working_usd_path=str(source),
                    working_root_path="/World",
                ),
                DecomposedAsset(
                    asset_id="asset_002",
                    label="Asset 002",
                    original_root_path="/World/B",
                    working_usd_path=str(source),
                    working_root_path="/World",
                ),
            ],
        ),
    )
    manifest_catalog_path = tmp_path / "decomposition" / "manifest_catalog.json"
    atomic_write_json(
        manifest_catalog_path,
        ManifestCatalog(
            original_usd_path=str(source),
            source_identity_digest=artifact_set_digest([source]),
            structural_analysis_id=file_sha256(manifest_path),
            manifests=[
                ManifestCatalogEntry(
                    manifest_id="default",
                    intent="multi_task_processing",
                    path=str(manifest_path),
                    manifest_digest=file_sha256(manifest_path),
                )
            ],
        ),
    )
    request_path = tmp_path / "processing" / "task_request.json"
    atomic_write_json(request_path, {"instructions": USER_GUIDANCE})
    task_catalog_path = tmp_path / "processing" / "task_catalog.json"
    atomic_write_json(
        task_catalog_path,
        TaskCatalog(
            tasks=[
                TaskSpec(
                    task_id="physics",
                    domain="physics",
                    skill="content-workflow-physics",
                    manifest_id="default",
                    request_path=str(request_path),
                    validator="physics",
                    collector="physics",
                ),
                TaskSpec(
                    task_id="material",
                    domain="material",
                    skill="content-workflow-material",
                    manifest_id="default",
                    request_path=str(request_path),
                    validator="material",
                    collector="material",
                    depends_on=["physics"],
                ),
            ]
        ),
    )
    run_dir = tmp_path / "processing"
    prepare_processing_run(
        manifest_catalog_path=manifest_catalog_path,
        task_catalog_path=task_catalog_path,
        output_dir=run_dir,
        input_digest="workflow-2-dependency-input",
    )
    return run_dir


def test_dependencies_are_scoped_to_matching_asset_work_item(tmp_path: Path) -> None:
    run_dir = _prepare_dependency_run(tmp_path)

    initial = processing_status(run_dir)
    assert initial["eligible_work_item_ids"] == [
        "physics:default:asset_001",
        "physics:default:asset_002",
    ]

    waive_work_item(
        run_dir,
        "physics:default:asset_001",
        reason="Accepted physics waiver for dependency scoping test.",
        accepted_by="test",
    )
    updated = processing_status(run_dir)

    assert "material:default:asset_001" in updated["eligible_work_item_ids"]
    assert "material:default:asset_002" not in updated["eligible_work_item_ids"]
    assert "physics:default:asset_002" in updated["eligible_work_item_ids"]


def _write_commit_payload(
    run_dir: Path,
    source: Path,
    request_digest: str,
    *,
    validation_payload: object = None,
    ledger_plan_revision: int = 1,
) -> tuple[Path, Path, Path]:
    item_dir = run_dir / "assets" / "default" / "asset_001" / "tasks" / "material"
    decisions_path = item_dir / "decisions.json"
    atomic_write_json(decisions_path, {"assignments": []})
    validation_path = item_dir / "validation.json"
    if validation_payload is None:
        atomic_write_json(validation_path, {"passed": True})
    else:
        validation_path.write_text(
            json.dumps(validation_payload) + "\n", encoding="utf-8"
        )
    result_path = item_dir / "result.json"
    atomic_write_json(
        result_path,
        AssetTaskResult(
            task_id="material",
            domain="material",
            manifest_id="default",
            asset_id="asset_001",
            original_root_path="/World",
            working_usd_path=str(source),
            domain_outputs={"decisions_path": str(decisions_path)},
            provenance={
                "agent_plan_revision": 1,
                "task_request_digest": request_digest,
            },
        ),
    )
    ledger_entry_path = item_dir / "ledger_entry.json"
    atomic_write_json(
        ledger_entry_path,
        DecisionLedgerEntry(
            work_item_id="material:default:asset_001",
            domain="material",
            task_id="material",
            evidence_summary="Single source mesh.",
            confidence=1.0,
            rationale="Explicit test assignment.",
            validation_status="passed",
            agent_plan_revision=ledger_plan_revision,
            task_request_digest=request_digest,
        ),
    )
    return result_path, validation_path, ledger_entry_path


def test_runtime_prepares_commits_and_finalizes(tmp_path: Path) -> None:
    run_dir, source, _input_digest, request_digest = _prepare_run(tmp_path)
    work_item_id = "material:default:asset_001"
    begin_work_item(run_dir, work_item_id)

    result_path, validation_path, ledger_entry_path = _write_commit_payload(
        run_dir,
        source,
        request_digest,
    )
    commit_work_item(
        run_dir,
        work_item_id,
        result_path=result_path,
        validation_path=validation_path,
        ledger_entry_path=ledger_entry_path,
    )

    status = processing_status(run_dir)
    assert status["status_counts"] == {"completed": 1}
    phase_result = finalize_processing_run(run_dir)
    assert phase_result.success
    assert phase_result.completed_required_count == 1
    assert Path(phase_result.work_item_state_path).is_file()

    inventory = AssetTaskInventory.model_validate(
        load_json(run_dir / "asset_task_inventory.json")
    )
    assert "status" not in inventory.work_items[0].model_dump()
    state = AssetTaskRunState.model_validate(
        load_json(run_dir / "asset_task_run_state.json")
    )
    assert state.work_items[0].status == "completed"


def test_commit_recovers_indexed_completion_after_state_write_crash(
    tmp_path: Path,
) -> None:
    run_dir, source, _input_digest, request_digest = _prepare_run(tmp_path)
    work_item_id = "material:default:asset_001"
    begin_work_item(run_dir, work_item_id)
    result_path, validation_path, ledger_entry_path = _write_commit_payload(
        run_dir,
        source,
        request_digest,
    )
    ledger_entry = DecisionLedgerEntry.model_validate(load_json(ledger_entry_path))
    (run_dir / "decision_ledger.jsonl").write_text(
        ledger_entry.model_dump_json() + "\n", encoding="utf-8"
    )
    atomic_write_json(
        run_dir / "asset_task_results_index.json",
        AssetTaskResultsIndex(
            entries=[
                ResultIndexEntry(
                    work_item_id=work_item_id,
                    status="completed",
                    result_path=str(result_path),
                    validation_path=str(validation_path),
                )
            ]
        ),
    )

    state = commit_work_item(
        run_dir,
        work_item_id,
        result_path=result_path,
        validation_path=validation_path,
        ledger_entry_path=ledger_entry_path,
    )

    assert state.work_items[0].status == "completed"
    assert state.transitions[-1].from_status == "running"
    assert "Recovered completed work item" in state.transitions[-1].reason


def test_finalize_recovers_indexed_completion_after_state_write_crash(
    tmp_path: Path,
) -> None:
    run_dir, source, _input_digest, request_digest = _prepare_run(tmp_path)
    work_item_id = "material:default:asset_001"
    begin_work_item(run_dir, work_item_id)
    result_path, validation_path, ledger_entry_path = _write_commit_payload(
        run_dir,
        source,
        request_digest,
    )
    ledger_entry = DecisionLedgerEntry.model_validate(load_json(ledger_entry_path))
    (run_dir / "decision_ledger.jsonl").write_text(
        ledger_entry.model_dump_json() + "\n", encoding="utf-8"
    )
    atomic_write_json(
        run_dir / "asset_task_results_index.json",
        AssetTaskResultsIndex(
            entries=[
                ResultIndexEntry(
                    work_item_id=work_item_id,
                    status="completed",
                    result_path=str(result_path),
                    validation_path=str(validation_path),
                )
            ]
        ),
    )

    phase_result = finalize_processing_run(run_dir)

    assert phase_result.success
    assert phase_result.completed_required_count == 1
    state = AssetTaskRunState.model_validate(
        load_json(run_dir / "asset_task_run_state.json")
    )
    assert state.work_items[0].status == "completed"
    assert "Recovered completed work item" in state.transitions[-1].reason


def test_commit_rejects_non_object_validation_report(tmp_path: Path) -> None:
    run_dir, source, _input_digest, request_digest = _prepare_run(tmp_path)
    work_item_id = "material:default:asset_001"
    begin_work_item(run_dir, work_item_id)
    result_path, validation_path, ledger_entry_path = _write_commit_payload(
        run_dir,
        source,
        request_digest,
        validation_payload=["passed"],
    )

    with pytest.raises(AssetTaskRuntimeError, match="JSON object|expected object"):
        commit_work_item(
            run_dir,
            work_item_id,
            result_path=result_path,
            validation_path=validation_path,
            ledger_entry_path=ledger_entry_path,
        )


def test_commit_rejects_ledger_plan_revision_mismatch(tmp_path: Path) -> None:
    run_dir, source, _input_digest, request_digest = _prepare_run(tmp_path)
    work_item_id = "material:default:asset_001"
    begin_work_item(run_dir, work_item_id)
    result_path, validation_path, ledger_entry_path = _write_commit_payload(
        run_dir,
        source,
        request_digest,
        ledger_plan_revision=2,
    )

    with pytest.raises(AssetTaskRuntimeError, match="plan revision"):
        commit_work_item(
            run_dir,
            work_item_id,
            result_path=result_path,
            validation_path=validation_path,
            ledger_entry_path=ledger_entry_path,
        )


def test_finalize_rejects_unresolved_required_work(tmp_path: Path) -> None:
    run_dir, _source, _input_digest, _request_digest = _prepare_run(tmp_path)
    try:
        finalize_processing_run(run_dir)
    except AssetTaskRuntimeError as exc:
        assert "Required work items remain unresolved" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("finalize should reject planned required work")


def test_waiver_keeps_completion_timestamp_reserved_for_completed_work(
    tmp_path: Path,
) -> None:
    run_dir, _source, _input_digest, _request_digest = _prepare_run(tmp_path)

    state = waive_work_item(
        run_dir,
        "material:default:asset_001",
        reason="Reviewer accepted omission.",
        accepted_by="reviewer",
    )

    assert state.work_items[0].status == "waived"
    assert state.work_items[0].completed_at is None
    assert state.accepted_waivers[0].accepted_at == state.transitions[-1].timestamp


def test_material_survey_index_carries_scene_level_guidance(tmp_path: Path) -> None:
    run_dir, _source, _input_digest, request_digest = _prepare_run(tmp_path)

    summary = survey_material_inventory(run_dir)
    index = load_json(summary["index_path"])

    assert summary["task_request_digest"] == request_digest
    assert index["task_request"] == {
        "path": str(run_dir / "material_request.json"),
        "sha256": request_digest,
        "additional_instructions": USER_GUIDANCE,
        "appearance_evidence_policy": {
            "schema_version": "content-agent-workflows.appearance-evidence-policy.v1",
            "default": "ignore",
            "global_sources": [],
            "scopes": [],
        },
    }


def test_runtime_rejects_task_request_mutation_after_prepare(
    tmp_path: Path,
) -> None:
    run_dir, _source, _input_digest, _request_digest = _prepare_run(tmp_path)
    atomic_write_json(
        run_dir / "material_request.json",
        {
            "schema_version": "content-agent-workflows.material-task-request.v2",
            "additional_instructions": "Changed guidance",
        },
    )

    with pytest.raises(AssetTaskRuntimeError, match="Task request changed"):
        processing_status(run_dir)


def test_saved_material_validation_rejects_same_leaf_at_wrong_manifest_path(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    library_path = _write_material_library(tmp_path / "materials.usda")
    library_yaml = tmp_path / "materials.yaml"
    library_yaml.write_text(
        """library_path: materials.usda
entries:
  - name: Plastic Green
    binding: /World/Looks/Plastic_Green
""",
        encoding="utf-8",
    )
    manifest = load_material_manifest(library_yaml)
    output_path = _write_materialized_mesh(
        tmp_path / "materialized.usda",
        material_path="/Counterfeit/Plastic_Green",
    )
    output_stage = Usd.Stage.Open(str(output_path))
    assert output_stage is not None
    decision = MaterialDecisionPatch(
        work_item_id="material:default:asset_001",
        source_usd=str(output_path),
        material_library_yaml=str(library_yaml),
        material_library_path=str(library_path),
        assignments=[
            MaterialAssignmentDecision(
                target_prim_path="/World/Mesh",
                covered_candidate_paths=["/World/Mesh"],
                material_name="Plastic Green",
                rationale="Matches the selected manifest material.",
                confidence=0.9,
            )
        ],
        evidence_summary="Selected from the authoritative manifest.",
        confidence=0.9,
    )

    errors = _validate_saved_material_bindings(
        output_stage=output_stage,
        decision=decision,
        manifest=manifest,
        scene_backend="usd-cli",
    )

    assert errors == [
        "Saved usd-cli derivative bound the wrong material at "
        "/World/Mesh: expected manifest path '/World/Looks/Plastic_Green', "
        "got '/Counterfeit/Plastic_Green'."
    ]


def test_saved_material_validation_accepts_default_prim_localization(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdShade

    library_path = _write_material_library(tmp_path / "materials.usda")
    library_yaml = tmp_path / "materials.yaml"
    library_yaml.write_text(
        """library_path: materials.usda
entries:
  - name: Plastic Green
    binding: /World/Looks/Plastic_Green
""",
        encoding="utf-8",
    )
    manifest = load_material_manifest(library_yaml)
    output_path = tmp_path / "localized.usda"
    output_stage = Usd.Stage.CreateNew(str(output_path))
    asset = output_stage.DefinePrim("/Asset", "Xform")
    output_stage.SetDefaultPrim(asset)
    mesh = output_stage.DefinePrim("/Asset/Mesh", "Mesh")
    localized = UsdShade.Material.Define(output_stage, "/Asset/Looks/Plastic_Green")
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(localized)
    output_stage.GetRootLayer().Save()
    decision = MaterialDecisionPatch(
        work_item_id="material:default:asset_001",
        source_usd=str(output_path),
        material_library_yaml=str(library_yaml),
        material_library_path=str(library_path),
        assignments=[
            MaterialAssignmentDecision(
                target_prim_path="/Asset/Mesh",
                covered_candidate_paths=["/Asset/Mesh"],
                material_name="Plastic Green",
                rationale="Matches the selected manifest material.",
                confidence=0.9,
            )
        ],
        evidence_summary="Selected from the authoritative manifest.",
        confidence=0.9,
    )

    assert (
        _validate_saved_material_bindings(
            output_stage=output_stage,
            decision=decision,
            manifest=manifest,
            scene_backend="usd-cli",
        )
        == []
    )


def test_saved_material_validation_rejects_usd_cli_same_leaf_outside_default_prim(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdShade

    library_path = _write_material_library(tmp_path / "materials.usda")
    library_yaml = tmp_path / "materials.yaml"
    library_yaml.write_text(
        """library_path: materials.usda
entries:
  - name: Plastic Green
    binding: /World/Looks/Plastic_Green
""",
        encoding="utf-8",
    )
    manifest = load_material_manifest(library_yaml)
    output_path = tmp_path / "counterfeit.usda"
    output_stage = Usd.Stage.CreateNew(str(output_path))
    asset = output_stage.DefinePrim("/Asset", "Xform")
    output_stage.SetDefaultPrim(asset)
    mesh = output_stage.DefinePrim("/Asset/Mesh", "Mesh")
    counterfeit = UsdShade.Material.Define(
        output_stage, "/Counterfeit/Looks/Plastic_Green"
    )
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(counterfeit)
    output_stage.GetRootLayer().Save()
    decision = MaterialDecisionPatch(
        work_item_id="material:default:asset_001",
        source_usd=str(output_path),
        material_library_yaml=str(library_yaml),
        material_library_path=str(library_path),
        assignments=[
            MaterialAssignmentDecision(
                target_prim_path="/Asset/Mesh",
                covered_candidate_paths=["/Asset/Mesh"],
                material_name="Plastic Green",
                rationale="Matches the selected manifest material.",
                confidence=0.9,
            )
        ],
        evidence_summary="Selected from the authoritative manifest.",
        confidence=0.9,
    )

    assert _validate_saved_material_bindings(
        output_stage=output_stage,
        decision=decision,
        manifest=manifest,
        scene_backend="usd-cli",
    ) == [
        "Saved usd-cli derivative bound the wrong material at /Asset/Mesh: "
        "expected manifest path '/World/Looks/Plastic_Green', got "
        "'/Counterfeit/Looks/Plastic_Green'."
    ]


def test_usd_cli_material_batch_isolates_assets_and_resumes_only_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library_path = _write_material_library(tmp_path / "materials.usda")
    library_yaml = tmp_path / "materials.yaml"
    library_yaml.write_text(
        """library_path: materials.usda
entries:
  - name: Plastic Green
    description: Durable green coating
    binding: /World/Looks/Plastic_Green
    tags: [plastic, green]
""",
        encoding="utf-8",
    )
    run_dir, source, _input_digest, request_digest = _prepare_run(
        tmp_path,
        asset_count=2,
        material_request_overrides={
            "material_library_yaml": str(library_yaml),
            "material_library_path": str(library_path),
            "processing_policy": {
                "scene_backend": "usd-cli",
                "scene_session_scope": "per_asset",
            },
        },
    )
    source_digest = file_sha256(source)
    work_item_ids = [
        "material:default:asset_001",
        "material:default:asset_002",
    ]
    plan_items: list[MaterialBatchItem] = []
    for work_item_id in work_item_ids:
        decision_path = run_dir / f"{work_item_id.rsplit(':', 1)[-1]}-decision.json"
        atomic_write_json(
            decision_path,
            MaterialDecisionPatch(
                work_item_id=work_item_id,
                source_usd=str(source),
                material_library_yaml=str(library_yaml),
                material_library_path=str(library_path),
                task_request_digest=request_digest,
                assignments=[
                    MaterialAssignmentDecision(
                        target_prim_path="/World/Mesh",
                        covered_candidate_paths=["/World/Mesh"],
                        material_name="Plastic Green",
                        rationale="Uses the accepted workflow decision.",
                        confidence=0.95,
                    )
                ],
                evidence_summary="Accepted reference and geometry evidence.",
                confidence=0.95,
            ),
        )
        plan_items.append(
            MaterialBatchItem(
                work_item_id=work_item_id,
                decision_path=str(decision_path),
            )
        )
    plan_path = atomic_write_json(
        run_dir / "usd_cli_batch_plan.json",
        MaterialBatchPlan(items=plan_items),
    )

    calls: list[tuple[Path, str, tuple[str, ...]]] = []
    stopped_projects: list[Path] = []
    failed_once = False

    class FakeWorkflowUsdCliSession:
        def __init__(
            self, *, project_dir: Path, session_id: str, identity: str
        ) -> None:
            self.project_dir = project_dir
            self.session_id = session_id
            self.identity = identity

        @classmethod
        def create(
            cls,
            *,
            project_dir: Path,
            identity: str,
            **_kwargs: object,
        ) -> FakeWorkflowUsdCliSession:
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / ".usd-cli").mkdir(exist_ok=True)
            return cls(
                project_dir=project_dir,
                session_id=f"material-{hashlib.sha256(identity.encode()).hexdigest()[:20]}",
                identity=identity,
            )

        def open(self, path: Path) -> dict[str, object]:
            return self.run_json(["open", str(path.resolve())])

        def require_ovrtx(self, output_dir: Path) -> dict[str, object]:
            return {
                **self.run_json(
                    [
                        "render-probe",
                        "--require-engine",
                        "ovrtx",
                        "--output-dir",
                        str(output_dir),
                    ]
                ),
                "execution_source": "render-probe",
            }

        def close(self) -> None:
            stopped_projects.append(self.project_dir)

        def run_json(
            self,
            arguments: list[str],
            *,
            timeout_seconds: float = 1800.0,
        ) -> dict[str, object]:
            del timeout_seconds
            nonlocal failed_once
            calls.append((self.project_dir, self.session_id, tuple(arguments)))
            if arguments[:1] == ["render-probe"]:
                probe_dir = Path(arguments[arguments.index("--output-dir") + 1])
                probe_dir.mkdir(parents=True, exist_ok=True)
                probe_image = probe_dir / "ovrtx_readiness_probe.png"
                Image.new("RGB", (64, 64), "green").save(probe_image)
                return {
                    "schema_version": "usd-cli.render-probe.v1",
                    "capabilities": ["appearance.clear.v1"],
                    "resolved_renderer": "ovrtx",
                    "engine": "ovrtx",
                    "transport": "local",
                    "ready": True,
                    "render": {
                        "path": str(probe_image),
                        "width": 64,
                        "height": 64,
                        "size_bytes": probe_image.stat().st_size,
                        "backend": "ovrtx",
                    },
                }
            if (
                self.identity.startswith(f"{work_item_ids[1]}:attempt:")
                and arguments[:1] == ["material"]
                and arguments[1:2] != ["audit"]
                and not failed_once
            ):
                failed_once = True
                raise AssetTaskRuntimeError("transient usd-cli bind failure")
            if arguments == ["appearance", "audit"]:
                return {
                    "ok": True,
                    "data": {
                        "clear": True,
                        "overlay_active": True,
                        "counts": {
                            "binding_relationships_with_targets": 0,
                            "effective_material_bindings": 0,
                            "effective_shader_appearances": 0,
                            "display_values": 0,
                            "instance_proxies": 0,
                        },
                    },
                }
            if arguments[:1] == ["save"]:
                output_path = Path(arguments[1])
                if output_path.name == "clean_slate_layer.usda":
                    _write_mesh(output_path)
                else:
                    _write_materialized_mesh(output_path)
            return {"ok": True, "data": {}}

    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task.WorkflowUsdCliSession",
        FakeWorkflowUsdCliSession,
    )

    first = run_material_batch(
        run_dir,
        plan_path,
    )
    assert first["completed_work_item_ids"] == [work_item_ids[0]]
    assert list(first["failures"]) == [work_item_ids[1]]
    assert processing_status(run_dir)["status_counts"] == {
        "completed": 1,
        "failed": 1,
    }

    resumed = run_material_batch(
        run_dir,
        plan_path,
    )
    assert resumed["completed_work_item_ids"] == work_item_ids
    assert resumed["failures"] == {}
    assert file_sha256(source) == source_digest
    assert len(stopped_projects) == 3

    source_opens = [
        (project, session)
        for project, session, arguments in calls
        if arguments == ("open", str(source.resolve()))
    ]
    clean_slate_opens = [
        (project, session, arguments)
        for project, session, arguments in calls
        if arguments[:1] == ("open",)
        and len(arguments) == 3
        and Path(arguments[1]).name == "clean_slate_layer.usda"
    ]

    def session_id(work_item_id: str, attempt: int) -> str:
        identity = f"{work_item_id}:attempt:{attempt}"
        return f"material-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"

    asset_001_opens = [
        item for item in source_opens if item[1] == session_id(work_item_ids[0], 1)
    ]
    asset_002_opens = [
        item
        for item in source_opens
        if item[1] in {session_id(work_item_ids[1], 1), session_id(work_item_ids[1], 2)}
    ]
    assert len(asset_001_opens) == 1
    assert len(asset_002_opens) == 2
    assert len(clean_slate_opens) == 3
    assert all(
        arguments[-1] == "--force-reload" for *_item, arguments in clean_slate_opens
    )
    assert asset_001_opens[0][0] != asset_002_opens[0][0]
    assert asset_001_opens[0][1] != asset_002_opens[0][1]
    assert asset_002_opens[0][0] != asset_002_opens[1][0]
    assert asset_002_opens[0][1] != asset_002_opens[1][1]
    if os.name == "nt":
        assert asset_001_opens[0][0].name.endswith("-a0001")
        assert asset_002_opens[0][0].name.endswith("-a0001")
        assert asset_002_opens[1][0].name.endswith("-a0002")
        assert all(project.parent == run_dir for project, _session in source_opens)
        for project, _session in source_opens:
            clean_slate = project / "clean_slate_layer.usda"
            code_units = len(str(clean_slate).encode("utf-16-le")) // 2
            assert code_units + WINDOWS_OPENUSD_REPLACE_SUFFIX_CODE_UNITS < 260
    else:
        assert asset_001_opens[0][0].name == "attempt-0001"
        assert asset_002_opens[0][0].name == "attempt-0001"
        assert asset_002_opens[1][0].name == "attempt-0002"
    assert all((project / ".usd-cli").is_dir() for project, _session in source_opens)
    for project, session in source_opens:
        project_calls = [
            arguments
            for call_project, call_session, arguments in calls
            if call_project == project and call_session == session
        ]
        assert project_calls[0][:1] == ("render-probe",)
        assert project_calls[1] == ("open", str(source.resolve()))
        assert project_calls[3] == ("appearance", "clear")
        assert project_calls[4] == ("appearance", "audit")
        assert project_calls[5][0] == "save"
        assert Path(project_calls[5][1]).name == "clean_slate_layer.usda"
        assert project_calls[5][2] == "--flatten"
        assert project_calls[6][0] == "open"
        assert Path(project_calls[6][1]).name == "clean_slate_layer.usda"
        assert project_calls[6][2] == "--force-reload"
        assert project_calls[7] == (
            "checkpoint",
            "save",
            "workflow-clean-slate",
            "--full",
        )
        assert project_calls[8][0] == "material"
    assert stopped_projects == [
        asset_001_opens[0][0],
        asset_002_opens[0][0],
        asset_002_opens[1][0],
    ]

    result = load_json(
        run_dir
        / "assets"
        / "default"
        / "asset_001"
        / "tasks"
        / "material"
        / "result.json"
    )
    palette = load_json(result["domain_outputs"]["material_palette_path"])
    palette_entry = palette["materials"][0]
    assert palette_entry["name"] == "Plastic Green"
    assert palette_entry["material_path"] == "/World/Looks/Plastic_Green"
    assert palette_entry["description"] == "Durable green coating"
    assert palette_entry["tags"] == ["plastic", "green"]
    assert Path(result["domain_outputs"]["backend_artifact_3"]).name == (
        "clean_slate_layer.usda"
    )
    assert Path(result["domain_outputs"]["backend_artifact_3"]).is_file()


def test_material_batch_fail_fast_stops_after_first_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = MaterialBatchPlan(
        items=[
            MaterialBatchItem(
                work_item_id="material:default:asset_001",
                decision_path="asset_001.json",
            ),
            MaterialBatchItem(
                work_item_id="material:default:asset_002",
                decision_path="asset_002.json",
            ),
        ],
        stop_on_error=False,
    )
    plan_path = atomic_write_json(tmp_path / "batch.json", plan)
    attempted: list[str] = []

    def fail_work_item(
        _processing_dir: str | Path,
        work_item_id: str,
        **_kwargs: object,
    ) -> AssetTaskResult:
        attempted.append(work_item_id)
        raise AssetTaskRuntimeError("system boundary failed")

    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task.run_material_work_item",
        fail_work_item,
    )

    result = run_material_batch(tmp_path, plan_path, fail_fast=True)

    assert attempted == ["material:default:asset_001"]
    assert result["completed_count"] == 0
    assert result["failed_count"] == 1


def test_material_batch_fail_fast_cli_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def failed_batch(
        processing_dir: str | Path,
        batch_plan_path: str | Path,
        *,
        actor: str,
        fail_fast: bool,
    ) -> dict[str, object]:
        observed.update(
            processing_dir=processing_dir,
            batch_plan_path=batch_plan_path,
            actor=actor,
            fail_fast=fail_fast,
        )
        return {
            "completed_count": 0,
            "failed_count": 1,
            "completed_work_item_ids": [],
            "failures": {"material:default:asset_001": "system boundary failed"},
        }

    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task.run_material_batch",
        failed_batch,
    )
    plan_path = tmp_path / "batch.json"

    returncode = material_task_main(
        [
            "run-batch",
            "--processing-dir",
            str(tmp_path),
            "--batch-plan",
            str(plan_path),
            "--fail-fast",
        ]
    )

    assert returncode == 2
    assert observed["fail_fast"] is True
    assert json.loads(capsys.readouterr().out)["failed_count"] == 1


def test_usd_cli_material_probe_requires_strict_contained_64px_evidence(
    tmp_path: Path,
) -> None:
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe_image = probe_dir / "ovrtx_readiness_probe.png"
    Image.new("RGB", (64, 64), "green").save(probe_image)
    valid_probe: dict[str, object] = {
        "schema_version": "usd-cli.render-probe.v1",
        "capabilities": ["appearance.clear.v1"],
        "resolved_renderer": "ovrtx",
        "engine": "ovrtx",
        "transport": "local",
        "ready": True,
        "render": {
            "path": str(probe_image),
            "width": 64,
            "height": 64,
            "size_bytes": probe_image.stat().st_size,
            "backend": "ovrtx",
        },
    }

    assert (
        _require_ovrtx_probe(
            valid_probe,
            probe_dir=probe_dir,
            require_clear_appearance=True,
        )
        == probe_image
    )

    forged_identity = dict(valid_probe)
    forged_identity.pop("transport")
    with pytest.raises(AssetTaskRuntimeError, match="invalid transport"):
        _require_ovrtx_probe(
            forged_identity,
            probe_dir=probe_dir,
            require_clear_appearance=True,
        )

    external_image = tmp_path / "outside.png"
    Image.new("RGB", (64, 64), "red").save(external_image)
    escaped_render = dict(valid_probe["render"])
    escaped_render["path"] = str(external_image)
    escaped_probe = {**valid_probe, "render": escaped_render}
    with pytest.raises(AssetTaskRuntimeError, match="outside the workflow run"):
        _require_ovrtx_probe(
            escaped_probe,
            probe_dir=probe_dir,
            require_clear_appearance=True,
        )

    escaped_symlink = probe_dir / "escaped.png"
    try:
        escaped_symlink.symlink_to(external_image)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")
    symlinked_render = dict(valid_probe["render"])
    symlinked_render["path"] = str(escaped_symlink)
    symlinked_probe = {**valid_probe, "render": symlinked_render}
    with pytest.raises(AssetTaskRuntimeError, match="must not contain symlinks"):
        _require_ovrtx_probe(
            symlinked_probe,
            probe_dir=probe_dir,
            require_clear_appearance=True,
        )

    corrupt_image = probe_dir / "corrupt.png"
    corrupt_image.write_bytes(b"not an image")
    corrupt_render = dict(valid_probe["render"])
    corrupt_render.update(
        {
            "path": str(corrupt_image),
            "size_bytes": corrupt_image.stat().st_size,
        }
    )
    corrupt_probe = {**valid_probe, "render": corrupt_render}
    with pytest.raises(AssetTaskRuntimeError, match="not decodable"):
        _require_ovrtx_probe(
            corrupt_probe,
            probe_dir=probe_dir,
            require_clear_appearance=True,
        )

    Image.new("RGB", (32, 32), "blue").save(probe_image)
    wrong_dimensions_render = dict(valid_probe["render"])
    wrong_dimensions_render["size_bytes"] = probe_image.stat().st_size
    wrong_dimensions_probe = {
        **valid_probe,
        "render": wrong_dimensions_render,
    }
    with pytest.raises(AssetTaskRuntimeError, match="artifact has unexpected"):
        _require_ovrtx_probe(
            wrong_dimensions_probe,
            probe_dir=probe_dir,
            require_clear_appearance=True,
        )


@pytest.mark.parametrize(
    ("library_kind", "manifest_entries", "message"),
    [
        (
            "materials",
            [
                {"name": "Duplicate", "binding": "/World/Looks/Plastic_Green"},
                {"name": "Duplicate", "binding": "/World/Looks/Other"},
            ],
            "Duplicate material manifest name",
        ),
        (
            "materials",
            [
                {"name": "First", "binding": "/World/Looks/Plastic_Green"},
                {"name": "Second", "binding": "/World/Looks/Plastic_Green"},
            ],
            "Duplicate material manifest binding",
        ),
        (
            "non_material",
            [{"name": "Wrong", "binding": "/World/Looks/NotMaterial"}],
            "must name UsdShade.Material",
        ),
    ],
)
def test_usd_cli_manifest_contract_fails_before_scene_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    library_kind: str,
    manifest_entries: list[dict[str, str]],
    message: str,
) -> None:
    from pxr import Usd

    library_path = tmp_path / "materials.usda"
    if library_kind == "non_material":
        stage = Usd.Stage.CreateNew(str(library_path))
        stage.DefinePrim("/World/Looks/NotMaterial", "Scope")
        stage.GetRootLayer().Save()
    else:
        _write_material_library(
            library_path,
            material_paths=(
                "/World/Looks/Plastic_Green",
                "/World/Looks/Other",
            ),
        )
    library_yaml = tmp_path / "materials.yaml"
    library_yaml.write_text(
        "library_path: materials.usda\nentries:\n"
        + "".join(
            f"  - name: {entry['name']}\n    binding: {entry['binding']}\n"
            for entry in manifest_entries
        ),
        encoding="utf-8",
    )
    run_dir, source, _input_digest, request_digest = _prepare_run(
        tmp_path,
        material_request_overrides={
            "material_library_yaml": str(library_yaml),
            "material_library_path": str(library_path.resolve()),
            "processing_policy": {
                "scene_backend": "usd-cli",
                "scene_session_scope": "per_asset",
            },
        },
    )
    material_name = manifest_entries[0]["name"]
    decision_path = atomic_write_json(
        run_dir / "decision.json",
        MaterialDecisionPatch(
            work_item_id="material:default:asset_001",
            source_usd=str(source),
            material_library_yaml=str(library_yaml),
            material_library_path=str(library_path.resolve()),
            task_request_digest=request_digest,
            assignments=[
                MaterialAssignmentDecision(
                    target_prim_path="/World/Mesh",
                    covered_candidate_paths=["/World/Mesh"],
                    material_name=material_name,
                    rationale="Must be validated before scene mutation.",
                    confidence=0.9,
                )
            ],
            evidence_summary="Contract test.",
            confidence=0.9,
        ),
    )
    session_creations: list[str] = []
    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task."
        "WorkflowUsdCliSession.create",
        lambda **_kwargs: session_creations.append("usd-cli"),
    )
    with pytest.raises(AssetTaskRuntimeError, match=message):
        run_material_work_item(
            run_dir,
            "material:default:asset_001",
            decision_path=decision_path,
        )
    assert session_creations == []
    assert processing_status(run_dir)["status_counts"] == {"planned": 1}


def test_usd_cli_unsupported_manifest_lookup_fails_before_scene_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library_path = _write_material_library(
        tmp_path / "materials.usda",
        material_paths=("/World/Looks/Internal_Prim_Name",),
    )
    library_yaml = tmp_path / "materials.yaml"
    library_yaml.write_text(
        """library_path: materials.usda
entries:
  - name: Friendly Display Alias
    binding: /World/Looks/Internal_Prim_Name
""",
        encoding="utf-8",
    )
    run_dir, source, _input_digest, request_digest = _prepare_run(
        tmp_path,
        material_request_overrides={
            "material_library_yaml": str(library_yaml),
            "material_library_path": str(library_path),
            "processing_policy": {
                "scene_backend": "usd-cli",
                "scene_session_scope": "per_asset",
            },
        },
    )
    decision_path = atomic_write_json(
        run_dir / "decision.json",
        MaterialDecisionPatch(
            work_item_id="material:default:asset_001",
            source_usd=str(source),
            material_library_yaml=str(library_yaml),
            material_library_path=str(library_path),
            task_request_digest=request_digest,
            assignments=[
                MaterialAssignmentDecision(
                    target_prim_path="/World/Mesh",
                    covered_candidate_paths=["/World/Mesh"],
                    material_name="Friendly Display Alias",
                    rationale="Exercises an unavailable low-level lookup.",
                    confidence=0.9,
                )
            ],
            evidence_summary="Contract test.",
            confidence=0.9,
        ),
    )
    session_creations: list[str] = []
    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task."
        "WorkflowUsdCliSession.create",
        lambda **_kwargs: session_creations.append("usd-cli"),
    )

    with pytest.raises(AssetTaskRuntimeError, match="cannot unambiguously address"):
        run_material_work_item(
            run_dir,
            "material:default:asset_001",
            decision_path=decision_path,
        )
    assert session_creations == []
    assert processing_status(run_dir)["status_counts"] == {"planned": 1}


def test_usd_cli_existing_local_material_destination_fails_before_scene_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library_path = _write_material_library(tmp_path / "materials.usda")
    library_yaml = tmp_path / "materials.yaml"
    library_yaml.write_text(
        """library_path: materials.usda
entries:
  - name: Plastic Green
    binding: /World/Looks/Plastic_Green
""",
        encoding="utf-8",
    )
    run_dir, source, _input_digest, request_digest = _prepare_run(
        tmp_path,
        material_request_overrides={
            "material_library_yaml": str(library_yaml),
            "material_library_path": str(library_path),
            "processing_policy": {
                "scene_backend": "usd-cli",
                "scene_session_scope": "per_asset",
            },
        },
        source_with_conflicting_material=True,
    )
    decision_path = atomic_write_json(
        run_dir / "decision.json",
        MaterialDecisionPatch(
            work_item_id="material:default:asset_001",
            source_usd=str(source),
            material_library_yaml=str(library_yaml),
            material_library_path=str(library_path),
            task_request_digest=request_digest,
            assignments=[
                MaterialAssignmentDecision(
                    target_prim_path="/World/Mesh",
                    covered_candidate_paths=["/World/Mesh"],
                    material_name="Plastic Green",
                    rationale=(
                        "Exercises a same-path source material with stronger red "
                        "shader opinions than the selected library material."
                    ),
                    confidence=0.9,
                )
            ],
            evidence_summary="Collision safety contract test.",
            confidence=0.9,
        ),
    )
    session_creations: list[str] = []
    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task."
        "WorkflowUsdCliSession.create",
        lambda **_kwargs: session_creations.append("usd-cli"),
    )

    with pytest.raises(
        AssetTaskRuntimeError,
        match="library import destination already exists",
    ):
        run_material_work_item(
            run_dir,
            "material:default:asset_001",
            decision_path=decision_path,
        )

    assert session_creations == []
    assert processing_status(run_dir)["status_counts"] == {"planned": 1}


def test_material_adapter_enforces_frozen_material_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_library_path = _write_material_library(
        tmp_path / "expected_materials.usda"
    )
    expected_yaml = tmp_path / "expected_materials.yaml"
    expected_yaml.write_text(
        """library_path: expected_materials.usda
entries:
  - name: Plastic Green
    binding: /World/Looks/Plastic_Green
""",
        encoding="utf-8",
    )
    other_library_path = _write_material_library(tmp_path / "other_materials.usda")
    other_yaml = tmp_path / "other_materials.yaml"
    other_yaml.write_text(
        """library_path: other_materials.usda
entries:
  - name: Plastic Green
    binding: /World/Looks/Plastic_Green
""",
        encoding="utf-8",
    )
    run_dir, source, _input_digest, request_digest = _prepare_run(
        tmp_path,
        material_request_overrides={
            "material_library_yaml": str(expected_yaml),
            "material_library_path": str(expected_library_path),
        },
    )
    work_item_id = "material:default:asset_001"
    decision_path = run_dir / "material_decision.json"
    atomic_write_json(
        decision_path,
        MaterialDecisionPatch(
            work_item_id=work_item_id,
            source_usd=str(source),
            material_library_yaml=str(other_yaml),
            material_library_path=str(other_library_path),
            task_request_digest=request_digest,
            assignments=[
                MaterialAssignmentDecision(
                    target_prim_path="/World/Mesh",
                    covered_candidate_paths=["/World/Mesh"],
                    material_name="Plastic Green",
                    rationale="Uses an out-of-scope library.",
                    confidence=0.9,
                )
            ],
            evidence_summary="Reference shows a green surface.",
            confidence=0.9,
        ),
    )

    with pytest.raises(AssetTaskRuntimeError, match="frozen material task request"):
        run_material_work_item(
            run_dir,
            work_item_id,
            decision_path=decision_path,
            render=False,
        )
    assert processing_status(run_dir)["status_counts"] == {"planned": 1}


def test_material_survey_skips_computed_invisible_meshes(tmp_path: Path) -> None:
    source = _write_visible_and_hidden_meshes(tmp_path / "visibility.usda")

    survey = survey_usd_material_candidates(
        work_item_id="material:default:asset_001",
        asset_label="Visibility",
        usd_path=source,
        original_root_path="/World",
    )

    assert [candidate.prim_path for candidate in survey.candidates] == [
        "/World/VisibleMesh"
    ]
    assert survey.visibility_policy == "visible_only"
    assert survey.skipped_invisible_mesh_count == 1


def test_material_survey_redacts_authored_appearance_by_default(
    tmp_path: Path,
) -> None:
    source = _write_bound_material_mesh(tmp_path / "bound.usda")

    survey = survey_usd_material_candidates(
        work_item_id="material:default:asset_001",
        asset_label="CAD",
        usd_path=source,
        original_root_path="/World",
    )

    candidate = survey.candidates[0]
    assert candidate.bound_material_name is None
    assert candidate.bound_material_path is None
    assert candidate.diffuse_color is None
    assert candidate.metallic is None
    assert candidate.roughness is None


def test_material_survey_exposes_scoped_authored_appearance(
    tmp_path: Path,
) -> None:
    source = _write_display_color_mesh(tmp_path / "display_color.usda")

    survey = survey_usd_material_candidates(
        work_item_id="material:default:asset_001",
        asset_label="Robot",
        usd_path=source,
        original_root_path="/World",
        appearance_evidence_policy={
            "scopes": [
                {
                    "root": "/World",
                    "sources": ["display_color"],
                    "reason": "User explicitly asked to use robot display colors.",
                }
            ]
        },
    )

    candidate = survey.candidates[0]
    assert candidate.display_color == pytest.approx([0.898, 0.447, 0.102])
    assert candidate.display_color_interpolation == "constant"
    assert candidate.display_color_value_count == 1
    assert candidate.diffuse_color is None


def test_material_survey_exposes_scoped_bound_material_hint(tmp_path: Path) -> None:
    source = _write_bound_material_mesh(tmp_path / "bound.usda")

    survey = survey_usd_material_candidates(
        work_item_id="material:default:asset_001",
        asset_label="CAD",
        usd_path=source,
        original_root_path="/World",
        appearance_evidence_policy={
            "scopes": [
                {
                    "root": "/World",
                    "sources": ["material_binding"],
                    "reason": "User explicitly asked to use CAD material hints.",
                }
            ]
        },
    )

    candidate = survey.candidates[0]
    assert candidate.bound_material_name == "DarkCADHint"
    assert candidate.diffuse_color == pytest.approx([0.2, 0.2, 0.2])
    assert candidate.metallic == pytest.approx(0.5)
    assert candidate.display_color is None


def test_display_color_matching_requires_authorized_scope(tmp_path: Path) -> None:
    run_dir, _source, _input_digest, _request_digest = _prepare_run(tmp_path)

    with pytest.raises(
        AssetTaskRuntimeError,
        match="appearance_evidence_policy scopes that include display_color",
    ):
        match_work_item_display_colors(
            run_dir,
            "material:default:asset_001",
            scope_paths=["/World"],
        )


def test_representative_srgb_measures_center_swatch_patch(tmp_path: Path) -> None:
    image_path = tmp_path / "swatch.png"
    image = Image.new("RGB", (100, 100), (220, 220, 220))
    image.paste((64, 128, 192), (35, 35, 65, 65))
    image.save(image_path)

    measured = representative_srgb(image_path, crop_fraction=0.2)

    assert measured == pytest.approx([64 / 255, 128 / 255, 192 / 255])
    assert srgb_to_lab([1.0, 1.0, 1.0])[0] == pytest.approx(100.0)
    assert srgb_to_lab([0.0, 0.0, 0.0])[0] == pytest.approx(0.0)


def test_representative_srgb_rejects_corrupt_image(tmp_path: Path) -> None:
    image_path = tmp_path / "corrupt.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(
        material_appearance.MaterialAppearanceError,
        match="Could not measure rendered swatch image",
    ):
        representative_srgb(image_path)


def test_display_color_target_cache_tracks_render_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "template.usd"
    template.write_text("#usda 1.0\n", encoding="utf-8")
    output_dir = tmp_path / "targets"
    rendered_names: list[str] = []
    session_projects: list[Path] = []

    class FakeSession:
        @classmethod
        def create(cls, **kwargs: object) -> FakeSession:
            session_projects.append(Path(str(kwargs["project_dir"])))
            return cls()

        def require_ovrtx(self, _output_dir: Path) -> dict[str, object]:
            return {"ready": True, "engine": "ovrtx"}

        def open(self, _path: Path) -> dict[str, object]:
            return {"ok": True}

        def run_json(self, _arguments: list[str]) -> dict[str, object]:
            return {"ok": True}

        def close(self) -> None:
            return None

    monkeypatch.setattr(material_appearance, "WorkflowUsdCliSession", FakeSession)

    def fake_render_override(**kwargs: object) -> RenderedAppearance:
        name = str(kwargs["name"])
        rendered_names.append(name)
        image_path = Path(str(kwargs["output_dir"])) / f"{name}.png"
        Image.new("RGB", (16, 16), (255, 0, 0)).save(image_path)
        return RenderedAppearance(
            swatch_path=str(image_path),
            representative_srgb=[1.0, 0.0, 0.0],
            representative_lab=srgb_to_lab([1.0, 0.0, 0.0]),
        )

    monkeypatch.setattr(material_appearance, "_render_override", fake_render_override)
    color = [1.0, 0.0, 0.0]
    material_appearance.render_display_color_targets(
        colors=[color],
        swatch_template_path=template,
        output_dir=output_dir,
    )
    monkeypatch.setattr(
        material_appearance,
        "_SWATCH_RENDER_CONFIG",
        {**material_appearance._SWATCH_RENDER_CONFIG, "camera_mode": "reframed"},
    )
    material_appearance.render_display_color_targets(
        colors=[color],
        swatch_template_path=template,
        output_dir=output_dir,
    )

    color_digest = hashlib.sha256(b"1.000000,0.000000,0.000000").hexdigest()[:12]
    assert len(rendered_names) == 2
    assert session_projects == [output_dir.resolve(), output_dir.resolve()]
    assert rendered_names[0].startswith(f"display_color_{color_digest}_")
    assert rendered_names[0] != rendered_names[1]


def test_display_color_matching_ranks_rendered_materials_within_scope(
    tmp_path: Path,
) -> None:
    gold_swatch = tmp_path / "gold.png"
    orange_swatch = tmp_path / "orange.png"
    target_swatch = tmp_path / "target.png"
    for path in (gold_swatch, orange_swatch, target_swatch):
        Image.new("RGB", (16, 16), (128, 128, 128)).save(path)
    appearance_index = MaterialAppearanceIndex(
        cache_key="cache-key",
        material_library_yaml=str(tmp_path / "materials.yaml"),
        material_library_path=str(tmp_path / "materials.usd"),
        material_library_yaml_digest="yaml-digest",
        material_library_usd_digest="usd-digest",
        swatch_template_path=str(tmp_path / "template.usd"),
        swatch_template_digest="template-digest",
        render_config={},
        materials=[
            MaterialAppearanceEntry(
                material_name="Gold Matte",
                material_path="/World/Looks/Gold_Matte",
                description="Muted gold metal",
                swatch_path=str(gold_swatch),
                representative_srgb=[0.72, 0.58, 0.28],
                representative_lab=[62.0, 4.0, 42.0],
            ),
            MaterialAppearanceEntry(
                material_name="Plastic Orange",
                material_path="/World/Looks/Plastic_Orange",
                description="Orange plastic",
                swatch_path=str(orange_swatch),
                representative_srgb=[0.9, 0.25, 0.05],
                representative_lab=[54.0, 48.0, 58.0],
            ),
        ],
    )
    target = RenderedAppearance(
        swatch_path=str(target_swatch),
        representative_srgb=[0.7, 0.57, 0.3],
        representative_lab=[61.0, 5.0, 40.0],
    )
    survey = {
        "candidates": [
            {
                "prim_path": "/World/G2BT/GoldTrim",
                "display_color": [1.0, 0.623529, 0.25098],
            },
            {
                "prim_path": "/World/Other/GoldTrim",
                "display_color": [1.0, 0.623529, 0.25098],
            },
        ]
    }

    matches = rank_display_color_candidates(
        work_item_id="material:default:g2bt",
        task_request_path=tmp_path / "material_request.json",
        task_request_digest="request-digest",
        survey_path=tmp_path / "survey.json",
        survey=survey,
        appearance_index_path=tmp_path / "appearance.json",
        appearance_index=appearance_index,
        target_appearances={"1.000000,0.623529,0.250980": target},
        scope_paths=["/World/G2BT"],
        top_k=2,
    )

    assert [match.prim_path for match in matches.matches] == ["/World/G2BT/GoldTrim"]
    assert [
        candidate.material_name for candidate in matches.matches[0].nearest_materials
    ] == ["Gold Matte", "Plastic Orange"]
    assert matches.matches[0].nearest_materials[0].delta_e_76 < 3.0
    assert matches.task_request_digest == "request-digest"

    root_matches = rank_display_color_candidates(
        work_item_id="material:default:g2bt",
        task_request_path=tmp_path / "material_request.json",
        task_request_digest="request-digest",
        survey_path=tmp_path / "survey.json",
        survey=survey,
        appearance_index_path=tmp_path / "appearance.json",
        appearance_index=appearance_index,
        target_appearances={"1.000000,0.623529,0.250980": target},
        scope_paths=["/"],
        top_k=1,
    )
    assert [match.prim_path for match in root_matches.matches] == [
        "/World/G2BT/GoldTrim",
        "/World/Other/GoldTrim",
    ]
