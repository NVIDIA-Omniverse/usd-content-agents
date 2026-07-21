# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Workflow 3 material collection."""

from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from content_agent_workflows.asset_task_processing.contracts import (
    AssetTaskInventory,
    AssetTaskResult,
    AssetTaskResultsIndex,
    AssetTaskWorkItem,
    ProcessingPhaseResult,
    ResultIndexEntry,
    ResultMapping,
    ResultProvenance,
    TaskCatalog,
    TaskSpec,
)
from content_agent_workflows.asset_task_processing.material_task import (
    MaterialAssignmentDecision,
    MaterialCandidateEvidence,
    MaterialDecisionPatch,
    MaterialSurvey,
)
from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
    seal_phase_result,
)
from content_agent_workflows.scene_collection import (
    CollectionRequest,
    ProjectedMaterialBinding,
    prepare_collection,
    run_collection,
)
from content_agent_workflows.scene_collection.collector import (
    CollectionRuntimeError,
    _authoring_target,
    _map_member_candidates,
    _surface_candidates_under,
    _topology_report,
    _validate_material_output,
)
from content_agent_workflows.scene_decomposition import (
    DecomposedAsset,
    ManifestCatalog,
    ManifestCatalogEntry,
    SceneDecompositionManifest,
    SceneInstanceGroup,
)


def _define_triangle(stage: Usd.Stage, path: str) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])


def _write_source(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/ProtoA")
    UsdGeom.Xform.Define(stage, "/ProtoB")
    _define_triangle(stage, "/ProtoA/Part")
    _define_triangle(stage, "/ProtoB/Part")
    for prototype in ("/ProtoA", "/ProtoB"):
        hidden_path = f"{prototype}/Hidden"
        _define_triangle(stage, hidden_path)
        UsdGeom.Imageable(stage.GetPrimAtPath(hidden_path)).CreateVisibilityAttr().Set(
            UsdGeom.Tokens.invisible
        )
    for name, prototype in (("A", "/ProtoA"), ("B", "/ProtoB")):
        prim = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        prim.GetReferences().AddInternalReference(prototype)
        prim.SetInstanceable(True)
    stage.GetRootLayer().Save()
    return path


def _write_material_library(root: Path) -> tuple[Path, Path]:
    usd_path = root / "materials.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    material = UsdShade.Material.Define(stage, "/World/Looks/Plastic_Red")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Plastic_Red/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.8, 0.02, 0.02)
    )
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    stage.GetRootLayer().Save()
    yaml_path = root / "materials.yaml"
    yaml_path.write_text(
        """materials:
  library_path: materials.usda
  entries:
    - name: Plastic Red
      binding: /World/Looks/Plastic_Red
""",
        encoding="utf-8",
    )
    return yaml_path, usd_path


def test_authoring_target_preserves_composed_reference_path(tmp_path: Path) -> None:
    asset_path = tmp_path / "referenced_asset.usda"
    asset_stage = Usd.Stage.CreateNew(str(asset_path))
    asset_root = UsdGeom.Xform.Define(asset_stage, "/Asset")
    asset_stage.SetDefaultPrim(asset_root.GetPrim())
    _define_triangle(asset_stage, "/Asset/Part")
    asset_stage.GetRootLayer().Save()

    source_path = tmp_path / "source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    world = UsdGeom.Xform.Define(source_stage, "/World")
    source_stage.SetDefaultPrim(world.GetPrim())
    referenced = UsdGeom.Xform.Define(source_stage, "/World/Referenced").GetPrim()
    referenced.GetReferences().AddReference(str(asset_path), "/Asset")
    source_stage.GetRootLayer().Save()

    target_path = "/World/Referenced/Part"
    assert source_stage.GetRootLayer().GetPrimAtPath(target_path) is None
    assert source_stage.GetPrimAtPath(target_path).IsValid()
    assert _authoring_target(source_stage, target_path) == target_path


def test_material_collection_projects_to_separate_instance_prototypes(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source.usda")
    material_yaml, _material_usd = _write_material_library(tmp_path)
    processing_dir = tmp_path / "02-asset-tasks"
    decomposition_dir = tmp_path / "01-decomposition"
    processing_dir.mkdir()
    decomposition_dir.mkdir()

    manifest_path = decomposition_dir / "scene_manifest.json"
    manifest = SceneDecompositionManifest(
        scene_id="scene",
        original_usd_path=str(source),
        assets=[
            DecomposedAsset(
                asset_id="asset_a",
                label="A",
                original_root_path="/World/A",
                working_usd_path=str(source),
                working_root_path="/World/A",
                source_path_prefixes=["/World/A"],
                processable=True,
                status="extracted",
            ),
            DecomposedAsset(
                asset_id="asset_waived",
                label="Waived",
                original_root_path="/World/Waived",
                working_usd_path=str(source),
                working_root_path="/World/Waived",
                source_path_prefixes=["/World/Waived"],
                processable=True,
                status="extracted",
            ),
        ],
        instance_groups=[
            SceneInstanceGroup(
                group_id="group",
                label="group",
                instance_count=2,
                member_paths=["/World/B"],
                representative_asset_id="asset_a",
            )
        ],
    )
    atomic_write_json(manifest_path, manifest)
    manifest_catalog_path = decomposition_dir / "manifest_catalog.json"
    atomic_write_json(
        manifest_catalog_path,
        ManifestCatalog(
            original_usd_path=str(source),
            source_identity_digest="source-digest",
            structural_analysis_id="analysis",
            manifests=[
                ManifestCatalogEntry(
                    manifest_id="material-view",
                    intent="material_processing",
                    path=str(manifest_path),
                    manifest_digest=file_sha256(manifest_path),
                )
            ],
        ),
    )

    request_path = processing_dir / "material_request.json"
    atomic_write_json(
        request_path,
        {
            "schema_version": "content-agent-workflows.material-task-request.v2",
            "domain": "material",
            "additional_instructions": (
                "Use white structural frames and transparent safety glass."
            ),
        },
    )
    request_digest = file_sha256(request_path)
    state_path = processing_dir / "state.json"
    atomic_write_json(state_path, {"work_items": []})
    plan_pointer_path = processing_dir / "plan.json"
    atomic_write_json(plan_pointer_path, {"current_revision": 1})
    ledger_path = processing_dir / "ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    task_catalog_path = processing_dir / "task_catalog.json"
    atomic_write_json(
        task_catalog_path,
        TaskCatalog(
            tasks=[
                TaskSpec(
                    task_id="material",
                    domain="material",
                    skill="content-workflow-material",
                    manifest_id="material-view",
                    request_path=str(request_path),
                    validator="material",
                    collector="material",
                )
            ]
        ),
    )
    work_item_id = "material:material-view:asset_a"
    waived_work_item_id = "material:material-view:asset_waived"
    inventory_path = processing_dir / "asset_task_inventory.json"
    atomic_write_json(
        inventory_path,
        AssetTaskInventory(
            input_digest="decomposition-digest",
            task_request_digests={"material": request_digest},
            work_items=[
                AssetTaskWorkItem(
                    work_item_id=work_item_id,
                    manifest_id="material-view",
                    asset_id="asset_a",
                    asset_label="A",
                    task_id="material",
                    original_root_path="/World/A",
                    working_usd_path=str(source),
                    working_root_path="/World/A",
                    source_path_prefixes=["/World/A"],
                ),
                AssetTaskWorkItem(
                    work_item_id=waived_work_item_id,
                    manifest_id="material-view",
                    asset_id="asset_waived",
                    asset_label="Waived",
                    task_id="material",
                    original_root_path="/World/Waived",
                    working_usd_path=str(source),
                    working_root_path="/World/Waived",
                    source_path_prefixes=["/World/Waived"],
                ),
            ],
        ),
    )
    survey_path = processing_dir / "survey.json"
    atomic_write_json(
        survey_path,
        MaterialSurvey(
            work_item_id=work_item_id,
            asset_label="A",
            source_usd=str(source),
            original_root_path="/World/A",
            visibility_policy="visible_only",
            candidates=[
                MaterialCandidateEvidence(
                    prim_path="/World/A/Part",
                    prim_type="Mesh",
                    mesh_path="/World/A/Part",
                    face_count=1,
                )
            ],
        ),
    )
    decision_path = processing_dir / "decision.json"
    atomic_write_json(
        decision_path,
        MaterialDecisionPatch(
            work_item_id=work_item_id,
            source_usd=str(source),
            material_library_yaml=str(material_yaml),
            material_library_path=str(tmp_path / "materials.usda"),
            task_request_digest=request_digest,
            assignments=[
                MaterialAssignmentDecision(
                    target_prim_path="/World/A/Part",
                    covered_candidate_paths=["/World/A/Part"],
                    material_name="Plastic Red",
                    rationale="Test assignment",
                    confidence=1.0,
                )
            ],
            evidence_summary="Test evidence",
            confidence=1.0,
        ),
    )
    result_path = processing_dir / "result.json"
    atomic_write_json(
        result_path,
        AssetTaskResult(
            task_id="material",
            domain="material",
            manifest_id="material-view",
            asset_id="asset_a",
            original_root_path="/World/A",
            working_usd_path=str(source),
            domain_outputs={
                "decision_path": str(decision_path),
                "survey_path": str(survey_path),
            },
            mapping=ResultMapping(),
            provenance=ResultProvenance(
                agent_plan_revision=1,
                task_request_digest=request_digest,
            ),
        ),
    )
    results_index_path = processing_dir / "asset_task_results_index.json"
    atomic_write_json(
        results_index_path,
        AssetTaskResultsIndex(
            entries=[
                ResultIndexEntry(
                    work_item_id=work_item_id,
                    status="completed",
                    result_path=str(result_path),
                ),
                ResultIndexEntry(
                    work_item_id=waived_work_item_id,
                    status="waived",
                ),
            ]
        ),
    )
    processing_result_path = processing_dir / "processing_result.json"
    processing_result = seal_phase_result(
        ProcessingPhaseResult(
            success=True,
            input_digest="decomposition-digest",
            task_catalog_path=str(task_catalog_path),
            manifest_catalog_path=str(manifest_catalog_path),
            asset_task_inventory_path=str(inventory_path),
            work_item_state_path=str(state_path),
            agent_plan_pointer_path=str(plan_pointer_path),
            decision_ledger_path=str(ledger_path),
            results_index_path=str(results_index_path),
            task_request_digests={"material": request_digest},
            required_work_item_count=2,
            completed_required_count=1,
            artifact_paths=[
                str(decision_path),
                str(inventory_path),
                str(ledger_path),
                str(manifest_catalog_path),
                str(manifest_path),
                str(plan_pointer_path),
                str(request_path),
                str(result_path),
                str(results_index_path),
                str(state_path),
                str(survey_path),
                str(task_catalog_path),
            ],
            completion_policy_satisfied=True,
        ),
        processing_result_path,
    )
    assert processing_result.output_digest is not None

    collection_dir = tmp_path / "03-collection"
    result = run_collection(
        CollectionRequest(
            source_scene=str(source),
            processing_result_path=str(processing_result_path),
            manifest_catalog_path=str(manifest_catalog_path),
            task_catalog_path=str(task_catalog_path),
            asset_task_inventory_path=str(inventory_path),
            results_index_path=str(results_index_path),
            output_dir=str(collection_dir),
            input_digest=processing_result.output_digest,
            requested_domains=["material"],
            material_library_yaml=str(material_yaml),
        )
    )
    assert result.success
    assert len(result.artifact_paths) == len(set(result.artifact_paths))
    composed = Usd.Stage.Open(result.final_output_paths[0], Usd.Stage.LoadNone)
    assert composed is not None
    for path in ("/World/A/Part", "/World/B/Part"):
        material, _relationship = UsdShade.MaterialBindingAPI(
            composed.GetPrimAtPath(path)
        ).ComputeBoundMaterial()
        assert str(material.GetPath()) == "/World/Looks/Plastic_Red"
    for path in ("/World/A/Hidden", "/World/B/Hidden"):
        material, _relationship = UsdShade.MaterialBindingAPI(
            composed.GetPrimAtPath(path)
        ).ComputeBoundMaterial()
        assert not material

    material_layer = Sdf.Layer.FindOrOpen(
        str(collection_dir / "domains" / "material" / "material_layer.usda")
    )
    assert material_layer is not None
    assert material_layer.GetPrimAtPath("/ProtoA/Part") is not None
    assert material_layer.GetPrimAtPath("/ProtoB/Part") is not None
    report = load_json(
        collection_dir / "domains" / "material" / "collection_report.json"
    )
    assert report["task_request_digest"] == request_digest
    assert report["covered_member_count"] == 2
    assert report["propagated_member_count"] == 1
    topology = load_json(collection_dir / "composition" / "topology_report.json")
    assert topology["passed"]
    collection_input = load_json(collection_dir / "collection_input_index.json")
    assert collection_input["task_request_digests"] == {"material": request_digest}
    assert any(
        artifact["role"] == "source_scene" and artifact["sha256"] == file_sha256(source)
        for artifact in collection_input["artifacts"]
    )
    assert any(
        artifact["role"] == "task_request:material"
        and artifact["sha256"] == request_digest
        for artifact in collection_input["artifacts"]
    )
    assert any(
        artifact["role"].startswith("processing_artifact:")
        and artifact["path"] == str(decision_path.resolve())
        for artifact in collection_input["artifacts"]
    )
    assert any(
        artifact["role"].startswith("processing_artifact:")
        and artifact["path"] == str(survey_path.resolve())
        for artifact in collection_input["artifacts"]
    )


def test_prepare_collection_rejects_tampered_processing_result_seal(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source.usda")
    processing_dir = tmp_path / "02-asset-tasks"
    decomposition_dir = tmp_path / "01-decomposition"
    collection_dir = tmp_path / "03-collection"
    processing_dir.mkdir()
    decomposition_dir.mkdir()
    request_path = processing_dir / "material_request.json"
    atomic_write_json(request_path, {"task": "material"})
    manifest_catalog_path = decomposition_dir / "manifest_catalog.json"
    atomic_write_json(
        manifest_catalog_path,
        ManifestCatalog(
            original_usd_path=str(source),
            source_identity_digest="source-digest",
            structural_analysis_id="analysis",
            manifests=[],
        ),
    )
    task_catalog_path = processing_dir / "task_catalog.json"
    atomic_write_json(task_catalog_path, TaskCatalog(tasks=[]))
    inventory_path = processing_dir / "asset_task_inventory.json"
    atomic_write_json(
        inventory_path,
        AssetTaskInventory(
            input_digest="decomposition-digest",
            task_request_digests={},
            work_items=[],
        ),
    )
    results_index_path = processing_dir / "asset_task_results_index.json"
    atomic_write_json(results_index_path, AssetTaskResultsIndex(entries=[]))
    state_path = processing_dir / "state.json"
    atomic_write_json(state_path, {"work_items": []})
    plan_pointer_path = processing_dir / "plan.json"
    atomic_write_json(plan_pointer_path, {"current_revision": 1})
    ledger_path = processing_dir / "ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    processing_result_path = processing_dir / "processing_result.json"
    sealed = seal_phase_result(
        ProcessingPhaseResult(
            success=True,
            input_digest="decomposition-digest",
            task_catalog_path=str(task_catalog_path),
            manifest_catalog_path=str(manifest_catalog_path),
            asset_task_inventory_path=str(inventory_path),
            work_item_state_path=str(state_path),
            agent_plan_pointer_path=str(plan_pointer_path),
            decision_ledger_path=str(ledger_path),
            results_index_path=str(results_index_path),
            required_work_item_count=0,
            completed_required_count=0,
            artifact_paths=[
                str(inventory_path),
                str(ledger_path),
                str(manifest_catalog_path),
                str(plan_pointer_path),
                str(request_path),
                str(results_index_path),
                str(state_path),
                str(task_catalog_path),
            ],
            completion_policy_satisfied=True,
        ),
        processing_result_path,
    )
    assert sealed.output_digest is not None
    payload = load_json(processing_result_path)
    payload["required_work_item_count"] = 1
    atomic_write_json(processing_result_path, payload)

    with pytest.raises(CollectionRuntimeError, match="seal"):
        prepare_collection(
            CollectionRequest(
                source_scene=str(source),
                processing_result_path=str(processing_result_path),
                manifest_catalog_path=str(manifest_catalog_path),
                task_catalog_path=str(task_catalog_path),
                asset_task_inventory_path=str(inventory_path),
                results_index_path=str(results_index_path),
                output_dir=str(collection_dir),
                input_digest=sealed.output_digest,
                requested_domains=[],
            )
        )


def test_surface_candidates_see_loaded_payload_contents(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.usda"
    payload_stage = Usd.Stage.CreateNew(str(payload_path))
    payload_root = UsdGeom.Xform.Define(payload_stage, "/Asset")
    payload_stage.SetDefaultPrim(payload_root.GetPrim())
    _define_triangle(payload_stage, "/Asset/Part")
    payload_stage.GetRootLayer().Save()

    source_path = tmp_path / "payload_source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    world = UsdGeom.Xform.Define(source_stage, "/World")
    source_stage.SetDefaultPrim(world.GetPrim())
    asset = UsdGeom.Xform.Define(source_stage, "/World/A").GetPrim()
    asset.GetPayloads().AddPayload(str(payload_path), "/Asset")
    source_stage.GetRootLayer().Save()

    unloaded = Usd.Stage.Open(str(source_path), Usd.Stage.LoadNone)
    loaded = Usd.Stage.Open(str(source_path), Usd.Stage.LoadAll)

    assert _surface_candidates_under(unloaded, "/World/A") == []
    assert [
        candidate["prim_path"]
        for candidate in _surface_candidates_under(loaded, "/World/A")
    ] == ["/World/A/Part"]


def test_validate_material_output_loads_payload_targets(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.usda"
    payload_stage = Usd.Stage.CreateNew(str(payload_path))
    payload_root = UsdGeom.Xform.Define(payload_stage, "/Asset")
    payload_stage.SetDefaultPrim(payload_root.GetPrim())
    _define_triangle(payload_stage, "/Asset/Part")
    payload_stage.GetRootLayer().Save()

    source_path = tmp_path / "payload_source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    world = UsdGeom.Xform.Define(source_stage, "/World")
    source_stage.SetDefaultPrim(world.GetPrim())
    asset = UsdGeom.Xform.Define(source_stage, "/World/A").GetPrim()
    asset.GetPayloads().AddPayload(str(payload_path), "/Asset")
    source_stage.GetRootLayer().Save()

    material_layer_path = tmp_path / "materials_layer.usda"
    material_stage = Usd.Stage.CreateNew(str(material_layer_path))
    material = UsdShade.Material.Define(material_stage, "/World/Looks/Plastic_Red")
    target = material_stage.OverridePrim("/World/A/Part")
    UsdShade.MaterialBindingAPI.Apply(target).Bind(material)
    material_stage.GetRootLayer().Save()

    composed_path = tmp_path / "composed.usda"
    composed_layer = Sdf.Layer.CreateNew(str(composed_path))
    composed_layer.subLayerPaths = [str(material_layer_path), str(source_path)]
    composed_layer.defaultPrim = "World"
    composed_layer.Save()

    validation = _validate_material_output(
        material_layer_path=material_layer_path,
        composed_scene_path=composed_path,
        projected=[
            ProjectedMaterialBinding(
                work_item_id="material:material-view:asset_a",
                representative_asset_id="asset_a",
                member_asset_id="asset_a",
                representative_root_path="/World/A",
                member_root_path="/World/A",
                decision_target_path="/World/A/Part",
                source_candidate_path="/World/A/Part",
                instance_target_path="/World/A/Part",
                authoring_target_path="/World/A/Part",
                material_name="Plastic Red",
                propagation_basis="explicit",
                mapping_method="exact_relative",
            )
        ],
        harmonized={"/World/A/Part": "Plastic Red"},
        material_paths={"Plastic Red": "/World/Looks/Plastic_Red"},
    )

    assert validation["passed"] is True
    assert validation["validated_composed_binding_count"] == 1


def test_topology_report_counts_loaded_payload_contents(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.usda"
    payload_stage = Usd.Stage.CreateNew(str(payload_path))
    payload_root = UsdGeom.Xform.Define(payload_stage, "/Asset")
    payload_stage.SetDefaultPrim(payload_root.GetPrim())
    _define_triangle(payload_stage, "/Asset/Part")
    payload_stage.GetRootLayer().Save()

    source_path = tmp_path / "payload_source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    world = UsdGeom.Xform.Define(source_stage, "/World")
    source_stage.SetDefaultPrim(world.GetPrim())
    asset = UsdGeom.Xform.Define(source_stage, "/World/A").GetPrim()
    asset.GetPayloads().AddPayload(str(payload_path), "/Asset")
    source_stage.GetRootLayer().Save()

    report = _topology_report(source_path, source_path)

    assert report["passed"] is True
    assert report["source_counts"]["mesh_count"] == 1
    assert report["composed_counts"]["mesh_count"] == 1


def test_member_candidate_mapping_rejects_order_only_structural_match(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "ambiguous.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    _define_triangle(stage, "/Rep/Alpha")
    _define_triangle(stage, "/Rep/Beta")
    _define_triangle(stage, "/Member/Gamma")
    _define_triangle(stage, "/Member/Delta")
    stage.GetRootLayer().Save()

    source_candidates = [
        MaterialCandidateEvidence(
            prim_path="/Rep/Alpha",
            prim_type="Mesh",
            mesh_path="/Rep/Alpha",
            face_count=1,
        ),
        MaterialCandidateEvidence(
            prim_path="/Rep/Beta",
            prim_type="Mesh",
            mesh_path="/Rep/Beta",
            face_count=1,
        ),
    ]

    with pytest.raises(CollectionRuntimeError, match="Could not project"):
        _map_member_candidates(stage, "/Rep", "/Member", source_candidates)
