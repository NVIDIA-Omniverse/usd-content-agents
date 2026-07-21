# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Workflow 2 runtime state and the material adapter."""

from __future__ import annotations

import hashlib
import json
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
    MaterialAssignmentDecision,
    MaterialDecisionPatch,
    _safe_path_component,
    match_work_item_display_colors,
    run_material_work_item,
    survey_material_inventory,
    survey_usd_material_candidates,
    survey_work_item,
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
) -> tuple[Path, Path, str, str]:
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
                    original_root_path="/World",
                    working_usd_path=str(source),
                    working_root_path="/World",
                )
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
        "Process the one material representative.\n", encoding="utf-8"
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


@pytest.mark.parametrize("blank_render", [False, True])
def test_material_adapter_surveys_applies_and_commits(
    tmp_path: Path,
    monkeypatch,
    blank_render: bool,
) -> None:
    library_path = _write_mesh(tmp_path / "materials.usda")
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
            "respect_existing_material_bindings": True,
        },
    )
    work_item_id = "material:default:asset_001"
    survey, survey_path = survey_work_item(run_dir, work_item_id)
    assert survey_path.is_file()
    assert [candidate.prim_path for candidate in survey.candidates] == ["/World/Mesh"]

    decision_path = run_dir / "material_decision.json"
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
                    rationale="Matches the supplied green surface evidence.",
                    confidence=0.9,
                )
            ],
            evidence_summary="Reference shows a green surface.",
            confidence=0.9,
        ),
    )

    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task.wait_until_healthy",
        lambda *args, **kwargs: None,
    )
    created_sessions: list[dict[str, object]] = []

    def fake_create_session(
        _workbench_url: str, payload: dict[str, object]
    ) -> dict[str, str]:
        created_sessions.append(payload)
        return {"session_id": "session-1"}

    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task.create_session",
        fake_create_session,
    )
    applied_commands: list[dict[str, object]] = []

    def fake_apply_command(
        _workbench_url: str,
        _session_id: str,
        command: str,
        payload: dict[str, object],
    ) -> dict[str, str]:
        applied_commands.append({"command": command, "payload": payload})
        return {"status": "applied"}

    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task.apply_command",
        fake_apply_command,
    )
    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task.get_material_assignments",
        lambda *args, **kwargs: {"assignments": [{"prim_path": "/World/Mesh"}]},
    )

    def fake_post_json(_url: str, payload: dict[str, object]) -> dict[str, object]:
        output_path = Path(str(payload["output_usd_path"]))
        _write_mesh(output_path)
        return {"status": "applied"}

    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task.post_json",
        fake_post_json,
    )
    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task.close_session",
        lambda *args, **kwargs: None,
    )
    rendered: dict[str, object] = {}

    def fake_render_view(**kwargs: object) -> dict[str, str]:
        rendered.update(kwargs)
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "final_oblique.png"
        image = Image.new("RGB", (64, 64), (194, 194, 194))
        if not blank_render:
            image.paste((20, 40, 80), (16, 16, 48, 48))
        image.save(image_path)
        return {"image_path": str(image_path)}

    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task.render_view",
        fake_render_view,
    )

    if blank_render:
        with pytest.raises(AssetTaskRuntimeError, match="render is blank"):
            run_material_work_item(
                run_dir,
                work_item_id,
                decision_path=decision_path,
                workbench_url="http://workbench.test",
                render=True,
            )
        validation = load_json(
            run_dir
            / "assets"
            / "default"
            / "asset_001"
            / "tasks"
            / "material"
            / "validation.json"
        )
        assert validation["passed"] is False
        assert processing_status(run_dir)["status_counts"] == {"failed": 1}
        assert applied_commands[0]["command"] == "material_override"
        payload = applied_commands[0]["payload"]
        assert isinstance(payload, dict)
        assert payload["unbind_existing"] is False
        assert isinstance(payload["material"], dict)
        return

    result = run_material_work_item(
        run_dir,
        work_item_id,
        decision_path=decision_path,
        workbench_url="http://workbench.test",
        render=True,
    )
    assert result.status == "completed"
    assert result.provenance.task_request_digest == request_digest
    assert Path(result.domain_outputs["preview_layer_path"]).is_file()
    assert created_sessions[0]["clear_materials"] is False
    assert applied_commands[0]["command"] == "material_override"
    payload = applied_commands[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["unbind_existing"] is False
    assert isinstance(payload["material"], dict)
    assert rendered["render_quality"] == "inspection"
    assert processing_status(run_dir)["status_counts"] == {"completed": 1}


def test_material_adapter_enforces_frozen_material_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_library_path = _write_mesh(tmp_path / "expected_materials.usda")
    expected_yaml = tmp_path / "expected_materials.yaml"
    expected_yaml.write_text(
        """library_path: expected_materials.usda
entries:
  - name: Plastic Green
    binding: /World/Looks/Plastic_Green
""",
        encoding="utf-8",
    )
    other_library_path = _write_mesh(tmp_path / "other_materials.usda")
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

    workbench_calls: list[str] = []
    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task.wait_until_healthy",
        lambda *args, **kwargs: workbench_calls.append("wait_until_healthy"),
    )
    monkeypatch.setattr(
        "content_agent_workflows.asset_task_processing.material_task.create_session",
        lambda *args, **kwargs: workbench_calls.append("create_session"),
    )

    with pytest.raises(AssetTaskRuntimeError, match="frozen material task request"):
        run_material_work_item(
            run_dir,
            work_item_id,
            decision_path=decision_path,
            workbench_url="http://workbench.test",
            render=False,
        )
    assert workbench_calls == []
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
            workbench_url="http://workbench.test",
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

    monkeypatch.setattr(
        material_appearance, "wait_until_healthy", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        material_appearance,
        "create_session",
        lambda *_a, **_k: {"session_id": "session"},
    )
    monkeypatch.setattr(material_appearance, "apply_command", lambda *_a, **_k: {})
    monkeypatch.setattr(material_appearance, "close_session", lambda *_a, **_k: None)

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
        workbench_url="http://workbench",
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
        workbench_url="http://workbench",
    )

    color_digest = hashlib.sha256(b"1.000000,0.000000,0.000000").hexdigest()[:12]
    assert len(rendered_names) == 2
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
