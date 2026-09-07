# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for Geometry in the durable composed asset workflow."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from content_agent_workflows.asset_composition import (
    GEOMETRY_STAGE_ORDER,
    LEGACY_STAGE_ORDER,
    ArtifactBinding,
    AssetCompositionRun,
    AssetCompositionStateError,
    AssetGeometryRequest,
    AssetGeometryStageResult,
    AssetStageHandoff,
    begin_stage,
    complete_stage,
    execute_geometry_stage,
    fail_stage,
    load_verified_asset_request,
    load_verified_run,
    record_coordinator_evidence_review,
    record_coordinator_plan,
    recover_stage,
    stage_directory,
)
from content_agent_workflows.asset_composition import state as asset_state
from content_agent_workflows.asset_composition.cli import main as state_cli_main
from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.geometry import scene_ops
from content_agent_workflows.geometry import (
    workflow as geometry_workflow,
)
from content_agent_workflows.geometry.rendering import (
    GeometryRenderEvidence,
    GeometryRenderImageBinding,
    OvRTXRenderMode,
)
from content_agent_workflows.geometry.segmentation import GeometrySegmentationHandoff
from PIL import Image
from pydantic import ValidationError

import content_workflow_cli.asset_runner as asset_runner
from content_workflow_cli.asset_runner import AssetRunConfig, run_asset_workflow


def _write_source(path: Path, *, closed: bool = True) -> Path:
    counts = "[3, 3, 3, 3]" if closed else "[3]"
    indices = "[0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]" if closed else "[0, 2, 1]"
    path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Asset"
{{
    def Mesh "Body"
    {{
        uniform token subdivisionScheme = "none"
        float3[] extent = [(0, 0, 0), (0.05, 0.05, 0.05)]
        int[] faceVertexCounts = {counts}
        int[] faceVertexIndices = {indices}
        point3f[] points = [(0, 0, 0), (0.05, 0, 0), (0, 0.05, 0), (0, 0, 0.05)]
    }}
}}
""",
        encoding="utf-8",
    )
    return path


def test_parameter_variant_requires_a_new_immutable_source_revision() -> None:
    with pytest.raises(
        ValueError,
        match="variant reused the immutable baseline source revision",
    ):
        asset_state._require_new_variant_source_revision(
            baseline_revision="revision-1",
            variant_revision="revision-1",
        )

    asset_state._require_new_variant_source_revision(
        baseline_revision="revision-1",
        variant_revision="revision-2",
    )


def _start_run(
    tmp_path: Path,
    *,
    closed: bool = True,
    optimization_policy: str = "skip",
    render_evidence: bool = False,
    include_geometry_stage: bool = True,
    segmentation_run_dir: Path | None = None,
    segmentation_required: bool = False,
) -> Path:
    repository = tmp_path / "repo"
    (repository / "agentic" / ".agents" / "skills").mkdir(parents=True)
    source = _write_source(repository / "source.usda", closed=closed)
    joint = repository / "joint.yaml"
    joint.write_text("review_policy: all\n", encoding="utf-8")
    materials_usd = repository / "materials.usda"
    materials_usd.write_text("#usda 1.0\n", encoding="utf-8")
    materials = repository / "materials.yaml"
    materials.write_text(
        "library_path: materials.usda\nentries: []\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_asset_workflow(
        AssetRunConfig(
            repo_root=repository,
            usd_path=source,
            prompt="Prepare this rigid object for downstream asset authoring.",
            selected_mode="compatibility_fixed",
            joint_config=joint,
            materials_yaml=materials,
            materials_usd=materials_usd,
            output_dir=run_dir,
            run_id="geometry-composed",
            geometry_optimization_policy=optimization_policy,
            geometry_render_evidence=render_evidence,
            geometry_segmentation_run_dir=segmentation_run_dir,
            geometry_segmentation_required=segmentation_required,
            include_geometry_stage=include_geometry_stage,
            dry_run=True,
        )
    )
    return run_dir / "asset_run.json"


def _fake_segmentation_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "completed-segmentation"
    staged_source = run_dir / "inputs" / "source" / "source.usda"
    staged_source.parent.mkdir(parents=True)
    staged_source.write_text("#usda 1.0\n", encoding="utf-8")
    segmented_usd = run_dir / "final" / "x.usdc"
    segmented_usd.parent.mkdir(parents=True)
    segmented_usd.write_bytes(b"PXR-USDCfixture")
    render_dir = run_dir / "final" / "renders"
    render_dir.mkdir(parents=True)
    image_path = render_dir / "hero.png"
    image_path.write_bytes(b"render")
    camera_path = render_dir / "hero_camera.json"
    camera_payload = {"scene": str(segmented_usd)}
    camera_path.write_text(json.dumps(camera_payload) + "\n", encoding="utf-8")
    response_path = render_dir / "hero_response.json"
    response_payload = {"ok": True, "artifact": str(image_path)}
    response_path.write_text(json.dumps(response_payload) + "\n", encoding="utf-8")
    receipt_dir = render_dir / ".usd_cli_receipts"
    receipt_dir.mkdir()
    receipt_path = receipt_dir / "commands.jsonl"
    receipt_path.write_text(
        "".join(
            json.dumps(item, sort_keys=True) + "\n"
            for item in (
                {
                    "arguments": ["open", str(segmented_usd)],
                    "response": {"ok": True},
                    "artifact_bindings": [],
                },
                {
                    "arguments": ["render"],
                    "response": response_payload,
                    "artifact_bindings": [
                        {
                            "label": "rgb",
                            "path": str(image_path),
                            "sha256": file_sha256(image_path),
                            "size_bytes": image_path.stat().st_size,
                        }
                    ],
                },
            )
        ),
        encoding="utf-8",
    )
    receipt_stat = receipt_path.stat()
    checkpoint_path = receipt_dir / "commands.checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "receipt_device": receipt_stat.st_dev,
                "receipt_inode": receipt_stat.st_ino,
                "receipt_sha256": file_sha256(receipt_path),
                "receipt_size_bytes": receipt_stat.st_size,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    documents = {
        "request.json": {"run_dir": str(run_dir), "asset": str(staged_source)},
        "prepare/topology.json": {"source_asset": str(staged_source)},
        "segments.json": {"source": {"path": str(staged_source)}},
        "final/renders/render_manifest.json": {
            "scene": str(segmented_usd),
            "scene_tool": "usd-cli",
            "usd_cli_command_receipts": str(receipt_path),
            "usd_cli_command_receipts_sha256": file_sha256(receipt_path),
            "usd_cli_receipt_checkpoint": str(checkpoint_path),
            "usd_cli_receipt_checkpoint_sha256": file_sha256(checkpoint_path),
            "renders": [
                {
                    "camera": str(camera_path),
                    "camera_sha256": file_sha256(camera_path),
                    "response": str(response_path),
                    "response_sha256": file_sha256(response_path),
                }
            ],
        },
        "final/export_manifest.json": {
            "source_asset": str(staged_source),
            "segments": str(run_dir / "segments.json"),
            "segments_sha256": "0" * 64,
        },
    }
    for relative, payload in documents.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return run_dir


def test_segmentation_run_files_use_artifact_contract_string_order(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "segmentation-run"
    nested = run_dir / "fragments" / "fragment.usdc"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"nested")
    sibling = run_dir / "fragments.json"
    sibling.write_text("{}\n", encoding="utf-8")

    files = asset_runner._segmentation_run_files(run_dir)

    assert files == [sibling, nested]
    assert files == sorted(files, key=lambda path: str(path))


def _allow_fake_segmentation_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> list[Path]:
    from content_agent_workflows.geometry import segmentation

    consumed: list[Path] = []

    def consume(**kwargs: object) -> GeometrySegmentationHandoff:
        run_dir = Path(str(kwargs["run_dir"])).resolve()
        consumed.append(run_dir)
        return GeometrySegmentationHandoff(
            requested=True,
            required=bool(kwargs["required"]),
            outcome="conditional",
            run_dir=str(run_dir),
            producer_run_id=run_dir.name,
            source_asset_sha256="a" * 64,
            topology_digest="b" * 64,
            fragment_labels_sha256="c" * 64,
            face_labels_sha256="d" * 64,
            segmented_usd_sha256="e" * 64,
        )

    monkeypatch.setattr(segmentation, "consume_segmentation_handoff", consume)
    return consumed


def _plan_and_begin(state_path: Path) -> None:
    draft = state_path.parent / "raw" / "geometry-plan-draft.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-plan-draft.v1"
                ),
                "stage": "geometry",
                "objective": "Prepare and validate durable Geometry output.",
                "steps": [
                    {
                        "stage": "geometry",
                        "objective": "Run the typed Geometry executor.",
                        "acceptance_evidence": [
                            "Geometry stage result, manifest, and validation evidence"
                        ],
                        "may_revisit": True,
                    }
                ],
                "evidence_paths": [str(state_path.parent / "request.json")],
                "revision_reason": "Initial plan from frozen source and policy.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record_coordinator_plan(state_path, plan_path=draft, actor="test")
    begin_stage(state_path, "geometry", actor="test")


def _evidence_paths(result: AssetGeometryStageResult) -> list[str]:
    return [
        result.result_path,
        result.workflow_result.path,
        *(binding.path for binding in result.artifacts.values()),
    ]


def _load_stage_result(state_path: Path) -> AssetGeometryStageResult:
    path = stage_directory(state_path, "geometry") / "geometry_stage_result.json"
    return AssetGeometryStageResult.model_validate_json(path.read_bytes())


def _review(
    state_path: Path,
    result: AssetGeometryStageResult,
    *,
    decision: str,
    repair_scope: list[str] | None = None,
) -> None:
    draft = stage_directory(state_path, "geometry") / f"{decision}-review.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-review-draft.v1"
                ),
                "stage": "geometry",
                "output_asset_path": (
                    result.output_asset.path
                    if result.output_asset is not None
                    else None
                ),
                "evidence_paths": _evidence_paths(result),
                "findings": ["Reviewed the exact typed Geometry result."],
                "decision": decision,
                "target_stage": None,
                "decision_summary": f"{decision} the Geometry attempt.",
                "repair_scope": repair_scope or [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record_coordinator_evidence_review(state_path, review_path=draft, actor="test")


def test_geometry_acceptance_routes_exact_usdc_to_articulation(tmp_path: Path) -> None:
    state_path = _start_run(tmp_path)
    _plan_and_begin(state_path)

    assert (
        state_cli_main(
            ["execute-geometry", "--run-state", str(state_path), "--actor", "test"]
        )
        == 0
    )
    result = _load_stage_result(state_path)

    assert result.success
    assert result.handoff_ready in {"yes", "conditional"}
    assert result.output_asset is not None
    output = Path(result.output_asset.path)
    assert output.suffix == ".usdc"
    assert output.read_bytes()[:8] == b"PXR-USDC"
    _review(state_path, result, decision="accept")
    complete_stage(
        state_path,
        "geometry",
        output_asset=output,
        evidence_paths=_evidence_paths(result),
        summary="Accepted the typed durable Geometry handoff.",
        actor="test",
    )

    run = load_verified_run(state_path)
    assert run.current_stage == "articulation"
    assert run.stages["geometry"].status == "completed"
    assert run.stages["articulation"].input_asset == result.output_asset
    assert run.stages["articulation"].input_dependencies == result.output_dependencies


def test_geometry_optimizer_fallback_is_only_conditional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = _start_run(tmp_path, optimization_policy="preserve_correspondence")
    _plan_and_begin(state_path)

    def unavailable_optimizer(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("optimizer intentionally unavailable")

    monkeypatch.setattr(scene_ops.OptimizeUSDTask, "run", unavailable_optimizer)
    result = execute_geometry_stage(state_path, actor="test")

    assert result.success
    assert result.validation_status == "conditional"
    assert result.handoff_ready == "conditional"
    optimization = json.loads(
        Path(result.artifacts["optimization_metadata"].path).read_text(encoding="utf-8")
    )
    assert optimization["status"] == "optimization_unavailable"
    _review(state_path, result, decision="accept")


def test_geometry_dependency_failure_returns_a_typed_failed_stage_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = _start_run(tmp_path)
    _plan_and_begin(state_path)
    real_dependency_bindings = asset_state._dependency_bindings

    def reject_geometry_output(
        path: Path,
        *,
        label: str,
        required_root: Path | None = None,
    ) -> list[ArtifactBinding]:
        if Path(path).name == "geometry.usdc":
            raise AssetCompositionStateError(
                "geometry.usdc has unresolved USD dependencies: missing.png"
            )
        return real_dependency_bindings(
            path,
            label=label,
            required_root=required_root,
        )

    monkeypatch.setattr(asset_state, "_dependency_bindings", reject_geometry_output)

    result = execute_geometry_stage(state_path, actor="test")

    assert not result.success
    assert result.validation_status == "fail"
    assert result.handoff_ready == "no"
    assert result.error is not None
    assert "unresolved USD dependencies: missing.png" in result.error
    assert Path(result.result_path).is_file()


def test_geometry_acceptance_rejects_unconditional_optimizer_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = _start_run(tmp_path, optimization_policy="preserve_correspondence")
    _plan_and_begin(state_path)

    def unavailable_optimizer(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("optimizer intentionally unavailable")

    monkeypatch.setattr(scene_ops.OptimizeUSDTask, "run", unavailable_optimizer)
    result = execute_geometry_stage(state_path, actor="test")
    assert result.success

    def rewrite_binding(
        binding: ArtifactBinding,
        mutate: Callable[[dict[str, Any]], None],
    ) -> dict[str, str | int]:
        path = Path(binding.path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }

    workflow_binding = rewrite_binding(
        result.workflow_result,
        lambda payload: payload.update(
            validation_status="pass",
            handoff_ready="yes",
        ),
    )
    artifact_bindings = {
        label: binding.model_dump(mode="json")
        for label, binding in result.artifacts.items()
    }
    artifact_bindings["handoff_manifest"] = rewrite_binding(
        result.artifacts["handoff_manifest"],
        lambda payload: payload.update(handoff_ready="yes"),
    )
    artifact_bindings["validation_evidence"] = rewrite_binding(
        result.artifacts["validation_evidence"],
        lambda payload: payload["metadata"].update(handoff_ready="yes"),
    )
    artifact_bindings["evidence_bundle"] = rewrite_binding(
        result.artifacts["evidence_bundle"],
        lambda payload: payload.update(geometry_validation_status="pass"),
    )
    result_path = Path(result.result_path)
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    result_payload.update(
        workflow_result=workflow_binding,
        artifacts=artifact_bindings,
        validation_status="pass",
        handoff_ready="yes",
    )
    result_path.write_text(
        json.dumps(result_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="optimization fallback cannot be represented as an unconditional pass",
    ):
        _review(state_path, result, decision="accept")


def test_geometry_acceptance_binds_every_ovrtx_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = _start_run(tmp_path, render_evidence=True)
    _plan_and_begin(state_path)

    def fake_render_geometry_evidence(
        *,
        usd_path: Path,
        output_dir: Path,
        preset: str,
        backend: str,
        ovrtx_mode: OvRTXRenderMode,
        ovrtx_num_sensor_updates: int,
        **_kwargs: object,
    ) -> GeometryRenderEvidence:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "front.png"
        image = Image.new("RGB", (256, 256))
        image.putdata(
            [
                ((x * 2) % 256, (y * 2) % 256, ((x + y) * 3) % 256)
                for y in range(256)
                for x in range(256)
            ]
        )
        image.save(image_path)
        image.close()
        report_path = output_dir / f"geometry_render_evidence_{preset}.json"
        source_digest = file_sha256(usd_path)
        result = GeometryRenderEvidence(
            status="pass",
            backend=backend,
            renderer="ovrtx",
            renderer_identity_verified=True,
            renderer_identity_evidence={
                "source": "explicit_local_backend",
                "renderer": "ovrtx",
                "ready": True,
            },
            preset=preset,
            source_usd_path=str(usd_path.resolve()),
            source_usd_sha256=source_digest,
            source_usd_sha256_after_render=source_digest,
            ovrtx_render_mode=ovrtx_mode,
            ovrtx_num_sensor_updates=ovrtx_num_sensor_updates,
            active_aov="rgb",
            image_paths=[str(image_path.resolve())],
            image_bindings=[
                GeometryRenderImageBinding(
                    path=str(image_path.resolve()),
                    sha256=file_sha256(image_path),
                    role="view",
                    view="front",
                    source_usd_sha256=source_digest,
                    ovrtx_render_mode=ovrtx_mode,
                    ovrtx_num_sensor_updates=ovrtx_num_sensor_updates,
                    active_aov="rgb",
                )
            ],
            shared_render_status="completed",
            metadata={
                "backend": "ovrtx",
                "stage_preparation": [
                    {
                        "usd_path": str(usd_path.resolve()),
                        "usd_sha256": source_digest,
                    }
                ],
                "response_cameras": ["front"],
                "image_count": 1,
                "requested_ovrtx_render_mode": ovrtx_mode,
                "requested_ovrtx_num_sensor_updates": ovrtx_num_sensor_updates,
                "active_aov": "rgb",
                "executed_ovrtx_settings": {
                    "ovrtx_render_mode": ovrtx_mode,
                    "ovrtx_num_sensor_updates": ovrtx_num_sensor_updates,
                    "active_aov": "rgb",
                },
                "executed_ovrtx_settings_verified": True,
            },
            report_path=str(report_path.resolve()),
        )
        report_path.write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return result

    monkeypatch.setattr(
        geometry_workflow,
        "render_geometry_evidence",
        fake_render_geometry_evidence,
    )
    result = execute_geometry_stage(state_path, actor="test")

    assert result.success, result.error
    assert result.artifacts["render_image_000"].sha256 == file_sha256(
        Path(result.artifacts["render_image_000"].path)
    )
    _review(state_path, result, decision="accept")


def test_geometry_manifest_digest_drift_blocks_acceptance(tmp_path: Path) -> None:
    state_path = _start_run(tmp_path)
    _plan_and_begin(state_path)
    result = execute_geometry_stage(state_path, actor="test")
    manifest_path = Path(result.artifacts["handoff_manifest"].path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tampered_after_execution"] = True
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match="omitted"):
        _review(state_path, result, decision="accept")


def test_rejected_geometry_cannot_enter_articulation_and_can_refine(
    tmp_path: Path,
) -> None:
    state_path = _start_run(tmp_path, closed=False)
    _plan_and_begin(state_path)
    assert (
        state_cli_main(
            ["execute-geometry", "--run-state", str(state_path), "--actor", "test"]
        )
        == 1
    )
    result = _load_stage_result(state_path)

    assert not result.success
    assert result.handoff_ready == "no"
    with pytest.raises(AssetCompositionStateError, match="rejected"):
        _review(state_path, result, decision="accept")

    _review(
        state_path,
        result,
        decision="refine",
        repair_scope=["Repair the open mesh before another Geometry attempt."],
    )
    run = load_verified_run(state_path)
    assert run.current_stage == "geometry"
    assert run.stages["geometry"].status == "ready"
    assert stage_directory(state_path, "geometry").name == "02"
    assert run.stages["articulation"].status == "pending"


def test_interrupted_geometry_attempt_resumes_in_place(tmp_path: Path) -> None:
    state_path = _start_run(tmp_path)
    _plan_and_begin(state_path)
    first_attempt = stage_directory(state_path, "geometry")
    fail_stage(
        state_path,
        "geometry",
        reason="Executor process was interrupted before evidence review.",
        actor="test",
    )

    recover_stage(
        state_path,
        "geometry",
        reason="Executor environment restored.",
        actor="test",
    )
    _plan_and_begin(state_path)
    assert stage_directory(state_path, "geometry") == first_attempt

    result = execute_geometry_stage(state_path, actor="test")
    assert result.success
    assert result.stage_attempt == 1


def test_pre_geometry_v2_run_without_stage_order_remains_loadable(
    tmp_path: Path,
) -> None:
    state_path = _start_run(tmp_path, include_geometry_stage=False)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "content-agent-workflows.asset-composition-run.v2"
    payload.pop("stage_order")
    state_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    run = load_verified_run(state_path)

    assert run.stage_order == LEGACY_STAGE_ORDER
    assert run.current_stage == "articulation"
    assert "geometry" not in run.stages


@pytest.mark.parametrize(
    "schema_version",
    [
        "content-agent-workflows.asset-composition-run.v1",
        "content-agent-workflows.asset-composition-run.v2",
    ],
)
def test_legacy_run_serialization_preserves_original_strict_shape(
    tmp_path: Path,
    schema_version: str,
) -> None:
    state_path = _start_run(tmp_path, include_geometry_stage=False)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    run = AssetCompositionRun.model_validate(payload)

    serialized = run.model_dump(mode="json")

    assert "stage_order" not in serialized
    assert all(
        "input_readiness" not in stage for stage in serialized["stages"].values()
    )
    restored = AssetCompositionRun.model_validate(serialized)
    assert restored.stage_order == LEGACY_STAGE_ORDER


def test_v3_run_serialization_infers_geometry_order_and_strips_nested_readiness(
    tmp_path: Path,
) -> None:
    state_path = _start_run(tmp_path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "content-agent-workflows.asset-composition-run.v3"
    payload["stages"]["geometry"]["superseded_attempts"] = [
        {
            "archived_at": "2026-08-18T00:00:00Z",
            "reason_review": payload["request"],
            "status": "ready",
            "attempt_count": 1,
            "input_asset": payload["source_asset"],
            "input_readiness": "conditional",
        }
    ]
    run = AssetCompositionRun.model_validate(payload)

    serialized = run.model_dump(mode="json")

    assert "stage_order" not in serialized
    assert "input_readiness" not in serialized["stages"]["geometry"]
    assert (
        "input_readiness"
        not in serialized["stages"]["geometry"]["superseded_attempts"][0]
    )
    restored = AssetCompositionRun.model_validate(serialized)
    assert restored.stage_order == GEOMETRY_STAGE_ORDER


def test_stage_handoff_v2_versions_readiness_without_breaking_v1() -> None:
    binding = ArtifactBinding(path="/tmp/asset.usdc", sha256="a" * 64, size_bytes=1)
    legacy = AssetStageHandoff(
        schema_version="content-agent-workflows.asset-stage-handoff.v1",
        stage="geometry",
        input_asset=binding,
        output_asset=binding,
        evidence=[binding],
        readiness="conditional",
        summary="Legacy handoff.",
    )
    current = legacy.model_copy(
        update={"schema_version": "content-agent-workflows.asset-stage-handoff.v2"}
    )

    legacy_payload = legacy.model_dump(mode="json")
    current_payload = current.model_dump(mode="json")

    assert "readiness" not in legacy_payload
    assert current_payload["readiness"] == "conditional"
    assert AssetStageHandoff.model_validate(legacy_payload).readiness == "yes"
    assert AssetStageHandoff.model_validate(current_payload).readiness == "conditional"


def test_pre_cad_run_with_explicit_geometry_order_accepts_fixed_v5_request(
    tmp_path: Path,
) -> None:
    state_path = _start_run(tmp_path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "content-agent-workflows.asset-composition-run.v3"
    state_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    run = load_verified_run(state_path)
    request = load_verified_asset_request(state_path, run=run)

    assert run.stage_order == GEOMETRY_STAGE_ORDER
    assert run.current_stage == "geometry"
    assert request.schema_version == "content-agents.asset-composition-request.v5"
    assert request.geometry is not None
    assert request.cad_modeling is None


def test_geometry_policy_rejects_path_like_semantic_part_names() -> None:
    with pytest.raises(ValidationError, match="bounded plain names"):
        AssetGeometryRequest(
            target_profile="geometry-agent.static-visual-asset.v1",
            segmentation_required=True,
            segmentation_required_parts=["../drawer"],
        )


def test_composed_geometry_consumes_frozen_completed_segmentation_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_run = _fake_segmentation_run(tmp_path / "producer")
    consumed = _allow_fake_segmentation_handoff(monkeypatch)
    state_path = _start_run(
        tmp_path / "asset",
        segmentation_run_dir=external_run,
        segmentation_required=True,
    )
    request = load_verified_asset_request(state_path)
    assert request.geometry is not None
    assert request.geometry.segmentation_run is not None
    staged_run = Path(request.geometry.segmentation_run.run_dir)
    assert staged_run.is_relative_to(state_path.parent / "inputs")
    assert consumed == [external_run.resolve(), staged_run]
    staged_manifest = json.loads(
        (staged_run / "final/renders/render_manifest.json").read_text(encoding="utf-8")
    )
    staged_receipts = Path(staged_manifest["usd_cli_command_receipts"])
    staged_checkpoint_path = Path(staged_manifest["usd_cli_receipt_checkpoint"])
    staged_checkpoint = json.loads(staged_checkpoint_path.read_text(encoding="utf-8"))
    staged_receipt_stat = staged_receipts.stat()
    staged_receipt_records = [
        json.loads(line)
        for line in staged_receipts.read_text(encoding="utf-8").splitlines()
    ]
    assert staged_receipt_records[0]["arguments"] == [
        "open",
        str(staged_run / "final/x.usdc"),
    ]
    assert staged_receipt_records[1]["response"]["artifact"] == str(
        staged_run / "final/renders/hero.png"
    )
    assert staged_receipt_records[1]["artifact_bindings"] == [
        {
            "label": "rgb",
            "path": str(staged_run / "final/renders/hero.png"),
            "sha256": file_sha256(staged_run / "final/renders/hero.png"),
            "size_bytes": (staged_run / "final/renders/hero.png").stat().st_size,
        }
    ]
    assert staged_checkpoint["receipt_device"] == staged_receipt_stat.st_dev
    assert staged_checkpoint["receipt_inode"] == staged_receipt_stat.st_ino
    assert staged_checkpoint["receipt_sha256"] == file_sha256(staged_receipts)
    assert staged_manifest["usd_cli_command_receipts_sha256"] == file_sha256(
        staged_receipts
    )
    assert staged_manifest["usd_cli_receipt_checkpoint_sha256"] == file_sha256(
        staged_checkpoint_path
    )

    original_workflow = geometry_workflow.run_geometry_workflow
    captured: list[Path | None] = []

    def capture_staged_run(params: geometry_workflow.GeometryWorkflowInput):
        captured.append(params.segmentation_run_dir)
        return original_workflow(
            params.model_copy(
                update={
                    "segmentation_run_dir": None,
                    "segmentation_required": False,
                    "segmentation_required_parts": [],
                }
            )
        )

    monkeypatch.setattr(
        geometry_workflow,
        "run_geometry_workflow",
        capture_staged_run,
    )
    _plan_and_begin(state_path)
    result = execute_geometry_stage(state_path, actor="test")

    assert result.success
    assert captured == [staged_run]


def test_composed_geometry_resume_rejects_staged_segmentation_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_run = _fake_segmentation_run(tmp_path / "producer")
    _allow_fake_segmentation_handoff(monkeypatch)
    state_path = _start_run(
        tmp_path / "asset",
        segmentation_run_dir=external_run,
        segmentation_required=True,
    )
    request = load_verified_asset_request(state_path)
    assert request.geometry is not None
    assert request.geometry.segmentation_run is not None
    staged_segments = Path(request.geometry.segmentation_run.run_dir) / "segments.json"
    staged_segments.write_text("{}\n", encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match="identity changed"):
        load_verified_asset_request(state_path)


def test_segmentation_staging_enforces_streaming_byte_budget(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"four")
    destination = tmp_path / "destination.bin"

    with pytest.raises(ValueError, match="staged byte limit"):
        asset_runner._copy_stable_regular_file(
            source,
            destination,
            maximum_bytes=3,
        )

    assert not destination.exists()
