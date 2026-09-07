# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from content_agent_workflows.articulation.asset_leaf_adapter import (
    ArticulationFocusedLeafResult,
)
from content_agent_workflows.asset_composition import release_leaf_adapters
from content_agent_workflows.asset_composition.catalog_adapters import (
    CanonicalOvrtxEvidenceLeafInvocation,
)
from content_agent_workflows.asset_composition.models import (
    ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
    ArtifactBinding,
    AssetCompositionRun,
    AssetCoordinatorState,
    AssetExecutionGraph,
    AssetExecutionNode,
    AssetLeafReceipt,
    AssetLeafState,
)
from content_agent_workflows.asset_composition.release_leaf_adapters import (
    ARTICULATION_PUBLISH_LEAF_ID,
    COMBINED_RESULT_LEAF_ID,
    FINAL_OVRTX_EVIDENCE_LEAF_ID,
    MATERIAL_ASSIGNMENT_LEAF_ID,
    PHYSICS_APPLY_LEAF_ID,
    PHYSICS_INSPECTION_LEAF_ID,
    PORTABLE_PACKAGE_LEAF_ID,
    SIMREADY_CONFORMANCE_LEAF_ID,
    SIMREADY_VALIDATION_LEAF_ID,
    TEXTURE_PUBLISH_LEAF_ID,
    CombinedAssetResultReceipt,
    CombinedResultLeafInvocation,
    ComposedAssetLeafResult,
    MaterialAssignmentLeafInvocation,
    PhysicsApplyLeafInvocation,
    PhysicsApplyLeafReceipt,
    PhysicsComponentInspectionReceipt,
    PhysicsInspectionLeafInvocation,
    PortablePackageLeafInvocation,
    PortablePackageReceipt,
    SimReadyConformanceLeafInvocation,
    SimReadyConformanceLeafReceipt,
    SimReadyValidationLeafInvocation,
    artifact_binding,
    release_composed_asset_leaf_runtime_bundle,
    write_canonical_json,
)
from content_agent_workflows.common.validation_evidence import (
    physics_validation_evidence,
)
from content_agent_workflows.physics import (
    PhysicsApplyWorkflowInput,
    PhysicsApplyWorkflowResult,
    PhysicsComponent,
)
from content_agent_workflows.simready import (
    SimReadyConformanceInput,
    SimReadyConformanceReport,
    SimReadyValidationInput,
    SimReadyValidationReport,
)
from content_agent_workflows.texture.asset_leaf_adapter import TextureFocusedLeafResult
from content_agent_workflows.validation import (
    CanonicalVisualEvidencePayload,
    execution_artifact_binding,
)
from pydantic import BaseModel

from content_workflow_cli import composed_leaf_cli


def _package_runtime():  # type: ignore[no-untyped-def]
    return next(
        binding
        for binding in release_composed_asset_leaf_runtime_bundle().bindings
        if binding.descriptor.leaf_id == PORTABLE_PACKAGE_LEAF_ID
    )


def _runtime(leaf_id: str):  # type: ignore[no-untyped-def]
    return next(
        binding
        for binding in release_composed_asset_leaf_runtime_bundle().bindings
        if binding.descriptor.leaf_id == leaf_id
    )


class _Blob(BaseModel):
    value: Any


def _write_blob(path: Path, value: Any) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_canonical_json(path, _Blob(value=value))


def _write_predecessor_receipt(
    *,
    run_root: Path,
    index: int,
    leaf_id: str,
    invocation: Any,
    native_terminal: Any,
    readbacks: list[Any],
    result_factory: Callable[[Any], BaseModel] | None = None,
) -> Any:
    attempt = run_root / "leaves" / f"{index:03d}-{leaf_id}" / "attempts" / "01"
    attempt.mkdir(parents=True, exist_ok=True)
    invocation_path = attempt / "invocation.json"
    invocation_binding = (
        write_canonical_json(invocation_path, invocation)
        if isinstance(invocation, BaseModel)
        else _write_blob(invocation_path, invocation)
    )
    result_model = (
        result_factory(invocation_binding)
        if result_factory is not None
        else _Blob(value={"status": "pass"})
    )
    result = write_canonical_json(attempt / "result.json", result_model)
    projection = _write_blob(attempt / "leaf_projection.json", {"status": "pass"})
    receipt = AssetLeafReceipt(
        run_id="combined-fixture",
        run_revision=index,
        graph_digest="1" * 64,
        leaf_catalog_digest="2" * 64,
        sole_coordinator_identity_digest="3" * 64,
        leaf_id=leaf_id,
        descriptor_digest="4" * 64,
        invocation_schema_digest="5" * 64,
        result_schema_digest="6" * 64,
        projection_schema_digest="7" * 64,
        projector_id=(
            "asset.projector.final-ovrtx-evidence.v1"
            if leaf_id == FINAL_OVRTX_EVIDENCE_LEAF_ID
            else f"asset.projector.{leaf_id}"
        ),
        projector_digest="8" * 64,
        required_artifact_categories=[],
        requirement="required",
        depends_on=[],
        dependency_receipts={},
        attempt=1,
        invocation=invocation_binding,
        result=result,
        projection=projection,
        native_terminal_receipt=native_terminal,
        evidence=[native_terminal],
        saved_stage_readbacks=readbacks,
        native_disposition="passed",
        native_status="pass",
        started_at="2026-08-25T00:00:00+00:00",
        finished_at="2026-08-25T00:00:00+00:00",
        duration_ms=0,
        summary=f"{leaf_id} passed.",
        actor="asset-coordinator",
    )
    return write_canonical_json(attempt / "leaf_receipt.json", receipt)


def _run_package_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):  # type: ignore[no-untyped-def]
    source = tmp_path / "source.usdz"
    source.write_bytes(b"fixture-usdz-bytes")
    attempt = tmp_path / "run" / "leaves" / "package" / "attempts" / "01"
    attempt.mkdir(parents=True)
    invocation = PortablePackageLeafInvocation(
        attempt_root=str(attempt),
        source=artifact_binding(source),
        source_dependencies=(),
        output_asset_path=str(attempt / "final.usdz"),
    )
    invocation_path = attempt / "invocation.json"
    invocation_binding = write_canonical_json(invocation_path, invocation)
    monkeypatch.setattr(composed_leaf_cli, "bind_usd_dependency_closure", lambda _p: [])
    package_members = ("keyboard-root.usdc", "0/keycap.png")
    monkeypatch.setattr(
        composed_leaf_cli,
        "validate_usdz_package_layout",
        lambda _p: package_members,
    )
    monkeypatch.setattr(
        release_leaf_adapters,
        "validate_usdz_package_layout",
        lambda _p: package_members,
    )
    result = composed_leaf_cli.run_portable_package_leaf(invocation_path)
    result_binding = write_canonical_json(attempt / "result.json", result)
    return source, invocation, invocation_binding, result, result_binding


def test_invocation_binding_and_payload_share_one_pinned_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    source = tmp_path / "source.usdz"
    source.write_bytes(b"source")
    original = PortablePackageLeafInvocation(
        attempt_root=str(attempt),
        source=artifact_binding(source),
        source_dependencies=(),
        output_asset_path=str(attempt / "original.usdz"),
    )
    replacement = original.model_copy(
        update={"output_asset_path": str(attempt / "replacement.usdz")}
    )
    invocation_path = attempt / "invocation.json"
    write_canonical_json(invocation_path, original)
    real_reader = composed_leaf_cli.read_artifact_binding_and_bytes

    def swap_after_pinned_read(path: str | Path) -> tuple[Any, bytes]:
        binding, payload_bytes = real_reader(path)
        invocation_path.unlink()
        invocation_path.write_text(
            replacement.model_dump_json(),
            encoding="utf-8",
        )
        return binding, payload_bytes

    monkeypatch.setattr(
        composed_leaf_cli,
        "read_artifact_binding_and_bytes",
        swap_after_pinned_read,
    )

    _root, binding, loaded = composed_leaf_cli._load_invocation(
        invocation_path,
        PortablePackageLeafInvocation,
    )

    assert loaded == original
    assert binding.sha256 != artifact_binding(invocation_path).sha256


def test_native_workflow_params_create_once_then_resume_exact_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    native_root = tmp_path / "native"
    native_root.mkdir()
    params = PhysicsApplyWorkflowInput(
        usd_path=source,
        output_dir=native_root,
        output_usd_path=native_root / "physics.usda",
        resume=True,
    )

    first = composed_leaf_cli._native_workflow_params(
        params,
        run_dir=native_root,
    )
    assert first.resume is False

    request_path = native_root / "request.json"
    request_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="request without its manifest"):
        composed_leaf_cli._native_workflow_params(params, run_dir=native_root)

    request_path.unlink()
    (native_root / "workflow_run_manifest.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    resumed = composed_leaf_cli._native_workflow_params(
        params,
        run_dir=native_root,
    )
    assert resumed.resume is True


def test_material_retry_rejects_a_changed_composed_invocation(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    native_root = attempt / "native"
    native_root.mkdir(parents=True)
    invocation_path = attempt / "invocation.json"
    invocation_path.write_text('{"version": 1}\n', encoding="utf-8")
    original = artifact_binding(invocation_path)

    seal = composed_leaf_cli._seal_material_invocation(native_root, original)
    assert Path(seal.path).is_file()

    invocation_path.write_text('{"version": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical composed leaf artifact changed"):
        composed_leaf_cli._seal_material_invocation(
            native_root,
            artifact_binding(invocation_path),
        )

    legacy_root = attempt / "legacy-native"
    legacy_root.mkdir()
    (legacy_root / "coordinator_preparation.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lacks its exact composed invocation"):
        composed_leaf_cli._seal_material_invocation(legacy_root, original)


def test_portable_package_leaf_copies_exact_usdz_and_projects_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, invocation, invocation_binding, result, result_binding = (
        _run_package_fixture(tmp_path, monkeypatch)
    )
    projection = _package_runtime().project(
        invocation,
        result,
        invocation_artifact=invocation_binding,
        result_artifact=result_binding,
    )

    assert result.status == "passed"
    assert result.output_asset is not None
    assert Path(result.output_asset.path).read_bytes() == source.read_bytes()
    package_receipt = PortablePackageReceipt.model_validate_json(
        Path(result.native_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert package_receipt.root_layer_name == "keyboard-root.usdc"
    assert projection.payload.native_disposition == "passed"
    assert projection.payload.saved_stage_readbacks[-1] == result.output_asset


def test_package_projector_rejects_cross_leaf_output_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, invocation, invocation_binding, result, result_binding = (
        _run_package_fixture(tmp_path, monkeypatch)
    )
    receipt_path = Path(result.native_terminal_receipt.path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["package"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="identity changed"):
        _package_runtime().project(
            invocation,
            result,
            invocation_artifact=invocation_binding,
            result_artifact=result_binding,
        )


def test_package_projector_rejects_a_false_copied_usdz_root_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, invocation, invocation_binding, result, _result_binding = (
        _run_package_fixture(tmp_path, monkeypatch)
    )
    receipt = PortablePackageReceipt.model_validate_json(
        Path(result.native_terminal_receipt.path).read_text(encoding="utf-8")
    )
    drifted_receipt = write_canonical_json(
        Path(invocation.attempt_root) / "false_root_receipt.json",
        receipt.model_copy(update={"root_layer_name": "asset.usdc"}),
    )
    assert result.output_asset is not None
    drifted_result = result.model_copy(
        update={
            "native_terminal_receipt": drifted_receipt,
            "evidence": (drifted_receipt,),
            "saved_stage_readbacks": (drifted_receipt, result.output_asset),
        }
    )
    drifted_result_binding = write_canonical_json(
        Path(invocation.attempt_root) / "false_root_result.json",
        drifted_result,
    )

    with pytest.raises(ValueError, match="portable package receipt identity"):
        _package_runtime().project(
            invocation,
            drifted_result,
            invocation_artifact=invocation_binding,
            result_artifact=drifted_result_binding,
        )


def test_package_projector_rejects_output_at_another_attempt_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, invocation, invocation_binding, result, _result_binding = (
        _run_package_fixture(tmp_path, monkeypatch)
    )
    attempt = Path(invocation.attempt_root)
    alternate = attempt / "alternate.usdz"
    alternate.write_bytes(source.read_bytes())
    alternate_binding = artifact_binding(alternate)
    alternate_receipt = write_canonical_json(
        attempt / "alternate_receipt.json",
        PortablePackageReceipt(
            source=invocation.source,
            source_dependencies=invocation.source_dependencies,
            package=alternate_binding,
            package_dependencies=(),
            root_layer_name=invocation.root_layer_name,
        ),
    )
    drifted = result.model_copy(
        update={
            "native_terminal_receipt": alternate_receipt,
            "evidence": (alternate_receipt,),
            "saved_stage_readbacks": (alternate_receipt, alternate_binding),
            "output_asset": alternate_binding,
        }
    )
    drifted_binding = write_canonical_json(attempt / "drifted_result.json", drifted)

    with pytest.raises(ValueError, match="portable package receipt identity"):
        _package_runtime().project(
            invocation,
            drifted,
            invocation_artifact=invocation_binding,
            result_artifact=drifted_binding,
        )


def test_physics_leaves_bind_typed_inspection_and_validated_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "texture.usdz"
    source.write_bytes(b"textured-asset")
    source_binding = artifact_binding(source)
    component = PhysicsComponent(
        component_id="component_001",
        body_root_path="/Keyboard",
        visual_evidence_paths=["/Keyboard"],
    )
    monkeypatch.setattr(
        composed_leaf_cli,
        "inspect_physics_components",
        lambda _path: [component],
    )

    inspection_attempt = tmp_path / "inspection" / "attempts" / "01"
    inspection_attempt.mkdir(parents=True)
    inspection_invocation = PhysicsInspectionLeafInvocation(
        attempt_root=str(inspection_attempt),
        source=source_binding,
        component_catalog_path=str(inspection_attempt / "components.json"),
    )
    inspection_invocation_path = inspection_attempt / "invocation.json"
    inspection_invocation_binding = write_canonical_json(
        inspection_invocation_path,
        inspection_invocation,
    )
    inspection_result = composed_leaf_cli.run_physics_inspection_leaf(
        inspection_invocation_path
    )
    inspection_result_binding = write_canonical_json(
        inspection_attempt / "result.json",
        inspection_result,
    )
    inspection_projection = _runtime(PHYSICS_INSPECTION_LEAF_ID).project(
        inspection_invocation,
        inspection_result,
        invocation_artifact=inspection_invocation_binding,
        result_artifact=inspection_result_binding,
    )
    inspection_receipt = PhysicsComponentInspectionReceipt.model_validate_json(
        Path(inspection_result.native_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert inspection_receipt.source == source_binding
    assert len(inspection_receipt.components) == 1
    assert inspection_projection.payload.native_disposition == "passed"

    apply_attempt = tmp_path / "apply" / "attempts" / "01"
    apply_attempt.mkdir(parents=True)
    decision = _write_blob(apply_attempt / "decision.json", {"accepted": True})
    output = apply_attempt / "physics.usdz"
    resume_modes: list[bool] = []

    def apply(params: PhysicsApplyWorkflowInput) -> PhysicsApplyWorkflowResult:
        resume_modes.append(params.resume)
        output.write_bytes(b"runtime-validated-physics")
        output_binding = artifact_binding(output)
        validation_path = apply_attempt / "native" / "validation_evidence.json"
        validation_path.parent.mkdir(exist_ok=True)
        validation = physics_validation_evidence(
            asset=output_binding.path,
            target_runtime="newton",
            physics_properties_status="pass",
            runtime_loadability_status="pass",
            no_explosions_status="pass",
            metadata={"asset_sha256": output_binding.sha256},
        )
        validation_path.write_text(
            validation.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return PhysicsApplyWorkflowResult(
            success=True,
            asset=source_binding.path,
            output_dir=params.output_dir.as_posix(),
            physics_usd_path=str(output),
            decision_patch_path=decision.path,
            validation_evidence_path=str(validation_path),
            validation_status="pass",
        )

    monkeypatch.setattr(composed_leaf_cli, "run_physics_apply_workflow", apply)
    apply_invocation = PhysicsApplyLeafInvocation(
        attempt_root=str(apply_attempt),
        source=source_binding,
        component_catalog=inspection_result.native_terminal_receipt,
        decision_patch=decision,
        params=PhysicsApplyWorkflowInput(
            usd_path=source,
            inspection_asset_sha256=source_binding.sha256,
            output_dir=apply_attempt / "native",
            output_usd_path=output,
            decision_patch_path=Path(decision.path),
            resume=True,
        ),
    )
    apply_invocation_path = apply_attempt / "invocation.json"
    apply_invocation_binding = write_canonical_json(
        apply_invocation_path,
        apply_invocation,
    )
    apply_result = composed_leaf_cli.run_physics_apply_leaf(apply_invocation_path)
    apply_result_binding = write_canonical_json(
        apply_attempt / "result.json",
        apply_result,
    )
    apply_projection = _runtime(PHYSICS_APPLY_LEAF_ID).project(
        apply_invocation,
        apply_result,
        invocation_artifact=apply_invocation_binding,
        result_artifact=apply_result_binding,
    )
    apply_receipt = PhysicsApplyLeafReceipt.model_validate_json(
        Path(apply_result.native_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert apply_receipt.output_asset == apply_result.output_asset
    assert apply_receipt.validation_evidence is not None
    assert apply_projection.payload.native_disposition == "passed"
    assert resume_modes == [False]

    validation_path = Path(apply_receipt.validation_evidence.path)
    tampered_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    tampered_validation["asset"] = source_binding.path
    validation_path.write_text(
        json.dumps(tampered_validation, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered_validation_binding = artifact_binding(validation_path)
    tampered_receipt = apply_receipt.model_copy(
        update={"validation_evidence": tampered_validation_binding}
    )
    tampered_receipt_binding = write_canonical_json(
        apply_attempt / "tampered_terminal.json",
        tampered_receipt,
    )
    tampered_result = apply_result.model_copy(
        update={
            "native_terminal_receipt": tampered_receipt_binding,
            "evidence": tuple(
                tampered_receipt_binding
                if item == apply_result.native_terminal_receipt
                else tampered_validation_binding
                if item == apply_receipt.validation_evidence
                else item
                for item in apply_result.evidence
            ),
            "saved_stage_readbacks": tuple(
                tampered_receipt_binding
                if item == apply_result.native_terminal_receipt
                else item
                for item in apply_result.saved_stage_readbacks
            ),
        }
    )
    tampered_result_binding = write_canonical_json(
        apply_attempt / "tampered_result.json",
        tampered_result,
    )
    with pytest.raises(
        ValueError,
        match="validation evidence does not bind the exact clean output",
    ):
        _runtime(PHYSICS_APPLY_LEAF_ID).project(
            apply_invocation,
            tampered_result,
            invocation_artifact=apply_invocation_binding,
            result_artifact=tampered_result_binding,
        )


def test_physics_pass_rejects_missing_populated_validation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    source_binding = artifact_binding(source)
    monkeypatch.setattr(
        composed_leaf_cli,
        "inspect_physics_components",
        lambda _path: [],
    )

    attempt = tmp_path / "apply" / "attempts" / "01"
    attempt.mkdir(parents=True)
    component_catalog = write_canonical_json(
        attempt / "components.json",
        PhysicsComponentInspectionReceipt(source=source_binding, components=()),
    )
    decision = _write_blob(attempt / "decision.json", {"accepted": True})
    output = attempt / "physics.usda"
    missing_validation = attempt / "native" / "validation_evidence.json"

    def apply(params: PhysicsApplyWorkflowInput) -> PhysicsApplyWorkflowResult:
        output.write_text("#usda 1.0\n", encoding="utf-8")
        return PhysicsApplyWorkflowResult(
            success=True,
            asset=source_binding.path,
            output_dir=params.output_dir.as_posix(),
            physics_usd_path=str(output),
            decision_patch_path=decision.path,
            validation_evidence_path=str(missing_validation),
            validation_status="pass",
        )

    monkeypatch.setattr(composed_leaf_cli, "run_physics_apply_workflow", apply)
    invocation = PhysicsApplyLeafInvocation(
        attempt_root=str(attempt),
        source=source_binding,
        component_catalog=component_catalog,
        decision_patch=decision,
        params=PhysicsApplyWorkflowInput(
            usd_path=source,
            inspection_asset_sha256=source_binding.sha256,
            output_dir=attempt / "native",
            output_usd_path=output,
            decision_patch_path=Path(decision.path),
            resume=True,
        ),
    )
    invocation_path = attempt / "invocation.json"
    write_canonical_json(invocation_path, invocation)

    with pytest.raises(
        ValueError,
        match="validation_evidence_path is not a safe regular file",
    ):
        composed_leaf_cli.run_physics_apply_leaf(invocation_path)


def test_physics_native_result_rejects_lexical_path_escape(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    (attempt / "native").mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    disguised_outside = attempt / "native" / ".." / ".." / outside.name
    result = PhysicsApplyWorkflowResult(
        success=False,
        asset=str(attempt / "source.usda"),
        output_dir=str(attempt / "native"),
        validation_evidence_path=str(disguised_outside),
        validation_status="failed",
    )

    with pytest.raises(
        ValueError,
        match="validation_evidence_path escaped its composed attempt",
    ):
        composed_leaf_cli._physics_result_artifacts(result, root=attempt)


def test_native_output_preflight_rejects_a_late_parent_symlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usdz"
    source.write_bytes(b"source")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    destination = attempt / "future" / "component_catalog.json"
    invocation_path = attempt / "invocation.json"
    write_canonical_json(
        invocation_path,
        PhysicsInspectionLeafInvocation(
            attempt_root=str(attempt),
            source=artifact_binding(source),
            component_catalog_path=str(destination),
        ),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    destination.parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe directory component"):
        composed_leaf_cli.run_physics_inspection_leaf(invocation_path)

    assert not (outside / destination.name).exists()


def test_simready_conformance_leaf_binds_exact_source_report_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "physics.usdz"
    source.write_bytes(b"physics-asset")
    source_binding = artifact_binding(source)
    attempt = tmp_path / "conformance" / "attempts" / "01"
    attempt.mkdir(parents=True)
    native_root = attempt / "native"
    native_output = native_root / "conformed.usdz"
    native_report = native_root / "simready_conformance.json"
    resume_modes: list[bool] = []

    def conform(params: SimReadyConformanceInput) -> SimReadyConformanceReport:
        resume_modes.append(params.resume)
        native_root.mkdir(exist_ok=True)
        native_output.write_bytes(b"conformed-asset")
        report = SimReadyConformanceReport(
            input_usd_path=source_binding.path,
            output_usd_path=str(native_output),
            output_dir=str(native_root),
            profile="SimReady Foundation",
            profile_version="1.0",
            passed=True,
            status="PASS",
            report_path=str(native_report),
        )
        native_report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report

    monkeypatch.setattr(
        composed_leaf_cli,
        "run_simready_profile_conformance",
        conform,
    )
    invocation = SimReadyConformanceLeafInvocation(
        attempt_root=str(attempt),
        source=source_binding,
        params=SimReadyConformanceInput(
            asset_path=source_binding.path,
            output_dir=str(native_root),
            report_path=str(native_report),
            resume=True,
        ),
    )
    invocation_path = attempt / "invocation.json"
    invocation_binding = write_canonical_json(invocation_path, invocation)
    result = composed_leaf_cli.run_simready_conformance_leaf(invocation_path)
    result_binding = write_canonical_json(attempt / "result.json", result)
    projection = _runtime(SIMREADY_CONFORMANCE_LEAF_ID).project(
        invocation,
        result,
        invocation_artifact=invocation_binding,
        result_artifact=result_binding,
    )
    receipt = SimReadyConformanceLeafReceipt.model_validate_json(
        Path(result.native_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert receipt.source == source_binding
    assert receipt.output_asset == result.output_asset
    assert projection.payload.native_disposition == "passed"
    assert resume_modes == [False]


@pytest.mark.parametrize(
    ("drift", "expected_error"),
    (
        ("input", "report input differs from source binding"),
        ("report", "report path differs from its invocation"),
    ),
)
def test_simready_conformance_rejects_mismatched_native_report_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    expected_error: str,
) -> None:
    source = tmp_path / "source.usdz"
    source.write_bytes(b"source")
    source_binding = artifact_binding(source)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    native_root = attempt / "native"
    report_path = native_root / "conformance.json"
    native_output = native_root / "conformed.usdz"

    def conform(_params: SimReadyConformanceInput) -> SimReadyConformanceReport:
        native_root.mkdir()
        native_output.write_bytes(b"conformed")
        report = SimReadyConformanceReport(
            input_usd_path=(
                str(tmp_path / "other.usdz")
                if drift == "input"
                else source_binding.path
            ),
            output_usd_path=str(native_output),
            output_dir=str(native_root),
            profile="SimReady Foundation",
            profile_version="1.0",
            passed=True,
            status="PASS",
            report_path=(
                str(native_root / "other.json")
                if drift == "report"
                else str(report_path)
            ),
        )
        Path(report.report_path or "").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        return report

    monkeypatch.setattr(
        composed_leaf_cli,
        "run_simready_profile_conformance",
        conform,
    )
    invocation = SimReadyConformanceLeafInvocation(
        attempt_root=str(attempt),
        source=source_binding,
        params=SimReadyConformanceInput(
            asset_path=source_binding.path,
            output_dir=str(native_root),
            report_path=str(report_path),
            resume=True,
        ),
    )
    invocation_path = attempt / "invocation.json"
    write_canonical_json(invocation_path, invocation)

    with pytest.raises(ValueError, match=expected_error):
        composed_leaf_cli.run_simready_conformance_leaf(invocation_path)


def test_simready_conformance_rejects_output_outside_attempt(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    outside = tmp_path / "outside.usda"
    outside.write_text("#usda 1.0\n", encoding="utf-8")
    report = SimReadyConformanceReport(
        input_usd_path=str(attempt / "source.usda"),
        output_usd_path=str(outside),
        output_dir=str(attempt),
        profile="SimReady Foundation",
        profile_version="1.0",
        passed=True,
        status="PASS",
    )

    with pytest.raises(ValueError, match="escaped its composed attempt"):
        composed_leaf_cli._publish_conformance_output(attempt, report)


def test_simready_validation_creates_before_resuming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usdz"
    source.write_bytes(b"source")
    attempt = tmp_path / "validation" / "attempts" / "01"
    attempt.mkdir(parents=True)
    report_path = attempt / "simready_validation.json"
    resume_modes: list[bool] = []

    def validate(params: SimReadyValidationInput) -> SimReadyValidationReport:
        resume_modes.append(params.resume)
        report = SimReadyValidationReport(
            asset_path=str(source),
            asset_sha256=artifact_binding(source).sha256,
            passed=True,
            status="PASS",
            profile_name="SimReady Foundation",
            profile_version="1.0",
            profile_target="default",
            report_path=str(report_path),
        )
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report

    monkeypatch.setattr(
        composed_leaf_cli,
        "run_simready_profile_validation",
        validate,
    )
    invocation = SimReadyValidationLeafInvocation(
        attempt_root=str(attempt),
        source=artifact_binding(source),
        params=SimReadyValidationInput(
            asset_path=str(source),
            report_path=str(report_path),
            resume=True,
        ),
    )
    invocation_path = attempt / "invocation.json"
    write_canonical_json(invocation_path, invocation)

    result = composed_leaf_cli.run_simready_validation_leaf(invocation_path)

    assert result.status == "passed"
    assert resume_modes == [False]


@pytest.mark.parametrize(
    ("drift", "expected_error"),
    (
        ("asset", "report asset differs from source binding"),
        ("digest", "report digest differs from source binding"),
        ("report", "report path differs from its invocation"),
    ),
)
def test_simready_validation_rejects_mismatched_native_report_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    expected_error: str,
) -> None:
    source = tmp_path / "source.usdz"
    source.write_bytes(b"source")
    source_binding = artifact_binding(source)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    report_path = attempt / "validation.json"

    def validate(_params: SimReadyValidationInput) -> SimReadyValidationReport:
        report = SimReadyValidationReport(
            asset_path=(
                str(tmp_path / "other.usdz") if drift == "asset" else str(source)
            ),
            asset_sha256=("0" * 64 if drift == "digest" else source_binding.sha256),
            passed=True,
            status="PASS",
            profile_name="SimReady Foundation",
            profile_version="1.0",
            profile_target="default",
            report_path=(
                str(attempt / "other.json") if drift == "report" else str(report_path)
            ),
        )
        Path(report.report_path or "").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        return report

    monkeypatch.setattr(
        composed_leaf_cli,
        "run_simready_profile_validation",
        validate,
    )
    invocation = SimReadyValidationLeafInvocation(
        attempt_root=str(attempt),
        source=source_binding,
        params=SimReadyValidationInput(
            asset_path=str(source),
            report_path=str(report_path),
            resume=True,
        ),
    )
    invocation_path = attempt / "invocation.json"
    write_canonical_json(invocation_path, invocation)

    with pytest.raises(ValueError, match=expected_error):
        composed_leaf_cli.run_simready_validation_leaf(invocation_path)


def test_material_failure_retains_corrupt_preparation_cleanup_evidence(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    invocation_path = attempt / "invocation.json"
    invocation_path.parent.mkdir(parents=True)
    invocation_path.write_text("{}\n", encoding="utf-8")
    preparation_path = attempt / "native" / "coordinator_preparation.json"
    preparation_path.parent.mkdir(parents=True)
    preparation_path.write_text("not-json\n", encoding="utf-8")

    result = composed_leaf_cli._failure_result(
        operation="material",
        invocation_path=invocation_path,
        error=RuntimeError("material failed"),
    )

    assert result.status == "failed"
    assert result.resource_claims == ()
    assert len(result.resource_release_receipts) == 1
    release = json.loads(
        Path(result.resource_release_receipts[0].path).read_text(encoding="utf-8")
    )
    assert release["status"] == "failed"
    assert "Cannot verify retained Material preparation" in release["error"]


@pytest.mark.parametrize(
    ("native_status", "expected_status"),
    (("pass", "passed"), ("conditional", "failed")),
)
def test_material_terminal_result_converts_native_materialized_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_status: str,
    expected_status: str,
) -> None:
    attempt = tmp_path / "attempt"
    native = attempt / "native"
    raw = native / "raw"
    raw.mkdir(parents=True)

    def fixture_file(path: Path, payload: bytes = b"fixture") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    source = fixture_file(tmp_path / "source.usdz", b"source")
    materials_yaml = fixture_file(tmp_path / "materials.yaml", b"materials")
    materials_usd = fixture_file(tmp_path / "materials.usda", b"#usda 1.0\n")
    request = fixture_file(native / "coordinator_request.json")
    packet = fixture_file(raw / "material_run_packet.json")
    visible = fixture_file(raw / "visible_candidate_prims.json")
    palette = fixture_file(raw / "material_palette.json")
    context = fixture_file(raw / "material_authoring_context.md")
    seed = fixture_file(raw / "material_assignment_seed.json")
    table = fixture_file(raw / "visible_candidate_table.tsv")
    initial_records = fixture_file(raw / "initial_render_records.json")
    checkpoint = fixture_file(raw / "material_receipt_checkpoint.json")
    decision = fixture_file(raw / "material_decision_patch.json")
    applied_decision = fixture_file(raw / "material_applied_decision_patch.json")
    policy = fixture_file(raw / "material_finalization_policy.json")
    materialized = fixture_file(raw / "materialized.usda", b"materialized")
    render = fixture_file(raw / "final.png", b"png")
    evidence = fixture_file(raw / "evidence.json")
    review = fixture_file(raw / "material_post_apply_review.json")
    output = fixture_file(raw / "published.usdz", b"published")
    fixture_file(raw / "material_session_release.json")

    def native_binding(path: Path) -> dict[str, object]:
        return artifact_binding(path).model_dump(mode="json")

    preparation = composed_leaf_cli.MaterialCoordinatorPreparation.model_validate(
        {
            "request_path": str(request),
            "request_sha256": artifact_binding(request).sha256,
            "packet_path": str(packet),
            "packet_binding": native_binding(packet),
            "session_id": "material-test-session",
            "visible_candidates_path": str(visible),
            "visible_candidates_binding": native_binding(visible),
            "palette_path": str(palette),
            "palette_binding": native_binding(palette),
            "authoring_context_path": str(context),
            "authoring_context_binding": native_binding(context),
            "assignment_seed_path": str(seed),
            "assignment_seed_binding": native_binding(seed),
            "candidate_table_path": str(table),
            "candidate_table_binding": native_binding(table),
            "initial_render_paths": [],
            "initial_render_bindings": [],
            "initial_render_records_binding": native_binding(initial_records),
            "receipt_checkpoint_binding": native_binding(checkpoint),
        }
    )
    (native / "coordinator_preparation.json").write_text(
        preparation.model_dump_json(), encoding="utf-8"
    )
    application = composed_leaf_cli.MaterialCoordinatorReviewRequired.model_validate(
        {
            "request": native_binding(request),
            "decision_patch": native_binding(applied_decision),
            "policy": native_binding(policy),
            "materialized_usd": native_binding(materialized),
            "receipt_checkpoint_binding": native_binding(checkpoint),
            "final_render_bindings": [native_binding(render)],
            "evidence": [native_binding(evidence)],
            "applied_source_prim_paths": [],
            "unresolved_issues": ["review required"],
        }
    )
    (raw / "material_application_receipt.json").write_text(
        application.model_dump_json(), encoding="utf-8"
    )
    result = composed_leaf_cli.MaterialCoordinatorResult.model_validate(
        {
            "status": native_status,
            "output_usd_path": str(output),
            "output_usd_sha256": artifact_binding(output).sha256,
            "request": native_binding(request),
            "decision_patch": native_binding(review),
            "evidence": [native_binding(evidence)],
            "unresolved_issues": (
                [] if native_status == "pass" else ["review remains conditional"]
            ),
        }
    )
    (native / "coordinator_result.json").write_text(
        result.model_dump_json(), encoding="utf-8"
    )

    invocation = MaterialAssignmentLeafInvocation(
        attempt_root=str(attempt),
        source=artifact_binding(source),
        repository_root=str(tmp_path),
        materials_yaml=artifact_binding(materials_yaml),
        materials_usd=artifact_binding(materials_usd),
        output_asset_path=str(output),
        decision_patch_path=str(decision),
        review_patch_path=str(review),
    )
    invocation_path = attempt / "invocation.json"
    invocation_binding = write_canonical_json(invocation_path, invocation)
    write_canonical_json(
        native / "composed_invocation_binding.json",
        invocation_binding,
    )
    monkeypatch.setattr(
        composed_leaf_cli,
        "bind_usd_dependency_closure",
        lambda _path: [],
    )

    composed_result = composed_leaf_cli.run_material_leaf(invocation_path)

    assert composed_result.status == expected_status
    assert all(
        isinstance(binding, ArtifactBinding)
        for binding in composed_result.saved_stage_readbacks
    )
    assert any(
        binding.path == str(materialized)
        for binding in composed_result.saved_stage_readbacks
    )


def test_physics_inspection_exception_projects_as_exact_failed_leaf(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    source = attempt / "source.usdz"
    source.write_bytes(b"source")
    invocation = PhysicsInspectionLeafInvocation(
        attempt_root=str(attempt),
        source=artifact_binding(source),
        component_catalog_path=str(attempt / "components.json"),
    )
    invocation_path = attempt / "invocation.json"
    invocation_binding = write_canonical_json(invocation_path, invocation)

    result = composed_leaf_cli._failure_result(
        operation="physics-inspect",
        invocation_path=invocation_path,
        error=RuntimeError("inspection failed"),
    )
    result_binding = write_canonical_json(attempt / "result.json", result)
    projection = _runtime(PHYSICS_INSPECTION_LEAF_ID).project(
        invocation,
        result,
        invocation_artifact=invocation_binding,
        result_artifact=result_binding,
    )

    assert projection.payload.native_disposition == "failed"
    assert projection.payload.error == "RuntimeError: inspection failed"


def _combined_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    graph_name: str = "execution_graph.json",
):  # type: ignore[no-untyped-def]
    run_root = tmp_path / "run"
    source = run_root / "source.usdz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source-package")
    source_binding = artifact_binding(source)

    def attempt(index: int, leaf_id: str) -> Path:
        path = run_root / "leaves" / f"{index:03d}-{leaf_id}" / "attempts" / "01"
        path.mkdir(parents=True, exist_ok=True)
        return path

    articulation_attempt = attempt(1, ARTICULATION_PUBLISH_LEAF_ID)
    articulated_asset = articulation_attempt / "articulated.usdz"
    articulated_asset.write_bytes(b"articulated-package")
    articulated_binding = artifact_binding(articulated_asset)
    articulation_native = _write_blob(
        articulation_attempt / "articulation_terminal.json",
        {"status": "pass"},
    )
    articulation_receipt = _write_predecessor_receipt(
        run_root=run_root,
        index=1,
        leaf_id=ARTICULATION_PUBLISH_LEAF_ID,
        invocation={"source": source_binding.model_dump(mode="json")},
        native_terminal=articulation_native,
        readbacks=[articulation_native],
        result_factory=lambda invocation: ArticulationFocusedLeafResult(
            leaf_id=ARTICULATION_PUBLISH_LEAF_ID,
            invocation=execution_artifact_binding(invocation.path),
            native_disposition="passed",
            native_status="completed",
            native_terminal_receipt=execution_artifact_binding(
                articulation_native.path
            ),
            evidence=(execution_artifact_binding(articulation_native.path),),
            saved_stage_readbacks=(
                execution_artifact_binding(articulation_native.path),
            ),
            output=execution_artifact_binding(articulated_asset),
        ),
    )

    material_attempt = attempt(2, MATERIAL_ASSIGNMENT_LEAF_ID)
    materials_yaml = run_root / "materials.yaml"
    materials_yaml.write_text("materials: []\n", encoding="utf-8")
    materials_usd = run_root / "materials.usda"
    materials_usd.write_text("#usda 1.0\n", encoding="utf-8")
    material_asset = material_attempt / "native" / "material.usdz"
    material_asset.parent.mkdir()
    material_asset.write_bytes(b"material-package")
    material_binding = artifact_binding(material_asset)
    material_native = _write_blob(
        material_attempt / "material_terminal.json",
        {"status": "pass"},
    )
    material_invocation = MaterialAssignmentLeafInvocation(
        attempt_root=str(material_attempt),
        source=articulated_binding,
        repository_root=str(tmp_path),
        materials_yaml=artifact_binding(materials_yaml),
        materials_usd=artifact_binding(materials_usd),
        output_asset_path=str(material_asset),
        decision_patch_path=str(
            material_attempt / "native" / "raw" / "material_decision_patch.json"
        ),
        review_patch_path=str(
            material_attempt / "native" / "raw" / "material_post_apply_review.json"
        ),
    )
    material_receipt = _write_predecessor_receipt(
        run_root=run_root,
        index=2,
        leaf_id=MATERIAL_ASSIGNMENT_LEAF_ID,
        invocation=material_invocation,
        native_terminal=material_native,
        readbacks=[material_native, material_binding],
        result_factory=lambda invocation: ComposedAssetLeafResult(
            leaf_id=MATERIAL_ASSIGNMENT_LEAF_ID,
            status="passed",
            native_status="pass",
            invocation=invocation,
            native_terminal_receipt=material_native,
            evidence=(material_native,),
            saved_stage_readbacks=(material_native, material_binding),
            output_asset=material_binding,
            summary="Material passed.",
        ),
    )

    texture_attempt = attempt(3, TEXTURE_PUBLISH_LEAF_ID)
    texture_asset = texture_attempt / "texture.usdz"
    texture_asset.write_bytes(b"texture-package")
    texture_binding = artifact_binding(texture_asset)
    texture_native = _write_blob(
        texture_attempt / "texture_terminal.json",
        {"status": "pass"},
    )
    texture_receipt = _write_predecessor_receipt(
        run_root=run_root,
        index=3,
        leaf_id=TEXTURE_PUBLISH_LEAF_ID,
        invocation={"source": material_binding.model_dump(mode="json")},
        native_terminal=texture_native,
        readbacks=[texture_native],
        result_factory=lambda invocation: TextureFocusedLeafResult(
            leaf_id=TEXTURE_PUBLISH_LEAF_ID,
            invocation=execution_artifact_binding(invocation.path),
            native_packet=execution_artifact_binding(texture_native.path),
            native_disposition="passed",
            source=execution_artifact_binding(material_asset),
            output=execution_artifact_binding(texture_asset),
            output_dependencies=(),
            evidence=(execution_artifact_binding(texture_native.path),),
            saved_stage_readbacks=(execution_artifact_binding(texture_native.path),),
            renderer_invoked=False,
        ),
    )

    inspection_attempt = attempt(4, PHYSICS_INSPECTION_LEAF_ID)
    inspection_native = write_canonical_json(
        inspection_attempt / "physics_components.json",
        PhysicsComponentInspectionReceipt(
            source=texture_binding,
            components=(),
        ),
    )
    inspection_invocation = PhysicsInspectionLeafInvocation(
        attempt_root=str(inspection_attempt),
        source=texture_binding,
        component_catalog_path=inspection_native.path,
    )
    inspection_receipt = _write_predecessor_receipt(
        run_root=run_root,
        index=4,
        leaf_id=PHYSICS_INSPECTION_LEAF_ID,
        invocation=inspection_invocation,
        native_terminal=inspection_native,
        readbacks=[inspection_native],
    )

    physics_attempt = attempt(5, PHYSICS_APPLY_LEAF_ID)
    physics_asset = physics_attempt / "physics.usdz"
    physics_asset.write_bytes(b"physics-package")
    physics_binding = artifact_binding(physics_asset)
    physics_decision = _write_blob(
        physics_attempt / "physics_decision.json",
        {"status": "accepted"},
    )
    physics_native = _write_blob(
        physics_attempt / "physics_terminal.json",
        {"status": "pass"},
    )
    physics_invocation = PhysicsApplyLeafInvocation(
        attempt_root=str(physics_attempt),
        source=texture_binding,
        component_catalog=inspection_native,
        decision_patch=physics_decision,
        params=PhysicsApplyWorkflowInput(
            usd_path=Path(texture_binding.path),
            inspection_asset_sha256=texture_binding.sha256,
            output_dir=physics_attempt / "native",
            output_usd_path=physics_asset,
            decision_patch_path=Path(physics_decision.path),
            resume=True,
        ),
    )
    physics_receipt = _write_predecessor_receipt(
        run_root=run_root,
        index=5,
        leaf_id=PHYSICS_APPLY_LEAF_ID,
        invocation=physics_invocation,
        native_terminal=physics_native,
        readbacks=[physics_native, physics_binding],
        result_factory=lambda invocation: ComposedAssetLeafResult(
            leaf_id=PHYSICS_APPLY_LEAF_ID,
            status="passed",
            native_status="pass",
            invocation=invocation,
            native_terminal_receipt=physics_native,
            evidence=(physics_native,),
            saved_stage_readbacks=(physics_native, physics_binding),
            output_asset=physics_binding,
            summary="Physics passed.",
        ),
    )

    conformance_attempt = attempt(6, SIMREADY_CONFORMANCE_LEAF_ID)
    conformed_asset = conformance_attempt / "conformed.usdz"
    conformed_asset.write_bytes(b"conformed-package")
    conformed_binding = artifact_binding(conformed_asset)
    conformance_native = _write_blob(
        conformance_attempt / "conformance_terminal.json",
        {"status": "pass"},
    )
    conformance_invocation = SimReadyConformanceLeafInvocation(
        attempt_root=str(conformance_attempt),
        source=physics_binding,
        params=SimReadyConformanceInput(
            asset_path=physics_binding.path,
            output_dir=str(conformance_attempt / "native"),
            report_path=str(
                conformance_attempt / "native" / "simready_conformance.json"
            ),
            resume=True,
        ),
    )
    conformance_receipt = _write_predecessor_receipt(
        run_root=run_root,
        index=6,
        leaf_id=SIMREADY_CONFORMANCE_LEAF_ID,
        invocation=conformance_invocation,
        native_terminal=conformance_native,
        readbacks=[conformance_native, conformed_binding],
        result_factory=lambda invocation: ComposedAssetLeafResult(
            leaf_id=SIMREADY_CONFORMANCE_LEAF_ID,
            status="passed",
            native_status="pass",
            invocation=invocation,
            native_terminal_receipt=conformance_native,
            evidence=(conformance_native,),
            saved_stage_readbacks=(conformance_native, conformed_binding),
            output_asset=conformed_binding,
            summary="SimReady conformance passed.",
        ),
    )

    package_attempt = attempt(7, PORTABLE_PACKAGE_LEAF_ID)
    final_asset = package_attempt / "final.usdz"
    final_asset.write_bytes(b"qualified-final-package")
    final_binding = artifact_binding(final_asset)
    package_native = write_canonical_json(
        package_attempt / "portable_package_receipt.json",
        PortablePackageReceipt(
            source=conformed_binding,
            source_dependencies=(),
            package=final_binding,
            package_dependencies=(),
            root_layer_name="asset.usdc",
        ),
    )
    package_graph_receipt = _write_predecessor_receipt(
        run_root=run_root,
        index=7,
        leaf_id=PORTABLE_PACKAGE_LEAF_ID,
        invocation=PortablePackageLeafInvocation(
            attempt_root=str(package_attempt),
            source=conformed_binding,
            source_dependencies=(),
            output_asset_path=str(final_asset),
        ),
        native_terminal=package_native,
        readbacks=[package_native, final_binding],
    )

    validation_attempt = attempt(8, SIMREADY_VALIDATION_LEAF_ID)
    validation_native = write_canonical_json(
        validation_attempt / "simready_validation.json",
        SimReadyValidationReport(
            asset_path=str(final_asset),
            asset_sha256=final_binding.sha256,
            passed=True,
            status="PASS",
            profile_name="SimReady Foundation",
            profile_version="1.0",
            profile_target="simulation",
        ),
    )
    validation_graph_receipt = _write_predecessor_receipt(
        run_root=run_root,
        index=8,
        leaf_id=SIMREADY_VALIDATION_LEAF_ID,
        invocation=SimReadyValidationLeafInvocation(
            attempt_root=str(validation_attempt),
            source=final_binding,
            params=SimReadyValidationInput(
                asset_path=str(final_asset),
                report_path=str(validation_native.path),
                resume=True,
            ),
        ),
        native_terminal=validation_native,
        readbacks=[validation_native],
    )

    ovrtx_attempt = attempt(9, FINAL_OVRTX_EVIDENCE_LEAF_ID)
    render_report = _write_blob(ovrtx_attempt / "render_report.json", {"ok": True})
    image = ovrtx_attempt / "view.png"
    image.write_bytes(b"ovrtx-render")
    response = _write_blob(ovrtx_attempt / "response.json", {"ok": True})
    camera = _write_blob(ovrtx_attempt / "camera.json", {"view": "+x+y+z"})
    command_receipt = _write_blob(
        ovrtx_attempt / "command_receipt.json", {"status": "pass"}
    )
    command_checkpoint = _write_blob(
        ovrtx_attempt / "command_checkpoint.json", {"status": "sealed"}
    )
    payload = CanonicalVisualEvidencePayload(
        source=execution_artifact_binding(source),
        post_mutation_output=execution_artifact_binding(final_asset),
        render_report=execution_artifact_binding(render_report.path),
        images=(execution_artifact_binding(image),),
        render_responses=(execution_artifact_binding(response.path),),
        camera_records=(execution_artifact_binding(camera.path),),
        usd_cli_command_receipt=execution_artifact_binding(command_receipt.path),
        usd_cli_receipt_checkpoint=execution_artifact_binding(command_checkpoint.path),
        usd_cli_source_revision="a" * 40,
        backend_alias="ovrtx",
        render_metadata={"renderer": "ovrtx"},
    )
    payload_binding = write_canonical_json(ovrtx_attempt / "payload.json", payload)
    ovrtx_native = _write_blob(
        ovrtx_attempt / "envelope.json", {"native_status": "pass"}
    )
    ovrtx_graph_receipt = _write_predecessor_receipt(
        run_root=run_root,
        index=9,
        leaf_id=FINAL_OVRTX_EVIDENCE_LEAF_ID,
        invocation=CanonicalOvrtxEvidenceLeafInvocation(
            output_dir=str(ovrtx_attempt),
            post_mutation_usd=str(final_asset),
            source_usd=str(source),
            backend="ovrtx",
            views=("+x+y+z",),
            image_width=1024,
            image_height=1024,
        ),
        native_terminal=ovrtx_native,
        readbacks=[payload_binding],
    )

    predecessor_receipts = {
        ARTICULATION_PUBLISH_LEAF_ID: articulation_receipt,
        FINAL_OVRTX_EVIDENCE_LEAF_ID: ovrtx_graph_receipt,
        PORTABLE_PACKAGE_LEAF_ID: package_graph_receipt,
        MATERIAL_ASSIGNMENT_LEAF_ID: material_receipt,
        PHYSICS_APPLY_LEAF_ID: physics_receipt,
        PHYSICS_INSPECTION_LEAF_ID: inspection_receipt,
        SIMREADY_CONFORMANCE_LEAF_ID: conformance_receipt,
        SIMREADY_VALIDATION_LEAF_ID: validation_graph_receipt,
        TEXTURE_PUBLISH_LEAF_ID: texture_receipt,
    }
    predecessor_ids = sorted(predecessor_receipts)
    graph = AssetExecutionGraph.create(
        sole_coordinator_identity_digest="3" * 64,
        prompt_digest="a" * 64,
        source_digest=source_binding.sha256,
        configuration_digest="b" * 64,
        reference_digest="c" * 64,
        leaf_catalog_digest="2" * 64,
        nodes=[
            *[
                AssetExecutionNode(
                    leaf_id=leaf_id,
                    requirement="required",
                    descriptor_digest="4" * 64,
                )
                for leaf_id in predecessor_ids
            ],
            AssetExecutionNode(
                leaf_id=COMBINED_RESULT_LEAF_ID,
                depends_on=predecessor_ids,
                requirement="required",
                descriptor_digest="4" * 64,
                terminal_output=True,
            ),
        ],
        omitted_leaf_ids=[],
    )
    graph_binding = write_canonical_json(run_root / graph_name, graph)
    rebound_receipts: dict[str, Any] = {}
    for leaf_id, binding in predecessor_receipts.items():
        receipt_path = Path(binding.path)
        receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_payload["graph_digest"] = graph.graph_digest
        receipt_path.write_text(
            json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rebound_receipts[leaf_id] = artifact_binding(receipt_path)
    predecessor_receipts = rebound_receipts

    combined_index = graph.selected_leaf_ids.index(COMBINED_RESULT_LEAF_ID) + 1
    combined_attempt = attempt(combined_index, COMBINED_RESULT_LEAF_ID)
    run_request = _write_blob(run_root / "request.json", {"mode": "agentic"})
    leaf_states = {
        leaf_id: AssetLeafState(
            leaf_id=leaf_id,
            requirement="required",
            status="passed",
            attempt_count=1,
            receipt=binding,
        )
        for leaf_id, binding in predecessor_receipts.items()
    }
    leaf_states[COMBINED_RESULT_LEAF_ID] = AssetLeafState(
        leaf_id=COMBINED_RESULT_LEAF_ID,
        requirement="required",
        depends_on=predecessor_ids,
        terminal_output=True,
        status="running",
        attempt_count=1,
        started_at="2026-08-25T00:00:00+00:00",
    )
    write_canonical_json(
        run_root / "asset_run.json",
        AssetCompositionRun(
            schema_version=ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
            revision=10,
            run_id="combined-fixture",
            request=run_request,
            source_asset=source_binding,
            selected_mode="agentic",
            execution_graph=graph_binding,
            graph_started_at="2026-08-25T00:00:00+00:00",
            current_leaf_id=COMBINED_RESULT_LEAF_ID,
            leaf_states=leaf_states,
            current_stage=None,
            stages={},
            coordinator=AssetCoordinatorState(
                mode="single_reasoning_loop",
                next_action="execute_leaf",
            ),
        ),
    )
    invocation = CombinedResultLeafInvocation(
        attempt_root=str(combined_attempt),
        final_asset=final_binding,
        required_predecessor_receipts=predecessor_receipts,
        output_report_path=str(combined_attempt / "combined_result.json"),
    )
    invocation_path = combined_attempt / "invocation.json"
    invocation_binding = write_canonical_json(invocation_path, invocation)
    monkeypatch.setattr(
        composed_leaf_cli, "validate_usdz_package_layout", lambda _p: None
    )
    result = composed_leaf_cli.run_combined_result_leaf(invocation_path)
    result_binding = write_canonical_json(combined_attempt / "result.json", result)
    return (
        combined_attempt,
        invocation,
        invocation_binding,
        result,
        result_binding,
        validation_native,
    )


def test_combined_leaf_binds_exact_predecessors_and_keeps_projection_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        combined_attempt,
        invocation,
        invocation_binding,
        result,
        result_binding,
        validation_native,
    ) = _combined_fixture(tmp_path, monkeypatch)
    projection = _runtime(COMBINED_RESULT_LEAF_ID).project(
        invocation,
        result,
        invocation_artifact=invocation_binding,
        result_artifact=result_binding,
    )

    assert result.status == "passed"
    assert result.output_asset is None
    assert (
        composed_leaf_cli.run_combined_result_leaf(combined_attempt / "invocation.json")
        == result
    )
    projected = (
        *projection.payload.evidence,
        *projection.payload.saved_stage_readbacks,
    )
    assert projected
    assert all(Path(item.path).is_relative_to(combined_attempt) for item in projected)

    validation_path = Path(validation_native.path)
    validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    validation_payload["asset_sha256"] = "0" * 64
    validation_path.write_text(
        json.dumps(validation_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity changed"):
        _runtime(COMBINED_RESULT_LEAF_ID).project(
            invocation,
            result,
            invocation_artifact=invocation_binding,
            result_artifact=result_binding,
        )


def test_combined_leaf_accepts_the_exact_run_bound_graph_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    *_, result, _result_binding, _validation_native = _combined_fixture(
        tmp_path,
        monkeypatch,
        graph_name="caller-selected-graph.json",
    )

    assert result.status == "passed"


def test_combined_projector_rejects_copied_same_run_predecessor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        combined_attempt,
        invocation,
        _invocation_binding,
        result,
        _result_binding,
        _validation_native,
    ) = _combined_fixture(tmp_path, monkeypatch)
    leaf_id = PHYSICS_INSPECTION_LEAF_ID
    original = invocation.required_predecessor_receipts[leaf_id]
    run_root = combined_attempt.parents[3]
    copied_path = (
        run_root / "leaves" / f"999-{leaf_id}" / "attempts" / "01" / "leaf_receipt.json"
    )
    copied_path.parent.mkdir(parents=True)
    copied_path.write_bytes(Path(original.path).read_bytes())
    copied = artifact_binding(copied_path)
    predecessor_receipts = dict(invocation.required_predecessor_receipts)
    predecessor_receipts[leaf_id] = copied
    drifted_invocation = invocation.model_copy(
        update={
            "required_predecessor_receipts": predecessor_receipts,
            "output_report_path": str(
                combined_attempt / "drifted_combined_result.json"
            ),
        }
    )
    drifted_invocation_binding = write_canonical_json(
        combined_attempt / "drifted_invocation.json",
        drifted_invocation,
    )
    native = CombinedAssetResultReceipt.model_validate_json(
        Path(result.native_terminal_receipt.path).read_text(encoding="utf-8")
    )
    drifted_native_binding = write_canonical_json(
        drifted_invocation.output_report_path,
        native.model_copy(
            update={"required_predecessor_receipts": predecessor_receipts}
        ),
    )
    drifted_result = result.model_copy(
        update={
            "invocation": drifted_invocation_binding,
            "native_terminal_receipt": drifted_native_binding,
            "evidence": (drifted_native_binding,),
            "saved_stage_readbacks": (drifted_native_binding,),
        }
    )
    drifted_result_binding = write_canonical_json(
        combined_attempt / "drifted_result.json",
        drifted_result,
    )

    with pytest.raises(ValueError, match="not the graph-state receipt"):
        _runtime(COMBINED_RESULT_LEAF_ID).project(
            drifted_invocation,
            drifted_result,
            invocation_artifact=drifted_invocation_binding,
            result_artifact=drifted_result_binding,
        )
