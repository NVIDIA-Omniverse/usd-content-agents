# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checked-in launchers for complete composed release-domain asset leaves."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from content_agent_workflows.asset_composition import (
    ArtifactBinding,
    bind_usd_dependency_closure,
)
from content_agent_workflows.asset_composition.release_leaf_adapters import (
    COMBINED_RESULT_LEAF_ID,
    MATERIAL_ASSIGNMENT_LEAF_ID,
    PHYSICS_APPLY_LEAF_ID,
    PHYSICS_INSPECTION_LEAF_ID,
    PORTABLE_PACKAGE_LEAF_ID,
    SIMREADY_CONFORMANCE_LEAF_ID,
    SIMREADY_VALIDATION_LEAF_ID,
    CombinedAssetResultReceipt,
    CombinedResultLeafInvocation,
    ComposedAssetLeafResult,
    ComposedLeafFailureReceipt,
    ComposedLeafInvocation,
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
    prepare_confined_output_path,
    read_artifact_binding_and_bytes,
    resolve_combined_predecessor_evidence,
    write_or_verify_canonical_json,
)
from content_agent_workflows.common.validation_evidence import (
    VALIDATION_EVIDENCE_SCHEMA_VERSION,
    ValidationEvidence,
)
from content_agent_workflows.physics import (
    inspect_physics_components,
    run_physics_apply_workflow,
)
from content_agent_workflows.simready import (
    SimReadyConformanceReport,
    run_simready_profile_conformance,
    run_simready_profile_validation,
)
from pydantic import BaseModel, ConfigDict
from world_understanding.utils.usd.package import validate_usdz_package_layout

from .material_coordinator import (
    MaterialCoordinatorPreparation,
    MaterialCoordinatorResult,
    MaterialCoordinatorReviewRequired,
    file_sha256,
    finalize_material_for_coordinator,
    prepare_material_for_coordinator,
    release_material_for_coordinator,
    review_material_for_coordinator,
)
from .runner import OPTIMIZER_SELECTION_FIXED, MaterialAssignConfig


class _MaterialReleaseFallback(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["released", "not_claimed", "failed"]
    resource: Literal["usd-cli"] = "usd-cli"
    error: str | None = None


def _prepare_invocation_output_paths(
    root: Path,
    invocation: ComposedLeafInvocation,
) -> None:
    """Materialize native output parents through the confined attempt root."""

    paths: list[tuple[str, str | Path, bool]]
    if isinstance(invocation, MaterialAssignmentLeafInvocation):
        paths = [
            ("Material output_asset_path", invocation.output_asset_path, False),
            ("Material decision_patch_path", invocation.decision_patch_path, False),
            ("Material review_patch_path", invocation.review_patch_path, False),
        ]
    elif isinstance(invocation, PhysicsInspectionLeafInvocation):
        paths = [
            (
                "Physics component_catalog_path",
                invocation.component_catalog_path,
                False,
            )
        ]
    elif isinstance(invocation, PhysicsApplyLeafInvocation):
        if invocation.params.output_usd_path is None:
            raise ValueError("Physics invocation omitted output_usd_path")
        paths = [
            ("Physics output_dir", invocation.params.output_dir, True),
            ("Physics output_usd_path", invocation.params.output_usd_path, False),
        ]
    elif isinstance(invocation, SimReadyConformanceLeafInvocation):
        paths = [
            ("SimReady conformance output_dir", invocation.params.output_dir, True)
        ]
        if invocation.params.report_path is not None:
            report_path = Path(invocation.params.report_path)
            output_dir = Path(invocation.params.output_dir)
            if not report_path.is_relative_to(output_dir):
                paths.append(
                    (
                        "SimReady conformance report_path",
                        report_path,
                        False,
                    )
                )
    elif isinstance(invocation, SimReadyValidationLeafInvocation):
        if invocation.params.report_path is None:
            raise ValueError("SimReady validation invocation omitted report_path")
        paths = [
            (
                "SimReady validation report_path",
                invocation.params.report_path,
                False,
            )
        ]
        paths.extend(
            (f"SimReady validation {label}", value, False)
            for label, value in (
                ("stdout_log_path", invocation.params.stdout_log_path),
                ("stderr_log_path", invocation.params.stderr_log_path),
            )
            if value is not None
        )
    elif isinstance(invocation, PortablePackageLeafInvocation):
        paths = [("portable package output", invocation.output_asset_path, False)]
    elif isinstance(invocation, CombinedResultLeafInvocation):
        paths = [("combined result output", invocation.output_report_path, False)]
    else:  # pragma: no cover - closed invocation union
        raise TypeError(f"unsupported composed leaf invocation: {type(invocation)}")
    for label, candidate, directory in paths:
        prepare_confined_output_path(
            root,
            candidate,
            label=label,
            directory=directory,
        )


def _load_invocation[InvocationT: ComposedLeafInvocation](
    path: str | Path,
    model: type[InvocationT],
) -> tuple[Path, ArtifactBinding, InvocationT]:
    binding, payload_bytes = read_artifact_binding_and_bytes(path)
    candidate = Path(binding.path)
    payload = json.loads(payload_bytes.decode("utf-8"))
    invocation = model.model_validate(payload)
    root = Path(invocation.attempt_root).resolve(strict=True)
    if candidate.parent != root or not root.is_dir() or root.is_symlink():
        raise ValueError("composed leaf invocation must be a direct attempt artifact")
    _prepare_invocation_output_paths(root, invocation)
    return root, binding, invocation


def _binding_tuple(paths: Sequence[str | Path]) -> tuple[ArtifactBinding, ...]:
    unique: dict[tuple[str, str, int], ArtifactBinding] = {}
    for path in paths:
        candidate = Path(path)
        if not candidate.is_file() or candidate.is_symlink():
            continue
        binding = artifact_binding(candidate)
        unique[(binding.path, binding.sha256, binding.size_bytes)] = binding
    return tuple(unique.values())


def _existing_model[ModelT: BaseModel](
    path: Path, model: type[ModelT]
) -> ModelT | None:
    if not path.is_file() or path.is_symlink():
        return None
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _native_workflow_params[ParamsT: BaseModel](
    params: ParamsT,
    *,
    run_dir: Path,
    record_subdir: str | Path | None = None,
) -> ParamsT:
    """Create a native recorder once, then resume that exact recorder."""

    record_root = run_dir
    if record_subdir is not None:
        relative = Path(record_subdir)
        if relative.is_absolute() or any(
            component in {"", ".", ".."} for component in relative.parts
        ):
            raise ValueError("native workflow recorder path is unsafe")
        record_root = run_dir / relative
    request_path = record_root / "request.json"
    manifest_path = record_root / "workflow_run_manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        artifact_binding(manifest_path)
        return params.model_copy(update={"resume": True})
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError("native workflow manifest is not a safe regular file")
    if request_path.exists() or request_path.is_symlink():
        raise ValueError("native workflow retained a request without its manifest")
    return params.model_copy(update={"resume": False})


def _seal_material_invocation(
    native_root: Path,
    invocation_binding: ArtifactBinding,
) -> ArtifactBinding:
    """Bind every retained Material phase to the first exact invocation."""

    seal_path = native_root / "composed_invocation_binding.json"
    retained_state = (
        native_root / "coordinator_preparation.json",
        native_root / "raw" / "material_application_receipt.json",
        native_root / "coordinator_result.json",
    )
    seal_exists = seal_path.exists() or seal_path.is_symlink()
    if not seal_exists and any(
        path.exists() or path.is_symlink() for path in retained_state
    ):
        raise ValueError(
            "Material retained state lacks its exact composed invocation identity"
        )
    return write_or_verify_canonical_json(seal_path, invocation_binding)


def _result(
    *,
    leaf_id: str,
    status: Literal["awaiting_decision", "awaiting_review", "passed", "failed"],
    native_status: str,
    invocation: Any,
    terminal: Any,
    evidence: tuple[Any, ...],
    readbacks: tuple[Any, ...],
    summary: str,
    output_asset: Any | None = None,
    resource_claims: tuple[str, ...] = (),
    releases: tuple[Any, ...] = (),
    error: str | None = None,
) -> ComposedAssetLeafResult:
    return ComposedAssetLeafResult(
        leaf_id=leaf_id,
        status=status,
        native_status=native_status,
        invocation=invocation,
        native_terminal_receipt=terminal,
        evidence=evidence,
        saved_stage_readbacks=readbacks,
        output_asset=output_asset,
        resource_claims=resource_claims,
        resource_release_receipts=releases,
        summary=summary,
        error=error,
    )


def _material_native_paths(
    preparation: MaterialCoordinatorPreparation,
) -> list[str | Path]:
    return [
        preparation.request_path,
        preparation.packet_path,
        preparation.visible_candidates_path,
        preparation.palette_path,
        preparation.authoring_context_path,
        preparation.assignment_seed_path,
        preparation.candidate_table_path,
        preparation.initial_render_records_binding.path,
        preparation.receipt_checkpoint_binding.path,
        *preparation.initial_render_paths,
        *(item.path for item in preparation.initial_render_bindings),
    ]


def _material_config(
    invocation: MaterialAssignmentLeafInvocation,
    *,
    native_root: Path,
) -> MaterialAssignConfig:
    return MaterialAssignConfig(
        repo_root=Path(invocation.repository_root),
        usd_path=Path(invocation.source.path),
        reference_images=[Path(item.path) for item in invocation.reference_images],
        reference_files=[Path(item.path) for item in invocation.reference_files],
        materials_yaml=Path(invocation.materials_yaml.path),
        materials_usd=Path(invocation.materials_usd.path),
        output_dir=native_root,
        output_usd_path=Path(invocation.output_asset_path),
        optimize=invocation.optimize,
        optimizer_selection=OPTIMIZER_SELECTION_FIXED,
        root_prim_path=invocation.root_prim_path,
        material_candidate_space=invocation.material_candidate_space,
        skip_instances=invocation.skip_instances,
        skip_prototypes=invocation.skip_prototypes,
        skip_invisible=invocation.skip_invisible,
        flatten_prototypes=invocation.flatten_prototypes,
        enable_deinstance=invocation.enable_deinstance,
        enable_split=invocation.enable_split,
        enable_deduplicate=invocation.enable_deduplicate,
        respect_existing_material_bindings=(
            invocation.respect_existing_material_bindings
        ),
        preflight=True,
        scene_tool_timeout_seconds=invocation.scene_tool_timeout_seconds,
    )


def run_material_leaf(path: str | Path) -> ComposedAssetLeafResult:
    root, invocation_binding, invocation = _load_invocation(
        path, MaterialAssignmentLeafInvocation
    )
    for binding in (
        invocation.source,
        invocation.materials_yaml,
        invocation.materials_usd,
        *invocation.materials_usd_dependencies,
        *invocation.reference_images,
        *invocation.reference_files,
    ):
        if artifact_binding(binding.path) != binding:
            raise ValueError(f"Material invocation artifact changed: {binding.path}")
    observed_dependencies = tuple(
        bind_usd_dependency_closure(invocation.materials_usd.path)
    )
    if observed_dependencies != invocation.materials_usd_dependencies:
        raise ValueError("Material library dependency closure changed")

    native_root = root / "native"
    invocation_seal = _seal_material_invocation(native_root, invocation_binding)
    preparation_path = native_root / "coordinator_preparation.json"
    preparation = _existing_model(preparation_path, MaterialCoordinatorPreparation)
    if preparation is None:
        preparation = prepare_material_for_coordinator(
            _material_config(invocation, native_root=native_root)
        )
    preparation_binding = artifact_binding(preparation_path)
    preparation_evidence = _binding_tuple(
        [invocation_seal.path, *_material_native_paths(preparation)]
    )

    decision_path = Path(invocation.decision_patch_path)
    canonical_decision = native_root / "raw" / "material_decision_patch.json"
    if decision_path != canonical_decision:
        raise ValueError(f"Material decision patch must use {canonical_decision}")
    if not decision_path.is_file():
        return _result(
            leaf_id=MATERIAL_ASSIGNMENT_LEAF_ID,
            status="awaiting_decision",
            native_status="awaiting_decision",
            invocation=invocation_binding,
            terminal=preparation_binding,
            evidence=preparation_evidence,
            readbacks=(preparation_binding,),
            resource_claims=(f"usd-cli:{preparation.session_id}",),
            summary="Material preparation is sealed and awaits an outer decision.",
        )

    application_path = native_root / "raw" / "material_application_receipt.json"
    application = _existing_model(application_path, MaterialCoordinatorReviewRequired)
    if application is None:
        application = finalize_material_for_coordinator(
            native_root,
            decision_patch_path=decision_path,
            preparation_sha256=file_sha256(preparation_path),
        )
    application_binding = artifact_binding(application_path)
    application_evidence = _binding_tuple(
        [
            application.request.path,
            application.decision_patch.path,
            application.policy.path,
            application.materialized_usd.path,
            application.receipt_checkpoint_binding.path,
            *[item.path for item in application.final_render_bindings],
            *[item.path for item in application.evidence],
        ]
    )

    review_path = Path(invocation.review_patch_path)
    canonical_review = native_root / "raw" / "material_post_apply_review.json"
    if review_path != canonical_review:
        raise ValueError(f"Material review patch must use {canonical_review}")
    if not review_path.is_file():
        return _result(
            leaf_id=MATERIAL_ASSIGNMENT_LEAF_ID,
            status="awaiting_review",
            native_status="awaiting_review",
            invocation=invocation_binding,
            terminal=application_binding,
            evidence=tuple(
                dict.fromkeys((*preparation_evidence, *application_evidence))
            ),
            readbacks=(preparation_binding, application_binding),
            resource_claims=(f"usd-cli:{preparation.session_id}",),
            summary="Material application is sealed and awaits exact OVRTX review.",
        )

    result_path = native_root / "coordinator_result.json"
    native_result = _existing_model(result_path, MaterialCoordinatorResult)
    if native_result is None:
        native_result = review_material_for_coordinator(
            native_root,
            review_patch_path=review_path,
            preparation_sha256=file_sha256(preparation_path),
            application_receipt_sha256=file_sha256(application_path),
        )
    result_binding = artifact_binding(result_path)
    release_path = native_root / "raw" / "material_session_release.json"
    release_binding = artifact_binding(release_path)
    materialized_readback = artifact_binding(application.materialized_usd.path)
    output = artifact_binding(native_result.output_usd_path)
    final_evidence = _binding_tuple(
        [
            *[item.path for item in native_result.evidence],
            preparation_path,
            application_path,
            review_path,
            result_path,
            release_path,
        ]
    )
    if native_result.status != "pass":
        return _result(
            leaf_id=MATERIAL_ASSIGNMENT_LEAF_ID,
            status="failed",
            native_status=native_result.status,
            invocation=invocation_binding,
            terminal=result_binding,
            evidence=final_evidence,
            readbacks=(
                result_binding,
                application_binding,
                materialized_readback,
                output,
            ),
            output_asset=output,
            resource_claims=(f"usd-cli:{preparation.session_id}",),
            releases=(release_binding,),
            summary="Material final review retained unresolved issues.",
            error="Material coordinator result is conditional, not a clean pass.",
        )
    return _result(
        leaf_id=MATERIAL_ASSIGNMENT_LEAF_ID,
        status="passed",
        native_status="pass",
        invocation=invocation_binding,
        terminal=result_binding,
        evidence=final_evidence,
        readbacks=(
            result_binding,
            application_binding,
            materialized_readback,
            output,
        ),
        output_asset=output,
        resource_claims=(f"usd-cli:{preparation.session_id}",),
        releases=(release_binding,),
        summary="Material assignment, OVRTX review, publication, and release passed.",
    )


def run_physics_inspection_leaf(path: str | Path) -> ComposedAssetLeafResult:
    root, invocation_binding, invocation = _load_invocation(
        path, PhysicsInspectionLeafInvocation
    )
    if artifact_binding(invocation.source.path) != invocation.source:
        raise ValueError("Physics inspection source changed")
    components = inspect_physics_components(invocation.source.path)
    receipt_path = Path(invocation.component_catalog_path)
    receipt = write_or_verify_canonical_json(
        receipt_path,
        PhysicsComponentInspectionReceipt(
            source=invocation.source,
            components=tuple(item.model_dump(mode="json") for item in components),
        ),
    )
    del root
    return _result(
        leaf_id=PHYSICS_INSPECTION_LEAF_ID,
        status="passed",
        native_status="pass",
        invocation=invocation_binding,
        terminal=receipt,
        evidence=(receipt,),
        readbacks=(receipt,),
        summary=f"Physics inspection sealed {len(components)} logical components.",
    )


_PHYSICS_RESULT_FIELDS = (
    "assignments_path",
    "decision_patch_path",
    "components_path",
    "candidate_prims_path",
    "predictions_path",
    "apply_report_path",
    "topology_report_path",
    "validation_evidence_path",
    "simulation_report_path",
    "behavior_assessment_path",
    "scene_operation_record_path",
    "vomp_result_path",
    "vomp_provenance_path",
    "workflow_run_manifest_path",
)


def _physics_result_artifacts(
    native_result: Any,
    *,
    root: Path,
) -> tuple[dict[str, ArtifactBinding], dict[str, bytes]]:
    """Capture every populated native-result path as exact attempt evidence."""

    bindings: dict[str, ArtifactBinding] = {}
    payloads: dict[str, bytes] = {}
    for field in _PHYSICS_RESULT_FIELDS:
        raw_path = getattr(native_result, field)
        if raw_path is None:
            continue
        lexical_candidate = Path(raw_path).expanduser()
        if not lexical_candidate.is_absolute():
            raise ValueError(f"Physics {field} escaped its composed attempt")
        candidate = Path(os.path.abspath(lexical_candidate))
        if not candidate.is_relative_to(root):
            raise ValueError(f"Physics {field} escaped its composed attempt")
        try:
            binding, payload = read_artifact_binding_and_bytes(candidate)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Physics {field} is not a safe regular file: {candidate}"
            ) from exc
        bindings[field] = binding
        payloads[field] = payload
    return bindings, payloads


def _physics_validation_binding(
    native_result: Any,
    *,
    artifacts: dict[str, ArtifactBinding],
    payloads: dict[str, bytes],
    output: ArtifactBinding | None,
) -> ArtifactBinding | None:
    raw_path = native_result.validation_evidence_path
    if raw_path is None:
        if native_result.success and native_result.validation_status == "pass":
            raise ValueError("Physics pass omitted validation_evidence_path")
        return None
    binding = artifacts["validation_evidence_path"]
    try:
        evidence = ValidationEvidence.model_validate_json(
            payloads["validation_evidence_path"]
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("Physics validation evidence is not valid typed JSON") from exc
    if native_result.success and native_result.validation_status == "pass":
        if output is None:
            raise ValueError("Physics pass omitted its authored output")
        if (
            evidence.schema_version != VALIDATION_EVIDENCE_SCHEMA_VERSION
            or evidence.workflow != "physics_authoring"
            or Path(evidence.asset) != Path(output.path)
            or evidence.metadata.get("asset_sha256") != output.sha256
            or evidence.sim_ready_status != "pass"
            or evidence.failures
            or evidence.warnings
            or evidence.unresolved_issues
        ):
            raise ValueError(
                "Physics validation evidence does not bind an exact clean output pass"
            )
    return binding


def run_physics_apply_leaf(path: str | Path) -> ComposedAssetLeafResult:
    root, invocation_binding, invocation = _load_invocation(
        path, PhysicsApplyLeafInvocation
    )
    for binding in (
        invocation.source,
        invocation.component_catalog,
        invocation.decision_patch,
    ):
        if artifact_binding(binding.path) != binding:
            raise ValueError(f"Physics invocation artifact changed: {binding.path}")
    catalog = json.loads(
        Path(invocation.component_catalog.path).read_text(encoding="utf-8")
    )
    observed = [
        item.model_dump(mode="json")
        for item in inspect_physics_components(invocation.source.path)
    ]
    if (
        catalog.get("source") != invocation.source.model_dump(mode="json")
        or catalog.get("components") != observed
    ):
        raise ValueError("Physics component catalog differs from the exact source")

    native_params = _native_workflow_params(
        invocation.params,
        run_dir=Path(invocation.params.output_dir),
        record_subdir=invocation.params.workflow_run_record_subdir,
    )
    native_result = run_physics_apply_workflow(native_params)
    native_result_path = root / "physics_apply_native_result.json"
    native_result_binding = write_or_verify_canonical_json(
        native_result_path,
        native_result,
    )
    native_artifacts, native_payloads = _physics_result_artifacts(
        native_result,
        root=root,
    )
    output = (
        artifact_binding(native_result.physics_usd_path)
        if native_result.physics_usd_path
        and Path(native_result.physics_usd_path).is_file()
        else None
    )
    validation_evidence = _physics_validation_binding(
        native_result,
        artifacts=native_artifacts,
        payloads=native_payloads,
        output=output,
    )
    evidence = tuple(dict.fromkeys((native_result_binding, *native_artifacts.values())))
    terminal_path = root / "physics_apply_leaf_terminal.json"
    terminal = write_or_verify_canonical_json(
        terminal_path,
        PhysicsApplyLeafReceipt(
            source=invocation.source,
            component_catalog=invocation.component_catalog,
            decision_patch=invocation.decision_patch,
            native_result=native_result_binding,
            validation_evidence=validation_evidence,
            output_asset=output,
        ),
    )
    evidence = tuple(dict.fromkeys((*evidence, native_result_binding, terminal)))
    if (
        not native_result.success
        or native_result.validation_status != "pass"
        or output is None
    ):
        return _result(
            leaf_id=PHYSICS_APPLY_LEAF_ID,
            status="failed",
            native_status=native_result.validation_status or "failed",
            invocation=invocation_binding,
            terminal=terminal,
            evidence=evidence,
            readbacks=(terminal, *((output,) if output is not None else ())),
            output_asset=output,
            summary="Physics authoring or runtime validation failed.",
            error=(
                native_result.error
                or "Physics workflow did not produce an exact runtime pass."
            ),
        )
    return _result(
        leaf_id=PHYSICS_APPLY_LEAF_ID,
        status="passed",
        native_status=native_result.validation_status,
        invocation=invocation_binding,
        terminal=terminal,
        evidence=evidence,
        readbacks=(terminal, output),
        output_asset=output,
        summary="Physics decisions, authoring, and runtime evidence passed.",
    )


def _publish_conformance_output(
    root: Path,
    report: SimReadyConformanceReport,
) -> ArtifactBinding:
    output = artifact_binding(report.output_usd_path)
    try:
        Path(output.path).relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "SimReady conformance output escaped its composed attempt"
        ) from exc
    return output


def run_simready_conformance_leaf(path: str | Path) -> ComposedAssetLeafResult:
    root, invocation_binding, invocation = _load_invocation(
        path, SimReadyConformanceLeafInvocation
    )
    if artifact_binding(invocation.source.path) != invocation.source:
        raise ValueError("SimReady conformance source changed")
    if invocation.params.report_path is None:  # pragma: no cover - model guard
        raise ValueError("SimReady conformance omitted its report path")
    native_params = _native_workflow_params(
        invocation.params,
        run_dir=Path(invocation.params.output_dir),
        record_subdir=f".{Path(invocation.params.report_path).name}.workflow-run",
    )
    report = run_simready_profile_conformance(native_params)
    if report.report_path is None:
        raise ValueError("SimReady conformance omitted its report")
    if Path(report.input_usd_path) != Path(invocation.source.path):
        raise ValueError(
            "SimReady conformance report input differs from source binding"
        )
    if Path(report.report_path) != Path(invocation.params.report_path):
        raise ValueError("SimReady conformance report path differs from its invocation")
    native_report_binding = artifact_binding(report.report_path)
    evidence = _binding_tuple(
        [
            report.report_path,
            *(report.reports.values()),
            *(
                [report.workflow_run_manifest_path]
                if report.workflow_run_manifest_path
                else []
            ),
        ]
    )
    output = _publish_conformance_output(root, report) if report.passed else None
    bound_report = (
        report.model_copy(update={"output_usd_path": output.path})
        if output is not None
        else report
    )
    bound_report_path = root / "simready_conformance_bound_report.json"
    bound_report_binding = write_or_verify_canonical_json(
        bound_report_path,
        bound_report,
    )
    terminal_path = root / "simready_conformance_leaf_terminal.json"
    terminal = write_or_verify_canonical_json(
        terminal_path,
        SimReadyConformanceLeafReceipt(
            source=invocation.source,
            native_report=bound_report_binding,
            output_asset=output,
        ),
    )
    evidence = tuple(
        dict.fromkeys(
            (
                *evidence,
                native_report_binding,
                bound_report_binding,
                terminal,
            )
        )
    )
    if not report.passed or output is None:
        return _result(
            leaf_id=SIMREADY_CONFORMANCE_LEAF_ID,
            status="failed",
            native_status=report.status,
            invocation=invocation_binding,
            terminal=terminal,
            evidence=evidence,
            readbacks=(terminal,),
            summary="SimReady conformance did not pass.",
            error="; ".join(report.errors) or f"SimReady status: {report.status}",
        )
    return _result(
        leaf_id=SIMREADY_CONFORMANCE_LEAF_ID,
        status="passed",
        native_status=report.status,
        invocation=invocation_binding,
        terminal=terminal,
        evidence=evidence,
        readbacks=(terminal, output),
        output_asset=output,
        summary="SimReady conformance passed and sealed an exact derivative.",
    )


def run_simready_validation_leaf(path: str | Path) -> ComposedAssetLeafResult:
    _root, invocation_binding, invocation = _load_invocation(
        path, SimReadyValidationLeafInvocation
    )
    if artifact_binding(invocation.source.path) != invocation.source:
        raise ValueError("SimReady validation source changed")
    if invocation.params.report_path is None:  # pragma: no cover - model guard
        raise ValueError("SimReady validation omitted its report path")
    validation_report_path = Path(invocation.params.report_path)
    native_params = _native_workflow_params(
        invocation.params,
        run_dir=validation_report_path.parent,
        record_subdir=f".{validation_report_path.name}.workflow-run",
    )
    report = run_simready_profile_validation(native_params)
    if report.report_path is None:
        raise ValueError("SimReady validation omitted its report")
    if Path(report.asset_path) != Path(invocation.source.path):
        raise ValueError("SimReady validation report asset differs from source binding")
    if report.asset_sha256 != invocation.source.sha256:
        raise ValueError(
            "SimReady validation report digest differs from source binding"
        )
    if Path(report.report_path) != validation_report_path:
        raise ValueError("SimReady validation report path differs from its invocation")
    report_binding = artifact_binding(report.report_path)
    evidence = _binding_tuple(
        [
            report.report_path,
            *([report.raw_report_path] if report.raw_report_path else []),
            *([report.stdout_log_path] if report.stdout_log_path else []),
            *([report.stderr_log_path] if report.stderr_log_path else []),
            *(
                [report.workflow_run_manifest_path]
                if report.workflow_run_manifest_path
                else []
            ),
        ]
    )
    if not report.passed:
        return _result(
            leaf_id=SIMREADY_VALIDATION_LEAF_ID,
            status="failed",
            native_status=report.status,
            invocation=invocation_binding,
            terminal=report_binding,
            evidence=evidence or (report_binding,),
            readbacks=(report_binding,),
            summary="SimReady validation did not pass.",
            error="; ".join(report.errors) or f"SimReady status: {report.status}",
        )
    return _result(
        leaf_id=SIMREADY_VALIDATION_LEAF_ID,
        status="passed",
        native_status=report.status,
        invocation=invocation_binding,
        terminal=report_binding,
        evidence=evidence,
        readbacks=(report_binding,),
        summary="SimReady Foundation validation passed.",
    )


def run_portable_package_leaf(path: str | Path) -> ComposedAssetLeafResult:
    root, invocation_binding, invocation = _load_invocation(
        path, PortablePackageLeafInvocation
    )
    if artifact_binding(invocation.source.path) != invocation.source:
        raise ValueError("portable package source changed")
    observed_dependencies = tuple(bind_usd_dependency_closure(invocation.source.path))
    if observed_dependencies != invocation.source_dependencies:
        raise ValueError("portable package source dependency closure changed")
    output = Path(invocation.output_asset_path)
    receipt_path = root / "portable_package_receipt.json"
    output_preexisting = output.exists() or output.is_symlink()
    if output_preexisting:
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise ValueError(
                "portable package output exists without its immutable receipt"
            )
    else:
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if Path(invocation.source.path).suffix.lower() == ".usdz":
            shutil.copy2(invocation.source.path, output)
        else:
            from content_agent_workflows.common.usd_package_localizer import (
                create_localized_usdz_package,
            )

            create_localized_usdz_package(
                invocation.source.path,
                output,
                invocation.root_layer_name,
            )
    package_member_order = validate_usdz_package_layout(output)
    actual_root_layer_name = package_member_order[0]
    if (
        Path(invocation.source.path).suffix.lower() != ".usdz"
        and actual_root_layer_name != invocation.root_layer_name
    ):
        raise ValueError("localized package root layer differs from its invocation")
    package = artifact_binding(output)
    package_dependencies = tuple(bind_usd_dependency_closure(output))
    if package_dependencies:
        raise ValueError("portable USDZ retained external dependencies")
    receipt_model = PortablePackageReceipt(
        source=invocation.source,
        source_dependencies=invocation.source_dependencies,
        package=package,
        package_dependencies=package_dependencies,
        root_layer_name=actual_root_layer_name,
    )
    receipt = write_or_verify_canonical_json(receipt_path, receipt_model)
    return _result(
        leaf_id=PORTABLE_PACKAGE_LEAF_ID,
        status="passed",
        native_status="pass",
        invocation=invocation_binding,
        terminal=receipt,
        evidence=(receipt,),
        readbacks=(receipt, package),
        output_asset=package,
        summary="Portable package is canonical, dependency-closed, and exact.",
    )


def run_combined_result_leaf(path: str | Path) -> ComposedAssetLeafResult:
    root, invocation_binding, invocation = _load_invocation(
        path, CombinedResultLeafInvocation
    )
    for binding in (
        invocation.final_asset,
        *invocation.required_predecessor_receipts.values(),
    ):
        if artifact_binding(binding.path) != binding:
            raise ValueError(f"combined result artifact changed: {binding.path}")
    validate_usdz_package_layout(Path(invocation.final_asset.path))
    package_receipt, simready_validation, canonical_ovrtx_evidence = (
        resolve_combined_predecessor_evidence(invocation)
    )
    receipt_model = CombinedAssetResultReceipt(
        final_asset=invocation.final_asset,
        package_receipt=package_receipt,
        simready_validation=simready_validation,
        canonical_ovrtx_evidence=canonical_ovrtx_evidence,
        required_predecessor_receipts=invocation.required_predecessor_receipts,
    )
    receipt = write_or_verify_canonical_json(
        invocation.output_report_path, receipt_model
    )
    return _result(
        leaf_id=COMBINED_RESULT_LEAF_ID,
        status="passed",
        native_status="pass",
        invocation=invocation_binding,
        terminal=receipt,
        evidence=(receipt,),
        readbacks=(receipt,),
        summary="Combined terminal binds the exact package and every required gate.",
    )


_OPERATIONS: dict[str, Callable[[str | Path], ComposedAssetLeafResult]] = {
    "material": run_material_leaf,
    "physics-inspect": run_physics_inspection_leaf,
    "physics-apply": run_physics_apply_leaf,
    "simready-conform": run_simready_conformance_leaf,
    "simready-validate": run_simready_validation_leaf,
    "package": run_portable_package_leaf,
    "combined": run_combined_result_leaf,
}


def _failure_result(
    *,
    operation: str,
    invocation_path: Path,
    error: Exception,
) -> ComposedAssetLeafResult:
    leaf_ids = {
        "material": MATERIAL_ASSIGNMENT_LEAF_ID,
        "physics-inspect": PHYSICS_INSPECTION_LEAF_ID,
        "physics-apply": PHYSICS_APPLY_LEAF_ID,
        "simready-conform": SIMREADY_CONFORMANCE_LEAF_ID,
        "simready-validate": SIMREADY_VALIDATION_LEAF_ID,
        "package": PORTABLE_PACKAGE_LEAF_ID,
        "combined": COMBINED_RESULT_LEAF_ID,
    }
    root = invocation_path.parent.resolve(strict=True)
    failure_path = root / f"{operation}_leaf_failure.json"
    failure_model = ComposedLeafFailureReceipt(
        leaf_id=leaf_ids[operation],
        error_type=type(error).__name__,
        error=str(error).strip() or "no exception detail",
    )
    retained_failure = _existing_model(failure_path, ComposedLeafFailureReceipt)
    if retained_failure is None:
        failure = write_or_verify_canonical_json(failure_path, failure_model)
        retained_failure = failure_model
    else:
        failure = artifact_binding(failure_path)
    releases: tuple[Any, ...] = ()
    resource_claims: tuple[str, ...] = ()
    if operation == "material":
        release_path = root / "material_leaf_resource_release.json"
        native_root = root / "native"
        preparation_path = native_root / "coordinator_preparation.json"
        preparation_error: Exception | None = None
        try:
            preparation = _existing_model(
                preparation_path,
                MaterialCoordinatorPreparation,
            )
        except Exception as exc:  # noqa: BLE001 - retain corrupt cleanup evidence
            preparation = None
            preparation_error = exc
        if preparation is not None:
            resource_claims = (f"usd-cli:{preparation.session_id}",)
        retained_release = _existing_model(release_path, _MaterialReleaseFallback)
        if retained_release is None:
            if preparation_error is not None:
                release_model = _MaterialReleaseFallback(
                    status="failed",
                    error=(
                        "Cannot verify retained Material preparation for release: "
                        f"{type(preparation_error).__name__}: {preparation_error}"
                    ),
                )
            else:
                try:
                    application_path = (
                        native_root / "raw" / "material_application_receipt.json"
                    )
                    if preparation is not None:
                        release_material_for_coordinator(
                            native_root,
                            preparation_sha256=file_sha256(preparation_path),
                            application_receipt_sha256=(
                                file_sha256(application_path)
                                if application_path.is_file()
                                else None
                            ),
                        )
                        release_model = _MaterialReleaseFallback(status="released")
                    else:
                        release_model = _MaterialReleaseFallback(status="not_claimed")
                except Exception as release_error:  # noqa: BLE001
                    release_model = _MaterialReleaseFallback(
                        status="failed",
                        error=f"{type(release_error).__name__}: {release_error}",
                    )
            release_binding = write_or_verify_canonical_json(
                release_path, release_model
            )
        else:
            release_binding = artifact_binding(release_path)
        releases = (release_binding,)
    return _result(
        leaf_id=leaf_ids[operation],
        status="failed",
        native_status="exception",
        invocation=artifact_binding(invocation_path),
        terminal=failure,
        evidence=(failure,),
        readbacks=(failure,),
        resource_claims=resource_claims,
        releases=releases,
        summary=f"{leaf_ids[operation]} failed closed.",
        error=f"{retained_failure.error_type}: {retained_failure.error}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=sorted(_OPERATIONS))
    parser.add_argument("--invocation", type=Path, required=True)
    args = parser.parse_args(argv)
    invocation_path = args.invocation.expanduser().resolve(strict=True)
    try:
        result = _OPERATIONS[args.operation](invocation_path)
    except Exception as exc:  # noqa: BLE001 - stable typed failure boundary
        result = _failure_result(
            operation=args.operation,
            invocation_path=invocation_path,
            error=exc,
        )
    print(result.model_dump_json(indent=2))
    return 2 if result.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
