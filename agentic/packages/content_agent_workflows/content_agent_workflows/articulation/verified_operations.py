# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint-owned projectors for shared verified-operation ingress.

These leaves verify already produced Joint artifacts.  They never invoke Joint
inference, a renderer, or a simulator.  Shared Validation authenticates the
resulting envelope and retains it without re-running the domain capability.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from world_understanding.utils.captured_artifacts import (
        CapturedOpaqueFile,
        OpaqueArtifactRequest,
    )

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)
from content_agent_workflows.validation.verified_operations import (
    VerifiedNativeStatus,
    VerifiedOperationComponentIdentity,
    VerifiedOperationError,
    VerifiedValidationOperationEnvelope,
    VerifiedValidationOperationProjection,
    execution_artifact_binding,
    verify_execution_artifact_binding,
    verify_operation_envelope,
)

from .embedded_decision import (
    EmbeddedArticulationCanonicalGraph,
    EmbeddedArticulationReadback,
    validate_completed_embedded_articulation_checkpoint,
)
from .models import ArticulationAuthoringResult, ArticulationRunState
from .output_evidence import (
    EmbeddedArticulationOutputEvidence,
    EmbeddedArticulationTerminalReceipt,
    validate_embedded_articulation_output_evidence,
)

VERIFIED_OPERATION_CONTRACT_CHECKPOINT: Final = (
    "2ba10dbd904ab8f5ed313f345876d7c8cdddeded"
)
JOINT_VERIFIED_OPERATION_PUBLICATION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.joint-verified-operation-publication.v1"
)
JOINT_DYNAMIC_ARTIFACT_MAP_SCHEMA_VERSION: Final = (
    "content-agent-workflows.joint-dynamic-artifact-map.v1"
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JointVerifiedOperationPublication(_FrozenModel):
    """Exact paths and shared v1 result emitted by one Joint projector leaf."""

    schema_version: Literal[
        "content-agent-workflows.joint-verified-operation-publication.v1"
    ] = JOINT_VERIFIED_OPERATION_PUBLICATION_SCHEMA_VERSION
    projection: ExecutionArtifactBinding
    envelope: ExecutionArtifactBinding
    result: VerifiedValidationOperationEnvelope
    shared_contract_checkpoint: Literal["2ba10dbd904ab8f5ed313f345876d7c8cdddeded"] = (
        VERIFIED_OPERATION_CONTRACT_CHECKPOINT
    )
    joint_inference_invoked: Literal[False] = False
    simulator_invoked: Literal[False] = False
    renderer_invoked: Literal[False] = False


class JointDynamicArtifactMap(_FrozenModel):
    """Explicit local resolver inputs for trusted dynamic re-verification."""

    schema_version: Literal["content-agent-workflows.joint-dynamic-artifact-map.v1"] = (
        JOINT_DYNAMIC_ARTIFACT_MAP_SCHEMA_VERSION
    )
    artifacts: dict[str, str]


class _LocalArtifactResolver:
    """Resolve only explicitly mapped local artifacts through the trusted capture API."""

    def __init__(self, artifacts: Mapping[str, str]) -> None:
        self._artifacts = dict(artifacts)

    @contextmanager
    def capture(self, request: OpaqueArtifactRequest) -> Iterator[CapturedOpaqueFile]:
        from world_understanding.utils.captured_artifacts import (
            capture_local_opaque_file,
        )

        raw_path = self._artifacts.get(request.uri)
        if raw_path is None:
            raise ValueError(f"dynamic artifact URI is not mapped: {request.uri}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise ValueError("dynamic artifact map paths must be absolute")
        with capture_local_opaque_file(path, request) as captured:
            yield captured


def _load_json_object(
    binding: ExecutionArtifactBinding, *, label: str
) -> dict[str, object]:
    payload = verify_execution_artifact_binding(binding, label=label)
    try:
        document = json.loads(payload)
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError) as exc:
        raise VerifiedOperationError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise VerifiedOperationError(f"{label} must be a JSON object")
    return document


def _load_bound_model[ModelT: BaseModel](
    binding: ExecutionArtifactBinding,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    try:
        return model.model_validate_json(
            verify_execution_artifact_binding(binding, label=label)
        )
    except ValueError as exc:
        raise VerifiedOperationError(f"invalid {label}: {exc}") from exc


def _load_strict_bound_model[ModelT: BaseModel](
    binding: ExecutionArtifactBinding,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    try:
        return model.model_validate_json(
            verify_execution_artifact_binding(binding, label=label),
            strict=True,
        )
    except ValueError as exc:
        raise VerifiedOperationError(f"invalid {label}: {exc}") from exc


def _component(
    *,
    component_id: str,
    version: str,
    contract: str | Path,
    configuration: ExecutionArtifactBinding | None,
) -> VerifiedOperationComponentIdentity:
    return VerifiedOperationComponentIdentity(
        component_id=component_id,
        version=version,
        contract=execution_artifact_binding(Path(contract).resolve()),
        configuration=configuration,
    )


def _dependency_bindings(
    output: ExecutionArtifactBinding,
) -> tuple[ExecutionArtifactBinding, ...]:
    from content_agent_workflows.simready.asset_identity import (
        asset_dependency_root_sha256,
        build_asset_dependency_manifest,
    )

    manifest = build_asset_dependency_manifest(output.path)
    if asset_dependency_root_sha256(manifest) != output.sha256:
        raise VerifiedOperationError("USD dependency manifest root differs from output")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise VerifiedOperationError("USD dependency manifest has no file list")
    dependencies_list: list[ExecutionArtifactBinding] = []
    for item in raw_files:
        if not isinstance(item, dict) or item.get("role") != "dependency":
            continue
        dependency_path = item.get("path")
        if not isinstance(dependency_path, str) or not dependency_path:
            raise VerifiedOperationError(
                "USD dependency manifest has a dependency without a valid path"
            )
        try:
            dependencies_list.append(execution_artifact_binding(dependency_path))
        except (OSError, TypeError, ValueError) as exc:
            raise VerifiedOperationError(
                "USD dependency manifest has an invalid dependency path"
            ) from exc
    dependencies = tuple(dependencies_list)
    for binding in dependencies:
        verify_execution_artifact_binding(binding, label="Joint USD dependency")
    return dependencies


def _fresh_output_root(output_dir: str | Path) -> Path:
    raw_root = Path(output_dir).expanduser()
    if raw_root.is_symlink():
        raise VerifiedOperationError(f"Joint projector output is a symlink: {raw_root}")
    root = raw_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    return root


def _publish(
    *,
    output_dir: str | Path,
    operation_id: str,
    gate_id: str,
    evidence_type: str,
    native_report_type: str,
    native_payload_type: str | None,
    claim_scope: str,
    native_status: VerifiedNativeStatus,
    source: ExecutionArtifactBinding,
    output: ExecutionArtifactBinding,
    dependencies: tuple[ExecutionArtifactBinding, ...],
    artifacts: tuple[ExecutionArtifactBinding, ...],
    native_report: ExecutionArtifactBinding,
    native_payload: ExecutionArtifactBinding | None,
    producer: VerifiedOperationComponentIdentity,
    tool: VerifiedOperationComponentIdentity,
    profile: VerifiedOperationComponentIdentity,
    backend: VerifiedOperationComponentIdentity,
    verifier: VerifiedOperationComponentIdentity,
    projector: VerifiedOperationComponentIdentity,
) -> JointVerifiedOperationPublication:
    root = _fresh_output_root(output_dir)
    values = {
        "operation_id": operation_id,
        "gate_id": gate_id,
        "evidence_type": evidence_type,
        "native_report_type": native_report_type,
        "native_payload_type": native_payload_type,
        "claim_scope": claim_scope,
        "native_status": native_status,
        "required": True,
        "authority": "deterministic_fact",
        "source": source,
        "output": output,
        "dependencies": dependencies,
        "artifacts": artifacts,
        "native_report": native_report,
        "native_payload": native_payload,
    }
    projection = VerifiedValidationOperationProjection(
        **values,
        producer_identity_sha256=canonical_json_digest(producer),
        tool_identity_sha256=canonical_json_digest(tool),
        profile_identity_sha256=canonical_json_digest(profile),
        backend_identity_sha256=canonical_json_digest(backend),
        verifier_identity_sha256=canonical_json_digest(verifier),
        projector_identity_sha256=canonical_json_digest(projector),
    )
    projection_path = root / "verified_operation_projection.json"
    atomic_write_json(projection_path, projection)
    projection_binding = execution_artifact_binding(projection_path)
    envelope = VerifiedValidationOperationEnvelope(
        **values,
        producer=producer,
        tool=tool,
        profile=profile,
        backend=backend,
        verifier=verifier,
        projector=projector,
        projection=projection_binding,
    )
    envelope_path = root / "verified_operation_envelope.json"
    atomic_write_json(envelope_path, envelope)
    envelope_binding = execution_artifact_binding(envelope_path)
    verified = verify_operation_envelope(
        VerifiedValidationOperationEnvelope.model_validate_json(
            verify_execution_artifact_binding(
                envelope_binding,
                label="Joint verified operation envelope",
            )
        )
    )
    return JointVerifiedOperationPublication(
        projection=projection_binding,
        envelope=envelope_binding,
        result=verified,
    )


def _state_binding(state_binding: object, *, label: str) -> ExecutionArtifactBinding:
    if state_binding is None:
        raise VerifiedOperationError(f"completed Articulation lacks {label}")
    path = getattr(state_binding, "path", None)
    digest = getattr(state_binding, "sha256", None)
    if not isinstance(path, str) or not isinstance(digest, str):
        raise VerifiedOperationError(f"completed Articulation has malformed {label}")
    binding = execution_artifact_binding(path)
    if binding.sha256 != digest:
        raise VerifiedOperationError(f"completed Articulation {label} is stale")
    return binding


def project_joint_graph_apply_result(
    run_dir: str | Path,
    *,
    output_dir: str | Path,
) -> JointVerifiedOperationPublication:
    """Project exact graph apply/readback from a completed embedded run."""

    root = Path(run_dir).expanduser().resolve()
    try:
        state = ArticulationRunState.model_validate_json(
            (root / "checkpoint.json").read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise VerifiedOperationError(f"invalid Articulation checkpoint: {exc}") from exc
    authoring_binding = _state_binding(state.authoring_result, label="authoring result")
    authoring = _load_bound_model(
        authoring_binding,
        ArticulationAuthoringResult,
        label="authoring result",
    )
    validate_completed_embedded_articulation_checkpoint(
        state,
        authoring=authoring,
    )
    graph_binding = _state_binding(
        state.embedded_canonical_graph,
        label="canonical graph",
    )
    readback_binding = _state_binding(state.embedded_readback, label="saved readback")
    output_evidence_binding = _state_binding(
        state.embedded_output_evidence,
        label="output evidence",
    )
    terminal_binding = _state_binding(
        state.embedded_terminal_receipt,
        label="terminal receipt",
    )
    graph = _load_bound_model(
        graph_binding,
        EmbeddedArticulationCanonicalGraph,
        label="canonical graph",
    )
    readback = _load_bound_model(
        readback_binding,
        EmbeddedArticulationReadback,
        label="saved readback",
    )
    output_evidence = _load_bound_model(
        output_evidence_binding,
        EmbeddedArticulationOutputEvidence,
        label="output evidence",
    )
    terminal = _load_bound_model(
        terminal_binding,
        EmbeddedArticulationTerminalReceipt,
        label="terminal receipt",
    )
    validate_embedded_articulation_output_evidence(output_evidence)
    source = output_evidence.source
    if (
        readback.canonical_graph_digest != canonical_json_digest(graph)
        or terminal.canonical_graph_digest != canonical_json_digest(graph)
        or terminal.output_evidence.path != output_evidence_binding.path
        or terminal.output_evidence.sha256 != output_evidence_binding.sha256
        or terminal.output_asset != output_evidence.post_mutation_output
    ):
        raise VerifiedOperationError("Joint graph/apply terminal chain is stale")
    contract = Path(__file__).resolve()
    embedded_contract = Path(__file__).with_name("embedded_decision.py").resolve()
    authoring_contract = Path(__file__).with_name("client.py").resolve()
    common_configuration = terminal_binding
    return _publish(
        output_dir=output_dir,
        operation_id="joint.graph-apply-readback",
        gate_id="joint.graph-apply",
        evidence_type="joint.graph-apply-readback",
        native_report_type="joint.articulation-readback",
        native_payload_type="joint.articulation-terminal-receipt",
        claim_scope="exact canonical graph apply and saved-stage readback",
        native_status="pass",
        source=source,
        output=output_evidence.post_mutation_output,
        dependencies=output_evidence.dependencies,
        artifacts=(
            graph_binding,
            output_evidence_binding,
            output_evidence.canonical_visual_envelope,
            output_evidence.canonical_visual_projection,
            output_evidence.canonical_visual_request,
            output_evidence.canonical_visual_payload,
            output_evidence.render_report,
            *output_evidence.images,
        ),
        native_report=readback_binding,
        native_payload=terminal_binding,
        producer=_component(
            component_id="joint-agent-graph-authoring-client",
            version="content-agent-workflows.joint-graph-authoring.v1",
            contract=authoring_contract,
            configuration=graph_binding,
        ),
        tool=_component(
            component_id="embedded-articulation-graph-apply",
            version="content-agent-workflows.embedded-articulation.v3",
            contract=embedded_contract,
            configuration=graph_binding,
        ),
        profile=_component(
            component_id="embedded-articulation-graph-readback",
            version="content-agent-workflows.embedded-articulation-readback.v1",
            contract=embedded_contract,
            configuration=readback_binding,
        ),
        backend=_component(
            component_id="joint-rigger-owned-core",
            version="joint-agent.graph-authoring.v1",
            contract=authoring_contract,
            configuration=common_configuration,
        ),
        verifier=_component(
            component_id="embedded-articulation-terminal-verifier",
            version="content-agent-workflows.embedded-articulation-terminal-receipt.v1",
            contract=embedded_contract,
            configuration=terminal_binding,
        ),
        projector=_component(
            component_id="joint-graph-apply-verified-operation-projector",
            version="content-agent-workflows.joint-projector.v1",
            contract=contract,
            configuration=terminal_binding,
        ),
    )


@dataclass(frozen=True)
class _StaticGateInputs:
    source: ExecutionArtifactBinding
    output: ExecutionArtifactBinding
    dependencies: tuple[ExecutionArtifactBinding, ...]
    report_binding: ExecutionArtifactBinding
    closeout_binding: ExecutionArtifactBinding
    plan_binding: ExecutionArtifactBinding
    intake_binding: ExecutionArtifactBinding
    authoring_receipt_binding: ExecutionArtifactBinding
    report: dict[str, object]
    plan: BaseModel
    generated: BaseModel


def _static_gate_inputs(
    *,
    source_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    closeout_path: str | Path,
    run_plan_path: str | Path,
    intake_path: str | Path,
    authoring_receipt_path: str | Path,
    stage_name: Literal["gate3a", "gate3b"],
) -> _StaticGateInputs:
    from joint_agent.articulation_v2_static_artifact_identity import (
        ArticulationV2StaticGeneratedArtifactIdentitiesV1,
        gate3_v3_dependency_manifest_from_capture,
        identify_articulation_v2_static_generated_artifact,
    )
    from joint_agent.articulation_v2_static_intake import (
        ArticulationV2StaticIntakeV1,
        articulation_v2_static_intake_sha256,
        validate_articulation_v2_static_intake,
    )
    from joint_agent.articulation_v2_static_receipts import (
        ArticulationV2StaticAuthoringReceiptV1,
        canonical_articulation_v2_static_gate3_closeout_bytes,
    )
    from joint_agent.articulation_v2_static_run_plan import (
        ArticulationV2StaticRunPlanV1,
        articulation_v2_static_run_plan_sha256,
    )
    from joint_agent.capability_manifest import (
        CapabilityManifestError,
        load_capability_manifest,
    )
    from world_understanding.utils.captured_artifacts import (
        OpaqueArtifactRequest,
        capture_local_opaque_file,
    )

    source = execution_artifact_binding(source_path)
    output = execution_artifact_binding(output_path)
    dependencies = _dependency_bindings(output)
    report_binding = execution_artifact_binding(report_path)
    closeout_binding = execution_artifact_binding(closeout_path)
    plan_binding = execution_artifact_binding(run_plan_path)
    intake_binding = execution_artifact_binding(intake_path)
    authoring_receipt_binding = execution_artifact_binding(authoring_receipt_path)
    plan = _load_strict_bound_model(
        plan_binding,
        ArticulationV2StaticRunPlanV1,
        label="Joint static run plan",
    )
    intake = _load_strict_bound_model(
        intake_binding,
        ArticulationV2StaticIntakeV1,
        label="Joint static intake",
    )
    authoring_receipt = _load_strict_bound_model(
        authoring_receipt_binding,
        ArticulationV2StaticAuthoringReceiptV1,
        label="Joint retained authoring receipt",
    )
    if (
        intake.run_plan_sha256 != articulation_v2_static_run_plan_sha256(plan)
        or intake.run != plan.run
        or intake.output != plan.output
    ):
        raise VerifiedOperationError("Joint static plan and intake identities differ")
    try:
        validate_articulation_v2_static_intake(
            load_capability_manifest(),
            plan,
            intake,
        )
    except (CapabilityManifestError, TypeError, ValueError) as exc:
        raise VerifiedOperationError(
            "Joint static plan or intake is not admitted by the packaged manifest"
        ) from exc
    selected_source = plan.run.selected_source
    evidence_root = Path(plan_binding.path).parent.parent
    expected_paths = {
        "run plan": evidence_root / "input/run-plan.json",
        "intake": evidence_root / "input/intake.json",
        "closeout": evidence_root / "input/gate3-closeout.json",
        "authoring receipt": evidence_root / "receipts/authoring.json",
        "report": evidence_root / f"reports/{stage_name}.json",
        "output": evidence_root / plan.output.output_key,
    }
    observed_paths = {
        "run plan": Path(plan_binding.path),
        "intake": Path(intake_binding.path),
        "closeout": Path(closeout_binding.path),
        "authoring receipt": Path(authoring_receipt_binding.path),
        "report": Path(report_binding.path),
        "output": Path(output.path),
    }
    for label, expected in expected_paths.items():
        if observed_paths[label] != expected.resolve():
            raise VerifiedOperationError(
                f"Joint static {label} path differs from the canonical retained root"
            )
    locator_parts = tuple(part for part in Path(selected_source.locator).parts if part)
    source_parts = Path(source.path).parts
    if (
        source.sha256 != selected_source.root_sha256
        or source.size_bytes > selected_source.max_bytes
        or not locator_parts
        or tuple(source_parts[-len(locator_parts) :]) != locator_parts
    ):
        raise VerifiedOperationError(
            "Joint static source realization differs from retained URI and digest"
        )
    request = OpaqueArtifactRequest(
        uri=selected_source.locator,
        sha256=source.sha256,
        size_bytes=source.size_bytes,
        max_bytes=selected_source.max_bytes,
    )
    with capture_local_opaque_file(source.path, request) as captured_source:
        source_manifest = gate3_v3_dependency_manifest_from_capture(
            captured_source,
            representation=selected_source.representation,
        )
    if (
        source_manifest.sha256 != selected_source.gate3_dependency_bundle.sha256
        or len(source_manifest.entries)
        != selected_source.gate3_dependency_bundle.entry_count
    ):
        raise VerifiedOperationError(
            "Joint static source dependency closure differs from retained plan"
        )
    generated_identities = authoring_receipt.generated_identities
    if (
        authoring_receipt.status != "pass"
        or authoring_receipt.binding.run_plan_sha256
        != articulation_v2_static_run_plan_sha256(plan)
        or authoring_receipt.binding.intake_sha256
        != articulation_v2_static_intake_sha256(intake)
        or authoring_receipt.binding.run != plan.run
        or authoring_receipt.stage.plan != plan.stages.authoring
        or authoring_receipt.generated_file is None
        or generated_identities is None
        or authoring_receipt.generated_file.path != plan.output.output_key
        or authoring_receipt.generated_file.sha256 != output.sha256
        or authoring_receipt.generated_file.size_bytes != output.size_bytes
    ):
        raise VerifiedOperationError(
            "Joint retained authoring receipt differs from plan, intake, or output"
        )
    if not isinstance(
        generated_identities,
        ArticulationV2StaticGeneratedArtifactIdentitiesV1,
    ):
        raise VerifiedOperationError("Joint retained authoring identities are missing")
    observed_generated = identify_articulation_v2_static_generated_artifact(
        output.path,
        output=plan.output,
    )
    if observed_generated != generated_identities:
        raise VerifiedOperationError(
            "Joint static output differs from retained generated identities"
        )
    report = _load_json_object(report_binding, label="Joint Gate 3 report")
    closeout = _load_json_object(closeout_binding, label="Joint Gate 3 closeout")
    expected_closeout = canonical_articulation_v2_static_gate3_closeout_bytes(
        plan,
        intake,
        generated_identities,
    )
    if (
        verify_execution_artifact_binding(
            closeout_binding,
            label="Joint Gate 3 closeout",
        )
        != expected_closeout
        or report.get("closeout_sha256") != closeout_binding.sha256
        or closeout.get("intake_sha256") != articulation_v2_static_intake_sha256(intake)
    ):
        raise VerifiedOperationError(
            "Joint Gate 3 report or closeout differs from the exact retained run"
        )
    return _StaticGateInputs(
        source=source,
        output=output,
        dependencies=dependencies,
        report_binding=report_binding,
        closeout_binding=closeout_binding,
        plan_binding=plan_binding,
        intake_binding=intake_binding,
        authoring_receipt_binding=authoring_receipt_binding,
        report=report,
        plan=plan,
        generated=generated_identities,
    )


def project_joint_gate3a_result(
    *,
    source_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    closeout_path: str | Path,
    run_plan_path: str | Path,
    intake_path: str | Path,
    authoring_receipt_path: str | Path,
    output_dir: str | Path,
) -> JointVerifiedOperationPublication:
    """Verify one retained Isaac Asset Validator report and project its fact."""

    inputs = _static_gate_inputs(
        source_path=source_path,
        output_path=output_path,
        report_path=report_path,
        closeout_path=closeout_path,
        run_plan_path=run_plan_path,
        intake_path=intake_path,
        authoring_receipt_path=authoring_receipt_path,
        stage_name="gate3a",
    )
    from joint_agent.articulation_v2_static_artifact_identity import (
        ArticulationV2StaticGeneratedArtifactIdentitiesV1,
    )
    from joint_agent.articulation_v2_static_gate3a_contracts import (
        validate_articulation_v2_static_gate3a_report_consistency,
    )
    from joint_agent.articulation_v2_static_run_plan import (
        ArticulationV2StaticRunPlanV1,
    )
    from joint_agent.internal.articulation_v2_static_gate3_closeout import (
        _require_generated_result_identity,
    )

    plan = cast(ArticulationV2StaticRunPlanV1, inputs.plan)
    generated = cast(
        ArticulationV2StaticGeneratedArtifactIdentitiesV1,
        inputs.generated,
    )
    stage = plan.stages.gate3a
    report = inputs.report
    if report.get("schema_version") != stage.admission.validation_contract:
        raise VerifiedOperationError("Gate 3A report uses an unsupported contract")
    validate_articulation_v2_static_gate3a_report_consistency(report)
    results = report.get("results")
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
    ):
        raise VerifiedOperationError("Gate 3A projector requires exactly one result")
    result = results[0]
    _require_generated_result_identity(
        result,
        plan=plan,
        generated_identities=generated,
        label="Gate 3A",
    )
    status = {
        "pass": "pass",
        "warning": "warn",
        "fail": "fail",
        "validator_exception": "error",
        "not_run": "not_evaluated",
    }.get(result.get("status"))
    if status is None:
        raise VerifiedOperationError("Gate 3A result has an unsupported status")
    profile = report.get("validation_profile")
    profile_id = profile.get("name") if isinstance(profile, dict) else None
    if profile_id != stage.admission.profile_id:
        raise VerifiedOperationError("Gate 3A report lacks exact admitted profile")
    contract = Path(__file__).resolve()
    gate_contract = Path(
        __import__(
            "joint_agent.articulation_v2_static_gate3a_contracts",
            fromlist=["__file__"],
        ).__file__
    ).resolve()
    return _publish(
        output_dir=output_dir,
        operation_id="joint.gate3a-isaac-asset-validator",
        gate_id="joint.gate3a",
        evidence_type="joint.static-schema-validation",
        native_report_type="joint.gate3a-report",
        native_payload_type="joint.gate3-closeout",
        claim_scope="Joint Gate 3A static Isaac Asset Validator result",
        native_status=status,
        source=inputs.source,
        output=inputs.output,
        dependencies=inputs.dependencies,
        artifacts=(
            inputs.plan_binding,
            inputs.intake_binding,
            inputs.authoring_receipt_binding,
        ),
        native_report=inputs.report_binding,
        native_payload=inputs.closeout_binding,
        producer=_component(
            component_id="joint-gate3a-report-producer",
            version=stage.admission.validation_contract,
            contract=gate_contract,
            configuration=inputs.closeout_binding,
        ),
        tool=_component(
            component_id=stage.command.tool_id,
            version=stage.command.tool_version,
            contract=gate_contract,
            configuration=inputs.plan_binding,
        ),
        profile=_component(
            component_id=profile_id,
            version=stage.admission.validation_contract,
            contract=gate_contract,
            configuration=inputs.report_binding,
        ),
        backend=_component(
            component_id="isaac-sim-gate3a-runtime",
            version=stage.command.tool_version,
            contract=gate_contract,
            configuration=inputs.plan_binding,
        ),
        verifier=_component(
            component_id="joint-gate3a-report-consistency-verifier",
            version=stage.admission.validation_contract,
            contract=gate_contract,
            configuration=inputs.report_binding,
        ),
        projector=_component(
            component_id="joint-gate3a-verified-operation-projector",
            version="content-agent-workflows.joint-projector.v1",
            contract=contract,
            configuration=inputs.report_binding,
        ),
    )


def project_joint_gate3b_result(
    *,
    source_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    closeout_path: str | Path,
    run_plan_path: str | Path,
    intake_path: str | Path,
    authoring_receipt_path: str | Path,
    output_dir: str | Path,
) -> JointVerifiedOperationPublication:
    """Verify one retained SimReady Foundation report and project its fact."""

    inputs = _static_gate_inputs(
        source_path=source_path,
        output_path=output_path,
        report_path=report_path,
        closeout_path=closeout_path,
        run_plan_path=run_plan_path,
        intake_path=intake_path,
        authoring_receipt_path=authoring_receipt_path,
        stage_name="gate3b",
    )
    from joint_agent.articulation_v2_static_artifact_identity import (
        ArticulationV2StaticGeneratedArtifactIdentitiesV1,
    )
    from joint_agent.articulation_v2_static_gate3b_contracts import (
        validate_articulation_v2_static_gate3b_report_consistency,
    )
    from joint_agent.articulation_v2_static_run_plan import (
        ArticulationV2StaticRunPlanV1,
    )
    from joint_agent.internal.articulation_v2_static_gate3_closeout import (
        ARTICULATION_V2_STATIC_GATE3B_APPROVED_FEATURES,
        _require_generated_result_identity,
    )

    plan = cast(ArticulationV2StaticRunPlanV1, inputs.plan)
    generated = cast(
        ArticulationV2StaticGeneratedArtifactIdentitiesV1,
        inputs.generated,
    )
    stage = plan.stages.gate3b
    report = inputs.report
    if report.get("schema_version") != stage.admission.validation_contract:
        raise VerifiedOperationError("Gate 3B report uses an unsupported contract")
    profile = report.get("profile")
    profile_id = profile.get("target") if isinstance(profile, dict) else None
    if profile_id != stage.admission.profile_id:
        raise VerifiedOperationError("Gate 3B report lacks exact admitted profile")
    feature_ids = ARTICULATION_V2_STATIC_GATE3B_APPROVED_FEATURES.get(profile_id)
    if feature_ids is None:
        raise VerifiedOperationError("Gate 3B profile has no approved feature roster")
    validate_articulation_v2_static_gate3b_report_consistency(
        report,
        profile_id=profile_id,
        feature_ids=feature_ids,
    )
    results = report.get("results")
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
    ):
        raise VerifiedOperationError("Gate 3B projector requires exactly one result")
    result = results[0]
    _require_generated_result_identity(
        result,
        plan=plan,
        generated_identities=generated,
        label="Gate 3B",
    )
    if result.get("profile_target") != stage.admission.profile_id:
        raise VerifiedOperationError("Gate 3B result profile differs from admission")
    status = {
        "PASS": "pass",
        "FAIL": "fail",
        "BLOCKED": "blocked",
        "ERROR": "error",
        "NOT_RUN": "not_evaluated",
    }.get(result.get("status"))
    if status is None:
        raise VerifiedOperationError("Gate 3B result has an unsupported status")
    contract = Path(__file__).resolve()
    gate_contract = Path(
        __import__(
            "joint_agent.articulation_v2_static_gate3b_contracts",
            fromlist=["__file__"],
        ).__file__
    ).resolve()
    foundation_commit = result.get("foundation_commit")
    if status in {"pass", "fail"} and (
        not isinstance(foundation_commit, str) or not foundation_commit
    ):
        raise VerifiedOperationError(
            "Gate 3B evaluated result lacks an exact foundation commit"
        )
    backend_version = str(foundation_commit or "not-evaluated")
    return _publish(
        output_dir=output_dir,
        operation_id="joint.gate3b-simready-foundation",
        gate_id="joint.gate3b",
        evidence_type="joint.simready-foundation-validation",
        native_report_type="joint.gate3b-report",
        native_payload_type="joint.gate3-closeout",
        claim_scope="Joint Gate 3B SimReady Foundation profile result",
        native_status=status,
        source=inputs.source,
        output=inputs.output,
        dependencies=inputs.dependencies,
        artifacts=(
            inputs.plan_binding,
            inputs.intake_binding,
            inputs.authoring_receipt_binding,
        ),
        native_report=inputs.report_binding,
        native_payload=inputs.closeout_binding,
        producer=_component(
            component_id="joint-gate3b-report-producer",
            version=stage.admission.validation_contract,
            contract=gate_contract,
            configuration=inputs.closeout_binding,
        ),
        tool=_component(
            component_id=stage.command.tool_id,
            version=stage.command.tool_version,
            contract=gate_contract,
            configuration=inputs.plan_binding,
        ),
        profile=_component(
            component_id=profile_id,
            version=stage.admission.validation_contract,
            contract=gate_contract,
            configuration=inputs.report_binding,
        ),
        backend=_component(
            component_id="simready-foundation-runtime",
            version=backend_version,
            contract=gate_contract,
            configuration=inputs.report_binding,
        ),
        verifier=_component(
            component_id="joint-gate3b-report-consistency-verifier",
            version=stage.admission.validation_contract,
            contract=gate_contract,
            configuration=inputs.report_binding,
        ),
        projector=_component(
            component_id="joint-gate3b-verified-operation-projector",
            version="content-agent-workflows.joint-projector.v1",
            contract=contract,
            configuration=inputs.report_binding,
        ),
    )


def project_joint_dynamic_result(
    *,
    source_path: str | Path,
    output_path: str | Path,
    receipt_path: str | Path,
    profile_id: str,
    artifact_map_path: str | Path,
    output_dir: str | Path,
) -> JointVerifiedOperationPublication:
    """Re-verify captured dynamic evidence and project the trusted result."""

    from joint_agent.dynamic_qualification.admission import (
        _captured_profile_file,
        admit_dynamic_profile,
    )
    from joint_agent.dynamic_qualification.contracts import (
        DynamicQualificationReceiptV1,
    )
    from joint_agent.dynamic_qualification.verification import (
        verify_dynamic_qualification,
    )

    source = execution_artifact_binding(source_path)
    output = execution_artifact_binding(output_path)
    if source != output:
        raise VerifiedOperationError(
            "Joint dynamic source must be the exact verified authored input"
        )
    dependencies = _dependency_bindings(output)
    receipt_binding = execution_artifact_binding(receipt_path)
    receipt = _load_bound_model(
        receipt_binding,
        DynamicQualificationReceiptV1,
        label="Joint dynamic qualification receipt",
    )
    artifact_map_binding = execution_artifact_binding(artifact_map_path)
    artifact_map = _load_bound_model(
        artifact_map_binding,
        JointDynamicArtifactMap,
        label="Joint dynamic artifact map",
    )
    resolver = _LocalArtifactResolver(artifact_map.artifacts)
    with admit_dynamic_profile(profile_id) as admitted:
        verified = verify_dynamic_qualification(
            admitted,
            receipt,
            resolver=resolver,
        )
        record = verified.record
        profile_capture = _captured_profile_file(admitted)
        profile_payload = b"".join(profile_capture.iter_chunks())
        try:
            profile_text = profile_payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VerifiedOperationError(
                "admitted dynamic profile is not canonical UTF-8 JSON"
            ) from exc
        profile_capture.require_intact()
    output_claims = tuple(
        item
        for item in record.artifacts
        if item.source == "input"
        and item.sha256 == output.sha256
        and item.size_bytes == output.size_bytes
        and Path(artifact_map.artifacts.get(item.uri, "")).expanduser().resolve()
        == Path(output.path).resolve()
    )
    if not output_claims:
        raise VerifiedOperationError(
            "trusted dynamic result does not bind the projected output"
        )
    claimed_by_uri = {item.uri: item for item in record.artifacts}
    profile_claims = tuple(
        item for item in record.artifacts if item.source == "profile"
    )
    if len(profile_claims) != 1:
        raise VerifiedOperationError(
            "trusted dynamic result lacks one admitted profile artifact"
        )
    mapped_claims = {
        uri: claim for uri, claim in claimed_by_uri.items() if claim.source != "profile"
    }
    if set(artifact_map.artifacts) != set(mapped_claims):
        raise VerifiedOperationError(
            "dynamic artifact map differs from the trusted result artifact roster"
        )
    mapped_artifacts = tuple(
        execution_artifact_binding(artifact_map.artifacts[uri])
        for uri in sorted(mapped_claims)
    )
    for uri, binding in zip(sorted(mapped_claims), mapped_artifacts, strict=True):
        claim = mapped_claims[uri]
        if binding.sha256 != claim.sha256 or binding.size_bytes != claim.size_bytes:
            raise VerifiedOperationError(
                "dynamic artifact bytes differ from the trusted result"
            )
    profile_claim = profile_claims[0]
    if (
        profile_capture.sha256 != profile_claim.sha256
        or profile_capture.size_bytes != profile_claim.size_bytes
    ):
        raise VerifiedOperationError(
            "admitted dynamic profile differs from the trusted result"
        )
    root = Path(output_dir).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise VerifiedOperationError(f"Joint projector output already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    profile_path = root / "joint_dynamic_admitted_profile.json"
    atomic_write_text(profile_path, profile_text)
    profile_binding = execution_artifact_binding(profile_path)
    if (
        profile_binding.sha256 != profile_claim.sha256
        or profile_binding.size_bytes != profile_claim.size_bytes
    ):
        raise VerifiedOperationError(
            "persisted dynamic profile differs from admitted profile"
        )
    captured_artifacts = (profile_binding, *mapped_artifacts)
    record_path = root / "joint_dynamic_verified_result.json"
    atomic_write_json(record_path, record)
    record_binding = execution_artifact_binding(record_path)
    # _publish owns creation of its directory, so retain the verified record in a
    # sibling and use a dedicated child for the shared projection artifacts.
    projection_root = root / "projection"
    status = {
        "PASS": "pass",
        "FAIL": "fail",
        "NOT_RUN": "not_evaluated",
        "NA": "blocked",
    }.get(record.status)
    if status is None:
        raise VerifiedOperationError(
            "dynamic qualification result has an unsupported status"
        )
    status = cast(VerifiedNativeStatus, status)
    verification_module = __import__(
        "joint_agent.dynamic_qualification.verification",
        fromlist=["__file__"],
    )
    admission_module = __import__(
        "joint_agent.dynamic_qualification.admission",
        fromlist=["__file__"],
    )
    contract = Path(__file__).resolve()
    runtime_version = (
        f"{record.runtime.runtime_version}:{record.runtime.implementation_sha256}"
    )
    return _publish(
        output_dir=projection_root,
        operation_id="joint.dynamic-qualification",
        gate_id="joint.dynamic",
        evidence_type="joint.dynamic-qualification",
        native_report_type="joint.dynamic-result",
        native_payload_type=None,
        claim_scope="trusted Joint dynamic qualification result",
        native_status=status,
        source=source,
        output=output,
        dependencies=dependencies,
        artifacts=(receipt_binding, artifact_map_binding, *captured_artifacts),
        native_report=record_binding,
        native_payload=None,
        producer=_component(
            component_id="joint-dynamic-qualification-verifier",
            version="joint-dynamic-result-v1",
            contract=Path(verification_module.__file__).resolve(),
            configuration=receipt_binding,
        ),
        tool=_component(
            component_id="joint-dynamic-evidence-adapter",
            version=record.adapter.adapter_version,
            contract=Path(verification_module.__file__).resolve(),
            configuration=record_binding,
        ),
        profile=_component(
            component_id=record.profile_id,
            version=record.capability_manifest_version,
            contract=Path(admission_module.__file__).resolve(),
            configuration=record_binding,
        ),
        backend=_component(
            component_id=record.runtime.runtime_id,
            version=runtime_version,
            contract=Path(verification_module.__file__).resolve(),
            configuration=record_binding,
        ),
        verifier=_component(
            component_id="joint-dynamic-trusted-verifier",
            version="joint-dynamic-verification.v1",
            contract=Path(verification_module.__file__).resolve(),
            configuration=record_binding,
        ),
        projector=_component(
            component_id="joint-dynamic-verified-operation-projector",
            version="content-agent-workflows.joint-projector.v1",
            contract=contract,
            configuration=record_binding,
        ),
    )


__all__ = [
    "JOINT_DYNAMIC_ARTIFACT_MAP_SCHEMA_VERSION",
    "JOINT_VERIFIED_OPERATION_PUBLICATION_SCHEMA_VERSION",
    "VERIFIED_OPERATION_CONTRACT_CHECKPOINT",
    "JointDynamicArtifactMap",
    "JointVerifiedOperationPublication",
    "project_joint_dynamic_result",
    "project_joint_gate3a_result",
    "project_joint_gate3b_result",
    "project_joint_graph_apply_result",
]
