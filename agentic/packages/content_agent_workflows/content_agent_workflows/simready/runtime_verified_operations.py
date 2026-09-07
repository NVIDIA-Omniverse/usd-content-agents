# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-executing SimReady runtime projectors for verified-operation ingress."""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue

from content_agent_workflows.common.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    read_contained_artifact,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)
from content_agent_workflows.validation.verified_operations import (
    VerifiedNativeStatus,
    VerifiedOperationBindingCache,
    VerifiedOperationComponentIdentity,
    VerifiedOperationError,
    VerifiedValidationOperationEnvelope,
    VerifiedValidationOperationProjection,
    execution_artifact_binding,
    verify_execution_artifact_binding,
    verify_operation_envelope,
)

from .runtime_benchmark import (
    SIMREADY_BENCHMARK_NATIVE_REPORT_SCHEMA_VERSION,
    SIMREADY_BENCHMARK_VERSION,
    SIMREADY_RUNTIME_PUBLICATION_INVALIDATED_NAME,
    SIMREADY_RUNTIME_PUBLICATION_VALIDITY_NAME,
    SIMREADY_RUNTIME_VALIDATION_SCHEMA_VERSION,
    SimReadyRuntimeCheckResult,
    SimReadyRuntimeValidationReport,
    _normalize_benchmark_result,
    _parse_runtime_error_events,
    _runtime_input_binding_errors,
    native_benchmark_checks,
)

SIMREADY_RUNTIME_CHECK_PAYLOAD_SCHEMA_VERSION: Final = (
    "content-agent-workflows.simready-runtime-check-payload.v1"
)
SIMREADY_RUNTIME_VERIFIED_OPERATIONS_SCHEMA_VERSION: Final = (
    "content-agent-workflows.simready-runtime-verified-operations.v1"
)
SIMREADY_RUNTIME_VERIFIED_OPERATIONS_NAME: Final = (
    "simready_runtime_verified_operations.json"
)
_MAX_EXECUTION_CONTRACT_BYTES: Final = 4 * 1024 * 1024
_SAFE_ID_CHARS = re.compile(r"[^a-z0-9]+")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SimReadyRuntimeCheckPayload(_FrozenModel):
    """Exact native test entry and its non-lossy WU interpretation."""

    schema_version: Literal[
        "content-agent-workflows.simready-runtime-check-payload.v1"
    ] = SIMREADY_RUNTIME_CHECK_PAYLOAD_SCHEMA_VERSION
    operation_id: str
    gate_id: str
    check: SimReadyRuntimeCheckResult
    native_result: dict[str, Any]
    runtime_status: VerifiedNativeStatus
    runtime_errors: tuple[str, ...] = ()
    runtime_error_events: tuple[dict[str, Any], ...] = ()


class SimReadyRuntimeVerifiedOperationPublication(_FrozenModel):
    """One independently ingestible native benchmark test result."""

    payload: ExecutionArtifactBinding
    projection: ExecutionArtifactBinding
    envelope: ExecutionArtifactBinding
    result: VerifiedValidationOperationEnvelope


class SimReadyRuntimeVerifiedOperationsPublication(_FrozenModel):
    """Index of all native tests projected from one completed benchmark report."""

    schema_version: Literal[
        "content-agent-workflows.simready-runtime-verified-operations.v1"
    ] = SIMREADY_RUNTIME_VERIFIED_OPERATIONS_SCHEMA_VERSION
    runtime_report: ExecutionArtifactBinding
    native_report: ExecutionArtifactBinding
    operations: tuple[SimReadyRuntimeVerifiedOperationPublication, ...]
    projector_benchmark_invoked: Literal[False] = False
    projector_simulator_invoked: Literal[False] = False
    projector_provider_invoked: Literal[False] = False
    projector_nested_agent_launched: Literal[False] = False


def project_simready_runtime_verified_operations(
    report_path: str | Path,
    *,
    output_dir: str | Path,
) -> SimReadyRuntimeVerifiedOperationsPublication:
    """Project retained runtime results without launching Benchmark or a simulator."""

    _reject_invalidated_publication(
        Path(output_dir) / SIMREADY_RUNTIME_VERIFIED_OPERATIONS_NAME
    )
    report_binding = execution_artifact_binding(report_path)
    report = _load_bound_model(
        report_binding,
        SimReadyRuntimeValidationReport,
        label="SimReady runtime report",
    )
    if report.schema_version != SIMREADY_RUNTIME_VALIDATION_SCHEMA_VERSION:
        raise VerifiedOperationError("unsupported SimReady runtime report contract")
    if report.report_path != report_binding.path:
        raise VerifiedOperationError(
            "SimReady runtime report path differs from its retained identity"
        )
    if not report.checks:
        raise VerifiedOperationError(
            "SimReady runtime report contains no native checks to project"
        )
    if report.native_report_path is None:
        raise VerifiedOperationError("SimReady runtime report lacks its native report")

    native_report_binding = _bound_report_artifact(
        report,
        key="native_report",
        path=report.native_report_path,
        label="native SimReady Benchmark report",
    )
    native_report = _load_json_object(
        native_report_binding,
        label="native SimReady Benchmark report",
    )
    _validate_projectable_runtime_contract(report, native_report)
    native_plan_binding, native_plan = _native_plan(report)
    _validate_retained_runtime_evidence(
        report,
        native_report=native_report,
        native_plan=native_plan,
    )
    raw_results = _native_results(
        native_report,
        report.checks,
        expected_asset_path=report.asset_path,
        native_plan=native_plan,
    )
    source, dependencies = _asset_bindings(report)
    tool_contract = _benchmark_contract(report)
    common_artifacts = _common_artifacts(
        report,
        report_binding,
        native_report,
        native_plan_binding,
    )
    root = _fresh_output_root(output_dir)
    validity_binding = _publication_validity_binding(root, create=True)
    runtime_input_artifacts = _runtime_input_artifacts(
        report,
        root=root,
        create_manifest=True,
    )
    adapter_contract, projector_contract = _snapshot_execution_contracts(root)

    publications: list[SimReadyRuntimeVerifiedOperationPublication] = []
    operation_ids: set[str] = set()
    gate_ids: set[str] = set()
    verification_cache = VerifiedOperationBindingCache()
    for index, (check, native_result) in enumerate(
        zip(report.checks, raw_results, strict=True)
    ):
        operation_id, gate_id = _operation_identity(check)
        if operation_id in operation_ids or gate_id in gate_ids:
            raise VerifiedOperationError(
                "SimReady runtime report contains duplicate native check identities"
            )
        operation_ids.add(operation_id)
        gate_ids.add(gate_id)
        check_root = root / f"{index:04d}-{operation_id.rsplit('.', 1)[-1]}"
        check_root.mkdir(parents=False, exist_ok=False)
        payload = SimReadyRuntimeCheckPayload(
            operation_id=operation_id,
            gate_id=gate_id,
            check=check,
            native_result=native_result,
            runtime_status=cast(VerifiedNativeStatus, report.status),
            runtime_errors=tuple(report.errors),
            runtime_error_events=tuple(report.runtime_error_events),
        )
        payload_path = check_root / "simready_runtime_check_payload.json"
        atomic_write_json(payload_path, payload)
        payload_binding = execution_artifact_binding(payload_path)
        test_artifacts = _test_artifacts(
            report=report,
            check=check,
            native_result=native_result,
        )
        publications.append(
            _publish_check(
                root=check_root,
                operation_id=operation_id,
                gate_id=gate_id,
                check=check,
                source=source,
                dependencies=dependencies,
                artifacts=_dedupe_bindings(
                    (
                        report_binding,
                        validity_binding,
                        *runtime_input_artifacts,
                        *common_artifacts,
                        *test_artifacts,
                    )
                ),
                native_report=native_report_binding,
                payload=payload_binding,
                tool_contract=tool_contract,
                adapter_contract=adapter_contract,
                projector_contract=projector_contract,
                runtime_report=report_binding,
                report=report,
                binding_cache=verification_cache,
            )
        )

    verification_cache.assert_unchanged()
    publication = SimReadyRuntimeVerifiedOperationsPublication(
        runtime_report=report_binding,
        native_report=native_report_binding,
        operations=tuple(publications),
    )
    index_path = root / SIMREADY_RUNTIME_VERIFIED_OPERATIONS_NAME
    atomic_write_json(index_path, publication)
    return load_simready_runtime_verified_operations(index_path)


def load_simready_runtime_verified_operations(
    path: str | Path,
) -> SimReadyRuntimeVerifiedOperationsPublication:
    """Load and reverify every byte retained by one projector publication."""

    _reject_invalidated_publication(path)
    index_binding = execution_artifact_binding(path)
    publication = _load_bound_model(
        index_binding,
        SimReadyRuntimeVerifiedOperationsPublication,
        label="SimReady runtime verified-operation index",
    )
    report = _load_bound_model(
        publication.runtime_report,
        SimReadyRuntimeValidationReport,
        label="SimReady runtime report",
    )
    if report.schema_version != SIMREADY_RUNTIME_VALIDATION_SCHEMA_VERSION:
        raise VerifiedOperationError("unsupported SimReady runtime report contract")
    if report.report_path != publication.runtime_report.path:
        raise VerifiedOperationError(
            "SimReady runtime report path differs from its publication binding"
        )
    if report.native_report_path is None:
        raise VerifiedOperationError("SimReady runtime report lacks its native report")
    expected_native_report = _bound_report_artifact(
        report,
        key="native_report",
        path=report.native_report_path,
        label="native SimReady Benchmark report",
    )
    if publication.native_report != expected_native_report:
        raise VerifiedOperationError(
            "SimReady publication native report differs from its runtime report"
        )
    native_report = _load_json_object(
        publication.native_report,
        label="native SimReady Benchmark report",
    )
    _validate_projectable_runtime_contract(report, native_report)
    native_plan_binding, native_plan = _native_plan(report)
    _validate_retained_runtime_evidence(
        report,
        native_report=native_report,
        native_plan=native_plan,
    )
    raw_results = _native_results(
        native_report,
        report.checks,
        expected_asset_path=report.asset_path,
        native_plan=native_plan,
    )
    source, dependencies = _asset_bindings(report)
    tool_contract = _benchmark_contract(report)
    common_artifacts = _common_artifacts(
        report,
        publication.runtime_report,
        native_report,
        native_plan_binding,
    )
    publication_root = Path(path).expanduser().resolve().parent
    validity_binding = _publication_validity_binding(
        publication_root,
        create=False,
    )
    runtime_input_artifacts = _runtime_input_artifacts(
        report,
        root=publication_root,
        create_manifest=False,
    )
    if len(publication.operations) != len(report.checks):
        raise VerifiedOperationError(
            "SimReady publication does not cover every native runtime check"
        )
    operation_ids: set[str] = set()
    gate_ids: set[str] = set()
    verification_cache = VerifiedOperationBindingCache()
    adapter_contract: ExecutionArtifactBinding | None = None
    projector_contract: ExecutionArtifactBinding | None = None
    for operation, check, native_result in zip(
        publication.operations,
        report.checks,
        raw_results,
        strict=True,
    ):
        payload = _load_bound_model(
            operation.payload,
            SimReadyRuntimeCheckPayload,
            label="SimReady runtime check payload",
        )
        projection = _load_bound_model(
            operation.projection,
            VerifiedValidationOperationProjection,
            label="SimReady verified-operation projection",
        )
        envelope = _load_bound_model(
            operation.envelope,
            VerifiedValidationOperationEnvelope,
            label="SimReady verified-operation envelope",
        )
        if adapter_contract is None:
            adapter_contract = _verify_execution_contract_snapshot(
                envelope.producer.contract,
                Path(__file__).with_name("runtime_benchmark.py"),
                label="SimReady runtime adapter",
                binding_cache=verification_cache,
            )
            projector_contract = _verify_execution_contract_snapshot(
                envelope.projector.contract,
                Path(__file__),
                label="SimReady runtime projector",
                binding_cache=verification_cache,
            )
        assert projector_contract is not None
        expected_operation_id, expected_gate_id = _operation_identity(check)
        expected_artifacts = _dedupe_bindings(
            (
                publication.runtime_report,
                validity_binding,
                *runtime_input_artifacts,
                *common_artifacts,
                *_test_artifacts(
                    report=report,
                    check=check,
                    native_result=native_result,
                ),
            )
        )
        producer, tool, profile, backend, verifier, projector = _operation_components(
            check=check,
            native_report=publication.native_report,
            payload=operation.payload,
            tool_contract=tool_contract,
            adapter_contract=adapter_contract,
            projector_contract=projector_contract,
            runtime_report=publication.runtime_report,
            report=report,
        )
        if (
            envelope != operation.result
            or payload.operation_id != expected_operation_id
            or payload.gate_id != expected_gate_id
            or payload.check != check
            or payload.native_result != native_result
            or payload.runtime_status != report.status
            or payload.runtime_errors != tuple(report.errors)
            or payload.runtime_error_events != tuple(report.runtime_error_events)
            or payload.operation_id != envelope.operation_id
            or payload.gate_id != envelope.gate_id
            or projection.operation_id != envelope.operation_id
            or projection.gate_id != envelope.gate_id
            or envelope.projection != operation.projection
            or envelope.native_status != check.disposition
            or envelope.evidence_type != "simready.runtime-test"
            or envelope.native_report_type != "simready.benchmark-report"
            or envelope.native_payload_type != "simready.runtime-check-payload"
            or envelope.claim_scope != _claim_scope(check)
            or envelope.required is not True
            or envelope.authority != "deterministic_fact"
            or envelope.source != source
            or envelope.output != source
            or envelope.dependencies != dependencies
            or envelope.artifacts != expected_artifacts
            or envelope.native_report != publication.native_report
            or envelope.native_payload != operation.payload
            or envelope.producer != producer
            or envelope.tool != tool
            or envelope.profile != profile
            or envelope.backend != backend
            or envelope.verifier != verifier
            or envelope.projector != projector
        ):
            raise VerifiedOperationError(
                "SimReady verified-operation index differs from its retained operation"
            )
        if envelope.operation_id in operation_ids or envelope.gate_id in gate_ids:
            raise VerifiedOperationError(
                "SimReady publication contains duplicate native check identities"
            )
        operation_ids.add(envelope.operation_id)
        gate_ids.add(envelope.gate_id)
        verify_operation_envelope(
            envelope,
            binding_cache=verification_cache,
        )
    verification_cache.assert_unchanged()
    return publication


def _publish_check(
    *,
    root: Path,
    operation_id: str,
    gate_id: str,
    check: SimReadyRuntimeCheckResult,
    source: ExecutionArtifactBinding,
    dependencies: tuple[ExecutionArtifactBinding, ...],
    artifacts: tuple[ExecutionArtifactBinding, ...],
    native_report: ExecutionArtifactBinding,
    payload: ExecutionArtifactBinding,
    tool_contract: ExecutionArtifactBinding,
    adapter_contract: ExecutionArtifactBinding,
    projector_contract: ExecutionArtifactBinding,
    runtime_report: ExecutionArtifactBinding,
    report: SimReadyRuntimeValidationReport,
    binding_cache: VerifiedOperationBindingCache,
) -> SimReadyRuntimeVerifiedOperationPublication:
    producer, tool, profile, backend, verifier, projector = _operation_components(
        check=check,
        native_report=native_report,
        payload=payload,
        tool_contract=tool_contract,
        adapter_contract=adapter_contract,
        projector_contract=projector_contract,
        runtime_report=runtime_report,
        report=report,
    )
    values = {
        "operation_id": operation_id,
        "gate_id": gate_id,
        "evidence_type": "simready.runtime-test",
        "native_report_type": "simready.benchmark-report",
        "native_payload_type": "simready.runtime-check-payload",
        "claim_scope": _claim_scope(check),
        "native_status": cast(VerifiedNativeStatus, check.disposition),
        "required": True,
        "authority": "deterministic_fact",
        "source": source,
        "output": source,
        "dependencies": dependencies,
        "artifacts": artifacts,
        "native_report": native_report,
        "native_payload": payload,
    }
    projection = VerifiedValidationOperationProjection.model_validate(
        {
            **values,
            "producer_identity_sha256": canonical_json_digest(producer),
            "tool_identity_sha256": canonical_json_digest(tool),
            "profile_identity_sha256": canonical_json_digest(profile),
            "backend_identity_sha256": (
                canonical_json_digest(backend) if backend is not None else None
            ),
            "verifier_identity_sha256": canonical_json_digest(verifier),
            "projector_identity_sha256": canonical_json_digest(projector),
        }
    )
    projection_path = root / "verified_operation_projection.json"
    atomic_write_json(projection_path, projection)
    projection_binding = execution_artifact_binding(projection_path)
    envelope = VerifiedValidationOperationEnvelope.model_validate(
        {
            **values,
            "producer": producer,
            "tool": tool,
            "profile": profile,
            "backend": backend,
            "verifier": verifier,
            "projector": projector,
            "projection": projection_binding,
        }
    )
    envelope_path = root / "verified_operation_envelope.json"
    atomic_write_json(envelope_path, envelope)
    envelope_binding = execution_artifact_binding(envelope_path)
    verified = verify_operation_envelope(
        _load_bound_model(
            envelope_binding,
            VerifiedValidationOperationEnvelope,
            label="SimReady verified-operation envelope",
        ),
        binding_cache=binding_cache,
    )
    return SimReadyRuntimeVerifiedOperationPublication(
        payload=payload,
        projection=projection_binding,
        envelope=envelope_binding,
        result=verified,
    )


def _snapshot_execution_contracts(
    root: Path,
) -> tuple[ExecutionArtifactBinding, ExecutionArtifactBinding]:
    contracts_root = root / "execution-contracts"
    contracts_root.mkdir(parents=False, exist_ok=False)
    bindings: list[ExecutionArtifactBinding] = []
    for source, name, label in (
        (
            Path(__file__).with_name("runtime_benchmark.py"),
            "runtime_benchmark.py",
            "SimReady runtime adapter",
        ),
        (
            Path(__file__),
            "runtime_verified_operations.py",
            "SimReady runtime projector",
        ),
    ):
        payload, _digest, _size = _read_execution_contract(source, label=label)
        snapshot = contracts_root / name
        atomic_write_bytes(snapshot, payload, within=root)
        bindings.append(execution_artifact_binding(snapshot))
    return bindings[0], bindings[1]


def _verify_execution_contract_snapshot(
    binding: ExecutionArtifactBinding,
    source: Path,
    *,
    label: str,
    binding_cache: VerifiedOperationBindingCache,
) -> ExecutionArtifactBinding:
    binding_cache.verify(binding, label=f"{label} contract snapshot")
    _payload, digest, size = _read_execution_contract(source, label=label)
    if binding.sha256 != digest or binding.size_bytes != size:
        raise VerifiedOperationError(
            f"{label} contract snapshot differs from the executing source"
        )
    return binding


def _read_execution_contract(
    source: Path,
    *,
    label: str,
) -> tuple[bytes, str, int]:
    try:
        resolved = source.expanduser().resolve(strict=True)
        captured = read_contained_artifact(
            resolved.parent,
            resolved,
            max_bytes=_MAX_EXECUTION_CONTRACT_BYTES,
            capture_bytes=True,
            allow_hardlinks=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise VerifiedOperationError(
            f"could not bind {label} execution contract: {exc}"
        ) from exc
    if captured.data is None:
        raise VerifiedOperationError(f"could not capture {label} execution contract")
    return captured.data, captured.sha256, captured.size_bytes


def _operation_components(
    *,
    check: SimReadyRuntimeCheckResult,
    native_report: ExecutionArtifactBinding,
    payload: ExecutionArtifactBinding,
    tool_contract: ExecutionArtifactBinding,
    adapter_contract: ExecutionArtifactBinding,
    projector_contract: ExecutionArtifactBinding,
    runtime_report: ExecutionArtifactBinding,
    report: SimReadyRuntimeValidationReport,
) -> tuple[
    VerifiedOperationComponentIdentity,
    VerifiedOperationComponentIdentity,
    VerifiedOperationComponentIdentity,
    VerifiedOperationComponentIdentity | None,
    VerifiedOperationComponentIdentity,
    VerifiedOperationComponentIdentity,
]:
    producer = _component(
        component_id="simready-runtime-adapter",
        version=SIMREADY_RUNTIME_VALIDATION_SCHEMA_VERSION,
        contract=adapter_contract,
        configuration=runtime_report,
    )
    tool = VerifiedOperationComponentIdentity(
        component_id="simready-benchmark",
        version=report.benchmark_version or "unknown",
        contract=tool_contract,
        configuration=native_report,
    )
    profile = VerifiedOperationComponentIdentity(
        component_id=_component_id("simready-profile", check.profile_id),
        version=check.profile_version or "unspecified",
        contract=adapter_contract,
        configuration=payload,
    )
    backend = (
        VerifiedOperationComponentIdentity(
            component_id=_component_id("simready-backend", check.engine),
            version=check.engine_version or "unspecified",
            contract=tool_contract,
            configuration=payload,
        )
        if check.engine
        else None
    )
    verifier = _component(
        component_id="simready-runtime-result-verifier",
        version=SIMREADY_RUNTIME_VALIDATION_SCHEMA_VERSION,
        contract=adapter_contract,
        configuration=runtime_report,
    )
    projector = VerifiedOperationComponentIdentity(
        component_id="simready-runtime-verified-operation-projector",
        version=SIMREADY_RUNTIME_VERIFIED_OPERATIONS_SCHEMA_VERSION,
        contract=projector_contract,
        configuration=payload,
    )
    return producer, tool, profile, backend, verifier, projector


def _claim_scope(check: SimReadyRuntimeCheckResult) -> str:
    return (
        f"SimReady Benchmark {check.profile_id}/{check.feature_id}/"
        f"{check.test_name} result for the exact source asset"
    )


def _component(
    *,
    component_id: str,
    version: str,
    contract: ExecutionArtifactBinding,
    configuration: ExecutionArtifactBinding,
) -> VerifiedOperationComponentIdentity:
    return VerifiedOperationComponentIdentity(
        component_id=component_id,
        version=version,
        contract=contract,
        configuration=configuration,
    )


def _component_id(prefix: str, value: str | None) -> str:
    slug = _slug(value or "unspecified")
    return f"{prefix}-{slug}"


def _operation_identity(check: SimReadyRuntimeCheckResult) -> tuple[str, str]:
    identity: dict[str, JsonValue] = {
        "profile_id": check.profile_id,
        "profile_version": check.profile_version,
        "feature_id": check.feature_id,
        "feature_version": check.feature_version,
        "test_name": check.test_name,
        "test_version": check.test_version,
        "engine": check.engine,
        "engine_version": check.engine_version,
    }
    digest = canonical_json_digest(identity)
    suffix = f"{_slug(check.test_name)[:48].rstrip('-')}-{digest[:12]}"
    return (
        f"simready.runtime-check.{suffix}",
        f"simready.runtime-gate.{suffix}",
    )


def _slug(value: str) -> str:
    return _SAFE_ID_CHARS.sub("-", value.lower()).strip("-") or "unspecified"


def _reject_invalidated_publication(path: str | Path) -> None:
    publication_path = Path(path).expanduser().resolve()
    marker = publication_path.parent / SIMREADY_RUNTIME_PUBLICATION_INVALIDATED_NAME
    if marker.is_symlink() or marker.exists():
        raise VerifiedOperationError(
            "SimReady verified-operation publication was invalidated by a newer run"
        )


def _validate_projectable_runtime_contract(
    report: SimReadyRuntimeValidationReport,
    native_report: Mapping[str, Any],
) -> None:
    metadata = native_report.get("metadata")
    if not isinstance(metadata, Mapping):
        raise VerifiedOperationError("native SimReady Benchmark metadata is malformed")
    native_schema = str(metadata.get("schema_version") or "")
    if (
        report.native_report_schema_version
        != SIMREADY_BENCHMARK_NATIVE_REPORT_SCHEMA_VERSION
        or native_schema != SIMREADY_BENCHMARK_NATIVE_REPORT_SCHEMA_VERSION
    ):
        raise VerifiedOperationError(
            "native SimReady Benchmark schema differs from the pinned contract"
        )
    native_framework = str(metadata.get("framework_version") or "")
    if (
        report.benchmark_version != SIMREADY_BENCHMARK_VERSION
        or native_framework != SIMREADY_BENCHMARK_VERSION
    ):
        raise VerifiedOperationError(
            "native SimReady Benchmark framework differs from the pinned contract"
        )
    if report.status not in {"pass", "warn", "fail", "not_evaluated"}:
        raise VerifiedOperationError(
            f"SimReady runtime status is not publishable: {report.status}"
        )


def _validate_retained_runtime_evidence(
    report: SimReadyRuntimeValidationReport,
    *,
    native_report: dict[str, Any],
    native_plan: dict[str, Any],
) -> None:
    if report.run_summary_path is None:
        raise VerifiedOperationError("SimReady runtime report lacks its run summary")
    if report.events_path is None:
        raise VerifiedOperationError("SimReady runtime report lacks its event stream")

    run_summary = _load_json_object(
        _bound_report_artifact(
            report,
            key="run_summary",
            path=report.run_summary_path,
            label="run summary",
        ),
        label="SimReady Benchmark run summary",
    )
    events_binding = _bound_report_artifact(
        report,
        key="events",
        path=report.events_path,
        label="event stream",
    )
    events_payload = verify_execution_artifact_binding(
        events_binding,
        label="SimReady Benchmark event stream",
    )
    try:
        runtime_error_events = _parse_runtime_error_events(
            events_payload.decode("utf-8")
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise VerifiedOperationError(
            f"SimReady Benchmark event stream is malformed: {exc}"
        ) from exc

    retained_status, retained_checks, _warnings, retained_errors = (
        _normalize_benchmark_result(
            exit_code=report.exit_code,
            process_error=None,
            expected_asset_path=Path(report.asset_path),
            native_plan=native_plan,
            native_plan_error=None,
            native_report=native_report,
            native_report_error=None,
            run_summary=run_summary,
            run_summary_error=None,
            runtime_error_events=runtime_error_events,
            events_error=None,
        )
    )
    if retained_errors:
        raise VerifiedOperationError(
            "retained SimReady Benchmark evidence cannot be projected: "
            + "; ".join(retained_errors)
        )
    if retained_status != report.status:
        raise VerifiedOperationError(
            "SimReady runtime status differs from the retained native evidence: "
            f"reported={report.status}, derived={retained_status}"
        )
    if retained_checks != report.checks:
        raise VerifiedOperationError(
            "SimReady runtime checks differ from the retained native evidence"
        )
    if runtime_error_events != report.runtime_error_events:
        raise VerifiedOperationError(
            "SimReady runtime error events differ from the retained event stream"
        )
    if report.errors:
        raise VerifiedOperationError(
            "publishable SimReady runtime report retains adapter errors"
        )
    if report.passed != (retained_status == "pass"):
        raise VerifiedOperationError(
            "SimReady runtime passed flag differs from the retained native evidence"
        )


def _native_results(
    native_report: Mapping[str, Any],
    expected_checks: Sequence[SimReadyRuntimeCheckResult],
    *,
    expected_asset_path: str,
    native_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    normalized, errors, _validation_failed = native_benchmark_checks(
        native_report,
        expected_asset_path=expected_asset_path,
        native_plan=native_plan,
    )
    if errors:
        raise VerifiedOperationError(
            "native SimReady Benchmark report cannot be projected: " + "; ".join(errors)
        )
    if normalized != list(expected_checks):
        raise VerifiedOperationError(
            "SimReady runtime report checks differ from the native report"
        )
    raw_results: list[dict[str, Any]] = []
    assets = native_report.get("assets")
    assert isinstance(assets, Mapping)
    for asset in assets.values():
        assert isinstance(asset, Mapping)
        profiles = asset.get("profiles")
        assert isinstance(profiles, list)
        for profile in profiles:
            assert isinstance(profile, Mapping)
            features = profile.get("features")
            assert isinstance(features, list)
            for feature in features:
                assert isinstance(feature, Mapping)
                tests = feature.get("tests") or []
                assert isinstance(tests, list)
                raw_results.extend(dict(test) for test in tests)
    return raw_results


def _native_plan(
    report: SimReadyRuntimeValidationReport,
) -> tuple[ExecutionArtifactBinding, dict[str, Any]]:
    if report.native_plan_path is None:
        raise VerifiedOperationError("SimReady runtime report lacks its native plan")
    binding = _bound_report_artifact(
        report,
        key="native_plan",
        path=report.native_plan_path,
        label="native SimReady Benchmark plan",
    )
    return binding, _load_json_object(
        binding,
        label="native SimReady Benchmark plan",
    )


def _asset_bindings(
    report: SimReadyRuntimeValidationReport,
) -> tuple[ExecutionArtifactBinding, tuple[ExecutionArtifactBinding, ...]]:
    files = report.asset_dependency_manifest.get("files")
    if not isinstance(files, list):
        raise VerifiedOperationError("runtime report lacks an asset dependency roster")
    roots: list[ExecutionArtifactBinding] = []
    dependencies: list[ExecutionArtifactBinding] = []
    for item in files:
        if not isinstance(item, Mapping):
            raise VerifiedOperationError("asset dependency roster is malformed")
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("size")
        role = item.get("role")
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not isinstance(size, int)
            or role not in {"root", "dependency"}
        ):
            raise VerifiedOperationError("asset dependency identity is malformed")
        binding = execution_artifact_binding(path)
        if binding.sha256 != digest or binding.size_bytes != size:
            raise VerifiedOperationError(
                f"asset dependency differs from the runtime report: {path}"
            )
        (roots if role == "root" else dependencies).append(binding)
    if len(roots) != 1 or roots[0].path != report.asset_path:
        raise VerifiedOperationError("runtime report lacks one exact source asset")
    if roots[0].sha256 != report.asset_sha256:
        raise VerifiedOperationError("runtime source digest differs from its report")
    return roots[0], tuple(dependencies)


def _benchmark_contract(
    report: SimReadyRuntimeValidationReport,
) -> ExecutionArtifactBinding:
    if (
        report.benchmark_executable is None
        or report.benchmark_executable_sha256 is None
    ):
        raise VerifiedOperationError(
            "runtime report lacks benchmark executable identity"
        )
    binding = execution_artifact_binding(report.benchmark_executable)
    if binding.sha256 != report.benchmark_executable_sha256:
        raise VerifiedOperationError(
            "benchmark executable differs from its runtime report"
        )
    return binding


def _command_values(command: Sequence[str], flag: str) -> tuple[str, ...]:
    if flag not in command:
        return ()
    start = command.index(flag) + 1
    end = next(
        (
            index
            for index in range(start, len(command))
            if command[index].startswith("--")
        ),
        len(command),
    )
    return tuple(command[start:end])


def _publication_validity_binding(
    root: Path,
    *,
    create: bool,
) -> ExecutionArtifactBinding:
    path = root / SIMREADY_RUNTIME_PUBLICATION_VALIDITY_NAME
    if create:
        atomic_write_json(
            path,
            {
                "schema_version": (
                    "content-agent-workflows.simready-publication-validity.v1"
                ),
                "status": "valid",
                # A unique generation prevents an invalidated envelope from becoming
                # valid again when the same output directory is reused.
                "generation": secrets.token_hex(32),
            },
        )
    elif not path.is_file():
        raise VerifiedOperationError("runtime publication lacks its validity token")
    return execution_artifact_binding(path)


def _runtime_input_artifacts(
    report: SimReadyRuntimeValidationReport,
    *,
    root: Path,
    create_manifest: bool,
) -> tuple[ExecutionArtifactBinding, ...]:
    expected_paths: dict[str, str] = {}
    for label, flag in (
        ("sr_specs_path", "--sr-specs"),
        ("engines_toml_path", "--engines-toml"),
    ):
        values = _command_values(report.command, flag)
        if len(values) != 1:
            raise VerifiedOperationError(
                f"runtime command lacks exactly one {flag} input"
            )
        expected_paths[label] = values[0]
    project_config = _command_values(report.command, "--project-config")
    if len(project_config) > 1:
        raise VerifiedOperationError(
            "runtime command has more than one --project-config input"
        )
    if project_config:
        expected_paths["project_config_path"] = project_config[0]
    for index, path in enumerate(_command_values(report.command, "--tests-path")):
        expected_paths[f"tests_paths[{index}]"] = path

    bindings_by_label = {
        binding.label: binding for binding in report.runtime_input_bindings
    }
    if len(bindings_by_label) != len(report.runtime_input_bindings):
        raise VerifiedOperationError("runtime input binding labels are not unique")
    if set(bindings_by_label) != set(expected_paths):
        raise VerifiedOperationError(
            "runtime input bindings differ from the benchmark command inputs"
        )
    for label, expected_path in expected_paths.items():
        binding = bindings_by_label[label]
        expected_kind = (
            "directory"
            if label == "sr_specs_path" or label.startswith("tests_paths[")
            else "file"
        )
        if binding.path != expected_path or binding.kind != expected_kind:
            raise VerifiedOperationError(
                f"runtime input binding differs from command input {label}"
            )

    identity_errors = _runtime_input_binding_errors(report.runtime_input_bindings)
    if identity_errors:
        raise VerifiedOperationError("; ".join(identity_errors))

    manifest_path = root / "runtime_input_bindings.json"
    manifest = {
        "schema_version": (
            "content-agent-workflows.simready-runtime-input-bindings.v1"
        ),
        "inputs": [
            binding.model_dump(mode="json") for binding in report.runtime_input_bindings
        ],
    }
    if create_manifest:
        atomic_write_json(manifest_path, manifest)
    else:
        retained_manifest = _load_json_object(
            execution_artifact_binding(manifest_path),
            label="SimReady runtime input binding manifest",
        )
        if retained_manifest != manifest:
            raise VerifiedOperationError(
                "runtime input binding manifest differs from the runtime report"
            )
    flattened = [
        file_binding
        for input_binding in report.runtime_input_bindings
        for file_binding in input_binding.files
    ]
    return _dedupe_bindings((execution_artifact_binding(manifest_path), *flattened))


def _common_artifacts(
    report: SimReadyRuntimeValidationReport,
    report_binding: ExecutionArtifactBinding,
    native_report: Mapping[str, Any],
    native_plan: ExecutionArtifactBinding,
) -> tuple[ExecutionArtifactBinding, ...]:
    bindings = [report_binding, native_plan]
    paths = {
        "run_summary": report.run_summary_path,
        "events": report.events_path,
        "stdout": report.stdout_log_path,
        "stderr": report.stderr_log_path,
    }
    for key, path in paths.items():
        if path is not None:
            bindings.append(
                _bound_report_artifact(report, key=key, path=path, label=key)
            )
    native_root = _native_root(report)
    metadata = native_report.get("metadata")
    sessions = metadata.get("sessions") if isinstance(metadata, Mapping) else None
    if isinstance(sessions, list):
        for session in sessions:
            if not isinstance(session, Mapping):
                raise VerifiedOperationError("native benchmark session is malformed")
            log_file = session.get("log_file")
            if log_file:
                bindings.append(
                    execution_artifact_binding(
                        _native_relative_file(native_root, str(log_file), label="log")
                    )
                )
    return _dedupe_bindings(bindings)


def _test_artifacts(
    *,
    report: SimReadyRuntimeValidationReport,
    check: SimReadyRuntimeCheckResult,
    native_result: Mapping[str, Any],
) -> tuple[ExecutionArtifactBinding, ...]:
    native_root = _native_root(report)
    asset_key = PurePosixPath(check.asset_key)
    if asset_key.is_absolute() or ".." in asset_key.parts or not asset_key.name:
        raise VerifiedOperationError("native benchmark asset key is unsafe")
    test_name = PurePosixPath(check.test_name)
    if (
        test_name.is_absolute()
        or len(test_name.parts) != 1
        or test_name.name in {"", ".", ".."}
    ):
        raise VerifiedOperationError("native benchmark test name is unsafe")
    # Benchmark 2026.6.5 keys per-test storage by (asset, test), while engine
    # selection is a filter rather than a test cross-product. Ingress rejects a
    # duplicate asset/test roster before this canonical path is resolved.
    test_root = (
        native_root
        / "results"
        / Path(*asset_key.parent.parts)
        / ".simready"
        / "runtime"
        / test_name.name
    )
    result_path = test_root / "result.json"
    result_binding = execution_artifact_binding(result_path)
    result = _load_json_object(result_binding, label="native per-test result")
    result_media = _mapping_roster(result.get("media"))
    result_kit_logs = _list_roster(result.get("kit_logs"))
    if (
        _aggregate_native_status(result.get("status")) != check.native_status
        or str(result.get("test_name") or "") != check.test_name
        or str(result.get("test_version") or "") != check.test_version
        or str(result.get("asset") or "") != check.asset_key
        or _optional_text(result.get("engine")) != check.engine
        or _optional_text(result.get("engine_version")) != check.engine_version
        or _optional_number(result.get("duration")) != check.duration_s
        or str(result.get("message") or "") != check.message
        or _mapping_value(result.get("metrics")) != check.metrics
        or _canonical_roster(result_media) != _canonical_roster(check.media)
        or _canonical_roster(result_kit_logs) != _canonical_roster(check.kit_logs)
        or any(
            str(result.get(field) or "") != str(native_result.get(field) or "")
            for field in ("description", "expected_video", "scene_file")
        )
    ):
        raise VerifiedOperationError(
            "native per-test result differs from the aggregate benchmark report"
        )
    bindings = [result_binding]
    media = native_result.get("media") or []
    if not isinstance(media, list):
        raise VerifiedOperationError("native benchmark media roster is malformed")
    for item in media:
        if not isinstance(item, Mapping) or not isinstance(item.get("filename"), str):
            raise VerifiedOperationError("native benchmark media entry is malformed")
        bindings.append(
            execution_artifact_binding(
                _native_relative_file(
                    test_root,
                    cast(str, item["filename"]),
                    label="test media",
                )
            )
        )
    return _dedupe_bindings(bindings)


def _aggregate_native_status(value: Any) -> str:
    status = str(value or "")
    if status in {"pass", "fail", "skipped", "incomplete", "blocked"}:
        return status
    if status == "error":
        return "fail"
    return "incomplete"


def _optional_text(value: Any) -> str | None:
    return None if value is None or value == "" else str(value)


def _optional_number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _mapping_value(value: Any) -> dict[str, Any] | None:
    if value is None:
        return {}
    return dict(value) if isinstance(value, Mapping) else None


def _mapping_roster(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        return None
    return [dict(item) for item in value]


def _list_roster(value: Any) -> list[Any] | None:
    if value is None:
        return []
    return value if isinstance(value, list) else None


def _canonical_roster(value: Sequence[Any] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    try:
        return tuple(
            sorted(
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in value
            )
        )
    except (TypeError, ValueError):
        return None


def _native_root(report: SimReadyRuntimeValidationReport) -> Path:
    if report.native_output_dir is None:
        raise VerifiedOperationError("runtime report lacks its native output root")
    root = Path(report.native_output_dir).expanduser()
    if not root.is_absolute():
        raise VerifiedOperationError("native benchmark output root must be absolute")
    return root


def _native_relative_file(root: Path, value: str, *, label: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise VerifiedOperationError(
            f"native benchmark {label} path is unsafe: {value}"
        )
    candidate = root / Path(*relative.parts)
    if not candidate.is_file():
        raise VerifiedOperationError(
            f"native benchmark {label} artifact is missing: {candidate}"
        )
    return candidate


def _bound_report_artifact(
    report: SimReadyRuntimeValidationReport,
    *,
    key: str,
    path: str,
    label: str,
) -> ExecutionArtifactBinding:
    expected = report.artifact_sha256.get(key)
    if expected is None:
        raise VerifiedOperationError(f"runtime report lacks {label} artifact identity")
    binding = execution_artifact_binding(path)
    if binding.sha256 != expected:
        raise VerifiedOperationError(f"{label} artifact differs from runtime report")
    return binding


def _load_json_object(
    binding: ExecutionArtifactBinding,
    *,
    label: str,
) -> dict[str, Any]:
    payload = verify_execution_artifact_binding(binding, label=label)
    try:
        result = json.loads(payload)
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError) as exc:
        raise VerifiedOperationError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise VerifiedOperationError(f"{label} must be a JSON object")
    return result


def _load_bound_model[ModelT: BaseModel](
    binding: ExecutionArtifactBinding,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    payload = verify_execution_artifact_binding(binding, label=label)
    try:
        return model.model_validate_json(payload)
    except ValueError as exc:
        raise VerifiedOperationError(f"invalid {label}: {exc}") from exc


def _fresh_output_root(output_dir: str | Path) -> Path:
    raw_root = Path(output_dir).expanduser()
    if raw_root.is_symlink():
        raise VerifiedOperationError(
            f"SimReady projector output is a symlink: {raw_root}"
        )
    root = raw_root.resolve()
    if root.exists():
        raise VerifiedOperationError(
            f"SimReady projector output already exists: {root}"
        )
    root.mkdir(parents=True, exist_ok=False)
    return root


def _dedupe_bindings(
    bindings: Sequence[ExecutionArtifactBinding],
) -> tuple[ExecutionArtifactBinding, ...]:
    unique: dict[tuple[str, str, int], ExecutionArtifactBinding] = {}
    for binding in bindings:
        unique[(binding.path, binding.sha256, binding.size_bytes)] = binding
    return tuple(unique.values())


__all__ = [
    "SIMREADY_RUNTIME_CHECK_PAYLOAD_SCHEMA_VERSION",
    "SIMREADY_RUNTIME_VERIFIED_OPERATIONS_NAME",
    "SIMREADY_RUNTIME_VERIFIED_OPERATIONS_SCHEMA_VERSION",
    "SimReadyRuntimeCheckPayload",
    "SimReadyRuntimeVerifiedOperationPublication",
    "SimReadyRuntimeVerifiedOperationsPublication",
    "load_simready_runtime_verified_operations",
    "project_simready_runtime_verified_operations",
]
