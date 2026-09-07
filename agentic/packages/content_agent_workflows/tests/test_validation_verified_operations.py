# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

import content_agent_workflows.validation.verified_operations as verified_operations_module
from content_agent_workflows.asset_composition import ArtifactBinding
from content_agent_workflows.asset_composition.catalog_adapters import (
    PROVIDED_VALIDATION_INGRESS_LEAF_ID,
    ProvidedValidationIngressLeafInvocation,
    ProvidedValidationIngressLeafResult,
    shared_asset_leaf_runtime_bundle,
)
from content_agent_workflows.common.artifacts import atomic_write_json
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)
from content_agent_workflows.validation import (
    EXECUTED_VALIDATION_ARTIFACT_NAMES,
    VerifiedOperationAssessmentDisposition,
    VerifiedOperationComponentIdentity,
    VerifiedOperationCoordinatorAssessment,
    VerifiedOperationError,
    VerifiedOperationEvidenceIndex,
    VerifiedValidationOperationEnvelope,
    VerifiedValidationOperationProjection,
    assess_verified_operation_evidence,
    collect_verified_operation_evidence,
    execution_artifact_binding,
    ingest_verified_operation_result,
    is_verified_operation_ingest_run,
    load_verified_operation_terminal_receipt,
    review_verified_operation_assessment,
)
from content_agent_workflows.validation.embedded_assessment import (
    ValidationCoordinatorReviewDraft,
)
from content_agent_workflows.validation.verified_operations import (
    VerifiedOperationDisposition,
    VerifiedTerminalDisposition,
)


def _write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _component(root: Path, name: str) -> VerifiedOperationComponentIdentity:
    return VerifiedOperationComponentIdentity(
        component_id=f"test.{name}",
        version="test.contract.v1",
        contract=execution_artifact_binding(
            _write(root / f"{name}_contract.py", f"# {name}\n")
        ),
        configuration=execution_artifact_binding(
            _write(root / f"{name}_config.json", f'{{"name":"{name}"}}\n')
        ),
    )


def _make_envelope(
    root: Path,
    *,
    operation_id: str = "physics.mass-properties.verified",
    gate_id: str = "physics.mass-properties",
    status: str = "pass",
    authority: str = "deterministic_fact",
    required: bool = True,
) -> tuple[Path, VerifiedValidationOperationEnvelope, dict[str, Path]]:
    root.mkdir(parents=True)
    paths = {
        "source": _write(root / "source.usda", "#usda 1.0\n"),
        "output": _write(root / "output.usda", '#usda 1.0\ndef Xform "A" {}\n'),
        "dependency": _write(root / "texture.png", "image-bytes"),
        "artifact": _write(root / "runtime.log", "verified runtime\n"),
        "report": _write(root / "report.json", '{"mass": 1.0, "passed": true}\n'),
        "payload": _write(root / "payload.json", '{"native": "physics"}\n'),
    }
    components = {
        name: _component(root, name)
        for name in ("producer", "tool", "profile", "backend", "verifier", "projector")
    }
    for name, component in components.items():
        paths[f"{name}_contract"] = Path(component.contract.path)
        assert component.configuration is not None
        paths[f"{name}_config"] = Path(component.configuration.path)
    values = {
        "operation_id": operation_id,
        "gate_id": gate_id,
        "evidence_type": "physics.mass-properties-report",
        "native_report_type": "physics.mass-properties-report",
        "native_payload_type": "physics.mass-properties-payload",
        "claim_scope": "exact mass-property report for one authored USD",
        "native_status": status,
        "required": required,
        "authority": authority,
        "source": execution_artifact_binding(paths["source"]),
        "output": execution_artifact_binding(paths["output"]),
        "dependencies": (execution_artifact_binding(paths["dependency"]),),
        "artifacts": (execution_artifact_binding(paths["artifact"]),),
        "native_report": execution_artifact_binding(paths["report"]),
        "native_payload": execution_artifact_binding(paths["payload"]),
        "producer_identity_sha256": canonical_json_digest(components["producer"]),
        "tool_identity_sha256": canonical_json_digest(components["tool"]),
        "profile_identity_sha256": canonical_json_digest(components["profile"]),
        "backend_identity_sha256": canonical_json_digest(components["backend"]),
        "verifier_identity_sha256": canonical_json_digest(components["verifier"]),
        "projector_identity_sha256": canonical_json_digest(components["projector"]),
    }
    projection = VerifiedValidationOperationProjection.model_validate(values)
    projection_path = root / "projection.json"
    atomic_write_json(projection_path, projection)
    envelope = VerifiedValidationOperationEnvelope(
        **{
            key: value
            for key, value in values.items()
            if not key.endswith("_identity_sha256")
        },
        producer=components["producer"],
        tool=components["tool"],
        profile=components["profile"],
        backend=components["backend"],
        verifier=components["verifier"],
        projector=components["projector"],
        projection=execution_artifact_binding(projection_path),
    )
    envelope_path = root / "envelope.json"
    atomic_write_json(envelope_path, envelope)
    return envelope_path, envelope, paths


def _write_assessment(
    path: Path,
    evidence: VerifiedOperationEvidenceIndex,
    *,
    dispositions: tuple[VerifiedOperationDisposition, ...],
    terminal: VerifiedTerminalDisposition = "pass",
) -> Path:
    records = evidence.records
    assessment = VerifiedOperationCoordinatorAssessment(
        assessment_id="outer-assessment-001",
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        assessment_identity_sha256=evidence.assessment_identity_sha256,
        operations=tuple(
            VerifiedOperationAssessmentDisposition(
                operation_id=record.operation_id,
                gate_id=record.gate_id,
                ingest_receipt_sha256=record.ingest_receipt.sha256,
                native_status=record.native_status,
                disposition=disposition,
                rationale="Outer exact-byte review of the native result.",
            )
            for record, disposition in zip(records, dispositions, strict=True)
        ),
        terminal_disposition=terminal,
        summary="Outer organizer accepted each required native operation.",
    )
    atomic_write_json(path, assessment)
    return path


def test_provided_operations_survive_independently_through_terminal_receipt(
    tmp_path: Path,
) -> None:
    first_path, _, _ = _make_envelope(tmp_path / "first")
    second_path, _, _ = _make_envelope(
        tmp_path / "second",
        operation_id="material.package-integrity.verified",
        gate_id="material.package-integrity",
        status="warn",
    )
    run_dir = tmp_path / "run"
    ingest_verified_operation_result(first_path, output_dir=run_dir)
    ingest_verified_operation_result(second_path, output_dir=run_dir)

    evidence = collect_verified_operation_evidence(run_dir)
    assert evidence.execution_mode == "provided"
    assert [record.gate_id for record in evidence.records] == [
        "physics.mass-properties",
        "material.package-integrity",
    ]
    assert evidence.records[0].native_report_type == "physics.mass-properties-report"
    assessment_path = _write_assessment(
        tmp_path / "assessment.json",
        evidence,
        dispositions=("pass", "waive"),
    )
    execution = assess_verified_operation_evidence(
        run_dir,
        assessment_path=assessment_path,
    )
    assert execution.execution_mode == "provided"
    assert execution.authorized is True
    review_path = tmp_path / "review.json"
    atomic_write_json(
        review_path,
        ValidationCoordinatorReviewDraft(
            created_at=datetime(2026, 8, 12, tzinfo=UTC),
            disposition="accept",
            findings=("Exact-digest outer review accepted both operations.",),
        ),
    )
    terminal = review_verified_operation_assessment(run_dir, review_path=review_path)

    assert terminal.mode == "provided"
    assert [item.disposition for item in terminal.operation_dispositions] == [
        "pass",
        "waive",
    ]
    assert terminal.operation_evidence[0].native_report_type == (
        "physics.mass-properties-report"
    )
    assert terminal.operation_evidence[0].native_payload is not None
    assert terminal.operation_evidence[0].producer == evidence.records[0].producer
    assert terminal.operation_evidence[0].backend == evidence.records[0].backend
    assert terminal.operation_evidence[0].projector == evidence.records[0].projector
    assert terminal.operation_evidence[0].ingest_receipt.sha256 == (
        terminal.operation_dispositions[0].ingest_receipt_sha256
    )
    assert set(terminal.gate_dispositions.values()) == {"not_evaluated"}
    assert terminal.receipt_status == "completed"
    assert load_verified_operation_terminal_receipt(run_dir) == terminal


def test_provided_asset_projection_retains_external_envelope_in_local_receipt(
    tmp_path: Path,
) -> None:
    envelope_path, envelope, _paths = _make_envelope(tmp_path / "graph-a")
    attempt = tmp_path / "graph-b" / "attempt"
    output_dir = attempt / "native"
    receipt = ingest_verified_operation_result(envelope_path, output_dir=output_dir)
    noncanonical_envelope_path = (
        envelope_path.parent / ".." / envelope_path.parent.name / envelope_path.name
    )
    assert str(noncanonical_envelope_path) != receipt.envelope_binding.path
    invocation = ProvidedValidationIngressLeafInvocation(
        envelope_path=str(noncanonical_envelope_path),
        output_dir=str(output_dir),
    )
    result = ProvidedValidationIngressLeafResult(root=receipt)
    invocation_path = attempt / "invocation.json"
    result_path = attempt / "result.json"
    atomic_write_json(invocation_path, invocation)
    atomic_write_json(result_path, result)

    binding = next(
        candidate
        for candidate in shared_asset_leaf_runtime_bundle().bindings
        if candidate.descriptor.leaf_id == PROVIDED_VALIDATION_INGRESS_LEAF_ID
    )
    projection = binding.project(
        invocation,
        result,
        invocation_artifact=ArtifactBinding.model_validate(
            execution_artifact_binding(invocation_path).model_dump(mode="json")
        ),
        result_artifact=ArtifactBinding.model_validate(
            execution_artifact_binding(result_path).model_dump(mode="json")
        ),
    ).payload

    local_receipt = (
        output_dir
        / "verified_operations"
        / receipt.envelope_contract_sha256
        / "ingest_receipt.json"
    ).resolve()
    local_index = (output_dir / "verified_operation_ingest_index.json").resolve()
    assert Path(projection.native_terminal_receipt.path) == local_receipt
    assert [Path(item.path) for item in projection.operation_indexes] == [local_index]
    assert [Path(item.path) for item in projection.evidence] == [local_receipt]
    assert [Path(item.path) for item in projection.saved_stage_readbacks] == [
        local_receipt,
        local_index,
    ]
    assert all(
        Path(item.path).is_relative_to(attempt.resolve())
        for item in (
            projection.native_terminal_receipt,
            *projection.operation_indexes,
            *projection.evidence,
            *projection.saved_stage_readbacks,
        )
    )
    assert receipt.envelope_binding.path == str(envelope_path.resolve())
    assert receipt.envelope == envelope

    local_index.unlink()
    with pytest.raises(
        ValueError,
        match="local ingest artifacts are missing or invalid",
    ):
        binding.project(
            invocation,
            result,
            invocation_artifact=ArtifactBinding.model_validate(
                execution_artifact_binding(invocation_path).model_dump(mode="json")
            ),
            result_artifact=ArtifactBinding.model_validate(
                execution_artifact_binding(result_path).model_dump(mode="json")
            ),
        )


@pytest.mark.parametrize(
    "review_disposition",
    ("reject", "revise", "retry", "stop", "cancelled"),
)
def test_non_accept_review_preserves_semantics_separately_from_process_status(
    tmp_path: Path,
    review_disposition: Literal["reject", "revise", "retry", "stop", "cancelled"],
) -> None:
    envelope_path, _, _ = _make_envelope(tmp_path / "provided")
    run_dir = tmp_path / "run"
    ingest_verified_operation_result(envelope_path, output_dir=run_dir)
    evidence = collect_verified_operation_evidence(run_dir)
    assessment_path = _write_assessment(
        tmp_path / "assessment.json",
        evidence,
        dispositions=("pass",),
    )
    assess_verified_operation_evidence(run_dir, assessment_path=assessment_path)
    review_path = tmp_path / "review.json"
    atomic_write_json(
        review_path,
        ValidationCoordinatorReviewDraft(
            created_at=datetime(2026, 8, 12, tzinfo=UTC),
            disposition=review_disposition,
            findings=("Outer review did not accept the exact assessment.",),
        ),
    )

    terminal = review_verified_operation_assessment(run_dir, review_path=review_path)

    assert terminal.receipt_status == "completed"
    assert terminal.review_disposition == review_disposition
    assert terminal.terminal_disposition == "pass"
    assert load_verified_operation_terminal_receipt(run_dir) == terminal


def test_terminal_readback_rejects_mismatched_independent_disposition(
    tmp_path: Path,
) -> None:
    envelope_path, _, _ = _make_envelope(tmp_path / "provided")
    run_dir = tmp_path / "run"
    ingest_verified_operation_result(envelope_path, output_dir=run_dir)
    evidence = collect_verified_operation_evidence(run_dir)
    assessment_path = _write_assessment(
        tmp_path / "assessment.json",
        evidence,
        dispositions=("pass",),
    )
    assess_verified_operation_evidence(run_dir, assessment_path=assessment_path)
    review_path = tmp_path / "review.json"
    atomic_write_json(
        review_path,
        ValidationCoordinatorReviewDraft(
            created_at=datetime(2026, 8, 12, tzinfo=UTC),
            disposition="accept",
            findings=("Exact-digest outer review accepted the operation.",),
        ),
    )
    review_verified_operation_assessment(run_dir, review_path=review_path)
    terminal_path = run_dir / "validation_terminal_receipt.json"
    tampered = json.loads(terminal_path.read_text(encoding="utf-8"))
    tampered["operation_dispositions"][0]["gate_id"] = "physics.changed-gate"
    atomic_write_json(terminal_path, tampered)

    with pytest.raises(VerifiedOperationError, match="terminal receipt"):
        load_verified_operation_terminal_receipt(run_dir)


@pytest.mark.parametrize(
    "artifact_name",
    (
        "source",
        "output",
        "dependency",
        "artifact",
        "report",
        "payload",
        "producer_contract",
        "projector_config",
        "verifier_contract",
    ),
)
def test_ingest_rejects_stale_bound_artifact(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    envelope_path, _, paths = _make_envelope(tmp_path / "provided")
    paths[artifact_name].write_text("altered bytes", encoding="utf-8")

    with pytest.raises(VerifiedOperationError, match="stale"):
        ingest_verified_operation_result(envelope_path, output_dir=tmp_path / "run")


def test_ingest_streams_unparsed_artifact_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope_path, _, paths = _make_envelope(tmp_path / "provided")
    observed_reads: list[tuple[Path, bool]] = []
    original_read = verified_operations_module._read_regular_file

    def tracking_read(
        path: Path,
        *,
        capture_bytes: bool,
    ) -> tuple[bytes | None, str, os.stat_result]:
        observed_reads.append((path, capture_bytes))
        return original_read(path, capture_bytes=capture_bytes)

    monkeypatch.setattr(
        verified_operations_module,
        "_read_regular_file",
        tracking_read,
    )

    ingest_verified_operation_result(envelope_path, output_dir=tmp_path / "run")

    artifact_reads = [
        capture for path, capture in observed_reads if path == paths["artifact"]
    ]
    assert artifact_reads
    assert all(capture is False for capture in artifact_reads)


def test_binding_reverification_rejects_same_inode_same_size_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "large-evidence.bin"
    artifact.write_bytes(b"a" * (verified_operations_module._CHUNK_SIZE + 17))
    binding = verified_operations_module.execution_artifact_binding(artifact)
    before = artifact.stat()
    original_read = os.read
    mutated = False

    def mutating_read(fd: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(fd, size)
        if chunk and not mutated:
            mutated = True
            with artifact.open("r+b", buffering=0) as stream:
                stream.write(b"b")
                os.fsync(stream.fileno())
            os.utime(
                artifact,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
        return chunk

    monkeypatch.setattr(os, "read", mutating_read)

    with pytest.raises(
        VerifiedOperationError,
        match="verified artifact identity changed while reading",
    ):
        verified_operations_module.verify_execution_artifact_binding(binding)

    after = artifact.stat()
    assert mutated is True
    assert (after.st_dev, after.st_ino, after.st_size) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
    )


def test_binding_cache_hashes_once_and_rejects_later_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _envelope_path, envelope, paths = _make_envelope(tmp_path / "provided")
    original_verify = (
        verified_operations_module._verify_execution_artifact_binding_streaming
    )
    artifact_hashes = 0

    def tracking_verify(binding, *, label: str):
        nonlocal artifact_hashes
        if Path(binding.path) == paths["artifact"]:
            artifact_hashes += 1
        return original_verify(binding, label=label)

    monkeypatch.setattr(
        verified_operations_module,
        "_verify_execution_artifact_binding_streaming",
        tracking_verify,
    )
    cache = verified_operations_module.VerifiedOperationBindingCache()

    verified_operations_module.verify_operation_envelope(
        envelope,
        binding_cache=cache,
    )
    verified_operations_module.verify_operation_envelope(
        envelope,
        binding_cache=cache,
    )
    assert artifact_hashes == 1

    paths["artifact"].write_text("changed runtime\n", encoding="utf-8")
    with pytest.raises(VerifiedOperationError, match="changed after verification"):
        cache.assert_unchanged()


def test_ingest_rejects_symlinked_bound_artifact(tmp_path: Path) -> None:
    envelope_path, _, paths = _make_envelope(tmp_path / "provided")
    source = paths["source"]
    source.unlink()
    try:
        source.symlink_to(_write(tmp_path / "replacement.usda", "#usda 1.0\n"))
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")

    with pytest.raises(VerifiedOperationError, match="unsafe"):
        ingest_verified_operation_result(envelope_path, output_dir=tmp_path / "run")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("operation_id", "physics.mass-properties.changed"),
        ("gate_id", "physics.mass-properties-changed"),
        ("native_status", "warn"),
        ("claim_scope", "altered claim scope"),
    ),
)
def test_ingest_rejects_projection_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    envelope_path, envelope, _ = _make_envelope(tmp_path / "provided")
    atomic_write_json(envelope_path, envelope.model_copy(update={field: value}))

    with pytest.raises(VerifiedOperationError, match="projection differs"):
        ingest_verified_operation_result(envelope_path, output_dir=tmp_path / "run")


def test_ingest_rejects_altered_producer_identity(tmp_path: Path) -> None:
    envelope_path, envelope, _ = _make_envelope(tmp_path / "provided")
    changed = envelope.producer.model_copy(update={"component_id": "test.changed"})
    atomic_write_json(envelope_path, envelope.model_copy(update={"producer": changed}))

    with pytest.raises(VerifiedOperationError, match="projection differs"):
        ingest_verified_operation_result(envelope_path, output_dir=tmp_path / "run")


@pytest.mark.parametrize(
    "field",
    ("source", "output", "dependencies", "native_report", "native_payload"),
)
def test_ingest_rejects_exact_binding_that_differs_from_projection(
    tmp_path: Path,
    field: str,
) -> None:
    envelope_path, envelope, _ = _make_envelope(tmp_path / "provided")
    replacement = execution_artifact_binding(
        _write(tmp_path / f"replacement-{field}.bin", f"replacement {field}\n")
    )
    changed: object = (replacement,) if field == "dependencies" else replacement
    atomic_write_json(envelope_path, envelope.model_copy(update={field: changed}))

    with pytest.raises(VerifiedOperationError, match="projection differs"):
        ingest_verified_operation_result(envelope_path, output_dir=tmp_path / "run")


@pytest.mark.parametrize("duplicate", ("operation", "gate"))
def test_ingest_rejects_duplicate_ids_before_writing_receipt(
    tmp_path: Path,
    duplicate: str,
) -> None:
    first_path, _, _ = _make_envelope(tmp_path / "first")
    second_path, _, _ = _make_envelope(
        tmp_path / "second",
        operation_id=(
            "physics.mass-properties.verified"
            if duplicate == "operation"
            else "physics.collision.verified"
        ),
        gate_id=(
            "physics.mass-properties" if duplicate == "gate" else "physics.collision"
        ),
    )
    run_dir = tmp_path / "run"
    ingest_verified_operation_result(first_path, output_dir=run_dir)

    with pytest.raises(VerifiedOperationError, match=f"{duplicate} ID"):
        ingest_verified_operation_result(second_path, output_dir=run_dir)
    assert len(tuple((run_dir / "verified_operations").iterdir())) == 1


@pytest.mark.parametrize("artifact_name", EXECUTED_VALIDATION_ARTIFACT_NAMES)
def test_ingest_rejects_every_execute_mode_artifact(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    envelope_path, _, _ = _make_envelope(tmp_path / "provided")
    run_dir = tmp_path / "run"
    artifact_path = run_dir / artifact_name
    if artifact_path.suffix:
        _write(artifact_path, "{}\n")
    else:
        artifact_path.mkdir(parents=True)

    with pytest.raises(VerifiedOperationError, match="mutually exclusive"):
        ingest_verified_operation_result(envelope_path, output_dir=run_dir)


def test_incomplete_provided_mode_never_falls_back_to_execute(tmp_path: Path) -> None:
    envelope_path, _, _ = _make_envelope(tmp_path / "provided")
    run_dir = tmp_path / "run"
    ingest_verified_operation_result(envelope_path, output_dir=run_dir)
    (run_dir / "verified_operation_ingest_index.json").unlink()

    assert is_verified_operation_ingest_run(run_dir) is True
    with pytest.raises(VerifiedOperationError, match="missing or unsafe"):
        collect_verified_operation_evidence(run_dir)


def test_provided_mode_rejects_symlinked_run_root_on_later_steps(
    tmp_path: Path,
) -> None:
    envelope_path, _, _ = _make_envelope(tmp_path / "provided")
    run_dir = tmp_path / "run"
    ingest_verified_operation_result(envelope_path, output_dir=run_dir)
    alias = tmp_path / "run-alias"
    try:
        alias.symlink_to(run_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")

    with pytest.raises(VerifiedOperationError, match="missing or unsafe"):
        is_verified_operation_ingest_run(alias)
    with pytest.raises(VerifiedOperationError, match="missing or unsafe"):
        collect_verified_operation_evidence(alias)


def test_ingest_never_imports_or_calls_execution_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope_path, _, _ = _make_envelope(tmp_path / "provided")
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        forbidden = ("usd_rendering", "look_right", "simulation")
        if any(token in name for token in forbidden):
            raise AssertionError(f"ingest imported execution dependency {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    receipt = ingest_verified_operation_result(
        envelope_path,
        output_dir=tmp_path / "run",
    )
    assert receipt.execution_mode == "provided"
    assert receipt.nested_agent_launched is False


def test_outer_assessment_cannot_relabel_unevaluated_or_advisory_as_pass(
    tmp_path: Path,
) -> None:
    envelope_path, _, _ = _make_envelope(
        tmp_path / "provided",
        status="not_evaluated",
        authority="advisory_critique",
        required=False,
    )
    run_dir = tmp_path / "run"
    ingest_verified_operation_result(envelope_path, output_dir=run_dir)
    evidence = collect_verified_operation_evidence(run_dir)
    assessment_path = _write_assessment(
        tmp_path / "assessment.json",
        evidence,
        dispositions=("pass",),
    )

    with pytest.raises(VerifiedOperationError, match="advisory operation"):
        assess_verified_operation_evidence(run_dir, assessment_path=assessment_path)


def test_outer_assessment_cannot_relabel_unevaluated_operation_as_pass(
    tmp_path: Path,
) -> None:
    envelope_path, _, _ = _make_envelope(
        tmp_path / "provided",
        status="not_evaluated",
        required=False,
    )
    run_dir = tmp_path / "run"
    ingest_verified_operation_result(envelope_path, output_dir=run_dir)
    evidence = collect_verified_operation_evidence(run_dir)
    assessment_path = _write_assessment(
        tmp_path / "assessment.json",
        evidence,
        dispositions=("pass",),
    )

    with pytest.raises(VerifiedOperationError, match="unevaluated operation"):
        assess_verified_operation_evidence(run_dir, assessment_path=assessment_path)


def test_path_only_legacy_evidence_is_not_a_canonical_envelope(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy_validation_evidence.json"
    legacy_path.write_text(
        '{"schema_version":"content-agent-workflows.validation-evidence.v1",'
        '"artifacts":[],"checks":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        VerifiedOperationError, match="invalid verified operation envelope"
    ):
        ingest_verified_operation_result(legacy_path, output_dir=tmp_path / "run")
