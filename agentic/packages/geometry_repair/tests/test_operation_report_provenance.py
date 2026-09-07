# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable operation-report and accepted-backend provenance tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import trimesh

import geometry_repair.orchestrator as orchestrator_module
from geometry_repair import RepairRequest, run_geometry_repair
from geometry_repair.artifacts import atomic_write_json, file_sha256
from geometry_repair.models import AttemptRecord, RepairOperation, RepairPlan
from geometry_repair.orchestrator import _verified_worker_report
from geometry_repair.workers.base import WorkerResult

_BACKEND_IDENTITY = {
    "backend_id": "openvdb",
    "driver_distribution": "sdf-tools",
    "implementation_version": "13.0.0",
    "library_version": [13, 0, 0],
}
_QUALIFICATION_ID = "geometry-repair.openvdb13.v1"


def test_sdf_cross_layer_contract_is_explicit_fail_closed_and_digest_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "cube.stl"
    trimesh.creation.box().export(source)

    unavailable_calls: list[str] = []

    def unavailable_plan(request, diagnosis, normalized_source, plan_path):
        assert request.enabled_workers == ["sdf_rebuild"]
        operation = RepairOperation(
            operation_id="attempt-00-unavailable-sdf-rebuild",
            worker="openvdb_rebuild",
            implementation="test-qualified-sdf-backend",
            parameters={"backend_id": "openvdb"},
            drift_band="identity",
            source_checkpoint=str(normalized_source),
        )
        assert operation.worker == "sdf_rebuild"
        plan = RepairPlan(
            source_sha256=diagnosis.source_sha256,
            profile=request.profile,
            operations=[operation],
            budgets=request.budgets,
            deterministic_seed=request.deterministic_seed,
            rationale="Exercise explicit unavailable-without-fallback behavior.",
            plan_path=str(plan_path),
        )
        atomic_write_json(plan_path, plan)
        return plan

    def unavailable_sdf_worker(**kwargs) -> WorkerResult:
        operation = kwargs["operation"]
        assert operation.worker == "sdf_rebuild"
        unavailable_calls.append(operation.parameters["backend_id"])
        return WorkerResult(
            status="unavailable",
            failures=["the explicitly requested qualified backend is unavailable"],
            metadata={
                "sdf_backend_id": "openvdb",
                "sdf_status": "unavailable",
            },
        )

    monkeypatch.setattr("geometry_repair.orchestrator._plan_repair", unavailable_plan)
    monkeypatch.setattr(
        "geometry_repair.orchestrator._run_worker_subprocess",
        unavailable_sdf_worker,
    )

    unavailable_result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "unavailable",
            profile="visual_only",
            mode="auto",
            enabled_workers=["openvdb_rebuild"],
        )
    )

    assert unavailable_calls == ["openvdb"]
    assert [attempt.status for attempt in unavailable_result.attempts] == ["unavailable"]
    assert unavailable_result.attempts[0].operation.worker == "sdf_rebuild"

    def sdf_plan(request, diagnosis, normalized_source, plan_path):
        assert request.enabled_workers == ["sdf_rebuild"]
        operations = [
            RepairOperation(
                operation_id=f"attempt-{index:02d}-sdf-rebuild",
                worker="openvdb_rebuild" if index == 0 else "sdf_rebuild",
                implementation="test-qualified-sdf-backend",
                parameters={"backend_id": "openvdb"},
                drift_band="identity",
                source_checkpoint=str(normalized_source),
            )
            for index in range(2)
        ]
        assert all(operation.worker == "sdf_rebuild" for operation in operations)
        plan = RepairPlan(
            source_sha256=diagnosis.source_sha256,
            profile=request.profile,
            operations=operations,
            budgets=request.budgets,
            deterministic_seed=request.deterministic_seed,
            rationale="Reject one backend result, then bind the accepted provenance.",
            plan_path=str(plan_path),
        )
        atomic_write_json(plan_path, plan)
        return plan

    completed_calls: list[str] = []

    def completed_sdf_worker(**kwargs) -> WorkerResult:
        output = Path(kwargs["output"])
        operation = kwargs["operation"]
        assert operation.worker == "sdf_rebuild"
        assert operation.parameters["backend_id"] == "openvdb"
        completed_calls.append(operation.operation_id)
        if len(completed_calls) == 1:
            trimesh.creation.icosphere(subdivisions=1, radius=2.0).export(output)
        else:
            shutil.copy2(kwargs["source"], output)
        return WorkerResult(
            status="completed",
            output_path=str(output),
            output_sha256=file_sha256(output),
            changed=len(completed_calls) == 1,
            metadata={
                "sdf_backend": dict(_BACKEND_IDENTITY),
                "backend_qualification_id": _QUALIFICATION_ID,
                "backend_success_is_acceptance": False,
            },
        )

    monkeypatch.setattr("geometry_repair.orchestrator._plan_repair", sdf_plan)
    monkeypatch.setattr(
        "geometry_repair.orchestrator._run_worker_subprocess",
        completed_sdf_worker,
    )

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="visual_only",
            mode="auto",
            enabled_workers=["openvdb_rebuild"],
        )
    )

    assert completed_calls == [
        "attempt-00-sdf-rebuild",
        "attempt-01-sdf-rebuild",
    ]
    assert [attempt.status for attempt in result.attempts] == ["rejected", "accepted"]
    attempt = result.attempts[1]
    assert attempt.status == "accepted"
    assert attempt.worker_report_sha256 == file_sha256(attempt.worker_report_path or "")

    ledger = json.loads(Path(result.attempts_path).read_text(encoding="utf-8"))
    assert [item["worker_report_sha256"] for item in ledger["attempts"]] == [
        item.worker_report_sha256 for item in result.attempts
    ]

    evidence = json.loads(
        Path(result.geometry_validation_evidence_path).read_text(encoding="utf-8")
    )
    operation_reports = [
        artifact
        for artifact in evidence["evidence_artifacts"]
        if artifact["kind"] == "operation_report"
    ]
    assert operation_reports == [
        {
            "kind": "operation_report",
            "path": str(Path(item.worker_report_path or "").resolve()),
            "sha256": item.worker_report_sha256,
        }
        for item in result.attempts
    ]

    certificate = json.loads(Path(result.certificate_path).read_text(encoding="utf-8"))
    assert certificate["accepted_backend_identity"] == _BACKEND_IDENTITY
    assert certificate["accepted_backend_qualification_id"] == _QUALIFICATION_ID

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["accepted_backend_identity"] == _BACKEND_IDENTITY
    assert manifest["accepted_backend_qualification_id"] == _QUALIFICATION_ID
    assert manifest["attempt_ledger_sha256"] == file_sha256(result.attempts_path)
    assert manifest["repair_certificate_sha256"] == file_sha256(result.certificate_path)
    assert manifest["geometry_validation_evidence_sha256"] == file_sha256(
        result.geometry_validation_evidence_path
    )


def test_worker_report_verification_rejects_tampering(tmp_path: Path) -> None:
    report_path = tmp_path / "operation.json"
    atomic_write_json(
        report_path,
        WorkerResult(status="failed", failures=["expected test failure"]),
    )
    attempt = AttemptRecord(
        attempt_id="attempt-00",
        operation=RepairOperation(
            operation_id="attempt-00-noop",
            worker="noop",
            implementation="test",
            drift_band="identity",
            source_checkpoint=str(tmp_path / "source.usd"),
        ),
        status="failed",
        worker_report_path=str(report_path),
        worker_report_sha256=file_sha256(report_path),
    )

    _, verified = _verified_worker_report(attempt)
    assert verified.failures == ["expected test failure"]

    report_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 does not match"):
        _verified_worker_report(attempt)


@pytest.mark.parametrize(
    ("failure_kind", "expected_detail"),
    [
        (
            "operation_report_digest",
            "attempt-00 worker report SHA-256 does not match its ledger",
        ),
        ("accepted_output_missing", "accepted attempt output differs from its worker report"),
        (
            "accepted_output_unreadable",
            "accepted attempt output cannot be read for SHA-256 verification",
        ),
    ],
)
def test_final_operation_evidence_failure_returns_rejected_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    expected_detail: str,
) -> None:
    source = tmp_path / "cube.stl"
    trimesh.creation.box().export(source)

    if failure_kind == "operation_report_digest":
        original_report_verifier = orchestrator_module._verified_operation_report_artifacts

        def tamper_operation_report(attempts: list[AttemptRecord]) -> list[dict[str, str]]:
            assert attempts and attempts[0].worker_report_path is not None
            Path(attempts[0].worker_report_path).write_text("{}\n", encoding="utf-8")
            return original_report_verifier(attempts)

        monkeypatch.setattr(
            orchestrator_module,
            "_verified_operation_report_artifacts",
            tamper_operation_report,
        )
    elif failure_kind == "accepted_output_missing":
        original_backend_verifier = orchestrator_module._accepted_backend_provenance

        def remove_accepted_output(
            accepted_attempt: AttemptRecord | None,
        ) -> tuple[dict[str, Any] | None, str | None]:
            assert accepted_attempt is not None and accepted_attempt.output_path is not None
            Path(accepted_attempt.output_path).unlink()
            return original_backend_verifier(accepted_attempt)

        monkeypatch.setattr(
            orchestrator_module,
            "_accepted_backend_provenance",
            remove_accepted_output,
        )
    else:
        original_report_verifier = orchestrator_module._verified_operation_report_artifacts
        original_file_sha256 = orchestrator_module.file_sha256

        def arm_output_digest_failure(
            attempts: list[AttemptRecord],
        ) -> list[dict[str, str]]:
            artifacts = original_report_verifier(attempts)
            accepted_attempt = next(attempt for attempt in attempts if attempt.status == "accepted")
            assert accepted_attempt.output_path is not None
            accepted_output = Path(accepted_attempt.output_path).resolve()

            def unreadable_output_digest(path: str | Path) -> str:
                if Path(path).resolve() == accepted_output:
                    raise OSError("forced unreadable accepted output")
                return original_file_sha256(path)

            monkeypatch.setattr(orchestrator_module, "file_sha256", unreadable_output_digest)
            return artifacts

        monkeypatch.setattr(
            orchestrator_module,
            "_verified_operation_report_artifacts",
            arm_output_digest_failure,
        )

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="visual_only",
            mode="auto",
        )
    )

    expected_error = f"final operation evidence verification failed: {expected_detail}"
    assert result.outcome == "rejected"
    assert result.error == expected_error

    certificate = json.loads(Path(result.certificate_path).read_text(encoding="utf-8"))
    assert certificate["outcome"] == "rejected"
    assert certificate["validation_results"]["operation_report_provenance"] == "fail"
    assert expected_error in certificate["blockers"]
    assert certificate["accepted_backend_identity"] is None
    assert certificate["accepted_backend_qualification_id"] is None

    evidence = json.loads(
        Path(result.geometry_validation_evidence_path).read_text(encoding="utf-8")
    )
    assert expected_error in evidence["failures"]

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["outcome"] == "rejected"
