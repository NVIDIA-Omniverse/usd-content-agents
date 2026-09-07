# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolation and evidence tests for the complete collision pipeline worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest

import geometry_repair.worker_runner as worker_runner
from geometry_repair.artifacts import atomic_write_json, file_sha256
from geometry_repair.models import CollisionReport, RepairBudgets, RepairOperation, RepairProfile
from geometry_repair.orchestrator import _run_collision_geometry_subprocess
from geometry_repair.process_limits import (
    ADDRESS_SPACE_LIMIT_MODE_HARD,
    ADDRESS_SPACE_LIMIT_MODE_SOFT,
    ADDRESS_SPACE_LIMIT_SCOPE_PROCESS_TREE,
    OVPHYSX_ADDRESS_SPACE_LIMIT_SCOPE,
    OVPHYSX_DAEMON_ADDRESS_SPACE_LIMIT_EXEMPTION,
)
from geometry_repair.workers.base import WorkerResult
from geometry_repair.workers.collision_geometry import (
    CollisionGeometryParameters,
    CollisionGeometryWorker,
)
from geometry_repair.workers.sdf_rebuild import SdfRebuildWorker


def _operation(
    source: Path,
    output: Path,
    *,
    report_path: Path | None = None,
    audit_path: Path | None = None,
    profile: RepairProfile = "rigid_pick_place",
    runtime_engine: Literal["skip", "fake", "ovphysx"] = "fake",
    max_memory_mb: int = 512,
) -> RepairOperation:
    return RepairOperation(
        operation_id="build-collision-geometry",
        worker=CollisionGeometryWorker.name,
        implementation="geometry_repair.workers.collision_geometry",
        parameters={
            "operation": "build_collision_geometry",
            "profile": profile,
            "budgets": RepairBudgets(max_memory_mb=max_memory_mb).model_dump(mode="json"),
            "protected_features": [],
            "deterministic_seed": 17,
            "coacd_enabled": False,
            "sdf_collision_rebuild_enabled": True,
            "timeout_s": 3.5,
            "runtime_engine": runtime_engine,
            "report_path": str(report_path or output.with_name("collision_report.json")),
            "source_collision_audit_path": str(
                audit_path or output.with_name("source_collision_audit.json")
            ),
        },
        drift_band="reconstructive",
        source_checkpoint=str(source),
        target_role="collision",
    )


def _write_collision_report(
    source: Path,
    output: Path,
    report_path: Path,
    *,
    status: str = "pass",
    audit_path: Path | None = None,
    runtime_engine: Literal["skip", "fake", "ovphysx"] = "fake",
) -> CollisionReport:
    source_sha256 = file_sha256(source)
    output.write_text("collision-layer\n", encoding="utf-8")
    report = CollisionReport(
        status=status,
        representation="static_triangle_mesh",
        source_render_path=str(source.resolve()),
        source_render_sha256=source_sha256,
        source_render_sha256_after=source_sha256,
        source_render_unchanged=True,
        collision_path=str(output.resolve()),
        collision_sha256=file_sha256(output),
        runtime_engine=runtime_engine,
        report_path=str(report_path.resolve()),
        source_collision_audit_path=str(audit_path.resolve()) if audit_path else None,
    )
    atomic_write_json(report_path, report)
    return report


def _address_space_policy_metadata(*, uses_ovphysx_daemon: bool) -> dict[str, str]:
    if uses_ovphysx_daemon:
        return {
            "address_space_limit_mode": ADDRESS_SPACE_LIMIT_MODE_SOFT,
            "address_space_limit_scope": OVPHYSX_ADDRESS_SPACE_LIMIT_SCOPE,
            "memory_limit_exemption": OVPHYSX_DAEMON_ADDRESS_SPACE_LIMIT_EXEMPTION,
        }
    return {
        "address_space_limit_mode": ADDRESS_SPACE_LIMIT_MODE_HARD,
        "address_space_limit_scope": ADDRESS_SPACE_LIMIT_SCOPE_PROCESS_TREE,
    }


def test_collision_worker_delegates_complete_typed_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"
    report_path = tmp_path / "collision_report.json"
    captured: dict[str, Any] = {}

    def _build(render: Path, target: Path, **kwargs: Any) -> CollisionReport:
        captured.update({"render": render, "target": target, **kwargs})
        audit_path = Path(kwargs["source_collision_audit_path"])
        audit_path.write_text("source-audit\n", encoding="utf-8")
        return _write_collision_report(
            render,
            target,
            kwargs["report_path"],
            audit_path=audit_path,
        )

    monkeypatch.setattr("geometry_repair.collision.build_collision_geometry", _build)

    result = CollisionGeometryWorker().execute(
        source=source,
        output=output,
        operation=_operation(source, output),
    )

    assert result.status == "completed"
    assert result.changed is True
    assert result.operations == ["build_collision_geometry"]
    assert result.output_path == str(output.resolve())
    assert result.metadata["collision_report_path"] == str(report_path.resolve())
    assert result.metadata["collision_report_sha256"] == file_sha256(report_path)
    audit_path = tmp_path / "source_collision_audit.json"
    assert result.metadata["source_collision_audit_path"] == str(audit_path.resolve())
    assert result.metadata["source_collision_audit_sha256"] == file_sha256(audit_path)
    assert captured["render"] == source.resolve()
    assert captured["target"] == output.resolve()
    assert captured["profile"] == "rigid_pick_place"
    assert captured["budgets"].max_memory_mb == 512
    assert captured["deterministic_seed"] == 17
    assert captured["coacd_enabled"] is False
    assert captured["sdf_collision_rebuild_enabled"] is True
    assert captured["timeout_s"] == 3.5
    assert captured["runtime_engine"] == "fake"


def test_collision_worker_normalizes_legacy_sdf_toggle(tmp_path: Path) -> None:
    source = tmp_path / "render.usd"
    output = tmp_path / "collision.usda"
    parameters = dict(_operation(source, output).parameters)
    enabled = parameters.pop("sdf_collision_rebuild_enabled")
    parameters["openvdb_collision_rebuild_enabled"] = enabled

    normalized = CollisionGeometryParameters.model_validate(parameters)

    assert normalized.sdf_collision_rebuild_enabled is True
    assert "openvdb_collision_rebuild_enabled" not in normalized.model_dump(mode="json")


def test_collision_worker_rejects_conflicting_sdf_toggle_aliases(tmp_path: Path) -> None:
    source = tmp_path / "render.usd"
    output = tmp_path / "collision.usda"
    parameters = dict(_operation(source, output).parameters)
    parameters["openvdb_collision_rebuild_enabled"] = False

    with pytest.raises(ValueError, match="conflicting canonical and legacy SDF"):
        CollisionGeometryParameters.model_validate(parameters)


def test_collision_worker_rejects_evidence_path_escape(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    source = work_dir / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = work_dir / "collision.usda"

    with pytest.raises(ValueError, match="evidence paths must remain beside"):
        CollisionGeometryWorker().execute(
            source=source,
            output=output,
            operation=_operation(
                source,
                output,
                report_path=tmp_path / "collision_report.json",
            ),
        )


def test_collision_worker_treats_valid_failed_report_as_completed_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"

    def _build(render: Path, _target: Path, **kwargs: Any) -> CollisionReport:
        source_sha256 = file_sha256(render)
        report = CollisionReport(
            status="fail",
            representation="none",
            source_render_path=str(render.resolve()),
            source_render_sha256=source_sha256,
            source_render_sha256_after=source_sha256,
            source_render_unchanged=True,
            failures=["no acceptable collision candidate"],
            runtime_engine="fake",
            report_path=str(Path(kwargs["report_path"]).resolve()),
        )
        atomic_write_json(kwargs["report_path"], report)
        return report

    monkeypatch.setattr("geometry_repair.collision.build_collision_geometry", _build)

    result = CollisionGeometryWorker().execute(
        source=source,
        output=output,
        operation=_operation(source, output),
    )

    assert result.status == "completed"
    assert result.changed is False
    assert result.output_path is None
    assert result.metadata["collision_status"] == "fail"
    assert not output.exists()


def test_collision_worker_rejects_unreported_source_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"
    audit_path = tmp_path / "source_collision_audit.json"

    def _build(render: Path, target: Path, **kwargs: Any) -> CollisionReport:
        Path(kwargs["source_collision_audit_path"]).write_text(
            "unreported-audit\n",
            encoding="utf-8",
        )
        return _write_collision_report(render, target, kwargs["report_path"])

    monkeypatch.setattr("geometry_repair.collision.build_collision_geometry", _build)

    with pytest.raises(RuntimeError, match="unreported source-audit"):
        CollisionGeometryWorker().execute(
            source=source,
            output=output,
            operation=_operation(source, output),
        )

    assert not audit_path.exists()


def test_collision_orchestrator_forwards_limits_and_validates_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"
    report_path = tmp_path / "collision_report.json"
    audit_path = tmp_path / "source_collision_audit.json"
    attempt_dir = tmp_path / "collision_worker"
    budgets = RepairBudgets(max_memory_mb=768)
    captured: dict[str, Any] = {}

    def _run_worker(**kwargs: Any) -> WorkerResult:
        captured.update(kwargs)
        report = _write_collision_report(source, output, report_path)
        return WorkerResult(
            status="completed",
            output_path=str(output.resolve()),
            output_sha256=file_sha256(output),
            changed=True,
            operations=["build_collision_geometry"],
            metadata={
                "execution_scope": "isolated_subprocess",
                "memory_budget_mb": budgets.max_memory_mb,
                "memory_limit_mb": budgets.max_memory_mb,
                **_address_space_policy_metadata(uses_ovphysx_daemon=False),
                "collision_report_path": str(report_path.resolve()),
                "collision_report_sha256": file_sha256(report_path),
                "collision_status": report.status,
            },
        )

    monkeypatch.setattr("geometry_repair.orchestrator._run_worker_subprocess", _run_worker)

    report = _run_collision_geometry_subprocess(
        render_path=source,
        output_path=output,
        profile="rigid_pick_place",
        budgets=budgets,
        protected_features=[],
        deterministic_seed=23,
        coacd_enabled=True,
        sdf_collision_rebuild_enabled=False,
        timeout_s=0.25,
        runtime_engine="fake",
        report_path=report_path,
        source_collision_audit_path=audit_path,
        attempt_dir=attempt_dir,
    )

    assert report.status == "pass"
    assert report.collision_sha256 == file_sha256(output)
    assert captured["memory_mb"] == 768
    assert captured["timeout_s"] == 0.25
    assert captured["attempt_dir"] == attempt_dir.resolve()
    operation = captured["operation"]
    assert operation.worker == CollisionGeometryWorker.name
    assert operation.parameters["deterministic_seed"] == 23
    assert operation.parameters["coacd_enabled"] is True
    assert operation.parameters["sdf_collision_rebuild_enabled"] is False
    assert operation.parameters["timeout_s"] == pytest.approx(0.125)


def test_collision_orchestrator_accepts_ovphysx_with_parent_memory_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"
    report_path = tmp_path / "collision_report.json"
    audit_path = tmp_path / "source_collision_audit.json"

    def _run_worker(**_kwargs: Any) -> WorkerResult:
        report = _write_collision_report(
            source,
            output,
            report_path,
            runtime_engine="ovphysx",
        )
        return WorkerResult(
            status="completed",
            output_path=str(output.resolve()),
            output_sha256=file_sha256(output),
            changed=True,
            operations=["build_collision_geometry"],
            metadata={
                "execution_scope": "isolated_subprocess",
                "memory_budget_mb": 768,
                "memory_limit_mb": 768,
                **_address_space_policy_metadata(uses_ovphysx_daemon=True),
                "collision_report_path": str(report_path.resolve()),
                "collision_report_sha256": file_sha256(report_path),
                "collision_status": report.status,
            },
        )

    monkeypatch.setattr("geometry_repair.orchestrator._run_worker_subprocess", _run_worker)

    report = _run_collision_geometry_subprocess(
        render_path=source,
        output_path=output,
        profile="rigid_pick_place",
        budgets=RepairBudgets(max_memory_mb=768),
        protected_features=[],
        deterministic_seed=0,
        coacd_enabled=False,
        sdf_collision_rebuild_enabled=False,
        timeout_s=1.0,
        runtime_engine="ovphysx",
        report_path=report_path,
        source_collision_audit_path=audit_path,
        attempt_dir=tmp_path / "collision_worker",
    )

    assert report.status == "pass"
    assert report.runtime_engine == "ovphysx"


@pytest.mark.parametrize(
    ("runtime_engine", "tampered_field", "tampered_value", "expected_failure"),
    [
        ("ovphysx", "address_space_limit_mode", ADDRESS_SPACE_LIMIT_MODE_HARD, "mode"),
        (
            "ovphysx",
            "address_space_limit_scope",
            ADDRESS_SPACE_LIMIT_SCOPE_PROCESS_TREE,
            "scope",
        ),
        ("ovphysx", "memory_limit_exemption", None, "exemption"),
        ("fake", "address_space_limit_mode", ADDRESS_SPACE_LIMIT_MODE_SOFT, "mode"),
        (
            "fake",
            "address_space_limit_scope",
            OVPHYSX_ADDRESS_SPACE_LIMIT_SCOPE,
            "scope",
        ),
        (
            "fake",
            "memory_limit_exemption",
            OVPHYSX_DAEMON_ADDRESS_SPACE_LIMIT_EXEMPTION,
            "exemption",
        ),
    ],
)
def test_collision_orchestrator_rejects_address_space_policy_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_engine: Literal["fake", "ovphysx"],
    tampered_field: str,
    tampered_value: str | None,
    expected_failure: str,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"
    report_path = tmp_path / "collision_report.json"
    audit_path = tmp_path / "source_collision_audit.json"

    def _run_worker(**_kwargs: Any) -> WorkerResult:
        report = _write_collision_report(
            source,
            output,
            report_path,
            runtime_engine=runtime_engine,
        )
        metadata: dict[str, Any] = {
            "execution_scope": "isolated_subprocess",
            "memory_budget_mb": 768,
            "memory_limit_mb": 768,
            **_address_space_policy_metadata(uses_ovphysx_daemon=runtime_engine == "ovphysx"),
            "collision_report_path": str(report_path.resolve()),
            "collision_report_sha256": file_sha256(report_path),
            "collision_status": report.status,
        }
        metadata[tampered_field] = tampered_value
        return WorkerResult(
            status="completed",
            output_path=str(output.resolve()),
            output_sha256=file_sha256(output),
            changed=True,
            operations=["build_collision_geometry"],
            metadata=metadata,
        )

    monkeypatch.setattr("geometry_repair.orchestrator._run_worker_subprocess", _run_worker)

    report = _run_collision_geometry_subprocess(
        render_path=source,
        output_path=output,
        profile="rigid_pick_place",
        budgets=RepairBudgets(max_memory_mb=768),
        protected_features=[],
        deterministic_seed=0,
        coacd_enabled=False,
        sdf_collision_rebuild_enabled=False,
        timeout_s=1.0,
        runtime_engine=runtime_engine,
        report_path=report_path,
        source_collision_audit_path=audit_path,
        attempt_dir=tmp_path / "collision_worker",
    )

    assert report.status == "fail"
    assert expected_failure in report.failures[0]
    assert not output.exists()


def test_collision_orchestrator_rejects_runtime_engine_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"
    report_path = tmp_path / "collision_report.json"
    audit_path = tmp_path / "source_collision_audit.json"

    def _run_worker(**_kwargs: Any) -> WorkerResult:
        report = _write_collision_report(source, output, report_path, runtime_engine="fake")
        return WorkerResult(
            status="completed",
            output_path=str(output.resolve()),
            output_sha256=file_sha256(output),
            changed=True,
            operations=["build_collision_geometry"],
            metadata={
                "execution_scope": "isolated_subprocess",
                "memory_budget_mb": 768,
                "memory_limit_mb": 768,
                **_address_space_policy_metadata(uses_ovphysx_daemon=True),
                "collision_report_path": str(report_path.resolve()),
                "collision_report_sha256": file_sha256(report_path),
                "collision_status": report.status,
            },
        )

    monkeypatch.setattr("geometry_repair.orchestrator._run_worker_subprocess", _run_worker)

    report = _run_collision_geometry_subprocess(
        render_path=source,
        output_path=output,
        profile="rigid_pick_place",
        budgets=RepairBudgets(max_memory_mb=768),
        protected_features=[],
        deterministic_seed=0,
        coacd_enabled=False,
        sdf_collision_rebuild_enabled=False,
        timeout_s=1.0,
        runtime_engine="ovphysx",
        report_path=report_path,
        source_collision_audit_path=audit_path,
        attempt_dir=tmp_path / "collision_worker",
    )

    assert report.status == "fail"
    assert "runtime engine does not match" in report.failures[0]
    assert not output.exists()


def test_collision_orchestrator_accepts_hashed_source_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"
    report_path = tmp_path / "collision_report.json"
    audit_path = tmp_path / "source_collision_audit.json"

    def _run_worker(**_kwargs: Any) -> WorkerResult:
        audit_path.write_text("source-audit\n", encoding="utf-8")
        report = _write_collision_report(
            source,
            output,
            report_path,
            audit_path=audit_path,
        )
        return WorkerResult(
            status="completed",
            output_path=str(output.resolve()),
            output_sha256=file_sha256(output),
            changed=True,
            operations=["build_collision_geometry"],
            metadata={
                "execution_scope": "isolated_subprocess",
                "memory_budget_mb": 512,
                "memory_limit_mb": 512,
                **_address_space_policy_metadata(uses_ovphysx_daemon=False),
                "collision_report_path": str(report_path.resolve()),
                "collision_report_sha256": file_sha256(report_path),
                "collision_status": report.status,
                "source_collision_audit_path": str(audit_path.resolve()),
                "source_collision_audit_sha256": file_sha256(audit_path),
            },
        )

    monkeypatch.setattr("geometry_repair.orchestrator._run_worker_subprocess", _run_worker)

    report = _run_collision_geometry_subprocess(
        render_path=source,
        output_path=output,
        profile="rigid_pick_place",
        budgets=RepairBudgets(max_memory_mb=512),
        protected_features=[],
        deterministic_seed=0,
        coacd_enabled=True,
        sdf_collision_rebuild_enabled=True,
        timeout_s=1.0,
        runtime_engine="fake",
        report_path=report_path,
        source_collision_audit_path=audit_path,
        attempt_dir=tmp_path / "collision_worker",
    )

    assert report.status == "pass"
    assert report.source_collision_audit_path == str(audit_path.resolve())
    assert audit_path.is_file()


def test_collision_orchestrator_rejects_tampered_source_audit_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"
    report_path = tmp_path / "collision_report.json"
    audit_path = tmp_path / "source_collision_audit.json"

    def _run_worker(**_kwargs: Any) -> WorkerResult:
        audit_path.write_text("source-audit\n", encoding="utf-8")
        report = _write_collision_report(
            source,
            output,
            report_path,
            audit_path=audit_path,
        )
        return WorkerResult(
            status="completed",
            output_path=str(output.resolve()),
            output_sha256=file_sha256(output),
            changed=True,
            operations=["build_collision_geometry"],
            metadata={
                "execution_scope": "isolated_subprocess",
                "memory_budget_mb": 512,
                "memory_limit_mb": 512,
                **_address_space_policy_metadata(uses_ovphysx_daemon=False),
                "collision_report_path": str(report_path.resolve()),
                "collision_report_sha256": file_sha256(report_path),
                "collision_status": report.status,
                "source_collision_audit_path": str(audit_path.resolve()),
                "source_collision_audit_sha256": "0" * 64,
            },
        )

    monkeypatch.setattr("geometry_repair.orchestrator._run_worker_subprocess", _run_worker)

    report = _run_collision_geometry_subprocess(
        render_path=source,
        output_path=output,
        profile="rigid_pick_place",
        budgets=RepairBudgets(max_memory_mb=512),
        protected_features=[],
        deterministic_seed=0,
        coacd_enabled=True,
        sdf_collision_rebuild_enabled=True,
        timeout_s=1.0,
        runtime_engine="fake",
        report_path=report_path,
        source_collision_audit_path=audit_path,
        attempt_dir=tmp_path / "collision_worker",
    )

    assert report.status == "fail"
    assert "source-audit digest does not match" in report.failures[0]
    assert not output.exists()
    assert not audit_path.exists()


@pytest.mark.parametrize("tamper_kind", ["artifact", "metadata"])
def test_collision_orchestrator_rejects_unclaimed_source_audit_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"
    report_path = tmp_path / "collision_report.json"
    audit_path = tmp_path / "source_collision_audit.json"

    def _run_worker(**_kwargs: Any) -> WorkerResult:
        report = _write_collision_report(source, output, report_path)
        metadata = {
            "execution_scope": "isolated_subprocess",
            "memory_budget_mb": 512,
            "memory_limit_mb": 512,
            **_address_space_policy_metadata(uses_ovphysx_daemon=False),
            "collision_report_path": str(report_path.resolve()),
            "collision_report_sha256": file_sha256(report_path),
            "collision_status": report.status,
        }
        if tamper_kind == "artifact":
            audit_path.write_text("unreported-audit\n", encoding="utf-8")
        else:
            metadata.update(
                {
                    "source_collision_audit_path": str(audit_path.resolve()),
                    "source_collision_audit_sha256": "0" * 64,
                }
            )
        return WorkerResult(
            status="completed",
            output_path=str(output.resolve()),
            output_sha256=file_sha256(output),
            changed=True,
            operations=["build_collision_geometry"],
            metadata=metadata,
        )

    monkeypatch.setattr("geometry_repair.orchestrator._run_worker_subprocess", _run_worker)

    report = _run_collision_geometry_subprocess(
        render_path=source,
        output_path=output,
        profile="rigid_pick_place",
        budgets=RepairBudgets(max_memory_mb=512),
        protected_features=[],
        deterministic_seed=0,
        coacd_enabled=True,
        sdf_collision_rebuild_enabled=True,
        timeout_s=1.0,
        runtime_engine="fake",
        report_path=report_path,
        source_collision_audit_path=audit_path,
        attempt_dir=tmp_path / "collision_worker",
    )

    assert report.status == "fail"
    assert "source-audit" in report.failures[0]
    assert not output.exists()
    assert not audit_path.exists()


@pytest.mark.parametrize("mutate_source", [False, True], ids=["false-flag", "changed-bytes"])
def test_collision_orchestrator_requires_affirmative_source_immutability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate_source: bool,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    source_sha256_before = file_sha256(source)
    output = tmp_path / "collision.usda"
    report_path = tmp_path / "collision_report.json"
    audit_path = tmp_path / "source_collision_audit.json"

    def _run_worker(**_kwargs: Any) -> WorkerResult:
        if mutate_source:
            source.write_text("mutated-render-layer\n", encoding="utf-8")
        source_sha256_after = file_sha256(source)
        output.write_text("collision-layer\n", encoding="utf-8")
        report = CollisionReport(
            status="pass",
            representation="static_triangle_mesh",
            source_render_path=str(source.resolve()),
            source_render_sha256=source_sha256_before,
            source_render_sha256_after=source_sha256_after,
            source_render_unchanged=False,
            collision_path=str(output.resolve()),
            collision_sha256=file_sha256(output),
            runtime_engine="fake",
            report_path=str(report_path.resolve()),
        )
        atomic_write_json(report_path, report)
        return WorkerResult(
            status="completed",
            output_path=str(output.resolve()),
            output_sha256=file_sha256(output),
            changed=True,
            operations=["build_collision_geometry"],
            metadata={
                "execution_scope": "isolated_subprocess",
                "memory_budget_mb": 512,
                "memory_limit_mb": 512,
                **_address_space_policy_metadata(uses_ovphysx_daemon=False),
                "collision_report_path": str(report_path.resolve()),
                "collision_report_sha256": file_sha256(report_path),
                "collision_status": report.status,
            },
        )

    monkeypatch.setattr("geometry_repair.orchestrator._run_worker_subprocess", _run_worker)

    report = _run_collision_geometry_subprocess(
        render_path=source,
        output_path=output,
        profile="rigid_pick_place",
        budgets=RepairBudgets(max_memory_mb=512),
        protected_features=[],
        deterministic_seed=0,
        coacd_enabled=True,
        sdf_collision_rebuild_enabled=True,
        timeout_s=1.0,
        runtime_engine="fake",
        report_path=report_path,
        source_collision_audit_path=audit_path,
        attempt_dir=tmp_path / "collision_worker",
    )

    assert report.status == "fail"
    assert report.source_render_sha256 == source_sha256_before
    assert report.source_render_unchanged is (not mutate_source)
    assert "did not preserve the immutable render source" in report.failures[0]
    assert not output.exists()


def test_collision_orchestrator_converts_timeout_to_failed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    source_sha256 = file_sha256(source)
    output = tmp_path / "collision.usda"
    report_path = tmp_path / "collision_report.json"
    audit_path = tmp_path / "source_collision_audit.json"
    for path in (output, report_path, audit_path):
        path.write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr(
        "geometry_repair.orchestrator._run_worker_subprocess",
        lambda **_kwargs: WorkerResult(
            status="failed",
            failures=["worker exceeded remaining wall-clock budget of 0.010s"],
            metadata={
                "execution_scope": "isolated_subprocess",
                "memory_limit_mb": 512,
            },
        ),
    )

    report = _run_collision_geometry_subprocess(
        render_path=source,
        output_path=output,
        profile="rigid_pick_place",
        budgets=RepairBudgets(max_memory_mb=512),
        protected_features=[],
        deterministic_seed=0,
        coacd_enabled=True,
        sdf_collision_rebuild_enabled=True,
        timeout_s=0.01,
        runtime_engine="skip",
        report_path=report_path,
        source_collision_audit_path=audit_path,
        attempt_dir=tmp_path / "collision_worker",
    )

    assert report.status == "fail"
    assert report.representation == "none"
    assert report.collision_path is None
    assert report.source_render_sha256 == source_sha256
    assert report.source_render_unchanged is True
    assert "exceeded remaining wall-clock budget" in report.failures[0]
    assert not output.exists()
    assert not audit_path.exists()
    assert CollisionReport.model_validate_json(report_path.read_text(encoding="utf-8")) == report


def test_collision_orchestrator_runs_real_worker_in_fresh_attempt_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"
    report_path = tmp_path / "collision_report.json"
    attempt_dir = tmp_path / "collision_worker"

    report = _run_collision_geometry_subprocess(
        render_path=source,
        output_path=output,
        profile="visual_only",
        budgets=RepairBudgets(max_memory_mb=2048),
        protected_features=[],
        deterministic_seed=0,
        coacd_enabled=False,
        sdf_collision_rebuild_enabled=False,
        # Harness bound only: this exercises the real isolated entry point,
        # including its import/startup cost under coverage and xdist.  It is
        # not a worker-latency assertion, so leave enough headroom for a
        # loaded shared runner while still catching a genuine hang.
        timeout_s=60.0,
        runtime_engine="skip",
        report_path=report_path,
        source_collision_audit_path=tmp_path / "source_collision_audit.json",
        attempt_dir=attempt_dir,
    )

    assert report.status == "not_required"
    assert report.collision_path is None
    assert attempt_dir.is_dir()
    assert (attempt_dir / "operation_request.json").is_file()
    worker_result = WorkerResult.model_validate_json(
        (attempt_dir / "worker_result.json").read_text(encoding="utf-8")
    )
    assert worker_result.status == "completed"
    assert worker_result.metadata["execution_scope"] == "isolated_subprocess"
    assert worker_result.metadata["memory_budget_mb"] == 2048
    assert worker_result.metadata["memory_limit_mb"] == 2048
    assert (
        worker_result.metadata["address_space_limit_mode"],
        worker_result.metadata["address_space_limit_scope"],
        worker_result.metadata.get("memory_limit_exemption"),
    ) == (
        ADDRESS_SPACE_LIMIT_MODE_HARD,
        ADDRESS_SPACE_LIMIT_SCOPE_PROCESS_TREE,
        None,
    )
    assert "memory_limit_exemption" not in worker_result.metadata
    assert not output.exists()


def test_collision_orchestrator_rejects_symlink_attempt_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    real_attempt_dir = tmp_path / "real-worker-dir"
    real_attempt_dir.mkdir()
    attempt_dir = tmp_path / "collision_worker"
    attempt_dir.symlink_to(real_attempt_dir, target_is_directory=True)
    monkeypatch.setattr(
        "geometry_repair.orchestrator._run_worker_subprocess",
        lambda **_kwargs: pytest.fail("unsafe attempt directory reached the worker launcher"),
    )

    report = _run_collision_geometry_subprocess(
        render_path=source,
        output_path=tmp_path / "collision.usda",
        profile="visual_only",
        budgets=RepairBudgets(max_memory_mb=512),
        protected_features=[],
        deterministic_seed=0,
        coacd_enabled=False,
        sdf_collision_rebuild_enabled=False,
        timeout_s=1.0,
        runtime_engine="skip",
        report_path=tmp_path / "collision_report.json",
        source_collision_audit_path=tmp_path / "source_collision_audit.json",
        attempt_dir=attempt_dir,
    )

    assert report.status == "fail"
    assert "must not be a symbolic link" in report.failures[0]
    assert list(real_attempt_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("runtime_engine", "profile"),
    [
        ("skip", "rigid_pick_place"),
        ("fake", "rigid_pick_place"),
        ("ovphysx", "visual_only"),
        ("ovphysx", "articulated_rigid"),
    ],
)
def test_collision_runner_keeps_address_space_limit_without_ovphysx_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_engine: Literal["skip", "fake", "ovphysx"],
    profile: RepairProfile,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"
    operation_path = tmp_path / "operation.json"
    result_path = tmp_path / "worker_result.json"
    atomic_write_json(
        operation_path,
        _operation(
            source,
            output,
            profile=profile,
            runtime_engine=runtime_engine,
            max_memory_mb=640,
        ),
    )
    limits: list[tuple[str, int]] = []

    class _StubCollisionWorker:
        name = CollisionGeometryWorker.name
        operations = CollisionGeometryWorker.operations

        def available(self) -> tuple[bool, None]:
            return True, None

        def execute(self, **_kwargs: Any) -> WorkerResult:
            return WorkerResult(
                status="completed",
                operations=["build_collision_geometry"],
            )

    monkeypatch.setitem(
        worker_runner._WORKERS,
        CollisionGeometryWorker.name,
        _StubCollisionWorker(),
    )
    monkeypatch.setattr(
        "geometry_repair.worker_runner.limit_cpu_affinity",
        lambda count: limits.append(("cpu", count)),
    )
    monkeypatch.setattr(
        "geometry_repair.worker_runner.limit_address_space",
        lambda memory_mb: limits.append(("hard", memory_mb)),
    )
    monkeypatch.setattr(
        "geometry_repair.worker_runner.limit_address_space_soft",
        lambda memory_mb: limits.append(("soft", memory_mb)),
    )

    exit_code = worker_runner.main(
        [
            "--worker",
            CollisionGeometryWorker.name,
            "--source",
            str(source),
            "--output",
            str(output),
            "--operation",
            str(operation_path),
            "--result",
            str(result_path),
            "--memory-mb",
            "640",
        ]
    )

    assert exit_code == 0
    assert limits == [("hard", 640)]
    result = WorkerResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    assert result.metadata["execution_scope"] == "isolated_subprocess"
    assert result.metadata["memory_budget_mb"] == 640
    assert result.metadata["memory_limit_mb"] == 640
    assert (
        result.metadata["address_space_limit_mode"],
        result.metadata["address_space_limit_scope"],
        result.metadata.get("memory_limit_exemption"),
    ) == (
        ADDRESS_SPACE_LIMIT_MODE_HARD,
        ADDRESS_SPACE_LIMIT_SCOPE_PROCESS_TREE,
        None,
    )
    assert "memory_limit_exemption" not in result.metadata


@pytest.mark.parametrize("profile", ["rigid_pick_place", "static_environment"])
def test_collision_runner_keeps_soft_parent_limit_for_ovphysx_preprocessing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: RepairProfile,
) -> None:
    source = tmp_path / "render.usd"
    source.write_text("render-layer\n", encoding="utf-8")
    output = tmp_path / "collision.usda"
    operation_path = tmp_path / "operation.json"
    result_path = tmp_path / "worker_result.json"
    atomic_write_json(
        operation_path,
        _operation(
            source,
            output,
            profile=profile,
            runtime_engine="ovphysx",
            max_memory_mb=640,
        ),
    )
    limits: list[tuple[str, int]] = []

    class _StubCollisionWorker:
        name = CollisionGeometryWorker.name
        operations = CollisionGeometryWorker.operations

        def available(self) -> tuple[bool, None]:
            return True, None

        def execute(self, **_kwargs: Any) -> WorkerResult:
            assert limits == [("soft", 640)]
            return WorkerResult(
                status="completed",
                operations=["build_collision_geometry"],
            )

    monkeypatch.setitem(
        worker_runner._WORKERS,
        CollisionGeometryWorker.name,
        _StubCollisionWorker(),
    )
    monkeypatch.setattr(
        "geometry_repair.worker_runner.limit_address_space",
        lambda memory_mb: limits.append(("hard", memory_mb)),
    )
    monkeypatch.setattr(
        "geometry_repair.worker_runner.limit_address_space_soft",
        lambda memory_mb: limits.append(("soft", memory_mb)),
    )

    exit_code = worker_runner.main(
        [
            "--worker",
            CollisionGeometryWorker.name,
            "--source",
            str(source),
            "--output",
            str(output),
            "--operation",
            str(operation_path),
            "--result",
            str(result_path),
            "--memory-mb",
            "640",
        ]
    )

    assert exit_code == 0
    assert limits == [("soft", 640)]
    result = WorkerResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    assert result.metadata["execution_scope"] == "isolated_subprocess"
    assert result.metadata["memory_budget_mb"] == 640
    assert result.metadata["memory_limit_mb"] == 640
    assert (
        result.metadata["address_space_limit_mode"],
        result.metadata["address_space_limit_scope"],
        result.metadata["memory_limit_exemption"],
    ) == (
        ADDRESS_SPACE_LIMIT_MODE_SOFT,
        OVPHYSX_ADDRESS_SPACE_LIMIT_SCOPE,
        OVPHYSX_DAEMON_ADDRESS_SPACE_LIMIT_EXEMPTION,
    )


def test_sdf_runner_applies_single_cpu_affinity_before_memory_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "output.usda"
    operation_path = tmp_path / "operation.json"
    result_path = tmp_path / "worker_result.json"
    operation = RepairOperation(
        operation_id="sdf-affinity",
        worker=SdfRebuildWorker.name,
        implementation="test",
        parameters={},
        drift_band="reconstructive",
        source_checkpoint=str(source),
    )
    legacy_operation = operation.model_dump(mode="json")
    legacy_operation["worker"] = "openvdb_rebuild"
    atomic_write_json(operation_path, legacy_operation)
    calls: list[tuple[str, int]] = []

    class _StubSdfWorker:
        name = SdfRebuildWorker.name
        operations = SdfRebuildWorker.operations

        def available(self) -> tuple[bool, None]:
            return True, None

        def execute(self, **_kwargs: Any) -> WorkerResult:
            return WorkerResult(status="completed")

    monkeypatch.setitem(
        worker_runner._WORKERS,
        SdfRebuildWorker.name,
        _StubSdfWorker(),
    )
    monkeypatch.setattr(
        "geometry_repair.worker_runner.limit_cpu_affinity",
        lambda count: calls.append(("cpu", count)),
    )
    monkeypatch.setattr(
        "geometry_repair.worker_runner.limit_address_space",
        lambda memory_mb: calls.append(("memory", memory_mb)),
    )

    exit_code = worker_runner.main(
        [
            "--worker",
            "openvdb_rebuild",
            "--source",
            str(source),
            "--output",
            str(output),
            "--operation",
            str(operation_path),
            "--result",
            str(result_path),
            "--memory-mb",
            "8192",
        ]
    )

    assert exit_code == 0
    assert calls == [("cpu", 1), ("memory", 8192)]
    result = WorkerResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    assert (
        result.metadata["address_space_limit_mode"],
        result.metadata["address_space_limit_scope"],
        result.metadata.get("memory_limit_exemption"),
    ) == (
        ADDRESS_SPACE_LIMIT_MODE_HARD,
        ADDRESS_SPACE_LIMIT_SCOPE_PROCESS_TREE,
        None,
    )
    assert "memory_limit_exemption" not in result.metadata


def test_sdf_runner_checks_the_backend_requested_by_the_operation(monkeypatch) -> None:
    operation = RepairOperation(
        operation_id="sdf-backend-availability",
        worker=SdfRebuildWorker.name,
        implementation="test",
        parameters={"backend_id": "qualified-alternate"},
        drift_band="reconstructive",
        source_checkpoint="source.usda",
    )
    checked: list[str] = []

    def available_for_backend(_self, backend_id: str) -> tuple[bool, None]:
        checked.append(backend_id)
        return True, None

    monkeypatch.setattr(SdfRebuildWorker, "available_for_backend", available_for_backend)

    assert worker_runner._worker_availability(SdfRebuildWorker(), operation) == (True, None)
    assert checked == ["qualified-alternate"]
