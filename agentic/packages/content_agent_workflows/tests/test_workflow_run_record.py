# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from content_agent_workflows.common import artifacts as artifact_helpers
from content_agent_workflows.common.run_record import WorkflowRunRecorder


def test_run_recorder_writes_request_artifacts_checkpoints_and_success(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    recorder = WorkflowRunRecorder.create(
        run_dir,
        workflow="materials.assign",
        request={"schema_version": "request.v1"},
        source_path=source,
        backend={"scene_tool": "usd-cli"},
        policy={"clear_materials": True},
        required_artifacts=["output", "validation"],
    )
    output = run_dir / "output" / "materialized.usda"
    output.parent.mkdir()
    output.write_text("#usda 1.0\n", encoding="utf-8")
    validation = run_dir / "validation_evidence.json"
    validation.write_text('{"status":"pass"}\n', encoding="utf-8")

    recorder.record_artifact("output", output, kind="usd")
    recorder.record_artifact("validation", validation, kind="validation")
    recorder.checkpoint("validated", ["output", "validation"])
    manifest = recorder.finalize("pass")

    assert manifest.status == "pass"
    assert manifest.source_sha256
    assert manifest.checkpoints[0].sealed
    assert set(manifest.checkpoints[0].artifact_sha256) == {"output", "validation"}
    assert all(
        ".workflow_checkpoints" in Path(path).parts
        for path in manifest.checkpoints[0].artifact_paths.values()
    )
    resumed = WorkflowRunRecorder.resume(run_dir)
    assert resumed.manifest.status == "pass"
    assert {item.logical_name for item in resumed.manifest.artifacts} == {
        "output",
        "validation",
    }


def test_run_recorder_converts_incomplete_success_to_failure(tmp_path: Path) -> None:
    recorder = WorkflowRunRecorder.create(
        tmp_path / "run",
        workflow="simready",
        request={},
        required_artifacts=["report"],
    )

    manifest = recorder.finalize("pass")

    assert manifest.status == "fail"
    assert manifest.failure == {
        "code": "missing_required_artifacts",
        "missing": ["report"],
    }


def test_run_recorder_can_use_a_contained_namespace_without_overwriting_outer_files(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outer_request = run_dir / "request.json"
    outer_manifest = run_dir / "workflow_run_manifest.json"
    outer_request.write_text('{"owner":"outer"}\n', encoding="utf-8")
    outer_manifest.write_text('{"owner":"outer"}\n', encoding="utf-8")

    recorder = WorkflowRunRecorder.create(
        run_dir,
        workflow="physics_authoring",
        request={"owner": "nested"},
        record_subdir=Path("raw") / "physics_workflow" / "iteration-0001",
    )
    manifest = recorder.finalize("pass")

    assert manifest.status == "pass"
    assert outer_request.read_text(encoding="utf-8") == '{"owner":"outer"}\n'
    assert outer_manifest.read_text(encoding="utf-8") == '{"owner":"outer"}\n'
    assert recorder.manifest.request_path == (
        "raw/physics_workflow/iteration-0001/request.json"
    )
    assert recorder.path == (
        run_dir
        / "raw"
        / "physics_workflow"
        / "iteration-0001"
        / "workflow_run_manifest.json"
    )
    resumed = WorkflowRunRecorder.resume(
        run_dir,
        record_subdir=Path("raw") / "physics_workflow" / "iteration-0001",
    )
    assert resumed.manifest.status == "pass"


@pytest.mark.parametrize(
    "record_subdir",
    [Path("/outside"), Path("../outside"), Path("raw/../outside")],
)
def test_run_recorder_rejects_unsafe_record_namespace(
    tmp_path: Path,
    record_subdir: Path,
) -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        WorkflowRunRecorder.create(
            tmp_path / "run",
            workflow="physics_authoring",
            request={},
            record_subdir=record_subdir,
        )


def test_run_recorder_rejects_tampered_required_artifact_at_finalize(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    recorder = WorkflowRunRecorder.create(
        run_dir,
        workflow="physics",
        request={},
        required_artifacts=["validation"],
    )
    validation = run_dir / "validation.json"
    validation.write_text('{"status":"pass"}\n', encoding="utf-8")
    recorder.record_artifact("validation", validation, kind="validation")
    validation.write_text('{"status":"fail"}\n', encoding="utf-8")

    manifest = recorder.finalize("pass")

    assert manifest.status == "fail"
    assert manifest.failure == {
        "code": "invalid_required_artifacts",
        "invalid": ["validation"],
        "missing": [],
    }


def test_run_recorder_rejects_escaping_artifact_symlink(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    recorder = WorkflowRunRecorder.create(
        run_dir,
        workflow="conversion",
        request={},
    )
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    link = run_dir / "report.json"
    link.symlink_to(outside)

    with pytest.raises(ValueError, match="symlinks"):
        recorder.record_artifact("report", link, kind="report")


def test_run_recorder_records_digest_and_size_from_same_held_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    recorder = WorkflowRunRecorder.create(
        run_dir,
        workflow="conversion",
        request={},
        required_artifacts=["report"],
    )
    report = run_dir / "report.json"
    original = b'{"status":"pass"}\n'
    report.write_bytes(original)
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret":"must-not-be-read"}\n', encoding="utf-8")

    original_open = artifact_helpers.os.open
    swapped = False

    def open_and_swap(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        file_fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == report.name and dir_fd is not None and not swapped:
            swapped = True
            report.unlink()
            report.symlink_to(outside)
        return file_fd

    monkeypatch.setattr(artifact_helpers.os, "open", open_and_swap)

    record = recorder.record_artifact("report", report, kind="report")

    assert record.sha256 == hashlib.sha256(original).hexdigest()
    assert record.size_bytes == len(original)
    manifest = recorder.finalize("pass")
    assert manifest.status == "fail"
    assert manifest.failure == {
        "code": "invalid_required_artifacts",
        "missing": [],
        "invalid": ["report"],
    }


def test_run_recorder_resume_preserves_checkpoints_and_rejects_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    contract = {
        "workflow": "conversion",
        "request": {"source": str(source)},
        "source_path": source,
        "backend": {"kind": "converter_router"},
        "policy": {"install_missing": False},
        "required_artifacts": ["report"],
    }
    recorder = WorkflowRunRecorder.start(run_dir, **contract)
    report = run_dir / "report.json"
    report.write_text('{"status":"pass"}\n', encoding="utf-8")
    recorder.record_artifact("report", report, kind="report")
    recorder.checkpoint("reported", ["report"])
    recorder.finalize("pass")

    resumed = WorkflowRunRecorder.start(run_dir, **contract, resume=True)

    assert resumed.manifest.status == "running"
    assert [item.phase for item in resumed.manifest.checkpoints] == ["reported"]
    assert [item.logical_name for item in resumed.manifest.artifacts] == ["report"]

    report.write_text('{"status":"fail"}\n', encoding="utf-8")
    WorkflowRunRecorder.start(run_dir, **contract, resume=True)

    checkpoint_path = (
        run_dir / recorder.manifest.checkpoints[0].artifact_paths["report"]
    )
    checkpoint_path.chmod(0o600)
    checkpoint_path.write_text('{"status":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_artifacts"):
        WorkflowRunRecorder.start(run_dir, **contract, resume=True)

    checkpoint_path.write_text('{"status":"pass"}\n', encoding="utf-8")
    report.write_text('{"status":"pass"}\n', encoding="utf-8")
    source.write_text("#usda 1.0\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_sha256"):
        WorkflowRunRecorder.start(run_dir, **contract, resume=True)


def test_run_recorder_resume_validates_every_sealed_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    contract = {
        "workflow": "scene.run",
        "request": {"schema_version": "request.v1"},
        "required_artifacts": [],
    }
    recorder = WorkflowRunRecorder.start(run_dir, **contract)
    early = run_dir / "state-early.json"
    late = run_dir / "state-late.json"
    early.write_text('{"phase":"early"}\n', encoding="utf-8")
    late.write_text('{"phase":"late"}\n', encoding="utf-8")
    recorder.record_artifact("state", early, kind="checkpoint")
    recorder.checkpoint("early", ["state"])
    recorder.record_artifact("state", late, kind="checkpoint")
    recorder.checkpoint("late", ["state"])
    recorder.finalize("blocked", failure={"code": "interrupted"})

    resumed = WorkflowRunRecorder.start(run_dir, **contract, resume=True)
    assert resumed.manifest.status == "running"

    early_checkpoint = (
        run_dir / recorder.manifest.checkpoints[0].artifact_paths["state"]
    )
    early_checkpoint.chmod(0o600)
    early_checkpoint.write_text('{"phase":"tampered"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint_artifacts"):
        WorkflowRunRecorder.start(run_dir, **contract, resume=True)


def test_run_recorder_repeated_resume_allows_logical_artifact_to_evolve_in_place(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    contract = {
        "workflow": "scene.run",
        "request": {"schema_version": "request.v1"},
        "required_artifacts": [],
    }
    recorder = WorkflowRunRecorder.start(run_dir, **contract)
    operation_trace = run_dir / "trace" / "operation_trace.json"
    operation_trace.parent.mkdir()

    operation_trace.write_text('{"turn":1}\n', encoding="utf-8")
    recorder.record_artifact("operation_trace", operation_trace, kind="trace")
    recorder.checkpoint("turn-1", ["operation_trace"])
    operation_trace.write_text('{"turn":2}\n', encoding="utf-8")
    recorder.record_artifact("operation_trace", operation_trace, kind="trace")
    recorder.checkpoint("turn-2", ["operation_trace"])
    recorder.finalize("blocked", failure={"code": "interrupted"})

    first_resume = WorkflowRunRecorder.start(run_dir, **contract, resume=True)
    operation_trace.write_text('{"turn":3}\n', encoding="utf-8")
    first_resume.record_artifact("operation_trace", operation_trace, kind="trace")
    first_resume.checkpoint("turn-3", ["operation_trace"])
    first_resume.finalize("blocked", failure={"code": "interrupted"})

    second_resume = WorkflowRunRecorder.start(run_dir, **contract, resume=True)

    assert second_resume.manifest.status == "running"
    assert [item.phase for item in second_resume.manifest.checkpoints] == [
        "turn-1",
        "turn-2",
        "turn-3",
    ]
    snapshot_paths = [
        item.artifact_paths["operation_trace"]
        for item in second_resume.manifest.checkpoints
    ]
    assert len(set(snapshot_paths)) == 3


@pytest.mark.parametrize("legacy_paths", [False, True])
def test_run_recorder_resume_accepts_superseded_legacy_live_checkpoint_paths(
    tmp_path: Path,
    legacy_paths: bool,
) -> None:
    run_dir = tmp_path / "run"
    contract = {
        "workflow": "scene.run",
        "request": {"schema_version": "request.v1"},
        "required_artifacts": [],
    }
    recorder = WorkflowRunRecorder.start(run_dir, **contract)
    state = run_dir / "operation_trace.json"
    state.write_text('{"turn":1}\n', encoding="utf-8")
    recorder.record_artifact("operation_trace", state, kind="trace")
    recorder.checkpoint("turn-1", ["operation_trace"])
    state.write_text('{"turn":2}\n', encoding="utf-8")
    recorder.record_artifact("operation_trace", state, kind="trace")
    recorder.checkpoint("turn-2", ["operation_trace"])
    recorder.finalize("blocked", failure={"code": "legacy"})

    manifest_path = run_dir / "workflow_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for checkpoint in manifest["checkpoints"]:
        checkpoint["artifact_paths"] = (
            {"operation_trace": "operation_trace.json"} if legacy_paths else {}
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resumed = WorkflowRunRecorder.start(run_dir, **contract, resume=True)

    assert resumed.manifest.status == "running"


def test_run_recorder_rejects_request_byte_tampering(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    contract = {
        "workflow": "scene.run",
        "request": {"schema_version": "request.v1", "policy": {"mode": "safe"}},
        "backend": {"scene_backend": "usd-cli"},
        "policy": {"mode": "safe"},
        "required_artifacts": [],
    }
    recorder = WorkflowRunRecorder.start(run_dir, **contract)
    recorder.record_artifact("request", run_dir / "request.json", kind="request")
    recorder.checkpoint("prepared", ["request"])
    recorder.finalize("blocked", failure={"code": "dry_run"})

    # The parsed object is unchanged, but the frozen request bytes are not.
    (run_dir / "request.json").write_text(
        '{"schema_version":"request.v1","policy":{"mode":"safe"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="request_sha256"):
        WorkflowRunRecorder.start(run_dir, **contract, resume=True)


def test_run_recorder_adopts_existing_request_without_rewriting_it(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    request_path = run_dir / "request.json"
    original = '{\n  "schema_version": "legacy.v1",\n  "mode": "workbench"\n}\n'
    request_path.write_text(original, encoding="utf-8")

    recorder = WorkflowRunRecorder.adopt(
        run_dir,
        workflow="scene.run",
        request={"schema_version": "legacy.v1", "mode": "workbench"},
        backend={"scene_backend": "workbench"},
    )

    assert request_path.read_text(encoding="utf-8") == original
    assert recorder.manifest.request_sha256
    assert recorder.manifest.request_path == "request.json"


def test_run_recorder_rejects_input_drift_at_success_finalization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    recorder = WorkflowRunRecorder.create(
        run_dir,
        workflow="materials.assign",
        request={"schema_version": "request.v1"},
        source_path=source,
    )
    (run_dir / "request.json").write_text(
        '{"schema_version": "request.v1"}\n',
        encoding="utf-8",
    )
    source.write_text("#usda 1.0\n# changed\n", encoding="utf-8")

    manifest = recorder.finalize("pass")

    assert manifest.status == "fail"
    assert manifest.failure == {
        "code": "workflow_input_drift",
        "invalid": ["request_sha256", "source_sha256"],
    }


def test_run_recorder_refuses_symlinked_manifest_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    recorder = WorkflowRunRecorder.create(
        run_dir,
        workflow="materials.assign",
        request={"schema_version": "request.v1"},
    )
    manifest_path = run_dir / "workflow_run_manifest.json"
    manifest_path.unlink()
    outside = tmp_path / "outside-manifest.json"
    outside.write_text('{"sentinel":"unchanged"}\n', encoding="utf-8")
    manifest_path.symlink_to(outside)

    with pytest.raises(ValueError, match="regular file"):
        recorder.finalize("pass")

    assert outside.read_text(encoding="utf-8") == '{"sentinel":"unchanged"}\n'


def test_run_recorder_refuses_symlinked_manifest_input(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    WorkflowRunRecorder.create(
        run_dir,
        workflow="materials.assign",
        request={"schema_version": "request.v1"},
    )
    manifest_path = run_dir / "workflow_run_manifest.json"
    outside = tmp_path / "outside-manifest.json"
    outside.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(outside)

    with pytest.raises(ValueError, match="symlinks"):
        WorkflowRunRecorder.resume(run_dir)


def test_run_recorder_refuses_symlinked_request_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside-request.json"
    outside.write_text('{"sentinel":"unchanged"}\n', encoding="utf-8")
    (run_dir / "request.json").symlink_to(outside)

    with pytest.raises(ValueError, match="regular file"):
        WorkflowRunRecorder.create(
            run_dir,
            workflow="materials.assign",
            request={"schema_version": "request.v1"},
        )

    assert outside.read_text(encoding="utf-8") == '{"sentinel":"unchanged"}\n'


def test_finalize_surfaces_integrity_violations_on_failed_runs(
    tmp_path: Path,
) -> None:
    """A run finalized as "fail" must still surface contract violations:
    silently dropping them lets a generic failure code coexist with a missing
    required artifact or a rewritten sealed request, and consumers that trust
    the failure object would treat the incomplete contract as intact."""

    run_dir = tmp_path / "run"
    recorder = WorkflowRunRecorder.create(
        run_dir,
        workflow="physics.apply",
        request={"schema_version": "request.v1"},
        required_artifacts=["output", "validation"],
    )
    output = run_dir / "output.usda"
    output.write_text("#usda 1.0\n", encoding="utf-8")
    recorder.record_artifact("output", output, kind="usd")
    # "validation" is required but never recorded; the sealed request is
    # also rewritten after sealing.
    (run_dir / "request.json").write_text('{"tampered": true}\n', encoding="utf-8")

    manifest = recorder.finalize(
        "fail", failure={"code": "physics_apply_failed", "returncode": 1}
    )

    assert manifest.status == "fail"
    assert manifest.failure is not None
    assert manifest.failure["code"] == "physics_apply_failed"
    integrity = manifest.failure["integrity"]
    assert integrity["missing_required_artifacts"] == ["validation"]
    assert "request_sha256" in integrity["input_drift"]


def test_finalize_keeps_failed_runs_with_intact_contracts_unannotated(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    recorder = WorkflowRunRecorder.create(
        run_dir,
        workflow="physics.apply",
        request={"schema_version": "request.v1"},
        required_artifacts=["output"],
    )
    output = run_dir / "output.usda"
    output.write_text("#usda 1.0\n", encoding="utf-8")
    recorder.record_artifact("output", output, kind="usd")

    manifest = recorder.finalize(
        "fail", failure={"code": "physics_apply_failed", "returncode": 1}
    )

    assert manifest.status == "fail"
    assert manifest.failure is not None
    assert "integrity" not in manifest.failure


def test_finalize_leaves_blocked_runs_unannotated(tmp_path: Path) -> None:
    """Dry runs and other intentional blocked finalizations legitimately
    lack terminal artifacts; only failed runs get the integrity annotation."""

    recorder = WorkflowRunRecorder.create(
        tmp_path / "run",
        workflow="physics.apply",
        request={"schema_version": "request.v1"},
        required_artifacts=["output", "validation"],
    )
    manifest = recorder.finalize(
        "blocked",
        failure={"code": "dry_run", "message": "prepared but not executed"},
    )

    assert manifest.status == "blocked"
    assert manifest.failure is not None
    assert "integrity" not in manifest.failure
