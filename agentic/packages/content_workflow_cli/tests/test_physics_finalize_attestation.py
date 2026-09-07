# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parent attestation of the physics finalize record.

raw/ stays child-writable through refinement turns, so the manifest seal must
bind to the exact bytes the parent wrote LAST (after render bookkeeping), via
a bounded, descriptor-safe digest that rejects substituted paths before
reading.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from content_agent_workflows.common.run_record import WorkflowRunRecorder

from content_workflow_cli import runner


def test_bounded_record_sha256_accepts_only_plain_regular_files(
    tmp_path: Path,
) -> None:
    record = tmp_path / "record.json"
    payload = b'{"validation_status": "conditional"}'
    record.write_bytes(payload)
    assert runner._bounded_record_sha256(record) == hashlib.sha256(payload).hexdigest()

    # Symlink substitution: the digest must never follow the link even when
    # the target holds identical bytes.
    outside = tmp_path / "outside.json"
    outside.write_bytes(payload)
    link = tmp_path / "link.json"
    link.symlink_to(outside)
    assert runner._bounded_record_sha256(link) is None

    # Hard-link substitution raises nlink above 1 and is refused.
    hard = tmp_path / "hard.json"
    import os

    os.link(record, hard)
    assert runner._bounded_record_sha256(hard) is None
    assert runner._bounded_record_sha256(record) is None  # nlink now 2 too
    hard.unlink()
    assert runner._bounded_record_sha256(record) is not None

    # Oversized files are refused BEFORE buffering into parent memory.
    assert runner._bounded_record_sha256(record, max_bytes=8) is None

    assert runner._bounded_record_sha256(tmp_path / "missing.json") is None


def _seal_manifest(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    attested: tuple[Path, str] | None,
    terminal_stop: dict | None = None,
):
    recorder = WorkflowRunRecorder.create(
        run_dir, workflow="physics.apply", request={"workflow": "physics.apply"}
    )
    monkeypatch.setattr(
        runner,
        "_physics_run_manifest_backend",
        lambda config: recorder.manifest.backend,
    )
    monkeypatch.setattr(
        runner,
        "_physics_run_manifest_policy",
        lambda config: recorder.manifest.policy,
    )
    monkeypatch.setattr(
        runner,
        "_write_external_physics_output_integrity",
        lambda **_: None,
    )
    return runner._finalize_physics_run_manifest(
        recorder=recorder,
        config=None,
        run_dir=run_dir,
        finalize_attestation=attested,
        terminal_stop=terminal_stop,
        status="fail",
        failure={"code": "physics_apply_failed", "returncode": 1},
    )


def _write_final_record(run_dir: Path) -> Path:
    """The record in its FINAL shape — after the render-bookkeeping rewrite."""
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    record_path = raw / "physics_finalize_result_1.json"
    record_path.write_text(
        json.dumps(
            {
                "validation_status": "conditional",
                "error": None,
                "rendered_frames": ["raw/frame_0.png"],
                "render_frame_receipt_path": "raw/physics_render_frame_receipt_1.json",
            }
        ),
        encoding="utf-8",
    )
    return record_path


def test_manifest_seals_record_attested_after_final_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest captured from the final bytes (including rendered_frames and
    render_frame_receipt_path) seals cleanly. Attesting before the render
    bookkeeping rewrite would make every finalized run report
    'finalize_record: bytes changed' — the regression this pins down."""

    run_dir = (tmp_path / "run").resolve()
    record_path = _write_final_record(run_dir)
    sha = runner._bounded_record_sha256(record_path)
    assert sha is not None
    manifest = _seal_manifest(run_dir, monkeypatch, attested=(record_path, sha))

    sealed = {artifact.logical_name for artifact in manifest.artifacts}
    assert "finalize_record" in sealed
    errors = (manifest.failure or {}).get("artifact_errors", [])
    assert not [error for error in errors if "finalize_record" in error]


def test_manifest_rejects_record_rewritten_after_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = (tmp_path / "run").resolve()
    record_path = _write_final_record(run_dir)
    sha = runner._bounded_record_sha256(record_path)
    assert sha is not None
    # A child turn (or a stale pre-rewrite attestation) leaves bytes that no
    # longer match the attested digest.
    record_path.write_text(json.dumps({"validation_status": "pass"}), encoding="utf-8")
    manifest = _seal_manifest(run_dir, monkeypatch, attested=(record_path, sha))

    sealed = {artifact.logical_name for artifact in manifest.artifacts}
    assert "finalize_record" not in sealed
    errors = (manifest.failure or {}).get("artifact_errors", [])
    assert any("finalize_record" in error for error in errors)


def test_manifest_rejects_record_substituted_by_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same bytes through a substituted link must not seal: the bounded
    reader refuses the path before hashing."""

    run_dir = (tmp_path / "run").resolve()
    record_path = _write_final_record(run_dir)
    sha = runner._bounded_record_sha256(record_path)
    assert sha is not None
    outside = tmp_path / "outside.json"
    outside.write_bytes(record_path.read_bytes())
    record_path.unlink()
    record_path.symlink_to(outside)
    manifest = _seal_manifest(run_dir, monkeypatch, attested=(record_path, sha))

    sealed = {artifact.logical_name for artifact in manifest.artifacts}
    assert "finalize_record" not in sealed
    errors = (manifest.failure or {}).get("artifact_errors", [])
    assert any("finalize_record" in error for error in errors)


def test_manifest_attaches_parent_terminal_stop_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The conditional stop marker recorded by the visual refinement loop
    must reach the manifest failure object: "physics_apply_failed" alone is
    ambiguous with a failed visual-review child."""

    record_path = _write_final_record(tmp_path)
    sha = runner._bounded_record_sha256(record_path)
    manifest = _seal_manifest(
        tmp_path,
        monkeypatch,
        attested=(record_path, sha),
        terminal_stop={
            "kind": "conditional_validation_stop",
            "history_status": "max_iterations_reached",
        },
    )
    assert manifest.failure is not None
    assert manifest.failure["terminal_stop"]["kind"] == "conditional_validation_stop"


def test_manifest_omits_terminal_stop_without_parent_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No parent-recorded stop (child_failed, runtime_failed, or any other
    nonzero path) → no marker in the manifest, so scoring keeps the run FAIL."""

    record_path = _write_final_record(tmp_path)
    sha = runner._bounded_record_sha256(record_path)
    manifest = _seal_manifest(
        tmp_path,
        monkeypatch,
        attested=(record_path, sha),
    )
    assert manifest.failure is not None
    assert "terminal_stop" not in manifest.failure


def test_finalize_once_invalidates_prior_attestation_before_each_attempt(
    tmp_path: Path,
) -> None:
    """A finalize attempt that fails must not leave an earlier iteration
    attested: the state is cleared before the attempt starts, so the sealed
    manifest can never bind a stale terminal record."""

    record_path = _write_final_record(tmp_path)
    sha = runner._bounded_record_sha256(record_path)
    run_state = {"finalize_attestation": (record_path, sha)}
    with pytest.raises(Exception):
        runner._finalize_physics_once(
            config=None,
            run_dir=tmp_path,
            inspection_pin=None,
            session_id=None,
            iteration=2,
            trace_writer=None,
            usd_cli_session=None,
            run_state=run_state,
        )
    assert run_state["finalize_attestation"] is None


def test_multi_body_runtime_skip_evidence_requires_explicit_report() -> None:
    """The conditional-stop marker binds to the workflow's explicit
    multi-body runtime-skip report; conditional verdicts from incomplete
    authoring (runtime actually evaluated) must not qualify."""

    skip = {"not_evaluated": True, "enabled_rigid_body_count": 3}
    assert runner._multi_body_runtime_skip_evidence(skip) is True

    evaluated = {"not_evaluated": False, "enabled_rigid_body_count": 3}
    assert runner._multi_body_runtime_skip_evidence(evaluated) is False
    single_body = {"not_evaluated": True, "enabled_rigid_body_count": 1}
    assert runner._multi_body_runtime_skip_evidence(single_body) is False
    assert runner._multi_body_runtime_skip_evidence({}) is False
    forged_truthy = {"not_evaluated": 1, "enabled_rigid_body_count": 3}
    assert runner._multi_body_runtime_skip_evidence(forged_truthy) is False
    bool_count = {"not_evaluated": True, "enabled_rigid_body_count": True}
    assert runner._multi_body_runtime_skip_evidence(bool_count) is False


def test_multi_body_runtime_skip_accepts_multi_root_reports() -> None:
    """validate_physics_runtime_multi_body records skip_reason and
    body_prim_paths (no enabled_rigid_body_count) when the bodies share no
    common Xformable placement ancestor — a supported multi-root shape."""

    multi_root = {
        "not_evaluated": True,
        "skip_reason": "no_common_xformable_ancestor",
        "body_prim_paths": ["/a/body1", "/b/body2"],
    }
    assert runner._multi_body_runtime_skip_evidence(multi_root) is True

    single_root_body = {
        "not_evaluated": True,
        "skip_reason": "no_common_xformable_ancestor",
        "body_prim_paths": ["/a/body1"],
    }
    assert runner._multi_body_runtime_skip_evidence(single_root_body) is False
    engine_none = {
        "not_evaluated": True,
        "skip_reason": "engine_none",
        "body_prim_paths": ["/a/body1", "/b/body2"],
    }
    assert runner._multi_body_runtime_skip_evidence(engine_none) is False
    forged_paths = {
        "not_evaluated": True,
        "skip_reason": "no_common_xformable_ancestor",
        "body_prim_paths": "/a,/b",
    }
    assert runner._multi_body_runtime_skip_evidence(forged_paths) is False


def test_output_closure_digest_covers_sidecar_dependencies(
    tmp_path: Path,
) -> None:
    """The pinned digest must change when a localized dependency changes,
    and fail closed on symlinked members — the root layer alone cannot
    prove the composed asset is the finalized one."""

    output = tmp_path / "physics.usdc"
    output.write_text("#usda 1.0\n", encoding="utf-8")
    root_only = runner._physics_output_closure_sha256(output, within=tmp_path)
    assert root_only is not None

    sidecar = tmp_path / "physics_assets"
    sidecar.mkdir()
    dep = sidecar / "geometry.usd"
    dep.write_text("#usda 1.0\n# geo\n", encoding="utf-8")
    with_dep = runner._physics_output_closure_sha256(output, within=tmp_path)
    assert with_dep is not None
    assert with_dep != root_only

    dep.write_text("#usda 1.0\n# mutated\n", encoding="utf-8")
    mutated = runner._physics_output_closure_sha256(output, within=tmp_path)
    assert mutated not in (None, with_dep, root_only)

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"external")
    link = sidecar / "swapped.bin"
    link.symlink_to(outside)
    assert runner._physics_output_closure_sha256(output, within=tmp_path) is None


def test_bounded_record_sha256_rejects_fifo_without_blocking(
    tmp_path: Path,
) -> None:
    """A child-planted FIFO must be rejected promptly: without O_NONBLOCK the
    open itself blocks until a writer appears, hanging manifest finalization."""

    import os

    fifo = tmp_path / "physics_finalize_result_1.json"
    os.mkfifo(fifo)
    assert runner._bounded_record_sha256(fifo) is None


def test_closure_digest_refuses_oversized_sidecar_members(tmp_path: Path) -> None:
    """A child-planted multi-terabyte sparse file is cheap to create but must
    overflow the closure budget (digest -> None, downgrade refused) instead
    of forcing unbounded reads after the child timeout."""

    output = tmp_path / "physics.usdc"
    output.write_text("#usda 1.0\n", encoding="utf-8")
    sidecar = tmp_path / "physics_assets"
    sidecar.mkdir()
    (sidecar / "ok.png").write_bytes(b"png-bytes")
    assert runner._physics_output_closure_sha256(output, within=tmp_path) is not None

    sparse = sidecar / "huge.bin"
    with open(sparse, "wb") as handle:
        handle.truncate(runner.MAX_CLOSURE_MEMBER_BYTES + 1)
    assert runner._physics_output_closure_sha256(output, within=tmp_path) is None


def test_bounded_file_sha256_caps_reads(tmp_path: Path) -> None:
    small = tmp_path / "small.bin"
    small.write_bytes(b"x" * 64)
    bounded = runner._bounded_file_sha256(small, max_bytes=64)
    assert bounded is not None and bounded[1] == 64
    assert runner._bounded_file_sha256(small, max_bytes=63) is None
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"y")
    link = tmp_path / "link.bin"
    link.symlink_to(outside)
    assert runner._bounded_file_sha256(link, max_bytes=64) is None


def test_closure_digest_refuses_symlinked_sidecar_directory(tmp_path: Path) -> None:
    """A sidecar DIRECTORY replaced by a symlink fails closed like a
    symlinked member: a root-only digest could still match a pin computed
    before the swap."""

    import shutil as _shutil

    output = tmp_path / "physics.usdc"
    output.write_text("#usda 1.0\n", encoding="utf-8")
    sidecar = tmp_path / "physics_assets"
    sidecar.mkdir()
    (sidecar / "ok.png").write_bytes(b"png-bytes")
    assert runner._physics_output_closure_sha256(output, within=tmp_path) is not None

    real = tmp_path / "elsewhere"
    _shutil.move(str(sidecar), str(real))
    (tmp_path / "physics_assets").symlink_to(real)
    assert runner._physics_output_closure_sha256(output, within=tmp_path) is None
