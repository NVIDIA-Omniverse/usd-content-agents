# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared Joint Rigger artifact transactions."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import multiprocessing
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any

import pytest

from world_understanding.functions.physics.joint_rigger import artifacts
from world_understanding.functions.physics.joint_rigger.artifacts import (
    CommittedArtifactPublicationCleanupError,
    ConcurrentArtifactPublicationError,
    JointRiggerArtifactTargets,
    StagedArtifact,
    create_staged_artifact_targets,
    invalidate_artifact_targets,
    promote_staged_artifacts,
    sidecar_dependency_bundle_sha256,
    staged_promotion_artifacts,
    validate_artifact_targets,
)


def _controlled_promotion_process(
    raw_artifacts: list[tuple[str, str, str]],
    pause_after: str,
    fail_on: str,
    entered: Any,
    release: Any,
    outcome: Any,
) -> None:
    """Promote in a child, pausing after one move and then forcing failure."""

    promotion = [
        StagedArtifact(Path(staged), Path(target), label)
        for staged, target, label in raw_artifacts
    ]
    original_replace = artifacts._replace_entry

    def controlled_replace(source: Any, target: Any) -> None:
        if source.path == Path(fail_on):
            raise OSError("forced concurrent promotion failure")
        original_replace(source, target)
        if source.path == Path(pause_after):
            entered.set()
            if not release.wait(timeout=15):
                raise TimeoutError("parent did not release paused promotion")

    artifacts._replace_entry = controlled_replace
    try:
        promote_staged_artifacts(promotion)
    except Exception as exc:
        outcome.put((type(exc).__name__, str(exc)))
    else:  # pragma: no cover - this worker is required to force a failure
        outcome.put(("succeeded", ""))
    finally:
        artifacts._replace_entry = original_replace


def _crash_while_holding_publication_locks(
    raw_targets: list[str],
    acquired: Any,
) -> None:
    """Exit without context cleanup after acquiring interprocess locks."""

    with artifacts._publication_target_locks(Path(path) for path in raw_targets):
        acquired.set()
        os._exit(23)


def _targets(tmp_path: Path, *, sidecar: bool = False) -> JointRiggerArtifactTargets:
    return JointRiggerArtifactTargets(
        output_path=tmp_path / "rigged.usda",
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
        sidecar_path=tmp_path / "rigged_assets" if sidecar else None,
    )


def _write_staged_bundle(
    staged: JointRiggerArtifactTargets,
    *,
    marker: str = "new",
) -> None:
    staged.output_path.write_text(f"{marker} output", encoding="utf-8")
    staged.diagnostics_path.write_text(f"{marker} diagnostics", encoding="utf-8")
    staged.result_path.write_text(f"{marker} result", encoding="utf-8")
    if staged.sidecar_path is not None:
        staged.sidecar_path.mkdir()
        (staged.sidecar_path / "asset.txt").write_text(
            f"{marker} sidecar",
            encoding="utf-8",
        )


def test_staging_cleanup_identity_uses_parent_when_basenames_collide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifacts.secrets, "token_hex", lambda _size: "same")
    targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "output" / "artifact.usda",
        diagnostics_path=tmp_path / "diagnostics" / "artifact.usda",
        result_path=tmp_path / "result" / "artifact.usda",
    )
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    staged_paths = (
        bundle.staged_targets.output_path,
        bundle.staged_targets.diagnostics_path,
        bundle.staged_targets.result_path,
    )
    assert staged_paths[1].name == staged_paths[2].name == "artifact.usda"

    try:
        staged_promotion_artifacts(bundle)
        reservations = {
            reservation.parent.locator_path / reservation.name: reservation
            for reservation in bundle._cleanup_reservations
        }
        for path in staged_paths:
            owned_path = path if path in reservations else path.parent
            metadata = os.stat(owned_path, follow_symlinks=False)
            reservation = reservations[owned_path]
            assert reservation.owned_identity == (metadata.st_dev, metadata.st_ino)
    finally:
        bundle.cleanup()

    assert not any(path.exists() for path in staged_paths)


def test_staging_cleanup_revokes_created_file_binding_capability(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    staged = bundle.staged_targets
    stale_binder = staged._created_file_binder
    assert stale_binder is not None
    stale_path = staged.output_path

    bundle.cleanup()

    assert staged._created_file_binder is None
    stale_path.write_bytes(b"post-cleanup replacement")
    metadata = os.stat(stale_path, follow_symlinks=False)
    try:
        with pytest.raises(RuntimeError, match="binding has been revoked"):
            stale_binder(stale_path, metadata)
        with pytest.raises(
            RuntimeError,
            match="requires facade-owned staging cleanup",
        ):
            staged._bind_created_file(stale_path, metadata)
        assert all(
            reservation.owned_descriptor < 0 and reservation.payload_descriptor < 0
            for reservation in bundle._cleanup_reservations
        )
    finally:
        stale_path.unlink()


def test_backend_report_owner_substitution_preserves_foreign_and_owned_trees(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    staged_report = bundle.staged_targets.result_path
    owner = staged_report.parent
    displaced_owner = tmp_path / "owned-report-owner-displaced"
    staged_report.write_text("owned backend report", encoding="utf-8")

    owner.rename(displaced_owner)
    owner.mkdir(mode=0o700)
    foreign_report = owner / staged_report.name
    foreign_report.write_text("foreign replacement", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Staging owner changed inode"):
        artifacts._require_staging_owner_unchanged(bundle, staged_report)

    with pytest.raises(RuntimeError, match="replacement preserved"):
        bundle.cleanup()

    assert foreign_report.read_text(encoding="utf-8") == "foreign replacement"
    assert (displaced_owner / staged_report.name).read_text(encoding="utf-8") == (
        "owned backend report"
    )
    assert all(reservation.closed for reservation in bundle._cleanup_reservations)


def test_backend_report_owner_rename_is_reported_and_preserved(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    staged_report = bundle.staged_targets.diagnostics_path
    owner = staged_report.parent
    displaced_owner = tmp_path / "owned-report-owner-displaced"
    staged_report.write_text("owned backend report", encoding="utf-8")

    owner.rename(displaced_owner)

    with pytest.raises(RuntimeError, match="disappeared from its reserved name"):
        bundle.cleanup()

    assert (displaced_owner / staged_report.name).read_text(encoding="utf-8") == (
        "owned backend report"
    )
    assert not owner.exists()
    assert all(reservation.closed for reservation in bundle._cleanup_reservations)


def test_cleanup_does_not_revisit_same_parent_after_removing_owned_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    staged_report = bundle.staged_targets.diagnostics_path
    owner = staged_report.parent
    report_reservation = next(
        reservation
        for reservation in bundle._cleanup_reservations
        if reservation.parent.locator_path / reservation.name == owner
    )
    original_identity = report_reservation.owned_identity
    original_remove = artifacts._remove_descriptor_entry
    substituted = False

    def replace_after_owned_removal(*args: Any, **kwargs: Any) -> None:
        nonlocal substituted
        original_remove(*args, **kwargs)
        if kwargs.get("source_descriptor") != report_reservation.owned_descriptor:
            return
        owner.mkdir()
        (owner / "foreign.txt").write_text("foreign", encoding="utf-8")
        substituted = True

    monkeypatch.setattr(
        artifacts,
        "_remove_descriptor_entry",
        replace_after_owned_removal,
    )

    bundle.cleanup()

    assert substituted
    replacement_metadata = owner.stat()
    assert original_identity != (
        replacement_metadata.st_dev,
        replacement_metadata.st_ino,
    )
    assert (owner / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert report_reservation.closed
    assert report_reservation.owned_descriptor == -1
    assert report_reservation.owned_identity is None


def test_second_cleanup_does_not_revisit_retired_owner_reservations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    owner_reservations = [
        reservation
        for reservation in bundle._cleanup_reservations
        if reservation.is_owner_directory
    ]
    assert owner_reservations
    assert all(
        reservation.owned_identity is not None for reservation in owner_reservations
    )

    bundle.cleanup()

    assert all(reservation.closed for reservation in bundle._cleanup_reservations)
    assert all(
        reservation.owned_descriptor == -1
        and reservation.owned_identity is None
        and reservation.payload_descriptor == -1
        and reservation.payload_identity is None
        for reservation in bundle._cleanup_reservations
    )
    report_reservation = owner_reservations[0]
    owner = report_reservation.parent.locator_path / report_reservation.name
    owner.mkdir()
    (owner / "foreign.txt").write_text("foreign", encoding="utf-8")

    # Exact inode reuse cannot be forced portably. The stronger security
    # invariant is that a retired cleanup capability never consults the
    # namespace again, even if the filesystem recycles the original identity.
    def fail_if_revisited(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("retired cleanup reservation was revisited")

    monkeypatch.setattr(artifacts, "_open_bound_directory", fail_if_revisited)
    monkeypatch.setattr(
        artifacts,
        "_preserve_unbound_staging_reservation",
        fail_if_revisited,
    )
    monkeypatch.setattr(
        artifacts,
        "_remove_staging_reservation_entry",
        fail_if_revisited,
    )
    bundle.cleanup()

    assert (owner / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert report_reservation.closed
    assert report_reservation.owned_descriptor == -1
    assert report_reservation.owned_identity is None


def test_unvalidated_backend_root_is_preserved_with_cleanup_evidence(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    staged_output = bundle.staged_targets.output_path
    staged_output.write_text("unvalidated backend output", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no descriptor-bound cleanup identity"):
        bundle.cleanup()

    assert staged_output.read_text(encoding="utf-8") == "unvalidated backend output"
    assert all(reservation.closed for reservation in bundle._cleanup_reservations)
    staged_output.unlink()


def test_unvalidated_backend_root_in_recreated_parent_is_reported_and_preserved(
    tmp_path: Path,
) -> None:
    live_parent = tmp_path / "live"
    displaced_parent = tmp_path / "displaced-live"
    bundle = create_staged_artifact_targets(_targets(live_parent))
    staged_output = bundle.staged_targets.output_path

    live_parent.rename(displaced_parent)
    live_parent.mkdir()
    staged_output.write_text("unvalidated output in recreated parent", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no descriptor-bound cleanup identity"):
        bundle.cleanup()

    assert staged_output.read_text(encoding="utf-8") == (
        "unvalidated output in recreated parent"
    )
    assert not (displaced_parent / staged_output.name).exists()
    assert all(reservation.closed for reservation in bundle._cleanup_reservations)
    staged_output.unlink()


def test_backend_root_substitution_during_descriptor_bind_preserves_both_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    staged_output = bundle.staged_targets.output_path
    staged_output.write_text("owned backend output", encoding="utf-8")
    validated_metadata = staged_output.lstat()
    displaced_output = tmp_path / "owned-output-displaced.usda"
    original_optional_identity = artifacts._optional_descriptor_entry_identity
    original_open = artifacts.os.open
    real_close = artifacts.os.close
    binding_descriptor: int | None = None
    close_attempts = 0
    substituted = False
    output_reservation = next(
        reservation
        for reservation in bundle._cleanup_reservations
        if reservation.parent.locator_path == staged_output.parent
        and reservation.name == staged_output.name
    )

    def track_binding_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal binding_descriptor
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if (
            path == output_reservation.name
            and dir_fd == output_reservation.parent.descriptor
        ):
            binding_descriptor = descriptor
        return descriptor

    def substitute_before_lexical_check(
        parent_descriptor: int,
        entry_name: str,
    ) -> tuple[int, int] | None:
        nonlocal substituted
        if (
            parent_descriptor == output_reservation.parent.descriptor
            and entry_name == output_reservation.name
            and not substituted
        ):
            substituted = True
            staged_output.rename(displaced_output)
            staged_output.write_text("foreign replacement", encoding="utf-8")
        return original_optional_identity(parent_descriptor, entry_name)

    def track_binding_close(descriptor: int) -> None:
        nonlocal close_attempts
        real_close(descriptor)
        if descriptor == binding_descriptor:
            close_attempts += 1

    monkeypatch.setattr(artifacts.os, "open", track_binding_open)
    monkeypatch.setattr(
        artifacts,
        "_optional_descriptor_entry_identity",
        substitute_before_lexical_check,
    )
    monkeypatch.setattr(artifacts.os, "close", track_binding_close)
    with pytest.raises(RuntimeError, match="changed inode"):
        artifacts._bind_staging_cleanup_identity(
            bundle,
            staged_output,
            validated_metadata,
        )

    assert substituted
    assert binding_descriptor is not None
    assert close_attempts == 1
    assert output_reservation.owned_identity is None
    with pytest.raises(RuntimeError, match="no descriptor-bound cleanup identity"):
        bundle.cleanup()
    assert staged_output.read_text(encoding="utf-8") == "foreign replacement"
    assert displaced_output.read_text(encoding="utf-8") == "owned backend output"


def test_staging_bind_skips_unrelated_and_rejects_incomplete_reservation(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    staged_output = bundle.staged_targets.output_path
    staged_output.write_text("owned backend output", encoding="utf-8")
    metadata = staged_output.lstat()
    output_reservation = next(
        reservation
        for reservation in bundle._cleanup_reservations
        if reservation.parent.locator_path == staged_output.parent
        and reservation.name == staged_output.name
    )
    object.__setattr__(
        bundle,
        "_cleanup_reservations",
        tuple(reversed(bundle._cleanup_reservations)),
    )
    output_reservation.owned_identity = (metadata.st_dev, metadata.st_ino)

    with pytest.raises(
        RuntimeError,
        match="incomplete descriptor-bound cleanup ownership",
    ):
        artifacts._bind_staging_cleanup_identity(bundle, staged_output, metadata)

    output_reservation.owned_identity = None
    artifacts._bind_staging_cleanup_identity(bundle, staged_output, metadata)
    assert output_reservation.owned_descriptor >= 0
    bundle.cleanup()
    assert not staged_output.exists()


def test_bound_backend_root_descriptor_prevents_reused_inode_cleanup(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    staged_output = bundle.staged_targets.output_path
    staged_output.write_text("owned backend output", encoding="utf-8")
    metadata = staged_output.lstat()
    output_reservation = next(
        reservation
        for reservation in bundle._cleanup_reservations
        if reservation.parent.locator_path == staged_output.parent
        and reservation.name == staged_output.name
    )
    artifacts._bind_staging_cleanup_identity(bundle, staged_output, metadata)
    owned_descriptor = output_reservation.owned_descriptor
    owned_identity = output_reservation.owned_identity
    assert owned_descriptor >= 0
    assert owned_identity is not None

    staged_output.unlink()
    staged_output.write_text("foreign replacement", encoding="utf-8")
    replacement = staged_output.stat()
    assert (replacement.st_dev, replacement.st_ino) != owned_identity

    with pytest.raises(RuntimeError, match="replacement preserved"):
        bundle.cleanup()

    assert staged_output.read_text(encoding="utf-8") == "foreign replacement"
    assert output_reservation.closed
    assert output_reservation.owned_descriptor == -1
    assert output_reservation.owned_identity is None
    with pytest.raises(OSError):
        os.fstat(owned_descriptor)


def test_bound_backend_root_rename_away_is_reported_and_preserved(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    staged_output = bundle.staged_targets.output_path
    displaced_output = tmp_path / "owned-output-displaced.usda"
    staged_output.write_text("owned backend output", encoding="utf-8")
    artifacts._bind_staging_cleanup_identity(
        bundle,
        staged_output,
        staged_output.lstat(),
    )

    staged_output.rename(displaced_output)

    with pytest.raises(RuntimeError, match="without reaching its exact publication"):
        bundle.cleanup()

    assert displaced_output.read_text(encoding="utf-8") == "owned backend output"


def test_bound_backend_root_at_final_name_requires_recorded_promotion(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path)
    bundle = create_staged_artifact_targets(targets)
    staged_output = bundle.staged_targets.output_path
    staged_output.write_text("owned backend output", encoding="utf-8")
    artifacts._bind_staging_cleanup_identity(
        bundle,
        staged_output,
        staged_output.lstat(),
    )

    staged_output.rename(targets.output_path)

    with pytest.raises(RuntimeError, match="without reaching its exact publication"):
        bundle.cleanup()

    assert targets.output_path.read_text(encoding="utf-8") == "owned backend output"


def test_bound_backend_root_hardlink_is_reported_without_deleting_either_name(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    staged_output = bundle.staged_targets.output_path
    alias = tmp_path / "root-alias.usda"
    staged_output.write_text("owned backend output", encoding="utf-8")
    artifacts._bind_staging_cleanup_identity(
        bundle,
        staged_output,
        staged_output.lstat(),
    )

    os.link(staged_output, alias)

    with pytest.raises(RuntimeError, match="gained additional links"):
        bundle.cleanup()

    assert staged_output.read_text(encoding="utf-8") == "owned backend output"
    assert alias.read_text(encoding="utf-8") == "owned backend output"


def test_bound_backend_root_hardlink_race_is_reported_after_staged_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    staged_output = bundle.staged_targets.output_path
    alias = tmp_path / "raced-root-alias.usda"
    staged_output.write_text("owned backend output", encoding="utf-8")
    artifacts._bind_staging_cleanup_identity(
        bundle,
        staged_output,
        staged_output.lstat(),
    )
    output_reservation = next(
        reservation
        for reservation in bundle._cleanup_reservations
        if reservation.parent.locator_path == staged_output.parent
        and reservation.name == staged_output.name
    )
    original_remove = artifacts._remove_descriptor_entry
    injected = False

    def hardlink_after_precheck(*args: Any, **kwargs: Any) -> None:
        nonlocal injected
        if kwargs.get("source_descriptor") == output_reservation.owned_descriptor:
            os.link(staged_output, alias)
            injected = True
        original_remove(*args, **kwargs)

    monkeypatch.setattr(
        artifacts,
        "_remove_descriptor_entry",
        hardlink_after_precheck,
    )

    with pytest.raises(RuntimeError, match="retained an unauthorized linked name"):
        bundle.cleanup()

    assert injected
    assert not staged_output.exists()
    assert alias.read_text(encoding="utf-8") == "owned backend output"


def test_promoted_backend_root_hardlink_is_reported_and_preserved(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    promote_staged_artifacts(staged_promotion_artifacts(bundle))
    alias = tmp_path / "promoted-root-alias.usda"
    os.link(targets.output_path, alias)

    with pytest.raises(RuntimeError, match="held inode preserved elsewhere"):
        bundle.cleanup()

    assert targets.output_path.read_text(encoding="utf-8") == "new output"
    assert alias.read_text(encoding="utf-8") == "new output"
    assert targets.output_path.stat().st_nlink == 2


def test_promotion_rejects_target_created_after_staging_reservation(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    targets.output_path.write_text("late foreign output", encoding="utf-8")

    try:
        with pytest.raises(
            RuntimeError,
            match="Artifact target changed after staged targets were created",
        ):
            promote_staged_artifacts(staged_promotion_artifacts(bundle))

        assert targets.output_path.read_text(encoding="utf-8") == (
            "late foreign output"
        )
        assert not targets.diagnostics_path.exists()
        assert not targets.result_path.exists()
        assert bundle.staged_targets.output_path.read_text(encoding="utf-8") == (
            "new output"
        )
    finally:
        bundle.cleanup()


def test_promotion_rejects_existing_target_inode_replaced_after_capture(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path)
    targets.output_path.write_text("captured old output", encoding="utf-8")
    targets.diagnostics_path.write_text("captured old diagnostics", encoding="utf-8")
    targets.result_path.write_text("captured old result", encoding="utf-8")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    replacement = tmp_path / "foreign-output-replacement.usda"
    replacement.write_text("late foreign output", encoding="utf-8")
    replacement.replace(targets.output_path)

    try:
        with pytest.raises(
            RuntimeError,
            match="Artifact target changed after staged targets were created",
        ):
            promote_staged_artifacts(staged_promotion_artifacts(bundle))

        assert targets.output_path.read_text(encoding="utf-8") == (
            "late foreign output"
        )
        assert targets.diagnostics_path.read_text(encoding="utf-8") == (
            "captured old diagnostics"
        )
        assert targets.result_path.read_text(encoding="utf-8") == (
            "captured old result"
        )
        assert bundle.staged_targets.output_path.read_text(encoding="utf-8") == (
            "new output"
        )
    finally:
        bundle.cleanup()


@pytest.mark.parametrize(
    "target_field",
    ["output_path", "diagnostics_path", "result_path"],
)
def test_promotion_rejects_existing_target_same_inode_mutation_after_capture(
    tmp_path: Path,
    target_field: str,
) -> None:
    targets = _targets(tmp_path)
    for path, payload in (
        (targets.output_path, "captured old output"),
        (targets.diagnostics_path, "captured old diagnostics"),
        (targets.result_path, "captured old result"),
    ):
        path.write_text(payload, encoding="utf-8")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    mutated = getattr(targets, target_field)
    captured_identity = (mutated.stat().st_dev, mutated.stat().st_ino)
    mutated.write_text("same inode foreign mutation", encoding="utf-8")
    assert (mutated.stat().st_dev, mutated.stat().st_ino) == captured_identity

    try:
        with pytest.raises(
            RuntimeError,
            match="Artifact target changed after staged targets were created",
        ):
            promote_staged_artifacts(staged_promotion_artifacts(bundle))

        assert mutated.read_text(encoding="utf-8") == "same inode foreign mutation"
        assert bundle.staged_targets.output_path.read_text(encoding="utf-8") == (
            "new output"
        )
    finally:
        bundle.cleanup()


@pytest.mark.parametrize("mutation", ["same-inode-write", "identical-replacement"])
def test_promotion_rejects_existing_sidecar_descendant_mutation_after_capture(
    tmp_path: Path,
    mutation: str,
) -> None:
    targets = _targets(tmp_path, sidecar=True)
    targets.output_path.write_text("captured old output", encoding="utf-8")
    targets.diagnostics_path.write_text("captured old diagnostics", encoding="utf-8")
    targets.result_path.write_text("captured old result", encoding="utf-8")
    assert targets.sidecar_path is not None
    targets.sidecar_path.mkdir()
    child = targets.sidecar_path / "asset.bin"
    child.write_bytes(b"captured sidecar bytes")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)

    if mutation == "same-inode-write":
        child.write_bytes(b"mutated sidecar bytes")
        expected = b"mutated sidecar bytes"
    else:
        replacement = tmp_path / "byte-identical-sidecar-replacement.bin"
        replacement.write_bytes(child.read_bytes())
        replacement.replace(child)
        expected = b"captured sidecar bytes"

    try:
        with pytest.raises(
            RuntimeError,
            match="Artifact target changed|initial publication target",
        ):
            promote_staged_artifacts(staged_promotion_artifacts(bundle))

        assert child.read_bytes() == expected
        assert bundle.staged_targets.output_path.read_text(encoding="utf-8") == (
            "new output"
        )
    finally:
        bundle.cleanup()


def test_promotion_rechecks_initial_targets_after_prebackup_validator(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    validator_calls = 0

    def create_late_target() -> None:
        nonlocal validator_calls
        validator_calls += 1
        targets.result_path.write_text("late foreign result", encoding="utf-8")

    try:
        with pytest.raises(
            RuntimeError,
            match="Artifact target changed after staged targets were created",
        ):
            promote_staged_artifacts(
                staged_promotion_artifacts(bundle),
                prebackup_validator=create_late_target,
            )

        assert validator_calls == 1
        assert targets.result_path.read_text(encoding="utf-8") == (
            "late foreign result"
        )
        assert not targets.output_path.exists()
        assert not targets.diagnostics_path.exists()
        assert bundle.staged_targets.result_path.read_text(encoding="utf-8") == (
            "new result"
        )
    finally:
        bundle.cleanup()


def test_promotion_rechecks_same_inode_content_after_prebackup_validator(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path)
    targets.output_path.write_text("captured old output", encoding="utf-8")
    targets.diagnostics_path.write_text("captured old diagnostics", encoding="utf-8")
    targets.result_path.write_text("captured old result", encoding="utf-8")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    captured_identity = (
        targets.result_path.stat().st_dev,
        targets.result_path.stat().st_ino,
    )

    def mutate_existing_result() -> None:
        targets.result_path.write_text(
            "same inode validator mutation", encoding="utf-8"
        )
        assert (
            targets.result_path.stat().st_dev,
            targets.result_path.stat().st_ino,
        ) == captured_identity

    try:
        with pytest.raises(
            RuntimeError,
            match="Artifact target changed after staged targets were created",
        ):
            promote_staged_artifacts(
                staged_promotion_artifacts(bundle),
                prebackup_validator=mutate_existing_result,
            )
        assert targets.result_path.read_text(encoding="utf-8") == (
            "same inode validator mutation"
        )
    finally:
        bundle.cleanup()


def test_post_backup_mutation_aborts_and_restores_mutated_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path)
    targets.output_path.write_text("captured old output", encoding="utf-8")
    targets.diagnostics_path.write_text("captured old diagnostics", encoding="utf-8")
    targets.result_path.write_text("captured old result", encoding="utf-8")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    real_replace = artifacts._replace_entry
    mutated = False

    def mutate_after_result_backup(source: Any, destination: Any) -> None:
        nonlocal mutated
        real_replace(source, destination)
        if source.path == targets.result_path and ".rollback-" in str(
            destination.path.parent
        ):
            destination.path.write_text(
                "post-backup same inode mutation",
                encoding="utf-8",
            )
            mutated = True

    monkeypatch.setattr(artifacts, "_replace_entry", mutate_after_result_backup)

    try:
        with pytest.raises(
            RuntimeError,
            match="Artifact backup (metadata|content) changed",
        ):
            promote_staged_artifacts(staged_promotion_artifacts(bundle))

        assert mutated
        assert targets.result_path.read_text(encoding="utf-8") == (
            "post-backup same inode mutation"
        )
        assert targets.output_path.read_text(encoding="utf-8") == (
            "captured old output"
        )
        assert bundle.staged_targets.output_path.read_text(encoding="utf-8") == (
            "new output"
        )
    finally:
        bundle.cleanup()


def test_committed_cleanup_preserves_postcommit_mutated_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path)
    targets.output_path.write_text("captured old output", encoding="utf-8")
    targets.diagnostics_path.write_text("captured old diagnostics", encoding="utf-8")
    targets.result_path.write_text("captured old result", encoding="utf-8")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    real_remove = artifacts._remove_committed_backup_if_unchanged
    preserved_backup: Path | None = None

    def mutate_then_remove(backup: Any) -> None:
        nonlocal preserved_backup
        if preserved_backup is None:
            preserved_backup = backup.artifact_entry.path
            preserved_backup.write_text(
                "postcommit foreign backup mutation",
                encoding="utf-8",
            )
        real_remove(backup)

    monkeypatch.setattr(
        artifacts,
        "_remove_committed_backup_if_unchanged",
        mutate_then_remove,
    )

    try:
        with pytest.raises(CommittedArtifactPublicationCleanupError):
            promote_staged_artifacts(staged_promotion_artifacts(bundle))

        assert targets.output_path.read_text(encoding="utf-8") == "new output"
        assert targets.diagnostics_path.read_text(encoding="utf-8") == (
            "new diagnostics"
        )
        assert targets.result_path.read_text(encoding="utf-8") == "new result"
        assert preserved_backup is not None
        assert preserved_backup.read_text(encoding="utf-8") == (
            "postcommit foreign backup mutation"
        )
    finally:
        bundle.cleanup()
        if preserved_backup is not None and preserved_backup.parent.exists():
            artifacts.remove_artifact(preserved_backup.parent)


def test_final_target_parent_recheck_failure_closes_captured_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.usda"
    target.write_text("captured target", encoding="utf-8")
    parent = artifacts._open_bound_directory(tmp_path)
    real_open = artifacts.os.open
    captured_descriptors: list[int] = []
    recheck_count = 0

    def tracked_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == target.name and dir_fd == parent.descriptor:
            captured_descriptors.append(descriptor)
        return descriptor

    def fail_final_parent_recheck(_parent: Any) -> None:
        nonlocal recheck_count
        recheck_count += 1
        if recheck_count == 2:
            raise RuntimeError("forced final parent recheck failure")

    monkeypatch.setattr(artifacts.os, "open", tracked_open)
    monkeypatch.setattr(
        artifacts,
        "_require_bound_directory_unchanged",
        fail_final_parent_recheck,
    )
    try:
        with pytest.raises(RuntimeError, match="forced final parent recheck failure"):
            artifacts._capture_target_state(parent, target)
    finally:
        os.close(parent.descriptor)

    assert len(captured_descriptors) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(captured_descriptors[0])
    assert closed.value.errno == errno.EBADF


@pytest.mark.parametrize("nested", [False, True])
def test_initial_target_capture_reads_symlink_payload_through_retained_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
) -> None:
    target = tmp_path / "target"
    if nested:
        target.mkdir()
        (target / "link").symlink_to("relative-payload")
    else:
        target.symlink_to("relative-payload")
    parent = artifacts._open_bound_directory(tmp_path)
    real_readlink = artifacts.os.readlink
    observed: list[tuple[Any, int | None]] = []

    def record_readlink(path: Any, *, dir_fd: int | None = None) -> str:
        observed.append((path, dir_fd))
        return real_readlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "readlink", record_readlink)
    state = None
    try:
        state = artifacts._capture_target_state(parent, target)
    finally:
        if state is not None and state.entry_handle is not None:
            state.entry_handle.close()
        os.close(parent.descriptor)

    assert observed
    assert all(path == "" and dir_fd is not None for path, dir_fd in observed)


def test_initial_target_capture_rejects_nested_file_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path, sidecar=True)
    targets.output_path.write_text("captured old output", encoding="utf-8")
    targets.diagnostics_path.write_text("captured old diagnostics", encoding="utf-8")
    targets.result_path.write_text("captured old result", encoding="utf-8")
    assert targets.sidecar_path is not None
    targets.sidecar_path.mkdir()
    member = targets.sidecar_path / "member.bin"
    member.write_bytes(b"captured sidecar member")
    member_identity = (member.stat().st_dev, member.stat().st_ino)
    real_mount_id = artifacts._descriptor_mount_id

    def simulate_nested_file_mount(descriptor: int) -> int:
        mount_id = real_mount_id(descriptor)
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == member_identity:
            return mount_id + 1
        return mount_id

    monkeypatch.setattr(
        artifacts,
        "_descriptor_mount_id",
        simulate_nested_file_mount,
    )

    with pytest.raises(ValueError, match="crossed a mount at member.bin"):
        create_staged_artifact_targets(targets)

    assert member.read_bytes() == b"captured sidecar member"
    assert not list(tmp_path.glob(".*.stage-*"))


@pytest.mark.parametrize("payload_kind", ["diagnostics", "sidecar"])
def test_owner_payload_at_final_requires_exact_recorded_promotion(
    tmp_path: Path,
    payload_kind: str,
) -> None:
    targets = _targets(tmp_path, sidecar=payload_kind == "sidecar")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    label = (
        "diagnostics report" if payload_kind == "diagnostics" else "composition sidecar"
    )
    payload = next(artifact for artifact in promotion if artifact.label == label)
    detached = tmp_path / f"sealed-{payload_kind}"
    if payload_kind == "sidecar":
        detached.mkdir()
        (detached / "asset.txt").write_text("sealed sidecar", encoding="utf-8")
    else:
        detached.write_text("sealed diagnostics", encoding="utf-8")
    promote_staged_artifacts(
        [StagedArtifact(detached, payload.target_path, payload.label)]
    )
    displaced = tmp_path / f"displaced-sealed-{payload_kind}"
    payload.target_path.rename(displaced)
    if payload_kind == "sidecar":
        os.chmod(payload.staged_path, 0o700)
    payload.staged_path.rename(payload.target_path)

    with pytest.raises(RuntimeError, match="without an exact recorded promotion"):
        bundle.cleanup()

    assert payload.target_path.exists()
    assert displaced.exists()
    owner = payload.staged_path.parent
    assert owner.is_dir()
    shutil.rmtree(owner)


def test_direct_promotion_rejects_payload_replacement_after_validation(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    diagnostics = promotion[0]
    displaced = tmp_path / "validated-diagnostics-displaced.json"
    diagnostics.staged_path.rename(displaced)
    diagnostics.staged_path.write_text("foreign diagnostics", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed inode after validation"):
        promote_staged_artifacts(promotion)
    with pytest.raises(RuntimeError, match="replacement and descriptor-owned"):
        bundle.cleanup()

    assert diagnostics.staged_path.read_text(encoding="utf-8") == (
        "foreign diagnostics"
    )
    assert displaced.read_text(encoding="utf-8") == "new diagnostics"
    shutil.rmtree(diagnostics.staged_path.parent)
    displaced.unlink()


def test_direct_promotion_rejects_payload_owner_move_after_validation(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    diagnostics = promotion[0]
    owner = diagnostics.staged_path.parent
    displaced_owner = tmp_path / "validated-diagnostics-owner-displaced"
    owner.rename(displaced_owner)
    owner.mkdir()
    (displaced_owner / diagnostics.staged_path.name).rename(
        owner / diagnostics.staged_path.name
    )

    with pytest.raises(RuntimeError, match="owner changed inode after validation"):
        promote_staged_artifacts(promotion)
    with pytest.raises(RuntimeError, match="Staging owner changed inode"):
        bundle.cleanup()

    assert diagnostics.staged_path.read_text(encoding="utf-8") == "new diagnostics"
    shutil.rmtree(owner)
    displaced_owner.rmdir()


def test_direct_sidecar_rejects_payload_owner_move_after_validation(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path, sidecar=True)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    sidecar = next(
        artifact for artifact in promotion if artifact.label == "composition sidecar"
    )
    owner = sidecar.staged_path.parent
    displaced_owner = tmp_path / "validated-sidecar-owner-displaced"
    owner.rename(displaced_owner)
    owner.mkdir()
    displaced_payload = displaced_owner / sidecar.staged_path.name
    sealed_mode = stat.S_IMODE(displaced_payload.stat().st_mode)
    os.chmod(displaced_payload, 0o700)
    displaced_payload.rename(sidecar.staged_path)
    os.chmod(sidecar.staged_path, sealed_mode)

    with pytest.raises(RuntimeError, match="owner changed inode after validation"):
        promote_staged_artifacts(promotion)
    with pytest.raises(RuntimeError, match="Staging owner changed inode"):
        bundle.cleanup()

    os.chmod(sidecar.staged_path, 0o700)
    shutil.rmtree(owner)
    displaced_owner.rmdir()


def test_direct_promotion_rejects_payload_hardlink_after_validation(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    diagnostics = promotion[0]
    alias = tmp_path / "diagnostics-alias.json"
    os.link(diagnostics.staged_path, alias)

    with pytest.raises(RuntimeError, match="gained additional links after validation"):
        promote_staged_artifacts(promotion)
    with pytest.raises(RuntimeError, match="gained additional links"):
        bundle.cleanup()

    assert diagnostics.staged_path.read_text(encoding="utf-8") == "new diagnostics"
    assert alias.read_text(encoding="utf-8") == "new diagnostics"
    shutil.rmtree(diagnostics.staged_path.parent)
    alias.unlink()


def test_staging_cleanup_invalidates_retained_promotion_authority(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path, sidecar=True))
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    states = [
        state
        for state in (
            *(artifact._promotion_state for artifact in promotion),
            *(
                reservation.payload_promotion_state
                for reservation in bundle._cleanup_reservations
            ),
        )
        if state is not None
    ]
    descriptors = [state.source_descriptor for state in states]

    bundle.cleanup()

    assert states
    for state in states:
        assert state.source_descriptor == -1
        assert state.source_identity is None
        assert state.source_parent_identity is None
        assert state.committed_identity is None
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)

    root_artifact = promotion[-1]
    root_artifact.staged_path.write_text("recreated root", encoding="utf-8")
    with pytest.raises(RuntimeError, match="promotion authority is no longer active"):
        promote_staged_artifacts([root_artifact])


def test_bound_backend_root_unlink_is_safe_when_held_inode_has_no_links(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    staged_output = bundle.staged_targets.output_path
    staged_output.write_text("owned backend output", encoding="utf-8")
    output_reservation = next(
        reservation
        for reservation in bundle._cleanup_reservations
        if reservation.parent.locator_path == staged_output.parent
        and reservation.name == staged_output.name
    )
    artifacts._bind_staging_cleanup_identity(
        bundle,
        staged_output,
        staged_output.lstat(),
    )
    owned_descriptor = output_reservation.owned_descriptor

    staged_output.unlink()
    assert os.fstat(owned_descriptor).st_nlink == 0
    bundle.cleanup()

    assert output_reservation.closed
    with pytest.raises(OSError):
        os.fstat(owned_descriptor)


def test_repeated_staging_validation_reuses_bound_root_descriptor(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    _write_staged_bundle(bundle.staged_targets)
    staged_output = bundle.staged_targets.output_path
    output_reservation = next(
        reservation
        for reservation in bundle._cleanup_reservations
        if reservation.parent.locator_path == staged_output.parent
        and reservation.name == staged_output.name
    )

    staged_promotion_artifacts(bundle)
    owned_descriptor = output_reservation.owned_descriptor
    descriptor_count = len(os.listdir("/proc/self/fd"))
    staged_promotion_artifacts(bundle)

    assert output_reservation.owned_descriptor == owned_descriptor
    assert len(os.listdir("/proc/self/fd")) == descriptor_count
    os.fstat(owned_descriptor)
    bundle.cleanup()
    with pytest.raises(OSError):
        os.fstat(owned_descriptor)


def test_repeated_staging_validation_reseals_nested_sidecar_without_fd_leaks(
    tmp_path: Path,
) -> None:
    descriptor_count_before_bundle = len(os.listdir("/proc/self/fd"))
    bundle = create_staged_artifact_targets(_targets(tmp_path, sidecar=True))
    _write_staged_bundle(bundle.staged_targets)
    sidecar = bundle.staged_targets.sidecar_path
    assert sidecar is not None
    nested = sidecar / "textures" / "variants"
    nested.mkdir(parents=True)
    nested_file = nested / "albedo.bin"
    nested_file.write_bytes(b"nested sidecar bytes")
    sidecar_reservation, owner_payload = artifacts._find_staging_cleanup_reservation(
        bundle,
        sidecar,
    )
    assert owner_payload

    first_promotion = staged_promotion_artifacts(bundle)
    payload_descriptor = sidecar_reservation.payload_descriptor
    source_digest = sidecar_reservation.payload_promotion_state
    assert payload_descriptor >= 0
    assert source_digest is not None
    assert source_digest.source_tree_sha256 is not None
    descriptor_count_after_first_validation = len(os.listdir("/proc/self/fd"))
    first_sidecar = next(
        item for item in first_promotion if item.label == "composition sidecar"
    )
    assert first_sidecar.source_descriptor == payload_descriptor
    assert first_sidecar.source_sha256 == source_digest.source_tree_sha256
    for path in (sidecar, nested.parent, nested, nested_file):
        assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0

    second_promotion = staged_promotion_artifacts(bundle)
    second_sidecar = next(
        item for item in second_promotion if item.label == "composition sidecar"
    )
    assert sidecar_reservation.payload_descriptor == payload_descriptor
    assert second_sidecar.source_descriptor == payload_descriptor
    assert second_sidecar.source_sha256 == first_sidecar.source_sha256
    assert len(os.listdir("/proc/self/fd")) == descriptor_count_after_first_validation

    bundle.cleanup()

    with pytest.raises(OSError):
        os.fstat(payload_descriptor)
    assert len(os.listdir("/proc/self/fd")) == descriptor_count_before_bundle


def test_repeated_staging_validation_rejects_root_replacement_without_rebinding(
    tmp_path: Path,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    _write_staged_bundle(bundle.staged_targets)
    staged_output = bundle.staged_targets.output_path
    displaced_output = tmp_path / "owned-output-displaced.usda"
    output_reservation = next(
        reservation
        for reservation in bundle._cleanup_reservations
        if reservation.parent.locator_path == staged_output.parent
        and reservation.name == staged_output.name
    )
    staged_promotion_artifacts(bundle)
    owned_descriptor = output_reservation.owned_descriptor
    owned_identity = output_reservation.owned_identity

    staged_output.rename(displaced_output)
    staged_output.write_text("foreign replacement", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed inode after cleanup ownership"):
        staged_promotion_artifacts(bundle)

    assert output_reservation.owned_descriptor == owned_descriptor
    opened = os.fstat(owned_descriptor)
    assert owned_identity == (opened.st_dev, opened.st_ino)
    assert displaced_output.read_text(encoding="utf-8") == "new output"
    assert staged_output.read_text(encoding="utf-8") == "foreign replacement"
    with pytest.raises(RuntimeError, match="replacement preserved"):
        bundle.cleanup()
    assert staged_output.read_text(encoding="utf-8") == "foreign replacement"
    assert displaced_output.read_text(encoding="utf-8") == "new output"
    with pytest.raises(OSError):
        os.fstat(owned_descriptor)


def _seal_directory_tree(root: Path) -> None:
    """Remove every write bit after a test tree has been fully authored."""

    entries = list(root.rglob("*"))
    for entry in entries:
        if entry.is_file() and not entry.is_symlink():
            entry.chmod(0o444)
    for entry in sorted(entries, key=lambda path: len(path.parts), reverse=True):
        if entry.is_dir() and not entry.is_symlink():
            entry.chmod(0o555)
    root.chmod(0o555)


def _make_directory_tree_writable(root: Path) -> None:
    """Restore owner access so pytest can remove a sealed test tree."""

    if not root.exists() or root.is_symlink():
        return
    root.chmod(0o700)
    for entry in root.rglob("*"):
        if entry.is_dir() and not entry.is_symlink():
            entry.chmod(0o700)
        elif entry.is_file() and not entry.is_symlink():
            entry.chmod(0o600)


def test_targets_normalize_all_paths_once_without_resolving_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    output_alias = workspace / "output-alias"
    output_alias.symlink_to(real_output, target_is_directory=True)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(workspace)

    targets = JointRiggerArtifactTargets(
        output_path=Path("output-alias") / "nested" / ".." / "rigged.usda",
        diagnostics_path=Path("~/reports/diagnostics.json"),
        result_path=Path("./results/../result.json"),
        sidecar_path=Path("output-alias/./rigged_assets"),
        publication_output_path=Path("./output-alias/rigged.usda"),
        publication_sidecar_path=Path("output-alias/rigged_assets"),
        publication_diagnostics_path=Path("~/reports/./diagnostics.json"),
        publication_result_path=Path("./result.json"),
    )

    assert targets.output_path == output_alias / "rigged.usda"
    assert targets.diagnostics_path == home / "reports" / "diagnostics.json"
    assert targets.result_path == workspace / "result.json"
    assert targets.sidecar_path == output_alias / "rigged_assets"
    assert targets.publication_output_path == targets.output_path
    assert targets.publication_sidecar_path == targets.sidecar_path
    assert targets.publication_diagnostics_path == targets.diagnostics_path
    assert targets.publication_result_path == targets.result_path
    assert all(
        path is None or path.is_absolute()
        for path in (
            targets.output_path,
            targets.diagnostics_path,
            targets.result_path,
            targets.sidecar_path,
            targets.publication_output_path,
            targets.publication_sidecar_path,
            targets.publication_diagnostics_path,
            targets.publication_result_path,
        )
    )
    assert targets.output_path != real_output / "rigged.usda"


def test_normalized_targets_are_stable_across_working_directory_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()
    monkeypatch.chdir(first_cwd)
    targets = JointRiggerArtifactTargets(
        output_path=Path("published/rigged.usda"),
        diagnostics_path=Path("reports/diagnostics.json"),
        result_path=Path("reports/result.json"),
        sidecar_path=Path("published/rigged_assets"),
    )

    monkeypatch.chdir(second_cwd)
    validate_artifact_targets(targets)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    try:
        promote_staged_artifacts(staged_promotion_artifacts(bundle))
    finally:
        bundle.cleanup()

    assert targets.output_path == first_cwd / "published" / "rigged.usda"
    assert targets.publication_output_path == targets.output_path
    assert targets.publication_sidecar_path == targets.sidecar_path
    assert targets.publication_diagnostics_path == targets.diagnostics_path
    assert targets.publication_result_path == targets.result_path
    assert targets.output_path.read_text(encoding="utf-8") == "new output"
    assert targets.diagnostics_path.read_text(encoding="utf-8") == "new diagnostics"
    assert targets.result_path.read_text(encoding="utf-8") == "new result"
    assert targets.sidecar_path is not None
    assert (targets.sidecar_path / "asset.txt").read_text(encoding="utf-8") == (
        "new sidecar"
    )
    assert not (second_cwd / "published").exists()
    assert not (second_cwd / "reports").exists()


def test_relative_target_alias_is_stable_across_working_directory_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()
    source = first_cwd / "source.usda"
    source.write_text("protected source", encoding="utf-8")
    monkeypatch.chdir(first_cwd)
    targets = JointRiggerArtifactTargets(
        output_path=Path("source.usda"),
        diagnostics_path=Path("diagnostics.json"),
        result_path=Path("result.json"),
    )

    monkeypatch.chdir(second_cwd)
    with pytest.raises(ValueError, match="output_path must not alias source_asset"):
        validate_artifact_targets(targets, read_paths=[("source_asset", source)])

    assert source.read_text(encoding="utf-8") == "protected source"
    assert not (second_cwd / "source.usda").exists()


def test_staging_targets_expose_final_publication_layout_and_cleanup_full_bundle(
    tmp_path: Path,
) -> None:
    targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "output" / "rigged.usda",
        diagnostics_path=tmp_path / "reports" / "diagnostics.json",
        result_path=tmp_path / "results" / "result.json",
        sidecar_path=tmp_path / "output" / "rigged_assets",
    )
    bundle = create_staged_artifact_targets(targets)
    staged = bundle.staged_targets

    assert targets.publication_output_path == targets.output_path
    assert targets.publication_sidecar_path == targets.sidecar_path
    assert targets.publication_diagnostics_path == targets.diagnostics_path
    assert targets.publication_result_path == targets.result_path
    assert staged.publication_output_path == targets.output_path
    assert staged.publication_sidecar_path == targets.sidecar_path
    assert staged.publication_diagnostics_path == targets.diagnostics_path
    assert staged.publication_result_path == targets.result_path
    assert staged.output_path != staged.publication_output_path

    for final, temporary, descriptor_owned in zip(
        (
            targets.output_path,
            targets.diagnostics_path,
            targets.result_path,
        ),
        (
            staged.output_path,
            staged.diagnostics_path,
            staged.result_path,
        ),
        (False, True, True),
        strict=True,
    ):
        assert temporary != final
        if descriptor_owned:
            assert temporary.parent.parent == final.parent
            assert temporary.name == final.name
            assert temporary.parent.name.startswith(f".{final.name}.stage-")
        else:
            assert temporary.parent == final.parent
            assert temporary.name.startswith(f".{final.stem}.stage-")
        assert temporary.suffix == final.suffix
        assert not temporary.exists()
    assert targets.sidecar_path is not None
    assert staged.sidecar_path is not None
    assert bundle.sidecar_owner_path is not None
    assert bundle.sidecar_owner_path.parent == targets.sidecar_path.parent
    assert staged.sidecar_path == bundle.sidecar_owner_path / targets.sidecar_path.name
    assert staged.sidecar_path.name == targets.sidecar_path.name
    assert staged.sidecar_path != staged.publication_sidecar_path
    assert staged.sidecar_path.parent != staged.output_path.parent
    assert staged.publication_sidecar_path is not None
    assert staged.publication_output_path is not None
    assert (
        staged.publication_sidecar_path.parent == staged.publication_output_path.parent
    )
    assert not staged.sidecar_path.exists()

    _write_staged_bundle(staged)
    staged_promotion_artifacts(bundle)
    bundle.cleanup()

    assert all(not path.exists() for path in artifacts._target_paths(staged))
    assert not bundle.sidecar_owner_path.exists()
    assert not any(tmp_path.rglob(".*.stage-*"))


def test_staging_cleanup_preserves_recreated_parent_entries(
    tmp_path: Path,
) -> None:
    live_parent = tmp_path / "live"
    displaced_parent = tmp_path / "displaced-live"
    targets = _targets(live_parent, sidecar=True)
    bundle = create_staged_artifact_targets(targets)
    staged = bundle.staged_targets
    assert bundle.sidecar_owner_path is not None
    assert staged.sidecar_path is not None
    reservations = bundle._cleanup_reservations
    reserved_descriptors = [item.parent.descriptor for item in reservations]
    _write_staged_bundle(staged, marker="displaced")
    staged_promotion_artifacts(bundle)
    staged_files = (
        staged.output_path,
        staged.diagnostics_path,
        staged.result_path,
    )
    relative_files = tuple(path.relative_to(live_parent) for path in staged_files)
    relative_sidecar = staged.sidecar_path.relative_to(live_parent)

    live_parent.rename(displaced_parent)
    live_parent.mkdir()
    current_files: list[Path] = []
    for relative_path in relative_files:
        current_path = live_parent / relative_path
        current_path.parent.mkdir(parents=True, exist_ok=True)
        current_path.write_text(
            "current staging",
            encoding="utf-8",
        )
        current_files.append(current_path)

    current_sidecar = live_parent / relative_sidecar
    current_sidecar.mkdir(parents=True)
    (current_sidecar / "asset.txt").write_text(
        "current sidecar",
        encoding="utf-8",
    )

    bundle.cleanup()

    assert not any(displaced_parent.rglob(".*.stage-*"))
    assert any(live_parent.rglob(".*.stage-*"))
    for current_path in current_files:
        assert current_path.read_text(encoding="utf-8") == "current staging"
    assert (current_sidecar / "asset.txt").read_text(encoding="utf-8") == (
        "current sidecar"
    )
    assert all(item.closed for item in reservations)
    for descriptor in reserved_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_staging_cleanup_preserves_errors_and_closes_all_reservations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    _write_staged_bundle(bundle.staged_targets)
    staged_promotion_artifacts(bundle)
    reservations = bundle._cleanup_reservations
    original_remove = artifacts._remove_descriptor_entry
    failed_once = False

    def fail_first_removal(*args: Any, **kwargs: Any) -> None:
        nonlocal failed_once
        original_remove(*args, **kwargs)
        if not failed_once:
            failed_once = True
            raise OSError("forced staging cleanup failure")

    monkeypatch.setattr(artifacts, "_remove_descriptor_entry", fail_first_removal)

    with pytest.raises(OSError, match="forced staging cleanup failure"):
        bundle.cleanup()

    assert failed_once
    assert all(item.closed for item in reservations)
    assert not any(tmp_path.glob(".*.stage-*"))


def test_staging_cleanup_relinquishes_close_then_eio_reused_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    _write_staged_bundle(bundle.staged_targets)
    staged_promotion_artifacts(bundle)
    reservation = bundle._cleanup_reservations[0]
    owned_descriptor = reservation.parent.descriptor
    decoy = tmp_path / "decoy.txt"
    decoy.write_text("do not close reused fd", encoding="utf-8")
    real_close = artifacts.os.close
    reused_descriptor: int | None = None
    close_attempts = 0

    def close_then_reuse_and_fail(descriptor: int) -> None:
        nonlocal reused_descriptor, close_attempts
        if descriptor != owned_descriptor or reused_descriptor is not None:
            real_close(descriptor)
            return
        close_attempts += 1
        real_close(descriptor)
        reused_descriptor = os.open(decoy, os.O_RDONLY)
        assert reused_descriptor == owned_descriptor
        raise OSError(errno.EIO, "forced close-then-EIO")

    monkeypatch.setattr(artifacts.os, "close", close_then_reuse_and_fail)
    try:
        with pytest.raises(OSError, match="forced close-then-EIO"):
            bundle.cleanup()

        assert reservation.closed
        assert close_attempts == 1
        assert reused_descriptor is not None
        assert os.read(reused_descriptor, 64) == b"do not close reused fd"
        assert not any(tmp_path.glob(".*.stage-*"))
    finally:
        if reused_descriptor is not None:
            real_close(reused_descriptor)


def test_staging_cleanup_baseexception_attempts_all_reservations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    _write_staged_bundle(bundle.staged_targets)
    staged_promotion_artifacts(bundle)
    reservations = bundle._cleanup_reservations
    descriptors = [reservation.parent.descriptor for reservation in reservations]
    owned_descriptors = {reservation.owned_descriptor for reservation in reservations}
    original_remove = artifacts._remove_descriptor_entry
    fatal_error = SystemExit("forced staging cleanup fatal")
    removal_attempts: list[str] = []

    def fail_first_removal(*args: Any, **kwargs: Any) -> None:
        original_remove(*args, **kwargs)
        if kwargs.get("source_descriptor") not in owned_descriptors:
            return
        removal_attempts.append(args[1])
        if len(removal_attempts) == 1:
            raise fatal_error

    monkeypatch.setattr(
        artifacts,
        "_remove_descriptor_entry",
        fail_first_removal,
    )

    with pytest.raises(SystemExit) as raised:
        bundle.cleanup()

    assert raised.value is fatal_error
    assert len(removal_attempts) >= len(reservations)
    assert all(reservation.closed for reservation in reservations)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert not any(tmp_path.glob(".*.stage-*"))


def test_staging_cleanup_fatal_close_is_single_shot_and_does_not_stop_later_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path))
    _write_staged_bundle(bundle.staged_targets)
    staged_promotion_artifacts(bundle)
    reservations = bundle._cleanup_reservations
    descriptors = [reservation.parent.descriptor for reservation in reservations]
    failed_descriptor = descriptors[0]
    fatal_error = KeyboardInterrupt("forced staging descriptor close fatal")
    real_close = artifacts.os.close
    failed_close_attempts = 0

    def close_then_interrupt(descriptor: int) -> None:
        nonlocal failed_close_attempts
        real_close(descriptor)
        if descriptor == failed_descriptor and failed_close_attempts == 0:
            failed_close_attempts += 1
            raise fatal_error

    monkeypatch.setattr(artifacts.os, "close", close_then_interrupt)

    with pytest.raises(KeyboardInterrupt) as raised:
        bundle.cleanup()

    assert raised.value is fatal_error
    assert failed_close_attempts == 1
    assert all(reservation.closed for reservation in reservations)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert not any(tmp_path.glob(".*.stage-*"))


def test_staging_construction_fatal_cleans_prior_reservations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_reserve = artifacts._reserve_backend_staging_name
    fatal_error = SystemExit("forced later staging reservation fatal")
    retained_reservations: list[Any] = []
    calls = 0

    def fail_second_reservation(target: Path, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise fatal_error
        result = original_reserve(target, **kwargs)
        retained_reservations.append(result[1])
        return result

    monkeypatch.setattr(
        artifacts,
        "_reserve_backend_staging_name",
        fail_second_reservation,
    )

    with pytest.raises(SystemExit) as raised:
        create_staged_artifact_targets(_targets(tmp_path))

    assert raised.value is fatal_error
    assert len(retained_reservations) == 1
    reservation = retained_reservations[0]
    assert reservation.closed
    with pytest.raises(OSError):
        os.fstat(reservation.parent.descriptor)
    assert not any(tmp_path.glob(".*.stage-*"))


def test_target_preflight_rejects_nested_paths_before_cleanup(tmp_path: Path) -> None:
    output = tmp_path / "rigged.usda"
    output.write_text("old output", encoding="utf-8")
    sidecar = tmp_path / "rigged_assets"
    targets = JointRiggerArtifactTargets(
        output_path=output,
        diagnostics_path=sidecar / "diagnostics.json",
        result_path=tmp_path / "result.json",
        sidecar_path=sidecar,
    )

    with pytest.raises(ValueError, match="Nested Joint Rigger artifact targets"):
        validate_artifact_targets(targets)

    assert output.read_text(encoding="utf-8") == "old output"


def test_target_preflight_rejects_read_input_inside_sidecar(tmp_path: Path) -> None:
    targets = _targets(tmp_path, sidecar=True)
    assert targets.sidecar_path is not None
    source = targets.sidecar_path / "source.usda"

    with pytest.raises(ValueError, match="must not be inside sidecar_path"):
        validate_artifact_targets(targets, read_paths=[("source_asset", source)])


def test_target_preflight_refuses_to_replace_report_directory(tmp_path: Path) -> None:
    targets = _targets(tmp_path)
    targets.diagnostics_path.mkdir()
    marker = targets.diagnostics_path / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(IsADirectoryError, match="diagnostics_path directory"):
        validate_artifact_targets(targets)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_target_preflight_refuses_existing_sidecar_file(tmp_path: Path) -> None:
    targets = _targets(tmp_path, sidecar=True)
    assert targets.sidecar_path is not None
    targets.sidecar_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(
        ValueError, match="sidecar_path must be a non-symlink directory"
    ):
        validate_artifact_targets(targets)

    assert targets.sidecar_path.read_text(encoding="utf-8") == "not a directory"


def test_sidecar_digest_is_order_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "nested").mkdir(parents=True)
    (first / "z.usda").write_text("z", encoding="utf-8")
    (first / "nested" / "a.bin").write_bytes(b"a\x00b")
    (second / "nested").mkdir(parents=True)
    (second / "nested" / "a.bin").write_bytes(b"a\x00b")
    (second / "z.usda").write_text("z", encoding="utf-8")

    first_digest = sidecar_dependency_bundle_sha256(first)
    assert first_digest == sidecar_dependency_bundle_sha256(second)

    (second / "nested" / "a.bin").write_bytes(b"mutated")
    assert sidecar_dependency_bundle_sha256(second) != first_digest


def test_sidecar_digest_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        sidecar_dependency_bundle_sha256(tmp_path / "missing")

    regular_file = tmp_path / "not-a-directory"
    regular_file.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="non-symlink directory"):
        sidecar_dependency_bundle_sha256(regular_file)


@pytest.mark.parametrize(
    "entry_kind",
    ["symlink", "symlink_directory", "hardlink", "fifo"],
)
def test_sidecar_digest_rejects_links_and_special_files(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    entry = sidecar / "invalid"
    if entry_kind == "symlink":
        target = tmp_path / "outside.usda"
        target.write_text("outside", encoding="utf-8")
        entry.symlink_to(target)
        expected = "symlink"
    elif entry_kind == "symlink_directory":
        target = tmp_path / "outside"
        target.mkdir()
        (target / "must-not-be-traversed.usda").write_text(
            "outside",
            encoding="utf-8",
        )
        entry.symlink_to(target, target_is_directory=True)
        expected = "symlink"
    elif entry_kind == "hardlink":
        target = tmp_path / "outside.usda"
        target.write_text("outside", encoding="utf-8")
        os.link(target, entry)
        expected = "exactly one hard link"
    else:
        os.mkfifo(entry)
        expected = "special file"

    with pytest.raises(ValueError, match=expected):
        sidecar_dependency_bundle_sha256(sidecar)


def test_sidecar_digest_fifo_swap_between_stat_and_open_fails_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    member = sidecar / "asset.bin"
    member.write_bytes(b"trusted bytes")
    real_open = artifacts.os.open
    swapped = False

    def swap_member_to_fifo(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and (
            Path(path) == member
            or (os.fspath(path) == member.name and dir_fd is not None)
        ):
            swapped = True
            assert flags & getattr(os, "O_NONBLOCK", 0)
            if dir_fd is None:
                member.unlink()
                os.mkfifo(member)
            else:
                os.unlink(member.name, dir_fd=dir_fd)
                os.mkfifo(member.name, dir_fd=dir_fd)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", swap_member_to_fifo)

    with pytest.raises(
        ValueError,
        match=r"regular file changed|changed before hashing|special file",
    ):
        sidecar_dependency_bundle_sha256(sidecar)

    assert swapped
    assert stat.S_ISFIFO(member.lstat().st_mode)


def test_staging_creation_failure_leaves_no_reserved_sibling_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_reserve = artifacts._reserve_backend_staging_name
    calls = 0

    def fail_second_reservation(
        target: Path,
        **kwargs: Any,
    ) -> tuple[Path, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced staging creation failure")
        return real_reserve(target, **kwargs)

    monkeypatch.setattr(
        artifacts,
        "_reserve_backend_staging_name",
        fail_second_reservation,
    )

    with pytest.raises(OSError, match="forced staging creation failure"):
        create_staged_artifact_targets(_targets(tmp_path))

    assert not any(tmp_path.glob(".*.stage-*"))


def test_staging_requires_sidecar_to_share_output_parent(tmp_path: Path) -> None:
    targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "output" / "rigged.usda",
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
        sidecar_path=tmp_path / "dependencies" / "rigged_assets",
    )

    for operation in (validate_artifact_targets, create_staged_artifact_targets):
        with pytest.raises(ValueError, match="share output_path's parent"):
            operation(targets)


def test_caller_publication_layout_cannot_override_final_targets(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires a physical sidecar_path"):
        JointRiggerArtifactTargets(
            output_path=tmp_path / "rigged.usda",
            diagnostics_path=tmp_path / "diagnostics.json",
            result_path=tmp_path / "result.json",
            publication_sidecar_path=tmp_path / "published" / "rigged_assets",
        )

    mismatches = (
        (
            "publication_output_path",
            {"publication_output_path": tmp_path / "published" / "rigged.usda"},
        ),
        (
            "publication_sidecar_path",
            {"publication_sidecar_path": tmp_path / "published" / "rigged_assets"},
        ),
        (
            "publication_diagnostics_path",
            {
                "publication_diagnostics_path": tmp_path
                / "published"
                / "diagnostics.json"
            },
        ),
        (
            "publication_result_path",
            {"publication_result_path": tmp_path / "published" / "result.json"},
        ),
    )
    for field, override in mismatches:
        targets = JointRiggerArtifactTargets(
            output_path=tmp_path / "rigged.usda",
            diagnostics_path=tmp_path / "diagnostics.json",
            result_path=tmp_path / "result.json",
            sidecar_path=tmp_path / "rigged_assets",
            **override,
        )
        for operation in (validate_artifact_targets, create_staged_artifact_targets):
            with pytest.raises(ValueError, match=f"Caller-facing {field}"):
                operation(targets)

    assert not any(tmp_path.glob(".*.stage-*"))


def test_staging_failure_removes_temporary_sidecar_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_reserve = artifacts._reserve_backend_staging_name
    calls = 0

    def fail_after_owner_created(
        target: Path,
        **kwargs: Any,
    ) -> tuple[Path, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced report staging failure")
        return real_reserve(target, **kwargs)

    monkeypatch.setattr(
        artifacts,
        "_reserve_backend_staging_name",
        fail_after_owner_created,
    )

    with pytest.raises(OSError, match="forced report staging failure"):
        create_staged_artifact_targets(_targets(tmp_path, sidecar=True))

    assert not any(tmp_path.glob(".*.stage-*"))


def test_invalidation_removes_commit_root_and_all_evidence(tmp_path: Path) -> None:
    targets = _targets(tmp_path, sidecar=True)
    targets.output_path.write_text("old output", encoding="utf-8")
    targets.diagnostics_path.write_text("old diagnostics", encoding="utf-8")
    targets.result_path.write_text("old result", encoding="utf-8")
    assert targets.sidecar_path is not None
    targets.sidecar_path.mkdir()
    (targets.sidecar_path / "asset.txt").write_text("old", encoding="utf-8")

    invalidate_artifact_targets(targets)

    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()
    assert not targets.sidecar_path.exists()

    targets_without_sidecar = _targets(tmp_path)
    targets_without_sidecar.output_path.write_text("old output", encoding="utf-8")
    targets_without_sidecar.diagnostics_path.write_text(
        "old diagnostics",
        encoding="utf-8",
    )
    targets_without_sidecar.result_path.write_text("old result", encoding="utf-8")

    invalidate_artifact_targets(targets_without_sidecar)

    assert not targets_without_sidecar.output_path.exists()
    assert not targets_without_sidecar.diagnostics_path.exists()
    assert not targets_without_sidecar.result_path.exists()


def test_remove_artifact_unlinks_symlink_without_traversing_referent(
    tmp_path: Path,
) -> None:
    referent = tmp_path / "referent"
    referent.mkdir()
    member = referent / "member.txt"
    member.write_text("preserve", encoding="utf-8")
    artifact = tmp_path / "artifact"
    artifact.symlink_to(referent, target_is_directory=True)

    artifacts.remove_artifact(artifact)
    artifacts.remove_artifact(tmp_path / "missing-parent" / "missing-artifact")
    with pytest.raises(ValueError, match="must name a directory entry"):
        artifacts.remove_artifact(Path("/"))

    assert not artifact.exists()
    assert not artifact.is_symlink()
    assert member.read_text(encoding="utf-8") == "preserve"


def test_remove_artifact_preserves_replacement_swapped_before_nofollow_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.json"
    displaced = tmp_path / "displaced.json"
    artifact.write_text("original", encoding="utf-8")
    real_open = artifacts.os.open
    swapped = False
    observed_flags = 0

    def swap_before_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal observed_flags, swapped
        if not swapped and path == artifact.name and dir_fd is not None:
            swapped = True
            observed_flags = flags
            artifact.rename(displaced)
            artifact.write_text("replacement", encoding="utf-8")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", swap_before_open)

    with pytest.raises(RuntimeError, match="changed inode while opening"):
        artifacts.remove_artifact(artifact)

    assert swapped
    assert observed_flags & os.O_NOFOLLOW
    assert artifact.read_text(encoding="utf-8") == "replacement"
    assert displaced.read_text(encoding="utf-8") == "original"


def test_remove_artifact_rejects_root_mount_boundary_before_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact-tree"
    artifact.mkdir()
    member = artifact / "member.txt"
    member.write_text("preserve", encoding="utf-8")
    artifact_metadata = artifact.stat()
    artifact_identity = (artifact_metadata.st_dev, artifact_metadata.st_ino)
    real_mount_id = artifacts._descriptor_mount_id

    def simulate_artifact_mount(descriptor: int) -> int:
        mount_id = real_mount_id(descriptor)
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == artifact_identity:
            return mount_id + 1
        return mount_id

    monkeypatch.setattr(artifacts, "_descriptor_mount_id", simulate_artifact_mount)

    with pytest.raises(ValueError, match="is or crossed a mount point"):
        artifacts.remove_artifact(artifact)

    assert member.read_text(encoding="utf-8") == "preserve"
    assert not any(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_complete_bundle_promotes_reports_sidecar_then_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path, sidecar=True)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    expected_order = [
        targets.diagnostics_path,
        targets.result_path,
        targets.sidecar_path,
        targets.output_path,
    ]
    observed: list[Path] = []
    original_replace = artifacts._replace_entry

    def record_replace(source: Any, target: Any) -> None:
        if target.path in expected_order:
            observed.append(target.path)
        original_replace(source, target)

    monkeypatch.setattr(artifacts, "_replace_entry", record_replace)
    try:
        promote_staged_artifacts(promotion)
    finally:
        bundle.cleanup()

    assert observed == expected_order
    assert targets.output_path.read_text(encoding="utf-8") == "new output"
    assert targets.diagnostics_path.read_text(encoding="utf-8") == "new diagnostics"
    assert targets.result_path.read_text(encoding="utf-8") == "new result"
    assert targets.sidecar_path is not None
    assert (targets.sidecar_path / "asset.txt").read_text(encoding="utf-8") == (
        "new sidecar"
    )


def test_sealed_sidecar_backup_and_rollback_restore_exact_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path, sidecar=True)
    _write_staged_bundle(targets, marker="old")
    assert targets.sidecar_path is not None
    _seal_directory_tree(targets.sidecar_path)
    original_mode = stat.S_IMODE(targets.sidecar_path.stat().st_mode)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    original_replace = artifacts._replace_entry
    observed_temporary_write: list[str] = []

    def observe_guarded_rename(source: Any, target: Any) -> None:
        if source.path == targets.sidecar_path:
            assert stat.S_IMODE(source.path.stat().st_mode) & stat.S_IWUSR
            observed_temporary_write.append("backup")
        elif (
            target.path == targets.sidecar_path
            and source.name == "artifact"
            and source.parent.locator_path.name.startswith(".joint-rigger.rollback-")
        ):
            assert stat.S_IMODE(source.path.stat().st_mode) & stat.S_IWUSR
            observed_temporary_write.append("restore")
        original_replace(source, target)

    monkeypatch.setattr(artifacts, "_replace_entry", observe_guarded_rename)
    try:
        with pytest.raises(RuntimeError, match="forced precommit failure"):
            promote_staged_artifacts(
                promotion,
                precommit_validator=lambda: (_ for _ in ()).throw(
                    RuntimeError("forced precommit failure")
                ),
            )
    finally:
        bundle.cleanup()

    assert observed_temporary_write == ["backup", "restore"]
    assert stat.S_IMODE(targets.sidecar_path.stat().st_mode) == original_mode
    assert (targets.sidecar_path / "asset.txt").read_text(encoding="utf-8") == (
        "old sidecar"
    )
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))
    _make_directory_tree_writable(targets.sidecar_path)


def test_sealed_sidecar_backup_failure_restores_exact_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path, sidecar=True)
    _write_staged_bundle(targets, marker="old")
    assert targets.sidecar_path is not None
    _seal_directory_tree(targets.sidecar_path)
    original_mode = stat.S_IMODE(targets.sidecar_path.stat().st_mode)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    original_replace = artifacts._replace_entry
    failed_with_temporary_write = False

    def fail_guarded_sidecar_backup(source: Any, target: Any) -> None:
        nonlocal failed_with_temporary_write
        if source.path == targets.sidecar_path:
            assert stat.S_IMODE(source.path.stat().st_mode) & stat.S_IWUSR
            failed_with_temporary_write = True
            raise OSError(errno.EIO, "forced sealed sidecar backup failure")
        original_replace(source, target)

    monkeypatch.setattr(artifacts, "_replace_entry", fail_guarded_sidecar_backup)
    try:
        with pytest.raises(OSError, match="forced sealed sidecar backup failure"):
            promote_staged_artifacts(promotion)
    finally:
        bundle.cleanup()

    assert failed_with_temporary_write
    assert stat.S_IMODE(targets.sidecar_path.stat().st_mode) == original_mode
    assert (targets.sidecar_path / "asset.txt").read_text(encoding="utf-8") == (
        "old sidecar"
    )
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))
    _make_directory_tree_writable(targets.sidecar_path)


def test_descriptor_source_contract_requires_paired_valid_fields(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"

    with pytest.raises(ValueError, match="must be provided together"):
        StagedArtifact(staged, target, "root", source_descriptor=1)
    with pytest.raises(ValueError, match="must be provided together"):
        StagedArtifact(staged, target, "root", source_sha256="0" * 64)
    with pytest.raises(ValueError, match="non-negative file descriptor"):
        StagedArtifact(
            staged,
            target,
            "root",
            source_descriptor=-1,
            source_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        StagedArtifact(
            staged,
            target,
            "root",
            source_descriptor=1,
            source_sha256="not-a-digest",
        )


def test_descriptor_source_requires_readonly_unwritable_singly_linked_file(
    tmp_path: Path,
) -> None:
    expected_bytes = b"trusted descriptor bytes"
    expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()

    writable_source = tmp_path / "writable-fd.usda"
    writable_source.write_bytes(expected_bytes)
    writable_descriptor = os.open(writable_source, os.O_RDWR)
    writable_source.chmod(0o400)
    try:
        with pytest.raises(ValueError, match="opened read-only"):
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        writable_source,
                        tmp_path / "writable-fd-target.usda",
                        "generated root",
                        source_descriptor=writable_descriptor,
                        source_sha256=expected_sha256,
                    )
                ]
            )
    finally:
        os.close(writable_descriptor)

    writable_mode_source = tmp_path / "writable-mode.usda"
    writable_mode_source.write_bytes(expected_bytes)
    with writable_mode_source.open("rb") as source:
        with pytest.raises(RuntimeError, match="no write permissions"):
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        writable_mode_source,
                        tmp_path / "writable-mode-target.usda",
                        "generated root",
                        source_descriptor=source.fileno(),
                        source_sha256=expected_sha256,
                    )
                ]
            )

    linked_source = tmp_path / "linked.usda"
    linked_alias = tmp_path / "linked-alias.usda"
    linked_source.write_bytes(expected_bytes)
    linked_source.chmod(0o400)
    os.link(linked_source, linked_alias)
    with linked_source.open("rb") as source:
        with pytest.raises(RuntimeError, match="exactly 1 link"):
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        linked_source,
                        tmp_path / "linked-target.usda",
                        "generated root",
                        source_descriptor=source.fileno(),
                        source_sha256=expected_sha256,
                    )
                ]
            )


def test_descriptor_backed_source_publishes_detached_copy_and_unlinks_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    staged.write_bytes(trusted_bytes)
    writable_descriptor = os.open(staged, os.O_WRONLY)
    staged.chmod(0o400)
    target.write_bytes(b"old target")

    def reject_hard_link(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        pytest.fail("descriptor publication must not use hard links")

    original_write_all = artifacts._write_all
    writes_observed = 0

    def assert_final_absent_while_copying(
        descriptor: int,
        content: bytes,
        *,
        label: str,
    ) -> None:
        nonlocal writes_observed
        writes_observed += 1
        assert not target.exists()
        assert any(tmp_path.glob(".joint-rigger-copy-*"))
        original_write_all(descriptor, content, label=label)

    monkeypatch.setattr(artifacts.os, "link", reject_hard_link)
    monkeypatch.setattr(
        artifacts,
        "_write_all",
        assert_final_absent_while_copying,
    )
    try:
        with staged.open("rb") as source:
            source_metadata = os.fstat(source.fileno())
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        staged,
                        target,
                        "generated root",
                        source_descriptor=source.fileno(),
                        source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
                    )
                ]
            )
            assert os.fstat(source.fileno()).st_nlink == 0
            target_metadata = target.stat()
            os.ftruncate(writable_descriptor, 0)
            os.pwrite(writable_descriptor, b"mutated caller snapshot", 0)
    finally:
        os.close(writable_descriptor)

    assert target.read_bytes() == trusted_bytes
    assert (target_metadata.st_dev, target_metadata.st_ino) != (
        source_metadata.st_dev,
        source_metadata.st_ino,
    )
    assert source_metadata.st_nlink == target_metadata.st_nlink == 1
    assert stat.S_IMODE(target_metadata.st_mode) == 0o444
    assert writes_observed > 0
    assert not staged.exists()
    assert not any(tmp_path.glob(".joint-rigger-copy-*"))
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_descriptor_promotion_refuses_foreign_destination_created_at_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    foreign_bytes = b"foreign destination"
    staged.write_bytes(trusted_bytes)
    staged.chmod(0o400)
    original_replace = artifacts._replace_entry
    injected = False

    def inject_foreign_destination(source: Any, destination: Any) -> None:
        nonlocal injected
        if not injected and source.name.startswith(".joint-rigger-copy-"):
            injected = True
            target.write_bytes(foreign_bytes)
        original_replace(source, destination)

    monkeypatch.setattr(artifacts, "_replace_entry", inject_foreign_destination)
    with staged.open("rb") as source:
        with pytest.raises(FileExistsError):
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        staged,
                        target,
                        "generated root",
                        source_descriptor=source.fileno(),
                        source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
                    )
                ]
            )

    assert injected
    assert target.read_bytes() == foreign_bytes
    assert staged.read_bytes() == trusted_bytes
    assert not any(tmp_path.glob(".joint-rigger-copy-*"))
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_rollback_preserves_swapped_promoted_target_and_quarantines_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_evidence = tmp_path / "staged-evidence.json"
    staged_root = tmp_path / "staged-root.usda"
    target_evidence = tmp_path / "evidence.json"
    target_root = tmp_path / "root.usda"
    detached_new_evidence = tmp_path / "detached-new-evidence.json"
    staged_evidence.write_bytes(b"new evidence")
    staged_root.write_bytes(b"new root")
    target_evidence.write_bytes(b"old evidence")
    target_root.write_bytes(b"old root")
    original_replace = artifacts._replace_entry
    swapped = False

    def swap_promoted_evidence(source: Any, destination: Any) -> None:
        nonlocal swapped
        original_replace(source, destination)
        if destination.path == target_evidence and not swapped:
            swapped = True
            target_evidence.rename(detached_new_evidence)
            target_evidence.write_bytes(b"foreign evidence")

    monkeypatch.setattr(artifacts, "_replace_entry", swap_promoted_evidence)
    with pytest.raises(
        RuntimeError,
        match="Artifact promotion changed inode after rename",
    ) as raised:
        promote_staged_artifacts(
            [
                StagedArtifact(staged_evidence, target_evidence, "evidence"),
                StagedArtifact(staged_root, target_root, "generated root"),
            ]
        )

    assert swapped
    notes = "\n".join(raised.value.__notes__)
    assert "Artifact rollback was incomplete" in notes
    assert "Artifact backup restore also failed for evidence" in notes
    assert target_evidence.read_bytes() == b"foreign evidence"
    assert detached_new_evidence.read_bytes() == b"new evidence"
    assert target_root.read_bytes() == b"old root"
    backup_files = [
        path
        for directory in tmp_path.glob(".joint-rigger.rollback-*")
        for path in directory.iterdir()
        if path.is_file()
    ]
    assert any(path.read_bytes() == b"old evidence" for path in backup_files)


def test_descriptor_backed_source_path_replacement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    replacement_bytes = b"pathname replacement bytes"
    staged.write_bytes(trusted_bytes)
    staged.chmod(0o400)
    target.write_bytes(b"old target")
    original_replace = artifacts._replace_entry
    path_replaced = False

    def replace_path_after_backup(source: Any, destination: Any) -> None:
        nonlocal path_replaced
        original_replace(source, destination)
        if path_replaced:
            return
        path_replaced = True
        staged.unlink()
        staged.write_bytes(replacement_bytes)

    monkeypatch.setattr(artifacts, "_replace_entry", replace_path_after_backup)
    with staged.open("rb") as source:
        promotion = [
            StagedArtifact(
                staged,
                target,
                "generated root",
                source_descriptor=source.fileno(),
                source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
            )
        ]
        with pytest.raises(RuntimeError, match="entry changed inode"):
            promote_staged_artifacts(promotion)

    assert path_replaced
    assert target.read_bytes() == b"old target"
    assert staged.read_bytes() == replacement_bytes
    assert target.read_bytes() != replacement_bytes
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_descriptor_content_mismatch_after_backup_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    staged.write_bytes(trusted_bytes)
    writable_descriptor = os.open(staged, os.O_WRONLY)
    staged.chmod(0o400)
    target.write_bytes(b"old target")
    original_replace = artifacts._replace_entry
    source_modified = False

    def modify_source_after_backup(source: Any, destination: Any) -> None:
        nonlocal source_modified
        original_replace(source, destination)
        if source_modified:
            return
        source_modified = True
        os.ftruncate(writable_descriptor, 0)
        os.pwrite(writable_descriptor, b"modified descriptor bytes", 0)

    monkeypatch.setattr(artifacts, "_replace_entry", modify_source_after_backup)
    try:
        with staged.open("rb") as source:
            promotion = [
                StagedArtifact(
                    staged,
                    target,
                    "generated root",
                    source_descriptor=source.fileno(),
                    source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
                )
            ]
            with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
                promote_staged_artifacts(promotion)
    finally:
        os.close(writable_descriptor)

    assert source_modified
    assert target.read_bytes() == b"old target"
    assert staged.read_bytes() == b"modified descriptor bytes"
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_descriptor_source_unlink_failure_reports_committed_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    staged.write_bytes(trusted_bytes)
    staged.chmod(0o400)
    target.write_bytes(b"old target")

    def fail_source_unlink(bound_artifact: Any) -> None:
        del bound_artifact
        raise OSError("forced descriptor source unlink failure")

    monkeypatch.setattr(
        artifacts,
        "_unlink_descriptor_source_name",
        fail_source_unlink,
    )
    with staged.open("rb") as source:
        promotion = [
            StagedArtifact(
                staged,
                target,
                "generated root",
                source_descriptor=source.fileno(),
                source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
            )
        ]
        with pytest.raises(CommittedArtifactPublicationCleanupError) as raised:
            promote_staged_artifacts(promotion)

    assert any(
        "forced descriptor source unlink failure" in str(error)
        for error in raised.value.cleanup_errors
    )
    assert target.read_bytes() == trusted_bytes
    assert staged.read_bytes() == trusted_bytes
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_descriptor_source_unlink_has_no_postcommit_target_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    staged.write_bytes(trusted_bytes)
    staged.chmod(0o400)
    target.write_bytes(b"old target")
    original_require_target = artifacts._require_descriptor_target

    validation_calls = 0

    def reject_postcommit_target_revalidation(
        bound_artifact: Any,
        detached_target: Any,
    ) -> None:
        nonlocal validation_calls
        validation_calls += 1
        original_require_target(bound_artifact, detached_target)
        if not staged.exists():
            raise AssertionError("target was revalidated after root commit")

    monkeypatch.setattr(
        artifacts,
        "_require_descriptor_target",
        reject_postcommit_target_revalidation,
    )
    with staged.open("rb") as source:
        promotion = [
            StagedArtifact(
                staged,
                target,
                "generated root",
                source_descriptor=source.fileno(),
                source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
            )
        ]
        promote_staged_artifacts(promotion)

    assert validation_calls == 0
    assert target.read_bytes() == trusted_bytes
    assert not staged.exists()
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


@pytest.mark.parametrize("fail_source_close", [False, True])
def test_descriptor_restore_closes_source_after_target_close_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_source_close: bool,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    target.write_bytes(trusted_bytes)
    target.chmod(0o444)
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    restore_source_descriptor: int | None = None
    restore_target_descriptor: int | None = None
    restore_source_closed = False

    def track_restore_opens(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal restore_source_descriptor, restore_target_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == target.name and flags & os.O_ACCMODE == os.O_RDONLY:
            assert flags & getattr(os, "O_NONBLOCK", 0)
            restore_source_descriptor = descriptor
        elif path == staged.name and flags & os.O_CREAT:
            restore_target_descriptor = descriptor
        return descriptor

    def close_restore_descriptors(descriptor: int) -> None:
        nonlocal restore_source_closed
        if descriptor == restore_target_descriptor:
            real_close(descriptor)
            raise OSError(errno.EIO, "forced restore target close failure")
        if descriptor == restore_source_descriptor:
            restore_source_closed = True
            real_close(descriptor)
            if fail_source_close:
                raise OSError(errno.EIO, "forced restore source close failure")
            return
        real_close(descriptor)

    monkeypatch.setattr(artifacts.os, "open", track_restore_opens)
    monkeypatch.setattr(artifacts.os, "close", close_restore_descriptors)
    parent_descriptor = real_open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    source_contract_descriptor = real_open(target, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        parent_metadata = os.fstat(parent_descriptor)
        directory = artifacts._BoundDirectory(
            locator_path=tmp_path,
            opened_path=tmp_path,
            descriptor=parent_descriptor,
            identity=(parent_metadata.st_dev, parent_metadata.st_ino),
        )
        target_metadata = target.stat()
        artifact = StagedArtifact(
            staged,
            target,
            "generated root",
            source_descriptor=source_contract_descriptor,
            source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
        )
        bound_artifact = artifacts._BoundArtifact(
            artifact=artifact,
            staged_entry=artifacts._BoundEntry(directory, staged.name),
            target_entry=artifacts._BoundEntry(directory, target.name),
            descriptor_source=artifacts._BoundDescriptorSource(
                descriptor=source_contract_descriptor,
                identity=(target_metadata.st_dev, target_metadata.st_ino),
                sha256=hashlib.sha256(trusted_bytes).hexdigest(),
                mode=0o400,
                is_directory=False,
            ),
        )
        detached_target = artifacts._DetachedTarget(
            identity=(target_metadata.st_dev, target_metadata.st_ino),
            sha256=hashlib.sha256(trusted_bytes).hexdigest(),
            mode=stat.S_IMODE(target_metadata.st_mode),
            is_directory=False,
        )
        if fail_source_close:
            with pytest.raises(
                BaseExceptionGroup,
                match="Descriptor-source restoration cleanup failed",
            ) as raised:
                artifacts._restore_descriptor_source_name(
                    bound_artifact,
                    detached_target,
                )
            assert len(raised.value.exceptions) == 2
        else:
            with pytest.raises(OSError, match="forced restore target close failure"):
                artifacts._restore_descriptor_source_name(
                    bound_artifact,
                    detached_target,
                )
    finally:
        real_close(source_contract_descriptor)
        real_close(parent_descriptor)

    assert restore_target_descriptor is not None
    assert restore_source_descriptor is not None
    assert restore_source_closed
    assert staged.read_bytes() == trusted_bytes


def test_directory_descriptor_source_publishes_distinct_exact_tree_copy(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged-sidecar"
    nested = staged / "nested"
    empty = nested / "empty"
    empty.mkdir(parents=True)
    source_member = nested / "asset.bin"
    source_member.write_bytes(b"trusted sidecar bytes")
    _seal_directory_tree(staged)
    target = tmp_path / "published-sidecar"
    target.mkdir()
    (target / "old.txt").write_text("old sidecar", encoding="utf-8")

    descriptor = os.open(
        staged,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        expected_sha256 = artifacts.directory_descriptor_tree_sha256(descriptor)
        source_root_identity = (staged.stat().st_dev, staged.stat().st_ino)
        source_member_identity = (
            source_member.stat().st_dev,
            source_member.stat().st_ino,
        )
        promote_staged_artifacts(
            [
                StagedArtifact(
                    staged,
                    target,
                    "composition sidecar",
                    source_descriptor=descriptor,
                    source_sha256=expected_sha256,
                )
            ]
        )

        target_member = target / "nested" / "asset.bin"
        assert target_member.read_bytes() == b"trusted sidecar bytes"
        assert (target / "nested" / "empty").is_dir()
        assert artifacts.directory_tree_sha256(target) == expected_sha256
        assert (target.stat().st_dev, target.stat().st_ino) != source_root_identity
        assert (target_member.stat().st_dev, target_member.stat().st_ino) != (
            source_member_identity
        )
        assert stat.S_IMODE(target.stat().st_mode) == 0o555
        assert stat.S_IMODE(target_member.stat().st_mode) == 0o444

        # The caller owns the private source tree after success. Even a writer
        # that changes it cannot affect the promoter-owned detached target.
        source_member.chmod(0o644)
        source_member.write_bytes(b"caller mutation after publication")
        source_member.chmod(0o444)
        assert target_member.read_bytes() == b"trusted sidecar bytes"
    finally:
        os.close(descriptor)
        _make_directory_tree_writable(staged)
        _make_directory_tree_writable(target)

    assert staged.exists()
    assert not any(tmp_path.glob(".joint-rigger-tree-copy-*"))
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_directory_descriptor_path_replacement_after_backup_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged-sidecar"
    staged.mkdir()
    (staged / "asset.bin").write_bytes(b"trusted sidecar bytes")
    _seal_directory_tree(staged)
    displaced = tmp_path / "displaced-sidecar"
    target = tmp_path / "published-sidecar"
    target.mkdir()
    (target / "old.txt").write_text("old sidecar", encoding="utf-8")
    original_replace = artifacts._replace_entry
    path_replaced = False

    def replace_path_after_backup(source: Any, destination: Any) -> None:
        nonlocal path_replaced
        original_replace(source, destination)
        if path_replaced:
            return
        path_replaced = True
        staged.rename(displaced)
        staged.mkdir()
        (staged / "asset.bin").write_bytes(b"attacker replacement")

    monkeypatch.setattr(artifacts, "_replace_entry", replace_path_after_backup)
    descriptor = os.open(
        staged,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        with pytest.raises(RuntimeError, match="entry changed inode"):
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        staged,
                        target,
                        "composition sidecar",
                        source_descriptor=descriptor,
                        source_sha256=(
                            artifacts.directory_descriptor_tree_sha256(descriptor)
                        ),
                    )
                ]
            )
    finally:
        os.close(descriptor)
        _make_directory_tree_writable(staged)
        _make_directory_tree_writable(displaced)
        _make_directory_tree_writable(target)

    assert path_replaced
    assert (target / "old.txt").read_text(encoding="utf-8") == "old sidecar"
    assert (staged / "asset.bin").read_bytes() == b"attacker replacement"
    assert (displaced / "asset.bin").read_bytes() == b"trusted sidecar bytes"
    assert not any(tmp_path.glob(".joint-rigger-tree-copy-*"))
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_directory_descriptor_content_mutation_after_backup_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged-sidecar"
    staged.mkdir()
    member = staged / "asset.bin"
    member.write_bytes(b"trusted sidecar bytes")
    writer = os.open(member, os.O_WRONLY)
    _seal_directory_tree(staged)
    target = tmp_path / "published-sidecar"
    target.mkdir()
    (target / "old.txt").write_text("old sidecar", encoding="utf-8")
    original_replace = artifacts._replace_entry
    content_mutated = False

    def mutate_content_after_backup(source: Any, destination: Any) -> None:
        nonlocal content_mutated
        original_replace(source, destination)
        if content_mutated:
            return
        content_mutated = True
        os.ftruncate(writer, 0)
        os.pwrite(writer, b"attacker mutation", 0)

    monkeypatch.setattr(artifacts, "_replace_entry", mutate_content_after_backup)
    descriptor = os.open(
        staged,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        expected_sha256 = artifacts.directory_descriptor_tree_sha256(descriptor)
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        staged,
                        target,
                        "composition sidecar",
                        source_descriptor=descriptor,
                        source_sha256=expected_sha256,
                    )
                ]
            )
    finally:
        os.close(descriptor)
        os.close(writer)
        _make_directory_tree_writable(staged)
        _make_directory_tree_writable(target)

    assert content_mutated
    assert member.read_bytes() == b"attacker mutation"
    assert (target / "old.txt").read_text(encoding="utf-8") == "old sidecar"
    assert not any(tmp_path.glob(".joint-rigger-tree-copy-*"))
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


@pytest.mark.parametrize("operation", ["hash", "copy"])
def test_directory_descriptor_fifo_swap_between_stat_and_open_fails_promptly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    source = tmp_path / "source-tree"
    source.mkdir()
    member = source / "asset.bin"
    member.write_bytes(b"trusted bytes")
    _seal_directory_tree(source)
    target = tmp_path / "target-tree"
    target.mkdir()
    source_descriptor = os.open(
        source,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    target_descriptor = os.open(
        target,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    real_open = artifacts.os.open
    swapped = False

    def swap_member_to_fifo(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and path == member.name and dir_fd == source_descriptor:
            swapped = True
            assert flags & getattr(os, "O_NONBLOCK", 0)
            os.fchmod(source_descriptor, 0o700)
            os.unlink(member.name, dir_fd=source_descriptor)
            os.mkfifo(member.name, dir_fd=source_descriptor)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", swap_member_to_fifo)
    try:
        with pytest.raises(
            RuntimeError, match=r"changed inode|special (entry|file)|not sealed"
        ):
            if operation == "hash":
                artifacts.directory_descriptor_tree_sha256(source_descriptor)
            else:
                artifacts.copy_directory_descriptor_tree(
                    source_descriptor,
                    target_descriptor,
                    label="composition sidecar",
                )
    finally:
        os.close(target_descriptor)
        os.close(source_descriptor)
        _make_directory_tree_writable(source)
        _make_directory_tree_writable(target)

    assert swapped
    assert stat.S_ISFIFO(member.lstat().st_mode)


def test_descriptor_private_target_fifo_swap_before_commit_rolls_back_promptly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    staged.write_bytes(trusted_bytes)
    staged.chmod(0o400)
    target.write_bytes(b"old target")
    real_open = artifacts.os.open
    swapped = False

    def swap_private_target_to_fifo(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if (
            not swapped
            and isinstance(path, str)
            and path.startswith(".joint-rigger-copy-")
            and dir_fd is not None
            and flags & os.O_ACCMODE == os.O_RDONLY
        ):
            swapped = True
            assert flags & getattr(os, "O_NONBLOCK", 0)
            os.unlink(path, dir_fd=dir_fd)
            os.mkfifo(path, dir_fd=dir_fd)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", swap_private_target_to_fifo)
    with staged.open("rb") as source:
        with pytest.raises(RuntimeError, match="changed after detached copy"):
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        staged,
                        target,
                        "generated root",
                        source_descriptor=source.fileno(),
                        source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
                    )
                ]
            )

    assert swapped
    assert target.read_bytes() == b"old target"
    assert staged.read_bytes() == trusted_bytes
    preserved_replacements = list(tmp_path.glob(".joint-rigger-copy-*"))
    assert len(preserved_replacements) == 1
    assert stat.S_ISFIFO(preserved_replacements[0].lstat().st_mode)
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_prebackup_validator_runs_once_under_locks_before_any_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path)
    targets.output_path.write_text("old output", encoding="utf-8")
    targets.diagnostics_path.write_text("old diagnostics", encoding="utf-8")
    targets.result_path.write_text("old result", encoding="utf-8")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    events: list[str] = []
    validator_calls = 0
    original_parent_validation = artifacts._require_bound_artifact_parents_unchanged
    original_replace = artifacts._replace_entry

    def record_parent_validation(bound_artifacts: list[Any]) -> None:
        events.append("parents validated")
        original_parent_validation(bound_artifacts)

    def record_replace(source: Any, target: Any) -> None:
        events.append("move")
        original_replace(source, target)

    def validate_before_backup() -> None:
        nonlocal validator_calls
        validator_calls += 1
        events.append("prebackup validator")
        assert targets.output_path.read_text(encoding="utf-8") == "old output"
        assert targets.diagnostics_path.read_text(encoding="utf-8") == (
            "old diagnostics"
        )
        assert targets.result_path.read_text(encoding="utf-8") == "old result"
        assert bundle.staged_targets.output_path.read_text(encoding="utf-8") == (
            "new output"
        )
        assert (
            bundle.staged_targets.diagnostics_path.read_text(encoding="utf-8")
            == "new diagnostics"
        )
        assert bundle.staged_targets.result_path.read_text(encoding="utf-8") == (
            "new result"
        )
        with pytest.raises(
            ConcurrentArtifactPublicationError,
            match="already targeting parent",
        ):
            with artifacts._publication_target_locks([targets.output_path]):
                pytest.fail("prebackup validator must run under the target lock")

    monkeypatch.setattr(
        artifacts,
        "_require_bound_artifact_parents_unchanged",
        record_parent_validation,
    )
    monkeypatch.setattr(artifacts, "_replace_entry", record_replace)
    try:
        promote_staged_artifacts(
            staged_promotion_artifacts(bundle),
            prebackup_validator=validate_before_backup,
        )
    finally:
        bundle.cleanup()

    assert validator_calls == 1
    assert events[:5] == [
        "parents validated",
        "parents validated",
        "prebackup validator",
        "parents validated",
        "move",
    ]
    assert targets.output_path.read_text(encoding="utf-8") == "new output"
    assert targets.diagnostics_path.read_text(encoding="utf-8") == "new diagnostics"
    assert targets.result_path.read_text(encoding="utf-8") == "new result"


def test_prebackup_validator_failure_preserves_old_bundle_and_staged_sources(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path)
    targets.output_path.write_text("old output", encoding="utf-8")
    targets.diagnostics_path.write_text("old diagnostics", encoding="utf-8")
    targets.result_path.write_text("old result", encoding="utf-8")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    validator_calls = 0

    def reject_before_backup() -> None:
        nonlocal validator_calls
        validator_calls += 1
        raise RuntimeError("forced prebackup rejection")

    try:
        with pytest.raises(RuntimeError, match="forced prebackup rejection"):
            promote_staged_artifacts(
                staged_promotion_artifacts(bundle),
                prebackup_validator=reject_before_backup,
            )

        assert validator_calls == 1
        assert targets.output_path.read_text(encoding="utf-8") == "old output"
        assert targets.diagnostics_path.read_text(encoding="utf-8") == (
            "old diagnostics"
        )
        assert targets.result_path.read_text(encoding="utf-8") == "old result"
        assert bundle.staged_targets.output_path.read_text(encoding="utf-8") == (
            "new output"
        )
        assert (
            bundle.staged_targets.diagnostics_path.read_text(encoding="utf-8")
            == "new diagnostics"
        )
        assert bundle.staged_targets.result_path.read_text(encoding="utf-8") == (
            "new result"
        )
        assert not any(tmp_path.glob(".joint-rigger.rollback-*"))
        assert not any(tmp_path.glob(".joint-rigger-copy-*"))
        with artifacts._publication_target_locks([targets.output_path]):
            pass
    finally:
        bundle.cleanup()


def test_prebackup_validator_is_not_invoked_when_staged_binding_fails(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "writable-staged.usda"
    target = tmp_path / "target.usda"
    staged_bytes = b"new staged root"
    staged.write_bytes(staged_bytes)
    staged.chmod(0o600)
    target.write_bytes(b"old target")
    validator_calls = 0

    def validate_before_backup() -> None:
        nonlocal validator_calls
        validator_calls += 1

    with staged.open("rb") as source:
        with pytest.raises(RuntimeError, match="no write permissions"):
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        staged,
                        target,
                        "generated root",
                        source_descriptor=source.fileno(),
                        source_sha256=hashlib.sha256(staged_bytes).hexdigest(),
                    )
                ],
                prebackup_validator=validate_before_backup,
            )

    assert validator_calls == 0
    assert staged.read_bytes() == staged_bytes
    assert target.read_bytes() == b"old target"
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))
    with artifacts._publication_target_locks([target]):
        pass


def test_prebackup_validator_is_not_invoked_when_target_lock_fails(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    staged.write_bytes(b"new staged root")
    target.write_bytes(b"old target")
    validator_calls = 0

    def validate_before_backup() -> None:
        nonlocal validator_calls
        validator_calls += 1

    with artifacts._publication_target_locks([target]):
        with pytest.raises(
            ConcurrentArtifactPublicationError,
            match="already targeting parent",
        ):
            promote_staged_artifacts(
                [StagedArtifact(staged, target, "generated root")],
                prebackup_validator=validate_before_backup,
            )

    assert validator_calls == 0
    assert staged.read_bytes() == b"new staged root"
    assert target.read_bytes() == b"old target"
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_precommit_validator_runs_once_after_private_root_is_sealed(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    staged.write_bytes(trusted_bytes)
    staged.chmod(0o400)
    target.write_bytes(b"old target")
    validator_calls = 0

    def validate_commit_gate() -> None:
        nonlocal validator_calls
        validator_calls += 1
        assert not target.exists()
        private_targets = list(tmp_path.glob(".joint-rigger-copy-*"))
        assert len(private_targets) == 1
        assert private_targets[0].read_bytes() == trusted_bytes
        assert stat.S_IMODE(private_targets[0].stat().st_mode) == 0o444

    with staged.open("rb") as source:
        promote_staged_artifacts(
            [
                StagedArtifact(
                    staged,
                    target,
                    "generated root",
                    source_descriptor=source.fileno(),
                    source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
                )
            ],
            precommit_validator=validate_commit_gate,
        )

    assert validator_calls == 1
    assert target.read_bytes() == trusted_bytes
    assert not staged.exists()
    assert not any(tmp_path.glob(".joint-rigger-copy-*"))
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_precommit_baseexception_restores_previous_complete_bundle(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path)
    targets.output_path.write_text("old output", encoding="utf-8")
    targets.diagnostics_path.write_text("old diagnostics", encoding="utf-8")
    targets.result_path.write_text("old result", encoding="utf-8")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    validator_calls = 0

    def interrupt_commit_gate() -> None:
        nonlocal validator_calls
        validator_calls += 1
        raise KeyboardInterrupt("forced precommit interrupt")

    try:
        with pytest.raises(KeyboardInterrupt, match="forced precommit interrupt"):
            promote_staged_artifacts(
                staged_promotion_artifacts(bundle),
                precommit_validator=interrupt_commit_gate,
            )
    finally:
        bundle.cleanup()

    assert validator_calls == 1
    assert targets.output_path.read_text(encoding="utf-8") == "old output"
    assert targets.diagnostics_path.read_text(encoding="utf-8") == "old diagnostics"
    assert targets.result_path.read_text(encoding="utf-8") == "old result"
    assert not any(tmp_path.glob(".*.rollback-*"))
    assert not any(tmp_path.glob(".joint-rigger-copy-*"))


def test_precommit_baseexception_remains_primary_when_backup_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    staged.write_text("new target", encoding="utf-8")
    target.write_text("old target", encoding="utf-8")
    primary_error = KeyboardInterrupt("forced precommit interrupt")
    original_create_backup = artifacts._create_artifact_backup
    real_close = artifacts.os.close
    backup_descriptor: int | None = None
    backup_close_failed = False

    def track_backup(
        bound_artifact: Any,
        *,
        artifact_identity: tuple[int, int],
    ) -> Any:
        nonlocal backup_descriptor
        backup = original_create_backup(
            bound_artifact,
            artifact_identity=artifact_identity,
        )
        backup_descriptor = backup.directory.descriptor
        return backup

    def interrupt_commit_gate() -> None:
        raise primary_error

    def fail_backup_descriptor_close(descriptor: int) -> None:
        nonlocal backup_close_failed
        if not backup_close_failed and descriptor == backup_descriptor:
            backup_close_failed = True
            real_close(descriptor)
            raise OSError("forced backup descriptor close failure")
        real_close(descriptor)

    monkeypatch.setattr(artifacts, "_create_artifact_backup", track_backup)
    monkeypatch.setattr(artifacts.os, "close", fail_backup_descriptor_close)

    with pytest.raises(KeyboardInterrupt) as raised:
        promote_staged_artifacts(
            [StagedArtifact(staged, target, "generated root")],
            precommit_validator=interrupt_commit_gate,
        )

    assert raised.value is primary_error
    assert backup_close_failed
    assert "OSError: forced backup descriptor close failure" in "\n".join(
        raised.value.__notes__
    )
    assert staged.read_text(encoding="utf-8") == "new target"
    assert target.read_text(encoding="utf-8") == "old target"
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_post_backup_rename_baseexception_restores_previous_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    staged.write_text("new target", encoding="utf-8")
    target.write_text("old target", encoding="utf-8")
    real_replace = artifacts._replace_entry
    interrupted = False

    def interrupt_after_backup_rename(source: Any, destination: Any) -> None:
        nonlocal interrupted
        real_replace(source, destination)
        if not interrupted and source.path == target and destination.name == "artifact":
            interrupted = True
            raise KeyboardInterrupt("forced post-backup interrupt")

    monkeypatch.setattr(
        artifacts,
        "_replace_entry",
        interrupt_after_backup_rename,
    )

    with pytest.raises(KeyboardInterrupt, match="forced post-backup interrupt"):
        promote_staged_artifacts([StagedArtifact(staged, target, "generated root")])

    assert interrupted
    assert staged.read_text(encoding="utf-8") == "new target"
    assert target.read_text(encoding="utf-8") == "old target"
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_unmoved_backup_cleanup_preserves_primary_and_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    staged.write_bytes(b"new target")
    target.write_bytes(b"old target")
    primary_error = RuntimeError("forced target-to-backup move failure")
    cleanup_error = SystemExit("forced unmoved backup cleanup failure")
    original_create_backup = artifacts._create_artifact_backup
    original_replace = artifacts._replace_entry
    original_remove_backup = artifacts._remove_backup_directory
    backup_descriptor: int | None = None

    def track_backup(
        bound_artifact: Any,
        *,
        artifact_identity: tuple[int, int],
    ) -> Any:
        nonlocal backup_descriptor
        backup = original_create_backup(
            bound_artifact,
            artifact_identity=artifact_identity,
        )
        backup_descriptor = backup.directory.descriptor
        return backup

    def fail_backup_move(source: Any, destination: Any) -> None:
        if source.path == target and destination.name == "artifact":
            raise primary_error
        original_replace(source, destination)

    def remove_backup_then_terminate(backup: Any) -> None:
        original_remove_backup(backup)
        raise cleanup_error

    monkeypatch.setattr(artifacts, "_create_artifact_backup", track_backup)
    monkeypatch.setattr(artifacts, "_replace_entry", fail_backup_move)
    monkeypatch.setattr(
        artifacts,
        "_remove_backup_directory",
        remove_backup_then_terminate,
    )

    with pytest.raises(RuntimeError) as raised:
        promote_staged_artifacts([StagedArtifact(staged, target, "generated root")])

    assert raised.value is primary_error
    assert "SystemExit: forced unmoved backup cleanup failure" in "\n".join(
        raised.value.__notes__
    )
    assert backup_descriptor is not None
    with pytest.raises(OSError) as closed_descriptor:
        os.fstat(backup_descriptor)
    assert closed_descriptor.value.errno == errno.EBADF
    assert staged.read_bytes() == b"new target"
    assert target.read_bytes() == b"old target"
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_rollback_attempts_all_restores_and_notes_post_rename_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_evidence = tmp_path / "staged-evidence.json"
    staged_root = tmp_path / "staged-root.usda"
    target_evidence = tmp_path / "evidence.json"
    target_root = tmp_path / "root.usda"
    staged_evidence.write_bytes(b"new evidence")
    staged_root.write_bytes(b"new root")
    target_evidence.write_bytes(b"old evidence")
    target_root.write_bytes(b"old root")
    primary_error = RuntimeError("forced root promotion failure")
    restore_error = KeyboardInterrupt("forced post-rename restore interrupt")
    original_replace = artifacts._replace_entry
    restore_targets: list[Path] = []

    def fail_root_and_interrupt_first_restore(
        source: Any,
        destination: Any,
    ) -> None:
        if source.path == staged_root and destination.path == target_root:
            raise primary_error
        if source.name == "artifact" and destination.path in {
            target_evidence,
            target_root,
        }:
            restore_targets.append(destination.path)
            original_replace(source, destination)
            if destination.path == target_evidence:
                raise restore_error
            return
        original_replace(source, destination)

    monkeypatch.setattr(
        artifacts,
        "_replace_entry",
        fail_root_and_interrupt_first_restore,
    )

    with pytest.raises(RuntimeError) as raised:
        promote_staged_artifacts(
            [
                StagedArtifact(
                    staged_evidence,
                    target_evidence,
                    "evidence report",
                ),
                StagedArtifact(staged_root, target_root, "generated root"),
            ]
        )

    assert raised.value is primary_error
    notes = "\n".join(raised.value.__notes__)
    assert "KeyboardInterrupt: forced post-rename restore interrupt" in notes
    assert "Artifact rollback was incomplete" in notes
    assert restore_targets == [target_evidence, target_root]
    assert target_evidence.read_bytes() == b"old evidence"
    assert target_root.read_bytes() == b"old root"
    rollback_directories = list(tmp_path.glob(".joint-rigger.rollback-*"))
    assert len(rollback_directories) == 2
    assert all(not any(directory.iterdir()) for directory in rollback_directories)


def test_rolled_back_backup_cleanup_fatal_preserves_operation_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    staged.write_bytes(b"new target")
    target.write_bytes(b"old target")
    primary_error = RuntimeError("forced promotion failure before commit")
    cleanup_error = SystemExit("forced rolled-back backup cleanup failure")
    original_replace = artifacts._replace_entry
    original_remove_backup = artifacts._remove_backup_directory
    cleanup_calls = 0

    def fail_promotion(source: Any, destination: Any) -> None:
        if source.path == staged and destination.path == target:
            raise primary_error
        original_replace(source, destination)

    def remove_backup_then_terminate(backup: Any) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        original_remove_backup(backup)
        raise cleanup_error

    monkeypatch.setattr(artifacts, "_replace_entry", fail_promotion)
    monkeypatch.setattr(
        artifacts,
        "_remove_backup_directory",
        remove_backup_then_terminate,
    )

    with pytest.raises(RuntimeError) as raised:
        promote_staged_artifacts([StagedArtifact(staged, target, "generated root")])

    assert raised.value is primary_error
    assert cleanup_calls == 1
    assert "SystemExit: forced rolled-back backup cleanup failure" in "\n".join(
        raised.value.__notes__
    )
    assert staged.read_bytes() == b"new target"
    assert target.read_bytes() == b"old target"
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_backup_move_rejects_substituted_payload_and_preserves_both_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    displaced = tmp_path / "owned-target-displaced.usda"
    staged.write_text("new target", encoding="utf-8")
    target.write_text("old target", encoding="utf-8")
    real_replace = artifacts._replace_entry
    substituted = False

    def substitute_before_backup(source: Any, destination: Any) -> None:
        nonlocal substituted
        if not substituted and source.path == target and destination.name == "artifact":
            substituted = True
            target.rename(displaced)
            target.write_text("foreign target", encoding="utf-8")
        real_replace(source, destination)

    monkeypatch.setattr(artifacts, "_replace_entry", substitute_before_backup)

    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        promote_staged_artifacts([StagedArtifact(staged, target, "generated root")])

    assert substituted
    assert not target.exists()
    assert displaced.read_text(encoding="utf-8") == "old target"
    rollback_directories = list(tmp_path.glob(".joint-rigger.rollback-*"))
    assert len(rollback_directories) == 1
    assert (rollback_directories[0] / "artifact").read_text(
        encoding="utf-8"
    ) == "foreign target"
    assert staged.read_text(encoding="utf-8") == "new target"


@pytest.mark.parametrize("race_recovery", [False, True])
def test_backup_restore_rejects_late_payload_swap_and_recovers_foreign_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race_recovery: bool,
) -> None:
    staged_evidence = tmp_path / "staged-evidence.json"
    staged_root = tmp_path / "staged-root.usda"
    target_evidence = tmp_path / "evidence.json"
    target_root = tmp_path / "root.usda"
    staged_evidence.write_text("new evidence", encoding="utf-8")
    staged_root.write_text("new root", encoding="utf-8")
    target_evidence.write_text("old evidence", encoding="utf-8")
    target_root.write_text("old root", encoding="utf-8")
    real_replace = artifacts._replace_entry
    real_rename = artifacts._rename_descriptor_entry_noreplace
    substituted = False
    primary_error = OSError("forced root promotion failure")

    def substitute_during_rollback(source: Any, destination: Any) -> None:
        nonlocal substituted
        if source.path == staged_root and destination.path == target_root:
            raise primary_error
        if (
            not substituted
            and source.name == "artifact"
            and destination.path == target_evidence
        ):
            substituted = True
            source.path.rename(source.parent.opened_path / "owned-displaced")
            source.path.write_text("foreign evidence", encoding="utf-8")
        real_replace(source, destination)

    def displace_foreign_after_recovery(
        source_parent_descriptor: int,
        source_name: str,
        target_parent_descriptor: int,
        target_name: str,
        *,
        label: str,
    ) -> None:
        real_rename(
            source_parent_descriptor,
            source_name,
            target_parent_descriptor,
            target_name,
            label=label,
        )
        if label == "mismatched artifact backup restore recovery":
            os.rename(
                target_name,
                "foreign-preserved",
                src_dir_fd=target_parent_descriptor,
                dst_dir_fd=target_parent_descriptor,
            )

    monkeypatch.setattr(artifacts, "_replace_entry", substitute_during_rollback)
    if race_recovery:
        monkeypatch.setattr(
            artifacts,
            "_rename_descriptor_entry_noreplace",
            displace_foreign_after_recovery,
        )

    with pytest.raises(OSError) as raised:
        promote_staged_artifacts(
            [
                StagedArtifact(
                    staged_evidence,
                    target_evidence,
                    "evidence report",
                ),
                StagedArtifact(staged_root, target_root, "generated root"),
            ]
        )

    assert raised.value is primary_error
    notes = "\n".join(raised.value.__notes__)
    assert "Artifact rollback was incomplete" in notes
    assert "Artifact backup state changed after rollback restore" in notes
    assert substituted
    assert not target_evidence.exists()
    assert target_root.read_text(encoding="utf-8") == "old root"
    evidence_backups = [
        path
        for path in tmp_path.glob(".joint-rigger.rollback-*")
        if (path / "owned-displaced").exists()
    ]
    assert len(evidence_backups) == 1
    assert (evidence_backups[0] / "owned-displaced").read_text(
        encoding="utf-8"
    ) == "old evidence"
    foreign_name = "foreign-preserved" if race_recovery else "artifact"
    assert (evidence_backups[0] / foreign_name).read_text(encoding="utf-8") == (
        "foreign evidence"
    )


def test_post_noncommit_rename_baseexception_restores_previous_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path)
    targets.output_path.write_text("old output", encoding="utf-8")
    targets.diagnostics_path.write_text("old diagnostics", encoding="utf-8")
    targets.result_path.write_text("old result", encoding="utf-8")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    real_replace = artifacts._replace_entry
    interrupted = False

    def interrupt_after_noncommit_rename(source: Any, destination: Any) -> None:
        nonlocal interrupted
        real_replace(source, destination)
        if (
            not interrupted
            and source.path == bundle.staged_targets.diagnostics_path
            and destination.path == targets.diagnostics_path
        ):
            interrupted = True
            raise KeyboardInterrupt("forced post-promotion interrupt")

    monkeypatch.setattr(
        artifacts,
        "_replace_entry",
        interrupt_after_noncommit_rename,
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="forced post-promotion interrupt"):
            promote_staged_artifacts(staged_promotion_artifacts(bundle))
    finally:
        bundle.cleanup()

    assert interrupted
    assert targets.output_path.read_text(encoding="utf-8") == "old output"
    assert targets.diagnostics_path.read_text(encoding="utf-8") == "old diagnostics"
    assert targets.result_path.read_text(encoding="utf-8") == "old result"
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_post_commit_rename_baseexception_preserves_committed_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    staged.write_text("new target", encoding="utf-8")
    target.write_text("old target", encoding="utf-8")
    real_replace = artifacts._replace_entry
    interrupted = False

    def interrupt_after_commit_rename(source: Any, destination: Any) -> None:
        nonlocal interrupted
        real_replace(source, destination)
        if not interrupted and source.path == staged and destination.path == target:
            interrupted = True
            raise KeyboardInterrupt("forced post-commit interrupt")

    monkeypatch.setattr(
        artifacts,
        "_replace_entry",
        interrupt_after_commit_rename,
    )

    with pytest.raises(KeyboardInterrupt, match="forced post-commit interrupt"):
        promote_staged_artifacts([StagedArtifact(staged, target, "generated root")])

    assert interrupted
    assert not staged.exists()
    assert target.read_text(encoding="utf-8") == "new target"
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_post_commit_rename_baseexception_marks_staging_commit_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    root_artifact = staged_promotion_artifacts(bundle)[-1]
    staged = root_artifact.staged_path
    target = root_artifact.target_path
    state = root_artifact._promotion_state
    assert state is not None
    real_replace = artifacts._replace_entry
    expected = KeyboardInterrupt("forced post-commit state interrupt")

    def interrupt_after_commit_rename(source: Any, destination: Any) -> None:
        real_replace(source, destination)
        raise expected

    monkeypatch.setattr(
        artifacts,
        "_replace_entry",
        interrupt_after_commit_rename,
    )

    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            promote_staged_artifacts([root_artifact])

        assert raised.value is expected
        assert state.committed
        assert not staged.exists()
        assert target.read_text(encoding="utf-8") == "new output"
        assert not any(tmp_path.glob(".*.rollback-*"))
    finally:
        bundle.cleanup()


def test_promotion_refuses_commit_state_when_target_is_replaced_after_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    root_artifact = staged_promotion_artifacts(bundle)[-1]
    state = root_artifact._promotion_state
    assert state is not None
    displaced = tmp_path / "exact-promoted-displaced.usda"
    real_replace = artifacts._replace_entry

    def replace_then_substitute(source: Any, destination: Any) -> None:
        real_replace(source, destination)
        if destination.path == targets.output_path:
            destination.path.rename(displaced)
            destination.path.write_text("foreign root", encoding="utf-8")

    monkeypatch.setattr(artifacts, "_replace_entry", replace_then_substitute)

    with pytest.raises(RuntimeError, match="changed inode after rename"):
        promote_staged_artifacts([root_artifact])

    assert not state.committed
    assert targets.output_path.read_text(encoding="utf-8") == "foreign root"
    assert displaced.read_text(encoding="utf-8") == "new output"
    with pytest.raises(RuntimeError, match="held inode preserved elsewhere"):
        bundle.cleanup()


def test_promotion_revalidates_retained_file_after_rename_window_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    root_artifact = staged_promotion_artifacts(bundle)[-1]
    state = root_artifact._promotion_state
    assert state is not None
    alias = tmp_path / "rename-window-root-alias.usda"
    real_replace = artifacts._replace_entry

    def hardlink_then_replace(source: Any, destination: Any) -> None:
        if destination.path == targets.output_path:
            os.link(source.path, alias)
        real_replace(source, destination)

    monkeypatch.setattr(artifacts, "_replace_entry", hardlink_then_replace)

    with pytest.raises(RuntimeError, match="gained additional links"):
        promote_staged_artifacts([root_artifact])

    assert not state.committed
    assert not targets.output_path.exists()
    assert alias.read_text(encoding="utf-8") == "new output"
    with pytest.raises(RuntimeError, match="held inode preserved elsewhere"):
        bundle.cleanup()
    alias.unlink()


def test_precommit_gate_rolls_back_mutated_directory_target_before_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_sidecar = tmp_path / "staged-sidecar"
    staged_sidecar.mkdir()
    (staged_sidecar / "asset.bin").write_bytes(b"trusted sidecar bytes")
    _seal_directory_tree(staged_sidecar)
    staged_root = tmp_path / "staged-root.usda"
    staged_root.write_text("new root", encoding="utf-8")
    target_sidecar = tmp_path / "published-sidecar"
    target_sidecar.mkdir()
    (target_sidecar / "old.txt").write_text("old sidecar", encoding="utf-8")
    target_root = tmp_path / "published-root.usda"
    target_root.write_text("old root", encoding="utf-8")
    original_precommit = artifacts._require_precommit_descriptor_artifacts
    gate_called = False

    def mutate_target_at_precommit_gate(
        promoted: list[Any],
        detached_targets: dict[int, Any],
    ) -> None:
        nonlocal gate_called
        gate_called = True
        assert not target_root.exists()
        member = target_sidecar / "asset.bin"
        member.chmod(0o644)
        member.write_bytes(b"attacker mutation")
        member.chmod(0o444)
        original_precommit(promoted, detached_targets)

    monkeypatch.setattr(
        artifacts,
        "_require_precommit_descriptor_artifacts",
        mutate_target_at_precommit_gate,
    )
    descriptor = os.open(
        staged_sidecar,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        with pytest.raises(RuntimeError, match="failed SHA-256 verification"):
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        staged_sidecar,
                        target_sidecar,
                        "composition sidecar",
                        source_descriptor=descriptor,
                        source_sha256=(
                            artifacts.directory_descriptor_tree_sha256(descriptor)
                        ),
                    ),
                    StagedArtifact(
                        staged_root,
                        target_root,
                        "generated root",
                    ),
                ]
            )
    finally:
        os.close(descriptor)
        _make_directory_tree_writable(staged_sidecar)
        _make_directory_tree_writable(target_sidecar)

    assert gate_called
    assert target_root.read_text(encoding="utf-8") == "old root"
    assert (target_sidecar / "old.txt").read_text(encoding="utf-8") == "old sidecar"
    assert staged_root.read_text(encoding="utf-8") == "new root"
    assert not any(tmp_path.glob(".joint-rigger-tree-copy-*"))
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_precommit_revalidates_promoted_report_hardlinks(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    alias = tmp_path / "promoted-diagnostics-alias.json"

    def hardlink_promoted_diagnostics() -> None:
        os.link(targets.diagnostics_path, alias)

    with pytest.raises(RuntimeError, match="gained additional links"):
        promote_staged_artifacts(
            promotion,
            precommit_validator=hardlink_promoted_diagnostics,
        )

    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()
    assert alias.read_text(encoding="utf-8") == "new diagnostics"
    with pytest.raises(RuntimeError, match="without an exact recorded promotion"):
        bundle.cleanup()
    alias.unlink()


@pytest.mark.parametrize(
    "mutation",
    ["hardlink", "new_file", "modify", "remove"],
)
def test_precommit_revalidates_promoted_sidecar_tree(
    tmp_path: Path,
    mutation: str,
) -> None:
    targets = _targets(tmp_path, sidecar=True)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    assert targets.sidecar_path is not None
    sidecar_file = targets.sidecar_path / "asset.txt"
    alias = tmp_path / "promoted-sidecar-alias.txt"

    def mutate_promoted_sidecar() -> None:
        if mutation == "hardlink":
            os.link(sidecar_file, alias)
            return
        if mutation == "new_file":
            os.chmod(targets.sidecar_path, 0o755)
            (targets.sidecar_path / "added.txt").write_text(
                "added",
                encoding="utf-8",
            )
            return
        if mutation == "modify":
            os.chmod(sidecar_file, 0o644)
            sidecar_file.write_text("changed", encoding="utf-8")
            return
        os.chmod(targets.sidecar_path, 0o755)
        sidecar_file.unlink()

    with pytest.raises(RuntimeError, match="tree|Descriptor-backed target"):
        promote_staged_artifacts(
            promotion,
            precommit_validator=mutate_promoted_sidecar,
        )

    assert not targets.output_path.exists()
    assert not targets.sidecar_path.exists()
    if alias.exists():
        alias.unlink()
    bundle.cleanup()


def test_private_directory_operation_keeps_primary_when_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged-sidecar"
    staged.mkdir()
    (staged / "asset.bin").write_bytes(b"trusted sidecar bytes")
    _seal_directory_tree(staged)
    target = tmp_path / "published-sidecar"
    descriptor = os.open(
        staged,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    expected = RuntimeError("forced private directory operation failure")
    cleanup_error = RuntimeError("forced private directory cleanup failure")
    real_remove = artifacts._remove_bound_entry

    def fail_copy(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise expected

    def fail_private_cleanup(entry: Any, **kwargs: object) -> None:
        if entry.name.startswith(".joint-rigger-tree-copy-"):
            raise cleanup_error
        real_remove(entry, **kwargs)

    monkeypatch.setattr(
        artifacts,
        "_copy_directory_descriptor_source_to_target",
        fail_copy,
    )
    monkeypatch.setattr(artifacts, "_remove_bound_entry", fail_private_cleanup)
    try:
        with pytest.raises(RuntimeError) as raised:
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        staged,
                        target,
                        "composition sidecar",
                        source_descriptor=descriptor,
                        source_sha256=(
                            artifacts.directory_descriptor_tree_sha256(descriptor)
                        ),
                    )
                ]
            )
    finally:
        os.close(descriptor)
        _make_directory_tree_writable(staged)

    assert raised.value is expected
    assert any(
        "forced private directory cleanup failure" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert not target.exists()


@pytest.mark.parametrize("source_kind", ["file", "directory"])
def test_committed_private_copy_cleanup_error_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
) -> None:
    staged = tmp_path / "staged"
    target = tmp_path / "target"
    if source_kind == "directory":
        staged.mkdir()
        (staged / "asset.bin").write_bytes(b"trusted directory bytes")
        _seal_directory_tree(staged)
        descriptor = os.open(
            staged,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        source_sha256 = artifacts.directory_descriptor_tree_sha256(descriptor)
        private_prefix = ".joint-rigger-tree-copy-"
    else:
        staged.write_bytes(b"trusted file bytes")
        staged.chmod(0o400)
        descriptor = os.open(staged, os.O_RDONLY | os.O_NOFOLLOW)
        source_sha256 = hashlib.sha256(b"trusted file bytes").hexdigest()
        private_prefix = ".joint-rigger-copy-"
    cleanup_error = RuntimeError(f"forced {source_kind} cleanup-only failure")
    real_remove = artifacts._remove_bound_entry

    def fail_private_cleanup(entry: Any, **kwargs: object) -> None:
        if entry.name.startswith(private_prefix):
            raise cleanup_error
        real_remove(entry, **kwargs)

    monkeypatch.setattr(artifacts, "_remove_bound_entry", fail_private_cleanup)
    try:
        with pytest.raises(
            CommittedArtifactPublicationCleanupError,
            match=f"forced {source_kind} cleanup-only failure",
        ) as raised:
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        staged,
                        target,
                        "generated artifact",
                        source_descriptor=descriptor,
                        source_sha256=source_sha256,
                    )
                ]
            )
    finally:
        os.close(descriptor)
        if source_kind == "directory":
            _make_directory_tree_writable(staged)

    assert raised.value.committed
    if source_kind == "directory":
        assert (target / "asset.bin").read_bytes() == b"trusted directory bytes"
    else:
        assert target.read_bytes() == b"trusted file bytes"


def test_committed_private_finalizer_fatal_outranks_normal_and_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted file bytes"
    staged.write_bytes(trusted_bytes)
    staged.chmod(0o400)
    source_descriptor = os.open(staged, os.O_RDONLY | os.O_NOFOLLOW)
    ordinary_error = OSError("forced private-name cleanup failure")
    fatal_error = SystemExit("forced private descriptor close failure")
    original_create_private = artifacts._create_private_detached_target
    original_remove = artifacts._remove_bound_entry
    original_close = artifacts.os.close
    private_descriptor: int | None = None
    close_injected = False

    def track_private_target(bound_artifact: Any) -> Any:
        nonlocal private_descriptor
        result = original_create_private(bound_artifact)
        private_descriptor = result[1]
        return result

    def fail_private_name_cleanup(entry: Any, **kwargs: object) -> None:
        if entry.name.startswith(".joint-rigger-copy-"):
            raise ordinary_error
        original_remove(entry, **kwargs)

    def close_private_then_terminate(descriptor: int) -> None:
        nonlocal close_injected
        original_close(descriptor)
        if descriptor == private_descriptor and not close_injected:
            close_injected = True
            raise fatal_error

    monkeypatch.setattr(
        artifacts,
        "_create_private_detached_target",
        track_private_target,
    )
    monkeypatch.setattr(artifacts, "_remove_bound_entry", fail_private_name_cleanup)
    monkeypatch.setattr(artifacts.os, "close", close_private_then_terminate)
    try:
        with pytest.raises(SystemExit) as raised:
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        staged,
                        target,
                        "generated root",
                        source_descriptor=source_descriptor,
                        source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
                    )
                ]
            )
    finally:
        original_close(source_descriptor)

    assert raised.value is fatal_error
    assert close_injected
    assert "OSError: forced private-name cleanup failure" in "\n".join(
        raised.value.__notes__
    )
    assert private_descriptor is not None
    with pytest.raises(OSError) as closed_descriptor:
        os.fstat(private_descriptor)
    assert closed_descriptor.value.errno == errno.EBADF
    assert target.read_bytes() == trusted_bytes


def test_existing_target_nested_mount_is_rejected_before_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged-sidecar"
    staged.mkdir()
    (staged / "new.txt").write_text("new sidecar", encoding="utf-8")
    target = tmp_path / "published-sidecar"
    nested = target / "nested"
    nested.mkdir(parents=True)
    (nested / "old.txt").write_text("old sidecar", encoding="utf-8")
    nested_identity = (nested.stat().st_dev, nested.stat().st_ino)
    real_mount_id = artifacts._descriptor_mount_id

    def simulated_mount_id(descriptor: int) -> int:
        mount_id = real_mount_id(descriptor)
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == nested_identity:
            return mount_id + 1
        return mount_id

    monkeypatch.setattr(artifacts, "_descriptor_mount_id", simulated_mount_id)

    with pytest.raises(ValueError, match="mount point at nested"):
        promote_staged_artifacts(
            [StagedArtifact(staged, target, "composition sidecar")]
        )

    assert (staged / "new.txt").read_text(encoding="utf-8") == "new sidecar"
    assert (nested / "old.txt").read_text(encoding="utf-8") == "old sidecar"
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_backup_cleanup_rechecks_mounts_after_initial_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged-sidecar"
    staged.mkdir()
    (staged / "new.txt").write_text("new sidecar", encoding="utf-8")
    target = tmp_path / "published-sidecar"
    nested = target / "nested"
    nested.mkdir(parents=True)
    (nested / "old.txt").write_text("old sidecar", encoding="utf-8")
    nested_identity = (nested.stat().st_dev, nested.stat().st_ino)
    real_mount_id = artifacts._descriptor_mount_id
    real_replace = artifacts._replace_entry
    backup_moved = False

    def mount_id_changes_after_backup(descriptor: int) -> int:
        mount_id = real_mount_id(descriptor)
        metadata = os.fstat(descriptor)
        if backup_moved and (metadata.st_dev, metadata.st_ino) == nested_identity:
            return mount_id + 1
        return mount_id

    def mark_backup_moved(source: Any, destination: Any) -> None:
        nonlocal backup_moved
        real_replace(source, destination)
        if source.path == target and destination.name == "artifact":
            backup_moved = True

    monkeypatch.setattr(
        artifacts, "_descriptor_mount_id", mount_id_changes_after_backup
    )
    monkeypatch.setattr(artifacts, "_replace_entry", mark_backup_moved)

    with pytest.raises(
        CommittedArtifactPublicationCleanupError,
        match="cleanup crossed a mount",
    ) as caught:
        promote_staged_artifacts(
            [StagedArtifact(staged, target, "composition sidecar")]
        )

    assert caught.value.committed
    assert backup_moved
    assert not staged.exists()
    assert (target / "new.txt").read_text(encoding="utf-8") == "new sidecar"
    rollback_directories = list(tmp_path.glob(".joint-rigger.rollback-*"))
    assert len(rollback_directories) == 1
    assert (rollback_directories[0] / "artifact/nested/old.txt").read_text(
        encoding="utf-8"
    ) == "old sidecar"


def test_descriptor_mount_id_requires_one_positive_linux_mount_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = 19
    monkeypatch.setattr(artifacts, "_PROC_SELF_FDINFO", tmp_path)
    fdinfo = tmp_path / str(descriptor)
    fdinfo.write_text("pos:\t0\nmnt_id:\t42\n", encoding="utf-8")

    assert artifacts._descriptor_mount_id(descriptor) == 42

    for contents, error in (
        ("pos:\t0\n", "exactly one mount ID"),
        ("mnt_id\n", "Malformed mount ID"),
        ("mnt_id:\tinvalid\n", "Malformed mount ID"),
        ("mnt_id:\t0\n", "Malformed mount ID"),
        ("mnt_id:\t1\nmnt_id:\t2\n", "exactly one mount ID"),
    ):
        fdinfo.write_text(contents, encoding="utf-8")
        with pytest.raises(RuntimeError, match=error):
            artifacts._descriptor_mount_id(descriptor)

    monkeypatch.setattr(artifacts, "_PROC_SELF_FDINFO", tmp_path / "missing")
    with pytest.raises(RuntimeError, match="requires Linux /proc/self/fdinfo"):
        artifacts._descriptor_mount_id(descriptor)


def test_existing_target_root_mount_is_rejected_before_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged-sidecar"
    staged.mkdir()
    (staged / "new.txt").write_text("new sidecar", encoding="utf-8")
    target = tmp_path / "published-sidecar"
    target.mkdir()
    (target / "old.txt").write_text("old sidecar", encoding="utf-8")
    target_identity = (target.stat().st_dev, target.stat().st_ino)
    real_mount_id = artifacts._descriptor_mount_id

    def simulated_mount_id(descriptor: int) -> int:
        mount_id = real_mount_id(descriptor)
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == target_identity:
            return mount_id + 1
        return mount_id

    monkeypatch.setattr(artifacts, "_descriptor_mount_id", simulated_mount_id)

    with pytest.raises(ValueError, match="root is a mount point"):
        promote_staged_artifacts(
            [StagedArtifact(staged, target, "composition sidecar")]
        )

    assert (staged / "new.txt").read_text(encoding="utf-8") == "new sidecar"
    assert (target / "old.txt").read_text(encoding="utf-8") == "old sidecar"
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_existing_target_inode_swap_is_rejected_before_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged-sidecar"
    staged.mkdir()
    target = tmp_path / "published-sidecar"
    target.mkdir()
    replacement = tmp_path / "replacement-sidecar"
    replacement.mkdir()
    real_open = artifacts.os.open

    def substitute_target_descriptor(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == target.name and dir_fd is not None and flags & os.O_DIRECTORY:
            return real_open(replacement, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", substitute_target_descriptor)

    with pytest.raises(RuntimeError, match="changed inode during mount validation"):
        promote_staged_artifacts(
            [StagedArtifact(staged, target, "composition sidecar")]
        )

    assert staged.is_dir()
    assert target.is_dir()
    assert replacement.is_dir()
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_directory_mount_walk_rejects_races_and_ignores_special_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    child = root / "child.txt"
    child.write_text("payload", encoding="utf-8")
    nested = root / "nested"
    nested.mkdir()
    (nested / "member.txt").write_text("nested payload", encoding="utf-8")
    fifo = root / "ignored.fifo"
    os.mkfifo(fifo)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    expected_mount_id = artifacts._descriptor_mount_id(descriptor)
    try:
        artifacts._require_directory_tree_mount_id(
            descriptor,
            expected_mount_id=expected_mount_id,
            label="Existing sidecar",
        )
        with pytest.raises(ValueError, match=r"mount point at \."):
            artifacts._require_directory_tree_mount_id(
                descriptor,
                expected_mount_id=expected_mount_id + 1,
                label="Existing sidecar",
            )

        child_identity = (child.stat().st_dev, child.stat().st_ino)
        real_fstat = artifacts.os.fstat

        class ChangedStat:
            def __init__(self, original: Any) -> None:
                self.original = original

            def __getattr__(self, name: str) -> Any:
                if name == "st_ino":
                    return self.original.st_ino + 1
                return getattr(self.original, name)

        def changed_child_stat(child_descriptor: int) -> Any:
            metadata = real_fstat(child_descriptor)
            if (metadata.st_dev, metadata.st_ino) == child_identity:
                return ChangedStat(metadata)
            return metadata

        monkeypatch.setattr(artifacts.os, "fstat", changed_child_stat)
        with pytest.raises(RuntimeError, match=r"changed inode at \./child.txt"):
            artifacts._require_directory_tree_mount_id(
                descriptor,
                expected_mount_id=expected_mount_id,
                label="Existing sidecar",
            )
    finally:
        os.close(descriptor)


def _mount_boundary_fixture(root: Path, entry_kind: str) -> Path:
    """Create one root, nested-directory, or nested-file mount candidate."""

    root.mkdir()
    if entry_kind == "root":
        (root / "member.bin").write_bytes(b"root member")
        root.chmod(0o777)
        return root
    if entry_kind == "directory":
        nested = root / "nested"
        nested.mkdir()
        (nested / "member.bin").write_bytes(b"nested member")
        nested.chmod(0o777)
        return nested
    member = root / "member.bin"
    member.write_bytes(b"mounted member")
    member.chmod(0o666)
    return member


def _mount_id_substitution(
    victim: Path,
    real_mount_id: Any,
    *,
    appears_after_first_check: bool,
) -> Any:
    """Return a deterministic mount-ID change for one exact inode."""

    victim_metadata = victim.stat()
    victim_identity = (victim_metadata.st_dev, victim_metadata.st_ino)
    checks = 0

    def substituted(descriptor: int) -> int:
        nonlocal checks
        mount_id = real_mount_id(descriptor)
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) != victim_identity:
            return mount_id
        checks += 1
        if appears_after_first_check and checks == 1:
            return mount_id
        return mount_id + 1

    return substituted


@pytest.mark.parametrize("entry_kind", ["root", "directory", "file"])
@pytest.mark.parametrize("appears_after_first_check", [False, True])
def test_sidecar_digest_rejects_mounts_before_hashing_external_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
    appears_after_first_check: bool,
) -> None:
    sidecar = tmp_path / "sidecar"
    victim = _mount_boundary_fixture(sidecar, entry_kind)
    original_mode = stat.S_IMODE(victim.stat().st_mode)
    real_mount_id = artifacts._descriptor_mount_id
    hash_calls: list[int] = []
    real_hash = artifacts._descriptor_sha256

    def track_hash(descriptor: int, *, label: str) -> str:
        hash_calls.append(descriptor)
        return real_hash(descriptor, label=label)

    monkeypatch.setattr(
        artifacts,
        "_descriptor_mount_id",
        _mount_id_substitution(
            victim,
            real_mount_id,
            appears_after_first_check=appears_after_first_check,
        ),
    )
    monkeypatch.setattr(artifacts, "_descriptor_sha256", track_hash)

    with pytest.raises(ValueError, match="mount point"):
        sidecar_dependency_bundle_sha256(sidecar)

    assert hash_calls == []
    assert stat.S_IMODE(victim.stat().st_mode) == original_mode


@pytest.mark.parametrize("entry_kind", ["root", "directory", "file"])
@pytest.mark.parametrize("appears_after_first_check", [False, True])
def test_staged_sidecar_rejects_mounts_before_any_fchmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
    appears_after_first_check: bool,
) -> None:
    bundle = create_staged_artifact_targets(_targets(tmp_path, sidecar=True))
    _write_staged_bundle(bundle.staged_targets)
    sidecar = bundle.staged_targets.sidecar_path
    assert sidecar is not None
    if entry_kind == "root":
        victim = sidecar
        victim.chmod(0o777)
    elif entry_kind == "directory":
        victim = sidecar / "nested"
        victim.mkdir()
        (victim / "member.bin").write_bytes(b"nested")
        victim.chmod(0o777)
    else:
        victim = sidecar / "asset.txt"
        victim.chmod(0o666)
    original_mode = stat.S_IMODE(victim.stat().st_mode)
    real_mount_id = artifacts._descriptor_mount_id
    real_fchmod = artifacts.os.fchmod
    fchmod_calls: list[int] = []

    def track_fchmod(descriptor: int, mode: int) -> None:
        fchmod_calls.append(descriptor)
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(
        artifacts,
        "_descriptor_mount_id",
        _mount_id_substitution(
            victim,
            real_mount_id,
            appears_after_first_check=appears_after_first_check,
        ),
    )
    monkeypatch.setattr(artifacts.os, "fchmod", track_fchmod)

    with pytest.raises(ValueError, match="mount point"):
        staged_promotion_artifacts(bundle)

    assert fchmod_calls == []
    assert stat.S_IMODE(victim.stat().st_mode) == original_mode

    monkeypatch.setattr(artifacts, "_descriptor_mount_id", real_mount_id)
    monkeypatch.setattr(artifacts.os, "fchmod", real_fchmod)
    staged_promotion_artifacts(bundle)
    bundle.cleanup()


@pytest.mark.parametrize("entry_kind", ["root", "directory", "file"])
@pytest.mark.parametrize("appears_after_first_check", [False, True])
def test_sidecar_copy_rejects_mounts_before_target_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
    appears_after_first_check: bool,
) -> None:
    source = tmp_path / "source"
    victim = _mount_boundary_fixture(source, entry_kind)
    target = tmp_path / "target"
    target.mkdir()
    target_descriptor = os.open(
        target,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    real_mount_id = artifacts._descriptor_mount_id
    hash_calls: list[int] = []
    real_hash = artifacts._descriptor_sha256

    def track_hash(descriptor: int, *, label: str) -> str:
        hash_calls.append(descriptor)
        return real_hash(descriptor, label=label)

    monkeypatch.setattr(
        artifacts,
        "_descriptor_mount_id",
        _mount_id_substitution(
            victim,
            real_mount_id,
            appears_after_first_check=appears_after_first_check,
        ),
    )
    monkeypatch.setattr(artifacts, "_descriptor_sha256", track_hash)
    try:
        with pytest.raises(ValueError, match="mount point"):
            artifacts.copy_sidecar_directory(
                source,
                target_descriptor,
                label="mounted sidecar",
            )
    finally:
        os.close(target_descriptor)

    assert hash_calls == []
    assert list(target.iterdir()) == []


def test_sidecar_copy_accepts_stable_writable_backend_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    member = nested / "asset.bin"
    member.write_bytes(b"backend sidecar bytes")
    source.chmod(0o755)
    nested.chmod(0o755)
    member.chmod(0o644)
    target = tmp_path / "target"
    target.mkdir()
    target_descriptor = os.open(
        target,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        artifacts.copy_sidecar_directory(
            source,
            target_descriptor,
            label="writable backend sidecar",
        )
    finally:
        os.close(target_descriptor)

    assert (target / "nested" / "asset.bin").read_bytes() == b"backend sidecar bytes"
    assert stat.S_IMODE(source.stat().st_mode) == 0o755
    assert stat.S_IMODE(nested.stat().st_mode) == 0o755
    assert stat.S_IMODE(member.stat().st_mode) == 0o644


def test_staged_sidecar_rejects_real_root_and_nested_bind_mounts(
    tmp_path: Path,
) -> None:
    unshare = shutil.which("unshare")
    mount = shutil.which("mount")
    if unshare is None or mount is None:
        pytest.skip("unshare and mount are required for the bind-mount regression")

    probe_source = tmp_path / "probe-source"
    probe_target = tmp_path / "probe-target"
    probe_source.mkdir()
    probe_target.mkdir()
    probe = subprocess.run(
        [
            unshare,
            "--user",
            "--map-root-user",
            "--mount",
            "--fork",
            mount,
            "--bind",
            str(probe_source),
            str(probe_target),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if probe.returncode != 0:
        pytest.skip(
            "unprivileged user-namespace bind mounts are unavailable: "
            + (probe.stderr.strip() or probe.stdout.strip())
        )

    script = textwrap.dedent(
        """
        import os
        import stat
        import subprocess
        import sys
        from pathlib import Path

        from world_understanding.functions.physics.joint_rigger.artifacts import (
            JointRiggerArtifactTargets,
            create_staged_artifact_targets,
            staged_promotion_artifacts,
        )

        root = Path(sys.argv[1])
        mount = sys.argv[2]
        for entry_kind in ("root", "directory", "file"):
            case = root / f"real-bind-{entry_kind}"
            case.mkdir()
            targets = JointRiggerArtifactTargets(
                output_path=case / "rigged.usda",
                diagnostics_path=case / "diagnostics.json",
                result_path=case / "result.json",
                sidecar_path=case / "rigged_assets",
            )
            bundle = create_staged_artifact_targets(targets)
            staged = bundle.staged_targets
            staged.output_path.write_text("root", encoding="utf-8")
            staged.diagnostics_path.write_text("diagnostics", encoding="utf-8")
            staged.result_path.write_text("result", encoding="utf-8")
            assert staged.sidecar_path is not None
            staged.sidecar_path.mkdir()

            external = case / "external"
            if entry_kind in {"root", "directory"}:
                external.mkdir()
                (external / "sentinel.bin").write_bytes(b"external")
                external.chmod(0o777)
                external_member = external / "sentinel.bin"
                external_member.chmod(0o666)
                mount_target = staged.sidecar_path
                if entry_kind == "directory":
                    mount_target = staged.sidecar_path / "nested"
                    mount_target.mkdir()
            else:
                external.write_bytes(b"external")
                external.chmod(0o666)
                external_member = external
                mount_target = staged.sidecar_path / "member.bin"
                mount_target.write_bytes(b"placeholder")

            expected_mode = stat.S_IMODE(external.stat().st_mode)
            expected_member_mode = stat.S_IMODE(external_member.stat().st_mode)
            subprocess.run(
                [mount, "--bind", str(external), str(mount_target)],
                check=True,
            )
            try:
                staged_promotion_artifacts(bundle)
            except ValueError as exc:
                assert "mount point" in str(exc)
            else:
                raise AssertionError(f"{entry_kind} bind mount was not rejected")
            assert stat.S_IMODE(external.stat().st_mode) == expected_mode
            assert stat.S_IMODE(external_member.stat().st_mode) == expected_member_mode
            assert external_member.read_bytes() == b"external"
        """
    )
    completed = subprocess.run(
        [
            unshare,
            "--user",
            "--map-root-user",
            "--mount",
            "--fork",
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            mount,
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("entry_kind", ["symlink", "hardlink", "fifo"])
def test_directory_tree_digest_rejects_unsafe_entry_kinds(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    tree = tmp_path / "sidecar"
    tree.mkdir()
    entry = tree / "unsafe"
    if entry_kind == "symlink":
        entry.symlink_to(tmp_path / "outside")
        expected = "symlink"
    elif entry_kind == "hardlink":
        source = tree / "source"
        source.write_bytes(b"linked")
        os.link(source, entry)
        expected = "exactly 1 link"
    else:
        os.mkfifo(entry)
        expected = "special file"

    with pytest.raises(RuntimeError, match=expected):
        artifacts.directory_tree_sha256(tree)


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("output_path", "generated root"),
        ("diagnostics_path", "diagnostics report"),
        ("result_path", "result report"),
    ],
)
def test_staged_promotion_rejects_multiply_linked_files(
    tmp_path: Path,
    field: str,
    label: str,
) -> None:
    targets = _targets(tmp_path)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    output_metadata = bundle.staged_targets.output_path.lstat()
    artifacts._bind_staging_cleanup_identity(
        bundle,
        bundle.staged_targets.output_path,
        output_metadata,
    )
    staged_file = getattr(bundle.staged_targets, field)
    outside_alias = tmp_path / f"{field}.alias"
    os.link(staged_file, outside_alias)

    try:
        with pytest.raises(RuntimeError, match=f"{label}.*exactly one hard link"):
            staged_promotion_artifacts(bundle)
    finally:
        if field == "output_path":
            with pytest.raises(RuntimeError, match="gained additional links"):
                bundle.cleanup()
        else:
            bundle.cleanup()

    if field == "output_path":
        assert outside_alias.stat().st_nlink == 2
        assert staged_file.stat().st_nlink == 2
        staged_file.unlink()
    assert outside_alias.stat().st_nlink == 1
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()
    assert not any(tmp_path.glob(".*.stage-*"))


def test_staged_promotion_rejects_multiply_linked_sidecar_member(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path, sidecar=True)
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    staged_sidecar = bundle.staged_targets.sidecar_path
    assert staged_sidecar is not None
    member = staged_sidecar / "asset.txt"
    outside_alias = tmp_path / "request-dependency.usda"
    outside_alias.write_bytes(member.read_bytes())
    member.unlink()
    os.link(outside_alias, member)

    try:
        with pytest.raises(ValueError, match="exactly one hard link"):
            staged_promotion_artifacts(bundle)
    finally:
        bundle.cleanup()

    assert outside_alias.stat().st_nlink == 1
    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()
    assert targets.sidecar_path is not None
    assert not targets.sidecar_path.exists()
    assert not any(tmp_path.glob(".*.stage-*"))


def test_promotion_failure_restores_previous_complete_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path, sidecar=True)
    targets.output_path.write_text("old output", encoding="utf-8")
    targets.diagnostics_path.write_text("old diagnostics", encoding="utf-8")
    targets.result_path.write_text("old result", encoding="utf-8")
    assert targets.sidecar_path is not None
    targets.sidecar_path.mkdir()
    (targets.sidecar_path / "asset.txt").write_text("old sidecar", encoding="utf-8")

    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    original_replace = artifacts._replace_entry

    def fail_result_promotion(source: Any, target: Any) -> None:
        if (
            source.path == bundle.staged_targets.result_path
            and target.path == targets.result_path
        ):
            raise OSError("forced result promotion failure")
        original_replace(source, target)

    monkeypatch.setattr(artifacts, "_replace_entry", fail_result_promotion)
    try:
        with pytest.raises(OSError, match="forced result promotion failure"):
            promote_staged_artifacts(promotion)
    finally:
        bundle.cleanup()

    assert targets.output_path.read_text(encoding="utf-8") == "old output"
    assert targets.diagnostics_path.read_text(encoding="utf-8") == "old diagnostics"
    assert targets.result_path.read_text(encoding="utf-8") == "old result"
    assert (targets.sidecar_path / "asset.txt").read_text(encoding="utf-8") == (
        "old sidecar"
    )
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_parent_swap_during_promotion_does_not_touch_recreated_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_parent = tmp_path / "live"
    displaced_parent = tmp_path / "displaced-live"
    live_parent.mkdir()
    targets = _targets(live_parent, sidecar=True)
    targets.output_path.write_text("old output", encoding="utf-8")
    targets.diagnostics_path.write_text("old diagnostics", encoding="utf-8")
    targets.result_path.write_text("old result", encoding="utf-8")
    assert targets.sidecar_path is not None
    targets.sidecar_path.mkdir()
    (targets.sidecar_path / "asset.txt").write_text(
        "old sidecar",
        encoding="utf-8",
    )
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    promotion = staged_promotion_artifacts(bundle)
    original_replace = artifacts._replace_entry
    parent_swapped = False

    def swap_parent_after_first_move(source: Any, target: Any) -> None:
        nonlocal parent_swapped
        original_replace(source, target)
        if parent_swapped:
            return
        parent_swapped = True
        live_parent.rename(displaced_parent)
        live_parent.mkdir()
        unrelated_targets = _targets(live_parent, sidecar=True)
        unrelated_targets.output_path.write_text(
            "unrelated output",
            encoding="utf-8",
        )
        unrelated_targets.diagnostics_path.write_text(
            "unrelated diagnostics",
            encoding="utf-8",
        )
        unrelated_targets.result_path.write_text(
            "unrelated result",
            encoding="utf-8",
        )
        assert unrelated_targets.sidecar_path is not None
        unrelated_targets.sidecar_path.mkdir()
        (unrelated_targets.sidecar_path / "asset.txt").write_text(
            "unrelated sidecar",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        artifacts,
        "_replace_entry",
        swap_parent_after_first_move,
    )
    try:
        with pytest.raises(
            RuntimeError,
            match="Publication parent changed during transaction",
        ):
            promote_staged_artifacts(promotion)
    finally:
        bundle.cleanup()

    assert parent_swapped
    assert targets.output_path.read_text(encoding="utf-8") == "unrelated output"
    assert targets.diagnostics_path.read_text(encoding="utf-8") == (
        "unrelated diagnostics"
    )
    assert targets.result_path.read_text(encoding="utf-8") == "unrelated result"
    assert (targets.sidecar_path / "asset.txt").read_text(encoding="utf-8") == (
        "unrelated sidecar"
    )
    assert (displaced_parent / targets.output_path.name).read_text(
        encoding="utf-8"
    ) == "old output"
    assert (displaced_parent / targets.diagnostics_path.name).read_text(
        encoding="utf-8"
    ) == "old diagnostics"
    assert (displaced_parent / targets.result_path.name).read_text(
        encoding="utf-8"
    ) == "old result"
    assert (displaced_parent / targets.sidecar_path.name / "asset.txt").read_text(
        encoding="utf-8"
    ) == "old sidecar"
    assert not any(live_parent.glob(".joint-rigger-publish-*.lock"))
    assert not any(displaced_parent.glob(".joint-rigger-publish-*.lock"))
    assert not any(displaced_parent.glob(".joint-rigger.rollback-*"))


def test_parent_swap_inside_root_rename_never_reports_plain_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_parent = tmp_path / "live"
    moved_parent = tmp_path / "moved-live"
    live_parent.mkdir()
    staged = tmp_path / "staged.usda"
    target = live_parent / "target.usda"
    trusted_bytes = b"trusted generated root"
    staged.write_bytes(trusted_bytes)
    staged.chmod(0o400)
    original_replace = artifacts._replace_entry
    parent_swapped = False

    def swap_parent_at_root_commit(source: Any, destination: Any) -> None:
        nonlocal parent_swapped
        if source.name.startswith(".joint-rigger-copy-"):
            live_parent.rename(moved_parent)
            live_parent.mkdir()
            parent_swapped = True
        original_replace(source, destination)

    monkeypatch.setattr(artifacts, "_replace_entry", swap_parent_at_root_commit)
    with staged.open("rb") as source:
        with pytest.raises(CommittedArtifactPublicationCleanupError) as raised:
            promote_staged_artifacts(
                [
                    StagedArtifact(
                        staged,
                        target,
                        "generated root",
                        source_descriptor=source.fileno(),
                        source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
                    )
                ]
            )

    assert parent_swapped
    assert any(
        "Publication parent changed during transaction" in str(error)
        for error in raised.value.cleanup_errors
    )
    assert not target.exists()
    assert (moved_parent / target.name).read_bytes() == trusted_bytes


def test_successful_promotion_replaces_existing_bundle_and_removes_backups(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path, sidecar=True)
    targets.output_path.write_text("old output", encoding="utf-8")
    targets.diagnostics_path.write_text("old diagnostics", encoding="utf-8")
    targets.result_path.write_text("old result", encoding="utf-8")
    assert targets.sidecar_path is not None
    targets.sidecar_path.mkdir()
    (targets.sidecar_path / "asset.txt").write_text(
        "old sidecar",
        encoding="utf-8",
    )

    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    try:
        promote_staged_artifacts(staged_promotion_artifacts(bundle))
    finally:
        bundle.cleanup()

    assert targets.output_path.read_text(encoding="utf-8") == "new output"
    assert targets.diagnostics_path.read_text(encoding="utf-8") == "new diagnostics"
    assert targets.result_path.read_text(encoding="utf-8") == "new result"
    assert (targets.sidecar_path / "asset.txt").read_text(encoding="utf-8") == (
        "new sidecar"
    )
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_post_commit_real_close_failure_reports_committed_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path)
    targets.output_path.write_text("old output", encoding="utf-8")
    targets.diagnostics_path.write_text("old diagnostics", encoding="utf-8")
    targets.result_path.write_text("old result", encoding="utf-8")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    real_close = artifacts.os.close
    real_flock = artifacts.fcntl.flock
    close_failure_injected = False
    unlocked_descriptor: int | None = None

    def track_unlock(descriptor: int, operation: int) -> None:
        nonlocal unlocked_descriptor
        real_flock(descriptor, operation)
        if operation == fcntl.LOCK_UN and unlocked_descriptor is None:
            unlocked_descriptor = descriptor

    def close_unlocked_parent_then_fail(descriptor: int) -> None:
        nonlocal close_failure_injected
        real_close(descriptor)
        if descriptor == unlocked_descriptor and not close_failure_injected:
            close_failure_injected = True
            raise OSError(errno.EIO, "forced post-commit close failure")

    monkeypatch.setattr(artifacts.fcntl, "flock", track_unlock)
    monkeypatch.setattr(artifacts.os, "close", close_unlocked_parent_then_fail)
    try:
        with pytest.raises(CommittedArtifactPublicationCleanupError) as raised:
            promote_staged_artifacts(staged_promotion_artifacts(bundle))
    finally:
        bundle.cleanup()

    assert close_failure_injected
    assert raised.value.committed is True
    assert any(
        "forced post-commit close failure" in str(error)
        for error in raised.value.cleanup_errors
    )
    assert targets.output_path.read_text(encoding="utf-8") == "new output"
    assert targets.diagnostics_path.read_text(encoding="utf-8") == "new diagnostics"
    assert targets.result_path.read_text(encoding="utf-8") == "new result"
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_post_commit_backup_descriptor_close_failure_reports_committed_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    staged.write_bytes(b"new target")
    target.write_bytes(b"old target")
    original_create_backup = artifacts._create_artifact_backup
    real_close = artifacts.os.close
    backup_descriptor: int | None = None
    close_failure_injected = False

    def track_backup(
        bound_artifact: Any,
        *,
        artifact_identity: tuple[int, int],
    ) -> Any:
        nonlocal backup_descriptor
        backup = original_create_backup(
            bound_artifact,
            artifact_identity=artifact_identity,
        )
        backup_descriptor = backup.directory.descriptor
        return backup

    def close_backup_then_fail(descriptor: int) -> None:
        nonlocal close_failure_injected
        real_close(descriptor)
        if descriptor == backup_descriptor and not close_failure_injected:
            close_failure_injected = True
            raise OSError(errno.EIO, "forced committed backup close failure")

    monkeypatch.setattr(artifacts, "_create_artifact_backup", track_backup)
    monkeypatch.setattr(artifacts.os, "close", close_backup_then_fail)

    caller_error = ValueError("caller-owned handled exception")
    try:
        raise caller_error
    except ValueError:
        with pytest.raises(CommittedArtifactPublicationCleanupError) as raised:
            promote_staged_artifacts([StagedArtifact(staged, target, "generated root")])

    assert close_failure_injected
    assert not getattr(caller_error, "__notes__", ())
    assert raised.value.committed is True
    assert any(
        "forced committed backup close failure" in str(error)
        for error in raised.value.cleanup_errors
    )
    assert not staged.exists()
    assert target.read_bytes() == b"new target"
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_post_commit_fatal_backup_close_attempts_all_and_remains_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_evidence = tmp_path / "staged-evidence.json"
    staged_root = tmp_path / "staged-root.usda"
    target_evidence = tmp_path / "evidence.json"
    target_root = tmp_path / "root.usda"
    staged_evidence.write_bytes(b"new evidence")
    staged_root.write_bytes(b"new root")
    target_evidence.write_bytes(b"old evidence")
    target_root.write_bytes(b"old root")
    original_create_backup = artifacts._create_artifact_backup
    real_close = artifacts.os.close
    backup_descriptors: list[int] = []
    normal_error = OSError(errno.EIO, "forced earlier backup close failure")
    normal_error.add_note("forced nested close detail")
    fatal_error = SystemExit("forced fatal backup close failure")
    failed_descriptors: list[int] = []

    def track_backup(
        bound_artifact: Any,
        *,
        artifact_identity: tuple[int, int],
    ) -> Any:
        backup = original_create_backup(
            bound_artifact,
            artifact_identity=artifact_identity,
        )
        backup_descriptors.append(backup.directory.descriptor)
        return backup

    def close_backups_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        if backup_descriptors and descriptor == backup_descriptors[0]:
            failed_descriptors.append(descriptor)
            raise normal_error
        if len(backup_descriptors) > 1 and descriptor == backup_descriptors[1]:
            failed_descriptors.append(descriptor)
            raise fatal_error

    monkeypatch.setattr(artifacts, "_create_artifact_backup", track_backup)
    monkeypatch.setattr(artifacts.os, "close", close_backups_then_fail)

    with pytest.raises(SystemExit) as raised:
        promote_staged_artifacts(
            [
                StagedArtifact(
                    staged_evidence,
                    target_evidence,
                    "evidence report",
                ),
                StagedArtifact(staged_root, target_root, "generated root"),
            ]
        )

    assert raised.value is fatal_error
    assert failed_descriptors == backup_descriptors
    notes = "\n".join(raised.value.__notes__)
    assert "OSError: [Errno 5] forced earlier backup close failure" in notes
    assert "forced nested close detail" in notes
    assert not staged_evidence.exists()
    assert not staged_root.exists()
    assert target_evidence.read_bytes() == b"new evidence"
    assert target_root.read_bytes() == b"new root"
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_post_commit_backup_cleanup_fatal_remains_primary_over_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    staged.write_bytes(b"new target")
    target.write_bytes(b"old target")
    original_create_backup = artifacts._create_artifact_backup
    original_remove_backup = artifacts._remove_backup_directory
    real_close = artifacts.os.close
    backup_descriptor: int | None = None
    fatal_error = SystemExit("forced fatal backup cleanup failure")
    close_failure_injected = False

    def track_backup(
        bound_artifact: Any,
        *,
        artifact_identity: tuple[int, int],
    ) -> Any:
        nonlocal backup_descriptor
        backup = original_create_backup(
            bound_artifact,
            artifact_identity=artifact_identity,
        )
        backup_descriptor = backup.directory.descriptor
        return backup

    def remove_backup_then_terminate(backup: Any) -> None:
        original_remove_backup(backup)
        raise fatal_error

    def close_backup_then_fail(descriptor: int) -> None:
        nonlocal close_failure_injected
        real_close(descriptor)
        if descriptor == backup_descriptor and not close_failure_injected:
            close_failure_injected = True
            raise OSError("forced backup close failure during fatal cleanup")

    monkeypatch.setattr(artifacts, "_create_artifact_backup", track_backup)
    monkeypatch.setattr(
        artifacts,
        "_remove_backup_directory",
        remove_backup_then_terminate,
    )
    monkeypatch.setattr(artifacts.os, "close", close_backup_then_fail)

    with pytest.raises(SystemExit) as raised:
        promote_staged_artifacts([StagedArtifact(staged, target, "generated root")])

    assert raised.value is fatal_error
    assert close_failure_injected
    assert "OSError: forced backup close failure during fatal cleanup" in "\n".join(
        raised.value.__notes__
    )
    assert not staged.exists()
    assert target.read_bytes() == b"new target"
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_committed_operation_primary_survives_fatal_backup_cleanup_run_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_evidence = tmp_path / "staged-evidence.json"
    staged_root = tmp_path / "staged-root.usda"
    target_evidence = tmp_path / "evidence.json"
    target_root = tmp_path / "root.usda"
    staged_evidence.write_bytes(b"new evidence")
    staged_root.write_bytes(b"new root")
    target_evidence.write_bytes(b"old evidence")
    target_root.write_bytes(b"old root")
    primary_error = KeyboardInterrupt("forced post-commit operation failure")
    cleanup_error = SystemExit("forced committed backup cleanup failure")
    original_replace = artifacts._replace_entry
    original_remove_backup = artifacts._remove_backup_directory
    removal_calls: list[str] = []

    def interrupt_after_root_commit(source: Any, destination: Any) -> None:
        original_replace(source, destination)
        if source.path == staged_root and destination.path == target_root:
            raise primary_error

    def remove_all_with_first_fatal(backup: Any) -> None:
        removal_calls.append(backup.bound_artifact.artifact.label)
        original_remove_backup(backup)
        if len(removal_calls) == 1:
            raise cleanup_error

    monkeypatch.setattr(artifacts, "_replace_entry", interrupt_after_root_commit)
    monkeypatch.setattr(
        artifacts,
        "_remove_backup_directory",
        remove_all_with_first_fatal,
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        promote_staged_artifacts(
            [
                StagedArtifact(
                    staged_evidence,
                    target_evidence,
                    "evidence report",
                ),
                StagedArtifact(staged_root, target_root, "generated root"),
            ]
        )

    assert raised.value is primary_error
    assert "SystemExit: forced committed backup cleanup failure" in "\n".join(
        raised.value.__notes__
    )
    assert removal_calls == ["generated root", "evidence report"]
    assert target_evidence.read_bytes() == b"new evidence"
    assert target_root.read_bytes() == b"new root"
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_successful_commit_fatal_backup_cleanup_attempts_every_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_evidence = tmp_path / "staged-evidence.json"
    staged_root = tmp_path / "staged-root.usda"
    target_evidence = tmp_path / "evidence.json"
    target_root = tmp_path / "root.usda"
    staged_evidence.write_bytes(b"new evidence")
    staged_root.write_bytes(b"new root")
    target_evidence.write_bytes(b"old evidence")
    target_root.write_bytes(b"old root")
    cleanup_error = SystemExit("forced first committed backup cleanup failure")
    original_remove_backup = artifacts._remove_backup_directory
    removal_calls: list[str] = []

    def remove_all_with_first_fatal(backup: Any) -> None:
        removal_calls.append(backup.bound_artifact.artifact.label)
        original_remove_backup(backup)
        if len(removal_calls) == 1:
            raise cleanup_error

    monkeypatch.setattr(
        artifacts,
        "_remove_backup_directory",
        remove_all_with_first_fatal,
    )

    with pytest.raises(SystemExit) as raised:
        promote_staged_artifacts(
            [
                StagedArtifact(
                    staged_evidence,
                    target_evidence,
                    "evidence report",
                ),
                StagedArtifact(staged_root, target_root, "generated root"),
            ]
        )

    assert raised.value is cleanup_error
    assert removal_calls == ["generated root", "evidence report"]
    assert target_evidence.read_bytes() == b"new evidence"
    assert target_root.read_bytes() == b"new root"
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_post_commit_backup_cleanup_failure_reports_committed_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(tmp_path)
    targets.output_path.write_text("old output", encoding="utf-8")
    targets.diagnostics_path.write_text("old diagnostics", encoding="utf-8")
    targets.result_path.write_text("old result", encoding="utf-8")
    bundle = create_staged_artifact_targets(targets)
    _write_staged_bundle(bundle.staged_targets)
    original_remove_backup = artifacts._remove_backup_directory
    cleanup_failure_injected = False

    def remove_then_fail(backup: Any) -> None:
        nonlocal cleanup_failure_injected
        original_remove_backup(backup)
        if not cleanup_failure_injected:
            cleanup_failure_injected = True
            raise OSError("forced committed backup cleanup failure")

    monkeypatch.setattr(artifacts, "_remove_backup_directory", remove_then_fail)
    try:
        with pytest.raises(CommittedArtifactPublicationCleanupError) as raised:
            promote_staged_artifacts(staged_promotion_artifacts(bundle))
    finally:
        bundle.cleanup()

    assert cleanup_failure_injected
    assert raised.value.committed is True
    assert targets.output_path.read_text(encoding="utf-8") == "new output"
    assert targets.diagnostics_path.read_text(encoding="utf-8") == "new diagnostics"
    assert targets.result_path.read_text(encoding="utf-8") == "new result"
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_incomplete_staged_bundle_cannot_publish_root(tmp_path: Path) -> None:
    targets = _targets(tmp_path)
    bundle = create_staged_artifact_targets(targets)
    bundle.staged_targets.output_path.write_text("new output", encoding="utf-8")
    bundle.staged_targets.diagnostics_path.write_text(
        "new diagnostics",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="result report is missing") as raised:
        staged_promotion_artifacts(bundle)
    artifacts._cleanup_staging_reservations(
        bundle._cleanup_reservations,
        primary_error=raised.value,
    )

    assert not targets.output_path.exists()
    assert bundle.staged_targets.output_path.exists()
    assert "no descriptor-bound cleanup identity" in "\n".join(raised.value.__notes__)
    bundle.staged_targets.output_path.unlink()


def test_publication_parent_locks_dedupe_and_sort_physical_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_parent = tmp_path / "z-parent"
    second_parent = tmp_path / "a-parent"
    first_parent.mkdir()
    second_parent.mkdir()
    expected_identities = sorted(
        {
            (first_parent.stat().st_dev, first_parent.stat().st_ino),
            (second_parent.stat().st_dev, second_parent.stat().st_ino),
        }
    )
    acquired_identities: list[tuple[int, int]] = []
    real_flock = artifacts.fcntl.flock

    def record_flock(descriptor: int, operation: int) -> None:
        if operation == fcntl.LOCK_EX | fcntl.LOCK_NB:
            metadata = os.fstat(descriptor)
            acquired_identities.append((metadata.st_dev, metadata.st_ino))
        real_flock(descriptor, operation)

    monkeypatch.setattr(artifacts.fcntl, "flock", record_flock)

    with artifacts._publication_target_locks(
        [
            second_parent / "result.json",
            first_parent / "output.usda",
            second_parent / "diagnostics.json",
        ]
    ):
        pass

    assert acquired_identities == expected_identities


def test_duplicate_physical_targets_fail_before_lock_or_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_parent = tmp_path / "first-mount"
    second_parent = tmp_path / "second-mount"
    staging_parent = tmp_path / "staging"
    first_parent.mkdir()
    second_parent.mkdir()
    staging_parent.mkdir()
    first_target = first_parent / "result.json"
    second_target = second_parent / "result.json"
    first_target.write_text("old first", encoding="utf-8")
    second_target.write_text("old second", encoding="utf-8")
    first_staged = staging_parent / "first.json"
    second_staged = staging_parent / "second.json"
    first_staged.write_text("new first", encoding="utf-8")
    second_staged.write_text("new second", encoding="utf-8")
    aliased_parents = {first_parent.resolve(), second_parent.resolve()}
    original_identity = artifacts._physical_directory_identity

    def shared_target_parent_identity(
        parent: Path,
        *,
        descriptor: int | None = None,
    ) -> tuple[int, int]:
        if parent in aliased_parents:
            return 123, 456
        return original_identity(parent, descriptor=descriptor)

    monkeypatch.setattr(
        artifacts,
        "_physical_directory_identity",
        shared_target_parent_identity,
    )
    promotion = [
        StagedArtifact(first_staged, first_target, "first report"),
        StagedArtifact(second_staged, second_target, "second report"),
    ]

    with pytest.raises(ValueError, match="Duplicate physical transaction target"):
        promote_staged_artifacts(promotion)

    assert first_staged.read_text(encoding="utf-8") == "new first"
    assert second_staged.read_text(encoding="utf-8") == "new second"
    assert first_target.read_text(encoding="utf-8") == "old first"
    assert second_target.read_text(encoding="utf-8") == "old second"
    assert not any(first_parent.glob(".joint-rigger-publish-*.lock"))
    assert not any(second_parent.glob(".joint-rigger-publish-*.lock"))


def test_concurrent_successful_alias_promotions_reject_then_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    first_targets = _targets(real_parent, sidecar=True)
    second_targets = _targets(alias_parent, sidecar=True)
    first = create_staged_artifact_targets(first_targets)
    second = create_staged_artifact_targets(second_targets)
    _write_staged_bundle(first.staged_targets, marker="first")
    _write_staged_bundle(second.staged_targets, marker="second")
    first_promotion = staged_promotion_artifacts(first)
    second_promotion = staged_promotion_artifacts(second)
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    original_replace = artifacts._replace_entry

    def pause_first_after_one_replacement(source: Any, target: Any) -> None:
        original_replace(source, target)
        if source.path == first.staged_targets.diagnostics_path:
            entered.set()
            if not release.wait(timeout=15):
                raise TimeoutError("test did not release first promotion")

    def publish_first() -> None:
        try:
            promote_staged_artifacts(first_promotion)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    monkeypatch.setattr(
        artifacts,
        "_replace_entry",
        pause_first_after_one_replacement,
    )
    thread = threading.Thread(target=publish_first, name="first-publication")
    thread.start()
    try:
        assert entered.wait(timeout=15)
        with pytest.raises(
            ConcurrentArtifactPublicationError,
            match="already targeting",
        ):
            promote_staged_artifacts(second_promotion)
    finally:
        release.set()
        thread.join(timeout=15)
        first.cleanup()

    assert not thread.is_alive()
    assert errors == []
    assert first_targets.output_path.read_text(encoding="utf-8") == "first output"

    # Rejection occurs before any staged file moves, but the successful first
    # transaction invalidates the second bundle's captured absent-target state.
    # A retry must acquire a fresh staging reservation against the now-current
    # target identities instead of legitimizing drift in a stale transaction.
    try:
        with pytest.raises(
            RuntimeError,
            match="Artifact target changed after staged targets were created",
        ):
            promote_staged_artifacts(second_promotion)
    finally:
        second.cleanup()

    retry = create_staged_artifact_targets(second_targets)
    _write_staged_bundle(retry.staged_targets, marker="second")
    try:
        promote_staged_artifacts(staged_promotion_artifacts(retry))
    finally:
        retry.cleanup()
    assert first_targets.output_path.read_text(encoding="utf-8") == "second output"
    assert first_targets.diagnostics_path.read_text(encoding="utf-8") == (
        "second diagnostics"
    )
    assert first_targets.result_path.read_text(encoding="utf-8") == "second result"
    assert first_targets.sidecar_path is not None
    assert (first_targets.sidecar_path / "asset.txt").read_text(
        encoding="utf-8"
    ) == "second sidecar"


def test_concurrent_process_failure_after_one_replacement_restores_success(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path, sidecar=True)
    targets.output_path.write_text("previous output", encoding="utf-8")
    targets.diagnostics_path.write_text("previous diagnostics", encoding="utf-8")
    targets.result_path.write_text("previous result", encoding="utf-8")
    assert targets.sidecar_path is not None
    targets.sidecar_path.mkdir()
    (targets.sidecar_path / "asset.txt").write_text(
        "previous sidecar",
        encoding="utf-8",
    )
    failing = create_staged_artifact_targets(targets)
    succeeding = create_staged_artifact_targets(targets)
    _write_staged_bundle(failing.staged_targets, marker="failed")
    _write_staged_bundle(succeeding.staged_targets, marker="success")
    failing_promotion = staged_promotion_artifacts(failing)
    succeeding_promotion = staged_promotion_artifacts(succeeding)
    raw_promotion = [
        (str(item.staged_path), str(item.target_path), item.label)
        for item in failing_promotion
    ]
    # The supported runtime is Linux/WSL2, where ``fork`` also avoids requiring
    # the non-package ``tests/`` directory to be importable in the child.
    context = multiprocessing.get_context("fork")
    entered = context.Event()
    release = context.Event()
    outcome = context.Queue()
    process = context.Process(
        target=_controlled_promotion_process,
        args=(
            raw_promotion,
            str(failing.staged_targets.diagnostics_path),
            str(failing.staged_targets.result_path),
            entered,
            release,
            outcome,
        ),
    )
    process.start()
    try:
        assert entered.wait(timeout=30)
        with pytest.raises(
            ConcurrentArtifactPublicationError,
            match="already targeting",
        ):
            promote_staged_artifacts(succeeding_promotion)
    finally:
        release.set()
        process.join(timeout=30)

    assert not process.is_alive()
    assert process.exitcode == 0
    assert outcome.get(timeout=5) == (
        "OSError",
        "forced concurrent promotion failure",
    )
    assert targets.output_path.read_text(encoding="utf-8") == "previous output"
    assert targets.diagnostics_path.read_text(encoding="utf-8") == (
        "previous diagnostics"
    )
    assert targets.result_path.read_text(encoding="utf-8") == "previous result"
    assert (targets.sidecar_path / "asset.txt").read_text(encoding="utf-8") == (
        "previous sidecar"
    )

    # The failed process released every kernel lock and restored the old bytes,
    # but moving those inodes through rollback changed their captured state.
    # A stale transaction must fail closed and acquire fresh target snapshots.
    try:
        with pytest.raises(
            RuntimeError,
            match="Artifact target changed after staged targets were created",
        ):
            promote_staged_artifacts(succeeding_promotion)
    finally:
        failing.cleanup()
        succeeding.cleanup()
        outcome.close()
        outcome.join_thread()

    fresh = create_staged_artifact_targets(targets)
    _write_staged_bundle(fresh.staged_targets, marker="success")
    try:
        promote_staged_artifacts(staged_promotion_artifacts(fresh))
    finally:
        fresh.cleanup()
    assert targets.output_path.read_text(encoding="utf-8") == "success output"
    assert targets.diagnostics_path.read_text(encoding="utf-8") == (
        "success diagnostics"
    )
    assert targets.result_path.read_text(encoding="utf-8") == "success result"
    assert (targets.sidecar_path / "asset.txt").read_text(encoding="utf-8") == (
        "success sidecar"
    )
    assert not any(tmp_path.glob(".*.rollback-*"))


def test_distinct_target_promotions_can_run_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_targets = _targets(tmp_path / "first", sidecar=True)
    second_targets = _targets(tmp_path / "second", sidecar=True)
    first = create_staged_artifact_targets(first_targets)
    second = create_staged_artifact_targets(second_targets)
    _write_staged_bundle(first.staged_targets, marker="first")
    _write_staged_bundle(second.staged_targets, marker="second")
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    original_replace = artifacts._replace_entry

    def pause_first_after_one_replacement(source: Any, target: Any) -> None:
        original_replace(source, target)
        if source.path == first.staged_targets.diagnostics_path:
            entered.set()
            if not release.wait(timeout=15):
                raise TimeoutError("test did not release first promotion")

    def publish_first() -> None:
        try:
            promote_staged_artifacts(staged_promotion_artifacts(first))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    monkeypatch.setattr(
        artifacts,
        "_replace_entry",
        pause_first_after_one_replacement,
    )
    thread = threading.Thread(target=publish_first, name="first-publication")
    thread.start()
    try:
        assert entered.wait(timeout=15)
        # This completes while the first transaction is deliberately paused.
        promote_staged_artifacts(staged_promotion_artifacts(second))
        assert second_targets.output_path.read_text(encoding="utf-8") == (
            "second output"
        )
    finally:
        release.set()
        thread.join(timeout=15)
        first.cleanup()
        second.cleanup()

    assert not thread.is_alive()
    assert errors == []
    assert first_targets.output_path.read_text(encoding="utf-8") == "first output"


def test_same_parent_disjoint_target_sets_share_a_lock(tmp_path: Path) -> None:
    targets = [tmp_path / name for name in ("a.json", "b.json", "c.json")]
    with artifacts._publication_target_locks(targets[:1]):
        with pytest.raises(
            ConcurrentArtifactPublicationError,
            match="already targeting parent",
        ):
            with artifacts._publication_target_locks(targets[1:]):
                pytest.fail("same-parent target locks must not be acquired")

    with artifacts._publication_target_locks(targets[1:]):
        pass


def test_process_crash_releases_publication_locks_for_reuse(tmp_path: Path) -> None:
    targets = [tmp_path / "diagnostics.json", tmp_path / "result.json"]
    context = multiprocessing.get_context("fork")
    acquired = context.Event()
    process = context.Process(
        target=_crash_while_holding_publication_locks,
        args=([str(target) for target in targets], acquired),
    )
    process.start()
    assert acquired.wait(timeout=15)
    process.join(timeout=15)

    assert not process.is_alive()
    assert process.exitcode == 23
    assert not any(tmp_path.glob(".joint-rigger-publish-*.lock"))
    # Process-owned directory locks disappear on exit, so the next transaction
    # acquires the same physical parent without stale recovery.
    with artifacts._publication_target_locks(targets):
        pass


def test_hostile_lock_entry_is_ignored_but_parent_alias_contends(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    outside = tmp_path / "outside"
    outside.write_text("protected", encoding="utf-8")
    hostile_entry = real_parent / ".joint-rigger-publish-hostile.lock"
    hostile_entry.symlink_to(outside)

    with artifacts._publication_target_locks([real_parent / "first.json"]):
        with pytest.raises(
            ConcurrentArtifactPublicationError,
            match="already targeting parent",
        ):
            with artifacts._publication_target_locks([alias_parent / "different.json"]):
                pytest.fail("physical parent aliases must contend")

    assert hostile_entry.is_symlink()
    assert outside.read_text(encoding="utf-8") == "protected"
    with artifacts._publication_target_locks([alias_parent / "different.json"]):
        pass


def test_repeated_random_staging_publications_create_no_lock_files(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path, sidecar=True)

    for iteration in range(12):
        bundle = create_staged_artifact_targets(targets)
        marker = f"iteration-{iteration}"
        _write_staged_bundle(bundle.staged_targets, marker=marker)
        try:
            promote_staged_artifacts(staged_promotion_artifacts(bundle))
        finally:
            bundle.cleanup()
        assert targets.output_path.read_text(encoding="utf-8") == f"{marker} output"

    assert not any(tmp_path.rglob(".joint-rigger-publish-*.lock"))
    assert not any(tmp_path.rglob(".*.stage-*"))


def test_single_context_cleanup_error_is_raised_directly() -> None:
    expected = OSError("single cleanup failure")

    with pytest.raises(OSError) as raised:
        artifacts._route_context_cleanup_errors(
            [expected],
            None,
            label="unused aggregate label",
        )

    assert raised.value is expected


def test_committed_standalone_fatal_includes_recorded_and_peer_cleanup_errors() -> None:
    cleanup_state = artifacts._PublicationCleanupState(committed=True)
    recorded_error = OSError("recorded committed cleanup failure")
    cleanup_state.errors.append(recorded_error)
    fatal_error = SystemExit("standalone cleanup fatal")
    peer_error = OSError("peer cleanup failure")

    with pytest.raises(SystemExit) as raised:
        artifacts._route_cleanup_failures(
            [("fatal cleanup", fatal_error), ("peer cleanup", peer_error)],
            cleanup_state=cleanup_state,
            label="standalone cleanup failed",
        )

    assert raised.value is fatal_error
    assert cleanup_state.errors == []
    notes = "\n".join(raised.value.__notes__)
    assert "recorded committed cleanup failure" in notes
    assert "peer cleanup failure" in notes


def test_unbound_staging_reservation_entry_cleanup_is_noop(tmp_path: Path) -> None:
    parent = artifacts._open_bound_directory(tmp_path)
    reservation = artifacts._StagingCleanupReservation(
        parent=parent,
        name="absent.stage",
    )
    try:
        artifacts._remove_staging_reservation_entry(reservation, parent)
    finally:
        os.close(parent.descriptor)


def test_publication_contexts_keep_active_fatal_and_attempt_every_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_paths: list[Path] = []
    target_paths: list[Path] = []
    for index in range(2):
        staged_parent = tmp_path / f"staged-{index}"
        target_parent = tmp_path / f"target-{index}"
        staged_parent.mkdir()
        target_parent.mkdir()
        staged_path = staged_parent / "artifact"
        staged_path.write_text(f"staged-{index}", encoding="utf-8")
        staged_paths.append(staged_path)
        target_paths.append(target_parent / "artifact")

    original_open = artifacts._open_bound_directory
    original_close = artifacts.os.close
    original_flock = artifacts.fcntl.flock
    staged_descriptors: list[int] = []
    target_descriptors: list[int] = []
    close_attempts: list[int] = []
    unlock_attempts: list[int] = []
    primary_error = KeyboardInterrupt("forced publication body fatal")
    staged_close_error = OSError("forced staged descriptor close failure")
    target_close_error = SystemExit("forced target descriptor close fatal")
    unlock_error = OSError("forced publication unlock failure")
    unlock_error.add_note("nested unlock detail")
    unlock_fatal = SystemExit("forced publication unlock fatal")

    def track_open(path: Path) -> Any:
        directory = original_open(path)
        if path.name.startswith("staged-"):
            staged_descriptors.append(directory.descriptor)
        elif path.name.startswith("target-"):
            target_descriptors.append(directory.descriptor)
        return directory

    def close_with_failures(descriptor: int) -> None:
        close_attempts.append(descriptor)
        original_close(descriptor)
        if staged_descriptors and descriptor == staged_descriptors[0]:
            raise staged_close_error
        if target_descriptors and descriptor == target_descriptors[0]:
            raise target_close_error

    def flock_with_failures(descriptor: int, operation: int) -> None:
        original_flock(descriptor, operation)
        if operation != fcntl.LOCK_UN:
            return
        unlock_attempts.append(descriptor)
        if len(unlock_attempts) == 1:
            raise unlock_error
        if len(unlock_attempts) == 2:
            raise unlock_fatal

    def interrupt_before_backup() -> None:
        raise primary_error

    monkeypatch.setattr(artifacts, "_open_bound_directory", track_open)
    monkeypatch.setattr(artifacts.os, "close", close_with_failures)
    monkeypatch.setattr(artifacts.fcntl, "flock", flock_with_failures)

    with pytest.raises(KeyboardInterrupt) as raised:
        promote_staged_artifacts(
            [
                StagedArtifact(staged, target, f"artifact-{index}")
                for index, (staged, target) in enumerate(
                    zip(staged_paths, target_paths, strict=True)
                )
            ],
            prebackup_validator=interrupt_before_backup,
        )

    assert raised.value is primary_error
    assert len(unlock_attempts) == 2
    # ``artifacts.os`` is the process-wide ``os`` module, so the monkeypatch
    # can also observe unrelated background-thread closes during a full-suite
    # run. Require every descriptor owned by this transaction without making
    # those unrelated closes a test failure.
    owned_descriptors = set(staged_descriptors + target_descriptors)
    observed_owned_descriptors = {
        descriptor for descriptor in close_attempts if descriptor in owned_descriptors
    }
    assert observed_owned_descriptors == owned_descriptors
    notes = "\n".join(raised.value.__notes__)
    assert "forced staged descriptor close failure" in notes
    assert "forced target descriptor close fatal" in notes
    assert "forced publication unlock failure" in notes
    assert "nested unlock detail" in notes
    assert "forced publication unlock fatal" in notes
    for descriptor in staged_descriptors + target_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_postcommit_fatal_includes_previously_recorded_cleanup_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    cleanup_state = artifacts._PublicationCleanupState(committed=True)
    recorded_error = OSError("previous committed cleanup failure")
    recorded_error.add_note("recorded nested detail")
    cleanup_state.errors.append(recorded_error)
    primary_error = SystemExit("forced postcommit fatal")
    original_close = artifacts.os.close
    close_attempts: list[int] = []

    def track_close(descriptor: int) -> None:
        close_attempts.append(descriptor)
        original_close(descriptor)

    monkeypatch.setattr(artifacts.os, "close", track_close)

    with pytest.raises(SystemExit) as raised:
        with artifacts._bound_publication_targets(
            [target],
            cleanup_state=cleanup_state,
        ):
            raise primary_error

    assert raised.value is primary_error
    assert len(close_attempts) == 1
    assert cleanup_state.errors == []
    notes = "\n".join(raised.value.__notes__)
    assert "previous committed cleanup failure" in notes
    assert "recorded nested detail" in notes


def test_bound_target_setup_fatal_closes_parent_without_replacing_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_error = KeyboardInterrupt("forced target identity fatal")
    close_error = OSError("forced target setup close failure")
    original_open = artifacts._open_bound_directory
    original_close = artifacts.os.close
    opened_descriptor: int | None = None

    def track_open(path: Path) -> Any:
        nonlocal opened_descriptor
        directory = original_open(path)
        opened_descriptor = directory.descriptor
        return directory

    def fail_identity(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise primary_error

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        if descriptor == opened_descriptor:
            raise close_error

    monkeypatch.setattr(artifacts, "_open_bound_directory", track_open)
    monkeypatch.setattr(artifacts, "_physical_entry_identity", fail_identity)
    monkeypatch.setattr(artifacts.os, "close", close_then_fail)

    with pytest.raises(KeyboardInterrupt) as raised:
        with artifacts._bound_publication_targets([tmp_path / "target"]):
            pytest.fail("target binding must fail before yielding")

    assert raised.value is primary_error
    assert opened_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(opened_descriptor)
    assert "forced target setup close failure" in "\n".join(raised.value.__notes__)


def test_legacy_staging_bundle_cleanup_is_complete_and_idempotent(
    tmp_path: Path,
) -> None:
    final_targets = _targets(tmp_path / "final", sidecar=True)
    owner = tmp_path / "legacy-owner"
    staged_targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "legacy-output.usda",
        diagnostics_path=tmp_path / "legacy-diagnostics.json",
        result_path=tmp_path / "legacy-result.json",
        sidecar_path=owner / "legacy-sidecar",
    )
    staged_targets.output_path.write_text("staged output", encoding="utf-8")
    staged_targets.diagnostics_path.write_text(
        "staged diagnostics",
        encoding="utf-8",
    )
    staged_targets.result_path.write_text("staged result", encoding="utf-8")
    assert staged_targets.sidecar_path is not None
    staged_targets.sidecar_path.mkdir(parents=True)
    (staged_targets.sidecar_path / "member.txt").write_text(
        "staged sidecar",
        encoding="utf-8",
    )
    bundle = artifacts.StagedJointRiggerArtifacts(
        final_targets=final_targets,
        staged_targets=staged_targets,
        sidecar_owner_path=owner,
    )

    bundle.cleanup()
    bundle.cleanup()

    assert not any(path.exists() for path in artifacts._target_paths(staged_targets))
    assert not owner.exists()


def test_legacy_staging_cleanup_baseexception_attempts_every_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_targets = _targets(tmp_path / "final", sidecar=True)
    owner = tmp_path / "legacy-owner"
    staged_targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "legacy-output.usda",
        diagnostics_path=tmp_path / "legacy-diagnostics.json",
        result_path=tmp_path / "legacy-result.json",
        sidecar_path=owner / "legacy-sidecar",
    )
    staged_targets.output_path.write_text("output", encoding="utf-8")
    staged_targets.diagnostics_path.write_text("diagnostics", encoding="utf-8")
    staged_targets.result_path.write_text("result", encoding="utf-8")
    assert staged_targets.sidecar_path is not None
    staged_targets.sidecar_path.mkdir(parents=True)
    (staged_targets.sidecar_path / "member.txt").write_text(
        "sidecar",
        encoding="utf-8",
    )
    bundle = artifacts.StagedJointRiggerArtifacts(
        final_targets=final_targets,
        staged_targets=staged_targets,
        sidecar_owner_path=owner,
    )
    original_remove = artifacts.remove_artifact
    fatal_error = KeyboardInterrupt("forced legacy staging cleanup fatal")
    removed_paths: list[Path] = []

    def remove_then_interrupt(path: Path) -> None:
        original_remove(path)
        removed_paths.append(path)
        if len(removed_paths) == 1:
            raise fatal_error

    monkeypatch.setattr(artifacts, "remove_artifact", remove_then_interrupt)

    with pytest.raises(KeyboardInterrupt) as raised:
        bundle.cleanup()

    assert raised.value is fatal_error
    assert removed_paths == [*artifacts._target_paths(staged_targets), owner]
    assert not any(path.exists() for path in artifacts._target_paths(staged_targets))
    assert not owner.exists()


def test_noncommit_descriptor_file_is_revalidated_before_root_commit(
    tmp_path: Path,
) -> None:
    staged_evidence = tmp_path / "staged-evidence.json"
    target_evidence = tmp_path / "evidence.json"
    staged_root = tmp_path / "staged-root.usda"
    target_root = tmp_path / "root.usda"
    evidence = b"trusted descriptor evidence"
    staged_evidence.write_bytes(evidence)
    staged_evidence.chmod(0o400)
    staged_root.write_bytes(b"new root")

    with staged_evidence.open("rb") as source:
        promote_staged_artifacts(
            [
                StagedArtifact(
                    staged_evidence,
                    target_evidence,
                    "descriptor evidence",
                    source_descriptor=source.fileno(),
                    source_sha256=hashlib.sha256(evidence).hexdigest(),
                ),
                StagedArtifact(staged_root, target_root, "generated root"),
            ]
        )

    assert target_evidence.read_bytes() == evidence
    assert target_root.read_bytes() == b"new root"
    assert not staged_evidence.exists()


def test_precommit_failure_remains_primary_when_backup_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    staged.write_bytes(b"new target")
    target.write_bytes(b"old target")
    original_create_backup = artifacts._create_artifact_backup
    original_replace = artifacts._replace_entry
    real_close = artifacts.os.close
    primary_error = RuntimeError("forced precommit promotion failure")
    backup_descriptor: int | None = None
    close_failure_injected = False

    def track_backup(
        bound_artifact: Any,
        *,
        artifact_identity: tuple[int, int],
    ) -> Any:
        nonlocal backup_descriptor
        backup = original_create_backup(
            bound_artifact,
            artifact_identity=artifact_identity,
        )
        backup_descriptor = backup.directory.descriptor
        return backup

    def fail_staged_promotion(source: Any, destination: Any) -> None:
        if source.path == staged and destination.path == target:
            raise primary_error
        original_replace(source, destination)

    def close_backup_then_fail(descriptor: int) -> None:
        nonlocal close_failure_injected
        real_close(descriptor)
        if descriptor == backup_descriptor and not close_failure_injected:
            close_failure_injected = True
            raise OSError(errno.EIO, "forced backup descriptor close failure")

    monkeypatch.setattr(artifacts, "_create_artifact_backup", track_backup)
    monkeypatch.setattr(artifacts, "_replace_entry", fail_staged_promotion)
    monkeypatch.setattr(artifacts.os, "close", close_backup_then_fail)

    with pytest.raises(RuntimeError) as raised:
        promote_staged_artifacts([StagedArtifact(staged, target, "generated root")])

    assert raised.value is primary_error
    assert close_failure_injected
    assert "OSError: [Errno 5] forced backup descriptor close failure" in "\n".join(
        raised.value.__notes__
    )
    assert target.read_bytes() == b"old target"
    assert staged.read_bytes() == b"new target"
    assert not any(tmp_path.glob(".joint-rigger.rollback-*"))


def test_bound_directory_open_failure_closes_new_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RuntimeError("forced bound-directory identity failure")
    real_open = artifacts.os.open
    opened_descriptor: int | None = None

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal opened_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened_descriptor = descriptor
        return descriptor

    def fail_identity(*args: object, **kwargs: object) -> tuple[int, int]:
        del args, kwargs
        raise expected

    monkeypatch.setattr(artifacts.os, "open", track_open)
    monkeypatch.setattr(artifacts, "_physical_directory_identity", fail_identity)

    with pytest.raises(RuntimeError) as raised:
        artifacts._open_bound_directory(tmp_path)

    assert raised.value is expected
    assert opened_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(opened_descriptor)


def test_bound_directory_open_fatal_keeps_primary_when_close_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_error = KeyboardInterrupt("forced bound-directory identity fatal")
    close_error = OSError("forced bound-directory close failure")
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    opened_descriptor: int | None = None

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal opened_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened_descriptor = descriptor
        return descriptor

    def fail_identity(*args: object, **kwargs: object) -> tuple[int, int]:
        del args, kwargs
        raise primary_error

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == opened_descriptor:
            raise close_error

    monkeypatch.setattr(artifacts.os, "open", track_open)
    monkeypatch.setattr(artifacts.os, "close", close_then_fail)
    monkeypatch.setattr(artifacts, "_physical_directory_identity", fail_identity)

    with pytest.raises(KeyboardInterrupt) as raised:
        artifacts._open_bound_directory(tmp_path)

    assert raised.value is primary_error
    assert opened_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(opened_descriptor)
    assert "forced bound-directory close failure" in "\n".join(raised.value.__notes__)


def test_bound_child_directory_open_failure_closes_child_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    parent = artifacts._open_bound_directory(tmp_path)
    expected = RuntimeError("forced child identity failure")
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    child_descriptor: int | None = None

    def track_child_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal child_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd == parent.descriptor:
            child_descriptor = descriptor
        return descriptor

    def fail_identity(*args: object, **kwargs: object) -> tuple[int, int]:
        del args, kwargs
        raise expected

    monkeypatch.setattr(artifacts.os, "open", track_child_open)
    monkeypatch.setattr(artifacts, "_physical_directory_identity", fail_identity)
    try:
        with pytest.raises(RuntimeError) as raised:
            artifacts._open_bound_child_directory(parent, child.name)
    finally:
        real_close(parent.descriptor)

    assert raised.value is expected
    assert child_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(child_descriptor)


def test_bound_child_directory_open_binds_real_child_identity(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    parent = artifacts._open_bound_directory(tmp_path)
    bound_child: Any | None = None
    try:
        bound_child = artifacts._open_bound_child_directory(parent, child.name)
        metadata = os.fstat(bound_child.descriptor)
        assert bound_child.locator_path == child
        assert bound_child.opened_path == child
        assert bound_child.identity == (metadata.st_dev, metadata.st_ino)
    finally:
        if bound_child is not None:
            os.close(bound_child.descriptor)
        os.close(parent.descriptor)


def test_directory_copy_target_child_open_failure_closes_source_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source-tree"
    target = tmp_path / "target-tree"
    (source / "nested").mkdir(parents=True)
    target.mkdir()
    _seal_directory_tree(source)
    source_descriptor = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    target_descriptor = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    source_child_descriptor: int | None = None

    def fail_target_child_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal source_child_descriptor
        if dir_fd == target_descriptor and path == "nested":
            raise OSError(errno.EIO, "forced target child open failure")
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd == source_descriptor and path == "nested":
            source_child_descriptor = descriptor
        return descriptor

    monkeypatch.setattr(artifacts.os, "open", fail_target_child_open)
    try:
        with pytest.raises(OSError, match="forced target child open failure"):
            artifacts._copy_directory_descriptor_tree(
                source_descriptor,
                target_descriptor,
                label="sidecar",
            )
    finally:
        real_close(target_descriptor)
        real_close(source_descriptor)

    assert source_child_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(source_child_descriptor)


def test_directory_copy_reports_single_child_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source-tree"
    target = tmp_path / "target-tree"
    (source / "nested").mkdir(parents=True)
    target.mkdir()
    _seal_directory_tree(source)
    source_descriptor = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    target_descriptor = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    source_child_descriptor: int | None = None
    close_failure_injected = False

    def track_source_child(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal source_child_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd == source_descriptor and path == "nested":
            source_child_descriptor = descriptor
        return descriptor

    def close_source_child_then_fail(descriptor: int) -> None:
        nonlocal close_failure_injected
        real_close(descriptor)
        if descriptor == source_child_descriptor and not close_failure_injected:
            close_failure_injected = True
            raise OSError(errno.EIO, "forced directory child close failure")

    monkeypatch.setattr(artifacts.os, "open", track_source_child)
    monkeypatch.setattr(artifacts.os, "close", close_source_child_then_fail)
    try:
        with pytest.raises(OSError, match="forced directory child close failure"):
            artifacts._copy_directory_descriptor_tree(
                source_descriptor,
                target_descriptor,
                label="sidecar",
            )
    finally:
        real_close(target_descriptor)
        real_close(source_descriptor)

    assert close_failure_injected


def test_directory_file_copy_reports_single_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source-tree"
    target = tmp_path / "target-tree"
    source.mkdir()
    (source / "member.bin").write_bytes(b"sealed member")
    target.mkdir()
    _seal_directory_tree(source)
    source_descriptor = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    target_descriptor = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    target_file_descriptor: int | None = None
    close_failure_injected = False

    def track_target_file(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal target_file_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd == target_descriptor and path == "member.bin":
            target_file_descriptor = descriptor
        return descriptor

    def close_target_file_then_fail(descriptor: int) -> None:
        nonlocal close_failure_injected
        real_close(descriptor)
        if descriptor == target_file_descriptor and not close_failure_injected:
            close_failure_injected = True
            raise OSError(errno.EIO, "forced copied-file close failure")

    monkeypatch.setattr(artifacts.os, "open", track_target_file)
    monkeypatch.setattr(artifacts.os, "close", close_target_file_then_fail)
    try:
        with pytest.raises(OSError, match="forced copied-file close failure"):
            artifacts._copy_directory_descriptor_tree(
                source_descriptor,
                target_descriptor,
                label="sidecar",
            )
    finally:
        real_close(target_descriptor)
        real_close(source_descriptor)

    assert close_failure_injected


def test_directory_file_copy_target_open_failure_closes_source_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source-tree"
    target = tmp_path / "target-tree"
    source.mkdir()
    member = source / "member.bin"
    member.write_bytes(b"sealed member")
    target.mkdir()
    _seal_directory_tree(source)
    source_descriptor = os.open(
        source,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    target_descriptor = os.open(
        target,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    source_file_descriptor: int | None = None

    def fail_target_file_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal source_file_descriptor
        if dir_fd == target_descriptor and path == member.name:
            raise OSError(errno.EIO, "forced target file open failure")
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd == source_descriptor and path == member.name:
            source_file_descriptor = descriptor
        return descriptor

    monkeypatch.setattr(artifacts.os, "open", fail_target_file_open)
    try:
        with pytest.raises(OSError, match="forced target file open failure"):
            artifacts._copy_directory_descriptor_tree(
                source_descriptor,
                target_descriptor,
                label="sidecar",
            )
    finally:
        real_close(target_descriptor)
        real_close(source_descriptor)

    assert source_file_descriptor is not None
    _assert_descriptor_is_closed(source_file_descriptor)
    assert list(target.iterdir()) == []


def test_write_all_retries_interrupted_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "destination.bin"
    descriptor = os.open(destination, os.O_CREAT | os.O_WRONLY, 0o600)
    real_write = artifacts.os.write
    attempts = 0

    def interrupt_once(target_descriptor: int, content: Any) -> int:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InterruptedError
        return real_write(target_descriptor, content)

    monkeypatch.setattr(artifacts.os, "write", interrupt_once)
    try:
        artifacts._write_all(descriptor, b"complete payload", label="payload")
    finally:
        os.close(descriptor)

    assert attempts == 2
    assert destination.read_bytes() == b"complete payload"


def test_descriptor_source_restore_copy_failure_removes_partial_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    target.write_bytes(trusted_bytes)
    target.chmod(0o444)
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    source_contract_descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    parent_metadata = os.fstat(parent_descriptor)
    directory = artifacts._BoundDirectory(
        locator_path=tmp_path,
        opened_path=tmp_path,
        descriptor=parent_descriptor,
        identity=(parent_metadata.st_dev, parent_metadata.st_ino),
    )
    target_metadata = target.stat()
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            staged,
            target,
            "generated root",
            source_descriptor=source_contract_descriptor,
            source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
        ),
        staged_entry=artifacts._BoundEntry(directory, staged.name),
        target_entry=artifacts._BoundEntry(directory, target.name),
        descriptor_source=artifacts._BoundDescriptorSource(
            descriptor=source_contract_descriptor,
            identity=(target_metadata.st_dev, target_metadata.st_ino),
            sha256=hashlib.sha256(trusted_bytes).hexdigest(),
            mode=0o400,
            is_directory=False,
        ),
    )
    detached_target = artifacts._DetachedTarget(
        identity=(target_metadata.st_dev, target_metadata.st_ino),
        sha256=hashlib.sha256(trusted_bytes).hexdigest(),
        mode=stat.S_IMODE(target_metadata.st_mode),
        is_directory=False,
    )
    expected = RuntimeError("forced descriptor restore copy failure")

    def fail_copy(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise expected

    monkeypatch.setattr(artifacts, "_copy_stable_descriptor", fail_copy)
    try:
        with pytest.raises(RuntimeError) as raised:
            artifacts._restore_descriptor_source_name(
                bound_artifact,
                detached_target,
            )
    finally:
        os.close(source_contract_descriptor)
        os.close(parent_descriptor)

    assert raised.value is expected
    assert not staged.exists()


def test_descriptor_source_restore_baseexception_cleans_exact_partial_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    target.write_bytes(trusted_bytes)
    target.chmod(0o444)
    parent = artifacts._open_bound_directory(tmp_path)
    source_contract_descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    target_metadata = target.stat()
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            staged,
            target,
            "generated root",
            source_descriptor=source_contract_descriptor,
            source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
        ),
        staged_entry=artifacts._BoundEntry(parent, staged.name),
        target_entry=artifacts._BoundEntry(parent, target.name),
        descriptor_source=artifacts._BoundDescriptorSource(
            descriptor=source_contract_descriptor,
            identity=(target_metadata.st_dev, target_metadata.st_ino),
            sha256=hashlib.sha256(trusted_bytes).hexdigest(),
            mode=0o444,
            is_directory=False,
        ),
    )
    detached_target = artifacts._DetachedTarget(
        identity=(target_metadata.st_dev, target_metadata.st_ino),
        sha256=hashlib.sha256(trusted_bytes).hexdigest(),
        mode=0o444,
        is_directory=False,
    )
    expected = KeyboardInterrupt("forced descriptor restore interrupt")
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    restore_target_descriptor: int | None = None

    def track_restore_target(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal restore_target_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == staged.name and flags & os.O_CREAT:
            restore_target_descriptor = descriptor
        return descriptor

    def interrupt_copy(*args: object, **kwargs: object) -> None:
        del args
        target_descriptor = kwargs["target_descriptor"]
        assert isinstance(target_descriptor, int)
        os.write(target_descriptor, b"partial")
        raise expected

    def fail_after_target_close(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == restore_target_descriptor:
            raise OSError(errno.EIO, "forced restore target close failure")

    monkeypatch.setattr(artifacts.os, "open", track_restore_target)
    monkeypatch.setattr(artifacts.os, "close", fail_after_target_close)
    monkeypatch.setattr(artifacts, "_copy_stable_descriptor", interrupt_copy)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            artifacts._restore_descriptor_source_name(
                bound_artifact,
                detached_target,
            )
    finally:
        real_close(source_contract_descriptor)
        real_close(parent.descriptor)

    assert raised.value is expected
    assert restore_target_descriptor is not None
    assert any(
        "forced restore target close failure" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert not staged.exists()
    assert not any(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_descriptor_source_restore_open_race_preserves_concurrent_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    target.write_bytes(trusted_bytes)
    target.chmod(0o444)
    parent = artifacts._open_bound_directory(tmp_path)
    source_descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    target_metadata = target.stat()
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            staged,
            target,
            "generated root",
            source_descriptor=source_descriptor,
            source_sha256=hashlib.sha256(trusted_bytes).hexdigest(),
        ),
        staged_entry=artifacts._BoundEntry(parent, staged.name),
        target_entry=artifacts._BoundEntry(parent, target.name),
        descriptor_source=artifacts._BoundDescriptorSource(
            descriptor=source_descriptor,
            identity=(target_metadata.st_dev, target_metadata.st_ino),
            sha256=hashlib.sha256(trusted_bytes).hexdigest(),
            mode=0o444,
            is_directory=False,
        ),
    )
    detached_target = artifacts._DetachedTarget(
        identity=(target_metadata.st_dev, target_metadata.st_ino),
        sha256=hashlib.sha256(trusted_bytes).hexdigest(),
        mode=0o444,
        is_directory=False,
    )
    real_open = artifacts.os.open
    injected = False

    def create_concurrent_entry_before_exclusive_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal injected
        if not injected and path == staged.name and flags & os.O_EXCL:
            injected = True
            staged.write_bytes(b"concurrent bytes")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        artifacts.os,
        "open",
        create_concurrent_entry_before_exclusive_open,
    )
    try:
        with pytest.raises(FileExistsError):
            artifacts._restore_descriptor_source_name(
                bound_artifact,
                detached_target,
            )
    finally:
        os.close(source_descriptor)
        os.close(parent.descriptor)

    assert injected
    assert staged.read_bytes() == b"concurrent bytes"


def test_backup_post_create_mkdir_fatal_preserves_name_with_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = artifacts._open_bound_directory(tmp_path)
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            tmp_path / "staged.usda",
            tmp_path / "target.usda",
            "generated root",
        ),
        staged_entry=artifacts._BoundEntry(parent, "staged.usda"),
        target_entry=artifacts._BoundEntry(parent, "target.usda"),
        descriptor_source=None,
    )
    real_mkdir = artifacts.os.mkdir
    expected = KeyboardInterrupt("forced post-create rollback mkdir fatal")

    def mkdir_then_interrupt(
        path: str | os.PathLike[str],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        real_mkdir(path, mode, dir_fd=dir_fd)
        if dir_fd == parent.descriptor:
            raise expected

    monkeypatch.setattr(artifacts.os, "mkdir", mkdir_then_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            artifacts._create_artifact_backup(
                bound_artifact,
                artifact_identity=(1, 1),
            )
        os.fstat(parent.descriptor)
    finally:
        os.close(parent.descriptor)

    assert raised.value is expected
    residual_names = list(tmp_path.glob(".joint-rigger.rollback-*"))
    assert len(residual_names) == 1
    assert "unpredictable private name preserved" in "\n".join(raised.value.__notes__)
    residual_names[0].rmdir()


def test_backup_child_open_failure_preserves_unbound_reserved_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = artifacts._open_bound_directory(tmp_path)
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            tmp_path / "staged.usda",
            tmp_path / "target.usda",
            "generated root",
        ),
        staged_entry=artifacts._BoundEntry(parent, "staged.usda"),
        target_entry=artifacts._BoundEntry(parent, "target.usda"),
        descriptor_source=None,
    )
    expected = RuntimeError("forced backup child open failure")

    def fail_child_open(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise expected

    monkeypatch.setattr(
        artifacts,
        "_open_child_directory_descriptor",
        fail_child_open,
    )
    try:
        with pytest.raises(RuntimeError) as raised:
            artifacts._create_artifact_backup(
                bound_artifact,
                artifact_identity=(1, 1),
            )
    finally:
        os.close(parent.descriptor)

    assert raised.value is expected
    residual_names = list(tmp_path.glob(".joint-rigger.rollback-*"))
    assert len(residual_names) == 1
    assert "unpredictable private name preserved" in "\n".join(raised.value.__notes__)
    residual_names[0].rmdir()


def test_backup_child_open_fatal_preserves_unbound_reserved_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = artifacts._open_bound_directory(tmp_path)
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            tmp_path / "staged.usda",
            tmp_path / "target.usda",
            "generated root",
        ),
        staged_entry=artifacts._BoundEntry(parent, "staged.usda"),
        target_entry=artifacts._BoundEntry(parent, "target.usda"),
        descriptor_source=None,
    )
    expected = KeyboardInterrupt("forced fatal backup child open failure")

    def fail_child_open(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise expected

    monkeypatch.setattr(
        artifacts,
        "_open_child_directory_descriptor",
        fail_child_open,
    )
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            artifacts._create_artifact_backup(
                bound_artifact,
                artifact_identity=(1, 1),
            )
    finally:
        os.close(parent.descriptor)

    assert raised.value is expected
    residual_names = list(tmp_path.glob(".joint-rigger.rollback-*"))
    assert len(residual_names) == 1
    assert "unpredictable private name preserved" in "\n".join(raised.value.__notes__)
    residual_names[0].rmdir()


def test_backup_transient_identity_fatal_cleans_name_and_child_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = artifacts._open_bound_directory(tmp_path)
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            tmp_path / "staged.usda",
            tmp_path / "target.usda",
            "generated root",
        ),
        staged_entry=artifacts._BoundEntry(parent, "staged.usda"),
        target_entry=artifacts._BoundEntry(parent, "target.usda"),
        descriptor_source=None,
    )
    original_open_child = artifacts._open_child_directory_descriptor
    real_fstat = artifacts.os.fstat
    real_close = artifacts.os.close
    primary_error = KeyboardInterrupt("forced transient backup identity fatal")
    child_descriptor: int | None = None
    child_close_attempts = 0
    stat_calls = 0

    def track_child_open(parent_descriptor: int, name: str) -> int:
        nonlocal child_descriptor
        child_descriptor = original_open_child(parent_descriptor, name)
        return child_descriptor

    def fail_first_identity_fstat(descriptor: int) -> os.stat_result:
        nonlocal stat_calls
        if descriptor == child_descriptor:
            stat_calls += 1
            if stat_calls == 1:
                raise primary_error
        return real_fstat(descriptor)

    def track_child_close(descriptor: int) -> None:
        nonlocal child_close_attempts
        real_close(descriptor)
        if descriptor == child_descriptor:
            child_close_attempts += 1

    monkeypatch.setattr(
        artifacts,
        "_open_child_directory_descriptor",
        track_child_open,
    )
    monkeypatch.setattr(artifacts.os, "fstat", fail_first_identity_fstat)
    monkeypatch.setattr(artifacts.os, "close", track_child_close)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            artifacts._create_artifact_backup(
                bound_artifact,
                artifact_identity=(1, 1),
            )

        assert raised.value is primary_error
        assert stat_calls >= 2
        assert child_descriptor is not None
        assert child_close_attempts == 1
        with pytest.raises(OSError):
            os.fstat(child_descriptor)
        assert not any(tmp_path.glob(".joint-rigger.rollback-*"))
    finally:
        real_close(parent.descriptor)


def test_backup_persistent_identity_fatal_preserves_name_with_note_and_closes_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = artifacts._open_bound_directory(tmp_path)
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            tmp_path / "staged.usda",
            tmp_path / "target.usda",
            "generated root",
        ),
        staged_entry=artifacts._BoundEntry(parent, "staged.usda"),
        target_entry=artifacts._BoundEntry(parent, "target.usda"),
        descriptor_source=None,
    )
    original_open_child = artifacts._open_child_directory_descriptor
    real_fstat = artifacts.os.fstat
    real_close = artifacts.os.close
    primary_error = KeyboardInterrupt("forced initial backup identity fatal")
    cleanup_error = SystemExit("forced persistent backup identity fatal")
    child_descriptor: int | None = None
    child_close_attempts = 0
    stat_calls = 0

    def track_child_open(parent_descriptor: int, name: str) -> int:
        nonlocal child_descriptor
        child_descriptor = original_open_child(parent_descriptor, name)
        return child_descriptor

    def fail_identity_fstat(descriptor: int) -> os.stat_result:
        nonlocal stat_calls
        if descriptor == child_descriptor:
            stat_calls += 1
            if stat_calls == 1:
                raise primary_error
            if stat_calls == 2:
                raise cleanup_error
        return real_fstat(descriptor)

    def track_child_close(descriptor: int) -> None:
        nonlocal child_close_attempts
        real_close(descriptor)
        if descriptor == child_descriptor:
            child_close_attempts += 1

    monkeypatch.setattr(
        artifacts,
        "_open_child_directory_descriptor",
        track_child_open,
    )
    monkeypatch.setattr(artifacts.os, "fstat", fail_identity_fstat)
    monkeypatch.setattr(artifacts.os, "close", track_child_close)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            artifacts._create_artifact_backup(
                bound_artifact,
                artifact_identity=(1, 1),
            )

        assert raised.value is primary_error
        assert stat_calls == 2
        assert child_descriptor is not None
        assert child_close_attempts == 1
        with pytest.raises(OSError):
            os.fstat(child_descriptor)
        os.fstat(parent.descriptor)
        residual_names = list(tmp_path.glob(".joint-rigger.rollback-*"))
        assert len(residual_names) == 1
        assert "forced persistent backup identity fatal" in "\n".join(
            raised.value.__notes__
        )
        residual_names[0].rmdir()
    finally:
        real_close(parent.descriptor)


def test_backup_acquisition_substitution_preserves_foreign_and_owned_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = artifacts._open_bound_directory(tmp_path)
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            tmp_path / "staged.usda",
            tmp_path / "target.usda",
            "generated root",
        ),
        staged_entry=artifacts._BoundEntry(parent, "staged.usda"),
        target_entry=artifacts._BoundEntry(parent, "target.usda"),
        descriptor_source=None,
    )
    original_open_child = artifacts._open_child_directory_descriptor
    original_optional_identity = artifacts._optional_descriptor_entry_identity
    real_close = artifacts.os.close
    displaced = tmp_path / "owned-rollback-displaced"
    child_descriptor: int | None = None
    held_identity: tuple[int, int] | None = None
    close_attempts = 0
    substituted_name: str | None = None

    def track_child_open(parent_descriptor: int, name: str) -> int:
        nonlocal child_descriptor
        child_descriptor = original_open_child(parent_descriptor, name)
        return child_descriptor

    def substitute_before_lexical_check(
        parent_descriptor: int,
        name: str,
    ) -> tuple[int, int] | None:
        nonlocal held_identity, substituted_name
        if name.startswith(".joint-rigger.rollback-") and substituted_name is None:
            assert child_descriptor is not None
            opened = os.fstat(child_descriptor)
            held_identity = (opened.st_dev, opened.st_ino)
            os.rename(
                name,
                displaced.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            (displaced / "owned.txt").write_text("owned", encoding="utf-8")
            (tmp_path / name / "foreign.txt").write_text(
                "foreign",
                encoding="utf-8",
            )
            substituted_name = name
        return original_optional_identity(parent_descriptor, name)

    def track_child_close(descriptor: int) -> None:
        nonlocal close_attempts
        real_close(descriptor)
        if descriptor == child_descriptor:
            close_attempts += 1

    monkeypatch.setattr(
        artifacts,
        "_open_child_directory_descriptor",
        track_child_open,
    )
    monkeypatch.setattr(
        artifacts,
        "_optional_descriptor_entry_identity",
        substitute_before_lexical_check,
    )
    monkeypatch.setattr(artifacts.os, "close", track_child_close)
    try:
        with pytest.raises(RuntimeError, match="changed inode") as raised:
            artifacts._create_artifact_backup(
                bound_artifact,
                artifact_identity=(1, 1),
            )

        assert substituted_name is not None
        foreign = tmp_path / substituted_name
        assert (foreign / "foreign.txt").read_text(encoding="utf-8") == "foreign"
        assert (displaced / "owned.txt").read_text(encoding="utf-8") == "owned"
        displaced_metadata = displaced.stat()
        assert held_identity == (displaced_metadata.st_dev, displaced_metadata.st_ino)
        assert child_descriptor is not None
        assert close_attempts == 1
        with pytest.raises(OSError):
            os.fstat(child_descriptor)
        assert "replacement preserved" in "\n".join(raised.value.__notes__)
    finally:
        real_close(parent.descriptor)


def test_backup_acquisition_rename_away_reports_owned_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = artifacts._open_bound_directory(tmp_path)
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            tmp_path / "staged.usda",
            tmp_path / "target.usda",
            "generated root",
        ),
        staged_entry=artifacts._BoundEntry(parent, "staged.usda"),
        target_entry=artifacts._BoundEntry(parent, "target.usda"),
        descriptor_source=None,
    )
    original_optional_identity = artifacts._optional_descriptor_entry_identity
    displaced = tmp_path / "owned-rollback-displaced"
    renamed = False

    def rename_before_lexical_check(
        parent_descriptor: int,
        name: str,
    ) -> tuple[int, int] | None:
        nonlocal renamed
        if name.startswith(".joint-rigger.rollback-") and not renamed:
            os.rename(
                name,
                displaced.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            renamed = True
        return original_optional_identity(parent_descriptor, name)

    monkeypatch.setattr(
        artifacts,
        "_optional_descriptor_entry_identity",
        rename_before_lexical_check,
    )
    try:
        with pytest.raises(
            RuntimeError, match="changed inode while it was bound"
        ) as raised:
            artifacts._create_artifact_backup(
                bound_artifact,
                artifact_identity=(1, 1),
            )

        assert renamed
        assert displaced.is_dir()
        assert "descriptor-owned inode remains linked elsewhere" in "\n".join(
            raised.value.__notes__
        )
        displaced.rmdir()
    finally:
        os.close(parent.descriptor)


def test_backup_cleanup_preserves_substituted_payload_and_owned_inode(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.usda"
    target.write_text("owned target", encoding="utf-8")
    parent = artifacts._open_bound_directory(tmp_path)
    target_entry = artifacts._BoundEntry(parent, target.name)
    target_identity = artifacts._bound_entry_identity(target_entry)
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            tmp_path / "staged.usda",
            target,
            "generated root",
        ),
        staged_entry=artifacts._BoundEntry(parent, "staged.usda"),
        target_entry=target_entry,
        descriptor_source=None,
    )
    backup = artifacts._create_artifact_backup(
        bound_artifact,
        artifact_identity=target_identity,
    )
    try:
        artifacts._replace_entry(target_entry, backup.artifact_entry)
        owned_displaced = backup.directory.opened_path / "owned-displaced"
        replacement = backup.directory.opened_path / backup.artifact_entry.name
        replacement.rename(owned_displaced)
        replacement.write_text("replacement", encoding="utf-8")

        with pytest.raises(RuntimeError, match="replacement preserved"):
            artifacts._remove_backup_directory(backup)

        assert replacement.read_text(encoding="utf-8") == "replacement"
        assert owned_displaced.read_text(encoding="utf-8") == "owned target"
        assert backup.directory.opened_path.is_dir()
    finally:
        os.close(backup.directory.descriptor)
        os.close(parent.descriptor)


def test_private_detached_file_post_create_open_fatal_preserves_name_with_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = artifacts._open_bound_directory(tmp_path)
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            tmp_path / "staged.usda",
            tmp_path / "target.usda",
            "generated root",
        ),
        staged_entry=artifacts._BoundEntry(parent, "staged.usda"),
        target_entry=artifacts._BoundEntry(parent, "target.usda"),
        descriptor_source=None,
    )
    real_open = artifacts.os.open
    hidden_descriptor: int | None = None
    expected = KeyboardInterrupt("forced post-create detached target open fatal")

    def create_then_interrupt(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal hidden_descriptor
        if os.fspath(path).startswith(".joint-rigger-copy-"):
            hidden_descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            raise expected
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", create_then_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            artifacts._create_private_detached_target(bound_artifact)
    finally:
        if hidden_descriptor is not None:
            os.close(hidden_descriptor)
        os.close(parent.descriptor)

    assert raised.value is expected
    residual_names = list(tmp_path.glob(".joint-rigger-copy-*"))
    assert len(residual_names) == 1
    assert "unpredictable private name" in "\n".join(raised.value.__notes__)
    residual_names[0].unlink()


def test_private_detached_file_creation_failure_preserves_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = artifacts._open_bound_directory(tmp_path)
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            tmp_path / "staged.usda",
            tmp_path / "target.usda",
            "generated root",
        ),
        staged_entry=artifacts._BoundEntry(parent, "staged.usda"),
        target_entry=artifacts._BoundEntry(parent, "target.usda"),
        descriptor_source=None,
    )
    displaced = tmp_path / "owned-copy-displaced"
    real_rename = artifacts._rename_descriptor_entry_noreplace
    substituted = False

    def substitute_before_cleanup(
        source_parent_descriptor: int,
        source_name: str,
        target_parent_descriptor: int,
        target_name: str,
        *,
        label: str,
    ) -> None:
        nonlocal substituted
        if not substituted and source_name.startswith(".joint-rigger-copy-"):
            substituted = True
            (tmp_path / source_name).rename(displaced)
            (tmp_path / source_name).write_bytes(b"foreign copy")
        real_rename(
            source_parent_descriptor,
            source_name,
            target_parent_descriptor,
            target_name,
            label=label,
        )

    monkeypatch.setattr(
        artifacts,
        "_rename_descriptor_entry_noreplace",
        substitute_before_cleanup,
    )
    try:
        with pytest.raises(AssertionError) as raised:
            artifacts._create_private_detached_target(bound_artifact)
    finally:
        os.close(parent.descriptor)

    assert substituted
    assert displaced.exists()
    private_names = list(tmp_path.glob(".joint-rigger-copy-*"))
    assert len(private_names) == 1
    assert private_names[0].read_bytes() == b"foreign copy"
    assert any(
        "cleanup also failed" in note for note in getattr(raised.value, "__notes__", ())
    )
    assert not any(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_private_detached_tree_post_create_mkdir_fatal_preserves_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source-tree"
    source.mkdir()
    source_descriptor = os.open(
        source,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    parent = artifacts._open_bound_directory(tmp_path)
    source_metadata = os.fstat(source_descriptor)
    source_sha256 = artifacts.directory_descriptor_tree_sha256(source_descriptor)
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            source,
            tmp_path / "target-tree",
            "composition sidecar",
            source_descriptor=source_descriptor,
            source_sha256=source_sha256,
        ),
        staged_entry=artifacts._BoundEntry(parent, source.name),
        target_entry=artifacts._BoundEntry(parent, "target-tree"),
        descriptor_source=artifacts._BoundDescriptorSource(
            descriptor=source_descriptor,
            identity=(source_metadata.st_dev, source_metadata.st_ino),
            sha256=source_sha256,
            mode=0o700,
            is_directory=True,
        ),
    )
    real_mkdir = artifacts.os.mkdir
    expected = KeyboardInterrupt("forced post-create detached tree mkdir fatal")

    def mkdir_then_interrupt(
        path: str | os.PathLike[str],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        real_mkdir(path, mode, dir_fd=dir_fd)
        if os.fspath(path).startswith(".joint-rigger-tree-copy-"):
            raise expected

    monkeypatch.setattr(artifacts.os, "mkdir", mkdir_then_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            artifacts._create_private_detached_directory(bound_artifact)
    finally:
        os.close(source_descriptor)
        os.close(parent.descriptor)

    assert raised.value is expected
    residual_names = list(tmp_path.glob(".joint-rigger-tree-copy-*"))
    assert len(residual_names) == 1
    assert "unpredictable private name" in "\n".join(raised.value.__notes__)
    residual_names[0].rmdir()


def test_private_detached_tree_creation_failure_preserves_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source-tree"
    source.mkdir()
    (source / "asset.bin").write_bytes(b"source")
    source_descriptor = os.open(
        source,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    parent = artifacts._open_bound_directory(tmp_path)
    source_metadata = os.fstat(source_descriptor)
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            source,
            tmp_path / "target-tree",
            "composition sidecar",
            source_descriptor=source_descriptor,
            source_sha256=artifacts.directory_descriptor_tree_sha256(source_descriptor),
        ),
        staged_entry=artifacts._BoundEntry(parent, source.name),
        target_entry=artifacts._BoundEntry(parent, "target-tree"),
        descriptor_source=artifacts._BoundDescriptorSource(
            descriptor=source_descriptor,
            identity=(source_metadata.st_dev, source_metadata.st_ino),
            sha256=artifacts.directory_descriptor_tree_sha256(source_descriptor),
            mode=stat.S_IMODE(source_metadata.st_mode),
            is_directory=True,
        ),
    )
    displaced = tmp_path / "owned-tree-displaced"
    real_rename = artifacts._rename_descriptor_entry_noreplace
    substituted = False
    expected = RuntimeError("forced detached target construction failure")

    def fail_detached_target(**kwargs: object) -> Any:
        del kwargs
        raise expected

    def substitute_before_cleanup(
        source_parent_descriptor: int,
        source_name: str,
        target_parent_descriptor: int,
        target_name: str,
        *,
        label: str,
    ) -> None:
        nonlocal substituted
        if not substituted and source_name.startswith(".joint-rigger-tree-copy-"):
            substituted = True
            (tmp_path / source_name).rename(displaced)
            replacement = tmp_path / source_name
            replacement.mkdir()
            (replacement / "foreign.txt").write_text("foreign", encoding="utf-8")
        real_rename(
            source_parent_descriptor,
            source_name,
            target_parent_descriptor,
            target_name,
            label=label,
        )

    monkeypatch.setattr(artifacts, "_DetachedTarget", fail_detached_target)
    monkeypatch.setattr(
        artifacts,
        "_rename_descriptor_entry_noreplace",
        substitute_before_cleanup,
    )
    try:
        with pytest.raises(RuntimeError) as raised:
            artifacts._create_private_detached_directory(bound_artifact)
    finally:
        os.close(source_descriptor)
        os.close(parent.descriptor)

    assert raised.value is expected
    assert substituted
    assert displaced.is_dir()
    private_names = list(tmp_path.glob(".joint-rigger-tree-copy-*"))
    assert len(private_names) == 1
    assert (private_names[0] / "foreign.txt").read_text(encoding="utf-8") == ("foreign")
    assert any(
        "cleanup also failed" in note for note in getattr(raised.value, "__notes__", ())
    )
    assert not any(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_backend_staging_validation_failure_cleans_name_and_parent_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open_bound = artifacts._open_bound_directory
    parent_descriptor: int | None = None
    expected = RuntimeError("forced backend staging parent drift")

    def track_parent(path: Path) -> Any:
        nonlocal parent_descriptor
        parent = original_open_bound(path)
        parent_descriptor = parent.descriptor
        return parent

    def fail_validation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise expected

    monkeypatch.setattr(artifacts, "_open_bound_directory", track_parent)
    monkeypatch.setattr(
        artifacts, "_require_bound_directory_unchanged", fail_validation
    )

    with pytest.raises(RuntimeError) as raised:
        artifacts._reserve_backend_staging_name(tmp_path / "rigged.usda")

    assert raised.value is expected
    assert parent_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    assert not any(tmp_path.glob(".*.stage-*"))


def test_sidecar_owner_validation_failure_cleans_name_and_parent_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open_bound = artifacts._open_bound_directory
    parent_descriptor: int | None = None
    expected = RuntimeError("forced sidecar owner parent drift")

    def track_parent(path: Path) -> Any:
        nonlocal parent_descriptor
        parent = original_open_bound(path)
        parent_descriptor = parent.descriptor
        return parent

    def fail_validation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise expected

    monkeypatch.setattr(artifacts, "_open_bound_directory", track_parent)
    monkeypatch.setattr(
        artifacts, "_require_bound_directory_unchanged", fail_validation
    )

    with pytest.raises(RuntimeError) as raised:
        artifacts._create_sidecar_owner_reservation(
            tmp_path,
            target_name="rigged_assets",
        )

    assert raised.value is expected
    assert parent_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    assert not any(tmp_path.glob(".*.stage-*"))


def test_sidecar_owner_post_create_mkdir_fatal_preserves_name_with_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open_bound = artifacts._open_bound_directory
    real_mkdir = artifacts.os.mkdir
    parent_descriptor: int | None = None
    expected = KeyboardInterrupt("forced post-create staging owner mkdir fatal")

    def track_parent(path: Path) -> Any:
        nonlocal parent_descriptor
        parent = original_open_bound(path)
        parent_descriptor = parent.descriptor
        return parent

    def mkdir_then_interrupt(
        path: str | os.PathLike[str],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        real_mkdir(path, mode, dir_fd=dir_fd)
        if dir_fd == parent_descriptor:
            raise expected

    monkeypatch.setattr(artifacts, "_open_bound_directory", track_parent)
    monkeypatch.setattr(artifacts.os, "mkdir", mkdir_then_interrupt)

    with pytest.raises(KeyboardInterrupt) as raised:
        artifacts._create_sidecar_owner_reservation(
            tmp_path,
            target_name="rigged_assets",
        )

    assert raised.value is expected
    assert parent_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    residual_names = list(tmp_path.glob(".rigged_assets.stage-*"))
    assert len(residual_names) == 1
    assert "unpredictable private name preserved" in "\n".join(raised.value.__notes__)
    residual_names[0].rmdir()


def test_sidecar_owner_child_open_failure_preserves_unbound_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open_bound = artifacts._open_bound_directory
    parent_descriptor: int | None = None
    expected = RuntimeError("forced sidecar child open failure")

    def track_parent(path: Path) -> Any:
        nonlocal parent_descriptor
        parent = original_open_bound(path)
        parent_descriptor = parent.descriptor
        return parent

    def fail_child_open(*args: object, **kwargs: object) -> int:
        del args, kwargs
        raise expected

    monkeypatch.setattr(artifacts, "_open_bound_directory", track_parent)
    monkeypatch.setattr(
        artifacts,
        "_open_child_directory_descriptor",
        fail_child_open,
    )

    with pytest.raises(RuntimeError) as raised:
        artifacts._create_sidecar_owner_reservation(
            tmp_path,
            target_name="rigged_assets",
        )

    assert raised.value is expected
    assert parent_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    residual_names = list(tmp_path.glob(".*.stage-*"))
    assert len(residual_names) == 1
    assert "unpredictable private name preserved" in "\n".join(raised.value.__notes__)
    residual_names[0].rmdir()


def test_sidecar_owner_transient_identity_fatal_rebinds_and_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open_bound = artifacts._open_bound_directory
    original_open_child = artifacts._open_child_directory_descriptor
    real_fstat = artifacts.os.fstat
    real_close = artifacts.os.close
    parent_descriptor: int | None = None
    child_descriptor: int | None = None
    child_close_attempts = 0
    stat_calls = 0
    primary_error = KeyboardInterrupt("forced transient sidecar identity fatal")

    def track_parent(path: Path) -> Any:
        nonlocal parent_descriptor
        parent = original_open_bound(path)
        parent_descriptor = parent.descriptor
        return parent

    def track_child_open(parent_descriptor: int, name: str) -> int:
        nonlocal child_descriptor
        child_descriptor = original_open_child(parent_descriptor, name)
        return child_descriptor

    def fail_first_identity_fstat(descriptor: int) -> os.stat_result:
        nonlocal stat_calls
        if descriptor == child_descriptor:
            stat_calls += 1
            if stat_calls == 1:
                raise primary_error
        return real_fstat(descriptor)

    def track_child_close(descriptor: int) -> None:
        nonlocal child_close_attempts
        real_close(descriptor)
        if descriptor == child_descriptor:
            child_close_attempts += 1

    monkeypatch.setattr(artifacts, "_open_bound_directory", track_parent)
    monkeypatch.setattr(
        artifacts,
        "_open_child_directory_descriptor",
        track_child_open,
    )
    monkeypatch.setattr(artifacts.os, "fstat", fail_first_identity_fstat)
    monkeypatch.setattr(artifacts.os, "close", track_child_close)

    with pytest.raises(KeyboardInterrupt) as raised:
        artifacts._create_sidecar_owner_reservation(
            tmp_path,
            target_name="rigged_assets",
        )

    assert raised.value is primary_error
    assert stat_calls >= 2
    assert child_descriptor is not None
    assert child_close_attempts == 1
    with pytest.raises(OSError):
        os.fstat(child_descriptor)
    assert parent_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    assert not any(tmp_path.glob(".*.stage-*"))


def test_sidecar_owner_persistent_identity_fatal_preserves_name_with_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open_bound = artifacts._open_bound_directory
    original_open_child = artifacts._open_child_directory_descriptor
    real_fstat = artifacts.os.fstat
    real_close = artifacts.os.close
    parent_descriptor: int | None = None
    child_descriptor: int | None = None
    child_close_attempts = 0
    stat_calls = 0
    primary_error = KeyboardInterrupt("forced initial sidecar identity fatal")
    cleanup_error = SystemExit("forced persistent sidecar identity fatal")

    def track_parent(path: Path) -> Any:
        nonlocal parent_descriptor
        parent = original_open_bound(path)
        parent_descriptor = parent.descriptor
        return parent

    def track_child_open(parent_descriptor: int, name: str) -> int:
        nonlocal child_descriptor
        child_descriptor = original_open_child(parent_descriptor, name)
        return child_descriptor

    def fail_identity_fstat(descriptor: int) -> os.stat_result:
        nonlocal stat_calls
        if descriptor == child_descriptor:
            stat_calls += 1
            if stat_calls == 1:
                raise primary_error
            if stat_calls == 2:
                raise cleanup_error
        return real_fstat(descriptor)

    def track_child_close(descriptor: int) -> None:
        nonlocal child_close_attempts
        real_close(descriptor)
        if descriptor == child_descriptor:
            child_close_attempts += 1

    monkeypatch.setattr(artifacts, "_open_bound_directory", track_parent)
    monkeypatch.setattr(
        artifacts,
        "_open_child_directory_descriptor",
        track_child_open,
    )
    monkeypatch.setattr(artifacts.os, "fstat", fail_identity_fstat)
    monkeypatch.setattr(artifacts.os, "close", track_child_close)

    with pytest.raises(KeyboardInterrupt) as raised:
        artifacts._create_sidecar_owner_reservation(
            tmp_path,
            target_name="rigged_assets",
        )

    assert raised.value is primary_error
    assert stat_calls == 2
    assert child_descriptor is not None
    assert child_close_attempts == 1
    with pytest.raises(OSError):
        os.fstat(child_descriptor)
    assert parent_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    residual_names = list(tmp_path.glob(".*.stage-*"))
    assert len(residual_names) == 1
    assert "forced persistent sidecar identity fatal" in "\n".join(
        raised.value.__notes__
    )
    residual_names[0].rmdir()


def test_sidecar_owner_acquisition_substitution_preserves_foreign_and_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open_bound = artifacts._open_bound_directory
    original_open_child = artifacts._open_child_directory_descriptor
    original_optional_identity = artifacts._optional_descriptor_entry_identity
    real_close = artifacts.os.close
    parent_descriptor: int | None = None
    child_descriptor: int | None = None
    held_identity: tuple[int, int] | None = None
    close_attempts = 0
    substituted_name: str | None = None
    displaced = tmp_path / "owned-sidecar-owner-displaced"

    def track_parent(path: Path) -> Any:
        nonlocal parent_descriptor
        parent = original_open_bound(path)
        parent_descriptor = parent.descriptor
        return parent

    def track_child_open(descriptor: int, name: str) -> int:
        nonlocal child_descriptor
        child_descriptor = original_open_child(descriptor, name)
        return child_descriptor

    def substitute_before_lexical_check(
        descriptor: int,
        name: str,
    ) -> tuple[int, int] | None:
        nonlocal held_identity, substituted_name
        if name.startswith(".rigged_assets.stage-") and substituted_name is None:
            assert child_descriptor is not None
            opened = os.fstat(child_descriptor)
            held_identity = (opened.st_dev, opened.st_ino)
            os.rename(
                name,
                displaced.name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
            os.mkdir(name, mode=0o700, dir_fd=descriptor)
            (displaced / "owned.txt").write_text("owned", encoding="utf-8")
            (tmp_path / name / "foreign.txt").write_text(
                "foreign",
                encoding="utf-8",
            )
            substituted_name = name
        return original_optional_identity(descriptor, name)

    def track_child_close(descriptor: int) -> None:
        nonlocal close_attempts
        real_close(descriptor)
        if descriptor == child_descriptor:
            close_attempts += 1

    monkeypatch.setattr(artifacts, "_open_bound_directory", track_parent)
    monkeypatch.setattr(
        artifacts,
        "_open_child_directory_descriptor",
        track_child_open,
    )
    monkeypatch.setattr(
        artifacts,
        "_optional_descriptor_entry_identity",
        substitute_before_lexical_check,
    )
    monkeypatch.setattr(artifacts.os, "close", track_child_close)

    with pytest.raises(RuntimeError, match="changed inode") as raised:
        artifacts._create_sidecar_owner_reservation(
            tmp_path,
            target_name="rigged_assets",
        )

    assert substituted_name is not None
    foreign = tmp_path / substituted_name
    assert (foreign / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert (displaced / "owned.txt").read_text(encoding="utf-8") == "owned"
    displaced_metadata = displaced.stat()
    assert held_identity == (displaced_metadata.st_dev, displaced_metadata.st_ino)
    assert child_descriptor is not None
    assert close_attempts == 1
    with pytest.raises(OSError):
        os.fstat(child_descriptor)
    assert parent_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    assert "replacement preserved" in "\n".join(raised.value.__notes__)


def test_sidecar_owner_acquisition_rename_away_reports_owned_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_optional_identity = artifacts._optional_descriptor_entry_identity
    displaced = tmp_path / "owned-sidecar-owner-displaced"
    renamed = False

    def rename_before_lexical_check(
        parent_descriptor: int,
        name: str,
    ) -> tuple[int, int] | None:
        nonlocal renamed
        if name.startswith(".rigged_assets.stage-") and not renamed:
            os.rename(
                name,
                displaced.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            renamed = True
        return original_optional_identity(parent_descriptor, name)

    monkeypatch.setattr(
        artifacts,
        "_optional_descriptor_entry_identity",
        rename_before_lexical_check,
    )

    with pytest.raises(
        RuntimeError, match="changed inode while it was bound"
    ) as raised:
        artifacts._create_sidecar_owner_reservation(
            tmp_path,
            target_name="rigged_assets",
        )

    assert renamed
    assert displaced.is_dir()
    assert "descriptor-owned inode remains linked elsewhere" in "\n".join(
        raised.value.__notes__
    )
    displaced.rmdir()


@pytest.mark.parametrize("reservation_kind", ["backend", "sidecar"])
def test_staging_reservation_fatal_cleans_name_and_keeps_primary_over_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reservation_kind: str,
) -> None:
    original_open_bound = artifacts._open_bound_directory
    real_close = artifacts.os.close
    parent_descriptor: int | None = None
    primary_error = KeyboardInterrupt(f"forced {reservation_kind} validation fatal")
    close_error = OSError(f"forced {reservation_kind} parent close failure")

    def track_parent(path: Path) -> Any:
        nonlocal parent_descriptor
        parent = original_open_bound(path)
        parent_descriptor = parent.descriptor
        return parent

    def fail_validation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise primary_error

    def close_parent_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == parent_descriptor:
            raise close_error

    monkeypatch.setattr(artifacts, "_open_bound_directory", track_parent)
    monkeypatch.setattr(
        artifacts,
        "_require_bound_directory_unchanged",
        fail_validation,
    )
    monkeypatch.setattr(artifacts.os, "close", close_parent_then_fail)

    with pytest.raises(KeyboardInterrupt) as raised:
        if reservation_kind == "backend":
            artifacts._reserve_backend_staging_name(tmp_path / "rigged.usda")
        else:
            artifacts._create_sidecar_owner_reservation(
                tmp_path,
                target_name="rigged_assets",
            )

    assert raised.value is primary_error
    assert parent_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    assert f"forced {reservation_kind} parent close failure" in "\n".join(
        raised.value.__notes__
    )
    assert not any(tmp_path.glob(".*.stage-*"))


def test_backend_staging_placeholder_post_create_open_fatal_preserves_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = artifacts.os.open
    hidden_descriptor: int | None = None
    hidden_name: str | None = None
    expected = KeyboardInterrupt("forced post-create placeholder open fatal")

    def create_then_interrupt(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal hidden_descriptor, hidden_name
        if flags & os.O_CREAT and flags & os.O_EXCL:
            hidden_descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            hidden_name = os.fspath(path)
            raise expected
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", create_then_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            artifacts._reserve_backend_staging_name(
                tmp_path / "rigged.usda",
                descriptor_owned=False,
            )
    finally:
        if hidden_descriptor is not None:
            os.close(hidden_descriptor)

    assert raised.value is expected
    assert hidden_name is not None
    residual = tmp_path / hidden_name
    assert residual.is_file()
    assert "unpredictable private name" in "\n".join(raised.value.__notes__)
    residual.unlink()


def test_backend_staging_placeholder_rename_away_reports_owned_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_remove = artifacts._remove_descriptor_entry
    original_optional_identity = artifacts._optional_bound_entry_identity
    primary_error = KeyboardInterrupt("forced initial placeholder removal failure")
    displaced = tmp_path / "owned-placeholder-displaced.usda"
    removal_calls = 0
    renamed = False

    def fail_initial_remove(*args: Any, **kwargs: Any) -> None:
        nonlocal removal_calls
        removal_calls += 1
        if removal_calls == 1:
            raise primary_error
        original_remove(*args, **kwargs)

    def rename_before_cleanup_identity(
        entry: artifacts._BoundEntry,
    ) -> tuple[int, int] | None:
        nonlocal renamed
        if entry.name.startswith(".rigged.stage-") and not renamed:
            os.rename(
                entry.name,
                displaced.name,
                src_dir_fd=entry.parent.descriptor,
                dst_dir_fd=entry.parent.descriptor,
            )
            renamed = True
        return original_optional_identity(entry)

    monkeypatch.setattr(artifacts, "_remove_descriptor_entry", fail_initial_remove)
    monkeypatch.setattr(
        artifacts,
        "_optional_bound_entry_identity",
        rename_before_cleanup_identity,
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        artifacts._reserve_backend_staging_name(
            tmp_path / "rigged.usda",
            descriptor_owned=False,
        )

    assert raised.value is primary_error
    assert removal_calls == 1
    assert renamed
    assert displaced.is_file()
    assert "descriptor-owned inode remains linked elsewhere" in "\n".join(
        raised.value.__notes__
    )
    displaced.unlink()


def test_backend_staging_placeholder_close_reuse_preserves_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_close = artifacts.os.close
    placeholder_descriptor: int | None = None
    placeholder_name: str | None = None
    parent_descriptor: int | None = None
    reused_identity: tuple[int, int] | None = None

    original_open = artifacts.os.open

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal parent_descriptor, placeholder_descriptor, placeholder_name
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is None:
            parent_descriptor = descriptor
        elif flags & os.O_EXCL:
            placeholder_descriptor = descriptor
            placeholder_name = os.fspath(path)
        return descriptor

    def close_then_reuse_placeholder_inode(descriptor: int) -> None:
        nonlocal reused_identity
        if descriptor != placeholder_descriptor:
            real_close(descriptor)
            return
        assert parent_descriptor is not None
        assert placeholder_name is not None
        opened = os.fstat(descriptor)
        expected_identity = (opened.st_dev, opened.st_ino)
        real_close(descriptor)
        for _ in range(4096):
            replacement = original_open(
                placeholder_name,
                os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_descriptor,
            )
            replacement_metadata = os.fstat(replacement)
            real_close(replacement)
            candidate = (
                replacement_metadata.st_dev,
                replacement_metadata.st_ino,
            )
            if candidate == expected_identity:
                reused_identity = candidate
                return
            os.unlink(placeholder_name, dir_fd=parent_descriptor)
        raise AssertionError("filesystem did not recycle the placeholder inode")

    monkeypatch.setattr(artifacts.os, "open", track_open)
    monkeypatch.setattr(artifacts.os, "close", close_then_reuse_placeholder_inode)

    with pytest.raises(RuntimeError, match="replacement preserved"):
        artifacts._reserve_backend_staging_name(
            tmp_path / "rigged.usda",
            descriptor_owned=False,
        )

    assert reused_identity is not None
    assert placeholder_name is not None
    replacement_path = tmp_path / placeholder_name
    assert replacement_path.is_file()
    replacement_path.write_text("foreign replacement", encoding="utf-8")
    assert replacement_path.read_text(encoding="utf-8") == "foreign replacement"
    assert parent_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)


def test_backend_staging_placeholder_close_fatal_cleans_name_and_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    placeholder_descriptor: int | None = None
    parent_descriptor: int | None = None
    close_attempts = 0
    fatal_error = SystemExit("forced placeholder close fatal")

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal parent_descriptor, placeholder_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is None:
            parent_descriptor = descriptor
        elif os.fspath(path).startswith("."):
            placeholder_descriptor = descriptor
        return descriptor

    def close_placeholder_then_fail(descriptor: int) -> None:
        nonlocal close_attempts
        real_close(descriptor)
        if descriptor == placeholder_descriptor and close_attempts == 0:
            close_attempts += 1
            raise fatal_error

    monkeypatch.setattr(artifacts.os, "open", track_open)
    monkeypatch.setattr(artifacts.os, "close", close_placeholder_then_fail)

    with pytest.raises(SystemExit) as raised:
        artifacts._reserve_backend_staging_name(
            tmp_path / "rigged.usda",
            descriptor_owned=False,
        )

    assert raised.value is fatal_error
    assert close_attempts == 1
    assert placeholder_descriptor is not None
    assert parent_descriptor is not None
    for descriptor in (placeholder_descriptor, parent_descriptor):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert not any(tmp_path.glob(".*.stage-*"))


def test_staging_cleanup_tolerates_missing_current_parent(tmp_path: Path) -> None:
    live_parent = tmp_path / "live"
    displaced_parent = tmp_path / "displaced"
    bundle = create_staged_artifact_targets(_targets(live_parent, sidecar=True))
    _write_staged_bundle(bundle.staged_targets)
    staged_promotion_artifacts(bundle)
    reservations = bundle._cleanup_reservations

    live_parent.rename(displaced_parent)
    bundle.cleanup()

    assert all(reservation.closed for reservation in reservations)
    assert not any(displaced_parent.rglob(".*.stage-*"))


def test_owned_entry_cleanup_tolerates_disappearance_after_identity_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    staged.write_text("payload", encoding="utf-8")
    parent = artifacts._open_bound_directory(tmp_path)
    entry = artifacts._BoundEntry(parent, staged.name)
    identity = artifacts._bound_entry_identity(entry)

    def disappear_before_unlink(
        path: str,
        *,
        dir_fd: int | None = None,
    ) -> None:
        del path, dir_fd
        raise FileNotFoundError

    monkeypatch.setattr(artifacts.os, "unlink", disappear_before_unlink)
    try:
        artifacts._remove_bound_entry_if_identity(entry, identity)
    finally:
        os.close(parent.descriptor)


def test_ambiguous_directory_creation_notes_missing_lexical_entry(
    tmp_path: Path,
) -> None:
    parent = artifacts._open_bound_directory(tmp_path)
    expected = KeyboardInterrupt("forced pre-mkdir interrupt")
    name = ".never-created-directory"
    try:
        artifacts._note_ambiguous_directory_creation(
            expected,
            parent,
            name,
            label="Coverage owner",
        )
    finally:
        os.close(parent.descriptor)

    assert not (tmp_path / name).exists()
    assert "no lexical entry was observed" in "\n".join(expected.__notes__)
    assert "no cleanup deletion was attempted" in "\n".join(expected.__notes__)


def test_ambiguous_file_creation_notes_missing_lexical_entry(tmp_path: Path) -> None:
    parent = artifacts._open_bound_directory(tmp_path)
    expected = KeyboardInterrupt("forced pre-open interrupt")
    name = ".never-created-file"
    try:
        artifacts._note_ambiguous_file_creation(
            expected,
            parent,
            name,
            label="Coverage file",
        )
    finally:
        os.close(parent.descriptor)

    assert not (tmp_path / name).exists()
    assert "no lexical entry was observed" in "\n".join(expected.__notes__)
    assert "no cleanup deletion was attempted" in "\n".join(expected.__notes__)


def test_created_directory_cleanup_tolerates_missing_lexical_entry(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    parent = artifacts._open_bound_directory(tmp_path)
    descriptor = artifacts._open_child_directory_descriptor(
        parent.descriptor,
        owned.name,
    )
    metadata = os.fstat(descriptor)
    identity = (metadata.st_dev, metadata.st_ino)
    os.rmdir(owned.name, dir_fd=parent.descriptor)
    try:
        artifacts._remove_created_directory_if_bound(
            parent,
            owned.name,
            descriptor=descriptor,
            identity=identity,
            label="coverage owner",
        )
        assert os.fstat(descriptor).st_nlink == 0
    finally:
        os.close(descriptor)
        os.close(parent.descriptor)

    assert not owned.exists()


def test_owned_file_identity_helper_preserves_mismatched_entry(tmp_path: Path) -> None:
    owned = tmp_path / "owned.bin"
    foreign = tmp_path / "foreign.bin"
    owned.write_bytes(b"owned")
    foreign.write_bytes(b"foreign")
    parent = artifacts._open_bound_directory(tmp_path)
    entry = artifacts._BoundEntry(parent, owned.name)
    foreign_metadata = foreign.stat()
    try:
        artifacts._remove_bound_entry_if_identity(
            entry,
            (foreign_metadata.st_dev, foreign_metadata.st_ino),
        )
    finally:
        os.close(parent.descriptor)

    assert owned.read_bytes() == b"owned"
    assert foreign.read_bytes() == b"foreign"


def test_owned_tree_helper_preserves_mismatch_then_removes_exact_tree(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "owned-tree"
    owned.mkdir()
    (owned / "member.bin").write_bytes(b"owned")
    foreign = tmp_path / "foreign.bin"
    foreign.write_bytes(b"foreign")
    parent = artifacts._open_bound_directory(tmp_path)
    entry = artifacts._BoundEntry(parent, owned.name)
    owned_metadata = owned.stat()
    foreign_metadata = foreign.stat()
    try:
        artifacts._remove_bound_entry_if_owned(
            entry,
            (foreign_metadata.st_dev, foreign_metadata.st_ino),
        )
        assert (owned / "member.bin").read_bytes() == b"owned"
        artifacts._remove_bound_entry_if_owned(
            entry,
            (owned_metadata.st_dev, owned_metadata.st_ino),
        )
    finally:
        os.close(parent.descriptor)

    assert not owned.exists()
    assert foreign.read_bytes() == b"foreign"


@pytest.mark.parametrize("entry_kind", ["file", "directory"])
def test_recursive_cleanup_quarantine_preserves_child_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    owned = root / "member"
    displaced = root / "owned-displaced"
    if entry_kind == "directory":
        owned.mkdir()
        (owned / "owned.txt").write_text("owned", encoding="utf-8")
    else:
        owned.write_text("owned", encoding="utf-8")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    mount_id = artifacts._descriptor_mount_id(descriptor)
    real_rename = artifacts._rename_descriptor_entry_noreplace
    substituted = False

    def substitute_before_quarantine(
        source_parent_descriptor: int,
        source_name: str,
        target_parent_descriptor: int,
        target_name: str,
        *,
        label: str,
    ) -> None:
        nonlocal substituted
        if not substituted and source_name == owned.name:
            substituted = True
            owned.rename(displaced)
            if entry_kind == "directory":
                owned.mkdir()
                (owned / "replacement.txt").write_text(
                    "replacement",
                    encoding="utf-8",
                )
            else:
                owned.write_text("replacement", encoding="utf-8")
        real_rename(
            source_parent_descriptor,
            source_name,
            target_parent_descriptor,
            target_name,
            label=label,
        )

    monkeypatch.setattr(
        artifacts,
        "_rename_descriptor_entry_noreplace",
        substitute_before_quarantine,
    )
    try:
        with pytest.raises(
            RuntimeError, match="changed inode during atomic quarantine"
        ):
            artifacts._remove_directory_descriptor_contents(
                descriptor,
                expected_mount_id=mount_id,
                label="test tree",
            )
    finally:
        os.close(descriptor)

    assert substituted
    if entry_kind == "directory":
        assert (owned / "replacement.txt").read_text(encoding="utf-8") == (
            "replacement"
        )
        assert (displaced / "owned.txt").read_text(encoding="utf-8") == "owned"
    else:
        assert owned.read_text(encoding="utf-8") == "replacement"
        assert displaced.read_text(encoding="utf-8") == "owned"
    assert not any(root.glob(".joint-rigger.cleanup-*"))


@pytest.mark.parametrize(
    "injected_error",
    [KeyboardInterrupt("forced quarantine interrupt"), FileExistsError("late EEXIST")],
)
def test_cleanup_quarantine_restores_completed_rename_before_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injected_error: BaseException,
) -> None:
    owned = tmp_path / "owned.bin"
    owned.write_bytes(b"owned bytes")
    identity = (owned.stat().st_dev, owned.stat().st_ino)
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    real_rename = artifacts._rename_descriptor_entry_noreplace
    interrupted = False

    def rename_then_interrupt(
        source_parent_descriptor: int,
        source_name: str,
        target_parent_descriptor: int,
        target_name: str,
        *,
        label: str,
    ) -> None:
        nonlocal interrupted
        real_rename(
            source_parent_descriptor,
            source_name,
            target_parent_descriptor,
            target_name,
            label=label,
        )
        if not interrupted and source_name == owned.name:
            interrupted = True
            raise injected_error

    monkeypatch.setattr(
        artifacts,
        "_rename_descriptor_entry_noreplace",
        rename_then_interrupt,
    )
    try:
        with pytest.raises(type(injected_error)) as raised:
            artifacts._remove_descriptor_entry(
                parent_descriptor,
                owned.name,
                expected_identity=identity,
                label="test owned entry",
            )
    finally:
        os.close(parent_descriptor)

    assert raised.value is injected_error
    assert interrupted
    assert owned.read_bytes() == b"owned bytes"
    assert not any(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_cleanup_quarantine_exception_preserves_foreign_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = tmp_path / "owned.bin"
    displaced = tmp_path / "owned-displaced.bin"
    owned.write_bytes(b"owned bytes")
    identity = (owned.stat().st_dev, owned.stat().st_ino)
    quarantine = tmp_path / f".joint-rigger.cleanup-{'a' * 32}"
    quarantine.write_bytes(b"foreign bytes")
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )

    def fail_after_source_disappears(*args: object, **kwargs: object) -> None:
        del args, kwargs
        owned.rename(displaced)
        raise FileExistsError("forced ambiguous EEXIST")

    monkeypatch.setattr(artifacts.secrets, "token_hex", lambda count: "a" * (count * 2))
    monkeypatch.setattr(
        artifacts,
        "_rename_descriptor_entry_noreplace",
        fail_after_source_disappears,
    )
    try:
        with pytest.raises(FileExistsError, match="forced ambiguous EEXIST"):
            artifacts._quarantine_descriptor_entry(
                parent_descriptor,
                owned.name,
                expected_identity=identity,
                label="test owned entry",
            )
    finally:
        os.close(parent_descriptor)

    assert not owned.exists()
    assert displaced.read_bytes() == b"owned bytes"
    assert quarantine.read_bytes() == b"foreign bytes"


@pytest.mark.parametrize(
    ("error_number", "expected_error", "message"),
    [
        (errno.ENOSYS, RuntimeError, "does not support atomic no-replace"),
        (errno.EACCES, OSError, "Permission denied"),
    ],
)
def test_cleanup_rename_reports_platform_and_filesystem_errors(
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    expected_error: type[BaseException],
    message: str,
) -> None:
    monkeypatch.setattr(artifacts, "_RENAMEAT2", lambda *args: -1)
    monkeypatch.setattr(artifacts.ctypes, "get_errno", lambda: error_number)

    with pytest.raises(expected_error, match=message):
        artifacts._rename_descriptor_entry_noreplace(
            11,
            "source",
            12,
            "target",
            label="test cleanup rename",
        )


def test_quarantine_restore_refuses_missing_changed_and_occupied_names(
    tmp_path: Path,
) -> None:
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        missing = artifacts._best_effort_restore_quarantined_entry(
            parent_descriptor,
            "missing-quarantine",
            "original",
            preserved_identity=(1, 2),
            label="missing cleanup",
        )

        changed_quarantine = tmp_path / "changed-quarantine"
        changed_quarantine.write_bytes(b"foreign")
        changed = artifacts._best_effort_restore_quarantined_entry(
            parent_descriptor,
            changed_quarantine.name,
            "changed-original",
            preserved_identity=(3, 4),
            label="changed cleanup",
        )

        occupied_quarantine = tmp_path / "occupied-quarantine"
        occupied_original = tmp_path / "occupied-original"
        occupied_quarantine.write_bytes(b"owned")
        occupied_original.write_bytes(b"replacement")
        occupied_metadata = occupied_quarantine.stat()
        occupied = artifacts._best_effort_restore_quarantined_entry(
            parent_descriptor,
            occupied_quarantine.name,
            occupied_original.name,
            preserved_identity=(
                occupied_metadata.st_dev,
                occupied_metadata.st_ino,
            ),
            label="occupied cleanup",
        )
    finally:
        os.close(parent_descriptor)

    assert "disappeared before restoration" in missing
    assert "quarantine changed inode" in changed
    assert changed_quarantine.read_bytes() == b"foreign"
    assert "original name is occupied" in occupied
    assert occupied_quarantine.read_bytes() == b"owned"
    assert occupied_original.read_bytes() == b"replacement"


def test_cleanup_quarantine_detects_disappearance_after_successful_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = tmp_path / "owned.bin"
    displaced = tmp_path / "owned-displaced.bin"
    owned.write_bytes(b"owned bytes")
    metadata = owned.stat()
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )

    def move_without_creating_quarantine(*args: object, **kwargs: object) -> None:
        del args, kwargs
        owned.rename(displaced)

    monkeypatch.setattr(
        artifacts,
        "_rename_descriptor_entry_noreplace",
        move_without_creating_quarantine,
    )
    try:
        with pytest.raises(RuntimeError, match="quarantine entry disappeared"):
            artifacts._quarantine_descriptor_entry(
                parent_descriptor,
                owned.name,
                expected_identity=(metadata.st_dev, metadata.st_ino),
                label="disappearing cleanup",
            )
    finally:
        os.close(parent_descriptor)

    assert displaced.read_bytes() == b"owned bytes"


def _assert_descriptor_is_closed(
    descriptor: int,
    fstat: Any = os.fstat,
) -> None:
    with pytest.raises(OSError) as raised:
        fstat(descriptor)
    assert raised.value.errno == errno.EBADF


def test_directory_tree_hash_preserves_primary_when_root_close_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    primary = KeyboardInterrupt("forced directory hash failure")
    close_failure = SystemExit("forced directory root close failure")
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    opened_descriptor: int | None = None

    def track_open(*args: Any, **kwargs: Any) -> int:
        nonlocal opened_descriptor
        opened_descriptor = real_open(*args, **kwargs)
        return opened_descriptor

    def fail_hash(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise primary

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == opened_descriptor:
            raise close_failure

    monkeypatch.setattr(artifacts.os, "open", track_open)
    monkeypatch.setattr(artifacts.os, "close", close_then_fail)
    monkeypatch.setattr(artifacts, "_directory_descriptor_tree_sha256", fail_hash)

    with pytest.raises(KeyboardInterrupt) as raised:
        artifacts.directory_tree_sha256(tree)

    assert raised.value is primary
    assert opened_descriptor is not None
    _assert_descriptor_is_closed(opened_descriptor)
    assert "forced directory root close failure" in "\n".join(primary.__notes__)


@pytest.mark.parametrize("entry_kind", ["directory", "file"])
def test_directory_tree_hash_closes_child_without_replacing_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    member = tree / "member"
    if entry_kind == "directory":
        member.mkdir()
    else:
        member.write_bytes(b"member")
    primary = KeyboardInterrupt(f"forced {entry_kind} validation failure")
    close_failure = SystemExit(f"forced {entry_kind} close failure")
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    real_fstat = artifacts.os.fstat
    child_descriptor: int | None = None

    def track_child_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal child_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == member.name and dir_fd is not None:
            child_descriptor = descriptor
        return descriptor

    def fail_child_fstat(descriptor: int) -> os.stat_result:
        if descriptor == child_descriptor:
            raise primary
        return real_fstat(descriptor)

    def close_child_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == child_descriptor:
            raise close_failure

    monkeypatch.setattr(artifacts.os, "open", track_child_open)
    monkeypatch.setattr(artifacts.os, "fstat", fail_child_fstat)
    monkeypatch.setattr(artifacts.os, "close", close_child_then_fail)

    with pytest.raises(KeyboardInterrupt) as raised:
        artifacts.directory_tree_sha256(tree)

    assert raised.value is primary
    assert child_descriptor is not None
    _assert_descriptor_is_closed(child_descriptor)
    assert f"forced {entry_kind} close failure" in "\n".join(primary.__notes__)


def test_directory_copy_closes_both_children_and_keeps_operation_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "nested").mkdir(parents=True)
    target.mkdir()
    _seal_directory_tree(source)
    source_descriptor = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    target_descriptor = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    real_fsync = artifacts.os.fsync
    children: dict[str, int] = {}
    close_calls: list[int] = []
    primary = KeyboardInterrupt("forced directory copy failure")
    close_failure = SystemExit("forced target child close failure")

    def track_children(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "nested" and dir_fd == source_descriptor:
            children["source"] = descriptor
        elif path == "nested" and dir_fd == target_descriptor:
            children["target"] = descriptor
        return descriptor

    def fail_target_sync(descriptor: int) -> None:
        if descriptor == children.get("target"):
            raise primary
        real_fsync(descriptor)

    def close_children(descriptor: int) -> None:
        if descriptor in children.values():
            close_calls.append(descriptor)
        real_close(descriptor)
        if descriptor == children.get("target"):
            raise close_failure

    monkeypatch.setattr(artifacts.os, "open", track_children)
    monkeypatch.setattr(artifacts.os, "fsync", fail_target_sync)
    monkeypatch.setattr(artifacts.os, "close", close_children)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            artifacts._copy_directory_descriptor_tree(
                source_descriptor,
                target_descriptor,
                label="sidecar",
            )
    finally:
        real_close(target_descriptor)
        real_close(source_descriptor)

    assert raised.value is primary
    assert set(children) == {"source", "target"}
    for descriptor in children.values():
        assert close_calls.count(descriptor) >= 1
        _assert_descriptor_is_closed(descriptor)
    assert "forced target child close failure" in "\n".join(primary.__notes__)


def test_mount_walk_preserves_primary_and_closes_child_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    member = root / "member.bin"
    member.write_bytes(b"member")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    expected_mount_id = artifacts._descriptor_mount_id(descriptor)
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    real_mount_id = artifacts._descriptor_mount_id
    child_descriptor: int | None = None
    primary = KeyboardInterrupt("forced mount traversal failure")
    close_failure = SystemExit("forced mount child close failure")

    def track_child_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal child_descriptor
        child_descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        return child_descriptor

    def fail_child_mount_id(candidate: int) -> int:
        if candidate == child_descriptor:
            raise primary
        return real_mount_id(candidate)

    def close_child_then_fail(candidate: int) -> None:
        real_close(candidate)
        if candidate == child_descriptor:
            raise close_failure

    monkeypatch.setattr(artifacts.os, "open", track_child_open)
    monkeypatch.setattr(artifacts, "_descriptor_mount_id", fail_child_mount_id)
    monkeypatch.setattr(artifacts.os, "close", close_child_then_fail)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            artifacts._require_directory_tree_mount_id(
                descriptor,
                expected_mount_id=expected_mount_id,
                label="Existing sidecar",
            )
    finally:
        real_close(descriptor)

    assert raised.value is primary
    assert child_descriptor is not None
    _assert_descriptor_is_closed(child_descriptor)
    assert "forced mount child close failure" in "\n".join(primary.__notes__)


def test_recursive_cleanup_runs_later_siblings_after_fatal_child_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    first = root / "a-first.bin"
    second = root / "b-second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    expected_mount_id = artifacts._descriptor_mount_id(descriptor)
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    real_remove = artifacts._remove_descriptor_entry
    opened_children: dict[str, int] = {}
    active_children: dict[int, str] = {}
    closed_children: list[str] = []
    primary = KeyboardInterrupt("forced first child removal failure")
    close_failure = SystemExit("forced first child close failure")
    removal_started = False

    def track_child_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        candidate = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd == descriptor:
            child_name = os.fspath(path)
            opened_children[child_name] = candidate
            active_children[candidate] = child_name
        return candidate

    def fail_first_removal(
        parent_descriptor: int,
        entry_name: str,
        **kwargs: Any,
    ) -> None:
        nonlocal removal_started
        if entry_name == first.name:
            removal_started = True
            raise primary
        real_remove(parent_descriptor, entry_name, **kwargs)

    def close_children(candidate: int) -> None:
        child_name = active_children.pop(candidate, None)
        if child_name is not None:
            closed_children.append(child_name)
        real_close(candidate)
        if child_name == first.name and removal_started:
            raise close_failure

    monkeypatch.setattr(artifacts.os, "open", track_child_open)
    monkeypatch.setattr(artifacts.os, "close", close_children)
    monkeypatch.setattr(artifacts, "_remove_descriptor_entry", fail_first_removal)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            artifacts._remove_directory_descriptor_contents(
                descriptor,
                expected_mount_id=expected_mount_id,
                label="test tree",
            )
    finally:
        real_close(descriptor)

    assert raised.value is primary
    assert first.read_bytes() == b"first"
    assert not second.exists()
    assert set(opened_children) == {first.name, second.name}
    assert closed_children.count(first.name) >= 1
    assert closed_children.count(second.name) >= 1
    for child_descriptor in set(opened_children.values()):
        _assert_descriptor_is_closed(child_descriptor)
    assert "forced first child close failure" in "\n".join(primary.__notes__)
    assert not any(root.glob(".joint-rigger.cleanup-*"))


def test_descriptor_restore_post_create_open_fatal_preserves_name_with_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    target.write_bytes(trusted_bytes)
    target.chmod(0o444)
    parent = artifacts._open_bound_directory(tmp_path)
    source_contract_descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    target_metadata = target.stat()
    source_sha256 = hashlib.sha256(trusted_bytes).hexdigest()
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            staged,
            target,
            "generated root",
            source_descriptor=source_contract_descriptor,
            source_sha256=source_sha256,
        ),
        staged_entry=artifacts._BoundEntry(parent, staged.name),
        target_entry=artifacts._BoundEntry(parent, target.name),
        descriptor_source=artifacts._BoundDescriptorSource(
            descriptor=source_contract_descriptor,
            identity=(target_metadata.st_dev, target_metadata.st_ino),
            sha256=source_sha256,
            mode=0o400,
            is_directory=False,
        ),
    )
    detached_target = artifacts._DetachedTarget(
        identity=(target_metadata.st_dev, target_metadata.st_ino),
        sha256=source_sha256,
        mode=0o444,
        is_directory=False,
    )
    real_open = artifacts.os.open
    hidden_descriptor: int | None = None
    expected = KeyboardInterrupt("forced post-create restoration open fatal")

    def create_then_interrupt(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal hidden_descriptor
        if path == staged.name and flags & os.O_CREAT:
            hidden_descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            raise expected
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", create_then_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            artifacts._restore_descriptor_source_name(
                bound_artifact,
                detached_target,
            )
    finally:
        if hidden_descriptor is not None:
            os.close(hidden_descriptor)
        os.close(source_contract_descriptor)
        os.close(parent.descriptor)

    assert raised.value is expected
    assert staged.is_file()
    assert "unpredictable private name" in "\n".join(raised.value.__notes__)
    staged.unlink()


def test_descriptor_restore_prefers_exact_fatal_after_all_close_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    trusted_bytes = b"trusted descriptor bytes"
    target.write_bytes(trusted_bytes)
    target.chmod(0o444)
    parent = artifacts._open_bound_directory(tmp_path)
    source_contract_descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    target_metadata = target.stat()
    source_sha256 = hashlib.sha256(trusted_bytes).hexdigest()
    bound_artifact = artifacts._BoundArtifact(
        artifact=StagedArtifact(
            staged,
            target,
            "generated root",
            source_descriptor=source_contract_descriptor,
            source_sha256=source_sha256,
        ),
        staged_entry=artifacts._BoundEntry(parent, staged.name),
        target_entry=artifacts._BoundEntry(parent, target.name),
        descriptor_source=artifacts._BoundDescriptorSource(
            descriptor=source_contract_descriptor,
            identity=(target_metadata.st_dev, target_metadata.st_ino),
            sha256=source_sha256,
            mode=0o400,
            is_directory=False,
        ),
    )
    detached_target = artifacts._DetachedTarget(
        identity=(target_metadata.st_dev, target_metadata.st_ino),
        sha256=source_sha256,
        mode=0o444,
        is_directory=False,
    )
    real_open = artifacts.os.open
    real_close = artifacts.os.close
    restore_source: int | None = None
    restore_target: int | None = None
    close_calls: list[int] = []
    ordinary_close = OSError(errno.EIO, "forced restore target close failure")
    fatal_close = SystemExit("forced restore source close failure")

    def track_restore_opens(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal restore_source, restore_target
        candidate = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == target.name and flags & os.O_ACCMODE == os.O_RDONLY:
            restore_source = candidate
        elif path == staged.name and flags & os.O_CREAT:
            restore_target = candidate
        return candidate

    def close_restore_descriptors(candidate: int) -> None:
        if candidate in {restore_source, restore_target}:
            close_calls.append(candidate)
        real_close(candidate)
        if candidate == restore_target:
            raise ordinary_close
        if candidate == restore_source:
            raise fatal_close

    monkeypatch.setattr(artifacts.os, "open", track_restore_opens)
    monkeypatch.setattr(artifacts.os, "close", close_restore_descriptors)
    try:
        with pytest.raises(SystemExit) as raised:
            artifacts._restore_descriptor_source_name(
                bound_artifact,
                detached_target,
            )
    finally:
        real_close(source_contract_descriptor)
        real_close(parent.descriptor)

    assert raised.value is fatal_close
    assert restore_source is not None
    assert restore_target is not None
    for descriptor in (restore_source, restore_target):
        assert close_calls.count(descriptor) == 1
        _assert_descriptor_is_closed(descriptor)
    assert "forced restore target close failure" in "\n".join(fatal_close.__notes__)
    assert staged.read_bytes() == trusted_bytes


def test_require_present_invariant_returns_value_and_rejects_none() -> None:
    marker = object()

    assert artifacts._require_present_invariant(marker, label="test marker") is marker
    with pytest.raises(RuntimeError, match="test marker"):
        artifacts._require_present_invariant(None, label="test marker")


def test_captured_target_tree_depth_budget_fails_before_deep_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "captured"
    nested = root / "one" / "two"
    nested.mkdir(parents=True)
    monkeypatch.setattr(artifacts, "_ARTIFACT_TREE_MAX_DEPTH", 1)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(RuntimeError, match="artifact-tree depth limit.*one/two"):
            artifacts._collect_captured_target_tree_state(
                descriptor,
                relative_directory=".",
                expected_mount_id=artifacts._descriptor_mount_id(descriptor),
                label="depth-limited target",
                entries=[],
            )
    finally:
        os.close(descriptor)


def test_directory_tree_hash_entry_budget_is_shared_across_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "hashed"
    root.mkdir()
    (root / "first.bin").write_bytes(b"first")
    (root / "second.bin").write_bytes(b"second")
    monkeypatch.setattr(artifacts, "_ARTIFACT_TREE_MAX_ENTRIES", 2)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(RuntimeError, match="artifact-tree entry limit"):
            artifacts._collect_directory_tree_entries(
                descriptor,
                relative_directory=".",
                label="entry-limited tree",
                require_no_write_bits=False,
                expected_mount_id=artifacts._descriptor_mount_id(descriptor),
                entries=[],
            )
    finally:
        os.close(descriptor)


def test_directory_tree_copy_byte_budget_leaves_private_target_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    payload = source / "payload.bin"
    payload.write_bytes(b"four")
    payload.chmod(0o400)
    source.chmod(0o500)
    monkeypatch.setattr(artifacts, "_ARTIFACT_TREE_MAX_BYTES", 3)
    source_descriptor = os.open(
        source,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    target_descriptor = os.open(
        target,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        with pytest.raises(RuntimeError, match="artifact-tree byte limit"):
            artifacts._copy_directory_descriptor_tree(
                source_descriptor,
                target_descriptor,
                label="byte-limited tree",
            )
    finally:
        os.close(target_descriptor)
        os.close(source_descriptor)
    assert list(target.iterdir()) == []


def test_directory_tree_copy_rejects_growth_between_stat_and_descriptor_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    payload = source / "payload.bin"
    payload.write_bytes(b"four")
    payload.chmod(0o400)
    source.chmod(0o500)
    monkeypatch.setattr(artifacts, "_ARTIFACT_TREE_MAX_BYTES", 4)
    source_descriptor = os.open(
        source,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    target_descriptor = os.open(
        target,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    real_open = artifacts.os.open
    source_file_opens = 0

    def grow_before_copy_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal source_file_opens
        if path == payload.name and dir_fd == source_descriptor:
            source_file_opens += 1
            # The first open belongs to the mount-boundary preflight. Grow the
            # same inode after the copy traversal has charged its earlier stat.
            if source_file_opens == 2:
                payload.chmod(0o600)
                payload.write_bytes(b"x" * 1024)
                payload.chmod(0o400)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts.os, "open", grow_before_copy_open)
    try:
        with pytest.raises(RuntimeError, match="tree changed inode"):
            artifacts.copy_directory_descriptor_tree(
                source_descriptor,
                target_descriptor,
                label="raced byte-limited tree",
            )
    finally:
        os.close(target_descriptor)
        os.close(source_descriptor)

    assert source_file_opens == 2
    assert list(target.iterdir()) == []


def test_captured_tree_hash_reads_only_accounted_state_after_same_inode_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "captured"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"four")
    monkeypatch.setattr(artifacts, "_ARTIFACT_TREE_MAX_BYTES", 4)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    payload_identity = (payload.stat().st_dev, payload.stat().st_ino)
    real_pread = artifacts.os.pread
    read_lengths: list[int] = []
    grew = False

    def grow_during_hash(file_descriptor: int, length: int, offset: int) -> bytes:
        nonlocal grew
        metadata = os.fstat(file_descriptor)
        if (metadata.st_dev, metadata.st_ino) == payload_identity:
            if not grew and offset == 0:
                grew = True
                payload.write_bytes(b"x" * 1024)
            content = real_pread(file_descriptor, length, offset)
            read_lengths.append(len(content))
            return content
        return real_pread(file_descriptor, length, offset)

    monkeypatch.setattr(artifacts.os, "pread", grow_during_hash)
    budget = artifacts._ArtifactTreeTraversalBudget(label="raced captured tree")
    try:
        with pytest.raises(RuntimeError, match="changed while it was hashed"):
            artifacts._collect_captured_target_tree_state(
                descriptor,
                relative_directory=".",
                expected_mount_id=artifacts._descriptor_mount_id(descriptor),
                label="raced captured target",
                entries=[],
                traversal_budget=budget,
            )
    finally:
        os.close(descriptor)

    assert grew
    assert budget.total_bytes == 4
    assert read_lengths == [4, 1]


def test_cleanup_tree_budget_rejects_before_quarantine_or_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "owned-tree"
    target.mkdir()
    (target / "first.bin").write_bytes(b"first")
    (target / "second.bin").write_bytes(b"second")
    metadata = target.stat()
    monkeypatch.setattr(artifacts, "_ARTIFACT_TREE_MAX_ENTRIES", 2)
    parent = artifacts._open_bound_directory(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="artifact-tree entry limit"):
            artifacts._remove_descriptor_entry(
                parent.descriptor,
                target.name,
                expected_identity=(metadata.st_dev, metadata.st_ino),
                label="budget-limited cleanup",
            )
    finally:
        os.close(parent.descriptor)
    assert (target / "first.bin").read_bytes() == b"first"
    assert (target / "second.bin").read_bytes() == b"second"
    assert not any(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_cleanup_validates_symlink_mount_before_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "owned-tree"
    target.mkdir()
    victim = tmp_path / "external-victim"
    victim.write_bytes(b"external")
    (target / "link").symlink_to(victim)
    metadata = target.stat()
    parent = artifacts._open_bound_directory(tmp_path)
    expected_mount_id = artifacts._descriptor_mount_id(parent.descriptor)
    real_mount_id = artifacts._descriptor_mount_id
    real_quarantine = artifacts._quarantine_descriptor_entry
    quarantine_called = False

    def report_nested_symlink_mount(descriptor: int) -> int:
        if stat.S_ISLNK(os.fstat(descriptor).st_mode):
            return expected_mount_id + 1
        return real_mount_id(descriptor)

    def track_quarantine(*args: Any, **kwargs: Any) -> Any:
        nonlocal quarantine_called
        quarantine_called = True
        return real_quarantine(*args, **kwargs)

    monkeypatch.setattr(artifacts, "_descriptor_mount_id", report_nested_symlink_mount)
    monkeypatch.setattr(artifacts, "_quarantine_descriptor_entry", track_quarantine)
    try:
        with pytest.raises(ValueError, match="contains a mount point at link"):
            artifacts._remove_descriptor_entry(
                parent.descriptor,
                target.name,
                expected_identity=(metadata.st_dev, metadata.st_ino),
                label="mount-limited cleanup",
            )
    finally:
        os.close(parent.descriptor)

    assert not quarantine_called
    assert (target / "link").is_symlink()
    assert victim.read_bytes() == b"external"


def test_bounded_directory_name_collection_stops_at_limit_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yielded: list[int] = []
    closed = False

    class FakeEntry:
        def __init__(self, index: int) -> None:
            self.name = f"entry-{index}"

    class FakeScan:
        def __enter__(self) -> FakeScan:
            return self

        def __exit__(self, *_args: object) -> None:
            nonlocal closed
            closed = True

        def __iter__(self) -> Any:
            for index in range(1_000_000):
                yielded.append(index)
                yield FakeEntry(index)

    monkeypatch.setattr(artifacts.os, "scandir", lambda _descriptor: FakeScan())

    with pytest.raises(RuntimeError, match="bounded scan overflow"):
        artifacts._bounded_sorted_directory_names(
            123,
            maximum_names=2,
            overflow_message="bounded scan overflow",
        )

    assert yielded == [0, 1, 2]
    assert closed


def test_sidecar_entry_validation_enforces_shared_depth_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sidecar"
    (root / "one" / "two").mkdir(parents=True)
    monkeypatch.setattr(artifacts, "_ARTIFACT_TREE_MAX_DEPTH", 1)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(RuntimeError, match="artifact-tree depth limit.*one/two"):
            artifacts._require_sidecar_tree_entries(
                descriptor,
                expected_mount_id=artifacts._descriptor_mount_id(descriptor),
                label="depth-limited sidecar",
            )
    finally:
        os.close(descriptor)


def test_staging_tree_seal_rejects_entry_budget_before_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "staged-sidecar"
    root.mkdir(mode=0o700)
    (root / "first.bin").write_bytes(b"first")
    (root / "second.bin").write_bytes(b"second")
    monkeypatch.setattr(artifacts, "_ARTIFACT_TREE_MAX_ENTRIES", 2)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(RuntimeError, match="artifact-tree entry limit"):
            artifacts._seal_staging_directory_descriptor_tree(
                descriptor,
                expected_mount_id=artifacts._descriptor_mount_id(descriptor),
                label="entry-limited sidecar",
                traversal_budget=artifacts._ArtifactTreeTraversalBudget(
                    label="entry-limited sidecar"
                ),
            )
    finally:
        os.close(descriptor)

    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_directory_copy_accounts_child_before_creating_over_budget_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "a" / "nested").mkdir(parents=True)
    (source / "b").mkdir()
    target.mkdir()
    for directory in (source / "a" / "nested", source / "a", source / "b", source):
        directory.chmod(0o500)
    monkeypatch.setattr(artifacts, "_ARTIFACT_TREE_MAX_ENTRIES", 3)
    source_descriptor = os.open(
        source,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    target_descriptor = os.open(
        target,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    budget = artifacts._ArtifactTreeTraversalBudget(label="entry-limited copy")
    try:
        with pytest.raises(RuntimeError, match="artifact-tree entry limit.*b"):
            artifacts._copy_directory_descriptor_tree(
                source_descriptor,
                target_descriptor,
                label="entry-limited tree",
                traversal_budget=budget,
            )
    finally:
        os.close(target_descriptor)
        os.close(source_descriptor)

    assert (target / "a" / "nested").is_dir()
    assert not (target / "b").exists()
    assert budget.entries == 3


def test_recursive_cleanup_budget_counts_each_entry_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "owned-tree"
    nested = target / "nested"
    nested.mkdir(parents=True)
    (nested / "payload.bin").write_bytes(b"payload")
    metadata = target.stat()
    monkeypatch.setattr(artifacts, "_ARTIFACT_TREE_MAX_ENTRIES", 3)
    parent = artifacts._open_bound_directory(tmp_path)
    try:
        artifacts._remove_descriptor_entry(
            parent.descriptor,
            target.name,
            expected_identity=(metadata.st_dev, metadata.st_ino),
            label="exact-budget cleanup",
        )
    finally:
        os.close(parent.descriptor)

    assert not target.exists()


def test_captured_target_handle_close_is_idempotent() -> None:
    read_descriptor, write_descriptor = os.pipe()
    handle = artifacts._CapturedTargetHandle(read_descriptor)
    try:
        handle.close()
        handle.close()

        assert handle.closed
        assert handle.descriptor == -1
        with pytest.raises(OSError) as closed_descriptor:
            os.fstat(read_descriptor)
        assert closed_descriptor.value.errno == errno.EBADF
    finally:
        os.close(write_descriptor)


def test_initial_target_lookup_distinguishes_untracked_and_missing_states(
    tmp_path: Path,
) -> None:
    targets = _targets(tmp_path)
    untracked = artifacts.StagedJointRiggerArtifacts(
        final_targets=targets,
        staged_targets=targets,
    )
    assert artifacts._initial_target_state(untracked, targets.output_path) is None

    unrelated_state = artifacts._CapturedTargetState(
        requested_path=artifacts._absolute_lexical_path(tmp_path / "other.usda"),
        parent_identity=(1, 2),
        entry_state=None,
    )
    assert unrelated_state.entry_identity is None
    tracked = artifacts.StagedJointRiggerArtifacts(
        final_targets=targets,
        staged_targets=targets,
        _initial_target_states=(unrelated_state,),
    )
    with pytest.raises(
        RuntimeError, match="Initial publication target state is missing"
    ):
        artifacts._initial_target_state(tracked, targets.output_path)


def test_captured_target_tree_collects_nested_directory_and_rejects_fifo(
    tmp_path: Path,
) -> None:
    root = tmp_path / "captured"
    nested = root / "nested"
    nested.mkdir(parents=True)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        entries: list[dict[str, str | int | list[int]]] = []
        artifacts._collect_captured_target_tree_state(
            descriptor,
            relative_directory=".",
            expected_mount_id=artifacts._descriptor_mount_id(descriptor),
            label="nested target",
            entries=entries,
        )
        assert [entry["path"] for entry in entries] == [".", "nested"]
    finally:
        os.close(descriptor)

    os.mkfifo(root / "special")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    special_budget = artifacts._ArtifactTreeTraversalBudget(
        label="captured special-entry coverage"
    )
    try:
        with pytest.raises(ValueError, match="contains a special file: special"):
            artifacts._collect_captured_target_tree_state(
                descriptor,
                relative_directory=".",
                expected_mount_id=artifacts._descriptor_mount_id(descriptor),
                label="special target",
                entries=[],
                traversal_budget=special_budget,
            )
    finally:
        os.close(descriptor)
    # Root, the pre-existing nested directory, and the rejected FIFO were all
    # charged before the special-entry failure.
    assert special_budget.entries == 3


def test_directory_descriptor_copy_rejects_special_entry_after_budget_charge(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    os.mkfifo(source / "special")
    source.chmod(0o555)
    source_descriptor = os.open(
        source,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    target_descriptor = os.open(
        target,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    special_budget = artifacts._ArtifactTreeTraversalBudget(
        label="copied special-entry coverage"
    )
    try:
        with pytest.raises(RuntimeError, match="contains a special entry: special"):
            artifacts._copy_directory_descriptor_tree(
                source_descriptor,
                target_descriptor,
                label="special tree",
                traversal_budget=special_budget,
            )
    finally:
        os.close(target_descriptor)
        os.close(source_descriptor)
        source.chmod(0o700)
        (source / "special").unlink()

    assert list(target.iterdir()) == []
    assert special_budget.entries == 2


def test_commit_point_move_records_rollback_when_backup_recheck_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged.usda"
    target = tmp_path / "target.usda"
    staged.write_text("new artifact", encoding="utf-8")
    real_replace = artifacts._replace_entry
    moved = False

    def move_then_fail(source: Any, destination: Any) -> None:
        nonlocal moved
        real_replace(source, destination)
        moved = True
        raise RuntimeError("forced interruption after commit-point move")

    def reject_moved_commit_point(backups: Any) -> None:
        if moved:
            raise RuntimeError("forced post-move backup invariant failure")

    monkeypatch.setattr(artifacts, "_replace_entry", move_then_fail)
    monkeypatch.setattr(
        artifacts,
        "_require_artifact_backups_unchanged",
        reject_moved_commit_point,
    )

    with pytest.raises(RuntimeError, match="post-move backup invariant failure"):
        promote_staged_artifacts([StagedArtifact(staged, target, "commit point")])

    assert moved
    assert not target.exists()
