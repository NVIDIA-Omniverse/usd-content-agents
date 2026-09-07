# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from content_agent_workflows.common import artifacts


def test_file_sha256_reads_binary_payload_verbatim(tmp_path: Path) -> None:
    payload = (
        b"a" * (artifacts._CHUNK_SIZE // 2) + b"\x1a" + b"b" * artifacts._CHUNK_SIZE
    )
    source = tmp_path / "payload.bin"
    source.write_bytes(payload)

    assert artifacts.file_sha256(source) == hashlib.sha256(payload).hexdigest()


def test_open_contained_text_writer_truncates_existing_live_log(
    tmp_path: Path,
) -> None:
    target = tmp_path / "child-output.log"
    target.write_text("stale", encoding="utf-8")

    stream = artifacts.open_contained_text_writer(tmp_path, target)
    try:
        stream.write("live output\n")
        stream.flush()
        assert target.read_text(encoding="utf-8") == "live output\n"
    finally:
        stream.close()


@pytest.mark.skipif(os.name == "posix", reason="non-POSIX confined copy backend")
def test_snapshot_validates_staged_bytes_before_replacing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"new source bytes")
    destination = tmp_path / "snapshots" / "sealed.bin"
    destination.parent.mkdir()
    destination.write_bytes(b"previous sealed bytes")

    with pytest.raises(ValueError, match="digest changed before sealing"):
        artifacts.snapshot_contained_artifact(
            tmp_path,
            source,
            destination,
            expected_sha256=hashlib.sha256(b"different bytes").hexdigest(),
        )

    assert destination.read_bytes() == b"previous sealed bytes"
    assert list(destination.parent.glob(".*.snapshot")) == []


def test_snapshot_supports_a_separate_destination_root(tmp_path: Path) -> None:
    source_root = tmp_path / "provider"
    destination_root = tmp_path / "workflow"
    source_root.mkdir()
    destination_root.mkdir()
    source = source_root / "source.bin"
    source.write_bytes(b"immutable provider bytes")
    destination = destination_root / "closure" / "source.bin"

    snapshot = artifacts.snapshot_contained_artifact(
        source_root,
        source,
        destination,
        destination_run_dir=destination_root,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        expected_size_bytes=source.stat().st_size,
        allow_hardlinks=False,
    )

    assert snapshot.path == destination.resolve()
    assert destination.read_bytes() == source.read_bytes()
    assert list(destination.parent.glob(".*.snapshot")) == []


def test_atomic_write_text_fsyncs_file_then_parent_directory(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    fsync_modes: list[int] = []
    fsynced_directories: list[Path] = []
    real_fsync = os.fsync
    real_fsync_directory = artifacts._fsync_directory

    def record_fsync(file_descriptor: int) -> None:
        fsync_modes.append(os.fstat(file_descriptor).st_mode)
        real_fsync(file_descriptor)

    def record_fsync_directory(path: Path) -> None:
        fsynced_directories.append(path.resolve())
        real_fsync_directory(path)

    monkeypatch.setattr(artifacts.os, "fsync", record_fsync)
    monkeypatch.setattr(artifacts, "_fsync_directory", record_fsync_directory)

    destination = tmp_path / "nested" / "artifact.json"
    destination.parent.mkdir()
    artifacts.atomic_write_text(destination, '{"status": "durable"}\n')

    assert destination.read_text(encoding="utf-8") == '{"status": "durable"}\n'
    if os.name == "posix":
        assert stat.S_ISREG(fsync_modes[-2])
        assert stat.S_ISDIR(fsync_modes[-1])
    else:
        assert stat.S_ISREG(fsync_modes[-1])
        assert fsynced_directories[-1] == destination.parent


def test_atomic_write_text_durably_creates_missing_ancestors(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    fsynced_directories: list[Path] = []
    real_fsync_directory = artifacts._fsync_directory

    def record_fsync_directory(path: Path) -> None:
        fsynced_directories.append(path.resolve())
        real_fsync_directory(path)

    monkeypatch.setattr(
        artifacts,
        "_fsync_directory",
        record_fsync_directory,
    )

    first_parent = tmp_path / "runs"
    second_parent = first_parent / "joint"
    destination = second_parent / "checkpoint.json"
    artifacts.atomic_write_text(destination, '{"phase": "inferring"}\n')

    assert destination.is_file()
    assert fsynced_directories[:3] == [
        second_parent,
        first_parent,
        tmp_path,
    ]
    if os.name == "posix":
        # POSIX finalizes the parent through the pinned directory fd rather
        # than this path-based helper, so the walk to the mount boundary is
        # what this helper records last. The post-write parent sync itself is
        # asserted directly by
        # test_atomic_write_text_fsyncs_file_then_parent_directory.
        assert second_parent in fsynced_directories
    else:  # pragma: win32 cover
        assert fsynced_directories[-1] == second_parent


def test_atomic_write_text_repairs_interrupted_ancestor_sync(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    fsynced_directories: list[Path] = []
    real_fsync_directory = artifacts._fsync_directory
    fail_boundary_sync = True

    def fail_once_at_existing_boundary(path: Path) -> None:
        nonlocal fail_boundary_sync
        resolved = path.resolve()
        fsynced_directories.append(resolved)
        if resolved == tmp_path and fail_boundary_sync:
            fail_boundary_sync = False
            raise OSError("simulated interrupted ancestor sync")
        real_fsync_directory(path)

    monkeypatch.setattr(
        artifacts,
        "_fsync_directory",
        fail_once_at_existing_boundary,
    )

    run_dir = tmp_path / "runs"
    destination = run_dir / "checkpoint.json"
    with pytest.raises(OSError, match="interrupted ancestor sync"):
        artifacts.atomic_write_text(destination, '{"phase": "inferring"}\n')

    assert run_dir.is_dir()
    assert not destination.exists()
    fsynced_directories.clear()

    artifacts.atomic_write_text(destination, '{"phase": "inferring"}\n')

    assert destination.is_file()
    assert fsynced_directories[:2] == [run_dir, tmp_path]


def test_directory_durability_cache_rejects_recreated_directory(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    fsynced_directories: list[Path] = []
    real_fsync_directory = artifacts._fsync_directory

    def record_fsync_directory(path: Path) -> None:
        fsynced_directories.append(path.resolve())
        real_fsync_directory(path)

    monkeypatch.setattr(
        artifacts,
        "_fsync_directory",
        record_fsync_directory,
    )

    run_dir = tmp_path / "reused-run"
    artifacts._create_directory_tree_durably(run_dir)
    run_dir.rmdir()
    fsynced_directories.clear()

    artifacts._create_directory_tree_durably(run_dir)

    assert fsynced_directories[:2] == [run_dir, tmp_path]


@pytest.mark.skipif(os.name != "nt", reason="Windows durability boundary")
def test_directory_durability_stops_at_first_existing_windows_parent(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    fsynced_directories: list[Path] = []
    run_dir = tmp_path / "runs" / "scene"

    monkeypatch.setattr(
        artifacts,
        "_fsync_directory",
        lambda path: fsynced_directories.append(path),
    )

    artifacts._create_directory_tree_durably(run_dir)

    assert fsynced_directories == [run_dir, run_dir.parent, tmp_path]


@pytest.mark.skipif(os.name != "nt", reason="Windows durability boundary")
def test_windows_directory_flush_failure_is_not_suppressed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fail_flush(_path: Path) -> None:
        raise PermissionError("directory flush denied")

    monkeypatch.setattr(artifacts, "fsync_directory", fail_flush)

    with pytest.raises(PermissionError, match="directory flush denied"):
        artifacts._fsync_directory(tmp_path)


def test_directory_durability_cache_retries_recreated_identity_collision(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = tmp_path / "reused-run"
    artifacts._create_directory_tree_durably(run_dir)
    cached_identity = artifacts._DURABLY_SYNCED_DIRECTORY_IDENTITIES[str(run_dir)]
    run_dir.rmdir()

    real_directory_identity = artifacts._directory_identity

    def reuse_cached_identity(path: Path) -> tuple[int, int, int]:
        if path == run_dir:
            return cached_identity
        return real_directory_identity(path)

    fsynced_directories: list[Path] = []
    real_fsync_directory = artifacts._fsync_directory
    fail_boundary_sync = True

    def fail_once_at_existing_boundary(path: Path) -> None:
        nonlocal fail_boundary_sync
        resolved = path.resolve()
        fsynced_directories.append(resolved)
        if resolved == tmp_path and fail_boundary_sync:
            fail_boundary_sync = False
            raise OSError("simulated interrupted recreated-directory sync")
        real_fsync_directory(path)

    monkeypatch.setattr(artifacts, "_directory_identity", reuse_cached_identity)
    monkeypatch.setattr(
        artifacts,
        "_fsync_directory",
        fail_once_at_existing_boundary,
    )

    with pytest.raises(OSError, match="interrupted recreated-directory sync"):
        artifacts._create_directory_tree_durably(run_dir)

    fsynced_directories.clear()
    artifacts._create_directory_tree_durably(run_dir)

    assert fsynced_directories[:2] == [run_dir, tmp_path]
