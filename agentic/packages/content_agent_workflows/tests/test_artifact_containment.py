# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
from pathlib import Path

import pytest
from PIL import Image

from content_agent_workflows.common import artifacts as artifact_helpers
from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    contained_regular_file,
    read_contained_artifact,
)


def test_contained_regular_file_accepts_direct_run_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifact = run_dir / "final_renders" / "view.png"
    artifact.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), "red").save(artifact)

    assert (
        contained_regular_file(run_dir, "final_renders/view.png", image=True)
        == artifact.resolve()
    )


def test_contained_regular_file_rejects_direct_and_parent_symlinks(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.png"
    Image.new("RGB", (2, 2), "red").save(secret)

    direct = run_dir / "direct.png"
    direct.symlink_to(secret)
    with pytest.raises(ValueError, match="symlinks"):
        contained_regular_file(run_dir, direct)

    parent = run_dir / "final_renders"
    parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        contained_regular_file(run_dir, parent / "secret.png")


def test_contained_regular_file_rejects_escape_directory_and_invalid_image(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="outside"):
        contained_regular_file(run_dir, outside)

    invalid = run_dir / "invalid.png"
    invalid.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="not decodable"):
        contained_regular_file(run_dir, invalid, image=True)


def test_contained_regular_file_enforces_size_limit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "large.json"
    artifact.write_text("12345", encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        contained_regular_file(run_dir, artifact, max_bytes=4)


def test_read_contained_artifact_derives_json_digest_and_size_from_one_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "record.json"
    original = b'{"value":"original"}\n'
    artifact.write_bytes(original)
    outside = tmp_path / "outside.json"
    outside.write_text('{"value":"outside"}\n', encoding="utf-8")

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
        if path == artifact.name and dir_fd is not None and not swapped:
            swapped = True
            artifact.unlink()
            artifact.symlink_to(outside)
        return file_fd

    monkeypatch.setattr(artifact_helpers.os, "open", open_and_swap)

    result = read_contained_artifact(run_dir, artifact, parse_json=True)

    assert result.json_object == {"value": "original"}
    assert result.sha256 == hashlib.sha256(original).hexdigest()
    assert result.size_bytes == len(original)
    assert artifact.is_symlink()


def test_read_contained_artifact_rejects_in_place_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "record.bin"
    artifact.write_bytes(b"original")

    original_read = artifact_helpers.os.read
    mutated = False

    def read_and_mutate(file_fd: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(file_fd, size)
        if chunk and not mutated:
            mutated = True
            artifact.write_bytes(b"replacement-is-longer")
        return chunk

    monkeypatch.setattr(artifact_helpers.os, "read", read_and_mutate)

    with pytest.raises(ValueError, match="changed while it was being read"):
        read_contained_artifact(run_dir, artifact)


def test_read_contained_artifact_validates_image_bytes_from_open_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "render.png"
    Image.new("RGB", (2, 2), "green").save(artifact)
    original = artifact.read_bytes()
    outside = tmp_path / "outside.png"
    outside.write_text("not an image", encoding="utf-8")

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
        if path == artifact.name and dir_fd is not None and not swapped:
            swapped = True
            artifact.unlink()
            artifact.symlink_to(outside)
        return file_fd

    monkeypatch.setattr(artifact_helpers.os, "open", open_and_swap)

    result = read_contained_artifact(run_dir, artifact, image=True)

    assert result.sha256 == hashlib.sha256(original).hexdigest()
    assert result.size_bytes == len(original)
    assert artifact.is_symlink()


def test_read_contained_artifact_rejects_hard_linked_run_artifact(
    tmp_path: Path,
) -> None:
    # O_NOFOLLOW does not stop a same-UID hard link: a contained path could
    # alias bytes outside the run root. Multi-link artifacts fail closed.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"leaked": true}', encoding="utf-8")
    linked = run_dir / "artifact.json"
    os.link(outside, linked)

    with pytest.raises(ValueError, match="single-link regular file"):
        read_contained_artifact(run_dir, "artifact.json", max_bytes=1024)


def test_read_contained_artifact_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is not available on this platform")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fifo = run_dir / "artifact.pipe"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="not a regular file"):
        read_contained_artifact(run_dir, fifo)


def test_atomic_write_json_refuses_symlinked_parent_within_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (run_dir / "trace").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        atomic_write_json(
            run_dir / "trace" / "operation_trace.json",
            {"must": "stay contained"},
            within=run_dir,
        )

    assert not (outside / "operation_trace.json").exists()
