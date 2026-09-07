# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Descriptor-confinement regressions for durable artifact storage."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import BinaryIO

import pytest

from world_understanding.utils import artifacts as artifact_utils
from world_understanding.utils.artifacts import (
    confined_directory_identity,
    copy_open_file_to_confined,
    is_pipeline_temp_path,
    open_confined_directory,
    open_confined_directory_at,
    open_confined_regular_file_leaf,
    open_regular_file_no_follow,
    remove_confined_tree,
    write_bytes_to_confined,
)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX directory descriptors")
def test_confined_directory_identity_matches_held_descriptor(tmp_path: Path) -> None:
    with open_confined_directory(tmp_path) as descriptor:
        metadata = os.fstat(descriptor)
        assert confined_directory_identity(descriptor) == (
            metadata.st_dev,
            metadata.st_ino,
        )


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX descriptors")
def test_confined_directory_identity_rejects_regular_file(tmp_path: Path) -> None:
    regular = tmp_path / "artifact.bin"
    regular.write_bytes(b"payload")
    descriptor = os.open(regular, os.O_RDONLY)
    try:
        with pytest.raises(artifact_utils.ArtifactPathError, match="not a directory"):
            confined_directory_identity(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "path",
    [
        ".Pipeline_Temp/config.yaml",
        "cache/.PIPELINE_TEMP/credentials.json",
        Path("/sessions/one/.pipeline_TEMP/result.json"),
        r"cache\.PiPeLiNe_TeMp\config.yaml",
    ],
)
def test_pipeline_temp_path_detection_casefolds_components(path: str | Path) -> None:
    assert is_pipeline_temp_path(path)


@pytest.mark.parametrize(
    "leaf_name", [".pipeline_temp", r"a\request.json", "C:req.json"]
)
def test_confined_host_leaf_accepts_linux_filenames(
    tmp_path: Path,
    leaf_name: str,
) -> None:
    source = tmp_path / leaf_name
    source.write_bytes(b"request")

    with open_confined_directory(tmp_path) as parent_descriptor:
        with open_confined_regular_file_leaf(
            parent_descriptor,
            leaf_name,
        ) as (stream, metadata):
            assert metadata.st_size == len(b"request")
            assert stream.read() == b"request"


@pytest.mark.parametrize(
    "leaf_name",
    ["", ".", "..", "../secret", "nested/leaf", "a\x00b", Path("leaf")],
)
def test_confined_host_leaf_rejects_traversal_and_non_text(
    tmp_path: Path,
    leaf_name: object,
) -> None:
    (tmp_path.parent / "secret").write_bytes(b"outside")

    with open_confined_directory(tmp_path) as parent_descriptor:
        with pytest.raises(ValueError, match="one exact host filename"):
            with open_confined_regular_file_leaf(parent_descriptor, leaf_name):  # type: ignore[arg-type]
                raise AssertionError("unsafe leaf must not be opened")


def test_confined_host_leaf_rejects_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-request"
    outside.write_bytes(b"outside")
    (tmp_path / "request-link").symlink_to(outside)

    with open_confined_directory(tmp_path) as parent_descriptor:
        with pytest.raises(
            artifact_utils.ArtifactPathError,
            match="symlinked artifact",
        ):
            with open_confined_regular_file_leaf(
                parent_descriptor,
                "request-link",
            ):
                raise AssertionError("symlink must not be opened")


def test_atomic_write_default_mode_honors_process_umask(tmp_path: Path) -> None:
    root = tmp_path / "root"
    previous_umask = os.umask(0o077)
    try:
        with open_confined_directory(root, create=True) as root_descriptor:
            assert write_bytes_to_confined(
                root_descriptor,
                "artifact.bin",
                b"payload",
            )
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE((root / "artifact.bin").stat().st_mode) == 0o600


def test_confined_directory_exclusive_create_rejects_existing_leaf(
    tmp_path: Path,
) -> None:
    with open_confined_directory(tmp_path) as root_descriptor:
        with open_confined_directory_at(
            root_descriptor,
            "tuning/iter_1/evidence",
            create=True,
            exclusive_create=True,
        ):
            pass

        with pytest.raises(FileExistsError):
            with open_confined_directory_at(
                root_descriptor,
                "tuning/iter_1/evidence",
                create=True,
                exclusive_create=True,
            ):
                raise AssertionError("an existing leaf must not be reopened")


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX directory descriptors")
def test_confined_lock_file_exclusive_create_rejects_existing_leaf(
    tmp_path: Path,
) -> None:
    with open_confined_directory(tmp_path) as root_descriptor:
        with artifact_utils.open_confined_lock_file(
            root_descriptor,
            "locks/run.lock",
            exclusive_create=True,
        ) as descriptor:
            os.write(descriptor, b"held")

        with pytest.raises(FileExistsError):
            with artifact_utils.open_confined_lock_file(
                root_descriptor,
                "locks/run.lock",
                exclusive_create=True,
            ):
                raise AssertionError("an existing lock file must not be reopened")


def test_atomic_copy_keeps_held_destination_parent_after_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(b"original")
    destination_root = tmp_path / "destination"
    destination_parent = destination_root / "cache"
    destination_parent.mkdir(parents=True)
    held_parent = destination_root / "cache-held"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside")
    real_copy = artifact_utils.shutil.copyfileobj

    def swap_then_copy(source: BinaryIO, destination: BinaryIO) -> None:
        destination_parent.rename(held_parent)
        destination_parent.symlink_to(outside, target_is_directory=True)
        real_copy(source, destination)

    monkeypatch.setattr(artifact_utils.shutil, "copyfileobj", swap_then_copy)

    with open_regular_file_no_follow(source_path) as (source, metadata):
        with open_confined_directory(destination_root) as destination_descriptor:
            assert copy_open_file_to_confined(
                destination_descriptor,
                "cache/result.bin",
                source,
                metadata,
                overwrite=True,
            )

    assert (held_parent / "result.bin").read_bytes() == b"original"
    assert sentinel.read_bytes() == b"outside"
    assert not (outside / "result.bin").exists()
    assert list(held_parent.glob(".result.bin.*.tmp")) == []


def test_remove_confined_tree_refuses_swapped_symlink_leaf(tmp_path: Path) -> None:
    local_root = tmp_path / "sessions"
    local_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside")
    (local_root / "stale").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot be a symlink"):
        remove_confined_tree(local_root / "stale", local_root)

    assert sentinel.read_bytes() == b"outside"
