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
    copy_open_file_to_confined,
    is_pipeline_temp_path,
    open_confined_directory,
    open_regular_file_no_follow,
    remove_confined_tree,
    write_bytes_to_confined,
)


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
