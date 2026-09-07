# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release coverage for descriptor-confined artifact read helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from world_understanding.utils import artifacts


def test_s3_object_suffix_requires_prefix_and_returns_canonical_key() -> None:
    assert (
        artifacts.validated_s3_object_suffix(
            "sessions/run-1/nested/result.bin",
            "sessions/run-1/",
        )
        == "nested/result.bin"
    )

    with pytest.raises(ValueError, match="outside the session key prefix"):
        artifacts.validated_s3_object_suffix(
            "sessions/run-2/result.bin",
            "sessions/run-1/",
        )


def test_missing_confined_directory_and_listing_are_reported_as_absent(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        with artifacts.open_confined_directory(missing):
            raise AssertionError("a missing directory must not be opened")

    assert artifacts.list_confined_artifact_keys(missing) == []


def test_open_confined_directory_at_yields_nested_directory(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "artifacts"
    nested.mkdir(parents=True)

    with artifacts.open_confined_directory(tmp_path) as root_descriptor:
        with artifacts.open_confined_directory_at(
            root_descriptor,
            "nested/artifacts",
        ) as nested_descriptor:
            assert os.path.samefile(f"/proc/self/fd/{nested_descriptor}", nested)


def test_confined_exists_and_read_use_held_regular_file(tmp_path: Path) -> None:
    payload = b"release-artifact"
    (tmp_path / "result.bin").write_bytes(payload)

    assert artifacts.confined_artifact_exists(tmp_path, "result.bin")
    assert not artifacts.confined_artifact_exists(tmp_path, "missing.bin")
    assert artifacts.read_confined_artifact_bytes(tmp_path, "result.bin") == payload


def test_listing_recurses_and_skips_private_or_unselected_entries(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected" / "nested"
    selected.mkdir(parents=True)
    (selected / "result.bin").write_bytes(b"selected")
    private = tmp_path / ".pipeline_temp"
    private.mkdir()
    (private / "secret.bin").write_bytes(b"private")
    unselected = tmp_path / "other"
    unselected.mkdir()
    (unselected / "ignored.bin").write_bytes(b"other")

    assert artifacts.list_confined_artifact_keys(
        tmp_path,
        prefix="selected/",
    ) == ["selected/nested/result.bin"]
