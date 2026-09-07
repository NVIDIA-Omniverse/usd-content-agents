# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release coverage for confined artifact iteration and atomic writes."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from world_understanding.utils import artifacts


def test_iter_open_regular_files_recurses_filters_and_holds_selected_file(
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

    opened_files: list[tuple[str, bytes, bool]] = []
    with artifacts.open_confined_directory(tmp_path) as root_descriptor:
        for opened in artifacts.iter_open_regular_files(
            root_descriptor,
            prefix="selected/nested/",
        ):
            opened_files.append(
                (
                    opened.relative_key,
                    opened.stream.read(),
                    stat.S_ISREG(opened.metadata.st_mode),
                )
            )

    assert opened_files == [("selected/nested/result.bin", b"selected", True)]


def test_non_overwrite_atomic_write_publishes_then_preserves_existing_file(
    tmp_path: Path,
) -> None:
    with artifacts.open_confined_directory(tmp_path) as root_descriptor:
        assert artifacts.write_bytes_to_confined(
            root_descriptor,
            "nested/result.bin",
            b"first",
            overwrite=False,
        )
        assert not artifacts.write_bytes_to_confined(
            root_descriptor,
            "nested/result.bin",
            b"second",
            overwrite=False,
        )

    assert (tmp_path / "nested" / "result.bin").read_bytes() == b"first"


def test_atomic_writer_closes_and_removes_temporary_file_when_body_raises(
    tmp_path: Path,
) -> None:
    with artifacts.open_confined_directory(tmp_path) as root_descriptor:
        with pytest.raises(RuntimeError, match="abort publication"):
            with artifacts.confined_atomic_writer(
                root_descriptor,
                "nested/result.bin",
                overwrite=True,
            ) as state:
                assert state.stream is not None
                state.stream.write(b"partial")
                raise RuntimeError("abort publication")

    assert not (tmp_path / "nested" / "result.bin").exists()
    assert list((tmp_path / "nested").glob(".result.bin.*.tmp")) == []
