# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release coverage for descriptor-confined artifact mutation helpers."""

from __future__ import annotations

from pathlib import Path

from world_understanding.utils import artifacts, public_response


def test_delete_and_append_confined_artifacts(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    deleted = nested / "deleted.bin"
    deleted.write_bytes(b"stale")

    with artifacts.open_confined_directory(tmp_path) as root_descriptor:
        assert artifacts.delete_confined_file(root_descriptor, "nested/deleted.bin")
        artifacts.append_bytes_to_confined(
            root_descriptor,
            "nested/appended.bin",
            b"first",
        )
        artifacts.append_bytes_to_confined(
            root_descriptor,
            "nested/appended.bin",
            b"-second",
        )

    assert not deleted.exists()
    assert (nested / "appended.bin").read_bytes() == b"first-second"


def test_prune_snapshot_skips_other_prefix_and_removes_empty_tree(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "stale.bin").write_bytes(b"stale")
    ignored = tmp_path / "ignored.bin"
    ignored.write_bytes(b"keep")

    with artifacts.open_confined_directory(tmp_path) as root_descriptor:
        artifacts.prune_confined_snapshot(
            root_descriptor,
            "selected/",
            set(),
        )

    assert not selected.exists()
    assert ignored.read_bytes() == b"keep"


def test_remove_missing_confined_tree_is_idempotent(tmp_path: Path) -> None:
    owner = tmp_path / "sessions"
    owner.mkdir()

    assert not artifacts.remove_confined_tree(owner / "missing", owner)


def test_session_uri_rejects_absolute_path_outside_roots(tmp_path: Path) -> None:
    session_root = tmp_path / "sessions"
    outside = tmp_path / "outside" / "result.bin"

    assert public_response._session_uri(str(outside), (session_root,)) is None
