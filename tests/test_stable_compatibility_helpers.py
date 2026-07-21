# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for compatibility boundaries around persisted identifiers."""

from pathlib import Path

from world_understanding.utils.compat_hash import legacy_md5_hex
from world_understanding.utils.preview_paths import (
    find_existing_preview_filename,
    legacy_preview_filename_for_render_path,
    preview_filename_for_render_path,
    resolve_preview_filename,
)


def test_legacy_md5_hex_preserves_historical_session_suffixes() -> None:
    assert legacy_md5_hex("path/to/model.usd", length=6) == "aa7621"
    assert legacy_md5_hex("categoryStem", length=6) == "fa504a"


def test_preview_filename_uses_new_hash_for_new_sessions() -> None:
    assert (
        preview_filename_for_render_path("World/mesh_I3_prim_only.png")
        == "146c59b0_mesh_I3_prim_only.png"
    )


def test_preview_resolution_prefers_existing_preferred_and_handles_missing_dir(
    tmp_path: Path,
) -> None:
    preview_dir = tmp_path / "preview"
    preferred = preview_filename_for_render_path("World/mesh_I3_prim_only.png")
    assert (
        find_existing_preview_filename(preview_dir, "World/mesh_I3_prim_only.png")
        is None
    )
    assert (
        resolve_preview_filename(preview_dir, "World/mesh_I3_prim_only.png")
        == preferred
    )

    preview_dir.mkdir()
    (preview_dir / preferred).write_bytes(b"png")

    assert (
        find_existing_preview_filename(preview_dir, "World/mesh_I3_prim_only.png")
        == preferred
    )

    empty_preview_dir = tmp_path / "empty-preview"
    empty_preview_dir.mkdir()
    assert (
        find_existing_preview_filename(empty_preview_dir, "World/mesh_I3_prim_only.png")
        is None
    )


def test_preview_resolution_prefers_existing_legacy_filename(tmp_path: Path) -> None:
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    legacy_name = legacy_preview_filename_for_render_path("World/mesh_I3_prim_only.png")
    assert legacy_name == "f742ef29_mesh_I3_prim_only.png"
    (preview_dir / legacy_name).write_bytes(b"png")

    assert (
        find_existing_preview_filename(
            preview_dir,
            "usd/renders/World/mesh_I3_prim_only.png",
        )
        == legacy_name
    )
    assert (
        resolve_preview_filename(preview_dir, "World/mesh_I3_prim_only.png")
        == legacy_name
    )


def test_preview_resolution_uses_exact_legacy_path_hash(tmp_path: Path) -> None:
    preview_dir = tmp_path / "preview"
    preview_dir.mkdir()
    wrong_same_basename = "00000000_mesh_I3_prim_only.png"
    exact_legacy = legacy_preview_filename_for_render_path(
        "World/mesh_I3_prim_only.png"
    )
    (preview_dir / wrong_same_basename).write_bytes(b"wrong")
    (preview_dir / exact_legacy).write_bytes(b"right")

    assert (
        find_existing_preview_filename(
            preview_dir,
            "usd/renders/World/mesh_I3_prim_only.png",
        )
        == exact_legacy
    )
