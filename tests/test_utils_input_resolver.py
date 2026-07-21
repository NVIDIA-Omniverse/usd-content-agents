# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for validation input inventory resolution."""

import json
from pathlib import Path

import pytest

from world_understanding.utils.input_resolver import (
    InputResolutionError,
    InputResolver,
    _resolve_base_dir,
    resolve_input_inventory,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def test_resolves_mixed_inputs_by_extension(tmp_path: Path) -> None:
    _touch(tmp_path / "asset.usda")
    _touch(tmp_path / "reference.PNG")
    _touch(tmp_path / "clip.MOV")
    _touch(tmp_path / "bundle" / "camera_a.webp")
    _touch(tmp_path / "bundle" / "nested" / "camera_b.jpg")
    _touch(tmp_path / "bundle" / "notes.txt")

    inventory = resolve_input_inventory(
        [
            "asset.usda",
            "reference.PNG",
            "clip.MOV",
            "bundle",
        ],
        base_dir=tmp_path,
    )

    assert inventory.usd_paths == ((tmp_path / "asset.usda").resolve(),)
    assert inventory.image_paths == ((tmp_path / "reference.PNG").resolve(),)
    assert inventory.video_paths == ((tmp_path / "clip.MOV").resolve(),)
    assert inventory.render_bundle_dirs == ((tmp_path / "bundle").resolve(),)
    assert inventory.render_bundle_image_paths == (
        (tmp_path / "bundle" / "camera_a.webp").resolve(),
        (tmp_path / "bundle" / "nested" / "camera_b.jpg").resolve(),
    )
    assert [item.kind for item in inventory.items] == [
        "usd",
        "image",
        "video",
        "render_bundle",
    ]
    assert [item.extension for item in inventory.items] == [
        ".usda",
        ".png",
        ".mov",
        None,
    ]
    assert inventory.items[-1].extension is None


def test_accepts_bare_input_and_focus_prim_strings(tmp_path: Path) -> None:
    _touch(tmp_path / "asset.usd")

    inventory = resolve_input_inventory(
        "asset.usd",
        base_dir=tmp_path,
        focus_prim_paths="/World/Cart",
    )

    assert inventory.usd_paths == ((tmp_path / "asset.usd").resolve(),)
    assert inventory.focus_prim_paths == ("/World/Cart",)


def test_accepts_path_inputs_and_multiple_items_per_kind(tmp_path: Path) -> None:
    _touch(tmp_path / "asset_a.usd")
    _touch(tmp_path / "asset_b.usdc")
    _touch(tmp_path / "reference_a.jpg")
    _touch(tmp_path / "reference_b.webp")

    inventory = resolve_input_inventory(
        [
            tmp_path / "asset_a.usd",
            tmp_path / "asset_b.usdc",
            tmp_path / "reference_a.jpg",
            tmp_path / "reference_b.webp",
        ],
    )

    assert inventory.usd_paths == (
        (tmp_path / "asset_a.usd").resolve(),
        (tmp_path / "asset_b.usdc").resolve(),
    )
    assert inventory.image_paths == (
        (tmp_path / "reference_a.jpg").resolve(),
        (tmp_path / "reference_b.webp").resolve(),
    )


def test_to_dict_is_json_friendly(tmp_path: Path) -> None:
    _touch(tmp_path / "asset.usd")
    _touch(tmp_path / "bundle" / "camera.png")

    inventory = resolve_input_inventory(
        ["asset.usd", "bundle"],
        base_dir=tmp_path,
        focus_prim_paths=["/World/Cart"],
    )
    data = inventory.to_dict()

    json.dumps(data)

    assert data["usd_paths"] == [str((tmp_path / "asset.usd").resolve())]
    assert data["image_paths"] == []
    assert data["video_paths"] == []
    assert data["render_bundle_dirs"] == [str((tmp_path / "bundle").resolve())]
    assert data["render_bundle_image_paths"] == [
        str((tmp_path / "bundle" / "camera.png").resolve())
    ]
    assert data["focus_prim_paths"] == ["/World/Cart"]
    assert data["working_dir"] is None
    assert data["items"][0]["original"] == "asset.usd"
    assert data["items"][0]["kind"] == "usd"
    assert data["items"][0]["extension"] == ".usd"
    assert data["items"][0]["path"] == str((tmp_path / "asset.usd").resolve())
    assert data["items"][1]["image_paths"] == [
        str((tmp_path / "bundle" / "camera.png").resolve())
    ]
    assert "directories" not in data


def test_missing_input_fails_with_clear_message(tmp_path: Path) -> None:
    with pytest.raises(InputResolutionError, match="Input path does not exist"):
        resolve_input_inventory(["missing.usd"], base_dir=tmp_path)


def test_missing_base_dir_fails_with_clear_message(tmp_path: Path) -> None:
    with pytest.raises(InputResolutionError, match="Base directory does not exist"):
        resolve_input_inventory(["asset.usd"], base_dir=tmp_path / "missing")


def test_base_dir_file_fails_with_clear_message(tmp_path: Path) -> None:
    base_file = _touch(tmp_path / "base.txt")

    with pytest.raises(InputResolutionError, match="Base path is not a directory"):
        resolve_input_inventory(["asset.usd"], base_dir=base_file)


def test_unsupported_existing_file_fails_with_clear_message(tmp_path: Path) -> None:
    _touch(tmp_path / "notes.txt")

    with pytest.raises(InputResolutionError, match="Unsupported input type"):
        resolve_input_inventory(["notes.txt"], base_dir=tmp_path)


def test_file_without_extension_reports_none_extension(tmp_path: Path) -> None:
    _touch(tmp_path / "Makefile")

    with pytest.raises(InputResolutionError, match="extension=<none>"):
        resolve_input_inventory(["Makefile"], base_dir=tmp_path)


def test_directory_without_image_like_contents_fails(tmp_path: Path) -> None:
    _touch(tmp_path / "bundle" / "metadata.json")

    with pytest.raises(
        InputResolutionError,
        match="does not contain image-like files",
    ):
        resolve_input_inventory(["bundle"], base_dir=tmp_path)


def test_render_bundle_scan_skips_symlinks(tmp_path: Path) -> None:
    _touch(tmp_path / "bundle" / "camera.png")
    _touch(tmp_path / "outside" / "outside.png")
    (tmp_path / "bundle" / "linked_file.png").symlink_to(
        tmp_path / "outside" / "outside.png"
    )
    (tmp_path / "bundle" / "linked_dir").symlink_to(
        tmp_path / "outside",
        target_is_directory=True,
    )

    inventory = resolve_input_inventory(["bundle"], base_dir=tmp_path)

    assert inventory.render_bundle_image_paths == (
        (tmp_path / "bundle" / "camera.png").resolve(),
    )


def test_focus_prim_paths_are_passed_through_without_validation(
    tmp_path: Path,
) -> None:
    _touch(tmp_path / "asset.usdz")
    focus_paths = ["/Missing/Prim", "not/a/usd/path", ""]

    inventory = resolve_input_inventory(
        ["asset.usdz"],
        base_dir=tmp_path,
        focus_prim_paths=focus_paths,
    )

    assert inventory.focus_prim_paths == tuple(focus_paths)


def test_working_dir_is_resolved_and_created_when_requested(tmp_path: Path) -> None:
    _touch(tmp_path / "asset.usdc")
    working_dir = tmp_path / ".validation" / "run"

    resolver = InputResolver(
        base_dir=tmp_path,
        working_dir=".validation/run",
        create_working_dir=True,
    )
    inventory = resolver.resolve(["asset.usdc"])

    assert inventory.working_dir == working_dir.resolve()
    assert working_dir.is_dir()


def test_convenience_wrapper_creates_working_dir(tmp_path: Path) -> None:
    _touch(tmp_path / "asset.usd")
    working_dir = tmp_path / ".validation" / "run"

    inventory = resolve_input_inventory(
        ["asset.usd"],
        base_dir=tmp_path,
        working_dir=".validation/run",
        create_working_dir=True,
    )

    assert inventory.working_dir == working_dir.resolve()
    assert working_dir.is_dir()


def test_missing_working_dir_without_create_fails(tmp_path: Path) -> None:
    _touch(tmp_path / "asset.usd")

    resolver = InputResolver(base_dir=tmp_path, working_dir=".validation/run")
    with pytest.raises(InputResolutionError, match="Working directory does not exist"):
        resolver.resolve(["asset.usd"])


def test_working_dir_is_not_created_when_inputs_fail(tmp_path: Path) -> None:
    working_dir = tmp_path / ".validation" / "run"

    resolver = InputResolver(
        base_dir=tmp_path,
        working_dir=".validation/run",
        create_working_dir=True,
    )
    with pytest.raises(InputResolutionError, match="Input path does not exist"):
        resolver.resolve(["missing.usd"])

    assert not working_dir.exists()


def test_working_dir_file_fails(tmp_path: Path) -> None:
    _touch(tmp_path / "asset.usd")
    _touch(tmp_path / "not-a-dir")

    resolver = InputResolver(base_dir=tmp_path, working_dir="not-a-dir")
    with pytest.raises(InputResolutionError, match="Working directory"):
        resolver.resolve(["asset.usd"])


def test_empty_inputs_fail(tmp_path: Path) -> None:
    with pytest.raises(InputResolutionError, match="At least one input path"):
        resolve_input_inventory([], base_dir=tmp_path)


def test_resolve_base_dir_accepts_relative_existing_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.chdir(tmp_path)

    assert _resolve_base_dir("base") == base.resolve()
