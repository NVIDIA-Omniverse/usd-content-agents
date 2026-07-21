# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional coverage for material-agent dataset loading."""

from __future__ import annotations

from pathlib import Path

from material_agent.tasks.dataset import DatasetLoadingTask


class _Listener:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message: str) -> None:
        self.debugs.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def test_dataset_loader_validates_supported_image_formats(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("", encoding="utf-8")
    (tmp_path / "front.png").write_bytes(b"image")
    (tmp_path / "side.png").write_bytes(b"image")
    (tmp_path / "detail.png").write_bytes(b"image")

    task = DatasetLoadingTask()
    listener = _Listener()

    assert (
        task._validate_entry(
            {"id": "old-list", "images": ["front.png", "side.png"]},
            dataset_path,
            listener,
        )
        is True
    )
    assert (
        task._validate_entry(
            {"id": "old-single", "image_path": "front.png"},
            dataset_path,
            listener,
        )
        is True
    )
    assert (
        task._validate_entry(
            {"id": "media", "media": {"images": [{"path": "detail.png"}]}},
            dataset_path,
            listener,
        )
        is True
    )


def test_dataset_loader_filters_invalid_entries_and_updates_context(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("", encoding="utf-8")
    (tmp_path / "valid.png").write_bytes(b"image")

    valid_entry = {"id": "valid", "images": ["valid.png"]}
    invalid_entries = [
        {"images": ["valid.png"]},
        {"id": "no-images"},
        {"id": "missing-list", "images": ["missing.png"]},
        {"id": "missing-single", "image_path": "missing.png"},
        {"id": "missing-media", "media": {"images": [{"path": "missing.png"}]}},
        {"id": "invalid-media-object", "media": {"images": ["bad"]}},
    ]

    task = DatasetLoadingTask()
    listener = _Listener()
    task._listener = listener

    for entry in invalid_entries:
        assert task._validate_entry(entry, dataset_path, listener) is False

    filtered = task._validate_dataset([valid_entry, *invalid_entries], dataset_path)
    assert filtered == [valid_entry]
    assert "Entry no-images has no images in any supported format" in listener.debugs
    assert any("Image not found" in warning for warning in listener.warnings)
    assert any(
        "Invalid image object in media.images" in warning
        for warning in listener.warnings
    )
    assert any("Invalid entry: unknown" in warning for warning in listener.warnings)

    context: dict[str, object] = {}
    task._update_context(context, filtered, dataset_path, {"source": "unit"})
    assert context["dataset"] == filtered
    assert context["dataset_path"] == str(dataset_path)
    assert context["dataset_metadata"] == {"source": "unit"}
    assert context["image_base_dir"] == str(tmp_path)
