# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for v0.1 to v0.2 dataset migration helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from world_understanding.agentic.dataset import migrate


class _FakeConfig:
    inference = SimpleNamespace(
        prompts=[SimpleNamespace(system_prompt="system prompt")]
    )

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {"version": "0.2", "mode": mode}


class _FakeEntry:
    media = SimpleNamespace(
        images=["renders/a.png", "renders/b.jpg"],
        reference_images=["reference_image.png"],
    )

    def model_dump_json(self) -> str:
        return '{"id": "entry-1"}'


def _write(path: Path, text: str = "data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_migrate_dataset_writes_v02_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "dataset_v01"
    output_dir = tmp_path / "dataset_v02"
    _write(input_dir / "usd" / "renders" / "a.png")
    _write(input_dir / "usd" / "renders" / "b.jpg")
    _write(input_dir / "usd" / "usd_model.json", "{}")
    _write(input_dir / "reference_image.png")

    monkeypatch.setattr(migrate, "detect_dataset_version", lambda _path: "0.1")
    monkeypatch.setattr(migrate, "load_dataset_config", lambda _path: _FakeConfig())
    monkeypatch.setattr(
        migrate,
        "load_dataset_entries",
        lambda _path, _config: [_FakeEntry()],
    )

    stats = migrate.migrate_dataset(input_dir, output_dir=output_dir)

    assert stats["config_path"] == str(output_dir / "dataset.json")
    assert stats["entries_path"] == str(output_dir / "dataset.jsonl")
    assert stats["num_entries"] == 1
    assert stats["num_images_migrated"] == 2
    assert stats["usd_model_migrated"] is True
    assert stats["num_reference_images_migrated"] == 1
    assert (output_dir / "dataset.json").exists()
    assert (output_dir / "dataset.jsonl").read_text(encoding="utf-8").strip() == (
        '{"id": "entry-1"}'
    )
    assert (output_dir / "renders" / "a.png").exists()
    assert (output_dir / "usd_model.json").exists()
    assert (output_dir / "reference_image.png").exists()


def test_migrate_dataset_dry_run_and_validation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    monkeypatch.setattr(migrate, "detect_dataset_version", lambda _path: "0.1")
    monkeypatch.setattr(migrate, "load_dataset_config", lambda _path: _FakeConfig())
    monkeypatch.setattr(
        migrate,
        "load_dataset_entries",
        lambda _path, _config: [_FakeEntry(), _FakeEntry()],
    )

    stats = migrate.migrate_dataset(input_dir, dry_run=True)

    assert stats["num_entries"] == 2
    assert stats["num_images_migrated"] == 0
    assert not (tmp_path / "dataset_v02").exists()
    in_place_stats = migrate.migrate_dataset(input_dir, in_place=True, dry_run=True)
    assert in_place_stats["num_entries"] == 2

    with pytest.raises(FileNotFoundError):
        migrate.migrate_dataset(tmp_path / "missing")

    monkeypatch.setattr(
        migrate,
        "detect_dataset_version",
        lambda _path: (_ for _ in ()).throw(ValueError("bad layout")),
    )
    with pytest.raises(ValueError, match="Cannot detect dataset version"):
        migrate.migrate_dataset(input_dir)

    monkeypatch.setattr(migrate, "detect_dataset_version", lambda _path: "0.2")
    with pytest.raises(ValueError, match="already v0.2"):
        migrate.migrate_dataset(input_dir)


def test_migrate_file_helpers_cover_copy_move_cleanup_and_noops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write(input_dir / "usd" / "renders" / "nested.png")
    _write(input_dir / "usd" / "usd_model.json", "{}")
    _write(input_dir / "vlm_system_prompt.txt")
    _write(input_dir / "spec.txt")
    _write(input_dir / "usd" / "prims.jsonl")
    _write(input_dir / "usd" / "dataset.json")

    render_stats = migrate._migrate_renders(
        input_dir, input_dir, in_place=True, dry_run=False
    )
    model_stats = migrate._migrate_usd_model(
        input_dir, input_dir, in_place=True, dry_run=False
    )
    cleanup_stats = migrate._cleanup_v01_files(input_dir, in_place=True)

    assert render_stats == {"num_images_migrated": 1}
    assert model_stats == {"usd_model_migrated": True}
    assert cleanup_stats["files_cleaned_up"] == 4
    assert (input_dir / "renders" / "nested.png").exists()
    assert (input_dir / "usd_model.json").exists()

    top_input = tmp_path / "top_input"
    top_output = tmp_path / "top_output"
    _write(top_input / "renders" / "top.jpg")
    _write(top_input / "usd_model.json", "{}")
    top_output.mkdir()

    assert migrate._migrate_renders(
        top_input, top_output, in_place=False, dry_run=False
    ) == {"num_images_migrated": 1}
    assert migrate._migrate_usd_model(
        top_input, top_output, in_place=False, dry_run=False
    ) == {"usd_model_migrated": True}
    assert migrate._migrate_reference_images(
        top_input, top_output, in_place=False, dry_run=False
    ) == {"num_reference_images_migrated": 0}

    dry_input = tmp_path / "dry"
    _write(dry_input / "usd" / "renders" / "dry.png")
    _write(dry_input / "usd" / "usd_model.json", "{}")
    assert migrate._migrate_renders(
        dry_input, tmp_path / "dry_out", in_place=False, dry_run=True
    ) == {"num_images_migrated": 1}
    assert migrate._migrate_usd_model(
        dry_input, tmp_path / "dry_out", in_place=False, dry_run=True
    ) == {"usd_model_migrated": True}

    stubborn = input_dir / "usd"
    stubborn.mkdir(exist_ok=True)
    (stubborn / "renders" / "empty_child").mkdir(parents=True)
    monkeypatch.setattr(Path, "rmdir", lambda _self: (_ for _ in ()).throw(OSError()))
    migrate._cleanup_v01_files(input_dir, in_place=True)
    usd_dir_error = tmp_path / "usd_dir_error"
    (usd_dir_error / "usd").mkdir(parents=True)
    migrate._cleanup_v01_files(usd_dir_error, in_place=True)


def test_migrate_datasets_batch_summarizes_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    parent = tmp_path / "out"

    def fake_migrate_dataset(
        input_dir: Path,
        output_dir: Path,
        in_place: bool,
        keep_intermediate: bool,
        dry_run: bool,
    ) -> dict[str, Any]:
        assert in_place is False
        assert keep_intermediate is True
        assert dry_run is True
        if Path(input_dir) == bad:
            raise RuntimeError("cannot migrate")
        assert output_dir == parent / "good_v02"
        return {"num_entries": 3, "num_images_migrated": 4}

    monkeypatch.setattr(migrate, "migrate_dataset", fake_migrate_dataset)

    stats = migrate.migrate_datasets_batch(
        [good, bad],
        output_parent_dir=parent,
        keep_intermediate=True,
        dry_run=True,
    )

    assert stats["total_datasets"] == 2
    assert stats["successful"] == 1
    assert stats["failed"] == 1
    assert stats["failed_datasets"] == [str(bad)]
    assert stats["total_entries"] == 3
    assert stats["total_images"] == 4
    assert stats["results"][0]["status"] == "success"
    assert stats["results"][1]["status"] == "failed"

    seen_outputs: list[Path] = []

    def fake_default_output(
        input_dir: Path,
        output_dir: Path,
        in_place: bool,
        keep_intermediate: bool,
        dry_run: bool,
    ) -> dict[str, Any]:
        seen_outputs.append(output_dir)
        return {"num_entries": 1, "num_images_migrated": 2}

    monkeypatch.setattr(migrate, "migrate_dataset", fake_default_output)
    default_stats = migrate.migrate_datasets_batch([good])

    assert seen_outputs == [good.parent / "good_v02"]
    assert default_stats["successful"] == 1
