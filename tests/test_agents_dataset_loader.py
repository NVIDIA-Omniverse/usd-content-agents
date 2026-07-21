# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dataset loader with v0.1/v0.2 auto-detection."""

import json
import tempfile
from pathlib import Path

import pytest

from world_understanding.agentic.dataset.loader import (
    detect_dataset_version,
    load_dataset,
    load_dataset_config,
    load_dataset_config_v01,
    load_dataset_entries,
)
from world_understanding.agentic.dataset.schema import (
    DatasetConfig,
    DatasetEntry,
)


@pytest.fixture
def temp_dataset_dir():
    """Create temporary directory for test datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def v01_dataset(temp_dataset_dir):
    """Create a minimal v0.1 dataset for testing."""
    # Create vlm_system_prompt.txt
    system_prompt = "You are an expert at identifying materials."
    (temp_dataset_dir / "vlm_system_prompt.txt").write_text(system_prompt)

    # Create dataset.jsonl
    entries = [
        {
            "id": "/prim/path/1",
            "text": "Identify the material for this component.",
            "images": ["usd/renders/part1_view1.png", "usd/renders/part1_view2.png"],
            "image_metadata": [
                {
                    "path": "usd/renders/part1_view1.png",
                    "view": "posx_posy_posz",
                    "camera": "/Cameras/Camera1",
                    "render_mode": "prim_with_stage",
                    "vlm_prompt": "This is a rendered part highlighted with an orange outline.",
                },
                {
                    "path": "usd/renders/part1_view2.png",
                    "view": "negx_negy_negz",
                    "camera": "/Cameras/Camera2",
                    "render_mode": "prim_only",
                    "vlm_prompt": "This is a rendered part only without highlighting.",
                },
            ],
            "ground_truth": "002_plastic_black",
        },
        {
            "id": "/prim/path/2",
            "text": "Identify the material for this component.",
            "images": ["usd/renders/part2_view1.png"],
            "image_metadata": [
                {
                    "path": "usd/renders/part2_view1.png",
                    "view": "posx_posy_posz",
                    "camera": "/Cameras/Camera1",
                    "render_mode": "prim_with_stage",
                }
            ],
            "ground_truth": "007_tin_plating",
        },
    ]

    with open(temp_dataset_dir / "dataset.jsonl", "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Create usd/dataset.json (optional metadata)
    usd_dir = temp_dataset_dir / "usd"
    usd_dir.mkdir()

    usd_dataset = {
        "version": "1.0",
        "metadata": {
            "source_usd": "/path/to/model.usd",
            "created": "2025-01-15T10:30:00Z",
        },
        "statistics": {"total_prims": 2, "total_images": 3},
        "prims_file": "prims.jsonl",
    }

    with open(usd_dir / "dataset.json", "w") as f:
        json.dump(usd_dataset, f)

    return temp_dataset_dir


@pytest.fixture
def v02_dataset(temp_dataset_dir):
    """Create a minimal v0.2 dataset for testing."""
    # Create dataset.json
    config = {
        "schema_version": "0.2",
        "metadata": {
            "created": "2025-01-15T10:30:00Z",
            "creator": "test-agent",
            "source_usd": "/path/to/model.usd",
            "description": "Test dataset",
            "num_entries": 2,
        },
        "task": {"type": "material_assignment", "description": "Material assignment"},
        "inference": {
            "prompts": [
                {
                    "step_name": "main",
                    "step_index": 0,
                    "system_prompt": "You are an expert at identifying materials.",
                }
            ]
        },
        "prims_file": "dataset.jsonl",
        "usd_model_file": "usd_model.json",
    }

    with open(temp_dataset_dir / "dataset.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create dataset.jsonl
    entries = [
        {
            "id": "/prim/path/1",
            "source": {"type": "usd_prim", "prim_path": "/prim/path/1"},
            "user_prompt": "Identify the material for this component.",
            "media": {
                "images": [
                    {
                        "path": "renders/part1_view1.png",
                        "type": "render",
                        "metadata": {
                            "view": "posx_posy_posz",
                            "camera": "/Cameras/Camera1",
                            "render_mode": "prim_with_stage",
                            "vlm_prompt": "This is a rendered part highlighted with an orange outline.",
                        },
                    }
                ]
            },
            "ground_truth": {
                "material": "002_plastic_black",
                "metadata": {"source": "oracle"},
            },
        },
        {
            "id": "/prim/path/2",
            "source": {"type": "usd_prim", "prim_path": "/prim/path/2"},
            "user_prompt": "Identify the material for this component.",
            "media": {
                "images": [
                    {
                        "path": "renders/part2_view1.png",
                        "type": "render",
                        "metadata": {"view": "posx_posy_posz"},
                    }
                ]
            },
            "ground_truth": {"material": "007_tin_plating"},
        },
    ]

    with open(temp_dataset_dir / "dataset.jsonl", "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return temp_dataset_dir


class TestVersionDetection:
    """Test dataset version detection."""

    def test_detect_v01_with_vlm_prompt(self, v01_dataset):
        """Test detecting v0.1 dataset with vlm_system_prompt.txt."""
        version = detect_dataset_version(v01_dataset)
        assert version == "0.1"

    def test_detect_v02_with_schema_version(self, v02_dataset):
        """Test detecting v0.2 dataset with schema_version field."""
        version = detect_dataset_version(v02_dataset)
        assert version == "0.2"

    def test_detect_invalid_dataset(self, temp_dataset_dir):
        """Test detecting invalid dataset raises error."""
        with pytest.raises(ValueError, match="Cannot determine dataset format"):
            detect_dataset_version(temp_dataset_dir)

    def test_detect_malformed_dataset_json_falls_back_to_v01_indicator(
        self, temp_dataset_dir
    ):
        (temp_dataset_dir / "dataset.json").write_text(
            "{invalid json", encoding="utf-8"
        )
        (temp_dataset_dir / "vlm_system_prompt.txt").write_text(
            "Prompt", encoding="utf-8"
        )

        assert detect_dataset_version(temp_dataset_dir) == "0.1"

    def test_detect_dataset_json_without_schema_version_as_v01(self, temp_dataset_dir):
        (temp_dataset_dir / "dataset.json").write_text(
            json.dumps({"metadata": {"created": "2025-01-15T10:30:00Z"}}),
            encoding="utf-8",
        )

        assert detect_dataset_version(temp_dataset_dir) == "0.1"


class TestV01Loading:
    """Test loading v0.1 datasets."""

    def test_load_v01_config(self, v01_dataset):
        """Test loading v0.1 dataset config (converted to v0.2)."""
        config = load_dataset_config(v01_dataset)

        assert isinstance(config, DatasetConfig)
        assert config.schema_version == "0.2"
        assert config.task.type == "material_assignment"
        assert config.metadata.num_entries == 2
        assert (
            "expert at identifying materials"
            in config.inference.prompts[0].system_prompt
        )

    def test_load_v01_config_counts_jsonl_when_metadata_count_missing(
        self, temp_dataset_dir
    ):
        (temp_dataset_dir / "vlm_system_prompt.txt").write_text(
            "System prompt", encoding="utf-8"
        )
        usd_dir = temp_dataset_dir / "usd"
        usd_dir.mkdir()
        (usd_dir / "dataset.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "source_usd": "/path/to/model.usd",
                        "created": "2025-01-15T10:30:00Z",
                    },
                    "statistics": {"total_prims": 0},
                }
            ),
            encoding="utf-8",
        )
        entries = [
            {"id": "/prim/one", "text": "Prompt", "images": ["one.png"]},
            {"id": "/prim/two", "text": "Prompt", "images": ["two.png"]},
        ]
        (temp_dataset_dir / "dataset.jsonl").write_text(
            "\n".join(json.dumps(entry) for entry in entries) + "\n\n",
            encoding="utf-8",
        )

        config = load_dataset_config_v01(temp_dataset_dir)

        assert config.metadata.num_entries == 2

    def test_load_v01_entries(self, v01_dataset):
        """Test loading v0.1 dataset entries (converted to v0.2)."""
        entries = list(load_dataset_entries(v01_dataset))

        assert len(entries) == 2

        # Check first entry
        entry1 = entries[0]
        assert isinstance(entry1, DatasetEntry)
        assert entry1.id == "/prim/path/1"
        assert entry1.source.type == "usd_prim"
        assert entry1.user_prompt == "Identify the material for this component."
        assert len(entry1.media.images) == 2
        assert entry1.ground_truth.material == "002_plastic_black"

        # Check image metadata preservation
        img1 = entry1.media.images[0]
        assert img1.metadata.view == "posx_posy_posz"
        assert img1.metadata.render_mode == "prim_with_stage"
        assert (
            img1.metadata.vlm_prompt
            == "This is a rendered part highlighted with an orange outline."
        )

    def test_load_v01_entries_skips_blank_lines(self, v01_dataset):
        jsonl_path = v01_dataset / "dataset.jsonl"
        jsonl_path.write_text(
            "\n" + jsonl_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

        entries = list(load_dataset_entries(v01_dataset))

        assert len(entries) == 2

    def test_load_v01_with_filter(self, v01_dataset):
        """Test loading v0.1 entries with filter."""

        # Filter for entries with specific material
        def material_filter(entry: DatasetEntry) -> bool:
            return (
                entry.ground_truth
                and entry.ground_truth.material == "002_plastic_black"
            )

        entries = list(load_dataset_entries(v01_dataset, entry_filter=material_filter))

        assert len(entries) == 1
        assert entries[0].ground_truth.material == "002_plastic_black"


class TestV02Loading:
    """Test loading v0.2 datasets."""

    def test_load_v02_config(self, v02_dataset):
        """Test loading v0.2 dataset config."""
        config = load_dataset_config(v02_dataset)

        assert isinstance(config, DatasetConfig)
        assert config.schema_version == "0.2"
        assert config.task.type == "material_assignment"
        assert config.metadata.creator == "test-agent"
        assert config.metadata.num_entries == 2

    def test_load_v02_entries(self, v02_dataset):
        """Test loading v0.2 dataset entries."""
        entries = list(load_dataset_entries(v02_dataset))

        assert len(entries) == 2

        # Check first entry
        entry1 = entries[0]
        assert isinstance(entry1, DatasetEntry)
        assert entry1.id == "/prim/path/1"
        assert entry1.source.prim_path == "/prim/path/1"
        assert entry1.user_prompt == "Identify the material for this component."
        assert len(entry1.media.images) == 1
        assert entry1.ground_truth.material == "002_plastic_black"

        # Check image metadata
        img1 = entry1.media.images[0]
        assert img1.path == "renders/part1_view1.png"
        assert img1.type == "render"
        assert (
            img1.metadata.vlm_prompt
            == "This is a rendered part highlighted with an orange outline."
        )

    def test_load_v02_entries_skips_blank_lines(self, v02_dataset):
        jsonl_path = v02_dataset / "dataset.jsonl"
        jsonl_path.write_text(
            "\n" + jsonl_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

        entries = list(load_dataset_entries(v02_dataset))

        assert len(entries) == 2

    def test_load_v02_with_filter(self, v02_dataset):
        """Test loading v0.2 entries with filter."""

        def id_filter(entry: DatasetEntry) -> bool:
            return entry.id == "/prim/path/1"

        entries = list(load_dataset_entries(v02_dataset, entry_filter=id_filter))

        assert len(entries) == 1
        assert entries[0].id == "/prim/path/1"


class TestUnifiedInterface:
    """Test unified loading interface."""

    def test_load_dataset_v01(self, v01_dataset):
        """Test load_dataset() with v0.1 dataset."""
        config, entries = load_dataset(v01_dataset)

        assert isinstance(config, DatasetConfig)
        entries_list = list(entries)
        assert len(entries_list) == 2

    def test_load_dataset_v02(self, v02_dataset):
        """Test load_dataset() with v0.2 dataset."""
        config, entries = load_dataset(v02_dataset)

        assert isinstance(config, DatasetConfig)
        entries_list = list(entries)
        assert len(entries_list) == 2

    def test_load_dataset_with_filter(self, v02_dataset):
        """Test load_dataset() with entry filter."""

        def filter_func(entry: DatasetEntry) -> bool:
            return entry.ground_truth and entry.ground_truth.material.startswith("002")

        config, entries = load_dataset(v02_dataset, entry_filter=filter_func)

        entries_list = list(entries)
        assert len(entries_list) == 1
        assert entries_list[0].ground_truth.material == "002_plastic_black"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_load_missing_dataset(self, temp_dataset_dir):
        """Test loading non-existent dataset raises error."""
        missing_dir = temp_dataset_dir / "nonexistent"

        with pytest.raises(ValueError):
            detect_dataset_version(missing_dir)

    def test_load_empty_jsonl(self, temp_dataset_dir):
        """Test loading dataset with empty JSONL file."""
        # Create minimal v0.2 config
        config = {
            "schema_version": "0.2",
            "metadata": {
                "created": "2025-01-15T10:30:00Z",
                "creator": "test",
                "num_entries": 0,
            },
            "task": {"type": "material_assignment", "description": "Test"},
            "inference": {
                "prompts": [
                    {"step_name": "main", "step_index": 0, "system_prompt": "Test"}
                ]
            },
        }

        with open(temp_dataset_dir / "dataset.json", "w") as f:
            json.dump(config, f)

        # Create empty JSONL
        (temp_dataset_dir / "dataset.jsonl").touch()

        # Should load successfully but yield no entries
        entries = list(load_dataset_entries(temp_dataset_dir))
        assert len(entries) == 0

    def test_load_malformed_jsonl_entry(self, v02_dataset):
        """Test loading JSONL with malformed entry (should skip it)."""
        # Append malformed entry
        with open(v02_dataset / "dataset.jsonl", "a") as f:
            f.write("{invalid json}\n")

        # Should load successfully but skip malformed entry
        entries = list(load_dataset_entries(v02_dataset))
        assert len(entries) == 2  # Only valid entries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
