# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dataset schema (v0.2) Pydantic models."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from world_understanding.agentic.dataset.schema import (
    DatasetConfig,
    DatasetEntry,
    DatasetMetadata,
    GroundTruth,
    ImageMetadata,
    ImageObject,
    InferenceConfig,
    MediaConfig,
    PromptConfig,
    SourceInfo,
    TaskConfig,
    export_json_schema,
    validate_dataset_config_file,
    validate_dataset_entry,
)


class TestDatasetMetadata:
    """Test DatasetMetadata model."""

    def test_valid_metadata(self):
        """Test creating valid metadata."""
        metadata = DatasetMetadata(
            created="2025-01-15T10:30:00Z",
            creator="material-agent",
            source_usd="/path/to/model.usd",
            description="Test dataset",
            num_entries=100,
        )

        assert metadata.creator == "material-agent"
        assert metadata.num_entries == 100

    def test_timestamp_validation(self):
        """Test timestamp validation."""
        # Valid ISO 8601 timestamp
        metadata = DatasetMetadata(
            created="2025-01-15T10:30:00",
            creator="test",
            num_entries=1,
        )
        assert metadata.created == "2025-01-15T10:30:00"

        # Datetime object should be converted
        metadata = DatasetMetadata(
            created=datetime(2025, 1, 15, 10, 30, 0),
            creator="test",
            num_entries=1,
        )
        assert "2025-01-15" in metadata.created

    def test_optional_fields(self):
        """Test optional fields."""
        metadata = DatasetMetadata(
            created="2025-01-15T10:30:00",
            creator="test",
            num_entries=0,
        )

        assert metadata.source_usd is None
        assert metadata.description is None


class TestTaskConfig:
    """Test TaskConfig model."""

    def test_valid_task_types(self):
        """Test all valid task types."""
        task_types = ["material_assignment", "iterative_classification", "detection"]

        for task_type in task_types:
            task = TaskConfig(type=task_type, description=f"Test {task_type}")
            assert task.type == task_type

    def test_invalid_task_type(self):
        """Test invalid task type raises error."""
        with pytest.raises(ValidationError):
            TaskConfig(type="invalid_type", description="Test")


class TestPromptConfig:
    """Test PromptConfig model."""

    def test_valid_prompt(self):
        """Test creating valid prompt config."""
        prompt = PromptConfig(
            step_name="main",
            step_index=0,
            system_prompt="You are an expert...",
            classes=["class1", "class2"],
            temperature=0.7,
            max_tokens=1024,
        )

        assert prompt.step_index == 0
        assert prompt.temperature == 0.7
        assert len(prompt.classes) == 2


class TestInferenceConfig:
    """Test InferenceConfig model."""

    def test_single_prompt(self):
        """Test inference config with single prompt."""
        inference = InferenceConfig(
            prompts=[
                PromptConfig(
                    step_name="main",
                    step_index=0,
                    system_prompt="Test prompt",
                )
            ]
        )

        assert len(inference.prompts) == 1

    def test_multi_step_prompts(self):
        """Test multi-step prompts."""
        inference = InferenceConfig(
            prompts=[
                PromptConfig(step_name="step1", step_index=0, system_prompt="Prompt 1"),
                PromptConfig(step_name="step2", step_index=1, system_prompt="Prompt 2"),
            ]
        )

        assert len(inference.prompts) == 2

    def test_empty_prompts_invalid(self):
        """Test that empty prompts list is invalid."""
        with pytest.raises(ValidationError):
            InferenceConfig(prompts=[])

    def test_non_sequential_indices_invalid(self):
        """Test that non-sequential step indices are invalid."""
        with pytest.raises(ValidationError):
            InferenceConfig(
                prompts=[
                    PromptConfig(
                        step_name="step1", step_index=0, system_prompt="Prompt 1"
                    ),
                    PromptConfig(
                        step_name="step2", step_index=2, system_prompt="Prompt 2"
                    ),  # Missing index 1
                ]
            )

    def test_step_index_validator_allows_empty_input_fast_path(self):
        """The sequential-index validator itself leaves empty lists unchanged."""
        assert InferenceConfig.validate_step_indices([]) == []


class TestDatasetConfig:
    """Test DatasetConfig model."""

    def test_valid_config(self):
        """Test creating valid dataset config."""
        config = DatasetConfig(
            schema_version="0.2",
            metadata=DatasetMetadata(
                created="2025-01-15T10:30:00",
                creator="material-agent",
                num_entries=10,
            ),
            task=TaskConfig(type="material_assignment", description="Test task"),
            inference=InferenceConfig(
                prompts=[
                    PromptConfig(
                        step_name="main", step_index=0, system_prompt="Test prompt"
                    )
                ]
            ),
        )

        assert config.schema_version == "0.2"
        assert config.task.type == "material_assignment"

    def test_iterative_classification_validation(self):
        """Test that iterative_classification requires multiple prompts."""
        # Should fail with single prompt
        with pytest.raises(ValidationError):
            DatasetConfig(
                metadata=DatasetMetadata(
                    created="2025-01-15T10:30:00", creator="test", num_entries=1
                ),
                task=TaskConfig(
                    type="iterative_classification", description="Multi-step"
                ),
                inference=InferenceConfig(
                    prompts=[
                        PromptConfig(
                            step_name="main", step_index=0, system_prompt="Prompt"
                        )
                    ]
                ),
            )

        # Should succeed with multiple prompts
        config = DatasetConfig(
            metadata=DatasetMetadata(
                created="2025-01-15T10:30:00", creator="test", num_entries=1
            ),
            task=TaskConfig(type="iterative_classification", description="Multi-step"),
            inference=InferenceConfig(
                prompts=[
                    PromptConfig(
                        step_name="step1", step_index=0, system_prompt="Prompt 1"
                    ),
                    PromptConfig(
                        step_name="step2", step_index=1, system_prompt="Prompt 2"
                    ),
                ]
            ),
        )

        assert len(config.inference.prompts) == 2


class TestImageObject:
    """Test ImageObject model."""

    def test_render_image(self):
        """Test creating render image."""
        img = ImageObject(
            path="renders/part1.png",
            type="render",
            metadata=ImageMetadata(
                view="posx_posy_posz",
                render_mode="prim_with_stage",
                vlm_prompt="Test prompt",
            ),
        )

        assert img.type == "render"
        assert img.metadata.vlm_prompt == "Test prompt"

    def test_reference_image(self):
        """Test creating reference image."""
        img = ImageObject(
            path="references/ref1.jpg",
            type="reference",
        )

        assert img.type == "reference"
        assert img.metadata is None


class TestMediaConfig:
    """Test MediaConfig model."""

    def test_with_renders(self):
        """Test media config with render images."""
        media = MediaConfig(
            images=[
                ImageObject(path="renders/part1.png", type="render"),
                ImageObject(path="renders/part2.png", type="render"),
            ]
        )

        assert len(media.images) == 2
        assert media.reference_images is None

    def test_with_references(self):
        """Test media config with reference images."""
        media = MediaConfig(
            images=[ImageObject(path="renders/part1.png", type="render")],
            reference_images=[ImageObject(path="refs/ref1.jpg", type="reference")],
        )

        assert len(media.images) == 1
        assert len(media.reference_images) == 1

    def test_empty_images_invalid(self):
        """Test that empty images list is invalid."""
        with pytest.raises(ValidationError):
            MediaConfig(images=[])


class TestDatasetEntry:
    """Test DatasetEntry model."""

    def test_single_step_entry(self):
        """Test entry with single user prompt."""
        entry = DatasetEntry(
            id="/prim/path",
            source=SourceInfo(type="usd_prim", prim_path="/prim/path"),
            user_prompt="Test prompt",
            media=MediaConfig(
                images=[ImageObject(path="renders/part1.png", type="render")]
            ),
        )

        assert entry.user_prompt == "Test prompt"
        assert entry.user_prompts is None

    def test_multi_step_entry(self):
        """Test entry with multiple user prompts."""
        entry = DatasetEntry(
            id="/prim/path",
            source=SourceInfo(type="usd_prim", prim_path="/prim/path"),
            user_prompts=["Prompt 1", "Prompt 2"],
            media=MediaConfig(
                images=[ImageObject(path="renders/part1.png", type="render")]
            ),
        )

        assert entry.user_prompts == ["Prompt 1", "Prompt 2"]
        assert entry.user_prompt is None

    def test_no_prompts_invalid(self):
        """Test that entry without any prompts is invalid."""
        with pytest.raises(ValidationError):
            DatasetEntry(
                id="/prim/path",
                source=SourceInfo(type="usd_prim"),
                media=MediaConfig(
                    images=[ImageObject(path="renders/part1.png", type="render")]
                ),
            )

    def test_both_prompts_invalid(self):
        """Test that entry with both prompt types is invalid."""
        with pytest.raises(ValidationError):
            DatasetEntry(
                id="/prim/path",
                source=SourceInfo(type="usd_prim"),
                user_prompt="Single prompt",
                user_prompts=["Multi prompt 1", "Multi prompt 2"],
                media=MediaConfig(
                    images=[ImageObject(path="renders/part1.png", type="render")]
                ),
            )

    def test_with_ground_truth(self):
        """Test entry with ground truth."""
        entry = DatasetEntry(
            id="/prim/path",
            source=SourceInfo(type="usd_prim"),
            user_prompt="Test",
            media=MediaConfig(
                images=[ImageObject(path="renders/part1.png", type="render")]
            ),
            ground_truth=GroundTruth(material="002_plastic_black"),
        )

        assert entry.ground_truth.material == "002_plastic_black"


class TestSchemaHelpers:
    """Test schema export and standalone validation helpers."""

    def test_export_json_schema_includes_config_and_entry(self):
        schemas = export_json_schema()

        assert set(schemas) == {"dataset_config", "dataset_entry"}
        assert schemas["dataset_config"]["title"] == "DatasetConfig"
        assert schemas["dataset_entry"]["title"] == "DatasetEntry"

    def test_validate_dataset_config_file(self, tmp_path):
        config_path = tmp_path / "dataset.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "0.2",
                    "metadata": {
                        "created": "2025-01-15T10:30:00Z",
                        "creator": "test",
                        "num_entries": 1,
                    },
                    "task": {
                        "type": "material_assignment",
                        "description": "Assign materials",
                    },
                    "inference": {
                        "prompts": [
                            {
                                "step_name": "main",
                                "step_index": 0,
                                "system_prompt": "Prompt",
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

        config = validate_dataset_config_file(str(config_path))

        assert config.task.type == "material_assignment"

    def test_validate_dataset_config_file_rejects_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Dataset config not found"):
            validate_dataset_config_file(str(tmp_path / "missing.json"))

    def test_validate_dataset_entry_helper(self):
        entry = validate_dataset_entry(
            {
                "id": "/prim/path",
                "source": {"type": "usd_prim", "prim_path": "/prim/path"},
                "user_prompt": "Test prompt",
                "media": {
                    "images": [
                        {
                            "path": "renders/part1.png",
                            "type": "render",
                        }
                    ]
                },
            }
        )

        assert entry.id == "/prim/path"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
