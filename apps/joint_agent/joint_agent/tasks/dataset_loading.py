# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset loading and validation task for asset agent."""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.dataset import BaseDatasetLoadingTask
from world_understanding.agentic.events import get_listener
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class DatasetLoadingTask(BaseDatasetLoadingTask):
    """Load and validate dataset from JSONL file with image validation.

    This task extends the base dataset loading with joint-agent-specific
    validation logic for image paths and metadata.
    """

    def __init__(self, dataset_path: Path | None = None, validate: bool = True):
        """Initialize the dataset loading task.

        Args:
            dataset_path: Path to the JSONL dataset file (None to use from context)
            validate: Whether to validate dataset entries
        """
        super().__init__(
            dataset_path=dataset_path,
            validate=validate,
            name="DatasetLoading",
            description="Load and validate dataset from JSONL file",
        )

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Load dataset from JSONL file with event listener support.

        Args:
            context: Workflow context
            object_store: Storage for large objects

        Returns:
            Updated context with dataset metadata
        """
        # Store event listener in instance for validation methods
        self._listener = get_listener(context, logger_name=__name__)

        # Delegate to base class
        return super().run(context, object_store)

    def _validate_dataset(self, dataset: list[dict], dataset_path: Path) -> list[dict]:
        """Validate dataset entries with image existence checks.

        Args:
            dataset: List of dataset entries
            dataset_path: Path to dataset file for resolving relative paths

        Returns:
            Filtered list of valid entries
        """
        valid_entries = []
        for entry in dataset:
            if self._validate_entry(entry, dataset_path, self._listener):
                valid_entries.append(entry)
            else:
                self._listener.warning(f"Invalid entry: {entry.get('id', 'unknown')}")
        return valid_entries

    def _update_context(
        self,
        context: dict[str, Any],
        dataset: list[dict],
        dataset_path: Path,
        metadata: dict[str, Any],
    ) -> None:
        """Update context with joint-agent-specific fields.

        Args:
            context: Workflow context to update
            dataset: Loaded dataset entries
            dataset_path: Path to dataset file
            metadata: Dataset metadata
        """
        # Call base class to update standard fields
        super()._update_context(context, dataset, dataset_path, metadata)

        # Add joint-agent specific field
        context["image_base_dir"] = str(dataset_path.parent)

    def _validate_entry(self, entry: dict, dataset_path: Path, listener: Any) -> bool:
        """Validate a dataset entry.

        Args:
            entry: Dataset entry to validate
            dataset_path: Path to dataset for resolving relative image paths
            listener: Event listener for progress reporting

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if "id" not in entry:
            return False

        # Support multiple dataset formats:
        # 1. Old format: entry["images"] as list of paths
        # 2. Old format: entry["image_path"] as single path
        # 3. New format: entry["media"]["images"] as list of dicts with "path" key

        has_images = "images" in entry and entry["images"]
        has_image_path = "image_path" in entry and entry["image_path"]
        has_media_images = (
            "media" in entry
            and isinstance(entry["media"], dict)
            and "images" in entry["media"]
            and entry["media"]["images"]
        )

        if not has_images and not has_image_path and not has_media_images:
            listener.debug(
                f"Entry {entry.get('id')} has no images in any supported format"
            )
            return False

        # Check images exist if paths are provided
        if has_images:
            # Handle list of images (old format)
            for img_path in entry["images"]:
                if img_path:
                    image_path = Path(dataset_path.parent) / img_path
                    if not image_path.exists():
                        listener.warning(f"Image not found: {image_path}")
                        return False
        elif has_image_path:
            # Handle single image path (old format)
            image_path = Path(dataset_path.parent) / entry["image_path"]
            if not image_path.exists():
                listener.warning(f"Image not found: {image_path}")
                return False
        elif has_media_images:
            # Handle new format: media.images[] with path field
            for img_obj in entry["media"]["images"]:
                if isinstance(img_obj, dict) and "path" in img_obj:
                    img_path = img_obj["path"]
                    if img_path:
                        image_path = Path(dataset_path.parent) / img_path
                        if not image_path.exists():
                            listener.warning(f"Image not found: {image_path}")
                            return False
                else:
                    listener.warning(f"Invalid image object in media.images: {img_obj}")
                    return False

        return True
