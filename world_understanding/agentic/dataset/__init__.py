# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD Agent Dataset Schema v0.2.

This package provides schema definitions, loaders, and utilities for working with
unified USD agent datasets.

Key Modules:
    - schema: Pydantic models for dataset.json and dataset.jsonl
    - loader: Load datasets with auto-detection of v0.1 vs v0.2
    - migrate: Migration utilities for converting v0.1 → v0.2

Example:
    ```python
    from world_understanding.agentic.dataset import load_dataset

    # Auto-detect format and load
    config, entries = load_dataset("path/to/dataset")

    # Iterate through entries
    for entry in entries:
        print(f"Processing {entry.id}")
        print(f"  Images: {len(entry.media.images)}")
        print(f"  User prompt: {entry.user_prompt[:50]}...")
    ```
"""

from world_understanding.agentic.dataset.base_dataset_loading import (
    BaseDatasetLoadingTask,
)
from world_understanding.agentic.dataset.loader import (
    detect_dataset_version,
    load_dataset,
    load_dataset_config,
    load_dataset_entries,
)
from world_understanding.agentic.dataset.migrate import (
    migrate_dataset,
    migrate_datasets_batch,
)
from world_understanding.agentic.dataset.schema import (
    DatasetConfig,
    DatasetEntry,
    DatasetMetadata,
    GroundTruth,
    GroundTruthMetadata,
    ImageMetadata,
    ImageObject,
    InferenceConfig,
    MediaConfig,
    PromptConfig,
    SourceInfo,
    StepResult,
    TaskConfig,
    UntrustedSpecEvidence,
    UntrustedSpecReferencePage,
    export_json_schema,
    validate_dataset_config_file,
    validate_dataset_entry,
)
from world_understanding.utils.misc_utils import get_version

__version__ = get_version()

__all__ = [
    # Task classes
    "BaseDatasetLoadingTask",
    # Main models
    "DatasetConfig",
    "DatasetEntry",
    # dataset.json components
    "DatasetMetadata",
    "TaskConfig",
    "PromptConfig",
    "InferenceConfig",
    # dataset.jsonl entry components
    "SourceInfo",
    "ImageMetadata",
    "ImageObject",
    "MediaConfig",
    "GroundTruth",
    "GroundTruthMetadata",
    "StepResult",
    "UntrustedSpecEvidence",
    "UntrustedSpecReferencePage",
    # Loader functions
    "detect_dataset_version",
    "load_dataset",
    "load_dataset_config",
    "load_dataset_entries",
    # Migration functions
    "migrate_dataset",
    "migrate_datasets_batch",
    # Utilities
    "export_json_schema",
    "validate_dataset_config_file",
    "validate_dataset_entry",
]
