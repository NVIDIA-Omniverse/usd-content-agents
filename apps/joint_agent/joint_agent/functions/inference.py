# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference functions for VLM-based asset classification.

This module provides the core classification logic for Joint Agent.
It delegates to world_understanding.functions.classification for the generic
classification implementation, allowing configurable output_key for flexible
classification tasks.
"""

import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from PIL import Image as PILImage

# Import generic classification functions
from world_understanding.functions.classification.inference import (
    batch_classify_objects,
    classify_object,
    extract_answer_block,
    get_fibonacci_delay,
)
from world_understanding.functions.models.vision_language_models import (
    BaseVisionLanguageModel,
)
from world_understanding.utils.token_tracking import TokenTracker

from joint_agent.functions.stage1_schema import normalize_stage1_prediction_payload

logger = logging.getLogger(__name__)


def classify_asset(
    vlm: BaseVisionLanguageModel,
    text: str,
    images: list[str | Path | PILImage.Image],
    llm: BaseChatModel,
    system_prompt: str | None = None,
    invoke_kwargs: dict[str, Any] | None = None,
    image_prompts: list[str] | None = None,
    max_retries: int = 3,
    output_key: str = "classification",
    token_tracker: TokenTracker | None = None,
) -> dict[str, Any]:
    """Classify an asset using Vision-Language Model.

    This is the core inference function for the Joint Agent. It analyzes
    images of an asset and provides classification results based on the
    system prompt configuration.

    Args:
        vlm: Vision-Language Model instance to use for inference
        text: Context text containing object description and classification criteria
        images: List of images as file paths (str/Path) or PIL Image objects
        llm: LLM for parsing VLM response into structured format (fallback only)
        system_prompt: Optional custom system prompt (uses default if None)
        invoke_kwargs: Additional kwargs to pass to VLM (e.g., temperature, max_tokens)
        image_prompts: Optional list of prompts/captions for each image
        max_retries: Maximum number of retry attempts for VLM/LLM calls (default: 3)
        output_key: Key name for the classification result (default: "classification")
        token_tracker: Optional TokenTracker to collect usage statistics

    Returns:
        Dict with output_key and "original_response" keys

    Example:
        ```python
        from joint_agent.functions import classify_asset

        # Classify component type and material
        system_prompt = '''
        Analyze the component and provide:
        {
            "component_type": "what this is",
            "material": "predicted material",
            "properties": {"friction": 0.0, "density": 0.0}
        }
        '''

        response = classify_asset(
            vlm=vlm,
            text="Analyze this mechanical part",
            images=["part_view1.png", "part_view2.png"],
            llm=llm,
            system_prompt=system_prompt,
            output_key="analysis"
        )
        print(response)
        # Output: {"analysis": {...}, "original_response": "..."}
        ```
    """
    logger.debug(f"classify_asset() with output_key='{output_key}'")
    logger.debug(f"Running classification with {len(images)} images")

    # Delegate to generic classification, then normalize the parsed payload to
    # the Joint Agent Stage 1 contract.
    response = classify_object(
        vlm=vlm,
        text=text,
        images=images,
        llm=llm,
        system_prompt=system_prompt,
        invoke_kwargs=invoke_kwargs,
        image_prompts=image_prompts,
        max_retries=max_retries,
        output_key=output_key,
        token_tracker=token_tracker,
    )
    return normalize_stage1_prediction_payload(response, output_key=output_key)


def batch_classify_assets(
    vlm: BaseVisionLanguageModel,
    entries: list[dict[str, Any]],
    llm: BaseChatModel,
    image_base_dir: Path | None = None,
    system_prompt: str | None = None,
    invoke_kwargs: dict[str, Any] | None = None,
    on_progress: Any | None = None,
    on_error: Any | None = None,
    processed_ids: set[str] | None = None,
    on_result: Any | None = None,
    on_prediction: Any | None = None,
    max_workers: int | None = None,
    max_retries: int = 3,
    output_key: str = "classification",
    token_tracker: TokenTracker | None = None,
) -> list[dict[str, Any]]:
    """Process multiple classification tasks in batch with optional parallel execution.

    Args:
        vlm: Vision-Language Model instance to use for inference
        entries: List of dictionaries containing:
            - id: Unique identifier
            - text: Context text with classification criteria
            - images: List of images (str/Path filenames or PIL Image objects)
        llm: LLM for parsing VLM responses into structured format
        image_base_dir: Base directory for image files (if not absolute paths)
        system_prompt: Optional custom system prompt
        invoke_kwargs: Additional kwargs to pass to VLM
        on_progress: Optional callback function(entry_id, response)
        on_error: Optional callback function(entry_id, error)
        processed_ids: Set of entry IDs to skip (for resuming)
        on_result: Optional callback function(result, entry)
        on_prediction: Optional callback function(entry_id, classification_dict)
        max_workers: Maximum number of parallel workers
        max_retries: Maximum number of retry attempts
        output_key: Key name for the classification result (default: "classification")
        token_tracker: Optional TokenTracker to collect usage statistics

    Returns:
        List of dictionaries containing:
            - id: Entry identifier
            - vlm_response: Classification response
            - status: "success" or "error"
            - error: Error message (if status is "error")

    Example:
        ```python
        entries = [
            {
                "id": "gear_001",
                "text": "Identify this component",
                "images": ["gear1.png", "gear2.png"]
            },
            {
                "id": "shaft_002",
                "text": "Identify this component",
                "images": ["shaft1.png", "shaft2.png"]
            }
        ]

        results = batch_classify_assets(
            vlm=vlm,
            entries=entries,
            llm=llm,
            output_key="component"
        )
        ```
    """
    logger.debug(f"batch_classify_assets() with output_key='{output_key}'")

    entries_count = len(entries) if entries else 0
    skipped = len(processed_ids) if processed_ids else 0
    to_process = entries_count - skipped
    logger.info(f"Starting batch classification for {to_process} entries")

    def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(result)
        if normalized.get("status") == "success":
            normalized["vlm_response"] = normalize_stage1_prediction_payload(
                normalized.get("vlm_response"),
                output_key=output_key,
            )
        return normalized

    def _on_progress(entry_id: str, response: Any) -> None:
        if on_progress:
            on_progress(
                entry_id,
                normalize_stage1_prediction_payload(
                    response,
                    output_key=output_key,
                ),
            )

    def _on_prediction(entry_id: str, response: Any) -> None:
        if on_prediction:
            on_prediction(
                entry_id,
                normalize_stage1_prediction_payload(
                    response,
                    output_key=output_key,
                ),
            )

    def _on_result(result: dict[str, Any], entry: dict[str, Any]) -> None:
        if on_result:
            on_result(_normalize_result(result), entry)

    # Delegate to generic batch classification and normalize Joint Agent output.
    results = batch_classify_objects(
        vlm=vlm,
        entries=entries,
        llm=llm,
        image_base_dir=image_base_dir,
        system_prompt=system_prompt,
        invoke_kwargs=invoke_kwargs,
        on_progress=_on_progress,
        on_error=on_error,
        processed_ids=processed_ids,
        on_result=_on_result,
        on_prediction=_on_prediction,
        max_workers=max_workers,
        max_retries=max_retries,
        output_key=output_key,
        token_tracker=token_tracker,
    )
    return [_normalize_result(result) for result in results]


# Export all public functions
__all__ = [
    "classify_asset",
    "batch_classify_assets",
    "extract_answer_block",
    "get_fibonacci_delay",
]
