# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic classification functions for world_understanding.

This module provides generic VLM-based object classification that can work
with any class labels (materials, vehicle types, fabric patterns, etc.).

Key functions:
- classify_object(): Single object classification
- batch_classify_objects(): Batch classification with parallel/sequential processing

Example:
    ```python
    from world_understanding.functions.classification import classify_object
    from world_understanding.functions.models import create_vlm, create_chat_model

    vlm = create_vlm(backend="nim", model="google/gemma-4-31b-it")
    llm = create_chat_model(backend="nim")

    result = classify_object(
        vlm=vlm,
        text="This is a vehicle. Available types: sedan, SUV, truck",
        images=["vehicle.jpg"],
        llm=llm,
        output_key="vehicle_type"
    )
    print(result["vehicle_type"])  # e.g., "sedan"
    ```
"""

from world_understanding.functions.classification.inference import (
    batch_classify_objects,
    classify_object,
    classify_objects_multi_prim,
    extract_answer_block,
    get_fibonacci_delay,
)
from world_understanding.functions.classification.types import (
    ClassificationEntry,
    ClassificationResult,
)

__all__ = [
    "classify_object",
    "classify_objects_multi_prim",
    "batch_classify_objects",
    "extract_answer_block",
    "get_fibonacci_delay",
    "ClassificationEntry",
    "ClassificationResult",
]
