# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save predictions task for Joint Agent."""

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class SavePredictionsTask(Task):
    """Save predictions to file.

    This task saves predictions that were streamed during inference
    or are stored in the object store.

    Input context keys:
        - predictions_path: Path to predictions file (already saved if streaming)
        - predictions_count: Number of predictions

    Output context keys:
        - predictions_saved: Boolean indicating success
    """

    def __init__(self):
        """Initialize the task."""
        self.name = "SavePredictions"
        self.description = "Save predictions to file"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Save predictions.

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context
        """
        predictions_path = context.get("predictions_path")
        predictions_count = context.get("predictions_count", 0)

        if predictions_path:
            path = Path(predictions_path)
            if path.exists():
                logger.info(
                    "Predictions already saved to %s (%d entries)",
                    predictions_path,
                    predictions_count,
                )
                context["predictions_saved"] = True
                return context

        # If not streamed, save from object store
        if object_store and object_store.exists("predictions"):
            predictions = object_store.get("predictions")

            if not predictions_path:
                output_dir = context.get("output_dir", ".")
                predictions_path = Path(output_dir) / "predictions.jsonl"

            predictions_path = Path(predictions_path)
            predictions_path.parent.mkdir(parents=True, exist_ok=True)

            output_key = context.get("output_key", "classification")

            with open(predictions_path, "w", encoding="utf-8") as f:
                for pred in predictions:
                    entry = {
                        "id": pred.get("id"),
                        output_key: pred.get("vlm_response"),
                    }
                    f.write(json.dumps(entry) + "\n")

            logger.info(
                "Saved %d predictions to %s", len(predictions), predictions_path
            )
            context["predictions_path"] = str(predictions_path)
            context["predictions_saved"] = True
        else:
            logger.warning("No predictions to save")
            context["predictions_saved"] = False

        return context
