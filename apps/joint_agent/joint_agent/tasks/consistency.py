# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prediction consistency-pass task."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

from joint_agent.functions.consistency import (
    apply_prediction_consistency,
    load_predictions_jsonl,
    write_predictions_jsonl,
)

logger = logging.getLogger(__name__)


class ConsistencyPassTask(Task):
    """Annotate and optionally harmonize repeated prediction groups."""

    def __init__(self) -> None:
        self.name = "ConsistencyPass"
        self.description = "Post-process predictions for repeated-part consistency"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        predictions_path = Path(context["predictions_path"])
        output_predictions_path = Path(context["output_predictions_path"])
        output_stats_path = Path(context["output_stats_path"])

        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

        predictions = load_predictions_jsonl(predictions_path)
        listener.info(f"Loaded {len(predictions)} predictions for consistency pass")

        harmonize_motion_profiles = context.get("harmonize_motion_profiles", False)
        if not isinstance(harmonize_motion_profiles, bool):
            raise ValueError("harmonize_motion_profiles must be a boolean")

        consistent_predictions, stats = apply_prediction_consistency(
            predictions,
            output_key=context.get("output_key", "classification"),
            min_group_size=context.get("min_group_size", 2),
            min_majority_fraction=context.get("min_majority_fraction", 0.6),
            harmonize_fields=context.get("harmonize_fields", []),
            harmonize_motion_profiles=harmonize_motion_profiles,
            add_role=context.get("add_role", True),
            add_instance_id=context.get("add_instance_id", True),
            signature_depth=context.get("signature_depth", 2),
        )

        write_predictions_jsonl(output_predictions_path, consistent_predictions)
        output_stats_path.parent.mkdir(parents=True, exist_ok=True)
        with output_stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        listener.info(
            "Consistency pass complete: "
            f"{stats['groups_repeated']} repeated groups, "
            f"{stats['annotated_predictions']} annotated, "
            f"{stats['harmonized_predictions']} harmonized"
        )

        context["consistent_predictions_path"] = str(output_predictions_path)
        context["consistency_stats_path"] = str(output_stats_path)
        context["consistency_stats"] = stats
        context["predictions_count"] = len(consistent_predictions)
        return context
