# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for identifying unique materials from prediction files."""

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

from material_agent.materials import (
    FALLBACK_MATERIAL_NAME,
    PREDICTION_CONTAINER_KEYS,
    PREDICTION_ID_KEYS,
    PREDICTION_MATERIAL_KEYS,
    UNKNOWN_MATERIAL_SENTINEL,
    is_actionable_material_name,
    is_unknown_material_name,
    normalize_material_name,
)

logger = logging.getLogger(__name__)


class IdentifyUniqueMaterialsTask(Task):
    """Task to identify unique materials from prediction JSON files.

    This task loads a predictions file (JSON or JSONL format) and extracts
    all unique material names from the predictions. It handles both single
    JSON objects and JSONL format with multiple prediction entries.

    Input context keys:
        - predictions_path: Path to the predictions file (JSON or JSONL)

    Output context keys:
        - unique_materials: List of unique material names found in predictions
        - predictions_data: The loaded predictions data for potential future use
        - total_predictions: Number of prediction entries processed
    """

    def __init__(self):
        """Initialize the identify unique materials task."""
        self.name = "IdentifyUniqueMaterials"
        self.description = "Identify unique materials from prediction files"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Identify unique materials from predictions.

        Args:
            context: Workflow context containing predictions_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with unique materials
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)
        predictions_path = context.get("predictions_path")
        if not predictions_path:
            raise ValueError("predictions_path not provided in context")

        predictions_path = Path(predictions_path)
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

        listener.info(f"Loading predictions from {predictions_path}")

        # Load predictions data
        predictions_data = self._load_predictions(predictions_path, listener)

        # Extract unique materials
        unique_materials = self._extract_unique_materials(predictions_data, listener)
        unknown_predictions = self._extract_unknown_predictions(predictions_data)

        listener.info(
            f"Found {len(unique_materials)} unique materials from {len(predictions_data)} predictions"
        )
        if unknown_predictions:
            listener.warning(
                f"{len(unknown_predictions)} prediction(s) were classified as "
                f"'{UNKNOWN_MATERIAL_SENTINEL}' and will use "
                f"'{FALLBACK_MATERIAL_NAME}'."
            )

        if logger.isEnabledFor(logging.DEBUG):
            listener.debug(f"Unique materials: {unique_materials}")

        # Update context
        context["unique_materials"] = unique_materials
        context["predictions_data"] = predictions_data
        context["total_predictions"] = len(predictions_data)
        existing_unknown_count = context.get("unknown_material_predictions", 0)
        if not isinstance(existing_unknown_count, int):
            existing_unknown_count = 0
        context["unknown_material_predictions"] = max(
            existing_unknown_count,
            len(unknown_predictions),
        )
        context["unknown_material_prediction_ids"] = unknown_predictions

        return context

    def _load_predictions(self, predictions_path: Path, listener) -> list[Any]:
        """Load predictions from JSON or JSONL file.

        Args:
            predictions_path: Path to the predictions file
            listener: Event listener for logging

        Returns:
            List of prediction dictionaries

        Raises:
            ValueError: If file format is not supported or data is invalid
        """
        try:
            with open(predictions_path, encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                listener.warning("Predictions file is empty")
                return []

            # Try to parse as JSON first (single object or array)
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    # Single prediction object
                    return [data]
                elif isinstance(data, list):
                    # Array of predictions
                    return data
                else:
                    raise ValueError(f"Unexpected JSON root type: {type(data)}")
            except json.JSONDecodeError:
                # Try to parse as JSONL (one JSON object per line)
                predictions = []
                for line_num, line in enumerate(content.split("\n"), 1):
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    try:
                        prediction = json.loads(line)
                        predictions.append(prediction)
                    except json.JSONDecodeError as e:
                        listener.warning(
                            f"Skipping invalid JSON on line {line_num}: {e}"
                        )
                        continue

                if not predictions:
                    raise ValueError("No valid JSON objects found in file") from None

                return predictions

        except Exception as e:
            raise ValueError(
                f"Failed to load predictions from {predictions_path}: {str(e)}"
            ) from e

    def _extract_unique_materials(
        self, predictions_data: list[Any], listener
    ) -> list[str]:
        """Extract unique materials from prediction data.

        Args:
            predictions_data: List of prediction dictionaries
            listener: Event listener for logging

        Returns:
            List of unique material names (sorted)
        """
        unique_materials = set()

        for i, prediction in enumerate(predictions_data):
            try:
                materials = self._extract_materials_from_prediction(prediction)
                unique_materials.update(materials)

            except Exception as e:
                listener.warning(
                    f"Failed to extract materials from prediction {i}: {e}"
                )
                continue

        # Return sorted list for consistent ordering
        return sorted(unique_materials)

    def _extract_materials_from_container(self, container: Any) -> set[str]:
        """Extract actionable material names from nested prediction containers."""
        materials: set[str] = set()
        if isinstance(container, list):
            for item in container:
                materials.update(self._extract_materials_from_prediction(item))
        elif isinstance(container, dict):
            for value in container.values():
                materials.update(self._extract_materials_from_prediction(value))
        return materials

    def _extract_unknown_predictions(self, predictions_data: list[Any]) -> list[str]:
        """Return prediction IDs whose material is the unknown sentinel."""
        prediction_ids: list[str] = []

        for index, prediction in enumerate(predictions_data):
            prediction_ids.extend(
                self._extract_unknown_prediction_ids(
                    prediction,
                    fallback_id=f"index:{index}",
                )
            )

        return prediction_ids

    def _extract_unknown_prediction_ids(
        self,
        prediction: Any,
        *,
        fallback_id: str,
    ) -> list[str]:
        """Return unknown prediction IDs from a record or nested container."""
        if not isinstance(prediction, dict):
            return [fallback_id] if is_unknown_material_name(prediction) else []

        if self._prediction_contains_unknown_material(prediction):
            for key in PREDICTION_ID_KEYS:
                value = prediction.get(key)
                if isinstance(value, str) and value:
                    return [value]
            return [fallback_id]

        prediction_ids: list[str] = []
        for container_key in PREDICTION_CONTAINER_KEYS:
            container = prediction.get(container_key)
            if isinstance(container, list):
                for nested_index, sub_prediction in enumerate(container):
                    prediction_ids.extend(
                        self._extract_unknown_prediction_ids(
                            sub_prediction,
                            fallback_id=f"{fallback_id}.{nested_index}",
                        )
                    )
            elif isinstance(container, dict):
                for key, sub_prediction in container.items():
                    child_fallback = (
                        str(key)
                        if isinstance(key, str) and key.startswith("/")
                        else f"{fallback_id}.{key}"
                    )
                    prediction_ids.extend(
                        self._extract_unknown_prediction_ids(
                            sub_prediction,
                            fallback_id=child_fallback,
                        )
                    )

        for key, sub_prediction in prediction.items():
            if key in PREDICTION_CONTAINER_KEYS:
                continue
            if isinstance(key, str) and key.startswith("/"):
                prediction_ids.extend(
                    self._extract_unknown_prediction_ids(
                        sub_prediction,
                        fallback_id=key,
                    )
                )

        return prediction_ids

    def _prediction_contains_unknown_material(self, prediction: Any) -> bool:
        """Return True if a prediction contains the unknown material sentinel."""
        if not isinstance(prediction, dict):
            return is_unknown_material_name(prediction)

        return is_unknown_material_name(
            self._selected_material_from_prediction(prediction)
        )

    def _selected_material_from_prediction(
        self, prediction: dict[str, Any]
    ) -> str | None:
        """Extract the primary selected material from a prediction entry."""

        def normalize_selected_material(material: str) -> str:
            if is_unknown_material_name(material):
                return UNKNOWN_MATERIAL_SENTINEL
            return normalize_material_name(material)

        materials = prediction.get("materials")
        if isinstance(materials, dict):
            material = materials.get("material")
            return (
                normalize_selected_material(material)
                if isinstance(material, str)
                else None
            )
        if isinstance(materials, str):
            return normalize_selected_material(materials)

        for key in PREDICTION_MATERIAL_KEYS:
            value = prediction.get(key)
            if isinstance(value, str):
                return normalize_selected_material(value)

        return None

    def _extract_materials_from_prediction(self, prediction: Any) -> list[str]:
        """Extract materials from a single prediction entry.

        This method handles various possible formats for material data in predictions.
        It looks for common field names that might contain material information.

        Args:
            prediction: Single prediction dictionary

        Returns:
            List of material names from this prediction
        """
        selected_material = (
            self._selected_material_from_prediction(prediction)
            if isinstance(prediction, dict)
            else None
        )
        materials = (
            [selected_material]
            if selected_material is not None
            else self._extract_all_materials_from_prediction(prediction)
        )

        # Remove duplicates and filter out empty/non-actionable sentinel strings.
        normalized_materials = set()
        for material in materials:
            if is_unknown_material_name(material):
                normalized_materials.add(FALLBACK_MATERIAL_NAME)
            elif is_actionable_material_name(material):
                normalized_materials.add(normalize_material_name(material))
        return sorted(normalized_materials)

    def _extract_all_materials_from_prediction(self, prediction: Any) -> list[str]:
        """Extract all material-like strings from a prediction entry."""
        if isinstance(prediction, str):
            return [prediction]
        if not isinstance(prediction, dict):
            return []

        materials = []

        # Common field names that might contain material information
        material_fields = [
            "materials",
            *PREDICTION_MATERIAL_KEYS,
            "predicted_materials",
            "material_predictions",
            "assigned_materials",
        ]

        for field_name in material_fields:
            if field_name in prediction:
                field_value = prediction[field_name]
                extracted = self._extract_materials_from_field(field_value)
                materials.extend(extracted)

        for container_key in PREDICTION_CONTAINER_KEYS:
            materials.extend(
                self._extract_materials_from_container(prediction.get(container_key))
            )

        for key, value in prediction.items():
            if isinstance(key, str) and key.startswith("/"):
                materials.extend(self._extract_all_materials_from_prediction(value))

        return materials

    def _extract_materials_from_field(self, field_value: Any) -> list[str]:
        """Extract material names from a field value.

        Args:
            field_value: The value of a field that might contain materials

        Returns:
            List of material names extracted from the field
        """
        materials = []

        if isinstance(field_value, str):
            # Single material string
            materials.append(field_value)
        elif isinstance(field_value, list):
            # List of materials
            for item in field_value:
                if isinstance(item, str):
                    materials.append(item)
                elif isinstance(item, dict):
                    # Material might be in a dictionary structure
                    # Look for common keys
                    for key in ["name", "material", "type", "value"]:
                        if key in item and isinstance(item[key], str):
                            materials.append(item[key])
                            break
        elif isinstance(field_value, dict):
            # Materials might be in a dictionary structure
            # First check for specific material keys
            material_keys = [
                "material",
                "name",
                "type",
                "value",
                "material_name",
                "material_type",
            ]
            for key in material_keys:
                if key in field_value and isinstance(field_value[key], str):
                    materials.append(field_value[key])
                    # Don't break here - there might be multiple material keys

            # If no specific material keys found, look for lists within the dict
            if not materials:
                for _key, value in field_value.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                materials.append(item)

        return materials
