# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build USD dataset manifest task."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.usd_model import USDModel
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class USDDatasetManifestTask(Task):
    """Build USD dataset files from rendered prims."""

    def __init__(self):
        self.name = "USDDatasetManifest"
        self.description = "Create dataset.json and prims.jsonl files for USD data"

    def run(self, context: dict[str, Any], object_store: ObjectStore) -> dict[str, Any]:
        """Build and save USD dataset files.

        Expected context inputs:
            - output_dir: Base output directory for dataset
            - export_usd_model: Whether to include USD model (default: True)
            - usd_path: Path to original USD file

        Expected from object_store:
            - prim_data: List of prim info with image paths
            - usd_model: Optional USDModel for USD context
            - usd_stage: USD stage for additional metadata

        Updates context with:
            - dataset_path: Path to main dataset.json file
            - prims_path: Path to prims.jsonl file
            - num_prims: Number of prims in dataset
            - num_images: Total number of images
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Get inputs
        prim_data = object_store.get("prim_data", [])
        output_dir = Path(context.get("output_dir", "output"))
        export_usd_model = context.get("export_usd_model", True)
        usd_path = context.get("usd_path", "")

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        listener.info("Building USD dataset files")
        listener.info(f"  Prims to process: {len(prim_data)}")
        listener.info(f"  Output directory: {output_dir}")

        # Create prim entries with new structure
        prim_entries = self._create_prim_entries(prim_data)

        # Calculate statistics
        include_color_stats = context.get("include_display_color_statistics", False)
        statistics = self._calculate_statistics(prim_entries, include_color_stats)

        # Export USD model if requested
        usd_model_exported = False
        if export_usd_model:
            usd_model = object_store.get("usd_model")
            if isinstance(usd_model, USDModel):
                usd_model_path = output_dir / "usd_model.json"
                try:
                    usd_model.save_json(
                        usd_model_path, include_hierarchy=True, indent=2
                    )
                    listener.info(f"Saved USD model to {usd_model_path}")
                    usd_model_exported = True
                    context["usd_model_path"] = str(usd_model_path)
                except Exception as e:
                    listener.warning(f"Could not save USD model: {e}")

        # Create main dataset.json
        dataset = self._create_dataset_json(
            usd_path=str(usd_path),
            statistics=statistics,
            context=context,
            object_store=object_store,
            usd_model_exported=usd_model_exported,
        )

        # Save dataset.json
        dataset_path = output_dir / "dataset.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        listener.info(f"Saved dataset.json to {dataset_path}")

        # Save prims.jsonl
        prims_path = output_dir / "prims.jsonl"
        with open(prims_path, "w", encoding="utf-8") as f:
            for entry in prim_entries:
                # Clean the entry to ensure JSON serialization
                clean_entry = self._clean_for_json(entry)
                f.write(json.dumps(clean_entry) + "\n")
        listener.info(f"Saved prims.jsonl to {prims_path}")

        # Fail if no images were produced
        if statistics["total_images"] == 0:
            raise RuntimeError(
                "Dataset has 0 images. Rendering likely failed for all prims."
            )

        # Update context with results
        context["dataset_path"] = str(dataset_path)
        context["prims_path"] = str(prims_path)
        context["num_prims"] = statistics["total_prims"]
        context["num_images"] = statistics["total_images"]

        listener.info("USD dataset creation complete:")
        listener.info(f"  Total prims: {statistics['total_prims']}")
        listener.info(f"  Total images: {statistics['total_images']}")

        return context

    def _create_prim_entries(
        self,
        prim_data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create prim entries with new structure.

        Args:
            prim_data: List of prim info dicts from rendering

        Returns:
            List of prim entries for prims.jsonl
        """
        prim_entries = []
        for prim_info in prim_data:
            # Create prim entry with new structure
            entry = {
                "prim_path": prim_info["prim_path"],
                "renders": [
                    {
                        "view": img["view"],
                        "path": img["path"],
                        "camera": img.get("camera", "default"),
                        "render_mode": img.get("render_mode", "unknown"),
                    }
                    for img in prim_info.get("images", [])
                ],
            }

            # Add optional fields if present
            if prim_info.get("metadata"):
                entry["metadata"] = prim_info["metadata"]

            if prim_info.get("display_color"):
                entry["display_color"] = prim_info["display_color"]

            if prim_info.get("material_bindings"):
                entry["material_bindings"] = prim_info["material_bindings"]

            if prim_info.get("hierarchy"):
                entry["hierarchy"] = prim_info["hierarchy"]

            # Add world bounding box if present
            if prim_info.get("world_bbox"):
                entry["world_bbox"] = prim_info["world_bbox"]

            # Add MPU-scaled world bounding box if present
            if prim_info.get("world_bbox_meters"):
                entry["world_bbox_meters"] = prim_info["world_bbox_meters"]

            # Add relative metrics if present
            if prim_info.get("relative_metrics"):
                entry["relative_metrics"] = prim_info["relative_metrics"]

            prim_entries.append(entry)

        return prim_entries

    def _calculate_statistics(
        self,
        prim_entries: list[dict[str, Any]],
        include_display_color_stats: bool = False,
    ) -> dict[str, Any]:
        """Calculate USD dataset statistics.

        Args:
            prim_entries: List of prim entries
            include_display_color_stats: Whether to include unique display colors

        Returns:
            Statistics dictionary
        """
        total_images = sum(len(entry.get("renders", [])) for entry in prim_entries)

        # Count prim types
        type_distribution = {}
        for entry in prim_entries:
            prim_type = entry.get("metadata", {}).get("type", "unknown")
            if prim_type:
                type_distribution[str(prim_type)] = (
                    type_distribution.get(str(prim_type), 0) + 1
                )

        # Count collections
        all_collections = set()
        for entry in prim_entries:
            for coll in entry.get("hierarchy", {}).get("collections", []):
                all_collections.add(f"{coll['prim_path']}:{coll['name']}")

        statistics = {
            "total_prims": len(prim_entries),
            "total_images": total_images,
            "total_collections": len(all_collections),
            "type_distribution": type_distribution,
        }

        # Add unique display colors if requested
        if include_display_color_stats:
            unique_colors = set()
            prims_with_color = 0

            for entry in prim_entries:
                display_color = entry.get("display_color")
                if display_color:
                    prims_with_color += 1
                    # Convert list to tuple for set membership
                    # Round to 3 decimal places to group similar colors
                    rounded_color = tuple(round(c, 3) for c in display_color)
                    unique_colors.add(rounded_color)

            if unique_colors:
                # Convert back to lists for JSON serialization
                statistics["display_color_stats"] = {
                    "unique_colors": sorted([list(color) for color in unique_colors]),
                    "total_unique_colors": len(unique_colors),
                    "prims_with_color": prims_with_color,
                }

        return statistics

    def _create_dataset_json(
        self,
        usd_path: str,
        statistics: dict[str, Any],
        context: dict[str, Any],
        object_store: ObjectStore,
        usd_model_exported: bool,
    ) -> dict[str, Any]:
        """Create the main dataset.json structure.

        Args:
            usd_path: Path to original USD file
            statistics: Calculated statistics
            context: Workflow context
            object_store: Object store with USD data
            usd_model_exported: Whether USD model was exported to file

        Returns:
            Dataset dictionary
        """
        dataset = {
            "version": "1.0",
            "metadata": {
                "source_usd": usd_path,
                "created": datetime.now().isoformat(),
                "generator": "world-understanding-usd-data-prep-0.1.0",
            },
            "statistics": statistics,
            "prims_file": "prims.jsonl",
        }

        # Add stage up axis if available
        stage_up_axis = object_store.get("stage_up_axis")
        if stage_up_axis:
            dataset["stage_up_axis"] = stage_up_axis

        # Add meters per unit if available
        meters_per_unit = object_store.get("meters_per_unit")
        if meters_per_unit is not None:
            dataset["meters_per_unit"] = meters_per_unit

        # Add stage world bounding box if available
        stage_world_bbox = object_store.get("stage_world_bbox")
        if stage_world_bbox:
            dataset["stage_world_bbox"] = stage_world_bbox

        # Add MPU-scaled stage world bbox if available
        stage_world_bbox_meters = object_store.get("stage_world_bbox_meters")
        if stage_world_bbox_meters:
            dataset["stage_world_bbox_meters"] = stage_world_bbox_meters

        # Add reference to USD model file if it was exported
        if usd_model_exported:
            dataset["usd_model_file"] = "usd_model.json"

        # Add render settings from context
        renderer_config = context.get("renderer_config", {})
        dataset["render_settings"] = {
            "image_width": renderer_config.get("image_width", 512),
            "image_height": renderer_config.get("image_height", 512),
            "camera_type": renderer_config.get("camera_view_type", "corner"),
            "backend": renderer_config.get("backend", "remote"),
        }

        return dataset

    def _clean_for_json(self, obj: Any) -> Any:
        """Recursively clean an object for JSON serialization."""
        import numpy as np

        if obj is None:
            return None
        elif isinstance(obj, str | int | float | bool):
            return obj
        elif isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list | tuple):
            return [self._clean_for_json(item) for item in obj]
        elif hasattr(obj, "tolist"):  # NumPy arrays
            return obj.tolist()
        elif hasattr(obj, "__array__"):  # USD arrays
            return np.array(obj).tolist()
        else:
            # Try to convert to string
            try:
                return str(obj)
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Failed to serialize object of type {type(obj)}: {e}")
                return None
