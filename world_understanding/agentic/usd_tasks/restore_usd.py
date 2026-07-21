# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for restoring predictions from optimized USD to original USD structure."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from pxr import Usd, UsdGeom

from world_understanding.agentic.events import EventListener, get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)


@dataclass
class RestorationStats:
    """Statistics from a predictions restoration run."""

    total_originals: int = 0
    identity_count: int = 0
    dedup_count: int = 0
    split_count: int = 0
    split_dedup_count: int = 0
    predictions_consumed: set[str] = field(default_factory=set)
    predictions_written: int = 0
    uncovered_originals: list[str] = field(default_factory=list)
    unconsumed_predictions: list[str] = field(default_factory=list)
    restored_prim_sources: dict[str, str] = field(default_factory=dict)
    expected_target_count: int = 0
    mapping_complete: bool = True
    mapping_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "total_originals": self.total_originals,
            "identity_count": self.identity_count,
            "dedup_count": self.dedup_count,
            "split_count": self.split_count,
            "split_dedup_count": self.split_dedup_count,
            "predictions_consumed": len(self.predictions_consumed),
            "predictions_written": self.predictions_written,
            "uncovered_originals": self.uncovered_originals,
            "unconsumed_predictions": self.unconsumed_predictions,
            "restored_prim_sources": self.restored_prim_sources,
            "expected_target_count": self.expected_target_count,
            "mapping_complete": self.mapping_complete,
            "mapping_warnings": self.mapping_warnings,
        }


class RestoreUSDTask(Task):
    """Task to transform predictions.jsonl from optimized USD structure to original USD structure.

    This task takes predictions generated on an optimized USD and transforms the prim IDs/paths
    back to the original USD structure using the optimization metadata. This allows predictions
    to be applied to the original USD file.

    The optimizer performs 3 operations (in order): deinstance, split, deduplicate.
    All 8 combinations are supported (no_op, D, S, P, DS, DP, SP, DSP).

    Mapping patterns handled:
        - Identity (no_op/D): 1:1 — optimized path equals original path
        - Dedup (P/DP): N:1 — multiple originals share one prototype prediction
        - Split (S/DS): 1:N — one original maps to multiple prototype parts (GeomSubsets)
        - Split+Dedup (SP/DSP): combined — split parts may be deduplicated to same prototype

    Input context keys:
        - original_usd_path: Path to the original USD (before optimization)
        - optimization_metadata: Metadata from optimize_usd step (correspondence_map, etc.)
        - predictions_path: Path to predictions.jsonl from predict/benchmark step
        - output_predictions_path: Path where restored predictions will be saved

    Output context keys:
        - restored_predictions_path: Path to the restored predictions file
        - restore_success: Boolean indicating success
        - predictions_count: Number of predictions processed
        - restore_stats: Detailed restoration statistics
    """

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Execute predictions restoration.

        Args:
            context: Workflow context with input parameters
            object_store: Optional object store (not used)

        Returns:
            Updated context with restoration results

        Raises:
            ValueError: If required parameters are missing
            Exception: If restoration fails
        """
        listener = get_listener(context)

        # Get input parameters
        original_usd_path = context.get("original_usd_path")
        predictions_path = context.get("predictions_path")
        output_predictions_path = context.get("output_predictions_path")
        optimization_metadata = context.get("optimization_metadata", {})

        # Validate required inputs
        if not original_usd_path:
            raise ValueError("original_usd_path is required in context")
        if not predictions_path:
            raise ValueError("predictions_path is required in context")
        if not output_predictions_path:
            raise ValueError("output_predictions_path is required in context")

        original_usd_path = Path(original_usd_path)
        predictions_path = Path(predictions_path)
        output_predictions_path = Path(output_predictions_path)

        # Check that input files exist
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

        listener.info("Restoring predictions from optimized to original USD structure")
        listener.info(f"  Original USD: {original_usd_path}")
        listener.info(f"  Input predictions: {predictions_path}")
        listener.info(f"  Output predictions: {output_predictions_path}")

        if optimization_metadata:
            listener.info(
                f"Using optimization metadata: {list(optimization_metadata.keys())}"
            )
        else:
            listener.warning(
                "No optimization metadata provided - predictions will be copied as-is"
            )

        try:
            # Transform predictions
            predictions_count, stats = self._transform_predictions(
                predictions_path,
                output_predictions_path,
                original_usd_path,
                optimization_metadata,
                listener,
            )

            # Update context with results
            context["restored_predictions_path"] = str(output_predictions_path)
            context["restore_success"] = True
            context["predictions_count"] = predictions_count
            context["restore_stats"] = stats.to_dict()

            listener.info("Predictions restoration completed")
            listener.info(
                f"Restored {predictions_count} predictions to: {output_predictions_path}"
            )

            # Log stats summary
            listener.info(
                f"Stats: identity={stats.identity_count}, dedup={stats.dedup_count}, "
                f"split={stats.split_count}, split_dedup={stats.split_dedup_count}"
            )
            if stats.unconsumed_predictions:
                listener.warning(
                    f"{len(stats.unconsumed_predictions)} input predictions not "
                    f"consumed: {stats.unconsumed_predictions[:5]}"
                )
            if stats.uncovered_originals:
                listener.warning(
                    f"{len(stats.uncovered_originals)} original prims not covered: "
                    f"{stats.uncovered_originals[:5]}"
                )

        except Exception as e:
            listener.error(f"Predictions restoration failed: {e}")
            context["restore_success"] = False
            context["restore_error"] = str(e)
            raise

        return context

    def _transform_predictions(
        self,
        predictions_path: Path,
        output_predictions_path: Path,
        original_usd_path: Path,
        optimization_metadata: dict[str, Any],
        listener: EventListener,
    ) -> tuple[int, RestorationStats]:
        """Transform predictions from optimized to original USD structure.

        Args:
            predictions_path: Path to input predictions.jsonl
            output_predictions_path: Path to output restored predictions.jsonl
            original_usd_path: Path to original USD (before optimization)
            optimization_metadata: Metadata from optimize_usd step
            listener: Event listener for logging

        Returns:
            Tuple of (predictions_written, stats)
        """
        # Ensure output directory exists
        output_predictions_path.parent.mkdir(parents=True, exist_ok=True)

        stats = RestorationStats()

        # Extract mappings from correspondence_map
        if not isinstance(optimization_metadata, dict):
            optimization_metadata = {}
        correspondence_map = optimization_metadata.get("correspondence_map", {})
        if not isinstance(correspondence_map, dict):
            correspondence_map = {}
        full_mapping = correspondence_map.get("full_mapping", {})
        if not isinstance(full_mapping, dict):
            full_mapping = {}
        original_to_prototype = full_mapping.get("original_to_prototype", {})
        split_mapping = correspondence_map.get("split_mapping", {})

        if not isinstance(original_to_prototype, dict) or not original_to_prototype:
            stats.mapping_complete = False
            stats.mapping_warnings.append(
                "No usable original_to_prototype correspondence mapping was available."
            )
            listener.warning(
                "No original_to_prototype mapping found - "
                "predictions will be copied as-is"
            )
            # Copy predictions as-is if no mapping available
            with open(predictions_path, encoding="utf-8") as f_in:
                with open(output_predictions_path, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        line = line.strip()
                        if line:
                            f_out.write(line + "\n")
                            stats.predictions_written += 1
            return stats.predictions_written, stats

        if not isinstance(split_mapping, dict):
            stats.mapping_complete = False
            stats.mapping_warnings.append(
                "split_mapping was malformed; restored target scope is unqualified."
            )
            split_mapping = {}

        # Load all predictions keyed by their ID (optimized prim path)
        predictions_by_id: dict[str, dict] = {}
        with open(predictions_path, encoding="utf-8") as f_in:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    prediction = json.loads(line)
                    prediction_id = prediction.get("id")
                    if prediction_id:
                        predictions_by_id[prediction_id] = prediction
                except json.JSONDecodeError as e:
                    listener.warning(f"Skipping invalid JSON on line {line_num}: {e}")

        listener.info(f"Loaded {len(predictions_by_id)} predictions from file")

        # Log operations that were run
        summary = correspondence_map.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        ops_run = summary.get("operations_run", {})
        listener.info(f"Optimizer operations: {ops_run}")

        # Open original USD stage for GeomSubset inspection (needed for split cases)
        stage = self._open_stage(original_usd_path, listener)

        stats.total_originals = len(original_to_prototype)

        # Process each original prim
        with open(output_predictions_path, "w", encoding="utf-8") as f_out:
            for original_path, prototype_paths in original_to_prototype.items():
                if not self._is_absolute_prim_path(original_path):
                    stats.mapping_complete = False
                    stats.mapping_warnings.append(
                        f"Invalid original prim path in correspondence map: {original_path!r}."
                    )
                    continue

                # Normalize to list
                if not isinstance(prototype_paths, list):
                    prototype_paths = [prototype_paths]

                if not prototype_paths or not all(
                    self._is_absolute_prim_path(path) for path in prototype_paths
                ):
                    stats.mapping_complete = False
                    stats.mapping_warnings.append(
                        f"Invalid prototype mapping for original prim {original_path}."
                    )
                    continue

                is_split = original_path in split_mapping
                unique_prototypes = list(dict.fromkeys(prototype_paths))

                if is_split:
                    split_paths = split_mapping.get(original_path)
                    if not isinstance(split_paths, list) or len(split_paths) != len(
                        prototype_paths
                    ):
                        stats.mapping_complete = False
                        stats.mapping_warnings.append(
                            "Split correspondence count mismatch for "
                            f"{original_path}: expected {len(prototype_paths)} entries."
                        )
                elif len(unique_prototypes) != 1:
                    stats.mapping_complete = False
                    stats.mapping_warnings.append(
                        f"Multiple prototypes for {original_path} lacked split metadata."
                    )

                if not is_split and len(unique_prototypes) == 1:
                    # Single prototype: identity or dedup case
                    self._handle_single_prototype(
                        original_path,
                        unique_prototypes[0],
                        predictions_by_id,
                        stats,
                        f_out,
                        listener,
                    )
                else:
                    # Multiple prototypes or split: split or split+dedup case
                    self._handle_split(
                        original_path,
                        prototype_paths,
                        predictions_by_id,
                        stage,
                        stats,
                        f_out,
                        listener,
                    )

        # Compute unconsumed predictions
        stats.unconsumed_predictions = sorted(
            set(predictions_by_id.keys()) - stats.predictions_consumed
        )
        if len(stats.restored_prim_sources) != stats.expected_target_count:
            stats.mapping_complete = False
            stats.mapping_warnings.append(
                "Restored target mapping did not contain every expected target: "
                f"expected {stats.expected_target_count}, recorded "
                f"{len(stats.restored_prim_sources)}."
            )

        listener.info(f"Wrote {stats.predictions_written} restored predictions")
        return stats.predictions_written, stats

    def _handle_single_prototype(
        self,
        original_path: str,
        prototype_path: str,
        predictions_by_id: dict[str, dict],
        stats: RestorationStats,
        f_out: TextIO,
        listener: EventListener,
    ) -> None:
        """Handle 1:1 mapping (identity or dedup).

        Args:
            original_path: Original prim path
            prototype_path: Single prototype path
            predictions_by_id: All loaded predictions
            stats: Stats to update
            f_out: Output file handle
            listener: Event listener
        """
        mapping_accepted = self._record_restored_prim_source(
            stats,
            restored_prim_id=original_path,
            optimized_prim_id=prototype_path,
        )
        if not mapping_accepted:
            stats.uncovered_originals.append(original_path)
            return

        if prototype_path in predictions_by_id:
            prediction = predictions_by_id[prototype_path].copy()
            prediction["id"] = original_path
            f_out.write(json.dumps(prediction) + "\n")
            stats.predictions_written += 1
            stats.predictions_consumed.add(prototype_path)

            if prototype_path == original_path:
                stats.identity_count += 1
            else:
                stats.dedup_count += 1
        else:
            stats.uncovered_originals.append(original_path)
            listener.warning(
                f"No prediction found for prototype: {prototype_path} "
                f"(original: {original_path})"
            )

    def _handle_split(
        self,
        original_path: str,
        prototype_paths: list[str],
        predictions_by_id: dict[str, dict],
        stage: Usd.Stage | None,
        stats: RestorationStats,
        f_out: TextIO,
        listener: EventListener,
    ) -> None:
        """Handle 1:N mapping (split or split+dedup).

        Each prototype corresponds to a GeomSubset child of the original prim.
        The positional correspondence is: prototype_paths[i] -> GeomSubset[i].

        In the split+dedup case, prototype_paths may contain duplicate entries
        (the same deduplicated prototype appearing multiple times).

        Args:
            original_path: Original prim path
            prototype_paths: List of prototype paths (may have duplicates in dedup case)
            predictions_by_id: All loaded predictions
            stage: USD stage for GeomSubset inspection (may be None)
            stats: Stats to update
            f_out: Output file handle
            listener: Event listener
        """
        unique_prototypes = list(dict.fromkeys(prototype_paths))
        has_dedup = len(prototype_paths) != len(unique_prototypes)

        if has_dedup:
            stats.split_dedup_count += 1
        else:
            stats.split_count += 1

        # Get GeomSubset paths from the original USD
        geomsubset_paths = (
            self._get_geomsubset_paths(stage, original_path, listener) if stage else []
        )

        if len(geomsubset_paths) != len(prototype_paths):
            stats.mapping_complete = False
            stats.mapping_warnings.append(
                f"GeomSubset count mismatch for {original_path}: "
                f"{len(prototype_paths)} prototypes vs "
                f"{len(geomsubset_paths)} GeomSubsets."
            )
            listener.warning(
                f"Count mismatch for {original_path}: "
                f"{len(prototype_paths)} prototypes vs "
                f"{len(geomsubset_paths)} GeomSubsets"
            )

        any_found = False
        for i, prototype_path in enumerate(prototype_paths):
            restored_prim_id = (
                geomsubset_paths[i]
                if i < len(geomsubset_paths)
                else f"{original_path}_part_{i}"
            )
            mapping_accepted = self._record_restored_prim_source(
                stats,
                restored_prim_id=restored_prim_id,
                optimized_prim_id=prototype_path,
            )
            if not mapping_accepted:
                continue

            if prototype_path in predictions_by_id:
                prediction = predictions_by_id[prototype_path].copy()

                # Assign the restored ID: GeomSubset path or indexed fallback
                prediction["id"] = restored_prim_id

                f_out.write(json.dumps(prediction) + "\n")
                stats.predictions_written += 1
                stats.predictions_consumed.add(prototype_path)
                any_found = True
            else:
                listener.warning(
                    f"No prediction found for prototype: {prototype_path} "
                    f"(original: {original_path}, part {i})"
                )

        if not any_found:
            stats.uncovered_originals.append(original_path)

    @staticmethod
    def _is_absolute_prim_path(value: object) -> bool:
        """Return whether *value* is an absolute USD prim path string."""
        return isinstance(value, str) and value.startswith("/")

    def _record_restored_prim_source(
        self,
        stats: RestorationStats,
        *,
        restored_prim_id: str,
        optimized_prim_id: str,
    ) -> bool:
        """Record a canonical restored mapping and return whether it was accepted."""
        stats.expected_target_count += 1
        if not self._is_absolute_prim_path(
            restored_prim_id
        ) or not self._is_absolute_prim_path(optimized_prim_id):
            stats.mapping_complete = False
            stats.mapping_warnings.append(
                "Invalid restored/source prim mapping: "
                f"{restored_prim_id!r} -> {optimized_prim_id!r}."
            )
            return False

        if restored_prim_id in stats.restored_prim_sources:
            stats.mapping_complete = False
            stats.mapping_warnings.append(
                f"Duplicate restored target prim ID: {restored_prim_id}."
            )
            return False

        stats.restored_prim_sources[restored_prim_id] = optimized_prim_id
        return True

    def _open_stage(self, usd_path: Path, listener: EventListener) -> Usd.Stage | None:
        """Open a USD stage, returning None on failure.

        Args:
            usd_path: Path to USD file
            listener: Event listener for logging

        Returns:
            Opened stage or None
        """
        try:
            stage = Usd.Stage.Open(str(usd_path))
            if not stage:
                listener.warning(f"Failed to open USD stage: {usd_path}")
            return stage
        except Exception as e:
            listener.warning(f"Error opening USD stage: {e}")
            return None

    def _get_geomsubset_paths(
        self, stage: Usd.Stage, prim_path: str, listener: EventListener
    ) -> list[str]:
        """Get list of GeomSubset paths under a given prim.

        Args:
            stage: USD stage to inspect
            prim_path: Path to the prim to inspect
            listener: Event listener for logging

        Returns:
            List of full GeomSubset paths (e.g., ["/path/to/mesh/subset1", ...])
        """
        geomsubset_paths = []

        try:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                listener.warning(f"Invalid prim at path: {prim_path}")
                return geomsubset_paths

            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Subset):
                    geomsubset_paths.append(str(child.GetPath()))

            if geomsubset_paths:
                listener.debug(
                    f"Found {len(geomsubset_paths)} GeomSubsets under {prim_path}"
                )

        except Exception as e:
            listener.warning(f"Error inspecting GeomSubsets for {prim_path}: {e}")

        return geomsubset_paths
