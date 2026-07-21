# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch processing utilities for material agent workflows."""

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from world_understanding.utils.credentials import (
    create_directory_with_safe_diagnostics,
    path_exists_with_safe_diagnostics,
    redact_sensitive_path,
)

logger = logging.getLogger(__name__)

_BATCH_ITEM_FAILURE_MESSAGE = "USD file processing failed"
_BATCH_SOURCE_INSPECTION_FAILURE_MESSAGE = "Unable to inspect USD directory"
_BATCH_OUTPUT_SETUP_FAILURE_MESSAGE = "Unable to create batch output directory"


def _safe_batch_identifier(
    usd_file: Path,
    position: int,
    used_identifiers: set[str],
) -> str:
    """Return a non-secret, collision-safe identifier for batch status output."""
    safe_name = redact_sensitive_path(usd_file.name)
    base_identifier = (
        f"file_{position}" if safe_name != usd_file.name else usd_file.stem
    )
    identifier = base_identifier
    suffix = 2
    while identifier in used_identifiers:
        identifier = f"{base_identifier}_{suffix}"
        suffix += 1
    return identifier


async def process_usd_batch(
    usd_dir: Path,
    batch_output_dir: Path,
    workflow_runner: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    base_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process multiple USD files in batch mode asynchronously.

    This utility handles the common pattern of:
    1. Finding all USD files in a directory
    2. Running a workflow for each file asynchronously
    3. Tracking success/failure
    4. Aggregating results

    Args:
        usd_dir: Directory containing USD files
        batch_output_dir: Base output directory for results
        workflow_runner: Async callable that takes context dict and returns result dict
        base_context: Optional base context to merge with per-file context

    Returns:
        Dictionary with batch processing results:
        - output_dir: Base output directory
        - num_files_processed: Number of successfully processed files
        - num_files_failed: Number of failed files
        - total_files: Total number of files found
        - results: Dictionary mapping filename to result details

    Raises:
        RuntimeError: If no USD files found or if all files fail to process
    """
    if not path_exists_with_safe_diagnostics(usd_dir, label="USD directory"):
        raise RuntimeError("USD directory not found")

    # Find all USD files recursively
    source_inspection_failed = False
    try:
        usd_files = (
            list(usd_dir.rglob("*.usd"))
            + list(usd_dir.rglob("*.usda"))
            + list(usd_dir.rglob("*.usdc"))
        )
    except (OSError, RuntimeError):
        source_inspection_failed = True
        usd_files = []
    if source_inspection_failed:
        raise RuntimeError(_BATCH_SOURCE_INSPECTION_FAILURE_MESSAGE)

    if not usd_files:
        raise RuntimeError("No USD files found in directory")

    logger.info(f"Found {len(usd_files)} USD files to process")
    logger.info("  USD directory: %s", redact_sensitive_path(usd_dir))
    logger.info("  Output directory: %s", redact_sensitive_path(batch_output_dir))

    # Create base output directory
    output_setup_failed = False
    try:
        create_directory_with_safe_diagnostics(
            batch_output_dir,
            label="batch output directory",
        )
    except (OSError, RuntimeError):
        output_setup_failed = True
    if output_setup_failed:
        raise RuntimeError(_BATCH_OUTPUT_SETUP_FAILURE_MESSAGE)

    # Process each USD file
    successful = 0
    failed = 0
    results = {}
    used_result_keys: set[str] = set()
    base_context = base_context or {}

    for position, usd_file in enumerate(usd_files, start=1):
        usd_name = usd_file.stem
        dataset_output_dir = batch_output_dir / usd_name
        safe_result_key = _safe_batch_identifier(
            usd_file,
            position,
            used_result_keys,
        )
        used_result_keys.add(safe_result_key)
        safe_usd_file = redact_sensitive_path(usd_file)
        safe_output_dir = redact_sensitive_path(dataset_output_dir)

        logger.info("  Processing %s -> %s", safe_usd_file, safe_output_dir)

        try:
            # Prepare context for this specific file
            file_context = dict(base_context)  # Copy base context
            file_context["source_override"] = usd_file
            file_context["output_dir_override"] = dataset_output_dir

            # Run workflow for this file
            result = await workflow_runner(file_context)

            # Check result
            if not result or "error" in result:
                logger.warning("  ✗ Failed to process %s", safe_usd_file)
                results[safe_result_key] = {
                    "status": "failed",
                    "usd_file": safe_usd_file,
                    "output_dir": safe_output_dir,
                    "error": _BATCH_ITEM_FAILURE_MESSAGE,
                }
                failed += 1
            else:
                logger.info("  ✓ Successfully processed %s", safe_usd_file)
                dataset_path = result.get("dataset_path")
                results[safe_result_key] = {
                    "status": "success",
                    "usd_file": safe_usd_file,
                    "output_dir": safe_output_dir,
                    "dataset_path": (
                        redact_sensitive_path(dataset_path) if dataset_path else "N/A"
                    ),
                    "num_prims": result.get("num_prims", 0),
                    "num_images": result.get("num_images", 0),
                }
                successful += 1

        except Exception:
            logger.error("  ✗ Failed to process %s", safe_usd_file)
            results[safe_result_key] = {
                "status": "failed",
                "usd_file": safe_usd_file,
                "output_dir": safe_output_dir,
                "error": _BATCH_ITEM_FAILURE_MESSAGE,
            }
            failed += 1

    logger.info(f"Batch processing complete: {successful} successful, {failed} failed")

    if failed > 0 and successful == 0:
        raise RuntimeError(f"All {failed} USD files failed to process")

    # Return aggregated results
    return {
        "output_dir": redact_sensitive_path(batch_output_dir),
        "num_files_processed": successful,
        "num_files_failed": failed,
        "total_files": len(usd_files),
        "results": results,
    }
