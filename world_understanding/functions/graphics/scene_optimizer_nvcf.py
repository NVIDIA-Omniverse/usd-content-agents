# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD scene optimization functions using NVIDIA Cloud Functions (NVCF) microservice."""

import base64
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from world_understanding.config.s3 import WU_S3_BUCKET, WU_S3_PROFILE, WU_S3_REGION
from world_understanding.functions.graphics.so_export import (
    _atomic_output_file,
    _lexical_absolute_path,
)
from world_understanding.utils.data_uri import should_use_data_uri
from world_understanding.utils.nvcf_utils import (
    create_nvcf_headers,
    execute_nvcf_request_async,
    get_base_url,
    get_nvcf_api_key,
    s3_uri_to_https_url,
)
from world_understanding.utils.s3_utils import delete_s3_path, upload_file_to_s3

logger = logging.getLogger(__name__)


async def optimize_usd_from_url(
    input_url: str,
    output_path: Path | str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 600,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    optimization_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Optimize a USD file from URL using NVCF (async).

    This async function calls the NVCF optimization microservice with a USD file URL
    and writes the optimized result to the output path.

    Args:
        input_url: URL to the input USD file (HTTP/HTTPS)
        output_path: Path where optimized USD will be saved
        api_key: NVCF API key. If None, uses NGC_API_KEY env var
        base_url: NVCF base URL (without endpoint). If None, uses OPTIMIZER_ENDPOINT env var (or NVCF_OPTIMIZER_FUNCTION_ID fallback)
        timeout: Request timeout in seconds. Default: 600
        max_retries: Maximum number of retry attempts. Default: 3
        retry_delay: Initial delay between retries in seconds. Default: 1.0
        retry_backoff_factor: Factor to multiply delay by after each retry. Default: 2.0
        optimization_config: Optional dict with additional optimization parameters:
            - scene_optimizer_settings: Dict with operation settings
            - poll_seconds: int (optional) - Polling timeout for long-running operations
            - wait_for_assets: bool (default False) - Wait for assets to load
            - stage_timeout: float (default 180.0) - Stage opening timeout
            - output_format: str (default "usdc") - Output format: "usd", "usda", or "usdc"

    Returns:
        Dict containing:
            - status: Optimization status (success or error)
            - optimization_time: Total optimization time in seconds
            - stage_size_bytes: Size of optimized stage in bytes
            - operations_executed: List of optimization operations performed
            - report: Text report of optimization operations
            - correspondence_map: Mapping data for split/deduplicated meshes
            - error: Error message if optimization failed (optional)

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If optimization fails

    Example:
        >>> result = await optimize_usd_from_url(
        ...     input_url="https://example.com/scene.usd",
        ...     output_path="optimized.usdc",  # Format determined from .usdc extension
        ...     api_key="your-api-key"
        ... )
        >>> print("Optimized to optimized.usdc")
    """
    # Get API key and base URL using common utilities
    api_key = get_nvcf_api_key(api_key)
    base_url = get_base_url(
        base_url, "OPTIMIZER_ENDPOINT", "NVCF_OPTIMIZER_FUNCTION_ID"
    )

    # Construct full URL with optimize endpoint
    full_url = f"{base_url.rstrip('/')}/optimize"

    # Build request parameters with new API format
    # Determine output format from output_path suffix
    output_path_obj = Path(output_path)
    output_suffix = output_path_obj.suffix.lower().lstrip(".")

    # Default to usdc if no extension or invalid extension
    if output_suffix not in ["usd", "usda", "usdc"]:
        output_format = "usdc"
        logger.info("Output path has no valid USD extension, defaulting to usdc format")
    else:
        output_format = output_suffix

    # Extract scene optimizer settings from config
    scene_optimizer_settings = {}

    if optimization_config and "scene_optimizer_settings" in optimization_config:
        # New format: use provided scene_optimizer_settings directly
        scene_optimizer_settings = optimization_config[
            "scene_optimizer_settings"
        ].copy()
        logger.info("Using provided scene_optimizer_settings from config")
    elif optimization_config:
        # Legacy format: extract individual settings for backward compatibility
        logger.info("Using legacy flat config format (backward compatibility)")
        scene_optimizer_settings = {
            "wait_for_assets": optimization_config.get("wait_for_assets", False),
            "stage_timeout": optimization_config.get("stage_timeout", 180.0),
            "extract_geom_subset_indices": optimization_config.get(
                "extract_geom_subset_indices", True
            ),
        }
    else:
        # No config provided: use minimal defaults
        logger.info("No optimization config provided, using minimal defaults")
        scene_optimizer_settings = {
            "wait_for_assets": False,
            "stage_timeout": 180.0,
            "extract_geom_subset_indices": True,
        }

    # Always override output_format with the format derived from output_path
    scene_optimizer_settings["output_format"] = output_format

    params = {
        "url": input_url,
        "scene_optimizer_settings": scene_optimizer_settings,
        "timeout": timeout,
    }

    # Extract poll_seconds from optimization_config (default 300 = NVCF max long-poll)
    poll_seconds = (
        optimization_config.get("poll_seconds", 300) if optimization_config else 300
    )

    # NVCF runs behind AWS VPC (350s idle timeout); cap at 300s for safety margin
    aws_vpc_mode = (
        optimization_config.get("aws_vpc_mode", True) if optimization_config else True
    )
    if aws_vpc_mode:
        max_poll = 300
        if poll_seconds > max_poll:
            logger.warning(
                "AWS VPC mode: capping poll_seconds from %d to %d",
                poll_seconds,
                max_poll,
            )
            poll_seconds = max_poll

    # Create headers using common utility
    headers = create_nvcf_headers(api_key, timeout, poll_seconds=poll_seconds)

    logger.info("Optimizing USD from %s", input_url[:100])
    start_time = time.time()

    # Execute NVCF request with retry and polling
    try:
        result = await execute_nvcf_request_async(
            url=full_url,
            headers=headers,
            params=params,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff_factor=retry_backoff_factor,
        )
    except RuntimeError as e:
        return {
            "status": "error",
            "optimization_time": time.time() - start_time,
            "error": str(e),
        }

    optimization_time = time.time() - start_time
    logger.info("NVCF request completed in %.2fs", optimization_time)

    # Check if optimization succeeded
    success = result.get("success")
    if not success:
        error_msg = "Optimization failed (success=False in response)"
        logger.error(error_msg)
        return {
            "status": "error",
            "optimization_time": optimization_time,
            "error": error_msg,
        }

    # Decode and write optimized USD
    try:
        optimized_stage_base64 = result.get("optimized_stage_base64")
        if not optimized_stage_base64:
            error_msg = "No optimized_stage_base64 in response"
            logger.error(error_msg)
            return {
                "status": "error",
                "optimization_time": optimization_time,
                "error": error_msg,
            }

        stage_bytes = base64.b64decode(optimized_stage_base64)

        # Write to output path
        output_path = _lexical_absolute_path(output_path)
        with _atomic_output_file(
            output_path,
            clear_portable_sidecar=True,
        ) as transaction_output:
            transaction_output.write_bytes(stage_bytes)

        output_size = output_path.stat().st_size
        logger.info("Wrote optimized USD to %s (%d bytes)", output_path, output_size)

        return {
            "status": "success",
            "optimization_time": optimization_time,
            "stage_size_bytes": output_size,
            "operations_executed": result.get("operations_executed", []),
            "report": result.get("report", ""),
            "correspondence_map": result.get("correspondence_map", {}),
        }

    except Exception as e:
        error_msg = f"Failed to decode/write optimized USD: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "optimization_time": optimization_time,
            "error": error_msg,
        }


async def optimize_usd_from_path(
    input_path: Path | str,
    output_path: Path | str,
    api_key: str | None = None,
    base_url: str | None = None,
    s3_bucket: str = WU_S3_BUCKET,
    s3_region: str = WU_S3_REGION,
    s3_profile: str = WU_S3_PROFILE,
    timeout: int = 3600,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    optimization_config: dict[str, Any] | None = None,
    use_data_uri: bool | None = None,
) -> dict[str, Any]:
    """
    Optimize a USD file from local path using NVCF (async).

    This async function uploads the input USD to S3, calls the NVCF optimization
    microservice, and writes the optimized result to the output path.
    The uploaded S3 file is cleaned up automatically.

    Args:
        input_path: Path to the input USD file
        output_path: Path where optimized USD will be saved
        api_key: NVCF API key. If None, uses NGC_API_KEY env var
        base_url: NVCF base URL (without endpoint). If None, uses OPTIMIZER_ENDPOINT env var (or NVCF_OPTIMIZER_FUNCTION_ID fallback)
        s3_bucket: S3 bucket for file upload.
                  Default: WU_S3_BUCKET env var (required for S3 mode).
        s3_region: AWS region where the S3 bucket is located.
                  Default: WU_S3_REGION env var or "us-east-2"
        s3_profile: AWS profile for S3 upload.
                   Default: WU_S3_PROFILE env var (required for S3 mode).
        timeout: Request timeout in seconds. Default: 3600
        max_retries: Maximum number of retry attempts. Default: 3
        retry_delay: Initial delay between retries in seconds. Default: 1.0
        retry_backoff_factor: Factor to multiply delay by after each retry. Default: 2.0
        optimization_config: Optional dict with additional optimization parameters.
            Can be provided in two formats:

            1. Legacy flat format (for backward compatibility):
               - wait_for_assets: bool (default False) - Wait for assets to load
               - stage_timeout: float (default 180.0) - Stage opening timeout

            2. New nested format (recommended):
               - scene_optimizer_settings: dict with full SceneOptimizerSettings structure
                 containing operations, generate_report, capture_stats, verbose,
                 wait_for_assets, stage_timeout, output_format, extract_geom_subset_indices
               - poll_seconds: int (optional) - Polling timeout for long-running operations

            Note: output_format is auto-detected from output_path suffix if not specified

    Returns:
        Dict containing:
            - status: Optimization status (success or error)
            - optimization_time: Total optimization time in seconds
            - stage_size_bytes: Size of optimized stage in bytes
            - operations_executed: List of optimization operations performed
            - report: Text report of optimization operations
            - correspondence_map: Mapping data for split/deduplicated meshes
            - error: Error message if optimization failed (optional)

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If optimization fails

    Example:
        >>> result = await optimize_usd_from_path(
        ...     input_path="input.usd",
        ...     output_path="optimized.usdc",  # Format determined from .usdc extension
        ...     s3_bucket="my-bucket",
        ...     s3_region="us-west-2"
        ... )
        >>> print("Optimized to optimized.usdc")
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")

    # Determine whether to use data URI: explicit param > env var > auto-detect
    use_data_uri = should_use_data_uri(use_data_uri)

    if use_data_uri:
        # Base64-encode input and pass as data URI (no S3 needed)
        from world_understanding.utils.usd.stage import create_data_uri_from_file

        logger.info("Using data URI for optimization input (no S3)")
        mime_type = (
            "model/vnd.usdz+zip"
            if input_path.suffix.lower() == ".usdz"
            else "model/vnd.usd"
        )
        input_url = create_data_uri_from_file(input_path, mime_type=mime_type)

        result = await optimize_usd_from_url(
            input_url=input_url,
            output_path=output_path,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff_factor=retry_backoff_factor,
            optimization_config=optimization_config,
        )

        if result.get("status") == "success":
            logger.info("Optimization complete to %s", output_path)

        return result

    # S3 path: upload to S3, pass HTTPS URL, clean up after
    suffix = input_path.suffix.lower() if input_path.suffix else ".usd"
    if suffix not in (".usd", ".usda", ".usdc", ".usdz"):
        suffix = ".usd"
    unique_id = uuid.uuid4().hex
    s3_key = f"nvcf-optimization/{unique_id}/input{suffix}"
    s3_uri = None

    try:
        logger.info("Uploading input USD to S3...")
        s3_uri = upload_file_to_s3(
            file_path=str(input_path),
            s3_path=f"s3://{s3_bucket}/{s3_key}",
            profile_name=s3_profile,
        )
        input_url = s3_uri_to_https_url(s3_uri, s3_region)
        logger.info("Uploaded to S3: %s", input_url)

        # Call optimization API
        result = await optimize_usd_from_url(
            input_url=input_url,
            output_path=output_path,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff_factor=retry_backoff_factor,
            optimization_config=optimization_config,
        )

        if result.get("status") == "success":
            logger.info("Optimization complete to %s", output_path)

        return result

    finally:
        # Clean up S3 file
        if s3_uri:
            try:
                delete_s3_path(s3_uri, profile_name=s3_profile)
                logger.info("Cleaned up S3 file: %s", s3_uri)
            except Exception as e:
                logger.warning("Failed to clean up S3 file %s: %s", s3_uri, e)
