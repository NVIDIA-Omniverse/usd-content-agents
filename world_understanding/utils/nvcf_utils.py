# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common utilities for NVIDIA Cloud Functions (NVCF) API calls."""

import logging
import os
import random
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import requests
from opentelemetry.trace import Status, StatusCode
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

from world_understanding.telemetry import get_tracer
from world_understanding.telemetry.attributes import MAAttributes

logger = logging.getLogger(__name__)

NVCF_INVOCATION_HOST = ".invocation.api.nvcf.nvidia.com"
_HTTP_BASE_URL_SCHEMES = {"http", "https"}
_LOCAL_SERVICE_URL_SCHEME = "http"
_NVCF_URL_SCHEME = "https"
_URL_SCHEME_SEPARATOR = "://"

# Get tracer at module level
_tracer = get_tracer(__name__)


def get_nvcf_api_key(api_key: str | None = None) -> str:
    """Get NVCF API key from parameter or environment.

    Returns an empty string (with a warning) when no key is available,
    instead of raising. That way callers pointed at a local, non-NVCF
    `RENDER_ENDPOINT` (e.g. the OVRTX rendering API container) don't
    need to set NGC_API_KEY just to satisfy this helper — their HTTP
    server simply ignores the Authorization header. Real NVCF endpoints
    will still fail with a clear HTTP 401 if the key is actually
    required.

    Args:
        api_key: Optional API key. If None, reads from NGC_API_KEY env var

    Returns:
        The API key string, or "" if none is available.
    """
    if api_key is None:
        api_key = os.environ.get("NGC_API_KEY", "")
        if not api_key:
            logger.warning(
                "NGC_API_KEY is not set; requests to NVCF endpoints will "
                "fail with HTTP 401. Set NGC_API_KEY to authenticate, or "
                "ignore this warning if you are targeting a local render "
                "service via RENDER_ENDPOINT."
            )
    return api_key


def get_base_url(
    base_url: str | None,
    endpoint_env: str,
    function_id_env: str,
) -> str:
    """Resolve a service base URL with endpoint-then-function-id fallback.

    Resolution order:
        1. ``base_url`` argument (if not None)
        2. ``endpoint_env`` environment variable (URL like
           ``http://ovrtx-rendering-api:8000``, ``ovrtx-rendering-api:8000``, or
           ``https://abc12345.invocation.api.nvcf.nvidia.com``)
        3. ``function_id_env`` environment variable (NVCF function ID,
           expanded to ``https://{id}.invocation.api.nvcf.nvidia.com``)

    The endpoint form is preferred for clarity — point it at any compatible
    service (NVCF, local OVRTX, or another mock). The function-id form is
    kept for backward compatibility and convenience when targeting a real
    NVCF function by ID.

    Args:
        base_url: Optional explicit base URL or function ID.
        endpoint_env: Env var that holds a full URL (preferred).
        function_id_env: Env var that holds an NVCF function ID (fallback).

    Returns:
        The resolved base URL string.

    Raises:
        ValueError: If no value is provided or found in either env var.
    """
    if base_url is None:
        base_url = os.environ.get(endpoint_env) or os.environ.get(function_id_env)
        if not base_url:
            raise ValueError(
                f"Base URL required. Set {endpoint_env} (full URL) or "
                f"{function_id_env} (NVCF function ID), or pass base_url."
            )

    return resolve_endpoint_or_function_id(base_url)


def _has_http_base_url_scheme(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return _starts_with_http_base_url_prefix(value)
    return parsed.scheme.lower() in _HTTP_BASE_URL_SCHEMES


def _starts_with_http_base_url_prefix(value: str) -> bool:
    normalized = value.lower()
    return any(
        normalized.startswith(_url_with_scheme(scheme, ""))
        for scheme in _HTTP_BASE_URL_SCHEMES
    )


def _has_schemeless_service_location(value: str) -> bool:
    try:
        parsed = urlsplit(value if value.startswith("//") else f"//{value}")
    except ValueError:
        return False
    if not parsed.hostname:
        return False
    try:
        return parsed.port is not None
    except ValueError:
        return False


def is_service_base_url(value: str) -> bool:
    """Return True when ``value`` points to an HTTP service, not a function ID."""
    value = str(value).strip()
    if not value:
        return False
    return _has_http_base_url_scheme(value) or _has_schemeless_service_location(value)


def resolve_endpoint_or_function_id(value: str) -> str:
    """Resolve an endpoint URL or NVCF function ID into a service base URL."""
    value = str(value).strip()
    if not value:
        raise ValueError("Base URL value cannot be empty")
    if _has_http_base_url_scheme(value):
        return value
    if _has_schemeless_service_location(value):
        return _url_with_scheme(
            _LOCAL_SERVICE_URL_SCHEME,
            value.removeprefix("//"),
        )
    return _url_with_scheme(_NVCF_URL_SCHEME, f"{value}{NVCF_INVOCATION_HOST}")


def _url_with_scheme(scheme: str, location: str) -> str:
    return f"{scheme}{_URL_SCHEME_SEPARATOR}{location}"


def create_nvcf_headers(
    api_key: str, _timeout: int, poll_seconds: int | None = None
) -> dict[str, str]:
    """Create standard NVCF request headers.

    Args:
        api_key: NVCF API key
        _timeout: Request timeout in seconds (kept for API compatibility but not used;
                  timeout is handled by the HTTP client layer)
        poll_seconds: Optional polling timeout in seconds. If provided, enables long-polling.
                      If None, uses reconnection-based polling (no NVCF-POLL-SECONDS header).

    Returns:
        Dictionary of HTTP headers for NVCF requests
    """
    # Build base headers first (like client_common.py lines 189-193).
    # Skip the Authorization header when no api_key is set: httpx/requests
    # reject `Authorization: Bearer ` (trailing space) as an illegal header
    # value, and local RENDER_ENDPOINT targets (e.g. the bundled OVRTX
    # container) don't need the header at all.
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Conditionally add polling headers (like client_common.py lines 195-197)
    if poll_seconds is not None:
        headers["NVCF-POLL-SECONDS"] = str(poll_seconds)
        headers["nvcf-feature-enable-gateway-timeout"] = "true"

    return headers


def parse_zip_response(zip_content: bytes) -> dict[str, Any] | None:
    """Parse a ZIP response from NVCF optimizer API.

    Matches client_common.py lines 260-305.

    Args:
        zip_content: Raw ZIP file content as bytes

    Returns:
        Parsed result dictionary, or None if parsing fails
    """
    import io
    import json
    import zipfile

    try:
        # Open ZIP from memory
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
            # List all files in the ZIP
            file_list = zip_file.namelist()
            logger.debug("ZIP contains files: %s", file_list)

            # Find the .response file (contains JSON result)
            response_file = None
            for filename in file_list:
                if filename.endswith(".response"):
                    response_file = filename
                    break

            if response_file is None:
                logger.error("No .response file found in ZIP. Files: %s", file_list)
                return None

            # Extract and parse the JSON response
            logger.info("Extracting response file: %s", response_file)
            with zip_file.open(response_file) as f:
                response_bytes = f.read()
                result = json.loads(response_bytes)

            logger.info(
                "Successfully parsed ZIP response: success=%s, keys=%s",
                result.get("success", "unknown"),
                list(result.keys()),
            )
            return result

    except zipfile.BadZipFile:
        logger.error("Invalid ZIP file received")
        return None
    except json.JSONDecodeError as e:
        logger.error("Error parsing JSON from ZIP: %s", str(e))
        return None
    except Exception as e:
        logger.error("Error parsing ZIP response: %s", str(e))
        return None


async def poll_nvcf_status(
    client: Any,  # httpx.AsyncClient
    req_id: str,
    api_key: str,
    poll_seconds: int,
    timeout: float,
) -> tuple[int, dict[str, Any] | None]:
    """Poll NVCF status endpoint until completion (async version).

    Matches client_common.py poll_nvcf_status (lines 313-410).

    NVCF Long-Polling Behavior:
    - When client receives 202, it must IMMEDIATELY poll with same NVCF-POLL-SECONDS
    - Server holds connection open (no client-side sleep needed)
    - Poll repeatedly until 200 (success) or 504 (timeout/error)

    Args:
        client: httpx.AsyncClient instance
        req_id: NVCF request ID from 202 response header (nvcf-reqid)
        api_key: API key for authentication
        poll_seconds: Same NVCF-POLL-SECONDS value used in initial request
        timeout: Total timeout in seconds (client-side safety limit)

    Returns:
        Tuple of (status_code, result_dict) where result_dict is None on failure
    """
    # NVCF status endpoint format
    status_url = f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{req_id}"

    # Must use same NVCF-POLL-SECONDS as initial request. Skip the
    # Authorization header when no api_key is set (httpx rejects
    # `Bearer ` with trailing space as an illegal header value).
    headers = {
        "Accept": "application/json",
        "NVCF-POLL-SECONDS": str(poll_seconds),
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    logger.info(
        "Starting status polling for request %s (poll_seconds=%d)", req_id, poll_seconds
    )

    start_time = time.time()
    poll_count = 0

    while (time.time() - start_time) < timeout:
        poll_count += 1
        try:
            # Poll immediately - server holds connection open via NVCF-POLL-SECONDS
            response = await client.get(status_url, headers=headers)
            status_code = response.status_code

            logger.debug(
                "Poll %d for %s: status=%d",
                poll_count,
                req_id,
                status_code,
            )

            if status_code == 200:
                # Worker finished - return result
                content_type = response.headers.get("content-type", "")

                if "application/json" in content_type:
                    result = response.json()
                elif "application/zip" in content_type:
                    logger.info("Received ZIP response during polling, parsing...")
                    result = parse_zip_response(response.content)
                    if result is None:
                        logger.error(
                            "Failed to parse ZIP response for request %s", req_id
                        )
                        return status_code, None
                else:
                    logger.error(
                        "Unexpected content type '%s' for request %s",
                        content_type,
                        req_id,
                    )
                    return status_code, None

                logger.info(
                    "Request %s fulfilled after %d polls (%.1fs)",
                    req_id,
                    poll_count,
                    time.time() - start_time,
                )
                return status_code, result

            elif status_code == 202:
                # Still processing - poll again immediately (no sleep)
                logger.debug(
                    "Request %s still processing (poll %d)", req_id, poll_count
                )
                continue

            elif status_code == 504:
                # Gateway timeout - worker may have failed
                logger.warning(
                    "Request %s timed out at gateway (poll %d, %.1fs)",
                    req_id,
                    poll_count,
                    time.time() - start_time,
                )
                return status_code, None

            else:
                # Unexpected status code
                logger.error(
                    "Unexpected status %d for request %s (poll %d)",
                    status_code,
                    req_id,
                    poll_count,
                )
                return status_code, None

        except Exception:
            logger.exception("Error during polling for request %s", req_id)
            return 500, None

    # Timeout reached
    logger.error(
        "Polling timeout after %.1fs for request %s", time.time() - start_time, req_id
    )
    return 504, None


async def execute_nvcf_request_async(
    url: str,
    headers: dict[str, str],
    params: dict[str, Any],
    api_key: str,
    timeout: int,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_jitter: float = 0.1,
) -> dict[str, Any]:
    """Execute NVCF POST request asynchronously with retry logic and 202 polling.

    This is the async equivalent of execute_nvcf_request_with_retry.

    Args:
        url: URL to send the request to
        headers: HTTP headers
        params: JSON parameters
        api_key: NVCF API key for authentication (used for polling)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts. Default: 3
        retry_delay: Initial delay between retries in seconds. Default: 1.0
        retry_backoff_factor: Factor to multiply delay by after each retry. Default: 2.0
        retry_jitter: Random jitter factor (0-1) to add to delays. Default: 0.1

    Returns:
        Parsed result dictionary (from JSON or ZIP response)

    Raises:
        RuntimeError: If request fails after all retries
    """
    import asyncio
    import random

    import httpx

    with _tracer.start_as_current_span("nvcf.request.async") as span:
        span.set_attribute("http.url", url)
        span.set_attribute("http.method", "POST")

        try:
            if ".invocation.api.nvcf.nvidia.com" in url:
                function_id = url.split("//")[1].split(".")[0]
                span.set_attribute(MAAttributes.NVCF_FUNCTION_ID, function_id)
        except (IndexError, AttributeError):
            pass

        # Extract poll_seconds from headers for status polling
        poll_seconds_str = headers.get("NVCF-POLL-SECONDS")
        poll_seconds = int(poll_seconds_str) if poll_seconds_str else 300

        # Retryable HTTP status codes (5xx server errors + 408 timeout + 429 rate limit)
        retryable_codes = [408, 429, 500, 502, 503, 504]

        last_error = None
        current_delay = retry_delay

        for attempt in range(max_retries + 1):
            span.set_attribute(MAAttributes.NVCF_RETRY_COUNT, attempt)
            try:
                if attempt > 0:
                    # Add jitter to prevent thundering herd
                    jittered_delay = current_delay * (
                        1 + random.uniform(-retry_jitter, retry_jitter)
                    )
                    logger.info(
                        "Retrying NVCF request (attempt %d/%d) after %.2fs delay",
                        attempt + 1,
                        max_retries + 1,
                        jittered_delay,
                    )
                    await asyncio.sleep(jittered_delay)
                    current_delay *= retry_backoff_factor

                async with httpx.AsyncClient(
                    timeout=timeout + 30, follow_redirects=True
                ) as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=params,
                    )

                    # Handle NVCF 202 Accepted - need to poll status endpoint
                    if response.status_code == 202:
                        req_id = response.headers.get("nvcf-reqid")
                        if not req_id:
                            error_msg = "No nvcf-reqid header in 202 response"
                            logger.error(error_msg)
                            if attempt == max_retries:
                                raise RuntimeError(error_msg) from None
                            last_error = RuntimeError(error_msg)
                            continue

                        logger.info(
                            "Received 202 Accepted (nvcf-reqid: %s), starting status polling...",
                            req_id,
                        )

                        # Poll for completion
                        status_code, poll_result = await poll_nvcf_status(
                            client=client,
                            req_id=req_id,
                            api_key=api_key,
                            poll_seconds=poll_seconds,
                            timeout=timeout,
                        )

                        if poll_result is None:
                            error_msg = f"Polling failed for request {req_id} (status: {status_code})"
                            logger.error(error_msg)
                            if attempt == max_retries:
                                raise RuntimeError(error_msg) from None
                            last_error = RuntimeError(error_msg)
                            continue

                        return poll_result

                    else:
                        # Handle direct response (200 OK)
                        response.raise_for_status()
                        span.set_attribute("http.status_code", response.status_code)

                        # Check content type for ZIP vs JSON response
                        content_type = response.headers.get("content-type", "")

                        if "application/json" in content_type:
                            return response.json()
                        elif "application/zip" in content_type:
                            logger.info("Received ZIP response, parsing...")
                            result = parse_zip_response(response.content)
                            if result is None:
                                error_msg = "Failed to parse ZIP response"
                                logger.error(error_msg)
                                if attempt == max_retries:
                                    raise RuntimeError(error_msg)
                                last_error = RuntimeError(error_msg)
                                continue
                            return result
                        else:
                            # Try JSON as fallback
                            try:
                                return response.json()
                            except Exception as e:
                                error_msg = f"Unexpected content type '{content_type}': {str(e)}"
                                logger.error(error_msg)
                                if attempt == max_retries:
                                    raise RuntimeError(error_msg) from e
                                last_error = e
                                continue

            except httpx.HTTPStatusError as e:
                # HTTP status errors - check if we should retry
                span.set_attribute("http.status_code", e.response.status_code)
                span.record_exception(e)
                if e.response.status_code in retryable_codes:
                    last_error = e
                    logger.warning(
                        "NVCF request attempt %d failed with HTTP %d: %s",
                        attempt + 1,
                        e.response.status_code,
                        str(e),
                    )
                    if attempt == max_retries:
                        error_msg = f"NVCF request failed after {max_retries + 1} attempts: {str(last_error)}"
                        logger.error(error_msg)
                        span.set_status(Status(StatusCode.ERROR, error_msg))
                        raise RuntimeError(error_msg) from e
                else:
                    # Non-retryable status code - fail immediately
                    error_msg = f"NVCF request failed with HTTP {e.response.status_code}: {str(e)}"
                    logger.error(error_msg)
                    span.set_status(Status(StatusCode.ERROR, error_msg))
                    raise RuntimeError(error_msg) from e

            except httpx.RequestError as e:
                # Network errors - always retry
                last_error = e
                span.record_exception(e)
                logger.warning(
                    "NVCF request attempt %d failed with network error: %s",
                    attempt + 1,
                    str(e),
                )
                if attempt == max_retries:
                    error_msg = f"NVCF request failed after {max_retries + 1} attempts: {str(last_error)}"
                    logger.error(error_msg)
                    span.set_status(Status(StatusCode.ERROR, error_msg))
                    raise RuntimeError(error_msg) from e

        # Should never reach here due to raise in loop, but just in case
        raise RuntimeError(f"NVCF request failed: {str(last_error)}")


def execute_nvcf_request_with_retry(
    url: str,
    headers: dict[str, str],
    params: dict[str, Any],
    timeout: int,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_jitter: float = 0.1,
    error_response_factory: Callable[[str, float], dict[str, Any]] | None = None,
) -> dict[str, Any] | requests.Response:
    """Execute NVCF POST request with retry logic.

    Args:
        url: URL to send the request to
        headers: HTTP headers
        params: JSON parameters
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts. Default: 3
        retry_delay: Initial delay between retries in seconds. Default: 1.0
        retry_backoff_factor: Factor to multiply delay by after each retry. Default: 2.0
        retry_jitter: Random jitter factor (0-1) to add to delays. Default: 0.1
        error_response_factory: Optional function to create error response dict.
            If None, raises exception on error. Signature: (error_msg, elapsed_time) -> dict

    Returns:
        Response object on success, or error dict if error_response_factory provided

    Raises:
        RuntimeError: If request fails and error_response_factory is None
    """
    with _tracer.start_as_current_span("nvcf.request") as span:
        # Set initial span attributes
        span.set_attribute("http.url", url)
        span.set_attribute("http.method", "POST")

        # Try to extract function_id from URL for tracing
        # URL format: https://{function_id}.invocation.api.nvcf.nvidia.com
        try:
            if ".invocation.api.nvcf.nvidia.com" in url:
                function_id = url.split("//")[1].split(".")[0]
                span.set_attribute(MAAttributes.NVCF_FUNCTION_ID, function_id)
        except (IndexError, AttributeError):
            pass  # Unable to extract function_id, skip

        start_time = time.time()
        last_error = None
        current_delay = retry_delay

        # Retryable HTTP status codes (5xx server errors + 408 timeout + 429 rate limit)
        retryable_codes = [408, 429, 500, 502, 503, 504]

        for attempt in range(max_retries + 1):
            span.set_attribute(MAAttributes.NVCF_RETRY_COUNT, attempt)
            try:
                if attempt > 0:
                    # Add jitter to prevent thundering herd
                    jittered_delay = current_delay * (
                        1 + random.uniform(-retry_jitter, retry_jitter)
                    )
                    logger.info(
                        "Retrying NVCF request (attempt %d/%d) after %.2fs delay",
                        attempt + 1,
                        max_retries + 1,
                        jittered_delay,
                    )
                    time.sleep(jittered_delay)
                    current_delay *= retry_backoff_factor

                response = requests.post(
                    url,
                    headers=headers,
                    json=params,
                    timeout=timeout + 10,
                    allow_redirects=True,
                )
                response.raise_for_status()

                # Success - set status code and return response
                span.set_attribute("http.status_code", response.status_code)
                return response

            except (ConnectionError, Timeout) as e:
                # Network errors - retry
                last_error = e
                span.record_exception(e)
                logger.warning(
                    "NVCF request attempt %d failed with network error: %s",
                    attempt + 1,
                    str(e),
                )
                if attempt == max_retries:
                    error_msg = f"NVCF request failed after {max_retries + 1} attempts: {str(last_error)}"
                    logger.error(error_msg)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    if error_response_factory:
                        return error_response_factory(
                            error_msg, time.time() - start_time
                        )
                    else:
                        raise RuntimeError(error_msg) from e

            except HTTPError as e:
                # HTTP errors - check if we should retry
                span.set_attribute("http.status_code", e.response.status_code)
                span.record_exception(e)
                if e.response.status_code in retryable_codes:
                    last_error = e
                    logger.warning(
                        "NVCF request attempt %d failed with HTTP %d: %s",
                        attempt + 1,
                        e.response.status_code,
                        str(e),
                    )
                    if attempt == max_retries:
                        error_msg = f"NVCF request failed after {max_retries + 1} attempts: HTTP {e.response.status_code}"
                        logger.error(error_msg)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        if error_response_factory:
                            return error_response_factory(
                                error_msg, time.time() - start_time
                            )
                        else:
                            raise RuntimeError(error_msg) from e
                else:
                    # Non-retryable HTTP error (400, 401, 403, 404, etc.)
                    error_msg = f"NVCF request failed with non-retryable HTTP {e.response.status_code}: {str(e)}"
                    logger.error(
                        "Non-retryable error: HTTP %d. Will not retry. Error: %s",
                        e.response.status_code,
                        str(e),
                    )
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    if error_response_factory:
                        return error_response_factory(
                            error_msg, time.time() - start_time
                        )
                    else:
                        raise RuntimeError(error_msg) from e

            except RequestException as e:
                # Other request exceptions - don't retry
                error_msg = f"NVCF request failed with non-retryable error: {str(e)}"
                logger.error(
                    "Non-retryable request exception. Will not retry. Error: %s",
                    str(e),
                )
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                if error_response_factory:
                    return error_response_factory(error_msg, time.time() - start_time)
                else:
                    raise RuntimeError(error_msg) from e

        # Should never reach here, but just in case
        error_msg = "NVCF request failed: unexpected error"
        span.set_status(Status(StatusCode.ERROR, error_msg))
        if error_response_factory:
            return error_response_factory(error_msg, time.time() - start_time)
        else:
            raise RuntimeError(error_msg)


def s3_uri_to_https_url(s3_uri: str, s3_region: str) -> str:
    """Convert S3 URI to HTTPS URL with region.

    Args:
        s3_uri: S3 URI in format s3://bucket/key
        s3_region: AWS region where the bucket is located

    Returns:
        HTTPS URL in format https://bucket.s3.region.amazonaws.com/key
    """
    # s3_uri format: s3://bucket/key -> https://bucket.s3.region.amazonaws.com/key
    bucket_and_key = s3_uri.replace("s3://", "")
    bucket, key = bucket_and_key.split("/", 1)
    return f"https://{bucket}.s3.{s3_region}.amazonaws.com/{key}"
