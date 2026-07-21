# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""REST client for the Texture Variation API service.

Calls the remote service at /v1/texture-variations endpoints.
Drop-in replacement for the local TextureVariationClient.

Usage:
    from texture_agent.functions.rest_client import RestTextureVariationClient

    client = RestTextureVariationClient("http://dt1:8000")
    status = client.generate(
        source_asset_uri="file:///path/to/asset.usd",
        conditioning=Conditioning(text_prompt="rusted metal"),
        config=TextureVariationConfig(strength=0.8),
    )
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict
from typing import Any

import httpx

from texture_agent.functions.texture_generation import (
    BackendCapabilities,
    Conditioning,
    GeneratedTextures,
    GenerationResult,
    JobStatus,
    MapArtifact,
    TextureTarget,
    TextureVariationConfig,
)

logger = logging.getLogger(__name__)
_STATUS_POLL_MAX_ATTEMPTS = 4
_STATUS_POLL_RETRY_BASE_DELAY_SEC = 1.0
_SUBMIT_RETRYABLE_STATUS_CODES = {429}


class RestTextureVariationClient:
    """REST client implementing the Texture Variation API contract.

    Talks to a remote service running the texture-editing pipeline
    (Step1X-3D + Material Anything) via the REST API.
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: str | None = None,
        timeout: float = 600.0,
        submit_retry_timeout_sec: float = 600.0,
        submit_retry_base_delay_sec: float = 5.0,
        submit_retry_max_delay_sec: float = 30.0,
    ) -> None:
        self._endpoint_url = endpoint_url.rstrip("/")
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout
        self._submit_retry_timeout_sec = max(0.0, submit_retry_timeout_sec)
        self._submit_retry_base_delay_sec = max(0.1, submit_retry_base_delay_sec)
        self._submit_retry_max_delay_sec = max(
            self._submit_retry_base_delay_sec,
            submit_retry_max_delay_sec,
        )

    def generate(
        self,
        source_asset_uri: str,
        conditioning: Conditioning,
        config: TextureVariationConfig | None = None,
        wait: bool = True,
        timeout_sec: int = 600,
        target: TextureTarget | None = None,
        capabilities: BackendCapabilities | None = None,
    ) -> JobStatus:
        """Submit a texture variation job.

        Args:
            source_asset_uri: URI to the source USD asset.
            conditioning: Text prompt, reference images, etc.
            config: Generation configuration.
            wait: If True, poll until terminal status or timeout.
            timeout_sec: Max wait time in seconds.
            target: Optional selected material/prim scope.
            capabilities: Optional requested backend capabilities.

        Returns:
            JobStatus with result on completion.
        """
        config = config or TextureVariationConfig()
        conditioning.validate()

        body = self._build_request_body(
            source_asset_uri=source_asset_uri,
            conditioning=conditioning,
            config=config,
            target=target,
            capabilities=capabilities,
        )

        url = f"{self._endpoint_url}/v1/texture-variations"
        logger.info(
            "POST %s (prompt='%s', strength=%.2f)",
            url,
            conditioning.text_prompt,
            config.strength,
        )

        with httpx.Client(timeout=self._timeout, headers=self._headers) as client:
            # Submit job. A single-worker texture service may be busy with a
            # long GPU job; treat capacity responses as backpressure, not as a
            # material failure.
            try:
                resp = self._post_with_backpressure_retry(client, url, body)
            except httpx.HTTPError as exc:
                return JobStatus(
                    job_id="",
                    status="failed",
                    error_message=f"Texture service submit failed: {exc}",
                )
            if resp.status_code not in (200, 201, 202):
                return JobStatus(
                    job_id="",
                    status="failed",
                    error_message=f"HTTP {resp.status_code}: {resp.text}",
                )

            status = self._parse_status(resp.json())
            logger.info("Job submitted: %s (status=%s)", status.job_id, status.status)

            if not wait:
                return status

            # Poll until terminal or timeout
            deadline = time.time() + timeout_sec
            poll_interval = 2.0

            while status.status in ("queued", "processing"):
                if time.time() > deadline:
                    logger.warning("Timeout waiting for job %s", status.job_id)
                    return JobStatus(
                        job_id=status.job_id,
                        status="failed",
                        progress=status.progress,
                        message=status.message,
                        error_message=(
                            f"Timed out waiting for job {status.job_id} after "
                            f"{timeout_sec}s (last status: {status.status})"
                        ),
                    )

                time.sleep(poll_interval)
                try:
                    status = self.get_status(status.job_id, client=client)
                except (httpx.HTTPError, RuntimeError) as exc:
                    logger.warning(
                        "Texture service poll failed for job %s: %s",
                        status.job_id,
                        exc,
                    )
                    return JobStatus(
                        job_id=status.job_id,
                        status="failed",
                        progress=status.progress,
                        message=status.message,
                        error_message=(
                            f"Texture service poll failed for job "
                            f"{status.job_id}: {exc}"
                        ),
                    )
                logger.info(
                    "Job %s: %s (%d%%) %s",
                    status.job_id,
                    status.status,
                    status.progress,
                    status.message or "",
                )

                # Back off gradually
                poll_interval = min(poll_interval * 1.5, 10.0)

            return status

    def _post_with_backpressure_retry(
        self,
        client: httpx.Client,
        url: str,
        body: dict[str, Any],
    ) -> httpx.Response:
        deadline = time.time() + self._submit_retry_timeout_sec
        attempt = 0
        while True:
            try:
                resp = client.post(url, json=body)
            except httpx.TransportError as exc:
                now = time.time()
                if now >= deadline:
                    raise
                delay = self._submit_backoff_delay(attempt)
                delay = min(delay, max(0.0, deadline - now))
                if delay <= 0.0:
                    raise
                attempt += 1
                logger.warning(
                    "Transient error submitting texture job: %s; retrying in %.1fs",
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue
            if resp.status_code not in _SUBMIT_RETRYABLE_STATUS_CODES:
                return resp
            now = time.time()
            if now >= deadline:
                return resp
            delay = self._submit_retry_delay(resp, attempt)
            delay = min(delay, max(0.0, deadline - now))
            if delay <= 0.0:
                return resp
            attempt += 1
            logger.warning(
                "Texture service is busy (HTTP %s); retrying submit in %.1fs",
                resp.status_code,
                delay,
            )
            time.sleep(delay)

    def _submit_backoff_delay(self, attempt: int) -> float:
        backoff = self._submit_retry_base_delay_sec * (2**attempt)
        return min(backoff, self._submit_retry_max_delay_sec)

    def _submit_retry_delay(self, resp: httpx.Response, attempt: int) -> float:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                retry_after_sec = float(retry_after)
            except ValueError:
                retry_after_sec = 0.0
            if math.isfinite(retry_after_sec) and retry_after_sec > 0.0:
                return min(retry_after_sec, self._submit_retry_max_delay_sec)
        return self._submit_backoff_delay(attempt)

    def get_status(
        self,
        job_id: str,
        client: httpx.Client | None = None,
    ) -> JobStatus:
        """Query job status."""
        url = f"{self._endpoint_url}/v1/texture-variations/{job_id}"

        resp: httpx.Response | None = None
        for attempt in range(1, _STATUS_POLL_MAX_ATTEMPTS + 1):
            try:
                if client:
                    resp = client.get(url)
                else:
                    with httpx.Client(
                        timeout=self._timeout,
                        headers=self._headers,
                    ) as c:
                        resp = c.get(url)
                break
            except httpx.TransportError as exc:
                if attempt >= _STATUS_POLL_MAX_ATTEMPTS:
                    raise
                delay = min(_STATUS_POLL_RETRY_BASE_DELAY_SEC * attempt, 5.0)
                logger.warning(
                    "Transient error polling texture job %s (%s/%s): %s; "
                    "retrying in %.1fs",
                    job_id,
                    attempt,
                    _STATUS_POLL_MAX_ATTEMPTS,
                    exc,
                    delay,
                )
                time.sleep(delay)

        if resp is None:
            raise RuntimeError(f"Failed to poll texture job {job_id}")

        if resp.status_code == 404:
            return JobStatus(
                job_id=job_id,
                status="failed",
                error_message=self._job_not_found_message(client),
            )
        resp.raise_for_status()
        return self._parse_status(resp.json())

    def _job_not_found_message(self, client: httpx.Client | None) -> str:
        message = "Job not found"
        health = self._service_health_summary(client)
        if health:
            message = f"{message}; texture service health: {health}"
        return message

    def _service_health_summary(self, client: httpx.Client | None) -> str | None:
        url = f"{self._endpoint_url}/health"
        try:
            if client:
                resp = client.get(url)
            else:
                with httpx.Client(timeout=10.0, headers=self._headers) as c:
                    resp = c.get(url)
        except httpx.HTTPError as exc:
            return f"unreachable ({exc})"
        if resp.status_code >= 400:
            return f"HTTP {resp.status_code}"
        try:
            data = resp.json()
        except ValueError:
            text = resp.text.strip().replace("\n", " ")
            return f"non-JSON response {text[:80]!r}"
        if not isinstance(data, dict):
            return None
        summary_keys = (
            "status",
            "ready",
            "accepting_jobs",
            "active_jobs",
            "queued_jobs",
            "max_workers",
            "gpu_available",
            "error",
        )
        compact = {
            key: data[key]
            for key in summary_keys
            if key in data and data[key] not in (None, "")
        }
        if not compact:
            return None
        return ", ".join(f"{key}={value}" for key, value in compact.items())

    def cancel(self, job_id: str) -> None:
        """Cancel a job."""
        url = f"{self._endpoint_url}/v1/texture-variations/{job_id}"
        with httpx.Client(timeout=30, headers=self._headers) as client:
            resp = client.delete(url)
            if resp.status_code == 409:
                raise ValueError("Job already in terminal state")
            resp.raise_for_status()

    @staticmethod
    def _build_request_body(
        *,
        source_asset_uri: str,
        conditioning: Conditioning,
        config: TextureVariationConfig,
        target: TextureTarget | None = None,
        capabilities: BackendCapabilities | None = None,
    ) -> dict[str, Any]:
        """Build the normalized Texture Variation API request body."""
        body: dict[str, Any] = {
            "source_asset_uri": source_asset_uri,
            "conditioning": {
                "text_prompt": conditioning.text_prompt,
                "reference_image_uris": conditioning.reference_image_uris,
                "turntable_video_uri": conditioning.turntable_video_uri,
                "multiview_image_uris": conditioning.multiview_image_uris,
            },
            "configuration": {
                "strength": config.strength,
                "seed": config.seed,
                "variant_name": config.variant_name,
                "engine": config.engine,
                "texture_size": config.texture_size,
                "custom_parameters": config.custom_parameters,
            },
        }
        if target:
            body["target"] = asdict(target)
        if capabilities:
            body["capabilities"] = asdict(capabilities)
        return body

    @staticmethod
    def _parse_status(data: dict[str, Any]) -> JobStatus:
        """Parse a JSON response into a JobStatus."""
        result = None
        if data.get("result"):
            r = data["result"]
            gt = r.get("generated_textures") or {}
            maps = {
                key: MapArtifact(
                    uri=str(value.get("uri") or ""),
                    width=value.get("width"),
                    height=value.get("height"),
                    mime_type=value.get("mime_type", "image/png"),
                    colorspace=value.get("colorspace"),
                    packing=value.get("packing"),
                )
                for key, value in (r.get("maps") or {}).items()
                if isinstance(value, dict) and value.get("uri")
            }
            result = GenerationResult(
                variant_asset_uri=r.get("variant_asset_uri", ""),
                variant_name=r.get("variant_name", ""),
                generated_textures=GeneratedTextures(
                    albedo=gt.get("albedo"),
                    normal=gt.get("normal"),
                    orm=gt.get("orm"),
                ),
                maps=maps,
                auxiliary_artifacts=r.get("auxiliary_artifacts") or {},
                metadata=r.get("metadata") or {},
                diagnostics=r.get("diagnostics") or [],
            )

        return JobStatus(
            job_id=data.get("job_id", ""),
            status=data.get("status", "failed"),
            progress=data.get("progress", 0),
            message=data.get("message"),
            result=result,
            error_message=data.get("error_message"),
        )
