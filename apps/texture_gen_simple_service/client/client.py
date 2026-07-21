# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small HTTP client for the shared Texture Variation API contract."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from apps.texture_gen_service_common import CreateJobRequest, JobStatus


class TextureVariationClient:
    """Synchronous stdlib client for /v1/texture-variations endpoints."""

    def __init__(self, endpoint_url: str, *, timeout: float = 60.0) -> None:
        self._endpoint_url = endpoint_url.rstrip("/")
        self._timeout = timeout

    def create_texture_variation(
        self, request: CreateJobRequest | dict[str, Any]
    ) -> JobStatus:
        payload = (
            request.model_dump(mode="json")
            if isinstance(request, CreateJobRequest)
            else request
        )
        status = self._request(
            "POST",
            "/v1/texture-variations",
            payload=payload,
        )
        assert status is not None
        return status

    def get_texture_variation(self, job_id: str) -> JobStatus:
        status = self._request("GET", f"/v1/texture-variations/{job_id}")
        assert status is not None
        return status

    def cancel_texture_variation(self, job_id: str) -> None:
        self._request(
            "DELETE",
            f"/v1/texture-variations/{job_id}",
            expect_body=False,
            expected_statuses={204},
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        expect_body: bool = True,
        expected_statuses: set[int] | None = None,
    ) -> JobStatus | None:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self._endpoint_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                status_code = response.status
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Texture variation request failed with HTTP {exc.code}: "
                f"{response_body}"
            ) from exc

        if expected_statuses is not None and status_code not in expected_statuses:
            raise RuntimeError(
                "Texture variation request returned unexpected HTTP "
                f"{status_code}: {response_body}"
            )
        if not expect_body:
            return None
        if not response_body:
            raise RuntimeError(
                f"Texture variation request returned empty HTTP {status_code} body."
            )
        return JobStatus.model_validate_json(response_body)
