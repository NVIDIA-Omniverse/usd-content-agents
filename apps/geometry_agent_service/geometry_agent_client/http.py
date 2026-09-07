# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded HTTP client for the authenticated Geometry Agent service."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024


class GeometryAgentClientError(RuntimeError):
    """Typed, secret-free service client failure."""


def _service_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    try:
        parsed = urlparse(normalized)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Geometry Agent service URL is invalid") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Geometry Agent service URL must be credential-free")
    loopback = parsed.hostname.casefold() == "localhost"
    try:
        loopback = loopback or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        pass
    if parsed.scheme != "https" and not loopback:
        raise ValueError("Non-loopback Geometry Agent services require HTTPS")
    return normalized


class GeometryAgentClient:
    """Call one explicitly configured service without redirects or URL fetching."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 600.0,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Geometry Agent service API key is required")
        self.base_url = _service_url(base_url)
        self._api_key = api_key
        self._timeouts = (connect_timeout_seconds, read_timeout_seconds)
        if session is None:
            session = requests.Session()
            session.trust_env = False
        self._session = session

    def providers(self) -> dict[str, Any]:
        return self._json("GET", "/api/geometry/providers")

    def upload(
        self,
        path: str | Path,
        *,
        role: str,
        media_type: str,
        archive_entrypoint: str | None = None,
    ) -> dict[str, Any]:
        raw_source = Path(path).expanduser()
        if raw_source.is_symlink():
            raise ValueError(f"Upload source is not a regular file: {raw_source}")
        source = raw_source.resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Upload source is not a regular file: {source}")
        digest = _file_sha256(source)
        data = {"role": role}
        if archive_entrypoint is not None:
            data["archive_entrypoint"] = archive_entrypoint
        with source.open("rb") as stream:
            return self._json(
                "POST",
                "/api/geometry/sources",
                headers={"X-Content-SHA256": digest},
                data=data,
                files={"file": (source.name, stream, media_type)},
            )

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/api/geometry/generations", json_payload=payload)

    def revise(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/api/geometry/revisions", json_payload=payload)

    def family(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/api/geometry/families", json_payload=payload)

    def export(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/api/geometry/exports", json_payload=payload)

    def provider_export(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json(
            "POST",
            "/api/geometry/provider-exports",
            json_payload=payload,
        )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/api/geometry/runs", json_payload=payload)

    def job(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/api/geometry/jobs/{job_id}")

    def wait(
        self,
        job: dict[str, Any],
        *,
        timeout_seconds: float = 3600.0,
        poll_seconds: float = 1.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        current = job
        while current.get("status") in {"queued", "running"}:
            if time.monotonic() >= deadline:
                raise GeometryAgentClientError("Geometry Agent job timed out")
            job_id = current.get("job_id")
            if not isinstance(job_id, str):
                raise GeometryAgentClientError("Geometry Agent returned an invalid job")
            time.sleep(poll_seconds)
            current = self.job(job_id)
        return current

    def download(
        self,
        artifact_id: str,
        destination: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> Path:
        target = Path(destination).expanduser().absolute()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Download destination already exists: {target}")
        response = self._request(
            "GET",
            f"/api/geometry/artifacts/{artifact_id}",
            stream=True,
        )
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("xb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > _MAX_DOWNLOAD_BYTES:
                        raise GeometryAgentClientError(
                            "Geometry Agent artifact exceeds the download limit"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            observed = digest.hexdigest()
            header_digest = response.headers.get("X-Content-SHA256")
            required = expected_sha256 or header_digest
            if required is not None and observed != required:
                raise GeometryAgentClientError(
                    "Downloaded artifact differs from its SHA-256 binding"
                )
            os.replace(temporary, target)
        finally:
            response.close()
            temporary.unlink(missing_ok=True)
        return target

    def _json(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = self._request(method, path, json=json_payload, **kwargs)
        try:
            content = _bounded_response_content(response, max_bytes=_MAX_JSON_BYTES)
            payload = json.loads(content)
        except (UnicodeError, json.JSONDecodeError, requests.RequestException) as exc:
            raise GeometryAgentClientError(
                "Geometry Agent returned invalid JSON"
            ) from exc
        finally:
            response.close()
        if not isinstance(payload, dict):
            raise GeometryAgentClientError("Geometry Agent returned non-object JSON")
        return payload

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        headers = dict(kwargs.pop("headers", {}))
        headers["X-Geometry-Agent-Service-Key"] = self._api_key
        kwargs.setdefault("stream", True)
        try:
            response = self._session.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                timeout=self._timeouts,
                allow_redirects=False,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise GeometryAgentClientError(
                "Geometry Agent service request failed"
            ) from exc
        if 300 <= response.status_code < 400:
            response.close()
            raise GeometryAgentClientError("Geometry Agent service redirect rejected")
        if response.status_code >= 400:
            summary = f"Geometry Agent service returned HTTP {response.status_code}"
            try:
                payload = json.loads(
                    _bounded_response_content(response, max_bytes=_MAX_JSON_BYTES)
                )
                detail = payload.get("detail") if isinstance(payload, dict) else None
                if isinstance(detail, dict) and isinstance(detail.get("message"), str):
                    summary = detail["message"]
            except (
                GeometryAgentClientError,
                UnicodeError,
                json.JSONDecodeError,
                requests.RequestException,
            ):
                pass
            response.close()
            raise GeometryAgentClientError(summary)
        return response


def _bounded_response_content(
    response: requests.Response,
    *,
    max_bytes: int,
) -> bytes:
    raw_length = response.headers.get("Content-Length")
    if raw_length is not None:
        try:
            if int(raw_length) > max_bytes:
                raise GeometryAgentClientError(
                    "Geometry Agent returned an oversized JSON response"
                )
        except ValueError:
            pass
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        size += len(chunk)
        if size > max_bytes:
            raise GeometryAgentClientError(
                "Geometry Agent returned an oversized JSON response"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["GeometryAgentClient", "GeometryAgentClientError"]
