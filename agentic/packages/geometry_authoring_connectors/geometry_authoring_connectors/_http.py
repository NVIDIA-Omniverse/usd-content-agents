# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded HTTP transport shared by public reference connectors."""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast
from urllib.parse import urlparse

import requests

from .errors import (
    ArtifactLimitError,
    ConnectorConfigurationError,
    InvalidProviderResponseError,
    ProviderTransportError,
)

DEFAULT_MAX_JSON_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_REQUEST_BYTES = 192 * 1024 * 1024
DEFAULT_MAX_BINARY_BYTES = 256 * 1024 * 1024


class HttpResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def iter_content(self, chunk_size: int) -> Any: ...

    def close(self) -> None: ...


class HttpTransport(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse: ...


RequestHeaderProvider = Callable[[str, str, str], Mapping[str, str]]


def _is_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_credential_free_url(
    url: str,
    *,
    purpose: str,
    allowed_host_suffix: str | None = None,
) -> str:
    """Validate one administrator-selected endpoint without resolving DNS."""

    candidate = url.strip().rstrip("/")
    try:
        candidate.encode("ascii")
        parsed = urlparse(candidate)
        port = parsed.port
    except (UnicodeEncodeError, ValueError) as exc:
        raise ConnectorConfigurationError(f"{purpose} URL is invalid") from exc
    if any(ord(character) <= 0x20 or ord(character) == 0x7F for character in candidate):
        raise ConnectorConfigurationError(f"{purpose} URL contains control characters")
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
        or "\\" in candidate
    ):
        raise ConnectorConfigurationError(
            f"{purpose} URL must be a credential-free HTTP URL without query or fragment"
        )
    if port is not None and not 1 <= port <= 65_535:
        raise ConnectorConfigurationError(f"{purpose} URL has an invalid port")
    hostname = parsed.hostname.lower().rstrip(".")
    if parsed.scheme == "http" and not _is_loopback(hostname):
        raise ConnectorConfigurationError(f"non-loopback {purpose} endpoints require HTTPS")
    if allowed_host_suffix is not None:
        suffix = allowed_host_suffix.lower().lstrip(".")
        if hostname != suffix and not hostname.endswith(f".{suffix}"):
            raise ConnectorConfigurationError(
                f"{purpose} endpoint must use the official {suffix} API domain"
            )
    return candidate


class BoundedHttpClient:
    """A no-redirect, size-bounded client with secret-free failures."""

    def __init__(
        self,
        *,
        provider_id: str,
        endpoint_alias: str,
        base_url: str,
        bearer_token: str | None,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        transport: HttpTransport | None = None,
        allowed_host_suffix: str | None = None,
        request_header_provider: RequestHeaderProvider | None = None,
    ) -> None:
        alias = endpoint_alias.strip()
        if not alias:
            raise ConnectorConfigurationError("endpoint alias must not be empty")
        if not provider_id.strip():
            raise ConnectorConfigurationError("provider id must not be empty")
        if connect_timeout_seconds <= 0 or read_timeout_seconds <= 0:
            raise ConnectorConfigurationError("HTTP timeouts must be positive")
        if connect_timeout_seconds > 300 or read_timeout_seconds > 900:
            raise ConnectorConfigurationError("HTTP timeouts exceed connector bounds")
        if bearer_token is not None and (
            not bearer_token
            or len(bearer_token) > 16_384
            or not bearer_token.isascii()
            or bearer_token != bearer_token.strip()
            or any(ord(character) < 33 or ord(character) == 127 for character in bearer_token)
        ):
            raise ConnectorConfigurationError(
                "bearer token must be one bounded printable ASCII value"
            )
        if bearer_token is not None and request_header_provider is not None:
            raise ConnectorConfigurationError(
                "bearer token and request header provider are mutually exclusive"
            )
        self._provider_id = provider_id
        self._endpoint_alias = alias
        self._base_url = validate_credential_free_url(
            base_url,
            purpose=alias,
            allowed_host_suffix=allowed_host_suffix,
        )
        self._bearer_token = bearer_token
        self._request_header_provider = request_header_provider
        self._timeout = (connect_timeout_seconds, read_timeout_seconds)
        self._transport: HttpTransport
        if transport is None:
            session = requests.Session()
            session.trust_env = False
            self._transport = cast(HttpTransport, session)
        else:
            self._transport = transport

    @property
    def base_url(self) -> str:
        return self._base_url

    def _url(self, relative_path: str) -> str:
        if not relative_path:
            return self._base_url
        if not relative_path.startswith("/") or "?" in relative_path or "#" in relative_path:
            raise ConnectorConfigurationError("connector request path is invalid")
        if ".." in relative_path.split("/") or "\\" in relative_path:
            raise ConnectorConfigurationError("connector request path is unsafe")
        return f"{self._base_url}{relative_path}"

    def _request(
        self,
        method: str,
        *,
        relative_path: str,
        payload: Mapping[str, Any] | None,
        accept: str,
        expected_statuses: frozenset[int],
    ) -> HttpResponse:
        headers = {"Accept": accept}
        body: bytes | None = None
        if payload is not None:
            try:
                body = json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ConnectorConfigurationError("connector request is not valid JSON") from exc
            if len(body) > DEFAULT_MAX_REQUEST_BYTES:
                raise ArtifactLimitError(
                    "connector request exceeds the bounded JSON limit",
                    provider_id=self._provider_id,
                )
            headers["Content-Type"] = "application/json"
        if self._bearer_token is not None:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        url = self._url(relative_path)
        method_name = method.upper()
        if self._request_header_provider is not None:
            try:
                provided_headers = self._request_header_provider(
                    method_name,
                    url,
                    headers.get("Content-Type", ""),
                )
            except Exception as exc:
                raise ConnectorConfigurationError(
                    "connector authentication headers could not be created"
                ) from exc
            allowed_names = {"authorization", "date", "on-nonce"}
            for name, value in provided_headers.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(value, str)
                    or name.lower() not in allowed_names
                ):
                    raise ConnectorConfigurationError(
                        "connector authentication returned an unsupported header"
                    )
                try:
                    name.encode("ascii")
                    value.encode("ascii")
                except UnicodeEncodeError as exc:
                    raise ConnectorConfigurationError(
                        "connector authentication headers must be ASCII"
                    ) from exc
                if (
                    not name
                    or not value
                    or len(name) > 64
                    or len(value) > 4096
                    or value != value.strip()
                    or any(
                        ord(character) < 32 or ord(character) == 127 for character in name + value
                    )
                ):
                    raise ConnectorConfigurationError(
                        "connector authentication returned an invalid header"
                    )
                headers[name] = value
        try:
            response = self._transport.request(
                method_name,
                url,
                data=body,
                headers=headers,
                timeout=self._timeout,
                allow_redirects=False,
                stream=True,
            )
        except Exception as exc:
            raise ProviderTransportError(
                f"{self._endpoint_alias} request failed without fallback",
                provider_id=self._provider_id,
            ) from exc
        if 300 <= response.status_code < 400:
            try:
                response.close()
            finally:
                raise ProviderTransportError(
                    f"{self._endpoint_alias} redirects are not permitted",
                    provider_id=self._provider_id,
                )
        if response.status_code not in expected_statuses:
            status = response.status_code
            try:
                response.close()
            finally:
                error_type = (
                    ProviderTransportError
                    if status in {408, 425, 429} or 500 <= status < 600
                    else InvalidProviderResponseError
                )
                raise error_type(
                    f"{self._endpoint_alias} returned HTTP {status} without fallback",
                    provider_id=self._provider_id,
                )
        return response

    def _read_response(self, response: HttpResponse, *, max_bytes: int) -> bytes:
        content_length = response.headers.get("Content-Length")
        declared_length: int | None = None
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except (TypeError, ValueError) as exc:
                raise InvalidProviderResponseError(
                    f"{self._endpoint_alias} returned an invalid content length",
                    provider_id=self._provider_id,
                ) from exc
            if declared_length < 0:
                raise InvalidProviderResponseError(
                    f"{self._endpoint_alias} returned an invalid content length",
                    provider_id=self._provider_id,
                )
            if declared_length > max_bytes:
                raise ArtifactLimitError(
                    f"{self._endpoint_alias} response exceeds the byte limit",
                    provider_id=self._provider_id,
                )
        result = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                result.extend(chunk)
                if len(result) > max_bytes:
                    raise ArtifactLimitError(
                        f"{self._endpoint_alias} response exceeds the byte limit",
                        provider_id=self._provider_id,
                    )
        except (ArtifactLimitError, InvalidProviderResponseError):
            raise
        except Exception as exc:
            raise ProviderTransportError(
                f"{self._endpoint_alias} response stream failed without fallback",
                provider_id=self._provider_id,
            ) from exc
        if declared_length is not None and declared_length != len(result):
            raise InvalidProviderResponseError(
                f"{self._endpoint_alias} response differs from its content length",
                provider_id=self._provider_id,
            )
        return bytes(result)

    def request_json(
        self,
        method: str,
        *,
        relative_path: str = "",
        payload: Mapping[str, Any] | None = None,
        expected_statuses: frozenset[int] = frozenset({200}),
        max_bytes: int = DEFAULT_MAX_JSON_BYTES,
    ) -> dict[str, Any]:
        response = self._request(
            method,
            relative_path=relative_path,
            payload=payload,
            accept="application/json",
            expected_statuses=expected_statuses,
        )
        try:
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type and not (
                content_type == "application/json" or content_type.endswith("+json")
            ):
                raise InvalidProviderResponseError(
                    f"{self._endpoint_alias} returned a non-JSON content type",
                    provider_id=self._provider_id,
                )
            document = self._read_response(response, max_bytes=max_bytes)
            try:
                value = json.loads(document.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InvalidProviderResponseError(
                    f"{self._endpoint_alias} returned invalid JSON",
                    provider_id=self._provider_id,
                ) from exc
            if not isinstance(value, dict):
                raise InvalidProviderResponseError(
                    f"{self._endpoint_alias} JSON response must be an object",
                    provider_id=self._provider_id,
                )
            return value
        finally:
            try:
                response.close()
            except Exception:
                pass

    def request_bytes(
        self,
        method: str,
        *,
        relative_path: str,
        payload: Mapping[str, Any] | None = None,
        expected_statuses: frozenset[int] = frozenset({200}),
        max_bytes: int = DEFAULT_MAX_BINARY_BYTES,
    ) -> bytes:
        response = self._request(
            method,
            relative_path=relative_path,
            payload=payload,
            accept="application/octet-stream",
            expected_statuses=expected_statuses,
        )
        try:
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type in {"text/html", "application/xhtml+xml"}:
                raise InvalidProviderResponseError(
                    f"{self._endpoint_alias} returned an unexpected document",
                    provider_id=self._provider_id,
                )
            return self._read_response(response, max_bytes=max_bytes)
        finally:
            try:
                response.close()
            except Exception:
                pass
