# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility Texture Variation API adapter for rendered refinement trials.

This module remains local to the legacy Material Agent refinement workflow so
``material-agent`` does not acquire a reverse dependency on ``texture-agent``.
The provider-neutral upload/download transport and client convergence decision
is recorded in the material-authoring architecture documentation.
"""

from __future__ import annotations

import json
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode, urlparse

from pxr import UsdUtils

from .contracts import TextureVariationSettings

CancelCheck = Callable[[], bool]
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


class TextureVariationError(RuntimeError):
    """Raised when a Texture Variation API request cannot produce valid maps."""


@dataclass(frozen=True)
class TextureVariationRequest:
    """One seeded candidate request at the existing Texture Variation boundary."""

    source_asset_path: Path
    material_path: str
    prompt: str
    reference_image_paths: tuple[Path, ...]
    strength: float
    seed: int
    variant_name: str


@dataclass(frozen=True)
class TextureVariationArtifacts:
    """Normalized local PBR maps returned by one variation request."""

    albedo_path: Path
    normal_path: Path
    orm_path: Path
    variant_asset_uri: str
    metadata: dict[str, Any]
    diagnostics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _GenerationResult:
    """Validated subset of a completed Texture Variation result."""

    variant_asset_uri: str
    variant_name: str
    generated_textures: dict[str, str | None]
    maps: dict[str, str]
    metadata: dict[str, Any]
    diagnostics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _JobStatus:
    """Validated subset of the asynchronous Texture Variation job status."""

    job_id: str
    status: str
    message: str | None
    result: _GenerationResult | None
    error_message: str | None


class TextureVariationGenerator(Protocol):
    """Candidate generator boundary shared by real services and test fakes."""

    @property
    def name(self) -> str: ...

    def generate(
        self,
        request: TextureVariationRequest,
        *,
        output_dir: Path,
        cancel_check: CancelCheck | None = None,
    ) -> TextureVariationArtifacts: ...


def _validated_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("variation.endpoint must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "variation.endpoint must not contain credentials, query, or fragment"
        )
    return endpoint.rstrip("/")


def _http_origin(uri: str) -> tuple[str, str, int]:
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("URI must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("URI must not contain credentials")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname.casefold(), port


def _require_same_origin(uri: str, allowed_origin: tuple[str, str, int]) -> None:
    try:
        origin = _http_origin(uri)
    except ValueError as error:
        raise TextureVariationError(
            "texture variation artifact URI is invalid"
        ) from error
    if origin != allowed_origin:
        raise TextureVariationError(
            "texture variation artifact URI must match the configured endpoint origin"
        )


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_origin: tuple[str, str, int]) -> None:
        self._allowed_origin = allowed_origin

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _require_same_origin(newurl, self._allowed_origin)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _response_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TextureVariationError(
            f"texture variation response field {field_name} must be an object"
        )
    return {str(key): item for key, item in value.items()}


def _response_string(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise TextureVariationError(
            f"texture variation response field {field_name} must be a string"
        )
    return value


def _optional_response_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _response_string(value, field_name=field_name)


def _parse_generation_result(value: Any) -> _GenerationResult:
    result = _response_mapping(value, field_name="result")
    generated = _response_mapping(
        result.get("generated_textures"),
        field_name="result.generated_textures",
    )
    generated_textures: dict[str, str | None] = {}
    for channel in ("albedo", "normal", "orm"):
        uri = generated.get(channel)
        generated_textures[channel] = (
            _response_string(
                uri,
                field_name=f"result.generated_textures.{channel}",
            )
            if uri is not None
            else None
        )

    raw_maps = _response_mapping(result.get("maps", {}), field_name="result.maps")
    maps: dict[str, str] = {}
    for channel, raw_artifact in raw_maps.items():
        artifact = _response_mapping(
            raw_artifact,
            field_name=f"result.maps.{channel}",
        )
        maps[channel] = _response_string(
            artifact.get("uri"),
            field_name=f"result.maps.{channel}.uri",
        )

    metadata = _response_mapping(
        result.get("metadata", {}), field_name="result.metadata"
    )
    raw_diagnostics = result.get("diagnostics", [])
    if not isinstance(raw_diagnostics, list):
        raise TextureVariationError(
            "texture variation response field result.diagnostics must be a list"
        )
    diagnostics = tuple(
        _response_mapping(
            diagnostic,
            field_name=f"result.diagnostics[{index}]",
        )
        for index, diagnostic in enumerate(raw_diagnostics)
    )
    return _GenerationResult(
        variant_asset_uri=_response_string(
            result.get("variant_asset_uri"),
            field_name="result.variant_asset_uri",
        ),
        variant_name=_response_string(
            result.get("variant_name"), field_name="result.variant_name"
        ),
        generated_textures=generated_textures,
        maps=maps,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def _parse_job_status(value: Any) -> _JobStatus:
    payload = _response_mapping(value, field_name="job")
    status = _response_string(payload.get("status"), field_name="job.status")
    if status not in {"queued", "processing", "completed", "failed", "cancelled"}:
        raise TextureVariationError(
            f"texture variation response has unsupported job status: {status}"
        )
    raw_job_id = payload.get("job_id")
    if not isinstance(raw_job_id, str):
        raise TextureVariationError(
            "texture variation response field job.job_id must be a string"
        )
    raw_result = payload.get("result")
    return _JobStatus(
        job_id=raw_job_id,
        status=status,
        message=_optional_response_string(
            payload.get("message"), field_name="job.message"
        ),
        result=(
            _parse_generation_result(raw_result) if raw_result is not None else None
        ),
        error_message=_optional_response_string(
            payload.get("error_message"), field_name="job.error_message"
        ),
    )


class RestTextureVariationGenerator:
    """Synchronous client for the repository's Texture Variation API contract."""

    def __init__(self, settings: TextureVariationSettings) -> None:
        if settings.endpoint is None:
            raise ValueError(
                "variation.endpoint is required when no generator is injected"
            )
        self._settings = settings
        self._endpoint = _validated_endpoint(settings.endpoint)
        self._artifact_origin = _http_origin(self._endpoint)

    @property
    def name(self) -> str:
        return "texture-variation-api"

    def generate(
        self,
        request: TextureVariationRequest,
        *,
        output_dir: Path,
        cancel_check: CancelCheck | None = None,
    ) -> TextureVariationArtifacts:
        if cancel_check is not None and cancel_check():
            raise TextureVariationError("texture variation cancelled by caller")
        source = request.source_asset_path.resolve()
        if not source.is_file():
            raise FileNotFoundError("texture variation source USD does not exist")
        for reference in request.reference_image_paths:
            if not reference.resolve().is_file():
                raise FileNotFoundError(
                    f"texture variation reference image does not exist: {reference.name}"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="material-variation-upload-") as raw:
            package = Path(raw) / "source_material.usdz"
            if not UsdUtils.CreateNewUsdzPackage(str(source), str(package)):
                raise TextureVariationError(
                    "failed to package texture variation source USD"
                )
            source_uri = self._upload_asset(package)
            reference_uris = [
                self._upload_asset(path.resolve())
                for path in request.reference_image_paths
            ]
            configuration: dict[str, Any] = {
                "strength": request.strength,
                "seed": request.seed,
                "variant_name": request.variant_name,
                "texture_size": self._settings.texture_size,
                "custom_parameters": self._settings.custom_parameters,
            }
            if self._settings.engine is not None:
                configuration["engine"] = self._settings.engine
            payload = {
                "source_asset_uri": source_uri,
                "target": {"material_path": request.material_path},
                "conditioning": {
                    "text_prompt": request.prompt,
                    "reference_image_uris": reference_uris,
                },
                "configuration": configuration,
            }
        submitted = self._request_json(
            "POST",
            f"{self._endpoint}/v1/texture-variations",
            payload,
        )
        status = _parse_job_status(submitted)
        if not status.job_id:
            raise TextureVariationError("texture variation service returned no job id")
        deadline = time.monotonic() + self._settings.timeout_seconds
        while status.status not in {"completed", "failed", "cancelled"}:
            if cancel_check is not None and cancel_check():
                self._cancel_job(status.job_id)
                raise TextureVariationError("texture variation cancelled by caller")
            if time.monotonic() >= deadline:
                self._cancel_job(status.job_id)
                raise TextureVariationError("texture variation service timed out")
            time.sleep(self._settings.poll_interval_seconds)
            status = _parse_job_status(
                self._request_json(
                    "GET",
                    f"{self._endpoint}/v1/texture-variations/{status.job_id}",
                )
            )
        if status.status != "completed" or status.result is None:
            message = status.error_message or status.message or status.status
            raise TextureVariationError(
                f"texture variation job did not complete successfully: {message}"
            )
        return self._localize_result(status.result, output_dir)

    def _upload_asset(self, path: Path) -> str:
        payload = self._read_local_artifact(path)
        query = urlencode({"filename": path.name})
        request = urllib.request.Request(
            f"{self._endpoint}/v1/texture-variation-assets?{query}",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/octet-stream",
                "Accept": "application/json",
            },
        )
        opener = urllib.request.build_opener(
            _SameOriginRedirectHandler(self._artifact_origin)
        )
        try:
            with opener.open(
                request,  # noqa: S310 - validated HTTP(S) endpoint
                timeout=min(self._settings.timeout_seconds, 60.0),
            ) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise TextureVariationError(
                f"texture variation input upload failed: {type(error).__name__}"
            ) from error
        response_payload = _response_mapping(decoded, field_name="upload")
        uri = _response_string(
            response_payload.get("asset_uri"), field_name="upload.asset_uri"
        )
        _require_same_origin(uri, self._artifact_origin)
        return uri

    def _read_local_artifact(self, path: Path) -> bytes:
        try:
            with path.open("rb") as source:
                payload = source.read(self._settings.max_artifact_bytes + 1)
        except OSError as error:
            raise TextureVariationError(
                "failed to read texture variation input artifact"
            ) from error
        if len(payload) > self._settings.max_artifact_bytes:
            raise TextureVariationError(
                "texture variation input exceeds max_artifact_bytes"
            )
        return payload

    def _cancel_job(self, job_id: str) -> None:
        try:
            self._request_json(
                "DELETE", f"{self._endpoint}/v1/texture-variations/{job_id}"
            )
        except TextureVariationError:
            pass

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        opener = urllib.request.build_opener(
            _SameOriginRedirectHandler(self._artifact_origin)
        )
        try:
            with opener.open(
                request,  # noqa: S310 - validated HTTP(S) endpoint
                timeout=min(self._settings.timeout_seconds, 60.0),
            ) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise TextureVariationError(
                f"texture variation service request failed: {type(error).__name__}"
            ) from error
        if not isinstance(decoded, dict):
            raise TextureVariationError(
                "texture variation service returned non-object JSON"
            )
        return decoded

    def _localize_result(
        self,
        result: _GenerationResult | dict[str, Any],
        output_dir: Path,
    ) -> TextureVariationArtifacts:
        if isinstance(result, dict):
            result = _parse_generation_result(result)
        output_dir.mkdir(parents=True, exist_ok=True)
        uris = dict(result.generated_textures)
        for channel, uri in result.maps.items():
            if channel in uris and not uris[channel]:
                uris[channel] = uri
        missing = [name for name, uri in uris.items() if not uri]
        if missing:
            raise TextureVariationError(
                "texture variation result is missing required maps: "
                + ", ".join(sorted(missing))
            )
        localized = {
            name: self._localize_uri(str(uri), output_dir / f"{name}.png")
            for name, uri in uris.items()
        }
        return TextureVariationArtifacts(
            albedo_path=localized["albedo"],
            normal_path=localized["normal"],
            orm_path=localized["orm"],
            variant_asset_uri=result.variant_asset_uri,
            metadata=dict(result.metadata),
            diagnostics=tuple(dict(item) for item in result.diagnostics),
        )

    def _localize_uri(self, uri: str, destination: Path) -> Path:
        parsed = urlparse(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if parsed.scheme in {"http", "https"}:
            _require_same_origin(uri, self._artifact_origin)
            opener = urllib.request.build_opener(
                _SameOriginRedirectHandler(self._artifact_origin)
            )
            try:
                with (
                    opener.open(
                        uri,
                        timeout=min(self._settings.timeout_seconds, 60.0),
                    ) as response,
                    destination.open("wb") as target,
                ):
                    written = 0
                    while True:
                        chunk = response.read(
                            min(
                                _DOWNLOAD_CHUNK_BYTES,
                                self._settings.max_artifact_bytes + 1 - written,
                            )
                        )
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > self._settings.max_artifact_bytes:
                            raise TextureVariationError(
                                "texture variation artifact exceeds max_artifact_bytes"
                            )
                        target.write(chunk)
            except TextureVariationError:
                destination.unlink(missing_ok=True)
                raise
            except (urllib.error.URLError, TimeoutError) as error:
                destination.unlink(missing_ok=True)
                raise TextureVariationError(
                    f"failed to download texture variation artifact: {type(error).__name__}"
                ) from error
            return destination
        if parsed.scheme not in {"", "file"}:
            raise TextureVariationError(
                f"unsupported texture variation artifact URI scheme: {parsed.scheme}"
            )
        source = (
            Path(urllib.request.url2pathname(f"//{parsed.netloc}{parsed.path}"))
            if parsed.scheme == "file"
            else Path(uri)
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"texture variation artifact does not exist: {source.name}"
            )
        if source.resolve() == destination.resolve():
            self._read_local_artifact(source)
            return source
        try:
            with source.open("rb") as input_file, destination.open("wb") as target:
                written = 0
                while True:
                    chunk = input_file.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > self._settings.max_artifact_bytes:
                        raise TextureVariationError(
                            "texture variation artifact exceeds max_artifact_bytes"
                        )
                    target.write(chunk)
        except TextureVariationError:
            destination.unlink(missing_ok=True)
            raise
        except OSError as error:
            destination.unlink(missing_ok=True)
            raise TextureVariationError(
                "failed to copy texture variation artifact"
            ) from error
        return destination


__all__ = [
    "RestTextureVariationGenerator",
    "TextureVariationArtifacts",
    "TextureVariationError",
    "TextureVariationGenerator",
    "TextureVariationRequest",
]
