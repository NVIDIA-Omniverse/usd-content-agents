# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One package-authenticated usd-cli sidecar/session owned by a workflow run."""

from __future__ import annotations

import hashlib
import http.client
import importlib.metadata
import ipaddress
import json
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)
from world_understanding.utils.artifacts import (
    delete_confined_file,
    open_confined_directory,
    open_confined_lock_file,
)
from world_understanding.utils.file_locking import blocking_exclusive_descriptor_lock
from world_understanding.utils.nvcf_utils import (
    NVCF_INVOCATION_HOST,
    resolve_endpoint_or_function_id,
)
from world_understanding.utils.windows_process import windows_process_start_token

from .artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    read_contained_artifact,
)
from .usd_cli import (
    UsdCliPackageRoute,
    UsdCliSubprocessOutputError,
    controlled_usd_cli_telemetry_env,
    find_usd_cli_repository_root,
    resolve_package_owned_usd_cli_route,
    run_bounded_usd_cli_subprocess,
)

_SECRET_ARGUMENT_PATTERN = re.compile(
    r"(?:api[_-]?key|authorization|cookie|credential|password|secret|signature|token)",
    re.IGNORECASE,
)

PARENT_USD_CLI_SESSION_IDENTITY_ENV = "CONTENT_WORKFLOW_PARENT_USD_CLI_SESSION_IDENTITY"
PARENT_USD_CLI_SESSION_IDENTITY_SHA256_ENV = (
    "CONTENT_WORKFLOW_PARENT_USD_CLI_SESSION_IDENTITY_SHA256"
)
PARENT_USD_CLI_MANAGED_ENV = "CONTENT_WORKFLOW_PARENT_USD_CLI_MANAGED"
PARENT_USD_CLI_PROXY_ALLOWED_ENV = "CONTENT_WORKFLOW_PARENT_USD_CLI_PROXY_ALLOWED"
PARENT_USD_CLI_SERVER_URL_ENV = "CONTENT_WORKFLOW_USD_CLI_SERVER_URL"
PARENT_USD_CLI_SESSION_ID_ENV = "CONTENT_WORKFLOW_USD_CLI_SESSION_ID"
PARENT_USD_CLI_TOKEN_ENV = "CONTENT_WORKFLOW_PARENT_USD_CLI_TOKEN"
USD_CLI_LOCAL_GPU_FORBIDDEN_ENV = "USD_CLI_LOCAL_GPU_FORBIDDEN"
USD_CLI_EXTERNAL_LIFECYCLE_ENV = "USD_CLI_LIFECYCLE_EXTERNALLY_OWNED"
PARENT_USD_CLI_SESSION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.parent-usd-cli-session.v1"
)
MAX_PARENT_USD_CLI_SESSION_IDENTITY_BYTES = 128 * 1024
MAX_PARENT_USD_CLI_SERVER_STATE_BYTES = 64 * 1024
MAX_USD_CLI_RECEIPT_CHECKPOINT_BYTES = 64 * 1024
MAX_USD_CLI_LAUNCHER_BYTES = 1024 * 1024
ASSET_USD_CLI_PROVIDER_FREE_READINESS_SCHEMA_VERSION: Final = (
    "content-workflow-cli.asset-usd-cli-provider-free-readiness.v1"
)
_OVRTX_PROVISIONING_TIMEOUT_SECONDS = 900.0
_OVRTX_PROVISIONING_POLL_SECONDS = 5.0
_OVRTX_PROVISIONING_MESSAGES = (
    "ovrtx auto-install STARTED in the background",
    "ovrtx auto-install in progress",
)


def _pin_uv_executable_capability(environment: dict[str, str]) -> None:
    """Preserve a verified uv path after usd-cli PATH sanitization."""

    if environment.get("USD_CLI_UV_EXECUTABLE"):
        return
    uv_executable = shutil.which("uv", path=os.environ.get("PATH", os.defpath))
    if uv_executable:
        environment["USD_CLI_UV_EXECUTABLE"] = str(
            Path(uv_executable).resolve(strict=True)
        )


def parent_usd_cli_daemon_identity_sha256(
    *,
    pid: int,
    process_start_token: str,
    project_id: str,
    instance_id: str | None,
    process_group_id: int | None,
    os_session_id: int | None,
) -> str:
    """Digest the exact daemon identity fields available at a boundary."""

    payload = json.dumps(
        {
            "instance_id": instance_id,
            "os_session_id": os_session_id,
            "pid": pid,
            "process_group_id": process_group_id,
            "process_start_token": process_start_token,
            "project_id": project_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _linux_process_identity(pid: int) -> tuple[str, int, int] | None:
    """Return the platform birth token and supervision identity for a process."""

    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        return None
    if os.name == "nt":
        token = windows_process_start_token(pid)
        return (token, pid, pid) if token is not None else None
    try:
        fields = (
            Path(f"/proc/{pid}/stat")
            .read_text(encoding="utf-8")
            .rpartition(")")[2]
            .split()
        )
        return f"t{fields[19]}", int(fields[2]), int(fields[3])
    except (OSError, IndexError, ValueError):
        return None


class ParentUsdCliArtifactIdentity(BaseModel):
    """One immutable regular-file identity named by the parent handoff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ParentUsdCliStagedSourceIdentity(BaseModel):
    """Run-confined source closure the child is authorized to inspect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_source: ParentUsdCliArtifactIdentity
    staged_source: ParentUsdCliArtifactIdentity
    staging_manifest: ParentUsdCliArtifactIdentity
    dependency_digest_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ParentUsdCliGeneratedSourceIdentity(BaseModel):
    """Frozen source-free CAD intent available before a USD exists."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_mode: Literal["cad_modeling"] = "cad_modeling"
    source_intent: ParentUsdCliArtifactIdentity
    source_images: list[ParentUsdCliArtifactIdentity] = Field(
        default_factory=list,
        max_length=32,
    )


class ParentUsdCliSessionIdentity(BaseModel):
    """Credential-free identity for one parent-owned run daemon.

    This document is discovery evidence, never daemon lifecycle authority. The
    child and deterministic domain leaves reuse the daemon through isolated
    named sessions;
    supported CLI/session surfaces refuse child lifecycle changes, while the
    parent retains the authenticated process identity and verifies teardown.
    This same-user contract is not an operating-system security boundary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agent-workflows.parent-usd-cli-session.v1"] = (
        PARENT_USD_CLI_SESSION_SCHEMA_VERSION
    )
    created_at: str
    launch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    workflow: str = Field(
        default="asset.run",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    run_id: str = Field(min_length=1)
    run_dir: str
    repository_root: str
    parent_session_id: str = Field(min_length=1)
    project_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    instance_id: str = Field(min_length=1, max_length=256)
    daemon_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    server_host: Literal["127.0.0.1"] = "127.0.0.1"
    server_port: int = Field(ge=1, le=65535)
    allowed_roots: list[str] = Field(min_length=1, max_length=32)
    # Source provenance belongs to the workflow request. Legacy asset/scene
    # handoffs keep it here for compatibility; the shared renderer capability
    # intentionally does not require every domain to invent a source shape.
    source: (
        ParentUsdCliStagedSourceIdentity | ParentUsdCliGeneratedSourceIdentity | None
    ) = None
    readiness_artifact: ParentUsdCliArtifactIdentity
    launcher_implementation: ParentUsdCliArtifactIdentity
    usd_cli_version: str = Field(min_length=1)
    usd_cli_source_revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    usd_cli_wrapper: str
    usd_cli_executable: str
    renderer_credentials: Literal["parent_confined"] = "parent_confined"
    lifecycle_authority: Literal["parent_only"] = "parent_only"
    child_daemon_policy: Literal["reuse_only_no_autostart"] = "reuse_only_no_autostart"

    @model_validator(mode="after")
    def validate_run_confinement(self) -> ParentUsdCliSessionIdentity:
        try:
            run_root = Path(self.run_dir).expanduser().resolve(strict=True)
            # Existence is required here; exact route equality is re-attested by
            # WorkflowUsdCliSession.create immediately before child use.
            Path(self.repository_root).expanduser().resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                "parent usd-cli identity names an unavailable root"
            ) from exc
        if not self.allowed_roots or len(set(self.allowed_roots)) != len(
            self.allowed_roots
        ):
            raise ValueError("parent usd-cli identity has invalid allowed roots")
        resolved_allowed_roots: list[Path] = []
        for raw_root in self.allowed_roots:
            try:
                allowed_root = Path(raw_root).expanduser().resolve(strict=True)
            except OSError as exc:
                raise ValueError(
                    "parent usd-cli identity names an unavailable allowed root"
                ) from exc
            if str(allowed_root) != raw_root:
                raise ValueError("parent usd-cli allowed roots must be canonical")
            resolved_allowed_roots.append(allowed_root)
        if run_root not in resolved_allowed_roots:
            raise ValueError("parent usd-cli identity must allow the run root")
        run_artifacts: tuple[tuple[str, ParentUsdCliArtifactIdentity], ...] = ()
        external_artifacts: tuple[tuple[str, ParentUsdCliArtifactIdentity], ...] = ()
        if isinstance(self.source, ParentUsdCliStagedSourceIdentity):
            run_artifacts = (
                ("staged source", self.source.staged_source),
                ("staging manifest", self.source.staging_manifest),
            )
        elif isinstance(self.source, ParentUsdCliGeneratedSourceIdentity):
            run_artifacts = (("source intent", self.source.source_intent),)
            external_artifacts = tuple(
                (f"source image {index}", artifact)
                for index, artifact in enumerate(self.source.source_images, start=1)
            )
        for label, artifact in (
            *run_artifacts,
            ("readiness artifact", self.readiness_artifact),
        ):
            try:
                observed = read_contained_artifact(run_root, artifact.path)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"{label} is not an immutable parent-run artifact"
                ) from exc
            if (
                observed.sha256 != artifact.sha256
                or observed.size_bytes != artifact.size_bytes
            ):
                raise ValueError(f"{label} bytes changed")
        for label, artifact in external_artifacts:
            image_path = Path(artifact.path).expanduser()
            if not image_path.is_absolute():
                raise ValueError(f"{label} path must be absolute")
            try:
                observed = read_contained_artifact(image_path.parent, image_path)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"{label} is not an immutable source artifact"
                ) from exc
            if (
                observed.sha256 != artifact.sha256
                or observed.size_bytes != artifact.size_bytes
            ):
                raise ValueError(f"{label} bytes changed")
        try:
            launcher = (
                Path(self.launcher_implementation.path)
                .expanduser()
                .resolve(strict=True)
            )
            wrapper = Path(self.usd_cli_wrapper).expanduser().resolve(strict=True)
            executable = Path(self.usd_cli_executable).expanduser().resolve(strict=True)
        except OSError as exc:
            raise ValueError("parent usd-cli implementation is unavailable") from exc
        if not wrapper.is_file() or not executable.is_file():
            raise ValueError("parent usd-cli implementation is not a regular file")
        launcher_metadata = launcher.stat()
        if (
            not launcher.is_file()
            or launcher_metadata.st_size != self.launcher_implementation.size_bytes
            or file_sha256(launcher) != self.launcher_implementation.sha256
        ):
            raise ValueError("launcher implementation bytes changed")
        return self


@dataclass(frozen=True)
class ParentUsdCliConnection:
    """Authenticated route to one attested parent-owned daemon session."""

    server_url: str
    session_id: str
    authentication_token: SecretStr = field(repr=False, compare=False)


def _verify_live_parent_usd_cli_daemon(
    identity: ParentUsdCliSessionIdentity,
    require_os_identity: bool = True,
) -> ParentUsdCliConnection:
    """Authenticate the live daemon bound by one parent handoff.

    The launcher verifies the daemon's host OS identity before creating a child.
    A sandboxed child may live in a different PID namespace, so its recheck uses
    the authenticated loopback health identity without consulting local ``/proc``.
    """

    run_root = Path(identity.run_dir).expanduser().resolve(strict=True)
    try:
        observed = read_contained_artifact(
            run_root,
            run_root / ".usd-cli" / "server.json",
            max_bytes=MAX_PARENT_USD_CLI_SERVER_STATE_BYTES,
            capture_bytes=True,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError("parent usd-cli daemon state is unavailable") from exc
    assert observed.data is not None
    try:
        state = json.loads(observed.data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("parent usd-cli daemon state is invalid") from exc
    if not isinstance(state, dict):
        raise RuntimeError("parent usd-cli daemon state is not a JSON object")
    pid = state.get("pid")
    process_start_token = state.get("process_start_token")
    project_id = state.get("project_id")
    instance_id = state.get("instance_id")
    token = state.get("token")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or not isinstance(process_start_token, str)
        or project_id != identity.project_id
        or instance_id != identity.instance_id
        or state.get("lifecycle_owner") != "external"
        or state.get("host") != identity.server_host
        or state.get("port") != identity.server_port
        or not isinstance(token, str)
        or not token
    ):
        raise RuntimeError("parent usd-cli daemon state disagrees with its handoff")
    if require_os_identity:
        os_identity = _linux_process_identity(pid)
        if os_identity is None or os_identity[0] != process_start_token:
            raise RuntimeError("parent usd-cli daemon process identity is not live")
        observed_identity_sha256 = parent_usd_cli_daemon_identity_sha256(
            pid=pid,
            process_start_token=process_start_token,
            project_id=project_id,
            instance_id=instance_id,
            process_group_id=os_identity[1],
            os_session_id=os_identity[2],
        )
        if observed_identity_sha256 != identity.daemon_identity_sha256:
            raise RuntimeError("parent usd-cli daemon OS identity changed")

    headers = {
        "x-usd-cli-token": token,
        "x-ov-token": token,
        "x-3dsc-token": token,
    }
    response_status: int
    if os.environ.get(PARENT_USD_CLI_PROXY_ALLOWED_ENV) == "1":
        proxy = _attached_parent_http_proxy(identity)
        if proxy is None:
            raise RuntimeError("parent usd-cli daemon proxy route is unavailable")
        import httpx

        try:
            with httpx.Client(proxy=proxy, trust_env=False) as client:
                with client.stream(
                    "GET",
                    f"http://{identity.server_host}:{identity.server_port}/health",
                    headers=headers,
                    timeout=2.0,
                ) as proxy_response:
                    response_status = proxy_response.status_code
                    bounded_payload = bytearray()
                    for chunk in proxy_response.iter_bytes():
                        remaining = (
                            MAX_PARENT_USD_CLI_SERVER_STATE_BYTES
                            + 1
                            - len(bounded_payload)
                        )
                        if remaining <= 0:
                            break
                        bounded_payload.extend(chunk[:remaining])
                    payload = bytes(bounded_payload)
        except Exception as exc:  # noqa: BLE001 - transport failures fail closed
            raise RuntimeError("parent usd-cli daemon health is unavailable") from exc
    else:
        connection = http.client.HTTPConnection(
            identity.server_host,
            identity.server_port,
            timeout=2.0,
        )
        try:
            connection.request("GET", "/health", headers=headers)
            response = connection.getresponse()
            response_status = response.status
            payload = response.read(MAX_PARENT_USD_CLI_SERVER_STATE_BYTES + 1)
        except (OSError, http.client.HTTPException) as exc:
            raise RuntimeError("parent usd-cli daemon health is unavailable") from exc
        finally:
            connection.close()
    if len(payload) > MAX_PARENT_USD_CLI_SERVER_STATE_BYTES:
        raise RuntimeError("parent usd-cli daemon health exceeds its size limit")
    try:
        health = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("parent usd-cli daemon health is invalid") from exc
    if not isinstance(health, dict):
        raise RuntimeError("parent usd-cli daemon health is not a JSON object")
    sessions = health.get("sessions")
    if (
        response_status != 200
        or health.get("ok") is not True
        or any(
            health.get(field) != state.get(field)
            for field in (
                "pid",
                "process_start_token",
                "project_id",
                "instance_id",
                "lifecycle_owner",
            )
        )
        or not isinstance(sessions, dict)
        or identity.parent_session_id not in sessions
    ):
        raise RuntimeError("parent usd-cli daemon health identity changed")
    return ParentUsdCliConnection(
        server_url=f"http://{identity.server_host}:{identity.server_port}",
        session_id=identity.parent_session_id,
        authentication_token=SecretStr(token),
    )


def _attached_parent_http_proxy(
    identity: ParentUsdCliSessionIdentity,
) -> str | None:
    """Validate the SDK proxy for this exact attached parent session."""

    if os.environ.get(PARENT_USD_CLI_PROXY_ALLOWED_ENV) != "1":
        return None
    attached_project = os.environ.get("USD_CLI_ATTACHED_PROJECT_DIR")
    if not attached_project:
        return None
    try:
        resolved_project = Path(attached_project).resolve(strict=True)
        parent_run = Path(identity.run_dir).resolve(strict=True)
    except OSError:
        return None
    if (
        str(resolved_project) != attached_project
        or not resolved_project.is_dir()
        or resolved_project != parent_run
    ):
        return None

    raw_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if not raw_proxy:
        return None
    try:
        parsed = urlparse(raw_proxy)
        host = parsed.hostname
        port = parsed.port
        address = (
            None
            if host == "localhost"
            else ipaddress.ip_address(host)
            if host is not None
            else None
        )
    except ValueError:
        return None
    if (
        parsed.scheme != "http"
        or (host != "localhost" and (address is None or not address.is_loopback))
        or port is None
        or not 1 <= port <= 65535
        # Claude's SDK loopback conduit authenticates as
        # srt:<opaque-token>. Newer Claude CLI sandboxes scope that username as
        # srt.<opaque-context>:<opaque-token>. Credentials remain forbidden for
        # every other proxy namespace and must always include a password.
        or (
            (parsed.username is not None or parsed.password is not None)
            and (
                not parsed.password
                or (
                    parsed.username != "srt"
                    and not (
                        parsed.username is not None
                        and parsed.username.startswith("srt.")
                        and len(parsed.username) > len("srt.")
                    )
                )
            )
        )
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return None
    return raw_proxy


verify_live_parent_usd_cli_daemon = _verify_live_parent_usd_cli_daemon


def _parent_usd_cli_session_identity_from_environment() -> (
    tuple[ParentUsdCliSessionIdentity, Path] | None
):
    """Load the exact credential-free parent identity passed by the launcher."""

    raw_path = os.environ.get(PARENT_USD_CLI_SESSION_IDENTITY_ENV)
    expected_sha256 = os.environ.get(PARENT_USD_CLI_SESSION_IDENTITY_SHA256_ENV)
    if raw_path is None and expected_sha256 is None:
        return None
    if not raw_path or not expected_sha256:
        raise RuntimeError("parent usd-cli session identity environment is incomplete")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise RuntimeError("parent usd-cli session identity digest is invalid")
    identity_path = Path(raw_path).expanduser()
    if not identity_path.is_absolute():
        raise RuntimeError("parent usd-cli session identity path must be absolute")
    if identity_path.parent.name != "raw":
        raise RuntimeError("parent usd-cli session identity path is not canonical")
    inferred_run_root = identity_path.parent.parent
    try:
        observed = read_contained_artifact(
            inferred_run_root,
            identity_path,
            max_bytes=MAX_PARENT_USD_CLI_SESSION_IDENTITY_BYTES,
            capture_bytes=True,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "parent usd-cli session identity must be a run-confined regular file"
        ) from exc
    if observed.sha256 != expected_sha256:
        raise RuntimeError("parent usd-cli session identity digest changed")
    assert observed.data is not None
    payload = observed.data
    try:
        identity = _validated_parent_usd_cli_session_identity(
            payload,
            _parent_usd_cli_identity_artifact_fingerprint(payload),
        )
    except ValidationError as exc:
        raise RuntimeError(
            f"parent usd-cli session identity is invalid: {exc}"
        ) from exc
    expected_names = {
        f"asset_usd_cli_session_{identity.launch_id}.json",
        f"parent_usd_cli_session_{identity.launch_id}.json",
    }
    if identity_path.name not in expected_names or identity_path.parent.resolve(
        strict=True
    ) != (Path(identity.run_dir) / "raw").resolve(strict=True):
        raise RuntimeError("parent usd-cli session identity path is not canonical")
    return identity, identity_path.resolve(strict=True)


def _parent_usd_cli_identity_artifact_fingerprint(
    payload: bytes,
) -> tuple[tuple[object, ...], ...]:
    """Return cheap invalidation metadata for expensive artifact attestations."""

    try:
        raw = json.loads(payload)
        source = raw.get("source")
        if source is None:
            source_paths: tuple[object, ...] = ()
        elif "staged_source" in source:
            source_paths = (
                source["staged_source"]["path"],
                source["staging_manifest"]["path"],
            )
        else:
            source_images = source["source_images"]
            if not isinstance(source_images, list):
                return ()
            source_paths = (
                source["source_intent"]["path"],
                *(image["path"] for image in source_images if isinstance(image, dict)),
            )
        raw_paths = (
            *source_paths,
            raw["readiness_artifact"]["path"],
            raw["launcher_implementation"]["path"],
        )
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return ()
    fingerprint: list[tuple[object, ...]] = []
    for raw_path in raw_paths:
        try:
            path = Path(raw_path).expanduser()
            metadata = path.stat(follow_symlinks=False)
        except (OSError, TypeError):
            fingerprint.append((repr(raw_path), None))
            continue
        fingerprint.append(
            (
                str(path),
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
        )
    return tuple(fingerprint)


@lru_cache(maxsize=8)
def _validated_parent_usd_cli_session_identity(
    payload: bytes,
    artifact_fingerprint: tuple[tuple[object, ...], ...],
) -> ParentUsdCliSessionIdentity:
    """Validate immutable handoff bytes once per unchanged child-process view.

    Large staged sources are expensive to re-hash for every domain session. The
    identity bytes and no-follow artifact metadata invalidate this bounded cache;
    the parent independently re-attests all source bytes after child access ends.
    """

    del artifact_fingerprint  # Its value participates in the cache key.
    return ParentUsdCliSessionIdentity.model_validate_json(payload)


def validated_ovrtx_render_metadata(
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """Return complete executed OVRTX facts or reject the render envelope."""

    summary = response.get("summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError("usd-cli render omitted its execution summary")
    backend = summary.get("backend")
    render_mode = summary.get("ovrtx_render_mode")
    sensor_updates = summary.get("ovrtx_num_sensor_updates")
    active_aov = summary.get("active_aov")
    if (
        backend not in {"ovrtx", "remote"}
        or not isinstance(render_mode, str)
        or not render_mode
        or not isinstance(sensor_updates, int)
        or isinstance(sensor_updates, bool)
        or sensor_updates < 1
        or not isinstance(active_aov, str)
        or not active_aov
    ):
        raise RuntimeError("usd-cli render omitted complete executed OVRTX metadata")

    renderer_identity: dict[str, Any] | None = None
    data = response.get("data")
    results = data.get("results") if isinstance(data, Mapping) else None
    if isinstance(results, list) and len(results) == 1:
        result = results[0]
        candidate = (
            result.get("renderer_identity") if isinstance(result, Mapping) else None
        )
        if isinstance(candidate, Mapping):
            renderer_identity = dict(candidate)

    if backend == "remote":
        identity = renderer_identity or {}
        endpoint = identity.get("endpoint")
        engine = identity.get("engine")
        protocol_version = identity.get("protocol_version")
        status = identity.get("status")
        parsed_endpoint = urlparse(endpoint) if isinstance(endpoint, str) else None
        if (
            parsed_endpoint is None
            or parsed_endpoint.scheme not in {"http", "https"}
            or not parsed_endpoint.netloc
            or engine != "ovrtx"
            or not isinstance(protocol_version, int)
            or isinstance(protocol_version, bool)
            or protocol_version < 1
            or not isinstance(status, str)
            or not status
        ):
            raise RuntimeError(
                "usd-cli remote render omitted its exact OVRTX service identity"
            )

    return {
        "backend": backend,
        "transport": "remote" if backend == "remote" else "local",
        "ovrtx_render_mode": render_mode,
        "ovrtx_num_sensor_updates": sensor_updates,
        "active_aov": active_aov,
        "renderer_identity": renderer_identity,
    }


def _redact_arguments(arguments: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for argument in arguments:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if argument.startswith("-"):
            flag, separator, _value = argument.partition("=")
            if _SECRET_ARGUMENT_PATTERN.search(flag):
                if separator:
                    redacted.append(f"{flag}=<redacted>")
                else:
                    redacted.append(flag)
                    redact_next = True
                continue
        redacted.append(argument)
    return redacted


def _secret_argument_values(arguments: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    redact_next = False
    for argument in arguments:
        if redact_next:
            if argument:
                values.append(argument)
            redact_next = False
            continue
        if not argument.startswith("-"):
            continue
        flag, separator, value = argument.partition("=")
        if not _SECRET_ARGUMENT_PATTERN.search(flag):
            continue
        if separator:
            if value:
                values.append(value)
        else:
            redact_next = True
    return tuple(dict.fromkeys(values))


def _redact_text(value: str, secrets_to_redact: tuple[str, ...]) -> str:
    redacted = value
    for secret in secrets_to_redact:
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _redact_value(value: Any, secrets_to_redact: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, secrets_to_redact)
    if isinstance(value, dict):
        return {
            str(key): _redact_value(item, secrets_to_redact)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, secrets_to_redact) for item in value]
    return value


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def inherited_project_render_config(project_dir: Path) -> dict[str, Any]:
    """Return only the nearest caller project's explicit ``[render]`` table.

    This must run before a workflow creates its nearer ``.usd-cli`` state
    directory.  Global defaults and environment overrides remain owned by
    usd-cli; server/session state is deliberately never inherited.
    """

    from usd_core.config import CONFIG_NAME, _read_toml, find_project_dir, state_dir_for

    inherited_project = find_project_dir(project_dir)
    if inherited_project is None:
        return {}
    inherited_config = _read_toml(state_dir_for(inherited_project) / CONFIG_NAME)
    render_section = inherited_config.get("render")
    return dict(render_section) if isinstance(render_section, dict) else {}


def _environment_remote_render_config() -> dict[str, Any]:
    """Translate the documented workflow renderer environment into usd-cli.

    The endpoint is durable configuration, while an NGC bearer token is
    returned as ``remote_api_key`` so :func:`_separate_render_credentials`
    keeps it out of the child-visible TOML and injects it only into the trusted
    usd-cli process. Automatic NGC credential use is limited to canonical HTTPS
    NVCF invocation hosts.
    """

    endpoint_value = (os.environ.get("RENDER_ENDPOINT") or "").strip()
    function_value = (os.environ.get("NVCF_RENDER_FUNCTION_ID") or "").strip()
    raw_value = endpoint_value or function_value
    if not raw_value:
        return {}
    source_name = "RENDER_ENDPOINT" if endpoint_value else "NVCF_RENDER_FUNCTION_ID"
    try:
        endpoint = resolve_endpoint_or_function_id(raw_value).rstrip("/")
        parsed = urlparse(endpoint)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{source_name} does not identify a valid renderer") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            f"{source_name} must identify an HTTP(S) renderer without userinfo, "
            "query parameters, or a fragment"
        )

    render: dict[str, Any] = {
        "renderer": "remote",
        "remote_url": endpoint,
        # These explicit empty values shadow any global usd-cli credential or
        # backend pool when the project config is deep-merged at load time.
        "remote_api_key": "",
        "backends": [],
    }
    hostname = parsed.hostname.lower().rstrip(".")
    nvcf_suffix = NVCF_INVOCATION_HOST.lower()
    canonical_nvcf = (
        parsed.scheme.lower() == "https"
        and port in {None, 443}
        and hostname.endswith(nvcf_suffix)
        and len(hostname) > len(nvcf_suffix)
        and parsed.path in {"", "/"}
    )
    if canonical_nvcf:
        api_key = (os.environ.get("NGC_API_KEY") or "").strip()
        if api_key:
            render["remote_api_key"] = api_key
    return render


def _separate_render_credentials(
    render_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return child-visible render config and parent-only renderer credentials."""

    sanitized = dict(render_config)
    credentials: dict[str, Any] = {}
    has_remote_api_key = "remote_api_key" in sanitized
    remote_api_key = sanitized.pop("remote_api_key", None)
    if remote_api_key:
        credentials["remote_api_key"] = str(remote_api_key)
    if has_remote_api_key:
        # Persist an empty shadow while the real credential remains available
        # only to the parent-owned usd-cli process environment.
        sanitized["remote_api_key"] = ""
    backends = sanitized.get("backends")
    if isinstance(backends, list):
        sanitized_backends: list[Any] = []
        backend_api_keys: dict[str, str] = {}
        for raw_entry in backends:
            if not isinstance(raw_entry, dict):
                sanitized_backends.append(raw_entry)
                continue
            entry = dict(raw_entry)
            api_key = entry.pop("api_key", None)
            url = str(entry.get("url", "") or "").rstrip("/")
            if url and api_key and url not in backend_api_keys:
                backend_api_keys[url] = str(api_key)
            sanitized_backends.append(entry)
        sanitized["backends"] = sanitized_backends
        if backend_api_keys:
            credentials["backend_api_keys"] = backend_api_keys
    return sanitized, credentials


def _option_values(arguments: list[str], *names: str) -> tuple[str, ...]:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        for name in names:
            if argument == name:
                if index + 1 >= len(arguments):
                    raise RuntimeError(f"usd-cli option {name} requires a value")
                values.append(arguments[index + 1])
                break
            prefix = f"{name}="
            if argument.startswith(prefix):
                values.append(argument[len(prefix) :])
                break
    return tuple(values)


def _command_output_values(arguments: list[str]) -> tuple[str, ...]:
    if not arguments:
        raise RuntimeError("usd-cli command arguments cannot be empty")
    values = list(_option_values(arguments, "--output", "-o", "--output-dir"))
    if any(not value for value in values):
        raise RuntimeError("usd-cli output options require non-empty paths")
    command = arguments[0]
    if command == "save":
        positional = arguments[1] if len(arguments) > 1 else None
        if not positional or positional.startswith("-"):
            raise RuntimeError(
                "workflow-confined usd-cli save requires an explicit output path"
            )
        values.append(positional)
    elif command == "export":
        positional = arguments[2] if len(arguments) > 2 else None
        if not positional or positional.startswith("-"):
            raise RuntimeError(
                "workflow-confined usd-cli export requires an explicit output path"
            )
        values.append(positional)
    elif command == "convert":
        positional = arguments[2] if len(arguments) > 2 else None
        if not positional or positional.startswith("-"):
            raise RuntimeError(
                "workflow-confined usd-cli convert requires an explicit output path"
            )
        values.append(positional)
    return tuple(dict.fromkeys(values))


def _confined_output_paths(
    project_dir: Path,
    arguments: list[str],
) -> tuple[Path, ...]:
    project = project_dir.resolve(strict=True)
    outputs: list[Path] = []
    for raw_value in _command_output_values(arguments):
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = project / candidate
        try:
            resolved = candidate.resolve(strict=False)
            relative = resolved.relative_to(project)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"usd-cli output escapes workflow project: {candidate}"
            ) from exc
        current = project
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                raise RuntimeError(
                    f"usd-cli output contains a symlink component: {current}"
                )
        outputs.append(resolved)
    return tuple(outputs)


def _absolute_confined_output_arguments(
    arguments: list[str],
    output_paths: tuple[Path, ...],
) -> list[str]:
    """Bind relative output arguments to their domain project before dispatch.

    An attached session's daemon resolves request paths against the parent run
    root, not the child process cwd. The caller already confined each output to
    the domain project; replace only those output values with the exact absolute
    paths that were checked.
    """

    raw_outputs = _command_output_values(arguments)
    replacements = {
        raw: str(path) for raw, path in zip(raw_outputs, output_paths, strict=True)
    }
    resolved = list(arguments)
    option_names = ("--output", "-o", "--output-dir")
    for index, argument in enumerate(arguments):
        for name in option_names:
            if argument == name and index + 1 < len(arguments):
                resolved[index + 1] = replacements[arguments[index + 1]]
                break
            prefix = f"{name}="
            if argument.startswith(prefix):
                resolved[index] = f"{prefix}{replacements[argument[len(prefix) :]]}"
                break
    command = arguments[0] if arguments else ""
    positional_index = 1 if command == "save" else 2
    if command in {"save", "export", "convert"}:
        resolved[positional_index] = replacements[arguments[positional_index]]
    return resolved


def _resolved_input_argument_paths(
    project_dir: Path,
    arguments: list[str],
    *,
    output_paths: tuple[Path, ...],
) -> tuple[tuple[int, str | None, Path, bool], ...]:
    """Resolve input files, with schema-known paths eligible for rewriting.

    A generic "argument happens to name an existing file" heuristic is unsafe:
    prim names, material names, and other scalar operands can collide with files
    in the workflow directory. The table below names the usd-cli arguments whose
    public command schema actually treats them as files. Other existing files are
    retained as receipt evidence for compatibility but never mutate argv.
    """

    raw_outputs = set(_command_output_values(arguments))
    if not arguments:
        return ()
    command = arguments[0]
    command_key = (
        f"{command}.{arguments[1]}"
        if command == "physics" and len(arguments) > 1
        else command
    )
    positional_indices: dict[str, tuple[int, ...]] = {
        "convert": (1,),
        "import": (1,),
        "open": (1,),
        "verify": (1,),
    }
    option_names: dict[str, tuple[tuple[str, bool], ...]] = {
        "material": (
            ("--diffuse-tex", False),
            ("--library", True),
            ("--mdl", False),
            ("--metallic-tex", False),
            ("--normal-tex", False),
            ("--orm-tex", False),
            ("--roughness-tex", False),
        ),
        "physics.apply": (("--file", True), ("-f", True)),
        "physics.simulate": (("--scene", True),),
        "render": (("--against", True),),
        "render-frames": (("--scene", True),),
    }
    # ``import`` and Material texture/MDL operands are authored as Sdf asset paths.
    # Preserve their portable spelling while still resolving and hashing any real
    # run-confined file for the command receipt. The daemon validates the original
    # spelling against its parent run root; final saved-stage and packaging checks
    # remain responsible for proving those authored dependencies resolve.
    preserve_relative = command_key == "import"
    candidates: list[tuple[int, str | None, str, bool]] = []
    for index in positional_indices.get(command_key, ()):
        if index < len(arguments):
            candidates.append((index, None, arguments[index], not preserve_relative))
    for schema_option_name, absolutize in option_names.get(command_key, ()):
        prefix = f"{schema_option_name}="
        for index, argument in enumerate(arguments[1:], start=1):
            if argument == schema_option_name and index + 1 < len(arguments):
                candidates.append((index + 1, None, arguments[index + 1], absolutize))
            elif argument.startswith(prefix):
                candidates.append(
                    (
                        index,
                        schema_option_name,
                        argument[len(prefix) :],
                        absolutize,
                    )
                )

    schema_indices = {candidate[0] for candidate in candidates}
    for index, argument in enumerate(arguments[1:], start=1):
        if index in schema_indices:
            continue
        option_name: str | None = None
        candidate_value = argument
        if argument.startswith("-") and "=" in argument:
            option_name, _separator, candidate_value = argument.partition("=")
        candidates.append((index, option_name, candidate_value, False))

    resolved_inputs: list[tuple[int, str | None, Path, bool]] = []
    seen_indices: set[int] = set()
    for index, option_name, candidate_value, absolutize in candidates:
        if index in seen_indices:
            continue
        seen_indices.add(index)
        if not candidate_value or candidate_value in raw_outputs:
            continue
        candidate = Path(candidate_value).expanduser()
        if not candidate.is_absolute():
            candidate = project_dir / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved in output_paths or not resolved.is_file():
            continue
        resolved_inputs.append((index, option_name, resolved, absolutize))
    return tuple(sorted(resolved_inputs, key=lambda item: item[0]))


def _absolute_attached_session_arguments(
    project_dir: Path,
    arguments: list[str],
    output_paths: tuple[Path, ...],
) -> list[str]:
    """Bind every detected relative file path to the domain project."""

    resolved = _absolute_confined_output_arguments(arguments, output_paths)
    for index, option_name, path, absolutize in _resolved_input_argument_paths(
        project_dir,
        arguments,
        output_paths=output_paths,
    ):
        if not absolutize:
            continue
        resolved[index] = (
            f"{option_name}={path}" if option_name is not None else str(path)
        )
    return resolved


def _referenced_input_files(
    project_dir: Path,
    arguments: list[str],
    *,
    output_paths: tuple[Path, ...],
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for _index, _option_name, resolved, _absolutize in _resolved_input_argument_paths(
        project_dir,
        arguments,
        output_paths=output_paths,
    ):
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            sha256, size_bytes = _file_digest(resolved)
        except OSError:
            continue
        inputs.append(
            {
                "path": str(resolved),
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    return sorted(inputs, key=lambda item: str(item["path"]))


def _contained_response_artifacts(
    project_dir: Path,
    response: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if response is None:
        return []
    artifacts: list[dict[str, Any]] = []
    seen: set[Path] = set()
    pending: list[Any] = [response]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
            continue
        if isinstance(value, list):
            pending.extend(value)
            continue
        if not isinstance(value, str):
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = project_dir / candidate
        try:
            if candidate.is_symlink():
                continue
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(project_dir.resolve(strict=True))
        except (OSError, ValueError):
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            sha256, size_bytes = _file_digest(resolved)
        except OSError:
            continue
        artifacts.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    return sorted(artifacts, key=lambda artifact: str(artifact["path"]))


def stage_up_axis_is_y(scene_path: Path) -> bool:
    """Read the factual stage up axis used by usd-cli camera orbiting."""

    from pxr import Usd, UsdGeom

    path = scene_path.expanduser().resolve(strict=True)
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage for camera orientation: {path}")
    return UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y


def direction_angles(
    direction: str,
    *,
    up_axis_y: bool = False,
) -> tuple[float, float]:
    """Convert a signed axis direction into usd-cli orbit angles."""

    components = dict.fromkeys("xyz", 0.0)
    position = 0
    for match in re.finditer(r"([+-](?:\d+(?:\.\d*)?)?)([xyz])", direction):
        if match.start() != position:
            raise ValueError(f"Invalid render direction: {direction}")
        raw = match.group(1)
        components[match.group(2)] = (
            float(raw) if raw not in {"+", "-"} else 1.0 if raw == "+" else -1.0
        )
        position = match.end()
    length = math.sqrt(sum(value * value for value in components.values()))
    if position != len(direction) or length == 0:
        raise ValueError(f"Invalid render direction: {direction}")
    if up_axis_y:
        return (
            math.degrees(math.atan2(components["x"], components["z"])),
            math.degrees(math.asin(components["y"] / length)),
        )
    return (
        math.degrees(math.atan2(components["y"], components["x"])),
        math.degrees(math.asin(components["z"] / length)),
    )


@dataclass(frozen=True, slots=True)
class WorkflowUsdCliCommandResult:
    """One command response paired with its process-local exit status."""

    payload: dict[str, Any]
    returncode: int


@dataclass(frozen=True, slots=True)
class _UsdCliLauncherIdentity:
    """Exact bytes of one launcher admitted with a package route."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class WorkflowUsdCliPackageRoutePin:
    """One immutable package route reused for a workflow session lifetime."""

    repository_root: Path
    route: UsdCliPackageRoute
    wrapper: _UsdCliLauncherIdentity
    target: _UsdCliLauncherIdentity


def _pin_usd_cli_package_route(repository_root: Path) -> WorkflowUsdCliPackageRoutePin:
    """Resolve and bind one package route without creating workflow state."""

    root = repository_root.expanduser().resolve()
    route = resolve_package_owned_usd_cli_route(root)

    def launcher_identity(path: Path) -> _UsdCliLauncherIdentity:
        resolved = path.expanduser().resolve(strict=True)
        observed = read_contained_artifact(
            resolved.parent,
            resolved,
            max_bytes=MAX_USD_CLI_LAUNCHER_BYTES,
        )
        return _UsdCliLauncherIdentity(
            path=observed.path,
            sha256=observed.sha256,
            size_bytes=observed.size_bytes,
        )

    return WorkflowUsdCliPackageRoutePin(
        repository_root=root,
        route=route,
        wrapper=launcher_identity(route.wrapper),
        target=launcher_identity(route.target),
    )


@dataclass(frozen=True, slots=True)
class WorkflowUsdCliSession:
    """Trusted project, launcher route, and named session for one workflow scope.

    ``frozen=True`` pins the public security identity for the session lifetime; it
    does not make command execution pure or promise deep immutability.  Receipt
    inode/digest tracking is intentionally private mutable operational state and is
    excluded from construction, representation, equality, and hashing.
    """

    project_dir: Path
    session_id: str
    route: UsdCliPackageRoute
    workflow: str
    daemon_project_dir: Path | None = None
    daemon_server_url: str | None = None
    receipt_checkpoint_sha256: str | None = None
    _force_reload_on_every_open: bool = field(
        default=False,
        repr=False,
        compare=False,
    )
    raw_directory_identity: tuple[int, int] | None = None
    _render_credentials: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _parent_readiness_artifact: ParentUsdCliArtifactIdentity | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _package_route_pin: WorkflowUsdCliPackageRoutePin | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _receipt_state: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def _open_raw_directory(self) -> tuple[Path, int]:
        """Open the workflow raw directory without following symlinks."""

        project = self.project_dir.resolve(strict=True)
        if os.name == "nt":
            raw_directory = project / "raw"
            with open_confined_directory(raw_directory, create=True):
                pass
            raw_stat = raw_directory.stat(follow_symlinks=False)
            if not stat.S_ISDIR(raw_stat.st_mode):
                raise RuntimeError(
                    "workflow raw artifact directory must be a real directory"
                )
            expected_identity = self._receipt_state.get("raw_identity")
            if expected_identity is None:
                expected_identity = self.raw_directory_identity
            observed_identity = (raw_stat.st_dev, raw_stat.st_ino)
            if expected_identity is not None and expected_identity != observed_identity:
                raise RuntimeError(
                    "workflow raw artifact directory changed during the session"
                )
            self._receipt_state.setdefault("raw_identity", observed_identity)
            return raw_directory, -1
        directory_flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        project_descriptor = os.open(project, directory_flags)
        created = False
        try:
            try:
                os.mkdir("raw", mode=0o700, dir_fd=project_descriptor)
                created = True
            except FileExistsError:
                pass
            raw_descriptor = os.open(
                "raw",
                directory_flags,
                dir_fd=project_descriptor,
            )
        finally:
            os.close(project_descriptor)
        raw_stat = os.fstat(raw_descriptor)
        expected_identity = self._receipt_state.get("raw_identity")
        if expected_identity is None:
            expected_identity = self.raw_directory_identity
        observed_identity = (raw_stat.st_dev, raw_stat.st_ino)
        if expected_identity is not None and expected_identity != observed_identity:
            os.close(raw_descriptor)
            raise RuntimeError(
                "workflow raw artifact directory changed during the session"
            )
        # A setgid project directory propagates that bit to newly-created child
        # directories on Linux, so mkdir(mode=0o700) may yield 0o2700. Normalize
        # only the directory this call created. Some collaborative filesystems
        # subsequently reconcile a pinned directory to their exact shared mode
        # (02770); re-harden that one mode only after the inode identity has been
        # pinned by this session. Pre-existing permissive directories and replaced
        # directories must continue to fail closed.
        allowed_setgid_modes = (
            {0o2700, 0o2770} if expected_identity is not None else {0o2700}
        )
        if (
            (created or expected_identity is not None)
            and stat.S_ISDIR(raw_stat.st_mode)
            and raw_stat.st_uid == os.geteuid()
            and raw_stat.st_nlink >= 2
            and stat.S_IMODE(raw_stat.st_mode) in allowed_setgid_modes
        ):
            os.fchmod(raw_descriptor, 0o700)
            raw_stat = os.fstat(raw_descriptor)
        if (
            not stat.S_ISDIR(raw_stat.st_mode)
            or raw_stat.st_uid != os.geteuid()
            or stat.S_IMODE(raw_stat.st_mode) != 0o700
            or raw_stat.st_nlink < 2
        ):
            os.close(raw_descriptor)
            raise RuntimeError(
                "workflow raw artifact directory must be an owner-controlled 0700 "
                "directory"
            )
        self._receipt_state.setdefault("raw_identity", observed_identity)
        return project / "raw", raw_descriptor

    def prepare_raw_directory(self) -> Path:
        """Create, validate, and identity-pin the private raw artifact directory."""

        raw_directory, descriptor = self._open_raw_directory()
        if descriptor >= 0:
            os.close(descriptor)
        return raw_directory

    @property
    def telemetry_file(self) -> Path:
        """Return the workflow-owned usd-cli telemetry artifact path."""

        raw_directory, descriptor = self._open_raw_directory()
        if descriptor >= 0:
            os.close(descriptor)
        return raw_directory / "usd_cli_telemetry.jsonl"

    @property
    def receipt_file(self) -> Path:
        """Return the workflow-owned structured command receipt journal."""

        raw_directory, descriptor = self._open_raw_directory()
        if descriptor >= 0:
            os.close(descriptor)
        if "receipt_identity" in self._receipt_state:
            self._verify_receipt_journal()
        return raw_directory / "usd_cli_command_receipts.jsonl"

    @property
    def receipt_checkpoint_file(self) -> Path:
        """Return the digest-bound checkpoint required to resume in another process."""

        raw_directory, descriptor = self._open_raw_directory()
        if descriptor >= 0:
            os.close(descriptor)
        checkpoint = raw_directory / "usd_cli_command_receipts.checkpoint.json"
        if "receipt_identity" in self._receipt_state:
            self._verify_receipt_checkpoint()
        return checkpoint

    def verify_receipt_journal_integrity(self) -> None:
        """Re-attest an initialized command journal without changing lifecycle.

        Setup may fail before the first usd-cli command creates a journal. That
        empty state has no journal bytes to attest and is not a teardown failure.
        """

        if "receipt_identity" not in self._receipt_state:
            return
        self._verify_receipt_journal()
        self._verify_receipt_checkpoint()

    @staticmethod
    def _validated_receipt_stat(
        descriptor: int,
        *,
        artifact: str = "journal",
    ) -> os.stat_result:
        receipt_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(receipt_stat.st_mode)
            or receipt_stat.st_nlink != 1
            or (
                os.name == "posix"
                and (
                    receipt_stat.st_uid != os.geteuid()
                    or stat.S_IMODE(receipt_stat.st_mode) != 0o600
                )
            )
        ):
            raise RuntimeError(
                f"usd-cli receipt {artifact} must be an owner-controlled, singly "
                "linked 0600 regular file; an identity-pinned 0660 file may be "
                "re-hardened first"
            )
        return receipt_stat

    @staticmethod
    def _descriptor_digest(descriptor: int) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        return digest.hexdigest(), size

    def _reharden_pinned_receipt_stat(self, descriptor: int) -> os.stat_result:
        """Validate the pinned inode and repair exact collaborative mode drift."""

        expected_identity = self._receipt_state.get("receipt_identity")
        receipt_stat = os.fstat(descriptor)
        observed_identity = (receipt_stat.st_dev, receipt_stat.st_ino)
        if expected_identity != observed_identity:
            raise RuntimeError("usd-cli receipt journal was replaced or modified")
        if (
            stat.S_ISREG(receipt_stat.st_mode)
            and receipt_stat.st_nlink == 1
            and receipt_stat.st_uid == os.geteuid()
            and stat.S_IMODE(receipt_stat.st_mode) == 0o660
        ):
            os.fchmod(descriptor, 0o600)
        return self._validated_receipt_stat(descriptor)

    def _verify_open_receipt(self, descriptor: int) -> os.stat_result:
        expected_digest = self._receipt_state.get("receipt_digest")
        expected_size = self._receipt_state.get("receipt_size")
        receipt_stat = self._reharden_pinned_receipt_stat(descriptor)
        observed_digest, observed_size = self._descriptor_digest(descriptor)
        if expected_digest != observed_digest or expected_size != observed_size:
            raise RuntimeError("usd-cli receipt journal was replaced or modified")
        return receipt_stat

    def _verify_receipt_journal(self) -> None:
        if os.name == "nt":
            try:
                observed = read_contained_artifact(
                    self.project_dir,
                    self.project_dir / "raw" / "usd_cli_command_receipts.jsonl",
                )
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "usd-cli receipt journal was replaced, removed, or is no longer "
                    "a safe singly linked regular file"
                ) from exc
            if (
                self._receipt_state.get("receipt_identity")
                != (observed.path.stat().st_dev, observed.path.stat().st_ino)
                or self._receipt_state.get("receipt_digest") != observed.sha256
                or self._receipt_state.get("receipt_size") != observed.size_bytes
            ):
                raise RuntimeError("usd-cli receipt journal was replaced or modified")
            return
        _raw_directory, raw_descriptor = self._open_raw_directory()
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(
                "usd_cli_command_receipts.jsonl",
                flags,
                dir_fd=raw_descriptor,
            )
            try:
                self._verify_open_receipt(descriptor)
            finally:
                os.close(descriptor)
        finally:
            os.close(raw_descriptor)

    def _read_open_receipt_checkpoint(
        self,
        descriptor: int,
    ) -> tuple[os.stat_result, bytes, str]:
        checkpoint_stat = os.fstat(descriptor)
        if (
            stat.S_ISREG(checkpoint_stat.st_mode)
            and checkpoint_stat.st_nlink == 1
            and checkpoint_stat.st_uid == os.geteuid()
            and stat.S_IMODE(checkpoint_stat.st_mode) == 0o660
        ):
            os.fchmod(descriptor, 0o600)
        before = self._validated_receipt_stat(descriptor, artifact="checkpoint")
        if before.st_size > MAX_USD_CLI_RECEIPT_CHECKPOINT_BYTES:
            raise RuntimeError("usd-cli receipt checkpoint is too large")
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, 16 * 1024):
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_USD_CLI_RECEIPT_CHECKPOINT_BYTES:
                raise RuntimeError("usd-cli receipt checkpoint is too large")
        after = self._validated_receipt_stat(descriptor, artifact="checkpoint")
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or size != after.st_size:
            raise RuntimeError("usd-cli receipt checkpoint changed while read")
        checkpoint_bytes = b"".join(chunks)
        return after, checkpoint_bytes, hashlib.sha256(checkpoint_bytes).hexdigest()

    def _open_receipt_checkpoint(self, raw_descriptor: int) -> int:
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            return os.open(
                "usd_cli_command_receipts.checkpoint.json",
                flags,
                dir_fd=raw_descriptor,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("usd-cli receipt checkpoint is missing") from exc
        except OSError as exc:
            raise RuntimeError("usd-cli receipt checkpoint is unsafe") from exc

    def _pin_receipt_checkpoint(self) -> None:
        if os.name == "nt":
            checkpoint_path = (
                self.project_dir / "raw" / "usd_cli_command_receipts.checkpoint.json"
            )
            observed = read_contained_artifact(
                self.project_dir,
                checkpoint_path,
                max_bytes=MAX_USD_CLI_RECEIPT_CHECKPOINT_BYTES,
                capture_bytes=True,
            )
            metadata = observed.path.stat()
            self._receipt_state.update(
                {
                    "checkpoint_identity": (metadata.st_dev, metadata.st_ino),
                    "checkpoint_digest": observed.sha256,
                    "checkpoint_size": observed.size_bytes,
                }
            )
            return
        _raw_directory, raw_descriptor = self._open_raw_directory()
        try:
            descriptor = self._open_receipt_checkpoint(raw_descriptor)
            try:
                checkpoint_stat, checkpoint_bytes, checkpoint_digest = (
                    self._read_open_receipt_checkpoint(descriptor)
                )
            finally:
                os.close(descriptor)
        finally:
            os.close(raw_descriptor)
        self._receipt_state.update(
            {
                "checkpoint_identity": (
                    checkpoint_stat.st_dev,
                    checkpoint_stat.st_ino,
                ),
                "checkpoint_digest": checkpoint_digest,
                "checkpoint_size": len(checkpoint_bytes),
            }
        )

    def _verify_receipt_checkpoint(self) -> None:
        expected_identity = self._receipt_state.get("checkpoint_identity")
        expected_digest = self._receipt_state.get("checkpoint_digest")
        expected_size = self._receipt_state.get("checkpoint_size")
        if (
            expected_identity is None
            or expected_digest is None
            or expected_size is None
        ):
            raise RuntimeError("usd-cli receipt checkpoint was not identity-pinned")
        if os.name == "nt":
            checkpoint_path = (
                self.project_dir / "raw" / "usd_cli_command_receipts.checkpoint.json"
            )
            try:
                observed = read_contained_artifact(
                    self.project_dir,
                    checkpoint_path,
                    max_bytes=MAX_USD_CLI_RECEIPT_CHECKPOINT_BYTES,
                )
                metadata = observed.path.stat()
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "usd-cli receipt checkpoint was replaced or modified"
                ) from exc
            if (
                expected_identity != (metadata.st_dev, metadata.st_ino)
                or expected_digest != observed.sha256
                or expected_size != observed.size_bytes
            ):
                raise RuntimeError(
                    "usd-cli receipt checkpoint was replaced or modified"
                )
            return
        _raw_directory, raw_descriptor = self._open_raw_directory()
        try:
            descriptor = self._open_receipt_checkpoint(raw_descriptor)
            try:
                checkpoint_stat, checkpoint_bytes, checkpoint_digest = (
                    self._read_open_receipt_checkpoint(descriptor)
                )
            finally:
                os.close(descriptor)
        finally:
            os.close(raw_descriptor)
        if (
            expected_identity != (checkpoint_stat.st_dev, checkpoint_stat.st_ino)
            or expected_digest != checkpoint_digest
            or expected_size != len(checkpoint_bytes)
        ):
            raise RuntimeError("usd-cli receipt checkpoint was replaced or modified")

    def _resume_receipt_journal(self, raw_descriptor: int, flags: int) -> int:
        expected_checkpoint = self.receipt_checkpoint_sha256
        if expected_checkpoint is None:
            raise RuntimeError(
                "refusing a pre-existing usd-cli receipt journal without its sealed "
                "checkpoint digest; pass receipt_checkpoint_sha256 from the matching "
                "sealed checkpoint to resume, or start a new workflow project "
                "directory without reusing the unsealed evidence"
            )
        checkpoint_descriptor = self._open_receipt_checkpoint(raw_descriptor)
        try:
            checkpoint_stat, checkpoint_bytes, checkpoint_digest = (
                self._read_open_receipt_checkpoint(checkpoint_descriptor)
            )
        finally:
            os.close(checkpoint_descriptor)
        if checkpoint_digest != expected_checkpoint:
            raise RuntimeError("usd-cli receipt checkpoint identity changed")
        try:
            checkpoint = json.loads(checkpoint_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("usd-cli receipt checkpoint is invalid") from exc
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("schema_version")
            != "content-agent-workflows.usd-cli-receipt-checkpoint.v1"
            or checkpoint.get("workflow") != self.workflow
            or checkpoint.get("session_id") != self.session_id
        ):
            raise RuntimeError("usd-cli receipt checkpoint belongs to another session")
        try:
            identity = (
                int(checkpoint["receipt_device"]),
                int(checkpoint["receipt_inode"]),
            )
            receipt_digest = str(checkpoint["receipt_sha256"])
            receipt_size = int(checkpoint["receipt_size_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("usd-cli receipt checkpoint is incomplete") from exc
        if not re.fullmatch(r"[0-9a-f]{64}", receipt_digest) or receipt_size < 0:
            raise RuntimeError(
                "usd-cli receipt checkpoint has invalid journal metadata"
            )
        self._receipt_state.update(
            {
                "receipt_identity": identity,
                "receipt_digest": receipt_digest,
                "receipt_size": receipt_size,
                "checkpoint_identity": (
                    checkpoint_stat.st_dev,
                    checkpoint_stat.st_ino,
                ),
                "checkpoint_digest": checkpoint_digest,
                "checkpoint_size": len(checkpoint_bytes),
            }
        )
        try:
            descriptor = os.open(
                "usd_cli_command_receipts.jsonl",
                flags,
                dir_fd=raw_descriptor,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "usd-cli receipt journal sealed by the checkpoint is missing"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "usd-cli receipt journal sealed by the checkpoint is unsafe"
            ) from exc
        try:
            self._verify_open_receipt(descriptor)
        except Exception:
            os.close(descriptor)
            raise
        return descriptor

    def _write_receipt_checkpoint(self, receipt_stat: os.stat_result) -> None:
        raw_directory, raw_descriptor = self._open_raw_directory()
        if raw_descriptor >= 0:
            os.close(raw_descriptor)
        checkpoint_path = raw_directory / "usd_cli_command_receipts.checkpoint.json"
        atomic_write_json(
            checkpoint_path,
            {
                "schema_version": (
                    "content-agent-workflows.usd-cli-receipt-checkpoint.v1"
                ),
                "workflow": self.workflow,
                "session_id": self.session_id,
                "receipt_device": receipt_stat.st_dev,
                "receipt_inode": receipt_stat.st_ino,
                "receipt_sha256": self._receipt_state["receipt_digest"],
                "receipt_size_bytes": self._receipt_state["receipt_size"],
            },
            within=self.project_dir,
        )
        os.chmod(checkpoint_path, 0o600)
        self._pin_receipt_checkpoint()

    @staticmethod
    def preflight_package_route(
        *,
        repository_root: Path | None = None,
    ) -> WorkflowUsdCliPackageRoutePin:
        """Pin the exact package route without creating workflow state."""

        root = (
            find_usd_cli_repository_root(Path(__file__).resolve())
            if repository_root is None
            else repository_root.expanduser().resolve()
        )
        return _pin_usd_cli_package_route(root)

    @staticmethod
    def verify_package_route(
        route_pin: WorkflowUsdCliPackageRoutePin,
    ) -> WorkflowUsdCliPackageRoutePin:
        """Fail closed unless an admitted package route is still byte-identical."""

        current = _pin_usd_cli_package_route(route_pin.repository_root)
        if current != route_pin:
            raise RuntimeError(
                "usd-cli package route or implementation identity changed"
            )
        return route_pin

    @classmethod
    def create(
        cls,
        *,
        owner_root: Path,
        project_dir: Path,
        identity: str,
        workflow: str,
        input_roots: Iterable[Path] = (),
        receipt_checkpoint_sha256: str | None = None,
        render_config: Mapping[str, Any] | None = None,
        package_route: WorkflowUsdCliPackageRoutePin | None = None,
    ) -> WorkflowUsdCliSession:
        """Create an isolated workflow session.

        ``render_config={}`` is an explicit empty override and does not inherit
        project render configuration; use ``None`` to inherit only the nearest
        caller project's explicit render configuration. Environment renderer
        variables must be mapped by a caller that explicitly selected remote.
        """

        root = owner_root.expanduser().resolve(strict=True)
        project = project_dir.expanduser().absolute()
        try:
            project.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"usd-cli project escapes workflow root: {project}"
            ) from exc
        relative = project.relative_to(root)
        current = root
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                raise RuntimeError(
                    f"usd-cli project contains a symlink component: {current}"
                )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        session_id = f"workflow-{digest}"
        parent_handoff = _parent_usd_cli_session_identity_from_environment()
        if parent_handoff is not None and render_config is not None:
            raise RuntimeError(
                "attached usd-cli sessions cannot override the parent render "
                "configuration"
            )
        repository_root = find_usd_cli_repository_root(Path(__file__).resolve())
        route_pin = (
            cls.preflight_package_route(repository_root=repository_root)
            if package_route is None
            else cls.verify_package_route(package_route)
        )
        if route_pin.repository_root != repository_root.resolve():
            raise RuntimeError("usd-cli package route belongs to another repository")
        route = route_pin.route
        if parent_handoff is not None:
            parent, _identity_path = parent_handoff
            parent_root = Path(parent.run_dir).expanduser().resolve(strict=True)
            try:
                root.relative_to(parent_root)
                project.relative_to(parent_root)
            except ValueError as exc:
                raise RuntimeError(
                    "child usd-cli project escapes the parent run root"
                ) from exc
            allowed_roots = tuple(
                Path(raw_root).expanduser().resolve(strict=True)
                for raw_root in parent.allowed_roots
            )
            for raw_input_root in input_roots:
                try:
                    input_root = raw_input_root.expanduser().resolve(strict=True)
                except OSError as exc:
                    raise RuntimeError("child usd-cli input is unavailable") from exc
                if not any(
                    input_root == allowed_root or allowed_root in input_root.parents
                    for allowed_root in allowed_roots
                ):
                    raise RuntimeError(
                        "child usd-cli input escapes the parent allowed roots"
                    )
            if (
                repository_root.resolve() != Path(parent.repository_root).resolve()
                or route.source_revision != parent.usd_cli_source_revision
                or route.wrapper.resolve(strict=True)
                != Path(parent.usd_cli_wrapper).resolve(strict=True)
                or route.target.resolve(strict=True)
                != Path(parent.usd_cli_executable).resolve(strict=True)
            ):
                raise RuntimeError(
                    "parent usd-cli session implementation identity changed"
                )
            _verify_live_parent_usd_cli_daemon(
                parent,
                require_os_identity=False,
            )
            unexpected_state = next(
                (
                    candidate
                    for state_name in (".usd-cli", ".ov", ".3dsc")
                    if (candidate := project / state_name).exists()
                    or candidate.is_symlink()
                ),
                None,
            )
            if unexpected_state is not None:
                raise RuntimeError(
                    "child usd-cli project contains competing daemon state: "
                    f"{unexpected_state}"
                )
            # Create a missing domain output scope through the same descriptor-safe
            # contained writer used for workflow artifacts. No nested .usd-cli
            # project is created: every domain gets a named session on the one
            # parent daemon.
            scope_marker = project / ".parent-usd-cli-session"
            atomic_write_text(
                scope_marker,
                f"{session_id}\n",
                within=parent_root,
            )
            project = project.resolve(strict=True)
            return cls(
                project_dir=project,
                session_id=session_id,
                route=route,
                workflow=workflow,
                daemon_project_dir=parent_root,
                daemon_server_url=(f"http://{parent.server_host}:{parent.server_port}"),
                receipt_checkpoint_sha256=receipt_checkpoint_sha256,
                _force_reload_on_every_open=True,
                _render_credentials={},
                _parent_readiness_artifact=parent.readiness_artifact,
                _package_route_pin=route_pin,
            )
        # Admit the exact package route before creating any project-local state.
        # Callers may preflight earlier to avoid their own preparation footprint;
        # this second check closes drift between admission and session creation.
        # Capture the caller's nearest project-local render configuration before
        # creating this run's nearer .usd-cli directory.  The session locks render
        # settings at daemon startup, so copying only this tool configuration keeps
        # an explicitly configured remote OVRTX endpoint available without
        # inheriting the caller's server permissions or mutable daemon state.
        from usd_core.config import dump_toml

        inherited_render = (
            dict(render_config)
            if render_config is not None
            else inherited_project_render_config(project)
        )
        child_render, render_credentials = _separate_render_credentials(
            inherited_render
        )
        state_dir = project / ".usd-cli"
        atomic_write_text(
            state_dir / ".workflow-owned",
            f"{workflow}\n",
            within=root,
        )
        allowed_roots = {root}
        for raw_input_root in input_roots:
            input_root = raw_input_root.expanduser().resolve(strict=True)
            allowed_roots.add(input_root if input_root.is_dir() else input_root.parent)
        run_config: dict[str, Any] = {
            "server": {
                "allowed_roots": [str(path) for path in sorted(allowed_roots, key=str)],
                "allowed_write_roots": [str(project)],
                "host": "127.0.0.1",
            }
        }
        if child_render:
            run_config["render"] = child_render
        atomic_write_text(
            state_dir / "config.toml",
            dump_toml(run_config),
            within=root,
        )
        return cls(
            project_dir=project,
            session_id=session_id,
            route=route,
            workflow=workflow,
            daemon_project_dir=None,
            daemon_server_url=None,
            receipt_checkpoint_sha256=receipt_checkpoint_sha256,
            _render_credentials=render_credentials,
            _package_route_pin=route_pin,
        )

    def run_json(
        self,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
        allow_not_ok: bool = False,
    ) -> dict[str, Any]:
        return self.run_json_result(
            arguments,
            timeout_seconds=timeout_seconds,
            allow_not_ok=allow_not_ok,
        ).payload

    def run_json_result(
        self,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
        allow_not_ok: bool = False,
    ) -> WorkflowUsdCliCommandResult:
        """Run one command and return its response and exit code atomically."""

        if (
            self.daemon_project_dir is not None
            and len(arguments) >= 2
            and arguments[0] == "server"
            and arguments[1] in {"start", "stop", "restart"}
        ):
            raise RuntimeError(
                "attached workflow sessions cannot mutate parent daemon lifecycle"
            )
        if (
            self._force_reload_on_every_open
            and arguments
            and arguments[0] == "open"
            and "--force-reload" not in arguments
        ):
            # Callers that build `open` argv directly (rather than through
            # self.open()) must get the same treatment: a parent-attached child
            # can repeatedly use its named session, where a plain re-open of a
            # path holding unsaved edits is refused by the reload guard.
            arguments = [*arguments, "--force-reload"]
        output_paths = _confined_output_paths(self.project_dir, arguments)
        execution_arguments = (
            _absolute_attached_session_arguments(
                self.project_dir,
                arguments,
                output_paths,
            )
            if self.daemon_project_dir is not None
            else arguments
        )
        command = [
            str(self.route.wrapper),
            "--json",
            *(
                [
                    "--server",
                    self.daemon_server_url,
                ]
                if self.daemon_project_dir is not None
                and self.daemon_server_url is not None
                else []
            ),
            "--session",
            self.session_id,
            *execution_arguments,
        ]
        environment = controlled_usd_cli_telemetry_env(
            route=self.route,
            telemetry_file=self.telemetry_file,
            attrs=f"wu.workflow={self.workflow},wu.session={self.session_id}",
        )
        # The telemetry environment intentionally pins PATH to the verified
        # launcher and system directories. First-use OVRTX/OvPhysX provisioning
        # still needs the operator's uv installation, so pass it as an absolute
        # capability that usd-cli revalidates before execution.
        _pin_uv_executable_capability(environment)
        # The workflow owns this policy choice. usd-cli only exposes the generic
        # lock that prevents a child-writable project config from changing the
        # daemon's attested render endpoint after startup.
        environment["USD_CLI_LOCK_RENDER_CONFIG"] = "1"
        if self.daemon_project_dir is not None:
            # An attached child can reuse only the already-authenticated parent
            # daemon. If that daemon exits, command execution fails closed instead
            # of creating an unowned replacement sidecar.
            environment["USD_CLI_NO_DAEMON"] = "1"
            environment[PARENT_USD_CLI_MANAGED_ENV] = "1"
            environment[PARENT_USD_CLI_PROXY_ALLOWED_ENV] = "1"
            environment[USD_CLI_EXTERNAL_LIFECYCLE_ENV] = "1"
            state_observed = read_contained_artifact(
                self.daemon_project_dir,
                self.daemon_project_dir / ".usd-cli" / "server.json",
                max_bytes=MAX_PARENT_USD_CLI_SERVER_STATE_BYTES,
                capture_bytes=True,
            )
            assert state_observed.data is not None
            try:
                daemon_state = json.loads(state_observed.data)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("parent usd-cli daemon state is invalid") from exc
            if not isinstance(daemon_state, dict):
                raise RuntimeError("parent usd-cli daemon state is invalid")
            host = daemon_state.get("host")
            port = daemon_state.get("port")
            token = daemon_state.get("token")
            observed_url = (
                f"http://{host}:{port}"
                if host == "127.0.0.1"
                and isinstance(port, int)
                and not isinstance(port, bool)
                and 0 < port <= 65535
                else None
            )
            if (
                observed_url != self.daemon_server_url
                or not isinstance(token, str)
                or not token
            ):
                raise RuntimeError("parent usd-cli daemon endpoint identity changed")
            environment["USD_CLI_TOKEN"] = token
            for credential_env in (
                "OVRTX_API_KEY",
                "3DSC_RENDER_REMOTE_API_KEY",
                "OV_RENDER_REMOTE_API_KEY",
                "USD_CLI_RENDER_REMOTE_API_KEY",
                "3DSC_RENDER_BACKEND_API_KEYS_JSON",
                "OV_RENDER_BACKEND_API_KEYS_JSON",
                "USD_CLI_RENDER_BACKEND_API_KEYS_JSON",
            ):
                environment.pop(credential_env, None)
        remote_api_key = self._render_credentials.get("remote_api_key")
        if isinstance(remote_api_key, str) and remote_api_key:
            environment["USD_CLI_RENDER_REMOTE_API_KEY"] = remote_api_key
        backend_api_keys = self._render_credentials.get("backend_api_keys")
        if isinstance(backend_api_keys, dict) and backend_api_keys:
            from usd_core.config import RENDER_BACKEND_API_KEYS_ENV

            environment[RENDER_BACKEND_API_KEYS_ENV] = json.dumps(
                backend_api_keys,
                separators=(",", ":"),
                sort_keys=True,
            )
        operation_id = f"usd-cli-{secrets.token_hex(12)}"
        started_ns = time.time_ns()
        input_files = _referenced_input_files(
            self.project_dir,
            arguments,
            output_paths=output_paths,
        )
        display_arguments = _redact_arguments(arguments)
        secrets_to_redact = _secret_argument_values(arguments)
        if self._package_route_pin is not None:
            # Keep the initially admitted immutable route. Re-resolve only to
            # compare current paths, source revision, and launcher bytes; never
            # replace the pinned launch contract with the current result.
            self.verify_package_route(self._package_route_pin)
        try:
            completed = run_bounded_usd_cli_subprocess(
                command,
                # The nearest parent .usd-cli anchor is still discovered because
                # the project is proven to be below the parent run root. Attached
                # calls bind detected file arguments to this domain project before
                # dispatch because the daemon itself resolves relative paths from
                # the parent run root.
                cwd=self.project_dir,
                env=environment,
                timeout=timeout_seconds,
                check=False,
            )
        except KeyboardInterrupt:
            self._append_receipt(
                operation_id=operation_id,
                started_ns=started_ns,
                arguments=execution_arguments,
                returncode=None,
                stdout="",
                stderr="workflow interrupted while usd-cli was running",
                response=None,
                status="interrupted",
                input_files=input_files,
            )
            raise
        except (OSError, subprocess.TimeoutExpired, UsdCliSubprocessOutputError) as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                failure_stdout = exc.stdout or ""
                failure_stderr = exc.stderr or ""
                if isinstance(failure_stdout, bytes):
                    failure_stdout = failure_stdout.decode("utf-8", errors="replace")
                if isinstance(failure_stderr, bytes):
                    failure_stderr = failure_stderr.decode("utf-8", errors="replace")
                failure_status = "timed_out"
            else:
                failure_stdout = ""
                failure_stderr = str(exc)
                failure_status = "failed"
            self._append_receipt(
                operation_id=operation_id,
                started_ns=started_ns,
                arguments=execution_arguments,
                returncode=None,
                stdout=failure_stdout,
                stderr=failure_stderr,
                response=None,
                status=failure_status,
                input_files=input_files,
            )
            raise RuntimeError(
                "usd-cli command could not complete: "
                f"{display_arguments!r}: {_redact_text(str(exc), secrets_to_redact)}"
            ) from exc
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            self._append_receipt(
                operation_id=operation_id,
                started_ns=started_ns,
                arguments=execution_arguments,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                response=None,
                status="failed",
                input_files=input_files,
            )
            raise RuntimeError(
                f"usd-cli returned non-JSON for {display_arguments!r}: "
                f"{_redact_text(completed.stdout[:500], secrets_to_redact)!r}"
            ) from exc
        if not isinstance(payload, dict):
            self._append_receipt(
                operation_id=operation_id,
                started_ns=started_ns,
                arguments=execution_arguments,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                response=None,
                status="failed",
                input_files=input_files,
            )
            raise RuntimeError(
                f"usd-cli returned a non-object for {display_arguments!r}"
            )
        is_probe = payload.get("schema_version") == "usd-cli.render-probe.v1"
        succeeded = not (
            completed.returncode != 0
            or (payload.get("ok") is not True and not is_probe)
        )
        self._append_receipt(
            operation_id=operation_id,
            started_ns=started_ns,
            arguments=execution_arguments,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            response=payload,
            status="completed" if succeeded else "failed",
            input_files=input_files,
        )
        if not allow_not_ok and not succeeded:
            detail = completed.stderr.strip() or payload.get("issues") or payload
            raise RuntimeError(
                f"usd-cli command failed for {display_arguments!r}: "
                f"{_redact_text(str(detail), secrets_to_redact)}"
            )
        return WorkflowUsdCliCommandResult(
            payload=payload,
            returncode=completed.returncode,
        )

    def _append_receipt(
        self,
        *,
        operation_id: str,
        started_ns: int,
        arguments: list[str],
        returncode: int | None,
        stdout: str,
        stderr: str,
        response: dict[str, Any] | None,
        status: str,
        input_files: list[dict[str, Any]],
    ) -> None:
        completed_ns = time.time_ns()
        secrets_to_redact = _secret_argument_values(arguments)
        redacted_response = (
            _redact_value(response, secrets_to_redact) if response is not None else None
        )
        try:
            package_version = importlib.metadata.version("usd-cli")
        except importlib.metadata.PackageNotFoundError:
            package_version = None
        receipt = {
            "schema_version": "content-agent-workflows.usd-cli-command-receipt.v1",
            "operation_id": operation_id,
            "workflow": self.workflow,
            "session_id": self.session_id,
            "started_unix_nano": started_ns,
            "completed_unix_nano": completed_ns,
            "duration_ms": round((completed_ns - started_ns) / 1_000_000, 3),
            "arguments": _redact_arguments(arguments),
            "tool": {
                "name": "usd-cli",
                "package_version": package_version,
                "source_revision": getattr(self.route, "source_revision", None),
            },
            "returncode": returncode,
            "status": status,
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "stdout": _redact_text(stdout, secrets_to_redact),
            "stderr": _redact_text(stderr, secrets_to_redact),
            "response": redacted_response,
            "inputs": input_files,
            "artifacts": _contained_response_artifacts(
                self.project_dir,
                redacted_response,
            ),
        }
        encoded = (json.dumps(receipt, sort_keys=True) + "\n").encode()
        if os.name == "nt":
            self._append_receipt_windows(encoded)
            return
        _raw_directory, raw_descriptor = self._open_raw_directory()
        flags = os.O_RDWR | os.O_APPEND | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            if "receipt_identity" not in self._receipt_state:
                if self.receipt_checkpoint_sha256 is not None:
                    descriptor = self._resume_receipt_journal(raw_descriptor, flags)
                else:
                    try:
                        descriptor = os.open(
                            "usd_cli_command_receipts.jsonl",
                            flags | os.O_CREAT | os.O_EXCL,
                            0o600,
                            dir_fd=raw_descriptor,
                        )
                    except FileExistsError:
                        descriptor = self._resume_receipt_journal(
                            raw_descriptor,
                            flags,
                        )
                    else:
                        receipt_stat = self._validated_receipt_stat(descriptor)
                        self._receipt_state.update(
                            {
                                "receipt_identity": (
                                    receipt_stat.st_dev,
                                    receipt_stat.st_ino,
                                ),
                                "receipt_digest": hashlib.sha256(b"").hexdigest(),
                                "receipt_size": 0,
                            }
                        )
            else:
                self._verify_receipt_checkpoint()
                descriptor = os.open(
                    "usd_cli_command_receipts.jsonl",
                    flags,
                    dir_fd=raw_descriptor,
                )
            try:
                self._verify_open_receipt(descriptor)
                previous_digest = str(self._receipt_state["receipt_digest"])
                previous_size = int(self._receipt_state["receipt_size"])
                os.lseek(descriptor, 0, os.SEEK_END)
                written = os.write(descriptor, encoded)
                if written != len(encoded):
                    raise RuntimeError("Could not append the complete usd-cli receipt.")
                # The receipt is not complete evidence until it is durable.  In
                # particular, do this before updating the pinned digest/size or
                # publishing a checkpoint that a later process may resume from.
                os.fsync(descriptor)
                receipt_stat = self._reharden_pinned_receipt_stat(descriptor)
                os.lseek(descriptor, 0, os.SEEK_SET)
                journal = b""
                while chunk := os.read(descriptor, 1024 * 1024):
                    journal += chunk
                receipt_size = len(journal)
                if (
                    receipt_stat.st_size != previous_size + len(encoded)
                    or receipt_size != previous_size + len(encoded)
                    or hashlib.sha256(journal[:previous_size]).hexdigest()
                    != previous_digest
                    or journal[previous_size:] != encoded
                ):
                    raise RuntimeError(
                        "usd-cli receipt journal changed concurrently during append"
                    )
                receipt_digest = hashlib.sha256(journal).hexdigest()
                self._receipt_state.update(
                    {
                        "receipt_identity": (receipt_stat.st_dev, receipt_stat.st_ino),
                        "receipt_digest": receipt_digest,
                        "receipt_size": receipt_size,
                    }
                )
                self._write_receipt_checkpoint(receipt_stat)
            finally:
                os.close(descriptor)
        finally:
            os.close(raw_descriptor)

    def _append_receipt_windows(self, encoded: bytes) -> None:
        raw_directory, _descriptor = self._open_raw_directory()
        with open_confined_directory(raw_directory) as raw_root:
            with ExitStack() as journal_stack:
                try:
                    descriptor = journal_stack.enter_context(
                        open_confined_lock_file(
                            raw_root,
                            "usd_cli_command_receipts.jsonl",
                            file_mode=0o600,
                            exclusive_create=True,
                        )
                    )
                    journal_created = True
                except FileExistsError:
                    descriptor = journal_stack.enter_context(
                        open_confined_lock_file(
                            raw_root,
                            "usd_cli_command_receipts.jsonl",
                            file_mode=0o600,
                        )
                    )
                    journal_created = False
                if journal_created and self.receipt_checkpoint_sha256 is not None:
                    # A supplied checkpoint seals an earlier journal, so a
                    # successful exclusive create proves that journal was
                    # removed. Close and remove only the leaf we just created;
                    # never append into a replacement journal.
                    journal_stack.close()
                    if not delete_confined_file(
                        raw_root,
                        "usd_cli_command_receipts.jsonl",
                        missing_ok=False,
                    ):
                        raise RuntimeError(
                            "Could not remove the unsealed replacement usd-cli "
                            "receipt journal."
                        )
                    raise RuntimeError(
                        "usd-cli receipt journal sealed by the checkpoint is missing"
                    )
                with blocking_exclusive_descriptor_lock(descriptor):
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    journal = b""
                    while chunk := os.read(descriptor, 1024 * 1024):
                        journal += chunk
                    metadata = self._validated_receipt_stat(descriptor)
                    identity = (metadata.st_dev, metadata.st_ino)

                    if "receipt_identity" not in self._receipt_state:
                        if not journal_created:
                            expected_checkpoint = self.receipt_checkpoint_sha256
                            if expected_checkpoint is None:
                                raise RuntimeError(
                                    "refusing a pre-existing usd-cli receipt journal "
                                    "without its sealed checkpoint digest"
                                )
                            checkpoint = read_contained_artifact(
                                self.project_dir,
                                raw_directory
                                / "usd_cli_command_receipts.checkpoint.json",
                                max_bytes=MAX_USD_CLI_RECEIPT_CHECKPOINT_BYTES,
                                parse_json=True,
                            )
                            if checkpoint.sha256 != expected_checkpoint:
                                raise RuntimeError(
                                    "usd-cli receipt checkpoint identity changed"
                                )
                            payload = checkpoint.json_object
                            assert payload is not None
                            if (
                                payload.get("workflow") != self.workflow
                                or payload.get("session_id") != self.session_id
                                or payload.get("receipt_sha256")
                                != hashlib.sha256(journal).hexdigest()
                                or payload.get("receipt_size_bytes") != len(journal)
                                or (
                                    payload.get("receipt_device"),
                                    payload.get("receipt_inode"),
                                )
                                != identity
                            ):
                                raise RuntimeError(
                                    "usd-cli receipt checkpoint is inconsistent"
                                )
                        self._receipt_state.update(
                            {
                                "receipt_identity": identity,
                                "receipt_digest": hashlib.sha256(journal).hexdigest(),
                                "receipt_size": len(journal),
                            }
                        )
                    else:
                        self._verify_receipt_checkpoint()
                        if (
                            self._receipt_state.get("receipt_identity") != identity
                            or self._receipt_state.get("receipt_digest")
                            != hashlib.sha256(journal).hexdigest()
                            or self._receipt_state.get("receipt_size") != len(journal)
                        ):
                            raise RuntimeError(
                                "usd-cli receipt journal was replaced or modified"
                            )

                    os.lseek(descriptor, 0, os.SEEK_END)
                    view = memoryview(encoded)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise RuntimeError(
                                "Could not append the complete usd-cli receipt."
                            )
                        view = view[written:]
                    os.fsync(descriptor)
                    journal += encoded
                    final_stat = os.fstat(descriptor)
                    if (
                        final_stat.st_dev,
                        final_stat.st_ino,
                    ) != identity or final_stat.st_size != len(journal):
                        raise RuntimeError(
                            "usd-cli receipt journal changed concurrently during append"
                        )
                    self._receipt_state.update(
                        {
                            "receipt_identity": identity,
                            "receipt_digest": hashlib.sha256(journal).hexdigest(),
                            "receipt_size": len(journal),
                        }
                    )
                    self._write_receipt_checkpoint(final_stat)

    def open(
        self,
        scene_path: Path,
        *,
        read_only: bool = False,
        force_reload: bool = False,
    ) -> dict[str, Any]:
        arguments = ["open", str(scene_path.expanduser().resolve(strict=True))]
        if read_only:
            arguments.append("--read-only")
        if force_reload or self._force_reload_on_every_open:
            arguments.append("--force-reload")
        return self.run_json(arguments)

    def require_ovrtx(self, output_dir: Path) -> dict[str, Any]:
        def run_probe() -> dict[str, Any]:
            arguments = [
                "render-probe",
                "--require-engine",
                "ovrtx",
                "--output-dir",
                str(output_dir.expanduser().absolute()),
            ]
            deadline = time.monotonic() + _OVRTX_PROVISIONING_TIMEOUT_SECONDS
            while True:
                try:
                    return self.run_json(
                        arguments,
                        timeout_seconds=max(0.0, deadline - time.monotonic()),
                    )
                except RuntimeError as exc:
                    detail = str(exc)
                    if not any(
                        message in detail for message in _OVRTX_PROVISIONING_MESSAGES
                    ):
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(
                            "OVRTX provisioning did not finish before the workflow "
                            "session readiness timeout. See the usd-cli daemon log, "
                            "then retry the workflow or configure a remote OVRTX "
                            "service."
                        ) from exc
                    time.sleep(min(_OVRTX_PROVISIONING_POLL_SECONDS, remaining))

        if self._parent_readiness_artifact is not None:
            artifact = self._parent_readiness_artifact
            if self.daemon_project_dir is None:
                raise RuntimeError("parent readiness artifact has no daemon project")
            parent_root = self.daemon_project_dir.resolve(strict=True)
            observed = read_contained_artifact(
                parent_root,
                artifact.path,
                max_bytes=MAX_PARENT_USD_CLI_SESSION_IDENTITY_BYTES,
                parse_json=True,
            )
            if (
                observed.sha256 != artifact.sha256
                or observed.size_bytes != artifact.size_bytes
            ):
                raise RuntimeError("parent readiness artifact changed")
            payload = observed.json_object
            probe = payload.get("probe") if payload is not None else None
            render = probe.get("render") if isinstance(probe, dict) else None
            render_path = render.get("path") if isinstance(render, dict) else None
            if not isinstance(render_path, str) or not render_path:
                provider_free_readiness = (
                    isinstance(payload, dict)
                    and set(payload)
                    == {
                        "schema_version",
                        "usd_cli_version",
                        "usd_cli_source_revision",
                        "probe",
                    }
                    and payload.get("schema_version")
                    == ASSET_USD_CLI_PROVIDER_FREE_READINESS_SCHEMA_VERSION
                    and isinstance(payload.get("usd_cli_version"), str)
                    and bool(payload["usd_cli_version"].strip())
                    and isinstance(payload.get("usd_cli_source_revision"), str)
                    and bool(payload["usd_cli_source_revision"].strip())
                    and probe
                    == {
                        "selected_mode": "agentic",
                        "provider_readiness": "not_requested",
                    }
                )
                if not provider_free_readiness:
                    raise RuntimeError(
                        "parent readiness artifact has no render evidence"
                    )
                attached_probe = run_probe()
                return {
                    **attached_probe,
                    "execution_source": "attached-render-probe",
                }
            render_artifact = read_contained_artifact(
                parent_root,
                render_path,
                max_bytes=16 * 1024 * 1024,
                image=True,
                capture_bytes=True,
            )
            assert render_artifact.data is not None
            if (
                render.get("size_bytes") != render_artifact.size_bytes
                or render.get("sha256") != render_artifact.sha256
            ):
                raise RuntimeError("parent readiness render evidence changed")
            destination = output_dir.expanduser().absolute() / Path(render_path).name
            atomic_write_bytes(destination, render_artifact.data, within=parent_root)
            return {
                **probe,
                "render": {**render, "path": str(destination)},
                "execution_source": "parent-readiness-reuse",
            }
        probe = run_probe()
        return {**probe, "execution_source": "render-probe"}

    @staticmethod
    def stage_up_axis_is_y(scene_path: Path) -> bool:
        return stage_up_axis_is_y(scene_path)

    @staticmethod
    def direction_angles(
        direction: str,
        *,
        up_axis_y: bool = False,
    ) -> tuple[float, float]:
        return direction_angles(direction, up_axis_y=up_axis_y)

    def render_view(
        self,
        *,
        output_dir: Path,
        name: str,
        direction: str,
        backend: Literal["ovrtx", "remote"] | None = None,
        focus: str = "/",
        width: int = 640,
        height: int = 480,
        timeout_seconds: float = 300.0,
        up_axis_y: bool = False,
    ) -> dict[str, Any]:
        """Render one named, direction-controlled OVRTX evidence image."""

        if backend not in {None, "ovrtx", "remote"}:
            raise ValueError(f"Unsupported OVRTX render backend: {backend!r}")

        destination = output_dir.expanduser().absolute()
        try:
            destination.relative_to(self.project_dir)
        except ValueError as exc:
            raise RuntimeError(
                f"usd-cli render output escapes workflow project: {destination}"
            ) from exc
        destination.mkdir(parents=True, exist_ok=True)
        azimuth, elevation = self.direction_angles(
            direction,
            up_axis_y=up_axis_y,
        )
        self.run_json(
            ["camera", "fit", focus],
            timeout_seconds=timeout_seconds,
        )
        self.run_json(
            [
                "camera",
                "orbit",
                focus,
                "--az",
                str(azimuth),
                "--el",
                str(elevation),
            ],
            timeout_seconds=timeout_seconds,
        )
        image_path = destination / f"{name}.png"
        # The workflow daemon freezes its renderer from the nearest project
        # configuration at startup and intentionally rejects per-command
        # overrides.  Treat ``backend`` as an explicit required postcondition:
        # execute against that frozen configuration, then reject any observed
        # backend mismatch below.
        render_arguments = ["render", "--photoreal", "--mode", "quality"]
        render_arguments.extend(
            ["--res", f"{width}x{height}", "--output", str(image_path)]
        )
        response = self.run_json(
            render_arguments,
            timeout_seconds=timeout_seconds,
        )
        try:
            metadata = validated_ovrtx_render_metadata(response)
        except RuntimeError as exc:
            raise RuntimeError(
                "usd-cli render did not produce verified OVRTX evidence"
            ) from exc
        if backend is not None and metadata["backend"] != backend:
            raise RuntimeError(
                "usd-cli render used a different OVRTX backend: "
                f"requested {backend}, observed {metadata['backend']}"
            )
        summary = response.get("summary")
        assert isinstance(summary, dict)
        if not image_path.is_file():
            raise RuntimeError("usd-cli render did not produce verified OVRTX evidence")
        response_path = destination / f"{name}_response.json"
        camera_path = destination / f"{name}_camera.json"
        atomic_write_json(response_path, response, within=self.project_dir)
        atomic_write_json(
            camera_path,
            {
                "scene_tool": "usd-cli",
                "renderer": "ovrtx",
                "transport": metadata["transport"],
                "renderer_identity": metadata["renderer_identity"],
                "direction": direction,
                "azimuth": azimuth,
                "elevation": elevation,
                "camera_position": summary.get(
                    "camera_pos", summary.get("camera_position")
                ),
                "camera_view_direction": summary.get(
                    "camera_dir", summary.get("camera_view_direction")
                ),
                "image_width": width,
                "image_height": height,
            },
            within=self.project_dir,
        )
        return {
            "name": name,
            "direction": direction,
            "image_path": str(image_path),
            "camera_json_path": str(camera_path),
            "response_path": str(response_path),
            "renderer": metadata["backend"],
            "renderer_identity": metadata["renderer_identity"],
        }

    def close(self) -> None:
        """Release this session without masking an active primary failure."""

        primary_error = sys.exc_info()[1]
        try:
            self._close()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            note = f"usd-cli session cleanup also failed: {cleanup_error}"
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(note)
            try:
                setattr(primary_error, "usd_cli_cleanup_error", cleanup_error)
            except (AttributeError, TypeError):
                pass

    def _close(self) -> None:
        if self.daemon_project_dir is not None:
            # The parent launcher owns daemon lifecycle. Child domain sessions
            # retain only command/receipt authority and must never stop it. Release
            # this child's named session so a long sequential asset batch cannot
            # exhaust the bounded parent-daemon registry.
            if "receipt_identity" in self._receipt_state:
                self.verify_receipt_journal_integrity()
            if not self._receipt_state.get("attached_session_released"):
                self.run_json(
                    ["server", "release-session", "--name", self.session_id],
                    timeout_seconds=30.0,
                )
                self._receipt_state["attached_session_released"] = True
            if "receipt_identity" in self._receipt_state:
                self.verify_receipt_journal_integrity()
            return
        environment = controlled_usd_cli_telemetry_env(
            route=self.route,
            telemetry_file=self.telemetry_file,
            attrs=f"wu.workflow={self.workflow},wu.session={self.session_id}",
        )
        environment["USD_CLI_LOCK_RENDER_CONFIG"] = "1"
        if self._package_route_pin is not None:
            self.verify_package_route(self._package_route_pin)
        try:
            completed = run_bounded_usd_cli_subprocess(
                [str(self.route.wrapper), "server", "stop"],
                cwd=self.project_dir,
                env=environment,
                timeout=30.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired, UsdCliSubprocessOutputError) as exc:
            raise RuntimeError(f"could not stop usd-cli sidecar: {exc}") from exc
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"could not stop usd-cli sidecar: {detail}")
        if "receipt_identity" in self._receipt_state:
            self.verify_receipt_journal_integrity()
