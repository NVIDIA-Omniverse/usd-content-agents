# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synchronous helpers for Content Workbench agent HTTP APIs."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Literal, NotRequired, Required, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

DEFAULT_TIMEOUT_SECONDS = 300.0
HEALTH_TIMEOUT_SECONDS = 2.0
_ABSOLUTE_FILESYSTEM_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:[A-Za-z]:\\[^\s;,)]+|/(?:home|Users|tmp|var|mnt|Volumes|workspace|opt|data|srv|root)/[^\s;,)]+)"
)


class HealthResponse(TypedDict):
    status: str
    service: str
    version: NotRequired[str]
    active_sessions: NotRequired[int]
    output_roots: NotRequired[list[str]]


class WorkbenchHealthConfigurationError(RuntimeError):
    """A reachable endpoint does not match the required Workbench policy."""


class AgentApiDiscovery(TypedDict, total=False):
    service: str
    version: str
    agent_api_url: str
    openapi_url: str
    agent_openapi_url: str
    agent_discovery_endpoints: list[str]
    session_root: str
    primary_endpoints: list[str]
    primary_commands: list[str]


class SessionCreateRequest(TypedDict, total=False):
    scene_path: str
    optimize: bool
    clear_materials: bool
    width: int
    height: int
    optimizer_backend: str
    optimization_config: dict[str, Any]


class SceneSnapshotRequest(TypedDict, total=False):
    root_prim_path: str
    include_properties: bool
    include_material_bindings: bool
    include_path_translations: bool
    include_candidate_hints: bool
    max_prims: int


class SceneRestoreRequest(TypedDict, total=False):
    output_usd_path: str
    output_mode: Literal["layer", "composed", "flattened"]
    material_profile: str
    skip_instance_check: bool
    fail_on_invalid_assignment: bool
    overwrite: bool
    include_preview_artifact: bool


class RenderRequest(TypedDict, total=False):
    width: int
    height: int
    direction: str
    use_session_camera: bool
    margin: float
    render_quality: str
    ovrtx_render_mode: str
    ovrtx_num_sensor_updates: int
    save_camera_json: bool


class RenderFramesRequest(RenderRequest, total=False):
    scene_path: str
    output_dir: str
    frames: str | None
    directions: list[str]
    camera_path: str
    make_mp4: bool
    max_duration_seconds: float


class PickRequest(TypedDict, total=False):
    x: int
    y: int
    width: int
    height: int
    update_selection: bool
    mode: str
    ovrtx_render_mode: str
    ovrtx_num_sensor_updates: int


class PathTranslationRequest(TypedDict):
    prim_path: str
    source_space: Literal["source", "inspection"]
    target_space: Literal["source", "inspection"]


class PhysicsInspectRequest(TypedDict, total=False):
    usd_path: str
    root_prim_path: str
    include_existing_schema: bool
    path_space: Literal["source", "inspection"]


class PhysicsTopologyOperation(TypedDict):
    op: Literal[
        "ensure_rigid_body_api",
        "remove_rigid_body_api",
        "remove_fixed_joint",
    ]
    prim_path: str


class PhysicsApplyTopologyPlanRequest(TypedDict, total=False):
    schema_version: Literal["content-workflows.physics-topology-plan.v1"]
    input_usd_path: Required[str]
    output_usd_path: str | None
    expected_source_digest: Required[str]
    mobility_intent: Required[Literal["preserve", "movable", "static"]]
    operations: Required[list[PhysicsTopologyOperation]]
    invariants: Required[dict[str, Any]]


class PhysicsApplySchemaRequest(TypedDict, total=False):
    usd_path: Required[str]
    predictions_jsonl_path: Required[str]
    author_rigid_body: Required[bool]
    decision_patch_path: str
    output_usd_path: str | None
    collision_approximation: str
    output_key: str


class PhysicsRuntimeValidationRequest(TypedDict, total=False):
    physics_usd_path: str
    output_dir: str | None
    engine: Literal["ovphysx", "fake", "none"]
    duration_s: float
    dt: float
    sample_fps: int
    drop_height_m: float | None
    acceptance: dict[str, Any]


class RenderArtifactRecord(TypedDict):
    image_path: str
    response_path: str
    camera_json_path: str | None
    artifact_download_count: int


def normalize_url(workbench_url: str) -> str:
    """Return a Workbench base URL without a trailing slash."""
    return workbench_url.rstrip("/")


def session_url(workbench_url: str, session_id: str, suffix: str = "") -> str:
    """Return an encoded Workbench session URL."""
    encoded_session_id = quote(session_id, safe="")
    normalized_suffix = suffix if not suffix or suffix.startswith("/") else f"/{suffix}"
    return f"{normalize_url(workbench_url)}/sessions/{encoded_session_id}{normalized_suffix}"


def artifact_url(workbench_url: str, artifact_path: str) -> str:
    """Return an absolute Workbench artifact URL from a response path."""
    if artifact_path.startswith(("http://", "https://")):
        base = urlparse(normalize_url(workbench_url))
        artifact = urlparse(artifact_path)
        if (artifact.scheme, artifact.netloc) != (base.scheme, base.netloc):
            raise RuntimeError(
                "Workbench artifact URL must use the same origin as the "
                "Workbench endpoint."
            )
        return artifact_path
    normalized_path = (
        artifact_path if artifact_path.startswith("/") else f"/{artifact_path}"
    )
    return f"{normalize_url(workbench_url)}{normalized_path}"


def post_json(
    url: str, payload: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """POST JSON to Workbench and return an object response."""
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    return request_json(request, url, timeout=timeout)


def get_optional_json(
    url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> dict[str, Any] | None:
    """GET optional JSON, returning None when the request fails."""
    request = Request(url, method="GET")
    try:
        return request_json(request, url, timeout=timeout)
    except RuntimeError:
        return None


def request_json(
    request: Request,
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run a Workbench request and require an object JSON response."""
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = _sanitize_message(exc.read().decode("utf-8", errors="replace"))
        raise RuntimeError(
            f"Workbench request failed for {url}: HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Workbench request failed for {url}: {_sanitize_message(str(exc.reason))}"
        ) from exc
    except (OSError, TimeoutError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Workbench request failed for {url}: {_sanitize_message(str(exc))}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Workbench request returned non-object JSON for {url}")
    return payload


def download_to_file(
    url: str, path: Path, *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> None:
    """Download a Workbench artifact to a local path."""
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            path.write_bytes(response.read())
    except HTTPError as exc:
        detail = _sanitize_message(exc.read().decode("utf-8", errors="replace"))
        raise RuntimeError(
            f"Workbench download failed for {url}: HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Workbench download failed for {url}: {_sanitize_message(str(exc.reason))}"
        ) from exc
    except (OSError, TimeoutError) as exc:
        raise RuntimeError(
            f"Workbench download failed for {url}: {_sanitize_message(str(exc))}"
        ) from exc


def create_session(
    workbench_url: str,
    payload: SessionCreateRequest | dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Create a Workbench session."""
    return post_json(
        f"{normalize_url(workbench_url)}/sessions", payload, timeout=timeout
    )


def close_session(
    workbench_url: str,
    session_id: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Close a Workbench session if a session ID is present."""
    if not session_id:
        return
    request = Request(session_url(workbench_url, session_id), method="DELETE")
    with urlopen(request, timeout=timeout) as response:
        response.read()


def get_agent_api_discovery(
    workbench_url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> AgentApiDiscovery:
    """Return Workbench agent API discovery metadata."""
    return get_json(
        f"{normalize_url(workbench_url)}/agent-api.json",
        timeout=timeout,
    )


def download_agent_api_docs(
    workbench_url: str,
    output_dir: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, str]:
    """Download Workbench agent-facing API documents into an output directory."""
    docs = {
        "agent_api_html": ("/agent-api", output_dir / "agent-api.html"),
        "agent_api_json": ("/agent-api.json", output_dir / "agent-api.json"),
        "agent_capabilities": (
            "/agent/capabilities",
            output_dir / "agent-capabilities.json",
        ),
        "agent_tool_manifest": (
            "/agent/tool-manifest",
            output_dir / "agent-tool-manifest.json",
        ),
        "openapi_json": ("/openapi.json", output_dir / "openapi.json"),
    }
    result: dict[str, str] = {}
    for name, (endpoint, path) in docs.items():
        download_to_file(
            f"{normalize_url(workbench_url)}{endpoint}", path, timeout=timeout
        )
        result[name] = str(path)
    return result


def get_json(url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """GET JSON from Workbench and require an object response."""
    return request_json(Request(url, method="GET"), url, timeout=timeout)


def get_optimization(
    workbench_url: str,
    session_id: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Read the current Workbench session optimization state."""
    return get_json(
        session_url(workbench_url, session_id, "/optimization"), timeout=timeout
    )


def snapshot_scene(
    workbench_url: str,
    session_id: str,
    payload: SceneSnapshotRequest | dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Create a Workbench scene snapshot."""
    return post_json(
        session_url(workbench_url, session_id, "/scene/snapshot"),
        payload,
        timeout=timeout,
    )


def restore_scene(
    workbench_url: str,
    session_id: str,
    payload: SceneRestoreRequest | dict[str, Any] | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Restore current Workbench scene state into durable artifacts."""
    return post_json(
        session_url(workbench_url, session_id, "/scene/restore"),
        payload or {},
        timeout=timeout,
    )


def apply_command(
    workbench_url: str,
    session_id: str,
    command: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Apply a generic Workbench scene command."""
    return post_json(
        session_url(workbench_url, session_id, "/commands"),
        {"command": command, "payload": payload or {}},
        timeout=timeout,
    )


def render(
    workbench_url: str,
    session_id: str,
    payload: RenderRequest | dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Render the current Workbench session."""
    return post_json(
        session_url(workbench_url, session_id, "/render"), payload, timeout=timeout
    )


def render_frames(
    workbench_url: str,
    session_id: str,
    payload: RenderFramesRequest | dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Render a frame sequence from the session scene or a supplied USD."""
    return post_json(
        session_url(workbench_url, session_id, "/render-frames"),
        payload,
        timeout=timeout,
    )


def pick(
    workbench_url: str,
    session_id: str,
    payload: PickRequest | dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Pick a pixel in the current Workbench session."""
    return post_json(
        session_url(workbench_url, session_id, "/pick"), payload, timeout=timeout
    )


def translate_paths(
    workbench_url: str,
    session_id: str,
    requests: list[PathTranslationRequest] | list[dict[str, Any]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Translate a batch of prim paths between Workbench path spaces."""
    return post_json(
        session_url(workbench_url, session_id, "/paths/translate:batch"),
        {"requests": requests},
        timeout=timeout,
    )


def get_material_bindings_batch(
    workbench_url: str,
    session_id: str,
    prim_paths: list[str],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Read material bindings for several prim paths."""
    return post_json(
        session_url(workbench_url, session_id, "/material-binding:batch"),
        {"prim_paths": prim_paths},
        timeout=timeout,
    )


def get_material_assignments(
    workbench_url: str,
    session_id: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Read Workbench material assignment state."""
    return get_json(
        session_url(workbench_url, session_id, "/authoring/material-assignments"),
        timeout=timeout,
    )


def inspect_physics_candidates(
    workbench_url: str,
    session_id: str,
    payload: PhysicsInspectRequest | dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Ask Workbench to inspect mesh prims as physics candidates."""
    return post_json(
        session_url(workbench_url, session_id, "/physics/inspect-mesh-candidates"),
        payload,
        timeout=timeout,
    )


def inspect_physics_components(
    workbench_url: str,
    session_id: str,
    payload: PhysicsInspectRequest | dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Ask Workbench to inspect logical physics components."""
    return post_json(
        session_url(workbench_url, session_id, "/physics/inspect-components"),
        dict(payload),
        timeout=timeout,
    )


def inspect_physics_topology(
    workbench_url: str,
    session_id: str,
    payload: PhysicsInspectRequest | dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Ask Workbench to inspect authored physics topology facts."""
    return post_json(
        session_url(workbench_url, session_id, "/physics/inspect-topology"),
        dict(payload),
        timeout=timeout,
    )


def apply_physics_topology_plan(
    workbench_url: str,
    session_id: str,
    payload: PhysicsApplyTopologyPlanRequest | dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Ask Workbench to apply an explicit topology plan."""
    return post_json(
        session_url(workbench_url, session_id, "/physics/apply-topology-plan"),
        dict(payload),
        timeout=timeout,
    )


def apply_physics_schema(
    workbench_url: str,
    session_id: str,
    payload: PhysicsApplySchemaRequest | dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Ask Workbench to author USD physics schema from accepted decisions."""
    return post_json(
        session_url(workbench_url, session_id, "/physics/apply-schema"),
        payload,
        timeout=timeout,
    )


def validate_physics_runtime(
    workbench_url: str,
    session_id: str,
    payload: PhysicsRuntimeValidationRequest | dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Ask Workbench to run target-runtime physics validation."""
    return post_json(
        session_url(workbench_url, session_id, "/physics/validate-runtime"),
        payload,
        timeout=timeout,
    )


def _health_payload(workbench_url: str) -> dict[str, Any]:
    return get_json(
        f"{normalize_url(workbench_url)}/healthz",
        timeout=HEALTH_TIMEOUT_SECONDS,
    )


def check_health(
    workbench_url: str,
    *,
    output_root: Path | None = None,
) -> None:
    """Require a healthy Workbench endpoint with the expected output policy."""
    payload = _health_payload(workbench_url)
    if payload.get("service") != "content-workbench":
        raise WorkbenchHealthConfigurationError(
            "healthz did not report content-workbench service: "
            f"{payload.get('service')!r}"
        )
    if output_root is None:
        return
    expected_roots = [str(output_root.expanduser().resolve())]
    actual_roots = payload.get("output_roots")
    if actual_roots != expected_roots:
        raise WorkbenchHealthConfigurationError(
            "healthz did not report the required run-scoped output root: "
            f"expected {expected_roots!r}, got {actual_roots!r}"
        )


def is_healthy(
    workbench_url: str,
    *,
    output_root: Path | None = None,
) -> bool:
    """Return whether Workbench is healthy with the expected output policy."""
    try:
        check_health(workbench_url, output_root=output_root)
    except RuntimeError:
        return False
    return True


def wait_until_healthy(
    workbench_url: str,
    *,
    timeout_seconds: float,
    output_root: Path | None = None,
) -> None:
    """Wait until Workbench is healthy or raise RuntimeError."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            check_health(workbench_url, output_root=output_root)
        except WorkbenchHealthConfigurationError:
            raise
        except RuntimeError as exc:
            last_error = exc
        else:
            return
        time.sleep(0.5)
    message = f"Content Workbench endpoint is not healthy: {normalize_url(workbench_url)}/healthz"
    if last_error is not None:
        message = f"{message} ({last_error})"
    raise RuntimeError(message)


def render_view(
    *,
    workbench_url: str,
    session_id: str,
    output_dir: Path,
    name: str,
    direction: str,
    width: int,
    height: int,
    render_quality: str,
) -> dict[str, Any]:
    """Render a named Workbench view and download image/camera artifacts."""
    request = {
        "width": width,
        "height": height,
        "use_session_camera": False,
        "direction": direction,
        "margin": 1.25,
        "render_quality": render_quality,
        "save_camera_json": True,
    }
    response = render(workbench_url, session_id, request)
    image_path = output_dir / f"{name}.png"
    response_path = output_dir / f"{name}_response.json"
    camera_path = output_dir / f"{name}_camera.json"
    record = download_render_artifacts(
        workbench_url=workbench_url,
        response=response,
        image_path=image_path,
        response_path=response_path,
        camera_path=camera_path,
        missing_image_message=(
            f"Workbench render response for view {name!r} is missing image_url."
        ),
    )
    return {
        "name": name,
        "direction": direction,
        "width": width,
        "height": height,
        "render_quality": render_quality,
        **record,
        "elapsed_seconds": response.get("elapsed_seconds"),
        "ovrtx_render_mode": response.get("ovrtx_render_mode"),
        "ovrtx_num_sensor_updates": response.get("ovrtx_num_sensor_updates"),
    }


def download_render_artifacts(
    *,
    workbench_url: str,
    response: dict[str, Any],
    image_path: Path,
    response_path: Path,
    camera_path: Path | None = None,
    missing_image_message: str = "Workbench render response is missing image_url.",
) -> RenderArtifactRecord:
    """Download image and optional camera artifacts from a Workbench render response."""
    image_url = response.get("image_url")
    if not isinstance(image_url, str) or not image_url:
        raise RuntimeError(missing_image_message)
    artifact_download_count = 1
    download_to_file(artifact_url(workbench_url, image_url), image_path)
    camera_json_path: str | None = None
    camera_json_url = response.get("camera_json_url")
    if camera_path is not None and isinstance(camera_json_url, str) and camera_json_url:
        download_to_file(artifact_url(workbench_url, camera_json_url), camera_path)
        artifact_download_count += 1
        camera_json_path = str(camera_path)
    response_path.write_text(
        json.dumps(response, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "image_path": str(image_path),
        "camera_json_path": camera_json_path,
        "response_path": str(response_path),
        "artifact_download_count": artifact_download_count,
    }


def _sanitize_message(message: str) -> str:
    return _ABSOLUTE_FILESYSTEM_PATH_RE.sub("<path>", message)
