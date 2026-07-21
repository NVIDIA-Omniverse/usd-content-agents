# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI application for the content workbench."""

from __future__ import annotations

import logging
import mimetypes
import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse

from . import physics_ops
from .env import first_nonempty_env as _first_nonempty_env
from .material_apply_adapter import MaterialApplyUnavailableError
from .models import (
    MAX_RENDER_DIMENSION,
    BatchMaterialBindingResponse,
    BatchPathTranslationRequest,
    BatchPathTranslationResponse,
    BatchPrimPathsRequest,
    BatchPropertiesResponse,
    CameraState,
    CommandRequest,
    CommandResponse,
    CreateSessionRequest,
    DiagnosticsResponse,
    HealthResponse,
    LoadSceneRequest,
    MaterialApplyRequest,
    MaterialApplyResponse,
    MaterialAssignmentsResponse,
    MaterialBindingResponse,
    OptimizationState,
    PathTranslationRequest,
    PathTranslationResponse,
    PhysicsApplySchemaRequest,
    PhysicsApplyTopologyPlanRequest,
    PhysicsInspectCandidatesRequest,
    PhysicsInspectComponentsRequest,
    PhysicsRuntimeValidationRequest,
    PickRequest,
    PickResponse,
    PropertiesResponse,
    RenderFramesRequest,
    RenderFramesResponse,
    RenderRequest,
    RenderResponse,
    SceneRestoreRequest,
    SceneRestoreResponse,
    SceneSnapshotRequest,
    SceneSnapshotResponse,
    SessionResponse,
    TreeResponse,
)
from .sessions import SessionManager
from .version import SERVICE_VERSION

logger = logging.getLogger(__name__)
_IPV4_LOOPBACK_RE = r"127(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}"
LOCALHOST_CORS_ORIGIN_REGEX = (
    rf"^https?://(localhost|{_IPV4_LOOPBACK_RE}|\[::1\]|"
    rf"\[::ffff:{_IPV4_LOOPBACK_RE}\])(:\d+)?$"
)
MAX_SCREENSHOT_GET_DIMENSION = 2048
_ABSOLUTE_FILESYSTEM_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:[A-Za-z]:\\[^\s;,)]+|/(?:home|Users|tmp|var|mnt|Volumes|workspace|opt|data|srv|root)/[^\s;,)]+)"
)
_PRIMARY_AGENT_ENDPOINTS = [
    "/sessions",
    "/sessions/{session_id}",
    "/sessions/{session_id}/scene",
    "/sessions/{session_id}/scene/optimize",
    "/sessions/{session_id}/scene/restore",
    "/sessions/{session_id}/tree",
    "/sessions/{session_id}/scene/snapshot",
    "/sessions/{session_id}/properties",
    "/sessions/{session_id}/properties:batch",
    "/sessions/{session_id}/material-binding",
    "/sessions/{session_id}/material-binding:batch",
    "/sessions/{session_id}/authoring/material-assignments",
    "/sessions/{session_id}/authoring/material-assignments:apply",
    "/sessions/{session_id}/diagnostics",
    "/sessions/{session_id}/optimization",
    "/sessions/{session_id}/paths/translate",
    "/sessions/{session_id}/paths/translate:batch",
    "/sessions/{session_id}/camera",
    "/sessions/{session_id}/commands",
    "/sessions/{session_id}/pick",
    "/sessions/{session_id}/render",
    "/sessions/{session_id}/render-frames",
    "/sessions/{session_id}/renders/{filename:path}",
    "/sessions/{session_id}/screenshot",
    "/sessions/{session_id}/physics/inspect-mesh-candidates",
    "/sessions/{session_id}/physics/inspect-components",
    "/sessions/{session_id}/physics/inspect-topology",
    "/sessions/{session_id}/physics/apply-topology-plan",
    "/sessions/{session_id}/physics/apply-schema",
    "/sessions/{session_id}/physics/validate-runtime",
]
_AGENT_DISCOVERY_ENDPOINTS = [
    "/agent-api",
    "/agent-api.json",
    "/agent/capabilities",
    "/agent/openapi.json",
    "/agent/tool-manifest",
]
_PRIMARY_AGENT_COMMANDS = [
    "frame",
    "orbit",
    "pan",
    "dolly",
    "select",
    "hide",
    "show",
    "isolate",
    "clear_isolation",
    "material_override",
    "clear_material_override",
    "clear_visual_overrides",
    "reset_view",
    "change_aov",
]
_WORKBENCH_CAPABILITIES = [
    "scene_session_lifecycle",
    "scene_snapshot",
    "scene_optimization_at_session_start",
    "source_inspection_path_translation",
    "viewport_render",
    "batched_frame_render",
    "pixel_pick",
    "camera_control",
    "scene_commands",
    "material_assignment_state",
    "material_assignment_apply",
    "frame_sequence_render",
    "physics_candidate_inspection",
    "physics_component_inspection",
    "physics_topology_inspection",
    "physics_topology_plan_apply",
    "physics_schema_apply",
    "physics_runtime_validation",
]


def _agent_api_doc_path() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "agent_api.md"


def _agent_api_discovery_payload() -> dict[str, object]:
    return {
        "service": "content-workbench",
        "version": SERVICE_VERSION,
        "agent_api_url": "/agent-api",
        "openapi_url": "/openapi.json",
        "agent_openapi_url": "/agent/openapi.json",
        "capabilities_url": "/agent/capabilities",
        "tool_manifest_url": "/agent/tool-manifest",
        "canonical_agent_api_path": str(_agent_api_doc_path()),
        "session_root": "/sessions",
        "primary_endpoints": list(_PRIMARY_AGENT_ENDPOINTS),
        "agent_discovery_endpoints": list(_AGENT_DISCOVERY_ENDPOINTS),
        "primary_commands": list(_PRIMARY_AGENT_COMMANDS),
        "capabilities": list(_WORKBENCH_CAPABILITIES),
    }


def _agent_tool_manifest_payload() -> dict[str, object]:
    return {
        "service": "content-workbench",
        "version": SERVICE_VERSION,
        "transport": "rest",
        "base_url_hint": "http://127.0.0.1:<port>",
        "discovery": {
            "human_guide": "/agent-api",
            "legacy_discovery": "/agent-api.json",
            "openapi": "/agent/openapi.json",
            "capabilities": "/agent/capabilities",
        },
        "operations": [
            {
                "name": "create_session",
                "method": "POST",
                "path": "/sessions",
                "purpose": "Create a scene session and optionally load/optimize a USD scene.",
            },
            {
                "name": "load_scene",
                "method": "POST",
                "path": "/sessions/{session_id}/scene",
                "purpose": "Load or reload a USD scene into an existing session.",
            },
            {
                "name": "optimize_scene",
                "method": "POST",
                "path": "/sessions/{session_id}/scene/optimize",
                "purpose": "Optimize the currently loaded scene or a supplied scene path.",
            },
            {
                "name": "snapshot_scene",
                "method": "POST",
                "path": "/sessions/{session_id}/scene/snapshot",
                "purpose": "Collect hierarchy, properties, material bindings, path translations, and candidate hints.",
            },
            {
                "name": "render",
                "method": "POST",
                "path": "/sessions/{session_id}/render",
                "purpose": "Render the current scene/camera/view direction and return artifact URLs.",
            },
            {
                "name": "pick",
                "method": "POST",
                "path": "/sessions/{session_id}/pick",
                "purpose": "Pick a rendered pixel and return resolved prim paths.",
            },
            {
                "name": "translate_path",
                "method": "POST",
                "path": "/sessions/{session_id}/paths/translate",
                "purpose": "Translate one prim path between source and inspection scene spaces.",
            },
            {
                "name": "translate_paths",
                "method": "POST",
                "path": "/sessions/{session_id}/paths/translate:batch",
                "purpose": "Translate multiple prim paths between source and inspection scene spaces.",
            },
            {
                "name": "restore_scene",
                "method": "POST",
                "path": "/sessions/{session_id}/scene/restore",
                "purpose": "Restore current editable state into durable scene artifacts using source mapping where supported.",
            },
            {
                "name": "inspect_physics_candidates",
                "method": "POST",
                "path": "/sessions/{session_id}/physics/inspect-mesh-candidates",
                "purpose": "Inspect mesh prims and return geometry/material hints for physics property reasoning.",
            },
            {
                "name": "inspect_physics_components",
                "method": "POST",
                "path": "/sessions/{session_id}/physics/inspect-components",
                "purpose": "Inspect logical components with separate visual, collider, helper, body, and joint roles.",
            },
            {
                "name": "inspect_physics_topology",
                "method": "POST",
                "path": "/sessions/{session_id}/physics/inspect-topology",
                "purpose": "Inspect authored rigid-body, collider-ownership, joint, and articulation topology facts.",
            },
            {
                "name": "apply_physics_topology_plan",
                "method": "POST",
                "path": "/sessions/{session_id}/physics/apply-topology-plan",
                "purpose": "Apply a digest-bound, intent-gated topology plan to a derivative USD.",
            },
            {
                "name": "apply_physics_schema",
                "method": "POST",
                "path": "/sessions/{session_id}/physics/apply-schema",
                "purpose": "Apply USD physics schemas from accepted physics predictions.",
            },
            {
                "name": "validate_physics_runtime",
                "method": "POST",
                "path": "/sessions/{session_id}/physics/validate-runtime",
                "purpose": "Run physics runtime validation and return evidence artifacts.",
            },
            {
                "name": "render_frame_sequence",
                "method": "POST",
                "path": "/sessions/{session_id}/render-frames",
                "purpose": "Render an ordered frame sequence from the session scene or a supplied time-sampled USD.",
            },
            {
                "name": "apply_command",
                "method": "POST",
                "path": "/sessions/{session_id}/commands",
                "purpose": "Apply camera, selection, visibility, AOV, or material override commands.",
            },
        ],
        "common_errors": [
            "404 for missing session or render artifact",
            "400 for invalid paths, commands, or request shapes",
            "409 for renderer or material-apply timeouts",
        ],
        "limits": {
            "max_render_dimension": MAX_RENDER_DIMENSION,
            "network_scope": "loopback-local service by default",
        },
    }


def _http_error(error: Exception) -> HTTPException:
    detail = _sanitize_error_detail(str(error).strip("'"))
    if isinstance(error, KeyError):
        return HTTPException(status_code=404, detail=detail)
    if isinstance(error, FileNotFoundError):
        return HTTPException(status_code=404, detail=detail)
    if isinstance(error, ValueError):
        return HTTPException(status_code=400, detail=detail)
    if isinstance(error, TimeoutError):
        return HTTPException(status_code=409, detail=detail)
    if isinstance(error, MaterialApplyUnavailableError):
        return HTTPException(status_code=501, detail=detail)
    logger.exception("Unhandled service error")
    return HTTPException(status_code=500, detail="Internal server error")


def _sanitize_error_detail(detail: str) -> str:
    return _ABSOLUTE_FILESYSTEM_PATH_RE.sub("<path>", detail)


def _cors_origins_from_env() -> list[str]:
    raw = os.environ.get("CONTENT_WORKBENCH_CORS_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _cors_origin_regex_from_env() -> str | None:
    value = os.environ.get("CONTENT_WORKBENCH_CORS_ORIGIN_REGEX")
    if value is None or value == "":
        return LOCALHOST_CORS_ORIGIN_REGEX
    if value.strip().lower() in {"none", "off", "disabled"}:
        return None
    return value


class LazyASGIApp:
    """ASGI callable that defers FastAPI construction until first use."""

    def __init__(self, factory: Callable[[], FastAPI]) -> None:
        self._factory = factory
        self._app: FastAPI | None = None

    def _get_app(self) -> FastAPI:
        if self._app is None:
            self._app = self._factory()
        return self._app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        await self._get_app()(scope, receive, send)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_app(), name)


def create_app(manager: SessionManager | None = None) -> FastAPI:
    """Create the FastAPI app.

    The optional manager argument keeps tests deterministic and allows future
    process-level runtime wiring to inject a renderer-backed manager.
    """
    session_manager = manager or SessionManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            session_manager.shutdown()

    app = FastAPI(
        title="Content Workbench",
        description="Local scene inspection service for USD scenes",
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    app.state.session_manager = session_manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins_from_env(),
        allow_origin_regex=_cors_origin_regex_from_env(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            active_sessions=session_manager.active_session_count,
            output_roots=[str(root) for root in session_manager.output_roots],
        )

    @app.get("/agent-api", response_class=PlainTextResponse)
    def agent_api() -> str:
        """Return the canonical agent-facing workbench API guide."""
        return _agent_api_doc_path().read_text(encoding="utf-8")

    @app.get("/agent-api.json")
    def agent_api_discovery() -> dict[str, object]:
        """Return stable discovery metadata for agent wrappers."""
        return _agent_api_discovery_payload()

    @app.get("/agent/capabilities")
    def agent_capabilities() -> dict[str, object]:
        """Return concise Workbench capabilities for agent discovery."""
        discovery = _agent_api_discovery_payload()
        return {
            "service": discovery["service"],
            "version": discovery["version"],
            "capabilities": discovery["capabilities"],
            "primary_endpoints": discovery["primary_endpoints"],
            "agent_discovery_endpoints": discovery["agent_discovery_endpoints"],
            "tool_manifest_url": discovery["tool_manifest_url"],
            "openapi_url": discovery["agent_openapi_url"],
        }

    @app.get("/agent/openapi.json")
    def agent_openapi() -> dict[str, Any]:
        """Alias the FastAPI OpenAPI schema under the target agent namespace."""
        return app.openapi()

    @app.get("/agent/tool-manifest")
    def agent_tool_manifest() -> dict[str, object]:
        """Return a compact prompt-friendly tool manifest for agents."""
        return _agent_tool_manifest_payload()

    @app.post("/sessions", response_model=SessionResponse, status_code=201)
    def create_session(request: CreateSessionRequest) -> SessionResponse:
        try:
            return session_manager.create_session(request)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/sessions/{session_id}", response_model=SessionResponse)
    def get_session(session_id: str) -> SessionResponse:
        try:
            return session_manager.get_session(session_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.delete("/sessions/{session_id}", response_model=SessionResponse)
    def close_session(session_id: str) -> SessionResponse:
        try:
            return session_manager.close_session(session_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/sessions/{session_id}/scene", response_model=SessionResponse)
    def load_scene(session_id: str, request: LoadSceneRequest) -> SessionResponse:
        try:
            return session_manager.load_scene(session_id, request)
        except Exception as error:
            raise _http_error(error) from error

    @app.post(
        "/sessions/{session_id}/scene/optimize",
        response_model=SessionResponse,
    )
    def optimize_scene(
        session_id: str,
        request: CreateSessionRequest | None = None,
    ) -> SessionResponse:
        try:
            session = session_manager.get_session(session_id)
            options = request or CreateSessionRequest(optimize=True)
            scene_path = (
                options.scene_path
                or session.source_scene_path
                or session.scene_path
                or session.inspection_scene_path
            )
            if not scene_path:
                raise ValueError(
                    "No scene is loaded; provide scene_path or load a scene first."
                )
            return session_manager.load_scene(
                session_id,
                LoadSceneRequest(
                    scene_path=scene_path,
                    optimize=True,
                    optimizer_backend=options.optimizer_backend,
                    flatten_prototypes=options.flatten_prototypes,
                    enable_deinstance=options.enable_deinstance,
                    enable_split=options.enable_split,
                    enable_deduplicate=options.enable_deduplicate,
                    clear_materials=options.clear_materials or session.clear_materials,
                    optimization_config=options.optimization_config,
                ),
            )
        except Exception as error:
            raise _http_error(error) from error

    @app.post(
        "/sessions/{session_id}/scene/restore",
        response_model=SceneRestoreResponse,
    )
    def restore_scene(
        session_id: str,
        request: SceneRestoreRequest | None = None,
    ) -> SceneRestoreResponse:
        try:
            return session_manager.restore_scene(
                session_id,
                request or SceneRestoreRequest(),
            )
        except Exception as error:
            raise _http_error(error) from error

    @app.post(
        "/sessions/{session_id}/scene/snapshot",
        response_model=SceneSnapshotResponse,
    )
    def snapshot_scene(
        session_id: str,
        request: SceneSnapshotRequest,
    ) -> SceneSnapshotResponse:
        try:
            return session_manager.snapshot_scene(session_id, request)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/sessions/{session_id}/commands", response_model=CommandResponse)
    def apply_command(
        session_id: str,
        request: CommandRequest,
    ) -> CommandResponse:
        try:
            return session_manager.apply_command(session_id, request)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/sessions/{session_id}/tree", response_model=TreeResponse)
    def get_tree(
        session_id: str,
        prim_path: str | None = Query(default=None),
    ) -> TreeResponse:
        try:
            return session_manager.get_tree(session_id, prim_path)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/sessions/{session_id}/properties", response_model=PropertiesResponse)
    def get_properties(
        session_id: str,
        prim_path: str = Query(...),
    ) -> PropertiesResponse:
        try:
            return session_manager.get_properties(session_id, prim_path)
        except Exception as error:
            raise _http_error(error) from error

    @app.post(
        "/sessions/{session_id}/properties:batch",
        response_model=BatchPropertiesResponse,
    )
    def get_properties_batch(
        session_id: str,
        request: BatchPrimPathsRequest,
    ) -> BatchPropertiesResponse:
        try:
            return BatchPropertiesResponse(
                session_id=session_id,
                results=[
                    session_manager.get_properties(session_id, prim_path)
                    for prim_path in request.prim_paths
                ],
            )
        except Exception as error:
            raise _http_error(error) from error

    @app.get(
        "/sessions/{session_id}/material-binding",
        response_model=MaterialBindingResponse,
    )
    def get_material_binding(
        session_id: str,
        prim_path: str = Query(...),
    ) -> MaterialBindingResponse:
        try:
            return session_manager.get_material_binding(session_id, prim_path)
        except Exception as error:
            raise _http_error(error) from error

    @app.post(
        "/sessions/{session_id}/material-binding:batch",
        response_model=BatchMaterialBindingResponse,
    )
    def get_material_binding_batch(
        session_id: str,
        request: BatchPrimPathsRequest,
    ) -> BatchMaterialBindingResponse:
        try:
            return BatchMaterialBindingResponse(
                session_id=session_id,
                results=[
                    session_manager.get_material_binding(session_id, prim_path)
                    for prim_path in request.prim_paths
                ],
            )
        except Exception as error:
            raise _http_error(error) from error

    @app.get(
        "/sessions/{session_id}/authoring/material-assignments",
        response_model=MaterialAssignmentsResponse,
    )
    def get_material_assignments(session_id: str) -> MaterialAssignmentsResponse:
        try:
            return session_manager.get_material_assignments(session_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.post(
        "/sessions/{session_id}/authoring/material-assignments:apply",
        response_model=MaterialApplyResponse,
    )
    def apply_material_assignments(
        session_id: str,
        request: MaterialApplyRequest,
    ) -> MaterialApplyResponse:
        try:
            return session_manager.apply_material_assignments(session_id, request)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/sessions/{session_id}/physics/inspect-mesh-candidates")
    def inspect_physics_candidates(
        session_id: str,
        request: PhysicsInspectCandidatesRequest,
    ) -> dict[str, Any]:
        try:
            session_manager.get_session(session_id)

            return physics_ops.inspect_mesh_candidates(
                request.usd_path,
                root_prim_path=request.root_prim_path,
                include_existing_schema=request.include_existing_schema,
                path_space=request.path_space,
            )
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/sessions/{session_id}/physics/inspect-components")
    def inspect_physics_components(
        session_id: str,
        request: PhysicsInspectComponentsRequest,
    ) -> dict[str, Any]:
        try:
            session_manager.get_session(session_id)
            return physics_ops.inspect_components(
                request.usd_path,
                root_prim_path=request.root_prim_path,
                path_space=request.path_space,
            )
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/sessions/{session_id}/physics/inspect-topology")
    def inspect_physics_topology(
        session_id: str,
        request: PhysicsInspectComponentsRequest,
    ) -> dict[str, Any]:
        try:
            session_manager.get_session(session_id)
            return physics_ops.inspect_topology(
                request.usd_path,
                root_prim_path=request.root_prim_path,
                path_space=request.path_space,
            )
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/sessions/{session_id}/physics/apply-topology-plan")
    def apply_physics_topology_plan(
        session_id: str,
        request: PhysicsApplyTopologyPlanRequest,
    ) -> dict[str, Any]:
        workspace_op_started = False
        try:
            with session_manager._lock:
                session_manager.get_session(session_id)
                session_manager._begin_workspace_operation(session_id)
                workspace_op_started = True

            output_usd_path = session_manager.resolve_agent_output_usd_path(
                session_id,
                request.output_usd_path,
                default_subdir="physics",
                default_filename="prepared.usda",
            )
            return physics_ops.apply_topology_plan(
                input_usd_path=request.input_usd_path,
                output_usd_path=output_usd_path,
                expected_source_digest=request.expected_source_digest,
                mobility_intent=request.mobility_intent,
                operations=[operation.model_dump() for operation in request.operations],
                invariants=request.invariants.model_dump(),
            )
        except Exception as error:
            raise _http_error(error) from error
        finally:
            if workspace_op_started:
                with session_manager._lock:
                    session_manager._end_workspace_operation(session_id)

    @app.post("/sessions/{session_id}/physics/apply-schema")
    def apply_physics_schema(
        session_id: str,
        request: PhysicsApplySchemaRequest,
    ) -> dict[str, Any]:
        workspace_op_started = False
        try:
            with session_manager._lock:
                session_manager.get_session(session_id)
                session_manager._begin_workspace_operation(session_id)
                workspace_op_started = True

            output_usd_path = session_manager.resolve_agent_output_usd_path(
                session_id,
                request.output_usd_path,
                default_subdir="physics",
                default_filename="physics.usda",
            )
            return physics_ops.apply_schema(
                usd_path=request.usd_path,
                decision_patch_path=request.decision_patch_path,
                predictions_jsonl_path=request.predictions_jsonl_path,
                output_usd_path=output_usd_path,
                collision_approximation=request.collision_approximation,
                output_key=request.output_key,
                author_rigid_body=request.author_rigid_body,
            )
        except Exception as error:
            raise _http_error(error) from error
        finally:
            if workspace_op_started:
                with session_manager._lock:
                    session_manager._end_workspace_operation(session_id)

    @app.post("/sessions/{session_id}/physics/validate-runtime")
    def validate_physics_runtime(
        session_id: str,
        request: PhysicsRuntimeValidationRequest,
    ) -> dict[str, Any]:
        workspace_op_started = False
        try:
            with session_manager._lock:
                session_manager.get_session(session_id)
                session_manager._begin_workspace_operation(session_id)
                workspace_op_started = True

            output_dir = session_manager.resolve_agent_output_dir(
                session_id,
                request.output_dir,
                default_subdir="physics-runtime",
            )
            return physics_ops.validate_runtime(
                physics_usd=request.physics_usd_path,
                output_dir=output_dir,
                engine=request.engine,
                duration_s=request.duration_s,
                dt=request.dt,
                sample_fps=request.sample_fps,
                drop_height_m=request.drop_height_m,
                acceptance=request.acceptance.model_dump(exclude_unset=True)
                if request.acceptance is not None
                else None,
            )
        except Exception as error:
            raise _http_error(error) from error
        finally:
            if workspace_op_started:
                with session_manager._lock:
                    session_manager._end_workspace_operation(session_id)

    @app.get("/sessions/{session_id}/diagnostics", response_model=DiagnosticsResponse)
    def get_diagnostics(session_id: str) -> DiagnosticsResponse:
        try:
            return session_manager.get_diagnostics(session_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.get(
        "/sessions/{session_id}/optimization",
        response_model=OptimizationState,
    )
    def get_optimization(session_id: str) -> OptimizationState:
        try:
            return session_manager.get_session(session_id).optimization
        except Exception as error:
            raise _http_error(error) from error

    @app.post(
        "/sessions/{session_id}/paths/translate",
        response_model=PathTranslationResponse,
    )
    def translate_path(
        session_id: str,
        request: PathTranslationRequest,
    ) -> PathTranslationResponse:
        try:
            return session_manager.translate_path(session_id, request)
        except Exception as error:
            raise _http_error(error) from error

    @app.post(
        "/sessions/{session_id}/paths/translate:batch",
        response_model=BatchPathTranslationResponse,
    )
    def translate_path_batch(
        session_id: str,
        request: BatchPathTranslationRequest,
    ) -> BatchPathTranslationResponse:
        try:
            return BatchPathTranslationResponse(
                session_id=session_id,
                results=[
                    session_manager.translate_path(session_id, translation_request)
                    for translation_request in request.requests
                ],
            )
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/sessions/{session_id}/camera", response_model=CameraState)
    def get_camera(session_id: str) -> CameraState:
        try:
            return session_manager.get_camera(session_id)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/sessions/{session_id}/camera", response_model=CameraState)
    def set_camera(session_id: str, camera: CameraState) -> CameraState:
        try:
            return session_manager.set_camera(session_id, camera)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/sessions/{session_id}/render", response_model=RenderResponse)
    def render_session(
        session_id: str,
        request: RenderRequest,
    ) -> RenderResponse:
        try:
            return session_manager.render_session(session_id, request)
        except Exception as error:
            raise _http_error(error) from error

    @app.post(
        "/sessions/{session_id}/render-frames", response_model=RenderFramesResponse
    )
    def render_session_frames(
        session_id: str,
        request: RenderFramesRequest,
    ) -> RenderFramesResponse:
        try:
            return session_manager.render_session_frames(session_id, request)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/sessions/{session_id}/renders/{filename:path}")
    def get_render_artifact(session_id: str, filename: str) -> FileResponse:
        try:
            artifact_path = session_manager.render_artifact_path(session_id, filename)
        except Exception as error:
            raise _http_error(error) from error
        media_type = (
            mimetypes.guess_type(artifact_path.name)[0] or "application/octet-stream"
        )
        return FileResponse(
            artifact_path,
            media_type=media_type,
            filename=artifact_path.name,
        )

    @app.post("/sessions/{session_id}/pick", response_model=PickResponse)
    def pick_session(
        session_id: str,
        request: PickRequest,
    ) -> PickResponse:
        try:
            return session_manager.pick_session(session_id, request)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/sessions/{session_id}/screenshot")
    def get_screenshot(
        session_id: str,
        width: int = Query(default=1024, ge=1, le=MAX_SCREENSHOT_GET_DIMENSION),
        height: int = Query(default=768, ge=1, le=MAX_SCREENSHOT_GET_DIMENSION),
    ) -> FileResponse:
        try:
            result = session_manager.render_session(
                session_id,
                RenderRequest(width=width, height=height),
            )
        except Exception as error:
            raise _http_error(error) from error
        return FileResponse(
            result.image_path,
            media_type="image/png",
            filename=f"{session_id}.png",
        )

    return app


app = LazyASGIApp(create_app)


def main() -> None:
    """Run the service with uvicorn."""
    host = (
        os.environ.get("CONTENT_WORKBENCH_HOST")
        or os.environ.get("SCENE_INSPECTOR_HOST")
        or os.environ.get("RSI_HOST")
        or "127.0.0.1"
    )
    port = int(
        _first_nonempty_env(
            ("CONTENT_WORKBENCH_PORT", "SCENE_INSPECTOR_PORT", "RSI_PORT"),
            default="8088",
        )
    )
    uvicorn.run("content_workbench.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
