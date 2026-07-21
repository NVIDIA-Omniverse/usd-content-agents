# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateful scene inspection session manager."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import zipfile
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .correspondence import SceneOptimizerPathMap
from .env import first_nonempty_env
from .material_apply_adapter import run_material_apply_task as _run_material_apply_task
from .models import (
    ArtifactState,
    CameraState,
    CommandName,
    CommandRequest,
    CommandResponse,
    CommandStatus,
    CreateSessionRequest,
    DiagnosticsResponse,
    LoadSceneRequest,
    MaterialApplyRequest,
    MaterialApplyResponse,
    MaterialAssignmentRecord,
    MaterialAssignmentsResponse,
    MaterialBindingResponse,
    MaterialOverride,
    OptimizationState,
    PathTranslationRequest,
    PathTranslationResponse,
    PickRequest,
    PickResponse,
    RenderFramesRequest,
    RenderFramesResponse,
    RenderQuality,
    RenderRequest,
    RenderResponse,
    SceneRestoreRequest,
    SceneRestoreResponse,
    SceneSnapshotCandidate,
    SceneSnapshotNode,
    SceneSnapshotRequest,
    SceneSnapshotResponse,
    SessionResponse,
    SessionStatus,
    ViewportState,
    ViewState,
)
from .renderer_worker import (
    RENDER_PRODUCT_PATH,
    IsolatedOvRTXRendererWorker,
    OvRTXRendererWorker,
    validate_aov_name,
)
from .usd_queries import UsdSceneQueries

PREVIEW_SCENE_RETENTION_COUNT = 16
MATERIAL_LIBRARY_ROOTS_ENV = "CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS"
OUTPUT_ROOTS_ENV = "CONTENT_WORKBENCH_OUTPUT_ROOTS"
CLOSE_SESSION_WORKSPACE_TIMEOUT_SECONDS = 30.0
DEFAULT_RENDER_OPERATION_TIMEOUT_SECONDS = 300.0
DEFAULT_MATERIAL_APPLY_OPERATION_TIMEOUT_SECONDS = 300.0
MIN_CAMERA_DISTANCE = 1.0e-3
MAX_CAMERA_DISTANCE = 1.0e9
MAX_DOLLY_EXPONENT = 64.0
USD_ASSET_PATH_UNSAFE_CHARS_RE = re.compile(r"[@\r\n]|[\x00-\x1f]")
logger = logging.getLogger(__name__)


class SceneSession(BaseModel):
    """Internal session object."""

    session_id: str
    status: SessionStatus = SessionStatus.CREATED
    scene_path: str | None = None
    source_scene_path: str | None = None
    inspection_scene_path: str | None = None
    clear_materials: bool = False
    root_prim_path: str | None = None
    error: str | None = None
    viewport: ViewportState
    view: ViewState
    artifacts: ArtifactState = Field(default_factory=ArtifactState)
    optimization: OptimizationState = Field(default_factory=OptimizationState)


def resolve_local_scene_path(raw_path: str) -> Path:
    """Resolve and validate a local USD path."""
    parsed = urlparse(raw_path)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(f"Only local file paths are supported, got: {raw_path}")

    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"Only local file URLs are supported, got: {raw_path}")
        path = Path(unquote(parsed.path))
    else:
        path = Path(raw_path).expanduser()

    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"USD scene does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"USD scene path is not a file: {resolved}")
    if resolved.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise ValueError(
            f"USD scene must end with .usd, .usda, .usdc, or .usdz: {resolved}"
        )
    return resolved


def _validate_session_id(session_id: str) -> None:
    try:
        parsed = UUID(session_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("session_id must be a canonical UUID") from exc
    if str(parsed) != session_id:
        raise ValueError("session_id must be a canonical UUID")


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.1f", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Invalid %s=%r; using %.1f", name, raw, default)
        return default
    return value


class SessionManager:
    """In-memory manager for local scene inspection sessions."""

    def __init__(self, renderer: OvRTXRendererWorker | None = None) -> None:
        self._sessions: dict[str, SceneSession] = {}
        self._queries: dict[str, UsdSceneQueries] = {}
        self._source_queries: dict[str, UsdSceneQueries] = {}
        self._path_maps: dict[str, SceneOptimizerPathMap] = {}
        self._pinned_preview_paths: set[Path] = set()
        self._active_workspace_ops: dict[str, int] = {}
        self._lock = threading.RLock()
        self._workspace_condition = threading.Condition(self._lock)
        workspace_root = first_nonempty_env(
            (
                "CONTENT_WORKBENCH_WORKSPACE_DIR",
                # Legacy aliases retained for transition from scene inspector.
                "SCENE_INSPECTOR_WORKSPACE_DIR",
                "RSI_WORKSPACE_DIR",
            ),
            default="",
        )
        self._workspace_root = _prepare_workspace_root(
            Path(workspace_root) if workspace_root else _default_workspace_root()
        )
        self._output_roots = _output_roots_from_env()
        self._render_timeout_seconds = _positive_float_env(
            "CONTENT_WORKBENCH_RENDER_TIMEOUT_SECONDS",
            DEFAULT_RENDER_OPERATION_TIMEOUT_SECONDS,
        )
        self._renderer = renderer or IsolatedOvRTXRendererWorker(
            log_file_path=str(self._workspace_root / "ovrtx.log"),
            operation_timeout_seconds=self._render_timeout_seconds,
        )
        self._material_apply_timeout_seconds = _positive_float_env(
            "CONTENT_WORKBENCH_MATERIAL_APPLY_TIMEOUT_SECONDS",
            DEFAULT_MATERIAL_APPLY_OPERATION_TIMEOUT_SECONDS,
        )
        self._renderer_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="content-workbench-render",
        )
        self._material_apply_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="content-workbench-material-apply",
        )

    @property
    def active_session_count(self) -> int:
        """Return the number of non-closed sessions."""
        with self._lock:
            return sum(
                1
                for session in self._sessions.values()
                if session.status != SessionStatus.CLOSED
            )

    @property
    def output_roots(self) -> list[Path]:
        """Return configured roots for caller-selected durable outputs."""
        return list(self._output_roots)

    def create_session(self, request: CreateSessionRequest) -> SessionResponse:
        """Create a session and optionally load an initial scene."""
        session = SceneSession(
            session_id=str(uuid4()),
            viewport=ViewportState(width=request.width, height=request.height),
            view=ViewState(),
            artifacts=ArtifactState(),
        )
        workspace = self._workspace_for(session.session_id)
        session.artifacts.workspace_dir = str(workspace)
        with self._lock:
            self._sessions[session.session_id] = session

        if request.scene_path:
            try:
                self.load_scene(
                    session.session_id,
                    LoadSceneRequest(
                        scene_path=request.scene_path,
                        optimize=request.optimize,
                        optimizer_backend=request.optimizer_backend,
                        flatten_prototypes=request.flatten_prototypes,
                        enable_deinstance=request.enable_deinstance,
                        enable_split=request.enable_split,
                        enable_deduplicate=request.enable_deduplicate,
                        clear_materials=request.clear_materials,
                        optimization_config=request.optimization_config,
                    ),
                )
            except Exception:
                with self._lock:
                    self._sessions.pop(session.session_id, None)
                shutil.rmtree(workspace, ignore_errors=True)
                raise
        return self.get_session(session.session_id)

    def get_session(self, session_id: str) -> SessionResponse:
        """Return public session state."""
        with self._lock:
            session = self._require_session(session_id)
            return SessionResponse(**session.model_dump())

    def close_session(self, session_id: str) -> SessionResponse:
        """Close a session and release query state."""
        with self._lock:
            session = self._require_session(session_id)
            self._wait_for_workspace_operations(session_id, action="closing")
            session.status = SessionStatus.CLOSED
            queries = self._queries.pop(session_id, None)
            source_queries = self._source_queries.pop(session_id, None)
            self._path_maps.pop(session_id, None)
            workspace_path = self._workspace_path(session_id)
            preview_dir = (workspace_path / "previews").resolve()
            self._pinned_preview_paths = {
                path
                for path in self._pinned_preview_paths
                if path.parent != preview_dir
            }
            response = SessionResponse(**session.model_dump())
            self._sessions.pop(session_id, None)
            release_renderer = not self._sessions
        try:
            for query_set in (queries, source_queries):
                if query_set is not None:
                    query_set.close()
            if release_renderer:
                shutdown = getattr(self._renderer, "shutdown", None)
                if shutdown is not None:
                    try:
                        if isinstance(self._renderer, OvRTXRendererWorker):
                            shutdown(timeout_seconds=self._render_timeout_seconds)
                        else:
                            shutdown()
                    except Exception as exc:
                        logger.warning("Unable to shut down idle renderer: %s", exc)
        finally:
            shutil.rmtree(workspace_path, ignore_errors=True)
        return response

    def load_scene(self, session_id: str, request: LoadSceneRequest) -> SessionResponse:
        """Load or reload a local USD scene for a session."""
        with self._lock:
            self._require_session(session_id)
        source_scene_path = resolve_local_scene_path(request.scene_path)
        base_inspection_scene_path = source_scene_path
        optimization = OptimizationState(
            enabled=False,
            status="disabled",
            source_scene_path=str(source_scene_path),
            inspection_scene_path=str(source_scene_path),
        )
        path_map = SceneOptimizerPathMap()
        if request.optimize:
            base_inspection_scene_path, optimization, path_map = self._optimize_scene(
                session_id,
                source_scene_path,
                request.resolved_optimization_config(),
            )
        inspection_scene_path = base_inspection_scene_path
        material_cleared_scene_path: Path | None = None
        if request.clear_materials:
            material_cleared_scene_path = self._material_cleared_scene_path(
                session_id,
                base_inspection_scene_path,
            )
            _export_material_cleared_stage(
                source_path=base_inspection_scene_path,
                output_path=material_cleared_scene_path,
            )
            inspection_scene_path = material_cleared_scene_path
            optimization.inspection_scene_path = str(inspection_scene_path)

        queries = UsdSceneQueries(inspection_scene_path)
        source_queries: UsdSceneQueries | None = None
        previous_queries: UsdSceneQueries | None = None
        previous_source_queries: UsdSceneQueries | None = None
        installed = False
        try:
            source_queries = UsdSceneQueries(source_scene_path)
            root_prim_path = queries.root_prim_path()
            root_bounds = queries.get_bounds(root_prim_path)
            with self._lock:
                session = self._require_session(session_id)
                self._wait_for_workspace_operations(session_id, action="reloading")
                previous_queries = self._queries.get(session_id)
                previous_source_queries = self._source_queries.get(session_id)
                session.scene_path = str(inspection_scene_path)
                session.source_scene_path = str(source_scene_path)
                session.inspection_scene_path = str(inspection_scene_path)
                session.clear_materials = request.clear_materials
                session.root_prim_path = root_prim_path
                session.status = SessionStatus.READY
                session.error = None
                session.optimization = optimization
                session.view = ViewState()
                session.artifacts.preview_scene_path = None
                session.artifacts.optimized_scene_path = (
                    str(base_inspection_scene_path) if request.optimize else None
                )
                session.artifacts.material_cleared_scene_path = (
                    str(material_cleared_scene_path)
                    if material_cleared_scene_path is not None
                    else None
                )
                session.artifacts.optimization_metadata_path = (
                    optimization.metadata_path
                )
                session.artifacts.last_render_path = None
                session.artifacts.last_render_camera_json_path = None
                session.artifacts.last_apply_output_path = None
                session.artifacts.last_apply_assignments_path = None
                session.artifacts.last_apply_predictions_path = None
                self._queries[session_id] = queries
                self._source_queries[session_id] = source_queries
                self._path_maps[session_id] = path_map
                session.view.camera = _camera_state_from_bounds(
                    root_bounds,
                    direction="+x-y+z",
                    margin=1.25,
                    width=session.viewport.width,
                    height=session.viewport.height,
                    focus_path=root_prim_path,
                )
                session.viewport.mode = "still_render"
                response = SessionResponse(**session.model_dump())
                installed = True
        except Exception:
            if not installed:
                queries.close()
                if source_queries is not None:
                    source_queries.close()
            raise
        for query_set in (previous_queries, previous_source_queries):
            if query_set is not None:
                query_set.close()
        return response

    def get_tree(self, session_id: str, prim_path: str | None = None):
        """Return direct children for a prim."""
        with self._lock:
            session = self._require_ready_session(session_id)
            queries = self._require_queries(session_id)
            return queries.get_children(prim_path or session.root_prim_path)

    def get_properties(self, session_id: str, prim_path: str):
        """Return properties for a prim."""
        with self._lock:
            self._require_ready_session(session_id)
            return self._require_queries(session_id).get_properties(prim_path)

    def get_material_binding(
        self, session_id: str, prim_path: str
    ) -> MaterialBindingResponse:
        """Return material binding state, enriched with material override state."""
        with self._lock:
            session = self._require_ready_session(session_id)
            binding = self._require_queries(session_id).get_material_binding(prim_path)
            binding.material_override = self._material_override_for(
                session,
                prim_path,
                space="inspection",
            )
            return binding

    def get_material_assignments(self, session_id: str) -> MaterialAssignmentsResponse:
        """Return current Workbench material assignment state."""
        with self._lock:
            session = self._require_ready_session(session_id)
            source_scene_path = (
                Path(session.source_scene_path) if session.source_scene_path else None
            )
            return MaterialAssignmentsResponse(
                session_id=session_id,
                assignments=[
                    _material_assignment_record(
                        override,
                        source_scene_path=source_scene_path,
                    )
                    for override in session.view.material_overrides
                ],
            )

    def apply_material_assignments(
        self,
        session_id: str,
        request: MaterialApplyRequest,
        *,
        overrides: list[MaterialOverride] | None = None,
    ) -> MaterialApplyResponse:
        """Apply accepted Workbench material assignments to an output USD/USDZ."""
        with self._lock:
            session = self._require_ready_session(session_id)
            if session.source_scene_path is None:
                raise ValueError(f"Session has no source scene path: {session_id}")
            source_scene_path = Path(session.source_scene_path)
            inspection_scene_path = (
                Path(session.inspection_scene_path)
                if session.inspection_scene_path
                else None
            )
            overrides = [
                override.model_copy(deep=True)
                for override in (
                    session.view.material_overrides if overrides is None else overrides
                )
            ]
            if not overrides:
                raise ValueError("No Workbench material assignments to apply")
            apply_dir = self._workspace_for(session_id) / "authoring" / uuid4().hex
            self._begin_workspace_operation(session_id)

        try:
            output_path = _resolve_material_apply_output_path(
                request.output_usd_path,
                source_scene_path=source_scene_path,
                inspection_scene_path=inspection_scene_path,
                overwrite=request.overwrite,
                allowed_output_roots=self._allowed_output_roots_for_session(session_id),
            )
            apply_dir.mkdir(parents=True, exist_ok=True)
            payload = _build_material_apply_payload(
                overrides,
                source_scene_path=source_scene_path,
                fail_on_invalid_assignment=request.fail_on_invalid_assignment,
            )
            if not payload["prediction_records"]:
                raise ValueError("No durable material assignments remain to apply")

            assignments_path = apply_dir / "assignments.json"
            predictions_path = apply_dir / "predictions.jsonl"
            assignments_path.write_text(
                json.dumps(
                    {
                        "session_id": session_id,
                        "input_usd_path": str(source_scene_path),
                        "output_usd_path": str(output_path),
                        "output_mode": request.output_mode,
                        "material_profile": request.material_profile,
                        "assignments": payload["assignment_records"],
                        "warnings": payload["warnings"],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            predictions_path.write_text(
                "".join(
                    json.dumps(record, sort_keys=True) + "\n"
                    for record in payload["prediction_records"]
                ),
                encoding="utf-8",
            )

            task_output_path = _material_apply_task_output_path(
                apply_dir / output_path.name
            )
            context = {
                "input_usd_path": str(source_scene_path),
                "output_usd_path": str(task_output_path),
                "predictions_path": str(predictions_path),
                "resolved_materials": payload["resolved_materials"],
                "is_library_based_mapping": True,
                "material_library_path": str(payload["material_library_path"]),
                "layer_only": request.output_mode == "layer",
                "flatten_output": request.output_mode == "flattened",
                "skip_instance_check": request.skip_instance_check,
                "material_profile": request.material_profile,
                "allow_empty_predictions": False,
                "fail_on_unknown_material": True,
            }
            try:
                result_context = _run_material_apply_task(
                    context,
                    executor=self._material_apply_executor,
                    timeout_seconds=self._material_apply_timeout_seconds,
                )
                if not task_output_path.exists() or not task_output_path.is_file():
                    raise RuntimeError(
                        f"Material apply did not produce output USD: {task_output_path}"
                    )
                assignment_stats = dict(result_context.get("assignment_stats") or {})
                bound_source_paths, unbound_source_paths = (
                    _material_apply_bound_source_paths(
                        assignment_stats=assignment_stats,
                        prediction_records=payload["prediction_records"],
                        fail_on_invalid_assignment=(request.fail_on_invalid_assignment),
                    )
                )
                publish_path = _prepare_material_apply_output_for_publish(
                    task_output_path,
                    output_path=output_path,
                    staging_dir=apply_dir,
                )
                try:
                    _secure_publish_staged_output(
                        publish_path,
                        output_path=output_path,
                        allowed_output_roots=(
                            self._allowed_output_roots_for_session(session_id)
                        ),
                        overwrite=request.overwrite,
                    )
                finally:
                    publish_path.unlink(missing_ok=True)
            finally:
                task_output_path.unlink(missing_ok=True)
            apply_warnings = list(payload["warnings"])
            if unbound_source_paths:
                apply_warnings.append(
                    "Material apply left requested prim targets unbound: "
                    f"{unbound_source_paths}"
                )
            response = MaterialApplyResponse(
                session_id=session_id,
                status="applied",
                input_usd_path=str(source_scene_path),
                output_usd_path=str(output_path),
                output_mode=request.output_mode,
                material_profile=str(
                    result_context.get("material_profile", request.material_profile)
                ),
                assignments_path=str(assignments_path),
                predictions_path=str(predictions_path),
                material_library_path=str(payload["material_library_path"]),
                materials_applied=dict(result_context.get("materials_applied") or {}),
                assignment_stats=assignment_stats,
                applied_assignment_count=len(bound_source_paths),
                applied_source_prim_paths=bound_source_paths,
                unbound_source_prim_paths=unbound_source_paths,
                skipped_assignment_count=payload["skipped_assignment_count"],
                warnings=apply_warnings,
            )
            with self._lock:
                session = self._require_ready_session(session_id)
                session.artifacts.last_apply_output_path = str(output_path)
                session.artifacts.last_apply_assignments_path = str(assignments_path)
                session.artifacts.last_apply_predictions_path = str(predictions_path)
            return response
        finally:
            with self._lock:
                self._end_workspace_operation(session_id)

    def restore_scene(
        self,
        session_id: str,
        request: SceneRestoreRequest,
    ) -> SceneRestoreResponse:
        """Restore current editable session state into durable scene artifacts.

        This preview implementation restores material edits back to the source
        USD through the same source-space mapping used by durable material apply.
        View-only edits such as hide/isolate are retained in the preview artifact
        and reported as warnings until generic edit transactions land.
        """
        with self._lock:
            session = self._require_ready_session(session_id)
            if session.source_scene_path is None:
                raise ValueError(f"Session has no source scene path: {session_id}")
            source_scene_path = Path(session.source_scene_path)
            inspection_scene_path = (
                Path(session.inspection_scene_path)
                if session.inspection_scene_path
                else None
            )
            material_override_count = len(session.view.material_overrides)
            hidden_prims = list(session.view.hidden_prims)
            isolated_prims = list(session.view.isolated_prims)
            unresolved_mappings = self._restore_unresolved_mappings(
                session.view.material_overrides
            )
            overrides = [
                override.model_copy(deep=True)
                for override in session.view.material_overrides
            ]
            durable_material_overrides = [
                override
                for override in overrides
                if override.source_prim_paths
                and _is_durable_material_override(
                    override,
                    source_scene_path=source_scene_path,
                )
            ]

        preview_scene_path: Path | None = None
        if request.include_preview_artifact:
            preview_scene_path = self.export_preview_scene(session_id)

        warnings: list[str] = []
        if hidden_prims:
            warnings.append(
                "Visibility hide edits are included in the preview artifact but "
                "are not projected to source output in this preview build."
            )
        if isolated_prims:
            warnings.append(
                "Isolation edits are included in the preview artifact but are not "
                "projected to source output in this preview build."
            )
        if unresolved_mappings:
            warnings.append(
                "Some material edits could not be resolved to source prim paths."
            )
        if material_override_count and not durable_material_overrides:
            warnings.append(
                "Material overrides are preview-only and are included in the "
                "preview artifact but are not projected to source output in this "
                "preview build."
            )

        material_apply: MaterialApplyResponse | None = None
        output_usd_path = request.output_usd_path
        restored_edit_count = 0
        restored_source_prim_paths: list[str] = []
        unbound_source_prim_paths: list[str] = []
        if durable_material_overrides:
            if output_usd_path is None:
                output_usd_path = str(
                    self._default_restored_scene_path(
                        session_id,
                        source_scene_path=source_scene_path,
                    )
                )
            material_apply = self.apply_material_assignments(
                session_id,
                MaterialApplyRequest(
                    output_usd_path=output_usd_path,
                    output_mode=request.output_mode,
                    material_profile=request.material_profile,
                    skip_instance_check=request.skip_instance_check,
                    fail_on_invalid_assignment=request.fail_on_invalid_assignment,
                    overwrite=request.overwrite,
                ),
                overrides=durable_material_overrides,
            )
            output_usd_path = material_apply.output_usd_path
            restored_edit_count = material_apply.applied_assignment_count
            restored_source_prim_paths = list(material_apply.applied_source_prim_paths)
            unbound_source_prim_paths = list(material_apply.unbound_source_prim_paths)
            warnings.extend(material_apply.warnings)
        elif output_usd_path:
            output_path = _resolve_material_apply_output_path(
                output_usd_path,
                source_scene_path=source_scene_path,
                inspection_scene_path=inspection_scene_path,
                overwrite=request.overwrite,
                allowed_output_roots=self._allowed_output_roots_for_session(session_id),
            )
            restore_dir = self._workspace_for(session_id) / "restore" / uuid4().hex
            restore_dir.mkdir(parents=True, exist_ok=True)
            _export_unchanged_scene(
                source_path=source_scene_path,
                output_path=output_path,
                output_mode=request.output_mode,
                staging_dir=restore_dir,
                allowed_output_roots=self._allowed_output_roots_for_session(session_id),
                overwrite=request.overwrite,
            )
            output_usd_path = str(output_path)
            warnings.append("No durable source-space edits were present to restore.")
        else:
            if not request.include_preview_artifact:
                raise ValueError(
                    "restore requires output_usd_path or include_preview_artifact "
                    "when no durable source-space edits are present"
                )
            warnings.append(
                "No durable source-space edits were present to restore; returning "
                "the preview artifact only."
            )

        return SceneRestoreResponse(
            session_id=session_id,
            status="restored" if output_usd_path else "preview_exported",
            source_scene_path=str(source_scene_path),
            inspection_scene_path=(
                str(inspection_scene_path) if inspection_scene_path else None
            ),
            preview_scene_path=str(preview_scene_path) if preview_scene_path else None,
            output_usd_path=output_usd_path,
            output_mode=request.output_mode if output_usd_path else None,
            restored_edit_count=restored_edit_count,
            restored_source_prim_paths=restored_source_prim_paths,
            unbound_source_prim_paths=unbound_source_prim_paths,
            unresolved_mappings=unresolved_mappings,
            warnings=warnings,
            material_apply=material_apply,
        )

    def get_diagnostics(self, session_id: str) -> DiagnosticsResponse:
        """Return offline diagnostics for a session scene."""
        with self._lock:
            self._require_ready_session(session_id)
            diagnostics = self._require_queries(session_id).diagnostics()
            return DiagnosticsResponse(session_id=session_id, diagnostics=diagnostics)

    def get_camera(self, session_id: str) -> CameraState:
        """Return the current agent-controllable camera state."""
        with self._lock:
            session = self._require_ready_session(session_id)
            return session.view.camera.model_copy(deep=True)

    def set_camera(self, session_id: str, camera: CameraState) -> CameraState:
        """Replace the current camera state."""
        with self._lock:
            session = self._require_ready_session(session_id)
            session.view.camera = _sanitize_camera(camera)
            return session.view.camera.model_copy(deep=True)

    def translate_path(
        self, session_id: str, request: PathTranslationRequest
    ) -> PathTranslationResponse:
        """Translate a prim path between source and inspection coordinate spaces."""
        with self._lock:
            session = self._require_ready_session(session_id)
            if request.source_space not in {"source", "inspection"}:
                raise ValueError("source_space must be source or inspection")
            if request.target_space not in {"source", "inspection"}:
                raise ValueError("target_space must be source or inspection")
            self._validate_prim_in_space(
                session_id,
                request.prim_path,
                space=request.source_space,
            )
            return self._path_translation_response(
                session,
                session_id,
                request.prim_path,
                source_space=request.source_space,
                target_space=request.target_space,
            )

    def snapshot_scene(
        self,
        session_id: str,
        request: SceneSnapshotRequest,
    ) -> SceneSnapshotResponse:
        """Return a one-call hierarchy/properties/material/path snapshot."""
        with self._lock:
            session = self._require_ready_session(session_id).model_copy(deep=True)
            queries = self._require_queries(session_id)
            path_map = self._path_maps.get(session_id) or SceneOptimizerPathMap()
            root_prim_path = (
                request.root_prim_path
                or session.root_prim_path
                or queries.root_prim_path()
            )
            if not isinstance(root_prim_path, str) or not root_prim_path.startswith(
                "/"
            ):
                raise ValueError(f"Invalid prim path: {root_prim_path!r}")
            if not queries.has_prim(root_prim_path):
                raise KeyError(f"Prim not found: {root_prim_path}")
            self._begin_workspace_operation(session_id)

        try:
            candidate_hint_paths: list[str] = []
            candidate_hints_truncated = False
            if request.include_candidate_hints:
                candidate_hint_budget = request.max_prims // 2
                candidate_hint_paths, candidate_hints_truncated = (
                    _snapshot_candidate_hint_paths(
                        queries,
                        root_prim_path,
                        max_paths=candidate_hint_budget,
                    )
                )
            tree_budget = max(1, request.max_prims - len(candidate_hint_paths))
            paths, nodes, truncated, property_cache = _snapshot_tree(
                queries,
                root_prim_path,
                max_prims=tree_budget,
            )
            candidate_extra_paths: list[str] = []
            if request.include_candidate_hints:
                path_set = set(paths)
                candidate_extra_paths = [
                    path for path in candidate_hint_paths if path not in path_set
                ]
            candidate_extra_set = set(candidate_extra_paths)
            properties = (
                [
                    _snapshot_properties(queries, property_cache, prim_path)
                    for prim_path in [*paths, *candidate_extra_paths]
                ]
                if request.include_properties or request.include_candidate_hints
                else []
            )
            visibility_properties = properties
            if request.include_candidate_hints:
                visibility_properties = [
                    *properties,
                    *[
                        _snapshot_properties(queries, property_cache, ancestor_path)
                        for ancestor_path in _snapshot_ancestor_paths(
                            queries, root_prim_path
                        )
                        if ancestor_path not in paths
                    ],
                ]
            material_bindings = (
                [
                    _snapshot_material_binding(session, queries, prim_path)
                    for prim_path in [*paths, *candidate_extra_paths]
                ]
                if request.include_material_bindings or request.include_candidate_hints
                else []
            )
            path_translations = (
                [
                    self._path_translation_response(
                        session,
                        session_id,
                        prim_path,
                        source_space="inspection",
                        path_map=path_map,
                    )
                    for prim_path in paths
                ]
                if request.include_path_translations or request.include_candidate_hints
                else []
            )
            candidate_path_translations = path_translations
            if request.include_candidate_hints and candidate_extra_paths:
                candidate_path_translations = [
                    *path_translations,
                    *[
                        self._path_translation_response(
                            session,
                            session_id,
                            prim_path,
                            source_space="inspection",
                            path_map=path_map,
                        )
                        for prim_path in candidate_extra_paths
                    ],
                ]
            candidates = (
                _snapshot_candidates(
                    session_id=session_id,
                    properties=properties,
                    visibility_properties=visibility_properties,
                    material_bindings=material_bindings,
                    path_translations=candidate_path_translations,
                )
                if request.include_candidate_hints
                else []
            )
            properties_out = (
                [
                    record
                    for record in properties
                    if record.prim_path not in candidate_extra_set
                ]
                if request.include_properties
                else []
            )
            material_bindings_out = (
                [
                    record
                    for record in material_bindings
                    if record.prim_path not in candidate_extra_set
                ]
                if request.include_material_bindings
                else []
            )
            path_translations_out = (
                path_translations if request.include_path_translations else []
            )
            return SceneSnapshotResponse(
                session_id=session_id,
                root_prim_path=root_prim_path,
                source_scene_path=session.source_scene_path,
                inspection_scene_path=session.inspection_scene_path,
                optimization=session.optimization.model_copy(deep=True),
                paths=paths,
                nodes=nodes,
                properties=properties_out,
                material_bindings=material_bindings_out,
                path_translations=path_translations_out,
                candidates=candidates,
                excluded_non_candidates=[],
                summary={
                    "prim_count": len(paths),
                    "node_count": len(nodes),
                    "property_count": len(properties_out),
                    "material_binding_count": len(material_bindings_out),
                    "path_translation_count": len(path_translations_out),
                    "candidate_count": len(candidates),
                    "ambiguous_translation_count": sum(
                        1 for item in candidate_path_translations if item.ambiguous
                    ),
                    "truncated": truncated or candidate_hints_truncated,
                    "max_prims": request.max_prims,
                    "snapshot_path_count": len(paths) + len(candidate_extra_paths),
                    "candidate_hint_path_count": len(candidate_hint_paths),
                    "candidate_hint_extra_path_count": len(candidate_extra_paths),
                    "candidate_hints_truncated": candidate_hints_truncated,
                },
            )
        finally:
            with self._lock:
                self._end_workspace_operation(session_id)

    def render_session(self, session_id: str, request: RenderRequest) -> RenderResponse:
        """Render the current session preview using the owned OvRTX renderer."""
        preview_scene_path = self.export_preview_scene(session_id)
        with self._lock:
            session = self._require_ready_session(session_id)
            render_dir = self._workspace_for(session_id) / "renders"
            render_dir.mkdir(parents=True, exist_ok=True)
            stamp = f"{int(time.time() * 1000)}-{uuid4().hex}"
            image_path = render_dir / f"render-{stamp}.png"
            camera_json_path = render_dir / f"render-{stamp}.json"
            queries = self._require_queries(session_id)
            camera = session.view.camera
            selected_mesh_paths = self._selection_mesh_paths(session_id)
            active_aov = session.view.active_aov
            render_mode, num_updates = _render_settings_from_request(request)
            if not request.use_session_camera or request.focus or request.direction:
                focus = (
                    request.focus or session.root_prim_path or queries.root_prim_path()
                )
                bounds = queries.get_bounds(focus)
                camera = _camera_state_from_bounds(
                    bounds,
                    direction=request.direction or "+x-y+z",
                    margin=request.margin,
                    width=request.width,
                    height=request.height,
                    focus_path=focus,
                )
            camera_transform = _camera_transform_from_state(camera)
            camera_json_payload = (
                {
                    "camera_path": "/Session/Cameras/Main",
                    "camera_state": camera.model_dump(),
                    "camera_world_transform": camera_transform,
                    "image_width": request.width,
                    "image_height": request.height,
                    "use_session_camera": request.use_session_camera,
                    "direction": request.direction,
                    "margin": request.margin,
                    "render_quality": request.render_quality,
                    "ovrtx_render_mode": render_mode,
                    "ovrtx_num_sensor_updates": num_updates,
                    "hdri_light": request.hdri_light,
                    "dome_light": request.dome_light,
                    "distant_light": request.distant_light,
                    "active_aov": active_aov,
                }
                if request.save_camera_json
                else None
            )
            self._pinned_preview_paths.add(preview_scene_path.resolve())
            self._begin_workspace_operation(session_id)

        render_succeeded = False
        try:
            elapsed = self._run_renderer_call(
                lambda: self._renderer.render(
                    scene_path=preview_scene_path,
                    output_path=image_path,
                    width=request.width,
                    height=request.height,
                    camera_transform=camera_transform,
                    num_updates=num_updates,
                    render_mode=render_mode,
                    hdri_light=request.hdri_light,
                    dome_light=request.dome_light,
                    distant_light=request.distant_light,
                    selected_prim_paths=selected_mesh_paths,
                    active_aov=active_aov,
                    lock_timeout_seconds=self._render_timeout_seconds,
                ),
                timeout_label="OvRTX render",
            )
            render_succeeded = True
            if camera_json_payload is not None:
                temp_camera_json_path = camera_json_path.with_suffix(".json.tmp")
                temp_camera_json_path.write_text(
                    json.dumps(camera_json_payload, indent=2),
                    encoding="utf-8",
                )
                temp_camera_json_path.replace(camera_json_path)
            with self._lock:
                session = self._require_ready_session(session_id)
                session.artifacts.preview_scene_path = str(preview_scene_path)
                session.artifacts.last_render_path = str(image_path)
                session.artifacts.last_render_camera_json_path = (
                    str(camera_json_path) if request.save_camera_json else None
                )
                return RenderResponse(
                    session_id=session_id,
                    status="success",
                    preview_scene_path=str(preview_scene_path),
                    image_path=str(image_path),
                    image_url=f"/sessions/{session_id}/renders/{image_path.name}",
                    camera_json_path=(
                        str(camera_json_path) if request.save_camera_json else None
                    ),
                    camera_json_url=(
                        f"/sessions/{session_id}/renders/{camera_json_path.name}"
                        if request.save_camera_json
                        else None
                    ),
                    render_product_path=RENDER_PRODUCT_PATH,
                    render_quality=request.render_quality,
                    ovrtx_render_mode=render_mode,
                    ovrtx_num_sensor_updates=num_updates,
                    active_aov=active_aov,
                    elapsed_seconds=elapsed,
                )
        except Exception:
            if request.save_camera_json:
                camera_json_path.with_suffix(".json.tmp").unlink(missing_ok=True)
                if not render_succeeded:
                    camera_json_path.unlink(missing_ok=True)
            raise
        finally:
            with self._lock:
                self._release_preview_workspace_operation(
                    session_id, preview_scene_path
                )

    def render_session_frames(
        self,
        session_id: str,
        request: RenderFramesRequest,
    ) -> RenderFramesResponse:
        """Render ordered frames from the current session preview or a supplied USD."""
        external_scene = request.scene_path is not None
        preview_scene_path = (
            resolve_local_scene_path(request.scene_path)
            if request.scene_path is not None
            else self.export_preview_scene(session_id)
        )
        preview_stage = _open_render_stage(preview_scene_path)
        fps = _stage_time_codes_per_second(preview_scene_path, stage=preview_stage)
        frame_numbers = _resolve_render_frame_numbers(
            preview_scene_path,
            request.frames,
            infer_from_scene=external_scene,
            max_duration_seconds=request.max_duration_seconds,
            fps=fps,
            stage=preview_stage,
        )
        if not frame_numbers:
            raise ValueError("frames must select at least one frame")
        with self._lock:
            session = self._require_ready_session(session_id)
            artifact_urls_enabled = request.output_dir is None
            if request.output_dir is None:
                render_dir = self._workspace_for(session_id) / "renders"
                stamp = f"{int(time.time() * 1000)}-{uuid4().hex}"
                frames_dir = render_dir / f"frames-{stamp}"
            else:
                frames_dir = self.resolve_agent_output_dir(
                    session_id,
                    request.output_dir,
                    default_subdir="renders",
                )
            frames_dir.mkdir(parents=True, exist_ok=True)
            queries = self._require_queries(session_id)
            selected_mesh_paths = (
                [] if external_scene else self._selection_mesh_paths(session_id)
            )
            active_aov = session.view.active_aov
            render_mode, num_updates = _render_settings_from_request(request)
            authored_camera_path = (
                _resolve_render_camera_path(
                    preview_scene_path,
                    request.camera_path,
                    stage=preview_stage,
                )
                if request.camera_path is not None
                else None
            )
            if (
                external_scene
                and authored_camera_path is None
                and not request.use_session_camera
            ):
                raise ValueError(
                    "external scene frame renders require camera_path or "
                    "use_session_camera"
                )
            focus = request.focus or session.root_prim_path or queries.root_prim_path()
            directions = request.directions or []
            if directions and len(directions) != len(frame_numbers):
                raise ValueError(
                    "directions length must match the selected frame count"
                )
            camera_transforms: list[list[list[float]]] = []
            cameras = []
            if authored_camera_path is None:
                if directions:
                    bounds = queries.get_bounds(focus)
                    for direction in directions:
                        cameras.append(
                            _camera_state_from_bounds(
                                bounds,
                                direction=direction,
                                margin=request.margin,
                                width=request.width,
                                height=request.height,
                                focus_path=focus,
                            )
                        )
                elif request.use_session_camera:
                    cameras = [session.view.camera for _frame in frame_numbers]
                else:
                    bounds = queries.get_bounds(focus)
                    for _frame in frame_numbers:
                        cameras.append(
                            _camera_state_from_bounds(
                                bounds,
                                direction="+x-y+z",
                                margin=request.margin,
                                width=request.width,
                                height=request.height,
                                focus_path=focus,
                            )
                        )
                camera_transforms = [
                    _camera_transform_from_state(camera) for camera in cameras
                ]
            frame_paths = [
                frames_dir / f"frame_{frame_number:04d}.png"
                for frame_number in frame_numbers
            ]
            camera_json_paths = [
                frames_dir / f"frame_{frame_number:04d}.json"
                for frame_number in frame_numbers
            ]
            frame_spec = request.frames or _format_frame_spec(frame_numbers)
            if authored_camera_path is not None:
                camera_json_payloads = [
                    {
                        "camera_path": authored_camera_path,
                        "camera_state": None,
                        "camera_world_transform": None,
                        "image_width": request.width,
                        "image_height": request.height,
                        "frame": frame_number,
                        "frames": frame_spec,
                        "direction": None,
                        "margin": request.margin,
                        "render_quality": request.render_quality,
                        "ovrtx_render_mode": render_mode,
                        "ovrtx_num_sensor_updates": num_updates,
                        "hdri_light": request.hdri_light,
                        "dome_light": request.dome_light,
                        "distant_light": request.distant_light,
                        "active_aov": active_aov,
                    }
                    for frame_number in frame_numbers
                ]
            else:
                camera_json_payloads = [
                    {
                        "camera_path": "/Session/Cameras/Main",
                        "camera_state": camera.model_dump(),
                        "camera_world_transform": transform,
                        "image_width": request.width,
                        "image_height": request.height,
                        "frame": frame_number,
                        "frames": frame_spec,
                        "direction": directions[index] if directions else None,
                        "margin": request.margin,
                        "render_quality": request.render_quality,
                        "ovrtx_render_mode": render_mode,
                        "ovrtx_num_sensor_updates": num_updates,
                        "hdri_light": request.hdri_light,
                        "dome_light": request.dome_light,
                        "distant_light": request.distant_light,
                        "active_aov": active_aov,
                    }
                    for index, (frame_number, camera, transform) in enumerate(
                        zip(frame_numbers, cameras, camera_transforms, strict=True)
                    )
                ]
            if not external_scene:
                self._pinned_preview_paths.add(preview_scene_path.resolve())
            self._begin_workspace_operation(session_id)

        render_succeeded = False
        camera_json_succeeded = not request.save_camera_json
        mp4_paths: list[Path] = []
        try:
            elapsed = self._run_renderer_call(
                lambda: self._renderer.render_frames(
                    scene_path=preview_scene_path,
                    output_paths=frame_paths,
                    width=request.width,
                    height=request.height,
                    frame_numbers=frame_numbers,
                    camera_transforms=camera_transforms,
                    camera_path=authored_camera_path,
                    fps=fps,
                    num_updates=num_updates,
                    render_mode=render_mode,
                    hdri_light=request.hdri_light,
                    dome_light=request.dome_light,
                    distant_light=request.distant_light,
                    selected_prim_paths=selected_mesh_paths,
                    active_aov=active_aov,
                    lock_timeout_seconds=self._render_timeout_seconds,
                ),
                timeout_label="OvRTX frame render",
            )
            render_succeeded = True
            if request.make_mp4:
                mp4_path = frames_dir / "render.mp4"
                if _write_mp4(frame_paths, mp4_path, fps):
                    mp4_paths.append(mp4_path)
            if request.save_camera_json:
                for camera_json_path, payload in zip(
                    camera_json_paths, camera_json_payloads, strict=True
                ):
                    temp_camera_json_path = camera_json_path.with_suffix(".json.tmp")
                    temp_camera_json_path.write_text(
                        json.dumps(payload, indent=2),
                        encoding="utf-8",
                    )
                    temp_camera_json_path.replace(camera_json_path)
                camera_json_succeeded = True
            with self._lock:
                session = self._require_ready_session(session_id)
                session.artifacts.preview_scene_path = str(preview_scene_path)
                session.artifacts.last_render_path = str(frame_paths[-1])
                session.artifacts.last_render_camera_json_path = (
                    str(camera_json_paths[-1]) if request.save_camera_json else None
                )
                return RenderFramesResponse(
                    session_id=session_id,
                    status="success",
                    preview_scene_path=str(preview_scene_path),
                    frame_paths=[str(path) for path in frame_paths],
                    frame_urls=[
                        f"/sessions/{session_id}/renders/{path.parent.name}/{path.name}"
                        for path in frame_paths
                    ]
                    if artifact_urls_enabled
                    else [],
                    camera_json_paths=[str(path) for path in camera_json_paths]
                    if request.save_camera_json
                    else [],
                    camera_json_urls=[
                        f"/sessions/{session_id}/renders/{path.parent.name}/{path.name}"
                        for path in camera_json_paths
                    ]
                    if request.save_camera_json and artifact_urls_enabled
                    else [],
                    mp4_paths=[str(path) for path in mp4_paths],
                    mp4_urls=[
                        f"/sessions/{session_id}/renders/{path.parent.name}/{path.name}"
                        for path in mp4_paths
                    ]
                    if artifact_urls_enabled
                    else [],
                    render_product_path=RENDER_PRODUCT_PATH,
                    render_quality=request.render_quality,
                    ovrtx_render_mode=render_mode,
                    ovrtx_num_sensor_updates=num_updates,
                    active_aov=active_aov,
                    elapsed_seconds=elapsed,
                )
        except Exception:
            for mp4_path in mp4_paths:
                mp4_path.unlink(missing_ok=True)
            if request.save_camera_json:
                for camera_json_path in camera_json_paths:
                    camera_json_path.with_suffix(".json.tmp").unlink(missing_ok=True)
                    if not render_succeeded or not camera_json_succeeded:
                        camera_json_path.unlink(missing_ok=True)
            if frames_dir.is_dir() and not any(frames_dir.iterdir()):
                frames_dir.rmdir()
            raise
        finally:
            with self._lock:
                if external_scene:
                    self._end_workspace_operation(session_id)
                else:
                    self._release_preview_workspace_operation(
                        session_id, preview_scene_path
                    )

    def render_artifact_path(self, session_id: str, filename: str) -> Path:
        """Return a validated render artifact path for API download."""
        if not filename:
            raise ValueError("render artifact filename must not be empty")
        relative_path = Path(filename)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("render artifact filename must stay inside renders")
        with self._lock:
            self._require_ready_session(session_id)
            render_dir = self._workspace_for(session_id) / "renders"
            artifact_path = (render_dir / relative_path).resolve()
        try:
            artifact_path.relative_to(render_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                "render artifact path escapes the session workspace"
            ) from exc
        if artifact_path.suffix.lower() not in {".png", ".json", ".mp4"}:
            raise ValueError("unsupported render artifact type")
        if not artifact_path.exists():
            raise FileNotFoundError(f"render artifact does not exist: {filename}")
        return artifact_path

    def pick_session(self, session_id: str, request: PickRequest) -> PickResponse:
        """Pick a viewport pixel using the current session camera."""
        preview_scene_path = self.export_preview_scene(session_id)
        with self._lock:
            session = self._require_ready_session(session_id)
            width = request.width or session.viewport.width
            height = request.height or session.viewport.height
            camera_transform = _camera_transform_from_state(session.view.camera)
            camera_focal_length = session.view.camera.focal_length
            camera_horizontal_aperture = session.view.camera.horizontal_aperture
            selected_mesh_paths = self._selection_mesh_paths(session_id)
            self._pinned_preview_paths.add(preview_scene_path.resolve())
            self._begin_workspace_operation(session_id)

        try:
            result = self._run_renderer_call(
                lambda: self._renderer.pick(
                    scene_path=preview_scene_path,
                    x=request.x,
                    y=request.y,
                    width=width,
                    height=height,
                    camera_transform=camera_transform,
                    camera_focal_length=camera_focal_length,
                    camera_horizontal_aperture=camera_horizontal_aperture,
                    num_updates=request.ovrtx_num_sensor_updates,
                    render_mode=request.ovrtx_render_mode,
                    selected_prim_paths=selected_mesh_paths,
                    lock_timeout_seconds=self._render_timeout_seconds,
                ),
                timeout_label="OvRTX pick",
            )
            with self._lock:
                session = self._require_ready_session(session_id)
                if request.update_selection:
                    session.view.selected_prims = _apply_selection_mode(
                        session.view.selected_prims,
                        result.prim_paths,
                        request.mode,
                    )
                return PickResponse(
                    session_id=session_id,
                    x=request.x,
                    y=request.y,
                    prim_paths=result.prim_paths,
                    selected_prims=session.view.selected_prims,
                    render_product_path=RENDER_PRODUCT_PATH,
                    elapsed_seconds=result.elapsed_seconds,
                )
        finally:
            with self._lock:
                self._release_preview_workspace_operation(
                    session_id, preview_scene_path
                )

    def _release_preview_workspace_operation(
        self, session_id: str, preview_scene_path: Path
    ) -> None:
        self._pinned_preview_paths.discard(preview_scene_path.resolve())
        if preview_scene_path.parent.exists():
            _prune_preview_scenes(
                preview_scene_path.parent,
                keep_path=preview_scene_path,
                protected_paths=self._pinned_preview_paths,
            )
        self._end_workspace_operation(session_id)

    def export_preview_scene(self, session_id: str) -> Path:
        """Export a session-owned USD with material edits applied."""
        with self._lock:
            session = self._require_ready_session(session_id)
            if session.scene_path is None:
                raise ValueError(f"Session has no scene path: {session_id}")

            preview_dir = self._workspace_for(session_id) / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / f"preview-{uuid4().hex}.usda"
            source_path = Path(session.scene_path)
            root_prim_path = session.root_prim_path
            overrides = self._material_overrides_for_inspection(session_id)
            hidden_prims = list(session.view.hidden_prims)
            isolated_prims = list(session.view.isolated_prims)
        try:
            _export_preview_stage(
                source_path=source_path,
                output_path=preview_path,
                root_prim_path=root_prim_path,
                overrides=overrides,
                hidden_prims=hidden_prims,
                isolated_prims=isolated_prims,
            )
        except Exception:
            preview_path.unlink(missing_ok=True)
            raise
        try:
            with self._lock:
                session = self._require_ready_session(session_id)
                session.artifacts.preview_scene_path = str(preview_path)
                _prune_preview_scenes(
                    preview_dir,
                    keep_path=preview_path,
                    protected_paths=self._pinned_preview_paths,
                )
                return preview_path
        except Exception:
            preview_path.unlink(missing_ok=True)
            raise

    def shutdown(self) -> None:
        """Release renderer, USD stages, and active session workspaces."""
        with self._lock:
            session_ids = list(self._sessions)
            workspaces = [
                self._workspace_path(session_id) for session_id in session_ids
            ]
            query_sets = [
                query_set
                for query_set in [
                    *self._queries.values(),
                    *self._source_queries.values(),
                ]
                if query_set is not None
            ]
            self._sessions.clear()
            self._queries.clear()
            self._source_queries.clear()
            self._path_maps.clear()
            self._pinned_preview_paths.clear()
            self._active_workspace_ops.clear()
            self._workspace_condition.notify_all()

        for query_set in query_sets:
            query_set.close()
        for workspace in workspaces:
            shutil.rmtree(workspace, ignore_errors=True)
        self._renderer_executor.shutdown(wait=False, cancel_futures=True)
        self._material_apply_executor.shutdown(wait=False, cancel_futures=True)
        shutdown = getattr(self._renderer, "shutdown", None)
        if shutdown is None:
            return
        try:
            if isinstance(self._renderer, OvRTXRendererWorker):
                shutdown(timeout_seconds=1.0)
            else:
                shutdown()
        except Exception as exc:
            logger.warning("Unable to shut down Content Workbench renderer: %s", exc)

    def apply_command(
        self, session_id: str, request: CommandRequest
    ) -> CommandResponse:
        """Apply a scene inspection command to session state."""
        with self._lock:
            session = self._require_session(session_id)
            if request.command in {
                CommandName.PICK,
                CommandName.FOCUS,
                CommandName.FRAME,
                CommandName.ORBIT,
                CommandName.PAN,
                CommandName.DOLLY,
                CommandName.SET_CAMERA,
                CommandName.SELECT,
                CommandName.HIDE,
                CommandName.SHOW,
                CommandName.ISOLATE,
                CommandName.MATERIAL_OVERRIDE,
                CommandName.CLEAR_MATERIAL_OVERRIDE,
            }:
                self._require_ready_session(session_id)

            message: str | None = None
            result = CommandStatus.SUCCESS

            if request.command == CommandName.PICK:
                prim_path = request.payload.get("prim_path")
                if not prim_path:
                    result = CommandStatus.UNSUPPORTED
                    message = "Use POST /sessions/{session_id}/pick for pixel picking, or pass prim_path to select directly."
                else:
                    self._validate_prim(session_id, prim_path)
                    session.view.selected_prims = [prim_path]
            elif request.command == CommandName.FOCUS:
                prim_path = request.payload.get("prim_path")
                if prim_path:
                    self._validate_prim(session_id, prim_path)
                    session.view.selected_prims = [prim_path]
                    self._frame_camera(
                        session,
                        self._require_queries(session_id),
                        prim_path,
                        margin=float(request.payload.get("margin", 1.25)),
                    )
            elif request.command == CommandName.FRAME:
                prim_path = request.payload.get("prim_path") or session.root_prim_path
                if not isinstance(prim_path, str):
                    raise ValueError("frame requires payload.prim_path")
                self._validate_prim(session_id, prim_path)
                self._frame_camera(
                    session,
                    self._require_queries(session_id),
                    prim_path,
                    margin=float(request.payload.get("margin", 1.25)),
                    direction=(
                        request.payload.get("direction")
                        if isinstance(request.payload.get("direction"), str)
                        else None
                    ),
                )
            elif request.command == CommandName.ORBIT:
                _orbit_camera(
                    session.view.camera,
                    yaw_delta_degrees=float(
                        request.payload.get(
                            "yaw_delta_degrees",
                            request.payload.get("yaw", 0.0),
                        )
                    ),
                    pitch_delta_degrees=float(
                        request.payload.get(
                            "pitch_delta_degrees",
                            request.payload.get("pitch", 0.0),
                        )
                    ),
                )
            elif request.command == CommandName.PAN:
                _pan_camera(
                    session.view.camera,
                    right_delta=float(
                        request.payload.get("right", request.payload.get("dx", 0.0))
                    ),
                    up_delta=float(
                        request.payload.get("up", request.payload.get("dy", 0.0))
                    ),
                    scale=float(request.payload.get("scale", 0.1)),
                )
            elif request.command == CommandName.DOLLY:
                _dolly_camera(
                    session.view.camera,
                    amount=float(request.payload.get("amount", 0.0)),
                    factor=request.payload.get("factor"),
                )
            elif request.command == CommandName.SET_CAMERA:
                session.view.camera = _sanitize_camera(
                    CameraState(**request.payload.get("camera", request.payload))
                )
            elif request.command == CommandName.SELECT:
                session.view.selected_prims = self._validated_paths(
                    session_id,
                    request.payload.get("paths", []),
                )
            elif request.command == CommandName.HIDE:
                session.view.hidden_prims = sorted(
                    set(session.view.hidden_prims)
                    | set(
                        self._validated_paths(
                            session_id, request.payload.get("paths", [])
                        )
                    )
                )
            elif request.command == CommandName.SHOW:
                show_paths = set(
                    self._validated_paths(session_id, request.payload.get("paths", []))
                )
                session.view.hidden_prims = [
                    path for path in session.view.hidden_prims if path not in show_paths
                ]
            elif request.command == CommandName.ISOLATE:
                session.view.isolated_prims = self._validated_paths(
                    session_id,
                    request.payload.get("paths", []),
                )
            elif request.command == CommandName.CLEAR_ISOLATION:
                session.view.isolated_prims = []
            elif request.command == CommandName.MATERIAL_OVERRIDE:
                prim_path = request.payload.get("prim_path")
                if not isinstance(prim_path, str):
                    raise ValueError("material_override requires payload.prim_path")
                space = str(request.payload.get("space", "inspection"))
                if space not in {"inspection", "source"}:
                    raise ValueError(
                        "material_override space must be source or inspection"
                    )
                self._validate_prim_in_space(session_id, prim_path, space=space)
                material = request.payload.get("material")
                if material is None:
                    raise ValueError("material_override requires payload.material")
                source_scene_path = session.source_scene_path or session.scene_path
                if source_scene_path is None:
                    raise ValueError("Session has no source scene path")
                material_spec = self._validated_material_spec(
                    material,
                    source_path=Path(source_scene_path),
                )
                previous_overrides = [
                    override.model_copy(deep=True)
                    for override in session.view.material_overrides
                ]
                try:
                    self._set_material_override(
                        session,
                        session_id,
                        prim_path,
                        material_spec,
                        space=space,
                        unbind_existing=bool(
                            request.payload.get("unbind_existing", True)
                        ),
                        remove_material_libraries=bool(
                            request.payload.get("remove_material_libraries", False)
                        ),
                    )
                    self.export_preview_scene(session_id)
                except Exception:
                    session.view.material_overrides = previous_overrides
                    raise
            elif request.command == CommandName.CLEAR_MATERIAL_OVERRIDE:
                prim_path = request.payload.get("prim_path")
                if not isinstance(prim_path, str):
                    raise ValueError(
                        "clear_material_override requires payload.prim_path"
                    )
                space = str(request.payload.get("space", "inspection"))
                if space not in {"inspection", "source"}:
                    raise ValueError(
                        "clear_material_override space must be source or inspection"
                    )
                self._validate_prim_in_space(session_id, prim_path, space=space)
                translation = self._translate_path(
                    session_id,
                    prim_path,
                    source_space=space,
                )
                source_paths = set(translation.source_paths)
                inspection_paths = set(translation.inspection_paths)
                previous_overrides = [
                    override.model_copy(deep=True)
                    for override in session.view.material_overrides
                ]
                try:
                    session.view.material_overrides = [
                        override
                        for override in session.view.material_overrides
                        if not _material_override_overlaps(
                            override,
                            source_paths=source_paths,
                            inspection_paths=inspection_paths,
                        )
                    ]
                    self.export_preview_scene(session_id)
                except Exception:
                    session.view.material_overrides = previous_overrides
                    raise
            elif request.command == CommandName.CLEAR_VISUAL_OVERRIDES:
                previous_selected = list(session.view.selected_prims)
                previous_hidden = list(session.view.hidden_prims)
                previous_isolated = list(session.view.isolated_prims)
                previous_overrides = [
                    override.model_copy(deep=True)
                    for override in session.view.material_overrides
                ]
                session.view.selected_prims = []
                session.view.hidden_prims = []
                session.view.isolated_prims = []
                session.view.material_overrides = []
                try:
                    self.export_preview_scene(session_id)
                except Exception:
                    session.view.selected_prims = previous_selected
                    session.view.hidden_prims = previous_hidden
                    session.view.isolated_prims = previous_isolated
                    session.view.material_overrides = previous_overrides
                    raise
            elif request.command == CommandName.RESET_VIEW:
                session.view.selected_prims = []
                if session.root_prim_path:
                    self._frame_camera(
                        session,
                        self._require_queries(session_id),
                        session.root_prim_path,
                        direction="+x-y+z",
                        margin=1.25,
                    )
            elif request.command == CommandName.CHANGE_AOV:
                aov = request.payload.get("aov")
                if not isinstance(aov, str) or not aov:
                    raise ValueError("change_aov requires payload.aov")
                session.view.active_aov = validate_aov_name(aov)
            else:
                result = CommandStatus.UNSUPPORTED
                message = f"Unsupported command: {request.command}"

            return CommandResponse(
                session_id=session_id,
                command=request.command,
                result=result,
                message=message,
                session=SessionResponse(**session.model_dump()),
            )

    def _require_session(self, session_id: str) -> SceneSession:
        _validate_session_id(session_id)
        session = self._sessions.get(session_id)
        if session is None or session.status == SessionStatus.CLOSED:
            raise KeyError(f"Session not found: {session_id}")
        return session

    def _require_ready_session(self, session_id: str) -> SceneSession:
        session = self._require_session(session_id)
        if session.status != SessionStatus.READY:
            raise ValueError(f"Session has no loaded scene: {session_id}")
        return session

    def _require_queries(self, session_id: str) -> UsdSceneQueries:
        queries = self._queries.get(session_id)
        if queries is None:
            raise ValueError(f"Session has no query worker: {session_id}")
        return queries

    def _require_source_queries(self, session_id: str) -> UsdSceneQueries:
        queries = self._source_queries.get(session_id)
        if queries is None:
            raise ValueError(f"Session has no source query worker: {session_id}")
        return queries

    def _validate_prim(self, session_id: str, prim_path: str) -> None:
        self._validate_prim_in_space(session_id, prim_path, space="inspection")

    def _validate_prim_in_space(
        self, session_id: str, prim_path: str, *, space: str
    ) -> None:
        if not isinstance(prim_path, str) or not prim_path.startswith("/"):
            raise ValueError(f"Invalid prim path: {prim_path!r}")
        queries = (
            self._require_source_queries(session_id)
            if space == "source"
            else self._require_queries(session_id)
        )
        if not queries.has_prim(prim_path):
            raise KeyError(f"Prim not found: {prim_path}")

    def _validated_paths(self, session_id: str, paths: object) -> list[str]:
        if not isinstance(paths, list):
            raise ValueError("payload.paths must be a list of prim paths")
        validated: list[str] = []
        for path in paths:
            if not isinstance(path, str):
                raise ValueError("payload.paths must contain only strings")
            self._validate_prim(session_id, path)
            validated.append(path)
        return validated

    @staticmethod
    def _validated_material_spec(
        material: object, *, source_path: Path
    ) -> dict[str, Any]:
        if not isinstance(material, dict):
            raise ValueError("material_override payload.material must be an object")
        for key in ("material_path", "binding_path"):
            raw_path = material.get(key)
            if raw_path is not None and (
                not isinstance(raw_path, str) or not raw_path.startswith("/")
            ):
                raise ValueError(
                    "material_override payload.material."
                    f"{key} must be an absolute USD prim path"
                )
        _material_color(material)
        _material_library_path(material, source_path=source_path)
        _material_library_material_path(material)
        return material

    @staticmethod
    def _material_override_for(
        session: SceneSession, prim_path: str, *, space: str
    ) -> MaterialOverride | None:
        for override in session.view.material_overrides:
            paths = (
                override.inspection_prim_paths
                if space == "inspection"
                else _override_source_paths_or_fallback(override)
            )
            if any(
                prim_path == path or prim_path.startswith(f"{path}/") for path in paths
            ):
                return override
        return None

    def _set_material_override(
        self,
        session: SceneSession,
        session_id: str,
        prim_path: str,
        material: dict[str, Any],
        *,
        space: str,
        unbind_existing: bool = True,
        remove_material_libraries: bool = False,
    ) -> None:
        translation = self._translate_path(session_id, prim_path, source_space=space)
        path_map = self._path_maps.get(session_id) or SceneOptimizerPathMap()
        source_paths = sorted(set(translation.source_paths), key=_natural_path_key)
        inspection_paths = sorted(
            set(translation.inspection_paths),
            key=_natural_path_key,
        )
        stored_space = space
        stored_prim_path = prim_path
        if space == "inspection" and not path_map.enabled:
            stored_space = "source"
            stored_prim_path = source_paths[0] if len(source_paths) == 1 else prim_path
        override = MaterialOverride(
            prim_path=stored_prim_path,
            material=material,
            unbind_existing=unbind_existing,
            remove_material_libraries=remove_material_libraries,
            space=stored_space,
            source_prim_paths=source_paths,
            inspection_prim_paths=inspection_paths,
        )
        source_path_set = set(source_paths)
        inspection_path_set = set(inspection_paths)
        trimmed_overrides: list[MaterialOverride] = []
        for existing in session.view.material_overrides:
            trimmed = _trim_material_override_coverage(
                existing,
                source_paths=source_path_set,
                inspection_paths=inspection_path_set,
            )
            if trimmed is not None:
                trimmed_overrides.append(trimmed)
        trimmed_overrides.append(override)
        session.view.material_overrides = trimmed_overrides

    def _material_overrides_for_inspection(
        self, session_id: str
    ) -> list[MaterialOverride]:
        session = self._require_ready_session(session_id)
        result: list[MaterialOverride] = []
        for override in session.view.material_overrides:
            inspection_paths = (
                override.inspection_prim_paths
                or self._translate_path(
                    session_id,
                    override.prim_path,
                    source_space="source",
                ).inspection_paths
            )
            for inspection_path in inspection_paths:
                result.append(
                    MaterialOverride(
                        prim_path=inspection_path,
                        material=override.material,
                        mode=override.mode,
                        unbind_existing=override.unbind_existing,
                        remove_material_libraries=override.remove_material_libraries,
                        space="inspection",
                        source_prim_paths=_override_source_paths_or_fallback(override),
                        inspection_prim_paths=[inspection_path],
                    )
                )
        return result

    @staticmethod
    def _restore_unresolved_mappings(
        overrides: list[MaterialOverride],
    ) -> list[dict[str, Any]]:
        unresolved: list[dict[str, Any]] = []
        for override in overrides:
            if override.source_prim_paths:
                continue
            unresolved.append(
                {
                    "prim_path": override.prim_path,
                    "space": override.space,
                    "reason": "material override has no resolved source prim paths",
                    "inspection_prim_paths": list(override.inspection_prim_paths),
                }
            )
        return unresolved

    def _translate_path(self, session_id: str, prim_path: str, *, source_space: str):
        path_map = self._path_maps.get(session_id) or SceneOptimizerPathMap()
        if source_space == "source":
            return path_map.translate_source_to_inspection(prim_path)
        if source_space == "inspection":
            return path_map.translate_inspection_to_source(prim_path)
        raise ValueError("source_space must be source or inspection")

    def _path_translation_response(
        self,
        session: SceneSession,
        session_id: str,
        prim_path: str,
        *,
        source_space: str,
        target_space: str = "source",
        path_map: SceneOptimizerPathMap | None = None,
    ) -> PathTranslationResponse:
        translation = (
            _translate_path_with_map(path_map, prim_path, source_space=source_space)
            if path_map is not None
            else self._translate_path(
                session_id,
                prim_path,
                source_space=source_space,
            )
        )
        return PathTranslationResponse(
            session_id=session_id,
            input_path=prim_path,
            source_space=source_space,
            target_space=target_space,
            source_paths=translation.source_paths,
            inspection_paths=translation.inspection_paths,
            ambiguous=translation.ambiguous,
            optimization=session.optimization.model_copy(deep=True),
        )

    def _optimize_scene(
        self,
        session_id: str,
        source_scene_path: Path,
        optimization_config: dict[str, object],
    ) -> tuple[Path, OptimizationState, SceneOptimizerPathMap]:
        from world_understanding.agentic.usd_tasks.optimize_usd import OptimizeUSDTask

        output_dir = self._workspace_for(session_id) / "optimized"
        output_dir.mkdir(parents=True, exist_ok=True)
        optimized_path = output_dir / f"{source_scene_path.stem}.optimized.usdc"
        config = dict(optimization_config)
        config.setdefault("backend", "local")
        context = OptimizeUSDTask().run(
            {
                "input_usd_path": str(source_scene_path),
                "output_usd_path": str(optimized_path),
                "optimization_config": config,
            }
        )
        metadata = context.get("optimization_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata_path = optimized_path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        path_map = SceneOptimizerPathMap.from_metadata(
            original_usd_path=source_scene_path,
            optimization_metadata=metadata,
        )
        optimization = OptimizationState(
            enabled=True,
            status="ready",
            source_scene_path=str(source_scene_path),
            inspection_scene_path=str(optimized_path),
            metadata_path=str(metadata_path),
            operations_executed=metadata.get("operations_executed", []),
            correspondence_summary=path_map.summary(),
        )
        return optimized_path, optimization, path_map

    def _workspace_for(self, session_id: str) -> Path:
        workspace = self._workspace_path(session_id)
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _allowed_output_roots_for_session(
        self,
        session_id: str,
    ) -> list[Path] | None:
        """Return the complete output-root policy for one session."""
        if not self._output_roots:
            return None
        return [self._workspace_for(session_id), *self._output_roots]

    def _workspace_path(self, session_id: str) -> Path:
        _validate_session_id(session_id)
        return self._workspace_root / session_id

    def resolve_agent_output_usd_path(
        self,
        session_id: str,
        raw_path: str | None,
        *,
        default_subdir: str,
        default_filename: str,
    ) -> Path:
        """Resolve an agent-requested USD output path inside session workspace."""
        return self._resolve_agent_workspace_output_path(
            session_id,
            raw_path,
            default_subdir=default_subdir,
            default_filename=default_filename,
            allowed_suffixes={".usd", ".usda", ".usdc", ".usdz"},
        )

    def resolve_agent_output_dir(
        self,
        session_id: str,
        raw_path: str | None,
        *,
        default_subdir: str,
    ) -> Path:
        """Resolve an agent-requested output directory inside session workspace."""
        return self._resolve_agent_workspace_output_path(
            session_id,
            raw_path,
            default_subdir=default_subdir,
            default_filename=None,
            allowed_suffixes=None,
        )

    def _resolve_agent_workspace_output_path(
        self,
        session_id: str,
        raw_path: str | None,
        *,
        default_subdir: str,
        default_filename: str | None,
        allowed_suffixes: set[str] | None,
    ) -> Path:
        workspace = self._workspace_for(session_id).resolve()
        if raw_path is None:
            stamp = f"{int(time.time() * 1000)}-{uuid4().hex}"
            resolved = workspace / default_subdir / stamp
            if default_filename is not None:
                resolved = resolved / default_filename
        else:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = workspace / candidate
            resolved = candidate.resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(
                "agent output paths must stay inside the session workspace"
            ) from exc
        if (
            allowed_suffixes is not None
            and resolved.suffix.lower() not in allowed_suffixes
        ):
            raise ValueError(
                "agent output USD path must end with .usd, .usda, .usdc, or .usdz"
            )
        target_dir = resolved if default_filename is None else resolved.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        return resolved

    def _default_restored_scene_path(
        self,
        session_id: str,
        *,
        source_scene_path: Path,
    ) -> Path:
        suffix = source_scene_path.suffix.lower()
        if suffix not in {".usd", ".usda", ".usdc", ".usdz"}:
            suffix = ".usda"
        restore_dir = self._workspace_for(session_id) / "restored"
        restore_dir.mkdir(parents=True, exist_ok=True)
        return restore_dir / f"restored-{uuid4().hex}{suffix}"

    def _material_cleared_scene_path(self, session_id: str, scene_path: Path) -> Path:
        cleaned_dir = self._workspace_for(session_id) / "material-cleared"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        return cleaned_dir / f"{scene_path.stem}.material-cleared.usda"

    def _run_renderer_call(
        self, operation: Callable[[], Any], *, timeout_label: str
    ) -> Any:
        future: Future[Any] = self._renderer_executor.submit(operation)
        try:
            return future.result(timeout=self._render_timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Timed out waiting for {timeout_label} after "
                f"{self._render_timeout_seconds:g} seconds"
            ) from exc

    def _begin_workspace_operation(self, session_id: str) -> None:
        self._active_workspace_ops[session_id] = (
            self._active_workspace_ops.get(session_id, 0) + 1
        )

    def _end_workspace_operation(self, session_id: str) -> None:
        count = self._active_workspace_ops.get(session_id, 0)
        if count <= 1:
            self._active_workspace_ops.pop(session_id, None)
        else:
            self._active_workspace_ops[session_id] = count - 1
        self._workspace_condition.notify_all()

    def _wait_for_workspace_operations(self, session_id: str, *, action: str) -> None:
        timeout_seconds = max(
            CLOSE_SESSION_WORKSPACE_TIMEOUT_SECONDS,
            self._render_timeout_seconds,
            self._material_apply_timeout_seconds,
        )
        deadline = time.monotonic() + timeout_seconds
        while self._active_workspace_ops.get(session_id, 0) > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for active workspace operation "
                    f"after {timeout_seconds:g} seconds before {action} "
                    f"session: {session_id}"
                )
            self._workspace_condition.wait(timeout=remaining)

    def _selection_mesh_paths(self, session_id: str) -> list[str]:
        session = self._require_ready_session(session_id)
        if not session.view.selected_prims:
            return []
        return self._require_queries(session_id).expand_to_mesh_paths(
            session.view.selected_prims
        )

    @staticmethod
    def _frame_camera(
        session: SceneSession,
        queries: UsdSceneQueries,
        prim_path: str,
        *,
        margin: float,
        direction: str | None = None,
    ) -> None:
        bounds = queries.get_bounds(prim_path)
        if direction:
            session.view.camera = _camera_state_from_bounds(
                bounds,
                direction=direction,
                margin=margin,
                width=session.viewport.width,
                height=session.viewport.height,
                focus_path=prim_path,
            )
            return
        session.view.camera = _frame_existing_camera(
            session.view.camera,
            bounds,
            margin=margin,
            width=session.viewport.width,
            height=session.viewport.height,
            focus_path=prim_path,
        )


_SNAPSHOT_RENDERABLE_TYPES = frozenset(
    {
        "BasisCurves",
        "Capsule",
        "Cone",
        "Cube",
        "Cylinder",
        "GeomSubset",
        "Mesh",
        "Sphere",
    }
)


def _snapshot_tree(
    queries: UsdSceneQueries,
    root_prim_path: str,
    *,
    max_prims: int,
) -> tuple[list[str], list[SceneSnapshotNode], bool, dict[str, Any]]:
    paths: list[str] = []
    included_paths: set[str] = set()
    expanded_paths: set[str] = set()
    nodes_by_path: dict[str, SceneSnapshotNode] = {}
    property_cache: dict[str, Any] = {}
    queue = deque([root_prim_path])
    queued = {root_prim_path}
    truncated = False

    while queue:
        prim_path = queue.popleft()
        if prim_path not in included_paths:
            if len(paths) >= max_prims:
                truncated = True
                break
            paths.append(prim_path)
            included_paths.add(prim_path)
        if prim_path in expanded_paths:
            continue
        expanded_paths.add(prim_path)

        tree = queries.get_children(prim_path)
        if prim_path not in nodes_by_path:
            properties_response = _snapshot_properties(
                queries, property_cache, prim_path
            )
            properties = properties_response.properties
            nodes_by_path[prim_path] = SceneSnapshotNode(
                path=prim_path,
                name=str(properties.get("name") or prim_path.rsplit("/", 1)[-1]),
                type_name=str(properties.get("type_name") or ""),
                active=bool(properties.get("active", True)),
                loaded=bool(properties.get("loaded", True)),
                children=bool(tree.children),
                child_paths=[],
            )
        node = nodes_by_path[prim_path]
        child_paths = []

        for child in tree.children:
            if child.path not in nodes_by_path:
                nodes_by_path[child.path] = SceneSnapshotNode(
                    path=child.path,
                    name=child.name,
                    type_name=child.type_name,
                    active=child.active,
                    loaded=child.loaded,
                    children=child.children,
                    child_paths=[],
                )
            if child.path not in included_paths:
                if len(paths) >= max_prims:
                    truncated = True
                    break
                paths.append(child.path)
                included_paths.add(child.path)
            child_paths.append(child.path)
            if (
                child.children
                and child.active
                and child.loaded
                and child.path not in queued
            ):
                queue.append(child.path)
                queued.add(child.path)
        node.child_paths = child_paths
        node.children = bool(node.child_paths)
        if truncated:
            break

    nodes = [nodes_by_path[path] for path in paths if path in nodes_by_path]
    return paths, nodes, truncated, property_cache


def _snapshot_candidate_hint_paths(
    queries: UsdSceneQueries,
    root_prim_path: str,
    *,
    max_paths: int,
) -> tuple[list[str], bool]:
    """Collect renderable candidate paths even when the display tree is capped."""
    if max_paths <= 0:
        return [], False
    try:
        from pxr import Usd
    except Exception:
        return [], False

    root_prim = queries.stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        return [], False

    paths: list[str] = []
    seen: set[str] = set()
    predicate = Usd.TraverseInstanceProxies(
        Usd.PrimIsActive & Usd.PrimIsLoaded & Usd.PrimIsDefined
    )
    for prim in Usd.PrimRange(root_prim, predicate):
        type_name = prim.GetTypeName() or ""
        if type_name not in _SNAPSHOT_RENDERABLE_TYPES:
            continue
        path = str(prim.GetPath())
        if path in seen:
            continue
        if len(paths) >= max_paths:
            return paths, True
        seen.add(path)
        paths.append(path)
    return paths, False


def _snapshot_properties(
    queries: UsdSceneQueries,
    property_cache: dict[str, Any],
    prim_path: str,
):
    properties = property_cache.get(prim_path)
    if properties is None:
        properties = queries.get_properties(prim_path)
        property_cache[prim_path] = properties
    return properties


def _snapshot_ancestor_paths(
    queries: UsdSceneQueries,
    prim_path: str,
) -> list[str]:
    ancestors = []
    current = prim_path.rsplit("/", 1)[0] or "/"
    while current and current != "/":
        if queries.has_prim(current):
            ancestors.append(current)
        current = current.rsplit("/", 1)[0] or "/"
    return ancestors


def _translate_path_with_map(
    path_map: SceneOptimizerPathMap,
    prim_path: str,
    *,
    source_space: str,
):
    if source_space == "source":
        return path_map.translate_source_to_inspection(prim_path)
    if source_space == "inspection":
        return path_map.translate_inspection_to_source(prim_path)
    raise ValueError("source_space must be source or inspection")


def _natural_path_key(path: str) -> list[tuple[int, int | str]]:
    return [
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.split(r"(\d+)", path)
    ]


def _material_override_overlaps(
    override: MaterialOverride,
    *,
    source_paths: set[str],
    inspection_paths: set[str],
) -> bool:
    existing_source_paths = set(override.source_prim_paths)
    if not existing_source_paths and override.space == "source":
        existing_source_paths.add(override.prim_path)
    existing_inspection_paths = set(override.inspection_prim_paths)
    if not existing_inspection_paths and override.space == "inspection":
        existing_inspection_paths.add(override.prim_path)
    return bool(existing_source_paths & source_paths) or bool(
        existing_inspection_paths & inspection_paths
    )


def _override_source_paths_or_fallback(override: MaterialOverride) -> list[str]:
    """Return an override's source coverage, falling back to `prim_path` only
    when appropriate.

    `source_prim_paths` is empty in two different situations that must not
    be treated the same: the override never tracked explicit per-leaf source
    coverage (in which case `prim_path` *is* its coverage, and is the
    intended fallback), or `_trim_material_override_coverage` narrowed a
    previously-populated `source_prim_paths` down to nothing because every
    leaf it covered is now superseded by a later command (in which case
    falling back to `prim_path` would silently reintroduce exactly the
    coverage that trim was supposed to remove).
    """
    if override.source_prim_paths:
        return override.source_prim_paths
    if override.source_coverage_exhausted:
        return []
    return [override.prim_path]


def _trim_material_override_coverage(
    override: MaterialOverride,
    *,
    source_paths: set[str],
    inspection_paths: set[str],
) -> MaterialOverride | None:
    """Narrow an existing override's coverage instead of deleting it wholesale.

    A single optimized runtime prim can represent multiple canonical source
    prims (dedup), and a single canonical source prim can in turn have more
    than one runtime fragment (split/dedup combined). A new command that only
    covers part of an existing override's resolved prims must not discard the
    existing override's coverage of the *other* prims it still uniquely
    represents. Only drop the existing override entirely when every leaf prim
    it previously resolved to is now superseded by the new command.
    """
    existing_source_paths = list(override.source_prim_paths)
    existing_inspection_paths = list(override.inspection_prim_paths)
    has_explicit_paths = bool(existing_source_paths or existing_inspection_paths)

    if not has_explicit_paths:
        # No resolved leaf list was recorded; the override's own prim_path is
        # its sole implicit coverage, so it can only be kept or dropped whole.
        if override.space == "source" and override.prim_path in source_paths:
            return None
        if override.space == "inspection" and override.prim_path in inspection_paths:
            return None
        return override

    trimmed_source_paths = [
        path for path in existing_source_paths if path not in source_paths
    ]
    trimmed_inspection_paths = [
        path for path in existing_inspection_paths if path not in inspection_paths
    ]
    if trimmed_source_paths == existing_source_paths and (
        trimmed_inspection_paths == existing_inspection_paths
    ):
        return override
    if not trimmed_source_paths and not trimmed_inspection_paths:
        return None
    # Narrowing only one side must not discard the other side's still-valid,
    # unrelated coverage (e.g. a candidate with both a shared runtime alias
    # and its own unique fragment: losing the shared alias's representation
    # to a sibling's command must not also discard the unique fragment,
    # regardless of which of the two lists happens to carry it). But
    # `source_prim_paths` uniquely backs a fallback in downstream consumers
    # (`_material_overrides_for_inspection`, `_material_override_for`, and the
    # durable-apply/library-identity helpers below): whenever it is empty,
    # they substitute `override.prim_path` instead via
    # `_override_source_paths_or_fallback`, which is only correct for an
    # override that never tracked explicit source coverage. Record when
    # trimming (rather than "never tracked") is what emptied a
    # previously-populated `source_prim_paths`, so those call sites can tell
    # the difference and skip the fallback instead of silently reintroducing
    # exactly the coverage this trim was supposed to remove.
    source_coverage_exhausted = bool(existing_source_paths) and not trimmed_source_paths
    return override.model_copy(
        update={
            "source_prim_paths": trimmed_source_paths,
            "inspection_prim_paths": trimmed_inspection_paths,
            "source_coverage_exhausted": (
                override.source_coverage_exhausted or source_coverage_exhausted
            ),
        }
    )


def _snapshot_material_binding(
    session: SceneSession,
    queries: UsdSceneQueries,
    prim_path: str,
) -> MaterialBindingResponse:
    binding = queries.get_material_binding(prim_path)
    binding.material_override = SessionManager._material_override_for(
        session,
        prim_path,
        space="inspection",
    )
    return binding


def _snapshot_candidates(
    *,
    session_id: str,
    properties: list[Any],
    visibility_properties: list[Any] | None = None,
    material_bindings: list[MaterialBindingResponse],
    path_translations: list[PathTranslationResponse],
) -> list[SceneSnapshotCandidate]:
    properties_by_path = {item.prim_path: item.properties for item in properties}
    visibility_properties_by_path = {
        item.prim_path: item.properties for item in visibility_properties or properties
    }
    bindings_by_path = {item.prim_path: item for item in material_bindings}
    translations_by_path = {item.input_path: item for item in path_translations}
    candidates: list[SceneSnapshotCandidate] = []

    for prim_path, prim_properties in properties_by_path.items():
        type_name = str(prim_properties.get("type_name") or "")
        active = bool(prim_properties.get("active", True))
        loaded = bool(prim_properties.get("loaded", True))
        bounds = prim_properties.get("bounds")
        binding = bindings_by_path.get(prim_path)
        binding_type = binding.binding_type if binding is not None else "none"
        direct_targets = binding.direct_targets if binding is not None else []
        is_renderable = type_name in _SNAPSHOT_RENDERABLE_TYPES
        is_material_bound = binding_type != "none" or bool(direct_targets)
        effective_visible = _snapshot_effective_visible(
            prim_path,
            visibility_properties_by_path,
        )
        if not (
            active
            and loaded
            and effective_visible
            and isinstance(bounds, dict)
            and _snapshot_bounds_volume(bounds) > 0
            and (is_renderable or is_material_bound)
        ):
            continue

        translation = translations_by_path.get(prim_path)
        reason = "renderable_prim" if is_renderable else "material_bound_container"
        candidates.append(
            SceneSnapshotCandidate(
                inspection_path=prim_path,
                source_paths=(
                    translation.source_paths if translation is not None else [prim_path]
                ),
                type_name=type_name,
                active=active,
                loaded=loaded,
                effective_visible=effective_visible,
                bounds_center=_snapshot_bounds_center(bounds),
                bounds_size=_snapshot_bounds_size(bounds),
                material_binding_type=binding_type,
                bound_material_path=(
                    binding.bound_material_path if binding is not None else None
                ),
                binding_source_path=(
                    binding.binding_source_path if binding is not None else None
                ),
                direct_targets=direct_targets,
                material_override=(
                    binding.material_override if binding is not None else None
                ),
                ambiguous_translation=(
                    bool(translation.ambiguous) if translation is not None else False
                ),
                candidate_reason=reason,
            )
        )

    return candidates


def _snapshot_effective_visible(
    prim_path: str,
    properties_by_path: dict[str, dict[str, Any]],
) -> bool:
    current = prim_path
    while current:
        attributes = properties_by_path.get(current, {}).get("attributes", {})
        visibility = attributes.get("visibility", {}).get("value")
        if visibility == "invisible":
            return False
        if current == "/":
            return True
        current = current.rsplit("/", 1)[0]
    return True


def _snapshot_bounds_center(bounds: dict[str, Any]) -> list[float] | None:
    center = bounds.get("center")
    if not isinstance(center, list) or len(center) != 3:
        return None
    return [round(float(value), 6) for value in center]


def _snapshot_bounds_size(bounds: dict[str, Any]) -> list[float] | None:
    min_values = bounds.get("min")
    max_values = bounds.get("max")
    if not (
        isinstance(min_values, list)
        and isinstance(max_values, list)
        and len(min_values) == 3
        and len(max_values) == 3
    ):
        return None
    return [
        round(max(0.0, float(max_value) - float(min_value)), 6)
        for min_value, max_value in zip(min_values, max_values, strict=True)
    ]


def _snapshot_bounds_volume(bounds: dict[str, Any]) -> float:
    size = _snapshot_bounds_size(bounds)
    if size is None:
        return 0.0
    return max(size, default=0.0)


def _resolve_material_apply_output_path(
    raw_path: str,
    *,
    source_scene_path: Path,
    inspection_scene_path: Path | None,
    overwrite: bool,
    allowed_output_roots: list[Path] | None = None,
) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ValueError("output_usd_path must be an absolute local path")
    if path.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise ValueError("output_usd_path must end with .usd, .usda, .usdc, or .usdz")

    resolved = path.resolve()
    protected_paths = {source_scene_path.resolve()}
    if inspection_scene_path is not None:
        protected_paths.add(inspection_scene_path.resolve())
    if resolved in protected_paths:
        raise ValueError("output_usd_path must not overwrite the loaded scene")

    if allowed_output_roots is not None:
        roots = [root.expanduser().resolve() for root in allowed_output_roots]
        if not any(resolved.is_relative_to(root) for root in roots):
            raise ValueError(f"output_usd_path is outside {OUTPUT_ROOTS_ENV}")

    if resolved.exists():
        if not overwrite:
            raise ValueError(
                "output_usd_path already exists; set overwrite=true to replace it"
            )
        if not resolved.is_file():
            raise ValueError("output_usd_path exists but is not a file")
    return resolved


def _output_roots_from_env() -> list[Path]:
    raw = os.environ.get(OUTPUT_ROOTS_ENV, "")
    return [
        Path(item.strip()).expanduser().resolve()
        for item in raw.split(",")
        if item.strip()
    ]


def _material_apply_task_output_path(output_path: Path) -> Path:
    task_suffix = (
        ".usdc" if output_path.suffix.lower() == ".usdz" else output_path.suffix
    )
    return output_path.with_name(f".{output_path.stem}.{uuid4().hex}{task_suffix}")


def _material_apply_bound_source_paths(
    *,
    assignment_stats: dict[str, Any],
    prediction_records: list[dict[str, Any]],
    fail_on_invalid_assignment: bool = True,
) -> tuple[list[str], list[str]]:
    """Return exact bound/unbound coverage, optionally rejecting partial apply."""

    raw_bound = assignment_stats.get("bound_prim_ids")
    raw_unbound = assignment_stats.get("unbound_prim_ids")
    if not isinstance(raw_bound, list) or not isinstance(raw_unbound, list):
        raise RuntimeError(
            "Material apply did not return exact bound/unbound prim coverage."
        )
    if any(
        not isinstance(path, str) or not path.startswith("/")
        for path in [*raw_bound, *raw_unbound]
    ):
        raise RuntimeError("Material apply returned invalid prim binding coverage.")

    requested_paths = {str(record["id"]) for record in prediction_records}
    bound_paths = set(raw_bound)
    unbound_paths = set(raw_unbound)
    unexpected_paths = (bound_paths | unbound_paths) - requested_paths
    overlapping_paths = bound_paths & unbound_paths
    if unexpected_paths or overlapping_paths:
        raise RuntimeError(
            "Material apply returned inconsistent prim binding coverage: "
            f"unexpected={sorted(unexpected_paths, key=_natural_path_key)}, "
            f"both_bound_and_unbound={sorted(overlapping_paths, key=_natural_path_key)}"
        )
    incomplete_paths = sorted(
        (requested_paths - bound_paths) | unbound_paths,
        key=_natural_path_key,
    )
    if incomplete_paths and fail_on_invalid_assignment:
        raise RuntimeError(
            f"Material apply left requested prim targets unbound: {incomplete_paths}"
        )
    return sorted(bound_paths, key=_natural_path_key), incomplete_paths


def _export_unchanged_scene(
    *,
    source_path: Path,
    output_path: Path,
    output_mode: str,
    staging_dir: Path,
    allowed_output_roots: list[Path] | None,
    overwrite: bool,
) -> None:
    """Export a zero-edit restore in the requested USD mode and file format."""

    from pxr import Usd

    stage = Usd.Stage.Open(str(source_path))
    if stage is None:
        raise ValueError(f"Failed to open zero-edit restore source: {source_path}")
    task_output_path = _material_apply_task_output_path(staging_dir / output_path.name)
    publish_path: Path | None = None
    try:
        layer = stage.Flatten() if output_mode == "flattened" else stage.GetRootLayer()
        if not layer.Export(str(task_output_path)):
            raise RuntimeError(
                f"Failed to export zero-edit restore USD: {task_output_path}"
            )
        if Usd.Stage.Open(str(task_output_path)) is None:
            raise RuntimeError(
                f"Zero-edit restore export is not a valid USD: {task_output_path}"
            )
        publish_path = _prepare_material_apply_output_for_publish(
            task_output_path,
            output_path=output_path,
            staging_dir=staging_dir,
            asset_anchor_path=source_path,
        )
        _secure_publish_staged_output(
            publish_path,
            output_path=output_path,
            allowed_output_roots=allowed_output_roots,
            overwrite=overwrite,
        )
    finally:
        task_output_path.unlink(missing_ok=True)
        if publish_path is not None:
            publish_path.unlink(missing_ok=True)


def _prepare_material_apply_output_for_publish(
    task_output_path: Path,
    *,
    output_path: Path,
    staging_dir: Path,
    asset_anchor_path: Path | None = None,
) -> Path:
    """Build a fully validated publication artifact in service-owned storage."""

    publish_path = staging_dir / f"publish-{uuid4().hex}{output_path.suffix.lower()}"
    if output_path.suffix.lower() == ".usdz":
        package_source_path = staging_dir / (
            f"package-source-{uuid4().hex}{task_output_path.suffix.lower()}"
        )
        try:
            _export_usd_with_rebased_assets(
                task_output_path,
                package_source_path,
                logical_output_parent=package_source_path.parent,
                asset_anchor_path=asset_anchor_path,
            )
            _package_material_apply_usdz(package_source_path, publish_path)
        finally:
            package_source_path.unlink(missing_ok=True)
        return publish_path

    _export_usd_with_rebased_assets(
        task_output_path,
        publish_path,
        logical_output_parent=output_path.parent,
        asset_anchor_path=asset_anchor_path,
    )
    return publish_path


def _export_usd_with_rebased_assets(
    source_path: Path,
    output_path: Path,
    *,
    logical_output_parent: Path,
    asset_anchor_path: Path | None = None,
) -> None:
    """Export a detached layer with local assets anchored to its final parent."""

    from pxr import Sdf, UsdUtils

    anchor_path = asset_anchor_path or source_path
    anchor_layer = Sdf.Layer.FindOrOpen(str(anchor_path))
    editable_layer = Sdf.Layer.OpenAsAnonymous(str(source_path))
    if anchor_layer is None or editable_layer is None:
        raise RuntimeError(f"Could not open staged material output: {source_path}")

    def rebase_asset_path(asset_path: str) -> str:
        if (
            not asset_path
            or Path(asset_path).is_absolute()
            or urlparse(asset_path).scheme
        ):
            return asset_path
        absolute_path = anchor_layer.ComputeAbsolutePath(asset_path)
        if not absolute_path:
            return asset_path
        return os.path.relpath(absolute_path, logical_output_parent).replace("\\", "/")

    UsdUtils.ModifyAssetPaths(
        editable_layer,
        rebase_asset_path,
        keepEmptyPathsInArrays=True,
    )
    if not editable_layer.Export(str(output_path)):
        raise RuntimeError(f"Could not export staged material output: {output_path}")


def _secure_publish_staged_output(
    staged_path: Path,
    *,
    output_path: Path,
    allowed_output_roots: list[Path] | None,
    overwrite: bool,
) -> None:
    """Publish through an anchored directory descriptor without following links."""

    staged_metadata = staged_path.stat(follow_symlinks=False)
    if not stat.S_ISREG(staged_metadata.st_mode) or staged_metadata.st_nlink != 1:
        raise ValueError("Staged material output must be a single-link regular file")

    root, relative_path = _secure_output_root_and_relative_path(
        output_path,
        allowed_output_roots=allowed_output_roots,
    )
    parent_fd = _open_secure_output_parent(
        root,
        relative_path.parent,
        create=True,
    )
    output_name = relative_path.name
    parent_metadata = os.fstat(parent_fd)
    published = False
    published_output_verified = False
    temporary_name: str | None = None
    backup_name: str | None = None
    try:
        try:
            existing = os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not overwrite:
                raise ValueError(
                    "output_usd_path already exists; set overwrite=true to replace it"
                )
            if not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1:
                raise ValueError(
                    "output_usd_path exists but is not a single-link regular file"
                )

        temporary_name = _copy_staged_output_to_anchored_fd(
            staged_path,
            parent_fd=parent_fd,
            output_name=output_name,
        )

        _verify_anchored_output_parent(
            root,
            relative_path.parent,
            expected=parent_metadata,
        )
        if overwrite:
            if existing is not None:
                backup_name = f".{output_name}.{uuid4().hex}.bak"
                os.link(
                    output_name,
                    backup_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                backup_metadata = os.stat(
                    backup_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                current_metadata = os.stat(
                    output_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(backup_metadata.st_mode)
                    or (backup_metadata.st_dev, backup_metadata.st_ino)
                    != (current_metadata.st_dev, current_metadata.st_ino)
                    or (backup_metadata.st_dev, backup_metadata.st_ino)
                    != (existing.st_dev, existing.st_ino)
                    or backup_metadata.st_nlink != 2
                    or current_metadata.st_nlink != 2
                ):
                    raise RuntimeError(
                        "Could not securely preserve the existing material output"
                    )
            os.replace(
                temporary_name,
                output_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        else:
            try:
                os.link(
                    temporary_name,
                    output_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise ValueError(
                    "output_usd_path already exists; set overwrite=true to replace it"
                ) from exc
            os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_name = None
        published = True

        published_metadata = os.stat(
            output_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(published_metadata.st_mode)
            or published_metadata.st_nlink != 1
        ):
            raise RuntimeError(
                "Published material output is not a single-link regular file"
            )
        published_output_verified = True
        _verify_anchored_output_parent(
            root,
            relative_path.parent,
            expected=parent_metadata,
        )
        if backup_name is not None:
            backup_metadata = os.stat(
                backup_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                existing is None
                or not stat.S_ISREG(backup_metadata.st_mode)
                or backup_metadata.st_nlink != 1
                or (backup_metadata.st_dev, backup_metadata.st_ino)
                != (existing.st_dev, existing.st_ino)
            ):
                raise RuntimeError(
                    "Existing material output backup changed during publication"
                )
            os.unlink(backup_name, dir_fd=parent_fd)
            backup_name = None
    except Exception:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        if published:
            if backup_name is not None:
                try:
                    backup_metadata = os.stat(
                        backup_name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    if (
                        existing is None
                        or not stat.S_ISREG(backup_metadata.st_mode)
                        or backup_metadata.st_nlink != 1
                        or (backup_metadata.st_dev, backup_metadata.st_ino)
                        != (existing.st_dev, existing.st_ino)
                    ):
                        raise RuntimeError(
                            "Existing material output backup changed before rollback"
                        )
                    os.replace(
                        backup_name,
                        output_name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                    backup_name = None
                except Exception as rollback_error:
                    raise RuntimeError(
                        "Secure publication failed and the prior output could not "
                        f"be restored; it remains preserved as {backup_name}"
                    ) from rollback_error
            elif overwrite or not published_output_verified:
                # Replacing an absent output is still reversible: remove the
                # new name so a failed overwrite transaction leaves the path
                # in its original missing state. Also remove a non-overwrite
                # output that failed its own file verification. A successfully
                # verified non-overwrite link is the no-clobber commit point;
                # preserve it if later parent verification fails so the raised
                # error remains recoverable.
                try:
                    os.unlink(output_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
        elif backup_name is not None:
            try:
                os.unlink(backup_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            backup_name = None
        raise
    finally:
        os.close(parent_fd)


def _secure_output_root_and_relative_path(
    output_path: Path,
    *,
    allowed_output_roots: list[Path] | None,
) -> tuple[Path, Path]:
    resolved_output = output_path.expanduser().resolve()
    if allowed_output_roots:
        candidates = [
            root.expanduser().resolve()
            for root in allowed_output_roots
            if resolved_output.is_relative_to(root.expanduser().resolve())
        ]
        if not candidates:
            raise ValueError(f"output_usd_path is outside {OUTPUT_ROOTS_ENV}")
        root = max(candidates, key=lambda candidate: len(candidate.parts))
    else:
        root = Path(resolved_output.anchor)
    relative_path = resolved_output.relative_to(root)
    if not relative_path.name or any(
        part in {"", ".", ".."} for part in relative_path.parts
    ):
        raise ValueError("output_usd_path contains an unsafe path component")
    return root, relative_path


def _open_secure_output_parent(
    root: Path,
    relative_parent: Path,
    *,
    create: bool,
) -> int:
    common_flags = getattr(os, "O_DIRECTORY", 0)
    common_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    path_flags = getattr(os, "O_PATH", os.O_RDONLY) | common_flags
    read_flags = os.O_RDONLY | common_flags
    descriptor = os.open(root, path_flags)
    try:
        for component in relative_parent.parts:
            created = False
            try:
                next_descriptor = os.open(
                    component,
                    path_flags,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o711, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                next_descriptor = os.open(
                    component,
                    read_flags if created else path_flags,
                    dir_fd=descriptor,
                )
            if created:
                try:
                    # The Workbench can use a restrictive umask, so set the
                    # completed directory through its anchored descriptor.
                    # Grant only traversal to other UIDs; the configured
                    # output root remains the enclosing access boundary.
                    os.fchmod(next_descriptor, 0o711)
                    created_metadata = os.fstat(next_descriptor)
                    if stat.S_IMODE(created_metadata.st_mode) != 0o711:
                        raise RuntimeError(
                            "New output directory is not safely traversable"
                        )
                except Exception:
                    os.close(next_descriptor)
                    raise
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _verify_anchored_output_parent(
    root: Path,
    relative_parent: Path,
    *,
    expected: os.stat_result,
) -> None:
    descriptor = _open_secure_output_parent(root, relative_parent, create=False)
    try:
        actual = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)
        or (actual.st_uid, actual.st_gid) != (expected.st_uid, expected.st_gid)
        or stat.S_IMODE(actual.st_mode) != stat.S_IMODE(expected.st_mode)
    ):
        raise RuntimeError(
            "Output directory identity or permissions changed during secure publication"
        )


def _copy_staged_output_to_anchored_fd(
    staged_path: Path,
    *,
    parent_fd: int,
    output_name: str,
) -> str:
    """Copy staged bytes to a unique file anchored below ``parent_fd``."""

    temporary_name = f".{output_name}.{uuid4().hex}.tmp"
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_flags |= getattr(os, "O_CLOEXEC", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    destination_flags |= getattr(os, "O_NOFOLLOW", 0)
    destination_flags |= getattr(os, "O_CLOEXEC", 0)
    source_descriptor = os.open(staged_path, source_flags)
    try:
        source_metadata = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_metadata.st_mode) or source_metadata.st_nlink != 1:
            raise ValueError(
                "Staged material output must be a single-link regular file"
            )
        descriptor = os.open(
            temporary_name,
            destination_flags,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            for chunk in iter(lambda: os.read(source_descriptor, 1024 * 1024), b""):
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
            # Keep the partially copied inode private, then make the complete
            # durable artifact readable to a caller that accesses a remote or
            # containerized Workbench through a shared mount. Output roots and
            # their directory permissions remain the access boundary.
            os.fchmod(descriptor, 0o644)
            os.fsync(descriptor)
            linked = os.stat(
                temporary_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            current = os.fstat(descriptor)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or (linked.st_dev, linked.st_ino) != (current.st_dev, current.st_ino)
                or stat.S_IMODE(current.st_mode) != 0o644
            ):
                raise RuntimeError("Temporary output changed during secure publication")
        finally:
            os.close(descriptor)
    except Exception:
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(source_descriptor)
    return temporary_name


def _package_material_apply_usdz(source_usd_path: Path, usdz_path: Path) -> None:
    """Package a material-apply USD layer into a validated USDZ archive."""
    from pxr import UsdUtils

    usdz_path.unlink(missing_ok=True)
    try:
        success = UsdUtils.CreateNewUsdzPackage(str(source_usd_path), str(usdz_path))
    except Exception as exc:
        usdz_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to create USDZ package: {usdz_path}") from exc

    if not success or not usdz_path.exists():
        usdz_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to create USDZ package: {usdz_path}")

    if not zipfile.is_zipfile(usdz_path):
        usdz_path.unlink(missing_ok=True)
        raise RuntimeError(f"CreateNewUsdzPackage wrote non-ZIP data to {usdz_path}")


def _material_assignment_record(
    override: MaterialOverride,
    *,
    source_scene_path: Path | None,
) -> MaterialAssignmentRecord:
    library_path: str | None = None
    material_path = _material_library_material_path(override.material)
    if source_scene_path is not None:
        try:
            resolved_library_path = _material_library_path(
                override.material,
                source_path=source_scene_path,
            )
            library_path = (
                str(resolved_library_path)
                if resolved_library_path is not None
                else None
            )
        except Exception:
            library_path = None
    source_prim_paths = _override_source_paths_or_fallback(override)
    identity = {
        "material": override.material,
        "mode": override.mode,
        "prim_path": override.prim_path,
        "source_prim_paths": source_prim_paths,
    }
    digest = hashlib.sha1(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return MaterialAssignmentRecord(
        assignment_id=f"ma_{digest}",
        prim_path=override.prim_path,
        space=override.space,
        source_prim_paths=source_prim_paths,
        inspection_prim_paths=list(override.inspection_prim_paths),
        material=dict(override.material),
        material_library_path=library_path,
        material_path=material_path,
        mode=override.mode,
        unbind_existing=override.unbind_existing,
        remove_material_libraries=override.remove_material_libraries,
    )


def _is_durable_material_override(
    override: MaterialOverride,
    *,
    source_scene_path: Path,
) -> bool:
    return (
        bool(override.source_prim_paths)
        and _material_library_material_path(override.material) is not None
        and _material_library_path(override.material, source_path=source_scene_path)
        is not None
    )


def _build_material_apply_payload(
    overrides: list[MaterialOverride],
    *,
    source_scene_path: Path,
    fail_on_invalid_assignment: bool,
) -> dict[str, Any]:
    material_library_path: Path | None = None
    resolved_materials: dict[str, str] = {}
    prediction_records: list[dict[str, str]] = []
    assignment_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    skipped_assignment_count = 0

    for override in overrides:
        record = _material_assignment_record(
            override,
            source_scene_path=source_scene_path,
        )
        assignment_records.append(record.model_dump())
        try:
            library_path = _material_library_path(
                override.material,
                source_path=source_scene_path,
            )
            material_path = _material_library_material_path(override.material)
            if library_path is None or material_path is None:
                raise ValueError(
                    "durable material apply requires a material-library assignment"
                )
            if (
                material_library_path is not None
                and library_path != material_library_path
            ):
                raise ValueError(
                    "durable material apply currently supports one material "
                    "library per request"
                )
            material_library_path = library_path
            material_name = _material_name_for_apply(
                override.material,
                material_path=material_path,
            )
            existing_material_path = resolved_materials.get(material_name)
            if (
                existing_material_path is not None
                and existing_material_path != material_path
            ):
                raise ValueError(
                    "material assignments use the same material name for "
                    f"different library paths: {material_name}"
                )
            resolved_materials[material_name] = material_path
            source_prim_paths = _override_source_paths_or_fallback(override)
            for prim_path in source_prim_paths:
                prediction_records.append(
                    {
                        "id": prim_path,
                        "material": material_name,
                    }
                )
        except Exception as exc:
            message = f"Skipping material assignment at {override.prim_path}: {exc}"
            if fail_on_invalid_assignment:
                raise ValueError(message) from exc
            warnings.append(message)
            skipped_assignment_count += 1

    if material_library_path is None:
        raise ValueError("No material-library assignments are available to apply")
    return {
        "assignment_records": assignment_records,
        "prediction_records": prediction_records,
        "resolved_materials": resolved_materials,
        "material_library_path": material_library_path,
        "warnings": warnings,
        "skipped_assignment_count": skipped_assignment_count,
    }


def _export_material_cleared_stage(
    *,
    source_path: Path,
    output_path: Path,
) -> None:
    from pxr import Sdf, Usd
    from world_understanding.utils.usd.prim import nullify_materials

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    root_layer = Sdf.Layer.CreateNew(str(output_path))
    _copy_root_stage_metadata(source_path, root_layer)
    _validate_preview_sublayer_asset_path(source_path, label="Source scene")
    root_layer.subLayerPaths.append(str(source_path))
    stage = Usd.Stage.Open(root_layer.identifier)
    if stage is None:
        raise ValueError(f"Failed to open material-cleared USD stage: {output_path}")

    default_prim = stage.GetDefaultPrim()
    if default_prim.IsValid():
        stage.SetDefaultPrim(default_prim)
    nullify_materials(stage, traversal_method="traverse_all")

    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"Failed to export material-cleared USD: {output_path}")


def _material_name_for_apply(material: dict[str, Any], *, material_path: str) -> str:
    raw_name = (
        material.get("material_name")
        or material.get("name")
        or material.get("display_name")
    )
    if isinstance(raw_name, str) and raw_name.strip():
        return raw_name.strip()
    return material_path.rstrip("/").rsplit("/", 1)[-1]


def _export_preview_stage(
    *,
    source_path: Path,
    output_path: Path,
    root_prim_path: str | None,
    overrides: list[MaterialOverride],
    hidden_prims: list[str],
    isolated_prims: list[str],
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    root_layer = Sdf.Layer.CreateNew(str(output_path))
    _copy_root_stage_metadata(source_path, root_layer)
    _validate_preview_sublayer_asset_path(source_path, label="Source scene")
    root_layer.subLayerPaths.append(str(source_path))
    for library_path in _material_library_paths(overrides, source_path=source_path):
        root_layer.subLayerPaths.append(str(library_path))
    stage = Usd.Stage.Open(root_layer.identifier)
    if stage is None:
        raise ValueError(f"Failed to open USD preview stage: {output_path}")

    if root_prim_path:
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if root_prim.IsValid():
            stage.SetDefaultPrim(root_prim)

    if any(override.remove_material_libraries for override in overrides):
        _deactivate_existing_material_libraries(stage, UsdShade)

    instance_root_paths = _instance_root_paths_for_overrides(
        stage, overrides, Usd, UsdGeom
    )
    if instance_root_paths:
        for path in instance_root_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                prim.SetInstanceable(False)
        if not stage.GetRootLayer().Save():
            raise RuntimeError(f"Failed to export preview USD: {output_path}")
        # Reopen after disabling instance roots so subsequent material bindings
        # target editable prims instead of stale instance proxy state.
        stage = Usd.Stage.Open(root_layer.identifier)
        if stage is None:
            raise ValueError(f"Failed to reopen USD preview stage: {output_path}")
        if root_prim_path:
            root_prim = stage.GetPrimAtPath(root_prim_path)
            if root_prim.IsValid():
                stage.SetDefaultPrim(root_prim)

    for override in _material_overrides_by_specificity(overrides):
        material = _define_preview_material(
            stage,
            root_prim_path,
            override,
            Gf,
            Sdf,
            UsdShade,
        )
        target_prim = stage.GetPrimAtPath(override.prim_path)
        if not target_prim.IsValid():
            raise KeyError(f"Prim not found: {override.prim_path}")
        mesh_prims = _target_mesh_prims(target_prim, UsdGeom)
        if not mesh_prims:
            raise ValueError(f"No mesh prims found under {override.prim_path}")
        fallback_bound_paths: set[str] = set()
        for prim in mesh_prims:
            _bind_preview_material(
                prim,
                material,
                UsdShade,
                unbind_existing=override.unbind_existing,
                fallback_bound_paths=fallback_bound_paths,
            )

    _apply_visibility_state(stage, hidden_prims, isolated_prims, UsdGeom)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"Failed to export preview USD: {output_path}")


def _copy_root_stage_metadata(source_path: Path, target_layer: Any) -> None:
    from pxr import Usd, UsdGeom

    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise ValueError(f"Failed to open USD stage metadata source: {source_path}")
    source_root = source_stage.GetRootLayer()
    target_layer.pseudoRoot.SetInfo("upAxis", UsdGeom.GetStageUpAxis(source_stage))
    target_layer.pseudoRoot.SetInfo(
        "metersPerUnit", float(UsdGeom.GetStageMetersPerUnit(source_stage))
    )
    if source_root.pseudoRoot.HasInfo("kilogramsPerUnit"):
        target_layer.pseudoRoot.SetInfo(
            "kilogramsPerUnit",
            source_root.pseudoRoot.GetInfo("kilogramsPerUnit"),
        )
    target_layer.customLayerData = dict(source_root.customLayerData or {})


def _material_overrides_by_specificity(
    overrides: list[MaterialOverride],
) -> list[MaterialOverride]:
    return sorted(
        overrides,
        key=lambda override: (override.prim_path.count("/"), override.prim_path),
    )


def _deactivate_existing_material_libraries(stage, usd_shade) -> None:
    material_paths = [
        str(prim.GetPath())
        for prim in stage.TraverseAll()
        if prim.IsA(usd_shade.Material)
    ]
    for path in sorted(material_paths, key=lambda item: item.count("/"), reverse=True):
        stage.OverridePrim(path).SetActive(False)


def _define_preview_material(stage, root_prim_path, override, gf, sdf, usd_shade):
    library_material_path = _material_library_material_path(override.material)
    if library_material_path:
        prim = stage.GetPrimAtPath(library_material_path)
        if not prim.IsValid() or not prim.IsA(usd_shade.Material):
            raise ValueError(
                "Material library binding target is not a valid UsdShade.Material: "
                f"{library_material_path}"
            )
        return usd_shade.Material(prim)

    name = "PreviewMaterial"
    if isinstance(override.material, dict):
        name = str(
            override.material.get("display_name")
            or override.material.get("name")
            or "PreviewMaterial"
        )
    identity_path = override.prim_path
    material_name = _preview_material_prim_name(name, identity_path)
    if root_prim_path and root_prim_path != "/":
        material_path = f"{root_prim_path}/PreviewMaterials/{material_name}"
    else:
        material_path = f"/SessionPreview/Materials/{material_name}"
    stage.DefinePrim(str(sdf.Path(material_path).GetParentPath()), "Scope")
    material = usd_shade.Material.Define(stage, material_path)
    shader = usd_shade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    color = _material_color(override.material)
    shader.CreateInput("diffuseColor", sdf.ValueTypeNames.Color3f).Set(gf.Vec3f(*color))
    shader.CreateInput("roughness", sdf.ValueTypeNames.Float).Set(
        _material_float(override.material, "roughness", 0.45)
    )
    shader.CreateInput("metallic", sdf.ValueTypeNames.Float).Set(
        _material_float(override.material, "metallic", 0.0)
    )
    shader.CreateOutput("surface", sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _material_library_paths(
    overrides: list[MaterialOverride], *, source_path: Path
) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for override in overrides:
        library_path = _material_library_path(
            override.material, source_path=source_path
        )
        if library_path is None or library_path in seen:
            continue
        seen.add(library_path)
        paths.append(library_path)
    return paths


def _material_library_path(material: object, *, source_path: Path) -> Path | None:
    if not isinstance(material, dict):
        return None
    if (
        material.get("source") != "material_library"
        and "library_path" not in material
        and "material_library_path" not in material
    ):
        return None
    raw_path = material.get("library_path") or material.get("material_library_path")
    if raw_path is None and material.get("source") == "material_library":
        raw_path = material.get("library")
    if not isinstance(raw_path, str) or not raw_path:
        return None

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        _reject_symlink_file(path)
        resolved = path.resolve()
        display_path = resolved
    else:
        unresolved_candidate = source_path.parent / path
        _reject_symlink_file(unresolved_candidate)
        source_candidate = unresolved_candidate.resolve()
        if not source_candidate.exists():
            raise FileNotFoundError(
                f"Material library USD does not exist: {source_candidate}"
            )
        resolved = source_candidate
        display_path = source_candidate

    if not resolved.exists():
        raise FileNotFoundError(f"Material library USD does not exist: {display_path}")
    if not resolved.is_file():
        raise ValueError(f"Material library path is not a file: {resolved}")
    if resolved.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(
            f"Material library path must end with .usd, .usda, or .usdc: {resolved}"
        )
    _validate_material_library_root(resolved, source_path=source_path)
    _validate_material_library_asset_path(resolved)
    return resolved


def _reject_symlink_file(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Material library path must not be a symlink: {path}")


def _validate_material_library_asset_path(path: Path) -> None:
    _validate_preview_sublayer_asset_path(path, label="Material library")


def _validate_preview_sublayer_asset_path(path: Path, *, label: str) -> None:
    path_text = str(path)
    if USD_ASSET_PATH_UNSAFE_CHARS_RE.search(path_text):
        raise ValueError(
            f"{label} path contains USD-special characters that are not "
            f"supported in preview sublayers: {path}"
        )


def _validate_material_library_root(path: Path, *, source_path: Path) -> None:
    roots = _material_library_roots_from_env(source_path=source_path)
    if any(path.is_relative_to(root) for root in roots):
        return
    allowed = ", ".join(str(root) for root in roots)
    raise ValueError(
        f"Material library path is outside {MATERIAL_LIBRARY_ROOTS_ENV}: "
        f"{path}; allowed roots: {allowed}"
    )


def _material_library_roots_from_env(*, source_path: Path) -> list[Path]:
    raw = os.environ.get(MATERIAL_LIBRARY_ROOTS_ENV, "")
    # Roots are resolved once per validation; operators should point the env at
    # canonical directories rather than symlinks that may be retargeted later.
    roots = [
        Path(item.strip()).expanduser().resolve()
        for item in raw.split(",")
        if item.strip()
    ]
    return roots or [source_path.parent.resolve()]


def _default_workspace_root() -> Path:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return Path(runtime_dir).expanduser() / "content-workbench"
    try:
        suffix = str(os.getuid())
    except AttributeError:
        suffix = os.environ.get("USERNAME") or os.environ.get("USER") or "user"
    safe_suffix = re.sub(r"[^A-Za-z0-9_.-]", "_", suffix)
    return Path(tempfile.gettempdir()) / f"content-workbench-{safe_suffix}"


def _prepare_workspace_root(path: Path) -> Path:
    workspace_root = path.expanduser()
    if workspace_root.is_symlink():
        raise ValueError(f"Workspace root must not be a symlink: {workspace_root}")
    workspace_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    if workspace_root.is_symlink():
        raise ValueError(f"Workspace root must not be a symlink: {workspace_root}")
    if not workspace_root.is_dir():
        raise ValueError(f"Workspace root is not a directory: {workspace_root}")
    try:
        os.chmod(workspace_root, 0o700)
    except OSError as exc:
        logger.warning(
            "Unable to chmod Content Workbench workspace root %s to 0700: %s",
            workspace_root,
            exc,
        )
    return workspace_root


def _material_library_material_path(material: object) -> str | None:
    if not isinstance(material, dict):
        return None
    raw_path = material.get("material_path") or material.get("binding_path")
    if isinstance(raw_path, str) and raw_path.startswith("/"):
        return raw_path

    is_library_override = (
        material.get("source") == "material_library"
        or "library_path" in material
        or "material_library_path" in material
    )
    if not is_library_override:
        return None

    raw_name = (
        material.get("material_name")
        or material.get("name")
        or material.get("display_name")
    )
    if not isinstance(raw_name, str) or not raw_name:
        raise ValueError(
            "Material library override requires material_path, binding_path, "
            "material_name, or name"
        )
    return f"/World/Looks/{_safe_prim_name(raw_name)}"


def _instance_root_paths_for_overrides(stage, overrides, usd, usd_geom) -> list[str]:
    paths: set[str] = set()
    for override in overrides:
        target_prim = stage.GetPrimAtPath(override.prim_path)
        if not target_prim.IsValid():
            continue
        instance_root = _instance_root_for_proxy(target_prim)
        if instance_root is not None:
            paths.add(str(instance_root.GetPath()))
        for prim in _target_mesh_prims(target_prim, usd_geom, usd=usd):
            instance_root = _instance_root_for_proxy(prim)
            if instance_root is not None:
                paths.add(str(instance_root.GetPath()))
    return sorted(paths, key=lambda item: (item.count("/"), item))


def _instance_root_for_proxy(prim) -> object | None:
    if not prim.IsInstanceProxy():
        return None
    current = prim
    while current and current.IsValid() and current.IsInstanceProxy():
        current = current.GetParent()
    if current and current.IsValid():
        return current
    return None


def _target_mesh_prims(target_prim, usd_geom, *, usd=None) -> list[object]:
    if target_prim.IsA(usd_geom.Mesh) or target_prim.IsA(usd_geom.Subset):
        return [target_prim]
    if usd is None:
        from pxr import Usd

        usd = Usd
    descendants = list(usd.PrimRange(target_prim, usd.TraverseInstanceProxies()))
    return [
        prim for prim in descendants if prim != target_prim and prim.IsA(usd_geom.Mesh)
    ]


def _bind_preview_material(
    prim,
    material,
    usd_shade,
    *,
    unbind_existing: bool,
    fallback_bound_paths: set[str],
) -> None:
    target_prim = prim
    binding_strength = None
    if prim.IsInstanceProxy():
        instance_root = _instance_root_for_proxy(prim)
        if instance_root is None:
            raise ValueError(
                f"Cannot bind material to instance proxy: {prim.GetPath()}"
            )
        target_prim = instance_root
        binding_strength = usd_shade.Tokens.strongerThanDescendants

    target_path = str(target_prim.GetPath())
    if binding_strength and target_path in fallback_bound_paths:
        return

    binding_api = usd_shade.MaterialBindingAPI.Apply(target_prim)
    if unbind_existing:
        binding_api.GetDirectBindingRel().SetTargets([])
    if binding_strength:
        binding_api.Bind(material, binding_strength)
        fallback_bound_paths.add(target_path)
    else:
        binding_api.Bind(material)


def _apply_visibility_state(stage, hidden_prims, isolated_prims, usd_geom) -> None:
    for path in hidden_prims:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            usd_geom.Imageable(prim).MakeInvisible()

    if not isolated_prims:
        return

    isolated = tuple(isolated_prims)
    for prim in stage.TraverseAll():
        imageable = usd_geom.Imageable(prim)
        if not imageable:
            continue
        path = str(prim.GetPath())
        if not (
            _path_is_in_isolation(path, isolated)
            or _path_is_isolation_ancestor(path, isolated)
        ):
            imageable.MakeInvisible()


def _path_is_in_isolation(path: str, isolated: tuple[str, ...]) -> bool:
    if "/" in isolated:
        return True
    return any(path == item or path.startswith(f"{item}/") for item in isolated)


def _path_is_isolation_ancestor(path: str, isolated: tuple[str, ...]) -> bool:
    if path == "/":
        return True
    return any(item.startswith(f"{path}/") for item in isolated)


def _prune_preview_scenes(
    preview_dir: Path,
    *,
    keep_path: Path,
    protected_paths: set[Path] | None = None,
    retention_count: int = PREVIEW_SCENE_RETENTION_COUNT,
) -> None:
    if retention_count <= 0:
        return
    protected = {path.resolve() for path in protected_paths or set()}
    preview_paths = sorted(
        (
            path
            for path in preview_dir.glob("preview-*.usda")
            if (
                path.is_file()
                and path.resolve() != keep_path.resolve()
                and path.resolve() not in protected
            )
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for stale_path in preview_paths[max(0, retention_count - 1) :]:
        try:
            stale_path.unlink()
        except FileNotFoundError:
            continue


def _safe_prim_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not name:
        return "PreviewMaterial"
    if name[0].isdigit():
        name = f"Material_{name}"
    return name


def _preview_material_prim_name(name: str, prim_path: str) -> str:
    prim_label = _safe_prim_name(prim_path.strip("/").replace("/", "_") or "Root")
    prim_hash = hashlib.sha256(prim_path.encode("utf-8")).hexdigest()[:8]
    return _safe_prim_name(f"{name}_{prim_label}_{prim_hash}")


def _material_color(material: object) -> tuple[float, float, float]:
    if isinstance(material, dict):
        raw = (
            material.get("diffuse_color")
            or material.get("preview_color")
            or material.get("color")
        )
        if raw is not None:
            if not isinstance(raw, list | tuple):
                raise ValueError(
                    "material color values must be a list or tuple with 3 values"
                )
            if len(raw) != 3:
                raise ValueError("material color values must contain exactly 3 values")
            return tuple(_material_color_component(value) for value in raw)
    return (0.82, 0.82, 0.82)


def _material_color_component(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("material color values must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError("material color values must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError("material color values must be between 0 and 1")
    return numeric


def _material_float(material: object, key: str, default: float) -> float:
    if isinstance(material, dict):
        raw = material.get(key, default)
        if isinstance(raw, int | float):
            return float(raw)
    return default


def _apply_selection_mode(
    current_paths: list[str],
    picked_paths: list[str],
    mode: str,
) -> list[str]:
    if mode == "add":
        return sorted(set(current_paths) | set(picked_paths))
    if mode == "subtract":
        return [path for path in current_paths if path not in set(picked_paths)]
    if mode == "clear_on_miss" and not picked_paths:
        return []
    return list(picked_paths)


_RENDER_QUALITY_PRESETS: dict[RenderQuality, tuple[str, int]] = {
    RenderQuality.INTERACTIVE: ("rt2", 16),
    RenderQuality.INSPECTION: ("rt2", 64),
    RenderQuality.FINAL: ("rt2", 256),
}
MAX_RENDER_FRAME_SPEC_COUNT = 10_000


def _open_render_stage(scene_path: Path) -> Any:
    from pxr import Usd

    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {scene_path}")
    return stage


def _stage_time_codes_per_second(
    scene_path: Path, *, stage: Any | None = None
) -> float:
    if stage is None:
        stage = _open_render_stage(scene_path)
    fps = float(stage.GetTimeCodesPerSecond() or 24.0)
    if fps <= 0:
        raise ValueError(f"USD stage has invalid timeCodesPerSecond: {fps}")
    return fps


def _resolve_render_frame_numbers(
    scene_path: Path,
    frames: str | None,
    *,
    infer_from_scene: bool,
    max_duration_seconds: float | None,
    fps: float,
    stage: Any | None = None,
) -> list[int]:
    if frames is not None:
        frame_numbers = _parse_frame_spec(frames)
    elif infer_from_scene:
        frame_numbers = _infer_stage_frame_numbers(scene_path, stage=stage)
    else:
        frame_numbers = _parse_frame_spec("0")
    if max_duration_seconds is not None:
        _validate_frame_duration_cap(frame_numbers, fps, max_duration_seconds)
    return frame_numbers


def _parse_frame_spec(frames: str) -> list[int]:
    value = str(frames or "").strip()
    if not value:
        raise ValueError("frames must not be empty")
    if ":" in value:
        start_raw, end_raw = value.split(":", 1)
        start = int(start_raw.strip())
        end = int(end_raw.strip())
        if end < start:
            raise ValueError("frame range end must be >= start")
        frame_count = end - start + 1
        if frame_count > MAX_RENDER_FRAME_SPEC_COUNT:
            raise ValueError(
                f"frame range must include at most {MAX_RENDER_FRAME_SPEC_COUNT} frames"
            )
        return list(range(start, end + 1))
    if "," in value:
        frame_numbers = [int(part.strip()) for part in value.split(",") if part.strip()]
        if len(frame_numbers) > MAX_RENDER_FRAME_SPEC_COUNT:
            raise ValueError(
                f"frame list must include at most {MAX_RENDER_FRAME_SPEC_COUNT} frames"
            )
        return frame_numbers
    return [int(value)]


def _format_frame_spec(frame_numbers: list[int]) -> str:
    ordered = sorted(frame_numbers)
    if not ordered:
        return ""
    if len(ordered) == 1:
        return str(ordered[0])
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}:{ordered[-1]}"
    return ",".join(str(frame) for frame in ordered)


def _infer_stage_frame_numbers(
    scene_path: Path, *, stage: Any | None = None
) -> list[int]:
    if stage is None:
        stage = _open_render_stage(scene_path)
    if stage.HasAuthoredTimeCodeRange():
        start = _as_integral_frame(stage.GetStartTimeCode(), "startTimeCode")
        end = _as_integral_frame(stage.GetEndTimeCode(), "endTimeCode")
        if end < start:
            raise ValueError(
                f"Authored time-code range is reversed (start={start}, end={end})"
            )
        return list(range(start, end + 1))

    authored: set[int] = set()
    for prim in stage.TraverseAll():
        if prim.IsInstanceProxy():
            continue
        for attr in prim.GetAttributes():
            for time_code in attr.GetTimeSamples():
                authored.add(_as_integral_frame(time_code, attr.GetPath().pathString))
    if authored:
        ordered = sorted(authored)
        return list(range(ordered[0], ordered[-1] + 1))

    return [_as_integral_frame(stage.GetStartTimeCode(), "startTimeCode")]


def _as_integral_frame(value: Any, label: str) -> int:
    number = float(value)
    rounded = round(number)
    if abs(number - rounded) > 1e-6:
        raise ValueError(
            f"{label} uses non-integer time code {number}; "
            "current render backends accept integer frame selections"
        )
    return int(rounded)


def _validate_frame_duration_cap(
    frame_numbers: list[int], fps: float, max_duration_seconds: float
) -> None:
    max_frames = max(1, int(max_duration_seconds * fps))
    if len(frame_numbers) > max_frames + 1:
        raise ValueError(
            f"frame count {len(frame_numbers)} exceeds cap of {max_frames + 1} "
            f"(= max_duration_seconds {max_duration_seconds} x fps {fps} "
            "+ 1-frame closed-interval tolerance)"
        )
    if len(frame_numbers) > max_frames:
        logger.warning(
            "Frame-count discrepancy: recording has %d time samples, expected "
            "cap is %d (%g s x %g fps). Rendering all %d frames.",
            len(frame_numbers),
            max_frames,
            max_duration_seconds,
            fps,
            len(frame_numbers),
        )


def _sanitize_camera_prim_name(direction: str) -> str:
    sanitized = "".join(
        c if c.isalnum() or c == "_" else "_"
        for c in direction.replace("+", "plus_").replace("-", "minus_")
    )
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    sanitized = sanitized.strip("_") or "axis"
    if sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def _camera_path_candidates(camera: str) -> list[str]:
    if camera.startswith("/"):
        return [camera]
    if camera.startswith("Cameras/"):
        return [f"/{camera}"]
    return [f"/Cameras/{_sanitize_camera_prim_name(camera)}", camera]


def _resolve_render_camera_path(
    scene_path: Path,
    camera: str,
    *,
    stage: Any | None = None,
) -> str:
    from pxr import UsdGeom

    if stage is None:
        stage = _open_render_stage(scene_path)
    available = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)
    ]
    for candidate in _camera_path_candidates(camera):
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid() and prim.IsA(UsdGeom.Camera):
            return candidate
    raise ValueError(f"Camera not found: {camera}. Available cameras: {available}")


def _write_mp4(frame_paths: list[Path], output_path: Path, fps: float) -> bool:
    try:
        import imageio.v3 as iio  # type: ignore[import-not-found]
        import numpy as np
        from PIL import Image
    except ImportError:
        logger.info("imageio is unavailable; skipping mp4 creation")
        return False

    try:
        frames = [np.asarray(Image.open(path).convert("RGB")) for path in frame_paths]
        iio.imwrite(output_path, frames, fps=fps)
        return True
    except Exception as exc:  # pragma: no cover - depends on optional codecs
        logger.warning("Failed to write mp4 render %s: %s", output_path, exc)
        return False


def _render_settings_from_request(
    request: RenderRequest | RenderFramesRequest,
) -> tuple[str, int]:
    mode, updates = _RENDER_QUALITY_PRESETS[request.render_quality]
    if request.ovrtx_render_mode is not None:
        mode = _normalize_ovrtx_render_mode(request.ovrtx_render_mode)
    if request.ovrtx_num_sensor_updates is not None:
        updates = int(request.ovrtx_num_sensor_updates)
    return mode, updates


def _normalize_ovrtx_render_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in {"rt2", "pt"}:
        raise ValueError(f"Unsupported OvRTX render mode: {value}")
    return mode


def _sanitize_camera(camera: CameraState) -> CameraState:
    target = _finite_vec3(camera.target, default=[0.0, 0.0, 0.0])
    return CameraState(
        target=target,
        distance=_clamped_camera_distance(camera.distance, default=6.0),
        yaw_degrees=float(camera.yaw_degrees),
        pitch_degrees=max(-89.0, min(89.0, float(camera.pitch_degrees))),
        focal_length=max(1e-3, float(camera.focal_length)),
        horizontal_aperture=max(1e-3, float(camera.horizontal_aperture)),
        last_framed_prim_path=camera.last_framed_prim_path,
    )


def _camera_state_from_bounds(
    bounds: dict[str, object] | None,
    *,
    direction: str,
    margin: float,
    width: int,
    height: int,
    focus_path: str | None,
) -> CameraState:
    direction_vec = _parse_direction(direction)
    yaw = math.degrees(math.atan2(direction_vec[1], direction_vec[0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(direction_vec[2])))))
    center = _bounds_center(bounds)
    radius = _bounds_radius(bounds)
    camera = CameraState(
        target=center,
        distance=_framing_distance(
            radius=radius,
            margin=margin,
            width=width,
            height=height,
            focal_length=50.0,
            horizontal_aperture=36.0,
        ),
        yaw_degrees=yaw,
        pitch_degrees=pitch,
        last_framed_prim_path=focus_path,
    )
    return _sanitize_camera(camera)


def _frame_existing_camera(
    camera: CameraState,
    bounds: dict[str, object] | None,
    *,
    margin: float,
    width: int,
    height: int,
    focus_path: str | None,
) -> CameraState:
    radius = _bounds_radius(bounds)
    framed = camera.model_copy(deep=True)
    framed.target = _bounds_center(bounds)
    framed.distance = _framing_distance(
        radius=radius,
        margin=margin,
        width=width,
        height=height,
        focal_length=framed.focal_length,
        horizontal_aperture=framed.horizontal_aperture,
    )
    framed.last_framed_prim_path = focus_path
    return _sanitize_camera(framed)


def _camera_transform_from_state(camera: CameraState) -> list[list[float]]:
    clean = _sanitize_camera(camera)
    right, up, camera_z, eye = _camera_axes(clean)
    return [
        [right[0], right[1], right[2], 0.0],
        [up[0], up[1], up[2], 0.0],
        [camera_z[0], camera_z[1], camera_z[2], 0.0],
        [eye[0], eye[1], eye[2], 1.0],
    ]


def _orbit_camera(
    camera: CameraState,
    *,
    yaw_delta_degrees: float,
    pitch_delta_degrees: float,
) -> None:
    camera.yaw_degrees = float(camera.yaw_degrees) + yaw_delta_degrees
    camera.pitch_degrees = max(
        -89.0,
        min(89.0, float(camera.pitch_degrees) + pitch_delta_degrees),
    )


def _pan_camera(
    camera: CameraState,
    *,
    right_delta: float,
    up_delta: float,
    scale: float,
) -> None:
    if (
        not math.isfinite(right_delta)
        or not math.isfinite(up_delta)
        or not math.isfinite(scale)
    ):
        raise ValueError("pan values must be finite")
    clean = _sanitize_camera(camera)
    right, up, _camera_z, _eye = _camera_axes(clean)
    world_scale = max(0.0, scale) * max(1e-3, float(camera.distance))
    camera.target = [
        clean.target[index]
        + (right[index] * right_delta + up[index] * up_delta) * world_scale
        for index in range(3)
    ]


def _dolly_camera(
    camera: CameraState,
    *,
    amount: float,
    factor: object,
) -> None:
    if isinstance(factor, bool):
        raise ValueError("dolly factor must be a finite positive number")
    if isinstance(factor, int | float):
        multiplier = float(factor)
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("dolly factor must be a finite positive number")
        min_factor = math.pow(2.0, -MAX_DOLLY_EXPONENT)
        max_factor = math.pow(2.0, MAX_DOLLY_EXPONENT)
        multiplier = max(min_factor, min(max_factor, multiplier))
    else:
        clean_amount = _finite_float(amount, default=0.0)
        clean_amount = max(-MAX_DOLLY_EXPONENT, min(MAX_DOLLY_EXPONENT, clean_amount))
        multiplier = math.pow(2.0, clean_amount)
    current_distance = _clamped_camera_distance(camera.distance, default=6.0)
    camera.distance = _clamped_camera_distance(
        current_distance * multiplier,
        default=MAX_CAMERA_DISTANCE,
    )


def _camera_axes(
    camera: CameraState,
) -> tuple[list[float], list[float], list[float], list[float]]:
    direction = _camera_direction(camera)
    target = _finite_vec3(camera.target, default=[0.0, 0.0, 0.0])
    eye = [
        target[index] + direction[index] * float(camera.distance) for index in range(3)
    ]
    forward = _normalize([target[index] - eye[index] for index in range(3)])
    camera_z = [-forward[0], -forward[1], -forward[2]]
    world_up = [0.0, 0.0, 1.0]
    right = _normalize(_cross(world_up, camera_z))
    if _length(right) < 1e-6:
        right = [1.0, 0.0, 0.0]
    up = _normalize(_cross(camera_z, right))
    return right, up, camera_z, eye


def _camera_direction(camera: CameraState) -> list[float]:
    yaw = math.radians(float(camera.yaw_degrees))
    pitch = math.radians(float(camera.pitch_degrees))
    horizontal = math.cos(pitch)
    return _normalize(
        [
            horizontal * math.cos(yaw),
            horizontal * math.sin(yaw),
            math.sin(pitch),
        ]
    )


def _bounds_center(bounds: dict[str, object] | None) -> list[float]:
    if not bounds:
        return [0.0, 0.0, 0.0]
    return _finite_vec3(bounds.get("center"), default=[0.0, 0.0, 0.0])


def _bounds_radius(bounds: dict[str, object] | None) -> float:
    if not bounds:
        return 1.0
    min_v = _finite_vec3(bounds.get("min"), default=[-1.0, -1.0, -1.0])
    max_v = _finite_vec3(bounds.get("max"), default=[1.0, 1.0, 1.0])
    extent = [max_v[index] - min_v[index] for index in range(3)]
    return max(1e-3, math.sqrt(sum(value * value for value in extent)) * 0.5)


def _framing_distance(
    *,
    radius: float,
    margin: float,
    width: int,
    height: int,
    focal_length: float,
    horizontal_aperture: float,
) -> float:
    vertical_aperture = horizontal_aperture * float(height) / float(width)
    fov_y = 2.0 * math.atan(vertical_aperture / (2.0 * focal_length))
    return max(radius * 2.0, radius / math.tan(fov_y * 0.5) * float(margin))


def _finite_vec3(value: object, *, default: list[float]) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) < 3:
        return list(default)
    result = []
    for index in range(3):
        try:
            item = float(value[index])
        except (TypeError, ValueError):
            return list(default)
        if not math.isfinite(item):
            return list(default)
        result.append(item)
    return result


def _finite_float(value: object, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _clamped_camera_distance(value: object, *, default: float) -> float:
    clean = _finite_float(value, default=default)
    return max(MIN_CAMERA_DISTANCE, min(MAX_CAMERA_DISTANCE, clean))


def _parse_direction(value: str) -> list[float]:
    compact = value.replace(" ", "").lower()
    if not compact:
        return _normalize([1.0, -1.0, 1.0])
    token_re = re.compile(r"([+-])(\d+(?:\.\d+)?)?([xyz])")
    matches = list(token_re.finditer(compact))
    if not matches or "".join(match.group(0) for match in matches) != compact:
        return _normalize([1.0, -1.0, 1.0])
    vec = [0.0, 0.0, 0.0]
    axis_index = {"x": 0, "y": 1, "z": 2}
    for match in matches:
        sign, magnitude, axis = match.groups()
        weight = float(magnitude) if magnitude else 1.0
        if sign == "-":
            weight = -weight
        vec[axis_index[axis]] += weight
    return _normalize(vec)


def _normalize(vec: list[float]) -> list[float]:
    length = _length(vec)
    if length < 1e-9:
        inv = 1.0 / math.sqrt(3.0)
        return [inv, -inv, inv]
    return [value / length for value in vec]


def _length(vec: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vec))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
