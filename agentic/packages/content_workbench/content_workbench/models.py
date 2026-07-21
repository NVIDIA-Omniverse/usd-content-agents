# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API models for the content workbench."""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .version import SERVICE_VERSION

MAX_RENDER_DIMENSION = 8192
MAX_PATH_FIELD_LENGTH = 4096
MAX_PRIM_PATH_LENGTH = 2048
MAX_SHORT_TEXT_LENGTH = 128
MAX_COMMAND_PAYLOAD_KEYS = 256
MAX_BATCH_REQUEST_ITEMS = 4096
MAX_OPTIMIZATION_CONFIG_DEPTH = 32
MAX_OPTIMIZATION_CONFIG_CONTAINER_ITEMS = 1024
MAX_OPTIMIZATION_CONFIG_TOTAL_ITEMS = 4096
MAX_PHYSICS_RUNTIME_DURATION_S = 10.0
MIN_PHYSICS_RUNTIME_DT_S = 1.0 / 1000.0
MAX_PHYSICS_RUNTIME_DT_S = 0.1
MAX_PHYSICS_RUNTIME_SAMPLE_FPS = 120

PrimPath = Annotated[str, Field(max_length=MAX_PRIM_PATH_LENGTH)]


class SessionStatus(StrEnum):
    """Lifecycle state for a scene inspection session."""

    CREATED = "created"
    READY = "ready"
    ERROR = "error"
    CLOSED = "closed"


class OptimizerBackend(StrEnum):
    """Scene Optimizer backend options exposed by the workbench."""

    LOCAL = "local"
    REMOTE = "remote"


class RenderQuality(StrEnum):
    """Still-render quality presets for agent inspection workflows."""

    INTERACTIVE = "interactive"
    INSPECTION = "inspection"
    FINAL = "final"


class CommandName(StrEnum):
    """Scene inspection commands accepted by the control plane."""

    PICK = "pick"
    FOCUS = "focus"
    FRAME = "frame"
    ORBIT = "orbit"
    PAN = "pan"
    DOLLY = "dolly"
    SET_CAMERA = "set_camera"
    SELECT = "select"
    HIDE = "hide"
    SHOW = "show"
    ISOLATE = "isolate"
    CLEAR_ISOLATION = "clear_isolation"
    MATERIAL_OVERRIDE = "material_override"
    CLEAR_MATERIAL_OVERRIDE = "clear_material_override"
    CLEAR_VISUAL_OVERRIDES = "clear_visual_overrides"
    RESET_VIEW = "reset_view"
    CHANGE_AOV = "change_aov"


class CommandStatus(StrEnum):
    """Result state for an applied command."""

    SUCCESS = "success"
    UNSUPPORTED = "unsupported"


class ViewportState(BaseModel):
    """Still-render viewport metadata."""

    mode: str = Field(default="still_render", max_length=MAX_SHORT_TEXT_LENGTH)
    width: int = Field(default=1280, ge=1)
    height: int = Field(default=720, ge=1)


class MaterialOverride(BaseModel):
    """Session-scoped material binding override."""

    model_config = ConfigDict(extra="forbid")

    prim_path: str = Field(max_length=MAX_PRIM_PATH_LENGTH)
    material: dict[str, Any]
    mode: str = Field(default="material_assignment", max_length=MAX_SHORT_TEXT_LENGTH)
    unbind_existing: bool = True
    remove_material_libraries: bool = False
    space: str = Field(default="source", max_length=MAX_SHORT_TEXT_LENGTH)
    source_prim_paths: list[str] = Field(default_factory=list, max_length=1024)
    inspection_prim_paths: list[str] = Field(default_factory=list, max_length=1024)
    # True when `source_prim_paths` was narrowed to empty by
    # `_trim_material_override_coverage` (as opposed to never having tracked
    # explicit source coverage). Distinguishes "genuinely no source coverage
    # left, do not substitute prim_path" from "no per-leaf tracking was ever
    # recorded, prim_path is the intended fallback" at the call sites that
    # otherwise fall back to `prim_path` whenever `source_prim_paths` is
    # empty.
    source_coverage_exhausted: bool = False


class CameraState(BaseModel):
    """Agent-controllable viewport camera state."""

    model_config = ConfigDict(extra="forbid")

    target: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
    )
    distance: float = Field(default=6.0, gt=0)
    yaw_degrees: float = -45.0
    pitch_degrees: float = 35.264389682754654
    focal_length: float = Field(default=50.0, gt=0)
    horizontal_aperture: float = Field(default=36.0, gt=0)
    last_framed_prim_path: str | None = Field(
        default=None,
        max_length=MAX_PRIM_PATH_LENGTH,
    )


class ViewState(BaseModel):
    """State owned by the scene inspection session."""

    camera_path: str = Field(
        default="/Session/Cameras/Main",
        max_length=MAX_PRIM_PATH_LENGTH,
    )
    camera: CameraState = Field(default_factory=CameraState)
    active_aov: str = Field(default="LdrColor", max_length=MAX_SHORT_TEXT_LENGTH)
    selected_prims: list[str] = Field(default_factory=list, max_length=1024)
    hidden_prims: list[str] = Field(default_factory=list, max_length=1024)
    isolated_prims: list[str] = Field(default_factory=list, max_length=1024)
    material_overrides: list[MaterialOverride] = Field(default_factory=list)


class ArtifactState(BaseModel):
    """Generated files for the current session."""

    workspace_dir: str | None = None
    preview_scene_path: str | None = None
    last_render_path: str | None = None
    last_render_camera_json_path: str | None = None
    last_apply_output_path: str | None = None
    last_apply_assignments_path: str | None = None
    last_apply_predictions_path: str | None = None
    optimized_scene_path: str | None = None
    material_cleared_scene_path: str | None = None
    optimization_metadata_path: str | None = None


class OptimizationState(BaseModel):
    """Scene Optimizer state for a session."""

    enabled: bool = False
    status: str = "disabled"
    source_scene_path: str | None = None
    inspection_scene_path: str | None = None
    metadata_path: str | None = None
    error: str | None = None
    operations_executed: list[Any] = Field(default_factory=list)
    correspondence_summary: dict[str, int] = Field(default_factory=dict)


class OptimizerRequestOptions(BaseModel):
    """Scene Optimizer options shared by session creation and scene loading."""

    model_config = ConfigDict(extra="forbid")

    optimize: bool = False
    optimizer_backend: OptimizerBackend | None = None
    flatten_prototypes: bool | None = None
    enable_deinstance: bool | None = None
    enable_split: bool | None = None
    enable_deduplicate: bool | None = None
    clear_materials: bool = False
    optimization_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_optimizer_operations(self) -> OptimizerRequestOptions:
        if not self.optimize:
            return self
        operations = _resolved_optimizer_operations(self.resolved_optimization_config())
        if not any(operations.values()):
            raise ValueError(
                "At least one Scene Optimizer operation must be enabled when "
                "optimize is true."
            )
        return self

    def resolved_optimization_config(self) -> dict[str, Any]:
        """Return the optimizer config passed to OptimizeUSDTask."""
        return _resolve_optimization_config(self)


class CreateSessionRequest(OptimizerRequestOptions):
    """Create a local scene inspection session."""

    scene_path: str | None = Field(default=None, max_length=MAX_PATH_FIELD_LENGTH)
    width: int = Field(default=1280, ge=1)
    height: int = Field(default=720, ge=1)


class LoadSceneRequest(OptimizerRequestOptions):
    """Load or reload a local USD scene."""

    scene_path: str = Field(max_length=MAX_PATH_FIELD_LENGTH)


def _resolve_optimization_config(request: OptimizerRequestOptions) -> dict[str, Any]:
    config = deepcopy(request.optimization_config)
    if request.optimizer_backend is not None:
        config["backend"] = request.optimizer_backend.value
    if request.flatten_prototypes is not None:
        config["flatten_prototypes"] = request.flatten_prototypes

    settings = _settings_dict(config)
    if request.enable_deinstance is not None:
        settings["enable_deinstance"] = request.enable_deinstance
    if request.enable_split is not None:
        settings["enable_split_meshes"] = request.enable_split
    if request.enable_deduplicate is not None:
        settings["enable_deduplicate"] = request.enable_deduplicate

    _normalize_scene_optimizer_setting_aliases(settings)
    if settings:
        config["scene_optimizer_settings"] = settings
    _validate_optimization_config_shape(config)
    return config


def _validate_optimization_config_shape(value: Any, *, depth: int = 0) -> int:
    if depth > MAX_OPTIMIZATION_CONFIG_DEPTH:
        raise ValueError(
            f"optimization_config exceeds maximum depth {MAX_OPTIMIZATION_CONFIG_DEPTH}"
        )
    if isinstance(value, dict):
        if len(value) > MAX_OPTIMIZATION_CONFIG_CONTAINER_ITEMS:
            raise ValueError("optimization_config contains too many keys in one object")
        total = len(value)
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("optimization_config keys must be strings")
            total += _validate_optimization_config_shape(child, depth=depth + 1)
            if total > MAX_OPTIMIZATION_CONFIG_TOTAL_ITEMS:
                raise ValueError("optimization_config contains too many items")
        return total
    if isinstance(value, list):
        if len(value) > MAX_OPTIMIZATION_CONFIG_CONTAINER_ITEMS:
            raise ValueError("optimization_config contains too many list items")
        total = len(value)
        for child in value:
            total += _validate_optimization_config_shape(child, depth=depth + 1)
            if total > MAX_OPTIMIZATION_CONFIG_TOTAL_ITEMS:
                raise ValueError("optimization_config contains too many items")
        return total
    return 0


def _settings_dict(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("scene_optimizer_settings")
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise ValueError(
            "optimization_config.scene_optimizer_settings must be an object"
        )
    return deepcopy(settings)


def _normalize_scene_optimizer_setting_aliases(settings: dict[str, Any]) -> None:
    aliases = {
        "enableSplitMeshes": "enable_split_meshes",
        "enable_split": "enable_split_meshes",
        "enableDeinstance": "enable_deinstance",
        "enableDeduplicate": "enable_deduplicate",
    }
    for alias, canonical in aliases.items():
        if alias not in settings:
            continue
        if canonical not in settings:
            settings[canonical] = settings[alias]
        settings.pop(alias, None)


def _resolved_optimizer_operations(config: dict[str, Any]) -> dict[str, bool]:
    settings = _settings_dict(config)
    _normalize_scene_optimizer_setting_aliases(settings)
    return {
        "deinstance": _setting_bool(settings, "enable_deinstance", default=True),
        "split": _setting_bool(settings, "enable_split_meshes", default=True),
        "deduplicate": _setting_bool(settings, "enable_deduplicate", default=True),
    }


def _setting_bool(settings: dict[str, Any], key: str, *, default: bool) -> bool:
    value = settings.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(
            f"optimization_config.scene_optimizer_settings.{key} must be a boolean"
        )
    return value


class CommandRequest(BaseModel):
    """Apply a scene inspection command."""

    model_config = ConfigDict(extra="forbid")

    command: CommandName
    payload: dict[str, Any] = Field(
        default_factory=dict,
        max_length=MAX_COMMAND_PAYLOAD_KEYS,
    )


class SessionResponse(BaseModel):
    """Public session state."""

    session_id: str
    status: SessionStatus
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


DEFAULT_HDRI_LIGHT_INTENSITY = 600.0


class RenderRequest(BaseModel):
    """Render the current session preview scene."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=1024, ge=1, le=MAX_RENDER_DIMENSION)
    height: int = Field(default=768, ge=1, le=MAX_RENDER_DIMENSION)
    direction: str | None = Field(default=None, max_length=MAX_SHORT_TEXT_LENGTH)
    use_session_camera: bool = True
    margin: float = Field(default=1.25, gt=0)
    hdri_light: float | None = DEFAULT_HDRI_LIGHT_INTENSITY
    dome_light: float | None = None
    distant_light: float | None = None
    render_quality: RenderQuality = RenderQuality.INSPECTION
    ovrtx_render_mode: str | None = Field(
        default=None, max_length=MAX_SHORT_TEXT_LENGTH
    )
    ovrtx_num_sensor_updates: int | None = Field(default=None, ge=1)
    focus: str | None = Field(default=None, max_length=MAX_PRIM_PATH_LENGTH)
    save_camera_json: bool = True


class RenderResponse(BaseModel):
    """Rendered image result."""

    session_id: str
    status: str
    preview_scene_path: str
    image_path: str
    image_url: str | None = None
    camera_json_path: str | None = None
    camera_json_url: str | None = None
    renderer: str = "ovrtx"
    render_product_path: str
    render_quality: RenderQuality
    ovrtx_render_mode: str
    ovrtx_num_sensor_updates: int
    active_aov: str = "LdrColor"
    elapsed_seconds: float


class RenderFramesRequest(BaseModel):
    """Render an ordered frame sequence from a session or supplied USD scene."""

    model_config = ConfigDict(extra="forbid")

    scene_path: str | None = Field(default=None, max_length=MAX_PATH_FIELD_LENGTH)
    output_dir: str | None = Field(default=None, max_length=MAX_PATH_FIELD_LENGTH)
    width: int = Field(default=1024, ge=1, le=MAX_RENDER_DIMENSION)
    height: int = Field(default=768, ge=1, le=MAX_RENDER_DIMENSION)
    frames: str | None = Field(default=None, max_length=MAX_SHORT_TEXT_LENGTH)
    directions: list[str] | None = None
    camera_path: str | None = Field(default=None, max_length=MAX_PRIM_PATH_LENGTH)
    use_session_camera: bool = False
    margin: float = Field(default=1.25, gt=0)
    focus: str | None = Field(default=None, max_length=MAX_PRIM_PATH_LENGTH)
    hdri_light: float | None = DEFAULT_HDRI_LIGHT_INTENSITY
    dome_light: float | None = None
    distant_light: float | None = None
    render_quality: RenderQuality = RenderQuality.INSPECTION
    ovrtx_render_mode: str | None = Field(
        default=None, max_length=MAX_SHORT_TEXT_LENGTH
    )
    ovrtx_num_sensor_updates: int | None = Field(default=None, ge=1)
    save_camera_json: bool = False
    make_mp4: bool = False
    max_duration_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_camera_mode(self) -> RenderFramesRequest:
        if self.camera_path is not None and (
            self.directions or self.use_session_camera
        ):
            raise ValueError(
                "camera_path cannot be combined with directions or use_session_camera"
            )
        return self


class RenderFramesResponse(BaseModel):
    """Rendered frame sequence result."""

    session_id: str
    status: str
    preview_scene_path: str
    frame_paths: list[str]
    frame_urls: list[str]
    camera_json_paths: list[str] = Field(default_factory=list)
    camera_json_urls: list[str] = Field(default_factory=list)
    mp4_paths: list[str] = Field(default_factory=list)
    mp4_urls: list[str] = Field(default_factory=list)
    renderer: str = "ovrtx"
    render_product_path: str
    render_quality: RenderQuality
    ovrtx_render_mode: str
    ovrtx_num_sensor_updates: int
    active_aov: str = "LdrColor"
    elapsed_seconds: float


class PickRequest(BaseModel):
    """Pick a rendered viewport pixel using the current camera."""

    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int | None = Field(default=None, ge=1, le=MAX_RENDER_DIMENSION)
    height: int | None = Field(default=None, ge=1, le=MAX_RENDER_DIMENSION)
    update_selection: bool = True
    mode: str = Field(default="replace", max_length=MAX_SHORT_TEXT_LENGTH)
    ovrtx_num_sensor_updates: int = Field(default=1, ge=1)
    ovrtx_render_mode: str = Field(default="rt2", max_length=MAX_SHORT_TEXT_LENGTH)


class PickResponse(BaseModel):
    """Result of a viewport pixel pick."""

    session_id: str
    x: int
    y: int
    prim_paths: list[str] = Field(default_factory=list)
    selected_prims: list[str] = Field(default_factory=list)
    render_product_path: str
    elapsed_seconds: float


class PathTranslationRequest(BaseModel):
    """Translate a path between source and inspection scene spaces."""

    model_config = ConfigDict(extra="forbid")

    prim_path: str = Field(max_length=MAX_PRIM_PATH_LENGTH)
    source_space: Literal["source", "inspection"] = "inspection"
    target_space: Literal["source", "inspection"] = "source"


class PathTranslationResponse(BaseModel):
    """Path translation result for source/inspection coordinate spaces."""

    session_id: str
    input_path: str
    source_space: str
    target_space: str
    source_paths: list[str] = Field(default_factory=list)
    inspection_paths: list[str] = Field(default_factory=list)
    ambiguous: bool = False
    optimization: OptimizationState


class BatchPrimPathsRequest(BaseModel):
    """Batch request for endpoints that read several prim paths."""

    model_config = ConfigDict(extra="forbid")

    prim_paths: list[PrimPath] = Field(
        min_length=1,
        max_length=MAX_BATCH_REQUEST_ITEMS,
    )


class BatchPathTranslationRequest(BaseModel):
    """Batch request for source/inspection path translations."""

    model_config = ConfigDict(extra="forbid")

    requests: list[PathTranslationRequest] = Field(
        min_length=1,
        max_length=MAX_BATCH_REQUEST_ITEMS,
    )


class BatchPathTranslationResponse(BaseModel):
    """Path translation results for several prim paths."""

    session_id: str
    results: list[PathTranslationResponse] = Field(default_factory=list)


class SceneSnapshotRequest(BaseModel):
    """Build a compact scene inspection snapshot in one Workbench call."""

    model_config = ConfigDict(extra="forbid")

    root_prim_path: str | None = Field(default=None, max_length=MAX_PRIM_PATH_LENGTH)
    include_properties: bool = True
    include_material_bindings: bool = True
    include_path_translations: bool = True
    include_candidate_hints: bool = True
    max_prims: int = Field(
        default=MAX_BATCH_REQUEST_ITEMS,
        ge=1,
        le=MAX_BATCH_REQUEST_ITEMS,
    )


class SceneSnapshotNode(BaseModel):
    """Flattened tree node with direct child paths."""

    path: str = Field(max_length=MAX_PRIM_PATH_LENGTH)
    name: str = Field(max_length=MAX_SHORT_TEXT_LENGTH)
    type_name: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)
    active: bool
    loaded: bool
    children: bool
    child_paths: list[str] = Field(default_factory=list)


class SceneSnapshotCandidate(BaseModel):
    """Workbench-generated hint for visible/renderable material candidates."""

    inspection_path: str = Field(max_length=MAX_PRIM_PATH_LENGTH)
    source_paths: list[str] = Field(default_factory=list)
    type_name: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)
    active: bool = True
    loaded: bool = True
    effective_visible: bool = True
    bounds_center: list[float] | None = None
    bounds_size: list[float] | None = None
    material_binding_type: str = Field(default="none", max_length=MAX_SHORT_TEXT_LENGTH)
    bound_material_path: str | None = Field(
        default=None,
        max_length=MAX_PRIM_PATH_LENGTH,
    )
    binding_source_path: str | None = Field(
        default=None,
        max_length=MAX_PRIM_PATH_LENGTH,
    )
    direct_targets: list[str] = Field(default_factory=list)
    material_override: MaterialOverride | None = None
    ambiguous_translation: bool = False
    candidate_reason: str = Field(default="", max_length=MAX_SHORT_TEXT_LENGTH)


class SceneSnapshotResponse(BaseModel):
    """One-call scene snapshot for agent inspection workflows."""

    session_id: str
    root_prim_path: str
    source_scene_path: str | None = None
    inspection_scene_path: str | None = None
    optimization: OptimizationState
    paths: list[str] = Field(default_factory=list)
    nodes: list[SceneSnapshotNode] = Field(default_factory=list)
    properties: list[PropertiesResponse] = Field(default_factory=list)
    material_bindings: list[MaterialBindingResponse] = Field(default_factory=list)
    path_translations: list[PathTranslationResponse] = Field(default_factory=list)
    candidates: list[SceneSnapshotCandidate] = Field(default_factory=list)
    excluded_non_candidates: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class CommandResponse(BaseModel):
    """Command result plus updated session state."""

    session_id: str
    command: CommandName
    result: CommandStatus
    message: str | None = None
    session: SessionResponse


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    service: str = "content-workbench"
    version: str = SERVICE_VERSION
    active_sessions: int = 0
    output_roots: list[str] = Field(default_factory=list)


class TreeChild(BaseModel):
    """One child row in the stage hierarchy."""

    name: str
    path: str
    type_name: str
    active: bool
    loaded: bool
    children: bool


class TreeResponse(BaseModel):
    """Lazy hierarchy response."""

    prim_path: str
    children: list[TreeChild]


class PropertiesResponse(BaseModel):
    """Serializable USD prim properties."""

    prim_path: str
    properties: dict[str, Any]
    truncated: bool = False


class MaterialBindingResponse(BaseModel):
    """Material binding state for a prim."""

    prim_path: str
    binding_type: str
    bound_material_path: str | None = None
    binding_source_path: str | None = None
    relationship_path: str | None = None
    direct_targets: list[str] = Field(default_factory=list)
    material_override: MaterialOverride | None = None


class BatchPropertiesResponse(BaseModel):
    """Properties for several prim paths."""

    session_id: str
    results: list[PropertiesResponse] = Field(default_factory=list)


class BatchMaterialBindingResponse(BaseModel):
    """Material binding state for several prim paths."""

    session_id: str
    results: list[MaterialBindingResponse] = Field(default_factory=list)


class MaterialAssignmentRecord(BaseModel):
    """Current Workbench material assignment state."""

    assignment_id: str = Field(max_length=MAX_SHORT_TEXT_LENGTH)
    prim_path: str = Field(max_length=MAX_PRIM_PATH_LENGTH)
    space: str = Field(default="source", max_length=MAX_SHORT_TEXT_LENGTH)
    source_prim_paths: list[str] = Field(default_factory=list, max_length=1024)
    inspection_prim_paths: list[str] = Field(default_factory=list, max_length=1024)
    material: dict[str, Any]
    material_library_path: str | None = Field(
        default=None,
        max_length=MAX_PATH_FIELD_LENGTH,
    )
    material_path: str | None = Field(default=None, max_length=MAX_PRIM_PATH_LENGTH)
    mode: str = Field(default="material_assignment", max_length=MAX_SHORT_TEXT_LENGTH)
    unbind_existing: bool = True
    remove_material_libraries: bool = False


class MaterialAssignmentsResponse(BaseModel):
    """Material assignments currently owned by the Workbench session."""

    session_id: str
    assignments: list[MaterialAssignmentRecord] = Field(default_factory=list)


class MaterialApplyRequest(BaseModel):
    """Apply accepted Workbench material assignments to an output USD/USDZ."""

    model_config = ConfigDict(extra="forbid")

    output_usd_path: str | None = Field(default=None, max_length=MAX_PATH_FIELD_LENGTH)
    output_mode: Literal["layer", "composed", "flattened"] = "layer"
    material_profile: str = Field(default="auto", max_length=MAX_SHORT_TEXT_LENGTH)
    skip_instance_check: bool = False
    fail_on_invalid_assignment: bool = True
    overwrite: bool = False


class MaterialApplyResponse(BaseModel):
    """Durable material apply result."""

    session_id: str
    status: str
    input_usd_path: str
    output_usd_path: str
    output_mode: Literal["layer", "composed", "flattened"]
    material_profile: str
    assignments_path: str
    predictions_path: str
    material_library_path: str
    materials_applied: dict[str, Any] = Field(default_factory=dict)
    assignment_stats: dict[str, Any] = Field(default_factory=dict)
    applied_assignment_count: int = 0
    applied_source_prim_paths: list[str] = Field(default_factory=list)
    unbound_source_prim_paths: list[str] = Field(default_factory=list)
    skipped_assignment_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class PhysicsInspectCandidatesRequest(BaseModel):
    """Inspect mesh prims as physics-authoring candidates."""

    model_config = ConfigDict(extra="forbid")

    usd_path: str = Field(max_length=MAX_PATH_FIELD_LENGTH)
    root_prim_path: str | None = Field(default=None, max_length=MAX_PRIM_PATH_LENGTH)
    include_existing_schema: bool = True
    path_space: Literal["source", "inspection"] = "source"


class PhysicsInspectComponentsRequest(BaseModel):
    """Inspect logical physics components or authored topology."""

    model_config = ConfigDict(extra="forbid")

    usd_path: str = Field(max_length=MAX_PATH_FIELD_LENGTH)
    root_prim_path: str | None = Field(default=None, max_length=MAX_PRIM_PATH_LENGTH)
    path_space: Literal["source", "inspection"] = "source"


class PhysicsTopologyOperation(BaseModel):
    """One allowlisted operation in an accepted topology plan."""

    model_config = ConfigDict(extra="forbid")

    op: Literal[
        "ensure_rigid_body_api",
        "remove_rigid_body_api",
        "remove_fixed_joint",
    ]
    prim_path: str = Field(max_length=MAX_PRIM_PATH_LENGTH)


class PhysicsTopologyInvariants(BaseModel):
    """Postconditions that a topology derivative must preserve."""

    model_config = ConfigDict(extra="forbid")

    enabled_collider_count: int = Field(ge=0)
    reject_articulation_changes: Literal[True] = True


class PhysicsApplyTopologyPlanRequest(BaseModel):
    """Apply an explicit, digest-bound topology plan to a derivative USD."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["content-workflows.physics-topology-plan.v1"] = (
        "content-workflows.physics-topology-plan.v1"
    )
    input_usd_path: str = Field(max_length=MAX_PATH_FIELD_LENGTH)
    output_usd_path: str | None = Field(default=None, max_length=MAX_PATH_FIELD_LENGTH)
    expected_source_digest: str = Field(max_length=MAX_SHORT_TEXT_LENGTH)
    mobility_intent: Literal["preserve", "movable", "static"] = "preserve"
    operations: list[PhysicsTopologyOperation] = Field(max_length=1024)
    invariants: PhysicsTopologyInvariants


class PhysicsApplySchemaRequest(BaseModel):
    """Apply USD physics schemas from accepted physics predictions."""

    model_config = ConfigDict(extra="forbid")

    usd_path: str = Field(max_length=MAX_PATH_FIELD_LENGTH)
    decision_patch_path: str | None = Field(
        default=None,
        max_length=MAX_PATH_FIELD_LENGTH,
    )
    predictions_jsonl_path: str = Field(max_length=MAX_PATH_FIELD_LENGTH)
    output_usd_path: str | None = Field(
        default=None,
        max_length=MAX_PATH_FIELD_LENGTH,
    )
    collision_approximation: str = Field(
        default="convexHull",
        max_length=MAX_SHORT_TEXT_LENGTH,
    )
    output_key: str = Field(default="classification", max_length=MAX_SHORT_TEXT_LENGTH)
    author_rigid_body: bool = True


class PhysicsRuntimeAcceptance(BaseModel):
    """Hard solver-backed acceptance limits for drop validation."""

    model_config = ConfigDict(extra="forbid")

    detect_initial_pose_discontinuity: bool = True
    max_initial_pose_displacement_m: float | None = Field(default=None, gt=0)
    ballistic_displacement_multiplier: float = Field(default=3.0, ge=1.0, le=20.0)
    initial_pose_tolerance_m: float = Field(default=0.002, ge=0, le=0.1)
    max_ground_penetration_m: float = Field(default=0.005, ge=0, le=1.0)
    require_gravity_response: bool = True
    expected_body_count: int | None = Field(default=None, ge=0)


class PhysicsRuntimeValidationRequest(BaseModel):
    """Run physics runtime validation for an authored physics USD."""

    model_config = ConfigDict(extra="forbid")

    physics_usd_path: str = Field(max_length=MAX_PATH_FIELD_LENGTH)
    output_dir: str | None = Field(default=None, max_length=MAX_PATH_FIELD_LENGTH)
    engine: Literal["ovphysx", "fake", "none"] = "ovphysx"
    duration_s: float = Field(default=1.0, gt=0, le=MAX_PHYSICS_RUNTIME_DURATION_S)
    dt: float = Field(
        default=1.0 / 240.0,
        ge=MIN_PHYSICS_RUNTIME_DT_S,
        le=MAX_PHYSICS_RUNTIME_DT_S,
    )
    sample_fps: int = Field(default=30, ge=1, le=MAX_PHYSICS_RUNTIME_SAMPLE_FPS)
    drop_height_m: float | None = Field(default=None, ge=0)
    acceptance: PhysicsRuntimeAcceptance | None = None


class SceneRestoreRequest(BaseModel):
    """Restore current Workbench edits into a durable scene artifact."""

    model_config = ConfigDict(extra="forbid")

    output_usd_path: str | None = Field(default=None, max_length=MAX_PATH_FIELD_LENGTH)
    output_mode: Literal["layer", "composed", "flattened"] = "layer"
    material_profile: str = Field(default="auto", max_length=MAX_SHORT_TEXT_LENGTH)
    skip_instance_check: bool = False
    fail_on_invalid_assignment: bool = True
    overwrite: bool = False
    include_preview_artifact: bool = True


class SceneRestoreResponse(BaseModel):
    """Result of restoring current scene edits through source mapping."""

    session_id: str
    status: str
    source_scene_path: str | None = None
    inspection_scene_path: str | None = None
    preview_scene_path: str | None = None
    output_usd_path: str | None = None
    output_mode: Literal["layer", "composed", "flattened"] | None = None
    restored_edit_count: int = 0
    restored_source_prim_paths: list[str] = Field(default_factory=list)
    unbound_source_prim_paths: list[str] = Field(default_factory=list)
    unresolved_mappings: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    material_apply: MaterialApplyResponse | None = None


class DiagnosticRecord(BaseModel):
    """Scene diagnostic record."""

    type: str
    source: str
    prim_path: str | None = None
    attribute: str | None = None
    layer: str | None = None


class DiagnosticsResponse(BaseModel):
    """Collection of scene inspection diagnostics."""

    session_id: str
    diagnostics: list[DiagnosticRecord]
