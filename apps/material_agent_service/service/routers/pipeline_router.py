# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline API endpoints - Core workflow operations."""

import asyncio
import copy
import json
import logging
import os
import shutil
import stat
import tempfile
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, NamedTuple

import yaml
from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

# Import API defaults (replaces service config defaults)
from material_agent.api.defaults import (
    DEFAULT_CAMERA_DIRECTIONS,
    DEFAULT_CLUSTER_COMPLEXITY_THRESHOLDS,
    DEFAULT_CLUSTER_EMBEDDING_MODEL,
    DEFAULT_CLUSTER_NIM_EMBEDDING_MODEL,
    DEFAULT_USD_PRIM_WARNING_THRESHOLD,
)
from material_agent.config.schema import STEP_ORDER
from material_agent.simready import is_simready_library_id
from sse_starlette import EventSourceResponse
from world_understanding.utils.archive import (
    ArchiveSizeLimitExceeded,
    copy_stream_limited,
)
from world_understanding.utils.artifacts import (
    is_pipeline_temp_path,
    open_confined_directory,
    visible_local_artifact_key,
    write_bytes_to_confined,
)
from world_understanding.utils.credentials import (
    InlineSecretError,
    aensure_no_inline_secrets,
    drop_stale_endpoint_credentials,
    ensure_no_inline_secrets,
    format_env_reference,
    is_local_base_url,
    is_nvidia_provider_base_url,
    is_placeholder_api_key,
    resolve_endpoint_api_key,
)
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)
from world_understanding.utils.held_file_response import HeldFileResponse
from world_understanding.utils.usd.stage import get_stage_info_from_path

from ..artifact_lineage import (
    ARTIFACT_CANONICAL_KEYS,
    ARTIFACT_LINEAGE,
    ARTIFACT_PRODUCER_STEPS,
    artifact_is_valid,
    initial_artifact_validity,
    invalidate_artifacts_for_steps,
)
from ..config import config
from ..coverage import (
    CoveragePolicy,
    normalize_coverage_policy,
    normalize_legacy_completed_coverage,
)
from ..models.requests import RegenerateRequest
from ..models.responses import (
    PipelineError,
    PipelineResults,
    PipelineStatus,
    SessionCreated,
)
from ..runtime import DuplicateJobError, get_event_bus, get_job_registry
from ..runtime.events import ProgressEvent, StepState
from ..session.manager import (
    CANCEL_KEY,
    RegenerationClaim,
    RegenerationClaimConflictError,
    SessionManager,
)
from ..storage.base import METADATA_KEY, JsonPreconditionError
from ..workers.executor import execute_pipeline_async, execute_scene_pipeline_async

logger = logging.getLogger(__name__)

_GENERATED_REFERENCE_STATUS_READY = "ready"
_MAX_MATERIALS_ZIP_ENTRIES = 8192
_MAX_MATERIALS_ZIP_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
_REGENERATION_LEASE_SECONDS = 300.0
_REGENERATION_PREPARATION_HEARTBEAT_SECONDS = 30.0
_CANCELLATION_CAS_ATTEMPTS = 32
_MAX_HISTORICAL_DESCRIPTIONS_BYTES = 1024 * 1024
_HISTORICAL_DESCRIPTIONS_INVALID_DETAIL = (
    "Stored reference descriptions failed validation"
)
_REGENERATION_TERMINAL_FIELDS = (
    "cancelled_at",
    "completed_at",
    "coverage",
    "duration_seconds",
    "error",
    "failed_at",
    "failed_step",
    "partial_results",
    "results",
    "restored_predictions_valid",
    "step_timings",
    "timings",
    "timings_breakdown",
)
_INVALID_DURABLE_REQUEST_DETAIL = "Request content cannot contain inline credentials"
_INVALID_MATERIALS_YAML_DETAIL = "Invalid materials.yaml"
_INVALID_SAVED_MATERIALS_DETAIL = "Invalid saved materials.yaml"

# Create router
router = APIRouter(prefix="/pipeline", tags=["pipeline"])


# Global session manager (initialized by main app)
session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    if session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return session_manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the global session manager instance."""
    global session_manager
    session_manager = manager


class _ModelRouting(NamedTuple):
    vlm_backend: str
    vlm_model: str | None
    vlm_nim_base_url: str | None
    llm_backend: str
    llm_model: str | None
    llm_nim_base_url: str | None
    llm_uses_vlm_sidecar: bool
    vlm_base_url: str | None = None
    vlm_api_key: str | None = None
    vlm_api_key_env: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_api_key_env: str | None = None


class _RegenerationInputBundle(NamedTuple):
    """Non-mutating plan for inputs that must exist before regeneration starts."""

    input_usd_path: Path
    hydration_keys: tuple[str, ...]
    reference_image_paths: tuple[Path, ...]
    reference_pdf_paths: tuple[Path, ...]
    reference_descriptions: tuple[Any, ...]
    custom_materials: tuple[str, list[dict[str, Any]]] | None
    extract_materials_zip: bool
    generated_library_cache_available: bool


class _HistoricalMaterialsPlan(NamedTuple):
    """Validated historical material bytes captured before request mutation."""

    custom_materials: tuple[str, list[dict[str, Any]]]
    snapshot_files: tuple[tuple[str, bytes], ...]
    archive_bytes: bytes | None


def _resolve_pipeline_model_routing(vlm_model: str | None = None) -> _ModelRouting:
    """Resolve VLM/LLM backend routing from request, service config, and env."""
    selected_vlm_model = vlm_model if vlm_model else config.vlm_model

    selected_vlm_backend = config.vlm_backend
    if selected_vlm_model and selected_vlm_model.startswith("nim/"):
        selected_vlm_backend = "nim"
        selected_vlm_model = selected_vlm_model[4:]

    vlm_nim_base_url = os.environ.get("MA_VLM_NIM_BASE_URL")
    if vlm_nim_base_url:
        selected_vlm_backend = "nim"

    llm_nim_base_url = os.environ.get("MA_LLM_NIM_BASE_URL")
    llm_uses_vlm_sidecar = False
    if not llm_nim_base_url:
        llm_nim_base_url = vlm_nim_base_url
        llm_uses_vlm_sidecar = bool(llm_nim_base_url)

    selected_llm_backend = "nim" if llm_nim_base_url else config.llm_backend
    selected_llm_model = (
        selected_vlm_model
        if llm_uses_vlm_sidecar and selected_vlm_model
        else config.llm_model
    )
    carry_vlm_endpoint_config = selected_vlm_backend != "nim"
    carry_llm_endpoint_config = selected_llm_backend != "nim"

    return _ModelRouting(
        vlm_backend=selected_vlm_backend,
        vlm_model=selected_vlm_model,
        vlm_nim_base_url=vlm_nim_base_url,
        llm_backend=selected_llm_backend,
        llm_model=selected_llm_model,
        llm_nim_base_url=llm_nim_base_url,
        llm_uses_vlm_sidecar=llm_uses_vlm_sidecar,
        vlm_base_url=config.vlm_base_url if carry_vlm_endpoint_config else None,
        vlm_api_key=config.vlm_api_key if carry_vlm_endpoint_config else None,
        vlm_api_key_env=config.vlm_api_key_env if carry_vlm_endpoint_config else None,
        llm_base_url=config.llm_base_url if carry_llm_endpoint_config else None,
        llm_api_key=config.llm_api_key if carry_llm_endpoint_config else None,
        llm_api_key_env=config.llm_api_key_env if carry_llm_endpoint_config else None,
    )


def _configure_predict_model_routing(
    pipeline_config: dict,
    routing: _ModelRouting,
) -> None:
    """Apply endpoint-specific VLM/LLM routing to the predict step."""
    if "predict" not in pipeline_config.get("steps", {}):
        return

    vlm_config: dict[str, Any] = {
        "backend": routing.vlm_backend,
        "model": routing.vlm_model,
        "temperature": config.vlm_temperature,
        "max_tokens": config.vlm_max_tokens,
        **config.vlm_backend_options,
    }
    # VLM routing starts from a fresh dict on each call, unlike the LLM config
    # below which merges into a possibly pre-existing step config.

    if routing.vlm_backend == "nim" and routing.vlm_nim_base_url:
        vlm_config["base_url"] = routing.vlm_nim_base_url
    elif routing.vlm_base_url:
        vlm_config["base_url"] = routing.vlm_base_url
    if routing.vlm_api_key_env:
        vlm_config["api_key_env"] = format_env_reference(routing.vlm_api_key_env)
    elif routing.vlm_api_key:
        vlm_config["api_key"] = routing.vlm_api_key

    if routing.vlm_model and "cosmos-reason2" in routing.vlm_model:
        vlm_config.update(
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 16384,
                "reasoning_budget": 16384,
                "chat_template_kwargs": {"enable_thinking": True},
            }
        )
        prep_step = pipeline_config.get("steps", {}).get(
            "build_dataset_prepare_dataset", {}
        )
        if prep_step:
            prompts = prep_step.setdefault("prompts", {})
            from material_agent.tasks.prepare_dataset import (
                _VLM_SYSTEM_PROMPT_TEMPLATE,
            )

            base_prompt = prompts.get("vlm_system", _VLM_SYSTEM_PROMPT_TEMPLATE)
            prompts["vlm_system"] = base_prompt.replace(
                "<reasoning>", "<thinking>"
            ).replace("</reasoning>", "</thinking>")
        logger.info("Using Cosmos Reason 2 via NIM backend: %s", routing.vlm_model)

    predict_config = pipeline_config["steps"]["predict"]
    predict_config["vlm"] = vlm_config

    existing_llm = dict(predict_config.get("llm") or {})
    existing_llm.update(
        {
            "temperature": config.llm_temperature,
            "max_tokens": config.llm_max_tokens,
        }
    )
    if routing.llm_nim_base_url:
        # Switching the LLM section onto a NIM endpoint voids any prior
        # provider key/url left over from the unified config defaults.
        drop_stale_endpoint_credentials(
            existing_llm, preserve_local_nim_placeholder=True
        )
        existing_llm.update(
            {
                "backend": "nim",
                "model": routing.llm_model,
                "base_url": routing.llm_nim_base_url,
            }
        )
        logger.info(
            "Routing LLM through local NIM: %s @ %s",
            routing.llm_model,
            routing.llm_nim_base_url,
        )
    elif routing.llm_base_url or routing.llm_api_key_env or routing.llm_api_key:
        drop_stale_endpoint_credentials(existing_llm)
        existing_llm.update(
            {
                "backend": routing.llm_backend,
                "model": routing.llm_model,
            }
        )
        if routing.llm_base_url:
            existing_llm["base_url"] = routing.llm_base_url
        if routing.llm_api_key_env:
            existing_llm["api_key_env"] = format_env_reference(routing.llm_api_key_env)
        elif routing.llm_api_key:
            existing_llm["api_key"] = routing.llm_api_key
    predict_config["llm"] = existing_llm

    predict_config["report"] = {
        "image_max_size": 256,
        "image_format": "jpeg",
        "image_quality": 75,
    }


def _build_service_vlm_config(routing: _ModelRouting) -> dict[str, Any]:
    """Build VLM config for service-owned pipeline-internal calls."""
    vlm_config: dict[str, Any] = {
        "backend": routing.vlm_backend,
        "model": routing.vlm_model,
        "temperature": config.vlm_temperature,
        "max_tokens": config.vlm_max_tokens,
        **config.vlm_backend_options,
    }
    if routing.vlm_backend == "nim" and routing.vlm_nim_base_url:
        vlm_config["base_url"] = routing.vlm_nim_base_url
    return vlm_config


def _configure_generate_material_library_step(
    pipeline_config: dict[str, Any],
    routing: _ModelRouting,
    *,
    material_generation_guidance: str,
    material_generation_texture_size: int,
) -> None:
    """Configure generated-material-library authoring from service settings."""
    step_config = pipeline_config.setdefault("steps", {}).setdefault(
        "generate_material_library",
        {},
    )
    step_config["enabled"] = True
    vlm_config = _build_service_vlm_config(routing)
    step_config["vlm"] = vlm_config
    step_config["vlm_config"] = dict(vlm_config)
    if material_generation_guidance.strip():
        step_config["material_guidance"] = material_generation_guidance.strip()
    step_config["texture_generation"] = {
        "texture_size": material_generation_texture_size,
        "backend": config.image_gen_backend,
        "model": config.image_gen_model,
        "base_url": config.image_gen_base_url,
        "api_key_env_var": format_env_reference("MA_IMAGE_GEN_API_KEY"),
        "color_correct_albedo": True,
        "albedo_color_correction_strength": 1.0,
    }
    step_config["material_authoring"] = {
        "use_default_prototypes": True,
        "prototype_min_score": 0.55,
    }


def _build_service_llm_config(
    routing: _ModelRouting,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build a service-owned LLM config for pipeline-internal calls."""
    llm_config: dict[str, Any] = {
        "backend": routing.llm_backend,
        "model": routing.llm_model,
        "temperature": config.llm_temperature if temperature is None else temperature,
        "max_tokens": config.llm_max_tokens if max_tokens is None else max_tokens,
        **config.vlm_backend_options,
    }
    if routing.llm_nim_base_url:
        llm_config.update(
            {
                "backend": "nim",
                "model": routing.llm_model,
                "base_url": routing.llm_nim_base_url,
            }
        )
    elif routing.llm_base_url or routing.llm_api_key_env or routing.llm_api_key:
        if routing.llm_base_url:
            llm_config["base_url"] = routing.llm_base_url
        if routing.llm_api_key_env:
            llm_config["api_key_env"] = format_env_reference(routing.llm_api_key_env)
        elif routing.llm_api_key:
            llm_config["api_key"] = routing.llm_api_key
    return llm_config


def _configure_scene_model_routing(
    pipeline_config: dict[str, Any],
    routing: _ModelRouting,
) -> dict[str, Any]:
    """Apply service-owned LLM routing for large-scene analysis."""
    scene_config = pipeline_config.setdefault("scene", {})
    if not isinstance(scene_config, dict):
        scene_config = {}
        pipeline_config["scene"] = scene_config

    analyze_config = scene_config.setdefault("analyze", {})
    if not isinstance(analyze_config, dict):
        analyze_config = {}
        scene_config["analyze"] = analyze_config

    analyze_config["llm"] = _build_service_llm_config(routing)
    return scene_config


def _coerce_positive_int(value: object, fallback: int) -> int:
    try:
        parsed = int(str(value)) if value is not None else fallback
    except (TypeError, ValueError):
        parsed = fallback
    return max(1, parsed)


def _parse_positive_int_form(
    name: str,
    value: int | None,
    fallback: int,
) -> int:
    if value is None:
        return fallback
    if isinstance(value, bool) or value < 1:
        raise HTTPException(status_code=400, detail=f"{name} must be >= 1")
    return value


def _build_cluster_complexity_thresholds(
    *,
    low: float | None,
    medium: float | None,
    high: float | None,
) -> dict[str, list[float]] | None:
    overrides = {"low": low, "medium": medium, "high": high}
    if all(value is None for value in overrides.values()):
        return None

    thresholds = {
        tier: [float(values[0]), float(values[1]), float(values[2])]
        for tier, values in DEFAULT_CLUSTER_COMPLEXITY_THRESHOLDS.items()
    }
    for tier, value in overrides.items():
        if value is None:
            continue
        if value < 0.0 or value > 1.0:
            raise HTTPException(
                status_code=400,
                detail=f"cluster_similarity_threshold_{tier} must be in [0.0, 1.0]",
            )
        thresholds[tier][2] = float(value)
    return thresholds


def _cluster_model_for_backend(backend: str, model: str | None) -> str:
    if model:
        return model
    configured_model = (config.cluster_embedding_model or "").strip()
    if backend == "nim":
        if configured_model and configured_model != DEFAULT_CLUSTER_EMBEDDING_MODEL:
            return configured_model
        return DEFAULT_CLUSTER_NIM_EMBEDDING_MODEL
    return configured_model or DEFAULT_CLUSTER_EMBEDDING_MODEL


def _normalize_optional_url(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _resolve_cluster_embedding_base_url(requested_base_url: str | None) -> str | None:
    """Resolve a trusted cluster embedding endpoint for server-side calls."""
    requested = _normalize_optional_url(requested_base_url)
    configured = _normalize_optional_url(config.cluster_embedding_base_url)
    if requested is None:
        return configured

    if is_nvidia_provider_base_url(requested):
        return requested

    if configured and requested.rstrip("/") == configured.rstrip("/"):
        return configured

    raise HTTPException(
        status_code=400,
        detail=(
            "cluster_embedding_base_url request overrides are restricted. "
            "Use a hosted NVIDIA endpoint or configure MA_CLUSTER_EMBEDDING_BASE_URL "
            "on the service deployment."
        ),
    )


def _inject_cluster_step(
    pipeline_steps: list[str],
    *,
    enable_prim_clustering: bool,
    require_prepare_step: bool = True,
) -> list[str]:
    if not enable_prim_clustering:
        return pipeline_steps
    if "predict" not in pipeline_steps and "benchmark" not in pipeline_steps:
        raise HTTPException(
            status_code=400,
            detail=(
                "enable_prim_clustering=true requires predict or benchmark so "
                "representative predictions can be expanded."
            ),
        )
    if require_prepare_step and "build_dataset_prepare_dataset" not in pipeline_steps:
        raise HTTPException(
            status_code=400,
            detail=(
                "enable_prim_clustering=true requires "
                "build_dataset_prepare_dataset to run before prediction."
            ),
        )
    if "cluster_prims" in pipeline_steps:
        return pipeline_steps
    insert_before = "predict" if "predict" in pipeline_steps else "benchmark"
    insert_at = pipeline_steps.index(insert_before)
    return [
        *pipeline_steps[:insert_at],
        "cluster_prims",
        *pipeline_steps[insert_at:],
    ]


def _inject_restore_usd_step(
    pipeline_steps: list[str],
    *,
    optimize_usd_enabled: bool,
) -> list[str]:
    """Insert restore_usd before apply when optimized predictions will be applied.

    The restore_usd step remaps predictions from optimized topology back to the
    caller's original USD paths. The executor then wires apply to the original
    USD plus those restored predictions.
    """
    if not optimize_usd_enabled:
        return pipeline_steps
    if "apply" not in pipeline_steps or "restore_usd" in pipeline_steps:
        return pipeline_steps

    insert_at = pipeline_steps.index("apply")
    return [
        *pipeline_steps[:insert_at],
        "restore_usd",
        *pipeline_steps[insert_at:],
    ]


def _persisted_flag_enabled(value: object) -> bool:
    """Parse a bool persisted by current or legacy service versions."""
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and _parse_bool_form(value)


async def _read_regeneration_checkpoint(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
) -> dict[str, Any]:
    """Read checkpoint evidence without hydrating or mutating session files."""
    state_path = session_dir / "cache" / ".pipeline_state.json"
    try:
        metadata = await manager.get_session_metadata(session_id)
        state_key = manager.resolve_published_artifact_key(
            metadata or {},
            "cache/.pipeline_state.json",
            legacy_key="cache/.pipeline_state.json",
        )
        data = (
            await manager.read_from_store(session_id, state_key)
            if state_key is not None
            else None
        )
        legacy_local = (
            manager.store.kind == "local"
            and (metadata or {}).get("published_artifacts") is None
        )
        if data is None and legacy_local and state_path.exists():
            data = state_path.read_bytes()
        state = json.loads(data) if data is not None else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


async def _derive_regeneration_artifact_validity(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, bool]:
    """Derive reusable lineage from checkpoint outputs plus artifact evidence.

    This intentionally does not use route-level legacy default-true behavior.
    A legacy artifact is reusable by regeneration only when its producing step
    completed with the expected output and the referenced public artifact still
    exists locally or in the configured store.
    """
    checkpoint = await _read_regeneration_checkpoint(
        manager,
        session_id,
        session_dir,
    )
    completed = checkpoint.get("completed_steps")
    completed_steps = set(completed) if isinstance(completed, list) else set()
    step_outputs = checkpoint.get("step_outputs")
    outputs = step_outputs if isinstance(step_outputs, dict) else {}

    async def key_exists(key: str) -> bool:
        published_key = manager.resolve_published_artifact_key(
            metadata,
            key,
            legacy_key=key,
        )
        legacy_local = (
            manager.store.kind == "local"
            and metadata.get("published_artifacts") is None
        )
        return (legacy_local and (session_dir / key).is_file()) or bool(
            published_key and await manager.exists_in_store(session_id, published_key)
        )

    async def any_key_exists(keys: tuple[str, ...]) -> bool:
        for key in keys:
            if await key_exists(key):
                return True
        return False

    validity = initial_artifact_validity()
    for artifact, rule in ARTIFACT_LINEAGE.items():
        if artifact in {"prediction_report", "previews"}:
            continue
        checkpoint_emitted = any(
            step in completed_steps
            and isinstance(outputs.get(step), dict)
            and any(outputs[step].get(key) for key in output_keys)
            for step, output_keys in rule.emitted_outputs.items()
        )
        artifact_keys = ARTIFACT_CANONICAL_KEYS.get(artifact, ())
        validity[artifact] = checkpoint_emitted and await any_key_exists(artifact_keys)

    report_key = manager.resolve_prediction_report_key(
        metadata,
        legacy_key=ARTIFACT_CANONICAL_KEYS["prediction_report"][0],
    )
    validity["prediction_report"] = validity["raw_predictions"] and bool(
        report_key and await manager.exists_in_store(session_id, report_key)
    )

    preview_names = metadata.get("preview_images")
    preview_keys = (
        tuple(
            key
            for name in preview_names
            if isinstance(name, str) and name
            for key in (f"cache/preview/{name}", f"preview/{name}")
        )
        if isinstance(preview_names, list)
        else ()
    )
    build_outputs = outputs.get("build_dataset_usd")
    preview_checkpoint_emitted = (
        "build_dataset_usd" in completed_steps
        and isinstance(build_outputs, dict)
        and bool(build_outputs.get("num_images"))
    )
    legacy_local = (
        manager.store.kind == "local" and metadata.get("published_artifacts") is None
    )
    local_preview_exists = legacy_local and any(
        (session_dir / "cache" / "preview").glob("*.png")
    )
    if legacy_local and not local_preview_exists:
        local_preview_exists = any((session_dir / "preview").glob("*.png"))
    validity["previews"] = preview_checkpoint_emitted and (
        local_preview_exists
        or bool(preview_keys and await any_key_exists(preview_keys))
    )

    stored_contract = metadata.get("artifact_validity")
    if isinstance(stored_contract, dict):
        for artifact in validity:
            validity[artifact] = validity[artifact] and artifact_is_valid(
                metadata,
                artifact,
            )
    elif "restored_predictions_valid" in metadata:
        validity["restored_predictions"] = validity["restored_predictions"] and bool(
            metadata["restored_predictions_valid"]
        )
    return validity


async def _derive_regeneration_step_evidence(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
    validity: dict[str, bool],
) -> set[str]:
    """Return cached producer steps with checkpoint and file/store evidence."""
    checkpoint = await _read_regeneration_checkpoint(
        manager,
        session_id,
        session_dir,
    )
    completed = checkpoint.get("completed_steps")
    completed_steps = set(completed) if isinstance(completed, list) else set()
    raw_outputs = checkpoint.get("step_outputs")
    outputs = raw_outputs if isinstance(raw_outputs, dict) else {}
    metadata = await manager.get_session_metadata(session_id) or {}

    def emitted(step: str, *keys: str) -> bool:
        step_output = outputs.get(step)
        return (
            step in completed_steps
            and isinstance(step_output, dict)
            and any(step_output.get(key) for key in keys)
        )

    async def key_exists(key: str) -> bool:
        published_key = manager.resolve_published_artifact_key(
            metadata,
            key,
            legacy_key=key,
        )
        legacy_local = (
            manager.store.kind == "local"
            and metadata.get("published_artifacts") is None
        )
        return (legacy_local and (session_dir / key).is_file()) or bool(
            published_key and await manager.exists_in_store(session_id, published_key)
        )

    evidence: set[str] = set()
    build_dir = session_dir / "cache" / "dataset" / "usd"
    legacy_local = (
        manager.store.kind == "local" and metadata.get("published_artifacts") is None
    )
    local_build_output = (
        legacy_local
        and build_dir.exists()
        and any(path.is_file() for path in build_dir.rglob("*"))
    )
    published = metadata.get("published_artifacts")
    published_map = (
        published.get("artifacts", {}) if isinstance(published, dict) else {}
    )
    stored_build_output = bool(
        any(
            isinstance(key, str) and key.startswith("cache/dataset/usd/")
            for key in published_map
        )
        or await manager.store.list_keys(
            session_id,
            prefix="cache/dataset/usd/",
        )
    )
    if emitted("build_dataset_usd", "usd_dataset_dir", "output_dir") and (
        local_build_output or stored_build_output
    ):
        evidence.add("build_dataset_usd")

    if emitted(
        "build_dataset_prepare_dataset",
        "dataset_jsonl_path",
        "dataset_path",
    ) and await key_exists("cache/dataset/dataset.jsonl"):
        evidence.add("build_dataset_prepare_dataset")

    if (
        emitted("cluster_prims", "cluster_map_path")
        and validity["cluster_map"]
        and validity["cluster_representatives"]
    ):
        evidence.add("cluster_prims")

    for step in (
        "predict",
        "benchmark",
        "expand_cluster_predictions",
        "validate_predictions",
        "harmonize_predictions",
    ):
        if emitted(step, "predictions_path") and validity["raw_predictions"]:
            evidence.add(step)
    if (
        emitted("restore_usd", "restored_predictions_path")
        and validity["restored_predictions"]
    ):
        evidence.add("restore_usd")
    if emitted("apply", "output_usd_path") and validity["applied_output_usd"]:
        evidence.add("apply")
    if emitted("render", "flattened_usd_path") and validity["rendered_output_usd"]:
        evidence.add("render")
    return evidence


async def _hydrate_regeneration_inputs(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
    steps_to_run: list[str],
    *,
    optimize_usd_enabled: bool,
    preserved_input_keys: tuple[str, ...] = (),
) -> None:
    """Overwrite local cached inputs with the authoritative shared-store bytes."""
    metadata = await manager.get_session_metadata(session_id) or {}
    publication = metadata.get("published_artifacts")
    published_artifacts = (
        publication.get("artifacts", {}) if isinstance(publication, dict) else {}
    )
    planned = set(steps_to_run)
    prefixes: set[str] = set()
    keys = set(preserved_input_keys)
    if "build_dataset_prepare_dataset" in planned:
        prefixes.add("cache/dataset/usd/")
    if planned & {"cluster_prims", "predict", "benchmark"}:
        prefixes.add("cache/dataset/")
    if "expand_cluster_predictions" in planned:
        prefixes.add("cache/clusters/")
        keys.add("cache/predictions/predictions.jsonl")
    if "restore_usd" in planned:
        keys.add("cache/predictions/predictions.jsonl")
    if "apply" in planned:
        keys.add(
            "cache/restored/restored_predictions.jsonl"
            if optimize_usd_enabled
            else "cache/predictions/predictions.jsonl"
        )
    if "render" in planned:
        keys.add("output/scene_with_materials.usd")

    for prefix in prefixes:
        keys.update(await manager.store.list_keys(session_id, prefix=prefix))
        keys.update(
            logical_key
            for logical_key in published_artifacts
            if isinstance(logical_key, str) and logical_key.startswith(prefix)
        )
    for key in sorted(keys):
        source_key = (
            key
            if key.startswith(("input/", "materials/"))
            else manager.resolve_published_artifact_key(
                metadata,
                key,
                legacy_key=key,
            )
        )
        data = (
            await manager.read_from_store(session_id, source_key)
            if source_key is not None
            else None
        )
        if data is None:
            if manager.store.kind != "local" or publication is not None:
                (session_dir / key).unlink(missing_ok=True)
            continue
        path = session_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        pending_path = path.with_name(f".{path.name}.regeneration-refresh")
        pending_path.write_bytes(data)
        pending_path.replace(path)


def _regeneration_invalidated_steps(steps_to_run: list[str]) -> set[str]:
    """Return the checkpoint suffix invalidated by a regeneration plan."""
    requested_indices = [
        STEP_ORDER.index(step) for step in steps_to_run if step in STEP_ORDER
    ]
    if not requested_indices:
        return set()
    return set(STEP_ORDER[min(requested_indices) :])


def _inject_regeneration_restore_step(
    steps_to_run: list[str],
    *,
    optimize_usd_enabled: bool,
    metadata: dict[str, Any],
) -> list[str]:
    """Refresh restored predictions only when apply cannot reuse valid cache."""
    preliminary_invalidated = _regeneration_invalidated_steps(steps_to_run)
    restore_needs_refresh = (
        "restore_usd" in preliminary_invalidated
        or not artifact_is_valid(metadata, "restored_predictions")
    )
    if not restore_needs_refresh:
        return steps_to_run
    return _inject_restore_usd_step(
        steps_to_run,
        optimize_usd_enabled=optimize_usd_enabled,
    )


def _validate_regeneration_dependency_closure(
    steps_to_run: list[str],
    invalidated_steps: set[str],
    *,
    optimize_usd_enabled: bool,
    metadata: dict[str, Any],
) -> None:
    """Reject plans that would consume invalidated or stale upstream artifacts."""
    planned = set(steps_to_run)
    cached_steps_value = metadata.get("_regeneration_step_evidence")
    cached_steps = (
        set(cached_steps_value)
        if isinstance(cached_steps_value, set | list | tuple | frozenset)
        else set()
    )
    dependencies: dict[str, tuple[str, ...]] = {
        "build_dataset_prepare_dataset": ("build_dataset_usd",),
        "cluster_prims": ("build_dataset_prepare_dataset",),
        "predict": ("build_dataset_prepare_dataset",),
        "benchmark": ("build_dataset_prepare_dataset",),
        "expand_cluster_predictions": ("cluster_prims", "predict"),
        "restore_usd": ("predict",),
        "apply": (("restore_usd",) if optimize_usd_enabled else ("predict",)),
        "render": ("apply",),
    }
    for consumer in steps_to_run:
        for producer in dependencies.get(consumer, ()):
            if producer in invalidated_steps and producer not in planned:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Regeneration step '{consumer}' requires '{producer}', "
                        "which is invalidated by an earlier requested step. "
                        "Include the missing producer in steps."
                    ),
                )
            if producer not in planned and producer not in cached_steps:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Regeneration step '{consumer}' requires reusable "
                        f"checkpoint output from '{producer}', but no current "
                        "cached evidence was found. Include the missing producer "
                        "in steps."
                    ),
                )

    required_artifacts: dict[str, tuple[str, ...]] = {
        "expand_cluster_predictions": ("raw_predictions", "cluster_map"),
        "restore_usd": ("raw_predictions",),
        "apply": (
            "restored_predictions" if optimize_usd_enabled else "raw_predictions",
        ),
        "render": ("applied_output_usd",),
    }
    for consumer in steps_to_run:
        for artifact in required_artifacts.get(consumer, ()):
            produced_in_plan = bool(ARTIFACT_PRODUCER_STEPS[artifact] & planned)
            if not artifact_is_valid(metadata, artifact) and not produced_in_plan:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Regeneration step '{consumer}' requires current "
                        f"'{artifact}' artifacts. Include one of its producer "
                        "steps in the request."
                    ),
                )


def _configure_apply_step(
    pipeline_config: dict,
    *,
    layer_only: bool,
    request_context: str,
) -> None:
    """Apply output-mode options without implicitly enabling the apply step."""
    steps_config = pipeline_config.get("steps", {})
    if "apply" not in steps_config:
        if layer_only:
            raise HTTPException(
                status_code=400,
                detail=f"{request_context}: layer_only=true requires the apply step.",
            )
        return

    steps_config["apply"]["layer_only"] = layer_only
    if layer_only:
        steps_config["apply"]["flatten_output"] = False
        logger.info("%s: layer-only mode enabled", request_context)


def _build_cluster_prims_step_config(
    *,
    cluster_min_prims: int | None,
    cluster_embedding_backend: str | None,
    cluster_embedding_model: str | None,
    cluster_embedding_base_url: str | None,
    cluster_embedding_max_workers: int | None,
    cluster_embedding_batch_size: int | None,
    cluster_max_size: int | None,
    cluster_similarity_threshold_low: float | None,
    cluster_similarity_threshold_medium: float | None,
    cluster_similarity_threshold_high: float | None,
    cluster_report: str,
) -> dict:
    from material_agent.api import build_cluster_prims_config

    backend = (
        (cluster_embedding_backend or config.cluster_embedding_backend).strip().lower()
    )
    if not backend:
        backend = config.cluster_embedding_backend.strip().lower()
    base_url = _resolve_cluster_embedding_base_url(cluster_embedding_base_url)
    report_enabled = cluster_report.lower() == "true"
    cluster_api_key = resolve_endpoint_api_key(
        config.cluster_embedding_api_key,
        config.cluster_embedding_api_key_env,
        prefer_env=True,
    )
    if backend == "nim" and is_nvidia_provider_base_url(base_url):
        hosted_key = cluster_api_key or config.nvidia_api_key
        if not hosted_key or is_placeholder_api_key(hosted_key):
            raise HTTPException(
                status_code=400,
                detail=(
                    "enable_prim_clustering=true with hosted NIM embeddings "
                    "requires NVIDIA_API_KEY or MA_CLUSTER_EMBEDDING_API_KEY."
                ),
            )

    cluster_kwargs: dict[str, Any] = {}
    if config.cluster_embedding_api_key_env:
        cluster_kwargs["api_key_env"] = format_env_reference(
            config.cluster_embedding_api_key_env
        )

    cluster_config = build_cluster_prims_config(
        embedding_service=backend,
        embedding_model=_cluster_model_for_backend(backend, cluster_embedding_model),
        min_prims_to_activate=_parse_positive_int_form(
            "cluster_min_prims",
            cluster_min_prims,
            config.cluster_min_prims,
        ),
        max_workers=_parse_positive_int_form(
            "cluster_embedding_max_workers",
            cluster_embedding_max_workers,
            config.cluster_embedding_max_workers,
        ),
        batch_size=_parse_positive_int_form(
            "cluster_embedding_batch_size",
            cluster_embedding_batch_size,
            config.cluster_embedding_batch_size,
        ),
        max_cluster_size=_parse_positive_int_form(
            "cluster_max_size",
            cluster_max_size,
            config.cluster_max_size,
        ),
        complexity_thresholds=_build_cluster_complexity_thresholds(
            low=cluster_similarity_threshold_low,
            medium=cluster_similarity_threshold_medium,
            high=cluster_similarity_threshold_high,
        ),
        base_url=base_url or None,
        report=report_enabled,
        **cluster_kwargs,
    )
    if backend == "nim":
        if base_url and not is_nvidia_provider_base_url(base_url):
            if cluster_api_key:
                # ClusterPrimsTask resolves this env var at execution time.
                # Do not persist real endpoint credentials into session config
                # or temporary per-step YAML files.
                pass
            elif is_local_base_url(base_url):
                cluster_config["api_key"] = "not-used"
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "enable_prim_clustering=true with a custom NIM "
                        "embedding endpoint requires MA_CLUSTER_EMBEDDING_API_KEY."
                    ),
                )
    return cluster_config


def _cluster_session_config_from_step_config(
    *,
    enabled: bool,
    step_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the sanitized clustering config persisted in session metadata."""

    cluster_keys: dict[str, Any] = {
        "enable_prim_clustering": enabled,
        "cluster_min_prims": None,
        "cluster_embedding_backend": None,
        "cluster_embedding_model": None,
        "cluster_embedding_base_url": None,
        "cluster_embedding_max_workers": None,
        "cluster_embedding_batch_size": None,
        "cluster_max_size": None,
        "cluster_similarity_threshold_low": None,
        "cluster_similarity_threshold_medium": None,
        "cluster_similarity_threshold_high": None,
        "cluster_report": False,
    }
    if not enabled or step_config is None:
        return cluster_keys

    thresholds = step_config.get("complexity_thresholds")

    def _similarity_threshold(tier: str) -> float | None:
        if not isinstance(thresholds, dict):
            return None
        values = thresholds.get(tier)
        if not isinstance(values, list | tuple) or len(values) < 3:
            return None
        return float(values[2])

    report_config = step_config.get("report", {"enabled": True})
    report_enabled = (
        bool(report_config.get("enabled", True))
        if isinstance(report_config, dict)
        else bool(report_config)
    )
    cluster_keys.update(
        {
            "cluster_min_prims": step_config.get("min_prims_to_activate"),
            "cluster_embedding_backend": step_config.get("embedding_service"),
            "cluster_embedding_model": step_config.get("embedding_model"),
            "cluster_embedding_base_url": step_config.get("base_url"),
            "cluster_embedding_max_workers": step_config.get("max_workers"),
            "cluster_embedding_batch_size": step_config.get("batch_size"),
            "cluster_max_size": step_config.get("max_cluster_size"),
            "cluster_similarity_threshold_low": _similarity_threshold("low"),
            "cluster_similarity_threshold_medium": _similarity_threshold("medium"),
            "cluster_similarity_threshold_high": _similarity_threshold("high"),
            "cluster_report": report_enabled,
        }
    )
    return cluster_keys


def _effective_render_worker_limit(value: object, fallback: int) -> int:
    """Apply the service render-worker cap without increasing lower values."""
    return min(_coerce_positive_int(value, fallback), config.max_render_num_workers)


def _effective_render_request_limit(
    value: object,
    fallback: int,
    requested_render_num_workers: int | None,
    render_num_workers: int,
) -> int:
    """Apply explicit request and global caps to async render concurrency."""
    from world_understanding.functions.graphics.render_remote_async import (
        get_global_remote_render_limit,
    )

    limit = _coerce_positive_int(value, fallback)
    if requested_render_num_workers is not None:
        limit = min(limit, render_num_workers)
    global_limit = get_global_remote_render_limit()
    if global_limit is not None:
        limit = min(limit, global_limit)
    return max(1, limit)


def _apply_build_dataset_render_worker_limit(
    pipeline_config: dict,
    requested_render_num_workers: int | None,
) -> None:
    """Set build_dataset_usd worker and request concurrency limits."""
    build_dataset_config = pipeline_config.get("steps", {}).get("build_dataset_usd")
    if not isinstance(build_dataset_config, dict):
        return

    worker_value = (
        requested_render_num_workers
        if requested_render_num_workers is not None
        else build_dataset_config.get("num_workers", config.max_render_num_workers)
    )
    render_num_workers = _effective_render_worker_limit(
        worker_value,
        config.max_render_num_workers,
    )
    render_request_limit = _effective_render_request_limit(
        build_dataset_config.get("max_concurrent_requests", render_num_workers),
        render_num_workers,
        requested_render_num_workers,
        render_num_workers,
    )

    build_dataset_config["num_workers"] = render_num_workers
    build_dataset_config["max_concurrent_requests"] = render_request_limit


def _apply_large_scene_render_batch_limit(pipeline_config: dict) -> None:
    """Cap render batch size for large-scene child asset pipelines."""
    build_dataset_config = pipeline_config.get("steps", {}).get("build_dataset_usd")
    if not isinstance(build_dataset_config, dict):
        return

    current_batch_size = _coerce_positive_int(
        build_dataset_config.get("batch_size"),
        config.scene_render_batch_size,
    )
    build_dataset_config["batch_size"] = min(
        current_batch_size,
        config.scene_render_batch_size,
    )
    if build_dataset_config["batch_size"] != current_batch_size:
        logger.info(
            "Large-scene render batch size capped: %s -> %s",
            current_batch_size,
            build_dataset_config["batch_size"],
        )


def _parse_bool_form(value: str | None) -> bool:
    """Parse a lenient boolean form value."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_coverage_policy(value: object) -> CoveragePolicy:
    """Parse the fail-closed versus inspection-only coverage policy."""
    try:
        return normalize_coverage_policy(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _insert_step_before(
    steps: list[str],
    step_name: str,
    *,
    before_candidates: tuple[str, ...],
) -> list[str]:
    """Insert a pipeline step once before the first matching downstream step."""
    if step_name in steps:
        return steps
    insert_at = next(
        (
            steps.index(candidate)
            for candidate in before_candidates
            if candidate in steps
        ),
        len(steps),
    )
    return [*steps[:insert_at], step_name, *steps[insert_at:]]


def _normalize_user_email(user_email: str | None) -> str:
    """Return request email or the configured telemetry fallback."""
    normalized = (user_email or "").strip()
    if normalized:
        return normalized

    fallback = config.default_user_email.strip()
    return fallback or "anonymous@nvidia.com"


def _parse_csv_form(value: str | None) -> list[str]:
    """Parse a comma-separated form field."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


async def _validate_request_owned_durable_content(content: dict[str, Any]) -> None:
    """Reject request-owned credentials before a durable or diagnostic boundary."""
    try:
        await aensure_no_inline_secrets(
            content,
            context="material service durable request content",
        )
    except InlineSecretError:
        raise HTTPException(
            status_code=400,
            detail=_INVALID_DURABLE_REQUEST_DETAIL,
        ) from None


def _parse_json_list_form(value: str, field_name: str) -> list[Any]:
    """Parse an optional JSON-list form while preserving permissive semantics."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Invalid %s JSON, ignoring", field_name)
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_json_object_form(value: str, field_name: str) -> dict[str, Any] | None:
    """Parse an optional JSON-object form field."""
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a valid JSON object",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a valid JSON object",
        )
    return parsed


def _parse_iso_datetime(value: str) -> datetime:
    """Parse service ISO timestamps with or without timezone suffixes."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _current_step_with_fresh_elapsed(
    current_step: Any,
) -> dict[str, Any] | None:
    """Return current step info with elapsed time computed at read time."""
    if not isinstance(current_step, dict):
        return None

    refreshed = dict(current_step)
    started_at = refreshed.get("started_at")
    if not isinstance(started_at, str):
        return refreshed

    try:
        elapsed_seconds = int(
            (datetime.now(UTC) - _parse_iso_datetime(started_at)).total_seconds()
        )
    except ValueError:
        return refreshed

    refreshed["elapsed_seconds"] = max(0, elapsed_seconds)
    return refreshed


def _terminal_metadata_ready(metadata: dict[str, Any]) -> bool:
    """Return whether terminal session metadata is authoritative and complete."""
    status = metadata.get("status")
    if status == "completed":
        return bool(
            metadata.get("completed_at")
            and "results" in metadata
            and "coverage" in metadata
        )
    if status == "failed":
        return bool(
            metadata.get("failed_at")
            and (metadata.get("failed_step") or metadata.get("error"))
        )
    if status == "cancelled":
        return bool(metadata.get("cancelled_at"))
    return False


def _merge_terminal_status_metadata(
    snapshot: dict[str, Any],
    disk_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Reconcile terminal status fields without discarding richer evidence."""
    metadata = {**snapshot, **disk_metadata}

    completed_by_name: dict[str, dict[str, Any]] = {}
    for step in [
        *(disk_metadata.get("completed_steps") or []),
        *(snapshot.get("completed_steps") or []),
    ]:
        completed_by_name.setdefault(step["name"], step)
    metadata["completed_steps"] = list(completed_by_name.values())

    snapshot_progress = snapshot.get("overall_progress") or {}
    disk_progress = disk_metadata.get("overall_progress") or {}

    def progress_rank(progress: dict[str, Any]) -> tuple[int, int, int]:
        return (
            int(progress.get("percent", 0) or 0),
            int(progress.get("current_step", 0) or 0),
            int(progress.get("total_steps", 0) or 0),
        )

    chosen_progress = (
        disk_progress
        if progress_rank(disk_progress) >= progress_rank(snapshot_progress)
        else snapshot_progress
    )
    metadata["overall_progress"] = dict(chosen_progress)
    if metadata.get("status") == "completed":
        metadata["overall_progress"]["percent"] = 100

    preview_images = list(disk_metadata.get("preview_images") or [])
    for image_name in snapshot.get("preview_images") or []:
        if image_name not in preview_images:
            preview_images.append(image_name)
    metadata["preview_images"] = preview_images

    return metadata


def _effective_scene_predict_workers(
    pipeline_config: dict,
    scene_workers: int,
    requested_vlm_max_workers: int | None,
) -> int | None:
    """Return a per-asset predict worker count for large-scene mode."""
    predict_config = pipeline_config.get("steps", {}).get("predict")
    if not isinstance(predict_config, dict):
        return None

    if requested_vlm_max_workers is not None:
        predict_workers = requested_vlm_max_workers
    else:
        default_workers = _coerce_positive_int(
            predict_config.get("max_workers"),
            config.max_scene_vlm_concurrency,
        )
        predict_workers = min(
            default_workers,
            max(1, config.max_scene_vlm_concurrency // scene_workers),
        )

    total_vlm_workers = scene_workers * predict_workers
    if total_vlm_workers > config.max_scene_vlm_concurrency:
        raise HTTPException(
            status_code=400,
            detail=(
                "large-scene VLM concurrency is too high: "
                f"scene_workers ({scene_workers}) * vlm_max_workers "
                f"({predict_workers}) = {total_vlm_workers}, "
                f"max: {config.max_scene_vlm_concurrency}"
            ),
        )

    predict_config["max_workers"] = predict_workers
    return predict_workers


def _validate_large_scene_stage_file(input_usd_path: Path) -> str:
    """Validate the public large-scene input contract.

    Large-scene mode accepts one composed USD stage rooted by a default prim.
    The uploaded file is the stage entry point, not a list of independent USDs.
    """
    from pxr import Usd

    try:
        stage = Usd.Stage.Open(str(input_usd_path))
    except Exception as exc:
        raise ValueError(
            "large_scene input must be a valid composed USD stage; "
            f"failed to open {input_usd_path.name}: {exc}"
        ) from exc

    if not stage:
        raise ValueError(
            "large_scene input must be a valid composed USD stage; "
            f"failed to open {input_usd_path.name}"
        )

    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise ValueError(
            "large_scene input must be one composed USD stage with a valid "
            "default root prim (defaultPrim metadata). It is not accepted as "
            "a collection of USD files."
        )

    return str(default_prim.GetPath())


async def _ensure_large_scene_stage_file(input_usd_path: Path) -> str:
    """Validate large-scene USD input without blocking the event loop."""
    try:
        return await asyncio.to_thread(_validate_large_scene_stage_file, input_usd_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _get_generated_reference_entry(
    metadata: dict[str, Any] | None, reference_id: str
) -> dict[str, Any] | None:
    if not metadata:
        return None
    for ref in metadata.get("generated_reference_images", []):
        if isinstance(ref, dict) and ref.get("id") == reference_id:
            return ref
    return None


def _session_accepts_generated_reference(metadata: dict | None) -> bool:
    return bool(
        metadata
        and metadata.get("status", "pending") == _GENERATED_REFERENCE_STATUS_READY
    )


async def _ensure_input_render_local(
    manager: SessionManager, session_id: str, session_dir: Path
) -> Path | None:
    """Return the local input preview, hydrating it from the store if needed."""
    input_render_key = "input/input_render.png"
    input_render = session_dir / input_render_key
    if input_render.exists():
        return input_render

    await manager.sync_from_store(session_id, prefix=input_render_key)
    if input_render.exists():
        return input_render

    data = await manager.read_from_store(session_id, input_render_key)
    if data is None:
        return None

    input_render.parent.mkdir(parents=True, exist_ok=True)
    input_render.write_bytes(data)
    return input_render


def _session_files(directory: Path, pattern: str) -> list[Path]:
    """Return sorted regular files from a session artifact directory."""
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob(pattern) if path.is_file())


async def _restore_existing_session_files(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
    relative_dir: str,
    pattern: str,
) -> list[str]:
    """Return existing session files, hydrating them from external store."""
    directory = session_dir / relative_dir
    files = _session_files(directory, pattern)
    if files:
        return [str(path) for path in files]

    pulled = await manager.sync_from_store(session_id, prefix=f"{relative_dir}/")
    if pulled > 0:
        logger.info(
            "Pulled %s %s file(s) from store for session %s",
            pulled,
            relative_dir,
            session_id[:8],
        )

    return [str(path) for path in _session_files(directory, pattern)]


def _load_reference_descriptions(reference_dir: Path) -> list[Any]:
    """Load saved reference descriptions if present."""
    descriptions_path = reference_dir / "descriptions.json"
    if not descriptions_path.exists():
        return []

    try:
        descriptions = json.loads(descriptions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log_durable_failure(
            logger,
            "material_reference_descriptions_load_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        return []

    return descriptions if isinstance(descriptions, list) else []


async def _preflight_historical_reference_descriptions(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
) -> list[Any]:
    """Validate historical descriptions without mutating or hydrating a session."""
    descriptions_key = "input/reference_images/descriptions.json"
    try:
        local_artifact = await manager.open_local_artifact(
            session_id,
            session_dir / descriptions_key,
        )
        if local_artifact is not None:
            try:
                data = await asyncio.to_thread(
                    _read_historical_descriptions_prefix,
                    local_artifact.stream,
                )
            finally:
                local_artifact.stream.close()
        elif await manager.store.exists(session_id, descriptions_key):
            stream = await manager.store.open_read(session_id, descriptions_key)
            try:
                data = await asyncio.to_thread(
                    stream.read,
                    _MAX_HISTORICAL_DESCRIPTIONS_BYTES + 1,
                )
            finally:
                stream.close()
        else:
            return []
    except Exception:
        log_durable_failure(
            logger,
            "material_historical_descriptions_read_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=True,
        )
        raise HTTPException(
            status_code=409,
            detail=_HISTORICAL_DESCRIPTIONS_INVALID_DETAIL,
        ) from None

    if len(data) > _MAX_HISTORICAL_DESCRIPTIONS_BYTES:
        log_durable_failure(
            logger,
            "material_historical_descriptions_size_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=409,
            detail=_HISTORICAL_DESCRIPTIONS_INVALID_DETAIL,
        )
    try:
        descriptions = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        log_durable_failure(
            logger,
            "material_historical_descriptions_parse_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=409,
            detail=_HISTORICAL_DESCRIPTIONS_INVALID_DETAIL,
        ) from None
    if not isinstance(descriptions, list):
        raise HTTPException(
            status_code=409,
            detail=_HISTORICAL_DESCRIPTIONS_INVALID_DETAIL,
        )
    try:
        await aensure_no_inline_secrets(
            descriptions,
            context="historical material reference descriptions",
        )
    except InlineSecretError:
        log_durable_failure(
            logger,
            "material_historical_descriptions_security_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=409,
            detail=_HISTORICAL_DESCRIPTIONS_INVALID_DETAIL,
        ) from None
    return descriptions


def _read_historical_descriptions_prefix(stream: BinaryIO) -> bytes:
    """Read only enough local bytes to enforce the historical artifact bound."""
    return stream.read(_MAX_HISTORICAL_DESCRIPTIONS_BYTES + 1)


def _local_session_keys(session_dir: Path, prefix: str) -> set[str]:
    """Return regular local session files under ``prefix`` as POSIX keys."""
    root = session_dir / prefix
    if not root.exists():
        return set()
    return {
        path.relative_to(session_dir).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


async def _read_regeneration_plan_key(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
    key: str,
    metadata: dict[str, Any] | None = None,
) -> bytes | None:
    """Read authoritative planning bytes without hydrating the local session."""
    source_key = (
        key
        if key.startswith(("input/", "materials/"))
        else manager.resolve_published_artifact_key(
            metadata or {},
            key,
            legacy_key=key,
        )
    )
    data = (
        await manager.read_from_store(session_id, source_key)
        if source_key is not None
        else None
    )
    if data is not None:
        return data
    if manager.store.kind != "local":
        return None
    local_path = session_dir / key
    return local_path.read_bytes() if local_path.is_file() else None


def _plan_material_manifest(
    manifest_bytes: bytes,
    manifest_key: str,
    available_keys: set[str],
    session_dir: Path,
    *,
    default_library_name: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Validate a stored material manifest and resolve its future local path."""
    try:
        manifest = yaml.safe_load(manifest_bytes.decode("utf-8")) or {}
    except (UnicodeDecodeError, yaml.YAMLError):
        log_durable_failure(
            logger,
            "material_manifest_parse_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=400,
            detail=_INVALID_SAVED_MATERIALS_DETAIL,
        ) from None
    try:
        ensure_no_inline_secrets(
            manifest,
            context="stored material manifest",
            path_context=True,
        )
    except InlineSecretError:
        log_durable_failure(
            logger,
            "material_manifest_security_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=400,
            detail=_INVALID_SAVED_MATERIALS_DETAIL,
        ) from None
    if not isinstance(manifest, dict):
        raise HTTPException(
            status_code=400,
            detail="Saved materials.yaml must be a YAML dictionary",
        )
    section = manifest.get("materials", manifest)
    if not isinstance(section, dict):
        raise HTTPException(
            status_code=400,
            detail="Saved materials.yaml has no material dictionary",
        )
    entries = section.get("entries")
    if (
        not isinstance(entries, list)
        or not entries
        or not all(isinstance(entry, dict) for entry in entries)
    ):
        raise HTTPException(
            status_code=400,
            detail="Saved materials.yaml must contain a non-empty list of entries",
        )
    library_value = section.get("library_path") or default_library_name
    if not isinstance(library_value, str) or not library_value:
        raise HTTPException(
            status_code=400,
            detail="Saved materials.yaml must specify a material library path",
        )

    relative_library = PurePosixPath(library_value.replace("\\", "/"))
    if relative_library.is_absolute():
        relative_library = PurePosixPath(relative_library.name)
    if ".." in relative_library.parts:
        raise HTTPException(
            status_code=400,
            detail="Saved material library path escapes its manifest directory",
        )
    manifest_parent = PurePosixPath(manifest_key).parent
    library_key = str(manifest_parent / relative_library)
    if library_key not in available_keys:
        same_name = sorted(
            key
            for key in available_keys
            if PurePosixPath(key).parent == manifest_parent
            and PurePosixPath(key).name == relative_library.name
        )
        if not same_name:
            raise HTTPException(
                status_code=400,
                detail=f"Saved material library is missing: {library_key}",
            )
        library_key = same_name[0]
    return str((session_dir / library_key).resolve()), entries


def _plan_materials_zip(
    zip_bytes: bytes,
    session_dir: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Inspect a stored custom-material archive outside the live session tree."""
    with tempfile.TemporaryDirectory(prefix="material-regen-plan-") as tmp:
        temp_root = Path(tmp)
        zip_path = temp_root / "materials.zip"
        extract_dir = temp_root / "materials"
        zip_path.write_bytes(zip_bytes)
        library_path, entries = _extract_and_validate_materials_zip(
            zip_path,
            extract_dir,
            failure_phase=FailurePhase.PERSISTENCE_VERIFICATION,
        )
        relative_library = (
            Path(library_path).resolve().relative_to(extract_dir.resolve())
        )
    return str((session_dir / "materials" / relative_library).resolve()), entries


async def _read_bounded_historical_material_key(
    manager: SessionManager,
    session_id: str,
    key: str,
    *,
    max_bytes: int,
) -> bytes:
    """Read one historical material file without hydrating or following aliases."""

    try:
        stream = await manager.store.open_read(session_id, key)
        try:
            data = await asyncio.to_thread(stream.read, max_bytes + 1)
        finally:
            stream.close()
    except Exception:
        log_durable_failure(
            logger,
            "historical_material_read_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=400,
            detail=_INVALID_SAVED_MATERIALS_DETAIL,
        ) from None
    if len(data) > max_bytes:
        log_durable_failure(
            logger,
            "historical_material_size_limit_exceeded",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=400,
            detail=_INVALID_SAVED_MATERIALS_DETAIL,
        )
    return data


async def _preflight_historical_session_materials(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
) -> _HistoricalMaterialsPlan | None:
    """Validate and snapshot saved materials before mutating a ready session."""

    try:
        listed_material_keys = await manager.store.list_keys(
            session_id,
            prefix="materials/",
        )
    except Exception:
        log_durable_failure(
            logger,
            "historical_material_listing_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=True,
        )
        raise HTTPException(
            status_code=409,
            detail="Saved materials could not be verified",
        ) from None
    material_keys: set[str] = set()
    for key in listed_material_keys:
        if not isinstance(key, str):
            key_is_canonical = False
        else:
            posix_key = PurePosixPath(key)
            windows_key = PureWindowsPath(key)
            key_is_canonical = (
                bool(key)
                and "\0" not in key
                and "\\" not in key
                and posix_key.as_posix() == key
                and not posix_key.is_absolute()
                and not windows_key.is_absolute()
                and not windows_key.drive
                and len(posix_key.parts) >= 2
                and posix_key.parts[0] == "materials"
                and all(
                    part not in {"", ".", "..", ".pipeline_temp"} and ":" not in part
                    for part in posix_key.parts
                )
                and not is_pipeline_temp_path(key)
            )
        if not key_is_canonical:
            log_durable_failure(
                logger,
                "historical_material_key_invalid",
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=False,
            )
            raise HTTPException(
                status_code=400,
                detail=_INVALID_SAVED_MATERIALS_DETAIL,
            ) from None
        material_keys.add(key)
    if not material_keys:
        return None

    max_snapshot_bytes = config.max_upload_size_mb * 1024 * 1024
    archive_key = "materials/materials.zip"
    if archive_key in material_keys:
        archive_bytes = await _read_bounded_historical_material_key(
            manager,
            session_id,
            archive_key,
            max_bytes=max_snapshot_bytes,
        )
        try:
            custom_materials = await asyncio.to_thread(
                _plan_materials_zip,
                archive_bytes,
                session_dir,
            )
        except HTTPException:
            raise HTTPException(
                status_code=400,
                detail=_INVALID_SAVED_MATERIALS_DETAIL,
            ) from None
        return _HistoricalMaterialsPlan(
            custom_materials=custom_materials,
            snapshot_files=((archive_key, archive_bytes),),
            archive_bytes=archive_bytes,
        )

    manifest_keys = [
        key
        for key in (
            "materials/materials.yaml",
            *sorted(
                candidate
                for candidate in material_keys
                if candidate.startswith("materials/")
                and candidate.endswith("/materials.yaml")
            ),
        )
        if key in material_keys
    ]
    if not manifest_keys:
        return None
    manifest_key = manifest_keys[0]
    manifest_bytes = await _read_bounded_historical_material_key(
        manager,
        session_id,
        manifest_key,
        max_bytes=min(max_snapshot_bytes, 4 * 1024 * 1024),
    )
    try:
        custom_materials = await asyncio.to_thread(
            _plan_material_manifest,
            manifest_bytes,
            manifest_key,
            material_keys,
            session_dir,
        )
    except HTTPException:
        raise HTTPException(
            status_code=400,
            detail=_INVALID_SAVED_MATERIALS_DETAIL,
        ) from None

    snapshots: dict[str, bytes] = {manifest_key: manifest_bytes}
    remaining_bytes = max_snapshot_bytes - len(manifest_bytes)
    for key in sorted(material_keys):
        if key == manifest_key:
            continue
        snapshots[key] = await _read_bounded_historical_material_key(
            manager,
            session_id,
            key,
            max_bytes=max(remaining_bytes, 0),
        )
        remaining_bytes -= len(snapshots[key])
    return _HistoricalMaterialsPlan(
        custom_materials=custom_materials,
        snapshot_files=tuple(snapshots.items()),
        archive_bytes=None,
    )


def _write_historical_material_snapshot(
    session_dir: Path,
    key: str,
    data: bytes,
) -> None:
    with open_confined_directory(session_dir) as session_descriptor:
        write_bytes_to_confined(
            session_descriptor,
            key,
            data,
            overwrite=True,
            file_mode=0o600,
        )


def _materialize_historical_materials_plan(
    plan: _HistoricalMaterialsPlan,
    session_dir: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Publish only the exact material bytes accepted by preflight."""

    for key, data in plan.snapshot_files:
        _write_historical_material_snapshot(session_dir, key, data)
    if plan.archive_bytes is not None:
        materials_dir = session_dir / "materials"
        zip_path = materials_dir / "materials.zip"
        _extract_and_validate_materials_zip(
            zip_path,
            materials_dir,
            failure_phase=FailurePhase.PERSISTENCE_VERIFICATION,
        )
    return plan.custom_materials


async def _plan_regeneration_input_bundle(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
    metadata: dict[str, Any],
) -> _RegenerationInputBundle:
    """Plan the complete reusable input bundle without mutating local state."""
    (
        store_input_keys,
        store_material_keys,
        *store_generated_groups,
    ) = await asyncio.gather(
        manager.store.list_keys(session_id, prefix="input/"),
        manager.store.list_keys(session_id, prefix="materials/"),
        manager.store.list_keys(session_id, prefix="cache/generated_material_library/"),
        manager.store.list_keys(session_id, prefix="generated_material_library/"),
        manager.store.list_keys(
            session_id, prefix="output/generated_material_library/"
        ),
    )
    local_fallback = manager.store.kind == "local"
    input_keys = set(store_input_keys)
    material_keys = set(store_material_keys)
    if local_fallback:
        input_keys.update(_local_session_keys(session_dir, "input"))
        material_keys.update(_local_session_keys(session_dir, "materials"))
    generated_keys = set().union(*map(set, store_generated_groups))
    publication = metadata.get("published_artifacts")
    published_artifacts = (
        publication.get("artifacts", {}) if isinstance(publication, dict) else {}
    )
    generated_keys.update(
        key
        for key in published_artifacts
        if isinstance(key, str)
        and key.startswith(
            (
                "cache/generated_material_library/",
                "generated_material_library/",
                "output/generated_material_library/",
            )
        )
    )
    if local_fallback:
        generated_keys.update(
            _local_session_keys(session_dir, "cache/generated_material_library")
        )
        generated_keys.update(
            _local_session_keys(session_dir, "generated_material_library")
        )
        generated_keys.update(
            _local_session_keys(session_dir, "output/generated_material_library")
        )

    input_usd_key = next(
        (
            f"input/scene{extension}"
            for extension in (".usd", ".usda", ".usdc", ".usdz")
            if f"input/scene{extension}" in input_keys
        ),
        None,
    )
    if input_usd_key is None:
        raise HTTPException(status_code=400, detail="Input USD not found for session")

    reference_image_keys = sorted(
        key
        for key in input_keys
        if PurePosixPath(key).parent == PurePosixPath("input/reference_images")
        and PurePosixPath(key).name.startswith("reference_")
        and PurePosixPath(key).suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    reference_pdf_keys = sorted(
        key
        for key in input_keys
        if PurePosixPath(key).parent == PurePosixPath("input/reference_pdfs")
        and PurePosixPath(key).name.startswith("reference_")
        and PurePosixPath(key).suffix.lower() == ".pdf"
    )
    descriptions_key = "input/reference_images/descriptions.json"
    reference_descriptions: list[Any] = []
    if descriptions_key in input_keys:
        descriptions_data = await _read_regeneration_plan_key(
            manager,
            session_id,
            session_dir,
            descriptions_key,
            metadata,
        )
        try:
            parsed_descriptions = json.loads(descriptions_data or b"[]")
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed_descriptions = []
        if isinstance(parsed_descriptions, list):
            reference_descriptions = parsed_descriptions

    custom_materials: tuple[str, list[dict[str, Any]]] | None = None
    extract_materials_zip = False
    material_manifest_keys = [
        key
        for key in (
            "materials/materials.yaml",
            *sorted(
                candidate
                for candidate in material_keys
                if candidate.startswith("materials/")
                and candidate.endswith("/materials.yaml")
            ),
        )
        if key in material_keys
    ]
    if material_manifest_keys:
        manifest_key = material_manifest_keys[0]
        manifest_data = await _read_regeneration_plan_key(
            manager,
            session_id,
            session_dir,
            manifest_key,
            metadata,
        )
        if manifest_data is not None:
            custom_materials = await asyncio.to_thread(
                _plan_material_manifest,
                manifest_data,
                manifest_key,
                material_keys,
                session_dir,
            )
    elif "materials/materials.zip" in material_keys:
        zip_data = await _read_regeneration_plan_key(
            manager,
            session_id,
            session_dir,
            "materials/materials.zip",
            metadata,
        )
        if zip_data is not None:
            custom_materials = await asyncio.to_thread(
                _plan_materials_zip,
                zip_data,
                session_dir,
            )
            extract_materials_zip = True

    generated_library_cache_available = False
    selected_generated_prefix: str | None = None
    if metadata.get("config", {}).get("enable_material_generation"):
        for manifest_key in (
            "cache/generated_material_library/materials.yaml",
            "generated_material_library/materials.yaml",
            "output/generated_material_library/materials.yaml",
        ):
            if manifest_key not in generated_keys:
                continue
            manifest_data = await _read_regeneration_plan_key(
                manager,
                session_id,
                session_dir,
                manifest_key,
                metadata,
            )
            if manifest_data is None:
                continue
            await asyncio.to_thread(
                _plan_material_manifest,
                manifest_data,
                manifest_key,
                generated_keys,
                session_dir,
                default_library_name="material_library.usda",
            )
            generated_library_cache_available = True
            selected_generated_prefix = f"{PurePosixPath(manifest_key).parent}/"
            break

    hydration_keys = {
        input_usd_key,
        *(
            key
            for key in store_input_keys
            if key in reference_image_keys
            or key in reference_pdf_keys
            or key == descriptions_key
        ),
        *store_material_keys,
    }
    if selected_generated_prefix is not None:
        hydration_keys.update(
            key for key in generated_keys if key.startswith(selected_generated_prefix)
        )

    return _RegenerationInputBundle(
        input_usd_path=session_dir / input_usd_key,
        hydration_keys=tuple(sorted(hydration_keys)),
        reference_image_paths=tuple(session_dir / key for key in reference_image_keys),
        reference_pdf_paths=tuple(session_dir / key for key in reference_pdf_keys),
        reference_descriptions=tuple(reference_descriptions),
        custom_materials=custom_materials,
        extract_materials_zip=extract_materials_zip,
        generated_library_cache_available=generated_library_cache_available,
    )


def _load_cached_generated_material_library(
    session_dir: Path,
) -> dict[str, Any] | None:
    """Load previously generated material-library artifacts for regeneration."""
    candidates = [
        session_dir / "cache" / "generated_material_library" / "materials.yaml",
        session_dir / "generated_material_library" / "materials.yaml",
        session_dir / "output" / "generated_material_library" / "materials.yaml",
    ]
    manifest_path = next((path for path in candidates if path.exists()), None)
    if manifest_path is None:
        return None

    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        log_durable_failure(
            logger,
            "generated_material_manifest_parse_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        return None

    try:
        ensure_no_inline_secrets(
            manifest,
            context="cached generated material manifest",
            path_context=True,
        )
    except InlineSecretError:
        log_durable_failure(
            logger,
            "generated_material_manifest_security_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        return None

    if not isinstance(manifest, dict):
        log_durable_failure(
            logger,
            "generated_material_manifest_shape_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        return None

    materials_section = manifest.get("materials", manifest)
    if not isinstance(materials_section, dict):
        log_durable_failure(
            logger,
            "generated_material_manifest_shape_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        return None

    entries = materials_section.get("entries")
    if not isinstance(entries, list) or not entries:
        log_durable_failure(
            logger,
            "generated_material_manifest_shape_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        return None

    library_path_value = materials_section.get("library_path") or (
        manifest_path.parent / "material_library.usda"
    )
    library_path = Path(str(library_path_value))
    if not library_path.is_absolute():
        library_path = manifest_path.parent / library_path
    library_path = library_path.resolve()

    if not library_path.exists():
        log_durable_failure(
            logger,
            "generated_material_library_missing",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        return None

    generated_materials_data = {
        "library_path": str(library_path),
        "entries": entries,
    }
    return {
        "generated_material_library_path": str(library_path),
        "generated_materials_data": generated_materials_data,
        "generated_material_entries": entries,
    }


def _ensure_cached_generated_material_library_state(
    session_dir: Path,
    *,
    session_id: str,
) -> bool:
    """Seed pipeline state with cached generated materials when artifacts exist."""
    generated_outputs = _load_cached_generated_material_library(session_dir)
    if generated_outputs is None:
        return False

    working_dir = session_dir / "cache"
    state_path = working_dir / ".pipeline_state.json"
    state: dict[str, Any]
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log_durable_failure(
                logger,
                "regeneration_pipeline_state_read_failed",
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=False,
            )
            state = {}
    else:
        state = {}

    if not isinstance(state, dict):
        state = {}
    state.setdefault("session_id", session_id)
    state.setdefault("project_name", session_id)
    state.setdefault("completed_steps", [])
    state.setdefault("failed_steps", [])
    state.setdefault("step_errors", {})
    step_outputs = state.setdefault("step_outputs", {})
    if not isinstance(step_outputs, dict):
        step_outputs = {}
        state["step_outputs"] = step_outputs
    step_outputs["generate_material_library"] = generated_outputs

    completed_steps = state.setdefault("completed_steps", [])
    if (
        isinstance(completed_steps, list)
        and "generate_material_library" not in completed_steps
    ):
        completed_steps.append("generate_material_library")

    working_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    logger.info(
        "Regeneration will reuse cached generated material library: %s",
        generated_outputs["generated_material_library_path"],
    )
    return True


def _invalidate_regeneration_pipeline_state(
    session_dir: Path,
    steps_to_run: list[str],
) -> set[str]:
    """Discard checkpoint evidence at and after the earliest regenerated step."""
    invalidated_steps = _regeneration_invalidated_steps(steps_to_run)
    if not invalidated_steps:
        return set()
    state_path = session_dir / "cache" / ".pipeline_state.json"
    if not state_path.exists():
        return invalidated_steps

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Discarding unreadable pipeline state before regeneration: %s",
            exc,
        )
        state = {}
    if not isinstance(state, dict):
        state = {}

    completed_steps = state.get("completed_steps")
    if not isinstance(completed_steps, list):
        completed_steps = []
    state["completed_steps"] = [
        step for step in completed_steps if step not in invalidated_steps
    ]

    for field in ("step_outputs", "step_errors"):
        values = state.get(field)
        if not isinstance(values, dict):
            values = {}
            state[field] = values
        for step in invalidated_steps:
            values.pop(step, None)

    failed_steps = state.get("failed_steps")
    if not isinstance(failed_steps, list):
        failed_steps = []
    state["failed_steps"] = [
        step for step in failed_steps if step not in invalidated_steps
    ]
    state["current_step"] = None

    state_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = state_path.with_name(".pipeline_state.regeneration.json")
    pending_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    pending_path.replace(state_path)
    return invalidated_steps


# Streaming upload helper to avoid loading large files into memory
def _validate_materials_yaml_content(
    materials_data: object,
    base_dir: Path,
) -> tuple[str, list[dict]]:
    """Validate parsed materials.yaml content and resolve library path.

    This is a shared helper used by both the ZIP extraction flow and the legacy
    YAML-only fallback during regeneration.

    Validates:
    - materials_data is a dict
    - materials_data is either a flat manifest dict or contains a nested
      materials dict
    - library_path is a non-empty string
    - entries is a non-empty list of dicts
    - Resolved library_path stays within base_dir (path traversal protection)
    - Library file exists on disk

    Args:
        materials_data: Parsed YAML data (from yaml.safe_load)
        base_dir: Directory containing materials.yaml (for resolving library_path)

    Returns:
        Tuple of (absolute_library_path, entries_list)

    Raises:
        HTTPException if validation fails
    """
    # Enforce YAML shape - must be a dict with material data at the top level
    # or under the service-style "materials" key.
    if not isinstance(materials_data, dict):
        error_msg = f"materials.yaml must be a YAML dictionary, got {type(materials_data).__name__}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    materials_section = materials_data.get("materials", materials_data)
    if not isinstance(materials_section, dict):
        error_msg = (
            f"materials.yaml must be a material dictionary or have a "
            f"'materials' dictionary at top level. "
            f"Found top-level keys: {list(materials_data.keys())}, "
            f"materials type: {type(materials_section).__name__ if materials_section else 'None'}"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    library_path_relative = materials_section.get("library_path")
    entries = materials_section.get("entries", [])

    logger.info(
        f"Parsed materials.yaml: library_path={library_path_relative}, "
        f"entries_count={len(entries) if entries else 0}"
    )

    # Validate library_path is a non-empty string
    if not library_path_relative or not isinstance(library_path_relative, str):
        error_msg = (
            "materials.yaml must specify library_path as a non-empty string "
            "either at the top level or under materials.library_path. "
            f"Found top-level keys: {list(materials_data.keys())}, "
            f"materials section keys: {list(materials_section.keys())}, "
            f"library_path type: {type(library_path_relative).__name__}"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    if not isinstance(entries, list) or not entries:
        error_msg = (
            "materials.yaml must contain a non-empty list of entries either "
            "at the top level or under materials.entries. "
            f"Found type={type(entries).__name__}, "
            f"len={len(entries) if hasattr(entries, '__len__') else 'n/a'}"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    # Ensure each entry is a mapping (dict-like)
    if not all(isinstance(e, dict) for e in entries):
        types = {type(e).__name__ for e in entries}
        error_msg = (
            "entries must be a list of objects (YAML mappings). "
            f"Got element types: {sorted(types)}"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    # Resolve and validate USD library file exists (relative to base_dir)
    # Validate library_path doesn't escape base_dir (defense in depth)
    library_path = (base_dir / library_path_relative).resolve()
    base_dir_resolved = base_dir.resolve()

    # Ensure resolved path is within base_dir
    try:
        library_path.relative_to(base_dir_resolved)
    except ValueError:
        error_msg = (
            f"library_path escapes base directory: '{library_path_relative}' "
            f"(resolved to: {library_path}, base: {base_dir_resolved})"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    logger.info(f"Looking for USD library at: {library_path}")

    if not library_path.exists():
        # List available USD files for helpful error message
        available_usd = [
            f.name
            for f in base_dir.iterdir()
            if f.is_file() and f.suffix in (".usd", ".usda", ".usdc")
        ]
        available_msg = (
            f" Available: {available_usd}" if available_usd else " No USD files found"
        )
        error_msg = (
            f"USD library file not found: '{library_path_relative}' "
            f"(resolved to: {library_path}).{available_msg} "
            f"Base directory: {base_dir}. "
            f"Ensure library_path in materials.yaml matches the actual file name."
        )
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    logger.info(
        f"Validated materials.yaml: {len(entries)} materials, library: {library_path.name}"
    )

    return str(library_path), entries


async def _stream_copy(
    upload: UploadFile, dest: Path, chunk_size: int = 2 * 1024 * 1024
) -> int:
    """Stream upload file to disk in chunks to avoid memory spikes.

    Args:
        upload: FastAPI UploadFile to stream
        dest: Destination path on disk
        chunk_size: Chunk size in bytes (default 2MB)

    Returns:
        Total bytes written
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    with dest.open("wb") as f:
        while True:
            data = await upload.read(chunk_size)
            if not data:
                break
            f.write(data)
            total_bytes += len(data)

    return total_bytes


def _find_input_usd(session_dir: Path) -> Path | None:
    """Find the input USD file in a session directory.

    Looks for scene.* with any valid USD extension (.usd, .usda, .usdc, .usdz).

    Args:
        session_dir: Session directory path

    Returns:
        Path to the input USD file, or None if not found
    """
    input_dir = session_dir / "input"
    for ext in [".usd", ".usda", ".usdc", ".usdz"]:
        candidate = input_dir / f"scene{ext}"
        if candidate.exists():
            return candidate
    return None


def _safe_zip_member_target(filename: str, extract_dir: Path) -> Path:
    """Return a safe extraction path or raise for traversal attempts."""
    if not filename or "\x00" in filename:
        raise HTTPException(
            status_code=400,
            detail="Materials ZIP contains an invalid member name.",
        )

    posix_path = PurePosixPath(filename)
    windows_path = PureWindowsPath(filename)
    invalid_parts = {"", ".", ".."}
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in invalid_parts for part in posix_path.parts)
        or any(part in invalid_parts for part in windows_path.parts)
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Materials ZIP contains unsafe path: {filename}",
        )

    extract_root = extract_dir.resolve()
    target = (extract_root / posix_path).resolve()
    try:
        target.relative_to(extract_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Materials ZIP contains unsafe path: {filename}",
        ) from exc
    return target


def _safe_extract_materials_zip(
    zf: zipfile.ZipFile,
    extract_dir: Path,
) -> None:
    """Extract a materials ZIP after validating all member paths."""
    entries = zf.infolist()
    if len(entries) > _MAX_MATERIALS_ZIP_ENTRIES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Materials ZIP contains too many entries "
                f"({len(entries)} > {_MAX_MATERIALS_ZIP_ENTRIES})."
            ),
        )

    members = [
        (info, _safe_zip_member_target(info.filename, extract_dir)) for info in entries
    ]

    total_written = 0
    extracted_files: list[Path] = []
    try:
        for info, target in members:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            extracted_files.append(target)
            with zf.open(info) as source, target.open("wb") as dest:
                try:
                    total_written += copy_stream_limited(
                        source,
                        dest,
                        max_bytes=(
                            _MAX_MATERIALS_ZIP_UNCOMPRESSED_BYTES - total_written
                        ),
                    )
                except ArchiveSizeLimitExceeded as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Materials ZIP uncompressed contents exceed "
                            f"{_MAX_MATERIALS_ZIP_UNCOMPRESSED_BYTES} bytes."
                        ),
                    ) from exc
    except Exception:
        for path in reversed(extracted_files):
            path.unlink(missing_ok=True)
            parent = path.parent
            while parent != extract_dir and parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        raise


def _clean_materials_extract_dir(extract_dir: Path, preserve_path: Path) -> None:
    """Remove stale extracted materials while preserving the uploaded ZIP."""
    preserve_name: str | None = None
    try:
        preserve_relative = preserve_path.absolute().relative_to(extract_dir.absolute())
    except ValueError:
        pass
    else:
        if len(preserve_relative.parts) == 1:
            preserve_name = preserve_relative.name

    with open_confined_directory(extract_dir, create=True) as extract_descriptor:
        with os.scandir(extract_descriptor) as entries:
            child_names = sorted(entry.name for entry in entries)
        for child_name in child_names:
            if preserve_name is not None and child_name == preserve_name:
                continue
            metadata = os.stat(
                child_name,
                dir_fd=extract_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                shutil.rmtree(child_name, dir_fd=extract_descriptor)
            else:
                os.unlink(child_name, dir_fd=extract_descriptor)
        os.fsync(extract_descriptor)


def _discard_rejected_materials_archive(
    materials_dir: Path,
    zip_path: Path,
) -> None:
    """Remove a rejected upload without exposing cleanup backend details."""
    try:
        _clean_materials_extract_dir(materials_dir, zip_path)
        zip_path.unlink(missing_ok=True)
    except OSError:
        log_durable_failure(
            logger,
            "material_archive_rejection_cleanup_failed",
            phase=FailurePhase.ROLLBACK,
            retryable=True,
        )


def _extract_and_validate_materials_zip(
    zip_path: Path,
    extract_dir: Path,
    *,
    failure_phase: FailurePhase = FailurePhase.LOCAL_PUBLICATION,
) -> tuple[str, list[dict]]:
    """Extract materials zip and validate contents.

    The zip must contain:
    - materials.yaml: Material definitions in service format (materials.entries)
    - USD library file: Referenced by library_path in materials.yaml

    Icons (thumbs/) are optional.

    Expected zip structure (created via `zip -r my.zip custom_materials/`):
        my.zip
        └── custom_materials/
            ├── materials.yaml
            └── materials_libs.usda

    Also supports flat structure (materials.yaml at zip root).

    Args:
        zip_path: Path to the uploaded zip file
        extract_dir: Directory to extract contents to

    Returns:
        Tuple of (materials_library_path, materials_entries)

    Raises:
        HTTPException if validation fails
    """
    _clean_materials_extract_dir(extract_dir, zip_path)

    # Extract zip
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_materials_zip(zf, extract_dir)
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=400,
            detail="Invalid ZIP file. Please upload a valid ZIP archive.",
        )

    # Check materials.yaml exists - first at root, then in subdirectory
    materials_yaml_path = extract_dir / "materials.yaml"
    base_dir = extract_dir

    if not materials_yaml_path.exists():
        # Look for materials.yaml in a subdirectory (e.g., zip -r x.zip custom_materials/)
        subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        found = False
        for subdir in subdirs:
            candidate = subdir / "materials.yaml"
            if candidate.exists():
                materials_yaml_path = candidate
                base_dir = subdir
                found = True
                logger.info(f"Found materials.yaml in subdirectory: {subdir.name}/")
                break

        if not found:
            error_msg = (
                f"materials.zip must contain materials.yaml (at root or in a subdirectory). "
                f"Searched in: {extract_dir} and subdirectories: {[d.name for d in subdirs]}"
            )
            logger.error(error_msg)
            raise HTTPException(
                status_code=400,
                detail=error_msg,
            )

    # Parse materials.yaml
    try:
        with open(materials_yaml_path, encoding="utf-8") as f:
            materials_data = yaml.safe_load(f)
    except (OSError, UnicodeError, yaml.YAMLError):
        log_durable_failure(
            logger,
            "material_manifest_parse_failed",
            phase=failure_phase,
            retryable=False,
        )
        raise HTTPException(
            status_code=400,
            detail=_INVALID_MATERIALS_YAML_DETAIL,
        ) from None

    try:
        ensure_no_inline_secrets(
            materials_data,
            context="custom material archive manifest",
            path_context=True,
        )
    except InlineSecretError:
        log_durable_failure(
            logger,
            "material_manifest_security_failed",
            phase=failure_phase,
            retryable=False,
        )
        raise HTTPException(
            status_code=400,
            detail=_INVALID_MATERIALS_YAML_DETAIL,
        ) from None

    # Validate YAML content and resolve library path using shared helper
    library_path, entries = _validate_materials_yaml_content(materials_data, base_dir)

    logger.info(
        f"Validated materials zip: {len(entries)} materials, "
        f"library: {Path(library_path).name}"
    )

    return library_path, entries


def _load_extracted_materials_tree(
    materials_dir: Path,
) -> tuple[str, list[dict]] | None:
    """Read an already-extracted material tree without mutating the session."""
    if materials_dir.is_symlink():
        log_durable_failure(
            logger,
            "material_manifest_security_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=400,
            detail=_INVALID_SAVED_MATERIALS_DETAIL,
        )
    candidates = [
        materials_dir / "materials.yaml",
        *sorted(materials_dir.glob("*/materials.yaml")),
    ]
    materials_yaml_path = next((path for path in candidates if path.exists()), None)
    if materials_yaml_path is None:
        return None
    if materials_yaml_path.is_symlink() or materials_yaml_path.parent.is_symlink():
        log_durable_failure(
            logger,
            "material_manifest_security_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=400,
            detail=_INVALID_SAVED_MATERIALS_DETAIL,
        )
    try:
        materials_data = yaml.safe_load(materials_yaml_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        log_durable_failure(
            logger,
            "material_manifest_parse_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=400,
            detail=_INVALID_SAVED_MATERIALS_DETAIL,
        ) from None
    try:
        ensure_no_inline_secrets(
            materials_data,
            context="stored extracted material manifest",
            path_context=True,
        )
    except InlineSecretError:
        log_durable_failure(
            logger,
            "material_manifest_security_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=400,
            detail=_INVALID_SAVED_MATERIALS_DETAIL,
        ) from None
    try:
        return _validate_materials_yaml_content(
            materials_data,
            materials_yaml_path.parent,
        )
    except HTTPException:
        log_durable_failure(
            logger,
            "material_manifest_validation_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=400,
            detail=_INVALID_SAVED_MATERIALS_DETAIL,
        ) from None


async def _restore_existing_session_materials(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
) -> tuple[str, list[dict]] | None:
    """Load previously uploaded custom materials from a session."""
    materials_dir = session_dir / "materials"
    materials_zip_path = materials_dir / "materials.zip"
    materials_yaml_path = materials_dir / "materials.yaml"

    if not materials_zip_path.exists() and not materials_yaml_path.exists():
        pulled = await manager.sync_from_store(session_id, prefix="materials/")
        if pulled > 0:
            logger.info(
                "Pulled %s material file(s) from store for session %s",
                pulled,
                session_id[:8],
            )

    if materials_zip_path.exists():
        logger.info("Reusing custom materials zip from session %s", session_id[:8])
        return await asyncio.to_thread(
            _extract_and_validate_materials_zip,
            materials_zip_path,
            materials_dir,
            failure_phase=FailurePhase.PERSISTENCE_VERIFICATION,
        )

    if materials_yaml_path.exists():
        logger.info("Reusing custom materials YAML from session %s", session_id[:8])
        return await asyncio.to_thread(_load_extracted_materials_tree, materials_dir)

    return None


async def _render_input_preview(
    session_id: str,
    session_dir: Path,
    original_usd_path: Path | None = None,
) -> None:
    """Render preview of input USD (before material assignment).

    This runs in the background after upload to show users what their scene looks like.
    Creates a single rendered view stored as input/input_render.png.

    Uses the shared ``RenderScenePreviewTask`` via the material agent's
    ``create_render_preview_workflow_from_config`` factory.

    Args:
        session_id: Session identifier
        session_dir: Session directory
        original_usd_path: Original file path on disk (desktop mode). When
            provided, the renderer opens from the original location so that
            relative payload/sublayer references resolve correctly.
    """

    manager = get_session_manager()
    await manager.update_session(
        session_id,
        {"preview_render_status": "rendering", "preview_render_error": None},
    )

    try:
        logger.info(
            f"Rendering input preview for {session_id[:8]}... "
            f"(original_usd_path={original_usd_path})"
        )

        # Find input USD file (supports .usd, .usda, .usdc, .usdz)
        input_usd = _find_input_usd(session_dir)
        if not input_usd:
            message = f"No input USD found for session {session_id[:8]}"
            logger.warning(message)
            await manager.update_session(
                session_id,
                {"preview_render_status": "failed", "preview_render_error": message},
            )
            return
        output_path = session_dir / "input" / "input_render.png"

        # For desktop mode: use the original file path so that relative
        # payload/sublayer references (e.g. @./Payload/Contents.usda@)
        # resolve against the original directory on disk.
        if original_usd_path and original_usd_path.is_file():
            input_usd = original_usd_path
            logger.info(f"Using original USD path for render: {original_usd_path}")
        logger.info(f"Resolved input_usd for render: {input_usd}")

        # Create config for the render_preview workflow
        preview_config = {
            "usd_path": str(input_usd),
            "output_dir": str(session_dir / "input"),
            "backend": "remote",
            "image_width": 512,
            "image_height": 512,
            "cameras": ["+x+y+z"],
            "camera_margin": 1.0,
            "background_color": [1.0, 1.0, 1.0],
            "should_reset_materials": False,
            "use_lights": True,
            "flatten_before_render": False,
        }

        # Import and run render-preview workflow
        from material_agent.workflows import create_render_preview_workflow_from_config

        workflow = create_render_preview_workflow_from_config()

        # Run in thread pool (sync workflow)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            workflow.run,
            {"config_dict": preview_config},
        )

        # Get rendered image from result
        rendered_images = result.get("rendered_preview_paths", [])
        if rendered_images and Path(rendered_images[0]).exists():
            # Rename to standard name
            import shutil

            shutil.move(rendered_images[0], output_path)
            logger.info(f"✓ Input preview rendered: {output_path.name}")
            await manager.update_session(
                session_id,
                {"preview_render_status": "ready", "preview_render_error": None},
            )
            try:
                await manager.put_file_to_store(
                    session_id,
                    "input/input_render.png",
                    str(output_path),
                    content_type="image/png",
                )
            except Exception:
                log_durable_failure(
                    logger,
                    "pipeline_input_render_sync_failed",
                    phase=FailurePhase.SYNC_UPLOAD,
                    retryable=True,
                )
        else:
            message = "Input preview render failed - no output generated"
            logger.warning(message)
            await manager.update_session(
                session_id,
                {"preview_render_status": "failed", "preview_render_error": message},
            )

    except Exception:
        message = "Input preview render failed"
        log_durable_failure(
            logger,
            "pipeline_input_preview_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=True,
        )
        await manager.update_session(
            session_id,
            {"preview_render_status": "failed", "preview_render_error": message},
        )
        # Don't fail the pipeline - this is just a nice-to-have preview
    finally:
        # Remove markers left by older versions. Current progress is carried by
        # preview_render_status and the workflow config remains in memory.
        temp_marker = session_dir / ".input_render_config.yaml"
        temp_marker.unlink(missing_ok=True)


@router.post("/{session_id}/generate-reference-image")
async def generate_reference_image(
    session_id: str,
    prompt: str = Form(..., description="Text prompt describing the desired look"),
) -> dict:
    """Generate a photorealistic reference image from the input preview + prompt.

    This endpoint is called interactively after the preview render is ready.
    The user provides a text prompt describing desired materials/look, and
    the system generates a reference image using an image-generation model.

    The generated image is saved to the session and returned as an explicit
    reference_id. The full pipeline uses it only when that ID is submitted.

    Args:
        session_id: Session identifier (from upload-usd)
        prompt: Text description of desired look

    Returns:
        JSON with status and image URL
    """

    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = await manager.get_session_metadata(session_id)
    if not _session_accepts_generated_reference(metadata):
        raise HTTPException(
            status_code=409,
            detail="Generated references can only be created before the pipeline is queued.",
        )

    if not config.image_gen_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Image generation backend '{config.image_gen_backend}' is not "
                "configured. Check MA_IMAGE_GEN_* and the required API key."
            ),
        )

    session_dir = manager.get_session_dir(session_id)

    # Check that the preview render exists, hydrating local cache if needed.
    input_render = await _ensure_input_render_local(manager, session_id, session_dir)
    if input_render is None:
        raise HTTPException(
            status_code=400,
            detail="Input preview not yet available. Wait for preview rendering to complete.",
        )

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    reference_id = uuid.uuid4().hex
    output_dir = session_dir / "input" / "generated_references" / reference_id
    output_key = f"input/generated_references/{reference_id}/generated_ref_0.png"

    try:
        logger.info(
            f"Generating reference image for {session_id[:8]}: {prompt[:80]}..."
        )

        image_gen_config: dict[str, str] = {"backend": config.image_gen_backend}
        if config.image_gen_model:
            image_gen_config["model"] = config.image_gen_model
        if config.image_gen_base_url:
            image_gen_config["base_url"] = config.image_gen_base_url
        if config.image_gen_api_key_env:
            image_gen_config["api_key_env"] = config.image_gen_api_key_env
        elif config.image_gen_api_key:
            image_gen_config["api_key"] = config.image_gen_api_key

        # Build config for the generate_reference_image workflow
        gen_ref_config = {
            "rendered_preview_paths": [str(input_render)],
            "image_gen": image_gen_config,
            "prompt": prompt.strip(),
            "output_dir": str(output_dir),
            "num_images": 1,
        }

        # Keep service credentials in memory. Session directories are synced to
        # the artifact store, and even process-local temp files create an
        # unnecessary credential persistence boundary.
        from material_agent.workflows import (
            create_generate_reference_image_workflow_from_config,
        )

        workflow = create_generate_reference_image_workflow_from_config()

        # Run in thread pool (sync workflow, may take ~20-30s)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            workflow.run,
            {"config_dict": gen_ref_config},
        )

        # Check result
        generated_paths = result.get("generated_reference_image_paths", [])
        if generated_paths and Path(generated_paths[0]).exists():
            latest_metadata = await manager.get_session_metadata(session_id)
            if not _session_accepts_generated_reference(latest_metadata):
                shutil.rmtree(output_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Generated reference was discarded because the pipeline "
                        "has already been queued."
                    ),
                )

            logger.info(f"✓ Reference image generated for {session_id[:8]}")

            # Mirror before the metadata CAS so another pod can serve the
            # reference immediately after publication.  If the CAS loses to a
            # pipeline start/regeneration claim, remove this orphan below.
            mirrored_to_store = False
            try:
                await manager.put_file_to_store(
                    session_id,
                    output_key,
                    generated_paths[0],
                    content_type="image/png",
                )
                mirrored_to_store = True
            except Exception:
                log_durable_failure(
                    logger,
                    "pipeline_reference_image_sync_failed",
                    phase=FailurePhase.SYNC_UPLOAD,
                    retryable=True,
                )

            image_url = f"/assets/{session_id}/generated-ref/{reference_id}"
            try:
                added = await manager.add_generated_reference_image(
                    session_id,
                    {
                        "id": reference_id,
                        "key": output_key,
                        "path": generated_paths[0],
                        "prompt": prompt.strip(),
                        "image_url": image_url,
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                )
            except RegenerationClaimConflictError as exc:
                if mirrored_to_store:
                    await manager.store.delete_file(session_id, output_key)
                shutil.rmtree(output_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Generated reference was discarded because the pipeline "
                        "has already been queued."
                    ),
                ) from exc
            if not added:
                if mirrored_to_store:
                    await manager.store.delete_file(session_id, output_key)
                shutil.rmtree(output_dir, ignore_errors=True)
                raise HTTPException(status_code=404, detail="Session not found")

            return {
                "status": "ok",
                "reference_id": reference_id,
                "image_url": image_url,
            }
        else:
            raise RuntimeError("No image generated")

    except HTTPException:
        raise
    except Exception:
        log_durable_failure(
            logger,
            "pipeline_reference_image_generation_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate reference image. Check server logs for details.",
        ) from None


@router.delete("/{session_id}/generated-reference-image/{reference_id}")
async def delete_generated_reference_image(session_id: str, reference_id: str) -> dict:
    """Delete a generated-reference image from the session metadata."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = await manager.get_session_metadata(session_id)
    if not _session_accepts_generated_reference(metadata):
        raise HTTPException(
            status_code=409,
            detail="Generated references can only be deleted before the pipeline is queued.",
        )

    try:
        removed = await manager.remove_generated_reference_image(
            session_id, reference_id
        )
    except RegenerationClaimConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail="Generated references can only be deleted before the pipeline is queued.",
        ) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="Generated reference not found")

    key = removed.get("key")
    if isinstance(key, str):
        try:
            await manager.store.delete_file(session_id, key)
        except Exception:
            log_durable_failure(
                logger,
                "pipeline_reference_delete_failed",
                phase=FailurePhase.ROLLBACK,
                retryable=True,
            )
        local_path = manager.get_session_dir(session_id) / key
        local_path.unlink(missing_ok=True)
        parent = local_path.parent
        try:
            parent.rmdir()
        except OSError:
            pass

    return {"status": "deleted", "reference_id": reference_id}


@router.post("/upload-usd", response_model=SessionCreated, status_code=201)
async def upload_usd_immediate(
    usd_file: UploadFile = File(..., description="USD file to upload and preview"),
) -> SessionCreated:
    """Upload USD file immediately and trigger input preview render.

    This endpoint is called immediately when user selects a file (before pipeline configuration).
    It creates a session, saves the USD, and triggers a background preview render.

    Args:
        usd_file: USD file to upload

    Returns:
        Session creation response with session_id
    """
    manager = get_session_manager()

    # Generate unique session ID
    session_id = str(uuid.uuid4())

    # Validate file extension
    if usd_file.filename:
        ext = Path(usd_file.filename).suffix.lower()
        if ext not in config.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {ext}. Allowed: {config.allowed_extensions}",
            )

    # Create session directory structure
    session_dir = await manager.create_session(
        session_id,
        config={"status": "uploading", "filename": usd_file.filename},
    )

    # Save uploaded USD file using streaming, preserving original extension
    original_ext = (
        Path(usd_file.filename).suffix.lower() if usd_file.filename else ".usd"
    )
    usd_path = session_dir / "input" / f"scene{original_ext}"
    failure_phase = FailurePhase.LOCAL_PUBLICATION
    try:
        total_bytes = await _stream_copy(usd_file, usd_path)
        size_mb = total_bytes / (1024 * 1024)

        if size_mb > config.max_upload_size_mb:
            usd_path.unlink(missing_ok=True)
            await manager.delete_session(session_id)
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
            )

        logger.info(
            f"USD uploaded for session {session_id[:8]}: {size_mb:.2f}MB ({original_ext})"
        )

        # Store asset metadata in session for telemetry
        original_filename = usd_file.filename or f"scene{original_ext}"
        failure_phase = FailurePhase.PERSISTENCE_VERIFICATION
        await manager.update_session(
            session_id,
            {
                "asset": {
                    "filename": original_filename,
                    "file_size_bytes": total_bytes,
                    "file_extension": original_ext,
                }
            },
        )

        # Mirror uploaded USD to external store if configured
        try:
            await manager.put_file_to_store(
                session_id,
                f"input/scene{original_ext}",
                str(usd_path),
                content_type="application/octet-stream",
            )
        except Exception:
            log_durable_failure(
                logger,
                "pipeline_usd_sync_failed",
                phase=FailurePhase.SYNC_UPLOAD,
                retryable=True,
            )

        # Trigger background input preview render IMMEDIATELY
        failure_phase = FailurePhase.PERSISTENCE_VERIFICATION
        await manager.update_session(
            session_id,
            {"status": "ready", "preview_render_status": "rendering"},
        )
        failure_phase = FailurePhase.PIPELINE_EXECUTION
        asyncio.create_task(_render_input_preview(session_id, session_dir))
        logger.info(f"✓ Input preview render triggered for {session_id[:8]}...")

        return SessionCreated(
            session_id=session_id,
            status="ready",
            message="USD uploaded, preview rendering in background",
            estimated_duration_minutes=0,
        )

    except HTTPException:
        raise
    except Exception:
        log_durable_failure(
            logger,
            "pipeline_usd_ingest_failed",
            phase=failure_phase,
            retryable=True,
        )
        await manager.delete_session(session_id)
        raise HTTPException(status_code=500, detail="Failed to upload USD") from None


@router.post("/open-usd", response_model=SessionCreated, status_code=201)
async def open_usd_local(
    file_path: str = Body(
        ..., embed=True, description="Absolute path to a local USD file"
    ),
) -> SessionCreated:
    """Open a local USD file by path (desktop mode).

    Instead of uploading bytes, the server reads the file directly from the
    local filesystem.  Validation, session creation, and preview rendering
    match the ``upload-usd`` endpoint exactly.

    Args:
        file_path: Absolute path to a USD file on the local machine.

    Returns:
        Session creation response with session_id.
    """
    manager = get_session_manager()

    src = Path(file_path)

    # --- validate -----------------------------------------------------------
    if not src.is_absolute():
        raise HTTPException(
            status_code=400, detail="file_path must be an absolute path"
        )

    if not src.is_file():
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

    ext = src.suffix.lower()
    if ext not in config.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {ext}. Allowed: {config.allowed_extensions}",
        )

    size_bytes = src.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > config.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
        )

    # --- session setup -------------------------------------------------------
    session_id = str(uuid.uuid4())
    session_dir = await manager.create_session(
        session_id,
        config={"status": "uploading", "filename": src.name},
    )

    dest = session_dir / "input" / f"scene{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Copy the entire source directory so that relative payload/sublayer
    # references (e.g. @./Payload/Contents.usda@) are available for the
    # full pipeline (optimize_usd, build_dataset, etc.).
    max_dir_bytes = config.max_upload_size_mb * 1024 * 1024 * 5
    total_dir_size = sum(f.stat().st_size for f in src.parent.rglob("*") if f.is_file())
    if total_dir_size > max_dir_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Source directory too large: {total_dir_size / (1024 * 1024):.1f}MB. "
                f"Max: {max_dir_bytes / (1024 * 1024):.0f}MB"
            ),
        )
    shutil.copytree(str(src.parent), str(dest.parent), dirs_exist_ok=True)

    # Rename the main USD to the canonical scene{ext} name that the
    # rest of the pipeline expects (_find_input_usd looks for "scene.*").
    copied_src = dest.parent / src.name
    if copied_src.resolve() != dest.resolve() and copied_src.exists():
        copied_src.rename(dest)

    logger.info(f"USD opened for session {session_id[:8]}: {size_mb:.2f}MB ({ext})")

    # Store asset metadata (mirrors upload-usd)
    await manager.update_session(
        session_id,
        {
            "asset": {
                "filename": src.name,
                "file_size_bytes": size_bytes,
                "file_extension": ext,
            }
        },
    )

    # Mirror to external store if configured
    try:
        await manager.put_file_to_store(
            session_id,
            f"input/scene{ext}",
            str(dest),
            content_type="application/octet-stream",
        )
    except Exception:
        log_durable_failure(
            logger,
            "pipeline_usd_sync_failed",
            phase=FailurePhase.SYNC_UPLOAD,
            retryable=True,
        )

    # Trigger background input preview render — pass the original path so
    # that payload/sublayer references resolve against the source directory.
    await manager.update_session(
        session_id,
        {"status": "ready", "preview_render_status": "rendering"},
    )
    asyncio.create_task(
        _render_input_preview(session_id, session_dir, original_usd_path=src)
    )
    logger.info(f"Input preview render triggered for {session_id[:8]}...")

    return SessionCreated(
        session_id=session_id,
        status="ready",
        message="USD opened, preview rendering in background",
        estimated_duration_minutes=0,
    )


@router.post("", response_model=SessionCreated, status_code=202)
async def create_pipeline(
    usd_file: UploadFile = File(
        None, description="USD file to process (optional if ``session_id`` provided)"
    ),
    session_id: str = Form(
        None, description="Existing session ID (from ``/upload-usd`` endpoint)"
    ),
    user_email: str | None = Form(
        default=None,
        description=(
            "Optional user email address for usage tracking and telemetry. "
            "Omitted or blank values use MA_DEFAULT_USER_EMAIL, then "
            "anonymous@nvidia.com when that fallback is blank."
        ),
    ),
    reference_images: list[UploadFile] = File(
        default=[],
        description="Reference images to help VLM understand the object (optional)",
    ),
    reference_pdfs: list[UploadFile] = File(
        default=[],
        description=(
            "Reference PDFs retained as provenance outside visual-model inputs; "
            "this endpoint does not extract material claims from uploads (optional)"
        ),
    ),
    materials_zip: UploadFile | None = File(
        None,
        description="ZIP file containing custom materials (materials.yaml + USD library)",
    ),
    reference_descriptions: str = Form(
        default="",
        description='JSON array of descriptions for each reference image (e.g., \'["view 1", "view 2"]\') (optional)',
    ),
    generated_reference_id: str = Form(
        default="",
        description="Generated reference ID returned by generate-reference-image (optional)",
    ),
    enable_material_generation: str = Form(
        default="false",
        description=(
            "Generate an asset-specific material library before prediction "
            "(true/false, default: false)"
        ),
    ),
    material_generation_guidance: str = Form(
        default="",
        description=(
            "Optional guidance for generated material planning, such as required "
            "colors, finishes, or reference-image priorities."
        ),
    ),
    material_generation_texture_size: int = Form(
        default=1024,
        ge=64,
        le=4096,
        description="Texture map size for generated materials (default: 1024)",
    ),
    user_prompt: str = Form(
        default="",
        description="Custom user prompt for VLM (optional)",
    ),
    camera_views: str = Form(
        default="+x+y+z,-x-y-z",
        description="Comma-separated camera views for rendering (default: ``+x+y+z,-x-y-z``)",
    ),
    steps: str = Form(
        default="",
        description="Comma-separated steps to run (optional, default: all steps)",
    ),
    coverage_policy: str = Form(
        default="allow_partial",
        description=(
            "Material coverage policy: strict fails closed before final success; "
            "allow_partial preserves artifacts with explicit partial readiness."
        ),
    ),
    optimize_usd: str = Form(
        default="true",
        description="Enable USD optimization step (true/false, default: true)",
    ),
    enable_deinstance: str = Form(
        default="true",
        description="Enable deinstance operation when optimize_usd is true (true/false, default: true)",
    ),
    enable_split: str = Form(
        default="true",
        description="Enable split meshes operation when optimize_usd is true (true/false, default: true)",
    ),
    enable_deduplicate: str = Form(
        default="true",
        description="Enable deduplicate operation when optimize_usd is true (true/false, default: true)",
    ),
    skip_instances: str = Form(
        default="true",
        description="Skip instance prims during dataset building (true/false, default: true)",
    ),
    skip_prototypes: str = Form(
        default="false",
        description="Skip prototype prims during dataset building (true/false, default: false)",
    ),
    skip_existing_materials: str = Form(
        default="false",
        description="Skip prims with existing material bindings (true/false, default: false)",
    ),
    pdf_descriptions: str = Form(
        default="",
        description='JSON array of descriptions for each reference PDF (e.g., \'["spec sheet", "manual"]\') (optional)',
    ),
    pdf_first_page: int | None = Form(
        default=None,
        description="First page to convert from PDFs (1-indexed, optional)",
    ),
    pdf_last_page: int | None = Form(
        default=None,
        description="Last page to convert from PDFs (1-indexed, optional)",
    ),
    vlm_model: str | None = Form(
        default=None,
        description="VLM model to use for prediction (optional, uses server default if not specified)",
    ),
    vlm_max_workers: int | None = Form(
        default=None,
        description="Maximum parallel VLM workers for prediction (optional, default: 64)",
    ),
    render_num_workers: int | None = Form(
        default=None,
        ge=1,
        le=config.max_render_num_workers,
        description=(
            "Maximum parallel render workers for build_dataset_usd "
            "(optional, uses Material Agent default if unspecified; "
            f"max: {config.max_render_num_workers})"
        ),
    ),
    enable_prim_clustering: str = Form(
        default="false",
        description=(
            "Enable image-based prim clustering before prediction "
            "(true/false, default: false)"
        ),
    ),
    cluster_min_prims: int | None = Form(
        default=None,
        ge=1,
        description="Minimum prim count before prim clustering runs",
    ),
    cluster_embedding_backend: str | None = Form(
        default=None,
        description="Embedding backend for prim clustering (default: service config)",
    ),
    cluster_embedding_model: str | None = Form(
        default=None,
        description="Embedding model for prim clustering (default: service config)",
    ),
    cluster_embedding_base_url: str | None = Form(
        default=None,
        description="Optional embedding API base URL for prim clustering",
    ),
    cluster_embedding_max_workers: int | None = Form(
        default=None,
        ge=1,
        description="Maximum parallel embedding workers for prim clustering",
    ),
    cluster_embedding_batch_size: int | None = Form(
        default=None,
        ge=1,
        description="Embedding batch size for prim clustering",
    ),
    cluster_max_size: int | None = Form(
        default=None,
        ge=1,
        description=(
            "Maximum prims that can share one propagated representative "
            "prediction before the cluster is split"
        ),
    ),
    cluster_similarity_threshold_low: float | None = Form(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for low-complexity prim clusters",
    ),
    cluster_similarity_threshold_medium: float | None = Form(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for medium-complexity prim clusters",
    ),
    cluster_similarity_threshold_high: float | None = Form(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for high-complexity prim clusters",
    ),
    cluster_report: str = Form(
        default="true",
        description="Generate a cluster HTML report when clustering runs",
    ),
    material_library: str = Form(
        default="default",
        description="Material library ID to use (default: 'default'). Ignored when materials_zip is provided.",
    ),
    layer_only: str = Form(
        default="false",
        description=(
            "Output only a material binding layer instead of a full USD "
            "(true/false, default: false). When true, the output USD "
            "contains only material definitions and bindings as 'over' "
            "opinions, preserving the original scene structure."
        ),
    ),
    large_scene: str = Form(
        default="false",
        description=(
            "Run the public large-scene material workflow (true/false, default: false)."
        ),
    ),
    scene_workers: int | None = Form(
        default=None,
        ge=1,
        le=config.max_scene_workers,
        description=(
            "Maximum parallel large-scene sub-asset workers "
            f"(optional, max: {config.max_scene_workers})."
        ),
    ),
    scene_assets: str = Form(
        default="",
        description=(
            "Comma-separated scene sub-asset names or prim path prefixes to process "
            "(large-scene mode only)."
        ),
    ),
    scene_resume: str = Form(
        default="false",
        description="Reuse existing large-scene analysis/extraction outputs.",
    ),
    scene_from_step: str = Form(
        default="",
        description="Resume per-asset pipelines from this step name.",
    ),
    scene_skip_existing: str = Form(
        default="false",
        description="Skip large-scene assets already marked completed.",
    ),
    scene_no_render: str = Form(
        default="false",
        description="Skip final composed-scene rendering in large-scene mode.",
    ),
    scene_simulate: str = Form(
        default="false",
        description=(
            "Run large-scene mode with mock render/VLM backends and generated "
            "predictions for smoke testing."
        ),
    ),
    scene_simulate_mock_analyze: str = Form(
        default="false",
        description=(
            "Also mock the large-scene analysis LLM when scene_simulate=true."
        ),
    ),
    scene_fail_on_validation_error: str = Form(
        default="false",
        description=(
            "Mark the large-scene job failed when scene validation reports errors."
        ),
    ),
    scene_filters: str = Form(
        default="",
        description="JSON object of scene analyze filters for large-scene mode.",
    ),
) -> SessionCreated:
    """Create and execute a material assignment pipeline.

    Two modes:
    1. New session: Provide usd_file, creates new session and uploads USD
    2. Existing session: Provide session_id (from /upload-usd), skips USD upload
    """
    manager = get_session_manager()
    user_email = _normalize_user_email(user_email)

    # Inspect only caller-owned fields here. Service-owned VLM/LLM credentials
    # are injected later into the in-memory execution config and deliberately
    # remain outside this durable request projection.
    await _validate_request_owned_durable_content(
        {
            "form_values": (
                session_id,
                user_email,
                reference_descriptions,
                generated_reference_id,
                enable_material_generation,
                material_generation_guidance,
                user_prompt,
                camera_views,
                steps,
                coverage_policy,
                optimize_usd,
                enable_deinstance,
                enable_split,
                enable_deduplicate,
                skip_instances,
                skip_prototypes,
                skip_existing_materials,
                pdf_descriptions,
                vlm_model,
                enable_prim_clustering,
                cluster_embedding_backend,
                cluster_embedding_model,
                cluster_embedding_base_url,
                cluster_report,
                material_library,
                layer_only,
                large_scene,
                scene_assets,
                scene_resume,
                scene_from_step,
                scene_skip_existing,
                scene_no_render,
                scene_simulate,
                scene_simulate_mock_analyze,
                scene_fail_on_validation_error,
                scene_filters,
            ),
            "upload_filenames": (
                usd_file.filename if usd_file is not None else None,
                *(upload.filename for upload in reference_images),
                *(upload.filename for upload in reference_pdfs),
                materials_zip.filename if materials_zip is not None else None,
            ),
        }
    )

    # Parse camera views (use API default if not provided)
    camera_view_list = [v.strip() for v in camera_views.split(",") if v.strip()]
    if not camera_view_list:
        camera_view_list = DEFAULT_CAMERA_DIRECTIONS

    # Parse steps
    steps_list = None
    if steps:
        steps_list = [s.strip() for s in steps.split(",") if s.strip()]
    coverage_policy_value = _parse_coverage_policy(coverage_policy)

    # Use default user prompt if not provided
    user_prompt_text = user_prompt.strip() if user_prompt else None
    prim_clustering_enabled = enable_prim_clustering.lower() == "true"
    cluster_prims_step_config = (
        _build_cluster_prims_step_config(
            cluster_min_prims=cluster_min_prims,
            cluster_embedding_backend=cluster_embedding_backend,
            cluster_embedding_model=cluster_embedding_model,
            cluster_embedding_base_url=cluster_embedding_base_url,
            cluster_embedding_max_workers=cluster_embedding_max_workers,
            cluster_embedding_batch_size=cluster_embedding_batch_size,
            cluster_max_size=cluster_max_size,
            cluster_similarity_threshold_low=cluster_similarity_threshold_low,
            cluster_similarity_threshold_medium=cluster_similarity_threshold_medium,
            cluster_similarity_threshold_high=cluster_similarity_threshold_high,
            cluster_report=cluster_report,
        )
        if prim_clustering_enabled
        else None
    )

    large_scene_bool = _parse_bool_form(large_scene)
    scene_worker_count = scene_workers or 1
    scene_asset_list = _parse_csv_form(scene_assets)
    scene_resume_bool = _parse_bool_form(scene_resume)
    scene_skip_existing_bool = _parse_bool_form(scene_skip_existing)
    scene_no_render_bool = _parse_bool_form(scene_no_render)
    scene_simulate_bool = _parse_bool_form(scene_simulate)
    scene_simulate_mock_analyze_bool = _parse_bool_form(scene_simulate_mock_analyze)
    scene_fail_on_validation_error_bool = _parse_bool_form(
        scene_fail_on_validation_error
    )
    material_generation_enabled = _parse_bool_form(enable_material_generation)
    scene_from_step_value = scene_from_step.strip() or None
    scene_filters_config = _parse_json_object_form(scene_filters, "scene_filters")
    ref_descriptions = _parse_json_list_form(
        reference_descriptions,
        "reference_descriptions",
    )
    pdf_desc_list = _parse_json_list_form(pdf_descriptions, "pdf_descriptions")
    await _validate_request_owned_durable_content(
        {
            "reference_descriptions": ref_descriptions,
            "pdf_descriptions": pdf_desc_list,
            "scene_filters": scene_filters_config,
        }
    )
    created_new_session = False
    historical_reference_descriptions: list[Any] | None = None
    historical_materials_plan: _HistoricalMaterialsPlan | None = None

    if large_scene_bool and coverage_policy_value == "strict":
        raise HTTPException(
            status_code=400,
            detail=(
                "coverage_policy=strict currently requires the single-asset "
                "pipeline because large-scene prim-level binding evidence is not "
                "yet qualified. Use allow_partial for an explicit not_evaluated "
                "coverage result."
            ),
        )

    if material_generation_enabled and large_scene_bool:
        raise HTTPException(
            status_code=400,
            detail="Material generation mode is not yet supported for large-scene workflows.",
        )
    if material_generation_enabled and not config.image_gen_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Material generation mode requires deployment image-generation "
                "configuration (MA_IMAGE_GEN_BACKEND, MA_IMAGE_GEN_MODEL, "
                "MA_IMAGE_GEN_BASE_URL, and MA_IMAGE_GEN_API_KEY as needed)."
            ),
        )

    pipeline_session_config = {
        "camera_views": camera_view_list,
        "user_prompt": user_prompt_text,
        "has_reference_images": len(reference_images) > 0,
        "num_reference_images": len(reference_images),
        "has_reference_pdfs": len(reference_pdfs) > 0,
        "num_reference_pdfs": len(reference_pdfs),
        "optimize_usd": optimize_usd.lower() == "true",
        "vlm_model": vlm_model,
        "render_num_workers": render_num_workers,
        "steps": steps_list,
        "coverage_policy": coverage_policy_value,
        "generated_reference_id": generated_reference_id or None,
        "enable_material_generation": material_generation_enabled,
        "material_generation_guidance": material_generation_guidance or None,
        "material_generation_texture_size": material_generation_texture_size,
        "large_scene": large_scene_bool,
        "scene_workers": scene_worker_count if large_scene_bool else None,
        "scene_assets": scene_asset_list if large_scene_bool else [],
        "scene_resume": scene_resume_bool if large_scene_bool else False,
        "scene_from_step": scene_from_step_value if large_scene_bool else None,
        "scene_skip_existing": (
            scene_skip_existing_bool if large_scene_bool else False
        ),
        "scene_no_render": scene_no_render_bool if large_scene_bool else False,
        "scene_simulate": scene_simulate_bool if large_scene_bool else False,
        "scene_simulate_mock_analyze": (
            scene_simulate_mock_analyze_bool if large_scene_bool else False
        ),
        "scene_fail_on_validation_error": (
            scene_fail_on_validation_error_bool if large_scene_bool else False
        ),
        **_cluster_session_config_from_step_config(
            enabled=prim_clustering_enabled,
            step_config=cluster_prims_step_config,
        ),
    }

    # Re-scan the exact normalized/parsed values that can enter session
    # metadata, description artifacts, or worker-derived durable output. This
    # catches structured short credentials such as {"api_key": "..."} that a
    # raw JSON string alone cannot classify reliably.
    await _validate_request_owned_durable_content(
        {
            "pipeline_session_config": pipeline_session_config,
            "user_email": user_email,
            "reference_descriptions": ref_descriptions,
            "pdf_descriptions": pdf_desc_list,
            "scene_filters": scene_filters_config,
            "material_library": material_library,
            "layer_only": layer_only,
        }
    )

    # Two execution paths:
    if session_id:
        # Path 1: Use existing session (USD already uploaded via /upload-usd)
        logger.info(f"Using existing session {session_id[:8]}...")

        metadata = await manager.get_session_metadata(session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Session not found")
        if metadata.get("status") != "ready":
            raise HTTPException(
                status_code=409,
                detail=(
                    "An existing session can only start a pipeline while it is ready"
                ),
            )

        session_dir = manager.get_session_dir(session_id)
        if not (materials_zip and materials_zip.filename):
            historical_materials_plan = await _preflight_historical_session_materials(
                manager,
                session_id,
                session_dir,
            )
        if not reference_images and not ref_descriptions:
            historical_reference_descriptions = (
                await _preflight_historical_reference_descriptions(
                    manager,
                    session_id,
                    session_dir,
                )
            )
        existing_config = metadata.get("config", {})
        if not isinstance(existing_config, dict):
            existing_config = {}
        updated_config = {**existing_config, **pipeline_session_config}

        # Persist upload-first configuration only after the disjoint admission
        # check above. Regeneration accepts terminal sessions only, and generic
        # writes reject any active regeneration claim.
        try:
            await manager.update_session(
                session_id,
                {
                    "config": updated_config,
                },
            )
        except RegenerationClaimConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    else:
        # Path 2: New session (legacy flow - upload USD now)
        if not usd_file:
            raise HTTPException(
                status_code=400, detail="Either usd_file or session_id must be provided"
            )

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Validate file extension
        if usd_file.filename:
            ext = Path(usd_file.filename).suffix.lower()
            if ext not in config.allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {ext}. Allowed: {config.allowed_extensions}",
                )

        # Create session directory structure
        session_dir = await manager.create_session(
            session_id,
            config=pipeline_session_config,
        )
        created_new_session = True

        # Save uploaded USD file using streaming, preserving original extension
        original_ext = (
            Path(usd_file.filename).suffix.lower() if usd_file.filename else ".usd"
        )
        usd_path = session_dir / "input" / f"scene{original_ext}"
        failure_phase = FailurePhase.LOCAL_PUBLICATION
        try:
            # Stream file to disk in chunks (2MB at a time)
            total_bytes = await _stream_copy(usd_file, usd_path)

            # Check file size after streaming
            size_mb = total_bytes / (1024 * 1024)
            if size_mb > config.max_upload_size_mb:
                # Remove the file if it exceeds limit
                usd_path.unlink(missing_ok=True)
                await manager.delete_session(session_id)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
                )

            logger.info(
                f"Saved USD file for session {session_id}: {size_mb:.2f}MB ({original_ext})"
            )

            # Store asset metadata in session for telemetry
            original_filename = usd_file.filename or f"scene{original_ext}"
            failure_phase = FailurePhase.PERSISTENCE_VERIFICATION
            await manager.update_session(
                session_id,
                {
                    "asset": {
                        "filename": original_filename,
                        "file_size_bytes": total_bytes,
                        "file_extension": original_ext,
                    }
                },
            )

            # Mirror uploaded USD to external store if configured
            try:
                await manager.put_file_to_store(
                    session_id,
                    f"input/scene{original_ext}",
                    str(usd_path),
                    content_type="application/octet-stream",
                )
            except Exception:
                log_durable_failure(
                    logger,
                    "pipeline_usd_sync_failed",
                    phase=FailurePhase.SYNC_UPLOAD,
                    retryable=True,
                )

            if not large_scene_bool:
                # Trigger background render of input USD (preview before material assignment)
                # This runs in parallel while user configures other settings.
                failure_phase = FailurePhase.PIPELINE_EXECUTION
                asyncio.create_task(_render_input_preview(session_id, session_dir))
                logger.info(f"Triggered input preview render for {session_id[:8]}...")

        except HTTPException:
            raise  # Re-raise HTTP exceptions as-is
        except Exception:
            log_durable_failure(
                logger,
                "pipeline_usd_ingest_failed",
                phase=failure_phase,
                retryable=True,
            )
            await manager.delete_session(session_id)
            raise HTTPException(
                status_code=500, detail="Failed to save USD file"
            ) from None

    # Store user_email at the top level of session metadata
    await manager.update_session(session_id, {"user_email": user_email})

    # Validate input USD exists (both new + existing session flows)
    # Supports .usd, .usda, .usdc, .usdz extensions
    input_usd_path = _find_input_usd(session_dir)
    if not input_usd_path:
        # May be on a different instance — pull input/ from store and retry
        pulled = await manager.sync_from_store(session_id, prefix="input/")
        if pulled > 0:
            logger.info(
                f"Pulled {pulled} input file(s) from store for session {session_id[:8]}"
            )
        input_usd_path = _find_input_usd(session_dir)
    if not input_usd_path:
        raise HTTPException(
            status_code=400,
            detail="Input USD not found for session",
        )

    if large_scene_bool:
        try:
            default_prim_path = await _ensure_large_scene_stage_file(input_usd_path)
        except HTTPException:
            if created_new_session:
                await manager.delete_session(session_id)
            raise
        await manager.update_session(
            session_id,
            {
                "scene_input": {
                    "usd_path": str(input_usd_path),
                    "default_prim_path": default_prim_path,
                }
            },
        )
        logger.info(
            "Validated large-scene input %s with default root prim %s",
            session_id[:8],
            default_prim_path,
        )

    # Save reference images if provided using streaming
    ref_image_paths = []
    if reference_images:
        reference_dir = session_dir / "input" / "reference_images"
        reference_dir.mkdir(parents=True, exist_ok=True)

        for i, ref_image in enumerate(reference_images):
            reference_failure_phase = FailurePhase.LOCAL_PUBLICATION
            try:
                # Stream reference image to disk
                ref_ext = (
                    Path(ref_image.filename).suffix if ref_image.filename else ".png"
                )
                ref_path = reference_dir / f"reference_{i:04d}{ref_ext}"

                await _stream_copy(ref_image, ref_path)
                ref_image_paths.append(str(ref_path))

                logger.info(f"Saved reference image {i + 1}/{len(reference_images)}")

                # Mirror to external store if configured
                try:
                    ct = "image/png" if str(ref_ext).lower() == ".png" else "image/jpeg"
                    reference_failure_phase = FailurePhase.SYNC_UPLOAD
                    await manager.put_file_to_store(
                        session_id,
                        f"input/reference_images/reference_{i:04d}{ref_ext}",
                        str(ref_path),
                        content_type=ct,
                    )
                except Exception:
                    log_durable_failure(
                        logger,
                        "pipeline_reference_image_publication_failed",
                        phase=reference_failure_phase,
                        retryable=True,
                    )

            except Exception:
                log_durable_failure(
                    logger,
                    "pipeline_reference_image_publication_failed",
                    phase=reference_failure_phase,
                    retryable=True,
                )
                # Continue with other images

        # Save descriptions metadata if provided
        if ref_descriptions:
            ref_metadata = reference_dir / "descriptions.json"
            with open(ref_metadata, "w") as f:
                json.dump(ref_descriptions, f)
            logger.info(f"Saved {len(ref_descriptions)} reference image descriptions")
    else:
        ref_image_paths = await _restore_existing_session_files(
            manager,
            session_id,
            session_dir,
            "input/reference_images",
            "reference_*",
        )
        if ref_image_paths:
            logger.info(
                "Reusing %s reference image(s) from session %s",
                len(ref_image_paths),
                session_id[:8],
            )
            if not ref_descriptions:
                ref_descriptions = (
                    copy.deepcopy(historical_reference_descriptions)
                    if historical_reference_descriptions is not None
                    else _load_reference_descriptions(
                        session_dir / "input" / "reference_images"
                    )
                )

    # Save reference PDFs if provided using streaming
    ref_pdf_paths = []
    if reference_pdfs:
        pdf_dir = session_dir / "input" / "reference_pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        for i, ref_pdf in enumerate(reference_pdfs):
            reference_failure_phase = FailurePhase.LOCAL_PUBLICATION
            try:
                # Validate PDF extension
                pdf_ext = (
                    Path(ref_pdf.filename).suffix.lower()
                    if ref_pdf.filename
                    else ".pdf"
                )
                if pdf_ext != ".pdf":
                    logger.warning(
                        f"Skipping non-PDF file: {ref_pdf.filename} (extension: {pdf_ext})"
                    )
                    continue

                # Stream PDF to disk
                pdf_path = pdf_dir / f"reference_{i:04d}.pdf"
                await _stream_copy(ref_pdf, pdf_path)
                ref_pdf_paths.append(str(pdf_path))

                logger.info(f"Saved reference PDF {i + 1}/{len(reference_pdfs)}")

                # Mirror to external store if configured
                try:
                    reference_failure_phase = FailurePhase.SYNC_UPLOAD
                    await manager.put_file_to_store(
                        session_id,
                        f"input/reference_pdfs/reference_{i:04d}.pdf",
                        str(pdf_path),
                        content_type="application/pdf",
                    )
                except Exception:
                    log_durable_failure(
                        logger,
                        "pipeline_reference_pdf_publication_failed",
                        phase=reference_failure_phase,
                        retryable=True,
                    )

            except Exception:
                log_durable_failure(
                    logger,
                    "pipeline_reference_pdf_publication_failed",
                    phase=reference_failure_phase,
                    retryable=True,
                )
                # Continue with other PDFs
    else:
        ref_pdf_paths = await _restore_existing_session_files(
            manager,
            session_id,
            session_dir,
            "input/reference_pdfs",
            "reference_*.pdf",
        )
        if ref_pdf_paths:
            logger.info(
                "Reusing %s reference PDF(s) from session %s",
                len(ref_pdf_paths),
                session_id[:8],
            )

    # Resolve materials: custom zip > saved custom materials > selected library
    # > default library. SimReady IDs are intentionally resolved after custom
    # material checks so an uploaded custom ZIP can override a SimReady selection.
    has_custom_materials = False
    selected_lib = None
    session_materials_library = config.materials_library_path
    session_materials_entries = config.materials

    if materials_zip and materials_zip.filename:
        logger.info(f"Processing custom materials zip: {materials_zip.filename}")

        # Create materials directory in session
        materials_dir = session_dir / "materials"
        materials_dir.mkdir(parents=True, exist_ok=True)

        # Save zip file with size check
        zip_path = materials_dir / "materials.zip"
        try:
            total_bytes = await _stream_copy(materials_zip, zip_path)
            size_mb = total_bytes / (1024 * 1024)

            # Apply same size limit as USD files
            if size_mb > config.max_upload_size_mb:
                zip_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Materials ZIP too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
                )

            logger.info(f"Saved materials zip: {zip_path} ({size_mb:.2f}MB)")

            # Extract and validate
            try:
                (
                    session_materials_library,
                    session_materials_entries,
                ) = await asyncio.to_thread(
                    _extract_and_validate_materials_zip,
                    zip_path,
                    materials_dir,
                )
                has_custom_materials = True
            except HTTPException:
                log_durable_failure(
                    logger,
                    "pipeline_materials_archive_validation_failed",
                    phase=FailurePhase.LOCAL_PUBLICATION,
                    retryable=False,
                )
                raise

            # Update session metadata
            await manager.update_session(
                session_id,
                {
                    "has_custom_materials": True,
                    "custom_materials_count": len(session_materials_entries),
                },
            )

            logger.info(
                f"Using custom materials: {len(session_materials_entries)} entries, "
                f"library: {session_materials_library}"
            )

        except HTTPException:
            await asyncio.to_thread(
                _discard_rejected_materials_archive,
                materials_dir,
                zip_path,
            )
            raise
        except Exception:
            log_durable_failure(
                logger,
                "pipeline_materials_archive_publication_failed",
                phase=FailurePhase.LOCAL_PUBLICATION,
                retryable=False,
            )
            raise HTTPException(
                status_code=400,
                detail="Failed to process materials zip",
            ) from None
    else:
        if historical_materials_plan is not None:
            restored_materials = await asyncio.to_thread(
                _materialize_historical_materials_plan,
                historical_materials_plan,
                session_dir,
            )
        elif created_new_session:
            restored_materials = await _restore_existing_session_materials(
                manager,
                session_id,
                session_dir,
            )
        else:
            restored_materials = None
        if restored_materials is not None:
            session_materials_library, session_materials_entries = restored_materials
            has_custom_materials = True
            await manager.update_session(
                session_id,
                {
                    "has_custom_materials": True,
                    "custom_materials_count": len(session_materials_entries),
                },
            )
            logger.info(
                "Reusing custom materials from session %s: %s entries, library: %s",
                session_id[:8],
                len(session_materials_entries),
                session_materials_library,
            )

    if not has_custom_materials:
        try:
            selected_lib = config.resolve_material_library(material_library)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if selected_lib:
            session_materials_library = selected_lib.library_path
            session_materials_entries = selected_lib.entries
        elif material_library != config.default_library_id:
            logger.warning(
                f"Unknown material library '{material_library}', "
                f"falling back to default"
            )

    # Build complete MAA API config dict here at entry point
    from material_agent.api import build_unified_pipeline_config

    # Determine steps
    pipeline_steps = steps_list or [
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
        "apply",
        "render",
    ]
    pipeline_steps = _inject_cluster_step(
        pipeline_steps,
        enable_prim_clustering=prim_clustering_enabled,
    )
    if material_generation_enabled:
        pipeline_steps = _insert_step_before(
            pipeline_steps,
            "generate_material_library",
            before_candidates=(
                "build_dataset_usd",
                "build_dataset_prepare_dataset",
                "predict",
                "apply",
            ),
        )

    # Add optimize_usd step if enabled (prepend to run first)
    optimize_usd_enabled = optimize_usd.lower() == "true"
    if optimize_usd_enabled and "optimize_usd" not in pipeline_steps:
        pipeline_steps = ["optimize_usd"] + pipeline_steps
        logger.info("USD optimization step enabled")
    restored_pipeline_steps = _inject_restore_usd_step(
        pipeline_steps,
        optimize_usd_enabled=optimize_usd_enabled,
    )
    if restored_pipeline_steps != pipeline_steps:
        pipeline_steps = restored_pipeline_steps
        logger.info(
            "USD restoration step enabled to preserve original topology before apply"
        )

    # Warn early if USD is very large (many prims) so UI can communicate latency
    threshold = DEFAULT_USD_PRIM_WARNING_THRESHOLD
    stage_info = await asyncio.to_thread(get_stage_info_from_path, input_usd_path)
    prim_count = stage_info.get("prim_count") if stage_info else None
    if prim_count is not None and prim_count > threshold:
        warn_step = (
            "build_dataset_usd"
            if "build_dataset_usd" in pipeline_steps
            else (pipeline_steps[0] if pipeline_steps else "pipeline")
        )
        warn_msg = (
            f"WARNING: Input USD contains {prim_count} prims (>{threshold}). "
            "Processing may be slow."
        )
        logger.warning("[%s] %s", session_id[:8], warn_msg)
        await get_event_bus().emit_for_owner(
            ProgressEvent(
                session_id=session_id,
                step=warn_step,
                state=StepState.RUNNING,
                percent=0,
                message=warn_msg,
                extra={"prim_count": prim_count, "prim_warning_threshold": threshold},
            ),
            regeneration_claim=None,
        )

    routing = _resolve_pipeline_model_routing(vlm_model)

    # Build base config (use session-specific materials if custom zip was provided)
    pipeline_config = build_unified_pipeline_config(
        project_name=session_id,
        session_id=session_id,
        input_usd_path=str(input_usd_path),
        output_usd_path=str(session_dir / "output" / "scene_with_materials.usd"),
        materials_library_path=session_materials_library,
        materials_entries=session_materials_entries,
        vlm_backend=routing.vlm_backend,
        vlm_model=routing.vlm_model,
        llm_backend=routing.llm_backend,
        llm_model=routing.llm_model,
        user_prompt=user_prompt_text,
        enabled_steps=pipeline_steps,
        working_dir=str(session_dir / "cache"),
    )
    if selected_lib and is_simready_library_id(selected_lib.id):
        pipeline_config["materials"]["simready"] = {
            "library_id": selected_lib.id,
            "release_tag": config.simready_release_tag,
            "manifest_path": config.simready_manifest_path,
            "cache_dir": config.simready_cache_dir,
            "split_archives_enabled": config.simready_split_archives_enabled,
        }

    # Override max_workers for predict step if specified
    if vlm_max_workers is not None and "predict" in pipeline_config.get("steps", {}):
        pipeline_config["steps"]["predict"]["max_workers"] = vlm_max_workers

    if prim_clustering_enabled:
        if cluster_prims_step_config is None:
            raise RuntimeError("Prim clustering config was not built")
        pipeline_config["steps"]["cluster_prims"] = cluster_prims_step_config
        logger.info(
            "Prim clustering enabled: backend=%s model=%s min_prims=%s",
            pipeline_config["steps"]["cluster_prims"]["embedding_service"],
            pipeline_config["steps"]["cluster_prims"]["embedding_model"],
            pipeline_config["steps"]["cluster_prims"]["min_prims_to_activate"],
        )

    # Configure optimize_usd step if enabled
    if optimize_usd_enabled:
        # Validate at least one operation is enabled
        enable_deinstance_bool = enable_deinstance.lower() == "true"
        enable_split_bool = enable_split.lower() == "true"
        enable_deduplicate_bool = enable_deduplicate.lower() == "true"

        if not any(
            [enable_deinstance_bool, enable_split_bool, enable_deduplicate_bool]
        ):
            raise HTTPException(
                status_code=400,
                detail="At least one optimization operation must be enabled when optimize_usd is true. "
                "Please select Deinstance, Split Meshes, or Deduplicate Geometry.",
            )

        optimization_config = {
            "scene_optimizer_settings": {
                "enable_deinstance": enable_deinstance_bool,
                "enable_split_meshes": enable_split_bool,
                "enable_deduplicate": enable_deduplicate_bool,
                # Use defaults for other settings
                "generate_report": True,
                "capture_stats": True,
                "verbose": False,
                "wait_for_assets": False,
                "stage_timeout": 180.0,
                "output_format": "usdc",
                "extract_geom_subset_indices": True,
            },
            # Flatten prototypes before optimization:
            # - Converts abstract prototypes (over/class) to def
            # - Inlines all referenced geometry
            # - Removes prototype prims
            "flatten_prototypes": True,
        }

        # Add to optimize_usd step config
        if "optimize_usd" not in pipeline_config["steps"]:
            pipeline_config["steps"]["optimize_usd"] = {}
        pipeline_config["steps"]["optimize_usd"]["optimization_config"] = (
            optimization_config
        )

        logger.info(
            f"Optimization config: deinstance={enable_deinstance}, "
            f"split={enable_split}, deduplicate={enable_deduplicate}"
        )

    # Parse skip_instances, skip_prototypes, and skip_existing_materials flags
    skip_instances_bool = skip_instances.lower() == "true"
    skip_prototypes_bool = skip_prototypes.lower() == "true"
    skip_existing_materials_bool = skip_existing_materials.lower() == "true"

    # Force skip_instances=true, skip_prototypes=false when optimize_usd is enabled
    # This allows processing of prototype prims after they are converted from abstract to def
    if optimize_usd_enabled:
        skip_instances_bool = True
        skip_prototypes_bool = False
        logger.info(
            "optimize_usd enabled: forcing skip_instances=true, skip_prototypes=false, flatten_prototypes=true"
        )

    # Log VLM model selection
    if vlm_model:
        logger.info("Using user-selected VLM model: %s", routing.vlm_model)

    # Log materials source for debugging
    if has_custom_materials:
        logger.info("Pipeline using CUSTOM materials from uploaded zip")
    elif selected_lib and selected_lib.id != config.default_library_id:
        logger.info(
            f"Pipeline using library '{selected_lib.id}' "
            f"({len(session_materials_entries)} materials)"
        )
    else:
        logger.info("Pipeline using SERVER DEFAULT materials")

    # Add reference images to input config
    if ref_image_paths:
        pipeline_config["input"]["reference_images"] = ref_image_paths

    # Explicitly inject a generated reference image when the caller selected one.
    if generated_reference_id:
        metadata = await manager.get_session_metadata(session_id)
        generated_ref = _get_generated_reference_entry(metadata, generated_reference_id)
        if not generated_ref:
            raise HTTPException(
                status_code=400,
                detail=f"Generated reference not found: {generated_reference_id}",
            )

        generated_key = generated_ref.get("key")
        if not isinstance(generated_key, str) or not generated_key:
            raise HTTPException(
                status_code=400,
                detail=f"Generated reference is missing a file key: {generated_reference_id}",
            )

        generated_ref_path = session_dir / generated_key
        if not generated_ref_path.exists():
            await manager.sync_from_store(session_id, prefix=generated_key)

        if not generated_ref_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Generated reference file is not available: {generated_reference_id}",
            )

        existing_refs = pipeline_config["input"].get("reference_images", [])
        pipeline_config["input"]["reference_images"] = existing_refs + [
            str(generated_ref_path)
        ]
        logger.info(
            "Injected selected generated reference image into pipeline config: %s",
            generated_reference_id,
        )

    if material_generation_enabled:
        reference_image_paths = pipeline_config.get("input", {}).get(
            "reference_images",
            [],
        )
        if not reference_image_paths:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Material generation mode requires at least one uploaded or "
                    "generated reference image."
                ),
            )
        _configure_generate_material_library_step(
            pipeline_config,
            routing,
            material_generation_guidance=material_generation_guidance,
            material_generation_texture_size=material_generation_texture_size,
        )
        logger.info(
            "Material generation mode enabled with %d reference image(s), "
            "texture_size=%d, image_backend=%s, image_model=%s",
            len(reference_image_paths),
            material_generation_texture_size,
            config.image_gen_backend,
            config.image_gen_model,
        )

    # Add reference PDFs to input config with conversion settings
    if ref_pdf_paths:
        pipeline_config["input"]["reference_pdfs"] = ref_pdf_paths

        # Add PDF conversion settings (dpi=150, format=png are defaults)
        if "build_dataset_prepare_dataset" not in pipeline_config.get("steps", {}):
            pipeline_config["steps"]["build_dataset_prepare_dataset"] = {"prompts": {}}

        pdf_conversion_config = {
            "dpi": 150,  # Default DPI
            "format": "png",  # Default format
        }
        if pdf_first_page is not None:
            pdf_conversion_config["first_page"] = pdf_first_page
        if pdf_last_page is not None:
            pdf_conversion_config["last_page"] = pdf_last_page

        pipeline_config["steps"]["build_dataset_prepare_dataset"]["pdf_conversion"] = (
            pdf_conversion_config
        )
        logger.info(
            f"Configured PDF conversion: {len(ref_pdf_paths)} PDFs, "
            f"pages {pdf_first_page or 'all'}-{pdf_last_page or 'all'}"
        )

    # Configure rendering for build_dataset_usd
    if "build_dataset_usd" in pipeline_config.get("steps", {}):
        # Use dict format for per-mode rendering configuration
        pipeline_config["steps"]["build_dataset_usd"]["renderer"].update(
            {
                "rendering_modes": {
                    "prim_only": {
                        "margin": 1.2,
                        "cameras": camera_view_list,
                        "camera_focus_mode": "prim",
                    },
                    "composition": {
                        "margin": 6.0,
                        "cameras": ["+x", "+y", "+z"],
                        "camera_focus_mode": "stage",
                        "skip_occluded_images": False,
                    },
                },
                "num_views": len(camera_view_list),
            }
        )

        # Configure prim_filters for skip_instances and skip_prototypes
        if "prim_filters" not in pipeline_config["steps"]["build_dataset_usd"]:
            pipeline_config["steps"]["build_dataset_usd"]["prim_filters"] = {}
        pipeline_config["steps"]["build_dataset_usd"]["prim_filters"].update(
            {
                "skip_instances": skip_instances_bool,
                "skip_prototypes": skip_prototypes_bool,
            }
        )

        # Set batch_size for async NVCF rendering (validated: 64 optimal for 128 instances)
        if "batch_size" not in pipeline_config["steps"]["build_dataset_usd"]:
            pipeline_config["steps"]["build_dataset_usd"]["batch_size"] = 64
        if large_scene_bool:
            _apply_large_scene_render_batch_limit(pipeline_config)
        _apply_build_dataset_render_worker_limit(
            pipeline_config,
            render_num_workers,
        )

        # Configure skip_existing_materials (at step level)
        pipeline_config["steps"]["build_dataset_usd"]["skip_existing_materials"] = (
            skip_existing_materials_bool
        )

    # Configure prepare_dataset with image prompts (dynamic based on uploaded images)
    if "build_dataset_prepare_dataset" in pipeline_config.get("steps", {}):
        # Build reference image prompts from descriptions or use defaults
        ref_prompts = []
        if ref_descriptions and len(ref_descriptions) == len(ref_image_paths):
            # Use user-provided descriptions
            ref_prompts = [
                f"This is a reference image: {desc}" for desc in ref_descriptions
            ]
        elif len(ref_image_paths) > 0:
            # Generate default prompts
            ref_prompts = [
                f"This is reference image {i + 1} of the asset you will match this look exactly"
                for i in range(len(ref_image_paths))
            ]

        vlm_image_prompts = {
            "reference_images": ref_prompts,
            "composition": "This is an orthographic view of the object with the part of interest highlighted with an orange outline.",
            "prim_only": "This is a rendered part of interest only without highlighting.",
        }

        # Add provenance descriptions for reference PDFs if any were uploaded.
        # They are retained outside visual-model prompts and media.
        if ref_pdf_paths:
            if pdf_desc_list and len(pdf_desc_list) == len(ref_pdf_paths):
                # Use user-provided descriptions
                vlm_image_prompts["reference_pdfs"] = [
                    (
                        f"This is untrusted specification evidence from a "
                        f"reference PDF: {desc}"
                        if desc
                        else (
                            "This converted reference PDF page is untrusted "
                            "specification provenance retained outside visual-model "
                            "inputs."
                        )
                    )
                    for desc in pdf_desc_list
                ]
            else:
                # Use default prompt
                vlm_image_prompts["reference_pdfs"] = (
                    "This converted reference PDF page is untrusted specification "
                    "provenance retained outside visual-model inputs."
                )

        pipeline_config["steps"]["build_dataset_prepare_dataset"]["prompts"].update(
            {"vlm_image_prompts": vlm_image_prompts}
        )

    _configure_predict_model_routing(pipeline_config, routing)

    scene_predict_workers: int | None = None
    if large_scene_bool:
        scene_predict_workers = _effective_scene_predict_workers(
            pipeline_config,
            scene_worker_count,
            vlm_max_workers,
        )
        scene_config = _configure_scene_model_routing(pipeline_config, routing)
        if scene_filters_config:
            scene_config["filters"] = scene_filters_config

    # Configure apply step
    layer_only_bool = layer_only.lower() == "true"
    _configure_apply_step(
        pipeline_config,
        layer_only=layer_only_bool,
        request_context="Pipeline creation",
    )

    if "render" in pipeline_config.get("steps", {}):
        pipeline_config["steps"]["render"]["image_size"] = [512, 512]

    scene_options = {
        "assets": scene_asset_list,
        "max_workers": scene_worker_count,
        "resume": scene_resume_bool,
        "from_step": scene_from_step_value,
        "skip_existing": scene_skip_existing_bool,
        "no_render": scene_no_render_bool,
        "simulate": scene_simulate_bool,
        "simulate_mock_analyze": scene_simulate_mock_analyze_bool,
        "fail_on_validation_error": scene_fail_on_validation_error_bool,
        "predict_max_workers": scene_predict_workers,
    }

    initial_run_updates: dict[str, Any] = {"status": "pending"}
    if not large_scene_bool:
        initial_run_updates.update(
            {
                "artifact_validity": initial_artifact_validity(),
                "prediction_lineage_token": str(uuid.uuid4()),
            }
        )
    try:
        await manager.update_session(session_id, initial_run_updates)
    except RegenerationClaimConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await get_event_bus().seed_pending_session(session_id)

    job_registry = get_job_registry()
    if large_scene_bool:
        job = execute_scene_pipeline_async(
            session_id=session_id,
            config_dict=pipeline_config,
            session_manager=manager,
            user_email=user_email,
            scene_options=scene_options,
            coverage_policy=coverage_policy_value,
        )
        message = "Large-scene pipeline queued for execution"
        estimated_minutes = 45
    else:
        job = execute_pipeline_async(
            session_id=session_id,
            config_dict=pipeline_config,
            session_manager=manager,
            user_email=user_email,
            coverage_policy=coverage_policy_value,
        )
        message = "Pipeline queued for execution"
        estimated_minutes = 15

    try:
        await job_registry.register(session_id, job)
    except DuplicateJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to register pipeline for %s", session_id)
        raise HTTPException(
            status_code=503,
            detail="Failed to schedule pipeline",
        ) from exc

    logger.info(f"Pipeline registered and queued for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message=message,
        estimated_duration_minutes=estimated_minutes,
    )


@router.get("/{session_id}/status", response_model=PipelineStatus)
async def get_pipeline_status(session_id: str) -> PipelineStatus:
    """Get pipeline execution status with detailed progress.

    Reads from in-memory event bus state for fast, real-time accuracy.
    Falls back to disk-based SessionManager only for completed/stopped sessions.

    Args:
        session_id: Session identifier

    Returns:
        Detailed status including current step progress and preview images
    """
    event_bus = get_event_bus()
    manager = get_session_manager()
    job_registry = get_job_registry()

    # Try in-memory state first (active sessions)
    snapshot = await event_bus.get_fenced_snapshot(session_id)
    active_statuses = {"pending", "running", "cancelling"}
    pipeline_active = job_registry.is_running(session_id)
    if snapshot and snapshot.get("status") in active_statuses and not pipeline_active:
        snapshot = None
    metadata: dict[str, Any]

    if snapshot:
        metadata = snapshot
        if snapshot.get("status") not in active_statuses:
            disk_metadata = await manager.get_session_metadata(session_id)
            if disk_metadata:
                disk_metadata = normalize_legacy_completed_coverage(
                    disk_metadata,
                    pipeline_active=pipeline_active,
                )
            final_metadata_ready = bool(
                disk_metadata and _terminal_metadata_ready(disk_metadata)
            )
            if final_metadata_ready:
                assert disk_metadata is not None
                # Terminal metadata is authoritative once the executor has
                # atomically persisted final stats and coverage.  Before that,
                # keep the richer EventBus snapshot to avoid exposing the
                # short status-only persistence race.
                metadata = _merge_terminal_status_metadata(snapshot, disk_metadata)
            elif pipeline_active:
                # A listener may emit/persist a terminal event before the
                # executor writes final stats and coverage.  Do not let callers
                # observe terminal success/failure until that atomic write is
                # available.
                metadata = dict(snapshot)
                metadata["status"] = "running"
        preview_images = metadata.get("preview_images", [])

    else:
        # Session not in event bus - check disk for completed/old sessions
        disk_metadata = await manager.get_session_metadata(session_id)
        if not disk_metadata:
            raise HTTPException(status_code=404, detail="Session not found")

        metadata = normalize_legacy_completed_coverage(
            disk_metadata,
            pipeline_active=pipeline_active,
        )
        if (
            pipeline_active
            and metadata.get("status") in {"completed", "failed", "cancelled"}
            and not _terminal_metadata_ready(metadata)
        ):
            metadata = {**metadata, "status": "running"}
        preview_images = metadata.get("preview_images", [])

    # Build preview image URLs (using new assets router path)
    preview_urls = [f"/assets/{session_id}/preview/{img}" for img in preview_images]

    # Calculate elapsed time dynamically
    created_at = _parse_iso_datetime(metadata["created_at"])
    elapsed_seconds = int((datetime.now(UTC) - created_at).total_seconds())

    # Determine if can cancel (only if running)
    can_cancel = metadata.get("status") in ["pending", "running"]

    return PipelineStatus(
        session_id=session_id,
        status=metadata["status"],
        current_step=_current_step_with_fresh_elapsed(metadata.get("current_step")),
        completed_steps=metadata.get("completed_steps", []),
        overall_progress=metadata.get("overall_progress", {}),
        preview_images=preview_urls,
        can_cancel=can_cancel,
        elapsed_seconds=elapsed_seconds,
        created_at=metadata["created_at"],
        updated_at=metadata["updated_at"],
        coverage=metadata.get("coverage"),
    )


async def _pipeline_download_urls(
    manager: SessionManager,
    session_id: str,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Return only the artifact endpoints backed by preserved pipeline files."""
    session_dir = manager.get_session_dir(session_id)

    async def artifact_exists(key: str) -> bool:
        if key == "cache/predictions/prediction_report.html":
            store_key = manager.resolve_prediction_report_key(
                metadata,
                legacy_key=key,
            )
        else:
            store_key = manager.resolve_published_artifact_key(
                metadata,
                key,
                legacy_key=key,
            )
        legacy_local = (
            manager.store.kind == "local"
            and metadata.get("published_artifacts") is None
        )
        return (legacy_local and (session_dir / key).exists()) or bool(
            store_key and await manager.exists_in_store(session_id, store_key)
        )

    async def any_artifact_exists(keys: list[str]) -> bool:
        for key in keys:
            if await artifact_exists(key):
                return True
        return False

    async def all_artifacts_exist(keys: list[str]) -> bool:
        for key in keys:
            if not await artifact_exists(key):
                return False
        return True

    download_urls: dict[str, str] = {}
    if metadata.get("pipeline_type") == "large_scene":
        # Completed scene runs have stable artifact routes even when their final
        # metadata update races the results request.  Failed runs advertise only
        # files that were actually preserved before validation stopped the job.
        completed = metadata.get("status") == "completed"
        valid_output = artifact_is_valid(
            metadata, "rendered_output_usd"
        ) or artifact_is_valid(metadata, "applied_output_usd")
        if valid_output and (
            completed
            or await any_artifact_exists(
                [
                    "output/scene_with_materials_flat.usd",
                    "output/composed_scene_flat.usd",
                    "output/scene_with_materials.usd",
                ]
            )
        ):
            download_urls["output_usd"] = f"/artifacts/{session_id}/output"
        if completed or await artifact_exists("scene/manifest.json"):
            download_urls["scene_manifest"] = f"/artifacts/{session_id}/scene-manifest"
        scene_metadata = metadata.get("scene", {})
        if (
            completed
            and isinstance(scene_metadata, dict)
            and scene_metadata.get("validation_report_path")
        ) or await artifact_exists("scene/validation_report.json"):
            download_urls["scene_validation_report"] = (
                f"/artifacts/{session_id}/scene-validation-report"
            )
        if (
            completed
            and isinstance(scene_metadata, dict)
            and scene_metadata.get("scene_predictions_path")
        ) or await artifact_exists("scene/predictions.jsonl"):
            download_urls["scene_predictions"] = (
                f"/artifacts/{session_id}/scene-predictions"
            )
        if artifact_is_valid(metadata, "final_render") and (
            completed or await artifact_exists("output/scene_with_materials.png")
        ):
            download_urls["final_render"] = f"/artifacts/{session_id}/final-render"
        return download_urls

    flat_output_available = artifact_is_valid(
        metadata, "rendered_output_usd"
    ) and await any_artifact_exists(
        ["output/scene_with_materials_flat.usd", "output/composed_scene_flat.usd"]
    )
    applied_output_available = artifact_is_valid(
        metadata, "applied_output_usd"
    ) and await artifact_exists("output/scene_with_materials.usd")
    if flat_output_available or applied_output_available:
        download_urls["output_usd"] = f"/artifacts/{session_id}/output"
    restored_predictions_available = artifact_is_valid(
        metadata, "restored_predictions"
    ) and await artifact_exists("cache/restored/restored_predictions.jsonl")
    raw_predictions_available = artifact_is_valid(
        metadata, "raw_predictions"
    ) and await artifact_exists("cache/predictions/predictions.jsonl")
    if restored_predictions_available or raw_predictions_available:
        download_urls["predictions"] = f"/artifacts/{session_id}/predictions"
    materialized_report_available = artifact_is_valid(
        metadata, "prediction_report"
    ) and await artifact_exists("cache/predictions/prediction_report.html")
    report_inputs_available = artifact_is_valid(
        metadata, "raw_predictions"
    ) and await all_artifacts_exist(
        ["cache/predictions/predictions.jsonl", "cache/dataset/dataset.jsonl"]
    )
    if materialized_report_available or report_inputs_available:
        download_urls["report"] = f"/artifacts/{session_id}/report"
    if metadata.get("results", {}).get("cluster_prims_ran"):
        cluster_artifacts = {
            "cluster_map": (
                "cache/clusters/cluster_map.jsonl",
                f"/artifacts/{session_id}/cluster-map",
                "cluster_map",
            ),
            "cluster_report": (
                "cache/clusters/cluster_report.html",
                f"/artifacts/{session_id}/cluster-report",
                "cluster_report",
            ),
            "cluster_summary": (
                "cache/clusters/cluster_summary.json",
                f"/artifacts/{session_id}/cluster-summary",
                "cluster_summary",
            ),
            "cluster_representatives": (
                "cache/clusters/dataset_representatives.jsonl",
                f"/artifacts/{session_id}/cluster-representatives",
                "cluster_representatives",
            ),
        }
        for name, (key, url, artifact) in cluster_artifacts.items():
            if artifact_is_valid(metadata, artifact) and await artifact_exists(key):
                download_urls[name] = url
    return download_urls


@router.get("/{session_id}/results", response_model=PipelineResults | PipelineError)
async def get_pipeline_results(session_id: str) -> PipelineResults | PipelineError:
    """Get pipeline execution results (only available when completed).

    Args:
        session_id: Session identifier

    Returns:
        Results if completed, error if failed, or 202 if still running
    """
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")
    if metadata.get("published_artifacts") is None:
        # Legacy sessions used mutable canonical keys and still need the old
        # local-to-store synchronization path. Claimed generations publish only
        # immutable pointers during executor finalization.
        await manager.sync_session_to_store(session_id)

    metadata = normalize_legacy_completed_coverage(
        metadata,
        pipeline_active=get_job_registry().is_running(session_id),
    )
    status = metadata["status"]

    if status in {"completed", "failed"} and not _terminal_metadata_ready(metadata):
        # EventBus status persistence can precede the executor's atomic final
        # metadata write. Wait for either authoritative success/failure or a
        # terminal legacy record whose worker is no longer active.
        for _attempt in range(6):
            await asyncio.sleep(0.5)
            metadata = await manager.get_session_metadata(session_id)
            if not metadata:
                raise HTTPException(status_code=404, detail="Session not found")
            metadata = normalize_legacy_completed_coverage(
                metadata,
                pipeline_active=get_job_registry().is_running(session_id),
            )
            status = metadata["status"]
            if status not in {"completed", "failed"} or _terminal_metadata_ready(
                metadata
            ):
                break

        status = metadata["status"]

    if status == "completed":
        if not _terminal_metadata_ready(metadata):
            raise HTTPException(
                status_code=202,
                detail="Pipeline is finalizing results and coverage.",
            )
        download_urls = await _pipeline_download_urls(manager, session_id, metadata)

        return PipelineResults(
            session_id=session_id,
            status=status,
            stats=metadata.get("results", {}),
            timings=metadata.get("timings_breakdown"),
            download_urls=download_urls,
            duration_seconds=metadata.get("duration_seconds", 0),
            completed_at=metadata.get("completed_at", ""),
            coverage=metadata.get("coverage"),
        )

    elif status == "failed":
        if not _terminal_metadata_ready(metadata):
            raise HTTPException(
                status_code=202,
                detail="Pipeline is finalizing failure diagnostics.",
            )
        download_urls = await _pipeline_download_urls(manager, session_id, metadata)
        return PipelineError(
            session_id=session_id,
            status=status,
            error_message=metadata.get("error", "Unknown error"),
            failed_step=metadata.get("failed_step", "unknown"),
            completed_steps=[s["name"] for s in metadata.get("completed_steps", [])],
            partial_results=metadata.get("partial_results"),
            download_urls=download_urls,
            coverage=metadata.get("coverage"),
        )

    else:
        # Still running, pending, or cancelling
        raise HTTPException(
            status_code=202,
            detail=f"Pipeline still {status}. Check status endpoint for progress.",
        )


@router.post("/{session_id}/cancel")
async def cancel_pipeline(session_id: str) -> dict[str, str]:
    """Cancel a running pipeline.

    Uses JobRegistry to cancel the asyncio.Task directly for immediate,
    deterministic cancellation (no file markers needed).

    Args:
        session_id: Session identifier

    Returns:
        Cancellation acknowledgment
    """
    job_registry = get_job_registry()
    manager = get_session_manager()

    snapshot = await manager.get_session_metadata_versioned(session_id)
    metadata = snapshot.value
    metadata_version = snapshot.version
    if not metadata or metadata_version is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # A local reservation has no worker task yet. Cancel its registering owner
    # first so a regeneration ``before_start`` callback cannot erase the durable
    # marker after this endpoint writes it.
    reservation_cancelled = False
    if job_registry.is_reserved(session_id):
        reservation_cancelled = await job_registry.cancel(session_id)
    if reservation_cancelled:
        refreshed = await manager.get_session_metadata_versioned(session_id)
        if refreshed.value is None or refreshed.version is None:
            raise HTTPException(status_code=404, detail="Session not found")
        metadata = refreshed.value
        metadata_version = refreshed.version
        refreshed_claim = RegenerationClaim.from_metadata(metadata)
        refreshed_raw_claim = metadata.get("regeneration_claim")
        refreshed_claim_active = bool(
            refreshed_claim is not None
            and isinstance(refreshed_raw_claim, dict)
            and refreshed_raw_claim.get("active") is True
        )
        if not refreshed_claim_active and metadata.get("status") not in {
            "pending",
            "running",
        }:
            # A queued regeneration was cancelled before ``before_start``
            # claimed or mutated the session. Preserve the prior terminal run
            # instead of relabeling its artifacts as a cancelled generation.
            return {
                "session_id": session_id,
                "status": str(metadata["status"]),
                "message": "Queued regeneration cancelled before start",
            }

    # Put the durable marker before the metadata CAS. If a remote regeneration
    # claim wins the version first and clears this marker, the retry below sees
    # that exact claim and atomically sets its token-scoped cancel flag.
    await manager.store.put_bytes(session_id, CANCEL_KEY, b"")

    for _attempt in range(_CANCELLATION_CAS_ATTEMPTS):
        regeneration_claim = RegenerationClaim.from_metadata(metadata)
        raw_claim = metadata.get("regeneration_claim")
        active_claim = bool(
            regeneration_claim is not None
            and isinstance(raw_claim, dict)
            and raw_claim.get("active") is True
        )
        expired_claim = bool(
            active_claim
            and regeneration_claim is not None
            and regeneration_claim.lease_expires_at <= datetime.now(UTC)
        )

        if active_claim and regeneration_claim is not None:
            if expired_claim:
                cancelled = await manager.cancel_expired_regeneration_claim(
                    session_id,
                    regeneration_claim,
                )
                terminal_status = "cancelled"
                message = "Expired regeneration cancelled"
            else:
                cancelled = await manager.cancel_regeneration_claim(
                    session_id,
                    regeneration_claim,
                )
                terminal_status = "cancelling"
                message = "Pipeline cancellation requested"
            if not cancelled:
                refreshed = await manager.get_session_metadata_versioned(session_id)
                if refreshed.value is None or refreshed.version is None:
                    raise HTTPException(status_code=404, detail="Session not found")
                metadata = refreshed.value
                metadata_version = refreshed.version
                continue
            # Claim cancellation is durable; remove any marker from the losing
            # no-claim attempt so it cannot poison a later standard run.
            await manager.clear_cancellation(session_id)
            if job_registry.is_running(session_id):
                await job_registry.cancel(session_id)
            return {
                "session_id": session_id,
                "status": terminal_status,
                "message": message,
            }

        if metadata["status"] in {"cancelled", "canceled"}:
            await manager.clear_cancellation(session_id)
            return {
                "session_id": session_id,
                "status": "cancelled",
                "message": "Pipeline cancellation completed",
            }

        if metadata["status"] not in ["pending", "running"]:
            await manager.clear_cancellation(session_id)
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel pipeline with status: {metadata['status']}",
            )

        now = datetime.now(UTC)
        cancellation_metadata = dict(metadata)
        cancellation_updates: dict[str, Any] = {
            "status": "cancelled" if reservation_cancelled else "cancelling",
        }
        if reservation_cancelled:
            cancellation_updates.update(
                {
                    "cancelled_at": now.isoformat(),
                    "can_cancel": False,
                }
            )
        manager._apply_metadata_updates(
            cancellation_metadata,
            cancellation_updates,
            (),
            now=now,
        )
        try:
            await manager.store.replace_json_if_version(
                session_id,
                METADATA_KEY,
                cancellation_metadata,
                metadata_version,
            )
        except JsonPreconditionError:
            refreshed = await manager.get_session_metadata_versioned(session_id)
            if refreshed.value is None or refreshed.version is None:
                raise HTTPException(status_code=404, detail="Session not found")
            metadata = refreshed.value
            metadata_version = refreshed.version
            continue

        # Running local tasks use direct asyncio cancellation as the fast path.
        # Remote standard workers poll the marker written before the CAS.
        if not reservation_cancelled and job_registry.is_running(session_id):
            await job_registry.cancel(session_id)
        return {
            "session_id": session_id,
            "status": "cancelled" if reservation_cancelled else "cancelling",
            "message": "Pipeline cancellation requested",
        }

    await manager.clear_cancellation(session_id)
    raise HTTPException(
        status_code=409,
        detail="Session ownership remained contended during cancellation",
    )


@router.get("/{session_id}/events")
async def stream_progress_events(session_id: str) -> EventSourceResponse:
    """Stream real-time progress events via Server-Sent Events (SSE).

    This endpoint provides live updates as the pipeline executes. The web UI
    can subscribe to this stream to show real-time progress without polling.

    Args:
        session_id: Session identifier

    Returns:
        SSE event stream with progress updates

    Example client (JavaScript):
        const eventSource = new EventSource(`/pipeline/${sessionId}/events`);
        eventSource.addEventListener('progress', (e) => {
            const data = JSON.parse(e.data);
            console.log(`Step: ${data.step}, Progress: ${data.percent}%`);
        });
    """
    event_bus = get_event_bus()

    # Verify session exists (either in EventBus or SessionManager)
    snapshot = await event_bus.get_fenced_snapshot(session_id)
    if snapshot is None:
        # Check if it exists in session manager but hasn't started yet
        manager = get_session_manager()
        if not await manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        # If an active claim has no matching local fenced snapshot, this pod is
        # not the regeneration owner (or ownership preparation has not seeded
        # this pod yet).  Return 503 for every active claimed state so a new
        # stream cannot attach to an empty local queue and hang.  Unclaimed
        # pending sessions may still be waiting for this pod's executor.
        metadata = await manager.get_session_metadata(session_id)
        status = (metadata or {}).get("status", "unknown")
        raw_claim = (metadata or {}).get("regeneration_claim")
        active_claim = isinstance(raw_claim, dict) and raw_claim.get("active") is True
        if status == "running" or (
            active_claim and status in {"pending", "running", "cancelling"}
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pipeline is running on a different instance; use polling instead"
                ),
            )

    async def event_generator() -> Any:
        """Generate SSE events from the session's event queue."""
        queue = event_bus.get_queue(session_id)

        try:
            while True:
                # Wait for next event (with timeout to allow connection checks)
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    if not await event_bus.queued_event_is_current(event):
                        # This pod's regeneration claim was superseded. End the
                        # pinned stream without leaking its queued stale event.
                        break

                    # Serialize event as JSON
                    event_data = event.model_dump_json()

                    # Yield SSE-formatted message
                    yield {
                        "event": "progress",
                        "data": event_data,
                    }

                    # Stop streaming only when OVERALL pipeline completes or fails
                    # Don't stop when individual steps complete (e.g., render at 50%)
                    should_close = False

                    if (
                        event.state == "failed"
                        and isinstance(event.extra, dict)
                        and event.extra.get("pipeline_failed") is True
                    ):
                        # Step-level failures can precede the executor's
                        # persisted failure diagnostics. Close only on its
                        # authoritative post-persistence failure event.
                        should_close = True
                    elif event.state == "cancelled":
                        # Cancellation is terminal for the whole registered job.
                        should_close = True
                    elif (
                        event.state == "completed"
                        and (event.overall_percent or 0) >= 100
                        and isinstance(event.extra, dict)
                        and event.extra.get("pipeline_completed") is True
                        and "coverage" in event.extra
                    ):
                        # Only the executor's post-persistence completion event
                        # is terminal. Underlying workflow completion can precede
                        # strict coverage qualification.
                        should_close = True

                    if should_close:
                        # Send final event then close stream
                        yield {
                            "event": "done",
                            "data": f'{{"session_id": "{session_id}", "final_state": "{event.state}"}}',
                        }
                        break

                except TimeoutError:
                    if await event_bus.get_fenced_snapshot(session_id) is None:
                        break
                    # Send keepalive ping
                    yield {"event": "ping", "data": "keepalive"}

        except asyncio.CancelledError:
            logger.debug(f"SSE stream cancelled for {session_id[:8]}...")
            raise

    return EventSourceResponse(event_generator(), ping=15)


@router.post("/{session_id}/regenerate", response_model=SessionCreated, status_code=202)
async def regenerate_pipeline(
    session_id: str,
    request: RegenerateRequest,
) -> SessionCreated:
    """Regenerate specific pipeline steps from cached data.

    Useful for re-running apply step with different settings without re-rendering.

    Args:
        session_id: Session identifier
        request: Regeneration request with steps and overrides

    Returns:
        Session status (same session_id)
    """
    manager = get_session_manager()

    planned_session = await manager.get_session_metadata_versioned(session_id)
    metadata = planned_session.value
    planning_version = planned_session.version
    if not metadata or planning_version is None:
        raise HTTPException(status_code=404, detail="Session not found")

    stored_config = metadata.get("config")
    configured_large_scene = (
        isinstance(stored_config, dict) and stored_config.get("large_scene") is True
    )
    if metadata.get("pipeline_type") == "large_scene" or configured_large_scene:
        raise HTTPException(
            status_code=400,
            detail=(
                "Large-scene regeneration is not supported by this endpoint; "
                "start a new large-scene pipeline instead."
            ),
        )

    if metadata.get("status") in {"completed", "failed", "cancelled"} and not (
        _terminal_metadata_ready(metadata)
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Pipeline terminal status is still being finalized; retry after "
                "authoritative results are available"
            ),
        )
    if metadata.get("terminal_events_quiesced") is False:
        raise HTTPException(
            status_code=409,
            detail="Pipeline terminal events are still being finalized",
        )

    # A crashed pod may leave an active status behind. Permit only an expired,
    # tokenized lease to reach the manager's atomic takeover CAS; ordinary jobs
    # and live claims remain non-regenerable.
    current_claim = RegenerationClaim.from_metadata(metadata)
    raw_current_claim = metadata.get("regeneration_claim")
    expired_claim_takeover = bool(
        current_claim is not None
        and isinstance(raw_current_claim, dict)
        and raw_current_claim.get("active") is True
        and current_claim.lease_expires_at <= datetime.now(UTC)
    )
    if metadata["status"] not in {"completed", "failed", "cancelled"} and not (
        expired_claim_takeover
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot regenerate while pipeline is {metadata['status']}",
        )

    # Update config with overrides
    original_config = metadata.get("config", {}).copy()

    # Override user prompt in metadata if provided (None means "no override")
    if request.user_prompt is not None:
        original_config["user_prompt"] = request.user_prompt
    coverage_policy_value = _parse_coverage_policy(
        request.coverage_policy
        if request.coverage_policy is not None
        else original_config.get("coverage_policy", "allow_partial")
    )
    original_config["coverage_policy"] = coverage_policy_value

    # Reject the prompt-overlaid durable config before planning can lead to a
    # claim, hydration, checkpoint rewrite, or metadata update.
    await _validate_request_owned_durable_content(
        {
            "config": original_config,
            "user_email": metadata.get("user_email"),
        }
    )

    # Get session directory and build complete config for regeneration
    from material_agent.api import build_unified_pipeline_config

    session_dir = manager.get_session_dir(session_id)
    input_bundle = await _plan_regeneration_input_bundle(
        manager,
        session_id,
        session_dir,
        metadata,
    )
    await _validate_request_owned_durable_content(
        {
            "reference_descriptions": input_bundle.reference_descriptions,
            "custom_materials": input_bundle.custom_materials,
        }
    )
    regeneration_validity = await _derive_regeneration_artifact_validity(
        manager,
        session_id,
        session_dir,
        metadata,
    )
    regeneration_step_evidence = await _derive_regeneration_step_evidence(
        manager,
        session_id,
        session_dir,
        regeneration_validity,
    )
    lineage_metadata = {
        **metadata,
        "artifact_validity": regeneration_validity,
        "_regeneration_step_evidence": regeneration_step_evidence,
    }
    camera_view_list = original_config.get("camera_views", DEFAULT_CAMERA_DIRECTIONS)
    render_num_workers = original_config.get("render_num_workers")
    steps_to_run = [s.value for s in request.steps]
    if request.layer_only and "apply" not in steps_to_run:
        raise HTTPException(
            status_code=400,
            detail="Regeneration: layer_only=true requires the apply step.",
        )
    regenerate_clustering = bool(
        original_config.get("enable_prim_clustering")
        and any(step in steps_to_run for step in ("predict", "benchmark"))
    )
    steps_to_run = _inject_cluster_step(
        steps_to_run,
        enable_prim_clustering=regenerate_clustering,
        require_prepare_step=False,
    )
    optimize_usd_enabled = _persisted_flag_enabled(original_config.get("optimize_usd"))
    steps_to_run = _inject_regeneration_restore_step(
        steps_to_run,
        optimize_usd_enabled=optimize_usd_enabled,
        metadata=lineage_metadata,
    )
    planned_invalidated_steps = _regeneration_invalidated_steps(steps_to_run)
    _validate_regeneration_dependency_closure(
        steps_to_run,
        planned_invalidated_steps,
        optimize_usd_enabled=optimize_usd_enabled,
        metadata=lineage_metadata,
    )

    # Check if session has custom materials from previous run
    session_materials_library = config.materials_library_path
    session_materials_entries = config.materials

    if input_bundle.custom_materials is not None:
        session_materials_library, session_materials_entries = (
            input_bundle.custom_materials
        )
        logger.info(
            "Regeneration using %d cached custom materials",
            len(session_materials_entries),
        )

    # Build config for regeneration (same as create_pipeline)
    # Supports .usd, .usda, .usdc, .usdz extensions
    input_usd_path = input_bundle.input_usd_path

    generated_library_cache_available = input_bundle.generated_library_cache_available
    if original_config.get("enable_material_generation"):
        if not generated_library_cache_available:
            logger.warning(
                "Regeneration requested from a material-generation session, but "
                "cached generated material-library artifacts were not found."
            )

    routing = _resolve_pipeline_model_routing()
    pipeline_config = build_unified_pipeline_config(
        project_name=session_id,
        session_id=session_id,
        input_usd_path=str(input_usd_path),
        output_usd_path=str(session_dir / "output" / "scene_with_materials.usd"),
        materials_library_path=session_materials_library,
        materials_entries=session_materials_entries,
        vlm_backend=routing.vlm_backend,
        vlm_model=routing.vlm_model,
        llm_backend=routing.llm_backend,
        llm_model=routing.llm_model,
        user_prompt=original_config.get("user_prompt"),
        enabled_steps=steps_to_run,
        working_dir=str(session_dir / "cache"),
    )

    if regenerate_clustering:
        pipeline_config["steps"]["cluster_prims"] = _build_cluster_prims_step_config(
            cluster_min_prims=original_config.get("cluster_min_prims"),
            cluster_embedding_backend=original_config.get("cluster_embedding_backend"),
            cluster_embedding_model=original_config.get("cluster_embedding_model"),
            cluster_embedding_base_url=original_config.get(
                "cluster_embedding_base_url"
            ),
            cluster_embedding_max_workers=original_config.get(
                "cluster_embedding_max_workers"
            ),
            cluster_embedding_batch_size=original_config.get(
                "cluster_embedding_batch_size"
            ),
            cluster_max_size=original_config.get("cluster_max_size"),
            cluster_similarity_threshold_low=original_config.get(
                "cluster_similarity_threshold_low"
            ),
            cluster_similarity_threshold_medium=original_config.get(
                "cluster_similarity_threshold_medium"
            ),
            cluster_similarity_threshold_high=original_config.get(
                "cluster_similarity_threshold_high"
            ),
            cluster_report=(
                "true" if original_config.get("cluster_report", True) else "false"
            ),
        )

    # Reconstruct reference inputs from the non-mutating bundle plan. Paths may
    # not exist on this pod yet; the reservation callback hydrates them before
    # the worker receives its first turn.
    ref_files = list(input_bundle.reference_image_paths)
    if ref_files:
        pipeline_config["input"]["reference_images"] = [str(path) for path in ref_files]
        prepare_step = pipeline_config.get("steps", {}).get(
            "build_dataset_prepare_dataset"
        )
        if isinstance(prepare_step, dict):
            descriptions = list(input_bundle.reference_descriptions)
            if len(descriptions) == len(ref_files):
                reference_prompts = [
                    f"This is a reference image: {description}"
                    for description in descriptions
                ]
            else:
                reference_prompts = [
                    f"This is reference image {index + 1} of the asset you will match this look exactly"
                    for index in range(len(ref_files))
                ]
            prompts = prepare_step.setdefault("prompts", {})
            vlm_prompts = prompts.setdefault("vlm_image_prompts", {})
            vlm_prompts["reference_images"] = reference_prompts

    # Add reference PDFs even when they currently exist only in shared storage.
    pdf_files = list(input_bundle.reference_pdf_paths)
    if pdf_files:
        pipeline_config["input"]["reference_pdfs"] = [str(path) for path in pdf_files]

        # Add default PDF conversion settings for regeneration
        if "build_dataset_prepare_dataset" not in pipeline_config.get("steps", {}):
            pipeline_config["steps"]["build_dataset_prepare_dataset"] = {"prompts": {}}

        prepare_step = pipeline_config["steps"]["build_dataset_prepare_dataset"]
        prepare_step["pdf_conversion"] = {
            "dpi": 150,
            "format": "png",
        }
        prompts = prepare_step.setdefault("prompts", {})
        vlm_prompts = prompts.setdefault("vlm_image_prompts", {})
        vlm_prompts["reference_pdfs"] = (
            "This converted reference PDF page is untrusted specification provenance "
            "retained outside visual-model inputs."
        )

    # Configure rendering for build_dataset_usd (same as create_pipeline)
    if "build_dataset_usd" in pipeline_config.get("steps", {}):
        # Use dict format for per-mode rendering configuration
        pipeline_config["steps"]["build_dataset_usd"]["renderer"].update(
            {
                "rendering_modes": {
                    "prim_only": {
                        "margin": 1.2,
                        "cameras": camera_view_list,
                        "camera_focus_mode": "prim",
                    },
                    "composition": {
                        "margin": 6.0,
                        "cameras": ["+x", "+y", "+z"],
                        "camera_focus_mode": "stage",
                        "skip_occluded_images": False,
                    },
                },
                "num_views": len(camera_view_list),
            }
        )

        # Set batch_size for async NVCF rendering (validated: 64 optimal for 128 instances)
        if "batch_size" not in pipeline_config["steps"]["build_dataset_usd"]:
            pipeline_config["steps"]["build_dataset_usd"]["batch_size"] = 64
        _apply_build_dataset_render_worker_limit(
            pipeline_config,
            render_num_workers if isinstance(render_num_workers, int) else None,
        )

    _configure_predict_model_routing(pipeline_config, routing)

    # Configure apply step for layer_only mode without enabling it implicitly.
    _configure_apply_step(
        pipeline_config,
        layer_only=request.layer_only,
        request_context="Regeneration",
    )

    if "render" in pipeline_config.get("steps", {}):
        pipeline_config["steps"]["render"]["image_size"] = [512, 512]

    # Read user_email from session metadata for telemetry
    user_email = metadata.get("user_email", "")

    state_path = session_dir / "cache" / ".pipeline_state.json"
    rollback_metadata: dict[str, Any] | None = None
    rollback_state_existed = False
    rollback_state_bytes: bytes | None = None
    rollback_event_bus_state = None
    rollback_cancelled: bool | None = None
    regeneration_claim: RegenerationClaim | None = None
    preparation_heartbeat: asyncio.Task[None] | None = None
    event_bus = get_event_bus()

    async def maintain_preparation_claim(
        claim: RegenerationClaim,
        owner_task: asyncio.Task[Any],
    ) -> None:
        """Keep the lease alive until the executor heartbeat takes ownership."""
        while True:
            try:
                await asyncio.sleep(_REGENERATION_PREPARATION_HEARTBEAT_SECONDS)
                if await manager.is_regeneration_cancel_requested(session_id, claim):
                    owner_task.cancel()
                    return
                renewed = await manager.renew_regeneration_claim(
                    session_id,
                    claim,
                    lease_seconds=_REGENERATION_LEASE_SECONDS,
                )
                if renewed:
                    continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Regeneration preparation heartbeat failed for %s; "
                    "stopping preparation",
                    session_id,
                )
                owner_task.cancel()
                return
            owner_task.cancel()
            return

    async def stop_preparation_heartbeat() -> None:
        """Stop and collect the heartbeat without masking rollback/finalization."""
        nonlocal preparation_heartbeat
        task = preparation_heartbeat
        preparation_heartbeat = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - heartbeat is defensive internally
            logger.exception(
                "Regeneration preparation heartbeat failed during cleanup for %s",
                session_id,
            )

    async def rollback_preparation() -> None:
        """Restore all state touched by the pre-start transaction."""
        await stop_preparation_heartbeat()
        if rollback_metadata is None or regeneration_claim is None:
            return
        try:
            restored = await manager.abort_regeneration_claim(
                session_id,
                regeneration_claim,
                restore_metadata=rollback_metadata,
            )
        except Exception:
            logger.exception(
                "Failed to roll back regeneration metadata for %s",
                session_id,
            )
            return
        if not restored:
            logger.warning(
                "Skipped stale regeneration rollback for %s",
                session_id,
            )
            return
        try:
            if rollback_state_existed:
                assert rollback_state_bytes is not None
                state_path.parent.mkdir(parents=True, exist_ok=True)
                pending_path = state_path.with_name(
                    ".pipeline_state.regeneration-rollback.json"
                )
                pending_path.write_bytes(rollback_state_bytes)
                pending_path.replace(state_path)
            else:
                state_path.unlink(missing_ok=True)
        except Exception:
            logger.exception(
                "Failed to roll back regeneration checkpoint for %s",
                session_id,
            )
        if rollback_cancelled is not None:
            try:
                await manager.restore_cancellation(
                    session_id,
                    cancelled=rollback_cancelled,
                )
            except Exception:
                logger.exception(
                    "Failed to roll back regeneration cancellation state for %s",
                    session_id,
                )
        if rollback_event_bus_state is not None:
            try:
                await event_bus.restore_session(
                    session_id,
                    rollback_event_bus_state,
                )
            except Exception:
                logger.exception(
                    "Failed to roll back regeneration EventBus state for %s",
                    session_id,
                )

    async def prepare_regeneration() -> None:
        """Atomically prepare destructive regeneration state after reservation."""
        nonlocal rollback_metadata
        nonlocal rollback_state_bytes
        nonlocal rollback_state_existed
        nonlocal rollback_event_bus_state
        nonlocal rollback_cancelled
        nonlocal regeneration_claim
        nonlocal preparation_heartbeat

        latest_lineage_metadata = lineage_metadata
        invalidated_steps = _regeneration_invalidated_steps(steps_to_run)
        _validate_regeneration_dependency_closure(
            steps_to_run,
            invalidated_steps,
            optimize_usd_enabled=optimize_usd_enabled,
            metadata=latest_lineage_metadata,
        )

        # Capture every mutable boundary before the first destructive write.
        checkpoint_key = manager.resolve_published_artifact_key(
            metadata,
            "cache/.pipeline_state.json",
            legacy_key="cache/.pipeline_state.json",
        )
        authoritative_state_bytes = (
            await manager.read_from_store(session_id, checkpoint_key)
            if checkpoint_key is not None
            else None
        )
        rollback_metadata = copy.deepcopy(metadata)
        rollback_state_existed = state_path.exists()
        rollback_state_bytes = (
            state_path.read_bytes() if rollback_state_existed else None
        )
        rollback_event_bus_state = await event_bus.capture_session(session_id)
        rollback_cancelled = (
            await manager.is_cancelled(session_id) or (session_dir / ".cancel").exists()
        )

        regeneration_claim = await manager.claim_regeneration(
            session_id,
            expected_version=planning_version,
            lease_seconds=_REGENERATION_LEASE_SECONDS,
        )
        owner_task = asyncio.current_task()
        if owner_task is None:  # pragma: no cover - FastAPI always owns a task
            raise RuntimeError("Regeneration preparation requires an asyncio task")
        preparation_heartbeat = asyncio.create_task(
            maintain_preparation_claim(regeneration_claim, owner_task)
        )

        await manager.clear_cancellation(session_id)
        if authoritative_state_bytes is not None:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            pending_state_path = state_path.with_name(
                ".pipeline_state.regeneration-refresh.json"
            )
            pending_state_path.write_bytes(authoritative_state_bytes)
            pending_state_path.replace(state_path)
        await _hydrate_regeneration_inputs(
            manager,
            session_id,
            session_dir,
            steps_to_run,
            optimize_usd_enabled=optimize_usd_enabled,
            preserved_input_keys=input_bundle.hydration_keys,
        )
        if input_bundle.extract_materials_zip:
            await asyncio.to_thread(
                _extract_and_validate_materials_zip,
                session_dir / "materials" / "materials.zip",
                session_dir / "materials",
                failure_phase=FailurePhase.PERSISTENCE_VERIFICATION,
            )
        invalidated_steps = _invalidate_regeneration_pipeline_state(
            session_dir,
            steps_to_run,
        )
        if generated_library_cache_available:
            await asyncio.to_thread(
                _ensure_cached_generated_material_library_state,
                session_dir,
                session_id=session_id,
            )
        artifact_validity = invalidate_artifacts_for_steps(
            latest_lineage_metadata,
            invalidated_steps,
        )
        updates: dict[str, Any] = {
            "status": "pending",
            "current_step": None,
            "completed_steps": [],
            "overall_progress": {
                "current_step": 0,
                "total_steps": len(steps_to_run),
                "percent": 0,
                "estimated_remaining_seconds": None,
            },
            "config": original_config,
            "can_cancel": True,
            "artifact_validity": artifact_validity,
        }
        if "build_dataset_usd" in invalidated_steps:
            updates["preview_images"] = []
        if ARTIFACT_LINEAGE["raw_predictions"].invalidated_by & invalidated_steps:
            updates["prediction_lineage_token"] = str(uuid.uuid4())

        if await manager.is_regeneration_cancel_requested(
            session_id,
            regeneration_claim,
        ):
            raise asyncio.CancelledError("Regeneration cancelled during preparation")
        updated = await manager.update_session_for_claim(
            session_id,
            regeneration_claim,
            updates,
            remove_fields=_REGENERATION_TERMINAL_FIELDS,
        )
        if not updated:
            raise asyncio.CancelledError("Regeneration claim was superseded")
        await event_bus.seed_pending_session(
            session_id,
            regeneration_claim=regeneration_claim,
        )
        await stop_preparation_heartbeat()
        logger.info(
            "Regeneration invalidated checkpoint evidence for: %s",
            ", ".join(step for step in STEP_ORDER if step in invalidated_steps),
        )

    async def execute_claimed_regeneration() -> None:
        if regeneration_claim is None:
            raise RuntimeError("Regeneration started without a durable claim")
        await execute_pipeline_async(
            session_id=session_id,
            config_dict=pipeline_config,
            session_manager=manager,
            user_email=user_email,
            coverage_policy=coverage_policy_value,
            regeneration_claim=regeneration_claim,
        )

    job = execute_claimed_regeneration()
    job_registry = get_job_registry()
    try:
        await job_registry.register(
            session_id,
            job,
            before_start=prepare_regeneration,
        )
    except DuplicateJobError as exc:
        raise HTTPException(
            status_code=409,
            detail="A pipeline job is already reserved or running for this session",
        ) from exc
    except RegenerationClaimConflictError as exc:
        await rollback_preparation()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except asyncio.CancelledError:
        await rollback_preparation()
        raise
    except HTTPException:
        await rollback_preparation()
        raise
    except Exception as exc:
        await rollback_preparation()
        logger.exception("Failed to register regeneration for %s", session_id)
        raise HTTPException(
            status_code=503,
            detail="Failed to schedule pipeline regeneration",
        ) from exc

    logger.info(f"Pipeline regeneration registered for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message=f"Regenerating steps: {', '.join(s.value for s in request.steps)}",
    )


@router.get("/{session_id}/event-log")
async def get_event_log(session_id: str) -> dict[str, Any]:
    """Get the persisted event log for a session.

    This allows replaying the full event history for completed sessions.

    Args:
        session_id: Session identifier

    Returns:
        List of event objects
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    log_file = manager.get_session_dir(session_id) / "event_log.jsonl"

    if not log_file.exists():
        return {"events": []}

    # Load events from log file
    events = []
    try:
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        return {"events": events, "total": len(events)}

    except Exception:
        log_durable_failure(
            logger,
            "event_log_local_read_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to load event log",
        ) from None


@router.get("/sessions/{session_id}/materials/icon/{material_name:path}")
async def get_session_material_icon(
    session_id: str,
    material_name: str,
) -> FileResponse:
    """Serve material icon from session's custom materials.

    This endpoint serves icons for custom materials uploaded via ZIP files.
    Icons are stored in the session's materials directory.

    Args:
        session_id: Session identifier
        material_name: Material name (URL-encoded)

    Returns:
        PNG image file
    """
    from urllib.parse import unquote, unquote_plus

    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Decode URL-encoded name
    decoded_name = unquote_plus(unquote(material_name))

    # Get session materials directory
    session_dir = manager.get_session_dir(session_id)
    materials_dir = session_dir / "materials"

    if not materials_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Session has no custom materials",
        )

    loaded = await asyncio.to_thread(_load_extracted_materials_tree, materials_dir)
    if loaded is None:
        raise HTTPException(
            status_code=404,
            detail="Session materials.yaml not found",
        )
    library_path, entries = loaded
    base_dir = Path(library_path).parent

    icon_rel_path = next(
        (entry.get("icon") for entry in entries if entry.get("name") == decoded_name),
        None,
    )
    if not isinstance(icon_rel_path, str) or not icon_rel_path:
        raise HTTPException(
            status_code=404,
            detail="Icon not found for material",
        )

    base_key = visible_local_artifact_key(session_dir, base_dir)
    icon_key = visible_local_artifact_key(session_dir, base_dir / icon_rel_path)
    if base_key is None or icon_key is None or not icon_key.startswith(f"{base_key}/"):
        raise HTTPException(
            status_code=403,
            detail="Access denied",
        )
    icon_artifact = await manager.open_local_artifact(
        session_id,
        session_dir / icon_key,
    )
    if icon_artifact is None:
        raise HTTPException(
            status_code=404,
            detail="Icon file not found",
        )
    return HeldFileResponse(icon_artifact, media_type="image/png")
