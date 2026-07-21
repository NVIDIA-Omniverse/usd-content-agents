# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune API endpoints — Physics Agent tuning over an authored physics USD.

Mirrors the /pipeline router's lifecycle (create → status → events → cancel
→ results → artifact downloads) but skips the upload-USD pre-step: tune
sessions can either upload a USD inline, reference an S3 URI, or chain off
a completed pipeline session via ``source_session_id``.

Reuses the same SessionManager / JobRegistry / EventBus infrastructure that
/pipeline uses — no new persistence layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from physics_agent.tuning.visual_evidence import (
    DEFAULT_JUDGE_GENERATED_FRAMES,
    DEFAULT_JUDGE_REFERENCE_FRAMES,
    DEFAULT_REFERENCE_VIDEO_FRAMES,
    validate_visual_frame_count,
)
from sse_starlette import EventSourceResponse
from world_understanding.functions.physics import ovphysx_runtime_available
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)
from world_understanding.utils.held_file_response import HeldFileResponse
from world_understanding.utils.s3_utils import (
    S3BucketNotAllowedError,
    authorize_s3_uri_for_extensions,
    download_file_from_s3,
)

from ..artifact_contract import (
    TUNE_ARTIFACT_SPECS,
    artifact_name_from_key,
    available_artifact_keys,
)
from ..config import config
from ..config_persistence import avalidate_durable_request_content
from ..models.responses import (
    S3_INPUT_ERROR_RESPONSES,
    PipelineError,
    SessionCreated,
    TuneResults,
    TuneStatus,
)
from ..runtime import get_event_bus, get_job_registry
from ..runtime.events import StepState
from ..session.manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tune", tags=["tune"])

session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    if session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return session_manager


def set_session_manager(manager: SessionManager) -> None:
    global session_manager
    session_manager = manager


def _validate_ovphysx_runtime_for_request(engine: str) -> None:
    """Reject OvPhysX work when the provisioned daemon runtime is unavailable."""

    if engine == "ovphysx" and not ovphysx_runtime_available():
        raise HTTPException(
            status_code=503,
            detail="OvPhysX runtime is not available in this deployment",
        )


_VALID_USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
_VALID_REFERENCE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VALID_REFERENCE_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".webm",
    ".avi",
    ".mkv",
}

# Hard limits on tune-creation inputs. The optimizer trial budget is the
# primary cost driver for a tune session — capping it at the request layer
# prevents an accidentally-huge ``max_trials`` from queueing a multi-hour
# job without operator review. The scenario-YAML cap is a generous DoS
# guard against pathological-sized form payloads.
_MAX_TUNE_TRIALS = 1000
_MAX_SCENARIO_YAML_BYTES = 64 * 1024  # 64KB — drop_settle is < 1KB.
# user_prompt is bounded so a malicious caller cannot pin LLM cost or
# memory by sending a megabyte of text. 16KB is generous for any NL prompt
# we expect to see in practice.
_MAX_USER_PROMPT_BYTES = 16 * 1024
_MAX_REFERENCE_UPLOADS = 16
_MAX_REFERENCE_DESCRIPTIONS_BYTES = 16 * 1024
_MAX_REFERENCE_DESCRIPTION_BYTES = 2 * 1024


async def _tune_download_urls(
    manager: SessionManager,
    session_id: str,
    metadata: dict,
) -> dict[str, str]:
    available = await available_artifact_keys(manager, session_id, metadata, "tune")
    urls = {
        spec.logical_name: (
            f"/tune/{session_id}/artifacts/{artifact_name_from_key(spec.key, 'tune')}"
        )
        for spec in TUNE_ARTIFACT_SPECS
        if spec.key in available
    }
    legacy_tuned_usd = "tune/tuned_physics.usda"
    if "tuned_usd" not in urls and legacy_tuned_usd in available:
        urls["tuned_usd"] = f"/tune/{session_id}/artifacts/tuned_physics.usda"
    return urls


async def _stream_copy(
    upload: UploadFile,
    dest: Path,
    chunk_size: int = 2 * 1024 * 1024,
    *,
    max_bytes: int | None = None,
    too_large_detail: str | None = None,
) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with dest.open("wb") as f:
            while True:
                data = await upload.read(chunk_size)
                if not data:
                    break
                if max_bytes is not None and total + len(data) > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=too_large_detail or "Uploaded file is too large",
                    )
                f.write(data)
                total += len(data)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return total


def _parse_reference_descriptions(raw: str, field_name: str) -> list[str] | None:
    """Parse optional JSON-array descriptions from multipart form fields."""
    raw_text = (raw or "").strip()
    if not raw_text:
        return None
    if len(raw_text.encode("utf-8")) > _MAX_REFERENCE_DESCRIPTIONS_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{field_name} exceeds "
                f"{_MAX_REFERENCE_DESCRIPTIONS_BYTES // 1024} KB size limit"
            ),
        )
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a JSON array of strings: {exc}",
        )
    if not isinstance(parsed, list) or not all(isinstance(v, str) for v in parsed):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a JSON array of strings",
        )
    for idx, value in enumerate(parsed, 1):
        if len(value.encode("utf-8")) > _MAX_REFERENCE_DESCRIPTION_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"{field_name}[{idx}] exceeds "
                    f"{_MAX_REFERENCE_DESCRIPTION_BYTES // 1024} KB size limit"
                ),
            )
    return list(parsed)


def _nonempty_uploads(uploads: list[UploadFile] | None) -> list[UploadFile]:
    """Drop empty file parts some multipart clients send for omitted lists."""
    if not uploads:
        return []
    return [upload for upload in uploads if upload and upload.filename]


async def _copy_reference_uploads(
    *,
    uploads: list[UploadFile],
    session_dir: Path,
    subdir: str,
    file_prefix: str,
    valid_extensions: set[str],
    label: str,
    current_batch_bytes: int = 0,
    max_batch_bytes: int | None = None,
) -> tuple[list[Path], int]:
    """Validate and copy reference media uploads into the session input dir."""
    copied: list[Path] = []
    batch_bytes = current_batch_bytes
    for idx, upload in enumerate(_nonempty_uploads(uploads), 1):
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in valid_extensions:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid {label} file type: {ext}. "
                    f"Allowed: {', '.join(sorted(valid_extensions))}"
                ),
            )
        dest = session_dir / "input" / subdir / f"{file_prefix}_{idx:02d}{ext}"
        per_file_limit = config.max_upload_size_mb * 1024 * 1024
        remaining_batch_bytes = (
            max_batch_bytes - batch_bytes if max_batch_bytes is not None else None
        )
        if remaining_batch_bytes is not None and remaining_batch_bytes <= 0:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Reference media batch too large. Max total: "
                    f"{config.max_upload_size_mb}MB"
                ),
            )
        if remaining_batch_bytes is not None:
            copy_limit = min(per_file_limit, remaining_batch_bytes)
            too_large_detail = (
                "Reference media batch too large. Max total: "
                f"{config.max_upload_size_mb}MB"
                if remaining_batch_bytes < per_file_limit
                else f"{label} file too large. Max: {config.max_upload_size_mb}MB"
            )
        else:
            copy_limit = per_file_limit
            too_large_detail = (
                f"{label} file too large. Max: {config.max_upload_size_mb}MB"
            )
        total = await _stream_copy(
            upload,
            dest,
            max_bytes=copy_limit,
            too_large_detail=too_large_detail,
        )
        copied.append(dest)
        batch_bytes += total
    return copied, batch_bytes


def _coerce_finite_score(value: object) -> float | None:
    """Round 12 (CX P2#2): coerce a stored best_score to a JSON-serialisable
    float or ``None``.

    Starlette's JSON encoder rejects ``inf`` / ``-inf`` / ``nan``; a
    cancelled-before-first-trial run persists ``best_score == inf``,
    and the legacy ``float("nan")`` fallback used in
    :func:`get_tune_results` would also explode. Treat any non-finite
    or non-numeric input as "no best score yet" so clients get a clean
    JSON null in the terminal response.
    """
    import math

    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _validate_and_authorize_s3_usd_uri(s3_uri: str) -> str:
    """Validate and authorize a client S3 USD URI without performing I/O."""
    try:
        return authorize_s3_uri_for_extensions(
            s3_uri,
            config.s3_allowed_buckets,
            allowed_extensions=_VALID_USD_EXTENSIONS,
        )
    except S3BucketNotAllowedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _download_s3_to_session(s3_uri: str, session_dir: Path) -> Path:
    """Reauthorize and download a client-supplied S3 USD into session input."""
    ext = _validate_and_authorize_s3_usd_uri(s3_uri)

    local_path = session_dir / "input" / f"physics{ext}"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_file_from_s3(s3_uri, local_path)
    except FileNotFoundError:
        log_durable_failure(
            logger,
            "tune_s3_object_not_found",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=False,
        )
        raise HTTPException(status_code=404, detail="S3 object not found") from None
    except PermissionError:
        log_durable_failure(
            logger,
            "tune_s3_access_denied",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=403, detail="Access denied to S3 object"
        ) from None
    except Exception:
        log_durable_failure(
            logger,
            "tune_s3_download_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )
        raise HTTPException(status_code=502, detail="S3 download failed") from None
    size_mb = local_path.stat().st_size / (1024 * 1024)
    if size_mb > config.max_upload_size_mb:
        local_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"S3 file too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
        )
    return local_path


async def _copy_from_source_session(
    manager: SessionManager,
    source_session_id: str,
    target_session_dir: Path,
) -> Path:
    """Copy the apply_physics output USD from a completed pipeline session."""
    if not await manager.session_exists(source_session_id):
        raise HTTPException(
            status_code=404,
            detail=f"source_session_id not found: {source_session_id}",
        )
    src_path = await manager.get_artifact_path(source_session_id, "output_usd")
    if src_path is None:
        # Try pulling cache/physics from the store (cross-instance case).
        await manager.sync_from_store(source_session_id, prefix="cache/physics/")
        src_path = await manager.get_artifact_path(source_session_id, "output_usd")
    if src_path is None or not src_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"source_session_id {source_session_id} has no apply_physics "
                "output_usd; run the pipeline to completion first."
            ),
        )
    dest = target_session_dir / "input" / "physics.usda"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dest)
    return dest


def _find_input_physics(session_dir: Path) -> Path | None:
    input_dir = session_dir / "input"
    for ext in (".usd", ".usda", ".usdc", ".usdz"):
        candidate = input_dir / f"physics{ext}"
        if candidate.exists():
            return candidate
    return None


def _scenario_param_names_from_mapping(scenario_data: dict[str, object]) -> set[str]:
    raw_params = scenario_data.get("parameters")
    if not isinstance(raw_params, list):
        return set()

    names: set[str] = set()
    for raw_param in raw_params:
        if not isinstance(raw_param, dict):
            continue
        name = raw_param.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def _validate_engine_supports_param_names_for_request(
    engine: str,
    param_names: set[str],
) -> None:
    if not param_names:
        return
    try:
        from physics_agent.tuning.backend import validate_engine_supports_param_names
        from physics_agent.tuning.errors import TuningError

        validate_engine_supports_param_names(engine, param_names)
    except TuningError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Engine/parameter unsupported: {e}",
        ) from e


def _validate_engine_name_for_request(engine: str) -> None:
    from physics_agent.tuning.backend import SUPPORTED_ENGINES

    if engine not in SUPPORTED_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown engine {engine!r}. Supported: {sorted(SUPPORTED_ENGINES)}",
        )


@router.post(
    "",
    response_model=SessionCreated,
    status_code=202,
    responses=S3_INPUT_ERROR_RESPONSES,
)
async def create_tune(
    physics_usd: UploadFile = File(
        None,
        description="Physics-authored USD (output of apply_physics) to tune",
    ),
    s3_uri: str = Form(
        None,
        description="S3 URI to a physics-authored USD",
    ),
    source_session_id: str = Form(
        None,
        description="Pipeline session ID — copy its apply_physics output_usd",
    ),
    reference_images: list[UploadFile] = File(
        default=[],
        description="Optional reference images for the visual/VLM judge",
    ),
    reference_videos: list[UploadFile] = File(
        default=[],
        description="Optional reference videos for the visual/VLM judge",
    ),
    reference_descriptions: str = Form(
        default="",
        description=(
            "Optional JSON array of descriptions parallel to reference_images"
        ),
    ),
    reference_video_descriptions: str = Form(
        default="",
        description=(
            "Optional JSON array of descriptions parallel to reference_videos"
        ),
    ),
    reference_video_frames: int = Form(
        default=DEFAULT_REFERENCE_VIDEO_FRAMES,
        description="Frames to extract from each reference video for visual judging",
    ),
    judge_reference_frames: int = Form(
        default=DEFAULT_JUDGE_REFERENCE_FRAMES,
        description="Max reference images/video frames to send to the VLM judge",
    ),
    judge_generated_frames: int = Form(
        default=DEFAULT_JUDGE_GENERATED_FRAMES,
        description="Max generated render frames to send to the VLM judge",
    ),
    scenario_yaml: str = Form(
        default="",
        description=(
            "Scenario YAML body (drop_settle etc.). Optional when "
            "`user_prompt` is supplied — the NL interpreter authors a "
            "Scenario in that case. When both are supplied, explicit "
            "YAML fields override interpreter output."
        ),
    ),
    user_prompt: str = Form(
        default="",
        description=(
            "Free-form NL description of the desired tune run "
            "(e.g. 'make this object bouncy'). Persisted to "
            "tune_results.json['user_prompt'] and rendered into report.md "
            "for audit. Mirrors the user_prompt field on /pipeline."
        ),
    ),
    optimizer: str = Form(
        default="auto",
        description="auto (=botorch), botorch, random, cma-es",
    ),
    engine: str = Form(
        default="ovphysx",
        description=(
            "Tuning engine: ovphysx (PhysX 5 daemon, production), "
            "newton (NVIDIA Newton GPU/MuJoCo-warp; requires the "
            "apps/physics_agent[newton] extra; supports contact_ke/contact_kd "
            "bounce tuning; no static_friction or restitution tuning yet), "
            "or fake (tests)."
        ),
    ),
    max_trials: int = Form(default=30, description="Optimizer trial budget"),
    seed: int = Form(default=42, description="Seed for optimizer + backend"),
    enable_judge: bool = Form(
        default=True,
        description=(
            "Run the VLM-as-judge over scenario/history/best_params at "
            "the end of tune (default on). Set to false for byte-identical "
            "output to the pre-Part-1.1 baseline (no model calls)."
        ),
    ),
    judge_max_iterations: int = Form(
        default=3,
        description="Hard cap on refine-loop iterations when judge returns 'continue'.",
    ),
    judge_max_tokens: int | None = Form(
        default=None,
        description=(
            "Optional max output tokens for judge responses. "
            "Defaults to the physics judge configuration."
        ),
    ),
    judge_temperature: float | None = Form(
        default=None,
        description=(
            "Optional temperature for judge calls. Defaults to "
            "the scenario judge block or physics judge configuration."
        ),
    ),
) -> SessionCreated:
    """Create a tuning session and queue it for background execution."""
    sources_set = sum(1 for s in (physics_usd, s3_uri, source_session_id) if s)
    if sources_set != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly one of physics_usd, s3_uri, or source_session_id "
                "must be provided"
            ),
        )

    # Reject obviously-bad form values before any session/storage work.
    if not (1 <= max_trials <= _MAX_TUNE_TRIALS):
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_trials must be between 1 and {_MAX_TUNE_TRIALS}, "
                f"got {max_trials}."
            ),
        )
    _validate_engine_name_for_request(engine)
    _validate_ovphysx_runtime_for_request(engine)
    if len(scenario_yaml.encode("utf-8")) > _MAX_SCENARIO_YAML_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"scenario_yaml exceeds {_MAX_SCENARIO_YAML_BYTES // 1024} KB "
                "size limit"
            ),
        )
    user_prompt_text = (user_prompt or "").strip()
    if len(user_prompt_text.encode("utf-8")) > _MAX_USER_PROMPT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"user_prompt exceeds {_MAX_USER_PROMPT_BYTES // 1024} KB size limit"
            ),
        )
    scenario_yaml_text = scenario_yaml or ""
    has_scenario = bool(scenario_yaml_text.strip())
    if not has_scenario and not user_prompt_text:
        raise HTTPException(
            status_code=400,
            detail=(
                "Either scenario_yaml or user_prompt must be supplied (both are empty)."
            ),
        )
    canonical_yaml = await avalidate_durable_request_content(
        {
            "user_prompt": user_prompt_text or None,
            "reference_descriptions": reference_descriptions,
            "reference_video_descriptions": reference_video_descriptions,
            "s3_uri": s3_uri,
        },
        yaml_documents={"scenario_yaml": scenario_yaml_text},
        context="physics tune durable request content",
    )
    scenario_yaml_text = canonical_yaml["scenario_yaml"]
    if judge_max_iterations < 1 or judge_max_iterations > 10:
        raise HTTPException(
            status_code=400,
            detail=(
                f"judge_max_iterations must be between 1 and 10, "
                f"got {judge_max_iterations}."
            ),
        )
    if judge_max_tokens is not None and judge_max_tokens < 1:
        raise HTTPException(
            status_code=400,
            detail=f"judge_max_tokens must be >= 1, got {judge_max_tokens}.",
        )
    if judge_temperature is not None and (
        not math.isfinite(judge_temperature) or judge_temperature < 0.0
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"judge_temperature must be finite and >= 0, got {judge_temperature}."
            ),
        )
    try:
        reference_video_frames = validate_visual_frame_count(
            "reference_video_frames",
            reference_video_frames,
        )
        judge_reference_frames = validate_visual_frame_count(
            "judge_reference_frames",
            judge_reference_frames,
        )
        judge_generated_frames = validate_visual_frame_count(
            "judge_generated_frames",
            judge_generated_frames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    reference_image_uploads = _nonempty_uploads(reference_images)
    reference_video_uploads = _nonempty_uploads(reference_videos)
    reference_upload_count = len(reference_image_uploads) + len(reference_video_uploads)
    if reference_upload_count > _MAX_REFERENCE_UPLOADS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many reference media files. Max total: {_MAX_REFERENCE_UPLOADS}"
            ),
        )
    parsed_reference_descriptions = _parse_reference_descriptions(
        reference_descriptions,
        "reference_descriptions",
    )
    parsed_reference_video_descriptions = _parse_reference_descriptions(
        reference_video_descriptions,
        "reference_video_descriptions",
    )
    if parsed_reference_descriptions is not None and len(
        parsed_reference_descriptions
    ) != len(reference_image_uploads):
        raise HTTPException(
            status_code=400,
            detail=(
                "reference_descriptions must have one item per reference image "
                f"({len(reference_image_uploads)} expected)"
            ),
        )
    if parsed_reference_video_descriptions is not None and len(
        parsed_reference_video_descriptions
    ) != len(reference_video_uploads):
        raise HTTPException(
            status_code=400,
            detail=(
                "reference_video_descriptions must have one item per reference "
                f"video ({len(reference_video_uploads)} expected)"
            ),
        )
    if source_session_id is not None and source_session_id.strip():
        # Pre-validate the UUID shape so a malicious caller cannot smuggle
        # path-like values through to SessionManager and trigger a 500 from
        # InvalidSessionIdError.
        # Treat empty / whitespace-only strings as absent — multipart
        # clients commonly serialise unset Form fields as ``""`` instead
        # of omitting them entirely, and ``sources_set`` already treats
        # blank as "not provided" upstream. Without this guard the blank
        # value here would short-circuit to a 400 even though no source
        # session was actually requested.
        from ..session.manager import _SESSION_ID_PATTERN

        if not _SESSION_ID_PATTERN.fullmatch(source_session_id):
            raise HTTPException(
                status_code=400,
                detail=(
                    "source_session_id must be a UUID4-shaped string; "
                    f"got {source_session_id!r}"
                ),
            )

    # Validate scenario YAML eagerly so a malformed scenario fails the create
    # call rather than the background job. Skipped when only user_prompt is
    # supplied — the NL interpreter authors the Scenario inside the worker.
    #
    # Round 12 (CX P2#1): when BOTH ``scenario_yaml`` and ``user_prompt`` are
    # supplied, the runner treats the YAML as an *override* that the NL
    # interpreter merges (explicit fields win on every conflict). A partial
    # override (e.g. just ``parameters: ...`` to lock the search bounds) is
    # legitimate, so we only enforce the full ``load_scenario`` schema here
    # for YAML-only submissions. With ``user_prompt`` present we do the
    # cheap YAML-shape validation (parses to a mapping, has a known
    # ``name``) so we still gate engine/scenario capability up-front, but we
    # leave the full ``parameters`` / ``target`` requirements to the
    # interpreter+merge inside the worker.
    if has_scenario:
        try:
            scenario_data = yaml.safe_load(scenario_yaml_text)
        except yaml.YAMLError:
            raise HTTPException(
                status_code=400,
                detail="Invalid scenario YAML",
            ) from None
        if not isinstance(scenario_data, dict):
            raise HTTPException(
                status_code=400,
                detail="scenario_yaml must parse to a mapping",
            )
        scenario_param_names = _scenario_param_names_from_mapping(scenario_data)
        # Resolve the scenario name for the capability check.
        scenario_name_value = scenario_data.get("name")
        if user_prompt_text:
            # Override mode — only enforce that ``name`` (when present)
            # is a known scenario kind. Skip ``load_scenario``'s full
            # required-fields gate so a partial override does not 400.
            from physics_agent.tuning.types import SUPPORTED_SCENARIOS

            if scenario_name_value is not None and (
                not isinstance(scenario_name_value, str)
                or scenario_name_value not in SUPPORTED_SCENARIOS
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid scenario name {scenario_name_value!r}; "
                        f"supported: {sorted(SUPPORTED_SCENARIOS)}"
                    ),
                )
        else:
            try:
                # Local import — keeps physics_agent_service from depending on
                # tuning at module import time.
                from physics_agent.tuning.scenario import load_scenario

                parsed = load_scenario(scenario_data)
                scenario_name_value = parsed.name
                scenario_param_names = {param.name for param in parsed.params}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid scenario: {e}")
        # Codex round 7: refuse known-unrunnable engine/scenario pairs at
        # submission time so a doomed background job never queues. Mirrors
        # the runner-side capability map in physics_agent.tuning.runner.
        if isinstance(scenario_name_value, str):
            try:
                from physics_agent.tuning.errors import TuningError
                from physics_agent.tuning.runner import (
                    _validate_engine_supports_scenario,
                )

                # Capability map now lives in
                # ``physics_agent.tuning.scenarios.SUPPORTED_SCENARIOS_PER_ENGINE``;
                # the runner helper above reads from there. Same call signature.
                _validate_engine_supports_scenario(engine, scenario_name_value)
            except (ValueError, TuningError) as e:
                # Codex round 8: ``_validate_engine_supports_scenario`` raises
                # ``TuningError`` (not ``ValueError``) on capability mismatch;
                # without catching both the freeform+ovphysx submission would
                # fall through to a 500 instead of the intended 400.
                raise HTTPException(
                    status_code=400,
                    detail=f"Engine/scenario unsupported: {e}",
                )
        _validate_engine_supports_param_names_for_request(
            engine,
            scenario_param_names,
        )

    if s3_uri:
        _validate_and_authorize_s3_usd_uri(s3_uri)

    manager = get_session_manager()
    session_id = str(uuid.uuid4())
    session_dir = await manager.create_session(session_id)

    try:
        if physics_usd:
            ext = (
                Path(physics_usd.filename).suffix.lower()
                if physics_usd.filename
                else ".usd"
            )
            if ext not in _VALID_USD_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid USD file type: {ext}. "
                        f"Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}"
                    ),
                )
            usd_path = session_dir / "input" / f"physics{ext}"
            total = await _stream_copy(physics_usd, usd_path)
            size_mb = total / (1024 * 1024)
            if size_mb > config.max_upload_size_mb:
                usd_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: {size_mb:.1f}MB",
                )
        elif s3_uri:
            # ``download_file_from_s3`` performs synchronous network I/O
            # via boto3; calling it directly inside the async route would
            # block the FastAPI event loop and stall every other in-flight
            # request for the duration of the download. Push it onto the
            # default thread executor so the loop stays responsive.
            await asyncio.to_thread(_download_s3_to_session, s3_uri, session_dir)
        else:
            await _copy_from_source_session(manager, source_session_id, session_dir)
    except HTTPException:
        await manager.delete_session(session_id)
        raise
    except Exception:
        log_durable_failure(
            logger,
            "tune_input_provision_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )
        await manager.delete_session(session_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to provision input physics USD",
        ) from None

    input_physics = _find_input_physics(session_dir)
    if not input_physics:
        await manager.delete_session(session_id)
        raise HTTPException(
            status_code=400, detail="Failed to provision input physics USD"
        )

    try:
        max_reference_batch_bytes = config.max_upload_size_mb * 1024 * 1024
        reference_image_paths, reference_batch_bytes = await _copy_reference_uploads(
            uploads=reference_image_uploads,
            session_dir=session_dir,
            subdir="reference_images",
            file_prefix="reference_image",
            valid_extensions=_VALID_REFERENCE_IMAGE_EXTENSIONS,
            label="reference image",
            max_batch_bytes=max_reference_batch_bytes,
        )
        reference_video_paths, _reference_batch_bytes = await _copy_reference_uploads(
            uploads=reference_video_uploads,
            session_dir=session_dir,
            subdir="reference_videos",
            file_prefix="reference_video",
            valid_extensions=_VALID_REFERENCE_VIDEO_EXTENSIONS,
            label="reference video",
            current_batch_bytes=reference_batch_bytes,
            max_batch_bytes=max_reference_batch_bytes,
        )
    except HTTPException:
        await manager.delete_session(session_id)
        raise
    except Exception as e:
        await manager.delete_session(session_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to copy reference media: {type(e).__name__}",
        ) from e

    # Persist scenario YAML and user_prompt to the session input dir so
    # cancellation / debugging can reproduce the run. Both are optional now;
    # at least one is guaranteed to be present by the validation above.
    scenario_path: Path | None = None
    if has_scenario:
        scenario_path = session_dir / "input" / "scenario.yaml"
        scenario_path.write_text(scenario_yaml_text, encoding="utf-8")
    user_prompt_path: Path | None = None
    if user_prompt_text:
        user_prompt_path = session_dir / "input" / "user_prompt.txt"
        user_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        user_prompt_path.write_text(user_prompt_text, encoding="utf-8")

    await manager.update_session(
        session_id,
        {
            "status": "pending",
            "kind": "tune",
            "can_cancel": True,
            "config": {
                "kind": "tune",
                "engine": engine,
                "optimizer": optimizer,
                "max_trials": max_trials,
                "seed": seed,
                "physics_usd": str(input_physics),
                "scenario_path": str(scenario_path) if scenario_path else None,
                "user_prompt": user_prompt_text or None,
                "user_prompt_path": (
                    str(user_prompt_path) if user_prompt_path else None
                ),
                "reference_images": [str(p) for p in reference_image_paths],
                "reference_videos": [str(p) for p in reference_video_paths],
                "reference_descriptions": parsed_reference_descriptions,
                "reference_video_descriptions": (parsed_reference_video_descriptions),
                "reference_video_frames": reference_video_frames,
                "judge_reference_frames": judge_reference_frames,
                "judge_generated_frames": judge_generated_frames,
                "enable_judge": enable_judge,
                "judge_max_iterations": judge_max_iterations,
                "judge_max_tokens": judge_max_tokens,
                "judge_temperature": judge_temperature,
                "source_session_id": source_session_id,
                "s3_uri": s3_uri,
            },
        },
    )

    job_registry = get_job_registry()
    # Local import keeps the runtime dependency lazy — same pattern as
    # /pipeline (executor lives in workers/, imported only when registering).
    from ..workers.tune_executor import execute_tune_async

    await job_registry.register(
        session_id,
        execute_tune_async(
            session_id=session_id,
            session_manager=manager,
            scenario_path=scenario_path,
            user_prompt=user_prompt_text or None,
            physics_usd=input_physics,
            reference_images=reference_image_paths,
            reference_videos=reference_video_paths,
            reference_descriptions=parsed_reference_descriptions,
            reference_video_descriptions=parsed_reference_video_descriptions,
            reference_video_frames=reference_video_frames,
            judge_reference_frames=judge_reference_frames,
            judge_generated_frames=judge_generated_frames,
            engine=engine,
            optimizer=optimizer,
            max_trials=max_trials,
            seed=seed,
            enable_judge=enable_judge,
            judge_max_iterations=judge_max_iterations,
            judge_max_tokens=judge_max_tokens,
            judge_temperature=judge_temperature,
        ),
    )

    logger.info(f"Tune queued for session {session_id}")
    return SessionCreated(
        session_id=session_id,
        status="pending",
        message="Tune queued for execution",
        estimated_duration_minutes=5,
    )


@router.get("/{session_id}/status", response_model=TuneStatus)
async def get_tune_status(session_id: str) -> TuneStatus:
    """Tune session status — durable metadata from the SessionManager,
    overlayed with live trial-level progress from the in-memory EventBus
    snapshot when present.

    The shared :mod:`runtime.bus.EventBus` was designed for the pipeline's
    fixed multi-step model; it does not natively understand tune trial
    counters, so we read those out of the snapshot's ``current_step.progress``
    payload that :class:`_TuneEventListener` writes on every trial. Status
    is always taken from the durable manager metadata so a completed tune
    cannot be reported as ``running`` due to a stale in-memory snapshot.
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    created_at = datetime.fromisoformat(metadata["created_at"])
    now = datetime.now(UTC)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    elapsed = int((now - created_at).total_seconds())

    snapshot = event_bus.get_snapshot(session_id) or {}
    n_trials = 0
    max_trials = 0
    best_score: float | None = None
    best_params: dict[str, float] | None = None

    # Trial counters live on the live snapshot for in-flight runs.
    current_step = snapshot.get("current_step") or {}
    if current_step:
        progress = current_step.get("progress") or {}
        n_trials = int(progress.get("current") or 0)
        max_trials = int(progress.get("total") or 0)

    # When the run finishes, results are persisted on the durable session.
    results = metadata.get("results") or {}
    if "n_trials" in results:
        n_trials = max(n_trials, int(results.get("n_trials") or 0))
    # Round 14 (Codex CX P2#1): a tune cancelled before any trial
    # completes persists ``best_score = inf``; the results endpoint
    # already runs every read through ``_coerce_finite_score`` so the
    # JSON encoder never sees a non-finite float. The status endpoint
    # used to copy ``best_score`` straight through, which made
    # ``GET /tune/{id}/status`` return 500 on zero-trial cancellations.
    # Pass the raw value through the same coercion helper so the
    # symmetry holds and the status response stays JSON-serialisable.
    best_score = _coerce_finite_score(results.get("best_score", best_score))
    best_params = results.get("best_params", best_params)

    config = metadata.get("config") or {}
    if not max_trials:
        max_trials = int(config.get("max_trials") or 0)

    return TuneStatus(
        session_id=session_id,
        status=metadata["status"],
        n_trials=n_trials,
        max_trials=max_trials,
        best_score=best_score,
        best_params=best_params,
        elapsed_seconds=elapsed,
        can_cancel=metadata.get("status") in ("pending", "running"),
        error_message=metadata.get("error")
        if metadata.get("status") == "failed"
        else None,
        created_at=metadata["created_at"],
        updated_at=metadata["updated_at"],
    )


@router.get("/{session_id}/results", response_model=TuneResults | PipelineError)
async def get_tune_results(session_id: str):
    manager = get_session_manager()
    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    status = metadata["status"]
    if status == "completed":
        return TuneResults(
            session_id=session_id,
            status=status,
            best_params=metadata.get("results", {}).get("best_params", {}),
            best_score=_coerce_finite_score(
                metadata.get("results", {}).get("best_score")
            ),
            n_trials=metadata.get("results", {}).get("n_trials", 0),
            optimizer_used=metadata.get("results", {}).get("optimizer_used", ""),
            engine_used=metadata.get("results", {}).get("engine_used", ""),
            download_urls=await _tune_download_urls(manager, session_id, metadata),
            duration_seconds=metadata.get("duration_seconds", 0),
            completed_at=metadata.get("completed_at", ""),
        )
    if status == "failed":
        results = metadata.get("results") or {}
        if results:
            return TuneResults(
                session_id=session_id,
                status=status,
                best_params=results.get("best_params", {}),
                best_score=_coerce_finite_score(results.get("best_score")),
                n_trials=results.get("n_trials", 0),
                optimizer_used=results.get("optimizer_used", ""),
                engine_used=results.get("engine_used", ""),
                download_urls=await _tune_download_urls(manager, session_id, metadata),
                duration_seconds=metadata.get("duration_seconds", 0),
                completed_at=metadata.get("completed_at", ""),
                error_message=metadata.get("error", "Unknown error"),
            )
        return PipelineError(
            session_id=session_id,
            status=status,
            error_message=metadata.get("error", "Unknown error"),
            failed_step="tune",
            completed_steps=[],
            partial_results=metadata.get("partial_results"),
        )
    if status == "cancelled":
        # Cancelled is a terminal state. The executor stores partial
        # best_params + history.jsonl + report.md when cancellation
        # lands after at least one trial — surface them through the
        # same TuneResults shape so clients have a download path.
        # CX Round 12 P2#2: ``best_score`` is coerced to ``None`` when
        # absent or non-finite so Starlette's JSON encoder doesn't reject
        # ``inf`` / ``nan`` and 500 the response.
        results = metadata.get("results") or {}
        return TuneResults(
            session_id=session_id,
            status=status,
            best_params=results.get("best_params", {}),
            best_score=_coerce_finite_score(results.get("best_score")),
            n_trials=results.get("n_trials", 0),
            optimizer_used=results.get("optimizer_used", ""),
            engine_used=results.get("engine_used", ""),
            download_urls=await _tune_download_urls(manager, session_id, metadata),
            duration_seconds=metadata.get("duration_seconds", 0),
            completed_at=metadata.get("completed_at", ""),
        )
    raise HTTPException(status_code=202, detail=f"Tune still {status}")


@router.post("/{session_id}/cancel")
async def cancel_tune(session_id: str):
    """Cooperatively cancel a running tune.

    Tune jobs run their optimizer loop inside ``asyncio.to_thread`` (BoTorch /
    CMA-ES are sync). ``asyncio.Task.cancel()`` cannot interrupt a worker
    thread — it would just abandon the task while the optimizer keeps running.
    Instead we write the ``.cancel`` marker via ``request_cancellation``;
    the executor's cancel-watcher coroutine picks that up and flips the
    ``threading.Event`` the runner polls between trials, giving us a clean
    cooperative exit (and final artifact write).

    Round 15 (kimbyn blocker): tune, pipeline, and predict all share the
    same :class:`SessionManager` and cancellation-marker namespace. Without
    a route-kind guard a caller could pass a pending/running pipeline or
    predict session id to ``POST /tune/{id}/cancel`` and the call would
    happily flip the cancellation marker for the non-tune job while
    responding ``"Tune cancellation requested"``. The predict router has
    the same guard (``predict_router.py::cancel_predict``); mirror it here
    by gating on the ``kind == "tune"`` discriminator that
    :func:`create_tune` stamps into session metadata.
    """
    manager = get_session_manager()
    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    # Refuse to cancel a session that wasn't started via /tune. Both
    # the top-level ``kind`` and the ``config.kind`` discriminator are
    # stamped by :func:`create_tune`; check both so an older session
    # missing the top-level field is still protected by the config
    # block. The shared :class:`SessionManager` also serves
    # ``pipeline`` and ``predict`` jobs — direct those to their own
    # cancel endpoints rather than silently accepting the call.
    #
    # A session created via the bare :meth:`SessionManager.create_session`
    # default (``config={}`` with no top-level ``kind``) is NOT a tune
    # session: that shape is what unknown/third-party callers produce,
    # and accepting it here would let ``/tune/{id}/cancel`` flip the
    # shared cancellation marker on any pending/running session that
    # happens to share the manager. Require an explicit ``tune``
    # discriminator on at least one of the two fields and reject
    # everything else (kimbyn review 2026-05-12).
    session_config = metadata.get("config") or {}
    config_kind = session_config.get("kind")
    metadata_kind = metadata.get("kind")
    is_tune_session = metadata_kind == "tune" or config_kind == "tune"
    if not is_tune_session:
        other_kind = metadata_kind or config_kind or "unknown"
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session {session_id} is not a tune session "
                f"(kind={other_kind!r}). Use POST /{other_kind}/{{id}}/cancel "
                "instead."
            )
            if other_kind in ("pipeline", "predict")
            else (
                f"Session {session_id} is not a tune session "
                f"(kind={other_kind!r}); refusing to cancel via /tune."
            ),
        )

    if metadata["status"] not in ("pending", "running"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel tune with status: {metadata['status']}",
        )
    await manager.request_cancellation(session_id)
    return {
        "session_id": session_id,
        "status": "cancelling",
        "message": "Tune cancellation requested",
    }


@router.get("/{session_id}/events")
async def stream_tune_events(session_id: str):
    """Stream tune progress events via SSE.

    Mirrors /pipeline/{session_id}/events: serves live events when the tune
    is running on this instance and falls back to polling guidance for
    cross-instance cases.
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    snapshot = event_bus.get_snapshot(session_id)
    if snapshot is None and not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    terminal_states = ("completed", "failed", "cancelled")
    if snapshot is None:
        metadata = await manager.get_session_metadata(session_id)
        final_state = (metadata or {}).get("status", "unknown")
        if final_state not in terminal_states:
            raise HTTPException(
                status_code=503,
                detail="Tune is running on a different instance; use polling instead",
            )

    async def event_generator():  # pragma: no cover - SSE transport loop
        queue = event_bus.get_queue(session_id)
        if snapshot is not None and snapshot.get("status") in terminal_states:
            final_state = snapshot["status"]
            yield {
                "event": "done",
                "data": json.dumps(
                    {"session_id": session_id, "final_state": final_state}
                ),
            }
            return
        if snapshot is None:
            metadata = await manager.get_session_metadata(session_id)
            if metadata and metadata.get("status") in terminal_states:
                final_state = metadata["status"]
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {"session_id": session_id, "final_state": final_state}
                    ),
                }
                return
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {"event": "progress", "data": event.model_dump_json()}
                    if event.state in (StepState.FAILED, StepState.CANCELLED):
                        yield {
                            "event": "done",
                            "data": json.dumps(
                                {"session_id": session_id, "final_state": event.state}
                            ),
                        }
                        break
                    if event.extra and event.extra.get("tune_ready"):
                        yield {
                            "event": "done",
                            "data": json.dumps(
                                {"session_id": session_id, "final_state": event.state}
                            ),
                        }
                        break
                except TimeoutError:
                    metadata = await manager.get_session_metadata(session_id)
                    if metadata and metadata.get("status") in terminal_states:
                        yield {
                            "event": "done",
                            "data": json.dumps(
                                {
                                    "session_id": session_id,
                                    "final_state": metadata["status"],
                                }
                            ),
                        }
                        break
                    yield {"event": "ping", "data": "keepalive"}
        except asyncio.CancelledError:
            logger.debug(f"SSE stream cancelled for {session_id[:8]}...")
            raise

    return EventSourceResponse(event_generator(), ping=15)


@router.get("/{session_id}/artifacts/{name}")
async def download_tune_artifact(session_id: str, name: str):
    """Download one of the tune artifacts by canonical name.

    Allowed names are exactly the artifact filenames the runner writes —
    anything else is rejected with 404 to avoid path traversal.
    """
    manager = get_session_manager()
    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    allowed = {
        artifact_name_from_key(spec.key, "tune"): (
            spec.media_type,
            spec.download_name,
        )
        for spec in TUNE_ARTIFACT_SPECS
    }
    allowed.update(
        {
            # Backward-compatible alias for sessions/clients created before the
            # canonical tuned artifact switched to .usd.
            "tuned_physics.usda": (
                "application/octet-stream",
                "tuned_physics.usda",
            ),
        }
    )
    if name not in allowed:
        raise HTTPException(status_code=404, detail=f"Unknown artifact: {name}")

    metadata = await manager.get_session_metadata(session_id) or {}
    available = await available_artifact_keys(manager, session_id, metadata, "tune")
    media_type, filename = allowed[name]
    legacy_aliases = {"tuned_physics.usda": "tuned_physics.usd"}
    requested_key = f"tune/{name}"
    if requested_key not in available and name in legacy_aliases:
        requested_key = f"tune/{legacy_aliases[name]}"
    if requested_key not in available:
        raise HTTPException(status_code=404, detail=f"Artifact not available: {name}")

    selected_key = requested_key
    artifact = await manager.open_local_artifact_key(session_id, selected_key)
    if artifact is None:
        # Try pulling from store (cross-instance / S3 case).
        await manager.sync_from_store(session_id, prefix="tune/")
        artifact = await manager.open_local_artifact_key(session_id, selected_key)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact not available: {name}")
    if selected_key != f"tune/{name}":
        filename = Path(selected_key).name
    return HeldFileResponse(artifact, media_type=media_type, filename=filename)
