# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline execution for Texture Agent Service.

Wraps the synchronous texture-agent pipeline by running each task
individually via asyncio.to_thread(), emitting progress events between steps.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine, Iterator
from contextlib import nullcontext, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from world_understanding.utils.credentials import redact_sensitive_log_text
from world_understanding.utils.usd.package import (
    UsdzPackageError,
    extract_usdz_members_to_dir,
    extract_usdz_package_for_edit,
    find_usdz_root_layer,
    write_usdz_package_from_directory,
)

from ..config import config as service_config
from ..runtime.bus import get_event_bus
from ..runtime.events import ProgressEvent, StepState
from ..sanitization import sanitize_message, sanitize_payload, sanitize_step_stats

logger = logging.getLogger(__name__)

# Map task class names to step names
_TASK_CLASS_TO_STEP = {
    "PrepareUVsTask": "prepare_uvs",
    "DiscoverMaterialsTask": "discover_materials",
    "PlanTexturesTask": "plan_textures",
    "GeneratePromptsTask": "generate_prompts",
    "RenderMaterialPreviewsTask": "render_previews",
    "GenerateTexturesTask": "generate_textures",
    "ExecuteTexturePlanTask": "generate_textures",
    "BlendTexturesTask": "blend_textures",
    "ApplyTexturesTask": "apply_textures",
    "RenderOutputTask": "render",
}
_PIPELINE_STARTUP_STEP = "pipeline_startup"
_WORKER_RESERVATION_HEARTBEAT_SECONDS = 60.0
_MAX_LOG_COUNT = 2_147_483_647


def _bounded_log_count(value: Any) -> int:
    """Return a non-negative integer suitable for a fixed-schema log field."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return min(_MAX_LOG_COUNT, max(0, value))
    if isinstance(value, float) and value.is_integer():
        return min(_MAX_LOG_COUNT, max(0, int(value)))
    return 0


def _log_cancelled_step_drain_failure() -> None:
    logger.error("code=cancelled_step_drain_failed")


def _log_unhandled_pipeline_failure() -> None:
    logger.error("code=pipeline_unhandled_failure")


def _log_step_failure(
    *,
    step_index: int,
    total_steps: int,
) -> None:
    logger.error(
        "code=pipeline_step_failed step_index=%d total_steps=%d",
        step_index,
        total_steps,
    )


def _log_failed_artifact_sync() -> None:
    logger.error("code=failed_step_artifact_sync_failed")


def _log_pipeline_stats(stats: dict[str, Any]) -> None:
    """Log only bounded aggregate fields from the final result."""
    logger.info(
        "Pipeline stats: materials_found=%d textures_generated=%d "
        "output_usd_count=%d renders_count=%d textures_failed=%d",
        _bounded_log_count(stats.get("materials_found")),
        _bounded_log_count(stats.get("textures_generated")),
        _bounded_log_count(stats.get("output_usd_count")),
        _bounded_log_count(stats.get("renders_count")),
        _bounded_log_count(stats.get("textures_failed")),
    )


def _resolver_stable_spec_value(
    value: Any,
    *,
    sdf: Any,
    asset_path_signature: Callable[[str], Any],
) -> Any:
    """Normalize resolver-backed asset values inside Sdf containers."""

    def _normalize(item: Any) -> Any:
        return _resolver_stable_spec_value(
            item,
            sdf=sdf,
            asset_path_signature=asset_path_signature,
        )

    if isinstance(value, sdf.AssetPath):
        return ("Sdf.AssetPath", asset_path_signature(value.path))
    if isinstance(value, sdf.AssetPathArray):
        return ("Sdf.AssetPathArray", tuple(_normalize(item) for item in value))
    if isinstance(value, sdf.Reference):
        return (
            "Sdf.Reference",
            asset_path_signature(value.assetPath),
            _normalize(value.primPath),
            ("Sdf.LayerOffset", value.layerOffset.offset, value.layerOffset.scale),
            _normalize(value.customData),
        )
    if isinstance(value, sdf.Payload):
        return (
            "Sdf.Payload",
            asset_path_signature(value.assetPath),
            _normalize(value.primPath),
            ("Sdf.LayerOffset", value.layerOffset.offset, value.layerOffset.scale),
        )
    if isinstance(value, sdf.Path):
        return ("Sdf.Path", str(value))
    list_op_names = (
        (sdf.PathListOp, "Sdf.PathListOp"),
        (sdf.ReferenceListOp, "Sdf.ReferenceListOp"),
        (sdf.PayloadListOp, "Sdf.PayloadListOp"),
    )
    list_op_name = next(
        (
            name
            for list_op_type, name in list_op_names
            if isinstance(value, list_op_type)
        ),
        None,
    )
    if list_op_name is not None:
        item_fields = (
            ("explicitItems",)
            if value.isExplicit
            else (
                "addedItems",
                "prependedItems",
                "appendedItems",
                "deletedItems",
                "orderedItems",
            )
        )
        return (
            list_op_name,
            tuple(
                (field, tuple(_normalize(item) for item in getattr(value, field)))
                for field in item_fields
            ),
        )
    if isinstance(value, dict):
        normalized_items = [
            (_normalize(key), _normalize(item)) for key, item in value.items()
        ]
        return ("dict", tuple(sorted(normalized_items, key=repr)))
    if isinstance(value, list):
        return ("list", tuple(_normalize(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_normalize(item) for item in value))
    if isinstance(value, set | frozenset):
        return (
            type(value).__name__,
            tuple(sorted((_normalize(item) for item in value), key=repr)),
        )
    return value


def _map_sdf_paths_in_value(
    value: Any,
    *,
    sdf: Any,
    map_path: Callable[[Any], Any],
) -> Any:
    """Rebuild an Sdf value while transforming every embedded Sdf.Path."""

    def _map(item: Any) -> Any:
        return _map_sdf_paths_in_value(item, sdf=sdf, map_path=map_path)

    if isinstance(value, sdf.Reference):
        return sdf.Reference(
            value.assetPath,
            value.primPath if value.assetPath else map_path(value.primPath),
            value.layerOffset,
            _map(value.customData),
        )
    if isinstance(value, sdf.Payload):
        return sdf.Payload(
            value.assetPath,
            value.primPath if value.assetPath else map_path(value.primPath),
            value.layerOffset,
        )
    if isinstance(value, sdf.Path):
        return map_path(value)
    for list_op_type in (
        sdf.PathListOp,
        sdf.ReferenceListOp,
        sdf.PayloadListOp,
    ):
        if not isinstance(value, list_op_type):
            continue
        remapped = list_op_type()
        item_fields = (
            ("explicitItems",)
            if value.isExplicit
            else (
                "addedItems",
                "prependedItems",
                "appendedItems",
                "deletedItems",
                "orderedItems",
            )
        )
        for item_field in item_fields:
            setattr(
                remapped,
                item_field,
                [_map(item) for item in getattr(value, item_field)],
            )
        return remapped
    if isinstance(value, tuple):
        return tuple(_map(item) for item in value)
    if isinstance(value, list):
        return [_map(item) for item in value]
    if isinstance(value, dict):
        return {_map(key): _map(item) for key, item in value.items()}
    return value


class _CancellationDrainError(RuntimeError):
    """A blocking worker call outlived the bounded cancellation drain."""


def _clear_task_cancellation_requests() -> None:
    """Clear pending cancellation count while draining a shielded thread."""
    task = asyncio.current_task()
    if task is None:  # pragma: no cover - no current task outside asyncio task context
        return

    uncancel = getattr(task, "uncancel", None)
    if uncancel is None:  # pragma: no cover - Python <3.11 compatibility guard
        return

    while task.cancelling():
        uncancel()


def _mark_stalled_until_future_done(
    session_manager: Any,
    session_id: str,
    step_name: str,
    step_future: asyncio.Future,
    reason: str,
) -> None:
    """Block deletion while a cancelled worker thread continues in background."""
    mark_worker_stalled = getattr(session_manager, "mark_worker_stalled", None)
    if mark_worker_stalled is not None:
        mark_worker_stalled(session_id, reason)

    def _clear_marker(fut: asyncio.Future) -> None:
        try:
            fut.result()
        except BaseException:  # pragma: no cover - post-cancel thread failure logging
            logger.debug(
                "Cancelled worker thread finished after stall marker for %s/%s",
                session_id[:8],
                step_name,
                exc_info=True,
            )

        clear_worker_stalled = getattr(session_manager, "clear_worker_stalled", None)
        if clear_worker_stalled is not None:
            clear_worker_stalled(session_id)

    step_future.add_done_callback(_clear_marker)


async def _drain_cancelled_step(
    *,
    session_id: str,
    step_name: str,
    step_future: asyncio.Future,
    session_manager: Any,
) -> None:
    """Wait for a cancelled threaded step with a hard deadline.

    The synchronous task cannot be interrupted by cancelling the asyncio
    wrapper. We keep the worker lock while the thread drains, but only up to a
    configured deadline so registry capacity cannot be pinned forever. If the
    deadline is exceeded, a stalled-worker marker keeps DELETE/TTL from
    removing artifacts until the thread future eventually finishes.
    """
    timeout_seconds = max(0.0, service_config.cancel_drain_timeout_seconds)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    while True:
        _clear_task_cancellation_requests()
        if step_future.done():  # pragma: no cover - cancellation race/drain edge
            try:
                step_future.result()
            except Exception:
                _log_cancelled_step_drain_failure()
                raise
            return

        remaining = deadline - loop.time()
        if remaining <= 0:  # pragma: no cover - hard timeout edge
            reason = (
                f"Cancellation timed out while waiting for step {step_name} "
                f"to stop after {timeout_seconds:.1f}s. The worker thread may "
                "still be writing artifacts."
            )
            _mark_stalled_until_future_done(
                session_manager,
                session_id,
                step_name,
                step_future,
                reason,
            )
            raise _CancellationDrainError(reason)

        try:
            await asyncio.wait_for(asyncio.shield(step_future), timeout=remaining)
            return
        except TimeoutError:
            reason = (
                f"Cancellation timed out while waiting for step {step_name} "
                f"to stop after {timeout_seconds:.1f}s. The worker thread may "
                "still be writing artifacts."
            )
            _mark_stalled_until_future_done(
                session_manager,
                session_id,
                step_name,
                step_future,
                reason,
            )
            raise _CancellationDrainError(reason)
        except asyncio.CancelledError:  # pragma: no cover - repeated cancel edge
            if step_future.done():
                continue
            logger.debug(
                "Additional cancellation while draining %s for %s",
                step_name,
                session_id[:8],
            )
            continue


async def _run_threaded_call_with_cancel_drain(
    function: Callable[..., Any],
    *args: Any,
    session_id: str,
    step_name: str,
    session_manager: Any,
) -> Any:
    """Run a blocking pipeline call without releasing its lock on cancellation."""
    loop = asyncio.get_running_loop()
    step_future = loop.run_in_executor(None, function, *args)
    try:
        return await asyncio.shield(step_future)
    except asyncio.CancelledError:
        logger.info(
            "Cancellation requested during %s for %s; waiting for worker "
            "thread to finish before releasing the session worker lock",
            step_name,
            session_id[:8],
        )
        await _drain_cancelled_step(
            session_id=session_id,
            step_name=step_name,
            step_future=step_future,
            session_manager=session_manager,
        )
        raise


async def _sync_prefix_to_store(
    session_manager: Any,
    session_id: str,
    prefix: str,
    *,
    attempts: int = 3,
) -> int:
    """Sync one artifact prefix with bounded retries before completion."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await asyncio.to_thread(
                session_manager.sync_to_store,
                session_id,
                prefix,
            )
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = min(2.0, 0.25 * attempt)
            logger.warning(
                "Retrying sync of %s for %s after error on attempt %d/%d: %s",
                redact_sensitive_log_text(prefix),
                session_id[:8],
                attempt,
                attempts,
                redact_sensitive_log_text(exc),
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"Failed to sync {prefix} to shared store: {last_error}")


async def _sync_prefix_to_store_with_cancel_drain(
    session_manager: Any,
    session_id: str,
    prefix: str,
    *,
    step_name: str,
) -> int:
    """Finish a started durable artifact sync before propagating cancellation."""
    sync_task = asyncio.create_task(
        _sync_prefix_to_store(
            session_manager,
            session_id,
            prefix,
        )
    )
    try:
        return await asyncio.shield(sync_task)
    except asyncio.CancelledError:
        logger.info(
            "Cancellation requested during %s for %s; waiting for durable "
            "artifact sync before releasing the session worker lock",
            step_name,
            session_id[:8],
        )
        await _drain_cancelled_step(
            session_id=session_id,
            step_name=step_name,
            step_future=sync_task,
            session_manager=session_manager,
        )
        raise


async def _worker_reservation_heartbeat_loop(
    session_manager: Any,
    session_id: str,
    owner_token: str,
    *,
    interval_seconds: float | None = None,
) -> None:
    """Refresh the shared worker reservation while a pipeline owns it."""
    heartbeat_worker = getattr(session_manager, "heartbeat_worker", None)
    if not callable(heartbeat_worker):  # pragma: no cover - duck-typed manager guard
        return

    interval = max(
        0.01,
        (
            _WORKER_RESERVATION_HEARTBEAT_SECONDS
            if interval_seconds is None
            else interval_seconds
        ),
    )
    while True:
        heartbeat_future = asyncio.create_task(
            asyncio.to_thread(
                heartbeat_worker,
                session_id,
                owner_token=owner_token,
            )
        )
        try:
            await asyncio.shield(heartbeat_future)
        except asyncio.CancelledError:  # pragma: no cover - heartbeat shutdown race
            with suppress(Exception):
                await heartbeat_future
            raise
        except Exception as exc:
            logger.debug(
                "Failed to heartbeat worker reservation for %s: %s",
                session_id[:8],
                exc,
            )
        await asyncio.sleep(interval)


def _start_worker_reservation_heartbeat(
    session_manager: Any,
    session_id: str,
    owner_token: str | None,
) -> asyncio.Task | None:
    if owner_token is None:
        return None
    uses_shared_store = getattr(session_manager, "uses_shared_store", lambda: False)
    try:
        if (
            not uses_shared_store()
        ):  # pragma: no cover - owner tokens only exist for shared stores
            return None
    except Exception as exc:
        logger.debug(
            "Cannot determine shared-store status for worker heartbeat on %s: %s",
            session_id[:8],
            exc,
        )
        return None

    return asyncio.create_task(
        _worker_reservation_heartbeat_loop(session_manager, session_id, owner_token)
    )


async def _stop_worker_reservation_heartbeat(
    heartbeat_task: asyncio.Task | None,
) -> None:
    if heartbeat_task is None:
        return
    heartbeat_task.cancel()
    with suppress(asyncio.CancelledError):
        await heartbeat_task


def _task_to_step_name(task: Any) -> str:
    """Get the step name for a task instance."""
    class_name = type(task).__name__
    return _TASK_CLASS_TO_STEP.get(class_name, class_name)


def _prepare_config_and_context(
    config_dict: dict[str, Any],
    session_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a config dict with working_dir set, apply defaults, and convert to context.

    Returns:
        Tuple of (resolved_config, pipeline_context).
    """
    from texture_agent.config.rendering_backends import (
        validate_texture_rendering_steps,
    )
    from texture_agent.config.schema import DEFAULTS, STEP_ORDER, STEP_OUTPUT_DIRS
    from texture_agent.config.unified_config import (
        apply_runtime_endpoint_overrides,
        config_to_context,
    )

    working_dir = session_dir / "cache"

    # Set project working_dir
    config_dict.setdefault("project", {})
    config_dict["project"]["working_dir"] = str(working_dir)

    # Ensure input section exists
    config_dict.setdefault("input", {})

    # Apply defaults for texture config
    texture = config_dict.setdefault("texture", {})
    for key, val in DEFAULTS["texture"].items():
        texture.setdefault(key, val)

    # Apply defaults for variations
    variations = config_dict.setdefault("variations", {})
    for key, val in DEFAULTS["variations"].items():
        variations.setdefault(key, val)

    # Apply defaults for steps
    steps = config_dict.setdefault("steps", {})
    for step_name in STEP_ORDER:
        step_cfg = steps.setdefault(step_name, {})
        defaults = DEFAULTS["steps"].get(step_name, {})
        for key, val in defaults.items():
            step_cfg.setdefault(key, val)

    apply_runtime_endpoint_overrides(config_dict)
    validate_texture_rendering_steps(steps)

    # Create working directory structure
    working_dir.mkdir(parents=True, exist_ok=True)
    for _step_name, dir_name in STEP_OUTPUT_DIRS.items():
        (working_dir / dir_name).mkdir(parents=True, exist_ok=True)

    context = config_to_context(config_dict)
    # Pipeline tasks such as PrepareUVsTask may replace ``usd_path`` with a
    # cache-local prepared layer. Keep the immutable uploaded source identity
    # so layered USDZ reconstruction still runs after that replacement.
    context["source_usd_path"] = context["usd_path"]
    # Uploaded raw USD inputs may use ../shared arcs within the bounded input
    # bundle. Let apply preserve that closure without granting access to the
    # rest of the session or host filesystem.
    context["usd_dependency_root"] = str(session_dir / "input")
    # ApplyTexturesTask localizes direct-CLI USDZ inputs itself. The service
    # owns a stricter source-package reconstruction immediately after apply,
    # so mark that ownership explicitly instead of inferring it from paths.
    context["service_managed_usdz_reconstruction"] = True
    planning_config = context.get("planning_config") or {}
    plan_path = working_dir / "texture_plan.json"
    if plan_path.is_file() and (
        planning_config.get("resume_execution")
        or planning_config.get("resume_apply_textures")
    ):
        context["texture_plan_path"] = str(plan_path)
    if planning_config.get("resume_apply_textures") and planning_config.get(
        "apply_texture_plan_unit_ids"
    ):
        # Existing-plan apply regeneration must be in resume mode before
        # GeneratePromptsTask runs so it loads cached prompts and scopes/rekeys
        # units to the immutable plan without contacting an auto-prompt backend.
        # Pre-plan sessions intentionally defer resume until ApplyTexturesTask;
        # their blended-map filenames use legacy display-derived keys.
        context["resume"] = True
    return config_dict, context


def _hydrate_cached_apply_context(context: dict[str, Any]) -> None:
    """Restore the runtime inputs required by an apply-only regeneration.

    Blended maps are durable cache artifacts, but ``PrimTextureUnit`` objects
    live only in the process that ran discovery/prompt expansion.  The service
    rebuilds those units before apply.  Plan-aware sessions must then rekey the
    rebuilt units to their stable texture-plan IDs because those IDs name the
    cached blended maps; legacy pre-plan sessions keep their display-derived
    keys.
    """
    planning_config = context.get("planning_config") or {}
    if not planning_config.get("resume_apply_textures"):
        return

    context["resume"] = True
    if not planning_config.get("apply_texture_plan_unit_ids"):
        return

    from texture_agent.execution import bind_prim_texture_units_to_plan
    from texture_agent.tasks.plan_textures import require_executable_texture_plan

    plan = require_executable_texture_plan(context)
    context["prim_texture_units"] = bind_prim_texture_units_to_plan(
        plan,
        list(context.get("prim_texture_units") or []),
    )


def _requires_executable_texture_plan(task: Any, context: dict[str, Any]) -> bool:
    """Return whether the task must pass the executable-plan gate.

    A legacy cached-apply request may build a compatibility plan solely to
    recover prompt scope and display-keyed units. The router marks that narrow
    path explicitly; it must not weaken the gate for generation tasks or
    modern plan-ID apply requests.
    """
    task_name = task.__class__.__name__
    if task_name not in {"GeneratePromptsTask", "ExecuteTexturePlanTask"}:
        return False
    if task_name == "GeneratePromptsTask":
        planning_config = context.get("planning_config") or {}
        if (
            planning_config.get("allow_non_executable_cached_apply_plan")
            and planning_config.get("apply_texture_plan_unit_ids") is False
        ):
            from texture_agent.functions.cached_apply import is_cached_apply_context
            from texture_agent.planning import (
                TexturePlanDecisionState,
                validate_texture_plan_payload,
            )

            if is_cached_apply_context(context):
                try:
                    raw_plan = context.get("texture_plan")
                    if raw_plan is None:
                        plan_path = Path(
                            context.get("texture_plan_path")
                            or Path(context.get("working_dir", "."))
                            / "texture_plan.json"
                        )
                        if not plan_path.is_file():
                            return True
                        raw_plan = plan_path.read_bytes()
                        context["texture_plan_path"] = str(plan_path)
                    plan = validate_texture_plan_payload(raw_plan)
                except (OSError, ValueError):
                    # Let the normal executable-plan gate raise the canonical
                    # missing/invalid artifact diagnostic.
                    return True
                # GeneratePromptsTask intentionally avoids stable-ID rebinding
                # for legacy display-key caches, but it still needs the
                # validated compatibility plan in context to scope discovery
                # on subsequent apply-only runs.
                context["texture_plan"] = plan
                return not (
                    plan.decision.state
                    == TexturePlanDecisionState.REQUIRES_OPERATOR_OVERRIDE
                    and plan.counts.planned_generation_job_count <= plan.limits.hard_cap
                )
    return True


def _usdz_extraction_limits(max_upload_size_mb: int) -> tuple[int, int]:
    """Return bounded member and byte limits for uploaded USDZ extraction."""
    max_upload_mb = max(max_upload_size_mb, 1)
    return (
        max(4096, min(65_536, max_upload_mb * 64)),
        max_upload_mb * 1024 * 1024,
    )


def _prepare_source_usdz_stage(
    context: dict[str, Any],
    session_dir: Path,
) -> Path | None:
    """Return an editable output root with uploaded USDZ members in place.

    ``ApplyTexturesTask`` exports only the edited root layer.  For an uploaded
    layered USDZ, rendering or packaging that root in ``cache/output`` loses
    the source package's relative payload, reference, and sublayer members.
    Extract the complete source package into a bounded session workspace. When
    UV preparation flattened the stage, transfer only changed opinions back to
    the package layers that originally authored their nearest source prims;
    otherwise replace the single edited root in place. Keeping the original
    composition layout lets rendering and packaging consume the same stage
    without pinning active variants or loaded payload contents.
    """
    import hashlib
    import zipfile

    from pxr import Sdf, Usd
    from texture_agent.functions.artifact_manifest import make_diagnostic

    output_paths = context.get("output_usd_paths", [])
    if not output_paths:
        return None

    output_usd = Path(output_paths[0])
    if not output_usd.is_file():
        message = f"Output USD not found: {output_usd}"
        _record_usdz_packaging_failure(context, message)
        logger.warning(message)
        return None

    staged_value = context.get("source_usdz_stage_path")
    if isinstance(staged_value, str) and staged_value:
        staged_root = Path(staged_value)
        staged_extract_value = context.get("source_usdz_extract_root")
        staged_extract_root = (
            Path(staged_extract_value)
            if isinstance(staged_extract_value, str) and staged_extract_value
            else None
        )
        if (
            staged_root.is_file()
            and staged_extract_root is not None
            and staged_extract_root.is_dir()
        ):
            return staged_root
        context.pop("source_usdz_stage_path", None)
        context.pop("source_usdz_extract_root", None)
        context.pop("render_output_usd_paths", None)

    session_root = session_dir.resolve()
    input_root = (session_root / "input").resolve()

    source_value = context.get("source_usd_path") or context.get("usd_path")
    if not isinstance(source_value, str) or not source_value:
        return output_usd
    try:
        source_usdz = Path(source_value).resolve()
        source_usdz.relative_to(input_root)
    except (OSError, ValueError):
        return output_usd
    if source_usdz.suffix.lower() != ".usdz":
        return output_usd
    if not source_usdz.is_file():
        message = f"Uploaded source USDZ not found: {source_usdz}"
        _record_usdz_packaging_failure(
            context,
            message,
            diagnostic=make_diagnostic(
                "PACKAGE_MISSING_ARTIFACT",
                severity="error",
                stage="package",
                message=message,
                recommended_action="Re-upload the source USDZ and retry.",
                details={"source_usdz": str(source_usdz)},
            ),
        )
        return None

    try:
        find_usdz_root_layer(source_usdz)
    except OSError as exc:
        message = f"Failed to read uploaded source USDZ: {exc}"
        _record_usdz_packaging_failure(context, message)
        return None
    except UsdzPackageError as exc:
        try:
            is_zip = zipfile.is_zipfile(source_usdz)
        except OSError as io_exc:
            message = f"Failed to read uploaded source USDZ: {io_exc}"
            _record_usdz_packaging_failure(context, message)
            return None
        if is_zip:
            # Preserve the existing asset-bundle path for ZIP-compatible
            # uploads that carry textures but no editable USD root.
            logger.debug(
                "Skipping source-stage preparation for asset-only "
                "USDZ-compatible upload %s: %s",
                source_usdz,
                exc,
            )
            return output_usd
        message = f"Failed to inspect uploaded source USDZ dependencies: {exc}"
        _record_usdz_packaging_failure(context, message)
        return None

    source_stage = Usd.Stage.Open(str(source_usdz))
    if not source_stage:  # pragma: no cover - root was validated immediately above
        message = f"Failed to compose uploaded source USDZ: {source_usdz}"
        _record_usdz_packaging_failure(context, message)
        return None
    source_prim_paths = tuple(str(prim.GetPath()) for prim in source_stage.Traverse())
    source_default_prim = source_stage.GetDefaultPrim()
    source_default_path = (
        str(source_default_prim.GetPath()) if source_default_prim.IsValid() else None
    )

    # Keep a bounded entry cap for archive safety without rejecting ordinary
    # production USDZs that contain hundreds or thousands of tiny members.
    max_members, max_total_bytes = _usdz_extraction_limits(
        service_config.max_upload_size_mb
    )
    digest = hashlib.sha256(str(source_usdz).encode("utf-8")).hexdigest()[:10]
    extract_root = session_root / "cache" / ".texture_agent_source_usdz" / digest
    try:
        extracted_root = extract_usdz_package_for_edit(
            source_usdz,
            extract_root,
            max_members=max_members,
            max_total_bytes=max_total_bytes,
        )
    except (OSError, UsdzPackageError) as exc:
        message = f"Failed to extract uploaded source USDZ dependencies: {exc}"
        _record_usdz_packaging_failure(
            context,
            message,
            diagnostic=make_diagnostic(
                "PACKAGE_MISSING_ARTIFACT",
                severity="error",
                stage="package",
                message=message,
                recommended_action="Inspect the source USDZ package layout and retry.",
                details={"source_usdz": str(source_usdz)},
            ),
        )
        return None

    prepared_value = context.get("usd_path")
    prepared_from_flattened_stage = False
    if isinstance(prepared_value, str) and prepared_value:
        with suppress(OSError, ValueError):
            prepared_from_flattened_stage = (
                Path(prepared_value).resolve() != source_usdz
            )
    try:
        edited_stage = Usd.Stage.Open(str(output_usd))
        if edited_stage is None:
            raise RuntimeError("Edited USD stage could not be opened")
        if prepared_from_flattened_stage:
            extracted_stage = Usd.Stage.Open(str(extracted_root))
            if extracted_stage is None:
                raise RuntimeError("Extracted source USDZ stage could not be opened")
            source_flat = extracted_stage.Flatten()
            source_flat_stage = Usd.Stage.Open(source_flat)
            if source_flat_stage is None:
                raise RuntimeError("Flattened source USDZ stage could not be opened")
            prepared_path = Path(prepared_value).resolve()
            if output_usd.resolve() == prepared_path:
                # PrepareUVs already exports a flattened working layer.
                # Flattening that artifact again renames its synthetic instance
                # prototype and makes an unchanged prim look newly authored.
                edited_flat = edited_stage.GetRootLayer()
            else:
                # ApplyTextures publishes a wrapper whose generated graphs live
                # in localized sublayers, so compose that wrapper exactly once.
                edited_flat = edited_stage.Flatten()

            # ApplyTextures opens the prepared stage through OpenUSD's global
            # layer registry and edits that stage in memory before exporting E.
            # Read B through an anonymous projection so a dirty cached layer
            # cannot turn Apply edits into baseline state.  The real file-backed
            # layer is retained only as the resolver anchor for B/E asset paths.
            prepared_anchor_layer = Sdf.Layer.FindOrOpen(str(prepared_path))
            prepared_baseline = Sdf.Layer.OpenAsAnonymous(str(prepared_path))
            if prepared_anchor_layer is None or prepared_baseline is None:
                raise RuntimeError("Prepared USD baseline could not be read from disk")
            prepared_baseline_stage = Usd.Stage.Open(prepared_baseline)
            if prepared_baseline_stage is None:
                raise RuntimeError("Prepared USD baseline stage could not be opened")
            get_edited_root_layer = getattr(edited_stage, "GetRootLayer", None)
            edited_asset_anchor_layer = (
                get_edited_root_layer()
                if callable(get_edited_root_layer)
                else edited_flat
            )
            extraction_root = extract_root.resolve()
            changed_layers: set[Sdf.Layer] = set()

            edited_flat_stage = Usd.Stage.Open(edited_flat)
            if edited_flat_stage is None:
                raise RuntimeError("Flattened edited USD stage could not be opened")

            def _flattened_backing_path_index(
                stage: Usd.Stage,
                layer: Sdf.Layer,
            ) -> tuple[
                dict[Sdf.Path, set[Sdf.Path]],
                dict[Sdf.Path, set[Sdf.Path]],
            ]:
                by_composed_path: dict[Sdf.Path, set[Sdf.Path]] = {}
                composed_paths_by_spec: dict[Sdf.Path, set[Sdf.Path]] = {}
                for composed_prim in Usd.PrimRange.Stage(
                    stage,
                    Usd.TraverseInstanceProxies(),
                ):
                    composed_path = composed_prim.GetPath()
                    for spec in composed_prim.GetPrimStack():
                        if spec.layer != layer or spec.path == composed_path:
                            continue
                        by_composed_path.setdefault(composed_path, set()).add(spec.path)
                        composed_paths_by_spec.setdefault(spec.path, set()).add(
                            composed_path
                        )
                return by_composed_path, composed_paths_by_spec

            source_specs_by_composed, source_composed_paths_by_spec = (
                _flattened_backing_path_index(source_flat_stage, source_flat)
            )
            baseline_specs_by_composed, baseline_composed_paths_by_spec = (
                _flattened_backing_path_index(
                    prepared_baseline_stage,
                    prepared_baseline,
                )
            )
            edited_specs_by_composed, edited_composed_paths_by_spec = (
                _flattened_backing_path_index(edited_flat_stage, edited_flat)
            )
            pre_edited_composed_paths_by_spec: dict[Sdf.Path, set[Sdf.Path]] = {}
            if callable(get_edited_root_layer) and hasattr(edited_stage, "Traverse"):
                pre_edited_layer = get_edited_root_layer()
                _, pre_edited_composed_paths_by_spec = _flattened_backing_path_index(
                    edited_stage,
                    pre_edited_layer,
                )

            baseline_source_flat_paths: dict[Sdf.Path, set[Sdf.Path]] = {}
            for composed_path, baseline_paths in baseline_specs_by_composed.items():
                source_paths = source_specs_by_composed.get(composed_path, set())
                for baseline_path in baseline_paths:
                    baseline_source_flat_paths.setdefault(
                        baseline_path,
                        set(),
                    ).update(source_paths)

            edited_baseline_paths: dict[Sdf.Path, set[Sdf.Path]] = {}
            for composed_path, edited_paths in edited_specs_by_composed.items():
                baseline_paths = baseline_specs_by_composed.get(composed_path, set())
                for edited_path in edited_paths:
                    edited_baseline_paths.setdefault(edited_path, set()).update(
                        baseline_paths
                    )

            baseline_superseded_backing_paths: set[Sdf.Path] = {
                source_path
                for source_path, composed_paths in source_composed_paths_by_spec.items()
                if source_path not in baseline_composed_paths_by_spec
                and prepared_baseline.GetPrimAtPath(source_path) is not None
                and any(
                    baseline_specs_by_composed.get(composed_path)
                    for composed_path in composed_paths
                )
            }
            baseline_superseded_backing_paths.update(
                root_spec.path
                for root_spec in prepared_baseline.rootPrims
                if root_spec.specifier == Sdf.SpecifierOver
                and root_spec.name.startswith("Flattened_Prototype_")
                and root_spec.path not in baseline_composed_paths_by_spec
            )

            superseded_backing_paths: set[Sdf.Path] = set()
            for previous_index in (
                source_composed_paths_by_spec,
                baseline_composed_paths_by_spec,
                pre_edited_composed_paths_by_spec,
            ):
                superseded_backing_paths.update(
                    previous_path
                    for previous_path, composed_paths in previous_index.items()
                    if previous_path not in edited_composed_paths_by_spec
                    and edited_flat.GetPrimAtPath(previous_path) is not None
                    and any(
                        edited_specs_by_composed.get(composed_path)
                        for composed_path in composed_paths
                    )
                )
            # Every flatten generation retains prior synthetic prototype
            # trees as inert root ``over`` specs while the composed instance
            # points at the newest generation. Intermediates are absent from
            # both the source and active pre-edit indexes, so discard any
            # inactive reserved flatten root regardless of chain length.
            superseded_backing_paths.update(
                root_spec.path
                for root_spec in edited_flat.rootPrims
                if root_spec.specifier == Sdf.SpecifierOver
                and root_spec.name.startswith("Flattened_Prototype_")
                and root_spec.path not in edited_composed_paths_by_spec
            )

            def _replace_mapped_prim_prefix(
                path: Sdf.Path,
                mapped_paths_by_prefix: dict[Sdf.Path, set[Sdf.Path]],
                *,
                ambiguity_message: str,
            ) -> Sdf.Path:
                prim_path = path.GetPrimPath()
                if not prim_path.IsAbsolutePath():
                    return path
                cursor = prim_path
                while not cursor.IsAbsoluteRootPath():
                    mapped_paths = mapped_paths_by_prefix.get(cursor)
                    if mapped_paths:
                        if len(mapped_paths) != 1:
                            sample = ", ".join(
                                str(item) for item in sorted(mapped_paths)[:5]
                            )
                            raise RuntimeError(
                                f"Prepared edit at {path} {ambiguity_message} "
                                f"({sample})"
                            )
                        return path.ReplacePrefix(
                            cursor,
                            next(iter(mapped_paths)),
                        )
                    cursor = cursor.GetParentPath()
                return path

            def _baseline_source_flat_path(path: Sdf.Path) -> Sdf.Path:
                """Map a trusted prepared-baseline path to the source flat."""
                return _replace_mapped_prim_prefix(
                    path,
                    baseline_source_flat_paths,
                    ambiguity_message="maps to multiple source-flat paths",
                )

            def _prepared_baseline_path(path: Sdf.Path) -> Sdf.Path:
                """Map an Apply output path to the trusted prepared baseline."""
                return _replace_mapped_prim_prefix(
                    path,
                    edited_baseline_paths,
                    ambiguity_message="maps to multiple prepared-baseline paths",
                )

            def _prepared_source_flat_path(path: Sdf.Path) -> Sdf.Path:
                """Map an Apply output path through B to the source flat."""
                return _baseline_source_flat_path(_prepared_baseline_path(path))

            def _baseline_composed_path(path: Sdf.Path) -> Sdf.Path:
                """Map a flattened baseline backing path to its composed path."""
                return _replace_mapped_prim_prefix(
                    path,
                    baseline_composed_paths_by_spec,
                    ambiguity_message=(
                        "maps to a flattened prototype used by multiple composed "
                        "paths; refusing ambiguous package writeback"
                    ),
                )

            def _prepared_composed_path(path: Sdf.Path) -> Sdf.Path:
                """Map a flattened prototype edit back to its composed path."""
                return _replace_mapped_prim_prefix(
                    path,
                    edited_composed_paths_by_spec,
                    ambiguity_message=(
                        "maps to a flattened prototype used by multiple composed "
                        "paths; refusing ambiguous package writeback"
                    ),
                )

            def _is_extracted_layer(layer: Sdf.Layer) -> bool:
                if not layer.realPath:
                    return False
                try:
                    Path(layer.realPath).resolve().relative_to(extraction_root)
                except (OSError, ValueError):
                    return False
                return True

            def _asset_path_signature(
                path_value: str,
                *,
                anchor_layer: Sdf.Layer,
            ) -> tuple[str, str]:
                """Resolve an asset value against the layer that owns it."""
                if not path_value:
                    return "authored-path", path_value
                try:
                    resolved = Sdf.ComputeAssetPathRelativeToLayer(
                        anchor_layer,
                        path_value,
                    )
                except (RuntimeError, ValueError):
                    resolved = ""
                return (
                    ("resolved-path", resolved)
                    if resolved
                    else ("authored-path", path_value)
                )

            def _spec_value_signature(
                value: Any,
                *,
                map_path: Callable[[Sdf.Path], Sdf.Path],
                asset_anchor_layer: Sdf.Layer,
            ) -> str:
                value = _map_sdf_paths_in_value(
                    value,
                    sdf=Sdf,
                    map_path=map_path,
                )
                return repr(
                    _resolver_stable_spec_value(
                        value,
                        sdf=Sdf,
                        asset_path_signature=lambda path: _asset_path_signature(
                            path,
                            anchor_layer=asset_anchor_layer,
                        ),
                    )
                )

            def _spec_signature(
                spec: Any,
                *,
                map_path: Callable[[Sdf.Path], Sdf.Path],
                asset_anchor_layer: Sdf.Layer,
            ) -> tuple[tuple[str, str], ...]:
                return tuple(
                    (
                        key,
                        _spec_value_signature(
                            spec.GetInfo(key),
                            map_path=map_path,
                            asset_anchor_layer=asset_anchor_layer,
                        ),
                    )
                    for key in sorted(spec.ListInfoKeys())
                )

            def _editable_prim_spec(prim: Usd.Prim) -> Any | None:
                candidates = [
                    spec
                    for spec in prim.GetPrimStack()
                    if _is_extracted_layer(spec.layer)
                ]
                for spec in candidates:
                    if spec.specifier in (Sdf.SpecifierDef, Sdf.SpecifierClass):
                        return spec
                return candidates[0] if candidates else None

            def _spec_identity(spec: Any) -> tuple[str, str]:
                return spec.layer.identifier, str(spec.path)

            prim_spec_composed_paths: dict[tuple[str, str], set[str]] = {}
            property_spec_composed_paths: dict[tuple[str, str], set[str]] = {}
            for composed_prim in Usd.PrimRange.Stage(
                extracted_stage,
                Usd.TraverseInstanceProxies(),
            ):
                editable_spec = _editable_prim_spec(composed_prim)
                if editable_spec is not None:
                    prim_spec_composed_paths.setdefault(
                        _spec_identity(editable_spec),
                        set(),
                    ).add(str(composed_prim.GetPath()))
                for composed_property in composed_prim.GetProperties():
                    for property_spec in composed_property.GetPropertyStack():
                        if not _is_extracted_layer(property_spec.layer):
                            continue
                        property_spec_composed_paths.setdefault(
                            _spec_identity(property_spec),
                            set(),
                        ).add(str(composed_property.GetPath()))

            def _authored_prim_specs(prim_spec: Any) -> Iterator[Any]:
                yield prim_spec
                for child in prim_spec.nameChildren:
                    yield from _authored_prim_specs(child)
                for variant_set in prim_spec.variantSets.values():
                    for variant in variant_set.variants.values():
                        yield from _authored_prim_specs(variant.primSpec)

            def _reachable_extracted_layers() -> list[Sdf.Layer]:
                discovered: dict[str, Sdf.Layer] = {}
                queue = [extracted_stage.GetRootLayer()]
                while queue:
                    layer = queue.pop()
                    identifier = str(layer.identifier)
                    if identifier in discovered:
                        continue
                    discovered[identifier] = layer
                    dependencies = list(layer.subLayerPaths)
                    with suppress(AttributeError, RuntimeError, ValueError):
                        dependencies.extend(layer.GetCompositionAssetDependencies())
                    for dependency in dict.fromkeys(dependencies):
                        if not dependency:
                            continue
                        with suppress(AttributeError, RuntimeError, ValueError):
                            resolved = Sdf.ComputeAssetPathRelativeToLayer(
                                layer, dependency
                            )
                            next_layer = (
                                Sdf.Layer.FindOrOpen(resolved) if resolved else None
                            )
                            if next_layer is not None:
                                queue.append(next_layer)
                return [
                    layer for layer in discovered.values() if _is_extracted_layer(layer)
                ]

            def _authored_composition_arc_roots() -> dict[
                str,
                list[tuple[Sdf.Path, str]],
            ]:
                """Index source roots for arcs authored in every variant branch."""
                arc_roots: dict[str, list[tuple[Sdf.Path, str]]] = {}
                layers = _reachable_extracted_layers()
                layer_by_identifier = {layer.identifier: layer for layer in layers}
                neighbors = {layer.identifier: {layer.identifier} for layer in layers}
                for layer in layers:
                    for sublayer_path in layer.subLayerPaths:
                        resolved = Sdf.ComputeAssetPathRelativeToLayer(
                            layer,
                            sublayer_path,
                        )
                        sublayer = Sdf.Layer.FindOrOpen(resolved) if resolved else None
                        if (
                            sublayer is None
                            or sublayer.identifier not in layer_by_identifier
                        ):
                            continue
                        neighbors[layer.identifier].add(sublayer.identifier)
                        neighbors[sublayer.identifier].add(layer.identifier)

                stack_members: dict[str, set[str]] = {}
                for identifier in neighbors:
                    pending = [identifier]
                    component: set[str] = set()
                    while pending:
                        candidate = pending.pop()
                        if candidate in component:
                            continue
                        component.add(candidate)
                        pending.extend(neighbors[candidate] - component)
                    stack_members[identifier] = component

                for layer in layers:
                    for root_prim in layer.pseudoRoot.nameChildren:
                        for spec in _authored_prim_specs(root_prim):
                            for arc_name, arc_list in (
                                ("reference", spec.referenceList),
                                ("payload", spec.payloadList),
                            ):
                                with suppress(AttributeError, RuntimeError, ValueError):
                                    for arc_index, item in enumerate(
                                        arc_list.GetAddedOrExplicitItems()
                                    ):
                                        asset_path = str(
                                            getattr(item, "assetPath", "") or ""
                                        )
                                        target = layer if not asset_path else None
                                        if asset_path:
                                            resolved = (
                                                Sdf.ComputeAssetPathRelativeToLayer(
                                                    layer,
                                                    asset_path,
                                                )
                                            )
                                            if resolved:
                                                target = Sdf.Layer.FindOrOpen(resolved)
                                        if target is None or not _is_extracted_layer(
                                            target
                                        ):
                                            continue
                                        # Sdf rejects non-empty relative reference
                                        # and payload prim paths when authoring.
                                        target_path = item.primPath
                                        if target_path.isEmpty:
                                            if not target.defaultPrim:
                                                continue
                                            target_path = (
                                                Sdf.Path.absoluteRootPath.AppendChild(
                                                    target.defaultPrim
                                                )
                                            )
                                        arc_user = (
                                            f"{layer.identifier}{spec.path}:"
                                            f"{arc_name}:{arc_index}"
                                        )
                                        for member in stack_members[target.identifier]:
                                            arc_roots.setdefault(member, []).append(
                                                (target_path, arc_user)
                                            )
                            for arc_name, path_list in (
                                ("inherit", spec.inheritPathList),
                                ("specialize", spec.specializesList),
                            ):
                                # Sdf resolves authored list entries to absolute paths.
                                for arc_index, target_path in enumerate(
                                    path_list.GetAddedOrExplicitItems()
                                ):
                                    arc_user = (
                                        f"{layer.identifier}{spec.path}:"
                                        f"{arc_name}:{arc_index}"
                                    )
                                    for member in stack_members[layer.identifier]:
                                        arc_roots.setdefault(member, []).append(
                                            (target_path, arc_user)
                                        )
                return arc_roots

            authored_arc_roots = _authored_composition_arc_roots()

            def _reject_shared_source_spec(
                spec: Any,
                *,
                composed_path: Sdf.Path,
                composed_paths_by_spec: dict[tuple[str, str], set[str]],
            ) -> None:
                composed_paths = composed_paths_by_spec.get(
                    _spec_identity(spec),
                    set(),
                )
                if len(composed_paths) <= 1:
                    authored_spec_path = spec.path.StripAllVariantSelections()
                    arc_users = {
                        user
                        for target_path, user in authored_arc_roots.get(
                            spec.layer.identifier,
                            [],
                        )
                        if authored_spec_path == target_path
                        or authored_spec_path.HasPrefix(target_path)
                    }
                    if len(arc_users) <= 1:
                        return
                    raise RuntimeError(
                        f"Prepared edit at {composed_path} maps to source spec "
                        f"{spec.path} in a layer referenced by more than one "
                        "composition arc, including branches outside the active "
                        "variant selection; refusing to apply one instance's "
                        "texture edits to every instance"
                    )
                sample = ", ".join(sorted(composed_paths)[:5])
                raise RuntimeError(
                    f"Prepared edit at {composed_path} maps to shared source "
                    f"spec {spec.path} used by multiple composed paths "
                    f"({sample}); refusing to apply one instance's texture "
                    "edits to every instance"
                )

            def _source_anchor(
                composed_path: Sdf.Path,
            ) -> tuple[Sdf.Layer, Sdf.Path, Sdf.Path]:
                cursor = composed_path
                while not cursor.IsAbsoluteRootPath():
                    source_prim = extracted_stage.GetPrimAtPath(cursor)
                    if source_prim.IsValid():
                        source_spec = _editable_prim_spec(source_prim)
                        if source_spec is not None:
                            _reject_shared_source_spec(
                                source_spec,
                                composed_path=composed_path,
                                composed_paths_by_spec=prim_spec_composed_paths,
                            )
                            return source_spec.layer, source_spec.path, cursor
                    cursor = cursor.GetParentPath()
                return extracted_stage.GetRootLayer(), Sdf.Path.absoluteRootPath, cursor

            def _destination_prim_mapping(
                prepared_path: Sdf.Path,
                *,
                map_composed_path: Callable[[Sdf.Path], Sdf.Path] = (
                    _prepared_composed_path
                ),
            ) -> tuple[Sdf.Layer, Sdf.Path, Sdf.Path, Sdf.Path]:
                def _namespace_prefixes(
                    composed_anchor: Sdf.Path,
                    source_anchor: Sdf.Path,
                ) -> tuple[Sdf.Path, Sdf.Path]:
                    composed_parts = (
                        str(composed_anchor.StripAllVariantSelections())
                        .strip("/")
                        .split("/")
                    )
                    source_parts = (
                        str(source_anchor.StripAllVariantSelections())
                        .strip("/")
                        .split("/")
                    )
                    common_suffix = 0
                    for composed_part, source_part in zip(
                        reversed(composed_parts),
                        reversed(source_parts),
                    ):
                        if composed_part != source_part:
                            break
                        common_suffix += 1
                    composed_prefix_parts = (
                        composed_parts[:-common_suffix]
                        if common_suffix
                        else composed_parts
                    )
                    source_prefix_parts = (
                        source_parts[:-common_suffix] if common_suffix else source_parts
                    )
                    composed_prefix = Sdf.Path("/" + "/".join(composed_prefix_parts))
                    source_prefix = Sdf.Path("/" + "/".join(source_prefix_parts))
                    return composed_prefix, source_prefix

                composed_path = map_composed_path(prepared_path)
                layer, anchor_spec_path, anchor_composed_path = _source_anchor(
                    composed_path
                )
                if anchor_composed_path.IsAbsoluteRootPath():
                    source_prefix, destination_prefix = _namespace_prefixes(
                        prepared_path,
                        composed_path,
                    )
                    return (
                        layer,
                        composed_path,
                        source_prefix,
                        destination_prefix,
                    )
                destination_path = composed_path.ReplacePrefix(
                    anchor_composed_path,
                    anchor_spec_path,
                )
                source_prefix, destination_prefix = _namespace_prefixes(
                    prepared_path,
                    destination_path,
                )
                return (
                    layer,
                    destination_path,
                    source_prefix,
                    destination_prefix,
                )

            def _destination_prim_path(
                composed_path: Sdf.Path,
                *,
                map_composed_path: Callable[[Sdf.Path], Sdf.Path] = (
                    _prepared_composed_path
                ),
            ) -> tuple[Sdf.Layer, Sdf.Path]:
                layer, destination, _source_prefix, _destination_prefix = (
                    _destination_prim_mapping(
                        composed_path,
                        map_composed_path=map_composed_path,
                    )
                )
                return layer, destination

            def _destination_property_path(
                composed_path: Sdf.Path,
                *,
                map_composed_path: Callable[[Sdf.Path], Sdf.Path] = (
                    _prepared_composed_path
                ),
            ) -> tuple[Sdf.Layer, Sdf.Path]:
                source_composed_path = map_composed_path(composed_path)
                source_property = extracted_stage.GetPropertyAtPath(
                    source_composed_path
                )
                if source_property.IsValid():
                    for spec in source_property.GetPropertyStack():
                        if _is_extracted_layer(spec.layer):
                            _reject_shared_source_spec(
                                spec,
                                composed_path=source_composed_path,
                                composed_paths_by_spec=property_spec_composed_paths,
                            )
                            return spec.layer, spec.path
                layer, prim_path = _destination_prim_path(
                    composed_path.GetPrimPath(),
                    map_composed_path=map_composed_path,
                )
                return layer, prim_path.AppendProperty(composed_path.name)

            def _destination_property_deletion_paths(
                composed_path: Sdf.Path,
                *,
                map_composed_path: Callable[[Sdf.Path], Sdf.Path] = (
                    _prepared_composed_path
                ),
            ) -> list[tuple[Sdf.Layer, Sdf.Path]]:
                """Return every authored destination that must be removed.

                Deleting only the strongest property spec can expose a weaker
                authored opinion from a referenced layer. A deletion from the
                flattened edit means the composed property must disappear, so
                remove every extracted spec after applying the shared-source
                guard to each one. Preserve the ordinary prim-based fallback
                only when the source composition has no extracted property spec.
                """
                destinations: list[tuple[Sdf.Layer, Sdf.Path]] = []
                source_composed_path = map_composed_path(composed_path)
                source_property = extracted_stage.GetPropertyAtPath(
                    source_composed_path
                )
                if source_property.IsValid():
                    for spec in source_property.GetPropertyStack():
                        if not _is_extracted_layer(spec.layer):
                            continue
                        _reject_shared_source_spec(
                            spec,
                            composed_path=source_composed_path,
                            composed_paths_by_spec=property_spec_composed_paths,
                        )
                        destinations.append((spec.layer, spec.path))
                if destinations:
                    return destinations
                layer, prim_path = _destination_prim_path(
                    composed_path.GetPrimPath(),
                    map_composed_path=map_composed_path,
                )
                return [(layer, prim_path.AppendProperty(composed_path.name))]

            def _remap_sdf_paths(
                value: Any,
                source_prefix: Sdf.Path,
                destination_prefix: Sdf.Path,
            ) -> Any:
                def _remap_path(path: Sdf.Path) -> Sdf.Path:
                    if source_prefix.IsAbsoluteRootPath():
                        return path
                    return (
                        path.ReplacePrefix(source_prefix, destination_prefix)
                        if path.HasPrefix(source_prefix)
                        else path
                    )

                return _map_sdf_paths_in_value(
                    value,
                    sdf=Sdf,
                    map_path=_remap_path,
                )

            def _remap_spec_path_fields(
                spec: Any,
                source_prefix: Sdf.Path,
                destination_prefix: Sdf.Path,
                *,
                recurse: bool,
            ) -> None:
                for key in spec.ListInfoKeys():
                    value = spec.GetInfo(key)
                    remapped = _remap_sdf_paths(
                        value,
                        source_prefix,
                        destination_prefix,
                    )
                    if remapped != value:
                        if key in {"connectionPaths", "targetPaths"}:
                            path_list = (
                                spec.connectionPathList
                                if key == "connectionPaths"
                                else spec.targetPathList
                            )
                            path_list.ClearEdits()
                            if remapped.isExplicit:
                                path_list.explicitItems = remapped.explicitItems
                            else:  # pragma: no cover - Sdf.CopySpec rebases list ops
                                for item_field in (
                                    "addedItems",
                                    "prependedItems",
                                    "appendedItems",
                                    "deletedItems",
                                    "orderedItems",
                                ):
                                    setattr(
                                        path_list,
                                        item_field,
                                        getattr(remapped, item_field),
                                    )
                        else:
                            spec.SetInfo(key, remapped)
                if not recurse:
                    return
                for property_spec in getattr(spec, "properties", ()):
                    _remap_spec_path_fields(
                        property_spec,
                        source_prefix,
                        destination_prefix,
                        recurse=False,
                    )
                for child_spec in getattr(spec, "nameChildren", ()):
                    _remap_spec_path_fields(
                        child_spec,
                        source_prefix,
                        destination_prefix,
                        recurse=True,
                    )

            uv_property_names = frozenset({"primvars:st", "primvars:st:indices"})

            def _identity_path(path: Sdf.Path) -> Sdf.Path:
                return path

            def _strip_apply_uv_properties(prim_spec: Any) -> None:
                """Remove UV properties from an E-added prim tree."""
                for property_name in uv_property_names:
                    if property_name in prim_spec.properties:
                        del prim_spec.properties[property_name]
                for child in prim_spec.nameChildren:
                    _strip_apply_uv_properties(child)

            def _copy_property_delta(
                source_layer: Sdf.Layer,
                property_spec: Any,
                *,
                map_composed_path: Callable[[Sdf.Path], Sdf.Path],
                description: str,
            ) -> None:
                property_layer, property_path = _destination_property_path(
                    property_spec.path,
                    map_composed_path=map_composed_path,
                )
                if not Sdf.CopySpec(
                    source_layer,
                    property_spec.path,
                    property_layer,
                    property_path,
                ):
                    raise RuntimeError(
                        f"Failed to transfer {description} property "
                        f"{property_spec.path}"
                    )
                copied_property = property_layer.GetPropertyAtPath(property_path)
                if copied_property is None:
                    raise RuntimeError(
                        f"Transferred {description} property is missing at "
                        f"{property_path}"
                    )
                (
                    _property_layer,
                    _property_prim_path,
                    property_source_prefix,
                    property_destination_prefix,
                ) = _destination_prim_mapping(
                    property_spec.path.GetPrimPath(),
                    map_composed_path=map_composed_path,
                )
                _remap_spec_path_fields(
                    copied_property,
                    property_source_prefix,
                    property_destination_prefix,
                    recurse=False,
                )
                changed_layers.add(property_layer)

            def _delete_property_delta(
                property_path: Sdf.Path,
                *,
                map_composed_path: Callable[[Sdf.Path], Sdf.Path],
            ) -> None:
                for (
                    property_layer,
                    destination_path,
                ) in _destination_property_deletion_paths(
                    property_path,
                    map_composed_path=map_composed_path,
                ):
                    destination_prim = property_layer.GetPrimAtPath(
                        destination_path.GetPrimPath()
                    )
                    if (
                        destination_prim is None
                        or destination_path.name not in destination_prim.properties
                    ):
                        continue
                    del destination_prim.properties[destination_path.name]
                    changed_layers.add(property_layer)

            def _active_composed_prim_paths(stage: Usd.Stage) -> set[Sdf.Path]:
                """Return only paths that participate in the active composition."""
                return {
                    prim.GetPath()
                    for prim in Usd.PrimRange.Stage(
                        stage,
                        Usd.TraverseInstanceProxies(),
                    )
                }

            def _validate_prepared_composition() -> None:
                """Validate S -> B UV preservation and B -> E prim preservation."""
                source_paths = _active_composed_prim_paths(source_flat_stage)
                baseline_paths = _active_composed_prim_paths(prepared_baseline_stage)
                edited_paths = _active_composed_prim_paths(edited_flat_stage)

                for source_path in sorted(source_paths):
                    source_prim = source_flat_stage.GetPrimAtPath(source_path)
                    source_uv_properties = {
                        property_name
                        for property_name in uv_property_names
                        if source_prim.HasProperty(property_name)
                    }
                    if not source_uv_properties:
                        continue
                    if source_path not in baseline_paths:
                        raise RuntimeError(
                            "Prepared UV baseline removed an active source prim or "
                            "subtree containing UV opinions; refusing destructive "
                            f"package writeback at {source_path}"
                        )
                    baseline_prim = prepared_baseline_stage.GetPrimAtPath(source_path)
                    for property_name in sorted(source_uv_properties):
                        if not baseline_prim.HasProperty(property_name):
                            raise RuntimeError(
                                "Prepared UV baseline removed a source UV opinion; "
                                "refusing destructive package writeback at "
                                f"{source_path.AppendProperty(property_name)}"
                            )

                missing_apply_paths = sorted(baseline_paths - edited_paths)
                if missing_apply_paths:
                    sample = ", ".join(
                        str(path)
                        for path in missing_apply_paths[:_MAX_ERRORS_IN_PAYLOAD]
                    )
                    raise RuntimeError(
                        "Apply output removed active prepared prims or subtrees; "
                        "whole-prim deletion is not supported: "
                        f"{sample}"
                    )

            def _copy_trusted_uv_deltas(baseline_spec: Any) -> None:
                """Publish only B-vs-S UV property changes from preparation."""
                if baseline_spec.path in baseline_superseded_backing_paths:
                    return

                source_spec = source_flat.GetPrimAtPath(
                    _baseline_source_flat_path(baseline_spec.path)
                )
                baseline_properties = {
                    prop.name: prop
                    for prop in baseline_spec.properties
                    if prop.name in uv_property_names
                }
                source_properties = (
                    {
                        prop.name: prop
                        for prop in source_spec.properties
                        if prop.name in uv_property_names
                    }
                    if source_spec is not None
                    else {}
                )
                for property_name in sorted(
                    baseline_properties.keys() | source_properties.keys()
                ):
                    baseline_property = baseline_properties.get(property_name)
                    source_property = source_properties.get(property_name)
                    if (
                        baseline_property is not None
                        and source_property is not None
                        and _spec_signature(
                            baseline_property,
                            map_path=_baseline_source_flat_path,
                            asset_anchor_layer=prepared_anchor_layer,
                        )
                        == _spec_signature(
                            source_property,
                            map_path=_identity_path,
                            asset_anchor_layer=source_flat,
                        )
                    ):
                        continue
                    if baseline_property is None:
                        raise RuntimeError(
                            "Prepared UV baseline removed a source UV opinion; "
                            "refusing destructive package writeback at "
                            f"{baseline_spec.path.AppendProperty(property_name)}"
                        )
                    else:
                        _copy_property_delta(
                            prepared_baseline,
                            baseline_property,
                            map_composed_path=_baseline_composed_path,
                            description="trusted UV",
                        )

                for child in baseline_spec.nameChildren:
                    _copy_trusted_uv_deltas(child)

            def _copy_apply_deltas(edited_spec: Any) -> None:
                """Publish genuine E-vs-B Apply edits, ignoring prepare rewrites."""
                if edited_spec.path in superseded_backing_paths:
                    # Re-flattening an already flattened instance retains its old
                    # synthetic backing tree while composing through a newly
                    # generated one. The retained tree is unreachable.
                    return
                composed_path = edited_spec.path
                baseline_spec = prepared_baseline.GetPrimAtPath(
                    _prepared_baseline_path(composed_path)
                )
                if baseline_spec is None:
                    (
                        destination_layer,
                        destination_path,
                        source_prefix,
                        destination_prefix,
                    ) = _destination_prim_mapping(composed_path)
                    if not Sdf.CopySpec(
                        edited_flat,
                        composed_path,
                        destination_layer,
                        destination_path,
                    ):
                        raise RuntimeError(
                            f"Failed to transfer Apply-added prim {composed_path}"
                        )
                    copied_spec = destination_layer.GetPrimAtPath(destination_path)
                    if copied_spec is None:
                        raise RuntimeError(
                            f"Transferred Apply-added prim is missing at "
                            f"{destination_path}"
                        )
                    _strip_apply_uv_properties(copied_spec)
                    _remap_spec_path_fields(
                        copied_spec,
                        source_prefix,
                        destination_prefix,
                        recurse=True,
                    )
                    changed_layers.add(destination_layer)
                    return

                edited_info = set(edited_spec.ListInfoKeys())
                baseline_info = set(baseline_spec.ListInfoKeys())
                changed_info_keys: list[str] = []
                for key in sorted((edited_info | baseline_info) - {"specifier"}):
                    edited_value = (
                        _spec_value_signature(
                            edited_spec.GetInfo(key),
                            map_path=_prepared_source_flat_path,
                            asset_anchor_layer=edited_asset_anchor_layer,
                        )
                        if key in edited_info
                        else None
                    )
                    baseline_value = (
                        _spec_value_signature(
                            baseline_spec.GetInfo(key),
                            map_path=_baseline_source_flat_path,
                            asset_anchor_layer=prepared_anchor_layer,
                        )
                        if key in baseline_info
                        else None
                    )
                    if edited_value == baseline_value:
                        continue
                    changed_info_keys.append(key)

                changed_properties = [
                    edited_property
                    for edited_property in edited_spec.properties
                    if edited_property.name not in uv_property_names
                    if (
                        baseline_property := prepared_baseline.GetPropertyAtPath(
                            _prepared_baseline_path(edited_property.path)
                        )
                    )
                    is None
                    or _spec_signature(
                        edited_property,
                        map_path=_prepared_source_flat_path,
                        asset_anchor_layer=edited_asset_anchor_layer,
                    )
                    != _spec_signature(
                        baseline_property,
                        map_path=_baseline_source_flat_path,
                        asset_anchor_layer=prepared_anchor_layer,
                    )
                ]
                deleted_properties = {
                    property_spec.name
                    for property_spec in baseline_spec.properties
                    if property_spec.name not in uv_property_names
                } - {
                    property_spec.name
                    for property_spec in edited_spec.properties
                    if property_spec.name not in uv_property_names
                }
                if (
                    not changed_info_keys
                    and not changed_properties
                    and not deleted_properties
                ):
                    # Resolving a destination is what enforces the shared-source
                    # guard, so a prim carrying no edit must never reach it. On an
                    # instanced asset every prim composes at several paths, so an
                    # untouched material would otherwise refuse the whole package
                    # on behalf of edits it never received.
                    for child in edited_spec.nameChildren:
                        _copy_apply_deltas(child)
                    return

                if changed_info_keys:
                    (
                        destination_layer,
                        destination_path,
                        source_prefix,
                        destination_prefix,
                    ) = _destination_prim_mapping(composed_path)
                    destination_spec = Sdf.CreatePrimInLayer(
                        destination_layer,
                        destination_path,
                    )
                    if destination_spec is None:
                        raise RuntimeError(
                            f"Failed to locate prepared prim target {destination_path}"
                        )
                    for key in changed_info_keys:
                        if key in edited_info:
                            destination_spec.SetInfo(
                                key,
                                _remap_sdf_paths(
                                    edited_spec.GetInfo(key),
                                    source_prefix,
                                    destination_prefix,
                                ),
                            )
                        else:
                            destination_spec.ClearInfo(key)
                        changed_layers.add(destination_layer)

                for deleted_name in sorted(deleted_properties):
                    _delete_property_delta(
                        composed_path.AppendProperty(deleted_name),
                        map_composed_path=_prepared_composed_path,
                    )

                for edited_property in edited_spec.properties:
                    if edited_property.name in uv_property_names:
                        continue
                    baseline_property = prepared_baseline.GetPropertyAtPath(
                        _prepared_baseline_path(edited_property.path)
                    )
                    if baseline_property is not None and _spec_signature(
                        edited_property,
                        map_path=_prepared_source_flat_path,
                        asset_anchor_layer=edited_asset_anchor_layer,
                    ) == _spec_signature(
                        baseline_property,
                        map_path=_baseline_source_flat_path,
                        asset_anchor_layer=prepared_anchor_layer,
                    ):
                        continue
                    _copy_property_delta(
                        edited_flat,
                        edited_property,
                        map_composed_path=_prepared_composed_path,
                        description="Apply",
                    )

                for child in edited_spec.nameChildren:
                    _copy_apply_deltas(child)

            # Validate the complete active compositions before mutating any
            # extracted package member.
            _validate_prepared_composition()
            for root_prim in prepared_baseline.rootPrims:
                _copy_trusted_uv_deltas(root_prim)
            for root_prim in edited_flat.rootPrims:
                _copy_apply_deltas(root_prim)
            for layer in changed_layers:
                if not layer.Save():
                    raise RuntimeError(
                        f"Failed to save prepared edits to {layer.realPath}"
                    )
            context["source_usdz_prepared_edit_layer_paths"] = sorted(
                layer.realPath for layer in changed_layers
            )
        elif not edited_stage.GetRootLayer().Export(str(extracted_root)):
            raise RuntimeError("USD layer export returned false")
    except Exception as exc:
        message = (
            f"Failed to stage the edited USDZ root for rendering and packaging: {exc}"
        )
        _record_usdz_packaging_failure(context, message)
        logger.exception(message)
        return None

    reconstructed_stage = Usd.Stage.Open(str(extracted_root))
    if not reconstructed_stage:  # pragma: no cover - export succeeded immediately above
        message = f"Failed to compose reconstructed USDZ stage: {extracted_root}"
        _record_usdz_packaging_failure(context, message)
        return None
    missing_prim_paths = [
        prim_path
        for prim_path in source_prim_paths
        if not reconstructed_stage.GetPrimAtPath(prim_path).IsValid()
    ]
    reconstructed_default = reconstructed_stage.GetDefaultPrim()
    reconstructed_default_path = (
        str(reconstructed_default.GetPath())
        if reconstructed_default.IsValid()
        else None
    )
    if missing_prim_paths or reconstructed_default_path != source_default_path:
        sampled_missing = missing_prim_paths[:_MAX_ERRORS_IN_PAYLOAD]
        message = (
            "Reconstructed USDZ stage did not preserve the source composition: "
            f"{len(missing_prim_paths)} source prim(s) missing"
        )
        _record_usdz_packaging_failure(
            context,
            message,
            diagnostic=make_diagnostic(
                "PACKAGE_MISSING_ARTIFACT",
                severity="error",
                stage="package",
                message=message,
                recommended_action=(
                    "Inspect the source package's payload, reference, and sublayer "
                    "members, then retry with a complete USDZ."
                ),
                details={
                    "source_default_prim": source_default_path,
                    "reconstructed_default_prim": reconstructed_default_path,
                    "missing_prim_count": len(missing_prim_paths),
                    "missing_prim_paths": sampled_missing,
                },
            ),
        )
        return None

    context["source_usdz_stage_path"] = str(extracted_root)
    context["source_usdz_extract_root"] = str(extract_root)
    context["render_output_usd_paths"] = [str(extracted_root)]
    return extracted_root


def _layer_authored_asset_paths(layer: Any) -> list[str]:
    """Return authored asset paths from typed dependency fields on one layer.

    Serialized layer text cannot distinguish an asset dependency from an
    ordinary String value that happens to contain ``@``-delimited text, so
    inspect composition arcs, clip dependencies, and Asset-typed properties
    rather than parsing ``ExportToString()``. Paths are returned exactly as
    authored, so package-relative spellings stay relative.
    """
    from pxr import Sdf

    authored: list[str] = []
    for accessor in (
        "GetCompositionAssetDependencies",
        "GetExternalAssetDependencies",
    ):
        with suppress(AttributeError, RuntimeError, ValueError):
            authored.extend(getattr(layer, accessor)())

    def _asset_paths(value: Any) -> list[str]:
        if isinstance(value, Sdf.AssetPath):
            return [value.path] if value.path else []
        paths: list[str] = []
        if not isinstance(value, str | bytes):
            with suppress(TypeError):
                for item in value:
                    if isinstance(item, Sdf.AssetPath) and item.path:
                        paths.append(item.path)
        return paths

    asset_types = (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray)

    def _visit(spec_path: Any) -> None:
        spec = layer.GetObjectAtPath(spec_path)
        if not isinstance(spec, Sdf.AttributeSpec):
            return
        if spec.typeName not in asset_types:
            return
        authored.extend(_asset_paths(spec.default))
        with suppress(AttributeError, RuntimeError, ValueError):
            for sample in layer.ListTimeSamplesForPath(spec_path):
                authored.extend(_asset_paths(layer.QueryTimeSample(spec_path, sample)))

    with suppress(AttributeError, RuntimeError, ValueError):
        layer.Traverse(Sdf.Path("/"), _visit)
    return authored


def _package_usdz(context: dict[str, Any], session_dir: Path) -> str | None:
    """Package the output USD + textures into a self-contained USDZ.

    Rewrites generated texture paths to package-relative references, preserves
    a layered source archive exactly, and includes only the composed dependency
    closure for a single-layer output.

    Returns:
        Path to the USDZ file, or None if packaging failed.
    """
    import hashlib
    import os
    import re
    import shutil
    import zipfile

    from pxr import Ar, Sdf, Usd, UsdShade, UsdUtils
    from texture_agent.functions.artifact_manifest import (
        _output_texture_references,
        make_diagnostic,
        validate_output_texture_portability,
    )

    output_paths = context.get("output_usd_paths", [])
    if not output_paths:  # pragma: no cover - packaging skipped without apply output
        return None

    output_usd = Path(output_paths[0])
    package_usd = _prepare_source_usdz_stage(context, session_dir)
    if package_usd is None:
        return None

    uri_scheme_re = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
    safe_filename_re = re.compile(r"[^A-Za-z0-9._-]+")
    package_texture_suffixes = frozenset({".jpeg", ".jpg", ".png"})
    session_root = session_dir.resolve()
    package_workspace = (session_root / "cache").resolve()
    textures_dir = package_workspace / "textures"
    source_extract_value = context.get("source_usdz_extract_root")
    source_extract_root: Path | None = None
    if isinstance(source_extract_value, str) and source_extract_value:
        with suppress(OSError, ValueError):
            candidate_extract_root = Path(source_extract_value).resolve()
            package_usd.resolve().relative_to(candidate_extract_root)
            candidate_extract_root.relative_to(package_workspace)
            if candidate_extract_root.is_dir():
                source_extract_root = candidate_extract_root
    archive_root = source_extract_root or package_workspace

    packaged_generated_textures_dir = textures_dir
    if source_extract_root is not None:
        base_name = ".texture_agent_generated_textures"
        stored_generated_dir = context.get("usdz_generated_textures_member")
        candidate_generated_dir: Path | None = None
        if isinstance(stored_generated_dir, str) and stored_generated_dir:
            stored_path = Path(stored_generated_dir)
            with suppress(OSError, ValueError):
                stored_candidate = (source_extract_root / stored_path).resolve()
                stored_candidate.relative_to(source_extract_root)
                if (
                    not stored_path.is_absolute()
                    and stored_path.parent == Path(".")
                    and (
                        stored_candidate.name == base_name
                        or stored_candidate.name.startswith(f"{base_name}_")
                    )
                ):
                    candidate_generated_dir = stored_candidate
        if candidate_generated_dir is None:
            candidate_generated_dir = source_extract_root / base_name
            if candidate_generated_dir.exists():
                digest = hashlib.sha256(str(package_usd).encode("utf-8")).hexdigest()[
                    :10
                ]
                candidate_generated_dir = source_extract_root / f"{base_name}_{digest}"
                counter = 1
                while candidate_generated_dir.exists():
                    candidate_generated_dir = (
                        source_extract_root / f"{base_name}_{digest}_{counter}"
                    )
                    counter += 1
            context["usdz_generated_textures_member"] = (
                candidate_generated_dir.relative_to(source_extract_root).as_posix()
            )
        packaged_generated_textures_dir = candidate_generated_dir
    textures_relative_dir = Path(
        os.path.relpath(packaged_generated_textures_dir, start=package_usd.parent)
    ).as_posix()

    def _texture_ref(filename: str) -> str:
        if source_extract_root is not None:
            source_texture = textures_dir / filename
            destination = packaged_generated_textures_dir / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source_texture.is_file() and destination.resolve() != source_texture:
                shutil.copy2(source_texture, destination)
        return f"{textures_relative_dir}/{filename}"

    input_root = session_root / "input"
    try:
        input_root = input_root.resolve()
        input_root.relative_to(session_root)
    except (OSError, ValueError):  # pragma: no cover - resolved session path guard
        input_root = session_root / "__invalid_input_root__"
    usdz_extracts_root = input_root / ".texture_agent_usdz_extracts"
    usdz_extracts_ready = False
    max_usdz_asset_members, max_usdz_asset_bytes = _usdz_extraction_limits(
        service_config.max_upload_size_mb
    )

    context_usd_parent: Path | None = None
    context_usd_path = context.get("source_usd_path") or context.get("usd_path")
    if isinstance(context_usd_path, str) and context_usd_path:
        try:
            candidate_parent = Path(context_usd_path).resolve().parent
            candidate_parent.relative_to(input_root)
            context_usd_parent = candidate_parent
        except (
            OSError,
            ValueError,
        ):  # pragma: no cover - context path outside input bundle
            context_usd_parent = None

    def _is_uri_ref(path_value: str) -> bool:
        return bool(uri_scheme_re.match(path_value))

    def _is_under(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    def _relative_path_parts(path_value: str) -> tuple[str, ...] | None:
        path = Path(path_value)
        if path.is_absolute() or any(part == ".." for part in path.parts):
            return None  # pragma: no cover - unsafe authored asset path
        return tuple(part for part in path.parts if part not in ("", "."))

    def _texture_filename(path_value: str) -> str:
        """Return a texture basename, including from a package-relative path."""
        candidate = path_value
        while Ar.IsPackageRelativePath(candidate):
            _outer, inner = Ar.SplitPackageRelativePathInner(candidate)
            candidate = inner
        return Path(candidate).name

    source_upload_value = context.get("source_usd_path") or context.get("usd_path")
    source_upload_path: Path | None = None
    if isinstance(source_upload_value, str) and source_upload_value:
        with suppress(OSError, ValueError):
            candidate_source_upload = Path(source_upload_value).resolve()
            candidate_source_upload.relative_to(input_root)
            source_upload_path = candidate_source_upload

    def _staged_source_package_member(path_value: str) -> Path | None:
        """Map ``source.usdz[member]`` to the reconstructed source member."""
        if source_extract_root is None or not Ar.IsPackageRelativePath(path_value):
            return None
        try:
            outer_path, inner_path = Ar.SplitPackageRelativePathOuter(path_value)
            if (
                source_upload_path is None
                or Path(outer_path).resolve() != source_upload_path
                or Ar.IsPackageRelativePath(inner_path)
            ):
                return None
            parts = _relative_path_parts(inner_path)
            if not parts:  # pragma: no cover - empty package member is not a texture
                return None
            candidate = source_extract_root.joinpath(*parts).resolve()
            if _is_under(candidate, source_extract_root) and candidate.is_file():
                return candidate
        except (  # pragma: no cover - pxr/path parser defensive guard
            OSError,
            RuntimeError,
            ValueError,
        ):
            return None
        return None

    stage = Usd.Stage.Open(str(package_usd))
    if not stage:  # pragma: no cover - malformed USD guard
        _record_usdz_packaging_failure(
            context, f"Failed to open output USD for USDZ packaging: {package_usd}"
        )
        return None

    def _extract_uploaded_usdz_assets() -> None:
        nonlocal usdz_extracts_ready
        if usdz_extracts_ready:  # pragma: no cover - memoized nested helper guard
            return
        usdz_extracts_ready = True
        if not input_root.is_dir():  # pragma: no cover - missing upload dir guard
            return

        extracted_members = 0
        extracted_bytes = 0
        packageable_suffixes = {
            ".jpeg",
            ".jpg",
            ".png",
            ".usd",
            ".usda",
            ".usdc",
            ".usdz",
        }
        for archive in input_root.rglob("*"):
            try:
                archive_resolved = archive.resolve()
            except OSError:  # pragma: no cover - filesystem race while scanning uploads
                continue
            if (
                not archive.is_file()
                or archive.suffix.lower() != ".usdz"
                or _is_under(archive_resolved, usdz_extracts_root.resolve())
            ):
                continue

            digest = hashlib.sha256(str(archive_resolved).encode()).hexdigest()[:10]
            safe_stem = safe_filename_re.sub("_", archive.stem).strip("._-") or "scene"
            extract_root = usdz_extracts_root / f"{safe_stem}_{digest}"
            try:
                remaining_members = max_usdz_asset_members - extracted_members
                if (
                    remaining_members <= 0
                ):  # pragma: no cover - package member safety cap
                    logger.warning(
                        "Skipped remaining uploaded USDZ package assets after "
                        "extracting %d members",
                        max_usdz_asset_members,
                    )
                    return
                stats = extract_usdz_members_to_dir(
                    archive_resolved,
                    extract_root,
                    allowed_suffixes=packageable_suffixes,
                    max_members=remaining_members,
                    max_total_bytes=max_usdz_asset_bytes - extracted_bytes,
                )
                extracted_members += stats.extracted_members
                extracted_bytes += stats.extracted_bytes
                if (
                    stats.skipped_members
                ):  # pragma: no cover - archive safety warning path
                    logger.warning(
                        "Skipped %d uploaded USDZ package asset(s) from %s due "
                        "to safety, suffix, or size limits",
                        stats.skipped_members,
                        archive,
                    )
                if (
                    stats.member_limit_reached
                ):  # pragma: no cover - archive safety cap path
                    logger.warning(
                        "Skipped remaining uploaded USDZ package assets after "
                        "extracting %d members",
                        max_usdz_asset_members,
                    )
                    return
            except (
                OSError,
                UsdzPackageError,
            ):  # pragma: no cover - unsafe upload archive guard
                continue

    def _resolve_authored_texture_ref(
        path_value: str,
        attr: Usd.Attribute | None,
    ) -> Path | None:
        """Resolve a relative texture from the layer that authored its value."""
        if attr is None:  # pragma: no cover - all internal callers supply an attr
            return None
        try:
            property_stack = attr.GetPropertyStack()
        except Exception:  # pragma: no cover - malformed USD property stack
            return None
        for spec in property_stack:
            try:
                authored_value = spec.default
                if isinstance(authored_value, Sdf.AssetPath):
                    authored_value = authored_value.path
                if authored_value != path_value:  # pragma: no cover - weaker opinion
                    continue
                resolved_value = Sdf.ComputeAssetPathRelativeToLayer(
                    spec.layer,
                    path_value,
                )
                if not resolved_value or _is_uri_ref(
                    resolved_value
                ):  # pragma: no cover - malformed layer result
                    return None
                candidate = Path(resolved_value).resolve()
                return candidate if candidate.is_file() else None
            except (
                AttributeError,
                OSError,
                RuntimeError,
                ValueError,
            ):  # pragma: no cover - malformed USD property spec
                continue
        return None  # pragma: no cover - no matching authored opinion

    def _resolve_local_texture_ref(
        path_value: str,
        *,
        attr: Usd.Attribute | None = None,
        resolved_path: str = "",
    ) -> Path | None:
        if _is_uri_ref(
            path_value
        ):  # pragma: no cover - callers reject URIs before resolution
            return None
        try:
            package_member = _staged_source_package_member(path_value)
            if package_member is None and resolved_path:
                package_member = _staged_source_package_member(resolved_path)
            if package_member is not None:
                return package_member
            path = Path(path_value)
            if path.is_absolute():
                return path.resolve()
            if resolved_path and not _is_uri_ref(resolved_path):
                resolved_candidate = Path(resolved_path).resolve()
                if resolved_candidate.is_file():
                    return resolved_candidate
            authored_candidate = _resolve_authored_texture_ref(path_value, attr)
            if authored_candidate is not None:
                return authored_candidate
            package_candidate = (package_usd.parent / path).resolve()
            if package_candidate.is_file():  # pragma: no cover - resolver fallback
                return package_candidate
            # ApplyTexturesTask authors generated refs relative to
            # cache/output before a layered USDZ root is reconstructed at its
            # original package location. Retain that original anchor long
            # enough to identify and rewrite those generated files.
            output_candidate = (output_usd.parent / path).resolve()
            if output_candidate.is_file():
                return output_candidate
            return package_candidate
        except (OSError, ValueError):  # pragma: no cover - invalid texture ref guard
            return None

    generated_texture_paths_by_unit: dict[str, dict[str, Path]] = {}
    for unit_key, blended in (context.get("blended_textures") or {}).items():
        unit_paths = generated_texture_paths_by_unit.setdefault(
            str(unit_key),
            {},
        )
        for channel in ("albedo", "normal", "orm"):
            path_value = getattr(blended, channel, None)
            if not path_value and isinstance(blended, dict):
                path_value = blended.get(channel)
            if path_value:
                with suppress(OSError, ValueError):
                    candidate = Path(path_value).resolve()
                    if candidate.is_file() and _is_under(candidate, textures_dir):
                        unit_paths[channel] = candidate
        for channel in ("roughness", "metalness"):
            with suppress(OSError, ValueError):
                candidate = (textures_dir / f"{unit_key}_{channel}.png").resolve()
                if candidate.is_file():
                    unit_paths[channel] = candidate

    unit_keys_by_material_path: dict[str, set[str]] = {}
    for unit in context.get("prim_texture_units") or []:
        unit_key = str(getattr(unit, "key", ""))
        material_info = getattr(unit, "material_info", None)
        if (
            not unit_key or material_info is None
        ):  # pragma: no cover - malformed context
            continue
        material_paths = {
            str(getattr(material_info, "prim_path", "")),
            *(
                str(path)
                for path in getattr(material_info, "material_alias_paths", ()) or ()
            ),
        }
        for material_path in material_paths:
            if material_path:
                unit_keys_by_material_path.setdefault(material_path, set()).add(
                    unit_key
                )

    def _owning_texture_unit(prim: Usd.Prim) -> str | None:
        ancestors: list[Usd.Prim] = []
        cursor = prim
        while cursor.IsValid() and not cursor.GetPath().IsAbsoluteRootPath():
            ancestors.append(cursor)
            material_keys = unit_keys_by_material_path.get(str(cursor.GetPath()), set())
            if len(material_keys) == 1:
                return next(iter(material_keys))
            cursor = cursor.GetParent()
        for ancestor in ancestors:
            name = str(ancestor.GetName())
            if name in generated_texture_paths_by_unit:  # pragma: no cover - clone ID
                return name
        return None

    def _texture_channel_for_reference(
        path_value: str,
        attr: Usd.Attribute,
    ) -> str | None:
        attr_name = attr.GetName().lower()
        channel_markers = (
            ("metalness", ("metalness", "metallic")),
            ("roughness", ("roughness",)),
            ("normal", ("normal", "normalmap")),
            ("albedo", ("albedo", "diffuse", "base_color", "basecolor")),
            ("orm", ("orm",)),
        )

        def _has_marker(value: str, marker: str) -> bool:
            normalized = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
            return f"_{marker}_" in f"_{normalized}_"

        for channel, markers in channel_markers:
            if any(_has_marker(attr_name, marker) for marker in markers):
                return channel

        prim = attr.GetPrim()
        if not prim.IsA(UsdShade.Shader):
            return None
        texture_shader = UsdShade.Shader(prim)
        if str(texture_shader.GetIdAttr().Get() or "") != "UsdUVTexture":
            return None
        material_prim = prim.GetParent()
        while material_prim.IsValid() and not material_prim.IsA(UsdShade.Material):
            material_prim = material_prim.GetParent()
        if not material_prim.IsValid():
            return None

        preview_channels = {
            "diffusecolor": "albedo",
            "basecolor": "albedo",
            "normal": "normal",
            "occlusion": "orm",
            "roughness": "roughness",
            "metallic": "metalness",
        }
        packed_outputs = {
            "orm": "r",
            "roughness": "g",
            "metalness": "b",
        }
        consumers: dict[str, set[str]] = {}
        has_unknown_consumer = False
        for candidate in Usd.PrimRange(material_prim):
            if not candidate.IsA(UsdShade.Shader):
                continue
            preview = UsdShade.Shader(candidate)
            if str(preview.GetIdAttr().Get() or "") != "UsdPreviewSurface":
                continue
            for preview_input in preview.GetInputs():
                connected = preview_input.GetConnectedSource()
                if connected is None or connected[0].GetPrim() != prim:
                    continue
                consumer_channel = preview_channels.get(
                    preview_input.GetBaseName().lower()
                )
                if consumer_channel is None:
                    has_unknown_consumer = True
                    continue
                output_name = str(connected[1] or "").lower()
                if output_name.startswith("outputs:"):
                    output_name = output_name.split(":", 1)[1]
                consumers.setdefault(consumer_channel, set()).add(output_name)
        if has_unknown_consumer or not consumers:
            return None
        if len(consumers) == 1:
            return next(iter(consumers))
        if set(consumers) <= set(packed_outputs) and all(
            packed_outputs[channel] in outputs for channel, outputs in consumers.items()
        ):
            return "orm"
        return None

    def _generated_texture_for_reference(
        path_value: str,
        *,
        attr: Usd.Attribute,
        resolved_path: str = "",
    ) -> Path | None:
        if _is_uri_ref(path_value):
            return None
        src_resolved = _resolve_local_texture_ref(
            path_value,
            attr=attr,
            resolved_path=resolved_path,
        )
        if src_resolved is not None and src_resolved.is_file():
            try:
                src_resolved.relative_to(textures_dir.resolve())
            except ValueError:
                pass
            else:
                return src_resolved

        # ApplyTexturesTask can leave the original relative spelling on
        # secondary shader inputs even when the referenced source member still
        # exists. Prefer this run's verified map only when the full material
        # identity and requested channel select it; an unrelated basename
        # collision is not sufficient.
        if Path(path_value).is_absolute():
            return None
        unit_key = _owning_texture_unit(attr.GetPrim())
        if unit_key is None:
            return None
        unit_paths = generated_texture_paths_by_unit.get(unit_key, {})
        channel = _texture_channel_for_reference(path_value, attr)
        if channel is not None:
            generated_path = unit_paths.get(channel)
            if generated_path is not None:
                return generated_path
        return None  # pragma: no cover - owning unit has no matching map

    def _is_existing_package_texture(
        path_value: str,
        *,
        attr: Usd.Attribute,
        resolved_path: str = "",
    ) -> bool:
        """Return whether a relative texture already lives in staged package data."""
        if _is_uri_ref(path_value) or Path(path_value).is_absolute():
            if not Ar.IsPackageRelativePath(path_value):
                return False
        src_resolved = _resolve_local_texture_ref(
            path_value,
            attr=attr,
            resolved_path=resolved_path,
        )
        return bool(
            src_resolved
            and src_resolved.is_file()
            and _is_under(src_resolved, archive_root)
        )

    def _rewritten_source_package_member_ref(
        path_value: str,
        *,
        attr: Usd.Attribute,
        resolved_path: str = "",
    ) -> str | None:
        """Rewrite a source-package ref relative to the reconstructed root."""
        src = _resolve_local_texture_ref(
            path_value,
            attr=attr,
            resolved_path=resolved_path,
        )
        if (  # pragma: no cover - caller verifies the same staged member first
            src is None or not src.is_file() or not _is_under(src, archive_root)
        ):
            return None
        return Path(os.path.relpath(src, start=package_usd.parent)).as_posix()

    localized_package_texture_refs: dict[Path, str] = {}

    def _localized_package_texture_ref(
        path_value: str,
        *,
        attr: Usd.Attribute,
        resolved_path: str = "",
    ) -> str | None:
        """Copy one source texture to a collision-safe package-only location."""
        src = _resolve_local_texture_ref(
            path_value,
            attr=attr,
            resolved_path=resolved_path,
        )
        if (  # pragma: no cover - guarded by _is_existing_package_texture
            src is None or not src.is_file() or not _is_under(src, archive_root)
        ):
            return None
        src = src.resolve()
        existing_ref = localized_package_texture_refs.get(src)
        if existing_ref is not None:
            return existing_ref

        safe_stem = safe_filename_re.sub("_", src.stem).strip("._-") or "texture"
        digest = hashlib.sha256(str(src).encode("utf-8")).hexdigest()[:10]
        source_assets_dir = package_usd.parent / ".texture_agent_source_assets"
        suffix = src.suffix.lower()
        if suffix not in package_texture_suffixes:
            return None  # pragma: no cover - callers require a texture suffix
        dest = source_assets_dir / f"{safe_stem}_{digest}{suffix}"
        source_assets_dir.mkdir(parents=True, exist_ok=True)
        if dest.resolve() != src:
            shutil.copy2(src, dest)
        relative_ref = Path(os.path.relpath(dest, start=package_usd.parent)).as_posix()
        localized_package_texture_refs[src] = relative_ref
        return relative_ref

    upload_texture_index: dict[str, list[Path]] | None = None
    upload_layer_index: dict[str, list[Path]] | None = None

    def _upload_bundle_texture_index() -> dict[str, list[Path]]:
        nonlocal upload_texture_index
        if upload_texture_index is not None:
            return upload_texture_index
        indexed: dict[str, list[Path]] = {}
        _extract_uploaded_usdz_assets()
        if input_root.is_dir():
            for match in input_root.rglob("*"):
                with suppress(OSError):
                    if (
                        not match.is_file()
                        or match.suffix.lower() not in package_texture_suffixes
                    ):
                        continue
                    indexed.setdefault(match.name, []).append(match.resolve())
        upload_texture_index = indexed
        return indexed

    def _upload_bundle_layer_index() -> dict[str, list[Path]]:
        nonlocal upload_layer_index
        if (
            upload_layer_index is not None
        ):  # pragma: no cover - memoized nested helper guard
            return upload_layer_index
        indexed: dict[str, list[Path]] = {}
        _extract_uploaded_usdz_assets()
        if input_root.is_dir():
            for match in input_root.rglob("*"):
                with suppress(OSError):
                    if not match.is_file() or match.suffix.lower() not in {
                        ".usd",
                        ".usda",
                        ".usdc",
                        ".usdz",
                    }:
                        continue
                    indexed.setdefault(match.name, []).append(match.resolve())
        upload_layer_index = indexed
        return indexed

    def _find_upload_bundle_texture(path_value: str) -> Path | None:
        if (
            _is_uri_ref(path_value)
            or Path(_texture_filename(path_value)).suffix.lower()
            not in package_texture_suffixes
        ):
            return None
        if not input_root.is_dir():
            return None

        candidates: list[Path] = []
        path = Path(path_value)
        if path.is_absolute():  # pragma: no cover - absolute upload texture ref
            with suppress(OSError):
                resolved = path.resolve()
                if resolved.is_file() and _is_under(resolved, input_root):
                    candidates.append(resolved)
        else:
            for base in (context_usd_parent, input_root):
                if base is None:  # pragma: no cover - optional source USD parent
                    continue
                with suppress(OSError):
                    resolved = (base / path).resolve()
                    if resolved.is_file() and _is_under(
                        resolved, input_root
                    ):  # pragma: no cover - absolute upload texture ref
                        candidates.append(resolved)

            parts = _relative_path_parts(path_value)
            if parts:
                filename = parts[-1]
                for match in _upload_bundle_texture_index().get(filename, []):
                    with suppress(OSError):
                        resolved = match.resolve()
                        relative_parts = resolved.relative_to(input_root).parts
                        if (
                            resolved.is_file()
                            and relative_parts[-len(parts) :] == parts
                        ):
                            candidates.append(resolved)

        unique_candidates = sorted(set(candidates))
        if len(unique_candidates) != 1:
            if (
                len(unique_candidates) > 1
            ):  # pragma: no cover - ambiguous upload texture guard
                logger.warning(
                    "Skipped ambiguous upload texture localization for %s: %s",
                    path_value,
                    [str(path) for path in unique_candidates],
                )
            return None  # pragma: no cover - missing/ambiguous upload texture ref
        return unique_candidates[0]

    def _safe_upload_texture_filename(src: Path) -> str:
        stem = safe_filename_re.sub("_", src.stem).strip("._-") or "texture"
        return f"{stem}{src.suffix.lower()}"

    localized_upload_texture_refs: dict[Path, str] = {}

    def _localized_upload_texture_ref(path_value: str) -> str | None:
        src = _find_upload_bundle_texture(path_value)
        if src is None:
            return None
        try:
            src = src.resolve()
        except (
            OSError,
            ValueError,
        ):  # pragma: no cover - invalid localized texture path
            return None

        existing_ref = localized_upload_texture_refs.get(src)
        if existing_ref is not None:
            return existing_ref

        textures_dir.mkdir(parents=True, exist_ok=True)
        filename = _safe_upload_texture_filename(src)
        dest = textures_dir / filename
        if (
            dest.exists() and dest.resolve() != src
        ):  # pragma: no cover - upload basename collision
            digest = hashlib.sha256(str(src).encode("utf-8")).hexdigest()[:10]
            suffix = Path(filename).suffix
            dest = textures_dir / f"{Path(filename).stem}_{digest}{suffix}"
            counter = 1
            while (
                dest.exists() and dest.resolve() != src
            ):  # pragma: no cover - repeated collision guard
                dest = (
                    textures_dir / f"{Path(filename).stem}_{digest}_{counter}{suffix}"
                )
                counter += 1

        if dest.resolve() != src:  # pragma: no cover - upload texture localization copy
            shutil.copy2(src, dest)
        localized_ref = _texture_ref(dest.name)
        localized_upload_texture_refs[src] = localized_ref
        return localized_ref

    def _find_upload_bundle_layer(path_value: str) -> Path | None:
        if _is_uri_ref(path_value) or Path(
            path_value
        ).suffix.lower() not in {  # pragma: no cover - non-layer ref guard
            ".usd",
            ".usda",
            ".usdc",
            ".usdz",
        }:
            return None
        if not input_root.is_dir():
            return None

        candidates: list[Path] = []
        path = Path(path_value)
        if path.is_absolute():  # pragma: no cover - absolute upload layer ref
            with suppress(OSError):
                resolved = path.resolve()
                if resolved.is_file() and _is_under(resolved, input_root):
                    candidates.append(resolved)
        else:
            for base in (context_usd_parent, input_root):
                if base is None:  # pragma: no cover - optional source USD parent
                    continue
                with suppress(OSError):
                    resolved = (base / path).resolve()
                    if resolved.is_file() and _is_under(resolved, input_root):
                        candidates.append(resolved)

            parts = _relative_path_parts(path_value)
            if parts:
                filename = parts[-1]
                for match in _upload_bundle_layer_index().get(filename, []):
                    with suppress(OSError):
                        resolved = match.resolve()
                        relative_parts = resolved.relative_to(input_root).parts
                        if (  # pragma: no cover - suffix upload layer match
                            resolved.is_file()
                            and relative_parts[-len(parts) :] == parts
                        ):
                            candidates.append(resolved)

        unique_candidates = sorted(set(candidates))
        if len(unique_candidates) != 1:
            if (
                len(unique_candidates) > 1
            ):  # pragma: no cover - ambiguous upload layer guard
                logger.warning(
                    "Skipped ambiguous upload layer localization for %s: %s",
                    path_value,
                    [str(path) for path in unique_candidates],
                )
            return None  # pragma: no cover - missing/ambiguous upload layer ref
        return unique_candidates[0]

    def _localize_upload_layer_ref(path_value: str) -> bool:
        parts = _relative_path_parts(path_value)
        if parts is None or _is_uri_ref(path_value):
            return False
        try:
            expected = package_usd.parent.joinpath(*parts).resolve()
        except (OSError, ValueError):  # pragma: no cover - invalid layer target
            return False
        if not _is_under(
            expected, archive_root
        ):  # pragma: no cover - safe parts cannot escape
            return False
        if expected.is_file():  # pragma: no cover - caller already checks existence
            return True

        src = _find_upload_bundle_layer(path_value)
        if src is None:  # pragma: no cover - missing upload layer guard
            return False
        try:
            src = src.resolve()
        except (OSError, ValueError):  # pragma: no cover - unsafe layer target guard
            return False

        expected.parent.mkdir(parents=True, exist_ok=True)
        if expected != src:
            try:
                shutil.copy2(src, expected)
            except OSError:  # pragma: no cover - upload layer copy failure
                return False
        return True

    def _material_binding_targets() -> set[str]:
        targets: set[str] = set()
        for prim in stage.Traverse():
            if prim.IsInstanceProxy():  # pragma: no cover - USD instance proxy guard
                continue
            for rel in prim.GetRelationships():
                rel_name = rel.GetName()
                if not rel_name.startswith("material:binding"):
                    continue
                # Nonvisual physics materials from SimReady assets can point to
                # optional package-side layers that are absent from uploaded USDZs.
                # They must not block visual texture USDZ packaging.
                if rel_name == "material:binding:physics":
                    continue
                targets.update(str(target) for target in rel.GetTargets())
        return targets

    def _is_unbound_material_like_ref(
        prim_path: Sdf.Path,
        bound_material_targets: set[str],
    ) -> bool:
        path_text = str(prim_path)
        if path_text in bound_material_targets:
            return False
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid() and prim.IsA(UsdShade.Material):
            return True
        return "/Materials/" in path_text or "/Looks/" in path_text  # pragma: no cover

    def _cleanup_local_layer_refs() -> bool:
        root_layer = stage.GetRootLayer()
        bound_material_targets = _material_binding_targets()
        cleared_refs: list[str] = []
        localized_refs: list[str] = []
        missing_refs: list[str] = []

        def _walk_prim_spec(spec: Sdf.PrimSpec) -> None:
            refs = list(spec.referenceList.prependedItems)
            if refs:
                kept_refs: list[Sdf.Reference] = []
                changed = False
                for ref in refs:
                    asset_path = ref.assetPath
                    if (  # pragma: no cover - non-package layer reference
                        not asset_path
                        or _is_uri_ref(asset_path)
                        or Path(asset_path).suffix.lower()
                        not in {".usd", ".usda", ".usdc", ".usdz"}
                    ):
                        kept_refs.append(ref)
                        continue

                    try:
                        candidate_ref = Path(asset_path)
                        if candidate_ref.is_absolute():
                            resolved_ref = candidate_ref.resolve()
                            ref_exists = (
                                _is_under(resolved_ref, archive_root)
                                and resolved_ref.is_file()
                            )
                        else:
                            resolved_ref = (
                                package_usd.parent / candidate_ref
                            ).resolve()
                            ref_exists = (
                                _is_under(resolved_ref, archive_root)
                                and resolved_ref.is_file()
                            )
                    except (OSError, ValueError):  # pragma: no cover - bad Sdf path
                        ref_exists = False
                    if ref_exists:  # pragma: no cover - reference already packaged
                        kept_refs.append(ref)
                        continue

                    if _localize_upload_layer_ref(asset_path):
                        kept_refs.append(ref)
                        localized_refs.append(f"{spec.path}: {asset_path}")
                        continue

                    if _is_unbound_material_like_ref(spec.path, bound_material_targets):
                        cleared_refs.append(f"{spec.path}: {asset_path}")
                        changed = True
                        continue

                    missing_refs.append(f"{spec.path}: {asset_path}")
                    kept_refs.append(ref)

                if changed:
                    spec.referenceList.prependedItems = kept_refs

            for child in spec.nameChildren:
                _walk_prim_spec(child)

        for root_spec in root_layer.rootPrims:
            _walk_prim_spec(root_spec)

        max_layer_ref_records = _MAX_ERRORS_IN_PAYLOAD
        if localized_refs:
            context["usdz_layer_references_localized_count"] = len(localized_refs)
            context["usdz_layer_references_localized"] = localized_refs[
                :max_layer_ref_records
            ]
        if cleared_refs:
            context["usdz_layer_references_cleared_count"] = len(cleared_refs)
            context["usdz_layer_references_cleared"] = cleared_refs[
                :max_layer_ref_records
            ]
            logger.info(
                "Cleared %d unresolved unbound material layer references before USDZ "
                "packaging; sample: %s",
                len(cleared_refs),
                cleared_refs[:max_layer_ref_records],
            )
        if missing_refs:
            context["usdz_layer_references_missing_count"] = len(missing_refs)
            sampled_missing_refs = missing_refs[:max_layer_ref_records]
            context["usdz_layer_references_missing"] = sampled_missing_refs
            for missing_ref in sampled_missing_refs:
                _record_usdz_packaging_failure(
                    context,
                    f"Output USD contains missing local layer reference: {missing_ref}",
                    diagnostic=make_diagnostic(
                        "PACKAGE_MISSING_ARTIFACT",
                        severity="error",
                        stage="package",
                        message=(
                            "Output USD contains missing local layer reference: "
                            f"{missing_ref}"
                        ),
                        recommended_action=(
                            "Inspect the source USD dependencies or download "
                            "individual artifacts instead of the USDZ package."
                        ),
                        details={"reference": missing_ref},
                    ),
                )
            return False

        if localized_refs or cleared_refs:
            root_layer.Export(str(package_usd))
        return True

    rewritten = 0
    cleared_missing_mdl_source_assets: list[str] = []

    def _set_matching_authored_values(
        attr: Usd.Attribute,
        old_path: str,
        new_value: Sdf.AssetPath | str,
    ) -> None:
        """Rewrite matching authored opinions so hidden layers do not dangle."""

        def _value_for_authoring_layer(
            layer: Sdf.Layer,
        ) -> Sdf.AssetPath | str:
            authored_path = (
                new_value.path if isinstance(new_value, Sdf.AssetPath) else new_value
            )
            if (
                not authored_path
                or Path(authored_path).is_absolute()
                or _is_uri_ref(authored_path)
                or Ar.IsPackageRelativePath(authored_path)
                or not layer.realPath
            ):
                return new_value
            target = (package_usd.parent / authored_path).resolve()
            rebased = Path(
                os.path.relpath(target, start=Path(layer.realPath).resolve().parent)
            ).as_posix()
            if isinstance(new_value, Sdf.AssetPath):
                return Sdf.AssetPath(rebased)
            return rebased

        changed_layers: set[Sdf.Layer] = set()
        for spec in attr.GetPropertyStack():
            if not spec.layer.realPath or not _is_under(
                Path(spec.layer.realPath),
                package_workspace,
            ):
                continue
            authored_value = getattr(spec, "default", None)
            authored_path = (
                authored_value.path
                if isinstance(authored_value, Sdf.AssetPath)
                else authored_value
            )
            if authored_path != old_path:
                continue
            spec.default = _value_for_authoring_layer(spec.layer)
            changed_layers.add(spec.layer)
        if not changed_layers:
            attr.Set(new_value)
            return
        for layer in changed_layers:
            if not layer.anonymous:
                layer.Save()

    def _clear_matching_authored_values(
        attr: Usd.Attribute,
        old_path: str,
    ) -> bool:
        """Clear matching defaults in their authoring layers, not the edit target."""
        changed_layers: set[Sdf.Layer] = set()
        for spec in attr.GetPropertyStack():
            if not spec.layer.realPath or not _is_under(
                Path(spec.layer.realPath),
                package_workspace,
            ):
                continue
            authored_value = getattr(spec, "default", None)
            authored_path = (
                authored_value.path
                if isinstance(authored_value, Sdf.AssetPath)
                else authored_value
            )
            if authored_path != old_path:
                continue
            spec.ClearInfo("default")
            changed_layers.add(spec.layer)
        for layer in changed_layers:
            if not layer.anonymous:
                layer.Save()
        return bool(changed_layers)

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        # String/token rewrites are scoped to UsdShade.Shader inputs whose
        # name ends in ``_texture`` (Codex round-11 finding) so we never
        # mutate unrelated authored metadata that happens to be a string
        # ending in ``.png``. Asset-typed rewrites stay broad — the existing
        # OpenPBR / tiledimage write path produces them across the stage. Both
        # paths either require the original file to live under cache/textures
        # before rewriting, or to resolve under this session's upload bundle
        # before being localized there. Out-of-bundle refs remain visible to
        # portability checks.
        is_shader = prim.IsA(UsdShade.Shader)
        for attr in prim.GetAttributes():
            val = attr.Get()
            # Asset-typed in-bundle PNG path → rewrite to bundle-relative.
            if isinstance(val, Sdf.AssetPath) and val.path:
                old_path = val.path
                filename = _texture_filename(old_path)
                if source_extract_root is not None:
                    staged_member_ref = _rewritten_source_package_member_ref(
                        old_path,
                        attr=attr,
                        resolved_path=val.resolvedPath,
                    )
                    if (
                        staged_member_ref is not None
                        and Path(filename).suffix.lower()
                        not in package_texture_suffixes
                    ):
                        _set_matching_authored_values(
                            attr,
                            old_path,
                            Sdf.AssetPath(staged_member_ref),
                        )
                        rewritten += 1
                        continue
                if (
                    is_shader
                    and attr.GetName() == "info:mdl:sourceAsset"
                    and filename.lower().endswith(".mdl")
                ):  # pragma: no cover - MDL source asset cleanup branch
                    if (
                        Path(old_path).is_absolute()
                    ):  # pragma: no cover - absolute MDL refs are preserved
                        continue
                    src_resolved = _resolve_local_texture_ref(
                        old_path,
                        attr=attr,
                        resolved_path=val.resolvedPath,
                    )
                    if src_resolved is not None and not src_resolved.exists():
                        try:
                            if _clear_matching_authored_values(attr, old_path):
                                cleared_missing_mdl_source_assets.append(
                                    str(attr.GetPath())
                                )
                        except (
                            Exception
                        ) as err:  # pragma: no cover - USD attr clear failure
                            logger.warning(
                                "Failed to clear missing MDL source asset on %s: %s",
                                attr.GetPath(),
                                err,
                            )
                    continue
                if (
                    Path(filename).suffix.lower() not in package_texture_suffixes
                ):  # pragma: no cover - non-texture asset guard
                    continue
                new_path = None
                generated_texture = _generated_texture_for_reference(
                    old_path,
                    attr=attr,
                    resolved_path=val.resolvedPath,
                )
                if generated_texture is not None:
                    new_path = _texture_ref(generated_texture.name)
                elif _is_existing_package_texture(
                    old_path,
                    attr=attr,
                    resolved_path=val.resolvedPath,
                ):
                    if source_extract_root is not None:
                        # The exact layered-package writer preserves the
                        # source member layout. Prepared USD layers can flatten
                        # an existing member to ``source.usdz[member]``; rewrite
                        # that spelling back to the reconstructed tree.
                        new_path = _rewritten_source_package_member_ref(
                            old_path,
                            attr=attr,
                            resolved_path=val.resolvedPath,
                        )
                        if new_path is None:
                            continue
                    else:
                        new_path = _localized_package_texture_ref(
                            old_path,
                            attr=attr,
                            resolved_path=val.resolvedPath,
                        )
                else:
                    new_path = _localized_upload_texture_ref(old_path)
                if new_path:
                    _set_matching_authored_values(
                        attr,
                        old_path,
                        Sdf.AssetPath(new_path),
                    )
                    rewritten += 1
                continue
            # String/token-typed PNG path (MDL shaders can author texture
            # inputs as `string` / `token`) → rewrite the same way so a
            # downloaded USDZ resolves the file via the OpenPBR side's
            # asset-typed dependency on the same generated PNG. Only
            # `inputs:*_texture` attributes on Shader prims qualify, and
            # the *original* path must resolve to a file inside this
            # session's ``cache/textures`` (apply_textures is the only
            # writer there, and it only writes verified bundle-safe
            # files). A bare basename match is not enough — a shader
            # input pointing somewhere else on disk could collide with a
            # generated PNG by basename and the rewrite would silently
            # substitute the wrong texture (Codex round-15 finding).
            if not (
                isinstance(val, str) and val and is_shader
            ):  # pragma: no cover - non-shader string guard
                continue
            attr_name = attr.GetName()
            if not (
                attr_name.startswith("inputs:") and attr_name.endswith("_texture")
            ):  # pragma: no cover - unrelated shader input guard
                continue
            filename = _texture_filename(val)
            if (
                Path(filename).suffix.lower() not in package_texture_suffixes
            ):  # pragma: no cover - non-texture shader input guard
                continue
            new_path = None
            generated_texture = _generated_texture_for_reference(val, attr=attr)
            if generated_texture is not None:
                new_path = _texture_ref(generated_texture.name)
            elif _is_existing_package_texture(val, attr=attr):
                if source_extract_root is not None:
                    new_path = _rewritten_source_package_member_ref(
                        val,
                        attr=attr,
                    )
                    if new_path is None:
                        continue
                else:
                    new_path = _localized_package_texture_ref(val, attr=attr)
            else:
                new_path = _localized_upload_texture_ref(val)
            if new_path is None:
                continue
            try:
                _set_matching_authored_values(attr, val, new_path)
                rewritten += 1
            except Exception as err:  # pragma: no cover - USD attr set failure
                logger.warning(
                    "Failed to rewrite string texture path on %s: %s",
                    attr.GetPath(),
                    err,
                )

    if rewritten > 0 or cleared_missing_mdl_source_assets:
        stage.GetRootLayer().Export(str(package_usd))
        logger.info(
            "Rewrote %d texture paths to relative; cleared %d missing MDL source assets",
            rewritten,
            len(cleared_missing_mdl_source_assets),
        )
        if cleared_missing_mdl_source_assets:
            context["usdz_mdl_source_assets_cleared"] = (
                cleared_missing_mdl_source_assets
            )

    if not _cleanup_local_layer_refs():
        return None

    portability = validate_output_texture_portability(
        package_usd,
        bundle_root=archive_root,
    )
    context["usdz_source_portability"] = portability

    # A package-owned layer can resolve another package-owned file while still
    # authoring its temporary absolute path. The archive writer stores the
    # target under a relative member name, so retaining the authored absolute
    # spelling would make the USDZ dangle after session cleanup.
    #
    # Keep this guard behind texture portability validation so
    # texture-specific failures retain their established diagnostics.
    # Dependencies outside the package workspace remain the responsibility of
    # the existing dependency/writer containment gates below.
    absolute_dependencies: list[dict[str, str]] = []
    if portability.get("portable", False):
        for dependency_layer in stage.GetUsedLayers():
            layer_path = str(getattr(dependency_layer, "realPath", "") or "")
            if not layer_path:
                continue
            with suppress(OSError, ValueError):
                if not _is_under(
                    Path(layer_path).resolve(), archive_root
                ):  # pragma: no cover - only package-owned layers are scanned
                    continue
                for authored_path in _layer_authored_asset_paths(dependency_layer):
                    if not Path(authored_path).is_absolute():
                        continue
                    authored_dependency = Path(authored_path).resolve()
                    if _is_under(authored_dependency, archive_root):
                        absolute_dependencies.append(
                            {
                                "layer": layer_path,
                                "path": authored_path,
                            }
                        )
    if absolute_dependencies:
        sampled_dependencies = absolute_dependencies[:_MAX_ERRORS_IN_PAYLOAD]
        context["usdz_absolute_dependency_count"] = len(absolute_dependencies)
        context["usdz_absolute_dependencies"] = sampled_dependencies
        message = (
            "USDZ retains absolute authored dependencies; "
            "refusing to package temporary worker paths."
        )
        # Texture portability passed, but the package is still not portable.
        # Leaving portable=True here would make the failure manifest contradict
        # the recorded packaging error.
        portability["portable"] = False
        portability.setdefault("diagnostics", []).append(
            make_diagnostic(
                "PACKAGE_NON_RELATIVE_PATH",
                severity="error",
                stage="package",
                message=message,
                recommended_action=(
                    "Author sublayers, references, payloads, clips, and asset "
                    "attributes with paths relative to their USD layer."
                ),
                details={
                    "absolute_dependency_count": len(absolute_dependencies),
                    "absolute_dependencies": sampled_dependencies,
                },
            )
        )
        _record_usdz_packaging_failure(
            context,
            message,
            diagnostic=make_diagnostic(
                "PACKAGE_NON_RELATIVE_PATH",
                severity="error",
                stage="package",
                message=message,
                recommended_action=(
                    "Author sublayers, references, payloads, clips, and asset "
                    "attributes with paths relative to their USD layer."
                ),
                details={
                    "absolute_dependency_count": len(absolute_dependencies),
                    "absolute_dependencies": sampled_dependencies,
                },
            ),
        )
        return None

    if source_extract_root is not None:
        dependency_layers, dependency_assets, unresolved_dependencies = (
            UsdUtils.ComputeAllDependencies(str(package_usd))
        )
        missing_dependencies: list[str] = []
        non_relative_dependencies: list[str] = []
        for path in unresolved_dependencies:
            dependency = str(path)
            if (
                _is_uri_ref(dependency)
                or Ar.IsPackageRelativePath(dependency)
                or Path(dependency).is_absolute()
            ):
                non_relative_dependencies.append(dependency)
            else:
                missing_dependencies.append(dependency)

        dependency_paths: list[str] = []
        for layer in dependency_layers:
            layer_path = getattr(layer, "realPath", "") or getattr(
                layer,
                "identifier",
                "",
            )
            if layer_path:
                dependency_paths.append(str(layer_path))
        dependency_paths.extend(str(path) for path in dependency_assets)

        for dependency in dependency_paths:
            if (  # pragma: no cover - USD reports these as unresolved above
                _is_uri_ref(dependency) or Ar.IsPackageRelativePath(dependency)
            ):
                non_relative_dependencies.append(dependency)
                continue
            dependency_path = Path(dependency)
            if not dependency_path.is_absolute():  # pragma: no cover - USD resolves
                dependency_path = package_usd.parent / dependency_path
            try:
                dependency_path = dependency_path.resolve()
            except (OSError, ValueError):  # pragma: no cover - defensive path guard
                missing_dependencies.append(dependency)
                continue
            if (
                not dependency_path.is_file()
            ):  # pragma: no cover - USD reports unresolved
                missing_dependencies.append(dependency)
            elif not _is_under(dependency_path, archive_root):
                non_relative_dependencies.append(dependency)

        invalid_dependencies = [
            *missing_dependencies,
            *non_relative_dependencies,
        ]
        if invalid_dependencies:
            sampled_dependencies = sorted(set(invalid_dependencies))[
                :_MAX_ERRORS_IN_PAYLOAD
            ]
            message = (
                "Reconstructed USDZ has unresolved or external dependencies: "
                + ", ".join(sampled_dependencies)
            )
            diagnostic = make_diagnostic(
                "PACKAGE_MISSING_ARTIFACT",
                severity="error",
                stage="package",
                message=message,
                recommended_action=(
                    "Include every payload, reference, sublayer, and asset "
                    "inside the uploaded USDZ, then retry."
                ),
                details={
                    "invalid_dependency_count": len(set(invalid_dependencies)),
                    "invalid_dependencies": sampled_dependencies,
                },
            )
            portability["portable"] = False
            portability["non_relative_texture_paths"] = sorted(
                {
                    *portability.get("non_relative_texture_paths", []),
                    *non_relative_dependencies,
                }
            )
            portability["missing_texture_paths"] = sorted(
                {
                    *portability.get("missing_texture_paths", []),
                    *missing_dependencies,
                }
            )
            portability.setdefault("diagnostics", []).append(diagnostic)
            _record_usdz_packaging_failure(
                context,
                message,
                diagnostic=diagnostic,
            )
            return None

    if package_usd.resolve() == output_usd.resolve():
        context["output_portability"] = portability
    else:
        # The apply artifact remains the root-only compatibility output for a
        # layered upload; the reconstructed root is intentionally internal and
        # feeds both rendering and the self-contained USDZ. Do not label the
        # downloadable root-only USD with the reconstructed stage's result.
        apply_portability = validate_output_texture_portability(
            output_usd,
            bundle_root=package_workspace,
        )
        apply_portability["portable"] = False
        apply_portability["diagnostics"] = [
            make_diagnostic(
                "OUTPUT_LAYERED_USD_REQUIRES_USDZ",
                severity="warning",
                stage="package",
                message=(
                    "The individual USD apply artifact is the root layer of "
                    "a layered upload; use the self-contained USDZ artifact."
                ),
                recommended_action=(
                    "Download output_usdz for the complete layered asset."
                ),
                details={
                    "output_usd": str(output_usd),
                    "reconstructed_root": str(package_usd),
                },
            )
        ]
        context["output_portability"] = apply_portability
    if not portability.get("portable", False):
        diagnostics = portability.get("diagnostics", [])
        for diagnostic in diagnostics:
            _record_usdz_packaging_failure(
                context,
                diagnostic.get(
                    "message",
                    "Output USD contains non-portable texture references",
                ),
                diagnostic=diagnostic,
            )
        if (
            not diagnostics
        ):  # pragma: no cover - validator normally supplies diagnostics
            _record_usdz_packaging_failure(
                context,
                "Output USD contains non-portable texture references",
            )
        return None

    # Package into USDZ
    usdz_path = output_usd.parent / "textured_output.usdz"

    # Remove stale output so a failed writer cannot leave a partial file that
    # the download endpoint would serve.
    usdz_path.unlink(missing_ok=True)

    try:
        # Preserve authored paths exactly for both layered and single-layer
        # outputs. CreateNewUsdzPackage rewrites Asset-typed references while
        # leaving equivalent String/Token shader inputs untouched, which can
        # make the latter dangle. The cache root is the common relative anchor
        # for a single-layer apply output; a reconstructed source tree is the
        # anchor for an uploaded layered USDZ.
        root_member = package_usd.resolve().relative_to(archive_root).as_posix()
        archive_members: list[tuple[Path, str]] = []
        if source_extract_root is not None:
            candidate_members = sorted(archive_root.rglob("*"))
        else:
            layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
                str(package_usd)
            )
            if unresolved:
                unresolved_sample = [str(path) for path in unresolved[:25]]
                raise RuntimeError(
                    "Output USD has unresolved package dependencies: "
                    + ", ".join(unresolved_sample)
                )
            required_members: set[Path] = set()
            for layer in layers:
                layer_path = getattr(layer, "realPath", "") or getattr(
                    layer,
                    "identifier",
                    "",
                )
                if layer_path and not _is_uri_ref(str(layer_path)):
                    required_members.add(Path(layer_path))
            for asset in assets:
                asset_path = str(asset)
                if _is_uri_ref(asset_path):
                    raise RuntimeError(
                        f"Output USD has external package dependency: {asset_path}"
                    )
                asset_member = Path(asset_path)
                if not asset_member.is_absolute():  # pragma: no cover - USD resolves
                    asset_member = package_usd.parent / asset_member
                required_members.add(asset_member)
            # ComputeAllDependencies follows typed USD assets. Include
            # packageable String/Token shader inputs using the same
            # authoring-layer-aware collector as the portability gate,
            # including values in unselected variant branches.
            texture_references = _output_texture_references(package_usd) or []
            for reference in texture_references:
                if reference.get("value_type") != "string":
                    continue
                resolved_value = str(reference.get("resolved_path") or "")
                if not resolved_value or _is_uri_ref(resolved_value):
                    continue
                resolved = Path(resolved_value).resolve()
                if not resolved.is_file():
                    continue
                if not _is_under(resolved, archive_root):
                    raise RuntimeError(
                        "Output USD texture dependency is outside the package workspace"
                    )
                required_members.add(resolved)
            candidate_members = sorted(required_members)

        package_usd_resolved = package_usd.resolve()
        usdz_path_resolved = usdz_path.resolve()
        archive_member_names = {root_member}
        for member in candidate_members:
            if not member.is_file():
                if source_extract_root is None:
                    raise RuntimeError(
                        f"Output USD dependency is not a local file: {member}"
                    )
                continue
            lexical_member = Path(os.path.abspath(member))
            try:
                member_name = lexical_member.relative_to(archive_root).as_posix()
                resolved_member = lexical_member.resolve(strict=True)
                resolved_member.relative_to(archive_root)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "Output USD dependency is outside the package workspace"
                ) from exc
            if resolved_member in {package_usd_resolved, usdz_path_resolved}:
                continue
            if member_name in archive_member_names:
                continue  # pragma: no cover - duplicate dependency spelling
            archive_member_names.add(member_name)
            archive_members.append((resolved_member, member_name))

        # Enumerate before CreateNew: ZipFileWriter creates its aligned
        # temporary output beside the destination, which must never be
        # discovered as an input member. Some supported USD Python builds omit
        # that low-level writer. Reuse WU's aligned writer in that case so the
        # already-computed member names remain byte-for-byte portable.
        zip_writer = getattr(Usd, "ZipFileWriter", None)
        if zip_writer is None:
            import tempfile

            with tempfile.TemporaryDirectory(
                dir=usdz_path.parent,
                prefix=".texture-usdz-",
            ) as staging_value:
                staging_dir = Path(staging_value)
                member_order = (root_member, *(name for _, name in archive_members))
                for source, member_name in (
                    (package_usd, root_member),
                    *archive_members,
                ):
                    destination = staging_dir / member_name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                write_usdz_package_from_directory(
                    staging_dir,
                    Path(root_member),
                    usdz_path,
                    member_order=member_order,
                )
            written_path = str(usdz_path)
        else:
            with zip_writer.CreateNew(str(usdz_path)) as writer:
                if not writer.AddFile(str(package_usd), root_member):
                    raise RuntimeError(f"Failed to add USDZ root member: {root_member}")
                for member, member_name in archive_members:
                    if not writer.AddFile(str(member), member_name):
                        raise RuntimeError(f"Failed to add USDZ member: {member_name}")
                written_path = writer.Save()
    except Exception as exc:
        usdz_path.unlink(missing_ok=True)
        _record_usdz_packaging_failure(context, f"Failed to create USDZ package: {exc}")
        logger.exception("Failed to create USDZ package")
        return None

    if written_path and usdz_path.exists():
        # Validate the output is actually a ZIP archive (USDZ spec).
        if not zipfile.is_zipfile(
            usdz_path
        ):  # pragma: no cover - USDZ package corruption guard
            message = f"Usd.ZipFileWriter wrote non-ZIP data to {usdz_path}"
            _record_usdz_packaging_failure(context, message)
            logger.warning("%s, removing", message)
            usdz_path.unlink(missing_ok=True)
            return None

        size_mb = usdz_path.stat().st_size / (1024 * 1024)
        logger.info("Packaged USDZ: %s (%.1f MB)", usdz_path, size_mb)
        if source_extract_root is not None:
            # RenderOutputTask flattens this reconstructed stage into an
            # anonymous layer. String/Token inputs do not receive USD Asset
            # path anchoring during flattening, so give the render-only stage
            # absolute local opinions after the portable USDZ snapshot is
            # complete. The downloadable package retains its relative values.
            render_string_localizations = 0
            localized_render_specs: set[tuple[str, str]] = set()
            stage.SetEditTarget(stage.GetRootLayer())
            for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdShade.Shader):
                    continue
                for attr in prim.GetAttributes():
                    value = attr.Get()
                    if not (
                        isinstance(value, str)
                        and value
                        and attr.GetName().startswith("inputs:")
                        and attr.GetName().endswith("_texture")
                        and Path(_texture_filename(value)).suffix.lower()
                        in package_texture_suffixes
                    ):
                        continue
                    resolved_string_texture = _resolve_local_texture_ref(
                        value,
                        attr=attr,
                    )
                    if (
                        resolved_string_texture is None
                        or not resolved_string_texture.is_file()
                        or not _is_under(resolved_string_texture, archive_root)
                    ):
                        continue
                    matching_spec_keys = {
                        (spec.layer.identifier, str(spec.path))
                        for spec in attr.GetPropertyStack()
                        if spec.layer.realPath
                        and _is_under(
                            Path(spec.layer.realPath),
                            package_workspace,
                        )
                        and getattr(spec, "default", None) == value
                    }
                    if (
                        matching_spec_keys
                        and matching_spec_keys <= localized_render_specs
                    ):
                        continue
                    if (
                        prim.IsInstanceProxy() and not matching_spec_keys
                    ):  # pragma: no cover - unsafe proxy property-stack guard
                        continue
                    _set_matching_authored_values(
                        attr,
                        value,
                        str(resolved_string_texture),
                    )
                    localized_render_specs.update(matching_spec_keys)
                    render_string_localizations += 1
            if render_string_localizations:
                stage.GetRootLayer().Save()
                context["render_string_texture_localizations"] = (
                    render_string_localizations
                )
        context.pop("usdz_packaging_failed", None)
        context.pop("usdz_packaging_error", None)
        return str(usdz_path)

    # Clean up any partial file left behind on failure
    usdz_path.unlink(missing_ok=True)
    _record_usdz_packaging_failure(context, "Failed to create USDZ package")
    logger.warning("Failed to create USDZ package")
    return None


def _apply_textures_stats_summary(context: dict[str, Any]) -> dict[str, Any]:
    """Distil ``context['apply_textures_stats']`` into a flat stats dict.

    Both the per-step (``_extract_step_stats``) and final
    (``_extract_final_stats``) summaries need the same shape, so callers can see
    MDL override / clear / localize counts, UsdPreviewSurface fallback rewrites,
    and a human-readable ``warnings`` list whether they look at /status mid-run
    or at /results after completion.
    """
    out: dict[str, Any] = {}
    apply_stats = context.get("apply_textures_stats") or {}
    if "mdl_inputs_overridden" in apply_stats:
        out["mdl_inputs_overridden"] = apply_stats["mdl_inputs_overridden"]

    cleared = apply_stats.get("mdl_inputs_cleared") or []
    localized = apply_stats.get("mdl_inputs_localized") or []
    if cleared:
        out["mdl_inputs_cleared"] = list(cleared)
    if localized:
        out["mdl_inputs_localized"] = list(localized)

    preview_overridden = apply_stats.get("preview_texture_inputs_overridden") or []
    if preview_overridden:
        out["preview_texture_inputs_overridden"] = list(preview_overridden)

    if cleared:
        out["warnings"] = [
            "Cleared MDL texture inputs that could not be bundled (unbundleable "
            "URI refs or unresolvable local paths). Affected materials/inputs: "
            + ", ".join(cleared)
        ]
    return out


# Cap persisted per-unit error payloads. In per-prim mode with a backend-
# wide outage, the unbounded list could be one record per prim (thousands)
# in session.json, event_log.jsonl, SSE payloads, and /results. Counts +
# bounded sample preserve the diagnostic value while keeping persisted
# artifacts small during the very incidents we want diagnostics for.
_MAX_ERRORS_IN_PAYLOAD = 25
_MAX_ERROR_MESSAGE_CHARS = 500
_MAX_RENDER_STATS_ITEMS = 25


def _record_usdz_packaging_failure(
    context: dict[str, Any],
    message: str,
    *,
    diagnostic: dict[str, Any] | None = None,
) -> None:
    from texture_agent.functions.artifact_manifest import make_diagnostic

    context["usdz_packaging_failed"] = True
    context["usdz_packaging_error"] = message
    diagnostics = context.setdefault("package_diagnostics", [])
    diagnostics.append(
        diagnostic
        or make_diagnostic(
            "PACKAGE_MISSING_ARTIFACT",
            severity="error",
            stage="package",
            message=message,
            recommended_action=(
                "Inspect artifacts_manifest.json and download individual artifacts "
                "instead of the USDZ package."
            ),
        )
    )


def _artifact_download_urls(session_id: str) -> dict[str, str]:
    return {
        "materials": f"/artifacts/{session_id}/materials",
        "textures": f"/artifacts/{session_id}/textures",
        "output": f"/artifacts/{session_id}/output",
        "renders": f"/artifacts/{session_id}/renders",
        "manifest": f"/artifacts/{session_id}/manifest",
    }


def _artifact_manifest_status(context: dict[str, Any]) -> str:
    if (
        context.get("usdz_packaging_failed")
        or context.get("generate_textures_failed_count")
        or context.get("blend_textures_failed_count")
        or context.get("texture_execution_status") not in (None, "completed")
    ):
        return "partial"
    return "completed"


def _write_service_artifact_manifest(
    context: dict[str, Any],
    *,
    status: str,
    service_urls: dict[str, str],
    duration_seconds: int | None = None,
) -> str | None:
    from texture_agent.functions.artifact_manifest import (
        build_artifacts_manifest,
        write_artifacts_manifest,
    )

    try:
        payload = build_artifacts_manifest(
            context,
            status=status,
            service_urls=service_urls,
            duration_seconds=duration_seconds,
        )
        sanitized_payload = sanitize_payload(
            payload,
            service_config.session_storage_path,
        )
        manifest_path = write_artifacts_manifest(
            context,
            status=status,
            payload=sanitized_payload,
        )
        return str(manifest_path)
    except Exception as err:
        logger.warning("Failed to write artifact manifest: %s", err)
        return None


def _truncate_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cap the error list size and truncate per-record messages.

    The full count is exposed via the sibling ``*_failed_count`` /
    ``textures_failed`` keys, so dropping the tail here doesn't lose the
    "how bad is it" signal -- only the per-material detail. The tail is
    still in container logs (``logger.exception``) for the small fraction
    of incidents where deeper diagnostics are needed.
    """
    capped = errors[:_MAX_ERRORS_IN_PAYLOAD]
    out: list[dict[str, Any]] = []
    for record in capped:
        message = record.get("message", "")
        if isinstance(message, str) and len(message) > _MAX_ERROR_MESSAGE_CHARS:
            message = message[:_MAX_ERROR_MESSAGE_CHARS] + "...(truncated)"
        out.append({**record, "message": message})
    return out


def _redact_diagnostics_for_stats(diagnostics: Any) -> list[dict[str, Any]]:
    """Redact credentials before diagnostics reach persisted/public stats."""
    from texture_agent.functions.artifact_manifest import redact_sensitive

    redacted = redact_sensitive(diagnostics)
    if not isinstance(redacted, list):
        return []
    records = [item for item in redacted if isinstance(item, dict)]
    return _truncate_errors(records)


def _truncate_render_stats_items(items: Any) -> list[Any]:
    if not isinstance(items, list):
        return []
    return items[:_MAX_RENDER_STATS_ITEMS]


def _render_summary_stats(render_stats: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key in (
        "backend",
        "evidence_classification",
        "production_visual_evidence",
        "texture_detail_display_color_bakes",
        "texture_detail_package_texture_localizations",
        "texture_detail_uv_texture_fallbacks",
        "textured_preview_fallbacks",
    ):
        if key in render_stats:
            stats[key] = render_stats[key]
    return stats


def _projection_backend_stats(context: dict[str, Any]) -> dict[str, Any]:
    """Return projection-backend metadata for status/results payloads."""
    from texture_agent.functions.artifact_manifest import redact_sensitive

    raw_records = context.get("projection_backend_results")
    records = raw_records if isinstance(raw_records, dict) else {}
    raw_diagnostics = context.get("generate_textures_diagnostics")
    diagnostics = raw_diagnostics if isinstance(raw_diagnostics, list) else []
    if not records and not diagnostics:
        return {}

    metadata_by_unit = {
        key: value.get("metadata", {})
        for key, value in records.items()
        if isinstance(value, dict)
    }
    map_counts = {
        key: len(maps)
        for key, value in records.items()
        if isinstance(value, dict)
        for maps in [value.get("maps", {}) or {}]
        if isinstance(maps, dict)
    }
    warnings = [
        item
        for item in diagnostics
        if isinstance(item, dict) and item.get("severity") == "warning"
    ]
    redacted_diagnostics = redact_sensitive(diagnostics)
    if not isinstance(
        redacted_diagnostics, list
    ):  # pragma: no cover - redactor shape guard
        redacted_diagnostics = []
    diagnostic_records = [
        item for item in redacted_diagnostics if isinstance(item, dict)
    ]
    redacted_warnings = redact_sensitive(warnings)
    if not isinstance(
        redacted_warnings, list
    ):  # pragma: no cover - redactor shape guard
        redacted_warnings = []
    warning_records = [item for item in redacted_warnings if isinstance(item, dict)]
    return {
        "projection_backend_units": len(records),
        "projection_backend_map_counts": map_counts,
        "projection_backend_metadata": sanitize_payload(
            redact_sensitive(metadata_by_unit)
        ),
        "projection_backend_diagnostics": _truncate_errors(diagnostic_records),
        "projection_backend_warnings": _truncate_errors(warning_records),
    }


def _extract_step_stats(step_name: str, context: dict[str, Any]) -> dict:
    """Extract statistics from context after a step completes.

    For ``generate_textures`` and ``blend_textures``, propagate the
    structured per-unit error list and failure count surfaced by those
    tasks. Without this, partial-failure runs report ``state=completed``
    in SSE / ``/status`` with no diagnostic for which materials failed
    or why, preventing silent completion with no diagnostics.
    """
    stats: dict[str, Any] = {}

    if step_name == "discover_materials":
        materials = context.get("discovered_materials", [])
        stats["materials_found"] = len(materials)

    elif step_name == "plan_textures":
        plan = context.get("texture_plan")
        if plan is not None:
            stats["texture_plan_available"] = True
            stats["texture_plan_state"] = str(plan.decision.state)
            stats["texture_plan_execution_allowed"] = plan.decision.execution_allowed
            stats["texture_plan_counts"] = plan.counts.model_dump()
            stats["texture_plan_limits"] = plan.limits.model_dump()

    elif step_name == "prepare_uvs":
        uv_preparation = context.get("uv_preparation") or {}
        stats["uv_report_available"] = bool(uv_preparation.get("uv_report_path"))
        if uv_preparation.get("uv_report_path"):
            stats["uv_report_path"] = uv_preparation["uv_report_path"]
        for key in ("backend", "generated", "fixed_interpolation", "normalized"):
            if key in uv_preparation:
                stats[f"uv_{key}"] = uv_preparation[key]

    elif step_name == "generate_textures":
        generated = context.get("generated_textures", {})
        errors = context.get("generate_textures_errors", [])
        stats["textures_generated"] = len(generated)
        stats["textures_failed"] = context.get(
            "generate_textures_failed_count", len(errors)
        )
        if errors:
            stats["errors"] = _truncate_errors(errors)
        execution = context.get("texture_execution")
        if isinstance(execution, dict):
            stats["texture_execution_status"] = execution.get("status")
            stats["selected_units"] = len(execution.get("requested_unit_ids", []))
            stats["accepted_units"] = len(execution.get("accepted_unit_ids", []))
            stats["remaining_units"] = len(execution.get("remaining_unit_ids", []))
            stats["cache_hits"] = len(execution.get("cache_hit_unit_ids", []))
        stats.update(_projection_backend_stats(context))

    elif step_name == "blend_textures":
        blended = context.get("blended_textures", {})
        errors = context.get("blend_textures_errors", [])
        stats["textures_blended"] = len(blended)
        stats["textures_failed"] = context.get(
            "blend_textures_failed_count", len(errors)
        )
        if errors:
            stats["errors"] = _truncate_errors(errors)
        diagnostics = context.get("blend_textures_diagnostics", [])
        if diagnostics:
            stats["diagnostics"] = _redact_diagnostics_for_stats(diagnostics)

    elif step_name == "apply_textures":
        output_paths = context.get("output_usd_paths", [])
        stats["output_usd_count"] = len(output_paths)
        stats.update(_apply_textures_stats_summary(context))

    elif step_name == "render":
        rendered = context.get("rendered_image_paths", [])
        stats["renders_count"] = len(rendered)
        render_stats = context.get("render_stats") or {}
        if render_stats:
            stats["render_available"] = bool(render_stats.get("render_available"))
            stats.update(_render_summary_stats(render_stats))
            stats["camera_paths"] = _truncate_render_stats_items(
                render_stats.get("camera_paths")
            )
            stats["focus_cameras"] = _truncate_render_stats_items(
                render_stats.get("focus_cameras")
            )
        errors = context.get("render_errors", [])
        if errors:
            stats["errors"] = _truncate_errors(errors)
        diagnostics = context.get("render_diagnostics", [])
        if diagnostics:
            stats["diagnostics"] = _truncate_errors(diagnostics)

    return stats


def _extract_final_stats(context: dict[str, Any], session_dir: Path) -> dict[str, Any]:
    """Extract final pipeline statistics from context and files.

    Includes structured per-unit failure records when generate/blend
    completed below the threshold gate -- without this, a partial-failure
    run that completes (default threshold=1.0 + 1 success + N failures)
    looks identical to a clean run on ``GET /result/{session_id}`` after
    the SSE snapshot has been GC'd, leaving non-SSE consumers without the
    diagnostics this MR adds.
    """
    stats: dict[str, Any] = {
        "materials_found": len(context.get("discovered_materials", [])),
        "textures_generated": len(context.get("generated_textures", {})),
        "output_usd_count": len(context.get("output_usd_paths", [])),
        "renders_count": len(context.get("rendered_image_paths", [])),
    }
    plan = context.get("texture_plan")
    if plan is not None:
        stats.update(
            {
                "texture_plan_available": True,
                "texture_plan_state": str(plan.decision.state),
                "texture_plan_execution_allowed": plan.decision.execution_allowed,
                "texture_plan_counts": plan.counts.model_dump(),
                "texture_plan_limits": plan.limits.model_dump(),
            }
        )
    execution = context.get("texture_execution")
    if isinstance(execution, dict):
        stats.update(
            {
                "texture_execution_status": execution.get("status"),
                "selected_unit_count": len(execution.get("requested_unit_ids", [])),
                "accepted_unit_count": len(execution.get("accepted_unit_ids", [])),
                "remaining_unit_count": len(execution.get("remaining_unit_ids", [])),
                "cache_hit_count": len(execution.get("cache_hit_unit_ids", [])),
            }
        )

    # Fallback: count files if context stats are empty
    cache_dir = session_dir / "cache"

    if stats["textures_generated"] == 0:
        textures_dir = cache_dir / "textures"
        if textures_dir.exists():
            stats["textures_generated"] = len(list(textures_dir.glob("*.png")))

    if stats["output_usd_count"] == 0:
        output_dir = cache_dir / "output"
        if output_dir.exists():
            usd_files = list(output_dir.glob("*.usd")) + list(output_dir.glob("*.usda"))
            stats["output_usd_count"] = len(usd_files)

    if stats["renders_count"] == 0:
        renders_dir = cache_dir / "renders"
        if renders_dir.exists():
            stats["renders_count"] = len(list(renders_dir.glob("*.png")))

    render_stats = context.get("render_stats") or {}
    stats["render_available"] = bool(
        render_stats.get("render_available") or stats["renders_count"]
    )
    if render_stats:
        stats.update(_render_summary_stats(render_stats))
        stats["render_camera_paths"] = _truncate_render_stats_items(
            render_stats.get("camera_paths")
        )
        stats["render_focus_cameras"] = _truncate_render_stats_items(
            render_stats.get("focus_cameras")
        )
    render_errors = context.get("render_errors", [])
    if render_errors:
        stats.setdefault("errors", {})["render"] = _truncate_errors(render_errors)
    render_diagnostics = context.get("render_diagnostics", [])
    if render_diagnostics:
        stats.setdefault("diagnostics", {})["render"] = _truncate_errors(
            render_diagnostics
        )
    blend_diagnostics = context.get("blend_textures_diagnostics", [])
    if blend_diagnostics:
        stats.setdefault("diagnostics", {})["blend_textures"] = (
            _redact_diagnostics_for_stats(blend_diagnostics)
        )

    # Persist apply_textures MDL override/clear/localize counts and the
    # `warnings` list so /results consumers see the same signal as /status.
    stats.update(_apply_textures_stats_summary(context))

    cleared_layer_refs = context.get("usdz_layer_references_cleared") or []
    if cleared_layer_refs:
        cleared_layer_refs_count = int(
            context.get("usdz_layer_references_cleared_count", len(cleared_layer_refs))
        )
        stats["usdz_layer_references_cleared"] = cleared_layer_refs
        stats["usdz_layer_references_cleared_count"] = cleared_layer_refs_count
        omitted = cleared_layer_refs_count - len(cleared_layer_refs)
        suffix = f", ... and {omitted} more" if omitted > 0 else ""
        stats.setdefault("warnings", []).append(
            "Cleared unresolved unbound material layer references before USDZ "
            f"packaging ({cleared_layer_refs_count} total): "
            + ", ".join(cleared_layer_refs)
            + suffix
        )

    localized_layer_refs = context.get("usdz_layer_references_localized") or []
    if localized_layer_refs:  # pragma: no cover - optional upload layer reporting
        stats["usdz_layer_references_localized"] = localized_layer_refs
        stats["usdz_layer_references_localized_count"] = int(
            context.get(
                "usdz_layer_references_localized_count",
                len(localized_layer_refs),
            )
        )

    if context.get("usdz_packaging_failed"):
        stats["package_status"] = "failed"
        stats["usdz_packaging_failed"] = True
        diagnostics = context.get("package_diagnostics") or []
        if diagnostics:
            stats["package_diagnostics"] = diagnostics
        message = context.get("usdz_packaging_error") or "USDZ packaging failed"
        stats.setdefault("warnings", []).append(
            f"{message}. The textured USD output is still available, but the "
            "self-contained USDZ artifact was not produced."
        )
    elif context.get("output_usdz_path"):
        stats["package_status"] = "succeeded"
        stats["output_usdz_available"] = True
    elif context.get("output_usd_paths"):
        stats["package_status"] = "not_available"
    elif stats["output_usd_count"]:
        stats["package_status"] = "not_available"

    uv_preparation = context.get("uv_preparation") or {}
    if uv_preparation.get("uv_report_path"):
        stats["uv_report_available"] = True
        stats["uv_report_path"] = uv_preparation["uv_report_path"]

    manifest_path = context.get("artifacts_manifest_path")
    if manifest_path:
        stats["manifest_available"] = True
        stats["manifest_path"] = manifest_path

    # Generate/blend partial-failure surfacing: per-step counts plus a
    # disjoint sum so an auth-issue gen failure isn't hidden when blend
    # also drops a unit.
    gen_failed = context.get("generate_textures_failed_count", 0)
    blend_failed = context.get("blend_textures_failed_count", 0)
    if gen_failed:
        stats["textures_generated_failed"] = gen_failed
        gen_errors = context.get("generate_textures_errors")
        if gen_errors:
            stats.setdefault("errors", {})["generate_textures"] = _truncate_errors(
                gen_errors
            )
    if blend_failed:
        stats["textures_blended_failed"] = blend_failed
        blend_errors = context.get("blend_textures_errors")
        if blend_errors:
            stats.setdefault("errors", {})["blend_textures"] = _truncate_errors(
                blend_errors
            )
    # Total is the sum: gen failures and blend failures cover disjoint
    # units (a unit either failed gen, OR was generated and failed
    # blend, OR succeeded both). Without this an auth-issue gen failure
    # followed by a downstream blend failure would hide the gen count
    # behind the blend count -- losing the "the backend is broken"
    # signal that ``textures_failed`` exists to surface.
    if gen_failed or blend_failed:
        stats["textures_failed"] = gen_failed + blend_failed

    stats.update(_projection_backend_stats(context))

    return stats


def _get_step_validation_error(
    step_name: str,
    step_stats: dict[str, Any],
    planned_steps: list[str],
    context: dict[str, Any] | None = None,
) -> str | None:
    """Return a terminal validation error for a completed step, if any.

    Some task implementations can legitimately "complete" while doing no useful
    work. The service should convert those cases into failed sessions instead of
    reporting a false-positive success.

    ``context`` is the live pipeline context; when provided, the
    ``apply_textures`` empty-output message is enriched with the upstream
    cause (no textures generated / no textures blended) so the customer-
    visible failure points at the real root cause rather than the
    last-step symptom.
    """
    try:
        step_index = planned_steps.index(step_name)
    except ValueError:  # pragma: no cover - unknown step validation helper guard
        downstream_steps: list[str] = []
    else:
        downstream_steps = planned_steps[step_index + 1 :]

    if step_name == "discover_materials":
        materials_found = step_stats.get("materials_found", 0)
        if materials_found == 0 and downstream_steps:
            return (
                "No discoverable materials were found in the uploaded USD. "
                "Texture generation requires a USD with bound materials."
            )

    if step_name == "apply_textures":
        output_usd_count = step_stats.get("output_usd_count", 0)
        if output_usd_count == 0:
            base = (
                "Texture application produced no output USD files. "
                "The pipeline cannot be reported as completed."
            )
            if context is None:
                return base
            generated = context.get("generated_textures", {})
            blended = context.get("blended_textures", {})
            gen_errors = context.get("generate_textures_errors", [])
            blend_errors = context.get("blend_textures_errors", [])
            if not generated:
                cause = (
                    f"upstream generate_textures produced 0 textures "
                    f"({len(gen_errors)} per-material failure(s))"
                    if gen_errors
                    else "upstream generate_textures produced 0 textures"
                )
                return f"{base} Cause: {cause}."
            if not blended:
                cause = (
                    f"upstream blend_textures produced 0 textures "
                    f"({len(blend_errors)} per-material failure(s))"
                    if blend_errors
                    else "upstream blend_textures produced 0 textures"
                )
                return f"{base} Cause: {cause}."
            return base  # pragma: no cover - no upstream context supplied

    return None


async def execute_pipeline_async(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager: Any,
    only_steps: list[str] | None = None,
    skip_steps: list[str] | None = None,
    acquire_worker_lock: bool = True,
    worker_owner_token: str | None = None,
    on_artifacts_synced: Callable[[], Coroutine[Any, Any, None]] | None = None,
) -> None:
    """Execute texture pipeline by running each task in a thread.

    Emits ProgressEvent for each step start/completion, enabling
    real-time SSE streaming to clients.

    Args:
        session_id: Session identifier
        config_dict: Pipeline configuration dict
        session_manager: SessionManager instance
        only_steps: If set, run only these steps
        skip_steps: Steps to skip
        acquire_worker_lock: If False, caller already reserved the cross-process
            worker lock and will release it after registry cleanup.
        worker_owner_token: Shared-store reservation owner token for caller-owned
            locks when acquire_worker_lock is False.
        on_artifacts_synced: Optional finalization callback invoked after every
            pipeline artifact is durable, but before success is published.
    """
    logger.info(f"Pipeline execution started for {session_id[:8]}...")

    event_bus = get_event_bus()
    session_dir = session_manager.get_session_dir(session_id)
    lock_context = getattr(session_manager, "worker_lock", None)
    worker_lock = (
        lock_context(session_id)
        if acquire_worker_lock and lock_context is not None
        else nullcontext()
    )

    with worker_lock as acquired_worker_lock:
        owner_token = worker_owner_token or getattr(
            acquired_worker_lock,
            "_wu_shared_reservation_token",
            None,
        )
        heartbeat_task = _start_worker_reservation_heartbeat(
            session_manager,
            session_id,
            owner_token,
        )
        try:
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step=_PIPELINE_STARTUP_STEP,
                    state=StepState.RUNNING,
                    current=0,
                    total=1,
                    percent=0,
                    message="Worker started; loading input artifacts",
                )
            )
            uses_shared_store = getattr(
                session_manager, "uses_shared_store", lambda: False
            )
            if uses_shared_store():
                await asyncio.to_thread(
                    session_manager.sync_from_store,
                    session_id,
                    "input/",
                )
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step=_PIPELINE_STARTUP_STEP,
                    state=StepState.RUNNING,
                    current=0,
                    total=1,
                    percent=0,
                    message="Preparing texture pipeline workflow",
                )
            )
            from texture_agent.workflows.factory import create_texture_pipeline_workflow

            await _execute_pipeline_inner(
                session_id,
                config_dict,
                session_manager,
                event_bus,
                session_dir,
                only_steps,
                skip_steps,
                create_texture_pipeline_workflow,
                on_artifacts_synced,
            )
        except asyncio.CancelledError:
            # task.cancel() (e.g. from POST /cancel) raises CancelledError at the
            # next await point. If the worker has not yet reached the between-step
            # is_cancelled() checkpoint in _execute_pipeline_inner, the cooperative
            # cleanup that normally persists "cancelled" is skipped -- handle that
            # final transition here so /status flips from "cancelling" to
            # "cancelled" instead of stalling.
            #
            # Persist the disk state synchronously BEFORE awaiting the event emit:
            # JobRegistry.cancel wraps task.cancel() in wait_for(timeout=5s) and
            # may fire a second task.cancel() if cleanup is slow. A re-raised
            # CancelledError on the await would skip the disk update otherwise.
            #
            # Keep the session worker lock held through this terminal update so
            # DELETE cannot remove artifacts while cancellation cleanup is still
            # writing metadata or queued events.
            logger.info("Pipeline cancelled via task.cancel for %s", session_id[:8])
            try:
                await asyncio.to_thread(
                    session_manager.update_session,
                    session_id,
                    {"status": "cancelled"},
                )
                await asyncio.to_thread(session_manager.clear_cancellation, session_id)
            except Exception:
                logger.exception(  # pragma: no cover - cancelled event emit failure
                    "Failed to persist cancelled status for %s", session_id[:8]
                )
            try:
                await event_bus.emit(
                    ProgressEvent(
                        session_id=session_id,
                        step="pipeline",
                        state=StepState.CANCELLED,
                        message="Pipeline cancelled by user",
                    )
                )
            except Exception:  # pragma: no cover - cancelled event emit failure
                logger.exception(
                    "Failed to emit cancelled event for %s", session_id[:8]
                )
            raise
        except Exception as e:
            if getattr(e, "_wu_failure_handled", False):
                raise
            # Any uncaught error past the per-step guard (e.g. post-loop
            # packaging, final stats) must still flip the session to "failed"
            # so /status doesn't stay at "running" forever. The worker lock
            # remains held through this terminal update.
            _log_unhandled_pipeline_failure()
            try:
                await asyncio.to_thread(
                    session_manager.update_session,
                    session_id,
                    {
                        "status": "failed",
                        "error": sanitize_message(
                            str(e), service_config.session_storage_path
                        ),
                    },
                )
            except Exception:
                logger.exception(
                    "Failed to persist failed status for %s", session_id[:8]
                )
            sanitized_error = sanitize_message(
                str(e), service_config.session_storage_path
            )
            try:
                await event_bus.emit(
                    ProgressEvent(
                        session_id=session_id,
                        step="pipeline",
                        state=StepState.FAILED,
                        message=sanitized_error,
                    )
                )
            except Exception:
                logger.exception("Failed to emit failed event for %s", session_id[:8])
            raise
        finally:
            await _stop_worker_reservation_heartbeat(heartbeat_task)


async def _execute_pipeline_inner(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager: Any,
    event_bus: Any,
    session_dir: Path,
    only_steps: list[str] | None,
    skip_steps: list[str] | None,
    create_texture_pipeline_workflow: Any,
    on_artifacts_synced: Callable[[], Coroutine[Any, Any, None]] | None = None,
) -> None:
    """Body of execute_pipeline_async, kept separate so the outer function
    can wrap it in a try/except that persists failure state on unhandled
    errors.
    """
    # Build context from config
    config_dict, context = _prepare_config_and_context(config_dict, session_dir)

    # Persist per-unit checkpoints and accepted artifacts through the session
    # store so a replacement worker can resume without repeating backend work.
    from ..runtime.texture_execution import SessionTextureExecutionCheckpointStore

    session_store = getattr(session_manager, "store", None)
    if session_store is not None:
        context["texture_execution_checkpoint_store"] = (
            SessionTextureExecutionCheckpointStore(
                session_store,
                session_id,
                session_dir,
            )
        )
    is_cancelled = getattr(session_manager, "is_cancelled", None)
    if callable(is_cancelled):
        context["texture_execution_is_cancelled"] = lambda: is_cancelled(session_id)

    def _record_texture_execution_progress(checkpoint: Any) -> None:
        accepted = sum(
            record.accepted_result is not None for record in checkpoint.records
        )
        total = len(checkpoint.selected_unit_ids)
        update_step_progress = getattr(session_manager, "update_step_progress", None)
        if not callable(update_step_progress):
            return
        try:
            update_step_progress(
                session_id,
                "generate_textures",
                {
                    "accepted_units": accepted,
                    "remaining_units": total - accepted,
                    "total_units": total,
                    "percent": round((accepted / total) * 100, 1) if total else 100.0,
                },
            )
        except Exception:
            logger.warning(
                "Ignoring texture execution progress callback failure for %s",
                session_id,
                exc_info=True,
            )

    context["texture_execution_progress_callback"] = _record_texture_execution_progress

    # Create task list
    tasks = create_texture_pipeline_workflow(context, skip=skip_steps, only=only_steps)
    total_tasks = len(tasks)

    logger.info(
        f"Running texture pipeline ({total_tasks} steps) for {session_id[:8]}..."
    )
    await asyncio.to_thread(
        session_manager.update_session,
        session_id,
        {"status": "running"},
    )

    completed_step_names: list[str] = []
    planned_step_names = [_task_to_step_name(task) for task in tasks]

    for i, task in enumerate(tasks):
        step_name = _task_to_step_name(task)

        # Check cancellation between tasks
        if await asyncio.to_thread(
            session_manager.is_cancelled, session_id
        ):  # pragma: no cover - cooperative cancel between steps
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step=step_name,
                    state=StepState.CANCELLED,
                    message="Pipeline cancelled by user",
                )
            )
            await asyncio.to_thread(
                session_manager.update_session,
                session_id,
                {"status": "cancelled"},
            )
            await asyncio.to_thread(session_manager.clear_cancellation, session_id)
            logger.info(f"Pipeline cancelled for {session_id[:8]}...")
            return

        # Emit step start
        await event_bus.emit(
            ProgressEvent(
                session_id=session_id,
                step=step_name,
                state=StepState.RUNNING,
                current=i + 1,
                total=total_tasks,
                percent=0,
                message=f"Starting {task.name}",
            )
        )

        try:
            if step_name == "apply_textures":
                await asyncio.to_thread(_hydrate_cached_apply_context, context)
            # Only the new plan-aware tasks require the bounded plan here.
            # Keep accepting legacy/mock GenerateTexturesTask implementations
            # used by compatibility workflows; the production factory now
            # emits ExecuteTexturePlanTask for this public step.
            if await asyncio.to_thread(
                _requires_executable_texture_plan,
                task,
                context,
            ):
                from texture_agent.tasks.plan_textures import (
                    require_executable_texture_plan,
                )

                await asyncio.to_thread(require_executable_texture_plan, context)
            # Run synchronous task in the thread pool. The outer wrapper
            # holds the cross-process worker lock for the full pipeline
            # lifetime, including final metadata/event writes.
            loop = asyncio.get_running_loop()
            step_future = loop.run_in_executor(None, task.run, context)
            try:
                context = await asyncio.shield(step_future)
            except asyncio.CancelledError:
                logger.info(
                    "Cancellation requested during %s for %s; waiting for worker "
                    "thread to finish before releasing the session worker lock",
                    redact_sensitive_log_text(step_name),
                    session_id[:8],
                )
                await _drain_cancelled_step(
                    session_id=session_id,
                    step_name=step_name,
                    step_future=step_future,
                    session_manager=session_manager,
                )
                raise
            if step_name == "apply_textures" and context.get("output_usd_paths"):
                prepared_output = await _run_threaded_call_with_cancel_drain(
                    _prepare_source_usdz_stage,
                    context,
                    session_dir,
                    session_id=session_id,
                    step_name="prepare_source_usdz_stage",
                    session_manager=session_manager,
                )
                if prepared_output is None:
                    raise RuntimeError(
                        context.get(
                            "usdz_packaging_error",
                            "Failed to prepare textured USD output dependencies.",
                        )
                    )
                if context.get("render_output_usd_paths"):
                    # Layered USDZ roots must be fully rewritten/packageable
                    # before RenderOutputTask consumes the reconstructed stage.
                    # Otherwise composition is present but generated textures
                    # still resolve relative to cache/output and visual
                    # evidence can silently show the old material.
                    usdz_path = await _run_threaded_call_with_cancel_drain(
                        _package_usdz,
                        context,
                        session_dir,
                        session_id=session_id,
                        step_name="package_layered_usdz",
                        session_manager=session_manager,
                    )
                    if usdz_path is None:
                        raise RuntimeError(
                            context.get(
                                "usdz_packaging_error",
                                "Failed to package reconstructed textured USDZ.",
                            )
                        )
                    context["output_usdz_path"] = usdz_path
                    await _sync_prefix_to_store_with_cancel_drain(
                        session_manager,
                        session_id,
                        "cache/output/",
                        step_name="checkpoint_layered_usdz",
                    )
            # Prompt generation may contact an LLM. Checkpoint its reusable
            # output before any later step or between-step cancellation can
            # terminate this worker.
            if step_name == "generate_prompts":
                synced_prompts = await _sync_prefix_to_store(
                    session_manager,
                    session_id,
                    "cache/prompts/",
                )
                if synced_prompts:  # pragma: no cover - informational logging branch
                    logger.info(
                        "Checkpointed %d prompt artifact file(s) for %s",
                        synced_prompts,
                        session_id[:8],
                    )
        except Exception as e:
            _log_step_failure(
                step_index=i + 1,
                total_steps=total_tasks,
            )
            _write_service_artifact_manifest(
                context,
                status="failed",
                service_urls=_artifact_download_urls(session_id),
            )
            # A rejected bounded plan and a fail-closed layered-package result
            # are intentional inspectable failures. Make their partial
            # artifacts durable before surfacing the failed step so shared
            # storage deployments do not lose generated maps or the apply USD.
            failure_prefixes: list[str] = []
            if context.get("texture_plan_path"):
                failure_prefixes.append("cache/texture_plan.json")
            if context.get("output_usd_paths") and (
                step_name == "apply_textures" or context.get("output_usdz_path")
            ):
                failure_prefixes.extend(("cache/textures/", "cache/output/"))
            if context.get("artifacts_manifest_path"):
                failure_prefixes.append("cache/artifacts_manifest.json")
            for prefix in dict.fromkeys(failure_prefixes):
                try:
                    await _sync_prefix_to_store_with_cancel_drain(
                        session_manager,
                        session_id,
                        prefix,
                        step_name="checkpoint_failed_step_artifacts",
                    )
                except Exception:
                    _log_failed_artifact_sync()
            # Tasks mutate `context` with structured per-unit error records
            # (e.g. ``generate_textures_errors``) BEFORE raising the
            # threshold-gate RuntimeError. Surface those on the FAILED event
            # and persisted session metadata; without this the highest-value
            # failure mode (the threshold gate firing) loses the very
            # diagnostics this code path was added to provide.
            failed_stats = _extract_step_stats(step_name, context)
            if context.get("artifacts_manifest_path"):
                failed_stats["manifest_path"] = context["artifacts_manifest_path"]
                failed_stats["manifest_available"] = True
            sanitized_message = sanitize_message(
                str(e), service_config.session_storage_path
            )
            sanitized_stats = sanitize_step_stats(
                failed_stats, service_config.session_storage_path
            )
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step=step_name,
                    state=StepState.FAILED,
                    message=sanitized_message,
                    extra=sanitized_stats or None,
                )
            )
            await asyncio.to_thread(
                session_manager.update_session,
                session_id,
                {
                    "status": "failed",
                    "error": sanitized_message,
                    "failed_step": step_name,
                    "failed_step_stats": sanitized_stats,
                },
            )
            setattr(e, "_wu_failure_handled", True)
            raise

        step_stats = _extract_step_stats(step_name, context)
        sanitized_step_stats = sanitize_step_stats(
            step_stats, service_config.session_storage_path
        )
        validation_error = _get_step_validation_error(
            step_name, step_stats, planned_step_names, context
        )
        if validation_error:
            _write_service_artifact_manifest(
                context,
                status="failed",
                service_urls=_artifact_download_urls(session_id),
            )
            partial_results = _extract_final_stats(context, session_dir)
            logger.error(
                "Step %s produced invalid terminal state for %s: %s",
                redact_sensitive_log_text(step_name),
                session_id[:8],
                redact_sensitive_log_text(validation_error),
            )
            # Validation failures (e.g. apply_textures emitting no USD)
            # are caused by upstream gen/blend errors that already
            # populated structured records on context. Bundle the
            # failing step's own stats together with any upstream
            # ``*_errors`` lists so REST consumers see WHY -- without
            # this the FAILED event and ``/result`` only carry the
            # generic prose message.
            failed_stats = dict(step_stats)
            if context.get("artifacts_manifest_path"):
                failed_stats["manifest_path"] = context["artifacts_manifest_path"]
                failed_stats["manifest_available"] = True
            for upstream_key, count_key in (
                ("generate_textures_errors", "generate_textures_failed_count"),
                ("blend_textures_errors", "blend_textures_failed_count"),
            ):
                upstream_errors = context.get(upstream_key)
                if upstream_errors:
                    failed_stats.setdefault("upstream_errors", {})[
                        upstream_key.removesuffix("_errors")
                    ] = {
                        "count": context.get(count_key, len(upstream_errors)),
                        "errors": _truncate_errors(upstream_errors),
                    }
            sanitized_validation_error = sanitize_message(
                validation_error, service_config.session_storage_path
            )
            sanitized_failed_stats = sanitize_step_stats(
                failed_stats, service_config.session_storage_path
            )
            sanitized_partial_results = sanitize_step_stats(
                partial_results, service_config.session_storage_path
            )
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step=step_name,
                    state=StepState.FAILED,
                    message=sanitized_validation_error,
                    extra=sanitized_failed_stats or None,
                )
            )
            await asyncio.to_thread(
                session_manager.update_session,
                session_id,
                {
                    "status": "failed",
                    "error": sanitized_validation_error,
                    "failed_step": step_name,
                    "partial_results": sanitized_partial_results,
                    "failed_step_stats": sanitized_failed_stats,
                },
            )
            handled_error = RuntimeError(validation_error)
            setattr(handled_error, "_wu_failure_handled", True)
            raise handled_error

        # Emit step completed
        await event_bus.emit(
            ProgressEvent(
                session_id=session_id,
                step=step_name,
                state=StepState.COMPLETED,
                percent=100,
                message=f"Completed {task.name}",
                extra=sanitized_step_stats,
            )
        )

        completed_step_names.append(step_name)
        logger.info(
            "[%s/%s] %s complete for %s",
            i + 1,
            total_tasks,
            redact_sensitive_log_text(step_name),
            session_id[:8],
        )

    # Package single-layer output after the workflow. Layered source USDZs are
    # packaged fail-closed immediately after apply_textures so rendering and
    # download consume the same reconstructed stage. A late single-layer
    # packaging exception remains inspectable and must not leave the session
    # stuck at status=running / 95%.
    if "apply_textures" in completed_step_names and not context.get("output_usdz_path"):
        try:
            usdz_path = await _run_threaded_call_with_cancel_drain(
                _package_usdz,
                context,
                session_dir,
                session_id=session_id,
                step_name="package_single_layer_usdz",
                session_manager=session_manager,
            )
            if (
                usdz_path
            ):  # pragma: no cover - USDZ packaging success path depends on pxr packager
                context["output_usdz_path"] = usdz_path
        except _CancellationDrainError:
            raise
        except Exception as exc:
            _record_usdz_packaging_failure(
                context, f"USDZ packaging raised an unexpected exception: {exc}"
            )
            logger.exception(
                "USDZ packaging failed for %s; continuing with .usd output only",
                session_id[:8],
            )

    metadata = await asyncio.to_thread(session_manager.get_session_metadata, session_id)
    duration_seconds = 0
    if metadata and metadata.get("created_at"):
        created_at = datetime.fromisoformat(metadata["created_at"])
        duration_seconds = int((datetime.now(UTC) - created_at).total_seconds())

    service_urls = _artifact_download_urls(session_id)

    synced = 0
    sync_failures: list[str] = []
    for prefix in (
        "cache/discovery/",
        "cache/texture_plan.json",
        "cache/textures/",
        "cache/output/",
        "cache/renders/",
        "preview/",
        "input/config.yaml",
    ):
        try:
            synced += await _sync_prefix_to_store(session_manager, session_id, prefix)
        except Exception as e:
            sync_failures.append(f"{prefix}: {e}")
            logger.error(
                "Failed to sync %s to store for %s: %s",
                redact_sensitive_log_text(prefix),
                session_id[:8],
                redact_sensitive_log_text(e),
            )
    if sync_failures:
        raise RuntimeError(
            "Failed to sync pipeline artifacts to shared storage: "
            + "; ".join(sync_failures)
        )
    if synced:  # pragma: no cover - informational logging branch
        logger.info("Synced %d artifact file(s) for %s", synced, session_id[:8])

    if on_artifacts_synced is not None:
        # Finalization may atomically promote durable metadata that describes
        # the just-synchronized artifacts. Once it starts, drain it under the
        # session worker lock before propagating cancellation so a background
        # sync cannot race lock release and leave local/durable state split.
        finalization_task = asyncio.create_task(on_artifacts_synced())
        try:
            await asyncio.shield(finalization_task)
        except asyncio.CancelledError:
            logger.info(
                "Cancellation requested during artifact finalization for %s; "
                "waiting for the durable commit before releasing the worker lock",
                session_id[:8],
            )
            await _drain_cancelled_step(
                session_id=session_id,
                step_name="artifact_finalization",
                step_future=finalization_task,
                session_manager=session_manager,
            )
            raise

    # Write and sync the success manifest only after all other artifacts are
    # durable in the shared store. Otherwise a failed sync can leave a completed
    # manifest advertising objects that were never uploaded.
    manifest_path = _write_service_artifact_manifest(
        context,
        status=_artifact_manifest_status(context),
        service_urls=service_urls,
        duration_seconds=duration_seconds,
    )
    if manifest_path is not None:
        synced_manifest = await _sync_prefix_to_store(
            session_manager,
            session_id,
            "cache/artifacts_manifest.json",
        )
        if synced_manifest:
            logger.info(
                "Synced artifact manifest for %s",
                session_id[:8],
            )

    stats = _extract_final_stats(context, session_dir)
    sanitized_stats = sanitize_step_stats(stats, service_config.session_storage_path)
    _log_pipeline_stats(sanitized_stats)

    # Write stats to session metadata after artifact sync but before emitting
    # completion, so clients reacting to SSE "done" can immediately GET
    # /results and then fetch artifacts from the shared store.
    await asyncio.to_thread(
        session_manager.update_session,
        session_id,
        {
            "results": sanitized_stats,
            "duration_seconds": duration_seconds,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )

    # Emit final completion event. The pipeline_completed marker is the SSE
    # close contract; step-level COMPLETED events can reach 100% before
    # artifact sync and result metadata are durable.
    last_step = _task_to_step_name(tasks[-1]) if tasks else "pipeline"
    await event_bus.emit(
        ProgressEvent(
            session_id=session_id,
            step=last_step,
            state=StepState.COMPLETED,
            percent=100,
            message="Pipeline completed successfully",
            extra={
                **(sanitized_stats or {}),
                "pipeline_completed": True,
                "pipeline_ready": True,
            },
        )
    )

    logger.info(f"Pipeline execution completed for {session_id[:8]}")
