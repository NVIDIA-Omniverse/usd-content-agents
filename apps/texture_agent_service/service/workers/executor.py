# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline execution for Texture Agent Service.

Wraps the synchronous texture-agent pipeline by running each task
individually via asyncio.to_thread(), emitting progress events between steps.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from contextlib import nullcontext, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from world_understanding.utils.usd.package import (
    UsdzPackageError,
    extract_usdz_members_to_dir,
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
                logger.exception(
                    "Step %s raised while draining cancelled session %s",
                    step_name,
                    session_id[:8],
                )
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
            raise RuntimeError(reason)

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
            raise RuntimeError(reason)
        except asyncio.CancelledError:  # pragma: no cover - repeated cancel edge
            if step_future.done():
                continue
            logger.debug(
                "Additional cancellation while draining %s for %s",
                step_name,
                session_id[:8],
            )
            continue


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
                prefix,
                session_id[:8],
                attempt,
                attempts,
                exc,
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"Failed to sync {prefix} to shared store: {last_error}")


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
    plan_path = working_dir / "texture_plan.json"
    if plan_path.is_file():
        context["texture_plan_path"] = str(plan_path)
    planning_config = context.get("planning_config") or {}
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


def _package_usdz(context: dict[str, Any], session_dir: Path) -> str | None:
    """Package the output USD + textures into a self-contained USDZ.

    Rewrites absolute texture paths to relative, then bundles everything
    into a single .usdz archive for easy download.

    Returns:
        Path to the USDZ file, or None if packaging failed.
    """
    import hashlib
    import re
    import shutil
    import zipfile

    from pxr import Sdf, Usd, UsdShade, UsdUtils
    from texture_agent.functions.artifact_manifest import (
        make_diagnostic,
        validate_output_texture_portability,
    )

    output_paths = context.get("output_usd_paths", [])
    if not output_paths:  # pragma: no cover - packaging skipped without apply output
        return None

    output_usd = Path(output_paths[0])
    if not output_usd.exists():  # pragma: no cover - defensive missing output guard
        message = f"Output USD not found: {output_usd}"
        _record_usdz_packaging_failure(context, message)
        logger.warning(message)
        return None

    # Rewrite absolute texture paths to be relative to the USD file.
    # Textures live in cache/textures/ while USD is in cache/output/,
    # so the relative path from the USD is ../textures/<filename>.
    textures_dir = output_usd.parent.parent / "textures"

    stage = Usd.Stage.Open(str(output_usd))
    if not stage:  # pragma: no cover - malformed USD guard
        _record_usdz_packaging_failure(
            context, f"Failed to open output USD for USDZ packaging: {output_usd}"
        )
        return None

    uri_scheme_re = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
    safe_filename_re = re.compile(r"[^A-Za-z0-9._-]+")
    session_root = session_dir.resolve()
    input_root = session_root / "input"
    try:
        input_root = input_root.resolve()
        input_root.relative_to(session_root)
    except (OSError, ValueError):  # pragma: no cover - resolved session path guard
        input_root = session_root / "__invalid_input_root__"
    usdz_extracts_root = input_root / ".texture_agent_usdz_extracts"
    usdz_extracts_ready = False
    max_usdz_asset_members = 512
    max_usdz_asset_bytes = max(service_config.max_upload_size_mb, 1) * 1024 * 1024

    context_usd_parent: Path | None = None
    context_usd_path = context.get("usd_path")
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

    def _extract_uploaded_usdz_assets() -> None:
        nonlocal usdz_extracts_ready
        if usdz_extracts_ready:  # pragma: no cover - memoized nested helper guard
            return
        usdz_extracts_ready = True
        if not input_root.is_dir():  # pragma: no cover - missing upload dir guard
            return

        extracted_members = 0
        extracted_bytes = 0
        packageable_suffixes = {".png", ".usd", ".usda", ".usdc", ".usdz"}
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

    def _resolve_local_texture_ref(path_value: str) -> Path | None:
        if _is_uri_ref(path_value):
            return None
        try:
            path = Path(path_value)
            if path.is_absolute():
                return path.resolve()
            return (output_usd.parent / path).resolve()
        except (OSError, ValueError):  # pragma: no cover - invalid texture ref guard
            return None

    def _can_rewrite_to_textures_dir(path_value: str) -> bool:
        src_resolved = _resolve_local_texture_ref(path_value)
        if src_resolved is None or not src_resolved.is_file():
            return False
        try:
            src_resolved.relative_to(textures_dir.resolve())
        except ValueError:
            return False
        return True

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
                    if not match.is_file() or match.suffix.lower() != ".png":
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

    def _relative_path_parts(path_value: str) -> tuple[str, ...] | None:
        path = Path(path_value)
        if path.is_absolute() or any(part == ".." for part in path.parts):
            return None  # pragma: no cover - unsafe authored asset path
        return tuple(part for part in path.parts if part not in ("", "."))

    def _find_upload_bundle_texture(path_value: str) -> Path | None:
        if _is_uri_ref(path_value) or not path_value.lower().endswith(".png"):
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
        return f"{stem}.png"

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
            dest = textures_dir / f"{Path(filename).stem}_{digest}.png"
            counter = 1
            while (
                dest.exists() and dest.resolve() != src
            ):  # pragma: no cover - repeated collision guard
                dest = textures_dir / f"{Path(filename).stem}_{digest}_{counter}.png"
                counter += 1

        if dest.resolve() != src:  # pragma: no cover - upload texture localization copy
            shutil.copy2(src, dest)
        localized_ref = f"../textures/{dest.name}"
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
        if not parts:  # pragma: no cover - unsafe relative layer path guard
            return False
        expected = (output_usd.parent / Path(*parts)).resolve()
        if expected.is_file():  # pragma: no cover - already-localized layer guard
            return True

        src = _find_upload_bundle_layer(path_value)
        if src is None:  # pragma: no cover - missing upload layer guard
            return False
        try:
            src = src.resolve()
            expected.relative_to(output_usd.parent.resolve())
        except (OSError, ValueError):  # pragma: no cover - unsafe layer target guard
            return False

        expected.parent.mkdir(parents=True, exist_ok=True)
        if expected != src:
            shutil.copy2(src, expected)
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

                    if Path(asset_path).is_absolute():  # pragma: no cover
                        ref_exists = Path(asset_path).is_file()
                    else:
                        parts = _relative_path_parts(asset_path)
                        ref_exists = bool(
                            parts and (output_usd.parent / Path(*parts)).is_file()
                        )
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
            root_layer.Export(str(output_usd))
        return True

    rewritten = 0
    cleared_missing_mdl_source_assets: list[str] = []
    for prim in stage.Traverse():
        if prim.IsInstanceProxy():  # pragma: no cover - USD instance proxy guard
            continue
        if prim.IsInstance() or prim.IsInstanceable():  # pragma: no cover
            prim.SetInstanceable(False)
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
                filename = Path(old_path).name
                if filename.lower().endswith(
                    ".mdl"
                ):  # pragma: no cover - MDL source asset cleanup branch
                    if (
                        Path(old_path).is_absolute()
                    ):  # pragma: no cover - absolute MDL refs are preserved
                        continue
                    src_resolved = _resolve_local_texture_ref(old_path)
                    if src_resolved is not None and not src_resolved.exists():
                        try:
                            attr.Clear()
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
                if not filename.lower().endswith(
                    ".png"
                ):  # pragma: no cover - non-texture asset guard
                    continue
                new_path = None
                if _can_rewrite_to_textures_dir(old_path):
                    new_path = f"../textures/{filename}"
                else:
                    new_path = _localized_upload_texture_ref(old_path)
                if new_path:
                    attr.Set(Sdf.AssetPath(new_path))
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
            filename = Path(val).name
            if not filename.lower().endswith(
                ".png"
            ):  # pragma: no cover - non-PNG shader input guard
                continue
            new_path = None
            if _can_rewrite_to_textures_dir(val):
                new_path = f"../textures/{filename}"
            else:
                new_path = _localized_upload_texture_ref(val)
            if new_path is None:
                continue
            try:
                attr.Set(new_path)
                rewritten += 1
            except Exception as err:  # pragma: no cover - USD attr set failure
                logger.warning(
                    "Failed to rewrite string texture path on %s: %s",
                    attr.GetPath(),
                    err,
                )

    if rewritten > 0 or cleared_missing_mdl_source_assets:
        stage.GetRootLayer().Export(str(output_usd))
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

    portability = validate_output_texture_portability(output_usd)
    context["output_portability"] = portability
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

    # Remove stale file so a failed CreateNewUsdzPackage doesn't leave
    # a partial (raw USDC) file that the download endpoint would serve.
    usdz_path.unlink(missing_ok=True)

    try:
        success = UsdUtils.CreateNewUsdzPackage(str(output_usd), str(usdz_path))
    except Exception as exc:
        usdz_path.unlink(missing_ok=True)
        _record_usdz_packaging_failure(context, f"Failed to create USDZ package: {exc}")
        logger.exception("Failed to create USDZ package")
        return None

    if success and usdz_path.exists():
        # Validate the output is actually a ZIP archive (USDZ spec).
        # CreateNewUsdzPackage can leave raw USDC bytes on failure.
        if not zipfile.is_zipfile(
            usdz_path
        ):  # pragma: no cover - USDZ package corruption guard
            message = f"CreateNewUsdzPackage wrote non-ZIP data to {usdz_path}"
            _record_usdz_packaging_failure(context, message)
            logger.warning("%s, removing", message)
            usdz_path.unlink(missing_ok=True)
            return None

        size_mb = usdz_path.stat().st_size / (1024 * 1024)
        logger.info("Packaged USDZ: %s (%.1f MB)", usdz_path, size_mb)
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
            logger.exception("Unhandled pipeline error for %s: %s", session_id[:8], e)
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
            logger.error(f"Step {step_name} failed for {session_id[:8]}: {e}")
            _write_service_artifact_manifest(
                context,
                status="failed",
                service_urls=_artifact_download_urls(session_id),
            )
            # A rejected bounded plan is an intentional inspectable artifact,
            # not backend output. Make it (and the failure manifest that
            # records its decision) durable before surfacing the failed step.
            if context.get("texture_plan_path"):
                for prefix in (
                    "cache/texture_plan.json",
                    "cache/artifacts_manifest.json",
                ):
                    try:
                        await _sync_prefix_to_store(
                            session_manager,
                            session_id,
                            prefix,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to sync rejected plan artifact %s for %s",
                            prefix,
                            session_id[:8],
                        )
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
                step_name,
                session_id[:8],
                validation_error,
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
            f"[{i + 1}/{total_tasks}] {step_name} complete for {session_id[:8]}"
        )

    # Package output into USDZ (self-contained with textures).
    # Treat packaging failures as non-fatal — the textured .usd is already
    # written and useful on its own; the USDZ bundle is a convenience. A
    # pxr/UsdUtils exception here (e.g. unresolved asset references from
    # the original input USD) must not leave the session stuck at
    # status=running / 95%.
    if "apply_textures" in completed_step_names:
        try:
            usdz_path = await asyncio.to_thread(_package_usdz, context, session_dir)
            if (
                usdz_path
            ):  # pragma: no cover - USDZ packaging success path depends on pxr packager
                context["output_usdz_path"] = usdz_path
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
                prefix,
                session_id[:8],
                e,
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
    logger.info(f"Pipeline stats for {session_id[:8]}: {stats}")
    sanitized_stats = sanitize_step_stats(stats, service_config.session_storage_path)

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
