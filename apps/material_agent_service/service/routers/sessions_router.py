# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sessions API endpoints - Session CRUD operations."""

import asyncio
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query

from ..coverage import normalize_legacy_completed_coverage
from ..runtime.registry import get_job_registry
from ..session.manager import SessionManager
from ..utils import timer


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC-aware. Assumes naive datetimes are UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/sessions", tags=["sessions"])

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


@router.get("")
async def list_sessions(
    limit: int = Query(50, ge=1, le=500, description="Max sessions to return"),
    offset: int = Query(0, ge=0, description="Number of sessions to skip"),
):
    """List sessions with metadata (paginated).

    Returns:
        Dict with:
          - sessions: List of sessions with id, status, created_at
          - total: Total session count (before pagination)
          - offset / limit: Pagination parameters
          - active_count: Currently running jobs (from JobRegistry)
          - max_active_sessions: Configured concurrency limit
    """
    manager = get_session_manager()

    with timer("list_sessions"):
        all_session_ids = await manager.list_sessions()

    # Fetch metadata for all sessions using batch method (single S3 client)
    with timer("get_session_metadata"):
        metadata_list = await manager.get_session_metadata_batch(all_session_ids)

    sessions = []
    for session_id, metadata in zip(all_session_ids, metadata_list):
        if metadata:
            sessions.append(
                {
                    "session_id": session_id,
                    "status": metadata.get("status", "unknown"),
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at"),
                    "elapsed_seconds": metadata.get("elapsed_seconds", 0),
                    "user_email": metadata.get("user_email", ""),
                    "config": {
                        "has_reference_images": metadata.get("config", {}).get(
                            "has_reference_images", False
                        ),
                        "num_reference_images": metadata.get("config", {}).get(
                            "num_reference_images", 0
                        ),
                    },
                }
            )

    # Sort by created_at (newest first), then paginate
    sessions.sort(key=lambda x: x["created_at"] or "", reverse=True)
    total = len(sessions)
    paginated = sessions[offset : offset + limit]

    # Include actual active count from job registry (not derived from session status)
    job_registry = get_job_registry()

    return {
        "sessions": paginated,
        "total": total,
        "offset": offset,
        "limit": limit,
        "active_count": job_registry.active_count,
        "max_active_sessions": job_registry.max_concurrent,
    }


@router.get("/usage")
async def get_usage_stats(
    from_date: str | None = Query(
        None,
        description="Filter sessions created after this ISO 8601 date (e.g., 2026-02-01T00:00:00Z)",
    ),
    to_date: str | None = Query(
        None,
        description="Filter sessions created before this ISO 8601 date",
    ),
    user_email: str | None = Query(
        None,
        description="Filter sessions by user email",
    ),
):
    """Get aggregate usage statistics across all sessions.

    Scans existing session metadata to produce per-user and per-asset
    aggregate stats. No additional storage needed.

    Args:
        from_date: Optional ISO 8601 start date filter
        to_date: Optional ISO 8601 end date filter
        user_email: Optional user email filter

    Returns:
        Aggregate usage stats with by_user and by_asset breakdowns
    """
    manager = get_session_manager()

    # Parse date filters
    try:
        from_dt = _ensure_utc(datetime.fromisoformat(from_date)) if from_date else None
        to_dt = _ensure_utc(datetime.fromisoformat(to_date)) if to_date else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    # Load all sessions
    with timer("usage_list_sessions"):
        session_ids = await manager.list_sessions()

    with timer("usage_get_metadata"):
        metadata_list = await asyncio.gather(
            *[manager.get_session_metadata(sid) for sid in session_ids]
        )

    total_sessions = 0
    total_completed = 0
    total_failed = 0
    by_user: dict[str, dict] = {}
    by_asset: dict[str, dict] = {}

    for session_id, meta in zip(session_ids, metadata_list):
        if not meta:
            continue

        # Apply date filters
        created_at_str = meta.get("created_at")
        if created_at_str:
            try:
                created_at = _ensure_utc(datetime.fromisoformat(created_at_str))
                if from_dt and created_at < from_dt:
                    continue
                if to_dt and created_at > to_dt:
                    continue
            except (ValueError, TypeError):
                pass

        # Apply email filter
        session_email = meta.get("user_email", "")
        if user_email and session_email != user_email:
            continue

        status = meta.get("status", "unknown")
        # Skip sessions that haven't started a pipeline (e.g., just uploaded)
        if status in ("uploading", "ready", "unknown"):
            continue

        total_sessions += 1
        is_completed = status == "completed"
        is_failed = status == "failed"
        if is_completed:
            total_completed += 1
        if is_failed:
            total_failed += 1

        results = meta.get("results", {})
        asset_info = meta.get("asset", {})
        asset_filename = asset_info.get("filename") or meta.get("filename", "unknown")
        duration = meta.get("duration_seconds", 0) or 0
        step_timings = meta.get("step_timings", {})

        # Extract VLM model from config
        config = meta.get("config", {})
        vlm_model = config.get("vlm_model", "") or ""

        # Build session entry (shared between by_user and by_asset)
        session_entry = {
            "session_id": session_id,
            "asset_filename": asset_filename,
            "status": status,
            "created_at": created_at_str,
            "duration_seconds": duration,
            "original_prim_count": results.get("original_prim_count", 0),
            "prims_processed": results.get("prims_processed", 0),
            "predictions_made": results.get("predictions_made", 0),
            "materials_applied": results.get("materials_applied", 0),
            "images_generated": results.get("images_generated", 0),
            "vlm_model": vlm_model,
            "step_timings": step_timings,
        }

        # --- Aggregate by user ---
        if session_email:
            if session_email not in by_user:
                by_user[session_email] = {
                    "session_count": 0,
                    "completed": 0,
                    "failed": 0,
                    "total_original_prim_count": 0,
                    "total_prims_processed": 0,
                    "total_predictions_made": 0,
                    "total_materials_applied": 0,
                    "total_images_generated": 0,
                    "total_duration_seconds": 0,
                    "durations": [],
                    "assets": set(),
                    "last_session_at": "",
                    "sessions": [],
                }
            user_data = by_user[session_email]
            user_data["session_count"] += 1
            if is_completed:
                user_data["completed"] += 1
            if is_failed:
                user_data["failed"] += 1
            user_data["total_original_prim_count"] += results.get(
                "original_prim_count", 0
            )
            user_data["total_prims_processed"] += results.get("prims_processed", 0)
            user_data["total_predictions_made"] += results.get("predictions_made", 0)
            user_data["total_materials_applied"] += results.get("materials_applied", 0)
            user_data["total_images_generated"] += results.get("images_generated", 0)
            user_data["total_duration_seconds"] += duration
            if duration > 0:
                user_data["durations"].append(duration)
            user_data["assets"].add(asset_filename)
            if created_at_str and created_at_str > user_data["last_session_at"]:
                user_data["last_session_at"] = created_at_str
            user_data["sessions"].append(session_entry)

        # --- Aggregate by asset ---
        if asset_filename not in by_asset:
            by_asset[asset_filename] = {
                "session_count": 0,
                "completed": 0,
                "failed": 0,
                "users": set(),
                "total_original_prim_count": 0,
                "total_prims_processed": 0,
                "total_predictions_made": 0,
                "total_materials_applied": 0,
                "total_images_generated": 0,
                "total_duration_seconds": 0,
                "durations": [],
                "step_durations": {},
                "sessions": [],
            }
        asset_data = by_asset[asset_filename]
        asset_data["session_count"] += 1
        if is_completed:
            asset_data["completed"] += 1
        if is_failed:
            asset_data["failed"] += 1
        if session_email:
            asset_data["users"].add(session_email)
        asset_data["total_original_prim_count"] += results.get("original_prim_count", 0)
        asset_data["total_prims_processed"] += results.get("prims_processed", 0)
        asset_data["total_predictions_made"] += results.get("predictions_made", 0)
        asset_data["total_materials_applied"] += results.get("materials_applied", 0)
        asset_data["total_images_generated"] += results.get("images_generated", 0)
        asset_data["total_duration_seconds"] += duration
        if duration > 0:
            asset_data["durations"].append(duration)
        # Accumulate step durations for averaging
        for step_name, step_dur in step_timings.items():
            if step_name not in asset_data["step_durations"]:
                asset_data["step_durations"][step_name] = []
            asset_data["step_durations"][step_name].append(step_dur)

        # Add session entry with user_email for by_asset view
        asset_session = dict(session_entry)
        asset_session["user_email"] = session_email
        asset_data["sessions"].append(asset_session)

    # Finalize by_user: convert sets and compute duration stats
    for email, data in by_user.items():
        data["assets"] = sorted(data["assets"])
        durations = data.pop("durations")
        if durations:
            data["min_duration_seconds"] = min(durations)
            data["max_duration_seconds"] = max(durations)
            data["avg_duration_seconds"] = round(sum(durations) / len(durations))
        else:
            data["min_duration_seconds"] = 0
            data["max_duration_seconds"] = 0
            data["avg_duration_seconds"] = 0
        # Sort sessions by created_at (newest first)
        data["sessions"].sort(key=lambda s: s.get("created_at") or "", reverse=True)

    # Finalize by_asset: convert sets, compute duration and step avg stats
    for filename, data in by_asset.items():
        data["users"] = sorted(data["users"])
        durations = data.pop("durations")
        if durations:
            data["min_duration_seconds"] = min(durations)
            data["max_duration_seconds"] = max(durations)
            data["avg_duration_seconds"] = round(sum(durations) / len(durations))
        else:
            data["min_duration_seconds"] = 0
            data["max_duration_seconds"] = 0
            data["avg_duration_seconds"] = 0
        # Compute step average durations
        step_durations = data.pop("step_durations")
        data["step_avg_durations"] = {
            step: round(sum(durs) / len(durs))
            for step, durs in step_durations.items()
            if durs
        }
        # Sort sessions by created_at (newest first)
        data["sessions"].sort(key=lambda s: s.get("created_at") or "", reverse=True)

    return {
        "total_sessions": total_sessions,
        "total_completed": total_completed,
        "total_failed": total_failed,
        "by_user": by_user,
        "by_asset": by_asset,
    }


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Get detailed session information.

    Args:
        session_id: Session identifier

    Returns:
        Session metadata
    """
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    return normalize_legacy_completed_coverage(
        metadata,
        pipeline_active=get_job_registry().is_running(session_id),
    )


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Delete a session and all its artifacts.

    Args:
        session_id: Session identifier
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Cancel any running job first to release file handles
    job_registry = get_job_registry()
    if job_registry.is_running(session_id):
        await job_registry.cancel(session_id)

    # Retry deletion with backoff - file handles may take time to release after cancellation
    max_retries = 3
    for attempt in range(max_retries):
        success = await manager.delete_session(session_id)
        if success:
            return None

        # Check if session was deleted by another request (TOCTOU race)
        if not await manager.session_exists(session_id):
            return None  # Session gone - treat as success

        if attempt < max_retries - 1:
            await asyncio.sleep(0.05 * (attempt + 1))  # 50ms, 100ms backoff

    raise HTTPException(status_code=500, detail="Failed to delete session")


@router.post("/admin/cleanup")
async def trigger_cleanup(max_age_hours: float = 24.0):
    """Manually trigger session cleanup.

    Cleans up:
    1. Stale local cache entries (syncs to remote storage and removes local files)
    2. Expired sessions past their TTL

    Args:
        max_age_hours: Maximum age in hours before local cache cleanup (default: 24)

    Returns:
        Cleanup results with counts
    """
    manager = get_session_manager()

    logger.info(f"Manual cleanup triggered (max_age_hours={max_age_hours})")

    # Clean up stale local cache
    cleaned_cache = await manager.cleanup_stale_local_cache(max_age_hours=max_age_hours)

    # Clean up expired sessions
    expired_sessions = await manager.cleanup_expired_sessions()

    result = {
        "cleaned_local_cache": cleaned_cache,
        "expired_sessions_removed": expired_sessions,
        "max_age_hours": max_age_hours,
    }

    logger.info(f"Manual cleanup complete: {result}")

    return result
