# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event bus for pipeline progress events."""

import asyncio
import logging
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from .events import ProgressEvent, StepState

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_WORKER_HEARTBEAT_INTERVAL_SECONDS = 5 * 60
_LIVE_METADATA_FLUSH_INTERVAL_SECONDS = 2.0
_SHARED_SESSION_EXISTS_CACHE_SECONDS = 1.0
_LIVE_METADATA_FIELDS = (
    "current_step",
    "completed_steps",
    "overall_progress",
    "step_timings",
    "updated_at",
    "completed_at",
    "cancelled_at",
    "cancelling_at",
    "failed_at",
    "failed_step",
    "failed_step_stats",
    "error",
)


class EventBus:
    """Central event bus for pipeline progress.

    Manages:
    - Per-session event queues for SSE streaming
    - Canonical in-memory state snapshot for /status API
    - Event application logic to update state
    """

    def __init__(self, session_manager: Any = None):
        """Initialize event bus.

        Args:
            session_manager: Shared SessionManager instance for persistence.
                If None, persistence methods become no-ops.
        """
        self._queues: dict[str, asyncio.Queue[ProgressEvent]] = {}
        self._state: dict[str, dict[str, Any]] = {}
        self._deleted_sessions: set[str] = set()
        self._worker_heartbeat_at: dict[str, float] = {}
        self._shared_session_exists_checked_at: dict[str, float] = {}
        self._live_metadata_flush_at: dict[str, float] = {}
        self._live_metadata_pending: dict[str, dict[str, Any]] = {}
        self._live_metadata_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._worker_heartbeat_lock = asyncio.Lock()
        self._session_manager = session_manager

    def get_queue(self, session_id: str) -> asyncio.Queue[ProgressEvent]:
        """Get or create event queue for a session."""
        return self._queues.setdefault(session_id, asyncio.Queue())

    def get_snapshot(self, session_id: str) -> dict[str, Any] | None:
        """Get current in-memory state snapshot for a session."""
        return self._state.get(session_id)

    async def seed_pending_session(self, session_id: str) -> None:
        """Seed a lightweight local status snapshot for an accepted pipeline.

        This avoids backing-store access so same-instance status polling can
        return immediately after /pipeline accepts a job, before the worker has
        emitted its first progress event.
        """
        timestamp = datetime.now(UTC).isoformat()
        async with self._lock:
            existing = self._state.get(session_id)
            self._state[session_id] = {
                "session_id": session_id,
                "status": "pending",
                "created_at": existing.get("created_at", timestamp)
                if existing
                else timestamp,
                "updated_at": timestamp,
                "current_step": None,
                "completed_steps": [],
                "overall_progress": {
                    "current_step": 0,
                    "total_steps": 9,
                    "percent": 0,
                },
                "preview_images": [],
                "step_timings": {},
            }

    def clear_session_state(self, session_id: str) -> None:
        """Drop the in-memory snapshot AND any queued SSE events.

        Called by ``/regenerate`` so a retry of a previously-failed run
        does not (a) show stale ``failed_step`` / ``failed_step_stats``
        / ``error`` fields from the prior attempt via ``/status``, or
        (b) replay an old terminal FAILED event to a fresh SSE
        subscriber attaching mid-retry. The next event for this
        session_id will lazily rebuild the snapshot from ``pending``.
        """
        self._state.pop(session_id, None)
        # Drain the per-session queue so a queued FAILED event from
        # the prior run can't be delivered to a new subscriber.
        # ``stream_progress_events`` calls ``setdefault`` and reuses
        # the queue object, so we drain in place rather than dropping
        # the dict entry.
        queue = self._queues.get(session_id)
        if queue is not None:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - defensive race guard
                    break
        self._live_metadata_flush_at.pop(session_id, None)
        self._shared_session_exists_checked_at.pop(session_id, None)
        self._live_metadata_pending.pop(session_id, None)
        task = self._live_metadata_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()

    async def _session_exists_for_emit(self, session_id: str) -> bool:
        if self._session_manager is None:
            return True
        manager = self._session_manager
        uses_shared_store = getattr(manager, "uses_shared_store", lambda: False)
        if uses_shared_store():
            now = asyncio.get_running_loop().time()
            checked_at = self._shared_session_exists_checked_at.get(session_id)
            if (
                checked_at is not None
                and now - checked_at < _SHARED_SESSION_EXISTS_CACHE_SECONDS
            ):
                return True
            try:
                # Shared stores need the remote lookup so progress from one pod
                # stops after another pod deletes the session. Cache positive
                # checks briefly so rapid progress emits still debounce as a
                # batch instead of yielding for a remote HeadObject each time.
                # The cache is best-effort; concurrent first emits may race
                # into duplicate lookups before one records the timestamp.
                exists = bool(
                    await asyncio.to_thread(manager.session_exists, session_id)
                )
            except Exception as exc:
                logger.warning(
                    "Failed to verify session %s exists before progress emit: %s",
                    session_id[:8],
                    exc,
                )
                self._shared_session_exists_checked_at[session_id] = now
                return True
            if exists:
                self._shared_session_exists_checked_at[session_id] = now
            else:
                self._shared_session_exists_checked_at.pop(session_id, None)
            return exists
        return manager.session_exists(session_id)

    async def _heartbeat_worker_if_due(self, session_id: str) -> None:
        if self._session_manager is None:
            return
        manager = self._session_manager
        uses_shared_store = getattr(manager, "uses_shared_store", lambda: False)
        if not uses_shared_store():
            return
        heartbeat_worker = getattr(manager, "heartbeat_worker", None)
        if not callable(heartbeat_worker):
            return
        get_owner_token = getattr(manager, "get_worker_reservation_owner_token", None)
        owner_token = get_owner_token(session_id) if callable(get_owner_token) else None
        if owner_token is None:
            return
        async with self._worker_heartbeat_lock:
            now = asyncio.get_running_loop().time()
            last = self._worker_heartbeat_at.get(session_id)
            if last is not None and now - last < _WORKER_HEARTBEAT_INTERVAL_SECONDS:
                return
            self._worker_heartbeat_at[session_id] = now
        await asyncio.to_thread(heartbeat_worker, session_id, owner_token=owner_token)

    def _uses_shared_store(self) -> bool:
        if self._session_manager is None:
            return False
        uses_shared_store = getattr(
            self._session_manager,
            "uses_shared_store",
            lambda: False,
        )
        return bool(uses_shared_store())

    def _force_live_metadata_flush(self, event: ProgressEvent) -> bool:
        return event.state in (
            StepState.COMPLETED,
            StepState.FAILED,
            StepState.CANCELLED,
            StepState.CANCELLING,
        )

    def _schedule_live_metadata_flush_locked(
        self,
        session_id: str,
        updates: dict[str, Any],
        now: float,
    ) -> None:
        self._live_metadata_pending[session_id] = updates
        last_flush = self._live_metadata_flush_at.get(session_id, 0.0)
        delay = max(0.0, _LIVE_METADATA_FLUSH_INTERVAL_SECONDS - (now - last_flush))
        task = self._live_metadata_tasks.get(session_id)
        if task is not None and not task.done():
            return
        self._live_metadata_tasks[session_id] = asyncio.create_task(
            self._flush_live_metadata_after_delay(session_id, delay)
        )

    async def _flush_live_metadata_after_delay(
        self,
        session_id: str,
        delay: float,
    ) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            async with self._lock:
                updates = self._live_metadata_pending.pop(session_id, None)
                if updates is None or session_id in self._deleted_sessions:
                    return
                self._live_metadata_flush_at[session_id] = (
                    asyncio.get_running_loop().time()
                )
            await self._persist_live_metadata(session_id, updates)
        except asyncio.CancelledError:
            raise
        finally:
            current_task = asyncio.current_task()
            async with self._lock:
                if self._live_metadata_tasks.get(session_id) is current_task:
                    self._live_metadata_tasks.pop(session_id, None)

    async def emit(self, event: ProgressEvent) -> None:
        """Emit an event: update state and queue for subscribers."""
        pending_persists: list[tuple[str, str]] = []
        live_metadata_update: dict[str, Any] | None = None

        # Shared stores may need a network round-trip here. Keep it outside the
        # global bus lock so one slow S3 HeadObject cannot serialize progress
        # events for every session on the instance.
        if not await self._session_exists_for_emit(event.session_id):
            logger.debug(
                f"Dropping event for deleted session {event.session_id[:8]}... "
                f"(step={event.step}, state={event.state.value})"
            )
            return

        async with self._lock:
            if event.session_id in self._deleted_sessions:
                logger.debug(
                    f"Dropping event for deleted session {event.session_id[:8]}... "
                    f"(step={event.step}, state={event.state.value})"
                )
                return

            self._apply_event_to_state(event, pending_persists)

            state = self._state.get(event.session_id)
            if state:
                state_update = self._live_metadata_update_from_state(state)
                if self._uses_shared_store():
                    now = asyncio.get_running_loop().time()
                    last_flush = self._live_metadata_flush_at.get(event.session_id)
                    if (
                        last_flush is None
                        or now - last_flush >= _LIVE_METADATA_FLUSH_INTERVAL_SECONDS
                        or self._force_live_metadata_flush(event)
                    ):
                        live_metadata_update = state_update
                        self._live_metadata_flush_at[event.session_id] = now
                        self._live_metadata_pending.pop(event.session_id, None)
                        task = self._live_metadata_tasks.pop(event.session_id, None)
                        if task is not None:
                            task.cancel()
                    else:
                        self._schedule_live_metadata_flush_locked(
                            event.session_id,
                            state_update,
                            now,
                        )
                else:
                    live_metadata_update = state_update
                event.overall_percent = state.get("overall_progress", {}).get(
                    "percent", 0
                )
                logger.info(
                    f"[EventBus] {event.session_id[:8]}... {event.step}: "
                    f"step={event.percent or 0}% → overall={event.overall_percent}% (state={event.state.value})"
                )
            else:
                event.overall_percent = 0

            queue = self.get_queue(event.session_id)
            await queue.put(event)

        # Persist status changes and event log outside the lock
        await self._heartbeat_worker_if_due(event.session_id)
        for session_id, status in pending_persists:
            await self._persist_status(session_id, status)
        if live_metadata_update:
            await self._persist_live_metadata(
                event.session_id,
                live_metadata_update,
            )
        await self._save_event_to_log(event)

    def _live_metadata_update_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            key: deepcopy(state[key]) for key in _LIVE_METADATA_FIELDS if key in state
        }

    def _apply_event_to_state(
        self, event: ProgressEvent, pending_persists: list[tuple[str, str]]
    ) -> None:
        """Apply event to update canonical in-memory state."""
        session_id = event.session_id

        if session_id not in self._state:
            self._state[session_id] = {
                "session_id": session_id,
                "status": "pending",
                "created_at": event.timestamp,
                "updated_at": event.timestamp,
                "current_step": None,
                "completed_steps": [],
                "overall_progress": {
                    "current_step": 0,
                    "total_steps": 9,
                    "percent": 0,
                },
                "step_timings": {},
            }

        state = self._state[session_id]
        state["updated_at"] = event.timestamp

        if event.state == StepState.RUNNING:
            if (
                state.get("current_step") is None
                or state["current_step"].get("name") != event.step
            ):
                state["current_step"] = {
                    "name": event.step,
                    "display_name": self._get_display_name(event.step),
                    "started_at": event.timestamp,
                    "progress": {
                        "current": event.current or 0,
                        "total": event.total or 1,
                        "percent": event.percent or 0,
                        "message": event.message or "",
                    },
                    "elapsed_seconds": 0,
                }
                if state["status"] == "pending":
                    pending_persists.append((state["session_id"], "running"))
                state["status"] = "running"
            else:
                state["current_step"]["progress"] = {
                    "current": event.current or 0,
                    "total": event.total or 1,
                    "percent": event.percent or 0,
                    "message": event.message or "",
                }
                started_at = datetime.fromisoformat(state["current_step"]["started_at"])
                now = datetime.fromisoformat(event.timestamp)
                state["current_step"]["elapsed_seconds"] = int(
                    (now - started_at).total_seconds()
                )

            self._update_overall_progress(state, event.step, event.percent or 0)

        elif event.state == StepState.COMPLETED:
            if (
                state.get("current_step")
                and state["current_step"]["name"] == event.step
            ):
                started_at = datetime.fromisoformat(state["current_step"]["started_at"])
                now = datetime.fromisoformat(event.timestamp)
                duration = int((now - started_at).total_seconds())

                completed_step = {
                    "name": event.step,
                    "display_name": state["current_step"]["display_name"],
                    "started_at": state["current_step"]["started_at"],
                    "completed_at": event.timestamp,
                    "duration_seconds": duration,
                    "stats": event.extra or {},
                }
                state["completed_steps"].append(completed_step)
                state["step_timings"][event.step] = duration
                state["current_step"] = None

                self._update_overall_progress_on_completion(
                    state, event.step, pending_persists
                )

            elif event.extra and event.extra.get("pipeline_completed"):
                state["overall_progress"]["percent"] = 100
                state["status"] = "completed"
                state["completed_at"] = datetime.now(UTC).isoformat()
                state["current_step"] = None
                pending_persists.append((state["session_id"], "completed"))

        elif event.state == StepState.FAILED:
            state["status"] = "failed"
            state["error"] = event.message or "Unknown error"
            state["failed_step"] = event.step
            state["failed_at"] = event.timestamp
            # Carry the structured failed-step stats (textures_failed,
            # errors[], upstream_errors) into the snapshot so /status --
            # which reads the bus snapshot first -- can surface per-unit
            # failure detail to clients polling between SSE-disconnect
            # and /results. Without this, /status only carries prose.
            if event.extra:
                state["failed_step_stats"] = event.extra
            pending_persists.append((state["session_id"], "failed"))

        elif event.state == StepState.CANCELLED:
            state["status"] = "cancelled"
            state["cancelled_at"] = event.timestamp
            pending_persists.append((state["session_id"], "cancelled"))

        elif event.state == StepState.CANCELLING:
            # Don't downgrade a terminal state. If the worker finished or
            # failed between the cancel route's is_running check and this
            # event reaching the bus, the terminal status wins.
            if state.get("status") in ("completed", "failed", "cancelled"):
                return
            state["status"] = "cancelling"
            state["cancelling_at"] = event.timestamp
            pending_persists.append((state["session_id"], "cancelling"))

    def _get_display_name(self, step: str) -> str:
        """Get human-readable display name for step."""
        display_map = {
            "pipeline_startup": "Starting Pipeline Worker",
            "prepare_uvs": "Preparing UV Coordinates",
            "discover_materials": "Discovering Materials",
            "plan_textures": "Planning Bounded Texture Work",
            "generate_prompts": "Generating Texture Prompts",
            "render_previews": "Rendering Material Previews",
            "generate_textures": "Generating PBR Textures",
            "blend_textures": "Blending Textures",
            "apply_textures": "Applying Textures to USD",
            "render": "Rendering Final Output",
        }
        return display_map.get(step, step)

    def _update_overall_progress(
        self, state: dict, step: str, step_percent: int
    ) -> None:
        """Update overall progress based on current step progress.

        Uses weighted allocation across 9 texture pipeline steps:
        - prepare_uvs: 0-3%
        - discover_materials: 3-5%
        - plan_textures: 5-7%
        - generate_prompts: 7-12%
        - render_previews: 12-20%
        - generate_textures: 20-75% (dominant cost)
        - blend_textures: 75-85%
        - apply_textures: 85-95%
        - render: 95-100%
        """
        step_weights = {
            "pipeline_startup": (0, 0),
            "prepare_uvs": (0, 3),
            "discover_materials": (3, 5),
            "plan_textures": (5, 7),
            "generate_prompts": (7, 12),
            "render_previews": (12, 20),
            "generate_textures": (20, 75),
            "blend_textures": (75, 85),
            "apply_textures": (85, 95),
            "render": (95, 100),
        }

        if step in step_weights:
            start, end = step_weights[step]
            overall = start + int((end - start) * step_percent / 100)
            state["overall_progress"]["percent"] = min(100, overall)

    def _update_overall_progress_on_completion(
        self,
        state: dict,
        step: str,
        pending_persists: list[tuple[str, str]],
    ) -> None:
        """Update overall progress when a step completes."""
        completion_percent = {
            "prepare_uvs": 3,
            "discover_materials": 5,
            "plan_textures": 7,
            "generate_prompts": 12,
            "render_previews": 20,
            "generate_textures": 75,
            "blend_textures": 85,
            "apply_textures": 95,
            "render": 100,
        }

        if step in completion_percent:
            state["overall_progress"]["percent"] = completion_percent[step]

        completed_count = len(state["completed_steps"])
        state["overall_progress"]["current_step"] = completed_count

        # A final step can reach 100% before post-processing artifacts have
        # been synced to shared storage. Persist the terminal `completed`
        # status only when the executor emits the explicit pipeline-completed
        # event after artifact sync succeeds.

    async def _persist_status(self, session_id: str, status: str) -> None:
        """Persist session status to SessionManager on disk."""
        if self._session_manager is None:
            return
        try:
            manager = self._session_manager
            get_session_metadata = getattr(manager, "get_session_metadata", None)
            metadata = (
                await asyncio.to_thread(get_session_metadata, session_id)
                if callable(get_session_metadata)
                else {}
            )
            current_status = (metadata or {}).get("status")
            if (
                current_status in _TERMINAL_STATUSES
                and status not in _TERMINAL_STATUSES
            ):
                logger.info(
                    "Skipping %s status persist for terminal session %s (%s)",
                    status,
                    session_id,
                    current_status,
                )
                return
            await asyncio.to_thread(
                manager.update_session,
                session_id,
                {"status": status},
            )
            logger.info(f"Persisted {status} status for session {session_id}")

        except Exception as e:
            logger.warning(f"Failed to persist {status} status: {e}")

    async def _persist_live_metadata(
        self,
        session_id: str,
        updates: dict[str, Any],
    ) -> None:
        """Persist live progress fields without rewriting the global index."""
        if self._session_manager is None:
            return
        try:
            manager = self._session_manager

            def _update() -> None:
                try:
                    manager.update_session(
                        session_id,
                        updates,
                        update_index=False,
                    )
                except TypeError as exc:
                    if "update_index" not in str(exc):
                        raise
                    manager.update_session(session_id, updates)

            await asyncio.to_thread(_update)
        except Exception as e:
            logger.warning(f"Failed to persist live metadata: {e}")

    async def _save_event_to_log(self, event: ProgressEvent) -> None:
        """Save event to persistent log file for replay."""
        if self._session_manager is None:
            return
        try:
            manager = self._session_manager
            event_dict = event.model_dump()
            await asyncio.to_thread(
                manager.append_event,
                event.session_id,
                event_dict,
            )

        except Exception as e:
            logger.debug(f"Failed to save event to log: {e}")

    async def cleanup_session(self, session_id: str) -> None:
        """Drop session state from the bus and notify any attached subscriber.

        DELETE /sessions/{sid} calls this after the on-disk dir is gone. A
        client that opened /pipeline/{sid}/events before the DELETE already
        holds a reference to the per-session asyncio.Queue; once we pop it
        from ``self._queues``, the next ``self.get_queue()`` from a worker
        emit() will create a fresh queue and any further events miss the
        subscriber entirely. To keep that subscriber from blocking on
        ``queue.get()`` forever (receiving only the 30 s keepalive pings),
        push a terminal CANCELLED sentinel onto the existing queue first
        so ``stream_progress_events`` emits its ``done`` event and closes.
        """
        async with self._lock:
            self._deleted_sessions.add(session_id)
            self._shared_session_exists_checked_at.pop(session_id, None)
            queue = self._queues.get(session_id)
            if queue is not None:
                # Drain any stale RUNNING/COMPLETED backlog first so the
                # terminal sentinel is the next event the subscriber sees,
                # not the (N+1)th. Without this drain, a slow client with a
                # backlog yields every historic progress event for an
                # already-deleted session before reaching the close branch.
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except (
                        asyncio.QueueEmpty
                    ):  # pragma: no cover - defensive race guard
                        break
                sentinel = ProgressEvent(
                    session_id=session_id,
                    step="pipeline",
                    state=StepState.CANCELLED,
                    message="Session deleted",
                )
                try:
                    queue.put_nowait(sentinel)
                except asyncio.QueueFull:  # pragma: no cover - unbounded queue guard
                    # Unbounded asyncio.Queue() does not raise QueueFull, but
                    # guard anyway -- a stuck subscriber is preferable to a
                    # raised exception inside DELETE.
                    logger.warning(
                        f"Could not enqueue delete sentinel for {session_id} "
                        f"(queue full); subscriber may rely on disk recheck."
                    )
            self._queues.pop(session_id, None)
            self._state.pop(session_id, None)
            self._worker_heartbeat_at.pop(session_id, None)
            self._live_metadata_flush_at.pop(session_id, None)
            self._live_metadata_pending.pop(session_id, None)
            task = self._live_metadata_tasks.pop(session_id, None)
            if task is not None:
                task.cancel()

    async def cleanup_orphaned_sessions(self) -> list[str]:
        """Drop local bus state for sessions removed by another service instance."""
        if self._session_manager is None:
            return []

        async with self._lock:
            session_ids = set(self._state) | set(self._queues)

        cleaned: list[str] = []
        for session_id in sorted(session_ids):
            try:
                exists = await asyncio.to_thread(
                    self._session_manager.session_exists,
                    session_id,
                )
            except Exception as exc:
                logger.warning(
                    "Skipping orphaned bus cleanup for %s: %s",
                    session_id,
                    exc,
                )
                continue
            if not exists:
                await self.cleanup_session(session_id)
                cleaned.append(session_id)
        return cleaned


# Global singleton event bus
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def init_event_bus(session_manager: Any) -> EventBus:
    """Initialize the global event bus with a shared session manager."""
    global _event_bus
    _event_bus = EventBus(session_manager=session_manager)
    return _event_bus
