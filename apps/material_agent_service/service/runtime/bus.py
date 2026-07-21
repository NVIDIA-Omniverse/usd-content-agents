# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event bus for pipeline progress events."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)

from ..json_utils import to_json_safe
from .events import ProgressEvent, StepState

if TYPE_CHECKING:  # pragma: no cover
    from ..session.manager import RegenerationClaim, SessionManager

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"completed", "succeeded", "failed", "cancelled", "canceled"}

SCENE_STEP_METADATA: dict[str, dict[str, int | str | tuple[int, int]]] = {
    "scene_analyze": {
        "display_name": "Analyzing Large Scene",
        "running_range": (0, 10),
        "completion_percent": 10,
    },
    "scene_extract": {
        "display_name": "Extracting Scene Assets",
        "running_range": (10, 25),
        "completion_percent": 25,
    },
    "scene_run_assets": {
        "display_name": "Running Asset Pipelines",
        "running_range": (25, 70),
        "completion_percent": 70,
    },
    "scene_run_payloads": {
        "display_name": "Running Payload Pipelines",
        "running_range": (70, 78),
        "completion_percent": 78,
    },
    "scene_reconcile": {
        "display_name": "Reconciling Scene Predictions",
        "running_range": (78, 84),
        "completion_percent": 84,
    },
    "scene_harmonize": {
        "display_name": "Harmonizing Scene Predictions",
        "running_range": (84, 90),
        "completion_percent": 90,
    },
    "scene_collect": {
        "display_name": "Composing Scene Output",
        "running_range": (90, 96),
        "completion_percent": 96,
    },
    "scene_render": {
        "display_name": "Rendering Composed Scene",
        "running_range": (96, 99),
        "completion_percent": 99,
    },
    "scene_validate": {
        "display_name": "Validating Scene Output",
        "running_range": (99, 100),
        "completion_percent": 100,
    },
}


@dataclass(frozen=True)
class EventBusSessionSnapshot:
    """Restorable EventBus state captured before transactional preparation."""

    state_present: bool
    state: dict[str, Any] | None
    queue_present: bool
    queue: asyncio.Queue[ProgressEvent] | None
    queued_events: tuple[ProgressEvent, ...]
    regeneration_claim: RegenerationClaim | None


class EventBus:
    """Central event bus for pipeline progress.

    Manages:
    - Per-session event queues for SSE streaming
    - Canonical in-memory state snapshot for /status API
    - Event application logic to update state
    """

    def __init__(self) -> None:
        """Initialize event bus."""
        # Per-session event queues for SSE subscribers
        self._queues: dict[str, asyncio.Queue[ProgressEvent]] = {}

        # Canonical in-memory state (what /status reads)
        self._state: dict[str, dict[str, Any]] = {}

        # Lock for thread-safe state updates
        self._lock = asyncio.Lock()

        # Session manager reference (set by main app during startup)
        self._session_manager: SessionManager | None = None
        self._regeneration_claims: dict[str, RegenerationClaim] = {}

    def set_session_manager(self, manager: SessionManager) -> None:
        """Set the session manager instance for persistence.

        Args:
            manager: SessionManager instance with configured storage backend
        """
        self._session_manager = manager

    def _get_session_manager(self) -> SessionManager | None:
        """Get the session manager, falling back to creating a new one if needed.

        Returns:
            SessionManager instance or None if unavailable
        """
        if self._session_manager is not None:
            return self._session_manager

        # Fallback: create a local-only session manager (legacy behavior)
        try:
            from ..config import ServiceConfig
            from ..session.manager import SessionManager

            config = ServiceConfig()
            return SessionManager(
                storage_path=config.session_storage_path,
                ttl_hours=config.session_ttl_hours,
            )
        except Exception:
            log_durable_failure(
                logger,
                "event_bus_session_manager_unavailable",
                phase=FailurePhase.LOCAL_PUBLICATION,
                retryable=True,
            )
            return None

    def get_queue(self, session_id: str) -> asyncio.Queue[ProgressEvent]:
        """Get or create event queue for a session.

        Args:
            session_id: Session identifier

        Returns:
            Event queue for the session
        """
        if session_id not in self._queues:
            self._queues[session_id] = asyncio.Queue()
        return self._queues[session_id]

    def bind_regeneration_claim(
        self,
        session_id: str,
        claim: RegenerationClaim,
    ) -> None:
        """Fence durable EventBus status writes to one regeneration claim."""
        self._regeneration_claims[session_id] = claim

    def get_snapshot(self, session_id: str) -> dict[str, Any] | None:
        """Get current in-memory state snapshot for a session.

        This is what the /status endpoint reads (no disk I/O).

        Args:
            session_id: Session identifier

        Returns:
            State snapshot or None if session not found
        """
        return self._state.get(session_id)

    async def _unbound_state_is_current(self, session_id: str) -> bool:
        """Reject an old standard-run cache once any regeneration owns history."""
        manager = self._session_manager
        if manager is None or not hasattr(manager, "get_session_metadata"):
            # Preserve legacy standalone EventBus use where no durable session
            # manager is configured. Production wires one during app startup.
            return True
        metadata = await manager.get_session_metadata(session_id)
        if metadata is None:
            return False
        return not isinstance(metadata.get("regeneration_claim"), Mapping)

    async def _bound_claim_is_current(self, session_id: str) -> bool:
        """Return whether this pod's bound claim owns a live session lease."""
        claim = self._regeneration_claims.get(session_id)
        if claim is None:
            return await self._unbound_state_is_current(session_id)
        manager = self._get_session_manager()
        if manager is None:
            return False
        metadata = await manager.get_session_metadata(session_id)
        if metadata is None:
            return False
        from ..session.manager import RegenerationClaim

        current = RegenerationClaim.from_metadata(metadata)
        raw_claim = metadata.get("regeneration_claim")
        if (
            current is None
            or not isinstance(raw_claim, Mapping)
            or current.generation != claim.generation
            or current.token != claim.token
        ):
            return False
        return raw_claim.get("active") is True and (
            current.lease_expires_at > datetime.now(UTC)
        )

    async def event_is_current(self, event: ProgressEvent) -> bool:
        """Allow live claim events or a persisted matching terminal event."""
        claim = self._regeneration_claims.get(event.session_id)
        if claim is None:
            manager = self._session_manager
            if manager is None or not hasattr(manager, "get_session_metadata"):
                # Preserve legacy standalone EventBus use where no durable
                # manager is configured. Production wires one at startup.
                return True
            metadata = await manager.get_session_metadata(event.session_id)
            if metadata is None or isinstance(
                metadata.get("regeneration_claim"), Mapping
            ):
                return False
            persisted_status = metadata.get("status")
            if persisted_status not in _TERMINAL_STATUSES:
                return True
            # A standard or scene run has no claim token, so durable terminal
            # metadata is its close barrier. Only the executor-authored event
            # matching that terminal state may pass after persistence; delayed
            # listener events must not reopen the snapshot or event log.
            return self._event_matches_persisted_terminal(event, persisted_status)
        manager = self._get_session_manager()
        if manager is None:
            return False
        metadata = await manager.get_session_metadata(event.session_id)
        if metadata is None:
            return False
        from ..session.manager import RegenerationClaim

        current = RegenerationClaim.from_metadata(metadata)
        raw_claim = metadata.get("regeneration_claim")
        if (
            current is None
            or not isinstance(raw_claim, Mapping)
            or current.generation != claim.generation
            or current.token != claim.token
        ):
            return False
        if raw_claim.get("active") is True:
            return current.lease_expires_at > datetime.now(UTC)

        return self._event_matches_persisted_terminal(
            event,
            metadata.get("status"),
        )

    @staticmethod
    def _event_matches_persisted_terminal(
        event: ProgressEvent,
        persisted_status: object,
    ) -> bool:
        """Accept only an executor-authored event matching durable terminal state."""
        if event.state == StepState.CANCELLED:
            return persisted_status in {"cancelled", "canceled"} and bool(
                event.extra and event.extra.get("pipeline_cancelled")
            )
        if event.state == StepState.FAILED:
            return persisted_status == "failed" and bool(
                event.extra and event.extra.get("pipeline_failed")
            )
        return (
            event.state == StepState.COMPLETED
            and persisted_status in {"completed", "succeeded"}
            and bool(
                event.extra
                and event.extra.get("pipeline_completed")
                and "coverage" in event.extra
            )
        )

    async def queued_event_is_current(self, event: ProgressEvent) -> bool:
        """Validate an already-queued event without dropping same-run backlog."""
        claim = self._regeneration_claims.get(event.session_id)
        if claim is None:
            return await self._unbound_state_is_current(event.session_id)
        manager = self._get_session_manager()
        if manager is None:
            return False
        metadata = await manager.get_session_metadata(event.session_id)
        if metadata is None:
            return False
        from ..session.manager import RegenerationClaim

        current = RegenerationClaim.from_metadata(metadata)
        raw_claim = metadata.get("regeneration_claim")
        if (
            current is None
            or not isinstance(raw_claim, Mapping)
            or current.generation != claim.generation
            or current.token != claim.token
        ):
            return False
        if raw_claim.get("active") is True:
            return current.lease_expires_at > datetime.now(UTC)
        if raw_claim.get("aborted_at"):
            return False
        finalized_at_value = raw_claim.get("finalized_at")
        if not isinstance(finalized_at_value, str):
            return False
        try:
            finalized_at = datetime.fromisoformat(
                finalized_at_value.replace("Z", "+00:00")
            )
        except ValueError:
            return False
        # A normal worker finalizes before its lease expires. Expired-claim
        # recovery finalizes after expiry and must not drain the old pod's queue.
        return current.lease_expires_at > finalized_at and metadata.get("status") in {
            "completed",
            "succeeded",
            "failed",
            "cancelled",
            "canceled",
        }

    async def get_fenced_snapshot(self, session_id: str) -> dict[str, Any] | None:
        """Return local state only while its regeneration claim is current."""
        if not await self._bound_claim_is_current(session_id):
            return None
        return self.get_snapshot(session_id)

    async def seed_pending_session(
        self,
        session_id: str,
        regeneration_claim: RegenerationClaim | None = None,
    ) -> None:
        """Seed a lightweight local status snapshot for an accepted pipeline.

        This intentionally avoids event-log persistence and SSE queue writes so
        /pipeline can make same-instance /status polling independent of the
        backing metadata store immediately after the job is accepted.
        """
        timestamp = datetime.now(UTC).isoformat()
        async with self._lock:
            state = self._new_state(session_id, timestamp)
            existing = self._state.get(session_id)
            if existing and existing.get("created_at"):
                state["created_at"] = existing["created_at"]
            self._state[session_id] = state
            if regeneration_claim is None:
                self._regeneration_claims.pop(session_id, None)
            else:
                self._regeneration_claims[session_id] = regeneration_claim
            queue = self.get_queue(session_id)
            while not queue.empty():
                queue.get_nowait()

    async def capture_session(self, session_id: str) -> EventBusSessionSnapshot:
        """Capture one session's state and queue for transactional rollback."""
        async with self._lock:
            queue = self._queues.get(session_id)
            queued_events = (
                tuple(
                    cast(
                        deque[ProgressEvent],
                        getattr(queue, "_queue"),
                    )
                )
                if queue is not None
                else ()
            )
            return EventBusSessionSnapshot(
                state_present=session_id in self._state,
                state=copy.deepcopy(self._state.get(session_id)),
                queue_present=queue is not None,
                queue=queue,
                queued_events=queued_events,
                regeneration_claim=self._regeneration_claims.get(session_id),
            )

    async def restore_session(
        self,
        session_id: str,
        snapshot: EventBusSessionSnapshot,
    ) -> None:
        """Restore state captured by :meth:`capture_session`."""
        async with self._lock:
            if snapshot.state_present:
                assert snapshot.state is not None
                self._state[session_id] = copy.deepcopy(snapshot.state)
            else:
                self._state.pop(session_id, None)

            current_queue = self._queues.get(session_id)
            if snapshot.queue_present:
                assert snapshot.queue is not None
                queue = snapshot.queue
                self._queues[session_id] = queue
                while not queue.empty():
                    queue.get_nowait()
                for event in snapshot.queued_events:
                    queue.put_nowait(event)
            elif current_queue is not None:
                while not current_queue.empty():
                    current_queue.get_nowait()
                self._queues.pop(session_id, None)
            if snapshot.regeneration_claim is None:
                self._regeneration_claims.pop(session_id, None)
            else:
                self._regeneration_claims[session_id] = snapshot.regeneration_claim

    async def emit(self, event: ProgressEvent) -> None:
        """Emit an event: update state and queue for subscribers.

        Args:
            event: Progress event to emit
        """
        async with self._lock:
            if not await self.event_is_current(event):
                logger.info(
                    "Dropped stale claimed event for session %s",
                    event.session_id[:8],
                )
                return
            await self._emit_locked(event)

    @staticmethod
    def _claim_identity_matches(
        bound_claim: RegenerationClaim | None,
        expected_claim: RegenerationClaim | None,
    ) -> bool:
        """Compare an EventBus binding with an explicit originating owner."""
        if bound_claim is None or expected_claim is None:
            return bound_claim is expected_claim
        return (
            bound_claim.generation == expected_claim.generation
            and bound_claim.token == expected_claim.token
        )

    async def emit_for_owner(
        self,
        event: ProgressEvent,
        *,
        regeneration_claim: RegenerationClaim | None,
    ) -> bool:
        """Emit a run-scoped event only for its still-bound owner.

        ``None`` explicitly means an unbound standard or scene run.  Ownership
        is checked under the same lock used to seed a later regeneration so an
        old event cannot poison a newly rebound queue or snapshot.
        """
        async with self._lock:
            bound_claim = self._regeneration_claims.get(event.session_id)
            if not self._claim_identity_matches(bound_claim, regeneration_claim):
                logger.info(
                    "Dropped event from superseded owner for session %s",
                    event.session_id[:8],
                )
                return False
            if not await self.event_is_current(event):
                logger.info(
                    "Dropped stale claimed event for session %s",
                    event.session_id[:8],
                )
                return False
            await self._emit_locked(event)
            return True

    async def _emit_locked(self, event: ProgressEvent) -> None:
        """Apply, enqueue, and persist an event while ``_lock`` is held."""
        # Update canonical state
        await self._apply_event_to_state(event)

        # Enrich event with overall progress from state
        state = self._state.get(event.session_id)
        if state:
            event.overall_percent = state.get("overall_progress", {}).get("percent", 0)
            logger.info(
                f"[EventBus] {event.session_id[:8]}... {event.step}: "
                f"step={event.percent}% → overall={event.overall_percent}% (state={event.state.value})"
            )
        else:
            event.overall_percent = 0
            logger.warning(
                f"[EventBus] No state found for {event.session_id[:8]}... - setting overall_percent=0"
            )

        # Queue enriched event for SSE subscribers
        queue = self.get_queue(event.session_id)
        await queue.put(event)

        # Persist event to disk for replay when viewing old sessions
        await self._save_event_to_log(event)

    async def _apply_event_to_state(self, event: ProgressEvent) -> None:
        """Apply event to update canonical in-memory state.

        Args:
            event: Progress event
        """
        session_id = event.session_id

        # Initialize state if new session
        if session_id not in self._state:
            self._state[session_id] = self._new_state(session_id, event.timestamp)

        state = self._state[session_id]
        state["updated_at"] = event.timestamp
        if event.extra:
            total_steps = event.extra.get("total_steps")
            if isinstance(total_steps, int) and total_steps > 0:
                state["overall_progress"]["total_steps"] = total_steps
            rendered_images = event.extra.get("rendered_images")
            if isinstance(rendered_images, list):
                for image in rendered_images:
                    if not isinstance(image, str):
                        continue
                    image_name = image.rstrip("/").rsplit("/", 1)[-1]
                    if image_name and image_name not in state["preview_images"]:
                        state["preview_images"].append(image_name)

        # Handle state transitions
        if event.state == StepState.RUNNING:
            if (
                state.get("_pipeline_terminal")
                and state.get("status") in _TERMINAL_STATUSES
            ):
                self._reset_run_state(state, event.timestamp)

            # Step started
            if (
                state.get("current_step") is None
                or state["current_step"].get("name") != event.step
            ):
                # New step started
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
                # Persist "running" status on first transition from pending
                if state["status"] == "pending":
                    await self._persist_status(state["session_id"], "running")
                state["status"] = "running"
                scene_steps = self._scene_step_order()
                if event.step in scene_steps:
                    state["overall_progress"]["total_steps"] = len(scene_steps)
            else:
                # Update existing step progress
                state["current_step"]["progress"] = {
                    "current": event.current or 0,
                    "total": event.total or 1,
                    "percent": event.percent or 0,
                    "message": event.message or "",
                }
                # Update elapsed time
                started_at_str = state["current_step"]["started_at"].replace("Z", "")
                started_at = datetime.fromisoformat(started_at_str)
                now = datetime.fromisoformat(event.timestamp.replace("Z", ""))
                state["current_step"]["elapsed_seconds"] = int(
                    (now - started_at).total_seconds()
                )

            # Update overall progress based on step and percent
            self._update_overall_progress(state, event.step, event.percent or 0)

        elif event.state == StepState.COMPLETED:
            # A workflow-level completion marker is provisional until the
            # executor attaches the persisted coverage contract. Do not count
            # that synthetic ``pipeline`` marker as a completed workflow step.
            if (
                event.extra
                and event.extra.get("pipeline_completed")
                and "coverage" not in event.extra
            ):
                return

            # Step completed
            if (
                state.get("current_step")
                and state["current_step"]["name"] == event.step
            ):
                started_at_str = state["current_step"]["started_at"].replace("Z", "")
                started_at = datetime.fromisoformat(started_at_str)
                now = datetime.fromisoformat(event.timestamp.replace("Z", ""))
                duration = int((now - started_at).total_seconds())

                # Add to completed steps
                completed_step = {
                    "name": event.step,
                    "display_name": state["current_step"]["display_name"],
                    "started_at": state["current_step"]["started_at"],
                    "completed_at": event.timestamp,
                    "duration_seconds": duration,
                    "stats": to_json_safe(event.extra or {}),
                }
                state["completed_steps"].append(completed_step)

                # Store timing
                state["step_timings"][event.step] = duration

                # Clear current step
                state["current_step"] = None

                # Update overall progress
                await self._update_overall_progress_on_completion(state, event.step)

            # Only executor-authored completion events carrying the persisted
            # coverage contract are authoritative.  Underlying workflows also
            # emit ``pipeline_completed`` before the service has qualified and
            # atomically stored its final result.
            elif (
                event.extra
                and event.extra.get("pipeline_completed")
                and "coverage" in event.extra
            ):
                # This is the authoritative pipeline completion event - force
                # progress to 100%.
                state["overall_progress"]["percent"] = 100
                state["status"] = "completed"
                state["completed_at"] = datetime.now(UTC).isoformat()
                state["current_step"] = None
                state["_pipeline_terminal"] = True
                completed_count = len(state.get("completed_steps", []))
                total_steps = max(
                    int(state.get("overall_progress", {}).get("total_steps", 0) or 0),
                    completed_count,
                )
                state["overall_progress"]["current_step"] = completed_count
                state["overall_progress"]["total_steps"] = total_steps
                await self._persist_status(state["session_id"], "completed")
            else:
                # Fast terminal steps can complete after the executor has
                # already emitted the pipeline completion event. Preserve them
                # in the status snapshot instead of dropping the event.
                completed_names = {
                    step.get("name")
                    for step in state.get("completed_steps", [])
                    if isinstance(step, dict)
                }
                if event.step not in completed_names:
                    completed_step = {
                        "name": event.step,
                        "display_name": self._get_display_name(event.step),
                        "started_at": event.timestamp,
                        "completed_at": event.timestamp,
                        "duration_seconds": 0,
                        "stats": to_json_safe(event.extra or {}),
                    }
                    state["completed_steps"].append(completed_step)
                    state["step_timings"][event.step] = 0
                    await self._update_overall_progress_on_completion(state, event.step)

        elif event.state == StepState.FAILED:
            # Step failed
            state["status"] = "failed"
            state["error"] = event.message or "Unknown error"
            state["failed_step"] = event.step
            state["failed_at"] = event.timestamp
            state["_pipeline_terminal"] = True

            # Persist failed status to disk
            await self._persist_status(state["session_id"], "failed")

        elif event.state == StepState.CANCELLED:
            # Step cancelled
            state["status"] = "cancelled"
            state["cancelled_at"] = event.timestamp
            state["_pipeline_terminal"] = True

            # Persist cancelled status to disk
            await self._persist_status(state["session_id"], "cancelled")

    def _new_state(self, session_id: str, timestamp: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "status": "pending",
            "created_at": timestamp,
            "updated_at": timestamp,
            "current_step": None,
            "completed_steps": [],
            "preview_images": [],
            "overall_progress": {
                "current_step": 0,
                "total_steps": 3,  # render, predict, apply
                "percent": 0,
            },
            "step_timings": {},
            "_pipeline_terminal": False,
        }

    def _reset_run_state(self, state: dict[str, Any], timestamp: str) -> None:
        """Clear run-scoped progress when a session ID starts another pipeline."""
        state["status"] = "pending"
        state["updated_at"] = timestamp
        state["current_step"] = None
        state["completed_steps"] = []
        state["preview_images"] = []
        state["overall_progress"] = {
            "current_step": 0,
            "total_steps": 3,
            "percent": 0,
        }
        state["step_timings"] = {}
        state["_pipeline_terminal"] = False
        for key in (
            "completed_at",
            "failed_at",
            "cancelled_at",
            "error",
            "failed_step",
        ):
            state.pop(key, None)

    def _get_display_name(self, step: str) -> str:
        """Get human-readable display name for step.

        Args:
            step: Step internal name

        Returns:
            Display name
        """
        display_map = {
            "build_dataset_usd": "Rendering USD Scene",
            "build_dataset_prepare_dataset": "Preparing Dataset",
            "cluster_prims": "Clustering Prims",
            "expand_cluster_predictions": "Expanding Cluster Predictions",
            "prepare_dataset": "Preparing Dataset",
            "predict": "Running VLM Predictions",
            "apply": "Applying Materials",
            "render": "Rendering Final Output",
        }
        scene_metadata = SCENE_STEP_METADATA.get(step)
        if scene_metadata:
            return str(scene_metadata["display_name"])
        return display_map.get(step, step)

    def _scene_step_order(self) -> tuple[str, ...]:
        """Return large-scene service progress step order."""
        return tuple(SCENE_STEP_METADATA)

    def _is_cluster_progress_active(self, state: dict, step: str) -> bool:
        """Return whether the current run includes prim clustering steps."""
        if step in {"cluster_prims", "expand_cluster_predictions"}:
            return True
        completed_steps = state.get("completed_steps", [])
        return any(
            isinstance(completed_step, dict)
            and completed_step.get("name")
            in {"cluster_prims", "expand_cluster_predictions"}
            for completed_step in completed_steps
        )

    def _update_overall_progress(
        self, state: dict, step: str, step_percent: int
    ) -> None:
        """Update overall progress based on current step progress.

        Uses weighted allocation for the full material pipeline. Optional
        clustering steps get their own ranges so status does not stall during
        large-scene deduplication runs.

        Args:
            state: Session state dictionary
            step: Current step name
            step_percent: Progress percentage within step (0-100)
        """
        cluster_active = self._is_cluster_progress_active(state, step)
        step_weights = {
            "build_dataset_usd": (0, 45),  # 0-45%
            "prepare_dataset": (45, 45),  # Instant (part of render phase)
            "build_dataset_prepare_dataset": (45, 45),  # Instant
            "predict": (45, 80),  # 45-80%
            "apply": (80, 95),  # 80-95%
            "render": (95, 100),  # 95-100% (final render)
        }
        if cluster_active:
            step_weights.update(
                {
                    "cluster_prims": (45, 60),  # 45-60%
                    "predict": (60, 85),  # 60-85%
                    "expand_cluster_predictions": (85, 90),  # 85-90%
                    "apply": (90, 98),  # 90-98%
                    "render": (98, 100),  # 98-100%
                }
            )
        step_weights.update(
            {
                step: metadata["running_range"]
                for step, metadata in SCENE_STEP_METADATA.items()
            }
        )

        if step in step_weights:
            start, end = step_weights[step]
            # Scale step progress to overall range
            overall = start + int((end - start) * step_percent / 100)
            state["overall_progress"]["percent"] = min(100, overall)
            scene_steps = self._scene_step_order()
            if step in scene_steps:
                state["overall_progress"]["current_step"] = scene_steps.index(step) + 1

    async def _update_overall_progress_on_completion(
        self, state: dict, step: str
    ) -> None:
        """Update overall progress when a step completes.

        Args:
            state: Session state dictionary
            step: Completed step name
        """
        # Set to the end of the step's range
        # Note: 'apply' is 100% since 'render' is optional and may not run
        completion_percent = {
            "build_dataset_usd": 45,
            "build_dataset_prepare_dataset": 45,
            "cluster_prims": 60,
            "predict": 85,
            "expand_cluster_predictions": 90,
            "apply": 100,  # Final step (render is optional)
            "render": 100,  # Also final if it runs
        }
        completion_percent.update(
            {
                step: metadata["completion_percent"]
                for step, metadata in SCENE_STEP_METADATA.items()
            }
        )

        scene_current_step_set = False
        if step in completion_percent:
            state["overall_progress"]["percent"] = max(
                int(state["overall_progress"].get("percent", 0)),
                completion_percent[step],
            )
            scene_steps = self._scene_step_order()
            if step in scene_steps:
                state["overall_progress"]["current_step"] = scene_steps.index(step) + 1
                scene_current_step_set = True
                state["overall_progress"]["total_steps"] = len(scene_steps)

        # Update step counter
        completed_count = len(state["completed_steps"])
        if not scene_current_step_set:
            total_steps = max(
                int(state.get("overall_progress", {}).get("total_steps", 0) or 0),
                completed_count,
            )
            state["overall_progress"]["current_step"] = completed_count
            state["overall_progress"]["total_steps"] = total_steps

        # A step reaching 100% is progress, not authoritative pipeline
        # completion.  Coverage qualification and result persistence still run
        # after apply/render, and the executor emits the terminal event only
        # after those artifacts have been stored atomically.

    async def _persist_status(self, session_id: str, status: str) -> None:
        """Persist session status to SessionManager on disk.

        Args:
            session_id: Session identifier
            status: Status to persist (completed, failed, cancelled)
        """
        try:
            manager = self._get_session_manager()
            if manager is None:
                logger.warning(f"No session manager available to persist {status}")
                return

            if await manager.session_exists(session_id):
                claim = self._regeneration_claims.get(session_id)
                if claim is None:
                    await manager.update_session(session_id, {"status": status})
                    persisted = True
                else:
                    persisted = await manager.update_session_for_claim(
                        session_id,
                        claim,
                        {"status": status},
                    )
                if persisted:
                    logger.info(
                        f"Persisted {status} status for session {session_id[:8]}"
                    )

        except Exception:
            log_durable_failure(
                logger,
                "event_bus_status_persistence_failed",
                phase=FailurePhase.LOCAL_PUBLICATION,
                retryable=True,
            )

    async def _save_event_to_log(self, event: ProgressEvent) -> None:
        """Save event to persistent log file for replay.

        Args:
            event: Progress event to save
        """
        try:
            manager = self._get_session_manager()
            if manager is None:
                return

            if await manager.session_exists(event.session_id):
                session_dir = manager.get_session_dir(event.session_id)
                log_file = session_dir / "event_log.jsonl"

                # Append event to log file (one JSON object per line)
                event_dict = to_json_safe(event.model_dump())

                def _write() -> None:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(event_dict) + "\n")

                await asyncio.to_thread(_write)

        except Exception:
            log_durable_failure(
                logger,
                "event_log_persistence_failed",
                phase=FailurePhase.LOCAL_PUBLICATION,
                retryable=True,
            )

    def cleanup_session(self, session_id: str) -> None:
        """Clean up session from event bus.

        Args:
            session_id: Session identifier
        """
        if session_id in self._queues:
            del self._queues[session_id]
        if session_id in self._state:
            del self._state[session_id]
        self._regeneration_claims.pop(session_id, None)


# Global singleton event bus
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance.

    Returns:
        Global EventBus instance
    """
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
