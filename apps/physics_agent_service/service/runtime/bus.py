# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event bus for pipeline progress events.

Manages in-memory state for the LOCAL instance only. Cross-instance state
is handled by the SessionStore (S3). The EventBus provides:
- Per-session SSE queues (for clients connected to this instance)
- Fast in-memory state snapshots (avoids store reads for the executing instance)
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from .events import ProgressEvent, StepState
from .progress import (
    STEP_COMPLETION_PERCENT,
    STEP_DISPLAY_NAMES,
    STEP_NUMBER,
    STEP_WEIGHTS,
    TOTAL_VISIBLE_STEPS,
)

logger = logging.getLogger(__name__)


class EventBus:
    """Local-instance event bus for pipeline progress.

    Manages:
    - Per-session event queues for SSE streaming
    - Canonical in-memory state snapshot for /status API (fast path)
    """

    def __init__(self):
        self._queues: dict[str, asyncio.Queue[ProgressEvent]] = {}
        self._state: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def get_queue(self, session_id: str) -> asyncio.Queue[ProgressEvent]:
        """Get or create event queue for a session."""
        if session_id not in self._queues:
            self._queues[session_id] = asyncio.Queue()
        return self._queues[session_id]

    def get_snapshot(self, session_id: str) -> dict[str, Any] | None:
        """Get current in-memory state snapshot for a session."""
        return self._state.get(session_id)

    async def seed_pending_session(
        self,
        session_id: str,
        *,
        created_at: str | None = None,
    ) -> None:
        """Seed a lightweight local status snapshot for an accepted pipeline.

        This avoids any backing-store access so same-instance status polling can
        return immediately after /pipeline accepts a job, before the worker has
        emitted its first progress event.
        """
        timestamp = datetime.now(UTC).isoformat()
        async with self._lock:
            existing = self._state.get(session_id)
            self._state[session_id] = {
                "session_id": session_id,
                "status": "pending",
                "created_at": created_at
                or (existing.get("created_at", timestamp) if existing else timestamp),
                "updated_at": timestamp,
                "current_step": None,
                "completed_steps": [],
                "overall_progress": {
                    "current_step": 0,
                    "total_steps": TOTAL_VISIBLE_STEPS,
                    "percent": 0,
                },
                "preview_images": [],
                "step_timings": {},
            }

    async def mark_cancelling(self, session_id: str) -> None:
        """Expose an accepted local cancellation while its worker drains."""
        timestamp = datetime.now(UTC).isoformat()
        async with self._lock:
            state = self._state.get(session_id)
            if state is None or state.get("status") in {
                "completed",
                "failed",
                "cancelled",
            }:
                return
            state["status"] = "cancelling"
            state["updated_at"] = timestamp

    async def emit(self, event: ProgressEvent) -> None:
        """Emit an event: update local state and queue for SSE subscribers."""
        async with self._lock:
            self._apply_event_to_state(event)

            state = self._state.get(event.session_id)
            if state:
                event.overall_percent = state.get("overall_progress", {}).get(
                    "percent", 0
                )
                logger.info(
                    f"[EventBus] {event.session_id[:8]}... {event.step}: "
                    f"step={event.percent}% → overall={event.overall_percent}% (state={event.state.value})"
                )
            else:
                event.overall_percent = 0

            queue = self.get_queue(event.session_id)
            await queue.put(event)

    def _apply_event_to_state(self, event: ProgressEvent) -> None:
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
                    "total_steps": TOTAL_VISIBLE_STEPS,
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

                self._update_overall_progress_on_completion(state, event.step)

            elif (
                event.extra
                and event.extra.get("pipeline_completed")
                and state.get("status") not in {"cancelling", "cancelled", "failed"}
            ):
                state["overall_progress"]["percent"] = 100
                state["status"] = "completed"
                state["completed_at"] = datetime.utcnow().isoformat()
                state["current_step"] = None

        elif event.state == StepState.FAILED:
            state["status"] = "failed"
            state["error"] = event.message or "Unknown error"
            state["failed_step"] = event.step
            state["failed_at"] = event.timestamp

        elif event.state == StepState.CANCELLED:
            state["status"] = "cancelled"
            state["cancelled_at"] = event.timestamp
            state["completed_at"] = event.timestamp
            state["current_step"] = None

    def _get_display_name(self, step: str) -> str:
        return STEP_DISPLAY_NAMES.get(step, step)

    def _update_overall_progress(
        self, state: dict, step: str, step_percent: int
    ) -> None:
        if step in STEP_WEIGHTS:
            start, end = STEP_WEIGHTS[step]
            overall = start + int((end - start) * step_percent / 100)
            state["overall_progress"]["percent"] = min(100, overall)

    def _update_overall_progress_on_completion(self, state: dict, step: str) -> None:
        # Per runtime.progress: predict stops at 90 so only apply_physics
        # can trip the auto "status = completed" branch below.
        completion_percent = STEP_COMPLETION_PERCENT

        if step in completion_percent:
            state["overall_progress"]["percent"] = completion_percent[step]

        # current_step counts the user-visible position (1..total_steps).
        # optimize_usd collapses onto slot 1 in STEP_NUMBER, so enabling
        # the full optional pipeline doesn't push the counter past
        # total_steps. Keep it monotonic against the previous value to
        # guard against out-of-order completions.
        state["overall_progress"]["current_step"] = max(
            state["overall_progress"].get("current_step", 0),
            STEP_NUMBER.get(step, len(state["completed_steps"])),
        )

        if state["overall_progress"]["percent"] >= 100 and state.get("status") not in {
            "cancelling",
            "cancelled",
            "failed",
        }:
            state["status"] = "completed"
            state["completed_at"] = datetime.utcnow().isoformat()

    def cleanup_session(self, session_id: str) -> None:
        """Clean up session from event bus."""
        if session_id in self._queues:
            del self._queues[session_id]
        if session_id in self._state:
            del self._state[session_id]


# Global singleton event bus
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
