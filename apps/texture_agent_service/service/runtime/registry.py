# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job registry for managing pipeline task lifecycle."""

import asyncio
import inspect
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

MAX_ACTIVE_SESSIONS_ENV_VAR = "TA_MAX_ACTIVE_SESSIONS"
DEFAULT_MAX_ACTIVE_SESSIONS = 4


def resolve_max_active_sessions() -> int:
    """Resolve the configured session capacity with a safe fallback.

    Zero is an explicit valid capacity. Unset, non-integer, and negative values
    use :data:`DEFAULT_MAX_ACTIVE_SESSIONS`.
    """
    env_value = os.getenv(MAX_ACTIVE_SESSIONS_ENV_VAR)
    if env_value is None:
        return DEFAULT_MAX_ACTIVE_SESSIONS

    try:
        limit = int(env_value)
    except ValueError:
        logger.error(
            "%s must be a valid integer, got '%s'. Falling back to default: %d",
            MAX_ACTIVE_SESSIONS_ENV_VAR,
            env_value,
            DEFAULT_MAX_ACTIVE_SESSIONS,
        )
        return DEFAULT_MAX_ACTIVE_SESSIONS

    if limit < 0:
        logger.error(
            "%s must be non-negative, got '%s'. Falling back to default: %d",
            MAX_ACTIVE_SESSIONS_ENV_VAR,
            env_value,
            DEFAULT_MAX_ACTIVE_SESSIONS,
        )
        return DEFAULT_MAX_ACTIVE_SESSIONS
    return limit


class JobRegistry:
    """Registry for managing asyncio.Task lifecycle.

    Maintains strong references to pipeline tasks to prevent GC and enables
    proper cancellation and monitoring.
    """

    def __init__(
        self,
        max_concurrent: int,
        cancel_wait_seconds: float = 5.0,
    ):
        """Initialize job registry.

        Args:
            max_concurrent: Maximum concurrent pipeline jobs (semaphore limit)
            cancel_wait_seconds: Seconds to wait for cancellation cleanup before
                returning to the caller. The task remains registered if cleanup
                is still draining.
        """
        self._tasks: dict[str, asyncio.Task] = {}
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._lock = asyncio.Lock()
        self._cancel_wait_seconds = cancel_wait_seconds

    async def register(
        self,
        session_id: str,
        coro: Any,
        on_never_started: Callable[[], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        on_queued_heartbeat: Callable[[], Any] | None = None,
        queued_heartbeat_interval_seconds: float = 60.0,
    ) -> None:
        """Register and start a pipeline job.

        The job is created immediately (so the HTTP 202 returns right away)
        but waits on the semaphore internally before executing.

        Args:
            session_id: Session identifier
            coro: Coroutine to execute
            on_never_started: Cleanup called if the queued coroutine is
                cancelled/closed before it starts.
            on_finished: Cleanup called after the queued/running job is fully
                removed from the registry.
            on_queued_heartbeat: Optional heartbeat while the job is accepted
                but still waiting for a semaphore slot.
            queued_heartbeat_interval_seconds: Interval for queued heartbeats.

        Raises:
            RuntimeError: If session is already running.
        """
        existing = self._tasks.get(session_id)
        if existing is not None and not existing.done():
            if hasattr(coro, "close"):
                coro.close()
            raise RuntimeError(
                f"Session {session_id} is already running. "
                "Cancel it before re-registering."
            )

        cleanup_complete = False
        wrapper_started = False

        async def _runner() -> None:
            nonlocal cleanup_complete, wrapper_started
            wrapper_started = True
            try:
                await self._run_with_cleanup(
                    session_id,
                    coro,
                    on_never_started=on_never_started,
                    on_finished=on_finished,
                    on_queued_heartbeat=on_queued_heartbeat,
                    queued_heartbeat_interval_seconds=queued_heartbeat_interval_seconds,
                )
            finally:
                cleanup_complete = True

        def _cleanup_if_never_started(task: asyncio.Task) -> None:
            nonlocal cleanup_complete
            if cleanup_complete or wrapper_started:
                return

            cleanup_complete = True
            if hasattr(coro, "close"):
                coro.close()

            if on_never_started is not None:
                try:
                    on_never_started()
                except Exception:
                    logger.exception(
                        "Pre-start job cleanup failed for %s", session_id[:8]
                    )

            if self._tasks.get(session_id) is task:
                del self._tasks[session_id]

            if on_finished is not None:
                try:
                    on_finished()
                except Exception:
                    logger.exception("Job cleanup failed for %s", session_id[:8])

            logger.info(
                "Pipeline cancelled before startup for %s... (active: %s)",
                session_id[:8],
                self._active_count,
            )

        task = asyncio.create_task(_runner())
        task.add_done_callback(_cleanup_if_never_started)
        self._tasks[session_id] = task

        logger.info(f"Pipeline queued for {session_id[:8]}...")

    async def _run_with_cleanup(
        self,
        session_id: str,
        coro: Any,
        on_never_started: Callable[[], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        on_queued_heartbeat: Callable[[], Any] | None = None,
        queued_heartbeat_interval_seconds: float = 60.0,
    ) -> None:
        """Wait for a semaphore slot, run the pipeline, then clean up."""
        acquired = False
        coro_started = False
        try:
            await self._acquire_slot_with_queued_heartbeat(
                session_id,
                on_queued_heartbeat,
                queued_heartbeat_interval_seconds,
            )
            acquired = True

            async with self._lock:
                self._active_count += 1

            logger.info(
                f"Starting pipeline for {session_id[:8]}... "
                f"(active: {self._active_count}/{self._semaphore._value + self._active_count})"
            )

            coro_started = True
            await coro
        finally:
            if not coro_started and hasattr(coro, "close"):
                coro.close()
                if on_never_started is not None:
                    try:
                        on_never_started()
                    except Exception:
                        logger.exception(
                            "Queued-job cleanup failed for %s", session_id[:8]
                        )

            if acquired:
                self._semaphore.release()

                async with self._lock:
                    self._active_count -= 1

            if session_id in self._tasks:
                del self._tasks[session_id]

            if on_finished is not None:
                try:
                    on_finished()
                except Exception:
                    logger.exception("Job cleanup failed for %s", session_id[:8])

            logger.info(
                f"Pipeline completed/cancelled for {session_id[:8]}... "
                f"(active: {self._active_count})"
            )

    async def _acquire_slot_with_queued_heartbeat(
        self,
        session_id: str,
        on_queued_heartbeat: Callable[[], Any] | None,
        queued_heartbeat_interval_seconds: float,
    ) -> None:
        if on_queued_heartbeat is None:
            await self._semaphore.acquire()
            return

        interval = max(0.01, queued_heartbeat_interval_seconds)
        while True:
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=interval)
                return
            except TimeoutError:
                await self._run_queued_heartbeat(session_id, on_queued_heartbeat)

    async def _run_queued_heartbeat(
        self,
        session_id: str,
        on_queued_heartbeat: Callable[[], Any],
    ) -> None:
        try:
            result = on_queued_heartbeat()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Queued-job heartbeat failed for %s", session_id[:8])

    async def cancel(self, session_id: str) -> bool:
        """Cancel a running pipeline job.

        Args:
            session_id: Session identifier

        Returns:
            True if job was cancelled, False if not found or already done
        """
        task = self._tasks.get(session_id)
        if task is None:
            logger.warning(f"Cannot cancel - session not found: {session_id[:8]}...")
            return False

        if task.done():
            logger.info(f"Session already completed: {session_id[:8]}...")
            return False

        task.cancel()
        logger.info(f"Cancellation requested for {session_id[:8]}...")

        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._cancel_wait_seconds,
            )
        except TimeoutError:
            pass
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "Session failed while handling cancellation: %s", session_id
            )

        return True

    def get_task(self, session_id: str) -> asyncio.Task | None:
        """Get task for a session."""
        return self._tasks.get(session_id)

    def is_running(self, session_id: str) -> bool:
        """Check if a session is currently running."""
        task = self._tasks.get(session_id)
        return task is not None and not task.done()

    @property
    def active_count(self) -> int:
        """Get count of currently active jobs (past the semaphore)."""
        return self._active_count

    @property
    def registered_count(self) -> int:
        """Get count of all registered jobs (active + queued)."""
        return len(self._tasks)

    @property
    def max_concurrent(self) -> int:
        """Return the session capacity enforced by this registry."""
        return self._max_concurrent


# Global singleton job registry
_job_registry: JobRegistry | None = None


def get_job_registry() -> JobRegistry:
    """Get the global job registry instance."""
    global _job_registry
    if _job_registry is None:
        _job_registry = JobRegistry(max_concurrent=resolve_max_active_sessions())
    return _job_registry
