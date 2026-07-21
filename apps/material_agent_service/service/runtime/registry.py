# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job registry for managing pipeline task lifecycle."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)

MAX_ACTIVE_SESSIONS_ENV_VAR = "MA_MAX_ACTIVE_SESSIONS"
DEFAULT_MAX_ACTIVE_SESSIONS = 3


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


BeforeJobStart = Callable[[], Awaitable[None]]


class DuplicateJobError(RuntimeError):
    """Raised when a session already has a running or reserved job."""


def _close_awaitable(awaitable: Any) -> None:
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()


class JobRegistry:
    """Registry for managing asyncio.Task lifecycle.

    Maintains strong references to pipeline tasks to prevent GC and enables
    proper cancellation and monitoring.

    This replaces FastAPI's BackgroundTasks which doesn't provide task handles
    or real cancellation support.
    """

    def __init__(self, max_concurrent: int):
        """Initialize job registry.

        Args:
            max_concurrent: Maximum concurrent pipeline jobs (semaphore limit)
        """
        # Strong references to tasks (prevents GC mid-execution)
        self._tasks: dict[str, asyncio.Task] = {}
        self._reservations: dict[str, asyncio.Task] = {}

        # Configured concurrency limit
        self._max_concurrent = max_concurrent

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Track active count explicitly (don't use _semaphore._value)
        self._active_count = 0
        self._lock = asyncio.Lock()

    async def register(
        self,
        session_id: str,
        coro: Any,
        *,
        before_start: BeforeJobStart | None = None,
    ) -> None:
        """Register and start a pipeline job.

        Args:
            session_id: Session identifier
            coro: Coroutine to execute
        """
        registering_task = asyncio.current_task()
        if registering_task is None:  # pragma: no cover - asyncio always owns callers
            _close_awaitable(coro)
            raise RuntimeError("Job registration requires an active asyncio task")

        async with self._lock:
            existing = self._tasks.get(session_id)
            if session_id in self._reservations or (
                existing is not None and not existing.done()
            ):
                _close_awaitable(coro)
                raise DuplicateJobError(
                    f"Session already has a registered job: {session_id}"
                )
            self._reservations[session_id] = registering_task

        semaphore_acquired = False
        task: asyncio.Task | None = None
        wrapper_started = asyncio.Event()
        ready = asyncio.Event()
        try:
            await self._semaphore.acquire()
            semaphore_acquired = True
            if before_start is not None:
                await before_start()

            task = asyncio.create_task(
                self._run_with_cleanup(
                    session_id,
                    coro,
                    wrapper_started=wrapper_started,
                    ready=ready,
                )
            )
            task.add_done_callback(lambda _task: ready.set())
            self._tasks[session_id] = task
            self._reservations.pop(session_id, None)
            self._active_count += 1

            # Do not report acceptance until the wrapper has had a first turn.
            # If it is cancelled before entry, the done callback wakes this
            # waiter and the registering caller owns deterministic cleanup.
            await ready.wait()
            if not wrapper_started.is_set():
                await self._cleanup_unstarted_task(session_id, task, coro)
                semaphore_acquired = False
                raise asyncio.CancelledError(
                    f"Job was cancelled before start: {session_id}"
                )
        except BaseException:
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            if task is not None and not wrapper_started.is_set() and semaphore_acquired:
                await self._cleanup_unstarted_task(session_id, task, coro)
                semaphore_acquired = False
            elif task is None:
                _close_awaitable(coro)
            if semaphore_acquired and task is None:
                self._semaphore.release()
            async with self._lock:
                if self._reservations.get(session_id) is registering_task:
                    self._reservations.pop(session_id, None)
            raise

        logger.info(
            f"Starting pipeline for {session_id[:8]}... "
            f"(active: {self._active_count}/{self._semaphore._value + self._active_count})"
        )

    async def _run_with_cleanup(
        self,
        session_id: str,
        coro: Any,
        *,
        wrapper_started: asyncio.Event,
        ready: asyncio.Event,
    ) -> None:
        """Run coroutine and clean up resources.

        Args:
            session_id: Session identifier
            coro: Coroutine to execute
        """
        wrapper_started.set()
        ready.set()
        try:
            await coro
        finally:
            # Always release semaphore and decrement count for tracked jobs.
            self._semaphore.release()

            async with self._lock:
                self._active_count -= 1
                current = self._tasks.get(session_id)
                if current is asyncio.current_task():
                    del self._tasks[session_id]

            logger.info(
                f"Pipeline completed/cancelled for {session_id[:8]}... "
                f"(active: {self._active_count})"
            )

    async def _cleanup_unstarted_task(
        self,
        session_id: str,
        task: asyncio.Task,
        coro: Any,
    ) -> None:
        """Release resources when cancellation precedes wrapper entry."""
        _close_awaitable(coro)
        self._semaphore.release()
        async with self._lock:
            if self._tasks.get(session_id) is task:
                self._tasks.pop(session_id, None)
                self._active_count -= 1

    async def cancel(self, session_id: str) -> bool:
        """Cancel a running pipeline job.

        Args:
            session_id: Session identifier

        Returns:
            True if job was cancelled, False if not found or already done
        """
        async with self._lock:
            task = self._tasks.get(session_id)
            reservation = self._reservations.get(session_id)

        if task is None and reservation is None:
            logger.warning(f"Cannot cancel - session not found: {session_id[:8]}...")
            return False

        if task is not None and task.done():
            logger.info(f"Session already completed: {session_id[:8]}...")
            return False

        # A reservation waiting for a semaphore or running ``before_start`` is
        # owned by the registering request task. Cancelling that owner closes
        # the not-yet-started worker coroutine through ``register`` cleanup.
        target = task if task is not None else reservation
        assert target is not None
        target.cancel()
        logger.info(f"Cancellation requested for {session_id[:8]}...")

        try:
            # Wait for cancellation to complete
            await asyncio.wait_for(target, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            # Expected - task was cancelled or didn't finish in time
            pass

        return True

    def get_task(self, session_id: str) -> asyncio.Task | None:
        """Get task for a session.

        Args:
            session_id: Session identifier

        Returns:
            Task or None if not found
        """
        return self._tasks.get(session_id)

    def is_running(self, session_id: str) -> bool:
        """Check if a session is currently running.

        Args:
            session_id: Session identifier

        Returns:
            True if session has an active task
        """
        task = self._tasks.get(session_id)
        return session_id in self._reservations or (
            task is not None and not task.done()
        )

    def is_reserved(self, session_id: str) -> bool:
        """Return whether registration is waiting to create the worker task."""
        return session_id in self._reservations

    @property
    def active_count(self) -> int:
        """Get count of currently active jobs.

        Returns:
            Number of active jobs
        """
        return self._active_count

    @property
    def max_concurrent(self) -> int:
        """Get the configured maximum concurrent jobs.

        Returns:
            Maximum concurrent pipeline jobs
        """
        return self._max_concurrent


# Global singleton job registry
_job_registry: JobRegistry | None = None


def get_job_registry() -> JobRegistry:
    """Get the global job registry instance.

    Returns:
        Global JobRegistry instance
    """
    global _job_registry
    if _job_registry is None:
        _job_registry = JobRegistry(max_concurrent=resolve_max_active_sessions())
    return _job_registry
