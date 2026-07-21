# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job registry for managing pipeline task lifecycle."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

MAX_ACTIVE_SESSIONS_ENV_VAR = "JA_MAX_ACTIVE_SESSIONS"
DEFAULT_MAX_ACTIVE_SESSIONS = 1


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


_CANCEL_DRAIN_WAIT_SECONDS = 5.0


class JobRegistry:
    """Registry for managing asyncio.Task lifecycle.

    Maintains strong references to pipeline tasks to prevent GC and enables
    proper cancellation and monitoring.
    """

    def __init__(self, max_concurrent: int):
        """Initialize job registry.

        Args:
            max_concurrent: Maximum concurrent pipeline jobs (semaphore limit)
        """
        self._tasks: dict[str, asyncio.Task] = {}
        self._run_ids: dict[str, str | None] = {}
        self._admissions: dict[str, str] = {}
        self._cleanup_events: dict[str, asyncio.Event] = {}
        self._background_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._lock = asyncio.Lock()

    async def reserve_admission(self, session_id: str, run_id: str) -> bool:
        """Reserve same-process admission until the run becomes a registered task."""

        async with self._lock:
            existing = self._tasks.get(session_id)
            if existing is not None and not existing.done():
                return False
            cleanup_complete = self._cleanup_events.get(session_id)
            if cleanup_complete is not None and not cleanup_complete.is_set():
                return False
            if session_id in self._admissions:
                return False
            self._admissions[session_id] = run_id
            return True

    async def release_admission(self, session_id: str, run_id: str) -> bool:
        """Release only the matching same-process admission generation."""

        async with self._lock:
            if self._admissions.get(session_id) != run_id:
                return False
            del self._admissions[session_id]
            return True

    async def register(
        self,
        session_id: str,
        coro: Any,
        *,
        run_id: str | None = None,
        liveness_guard: Awaitable[None] | None = None,
        on_finish: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Register and start a pipeline job.

        The job is created immediately (so the HTTP 202 returns right away)
        but waits on the semaphore internally before executing.  This means
        concurrent requests are queued rather than blocking the HTTP handler
        or being rejected.

        Args:
            session_id: Session identifier
            coro: Coroutine to execute
            run_id: Accepted run generation, when the caller has one
            liveness_guard: Coroutine that must remain alive for the full queued
                and active lifecycle. Its completion or failure cancels the job.
        """
        lifecycle = {"entered": False}
        cleanup_complete = asyncio.Event()
        rejection_message: str | None = None
        async with self._lock:
            admission_run_id = self._admissions.get(session_id)
            if admission_run_id is not None and admission_run_id != run_id:
                rejection_message = (
                    f"Pipeline admission belongs to another run for {session_id}"
                )
            else:
                existing = self._tasks.get(session_id)
                if existing is not None and not existing.done():
                    rejection_message = f"Pipeline already registered for {session_id}"

            if rejection_message is None:
                task = asyncio.create_task(
                    self._run_with_cleanup(
                        session_id,
                        coro,
                        liveness_guard,
                        on_finish,
                        lifecycle,
                        cleanup_complete,
                    )
                )
                self._tasks[session_id] = task
                self._run_ids[session_id] = run_id
                self._cleanup_events[session_id] = cleanup_complete
                self._admissions.pop(session_id, None)
                task.add_done_callback(
                    lambda completed: self._close_task_that_never_started(
                        session_id,
                        completed,
                        coro,
                        liveness_guard,
                        on_finish,
                        lifecycle,
                        cleanup_complete,
                    )
                )

        if rejection_message is not None:
            await asyncio.gather(
                self._cancel_and_drain_awaitable(
                    coro,
                    timeout=_CANCEL_DRAIN_WAIT_SECONDS,
                ),
                self._cancel_and_drain_awaitable(
                    liveness_guard,
                    timeout=_CANCEL_DRAIN_WAIT_SECONDS,
                ),
            )
            raise RuntimeError(rejection_message)

        logger.info(f"Pipeline queued for {session_id[:8]}...")

    async def _run_with_cleanup(
        self,
        session_id: str,
        coro: Any,
        liveness_guard: Awaitable[None] | None,
        on_finish: Callable[[], Awaitable[None]] | None,
        lifecycle: dict[str, bool],
        cleanup_complete: asyncio.Event,
    ) -> None:
        """Wait for a semaphore slot, run the pipeline, then clean up.

        Args:
            session_id: Session identifier
            coro: Coroutine to execute
        """
        lifecycle["entered"] = True
        acquired = False
        counted_active = False
        coro_started = False
        guard_task: asyncio.Future[None] | None = None
        try:
            if liveness_guard is not None:
                guard_task = asyncio.ensure_future(liveness_guard)

            async def acquire_slot() -> None:
                nonlocal acquired
                await self._semaphore.acquire()
                # Record ownership inside the acquisition task so a same-tick
                # guard failure cannot leak the permit before this await returns.
                acquired = True

            await self._await_guarded(acquire_slot(), guard_task)

            async with self._lock:
                self._active_count += 1
                counted_active = True

            logger.info(
                f"Starting pipeline for {session_id[:8]}... "
                f"(active: {self._active_count}/{self._semaphore._value + self._active_count})"
            )

            coro_started = True
            await self._await_guarded(coro, guard_task)
        finally:
            owner_task = asyncio.current_task()
            assert owner_task is not None
            await self._await_cleanup_transaction(
                self._finish_registered_run(
                    session_id,
                    owner_task,
                    coro,
                    coro_started,
                    guard_task,
                    acquired,
                    counted_active,
                    on_finish,
                    cleanup_complete,
                )
            )

    async def _finish_registered_run(
        self,
        session_id: str,
        owner_task: asyncio.Task[Any],
        coro: Any,
        coro_started: bool,
        guard_task: asyncio.Future[None] | None,
        acquired: bool,
        counted_active: bool,
        on_finish: Callable[[], Awaitable[None]] | None,
        cleanup_complete: asyncio.Event,
    ) -> None:
        """Drain one run's complete cleanup transaction outside its task."""

        try:
            if not coro_started:
                self._close_awaitable(coro)

            if guard_task is not None:
                if not guard_task.done():
                    guard_task.cancel()
                await asyncio.gather(guard_task, return_exceptions=True)

            if acquired:
                self._semaphore.release()

            if counted_active:
                async with self._lock:
                    self._active_count -= 1

            async with self._lock:
                if self._tasks.get(session_id) is owner_task:
                    del self._tasks[session_id]
                    self._run_ids.pop(session_id, None)

            if on_finish is not None:
                try:
                    await on_finish()
                except asyncio.CancelledError:
                    logger.debug(
                        "Pipeline cleanup was cancelled after draining for %s...",
                        session_id[:8],
                    )
                except Exception:
                    logger.exception(
                        "Pipeline cleanup failed for %s...", session_id[:8]
                    )
        finally:
            async with self._lock:
                if self._cleanup_events.get(session_id) is cleanup_complete:
                    del self._cleanup_events[session_id]
            cleanup_complete.set()

            logger.info(
                f"Pipeline completed/cancelled for {session_id[:8]}... "
                f"(active: {self._active_count})"
            )

    @staticmethod
    async def _await_cleanup_transaction(awaitable: Awaitable[None]) -> None:
        """Drain cleanup despite repeated cancellation of the owner task."""

        task = asyncio.ensure_future(awaitable)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                raise
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if not task.cancelled():
                try:
                    task.result()
                except BaseException:
                    pass
            raise

    @staticmethod
    async def _await_guarded(
        awaitable: Awaitable[Any],
        guard_task: asyncio.Future[None] | None,
    ) -> Any:
        if guard_task is None:
            return await awaitable

        operation_task = asyncio.ensure_future(awaitable)
        try:
            done, _ = await asyncio.wait(
                {operation_task, guard_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if guard_task in done:
                guard_task.result()
                raise RuntimeError("Job liveness guard stopped unexpectedly")
            return operation_task.result()
        finally:
            if not operation_task.done():
                operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)

    @staticmethod
    def _close_awaitable(awaitable: Any | None) -> None:
        if awaitable is not None and hasattr(awaitable, "close"):
            awaitable.close()

    @classmethod
    async def _cancel_and_drain_awaitable(
        cls,
        awaitable: Any | None,
        *,
        timeout: float | None = None,
    ) -> None:
        if awaitable is None:
            return
        has_close = hasattr(awaitable, "close")
        cls._close_awaitable(awaitable)
        task = asyncio.ensure_future(awaitable)
        if not has_close and not task.done():
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if task in done:
            await asyncio.gather(task, return_exceptions=True)
        else:
            task.add_done_callback(cls._retrieve_task_result)

    @staticmethod
    def _retrieve_task_result(task: asyncio.Future[Any]) -> None:
        try:
            task.result()
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                logger.error(
                    "Asynchronous pipeline cleanup failed after its caller stopped "
                    "waiting",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

    def _close_task_that_never_started(
        self,
        session_id: str,
        task: asyncio.Task,
        coro: Any,
        liveness_guard: Awaitable[None] | None,
        on_finish: Callable[[], Awaitable[None]] | None,
        lifecycle: dict[str, bool],
        cleanup_complete: asyncio.Event,
    ) -> None:
        """Close and release a job cancelled before its wrapper first ran."""

        if lifecycle["entered"]:
            return
        cleanup_task = asyncio.create_task(
            self._cleanup_task_that_never_started(
                session_id,
                task,
                coro,
                liveness_guard,
                on_finish,
                cleanup_complete,
            )
        )
        self._background_cleanup_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._background_cleanup_tasks.discard)
        cleanup_task.add_done_callback(self._retrieve_task_result)

    async def _cleanup_task_that_never_started(
        self,
        session_id: str,
        task: asyncio.Task,
        coro: Any,
        liveness_guard: Awaitable[None] | None,
        on_finish: Callable[[], Awaitable[None]] | None,
        cleanup_complete: asyncio.Event,
    ) -> None:
        try:
            try:
                await asyncio.gather(
                    self._cancel_and_drain_awaitable(coro),
                    self._cancel_and_drain_awaitable(liveness_guard),
                )
            finally:
                async with self._lock:
                    if self._tasks.get(session_id) is task:
                        del self._tasks[session_id]
                        self._run_ids.pop(session_id, None)
            if on_finish is not None:
                try:
                    await on_finish()
                except asyncio.CancelledError:
                    logger.debug(
                        "Pipeline cleanup was cancelled after draining for %s...",
                        session_id[:8],
                    )
                except Exception:
                    logger.exception(
                        "Pipeline cleanup failed for %s...", session_id[:8]
                    )
        finally:
            async with self._lock:
                if self._cleanup_events.get(session_id) is cleanup_complete:
                    del self._cleanup_events[session_id]
            cleanup_complete.set()

    async def cancel(self, session_id: str, *, run_id: str | None = None) -> bool:
        """Cancel a running pipeline job.

        Args:
            session_id: Session identifier
            run_id: Optional accepted run generation. When supplied, a successor
                task for the same session is never cancelled.

        Returns:
            True if job was cancelled, False if not found or already done
        """
        async with self._lock:
            task = self._tasks.get(session_id)
            cleanup_complete = self._cleanup_events.get(session_id)
            if task is None:
                logger.warning(
                    f"Cannot cancel - session not found: {session_id[:8]}..."
                )
                return False

            registered_run_id = self._run_ids.get(session_id)
            if run_id is not None and registered_run_id != run_id:
                logger.info(
                    "Cancellation ignored for stale session/run: %s.../%s",
                    session_id[:8],
                    run_id,
                )
                return False

            if task.done():
                logger.info(f"Session already completed: {session_id[:8]}...")
                return False

            task.cancel()
        logger.info(f"Cancellation requested for {session_id[:8]}...")

        done, _ = await asyncio.wait(
            {task},
            timeout=_CANCEL_DRAIN_WAIT_SECONDS,
        )
        if task in done:
            await asyncio.gather(task, return_exceptions=True)
            if cleanup_complete is not None and not cleanup_complete.is_set():
                cleanup_wait = asyncio.create_task(cleanup_complete.wait())
                done, _ = await asyncio.wait(
                    {cleanup_wait},
                    timeout=_CANCEL_DRAIN_WAIT_SECONDS,
                )
                if cleanup_wait in done:
                    await cleanup_wait
                else:
                    cleanup_wait.cancel()
                    await asyncio.gather(cleanup_wait, return_exceptions=True)

        return True

    def get_task(self, session_id: str) -> asyncio.Task | None:
        """Get task for a session."""
        return self._tasks.get(session_id)

    def is_running(self, session_id: str) -> bool:
        """Check if a session is currently running."""
        task = self._tasks.get(session_id)
        cleanup_complete = self._cleanup_events.get(session_id)
        return (
            session_id in self._admissions
            or (task is not None and not task.done())
            or (cleanup_complete is not None and not cleanup_complete.is_set())
        )

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
