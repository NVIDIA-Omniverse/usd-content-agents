# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job registry for managing pipeline task lifecycle."""

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

MAX_ACTIVE_SESSIONS_ENV_VAR = "PA_MAX_ACTIVE_SESSIONS"
DEFAULT_MAX_ACTIVE_SESSIONS = 1
_CANCEL_WAIT_TIMEOUT_SECONDS = 5.0


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


# Sentinel placed into JobRegistry._tasks by reserve() before the actual
# asyncio.Task is created. It holds the slot so concurrent reserve() /
# register() / is_running() calls observe the reservation and can't race past
# it. We never await this future from the user side; it is set/cancelled by
# the reservation owner via ``JobReservation.start()`` / ``release()``.
_RESERVED = object()


def _create_job_task(coro: Any) -> asyncio.Task[Any]:
    """Create one registry wrapper task (isolated for failure injection)."""
    return asyncio.create_task(coro)


class JobReservation:
    """Atomic slot reservation returned by ``JobRegistry.reserve()``.

    The reservation holds the ``session_id`` slot in the registry so that:

    * concurrent ``reserve()`` / ``register()`` / ``is_running()`` callers
      see the slot as occupied (and the *losing* concurrent request gets a
      ``ValueError`` immediately, BEFORE it can mutate any session state);
    * the *winning* request can safely write session metadata to the
      backing store, knowing no other in-process request can overwrite it
      out from under us;
    * once the metadata is durable, ``start(coro)`` swaps the reservation
      for the real running task. If anything fails between ``reserve()``
      and ``start()``, the caller must call ``release()`` to free the slot.

    Use as a context manager to guarantee release on early-exit paths::

        async with await registry.reserve(session_id) as reservation:
            await manager.update_session(...)
            await reservation.start(executor_coro())
    """

    def __init__(self, registry: "JobRegistry", session_id: str) -> None:
        self._registry = registry
        self._session_id = session_id
        self._consumed = False

    @property
    def session_id(self) -> str:
        return self._session_id

    async def start(self, coro: Any) -> None:
        """Swap the reservation for a real running task.

        Must be called at most once per reservation. After ``start()``,
        the reservation is considered consumed and ``release()`` becomes a
        no-op. If cancellation or another ``BaseException`` interrupts startup
        before a task owns ``coro``, the unstarted coroutine is closed and the
        reservation remains available for the caller to release.
        """
        if self._consumed:
            # Mirror the lost-reservation branch below: close the incoming
            # coroutine before raising so a misuse doesn't surface as a
            # noisy "coroutine was never awaited" RuntimeWarning at GC.
            if hasattr(coro, "close"):
                coro.close()
            raise RuntimeError(f"Reservation for {self._session_id} already consumed")

        task_owns_coro = False
        lock_acquired = False
        try:
            await self._registry._lock.acquire()
            lock_acquired = True
            current = self._registry._tasks.get(self._session_id)
            if current is not _RESERVED:
                # Defensive: somebody (cancel/release) cleared our slot
                # under us. Refuse to start so we don't leak a coroutine
                # into an unowned slot.
                raise RuntimeError(
                    f"Reservation for {self._session_id} was lost before start()"
                )
            wrapper = self._registry._run_with_cleanup(self._session_id, coro)
            try:
                task = _create_job_task(wrapper)
            except BaseException:
                wrapper.close()
                raise
            task_owns_coro = True
            # Stash the inner coroutine on the task so cancel() can close
            # it if the wrapping task is cancelled before _run_with_cleanup
            # ever runs (which would otherwise leak the coroutine and
            # surface a "coroutine was never awaited" RuntimeWarning).
            task._wu_inner_coro = coro  # type: ignore[attr-defined]
            self._registry._tasks[self._session_id] = task
        except BaseException:
            if not task_owns_coro and hasattr(coro, "close"):
                coro.close()
            raise
        finally:
            if lock_acquired:
                self._registry._lock.release()
        self._consumed = True
        logger.info(f"Pipeline queued for {self._session_id[:8]}...")

    async def release(self) -> None:
        """Free the reserved slot if ``start()`` was never called."""
        if self._consumed:
            return
        async with self._registry._lock:
            current = self._registry._tasks.get(self._session_id)
            if current is _RESERVED:
                del self._registry._tasks[self._session_id]
        self._consumed = True

    async def __aenter__(self) -> "JobReservation":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Always release on context exit if start() was never called. This
        # is what guarantees the BLOCKING fix: any exception between
        # reserve() and start() (including HTTPException raised by the
        # route) drops the reservation cleanly without holding the slot.
        await self.release()


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
        self._tasks: dict[str, Any] = {}
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._lock = asyncio.Lock()

    async def reserve(self, session_id: str) -> JobReservation:
        """Atomically claim the slot for ``session_id``.

        Returns a :class:`JobReservation` whose ``start()`` method must be
        called once the caller has finished writing any session state that
        depends on "we are about to launch a job for this session". Until
        ``start()`` is called, the slot is held by a sentinel and any
        concurrent ``reserve()`` / ``register()`` for the same session_id
        will raise ``ValueError`` immediately — that is what stops a losing
        concurrent rerun from overwriting the winning request's metadata.

        Raises:
            ValueError: when ``session_id`` already has a live task or an
                outstanding reservation on this instance. Callers should
                map this to HTTP 409.
        """
        async with self._lock:
            existing = self._tasks.get(session_id)
            if existing is _RESERVED:
                raise ValueError(
                    f"Session {session_id} already has an active job on this instance"
                )
            if isinstance(existing, asyncio.Task) and not existing.done():
                raise ValueError(
                    f"Session {session_id} already has an active job on this instance"
                )
            self._tasks[session_id] = _RESERVED
        return JobReservation(self, session_id)

    async def register(self, session_id: str, coro: Any) -> None:
        """Register and start a pipeline job.

        Convenience wrapper around :meth:`reserve` + :meth:`JobReservation.start`
        for callers that have no session state to write between the
        reservation and the launch (e.g. fresh-session paths). Routes that
        do mutate session state on rerun MUST call :meth:`reserve` first,
        write state inside the reservation, then call ``start()`` — that
        is the ordering that makes the same-pod rerun race safe.

        The job is created immediately (so the HTTP 202 returns right away)
        but waits on the semaphore internally before executing. This means
        concurrent requests are queued rather than blocking the HTTP handler
        or being rejected.

        Atomically rejects a register call when ``session_id`` is already
        active in this registry: silently overwriting a live task would
        leak the prior task into the asyncio event loop and make
        ``cancel()`` and ``is_running()`` target only the most recent
        registration.

        Raises:
            ValueError: when ``session_id`` already has a live task on
                this instance. Callers should map this to HTTP 409. The
                incoming ``coro`` is closed before the exception propagates
                so callers don't get a "coroutine was never awaited"
                RuntimeWarning on the 409 path.
        """
        try:
            reservation = await self.reserve(session_id)
        except ValueError:
            if hasattr(coro, "close"):
                coro.close()
            raise
        try:
            await reservation.start(coro)
        except BaseException:
            await reservation.release()
            raise

    async def _run_with_cleanup(self, session_id: str, coro: Any) -> None:
        """Wait for a semaphore slot, run the pipeline, then clean up.

        Args:
            session_id: Session identifier
            coro: Coroutine to execute
        """
        acquired = False
        active_incremented = False
        coro_started = False
        try:
            await self._semaphore.acquire()
            acquired = True

            async with self._lock:
                self._active_count += 1
                active_incremented = True

            logger.info(
                f"Starting pipeline for {session_id[:8]}... "
                f"(active: {self._active_count}/{self._semaphore._value + self._active_count})"
            )

            coro_started = True
            await coro
        finally:
            if not coro_started and hasattr(coro, "close"):
                coro.close()

            if acquired:
                self._semaphore.release()

            if active_incremented:
                async with self._lock:
                    self._active_count -= 1

            current_task = asyncio.current_task()
            async with self._lock:
                # Only clear our own slot — defend against a future reserve()
                # for the same session_id having already swapped a new entry
                # in while we were unwinding.
                if self._tasks.get(session_id) is current_task:
                    del self._tasks[session_id]

            logger.info(
                f"Pipeline completed/cancelled for {session_id[:8]}... "
                f"(active: {self._active_count})"
            )

    async def cancel(self, session_id: str) -> bool:
        """Cancel a running pipeline job.

        Args:
            session_id: Session identifier

        Returns:
            True if job was cancelled, False if not found or already done
        """
        task = self._tasks.get(session_id)
        if task is None or task is _RESERVED:
            # _RESERVED means a route is mid-handshake (reserve() returned
            # but start() hasn't run yet); there is no asyncio.Task to
            # cancel. Treat this the same as "no task" — the route owning
            # the reservation will release it on its own error path.
            logger.warning(f"Cannot cancel - session not found: {session_id[:8]}...")
            return False

        if task.done():
            logger.info(f"Session already completed: {session_id[:8]}...")
            return False

        task.cancel()
        logger.info(f"Cancellation requested for {session_id[:8]}...")

        # Do not use wait_for here: it sends a second cancellation at timeout
        # and then waits indefinitely for a cancellation-suppressing task. A
        # thread-backed pipeline deliberately keeps its wrapper alive while
        # the worker drains, so return after the bounded grace period and let
        # the registry slot remain occupied until normal cleanup runs.
        await asyncio.wait({task}, timeout=_CANCEL_WAIT_TIMEOUT_SECONDS)

        # If the task was cancelled before _run_with_cleanup ever started,
        # the inner coroutine never reached its `await` point and Python
        # would emit "coroutine was never awaited" on GC. Close it
        # explicitly here. close() is safe on already-completed coros.
        inner_coro = getattr(task, "_wu_inner_coro", None)
        if task.done() and inner_coro is not None and hasattr(inner_coro, "close"):
            inner_coro.close()

        # A task cancelled before ``_run_with_cleanup`` gets its first event
        # loop turn never executes that coroutine's ``finally`` block. Remove
        # the completed wrapper here as the cancellation-side fallback so an
        # injected post-registration startup failure cannot leave a dead slot.
        async with self._lock:
            if self._tasks.get(session_id) is task and task.done():
                del self._tasks[session_id]

        return True

    async def wait_for_quiescence(self, session_id: str) -> None:
        """Wait until a registered job can no longer mutate session artifacts."""
        task = self.get_task(session_id)
        if task is None:
            return
        # asyncio.wait does not re-raise the target task's cancellation or
        # failure, but cancellation of this destructive caller still propagates.
        await asyncio.wait({task})

    def get_task(self, session_id: str) -> asyncio.Task | None:
        """Get task for a session."""
        entry = self._tasks.get(session_id)
        if entry is None or entry is _RESERVED:
            return None
        return entry  # type: ignore[no-any-return]

    def is_running(self, session_id: str) -> bool:
        """Check if a session is currently running.

        A session is "running" from the point ``reserve()`` claims its
        slot all the way through to task completion — including the brief
        window where the slot holds a ``_RESERVED`` sentinel. This is what
        makes the up-front 409 check in routes safe against same-pod rerun
        races: a concurrent rerun that lost the reservation will observe
        ``is_running == True`` even before the winner has written
        ``status=pending`` to the session store.
        """
        entry = self._tasks.get(session_id)
        if entry is None:
            return False
        if entry is _RESERVED:
            return True
        return not entry.done()

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
