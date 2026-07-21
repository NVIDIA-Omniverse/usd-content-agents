# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Callable

import pytest

from ...service.runtime import registry as registry_module
from ...service.runtime.registry import JobRegistry


class _TaskBackedGuard:
    def __init__(self, task: asyncio.Task[None]) -> None:
        self.task = task

    def __await__(self):
        return self.task.__await__()

    def close(self) -> None:
        self.task.cancel()


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not met before timeout")
        await asyncio.sleep(0.01)


async def test_cancel_running_job_releases_slot_for_queued_job() -> None:
    registry = JobRegistry(max_concurrent=1)

    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    first_release = asyncio.Event()
    queued_started = asyncio.Event()

    async def first_job() -> None:
        first_started.set()
        try:
            await first_release.wait()
        except asyncio.CancelledError:
            first_cancelled.set()
            raise

    async def queued_job() -> None:
        queued_started.set()

    await registry.register("first", first_job())
    await _wait_until(first_started.is_set)

    await registry.register("queued", queued_job())
    await _wait_until(lambda: registry.registered_count == 2)
    await asyncio.sleep(0.05)

    assert registry.active_count == 1
    assert registry.is_running("first")
    assert registry.is_running("queued")
    assert not queued_started.is_set()

    assert await registry.cancel("first") is True
    await _wait_until(first_cancelled.is_set)
    await _wait_until(queued_started.is_set)
    await _wait_until(lambda: registry.registered_count == 0)

    assert registry.active_count == 0
    assert not registry.is_running("first")
    assert not registry.is_running("queued")


async def test_guard_failure_tied_with_acquire_releases_semaphore_permit() -> None:
    registry = JobRegistry(max_concurrent=1)
    guard = asyncio.get_running_loop().create_future()

    class SimultaneousSemaphore:
        def __init__(self) -> None:
            self._value = 1
            self.release_count = 0

        async def acquire(self) -> bool:
            self._value -= 1
            guard.set_exception(RuntimeError("run claim lost"))
            return True

        def release(self) -> None:
            self._value += 1
            self.release_count += 1

    semaphore = SimultaneousSemaphore()
    registry._semaphore = semaphore  # type: ignore[assignment]
    pipeline = asyncio.sleep(0)
    await registry.register(
        "guard-acquire-tie",
        pipeline,
        liveness_guard=guard,
    )
    task = registry.get_task("guard-acquire-tie")
    assert task is not None

    with pytest.raises(RuntimeError, match="run claim lost"):
        await task

    assert semaphore.release_count == 1
    assert semaphore._value == 1
    assert registry.active_count == 0
    assert registry.registered_count == 0
    assert pipeline.cr_frame is None


async def test_cancel_returns_while_non_cancellable_work_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    job_started = asyncio.Event()
    allow_cancel = asyncio.Event()

    async def draining_job() -> None:
        job_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await allow_cancel.wait()
            raise

    monkeypatch.setattr(registry_module, "_CANCEL_DRAIN_WAIT_SECONDS", 0.01)
    await registry.register("draining", draining_job())
    await job_started.wait()

    assert await registry.cancel("draining") is True
    assert registry.is_running("draining")

    allow_cancel.set()
    task = registry.get_task("draining")
    assert task is not None
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not registry.is_running("draining")


async def test_overlapping_cancel_drains_entire_cleanup_transaction() -> None:
    registry = JobRegistry(max_concurrent=1)
    session_id = "overlapping-cancel"
    job_started = asyncio.Event()
    guard_started = asyncio.Event()
    guard_cancelled = asyncio.Event()
    release_guard = asyncio.Event()
    cleanup_called = asyncio.Event()

    async def job() -> None:
        job_started.set()
        await asyncio.Event().wait()

    async def guard() -> None:
        guard_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            guard_cancelled.set()
            await release_guard.wait()
            raise

    async def cleanup() -> None:
        cleanup_called.set()

    await registry.register(
        session_id,
        job(),
        liveness_guard=guard(),
        on_finish=cleanup,
    )
    task = registry.get_task(session_id)
    assert task is not None
    cleanup_complete = registry._cleanup_events[session_id]
    await job_started.wait()
    await guard_started.wait()

    first_cancel = asyncio.create_task(registry.cancel(session_id))
    await guard_cancelled.wait()
    await registry._lock.acquire()
    try:
        second_cancel = asyncio.create_task(registry.cancel(session_id))
        await asyncio.sleep(0)
        release_guard.set()
        await asyncio.sleep(0)
    finally:
        registry._lock.release()

    assert await asyncio.wait_for(first_cancel, timeout=1) is True
    assert await asyncio.wait_for(second_cancel, timeout=1) is True
    assert cleanup_called.is_set()
    assert cleanup_complete.is_set()
    assert registry._cleanup_events == {}
    assert registry.registered_count == 0
    assert registry.active_count == 0


async def test_running_job_consumes_cancelled_finish_callback() -> None:
    registry = JobRegistry(max_concurrent=1)

    async def cancelled_cleanup() -> None:
        raise asyncio.CancelledError

    await registry.register(
        "cancelled-finish",
        asyncio.sleep(0),
        on_finish=cancelled_cleanup,
    )
    task = registry.get_task("cancelled-finish")
    assert task is not None
    await task
    assert registry.registered_count == 0
    assert registry._cleanup_events == {}


async def test_admission_waits_for_finish_callback() -> None:
    registry = JobRegistry(max_concurrent=1)
    session_id = "finish-before-successor"
    finish_started = asyncio.Event()
    release_finish = asyncio.Event()

    async def cleanup() -> None:
        finish_started.set()
        await release_finish.wait()

    await registry.register(
        session_id,
        asyncio.sleep(0),
        run_id="a" * 32,
        on_finish=cleanup,
    )
    task = registry.get_task(session_id)
    assert task is not None
    await finish_started.wait()

    assert registry.get_task(session_id) is None
    assert registry.is_running(session_id)
    assert not await registry.reserve_admission(session_id, "b" * 32)

    release_finish.set()
    await task
    assert not registry.is_running(session_id)
    assert await registry.reserve_admission(session_id, "b" * 32)
    assert await registry.release_admission(session_id, "b" * 32)


async def test_cleanup_transaction_drains_repeated_outer_cancellation() -> None:
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def cleanup() -> None:
        cleanup_started.set()
        await release_cleanup.wait()

    owner = asyncio.create_task(JobRegistry._await_cleanup_transaction(cleanup()))
    await cleanup_started.wait()
    owner.cancel()
    await asyncio.sleep(0)
    owner.cancel()
    await asyncio.sleep(0)
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await owner


async def test_cleanup_transaction_consumes_child_failure_after_cancellation() -> None:
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def failing_cleanup() -> None:
        cleanup_started.set()
        await release_cleanup.wait()
        raise RuntimeError("cleanup failed")

    owner = asyncio.create_task(
        JobRegistry._await_cleanup_transaction(failing_cleanup())
    )
    await cleanup_started.wait()
    owner.cancel()
    await asyncio.sleep(0)
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await owner


async def test_liveness_guard_runs_and_fails_while_job_is_queued() -> None:
    registry = JobRegistry(max_concurrent=1)
    blocker_started = asyncio.Event()
    blocker_release = asyncio.Event()
    guard_started = asyncio.Event()
    guard_failure = asyncio.Event()
    queued_started = asyncio.Event()

    async def blocker() -> None:
        blocker_started.set()
        await blocker_release.wait()

    async def queued_job() -> None:
        queued_started.set()

    async def failing_guard() -> None:
        guard_started.set()
        await guard_failure.wait()
        raise RuntimeError("lease lost")

    await registry.register("blocker", blocker())
    await blocker_started.wait()
    queued = queued_job()
    await registry.register(
        "queued",
        queued,
        liveness_guard=failing_guard(),
    )
    queued_task = registry.get_task("queued")
    assert queued_task is not None
    await guard_started.wait()
    assert not queued_started.is_set()

    guard_failure.set()
    with pytest.raises(RuntimeError, match="lease lost"):
        await queued_task

    assert queued.cr_frame is None
    assert not queued_started.is_set()
    blocker_release.set()
    blocker_task = registry.get_task("blocker")
    assert blocker_task is not None
    await blocker_task


async def test_liveness_guard_failure_cancels_active_job() -> None:
    registry = JobRegistry(max_concurrent=1)
    job_started = asyncio.Event()
    job_cancelled = asyncio.Event()
    guard_failure = asyncio.Event()
    cleanup_called = asyncio.Event()

    async def active_job() -> None:
        job_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            job_cancelled.set()
            raise

    async def failing_guard() -> None:
        await guard_failure.wait()
        raise RuntimeError("lease lost")

    async def cleanup() -> None:
        cleanup_called.set()

    await registry.register(
        "active",
        active_job(),
        liveness_guard=failing_guard(),
        on_finish=cleanup,
    )
    task = registry.get_task("active")
    assert task is not None
    await job_started.wait()

    guard_failure.set()
    with pytest.raises(RuntimeError, match="lease lost"):
        await task

    assert job_cancelled.is_set()
    assert cleanup_called.is_set()
    assert registry.active_count == 0


async def test_liveness_guard_is_drained_before_finish_callback() -> None:
    registry = JobRegistry(max_concurrent=1)
    job_started = asyncio.Event()
    finish_job = asyncio.Event()
    guard_started = asyncio.Event()
    guard_stopped = asyncio.Event()
    cleanup_called = asyncio.Event()

    async def job() -> None:
        job_started.set()
        await finish_job.wait()

    async def guard() -> None:
        guard_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            guard_stopped.set()

    async def cleanup() -> None:
        assert guard_stopped.is_set()
        assert registry.get_task("finished") is None
        assert registry.is_running("finished")
        assert registry.active_count == 0
        cleanup_called.set()

    await registry.register(
        "finished",
        job(),
        liveness_guard=guard(),
        on_finish=cleanup,
    )
    task = registry.get_task("finished")
    assert task is not None
    await job_started.wait()
    await guard_started.wait()

    finish_job.set()
    await task

    assert guard_stopped.is_set()
    assert cleanup_called.is_set()


async def test_rejected_liveness_guard_is_drained_before_register_returns() -> None:
    registry = JobRegistry(max_concurrent=1)
    release_owner = asyncio.Event()
    guard_started = asyncio.Event()
    guard_stopped = asyncio.Event()

    async def owner() -> None:
        await release_owner.wait()

    async def rejected_job() -> None:
        await asyncio.Event().wait()

    async def guard() -> None:
        guard_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            guard_stopped.set()

    await registry.register("same-session", owner())
    guard_task = asyncio.create_task(guard())
    await guard_started.wait()
    rejected = rejected_job()

    with pytest.raises(RuntimeError, match="already registered"):
        await registry.register(
            "same-session",
            rejected,
            liveness_guard=_TaskBackedGuard(guard_task),
        )

    assert rejected.cr_frame is None
    assert guard_task.done()
    assert guard_stopped.is_set()

    release_owner.set()
    owner_task = registry.get_task("same-session")
    assert owner_task is not None
    await owner_task


async def test_rejected_task_guard_is_cancelled_and_drained() -> None:
    registry = JobRegistry(max_concurrent=1)
    release_owner = asyncio.Event()
    guard_started = asyncio.Event()
    guard_stopped = asyncio.Event()

    async def owner() -> None:
        await release_owner.wait()

    async def rejected_job() -> None:
        await asyncio.Event().wait()

    async def guard() -> None:
        guard_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            guard_stopped.set()

    await registry.register("same-session", owner())
    guard_task = asyncio.create_task(guard())
    await guard_started.wait()
    rejected = rejected_job()

    with pytest.raises(RuntimeError, match="already registered"):
        await registry.register(
            "same-session",
            rejected,
            liveness_guard=guard_task,
        )

    assert rejected.cr_frame is None
    assert guard_task.done()
    assert guard_stopped.is_set()

    release_owner.set()
    owner_task = registry.get_task("same-session")
    assert owner_task is not None
    await owner_task


async def test_rejected_liveness_guard_drain_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    release_owner = asyncio.Event()
    guard_started = asyncio.Event()
    guard_cancelled = asyncio.Event()
    release_guard = asyncio.Event()

    async def owner() -> None:
        await release_owner.wait()

    async def rejected_job() -> None:
        await asyncio.Event().wait()

    async def cancellation_resistant_guard() -> None:
        guard_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            guard_cancelled.set()
            await release_guard.wait()
            raise RuntimeError("guard failed after cancellation")

    monkeypatch.setattr(registry_module, "_CANCEL_DRAIN_WAIT_SECONDS", 0.01)
    await registry.register("same-session", owner())
    guard_task = asyncio.create_task(cancellation_resistant_guard())
    await guard_started.wait()
    rejected = rejected_job()

    with pytest.raises(RuntimeError, match="already registered"):
        await asyncio.wait_for(
            registry.register(
                "same-session",
                rejected,
                liveness_guard=_TaskBackedGuard(guard_task),
            ),
            timeout=0.1,
        )

    assert rejected.cr_frame is None
    assert guard_cancelled.is_set()
    assert not guard_task.done()

    release_guard.set()
    guard_result = await asyncio.gather(guard_task, return_exceptions=True)
    assert isinstance(guard_result[0], RuntimeError)
    await _wait_until(lambda: "Asynchronous pipeline cleanup failed" in caplog.text)
    release_owner.set()
    owner_task = registry.get_task("same-session")
    assert owner_task is not None
    await owner_task


async def test_prestart_liveness_guard_is_drained_before_finish_callback() -> None:
    registry = JobRegistry(max_concurrent=1)
    guard_started = asyncio.Event()
    guard_stopped = asyncio.Event()
    cleanup_called = asyncio.Event()

    async def job() -> None:
        await asyncio.Event().wait()

    async def guard() -> None:
        guard_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            guard_stopped.set()

    async def cleanup() -> None:
        assert guard_stopped.is_set()
        cleanup_called.set()

    guard_task = asyncio.create_task(guard())
    await guard_started.wait()
    pipeline = job()
    await registry.register(
        "cancelled-before-start",
        pipeline,
        liveness_guard=_TaskBackedGuard(guard_task),
        on_finish=cleanup,
    )
    task = registry.get_task("cancelled-before-start")
    assert task is not None
    assert await registry.cancel("cancelled-before-start")

    assert pipeline.cr_frame is None
    assert guard_task.done()
    assert guard_stopped.is_set()
    assert cleanup_called.is_set()
    assert registry.get_task("cancelled-before-start") is None


async def test_prestart_cleanup_waits_for_cancellation_resistant_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    guard_started = asyncio.Event()
    guard_cancelled = asyncio.Event()
    release_guard = asyncio.Event()
    cleanup_called = asyncio.Event()

    async def job() -> None:
        await asyncio.Event().wait()

    async def cancellation_resistant_guard() -> None:
        guard_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            guard_cancelled.set()
            await release_guard.wait()

    async def cleanup() -> None:
        cleanup_called.set()

    monkeypatch.setattr(registry_module, "_CANCEL_DRAIN_WAIT_SECONDS", 0.01)
    guard_task = asyncio.create_task(cancellation_resistant_guard())
    await guard_started.wait()
    pipeline = job()
    await registry.register(
        "resistant-prestart-guard",
        pipeline,
        liveness_guard=_TaskBackedGuard(guard_task),
        on_finish=cleanup,
    )

    assert await asyncio.wait_for(
        registry.cancel("resistant-prestart-guard"),
        timeout=1,
    )
    assert guard_cancelled.is_set()
    assert not cleanup_called.is_set()
    assert registry.is_running("resistant-prestart-guard")
    assert len(registry._background_cleanup_tasks) == 1

    release_guard.set()
    await asyncio.wait_for(cleanup_called.wait(), timeout=1)
    await _wait_until(lambda: not registry._background_cleanup_tasks)
    assert pipeline.cr_frame is None
    assert guard_task.done()
    assert registry.get_task("resistant-prestart-guard") is None
    assert not registry.is_running("resistant-prestart-guard")
