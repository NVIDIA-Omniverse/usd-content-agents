# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Callable

import pytest

from ...service.runtime import registry as registry_module
from ...service.runtime.bus import EventBus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.runtime.registry import DuplicateJobError, JobRegistry


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not met before timeout")
        await asyncio.sleep(0.01)


async def test_register_waits_for_free_slot_before_tracking_next_job() -> None:
    registry = JobRegistry(max_concurrent=1)

    first_started = asyncio.Event()
    first_release = asyncio.Event()
    first_cancelled = asyncio.Event()
    second_started = asyncio.Event()
    second_release = asyncio.Event()

    async def first_job() -> None:
        first_started.set()
        try:
            await first_release.wait()
        except asyncio.CancelledError:
            first_cancelled.set()
            raise

    async def second_job() -> None:
        second_started.set()
        await second_release.wait()

    await registry.register("first", first_job())
    await _wait_until(first_started.is_set)

    second_register = asyncio.create_task(registry.register("second", second_job()))
    await asyncio.sleep(0.05)

    assert not second_register.done()
    assert registry.active_count == 1
    assert registry.get_task("second") is None
    assert registry.is_running("first")

    assert await registry.cancel("first") is True
    await _wait_until(first_cancelled.is_set)

    await second_register
    await _wait_until(second_started.is_set)

    assert registry.active_count == 1
    assert registry.is_running("second")
    assert registry.get_task("first") is None
    assert registry.get_task("second") is not None

    second_release.set()
    await _wait_until(lambda: registry.get_task("second") is None)

    assert registry.active_count == 0
    assert not registry.is_running("second")


async def test_reservation_rejects_duplicate_during_before_start() -> None:
    registry = JobRegistry(max_concurrent=1)
    callback_started = asyncio.Event()
    callback_release = asyncio.Event()
    job_started = asyncio.Event()
    job_release = asyncio.Event()

    async def before_start() -> None:
        callback_started.set()
        await callback_release.wait()

    async def job() -> None:
        job_started.set()
        await job_release.wait()

    first = asyncio.create_task(
        registry.register("same", job(), before_start=before_start)
    )
    await callback_started.wait()
    duplicate = job()
    with pytest.raises(DuplicateJobError):
        await registry.register("same", duplicate)
    assert duplicate.cr_frame is None

    callback_release.set()
    await first
    await job_started.wait()
    job_release.set()
    await _wait_until(lambda: not registry.is_running("same"))


async def test_before_start_failure_and_caller_cancellation_release_resources() -> None:
    registry = JobRegistry(max_concurrent=1)

    async def job() -> None:
        await asyncio.sleep(10)

    failed_job = job()

    async def fail_before_start() -> None:
        raise RuntimeError("prepare failed")

    with pytest.raises(RuntimeError, match="prepare failed"):
        await registry.register(
            "failed",
            failed_job,
            before_start=fail_before_start,
        )
    assert failed_job.cr_frame is None
    assert registry.active_count == 0
    assert not registry.is_running("failed")

    callback_started = asyncio.Event()
    callback_release = asyncio.Event()
    cancelled_job = job()

    async def block_before_start() -> None:
        callback_started.set()
        await callback_release.wait()

    registration = asyncio.create_task(
        registry.register(
            "cancelled",
            cancelled_job,
            before_start=block_before_start,
        )
    )
    await callback_started.wait()
    assert registry.is_running("cancelled")
    assert registry.is_reserved("cancelled")
    assert await registry.cancel("cancelled") is True
    with pytest.raises(asyncio.CancelledError):
        await registration
    assert cancelled_job.cr_frame is None
    assert registry.active_count == 0
    assert not registry.is_running("cancelled")


async def test_immediate_pre_first_turn_cancellation_does_not_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    real_create_task = asyncio.create_task

    def create_cancelled_task(coro):
        task = real_create_task(coro)
        task.cancel()
        return task

    monkeypatch.setattr(registry_module.asyncio, "create_task", create_cancelled_task)

    async def job() -> None:
        await asyncio.sleep(10)

    job_coro = job()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(registry.register("instant", job_coro), timeout=1)

    assert job_coro.cr_frame is None
    assert registry.active_count == 0
    assert registry.get_task("instant") is None
    assert not registry.is_running("instant")

    # The released semaphore permits a subsequent registration.
    monkeypatch.setattr(registry_module.asyncio, "create_task", real_create_task)
    completed = asyncio.Event()

    async def next_job() -> None:
        completed.set()

    await asyncio.wait_for(registry.register("next", next_job()), timeout=1)
    await asyncio.wait_for(completed.wait(), timeout=1)
    await _wait_until(lambda: registry.active_count == 0)


async def test_caller_cancellation_collects_unstarted_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    wrapper_entered = asyncio.Event()

    async def stalled_wrapper(*_args: object, **_kwargs: object) -> None:
        wrapper_entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(registry, "_run_with_cleanup", stalled_wrapper)

    async def job() -> None:
        await asyncio.sleep(60)

    job_coro = job()
    registration = asyncio.create_task(registry.register("stalled", job_coro))
    await wrapper_entered.wait()
    registration.cancel()
    with pytest.raises(asyncio.CancelledError):
        await registration

    assert job_coro.cr_frame is None
    assert registry.active_count == 0
    assert registry.get_task("stalled") is None


async def test_seed_pending_preserves_open_queue_waiter_and_drops_stale_items() -> None:
    bus = EventBus()
    session_id = "session"
    queue = bus.get_queue(session_id)
    stale = ProgressEvent(
        session_id=session_id,
        step="old",
        state=StepState.COMPLETED,
    )
    await queue.put(stale)
    await bus.seed_pending_session(session_id)
    assert queue.empty()
    assert bus.get_queue(session_id) is queue

    waiter = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    await bus.seed_pending_session(session_id)
    current = ProgressEvent(
        session_id=session_id,
        step="new",
        state=StepState.RUNNING,
    )
    await bus.emit(current)
    assert await asyncio.wait_for(waiter, timeout=1) is current
