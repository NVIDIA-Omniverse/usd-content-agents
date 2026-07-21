# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Callable

import pytest

from ...service.runtime import registry as registry_module
from ...service.runtime.registry import _RESERVED, JobRegistry


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not met before timeout")
        await asyncio.sleep(0.01)


async def test_register_rejects_duplicate_active_session() -> None:
    """Two concurrent register() calls for the same session_id must not
    both succeed — JobRegistry raises ValueError on the second so the
    first task isn't silently leaked into the event loop."""
    registry = JobRegistry(max_concurrent=2)

    first_started = asyncio.Event()
    first_release = asyncio.Event()

    async def first_job() -> None:
        first_started.set()
        await first_release.wait()

    async def second_job() -> None:
        # Should never run — register() is expected to reject.
        raise AssertionError("second_job must not run for duplicate session_id")

    await registry.register("dup", first_job())
    await _wait_until(first_started.is_set)

    second_coro = second_job()
    try:
        await registry.register("dup", second_coro)
    except ValueError as e:
        assert "dup" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on duplicate register")
    finally:
        # Close the never-awaited coroutine to avoid a runtime warning.
        second_coro.close()

    # Cleanup: release first job so the registry drains.
    first_release.set()
    await _wait_until(lambda: registry.registered_count == 0)


async def test_register_allows_resubmit_after_completion() -> None:
    """Once the previous task is done, register() must allow re-use of
    the same session_id (the duplicate guard only fires while a task is
    still active)."""
    registry = JobRegistry(max_concurrent=2)

    first_release = asyncio.Event()
    second_release = asyncio.Event()
    second_started = asyncio.Event()

    async def first_job() -> None:
        await first_release.wait()

    async def second_job() -> None:
        second_started.set()
        await second_release.wait()

    await registry.register("reuse", first_job())
    first_release.set()
    await _wait_until(lambda: registry.registered_count == 0)

    # Same session_id again — should now succeed.
    await registry.register("reuse", second_job())
    await _wait_until(second_started.is_set)
    second_release.set()
    await _wait_until(lambda: registry.registered_count == 0)


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


# ---------------------------------------------------------------------------
# Direct coverage for the JobReservation API introduced by the round-3
# atomic slot-reservation refactor. The route-level smoke test exercises the
# happy path; these tests pin the internal contracts (lost-slot rejection,
# double-consume rejection, release-on-context-exit) so the executor / route
# can rely on them under future refactors.
# ---------------------------------------------------------------------------


async def test_reserve_holds_slot_with_sentinel_until_start() -> None:
    """``reserve()`` must mark ``is_running`` true and reject ``cancel`` —
    the sentinel slot has no asyncio.Task to cancel yet, so cancel() must
    return False rather than crashing on a missing ``.cancel`` attribute."""
    registry = JobRegistry(max_concurrent=1)

    reservation = await registry.reserve("alpha")
    try:
        assert registry.is_running("alpha")
        assert registry._tasks.get("alpha") is _RESERVED
        # cancel() against a sentinel must not raise
        cancelled = await registry.cancel("alpha")
        assert cancelled is False
        # The slot is still claimed by the reservation
        assert registry._tasks.get("alpha") is _RESERVED
    finally:
        await reservation.release()
        assert "alpha" not in registry._tasks
        assert not registry.is_running("alpha")


async def test_reservation_aexit_releases_on_route_exception() -> None:
    """The ``async with reservation:`` context must release the sentinel
    slot when an exception (e.g. an HTTPException raised by the route)
    fires between ``reserve()`` and ``start()``. This is what guarantees
    that a failed Mode-A staging step doesn't leak the slot forever."""
    registry = JobRegistry(max_concurrent=1)

    class _BoomError(Exception):
        pass

    with pytest.raises(_BoomError):
        async with await registry.reserve("beta"):
            assert registry.is_running("beta")
            raise _BoomError

    assert "beta" not in registry._tasks
    assert not registry.is_running("beta")


async def test_reservation_double_start_closes_second_coro() -> None:
    """Calling ``start()`` twice on the same reservation must raise and
    must close the rejected coroutine before raising. Python sets
    ``coro.cr_frame`` to ``None`` once a coroutine has been closed, so
    asserting on that pins the close-before-raise contract directly
    (without depending on a GC pass or a particular RuntimeWarning
    filter setting in the test harness)."""
    registry = JobRegistry(max_concurrent=1)

    async def real_job() -> None:
        await asyncio.sleep(0)

    async def second_job() -> None:
        raise AssertionError("second_job must not run after first start()")

    reservation = await registry.reserve("gamma")
    await reservation.start(real_job())

    second_coro = second_job()
    with pytest.raises(RuntimeError, match="already consumed"):
        await reservation.start(second_coro)

    assert second_coro.cr_frame is None, (
        "JobReservation.start() must close the incoming coro before raising "
        "on the 'already consumed' path"
    )

    # Drain the real task we started so the conftest registry-cleanup
    # fixture sees a clean state.
    await _wait_until(lambda: registry.registered_count == 0)


async def test_reserve_rejects_while_reserved() -> None:
    """Two consecutive ``reserve()`` calls for the same session must reject
    the second with ValueError, even before ``start()`` runs. This is the
    invariant that the predict route's same-pod rerun-loser test depends on
    — it ensures the loser cannot mutate any session state at all."""
    registry = JobRegistry(max_concurrent=1)

    first = await registry.reserve("delta")
    try:
        with pytest.raises(ValueError, match="active job"):
            await registry.reserve("delta")
    finally:
        await first.release()


async def test_release_is_idempotent_after_start() -> None:
    """Once ``start()`` has been called, ``release()`` must be a no-op so
    `async with` exit doesn't tear down a real running task. The slot is
    owned by ``_run_with_cleanup`` from that point on."""
    registry = JobRegistry(max_concurrent=1)

    job_release = asyncio.Event()

    async def long_job() -> None:
        await job_release.wait()

    reservation = await registry.reserve("epsilon")
    await reservation.start(long_job())
    assert registry.is_running("epsilon")

    # release() after start() must NOT delete the live task slot
    await reservation.release()
    assert registry.is_running("epsilon")

    # cleanup
    job_release.set()
    await _wait_until(lambda: registry.registered_count == 0)


async def test_reserve_release_on_unused_context_allows_resubmit() -> None:
    """When the route claims a reservation but never calls ``start()`` —
    e.g. because an HTTPException fires between ``reserve()`` and
    ``start()`` — the context manager's ``__aexit__`` must release the
    slot so a subsequent ``register()`` for the same session_id is not
    permanently 409'd."""
    registry = JobRegistry(max_concurrent=1)

    started = asyncio.Event()
    second_release = asyncio.Event()

    async def first_job() -> None:
        started.set()

    # Land a fresh successful register/run cycle so the slot starts empty.
    await registry.register("zeta", first_job())
    await _wait_until(started.is_set)
    await _wait_until(lambda: registry.registered_count == 0)

    # Now exercise the reserve()-then-bail path: claim the slot, then
    # exit the async-with without calling start(). __aexit__ must
    # release so the next register() succeeds.
    async with await registry.reserve("zeta"):
        pass

    async def second_job() -> None:
        await second_release.wait()

    await registry.register("zeta", second_job())
    assert registry.is_running("zeta")
    second_release.set()
    await _wait_until(lambda: registry.registered_count == 0)


async def test_register_releases_slot_on_start_failure() -> None:
    """If ``JobReservation.start()`` raises (e.g. an unexpected internal
    error like the slot being mutated under us), ``register()`` must
    release the slot before propagating so the same session_id can be
    submitted again. We force this by mutating ``_tasks`` between
    ``reserve()`` and ``start()`` so the lost-reservation branch fires."""
    registry = JobRegistry(max_concurrent=1)

    second_release = asyncio.Event()
    started = asyncio.Event()

    real_coro_holder: dict = {}

    async def real_job() -> None:
        started.set()
        await second_release.wait()

    # Manually walk through register()'s logic with an injected failure: the
    # public API is reserve()→start(), and we mutate the slot in between
    # to trigger start()'s "lost reservation" RuntimeError. start() must
    # close the coro and the route must release; afterwards, the slot is
    # free and a fresh register() succeeds.
    reservation = await registry.reserve("eta")
    # Steal the slot out from under the reservation
    registry._tasks.pop("eta", None)

    real_coro = real_job()
    real_coro_holder["coro"] = real_coro
    with pytest.raises(RuntimeError, match="lost before start"):
        await reservation.start(real_coro)
    # start() should have closed the rejected coro before raising
    assert real_coro.cr_frame is None

    # Slot is now free and a fresh register() succeeds.
    async def fresh_job() -> None:
        started.set()
        await second_release.wait()

    await registry.register("eta", fresh_job())
    await _wait_until(started.is_set)
    second_release.set()
    await _wait_until(lambda: registry.registered_count == 0)


async def test_start_closes_wrapper_and_inner_when_task_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    reservation = await registry.reserve("task-create-failure")
    captured: dict[str, object] = {}

    async def incoming_job() -> None:
        await asyncio.sleep(0)

    def fail_task_creation(wrapper: object) -> None:
        captured["wrapper"] = wrapper
        raise RuntimeError("simulated create_task failure")

    monkeypatch.setattr(
        registry_module,
        "_create_job_task",
        fail_task_creation,
    )
    incoming = incoming_job()
    with pytest.raises(RuntimeError, match="create_task failure"):
        await reservation.start(incoming)

    wrapper = captured["wrapper"]
    assert getattr(wrapper, "cr_frame") is None
    assert incoming.cr_frame is None
    await reservation.release()
    assert registry.registered_count == 0
    assert registry.active_count == 0


async def test_start_cancellation_while_waiting_for_lock_closes_incoming_coro() -> None:
    registry = JobRegistry(max_concurrent=1)
    reservation = await registry.reserve("start-lock-cancel")

    async def incoming_job() -> None:
        raise AssertionError("incoming_job must not start before task ownership")

    incoming = incoming_job()

    async def start_reserved_job() -> None:
        async with reservation:
            await reservation.start(incoming)

    await registry._lock.acquire()
    try:
        start_task = asyncio.create_task(start_reserved_job())
        await _wait_until(lambda: bool(registry._lock._waiters))

        start_task.cancel()
        await _wait_until(lambda: incoming.cr_frame is None)
        assert not start_task.done()
        assert registry._tasks.get("start-lock-cancel") is _RESERVED
        assert registry.is_running("start-lock-cancel")
    finally:
        registry._lock.release()

    # start() closes the still-unowned coroutine, while the reservation context
    # waits for the same lock so it can release the sentinel before propagating
    # cancellation to the route.
    with pytest.raises(asyncio.CancelledError):
        await start_task
    assert registry.registered_count == 0
    assert not registry.is_running("start-lock-cancel")


async def test_cancel_before_active_increment_preserves_capacity() -> None:
    registry = JobRegistry(max_concurrent=1)
    reservation = await registry.reserve("increment-race")

    async def never_started() -> None:
        await asyncio.Event().wait()

    await reservation.start(never_started())
    await registry._lock.acquire()
    try:
        # The wrapper acquires capacity, then blocks on the held registry lock
        # before it can increment active_count.
        await asyncio.sleep(0)
        task = registry.get_task("increment-race")
        assert task is not None
        assert registry._semaphore._value == 0
        task.cancel()
    finally:
        registry._lock.release()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert registry.active_count == 0
    assert registry._semaphore._value == 1
    assert registry.registered_count == 0
    assert not registry.is_running("increment-race")
