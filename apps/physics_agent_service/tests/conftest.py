# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test configuration and fixtures for Physics Agent Service.

This module provides:
- Environment setup with temp directories
- FastAPI app and AsyncClient fixtures
- Deterministic stub executor that respects the real semaphore
- Concurrency tracking for validation
"""

import asyncio
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from ..service.runtime.registry import JobRegistry


def make_pipeline_files(
    usd_content: bytes = b"#usda 1.0\n",
    usd_filename: str = "scene.usda",
):
    """Create multipart files for pipeline creation."""
    return [
        ("usd_file", (usd_filename, usd_content, "application/octet-stream")),
    ]


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def _env_for_service(tmp_path_factory):
    """Configure environment with temp paths before importing service modules."""
    sessions = tmp_path_factory.mktemp("sessions")

    os.environ["PA_SESSION_STORAGE_PATH"] = str(sessions)
    os.environ["PA_MAX_ACTIVE_SESSIONS"] = "1"
    os.environ["PA_SESSION_TTL_HOURS"] = "1"
    os.environ["PA_STORAGE_KIND"] = "local"
    # Per-test tmp dirs sit under tmp_path_factory's base — allow that so
    # Mode-A tests using `dataset_path=tmp_path / ...` are accepted by the
    # /predict safety check.
    os.environ["PA_DATASET_ALLOWED_ROOTS"] = str(tmp_path_factory.getbasetemp())

    return {"sessions": sessions}


@pytest.fixture(scope="session")
def app(_env_for_service):
    """Create and configure FastAPI app."""
    from ..service.main import app
    from ..service.routers import (
        artifacts_router,
        pipeline_router,
        predict_router,
        refine_router,
        sessions_router,
        tune_router,
    )
    from ..service.session.manager import SessionManager
    from ..service.storage import LocalSessionStore

    store = LocalSessionStore(root_dir=str(_env_for_service["sessions"]))
    session_mgr = SessionManager(
        storage_path=Path(_env_for_service["sessions"]),
        ttl_hours=1,
        store=store,
    )
    pipeline_router.set_session_manager(session_mgr)
    predict_router.set_session_manager(session_mgr)
    artifacts_router.set_session_manager(session_mgr)
    sessions_router.set_session_manager(session_mgr)
    tune_router.set_session_manager(session_mgr)
    refine_router.set_session_manager(session_mgr)

    return app


@pytest.fixture
async def client(app: object) -> AsyncIterator[httpx.AsyncClient]:
    """Create AsyncClient for making test requests."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ============================================================================
# STUB EXECUTOR AND CONCURRENCY TRACKING
# ============================================================================


@pytest.fixture(autouse=True, scope="function")
async def _reset_job_registry() -> AsyncGenerator[None, None]:
    """Reset the global JobRegistry between tests."""
    from ..service.runtime import registry as registry_module

    async def _cancel_lingering_tasks() -> None:
        registry: JobRegistry | None = registry_module._job_registry
        if registry is None:
            return

        # Skip non-task entries (e.g. an in-flight reservation sentinel)
        # so we don't AttributeError on .cancel(); they're cleaned up when
        # we drop the registry reference below.
        tasks: list[asyncio.Task[Any]] = [
            t for t in registry._tasks.values() if isinstance(t, asyncio.Task)
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Close any inner coroutines whose wrapping _run_with_cleanup task
        # was cancelled before it ever started — otherwise Python emits a
        # "coroutine was never awaited" RuntimeWarning when the coro is
        # GC'd between tests. Mirrors registry.cancel()'s explicit close.
        for task in tasks:
            inner_coro = getattr(task, "_wu_inner_coro", None)
            if inner_coro is not None and hasattr(inner_coro, "close"):
                inner_coro.close()

    await _cancel_lingering_tasks()
    registry_module._job_registry = None

    yield

    await _cancel_lingering_tasks()
    registry_module._job_registry = None


@pytest.fixture(autouse=True, scope="function")
def _stub_executor(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> dict[str, Callable[[], int]]:
    """Replace the expensive execute_pipeline_async with a deterministic stub.

    The stub:
    - Respects the REAL global semaphore from JobRegistry
    - Simulates progress through all steps
    - Creates minimal but valid artifacts
    - Tracks peak concurrency
    """
    if request.node.get_closest_marker("real_executor"):
        return {
            "max_concurrency_seen": lambda: 0,
            "current_concurrency": lambda: 0,
        }

    from ..service.routers import tune_router

    # API tests exercise the deterministic stub runtime, not the separately
    # provisioned OvPhysX daemon used by production deployments.
    monkeypatch.setattr(tune_router, "ovphysx_runtime_available", lambda: True)

    from ..service.runtime import get_job_registry
    from ..service.session.manager import SessionManager

    get_job_registry()

    # Track concurrency
    max_seen = {"value": 0}
    current = {"value": 0}
    lock = asyncio.Lock()

    async def _inc():
        async with lock:
            current["value"] += 1
            max_seen["value"] = max(max_seen["value"], current["value"])

    async def _dec():
        async with lock:
            current["value"] -= 1

    async def fake_execute(
        session_id: str,
        config_dict: dict,
        session_manager: SessionManager,
        only_steps: list[str] | None = None,
    ):
        """Deterministic stub executor."""
        manager = session_manager
        await _inc()
        try:
            session_dir = manager.get_session_dir(session_id)

            await manager.update_session(session_id, {"status": "running"})

            ds = session_dir / "cache" / "dataset"
            preds = session_dir / "cache" / "predictions"
            physics = session_dir / "cache" / "physics"
            ds.mkdir(parents=True, exist_ok=True)
            preds.mkdir(parents=True, exist_ok=True)
            physics.mkdir(parents=True, exist_ok=True)

            # STEP 1: Rendering (0-50% overall)
            for pct in (10, 25, 50):
                await manager.update_step_progress(
                    session_id,
                    "build_dataset_usd",
                    {
                        "current": pct,
                        "total": 100,
                        "percent": pct,
                        "message": f"Rendering: {pct}%",
                    },
                )
                delay = float(os.getenv("TEST_STEP_DELAY", "0.01"))
                await asyncio.sleep(delay)

            dataset_file = ds / "dataset.jsonl"
            with dataset_file.open("w") as f:
                for i in range(10):
                    f.write(
                        json.dumps(
                            {
                                "id": f"/p{i}",
                                "type": "Mesh",
                                "images": {"prim_only": f"img_{i}.png"},
                            }
                        )
                        + "\n"
                    )

            await manager.mark_step_completed(session_id, "build_dataset_usd")

            # STEP 2: Prepare Dataset (stays at 50%)
            await manager.update_step_progress(
                session_id,
                "build_dataset_prepare_dataset",
                {
                    "current": 10,
                    "total": 10,
                    "percent": 50,
                    "message": "Preparing dataset",
                },
            )
            delay = float(os.getenv("TEST_STEP_DELAY", "0.01"))
            await asyncio.sleep(delay)

            await manager.mark_step_completed(
                session_id, "build_dataset_prepare_dataset"
            )

            # STEP 3: Prediction (50-100% overall)
            for i, pct in enumerate((60, 80, 100)):
                with (preds / "predictions.jsonl").open("a") as f:
                    category = ["furniture", "electronics", "decor"][i % 3]
                    f.write(
                        json.dumps(
                            {
                                "id": f"/p{i}",
                                "classification": category,
                                "confidence": 0.95,
                            }
                        )
                        + "\n"
                    )

                await manager.update_step_progress(
                    session_id,
                    "predict",
                    {
                        "current": i + 1,
                        "total": 10,
                        "percent": pct,
                        "message": f"Predicting: {i + 1}/10",
                    },
                )
                delay = float(os.getenv("TEST_STEP_DELAY", "0.01"))
                await asyncio.sleep(delay)

            await manager.mark_step_completed(session_id, "predict")

            # STEP 4: apply_physics — author a minimal fake output USD so that
            # /artifacts/{id}/output-usd returns the expected file. We have to
            # drive update_step_progress first because mark_step_completed is a
            # no-op when current_step doesn't match (predict cleared it).
            await manager.update_step_progress(
                session_id,
                "apply_physics",
                {
                    "current": 1,
                    "total": 1,
                    "percent": 100,
                    "message": "Applying physics",
                },
            )
            delay = float(os.getenv("TEST_STEP_DELAY", "0.01"))
            await asyncio.sleep(delay)
            input_suffix = Path(
                config_dict.get("input", {}).get("usd_path", "scene.usda")
            ).suffix.lower()
            if input_suffix not in {".usd", ".usda", ".usdc", ".usdz"}:
                input_suffix = ".usd"
            output_suffix = ".usda" if input_suffix == ".usdz" else input_suffix
            output_path = physics / f"scene_physics{output_suffix}"
            output_path.write_text("#usda 1.0\n# stub apply_physics output\n")
            await manager.mark_step_completed(session_id, "apply_physics")

            await manager.update_session(
                session_id,
                {
                    "status": "completed",
                    "results": {
                        "prims_processed": 10,
                        "images_generated": 20,
                        "predictions_made": 10,
                    },
                    "completed_at": "1970-01-01T00:00:01Z",
                    "can_cancel": False,
                },
            )

        except asyncio.CancelledError:
            await manager.update_session(session_id, {"status": "cancelled"})
            raise

        except Exception as e:
            await manager.update_session(
                session_id,
                {
                    "status": "failed",
                    "error": str(e),
                    "failed_step": "unknown",
                },
            )
            raise

        finally:
            await _dec()

    from ..service.routers import pipeline_router

    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute, raising=True
    )

    return {
        "max_concurrency_seen": lambda: max_seen["value"],
        "current_concurrency": lambda: current["value"],
    }


# ============================================================================
# HELPER FIXTURES
# ============================================================================


@pytest.fixture
def session_manager(_env_for_service):
    """Create a SessionManager instance for direct testing."""
    from ..service.session.manager import SessionManager
    from ..service.storage import LocalSessionStore

    store = LocalSessionStore(root_dir=str(_env_for_service["sessions"]))
    return SessionManager(
        storage_path=Path(_env_for_service["sessions"]),
        ttl_hours=1,
        store=store,
    )


# ============================================================================
# TEST MARKERS
# ============================================================================


def pytest_configure(config):
    """Register custom test markers."""
    config.addinivalue_line("markers", "unit: Unit tests for isolated components")
    config.addinivalue_line(
        "markers", "api: API/integration tests with stubbed executor"
    )
    config.addinivalue_line(
        "markers",
        "real_executor: API tests that run through the real service executor",
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests including concurrency")
    config.addinivalue_line(
        "markers", "concurrency: Tests for concurrent execution limits"
    )
