# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test configuration and fixtures for Joint Agent Service.

This module provides:
- Environment setup with temp directories
- FastAPI app and AsyncClient fixtures
- Deterministic stub executor that respects the real semaphore
- Concurrency tracking for validation
"""

import asyncio
import json
import os
import tempfile
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from ..service.runtime.registry import JobRegistry

os.environ.setdefault("TMPDIR", tempfile.gettempdir())


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

    os.environ["JA_SESSION_STORAGE_PATH"] = str(sessions)
    os.environ["JA_MAX_ACTIVE_SESSIONS"] = "1"
    os.environ["JA_SESSION_TTL_HOURS"] = "1"
    os.environ["JA_STORAGE_KIND"] = "local"

    return {"sessions": sessions}


@pytest.fixture(scope="session")
def app(_env_for_service):
    """Create and configure FastAPI app."""
    from ..service.main import app
    from ..service.routers import (
        artifacts_router,
        pipeline_router,
        sessions_router,
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
    artifacts_router.set_session_manager(session_mgr)
    sessions_router.set_session_manager(session_mgr)

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

        tasks: list[asyncio.Task[Any]] = list(registry._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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

    from ..service.runtime import get_job_registry
    from ..service.session.cache_publications import CACHE_PUBLICATIONS_FIELD
    from ..service.session.manager import SessionManager
    from ..service.workers.executor import (
        _merged_cache_publications,
        _publish_cache_publications,
    )

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
        run_id: str,
        config_dict: dict,
        session_manager: SessionManager,
        only_steps: list[str] | None = None,
    ):
        """Deterministic stub executor."""
        manager = session_manager
        await _inc()
        try:
            session_dir = manager.get_session_dir(session_id)

            assert await manager.update_session_for_run(
                session_id,
                run_id,
                {"status": "running"},
            )

            ds = session_dir / "cache" / "dataset"
            preds = session_dir / "cache" / "predictions"
            ds.mkdir(parents=True, exist_ok=True)
            preds.mkdir(parents=True, exist_ok=True)

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

            # STEP 2: Prepare Dataset
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

            # STEP 3: Prediction
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

            await manager.update_step_progress(
                session_id,
                "consistency_pass",
                {
                    "current": 1,
                    "total": 1,
                    "percent": 100,
                    "message": "Checking prediction consistency",
                },
            )
            delay = float(os.getenv("TEST_STEP_DELAY", "0.01"))
            await asyncio.sleep(delay)

            await manager.mark_step_completed(session_id, "consistency_pass")

            (preds / "articulation_candidates.json").write_text(
                json.dumps(
                    {
                        "schema_version": "joint-agent-stage2-v0",
                        "summary": {"candidate_count": 0},
                        "candidates": [],
                    }
                )
            )
            (preds / "articulation_candidates.html").write_text(
                "<html><body>Articulation Candidates</body></html>"
            )
            (preds / "report.html").write_text("<html><body>Predictions</body></html>")
            await manager.update_step_progress(
                session_id,
                "infer_articulation_candidates",
                {
                    "current": 1,
                    "total": 1,
                    "percent": 100,
                    "message": "Inferring articulation candidates",
                },
            )
            delay = float(os.getenv("TEST_STEP_DELAY", "0.01"))
            await asyncio.sleep(delay)

            await manager.mark_step_completed(
                session_id, "infer_articulation_candidates"
            )

            produced, published = await _publish_cache_publications(
                manager,
                session_id,
                run_id,
                only_steps,
            )
            cache_publications = await _merged_cache_publications(
                manager,
                session_id,
                produced,
                published,
            )
            assert await manager.terminalize_and_release_run(
                session_id,
                run_id,
                {
                    "status": "completed",
                    CACHE_PUBLICATIONS_FIELD: cache_publications,
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
            await manager.terminalize_and_release_run(
                session_id,
                run_id,
                {"status": "cancelled"},
            )
            raise

        except Exception as e:
            await manager.terminalize_and_release_run(
                session_id,
                run_id,
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
