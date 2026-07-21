# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test configuration and fixtures for Material Agent Service.

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

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def _env_for_service(tmp_path_factory):
    """Configure environment with temp paths before importing service modules.

    This fixture must run early to set env vars before service.config loads.
    Using tmp_path_factory ensures directories exist for the entire test session.

    Returns:
        Dict with sessions and materials paths
    """
    # Create temp directories
    sessions = tmp_path_factory.mktemp("sessions")
    materials_dir = tmp_path_factory.mktemp("materials")
    mats_path = materials_dir / "materials.yaml"

    # Create minimal materials.yaml for config generation
    # Note: Event-driven executor requires 'binding' field for each material
    materials_yaml = """materials:
  library_path: "/app/materials/material_libs_v2.usd"
  entries:
    - name: "Aluminum"
      description: "Silver metal, corrosion resistant"
      binding: "/bindingPath/Aluminum"
    - name: "Copper"
      description: "Red metal, conductive"
      binding: "/bindingPath/Copper"
    - name: "Plastic"
      description: "Polymer material, lightweight"
      binding: "/bindingPath/Plastic"
    - name: "Rubber"
      description: "Elastomer, flexible"
      binding: "/bindingPath/Rubber"
    - name: "Fabric"
      description: "Textile material"
      binding: "/bindingPath/Fabric"
"""
    mats_path.write_text(materials_yaml)

    # Set environment variables (must be set BEFORE importing service modules)
    os.environ["MA_SESSION_STORAGE_PATH"] = str(sessions)
    os.environ["MA_MATERIALS_CONFIG_PATH"] = str(mats_path)
    os.environ["MA_MAX_ACTIVE_SESSIONS"] = "2"  # Tight limit for concurrency tests
    os.environ["MA_PROGRESS_POLL_INTERVAL_SECONDS"] = "0"
    os.environ["MA_GEOM_LIMIT"] = "0"

    # Use default workflow executor (not event-driven)
    # The stub executor properly intercepts and prevents real Material Agent from running
    # This allows tests to verify service integration without changing material_agent
    os.environ["MA_USE_EVENT_DRIVEN_EXECUTOR"] = "false"

    # Reduce timeouts for testing
    os.environ["MA_SESSION_TTL_HOURS"] = "1"

    return {"sessions": sessions, "materials": mats_path}


@pytest.fixture(scope="session")
def app(_env_for_service):
    """Create and configure FastAPI app.

    Must be called AFTER environment setup to pick up env vars.
    Manually initialize SessionManager since lifespan won't trigger in ASGI testing.
    """
    # Now safe to import - env vars are set
    from pathlib import Path

    from ..service.main import app
    from ..service.routers import (
        artifacts_router,
        assets_router,
        pipeline_router,
        sessions_router,
    )
    from ..service.session.manager import SessionManager

    # Manually call the initialization that would normally happen in lifespan
    session_mgr = SessionManager(
        storage_path=Path(_env_for_service["sessions"]),
        ttl_hours=1,
    )
    # Set session manager for all routers
    pipeline_router.set_session_manager(session_mgr)
    artifacts_router.set_session_manager(session_mgr)
    assets_router.set_session_manager(session_mgr)
    sessions_router.set_session_manager(session_mgr)

    return app


@pytest.fixture
async def client(app: object) -> AsyncIterator[httpx.AsyncClient]:
    """Create AsyncClient for making test requests.

    Uses ASGITransport to test against app without network.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ============================================================================
# STUB EXECUTOR AND CONCURRENCY TRACKING
# ============================================================================


@pytest.fixture(autouse=True, scope="function")
async def _reset_job_registry() -> AsyncGenerator[None, None]:
    """Reset the global JobRegistry between tests.

    The JobRegistry is a singleton that creates asyncio.Semaphore/Lock
    at initialization. These become bound to the event loop at creation time.
    Since pytest-asyncio creates a new event loop per test function, we must
    reset the registry to avoid "bound to a different event loop" errors.
    """
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

    # Reset before test
    await _cancel_lingering_tasks()
    registry_module._job_registry = None

    yield

    # Clean up after test
    await _cancel_lingering_tasks()
    registry_module._job_registry = None


# NOTE: Step executors were deleted - tests now stub the MAA API wrapper instead
# The _stub_executor fixture below handles stubbing at the pipeline_wrapper level


@pytest.fixture(autouse=True, scope="function")
def _stub_input_preview_render(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Avoid real USD preview rendering in API tests.

    Preview rendering is a background convenience path. Unit/API tests exercise
    pipeline and session behavior, and the real renderer can enter USD native
    code after the test has already moved on to other sessions. Tests marked
    with ``@pytest.mark.real_executor`` can opt in to real preview rendering.
    """
    if request.node.get_closest_marker("real_executor"):
        return

    from ..service.routers import pipeline_router

    async def fake_render_input_preview(
        session_id: str,
        session_dir: str | Path,
        original_usd_path: Path | None = None,
    ) -> None:
        input_dir = Path(session_dir) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "input_render.png").write_bytes(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
            b"\xf6\x178U"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )

    monkeypatch.setattr(
        pipeline_router, "_render_input_preview", fake_render_input_preview
    )


@pytest.fixture(autouse=True, scope="function")
def _stub_executor(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> dict[str, Callable[[], int]]:
    """Replace the expensive execute_pipeline_async with a deterministic stub.

    The stub:
    - Respects the REAL global semaphore from JobRegistry
    - Simulates progress through all steps (rendering, predict, apply)
    - Creates minimal but valid artifacts
    - Tracks peak concurrency
    - Allows tests to observe concurrency patterns

    Returns:
        Dict with helper functions for accessing concurrency metrics
    """
    if request.node.get_closest_marker("real_executor"):
        return {
            "max_concurrency_seen": lambda: 0,
            "current_concurrency": lambda: 0,
        }

    # Access the real semaphore and executor module
    # Import here (not at module level) to ensure fresh imports
    from ..service.runtime import get_job_registry
    from ..service.session.manager import SessionManager

    get_job_registry()

    # Track concurrency
    max_seen = {"value": 0}
    current = {"value": 0}
    lock = asyncio.Lock()

    async def _inc():
        """Increment active job count."""
        async with lock:
            current["value"] += 1
            max_seen["value"] = max(max_seen["value"], current["value"])

    async def _dec():
        """Decrement active job count."""
        async with lock:
            current["value"] -= 1

    async def fake_execute(
        session_id: str,
        config_dict: dict,
        session_manager: SessionManager,
        user_email: str = "",
        coverage_policy: str = "allow_partial",
        regeneration_claim: Any | None = None,
    ):
        """Deterministic stub executor.

        Simulates the three pipeline steps with minimal delay.
        Creates artifacts in the right places so download routes work.
        """
        # Note: config_dict is the complete config built in pipeline.py
        manager = session_manager
        # Note: Concurrency is already controlled by JobRegistry semaphore
        # The executor is called AFTER acquiring the semaphore
        await _inc()
        try:
            session_dir = manager.get_session_dir(session_id)

            # Mark session as running
            if regeneration_claim is None:
                await manager.update_session(session_id, {"status": "running"})
            else:
                updated = await manager.update_session_for_claim(
                    session_id,
                    regeneration_claim,
                    {"status": "running"},
                )
                if not updated:
                    raise asyncio.CancelledError("Regeneration claim was superseded")

            # Create cache directories for artifacts
            ds = session_dir / "cache" / "dataset"
            clusters = session_dir / "cache" / "clusters"
            preds = session_dir / "cache" / "predictions"
            out = session_dir / "output"
            ds.mkdir(parents=True, exist_ok=True)
            clusters.mkdir(parents=True, exist_ok=True)
            preds.mkdir(parents=True, exist_ok=True)
            out.mkdir(parents=True, exist_ok=True)

            # ---- STEP 1: Rendering (0-50% overall) ----
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
                # Small delay to simulate work
                delay = float(os.getenv("TEST_STEP_DELAY", "0.01"))
                await asyncio.sleep(delay)

            # Create dataset artifacts
            prims_file = ds / "prims.jsonl"
            dataset_file = ds / "dataset.jsonl"
            usd_dataset_dir = ds / "usd"
            usd_dataset_dir.mkdir(parents=True, exist_ok=True)

            with prims_file.open("w") as f:
                for i in range(10):
                    f.write(json.dumps({"prim_path": f"/p{i}", "type": "Mesh"}) + "\n")
            (usd_dataset_dir / "prims.jsonl").write_text(prims_file.read_text())

            with dataset_file.open("w") as f:
                for i in range(10):
                    f.write(
                        json.dumps(
                            {
                                "id": f"/p{i}",
                                "type": "Mesh",
                                "images": {"composition": f"img_{i}.png"},
                            }
                        )
                        + "\n"
                    )

            # Mark rendering complete (snaps overall to 50%)
            await manager.mark_step_completed(session_id, "build_dataset_usd")

            has_cluster_prims = "cluster_prims" in config_dict.get("steps", {})

            if has_cluster_prims:
                with (clusters / "cluster_map.jsonl").open("w") as f:
                    for i in range(10):
                        cluster_id = i // 2
                        rep_id = f"/p{cluster_id * 2}"
                        f.write(
                            json.dumps(
                                {
                                    "id": f"/p{i}",
                                    "cluster_id": cluster_id,
                                    "is_representative": i % 2 == 0,
                                    "cluster_representative_id": rep_id,
                                    "cluster_size": 2,
                                    "complexity_score": 0.0,
                                    "complexity_tier": "low",
                                }
                            )
                            + "\n"
                        )
                with (clusters / "dataset_representatives.jsonl").open("w") as f:
                    for i in range(0, 10, 2):
                        f.write(
                            json.dumps(
                                {
                                    "id": f"/p{i}",
                                    "type": "Mesh",
                                    "images": {"composition": f"img_{i}.png"},
                                }
                            )
                            + "\n"
                        )
                (clusters / "cluster_summary.json").write_text(
                    json.dumps(
                        {
                            "total_prims": 10,
                            "cluster_count": 5,
                            "representative_count": 5,
                            "reduction_percent": 50.0,
                            "multi_member_count": 5,
                            "singleton_count": 0,
                            "max_cluster_size": 25,
                            "observed_max_cluster_size": 2,
                            "capped_cluster_count": 0,
                            "tiers": {
                                "low": {
                                    "prim_count": 10,
                                    "cluster_count": 5,
                                    "similarity_threshold": 0.99,
                                }
                            },
                            "report_limits": {},
                        }
                    )
                    + "\n"
                )
                cluster_report = (
                    config_dict.get("steps", {})
                    .get("cluster_prims", {})
                    .get("report", {})
                    .get("enabled", True)
                )
                if cluster_report:
                    (clusters / "cluster_report.html").write_text(
                        "<html><body>cluster report</body></html>"
                    )
                await manager.mark_step_completed(session_id, "cluster_prims")

            # ---- STEP 2: Prediction (50-90% overall) ----
            for i, pct in enumerate((60, 75, 90)):
                # Append predictions incrementally
                with (preds / "predictions.jsonl").open("a") as f:
                    material = ["Aluminum", "Copper", "Plastic"][i % 3]
                    f.write(
                        json.dumps(
                            {
                                "id": f"/p{i}",
                                "material": material,
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

            # Mark prediction complete (snaps overall to 90%)
            await manager.mark_step_completed(session_id, "predict")

            apply_enabled = "apply" in config_dict.get("steps", {})
            if apply_enabled:
                # ---- STEP 3: Apply (90-100% overall) ----
                for pct in (95, 100):
                    await manager.update_step_progress(
                        session_id,
                        "apply",
                        {
                            "current": pct,
                            "total": 100,
                            "percent": pct,
                            "message": "Applying materials",
                        },
                    )
                    delay = float(os.getenv("TEST_STEP_DELAY", "0.01"))
                    await asyncio.sleep(delay)

                # Create output USD (minimal but valid)
                (out / "scene_with_materials.usd").write_text("#usda 1.0\n")

                # Mark apply complete (snaps overall to 100%)
                await manager.mark_step_completed(session_id, "apply")

            # Finalize session metadata
            completed_steps = [
                "build_dataset_usd",
                "build_dataset_prepare_dataset",
                "predict",
            ]
            step_outputs = {
                "build_dataset_usd": {
                    "output_dir": str(usd_dataset_dir),
                    "usd_dataset_dir": str(usd_dataset_dir),
                    "num_prims": 10,
                    "num_images": 10,
                },
                "build_dataset_prepare_dataset": {
                    "dataset_path": str(ds),
                    "dataset_jsonl_path": str(dataset_file),
                    "num_entries": 10,
                },
                "predict": {"predictions_path": str(preds / "predictions.jsonl")},
            }
            if apply_enabled:
                completed_steps.append("apply")
                step_outputs["apply"] = {
                    "output_usd_path": str(out / "scene_with_materials.usd")
                }
            if has_cluster_prims:
                completed_steps.insert(2, "cluster_prims")
                step_outputs["cluster_prims"] = {
                    "cluster_map_path": str(clusters / "cluster_map.jsonl"),
                    "dataset_representatives_path": str(
                        clusters / "dataset_representatives.jsonl"
                    ),
                    "cluster_summary_path": str(clusters / "cluster_summary.json"),
                    "cluster_report_path": (
                        str(clusters / "cluster_report.html")
                        if (clusters / "cluster_report.html").exists()
                        else None
                    ),
                }
            (session_dir / "cache" / ".pipeline_state.json").write_text(
                json.dumps(
                    {
                        "completed_steps": completed_steps,
                        "failed_steps": [],
                        "step_errors": {},
                        "step_outputs": step_outputs,
                    }
                )
            )
            artifact_validity = {
                "raw_predictions": True,
                "prediction_report": False,
                "restored_predictions": False,
                "applied_output_usd": apply_enabled,
                "rendered_output_usd": False,
                "final_render": False,
                "cluster_map": has_cluster_prims,
                "cluster_report": has_cluster_prims
                and (clusters / "cluster_report.html").exists(),
                "cluster_summary": has_cluster_prims,
                "cluster_representatives": has_cluster_prims,
                "previews": False,
            }
            terminal_updates = {
                "status": "completed",
                "coverage": None,
                "results": {
                    "prims_processed": 10,
                    "predictions_made": 10,
                    "materials_applied": 3 if apply_enabled else 0,
                    "cluster_prims_ran": has_cluster_prims,
                    "cluster_total_prims": 10 if has_cluster_prims else 0,
                    "cluster_count": 5 if has_cluster_prims else 0,
                    "cluster_representative_count": 5 if has_cluster_prims else 0,
                    "cluster_reduction_percent": 50.0 if has_cluster_prims else 0.0,
                    "cluster_multi_member_count": 5 if has_cluster_prims else 0,
                    "cluster_singleton_count": 0,
                },
                "completed_at": "1970-01-01T00:00:01Z",
                "can_cancel": False,
                "artifact_validity": artifact_validity,
                "timings_breakdown": {
                    "preparation_seconds": 0.1,
                    "rendering_total_seconds": 0.1,
                    "rendering_per_prim_seconds": 0.01,
                    "prediction_total_seconds": 0.1,
                    "prediction_per_prim_seconds": 0.01,
                    "apply_seconds": 0.1,
                    "total_seconds": 0.3,
                },
            }
            if regeneration_claim is None:
                await manager.update_session(session_id, terminal_updates)
            else:
                from ..service.workers.executor import _promote_current_run_artifacts

                artifact_map: dict[str, str] = {}
                await _promote_current_run_artifacts(
                    manager,
                    session_id,
                    session_dir,
                    completed_steps,
                    step_outputs,
                    regeneration_claim=regeneration_claim,
                    artifact_map=artifact_map,
                )
                finalized = await manager.finalize_regeneration_claim(
                    session_id,
                    regeneration_claim,
                    updates=terminal_updates,
                    artifact_map=artifact_map,
                )
                if not finalized:
                    raise asyncio.CancelledError("Regeneration claim was superseded")

        except asyncio.CancelledError:
            # Session was cancelled
            if regeneration_claim is None:
                await manager.update_session(session_id, {"status": "cancelled"})
            else:
                await manager.finalize_regeneration_claim(
                    session_id,
                    regeneration_claim,
                    updates={"status": "cancelled", "can_cancel": False},
                )
            raise

        except Exception as e:
            # Session failed
            failure_updates = {
                "status": "failed",
                "error": str(e),
                "failed_step": "unknown",
            }
            if regeneration_claim is None:
                await manager.update_session(session_id, failure_updates)
            else:
                await manager.finalize_regeneration_claim(
                    session_id,
                    regeneration_claim,
                    updates=failure_updates,
                )
            raise

        finally:
            await _dec()

    # Install stub in place of real executor
    # IMPORTANT: Must patch where it's USED (router), not where it's defined (executor)
    # The router does: from service.workers.executor import execute_pipeline_async
    # So we need to patch the reference in the router's namespace
    from ..service.routers import pipeline_router

    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute, raising=True
    )

    # Return helper to access concurrency metrics
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

    return SessionManager(
        storage_path=Path(_env_for_service["sessions"]),
        ttl_hours=1,
    )


@pytest.fixture
def materials_config_path(_env_for_service):
    """Get path to test materials.yaml."""
    return Path(_env_for_service["materials"])


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
