# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SessionManager progress tracking and lifecycle.

Tests the core contracts for:
- Session creation and metadata management
- Progress math for the 0.5 seven-step pipeline
- Step completion and timing
- Preview image management

All tests are parameterized across local, S3 (MinIO), and default storage backends.
"""

import os
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from pytest import FixtureRequest

from ...service.session import manager as session_manager_module
from ...service.session.manager import SessionManager
from ...service.storage.config import StorageConfig
from ...service.storage.local_store import LocalSessionStore
from ...service.storage.s3_store import S3SessionStore

# S3 tests require a running MinIO instance; opt-in via RUN_S3_SESSION_MANAGER_TESTS=true
SKIP_S3_TESTS = os.getenv("RUN_S3_SESSION_MANAGER_TESTS", "").lower() != "true"


@pytest.fixture(params=["local", "s3", None])
def storage_type(request: FixtureRequest):
    return request.param


@pytest.fixture
def tmp_path() -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def storage_prefix() -> str:
    return str(uuid4())


@pytest.fixture
def manager(storage_type: str, tmp_path: str, storage_prefix: str):
    if storage_type == "local":
        return SessionManager(tmp_path, store=LocalSessionStore(tmp_path))
    elif storage_type == "s3":
        if SKIP_S3_TESTS:
            pytest.skip("S3 tests are skipped")
        else:
            return SessionManager(
                tmp_path,
                store=S3SessionStore.from_config(
                    StorageConfig(kind="s3", s3_prefix=storage_prefix)
                ),
            )
    else:
        return SessionManager(tmp_path)


@pytest.fixture
def session_id() -> str:
    return str(uuid4())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_sessions_filters_untrusted_backend_names(tmp_path: str) -> None:
    store = LocalSessionStore(tmp_path)
    manager = SessionManager(tmp_path, store=store)
    safe_id = str(uuid4())
    for backend_name in (safe_id, ".pipeline_temp", "access_token=do-not-return"):
        backend_dir = Path(tmp_path) / backend_name
        backend_dir.mkdir()
        (backend_dir / "session.json").write_text("{}", encoding="utf-8")

    assert await manager.list_sessions() == [safe_id]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uuid_symlink_session_cannot_expose_outside_metadata(
    tmp_path: str,
) -> None:
    storage_root = Path(tmp_path)
    outside_root = storage_root.parent / f"{storage_root.name}-outside-{uuid4()}"
    outside_root.mkdir()
    sentinel = "outside-joint-session-metadata-727"
    (outside_root / "session.json").write_text(
        '{"credential":"' + sentinel + '"}',
        encoding="utf-8",
    )
    session_id = str(uuid4())
    (storage_root / session_id).symlink_to(outside_root, target_is_directory=True)
    store = LocalSessionStore(tmp_path)
    manager = SessionManager(tmp_path, store=store)

    assert session_id not in await store.list_sessions()
    assert session_id not in await manager.list_sessions()
    assert not await store.exists(session_id, "session.json")
    assert await store.get_json(session_id, "session.json") is None
    assert await manager.get_session_metadata(session_id) is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_session_rejects_swapped_session_ancestor(
    tmp_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = Path(tmp_path)
    manager = SessionManager(storage_root, store=LocalSessionStore(storage_root))
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    (session_dir / "owned.txt").write_text("owned", encoding="utf-8")
    detached = storage_root / f"{session_id}.held"
    outside = storage_root.parent / f"outside-joint-delete-{uuid4()}"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("outside-delete-sentinel", encoding="utf-8")
    sentinel_mode = sentinel.stat().st_mode

    async def keep_local_session(_session_id: str) -> None:
        return None

    async def no_sleep(_delay: float) -> None:
        return None

    original_remove = session_manager_module.remove_confined_tree
    swapped = False

    def swap_then_remove(working_dir: Path, allowed_root: Path) -> bool:
        nonlocal swapped
        if not swapped:
            swapped = True
            session_dir.rename(detached)
            session_dir.symlink_to(outside, target_is_directory=True)
        return original_remove(working_dir, allowed_root)

    monkeypatch.setattr(manager.store, "delete_session", keep_local_session)
    monkeypatch.setattr(session_manager_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        session_manager_module,
        "remove_confined_tree",
        swap_then_remove,
    )

    assert await manager.delete_session(session_id)
    assert sentinel.read_text(encoding="utf-8") == "outside-delete-sentinel"
    assert sentinel.stat().st_mode == sentinel_mode
    assert (detached / "owned.txt").read_text(encoding="utf-8") == "owned"


@pytest.mark.unit
@pytest.mark.asyncio
class TestSessionManagerLifecycle:
    """Test basic session lifecycle operations."""

    async def test_create_session_initializes_metadata(
        self, manager: SessionManager, session_id: str
    ):
        """Test that create_session initializes proper directory structure and metadata."""
        session_dir = await manager.create_session(session_id)

        assert session_dir.exists()
        assert (session_dir / "input").exists()
        assert (session_dir / "cache" / "dataset").exists()
        assert (session_dir / "cache" / "predictions").exists()
        assert (session_dir / "preview").exists()

        metadata = await manager.get_session_metadata(session_id)
        assert metadata is not None
        assert metadata["session_id"] == session_id
        assert metadata["status"] == "pending"
        assert metadata["overall_progress"]["percent"] == 0
        assert metadata["overall_progress"]["total_steps"] == 7

    async def test_session_exists_check(self, manager: SessionManager, session_id: str):
        """Test session existence checking."""
        assert not await manager.session_exists(session_id)
        await manager.create_session(session_id)
        assert await manager.session_exists(session_id)

    async def test_get_session_dir(self, manager: SessionManager, session_id: str):
        """Test retrieving session directory path."""
        await manager.create_session(session_id)
        session_dir = manager.get_session_dir(session_id)

        assert session_dir.exists()

    async def test_delete_session_removes_all_artifacts(
        self, manager: SessionManager, session_id: str
    ):
        """Test that delete_session removes the entire session directory."""
        session_dir = await manager.create_session(session_id)
        (session_dir / "test.txt").write_text("test data")

        assert await manager.session_exists(session_id)

        success = await manager.delete_session(session_id)
        assert success
        assert not await manager.session_exists(session_id)


@pytest.mark.unit
@pytest.mark.asyncio
class TestProgressMath:
    """Test weighted progress scaling for the service pipeline."""

    async def test_initial_progress_is_zero(
        self, manager: SessionManager, session_id: str
    ):
        """Test that new sessions start at 0% progress."""
        await manager.create_session(session_id)

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 0

    async def test_rendering_step_progress_25_to_60(
        self, manager: SessionManager, session_id: str
    ):
        """Test that rendering step progress maps into the 25-60% range."""
        await manager.create_session(session_id)

        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 10, "total": 100, "percent": 10, "message": "rendering"},
        )
        meta = await manager.get_session_metadata(session_id)
        assert meta["overall_progress"]["percent"] == 28

        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 100, "total": 100, "percent": 100, "message": "rendering"},
        )
        meta = await manager.get_session_metadata(session_id)
        assert meta["overall_progress"]["percent"] == 60

    async def test_rendering_completion_snaps_to_60(
        self, manager: SessionManager, session_id: str
    ):
        """Test that completing rendering step snaps overall progress to 60%."""
        await manager.create_session(session_id)

        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 10, "total": 100, "percent": 10, "message": "rendering"},
        )

        await manager.mark_step_completed(session_id, "build_dataset_usd")

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 60
        assert metadata["current_step"] is None
        assert len(metadata["completed_steps"]) == 1

    async def test_prepare_dataset_completion_snaps_to_70(
        self, manager: SessionManager, session_id: str
    ):
        """Test that completing prepare_dataset snaps overall progress to 70%."""
        await manager.create_session(session_id)

        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 100, "total": 100, "percent": 50, "message": "render"},
        )
        await manager.mark_step_completed(session_id, "build_dataset_usd")

        await manager.update_step_progress(
            session_id,
            "build_dataset_prepare_dataset",
            {"current": 10, "total": 10, "percent": 50, "message": "preparing"},
        )
        await manager.mark_step_completed(session_id, "build_dataset_prepare_dataset")

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 70
        assert len(metadata["completed_steps"]) == 2

    async def test_predict_step_progress_70_to_88(
        self, manager: SessionManager, session_id: str
    ):
        """Test that predict step progress maps into the 70-88% range."""
        await manager.create_session(session_id)

        # Complete rendering and prepare_dataset
        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 100, "total": 100, "percent": 50, "message": "render"},
        )
        await manager.mark_step_completed(session_id, "build_dataset_usd")

        await manager.update_step_progress(
            session_id,
            "build_dataset_prepare_dataset",
            {"current": 10, "total": 10, "percent": 50, "message": "prepare"},
        )
        await manager.mark_step_completed(session_id, "build_dataset_prepare_dataset")

        await manager.update_step_progress(
            session_id,
            "predict",
            {"current": 5, "total": 10, "percent": 75, "message": "predicting"},
        )
        meta = await manager.get_session_metadata(session_id)
        assert meta["overall_progress"]["percent"] == 83

        await manager.update_step_progress(
            session_id,
            "predict",
            {"current": 10, "total": 10, "percent": 100, "message": "predicting"},
        )
        meta = await manager.get_session_metadata(session_id)
        assert meta["overall_progress"]["percent"] == 88

    async def test_predict_completion_snaps_to_88(
        self, manager: SessionManager, session_id: str
    ):
        """Test that completing predict step snaps overall progress to 88%."""
        await manager.create_session(session_id)

        for step in ["build_dataset_usd", "build_dataset_prepare_dataset", "predict"]:
            await manager.update_step_progress(
                session_id,
                step,
                {"current": 100, "total": 100, "percent": 50, "message": step},
            )
            await manager.mark_step_completed(session_id, step)

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 88
        assert metadata["status"] == "pending"  # Status only changes on completion
        assert len(metadata["completed_steps"]) == 3

    async def test_consistency_candidates_and_restore_complete_final_progress(
        self, manager: SessionManager, session_id: str
    ):
        """Test late-stage percentages stay below done until final."""
        await manager.create_session(session_id)

        for step in [
            "identify_asset",
            "analyze_structure",
            "build_dataset_usd",
            "build_dataset_prepare_dataset",
            "predict",
            "consistency_pass",
        ]:
            await manager.update_step_progress(
                session_id,
                step,
                {"current": 1, "total": 1, "percent": 100, "message": step},
            )
            await manager.mark_step_completed(session_id, step)

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 92
        assert metadata["current_step"] is None
        assert metadata["completed_steps"][-1]["name"] == "consistency_pass"

        await manager.update_step_progress(
            session_id,
            "infer_articulation_candidates",
            {"current": 1, "total": 2, "percent": 50, "message": "infer"},
        )
        metadata = await manager.get_session_metadata(session_id)
        assert metadata["current_step"]["display_name"] == (
            "Inferring Articulation Candidates"
        )
        assert metadata["overall_progress"]["current_step"] == 7
        assert metadata["overall_progress"]["percent"] == 94

        await manager.mark_step_completed(session_id, "infer_articulation_candidates")
        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 96
        assert metadata["overall_progress"]["total_steps"] == 7

        await manager.update_step_progress(
            session_id,
            "restore_usd",
            {"current": 1, "total": 2, "percent": 50, "message": "restore"},
        )
        metadata = await manager.get_session_metadata(session_id)
        assert metadata["current_step"]["display_name"] == "Restoring Predictions"
        assert metadata["overall_progress"]["total_steps"] == 8
        assert metadata["overall_progress"]["current_step"] == 8
        assert metadata["overall_progress"]["percent"] == 97

        await manager.mark_step_completed(session_id, "restore_usd")
        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 98
        assert metadata["overall_progress"]["total_steps"] == 8
        assert metadata["overall_progress"]["current_step"] == 8

        await manager.update_session(session_id, {"status": "completed"})
        metadata = await manager.get_session_metadata(session_id)
        assert metadata["status"] == "completed"
        assert metadata["overall_progress"]["percent"] == 100
        assert metadata["overall_progress"]["total_steps"] == 8
        assert metadata["overall_progress"]["current_step"] == 8

    async def test_joint_rigger_progress_extends_after_restore(
        self, manager: SessionManager, session_id: str
    ):
        """Test the opt-in Joint Rigger step uses the final progress slot."""
        await manager.create_session(session_id)

        for step in [
            "identify_asset",
            "analyze_structure",
            "build_dataset_usd",
            "build_dataset_prepare_dataset",
            "predict",
            "consistency_pass",
            "infer_articulation_candidates",
            "restore_usd",
        ]:
            await manager.update_step_progress(
                session_id,
                step,
                {"current": 1, "total": 1, "percent": 100, "message": step},
            )
            await manager.mark_step_completed(session_id, step)

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 98
        assert metadata["overall_progress"]["total_steps"] == 8

        await manager.update_step_progress(
            session_id,
            "apply_joint_rigger",
            {"current": 1, "total": 2, "percent": 50, "message": "rigger"},
        )
        metadata = await manager.get_session_metadata(session_id)
        assert metadata["current_step"]["display_name"] == "Applying Joint Rigger"
        assert metadata["overall_progress"]["current_step"] == 9
        assert metadata["overall_progress"]["total_steps"] == 9
        assert metadata["overall_progress"]["percent"] == 98

        await manager.mark_step_completed(session_id, "apply_joint_rigger")
        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 99
        assert metadata["overall_progress"]["current_step"] == 9
        assert metadata["overall_progress"]["total_steps"] == 9

        await manager.update_session(session_id, {"status": "completed"})
        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 100
        assert metadata["overall_progress"]["current_step"] == 9
        assert metadata["overall_progress"]["total_steps"] == 9

    async def test_completed_session_normalizes_progress_to_100(
        self, manager: SessionManager, session_id: str
    ):
        """Test completed status reports 100% with the configured total."""
        await manager.create_session(session_id)

        for step in [
            "identify_asset",
            "analyze_structure",
            "build_dataset_usd",
            "build_dataset_prepare_dataset",
            "predict",
            "consistency_pass",
            "infer_articulation_candidates",
        ]:
            await manager.update_step_progress(
                session_id,
                step,
                {"current": 1, "total": 1, "percent": 100, "message": step},
            )
            await manager.mark_step_completed(session_id, step)

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 96

        await manager.update_session(session_id, {"status": "completed"})

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["status"] == "completed"
        assert metadata["overall_progress"]["percent"] == 100
        assert metadata["overall_progress"]["current_step"] == 7


@pytest.mark.unit
@pytest.mark.asyncio
class TestStepTracking:
    """Test current step tracking and step completion."""

    async def test_current_step_updates_on_progress(
        self, manager: SessionManager, session_id: str
    ):
        """Test that current_step field updates during progress."""
        await manager.create_session(session_id)

        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 10, "total": 100, "percent": 10, "message": "rendering"},
        )

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["current_step"]["name"] == "build_dataset_usd"
        assert metadata["current_step"]["display_name"] == "Rendering USD Scene"
        assert metadata["current_step"]["progress"]["percent"] == 10

    async def test_step_completion_moves_to_completed_steps(
        self, manager: SessionManager, session_id: str
    ):
        """Test that mark_step_completed moves step from current to completed."""
        await manager.create_session(session_id)

        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 100, "total": 100, "percent": 50, "message": "render"},
        )
        meta = await manager.get_session_metadata(session_id)
        assert meta["current_step"] is not None

        await manager.mark_step_completed(
            session_id, "build_dataset_usd", stats={"prims_rendered": 100}
        )

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["current_step"] is None
        assert len(metadata["completed_steps"]) == 1
        assert metadata["completed_steps"][0]["name"] == "build_dataset_usd"
        assert metadata["completed_steps"][0]["stats"]["prims_rendered"] == 100

    async def test_step_timing_recorded(self, manager: SessionManager, session_id: str):
        """Test that step duration is recorded on completion."""
        await manager.create_session(session_id)

        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 10, "total": 100, "percent": 10, "message": "render"},
        )

        await manager.mark_step_completed(session_id, "build_dataset_usd")

        metadata = await manager.get_session_metadata(session_id)
        completed_step = metadata["completed_steps"][0]
        assert "started_at" in completed_step
        assert "completed_at" in completed_step
        assert completed_step["duration_seconds"] >= 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestPreviewImages:
    """Test preview image management."""

    async def test_add_preview_image(self, manager: SessionManager, session_id: str):
        """Test adding a single preview image."""
        await manager.create_session(session_id)

        await manager.add_preview_image(session_id, "preview_0.png")

        metadata = await manager.get_session_metadata(session_id)
        assert "preview_0.png" in metadata["preview_images"]

    async def test_add_multiple_preview_images(
        self, manager: SessionManager, session_id: str
    ):
        """Test adding multiple preview images."""
        await manager.create_session(session_id)

        for i in range(5):
            await manager.add_preview_image(session_id, f"preview_{i}.png")

        metadata = await manager.get_session_metadata(session_id)
        assert len(metadata["preview_images"]) == 5

    async def test_preview_images_not_duplicated(
        self, manager: SessionManager, session_id: str
    ):
        """Test that duplicate preview images are not added."""
        await manager.create_session(session_id)

        await manager.add_preview_image(session_id, "preview_0.png")
        await manager.add_preview_image(session_id, "preview_0.png")

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["preview_images"].count("preview_0.png") == 1

    async def test_update_preview_images_replaces_list(
        self, manager: SessionManager, session_id: str
    ):
        """Test that update_preview_images replaces the entire list."""
        await manager.create_session(session_id)

        await manager.add_preview_image(session_id, "old_0.png")
        await manager.update_preview_images(session_id, ["new_0.png", "new_1.png"])

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["preview_images"] == ["new_0.png", "new_1.png"]


@pytest.mark.unit
@pytest.mark.asyncio
class TestMetadataAtomicity:
    """Test that metadata updates are atomic and corruption-resistant."""

    async def test_metadata_created_and_readable(
        self, manager: SessionManager, session_id: str
    ):
        """Test that session.json is properly created and readable."""
        await manager.create_session(session_id)

        metadata = await manager.get_session_metadata(session_id)
        assert metadata is not None
        assert metadata["session_id"] == session_id

    async def test_metadata_persists_across_reads(
        self, manager: SessionManager, session_id: str
    ):
        """Test that metadata persists and is consistent across multiple reads."""
        await manager.create_session(session_id)

        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 50, "total": 100, "percent": 50, "message": "halfway"},
        )

        meta1 = await manager.get_session_metadata(session_id)
        meta2 = await manager.get_session_metadata(session_id)

        assert meta1 == meta2
        assert meta1["current_step"]["progress"]["percent"] == 50


@pytest.mark.unit
@pytest.mark.asyncio
class TestSyncToStore:
    """Test syncing local artifacts to the store."""

    async def test_sync_to_store(self, manager: SessionManager, session_id: str):
        """Test that local artifacts can be synced to the store."""
        session_dir = await manager.create_session(session_id)

        # Create a local artifact
        preds_dir = session_dir / "cache" / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        (preds_dir / "predictions.jsonl").write_text('{"id": "/Root"}\n')

        if manager.store.kind == "s3":
            # S3: file should not exist on remote until synced
            assert not await manager.store.exists(
                session_id, "cache/predictions/predictions.jsonl"
            )

        synced = await manager.sync_to_store(session_id)

        if manager.store.kind == "s3":
            assert synced >= 1
            assert await manager.store.exists(
                session_id, "cache/predictions/predictions.jsonl"
            )
        else:
            # Local store sync is a no-op
            assert synced == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestListSessions:
    """Test session listing from the configured storage backend."""

    async def test_list_sessions_empty(self, manager: SessionManager):
        """Test that list_sessions returns empty list when no sessions exist."""
        sessions = await manager.list_sessions()
        assert sessions == []

    async def test_list_sessions_single(self, manager: SessionManager, session_id: str):
        """Test that list_sessions returns a single created session."""
        await manager.create_session(session_id)

        sessions = await manager.list_sessions()
        assert session_id in sessions

    async def test_list_sessions_multiple(self, manager: SessionManager):
        """Test that list_sessions returns all created sessions."""
        session_ids = [str(uuid4()) for _ in range(3)]

        for sid in session_ids:
            await manager.create_session(sid)

        sessions = await manager.list_sessions()
        for sid in session_ids:
            assert sid in sessions

    async def test_list_sessions_excludes_deleted(
        self, manager: SessionManager, session_id: str
    ):
        """Test that deleted sessions are not returned by list_sessions."""
        await manager.create_session(session_id)
        assert session_id in await manager.list_sessions()

        await manager.delete_session(session_id)
        assert session_id not in await manager.list_sessions()

    async def test_list_sessions_from_store_not_local(
        self, manager: SessionManager, session_id: str
    ):
        """Test that S3 store lists sessions from remote, not local directory."""
        await manager.create_session(session_id)

        if manager.store.kind == "s3":
            # Sync to remote then remove local directory
            await manager.sync_to_store(session_id)

            local_dir = manager.get_session_dir(session_id)
            if local_dir.exists():
                import shutil

                shutil.rmtree(local_dir)

            # Session should still be listed (from S3)
            sessions = await manager.list_sessions()
            assert session_id in sessions
            assert not local_dir.exists()
            assert await manager.session_exists(session_id)
        else:
            sessions = await manager.list_sessions()
            assert session_id in sessions


@pytest.mark.unit
@pytest.mark.asyncio
class TestCancellation:
    """Test cross-instance cancellation via store."""

    async def test_cancellation_signal(self, manager: SessionManager, session_id: str):
        """Test that cancellation signal is persisted to the store."""
        await manager.create_session(session_id)
        run_id = "a" * 32
        assert await manager.reserve_run(session_id, run_id)
        assert await manager.update_session_for_run(
            session_id,
            run_id,
            {"status": "running"},
        )

        assert not await manager.is_cancelled(session_id)

        assert await manager.request_cancellation(session_id, run_id)

        assert await manager.is_cancelled(session_id)

        # Metadata should reflect cancelling status
        meta = await manager.get_session_metadata(session_id)
        assert meta["status"] == "cancelling"

    async def test_cancellation_visible_cross_instance(
        self, manager: SessionManager, session_id: str, tmp_path: str
    ):
        """Test that cancellation from one manager is visible to another."""
        await manager.create_session(session_id)
        run_id = "a" * 32
        assert await manager.reserve_run(session_id, run_id)
        assert await manager.update_session_for_run(
            session_id,
            run_id,
            {"status": "running"},
        )

        # Create a second manager pointing to the same store
        manager2 = SessionManager(
            storage_path=tmp_path + "_pod2",
            ttl_hours=1,
            store=manager.store,
        )

        assert await manager.request_cancellation(session_id, run_id)
        assert await manager2.is_cancelled(session_id)


@pytest.mark.unit
class TestSessionIdValidation:
    """Verify get_session_dir rejects traversal / non-UUID session IDs."""

    @pytest.mark.parametrize(
        "bad_id",
        [
            "../etc/passwd",
            "/etc/passwd",
            "..",
            ".",
            "",
            "a" * 36,
            "not-a-uuid",
            "00000000-0000-0000-0000-00000000000",  # one char short
            "00000000-0000-0000-0000-000000000000/../boom",
        ],
    )
    def test_rejects_non_uuid_session_id(self, tmp_path: str, bad_id: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            manager.get_session_dir(bad_id)

    def test_accepts_valid_uuid(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        sid = str(uuid4())
        assert manager.get_session_dir(sid) == Path(tmp_path) / sid

    def test_accepts_uppercase_uuid(self, tmp_path: str):
        """Regex must tolerate case variance — clients may echo UUIDs uppercased."""
        manager = SessionManager(tmp_path)
        sid = str(uuid4()).upper()
        assert manager.get_session_dir(sid) == Path(tmp_path) / sid


@pytest.mark.unit
@pytest.mark.asyncio
class TestSessionIdValidationEveryMethod:
    """Every public SessionManager method that takes session_id rejects bad input.

    This guards against future refactors silently dropping the
    `_validate_session_id` line from a single method.
    """

    BAD_ID = "../etc/passwd"

    async def test_create_session_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.create_session(self.BAD_ID)

    async def test_session_exists_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.session_exists(self.BAD_ID)

    async def test_get_session_metadata_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.get_session_metadata(self.BAD_ID)

    async def test_update_session_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.update_session(self.BAD_ID, {"status": "running"})

    async def test_delete_session_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.delete_session(self.BAD_ID)

    async def test_is_cancelled_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.is_cancelled(self.BAD_ID)

    async def test_sync_to_store_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.sync_to_store(self.BAD_ID)

    async def test_sync_from_store_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.sync_from_store(self.BAD_ID)

    async def test_get_artifact_path_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.get_artifact_path(self.BAD_ID, "predictions")

    async def test_get_artifact_stream_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.get_artifact_stream(self.BAD_ID, "predictions")
