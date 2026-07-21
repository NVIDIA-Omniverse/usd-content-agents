# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SessionManager progress tracking and lifecycle.

Tests the core contracts for:
- Session creation and metadata management
- Progress math (0-50-90-100% scaling)
- Step completion and timing
- Preview image management
- Local cache cleanup
"""

import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pytest import FixtureRequest

from ...service.session.manager import SessionManager
from ...service.storage.base import METADATA_KEY
from ...service.storage.config import StorageConfig
from ...service.storage.local_store import LocalSessionStore
from ...service.storage.s3_store import S3SessionStore

# S3 tests skipped by default; set RUN_S3_SESSION_MANAGER_TESTS=true and provide
# MA_STORAGE_S3_* env (e.g. a MinIO sidecar) to enable them.
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
async def test_local_store_rejects_nested_session_before_creating_paths(
    tmp_path: str,
) -> None:
    store = LocalSessionStore(tmp_path)

    with pytest.raises(ValueError, match="storage child"):
        await store.init_session("nested/session")
    with pytest.raises(ValueError, match="storage child"):
        await store.put_json("nested/session", METADATA_KEY, {})

    assert not (Path(tmp_path) / "nested").exists()
    assert not (Path(tmp_path) / ".locks" / "nested").exists()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uuid_symlink_session_cannot_expose_outside_metadata(
    tmp_path: str,
) -> None:
    storage_root = Path(tmp_path)
    outside_root = storage_root.parent / f"{storage_root.name}-outside-{uuid4()}"
    outside_root.mkdir()
    sentinel = "outside-material-session-metadata-727"
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
async def test_local_store_reserved_paths_are_invisible_and_writes_fail_loudly(
    tmp_path: str,
) -> None:
    store = LocalSessionStore(tmp_path)
    session_dir = Path(tmp_path) / "s1"
    reserved = session_dir / "cache" / ".pipeline_temp" / "config.json"
    reserved.parent.mkdir(parents=True)
    reserved.write_text('{"api_key": "sentinel"}', encoding="utf-8")
    alias = session_dir / "cache" / "alias.json"
    alias.symlink_to(reserved)

    assert await store.list_keys("s1") == []
    assert await store.list_keys("s1", r"cache\.pipeline_temp") == []
    assert not await store.exists("s1", "cache/.pipeline_temp/config.json")
    assert not await store.exists("s1", "cache/alias.json")
    assert await store.get_json("s1", "cache/.pipeline_temp/config.json") is None
    with pytest.raises(FileNotFoundError):
        await store.open_read("s1", r"cache\.pipeline_temp\config.json")
    with pytest.raises(ValueError, match="reserved"):
        await store.put_bytes("s1", "cache/.pipeline_temp/new.json", b"{}")

    safe_target = session_dir / "cache" / "public.json"
    safe_target.write_bytes(b"public")
    safe_alias = session_dir / "cache" / "public-alias.json"
    safe_alias.symlink_to(safe_target)
    with pytest.raises(FileNotFoundError):
        await store.open_read("s1", "cache/public-alias.json")

    outside = Path(tmp_path) / "outside.json"
    outside.write_bytes(b"outside")
    outside_alias = session_dir / "cache" / "outside-alias.json"
    outside_alias.symlink_to(outside)
    with pytest.raises(FileNotFoundError):
        await store.open_read("s1", "cache/outside-alias.json")


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
        assert (session_dir / "output").exists()
        assert (session_dir / "cache" / "dataset").exists()
        assert (session_dir / "cache" / "predictions").exists()
        assert (session_dir / "preview").exists()

        # Check metadata file
        metadata = await manager.get_session_metadata(session_id)
        assert metadata is not None
        assert metadata["session_id"] == session_id
        assert metadata["status"] == "pending"
        assert metadata["overall_progress"]["percent"] == 0

        response = await manager.sync_session_to_store(session_id)
        # nothing to sync, as session folder is empty
        assert response == 0

    async def test_session_exists_check(self, manager: SessionManager, session_id: str):
        """Test session existence checking."""
        assert not await manager.session_exists(session_id)
        await manager.create_session(session_id)
        assert await manager.session_exists(session_id)

    async def test_delete_session_removes_all_artifacts(
        self, manager: SessionManager, session_id: str
    ):
        """Test that delete_session removes the entire session directory."""
        _ = await manager.create_session(session_id)
        assert await manager.session_exists(session_id)

        await manager.store.put_json(session_id, "test", "test data")

        assert await manager.store.exists(session_id, "test")
        assert await manager.delete_session(session_id)
        assert not await manager.session_exists(session_id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_store_put_file_to_same_path_is_noop(tmp_path: str) -> None:
    """Local mirroring should tolerate artifacts already in the session store."""
    store = LocalSessionStore(tmp_path)
    session_id = str(uuid4())
    await store.init_session(session_id)

    artifact_path = Path(tmp_path) / session_id / "output" / "scene.usd"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("#usda 1.0\n")

    await store.put_file(session_id, "output/scene.usd", str(artifact_path))

    assert artifact_path.read_text() == "#usda 1.0\n"


@pytest.mark.unit
@pytest.mark.asyncio
class TestProgressMath:
    """Test progress scaling: 0-50% (rendering), 50-90% (predict), 90-100% (apply)."""

    async def test_initial_progress_is_zero(
        self, manager: SessionManager, session_id: str
    ):
        """Test that new sessions start at 0% progress."""
        await manager.create_session(session_id)

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 0

    async def test_rendering_step_progress_0_to_50(
        self, manager: SessionManager, session_id: str
    ):
        """Test that rendering step progress ranges from 0-50%."""
        await manager.create_session(session_id)

        # Rendering at 10%
        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 10, "total": 100, "percent": 10, "message": "rendering"},
        )
        assert (await manager.get_session_metadata(session_id))["overall_progress"][
            "percent"
        ] == 10

        # Rendering at 50%
        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 100, "total": 100, "percent": 50, "message": "rendering"},
        )
        assert (await manager.get_session_metadata(session_id))["overall_progress"][
            "percent"
        ] == 50

    async def test_rendering_completion_snaps_to_50(
        self, manager: SessionManager, session_id: str
    ):
        """Test that completing rendering step snaps overall progress to 50%."""
        await manager.create_session(session_id)

        # Update with 10%
        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 10, "total": 100, "percent": 10, "message": "rendering"},
        )

        # Complete rendering
        await manager.mark_step_completed(session_id, "build_dataset_usd")

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 50
        assert metadata["current_step"] is None
        assert len(metadata["completed_steps"]) == 1

    async def test_predict_step_progress_50_to_90(
        self, manager: SessionManager, session_id: str
    ):
        """Test that predict step progress ranges from 50-90%."""
        await manager.create_session(session_id)

        # Complete rendering first (gets to 50%)
        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 100, "total": 100, "percent": 50, "message": "render"},
        )
        await manager.mark_step_completed(session_id, "build_dataset_usd")

        # Predict at 5/10 items -> 50% within step -> scales to 50 + (90-50)*0.5 = 70%
        await manager.update_step_progress(
            session_id,
            "predict",
            {"current": 5, "total": 10, "percent": 60, "message": "predicting"},
        )
        assert (await manager.get_session_metadata(session_id))["overall_progress"][
            "percent"
        ] == 60

        # Predict at 10/10 -> 90%
        await manager.update_step_progress(
            session_id,
            "predict",
            {"current": 10, "total": 10, "percent": 90, "message": "predicting"},
        )
        assert (await manager.get_session_metadata(session_id))["overall_progress"][
            "percent"
        ] == 90

    async def test_predict_completion_snaps_to_90(
        self, manager: SessionManager, session_id: str
    ):
        """Test that completing predict step snaps overall progress to 90%."""
        await manager.create_session(session_id)

        # Complete rendering
        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 100, "total": 100, "percent": 50, "message": "render"},
        )
        await manager.mark_step_completed(session_id, "build_dataset_usd")

        # Update and complete predict
        await manager.update_step_progress(
            session_id,
            "predict",
            {"current": 10, "total": 10, "percent": 75, "message": "pred"},
        )
        await manager.mark_step_completed(session_id, "predict")

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 90
        assert len(metadata["completed_steps"]) == 2

    async def test_apply_step_progress_90_to_100(
        self, manager: SessionManager, session_id: str
    ):
        """Test that apply step progress ranges from 90-100%."""
        await manager.create_session(session_id)

        # Complete rendering and predict
        await manager.update_step_progress(
            session_id,
            "build_dataset_usd",
            {"current": 100, "total": 100, "percent": 50, "message": "render"},
        )
        await manager.mark_step_completed(session_id, "build_dataset_usd")

        await manager.update_step_progress(
            session_id,
            "predict",
            {"current": 10, "total": 10, "percent": 90, "message": "pred"},
        )
        await manager.mark_step_completed(session_id, "predict")

        # Apply at 50%
        await manager.update_step_progress(
            session_id,
            "apply",
            {"current": 5, "total": 10, "percent": 95, "message": "applying"},
        )
        assert (await manager.get_session_metadata(session_id))["overall_progress"][
            "percent"
        ] == 95

        # Apply at 100%
        await manager.update_step_progress(
            session_id,
            "apply",
            {"current": 10, "total": 10, "percent": 100, "message": "applying"},
        )
        assert (await manager.get_session_metadata(session_id))["overall_progress"][
            "percent"
        ] == 100

    async def test_apply_completion_snaps_to_100(
        self, manager: SessionManager, session_id: str
    ):
        """Test that completing apply step snaps overall progress to 100%."""
        await manager.create_session(session_id)

        # Complete all steps
        for step, step_num in [
            ("build_dataset_usd", 1),
            ("predict", 2),
            ("apply", 3),
        ]:
            await manager.update_step_progress(
                session_id,
                step,
                {
                    "current": 100,
                    "total": 100,
                    "percent": 50 + step_num * 10,
                    "message": step,
                },
            )
            await manager.mark_step_completed(session_id, step)

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["overall_progress"]["percent"] == 100
        assert metadata["status"] == "pending"  # Status only changes on completion
        assert len(metadata["completed_steps"]) == 3


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
        assert (await manager.get_session_metadata(session_id))[
            "current_step"
        ] is not None

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

    async def test_metadata_file_created_and_readable(
        self, manager: SessionManager, session_id: str
    ):
        """Test that metadata.json is properly created and readable."""
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

        # Read multiple times
        meta1 = await manager.get_session_metadata(session_id)
        meta2 = await manager.get_session_metadata(session_id)

        assert meta1 == meta2
        assert meta1["current_step"]["progress"]["percent"] == 50


@pytest.mark.unit
@pytest.mark.asyncio
class TestSyncSessionToStore:
    """Test that session can be synced to store."""

    async def test_sync_session_to_store(
        self, manager: SessionManager, session_id: str
    ):
        """Test that session can be synced to store."""
        session_dir = await manager.create_session(session_id)
        materials_yaml_path = session_dir / "materials" / "materials.yaml"
        materials_yaml_path.write_text("test")

        # check that the file exists locally but not on the S3 store
        if manager.store.kind == "s3":
            # s3 store should sync the materials.yaml file
            assert not await manager.store.exists(
                session_id, "materials/materials.yaml"
            )
        else:
            # local store should not sync anything
            assert await manager.store.exists(session_id, "materials/materials.yaml")

        response = await manager.sync_session_to_store(session_id)
        if manager.store.kind == "s3":
            # s3 store should sync the materials.yaml file
            assert response == 1
        else:
            # local store should not sync anything
            assert response == 0

        # sync again, should be nothing to sync
        response = await manager.sync_session_to_store(session_id)
        assert response == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestCleanupStaleLocalCache:
    """Test local cache cleanup functionality."""

    async def test_fresh_session_not_cleaned(
        self, manager: SessionManager, session_id: str
    ):
        """Test that recently updated sessions are not cleaned up."""
        session_dir = await manager.create_session(session_id)

        # Create a test file
        test_file = session_dir / "test_data.txt"
        test_file.write_text("test content")

        # Run cleanup with 24h threshold - fresh session should not be cleaned
        cleaned = await manager.cleanup_stale_local_cache(max_age_hours=24.0)

        assert cleaned == 0
        assert session_dir.exists()
        assert test_file.exists()

    async def test_stale_session_cleaned_from_local(
        self, manager: SessionManager, session_id: str, tmp_path: str
    ):
        """Test that stale sessions are removed from local storage."""
        session_dir = await manager.create_session(session_id)

        # Create some test files
        test_file = session_dir / "test_data.txt"
        test_file.write_text("test content")

        materials_dir = session_dir / "materials"
        materials_dir.mkdir(exist_ok=True)
        (materials_dir / "materials.yaml").write_text("materials: []")

        # Make the session stale by updating metadata with old timestamp
        # Use the store API to update metadata consistently
        metadata = await manager.store.get_json(session_id, METADATA_KEY)
        assert metadata is not None
        old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        metadata["updated_at"] = old_time
        await manager.store.put_json(session_id, METADATA_KEY, metadata)

        # Run cleanup with 24h threshold
        cleaned = await manager.cleanup_stale_local_cache(max_age_hours=24.0)

        if manager.store.kind == "s3":
            # S3 store should have synced and removed local cache
            assert cleaned == 1
            assert not session_dir.exists()

            # Verify data was synced to S3
            assert await manager.store.exists(session_id, "test_data.txt")
            assert await manager.store.exists(session_id, "materials/materials.yaml")
        else:
            # Local store should not cleanup anything (no-op)
            assert cleaned == 0
            assert session_dir.exists()

    async def test_multiple_stale_sessions_cleaned(
        self, manager: SessionManager, tmp_path: str
    ):
        """Test that multiple stale sessions are all cleaned up."""
        session_ids = [str(uuid4()) for _ in range(3)]

        for sid in session_ids:
            session_dir = await manager.create_session(sid)
            (session_dir / "data.txt").write_text(f"data for {sid}")

            # Make session stale using the store API
            metadata = await manager.store.get_json(sid, METADATA_KEY)
            assert metadata is not None
            old_time = (datetime.now(UTC) - timedelta(hours=72)).isoformat()
            metadata["updated_at"] = old_time
            await manager.store.put_json(sid, METADATA_KEY, metadata)

        # Run cleanup
        cleaned = await manager.cleanup_stale_local_cache(max_age_hours=24.0)

        if manager.store.kind == "s3":
            assert cleaned == 3
            for sid in session_ids:
                local_dir = Path(tmp_path) / sid
                assert not local_dir.exists()
                # Data should be on S3
                assert await manager.store.exists(sid, "data.txt")
        else:
            assert cleaned == 0

    async def test_mixed_fresh_and_stale_sessions(
        self, manager: SessionManager, tmp_path: str
    ):
        """Test that only stale sessions are cleaned, fresh ones remain."""
        fresh_id = str(uuid4())
        stale_id = str(uuid4())

        # Create fresh session
        fresh_dir = await manager.create_session(fresh_id)
        (fresh_dir / "fresh_data.txt").write_text("fresh")

        # Create stale session
        stale_dir = await manager.create_session(stale_id)
        (stale_dir / "stale_data.txt").write_text("stale")

        # Make only stale session old using the store API
        metadata = await manager.store.get_json(stale_id, METADATA_KEY)
        assert metadata is not None
        old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        metadata["updated_at"] = old_time
        await manager.store.put_json(stale_id, METADATA_KEY, metadata)

        # Run cleanup
        cleaned = await manager.cleanup_stale_local_cache(max_age_hours=24.0)

        # Fresh session should always remain locally
        assert fresh_dir.exists()
        assert (fresh_dir / "fresh_data.txt").exists()

        if manager.store.kind == "s3":
            assert cleaned == 1
            assert not stale_dir.exists()
            assert await manager.store.exists(stale_id, "stale_data.txt")
        else:
            assert cleaned == 0
            assert stale_dir.exists()

    async def test_cleanup_with_custom_max_age(
        self, manager: SessionManager, session_id: str
    ):
        """Test cleanup with different max_age_hours thresholds."""
        session_dir = await manager.create_session(session_id)
        (session_dir / "data.txt").write_text("test")

        # Make session 6 hours old using the store API
        metadata = await manager.store.get_json(session_id, METADATA_KEY)
        assert metadata is not None
        old_time = (datetime.now(UTC) - timedelta(hours=6)).isoformat()
        metadata["updated_at"] = old_time
        await manager.store.put_json(session_id, METADATA_KEY, metadata)

        # Cleanup with 12h threshold should not clean this session
        cleaned = await manager.cleanup_stale_local_cache(max_age_hours=12.0)
        if manager.store.kind == "s3":
            assert cleaned == 0
            assert session_dir.exists()

        # Cleanup with 4h threshold should clean this session
        cleaned = await manager.cleanup_stale_local_cache(max_age_hours=4.0)
        if manager.store.kind == "s3":
            assert cleaned == 1
            assert not session_dir.exists()
        else:
            assert cleaned == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestCleanupExpiredSessions:
    """Test TTL-based session cleanup return values."""

    async def test_cleanup_expired_sessions_returns_zero_when_nothing_expired(
        self, manager: SessionManager, session_id: str
    ):
        await manager.create_session(session_id)

        cleaned = await manager.cleanup_expired_sessions()

        assert cleaned == 0

    async def test_cleanup_expired_sessions_deletes_expired_local_session(
        self, tmp_path: str, session_id: str
    ):
        manager = SessionManager(tmp_path, store=LocalSessionStore(tmp_path))
        await manager.create_session(session_id)
        metadata = await manager.store.get_json(session_id, METADATA_KEY)
        assert metadata is not None
        metadata["ttl_expires_at"] = (
            datetime.now(UTC) - timedelta(hours=1)
        ).isoformat()
        await manager.store.put_json(session_id, METADATA_KEY, metadata)

        cleaned = await manager.cleanup_expired_sessions()

        assert cleaned == 1
        assert not await manager.session_exists(session_id)


@pytest.mark.unit
@pytest.mark.asyncio
class TestListSessions:
    """Test session listing functionality from the configured storage backend."""

    async def test_list_sessions_empty(self, manager: SessionManager):
        """Test that list_sessions returns empty list when no sessions exist."""
        sessions = await manager.list_sessions()
        assert sessions == []

    async def test_list_sessions_single(self, manager: SessionManager, session_id: str):
        """Test that list_sessions returns a single created session."""
        await manager.create_session(session_id)

        sessions = await manager.list_sessions()
        assert session_id in sessions
        assert len(sessions) >= 1  # May have other sessions in S3

    async def test_list_sessions_multiple(self, manager: SessionManager):
        """Test that list_sessions returns all created sessions."""
        session_ids = [str(uuid4()) for _ in range(3)]

        for sid in session_ids:
            await manager.create_session(sid)

        sessions = await manager.list_sessions()

        # All created sessions should be in the list
        for sid in session_ids:
            assert sid in sessions

    async def test_list_sessions_excludes_deleted(
        self, manager: SessionManager, session_id: str
    ):
        """Test that deleted sessions are not returned by list_sessions."""
        await manager.create_session(session_id)

        # Verify session is listed
        sessions = await manager.list_sessions()
        assert session_id in sessions

        # Delete the session
        await manager.delete_session(session_id)

        # Verify session is no longer listed
        sessions = await manager.list_sessions()
        assert session_id not in sessions

    async def test_list_sessions_from_store_not_local(
        self, manager: SessionManager, session_id: str
    ):
        """Test that S3 store lists sessions from remote, not local directory."""
        await manager.create_session(session_id)

        if manager.store.kind == "s3":
            # For S3 storage, sync to remote then remove local directory
            await manager.sync_session_to_store(session_id)

            # Remove local directory but keep session in S3
            local_dir = manager.get_session_dir(session_id)
            if local_dir.exists():
                import shutil

                shutil.rmtree(local_dir)

            # Session should still be listed (from S3)
            sessions = await manager.list_sessions()
            assert session_id in sessions

            # Verify local directory is gone but session exists in S3
            assert not local_dir.exists()
            assert await manager.session_exists(session_id)
        else:
            # For local storage, listing comes from local directory
            sessions = await manager.list_sessions()
            assert session_id in sessions

    async def test_list_sessions_cache_invalidation(
        self, manager: SessionManager, session_id: str
    ):
        """Test that cache invalidation forces a fresh fetch from S3."""
        if manager.store.kind != "s3":
            pytest.skip("Cache invalidation test only applies to S3 storage")

        await manager.create_session(session_id)

        # First call should cache the result
        sessions1 = await manager.store.list_sessions()
        assert session_id in sessions1

        # Invalidate the cache
        manager.store.invalidate_sessions_cache()

        # Next call should fetch fresh data
        sessions2 = await manager.store.list_sessions()
        assert session_id in sessions2

    async def test_list_sessions_cache_bypass(
        self, manager: SessionManager, session_id: str
    ):
        """Test that use_cache=False bypasses the cache."""
        if manager.store.kind != "s3":
            pytest.skip("Cache bypass test only applies to S3 storage")

        await manager.create_session(session_id)

        # First call with cache enabled
        sessions1 = await manager.store.list_sessions(use_cache=True)
        assert session_id in sessions1

        # Second call with cache bypass
        sessions2 = await manager.store.list_sessions(use_cache=False)
        assert session_id in sessions2

    async def test_list_sessions_cache_auto_update_on_create(
        self, manager: SessionManager
    ):
        """Test that creating a session updates the cache automatically."""
        if manager.store.kind != "s3":
            pytest.skip("Cache auto-update test only applies to S3 storage")

        # Ensure cache is populated
        await manager.store.list_sessions()

        # Create a new session - should update cache
        new_session_id = str(uuid4())
        await manager.create_session(new_session_id)

        # List again (should use cache but include new session)
        sessions = await manager.store.list_sessions(use_cache=True)
        assert new_session_id in sessions

    async def test_list_sessions_cache_auto_update_on_delete(
        self, manager: SessionManager, session_id: str
    ):
        """Test that deleting a session updates the cache automatically."""
        if manager.store.kind != "s3":
            pytest.skip("Cache auto-update test only applies to S3 storage")

        await manager.create_session(session_id)

        # Ensure cache is populated
        sessions_before = await manager.store.list_sessions()
        assert session_id in sessions_before

        # Delete the session - should update cache
        await manager.delete_session(session_id)

        # List again (should use cache but exclude deleted session)
        sessions_after = await manager.store.list_sessions(use_cache=True)
        assert session_id not in sessions_after


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

    Guards against future refactors silently dropping the
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

    async def test_get_session_metadata_batch_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.get_session_metadata_batch([str(uuid4()), self.BAD_ID])

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

    async def test_request_cancellation_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.request_cancellation(self.BAD_ID)

    async def test_sync_session_to_store_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.sync_session_to_store(self.BAD_ID)

    async def test_sync_from_store_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.sync_from_store(self.BAD_ID)

    async def test_exists_in_store_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.exists_in_store(self.BAD_ID, "some/key")

    async def test_read_from_store_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.read_from_store(self.BAD_ID, "some/key")

    async def test_put_bytes_to_store_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.put_bytes_to_store(self.BAD_ID, "some/key", b"x")

    async def test_put_file_to_store_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.put_file_to_store(self.BAD_ID, "some/key", "/tmp/x")

    async def test_make_public_url_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.make_public_url(self.BAD_ID, "some/key")

    async def test_get_artifact_path_rejects(self, tmp_path: str):
        manager = SessionManager(tmp_path)
        with pytest.raises(ValueError, match="Invalid session_id"):
            await manager.get_artifact_path(self.BAD_ID, "predictions")
