# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional branch coverage for SessionManager."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from ...service.session.manager import CANCEL_KEY, SessionManager
from ...service.storage.base import METADATA_KEY
from ...service.storage.local_store import LocalSessionStore


class _ListingStore(LocalSessionStore):
    def __init__(
        self, root_dir: str, sessions: list[str], *, kind: str = "local"
    ) -> None:
        super().__init__(root_dir)
        self._sessions = sessions
        self._kind = kind

    @property
    def kind(self) -> str:
        return self._kind

    async def list_sessions(self, use_cache: bool = True) -> list[str]:
        return list(self._sessions)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_manager_missing_metadata_paths(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    sid = str(uuid4())

    await manager.update_session(sid, {"status": "running"})
    await manager.update_step_progress(sid, "predict", {"percent": 1})
    await manager.mark_step_completed(sid, "predict")
    await manager.add_preview_image(sid, "preview.png")
    await manager.update_preview_images(sid, ["preview.png"])
    await manager.add_generated_reference_image(sid, {"id": "ref"})
    assert await manager.remove_generated_reference_image(sid, "ref") is None
    await manager.request_cancellation(sid)
    assert not await manager.store.exists(sid, CANCEL_KEY)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_manager_preview_reference_and_artifact_branches(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    sid = str(uuid4())
    session_dir = await manager.create_session(sid)

    metadata = await manager.get_session_metadata(sid)
    metadata.pop("preview_images")
    await manager.store.put_json(sid, METADATA_KEY, metadata)
    await manager.add_preview_image(sid, "preview.png")
    await manager.add_preview_image(sid, "preview.png")
    metadata = await manager.get_session_metadata(sid)
    assert metadata["preview_images"] == ["preview.png"]

    await manager.update_session(sid, {"status": "ready"})
    await manager.add_generated_reference_image(sid, {"id": "ref-1", "name": "A"})
    assert await manager.remove_generated_reference_image(sid, "missing") is None
    removed = await manager.remove_generated_reference_image(sid, "ref-1")
    assert removed == {"id": "ref-1", "name": "A"}

    assert await manager.get_artifact_path(sid, "missing") is None
    output = session_dir / "output" / "scene_with_materials.usd"
    output.write_text("#usda 1.0\n")
    assert await manager.get_artifact_path(sid, "output_usd") == output


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_manager_completed_steps_fallbacks(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    sid = str(uuid4())
    await manager.create_session(sid)

    metadata = await manager.get_session_metadata(sid)
    metadata.pop("completed_steps")
    metadata.pop("timings", None)
    metadata["current_step"] = {
        "name": "extra",
        "display_name": "Extra Step",
        "started_at": (datetime.now(UTC) - timedelta(seconds=5)).isoformat(),
        "progress": {"percent": 100},
        "elapsed_seconds": 0,
    }
    await manager.store.put_json(sid, METADATA_KEY, metadata)
    await manager.mark_step_completed(sid, "extra")

    metadata = await manager.get_session_metadata(sid)
    assert metadata["overall_progress"]["percent"] == 50

    for index in range(3):
        metadata["current_step"] = {
            "name": f"extra-{index}",
            "display_name": f"Extra {index}",
            "started_at": datetime.now(UTC).isoformat(),
            "progress": {"percent": 100},
            "elapsed_seconds": 0,
        }
        await manager.store.put_json(sid, METADATA_KEY, metadata)
        await manager.mark_step_completed(sid, f"extra-{index}")
        metadata = await manager.get_session_metadata(sid)

    assert metadata["overall_progress"]["percent"] == 100


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_manager_store_helpers_and_sync_logging(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    sid = str(uuid4())
    await manager.create_session(sid)

    source = tmp_path / "source.txt"
    source.write_text("hello")
    await manager.put_file_to_store(sid, "input/source.txt", str(source), "text/plain")
    await manager.put_bytes_to_store(sid, "input/raw.bin", b"raw", "application/bin")
    assert await manager.exists_in_store(sid, "input/raw.bin") is True
    assert await manager.read_from_store(sid, "input/raw.bin") == b"raw"
    assert await manager.read_from_store(sid, "input/missing.bin") is None
    assert await manager.make_public_url(sid, "input/raw.bin") is None

    class BrokenOpenStore(LocalSessionStore):
        async def open_read(self, session_id: str, key: str):
            raise RuntimeError("broken")

    broken = SessionManager(
        tmp_path / "broken", store=BrokenOpenStore(str(tmp_path / "broken"))
    )
    await broken.create_session(sid)
    await broken.store.put_bytes(sid, "raw.bin", b"raw")
    assert await broken.read_from_store(sid, "raw.bin") is None

    class SyncingStore(LocalSessionStore):
        async def sync_from_local(
            self, session_id: str, local_session_dir: str, prefix: str = ""
        ) -> int:
            return 2

        async def sync_to_local(
            self, session_id: str, local_session_dir: str, prefix: str = ""
        ) -> int:
            Path(local_session_dir).mkdir(parents=True, exist_ok=True)
            return 3

    syncing = SessionManager(
        tmp_path / "syncing", store=SyncingStore(str(tmp_path / "syncing"))
    )
    await syncing.create_session(sid)
    assert await syncing.sync_session_to_store(sid) == 2
    assert await syncing.sync_from_store(sid) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_manager_cleanup_skip_paths(tmp_path: Path) -> None:
    sid = str(uuid4())
    no_metadata = SessionManager(
        tmp_path / "no-metadata",
        store=_ListingStore(str(tmp_path / "no-metadata"), [sid]),
    )
    assert await no_metadata.cleanup_expired_sessions() == 0

    non_local = SessionManager(
        tmp_path / "non-local",
        ttl_hours=0,
        store=_ListingStore(str(tmp_path / "non-local"), [sid], kind="remote"),
    )
    await non_local.create_session(sid)
    assert await non_local.cleanup_expired_sessions() == 0
