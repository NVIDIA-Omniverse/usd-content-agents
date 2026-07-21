# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused session and storage branch coverage for PR 629."""

from __future__ import annotations

import io
from pathlib import Path
from uuid import uuid4

import pytest
from world_understanding.utils.artifacts import OpenArtifactFile

from ...service.session import manager as manager_module
from ...service.session.cache_publications import (
    PREDICTION_REPORT_PUBLICATION_ID_FIELD,
    bound_cache_artifact_key,
    parse_cache_publications,
)
from ...service.session.manager import (
    ACTIVE_RUN_EXPIRES_AT_FIELD,
    ACTIVE_RUN_ID_FIELD,
    JOINT_RIGGER_ARTIFACT_KEYS_RESULT_FIELD,
    JOINT_RIGGER_PUBLICATION_ID_RESULT_FIELD,
    SessionManager,
)
from ...service.storage import base as storage_base_module
from ...service.storage import local_store as local_store_module
from ...service.storage import s3_store as s3_store_module
from ...service.storage.base import METADATA_KEY
from ...service.storage.local_store import LocalSessionStore
from .test_s3_store_additional_coverage import _FakeS3Client, _store

pytestmark = pytest.mark.unit


def _session_id() -> str:
    return str(uuid4())


def test_cache_publication_bindings_reject_malformed_and_support_legacy() -> None:
    relative_path = "cache/dataset/dataset.jsonl"

    assert (
        parse_cache_publications({"cache_publications": {"dataset": "not-a-run-id"}})
        == {}
    )
    assert bound_cache_artifact_key({}, "preview/image.png") is None
    assert bound_cache_artifact_key({}, relative_path) == relative_path
    assert bound_cache_artifact_key({"cache_publications": {}}, relative_path) is None


def test_active_run_claim_parser_rejects_malformed_expirations() -> None:
    run_id = "a" * 32

    assert (
        SessionManager._parse_active_run_claim(
            {
                ACTIVE_RUN_ID_FIELD: run_id,
                ACTIVE_RUN_EXPIRES_AT_FIELD: 123,
            }
        )
        is None
    )
    assert (
        SessionManager._parse_active_run_claim(
            {
                ACTIVE_RUN_ID_FIELD: run_id,
                ACTIVE_RUN_EXPIRES_AT_FIELD: "2026-01-01T00:00:00",
            }
        )
        is None
    )
    assert (
        SessionManager._parse_active_run_claim(
            {
                ACTIVE_RUN_ID_FIELD: run_id,
                ACTIVE_RUN_EXPIRES_AT_FIELD: "not-a-timestamp",
            }
        )
        is None
    )


@pytest.mark.asyncio
async def test_run_operations_reject_invalid_ids_and_missing_sessions(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = _session_id()
    run_id = "a" * 32

    assert not await manager.renew_run(session_id, "invalid")
    assert not await manager.terminalize_and_release_run(
        session_id, "invalid", {"status": "failed"}
    )
    assert not await manager.is_run_current(session_id, "invalid")
    assert not await manager.update_session_for_run(
        session_id, "invalid", {"status": "running"}
    )
    assert not await manager.release_run(session_id, "invalid")

    assert not await manager.reserve_run(session_id, run_id)
    assert not await manager.reserve_legacy_cache_run(session_id, run_id)
    assert not await manager.renew_run(session_id, run_id)
    assert not await manager.terminalize_and_release_run(
        session_id, run_id, {"status": "failed"}
    )
    assert not await manager.is_run_current(session_id, run_id)
    assert not await manager.update_session_for_run(
        session_id, run_id, {"status": "running"}
    )
    assert not await manager.release_run(session_id, run_id)


@pytest.mark.asyncio
async def test_reserve_run_rejects_partial_active_claim(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session_id = _session_id()
    await manager.create_session(session_id)
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    metadata[ACTIVE_RUN_ID_FIELD] = "a" * 32
    metadata[ACTIVE_RUN_EXPIRES_AT_FIELD] = 123
    await manager.store.put_json(session_id, METADATA_KEY, metadata)

    assert not await manager.reserve_run(session_id, "b" * 32)
    unchanged = await manager.get_session_metadata(session_id)
    assert unchanged is not None
    assert unchanged[ACTIVE_RUN_ID_FIELD] == "a" * 32
    assert unchanged[ACTIVE_RUN_EXPIRES_AT_FIELD] == 123


@pytest.mark.asyncio
async def test_step_and_cancellation_guards_preserve_session_state(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    missing_session_id = _session_id()

    assert not await manager.is_cancelled(missing_session_id)

    session_id = _session_id()
    await manager.create_session(session_id)
    assert not await manager.is_cancelled(session_id)
    assert not await manager.is_cancelled(session_id, "invalid")
    assert not await manager.request_cancellation(session_id, "invalid")
    assert not await manager.request_cancellation(session_id)
    assert await manager.store.list_keys(session_id, prefix=".cancel/") == []

    await manager.update_step_progress(session_id, "predict", {"percent": 10})
    await manager.mark_step_completed(session_id, "restore_usd")
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["current_step"]["name"] == "predict"
    assert metadata["completed_steps"] == []

    run_id = "a" * 32
    assert await manager.reserve_run(session_id, run_id)
    assert await manager.request_cancellation(session_id)
    assert await manager.is_cancelled(session_id, run_id)
    cancelled = await manager.get_session_metadata(session_id)
    assert cancelled is not None
    assert cancelled["status"] == "cancelling"


@pytest.mark.asyncio
async def test_immutable_local_artifact_rejects_symlink(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    session_id = _session_id()
    session_dir = await manager.create_session(session_id)
    publication_id = "a" * 32
    artifact_key = f"artifacts/joint_rigger/{publication_id}/rigged.usdz"
    target = tmp_path / "outside.usdz"
    target.write_bytes(b"outside")
    artifact_path = session_dir / artifact_key
    artifact_path.parent.mkdir(parents=True)
    artifact_path.symlink_to(target)
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "results": {
                JOINT_RIGGER_ARTIFACT_KEYS_RESULT_FIELD: {
                    "joint_rigger_output": artifact_key
                },
                JOINT_RIGGER_PUBLICATION_ID_RESULT_FIELD: publication_id,
            },
        },
    )

    assert (
        await manager.get_immutable_local_artifact_path_with_filename(
            session_id, "joint_rigger_output"
        )
        is None
    )
    assert (
        await manager.get_immutable_local_artifact_stream_with_filename(
            session_id, "joint_rigger_output"
        )
        is None
    )
    assert artifact_path.is_symlink()


@pytest.mark.asyncio
async def test_immutable_local_artifact_closes_fd_when_revalidation_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    session_id = _session_id()
    session_dir = await manager.create_session(session_id)
    publication_id = "d" * 32
    artifact_key = f"artifacts/joint_rigger/{publication_id}/rigged.usdz"
    artifact_path = session_dir / artifact_key
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"rigged")
    marker = ("publication", publication_id, "complete")
    lookup_count = 0

    async def failing_revalidation(
        _session_id: str,
        _artifact_type: str,
    ) -> tuple[tuple[str, ...], tuple[str, str, str] | None]:
        nonlocal lookup_count
        lookup_count += 1
        if lookup_count == 1:
            return (artifact_key,), marker
        raise RuntimeError("metadata revalidation failed")

    opened_artifacts: list[OpenArtifactFile] = []
    original_open = manager_module.open_held_confined_artifact

    def tracked_open(root: Path, key: str) -> OpenArtifactFile:
        artifact = original_open(root, key)
        opened_artifacts.append(artifact)
        return artifact

    monkeypatch.setattr(manager, "_artifact_lookup_for_session", failing_revalidation)
    monkeypatch.setattr(manager_module, "open_held_confined_artifact", tracked_open)

    with pytest.raises(RuntimeError, match="metadata revalidation failed"):
        await manager.get_immutable_local_artifact_stream_with_filename(
            session_id,
            "joint_rigger_output",
        )
    assert lookup_count == 2
    assert len(opened_artifacts) == 1
    assert opened_artifacts[0].stream.closed


@pytest.mark.asyncio
async def test_immutable_local_artifact_rejects_pipeline_temp_leaf_alias(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    session_id = _session_id()
    session_dir = await manager.create_session(session_id)
    publication_id = "c" * 32
    artifact_key = f"artifacts/joint_rigger/{publication_id}/rigged.usdz"
    secret = session_dir / "cache" / ".pipeline_temp" / "rigged.usdz"
    secret.parent.mkdir(parents=True)
    secret.write_bytes(b"joint-leaf-secret-sentinel")
    artifact_path = session_dir / artifact_key
    artifact_path.parent.mkdir(parents=True)
    artifact_path.symlink_to(secret)
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "results": {
                JOINT_RIGGER_ARTIFACT_KEYS_RESULT_FIELD: {
                    "joint_rigger_output": artifact_key
                },
                JOINT_RIGGER_PUBLICATION_ID_RESULT_FIELD: publication_id,
            },
        },
    )

    assert (
        await manager.get_immutable_local_artifact_stream_with_filename(
            session_id,
            "joint_rigger_output",
        )
        is None
    )
    assert secret.read_bytes() == b"joint-leaf-secret-sentinel"


@pytest.mark.asyncio
async def test_immutable_local_artifact_rejects_pipeline_temp_ancestor_alias(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    session_id = _session_id()
    session_dir = await manager.create_session(session_id)
    publication_id = "b" * 32
    artifact_key = f"artifacts/joint_rigger/{publication_id}/rigged.usdz"
    secret_dir = session_dir / "cache" / ".pipeline_temp"
    secret_dir.mkdir(parents=True)
    secret = secret_dir / "rigged.usdz"
    secret.write_bytes(b"joint-secret-sentinel")
    artifact_path = session_dir / artifact_key
    artifact_path.parent.parent.mkdir(parents=True)
    artifact_path.parent.symlink_to(secret_dir, target_is_directory=True)
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "results": {
                JOINT_RIGGER_ARTIFACT_KEYS_RESULT_FIELD: {
                    "joint_rigger_output": artifact_key
                },
                JOINT_RIGGER_PUBLICATION_ID_RESULT_FIELD: publication_id,
            },
        },
    )

    assert (
        await manager.get_immutable_local_artifact_stream_with_filename(
            session_id,
            "joint_rigger_output",
        )
        is None
    )
    assert secret.read_bytes() == b"joint-secret-sentinel"


@pytest.mark.asyncio
async def test_artifact_lookup_rejects_missing_and_invalid_publications(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = SessionManager(tmp_path)
    assert await manager.get_artifact_path(_session_id(), "predictions") is None

    session_id = _session_id()
    await manager.create_session(session_id)
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    metadata["status"] = "completed"
    metadata.pop("cache_publications")
    metadata[PREDICTION_REPORT_PUBLICATION_ID_FIELD] = "invalid"
    await manager.store.put_json(session_id, METADATA_KEY, metadata)

    with caplog.at_level("WARNING", logger=manager_module.__name__):
        assert await manager.get_artifact_path(session_id, "prediction_report") is None
    assert "Ignoring invalid prediction report publication" in caplog.text


@pytest.mark.asyncio
async def test_registered_cache_artifacts_always_have_bound_paths(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = _session_id()
    await manager.create_session(session_id)
    run_id = "a" * 32
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "cache_publications": {
                "dataset": run_id,
                "predictions": run_id,
            },
        },
    )

    for artifact_type in manager_module._ARTIFACT_RELATIVE_PATHS:
        if artifact_type in manager_module.JOINT_RIGGER_ARTIFACT_RELATIVE_PATHS:
            continue
        bound_paths, marker = await manager._artifact_lookup_for_session(
            session_id, artifact_type
        )
        assert bound_paths
        assert all(path.startswith("artifacts/run_cache/") for path in bound_paths)
        assert marker is not None
        assert marker[0] == "cache"


@pytest.mark.asyncio
async def test_cache_lookup_fails_closed_when_binding_helper_rejects_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = _session_id()
    await manager.create_session(session_id)
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "cache_publications": {"predictions": "a" * 32},
        },
    )
    monkeypatch.setattr(
        manager_module,
        "bound_cache_artifact_key",
        lambda *_args, **_kwargs: None,
    )

    assert await manager._artifact_lookup_for_session(
        session_id,
        "predictions",
    ) == ((), None)


@pytest.mark.asyncio
async def test_artifact_stream_closes_when_revalidation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = _session_id()
    session_dir = await manager.create_session(session_id)
    artifact_key = "cache/predictions/predictions.jsonl"
    artifact_path = session_dir / artifact_key
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"prediction")
    marker = ("cache", "predictions", "a" * 32)
    lookup_count = 0

    async def failing_revalidation(
        _session_id: str,
        _artifact_type: str,
    ) -> tuple[tuple[str, ...], tuple[str, str, str] | None]:
        nonlocal lookup_count
        lookup_count += 1
        if lookup_count == 1:
            return (artifact_key,), marker
        raise RuntimeError("metadata revalidation failed")

    original_open_read = manager.store.open_read
    opened_streams: list[io.BufferedReader] = []

    async def tracked_open_read(session: str, key: str) -> io.BufferedReader:
        stream = await original_open_read(session, key)
        opened_streams.append(stream)
        return stream

    monkeypatch.setattr(manager, "_artifact_lookup_for_session", failing_revalidation)
    monkeypatch.setattr(manager.store, "open_read", tracked_open_read)

    with pytest.raises(RuntimeError, match="metadata revalidation failed"):
        await manager.get_artifact_stream_with_filename(session_id, "predictions")
    assert lookup_count == 2
    assert len(opened_streams) == 1
    assert opened_streams[0].closed


def test_snapshot_pruning_tolerates_directory_removal_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_dir = tmp_path / "snapshot"
    raced_directory = local_dir / "cache" / "empty"
    raced_directory.mkdir(parents=True)
    real_rmdir = Path.rmdir

    def remove_then_report_missing(path: Path) -> None:
        if path == raced_directory:
            real_rmdir(path)
            raise FileNotFoundError(path)
        real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", remove_then_report_missing)

    storage_base_module._prune_local_snapshot(local_dir, "cache/", set())
    assert not raced_directory.exists()


@pytest.mark.asyncio
async def test_local_compare_and_swap_cleans_temp_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSessionStore(str(tmp_path))
    await store.init_session("s1")
    await store.put_bytes("s1", "claim", b"old")
    real_replace = local_store_module.os.replace
    replace_calls = 0

    def fail_first_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            raise OSError("injected replace failure")
        real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(local_store_module.os, "replace", fail_first_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        await store.compare_and_swap_bytes("s1", "claim", b"old", b"new")
    session_entries = list((tmp_path / "s1").iterdir())
    assert (tmp_path / "s1" / "claim").read_bytes() == b"old"
    assert all(not entry.name.endswith(".tmp") for entry in session_entries)

    assert await store.compare_and_swap_bytes("s1", "claim", b"old", b"new")
    assert (tmp_path / "s1" / "claim").read_bytes() == b"new"


@pytest.mark.asyncio
async def test_s3_conditional_writes_keep_content_types_and_skip_directories(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    store = _store(client)

    assert await store.put_bytes_if_absent(
        "s1", "claim", b"old", content_type="application/claim"
    )
    assert client.uploads[-1][1]["ContentType"] == "application/claim"
    assert await store.compare_and_swap_bytes(
        "s1",
        "claim",
        b"old",
        b"new",
        content_type="application/replacement",
    )
    assert client.uploads[-1][1]["ContentType"] == "application/replacement"

    local_dir = tmp_path / "local"
    nested_file = local_dir / "nested" / "artifact.txt"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("artifact", encoding="utf-8")
    assert await store.sync_from_local("s2", str(local_dir)) == 1
    assert client.objects["wu/sessions/s2/nested/artifact.txt"] == b"artifact"


@pytest.mark.asyncio
async def test_s3_open_read_closes_temp_stream_on_non_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeS3Client()
    client.raise_on_get = RuntimeError("transport failed")
    store = _store(client)
    streams: list[io.BytesIO] = []

    def temporary_file(*, mode: str) -> io.BytesIO:
        assert mode == "w+b"
        stream = io.BytesIO()
        streams.append(stream)
        return stream

    monkeypatch.setattr(s3_store_module.tempfile, "TemporaryFile", temporary_file)

    with pytest.raises(RuntimeError, match="transport failed"):
        await store.open_read("s1", "artifact.bin")
    assert len(streams) == 1
    assert streams[0].closed
