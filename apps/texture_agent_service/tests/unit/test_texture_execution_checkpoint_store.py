# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for durable per-unit Texture Plan execution checkpoints."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from texture_agent.execution import (
    TextureArtifactRef,
    TextureExecutionCheckpoint,
    TextureUnitExecutionRecord,
    TextureUnitExecutionResult,
)

from apps.texture_agent_service.service.runtime.texture_execution import (
    TEXTURE_EXECUTION_CHECKPOINT_KEY,
    SessionTextureExecutionCheckpointStore,
)
from apps.texture_agent_service.service.storage.local_store import LocalSessionStore


def test_session_store_round_trips_execution_checkpoint(tmp_path: Path) -> None:
    session_store = LocalSessionStore(str(tmp_path))
    session_store.init_session("session-1")
    store = SessionTextureExecutionCheckpointStore(
        session_store,
        "session-1",
        tmp_path / "session-1",
    )
    unit_id = "tu_00000000000000000000"
    checkpoint = TextureExecutionCheckpoint(
        plan_schema_version="texture-agent-plan.v1",
        plan_fingerprint="0" * 64,
        selected_unit_ids=(unit_id,),
        records=(TextureUnitExecutionRecord(unit_id=unit_id),),
        created_at=datetime(2026, 6, 29, tzinfo=UTC),
        updated_at=datetime(2026, 6, 29, tzinfo=UTC),
    )

    assert store.load() is None
    store.save(checkpoint)

    assert store.load() == checkpoint
    assert session_store.exists("session-1", TEXTURE_EXECUTION_CHECKPOINT_KEY)


def test_session_store_hydrates_accepted_artifacts_for_replacement_worker(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "store"
    first_worker = tmp_path / "worker-a" / "session-1"
    second_worker = tmp_path / "worker-b" / "session-1"
    session_store = LocalSessionStore(str(storage_root))
    session_store.init_session("session-1")
    source = first_worker / "cache" / "generated" / "albedo.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"accepted-texture")
    unit_id = "tu_00000000000000000000"
    checkpoint = TextureExecutionCheckpoint(
        plan_schema_version="texture-agent-plan.v1",
        plan_fingerprint="0" * 64,
        selected_unit_ids=(unit_id,),
        records=(
            TextureUnitExecutionRecord(
                unit_id=unit_id,
                state="completed",
                attempts=1,
                accepted_result=TextureUnitExecutionResult(
                    unit_id=unit_id,
                    artifacts=(TextureArtifactRef(name="albedo", uri=str(source)),),
                ),
            ),
        ),
    )

    SessionTextureExecutionCheckpointStore(
        session_store,
        "session-1",
        first_worker,
    ).save(checkpoint)
    source.unlink()
    loaded = SessionTextureExecutionCheckpointStore(
        session_store,
        "session-1",
        second_worker,
    ).load()

    assert loaded is not None
    accepted = loaded.records[0].accepted_result
    assert accepted is not None
    hydrated = Path(accepted.artifacts[0].uri)
    assert hydrated.is_file()
    assert hydrated.read_bytes() == b"accepted-texture"


def test_session_store_refuses_artifact_outside_worker_session(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "store"
    worker_session = tmp_path / "worker" / "session-1"
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not-session-data")
    session_store = LocalSessionStore(str(storage_root))
    session_store.init_session("session-1")
    unit_id = "tu_00000000000000000000"
    checkpoint = TextureExecutionCheckpoint(
        plan_schema_version="texture-agent-plan.v1",
        plan_fingerprint="0" * 64,
        selected_unit_ids=(unit_id,),
        records=(
            TextureUnitExecutionRecord(
                unit_id=unit_id,
                state="completed",
                attempts=1,
                accepted_result=TextureUnitExecutionResult(
                    unit_id=unit_id,
                    artifacts=(TextureArtifactRef(name="albedo", uri=str(outside)),),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="outside the session directory"):
        SessionTextureExecutionCheckpointStore(
            session_store,
            "session-1",
            worker_session,
        ).save(checkpoint)


def test_session_store_leaves_remote_and_missing_artifact_uris_unchanged(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "store"
    worker_session = tmp_path / "worker" / "session-1"
    session_store = LocalSessionStore(str(storage_root))
    session_store.init_session("session-1")
    unit_id = "tu_00000000000000000000"
    checkpoint = TextureExecutionCheckpoint(
        plan_schema_version="texture-agent-plan.v1",
        plan_fingerprint="0" * 64,
        selected_unit_ids=(unit_id,),
        records=(
            TextureUnitExecutionRecord(
                unit_id=unit_id,
                state="completed",
                attempts=1,
                accepted_result=TextureUnitExecutionResult(
                    unit_id=unit_id,
                    artifacts=(
                        TextureArtifactRef(
                            name="remote",
                            uri="https://example.invalid/texture.png",
                        ),
                        TextureArtifactRef(
                            name="missing",
                            uri=str(worker_session / "missing.png"),
                        ),
                    ),
                ),
            ),
        ),
    )

    store = SessionTextureExecutionCheckpointStore(
        session_store,
        "session-1",
        worker_session,
    )
    store.save(checkpoint)

    persisted = session_store.get_json("session-1", TEXTURE_EXECUTION_CHECKPOINT_KEY)
    artifacts = persisted["records"][0]["accepted_result"]["artifacts"]
    assert artifacts[0]["uri"] == "https://example.invalid/texture.png"
    assert artifacts[1]["uri"] == str(worker_session / "missing.png")


def test_session_store_leaves_existing_session_artifact_uri_unchanged(
    tmp_path: Path,
) -> None:
    session_store = LocalSessionStore(str(tmp_path / "store"))
    store = SessionTextureExecutionCheckpointStore(
        session_store,
        "session-1",
        tmp_path / "worker" / "session-1",
    )
    artifact = TextureArtifactRef(
        name="albedo",
        uri="session-artifact:///cache/execution/accepted/tu/a.png",
    )

    assert (
        store._persist_artifact(
            "tu_00000000000000000000",
            artifact,
            upload=True,
        )
        == artifact
    )


def test_session_store_leaves_non_session_artifact_uri_localization_unchanged(
    tmp_path: Path,
) -> None:
    session_store = LocalSessionStore(str(tmp_path / "store"))
    store = SessionTextureExecutionCheckpointStore(
        session_store,
        "session-1",
        tmp_path / "worker" / "session-1",
    )
    artifact = TextureArtifactRef(name="remote", uri="https://example.invalid/a.png")

    assert store._localize_artifact(artifact) == artifact


def test_session_store_rejects_invalid_session_artifact_keys(
    tmp_path: Path,
) -> None:
    session_store = LocalSessionStore(str(tmp_path / "store"))
    store = SessionTextureExecutionCheckpointStore(
        session_store,
        "session-1",
        tmp_path / "worker" / "session-1",
    )

    for uri in (
        "session-artifact:///cache/execution/accepted/../escape.png",
        "session-artifact:///cache/execution/other/tu/a.png",
    ):
        with pytest.raises(ValueError, match="session artifact key"):
            store._localize_artifact(TextureArtifactRef(name="albedo", uri=uri))


def test_session_store_rejects_invalid_artifact_name_before_persisting(
    tmp_path: Path,
) -> None:
    session_store = LocalSessionStore(str(tmp_path / "store"))
    session_store.init_session("session-1")
    worker_session = tmp_path / "worker" / "session-1"
    source = worker_session / "cache" / "generated" / "albedo.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"png")
    store = SessionTextureExecutionCheckpointStore(
        session_store,
        "session-1",
        worker_session,
    )
    artifact = TextureArtifactRef.model_construct(
        name="../albedo",
        uri=str(source),
        sha256=None,
    )

    with pytest.raises(ValueError, match="artifact name"):
        store._persist_artifact(
            "tu_00000000000000000000",
            artifact,
            upload=True,
        )


def test_session_store_refuses_localized_symlink_escape(tmp_path: Path) -> None:
    session_store = LocalSessionStore(str(tmp_path / "store"))
    worker_session = tmp_path / "worker" / "session-1"
    outside = tmp_path / "outside"
    outside.mkdir()
    worker_session.mkdir(parents=True)
    (worker_session / "cache").symlink_to(outside, target_is_directory=True)
    store = SessionTextureExecutionCheckpointStore(
        session_store,
        "session-1",
        worker_session,
    )

    with pytest.raises(ValueError, match="outside the session directory"):
        store._localize_artifact(
            TextureArtifactRef(
                name="albedo",
                uri=(
                    "session-artifact:///cache/execution/accepted/"
                    "tu_00000000000000000000/albedo.png"
                ),
            )
        )
