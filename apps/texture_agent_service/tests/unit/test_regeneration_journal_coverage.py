# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused fail-closed coverage for regeneration execution journals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ...service.routers import pipeline_router
from ...service.session.manager import SessionManager


@pytest.mark.parametrize(
    ("metadata", "execution_id", "request_digest", "expected_detail"),
    [
        (
            {
                "execution_id": "a" * 32,
                "execution_request_digest": "b" * 64,
                "execution_request_digests": {"a" * 32: "c" * 64},
            },
            None,
            None,
            "Session execution history conflicts with current metadata",
        ),
        (
            {
                "execution_request_digests": {"a" * 32: "b" * 64},
            },
            "a" * 32,
            "c" * 64,
            "execution_id history has a conflicting request digest",
        ),
    ],
)
def test_reset_rejects_conflicting_execution_journal(
    tmp_path: Path,
    metadata: dict[str, Any],
    execution_id: str | None,
    request_digest: str | None,
    expected_detail: str,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    session_id = "conflicting-execution-journal"
    manager.create_session(session_id)
    manager.update_session(session_id, metadata)

    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._reset_session_for_new_run(
            manager,
            session_id,
            fresh=False,
            execution_id=execution_id,
            execution_request_digest=request_digest,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == expected_detail


def test_execution_request_journal_rejects_malformed_metadata() -> None:
    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._execution_request_journal({"execution_request_digests": []})

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Session execution history is malformed"


def test_execution_match_rejects_journal_digest_mismatch() -> None:
    execution_id = "a" * 32
    request_digest = "b" * 64
    request = pipeline_router.RegenerateRequest.model_validate(
        {
            "steps": ["generate_prompts"],
            "execution_id": execution_id,
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._matches_execution_request(
            request=request,
            request_digest=request_digest,
            metadata={
                "execution_id": execution_id,
                "execution_request_digest": request_digest,
                "execution_request_digests": {execution_id: "c" * 64},
            },
        )

    assert exc_info.value.status_code == 409
    assert (
        exc_info.value.detail
        == "Session execution history conflicts with current metadata"
    )


def test_execution_request_journal_capacity_preserves_replay_and_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_router,
        "_MAX_EXECUTION_REQUEST_JOURNAL_ENTRIES",
        3,
    )
    manager = SessionManager(tmp_path, ttl_hours=2)
    session_id = "bounded-execution-journal"
    manager.create_session(session_id)

    requests: list[pipeline_router.RegenerateRequest] = []
    for index in range(3):
        request = pipeline_router.RegenerateRequest.model_validate(
            {
                "steps": ["generate_prompts"],
                "execution_id": f"{index:032x}",
            }
        )
        requests.append(request)
        pipeline_router._reset_session_for_new_run(
            manager,
            session_id,
            fresh=False,
            execution_id=request.execution_id,
            execution_request_digest=pipeline_router._regenerate_request_digest(
                request
            ),
        )

    metadata = manager.get_session_metadata(session_id)
    assert metadata is not None
    assert list(metadata["execution_request_digests"]) == [
        request.execution_id for request in requests
    ]
    current_request = requests[-1]
    current_digest = pipeline_router._regenerate_request_digest(current_request)
    assert pipeline_router._matches_execution_request(
        request=current_request,
        request_digest=current_digest,
        metadata=metadata,
    )

    with pytest.raises(HTTPException) as superseded_exc:
        pipeline_router._matches_execution_request(
            request=requests[0],
            request_digest=pipeline_router._regenerate_request_digest(requests[0]),
            metadata=metadata,
        )
    assert superseded_exc.value.status_code == 409
    assert (
        superseded_exc.value.detail
        == "execution_id belongs to a superseded regeneration request"
    )

    overflow_request = pipeline_router.RegenerateRequest.model_validate(
        {
            "steps": ["generate_prompts"],
            "execution_id": "f" * 32,
        }
    )
    with pytest.raises(HTTPException) as capacity_exc:
        pipeline_router._reset_session_for_new_run(
            manager,
            session_id,
            fresh=False,
            execution_id=overflow_request.execution_id,
            execution_request_digest=pipeline_router._regenerate_request_digest(
                overflow_request
            ),
        )
    assert capacity_exc.value.status_code == 409
    assert (
        capacity_exc.value.detail
        == "Session execution history reached its safe capacity; start a new session"
    )

    unchanged = manager.get_session_metadata(session_id)
    assert unchanged is not None
    assert unchanged["execution_id"] == current_request.execution_id
    assert (
        unchanged["execution_request_digests"] == metadata["execution_request_digests"]
    )


def test_reset_rollback_logs_event_bus_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    updates: list[tuple[str, dict[str, Any]]] = []

    class _Manager:
        def update_session(
            self,
            session_id: str,
            snapshot: dict[str, Any],
        ) -> None:
            updates.append((session_id, snapshot))

    class _FailingBus:
        def clear_session_state(self, session_id: str) -> None:
            raise RuntimeError(f"cannot clear {session_id}")

    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _FailingBus())

    pipeline_router._restore_session_after_reset_failure(
        _Manager(),  # type: ignore[arg-type]
        "rollback-session",
        {"status": "failed"},
    )

    assert updates == [("rollback-session", {"status": "failed"})]
    assert "Failed to clear pending bus state for rollback-session" in caplog.text


def test_terminal_replay_after_worker_reservation_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replay discovered after reservation must release its worker lock."""
    manager = SessionManager(tmp_path, ttl_hours=2)
    session_id = "terminal-replay-after-reservation"
    manager.create_session(session_id)
    execution_id = "d" * 32
    request_payload = {
        "steps": ["generate_prompts"],
        "execution_id": execution_id,
    }
    request = pipeline_router.RegenerateRequest.model_validate(request_payload)
    request_digest = pipeline_router._regenerate_request_digest(request)
    manager.update_session(
        session_id,
        {
            "status": "completed",
            "execution_id": execution_id,
            "execution_request_digest": request_digest,
        },
    )

    real_get_metadata = manager.get_session_metadata
    metadata_reads = 0

    def metadata_visible_after_reservation(
        requested_session_id: str,
    ) -> dict[str, Any] | None:
        nonlocal metadata_reads
        metadata_reads += 1
        metadata = real_get_metadata(requested_session_id)
        assert metadata is not None
        if metadata_reads == 1:
            metadata = dict(metadata)
            metadata["execution_id"] = None
            metadata["execution_request_digest"] = None
        return metadata

    class _IdleRegistry:
        def is_running(self, requested_session_id: str) -> bool:
            assert requested_session_id == session_id
            return False

    monkeypatch.setattr(
        manager,
        "get_session_metadata",
        metadata_visible_after_reservation,
    )
    monkeypatch.setattr(
        pipeline_router,
        "get_job_registry",
        lambda: _IdleRegistry(),
    )
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json=request_payload,
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "completed"
    assert response.json()["message"] == "Regeneration execution already accepted"
    assert metadata_reads == 2
    assert manager.is_worker_active(session_id) is False
