# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bounded mock texture workflow skeleton."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Event, Lock
from typing import Any
from zipfile import ZipFile

import pytest
import requests
from filelock import FileLock

import content_agent_workflows.texture.client as texture_client_module
import content_agent_workflows.texture.decision as texture_decision_module
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.texture import (
    CanonicalTextureWorkflowFinalizer,
    MockTexturePlannerExecutorClient,
    MockTextureSceneValidator,
    TextureAgentServiceCancellationRequested,
    TextureAgentServiceClient,
    TextureDecisionLedger,
    TextureDecisionPatch,
    TextureExecutionResult,
    TextureFinalizationResult,
    TextureFinalizerInput,
    TextureGeneratorInputs,
    TexturePlanDecision,
    TexturePlanDocument,
    TextureReferenceArtifact,
    TextureStepObservation,
    TextureUnitArtifact,
    TextureValidationFinding,
    TextureValidationResult,
    TextureWorkflowCheckpointStore,
    TextureWorkflowProgress,
    TextureWorkflowRequest,
    TextureWorkflowRuntimeError,
    TextureWorkflowValidationEvidence,
    build_texture_step_observation,
    record_texture_decision_patch,
    run_batch_texture_workflow,
    run_interactive_texture_workflow,
    run_texture_workflow_step,
    texture_source_identity_digest,
    verify_texture_resume_decision_state,
)
from content_agent_workflows.texture.workflow import (
    _require_execution_scope,
    _require_plan_request_scope,
    _require_validation_scope,
)


class _FakeResponse:
    def __init__(
        self,
        payload: object = None,
        *,
        content: bytes = b"",
        status_code: int = 200,
    ) -> None:
        self._payload = payload
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        response = requests.Response()
        response.status_code = self.status_code
        raise requests.HTTPError(response=response)

    def json(self) -> object:
        return self._payload


class _FakeTextureServiceSession:
    def __init__(
        self,
        plan: dict[str, object],
        texture_zip: bytes,
        *,
        status_responses: list[_FakeResponse] | None = None,
    ) -> None:
        self.headers: dict[str, str] = {}
        self.plan = plan
        self.texture_zip = texture_zip
        self.status_responses = list(status_responses or ())
        self.status_request_count = 0
        self.status_request_timeouts: list[object] = []
        self.posts: list[tuple[str, dict[str, object]]] = []
        self.output_content = texture_zip
        self.run_revision = 0
        self.run_statuses: list[_FakeResponse] = []
        self.execution_id: str | None = None
        self.requests_by_execution_id: dict[str, dict[str, object]] = {}
        self.uploaded_files: dict[str, bytes] = {}

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.posts.append((url, kwargs))
        files = kwargs.get("files")
        if isinstance(files, Mapping):
            for field_name, raw_spec in files.items():
                if not isinstance(raw_spec, tuple) or len(raw_spec) < 2:
                    continue
                stream = raw_spec[1]
                position = stream.tell()
                self.uploaded_files[str(field_name)] = stream.read()
                stream.seek(position)
        if url.endswith("/pipeline"):
            return _FakeResponse({"session_id": "session-466"})
        if url.endswith("/regenerate"):
            raw_payload = kwargs["json"]
            assert isinstance(raw_payload, Mapping)
            payload = dict(raw_payload)
            execution_id = str(payload.pop("execution_id"))
            previous_payload = self.requests_by_execution_id.get(execution_id)
            if previous_payload is not None:
                assert previous_payload == payload
                return _FakeResponse({"session_id": "session-466", "status": "pending"})
            self.requests_by_execution_id[execution_id] = payload
            self.execution_id = execution_id
            self.run_revision += 1
            self.run_statuses = [
                _FakeResponse(
                    {
                        "status": "pending",
                        "updated_at": f"revision-{self.run_revision}-pending",
                        "execution_id": self.execution_id,
                    }
                ),
                _FakeResponse(
                    {
                        "status": "completed",
                        "updated_at": f"revision-{self.run_revision}",
                        "execution_id": self.execution_id,
                    }
                ),
            ]
            return _FakeResponse({"session_id": "session-466", "status": "pending"})
        raise AssertionError(url)

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        if url.endswith("/status"):
            self.status_request_count += 1
            self.status_request_timeouts.append(kwargs["timeout"])
            if self.status_responses:
                return self.status_responses.pop(0)
            if self.run_statuses:
                return self.run_statuses.pop(0)
            return _FakeResponse(
                {
                    "status": "completed",
                    "updated_at": f"revision-{self.run_revision}",
                    "execution_id": self.execution_id,
                }
            )
        if url.endswith("/plan"):
            return _FakeResponse(self.plan)
        if url.endswith("/results"):
            return _FakeResponse({"stats": {"cache_hit_unit_ids": []}})
        if url.endswith("/textures"):
            return _FakeResponse(content=self.texture_zip)
        if url.endswith("/output"):
            return _FakeResponse(content=self.output_content)
        raise AssertionError(url)


class _CrashAfterAcceptedRegenerateSession(_FakeTextureServiceSession):
    """Model a process dying after the service accepts one regenerate request."""

    def __init__(
        self,
        plan: dict[str, object],
        texture_zip: bytes,
        *,
        crash_on_regenerate_call: int,
    ) -> None:
        super().__init__(plan, texture_zip)
        self.crash_on_regenerate_call = crash_on_regenerate_call
        self.regenerate_call_count = 0
        self.remote_revision = 0
        self.requests_by_execution_id: dict[str, dict[str, object]] = {}

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        if not url.endswith("/regenerate"):
            return super().post(url, **kwargs)
        self.posts.append((url, kwargs))
        raw_payload = kwargs["json"]
        assert isinstance(raw_payload, Mapping)
        payload = dict(raw_payload)
        execution_id = str(payload.pop("execution_id"))
        previous_payload = self.requests_by_execution_id.get(execution_id)
        if previous_payload is not None:
            if previous_payload != payload:
                return _FakeResponse(
                    {"detail": "execution_id payload conflict"},
                    status_code=409,
                )
            return _FakeResponse({"session_id": "session-466", "status": "completed"})
        self.requests_by_execution_id[execution_id] = payload
        self.execution_id = execution_id
        self.regenerate_call_count += 1
        self.remote_revision += 1
        if self.regenerate_call_count == self.crash_on_regenerate_call:
            raise RuntimeError("simulated process interruption after remote acceptance")
        return _FakeResponse({"session_id": "session-466", "status": "pending"})

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        if not url.endswith("/status"):
            return super().get(url, **kwargs)
        self.status_request_count += 1
        self.status_request_timeouts.append(kwargs["timeout"])
        return _FakeResponse(
            {
                "status": "completed",
                "updated_at": f"revision-{self.remote_revision}",
                "completed_steps": [{"name": f"remote-run-{self.remote_revision}"}],
                "execution_id": self.execution_id,
            }
        )


class _LaggingAcceptedRegenerateSession(_FakeTextureServiceSession):
    """Return one stale terminal status after each accepted regenerate."""

    def __init__(self, plan: dict[str, object], texture_zip: bytes) -> None:
        super().__init__(plan, texture_zip)
        self.regenerate_call_count = 0
        self.remote_revision = 0
        self.visible_revision = 0
        self.visible_execution_id: str | None = None
        self.phase_statuses: list[dict[str, object]] = []
        self.status_history: list[tuple[int, str, str]] = []
        self.artifact_gets: list[str] = []
        self.requests_by_execution_id: dict[str, dict[str, object]] = {}

    @staticmethod
    def _completed_status(
        revision: int,
        execution_id: str | None,
    ) -> dict[str, object]:
        return {
            "status": "completed",
            "updated_at": f"revision-{revision}",
            "completed_steps": [{"name": f"remote-run-{revision}"}],
            "execution_id": execution_id,
        }

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        if not url.endswith("/regenerate"):
            return super().post(url, **kwargs)
        self.posts.append((url, kwargs))
        raw_payload = kwargs["json"]
        assert isinstance(raw_payload, Mapping)
        payload = dict(raw_payload)
        execution_id = str(payload.pop("execution_id"))
        previous_payload = self.requests_by_execution_id.get(execution_id)
        if previous_payload is not None:
            assert previous_payload == payload
            return _FakeResponse({"session_id": "session-466", "status": "pending"})
        self.requests_by_execution_id[execution_id] = payload
        self.execution_id = execution_id
        self.regenerate_call_count += 1
        baseline_revision = self.visible_revision
        self.remote_revision += 1
        new_revision = self.remote_revision
        self.phase_statuses = [
            self._completed_status(
                baseline_revision,
                self.visible_execution_id,
            ),
            {
                "status": "pending",
                "updated_at": f"revision-{new_revision}-pending",
                "completed_steps": [],
                "execution_id": self.execution_id,
            },
            self._completed_status(new_revision, self.execution_id),
        ]
        return _FakeResponse({"session_id": "session-466", "status": "pending"})

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        if url.endswith("/status"):
            self.status_request_count += 1
            self.status_request_timeouts.append(kwargs["timeout"])
            if self.phase_statuses:
                status = self.phase_statuses.pop(0)
                if (
                    status["status"] == "completed"
                    and status["updated_at"] == f"revision-{self.remote_revision}"
                ):
                    self.visible_revision = self.remote_revision
                    self.visible_execution_id = self.execution_id
            else:
                status = self._completed_status(
                    self.visible_revision,
                    self.visible_execution_id,
                )
            self.status_history.append(
                (
                    self.regenerate_call_count,
                    str(status["status"]),
                    str(status["updated_at"]),
                )
            )
            return _FakeResponse(status)
        if (
            url.endswith("/results")
            or url.endswith("/textures")
            or url.endswith("/output")
        ):
            if self.visible_revision != 2:
                raise AssertionError(
                    "Texture artifacts were fetched before apply completed"
                )
            self.artifact_gets.append(url)
        return super().get(url, **kwargs)


class _IdempotentCrashWithStaleStatusSession(_FakeTextureServiceSession):
    """Replay one accepted phase while status still shows its old baseline."""

    def __init__(self, plan: dict[str, object], texture_zip: bytes) -> None:
        super().__init__(plan, texture_zip)
        self.current_status: dict[str, object] = {
            "status": "completed",
            "updated_at": "revision-0",
            "execution_id": None,
        }
        self.stale_status: dict[str, object] | None = None
        self.requests_by_execution_id: dict[str, dict[str, object]] = {}
        self.backend_registrations_by_execution_id: dict[str, int] = {}
        self._crash_after_first_accept = True

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        if not url.endswith("/regenerate"):
            return super().post(url, **kwargs)
        self.posts.append((url, kwargs))
        raw_payload = kwargs["json"]
        assert isinstance(raw_payload, Mapping)
        payload = dict(raw_payload)
        execution_id = str(payload.pop("execution_id"))
        previous_payload = self.requests_by_execution_id.get(execution_id)
        if previous_payload is not None:
            assert previous_payload == payload
            return _FakeResponse({"session_id": "session-466", "status": "completed"})

        self.requests_by_execution_id[execution_id] = payload
        self.backend_registrations_by_execution_id[execution_id] = 1
        prior_status = dict(self.current_status)
        revision = len(self.requests_by_execution_id)
        self.current_status = {
            "status": "completed",
            "updated_at": f"revision-{revision}",
            "execution_id": execution_id,
        }
        self.execution_id = execution_id
        if self._crash_after_first_accept:
            self._crash_after_first_accept = False
            self.stale_status = prior_status
            raise RuntimeError("simulated crash after accepted generation")
        return _FakeResponse({"session_id": "session-466", "status": "pending"})

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        if url.endswith("/status"):
            self.status_request_count += 1
            self.status_request_timeouts.append(kwargs["timeout"])
            if self.stale_status is not None:
                status = self.stale_status
                self.stale_status = None
                return _FakeResponse(status)
            return _FakeResponse(dict(self.current_status))
        return super().get(url, **kwargs)


class _OwnedOrphanRegenerateSession(_FakeTextureServiceSession):
    """Keep the first accepted phase pending until its exact request is replayed."""

    def __init__(self, plan: dict[str, object], texture_zip: bytes) -> None:
        super().__init__(plan, texture_zip)
        self.current_status: dict[str, object] = {
            "status": "completed",
            "updated_at": "revision-0",
            "execution_id": None,
        }
        self.status_queue: list[dict[str, object]] = []
        self.requests_by_execution_id: dict[str, dict[str, object]] = {}
        self.replays_by_execution_id: dict[str, int] = {}
        self._orphan_first_execution = True

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        if not url.endswith("/regenerate"):
            return super().post(url, **kwargs)
        self.posts.append((url, kwargs))
        raw_payload = kwargs["json"]
        assert isinstance(raw_payload, Mapping)
        payload = dict(raw_payload)
        execution_id = str(payload.pop("execution_id"))
        previous_payload = self.requests_by_execution_id.get(execution_id)
        if previous_payload is not None:
            assert previous_payload == payload
            self.replays_by_execution_id[execution_id] = (
                self.replays_by_execution_id.get(execution_id, 0) + 1
            )
            self.status_queue = [
                dict(self.current_status),
                {
                    "status": "completed",
                    "updated_at": f"revision-{self.run_revision}",
                    "execution_id": execution_id,
                },
            ]
            return _FakeResponse({"session_id": "session-466", "status": "pending"})

        self.requests_by_execution_id[execution_id] = payload
        self.run_revision += 1
        self.execution_id = execution_id
        self.current_status = {
            "status": "pending",
            "updated_at": f"revision-{self.run_revision}-pending",
            "execution_id": execution_id,
        }
        if self._orphan_first_execution:
            self._orphan_first_execution = False
        else:
            self.status_queue = [
                dict(self.current_status),
                {
                    "status": "completed",
                    "updated_at": f"revision-{self.run_revision}",
                    "execution_id": execution_id,
                },
            ]
        return _FakeResponse({"session_id": "session-466", "status": "pending"})

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        if url.endswith("/status"):
            self.status_request_count += 1
            self.status_request_timeouts.append(kwargs["timeout"])
            if self.status_queue:
                self.current_status = self.status_queue.pop(0)
            return _FakeResponse(dict(self.current_status))
        return super().get(url, **kwargs)


class _CollectPhaseOrphanSession(_FakeTextureServiceSession):
    """Lose the final worker after its terminal bus snapshot was checkpointed."""

    def __init__(self, plan: dict[str, object], texture_zip: bytes) -> None:
        super().__init__(plan, texture_zip)
        self.restart_status: dict[str, object] | None = None
        self.collect_replay_count = 0

    def restart_with_owned_orphan(self) -> None:
        assert self.execution_id is not None
        self.restart_status = {
            "status": "pending",
            "updated_at": "restart-pending",
            "execution_id": self.execution_id,
        }

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        if not url.endswith("/regenerate") or self.restart_status is None:
            return super().post(url, **kwargs)
        self.posts.append((url, kwargs))
        raw_payload = kwargs["json"]
        assert isinstance(raw_payload, Mapping)
        payload = dict(raw_payload)
        execution_id = str(payload.pop("execution_id"))
        assert execution_id == self.execution_id
        assert self.requests_by_execution_id[execution_id] == payload
        self.collect_replay_count += 1
        self.restart_status = {
            "status": "completed",
            "updated_at": "restart-completed",
            "execution_id": execution_id,
        }
        return _FakeResponse({"session_id": "session-466", "status": "pending"})

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        if url.endswith("/status") and self.restart_status is not None:
            self.status_request_count += 1
            self.status_request_timeouts.append(kwargs["timeout"])
            return _FakeResponse(dict(self.restart_status))
        return super().get(url, **kwargs)


class _TransientCollectReplayFailureSession(_CollectPhaseOrphanSession):
    """Reject the first collect-phase orphan replay with a transient 503."""

    def __init__(self, plan: dict[str, object], texture_zip: bytes) -> None:
        super().__init__(plan, texture_zip)
        self.fail_next_collect_replay = True

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        if (
            url.endswith("/regenerate")
            and self.restart_status is not None
            and self.fail_next_collect_replay
        ):
            self.fail_next_collect_replay = False
            self.posts.append((url, kwargs))
            return _FakeResponse(
                {"detail": "transient replay failure"},
                status_code=503,
            )
        return super().post(url, **kwargs)


class _RecordingArtifactGetsSession(_FakeTextureServiceSession):
    def __init__(self, plan: dict[str, object], texture_zip: bytes) -> None:
        super().__init__(plan, texture_zip)
        self.artifact_gets: list[str] = []

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        if (
            url.endswith("/results")
            or url.endswith("/textures")
            or url.endswith("/output")
        ):
            self.artifact_gets.append(url)
        return super().get(url, **kwargs)


class _ConflictingExecutionTokenSession(_FakeTextureServiceSession):
    """Return a definitive token-conflict response that must not be adopted."""

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        if not url.endswith("/regenerate"):
            return super().post(url, **kwargs)
        self.posts.append((url, kwargs))
        payload = kwargs["json"]
        assert isinstance(payload, Mapping)
        self.execution_id = str(payload["execution_id"])
        self.run_revision += 1
        return _FakeResponse(
            {"detail": "execution_id payload conflict"},
            status_code=409,
        )


def _texture_zip(
    unit_ids: tuple[str, ...],
    *,
    payload: bytes = b"png",
) -> bytes:
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for unit_id in unit_ids:
            archive.writestr(f"{unit_id}_albedo.png", payload)
            archive.writestr(f"textures/{unit_id}/albedo.png", payload)
    return stream.getvalue()


def _owned_regenerate_payloads(
    session: _FakeTextureServiceSession,
) -> tuple[list[dict[str, object]], set[str]]:
    payloads: list[dict[str, object]] = []
    execution_ids: set[str] = set()
    for url, kwargs in session.posts:
        if not url.endswith("/regenerate"):
            continue
        raw_payload = kwargs["json"]
        assert isinstance(raw_payload, Mapping)
        payload = dict(raw_payload)
        execution_id = payload.pop("execution_id")
        assert isinstance(execution_id, str)
        assert len(execution_id) == 32
        execution_ids.add(execution_id)
        payloads.append(payload)
    return payloads, execution_ids


class _RecordingFinalizer:
    def __init__(self) -> None:
        self.inputs: list[TextureFinalizerInput] = []
        self._delegate = CanonicalTextureWorkflowFinalizer()

    def finalize(self, payload: TextureFinalizerInput) -> TextureFinalizationResult:
        self.inputs.append(payload)
        return self._delegate.finalize(payload)


class _WrongScopeValidator:
    def validate(
        self,
        *,
        request: TextureWorkflowRequest,
        plan: object,
        output_asset_path: str,
        unit_artifacts: Mapping[str, TextureUnitArtifact],
        unit_ids: tuple[str, ...],
        iteration: int,
        output_dir: Path,
    ) -> TextureValidationResult:
        del request, plan, unit_artifacts, output_dir
        first_id = unit_ids[0]
        return TextureValidationResult(
            iteration=iteration,
            evaluated_unit_ids=(first_id,),
            findings=(
                TextureValidationFinding(
                    unit_id=first_id,
                    status="pass",
                    summary="Deliberately incomplete validator fixture.",
                    evidence_artifact_paths=("/mock/incomplete.json",),
                ),
            ),
            output_asset_path=output_asset_path,
        )


class _StaleCandidateValidator(MockTextureSceneValidator):
    def __init__(self, stale_field: str) -> None:
        super().__init__()
        self._stale_field = stale_field

    def validate(self, **kwargs: Any) -> TextureValidationResult:
        result = super().validate(**kwargs)
        if self._stale_field == "iteration":
            return result.model_copy(update={"iteration": result.iteration + 1})
        stale_path = str(
            Path(result.output_asset_path).with_name("stale-textured-output.usda")
        )
        return result.model_copy(update={"output_asset_path": stale_path})


def _request(tmp_path: Path, *, max_vqa_iterations: int = 2) -> TextureWorkflowRequest:
    return TextureWorkflowRequest(
        source_asset="/assets/board.usda",
        output_dir=tmp_path,
        intent="Generate surface-only board textures.",
        max_vqa_iterations=max_vqa_iterations,
    )


def _decision_for(observation: TextureStepObservation) -> TextureDecisionPatch:
    assert observation.action != "done"
    return TextureDecisionPatch(
        request_digest=observation.request_digest,
        source_identity_digest=observation.source_identity_digest,
        plan_digest=observation.plan_digest,
        checkpoint_decision_digest=observation.checkpoint_decision_digest,
        checkpoint_revision=observation.checkpoint_revision,
        action=observation.action,
        iteration=observation.iteration,
        target_unit_ids=observation.target_unit_ids,
        operations=observation.required_operations,
        evidence_sha256_by_path=observation.evidence_sha256_by_path,
        rationale=f"Reviewed evidence and approved {observation.action}.",
        confidence=0.9,
    )


def test_real_service_adapter_uses_plan_only_full_then_targeted_regeneration(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text('#usda 1.0\n\ndef Xform "Board" {}\n', encoding="utf-8")
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
        metadata={
            "auto_prompt_enabled": True,
            "discovery_mode": "explicit",
            "uv_policy": "validate",
            "uv_scope": "target_prims",
            "explicit_material_paths": ["/World/Looks/Paint"],
            "explicit_prim_paths": ["/World/Board"],
        },
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=2).plan(request)
    fake_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )

    plan = client.plan(request)
    result = client.execute(
        plan,
        plan.selected_unit_ids,
        output_dir=request.output_dir,
        preserved_artifacts={},
    )

    plan_post = fake_http.posts[0]
    assert plan_post[0] == "http://texture.test/pipeline"
    assert plan_post[1]["data"]["plan_only"] == "true"
    assert plan_post[1]["data"]["discovery_mode"] == "explicit"
    assert plan_post[1]["data"]["uv_policy"] == "validate"
    assert plan_post[1]["data"]["uv_scope"] == "target_prims"
    assert json.loads(plan_post[1]["data"]["explicit_material_paths_json"]) == [
        "/World/Looks/Paint"
    ]
    assert json.loads(plan_post[1]["data"]["explicit_prim_paths_json"]) == [
        "/World/Board"
    ]
    regeneration_post = fake_http.posts[1]
    assert regeneration_post[1]["json"]["steps"] == [
        "prepare_uvs",
        "generate_textures",
        "blend_textures",
    ]
    execution_id = regeneration_post[1]["json"]["execution_id"]
    assert isinstance(execution_id, str)
    assert len(execution_id) == 32
    assert "texture_unit_ids" not in regeneration_post[1]["json"]
    apply_post = fake_http.posts[2]
    apply_execution_id = apply_post[1]["json"]["execution_id"]
    assert apply_post[1]["json"] == {
        "steps": ["apply_textures"],
        "execution_id": apply_execution_id,
    }
    assert apply_execution_id != execution_id

    targeted_id = plan.selected_unit_ids[0]
    preserved_id = plan.selected_unit_ids[1]
    targeted_result = client.execute(
        plan,
        (targeted_id,),
        output_dir=request.output_dir,
        preserved_artifacts={
            preserved_id: next(
                artifact
                for artifact in result.unit_artifacts
                if artifact.unit_id == preserved_id
            )
        },
    )
    targeted_post = fake_http.posts[3]
    assert targeted_post[1]["json"]["steps"] == [
        "prepare_uvs",
        "generate_textures",
        "blend_textures",
        "apply_textures",
    ]
    assert targeted_post[1]["json"]["texture_unit_ids"] == [targeted_id]
    targeted_execution_id = targeted_post[1]["json"]["execution_id"]
    assert targeted_execution_id != execution_id
    assert targeted_result.requested_unit_ids == (targeted_id,)
    assert result.requested_unit_ids == plan.selected_unit_ids
    assert Path(result.output_asset_path).parent.name == apply_execution_id
    assert Path(targeted_result.output_asset_path).parent.name == targeted_execution_id
    assert [artifact.unit_id for artifact in result.unit_artifacts] == list(
        plan.selected_unit_ids
    )
    assert Path(result.output_asset_path).read_bytes() == fake_http.output_content
    assert all(
        any(unit_id in Path(path).parts for path in artifact.artifact_paths)
        for unit_id, artifact in zip(
            plan.selected_unit_ids, result.unit_artifacts, strict=True
        )
    )

    preserved_artifact = next(
        artifact
        for artifact in result.unit_artifacts
        if artifact.unit_id == preserved_id
    )
    preserved_bytes = tuple(
        Path(path).read_bytes() for path in preserved_artifact.artifact_paths
    )
    first_output_path = Path(result.output_asset_path)
    first_output_bytes = first_output_path.read_bytes()
    fake_http.output_content = _texture_zip(
        plan.selected_unit_ids,
        payload=b"changed-package",
    )
    with pytest.raises(
        RuntimeError,
        match="output package changed preserved artifacts",
    ):
        client.execute(
            plan,
            (targeted_id,),
            output_dir=request.output_dir,
            preserved_artifacts={preserved_id: preserved_artifact},
        )

    fake_http.texture_zip = _texture_zip(
        plan.selected_unit_ids,
        payload=b"changed",
    )
    fake_http.output_content = first_output_bytes
    with pytest.raises(RuntimeError, match="changed preserved artifacts"):
        client.execute(
            plan,
            (targeted_id,),
            output_dir=request.output_dir,
            preserved_artifacts={preserved_id: preserved_artifact},
        )

    assert (
        tuple(Path(path).read_bytes() for path in preserved_artifact.artifact_paths)
        == preserved_bytes
    )
    assert first_output_path.read_bytes() == first_output_bytes


def test_real_service_forwards_session_and_per_unit_outer_generator_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text('#usda 1.0\n\ndef Xform "Board" {}\n', encoding="utf-8")
    reference = tmp_path / "appearance.png"
    reference.write_bytes(b"exact-outer-reference-bytes")
    reference_binding = ExecutionArtifactBinding(
        path=str(reference.resolve()),
        sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        size_bytes=reference.stat().st_size,
    )
    source_binding = ExecutionArtifactBinding(
        path=str(source_asset.resolve()),
        sha256=hashlib.sha256(source_asset.read_bytes()).hexdigest(),
        size_bytes=source_asset.stat().st_size,
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Outer intent.",
        reference_artifacts=(
            TextureReferenceArtifact(
                role="appearance_reference",
                artifact=reference_binding,
            ),
        ),
        metadata={
            "auto_prompt_enabled": False,
            "texture_backend": "mock",
            "backend_engine": "outer-engine",
            "texture_size": 2048,
            "seed": 41,
            "backend_custom_parameters": {"strength": 0.85, "mode": "outer"},
            "source_asset_binding": source_binding.model_dump(mode="json"),
        },
    )
    fixture_payload = (
        MockTexturePlannerExecutorClient(unit_count=1)
        .plan(request)
        .model_dump(mode="json")
    )
    fixture_payload["request"]["texture_size"] = 2048
    fixture_payload["execution"]["texture_size"] = 2048
    fixture_plan = TexturePlanDocument.model_validate(fixture_payload)
    fake_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = client.plan(request)
    unit = plan.selected_units[0]
    unit_scope = {unit.unit_id: unit.model_dump(mode="json")}
    inputs = TextureGeneratorInputs(
        backend="mock",
        engine="outer-engine",
        prompt="outer-authored warm red paint",
        seed=41,
        texture_size=2048,
        parameters={"strength": 0.85, "mode": "outer"},
        reference_artifacts=(reference_binding,),
    )

    result = client.execute_outer_plan(
        plan,
        (unit.unit_id,),
        unit_scope_by_id=unit_scope,
        generator_inputs_by_unit={unit.unit_id: inputs},
        output_dir=request.output_dir,
        preserved_artifacts={},
        persist_resume_state=lambda: None,
    )

    plan_form = fake_http.posts[0][1]["data"]
    assert fake_http.uploaded_files["usd_file"] == source_asset.read_bytes()
    assert fake_http.uploaded_files["reference_image_file"] == reference.read_bytes()
    assert plan_form["texture_backend"] == "mock"
    assert plan_form["backend_engine"] == "outer-engine"
    assert plan_form["texture_size"] == "2048"
    assert plan_form["seed"] == "41"
    assert json.loads(plan_form["backend_custom_parameters_json"]) == {
        "strength": 0.85,
        "mode": "outer",
    }
    generation_payload = fake_http.posts[1][1]["json"]
    material_path = unit_scope[unit.unit_id]["material_prim_paths"][0]
    assert generation_payload["material_textures"][material_path] == {
        "prompt": "outer-authored warm red paint",
        "detail_policy": "surface_only",
        "material_path": material_path,
        "prim_paths": unit_scope[unit.unit_id]["member_prim_paths"],
    }
    assert str(reference) not in json.dumps(generation_payload)
    assert result.requested_unit_ids == (unit.unit_id,)

    source_bytes = source_asset.read_bytes()
    reference_bytes = reference.read_bytes()

    class MutatingUploadSession(_FakeTextureServiceSession):
        def post(self, url: str, **kwargs: object) -> _FakeResponse:
            if url.endswith("/pipeline"):
                source_asset.write_bytes(b"changed-after-source-snapshot")
                reference.write_bytes(b"changed-after-reference-snapshot")
            return super().post(url, **kwargs)

    snapshot_http = MutatingUploadSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    snapshot_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=snapshot_http,
    )
    snapshot_client.plan(request)
    assert snapshot_http.uploaded_files["usd_file"] == source_bytes
    assert snapshot_http.uploaded_files["reference_image_file"] == reference_bytes
    source_asset.write_bytes(source_bytes)
    reference.write_bytes(reference_bytes)

    dependent_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    dependent_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=dependent_http,
    )
    dependent_request = request.model_copy(
        update={
            "metadata": {
                **request.metadata,
                "source_dependencies": [reference_binding.model_dump(mode="json")],
            }
        }
    )
    with pytest.raises(ValueError, match="byte-self-contained source"):
        dependent_client.plan(dependent_request)
    assert dependent_http.posts == []

    untampered_reference = reference.read_bytes()
    tampering_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    tampering_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=tampering_http,
    )
    real_os_open = texture_client_module.os.open

    def tamper_immediately_before_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        if Path(path) == reference.resolve():
            reference.write_bytes(b"changed-at-service-upload-boundary")
        return real_os_open(path, flags, mode)

    monkeypatch.setattr(
        texture_client_module.os,
        "open",
        tamper_immediately_before_open,
    )
    with pytest.raises(
        ValueError,
        match="reference image bytes changed before upload",
    ):
        tampering_client.plan(request)
    assert tampering_http.posts == []
    reference.write_bytes(untampered_reference)


def test_service_reconciliation_probe_accepts_pending_outer_generator_inputs(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _CrashAfterAcceptedRegenerateSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
        crash_on_regenerate_call=1,
    )
    first_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = first_client.plan(request)
    unit = plan.selected_units[0]
    unit_scope = {unit.unit_id: unit.model_dump(mode="json")}
    inputs = TextureGeneratorInputs(
        backend="mock",
        prompt="outer-authored warm red paint",
    )
    durable_state: dict[str, Any] = {}

    def persist_resume_state() -> None:
        nonlocal durable_state
        durable_state = first_client.export_resume_state(plan)

    with pytest.raises(
        RuntimeError,
        match="simulated process interruption after remote acceptance",
    ):
        first_client.execute_outer_plan(
            plan,
            (unit.unit_id,),
            unit_scope_by_id=unit_scope,
            generator_inputs_by_unit={unit.unit_id: inputs},
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=persist_resume_state,
        )

    pending = durable_state["pending_execution"]
    assert pending["material_textures"]
    resumed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    resumed_client.restore_resume_state(plan, durable_state)

    assert resumed_client.can_reconcile_authorized_execution(
        plan,
        (unit.unit_id,),
        preserved_artifacts={},
    )
    with pytest.raises(
        RuntimeError,
        match="generator inputs differ from outer semantics",
    ):
        resumed_client.execute_resumable(
            plan,
            (unit.unit_id,),
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=lambda: None,
            material_textures={},
        )


def test_preserved_artifact_hashing_streams_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit_id = MockTexturePlannerExecutorClient(unit_count=1).unit_ids[0]
    artifact_path = tmp_path / f"{unit_id}_albedo.png"
    artifact_bytes = b"streamed-preserved-artifact"
    artifact_path.write_bytes(artifact_bytes)

    def reject_whole_file_read(_path: Path) -> bytes:
        raise AssertionError("preserved artifacts must be hashed as a stream")

    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)

    assert TextureAgentServiceClient._unit_artifact_hashes(
        (artifact_path,),
        unit_id,
    ) == {
        "albedo.png": {hashlib.sha256(artifact_bytes).hexdigest()},
    }


def test_remote_execution_state_rejects_malformed_baseline_owner() -> None:
    unit_id = MockTexturePlannerExecutorClient(unit_count=1).unit_ids[0]

    with pytest.raises(ValueError, match="baseline_execution_id"):
        texture_client_module._TextureRemoteExecutionState.model_validate(
            {
                "execution_id": "a" * 32,
                "session_id": "session-466",
                "plan_key": "b" * 64,
                "unit_ids": [unit_id],
                "generation_by_unit_before": {unit_id: 0},
                "full_plan_regeneration": False,
                "phase": "generation",
                "baseline_status_digest": "c" * 64,
                "baseline_execution_id": "not-an-execution-token",
            }
        )


def test_preserved_package_verification_scans_each_unique_package_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit_ids = MockTexturePlannerExecutorClient(unit_count=3).unit_ids
    previous_package = tmp_path / "previous.usdz"
    previous_package.write_bytes(_texture_zip(unit_ids))
    preserved_artifacts = {
        unit_id: TextureUnitArtifact(
            unit_id=unit_id,
            artifact_paths=(str(tmp_path / f"{unit_id}.png"),),
            metadata={"output_asset_path": str(previous_package)},
        )
        for unit_id in unit_ids
    }

    zip_scans: list[object] = []
    original_zip_file = texture_client_module.zipfile.ZipFile

    def counting_zip_file(file: Any, *args: Any, **kwargs: Any) -> Any:
        zip_scans.append(file)
        return original_zip_file(file, *args, **kwargs)

    monkeypatch.setattr(
        texture_client_module.zipfile,
        "ZipFile",
        counting_zip_file,
    )

    TextureAgentServiceClient._verify_preserved_package_artifacts(
        _texture_zip(unit_ids),
        preserved_artifacts,
    )

    assert len(zip_scans) == 2
    assert sum(isinstance(source, Path) for source in zip_scans) == 1
    assert sum(isinstance(source, BytesIO) for source in zip_scans) == 1

    changed_unit_id = unit_ids[1]
    changed_package = BytesIO()
    with ZipFile(changed_package, "w") as archive:
        for unit_id in unit_ids:
            payload = b"changed" if unit_id == changed_unit_id else b"png"
            archive.writestr(f"{unit_id}_albedo.png", payload)
            archive.writestr(f"textures/{unit_id}/albedo.png", payload)

    zip_scans.clear()
    with pytest.raises(
        RuntimeError,
        match=f"output package changed preserved artifacts for {changed_unit_id}",
    ):
        TextureAgentServiceClient._verify_preserved_package_artifacts(
            changed_package.getvalue(),
            preserved_artifacts,
        )

    assert len(zip_scans) == 2
    assert sum(isinstance(source, Path) for source in zip_scans) == 1
    assert sum(isinstance(source, BytesIO) for source in zip_scans) == 1


def test_real_service_adapter_retries_transient_status_failure(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text('#usda 1.0\n\ndef Xform "Board" {}\n', encoding="utf-8")
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Plan bounded board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
        status_responses=[
            _FakeResponse(status_code=500),
            _FakeResponse({"status": "completed"}),
        ],
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )

    assert client.plan(request).selected_unit_ids == fixture_plan.selected_unit_ids
    assert fake_http.status_request_count == 2


def test_real_service_adapter_does_not_retry_client_status_failure(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text('#usda 1.0\n\ndef Xform "Board" {}\n', encoding="utf-8")
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Plan bounded board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
        status_responses=[_FakeResponse(status_code=401)],
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )

    with pytest.raises(requests.HTTPError):
        client.plan(request)
    assert fake_http.status_request_count == 1


def test_real_service_adapter_bounds_consecutive_status_failures(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text('#usda 1.0\n\ndef Xform "Board" {}\n', encoding="utf-8")
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Plan bounded board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
        status_responses=[_FakeResponse(status_code=500) for _ in range(3)],
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        max_status_poll_failures=2,
        session=fake_http,
    )

    with pytest.raises(requests.HTTPError):
        client.plan(request)
    assert fake_http.status_request_count == 3


@pytest.mark.parametrize(
    ("timeout_seconds", "poll_interval_seconds", "expected_message"),
    [
        (0, 1, "timeout_seconds must be positive"),
        (-1, 1, "timeout_seconds must be positive"),
        (1, -0.1, "poll_interval_seconds must be non-negative"),
    ],
)
def test_real_service_adapter_rejects_invalid_poll_timing(
    timeout_seconds: float,
    poll_interval_seconds: float,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        TextureAgentServiceClient(
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )


def test_real_service_adapter_clamps_poll_to_remaining_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text('#usda 1.0\n\ndef Xform "Board" {}\n', encoding="utf-8")
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Plan bounded board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
        status_responses=[
            _FakeResponse({"status": "pending"}),
            _FakeResponse({"status": "completed"}),
        ],
    )
    monotonic_values = iter((100.0, 102.0, 103.0, 104.0))
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        texture_client_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(texture_client_module.time, "sleep", sleep_calls.append)
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        timeout_seconds=10,
        poll_interval_seconds=30,
        session=fake_http,
    )

    assert client.plan(request).selected_unit_ids == fixture_plan.selected_unit_ids
    assert fake_http.status_request_timeouts == [8.0, 6.0]
    assert sleep_calls == [7.0]


def test_real_service_adapter_restores_session_for_workflow_resume(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text('#usda 1.0\n\ndef Xform "Board" {}\n', encoding="utf-8")
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Plan bounded board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    planning_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    planning_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=planning_http,
    )
    plan = planning_client.plan(request)
    resume_state = planning_client.export_resume_state(plan)

    resumed_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    resumed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=resumed_http,
    )
    resumed_client.restore_resume_state(plan, resume_state)
    result = resumed_client.execute(
        plan,
        plan.selected_unit_ids,
        output_dir=request.output_dir,
        preserved_artifacts={},
    )

    assert resumed_http.posts[0][0].endswith("/session-466/regenerate")
    assert result.requested_unit_ids == plan.selected_unit_ids
    assert result.metadata["session_id"] == "session-466"


def test_service_execution_fails_before_post_when_status_has_no_execution_id(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
        status_responses=[
            _FakeResponse({"status": "completed"}),
            _FakeResponse({"status": "completed"}),
        ],
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = client.plan(request)

    with pytest.raises(
        RuntimeError,
        match="does not advertise execution-id reconciliation",
    ):
        client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=lambda: None,
        )

    assert not any(url.endswith("/regenerate") for url, _kwargs in fake_http.posts)


@pytest.mark.parametrize(
    ("crash_on_regenerate_call", "checkpointed_phase"),
    [
        (1, "generation"),
        (2, "apply"),
    ],
)
def test_workflow_adopts_accepted_remote_execution_without_duplicate_dispatch(
    tmp_path: Path,
    crash_on_regenerate_call: int,
    checkpointed_phase: str,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _CrashAfterAcceptedRegenerateSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
        crash_on_regenerate_call=crash_on_regenerate_call,
    )
    first_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated process interruption after remote acceptance",
    ):
        run_batch_texture_workflow(
            request,
            client=first_client,
            validator=MockTextureSceneValidator(),
        )

    interrupted = TextureWorkflowCheckpointStore(request.output_dir).load()
    assert interrupted.next_action == "execute"
    pending = interrupted.client_resume_state["pending_execution"]
    assert pending["phase"] == checkpointed_phase
    assert pending["unit_ids"] == list(fixture_plan.selected_unit_ids)

    class StopAfterAdoptedExecution(MockTextureSceneValidator):
        def validate(self, **kwargs: Any) -> TextureValidationResult:
            raise RuntimeError("stop after adopted remote execution")

    resumed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    with pytest.raises(RuntimeError, match="stop after adopted remote execution"):
        run_batch_texture_workflow(
            request,
            client=resumed_client,
            validator=StopAfterAdoptedExecution(),
            resume=True,
        )

    regenerate_posts, execution_ids = _owned_regenerate_payloads(fake_http)
    assert len(execution_ids) == 2
    generation_request = {
        "steps": [
            "prepare_uvs",
            "generate_textures",
            "blend_textures",
        ]
    }
    apply_request = {"steps": ["apply_textures"]}
    assert regenerate_posts == (
        [generation_request, generation_request, apply_request]
        if checkpointed_phase == "generation"
        else [generation_request, apply_request, apply_request]
    )
    adopted = TextureWorkflowCheckpointStore(request.output_dir).load()
    assert adopted.next_action == "validate"
    assert len(adopted.executions) == 1
    assert adopted.executions[0].requested_unit_ids == fixture_plan.selected_unit_ids
    assert adopted.client_resume_state["pending_execution"] is None


def test_service_poll_cancellation_finalizes_durably_and_resumes_owned_execution(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    first_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )

    result = run_batch_texture_workflow(
        request,
        client=first_client,
        validator=MockTextureSceneValidator(),
        cancellation_check=lambda: fake_http.status_request_count >= 4,
    )

    assert result.status == "cancelled"
    assert result.output_asset_path is None
    cancelled = TextureWorkflowCheckpointStore(request.output_dir).load()
    assert cancelled.next_action == "execute"
    assert cancelled.terminal_status == "cancelled"
    pending = cancelled.client_resume_state["pending_execution"]
    assert pending["phase"] == "generation"
    assert pending["dispatched"] is True
    assert len(fake_http.requests_by_execution_id) == 1

    class StopAfterResumedExecution(MockTextureSceneValidator):
        def validate(self, **kwargs: Any) -> TextureValidationResult:
            raise RuntimeError("stop after resumed remote execution")

    resumed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    with pytest.raises(RuntimeError, match="stop after resumed remote execution"):
        run_batch_texture_workflow(
            request,
            client=resumed_client,
            validator=StopAfterResumedExecution(),
            resume=True,
        )

    regenerate_posts, execution_ids = _owned_regenerate_payloads(fake_http)
    generation_request = {
        "steps": [
            "prepare_uvs",
            "generate_textures",
            "blend_textures",
        ]
    }
    assert len(execution_ids) == 2
    assert regenerate_posts == [
        generation_request,
        generation_request,
        {"steps": ["apply_textures"]},
    ]
    resumed = TextureWorkflowCheckpointStore(request.output_dir).load()
    assert resumed.next_action == "validate"
    assert resumed.terminal_status is None
    assert len(resumed.executions) == 1
    assert resumed.client_resume_state["pending_execution"] is None


def test_service_resume_adopts_terminal_generation_before_apply_transition(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _CrashAfterAcceptedRegenerateSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
        crash_on_regenerate_call=99,
    )
    first_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = first_client.plan(request)
    durable_state: dict[str, Any] = {}

    def persist_until_apply_transition() -> None:
        nonlocal durable_state
        candidate = first_client.export_resume_state(plan)
        pending = candidate["pending_execution"]
        if pending["phase"] == "apply":
            raise RuntimeError(
                "simulated interruption before apply transition persisted"
            )
        durable_state = candidate

    with pytest.raises(
        RuntimeError,
        match="simulated interruption before apply transition persisted",
    ):
        first_client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=persist_until_apply_transition,
        )

    assert durable_state["pending_execution"]["phase"] == "generation"
    assert durable_state["pending_execution"]["dispatched"] is True

    resumed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    resumed_client.restore_resume_state(plan, durable_state)
    result = resumed_client.execute_resumable(
        plan,
        plan.selected_unit_ids,
        output_dir=request.output_dir,
        preserved_artifacts={},
        persist_resume_state=lambda: None,
    )

    regenerate_posts, execution_ids = _owned_regenerate_payloads(fake_http)
    assert len(execution_ids) == 2
    assert regenerate_posts == [
        {
            "steps": [
                "prepare_uvs",
                "generate_textures",
                "blend_textures",
            ]
        },
        {
            "steps": [
                "prepare_uvs",
                "generate_textures",
                "blend_textures",
            ]
        },
        {"steps": ["apply_textures"]},
    ]
    assert result.requested_unit_ids == plan.selected_unit_ids
    assert result.unit_artifacts[0].generation == 1


def test_service_resume_revalidates_terminal_execution_payload(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _CrashAfterAcceptedRegenerateSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
        crash_on_regenerate_call=1,
    )
    first_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = first_client.plan(request)
    durable_state: dict[str, Any] = {}

    def persist_state() -> None:
        nonlocal durable_state
        durable_state = first_client.export_resume_state(plan)

    with pytest.raises(
        RuntimeError,
        match="simulated process interruption after remote acceptance",
    ):
        first_client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=persist_state,
        )

    execution_id = fake_http.execution_id
    assert execution_id is not None
    fake_http.requests_by_execution_id[execution_id] = {"steps": ["apply_textures"]}

    resumed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    resumed_client.restore_resume_state(plan, durable_state)
    with pytest.raises(requests.HTTPError) as exc_info:
        resumed_client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=lambda: None,
        )

    assert exc_info.value.response is not None
    assert exc_info.value.response.status_code == 409
    regenerate_posts, execution_ids = _owned_regenerate_payloads(fake_http)
    assert len(execution_ids) == 1
    assert len(regenerate_posts) == 2


def test_service_replay_after_crash_and_stale_status_registers_each_phase_once(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _IdempotentCrashWithStaleStatusSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    first_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = first_client.plan(request)
    durable_state: dict[str, Any] = {}

    def persist_state() -> None:
        nonlocal durable_state
        durable_state = first_client.export_resume_state(plan)

    with pytest.raises(
        RuntimeError,
        match="simulated crash after accepted generation",
    ):
        first_client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=persist_state,
        )

    assert durable_state["pending_execution"]["phase"] == "generation"
    assert durable_state["pending_execution"]["dispatched"] is False

    resumed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    resumed_client.restore_resume_state(plan, durable_state)
    result = resumed_client.execute_resumable(
        plan,
        plan.selected_unit_ids,
        output_dir=request.output_dir,
        preserved_artifacts={},
        persist_resume_state=lambda: None,
    )

    owned_posts = [
        kwargs["json"] for url, kwargs in fake_http.posts if url.endswith("/regenerate")
    ]
    generation_execution_id = owned_posts[0]["execution_id"]
    assert owned_posts[1]["execution_id"] == generation_execution_id
    apply_execution_id = owned_posts[2]["execution_id"]
    assert apply_execution_id != generation_execution_id
    assert fake_http.backend_registrations_by_execution_id == {
        generation_execution_id: 1,
        apply_execution_id: 1,
    }
    assert result.unit_artifacts[0].generation == 1


def test_service_resume_replays_owned_orphan_before_waiting(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _OwnedOrphanRegenerateSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    first_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        timeout_seconds=0.1,
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = first_client.plan(request)
    durable_state: dict[str, Any] = {}

    def persist_and_interrupt_after_acceptance() -> None:
        nonlocal durable_state
        durable_state = first_client.export_resume_state(plan)
        pending = durable_state["pending_execution"]
        if pending is not None and pending["dispatched"]:
            raise RuntimeError("simulated service restart with an owned orphan")

    with pytest.raises(
        RuntimeError,
        match="simulated service restart with an owned orphan",
    ):
        first_client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=persist_and_interrupt_after_acceptance,
        )

    pending = durable_state["pending_execution"]
    assert pending["phase"] == "generation"
    assert pending["dispatched"] is True
    assert fake_http.current_status["status"] == "pending"
    assert fake_http.replays_by_execution_id == {}

    resumed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        timeout_seconds=0.1,
        poll_interval_seconds=0,
        session=fake_http,
    )
    resumed_client.restore_resume_state(plan, durable_state)
    result = resumed_client.execute_resumable(
        plan,
        plan.selected_unit_ids,
        output_dir=request.output_dir,
        preserved_artifacts={},
        persist_resume_state=lambda: None,
    )

    owned_posts = [
        dict(kwargs["json"])
        for url, kwargs in fake_http.posts
        if url.endswith("/regenerate")
    ]
    generation_execution_id = str(owned_posts[0]["execution_id"])
    assert owned_posts[1] == owned_posts[0]
    assert fake_http.replays_by_execution_id == {generation_execution_id: 1}
    assert owned_posts[2]["execution_id"] != generation_execution_id
    assert result.requested_unit_ids == plan.selected_unit_ids
    assert result.unit_artifacts[0].generation == 1


def test_service_waits_for_new_terminal_status_after_successful_dispatch(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _LaggingAcceptedRegenerateSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )

    plan = client.plan(request)
    result = client.execute_resumable(
        plan,
        plan.selected_unit_ids,
        output_dir=request.output_dir,
        preserved_artifacts={},
        persist_resume_state=lambda: None,
    )

    regenerate_posts, execution_ids = _owned_regenerate_payloads(fake_http)
    assert len(execution_ids) == 2
    assert regenerate_posts == [
        {
            "steps": [
                "prepare_uvs",
                "generate_textures",
                "blend_textures",
            ]
        },
        {"steps": ["apply_textures"]},
    ]
    for phase in (1, 2):
        phase_history = [
            (status, revision)
            for observed_phase, status, revision in fake_http.status_history
            if observed_phase == phase
        ]
        assert phase_history[:3] == [
            ("completed", f"revision-{phase - 1}"),
            ("pending", f"revision-{phase}-pending"),
            ("completed", f"revision-{phase}"),
        ]
    assert [url.rsplit("/", 1)[-1] for url in fake_http.artifact_gets] == [
        "results",
        "textures",
        "output",
    ]
    assert result.requested_unit_ids == plan.selected_unit_ids


def test_service_generation_refreshes_delayed_plan_terminal_baseline(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = client.plan(request)
    drift_injected = False

    def persist_with_delayed_plan_flush() -> None:
        nonlocal drift_injected
        pending = client.export_resume_state(plan)["pending_execution"]
        if pending["phase"] != "generation" or pending["dispatched"] or drift_injected:
            return
        drift_injected = True
        fake_http.run_statuses = [
            _FakeResponse(
                {
                    "status": "completed",
                    "updated_at": "plan-revision-delayed-flush",
                    "completed_steps": [{"name": "plan-flush"}],
                    "execution_id": pending["baseline_execution_id"],
                }
            )
        ]

    result = client.execute_resumable(
        plan,
        plan.selected_unit_ids,
        output_dir=request.output_dir,
        preserved_artifacts={},
        persist_resume_state=persist_with_delayed_plan_flush,
    )

    regenerate_posts, execution_ids = _owned_regenerate_payloads(fake_http)
    assert drift_injected is True
    assert len(execution_ids) == 2
    assert regenerate_posts == [
        {
            "steps": [
                "prepare_uvs",
                "generate_textures",
                "blend_textures",
            ]
        },
        {"steps": ["apply_textures"]},
    ]
    assert result.requested_unit_ids == plan.selected_unit_ids


def test_service_apply_refreshes_delayed_generation_terminal_baseline(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = client.plan(request)
    drift_injected = False

    def persist_with_delayed_generation_flush() -> None:
        nonlocal drift_injected
        pending = client.export_resume_state(plan)["pending_execution"]
        if pending["phase"] != "apply" or drift_injected:
            return
        drift_injected = True
        generation_execution_id = fake_http.execution_id
        fake_http.run_statuses = [
            _FakeResponse(
                {
                    "status": "completed",
                    "updated_at": "revision-1-delayed-flush",
                    "completed_steps": [{"name": "generation-flush"}],
                    "execution_id": generation_execution_id,
                }
            )
        ]

    result = client.execute_resumable(
        plan,
        plan.selected_unit_ids,
        output_dir=request.output_dir,
        preserved_artifacts={},
        persist_resume_state=persist_with_delayed_generation_flush,
    )

    regenerate_posts, execution_ids = _owned_regenerate_payloads(fake_http)
    assert drift_injected is True
    assert len(execution_ids) == 2
    assert regenerate_posts == [
        {
            "steps": [
                "prepare_uvs",
                "generate_textures",
                "blend_textures",
            ]
        },
        {"steps": ["apply_textures"]},
    ]
    assert result.requested_unit_ids == plan.selected_unit_ids


def test_service_resume_waits_past_stale_terminal_for_dispatched_phase(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _LaggingAcceptedRegenerateSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    first_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = first_client.plan(request)
    durable_state: dict[str, Any] = {}

    def persist_and_stop_after_dispatch() -> None:
        nonlocal durable_state
        durable_state = first_client.export_resume_state(plan)
        pending = durable_state["pending_execution"]
        if pending is not None and pending["dispatched"]:
            raise RuntimeError("simulated interruption after dispatch checkpoint")

    with pytest.raises(
        RuntimeError,
        match="simulated interruption after dispatch checkpoint",
    ):
        first_client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=persist_and_stop_after_dispatch,
        )

    assert durable_state["pending_execution"]["dispatched"] is True
    assert fake_http.status_history[-1] == (0, "completed", "revision-0")

    resumed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    resumed_client.restore_resume_state(plan, durable_state)
    result = resumed_client.execute_resumable(
        plan,
        plan.selected_unit_ids,
        output_dir=request.output_dir,
        preserved_artifacts={},
        persist_resume_state=lambda: None,
    )

    regenerate_posts, execution_ids = _owned_regenerate_payloads(fake_http)
    assert len(execution_ids) == 2
    assert regenerate_posts == [
        {
            "steps": [
                "prepare_uvs",
                "generate_textures",
                "blend_textures",
            ]
        },
        {
            "steps": [
                "prepare_uvs",
                "generate_textures",
                "blend_textures",
            ]
        },
        {"steps": ["apply_textures"]},
    ]
    for phase in (1, 2):
        phase_history = [
            (status, revision)
            for observed_phase, status, revision in fake_http.status_history
            if observed_phase == phase
        ]
        assert phase_history[:3] == [
            ("completed", f"revision-{phase - 1}"),
            ("pending", f"revision-{phase}-pending"),
            ("completed", f"revision-{phase}"),
        ]
    assert result.requested_unit_ids == plan.selected_unit_ids


def test_service_resume_rejects_external_execution_before_collect(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _RecordingArtifactGetsSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    first_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = first_client.plan(request)
    durable_state: dict[str, Any] = {}

    def persist_until_collect() -> None:
        nonlocal durable_state
        durable_state = first_client.export_resume_state(plan)
        pending = durable_state["pending_execution"]
        if pending is not None and pending["phase"] == "collect":
            raise RuntimeError("simulated interruption before artifact collection")

    with pytest.raises(
        RuntimeError,
        match="simulated interruption before artifact collection",
    ):
        first_client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=persist_until_collect,
        )

    assert durable_state["pending_execution"]["phase"] == "collect"
    fake_http.run_revision += 1
    fake_http.execution_id = "f" * 32
    fake_http.run_statuses = []

    resumed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    resumed_client.restore_resume_state(plan, durable_state)
    with pytest.raises(
        RuntimeError,
        match="status belongs to a different execution",
    ):
        resumed_client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=lambda: None,
        )

    assert fake_http.artifact_gets == []


def test_service_resume_replays_owned_orphan_before_collect(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _CollectPhaseOrphanSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    first_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = first_client.plan(request)
    durable_state: dict[str, Any] = {}

    def persist_until_collect() -> None:
        nonlocal durable_state
        durable_state = first_client.export_resume_state(plan)
        pending = durable_state["pending_execution"]
        if pending is not None and pending["phase"] == "collect":
            raise RuntimeError("simulated interruption before artifact collection")

    with pytest.raises(
        RuntimeError,
        match="simulated interruption before artifact collection",
    ):
        first_client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=persist_until_collect,
        )

    pending = durable_state["pending_execution"]
    assert pending["phase"] == "collect"
    completed_execution_id = fake_http.execution_id
    assert completed_execution_id is not None
    fake_http.restart_with_owned_orphan()

    resumed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        timeout_seconds=0.05,
        poll_interval_seconds=0,
        session=fake_http,
    )
    resumed_client.restore_resume_state(plan, durable_state)
    result = resumed_client.execute_resumable(
        plan,
        plan.selected_unit_ids,
        output_dir=request.output_dir,
        preserved_artifacts={},
        persist_resume_state=lambda: None,
    )

    assert fake_http.collect_replay_count == 1
    assert fake_http.posts[-1][1]["json"] == {
        "steps": ["apply_textures"],
        "execution_id": completed_execution_id,
    }
    assert result.requested_unit_ids == plan.selected_unit_ids


def test_service_collect_orphan_replay_503_remains_resumable(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _TransientCollectReplayFailureSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    first_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = first_client.plan(request)
    durable_state: dict[str, Any] = {}

    def persist_until_collect() -> None:
        nonlocal durable_state
        durable_state = first_client.export_resume_state(plan)
        pending = durable_state["pending_execution"]
        if pending is not None and pending["phase"] == "collect":
            raise RuntimeError("simulated interruption before artifact collection")

    with pytest.raises(
        RuntimeError,
        match="simulated interruption before artifact collection",
    ):
        first_client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=persist_until_collect,
        )

    pending = durable_state["pending_execution"]
    assert pending["phase"] == "collect"
    completed_execution_id = fake_http.execution_id
    assert completed_execution_id is not None
    fake_http.restart_with_owned_orphan()
    replay_post_offset = len(fake_http.posts)
    status_count_before_replay = fake_http.status_request_count

    failed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        timeout_seconds=0.05,
        poll_interval_seconds=0,
        session=fake_http,
    )
    failed_client.restore_resume_state(plan, durable_state)
    with pytest.raises(requests.HTTPError) as exc_info:
        failed_client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=lambda: None,
        )

    assert exc_info.value.response is not None
    assert exc_info.value.response.status_code == 503
    assert fake_http.status_request_count == status_count_before_replay + 1
    failed_resume_state = failed_client.export_resume_state(plan)
    assert failed_resume_state["pending_execution"] == pending

    later_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        timeout_seconds=0.05,
        poll_interval_seconds=0,
        session=fake_http,
    )
    later_client.restore_resume_state(plan, failed_resume_state)
    result = later_client.execute_resumable(
        plan,
        plan.selected_unit_ids,
        output_dir=request.output_dir,
        preserved_artifacts={},
        persist_resume_state=lambda: None,
    )

    replay_posts = fake_http.posts[replay_post_offset:]
    assert len(replay_posts) == 2
    first_payload = replay_posts[0][1]["json"]
    second_payload = replay_posts[1][1]["json"]
    assert (
        first_payload
        == second_payload
        == {
            "steps": ["apply_textures"],
            "execution_id": completed_execution_id,
        }
    )
    assert fake_http.collect_replay_count == 1
    assert result.requested_unit_ids == plan.selected_unit_ids


def test_service_rejects_status_owned_by_another_execution(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = client.plan(request)
    fake_http.status_responses = [
        _FakeResponse(
            {
                "status": "completed",
                "updated_at": "baseline",
                "execution_id": None,
            }
        ),
        _FakeResponse(
            {
                "status": "completed",
                "updated_at": "external-run",
                "execution_id": "f" * 32,
            }
        ),
    ]

    with pytest.raises(
        RuntimeError,
        match="status belongs to a different execution",
    ):
        client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=lambda: None,
        )

    assert not any(url.endswith("/regenerate") for url, _kwargs in fake_http.posts)


def test_service_does_not_adopt_definitive_execution_token_conflict(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    fake_http = _ConflictingExecutionTokenSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip(fixture_plan.selected_unit_ids),
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = client.plan(request)

    with pytest.raises(requests.HTTPError) as exc_info:
        client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=lambda: None,
        )

    assert exc_info.value.response is not None
    assert exc_info.value.response.status_code == 409
    # Plan completion, execution baseline, and pre-dispatch ownership check.
    # A fourth status read would mean the client tried to adopt the conflict.
    assert fake_http.status_request_count == 3


def test_service_collect_failure_does_not_advance_generation_state(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text(
        '#usda 1.0\n\ndef Xform "Board" {}\n',
        encoding="utf-8",
    )
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Generate bounded surface-only board textures.",
    )
    fixture_plan = MockTexturePlannerExecutorClient(unit_count=2).plan(request)
    first_unit_id, second_unit_id = fixture_plan.selected_unit_ids
    fake_http = _FakeTextureServiceSession(
        fixture_plan.model_dump(mode="json"),
        _texture_zip((first_unit_id,)),
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=fake_http,
    )
    plan = client.plan(request)

    with pytest.raises(
        RuntimeError,
        match=f"no downloadable artifact for {second_unit_id}",
    ):
        client.execute_resumable(
            plan,
            plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
            persist_resume_state=lambda: None,
        )

    failed_state = client.export_resume_state(plan)
    assert failed_state["generation_by_unit"] == {}
    assert failed_state["pending_execution"]["phase"] == "collect"

    fake_http.texture_zip = _texture_zip(plan.selected_unit_ids)
    result = client.execute_resumable(
        plan,
        plan.selected_unit_ids,
        output_dir=request.output_dir,
        preserved_artifacts={},
        persist_resume_state=lambda: None,
    )

    assert result.requested_unit_ids == plan.selected_unit_ids
    resumed_state = client.export_resume_state(plan)
    assert resumed_state["generation_by_unit"] == {
        first_unit_id: 1,
        second_unit_id: 1,
    }
    assert resumed_state["pending_execution"] is None


def test_real_service_adapter_rejects_resume_when_session_plan_changed(
    tmp_path: Path,
) -> None:
    source_asset = tmp_path / "board.usda"
    source_asset.write_text('#usda 1.0\n\ndef Xform "Board" {}\n', encoding="utf-8")
    request = TextureWorkflowRequest(
        source_asset=str(source_asset),
        output_dir=tmp_path / "run",
        intent="Plan bounded board textures.",
    )
    checkpoint_plan = MockTexturePlannerExecutorClient(unit_count=1).plan(request)
    changed_payload = checkpoint_plan.model_dump(mode="json")
    changed_payload["execution"]["texture_size"] = 2048
    service_plan = TexturePlanDocument.model_validate(changed_payload)
    resumed_http = _FakeTextureServiceSession(
        service_plan.model_dump(mode="json"),
        _texture_zip(service_plan.selected_unit_ids),
    )
    resumed_client = TextureAgentServiceClient(
        base_url="http://texture.test",
        poll_interval_seconds=0,
        session=resumed_http,
    )

    with pytest.raises(ValueError, match="does not match the checkpointed plan"):
        resumed_client.restore_resume_state(
            checkpoint_plan,
            {
                "session_id": "session-466",
                "generation_by_unit": {},
            },
        )

    with pytest.raises(ValueError, match="plan was not created"):
        resumed_client.execute(
            checkpoint_plan,
            checkpoint_plan.selected_unit_ids,
            output_dir=request.output_dir,
            preserved_artifacts={},
        )
    assert resumed_http.posts == []


def test_skill_routed_steps_require_decisions_and_refine_only_failed_unit(
    tmp_path: Path,
) -> None:
    client = MockTexturePlannerExecutorClient(unit_count=2)
    failing_id = client.unit_ids[1]
    validator = MockTextureSceneValidator(failure_schedule=[(failing_id,), ()])
    request = _request(tmp_path)

    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    assert isinstance(observation, TextureStepObservation)
    assert observation.action == "execute"
    assert observation.required_operations == (
        "inspect_scope",
        "inspect_uv",
        "generate_candidate",
        "preview_apply",
    )
    assert client.execution_calls == []

    observed_actions: list[str] = []
    while isinstance(observation, TextureStepObservation):
        observed_actions.append(observation.action)
        patch = _decision_for(observation)
        observation = run_texture_workflow_step(
            request,
            mode="batch",
            client=client,
            validator=validator,
            decision_patch=patch,
        )

    assert observation.success
    assert observed_actions == [
        "execute",
        "validate",
        "refine",
        "validate",
        "finalize",
    ]
    assert [call.unit_ids for call in client.execution_calls] == [
        client.unit_ids,
        (failing_id,),
    ]
    assert [call.unit_ids for call in validator.calls] == [
        client.unit_ids,
        (failing_id,),
    ]
    assert observation.decision_ledger_path is not None
    ledger = TextureDecisionLedger.model_validate_json(
        Path(observation.decision_ledger_path).read_text(encoding="utf-8")
    )
    assert [record.action for record in ledger.records] == observed_actions
    summary = json.loads(Path(observation.final_summary_path).read_text())
    assert summary["artifacts"]["decision_ledger"] == observation.decision_ledger_path


def test_resume_decision_state_rejects_ledger_truncated_behind_checkpoint(
    tmp_path: Path,
) -> None:
    client = MockTexturePlannerExecutorClient(unit_count=2)
    failing_id = client.unit_ids[1]
    validator = MockTextureSceneValidator(failure_schedule=[(failing_id,), ()])
    request = _request(tmp_path)
    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    while isinstance(observation, TextureStepObservation) and (
        observation.action != "finalize"
    ):
        observation = run_texture_workflow_step(
            request,
            mode="batch",
            client=client,
            validator=validator,
            decision_patch=_decision_for(observation),
        )
    assert isinstance(observation, TextureStepObservation)
    ledger_path = tmp_path / "texture_decision_ledger.json"
    ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger_payload["records"] = ledger_payload["records"][:2]
    ledger_path.write_text(json.dumps(ledger_payload), encoding="utf-8")

    with pytest.raises(
        TextureWorkflowRuntimeError,
        match="does not cover the durable checkpoint",
    ):
        run_texture_workflow_step(
            request,
            mode="batch",
            client=client,
            validator=validator,
            decision_patch=_decision_for(observation),
        )


def test_resume_decision_state_recovers_exact_current_orphan_patch(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    client = MockTexturePlannerExecutorClient(unit_count=1)
    validator = MockTextureSceneValidator()
    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    assert isinstance(observation, TextureStepObservation)
    patch = _decision_for(observation)
    patch_path = Path(observation.decision_patch_path or "")
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch.model_dump_json(), encoding="utf-8")
    orphan_bytes = patch_path.read_bytes()
    checkpoint = TextureWorkflowCheckpointStore(tmp_path).load()

    assert (
        verify_texture_resume_decision_state(checkpoint, output_dir=tmp_path) == patch
    )
    resumed = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )

    assert isinstance(resumed, TextureStepObservation)
    assert resumed.action == "validate"
    assert patch_path.read_bytes() == orphan_bytes
    ledger = TextureDecisionLedger.model_validate_json(
        (tmp_path / "texture_decision_ledger.json").read_text(encoding="utf-8")
    )
    assert [record.action for record in ledger.records] == ["execute"]


def test_resume_decision_state_recovers_ledger_patch_without_progress(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    client = MockTexturePlannerExecutorClient(unit_count=1)
    validator = MockTextureSceneValidator()
    plan = client.plan(request)
    checkpoint = TextureWorkflowCheckpointStore(tmp_path).create(
        mode="batch",
        request=request,
        plan=plan,
        source_identity_digest=texture_source_identity_digest(request, plan=plan),
        next_action="execute",
        progress=(),
        client_resume_state={},
    )
    observation = build_texture_step_observation(checkpoint, output_dir=tmp_path)
    patch = _decision_for(observation)
    record_texture_decision_patch(patch, output_dir=tmp_path)

    assert (
        verify_texture_resume_decision_state(checkpoint, output_dir=tmp_path) == patch
    )
    resumed = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )

    assert isinstance(resumed, TextureStepObservation)
    assert resumed.action == "validate"
    assert len(client.execution_calls) == 1


@pytest.mark.parametrize("ledger_backed", [False, True])
def test_recovered_current_patch_cancellation_requires_fresh_decision(
    tmp_path: Path,
    ledger_backed: bool,
) -> None:
    request = _request(tmp_path)
    client = MockTexturePlannerExecutorClient(unit_count=1)
    validator = MockTextureSceneValidator()
    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    assert isinstance(observation, TextureStepObservation)
    patch = _decision_for(observation)
    if ledger_backed:
        record_texture_decision_patch(patch, output_dir=tmp_path)
    else:
        patch_path = Path(observation.decision_patch_path or "")
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(patch.model_dump_json(), encoding="utf-8")

    cancelled = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
        cancellation_check=lambda: True,
    )

    assert isinstance(cancelled, TextureFinalizationResult)
    assert cancelled.status == "cancelled"
    resumed = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    assert isinstance(resumed, TextureStepObservation)
    assert resumed.action == "execute"
    assert resumed.checkpoint_revision > patch.checkpoint_revision
    advanced = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
        decision_patch=_decision_for(resumed),
    )
    assert isinstance(advanced, TextureStepObservation)
    assert advanced.action == "validate"
    checkpoint = TextureWorkflowCheckpointStore(tmp_path).load()
    assert verify_texture_resume_decision_state(checkpoint, output_dir=tmp_path) is None
    ledger = TextureDecisionLedger.model_validate_json(
        (tmp_path / "texture_decision_ledger.json").read_text(encoding="utf-8")
    )
    assert [record.action for record in ledger.records] == ["execute", "execute"]
    assert len(client.execution_calls) == 1


def test_cancellation_precedes_validation_of_unpersisted_decision_patch(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    client = MockTexturePlannerExecutorClient(unit_count=1)
    validator = MockTextureSceneValidator()
    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    assert isinstance(observation, TextureStepObservation)
    stale_patch = _decision_for(observation).model_copy(
        update={"checkpoint_revision": observation.checkpoint_revision + 1}
    )

    cancelled = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
        decision_patch=stale_patch,
        cancellation_check=lambda: True,
    )

    assert isinstance(cancelled, TextureFinalizationResult)
    assert cancelled.status == "cancelled"
    checkpoint = TextureWorkflowCheckpointStore(tmp_path).load()
    assert checkpoint.terminal_status == "cancelled"
    assert not (tmp_path / "texture_decision_ledger.json").exists()
    assert client.execution_calls == []


def test_record_texture_decision_patch_serializes_with_checkpoint_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=MockTexturePlannerExecutorClient(unit_count=1),
        validator=MockTextureSceneValidator(),
    )
    assert isinstance(observation, TextureStepObservation)
    patch = _decision_for(observation)
    attempted = Event()
    lock_path = tmp_path / ".texture_workflow.lock"

    class _NotifyingFileLock:
        def __init__(self, path: str) -> None:
            assert Path(path) == lock_path
            self._lock = FileLock(path)

        def __enter__(self) -> object:
            attempted.set()
            return self._lock.__enter__()

        def __exit__(self, *args: object) -> None:
            self._lock.__exit__(*args)

    monkeypatch.setattr(texture_decision_module, "FileLock", _NotifyingFileLock)
    ledger_path = tmp_path / "texture_decision_ledger.json"
    with ThreadPoolExecutor(max_workers=1) as pool:
        with FileLock(str(lock_path)):
            future = pool.submit(
                record_texture_decision_patch,
                patch,
                output_dir=tmp_path,
            )
            assert attempted.wait(timeout=1)
            assert not future.done()
            assert not ledger_path.exists()
        assert future.result(timeout=3) == ledger_path

    ledger = TextureDecisionLedger.model_validate_json(
        ledger_path.read_text(encoding="utf-8")
    )
    assert [record.checkpoint_revision for record in ledger.records] == [
        observation.checkpoint_revision
    ]


def test_skill_routed_step_serializes_decision_and_checkpoint_transition(
    tmp_path: Path,
) -> None:
    class _BlockingFirstExecutionClient(MockTexturePlannerExecutorClient):
        def __init__(self) -> None:
            super().__init__(unit_count=1)
            self.first_execution_started = Event()
            self.release_first_execution = Event()
            self.second_execution_started = Event()
            self._invocation_lock = Lock()
            self._execution_invocations = 0

        def execute(self, *args: Any, **kwargs: Any) -> TextureExecutionResult:
            with self._invocation_lock:
                self._execution_invocations += 1
                invocation = self._execution_invocations
            if invocation == 1:
                self.first_execution_started.set()
                assert self.release_first_execution.wait(timeout=3)
            else:
                self.second_execution_started.set()
            return super().execute(*args, **kwargs)

    request = _request(tmp_path)
    client = _BlockingFirstExecutionClient()
    validator = MockTextureSceneValidator()
    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    assert isinstance(observation, TextureStepObservation)
    patch = _decision_for(observation)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            run_texture_workflow_step,
            request,
            mode="batch",
            client=client,
            validator=validator,
            decision_patch=patch,
        )
        assert client.first_execution_started.wait(timeout=3)
        second = pool.submit(
            run_texture_workflow_step,
            request,
            mode="batch",
            client=client,
            validator=validator,
            decision_patch=patch,
        )
        assert not client.second_execution_started.wait(timeout=0.2)
        assert not second.done()
        client.release_first_execution.set()
        next_observation = first.result(timeout=3)
        with pytest.raises(
            TextureWorkflowRuntimeError,
            match="current durable state",
        ):
            second.result(timeout=3)

    assert isinstance(next_observation, TextureStepObservation)
    assert next_observation.action == "validate"
    assert len(client.execution_calls) == 1


def test_skill_routed_step_with_custom_store_holds_canonical_ledger_lock(
    tmp_path: Path,
) -> None:
    class _NotifyingExecutionClient(MockTexturePlannerExecutorClient):
        def __init__(self) -> None:
            super().__init__(unit_count=1)
            self.execution_started = Event()

        def execute(self, *args: Any, **kwargs: Any) -> TextureExecutionResult:
            self.execution_started.set()
            return super().execute(*args, **kwargs)

    request = _request(tmp_path / "run")
    store = TextureWorkflowCheckpointStore(tmp_path / "durable-state")
    client = _NotifyingExecutionClient()
    validator = MockTextureSceneValidator()
    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
        checkpoint_store=store,
    )
    assert isinstance(observation, TextureStepObservation)

    canonical_lock = FileLock(str(request.output_dir / ".texture_workflow.lock"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        with canonical_lock:
            future = pool.submit(
                run_texture_workflow_step,
                request,
                mode="batch",
                client=client,
                validator=validator,
                decision_patch=_decision_for(observation),
                checkpoint_store=store,
            )
            assert not client.execution_started.wait(timeout=0.2)
            assert not future.done()
            assert client.execution_calls == []
        next_observation = future.result(timeout=3)

    assert isinstance(next_observation, TextureStepObservation)
    assert next_observation.action == "validate"
    assert len(client.execution_calls) == 1


def test_skill_routed_finalizer_rejects_tampered_prior_decision(
    tmp_path: Path,
) -> None:
    client = MockTexturePlannerExecutorClient(unit_count=1)
    validator = MockTextureSceneValidator()
    request = _request(tmp_path)
    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    assert isinstance(observation, TextureStepObservation)

    while observation.action != "finalize":
        observation = run_texture_workflow_step(
            request,
            mode="batch",
            client=client,
            validator=validator,
            decision_patch=_decision_for(observation),
        )
        assert isinstance(observation, TextureStepObservation)

    first_patch = next((tmp_path / "decisions").glob("*-execute-decision.json"))
    first_patch.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        TextureWorkflowRuntimeError,
        match="decision patch changed after acceptance",
    ):
        run_texture_workflow_step(
            request,
            mode="batch",
            client=client,
            validator=validator,
            decision_patch=_decision_for(observation),
        )
    assert not (tmp_path / "final_summary.json").exists()


def test_skill_routed_remote_source_uses_plan_digest_through_finalization(
    tmp_path: Path,
) -> None:
    source_digest = "a" * 64
    request = TextureWorkflowRequest(
        source_asset="https://assets.example.test/board.usdz",
        output_dir=tmp_path,
        intent="Generate surface-only board textures.",
    )
    plan_payload = (
        MockTexturePlannerExecutorClient(unit_count=1)
        .plan(request)
        .model_dump(mode="json")
    )
    plan_payload["request"]["source"]["source_asset_sha256"] = source_digest
    client = MockTexturePlannerExecutorClient(
        plan_document=TexturePlanDocument.model_validate(plan_payload)
    )
    validator = MockTextureSceneValidator()
    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    assert isinstance(observation, TextureStepObservation)
    assert observation.source_identity_digest == source_digest

    while isinstance(observation, TextureStepObservation):
        observation = run_texture_workflow_step(
            request,
            mode="batch",
            client=client,
            validator=validator,
            decision_patch=_decision_for(observation),
        )

    assert observation.success
    assert observation.status == "pass"


def test_skill_routed_cancelled_action_resumes_with_fresh_decision(
    tmp_path: Path,
) -> None:
    class CancelFirstExecutionClient(MockTexturePlannerExecutorClient):
        cancel_next_execution = True

        def execute(self, *args: Any, **kwargs: Any) -> TextureExecutionResult:
            if self.cancel_next_execution:
                self.cancel_next_execution = False
                raise TextureAgentServiceCancellationRequested
            return super().execute(*args, **kwargs)

    client = CancelFirstExecutionClient(unit_count=1)
    validator = MockTextureSceneValidator()
    request = _request(tmp_path)
    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    assert isinstance(observation, TextureStepObservation)
    cancelled = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
        decision_patch=_decision_for(observation),
    )
    assert isinstance(cancelled, TextureFinalizationResult)
    assert cancelled.status == "cancelled"

    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    assert isinstance(observation, TextureStepObservation)
    assert observation.action == "execute"
    while isinstance(observation, TextureStepObservation):
        observation = run_texture_workflow_step(
            request,
            mode="batch",
            client=client,
            validator=validator,
            decision_patch=_decision_for(observation),
        )
    assert observation.success
    assert observation.decision_ledger_path is not None
    ledger = TextureDecisionLedger.model_validate_json(
        Path(observation.decision_ledger_path).read_text(encoding="utf-8")
    )
    assert [record.action for record in ledger.records] == [
        "execute",
        "execute",
        "validate",
        "finalize",
    ]


def test_skill_routed_step_rejects_stale_patch_before_mutation(
    tmp_path: Path,
) -> None:
    client = MockTexturePlannerExecutorClient(unit_count=1)
    request = _request(tmp_path)
    observation = run_texture_workflow_step(
        request,
        mode="interactive",
        client=client,
        validator=MockTextureSceneValidator(),
    )
    assert isinstance(observation, TextureStepObservation)
    patch = _decision_for(observation).model_copy(
        update={"checkpoint_revision": observation.checkpoint_revision + 1}
    )

    with pytest.raises(TextureWorkflowRuntimeError, match="checkpoint_revision"):
        run_texture_workflow_step(
            request,
            mode="interactive",
            client=client,
            validator=MockTextureSceneValidator(),
            decision_patch=patch,
        )

    assert client.execution_calls == []


def test_skill_routed_completed_run_rejects_replayed_decision_patch(
    tmp_path: Path,
) -> None:
    client = MockTexturePlannerExecutorClient(unit_count=1)
    validator = MockTextureSceneValidator()
    request = _request(tmp_path)
    outcome = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    final_patch: TextureDecisionPatch | None = None
    while isinstance(outcome, TextureStepObservation):
        final_patch = _decision_for(outcome)
        outcome = run_texture_workflow_step(
            request,
            mode="batch",
            client=client,
            validator=validator,
            decision_patch=final_patch,
        )
    assert outcome.status == "pass"
    assert final_patch is not None

    with pytest.raises(
        TextureWorkflowRuntimeError,
        match="does not accept another decision patch",
    ):
        run_texture_workflow_step(
            request,
            mode="batch",
            client=client,
            validator=validator,
            decision_patch=final_patch,
        )


def test_texture_observation_rejects_evidence_outside_run_directory(
    tmp_path: Path,
) -> None:
    client = MockTexturePlannerExecutorClient(unit_count=1)
    validator = MockTextureSceneValidator()
    request = _request(tmp_path / "run")
    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
    )
    assert isinstance(observation, TextureStepObservation)
    observation = run_texture_workflow_step(
        request,
        mode="batch",
        client=client,
        validator=validator,
        decision_patch=_decision_for(observation),
    )
    assert isinstance(observation, TextureStepObservation)
    checkpoint = TextureWorkflowCheckpointStore(request.output_dir).load()
    outside = tmp_path / "outside.usdz"
    outside.write_bytes(b"outside")
    escaped = checkpoint.model_copy(update={"output_asset_path": str(outside)})

    with pytest.raises(ValueError, match="escapes the run directory"):
        build_texture_step_observation(
            escaped,
            output_dir=request.output_dir,
        )


def test_batch_workflow_regenerates_exact_failure_and_preserves_accepts(
    tmp_path: Path,
) -> None:
    client = MockTexturePlannerExecutorClient(unit_count=3)
    failing_id = client.unit_ids[1]
    validator = MockTextureSceneValidator(failure_schedule=[(failing_id,), ()])
    finalizer = _RecordingFinalizer()
    events: list[TextureWorkflowProgress] = []

    result = run_batch_texture_workflow(
        _request(tmp_path),
        client=client,
        validator=validator,
        finalizer=finalizer,
        progress_callback=events.append,
    )

    assert result.success
    assert result.status == "pass"
    assert result.accepted_unit_ids == client.unit_ids
    assert result.remaining_unit_ids == ()
    assert [call.unit_ids for call in client.execution_calls] == [
        client.unit_ids,
        (failing_id,),
    ]
    assert client.execution_calls[1].preserved_unit_ids == (
        client.unit_ids[0],
        client.unit_ids[2],
    )
    assert [call.unit_ids for call in validator.calls] == [
        client.unit_ids,
        (failing_id,),
    ]

    payload = finalizer.inputs[0]
    initial_artifacts = {
        item.unit_id: item for item in payload.executions[0].unit_artifacts
    }
    assert (
        payload.unit_artifacts[client.unit_ids[0]]
        == initial_artifacts[client.unit_ids[0]]
    )
    assert (
        payload.unit_artifacts[client.unit_ids[2]]
        == initial_artifacts[client.unit_ids[2]]
    )
    assert payload.unit_artifacts[failing_id].generation == 2

    assert events[-1].phase == "completed"
    assert events[-1].accepted_unit_count == 3
    assert events[-1].remaining_unit_count == 0
    assert all(
        set(event.accepted_unit_ids) | set(event.remaining_unit_ids)
        == set(event.selected_unit_ids)
        for event in events
    )

    for artifact_path in (
        result.request_path,
        result.texture_plan_path,
        result.execution_summary_path,
        result.visual_quality_assessment_path,
        result.validation_evidence_path,
        result.workflow_progress_path,
        result.final_summary_path,
        result.output_asset_path,
    ):
        assert Path(artifact_path).is_file()

    plan = json.loads(Path(result.texture_plan_path).read_text(encoding="utf-8"))
    assert plan["schema_version"] == "texture-agent-plan.v1"
    assert [unit["unit_id"] for unit in plan["selected_units"]] == list(client.unit_ids)
    evidence = json.loads(
        Path(result.validation_evidence_path).read_text(encoding="utf-8")
    )
    assert evidence["selected_unit_count"] == 3
    assert evidence["backend_job_count"] == 4
    assert evidence["remaining_unit_ids"] == []
    assert (
        evidence["schema_version"]
        == "content-agent-workflows.texture-validation-evidence.v3"
    )
    assert (
        evidence["output_asset_sha256"]
        == hashlib.sha256(Path(result.output_asset_path).read_bytes()).hexdigest()
    )
    summary = json.loads(Path(result.final_summary_path).read_text(encoding="utf-8"))
    assert summary["schema_version"] == "content-agent-workflows.texture-summary.v3"
    assert summary["output_asset_sha256"] == evidence["output_asset_sha256"]


def test_interactive_and_batch_use_same_contracts_and_finalizer(
    tmp_path: Path,
) -> None:
    captured: list[TextureFinalizerInput] = []

    for mode, runner in (
        ("interactive", run_interactive_texture_workflow),
        ("batch", run_batch_texture_workflow),
    ):
        client = MockTexturePlannerExecutorClient(unit_count=1)
        validator = MockTextureSceneValidator()
        finalizer = _RecordingFinalizer()

        result = runner(
            _request(tmp_path / mode),
            client=client,
            validator=validator,
            finalizer=finalizer,
        )

        assert result.mode == mode
        assert result.success
        captured.append(finalizer.inputs[0])

    assert type(captured[0].request) is type(captured[1].request)
    assert type(captured[0].plan) is type(captured[1].plan)
    assert type(captured[0].executions[0]) is type(captured[1].executions[0])
    assert type(captured[0].validations[0]) is type(captured[1].validations[0])
    assert type(captured[0]) is type(captured[1])


def test_non_cancelled_finalizer_input_requires_terminal_evidence(
    tmp_path: Path,
) -> None:
    finalizer = _RecordingFinalizer()
    run_batch_texture_workflow(
        _request(tmp_path),
        client=MockTexturePlannerExecutorClient(unit_count=1),
        validator=MockTextureSceneValidator(),
        finalizer=finalizer,
    )
    payload = finalizer.inputs[0].model_dump(mode="python")

    for field_name, empty_value, message in (
        ("executions", (), "execution evidence"),
        ("validations", (), "validation evidence"),
        ("output_asset_path", None, "output asset path"),
    ):
        invalid_payload = {**payload, field_name: empty_value}
        with pytest.raises(ValueError, match=message):
            TextureFinalizerInput.model_validate(invalid_payload)


def test_texture_validation_evidence_requires_paired_output_identity() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        TextureWorkflowValidationEvidence(
            target_runtime="test",
            status="cancelled",
            selected_unit_ids=(),
            accepted_unit_ids=(),
            remaining_unit_ids=(),
            selected_unit_count=0,
            backend_job_count=0,
            cache_hit_count=0,
            retry_count=0,
            output_asset_path="/tmp/missing.usdz",
            unit_artifact_paths={},
            visual_evidence_paths=(),
        )


def test_cancelled_finalizer_clears_stale_output_identity_after_cleanup(
    tmp_path: Path,
) -> None:
    recording = _RecordingFinalizer()
    run_batch_texture_workflow(
        _request(tmp_path),
        client=MockTexturePlannerExecutorClient(unit_count=1),
        validator=MockTextureSceneValidator(),
        finalizer=recording,
    )
    payload = recording.inputs[0].model_copy(
        update={
            "terminal_status": "cancelled",
            "cancellation_reason": "operator cancelled after output cleanup",
        }
    )
    assert payload.output_asset_path is not None
    assert payload.output_asset_sha256 is not None
    Path(payload.output_asset_path).unlink()

    result = CanonicalTextureWorkflowFinalizer().finalize(payload)

    assert result.status == "cancelled"
    evidence = json.loads(
        Path(result.validation_evidence_path).read_text(encoding="utf-8")
    )
    assert evidence["output_asset_path"] is None
    assert evidence["output_asset_sha256"] is None


def test_passing_finalizer_rejects_missing_output_even_with_checkpoint_digest(
    tmp_path: Path,
) -> None:
    recording = _RecordingFinalizer()
    run_batch_texture_workflow(
        _request(tmp_path),
        client=MockTexturePlannerExecutorClient(unit_count=1),
        validator=MockTextureSceneValidator(),
        finalizer=recording,
    )
    payload = recording.inputs[0]
    assert payload.output_asset_path is not None
    assert payload.output_asset_sha256 is not None
    Path(payload.output_asset_path).unlink()

    with pytest.raises(FileNotFoundError, match="output asset does not exist"):
        CanonicalTextureWorkflowFinalizer().finalize(payload)


def test_passing_finalizer_persists_resolved_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = _RecordingFinalizer()
    run_batch_texture_workflow(
        _request(tmp_path),
        client=MockTexturePlannerExecutorClient(unit_count=1),
        validator=MockTextureSceneValidator(),
        finalizer=recording,
    )
    payload = recording.inputs[0]
    output_path = Path(payload.output_asset_path or "")
    monkeypatch.chdir(tmp_path)
    relative_output_path = output_path.relative_to(tmp_path)

    result = CanonicalTextureWorkflowFinalizer().finalize(
        payload.model_copy(update={"output_asset_path": str(relative_output_path)})
    )

    resolved_output_path = str(output_path.resolve())
    evidence = json.loads(
        Path(result.validation_evidence_path).read_text(encoding="utf-8")
    )
    summary = json.loads(Path(result.final_summary_path).read_text(encoding="utf-8"))
    assert evidence["output_asset_path"] == resolved_output_path
    assert summary["output_asset_path"] == resolved_output_path
    assert result.output_asset_path == resolved_output_path


def test_conditional_finalizer_rejects_missing_output_after_cleanup(
    tmp_path: Path,
) -> None:
    recording = _RecordingFinalizer()
    run_batch_texture_workflow(
        _request(tmp_path),
        client=MockTexturePlannerExecutorClient(unit_count=1),
        validator=MockTextureSceneValidator(),
        finalizer=recording,
    )
    payload = recording.inputs[0].model_copy(
        update={
            "terminal_status": "conditional",
            "accepted_unit_ids": (),
            "remaining_unit_ids": recording.inputs[0].plan.selected_unit_ids,
        }
    )
    assert payload.output_asset_path is not None
    assert payload.output_asset_sha256 is not None
    Path(payload.output_asset_path).unlink()

    with pytest.raises(FileNotFoundError, match="output asset does not exist"):
        CanonicalTextureWorkflowFinalizer().finalize(payload)


def test_cancelled_finalizer_clears_missing_unbound_output_path(tmp_path: Path) -> None:
    recording = _RecordingFinalizer()
    run_batch_texture_workflow(
        _request(tmp_path),
        client=MockTexturePlannerExecutorClient(unit_count=1),
        validator=MockTextureSceneValidator(),
        finalizer=recording,
    )
    payload = recording.inputs[0].model_copy(
        update={
            "terminal_status": "cancelled",
            "cancellation_reason": "cancelled before an output receipt was sealed",
            "output_asset_sha256": None,
        }
    )
    assert payload.output_asset_path is not None
    Path(payload.output_asset_path).unlink()

    result = CanonicalTextureWorkflowFinalizer().finalize(payload)

    assert result.output_asset_path is None
    evidence = json.loads(
        Path(result.validation_evidence_path).read_text(encoding="utf-8")
    )
    assert evidence["output_asset_path"] is None
    assert evidence["output_asset_sha256"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("request", "finalizer mutated the request identity"),
        ("output_dir", "finalizer mutated the output directory"),
        ("plan", "finalizer mutated the immutable plan"),
    ],
)
def test_workflow_rejects_finalizer_identity_mutation(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    redirected_output_dir = tmp_path / "redirected"

    class MutatingFinalizer:
        def __init__(self) -> None:
            self._delegate = CanonicalTextureWorkflowFinalizer()

        def finalize(
            self,
            payload: TextureFinalizerInput,
        ) -> TextureFinalizationResult:
            result = self._delegate.finalize(payload)
            if mutation == "request":
                payload.request.intent = "mutated after finalization"
            elif mutation == "output_dir":
                payload.request.output_dir = redirected_output_dir
            else:
                execution = (payload.plan.model_extra or {}).get("execution")
                assert isinstance(execution, dict)
                execution["max_concurrency"] = 99
            return result

    with pytest.raises(TextureWorkflowRuntimeError, match=message):
        run_batch_texture_workflow(
            _request(tmp_path / mutation),
            client=MockTexturePlannerExecutorClient(unit_count=1),
            validator=MockTextureSceneValidator(),
            finalizer=MutatingFinalizer(),
        )

    assert not redirected_output_dir.exists()


def test_bounded_vqa_preserves_partial_result_at_iteration_cap(
    tmp_path: Path,
) -> None:
    client = MockTexturePlannerExecutorClient(unit_count=2)
    failing_id = client.unit_ids[1]
    validator = MockTextureSceneValidator(
        failure_schedule=[(failing_id,), (failing_id,)]
    )
    finalizer = _RecordingFinalizer()

    result = run_interactive_texture_workflow(
        _request(tmp_path, max_vqa_iterations=1),
        client=client,
        validator=validator,
        finalizer=finalizer,
    )

    assert not result.success
    assert result.status == "conditional"
    assert result.accepted_unit_ids == (client.unit_ids[0],)
    assert result.remaining_unit_ids == (failing_id,)
    assert client.execution_calls[1].unit_ids == (failing_id,)
    assert client.execution_calls[1].preserved_unit_ids == (client.unit_ids[0],)
    assert finalizer.inputs[0].unit_artifacts[client.unit_ids[0]].generation == 1


def test_non_executable_plan_stops_before_executor_work(tmp_path: Path) -> None:
    source_client = MockTexturePlannerExecutorClient(unit_count=1)
    plan = source_client.plan(_request(tmp_path / "source"))
    blocked_plan = plan.model_copy(
        update={
            "decision": TexturePlanDecision(
                state="requires_operator_override",
                execution_allowed=False,
            )
        }
    )
    client = MockTexturePlannerExecutorClient(plan_document=blocked_plan)

    with pytest.raises(RuntimeError, match="not executable"):
        run_batch_texture_workflow(
            _request(tmp_path / "run"),
            client=client,
            validator=MockTextureSceneValidator(),
        )

    assert client.execution_calls == []
    assert (tmp_path / "run" / "request.json").is_file()
    assert (tmp_path / "run" / "texture_plan.json").is_file()


def test_mock_scene_validator_rejects_failure_outside_exact_scope(
    tmp_path: Path,
) -> None:
    client = MockTexturePlannerExecutorClient(unit_count=2)
    validator = MockTextureSceneValidator(
        failure_schedule=[("tu_00000000000000000000",)]
    )

    with pytest.raises(ValueError, match="evaluated unit IDs"):
        run_batch_texture_workflow(
            _request(tmp_path),
            client=client,
            validator=validator,
        )


def test_workflow_rejects_validator_response_with_narrower_scope(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="validation response scope"):
        run_batch_texture_workflow(
            _request(tmp_path),
            client=MockTexturePlannerExecutorClient(unit_count=2),
            validator=_WrongScopeValidator(),
        )


@pytest.mark.parametrize(
    ("stale_field", "message"),
    [
        ("iteration", "response iteration"),
        ("output_asset_path", "candidate output asset"),
    ],
)
def test_workflow_rejects_validator_response_for_stale_candidate(
    tmp_path: Path,
    stale_field: str,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        run_batch_texture_workflow(
            _request(tmp_path),
            client=MockTexturePlannerExecutorClient(unit_count=1),
            validator=_StaleCandidateValidator(stale_field),
        )


def test_progress_rejects_duplicate_partition_ids() -> None:
    unit_id = "tu_00000000000000000000"

    with pytest.raises(ValueError, match="accepted_unit_ids must be unique"):
        TextureWorkflowProgress.build(
            mode="batch",
            phase="validating",
            selected_unit_ids=(unit_id,),
            accepted_unit_ids=(unit_id, unit_id),
            remaining_unit_ids=(),
            message="invalid duplicate progress",
        )


def test_workflow_scope_guards_reject_bad_adapter_payloads() -> None:
    first = "tu_00000000000000000000"
    second = "tu_00000000000000000001"
    artifact = TextureUnitArtifact(unit_id=first, artifact_paths=("/tmp/a.png",))
    execution = TextureExecutionResult.model_construct(
        requested_unit_ids=(first, second),
        unit_artifacts=(artifact,),
        output_asset_path="/tmp/out.usda",
        cache_hit_unit_ids=(),
        retry_count=0,
        metadata={},
    )

    with pytest.raises(RuntimeError, match="artifacts differ"):
        _require_execution_scope(execution, (first, second))

    class _DuplicateFailedValidation:
        evaluated_unit_ids = (first,)
        failed_unit_ids = (first, first)

    with pytest.raises(RuntimeError, match="duplicate failed"):
        _require_validation_scope(  # type: ignore[arg-type]
            _DuplicateFailedValidation(),
            (first,),
            output_asset_path="/tmp/out.usda",
            iteration=0,
        )

    class _UnknownFailedValidation:
        evaluated_unit_ids = (first,)
        failed_unit_ids = (second,)

    with pytest.raises(RuntimeError, match="outside the requested scope"):
        _require_validation_scope(  # type: ignore[arg-type]
            _UnknownFailedValidation(),
            (first,),
            output_asset_path="/tmp/out.usda",
            iteration=0,
        )


def test_exact_provider_plan_coverage_is_embedded_only(tmp_path: Path) -> None:
    request = TextureWorkflowRequest(
        source_asset="/assets/board.usda",
        output_dir=tmp_path,
        intent="Texture the requested materials.",
        metadata={
            "explicit_material_paths": [
                "/World/Looks/Paint",
                "/World/Looks/Trim",
            ]
        },
    )
    partial_plan = TexturePlanDocument.model_validate(
        {
            "counts": {"selected_unit_count": 1},
            "selected_units": [
                {
                    "unit_id": "tu_00000000000000000000",
                    "material_prim_paths": ["/World/Looks/Paint"],
                }
            ],
            "decision": {"state": "ready", "execution_allowed": True},
        }
    )

    _require_plan_request_scope(request, partial_plan)
    with pytest.raises(RuntimeError, match="exactly cover explicit material scope"):
        _require_plan_request_scope(request, partial_plan, exact=True)
