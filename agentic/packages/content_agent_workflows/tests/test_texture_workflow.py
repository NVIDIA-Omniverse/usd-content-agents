# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bounded mock texture workflow skeleton."""

from __future__ import annotations

import json
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
import requests

import content_agent_workflows.texture.client as texture_client_module
from content_agent_workflows.texture import (
    CanonicalTextureWorkflowFinalizer,
    MockTexturePlannerExecutorClient,
    MockWorkbenchTextureValidator,
    TextureAgentServiceClient,
    TextureExecutionResult,
    TextureFinalizationResult,
    TextureFinalizerInput,
    TexturePlanDecision,
    TextureUnitArtifact,
    TextureValidationFinding,
    TextureValidationResult,
    TextureWorkflowProgress,
    TextureWorkflowRequest,
    run_batch_texture_workflow,
    run_interactive_texture_workflow,
)
from content_agent_workflows.texture.workflow import (
    _require_execution_scope,
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

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.posts.append((url, kwargs))
        if url.endswith("/pipeline"):
            return _FakeResponse({"session_id": "session-466"})
        if url.endswith("/regenerate"):
            return _FakeResponse({"session_id": "session-466", "status": "pending"})
        raise AssertionError(url)

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        if url.endswith("/status"):
            self.status_request_count += 1
            self.status_request_timeouts.append(kwargs["timeout"])
            if self.status_responses:
                return self.status_responses.pop(0)
            return _FakeResponse({"status": "completed"})
        if url.endswith("/plan"):
            return _FakeResponse(self.plan)
        if url.endswith("/results"):
            return _FakeResponse({"stats": {"cache_hit_unit_ids": []}})
        if url.endswith("/textures"):
            return _FakeResponse(content=self.texture_zip)
        if url.endswith("/output"):
            return _FakeResponse(content=b"mock-usdz")
        raise AssertionError(url)


def _texture_zip(unit_ids: tuple[str, ...]) -> bytes:
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for unit_id in unit_ids:
            archive.writestr(f"{unit_id}_albedo.png", b"png")
            archive.writestr(f"textures/{unit_id}/albedo.png", b"png")
    return stream.getvalue()


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
        output_asset_path: str,
        unit_artifacts: Mapping[str, TextureUnitArtifact],
        unit_ids: tuple[str, ...],
        iteration: int,
        output_dir: Path,
    ) -> TextureValidationResult:
        del unit_artifacts, output_dir
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


def _request(tmp_path: Path, *, max_vqa_iterations: int = 2) -> TextureWorkflowRequest:
    return TextureWorkflowRequest(
        source_asset="/assets/board.usda",
        output_dir=tmp_path,
        intent="Generate surface-only board textures.",
        max_vqa_iterations=max_vqa_iterations,
    )


def test_real_service_adapter_uses_plan_only_and_exact_regeneration(
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
        "apply_textures",
    ]
    assert regeneration_post[1]["json"]["texture_unit_ids"] == list(
        plan.selected_unit_ids
    )
    assert result.requested_unit_ids == plan.selected_unit_ids
    assert [artifact.unit_id for artifact in result.unit_artifacts] == list(
        plan.selected_unit_ids
    )
    assert Path(result.output_asset_path).read_bytes() == b"mock-usdz"
    assert all(
        any(unit_id in Path(path).parts for path in artifact.artifact_paths)
        for unit_id, artifact in zip(
            plan.selected_unit_ids, result.unit_artifacts, strict=True
        )
    )


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
    )
    monotonic_values = iter((100.0, 102.0))
    monkeypatch.setattr(
        texture_client_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    client = TextureAgentServiceClient(
        base_url="http://texture.test",
        timeout_seconds=10,
        poll_interval_seconds=0,
        session=fake_http,
    )

    assert client.plan(request).selected_unit_ids == fixture_plan.selected_unit_ids
    assert fake_http.status_request_timeouts == [8.0]


def test_batch_workflow_regenerates_exact_failure_and_preserves_accepts(
    tmp_path: Path,
) -> None:
    client = MockTexturePlannerExecutorClient(unit_count=3)
    failing_id = client.unit_ids[1]
    validator = MockWorkbenchTextureValidator(failure_schedule=[(failing_id,), ()])
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


def test_interactive_and_batch_use_same_contracts_and_finalizer(
    tmp_path: Path,
) -> None:
    captured: list[TextureFinalizerInput] = []

    for mode, runner in (
        ("interactive", run_interactive_texture_workflow),
        ("batch", run_batch_texture_workflow),
    ):
        client = MockTexturePlannerExecutorClient(unit_count=1)
        validator = MockWorkbenchTextureValidator()
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


def test_bounded_vqa_preserves_partial_result_at_iteration_cap(
    tmp_path: Path,
) -> None:
    client = MockTexturePlannerExecutorClient(unit_count=2)
    failing_id = client.unit_ids[1]
    validator = MockWorkbenchTextureValidator(
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
            validator=MockWorkbenchTextureValidator(),
        )

    assert client.execution_calls == []
    assert (tmp_path / "run" / "request.json").is_file()
    assert (tmp_path / "run" / "texture_plan.json").is_file()


def test_mock_workbench_rejects_failure_outside_exact_scope(tmp_path: Path) -> None:
    client = MockTexturePlannerExecutorClient(unit_count=2)
    validator = MockWorkbenchTextureValidator(
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
        _require_validation_scope(_DuplicateFailedValidation(), (first,))  # type: ignore[arg-type]

    class _UnknownFailedValidation:
        evaluated_unit_ids = (first,)
        failed_unit_ids = (second,)

    with pytest.raises(RuntimeError, match="outside the requested scope"):
        _require_validation_scope(_UnknownFailedValidation(), (first,))  # type: ignore[arg-type]
