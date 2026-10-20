# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared model endpoint record and preflight."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from world_understanding.agentic import base_pipeline_executor as base_module
from world_understanding.agentic import model_endpoints
from world_understanding.agentic.model_endpoints import (
    ActiveModel,
    ModelEndpointPreflightError,
    describe_model_endpoints,
    endpoint_host,
    public_endpoint_records,
    resolve_provisioned_model_config,
    run_model_endpoint_preflight,
)
from world_understanding.functions.models import request_timeout

SECRET = "s3cret-token-value"
LEAKY_URL = f"https://user:{SECRET}@hub.example:8443/v1?api_key={SECRET}"


@pytest.fixture(autouse=True)
def _no_nim_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in ("WU", "PA", "TA", "MA"):
        monkeypatch.delenv(f"{prefix}_VLM_NIM_BASE_URL", raising=False)
        monkeypatch.delenv(f"{prefix}_LLM_NIM_BASE_URL", raising=False)


@pytest.mark.parametrize(
    ("backend", "configured_url", "expected"),
    (
        ("nim", None, "integrate.api.nvidia.com"),
        ("openai", "  ", "api.openai.com"),
        ("custom", None, "provider_default"),
        ("openai", "https://inference.example/v1", "inference.example"),
        ("openai", LEAKY_URL, "hub.example:8443"),
        ("openai", "hub.example/v1", "configured_endpoint"),
        ("openai", "http://[::1", "configured_endpoint"),
    ),
)
def test_endpoint_host_never_carries_userinfo_path_or_query(
    backend: str, configured_url: object, expected: str
) -> None:
    # Not named ``base_url``: pytest-base-url provides a session fixture of that name.
    host = endpoint_host(backend, configured_url)
    assert host == expected
    assert SECRET not in host


def test_records_use_the_agent_resolved_config_and_the_effective_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WU_VLM_NIM_BASE_URL", "https://nim.override.example/v1")
    raw = {"backend": "nim", "model": "m"}
    as_given = describe_model_endpoints([ActiveModel("s", "vlm", "vlm", raw)])
    provisioned = describe_model_endpoints(
        [ActiveModel("s", "vlm", "vlm", resolve_provisioned_model_config("vlm", raw))]
    )
    assert as_given[0]["endpoint_host"] == "integrate.api.nvidia.com"
    assert provisioned[0]["endpoint_host"] == "nim.override.example"
    assert as_given[0]["timeout_seconds"] == (
        request_timeout.DEFAULT_HOSTED_NIM_REQUEST_TIMEOUT_SECONDS
    )
    assert provisioned[0]["timeout_seconds"] == (
        request_timeout.DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS
    )
    # An empty base URL means the default hosted endpoint.
    blank = describe_model_endpoints(
        [ActiveModel("s", "vlm", "vlm", {**raw, "base_url": ""})]
    )
    assert blank[0]["timeout_seconds"] == (
        request_timeout.DEFAULT_HOSTED_NIM_REQUEST_TIMEOUT_SECONDS
    )
    public = public_endpoint_records(as_given)
    assert all(not key.startswith("_") for key in public[0])


class _Model:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append("generate")
        if self.error is not None:
            raise self.error
        return "OK"

    def invoke(self, prompt: str) -> str:
        self.calls.append("invoke")
        if self.error is not None:
            raise self.error
        return "OK"


def test_preflight_dispatches_by_interface_and_dedupes_by_full_route() -> None:
    vlm_models: list[dict[str, Any]] = []
    llm_models: list[dict[str, Any]] = []

    def create_vlm(config: dict[str, Any]) -> _Model:
        vlm_models.append(config)
        return _Model()

    def create_llm(config: dict[str, Any]) -> _Model:
        llm_models.append(config)
        return _Model()

    hub = {"backend": "openai", "model": "m", "base_url": "https://hub.example/v1"}
    other_path = dict(hub, base_url="https://hub.example/custom")
    records = describe_model_endpoints(
        [
            ActiveModel("a", "vlm", "vlm", hub),
            ActiveModel("b", "vlm", "vlm", hub),
            ActiveModel("c", "vlm", "vlm", other_path),
            ActiveModel("d", "llm", "llm", hub),
            ActiveModel("e", "vlm", "vlm", {"backend": "mock"}),
        ]
    )
    results = run_model_endpoint_preflight(
        records, create_vlm=create_vlm, create_llm=create_llm
    )
    assert [r["first_step"] for r in results] == ["a", "c", "d", "e"]
    assert [r["outcome"] for r in results] == [
        "answered",
        "answered",
        "answered",
        "not_probed",
    ]
    assert len(vlm_models) == 2
    assert len(llm_models) == 1


@pytest.mark.parametrize("stage", ["provision", "probe"])
@pytest.mark.parametrize("required", [True, False])
def test_only_required_endpoints_abort_the_preflight(
    stage: str, required: bool
) -> None:
    """An optional parsing LLM that cannot be built or reached must not abort."""
    records = describe_model_endpoints(
        [
            ActiveModel("predict", "vlm", "vlm", {"backend": "nim", "model": "v"}),
            ActiveModel(
                "predict",
                "llm",
                "llm",
                {"backend": "nim", "model": "l"},
                required=required,
            ),
        ]
    )

    def create_llm(config: dict[str, Any]) -> _Model:
        if stage == "provision":
            raise ValueError("API key is required for NIM backend")
        return _Model(TimeoutError("queued"))

    def run() -> list[dict[str, Any]]:
        return run_model_endpoint_preflight(
            records, create_vlm=lambda config: _Model(), create_llm=create_llm
        )

    assert [r["required"] for r in records] == [True, required]
    if required:
        with pytest.raises(ModelEndpointPreflightError):
            run()
        return
    results = run()
    assert [r["outcome"] for r in results] == ["answered", "optional_unavailable"]
    assert results[1]["unavailable_for"] == [{"step": "predict", "section": "llm"}]
    assert results[1]["error_type"] == (
        "ValueError" if stage == "provision" else "TimeoutError"
    )


def test_steps_drop_only_their_unavailable_optional_models() -> None:
    context = {
        base_module.UNAVAILABLE_OPTIONAL_MODELS_KEY: [("predict", "llm")],
    }
    drop = base_module.BasePipelineExecutor._without_unavailable_optional_models
    predict = {"vlm": {"backend": "nim"}, "llm": {"backend": "openai"}}
    identify = {"vlm": {"backend": "nim"}, "llm": {"backend": "openai"}}

    assert drop("predict", predict, context) == {"vlm": {"backend": "nim"}}
    assert drop("identify_asset", identify, context) == identify
    assert drop("predict", {"vlm": {}}, {}) == {"vlm": {}}


@pytest.mark.parametrize("required_answers", [True, False])
def test_probing_continues_after_an_optional_failure(required_answers: bool) -> None:
    """An optional failure listed first must not stop the required probes."""
    records = describe_model_endpoints(
        [
            ActiveModel(
                "predict",
                "llm",
                "llm",
                {"backend": "nim", "model": "l"},
                required=False,
            ),
            ActiveModel("predict", "vlm", "vlm", {"backend": "nim", "model": "v"}),
        ]
    )
    vlm = _Model(None if required_answers else TimeoutError("queued"))

    def run() -> list[dict[str, Any]]:
        return run_model_endpoint_preflight(
            records,
            create_vlm=lambda config: vlm,
            create_llm=lambda config: _Model(TimeoutError("queued")),
        )

    if required_answers:
        assert [r["outcome"] for r in run()] == ["optional_unavailable", "answered"]
    else:
        with pytest.raises(ModelEndpointPreflightError):
            run()
    assert vlm.calls == ["generate"]


def test_shared_endpoint_is_required_if_any_step_requires_it() -> None:
    """An optional record seen first must not mask a required one on the same route."""
    shared = {"backend": "nim", "model": "k3"}
    records = describe_model_endpoints(
        [
            ActiveModel("predict", "llm", "llm", shared, required=False),
            ActiveModel("analyze_structure", "llm", "llm", shared),
        ]
    )

    with pytest.raises(ModelEndpointPreflightError):
        run_model_endpoint_preflight(
            records,
            create_vlm=lambda config: pytest.fail("vlm not used"),
            create_llm=lambda config: _Model(TimeoutError("queued")),
        )


def test_preflight_failure_status_and_message_are_value_safe() -> None:
    records = describe_model_endpoints(
        [ActiveModel("a", "vlm", "vlm", {"backend": "openai", "base_url": LEAKY_URL})]
    )
    with pytest.raises(ModelEndpointPreflightError) as raised:
        run_model_endpoint_preflight(
            records,
            create_vlm=lambda config: _Model(RuntimeError(f"bad {LEAKY_URL}")),
            create_llm=lambda config: pytest.fail("llm not used"),
        )
    assert SECRET not in str(raised.value)
    assert raised.value.__cause__ is None
    assert SECRET not in json.dumps(raised.value.status)
    assert raised.value.status["endpoint_host"] == "hub.example:8443"
    assert raised.value.status["failed_step"] == "model_endpoint_preflight"
    assert "No pipeline step ran" in raised.value.status["recovery_action"]["detail"]


def test_preflight_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(model_endpoints.MODEL_ENDPOINT_PREFLIGHT_ENV, raising=False)
    assert model_endpoints.model_endpoint_preflight_enabled(True) is True
    monkeypatch.setenv(model_endpoints.MODEL_ENDPOINT_PREFLIGHT_ENV, "0")
    assert model_endpoints.model_endpoint_preflight_enabled(True) is False
    monkeypatch.setenv(model_endpoints.MODEL_ENDPOINT_PREFLIGHT_ENV, "1")
    assert model_endpoints.model_endpoint_preflight_enabled(False) is True


class _Executor(base_module.BasePipelineExecutor):
    def _execute_step(self, step_name, context, object_store):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def _get_step_list_key(self) -> str:
        return "steps"

    def _get_required_context_keys(self) -> list[str]:
        return []

    def _get_state_file(self, context: dict[str, Any]) -> Path:
        return Path(context["state_file"])


def _run_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    preflight: bool,
    model: _Model,
    completed: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    from world_understanding.agentic.domain_tasks import model_provisioning

    class FakeProvisioning:
        def create_vlm(self, config: dict[str, Any]) -> _Model:
            return model

        def _create_llm(self, config: dict[str, Any]) -> _Model:
            return model

    monkeypatch.setattr(model_provisioning, "ModelProvisioningTask", FakeProvisioning)
    state_file = tmp_path / ".pipeline_state.json"
    context: dict[str, Any] = {"model_endpoint_preflight": preflight}
    pipeline_state: dict[str, Any] = {
        "completed_steps": sorted(completed),
        "failed_steps": [],
        "step_outputs": {},
    }
    active = [
        ActiveModel(
            "identify", "vlm", "vlm", {"backend": "openai", "base_url": LEAKY_URL}
        ),
        ActiveModel("predict", "vlm", "vlm", {"backend": "openai", "model": "p"}),
    ]
    try:
        _Executor()._record_and_preflight_model_endpoints(
            context=context,
            active_models=active,
            pipeline_state=pipeline_state,
            state_file=state_file,
            completed_steps=completed,
        )
    finally:
        saved = json.loads(state_file.read_text(encoding="utf-8"))
    return context, saved


def test_hook_records_endpoints_and_probes_nothing_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _Model()
    _context, saved = _run_hook(tmp_path, monkeypatch, preflight=False, model=model)
    assert [r["step"] for r in saved["model_endpoints"]] == ["identify", "predict"]
    assert "model_endpoint_preflight" not in saved
    assert model.calls == []
    assert SECRET not in json.dumps(saved)


def test_hook_skips_completed_steps_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _Model()
    _context, saved = _run_hook(
        tmp_path,
        monkeypatch,
        preflight=True,
        model=model,
        completed=frozenset({"identify"}),
    )
    assert [r["first_step"] for r in saved["model_endpoint_preflight"]] == ["predict"]


def test_hook_persists_a_failed_preflight_without_leaking_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ModelEndpointPreflightError) as raised:
        _run_hook(
            tmp_path,
            monkeypatch,
            preflight=True,
            model=_Model(TimeoutError(f"queued at {LEAKY_URL}")),
        )
    saved = json.loads((tmp_path / ".pipeline_state.json").read_text(encoding="utf-8"))
    assert saved["terminal_failure"]["failed_step"] == "model_endpoint_preflight"
    assert saved["completed_steps"] == []
    assert SECRET not in json.dumps(saved)
    assert SECRET not in caplog.text
    assert SECRET not in str(raised.value)


def test_resume_clears_only_a_stale_preflight_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.agentic.domain_tasks import model_provisioning

    class FakeProvisioning:
        def create_vlm(self, config: dict[str, Any]) -> _Model:
            return _Model()

        def _create_llm(self, config: dict[str, Any]) -> _Model:
            return _Model()

    monkeypatch.setattr(model_provisioning, "ModelProvisioningTask", FakeProvisioning)
    active = [ActiveModel("predict", "vlm", "vlm", {"backend": "openai", "model": "p"})]
    stale = {"failed_step": "model_endpoint_preflight", "terminal": True}
    state_file = tmp_path / ".pipeline_state.json"
    context: dict[str, Any] = {
        "model_endpoint_preflight": True,
        "terminal_failure": stale,
    }
    pipeline_state: dict[str, Any] = {
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "terminal_failure": dict(stale),
        "model_endpoint_preflight": [dict(stale)],
    }
    _Executor()._record_and_preflight_model_endpoints(
        context=context,
        active_models=active,
        pipeline_state=pipeline_state,
        state_file=state_file,
    )
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert "terminal_failure" not in saved
    assert "terminal_failure" not in context
    assert saved["model_endpoint_preflight"][0]["outcome"] == "answered"

    step_failure = {"failed_step": "predict", "terminal": True}
    pipeline_state = {
        "completed_steps": [],
        "failed_steps": ["predict"],
        "step_outputs": {},
        "terminal_failure": dict(step_failure),
    }
    _Executor()._record_and_preflight_model_endpoints(
        context={"model_endpoint_preflight": False},
        active_models=active,
        pipeline_state=pipeline_state,
        state_file=state_file,
    )
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["terminal_failure"] == step_failure


def test_hosted_clients_share_one_default_deadline() -> None:
    from world_understanding.functions.models import vision_language_models
    from world_understanding.functions.models.backends.public import (
        anthropic,
        gemini,
        openai,
    )
    from world_understanding.functions.models.nim_transport import (
        default_nim_request_timeout,
    )

    shared = request_timeout.DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS
    for module in (vision_language_models, openai, anthropic, gemini):
        assert module._DEFAULT_TIMEOUT_SECONDS == shared
    # Hosted NIM gets a longer deadline so its bounded retries fit inside it;
    # self-hosted NIM keeps the shared one.
    hosted = request_timeout.DEFAULT_HOSTED_NIM_REQUEST_TIMEOUT_SECONDS
    assert default_nim_request_timeout(None) == hosted
    assert default_nim_request_timeout("") == hosted
    assert default_nim_request_timeout("https://integrate.api.nvidia.com/v1") == hosted
    assert default_nim_request_timeout("http://localhost:8000/v1") == shared


def test_default_deadline_covers_the_measured_nim_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    monkeypatch.delenv(request_timeout.MODEL_REQUEST_TIMEOUT_ENV, raising=False)
    reloaded = importlib.reload(request_timeout)
    # Public NIM returned the first byte after 123-211 s on 24 Sep 2026.
    assert reloaded.DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS == 300.0
    assert reloaded.DEFAULT_HOSTED_NIM_REQUEST_TIMEOUT_SECONDS == 600.0
    monkeypatch.setenv(request_timeout.MODEL_REQUEST_TIMEOUT_ENV, "90")
    reloaded = importlib.reload(request_timeout)
    assert reloaded.DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS == 90.0
    # The shared override also wins over the hosted NIM default.
    assert reloaded.DEFAULT_HOSTED_NIM_REQUEST_TIMEOUT_SECONDS == 90.0
    monkeypatch.setenv(request_timeout.MODEL_REQUEST_TIMEOUT_ENV, "not-a-number")
    assert (
        importlib.reload(request_timeout).DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS == 300.0
    )
    monkeypatch.delenv(request_timeout.MODEL_REQUEST_TIMEOUT_ENV, raising=False)
    importlib.reload(request_timeout)


def test_route_identity_covers_every_route_field() -> None:
    probed: list[dict[str, Any]] = []

    def create_vlm(config: dict[str, Any]) -> _Model:
        probed.append(config)
        return _Model()

    base = {"backend": "gradio", "model": "m"}
    gateway = {"backend": "custom_gateway", "model": "m"}
    records = describe_model_endpoints(
        [
            ActiveModel("a", "vlm", "vlm", dict(base, endpoint="https://one.example")),
            ActiveModel("b", "vlm", "vlm", dict(base, endpoint="https://two.example")),
            ActiveModel(
                "c",
                "vlm",
                "vlm",
                dict(gateway, gateway={"azure_endpoint": "https://gw-a.example/"}),
            ),
            ActiveModel(
                "d",
                "vlm",
                "vlm",
                dict(gateway, gateway={"azure_endpoint": "https://gw-b.example/"}),
            ),
            # Same route as "a" with different request settings: one probe.
            ActiveModel(
                "e",
                "vlm",
                "vlm",
                dict(base, endpoint="https://one.example", timeout=42, temperature=0.1),
            ),
        ]
    )
    assert [r["endpoint_host"] for r in records] == [
        "one.example",
        "two.example",
        "gw-a.example",
        "gw-b.example",
        "one.example",
    ]
    results = run_model_endpoint_preflight(
        records, create_vlm=create_vlm, create_llm=lambda config: pytest.fail("no llm")
    )
    assert [r["first_step"] for r in results] == ["a", "b", "c", "d"]
    assert len(probed) == 4


def test_failed_preflight_never_persists_a_credential_bearing_state_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.agentic.domain_tasks import model_provisioning

    class FakeProvisioning:
        def create_vlm(self, config: dict[str, Any]) -> _Model:
            return _Model(TimeoutError("queued"))

        def _create_llm(self, config: dict[str, Any]) -> _Model:
            return _Model(TimeoutError("queued"))

    monkeypatch.setattr(model_provisioning, "ModelProvisioningTask", FakeProvisioning)
    secret_dir = tmp_path / f"api_key={SECRET}"
    secret_dir.mkdir()
    state_file = secret_dir / ".pipeline_state.json"
    with pytest.raises(ModelEndpointPreflightError):
        _Executor()._record_and_preflight_model_endpoints(
            context={"model_endpoint_preflight": True},
            active_models=[
                ActiveModel(
                    "predict", "vlm", "vlm", {"backend": "openai", "model": "p"}
                )
            ],
            pipeline_state={
                "completed_steps": [],
                "failed_steps": [],
                "step_outputs": {},
            },
            state_file=state_file,
        )
    saved_text = state_file.read_text(encoding="utf-8")
    assert SECRET not in saved_text
    failure = json.loads(saved_text)["terminal_failure"]
    assert failure["checkpoint_file"] == ".pipeline_state.json"
    assert "checkpoint_path" not in failure


def test_failed_preflight_checkpoint_loads_on_resume_and_is_reprobed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved preflight failure is a valid resume checkpoint."""

    from world_understanding.agentic.domain_tasks import model_provisioning

    outcomes = iter([TimeoutError("queued"), None])

    class _Probe:
        def generate(self, prompt: str, **kwargs: Any) -> str:
            error = next(outcomes)
            if error is not None:
                raise error
            return "OK"

    class FakeProvisioning:
        def create_vlm(self, config: dict[str, Any]) -> _Probe:
            return _Probe()

        def _create_llm(self, config: dict[str, Any]) -> _Probe:
            return _Probe()

    monkeypatch.setattr(model_provisioning, "ModelProvisioningTask", FakeProvisioning)
    state_file = tmp_path / ".pipeline_state.json"
    executor = _Executor()
    context: dict[str, Any] = {
        "model_endpoint_preflight": True,
        "state_file": str(state_file),
        "working_dir": str(tmp_path),
    }
    active = [ActiveModel("predict", "vlm", "vlm", {"backend": "openai", "model": "p"})]

    first_state = executor._initialize_pipeline_state(context, resume=False)
    with pytest.raises(ModelEndpointPreflightError):
        executor._record_and_preflight_model_endpoints(
            context=context,
            active_models=active,
            pipeline_state=first_state,
            state_file=state_file,
        )

    resumed_context = dict(context)
    resumed_context.pop("terminal_failure", None)
    resumed = executor._initialize_pipeline_state(resumed_context, resume=True)
    assert resumed["terminal_failure"]["failed_step"] == "model_endpoint_preflight"
    base_module.reject_terminal_pipeline_resume(resumed, resume=True)

    executor._record_and_preflight_model_endpoints(
        context=resumed_context,
        active_models=active,
        pipeline_state=resumed,
        state_file=state_file,
    )
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert "terminal_failure" not in saved
    assert saved["model_endpoint_preflight"][0]["outcome"] == "answered"


def test_backend_named_section_route_wins_over_top_level_url() -> None:
    config = {
        "backend": "custom_gateway_chat",
        "model": "m",
        "base_url": "https://generic.example/v1",
        "custom_gateway": {"azure_endpoint": "https://effective.example/azure/"},
    }
    (record,) = describe_model_endpoints([ActiveModel("s", "vlm", "vlm", config)])
    assert record["endpoint_host"] == "effective.example"
    other = dict(config, backend="plain")
    (plain,) = describe_model_endpoints([ActiveModel("s", "vlm", "vlm", other)])
    assert plain["endpoint_host"] == "generic.example"
