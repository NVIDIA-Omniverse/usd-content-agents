# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The physics pipeline's active model plan and endpoint record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from world_understanding.agentic.model_endpoints import ModelEndpointPreflightError

from physics_agent.functions.model_endpoints import physics_active_models
from physics_agent.tasks import unified_pipeline_executor as executor_module


@pytest.fixture(autouse=True)
def _no_nim_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in ("WU", "PA", "TA", "MA"):
        monkeypatch.delenv(f"{prefix}_VLM_NIM_BASE_URL", raising=False)
        monkeypatch.delenv(f"{prefix}_LLM_NIM_BASE_URL", raising=False)


CONFIGS: dict[str, Any] = {
    "identify_asset": {"vlm": {"backend": "nim", "model": "id", "timeout": 180}},
    "build_dataset_usd": {"renderer": {"backend": "remote"}},
    "predict": {"vlm": {"backend": "nim", "model": "pv"}, "llm": {}},
    "apply_physics": {},
}
STEPS = ["identify_asset", "build_dataset_usd", "predict", "apply_physics"]


def test_plan_lists_identify_and_predict_models_only() -> None:
    plan = [
        (m.step, m.section, m.interface) for m in physics_active_models(STEPS, CONFIGS)
    ]
    assert plan == [
        ("identify_asset", "vlm", "vlm"),
        ("predict", "vlm", "vlm"),
    ]


def test_plan_includes_a_configured_parsing_llm() -> None:
    configs = dict(
        CONFIGS, predict={"vlm": {"backend": "nim"}, "llm": {"backend": "nim"}}
    )
    plan = [(m.step, m.interface) for m in physics_active_models(["predict"], configs)]
    assert plan == [("predict", "vlm"), ("predict", "llm")]


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, preflight: bool, fail: bool
) -> dict[str, Any]:
    from world_understanding.agentic.domain_tasks import model_provisioning

    class _Model:
        def generate(self, prompt: str, **kwargs: Any) -> str:
            if fail:
                raise TimeoutError("queued")
            return "OK"

    class FakeProvisioning:
        def create_vlm(self, config: dict[str, Any]) -> _Model:
            return _Model()

        def _create_llm(self, config: dict[str, Any]) -> _Model:
            return _Model()

    monkeypatch.setattr(model_provisioning, "ModelProvisioningTask", FakeProvisioning)
    state_file = tmp_path / ".pipeline_state.json"
    try:
        executor_module.UnifiedPipelineExecutorTask()._record_and_preflight_model_endpoints(
            context={"model_endpoint_preflight": preflight},
            active_models=physics_active_models(STEPS, CONFIGS),
            pipeline_state={
                "completed_steps": [],
                "failed_steps": [],
                "step_outputs": {},
            },
            state_file=state_file,
        )
    finally:
        saved = json.loads(state_file.read_text(encoding="utf-8"))
    return saved


def test_executor_records_physics_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = _run(tmp_path, monkeypatch, preflight=False, fail=False)
    assert [(r["step"], r["timeout_seconds"]) for r in saved["model_endpoints"]] == [
        ("identify_asset", 180),
        ("predict", saved["model_endpoints"][1]["timeout_seconds"]),
    ]
    assert saved["model_endpoints"][1]["timeout_seconds"] >= 300


def test_failed_physics_preflight_is_the_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ModelEndpointPreflightError):
        _run(tmp_path, monkeypatch, preflight=True, fail=True)
    saved = json.loads((tmp_path / ".pipeline_state.json").read_text(encoding="utf-8"))
    assert saved["terminal_failure"]["failed_step"] == "model_endpoint_preflight"
    assert saved["terminal_failure"]["first_step"] == "identify_asset"


@pytest.mark.parametrize("stage", ["build", "probe"])
def test_unbuildable_parsing_llm_does_not_abort_physics_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Prediction falls back to the VLM, so its parsing LLM is optional."""
    from world_understanding.agentic.domain_tasks import model_provisioning

    probed: list[str] = []

    class _Model:
        def generate(self, prompt: str, **kwargs: Any) -> str:
            probed.append("vlm")
            return "OK"

    class _Failing:
        def invoke(self, prompt: str) -> str:
            raise PermissionError("401 Unauthorized")

    class FakeProvisioning:
        def create_vlm(self, config: dict[str, Any]) -> _Model:
            return _Model()

        def _create_llm(self, config: dict[str, Any]) -> _Model:
            if stage == "build":
                raise ValueError("API key is required for NIM backend")
            return _Failing()

    monkeypatch.setattr(model_provisioning, "ModelProvisioningTask", FakeProvisioning)
    configs = dict(
        CONFIGS,
        predict={"vlm": {"backend": "nim", "model": "pv"}, "llm": {"backend": "nim"}},
    )
    state_file = tmp_path / ".pipeline_state.json"
    context: dict[str, Any] = {"model_endpoint_preflight": True}
    executor_module.UnifiedPipelineExecutorTask()._record_and_preflight_model_endpoints(
        context=context,
        active_models=physics_active_models(STEPS, configs),
        pipeline_state={"completed_steps": [], "failed_steps": [], "step_outputs": {}},
        state_file=state_file,
    )
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert "terminal_failure" not in saved
    assert probed == ["vlm", "vlm"]  # identify_asset and predict VLMs still probed
    outcomes = {
        (r["first_step"], r["interface"]): r["outcome"]
        for r in saved["model_endpoint_preflight"]
    }
    assert outcomes[("predict", "llm")] == "optional_unavailable"
    # The run then provisions predict without its parsing LLM (VLM fallback).
    assert context["_unavailable_optional_models"] == [("predict", "llm")]
    trimmed = executor_module.UnifiedPipelineExecutorTask._without_unavailable_optional_models(
        "predict", {"vlm": {"backend": "nim"}, "llm": {"backend": "nim"}}, context
    )
    assert trimmed == {"vlm": {"backend": "nim"}}  # no LLM, so the VLM parses


def test_pipeline_input_keeps_its_positional_fields() -> None:
    import dataclasses

    from physics_agent.api.pipeline import PipelineInput

    names = [f.name for f in dataclasses.fields(PipelineInput)]
    assert names == [
        "config",
        "skip_steps",
        "only_steps",
        "session_id",
        "resume",
        "dry_run",
        "clean",
        "verbose",
        "event_listener",
        "cancel_event",
        "model_endpoint_preflight",
    ]
