# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Physics identify-asset task defaults."""

import json
import threading

import anthropic
import httpx
import openai
import pytest
from botocore.exceptions import ConnectTimeoutError, ReadTimeoutError
from world_understanding.agentic.workflows import Workflow
from world_understanding.utils.model_auth import ModelAuthenticationFailure
from world_understanding.utils.model_timeout import (
    NON_RETRYABLE_VLM_TIMEOUT_MESSAGE,
    TERMINAL_VLM_TIMEOUT_CONTEXT_KEY,
    NonRetryableVLMTimeoutError,
    is_model_timeout_error,
    make_terminal_vlm_timeout_marker,
)

from physics_agent.tasks import identify_asset
from physics_agent.tasks.identify_asset import IdentifyAssetTask
from physics_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask


class _RecordingVLM:
    has_bounded_request_timeout = True

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return json.dumps(
            {
                "asset_type": "vehicle",
                "asset_subtype": "forklift",
                "asset_description": "A forklift",
                "confidence": "high",
                "reasoning": "Visible forks and mast",
            }
        )


class _SequencedVLM:
    has_bounded_request_timeout = True

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.generate_calls = []
        self.generate_thread_ids = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        self.generate_thread_ids.append(threading.get_ident())
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _provider_timeout_cases() -> list[object]:
    """Return concrete timeout types surfaced by supported VLM SDKs."""
    return [
        pytest.param(
            openai.APITimeoutError(
                request=httpx.Request("POST", "https://openai.invalid")
            ),
            id="openai-and-azure",
        ),
        pytest.param(
            anthropic.APITimeoutError(
                request=httpx.Request("POST", "https://anthropic.invalid")
            ),
            id="anthropic",
        ),
        pytest.param(
            ReadTimeoutError(
                endpoint_url="https://bedrock.invalid",
                error=TimeoutError(),
            ),
            id="bedrock-read",
        ),
        pytest.param(
            ConnectTimeoutError(
                endpoint_url="https://bedrock.invalid",
                error=TimeoutError(),
            ),
            id="bedrock-connect",
        ),
        pytest.param(
            httpx.ReadTimeout(
                "Gemini request timed out",
                request=httpx.Request("POST", "https://gemini.invalid"),
            ),
            id="gemini-httpx",
        ),
        pytest.param(TimeoutError("provider timed out"), id="builtin-gradio-or-nim"),
    ]


@pytest.mark.parametrize("provider_timeout", _provider_timeout_cases())
def test_model_timeout_classifier_recognizes_supported_sdk_families(
    provider_timeout,
):
    assert is_model_timeout_error(provider_timeout) is True


@pytest.mark.parametrize("link_name", ["__cause__", "__context__"])
def test_model_timeout_classifier_follows_wrapped_exception_links(link_name):
    provider_timeout = openai.APITimeoutError(
        request=httpx.Request("POST", "https://openai.invalid")
    )
    wrapper = RuntimeError("provider adapter failed")
    setattr(wrapper, link_name, provider_timeout)

    assert is_model_timeout_error(wrapper) is True


def test_model_timeout_classifier_does_not_match_ordinary_provider_errors():
    assert is_model_timeout_error(RuntimeError("provider returned 504")) is False


def test_identify_asset_rejects_unbounded_backend_before_dispatch(tmp_path):
    vlm = _SequencedVLM(
        [json.dumps({"asset_type": "vehicle", "asset_subtype": "forklift"})]
    )
    vlm.has_bounded_request_timeout = False

    with pytest.raises(RuntimeError, match="verified bounded request timeout"):
        IdentifyAssetTask().run(
            {
                "vlm": vlm,
                "composition_images": ["/tmp/view.png"],
                "output_dir": str(tmp_path),
            }
        )

    assert vlm.generate_calls == []
    assert not (tmp_path / "identification.json").exists()


def test_identify_asset_uses_default_vlm_temperature_when_invoke_kwargs_missing(
    tmp_path, monkeypatch
):
    """Identify-asset fallback should not hardcode the old 0.3 temperature."""
    monkeypatch.setattr(identify_asset, "DEFAULT_VLM_TEMPERATURE", 0.8)
    vlm = _RecordingVLM()

    IdentifyAssetTask().run(
        {
            "vlm": vlm,
            "composition_images": ["/tmp/view.png"],
            "output_dir": str(tmp_path),
        }
    )

    assert vlm.generate_kwargs["temperature"] == 0.8


def test_identify_asset_uses_vlm_config_temperature_when_invoke_kwargs_missing(
    tmp_path, monkeypatch
):
    """VLM config should be the fallback before the module default."""
    monkeypatch.setattr(identify_asset, "DEFAULT_VLM_TEMPERATURE", 0.8)
    vlm = _RecordingVLM()

    IdentifyAssetTask().run(
        {
            "vlm": vlm,
            "composition_images": ["/tmp/view.png"],
            "output_dir": str(tmp_path),
            "vlm_config": {"temperature": 0.6},
        }
    )

    assert vlm.generate_kwargs["temperature"] == 0.6


def test_identify_asset_forwards_model_aware_reasoning_effort(tmp_path):
    vlm = _RecordingVLM()

    IdentifyAssetTask().run(
        {
            "vlm": vlm,
            "composition_images": ["/tmp/view.png"],
            "output_dir": str(tmp_path),
            "vlm_invoke_kwargs": {
                "temperature": 1.0,
                "max_tokens": 24576,
                "reasoning_effort": "max",
            },
        }
    )

    assert vlm.generate_kwargs["reasoning_effort"] == "max"


def test_identify_asset_treats_none_vlm_config_as_empty(tmp_path, monkeypatch):
    """Explicit None VLM config should fall back to the module default."""
    monkeypatch.setattr(identify_asset, "DEFAULT_VLM_TEMPERATURE", 0.8)
    vlm = _RecordingVLM()

    IdentifyAssetTask().run(
        {
            "vlm": vlm,
            "composition_images": ["/tmp/view.png"],
            "output_dir": str(tmp_path),
            "vlm_config": None,
        }
    )

    assert vlm.generate_kwargs["temperature"] == 0.8


def test_identify_asset_retries_transient_vlm_failures(tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr(identify_asset.time, "sleep", delays.append)
    vlm = _SequencedVLM(
        [
            RuntimeError("provider returned 429"),
            RuntimeError("provider returned 429"),
            json.dumps(
                {
                    "asset_type": "vehicle",
                    "asset_subtype": "forklift",
                    "asset_description": "A forklift",
                    "confidence": "high",
                    "reasoning": "Visible forks and mast",
                }
            ),
        ]
    )

    result = IdentifyAssetTask().run(
        {
            "vlm": vlm,
            "composition_images": ["/tmp/view.png"],
            "output_dir": str(tmp_path),
        }
    )

    assert len(vlm.generate_calls) == 3
    assert delays == [1.0, 1.0]
    assert result["identification"]["asset_subtype"] == "forklift"
    assert (tmp_path / "identification.json").is_file()


@pytest.mark.parametrize("provider_timeout", _provider_timeout_cases())
def test_identify_asset_normalizes_provider_timeout_without_retry(
    tmp_path, monkeypatch, provider_timeout
):
    delays = []
    monkeypatch.setattr(identify_asset.time, "sleep", delays.append)
    caller_thread_id = threading.get_ident()
    vlm = _SequencedVLM([provider_timeout])

    with pytest.raises(NonRetryableVLMTimeoutError) as exc_info:
        IdentifyAssetTask().run(
            {
                "vlm": vlm,
                "composition_images": ["/tmp/view.png"],
                "output_dir": str(tmp_path),
            }
        )

    assert str(exc_info.value) == NON_RETRYABLE_VLM_TIMEOUT_MESSAGE
    assert len(vlm.generate_calls) == 1
    assert vlm.generate_thread_ids == [caller_thread_id]
    assert delays == []
    assert not (tmp_path / "identification.json").exists()


def test_identify_asset_does_not_retry_unverified_remote_timeout(tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr(identify_asset.time, "sleep", delays.append)
    vlm = _SequencedVLM(
        [
            NonRetryableVLMTimeoutError("remote completion unverified"),
            json.dumps({"asset_type": "vehicle", "asset_subtype": "forklift"}),
        ]
    )

    with pytest.raises(
        NonRetryableVLMTimeoutError, match="remote completion unverified"
    ):
        IdentifyAssetTask().run(
            {
                "vlm": vlm,
                "composition_images": ["/tmp/view.png"],
                "output_dir": str(tmp_path),
            }
        )

    assert len(vlm.generate_calls) == 1
    assert delays == []
    assert not (tmp_path / "identification.json").exists()


def test_identify_asset_workflow_preserves_terminal_timeout_marker(
    tmp_path, monkeypatch
):
    delays = []
    monkeypatch.setattr(identify_asset.time, "sleep", delays.append)
    vlm = _SequencedVLM(
        [NonRetryableVLMTimeoutError("provider payload must not be retained")]
    )

    result = Workflow([IdentifyAssetTask()]).run(
        {
            "vlm": vlm,
            "composition_images": ["/tmp/view.png"],
            "output_dir": str(tmp_path),
        }
    )

    assert len(vlm.generate_calls) == 1
    assert delays == []
    assert result["error"] == "Task execution failed"
    assert result["failed_task"] == "IdentifyAsset"
    assert result[TERMINAL_VLM_TIMEOUT_CONTEXT_KEY] == make_terminal_vlm_timeout_marker(
        "IdentifyAsset"
    )
    assert result["workflow_terminated"] is True
    assert not (tmp_path / "identification.json").exists()


def test_identify_asset_does_not_retry_authentication_failures(tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr(identify_asset.time, "sleep", delays.append)
    vlm = _SequencedVLM(
        [
            RuntimeError("401 unauthorized"),
            json.dumps({"asset_type": "vehicle", "asset_subtype": "forklift"}),
        ]
    )

    with pytest.raises(ModelAuthenticationFailure):
        IdentifyAssetTask().run(
            {
                "vlm": vlm,
                "composition_images": ["/tmp/view.png"],
                "output_dir": str(tmp_path),
            }
        )

    assert len(vlm.generate_calls) == 1
    assert delays == []
    assert not (tmp_path / "identification.json").exists()


def test_identify_asset_retry_exhaustion_records_failed_pipeline_step(
    tmp_path, monkeypatch
):
    delays = []
    monkeypatch.setattr(identify_asset.time, "sleep", delays.append)
    vlm = _SequencedVLM([RuntimeError("provider returned 429")] * 3)
    output_dir = tmp_path / "identification"
    executor = UnifiedPipelineExecutorTask()

    def execute_identify(*_args, **_kwargs):
        return IdentifyAssetTask().run(
            {
                "vlm": vlm,
                "composition_images": ["/tmp/view.png"],
                "output_dir": str(output_dir),
            }
        )

    monkeypatch.setattr(executor, "_execute_step", execute_identify)
    working_dir = tmp_path / "pipeline"
    context = {
        "steps_to_run": ["identify_asset"],
        "step_configs": {"identify_asset": {}},
        "working_dir": working_dir,
    }

    with pytest.raises(RuntimeError, match="identify_asset"):
        executor.run(context)

    state = json.loads(
        (working_dir / ".pipeline_state.json").read_text(encoding="utf-8")
    )
    assert len(vlm.generate_calls) == 3
    assert delays == [1.0, 1.0]
    assert state["completed_steps"] == []
    assert state["failed_steps"] == ["identify_asset"]
    assert state["step_outputs"] == {}
    assert context["pipeline_state"] == "failed"
    assert not (output_dir / "identification.json").exists()
