# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for portable VLM helper functions."""

import inspect
from typing import Any

import pytest

from world_understanding.functions.cv import vlm as vlm_module


class _RecordingVLM:
    def __init__(self, response: str = "caption") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self.response


class _FailingVLM:
    def generate(self, **_kwargs: Any) -> str:
        raise RuntimeError("vlm exploded")


def test_generate_vlm_response_success_and_error() -> None:
    vlm = _RecordingVLM(response="answer")

    result = vlm_module.generate_vlm_response(
        vlm,
        prompt="What is shown?",
        system_prompt="System",
        images=["image.png"],
        temperature=0.2,
    )

    assert result == {"response": "answer"}
    assert vlm.calls == [
        {
            "prompt": "What is shown?",
            "images": ["image.png"],
            "system_prompt": "System",
            "temperature": 0.2,
        }
    ]
    assert vlm_module.generate_vlm_response(
        _FailingVLM(),
        prompt="fail",
    ) == {"error": "Failed to generate response: vlm exploded"}


def test_create_vlm_instance_has_no_implicit_backend_or_model() -> None:
    signature = inspect.signature(vlm_module.create_vlm_instance)

    assert signature.parameters["backend"].default is inspect.Parameter.empty
    assert signature.parameters["model"].default is None
    assert "gpt-4o" not in str(signature)


def test_create_vlm_instance_passes_no_model_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_create_vlm(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(vlm_module, "create_vlm", fake_create_vlm)
    monkeypatch.setattr(
        vlm_module,
        "get_env_api_key_for_backend",
        lambda backend, explicit_api_key=None: None,
    )

    result = vlm_module.create_vlm_instance("nim")

    assert result is sentinel
    assert captured == {"backend": "nim"}


@pytest.mark.parametrize("backend", ["nim", "openai"])
def test_create_vlm_instance_defers_endpoint_env_key_resolution_to_factory(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_vlm(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    def fail_resolver(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("nim credentials should be resolved by the factory")

    monkeypatch.setattr(vlm_module, "create_vlm", fake_create_vlm)
    monkeypatch.setattr(vlm_module, "get_env_api_key_for_backend", fail_resolver)

    vlm_module.create_vlm_instance(
        backend,
        model="google/gemma-4-31b-it",
        base_url="https://custom.example.test/v1",
        timeout=7,
    )

    assert captured == {
        "backend": backend,
        "base_url": "https://custom.example.test/v1",
        "timeout": 7,
        "model": "google/gemma-4-31b-it",
    }


def test_create_vlm_instance_passes_explicit_endpoint_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_vlm(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(vlm_module, "create_vlm", fake_create_vlm)

    vlm_module.create_vlm_instance(
        "nim",
        base_url="https://custom.example.test/v1",
        api_key="endpoint-key",
    )

    assert captured == {
        "backend": "nim",
        "base_url": "https://custom.example.test/v1",
        "api_key": "endpoint-key",
    }


def test_create_vlm_instance_uses_generic_api_key_resolution_for_other_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    resolver_calls: list[tuple[str, str | None]] = []

    def fake_create_vlm(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    def fake_resolver(backend: str, explicit_api_key: str | None = None) -> str:
        resolver_calls.append((backend, explicit_api_key))
        return "resolved-key"

    monkeypatch.setattr(vlm_module, "create_vlm", fake_create_vlm)
    monkeypatch.setattr(vlm_module, "get_env_api_key_for_backend", fake_resolver)

    vlm_module.create_vlm_instance("gemini", model="gemini-model")

    assert resolver_calls == [("gemini", None)]
    assert captured == {
        "backend": "gemini",
        "model": "gemini-model",
        "api_key": "resolved-key",
    }


def test_create_vlm_instance_gradio_uses_endpoint_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_vlm(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    def fail_resolver(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("gradio should not resolve an API key")

    monkeypatch.setattr(vlm_module, "create_vlm", fake_create_vlm)
    monkeypatch.setattr(vlm_module, "get_env_api_key_for_backend", fail_resolver)

    vlm_module.create_vlm_instance(
        "gradio",
        endpoint="https://example.test",
        api_name="/process_media",
        timeout=7,
    )

    assert captured == {
        "backend": "gradio",
        "timeout": 7,
        "endpoint": "https://example.test",
        "api_name": "/process_media",
    }


def test_get_image_caption_normalizes_single_image_and_image_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, Any]] = []
    single_vlm = _RecordingVLM(response="single caption")
    list_vlm = _RecordingVLM(response="list caption")
    vlms = iter([single_vlm, list_vlm])

    def fake_create_vlm_instance(**kwargs: Any) -> _RecordingVLM:
        created.append(kwargs)
        return next(vlms)

    monkeypatch.setattr(vlm_module, "create_vlm_instance", fake_create_vlm_instance)

    assert (
        vlm_module.get_image_caption(
            "image.png",
            caption_prompt="Caption this",
            system_prompt="System",
            vlm_backend="gemini",
            vlm_model="gemini-model",
            vlm_api_key="gemini-key",
        )
        == "single caption"
    )
    assert vlm_module.get_image_caption(["a.png", "b.png"]) == "list caption"

    assert created[0] == {
        "backend": "gemini",
        "model": "gemini-model",
        "api_key": "gemini-key",
    }
    assert single_vlm.calls[0]["images"] == ["image.png"]
    assert list_vlm.calls[0]["images"] == ["a.png", "b.png"]
