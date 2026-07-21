# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the VLM tool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from world_understanding.tools.cv.vlm import (
    VLMInput,
    VLMOutput,
    _display_vlm_response,
    vlm_tool,
)


class RecordingConsole:
    """Minimal console double that records print calls."""

    def __init__(self) -> None:
        self.calls = []

    def print(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


def test_display_vlm_response_includes_optional_model() -> None:
    console = RecordingConsole()

    _display_vlm_response(
        {
            "backend_used": "nim",
            "model_used": "qwen/qwen3.5-397b-a17b",
            "images_analyzed": 2,
            "response": "Two objects are visible.",
        },
        console,
        indent="  ",
    )

    rendered = "\n".join(str(call[0][0]) for call in console.calls)
    assert "VLM Analysis Results" in rendered
    assert "Backend: nim" in rendered
    assert "Model: qwen/qwen3.5-397b-a17b" in rendered
    assert "Images Analyzed: 2" in rendered
    assert "Two objects are visible." in rendered


def test_vlm_tool_accepts_gemini_api_key_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that VLM tool accepts GEMINI_API_KEY without GOOGLE_API_KEY."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    fake_vlm = object()

    with (
        patch(
            "world_understanding.tools.cv.vlm.create_vlm",
            return_value=fake_vlm,
        ) as mock_create_vlm,
        patch(
            "world_understanding.tools.cv.vlm.generate_vlm_response",
            return_value={"response": "Gemini vision response"},
        ),
    ):
        output = vlm_tool(
            VLMInput(
                prompt="Describe this image",
                images=["image.png"],
                backend="gemini",
            )
        )

    assert isinstance(output, VLMOutput)
    assert output.response == "Gemini vision response"
    assert output.backend_used == "gemini"
    assert output.images_analyzed == 1
    mock_create_vlm.assert_called_once()
    call_kwargs = mock_create_vlm.call_args[1]
    assert call_kwargs["api_key"] == "gemini-key"


@pytest.mark.parametrize(
    ("backend", "expected_model", "custom_base_url"),
    (
        ("test-provider", None, None),
        ("nim", "qwen/qwen3.5-397b-a17b", None),
        ("openai", "gpt-5.4", "https://api.openai.example/v1"),
        ("anthropic", "claude-opus-4-6", None),
    ),
)
def test_vlm_tool_uses_backend_default_models(
    backend: str,
    expected_model: str | None,
    custom_base_url: str | None,
) -> None:
    fake_vlm = object()

    with (
        patch(
            "world_understanding.tools.cv.vlm.create_vlm",
            return_value=fake_vlm,
        ) as mock_create_vlm,
        patch(
            "world_understanding.tools.cv.vlm.generate_vlm_response",
            return_value={"response": f"{backend} response"},
        ) as mock_generate_response,
    ):
        output = vlm_tool(
            VLMInput(
                prompt="Describe this image",
                images=["image.png", "other.png"],
                backend=backend,
                api_key="explicit-key",
                base_url=custom_base_url,
            )
        )

    assert output.response == f"{backend} response"
    assert output.backend_used == backend
    assert output.model_used == expected_model
    assert output.images_analyzed == 2
    create_kwargs = mock_create_vlm.call_args.kwargs
    assert create_kwargs["backend"] == backend
    assert create_kwargs["api_key"] == "explicit-key"
    assert create_kwargs["model"] == expected_model
    if custom_base_url is None:
        assert "base_url" not in create_kwargs
    else:
        assert create_kwargs["base_url"] == custom_base_url
    mock_generate_response.assert_called_once_with(
        vlm=fake_vlm,
        prompt="Describe this image",
        images=["image.png", "other.png"],
        system_prompt="You are a helpful AI assistant that can analyze images.",
        temperature=0.7,
        max_tokens=1024,
    )


def test_vlm_tool_openai_rejects_hosted_key_with_env_redirected_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OPENAI_BASE_URL`` redirects the OpenAI SDK to a custom endpoint.

    The VLM tool path must not silently forward the hosted ``OPENAI_API_KEY``
    to that endpoint — same protection as the config-driven model path.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai-compatible.example/v1")

    with patch("world_understanding.tools.cv.vlm.create_vlm") as mock_create_vlm:
        with pytest.raises(ValueError, match="API key required"):
            vlm_tool(
                VLMInput(
                    prompt="Describe this image",
                    images=["image.png"],
                    backend="openai",
                )
            )

    # No VLM client was constructed with the hosted key against the
    # env-redirected URL.
    assert mock_create_vlm.call_count == 0
