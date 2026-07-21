# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for image generation models."""

import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage

from world_understanding.functions.models.image_generation_models import (
    BaseImageGenerationModel,
    GeminiImageGenerationModel,
    NIMImageGenerationModel,
    OpenAICompatibleChatImageGenerationModel,
    OpenAIImageGenerationModel,
    _NamedBytesIO,
    create_image_generation_model,
)

# Check if google-genai is available
try:
    import google.genai  # noqa: F401

    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

# Check if openai is available
try:
    import openai  # noqa: F401

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def _png_bytes(size: tuple[int, int] = (2, 3)) -> bytes:
    buffer = BytesIO()
    PILImage.new("RGB", size, color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _data_uri(size: tuple[int, int] = (2, 3)) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(size)).decode()


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
def test_create_image_generation_model_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test creating Gemini image generation model."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        create_image_generation_model("gemini")


def test_create_image_generation_model_unknown_backend() -> None:
    """Test creating image generation model with unknown backend."""
    with pytest.raises(ValueError, match="Unknown image generation backend"):
        create_image_generation_model("unknown_backend")


def test_create_image_generation_model_not_implemented() -> None:
    """Test creating image generation model with unsupported backends."""
    with pytest.raises(ValueError, match="Unknown image generation backend"):
        create_image_generation_model("openai_dalle")

    with pytest.raises(ValueError, match="Unknown image generation backend"):
        create_image_generation_model("stability")


def test_gemini_normalizes_chat_style_generation_kwargs() -> None:
    kwargs = GeminiImageGenerationModel._normalize_generate_content_kwargs(
        {
            "max_tokens": 123,
            "temperature": 0.7,
            "unrelated": "kept",
        }
    )

    assert kwargs == {
        "unrelated": "kept",
        "config": {
            "max_output_tokens": 123,
            "temperature": 0.7,
        },
    }


def test_gemini_preserves_typed_generation_config_object() -> None:
    class GenerateContentConfig:
        max_output_tokens = None

    config = GenerateContentConfig()
    kwargs = GeminiImageGenerationModel._normalize_generate_content_kwargs(
        {
            "config": config,
            "max_tokens": 123,
            "temperature": 0.7,
        }
    )

    assert kwargs["config"] is config
    assert config.max_output_tokens == 123
    assert config.temperature == 0.7


def test_gemini_typed_generation_config_keeps_explicit_values() -> None:
    class GenerateContentConfig:
        max_output_tokens = 456
        temperature = 0.2

    config = GenerateContentConfig()
    kwargs = GeminiImageGenerationModel._normalize_generate_content_kwargs(
        {
            "config": config,
            "max_tokens": 123,
            "temperature": 0.7,
        }
    )

    assert kwargs["config"] is config
    assert config.max_output_tokens == 456
    assert config.temperature == 0.2


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_create_image_generation_model_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test creating OpenAI image generation model raises without API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        create_image_generation_model("openai")


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_model_initialization_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that OpenAIImageGenerationModel requires an API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        OpenAIImageGenerationModel()


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_model_rejects_remote_base_url_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test OpenAI image generation requires a key for remote base_url."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageGenerationModel(
            base_url="https://api.openai-compatible.example/v1",
        )


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_model_rejects_placeholder_env_key_for_remote_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageGenerationModel(
            base_url="https://api.openai-compatible.example/v1",
        )


def test_openai_model_rejects_env_key_for_custom_remote_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom OpenAI-compatible endpoints must use an explicit endpoint key."""

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "hosted-openai-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageGenerationModel(
            base_url="https://api.openai-compatible.example/v1",
        )


def test_openai_model_accepts_explicit_key_for_custom_remote_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "hosted-openai-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    OpenAIImageGenerationModel(
        api_key="endpoint-openai-key",
        base_url="https://api.openai-compatible.example/v1",
    )

    assert captured["api_key"] == "endpoint-openai-key"
    assert captured["base_url"] == "https://api.openai-compatible.example/v1"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_model_rejects_scheme_less_remote_base_url_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageGenerationModel(base_url="api.openai-compatible.example:443/v1")


def test_openai_model_uses_dummy_key_for_local_base_url_before_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local OpenAI-compatible image endpoints must not receive hosted keys."""
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "real-hosted-openai-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    OpenAIImageGenerationModel(
        api_key="not-used",
        base_url="http://localhost:8000/v1",
    )

    assert captured["api_key"] == "not-used"
    assert captured["base_url"] == "http://localhost:8000/v1"


def test_openai_model_rejects_env_key_for_local_base_url_without_explicit_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hosted ``OPENAI_API_KEY`` must not silently flow to a local
    OpenAI-compatible endpoint. Local URLs are non-provider trust boundaries
    and require an explicit endpoint-scoped ``api_key`` (or the documented
    ``not-used`` no-auth placeholder)."""
    monkeypatch.setenv("OPENAI_API_KEY", "real-hosted-openai-key")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageGenerationModel(base_url="http://localhost:8000/v1")


def test_openai_model_rejects_local_base_url_without_explicit_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageGenerationModel(base_url="http://localhost:8000/v1")


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_model_with_api_key() -> None:
    """Test that OpenAIImageGenerationModel can be initialized with an API key."""
    model = OpenAIImageGenerationModel(api_key="test-key")
    assert model.model_name == "gpt-image-1"
    assert model.backend_name == "openai"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_model_custom_model_name() -> None:
    """Test that OpenAIImageGenerationModel accepts custom model names."""
    model = OpenAIImageGenerationModel(api_key="test-key", model="custom-model")
    assert model.model_name == "custom-model"


def test_base_image_generation_model_interface() -> None:
    """Test that BaseImageGenerationModel defines the correct interface."""

    class MockImageGenModel(BaseImageGenerationModel):
        def generate(
            self,
            prompt: str,
            images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
            **kwargs: Any,
        ) -> PILImage.Image:
            return PILImage.new("RGB", (100, 100))

        def generate_with_image_prompt_pairs(
            self,
            image_prompt_pairs: list[
                tuple[str, str | Path | PILImage.Image | np.ndarray]
            ],
            final_prompt: str,
            **kwargs: Any,
        ) -> PILImage.Image:
            return PILImage.new("RGB", (100, 100))

        @property
        def model_name(self) -> str:
            return "mock-model"

        @property
        def backend_name(self) -> str:
            return "mock"

    model = MockImageGenModel()
    assert model.model_name == "mock-model"
    assert model.backend_name == "mock"

    # Test generate
    result = model.generate("test prompt")
    assert isinstance(result, PILImage.Image)
    assert result.size == (100, 100)

    # Test generate_with_image_prompt_pairs
    result = model.generate_with_image_prompt_pairs([], "test prompt")
    assert isinstance(result, PILImage.Image)


def test_base_image_generation_abstract_stubs_are_callable() -> None:
    assert BaseImageGenerationModel.generate(None, "prompt") is None  # type: ignore[misc]
    assert (
        BaseImageGenerationModel.generate_with_image_prompt_pairs(None, [], "prompt")  # type: ignore[misc]
        is None
    )
    assert BaseImageGenerationModel.model_name.fget(None) is None  # type: ignore[arg-type, union-attr]
    assert BaseImageGenerationModel.backend_name.fget(None) is None  # type: ignore[arg-type, union-attr]


def test_load_image_from_different_formats() -> None:
    """Test loading images from various formats."""

    class MockImageGenModel(BaseImageGenerationModel):
        def generate(
            self,
            prompt: str,
            images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
            **kwargs: Any,
        ) -> PILImage.Image:
            return PILImage.new("RGB", (100, 100))

        def generate_with_image_prompt_pairs(
            self,
            image_prompt_pairs: list[
                tuple[str, str | Path | PILImage.Image | np.ndarray]
            ],
            final_prompt: str,
            **kwargs: Any,
        ) -> PILImage.Image:
            return PILImage.new("RGB", (100, 100))

        @property
        def model_name(self) -> str:
            return "mock"

        @property
        def backend_name(self) -> str:
            return "mock"

    model = MockImageGenModel()

    # Test PIL Image
    pil_img = PILImage.new("RGB", (50, 50))
    loaded = model._load_image(pil_img)
    assert isinstance(loaded, PILImage.Image)
    assert loaded.size == (50, 50)

    # Test numpy array
    np_img = np.zeros((50, 50, 3), dtype=np.uint8)
    loaded = model._load_image(np_img)
    assert isinstance(loaded, PILImage.Image)
    assert loaded.size == (50, 50)

    # Test unsupported type
    with pytest.raises(ValueError, match="Unsupported image type"):
        model._load_image(123)  # type: ignore[arg-type]


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
def test_gemini_model_initialization_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that GeminiImageGenerationModel requires an API key."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        GeminiImageGenerationModel()


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
def test_gemini_model_accepts_gemini_api_key_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that GeminiImageGenerationModel accepts GEMINI_API_KEY."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    model = GeminiImageGenerationModel()

    assert model.model_name == "gemini-3-pro-image-preview"
    assert model.backend_name == "gemini"


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
def test_gemini_model_replaces_placeholder_config_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "real-gemini-key")

    model = GeminiImageGenerationModel(api_key="YOUR_GOOGLE_API_KEY")

    assert model.model_name == "gemini-3-pro-image-preview"
    assert model.backend_name == "gemini"


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
def test_gemini_model_with_api_key() -> None:
    """Test that GeminiImageGenerationModel can be initialized with API key."""
    # Just test initialization, not actual API calls
    model = GeminiImageGenerationModel(api_key="test-key")
    assert model.model_name == "gemini-3-pro-image-preview"
    assert model.backend_name == "gemini"


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
def test_gemini_model_custom_model_name() -> None:
    """Test that GeminiImageGenerationModel accepts custom model names."""
    model = GeminiImageGenerationModel(api_key="test-key", model="custom-model")
    assert model.model_name == "custom-model"


def test_gemini_model_normalizes_openai_style_generation_kwargs() -> None:
    normalized = GeminiImageGenerationModel._normalize_generate_content_kwargs(
        {
            "max_tokens": 4096,
            "temperature": 0.7,
            "config": {"top_p": 0.9},
        }
    )

    assert normalized == {
        "config": {
            "max_output_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }


def test_create_image_generation_model_nim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test creating NIM image generation model raises without API key."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        create_image_generation_model("nim")


def test_nim_model_initialization_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that NIMImageGenerationModel requires an API key."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        NIMImageGenerationModel()


def test_nim_model_rejects_placeholder_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "YOUR_NVIDIA_API_KEY")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NIMImageGenerationModel()


def test_nim_model_rejects_ma_nim_key_for_hosted_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "local-sidecar-key")

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NIMImageGenerationModel(base_url="https://ai.api.nvidia.com/v1/genai")


def test_nim_model_rejects_nvidia_key_for_custom_remote_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NIMImageGenerationModel(base_url="https://nim.example.com/v1/genai")


def test_nim_model_accepts_explicit_key_for_custom_remote_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    model = NIMImageGenerationModel(
        api_key="endpoint-nim-key",
        base_url="https://nim.example.com/v1/genai",
    )

    assert model.model_name == "black-forest-labs/flux_2-klein-4b"
    assert model.backend_name == "nim"


def test_nim_model_accepts_explicit_local_sidecar_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")

    model = NIMImageGenerationModel(base_url="http://image-gen-nim:8000/v1")

    assert model.model_name == "black-forest-labs/flux_2-klein-4b"
    assert model.backend_name == "nim"


def test_nim_model_rejects_global_nvidia_key_for_local_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NIMImageGenerationModel(base_url="http://image-gen-nim:8000/v1")


def test_nim_model_with_api_key() -> None:
    """Test that NIMImageGenerationModel can be initialized with an API key."""
    model = NIMImageGenerationModel(api_key="test-key")
    assert model.model_name == "black-forest-labs/flux_2-klein-4b"
    assert model.backend_name == "nim"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_compatible_chat_image_model_requires_endpoint_and_key() -> None:
    with pytest.raises(ValueError, match="base_url is required"):
        OpenAICompatibleChatImageGenerationModel(api_key="endpoint-key")

    with pytest.raises(ValueError, match="endpoint-scoped api_key is required"):
        OpenAICompatibleChatImageGenerationModel(
            base_url="https://image-provider.example/v1"
        )


def test_nim_model_custom_model_name() -> None:
    """Test that NIMImageGenerationModel accepts custom model names."""
    model = NIMImageGenerationModel(api_key="test-key", model="org/my_model-v1")
    assert model.model_name == "org/my_model-v1"


def test_nim_model_url_slug_conversion() -> None:
    """Test that NIMImageGenerationModel converts model name to URL slug correctly."""
    assert (
        NIMImageGenerationModel._model_to_url_slug("black-forest-labs/flux_2-klein-4b")
        == "black-forest-labs/flux.2-klein-4b"
    )
    assert (
        NIMImageGenerationModel._model_to_url_slug("org/my_model_v2")
        == "org/my.model_v2"
    )
    assert (
        NIMImageGenerationModel._model_to_url_slug("no_slash_model") == "no.slash_model"
    )


def test_base_image_generation_model_supports_image_conditioning_and_paths(
    tmp_path: Path,
) -> None:
    class MockImageGenModel(BaseImageGenerationModel):
        def generate(
            self,
            prompt: str,
            images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
            **kwargs: Any,
        ) -> PILImage.Image:
            return PILImage.new("RGB", (1, 1))

        def generate_with_image_prompt_pairs(
            self,
            image_prompt_pairs: list[
                tuple[str, str | Path | PILImage.Image | np.ndarray]
            ],
            final_prompt: str,
            **kwargs: Any,
        ) -> PILImage.Image:
            return PILImage.new("RGB", (1, 1))

        @property
        def model_name(self) -> str:
            return "mock"

        @property
        def backend_name(self) -> str:
            return "mock"

    path = tmp_path / "image.png"
    path.write_bytes(_png_bytes((4, 5)))
    model = MockImageGenModel()

    assert model.supports_image_conditioning is True
    assert model._load_image(path).size == (4, 5)


def test_gemini_generate_extracts_inline_images() -> None:
    captured: dict[str, Any] = {}

    class _Models:
        def generate_content(self, **kwargs: Any) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    inline_data=SimpleNamespace(data=_png_bytes((7, 8)))
                                )
                            ]
                        )
                    )
                ]
            )

    model = object.__new__(GeminiImageGenerationModel)
    model.client = SimpleNamespace(models=_Models())
    model._model_name = "gemini-test"

    result = model.generate(
        "make image",
        images=[PILImage.new("RGBA", (1, 2))],
        max_completion_tokens=12,
        top_k=3,
        candidate_count=1,
    )

    assert result.size == (7, 8)
    assert captured["model"] == "gemini-test"
    assert captured["contents"][0] == "make image"
    assert captured["contents"][1].mode == "RGB"
    assert captured["config"] == {
        "max_output_tokens": 12,
        "top_k": 3,
        "candidate_count": 1,
    }


def test_gemini_generate_with_pairs_and_no_image_response() -> None:
    calls: list[list[object]] = []

    class _Models:
        def generate_content(self, **kwargs: Any) -> object:
            calls.append(kwargs["contents"])
            return SimpleNamespace(candidates=[])

    model = object.__new__(GeminiImageGenerationModel)
    model.client = SimpleNamespace(models=_Models())
    model._model_name = "gemini-test"

    with pytest.raises(ValueError, match="No image generated"):
        model.generate_with_image_prompt_pairs(
            [("reference", np.zeros((2, 2, 3), dtype=np.uint8))],
            "final prompt",
        )

    assert calls[0][0] == "reference"
    assert isinstance(calls[0][1], PILImage.Image)
    assert calls[0][2] == "final prompt"

    with pytest.raises(ValueError, match="No image generated"):
        model.generate("plain prompt")


def test_gemini_generate_with_pairs_returns_inline_image() -> None:
    class _Models:
        def generate_content(self, **kwargs: Any) -> object:
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    inline_data=SimpleNamespace(data=_png_bytes((6, 5)))
                                )
                            ]
                        )
                    )
                ]
            )

    model = object.__new__(GeminiImageGenerationModel)
    model.client = SimpleNamespace(models=_Models())
    model._model_name = "gemini-test"

    assert model.generate_with_image_prompt_pairs([], "final prompt").size == (6, 5)


class _FakeChatCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.response


def _openai_compatible_model_with_response(
    response: object,
) -> tuple[OpenAICompatibleChatImageGenerationModel, _FakeChatCompletions]:
    completions = _FakeChatCompletions(response)
    model = object.__new__(OpenAICompatibleChatImageGenerationModel)
    model._model_name = "provider-test"
    model._backend_name = "test-provider"
    model._base_url = "https://image-provider.example/v1"
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return model, completions


def _chat_response(message: object, finish_reason: str = "stop") -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


def test_openai_compatible_generate_builds_content_and_extracts_part_image() -> None:
    response = _chat_response(
        SimpleNamespace(
            content=[
                {"type": "text", "text": "ok"},
                {"type": "image_url", "image_url": {"url": _data_uri((5, 6))}},
            ]
        )
    )
    model, completions = _openai_compatible_model_with_response(response)

    result = model.generate(
        "prompt",
        images=[PILImage.new("RGB", (3, 4))],
        temperature=0.2,
        max_tokens=None,
    )

    assert result.size == (5, 6)
    request = completions.calls[0]
    assert request["model"] == "provider-test"
    assert request["temperature"] == 0.2
    assert "max_tokens" not in request
    content = request["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "prompt"}
    assert content[1]["type"] == "image_url"
    assert model.model_name == "provider-test"
    assert model.backend_name == "test-provider"


def test_openai_compatible_model_initialization_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    model = OpenAICompatibleChatImageGenerationModel(
        api_key="secret",
        model="custom-model",
        base_url="https://inference.example",
        timeout=5,
        backend_name="test-provider",
    )

    assert model.model_name == "custom-model"
    assert model.backend_name == "test-provider"
    assert captured == {
        "api_key": "secret",
        "base_url": "https://inference.example",
        "timeout": 5,
    }


def test_openai_compatible_generate_with_pairs_and_alternate_response_shapes() -> None:
    typed_part = SimpleNamespace(
        type="image_url", image_url=SimpleNamespace(url=_data_uri((8, 9)))
    )
    model, _completions = _openai_compatible_model_with_response(
        _chat_response(SimpleNamespace(content=[typed_part]))
    )
    result = model.generate_with_image_prompt_pairs(
        [("desc", PILImage.new("RGB", (1, 1)))],
        "final",
    )
    assert result.size == (8, 9)

    model, _ = _openai_compatible_model_with_response(
        _chat_response(SimpleNamespace(content=f"text before {_data_uri((4, 4))}"))
    )
    assert model._call_and_extract_image([{"type": "text", "text": "x"}]).size == (4, 4)

    model, _ = _openai_compatible_model_with_response(
        _chat_response(
            SimpleNamespace(
                content=None,
                images=[{"type": "image_url", "image_url": {"url": _data_uri((6, 6))}}],
            )
        )
    )
    assert model._call_and_extract_image([{"type": "text", "text": "x"}]).size == (6, 6)

    model, _ = _openai_compatible_model_with_response(
        _chat_response(
            SimpleNamespace(
                content=None,
                images=None,
                model_extra={
                    "images": [
                        {"type": "image_url", "image_url": {"url": _data_uri((7, 7))}}
                    ]
                },
            )
        )
    )
    assert model._call_and_extract_image([{"type": "text", "text": "x"}]).size == (7, 7)

    inline_data = {
        "inlineData": {
            "mimeType": "image/png",
            "data": base64.b64encode(_png_bytes((10, 11))).decode(),
        }
    }
    model, _ = _openai_compatible_model_with_response(
        _chat_response(
            SimpleNamespace(content=[{"type": "text", "text": "ok"}, inline_data])
        )
    )
    assert model._call_and_extract_image([{"type": "text", "text": "x"}]).size == (
        10,
        11,
    )

    model, _ = _openai_compatible_model_with_response(
        _chat_response(
            SimpleNamespace(content=json.dumps({"parts": [inline_data]})),
        )
    )
    assert model._call_and_extract_image([{"type": "text", "text": "x"}]).size == (
        10,
        11,
    )

    model, _ = _openai_compatible_model_with_response(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None, images=None, model_extra={}),
                    finish_reason="stop",
                )
            ],
            model_extra={"candidates": [{"content": {"parts": [inline_data]}}]},
        )
    )
    assert model._call_and_extract_image([{"type": "text", "text": "x"}]).size == (
        10,
        11,
    )

    model, _ = _openai_compatible_model_with_response(
        _chat_response(
            SimpleNamespace(
                content=[
                    {
                        "type": "image",
                        "image": {
                            "b64_json": base64.b64encode(_png_bytes((12, 13))).decode()
                        },
                    }
                ]
            )
        )
    )
    assert model._call_and_extract_image([{"type": "text", "text": "x"}]).size == (
        12,
        13,
    )


def test_openai_compatible_decode_helpers_and_missing_image() -> None:
    assert (
        OpenAICompatibleChatImageGenerationModel._try_decode_data_uri("no data") is None
    )
    assert (
        OpenAICompatibleChatImageGenerationModel._try_decode_data_uri(
            "data:image/png;base64,not-valid"
        )
        is None
    )
    assert (
        OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part({})
        is None
    )
    assert (
        OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
            [{"type": "text", "text": "ok"}, "no image"]
        )
        is None
    )
    assert (
        OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
            SimpleNamespace(type="image_url", image_url=SimpleNamespace(url=""))
        )
        is None
    )
    assert (
        OpenAICompatibleChatImageGenerationModel._try_extract_image_from_json_string(
            "{bad"
        )
        is None
    )
    assert (
        OpenAICompatibleChatImageGenerationModel._try_decode_image_payload(123) is None
    )
    assert (
        OpenAICompatibleChatImageGenerationModel._try_decode_image_payload("not-base64")
        is None
    )
    assert (
        OpenAICompatibleChatImageGenerationModel._try_decode_image_payload(b"\xff")
        is None
    )
    assert OpenAICompatibleChatImageGenerationModel._try_decode_image_payload(
        _png_bytes((14, 15))
    ).size == (14, 15)

    model, _ = _openai_compatible_model_with_response(
        _chat_response(SimpleNamespace(content="no image"), finish_reason="length")
    )
    with pytest.raises(ValueError, match="No image found"):
        model._call_and_extract_image([{"type": "text", "text": "x"}])


def test_openai_compatible_extracts_typed_overflow_shapes() -> None:
    class _DumpedPart:
        def model_dump(self) -> dict[str, Any]:
            return {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(_png_bytes((16, 17))).decode(),
                }
            }

    class _BrokenDumpPart:
        def model_dump(self) -> dict[str, Any]:
            raise RuntimeError("cannot dump")

    class _PartsObject:
        def __init__(self) -> None:
            self.parts = [
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64.b64encode(_png_bytes((18, 19))).decode(),
                    }
                }
            ]

    class _MimeObject:
        mimeType = "image/png"
        data = base64.b64encode(_png_bytes((20, 21))).decode()

    assert OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
        _DumpedPart()
    ).size == (16, 17)
    assert (
        OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
            _BrokenDumpPart()
        )
        is None
    )
    assert OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
        _PartsObject()
    ).size == (18, 19)
    assert OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
        _MimeObject()
    ).size == (20, 21)


class _ImageItem:
    def __init__(self, *, b64_json: str | None = None, url: str | None = None) -> None:
        self.b64_json = b64_json
        self.url = url


class _FakeImages:
    def __init__(self, response: object) -> None:
        self.response = response
        self.generate_calls: list[dict[str, Any]] = []
        self.edit_calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> object:
        self.generate_calls.append(kwargs)
        return self.response

    def edit(self, **kwargs: Any) -> object:
        self.edit_calls.append(kwargs)
        return self.response


def _openai_model_with_response(
    response: object,
) -> tuple[OpenAIImageGenerationModel, _FakeImages]:
    images = _FakeImages(response)
    model = object.__new__(OpenAIImageGenerationModel)
    model._model_name = "openai-test"
    model.client = SimpleNamespace(images=images)
    return model, images


def test_openai_generate_text_and_image_paths() -> None:
    response = SimpleNamespace(
        data=[
            _ImageItem(
                b64_json=base64.b64encode(_png_bytes((9, 10))).decode(),
            )
        ]
    )
    model, images = _openai_model_with_response(response)

    assert model.generate("prompt", quality="low").size == (9, 10)
    assert images.generate_calls[0]["model"] == "openai-test"
    assert images.generate_calls[0]["quality"] == "low"

    assert model.generate("edit", images=[PILImage.new("RGB", (2, 2))]).size == (9, 10)
    uploaded = images.edit_calls[0]["image"][0]
    assert isinstance(uploaded, _NamedBytesIO)
    assert uploaded.name == "image.png"
    assert model.model_name == "openai-test"
    assert model.backend_name == "openai"


def test_openai_generate_with_pairs_combines_descriptions() -> None:
    response = SimpleNamespace(
        data=[_ImageItem(b64_json=base64.b64encode(_png_bytes()).decode())]
    )
    model, images = _openai_model_with_response(response)

    model.generate_with_image_prompt_pairs(
        [
            ("first", PILImage.new("RGB", (1, 1))),
            ("second", PILImage.new("RGB", (1, 1))),
        ],
        "final",
        size="1024x1024",
    )

    assert images.edit_calls[0]["prompt"] == "first\nsecond\nfinal"
    assert images.edit_calls[0]["size"] == "1024x1024"
    assert len(images.edit_calls[0]["image"]) == 2


def test_openai_extracts_image_from_url_and_reports_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UrlResponse:
        def __enter__(self) -> "_UrlResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return _png_bytes((11, 12))

    opened: dict[str, Any] = {}

    def fake_urlopen(url: str, *, timeout: float) -> _UrlResponse:
        opened["url"] = url
        opened["timeout"] = timeout
        return _UrlResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    model, _images = _openai_model_with_response(
        SimpleNamespace(data=[_ImageItem(url="https://image.example/out.png")])
    )

    assert model.generate("prompt").size == (11, 12)
    assert opened["url"] == "https://image.example/out.png"

    with pytest.raises(ValueError, match="No image found"):
        model._extract_image(SimpleNamespace(data=[]))

    with pytest.raises(ValueError, match="No image found"):
        model._extract_image(SimpleNamespace(data=[_ImageItem()]))


class _NIMUrlResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_NIMUrlResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_nim_generate_posts_payload_and_decodes_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any], float]] = []

    def fake_urlopen(req: Any, *, timeout: float) -> _NIMUrlResponse:
        calls.append((req.full_url, json.loads(req.data.decode()), timeout))
        assert req.get_method() == "POST"
        assert req.headers["Authorization"] == "Bearer secret"
        return _NIMUrlResponse(
            {"artifacts": [{"base64": base64.b64encode(_png_bytes((13, 14))).decode()}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    model = NIMImageGenerationModel(
        api_key="secret",
        model="org/model_name",
        base_url="https://nim.example/v1/genai/",
        timeout=9,
    )

    result = model.generate(
        "prompt",
        images=[PILImage.new("RGB", (1, 1))],
        height=64,
    )

    assert result.size == (13, 14)
    assert calls == [
        (
            "https://nim.example/v1/genai/org/model.name",
            {"prompt": "prompt", "height": 64, "width": 1024},
            9,
        )
    ]
    assert model.supports_image_conditioning is False


def test_nim_generate_with_pairs_and_error_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = iter(
        [
            {"artifacts": [{"base64": base64.b64encode(_png_bytes((3, 3))).decode()}]},
            {"artifacts": []},
            {"artifacts": [{"base64": ""}]},
        ]
    )
    requests_seen: list[dict[str, Any]] = []

    def fake_urlopen(req: Any, *, timeout: float) -> _NIMUrlResponse:
        requests_seen.append(json.loads(req.data.decode()))
        return _NIMUrlResponse(next(payloads))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    model = NIMImageGenerationModel(api_key="secret")

    assert model.generate_with_image_prompt_pairs(
        [("desc", "ignored.png")], "final"
    ).size == (
        3,
        3,
    )
    assert requests_seen[0]["prompt"] == "desc\nfinal"

    with pytest.raises(ValueError, match="No artifacts"):
        model.generate("prompt")

    with pytest.raises(ValueError, match="Empty base64"):
        model.generate("prompt")


def test_create_image_generation_model_uses_registry_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.models.backends import registry

    made = object()
    monkeypatch.setattr(
        registry, "get_image_gen_factory", lambda backend: lambda **kwargs: made
    )

    assert create_image_generation_model("custom", api_key="secret") is made
