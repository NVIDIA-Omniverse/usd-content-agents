# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public model-registry compatibility facades."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from PIL import Image as PILImage

from world_understanding.functions.models import backends as _backends  # noqa: F401
from world_understanding.functions.models.backends import registry as backend_registry
from world_understanding.functions.models.backends.registry import (
    chat_backend_requires_api_key,
    image_gen_backend_requires_api_key,
    register_chat_backend,
    register_image_gen_backend,
)
from world_understanding.functions.models.chat_models import (
    EchoChatModel,
    create_chat_model,
    create_chat_model_from_config,
    create_echo_chat_model,
    create_nim_chat_model,
)
from world_understanding.functions.models.image_generation_models import (
    BaseImageGenerationModel,
    create_image_generation_model,
)
from world_understanding.registry import (
    ChatModelRegistry,
    ImageGenerationModelRegistry,
    get_chat_model_registry,
    get_image_generation_model_registry,
)


class _ImageModel(BaseImageGenerationModel):
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> PILImage.Image:
        return PILImage.new("RGB", (1, 1))

    def generate_with_image_prompt_pairs(
        self,
        image_prompt_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        **kwargs: Any,
    ) -> PILImage.Image:
        return PILImage.new("RGB", (1, 1))

    @property
    def model_name(self) -> str:
        return "test-image"

    @property
    def backend_name(self) -> str:
        return "custom-image"


@pytest.fixture
def isolated_backend_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Copy backend state so compatibility registrations cannot leak."""
    monkeypatch.setattr(
        backend_registry, "_chat_backends", dict(backend_registry._chat_backends)
    )
    monkeypatch.setattr(
        backend_registry,
        "_chat_backend_requires_api_key",
        dict(backend_registry._chat_backend_requires_api_key),
    )
    monkeypatch.setattr(
        backend_registry,
        "_image_gen_backends",
        dict(backend_registry._image_gen_backends),
    )
    monkeypatch.setattr(
        backend_registry,
        "_image_gen_backend_requires_api_key",
        dict(backend_registry._image_gen_backend_requires_api_key),
    )


def test_public_chat_registry_reflects_shipped_backends() -> None:
    registry = ChatModelRegistry()

    assert {"anthropic", "echo", "gemini", "nim", "openai"} <= set(
        registry.list_models()
    )
    assert registry.get_factory("echo") is backend_registry.get_chat_factory("echo")
    assert registry.get_factory("missing") is None
    assert registry.create_model("missing") is None
    assert get_chat_model_registry().list_models() == registry.list_models()


def test_public_image_gen_registry_is_populated() -> None:
    registry = ImageGenerationModelRegistry()

    assert {"gemini", "nim", "openai"} <= set(registry.list_models())
    assert registry.get_factory("gemini") is backend_registry.get_image_gen_factory(
        "gemini"
    )
    assert registry.get_factory("missing") is None
    assert registry.create_model("missing") is None
    assert get_image_generation_model_registry().list_models() == registry.list_models()


def test_public_chat_registration_affects_runtime_selection(
    isolated_backend_registries: None,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for prefix in ("WU", "PA", "TA", "MA"):
        monkeypatch.delenv(f"{prefix}_LLM_NIM_BASE_URL", raising=False)
        monkeypatch.delenv(f"{prefix}_VLM_NIM_BASE_URL", raising=False)
    registry = ChatModelRegistry()

    def create_custom_chat(prefix: str = "Custom: ", **_kwargs: Any) -> EchoChatModel:
        return create_echo_chat_model(prefix=prefix)

    registry.register("custom-chat", create_custom_chat, requires_api_key=False)

    assert registry.get_factory("custom-chat") is create_custom_chat
    assert backend_registry.get_chat_factory("custom-chat") is create_custom_chat
    assert chat_backend_requires_api_key("custom-chat") is False
    selected = create_chat_model(backend="custom-chat", prefix="Selected: ")
    assert isinstance(selected, EchoChatModel)
    assert selected.prefix == "Selected: "
    selected_from_config = create_chat_model_from_config({"backend": "custom-chat"})
    assert isinstance(selected_from_config, EchoChatModel)
    selected_from_facade = registry.create_model("custom-chat", prefix="Facade: ")
    assert isinstance(selected_from_facade, EchoChatModel)
    assert selected_from_facade.prefix == "Facade: "

    with caplog.at_level(logging.WARNING):
        registry.register("custom-chat", create_custom_chat)
    assert "already registered" in caplog.text
    assert chat_backend_requires_api_key("custom-chat") is False


def test_public_image_generation_registration_affects_runtime_selection(
    isolated_backend_registries: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = ImageGenerationModelRegistry()
    model = _ImageModel()

    def create_custom_image(**_kwargs: Any) -> BaseImageGenerationModel:
        return model

    registry.register("custom-image", create_custom_image, requires_api_key=False)

    assert registry.get_factory("custom-image") is create_custom_image
    assert backend_registry.get_image_gen_factory("custom-image") is create_custom_image
    assert image_gen_backend_requires_api_key("custom-image") is False
    assert create_image_generation_model("custom-image") is model
    assert registry.create_model("custom-image") is model

    with caplog.at_level(logging.WARNING):
        registry.register("custom-image", create_custom_image)
    assert "already registered" in caplog.text
    assert image_gen_backend_requires_api_key("custom-image") is False


def test_facade_overwrite_preserves_shipped_backend_auth_metadata(
    isolated_backend_registries: None,
) -> None:
    chat = ChatModelRegistry()
    image = ImageGenerationModelRegistry()

    echo_factory = backend_registry.get_chat_factory("echo")
    nim_image_factory = backend_registry.get_image_gen_factory("nim")
    chat.register("echo", echo_factory)
    image.register("nim", nim_image_factory)

    assert chat_backend_requires_api_key("echo") is False
    assert image_gen_backend_requires_api_key("nim") is True


def test_authoritative_overwrite_preserves_or_explicitly_changes_auth_metadata(
    isolated_backend_registries: None,
) -> None:
    echo_factory = backend_registry.get_chat_factory("echo")
    nim_image_factory = backend_registry.get_image_gen_factory("nim")

    register_chat_backend("echo", echo_factory)
    register_image_gen_backend("nim", nim_image_factory)
    assert chat_backend_requires_api_key("echo") is False
    assert image_gen_backend_requires_api_key("nim") is True

    register_chat_backend("echo", echo_factory, requires_api_key=True)
    register_image_gen_backend("nim", nim_image_factory, requires_api_key=False)
    assert chat_backend_requires_api_key("echo") is True
    assert image_gen_backend_requires_api_key("nim") is False


@pytest.mark.parametrize("wrap_factory", [False, True], ids=["direct", "lambda"])
def test_legacy_nim_wrapper_registration_does_not_recurse(
    isolated_backend_registries: None,
    monkeypatch: pytest.MonkeyPatch,
    wrap_factory: bool,
) -> None:
    from world_understanding.functions.models.backends.public import nim

    model = create_echo_chat_model()

    def create_test_nim(**_kwargs: Any) -> EchoChatModel:
        return model

    monkeypatch.setattr(nim, "create_nim_chat", create_test_nim)
    registry = ChatModelRegistry()
    wrapper_calls: list[dict[str, Any]] = []

    def wrapped_nim(**kwargs: Any) -> BaseChatModel:
        wrapper_calls.append(kwargs)
        return create_nim_chat_model(**kwargs)

    factory = wrapped_nim if wrap_factory else create_nim_chat_model
    registry.register("nim", factory)

    assert registry.create_model("nim", api_key="test") is model
    assert create_chat_model(backend="nim", api_key="test") is model
    assert len(wrapper_calls) == (2 if wrap_factory else 0)


def test_new_facade_registrations_require_api_keys_by_default(
    isolated_backend_registries: None,
) -> None:
    def create_custom_chat(**_kwargs: Any) -> EchoChatModel:
        return create_echo_chat_model()

    def create_custom_image(**_kwargs: Any) -> BaseImageGenerationModel:
        return _ImageModel()

    ChatModelRegistry().register("secure-chat", create_custom_chat)
    ImageGenerationModelRegistry().register("secure-image", create_custom_image)

    assert chat_backend_requires_api_key("secure-chat") is True
    assert image_gen_backend_requires_api_key("secure-image") is True
