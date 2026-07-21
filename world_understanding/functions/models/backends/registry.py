# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend registry for model factories.

Each backend registers a factory function that takes **kwargs and returns
a model instance. The factory functions are responsible for extracting
and validating their own parameters from kwargs.
"""

from collections.abc import Callable
from importlib import metadata

from langchain_core.language_models.chat_models import BaseChatModel

from world_understanding.functions.models.image_generation_models import (
    BaseImageGenerationModel,
)
from world_understanding.functions.models.text_embedding_models import (
    BaseTextEmbeddingModel,
)
from world_understanding.functions.models.vision_language_models import (
    BaseVisionLanguageModel,
)

# Type aliases for factory functions
ChatFactory = Callable[..., BaseChatModel]
VLMFactory = Callable[..., BaseVisionLanguageModel]
ImageGenFactory = Callable[..., BaseImageGenerationModel]
TextEmbeddingFactory = Callable[..., BaseTextEmbeddingModel]

# Registries: backend name -> factory function
_chat_backends: dict[str, ChatFactory] = {}
_chat_backend_requires_api_key: dict[str, bool] = {}
_vlm_backends: dict[str, VLMFactory] = {}
_vlm_backend_requires_api_key: dict[str, bool] = {}
_vlm_backend_capabilities: dict[str, frozenset[str]] = {}
_image_gen_backends: dict[str, ImageGenFactory] = {}
_image_gen_backend_requires_api_key: dict[str, bool] = {}
_text_embedding_backends: dict[str, TextEmbeddingFactory] = {}
_BACKEND_PLUGIN_GROUP = "world_understanding.model_backends"
_loaded_backend_plugins: set[str] = set()


def load_backend_plugins() -> tuple[str, ...]:
    """Load installed model-backend plugins through package entry points.

    The public package does not know plugin package or backend names. Optional
    distributions register a zero-argument callable in
    ``world_understanding.model_backends``; importing/loading that callable may
    register any combination of chat, VLM, and image-generation factories.
    """
    entry_points = metadata.entry_points(group=_BACKEND_PLUGIN_GROUP)
    for entry_point in entry_points:
        plugin_id = f"{entry_point.name}:{entry_point.value}"
        if plugin_id in _loaded_backend_plugins:
            continue
        registrar = entry_point.load()
        if not callable(registrar):
            raise TypeError(
                f"Model backend plugin {entry_point.name!r} must be callable"
            )
        registrar()
        _loaded_backend_plugins.add(plugin_id)
    return tuple(sorted(_loaded_backend_plugins))


def list_loaded_backend_plugins() -> tuple[str, ...]:
    """Return identifiers for successfully loaded backend entry points."""
    return tuple(sorted(_loaded_backend_plugins))


def register_chat_backend(
    name: str, factory: ChatFactory, *, requires_api_key: bool | None = None
) -> None:
    """Register a chat model backend factory.

    Omitting ``requires_api_key`` preserves an existing backend's setting and
    defaults a new backend to requiring credentials.
    """
    if requires_api_key is None:
        requires_api_key = _chat_backend_requires_api_key.get(name, True)
    _chat_backends[name] = factory
    _chat_backend_requires_api_key[name] = requires_api_key


def chat_backend_requires_api_key(name: str) -> bool:
    """Return whether a registered chat backend requires an API key."""
    if name not in _chat_backends:
        get_chat_factory(name)
    return _chat_backend_requires_api_key[name]


def register_vlm_backend(
    name: str,
    factory: VLMFactory,
    *,
    requires_api_key: bool = True,
    capabilities: frozenset[str] = frozenset(),
) -> None:
    """Register a VLM backend factory."""
    _vlm_backends[name] = factory
    _vlm_backend_requires_api_key[name] = requires_api_key
    _vlm_backend_capabilities[name] = capabilities


def register_image_gen_backend(
    name: str, factory: ImageGenFactory, *, requires_api_key: bool | None = None
) -> None:
    """Register an image generation backend factory.

    Omitting ``requires_api_key`` preserves an existing backend's setting and
    defaults a new backend to requiring credentials.
    """
    if requires_api_key is None:
        requires_api_key = _image_gen_backend_requires_api_key.get(name, True)
    _image_gen_backends[name] = factory
    _image_gen_backend_requires_api_key[name] = requires_api_key


def vlm_backend_requires_api_key(name: str) -> bool:
    """Return whether a registered VLM backend requires an API key."""
    if name not in _vlm_backends:
        get_vlm_factory(name)
    return _vlm_backend_requires_api_key[name]


def vlm_backend_supports(name: str, capability: str) -> bool:
    """Return whether a registered VLM backend declares a capability."""
    if name not in _vlm_backends:
        get_vlm_factory(name)
    return capability in _vlm_backend_capabilities[name]


def image_gen_backend_requires_api_key(name: str) -> bool:
    """Return whether a registered image backend requires an API key."""
    if name not in _image_gen_backends:
        get_image_gen_factory(name)
    return _image_gen_backend_requires_api_key[name]


def register_text_embedding_backend(name: str, factory: TextEmbeddingFactory) -> None:
    """Register a text-embedding backend factory."""
    _text_embedding_backends[name] = factory


def get_chat_factory(name: str) -> ChatFactory:
    """Get a registered chat backend factory by name."""
    if name not in _chat_backends:
        available = ", ".join(sorted(_chat_backends.keys()))
        raise ValueError(
            f"Unknown chat backend: {name}. Available backends: {available}"
        )
    return _chat_backends[name]


def get_vlm_factory(name: str) -> VLMFactory:
    """Get a registered VLM backend factory by name."""
    if name not in _vlm_backends:
        available = ", ".join(sorted(_vlm_backends.keys()))
        raise ValueError(
            f"Unknown VLM backend: {name}. Available backends: {available}"
        )
    return _vlm_backends[name]


def get_image_gen_factory(name: str) -> ImageGenFactory:
    """Get a registered image generation backend factory by name."""
    if name not in _image_gen_backends:
        available = ", ".join(sorted(_image_gen_backends.keys()))
        raise ValueError(
            f"Unknown image generation backend: {name}. Available backends: {available}"
        )
    return _image_gen_backends[name]


def get_text_embedding_factory(name: str) -> TextEmbeddingFactory:
    """Get a registered text-embedding backend factory by name."""
    if name not in _text_embedding_backends:
        available = ", ".join(sorted(_text_embedding_backends))
        raise ValueError(
            f"Unknown text-embedding backend: {name}. Available backends: {available}"
        )
    return _text_embedding_backends[name]


def list_chat_backends() -> list[str]:
    """List all registered chat backend names."""
    return sorted(_chat_backends.keys())


def list_vlm_backends() -> list[str]:
    """List all registered VLM backend names."""
    return sorted(_vlm_backends.keys())


def list_image_gen_backends() -> list[str]:
    """List all registered image generation backend names."""
    return sorted(_image_gen_backends.keys())


def list_text_embedding_backends() -> list[str]:
    """List all registered text-embedding backend names."""
    return sorted(_text_embedding_backends)
