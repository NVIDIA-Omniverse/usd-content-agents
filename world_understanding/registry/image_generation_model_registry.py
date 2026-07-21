# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for the authoritative image backend registry."""

import logging
from collections.abc import Callable
from typing import Any

from world_understanding.functions.models.backends.registry import (
    get_image_gen_factory,
    image_gen_backend_requires_api_key,
    list_image_gen_backends,
    register_image_gen_backend,
)

logger = logging.getLogger(__name__)


class ImageGenerationModelRegistry:
    """Public facade over the runtime image-generation backend registry.

    Instances do not own independent factory state. All methods delegate to the
    backend registry used by :func:`create_image_generation_model`, so shipped
    backends are visible here and registrations immediately affect runtime model
    selection.
    """

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        *,
        requires_api_key: bool | None = None,
    ) -> None:
        """Register an image generation model factory function.

        Args:
            name: Name to register the model under
            factory: Factory function that creates an image generation model
            requires_api_key: Whether callers must resolve an API key before
                selecting the factory. When omitted, an existing backend keeps
                its current setting and a new backend defaults to requiring a
                key.
        """
        try:
            image_gen_backend_requires_api_key(name)
        except ValueError:
            pass
        else:
            logger.warning(
                f"Image generation model '{name}' already registered, overwriting"
            )
        register_image_gen_backend(
            name,
            factory,
            requires_api_key=requires_api_key,
        )
        logger.info(f"Registered image generation model: {name}")

    def get_factory(self, name: str) -> Callable[..., Any] | None:
        """Get an image generation model factory function by name.

        Args:
            name: Name of the image generation model

        Returns:
            Factory function if found, None otherwise
        """
        try:
            return get_image_gen_factory(name)
        except ValueError:
            return None

    def list_models(self) -> list[str]:
        """List all registered image generation model names."""
        return list_image_gen_backends()

    def create_model(self, name: str, **kwargs: Any) -> Any | None:
        """Create an image generation model instance.

        Args:
            name: Name of the image generation model
            **kwargs: Model-specific arguments

        Returns:
            Image generation model instance if found, None otherwise
        """
        factory = self.get_factory(name)
        if factory:
            return factory(**kwargs)
        return None


# Global compatibility facade. The factory state lives only in the backend
# registry; constructing another ImageGenerationModelRegistry exposes the same
# state.
_image_generation_model_registry = ImageGenerationModelRegistry()


def get_image_generation_model_registry() -> ImageGenerationModelRegistry:
    """Get the global image generation model registry."""
    return _image_generation_model_registry
