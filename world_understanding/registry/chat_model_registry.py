# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for the authoritative chat backend registry."""

import logging
from collections.abc import Callable
from typing import Any

from world_understanding.functions.models.backends.registry import (
    chat_backend_requires_api_key,
    get_chat_factory,
    list_chat_backends,
    register_chat_backend,
)

logger = logging.getLogger(__name__)


class ChatModelRegistry:
    """Public facade over the runtime chat backend registry.

    Instances do not own independent factory state. All methods delegate to the
    backend registry used by :func:`create_chat_model`, so shipped backends are
    visible here and registrations immediately affect runtime model selection.
    """

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        *,
        requires_api_key: bool | None = None,
    ) -> None:
        """Register a chat model factory function.

        Args:
            name: Name to register the model under
            factory: Factory function that creates a chat model
            requires_api_key: Whether config-based provisioning must resolve an
                API key before calling the factory. When omitted, an existing
                backend keeps its current setting and a new backend defaults to
                requiring a key.
        """
        try:
            chat_backend_requires_api_key(name)
        except ValueError:
            pass
        else:
            logger.warning(f"Chat model '{name}' already registered, overwriting")
        register_chat_backend(
            name,
            factory,
            requires_api_key=requires_api_key,
        )
        logger.info(f"Registered chat model: {name}")

    def get_factory(self, name: str) -> Callable[..., Any] | None:
        """Get a chat model factory function by name.

        Args:
            name: Name of the chat model

        Returns:
            Factory function if found, None otherwise
        """
        try:
            return get_chat_factory(name)
        except ValueError:
            return None

    def list_models(self) -> list[str]:
        """List all registered chat model names."""
        return list_chat_backends()

    def create_model(self, name: str, **kwargs: Any) -> Any | None:
        """Create a chat model instance.

        Args:
            name: Name of the chat model
            **kwargs: Model-specific arguments

        Returns:
            Chat model instance if found, None otherwise
        """
        factory = self.get_factory(name)
        if factory:
            return factory(**kwargs)
        return None


# Global compatibility facade. The factory state lives only in the backend
# registry; constructing another ChatModelRegistry exposes the same state.
_chat_model_registry = ChatModelRegistry()


def get_chat_model_registry() -> ChatModelRegistry:
    """Get the global chat model registry."""
    return _chat_model_registry
