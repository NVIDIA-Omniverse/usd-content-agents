# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Echo backend for testing."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from world_understanding.functions.models.backends.registry import (
    register_chat_backend,
)
from world_understanding.functions.models.chat_models import EchoChatModel


def create_echo_chat(prefix: str = "Echo: ", **kwargs: Any) -> BaseChatModel:
    """Create echo chat model for testing."""
    return EchoChatModel(prefix=prefix)


register_chat_backend("echo", create_echo_chat, requires_api_key=False)
