# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timeout helpers for LangChain NVIDIA NIM chat clients."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _apply_nim_chat_timeout(
    chat_model: Any, timeout: float | None, *, label: str
) -> bool:
    """Apply the timeout and report whether synchronous requests are bounded."""
    if timeout is None:
        return False

    timeout_s = float(timeout)
    configured = False
    sync_configured = False
    for client_attr in ("_client", "_async_client"):
        client = getattr(chat_model, client_attr, None)
        if client is None:
            continue
        # Locked ChatNVIDIA 1.4.3 forwards this field to both the initial POST
        # and every polling GET; the exact-SDK transport test guards that contract.
        client.timeout = timeout_s
        configured = True
        if client_attr == "_client":
            sync_configured = True

    if not configured:
        logger.warning(
            "%s could not apply timeout=%s because ChatNVIDIA exposed no HTTP client.",
            label,
            timeout_s,
        )

    return sync_configured
