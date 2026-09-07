# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for normalizing provider response content."""

from __future__ import annotations

from typing import Any


def extract_text_content(content: Any) -> str:
    """Return visible text from plain or block-based model content.

    Reasoning-only and empty block lists intentionally return an empty string.
    Callers use that signal to retry requests whose reasoning exhausted the
    shared output-token budget before the model emitted visible text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ]
        return "\n".join(parts)
    return str(content)
