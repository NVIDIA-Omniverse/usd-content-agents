# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider request policies for model output-token limits."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_OPENAI_OUTPUT_TOKEN_CAPS: tuple[tuple[str, int], ...] = (
    ("gpt-4o-mini", 16_384),
    ("gpt-4o", 16_384),
    ("gpt-4.1-nano", 16_384),
    ("gpt-4.1-mini", 16_384),
    ("gpt-4.1", 16_384),
)
_MODEL_SNAPSHOT_SUFFIX = re.compile(r"(?:\d{8}|\d{4}-\d{2}-\d{2})\Z")
_GPT5_DEPLOYMENT_ALIAS = re.compile(r"(?:^|[^a-z0-9])gpt-5(?:$|[^a-z0-9])")
_MAX_TOKENS_KEY = "max_tokens"
_MAX_COMPLETION_TOKENS_KEY = "max_completion_tokens"


def _canonical_model_name(model_name: str | None) -> str:
    """Return the provider model leaf from plain or catalog-prefixed IDs."""
    if not model_name:
        return ""
    return model_name.strip().lower().rstrip("/").rsplit("/", 1)[-1]


def _matches_model_family(model_name: str, family: str) -> bool:
    """Match an exact family name or one of its dated deployment snapshots."""
    if model_name == family:
        return True
    prefix = f"{family}-"
    return model_name.startswith(prefix) and bool(
        _MODEL_SNAPSHOT_SUFFIX.fullmatch(model_name[len(prefix) :])
    )


def model_output_token_cap(model_name: str | None) -> int | None:
    """Return a known OpenAI-family output cap for a canonical model ID.

    Model IDs may be plain names (``gpt-4o``), dated deployment aliases, or
    provider/catalog-prefixed names (``azure/openai/gpt-4o``). Unknown names
    deliberately return ``None`` so an operator-selected model is not altered
    based on a loose substring match.
    """
    canonical_name = _canonical_model_name(model_name)
    for family, cap in _OPENAI_OUTPUT_TOKEN_CAPS:
        if _matches_model_family(canonical_name, family):
            return cap
    return None


def clamp_model_output_tokens(model_name: str | None, requested: Any) -> Any:
    """Lower an integer request to its known model cap, never raise it."""
    cap = model_output_token_cap(model_name)
    if cap is None or not isinstance(requested, int) or isinstance(requested, bool):
        return requested
    return min(requested, cap)


def openai_token_parameter(model_name: str | None) -> str:
    """Return the Chat Completions token parameter for the model family."""
    canonical_name = _canonical_model_name(model_name)
    if _GPT5_DEPLOYMENT_ALIAS.search(canonical_name):
        return _MAX_COMPLETION_TOKENS_KEY
    return _MAX_TOKENS_KEY


def normalize_openai_token_kwargs(
    model_name: str | None,
    max_tokens: int | None,
    kwargs: Mapping[str, Any],
    *,
    prefer_max_tokens_argument: bool = False,
) -> dict[str, Any]:
    """Normalize aliases, parameter choice, and cap at an OpenAI request edge.

    By default, an explicitly supplied provider-preferred key wins over its
    alternate key, and the alternate wins over the method's ``max_tokens``
    argument. ``prefer_max_tokens_argument`` preserves the public OpenAI
    adapter's historical method-argument precedence. Exactly one provider token
    parameter reaches the request in either mode.
    """
    options = dict(kwargs)
    token_key = openai_token_parameter(model_name)
    alternate_key = (
        _MAX_TOKENS_KEY
        if token_key == _MAX_COMPLETION_TOKENS_KEY
        else _MAX_COMPLETION_TOKENS_KEY
    )

    missing = object()
    selected = options.pop(token_key, missing)
    alternate = options.pop(alternate_key, missing)
    if prefer_max_tokens_argument and max_tokens is not None:
        selected = max_tokens
    elif selected is missing:
        selected = alternate if alternate is not missing else max_tokens
    if selected is not None:
        options[token_key] = clamp_model_output_tokens(model_name, selected)

    return {key: value for key, value in options.items() if value is not None}
