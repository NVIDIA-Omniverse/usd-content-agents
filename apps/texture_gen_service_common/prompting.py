# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared prompt-budget helpers for texture generation backends."""

from __future__ import annotations

# Hosted NIM image-generation requests accept at most 800 prompt characters.
NIM_MAX_PROMPT_CHARS = 800
_PROMPT_INSTRUCTION_SEPARATOR = ". "


class PromptBudgetError(ValueError):
    """The user prompt leaves no room for a required generation instruction."""

    def __init__(
        self,
        *,
        prompt_chars: int,
        max_chars: int,
        minimum_instruction_chars: int,
        service_prefix_chars: int = 0,
    ) -> None:
        self.service_prefix_chars = max(0, service_prefix_chars)
        self.composed_prompt_chars = prompt_chars
        self.prompt_chars = max(0, prompt_chars - self.service_prefix_chars)
        self.max_chars = max_chars
        self.minimum_instruction_chars = minimum_instruction_chars
        max_text_prompt_chars = (
            max_chars
            - minimum_instruction_chars
            - self.service_prefix_chars
            - len(_PROMPT_INSTRUCTION_SEPARATOR)
        )
        self.max_text_prompt_chars = max(0, max_text_prompt_chars)
        super().__init__(
            "TEXT_PROMPT_TOO_LONG: the configured image-generation backend "
            f"allows {max_chars} characters and requires room for a channel "
            "instruction; shorten conditioning.text_prompt to at most "
            f"{self.max_text_prompt_chars} characters."
        )


def append_bounded_instruction(
    text_prompt: str,
    instruction: str,
    *,
    max_chars: int | None,
    minimum_instruction: str,
    service_prefix_chars: int = 0,
) -> str:
    """Append a channel instruction within a provider's prompt limit.

    The full caller prompt and full service instruction are retained whenever
    they fit. When only the service-owned instruction overflows, it is shortened
    while reserving a concise channel-specific directive. A caller prompt that
    leaves no room for that directive is rejected before provider launch.
    ``service_prefix_chars`` accounts for channel labels prepended outside the
    caller-owned prompt so the reported usable maximum remains actionable.
    """
    prompt = text_prompt.strip()
    suffix = instruction.strip()
    minimum = minimum_instruction.strip()
    combined = f"{prompt}{_PROMPT_INSTRUCTION_SEPARATOR}{suffix}"
    if max_chars is None or len(combined) <= max_chars:
        return combined

    suffix_budget = max_chars - len(prompt) - len(_PROMPT_INSTRUCTION_SEPARATOR)
    if suffix_budget < len(minimum):
        raise PromptBudgetError(
            prompt_chars=len(prompt),
            max_chars=max_chars,
            minimum_instruction_chars=len(minimum),
            service_prefix_chars=service_prefix_chars,
        )

    remaining_budget = suffix_budget - len(minimum)
    shortened = minimum
    if remaining_budget > 1:
        optional = suffix[: remaining_budget - 1].rstrip()
        if len(optional) == remaining_budget - 1 and " " in optional:
            optional = optional.rsplit(" ", 1)[0].rstrip()
        if optional:
            shortened = f"{minimum} {optional}"
    return f"{prompt}{_PROMPT_INSTRUCTION_SEPARATOR}{shortened}"
