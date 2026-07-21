# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token usage tracking utilities for VLM and LLM invocations.

This module provides utilities to track and aggregate token usage across
model invocations, particularly useful for VLMs where image tokens are included.
"""

import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage statistics for a single model invocation.

    Attributes:
        input_tokens: Number of input/prompt tokens (includes image tokens for VLMs)
        output_tokens: Number of output/completion tokens
        total_tokens: Total tokens (input + output)
        input_token_details: Optional breakdown of input tokens (e.g., cache, audio)
        output_token_details: Optional breakdown of output tokens (e.g., reasoning)
        model_name: Name of the model used
        invocation_type: Type of invocation (e.g., 'vlm', 'llm', 'embedding')
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_details: dict[str, int] | None = None
    output_token_details: dict[str, int] | None = None
    model_name: str | None = None
    invocation_type: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_token_details": self.input_token_details,
            "output_token_details": self.output_token_details,
            "model_name": self.model_name,
            "invocation_type": self.invocation_type,
        }

    @classmethod
    def from_langchain_response(
        cls,
        response: Any,
        model_name: str | None = None,
        invocation_type: str = "unknown",
    ) -> "TokenUsage | None":
        """Create TokenUsage from a LangChain AIMessage response.

        Args:
            response: LangChain AIMessage with usage_metadata
            model_name: Optional model name for tracking
            invocation_type: Type of invocation (e.g., 'vlm', 'llm')

        Returns:
            TokenUsage object if usage metadata exists, None otherwise
        """
        if not hasattr(response, "usage_metadata") or response.usage_metadata is None:
            logger.debug(f"No usage metadata in response for {model_name}")
            return None

        usage = response.usage_metadata

        return cls(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            input_token_details=usage.get("input_token_details"),
            output_token_details=usage.get("output_token_details"),
            model_name=model_name,
            invocation_type=invocation_type,
        )

    def __str__(self) -> str:
        """String representation of token usage."""
        parts = [
            f"TokenUsage(input={self.input_tokens}, "
            f"output={self.output_tokens}, "
            f"total={self.total_tokens}"
        ]

        if self.model_name:
            parts.append(f", model={self.model_name}")

        if self.invocation_type != "unknown":
            parts.append(f", type={self.invocation_type}")

        # Add details if present
        if self.input_token_details:
            parts.append(f", input_details={self.input_token_details}")
        if self.output_token_details:
            parts.append(f", output_details={self.output_token_details}")

        parts.append(")")
        return "".join(parts)


@dataclass
class TokenTracker:
    """Thread-safe aggregator for token usage across multiple invocations.

    This class collects token usage from multiple model calls and provides
    aggregated statistics. It's thread-safe for use in parallel processing.

    Example:
        ```python
        tracker = TokenTracker()

        # During VLM inference
        response = vlm.chat_model.invoke(messages)
        usage = TokenUsage.from_langchain_response(response, model_name="example-vlm-model")
        tracker.add_usage(usage)

        # Get aggregated stats
        stats = tracker.get_stats()
        print(f"Total tokens used: {stats['total_tokens']}")
        print(f"Total cost: ${stats.get('estimated_cost', 0):.4f}")
        ```
    """

    usages: list[TokenUsage] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def add_usage(self, usage: TokenUsage | None) -> None:
        """Add a token usage record (thread-safe).

        Args:
            usage: TokenUsage object to add (None is ignored)
        """
        if usage is None:
            return

        with self._lock:
            self.usages.append(usage)
            logger.debug(f"Added token usage: {usage}")

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated token usage statistics.

        Returns:
            Dictionary containing:
                - total_input_tokens: Sum of all input tokens
                - total_output_tokens: Sum of all output tokens
                - total_tokens: Sum of all tokens
                - invocation_count: Number of model invocations tracked
                - by_model: Dict mapping model names to their token counts
                - by_type: Dict mapping invocation types to their token counts
                - all_usages: List of serialized individual usage dictionaries
        """
        with self._lock:
            stats = {
                "total_input_tokens": sum(u.input_tokens for u in self.usages),
                "total_output_tokens": sum(u.output_tokens for u in self.usages),
                "total_tokens": sum(u.total_tokens for u in self.usages),
                "invocation_count": len(self.usages),
                "by_model": {},
                "by_type": {},
                "all_usages": [usage.to_dict() for usage in self.usages],
            }

            # Aggregate by model
            for usage in self.usages:
                model_key = usage.model_name or "unknown"
                if model_key not in stats["by_model"]:
                    stats["by_model"][model_key] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "count": 0,
                    }
                stats["by_model"][model_key]["input_tokens"] += usage.input_tokens
                stats["by_model"][model_key]["output_tokens"] += usage.output_tokens
                stats["by_model"][model_key]["total_tokens"] += usage.total_tokens
                stats["by_model"][model_key]["count"] += 1

            # Aggregate by invocation type
            for usage in self.usages:
                type_key = usage.invocation_type
                if type_key not in stats["by_type"]:
                    stats["by_type"][type_key] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "count": 0,
                    }
                stats["by_type"][type_key]["input_tokens"] += usage.input_tokens
                stats["by_type"][type_key]["output_tokens"] += usage.output_tokens
                stats["by_type"][type_key]["total_tokens"] += usage.total_tokens
                stats["by_type"][type_key]["count"] += 1

            return stats

    def reset(self) -> None:
        """Clear all tracked usage data (thread-safe)."""
        with self._lock:
            self.usages.clear()
            logger.debug("Token tracker reset")

    def __str__(self) -> str:
        """String representation of aggregated stats."""
        stats = self.get_stats()
        return (
            f"TokenTracker(invocations={stats['invocation_count']}, "
            f"input={stats['total_input_tokens']}, "
            f"output={stats['total_output_tokens']}, "
            f"total={stats['total_tokens']})"
        )


def format_token_stats(stats: dict[str, Any], include_details: bool = True) -> str:
    """Format token statistics as a human-readable string.

    Args:
        stats: Statistics dict from TokenTracker.get_stats()
        include_details: Whether to include per-model and per-type breakdowns

    Returns:
        Formatted string representation
    """
    lines = [
        "Token Usage Statistics:",
        f"  Total Invocations: {stats['invocation_count']}",
        f"  Input Tokens:  {stats['total_input_tokens']:,}",
        f"  Output Tokens: {stats['total_output_tokens']:,}",
        f"  Total Tokens:  {stats['total_tokens']:,}",
    ]

    if include_details and stats.get("by_model"):
        lines.append("\n  By Model:")
        for model, model_stats in stats["by_model"].items():
            lines.append(
                f"    {model}: {model_stats['total_tokens']:,} tokens "
                f"({model_stats['count']} calls, "
                f"in={model_stats['input_tokens']:,}, "
                f"out={model_stats['output_tokens']:,})"
            )

    if include_details and stats.get("by_type"):
        lines.append("\n  By Type:")
        for inv_type, type_stats in stats["by_type"].items():
            lines.append(
                f"    {inv_type}: {type_stats['total_tokens']:,} tokens "
                f"({type_stats['count']} calls, "
                f"in={type_stats['input_tokens']:,}, "
                f"out={type_stats['output_tokens']:,})"
            )

    return "\n".join(lines)
