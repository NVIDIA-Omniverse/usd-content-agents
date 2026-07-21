# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared types and dataclasses for Material Agent API."""

from dataclasses import dataclass, field
from typing import Any


def projected_int(value: Any) -> int:
    """Return an exact integer from public metadata, or a safe typed default."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def projected_float(value: Any) -> float:
    """Return a numeric public metric, or a safe typed default."""
    return (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else 0.0
    )


@dataclass
class APIResult:
    """Base class for API results."""

    success: bool
    error: str | None = None


@dataclass
class MetricsResult:
    """Performance metrics result."""

    functional_correctness_score: float = 0.0
    success_rate: float = 0.0
    exact_match_rate: float = 0.0
    total_cases: int = 0
    valid_cases: int = 0
    successful_cases: int = 0
    exact_matches: int = 0
    failure_count: int = 0
    score_distribution: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricsResult":
        """Create MetricsResult from dictionary."""
        return cls(
            functional_correctness_score=data.get("functional_correctness_score", 0.0),
            success_rate=data.get("success_rate", 0.0),
            exact_match_rate=data.get("exact_match_rate", 0.0),
            total_cases=data.get("total_cases", 0),
            valid_cases=data.get("valid_cases", 0),
            successful_cases=data.get("successful_cases", 0),
            exact_matches=data.get("exact_matches", 0),
            failure_count=data.get("failure_count", 0),
            score_distribution=data.get("score_distribution", {}),
        )

    @classmethod
    def from_projected_dict(cls, data: dict[str, Any]) -> "MetricsResult":
        """Build type-stable metrics from detached public metadata.

        Credential projection can replace rejected values with strings such as
        ``<redacted>``. Public result models must not copy those strings into
        numeric fields, even when the raw runtime mapping used the expected key.
        """
        distribution = data.get("score_distribution")
        safe_distribution = (
            {
                key: value
                for key, value in distribution.items()
                if type(key) is str
                and isinstance(value, int)
                and not isinstance(value, bool)
            }
            if isinstance(distribution, dict)
            else {}
        )
        return cls(
            functional_correctness_score=projected_float(
                data.get("functional_correctness_score")
            ),
            success_rate=projected_float(data.get("success_rate")),
            exact_match_rate=projected_float(data.get("exact_match_rate")),
            total_cases=projected_int(data.get("total_cases")),
            valid_cases=projected_int(data.get("valid_cases")),
            successful_cases=projected_int(data.get("successful_cases")),
            exact_matches=projected_int(data.get("exact_matches")),
            failure_count=projected_int(data.get("failure_count")),
            score_distribution=safe_distribution,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "functional_correctness_score": self.functional_correctness_score,
            "success_rate": self.success_rate,
            "exact_match_rate": self.exact_match_rate,
            "total_cases": self.total_cases,
            "valid_cases": self.valid_cases,
            "successful_cases": self.successful_cases,
            "exact_matches": self.exact_matches,
            "failure_count": self.failure_count,
            "score_distribution": self.score_distribution,
        }


@dataclass
class MaterialSearchResult:
    """Result from material search."""

    material: str
    matches: list[dict[str, Any]]
    match_count: int


@dataclass
class AssignmentStats:
    """Material assignment statistics."""

    materials_created: int = 0
    materials_applied: int = 0
    total_prims: int = 0
    failed: int = 0
    bound_prim_ids: list[str] = field(default_factory=list)
    unbound_prim_ids: list[str] = field(default_factory=list)


@dataclass
class DownloadStats:
    """Material download statistics."""

    found_local: int = 0
    downloaded: int = 0
    failed: int = 0
    skipped: int = 0
