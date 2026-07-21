# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-fast, value-safe helpers for agent CLI ingress."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import load_config_mapping_from_context

type StepFilterValues = str | Sequence[str] | None


def load_cli_config_mapping(config_path: str | Path) -> dict[str, Any]:
    """Load one CLI YAML file as a non-empty mapping with safe diagnostics."""
    config, _ = load_config_mapping_from_context(
        {"config_path": Path(config_path)},
        missing_file_message="Pipeline configuration file not found",
        read_error_message="Unable to read pipeline configuration",
        parse_error_message="Unable to parse pipeline configuration",
        empty_message="Pipeline configuration is empty",
        file_non_mapping_message="Pipeline configuration must be a mapping",
    )
    return config


def _filter_was_supplied(values: StepFilterValues) -> bool:
    if isinstance(values, str):
        return True
    return bool(values)


def _normalize_step_filter(
    values: StepFilterValues,
    *,
    option_name: str,
    valid_steps: tuple[str, ...],
) -> list[str]:
    if values is None:
        return []

    raw_values: Sequence[str]
    if isinstance(values, str):
        raw_values = values.split(",")
    else:
        raw_values = values

    valid_step_set = set(valid_steps)
    normalized: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            raise ValueError(f"{option_name} step names must be strings")
        step_name = raw_value.strip()
        if not step_name:
            raise ValueError(
                f"{option_name} contains an empty step name. "
                f"Valid steps: {', '.join(valid_steps)}"
            )
        if step_name not in valid_step_set:
            if step_name not in unknown:
                unknown.append(step_name)
            continue
        if step_name not in seen:
            normalized.append(step_name)
            seen.add(step_name)

    if unknown:
        invalid = ", ".join(repr(step_name) for step_name in unknown)
        raise ValueError(
            f"Invalid {option_name} step name(s): {invalid}. "
            f"Valid steps: {', '.join(valid_steps)}"
        )
    return normalized


def normalize_cli_step_filters(
    *,
    skip: StepFilterValues,
    only: StepFilterValues,
    valid_steps: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Normalize CLI step filters and reject every ambiguous spelling early."""
    ordered_valid_steps = tuple(dict.fromkeys(valid_steps))
    if not ordered_valid_steps:
        raise ValueError("The pipeline step registry is empty")

    if _filter_was_supplied(skip) and _filter_was_supplied(only):
        raise ValueError("--skip and --only cannot be used together; choose one.")

    return (
        _normalize_step_filter(
            skip,
            option_name="--skip",
            valid_steps=ordered_valid_steps,
        ),
        _normalize_step_filter(
            only,
            option_name="--only",
            valid_steps=ordered_valid_steps,
        ),
    )


__all__ = [
    "StepFilterValues",
    "load_cli_config_mapping",
    "normalize_cli_step_filters",
]
