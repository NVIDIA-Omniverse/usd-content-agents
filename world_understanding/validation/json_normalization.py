# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal structured JSON normalization shared by Validation callers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

UnsupportedValuePolicy = Literal["preserve", "stringify"]


@dataclass(frozen=True)
class StructuredJsonNormalizer:
    """Normalize nested values while preserving caller-specific tail behavior."""

    model_types: tuple[type[Any], ...] = ()
    unsupported_value_policy: UnsupportedValuePolicy = "preserve"

    def mapping(self, value: Any) -> dict[str, Any]:
        """Normalize a required mapping, treating ``None`` as an empty mapping."""
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("Expected a mapping")
        return {str(key): self.value(item) for key, item in value.items()}

    def value(self, item: Any) -> Any:
        """Normalize paths, configured models, mappings, and sequences recursively."""
        if isinstance(item, Path):
            return str(item)
        if self.model_types and isinstance(item, self.model_types):
            return item.model_dump(mode="json")
        if isinstance(item, Mapping):
            return {str(key): self.value(value) for key, value in item.items()}
        if isinstance(item, Sequence) and not isinstance(
            item,
            str | bytes | bytearray,
        ):
            return [self.value(value) for value in item]
        if item is None or isinstance(item, str | int | float | bool):
            return item
        if self.unsupported_value_policy == "stringify":
            return str(item)
        return item
