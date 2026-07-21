# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for the display formatter registry."""

from __future__ import annotations

from typing import Any

from world_understanding.registry.display_registry import (
    DisplayRegistry,
    get_display_registry,
)


def test_display_registry_register_get_and_display() -> None:
    registry = DisplayRegistry()
    calls: list[tuple[dict[str, Any], object, str]] = []

    def formatter(outputs: dict[str, Any], console: object, indent: str) -> None:
        calls.append((outputs, console, indent))

    console = object()

    assert registry.get_formatter("missing") is None
    assert registry.has_formatter("missing") is False
    assert registry.display("missing", {"value": 1}, console) is False

    registry.register("tool", formatter)

    assert registry.get_formatter("tool") is formatter
    assert registry.has_formatter("tool") is True
    assert registry.display("tool", {"value": 2}, console, indent="  ") is True
    assert calls == [({"value": 2}, console, "  ")]
    assert get_display_registry().has_formatter("unlikely-test-tool") is False
