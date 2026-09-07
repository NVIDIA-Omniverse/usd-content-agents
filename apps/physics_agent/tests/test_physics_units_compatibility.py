# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility coverage for Physics Agent unit conversions."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from physics_agent import physics_units


def _load_compatibility_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    import_override: Callable[..., Any] | None = None,
) -> ModuleType:
    module_path = Path(physics_units.__file__).resolve()
    module_name = "physics_agent._physics_units_fallback_test"
    if import_override is None:
        monkeypatch.setitem(
            sys.modules,
            "world_understanding.utils.physics_units",
            None,
        )
    else:
        monkeypatch.setattr("builtins.__import__", import_override)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_physics_unit_helpers_are_used_when_available() -> None:
    shared = pytest.importorskip("world_understanding.utils.physics_units")

    assert (
        physics_units.acceleration_m_per_s2_to_stage_units
        is shared.acceleration_m_per_s2_to_stage_units
    )
    assert physics_units.stage_units_per_meter is shared.stage_units_per_meter
    assert physics_units.validate_meters_per_unit is shared.validate_meters_per_unit
    assert physics_units.STANDARD_GRAVITY_M_PER_S2 == pytest.approx(9.81)
    assert physics_units.acceleration_m_per_s2_to_stage_units(
        9.81,
        0.01,
    ) == pytest.approx(981.0)


def test_fallback_physics_unit_helpers_support_legacy_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = _load_compatibility_module(monkeypatch)

    assert fallback.STANDARD_GRAVITY_M_PER_S2 == pytest.approx(9.81)
    assert fallback.stage_units_per_meter(0.01) == pytest.approx(100.0)
    assert fallback.acceleration_m_per_s2_to_stage_units(
        1.62,
        0.01,
    ) == pytest.approx(162.0)
    with pytest.raises(ValueError, match="got bool"):
        fallback.validate_meters_per_unit(True)
    with pytest.raises(ValueError, match="must be finite"):
        fallback.validate_meters_per_unit(float("inf"))
    with pytest.raises(ValueError, match="must be positive"):
        fallback.validate_meters_per_unit(0.0)
    with pytest.raises(ValueError, match="acceleration_m_per_s2"):
        fallback.acceleration_m_per_s2_to_stage_units(float("nan"), 1.0)


def test_fallback_reraises_missing_shared_module_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def missing_nested_dependency(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "world_understanding.utils.physics_units":
            raise ModuleNotFoundError(
                "No module named 'nested_dependency'",
                name="nested_dependency",
            )
        return real_import(name, globals, locals, fromlist, level)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        _load_compatibility_module(
            monkeypatch,
            import_override=missing_nested_dependency,
        )

    assert exc_info.value.name == "nested_dependency"
