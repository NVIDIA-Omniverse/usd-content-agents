# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve parsed scenarios into backend-aware parameter bindings."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .capabilities import (
    BINDING_KIND_SIMULATOR_PARAMETER,
    BINDING_KIND_USD_ATTRIBUTE,
    BINDING_KIND_USD_MASS_SCALE,
    BindingCapability,
    capabilities_for_backend,
)
from .errors import TuningError
from .types import Scenario, TunableParam
from .usd_inspector import UsdTuningReport, inspect_usd_for_tuning

RESOLVED_BINDINGS_EXTRA_KEY = "resolved_parameter_bindings"
RESOLUTION_REPORT_EXTRA_KEY = "parameter_binding_resolution"
AUTHORED_BOUND_MULTIPLIER = 1.1


def _backend_capabilities(backend: Any) -> tuple[BindingCapability, ...]:
    backend_name = str(getattr(backend, "name", type(backend).__name__))
    provider = getattr(backend, "tuning_capabilities", None)
    if not callable(provider):
        try:
            return capabilities_for_backend(backend_name)
        except ValueError as exc:
            raise TuningError(
                f"Backend {backend_name!r} does not declare tuning capabilities."
            ) from exc
    capabilities = provider()
    if not isinstance(capabilities, tuple):
        capabilities = tuple(capabilities)
    if not all(isinstance(c, BindingCapability) for c in capabilities):
        raise TuningError(
            f"Backend {getattr(backend, 'name', type(backend).__name__)!r} returned "
            "invalid tuning capabilities."
        )
    return capabilities


def _resolve_usd_binding(
    *,
    capability: BindingCapability,
    report: UsdTuningReport,
    backend_name: str,
) -> dict[str, Any] | None:
    if capability.schema is None or capability.attribute is None:
        return None
    matches = report.find(
        schema=capability.schema,
        attribute=capability.attribute,
        require_authored_value=capability.requires_authored_value,
    )
    if not matches:
        return None
    return {
        **capability.to_dict(),
        "backend": backend_name,
        "prim_paths": [m.prim_path for m in matches],
        "source": "usd_inspection",
    }


def _resolve_simulator_binding(
    *,
    capability: BindingCapability,
    backend_name: str,
) -> dict[str, Any] | None:
    if capability.simulator_parameter is None:
        return None
    return {
        **capability.to_dict(),
        "backend": backend_name,
        "source": "backend_capability",
    }


def _resolve_param_binding(
    *,
    param_name: str,
    capabilities: tuple[BindingCapability, ...],
    report: UsdTuningReport,
    backend_name: str,
) -> dict[str, Any]:
    candidates = sorted(
        (c for c in capabilities if c.param_name == param_name),
        key=lambda c: c.priority,
        reverse=True,
    )
    if not candidates:
        raise TuningError(
            f"Backend {backend_name!r} does not support tunable parameter "
            f"{param_name!r}."
        )

    for capability in candidates:
        if capability.binding_kind in {
            BINDING_KIND_USD_ATTRIBUTE,
            BINDING_KIND_USD_MASS_SCALE,
        }:
            resolved = _resolve_usd_binding(
                capability=capability,
                report=report,
                backend_name=backend_name,
            )
        elif capability.binding_kind == BINDING_KIND_SIMULATOR_PARAMETER:
            resolved = _resolve_simulator_binding(
                capability=capability,
                backend_name=backend_name,
            )
        else:
            resolved = None
        if resolved is not None:
            return resolved

    raise TuningError(
        f"Could not resolve tunable parameter {param_name!r} for backend "
        f"{backend_name!r} against {report.usd_path}. The backend advertises "
        "a capability, but the USD asset does not expose the required "
        "schema/attribute."
    )


def _binding_source_values(
    *,
    binding: dict[str, Any],
    report: UsdTuningReport,
) -> tuple[float, ...]:
    """Return authored values used to derive one parameter's automatic bounds."""

    if binding.get("kind") == BINDING_KIND_USD_MASS_SCALE:
        # ``mass_scale`` is already relative to every authored mass value.
        return (1.0,)
    if binding.get("kind") == BINDING_KIND_SIMULATOR_PARAMETER:
        raise TuningError(
            "Could not derive automatic tuning bounds for simulator-only "
            f"parameter {binding.get('param')!r} from the USD asset. Specify "
            "min and max explicitly."
        )

    schema = binding.get("schema")
    attribute = binding.get("attribute")
    raw_paths = binding.get("prim_paths")
    if not isinstance(schema, str) or not isinstance(attribute, str):
        raise TuningError(
            "Could not derive automatic tuning bounds for parameter "
            f"{binding.get('param')!r}: its resolved binding does not identify "
            "an authored USD attribute. Specify min and max explicitly."
        )
    prim_paths = set(raw_paths) if isinstance(raw_paths, list) else None
    candidates = report.find(schema=schema, attribute=attribute)
    if prim_paths is not None:
        candidates = tuple(c for c in candidates if c.prim_path in prim_paths)
    if not candidates:
        raise TuningError(
            "Could not derive automatic tuning bounds for parameter "
            f"{binding.get('param')!r}: no matching USD attribute was found. "
            "Specify min and max explicitly."
        )

    values: list[float] = []
    for candidate in candidates:
        raw_value = candidate.current_value
        if not candidate.has_authored_value:
            continue
        if raw_value is None:
            raise TuningError(
                "Could not read the authored USD value used for automatic "
                f"bounds for parameter {binding.get('param')!r} at "
                f"{candidate.prim_path}."
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise TuningError(
                "Could not derive automatic tuning bounds from non-numeric USD "
                f"value {raw_value!r} at {candidate.prim_path}."
            ) from exc
        if not math.isfinite(value):
            raise TuningError(
                "Could not derive automatic tuning bounds from non-finite USD "
                f"value {raw_value!r} at {candidate.prim_path}."
            )
        values.append(value)
    return tuple(values)


def _binding_default_bounds(binding: dict[str, Any]) -> tuple[float, float]:
    raw_range = binding.get("default_range")
    if not isinstance(raw_range, list | tuple) or len(raw_range) != 2:
        raise TuningError(
            f"Parameter {binding.get('param')!r} has no valid capability fallback "
            "range for automatic bounds. Specify min and max explicitly."
        )
    try:
        min_value, max_value = (float(raw_range[0]), float(raw_range[1]))
    except (TypeError, ValueError) as exc:
        raise TuningError(
            f"Parameter {binding.get('param')!r} has a non-numeric capability "
            "fallback range. Specify min and max explicitly."
        ) from exc
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        raise TuningError(
            f"Parameter {binding.get('param')!r} has a non-finite capability "
            "fallback range. Specify min and max explicitly."
        )
    if min_value >= max_value:
        raise TuningError(
            f"Parameter {binding.get('param')!r} has a zero-width or inverted "
            "capability fallback range. Specify min and max explicitly."
        )
    return min_value, max_value


def _derived_bounds(values: tuple[float, ...]) -> tuple[float, float]:
    endpoints = [
        endpoint
        for value in values
        for endpoint in (
            value / AUTHORED_BOUND_MULTIPLIER,
            value * AUTHORED_BOUND_MULTIPLIER,
        )
    ]
    return min(endpoints), max(endpoints)


def _automatic_bounds(
    *,
    parameter_name: str,
    binding: dict[str, Any],
    source_values: tuple[float, ...],
) -> tuple[float, float]:
    if not source_values:
        return _binding_default_bounds(binding)

    min_value, max_value = _derived_bounds(source_values)
    if parameter_name == "restitution":
        # The capability fallback is also restitution's physical domain.
        physical_min, physical_max = _binding_default_bounds(binding)
        min_value = min(max(min_value, physical_min), physical_max)
        max_value = min(max(max_value, physical_min), physical_max)
    if min_value == max_value:
        return _binding_default_bounds(binding)
    return min_value, max_value


def _fallback_conflicting_friction_bounds(
    *,
    scenario: Scenario,
    resolved_params: list[TunableParam],
    bindings_by_param: dict[str, dict[str, Any]],
) -> list[TunableParam]:
    params_by_name = {parameter.name: parameter for parameter in resolved_params}
    static_friction = params_by_name.get("static_friction")
    dynamic_friction = params_by_name.get("dynamic_friction")
    if (
        static_friction is None
        or dynamic_friction is None
        or dynamic_friction.min_value <= static_friction.max_value
    ):
        return resolved_params

    for name in ("static_friction", "dynamic_friction"):
        if name not in scenario.auto_bound_fields:
            continue
        parameter = params_by_name[name]
        min_value, max_value = _binding_default_bounds(bindings_by_param[name])
        if min_value >= max_value:
            raise TuningError(
                f"Capability fallback bounds for parameter {name!r} do not leave "
                f"a tunable interval: {min_value} >= {max_value}. Specify min and "
                "max explicitly."
            )
        params_by_name[name] = TunableParam(
            name=name,
            min_value=min_value,
            max_value=max_value,
            integer=parameter.integer,
        )

    static_friction = params_by_name["static_friction"]
    dynamic_friction = params_by_name["dynamic_friction"]
    if dynamic_friction.min_value > static_friction.max_value:
        raise TuningError(
            "Resolved friction bounds remain infeasible after capability "
            "fallback: dynamic_friction minimum exceeds static_friction maximum."
        )
    return [params_by_name[parameter.name] for parameter in resolved_params]


def _apply_authored_parameter_bounds(
    scenario: Scenario,
    *,
    report: UsdTuningReport,
    bindings: list[dict[str, Any]],
) -> Scenario:
    if not scenario.auto_bound_fields:
        return scenario

    bindings_by_param = {str(binding.get("param")): binding for binding in bindings}
    resolved_params: list[TunableParam] = []
    for parameter in scenario.params:
        if parameter.name not in scenario.auto_bound_fields:
            resolved_params.append(parameter)
            continue
        binding = bindings_by_param[parameter.name]
        source_values = _binding_source_values(binding=binding, report=report)
        derived_min, derived_max = _automatic_bounds(
            parameter_name=parameter.name,
            binding=binding,
            source_values=source_values,
        )
        min_value, max_value = derived_min, derived_max
        if min_value > max_value:
            raise TuningError(
                f"Resolved bounds for parameter {parameter.name!r} have min > max: "
                f"{min_value} > {max_value}."
            )
        if min_value == max_value:
            raise TuningError(
                f"Automatic bounds for parameter {parameter.name!r} collapse to "
                f"the fixed value {min_value}. Specify min and max explicitly "
                "to keep this parameter tunable."
            )
        resolved_params.append(
            TunableParam(
                name=parameter.name,
                min_value=min_value,
                max_value=max_value,
                integer=parameter.integer,
            )
        )

    resolved_params = _fallback_conflicting_friction_bounds(
        scenario=scenario,
        resolved_params=resolved_params,
        bindings_by_param=bindings_by_param,
    )

    return Scenario(
        name=scenario.name,
        params=tuple(resolved_params),
        target=scenario.target,
        metric=scenario.metric,
        extra=scenario.extra,
    )


def _resolution_inputs(
    scenario: Scenario,
    *,
    physics_usd: Path | str,
    backend: Any,
) -> tuple[str, UsdTuningReport, list[dict[str, Any]]]:
    backend_name = str(getattr(backend, "name", type(backend).__name__))
    capabilities = _backend_capabilities(backend)
    report = inspect_usd_for_tuning(physics_usd)
    bindings = [
        _resolve_param_binding(
            param_name=parameter.name,
            capabilities=capabilities,
            report=report,
            backend_name=backend_name,
        )
        for parameter in scenario.params
    ]
    return backend_name, report, bindings


def resolve_scenario_parameter_bounds(
    scenario: Scenario,
    *,
    physics_usd: Path | str,
    backend: Any,
) -> Scenario:
    """Resolve omitted parameter bounds against authored USD values."""

    if not scenario.auto_bound_fields:
        return scenario
    _backend_name, report, bindings = _resolution_inputs(
        scenario,
        physics_usd=physics_usd,
        backend=backend,
    )
    return _apply_authored_parameter_bounds(
        scenario,
        report=report,
        bindings=bindings,
    )


def resolve_scenario_bindings(
    scenario: Scenario,
    *,
    physics_usd: Path | str,
    backend: Any,
) -> Scenario:
    """Return a copy of ``scenario`` with resolved parameter bindings in extra."""

    backend_name, report, bindings = _resolution_inputs(
        scenario,
        physics_usd=physics_usd,
        backend=backend,
    )
    scenario = _apply_authored_parameter_bounds(
        scenario,
        report=report,
        bindings=bindings,
    )

    extra = dict(scenario.extra)
    extra[RESOLVED_BINDINGS_EXTRA_KEY] = bindings
    extra[RESOLUTION_REPORT_EXTRA_KEY] = {
        "backend": backend_name,
        "usd_path": str(report.usd_path),
        "candidate_count": len(report.candidates),
    }
    return Scenario(
        name=scenario.name,
        params=scenario.params,
        target=scenario.target,
        metric=scenario.metric,
        extra=extra,
    )


def get_resolved_bindings(scenario: Scenario) -> list[dict[str, Any]] | None:
    """Return bindings stored by :func:`resolve_scenario_bindings`, if any."""

    raw = scenario.extra.get(RESOLVED_BINDINGS_EXTRA_KEY)
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TuningError(
            f"Scenario extra {RESOLVED_BINDINGS_EXTRA_KEY!r} must be a list, "
            f"got {type(raw).__name__}."
        )
    bindings: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TuningError(
                f"Scenario binding {RESOLVED_BINDINGS_EXTRA_KEY}[{i}] must be "
                f"a mapping, got {type(item).__name__}."
            )
        bindings.append(dict(item))
    return bindings


__all__ = [
    "RESOLVED_BINDINGS_EXTRA_KEY",
    "RESOLUTION_REPORT_EXTRA_KEY",
    "AUTHORED_BOUND_MULTIPLIER",
    "get_resolved_bindings",
    "resolve_scenario_parameter_bounds",
    "resolve_scenario_bindings",
]
