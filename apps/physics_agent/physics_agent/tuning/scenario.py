# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario YAML loading + validation.

Two scenario kinds are supported:

**drop_settle** (locked schema — drop a rigid body and measure how it settles)::

    name: drop_settle
    metric: settle_distance
    target:
      drop_height_m: 0.5
      duration_s: 2.0
      gravity: -9.81
    parameters:
      - name: mass_scale
        min: 0.5
        max: 2.0
      - name: static_friction
        min: 0.1
        max: 1.0

**freeform** (LLM-authored, NL-driven — single-rigid-body scene with free
initial conditions; the backend reads ``target`` keys it understands and
fills sensible defaults for the rest)::

    name: freeform
    metric: judge_score
    target:
      description: "spin a top on a smooth surface"
      duration_s: 3.0
      gravity: -9.81
      initial_pose:
        position: [0.0, 0.5, 0.0]
        rotation: [0.0, 0.0, 0.0]   # XYZ Euler radians
      initial_velocity: [0.0, 0.0, 0.0]
      initial_angular_velocity: [0.0, 30.0, 0.0]   # rad/s
      surface:
        friction: 0.5
      observations:
        - "did the top fall over"
    parameters:
      - name: mass_scale
        min: 0.5
        max: 2.0

Only fields defined in this module are honoured. Anything else under ``extra``
is preserved so backends can read scenario-specific knobs without forking the
parser, but is *not* type-checked here. Freeform target keys are validated by
the backend at evaluation time, not by this parser, so the LLM interpreter has
room to author scenario-specific knobs without churning the parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import (
    DEFAULT_PARAM_BOUNDS,
    SUPPORTED_PARAM_KEYS,
    SUPPORTED_SCENARIOS,
    Scenario,
    TunableParam,
)


class ScenarioParseError(ValueError):
    """Raised when a scenario YAML fails validation.

    Subclasses ValueError so callers can ``except ValueError`` if they want to
    handle parse + validation errors uniformly, but FastAPI handlers can also
    target this subclass specifically.
    """


def _coerce_float(value: Any, *, field: str) -> float:
    import math

    if isinstance(value, bool):
        # bool is a subclass of int in Python — explicitly reject to avoid
        # silently treating ``True`` as ``1.0``.
        raise ScenarioParseError(
            f"Field {field!r} must be a number, got bool: {value!r}"
        )
    if isinstance(value, int | float):
        f = float(value)
        # Reject NaN / +Inf / -Inf — YAML accepts ``.nan`` and ``.inf`` and
        # they would otherwise propagate into _params_from_vector and produce
        # nan/inf parameter values that confuse every optimizer.
        if not math.isfinite(f):
            raise ScenarioParseError(
                f"Field {field!r} must be a finite number, got {f!r}"
            )
        return f
    raise ScenarioParseError(
        f"Field {field!r} must be a number, got {type(value).__name__}: {value!r}"
    )


def _parse_params(raw: Any) -> tuple[TunableParam, ...]:
    if not isinstance(raw, list):
        raise ScenarioParseError(
            f"'parameters' must be a list, got {type(raw).__name__}"
        )
    if not raw:
        raise ScenarioParseError(
            "'parameters' must contain at least one tunable parameter"
        )
    params: list[TunableParam] = []
    seen_names: dict[str, int] = {}
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ScenarioParseError(
                f"parameters[{i}] must be a mapping, got {type(entry).__name__}"
            )
        if "name" not in entry:
            raise ScenarioParseError(f"parameters[{i}] missing required key 'name'")
        name = entry["name"]
        if not isinstance(name, str):
            raise ScenarioParseError(
                f"parameters[{i}].name must be a string, got {type(name).__name__}"
            )
        if name not in SUPPORTED_PARAM_KEYS:
            raise ScenarioParseError(
                f"parameters[{i}].name = {name!r} is not a supported tunable "
                f"parameter. Supported: {sorted(SUPPORTED_PARAM_KEYS)}"
            )
        if name in seen_names:
            raise ScenarioParseError(
                f"parameters[{i}].name = {name!r} duplicates "
                f"parameters[{seen_names[name]}]"
            )
        seen_names[name] = i
        default_lo, default_hi = DEFAULT_PARAM_BOUNDS[name]
        lo = _coerce_float(entry.get("min", default_lo), field=f"parameters[{i}].min")
        hi = _coerce_float(entry.get("max", default_hi), field=f"parameters[{i}].max")
        if lo > hi:
            raise ScenarioParseError(
                f"parameters[{i}] ({name}) has min > max: {lo} > {hi}"
            )
        params.append(TunableParam(name=name, min_value=lo, max_value=hi))
    params_by_name = {param.name: param for param in params}
    static_param = params_by_name.get("static_friction")
    dynamic_param = params_by_name.get("dynamic_friction")
    if (
        static_param is not None
        and dynamic_param is not None
        and dynamic_param.min_value > static_param.max_value
    ):
        raise ScenarioParseError(
            "dynamic_friction minimum must not exceed static_friction maximum"
        )
    return tuple(params)


def parse_scenario(raw: dict[str, Any]) -> Scenario:
    """Validate a scenario dict and return a :class:`Scenario`.

    Raises:
        ScenarioParseError: when required keys are missing, types are wrong,
            or values are out of allowed ranges.
    """
    if not isinstance(raw, dict):
        raise ScenarioParseError(
            f"Scenario must be a mapping, got {type(raw).__name__}"
        )

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ScenarioParseError("Scenario 'name' is required and must be a string")
    if name not in SUPPORTED_SCENARIOS:
        raise ScenarioParseError(
            f"Unsupported scenario name {name!r}. "
            f"v1 supports: {sorted(SUPPORTED_SCENARIOS)}"
        )

    metric = raw.get("metric")
    if metric is None:
        # Default per scenario kind (CodeRabbit Round 11 thread #11): the
        # previous unconditional ``"settle_distance"`` default leaked into
        # freeform runs and silently switched the objective from the
        # judge-driven ``judge_score`` to a metric that
        # ``freeform.evaluate`` does not honour. Each scenario kind owns
        # its default; new kinds add an entry here.
        if name == "drop_settle":
            metric = "settle_distance"
        elif name == "freeform":
            metric = "judge_score"
        else:  # pragma: no cover - guarded by SUPPORTED_SCENARIOS validation
            raise ScenarioParseError(
                f"No default metric configured for scenario {name!r}"
            )
    if not isinstance(metric, str) or not metric:
        raise ScenarioParseError("'metric' must be a non-empty string when provided")
    # Validate drop_settle metrics against the registry HERE so a typo
    # surfaces as a clean parse error before the runner enters the
    # per-trial loop. Without this gate the per-trial backend would
    # raise on every trial and the optimizer would burn the full budget
    # producing a generic "all trials failed" outcome instead of a
    # readable "Unsupported drop_settle metric" message. The import is
    # local to keep ``parse_scenario``'s import surface unchanged
    # (matching the design constraint at the top of judge_tune.py).
    if name == "drop_settle":
        from physics_agent.tuning.scenarios.drop_settle import _METRICS

        if metric not in _METRICS:
            raise ScenarioParseError(
                f"Unsupported drop_settle metric {metric!r}; "
                f"choose from {sorted(_METRICS)} or omit ``metric`` "
                "to default to 'settle_distance'."
            )
    elif name == "freeform":
        # ``freeform.evaluate`` always returns ``score = 1 - hybrid``;
        # ``scenario.metric`` is preserved only as a label in artifacts.
        # Reject any value other than ``judge_score`` at parse time so a
        # YAML/LLM typo like ``metric: settle_distance`` does not silently
        # mislabel artifacts and confuse downstream consumers that key
        # off ``metric``. (CodeRabbit R13 thread #2, follow-on.)
        if metric != "judge_score":
            raise ScenarioParseError(
                f"Unsupported freeform metric {metric!r}; "
                "freeform only supports 'judge_score' (omit ``metric`` "
                "to inherit the default)."
            )

    target = raw.get("target", {})
    if target is None:
        target = {}
    if not isinstance(target, dict):
        raise ScenarioParseError(
            f"'target' must be a mapping, got {type(target).__name__}"
        )
    # For drop_settle the recognised target keys are numeric simulation
    # knobs. Validate the known ones at parse time so a malformed value
    # (e.g. ``drop_height_m: [1, 2]``) surfaces here with a helpful
    # ``target.<key>`` path instead of failing inside the backend.
    if name == "drop_settle":
        _DROP_SETTLE_TARGET_NUMERIC_KEYS = (
            "drop_height_m",
            "duration_s",
            "gravity",
        )
        for key in _DROP_SETTLE_TARGET_NUMERIC_KEYS:
            if key in target:
                _coerce_float(target[key], field=f"target.{key}")

    params_raw = raw.get("parameters")
    if params_raw is None:
        raise ScenarioParseError("Scenario must define 'parameters'")
    params = _parse_params(params_raw)

    # Anything outside the canonical keys is preserved as extra so backends can
    # pick up scenario-specific knobs without forking this parser.
    known_keys = {"name", "metric", "target", "parameters"}
    extra = {k: v for k, v in raw.items() if k not in known_keys}

    return Scenario(
        name=name,
        params=params,
        target=target,
        metric=metric,
        extra=extra,
    )


def load_scenario(source: Path | str | dict[str, Any]) -> Scenario:
    """Load a scenario from a YAML file path or a pre-parsed dict.

    Args:
        source: A :class:`pathlib.Path` to a YAML file, or a pre-parsed dict.
            ``str`` is also accepted for ergonomics — it is always interpreted
            as a filesystem path (callers wanting to pass YAML text should
            ``yaml.safe_load`` it first or hand a dict directly).

    Returns:
        Parsed :class:`Scenario`.

    Raises:
        ScenarioParseError: on any validation failure.
        FileNotFoundError: when ``source`` is a path that does not exist.
    """
    if isinstance(source, dict):
        return parse_scenario(source)

    if isinstance(source, Path | str):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ScenarioParseError(f"Failed to read scenario file {path}: {e}") from e
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ScenarioParseError(
                f"Invalid YAML in scenario file {path}: {e}"
            ) from e
        if data is None:
            raise ScenarioParseError(f"Scenario file is empty: {path}")
        return parse_scenario(data)

    raise ScenarioParseError(
        f"Scenario source must be a path, str, or dict; got {type(source).__name__}"
    )
