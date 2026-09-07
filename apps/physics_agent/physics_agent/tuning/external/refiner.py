# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constrained LLM refiner for an external runtime's parameter search."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from world_understanding.functions.nlp.chat import generate_chat_response

from physics_agent.tasks.judge_tune import JudgeResult
from physics_agent.tuning.types import TunableParam

from .types import ExternalTuneSpec

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = """You refine a trusted customer simulation optimization.
The customer runtime, simulation setup, parameter catalog, and scalar objective are
fixed and must not be changed. You may select a non-empty subset of qualified
parameters and change their search bounds. Do not invent names. Bounds must be
finite with min < max; integer parameters require integer bounds.

Respond with strict JSON only:
{
  "search": {"parameters": {"name": {"min": 0.0, "max": 1.0}}},
  "reasoning": "<=500 characters"
}
"""


class ExternalRefineError(RuntimeError):
    """The refiner could not produce a valid constrained update."""


@dataclass(frozen=True)
class ExternalRefineDecision:
    """Validated parameter search for the next iteration."""

    params: tuple[TunableParam, ...]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "search": {
                "parameters": {
                    parameter.name: {
                        "min": parameter.min_value,
                        "max": parameter.max_value,
                    }
                    for parameter in self.params
                }
            },
            "reasoning": self.reasoning,
        }


def _json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        raise ExternalRefineError("refiner returned an empty response")
    text = value.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        if match is None:
            raise ExternalRefineError("refiner response is not JSON") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ExternalRefineError("refiner response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ExternalRefineError("refiner response must be an object")
    if set(payload) != {"search", "reasoning"}:
        raise ExternalRefineError(
            "refiner response fields must be search and reasoning"
        )
    return payload


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExternalRefineError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ExternalRefineError(f"{label} must be a finite number")
    return result


def _parse_params(payload: Any, spec: ExternalTuneSpec) -> tuple[TunableParam, ...]:
    if not isinstance(payload, dict) or set(payload) != {"parameters"}:
        raise ExternalRefineError("search must contain only parameters")
    raw_params = payload.get("parameters")
    if not isinstance(raw_params, dict) or not raw_params:
        raise ExternalRefineError("search.parameters must be a non-empty object")
    catalog = {parameter.name: parameter for parameter in spec.parameter_catalog}
    unknown = sorted(set(raw_params) - set(catalog))
    if unknown:
        raise ExternalRefineError(f"refiner used unqualified parameter(s): {unknown}")
    params: list[TunableParam] = []
    for name, raw in raw_params.items():
        if not isinstance(raw, dict) or set(raw) != {"min", "max"}:
            raise ExternalRefineError(
                f"search parameter {name!r} must contain min and max"
            )
        minimum = _finite(raw.get("min"), f"search.parameters.{name}.min")
        maximum = _finite(raw.get("max"), f"search.parameters.{name}.max")
        try:
            params.append(
                TunableParam(
                    name=name,
                    min_value=minimum,
                    max_value=maximum,
                    integer=catalog[name].integer,
                )
            )
        except ValueError as exc:
            raise ExternalRefineError(str(exc)) from exc
        if minimum >= maximum:
            raise ExternalRefineError(
                f"search parameter {name!r} must satisfy min < max"
            )
    return tuple(params)


def run_external_refiner(
    *,
    spec: ExternalTuneSpec,
    judge_result: JudgeResult,
    user_prompt: str,
    chat_model: Any | None,
    iteration: int,
    history_summary: list[dict[str, Any]],
    prior_refine_history: list[dict[str, Any]] | None = None,
) -> ExternalRefineDecision:
    """Request and validate one constrained external refine update."""

    if chat_model is None:
        raise ExternalRefineError("external refiner has no chat model")
    payload = {
        "iteration": iteration,
        "user_goal": user_prompt,
        "parameter_catalog": {
            parameter.name: {
                "type": "integer" if parameter.integer else "number",
                "nominal": spec.qualification.nominal_params[parameter.name],
            }
            for parameter in spec.parameter_catalog
        },
        "objective": {
            "name": spec.objective.name,
            "unit": spec.objective.unit,
            "direction": spec.objective.direction,
        },
        "current_search": {
            parameter.name: {
                "min": parameter.min_value,
                "max": parameter.max_value,
            }
            for parameter in spec.params
        },
        "judge": judge_result.to_dict(),
        "top_trials": history_summary[:8],
        "prior_iterations": list(prior_refine_history or []),
    }
    try:
        response = generate_chat_response(
            chat_model,
            "Refine the parameter search for the next iteration.\n\n"
            + json.dumps(payload, indent=2, sort_keys=True),
            system_prompt=_SYSTEM_PROMPT,
        )
    except Exception as exc:
        raise ExternalRefineError(
            f"external refiner invocation failed: {type(exc).__name__}"
        ) from exc
    if not isinstance(response, dict) or response.get("error"):
        raise ExternalRefineError("external refiner provider returned an error")
    parsed = _json_object(response.get("response"))
    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ExternalRefineError("refiner reasoning must be a non-empty string")
    return ExternalRefineDecision(
        params=_parse_params(parsed["search"], spec),
        reasoning=reasoning.strip()[:500],
    )


__all__ = [
    "ExternalRefineDecision",
    "ExternalRefineError",
    "run_external_refiner",
]
