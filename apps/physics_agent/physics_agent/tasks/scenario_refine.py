# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-driven scenario refinement task.

One LLM call per refine iteration. Inputs are intentionally
goal-neutral — the prompt receives:

* the user's free-form goal (``user_goal_text``),
* the current scenario as YAML text,
* the last :class:`JudgeResult` (programmatic + LLM critiques and
  combined score),
* a top-k summary of trial results from ``history.jsonl``,
* the current iteration number (1-indexed),

and the LLM decides what to widen, tighten, or swap. The output is a
refined scenario YAML string that ``parse_scenario`` accepts. We do
**not** hardcode "bouncy" or any other adjective into the template:
the loop must work for any goal the user types.

Design constraints (mirror ``judge_tune.py`` so behaviour is
predictable):

1. **Strict-JSON LLM contract** — the model emits a single JSON object
   ``{"scenario": {...}}`` and we re-serialise to YAML. Strict JSON is
   easier to validate and recover from than free-form YAML.
2. **Graceful degradation** — when the LLM is unavailable or returns
   an unparseable / invalid scenario, we fall back to the input
   scenario YAML so the refine loop can still complete its remaining
   iterations (the judge will keep returning ``continue`` but the
   outer loop will exit at ``max_iterations``).
3. **No optimizer imports** — same constraint as ``judge_tune``; the
   refine task only depends on ``physics_agent.tuning.scenario`` and
   ``world_understanding.functions.nlp.chat``.
4. **No mutation of input** — the input YAML text is preserved.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import yaml
from world_understanding.functions.nlp.chat import generate_chat_response

from physics_agent.tasks.judge_tune import JudgeResult
from physics_agent.tuning.scenario import ScenarioParseError, parse_scenario
from physics_agent.tuning.types import SUPPORTED_PARAM_KEYS, Scenario

__all__ = [
    "RefineError",
    "RefineResult",
    "run_scenario_refine",
]

logger = logging.getLogger(__name__)

# Top-k trials surfaced into the prompt — keeps the prompt budget small
# while still showing the LLM how scores moved across the search.
_TOP_K_TRIALS = 5

# Cap the goal text the same way ``judge_tune`` caps reasoning so the
# prompt does not balloon when callers paste a paragraph-length prompt.
_GOAL_TEXT_MAX = 500


class RefineError(RuntimeError):
    """Internal sentinel raised inside ``_extract_json_object`` when the
    LLM payload cannot be parsed.

    ``run_scenario_refine`` catches this on its own and converts it into
    a degraded ``RefineResult(llm_unavailable=True, reasoning=...)``, so
    ``RefineError`` is never propagated to the orchestrator. Tests that
    want to exercise the JSON-extraction failure modes assert on the
    raised form via ``_extract_json_object`` directly.
    """


@dataclass(frozen=True)
class RefineResult:
    """Outcome of one ``run_scenario_refine`` invocation.

    Attributes:
        refined_yaml: A YAML string that parses to a valid
            :class:`Scenario`. When the LLM call fails, this is the
            unchanged input YAML.
        scenario: The parsed :class:`Scenario` matching ``refined_yaml``.
        llm_unavailable: True when we degraded to the input scenario
            because of an LLM/parse failure.
        reasoning: Short human-readable summary of what the LLM said
            it changed (or why we degraded).
        notes: Free-form metadata block for tests/audit.
    """

    refined_yaml: str
    scenario: Scenario
    llm_unavailable: bool = False
    reasoning: str = ""
    notes: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt template — goal-neutral
# ---------------------------------------------------------------------------


def _normalize_supported_param_keys(
    supported_param_keys: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """Validate and order an optional backend-specific parameter allowlist."""
    if supported_param_keys is None:
        return SUPPORTED_PARAM_KEYS
    raw = tuple(str(k) for k in supported_param_keys)
    invalid = sorted(set(raw) - set(SUPPORTED_PARAM_KEYS))
    if invalid:
        raise ValueError(
            "supported_param_keys contains unsupported parameter(s): "
            f"{invalid}. Supported globally: {sorted(SUPPORTED_PARAM_KEYS)}"
        )
    allowed = set(raw)
    ordered = tuple(k for k in SUPPORTED_PARAM_KEYS if k in allowed)
    if not ordered:
        raise ValueError("supported_param_keys must contain at least one key")
    return ordered


def _build_system_prompt(
    *,
    backend_name: str | None,
    supported_param_keys: tuple[str, ...],
) -> str:
    """Build a backend-scoped refiner prompt."""
    supported_params = ", ".join(supported_param_keys)
    backend_context = (
        f"\nActive backend: {backend_name}. Only choose tunable parameters from "
        "that backend's allowlist below.\n"
        if backend_name
        else ""
    )
    contact_guidance = (
        "\n``contact_ke`` and ``contact_kd`` are Newton MuJoCo contact "
        "stiffness/damping knobs. Use them for Newton bounce goals instead of "
        "restitution.\n"
        if {"contact_ke", "contact_kd"}.issubset(set(supported_param_keys))
        else ""
    )

    template = """You are a physics-tuning scenario refiner.

You receive (a) a free-form user goal, (b) the current tuning scenario as
YAML text, (c) the latest judge verdict including programmatic and LLM
critiques and a combined score in [0, 1], (d) the top trials from the
last optimization sweep, and (e) the current iteration number.
{backend_context}

Your job is to author a REFINED scenario that is more likely to satisfy
the user's goal on the next sweep. You may:

* widen or tighten parameter [min, max] bounds,
* tweak target keys (drop_height_m, duration_s, gravity, sample_fps,
  cameras, camera_ground_bias_fraction, vlm_check, record_video, ...),
  IMPORTANT: when refining, carry forward target keys you don't intend
  to change — emitting a target dict without ``camera_ground_bias_fraction``
  (or any other previously-set key) causes the next iteration to revert
  to that key's default behavior, silently regressing whatever the
  prior iteration relied on. Re-emit unchanged keys verbatim.
* swap the metric to a more goal-aligned one — but ONLY when the input
  scenario's ``name`` is ``"drop_settle"``, and only pick from the
  closed registry: "settle_distance" or "max_bounce_height". For
  ``"freeform"`` scenarios the metric field is decorative — the
  freeform backend evaluates a fixed combined trajectory+VLM score and
  ignores ``metric`` — so KEEP the metric you were given.
* add or remove tunable parameters from the supported set
  ({supported_params}).
{contact_guidance}

You MUST keep the scenario kind ("name") unchanged from the input. You
MUST emit a non-empty parameters list, with min<=max for every entry.

Respond with strict JSON ONLY (no markdown, no preamble, no trailing
prose). Top-level shape:

  {
    "scenario": {
      "name": "drop_settle" | "freeform",
      "metric": <string>,  // drop_settle: "settle_distance" | "max_bounce_height";
                           // freeform: keep input value verbatim
      "target": {...},
      "parameters": [{"name": "<string>", "min": <float>, "max": <float>}, ...]
    },
    "reasoning": "<= 500 char human-readable summary of your changes"
  }

Do not invent parameter names. Stick to {supported_params}. Do not
change the scenario "name".
"""
    return (
        template.replace("{backend_context}", backend_context)
        .replace("{supported_params}", supported_params)
        .replace("{contact_guidance}", contact_guidance)
    )


def _trim(text: str, max_len: int) -> str:
    text = text or ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _coerce_jsonable_number(v: Any) -> Any:
    """Coerce numbers to JSON-safe values (NaN / inf become strings)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "Infinity" if f > 0 else "-Infinity"
    return f


def _scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    """Round-trip a :class:`Scenario` into a YAML-friendly dict."""
    out: dict[str, Any] = {
        "name": scenario.name,
        "metric": scenario.metric,
        "target": dict(scenario.target),
        "parameters": [
            {"name": p.name, "min": p.min_value, "max": p.max_value}
            for p in scenario.params
        ],
    }
    if scenario.extra:
        for k, v in scenario.extra.items():
            out[k] = v
    return out


def _preserve_omitted_judge_extra(
    *,
    current_scenario: Scenario,
    refined_dict: dict[str, Any],
) -> None:
    """Carry forward the current judge block unless the LLM overrides it.

    The refine prompt asks the model to return the core scenario shape
    (``name``/``metric``/``target``/``parameters``). Operational top-level
    blocks such as ``judge`` still belong to the scenario contract, so losing
    the judge settings after iteration 1 would silently switch later
    iterations to global defaults. Explicit refined values win; omissions
    inherit the current scenario's judge block.
    """
    if "judge" in current_scenario.extra and "judge" not in refined_dict:
        refined_dict["judge"] = current_scenario.extra["judge"]


def _preserve_omitted_target_keys(
    *,
    current_scenario: Scenario,
    refined_dict: dict[str, Any],
) -> None:
    """Carry forward target keys the LLM omitted from its refined target.

    The LLM's refined ``target`` dict fully replaces the current one
    downstream — so any key the model didn't re-author is gone for the
    rest of the refine loop. That's surprising for keys like
    ``camera_ground_bias_fraction``, ``record_video``, ``sample_fps``,
    etc., where the user set a value the model wasn't asked to change.

    Preserve every current target key the refined dict does not
    explicitly specify. Explicit refined values (including explicit
    ``None``) win; only true omissions inherit. This keeps the refine
    loop deterministic in the face of new target keys that may post-
    date the system prompt's enumeration.
    """
    # Three target-shape cases to handle:
    #   (a) Missing key  → ``parse_scenario`` defaults to ``{}`` and
    #       silently drops every current target key — install a fresh
    #       dict so the loop below can re-inject them.
    #   (b) Explicit ``null`` → same as (a); ``parse_scenario`` coerces
    #       ``None`` to ``{}``.
    #   (c) Wrong shape (list, scalar, ...) → ``parse_scenario`` raises,
    #       so defer there rather than mutating into a runtime mess.
    if "target" not in refined_dict or refined_dict["target"] is None:
        refined_dict["target"] = {}
    refined_target = refined_dict["target"]
    if not isinstance(refined_target, dict):
        return
    for k, v in current_scenario.target.items():
        if k not in refined_target:
            refined_target[k] = v


def _summarise_history(history_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the top-k (lowest-score) trials for the prompt body."""
    if not history_summary:
        return []
    keep: list[dict[str, Any]] = []
    for trial in history_summary:
        try:
            score = float(trial.get("score", float("inf")))
        except (TypeError, ValueError):
            score = float("inf")
        keep.append({**trial, "_score_for_sort": score})
    keep.sort(key=lambda t: t["_score_for_sort"])
    top = keep[:_TOP_K_TRIALS]
    return [{k: v for k, v in t.items() if k != "_score_for_sort"} for t in top]


def _build_user_message(
    *,
    user_goal_text: str,
    current_yaml: str,
    judge_result: JudgeResult,
    history_summary: list[dict[str, Any]],
    iteration: int,
    backend_name: str | None,
    supported_param_keys: tuple[str, ...],
    prior_refine_history: list[dict[str, Any]] | None = None,
) -> str:
    """Build the user-side prompt body. JSON-shaped for easy LLM consumption."""
    payload: dict[str, Any] = {
        "iteration": int(iteration),
        "active_backend": backend_name,
        "allowed_tunable_parameters": list(supported_param_keys),
        "user_goal_text": _trim(user_goal_text, _GOAL_TEXT_MAX),
        "current_scenario_yaml": current_yaml,
        "judge_result": {
            "decision": judge_result.decision,
            "score": _coerce_jsonable_number(judge_result.score),
            "programmatic_score": _coerce_jsonable_number(
                judge_result.programmatic_score
            ),
            "llm_score": _coerce_jsonable_number(judge_result.llm_score),
            "programmatic_critique": judge_result.programmatic_critique,
            "llm_critique": judge_result.llm_critique,
            "reasoning": judge_result.reasoning,
            "llm_unavailable": bool(judge_result.llm_unavailable),
        },
        "top_trials": _summarise_history(history_summary),
    }
    if prior_refine_history is not None:
        payload["prior_refine_history"] = list(prior_refine_history)
    body = json.dumps(payload, sort_keys=True, indent=2)
    return (
        "Refine the scenario for the next iteration. Respond with strict "
        "JSON as described in the system prompt.\n\n" + body
    )


# ---------------------------------------------------------------------------
# JSON extraction (mirrors interpret_user_prompt_tuning._extract_json)
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict[str, Any]:
    """Defensively extract the first balanced JSON object from ``text``."""
    if not isinstance(text, str) or not text.strip():
        raise RefineError("LLM returned empty response")
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise RefineError(f"No JSON object found in response: {text[:200]!r}")
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise RefineError("Unbalanced JSON in LLM response")
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError as e:
        raise RefineError(f"Could not parse JSON: {e}") from e


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_scenario_refine(
    *,
    current_scenario: Scenario,
    judge_result: JudgeResult,
    user_goal_text: str,
    history_summary: list[dict[str, Any]] | None = None,
    iteration: int = 1,
    chat_model: Any | None = None,
    backend_name: str | None = None,
    supported_param_keys: tuple[str, ...] | list[str] | None = None,
    prior_refine_history: list[dict[str, Any]] | None = None,
) -> RefineResult:
    """Refine ``current_scenario`` for the next iteration of the refine loop.

    Args:
        current_scenario: The scenario the previous tune ran on.
        judge_result: The judge verdict for that previous tune.
        user_goal_text: Free-form user goal (e.g. ``"make it bouncy"``).
        history_summary: Top trials from the previous tune as a list of
            ``{trial_index, score, params, failed}`` dicts. Optional.
        iteration: 1-indexed iteration number for prompt context.
        chat_model: Pre-built chat model (LangChain ``BaseChatModel``).
            When ``None`` we degrade — return the input scenario unchanged
            with ``llm_unavailable=True``.
        backend_name: Optional active simulation backend name. Used only to
            scope the LLM prompt.
        supported_param_keys: Optional backend-specific tunable parameter
            allowlist. Refined scenarios using names outside this set are
            rejected and the loop reuses the current scenario.
        prior_refine_history: Optional compact summaries of prior outer-loop
            iterations. Included as prompt context for the refiner.

    Returns:
        A :class:`RefineResult`. On any LLM/parse/validation failure the
        result echoes the input scenario and sets ``llm_unavailable=True``.
    """
    current_dict = _scenario_to_dict(current_scenario)
    current_yaml = yaml.safe_dump(current_dict, sort_keys=False)
    history = list(history_summary or [])
    prior_history = None if prior_refine_history is None else list(prior_refine_history)
    prior_history_size = len(prior_history) if prior_history is not None else 0
    active_param_keys = _normalize_supported_param_keys(supported_param_keys)

    if chat_model is None:
        return RefineResult(
            refined_yaml=current_yaml,
            scenario=current_scenario,
            llm_unavailable=True,
            reasoning="LLM unavailable: no chat_model supplied",
            notes={
                "history_size": len(history),
                "prior_refine_history_size": prior_history_size,
            },
        )

    user_message = _build_user_message(
        user_goal_text=user_goal_text,
        current_yaml=current_yaml,
        judge_result=judge_result,
        history_summary=history,
        iteration=iteration,
        backend_name=backend_name,
        supported_param_keys=active_param_keys,
        prior_refine_history=prior_history,
    )
    system_prompt = _build_system_prompt(
        backend_name=backend_name,
        supported_param_keys=active_param_keys,
    )

    try:
        result = generate_chat_response(
            chat_model, user_message, system_prompt=system_prompt
        )
    except Exception as exc:
        logger.warning(
            "scenario_refine LLM invoke raised (provider detail logged "
            "server-side only): %s",
            exc,
        )
        return RefineResult(
            refined_yaml=current_yaml,
            scenario=current_scenario,
            llm_unavailable=True,
            reasoning=f"LLM unavailable: invoke raised ({type(exc).__name__})",
            notes={
                "history_size": len(history),
                "prior_refine_history_size": prior_history_size,
            },
        )

    if not isinstance(result, dict) or "error" in result:
        err = result.get("error") if isinstance(result, dict) else type(result).__name__
        logger.warning("scenario_refine LLM returned error: %s", err)
        return RefineResult(
            refined_yaml=current_yaml,
            scenario=current_scenario,
            llm_unavailable=True,
            reasoning="LLM unavailable: provider error",
            notes={
                "history_size": len(history),
                "prior_refine_history_size": prior_history_size,
            },
        )
    response = result.get("response")
    if not isinstance(response, str) or not response.strip():
        return RefineResult(
            refined_yaml=current_yaml,
            scenario=current_scenario,
            llm_unavailable=True,
            reasoning="LLM unavailable: empty response",
            notes={
                "history_size": len(history),
                "prior_refine_history_size": prior_history_size,
            },
        )

    try:
        parsed = _extract_json_object(response)
    except RefineError as exc:
        logger.warning("scenario_refine JSON parse failed: %s", exc)
        return RefineResult(
            refined_yaml=current_yaml,
            scenario=current_scenario,
            llm_unavailable=True,
            reasoning=f"LLM unavailable: {exc}",
            notes={
                "history_size": len(history),
                "prior_refine_history_size": prior_history_size,
            },
        )

    refined_dict = parsed.get("scenario")
    if not isinstance(refined_dict, dict):
        logger.warning("scenario_refine missing 'scenario' key in response")
        return RefineResult(
            refined_yaml=current_yaml,
            scenario=current_scenario,
            llm_unavailable=True,
            reasoning="LLM unavailable: response missing 'scenario' key",
            notes={
                "history_size": len(history),
                "prior_refine_history_size": prior_history_size,
            },
        )

    # Hard rule: refining must not change the scenario kind.
    if refined_dict.get("name") != current_scenario.name:
        logger.warning(
            "scenario_refine attempted to change scenario name from %r to %r; "
            "snapping back to current.",
            current_scenario.name,
            refined_dict.get("name"),
        )
        refined_dict["name"] = current_scenario.name

    # For freeform the backend's score is a fixed combined trajectory+VLM
    # weighting that ignores ``scenario.metric``. Letting the LLM
    # ostensibly swap the metric would only produce artifacts that
    # CLAIM to optimize a different objective while the actual scoring
    # never changed. Snap metric back to the input value so the artifact
    # name and the optimised objective stay aligned.
    if current_scenario.name == "freeform" and refined_dict.get("metric") != (
        current_scenario.metric
    ):
        logger.warning(
            "scenario_refine attempted to change freeform metric from %r to %r; "
            "freeform.evaluate ignores scenario.metric, snapping back to current.",
            current_scenario.metric,
            refined_dict.get("metric"),
        )
        refined_dict["metric"] = current_scenario.metric

    # Preserve the current scenario's metric when the LLM omits one. Without
    # this, ``parse_scenario`` would default a missing metric to
    # ``"settle_distance"`` for drop_settle, silently switching a run that
    # was optimising ``max_bounce_height`` back to the default objective —
    # a "successful" refinement that actually undoes the user's intent.
    # Only fires when the LLM returned no metric field at all; an
    # explicit value (including the same one) flows through unchanged.
    if "metric" not in refined_dict or refined_dict.get("metric") in (None, ""):
        refined_dict["metric"] = current_scenario.metric

    _preserve_omitted_judge_extra(
        current_scenario=current_scenario,
        refined_dict=refined_dict,
    )
    _preserve_omitted_target_keys(
        current_scenario=current_scenario,
        refined_dict=refined_dict,
    )

    try:
        refined_scenario = parse_scenario(refined_dict)
    except ScenarioParseError as exc:
        # ``parse_scenario`` already enforces the drop_settle metric
        # registry membership (see physics_agent.tuning.scenario), so an
        # LLM hallucination like ``"max_velocity"`` lands here with a
        # readable message and we degrade to the input scenario rather
        # than letting the next tune iteration fail every trial.
        #
        # CodeRabbit Round 11 thread #8 asked us to special-case
        # invalid-metric parse failures into a hard ``ValueError`` instead
        # of folding them into the catch-all degrade-to-input path. We
        # **deliberately keep the catch-all** here because the canonical
        # R5+R8 design (validated by ``test_scenario_refine.py
        # ::test_unsupported_drop_settle_metric_falls_back_to_current``)
        # is "any single LLM hiccup — including a typo'd metric — should
        # not end the entire refine loop; the next iteration's prompt
        # will see the refusal and try again." Surfacing only the
        # *fully-failed* refine (no scenario can be parsed at all) as a
        # hard error keeps that contract; aborting on a metric typo
        # would degrade UX across the whole loop. Operators who want
        # strict refusal on metric hallucinations should drive
        # ``IterativePhysicsRefinementTask`` directly and trip on
        # ``RefineResult.llm_unavailable``.
        logger.warning("scenario_refine produced invalid scenario: %s", exc)
        return RefineResult(
            refined_yaml=current_yaml,
            scenario=current_scenario,
            llm_unavailable=True,
            reasoning=f"LLM unavailable: refined scenario invalid ({exc})",
            notes={
                "history_size": len(history),
                "prior_refine_history_size": prior_history_size,
            },
        )

    invalid_params = sorted(
        {param.name for param in refined_scenario.params} - set(active_param_keys)
    )
    if invalid_params:
        logger.warning(
            "scenario_refine produced backend-disallowed parameter(s): %s",
            invalid_params,
        )
        return RefineResult(
            refined_yaml=current_yaml,
            scenario=current_scenario,
            llm_unavailable=True,
            reasoning=(
                "Refined scenario rejected: used backend-disallowed "
                f"parameter(s) ({', '.join(invalid_params)})"
            ),
            notes={
                "history_size": len(history),
                "prior_refine_history_size": prior_history_size,
                "supported_param_keys": list(active_param_keys),
            },
        )

    refined_yaml = yaml.safe_dump(_scenario_to_dict(refined_scenario), sort_keys=False)

    raw_reasoning = parsed.get("reasoning", "")
    if not isinstance(raw_reasoning, str):
        raw_reasoning = str(raw_reasoning)
    reasoning = _trim(raw_reasoning.strip() or "(no LLM reasoning provided)", 500)

    return RefineResult(
        refined_yaml=refined_yaml,
        scenario=refined_scenario,
        llm_unavailable=False,
        reasoning=reasoning,
        notes={
            "history_size": len(history),
            "prior_refine_history_size": prior_history_size,
            "supported_param_keys": list(active_param_keys),
        },
    )
