# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLM-as-judge for ``physics-agent tune`` (single-shot, hybrid programmatic+VLM).

This module is the "judge" component of the tune-refine loop introduced in
issue #51 Part 1.1. It produces a :class:`JudgeResult` from a parsed
:class:`Scenario`, a list of :class:`TrialRecord` and a best-params dict, by
combining:

* a **programmatic** sub-score (60% weight) that does fast,
  deterministic plausibility checks (param ranges, failure rate, finite
  best-score), and
* a **VLM** sub-score (40% weight) that asks a vision-language model to
  evaluate whether the optimisation result is good enough, returning
  **strict JSON**. The media list may be empty for text-only judging.

The combined score (``0.6 * programmatic + 0.4 * vlm``) is compared against
``score_threshold`` (default 0.7) to derive the ``approve`` / ``continue``
decision driving the caller's refine loop. **This module is single-shot and
stateless**; the runner is responsible for incrementing ``iterations`` and
honouring the hard cap (default 3) on refine cycles.

Design constraints (enforced by tests, do not relax):

1. **No optimizer imports.** This module must not import anything from
   ``physics_agent.tuning.optimizers`` or transitively pull in
   ``botorch`` / ``torch`` / ``ovphysx``. A subprocess test installs an
   import blocker for those names and expects this module to load
   successfully. Allowed imports: ``physics_agent.tuning.types`` and
   ``physics_agent.tuning.visual_evidence``.
2. **Strict JSON VLM contract.** The VLM is asked to emit
   ``{"score": float, "decision": "approve"|"continue", "reasoning": str}``.
   The model's ``decision`` is informative only — the authoritative decision
   is computed from the combined score against ``score_threshold``.
3. **VLM failures degrade gracefully.** Any network/parse/key error sets
   ``llm_unavailable=True``, fills ``llm_score = programmatic_score`` (so
   the weighted combine is mathematically the programmatic score), and
   returns a normal :class:`JudgeResult`. We only raise :class:`JudgeError`
   for the rare case where no usable result can be produced at all.
4. **No caching.** Each ``run_tune_judge`` invocation is fresh.
   Determinism for re-runs comes from the daemon's seed contract
   plus the model provider's own behaviour, not a persisted
   judge-output cache.

Judge model default: the physics-agent VLM default. The caller is expected
to construct ``vlm_model`` and pass it in; we do not build one here so we
avoid pulling provider SDKs into the import graph.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from physics_agent.tuning.types import Scenario, TrialRecord, TunableParam
from physics_agent.tuning.visual_evidence import (
    DEFAULT_JUDGE_GENERATED_FRAMES,
    DEFAULT_JUDGE_REFERENCE_FRAMES,
    JudgeVisualEvidence,
    sample_visual_evidence_items,
    validate_visual_frame_count,
)

__all__ = ["JudgeResult", "JudgeError", "run_tune_judge"]

_logger = logging.getLogger(__name__)

# Programmatic sub-component weights (must sum to 1.0). Changing them
# changes the score contract — historical ``tune_results.json`` files
# remain readable, but their numeric scores are no longer comparable
# to fresh runs.
_W_PARAM_PLAUSIBILITY = 0.60
_W_FAILED_PENALTY = 0.30
_W_FINITE_BEST = 0.10

# Combined-score weights.
_W_PROGRAMMATIC = 0.60
_W_LLM = 0.40

# Reasoning summary cap (chars). Mirrors the spec's "≤ ~500 chars".
_REASONING_MAX = 500
_BACKEND_HARD_FAILURE_RE = re.compile(r"(?:^|[;,\s])[^;=]+=\s*fail\b", re.I)
_BACKEND_COMPONENT_RE = re.compile(
    r"^\s*(upright|settled|finite_position|ground_clearance)\s*=\s*(pass|fail)\b",
    re.I,
)
_CURRENT_BACKEND_COMPONENT_WEIGHTS = {
    "settled": 0.3,
    "finite_position": 0.3,
    "ground_clearance": 0.6,
}


def _drop_legacy_upright_component(score: float, critique: str) -> tuple[float, str]:
    """Remove the retired upright scorer from replayed backend metrics."""
    components: list[tuple[str, bool, str]] = []
    other_parts: list[str] = []
    found_upright = False
    for raw_part in critique.split(";"):
        part = raw_part.strip()
        match = _BACKEND_COMPONENT_RE.match(part)
        if match is None:
            if part:
                other_parts.append(part)
            continue
        name = match.group(1).lower()
        passed = match.group(2).lower() == "pass"
        if name == "upright":
            found_upright = True
            continue
        components.append((name, passed, part))

    if not found_upright:
        return score, critique

    kept_parts = [part for _, _, part in components] + other_parts
    cleaned_critique = "; ".join(kept_parts)
    if not components:
        return 1.0, cleaned_critique

    total_weight = sum(
        _CURRENT_BACKEND_COMPONENT_WEIGHTS[name] for name, _, _ in components
    )
    earned = sum(
        _CURRENT_BACKEND_COMPONENT_WEIGHTS[name]
        for name, passed, _ in components
        if passed
    )
    return earned / total_weight, cleaned_critique


def _has_backend_programmatic_hard_failure(critique: str) -> bool:
    """Return true when backend trajectory checks marked any component failed."""
    return bool(_BACKEND_HARD_FAILURE_RE.search(critique))


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeResult:
    """The outcome of a single judge invocation.

    The ``iterations`` field is set by the caller across refine cycles; this
    module always emits it as ``iteration`` (the input arg, default 1).
    """

    decision: Literal["approve", "continue"]
    score: float  # combined, 0..1
    programmatic_score: float  # 0..1
    llm_score: float  # 0..1, equals programmatic_score when LLM unavailable
    reasoning: str  # human-readable summary, ≤ ~500 chars
    iterations: int = 1
    llm_unavailable: bool = False
    programmatic_critique: str = ""
    llm_critique: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (e.g. for ``tune_results.json``)."""
        return asdict(self)


class JudgeError(RuntimeError):
    """Raised when the judge cannot produce any usable result.

    Most LLM failures degrade to programmatic-only with
    ``llm_unavailable=True``; this exception is reserved for the rare case
    where even the programmatic path raises (e.g. malformed inputs that
    survived dataclass validation).
    """


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_tune_judge(
    scenario: Scenario,
    history: list[TrialRecord],
    best_params: dict[str, float],
    *,
    user_prompt: str | None = None,
    chat_model: Any | None = None,
    vlm_model: Any | None = None,
    visual_evidence: JudgeVisualEvidence | None = None,
    visual_evidence_enabled: bool = True,
    judge_max_tokens: int | None = None,
    judge_temperature: float | None = None,
    judge_reference_frames: int = DEFAULT_JUDGE_REFERENCE_FRAMES,
    judge_generated_frames: int = DEFAULT_JUDGE_GENERATED_FRAMES,
    score_threshold: float = 0.7,
    iteration: int = 1,
    prior_refine_history: list[dict[str, Any]] | None = None,
) -> JudgeResult:
    """Score a tune run and decide whether to approve or continue refining.

    Args:
        scenario: Parsed scenario YAML (name, metric, target, tunable params).
        history: Trial records in evaluation order. May be empty.
        best_params: Best-found params (keys must be a subset of
            ``scenario.params`` names).
        user_prompt: Optional NL context — passed verbatim to the LLM.
        chat_model: Deprecated compatibility parameter. The judge no longer
            calls chat-model APIs; pass ``vlm_model`` instead.
        vlm_model: Optional VLM object with ``generate_with_image_caption_pairs``.
            Used for both media-backed and text-only judge calls. When no
            media is available the judge calls it with an empty media list.
        visual_evidence: Optional reference/generated image evidence for a
            VLM judge. Reference media and generated frames are included
            when present; otherwise the VLM receives text-only prompt data.
        visual_evidence_enabled: Whether the caller allowed visual evidence
            to be supplied to the VLM judge. When false, callers should pass
            ``visual_evidence=None`` and the metadata records the disabled state.
        judge_max_tokens: Optional max output tokens for the judge response.
            ``None`` uses ``scenario.extra["judge"]["max_tokens"]``
            if present, else the physics judge default.
        judge_temperature: Optional temperature for the judge call.
            ``None`` uses ``scenario.extra["judge"]["temperature"]`` if
            present, else the physics judge default.
        judge_reference_frames: Max reference images/video frames to send to
            the VLM judge. Reference media is sampled evenly when there are
            more items than this limit.
        judge_generated_frames: Max generated render frames to send to the VLM
            judge. Rendered videos are sampled evenly when there are more
            frames than this limit.
        score_threshold: Combined-score cut-off for ``approve``. Default 0.7.
        iteration: Caller-tracked refine-loop iteration number (1-indexed).
        prior_refine_history: Optional compact summaries of previous refine
            iterations. Used only as prompt context for the VLM judge.

    Returns:
        :class:`JudgeResult` with the decision, sub-scores, and critique
        text.

    Raises:
        JudgeError: Only when the programmatic path itself fails.
    """
    validated_judge_reference_frames = validate_visual_frame_count(
        "judge_reference_frames",
        judge_reference_frames,
    )
    validated_judge_generated_frames = validate_visual_frame_count(
        "judge_generated_frames",
        judge_generated_frames,
    )

    # ---------------------------------------------------------- programmatic
    try:
        prog_score, prog_critique = _score_programmatic(scenario, history, best_params)
    except Exception as exc:  # pragma: no cover - guarded by dataclass init
        raise JudgeError(f"programmatic judge failed: {exc}") from exc

    # ------------------------------------------------------------------- vlm
    if vlm_model is None and hasattr(chat_model, "generate_with_image_caption_pairs"):
        # Backward-compatible escape hatch for older programmatic callers that
        # passed a VLM-like object in the former chat_model slot.
        vlm_model = chat_model
    effective_visual_evidence = visual_evidence if visual_evidence_enabled else None
    llm_score, llm_critique, llm_unavailable = _score_vlm(
        scenario=scenario,
        history=history,
        best_params=best_params,
        user_prompt=user_prompt,
        vlm_model=vlm_model,
        visual_evidence=effective_visual_evidence,
        programmatic_score=prog_score,
        judge_max_tokens=judge_max_tokens,
        judge_temperature=judge_temperature,
        judge_reference_frames=validated_judge_reference_frames,
        judge_generated_frames=validated_judge_generated_frames,
        prior_refine_history=prior_refine_history,
    )

    # -------------------------------------------------- combine + decide
    combined = _W_PROGRAMMATIC * prog_score + _W_LLM * llm_score
    # Clamp into [0, 1] just in case sub-scores escaped their ranges.
    combined = max(0.0, min(1.0, combined))
    backend_hard_failure = _has_backend_programmatic_hard_failure(prog_critique)
    decision: Literal["approve", "continue"]
    # Round-12 follow-up (CI flake fix for
    # ``test_threshold_one_only_approves_perfect``): the programmatic
    # sub-score is itself a weighted sum (0.60 + 0.30 + 0.10) that
    # arithmetic-evaluates to ``0.9999999999999999`` rather than exactly
    # ``1.0`` due to standard IEEE-754 imprecision; the same FP bias
    # propagates into ``combined``. Compare with a tiny epsilon so a
    # perfect-on-paper run with ``score_threshold == 1.0`` does NOT fall
    # to ``"continue"`` because of FP. The epsilon is far below any
    # judge-meaningful score difference (the LLM emits two-decimal
    # values).
    _SCORE_EPS = 1e-9
    if backend_hard_failure:
        combined = min(combined, max(0.0, score_threshold - 1e-6))
        decision = "continue"
    else:
        decision = "approve" if combined + _SCORE_EPS >= score_threshold else "continue"

    reasoning = _summarise_reasoning(
        decision=decision,
        combined=combined,
        prog_score=prog_score,
        llm_score=llm_score,
        prog_critique=prog_critique,
        llm_critique=llm_critique,
        llm_unavailable=llm_unavailable,
    )

    return JudgeResult(
        decision=decision,
        score=round(combined, 6),
        programmatic_score=round(prog_score, 6),
        llm_score=round(llm_score, 6),
        reasoning=reasoning,
        iterations=iteration,
        llm_unavailable=llm_unavailable,
        programmatic_critique=prog_critique,
        llm_critique=llm_critique,
        extra={
            "judge_modality": "vlm",
            "visual_evidence_enabled": bool(visual_evidence_enabled),
            "reference_image_count": (
                len(effective_visual_evidence.reference_image_caption_pairs)
                if effective_visual_evidence is not None
                else 0
            ),
            "generated_image_count": (
                len(effective_visual_evidence.generated_image_paths)
                if effective_visual_evidence is not None
                else 0
            ),
            "judge_reference_frames": validated_judge_reference_frames,
            "judge_generated_frames": validated_judge_generated_frames,
            "visual_evidence": (
                effective_visual_evidence.to_metadata()
                if effective_visual_evidence is not None
                else None
            ),
        },
    )


# ---------------------------------------------------------------------------
# Programmatic score
# ---------------------------------------------------------------------------


def _score_programmatic(
    scenario: Scenario,
    history: list[TrialRecord],
    best_params: dict[str, float],
) -> tuple[float, str]:
    """Plausibility + run-health checks. Returns (score in [0,1], critique).

    Sub-component weights (score-contract stable):

    * **60%** — per-tunable param plausibility: 1.0 if ``best_params[name]``
      is strictly inside ``[min, max]``, else 0.0. Averaged across params.
    * **30%** — failed-trial penalty:
      ``1 - clamp(failed/max(len(history),1), 0, 1)``.
    * **10%** — finite best-score: 1.0 if the best history score is finite,
      else 0.0. With an empty history we default this to 1.0 (nothing to
      contradict).

    When a backend trial reports its own deterministic ``programmatic_score``
    (freeform does this for trajectory checks such as ground clearance), the
    outer judge clamps the run-health score to that backend score. That prevents
    the VLM/refine layer from approving a trial whose physics evaluator already
    proved a hard objective failed.
    """
    notes: list[str] = []

    # 1) Per-param plausibility ------------------------------------------------
    params: tuple[TunableParam, ...] = scenario.params
    if params:
        per_param: list[float] = []
        for p in params:
            if p.name not in best_params:
                per_param.append(0.0)
                notes.append(f"missing best_params[{p.name!r}]")
                continue
            try:
                v = float(best_params[p.name])
            except (TypeError, ValueError):
                per_param.append(0.0)
                notes.append(f"best_params[{p.name!r}] not numeric")
                continue
            if not math.isfinite(v):
                per_param.append(0.0)
                notes.append(f"best_params[{p.name!r}] not finite")
                continue
            if p.min_value <= v <= p.max_value:
                per_param.append(1.0)
            else:
                per_param.append(0.0)
                notes.append(
                    f"best_params[{p.name!r}]={v:g} outside "
                    f"[{p.min_value:g}, {p.max_value:g}]"
                )
        param_score = sum(per_param) / len(per_param)
    else:  # pragma: no cover - Scenario.__post_init__ rejects empty params
        param_score = 0.0
        notes.append("scenario has no tunable params")

    # 2) Failed-trial penalty --------------------------------------------------
    n = len(history)
    if n == 0:
        failed_penalty_score = 1.0
        notes.append("no trials in history")
    else:
        failed = sum(1 for t in history if getattr(t, "failed", False))
        ratio = max(0.0, min(1.0, failed / max(n, 1)))
        failed_penalty_score = 1.0 - ratio
        if failed:
            notes.append(f"{failed}/{n} trials failed")

    # 3) Finite best-score check ----------------------------------------------
    if n == 0:
        finite_score = 1.0
    else:
        # Lower is better in this codebase; pick min over non-failed trials,
        # or fall back to min over all trials.
        candidates = [t.score for t in history if not getattr(t, "failed", False)]
        if not candidates:
            candidates = [t.score for t in history]
        try:
            best = min(candidates)
        except ValueError:  # pragma: no cover
            best = float("nan")
        if isinstance(best, int | float) and math.isfinite(best):
            finite_score = 1.0
        else:
            finite_score = 0.0
            notes.append("best score is non-finite")

    score = (
        _W_PARAM_PLAUSIBILITY * param_score
        + _W_FAILED_PENALTY * failed_penalty_score
        + _W_FINITE_BEST * finite_score
    )
    score = max(0.0, min(1.0, score))

    backend_programmatic_score: float | None = None
    backend_programmatic_critique = ""
    winning = _winning_trial(history)
    if winning is not None:
        backend_metrics = winning.backend_metrics or {}
        raw_backend_score = backend_metrics.get("programmatic_score")
        if isinstance(raw_backend_score, int | float) and not isinstance(
            raw_backend_score, bool
        ):
            raw_backend_score = float(raw_backend_score)
            if math.isfinite(raw_backend_score):
                backend_programmatic_score = max(0.0, min(1.0, raw_backend_score))
                backend_programmatic_critique = str(
                    backend_metrics.get("programmatic_critique") or ""
                )
                (
                    backend_programmatic_score,
                    backend_programmatic_critique,
                ) = _drop_legacy_upright_component(
                    backend_programmatic_score,
                    backend_programmatic_critique,
                )
                score = min(score, backend_programmatic_score)

    critique_bits: list[str] = []
    if param_score < 1.0:
        critique_bits.append(f"param_plausibility={param_score:.2f}")
    if failed_penalty_score < 1.0:
        critique_bits.append(f"failed_penalty={failed_penalty_score:.2f}")
    if finite_score < 1.0:
        critique_bits.append(f"finite_best={finite_score:.2f}")
    if backend_programmatic_score is not None and backend_programmatic_score < 1.0:
        bit = f"backend_programmatic={backend_programmatic_score:.2f}"
        if backend_programmatic_critique:
            bit += f" ({backend_programmatic_critique})"
        critique_bits.append(bit)
    if not critique_bits:
        critique = "all programmatic checks pass"
    else:
        critique = "; ".join(critique_bits)
        if notes:
            critique += " (" + "; ".join(notes) + ")"

    return score, critique


# ---------------------------------------------------------------------------
# VLM score
# ---------------------------------------------------------------------------


_VLM_SYSTEM_PROMPT = (
    "You are an expert physics-simulation judge. Evaluate whether a parameter "
    "optimisation run produced a good result for the given scenario. Use any "
    "supplied reference media, generated simulation frames, physics metrics, "
    "and user goal. If no media is supplied, judge from the text and metrics "
    "only.\n\n"
    "You must respond with strict JSON ONLY (no markdown, no preamble) of "
    "the form:\n"
    '{"score": <float in [0,1]>, "decision": "approve" | "continue", '
    '"reasoning": "<= 500 chars"}\n\n'
    "Score 1.0 means the run clearly succeeded; 0.0 means it clearly failed. "
    'Use "approve" when you believe further refinement is unlikely to help, '
    'and "continue" when another refine iteration would likely improve the '
    "result. Treat visual media as behavioral guidance, not pixel-perfect "
    "ground truth."
)


def _score_vlm(
    *,
    scenario: Scenario,
    history: list[TrialRecord],
    best_params: dict[str, float],
    user_prompt: str | None,
    vlm_model: Any | None,
    visual_evidence: JudgeVisualEvidence | None,
    programmatic_score: float,
    judge_max_tokens: int | None,
    judge_temperature: float | None,
    judge_reference_frames: int,
    judge_generated_frames: int,
    prior_refine_history: list[dict[str, Any]] | None = None,
) -> tuple[float, str, bool]:
    """Ask a VLM to score the run.

    The media list is empty only when no visual evidence is constructed.
    Reference media and/or generated simulation frames are included when
    callers provide them.
    """
    if visual_evidence is not None and visual_evidence.reference_error:
        return (
            programmatic_score,
            f"VLM unavailable: reference media failed: {visual_evidence.reference_error}",
            True,
        )
    if visual_evidence is not None and visual_evidence.generated_error:
        return (
            programmatic_score,
            f"VLM unavailable: generated evidence failed: {visual_evidence.generated_error}",
            True,
        )
    if (
        visual_evidence is not None
        and visual_evidence.has_reference_media
        and not visual_evidence.generated_image_paths
    ):
        return (
            programmatic_score,
            "VLM unavailable: no generated render frames supplied",
            True,
        )
    if vlm_model is None:
        return (
            programmatic_score,
            "VLM unavailable: no vlm_model supplied",
            True,
        )
    try:
        max_tokens = _resolve_judge_max_tokens(
            scenario=scenario,
            judge_max_tokens=judge_max_tokens,
        )
        temperature = _resolve_judge_temperature(
            scenario=scenario,
            judge_temperature=judge_temperature,
        )
    except (TypeError, ValueError) as exc:
        return (
            programmatic_score,
            f"VLM unavailable: invalid judge config: {exc}",
            True,
        )

    image_caption_pairs: list[tuple[str, Any]] = []
    if visual_evidence is not None:
        try:
            reference_pairs, generated_items = sample_visual_evidence_items(
                visual_evidence,
                max_reference_images=judge_reference_frames,
                max_generated_images=judge_generated_frames,
            )
        except ValueError as exc:
            return (
                programmatic_score,
                f"VLM unavailable: invalid visual evidence config: {exc}",
                True,
            )
        image_caption_pairs.extend(reference_pairs)
        image_caption_pairs.extend(generated_items)
        dropped_reference = len(visual_evidence.reference_image_caption_pairs) - len(
            reference_pairs
        )
        dropped_generated = len(visual_evidence.generated_image_paths) - len(
            generated_items
        )
        if dropped_reference or dropped_generated:
            _logger.info(
                "Sampled visual judge media: kept %d/%d reference and %d/%d "
                "generated frame(s)",
                len(reference_pairs),
                len(visual_evidence.reference_image_caption_pairs),
                len(generated_items),
                len(visual_evidence.generated_image_paths),
            )

    try:
        prompt = _build_visual_prompt(
            scenario=scenario,
            history=history,
            best_params=best_params,
            user_prompt=user_prompt,
            prior_refine_history=prior_refine_history,
        )
    except Exception as exc:
        return programmatic_score, f"VLM unavailable: prompt build failed: {exc}", True

    try:
        response = vlm_model.generate_with_image_caption_pairs(
            image_caption_pairs=image_caption_pairs,
            final_prompt=prompt,
            system_prompt=_VLM_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        _logger.warning(
            "judge VLM invoke raised (provider detail logged server-side only): %s",
            exc,
        )
        return (
            programmatic_score,
            f"VLM unavailable: invoke raised {type(exc).__name__} (see server logs).",
            True,
        )

    if not isinstance(response, str) or not response.strip():
        return programmatic_score, "VLM unavailable: empty response", True

    parsed = _parse_strict_json(response)
    if parsed is None:
        return (
            programmatic_score,
            "VLM unavailable: could not parse JSON from response",
            True,
        )

    try:
        raw_score = parsed["score"]
        raw_reasoning = parsed.get("reasoning", "")
    except KeyError as exc:
        return (
            programmatic_score,
            f"VLM unavailable: missing key in JSON: {exc}",
            True,
        )

    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return (
            programmatic_score,
            f"VLM unavailable: non-numeric score {raw_score!r}",
            True,
        )

    if not math.isfinite(score):
        return programmatic_score, "VLM unavailable: non-finite score", True
    score = max(0.0, min(1.0, score))

    critique = str(raw_reasoning).strip()
    if len(critique) > _REASONING_MAX:
        critique = critique[: _REASONING_MAX - 3] + "..."
    if not critique:
        critique = "(no VLM reasoning provided)"
    return score, critique, False


def _scenario_judge_config(scenario: Scenario) -> dict[str, Any]:
    """Return the optional ``judge:`` block preserved from scenario YAML."""
    raw_config = scenario.extra.get("judge", {})
    if raw_config is None:
        return {}
    if not isinstance(raw_config, dict):
        raise ValueError(
            f"scenario judge config must be a mapping, got {type(raw_config).__name__}"
        )
    return raw_config


def _resolve_judge_max_tokens(
    *,
    scenario: Scenario,
    judge_max_tokens: int | None,
) -> int:
    """Resolve the judge response budget.

    Mirrors material-agent's split between the base VLM max-token budget and
    the judge-specific response budget. The import is intentionally lazy so
    text-only judge imports stay narrow.
    """
    if judge_max_tokens is None:
        scenario_config = _scenario_judge_config(scenario)
        scenario_max_tokens = scenario_config.get("max_tokens")
        if scenario_max_tokens is not None:
            judge_max_tokens = scenario_max_tokens
    if judge_max_tokens is None:
        from physics_agent.api.defaults import DEFAULT_JUDGE_MAX_TOKENS

        judge_max_tokens = DEFAULT_JUDGE_MAX_TOKENS
    value = int(judge_max_tokens)
    if value < 1:
        raise ValueError(f"must be >= 1, got {value}")
    return value


def _resolve_judge_temperature(
    *,
    scenario: Scenario,
    judge_temperature: float | None,
) -> float:
    """Resolve the judge sampling temperature from explicit, YAML, default."""
    if judge_temperature is None:
        scenario_config = _scenario_judge_config(scenario)
        scenario_temperature = scenario_config.get("temperature")
        if scenario_temperature is not None:
            judge_temperature = scenario_temperature
    if judge_temperature is None:
        from physics_agent.api.defaults import DEFAULT_JUDGE_TEMPERATURE

        judge_temperature = DEFAULT_JUDGE_TEMPERATURE
    value = float(judge_temperature)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"temperature must be finite and >= 0, got {value}")
    return value


# Whitelist of scalar fields the judge surfaces from each trial's
# ``backend_metrics`` into the judge prompt. Anything not on this list
# (in particular: in-memory trajectory objects, file paths, error
# strings) is dropped. Keep the list small — the prompt budget is tight
# and the LLM only needs the physics signals it can reason over.
_BACKEND_METRIC_WHITELIST: tuple[str, ...] = (
    "settle_distance",
    "max_bounce_height",
    "first_bounce_height",
    "final_position",
    "rest_position",
    "world_up",
    "drop_height_m",
    "bbox_size_m",
    "programmatic_score",
    "programmatic_critique",
    "metric",
)


def _select_metrics(backend_metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Pull JSON-safe scalars out of a TrialRecord's ``backend_metrics``.

    Skips anything not in the whitelist. Coerces NaN/inf to strings via
    :func:`_jsonable` so the prompt always serialises cleanly.
    """
    if not backend_metrics:
        return {}
    selected: dict[str, Any] = {}
    for key in _BACKEND_METRIC_WHITELIST:
        if key in backend_metrics:
            selected[key] = _jsonable(backend_metrics[key])
    return selected


def _winning_trial(history: list[TrialRecord]) -> TrialRecord | None:
    """Lowest-score non-failed trial, or None when every trial failed.

    Matches the metric convention (lower is better — drop_settle's
    ``settle_distance`` and the negated ``max_bounce_height`` both
    minimise).
    """
    successful = [t for t in history if not getattr(t, "failed", False)]
    if not successful:
        return None
    return min(successful, key=lambda t: t.score)


def _load_trajectory_jsonl(
    path: str,
) -> list[tuple[float, list[float], list[float]]]:
    """Read a ``trajectory.jsonl`` written by ``author_trajectory_jsonl``.

    Returns the daemon-shaped ``[(t, pose7, vel6), ...]`` list that
    ``trajectory_summary`` accepts. Skips malformed lines silently —
    judge cost on the rare bad line is just a missing summary, not a
    crash.
    """
    out: list[tuple[float, list[float], list[float]]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                t = float(obj["t"])
                pose = [float(x) for x in obj["pose"]]
                vel = [float(x) for x in obj["vel"]]
            except (KeyError, TypeError, ValueError):
                continue
            out.append((t, pose, vel))
    return out


def _best_trial_summary(best: TrialRecord) -> dict[str, Any] | None:
    """Behavioral summary of the winning trial for the judge prompt.

    Pulls the ``trajectory_jsonl`` path off ``best.backend_metrics``,
    loads it, and runs
    :func:`world_understanding.functions.physics.trajectory.trajectory_summary`
    to produce a JSON-safe dict (``n_samples``, ``duration_s``,
    ``final_position``, ``max_linear_speed``, ``max_angular_speed``,
    ``settle_time_s``, ``fell_over``). Returns ``None`` when the path
    is missing or any step fails — the judge's combined score still
    runs, the LLM just loses one input.

    Lazily imports ``trajectory_summary`` to keep the module's top-level
    import surface unchanged (the design constraint at the top of this
    file forbids transitive optimizer / pxr / numpy imports at module
    load time).
    """
    bm = best.backend_metrics or {}
    path = bm.get("trajectory_jsonl")
    if not path:
        return None
    try:
        from world_understanding.functions.physics.trajectory import (
            infer_world_up,
            trajectory_summary,
        )
    except ImportError as exc:
        _logger.warning("trajectory_summary unavailable: %s", exc)
        return None
    # Identify the stage up-axis. Prefer the authoritative ``world_up``
    # the scene builder stashed in backend_metrics — that's the actual
    # axis the scene was authored against, not a guess. Fall back to
    # ``infer_world_up(rest_position)`` for backend_metrics shapes that
    # predate the world_up field. The legacy Y-up default lives inside
    # ``trajectory_summary`` itself when neither is available.
    #
    # Why both paths exist: corner-origin assets (e.g. SimReady ladder)
    # have ``rest_position == [0, 0, 0]`` because the body's bbox-min
    # already sits at the stage origin. ``infer_world_up`` then can't
    # extract an axis from the position and falls back to Y-up — wrong
    # on a Z-up stage. The explicit world_up from the scene builder
    # cuts through that ambiguity.
    explicit_world_up = bm.get("world_up")
    world_up: tuple[float, float, float] | None
    try:
        if (
            explicit_world_up
            and isinstance(explicit_world_up, list | tuple)
            and len(explicit_world_up) >= 3
            and any(float(v) != 0.0 for v in explicit_world_up[:3])
        ):
            world_up = (
                float(explicit_world_up[0]),
                float(explicit_world_up[1]),
                float(explicit_world_up[2]),
            )
        else:
            rest = bm.get("rest_position")
            world_up = infer_world_up(rest) if rest else None
    except (TypeError, ValueError):
        # Non-numeric ``world_up`` payload — fall through to the
        # rest_position inference (or legacy default) rather than
        # crashing the judge prompt build.
        rest = bm.get("rest_position")
        world_up = infer_world_up(rest) if rest else None
    try:
        traj = _load_trajectory_jsonl(str(path))
        if not traj:
            return None
        return _jsonable(trajectory_summary(traj, world_up=world_up))
    except (OSError, FileNotFoundError) as exc:
        _logger.warning("trajectory.jsonl read failed (%s): %s", path, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - prompt enrichment is best-effort
        _logger.warning("best_trial_summary build failed: %s", exc)
        return None


def _build_llm_prompt(
    *,
    scenario: Scenario,
    history: list[TrialRecord],
    best_params: dict[str, float],
    user_prompt: str | None,
    prior_refine_history: list[dict[str, Any]] | None = None,
) -> str:
    """Build a compact prompt — do NOT dump the entire history.jsonl."""
    history_summary = [
        {
            "trial_index": t.trial_index,
            "score": _coerce_jsonable_number(t.score),
            "failed": bool(getattr(t, "failed", False)),
            "metrics": _select_metrics(getattr(t, "backend_metrics", None)),
        }
        for t in history
    ]
    bounds = {p.name: [p.min_value, p.max_value] for p in scenario.params}
    payload: dict[str, Any] = {
        "scenario": {
            "name": scenario.name,
            "metric": scenario.metric,
            "target": _jsonable(scenario.target),
            "param_bounds": bounds,
        },
        "best_params": {k: _coerce_jsonable_number(v) for k, v in best_params.items()},
        "history_summary": history_summary,
        "history_length": len(history),
    }
    if prior_refine_history is not None:
        payload["prior_refine_history"] = list(prior_refine_history)
    if user_prompt:
        payload["user_prompt"] = user_prompt

    # Behavioral evidence on the winning trial — lifted from
    # ``trajectory.jsonl`` (the recorder's judge-readable companion to
    # ``recording.usd``). Mirrors how material-agent's judge reads
    # ``predictions.jsonl``: the prompt now carries what the body
    # actually did, not just optimizer scoreboard numbers. Best-effort —
    # when the path is missing or unreadable the key is omitted.
    best = _winning_trial(history)
    if best is not None:
        summary = _best_trial_summary(best)
        if summary is not None:
            payload["best_trial_summary"] = summary

    body = json.dumps(payload, sort_keys=True, indent=2)
    return (
        "Evaluate this physics-tuning run and respond with strict JSON as "
        "described in the system prompt.\n\n"
        f"{body}"
    )


def _build_visual_prompt(
    *,
    scenario: Scenario,
    history: list[TrialRecord],
    best_params: dict[str, float],
    user_prompt: str | None,
    prior_refine_history: list[dict[str, Any]] | None = None,
) -> str:
    """Build the text half of the VLM prompt; images are supplied separately."""
    base = _build_llm_prompt(
        scenario=scenario,
        history=history,
        best_params=best_params,
        user_prompt=user_prompt,
        prior_refine_history=prior_refine_history,
    )
    return (
        "Use any supplied images in order: reference media first, then "
        "Generated Physics Output frames from the best simulation trial. If "
        "no images are supplied, judge the run from the text goal, metrics, "
        "trial history, and best parameters. Respond with strict JSON only.\n\n"
        f"{base}"
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_strict_json(text: str) -> dict[str, Any] | None:
    """Defensively parse a JSON object from possibly-noisy LLM output."""
    s = text.strip()
    # Strip ``` fences if present.
    if s.startswith("```"):
        s = s.strip("`")
        # Drop a leading ``json`` token if present.
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fall back to extracting the first {...} block.
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_jsonable_number(v: Any) -> Any:
    """Coerce numbers to JSON-safe values (NaN/inf become strings)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "Infinity" if f > 0 else "-Infinity"
    return f


def _jsonable(obj: Any) -> Any:
    """Recursively coerce ``obj`` to a JSON-safe structure."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, int | bool) or obj is None:
        return obj
    if isinstance(obj, float):
        return _coerce_jsonable_number(obj)
    if isinstance(obj, str):
        return obj
    return str(obj)


def _summarise_reasoning(
    *,
    decision: str,
    combined: float,
    prog_score: float,
    llm_score: float,
    prog_critique: str,
    llm_critique: str,
    llm_unavailable: bool,
) -> str:
    parts: list[str] = [
        f"{decision} (combined={combined:.2f}, "
        f"prog={prog_score:.2f}, llm={llm_score:.2f}"
        + (", llm_unavailable" if llm_unavailable else "")
        + ")"
    ]
    if prog_critique:
        parts.append(f"prog: {prog_critique}")
    if llm_critique and not llm_unavailable:
        parts.append(f"llm: {llm_critique}")
    text = " | ".join(parts)
    if len(text) > _REASONING_MAX:
        text = text[: _REASONING_MAX - 3] + "..."
    return text
