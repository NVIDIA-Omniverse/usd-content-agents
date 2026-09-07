# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLM judge for one external-runtime refinement iteration."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from world_understanding.optimization.contracts import combine_judge_scores

from physics_agent.api.defaults import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_TEMPERATURE,
)
from physics_agent.tasks.judge_tune import (
    JUDGE_PROGRAMMATIC_WEIGHT,
    JUDGE_VLM_WEIGHT,
    JudgeResult,
)
from physics_agent.tuning.visual_evidence import (
    DEFAULT_JUDGE_GENERATED_FRAMES,
    DEFAULT_JUDGE_REFERENCE_FRAMES,
    JudgeVisualEvidence,
    sample_visual_evidence_items,
    validate_visual_frame_count,
)

from .artifacts import summarize_external_trials
from .types import ExternalTuneOutput, ExternalTuneSpec

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_SYSTEM_PROMPT = """You are an expert judge for customer-provided robotics and physics simulations.
Evaluate the selected simulation result against the user's goal using the generated
simulation frames, optional reference media, the customer's fixed scalar objective,
and optimization history. Do not reward visual polish that conflicts with the
physical goal. Respond with strict JSON only:
{"score": <number from 0 to 1>, "decision": "approve" or "continue", "reasoning": "<=500 characters"}
Use approve only when the selected behavior satisfies the goal and another refine
iteration is unlikely to provide a meaningful improvement.
"""


def _parse_response(value: Any) -> tuple[float, str, str] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    raw_score = payload.get("score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
        return None
    score = float(raw_score)
    if not math.isfinite(score):
        return None
    decision = payload.get("decision")
    if decision not in {"approve", "continue"}:
        return None
    reasoning = str(payload.get("reasoning") or "").strip()
    if not reasoning:
        return None
    return max(0.0, min(1.0, score)), decision, reasoning[:500]


def _programmatic_score(output: ExternalTuneOutput) -> tuple[float, str, bool]:
    if not output.success or not output.history:
        return 0.0, "optimization did not produce a usable history", True
    failed = sum(1 for record in output.history if record.failed)
    run_health = 1.0 - failed / len(output.history)
    finite_best = float(
        math.isfinite(output.best_score)
        and output.best_objective is not None
        and math.isfinite(output.best_objective)
    )
    hard_failure = False
    notes: list[str] = []
    recording = output.artifacts.get("best_recording")
    if recording is None or not Path(recording).is_file():
        hard_failure = True
        notes.append("selected candidate recording is unavailable")
    if failed:
        notes.append(f"{failed}/{len(output.history)} trials failed")
    if not finite_best:
        hard_failure = True
        notes.append("selected objective is non-finite")
    score = 0.7 * run_health + 0.3 * finite_best
    return (
        max(0.0, min(1.0, score)),
        "; ".join(notes) or "run-health checks pass",
        hard_failure,
    )


def run_external_judge(
    *,
    spec: ExternalTuneSpec,
    output: ExternalTuneOutput,
    user_prompt: str,
    vlm_model: Any | None,
    visual_evidence: JudgeVisualEvidence,
    score_threshold: float,
    iteration: int,
    prior_refine_history: list[dict[str, Any]] | None = None,
    judge_max_tokens: int | None = None,
    judge_temperature: float | None = None,
    judge_reference_frames: int = DEFAULT_JUDGE_REFERENCE_FRAMES,
    judge_generated_frames: int = DEFAULT_JUDGE_GENERATED_FRAMES,
) -> JudgeResult:
    """Judge one selected BYOR result and fail closed without usable VLM evidence."""

    judge_reference_frames = validate_visual_frame_count(
        "judge_reference_frames", judge_reference_frames
    )
    judge_generated_frames = validate_visual_frame_count(
        "judge_generated_frames", judge_generated_frames
    )
    programmatic_score, programmatic_critique, hard_failure = _programmatic_score(
        output
    )
    unavailable_reason: str | None = None
    if visual_evidence.reference_error:
        unavailable_reason = (
            f"reference evidence failed: {visual_evidence.reference_error}"
        )
    elif visual_evidence.generated_error:
        unavailable_reason = (
            f"generated evidence failed: {visual_evidence.generated_error}"
        )
    elif not visual_evidence.generated_image_paths:
        unavailable_reason = "selected simulation produced no judge frames"
    elif vlm_model is None:
        unavailable_reason = "no VLM model supplied"

    reference_items: list[tuple[str, Any]] = []
    generated_items: list[tuple[str, Any]] = []
    if unavailable_reason is None:
        try:
            reference_items, generated_items = sample_visual_evidence_items(
                visual_evidence,
                max_reference_images=judge_reference_frames,
                max_generated_images=judge_generated_frames,
            )
        except ValueError as exc:
            unavailable_reason = str(exc)

    llm_score = 0.0
    llm_decision = "continue"
    llm_critique = ""
    if unavailable_reason is None:
        assert vlm_model is not None
        payload = {
            "iteration": iteration,
            "task": spec.task,
            "user_goal": user_prompt,
            "objective": {
                "name": spec.objective.name,
                "unit": spec.objective.unit,
                "direction": spec.objective.direction,
            },
            "active_search": {
                parameter.name: [parameter.min_value, parameter.max_value]
                for parameter in spec.params
            },
            "selected": {
                "params": output.best_params,
                "objective_value": output.best_objective,
                "optimizer_loss": output.best_score,
            },
            "top_trials": summarize_external_trials(output.history),
            "prior_iterations": list(prior_refine_history or []),
        }
        image_caption_pairs = [*reference_items, *generated_items]
        try:
            response = vlm_model.generate_with_image_caption_pairs(
                image_caption_pairs=image_caption_pairs,
                final_prompt=(
                    "Judge this external-runtime refinement iteration.\n\n"
                    + json.dumps(payload, indent=2, sort_keys=True)
                ),
                system_prompt=_SYSTEM_PROMPT,
                temperature=(
                    DEFAULT_JUDGE_TEMPERATURE
                    if judge_temperature is None
                    else float(judge_temperature)
                ),
                max_tokens=(
                    DEFAULT_JUDGE_MAX_TOKENS
                    if judge_max_tokens is None
                    else int(judge_max_tokens)
                ),
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary
            unavailable_reason = f"VLM invoke failed: {type(exc).__name__}"
        else:
            parsed = _parse_response(response)
            if parsed is None:
                unavailable_reason = "VLM response was not valid judge JSON"
            else:
                llm_score, llm_decision, llm_critique = parsed

    llm_unavailable = unavailable_reason is not None
    if llm_unavailable:
        llm_critique = f"VLM unavailable: {unavailable_reason}"
    combined = combine_judge_scores(
        programmatic_score,
        llm_score,
        programmatic_weight=JUDGE_PROGRAMMATIC_WEIGHT,
        vlm_weight=JUDGE_VLM_WEIGHT,
    )
    decision = (
        "approve"
        if not llm_unavailable
        and not hard_failure
        and llm_decision == "approve"
        and combined + 1.0e-9 >= score_threshold
        else "continue"
    )
    reasoning = (
        f"{decision}: score={combined:.3f}; programmatic={programmatic_critique}; "
        f"VLM={llm_critique}"
    )[:500]
    return JudgeResult(
        decision=decision,
        score=round(combined, 6),
        programmatic_score=round(programmatic_score, 6),
        llm_score=round(llm_score, 6),
        reasoning=reasoning,
        iterations=iteration,
        llm_unavailable=llm_unavailable,
        programmatic_critique=programmatic_critique,
        llm_critique=llm_critique,
        extra={
            "judge_modality": "external_runtime_vlm",
            "reference_image_count": len(reference_items),
            "generated_image_count": len(generated_items),
            "visual_evidence": visual_evidence.to_metadata(),
        },
    )


__all__ = ["run_external_judge"]
