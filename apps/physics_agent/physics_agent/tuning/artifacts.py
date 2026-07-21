# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact writers for tuning runs.

The runner emits five canonical artifacts into ``output_dir``:

* ``best_params.json`` — flat JSON with the winning parameter values.
* ``tune_results.json`` — full reproducible run record (config + best + history
  summary). All fields are JSON-stable so machine consumers can parse it.
* ``history.jsonl`` — one JSON line per trial, append-flushed so an SSE
  consumer / live tail can watch progress.
* ``report.md`` — human-readable summary.
* ``tuned_physics.usd`` — the tuned USD (written by :mod:`usd_patch`).

When visual reference media is supplied, the runner may also emit
``comparison.png`` as the judge contact sheet.

Naming is part of the public contract — see Acceptance Criteria.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import IO, Any

from .types import Scenario, TrialRecord, TuneInput

logger = logging.getLogger(__name__)


def _safe_fence(content: str) -> str:
    """Return a fence string longer than any backtick run in ``content``.

    Markdown fenced blocks are closed by a fence at least as long as the
    opener with the same character. To prevent malicious or accidental
    fence-escaping inside untrusted content (e.g. an attacker-supplied
    ``user_prompt`` containing ``` `` ` `` ``` to forge a fake "## Judge
    verdict" section in report.md), pick a fence longer than any
    consecutive backtick run already in the content. Minimum fence length
    is 3.
    """
    longest = 0
    run = 0
    for ch in content:
        if ch == "`":
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    return "`" * max(3, longest + 1)


def _json_safe(value: Any) -> Any:
    """Recursively replace Infinity/NaN floats with JSON-strict surrogates.

    Python's :func:`json.dumps` emits ``Infinity``/``NaN`` literals by default,
    which strict JSON consumers (browsers' ``JSON.parse``, ``jq``, etc.)
    reject. We map them to ``None`` so SSE consumers tailing
    ``history.jsonl`` can parse every line.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically.

    Writes to a sibling temp file then ``os.replace`` so a kill mid-write
    cannot leave half-written JSON for downstream consumers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


ARTIFACT_BEST_PARAMS = "best_params.json"
ARTIFACT_RESULTS = "tune_results.json"
ARTIFACT_HISTORY = "history.jsonl"
ARTIFACT_REPORT = "report.md"
ARTIFACT_TUNED_USD = "tuned_physics.usd"
ARTIFACT_VISUAL_COMPARISON = "comparison.png"


def ensure_output_dir(path: Path) -> Path:
    """Create ``path`` if necessary and return its absolute form."""
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def open_history_writer(output_dir: Path) -> IO[str]:
    """Open ``history.jsonl`` for line-buffered append.

    The runner writes one trial per line and ``flush()``-es after each write
    so an SSE consumer can tail the file safely.
    """
    p = ensure_output_dir(output_dir) / ARTIFACT_HISTORY
    # Truncate at run start. Each run owns its history file.
    return p.open("w", encoding="utf-8", buffering=1)  # buffering=1 = line-buf


def write_history_line(handle: IO[str], record: TrialRecord) -> None:
    """Append one trial record + flush.

    Uses ``allow_nan=False`` after first replacing NaN/Inf with ``None`` so
    every line is strict JSON parseable by browsers / jq.
    """
    payload = _json_safe(record.to_dict())
    handle.write(json.dumps(payload, sort_keys=True, allow_nan=False))
    handle.write("\n")
    handle.flush()


def write_best_params(
    output_dir: Path, best_params: dict[str, float], best_score: float
) -> Path:
    """Write the canonical ``best_params.json`` artifact.

    Atomic temp-file + replace so a process kill mid-write never leaves a
    half-written JSON for downstream consumers.
    """
    p = ensure_output_dir(output_dir) / ARTIFACT_BEST_PARAMS
    payload = _json_safe(
        {
            "best_score": float(best_score),
            "params": {k: float(v) for k, v in best_params.items()},
        }
    )
    _atomic_write_text(
        p, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    )
    return p


def write_tune_results(
    output_dir: Path,
    *,
    params_input: TuneInput,
    scenario: Scenario,
    optimizer_used: str,
    engine_used: str,
    best_params: dict[str, float],
    best_score: float,
    history: list[TrialRecord],
    cancelled: bool,
    started_at: str,
    completed_at: str,
    judge_result: dict[str, Any] | None = None,
) -> Path:
    """Write the canonical ``tune_results.json`` artifact.

    Includes ``scenario.extra`` so any backend-specific knobs the loader
    preserved are part of the reproducible run record. Uses
    ``physics_usd_basename`` alongside the absolute path so consumers can
    audit which file was tuned without having to re-resolve absolute paths
    that don't exist on a different machine.

    When ``params_input.user_prompt`` is set, it is persisted under the
    top-level ``user_prompt`` key per #51 spec (and rendered into the
    matching ``report.md``). When ``judge_result`` is supplied, it is
    persisted under ``judge`` for downstream consumers; absent when judging
    is disabled to preserve byte-identical output vs the pre-Part-1.1
    baseline.
    """
    p = ensure_output_dir(output_dir) / ARTIFACT_RESULTS
    physics_usd_path = Path(str(params_input.physics_usd))
    payload: dict[str, Any] = {
        "scenario": {
            "name": scenario.name,
            "metric": scenario.metric,
            "target": scenario.target,
            "parameters": [
                {
                    "name": tp.name,
                    "min": tp.min_value,
                    "max": tp.max_value,
                }
                for tp in scenario.params
            ],
            "extra": scenario.extra,
        },
        "config": {
            "engine": engine_used,
            "optimizer": optimizer_used,
            "max_trials": params_input.max_trials,
            "seed": params_input.seed,
            "physics_usd": str(physics_usd_path),
            "physics_usd_basename": physics_usd_path.name,
        },
        "started_at": started_at,
        "completed_at": completed_at,
        "cancelled": cancelled,
        "n_trials": len(history),
        "best": {
            "score": float(best_score),
            "params": {k: float(v) for k, v in best_params.items()},
        },
        "history_summary": [
            {
                "trial_index": t.trial_index,
                "score": t.score,
                "params": t.params,
                "failed": t.failed,
            }
            for t in history
        ],
    }
    # user_prompt is only emitted when set so existing artifacts (and tests
    # that compare them) stay byte-identical for the explicit-YAML path.
    if params_input.user_prompt:
        payload["user_prompt"] = params_input.user_prompt
    if judge_result is not None:
        payload["judge"] = judge_result
    _atomic_write_text(
        p,
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
    )
    return p


def write_report_md(
    output_dir: Path,
    *,
    scenario: Scenario,
    optimizer_used: str,
    engine_used: str,
    best_params: dict[str, float],
    best_score: float,
    history: list[TrialRecord],
    cancelled: bool,
    user_prompt: str | None = None,
    judge_result: dict[str, Any] | None = None,
) -> Path:
    """Write the human-readable ``report.md`` artifact.

    ``user_prompt`` and ``judge_result`` are optional for byte-identical
    backward compat: when both are absent, the report body is identical
    to the pre-Part-1.1 baseline.
    """
    p = ensure_output_dir(output_dir) / ARTIFACT_REPORT
    lines: list[str] = []
    lines.append(f"# Physics Agent tune report — `{scenario.name}`")
    lines.append("")
    if cancelled:
        lines.append("> Run was cancelled before completion.")
        lines.append("")
    lines.append(f"- Engine: `{engine_used}`")
    lines.append(f"- Optimizer: `{optimizer_used}`")
    lines.append(f"- Metric: `{scenario.metric}`")
    lines.append(f"- Trials: `{len(history)}`")
    lines.append("")
    if user_prompt:
        # Only emit when set so the explicit-YAML path stays byte-identical
        # to the PR #43 baseline. Fence width adapts to the longest run of
        # backticks already in user_prompt so a caller cannot forge
        # downstream sections (e.g. a fake "## Judge verdict") by closing
        # the fence with their own ```.
        fence = _safe_fence(user_prompt)
        lines.append("## User prompt")
        lines.append("")
        lines.append(fence)
        lines.append(user_prompt)
        lines.append(fence)
        lines.append("")
    lines.append("## Best parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("| --- | --- |")
    for k in sorted(best_params):
        lines.append(f"| `{k}` | `{best_params[k]:.6g}` |")
    lines.append("")
    lines.append(f"Best score: `{best_score:.6g}` (lower is better)")
    lines.append("")
    # Only render the Judge verdict section when there is an actual
    # verdict to show. Failed/timed-out judge attempts persist a
    # status=failed marker in tune_results.json for machine consumers
    # but skip the human-readable section so the report stays clean.
    if judge_result is not None and "decision" in judge_result:
        lines.append("## Judge verdict")
        lines.append("")
        decision = judge_result.get("decision", "?")
        score = judge_result.get("score")
        score_str = f"{score:.3f}" if isinstance(score, int | float) else "?"
        iterations = judge_result.get("iterations", 1)
        lines.append(f"- Decision: `{decision}`")
        lines.append(f"- Score: `{score_str}` (≥ 0.7 → approve)")
        lines.append(f"- Iterations: `{iterations}`")
        # ``judge_result`` here is guaranteed to have a decision (gated
        # above), so reading the reasoning is safe.
        reasoning = judge_result.get("reasoning")
        if reasoning:
            reasoning_text = str(reasoning)
            # Same fence-width hardening as the user_prompt block — judge
            # reasoning is LLM-controlled and reaches a downloadable
            # audit artifact, so embedded backticks must not be able to
            # forge sibling sections.
            fence = _safe_fence(reasoning_text)
            lines.append("")
            lines.append("**Reasoning:**")
            lines.append("")
            lines.append(fence)
            lines.append(reasoning_text)
            lines.append(fence)
        extra = judge_result.get("extra")
        evidence = extra.get("visual_evidence") if isinstance(extra, dict) else None
        if isinstance(evidence, dict):
            _append_visual_evidence_md(lines, evidence)
        lines.append("")
    lines.append("## Trial history")
    lines.append("")
    lines.append("| # | Score | Params |")
    lines.append("| --- | --- | --- |")
    for t in history:
        params_str = ", ".join(f"{k}={t.params[k]:.4g}" for k in sorted(t.params))
        score_str = "FAILED" if t.failed else f"{t.score:.6g}"
        lines.append(f"| {t.trial_index} | {score_str} | {params_str} |")
    lines.append("")
    _atomic_write_text(p, "\n".join(lines))
    return p


def _append_visual_evidence_md(lines: list[str], evidence: dict[str, Any]) -> None:
    reference_items = evidence.get("reference_images")
    generated_items = evidence.get("generated_images")
    comparison_image = evidence.get("comparison_image")
    comparison_error = evidence.get("comparison_error")
    reference_error = evidence.get("reference_error")
    generated_error = evidence.get("generated_error")

    has_evidence = (
        comparison_image
        or reference_items
        or generated_items
        or reference_error
        or generated_error
        or comparison_error
    )
    if not has_evidence:
        return

    lines.append("")
    lines.append("**Evidence:**")
    if comparison_image:
        lines.append(f"- Comparison image: {_inline_code(comparison_image)}")
    if isinstance(reference_items, list) and reference_items:
        lines.append(f"- Reference media: `{len(reference_items)}` item(s)")
        for item in reference_items:
            if isinstance(item, dict):
                _append_evidence_item(lines, item)
    if isinstance(generated_items, list) and generated_items:
        lines.append(f"- Generated frames: `{len(generated_items)}` item(s)")
        for item in generated_items:
            if isinstance(item, dict):
                _append_evidence_item(lines, item)
    if reference_error:
        lines.append(f"- Reference media note: {_inline_code(reference_error)}")
    if generated_error:
        lines.append(f"- Generated frame note: {_inline_code(generated_error)}")
    if comparison_error:
        lines.append(f"- Comparison image note: {_inline_code(comparison_error)}")


def _append_evidence_item(lines: list[str], item: dict[str, Any]) -> None:
    path = item.get("path")
    caption = item.get("caption")
    if path and caption:
        lines.append(f"  - {_inline_code(path)} - {_inline_code(caption)}")
    elif path:
        lines.append(f"  - {_inline_code(path)}")


def _inline_code(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    longest = 0
    run = 0
    for ch in text:
        if ch == "`":
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    fence = "`" * max(1, longest + 1)
    if longest:
        return f"{fence} {text} {fence}"
    return f"{fence}{text}{fence}"


__all__ = [
    "ARTIFACT_BEST_PARAMS",
    "ARTIFACT_HISTORY",
    "ARTIFACT_REPORT",
    "ARTIFACT_RESULTS",
    "ARTIFACT_TUNED_USD",
    "ARTIFACT_VISUAL_COMPARISON",
    "ensure_output_dir",
    "open_history_writer",
    "write_history_line",
    "write_best_params",
    "write_tune_results",
    "write_report_md",
]
