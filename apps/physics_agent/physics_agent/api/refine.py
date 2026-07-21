# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refine API for Physics Agent.

First-class programmatic surface for the iterative
``tune → judge → scenario_refine`` loop. Mirrors the material-agent shape
(``RefineInput`` / ``RefineOutput`` / ``run_refine`` / ``arun_refine``) so
cross-domain callers get a consistent contract.

This API delegates to
:class:`physics_agent.tasks.iterative_physics_refinement.\
IterativePhysicsRefinementTask` (the same orchestrator the
``physics-agent refine`` CLI drives). The CLI is now a thin shell that
collects flags, builds a chat model, and calls ``run_refine`` — the loop
logic itself lives behind this dataclass entry point.

Usage::

    from pathlib import Path
    from physics_agent.api import RefineInput, run_refine

    result = run_refine(
        RefineInput(
            scenario=Path("scenario.yaml"),
            physics_usd=Path("asset_physics.usda"),
            user_prompt="make it bouncy",
            output_dir=Path("output/refine"),
            max_iterations=3,
            score_threshold=0.7,
        )
    )
    if result.success:
        print(result.final_dir)

The runner-internal modules (``physics_agent.tuning.runner``,
``physics_agent.tasks.iterative_physics_refinement``) are imported lazily
so importing :mod:`physics_agent.api.refine` does **not** drag the full
``physics_agent.tuning`` package (and its optional botorch / ovphysx
dependencies) onto the import graph. The lazy-import contract is
mirrored at :mod:`physics_agent.api`'s top-level
``__getattr__`` so ``from physics_agent.api import run_refine`` is also
zero-cost until the first call.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from physics_agent.api.types import APIResult
from physics_agent.tuning.visual_evidence import (
    DEFAULT_JUDGE_GENERATED_FRAMES,
    DEFAULT_JUDGE_REFERENCE_FRAMES,
    DEFAULT_REFERENCE_VIDEO_FRAMES,
    validate_visual_frame_count,
)

logger = logging.getLogger(__name__)

_MAX_REFINE_TRIALS = 1000
_MAX_REFINE_ITERATIONS = 12
_REFINE_FAILURE_MESSAGE = "Physics refinement failed"
_REFINE_ITERATION_FAILURE_MESSAGE = "Physics refinement iteration failed"
_FINAL_RECORDING_FAILURE_MESSAGE = "Final recording could not be published"


@dataclass(kw_only=True)
class RefineInput:
    """Input parameters for the physics refine API.

    All fields are keyword-only to mirror :class:`TuneInput`. ``scenario``
    and ``user_prompt`` are both required — refine needs an initial
    scenario (the first iteration's bounds + target) and the NL prompt
    that drives the VLM judge + scenario refiner.
    """

    scenario: Path | dict[str, Any]
    """Initial scenario — Path to a YAML file or a pre-parsed dict.

    The first iteration's parameter bounds + target come from this
    file; subsequent iterations are LLM-refined.
    """

    physics_usd: Path
    """Path to the physics-authored USD to tune (output of ``apply_physics``)."""

    user_prompt: str
    """Free-form natural-language description of the desired tune outcome
    (e.g. ``"make it bouncy"``). Drives both the VLM judge and the
    scenario refiner."""

    output_dir: Path
    """Directory for per-iteration artifacts. Each iteration writes its
    own ``iter_N/`` subdirectory plus a ``final/`` snapshot."""

    reference_images: list[Path] | None = None
    """Optional reference images for the visual/VLM judge."""

    reference_videos: list[Path] | None = None
    """Optional reference videos for the visual/VLM judge."""

    reference_descriptions: list[str] | None = None
    """Optional descriptions parallel to ``reference_images``."""

    reference_video_descriptions: list[str] | None = None
    """Optional descriptions parallel to ``reference_videos``."""

    reference_video_frames: int = DEFAULT_REFERENCE_VIDEO_FRAMES
    """Number of frames to extract from each reference video for visual judging."""

    judge_reference_frames: int = DEFAULT_JUDGE_REFERENCE_FRAMES
    """Max reference images/video frames to send to the VLM judge."""

    judge_generated_frames: int = DEFAULT_JUDGE_GENERATED_FRAMES
    """Max generated render frames to send to the VLM judge."""

    engine: str = "ovphysx"
    """Simulation backend (passed through to ``run_tune`` each iteration)."""

    optimizer: str = "auto"
    """Optimizer name (``auto`` → BoTorch, ``random``, ``cma-es``)."""

    max_trials: int = 30
    """Trials per iteration."""

    seed: int = 42
    """Seed forwarded to the underlying tune step each iteration."""

    max_iterations: int = 5
    """Hard cap on (tune → judge → refine) iterations. The loop exits
    earlier when the judge returns ``approve``."""

    history_window: int = 20
    """Number of prior outer-loop iteration summaries to pass to the judge
    and scenario refiner. The best-so-far row is retained when it falls
    outside this recent window. Set ``0`` to disable prompt history; the final
    ``refine_summary.json`` still includes a compact audit row."""

    score_threshold: float = 0.7
    """Combined-score threshold above which the judge approves (loop
    terminates). Higher threshold = stricter approval requirement."""

    judge_max_tokens: int | None = None
    """Optional max output tokens for the judge response.
    ``None`` uses the physics-agent judge default."""

    judge_temperature: float | None = None
    """Optional temperature for judge calls.
    ``None`` uses the scenario YAML ``judge.temperature`` when present,
    otherwise the physics-agent judge default."""

    chat_model: Any | None = None
    """Optional pre-built chat model used by the scenario refiner.
    ``None`` causes scenario refine to re-use the previous scenario.
    The judge uses ``vlm_model`` instead, with an empty media list for
    text-only judge calls."""

    vlm_model: Any | None = None
    """Optional pre-built VLM instance used by the judge. When no media is
    supplied, the judge still calls this VLM with an empty media list."""

    force_record_video: str | None = "off"
    """When set, every iteration's ``scenario.target.record_video`` is
    overwritten to this value, overriding both the initial YAML and any
    LLM-refined value. The CLI default is ``"off"`` so per-trial render
    cost is avoided. Pass ``None`` to honor the YAML / refine flow instead."""

    render_winning_trial: bool = False
    """Post-tune render of the best trial's recording.usd into
    ``iter_N/render/`` as PNG judge evidence. Refine never encodes MP4;
    the time-sampled recording USD is its portable motion artifact."""

    visual_evidence_enabled: bool = True
    """Whether generated/reference media should be sent to the VLM judge.
    Disabling this keeps tune/render artifacts intact but forces the judge
    VLM call to use text-only evidence."""

    llm_timeout_seconds: float = 180.0
    """Wall-clock deadline (seconds) for each judge / refine LLM call.
    Mirrors the tune runner's safeguard so a hung NIM / ChatNVIDIA call
    cannot wedge the refine loop. Set ``0`` (or any non-positive value)
    to disable."""

    cancel_event: Any = None
    """Optional Event-like object with ``is_set()`` used for cooperative
    cancellation. The refine loop polls it between expensive phases and
    forwards it into each underlying tune iteration."""

    event_listener: Any = None
    """Optional :class:`world_understanding.agentic.events.EventListener`.
    The refine task emits per-iteration progress signals to this listener
    in addition to its own logger."""

    verbose: bool = False
    """Enable verbose progress logging."""

    def __post_init__(self) -> None:
        if not self.user_prompt or not str(self.user_prompt).strip():
            raise ValueError("user_prompt must be a non-empty string")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.max_iterations > _MAX_REFINE_ITERATIONS:
            raise ValueError(
                f"max_iterations must be <= {_MAX_REFINE_ITERATIONS}, "
                f"got {self.max_iterations}"
            )
        if self.history_window < 0:
            raise ValueError(f"history_window must be >= 0, got {self.history_window}")
        if self.max_trials < 1:
            raise ValueError(f"max_trials must be >= 1, got {self.max_trials}")
        if self.max_trials > _MAX_REFINE_TRIALS:
            raise ValueError(
                f"max_trials must be <= {_MAX_REFINE_TRIALS}, got {self.max_trials}"
            )
        try:
            score_threshold = float(self.score_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "score_threshold must be finite and between 0 and 1, "
                f"got {self.score_threshold}"
            ) from exc
        if not math.isfinite(score_threshold) or not (0.0 <= score_threshold <= 1.0):
            raise ValueError(
                "score_threshold must be finite and between 0 and 1, "
                f"got {self.score_threshold}"
            )
        self.score_threshold = score_threshold
        try:
            timeout_seconds = float(self.llm_timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"llm_timeout_seconds must be finite, got {self.llm_timeout_seconds}"
            ) from exc
        if not math.isfinite(timeout_seconds):
            raise ValueError(
                f"llm_timeout_seconds must be finite, got {self.llm_timeout_seconds}"
            )
        self.llm_timeout_seconds = timeout_seconds
        if self.judge_max_tokens is not None and self.judge_max_tokens < 1:
            raise ValueError(
                f"judge_max_tokens must be >= 1, got {self.judge_max_tokens}"
            )
        if self.judge_temperature is not None:
            temperature = float(self.judge_temperature)
            if not math.isfinite(temperature) or temperature < 0.0:
                raise ValueError(
                    "judge_temperature must be finite and >= 0, "
                    f"got {self.judge_temperature}"
                )
        self.reference_video_frames = validate_visual_frame_count(
            "reference_video_frames",
            self.reference_video_frames,
        )
        self.judge_reference_frames = validate_visual_frame_count(
            "judge_reference_frames",
            self.judge_reference_frames,
        )
        self.judge_generated_frames = validate_visual_frame_count(
            "judge_generated_frames",
            self.judge_generated_frames,
        )
        if self.cancel_event is not None and not callable(
            getattr(self.cancel_event, "is_set", None)
        ):
            raise TypeError(
                "cancel_event must be an Event-like object with an is_set() method"
            )
        # ``scenario`` may be a Path or dict; only normalize when Path-like.
        if isinstance(self.scenario, str):
            self.scenario = Path(self.scenario)
        if isinstance(self.scenario, Path):
            if not self.scenario.exists():
                raise FileNotFoundError(f"Scenario file not found: {self.scenario}")
        elif isinstance(self.scenario, dict):
            if not self.scenario:
                raise ValueError("Scenario dict cannot be empty")
        else:
            raise TypeError(
                f"scenario must be a Path or dict, got {type(self.scenario).__name__}"
            )
        self.physics_usd = Path(self.physics_usd)
        if not self.physics_usd.exists():
            raise FileNotFoundError(f"physics_usd file not found: {self.physics_usd}")
        if self.reference_images is not None:
            self.reference_images = [Path(p) for p in self.reference_images]
            for path in self.reference_images:
                if not path.exists():
                    raise FileNotFoundError(f"reference image not found: {path}")
                if not path.is_file():
                    raise ValueError(f"reference image must be a file: {path}")
        if self.reference_videos is not None:
            self.reference_videos = [Path(p) for p in self.reference_videos]
            for path in self.reference_videos:
                if not path.exists():
                    raise FileNotFoundError(f"reference video not found: {path}")
                if not path.is_file():
                    raise ValueError(f"reference video must be a file: {path}")
        if self.reference_descriptions is not None and len(
            self.reference_descriptions
        ) != len(self.reference_images or []):
            raise ValueError(
                "reference_descriptions must be supplied once per reference_images item"
            )
        if self.reference_video_descriptions is not None and len(
            self.reference_video_descriptions
        ) != len(self.reference_videos or []):
            raise ValueError(
                "reference_video_descriptions must be supplied once per "
                "reference_videos item"
            )
        self.output_dir = Path(self.output_dir)
        if self.force_record_video is not None and self.force_record_video not in {
            "off",
            "end_of_tune",
            "always",
        }:
            raise ValueError(
                "force_record_video must be one of {'off','end_of_tune','always'} "
                f"or None, got {self.force_record_video!r}"
            )


@dataclass
class IterationSummary:
    """One iteration's high-level summary — mirrors ``IterationRecord`` on
    disk minus the heavy ``best_params`` payload."""

    iteration: int
    iteration_dir: Path
    judge_decision: str  # "approve" | "continue" | "skipped"
    judge_score: float | None
    judge_reasoning: str
    best_score: float | None
    n_trials: int
    metric_name: str
    metric_value: float | None
    cancelled: bool = False
    error: str | None = None


@dataclass
class RefineOutput(APIResult):
    """Output from the physics refine API.

    The compact prior-history audit payload is persisted in
    ``output_dir/refine_summary.json`` as ``compact_history``; this public result
    keeps only the lighter per-iteration summaries.
    """

    output_dir: Path | None = None
    """Resolved root output directory (same as the input)."""

    iterations: list[IterationSummary] = field(default_factory=list)
    """One entry per iteration actually run, in execution order."""

    iteration_count: int = 0
    """Number of iterations actually executed (``len(iterations)``)."""

    final_iteration: int = 0
    """Index of the iteration whose artifacts populate ``final_dir``
    (1-based; 0 when no iteration completed)."""

    final_dir: Path | None = None
    """Snapshot directory containing the winning iteration's artifacts."""

    final_recording_usd: Path | None = None
    """Flattened time-sampled recording for the strict winning trial."""

    final_recording_error: str | None = None
    """Why the winning recording could not be published, when unavailable."""

    termination_reason: str = "unknown"
    """One of ``"approved"``, ``"max_iterations"``, ``"cancelled"``,
    ``"error"``, ``"unknown"``."""

    final_judge_score: float | None = None
    """Judge score of the final iteration (``None`` when no iteration
    produced a verdict)."""

    user_prompt: str = ""
    """Echo of the input user_prompt for audit."""


def _build_iteration_summary(record: Any) -> IterationSummary:
    """Map an ``IterationRecord`` (loop-internal) to ``IterationSummary``
    (public)."""
    import math as _math

    def _finite_or_none(value: object) -> float | None:
        if value is None:
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if _math.isfinite(f) else None

    return IterationSummary(
        iteration=int(record.iteration),
        iteration_dir=Path(record.iteration_dir),
        judge_decision=str(record.judge_decision),
        judge_score=_finite_or_none(record.judge_score),
        judge_reasoning=str(record.judge_reasoning),
        best_score=_finite_or_none(record.best_score),
        n_trials=int(record.n_trials),
        metric_name=str(record.metric_name),
        metric_value=_finite_or_none(record.metric_value),
        cancelled=bool(record.cancelled),
        error=_REFINE_ITERATION_FAILURE_MESSAGE if record.error else None,
    )


async def arun_refine(params: RefineInput) -> RefineOutput:
    """Asynchronously run the iterative refine loop.

    Delegates to :class:`IterativePhysicsRefinementTask`, which is
    synchronous (the per-iteration tune step blocks on a worker thread).
    We push the loop onto :func:`asyncio.to_thread` so callers in an
    event loop don't block.

    Args:
        params: Refine input parameters.

    Returns:
        :class:`RefineOutput`. ``success`` is ``True`` when
        ``termination_reason`` is ``"approved"`` or ``"max_iterations"``;
        ``"cancelled"`` / ``"error"`` map to ``success=False``.
    """
    # Lazy import — keep physics_agent.api free of tuning/refine modules
    # at module-load time. Same pattern as the lazy tune re-export.
    from physics_agent.tasks.iterative_physics_refinement import (
        IterativePhysicsRefinementTask,
    )

    logger.info(
        "physics-agent refine: user_prompt=%r max_iterations=%d threshold=%.3f",
        params.user_prompt,
        params.max_iterations,
        params.score_threshold,
    )

    task = IterativePhysicsRefinementTask(
        user_prompt=str(params.user_prompt),
        initial_scenario=params.scenario,
        physics_usd=params.physics_usd,
        output_dir=params.output_dir,
        engine=params.engine,
        optimizer=params.optimizer,
        max_trials=params.max_trials,
        seed=params.seed,
        max_iterations=params.max_iterations,
        history_window=params.history_window,
        score_threshold=params.score_threshold,
        judge_max_tokens=params.judge_max_tokens,
        judge_temperature=params.judge_temperature,
        chat_model=params.chat_model,
        vlm_model=params.vlm_model,
        reference_images=params.reference_images,
        reference_videos=params.reference_videos,
        reference_descriptions=params.reference_descriptions,
        reference_video_descriptions=params.reference_video_descriptions,
        reference_video_frames=params.reference_video_frames,
        judge_reference_frames=params.judge_reference_frames,
        judge_generated_frames=params.judge_generated_frames,
        force_record_video=params.force_record_video,
        render_winning_trial=params.render_winning_trial,
        visual_evidence_enabled=params.visual_evidence_enabled,
        llm_timeout_seconds=params.llm_timeout_seconds,
        cancel_event=params.cancel_event,
    )

    ctx: dict[str, Any] = {}
    if params.event_listener is not None:
        ctx["event_listener"] = params.event_listener

    try:
        result = await asyncio.to_thread(task.run, ctx)
    except Exception:  # pragma: no cover — defensive
        logger.error(_REFINE_FAILURE_MESSAGE)
        return RefineOutput(
            success=False,
            error=_REFINE_FAILURE_MESSAGE,
            output_dir=params.output_dir,
            termination_reason="error",
        )

    iteration_summaries = [_build_iteration_summary(r) for r in result.iterations]
    final_judge_score = (
        iteration_summaries[-1].judge_score if iteration_summaries else None
    )
    termination_reason = result.termination_reason

    # ``termination_reason`` is the canonical truth — surface it to the
    # ``success`` field so callers don't have to recompute the policy.
    cancelled_or_errored = termination_reason in ("cancelled", "error")
    success = not cancelled_or_errored
    error_msg: str | None = None
    if cancelled_or_errored:
        error_msg = _REFINE_FAILURE_MESSAGE

    raw_final_recording = getattr(result, "final_recording_usd", None)
    return RefineOutput(
        success=success,
        error=error_msg,
        output_dir=Path(result.output_dir),
        iterations=iteration_summaries,
        iteration_count=len(iteration_summaries),
        final_iteration=int(result.final_iteration),
        final_dir=Path(result.final_dir) if result.final_dir else None,
        final_recording_usd=(
            Path(raw_final_recording) if raw_final_recording else None
        ),
        final_recording_error=(
            _FINAL_RECORDING_FAILURE_MESSAGE
            if getattr(result, "final_recording_error", None)
            else None
        ),
        termination_reason=termination_reason,
        final_judge_score=final_judge_score,
        user_prompt=str(result.user_prompt),
    )


def run_refine(params: RefineInput) -> RefineOutput:
    """Synchronously run the iterative refine loop.

    Wrapper around :func:`arun_refine` for backward compatibility with
    sync callers (CLI, scripts). Inside an existing event loop, call
    :func:`arun_refine` directly instead.

    Args:
        params: Refine input parameters.

    Returns:
        :class:`RefineOutput`.
    """
    return asyncio.run(arun_refine(params))
