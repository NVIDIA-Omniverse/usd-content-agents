# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed input/output dataclasses and shared schema for tuning.

Mirrors the shape of :mod:`physics_agent.api.predict` (PredictInput/Output)
so callers can use the tuning API the same way they use the prediction API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from physics_agent.api.types import APIResult
from physics_agent.tuning.visual_evidence import (
    DEFAULT_JUDGE_GENERATED_FRAMES,
    DEFAULT_JUDGE_REFERENCE_FRAMES,
    DEFAULT_REFERENCE_VIDEO_FRAMES,
)

# Scenario kinds.
#
# ``drop_settle`` is the locked, validated scenario from #36 PR #43 — drop a
# rigid body from a fixed height and measure how it settles. The target dict
# is constrained to a known set of numeric keys.
#
# ``freeform`` is the NL-driven kind added in Part 1.1 (closed issue #51). The
# LLM interpreter authors a single-rigid-body scene with free-form initial
# conditions (pose, linear velocity, angular velocity, gravity, duration,
# surface friction). Multi-body scenes are out of scope for v1.1.
SCENARIO_DROP_SETTLE = "drop_settle"
SCENARIO_FREEFORM = "freeform"
SUPPORTED_SCENARIOS: tuple[str, ...] = (SCENARIO_DROP_SETTLE, SCENARIO_FREEFORM)

# Tunable physics parameter keys.
SUPPORTED_PARAM_KEYS: tuple[str, ...] = (
    "mass_scale",
    "static_friction",
    "dynamic_friction",
    "restitution",
    "contact_ke",
    "contact_kd",
)

# Parameter-specific reasonable bounds — used as fallbacks when a scenario
# YAML omits min/max for a parameter. These are widely-applicable physical
# defaults and intentionally conservative.
DEFAULT_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "mass_scale": (0.5, 2.0),
    "static_friction": (0.05, 1.5),
    "dynamic_friction": (0.05, 1.5),
    "restitution": (0.0, 1.0),
    "contact_ke": (100.0, 100000.0),
    "contact_kd": (0.0, 5000.0),
}


@dataclass(frozen=True)
class TunableParam:
    """A single tunable parameter — name + closed [min, max] interval."""

    name: str
    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_PARAM_KEYS:
            raise ValueError(
                f"Unsupported tunable parameter {self.name!r}. "
                f"Supported keys: {sorted(SUPPORTED_PARAM_KEYS)}"
            )
        if self.min_value > self.max_value:
            raise ValueError(
                f"Parameter {self.name!r} has min_value > max_value "
                f"({self.min_value} > {self.max_value})"
            )

    def clip(self, value: float) -> float:
        """Clip a value into the allowed range."""
        return max(self.min_value, min(self.max_value, float(value)))


@dataclass(frozen=True)
class Scenario:
    """A parsed tuning scenario YAML."""

    name: str
    params: tuple[TunableParam, ...]
    target: dict[str, Any]
    metric: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_SCENARIOS:
            raise ValueError(
                f"Unsupported scenario {self.name!r}. "
                f"v1 supports: {sorted(SUPPORTED_SCENARIOS)}"
            )
        if not self.params:
            raise ValueError(
                f"Scenario {self.name!r} must define at least one tunable parameter"
            )
        # Reject duplicate param names — silent override would be bug-prone.
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Scenario {self.name!r} has duplicate parameter names: {names}"
            )

    def param_dict(self) -> dict[str, TunableParam]:
        return {p.name: p for p in self.params}


@dataclass
class TrialRecord:
    """One optimizer trial — params evaluated + scalar score from the backend."""

    trial_index: int
    params: dict[str, float]
    score: float
    backend_metrics: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    failed: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "params": self.params,
            "score": self.score,
            "backend_metrics": self.backend_metrics,
            "duration_seconds": self.duration_seconds,
            "failed": self.failed,
            "error": self.error,
        }


@dataclass
class BackendArtifacts:
    """Files written by the backend per-trial that the runner may surface."""

    trajectory: Path | None = None
    raw_log: Path | None = None


@dataclass(kw_only=True)
class TuneInput:
    """Input parameters for the tuning API.

    Mirrors :class:`physics_agent.api.predict.PredictInput` style:
    a single dataclass that fully describes the run.

    All fields are keyword-only (``kw_only=True``). This is a deliberate
    break from PR #43's positional shape: Part 1.1 makes ``scenario``
    optional (the NL interpreter can author it from ``user_prompt``),
    which would silently rebind any old positional callers ``TuneInput(
    scenario_path, usd_path, ...)`` because the field types are all
    ``Path``-compatible. Forcing keyword-only construction surfaces such
    misuse at construction time rather than as a confusing
    file-not-found later in the run.
    """

    physics_usd: Path
    """Path to a simulation-ready USD authored by ``apply_physics``."""

    output_dir: Path
    """Directory where best_params.json, history.jsonl, etc. are written."""

    scenario: Path | dict[str, Any] | None = None
    """Scenario YAML path or pre-parsed dict.

    Optional when :attr:`user_prompt` is supplied — the NL interpreter
    authors a Scenario from the prompt in that case. When both are
    supplied, the parsed YAML wins on every conflict and the interpreter
    only fills in fields the YAML omits.
    """

    user_prompt: str | None = None
    """Free-form natural-language description of the desired tune run.

    Examples: ``"make this object bouncy"``, ``"spin a top on a smooth
    surface"``, ``"settle quickly with low rebound"``. When supplied, the
    NL interpreter produces a :class:`Scenario` (kind ``drop_settle`` or
    ``freeform``) and biased parameter bounds. Persisted to
    ``tune_results.json["user_prompt"]`` and rendered into ``report.md``
    for audit. See ``physics_agent.tasks.interpret_user_prompt_tuning``.
    """

    reference_images: list[Path] | None = None
    """Optional reference images for the visual/VLM judge. When supplied
    with judging enabled, the runner compares these against the rendered
    best-trial image sequence."""

    reference_videos: list[Path] | None = None
    """Optional reference videos for the visual/VLM judge. Videos are
    frame-extracted at intake and then treated as captioned reference
    images."""

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

    vlm_model: Any | None = None
    """Optional pre-built VLM instance for judging. When this is ``None``,
    the runner builds the default physics-agent VLM from environment-backed
    defaults. Text-only judge calls pass an empty media list."""

    engine: str = "ovphysx"
    """Backend engine. v1 supports ``"ovphysx"``, ``"newton"``, and ``"fake"``."""

    optimizer: str = "auto"
    """Optimizer name: ``auto`` (→ botorch), ``botorch``, ``random``, ``cma-es``."""

    max_trials: int = 30
    """Number of optimizer evaluations to run."""

    seed: int = 42
    """Seed for both optimizer and backend (when supported)."""

    enable_judge: bool = True
    """Run the VLM-as-judge over scenario YAML + history + best_params at
    the end of tune (and per refine iteration). Default-on per #51 spec.
    Set to ``False`` (CLI ``--no-judge``) for byte-identical-to-PR-#43
    output: no judge artifacts written, no model calls, no refine loop."""

    judge_max_iterations: int = 3
    """Pass-through hard cap on refine-loop iterations.

    .. important::
       This knob has **no effect on ``run_tune`` itself**. ``run_tune``
       is single-shot — when the judge returns ``continue`` the runner
       emits ``tune.judge.refine_skipped`` and returns. True iteration
       lives in the **first-class refine API**
       (``physics_agent.api.RefineInput.max_iterations`` /
       ``RefineInput``, see
       :class:`physics_agent.api.refine.RefineInput`) and the
       ``physics-agent refine`` CLI that delegates to it.

       The field is preserved on :class:`TuneInput` for wire-shape
       backward compatibility with the REST ``/tune`` route which
       advertised it before the dedicated refine surface existed
       (Round 15 added that surface). The validation (must be ``>= 1``)
       is kept so REST input coercion stays strict, but the value is
       only echoed back through artifacts — single-shot tune does not
       consume it. Callers that want true iteration must construct a
       :class:`RefineInput` and call :func:`run_refine` /
       :func:`arun_refine`, **not** :func:`run_tune`. (doyubkim Round 15
       blocker #3, building on CodeRabbit R13 thread #4.)"""

    judge_max_tokens: int | None = None
    """Optional max output tokens for the judge response.

    ``None`` uses the physics-agent judge default. This is intentionally
    separate from the base VLM construction ``max_tokens`` so the judge can
    keep a compact critique budget while prediction/VLM defaults remain large.
    """

    judge_temperature: float | None = None
    """Optional temperature for judge calls.

    ``None`` uses ``judge.temperature`` from the scenario YAML when present,
    otherwise the physics-agent judge default.
    """

    llm_timeout_seconds: float = 60.0
    """Hard deadline (seconds) on each LLM call invoked by Part-1.1
    (interpreter + judge). When the deadline expires:

    * The interpreter call raises :class:`TuningError` — the runner
      cannot proceed without a Scenario.
    * The judge call is skipped (logged + ``tune.judge.failed`` event);
      the tune artifacts are still written, just without a judge verdict.

    The orphaned LLM call continues in a background thread until the
    underlying provider client returns or the process exits — Python
    cannot kill a synchronous third-party call. This wrapper still
    unblocks the caller so a slow NIM/LangChain dependency cannot wedge
    the worker queue. Use ``-1`` to disable the timeout entirely (not
    recommended in production)."""

    cancel_event: Any = None
    """Optional :class:`threading.Event` / :class:`asyncio.Event` style object
    with an ``is_set()`` method. Polled between trials to support cancellation."""

    event_listener: Any = None
    """Optional EventListener (from world_understanding.agentic.events).
    The runner emits ``tune.trial.*`` and ``tune.completed`` events."""

    verbose: bool = False
    """Verbose progress logging."""


@dataclass
class TuneOutput(APIResult):
    """Output from the tuning API."""

    output_dir: Path | None = None
    """Resolved output directory containing the artifacts."""

    best_params: dict[str, float] = field(default_factory=dict)
    """Best parameter set found — keys are tunable param names."""

    best_score: float = float("inf")
    """Score of the best trial (lower is better)."""

    n_trials: int = 0
    """Number of trials actually evaluated (failed trials count)."""

    optimizer_used: str = ""
    """Optimizer name actually used (``auto`` is resolved here)."""

    engine_used: str = ""
    """Engine name actually used."""

    history: list[TrialRecord] = field(default_factory=list)
    """All trial records in evaluation order."""

    artifacts: dict[str, Path] = field(default_factory=dict)
    """Map of artifact name → on-disk path (best_params.json, etc.)."""

    cancelled: bool = False
    """True if the run terminated early due to a cancel signal."""

    needs_refinement: bool = False
    """True when the VLM judge returned ``decision == "continue"`` (i.e.
    the result is below the score threshold and would benefit from a
    refine iteration). Surfaced as an explicit field so REST/CLI
    consumers can detect needs-refinement state without parsing the
    judge dict; ``success`` remains independent of judge verdict so
    callers that only care about completion vs. cancellation behave
    unchanged. The v1.1 runner does not act on this signal — see
    ``judge_max_iterations`` plumbing for the forward-compat refine
    loop."""
