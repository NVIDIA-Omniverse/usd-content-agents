# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tuning runner — orchestrates scenario → optimizer → backend → artifacts.

Public entry points:
    :func:`run_tune` (synchronous wrapper around :func:`arun_tune`).
    :func:`arun_tune` (async coroutine — for ``await`` / FastAPI use).

These mirror the :mod:`physics_agent.api.predict` ``run_predict`` /
``arun_predict`` shape so existing callers can adopt the tuning API the same
way.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.events import EventListener

from .artifacts import (
    ARTIFACT_BEST_PARAMS,
    ARTIFACT_HISTORY,
    ARTIFACT_REPORT,
    ARTIFACT_RESULTS,
    ARTIFACT_TUNED_USD,
    ARTIFACT_VISUAL_COMPARISON,
    ensure_output_dir,
    open_history_writer,
    write_best_params,
    write_history_line,
    write_report_md,
    write_tune_results,
)
from .backend import (
    ENGINE_NEWTON,
    ENGINE_OVPHYSX,
    SUPPORTED_ENGINES,
    TuningBackend,
    get_backend,
    validate_engine_supports_param_names,
    validate_engine_supports_params,
)
from .errors import TuningCancelledError, TuningError
from .optimizers import (
    SUPPORTED_OPTIMIZERS,
    get_runner,
    resolve_optimizer,
)
from .scenario import load_scenario
from .scenario_resolution import get_resolved_bindings, resolve_scenario_bindings
from .types import SCENARIO_FREEFORM, Scenario, TrialRecord, TuneInput, TuneOutput
from .usd_patch import make_tuned_usd_path, patch_physics_usd
from .video_rendering import resolve_video_renderer
from .visual_evidence import (
    JudgeVisualEvidence,
    has_reference_media,
    prepare_reference_media,
    resolve_default_judge_vlm,
    validate_visual_frame_count,
    write_comparison_contact_sheet,
)

logger = logging.getLogger(__name__)

FAIL_CLOSED_JUDGE_ERROR_WITH_REF_MEDIA = (
    "Judge VLM unavailable with reference media; refusing to fall back "
    "to programmatic-only verdict."
)
FAIL_CLOSED_VISUAL_EVIDENCE_ERROR_WITH_REF_MEDIA = (
    "Visual judge evidence preparation failed with reference media; "
    "refusing to fall back to programmatic-only verdict."
)
_JUDGE_VLM_SETUP_FAILURE_MESSAGE = (
    "Failed to instantiate default judge VLM; judge will be marked unavailable."
)
_JUDGE_EXECUTION_FAILURE_MESSAGE = "Judge execution failed."


class _LLMTimeoutError(TuningError):
    """An LLM call did not return within ``llm_timeout_seconds``.

    Subclasses :class:`TuningError` so callers that already catch the
    base error type also catch deadline misses.
    """


def _run_with_llm_timeout(
    fn: Callable[..., Any],
    *args: Any,
    timeout_seconds: float,
    cancel_check: Callable[[], bool] | None = None,
    cancel_poll_seconds: float = 0.5,
    op_label: str,
    **kwargs: Any,
) -> Any:
    """Execute a synchronous model-adjacent call under a wall-clock deadline.

    Implementation notes (codex round 4 hardening):

    * The call runs on a **daemon** thread, not a ThreadPoolExecutor.
      Daemon threads do not block process exit, so a hung provider
      call cannot keep the service alive past shutdown. Successive
      timeouts no longer accumulate ThreadPoolExecutor instances.
    * The caller polls cancellation *while waiting*, not just before
      submission. A cancel signal mid-flight raises
      :class:`TuningCancelledError` immediately, so a cancelled
      session does not block on a hung LLM up to the full timeout.
    * Python cannot interrupt a blocked third-party call. The orphan
      daemon thread keeps the LLM client busy until the provider's own
      timeout fires (or the process exits). Provider-level deadlines
      and a bounded provider-client connection pool are the right
      complementary fixes — out of scope for this runner-level guard.

    Use ``timeout_seconds <= 0`` to disable the wall-clock deadline (the
    cancellation poll still applies). Test envs sometimes want this;
    production should always pin a real value.
    """
    if cancel_check is not None and cancel_check():
        raise TuningCancelledError(f"Tuning cancelled before {op_label} call")

    if timeout_seconds <= 0 and cancel_check is None:
        return fn(*args, **kwargs)

    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            result_box["value"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001
            # Capture every error type so it surfaces to the caller; we
            # specifically must not suppress KeyboardInterrupt /
            # SystemExit raised inside the LLM client.
            error_box["error"] = e
        finally:
            done.set()

    thread = threading.Thread(
        target=_worker,
        daemon=True,
        name=f"tune-{op_label}-llm",
    )
    thread.start()

    if timeout_seconds > 0:
        deadline: float | None = time.monotonic() + timeout_seconds
    else:
        deadline = None
    poll = max(0.05, float(cancel_poll_seconds))

    while True:
        if done.wait(poll):
            break
        if cancel_check is not None and cancel_check():
            raise TuningCancelledError(
                f"Tuning cancelled while waiting for {op_label} call"
            )
        if deadline is not None and time.monotonic() >= deadline:
            raise _LLMTimeoutError(
                f"{op_label} call exceeded {timeout_seconds}s deadline"
            )

    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")


def _validate_engine_supports_scenario(engine: str, scenario_name: str) -> None:
    """Raise :class:`TuningError` if the chosen ``engine`` cannot run the
    chosen ``scenario_name``.

    The truth lives in
    ``physics_agent.tuning.scenarios.SUPPORTED_SCENARIOS_PER_ENGINE``
    co-located with the ``resolve()`` dispatch table — adding a new
    scenario kind there is the single change needed to advertise it.
    Engines absent from the map are passed through (we don't know what
    they support); the existing ``SUPPORTED_ENGINES`` check already
    gates the engine name itself.
    """
    from physics_agent.tuning.scenarios import SUPPORTED_SCENARIOS_PER_ENGINE

    supported = SUPPORTED_SCENARIOS_PER_ENGINE.get(engine)
    if supported is None:
        return
    if scenario_name in supported:
        return
    raise TuningError(
        f"Engine {engine!r} does not support scenario kind "
        f"{scenario_name!r}. Supported pairs: "
        + ", ".join(
            f"{e}=[{', '.join(s)}]"
            for e, s in sorted(SUPPORTED_SCENARIOS_PER_ENGINE.items())
        )
        + ". Use a scenario kind your engine implements."
    )


def _validate_engine_supports_scenario_and_params(
    engine: str, scenario: Scenario
) -> None:
    """Reject engine/scenario/parameter combinations before trial execution."""
    _validate_engine_supports_scenario(engine, scenario.name)
    validate_engine_supports_params(engine, scenario)


def _resolve_judge_vlm_lazy() -> Any:
    """Build the default judge VLM, or return None on failure."""
    try:
        return resolve_default_judge_vlm()
    except Exception:
        logger.warning(_JUDGE_VLM_SETUP_FAILURE_MESSAGE)
        return None


# Suffixes treated as path-bearing scalar values when anchoring a
# scenario override YAML against its source directory. Keys with these
# suffixes are the canonical path-typed naming convention across the
# physics-agent configs (e.g. ``physics_usd``, ``ground_usd``,
# ``input.usd_path``); broadening this set risks anchoring a string
# field that the LLM intended to leave as a free-form label.
_SCENARIO_PATH_KEY_SUFFIXES: tuple[str, ...] = (
    "_path",
    "_usd",
    "_dir",
    "_file",
)


def _anchor_relative_paths_in_scenario_dict(node: Any, base_dir: Path) -> None:
    """In-place: resolve relative path-shaped string values against ``base_dir``.

    Recurses into nested dicts / lists. Absolute paths and non-string
    leaves pass through unchanged. URL-like values (``http://``,
    ``s3://``, …) are left alone too. The heuristic is intentionally
    conservative — only string values keyed by a recognised path
    suffix get anchored, so a YAML ``description: "load_dir/asset.yaml"``
    is never rewritten.

    Used by the runner's ``user_prompt + scenario`` override branch to
    preserve the config-relative-paths contract documented in the
    physics-agent CLAUDE.md (paths inside config YAML are relative to
    the YAML's directory). The plain ``scenario``-only branch already
    inherits this via ``load_scenario``'s file-path-typed input.
    """
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if isinstance(value, str) and any(
                key.endswith(suffix) for suffix in _SCENARIO_PATH_KEY_SUFFIXES
            ):
                # Skip URL-like and absolute paths.
                if "://" in value:
                    continue
                stripped = value.strip()
                if not stripped:
                    continue
                candidate = Path(stripped)
                if candidate.is_absolute():
                    continue
                node[key] = str((base_dir / candidate).resolve())
            else:
                _anchor_relative_paths_in_scenario_dict(value, base_dir)
    elif isinstance(node, list):
        for item in node:
            _anchor_relative_paths_in_scenario_dict(item, base_dir)


def _load_scenario_override_dict(scenario: Any) -> dict[str, Any]:
    """Load an explicit scenario override dict for the NL interpreter path."""
    if isinstance(scenario, dict):
        return dict(scenario)

    scenario_path = Path(scenario).resolve()
    text = scenario_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TuningError(
            f"scenario file {scenario!s} did not parse to a "
            f"mapping (got {type(data).__name__}); cannot use as "
            "explicit override for the NL interpreter."
        )

    # CodeRabbit R13 thread #3: anchor any config-relative paths inside
    # the override YAML so they resolve against the YAML file's directory
    # (matching the standard ``apps/<agent>/configs/*.yaml`` contract)
    # rather than silently inheriting the runner's CWD.
    _anchor_relative_paths_in_scenario_dict(data, scenario_path.parent)
    return data


def _explicit_scenario_param_names(scenario: Any) -> set[str]:
    """Return parameter names explicitly present in a scenario override."""
    data = _load_scenario_override_dict(scenario)
    raw_params = data.get("parameters")
    if not isinstance(raw_params, list):
        return set()

    names: set[str] = set()
    for raw_param in raw_params:
        if not isinstance(raw_param, dict):
            continue
        name = raw_param.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def _backend_param_keys_for_interpreter(backend: Any | None) -> tuple[str, ...] | None:
    """Return backend-advertised parameter names for NL scenario inference."""
    if backend is None:
        return None
    provider = getattr(backend, "tuning_capabilities", None)
    if not callable(provider):
        return None
    names: list[str] = []
    for capability in provider():
        name = getattr(capability, "param_name", None)
        if isinstance(name, str) and name not in names:
            names.append(name)
    return tuple(names) or None


def _resolve_scenario(
    params: TuneInput, *, backend: Any | None = None
) -> tuple[Scenario, Any]:
    """Resolve the scenario for this tune run.

    Three paths in priority order:

    1. ``user_prompt`` only → invoke the NL interpreter
       (:func:`physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt`)
       which authors a Scenario from scratch. The interpreter writes
       ``inferred_scenario.json`` (write-only audit record — never read
       back, no caching).
    2. ``user_prompt`` + ``scenario`` → load the YAML / dict and pass it as
       ``scenario_override`` to the interpreter so explicit user fields win
       on every conflict (per #51 spec).
    3. ``scenario`` only → use the existing :func:`load_scenario` path
       unchanged. No LLM call. Byte-identical to PR #43 baseline.

    Returns:
        ``(scenario, chat_model_or_none)`` — the chat model is returned so
        the caller can reuse it for the judge step without re-instantiating.
        It is ``None`` for the YAML-only path so we don't pull provider SDKs
        when the user didn't ask for any LLM call.
    """
    user_prompt = (params.user_prompt or "").strip() or None
    output_dir = Path(params.output_dir)

    if user_prompt is None:
        # YAML-only — preserves the pre-Part-1.1 import graph.
        return load_scenario(params.scenario), None

    # Lazy import — keeps the runner module cheap to import for callers that
    # never touch user_prompt. The interpreter is itself optimizer-free
    # (subprocess-test enforced) so this stays clean.
    from physics_agent.tasks.interpret_user_prompt_tuning import (
        infer_scenario_from_prompt,
    )

    scenario_override: dict[str, Any] | None = None
    if params.scenario is not None:
        scenario_override = _load_scenario_override_dict(params.scenario)

    physics_usd_for_context = (
        Path(params.physics_usd) if params.physics_usd is not None else None
    )

    # Wrap the LLM call in a hard deadline so a hung NIM/LangChain client
    # cannot wedge the tune worker. Cancel-event polling is best-effort —
    # see ``_run_with_llm_timeout`` for the orphan-thread caveat.
    cancel_check_local = _cancel_check_factory(params.cancel_event)
    scenario = _run_with_llm_timeout(
        infer_scenario_from_prompt,
        user_prompt,
        scenario_override=scenario_override,
        audit_dir=output_dir,
        physics_usd=physics_usd_for_context,
        backend_name=str(getattr(backend, "name", params.engine)),
        supported_param_keys=_backend_param_keys_for_interpreter(backend),
        timeout_seconds=params.llm_timeout_seconds,
        cancel_check=cancel_check_local,
        op_label="interpreter",
    )
    # The interpreter resolved the chat model lazily inside its body; we
    # don't get a handle to it. The judge step will resolve its own when
    # ``params.enable_judge`` is set. Both calls share the same default
    # provider/model.
    return scenario, None


def _emit(listener: Any, event_type: str, data: dict[str, Any]) -> None:
    """Best-effort event emit; never raises."""
    if listener is None:
        return
    try:
        listener.event(event_type, data)
    except Exception:  # pragma: no cover - listener bugs shouldn't crash tuning
        logger.debug("Event listener raised on %s", event_type, exc_info=True)


def _cancel_check_factory(
    cancel_event: Any,
) -> Callable[[], bool]:
    """Wrap a threading/asyncio Event-like object into a no-arg callable."""
    if cancel_event is None:
        return lambda: False
    is_set = getattr(cancel_event, "is_set", None)
    if not callable(is_set):
        raise TypeError(
            "cancel_event must be an Event-like object with an is_set() method"
        )
    return lambda: bool(is_set())


def _validate_inputs(params: TuneInput) -> None:
    if params.max_trials <= 0:
        raise ValueError(f"max_trials must be > 0, got {params.max_trials}")
    if params.engine not in SUPPORTED_ENGINES:
        raise ValueError(
            f"Unknown engine {params.engine!r}. Supported: {sorted(SUPPORTED_ENGINES)}"
        )
    if params.optimizer not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer {params.optimizer!r}. "
            f"Supported: {sorted(SUPPORTED_OPTIMIZERS)}"
        )
    if not isinstance(params.physics_usd, Path):
        # Be tolerant of strings — but don't accept arbitrary objects.
        if not isinstance(params.physics_usd, str):
            raise TypeError(
                "physics_usd must be a pathlib.Path or str, "
                f"got {type(params.physics_usd).__name__}"
            )
    # At least one of scenario / user_prompt must be supplied. The runner
    # invokes the NL interpreter when only ``user_prompt`` is set; both
    # missing is a configuration error worth catching before any
    # backend / optimizer setup runs.
    has_user_prompt = bool((params.user_prompt or "").strip())
    if params.scenario is None and not has_user_prompt:
        raise ValueError(
            "TuneInput requires either 'scenario' (a YAML path/dict) or "
            "'user_prompt' (a non-empty string). Both are missing."
        )
    if params.judge_max_iterations < 1:
        raise ValueError(
            f"judge_max_iterations must be >= 1, got {params.judge_max_iterations}"
        )
    if params.judge_max_tokens is not None and params.judge_max_tokens < 1:
        raise ValueError(
            f"judge_max_tokens must be >= 1, got {params.judge_max_tokens}"
        )
    if params.judge_temperature is not None:
        judge_temperature = float(params.judge_temperature)
        if not math.isfinite(judge_temperature) or judge_temperature < 0.0:
            raise ValueError(
                "judge_temperature must be finite and >= 0, "
                f"got {params.judge_temperature}"
            )
    params.reference_video_frames = validate_visual_frame_count(
        "reference_video_frames",
        params.reference_video_frames,
    )
    params.judge_reference_frames = validate_visual_frame_count(
        "judge_reference_frames",
        params.judge_reference_frames,
    )
    params.judge_generated_frames = validate_visual_frame_count(
        "judge_generated_frames",
        params.judge_generated_frames,
    )


def _evaluate_one(
    backend: Any,
    scenario: Scenario,
    params: dict[str, float],
    physics_usd: Path,
    *,
    seed: int,
    trial_index: int,
) -> TrialRecord:
    """Evaluate one trial and convert backend output → TrialRecord."""
    started = time.perf_counter()
    try:
        result = backend.evaluate(
            params=params,
            scenario=scenario,
            physics_usd=physics_usd,
            seed=seed,
        )
    except Exception as e:
        # Backend setup failures are NOT per-trial failures: a missing OvPhysX
        # daemon/venv or incomplete Newton install means every trial for the
        # rest of the budget will hit the same local precondition. Treating
        # them as failed trials burns the budget producing a generic "all
        # trials failed" outcome instead of surfacing the actionable install
        # hint up to CLI/REST. Re-raise so the runner surfaces the setup error.
        from .errors import NewtonUnavailableError, OvPhysXUnavailableError

        try:
            from world_understanding.functions.physics.ovphysx_daemon import (
                OvPhysXDaemonUnavailableError as _DaemonUnavailable,
            )
        except Exception:  # pragma: no cover — physics module always present
            _DaemonUnavailable = ()  # type: ignore[assignment]
        if isinstance(e, NewtonUnavailableError | OvPhysXUnavailableError) or (
            _DaemonUnavailable and isinstance(e, _DaemonUnavailable)
        ):
            raise
        elapsed = time.perf_counter() - started
        logger.warning("Trial %d failed: %s", trial_index, e, exc_info=True)
        return TrialRecord(
            trial_index=trial_index,
            params=dict(params),
            score=float("inf"),
            duration_seconds=elapsed,
            failed=True,
            error=str(e),
        )

    elapsed = time.perf_counter() - started
    if not isinstance(result, dict):
        return TrialRecord(
            trial_index=trial_index,
            params=dict(params),
            score=float("inf"),
            duration_seconds=elapsed,
            failed=True,
            error=f"Backend returned {type(result).__name__}, expected dict",
        )
    if "score" not in result:
        return TrialRecord(
            trial_index=trial_index,
            params=dict(params),
            score=float("inf"),
            duration_seconds=elapsed,
            failed=True,
            error="Backend result missing required key 'score'",
        )

    # Coerce score → float defensively so a malformed backend (None,
    # non-numeric string, NaN proxy) does not abort the whole run with an
    # unhandled TypeError. Any failure here is recorded as a failed trial.
    raw_score = result["score"]
    try:
        score = float(raw_score)
    except (TypeError, ValueError) as e:
        return TrialRecord(
            trial_index=trial_index,
            params=dict(params),
            score=float("inf"),
            duration_seconds=elapsed,
            failed=True,
            error=f"Backend returned non-numeric score {raw_score!r}: {e}",
        )
    if not math.isfinite(score):
        # Reject NaN, +inf, and -inf alike — a backend overflow that
        # returns -inf would otherwise be recorded as the best trial of
        # the sweep. Non-finite is treated as a failed trial; the
        # optimizer's "lower is better" view sees ``inf`` so this
        # candidate is never chosen.
        return TrialRecord(
            trial_index=trial_index,
            params=dict(params),
            score=float("inf"),
            duration_seconds=elapsed,
            failed=True,
            error=f"Backend returned non-finite score {raw_score!r}",
        )
    return TrialRecord(
        trial_index=trial_index,
        params=dict(params),
        score=score,
        backend_metrics={k: v for k, v in result.items() if k != "score"},
        duration_seconds=elapsed,
        failed=False,
    )


def _discover_camera_paths(stage_path: Path) -> list[str] | None:
    """Return camera prim paths from a USD stage, or None on failure."""
    try:
        from pxr import Usd, UsdGeom
    except ImportError:  # pragma: no cover - defensive
        return None
    try:
        stage = Usd.Stage.Open(str(stage_path))
    except Exception:
        return None
    if stage is None:
        return None
    paths: list[str] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            paths.append(str(prim.GetPath()))
    return paths or None


def _render_best_trial_for_visual_judge(
    *,
    output_dir: Path,
    history: list[TrialRecord],
    scenario: Scenario,
) -> tuple[list[Path], str | None]:
    """Render the winning trial's recording.usd for visual judging."""
    successful = [t for t in history if not t.failed]
    if not successful:
        return [], "every trial failed; no winning trial to render"
    best = min(successful, key=lambda t: t.score)
    bm = best.backend_metrics or {}
    recording = bm.get("recording_usd") or bm.get("recording_usda")
    if not recording:
        return [], "winning trial did not persist recording_usd"
    try:
        from world_understanding.functions.graphics import render_time_sampled_usd
    except ImportError:
        return [], "render_time_sampled_usd unavailable"

    target = scenario.target or {}
    renderer = resolve_video_renderer(target)
    render_dir = output_dir / "judge_render"
    try:
        frames = render_time_sampled_usd(
            Path(recording),
            render_dir,
            renderer=renderer,
            cameras=_discover_camera_paths(Path(recording)),
            fps=int(target.get("sample_fps", 30)),
            max_duration_seconds=float(target.get("duration_s", 2.0)),
            image_width=int(target.get("video_image_width", 512)),
            image_height=int(target.get("video_image_height", 512)),
            num_sensor_updates=int(target.get("video_sensor_updates", 32)),
            render_mode=str(target.get("video_render_mode", "rt2")),
        )
    except Exception as exc:  # noqa: BLE001 - judge render should degrade
        logger.warning("visual judge render failed: %s", exc, exc_info=True)
        return [], type(exc).__name__
    if not frames:
        return [], "renderer produced no frames"
    return frames, None


def _prepare_visual_evidence_for_judge(
    *,
    params: TuneInput,
    output_dir: Path,
    history: list[TrialRecord],
    scenario: Scenario,
    include_generated_without_reference: bool = False,
) -> JudgeVisualEvidence | None:
    """Prepare reference and/or generated image evidence for the judge."""
    reference_media_requested = has_reference_media(
        reference_images=params.reference_images,
        reference_videos=params.reference_videos,
    )
    if not reference_media_requested and not include_generated_without_reference:
        return None

    if reference_media_requested:
        try:
            reference_evidence = prepare_reference_media(
                reference_images=params.reference_images,
                reference_videos=params.reference_videos,
                reference_descriptions=params.reference_descriptions,
                reference_video_descriptions=params.reference_video_descriptions,
                output_dir=output_dir,
                frames_per_video=params.reference_video_frames,
            )
        except Exception as exc:  # noqa: BLE001 - judge should persist degraded status
            return JudgeVisualEvidence(reference_error=type(exc).__name__)
    else:
        reference_evidence = JudgeVisualEvidence()

    frames, error = _render_best_trial_for_visual_judge(
        output_dir=output_dir,
        history=history,
        scenario=scenario,
    )
    evidence = reference_evidence.with_generated_images(frames, generated_error=error)
    if evidence.has_reference_media:
        comparison_path, comparison_error = write_comparison_contact_sheet(
            evidence,
            output_dir / ARTIFACT_VISUAL_COMPARISON,
            max_reference_images=params.judge_reference_frames,
            max_generated_images=params.judge_generated_frames,
        )
        evidence = evidence.with_comparison_image(
            comparison_path,
            comparison_error=comparison_error,
        )
    return evidence


def _scenario_requests_generated_visual_judge(scenario: Scenario) -> bool:
    """Return True when judge quality depends on generated rollout frames."""
    return scenario.name == SCENARIO_FREEFORM


def _visual_evidence_fail_closed_error(
    visual_evidence: JudgeVisualEvidence | None,
) -> str | None:
    """Return a concise evidence-prep failure reason for media-backed judging."""
    if visual_evidence is None:
        return "visual evidence preparation returned no evidence"
    if visual_evidence.reference_error is not None:
        return f"reference media failed: {visual_evidence.reference_error}"
    if visual_evidence.generated_error is not None:
        return f"generated evidence failed: {visual_evidence.generated_error}"
    return None


def _do_run_tune(params: TuneInput) -> TuneOutput:
    """Synchronous body of the tuning run. ``arun_tune`` calls this in a thread."""
    _validate_inputs(params)

    output_dir = ensure_output_dir(Path(params.output_dir))
    listener = params.event_listener

    # Explicit scenario parameter names are cheap to inspect, so validate them
    # before backend construction/warmup. This catches deterministic
    # engine/parameter mismatches (for example OvPhysX + Newton-only
    # contact_ke/contact_kd) without starting a daemon, compiling Warp kernels,
    # or invoking the NL interpreter. It intentionally does not fully parse the
    # scenario kind for non-Newton engines, preserving the existing
    # install-hint-first ordering for malformed OvPhysX YAML.
    scenario: Scenario | None = None
    scenario_validated = False
    has_user_prompt = bool((params.user_prompt or "").strip())
    if params.scenario is not None:
        try:
            explicit_param_names = _explicit_scenario_param_names(params.scenario)
        except (OSError, TuningError, yaml.YAMLError):
            if params.engine != ENGINE_OVPHYSX or has_user_prompt:
                raise
            explicit_param_names = set()
        validate_engine_supports_param_names(params.engine, explicit_param_names)
        if not has_user_prompt and params.engine == ENGINE_NEWTON:
            scenario, _interpreter_chat_model = _resolve_scenario(params)
            _validate_engine_supports_scenario_and_params(params.engine, scenario)
            scenario_validated = True

    # Resolve the cheap, local capability gates BEFORE any LLM-touching
    # work. ``_resolve_scenario`` invokes the NL interpreter on the
    # ``user_prompt`` path which costs a real API call; running it before
    # ``resolve_optimizer`` / ``get_backend`` would burn that cost on a
    # box that's missing the tuning extra or has a typo'd engine name.
    # Surface those install-time precondition errors first.
    optimizer_used = resolve_optimizer(params.optimizer)
    backend = get_backend(params.engine)

    # Keep every backend-touching step under shutdown. OvPhysX lazy-creates
    # a daemon subprocess during warmup/evaluate; binding resolution can fail
    # after warmup, so the cleanup must cover more than just the trial loop.
    try:
        # Round 14 (Codex CX P2#3): for the OvPhysX backend the daemon
        # startup handshake (which fails fast on a missing ovphysx venv via
        # ``OvPhysXDaemonUnavailableError``) only happens on the first
        # ``evaluate`` call. On the ``user_prompt`` path we'd otherwise burn
        # a paid LLM call inside ``_resolve_scenario`` before a deterministic
        # local precondition failure. Warm up the backend now — on FakeBackend
        # this is a no-op (no ``warmup`` attribute). OvPhysX reuses the daemon
        # on a follow-up ``evaluate``; Newton re-runs the probe on any later
        # ``warmup`` call and otherwise reuses the simulator object.
        warmup = getattr(backend, "warmup", None)
        if callable(warmup):
            warmup()

        # Scenario resolution — three paths, see ``_resolve_scenario``:
        # YAML-only (existing path; no LLM), user_prompt-only (NL interpreter
        # authors a Scenario), or both (interpreter fills gaps; explicit YAML
        # wins on every conflict).
        if scenario is None:
            scenario, _interpreter_chat_model = _resolve_scenario(
                params,
                backend=backend,
            )
        # Up-front capability check: refuse to spend a trial budget on a
        # scenario the chosen engine cannot execute. The current dispatch
        # map (``SUPPORTED_SCENARIOS_PER_ENGINE``) advertises both
        # drop_settle and freeform on the daemon-based ``ovphysx`` and
        # FakeBackend, so this gate is a forward-compat hook rather than a
        # currently-rejecting check — added scenario kinds that haven't
        # registered an evaluator will still trip it.
        if not scenario_validated:
            _validate_engine_supports_scenario_and_params(params.engine, scenario)
        cancel_check = _cancel_check_factory(params.cancel_event)
        physics_usd = Path(params.physics_usd)
        scenario = resolve_scenario_bindings(
            scenario,
            physics_usd=physics_usd,
            backend=backend,
        )
        return _do_run_tune_inner(
            params=params,
            output_dir=output_dir,
            listener=listener,
            scenario=scenario,
            cancel_check=cancel_check,
            physics_usd=physics_usd,
            optimizer_used=optimizer_used,
            backend=backend,
        )
    finally:
        getattr(backend, "shutdown", lambda: None)()


def _do_run_tune_inner(
    *,
    params: TuneInput,
    output_dir: Path,
    listener: EventListener | None,
    scenario: Scenario,
    cancel_check: Callable[[], bool],
    physics_usd: Path,
    optimizer_used: str,
    backend: TuningBackend,
) -> TuneOutput:
    """The trial-loop body of :func:`_do_run_tune`. Split out so the
    parent function can reap the backend in a ``finally`` without
    indenting 350 lines of body.
    """

    history: list[TrialRecord] = []
    cancelled_flag = {"value": False}

    # Wrap the user's cancel_check so the optimizer's polite-exit path also
    # flips ``cancelled_flag`` — without this, an optimizer that polls
    # cancel_check() at the top of its loop and returns cleanly (random,
    # botorch) would leave the runner reporting ``cancelled=False`` even
    # though it stopped early.
    def cancel_check_wrapped() -> bool:
        if cancel_check():
            cancelled_flag["value"] = True
            return True
        return False

    started_at = datetime.now(UTC).isoformat()
    history_handle = open_history_writer(output_dir)

    _emit(
        listener,
        "tune.started",
        {
            "scenario": scenario.name,
            "engine": params.engine,
            "optimizer": optimizer_used,
            "max_trials": params.max_trials,
            "seed": params.seed,
            "output_dir": str(output_dir),
        },
    )

    def evaluate_and_record(candidate: dict[str, float]) -> float:
        # Stop accepting new trials once the cancel signal fires; the
        # optimizer's own cancel_check exit will follow shortly.
        if cancel_check():
            cancelled_flag["value"] = True
            raise TuningCancelledError("Tuning cancelled by caller")
        if len(history) >= params.max_trials:
            raise StopIteration  # pragma: no cover - guard against bad optimizer
        # Clip into bounds — optimizers occasionally return tiny FP overshoots.
        clipped = {tp.name: tp.clip(candidate[tp.name]) for tp in scenario.params}
        trial = _evaluate_one(
            backend,
            scenario,
            clipped,
            physics_usd,
            seed=params.seed + len(history),
            trial_index=len(history),
        )
        history.append(trial)
        write_history_line(history_handle, trial)
        _emit(
            listener,
            "tune.trial.completed",
            {
                "trial_index": trial.trial_index,
                "score": trial.score,
                "params": trial.params,
                "failed": trial.failed,
            },
        )
        return trial.score

    runner = get_runner(optimizer_used)
    try:
        runner(
            scenario,
            evaluate_and_record,
            max_trials=params.max_trials,
            seed=params.seed,
            cancel_check=cancel_check_wrapped,
        )
    except TuningCancelledError:
        cancelled_flag["value"] = True
    finally:
        history_handle.close()

    completed_at = datetime.now(UTC).isoformat()

    if not history:
        # Optimizer ran zero trials.
        if cancelled_flag["value"]:
            # Pre-first-trial cancellation — return a cancelled TuneOutput
            # AFTER emitting the canonical artifact set so callers see the
            # same on-disk layout they would for a normal run.
            empty_params: dict[str, float] = {}
            zero_artifacts: dict[str, Path] = {}
            zero_artifacts[ARTIFACT_HISTORY] = output_dir / ARTIFACT_HISTORY
            zero_artifacts[ARTIFACT_BEST_PARAMS] = write_best_params(
                output_dir, empty_params, float("inf")
            )
            zero_artifacts[ARTIFACT_RESULTS] = write_tune_results(
                output_dir,
                params_input=params,
                scenario=scenario,
                optimizer_used=optimizer_used,
                engine_used=params.engine,
                best_params=empty_params,
                best_score=float("inf"),
                history=[],
                cancelled=True,
                started_at=started_at,
                completed_at=completed_at,
            )
            zero_artifacts[ARTIFACT_REPORT] = write_report_md(
                output_dir,
                scenario=scenario,
                optimizer_used=optimizer_used,
                engine_used=params.engine,
                best_params=empty_params,
                best_score=float("inf"),
                history=[],
                cancelled=True,
                user_prompt=params.user_prompt,
            )
            # Emit ``tune.cancelled`` (not ``tune.completed``) so SSE
            # listeners that map terminal events to step state see a
            # CANCELLED frame instead of a successful COMPLETED frame
            # (CodeRabbit Round 11 thread #10).
            _emit(
                listener,
                "tune.cancelled",
                {
                    "best_score": None,
                    "best_params": {},
                    "n_trials": 0,
                    "cancelled": True,
                    "engine": params.engine,
                    "optimizer": optimizer_used,
                },
            )
            return TuneOutput(
                success=False,
                error="Tuning cancelled before any trial completed",
                output_dir=output_dir,
                best_params={},
                best_score=float("inf"),
                n_trials=0,
                optimizer_used=optimizer_used,
                engine_used=params.engine,
                history=[],
                artifacts=zero_artifacts,
                cancelled=True,
            )
        # Otherwise a real bug (optimizer never called evaluate). Surface
        # a clear failure.
        msg = "Tuning produced zero trials; cannot select a best parameter set."
        _emit(listener, "tune.failed", {"error": msg})
        raise TuningError(msg)

    # Successful trials only when picking the winner; if every trial failed,
    # fall back to the best (lowest score) failed trial so we still emit a
    # complete artifact set, but mark the run as failed.
    successful = [t for t in history if not t.failed]
    if successful:
        best = min(successful, key=lambda t: t.score)
    else:
        best = min(history, key=lambda t: t.score)

    # ---- Durable trial artifacts (write BEFORE judging) ------------------
    #
    # Codex round 6 caught that running the judge before persisting these
    # artifacts means a cancellation that lands during the judge model
    # wait throws away completed tune work. Persist best_params.json and
    # the tuned USD now so cancellation during judging keeps the
    # already-computed work auditable. tune_results.json and report.md
    # are written *after* the judge step because their bytes depend on
    # the verdict; we re-write them with judge_result attached when the
    # judge succeeds, and again with status=cancelled/failed if it
    # doesn't.
    artifacts: dict[str, Path] = {}
    artifacts[ARTIFACT_HISTORY] = output_dir / ARTIFACT_HISTORY
    artifacts[ARTIFACT_BEST_PARAMS] = write_best_params(
        output_dir, best.params, best.score
    )

    # ---- Judge (Part 1.1) ------------------------------------------------
    #
    # Spec (#51): the judge runs at the end of tune unless ``--no-judge``
    # disables it. When disabled, the artifact bytes are identical to the
    # pre-Part-1.1 baseline (no ``judge`` key in tune_results.json, no
    # judge section in report.md, no model
    # calls).
    #
    # Codex round 3: when ``enable_judge=True``, we ALWAYS write a ``judge``
    # block to tune_results.json — completed, failed, or cancelled. This
    # disambiguates "judge disabled" (no key, byte-identical baseline)
    # from "judge attempted but ${reason}". The byte-identical guarantee
    # for ``enable_judge=False`` is preserved.
    judge_result_dict: dict[str, Any] | None = None
    visual_evidence: JudgeVisualEvidence | None = None
    judge_vlm_model: Any | None = None
    fail_closed_judge_error: str | None = None
    reference_media_requested = has_reference_media(
        reference_images=params.reference_images,
        reference_videos=params.reference_videos,
    )
    generated_visual_requested = _scenario_requests_generated_visual_judge(scenario)
    visual_evidence_requested = reference_media_requested or generated_visual_requested
    if params.enable_judge and not cancelled_flag["value"]:
        # Lazy import — keeps ``physics_agent.tuning`` import-clean for
        # callers that disable judging (subprocess test enforces this).
        from physics_agent.tasks.judge_tune import (
            JudgeError,
            run_tune_judge,
        )

        try:
            if visual_evidence_requested:
                visual_evidence = _run_with_llm_timeout(
                    _prepare_visual_evidence_for_judge,
                    params=params,
                    output_dir=output_dir,
                    history=history,
                    scenario=scenario,
                    include_generated_without_reference=generated_visual_requested,
                    timeout_seconds=params.llm_timeout_seconds,
                    cancel_check=cancel_check,
                    op_label="visual evidence preparation",
                )
                evidence_error = _visual_evidence_fail_closed_error(visual_evidence)
                if evidence_error is not None and reference_media_requested:
                    fail_closed_judge_error = (
                        FAIL_CLOSED_VISUAL_EVIDENCE_ERROR_WITH_REF_MEDIA
                    )
                    raise JudgeError(evidence_error)
            judge_chat_model = None
            judge_vlm_model = (
                params.vlm_model
                if params.vlm_model is not None
                else _run_with_llm_timeout(
                    _resolve_judge_vlm_lazy,
                    timeout_seconds=params.llm_timeout_seconds,
                    cancel_check=cancel_check,
                    op_label="judge VLM setup",
                )
            )
            judge_result = _run_with_llm_timeout(
                run_tune_judge,
                scenario=scenario,
                history=history,
                best_params=dict(best.params),
                user_prompt=params.user_prompt,
                chat_model=judge_chat_model,
                vlm_model=judge_vlm_model,
                visual_evidence=visual_evidence,
                judge_max_tokens=params.judge_max_tokens,
                judge_temperature=params.judge_temperature,
                judge_reference_frames=params.judge_reference_frames,
                judge_generated_frames=params.judge_generated_frames,
                iteration=1,
                timeout_seconds=params.llm_timeout_seconds,
                cancel_check=cancel_check,
                op_label="judge",
            )
            # Codex round 9: fail closed when the VLM judge was
            # unavailable. Programmatic-only scores tend to clear the
            # approve threshold trivially (best_params are always clipped
            # into bounds, finite_best=1, no failed trials), so trusting
            # them as a real verdict would silently approve runs under
            # model misconfiguration or provider outage. ``status`` is
            # set to ``degraded`` (not ``completed``) and downstream
            # signals like ``needs_refinement`` ignore degraded results.
            status = "degraded" if judge_result.llm_unavailable else "completed"
            judge_result_dict = {
                "enabled": True,
                "status": status,
                "attempted_iterations": int(judge_result.iterations),
                **judge_result.to_dict(),
            }
            if judge_result.llm_unavailable and reference_media_requested:
                fail_closed_judge_error = FAIL_CLOSED_JUDGE_ERROR_WITH_REF_MEDIA
            _emit(
                listener,
                "tune.judge.completed",
                {
                    "decision": judge_result.decision,
                    "score": judge_result.score,
                    "programmatic_score": judge_result.programmatic_score,
                    "llm_score": judge_result.llm_score,
                    "iterations": judge_result.iterations,
                    "llm_unavailable": judge_result.llm_unavailable,
                },
            )
            if judge_result.decision == "continue" and params.judge_max_iterations > 1:
                _emit(
                    listener,
                    "tune.judge.refine_skipped",
                    {
                        "reason": (
                            "judge requested continue but the v1.1 runner "
                            "ships single-iteration judging; "
                            "judge_max_iterations is captured for forward "
                            "compat only."
                        ),
                        "judge_max_iterations": params.judge_max_iterations,
                    },
                )
        except TuningCancelledError as e:
            # Codex round 6: a cancel that fires during the judge model
            # wait must NOT discard the trial work we already completed.
            # Mark the run cancelled, persist a judge.status="cancelled"
            # marker, and fall through to write the canonical artifact
            # set so callers can audit best_params/history.
            logger.info(
                "Judge cancelled by caller; persisting tune artifacts "
                "with judge status='cancelled'."
            )
            cancelled_flag["value"] = True
            judge_result_dict = {
                "enabled": True,
                "status": "cancelled",
                "attempted_iterations": 1,
            }
            _emit(
                listener,
                "tune.judge.cancelled",
                {"reason": str(e)},
            )
        except (JudgeError, _LLMTimeoutError) as e:
            # Judge failures (including LLM-call timeouts) are non-fatal —
            # the tune itself succeeded; we persist a durable failure
            # marker (codex round 3) so consumers can distinguish
            # "judge attempted but failed" from "judge disabled". The
            # exception's str() is intentionally NOT included to avoid
            # leaking provider-internal error detail across REST.
            error_type = (
                "_LLMTimeoutError" if isinstance(e, _LLMTimeoutError) else "JudgeError"
            )
            logger.warning(
                "Judge execution failed; persisting tune artifacts with "
                "judge status='failed'."
            )
            judge_result_dict = {
                "enabled": True,
                "status": "failed",
                "error_type": error_type,
                "attempted_iterations": 1,
            }
            if reference_media_requested and fail_closed_judge_error is None:
                fail_closed_judge_error = FAIL_CLOSED_JUDGE_ERROR_WITH_REF_MEDIA
            _emit(
                listener,
                "tune.judge.failed",
                {
                    "error": _JUDGE_EXECUTION_FAILURE_MESSAGE,
                    "error_type": error_type,
                },
            )

    # tune_results.json + report.md are deferred until here so judge
    # state (completed/failed/cancelled/disabled) is captured in the
    # artifact bytes.
    #
    # Re-stamp ``completed_at`` so it reflects the *actual* end time —
    # i.e. after the judge / artifact-build phase, not just after the
    # optimizer loop. Otherwise persisted metadata reads back saying
    # the tune finished before the judge ran, which skews durable
    # status reads and makes a cancellation that lands during the
    # judge wait look like it happened post-completion. The zero-trial
    # cancellation branch above kept its own earlier ``completed_at``
    # because no judge phase runs there.
    completed_at = datetime.now(UTC).isoformat()
    artifacts[ARTIFACT_RESULTS] = write_tune_results(
        output_dir,
        params_input=params,
        scenario=scenario,
        optimizer_used=optimizer_used,
        engine_used=params.engine,
        best_params=best.params,
        best_score=best.score,
        history=history,
        cancelled=cancelled_flag["value"],
        started_at=started_at,
        completed_at=completed_at,
        judge_result=judge_result_dict,
    )
    artifacts[ARTIFACT_REPORT] = write_report_md(
        output_dir,
        scenario=scenario,
        optimizer_used=optimizer_used,
        engine_used=params.engine,
        best_params=best.params,
        best_score=best.score,
        history=history,
        cancelled=cancelled_flag["value"],
        user_prompt=params.user_prompt,
        judge_result=judge_result_dict,
    )
    if (
        visual_evidence is not None
        and visual_evidence.comparison_image_path is not None
        and visual_evidence.comparison_image_path.exists()
    ):
        artifacts[ARTIFACT_VISUAL_COMPARISON] = visual_evidence.comparison_image_path

    # Tuned USD — best-effort. The run is still considered successful when
    # USD patching fails (we still wrote best_params.json + report.md), but
    # we surface the failure on the API output and skip the artifact entry.
    try:
        tuned_usd_path = make_tuned_usd_path(output_dir)
        patch_physics_usd(
            physics_usd,
            tuned_usd_path,
            best.params,
            bindings=get_resolved_bindings(scenario),
        )
        artifacts[ARTIFACT_TUNED_USD] = tuned_usd_path
    except Exception as e:
        logger.warning("Failed to write tuned USD artifact: %s", e, exc_info=True)
        _emit(
            listener,
            "tune.warning",
            {"warning": "tuned_usd_write_failed", "error": str(e)},
        )

    # Pick a terminal event type that matches the actual outcome
    # (CodeRabbit Round 11 thread #10). Listeners like
    # ``physics_agent_service._TuneEventListener`` map ``tune.completed`` to
    # ``StepState.COMPLETED``; emitting that on a cancelled-or-best-failed
    # run flashes a successful terminal frame on the wire that SSE clients
    # see seconds before the worker corrects the durable session metadata.
    _terminal_data = {
        "best_score": best.score,
        "best_params": best.params,
        "n_trials": len(history),
        "cancelled": cancelled_flag["value"],
        "engine": params.engine,
        "optimizer": optimizer_used,
    }
    if cancelled_flag["value"]:
        _emit(listener, "tune.cancelled", _terminal_data)
    elif best.failed or fail_closed_judge_error:
        _emit(
            listener,
            "tune.failed",
            {
                **_terminal_data,
                "error": (
                    best.error
                    if best.failed
                    else fail_closed_judge_error
                    or "Every trial failed; no successful run."
                ),
            },
        )
    else:
        _emit(listener, "tune.completed", _terminal_data)

    # ``needs_refinement`` requires a real (LLM-attested) judge verdict
    # — degraded/programmatic-only verdicts are NOT trusted to drive a
    # refine signal because the programmatic component alone is not
    # discriminative enough (codex round 9).
    needs_refinement = bool(
        judge_result_dict
        and judge_result_dict.get("status") == "completed"
        and judge_result_dict.get("decision") == "continue"
    )
    # When every trial fails, ``best.failed`` is True but ``best.error``
    # already carries the canonical "first failure" message. Surface it
    # through TuneOutput.error so programmatic callers don't have to
    # re-derive it from history. (CodeRabbit R13 thread #5.)
    if cancelled_flag["value"]:
        terminal_error: str | None = "Tuning cancelled"
    elif best.failed:
        # ``best.error`` is set on every failed trial; fall back to a
        # generic message if the optimizer somehow returned a failed
        # record without one (shouldn't happen but the field is Optional).
        terminal_error = best.error or (
            f"All {len(history)} trial(s) failed; "
            "see history.jsonl for per-trial errors."
        )
    elif fail_closed_judge_error:
        terminal_error = fail_closed_judge_error
    else:
        terminal_error = None
    return TuneOutput(
        success=not (
            cancelled_flag["value"] or bool(best.failed) or fail_closed_judge_error
        ),
        error=terminal_error,
        output_dir=output_dir,
        best_params=dict(best.params),
        best_score=float(best.score),
        n_trials=len(history),
        optimizer_used=optimizer_used,
        engine_used=params.engine,
        history=list(history),
        artifacts=artifacts,
        cancelled=cancelled_flag["value"],
        needs_refinement=needs_refinement,
    )


def run_tune(params: TuneInput) -> TuneOutput:
    """Run a tuning sweep synchronously.

    Mirrors :func:`physics_agent.api.predict.run_predict` — for callers that
    just want a blocking call. Calls the synchronous implementation
    directly rather than wrapping :func:`arun_tune` in :func:`asyncio.run`,
    so this entry point is safe to call from threads that already have an
    event loop bound (e.g. notebook kernels). Async callers should use
    :func:`arun_tune` to integrate with their own loop.
    """
    return _do_run_tune(params)


async def arun_tune(params: TuneInput) -> TuneOutput:
    """Async entry point for the tuning runner.

    Runs the (CPU/IO-bound) inner loop in a thread so it does not block the
    event loop. The runner internals are synchronous because BoTorch /
    CMA-ES / pxr are all sync libraries; running them via ``to_thread`` keeps
    the async surface clean.
    """
    return await asyncio.to_thread(_do_run_tune, params)


__all__ = ["run_tune", "arun_tune"]
