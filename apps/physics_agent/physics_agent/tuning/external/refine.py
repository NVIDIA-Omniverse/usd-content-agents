# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification-gated iterative refinement for trusted external runtimes."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from world_understanding.optimization import RefinementLoop

from physics_agent.api.types import APIResult
from physics_agent.tuning.visual_evidence import (
    DEFAULT_JUDGE_GENERATED_FRAMES,
    DEFAULT_JUDGE_REFERENCE_FRAMES,
    JudgeVisualEvidence,
    prepare_reference_media,
    validate_reference_image_path,
    validate_visual_frame_count,
    write_comparison_contact_sheet,
)

from .artifacts import (
    BEST_PARAMS,
    BEST_RECORDING,
    HISTORY,
    RESULTS,
    RUN_SPEC,
    external_trial_payload,
    summarize_external_trials,
    write_manifest,
)
from .config import load_external_tune_spec
from .fingerprint import build_runtime_fingerprint
from .judge import run_external_judge
from .refiner import ExternalRefineError, run_external_refiner
from .runner import (
    _validate_approved_qualification,
    _validate_run_directory_layout,
    run_external_tune,
)
from .types import ExternalTuneInput, ExternalTuneOutput, ExternalTuneSpec

_MAX_ITERATIONS = 12


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
            for payload in payloads
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _cancelled(event: Any) -> bool:
    return bool(event is not None and event.is_set())


def _emit(listener: Any, event_type: str, data: dict[str, Any]) -> None:
    if listener is None:
        return
    try:
        listener.event(event_type, data)
    except Exception:  # pragma: no cover - listeners cannot break refinement
        return


def _run_with_timeout(
    fn: Callable[..., Any],
    *args: Any,
    timeout_seconds: float,
    **kwargs: Any,
) -> Any:
    if timeout_seconds <= 0:
        return fn(*args, **kwargs)
    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}
    done = threading.Event()

    def worker() -> None:
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - preserve provider errors
            error["value"] = exc
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True, name="external-refine-llm").start()
    if not done.wait(timeout_seconds):
        raise TimeoutError(f"LLM call exceeded {timeout_seconds:g}s")
    if "value" in error:
        raise error["value"]
    return result.get("value")


def _search_payload(spec: ExternalTuneSpec) -> dict[str, dict[str, float]]:
    return {
        parameter.name: {
            "min": parameter.min_value,
            "max": parameter.max_value,
        }
        for parameter in spec.params
    }


@dataclass(kw_only=True)
class ExternalRefineInput:
    """Inputs for qualification-gated external-runtime refinement."""

    config: Path | dict[str, Any] | ExternalTuneSpec
    output_dir: Path
    user_prompt: str
    approval_digest: str | None = None
    reference_images: list[Path] | None = None
    reference_descriptions: list[str] | None = None
    judge_reference_frames: int = DEFAULT_JUDGE_REFERENCE_FRAMES
    judge_generated_frames: int = DEFAULT_JUDGE_GENERATED_FRAMES
    max_iterations: int = 5
    score_threshold: float = 0.7
    judge_max_tokens: int | None = None
    judge_temperature: float | None = None
    llm_timeout_seconds: float = 180.0
    chat_model: Any | None = None
    vlm_model: Any | None = None
    cancel_event: Any = None
    event_listener: Any = None
    verbose: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if not isinstance(self.user_prompt, str) or not self.user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string")
        self.user_prompt = self.user_prompt.strip()
        self.max_iterations = int(self.max_iterations)
        if not 1 <= self.max_iterations <= _MAX_ITERATIONS:
            raise ValueError(f"max_iterations must be between 1 and {_MAX_ITERATIONS}")
        threshold = float(self.score_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("score_threshold must be finite and between 0 and 1")
        self.score_threshold = threshold
        timeout = float(self.llm_timeout_seconds)
        if not math.isfinite(timeout):
            raise ValueError("llm_timeout_seconds must be finite")
        self.llm_timeout_seconds = timeout
        self.judge_reference_frames = validate_visual_frame_count(
            "judge_reference_frames", self.judge_reference_frames
        )
        self.judge_generated_frames = validate_visual_frame_count(
            "judge_generated_frames", self.judge_generated_frames
        )
        self.reference_images = [
            validate_reference_image_path(path, label="reference media")
            for path in self.reference_images or []
        ]
        if self.reference_descriptions is not None and len(
            self.reference_descriptions
        ) != len(self.reference_images):
            raise ValueError("reference_descriptions must match reference_images")
        if self.cancel_event is not None and not callable(
            getattr(self.cancel_event, "is_set", None)
        ):
            raise TypeError("cancel_event must provide is_set()")


def _model_identity(model: Any | None) -> dict[str, Any]:
    if model is None:
        return {"source": "default"}
    identity: dict[str, Any] = {
        "source": "injected",
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
    }
    model_name = getattr(model, "model", None) or getattr(model, "model_name", None)
    if isinstance(model_name, str) and model_name:
        identity["model"] = model_name
    return identity


def _request_payload(params: ExternalRefineInput) -> dict[str, Any]:
    image_descriptions = list(params.reference_descriptions or [])
    return {
        "user_prompt": params.user_prompt,
        "reference_media": {
            "images": [
                {
                    "path": str(path),
                    "description": (
                        image_descriptions[index]
                        if index < len(image_descriptions)
                        else ""
                    ),
                }
                for index, path in enumerate(params.reference_images or [])
            ],
        },
        "settings": {
            "max_iterations": params.max_iterations,
            "score_threshold": params.score_threshold,
            "judge_reference_frames": params.judge_reference_frames,
            "judge_generated_frames": params.judge_generated_frames,
            "judge_max_tokens": params.judge_max_tokens,
            "judge_temperature": params.judge_temperature,
            "llm_timeout_seconds": params.llm_timeout_seconds,
            "chat_model": _model_identity(params.chat_model),
            "vlm_model": _model_identity(params.vlm_model),
        },
    }


@dataclass
class ExternalRefineIteration:
    """One persisted optimize, evidence, judge, and optional refine cycle."""

    iteration: int
    output_dir: Path
    active_search: dict[str, dict[str, float]]
    objective: dict[str, Any]
    best_params: dict[str, float] = field(default_factory=dict)
    best_objective: float | None = None
    optimizer_loss: float | None = None
    n_trials: int = 0
    judge: dict[str, Any] | None = None
    refinement: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "output_dir": str(self.output_dir),
            "active_search": self.active_search,
            "objective": self.objective,
            "best_params": self.best_params,
            "best_objective": self.best_objective,
            "optimizer_loss": self.optimizer_loss,
            "n_trials": self.n_trials,
            "judge": self.judge,
            "refinement": self.refinement,
            "error": self.error,
        }


@dataclass
class ExternalRefineOutput(APIResult):
    """Qualification or terminal result of external-runtime refinement."""

    status: str = "failed"
    termination_reason: str = "error"
    output_dir: Path | None = None
    qualification_digest: str | None = None
    qualification_path: Path | None = None
    iterations: list[ExternalRefineIteration] = field(default_factory=list)
    final_dir: Path | None = None
    final_best_params: dict[str, float] = field(default_factory=dict)
    final_objective: float | None = None
    final_optimizer_loss: float | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    validated: bool = False


def _summary_payload(result: ExternalRefineOutput) -> dict[str, Any]:
    artifact_paths: dict[str, str] = {}
    if result.output_dir is not None:
        root = result.output_dir.resolve()
        for name, path in result.artifacts.items():
            resolved = Path(path).resolve()
            try:
                artifact_paths[name] = resolved.relative_to(root).as_posix()
            except ValueError:
                artifact_paths[name] = str(resolved)
    return {
        "schema_version": 1,
        "success": result.success,
        "status": result.status,
        "termination_reason": result.termination_reason,
        "validated": result.validated,
        "error": result.error,
        "qualification_digest": result.qualification_digest,
        "qualification_path": (
            str(result.qualification_path) if result.qualification_path else None
        ),
        "iterations": [iteration.to_dict() for iteration in result.iterations],
        "final_dir": str(result.final_dir) if result.final_dir else None,
        "final_best_params": result.final_best_params,
        "final_objective": result.final_objective,
        "final_optimizer_loss": result.final_optimizer_loss,
        "request": result.request,
        "artifacts": artifact_paths,
        "cancelled": result.cancelled,
    }


def _persist_summary(result: ExternalRefineOutput) -> ExternalRefineOutput:
    if result.output_dir is None:
        return result
    path = result.output_dir / "external_refine_summary.json"
    result.artifacts["summary"] = path
    _write_json(path, _summary_payload(result))
    return result


def _portable_selected_evidence(tune: ExternalTuneOutput) -> dict[str, Any]:
    selected = dict(tune.selected_evidence)
    for name in ("scored_recording", "published_recording"):
        descriptor = selected.get(name)
        if isinstance(descriptor, dict):
            selected[name] = {**descriptor, "path": BEST_RECORDING}
    return selected


def _portable_reference_path(source: Path, root: Path) -> str:
    root_resolved = root.resolve()
    absolute = Path(os.path.abspath(source))
    try:
        relative = absolute.relative_to(root_resolved)
    except ValueError:
        raise ValueError(
            "external refine judge reference escaped the refine output directory"
        ) from None
    if not relative.parts or relative.parts[0] != "reference_media":
        raise ValueError(
            "external refine judge reference is outside managed reference evidence"
        )
    candidate = root_resolved
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError("external refine judge reference must not use symlinks")
    if not absolute.is_file():
        raise ValueError("external refine judge reference is missing")
    return relative.as_posix()


def _copy_managed_reference(
    source: Path,
    *,
    root: Path,
    final_dir: Path,
) -> str:
    relative = _portable_reference_path(source, root)
    destination = final_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)
    return relative


def _portable_judge_payload(
    path: Path,
    iteration_dir: Path,
    *,
    root: Path | None = None,
    final_dir: Path | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("external refine terminal judge artifact is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("external refine terminal judge artifact must be an object")
    extra = payload.get("extra")
    visual = extra.get("visual_evidence") if isinstance(extra, dict) else None
    if not isinstance(visual, dict):
        return payload

    refine_root = iteration_dir.parent if root is None else root
    references = visual.get("reference_images")
    if isinstance(references, list):
        for item in references:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            source = Path(item["path"])
            item["path"] = (
                _portable_reference_path(source, refine_root)
                if final_dir is None
                else _copy_managed_reference(
                    source,
                    root=refine_root,
                    final_dir=final_dir,
                )
            )

    source_render_dir = (iteration_dir / "render").resolve()
    generated = visual.get("generated_images")
    if isinstance(generated, list):
        for item in generated:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            source = Path(item["path"]).resolve()
            try:
                relative = source.relative_to(source_render_dir)
            except ValueError:
                raise ValueError(
                    "external refine judge frame escaped the iteration render directory"
                ) from None
            if not source.is_file():
                raise ValueError("external refine judge frame is missing")
            item["path"] = (Path("render") / relative).as_posix()
    comparison = visual.get("comparison_image")
    if isinstance(comparison, str):
        expected_comparison = (iteration_dir / "comparison.png").resolve()
        if Path(comparison).resolve() != expected_comparison:
            raise ValueError(
                "external refine judge comparison escaped the iteration directory"
            )
        if not expected_comparison.is_file():
            raise ValueError("external refine judge comparison is missing")
        visual["comparison_image"] = "comparison.png"
    return payload


def _portable_request_payload(
    request: dict[str, Any],
    root: Path,
    *,
    final_dir: Path | None = None,
) -> dict[str, Any]:
    payload = copy.deepcopy(request)
    reference_media = payload.get("reference_media")
    if not isinstance(reference_media, dict):
        return payload
    for key, directory, prefix in (("images", "images", "reference_image"),):
        entries = reference_media.get(key)
        if not isinstance(entries, list):
            continue
        for index, item in enumerate(entries, 1):
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            suffix = Path(item["path"]).suffix.lower()
            relative = (
                Path("reference_media") / directory / f"{prefix}_{index:02d}{suffix}"
            )
            source = root / relative
            item["path"] = (
                _portable_reference_path(source, root)
                if final_dir is None
                else _copy_managed_reference(
                    source,
                    root=root,
                    final_dir=final_dir,
                )
            )
    return payload


def _final_tune_payload(
    *,
    spec: ExternalTuneSpec,
    tune: ExternalTuneOutput,
    history: list[dict[str, Any]],
    rendered_frames: list[str],
) -> dict[str, Any]:
    artifacts = {
        "run_spec": RUN_SPEC,
        "history": HISTORY,
        "best_params": BEST_PARAMS,
        "best_recording": BEST_RECORDING,
    }
    artifacts.update(
        {
            f"output.{name}": descriptor["path"]
            for name, descriptor in tune.published_outputs.items()
        }
    )
    return {
        "schema_version": 1,
        "mode": "external_runtime",
        "task": spec.task,
        "success": tune.success,
        "status": tune.status,
        "error": tune.error,
        "optimizer_used": tune.optimizer_used,
        "optimizer_loss": tune.best_score,
        "n_trials": tune.n_trials,
        "objective": {
            "name": spec.objective.name,
            "unit": spec.objective.unit,
            "direction": spec.objective.direction,
            "best_value": tune.best_objective,
        },
        "active_parameters": [parameter.name for parameter in spec.params],
        "best_params": tune.best_params,
        "qualification_digest": tune.qualification_digest,
        "started_at": tune.started_at,
        "completed_at": tune.completed_at,
        "history": history,
        "rendered_frames": rendered_frames,
        "render_error": tune.render_error,
        "selected_evidence": _portable_selected_evidence(tune),
        "published_outputs": tune.published_outputs,
        "artifacts": artifacts,
        "cancelled": tune.cancelled,
    }


def _build_final(
    *,
    final_dir: Path,
    iteration_dir: Path,
    spec: ExternalTuneSpec,
    tune: ExternalTuneOutput,
    request: dict[str, Any],
) -> dict[str, Path]:
    """Build a self-describing terminal bundle without raw trial internals."""

    if tune.best_objective is None or not math.isfinite(tune.best_score):
        raise ValueError("external refine winner has no finite objective result")
    final_dir.mkdir(parents=True)
    root = iteration_dir.parent
    artifacts: dict[str, Path] = {}
    recording_source = tune.artifacts.get("best_recording")
    recording_destination = final_dir / BEST_RECORDING
    if recording_source is None or not Path(recording_source).is_file():
        raise ValueError("external refine final artifact 'best_recording' is missing")
    shutil.copy2(recording_source, recording_destination)
    artifacts["best_recording"] = recording_destination

    required_iteration_files = {
        "run_spec": RUN_SPEC,
        "search": "search.json",
        "best_params": BEST_PARAMS,
    }
    for logical_name, filename in required_iteration_files.items():
        source = iteration_dir / filename
        if not source.is_file():
            raise ValueError(
                f"external refine terminal iteration is missing {filename}"
            )
        destination = final_dir / filename
        shutil.copy2(source, destination)
        artifacts[logical_name] = destination

    portable_history = [
        external_trial_payload(record, include_internal_artifacts=False)
        for record in tune.history
    ]
    history_destination = final_dir / HISTORY
    _write_jsonl(history_destination, portable_history)
    artifacts["history"] = history_destination

    judge_source = iteration_dir / "judge.json"
    if not judge_source.is_file():
        raise ValueError("external refine terminal iteration is missing judge.json")
    judge_destination = final_dir / "judge.json"
    _write_json(
        judge_destination,
        _portable_judge_payload(
            judge_source,
            iteration_dir,
            root=root,
            final_dir=final_dir,
        ),
    )
    artifacts["judge"] = judge_destination

    comparison_source = iteration_dir / "comparison.png"
    if comparison_source.is_file():
        comparison_destination = final_dir / "comparison.png"
        shutil.copy2(comparison_source, comparison_destination)
        artifacts["comparison"] = comparison_destination

    source_render_dir = iteration_dir / "render"
    final_rendered_frames: list[str] = []
    for source in tune.rendered_frames:
        resolved = Path(source).resolve()
        try:
            relative = resolved.relative_to(source_render_dir.resolve())
        except ValueError:
            raise ValueError(
                "external refine rendered frame escaped the iteration render directory"
            ) from None
        if not resolved.is_file() or resolved.suffix.lower() != ".png":
            raise ValueError("external refine rendered frame is not a PNG file")
        destination = final_dir / "render" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, destination)
        final_rendered_frames.append(destination.relative_to(final_dir).as_posix())

    outputs_source = iteration_dir / "outputs"
    if outputs_source.is_dir():
        shutil.copytree(outputs_source, final_dir / "outputs")

    tune_results_destination = final_dir / RESULTS
    _write_json(
        tune_results_destination,
        _final_tune_payload(
            spec=spec,
            tune=tune,
            history=portable_history,
            rendered_frames=final_rendered_frames,
        ),
    )
    artifacts["tune_results"] = tune_results_destination

    payloads = {
        "refine_request": _portable_request_payload(
            request,
            root,
            final_dir=final_dir,
        ),
        "objective": {
            "name": spec.objective.name,
            "unit": spec.objective.unit,
            "direction": spec.objective.direction,
            "value": tune.best_objective,
            "optimizer_loss": tune.best_score,
        },
        "result": {
            "source_iteration": iteration_dir.name,
            "best_params": tune.best_params,
            "objective": {
                "name": spec.objective.name,
                "unit": spec.objective.unit,
                "direction": spec.objective.direction,
                "value": tune.best_objective,
            },
            "optimizer_loss": tune.best_score,
            "selected_evidence": _portable_selected_evidence(tune),
            "published_outputs": tune.published_outputs,
        },
    }
    for name, payload in payloads.items():
        path = final_dir / f"{name}.json"
        _write_json(path, payload)
        artifacts[name] = path
    manifest = write_manifest(final_dir)
    artifacts["manifest"] = manifest
    return artifacts


def _publish_final(
    *,
    root: Path,
    iteration_dir: Path,
    spec: ExternalTuneSpec,
    tune: ExternalTuneOutput,
    request: dict[str, Any],
) -> tuple[Path, dict[str, Path]]:
    """Atomically publish the curated terminal bundle."""

    final_dir = root / "final"
    temporary_dir = root / ".final.tmp"
    shutil.rmtree(temporary_dir, ignore_errors=True)
    try:
        temporary_artifacts = _build_final(
            final_dir=temporary_dir,
            iteration_dir=iteration_dir,
            spec=spec,
            tune=tune,
            request=request,
        )
        shutil.rmtree(final_dir, ignore_errors=True)
        temporary_dir.replace(final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    artifacts = {
        name: final_dir / path.relative_to(temporary_dir)
        for name, path in temporary_artifacts.items()
    }
    return final_dir, artifacts


def _run_external_refine(params: ExternalRefineInput) -> ExternalRefineOutput:
    root = params.output_dir.expanduser().resolve()
    request_payload = _request_payload(params)
    spec = load_external_tune_spec(params.config)
    if spec.evidence is None:
        raise ValueError(
            "external refine requires selected RTX frame evidence and a rollout recording"
        )
    qualification_dir = root / "qualification"
    try:
        _validate_run_directory_layout(
            spec,
            output_dir=root,
            qualification_dir=qualification_dir,
        )
    except ValueError as exc:
        return ExternalRefineOutput(
            success=False,
            error=str(exc),
            status="invalid_output_layout",
            termination_reason="error",
            output_dir=root,
            request=request_payload,
        )
    root.mkdir(parents=True, exist_ok=True)

    if params.approval_digest is None:
        qualification = run_external_tune(
            ExternalTuneInput(
                config=spec,
                output_dir=qualification_dir,
                cancel_event=params.cancel_event,
                event_listener=params.event_listener,
            )
        )
        awaiting_approval = qualification.status == "awaiting_approval"
        qualification_cancelled = (
            qualification.cancelled or qualification.status == "cancelled"
        )
        result = ExternalRefineOutput(
            success=qualification.success,
            error=qualification.error,
            status=qualification.status,
            termination_reason=(
                "cancelled"
                if qualification_cancelled
                else (
                    "awaiting_approval" if awaiting_approval else "qualification_failed"
                )
            ),
            output_dir=root,
            qualification_digest=qualification.qualification_digest,
            qualification_path=qualification.qualification_path,
            artifacts=dict(qualification.artifacts),
            request=request_payload,
            cancelled=qualification_cancelled,
        )
        if awaiting_approval:
            _emit(
                params.event_listener,
                "external_refine.qualification.awaiting_approval",
                {"qualification_digest": qualification.qualification_digest},
            )
        return _persist_summary(result)

    if _cancelled(params.cancel_event):
        return _persist_summary(
            ExternalRefineOutput(
                success=False,
                error="external refinement cancelled",
                status="cancelled",
                termination_reason="cancelled",
                output_dir=root,
                request=request_payload,
                cancelled=True,
            )
        )

    fingerprint = build_runtime_fingerprint(spec)
    (
        _qualification,
        qualification_path,
        _qualification_digest,
        _qualification_artifacts,
    ) = _validate_approved_qualification(
        spec=spec,
        qualification_dir=qualification_dir,
        approval_digest=params.approval_digest,
        fingerprint=fingerprint,
    )

    for stale in root.iterdir():
        if stale.is_dir() and (
            stale.name == "final"
            or (stale.name.startswith("iter_") and stale.name[5:].isdigit())
        ):
            shutil.rmtree(stale)

    reference_evidence = prepare_reference_media(
        reference_images=params.reference_images,
        reference_descriptions=params.reference_descriptions,
        output_dir=root,
    )
    pinned_params = dict(spec.qualification.nominal_params)
    iterations: list[ExternalRefineIteration] = []
    prior_history: list[dict[str, Any]] = []
    final_dir: Path | None = None
    final_artifacts: dict[str, Path] = {}
    final_objective: float | None = None
    final_optimizer_loss: float | None = None
    refinement = RefinementLoop(
        initial_state=spec,
        max_iterations=params.max_iterations,
    )

    _emit(
        params.event_listener,
        "external_refine.started",
        {
            "task": spec.task,
            "max_iterations": params.max_iterations,
            "score_threshold": params.score_threshold,
        },
    )
    while (refinement_iteration := refinement.begin_iteration()) is not None:
        iteration = refinement_iteration.iteration
        current_spec = refinement_iteration.state
        if _cancelled(params.cancel_event):
            refinement.stop("cancelled")
            break

        replica_seed = (
            spec.optimizer.seed
            if spec.optimizer.replica_seed is None
            else spec.optimizer.replica_seed
        )
        iteration_optimizer = replace(
            current_spec.optimizer,
            seed=spec.optimizer.seed + iteration - 1,
            replica_seed=replica_seed,
        )
        iteration_spec = replace(
            current_spec,
            optimizer=iteration_optimizer,
        )
        iter_dir = root / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        _write_json(iter_dir / "search.json", _search_payload(iteration_spec))
        _emit(
            params.event_listener,
            "external_refine.iteration.started",
            {"iteration": iteration},
        )
        tune = run_external_tune(
            ExternalTuneInput(
                config=iteration_spec,
                output_dir=iter_dir,
                approval_digest=params.approval_digest,
                qualification_dir=qualification_dir,
                fixed_params=pinned_params,
                render_winning_trial=True,
                cancel_event=params.cancel_event,
                event_listener=params.event_listener,
            )
        )
        record = ExternalRefineIteration(
            iteration=iteration,
            output_dir=iter_dir,
            active_search=_search_payload(iteration_spec),
            objective={
                "name": iteration_spec.objective.name,
                "unit": iteration_spec.objective.unit,
                "direction": iteration_spec.objective.direction,
            },
            best_params=dict(tune.best_params),
            best_objective=tune.best_objective,
            optimizer_loss=(
                float(tune.best_score) if math.isfinite(tune.best_score) else None
            ),
            n_trials=tune.n_trials,
        )
        iterations.append(record)
        if not tune.success or tune.status != "completed":
            record.error = tune.error or f"external tune ended with {tune.status}"
            refinement.stop("cancelled" if tune.cancelled else "error")
            break
        pinned_params = dict(tune.best_params)
        frames = [Path(path) for path in tune.rendered_frames if Path(path).is_file()]
        generated_error = tune.render_error
        if not frames and generated_error is None:
            generated_error = "renderer produced no frames"
        visual_evidence: JudgeVisualEvidence = reference_evidence.with_generated_images(
            frames, generated_error=generated_error
        )
        comparison, comparison_error = write_comparison_contact_sheet(
            visual_evidence,
            iter_dir / "comparison.png",
            max_reference_images=params.judge_reference_frames,
            max_generated_images=params.judge_generated_frames,
        )
        visual_evidence = visual_evidence.with_comparison_image(
            comparison, comparison_error=comparison_error
        )
        try:
            judge = _run_with_timeout(
                run_external_judge,
                spec=iteration_spec,
                output=tune,
                user_prompt=params.user_prompt,
                vlm_model=params.vlm_model,
                visual_evidence=visual_evidence,
                score_threshold=params.score_threshold,
                iteration=iteration,
                prior_refine_history=prior_history,
                judge_max_tokens=params.judge_max_tokens,
                judge_temperature=params.judge_temperature,
                judge_reference_frames=params.judge_reference_frames,
                judge_generated_frames=params.judge_generated_frames,
                timeout_seconds=params.llm_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed LLM boundary
            record.error = f"external judge failed: {exc}"
            refinement.stop("error")
            break
        if _cancelled(params.cancel_event):
            record.error = "external refinement cancelled after judge"
            refinement.stop("cancelled")
            break
        record.judge = judge.to_dict()
        _write_json(iter_dir / "judge.json", record.judge)
        _emit(
            params.event_listener,
            "external_refine.iteration.judged",
            {
                "iteration": iteration,
                "decision": judge.decision,
                "judge_score": judge.score,
            },
        )
        if judge.llm_unavailable:
            record.error = judge.llm_critique
            refinement.stop("error")
            break
        terminal_decision = judge.decision == "approve" or refinement_iteration.is_last
        if not terminal_decision:
            try:
                decision = _run_with_timeout(
                    run_external_refiner,
                    spec=iteration_spec,
                    judge_result=judge,
                    user_prompt=params.user_prompt,
                    chat_model=params.chat_model,
                    iteration=iteration,
                    history_summary=summarize_external_trials(tune.history),
                    prior_refine_history=prior_history,
                    timeout_seconds=params.llm_timeout_seconds,
                )
            except (ExternalRefineError, TimeoutError) as exc:
                record.error = str(exc)
                refinement.stop("error")
                break
            record.refinement = decision.to_dict()
            _write_json(iter_dir / "refine.json", record.refinement)
            prior_history.append(
                {
                    "iteration": iteration,
                    "best_objective": record.best_objective,
                    "optimizer_loss": record.optimizer_loss,
                    "best_params": record.best_params,
                    "judge_score": judge.score,
                    "judge_decision": judge.decision,
                    "refine_reasoning": decision.reasoning,
                }
            )
            refinement.continue_with(
                replace(
                    current_spec,
                    params=decision.params,
                )
            )
            continue

        if tune.best_objective is None:
            record.error = "selected candidate has no objective value"
            refinement.stop("error")
            break
        published_dir, published_artifacts = _publish_final(
            root=root,
            iteration_dir=iter_dir,
            spec=iteration_spec,
            tune=tune,
            request=request_payload,
        )
        if _cancelled(params.cancel_event):
            record.error = "external refinement cancelled during final publication"
            refinement.stop("cancelled")
            shutil.rmtree(published_dir, ignore_errors=True)
            break
        final_objective = float(tune.best_objective)
        final_optimizer_loss = float(tune.best_score)
        final_dir = published_dir
        final_artifacts = published_artifacts
        if judge.decision == "approve":
            refinement.approve()
        else:
            refinement.continue_with(iteration_spec)
        break

    termination_reason = refinement.termination_reason
    if termination_reason is None:
        raise RuntimeError("external refinement ended without a terminal decision")
    cancelled = termination_reason == "cancelled"
    success = termination_reason in {"approved", "max_iterations"}
    validated = termination_reason == "approved"
    error = None
    if not success:
        error = next(
            (iteration.error for iteration in reversed(iterations) if iteration.error),
            "external refinement cancelled"
            if cancelled
            else "external refinement failed",
        )
    result = ExternalRefineOutput(
        success=success,
        error=error,
        status=("cancelled" if cancelled else ("completed" if success else "failed")),
        termination_reason=termination_reason,
        validated=validated,
        output_dir=root,
        qualification_digest=params.approval_digest,
        qualification_path=qualification_path,
        iterations=iterations,
        final_dir=final_dir,
        final_best_params=pinned_params if final_dir is not None else {},
        final_objective=final_objective,
        final_optimizer_loss=final_optimizer_loss,
        artifacts={**final_artifacts},
        request=request_payload,
        cancelled=cancelled,
    )
    _emit(
        params.event_listener,
        "external_refine.completed",
        {
            "termination_reason": termination_reason,
            "iteration_count": len(iterations),
            "success": success,
            "validated": validated,
        },
    )
    return _persist_summary(result)


def run_external_refine(params: ExternalRefineInput) -> ExternalRefineOutput:
    """Run external refinement and return failures as structured results."""

    try:
        return _run_external_refine(params)
    except Exception as exc:  # noqa: BLE001 - public API result boundary
        result = ExternalRefineOutput(
            success=False,
            error=str(exc),
            status="failed",
            termination_reason="error",
            output_dir=params.output_dir.expanduser().resolve(),
            request=_request_payload(params),
        )
        return _persist_summary(result)


async def arun_external_refine(params: ExternalRefineInput) -> ExternalRefineOutput:
    """Run external refinement without blocking the caller's event loop."""

    return await asyncio.to_thread(run_external_refine, params)


__all__ = [
    "ExternalRefineInput",
    "ExternalRefineIteration",
    "ExternalRefineOutput",
    "arun_external_refine",
    "run_external_refine",
]
