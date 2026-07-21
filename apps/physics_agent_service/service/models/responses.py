# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Response models for Physics Agent Service API."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Standard FastAPI error payload returned by authorization failures."""

    detail: str = Field(description="Human-readable error detail")


S3_INPUT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    403: {
        "description": (
            "Client S3 URI rejected by the configured bucket allowlist, "
            "or S3 access denied."
        ),
        "model": ErrorDetail,
    }
}

PREDICT_INPUT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    403: {
        "description": (
            "dataset_path resolves outside allowed roots; client S3 URI rejected "
            "by the configured bucket allowlist; or S3 access denied."
        ),
        "model": ErrorDetail,
    }
}


class StepProgress(BaseModel):
    """Progress information for a single step."""

    current: int = Field(description="Current progress count")
    total: int = Field(description="Total items to process")
    percent: int = Field(description="Percentage complete (0-100)")
    message: str = Field(description="Human-readable progress message")


class CurrentStepInfo(BaseModel):
    """Information about the currently executing step."""

    name: str = Field(description="Step internal name")
    display_name: str = Field(description="Human-readable step name")
    started_at: str = Field(description="ISO timestamp when step started")
    progress: StepProgress
    elapsed_seconds: int = Field(description="Seconds since step started")


class CompletedStepInfo(BaseModel):
    """Information about a completed step."""

    name: str = Field(description="Step internal name")
    display_name: str = Field(description="Human-readable step name")
    started_at: str = Field(description="ISO timestamp when step started")
    completed_at: str = Field(description="ISO timestamp when step completed")
    duration_seconds: int = Field(description="Step duration in seconds")
    stats: dict[str, Any] = Field(
        default_factory=dict, description="Step-specific statistics"
    )


class OverallProgress(BaseModel):
    """Overall pipeline progress."""

    current_step: int = Field(description="Current step number (1-indexed)")
    total_steps: int = Field(description="Total number of steps")
    percent: int = Field(description="Overall percentage complete (0-100)")
    estimated_remaining_seconds: int | None = Field(
        default=None, description="Estimated seconds until completion"
    )


class PipelineStatus(BaseModel):
    """Enhanced pipeline execution status with progress."""

    session_id: str
    status: str = Field(
        description="Current status: pending, running, completed, failed, cancelled, cancelling"
    )
    current_step: CurrentStepInfo | None = None
    completed_steps: list[CompletedStepInfo] = Field(default_factory=list)
    overall_progress: OverallProgress
    preview_images: list[str] = Field(
        default_factory=list, description="URLs to preview images"
    )
    can_cancel: bool = Field(description="Whether pipeline can be cancelled")
    elapsed_seconds: int = Field(description="Total elapsed time in seconds")
    created_at: str = Field(description="ISO timestamp when session created")
    updated_at: str = Field(description="ISO timestamp of last update")


class PipelineResults(BaseModel):
    """Pipeline execution results."""

    session_id: str
    status: str
    stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution statistics",
        examples=[
            {
                "prims_processed": 142,
                "images_generated": 284,
                "predictions_made": 142,
            }
        ],
    )
    download_urls: dict[str, str] = Field(
        default_factory=dict,
        description="URLs to download artifacts",
        examples=[
            {
                "predictions": "/artifacts/abc123/predictions",
                "report": "/artifacts/abc123/report",
            }
        ],
    )
    duration_seconds: int = Field(description="Total pipeline duration in seconds")
    completed_at: str = Field(description="ISO timestamp when completed")


class PipelineError(BaseModel):
    """Pipeline error response."""

    session_id: str
    status: str = "failed"
    error_message: str = Field(description="Stable value-free diagnostic code")
    failed_step: str = Field(description="Step that failed")
    completed_steps: list[str] = Field(
        default_factory=list, description="Steps completed before failure"
    )
    partial_results: dict[str, Any] | None = Field(
        default=None, description="Partial results if available"
    )


class SessionCreated(BaseModel):
    """Response when session is created."""

    session_id: str
    status: str = "pending"
    message: str = "Pipeline queued for execution"
    estimated_duration_minutes: int | None = Field(
        default=None, description="Estimated completion time"
    )


class PredictResults(BaseModel):
    """Predict-only execution results.

    Returned by ``GET /predict/{session_id}/results`` when the predict job has
    completed. Mirrors the prediction-relevant subset of ``PipelineResults`` but
    additionally surfaces:

    * ``mode`` — which predict mode actually ran (``dataset_only`` or
      ``full_predict``). Lets clients confirm the route's auto-detection picked
      the mode they intended.
    * ``steps_run`` — the canonical step list /predict drove for this job.
    * ``predictions_count`` / ``failed_count`` / ``token_stats`` —
      hoisted out of ``stats`` to match the source-of-truth Python API
      (`PredictOutput`).
    """

    session_id: str
    status: str
    mode: str = Field(
        description="Detected predict mode: 'dataset_only' or 'full_predict'.",
    )
    steps_run: list[str] = Field(
        default_factory=list,
        description="Pipeline steps the predict route actually drove.",
    )
    stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution statistics",
    )
    predictions_count: int = Field(
        default=0,
        description="Number of predictions produced (from PredictOutput).",
    )
    failed_count: int = Field(
        default=0,
        description="Number of failed predictions (from PredictOutput).",
    )
    predictions_path: str | None = Field(
        default=None,
        description="Server-side path to predictions.jsonl on disk.",
    )
    token_stats: dict[str, Any] = Field(
        default_factory=dict,
        description="VLM token usage statistics when available.",
    )
    download_urls: dict[str, str] = Field(
        default_factory=dict,
        description="URLs to download artifacts (predictions, report, dataset).",
    )
    duration_seconds: int = Field(description="Total predict duration in seconds")
    completed_at: str = Field(description="ISO timestamp when completed")


class TuneStatus(BaseModel):
    """Status response for a tune session.

    Tune sessions have a much smaller progress surface than the full
    pipeline (a single iterative loop, no multi-step bookkeeping), so this
    response is intentionally flatter than ``PipelineStatus``.
    """

    session_id: str
    status: str = Field(
        description="pending, running, completed, failed, cancelled, cancelling"
    )
    n_trials: int = Field(default=0, description="Trials completed so far")
    max_trials: int = Field(default=0, description="Configured trial budget")
    best_score: float | None = Field(
        default=None, description="Best score so far (lower is better)"
    )
    best_params: dict[str, float] | None = Field(
        default=None, description="Best parameter set so far"
    )
    elapsed_seconds: int = Field(description="Total elapsed time in seconds")
    can_cancel: bool = Field(description="Whether tune can be cancelled")
    error_message: str | None = Field(
        default=None,
        description="Failure reason for terminal failed tune sessions.",
    )
    created_at: str
    updated_at: str


class TuneResults(BaseModel):
    """Results response for a completed tune session."""

    session_id: str
    status: str
    best_params: dict[str, float] = Field(default_factory=dict)
    # Round 12 (CX P2#2): nullable so a cancelled-before-first-trial run
    # can serialise — the previous ``float`` schema with a ``nan`` /
    # ``inf`` sentinel made Starlette's JSON encoder reject the payload
    # and the results endpoint returned 500 instead of the cancelled
    # state.
    best_score: float | None = Field(
        default=None,
        description=(
            "Best score (lower is better); null when the run was "
            "cancelled before any trial completed or when no successful "
            "trial was recorded."
        ),
    )
    n_trials: int
    optimizer_used: str = Field(description="Resolved optimizer name (auto→botorch)")
    engine_used: str
    download_urls: dict[str, str] = Field(
        default_factory=dict,
        description="URLs for tune artifacts that were produced and published.",
    )
    duration_seconds: int
    completed_at: str
    error_message: str | None = Field(
        default=None,
        description=(
            "Failure reason for terminal failed sessions that still expose "
            "partial tune results and artifact URLs."
        ),
    )


class RefineStatus(BaseModel):
    """Status response for an iterative refine session."""

    session_id: str
    status: str = Field(
        description="pending, running, completed, failed, cancelled, cancelling"
    )
    iteration: int = Field(default=0, description="Current or final iteration number")
    max_iterations: int = Field(default=0, description="Configured iteration cap")
    n_trials: int = Field(
        default=0, description="Trials completed in current iteration"
    )
    max_trials: int = Field(
        default=0, description="Configured trial budget per iteration"
    )
    best_score: float | None = Field(
        default=None, description="Best score observed in current/final iteration"
    )
    best_params: dict[str, float] | None = Field(
        default=None,
        description="Best parameter set observed in current/final iteration",
    )
    judge_score: float | None = Field(
        default=None, description="Latest judge score, when available"
    )
    termination_reason: str | None = Field(
        default=None,
        description="approved, max_iterations, cancelled, error, or null while running",
    )
    elapsed_seconds: int = Field(description="Total elapsed time in seconds")
    can_cancel: bool = Field(description="Whether refine can be cancelled")
    error_message: str | None = Field(
        default=None,
        description="Failure reason for terminal failed refine sessions.",
    )
    created_at: str
    updated_at: str


class RefineResults(BaseModel):
    """Results response for a terminal refine session."""

    session_id: str
    status: str
    termination_reason: str
    iteration_count: int
    final_iteration: int
    final_judge_score: float | None = None
    iterations: list[dict[str, Any]] = Field(default_factory=list)
    download_urls: dict[str, str] = Field(
        default_factory=dict,
        description="URLs for refine artifacts that were produced and published.",
    )
    duration_seconds: int
    completed_at: str
    error_message: str | None = None
