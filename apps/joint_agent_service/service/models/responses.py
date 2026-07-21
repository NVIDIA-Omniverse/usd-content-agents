# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Response models for Joint Agent Service API."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Standard FastAPI error payload returned by input failures."""

    detail: str = Field(description="Human-readable error detail")


S3_INPUT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": (
            "Missing, conflicting, or invalid input source, including an "
            "unsupported USD extension."
        ),
        "model": ErrorDetail,
    },
    403: {
        "description": (
            "Client S3 URI rejected by the configured bucket allowlist, "
            "or S3 access denied."
        ),
        "model": ErrorDetail,
    },
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
    error_message: str = Field(description="Error description")
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


class PipelineRunCreated(SessionCreated):
    """Response when one exact pipeline run has been accepted."""

    run_id: str = Field(
        min_length=32,
        max_length=32,
        pattern=r"^[0-9a-f]{32}$",
        description="Generation token required to cancel this exact run",
    )


class PipelineCancellationAccepted(BaseModel):
    """Response after cancellation is accepted for one exact run."""

    session_id: str
    run_id: str = Field(
        min_length=32,
        max_length=32,
        pattern=r"^[0-9a-f]{32}$",
        description="Generation token whose cancellation was accepted",
    )
    status: str = Field(description="Cancellation state")
    message: str = Field(description="Human-readable cancellation result")
