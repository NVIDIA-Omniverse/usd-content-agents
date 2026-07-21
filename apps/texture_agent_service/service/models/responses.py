# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Response models for Texture Agent Service API."""

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


class TexturePlanStatus(BaseModel):
    """Compact immutable-plan summary embedded in polling responses."""

    schema_version: str
    decision_state: str
    execution_allowed: bool
    counts: dict[str, int]
    limits: dict[str, int | None]
    plan_url: str


class PipelineStatus(BaseModel):
    """Pipeline execution status with progress."""

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
    error: str | None = Field(
        default=None,
        description="Sanitized top-level error message (failed runs)",
    )
    failed_step: str | None = Field(
        default=None, description="Step that failed (only set when status=failed)"
    )
    failed_step_stats: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Structured failed-step stats including any per-unit ``errors`` "
            "and ``textures_failed`` count. Mirrors the SSE FAILED event "
            "extra so polling clients see the same diagnostic detail."
        ),
    )
    texture_plan: TexturePlanStatus | None = Field(
        default=None,
        description=(
            "Bounded plan summary once planning has completed, including the "
            "decision, audited counts, and exact effective/hard caps."
        ),
    )


class PipelineResults(BaseModel):
    """Pipeline execution results."""

    session_id: str
    status: str
    stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution statistics",
        examples=[
            {
                "materials_found": 12,
                "textures_generated": 12,
                "output_usd_count": 1,
                "renders_count": 2,
            }
        ],
    )
    download_urls: dict[str, str] = Field(
        default_factory=dict,
        description="URLs to download artifacts",
        examples=[
            {
                "materials": "/artifacts/abc123/materials",
                "manifest": "/artifacts/abc123/manifest",
                "textures": "/artifacts/abc123/textures",
                "output": "/artifacts/abc123/output",
                "renders": "/artifacts/abc123/renders",
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
    failed_step_stats: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Stats from the failed step at the moment it raised, including "
            "any structured per-unit errors (e.g. ``errors`` and "
            "``textures_failed`` for texture-generation/blend steps). Lets "
            "REST consumers without an SSE subscription diagnose threshold-"
            "gated failures without grepping container logs."
        ),
    )


class SessionCreated(BaseModel):
    """Response when session is created."""

    session_id: str
    status: str = "pending"
    message: str = "Pipeline queued for execution"
    estimated_duration_minutes: int | None = Field(
        default=None, description="Estimated completion time"
    )
    plan_url: str | None = Field(
        default=None,
        description=(
            "Stable texture-plan endpoint for this session; returns 404 until "
            "planning has produced texture_plan.json."
        ),
    )


class SessionConfigSummary(BaseModel):
    """Sanitized subset of pipeline config echoed back on /sessions.

    Excludes absolute filesystem paths (the input USD lives at a server-
    internal location implied by the session id; surfacing it would leak
    the container's storage layout).
    """

    project_name: str | None = None
    original_filename: str | None = Field(
        default=None,
        description="Filename the client uploaded (None for S3 inputs)",
    )
    input_extension: str | None = Field(
        default=None, description="USD extension, e.g. '.usd' or '.usdz'"
    )
    has_usd_upload: bool | None = None
    s3_uri: str | None = Field(
        default=None,
        description="S3 URI when input was sourced from S3 (client-supplied)",
    )
    material_textures: dict[str, Any] | None = None


class SessionSummary(BaseModel):
    """One row in the ``GET /sessions`` listing."""

    session_id: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    elapsed_seconds: int = 0
    config: SessionConfigSummary = Field(default_factory=SessionConfigSummary)


class SessionListResponse(BaseModel):
    """Response payload for ``GET /sessions``."""

    sessions: list[SessionSummary] = Field(default_factory=list)
    total: int = 0


class SessionDetail(BaseModel):
    """Detail payload for ``GET /sessions/{session_id}``.

    Whitelists fields that are safe for public surface. Free-form strings
    in ``error`` and ``failed_step_stats`` are sanitized to redact NVCF
    function URLs and absolute session paths.
    """

    session_id: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    elapsed_seconds: int = 0
    ttl_expires_at: str | None = None
    config: SessionConfigSummary = Field(default_factory=SessionConfigSummary)

    current_step: CurrentStepInfo | None = None
    completed_steps: list[CompletedStepInfo] = Field(default_factory=list)
    overall_progress: OverallProgress | None = None
    preview_images: list[str] = Field(default_factory=list)
    can_cancel: bool = False

    error: str | None = Field(
        default=None, description="Sanitized top-level error message (failed runs)"
    )
    failed_step: str | None = None
    failed_step_stats: dict[str, Any] | None = None
    partial_results: dict[str, Any] | None = None

    results: dict[str, Any] | None = Field(
        default=None, description="Final stats (completed runs)"
    )
    duration_seconds: int | None = None
    completed_at: str | None = None
