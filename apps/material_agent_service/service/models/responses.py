# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Response models for Material Agent Service API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """Standard FastAPI error payload returned by input failures."""

    detail: str = Field(description="Human-readable error detail")


S3_INPUT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": (
            "Missing or invalid input source, including an unsupported USD extension."
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
    404: {
        "description": "Client-supplied S3 object was not found.",
        "model": ErrorDetail,
    },
    413: {
        "description": "S3 object exceeds the configured upload size limit.",
        "model": ErrorDetail,
    },
    502: {
        "description": "Failed to download the client-supplied S3 object.",
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


class MaterialCoverage(BaseModel):
    """Prim-level prediction and material-binding readiness."""

    schema_version: str = "1.0"
    policy: Literal["strict", "allow_partial"]
    readiness_grade: Literal[
        "complete", "complete_with_fallback", "partial", "not_evaluated"
    ]
    target_count: int = Field(ge=0)
    prepared_count: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    usable_prediction_count: int = Field(ge=0)
    unknown_prediction_count: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    bound_count: int = Field(ge=0)
    unbound_count: int = Field(ge=0)
    prediction_coverage_ratio: float = Field(ge=0.0, le=1.0)
    binding_coverage_ratio: float = Field(ge=0.0, le=1.0)
    missing_prepared_prim_ids: list[str] = Field(default_factory=list)
    missing_prediction_prim_ids: list[str] = Field(default_factory=list)
    unknown_prim_ids: list[str] = Field(default_factory=list)
    fallback_prim_ids: list[str] = Field(default_factory=list)
    unbound_prim_ids: list[str] = Field(default_factory=list)
    extra_prediction_prim_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DiagnosticEvidenceNames(BaseModel):
    """Fixed filenames in a renderer-failure evidence bundle."""

    report: Literal["report.json"]
    samples: list[str] = Field(default_factory=list, max_length=4)


class PipelineErrorDiagnostic(BaseModel):
    """Stable, redacted terminal pipeline diagnostic."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    schema_version: str = Field(alias="schema")
    code: str = Field(min_length=1, max_length=96)
    phase: str
    retryable: bool
    failed_step: str | None = None
    renderer_backend: Literal["warp", "ovrtx", "remote", "mock", "unknown"] | None = (
        None
    )
    checked_count: int | None = Field(default=None, ge=1, le=1_000_000)
    blank_count: int | None = Field(default=None, ge=1, le=1_000_000)
    blank_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    render_modes: list[str] = Field(default_factory=list, max_length=6)
    samples: list[dict[str, Any]] = Field(default_factory=list, max_length=4)
    evidence: DiagnosticEvidenceNames | None = None


class FailureEvidenceFile(BaseModel):
    """One downloadable renderer-failure evidence file."""

    name: str
    url: str


class FailureEvidence(BaseModel):
    """Evidence retained for the lifetime of a failed session."""

    report: FailureEvidenceFile
    samples: list[FailureEvidenceFile] = Field(default_factory=list, max_length=4)
    retention: Literal["until_session_expiry_or_deletion"]


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
    coverage: MaterialCoverage | None = Field(
        default=None, description="Material prediction and binding readiness"
    )
    error: str | None = Field(
        default=None, description="Stable terminal error code for failed sessions"
    )
    failed_step: str | None = None
    error_diagnostic: PipelineErrorDiagnostic | None = None
    failure_evidence: FailureEvidence | None = None


class StageTimings(BaseModel):
    """Detailed timing breakdown for pipeline stages."""

    preparation_seconds: float = Field(description="USD loading and setup time")
    rendering_total_seconds: float = Field(description="Total rendering time")
    rendering_per_prim_seconds: float = Field(description="Average time per prim")
    prediction_total_seconds: float = Field(description="Total VLM prediction time")
    prediction_per_prim_seconds: float = Field(
        description="Average prediction time per prim"
    )
    apply_seconds: float = Field(description="Material application time")
    total_seconds: float = Field(description="Total pipeline duration")


class PipelineResults(BaseModel):
    """Pipeline execution results."""

    session_id: str
    status: str
    stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution statistics",
        examples=[
            {
                "original_prim_count": 250,
                "prims_processed": 142,
                "images_generated": 284,
                "predictions_made": 95,
                "materials_applied": 5,
            }
        ],
    )
    timings: StageTimings | None = Field(
        default=None, description="Detailed timing breakdown by stage"
    )
    download_urls: dict[str, str] = Field(
        default_factory=dict,
        description="URLs to download artifacts",
        examples=[
            {
                "output_usd": "/artifacts/abc123/output",
                "predictions": "/artifacts/abc123/predictions",
            }
        ],
    )
    duration_seconds: int = Field(description="Total pipeline duration in seconds")
    completed_at: str = Field(description="ISO timestamp when completed")
    coverage: MaterialCoverage | None = Field(
        default=None,
        description="Material prediction and binding readiness",
    )


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
    download_urls: dict[str, str] = Field(
        default_factory=dict,
        description="URLs to download artifacts preserved before failure",
        examples=[
            {
                "output_usd": "/artifacts/abc123/output",
                "predictions": "/artifacts/abc123/predictions",
            }
        ],
    )
    coverage: MaterialCoverage | None = Field(
        default=None, description="Material prediction and binding readiness"
    )
    error_diagnostic: PipelineErrorDiagnostic | None = None
    failure_evidence: FailureEvidence | None = None


class SessionCreated(BaseModel):
    """Response when session is created."""

    session_id: str
    status: str = "pending"
    message: str = "Pipeline queued for execution"
    estimated_duration_minutes: int | None = Field(
        default=None, description="Estimated completion time"
    )


class PreviewImage(BaseModel):
    """Preview image information."""

    name: str = Field(description="Image filename")
    url: str = Field(description="URL to download image")
    prim_path: str | None = Field(default=None, description="USD prim path")
    created_at: str = Field(description="ISO timestamp when created")


class PreviewList(BaseModel):
    """List of available preview images."""

    session_id: str
    previews: list[PreviewImage]
    total: int = Field(description="Total number of previews")
