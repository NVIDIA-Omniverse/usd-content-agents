# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared USD validation adapter for Geometry workflow outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from world_understanding.functions.graphics.validate_usd import validate_usd

from content_agent_workflows.common.artifacts import atomic_write_json

GEOMETRY_USD_VALIDATION_SCHEMA_VERSION = (
    "content-agent-workflows.geometry-usd-validation.v1"
)
GEOMETRY_USD_VALIDATION_CATEGORIES = [
    "Basic",
    "Geometry",
    "Layer",
    "Layout",
    "Other",
]
GEOMETRY_NONBLOCKING_USD_RULES = {"UsdAsciiPerformanceChecker"}
_TEXTURE_EXTENSIONS = (
    ".bmp",
    ".dds",
    ".exr",
    ".hdr",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".tx",
)


def _geometry_policy_downgrade(issue: dict[str, Any]) -> str | None:
    rule = str(issue.get("rule") or "")
    if rule in GEOMETRY_NONBLOCKING_USD_RULES:
        return (
            "Storage-format performance does not invalidate geometry; "
            "the original severity remains in upstream_result."
        )
    message = str(issue.get("message") or "").lower()
    if rule == "MissingReferenceChecker" and any(
        extension in message for extension in _TEXTURE_EXTENSIONS
    ):
        return (
            "A missing image texture is material-owned and does not invalidate "
            "geometry handoff; the unresolved dependency remains visible to downstream agents."
        )
    return None


class GeometryUsdValidationReport(BaseModel):
    """Normalized shared USD validation result for geometry-owned categories."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = GEOMETRY_USD_VALIDATION_SCHEMA_VERSION
    producer: str = "world_understanding.functions.graphics.validate_usd"
    status: Literal["pass", "fail", "unavailable"]
    asset_path: str
    categories: list[str]
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    upstream_result: dict[str, Any] = Field(default_factory=dict)
    policy_downgrades: list[dict[str, str]] = Field(default_factory=list)
    report_path: str | None = None


def run_geometry_usd_validation(
    asset_path: Path, output_dir: Path
) -> GeometryUsdValidationReport:
    """Run shared structural and geometry-category USD validation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "usd_validation_report.json"
    try:
        upstream = validate_usd(
            asset_path,
            categories=list(GEOMETRY_USD_VALIDATION_CATEGORIES),
        )
        if upstream.get("status") != "success":
            failures = [str(upstream.get("error") or "USD validation failed.")]
            status: Literal["pass", "fail", "unavailable"] = "fail"
            warnings: list[str] = []
            policy_downgrades: list[dict[str, str]] = []
        else:
            downgrade_reasons = {
                id(issue): _geometry_policy_downgrade(issue)
                for issue in upstream.get("issues") or []
            }
            policy_downgrades = [
                {
                    "rule": str(issue.get("rule") or ""),
                    "original_severity": str(issue.get("severity") or ""),
                    "geometry_severity": "warning",
                    "reason": str(downgrade_reasons[id(issue)]),
                }
                for issue in upstream.get("issues") or []
                if downgrade_reasons[id(issue)] is not None
            ]
            failures = [
                str(issue.get("message") or issue.get("rule") or "USD issue")
                for issue in upstream.get("issues") or []
                if str(issue.get("severity") or "").lower() in {"failure", "error"}
                and downgrade_reasons[id(issue)] is None
            ]
            warnings = [
                str(issue.get("message") or issue.get("rule") or "USD warning")
                for issue in upstream.get("issues") or []
                if str(issue.get("severity") or "").lower() == "warning"
                or downgrade_reasons[id(issue)] is not None
            ]
            status = "fail" if failures else "pass"
        report = GeometryUsdValidationReport(
            status=status,
            asset_path=str(asset_path),
            categories=list(GEOMETRY_USD_VALIDATION_CATEGORIES),
            failures=failures,
            warnings=warnings,
            upstream_result=upstream,
            policy_downgrades=policy_downgrades,
            report_path=str(report_path),
        )
    except Exception as exc:
        report = GeometryUsdValidationReport(
            status="unavailable",
            asset_path=str(asset_path),
            categories=list(GEOMETRY_USD_VALIDATION_CATEGORIES),
            warnings=[
                f"Shared USD validation unavailable: {type(exc).__name__}: {exc}"
            ],
            report_path=str(report_path),
        )
    atomic_write_json(report_path, report)
    return report
