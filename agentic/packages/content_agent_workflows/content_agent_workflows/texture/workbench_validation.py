# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workbench validation contract and deterministic mock VQA flow."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from .models import (
    TextureUnitArtifact,
    TextureValidationFinding,
    TextureValidationResult,
    TextureValidationStatus,
)


class TextureWorkbenchValidator(Protocol):
    """Workbench render/VQA boundary used by both workflow launch modes."""

    def validate(
        self,
        *,
        output_asset_path: str,
        unit_artifacts: Mapping[str, TextureUnitArtifact],
        unit_ids: tuple[str, ...],
        iteration: int,
        output_dir: Path,
    ) -> TextureValidationResult:
        """Validate exactly ``unit_ids`` and identify per-unit failures."""


class MockWorkbenchValidationCall(BaseModel):
    """Recorded validation request for workflow assertions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    iteration: int
    unit_ids: tuple[str, ...]


class MockWorkbenchTextureValidator:
    """Mock Workbench render/VQA adapter with a per-pass failure schedule."""

    def __init__(
        self,
        failure_schedule: Sequence[Sequence[str]] = (),
    ) -> None:
        self._failure_schedule = tuple(
            tuple(failed_ids) for failed_ids in failure_schedule
        )
        self.calls: list[MockWorkbenchValidationCall] = []

    def validate(
        self,
        *,
        output_asset_path: str,
        unit_artifacts: Mapping[str, TextureUnitArtifact],
        unit_ids: tuple[str, ...],
        iteration: int,
        output_dir: Path,
    ) -> TextureValidationResult:
        if set(unit_ids) - set(unit_artifacts):
            raise ValueError("Workbench validation requires an artifact for every unit")
        call_index = len(self.calls)
        scheduled_failures = (
            self._failure_schedule[call_index]
            if call_index < len(self._failure_schedule)
            else ()
        )
        unknown_failures = set(scheduled_failures) - set(unit_ids)
        if unknown_failures:
            raise ValueError(
                "mock Workbench failures must be within the evaluated unit IDs: "
                f"{unknown_failures}"
            )
        self.calls.append(
            MockWorkbenchValidationCall(iteration=iteration, unit_ids=unit_ids)
        )

        findings: list[TextureValidationFinding] = []
        for unit_id in unit_ids:
            evidence_path = (
                output_dir
                / "workbench_validation"
                / f"iteration-{iteration}"
                / f"{unit_id}.json"
            )
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            status: TextureValidationStatus = (
                "fail" if unit_id in scheduled_failures else "pass"
            )
            evidence_path.write_text(
                json.dumps(
                    {
                        "mock": True,
                        "output_asset_path": output_asset_path,
                        "status": status,
                        "unit_id": unit_id,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            findings.append(
                TextureValidationFinding(
                    unit_id=unit_id,
                    status=status,
                    summary=(
                        "Mock Workbench VQA identified a unit-specific defect."
                        if status == "fail"
                        else "Mock Workbench VQA accepted the unit artifact."
                    ),
                    evidence_artifact_paths=(str(evidence_path.resolve()),),
                )
            )

        return TextureValidationResult(
            iteration=iteration,
            evaluated_unit_ids=unit_ids,
            findings=tuple(findings),
            output_asset_path=output_asset_path,
        )
