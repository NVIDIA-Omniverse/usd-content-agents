# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker protocol and result contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..models import RepairOperation


class WorkerResult(BaseModel):
    """Typed outcome from a single deterministic worker invocation."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "unavailable", "failed"]
    output_path: str | None = None
    output_sha256: str | None = None
    changed: bool = False
    operations: list[str] = Field(default_factory=list)
    deleted_entities: list[str] = Field(default_factory=list)
    merged_entities: list[str] = Field(default_factory=list)
    verified_issue_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RepairWorker(Protocol):
    """Bounded worker invoked by the orchestrator."""

    name: str
    operations: frozenset[str]

    def available(self) -> tuple[bool, str | None]: ...

    def execute(
        self,
        *,
        source: Path,
        output: Path,
        operation: RepairOperation,
    ) -> WorkerResult: ...
