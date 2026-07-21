# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Value-free diagnostics for logs, telemetry, and durable artifacts."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import StrEnum

DIAGNOSTIC_SCHEMA = "world-understanding-durable-diagnostic-v1"


class FailurePhase(StrEnum):
    """Stable failure phases shared by storage and pipeline surfaces."""

    LOCAL_PUBLICATION = "local_publication"
    SYNC_UPLOAD = "sync_upload"
    PERSISTENCE_VERIFICATION = "persistence_verification"
    ROLLBACK = "rollback"
    PIPELINE_EXECUTION = "pipeline_execution"


@dataclass(frozen=True)
class DurableDiagnostic:
    """A durable failure record containing no runtime or exception values."""

    schema: str
    code: str
    phase: str
    retryable: bool

    def to_dict(self) -> dict[str, str | bool]:
        """Return the JSON-compatible diagnostic envelope."""
        return asdict(self)


def durable_diagnostic(
    code: str,
    *,
    phase: FailurePhase,
    retryable: bool,
) -> DurableDiagnostic:
    """Build one bounded, code-defined diagnostic without exception text."""
    if not code or len(code) > 96 or not code.replace("_", "").isalnum():
        raise ValueError("Diagnostic code must be a bounded identifier")
    return DurableDiagnostic(
        schema=DIAGNOSTIC_SCHEMA,
        code=code,
        phase=phase.value,
        retryable=retryable,
    )


def log_durable_failure(
    logger: logging.Logger,
    code: str,
    *,
    phase: FailurePhase,
    retryable: bool,
) -> None:
    """Emit the same value-free fields used by durable diagnostic artifacts."""
    diagnostic = durable_diagnostic(code, phase=phase, retryable=retryable)
    logger.error(
        "durable_failure schema=%s code=%s phase=%s retryable=%s",
        diagnostic.schema,
        diagnostic.code,
        diagnostic.phase,
        diagnostic.retryable,
    )
