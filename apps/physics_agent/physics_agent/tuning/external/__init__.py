# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted local external-runtime tuning for customer simulation repositories."""

from .config import ExternalTuneConfigError, load_external_tune_spec
from .refine import (
    ExternalRefineInput,
    ExternalRefineIteration,
    ExternalRefineOutput,
    arun_external_refine,
    run_external_refine,
)
from .runner import arun_external_tune, run_external_tune
from .session import derive_iteration_spec, validate_qualification_approval
from .types import (
    EvidenceSettings,
    ExternalRuntime,
    ExternalTuneInput,
    ExternalTuneOutput,
    ExternalTuneSpec,
    QualificationSettings,
    QualifiedParameter,
)

__all__ = [
    "EvidenceSettings",
    "ExternalRefineInput",
    "ExternalRefineIteration",
    "ExternalRefineOutput",
    "ExternalRuntime",
    "ExternalTuneConfigError",
    "ExternalTuneInput",
    "ExternalTuneOutput",
    "ExternalTuneSpec",
    "QualifiedParameter",
    "QualificationSettings",
    "arun_external_refine",
    "arun_external_tune",
    "derive_iteration_spec",
    "load_external_tune_spec",
    "validate_qualification_approval",
    "run_external_refine",
    "run_external_tune",
]
