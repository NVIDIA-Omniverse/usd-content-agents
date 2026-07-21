# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Joint Rigger adapter options."""

from __future__ import annotations

from typing import Literal, cast, get_args

JointRiggerAdapterName = Literal["owned_core", "mock", "usd_joint_rigger"]
InternalJointRiggerAdapterName = Literal[
    "owned_core",
    "mock",
    "usd_joint_rigger",
    "stage2_candidate_edges",
]
MissingDependencyPolicy = Literal["skip", "block"]
CandidateReadinessPolicy = Literal["warn", "block"]

SUPPORTED_JOINT_RIGGER_ADAPTERS = cast(
    tuple[JointRiggerAdapterName, ...],
    get_args(JointRiggerAdapterName),
)
SUPPORTED_INTERNAL_JOINT_RIGGER_ADAPTERS = cast(
    tuple[InternalJointRiggerAdapterName, ...],
    get_args(InternalJointRiggerAdapterName),
)
SUPPORTED_MISSING_DEPENDENCY_POLICIES = cast(
    tuple[MissingDependencyPolicy, ...],
    get_args(MissingDependencyPolicy),
)
SUPPORTED_CANDIDATE_READINESS_POLICIES = cast(
    tuple[CandidateReadinessPolicy, ...],
    get_args(CandidateReadinessPolicy),
)

DEFAULT_JOINT_RIGGER_ADAPTER: JointRiggerAdapterName = "mock"
DEFAULT_SERVICE_JOINT_RIGGER_ADAPTER: JointRiggerAdapterName = "owned_core"
DEFAULT_MISSING_DEPENDENCY_POLICY: MissingDependencyPolicy = "skip"
DEFAULT_CANDIDATE_READINESS_POLICY: CandidateReadinessPolicy = "warn"
DEFAULT_USD_JOINT_RIGGER_TEMPLATE = "generic_prop"
DEFAULT_USD_JOINT_RIGGER_APPLY_MASSES = True
DEFAULT_USD_JOINT_RIGGER_APPLY_COLLISION = True
PREDICTION_FREE_JOINT_RIGGER_ADAPTERS = frozenset({"stage2_candidate_edges"})
PREDICTION_OPTIONAL_JOINT_RIGGER_ADAPTERS = frozenset({"owned_core"})
CANDIDATE_REQUIRED_JOINT_RIGGER_ADAPTERS = frozenset(
    {"owned_core", "stage2_candidate_edges"}
)


def format_allowed_values(values: tuple[str, ...]) -> str:
    return ", ".join(values)
