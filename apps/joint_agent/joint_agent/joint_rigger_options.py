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

# Exact source intake selected for the disabled-by-default 0.5 limited preview.
# The adapter records these declared pins in every handoff diagnostic. Runtime
# package version discovery supplements this identity; it does not turn the
# declaration into release evidence or a source-tree verification claim.
USD_JOINT_RIGGER_SOURCE_VERSION = "0.1.0"
USD_JOINT_RIGGER_SOURCE_COMMIT = "a56a5c54342b933c6d671bcd2e264b46a6076219"
USD_JOINT_RIGGER_SOURCE_TREE = "8454dd908e53579ae6553549ffe37ee3d5efc9ad"
USD_JOINT_RIGGER_HANDOFF_VERSION = "joint-agent-wp-m2-limited-preview-v0"


def format_allowed_values(values: tuple[str, ...]) -> str:
    return ", ".join(values)
