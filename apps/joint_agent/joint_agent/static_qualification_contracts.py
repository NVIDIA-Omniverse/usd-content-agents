# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lane-aware static qualification validation-contract registry.

The retained 0.5 release-gate lane and the articulation-v2 lane share Gate 3
validators but intentionally use different contract, authoring, and readback
contracts.  Keeping that distinction in one neutral registry prevents stage or
lane substitution without importing run/evidence models.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

type StaticQualificationContractLane = Literal[
    "release_gate_v0_5",
    "articulation_v2",
]
type StaticQualificationValidationStageName = Literal[
    "contract",
    "authoring",
    "readback",
    "gate3a",
    "gate3b",
]

RELEASE_GATE_V0_5_CONTRACT_LANE: Literal["release_gate_v0_5"] = "release_gate_v0_5"
ARTICULATION_V2_CONTRACT_LANE: Literal["articulation_v2"] = "articulation_v2"

ARTICULATION_V2_STATIC_ADOPTION_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-adoption-v1"
] = "joint-agent-articulation-v2-static-adoption-v1"
ARTICULATION_V2_STATIC_AUTHORING_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-authoring-v1"
] = "joint-agent-articulation-v2-static-authoring-v1"
ARTICULATION_V2_STATIC_READBACK_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-readback-v1"
] = "joint-agent-articulation-v2-static-readback-v1"
STATIC_GATE3A_VALIDATION_CONTRACT: Literal[
    "joint-agent-isaac-sim-asset-validator-v2"
] = "joint-agent-isaac-sim-asset-validator-v2"
STATIC_GATE3B_VALIDATION_CONTRACT: Literal[
    "joint-agent-simready-foundation-validation-v2"
] = "joint-agent-simready-foundation-validation-v2"

RELEASE_GATE_V0_5_STAGE_VALIDATION_CONTRACTS: Mapping[
    StaticQualificationValidationStageName,
    str,
] = MappingProxyType(
    {
        "contract": "joint-rigger-core-release-gate-plan-adoption-v4",
        "authoring": "joint-rigger-core-release-gate-authoring-v3",
        "readback": "joint-rigger-core-release-gate-closeout-v1",
        "gate3a": STATIC_GATE3A_VALIDATION_CONTRACT,
        "gate3b": STATIC_GATE3B_VALIDATION_CONTRACT,
    }
)
ARTICULATION_V2_STATIC_STAGE_VALIDATION_CONTRACTS: Mapping[
    StaticQualificationValidationStageName,
    str,
] = MappingProxyType(
    {
        "contract": ARTICULATION_V2_STATIC_ADOPTION_SCHEMA_VERSION,
        "authoring": ARTICULATION_V2_STATIC_AUTHORING_SCHEMA_VERSION,
        "readback": ARTICULATION_V2_STATIC_READBACK_SCHEMA_VERSION,
        "gate3a": STATIC_GATE3A_VALIDATION_CONTRACT,
        "gate3b": STATIC_GATE3B_VALIDATION_CONTRACT,
    }
)

STATIC_QUALIFICATION_VALIDATION_CONTRACT_REGISTRY: Mapping[
    StaticQualificationContractLane,
    Mapping[StaticQualificationValidationStageName, str],
] = MappingProxyType(
    {
        RELEASE_GATE_V0_5_CONTRACT_LANE: (RELEASE_GATE_V0_5_STAGE_VALIDATION_CONTRACTS),
        ARTICULATION_V2_CONTRACT_LANE: (
            ARTICULATION_V2_STATIC_STAGE_VALIDATION_CONTRACTS
        ),
    }
)

# Compatibility alias for existing 0.5 production consumers.  New code should
# select a lane explicitly through ``static_validation_contract``.
STATIC_STAGE_VALIDATION_CONTRACTS = RELEASE_GATE_V0_5_STAGE_VALIDATION_CONTRACTS


def static_validation_contract(
    lane: StaticQualificationContractLane,
    stage: StaticQualificationValidationStageName,
) -> str:
    """Return the sole contract admitted for one lane and stage."""

    try:
        return STATIC_QUALIFICATION_VALIDATION_CONTRACT_REGISTRY[lane][stage]
    except KeyError as exc:
        raise ValueError(
            f"unknown static qualification lane/stage: {lane!r}/{stage!r}"
        ) from exc


def require_registered_static_validation_contract(
    *,
    stage: StaticQualificationValidationStageName,
    value: str,
) -> str:
    """Reject a contract registered only for a different stage."""

    admitted = {
        contracts[stage]
        for contracts in STATIC_QUALIFICATION_VALIDATION_CONTRACT_REGISTRY.values()
    }
    if value not in admitted:
        raise ValueError(
            f"{stage} validation contract {value!r} is not registered for that stage"
        )
    return value


__all__ = [
    "ARTICULATION_V2_CONTRACT_LANE",
    "ARTICULATION_V2_STATIC_ADOPTION_SCHEMA_VERSION",
    "ARTICULATION_V2_STATIC_AUTHORING_SCHEMA_VERSION",
    "ARTICULATION_V2_STATIC_READBACK_SCHEMA_VERSION",
    "ARTICULATION_V2_STATIC_STAGE_VALIDATION_CONTRACTS",
    "RELEASE_GATE_V0_5_CONTRACT_LANE",
    "RELEASE_GATE_V0_5_STAGE_VALIDATION_CONTRACTS",
    "STATIC_GATE3A_VALIDATION_CONTRACT",
    "STATIC_GATE3B_VALIDATION_CONTRACT",
    "STATIC_QUALIFICATION_VALIDATION_CONTRACT_REGISTRY",
    "STATIC_STAGE_VALIDATION_CONTRACTS",
    "StaticQualificationContractLane",
    "StaticQualificationValidationStageName",
    "require_registered_static_validation_contract",
    "static_validation_contract",
]
