# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discriminated static authoring documents for controlled-distance v2.

These sibling contracts keep the frozen frame-authoring v1 wire surface
unchanged.  They retain the exact admitted #1118 selector, source contract,
target, and non-qualifying policy without invoking an authorer or validator.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    field_validator,
    model_validator,
)
from world_understanding.functions.physics.joint_rigger import ArtifactIdentityV1

from joint_agent.articulation_v2_static_semantics import (
    ArticulationV2StaticDependencyBundleV3,
)
from joint_agent.capability_manifest import (
    StaticQualificationControlledDistanceSelectorAdmissionV2,
)
from joint_agent.functions.articulation_contract_v2_frames import (
    CapturedUsdArtifactBindingV1,
    OpaqueArtifactIdentityV1,
)
from joint_agent.functions.articulation_v2_controlled_distance import (
    ControlledDistanceContractV2,
)
from joint_agent.functions.articulation_v2_controlled_distance_usd import (
    ControlledDistanceStageReadbackV2,
    ControlledDistanceV2SourceReadbackProtocol,
)

ARTICULATION_V2_CONTROLLED_DISTANCE_REPRESENTATION: Literal[
    "prismatic_axial_interval_drive_body1_to_body0_v2"
] = "prismatic_axial_interval_drive_body1_to_body0_v2"
ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_AUTHORING_CONTRACT_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-controlled-distance-authoring-contract-v2"
] = "joint-agent-articulation-v2-static-controlled-distance-authoring-contract-v2"
ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_AUTHORING_RESULT_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-controlled-distance-authoring-result-v2"
] = "joint-agent-articulation-v2-static-controlled-distance-authoring-result-v2"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _raw_usd_gate3_dependency_bundle_sha256(
    capture: OpaqueArtifactIdentityV1,
) -> str:
    document = {
        "schema_version": "joint-agent-usd-artifact-dependency-bundle-v3",
        "container": "raw_usd",
        "entries": [
            {
                "kind": "root_layer",
                "path": "<root>",
                "size": capture.size_bytes,
                "sha256": capture.sha256,
            }
        ],
    }
    return hashlib.sha256(
        json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _require_current_admitted_authoring_contract(
    contract: ArticulationV2StaticControlledDistanceAuthoringContractV2,
) -> None:
    from joint_agent.capability_manifest import load_capability_manifest

    authority = load_capability_manifest()
    capability_id = contract.selector.expected_joint_semantics.capability_id
    rows = tuple(
        row
        for row in authority.manifest.capabilities
        if row.capability_id == capability_id
    )
    if len(rows) != 1:
        raise ValueError("authoring result capability is not uniquely admitted")
    row = rows[0]
    admissions = row.qualification.selector_admissions
    if (
        authority.sha256 != contract.capability_manifest_sha256
        or row.admission_sha256 != contract.row_admission_sha256
        or admissions is None
        or len(admissions) != 1
        or admissions[0] != contract.selector
    ):
        raise ValueError("authoring result differs from current admitted authority")


class _ControlledDistanceStaticModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )


class ArticulationV2ControlledDistanceSourceContractV2(_ControlledDistanceStaticModel):
    """Exact local capture and admitted dependency closure for the v2 source."""

    capture: OpaqueArtifactIdentityV1
    format: Literal["usda"]
    dependency_bundle: ArticulationV2StaticDependencyBundleV3


class ArticulationV2StaticControlledDistanceAuthoringContractV2(
    _ControlledDistanceStaticModel
):
    """Deterministic, inert request contract for the admitted v2 overlay."""

    schema_version: Literal[
        "joint-agent-articulation-v2-static-controlled-distance-authoring-contract-v2"
    ]
    representation: Literal["prismatic_axial_interval_drive_body1_to_body0_v2"]
    capability_manifest_sha256: str
    row_admission_sha256: str
    selector: StaticQualificationControlledDistanceSelectorAdmissionV2
    source_contract: ArticulationV2ControlledDistanceSourceContractV2
    target: CapturedUsdArtifactBindingV1
    static_qualified: Literal[False]
    dynamic_qualified: Literal[False]
    public_enabled: Literal[False]
    release_selection_applied: Literal[False]

    @field_validator("capability_manifest_sha256", "row_admission_sha256")
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _exact_admitted_closure(
        self,
    ) -> ArticulationV2StaticControlledDistanceAuthoringContractV2:
        semantics = self.selector.expected_joint_semantics
        controlled = semantics.controlled_distance_contract
        source = self.source_contract
        if self.representation != controlled.representation:
            raise ValueError("authoring representation differs from admitted v2")
        if (
            source.capture.uri != semantics.source_contract.path
            or source.capture.sha256 != semantics.source_contract.sha256
            or source.capture.max_bytes <= 0
            or source.dependency_bundle != semantics.source_contract.dependency_bundle
            or controlled.source_artifact.root_sha256 != source.capture.sha256
        ):
            raise ValueError("authoring source differs from admitted v2 authority")
        if self.target.format != "usdz":
            raise ValueError("controlled-distance v2 target must remain exact USDZ")
        return self


class ArticulationV2StaticControlledDistanceAuthoringResultV2(
    _ControlledDistanceStaticModel
):
    """Typed result plumbing for a future v2 authorer invocation."""

    schema_version: Literal[
        "joint-agent-articulation-v2-static-controlled-distance-authoring-result-v2"
    ]
    representation: Literal["prismatic_axial_interval_drive_body1_to_body0_v2"]
    authoring_contract_sha256: str
    contract_artifact: OpaqueArtifactIdentityV1
    authoring_contract: ArticulationV2StaticControlledDistanceAuthoringContractV2
    source_contract: ArticulationV2ControlledDistanceSourceContractV2
    target: CapturedUsdArtifactBindingV1
    output_artifact: ArtifactIdentityV1
    controlled_distance_contract: ControlledDistanceContractV2
    readback: ControlledDistanceStageReadbackV2
    source_readback_protocol: ControlledDistanceV2SourceReadbackProtocol
    static_qualified: Literal[False]
    dynamic_qualified: Literal[False]
    public_enabled: Literal[False]
    release_selection_applied: Literal[False]

    @field_validator("authoring_contract_sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("authoring_contract_sha256 must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _exact_v2_result(
        self,
    ) -> ArticulationV2StaticControlledDistanceAuthoringResultV2:
        controlled = self.controlled_distance_contract
        contract = self.authoring_contract
        readback = self.readback
        contract_bytes = canonical_articulation_v2_static_controlled_distance_authoring_contract_bytes(
            contract
        )
        if (
            self.authoring_contract_sha256
            != articulation_v2_static_controlled_distance_authoring_contract_sha256(
                contract
            )
            or self.contract_artifact.sha256 != self.authoring_contract_sha256
            or self.contract_artifact.size_bytes != len(contract_bytes)
            or self.representation != contract.representation
            or self.source_contract != contract.source_contract
            or self.target != contract.target
            or controlled
            != contract.selector.expected_joint_semantics.controlled_distance_contract
        ):
            raise ValueError("authoring result differs from its exact v2 contract")
        if self.representation != controlled.representation:
            raise ValueError("authoring result representation differs from v2 contract")
        if (
            controlled.source_artifact.root_sha256
            != self.source_contract.capture.sha256
            or readback.joint_path != controlled.source_joint_path
            or readback.body0_prim_path != controlled.body0_prim_path
            or readback.body1_prim_path != controlled.body1_prim_path
            or readback.axis_stage != controlled.axis_stage
            or readback.axial_interval != controlled.axial_interval
            or readback.drive != controlled.drive
            or readback.state != controlled.state
        ):
            raise ValueError("authoring result readback differs from exact v2 contract")
        if (
            controlled.source_artifact.dependency_bundle_sha256
            == self.source_contract.dependency_bundle.sha256
        ):
            raise ValueError(
                "controlled-distance authorer and Gate 3 dependency domains "
                "must remain distinct"
            )
        if self.source_contract.dependency_bundle.entry_count != 1:
            raise ValueError(
                "controlled-distance raw source requires one Gate 3 dependency entry"
            )
        if self.source_contract.dependency_bundle.sha256 != (
            _raw_usd_gate3_dependency_bundle_sha256(self.source_contract.capture)
        ):
            raise ValueError(
                "controlled-distance Gate 3 dependency identity differs from "
                "the exact raw source capture"
            )
        _require_current_admitted_authoring_contract(contract)
        return self


_CONTRACT_ADAPTER = TypeAdapter(
    ArticulationV2StaticControlledDistanceAuthoringContractV2
)
_RESULT_ADAPTER = TypeAdapter(ArticulationV2StaticControlledDistanceAuthoringResultV2)


def _canonical_bytes(value: BaseModel, adapter: TypeAdapter[Any]) -> bytes:
    payload = json.dumps(
        adapter.dump_python(value, mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    validated = adapter.validate_json(payload, strict=True)
    return json.dumps(
        adapter.dump_python(validated, mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_articulation_v2_static_controlled_distance_authoring_contract_bytes(
    contract: ArticulationV2StaticControlledDistanceAuthoringContractV2,
) -> bytes:
    if type(contract) is not ArticulationV2StaticControlledDistanceAuthoringContractV2:
        raise TypeError("contract must be the exact controlled-distance v2 contract")
    return _canonical_bytes(contract, _CONTRACT_ADAPTER)


def articulation_v2_static_controlled_distance_authoring_contract_sha256(
    contract: ArticulationV2StaticControlledDistanceAuthoringContractV2,
) -> str:
    return hashlib.sha256(
        canonical_articulation_v2_static_controlled_distance_authoring_contract_bytes(
            contract
        )
    ).hexdigest()


def parse_articulation_v2_static_controlled_distance_authoring_contract(
    payload: bytes,
) -> ArticulationV2StaticControlledDistanceAuthoringContractV2:
    if type(payload) is not bytes or not payload:
        raise TypeError("controlled-distance v2 contract must be nonempty bytes")
    contract = _CONTRACT_ADAPTER.validate_json(payload, strict=True)
    if (
        canonical_articulation_v2_static_controlled_distance_authoring_contract_bytes(
            contract
        )
        != payload
    ):
        raise ValueError("controlled-distance v2 contract is not canonical")
    return contract


def canonical_articulation_v2_static_controlled_distance_authoring_result_bytes(
    result: ArticulationV2StaticControlledDistanceAuthoringResultV2,
) -> bytes:
    if type(result) is not ArticulationV2StaticControlledDistanceAuthoringResultV2:
        raise TypeError("result must be the exact controlled-distance v2 result")
    return _canonical_bytes(result, _RESULT_ADAPTER)


def parse_articulation_v2_static_controlled_distance_authoring_result(
    payload: bytes,
) -> ArticulationV2StaticControlledDistanceAuthoringResultV2:
    if type(payload) is not bytes or not payload:
        raise TypeError("controlled-distance v2 result must be nonempty bytes")
    result = _RESULT_ADAPTER.validate_json(payload, strict=True)
    if (
        canonical_articulation_v2_static_controlled_distance_authoring_result_bytes(
            result
        )
        != payload
    ):
        raise ValueError("controlled-distance v2 result is not canonical")
    return result


__all__ = [
    "ARTICULATION_V2_CONTROLLED_DISTANCE_REPRESENTATION",
    "ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_AUTHORING_CONTRACT_SCHEMA_VERSION",
    "ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_AUTHORING_RESULT_SCHEMA_VERSION",
    "ArticulationV2ControlledDistanceSourceContractV2",
    "ArticulationV2StaticControlledDistanceAuthoringContractV2",
    "ArticulationV2StaticControlledDistanceAuthoringResultV2",
    "articulation_v2_static_controlled_distance_authoring_contract_sha256",
    "canonical_articulation_v2_static_controlled_distance_authoring_contract_bytes",
    "canonical_articulation_v2_static_controlled_distance_authoring_result_bytes",
    "parse_articulation_v2_static_controlled_distance_authoring_contract",
    "parse_articulation_v2_static_controlled_distance_authoring_result",
]
