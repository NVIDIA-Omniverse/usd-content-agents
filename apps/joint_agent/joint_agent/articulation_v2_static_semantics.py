# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neutral selector and articulation-v2 static semantic authorities.

This module contains inert, public-safe contracts.  It deliberately carries
asset-relative paths rather than deployment locators.  Internal corpus code is
responsible for proving that those neutral identities came from canonical
storage.

The articulation-v2 semantic document is not an independently asserted copy of
USD facts.  :func:`derive_articulation_v2_static_joint_semantics` is the sole
transform from the exact ``selector-semantic-preimage-v1`` authority emitted by
the reference-asset preparer.  Admission therefore binds both domains and the
transform between them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

from joint_agent.functions.articulation_contract_v2 import (
    DistanceConstraintV2,
    ExplicitAttachmentFramesV2,
    FixedConstraintV2,
)
from joint_agent.functions.articulation_v2_controlled_distance import (
    CONTROLLED_DISTANCE_FIELD_PROPERTIES,
    ControlledDistanceContractV2,
    canonical_controlled_distance_v2_sha256,
)

SELECTOR_SEMANTIC_PREIMAGE_SCHEMA_VERSION: Literal[
    "joint-agent-selector-semantic-preimage-v1"
] = "joint-agent-selector-semantic-preimage-v1"
ARTICULATION_V2_STATIC_JOINT_SEMANTICS_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-joint-semantics-v1"
] = "joint-agent-articulation-v2-static-joint-semantics-v1"
ARTICULATION_V2_STATIC_SEMANTIC_TRANSFORM_VERSION: Literal[
    "joint-agent-selector-semantic-preimage-v1-to-articulation-v2-static-v1"
] = "joint-agent-selector-semantic-preimage-v1-to-articulation-v2-static-v1"
ARTICULATION_V2_CONTROLLED_DISTANCE_SELECTOR_SEMANTICS_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-controlled-distance-selector-semantics-v2"
] = "joint-agent-articulation-v2-controlled-distance-selector-semantics-v2"
ARTICULATION_V2_CONTROLLED_DISTANCE_SELECTOR_TRANSFORM_VERSION: Literal[
    "joint-agent-selector-semantic-preimage-v1-to-controlled-distance-v2"
] = "joint-agent-selector-semantic-preimage-v1-to-controlled-distance-v2"

type ArticulationV2StaticCapabilityId = Literal[
    "fixed.explicit_two_body_constraint",
    "distance.bounded_two_body_constraint",
]
type ArticulationV2StaticConstraintKind = Literal["fixed", "distance"]
type _StaticConstraint = Annotated[
    FixedConstraintV2 | DistanceConstraintV2,
    Field(discriminator="kind"),
]

_CAPABILITY_KIND: dict[
    ArticulationV2StaticCapabilityId,
    ArticulationV2StaticConstraintKind,
] = {
    "fixed.explicit_two_body_constraint": "fixed",
    "distance.bounded_two_body_constraint": "distance",
}
_USD_TYPE_BY_KIND: dict[ArticulationV2StaticConstraintKind, str] = {
    "fixed": "PhysicsFixedJoint",
    "distance": "PhysicsDistanceJoint",
}
_RELEASE_GATE_ASSET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


class _StaticAuthorityModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )


class ArticulationV2StaticDependencyBundleV3(_StaticAuthorityModel):
    """Exact v3 dependency-closure identity from the prepared corpus."""

    schema_version: Literal["joint-agent-usd-artifact-dependency-bundle-v3"]
    sha256: str
    entry_count: int = Field(gt=0)

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        return _lowercase_sha256(value, "dependency bundle")


class ArticulationV2StaticArtifactAuthorityV1(_StaticAuthorityModel):
    """Asset-relative artifact identity without a provider locator."""

    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def _canonical_asset_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value
            or "\\" in value
            or path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("artifact path must be canonical and asset-relative")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        return _lowercase_sha256(value, "artifact authority")


class ArticulationV2StaticUsdArtifactAuthorityV1(
    ArticulationV2StaticArtifactAuthorityV1
):
    """Prepared USD root plus its complete v3 closure identity."""

    dependency_bundle: ArticulationV2StaticDependencyBundleV3


class ArticulationV2StaticSelectorV1(_StaticAuthorityModel):
    kind: Literal["usd_prim"]
    value: str

    @field_validator("value")
    @classmethod
    def _canonical_prim_path(cls, value: str) -> str:
        return _canonical_prim_path(value, "selector")


class SelectorSemanticTransformOpV1(_StaticAuthorityModel):
    name: str
    type: str
    value: str

    @field_validator("name", "type", "value")
    @classmethod
    def _nonblank(cls, value: str, info: Any) -> str:
        if not value or value.strip() != value or "\x00" in value:
            raise ValueError(f"{info.field_name} must be exact and nonblank")
        return value


class SelectorSemanticHierarchyPrimV1(_StaticAuthorityModel):
    path: str
    type_name: str
    applied_schemas: tuple[str, ...]
    xform_ops: tuple[SelectorSemanticTransformOpV1, ...]

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        return _canonical_prim_path(value, "endpoint hierarchy path")

    @field_validator("type_name")
    @classmethod
    def _type_name(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("endpoint hierarchy type_name must be nonblank")
        return value


class SelectorSemanticEndpointHierarchyV1(_StaticAuthorityModel):
    body0: tuple[SelectorSemanticHierarchyPrimV1, ...] = Field(min_length=1)
    body1: tuple[SelectorSemanticHierarchyPrimV1, ...] = Field(min_length=1)


class SelectorSemanticAuthoredMetadataV1(_StaticAuthorityModel):
    applied_schemas: tuple[str, ...]
    authored_properties: tuple[str, ...]
    body0_applied_schemas: tuple[str, ...]
    body1_applied_schemas: tuple[str, ...]
    physics_attributes: dict[str, JsonValue]
    usd_type_name: Literal["PhysicsFixedJoint", "PhysicsDistanceJoint"]

    @field_validator("authored_properties")
    @classmethod
    def _unique_authored_properties(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(not item for item in value):
            raise ValueError("authored properties must be unique and nonblank")
        return value


class SelectorSemanticJointV1(_StaticAuthorityModel):
    authored_metadata: SelectorSemanticAuthoredMetadataV1
    axis: None
    axis_world: None
    body0: str
    body1: str
    joint_prim_path: str
    joint_type: ArticulationV2StaticConstraintKind
    lower_limit: None
    upper_limit: None

    @field_validator("body0", "body1", "joint_prim_path")
    @classmethod
    def _paths(cls, value: str) -> str:
        return _canonical_prim_path(value, "selector-semantic joint path")

    @model_validator(mode="after")
    def _joint_shape(self) -> SelectorSemanticJointV1:
        if self.body0 == self.body1:
            raise ValueError("selector-semantic joint endpoints must be distinct")
        if self.authored_metadata.usd_type_name != _USD_TYPE_BY_KIND[self.joint_type]:
            raise ValueError("selector-semantic joint type differs from USD type")
        return self


class SelectorSemanticStageV1(_StaticAuthorityModel):
    default_prim: str
    meters_per_unit: float = Field(gt=0.0)
    up_axis: Literal["Y", "Z"]

    @field_validator("default_prim")
    @classmethod
    def _default_prim(cls, value: str) -> str:
        return _canonical_prim_path(value, "selector-semantic default prim")

    @field_validator("meters_per_unit")
    @classmethod
    def _finite_scale(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("selector-semantic stage scale must be finite")
        return value


class SelectorSemanticPreimageV1(_StaticAuthorityModel):
    """Exact persisted #1025 selector-semantic-preimage-v1 domain."""

    schema_version: Literal["joint-agent-selector-semantic-preimage-v1"]
    selector: ArticulationV2StaticSelectorV1
    topology_decision: Literal["explicit_two_body_constraint"]
    stage: SelectorSemanticStageV1
    joint: SelectorSemanticJointV1
    endpoint_hierarchy: SelectorSemanticEndpointHierarchyV1

    @model_validator(mode="after")
    def _complete_source_semantics(self) -> SelectorSemanticPreimageV1:
        if self.selector.value != self.joint.joint_prim_path:
            raise ValueError("selector semantic joint path differs from selector")
        for endpoint in ("body0", "body1"):
            hierarchy = getattr(self.endpoint_hierarchy, endpoint)
            expected_endpoint = getattr(self.joint, endpoint)
            paths = tuple(item.path for item in hierarchy)
            if paths[0] != self.stage.default_prim or paths[-1] != expected_endpoint:
                raise ValueError(
                    f"{endpoint} hierarchy does not connect the default prim to its body"
                )
            if len(paths) != len(set(paths)) or any(
                PurePosixPath(child).parent.as_posix() != parent
                for parent, child in zip(paths, paths[1:], strict=False)
            ):
                raise ValueError(f"{endpoint} hierarchy is not one exact parent chain")
        return self


class ControlledDistanceSelectorSemanticAuthoredMetadataV2(_StaticAuthorityModel):
    """Exact #1118 prismatic selector metadata retained by the preparer."""

    applied_schemas: tuple[str, ...]
    authored_properties: tuple[str, ...]
    body0_applied_schemas: tuple[str, ...]
    body1_applied_schemas: tuple[str, ...]
    physics_attributes: dict[str, JsonValue]
    usd_type_name: Literal["PhysicsPrismaticJoint"]

    @model_validator(mode="after")
    def _complete_controlled_distance_shape(
        self,
    ) -> ControlledDistanceSelectorSemanticAuthoredMetadataV2:
        expected_properties = {
            "drive:linear:physics:damping",
            "drive:linear:physics:maxForce",
            "drive:linear:physics:stiffness",
            "drive:linear:physics:targetPosition",
            "drive:linear:physics:targetVelocity",
            "drive:linear:physics:type",
            "physics:axis",
            "physics:body0",
            "physics:body1",
            "physics:localPos0",
            "physics:localPos1",
            "physics:localRot0",
            "physics:localRot1",
            "physics:lowerLimit",
            "physics:upperLimit",
            "state:linear:physics:position",
            "state:linear:physics:velocity",
        }
        expected_attributes = {
            "physics:axis",
            "physics:localPos0",
            "physics:localPos1",
            "physics:localRot0",
            "physics:localRot1",
            "physics:lowerLimit",
            "physics:upperLimit",
        }
        applied_schemas = set(self.applied_schemas)
        if "PhysicsDriveAPI:linear" not in applied_schemas or not applied_schemas <= {
            "PhysicsDriveAPI:linear",
            "PhysicsJointStateAPI:linear",
        }:
            raise ValueError(
                "controlled distance requires the exact linear drive/state APIs"
            )
        if set(self.authored_properties) != expected_properties:
            raise ValueError(
                "controlled distance requires the complete drive/state property set"
            )
        if set(self.physics_attributes) != expected_attributes:
            raise ValueError(
                "controlled distance selector attributes differ from the retained set"
            )
        return self


class ControlledDistanceSelectorSemanticJointV2(_StaticAuthorityModel):
    authored_metadata: ControlledDistanceSelectorSemanticAuthoredMetadataV2
    axis: Literal["x"]
    axis_world: tuple[float, float, float]
    body0: str
    body1: str
    joint_prim_path: str
    joint_type: Literal["prismatic"]
    lower_limit: float
    upper_limit: float

    @field_validator("body0", "body1", "joint_prim_path")
    @classmethod
    def _paths(cls, value: str) -> str:
        return _canonical_prim_path(value, "controlled-distance selector path")

    @model_validator(mode="after")
    def _positive_axial_interval(self) -> ControlledDistanceSelectorSemanticJointV2:
        attributes = self.authored_metadata.physics_attributes
        if self.body0 == self.body1:
            raise ValueError("controlled-distance endpoints must be distinct")
        if self.axis_world != (1.0, 0.0, 0.0) or attributes["physics:axis"] != "X":
            raise ValueError("controlled distance requires canonical positive local X")
        if (
            not math.isfinite(self.lower_limit)
            or not math.isfinite(self.upper_limit)
            or not 0.0 < self.lower_limit < self.upper_limit
            or attributes["physics:lowerLimit"] != self.lower_limit
            or attributes["physics:upperLimit"] != self.upper_limit
        ):
            raise ValueError(
                "controlled distance requires exact positive ordered limits"
            )
        return self


class ControlledDistanceSelectorSemanticPreimageV2(_StaticAuthorityModel):
    """The immutable #1118 preimage, narrowed to the successor representation."""

    schema_version: Literal["joint-agent-selector-semantic-preimage-v1"]
    selector: ArticulationV2StaticSelectorV1
    topology_decision: Literal["explicit_two_body_constraint"]
    stage: SelectorSemanticStageV1
    joint: ControlledDistanceSelectorSemanticJointV2
    endpoint_hierarchy: SelectorSemanticEndpointHierarchyV1

    @model_validator(mode="after")
    def _complete_source_semantics(
        self,
    ) -> ControlledDistanceSelectorSemanticPreimageV2:
        if self.selector.value != self.joint.joint_prim_path:
            raise ValueError("controlled-distance joint path differs from selector")
        for endpoint in ("body0", "body1"):
            hierarchy = getattr(self.endpoint_hierarchy, endpoint)
            expected_endpoint = getattr(self.joint, endpoint)
            paths = tuple(item.path for item in hierarchy)
            if paths[0] != self.stage.default_prim or paths[-1] != expected_endpoint:
                raise ValueError(
                    f"{endpoint} hierarchy does not connect the default prim to its body"
                )
            if len(paths) != len(set(paths)) or any(
                PurePosixPath(child).parent.as_posix() != parent
                for parent, child in zip(paths, paths[1:], strict=False)
            ):
                raise ValueError(f"{endpoint} hierarchy is not one exact parent chain")
        return self


class ArticulationV2StaticSelectorAuthoringReferenceV1(_StaticAuthorityModel):
    """Neutral identity projection of one exact selector authoring reference."""

    joint_type: ArticulationV2StaticConstraintKind
    selector: ArticulationV2StaticSelectorV1
    parent_reference_sha256: str
    semantic_digest_sha256: str
    raw_reference: ArticulationV2StaticUsdArtifactAuthorityV1
    package_reference: ArticulationV2StaticUsdArtifactAuthorityV1
    effective_reference_manifest: ArticulationV2StaticArtifactAuthorityV1
    identity: ArticulationV2StaticArtifactAuthorityV1

    @field_validator("parent_reference_sha256", "semantic_digest_sha256")
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        return _lowercase_sha256(value, info.field_name)

    @model_validator(mode="after")
    def _one_canonical_selector_reference(
        self,
    ) -> ArticulationV2StaticSelectorAuthoringReferenceV1:
        actual_expected = (
            (
                self.raw_reference.path,
                f"usd/authoring/{self.joint_type}/reference.usda",
            ),
            (
                self.package_reference.path,
                f"usd/authoring/{self.joint_type}/reference.usdz",
            ),
            (
                self.effective_reference_manifest.path,
                (
                    f"references/authoring/{self.joint_type}/"
                    "effective_reference_manifest.json"
                ),
            ),
            (
                self.identity.path,
                f"references/authoring/{self.joint_type}/identity.json",
            ),
        )
        if any(actual != required for actual, required in actual_expected):
            raise ValueError("selector reference artifact paths disagree with role")
        return self


class ArticulationV2ControlledDistanceSelectorAuthoringReferenceV2(
    _StaticAuthorityModel
):
    """Neutral identity projection of the #1118 controlled-distance reference."""

    role: Literal["controlled_distance"]
    joint_type: Literal["prismatic"]
    selector: ArticulationV2StaticSelectorV1
    parent_reference_sha256: str
    semantic_digest_sha256: str
    raw_reference: ArticulationV2StaticUsdArtifactAuthorityV1
    package_reference: ArticulationV2StaticUsdArtifactAuthorityV1
    effective_reference_manifest: ArticulationV2StaticArtifactAuthorityV1
    identity: ArticulationV2StaticArtifactAuthorityV1

    @field_validator("parent_reference_sha256", "semantic_digest_sha256")
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        return _lowercase_sha256(value, info.field_name)

    @model_validator(mode="after")
    def _canonical_paths(
        self,
    ) -> ArticulationV2ControlledDistanceSelectorAuthoringReferenceV2:
        expected = (
            (
                self.raw_reference.path,
                "usd/authoring/controlled_distance/reference.usda",
            ),
            (
                self.package_reference.path,
                "usd/authoring/controlled_distance/reference.usdz",
            ),
            (
                self.effective_reference_manifest.path,
                "references/authoring/controlled_distance/effective_reference_manifest.json",
            ),
            (
                self.identity.path,
                "references/authoring/controlled_distance/identity.json",
            ),
        )
        if any(actual != required for actual, required in expected):
            raise ValueError("controlled-distance reference paths disagree with role")
        return self


class ArticulationV2StaticSemanticSourceAuthorityV1(_StaticAuthorityModel):
    """Neutral parent source and effective-manifest artifact identities."""

    parent_reference: ArticulationV2StaticUsdArtifactAuthorityV1
    effective_reference_manifest: ArticulationV2StaticArtifactAuthorityV1

    @model_validator(mode="after")
    def _one_parent_authority(self) -> ArticulationV2StaticSemanticSourceAuthorityV1:
        if self.parent_reference.path != "usd/reference.usdz" or (
            self.effective_reference_manifest.path
            != "references/effective_reference_manifest.json"
        ):
            raise ValueError("parent semantic authority uses noncanonical asset paths")
        return self


class ArticulationV2StaticJointSemanticsV1(_StaticAuthorityModel):
    """Deterministic articulation-v2 projection of one source preimage."""

    schema_version: Literal["joint-agent-articulation-v2-static-joint-semantics-v1"]
    transform_version: Literal[
        "joint-agent-selector-semantic-preimage-v1-to-articulation-v2-static-v1"
    ]
    source_semantic_preimage_sha256: str
    capability_id: ArticulationV2StaticCapabilityId
    release_gate_asset_id: str
    selector_type: Literal["usd_prim"]
    source_joint_path: str
    joint_id: str
    topology_decision: Literal["explicit_two_body_constraint"]
    source_authority: ArticulationV2StaticSemanticSourceAuthorityV1
    source_stage_default_prim: str
    source_stage_meters_per_unit: float = Field(gt=0.0)
    source_stage_up_axis: Literal["Y", "Z"]
    body0_link_id: str
    body1_link_id: str
    body0_prim_path: str
    body1_prim_path: str
    linear_unit_interpretation: Literal["source-stage-units-normalized-to-si-meters-v1"]
    orientation_interpretation: Literal["body-local-active-unit-quaternion-wxyz-v1"]
    attachments: ExplicitAttachmentFramesV2
    constraint: _StaticConstraint

    @field_validator("source_semantic_preimage_sha256")
    @classmethod
    def _source_digest(cls, value: str) -> str:
        return _lowercase_sha256(value, "source semantic preimage")

    @field_validator("release_gate_asset_id")
    @classmethod
    def _canonical_asset_id(cls, value: str) -> str:
        if _RELEASE_GATE_ASSET_ID_RE.fullmatch(value) is None:
            raise ValueError(
                "release_gate_asset_id must use canonical lowercase asset-ID syntax"
            )
        return value

    @field_validator(
        "source_joint_path",
        "source_stage_default_prim",
        "body0_prim_path",
        "body1_prim_path",
    )
    @classmethod
    def _canonical_paths(cls, value: str) -> str:
        return _canonical_prim_path(value, "joint semantic path")

    @field_validator("joint_id", "body0_link_id", "body1_link_id")
    @classmethod
    def _nonblank_id(cls, value: str) -> str:
        if not value or value.strip() != value or "\x00" in value:
            raise ValueError("joint and link IDs must be exact and nonblank")
        return value

    @field_validator("source_stage_meters_per_unit")
    @classmethod
    def _finite_scale(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("source stage scale must be finite")
        return value

    @model_validator(mode="after")
    def _complete_authoring_semantics(
        self,
    ) -> ArticulationV2StaticJointSemanticsV1:
        if self.constraint.kind != _CAPABILITY_KIND[self.capability_id]:
            raise ValueError("capability and static constraint kind must agree")
        if self.body0_prim_path == self.body1_prim_path:
            raise ValueError("directed fixed/distance endpoints must be distinct")
        if self.body0_link_id == self.body1_link_id:
            raise ValueError("directed fixed/distance link IDs must be distinct")
        if isinstance(self.constraint, DistanceConstraintV2) and (
            self.constraint.minimum_meters is None
            or self.constraint.maximum_meters is None
        ):
            raise ValueError("distance semantics require both exact meter bounds")
        return self


class ArticulationV2ControlledDistanceSelectorSemanticsV2(_StaticAuthorityModel):
    """Deterministic admission projection of the #1118 selector and v2 overlay."""

    schema_version: Literal[
        "joint-agent-articulation-v2-controlled-distance-selector-semantics-v2"
    ]
    transform_version: Literal[
        "joint-agent-selector-semantic-preimage-v1-to-controlled-distance-v2"
    ]
    source_semantic_preimage_sha256: str
    capability_id: Literal["distance.bounded_two_body_constraint"]
    release_gate_asset_id: str
    selector_type: Literal["usd_prim"]
    source_joint_path: str
    topology_decision: Literal["explicit_two_body_constraint"]
    source_authority: ArticulationV2StaticSemanticSourceAuthorityV1
    source_contract: ArticulationV2StaticUsdArtifactAuthorityV1
    controlled_distance_contract: ControlledDistanceContractV2
    controlled_distance_contract_sha256: str
    release_selection_applied: Literal[False]

    @field_validator(
        "source_semantic_preimage_sha256",
        "controlled_distance_contract_sha256",
    )
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        return _lowercase_sha256(value, info.field_name)

    @field_validator("release_gate_asset_id")
    @classmethod
    def _asset_id(cls, value: str) -> str:
        if _RELEASE_GATE_ASSET_ID_RE.fullmatch(value) is None:
            raise ValueError("controlled-distance asset ID is not canonical")
        return value

    @field_validator("source_joint_path")
    @classmethod
    def _joint_path(cls, value: str) -> str:
        return _canonical_prim_path(value, "controlled-distance source joint path")

    @model_validator(mode="after")
    def _exact_overlay(self) -> ArticulationV2ControlledDistanceSelectorSemanticsV2:
        contract = self.controlled_distance_contract
        if (
            contract.source_joint_path != self.source_joint_path
            or contract.source_artifact.root_sha256 != self.source_contract.sha256
            or contract.static_qualified is not False
            or contract.dynamic_qualified is not False
            or contract.public_enabled is not False
        ):
            raise ValueError(
                "controlled-distance overlay differs from its admitted selector authority"
            )
        if (
            contract.source_artifact.dependency_bundle_sha256
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
        if canonical_controlled_distance_v2_sha256(contract) != (
            self.controlled_distance_contract_sha256
        ):
            raise ValueError("controlled-distance contract digest is not canonical")
        return self


def _lowercase_sha256(value: str, label: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} requires a lowercase SHA-256")
    return value


def _canonical_prim_path(value: str, label: str) -> str:
    if (
        not value.startswith("/")
        or value == "/"
        or value.endswith("/")
        or "//" in value
        or "\\" in value
        or any(
            not part
            or not (part[0].isalpha() or part[0] == "_")
            or any(not (character.isalnum() or character == "_") for character in part)
            for part in value[1:].split("/")
        )
    ):
        raise ValueError(f"{label} must be a canonical USD prim path")
    return value


def _canonical_model_bytes(model: BaseModel, adapter: TypeAdapter[Any]) -> bytes:
    return json.dumps(
        adapter.dump_python(model, mode="json", exclude_none=False),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


_SOURCE_PREIMAGE_ADAPTER = TypeAdapter(SelectorSemanticPreimageV1)
_SEMANTICS_ADAPTER = TypeAdapter(ArticulationV2StaticJointSemanticsV1)
_CONTROLLED_SOURCE_PREIMAGE_ADAPTER = TypeAdapter(
    ControlledDistanceSelectorSemanticPreimageV2
)
_CONTROLLED_SEMANTICS_ADAPTER = TypeAdapter(
    ArticulationV2ControlledDistanceSelectorSemanticsV2
)


def canonical_selector_semantic_preimage_bytes(
    preimage: SelectorSemanticPreimageV1,
) -> bytes:
    """Return the exact canonical byte domain used by #1025."""

    if type(preimage) is not SelectorSemanticPreimageV1:
        raise TypeError("preimage must be an exact SelectorSemanticPreimageV1")
    return _canonical_model_bytes(preimage, _SOURCE_PREIMAGE_ADAPTER)


def selector_semantic_preimage_sha256(preimage: SelectorSemanticPreimageV1) -> str:
    return hashlib.sha256(
        canonical_selector_semantic_preimage_bytes(preimage)
    ).hexdigest()


def canonical_controlled_distance_selector_preimage_bytes(
    preimage: ControlledDistanceSelectorSemanticPreimageV2,
) -> bytes:
    """Return the exact canonical #1118 controlled selector preimage bytes."""

    if type(preimage) is not ControlledDistanceSelectorSemanticPreimageV2:
        raise TypeError(
            "preimage must be an exact ControlledDistanceSelectorSemanticPreimageV2"
        )
    return _canonical_model_bytes(preimage, _CONTROLLED_SOURCE_PREIMAGE_ADAPTER)


def controlled_distance_selector_preimage_sha256(
    preimage: ControlledDistanceSelectorSemanticPreimageV2,
) -> str:
    return hashlib.sha256(
        canonical_controlled_distance_selector_preimage_bytes(preimage)
    ).hexdigest()


def _decimal_components(
    value: JsonValue, *, length: int, label: str
) -> tuple[Decimal, ...]:
    if (
        not isinstance(value, str)
        or not value.startswith("(")
        or not value.endswith(")")
    ):
        raise ValueError(f"{label} must use the persisted USD tuple syntax")
    tokens = tuple(item.strip() for item in value[1:-1].split(","))
    if len(tokens) != length or any(
        _NUMBER_RE.fullmatch(item) is None for item in tokens
    ):
        raise ValueError(f"{label} must contain exactly {length} canonical numbers")
    try:
        return tuple(Decimal(item) for item in tokens)
    except InvalidOperation as exc:
        raise ValueError(f"{label} contains an invalid number") from exc


def _scaled_vector(value: JsonValue, scale: float, *, label: str) -> tuple[float, ...]:
    factor = Decimal(str(scale))
    return tuple(
        float(component * factor)
        for component in _decimal_components(value, length=3, label=label)
    )


def _quaternion(value: JsonValue, *, label: str) -> tuple[float, ...]:
    return tuple(
        float(component)
        for component in _decimal_components(value, length=4, label=label)
    )


def _distance(value: JsonValue, scale: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a persisted JSON number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(Decimal(str(value)) * Decimal(str(scale)))


def derive_articulation_v2_static_joint_semantics(
    preimage: SelectorSemanticPreimageV1,
    *,
    capability_id: ArticulationV2StaticCapabilityId,
    release_gate_asset_id: str,
    source_authority: ArticulationV2StaticSemanticSourceAuthorityV1,
) -> ArticulationV2StaticJointSemanticsV1:
    """Deterministically transform the exact source preimage into v2 semantics."""

    if type(preimage) is not SelectorSemanticPreimageV1:
        raise TypeError("preimage must be an exact SelectorSemanticPreimageV1")
    if type(source_authority) is not ArticulationV2StaticSemanticSourceAuthorityV1:
        raise TypeError(
            "source_authority must be an exact "
            "ArticulationV2StaticSemanticSourceAuthorityV1"
        )
    if _RELEASE_GATE_ASSET_ID_RE.fullmatch(release_gate_asset_id) is None:
        raise ValueError("release_gate_asset_id is not canonical")

    kind = _CAPABILITY_KIND[capability_id]
    joint = preimage.joint
    if joint.joint_type != kind:
        raise ValueError("source joint kind differs from the capability")
    attributes = joint.authored_metadata.physics_attributes
    base_properties = {
        "physics:body0",
        "physics:body1",
        "physics:localPos0",
        "physics:localPos1",
        "physics:localRot0",
        "physics:localRot1",
    }
    expected_properties = (
        base_properties
        if kind == "fixed"
        else base_properties | {"physics:minDistance", "physics:maxDistance"}
    )
    if set(joint.authored_metadata.authored_properties) != expected_properties or set(
        attributes
    ) != expected_properties - {"physics:body0", "physics:body1"}:
        raise ValueError("source joint authored-property set differs from its kind")

    scale = preimage.stage.meters_per_unit
    attachments = ExplicitAttachmentFramesV2.model_validate(
        {
            "kind": "explicit",
            "body0": {
                "position_meters": _scaled_vector(
                    attributes["physics:localPos0"],
                    scale,
                    label="physics:localPos0",
                ),
                "orientation_wxyz": _quaternion(
                    attributes["physics:localRot0"],
                    label="physics:localRot0",
                ),
                "sampling_mode": "static",
            },
            "body1": {
                "position_meters": _scaled_vector(
                    attributes["physics:localPos1"],
                    scale,
                    label="physics:localPos1",
                ),
                "orientation_wxyz": _quaternion(
                    attributes["physics:localRot1"],
                    label="physics:localRot1",
                ),
                "sampling_mode": "static",
            },
        },
        strict=True,
    )
    constraint: FixedConstraintV2 | DistanceConstraintV2
    if kind == "fixed":
        constraint = FixedConstraintV2(kind="fixed")
    else:
        constraint = DistanceConstraintV2(
            kind="distance",
            minimum_meters=_distance(
                attributes["physics:minDistance"],
                scale,
                label="physics:minDistance",
            ),
            maximum_meters=_distance(
                attributes["physics:maxDistance"],
                scale,
                label="physics:maxDistance",
            ),
        )

    joint_id = PurePosixPath(joint.joint_prim_path).name
    return ArticulationV2StaticJointSemanticsV1(
        schema_version=ARTICULATION_V2_STATIC_JOINT_SEMANTICS_SCHEMA_VERSION,
        transform_version=ARTICULATION_V2_STATIC_SEMANTIC_TRANSFORM_VERSION,
        source_semantic_preimage_sha256=selector_semantic_preimage_sha256(preimage),
        capability_id=capability_id,
        release_gate_asset_id=release_gate_asset_id,
        selector_type=preimage.selector.kind,
        source_joint_path=preimage.selector.value,
        joint_id=joint_id,
        topology_decision=preimage.topology_decision,
        source_authority=source_authority,
        source_stage_default_prim=preimage.stage.default_prim,
        source_stage_meters_per_unit=scale,
        source_stage_up_axis=preimage.stage.up_axis,
        body0_link_id=f"{joint_id}_body0",
        body1_link_id=f"{joint_id}_body1",
        body0_prim_path=joint.body0,
        body1_prim_path=joint.body1,
        linear_unit_interpretation=("source-stage-units-normalized-to-si-meters-v1"),
        orientation_interpretation=("body-local-active-unit-quaternion-wxyz-v1"),
        attachments=attachments,
        constraint=constraint,
    )


def canonical_articulation_v2_static_joint_semantics_bytes(
    semantics: ArticulationV2StaticJointSemanticsV1,
) -> bytes:
    """Return the canonical, transform-domain-versioned semantic preimage."""

    if type(semantics) is not ArticulationV2StaticJointSemanticsV1:
        raise TypeError(
            "semantics must be an exact ArticulationV2StaticJointSemanticsV1"
        )
    return _canonical_model_bytes(semantics, _SEMANTICS_ADAPTER)


def articulation_v2_static_joint_semantics_sha256(
    semantics: ArticulationV2StaticJointSemanticsV1,
) -> str:
    return hashlib.sha256(
        canonical_articulation_v2_static_joint_semantics_bytes(semantics)
    ).hexdigest()


def derive_articulation_v2_controlled_distance_selector_semantics(
    preimage: ControlledDistanceSelectorSemanticPreimageV2,
    *,
    release_gate_asset_id: str,
    source_authority: ArticulationV2StaticSemanticSourceAuthorityV1,
    source_contract: ArticulationV2StaticUsdArtifactAuthorityV1,
    controlled_distance_contract: ControlledDistanceContractV2,
) -> ArticulationV2ControlledDistanceSelectorSemanticsV2:
    """Bind the immutable selector domain to the merged v2 overlay contract."""

    if type(preimage) is not ControlledDistanceSelectorSemanticPreimageV2:
        raise TypeError(
            "preimage must be an exact ControlledDistanceSelectorSemanticPreimageV2"
        )
    if type(source_authority) is not ArticulationV2StaticSemanticSourceAuthorityV1:
        raise TypeError(
            "source_authority must be an exact "
            "ArticulationV2StaticSemanticSourceAuthorityV1"
        )
    if type(source_contract) is not ArticulationV2StaticUsdArtifactAuthorityV1:
        raise TypeError(
            "source_contract must be an exact ArticulationV2StaticUsdArtifactAuthorityV1"
        )
    if type(controlled_distance_contract) is not ControlledDistanceContractV2:
        raise TypeError(
            "controlled_distance_contract must be an exact ControlledDistanceContractV2"
        )
    if _RELEASE_GATE_ASSET_ID_RE.fullmatch(release_gate_asset_id) is None:
        raise ValueError("release_gate_asset_id is not canonical")

    joint = preimage.joint
    contract = controlled_distance_contract
    if (
        contract.source_joint_path != joint.joint_prim_path
        or contract.body0_prim_path != joint.body0
        or contract.body1_prim_path != joint.body1
        or contract.axis_stage != joint.axis_world
        or contract.axial_interval.minimum_meters != joint.lower_limit
        or contract.axial_interval.maximum_meters != joint.upper_limit
    ):
        raise ValueError(
            "controlled-distance contract differs from the source selector preimage"
        )
    required_control_properties = {
        property_name
        for properties in CONTROLLED_DISTANCE_FIELD_PROPERTIES.values()
        for property_name in properties
        if not property_name.startswith("usd:")
        and not property_name.startswith("xformOp")
    }
    if not required_control_properties.issubset(
        set(joint.authored_metadata.authored_properties)
    ):
        raise ValueError(
            "controlled-distance selector lacks required drive/state facts"
        )
    return ArticulationV2ControlledDistanceSelectorSemanticsV2(
        schema_version=(
            ARTICULATION_V2_CONTROLLED_DISTANCE_SELECTOR_SEMANTICS_SCHEMA_VERSION
        ),
        transform_version=(
            ARTICULATION_V2_CONTROLLED_DISTANCE_SELECTOR_TRANSFORM_VERSION
        ),
        source_semantic_preimage_sha256=(
            controlled_distance_selector_preimage_sha256(preimage)
        ),
        capability_id="distance.bounded_two_body_constraint",
        release_gate_asset_id=release_gate_asset_id,
        selector_type=preimage.selector.kind,
        source_joint_path=preimage.selector.value,
        topology_decision=preimage.topology_decision,
        source_authority=source_authority,
        source_contract=source_contract,
        controlled_distance_contract=contract,
        controlled_distance_contract_sha256=(
            canonical_controlled_distance_v2_sha256(contract)
        ),
        release_selection_applied=False,
    )


def canonical_articulation_v2_controlled_distance_selector_semantics_bytes(
    semantics: ArticulationV2ControlledDistanceSelectorSemanticsV2,
) -> bytes:
    if type(semantics) is not ArticulationV2ControlledDistanceSelectorSemanticsV2:
        raise TypeError(
            "semantics must be an exact "
            "ArticulationV2ControlledDistanceSelectorSemanticsV2"
        )
    return _canonical_model_bytes(semantics, _CONTROLLED_SEMANTICS_ADAPTER)


def articulation_v2_controlled_distance_selector_semantics_sha256(
    semantics: ArticulationV2ControlledDistanceSelectorSemanticsV2,
) -> str:
    return hashlib.sha256(
        canonical_articulation_v2_controlled_distance_selector_semantics_bytes(
            semantics
        )
    ).hexdigest()


__all__ = [
    "ARTICULATION_V2_CONTROLLED_DISTANCE_SELECTOR_SEMANTICS_SCHEMA_VERSION",
    "ARTICULATION_V2_CONTROLLED_DISTANCE_SELECTOR_TRANSFORM_VERSION",
    "ARTICULATION_V2_STATIC_JOINT_SEMANTICS_SCHEMA_VERSION",
    "ARTICULATION_V2_STATIC_SEMANTIC_TRANSFORM_VERSION",
    "SELECTOR_SEMANTIC_PREIMAGE_SCHEMA_VERSION",
    "ArticulationV2StaticArtifactAuthorityV1",
    "ArticulationV2StaticCapabilityId",
    "ArticulationV2StaticConstraintKind",
    "ArticulationV2StaticDependencyBundleV3",
    "ArticulationV2StaticJointSemanticsV1",
    "ArticulationV2ControlledDistanceSelectorAuthoringReferenceV2",
    "ArticulationV2ControlledDistanceSelectorSemanticsV2",
    "ArticulationV2StaticSelectorAuthoringReferenceV1",
    "ArticulationV2StaticSelectorV1",
    "ArticulationV2StaticSemanticSourceAuthorityV1",
    "ArticulationV2StaticUsdArtifactAuthorityV1",
    "SelectorSemanticPreimageV1",
    "ControlledDistanceSelectorSemanticPreimageV2",
    "articulation_v2_controlled_distance_selector_semantics_sha256",
    "articulation_v2_static_joint_semantics_sha256",
    "canonical_articulation_v2_controlled_distance_selector_semantics_bytes",
    "canonical_articulation_v2_static_joint_semantics_bytes",
    "canonical_selector_semantic_preimage_bytes",
    "canonical_controlled_distance_selector_preimage_bytes",
    "controlled_distance_selector_preimage_sha256",
    "derive_articulation_v2_controlled_distance_selector_semantics",
    "derive_articulation_v2_static_joint_semantics",
    "selector_semantic_preimage_sha256",
]
