# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned Joint Agent capability policy and evidence bindings.

The manifest is deliberately separate from implementation enums. A type appearing
in a prompt, contract union, or authorer is not a product-support claim until its
manifest row reaches the corresponding support level.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal, cast, get_args

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from joint_agent.articulation_v2_static_semantics import (
    ArticulationV2ControlledDistanceSelectorAuthoringReferenceV2,
    ArticulationV2ControlledDistanceSelectorSemanticsV2,
    ArticulationV2StaticJointSemanticsV1,
    ArticulationV2StaticSelectorAuthoringReferenceV1,
    ArticulationV2StaticUsdArtifactAuthorityV1,
    ControlledDistanceSelectorSemanticPreimageV2,
    SelectorSemanticPreimageV1,
    articulation_v2_controlled_distance_selector_semantics_sha256,
    articulation_v2_static_joint_semantics_sha256,
    controlled_distance_selector_preimage_sha256,
    derive_articulation_v2_controlled_distance_selector_semantics,
    derive_articulation_v2_static_joint_semantics,
    selector_semantic_preimage_sha256,
)
from joint_agent.static_qualification_contracts import (
    ARTICULATION_V2_CONTRACT_LANE,
    static_validation_contract,
)

CAPABILITY_MANIFEST_SCHEMA_VERSION: Literal["joint-agent-capability-manifest-v3"] = (
    "joint-agent-capability-manifest-v3"
)
CAPABILITY_MANIFEST_RESOURCE = "data/joint_capability_manifest.json"
SUPPORT_TABLE_START = "<!-- joint-capability-manifest:start -->"
SUPPORT_TABLE_END = "<!-- joint-capability-manifest:end -->"
ARTICULATION_V2_CONTRACT_VERSION = "joint-agent-articulation-v2"
CORPUS_COVERAGE_SCHEMA_VERSION = "joint-agent-reference-corpus-coverage-v2"
CORPUS_QUALIFICATION_TRUTH_SCHEMA_VERSION = "joint-agent-corpus-qualification-truth-v1"
CORPUS_QUALIFICATION_TRUTH_AUTHORITY = "reference_corpus_manifest"
PREPARED_ASSET_MANIFEST_SCHEMA_VERSION = "joint-agent-reference-asset-v1"
PREPARED_ASSET_MANIFEST_QUALIFICATION_ROLE = "non_qualifying_reproduction_metadata"

type SupportLevel = Literal[
    "recognized",
    "contract_ready",
    "authorable",
    "static_qualified",
    "dynamic_qualified",
]
type CapabilityDisposition = Literal[
    "supported",
    "review_required",
    "unsupported",
    "deferred",
]
type EvidenceRole = Literal[
    "frozen_0_5",
    "representative_source",
    "analytic_positive",
    "fail_closed_negative",
]
type FieldApplicability = Literal[
    "required",
    "optional",
    "not_applicable",
    "deferred",
]
type PublicExposure = Literal["enabled", "gated", "internal_only"]
type QualificationStatus = Literal["pass", "fail", "not_run", "na"]
# The lane a capability row advertises may additionally be `scheduled`: a run
# has been admitted and its identities pinned, but no outcome is claimed. Per
# stage evidence keeps the four terminal values above -- widening that would let
# a stage carry a partial identity set with neither terminal branch firing.
type QualificationLaneStatus = Literal[
    "pass",
    "fail",
    "not_run",
    "na",
    "scheduled",
]
type StaticQualificationStageName = Literal[
    "contract",
    "authoring",
    "readback",
    "gate3a",
    "gate3b",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.-]+)?$")
_RELEASE_GATE_ASSET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_VALIDATION_CONTRACT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*-v[1-9][0-9]*$")
_USD_PRIM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SUPPORT_RANK: dict[SupportLevel, int] = {
    "recognized": 0,
    "contract_ready": 1,
    "authorable": 2,
    "static_qualified": 3,
    "dynamic_qualified": 4,
}
_EVIDENCE_CLASS_BY_ROLE: dict[EvidenceRole, str | None] = {
    "frozen_0_5": None,
    "representative_source": "source_backed",
    "analytic_positive": "analytic_positive",
    "fail_closed_negative": "analytic_negative",
}
# The frozen lane's historical outcome. Grandfathering is pinned to this exact
# status so the immutable rows cannot acquire a different one unattested.
_FROZEN_0_5_STATIC_STATUS = "fail"
_FROZEN_0_5_CAPABILITY_IDS = frozenset(
    {
        "prismatic.cardinal_topology.0_5",
        "revolute.cardinal_topology.0_5",
    }
)
_STATIC_QUALIFICATION_SCOPE_ISSUE = 868
type FixedFullProfileAssetId = Literal[
    "joint_ref_analytic_fixed_full_profile_gate3_ready_positive"
]
_FIXED_FULL_PROFILE_ASSET_ID = cast(
    FixedFullProfileAssetId,
    get_args(FixedFullProfileAssetId.__value__)[0],
)


class CapabilityManifestError(ValueError):
    """Raised when capability policy disagrees with evidence or product surfaces."""


class _ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrozenBaselineV1(_ManifestModel):
    release_manifest: str
    release_manifest_sha256: str
    gate3_baseline: str
    gate3_baseline_sha256: str
    asset_count: int = Field(gt=0)
    joint_count: int = Field(gt=0)

    @field_validator("release_manifest", "gate3_baseline")
    @classmethod
    def _nonblank_path(cls, value: str) -> str:
        return _repository_relative_path(value, "baseline path")

    @field_validator("release_manifest_sha256", "gate3_baseline_sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        return _sha256(value, "baseline SHA-256")


class CorpusBindingV1(_ManifestModel):
    corpus_id: str
    corpus_sha256: str
    coverage_report: str
    coverage_report_sha256: str
    admission_receipt: str | None = None
    admission_receipt_sha256: str | None = None
    fixed_admission_receipt: str | None = None
    fixed_admission_receipt_sha256: str | None = None

    @field_validator("corpus_id")
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        return _nonblank(value, "corpus binding")

    @field_validator("coverage_report", "admission_receipt", "fixed_admission_receipt")
    @classmethod
    def _relative_report_path(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        labels = {
            "coverage_report": "corpus coverage report",
            "admission_receipt": "corpus admission receipt",
            "fixed_admission_receipt": "fixed corpus admission receipt",
        }
        label = labels[info.field_name]
        return _repository_relative_path(value, label)

    @field_validator(
        "corpus_sha256",
        "coverage_report_sha256",
        "admission_receipt_sha256",
        "fixed_admission_receipt_sha256",
    )
    @classmethod
    def _valid_sha256(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        labels = {
            "corpus_sha256": "corpus SHA-256",
            "coverage_report_sha256": "corpus coverage report SHA-256",
            "admission_receipt_sha256": "corpus admission receipt SHA-256",
            "fixed_admission_receipt_sha256": (
                "fixed corpus admission receipt SHA-256"
            ),
        }
        return _sha256(value, labels[info.field_name])

    @model_validator(mode="after")
    def _receipt_pair(self) -> CorpusBindingV1:
        if (self.admission_receipt is None) != (self.admission_receipt_sha256 is None):
            raise ValueError("corpus admission receipt path and digest must be paired")
        if (self.fixed_admission_receipt is None) != (
            self.fixed_admission_receipt_sha256 is None
        ):
            raise ValueError(
                "fixed corpus admission receipt path and digest must be paired"
            )
        if self.fixed_admission_receipt is not None and self.admission_receipt is None:
            raise ValueError(
                "fixed corpus admission receipt requires the controlled-distance "
                "predecessor receipt"
            )
        return self

    @model_serializer(mode="wrap")
    def _preserve_wire_shape(self, handler: Any) -> dict[str, Any]:
        document = cast(dict[str, Any], handler(self))
        if self.admission_receipt is None:
            document.pop("admission_receipt", None)
            document.pop("admission_receipt_sha256", None)
        if self.fixed_admission_receipt is None:
            document.pop("fixed_admission_receipt", None)
            document.pop("fixed_admission_receipt_sha256", None)
        return document


class EvidenceBindingV1(_ManifestModel):
    asset_id: str
    evidence_role: EvidenceRole
    artifact_key: str
    sha256: str

    @field_validator("asset_id", "artifact_key")
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        return _nonblank(value, "evidence binding")

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        return _sha256(value, "evidence SHA-256")

    @model_validator(mode="after")
    def _frozen_binding_shape(self) -> EvidenceBindingV1:
        if self.evidence_role == "frozen_0_5" and self.asset_id != "frozen_0_5":
            raise ValueError("frozen_0_5 evidence must use asset_id 'frozen_0_5'")
        if self.evidence_role != "frozen_0_5" and self.asset_id == "frozen_0_5":
            raise ValueError("only frozen_0_5 evidence may use the baseline asset ID")
        return self


class CapabilityFixturesV1(_ManifestModel):
    positive: tuple[EvidenceBindingV1, ...] = ()
    negative: tuple[EvidenceBindingV1, ...] = ()
    representative: tuple[EvidenceBindingV1, ...] = ()
    qualification_result_ids: tuple[str, ...] = ()

    @field_validator("qualification_result_ids")
    @classmethod
    def _canonical_result_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_strings(value, "qualification result ID")

    @model_validator(mode="after")
    def _unique_bindings(self) -> CapabilityFixturesV1:
        invalid_positive = [
            binding.evidence_role
            for binding in self.positive
            if binding.evidence_role not in {"analytic_positive", "frozen_0_5"}
        ]
        if invalid_positive:
            raise ValueError(
                "positive fixtures must use analytic_positive or frozen_0_5 evidence"
            )
        if any(
            binding.evidence_role != "fail_closed_negative" for binding in self.negative
        ):
            raise ValueError("negative fixtures must use fail_closed_negative evidence")
        if any(
            binding.evidence_role != "representative_source"
            for binding in self.representative
        ):
            raise ValueError(
                "representative fixtures must use representative_source evidence"
            )
        bindings = (*self.positive, *self.negative, *self.representative)
        keys = [
            (
                binding.asset_id,
                binding.evidence_role,
                binding.artifact_key,
                binding.sha256,
            )
            for binding in bindings
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("fixture evidence bindings must be unique within a row")
        if not bindings:
            raise ValueError(
                "each capability row requires at least one evidence binding"
            )
        return self


class PropertyApplicabilityV1(_ManifestModel):
    axis: FieldApplicability
    anchors_and_frames: FieldApplicability
    scalar_limit: FieldApplicability
    controls: FieldApplicability


class StaticQualificationStageAdmissionV1(_ManifestModel):
    """One indivisible validation-contract/profile/result admission."""

    validation_contract: str
    profile_id: str
    result_id: str

    @field_validator("validation_contract")
    @classmethod
    def _versioned_validation_contract(cls, value: str) -> str:
        return validate_static_validation_contract_id(value)

    @field_validator("profile_id", "result_id")
    @classmethod
    def _nonblank_identity(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)


class StaticQualificationStageBindingV1(_ManifestModel):
    """Legacy retained IDs or atomic admissions for one exact static stage.

    The official 0.5 rows retain their historical profile/result wire shape.
    Post-0.5 rows use atomic triples so independent lists cannot accidentally
    widen into a Cartesian product.  The two forms are mutually exclusive.
    """

    profile_ids: tuple[str, ...] = ()
    result_ids: tuple[str, ...] = ()
    admissions: tuple[StaticQualificationStageAdmissionV1, ...] | None = None

    @field_validator("profile_ids", "result_ids")
    @classmethod
    def _canonical_ids(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        return _canonical_strings(value, info.field_name)

    @field_validator("admissions")
    @classmethod
    def _canonical_admissions(
        cls,
        value: tuple[StaticQualificationStageAdmissionV1, ...] | None,
    ) -> tuple[StaticQualificationStageAdmissionV1, ...] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("static stage admissions must not be empty")
        keys = [
            (item.validation_contract, item.profile_id, item.result_id)
            for item in value
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("static stage admissions must be unique")
        return tuple(
            sorted(
                value,
                key=lambda item: (
                    item.validation_contract,
                    item.profile_id,
                    item.result_id,
                ),
            )
        )

    @model_validator(mode="after")
    def _exclusive_wire_shape(self) -> StaticQualificationStageBindingV1:
        if bool(self.profile_ids) != bool(self.result_ids):
            raise ValueError(
                "static stage profile and result IDs must be supplied together"
            )
        if self.admissions is not None and (self.profile_ids or self.result_ids):
            raise ValueError(
                "atomic static admissions cannot be mixed with legacy ID lists"
            )
        return self

    @model_serializer(mode="wrap")
    def _preserve_wire_shape(self, handler: Any) -> dict[str, Any]:
        document = handler(self)
        if self.admissions is None:
            document.pop("admissions", None)
        else:
            document.pop("profile_ids", None)
            document.pop("result_ids", None)
        return document

    def admitted_profile_ids(self) -> tuple[str, ...]:
        if self.admissions is None:
            return self.profile_ids
        return tuple(sorted({item.profile_id for item in self.admissions}))

    def admitted_result_ids(self) -> tuple[str, ...]:
        if self.admissions is None:
            return self.result_ids
        return tuple(sorted({item.result_id for item in self.admissions}))


class FixedFullProfileSelectorPrevalidationV1(_ManifestModel):
    """Neutral #1113 prevalidation projection; never production evidence."""

    fixture_result_sha256: str
    structural_audit_sha256: str
    validator_closeout_sha256: str
    gate3a_results_sha256: str
    gate3b_results_sha256: str
    independent_review_sha256: str
    source_sha256: str
    target_sha256: str
    authoring_reference_sha256: str
    independent_review_check_count: Literal[44]
    gate3a_target_count: Literal[2]
    gate3b_target_count: Literal[2]
    gate3b_features_passed_per_target: Literal[7]
    gate3b_features_failed_per_target: Literal[0]
    status: Literal["PASS"]
    fixture_prevalidation_only: Literal[True]
    production_execution: Literal[False]
    production_attestation: Literal[False]
    production_qualification: Literal[False]
    scoring_authority: Literal[False]
    selection_authority: Literal[False]
    capability_manifest_updated: Literal[False]

    @field_validator(
        "fixture_result_sha256",
        "structural_audit_sha256",
        "validator_closeout_sha256",
        "gate3a_results_sha256",
        "gate3b_results_sha256",
        "independent_review_sha256",
        "source_sha256",
        "target_sha256",
        "authoring_reference_sha256",
    )
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)


class FixedFullProfileFixturePrevalidationV1(FixedFullProfileSelectorPrevalidationV1):
    """Internal #1113 prevalidation receipt with repository provenance."""

    fixture_result_sha256: Literal[
        "55721d4a8b6f35ddd12ba619eac2532fd0b3b0124add723c7571eae02b171321"
    ]
    structural_audit_sha256: Literal[
        "6b372a801ae8b928ac1194de665bb970da60d657ba9d0f5071dc9f3ca9a9a02d"
    ]
    validator_closeout_sha256: Literal[
        "2285ed8ba05f052b8676c18df971a80620ef4fbffa222119a972e949c5d08dfc"
    ]
    gate3a_results_sha256: Literal[
        "bf7f67226ac0864966bc113e18ebc02679ceaba53df3b35aa7c442e3658b60ec"
    ]
    gate3b_results_sha256: Literal[
        "7e4ede3b4abc043d2279ed6c78cfc94d510426a029481ab284cb4f8c6b120b23"
    ]
    independent_review_sha256: Literal[
        "4a8cffa1778972be79c85139bf30853b73ef89ee29ce394aef6b75dc520f188a"
    ]
    source_sha256: Literal[
        "868c0d2169c0d6fd8639aa4292259203aa96fce5cdfe937b7d13e8a4b87715b3"
    ]
    target_sha256: Literal[
        "f8c80444dbb0dda9588a20b7de7aa194b1375d686333250a6527021444c0690b"
    ]
    authoring_reference_sha256: Literal[
        "67330d72ac507922eef096bdf5a48c48a72f4cf06918d2f30824f5a4f60e3a1f"
    ]
    artifact_id: str
    implementation_head_sha: str
    base_main_sha: str

    @field_validator("artifact_id")
    @classmethod
    def _artifact_id(cls, value: str) -> str:
        return _nonblank(value, "fixed prevalidation artifact ID")

    @field_validator("implementation_head_sha", "base_main_sha", mode="before")
    @classmethod
    def _git_sha(cls, value: object, info: Any) -> object:
        if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be a lowercase Git SHA")
        return value


class StaticQualificationSelectorAdmissionV1(_ManifestModel):
    """One typed selector plus its semantic and #1025 authoring authorities."""

    release_gate_asset_id: str
    selector_type: Literal["usd_prim"]
    source_joint_path: str
    constraint_kind: Literal["fixed", "distance"]
    source_semantic_preimage: SelectorSemanticPreimageV1
    source_semantic_preimage_sha256: str
    expected_joint_semantics: ArticulationV2StaticJointSemanticsV1
    expected_joint_semantics_sha256: str
    authoring_reference: ArticulationV2StaticSelectorAuthoringReferenceV1
    fixture_prevalidation: FixedFullProfileSelectorPrevalidationV1 | None = None

    @field_validator("release_gate_asset_id")
    @classmethod
    def _canonical_asset_id(cls, value: str) -> str:
        normalized = _nonblank(value, "release-gate asset ID")
        if _RELEASE_GATE_ASSET_ID_RE.fullmatch(normalized) is None:
            raise ValueError(
                "release-gate asset ID must use canonical lowercase asset-ID syntax"
            )
        return normalized

    @field_validator("source_joint_path")
    @classmethod
    def _canonical_usd_prim_path(cls, value: str) -> str:
        return _usd_prim_path(value, "source joint path")

    @field_validator("expected_joint_semantics", mode="before")
    @classmethod
    def _strict_semantic_wire_shape(cls, value: object) -> object:
        if isinstance(value, dict):
            return ArticulationV2StaticJointSemanticsV1.model_validate_json(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                strict=True,
            )
        return value

    @field_validator("source_semantic_preimage", mode="before")
    @classmethod
    def _strict_source_preimage_wire_shape(cls, value: object) -> object:
        if isinstance(value, dict):
            return SelectorSemanticPreimageV1.model_validate_json(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ),
                strict=True,
            )
        return value

    @field_validator("authoring_reference", mode="before")
    @classmethod
    def _strict_authoring_reference_wire_shape(cls, value: object) -> object:
        if isinstance(value, dict):
            return ArticulationV2StaticSelectorAuthoringReferenceV1.model_validate_json(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                strict=True,
            )
        return value

    @field_validator(
        "source_semantic_preimage_sha256",
        "expected_joint_semantics_sha256",
    )
    @classmethod
    def _valid_semantics_sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @model_validator(mode="after")
    def _fixture_prevalidation_is_scoped(
        self,
    ) -> StaticQualificationSelectorAdmissionV1:
        if self.release_gate_asset_id == _FIXED_FULL_PROFILE_ASSET_ID:
            if self.fixture_prevalidation is None:
                raise ValueError(
                    "fixed full-profile selector requires fixture prevalidation"
                )
        elif self.fixture_prevalidation is not None:
            raise ValueError(
                "fixed full-profile fixture prevalidation is defined only for "
                f"{_FIXED_FULL_PROFILE_ASSET_ID}"
            )
        return self

    @model_validator(mode="after")
    def _semantic_preimage_is_exact(
        self,
    ) -> StaticQualificationSelectorAdmissionV1:
        semantics = self.expected_joint_semantics
        source_preimage = self.source_semantic_preimage
        if (
            semantics.release_gate_asset_id != self.release_gate_asset_id
            or semantics.selector_type != self.selector_type
            or semantics.source_joint_path != self.source_joint_path
            or semantics.constraint.kind != self.constraint_kind
            or source_preimage.selector.kind != self.selector_type
            or source_preimage.selector.value != self.source_joint_path
            or source_preimage.joint.joint_type != self.constraint_kind
        ):
            raise ValueError(
                "expected joint semantic preimage differs from selector admission"
            )
        observed_source_sha256 = selector_semantic_preimage_sha256(source_preimage)
        if observed_source_sha256 != self.source_semantic_preimage_sha256:
            raise ValueError(
                "source selector semantic digest differs from its canonical preimage"
            )
        derived = derive_articulation_v2_static_joint_semantics(
            source_preimage,
            capability_id=semantics.capability_id,
            release_gate_asset_id=self.release_gate_asset_id,
            source_authority=semantics.source_authority,
        )
        if derived != semantics:
            raise ValueError(
                "articulation-v2 semantics are not the deterministic source transform"
            )
        if articulation_v2_static_joint_semantics_sha256(semantics) != (
            self.expected_joint_semantics_sha256
        ):
            raise ValueError(
                "expected joint semantic digest differs from its canonical preimage"
            )
        reference = self.authoring_reference
        if (
            reference.joint_type != self.constraint_kind
            or reference.selector.kind != self.selector_type
            or reference.selector.value != self.source_joint_path
        ):
            raise ValueError(
                "selector-scoped authoring reference differs from its admission"
            )
        if reference.parent_reference_sha256 != (
            semantics.source_authority.parent_reference.sha256
        ):
            raise ValueError(
                "selector-scoped authoring reference differs from parent authority"
            )
        if reference.semantic_digest_sha256 != observed_source_sha256:
            raise ValueError(
                "selector-scoped authoring reference differs from the source preimage"
            )
        return self

    @model_serializer(mode="wrap")
    def _preserve_wire_shape(self, handler: Any) -> dict[str, Any]:
        document = handler(self)
        if self.fixture_prevalidation is None:
            document.pop("fixture_prevalidation", None)
        return document


class ControlledDistanceFixturePrevalidationV1(_ManifestModel):
    """Immutable #1118 prevalidation identities; never production evidence."""

    retained_tree_identity_sha256: str
    primary_fixture_result_sha256: str
    structural_audit_receipt_sha256: str
    gate3a_results_sha256: str
    gate3b_results_sha256: str
    independent_postexec_receipt_sha256: str
    production_attestation: Literal[False]
    release_selection_applied: Literal[False]
    capability_manifest_updated: Literal[False]

    @field_validator(
        "retained_tree_identity_sha256",
        "primary_fixture_result_sha256",
        "structural_audit_receipt_sha256",
        "gate3a_results_sha256",
        "gate3b_results_sha256",
        "independent_postexec_receipt_sha256",
    )
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)


class ControlledDistanceAdmissionImmutableBaseV1(_ManifestModel):
    receipt: str
    receipt_sha256: str
    verified_asset_count: Literal[15]
    verified_artifact_count: Literal[289]

    @field_validator("receipt")
    @classmethod
    def _relative_receipt(cls, value: str) -> str:
        return _repository_relative_path(value, "immutable base receipt")

    @field_validator("receipt_sha256")
    @classmethod
    def _receipt_sha256(cls, value: str) -> str:
        return _sha256(value, "immutable base receipt SHA-256")


class ControlledDistanceAdmissionSuccessorV1(_ManifestModel):
    asset_id: str
    asset_manifest_sha256: str
    prepared_asset_artifact_count: Literal[27]
    corpus_artifact_identity_count: Literal[17]
    corpus_artifact_identities_sha256: str
    reference_sha256: str
    reference_dependency_bundle_sha256: str
    target_sha256: str
    target_dependency_bundle_sha256: str
    source_contract_sha256: str
    source_contract_authorer_dependency_bundle_schema_version: Literal[
        "world-understanding-usd-dependency-bundle-v3"
    ]
    source_contract_authorer_dependency_bundle_sha256: str
    source_contract_gate3_dependency_manifest_schema_version: Literal[
        "joint-agent-usd-artifact-dependency-bundle-v3"
    ]
    source_contract_gate3_dependency_manifest_sha256: str
    source_contract_gate3_dependency_manifest_entry_count: Literal[1]
    selector_semantic_preimage_sha256: str
    selector_semantics_sha256: str
    controlled_distance_contract_sha256: str
    authoring_identity_sha256: str
    retained_tree_identity_sha256: str
    primary_fixture_result_sha256: str
    structural_audit_receipt_sha256: str
    gate3a_results_sha256: str
    gate3b_results_sha256: str
    independent_postexec_receipt_sha256: str

    @field_validator("asset_id")
    @classmethod
    def _asset_id(cls, value: str) -> str:
        normalized = _nonblank(value, "controlled-distance successor asset ID")
        if _RELEASE_GATE_ASSET_ID_RE.fullmatch(normalized) is None:
            raise ValueError("successor asset ID must use canonical syntax")
        return normalized

    @field_validator(
        "asset_manifest_sha256",
        "corpus_artifact_identities_sha256",
        "reference_sha256",
        "reference_dependency_bundle_sha256",
        "target_sha256",
        "target_dependency_bundle_sha256",
        "source_contract_sha256",
        "source_contract_authorer_dependency_bundle_sha256",
        "source_contract_gate3_dependency_manifest_sha256",
        "selector_semantic_preimage_sha256",
        "selector_semantics_sha256",
        "controlled_distance_contract_sha256",
        "authoring_identity_sha256",
        "retained_tree_identity_sha256",
        "primary_fixture_result_sha256",
        "structural_audit_receipt_sha256",
        "gate3a_results_sha256",
        "gate3b_results_sha256",
        "independent_postexec_receipt_sha256",
    )
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)


class ControlledDistanceFreshSourceVerificationV1(_ManifestModel):
    method: Literal["deterministic_generate_prepare_byte_compare"]
    source_generator_sha256: str
    package_preparer_sha256: str
    preparation_spec_sha256: str
    prepared_tree_file_count: Literal[28]
    prepared_asset_artifact_count: Literal[27]
    retained_tree_mismatch_count: Literal[0]
    fresh_asset_manifest_sha256: str
    fresh_authoring_identity_sha256: str
    fresh_source_contract_sha256: str
    exact_retained_tree_bytes: Literal[True]

    @field_validator(
        "source_generator_sha256",
        "package_preparer_sha256",
        "preparation_spec_sha256",
        "fresh_asset_manifest_sha256",
        "fresh_authoring_identity_sha256",
        "fresh_source_contract_sha256",
    )
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)


class ControlledDistanceAdmissionAggregateV1(_ManifestModel):
    verified_asset_count: Literal[16]
    verified_artifact_count: Literal[306]


class ControlledDistanceAdmissionReceiptV1(_ManifestModel):
    """Complete additive #1119 identity closure; fixture-only and nonqualifying."""

    schema_version: Literal["joint-agent-controlled-distance-admission-receipt-v1"]
    issue: Literal[1119]
    operation: Literal["verify_additive_successor_against_immutable_base"]
    corpus_id: str
    corpus_sha256: str
    coverage_report_sha256: str
    immutable_base: ControlledDistanceAdmissionImmutableBaseV1
    successor: ControlledDistanceAdmissionSuccessorV1
    fresh_source_verification: ControlledDistanceFreshSourceVerificationV1
    aggregate: ControlledDistanceAdmissionAggregateV1
    source_publication_performed: Literal[False]
    immutable_object_overwrite_performed: Literal[False]
    fixture_prevalidation_only: Literal[True]
    production_attestation: Literal[False]
    static_qualified: Literal[False]
    dynamic_qualified: Literal[False]
    public_enabled: Literal[False]
    release_selection_applied: Literal[False]

    @field_validator("corpus_id")
    @classmethod
    def _corpus_id(cls, value: str) -> str:
        return _nonblank(value, "admission receipt corpus ID")

    @field_validator("corpus_sha256", "coverage_report_sha256")
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)


class FixedFullProfilePriorCorpusV1(_ManifestModel):
    """The admitted #1119 corpus snapshot that precedes the fixed rotation."""

    controlled_distance_admission_receipt: str
    controlled_distance_admission_receipt_sha256: str
    corpus_sha256: Literal[
        "6451c2fecbff4c19baad53000fec181618a7e21edc0c9f38c258cac519cc2830"
    ]
    coverage_report_sha256: Literal[
        "48aa06f94b6a30a80f60d666e6f95600021a3bf692a8f40c889e98b7ac43c94d"
    ]
    verified_asset_count: Literal[16]
    verified_artifact_count: Literal[306]

    @field_validator("controlled_distance_admission_receipt")
    @classmethod
    def _relative_receipt(cls, value: str) -> str:
        return _repository_relative_path(value, "prior controlled-distance receipt")

    @field_validator(
        "controlled_distance_admission_receipt_sha256",
        "corpus_sha256",
        "coverage_report_sha256",
    )
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)


class FixedFullProfileAdmissionSuccessorV1(_ManifestModel):
    asset_id: FixedFullProfileAssetId
    asset_manifest_sha256: str
    prepared_asset_artifact_count: Literal[25]
    corpus_artifact_identity_count: Literal[16]
    corpus_artifact_identities_sha256: str
    reference_sha256: str
    reference_dependency_bundle_sha256: str
    target_sha256: str
    target_dependency_bundle_sha256: str
    authoring_reference_sha256: str
    authoring_reference_dependency_bundle_sha256: str
    selector_semantic_preimage_sha256: str
    selector_semantics_sha256: str
    authoring_identity_sha256: str

    @field_validator(
        "asset_manifest_sha256",
        "corpus_artifact_identities_sha256",
        "reference_sha256",
        "reference_dependency_bundle_sha256",
        "target_sha256",
        "target_dependency_bundle_sha256",
        "authoring_reference_sha256",
        "authoring_reference_dependency_bundle_sha256",
        "selector_semantic_preimage_sha256",
        "selector_semantics_sha256",
        "authoring_identity_sha256",
    )
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)


class FixedFullProfileFreshSourceVerificationV1(_ManifestModel):
    method: Literal["deterministic_generate_prepare_byte_compare"]
    source_generator: str
    source_generator_sha256: str
    package_preparer: str
    package_preparer_sha256: str
    structural_auditor: str
    structural_auditor_sha256: str
    preparation_spec: str
    preparation_spec_sha256: str
    prepared_tree_file_count: Literal[26]
    prepared_asset_artifact_count: Literal[25]
    retained_tree_mismatch_count: Literal[0]
    fresh_asset_manifest_sha256: str
    fresh_authoring_identity_sha256: str
    fresh_reference_sha256: str
    fresh_target_sha256: str
    fresh_authoring_reference_sha256: str
    exact_retained_tree_bytes: Literal[True]

    @field_validator(
        "source_generator",
        "package_preparer",
        "structural_auditor",
        "preparation_spec",
    )
    @classmethod
    def _relative_source_path(cls, value: str, info: Any) -> str:
        return _repository_relative_path(value, info.field_name)

    @field_validator(
        "source_generator_sha256",
        "package_preparer_sha256",
        "structural_auditor_sha256",
        "preparation_spec_sha256",
        "fresh_asset_manifest_sha256",
        "fresh_authoring_identity_sha256",
        "fresh_reference_sha256",
        "fresh_target_sha256",
        "fresh_authoring_reference_sha256",
    )
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)


class FixedFullProfileAdmissionAggregateV1(_ManifestModel):
    verified_asset_count: int
    verified_artifact_count: int

    @field_validator("verified_asset_count", "verified_artifact_count")
    @classmethod
    def _positive_count(cls, value: int, info: Any) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value


class FixedFullProfileAdmissionReceiptV1(_ManifestModel):
    """Complete #1114 rotation; fixture-only and explicitly nonqualifying."""

    schema_version: Literal["joint-agent-fixed-full-profile-admission-receipt-v1"]
    issue: Literal[1114]
    operation: Literal["rotate_fixed_successor_against_admitted_corpus"]
    corpus_id: str
    corpus_sha256: str
    coverage_report_sha256: str
    prior_corpus: FixedFullProfilePriorCorpusV1
    successor: FixedFullProfileAdmissionSuccessorV1
    fresh_source_verification: FixedFullProfileFreshSourceVerificationV1
    fixture_prevalidation: FixedFullProfileFixturePrevalidationV1
    aggregate: FixedFullProfileAdmissionAggregateV1
    source_publication_performed: Literal[False]
    immutable_object_overwrite_performed: Literal[False]
    production_execution: Literal[False]
    production_attestation: Literal[False]
    static_qualified: Literal[False]
    dynamic_qualified: Literal[False]
    public_enabled: Literal[False]
    release_selection_applied: Literal[False]

    @field_validator("corpus_id")
    @classmethod
    def _corpus_id(cls, value: str) -> str:
        return _nonblank(value, "fixed admission receipt corpus ID")

    @field_validator("corpus_sha256", "coverage_report_sha256")
    @classmethod
    def _sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)


class StaticQualificationControlledDistanceSelectorAdmissionV2(_ManifestModel):
    """One exact #1118 selector bound to the merged controlled-distance v2 overlay."""

    admission_version: Literal["joint-agent-controlled-distance-selector-admission-v2"]
    release_gate_asset_id: str
    selector_type: Literal["usd_prim"]
    source_joint_path: str
    constraint_kind: Literal["distance"]
    source_semantic_preimage: ControlledDistanceSelectorSemanticPreimageV2
    source_semantic_preimage_sha256: str
    expected_joint_semantics: ArticulationV2ControlledDistanceSelectorSemanticsV2
    expected_joint_semantics_sha256: str
    authoring_reference: ArticulationV2ControlledDistanceSelectorAuthoringReferenceV2
    fixture_prevalidation: ControlledDistanceFixturePrevalidationV1
    release_selection_applied: Literal[False]

    @field_validator("release_gate_asset_id")
    @classmethod
    def _canonical_asset_id(cls, value: str) -> str:
        normalized = _nonblank(value, "release-gate asset ID")
        if _RELEASE_GATE_ASSET_ID_RE.fullmatch(normalized) is None:
            raise ValueError(
                "release-gate asset ID must use canonical lowercase asset-ID syntax"
            )
        return normalized

    @field_validator("source_joint_path")
    @classmethod
    def _canonical_usd_prim_path(cls, value: str) -> str:
        return _usd_prim_path(value, "source joint path")

    @field_validator("source_semantic_preimage", mode="before")
    @classmethod
    def _strict_source_preimage_wire_shape(cls, value: object) -> object:
        if isinstance(value, dict):
            return ControlledDistanceSelectorSemanticPreimageV2.model_validate_json(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ),
                strict=True,
            )
        return value

    @field_validator("expected_joint_semantics", mode="before")
    @classmethod
    def _strict_semantic_wire_shape(cls, value: object) -> object:
        if isinstance(value, dict):
            return (
                ArticulationV2ControlledDistanceSelectorSemanticsV2.model_validate_json(
                    json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    strict=True,
                )
            )
        return value

    @field_validator("authoring_reference", mode="before")
    @classmethod
    def _strict_authoring_reference_wire_shape(cls, value: object) -> object:
        if isinstance(value, dict):
            return ArticulationV2ControlledDistanceSelectorAuthoringReferenceV2.model_validate_json(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                strict=True,
            )
        return value

    @field_validator(
        "source_semantic_preimage_sha256",
        "expected_joint_semantics_sha256",
    )
    @classmethod
    def _valid_semantics_sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @model_validator(mode="after")
    def _one_exact_successor(
        self,
    ) -> StaticQualificationControlledDistanceSelectorAdmissionV2:
        preimage = self.source_semantic_preimage
        semantics = self.expected_joint_semantics
        observed_source_sha256 = controlled_distance_selector_preimage_sha256(preimage)
        if (
            semantics.release_gate_asset_id != self.release_gate_asset_id
            or semantics.selector_type != self.selector_type
            or semantics.source_joint_path != self.source_joint_path
            or preimage.selector.kind != self.selector_type
            or preimage.selector.value != self.source_joint_path
            or preimage.joint.joint_type != "prismatic"
            or observed_source_sha256 != self.source_semantic_preimage_sha256
        ):
            raise ValueError(
                "controlled-distance semantic preimage differs from selector admission"
            )
        derived = derive_articulation_v2_controlled_distance_selector_semantics(
            preimage,
            release_gate_asset_id=self.release_gate_asset_id,
            source_authority=semantics.source_authority,
            source_contract=semantics.source_contract,
            controlled_distance_contract=semantics.controlled_distance_contract,
        )
        if derived != semantics:
            raise ValueError(
                "controlled-distance semantics are not the deterministic source transform"
            )
        if (
            articulation_v2_controlled_distance_selector_semantics_sha256(semantics)
            != self.expected_joint_semantics_sha256
        ):
            raise ValueError(
                "controlled-distance semantic digest differs from its canonical preimage"
            )
        reference = self.authoring_reference
        if (
            reference.selector.kind != self.selector_type
            or reference.selector.value != self.source_joint_path
            or reference.parent_reference_sha256
            != semantics.source_authority.parent_reference.sha256
            or reference.semantic_digest_sha256 != observed_source_sha256
        ):
            raise ValueError(
                "controlled-distance authoring reference differs from its admission"
            )
        return self


type StaticQualificationSelectorAdmission = (
    StaticQualificationSelectorAdmissionV1
    | StaticQualificationControlledDistanceSelectorAdmissionV2
)


class StaticQualificationStageBindingsV1(_ManifestModel):
    """Stage-specific admission; IDs cannot be exchanged across validators."""

    contract: StaticQualificationStageBindingV1
    authoring: StaticQualificationStageBindingV1
    readback: StaticQualificationStageBindingV1
    gate3a: StaticQualificationStageBindingV1
    gate3b: StaticQualificationStageBindingV1

    @model_validator(mode="after")
    def _atomic_contracts_are_lane_and_stage_exact(
        self,
    ) -> StaticQualificationStageBindingsV1:
        for stage in ("contract", "authoring", "readback", "gate3a", "gate3b"):
            admissions = getattr(self, stage).admissions
            if admissions is None:
                continue
            expected = static_validation_contract(ARTICULATION_V2_CONTRACT_LANE, stage)
            if any(item.validation_contract != expected for item in admissions):
                raise ValueError(
                    f"{stage} atomic admissions must use articulation-v2 contract "
                    f"{expected!r}"
                )
        return self

    def profile_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    profile_id
                    for stage in (
                        self.contract,
                        self.authoring,
                        self.readback,
                        self.gate3a,
                        self.gate3b,
                    )
                    for profile_id in stage.admitted_profile_ids()
                }
            )
        )

    def result_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    result_id
                    for stage in (
                        self.contract,
                        self.authoring,
                        self.readback,
                        self.gate3a,
                        self.gate3b,
                    )
                    for result_id in stage.admitted_result_ids()
                }
            )
        )


class StaticOutcomeAttestationV1(_ManifestModel):
    """Bind a recorded static outcome to the run that produced it.

    Without this a lane could move from ``scheduled`` to ``pass`` as a one-word
    edit carrying no new identity material. Naming the scorecard, run plan, run,
    manifest digest, and row admission digest the run was admitted against ties
    the outcome to specific artifacts.

    This is a binding, not a proof of possession: every digest here is
    recomputable by anyone holding the artifacts. What it buys is that a
    recorded outcome cannot silently outlive the plan, manifest state, or
    scorecard it describes.

    ``capability_manifest_sha256`` is the manifest digest the run executed
    under -- necessarily the digest *before* this attestation was written back,
    which is why it must be recorded rather than recomputed.
    """

    scorecard_sha256: str
    run_plan_sha256: str
    capability_manifest_sha256: str
    run_id: str
    row_admission_sha256: str

    @field_validator(
        "scorecard_sha256",
        "run_plan_sha256",
        "capability_manifest_sha256",
        "row_admission_sha256",
    )
    @classmethod
    def _valid_sha256(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("run_id")
    @classmethod
    def _nonblank_run_id(cls, value: str) -> str:
        return _nonblank(value, "attestation run ID")


class QualificationBindingV1(_ManifestModel):
    static_status: QualificationLaneStatus
    static_stages: StaticQualificationStageBindingsV1
    selector_admissions: tuple[StaticQualificationSelectorAdmission, ...] | None = None
    static_attestation: StaticOutcomeAttestationV1 | None = None
    dynamic_status: QualificationStatus
    dynamic_profile_ids: tuple[str, ...] = ()
    dynamic_result_ids: tuple[str, ...] = ()

    @field_validator(
        "dynamic_profile_ids",
        "dynamic_result_ids",
    )
    @classmethod
    def _canonical_ids(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        return _canonical_strings(value, info.field_name)

    @field_validator("selector_admissions")
    @classmethod
    def _canonical_selector_admissions(
        cls,
        value: tuple[StaticQualificationSelectorAdmission, ...] | None,
    ) -> tuple[StaticQualificationSelectorAdmission, ...] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("static selector admissions must not be empty")
        keys = [
            (
                item.release_gate_asset_id,
                item.selector_type,
                item.source_joint_path,
                item.constraint_kind,
            )
            for item in value
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("static selector admissions must be unique")
        return tuple(
            sorted(
                value,
                key=lambda item: (
                    item.release_gate_asset_id,
                    item.selector_type,
                    item.source_joint_path,
                    item.constraint_kind,
                ),
            )
        )

    @model_validator(mode="after")
    def _result_binding(self) -> QualificationBindingV1:
        _validate_status_binding(
            self.static_status,
            self.static_profile_ids,
            self.static_result_ids,
            "static",
        )
        _validate_status_binding(
            self.dynamic_status,
            self.dynamic_profile_ids,
            self.dynamic_result_ids,
            "dynamic",
        )
        if self.static_attestation is not None and self.static_status not in {
            "pass",
            "fail",
        }:
            raise ValueError(
                "static attestation is only meaningful for a recorded pass/fail "
                f"outcome; got {self.static_status!r}"
            )
        return self

    @model_serializer(mode="wrap")
    def _preserve_legacy_wire_shape(self, handler: Any) -> dict[str, Any]:
        document = handler(self)
        if self.selector_admissions is None:
            document.pop("selector_admissions", None)
        return document

    @property
    def static_profile_ids(self) -> tuple[str, ...]:
        """Return the aggregate for fixture auditing, never stage admission."""

        return self.static_stages.profile_ids()

    @property
    def static_result_ids(self) -> tuple[str, ...]:
        """Return the aggregate for fixture auditing, never stage admission."""

        return self.static_stages.result_ids()


class ScopeRestrictionV1(_ManifestModel):
    source_families: tuple[str, ...] = ()
    asset_archetypes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @field_validator("source_families", "asset_archetypes", "notes")
    @classmethod
    def _canonical_values(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        return _canonical_strings(value, info.field_name)


class CapabilityRowV1(_ManifestModel):
    capability_id: str
    roadmap_issues: tuple[int, ...]
    canonical_type: str
    property_profile: str
    accepted_input_aliases: tuple[str, ...]
    contract_versions: tuple[str, ...]
    articulation_v2_constraint_kinds: tuple[str, ...] = ()
    candidate_artifact_versions: tuple[str, ...]
    properties: PropertyApplicabilityV1
    support_level: SupportLevel
    disposition: CapabilityDisposition
    public_exposure: PublicExposure
    authoring_backends: tuple[str, ...]
    reason_codes: tuple[str, ...]
    fixtures: CapabilityFixturesV1
    qualification: QualificationBindingV1
    scope: ScopeRestrictionV1

    # Advertisement fields state what a row CLAIMS; everything else states what
    # it was admitted to run. Evidence binds to the latter so that recording one
    # row's outcome cannot invalidate another row's in-flight evidence -- the
    # whole-manifest digest could not distinguish the two.
    _ADVERTISEMENT_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "support_level",
            "disposition",
            "public_exposure",
            "reason_codes",
        }
    )

    @property
    def admission_sha256(self) -> str:
        """Digest the row's admitted identity, eliding what it advertises.

        Stage bindings, authoring backends, fixtures, and scope stay in: those
        are what a run is admitted against. Support level, disposition, public
        exposure, reason codes, and the lane's own status/attestation stay out,
        so a row can record an outcome without rotating the digest its evidence
        was bound to.
        """

        document = json.loads(self.model_dump_json())
        for field in self._ADVERTISEMENT_FIELDS:
            document.pop(field, None)
        qualification = document.get("qualification")
        if isinstance(qualification, dict):
            qualification.pop("static_status", None)
            qualification.pop("dynamic_status", None)
            qualification.pop("static_attestation", None)
        payload = json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @field_validator("capability_id", "canonical_type", "property_profile")
    @classmethod
    def _nonblank_required_text(cls, value: str) -> str:
        return _nonblank(value, "capability row")

    @field_validator(
        "accepted_input_aliases",
        "contract_versions",
        "articulation_v2_constraint_kinds",
        "candidate_artifact_versions",
        "authoring_backends",
        "reason_codes",
    )
    @classmethod
    def _canonical_strings(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        return _canonical_strings(value, info.field_name)

    @field_validator("roadmap_issues")
    @classmethod
    def _canonical_issues(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(issue <= 0 for issue in value):
            raise ValueError("roadmap_issues must contain positive issue numbers")
        if len(value) != len(set(value)):
            raise ValueError("roadmap_issues must not contain duplicates")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _support_invariants(self) -> CapabilityRowV1:
        support_rank = _SUPPORT_RANK[self.support_level]
        has_articulation_v2 = ARTICULATION_V2_CONTRACT_VERSION in self.contract_versions
        if has_articulation_v2 != bool(self.articulation_v2_constraint_kinds):
            raise ValueError(
                "articulation-v2 rows require contract registration and at least "
                "one owned constraint kind"
            )
        if (
            support_rank >= _SUPPORT_RANK["contract_ready"]
            and not self.contract_versions
        ):
            raise ValueError("contract-ready rows require a contract version")
        if support_rank >= _SUPPORT_RANK["authorable"] and not self.authoring_backends:
            raise ValueError("authorable rows require an authoring backend")
        # The immutable 0.5 lane predates attestations, so it is grandfathered
        # explicitly rather than through a general "attestation optional"
        # escape. The exemption is pinned to its exact historical outcome:
        # keying it on capability ID alone would let a frozen row move
        # `fail -> pass` unattested, leaving the immutable lane holding the one
        # word edit path this rule removes everywhere else.
        legacy_0_5 = (
            self.capability_id in _FROZEN_0_5_CAPABILITY_IDS
            and self.qualification.static_status == _FROZEN_0_5_STATIC_STATUS
        )
        stages = (
            self.qualification.static_stages.contract,
            self.qualification.static_stages.authoring,
            self.qualification.static_stages.readback,
            self.qualification.static_stages.gate3a,
            self.qualification.static_stages.gate3b,
        )
        selector_admissions = self.qualification.selector_admissions
        atomic_stage_admissions = tuple(stage.admissions for stage in stages)
        if legacy_0_5 and (
            selector_admissions is not None
            or any(admissions is not None for admissions in atomic_stage_admissions)
        ):
            raise ValueError(
                "the immutable 0.5 lane must retain its legacy static wire shape"
            )
        if selector_admissions is not None:
            if self.capability_id not in {
                "fixed.explicit_two_body_constraint",
                "distance.bounded_two_body_constraint",
            }:
                raise ValueError(
                    "articulation-v2 static semantic admission is currently "
                    "defined only for fixed and distance rows"
                )
            if self.qualification.static_status not in {"scheduled", "pass", "fail"}:
                raise ValueError(
                    "static selector admissions require a scheduled or recorded lane"
                )
            if len(selector_admissions) != 1:
                raise ValueError(
                    "fixed/distance rows require one exact static selector admission"
                )
            selector = selector_admissions[0]
            if (
                selector.expected_joint_semantics.capability_id != self.capability_id
                or selector.constraint_kind not in self.articulation_v2_constraint_kinds
            ):
                raise ValueError(
                    "static selector semantics disagree with their capability row"
                )
            if any(
                admissions is None or len(admissions) != 1
                for admissions in atomic_stage_admissions
            ):
                raise ValueError(
                    "fixed/distance static admission requires one atomic entry "
                    "for each of the five stages"
                )
        elif any(admissions is not None for admissions in atomic_stage_admissions):
            raise ValueError(
                "atomic static stage admissions require a typed selector admission"
            )
        attestation = self.qualification.static_attestation
        if (
            self.qualification.static_status in {"pass", "fail"}
            and attestation is None
            and not legacy_0_5
        ):
            raise ValueError(
                "a recorded static outcome requires an attestation naming the "
                "scorecard, run plan, run, and row admission digest that "
                "produced it"
            )
        if attestation is not None and (
            attestation.row_admission_sha256 != self.admission_sha256
        ):
            raise ValueError(
                "static attestation is bound to a different row admission "
                "digest than this row"
            )
        if self.support_level == "static_qualified":
            if self.qualification.static_status != "pass":
                raise ValueError(
                    "static-qualified rows require passing static evidence"
                )
            # No separate attestation check here: reaching this point requires
            # static_status == "pass", and the rule above already rejects a
            # non-frozen pass that carries no attestation. A second check would
            # be unreachable rather than defensive.
        if self.support_level == "dynamic_qualified":
            if (
                self.qualification.static_status != "pass"
                or self.qualification.dynamic_status != "pass"
            ):
                raise ValueError(
                    "dynamic-qualified rows require passing static and dynamic evidence"
                )
        if self.public_exposure == "enabled" and self.disposition != "supported":
            raise ValueError("enabled public rows must have supported disposition")
        if self.disposition != "supported" and not self.reason_codes:
            raise ValueError("gated capability rows require stable reason codes")
        if (
            self.disposition == "supported"
            and support_rank < _SUPPORT_RANK["authorable"]
        ):
            raise ValueError("supported rows require at least authorable support level")
        bound_result_ids = frozenset(
            (
                *self.qualification.static_result_ids,
                *self.qualification.dynamic_result_ids,
            )
        )
        if frozenset(self.fixtures.qualification_result_ids) != bound_result_ids:
            raise ValueError(
                "fixture qualification result IDs must exactly match "
                "static and dynamic result bindings"
            )
        return self


class CapabilityManifestV1(_ManifestModel):
    schema_version: Literal["joint-agent-capability-manifest-v3"]
    manifest_version: str
    frozen_0_5: FrozenBaselineV1
    corpus: CorpusBindingV1
    selected_public_capability_ids: tuple[str, ...]
    capabilities: tuple[CapabilityRowV1, ...]

    @field_validator("manifest_version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        normalized = _nonblank(value, "manifest_version")
        if _VERSION_RE.fullmatch(normalized) is None:
            raise ValueError("manifest_version must be a semantic version")
        return normalized

    @field_validator("selected_public_capability_ids")
    @classmethod
    def _canonical_public_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_strings(value, "selected public capability ID")

    @field_validator("capabilities")
    @classmethod
    def _canonical_capabilities(
        cls,
        value: tuple[CapabilityRowV1, ...],
    ) -> tuple[CapabilityRowV1, ...]:
        ids = [row.capability_id for row in value]
        if len(ids) != len(set(ids)):
            raise ValueError("capability IDs must be unique")
        return tuple(sorted(value, key=lambda row: row.capability_id))

    @model_validator(mode="after")
    def _public_selection(self) -> CapabilityManifestV1:
        enabled_rows = tuple(
            row for row in self.capabilities if row.public_exposure == "enabled"
        )
        enabled = tuple(sorted(row.capability_id for row in enabled_rows))
        if self.selected_public_capability_ids != enabled:
            raise ValueError(
                "selected_public_capability_ids must exactly match enabled rows"
            )
        enabled_types = [row.canonical_type for row in enabled_rows]
        if len(enabled_types) != len(set(enabled_types)):
            raise ValueError(
                "at most one public capability row may be enabled per canonical type"
            )
        if any(
            _SUPPORT_RANK[row.support_level] < _SUPPORT_RANK["authorable"]
            for row in enabled_rows
        ):
            raise ValueError("enabled public rows must be authorable or qualified")
        for row in self.capabilities:
            _validate_frozen_row_evidence(row, self.frozen_0_5)
        v2_kind_owners: dict[str, str] = {}
        for row in self.capabilities:
            for kind in row.articulation_v2_constraint_kinds:
                existing = v2_kind_owners.get(kind)
                if existing is not None:
                    raise ValueError(
                        "articulation-v2 constraint kinds require one capability "
                        f"decision: {kind!r} is owned by {existing!r} and "
                        f"{row.capability_id!r}"
                    )
                v2_kind_owners[kind] = row.capability_id
        has_controlled_distance_admission = any(
            isinstance(
                admission,
                StaticQualificationControlledDistanceSelectorAdmissionV2,
            )
            for row in self.capabilities
            for admission in (row.qualification.selector_admissions or ())
        )
        if has_controlled_distance_admission and (
            self.corpus.admission_receipt is None
            or self.corpus.admission_receipt_sha256 is None
        ):
            raise ValueError(
                "controlled-distance v2 admission requires its exact corpus receipt"
            )
        has_fixed_full_profile_admission = any(
            isinstance(admission, StaticQualificationSelectorAdmissionV1)
            and admission.release_gate_asset_id == _FIXED_FULL_PROFILE_ASSET_ID
            for row in self.capabilities
            for admission in (row.qualification.selector_admissions or ())
        )
        has_fixed_receipt = (
            self.corpus.fixed_admission_receipt is not None
            and self.corpus.fixed_admission_receipt_sha256 is not None
        )
        if has_fixed_full_profile_admission != has_fixed_receipt:
            raise ValueError(
                "fixed full-profile admission and its exact corpus receipt must be paired"
            )
        return self


@dataclass(frozen=True)
class LoadedCapabilityManifest:
    manifest: CapabilityManifestV1
    sha256: str


def load_capability_manifest(
    path: str | Path | None = None,
) -> LoadedCapabilityManifest:
    """Load and strictly validate the checked-in capability manifest."""

    if path is None:
        payload = (
            resources.files("joint_agent")
            .joinpath(CAPABILITY_MANIFEST_RESOURCE)
            .read_bytes()
        )
    else:
        payload = Path(path).read_bytes()
    document = loads_json_document(payload, label="capability manifest")
    try:
        manifest = CapabilityManifestV1.model_validate(document)
    except ValueError as exc:
        raise CapabilityManifestError(f"capability manifest is invalid: {exc}") from exc
    return LoadedCapabilityManifest(
        manifest=manifest,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def require_static_selector_admission(
    loaded: LoadedCapabilityManifest,
    *,
    capability_id: str,
    admission: StaticQualificationSelectorAdmission,
) -> StaticQualificationSelectorAdmission:
    """Resolve one exact in-memory selector admission without any I/O."""

    if not isinstance(loaded, LoadedCapabilityManifest):
        raise CapabilityManifestError(
            "static selector authority must be a loaded capability manifest"
        )
    if type(admission) not in {
        StaticQualificationSelectorAdmissionV1,
        StaticQualificationControlledDistanceSelectorAdmissionV2,
    }:
        raise CapabilityManifestError(
            "static selector request must be an exact selector admission"
        )
    try:
        normalized_capability_id = _nonblank(capability_id, "capability ID")
    except ValueError as exc:
        raise CapabilityManifestError(
            f"static selector request is invalid: {exc}"
        ) from exc
    row = _capability_row(loaded, normalized_capability_id)
    admitted = row.qualification.selector_admissions
    if admitted is None or admission not in admitted:
        raise CapabilityManifestError(
            "static selector, URI authority, or semantic preimage does not "
            f"exactly match the manifest admission for {normalized_capability_id!r}"
        )
    return admission


def require_static_stage_admission(
    loaded: LoadedCapabilityManifest,
    *,
    capability_id: str,
    stage: StaticQualificationStageName,
    admission: StaticQualificationStageAdmissionV1,
) -> StaticQualificationStageAdmissionV1:
    """Resolve one exact atomic stage admission without Cartesian widening."""

    if not isinstance(loaded, LoadedCapabilityManifest):
        raise CapabilityManifestError(
            "static stage authority must be a loaded capability manifest"
        )
    if stage not in {"contract", "authoring", "readback", "gate3a", "gate3b"}:
        raise CapabilityManifestError(f"unknown static qualification stage: {stage!r}")
    if type(admission) is not StaticQualificationStageAdmissionV1:
        raise CapabilityManifestError(
            "static stage request must be an exact atomic admission"
        )
    try:
        normalized_capability_id = _nonblank(capability_id, "capability ID")
    except ValueError as exc:
        raise CapabilityManifestError(
            f"static stage request is invalid: {exc}"
        ) from exc
    row = _capability_row(loaded, normalized_capability_id)
    binding = getattr(row.qualification.static_stages, stage)
    if binding.admissions is None or admission not in binding.admissions:
        raise CapabilityManifestError(
            "static validation contract/profile/result triple is not admitted "
            f"for {normalized_capability_id!r}/{stage}"
        )
    return admission


def validate_corpus_bindings(
    loaded: LoadedCapabilityManifest,
    *,
    repo_root: str | Path,
) -> None:
    """Verify every manifest evidence binding against the exact #871 report."""

    assets = _load_validated_corpus_assets(loaded, repo_root=repo_root)
    qualification_roster = qualification_evidence_roster(
        loaded,
        repo_root=repo_root,
    )
    qualification_asset_ids = frozenset(
        binding.asset_id for binding in qualification_roster
    )
    for row in loaded.manifest.capabilities:
        _validate_row_evidence(row, assets, loaded.manifest.frozen_0_5)
        _validate_static_selector_bindings(
            row,
            assets,
            qualification_asset_ids=qualification_asset_ids,
        )


def _capability_row(
    loaded: LoadedCapabilityManifest,
    capability_id: str,
) -> CapabilityRowV1:
    row = next(
        (
            candidate
            for candidate in loaded.manifest.capabilities
            if candidate.capability_id == capability_id
        ),
        None,
    )
    if row is None:
        raise CapabilityManifestError(
            f"capability manifest has no row for {capability_id!r}"
        )
    return row


def _load_validated_corpus_assets(
    loaded: LoadedCapabilityManifest,
    *,
    repo_root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Admit assets only from the manifest-bound corpus coverage report."""

    root = Path(repo_root)
    manifest = loaded.manifest
    coverage_path = resolve_repository_path(
        root,
        manifest.corpus.coverage_report,
        label="corpus coverage report",
    )
    coverage_bytes = coverage_path.read_bytes()
    coverage_sha256 = hashlib.sha256(coverage_bytes).hexdigest()
    if coverage_sha256 != manifest.corpus.coverage_report_sha256:
        raise CapabilityManifestError(
            "corpus coverage report SHA-256 does not match the capability manifest"
        )
    coverage = loads_json_document(coverage_bytes, label="corpus coverage report")
    if coverage.get("schema_version") != CORPUS_COVERAGE_SCHEMA_VERSION:
        raise CapabilityManifestError(
            "corpus coverage report must use the qualification-authority schema "
            f"{CORPUS_COVERAGE_SCHEMA_VERSION!r}"
        )
    if coverage.get("corpus_id") != manifest.corpus.corpus_id:
        raise CapabilityManifestError(
            "corpus ID does not match the capability manifest"
        )
    if coverage.get("corpus_sha256") != manifest.corpus.corpus_sha256:
        raise CapabilityManifestError(
            "corpus semantic SHA-256 does not match the capability manifest"
        )
    controlled_admissions = tuple(
        admission
        for row in manifest.capabilities
        for admission in (row.qualification.selector_admissions or ())
        if isinstance(
            admission,
            StaticQualificationControlledDistanceSelectorAdmissionV2,
        )
    )
    fixed_full_profile_admissions = tuple(
        admission
        for row in manifest.capabilities
        for admission in (row.qualification.selector_admissions or ())
        if isinstance(admission, StaticQualificationSelectorAdmissionV1)
        and admission.release_gate_asset_id == _FIXED_FULL_PROFILE_ASSET_ID
    )
    receipt_path = manifest.corpus.admission_receipt
    receipt_sha256 = manifest.corpus.admission_receipt_sha256
    typed_receipt: ControlledDistanceAdmissionReceiptV1 | None = None
    if receipt_path is not None and receipt_sha256 is not None:
        resolved_receipt = resolve_repository_path(
            root,
            receipt_path,
            label="corpus admission receipt",
        )
        receipt_bytes = resolved_receipt.read_bytes()
        if hashlib.sha256(receipt_bytes).hexdigest() != receipt_sha256:
            raise CapabilityManifestError(
                "corpus admission receipt SHA-256 does not match the capability manifest"
            )
        raw_receipt = loads_json_document(
            receipt_bytes,
            label="corpus admission receipt",
        )
        try:
            typed_receipt = ControlledDistanceAdmissionReceiptV1.model_validate(
                raw_receipt
            )
        except ValueError as exc:
            raise CapabilityManifestError(
                "corpus admission receipt is not one strict #1119 identity closure: "
                f"{exc}"
            ) from exc
    fixed_receipt_path = manifest.corpus.fixed_admission_receipt
    fixed_receipt_sha256 = manifest.corpus.fixed_admission_receipt_sha256
    typed_fixed_receipt: FixedFullProfileAdmissionReceiptV1 | None = None
    if fixed_receipt_path is not None and fixed_receipt_sha256 is not None:
        resolved_fixed_receipt = resolve_repository_path(
            root,
            fixed_receipt_path,
            label="fixed corpus admission receipt",
        )
        fixed_receipt_bytes = resolved_fixed_receipt.read_bytes()
        if hashlib.sha256(fixed_receipt_bytes).hexdigest() != fixed_receipt_sha256:
            raise CapabilityManifestError(
                "fixed corpus admission receipt SHA-256 does not match the "
                "capability manifest"
            )
        raw_fixed_receipt = loads_json_document(
            fixed_receipt_bytes,
            label="fixed corpus admission receipt",
        )
        try:
            typed_fixed_receipt = FixedFullProfileAdmissionReceiptV1.model_validate(
                raw_fixed_receipt
            )
        except ValueError as exc:
            raise CapabilityManifestError(
                "fixed corpus admission receipt is not one strict #1114 identity "
                f"closure: {exc}"
            ) from exc
        if (
            typed_fixed_receipt.corpus_id != manifest.corpus.corpus_id
            or typed_fixed_receipt.corpus_sha256 != manifest.corpus.corpus_sha256
            or typed_fixed_receipt.coverage_report_sha256
            != manifest.corpus.coverage_report_sha256
        ):
            raise CapabilityManifestError(
                "fixed corpus admission receipt differs from its bound corpus identities"
            )
    if (
        typed_receipt is not None
        and typed_receipt.corpus_id != manifest.corpus.corpus_id
    ):
        raise CapabilityManifestError(
            "corpus admission receipt differs from its bound corpus ID"
        )
    if typed_receipt is not None and typed_fixed_receipt is None:
        if (
            typed_receipt.corpus_sha256 != manifest.corpus.corpus_sha256
            or typed_receipt.coverage_report_sha256
            != manifest.corpus.coverage_report_sha256
        ):
            raise CapabilityManifestError(
                "corpus admission receipt differs from its bound corpus identities"
            )
    if controlled_admissions and typed_receipt is None:
        raise CapabilityManifestError(
            "controlled-distance v2 admission and corpus receipt must be paired"
        )
    if typed_receipt is not None and not controlled_admissions:
        raise CapabilityManifestError(
            "controlled-distance corpus receipt requires one active v2 admission"
        )
    if len(controlled_admissions) > 1:
        raise CapabilityManifestError(
            "corpus admission receipt must bind one controlled-distance admission"
        )
    if bool(fixed_full_profile_admissions) != bool(typed_fixed_receipt):
        raise CapabilityManifestError(
            "fixed full-profile admission and corpus receipt must be paired"
        )
    if len(fixed_full_profile_admissions) > 1:
        raise CapabilityManifestError(
            "fixed corpus admission receipt must bind one full-profile admission"
        )
    _validate_frozen_baseline(manifest.frozen_0_5, coverage, root)

    raw_assets = coverage.get("assets")
    if not isinstance(raw_assets, list):
        raise CapabilityManifestError("corpus coverage assets must be a list")
    assets: dict[str, dict[str, Any]] = {}
    for asset in raw_assets:
        if not isinstance(asset, dict):
            raise CapabilityManifestError("corpus coverage asset rows must be objects")
        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise CapabilityManifestError(
                "corpus coverage asset rows require nonblank asset IDs"
            )
        if asset_id in assets:
            raise CapabilityManifestError(
                f"corpus coverage asset ID is duplicated: {asset_id}"
            )
        if asset.get("schema_version") == PREPARED_ASSET_MANIFEST_SCHEMA_VERSION:
            raise CapabilityManifestError(
                "prepared asset manifests are non-qualifying reproduction metadata; "
                "resolve qualification truth from the corpus coverage row"
            )
        if asset.get("preparation_state") == "prepared":
            assets[asset_id] = _validate_corpus_qualification_row(asset)
        else:
            if "qualification_truth" in asset:
                raise CapabilityManifestError(
                    "unprepared corpus assets cannot claim qualification authority: "
                    f"{asset_id}"
                )
            assets[asset_id] = asset
    if typed_receipt is not None:
        _validate_controlled_distance_admission_receipt(
            typed_receipt,
            admission=controlled_admissions[0],
            assets=assets,
            repo_root=root,
            additional_successor_asset_ids=(
                frozenset({_FIXED_FULL_PROFILE_ASSET_ID})
                if typed_fixed_receipt is not None
                else frozenset()
            ),
        )
    if typed_fixed_receipt is not None:
        # Defensive for programmatically constructed containers; the public
        # manifest schema normally enforces this predecessor pairing earlier.
        if typed_receipt is None:
            raise CapabilityManifestError(
                "fixed corpus admission receipt requires its predecessor receipt"
            )
        _validate_fixed_full_profile_admission_receipt(
            typed_fixed_receipt,
            admission=fixed_full_profile_admissions[0],
            controlled_receipt=typed_receipt,
            assets=assets,
            manifest=manifest,
            repo_root=root,
        )
    return assets


_BASE_RECEIPT_NON_REGISTRY_ARTIFACT_ROLES = frozenset(
    {
        "articulation_candidates",
        "source_file",
        "source_preparation_spec",
    }
)


def _corpus_artifact_identity_records(
    asset_id: str,
    identities: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Flatten every URI-bound corpus identity without counting bundle nodes."""

    records: list[dict[str, Any]] = []

    def visit(value: object, role: str) -> None:
        if isinstance(value, dict):
            if "uri" in value and "sha256" in value:
                record: dict[str, Any] = {
                    "asset_id": asset_id,
                    "role": role,
                    "uri": value["uri"],
                    "sha256": value["sha256"],
                }
                if "dependency_bundle" in value:
                    record["dependency_bundle"] = value["dependency_bundle"]
                if "status" in value:
                    record["status"] = value["status"]
                records.append(record)
                return
            for key, item in value.items():
                if not role:
                    nested_role = key
                elif role == "selector_authoring_references":
                    nested_role = f"{role}[{key!r}]"
                else:
                    nested_role = f"{role}.{key}"
                visit(item, nested_role)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{role}[{index}]")

    visit(identities, "")
    keys = [(record["asset_id"], record["role"], record["uri"]) for record in records]
    if len(set(keys)) != len(keys):
        raise CapabilityManifestError(
            f"corpus asset {asset_id!r} has duplicate artifact identities"
        )
    return tuple(
        sorted(
            records,
            key=lambda record: (
                str(record["asset_id"]),
                str(record["role"]),
                str(record["uri"]),
            ),
        )
    )


def _corpus_artifact_identities_sha256(identities: dict[str, Any]) -> str:
    payload = json.dumps(
        identities,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_immutable_base_artifact_identities(
    base_document: dict[str, Any],
    *,
    assets: dict[str, dict[str, Any]],
    successor_asset_id: str,
    additional_successor_asset_ids: frozenset[str] = frozenset(),
) -> None:
    """Compare every frozen registry identity with the immutable base receipt."""

    raw_artifacts = base_document.get("verified_artifacts")
    if not isinstance(raw_artifacts, list):
        raise CapabilityManifestError(
            "immutable base receipt lacks its verified artifact inventory"
        )
    expected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in raw_artifacts:
        if not isinstance(item, dict):
            raise CapabilityManifestError(
                "immutable base receipt artifact inventory is malformed"
            )
        role = item.get("role")
        if role in _BASE_RECEIPT_NON_REGISTRY_ARTIFACT_ROLES:
            continue
        asset_id = item.get("asset_id")
        uri = item.get("uri")
        if not all(isinstance(value, str) for value in (asset_id, role, uri)):
            raise CapabilityManifestError(
                "immutable base receipt artifact identity is malformed"
            )
        key = (asset_id, role, uri)
        if key in expected:
            raise CapabilityManifestError(
                "immutable base receipt duplicates a registry artifact identity"
            )
        expected[key] = {
            field: item[field] for field in ("asset_id", "role", "uri", "sha256")
        }
        for field in ("dependency_bundle", "status"):
            if field in item:
                expected[key][field] = item[field]

    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for asset_id, asset in assets.items():
        if (
            asset_id == successor_asset_id
            or asset_id in additional_successor_asset_ids
            or asset.get("preparation_state") != "prepared"
        ):
            continue
        identities = asset.get("artifact_identities")
        if not isinstance(identities, dict):
            raise CapabilityManifestError(
                f"immutable base asset {asset_id!r} lacks artifact identities"
            )
        for record in _corpus_artifact_identity_records(asset_id, identities):
            key = (record["asset_id"], record["role"], record["uri"])
            observed[key] = record

    if observed != expected:
        raise CapabilityManifestError(
            "corpus immutable-base artifact identities differ from the frozen receipt"
        )


def _validate_controlled_distance_admission_receipt(
    receipt: ControlledDistanceAdmissionReceiptV1,
    *,
    admission: StaticQualificationControlledDistanceSelectorAdmissionV2,
    assets: dict[str, dict[str, Any]],
    repo_root: Path,
    additional_successor_asset_ids: frozenset[str] = frozenset(),
) -> None:
    """Bind every additive receipt identity to immutable base and successor truth."""

    base = receipt.immutable_base
    base_path = resolve_repository_path(
        repo_root,
        base.receipt,
        label="immutable base receipt",
    )
    base_bytes = base_path.read_bytes()
    if hashlib.sha256(base_bytes).hexdigest() != base.receipt_sha256:
        raise CapabilityManifestError(
            "immutable base receipt SHA-256 differs from #1119 admission"
        )
    base_document = loads_json_document(
        base_bytes,
        label="immutable base receipt",
    )
    if (
        base_document.get("schema_version")
        != "joint-agent-reference-corpus-materialization-receipt-v1"
        or base_document.get("prepared_asset_count") != base.verified_asset_count
        or base_document.get("verified_artifact_count") != base.verified_artifact_count
    ):
        raise CapabilityManifestError(
            "immutable base receipt counts or schema differ from #1119 admission"
        )

    successor = receipt.successor
    _validate_immutable_base_artifact_identities(
        base_document,
        assets=assets,
        successor_asset_id=successor.asset_id,
        additional_successor_asset_ids=additional_successor_asset_ids,
    )
    if successor.asset_id != admission.release_gate_asset_id:
        raise CapabilityManifestError(
            "admission receipt successor differs from the controlled-distance row"
        )
    asset = assets.get(successor.asset_id)
    identities = asset.get("artifact_identities") if isinstance(asset, dict) else None
    if not isinstance(identities, dict):
        raise CapabilityManifestError(
            "admission receipt successor lacks corpus artifact identities"
        )
    reference = identities.get("reference_asset")
    target = identities.get("normalized_asset")
    source_contract = identities.get("controlled_distance_contract")
    authoring_roles = identities.get("selector_authoring_references")
    authoring = (
        authoring_roles.get("controlled_distance")
        if isinstance(authoring_roles, dict)
        else None
    )
    asset_manifest = identities.get("asset_manifest")
    if not all(
        isinstance(item, dict)
        for item in (reference, target, source_contract, authoring, asset_manifest)
    ):
        raise CapabilityManifestError(
            "admission receipt successor has a partial corpus identity closure"
        )
    assert isinstance(reference, dict)
    assert isinstance(target, dict)
    assert isinstance(source_contract, dict)
    assert isinstance(authoring, dict)
    assert isinstance(asset_manifest, dict)
    reference_bundle = reference.get("dependency_bundle")
    target_bundle = target.get("dependency_bundle")
    source_contract_bundle = source_contract.get("dependency_bundle")
    authoring_identity = authoring.get("identity")
    if not all(
        isinstance(item, dict)
        for item in (
            reference_bundle,
            target_bundle,
            source_contract_bundle,
            authoring_identity,
        )
    ):
        raise CapabilityManifestError(
            "admission receipt successor has a partial dependency closure"
        )
    assert isinstance(reference_bundle, dict)
    assert isinstance(target_bundle, dict)
    assert isinstance(source_contract_bundle, dict)
    assert isinstance(authoring_identity, dict)
    successor_records = _corpus_artifact_identity_records(
        successor.asset_id,
        identities,
    )
    if (
        len(successor_records) != successor.corpus_artifact_identity_count
        or _corpus_artifact_identities_sha256(identities)
        != successor.corpus_artifact_identities_sha256
    ):
        raise CapabilityManifestError(
            "admission receipt successor does not bind its complete corpus identity set"
        )
    expected = {
        "asset_manifest_sha256": asset_manifest.get("sha256"),
        "reference_sha256": reference.get("sha256"),
        "reference_dependency_bundle_sha256": reference_bundle.get("sha256"),
        "target_sha256": target.get("sha256"),
        "target_dependency_bundle_sha256": target_bundle.get("sha256"),
        "source_contract_sha256": source_contract.get("sha256"),
        "source_contract_authorer_dependency_bundle_schema_version": (
            "world-understanding-usd-dependency-bundle-v3"
        ),
        "source_contract_authorer_dependency_bundle_sha256": (
            admission.expected_joint_semantics.controlled_distance_contract.source_artifact.dependency_bundle_sha256
        ),
        "source_contract_gate3_dependency_manifest_schema_version": (
            source_contract_bundle.get("schema_version")
        ),
        "source_contract_gate3_dependency_manifest_sha256": (
            source_contract_bundle.get("sha256")
        ),
        "source_contract_gate3_dependency_manifest_entry_count": (
            source_contract_bundle.get("entry_count")
        ),
        "selector_semantic_preimage_sha256": (
            admission.source_semantic_preimage_sha256
        ),
        "selector_semantics_sha256": admission.expected_joint_semantics_sha256,
        "controlled_distance_contract_sha256": (
            admission.expected_joint_semantics.controlled_distance_contract_sha256
        ),
        "authoring_identity_sha256": authoring_identity.get("sha256"),
        "retained_tree_identity_sha256": (
            admission.fixture_prevalidation.retained_tree_identity_sha256
        ),
        "primary_fixture_result_sha256": (
            admission.fixture_prevalidation.primary_fixture_result_sha256
        ),
        "structural_audit_receipt_sha256": (
            admission.fixture_prevalidation.structural_audit_receipt_sha256
        ),
        "gate3a_results_sha256": (
            admission.fixture_prevalidation.gate3a_results_sha256
        ),
        "gate3b_results_sha256": (
            admission.fixture_prevalidation.gate3b_results_sha256
        ),
        "independent_postexec_receipt_sha256": (
            admission.fixture_prevalidation.independent_postexec_receipt_sha256
        ),
    }
    for field, value in expected.items():
        if getattr(successor, field) != value:
            raise CapabilityManifestError(
                f"admission receipt successor {field} differs from admitted truth"
            )
    fresh = receipt.fresh_source_verification
    if (
        fresh.fresh_asset_manifest_sha256 != successor.asset_manifest_sha256
        or fresh.fresh_authoring_identity_sha256 != successor.authoring_identity_sha256
        or fresh.fresh_source_contract_sha256 != successor.source_contract_sha256
        or fresh.prepared_asset_artifact_count
        != successor.prepared_asset_artifact_count
        or receipt.aggregate.verified_asset_count != base.verified_asset_count + 1
        or receipt.aggregate.verified_artifact_count
        != base.verified_artifact_count + successor.corpus_artifact_identity_count
    ):
        raise CapabilityManifestError(
            "admission receipt fresh-source or aggregate closure is inconsistent"
        )


def _validate_fixed_full_profile_admission_receipt(
    receipt: FixedFullProfileAdmissionReceiptV1,
    *,
    admission: StaticQualificationSelectorAdmissionV1,
    controlled_receipt: ControlledDistanceAdmissionReceiptV1,
    assets: dict[str, dict[str, Any]],
    manifest: CapabilityManifestV1,
    repo_root: Path,
) -> None:
    """Bind the corrected #1113 fixture without relabeling predecessor evidence."""

    prior = receipt.prior_corpus
    if (
        prior.controlled_distance_admission_receipt != manifest.corpus.admission_receipt
        or prior.controlled_distance_admission_receipt_sha256
        != manifest.corpus.admission_receipt_sha256
        or prior.corpus_sha256 != controlled_receipt.corpus_sha256
        or prior.coverage_report_sha256 != controlled_receipt.coverage_report_sha256
        or prior.verified_asset_count
        != controlled_receipt.aggregate.verified_asset_count
        or prior.verified_artifact_count
        != controlled_receipt.aggregate.verified_artifact_count
    ):
        raise CapabilityManifestError(
            "fixed admission receipt differs from its admitted #1119 predecessor"
        )

    successor = receipt.successor
    asset = assets.get(successor.asset_id)
    if not isinstance(asset, dict) or asset.get("preparation_state") != "prepared":
        raise CapabilityManifestError(
            "fixed admission receipt successor is not one prepared corpus row"
        )
    identities = asset.get("artifact_identities")
    if not isinstance(identities, dict):
        raise CapabilityManifestError(
            "fixed admission receipt successor lacks corpus artifact identities"
        )
    reference = identities.get("reference_asset")
    target = identities.get("normalized_asset")
    asset_manifest = identities.get("asset_manifest")
    authoring_roles = identities.get("selector_authoring_references")
    authoring = (
        authoring_roles.get("fixed") if isinstance(authoring_roles, dict) else None
    )
    if not all(
        isinstance(item, dict)
        for item in (reference, target, asset_manifest, authoring)
    ):
        raise CapabilityManifestError(
            "fixed admission receipt successor has a partial corpus identity closure"
        )
    assert isinstance(reference, dict)
    assert isinstance(target, dict)
    assert isinstance(asset_manifest, dict)
    assert isinstance(authoring, dict)
    reference_bundle = reference.get("dependency_bundle")
    target_bundle = target.get("dependency_bundle")
    package_reference = authoring.get("package_reference")
    authoring_identity = authoring.get("identity")
    if not all(
        isinstance(item, dict)
        for item in (
            reference_bundle,
            target_bundle,
            package_reference,
            authoring_identity,
        )
    ):
        raise CapabilityManifestError(
            "fixed admission receipt successor has a partial dependency closure"
        )
    assert isinstance(reference_bundle, dict)
    assert isinstance(target_bundle, dict)
    assert isinstance(package_reference, dict)
    assert isinstance(authoring_identity, dict)
    package_bundle = package_reference.get("dependency_bundle")
    if not isinstance(package_bundle, dict):
        raise CapabilityManifestError(
            "fixed admission receipt lacks the authoring-reference dependency closure"
        )
    successor_records = _corpus_artifact_identity_records(
        successor.asset_id,
        identities,
    )
    if (
        len(successor_records) != successor.corpus_artifact_identity_count
        or _corpus_artifact_identities_sha256(identities)
        != successor.corpus_artifact_identities_sha256
    ):
        raise CapabilityManifestError(
            "fixed admission receipt does not bind its complete corpus identity set"
        )

    reference_authority = admission.expected_joint_semantics.source_authority
    expected = {
        "asset_manifest_sha256": asset_manifest.get("sha256"),
        "reference_sha256": reference.get("sha256"),
        "reference_dependency_bundle_sha256": reference_bundle.get("sha256"),
        "target_sha256": target.get("sha256"),
        "target_dependency_bundle_sha256": target_bundle.get("sha256"),
        "authoring_reference_sha256": package_reference.get("sha256"),
        "authoring_reference_dependency_bundle_sha256": package_bundle.get("sha256"),
        "selector_semantic_preimage_sha256": (
            admission.source_semantic_preimage_sha256
        ),
        "selector_semantics_sha256": admission.expected_joint_semantics_sha256,
        "authoring_identity_sha256": authoring_identity.get("sha256"),
    }
    for field, value in expected.items():
        if getattr(successor, field) != value:
            raise CapabilityManifestError(
                f"fixed admission receipt successor {field} differs from admitted truth"
            )
    if (
        reference_authority.parent_reference.sha256 != successor.reference_sha256
        or reference_authority.parent_reference.dependency_bundle.sha256
        != successor.reference_dependency_bundle_sha256
        or admission.authoring_reference.package_reference.sha256
        != successor.authoring_reference_sha256
        or admission.authoring_reference.package_reference.dependency_bundle.sha256
        != successor.authoring_reference_dependency_bundle_sha256
    ):
        raise CapabilityManifestError(
            "fixed selector authority differs from the receipt source/package closure"
        )

    fresh = receipt.fresh_source_verification
    provenance = asset.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("path") != fresh.source_generator
        or provenance.get("source_sha256") != fresh.source_generator_sha256
    ):
        raise CapabilityManifestError(
            "fixed admission generator provenance differs from the fresh-source closure"
        )
    path_digests = (
        (fresh.source_generator, fresh.source_generator_sha256),
        (fresh.package_preparer, fresh.package_preparer_sha256),
        (fresh.structural_auditor, fresh.structural_auditor_sha256),
        (fresh.preparation_spec, fresh.preparation_spec_sha256),
    )
    deterministic_source_bytes: dict[str, bytes] = {}
    for relative_path, expected_sha256 in path_digests:
        path = resolve_repository_path(
            repo_root,
            relative_path,
            label="fixed admission deterministic source",
        )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise CapabilityManifestError(
                "fixed admission deterministic source cannot be read: "
                f"{relative_path}: {exc}"
            ) from exc
        deterministic_source_bytes[relative_path] = payload
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise CapabilityManifestError(
                "fixed admission deterministic source identity changed: "
                f"{relative_path}"
            )
    # Exact run provenance is internal authority and `internal/` is stripped
    # from public staging. Keep the import lazy so importing this public-shipped
    # capability module never depends on a staging-excluded module.
    authority_module = "joint_agent.internal.fixed_full_profile_prevalidation_authority"
    try:
        from joint_agent.internal.fixed_full_profile_prevalidation_authority import (
            validate_fixed_full_profile_internal_authority,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"joint_agent.internal", authority_module}:
            raise
        raise CapabilityManifestError(
            "fixed admission internal #1113 authority is unavailable"
        ) from exc

    try:
        validate_fixed_full_profile_internal_authority(
            artifact_id=receipt.fixture_prevalidation.artifact_id,
            implementation_head_sha=(
                receipt.fixture_prevalidation.implementation_head_sha
            ),
            base_main_sha=receipt.fixture_prevalidation.base_main_sha,
            structural_auditor=fresh.structural_auditor,
            structural_auditor_sha256=fresh.structural_auditor_sha256,
        )
    except ValueError as exc:
        raise CapabilityManifestError(
            "fixed admission internal #1113 authority differs from the accepted run"
        ) from exc
    try:
        preparation_spec = yaml.safe_load(
            deterministic_source_bytes[fresh.preparation_spec].decode("utf-8")
        )
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CapabilityManifestError(
            "fixed admission preparation spec is not valid UTF-8 YAML"
        ) from exc
    recipe = (
        preparation_spec.get("recipe") if isinstance(preparation_spec, dict) else None
    )
    expected_recipe = {
        "source_generator": fresh.source_generator,
        "source_generator_sha256": fresh.source_generator_sha256,
        "package_preparer": fresh.package_preparer,
        "package_preparer_sha256": fresh.package_preparer_sha256,
    }
    if (
        not isinstance(preparation_spec, dict)
        or preparation_spec.get("asset_id") != successor.asset_id
        or not isinstance(recipe, dict)
        or any(recipe.get(field) != value for field, value in expected_recipe.items())
    ):
        raise CapabilityManifestError(
            "fixed admission preparation recipe differs from the fresh-source closure"
        )
    if (
        fresh.fresh_asset_manifest_sha256 != successor.asset_manifest_sha256
        or fresh.fresh_authoring_identity_sha256 != successor.authoring_identity_sha256
        or fresh.fresh_reference_sha256 != successor.reference_sha256
        or fresh.fresh_target_sha256 != successor.target_sha256
        or fresh.fresh_authoring_reference_sha256
        != successor.authoring_reference_sha256
    ):
        raise CapabilityManifestError(
            "fixed admission fresh-source closure differs from the successor"
        )
    prevalidation = receipt.fixture_prevalidation
    selector_prevalidation_fields = set(
        FixedFullProfileSelectorPrevalidationV1.model_fields
    )
    fixture_only_prevalidation_fields = (
        set(FixedFullProfileFixturePrevalidationV1.model_fields)
        - selector_prevalidation_fields
    )
    selector_prevalidation = FixedFullProfileSelectorPrevalidationV1.model_validate(
        prevalidation.model_dump(
            mode="json",
            exclude=fixture_only_prevalidation_fields,
        )
    )
    if (
        admission.fixture_prevalidation != selector_prevalidation
        or prevalidation.source_sha256 != successor.reference_sha256
        or prevalidation.target_sha256 != successor.target_sha256
        or prevalidation.authoring_reference_sha256
        != successor.authoring_reference_sha256
    ):
        raise CapabilityManifestError(
            "fixed admission prevalidation closure is inconsistent"
        )
    if (
        receipt.aggregate.verified_asset_count != prior.verified_asset_count + 1
        or receipt.aggregate.verified_artifact_count
        != prior.verified_artifact_count + successor.corpus_artifact_identity_count
    ):
        raise CapabilityManifestError(
            "fixed admission aggregate closure is inconsistent"
        )


def require_articulation_v2_static_target_authority(
    loaded: LoadedCapabilityManifest,
    release_gate_asset_id: str,
    *,
    repo_root: str | Path,
) -> ArticulationV2StaticUsdArtifactAuthorityV1:
    """Return one manifest-bound #871 stripped target by exact asset ID."""

    assets = _load_validated_corpus_assets(loaded, repo_root=repo_root)
    asset = assets.get(release_gate_asset_id)
    if asset is None or asset.get("preparation_state") != "prepared":
        raise CapabilityManifestError(
            "static selector target is missing or unprepared in issue #871: "
            f"{release_gate_asset_id}"
        )
    identities = asset.get("artifact_identities")
    if not isinstance(identities, dict):
        raise CapabilityManifestError(
            "static selector target lacks #871 artifact identities: "
            f"{release_gate_asset_id}"
        )
    # Canonical storage is internal authority and `internal/` is stripped from
    # public staging. Keep this projection lazy so importing the public-shipped
    # capability module never depends on a staging-excluded module.
    from joint_agent.internal.articulation_v2_static_authority import (
        neutral_static_target_authority,
    )

    try:
        return neutral_static_target_authority(
            asset_id=release_gate_asset_id,
            normalized_asset=identities.get("normalized_asset"),
        )
    except ValueError as exc:
        raise CapabilityManifestError(
            "#871 stripped target authority is invalid for "
            f"{release_gate_asset_id}: {exc}"
        ) from exc


def _validate_corpus_qualification_row(
    asset: dict[str, Any],
) -> dict[str, Any]:
    """Validate one row after its hash-bound corpus report is admitted.

    Prepared ``asset.yaml`` manifests are immutable reproduction metadata. Their
    historical labels and truth selectors are never qualification claims. The
    corpus coverage row is the only admissible source because it is generated
    from the reviewed corpus manifest and hash-bound by the capability manifest.
    """

    authority = asset.get("qualification_truth")
    expected_authority = {
        "schema_version": CORPUS_QUALIFICATION_TRUTH_SCHEMA_VERSION,
        "authority": CORPUS_QUALIFICATION_TRUTH_AUTHORITY,
        "prepared_asset_manifest_role": PREPARED_ASSET_MANIFEST_QUALIFICATION_ROLE,
    }
    if authority != expected_authority:
        raise CapabilityManifestError(
            "qualification evidence must carry the exact corpus authority marker"
        )
    asset_id = asset["asset_id"]
    if asset.get("preparation_state") != "prepared":
        raise CapabilityManifestError(
            f"corpus qualification evidence is not prepared: {asset_id}"
        )
    if asset.get("evidence_class") not in {
        "source_backed",
        "analytic_positive",
        "analytic_negative",
    }:
        raise CapabilityManifestError(
            f"corpus qualification evidence has an invalid class: {asset_id}"
        )
    truth_selectors = asset.get("truth_selectors")
    if (
        not isinstance(truth_selectors, list)
        or not truth_selectors
        or any(not isinstance(selector, dict) for selector in truth_selectors)
    ):
        raise CapabilityManifestError(
            f"corpus qualification evidence lacks reviewed truth selectors: {asset_id}"
        )
    if not isinstance(asset.get("artifact_identities"), dict):
        raise CapabilityManifestError(
            f"corpus qualification evidence lacks artifact identities: {asset_id}"
        )
    return asset


def qualification_evidence_roster(
    loaded: LoadedCapabilityManifest,
    *,
    repo_root: str | Path,
) -> tuple[EvidenceBindingV1, ...]:
    """Derive one exact qualification asset roster.

    The corpus and its coverage report are #871's artifact; the selection of
    which of its assets are in scope is #868's, recorded as that issue's
    ``scope_decisions`` entry. The issue number in every message here derives
    from ``_STATIC_QUALIFICATION_SCOPE_ISSUE`` so the two cannot drift.
    """

    issue = _STATIC_QUALIFICATION_SCOPE_ISSUE
    manifest = loaded.manifest
    coverage_path = resolve_repository_path(
        repo_root,
        manifest.corpus.coverage_report,
        label="corpus coverage report",
    )
    coverage_bytes = coverage_path.read_bytes()
    if hashlib.sha256(coverage_bytes).hexdigest() != (
        manifest.corpus.coverage_report_sha256
    ):
        raise CapabilityManifestError(
            "corpus coverage report SHA-256 does not match the capability manifest"
        )
    coverage = loads_json_document(coverage_bytes, label="corpus coverage report")
    if (
        coverage.get("corpus_id") != manifest.corpus.corpus_id
        or coverage.get("corpus_sha256") != manifest.corpus.corpus_sha256
    ):
        raise CapabilityManifestError(
            "corpus identity does not match the capability manifest"
        )
    raw_assets = coverage.get("assets")
    if not isinstance(raw_assets, list):
        raise CapabilityManifestError("corpus coverage assets must be a list")
    assets: dict[str, Any] = {}
    for asset in raw_assets:
        if not isinstance(asset, dict):
            continue
        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str):
            continue
        if asset_id in assets:
            # `_load_validated_corpus_assets` rejects a repeated asset_id, so a
            # last-wins comprehension here would let the roster and the binding
            # validator disagree about which record one ID names. These are list
            # entries, so the duplicate JSON *key* hook cannot catch them.
            raise CapabilityManifestError(
                f"corpus coverage asset ID is duplicated: {asset_id}"
            )
        assets[asset_id] = asset
    selected_ids = _qualification_asset_ids_from_coverage(coverage, issue=issue)
    evidence_role_by_class: dict[str, EvidenceRole] = {
        "source_backed": "representative_source",
        "analytic_positive": "analytic_positive",
        "analytic_negative": "fail_closed_negative",
    }
    bindings = []
    for asset_id in sorted(selected_ids):
        asset = assets.get(asset_id)
        if asset is None or asset.get("preparation_state") != "prepared":
            raise CapabilityManifestError(
                f"issue #{issue} qualification asset is missing or unprepared: "
                f"{asset_id}"
            )
        evidence_class = asset.get("evidence_class")
        role = (
            evidence_role_by_class.get(evidence_class)
            if isinstance(evidence_class, str)
            else None
        )
        identities = asset.get("artifact_identities")
        reference = (
            identities.get("reference_asset") if isinstance(identities, dict) else None
        )
        sha256 = reference.get("sha256") if isinstance(reference, dict) else None
        if role is None or not isinstance(sha256, str):
            raise CapabilityManifestError(
                f"issue #{issue} qualification asset lacks reference identity: "
                f"{asset_id}"
            )
        bindings.append(
            EvidenceBindingV1(
                asset_id=asset_id,
                evidence_role=role,
                artifact_key="reference_asset",
                sha256=sha256,
            )
        )
    return tuple(bindings)


def public_authorable_types(manifest: CapabilityManifestV1) -> frozenset[str]:
    """Return canonical types currently enabled on the public product surface."""

    return frozenset(
        row.canonical_type
        for row in manifest.capabilities
        if row.public_exposure == "enabled"
    )


def represented_types(manifest: CapabilityManifestV1) -> frozenset[str]:
    """Return every canonical type with an explicit support decision."""

    return frozenset(row.canonical_type for row in manifest.capabilities)


def contract_types(
    manifest: CapabilityManifestV1,
    contract_version: str,
) -> frozenset[str]:
    """Return canonical types represented by one contract version."""

    return frozenset(
        row.canonical_type
        for row in manifest.capabilities
        if contract_version in row.contract_versions
    )


def articulation_v2_constraint_types(
    manifest: CapabilityManifestV1,
) -> frozenset[str]:
    """Return constraint kinds explicitly assigned to articulation-v2 rows."""

    return frozenset(
        kind
        for row in manifest.capabilities
        for kind in row.articulation_v2_constraint_kinds
    )


def articulation_v2_capability_ids(
    manifest: CapabilityManifestV1,
) -> frozenset[str]:
    """Return capability decisions explicitly registered to articulation-v2."""

    return frozenset(
        row.capability_id
        for row in manifest.capabilities
        if ARTICULATION_V2_CONTRACT_VERSION in row.contract_versions
    )


def validate_surface_alignment(
    manifest: CapabilityManifestV1,
    *,
    inference_types: frozenset[str],
    articulation_v1_types: frozenset[str],
    articulation_v2_types: frozenset[str],
    articulation_v2_implemented_capability_ids: frozenset[str],
    public_capability_ids: frozenset[str],
    stage2_promotable_types: frozenset[str],
    owned_authorer_types: frozenset[str],
    documented_public_types: frozenset[str],
) -> None:
    """Fail when implementation or documentation drifts from manifest policy."""

    represented = represented_types(manifest)
    missing_inference = inference_types - represented
    if missing_inference:
        raise CapabilityManifestError(
            f"inference types lack manifest rows: {sorted(missing_inference)}"
        )
    declared_v1 = contract_types(manifest, "joint-agent-articulation-v1")
    if articulation_v1_types != declared_v1:
        raise CapabilityManifestError(
            "articulation-v1 type union disagrees with manifest rows: "
            f"contract={sorted(articulation_v1_types)}, "
            f"manifest={sorted(declared_v1)}"
        )
    declared_v2_types = articulation_v2_constraint_types(manifest)
    if articulation_v2_types != declared_v2_types:
        raise CapabilityManifestError(
            "articulation-v2 constraint union disagrees with manifest decisions: "
            f"contract={sorted(articulation_v2_types)}, "
            f"manifest={sorted(declared_v2_types)}"
        )
    declared_v2_capability_ids = articulation_v2_capability_ids(manifest)
    if articulation_v2_implemented_capability_ids != declared_v2_capability_ids:
        raise CapabilityManifestError(
            "articulation-v2 capability implementation disagrees with manifest "
            "decisions: "
            f"implementation={sorted(articulation_v2_implemented_capability_ids)}, "
            f"manifest={sorted(declared_v2_capability_ids)}"
        )
    selected_public = frozenset(manifest.selected_public_capability_ids)
    if public_capability_ids != selected_public:
        raise CapabilityManifestError(
            "public capability IDs disagree with enabled manifest rows: "
            f"surface={sorted(public_capability_ids)}, "
            f"manifest={sorted(selected_public)}"
        )
    enabled = public_authorable_types(manifest)
    if stage2_promotable_types != enabled:
        raise CapabilityManifestError(
            "Stage 2 promotable types disagree with enabled public rows: "
            f"stage2={sorted(stage2_promotable_types)}, enabled={sorted(enabled)}"
        )
    if not stage2_promotable_types <= owned_authorer_types:
        raise CapabilityManifestError(
            "Stage 2 exposes a type missing from the owned authorer: "
            f"{sorted(stage2_promotable_types - owned_authorer_types)}"
        )
    if not owned_authorer_types <= represented:
        raise CapabilityManifestError(
            "owned authorer types lack manifest rows: "
            f"{sorted(owned_authorer_types - represented)}"
        )
    if documented_public_types != enabled:
        raise CapabilityManifestError(
            "public support documentation disagrees with enabled manifest rows: "
            f"docs={sorted(documented_public_types)}, enabled={sorted(enabled)}"
        )


def render_public_support_table(manifest: CapabilityManifestV1) -> str:
    """Render the canonical README block for currently enabled capability rows."""

    rows = [
        row
        for row in manifest.capabilities
        if row.capability_id in manifest.selected_public_capability_ids
    ]
    lines = [
        SUPPORT_TABLE_START,
        "## Joint capability support",
        "",
        (
            "The checked-in capability manifest is the source of truth. "
            "The 0.5 public surface remains limited to the rows below; other "
            "recognized or internal authorer types are gated."
        ),
        "",
        "| Type | Highest product level | Contract | Static status | Dynamic status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.canonical_type,
                    row.support_level,
                    ", ".join(row.contract_versions),
                    row.qualification.static_status,
                    row.qualification.dynamic_status,
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            (
                "A completed static run can still retain validator findings; "
                "dynamic behavior is a separate qualification lane."
            ),
            SUPPORT_TABLE_END,
        )
    )
    return "\n".join(lines)


def extract_public_support_table(markdown: str) -> str:
    """Extract the exact manifest-managed support block from Markdown."""

    start = markdown.find(SUPPORT_TABLE_START)
    end = markdown.find(SUPPORT_TABLE_END)
    if start < 0 or end < 0 or end < start:
        raise CapabilityManifestError("public support table markers are missing")
    end += len(SUPPORT_TABLE_END)
    if markdown.find(SUPPORT_TABLE_START, start + 1) >= 0:
        raise CapabilityManifestError("public support table start marker is duplicated")
    if markdown.find(SUPPORT_TABLE_END, end) >= 0:
        raise CapabilityManifestError("public support table end marker is duplicated")
    return markdown[start:end]


def _validate_frozen_baseline(
    baseline: FrozenBaselineV1,
    coverage: dict[str, Any],
    repo_root: Path,
) -> None:
    frozen = coverage.get("frozen_0_5")
    if not isinstance(frozen, dict):
        raise CapabilityManifestError("corpus coverage is missing frozen_0_5")
    expected = {
        "release_manifest": baseline.release_manifest,
        "release_manifest_sha256": baseline.release_manifest_sha256,
        "gate3_baseline": baseline.gate3_baseline,
        "gate3_baseline_sha256": baseline.gate3_baseline_sha256,
        "asset_count": baseline.asset_count,
        "joint_count": baseline.joint_count,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise CapabilityManifestError(
                f"frozen 0.5 {key} disagrees with the capability manifest"
            )
    for path, expected_sha256 in (
        (baseline.release_manifest, baseline.release_manifest_sha256),
        (baseline.gate3_baseline, baseline.gate3_baseline_sha256),
    ):
        evidence_path = resolve_repository_path(
            repo_root,
            path,
            label="frozen 0.5 evidence",
        )
        actual_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise CapabilityManifestError(
                f"frozen 0.5 evidence changed without a manifest decision: {path}"
            )


def _validate_row_evidence(
    row: CapabilityRowV1,
    assets: dict[str, dict[str, Any]],
    baseline: FrozenBaselineV1,
) -> None:
    _validate_frozen_row_evidence(row, baseline)
    bound_assets: list[dict[str, Any]] = []
    bindings = (
        *row.fixtures.positive,
        *row.fixtures.negative,
        *row.fixtures.representative,
    )
    for binding in bindings:
        if binding.evidence_role == "frozen_0_5":
            continue
        asset = assets.get(binding.asset_id)
        if asset is None:
            raise CapabilityManifestError(
                f"{row.capability_id} references unknown asset {binding.asset_id}"
            )
        if asset.get("preparation_state") != "prepared":
            raise CapabilityManifestError(
                f"{row.capability_id} references unprepared asset {binding.asset_id}"
            )
        expected_class = _EVIDENCE_CLASS_BY_ROLE[binding.evidence_role]
        if asset.get("evidence_class") != expected_class:
            raise CapabilityManifestError(
                f"{row.capability_id} evidence role {binding.evidence_role} "
                f"disagrees with {binding.asset_id}"
            )
        identities = asset.get("artifact_identities")
        identity = (
            identities.get(binding.artifact_key)
            if isinstance(identities, dict)
            else None
        )
        if not isinstance(identity, dict) or identity.get("sha256") != binding.sha256:
            raise CapabilityManifestError(
                f"{row.capability_id} evidence identity drifted for "
                f"{binding.asset_id}:{binding.artifact_key}"
            )
        bound_assets.append(asset)

    if bound_assets and row.scope.source_families:
        observed = {str(asset.get("source_family")) for asset in bound_assets}
        outside = observed - set(row.scope.source_families)
        if outside:
            raise CapabilityManifestError(
                f"{row.capability_id} evidence exceeds its source-family scope: "
                f"{sorted(outside)}"
            )
    if bound_assets and row.scope.asset_archetypes:
        allowed = set(row.scope.asset_archetypes)
        for asset in bound_assets:
            archetypes = asset.get("asset_archetypes")
            observed = set(archetypes) if isinstance(archetypes, list) else set()
            if not observed & allowed:
                raise CapabilityManifestError(
                    f"{row.capability_id} evidence asset {asset.get('asset_id')} "
                    "is outside its archetype scope"
                )


def _validate_static_selector_bindings(
    row: CapabilityRowV1,
    assets: dict[str, dict[str, Any]],
    *,
    qualification_asset_ids: frozenset[str],
) -> None:
    """Cross-check typed admissions against the hash-bound #871 truth row."""

    # Canonical storage is an internal authority.  Keep provider locators out of
    # this public-shipped module and manifest; this lazy import projects the
    # internal coverage rows onto their neutral asset-relative identities.
    from joint_agent.internal.articulation_v2_static_authority import (
        neutral_controlled_distance_selector_authoring_reference,
        neutral_controlled_distance_source_contract,
        neutral_selector_authoring_reference,
        neutral_static_source_authority,
    )

    admissions = row.qualification.selector_admissions
    if admissions is None:
        return
    fixture_bindings = (
        *row.fixtures.positive,
        *row.fixtures.negative,
        *row.fixtures.representative,
    )
    for admission in admissions:
        asset_id = admission.release_gate_asset_id
        if asset_id not in qualification_asset_ids:
            raise CapabilityManifestError(
                f"{row.capability_id} static selector asset is outside issue "
                f"#{_STATIC_QUALIFICATION_SCOPE_ISSUE}'s exact roster: {asset_id}"
            )
        asset = assets.get(asset_id)
        if asset is None or asset.get("preparation_state") != "prepared":
            raise CapabilityManifestError(
                f"{row.capability_id} static selector asset is missing or "
                f"unprepared in issue #871: {asset_id}"
            )
        controlled_distance = isinstance(
            admission,
            StaticQualificationControlledDistanceSelectorAdmissionV2,
        )
        expected_joint_type = (
            "prismatic" if controlled_distance else (admission.constraint_kind)
        )
        joint_types = asset.get("joint_types")
        if not isinstance(joint_types, list) or (
            expected_joint_type not in joint_types
        ):
            raise CapabilityManifestError(
                f"{row.capability_id} selector kind is absent from #871 asset "
                f"truth: {asset_id}/{expected_joint_type}"
            )
        matching_truth_selectors = [
            selector
            for selector in asset.get("truth_selectors", [])
            if isinstance(selector, dict)
            and selector.get("selector_type") == admission.selector_type
            and selector.get("value") == admission.source_joint_path
            and selector.get("artifact_role") == "reference_asset"
        ]
        if len(matching_truth_selectors) != 1:
            raise CapabilityManifestError(
                f"{row.capability_id} selector does not name one exact #871 "
                f"reference truth: {asset_id}/{admission.source_joint_path}"
            )
        identities = asset.get("artifact_identities")
        if not isinstance(identities, dict):
            raise CapabilityManifestError(
                f"#871 asset lacks artifact identities: {asset_id}"
            )
        reference = identities.get("reference_asset")
        effective = identities.get("effective_reference_manifest")
        if not isinstance(reference, dict) or not isinstance(effective, dict):
            raise CapabilityManifestError(
                f"#871 asset lacks semantic source identities: {asset_id}"
            )
        try:
            source_authority = neutral_static_source_authority(
                asset_id=asset_id,
                parent_reference=reference,
                effective_reference_manifest=effective,
            )
        except ValueError as exc:
            raise CapabilityManifestError(
                f"#871 semantic source authority is invalid for {asset_id}: {exc}"
            ) from exc
        if admission.expected_joint_semantics.source_authority != source_authority:
            raise CapabilityManifestError(
                f"{row.capability_id} parent semantic authority disagrees with "
                f"#871: {asset_id}"
            )
        if controlled_distance:
            raw_source_contract = identities.get("controlled_distance_contract")
            if not isinstance(raw_source_contract, dict):
                raise CapabilityManifestError(
                    f"#871 asset lacks controlled-distance source contract: {asset_id}"
                )
            try:
                source_contract = neutral_controlled_distance_source_contract(
                    asset_id=asset_id,
                    value=raw_source_contract,
                )
            except ValueError as exc:
                raise CapabilityManifestError(
                    f"#871 controlled-distance source contract is invalid for "
                    f"{asset_id}: {exc}"
                ) from exc
            if admission.expected_joint_semantics.source_contract != source_contract:
                raise CapabilityManifestError(
                    f"{row.capability_id} source contract disagrees with #871: "
                    f"{asset_id}"
                )
        raw_authoring_references = identities.get("selector_authoring_references")
        if not isinstance(raw_authoring_references, dict):
            raise CapabilityManifestError(
                f"#871 asset lacks selector-scoped authoring references: {asset_id}"
            )
        authoring_role = (
            "controlled_distance" if controlled_distance else admission.constraint_kind
        )
        raw_authoring_reference = raw_authoring_references.get(authoring_role)
        if not isinstance(raw_authoring_reference, dict):
            raise CapabilityManifestError(
                f"#871 asset lacks {authoring_role} authoring authority: {asset_id}"
            )
        try:
            if controlled_distance:
                authoring_reference = (
                    neutral_controlled_distance_selector_authoring_reference(
                        asset_id=asset_id,
                        value=raw_authoring_reference,
                    )
                )
            else:
                authoring_reference = neutral_selector_authoring_reference(
                    asset_id=asset_id,
                    value=raw_authoring_reference,
                )
        except ValueError as exc:
            raise CapabilityManifestError(
                f"#871 selector authoring authority is invalid for {asset_id}: {exc}"
            ) from exc
        if admission.authoring_reference != authoring_reference:
            raise CapabilityManifestError(
                f"{row.capability_id} selector authoring authority disagrees with "
                f"#871: {asset_id}/{authoring_role}"
            )
        matching_fixtures = [
            binding
            for binding in fixture_bindings
            if binding.asset_id == asset_id
            and binding.artifact_key == "reference_asset"
            and binding.sha256 == source_authority.parent_reference.sha256
        ]
        if len(matching_fixtures) != 1:
            raise CapabilityManifestError(
                f"{row.capability_id} static selector has no single matching "
                f"fixture identity for #871 asset {asset_id}"
            )


def _validate_frozen_row_evidence(
    row: CapabilityRowV1,
    baseline: FrozenBaselineV1,
) -> None:
    frozen_bindings = tuple(
        binding
        for binding in (
            *row.fixtures.positive,
            *row.fixtures.negative,
            *row.fixtures.representative,
        )
        if binding.evidence_role == "frozen_0_5"
    )
    if row.capability_id not in _FROZEN_0_5_CAPABILITY_IDS:
        if frozen_bindings:
            raise CapabilityManifestError(
                f"{row.capability_id} cannot inherit frozen 0.5 evidence"
            )
        return

    if (
        len(row.fixtures.positive) != 1
        or len(frozen_bindings) != 1
        or row.fixtures.negative
        or row.fixtures.representative
    ):
        raise CapabilityManifestError(
            f"{row.capability_id} must bind only the exact frozen 0.5 release manifest"
        )
    binding = frozen_bindings[0]
    if (
        binding.artifact_key != "release_manifest"
        or binding.sha256 != baseline.release_manifest_sha256
    ):
        raise CapabilityManifestError(
            f"{row.capability_id} frozen 0.5 evidence identity drifted"
        )


def _qualification_asset_ids_from_coverage(
    coverage: dict[str, Any],
    *,
    issue: int,
) -> set[str]:
    decisions = coverage.get("scope_decisions")
    if not isinstance(decisions, list):
        raise CapabilityManifestError("corpus coverage scope_decisions must be a list")
    matches = [
        decision
        for decision in decisions
        if isinstance(decision, dict) and decision.get("issue") == issue
    ]
    if len(matches) != 1:
        raise CapabilityManifestError(
            f"corpus coverage must contain exactly one issue #{issue} scope decision"
        )
    selected: set[str] = set()
    decision = matches[0]
    for field in (
        "source_backed_asset_ids",
        "analytic_positive_asset_ids",
        "analytic_negative_asset_ids",
    ):
        values = decision.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(values) != len(set(values))
        ):
            raise CapabilityManifestError(
                f"issue #{issue} scope decision {field} must contain unique asset IDs"
            )
        overlap = selected.intersection(values)
        if overlap:
            raise CapabilityManifestError(
                f"issue #{issue} scope decision repeats assets across evidence "
                f"classes: {sorted(overlap)}"
            )
        selected.update(values)
    return selected


def _validate_status_binding(
    status: QualificationLaneStatus,
    profiles: tuple[str, ...],
    results: tuple[str, ...],
    lane: str,
) -> None:
    if status in {"pass", "fail"} and (not profiles or not results):
        raise ValueError(f"{lane} pass/fail status requires profile and result IDs")
    if status in {"not_run", "na"} and results:
        raise ValueError(f"{lane} not_run/na status must not carry result IDs")
    if status == "scheduled" and (not profiles or not results):
        raise ValueError(f"{lane} scheduled status requires profile and result IDs")


def validate_static_validation_contract_id(value: str) -> str:
    """Return one canonical, explicitly versioned validation-contract ID."""

    normalized = _nonblank(value, "validation contract")
    if _VALIDATION_CONTRACT_RE.fullmatch(normalized) is None:
        raise ValueError(
            "validation contract must be a canonical lowercase ID ending in -vN"
        )
    return normalized


def _canonical_strings(value: tuple[str, ...], label: str) -> tuple[str, ...]:
    normalized = tuple(_nonblank(item, label) for item in value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} values must be unique")
    return tuple(sorted(normalized))


def _repository_relative_path(value: str, label: str) -> str:
    normalized = _nonblank(value, label)
    path = PurePosixPath(normalized)
    if (
        "\\" in normalized
        or path.is_absolute()
        or path.as_posix() != normalized
        or any(part in {".", ".."} for part in path.parts)
    ):
        # `CapabilityManifestError` subclasses `ValueError`, so the pydantic
        # validators calling this still see a value error, while
        # `resolve_repository_path` -- whose callers catch only
        # `CapabilityManifestError` -- no longer lets this escape as a bare one.
        raise CapabilityManifestError(
            f"{label} must be a canonical repository-relative path"
        )
    return normalized


def _usd_prim_path(value: str, label: str) -> str:
    normalized = _nonblank(value, label)
    if (
        normalized != value
        or "\\" in normalized
        or not normalized.startswith("/")
        or normalized == "/"
        or normalized.endswith("/")
        or any(
            _USD_PRIM_NAME_RE.fullmatch(part) is None
            for part in normalized[1:].split("/")
        )
    ):
        raise ValueError(f"{label} must be a canonical absolute USD prim path")
    return normalized


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys instead of silently taking the last.

    Duplicate keys are legal JSON but ``json.loads`` keeps only the final value,
    so a crafted document could carry one identity for a reader that inspects
    the bytes and another for this policy code. Every policy document is parsed
    through this hook so the two can never disagree.
    """

    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key: {key!r}")
        document[key] = value
    return document


def loads_json_document(payload: bytes | str, *, label: str) -> dict[str, Any]:
    """Parse one policy JSON object, rejecting duplicate keys and non-objects."""

    try:
        document = json.loads(payload, object_pairs_hook=reject_duplicate_json_keys)
    except ValueError as exc:
        raise CapabilityManifestError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise CapabilityManifestError(f"{label} root must be an object")
    return document


def resolve_repository_path(
    repo_root: str | Path,
    relative_path: str,
    *,
    label: str,
) -> Path:
    """Resolve a manifest path while preventing traversal or symlink escape.

    ``repo_root`` is operator-supplied on every runner that takes a
    ``--repo-root``, so a root that cannot be resolved at all is a refusal like
    any other. ``Path.resolve`` raises ``RuntimeError`` -- not ``OSError`` --
    on a symlink loop, and callers only catch ``CapabilityManifestError``.
    """

    try:
        root = Path(repo_root).resolve()
        candidate = (root / _repository_relative_path(relative_path, label)).resolve()
    except (OSError, RuntimeError) as exc:
        raise CapabilityManifestError(
            f"{label} cannot be resolved under repository root {repo_root}: {exc}"
        ) from exc
    if not candidate.is_relative_to(root):
        raise CapabilityManifestError(f"{label} escapes the repository root")
    return candidate


def _sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _nonblank(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be blank")
    return normalized


__all__ = [
    "CAPABILITY_MANIFEST_RESOURCE",
    "CAPABILITY_MANIFEST_SCHEMA_VERSION",
    "SUPPORT_TABLE_END",
    "SUPPORT_TABLE_START",
    "CapabilityManifestError",
    "CapabilityManifestV1",
    "CapabilityRowV1",
    "LoadedCapabilityManifest",
    "StaticQualificationControlledDistanceSelectorAdmissionV2",
    "StaticQualificationSelectorAdmissionV1",
    "StaticQualificationStageAdmissionV1",
    "StaticQualificationStageBindingV1",
    "contract_types",
    "extract_public_support_table",
    "load_capability_manifest",
    "public_authorable_types",
    "qualification_evidence_roster",
    "require_articulation_v2_static_target_authority",
    "require_static_selector_admission",
    "require_static_stage_admission",
    "render_public_support_table",
    "represented_types",
    "resolve_repository_path",
    "validate_corpus_bindings",
    "validate_static_validation_contract_id",
    "validate_surface_alignment",
]
