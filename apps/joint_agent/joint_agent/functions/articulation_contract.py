# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict first-class prim, link, and joint articulation records.

Ready records carry artifact-backed evidence for every canonical fact.
``template_default`` and ``owner_approved_plan`` are review policy inputs, not
source facts: a producer must either keep them on a diagnosed review record or
bind the accepted fact to a declared artifact source such as
``accepted_manifest`` before promotion.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from world_understanding.functions.physics.joint_rigger import (
    ArtifactIdentityV1,
    FieldProvenanceV1,
    JointLimitV1,
)

from joint_agent.functions.articulation_types import ArticulationReviewStatus

ARTICULATION_CONTRACT_SCHEMA_VERSION: Literal["joint-agent-articulation-v1"] = (
    "joint-agent-articulation-v1"
)

type AxisStage = tuple[float, float, float]
type ReviewStatus = ArticulationReviewStatus
type MotionType = Literal["revolute", "prismatic", "spherical"]
type RecordKind = Literal["prim", "link", "joint"]
type LinkRole = str
type BodyAuthoring = Literal["existing", "aggregate"]
type DiagnosticSeverity = Literal["error", "warning"]

_AXIS_TOLERANCE = 1e-6
_UNRESOLVED_ROLE_VALUES = frozenset({"", "unknown", "none", "null", "n/a", "na"})
_SOURCE_BACKED_PROVENANCE_SOURCES = frozenset(
    {
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
    }
)
_RECORD_KIND_ORDER: dict[RecordKind, int] = {
    "prim": 0,
    "link": 1,
    "joint": 2,
}


class _ContractModel(BaseModel):
    """Strict immutable base for the application contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _FrozenFieldEvidence(Mapping[str, FieldProvenanceV1]):
    """Small immutable mapping with stable deepcopy behavior.

    Keeping immutable tuples here lets Pydantic use its public ``BaseModel``
    deepcopy implementation. This avoids copying private Pydantic attributes or
    relying on ``mappingproxy`` pickling behavior.
    """

    __slots__ = ("_items",)
    _items: tuple[tuple[str, FieldProvenanceV1], ...]

    def __init__(self, value: Mapping[str, FieldProvenanceV1]) -> None:
        object.__setattr__(
            self,
            "_items",
            tuple((key, value[key]) for key in sorted(value)),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("_FrozenFieldEvidence is immutable")

    def __getitem__(self, key: str) -> FieldProvenanceV1:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenFieldEvidence:
        cached = memo.get(id(self))
        if isinstance(cached, _FrozenFieldEvidence):
            return cached
        memo[id(self)] = self
        return self

    def __reduce__(
        self,
    ) -> tuple[
        type[_FrozenFieldEvidence],
        tuple[dict[str, FieldProvenanceV1]],
    ]:
        return _FrozenFieldEvidence, (dict(self.items()),)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())


class _FieldEvidenceModel(_ContractModel):
    """Base for records carrying an immutable per-field evidence map."""

    field_evidence: Mapping[str, FieldProvenanceV1]

    @field_validator("field_evidence")
    @classmethod
    def _canonical_field_evidence(
        cls,
        value: Mapping[str, Any],
    ) -> Mapping[str, FieldProvenanceV1]:
        # Pydantic 2.11 can leave Mapping values as dictionaries when a
        # model_dump(mode="python") payload is revalidated. Validate each value
        # explicitly so Python and JSON round trips share one floor-safe path.
        canonical = {
            _nonblank(key, "field_evidence key"): FieldProvenanceV1.model_validate(
                value[key]
            )
            for key in sorted(value)
        }
        return _FrozenFieldEvidence(canonical)

    @field_serializer("field_evidence")
    def _serialize_field_evidence(
        self,
        value: Mapping[str, FieldProvenanceV1],
    ) -> dict[str, FieldProvenanceV1]:
        return {key: value[key] for key in sorted(value)}


class PrimRecordV1(_ContractModel):
    """One explicit member-prim assignment to a declared physical link."""

    kind: Literal["prim"]
    prim_path: str
    link_id: str = Field(min_length=1)
    membership_evidence: FieldProvenanceV1

    @field_validator("prim_path")
    @classmethod
    def _valid_prim_path(cls, value: str) -> str:
        _require_prim_path(value, "prim_path")
        return value

    @field_validator("link_id")
    @classmethod
    def _nonblank_link_id(cls, value: str) -> str:
        return _nonblank(value, "link_id")

    @model_validator(mode="after")
    def _membership_evidence_names_member(self) -> PrimRecordV1:
        if not _is_source_backed_provenance(self.membership_evidence):
            raise ValueError(
                "prim membership requires source-backed provenance, not "
                f"{self.membership_evidence.source}"
            )
        evidence_path = self.membership_evidence.prim_path
        if evidence_path is not None and evidence_path != self.prim_path:
            raise ValueError(
                "membership_evidence prim_path must equal the member prim_path"
            )
        return self


class LinkRecordV1(_FieldEvidenceModel):
    """One physical rigid-body link and its link-scoped facts."""

    kind: Literal["link"]
    link_id: str = Field(min_length=1)
    body_prim_path: str
    body_authoring: BodyAuthoring = "existing"
    role: LinkRole
    axis_stage: AxisStage | None = None
    review_status: ReviewStatus

    @field_validator("link_id", "role")
    @classmethod
    def _nonblank_text(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("body_prim_path")
    @classmethod
    def _valid_body_path(cls, value: str) -> str:
        _require_prim_path(value, "body_prim_path")
        return value

    @field_validator("axis_stage")
    @classmethod
    def _normalized_axis(cls, value: AxisStage | None) -> AxisStage | None:
        if value is not None:
            _require_normalized_axis(value, "axis_stage")
        return value

    @model_validator(mode="after")
    def _validate_evidence(self) -> LinkRecordV1:
        if (
            self.review_status == "ready_for_rigger_input"
            and self.role.strip().lower() in _UNRESOLVED_ROLE_VALUES
        ):
            raise ValueError("ready links require a resolved role")
        expected = {"body_prim_path", "role"}
        if self.axis_stage is not None:
            expected.add("axis_stage")
        _validate_evidence_keys(
            record_label=f"link {self.link_id!r}",
            actual=set(self.field_evidence),
            expected=expected,
            review_status=self.review_status,
        )
        _validate_ready_field_provenance(
            record_label=f"link {self.link_id!r}",
            field_evidence=self.field_evidence,
            review_status=self.review_status,
        )
        return self


class JointRecordV1(_FieldEvidenceModel):
    """One canonical directed physical edge between two declared links."""

    kind: Literal["joint"]
    joint_id: str = Field(min_length=1)
    body0_link: str = Field(min_length=1)
    body1_link: str = Field(min_length=1)
    motion_type: MotionType
    axis_stage: AxisStage | None = None
    limit: JointLimitV1 | None = None
    review_status: ReviewStatus

    @field_validator("joint_id", "body0_link", "body1_link")
    @classmethod
    def _nonblank_identifier(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("axis_stage")
    @classmethod
    def _normalized_axis(cls, value: AxisStage | None) -> AxisStage | None:
        if value is not None:
            _require_normalized_axis(value, "axis_stage")
        return value

    @model_validator(mode="after")
    def _validate_joint(self) -> JointRecordV1:
        if self.body0_link == self.body1_link:
            raise ValueError("body0_link and body1_link must reference distinct links")

        expected_evidence = {"body0_link", "body1_link", "motion_type"}
        if self.motion_type == "spherical":
            if self.axis_stage is not None:
                raise ValueError("spherical joints must not carry axis_stage")
            if self.limit is not None:
                raise ValueError("scalar spherical limits are unsupported in v1")
        else:
            if (
                self.review_status == "ready_for_rigger_input"
                and self.axis_stage is None
            ):
                raise ValueError(
                    "ready revolute and prismatic joints require axis_stage"
                )
            if self.axis_stage is not None:
                expected_evidence.add("axis_stage")
            if self.limit is not None:
                expected_unit = (
                    "degrees" if self.motion_type == "revolute" else "meters"
                )
                if self.limit.unit != expected_unit:
                    raise ValueError(
                        f"{self.motion_type} limits must use {expected_unit}"
                    )
                # JointLimitV1 owns its evidence as a required nested
                # provenance field. Duplicating ``limit`` in field_evidence
                # would create two independently editable sources of truth.

        _validate_evidence_keys(
            record_label=f"joint {self.joint_id!r}",
            actual=set(self.field_evidence),
            expected=expected_evidence,
            review_status=self.review_status,
        )
        _validate_ready_field_provenance(
            record_label=f"joint {self.joint_id!r}",
            field_evidence=self.field_evidence,
            review_status=self.review_status,
        )
        if (
            self.review_status == "ready_for_rigger_input"
            and self.limit is not None
            and not _is_source_backed_provenance(self.limit.provenance)
        ):
            raise ValueError(
                f"ready joint {self.joint_id!r} limit requires source-backed "
                "provenance; producers must bind accepted policy evidence to a "
                "declared artifact source before promotion"
            )
        return self


ArticulationRecordV1: TypeAlias = Annotated[  # noqa: UP040
    PrimRecordV1 | LinkRecordV1 | JointRecordV1,
    Field(discriminator="kind"),
]


class ContractDiagnosticV1(_ContractModel):
    """One stable reason that a link or joint requires review."""

    record_kind: Literal["link", "joint"]
    record_id: str = Field(min_length=1)
    code: str = Field(min_length=1)
    severity: DiagnosticSeverity
    field: str | None = None
    prim_paths: tuple[str, ...] = ()
    source_prediction_ids: tuple[str, ...] = ()
    detail: str = Field(min_length=1)

    @field_validator("record_id", "code", "detail")
    @classmethod
    def _nonblank_required_text(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("field")
    @classmethod
    def _nonblank_optional_field(cls, value: str | None) -> str | None:
        if value is not None:
            return _nonblank(value, "field")
        return value

    @field_validator("prim_paths")
    @classmethod
    def _canonical_prim_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for prim_path in value:
            _require_prim_path(prim_path, "diagnostic prim_path")
        _require_unique(value, "diagnostic prim_paths")
        return tuple(sorted(value))

    @field_validator("source_prediction_ids")
    @classmethod
    def _canonical_source_prediction_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(
            _nonblank(prediction_id, "source_prediction_id") for prediction_id in value
        )
        _require_unique(normalized, "diagnostic source_prediction_ids")
        return tuple(sorted(normalized))


class ContractSummaryV1(_ContractModel):
    """Validated record and review counts for fast contract inspection."""

    prim_count: int = Field(ge=0)
    link_count: int = Field(ge=0)
    joint_count: int = Field(ge=0)
    review_required_link_count: int = Field(ge=0)
    review_required_joint_count: int = Field(ge=0)
    diagnostic_count: int = Field(ge=0)


class ArticulationContractV1(_ContractModel):
    """Canonical first-class articulation contract for Joint Agent consumers.

    Producers may carry owner-approved or default policy on diagnosed review
    records. Promotion to ready requires binding each accepted fact to a
    declared artifact-backed provenance source.
    """

    schema_version: Literal["joint-agent-articulation-v1"]
    status: ReviewStatus
    articulation_roots: tuple[str, ...]
    source_identities: tuple[ArtifactIdentityV1, ...] = Field(min_length=1)
    records: tuple[ArticulationRecordV1, ...]
    diagnostics: tuple[ContractDiagnosticV1, ...] = ()
    summary: ContractSummaryV1

    @field_validator("source_identities")
    @classmethod
    def _canonical_source_identities(
        cls,
        value: tuple[ArtifactIdentityV1, ...],
    ) -> tuple[ArtifactIdentityV1, ...]:
        uris = [identity.uri for identity in value]
        _require_unique(uris, "source identity uri values")
        return tuple(
            sorted(
                value,
                key=lambda identity: (
                    identity.uri,
                    identity.root_sha256,
                    identity.dependency_bundle_sha256 or "",
                ),
            )
        )

    @field_validator("articulation_roots")
    @classmethod
    def _canonical_articulation_roots(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(
            _nonblank(item, "articulation root link id") for item in value
        )
        _require_unique(normalized, "articulation root link id values")
        return tuple(sorted(normalized))

    @field_validator("records")
    @classmethod
    def _canonical_records(
        cls,
        value: tuple[ArticulationRecordV1, ...],
    ) -> tuple[ArticulationRecordV1, ...]:
        return tuple(sorted(value, key=_record_sort_key))

    @field_validator("diagnostics")
    @classmethod
    def _canonical_diagnostics(
        cls,
        value: tuple[ContractDiagnosticV1, ...],
    ) -> tuple[ContractDiagnosticV1, ...]:
        # One code identifies one issue at one record field. Producers replace
        # that diagnostic when severity/detail changes instead of emitting a
        # second contradictory state for the same issue.
        keys = [
            (
                diagnostic.record_kind,
                diagnostic.record_id,
                diagnostic.field or "",
                diagnostic.code,
            )
            for diagnostic in value
        ]
        _require_unique(keys, "diagnostic target/field/code values")
        return tuple(
            sorted(
                value,
                key=lambda diagnostic: (
                    diagnostic.record_kind,
                    diagnostic.record_id,
                    diagnostic.field or "",
                    diagnostic.code,
                    diagnostic.detail,
                ),
            )
        )

    @model_validator(mode="after")
    def _validate_document(self) -> ArticulationContractV1:
        prims = tuple(
            record for record in self.records if isinstance(record, PrimRecordV1)
        )
        links = tuple(
            record for record in self.records if isinstance(record, LinkRecordV1)
        )
        joints = tuple(
            record for record in self.records if isinstance(record, JointRecordV1)
        )

        _require_unique(
            [record.prim_path for record in prims],
            "prim_path values",
        )
        _require_unique(
            [record.link_id for record in links],
            "link_id values",
        )
        _require_unique(
            [record.body_prim_path for record in links],
            "link body_prim_path values",
        )
        _require_unique(
            [record.joint_id for record in joints],
            "joint_id values",
        )

        links_by_id = {record.link_id: record for record in links}
        self._validate_membership(prims, links, links_by_id, joints)
        self._validate_joint_graph(joints, links_by_id)
        self._validate_diagnostics(links, joints)
        self._validate_summary(prims, links, joints)
        self._validate_evidence_sources(prims, links, joints)
        return self

    def _validate_membership(
        self,
        prims: tuple[PrimRecordV1, ...],
        links: tuple[LinkRecordV1, ...],
        links_by_id: Mapping[str, LinkRecordV1],
        joints: tuple[JointRecordV1, ...],
    ) -> None:
        member_paths_by_link: dict[str, list[str]] = {
            link.link_id: [] for link in links
        }
        for prim in prims:
            owner = links_by_id.get(prim.link_id)
            if owner is None:
                raise ValueError(
                    f"prim {prim.prim_path} references missing link {prim.link_id!r}"
                )
            member_paths_by_link[owner.link_id].append(prim.prim_path)

        for link in links:
            member_paths = member_paths_by_link[link.link_id]
            if not member_paths:
                raise ValueError(
                    f"link {link.link_id!r} must own at least one explicit member prim"
                )
            if link.body_authoring == "existing":
                if link.body_prim_path not in member_paths:
                    raise ValueError(
                        f"existing link {link.link_id!r} body_prim_path must be one "
                        "of its explicit member prim paths"
                    )
                if len(member_paths) != 1:
                    raise ValueError(
                        f"existing link {link.link_id!r} must own exactly one explicit "
                        "member root equal to body_prim_path; use "
                        "body_authoring='aggregate' for multiple flat sibling roots"
                    )

        parent_by_child = {
            joint.body1_link: joint.body0_link
            for joint in joints
            if joint.body0_link in links_by_id and joint.body1_link in links_by_id
        }
        for index, first in enumerate(prims):
            for second in prims[index + 1 :]:
                if not _prim_paths_overlap(first.prim_path, second.prim_path):
                    continue
                first_link = links_by_id[first.link_id]
                second_link = links_by_id[second.link_id]
                if (
                    first.prim_path == second.prim_path
                    or first.link_id == second.link_id
                ):
                    raise ValueError(
                        "explicit member prim roots in one ownership boundary must "
                        "not overlap by equality or ancestry: "
                        f"{first.prim_path} ({first.link_id!r}) and "
                        f"{second.prim_path} ({second.link_id!r})"
                    )
                if (
                    first_link.body_authoring == "existing"
                    and second_link.body_authoring == "existing"
                ):
                    ancestor, descendant = (
                        (first, second)
                        if _is_same_or_descendant_prim_path(
                            second.prim_path, first.prim_path
                        )
                        else (second, first)
                    )
                    if _is_link_ancestor(
                        ancestor.link_id,
                        descendant.link_id,
                        parent_by_child=parent_by_child,
                    ):
                        # Existing nested rigid bodies use nearest declared link
                        # ownership: the descendant boundary is excluded from its
                        # namespace ancestor's effective member subtree.
                        continue
                raise ValueError(
                    "nested explicit member prim roots require existing bodies "
                    "whose namespace ancestry matches articulation ancestry: "
                    f"{first.prim_path} ({first.link_id!r}) and "
                    f"{second.prim_path} ({second.link_id!r})"
                )

        member_roots = tuple(prim.prim_path for prim in prims)
        for index, first_link in enumerate(links):
            for second_link in links[index + 1 :]:
                if (
                    first_link.body_authoring != "aggregate"
                    and second_link.body_authoring != "aggregate"
                ):
                    continue
                if not _prim_paths_overlap(
                    first_link.body_prim_path,
                    second_link.body_prim_path,
                ):
                    continue
                raise ValueError(
                    "aggregate synthetic body_prim_path values must be pairwise "
                    "disjoint from every other link body target: "
                    f"{first_link.body_prim_path} ({first_link.link_id!r}) and "
                    f"{second_link.body_prim_path} ({second_link.link_id!r})"
                )

        for link in links:
            if link.body_authoring == "aggregate":
                collision = next(
                    (
                        member_root
                        for member_root in member_roots
                        if _prim_paths_overlap(link.body_prim_path, member_root)
                    ),
                    None,
                )
                if collision is None:
                    continue
                raise ValueError(
                    f"aggregate link {link.link_id!r} synthetic body_prim_path "
                    f"{link.body_prim_path} collides with explicit member root "
                    f"{collision}"
                )

    def _validate_joint_graph(
        self,
        joints: tuple[JointRecordV1, ...],
        links_by_id: Mapping[str, LinkRecordV1],
    ) -> None:
        ordered_pairs: list[tuple[str, str]] = []
        incoming_joint_by_link: dict[str, str] = {}
        parent_by_child: dict[str, str] = {}
        children_by_link: dict[str, list[str]] = {
            link_id: [] for link_id in links_by_id
        }

        for joint in joints:
            missing = tuple(
                link_id
                for link_id in (joint.body0_link, joint.body1_link)
                if link_id not in links_by_id
            )
            if missing:
                raise ValueError(
                    f"joint {joint.joint_id!r} references missing link(s): "
                    f"{', '.join(sorted(missing))}"
                )

            ordered_pair = (joint.body0_link, joint.body1_link)
            if ordered_pair in ordered_pairs:
                raise ValueError(
                    "only one canonical joint may exist per ordered link pair: "
                    f"{joint.body0_link!r} -> {joint.body1_link!r}"
                )
            ordered_pairs.append(ordered_pair)

            existing = incoming_joint_by_link.get(joint.body1_link)
            if existing is not None:
                raise ValueError(
                    f"link {joint.body1_link!r} has multiple incoming joints: "
                    f"{existing!r}, {joint.joint_id!r}"
                )
            incoming_joint_by_link[joint.body1_link] = joint.joint_id
            parent_by_child[joint.body1_link] = joint.body0_link
            children_by_link[joint.body0_link].append(joint.body1_link)

            body1 = links_by_id[joint.body1_link]
            if joint.motion_type == "spherical":
                if body1.axis_stage is not None:
                    raise ValueError(
                        f"spherical joint {joint.joint_id!r} requires body1 link "
                        f"{body1.link_id!r} axis_stage to be absent"
                    )
            elif joint.axis_stage is not None:
                if body1.axis_stage is None:
                    raise ValueError(
                        f"joint {joint.joint_id!r} axis_stage requires body1 link "
                        f"{body1.link_id!r} axis_stage"
                    )
                if not _axes_match(joint.axis_stage, body1.axis_stage):
                    raise ValueError(
                        f"joint {joint.joint_id!r} axis_stage conflicts with "
                        f"body1 link {body1.link_id!r} axis_stage"
                    )
            # An absent review-required joint axis is unresolved, not a value
            # that can conflict with link evidence. Contract status keeps that
            # record out of the rigger until its targeted diagnostic is fixed.

        roots = tuple(sorted(set(links_by_id) - set(incoming_joint_by_link)))
        cycle = _find_link_cycle(parent_by_child)
        if cycle is not None:
            raise ValueError(
                f"directed_cycle: joint links form a cycle: {' -> '.join(cycle)}"
            )

        if self.articulation_roots != roots:
            raise ValueError(
                "articulation_roots must exactly identify every directed graph "
                f"component root; expected {list(roots)}, got "
                f"{list(self.articulation_roots)}"
            )

        if self.status == "ready_for_rigger_input":
            if not roots:
                raise ValueError(
                    "ready articulation contract requires at least one explicit "
                    "articulation root"
                )
            reachable: set[str] = set()
            for root in roots:
                component: set[str] = set()
                pending = [root]
                while pending:
                    link_id = pending.pop()
                    component.add(link_id)
                    pending.extend(sorted(children_by_link[link_id], reverse=True))
                overlap = reachable.intersection(component)
                if overlap:  # pragma: no cover - single-parent graph invariant
                    raise ValueError(
                        f"articulation components overlap at links: {sorted(overlap)}"
                    )
                reachable.update(component)
            unreachable = sorted(set(links_by_id) - reachable)
            if unreachable:
                raise ValueError(
                    "ready articulation roots do not reach links: "
                    f"{', '.join(unreachable)}"
                )

    def _validate_diagnostics(
        self,
        links: tuple[LinkRecordV1, ...],
        joints: tuple[JointRecordV1, ...],
    ) -> None:
        reviewable_records: tuple[LinkRecordV1 | JointRecordV1, ...] = (
            *links,
            *joints,
        )
        review_records: dict[tuple[str, str], LinkRecordV1 | JointRecordV1] = {
            (record.kind, _record_identifier(record)): record
            for record in reviewable_records
            if record.review_status == "review_required"
        }
        all_records: dict[tuple[str, str], LinkRecordV1 | JointRecordV1] = {
            (record.kind, _record_identifier(record)): record
            for record in reviewable_records
        }
        diagnosed: set[tuple[str, str]] = set()
        for diagnostic in self.diagnostics:
            target = (diagnostic.record_kind, diagnostic.record_id)
            if target not in all_records:
                raise ValueError(
                    "diagnostic references missing or mismatched record "
                    f"{diagnostic.record_kind}:{diagnostic.record_id}"
                )
            if target not in review_records:
                raise ValueError(
                    "diagnostics may target only review_required records: "
                    f"{diagnostic.record_kind}:{diagnostic.record_id}"
                )
            diagnosed.add(target)

        undiagnosed = sorted(set(review_records) - diagnosed)
        if undiagnosed:
            formatted = ", ".join(
                f"{kind}:{record_id}" for kind, record_id in undiagnosed
            )
            raise ValueError(
                f"review_required records require diagnostics: {formatted}"
            )

        expected_status: ReviewStatus = (
            "review_required" if review_records else "ready_for_rigger_input"
        )
        if self.status != expected_status:
            raise ValueError(
                f"contract status must be {expected_status!r} for its record states"
            )

    def _validate_summary(
        self,
        prims: tuple[PrimRecordV1, ...],
        links: tuple[LinkRecordV1, ...],
        joints: tuple[JointRecordV1, ...],
    ) -> None:
        expected = ContractSummaryV1(
            prim_count=len(prims),
            link_count=len(links),
            joint_count=len(joints),
            review_required_link_count=sum(
                record.review_status == "review_required" for record in links
            ),
            review_required_joint_count=sum(
                record.review_status == "review_required" for record in joints
            ),
            diagnostic_count=len(self.diagnostics),
        )
        if self.summary != expected:
            raise ValueError(
                "summary does not match canonical record and diagnostic counts; "
                f"expected {expected.model_dump(mode='json')}"
            )

    def _validate_evidence_sources(
        self,
        prims: tuple[PrimRecordV1, ...],
        links: tuple[LinkRecordV1, ...],
        joints: tuple[JointRecordV1, ...],
    ) -> None:
        declared = set(self.source_identities)
        for provenance in _iter_contract_evidence(prims, links, joints):
            if provenance.artifact is not None and provenance.artifact not in declared:
                raise ValueError(
                    "evidence artifact must be declared in source_identities: "
                    f"{provenance.artifact.uri}"
                )

        diagnosed_fields: dict[tuple[str, str], set[str]] = {}
        for diagnostic in self.diagnostics:
            if diagnostic.field is not None:
                target = (diagnostic.record_kind, diagnostic.record_id)
                diagnosed_fields.setdefault(target, set()).add(diagnostic.field)

        reviewable_records: tuple[LinkRecordV1 | JointRecordV1, ...] = (
            *links,
            *joints,
        )
        for record in reviewable_records:
            target = (record.kind, _record_identifier(record))
            missing_fields = sorted(
                _review_fallback_fields(record) - diagnosed_fields.get(target, set())
            )
            if missing_fields:
                raise ValueError(
                    "non-source-backed review fallback provenance requires a "
                    "review_required record and targeted diagnostics for "
                    f"{target[0]}:{target[1]} fields: {', '.join(missing_fields)}"
                )


def _validate_evidence_keys(
    *,
    record_label: str,
    actual: set[str],
    expected: set[str],
    review_status: ReviewStatus,
) -> None:
    unknown = sorted(actual - expected)
    if unknown:
        raise ValueError(
            f"{record_label} field_evidence contains unsupported keys: "
            f"{', '.join(unknown)}"
        )
    if review_status == "ready_for_rigger_input" and actual != expected:
        missing = sorted(expected - actual)
        raise ValueError(
            f"ready {record_label} requires complete field_evidence; missing "
            f"{', '.join(missing)}"
        )


def _validate_ready_field_provenance(
    *,
    record_label: str,
    field_evidence: Mapping[str, FieldProvenanceV1],
    review_status: ReviewStatus,
) -> None:
    if review_status != "ready_for_rigger_input":
        return
    fallback_fields = sorted(
        f"{field}={provenance.source}"
        for field, provenance in field_evidence.items()
        if not _is_source_backed_provenance(provenance)
    )
    if fallback_fields:
        raise ValueError(
            f"ready {record_label} topology evidence must be source-backed; "
            f"fields: {', '.join(fallback_fields)}"
        )


def _is_source_backed_provenance(provenance: FieldProvenanceV1) -> bool:
    return bool(
        provenance.source in _SOURCE_BACKED_PROVENANCE_SOURCES
        and provenance.artifact is not None
        and provenance.prim_path is not None
        and provenance.properties
    )


def _review_fallback_fields(
    record: LinkRecordV1 | JointRecordV1,
) -> set[str]:
    fields = {
        field
        for field, provenance in record.field_evidence.items()
        if not _is_source_backed_provenance(provenance)
    }
    if (
        isinstance(record, JointRecordV1)
        and record.limit is not None
        and not _is_source_backed_provenance(record.limit.provenance)
    ):
        fields.add("limit")
    return fields


def _iter_contract_evidence(
    prims: tuple[PrimRecordV1, ...],
    links: tuple[LinkRecordV1, ...],
    joints: tuple[JointRecordV1, ...],
) -> Iterable[FieldProvenanceV1]:
    for prim in prims:
        yield prim.membership_evidence
    for record in (*links, *joints):
        yield from record.field_evidence.values()
    for joint in joints:
        if joint.limit is not None:
            yield joint.limit.provenance


def _record_identifier(record: ArticulationRecordV1) -> str:
    if isinstance(record, PrimRecordV1):
        return record.prim_path
    if isinstance(record, LinkRecordV1):
        return record.link_id
    return record.joint_id


def _record_sort_key(record: ArticulationRecordV1) -> tuple[int, str]:
    return (_RECORD_KIND_ORDER[record.kind], _record_identifier(record))


def _find_link_cycle(
    parent_by_child: Mapping[str, str],
) -> tuple[str, ...] | None:
    resolved: set[str] = set()
    for start in sorted(parent_by_child):
        path: list[str] = []
        positions: dict[str, int] = {}
        current = start
        while (
            current in parent_by_child
            and current not in positions
            and current not in resolved
        ):
            positions[current] = len(path)
            path.append(current)
            current = parent_by_child[current]
        if current in positions:
            cycle = path[positions[current] :]
            first = min(cycle)
            first_index = cycle.index(first)
            canonical = cycle[first_index:] + cycle[:first_index]
            return (*canonical, canonical[0])
        resolved.update(path)
    return None


def _nonblank(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value


def _require_unique(values: Iterable[Any], label: str) -> None:
    values_tuple = tuple(values)
    if len(values_tuple) != len(set(values_tuple)):
        raise ValueError(f"{label} must be unique")


def _require_normalized_axis(value: AxisStage, label: str) -> None:
    if not all(math.isfinite(component) for component in value):
        raise ValueError(f"{label} must contain only finite values")
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_AXIS_TOLERANCE):
        raise ValueError(f"{label} must be a normalized vector")


def _axes_match(first: AxisStage, second: AxisStage) -> bool:
    return all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=_AXIS_TOLERANCE)
        for left, right in zip(first, second, strict=True)
    )


def _require_prim_path(value: str, label: str) -> None:
    if not _is_absolute_prim_path(value):
        raise ValueError(
            f"{label} must be a valid absolute non-root USD prim path without "
            "property or variant selections"
        )


def _is_absolute_prim_path(value: str) -> bool:
    try:
        from pxr import Sdf
    except ImportError:
        return _is_portable_absolute_prim_path(value)

    validation = Sdf.Path.IsValidPathString(value)
    is_valid = validation[0] if isinstance(validation, tuple) else validation
    if not is_valid:
        return False
    path = Sdf.Path(value)
    return bool(
        str(path) == value
        and path.IsAbsolutePath()
        and path.IsPrimPath()
        and not path.ContainsPrimVariantSelection()
    )


def _is_portable_absolute_prim_path(value: str) -> bool:
    # USD prim names are identifier tokens. Namespaces and hyphens are valid in
    # other USD name/path contexts, but Sdf rejects them in prim path segments;
    # ``isidentifier`` also preserves the Unicode identifiers accepted by Sdf.
    if not value.startswith("/") or value == "/":
        return False
    return all(segment.isidentifier() for segment in value[1:].split("/"))


def _is_same_or_descendant_prim_path(value: str, ancestor: str) -> bool:
    try:
        from pxr import Sdf
    except ImportError:
        return value == ancestor or value.startswith(f"{ancestor}/")

    return bool(Sdf.Path(value).HasPrefix(Sdf.Path(ancestor)))


def _prim_paths_overlap(first: str, second: str) -> bool:
    return _is_same_or_descendant_prim_path(
        first, second
    ) or _is_same_or_descendant_prim_path(second, first)


def _is_link_ancestor(
    ancestor: str,
    descendant: str,
    *,
    parent_by_child: Mapping[str, str],
) -> bool:
    visited: set[str] = set()
    current = descendant
    while current not in visited:
        visited.add(current)
        parent = parent_by_child.get(current)
        if parent is None:
            return False
        if parent == ancestor:
            return True
        current = parent
    return False


__all__ = [
    "ARTICULATION_CONTRACT_SCHEMA_VERSION",
    "ArticulationContractV1",
    "ArticulationRecordV1",
    "BodyAuthoring",
    "ContractDiagnosticV1",
    "ContractSummaryV1",
    "DiagnosticSeverity",
    "JointRecordV1",
    "LinkRole",
    "LinkRecordV1",
    "PrimRecordV1",
]
