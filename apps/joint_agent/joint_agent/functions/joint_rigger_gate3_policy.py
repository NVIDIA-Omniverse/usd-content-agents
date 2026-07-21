# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic owner-approved Gate 3 physics policy production.

The producer reads one identity-bound source stage and derives only the static
physics facts authorized by an exact policy. Geometry ownership comes solely
from first-class ``PrimRecordV1`` memberships. No asset label, filename, model,
provider, or hidden physical default participates in a decision.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from world_understanding.functions.physics.joint_rigger import (
    PLAN_SCHEMA_VERSION,
    ArtifactIdentityV1,
    ColliderPlanV1,
    FieldProvenanceV1,
    JointDriveV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerInputV1,
    JointRiggerPlanV1,
    JointStateV1,
    MassPropertiesV1,
    RigidBodyPlanV1,
    canonical_json,
    canonical_sha256,
    identify_usd_artifact,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    BoundInputDirectory,
    SealedSourceBinding,
    bound_input_dependency_snapshots,
    close_source_binding,
    create_sealed_source_binding,
    materialize_bound_input,
    remove_bound_input_directory,
    require_sealed_source_binding,
)

from joint_agent.functions.articulation_contract import (
    ArticulationContractV1,
    LinkRecordV1,
    PrimRecordV1,
)
from joint_agent.functions.joint_rigger_contract_bridge import (
    build_joint_rigger_input_from_contract,
)
from joint_agent.functions.joint_rigger_gate3_plan import (
    GATE3_PHYSICS_PLAN_ENVELOPE_SCHEMA_VERSION,
    Gate3PhysicsPlanEnvelopeV1,
    build_gate3_joint_rigger_input_from_contract,
)

GATE3_PHYSICS_POLICY_SCHEMA_VERSION: Literal["joint-agent-gate3-physics-policy-v1"] = (
    "joint-agent-gate3-physics-policy-v1"
)

type MeshColliderApproximation = Literal[
    "none",
    "convexHull",
    "convexDecomposition",
    "sdf",
]
type ColliderPrimType = Literal[
    "Mesh",
    "Cube",
    "Sphere",
    "Capsule",
    "Cylinder",
    "Cone",
]

SUPPORTED_COLLIDER_PRIM_TYPES: tuple[ColliderPrimType, ...] = (
    "Mesh",
    "Cube",
    "Sphere",
    "Capsule",
    "Cylinder",
    "Cone",
)

_PRODUCER_DERIVATION = "joint_agent_owner_approved_gate3_policy_v1"
_PHYSICS_PLAN_URN_PREFIX = "urn:world-understanding:joint-rigger-physics-plan:sha256:"
_MESH_GEOMETRY_ATTRIBUTES = (
    "extent",
    "faceVertexCounts",
    "faceVertexIndices",
    "holeIndices",
    "points",
)
_NATIVE_SHAPE_GEOMETRY_ATTRIBUTES: Mapping[ColliderPrimType, tuple[str, ...]] = {
    "Cube": ("extent", "size"),
    "Sphere": ("extent", "radius"),
    "Capsule": ("axis", "extent", "height", "radius"),
    "Cylinder": ("axis", "extent", "height", "radius"),
    "Cone": ("axis", "extent", "height", "radius"),
}


class _PolicyModel(BaseModel):
    """Strict immutable base for owner-approved policy documents."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Gate3PhysicsPolicyV1(_PolicyModel):
    """Every physical choice authorized by the first deterministic policy."""

    schema_version: Literal["joint-agent-gate3-physics-policy-v1"]
    density_kg_m3: float
    volume_fill_fraction: float
    passive_drive_type: Literal["force"]
    passive_drive_stiffness: float
    passive_drive_damping: float
    passive_drive_max_force: float
    passive_drive_target_position: float
    passive_drive_target_velocity: float
    passive_drive_max_velocity: float
    allowed_collider_prim_types: tuple[ColliderPrimType, ...]
    mesh_collider_approximation: MeshColliderApproximation
    approval_evidence: str = Field(min_length=1)
    approval_identity: ArtifactIdentityV1

    @field_validator("approval_evidence")
    @classmethod
    def _canonical_evidence(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("approval_evidence must not be blank")
        return stripped

    @field_validator("allowed_collider_prim_types")
    @classmethod
    def _canonical_collider_prim_types(
        cls,
        value: tuple[ColliderPrimType, ...],
    ) -> tuple[ColliderPrimType, ...]:
        if not value:
            raise ValueError("allowed_collider_prim_types must not be empty")
        canonical = tuple(
            prim_type
            for prim_type in SUPPORTED_COLLIDER_PRIM_TYPES
            if prim_type in value
        )
        if value != canonical:
            raise ValueError(
                "allowed_collider_prim_types must be unique and ordered as "
                f"{SUPPORTED_COLLIDER_PRIM_TYPES}"
            )
        return value

    @field_validator("density_kg_m3")
    @classmethod
    def _positive_density(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("density_kg_m3 must be finite and positive")
        return value

    @field_validator("volume_fill_fraction")
    @classmethod
    def _valid_fill_fraction(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(
                "volume_fill_fraction must be finite and in the interval (0, 1]"
            )
        return value

    @field_validator(
        "passive_drive_stiffness",
        "passive_drive_target_position",
        "passive_drive_target_velocity",
    )
    @classmethod
    def _required_zero_drive_choice(cls, value: float, info: Any) -> float:
        if not math.isfinite(value) or value != 0.0:
            raise ValueError(f"{info.field_name} must be finite and exactly zero")
        return value

    @field_validator(
        "passive_drive_damping",
        "passive_drive_max_force",
        "passive_drive_max_velocity",
    )
    @classmethod
    def _nonnegative_drive_choice(cls, value: float, info: Any) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{info.field_name} must be finite and nonnegative")
        return value


@dataclass(frozen=True)
class _ColliderGeometry:
    """One exact supported source Gprim selected as a collider."""

    source_prim_path: str
    authored_prim_path: str
    prim_type: ColliderPrimType


@dataclass(frozen=True)
class _LinkGeometry:
    """Resolved exact collider ownership and body-local bounds for one link."""

    colliders: tuple[_ColliderGeometry, ...]
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


def produce_owner_approved_gate3_joint_rigger_input(
    contract: ArticulationContractV1,
    *,
    contract_artifact: ArtifactIdentityV1,
    source_usd_path: str | Path,
    source_asset: ArtifactIdentityV1,
    policy: Gate3PhysicsPolicyV1,
    policy_artifact: ArtifactIdentityV1,
) -> JointRiggerInputV1:
    """Produce and finally admit one complete deterministic Gate 3 request.

    ``policy_artifact`` must identify the canonical JSON representation of
    ``policy``. ``source_asset`` must identify the complete source USD closure
    at ``source_usd_path`` and must already be declared by ``contract``.
    """

    if not isinstance(contract, ArticulationContractV1):
        raise TypeError("contract must be an ArticulationContractV1")
    if not isinstance(contract_artifact, ArtifactIdentityV1):
        raise TypeError("contract_artifact must be an ArtifactIdentityV1")
    if not isinstance(source_asset, ArtifactIdentityV1):
        raise TypeError("source_asset must be an ArtifactIdentityV1")
    if not isinstance(policy, Gate3PhysicsPolicyV1):
        raise TypeError("policy must be a Gate3PhysicsPolicyV1")
    if not isinstance(policy_artifact, ArtifactIdentityV1):
        raise TypeError("policy_artifact must be an ArtifactIdentityV1")

    policy_sha256 = canonical_sha256(policy)
    if policy_artifact.root_sha256 != policy_sha256:
        raise JointRiggerContractError(
            "gate3_policy_identity_mismatch",
            "policy_artifact root_sha256 does not match canonical policy JSON",
        )

    topology_request = build_joint_rigger_input_from_contract(
        contract,
        contract_artifact=contract_artifact,
        source_asset=source_asset,
    )
    source_path = Path(source_usd_path)
    _require_source_identity(
        source_path,
        expected=source_asset,
        mismatch_code="source_asset_identity_mismatch",
    )
    with _identity_bound_source_stage(source_path, expected=source_asset) as stage:
        request = _produce_from_bound_stage(
            stage,
            contract=contract,
            contract_artifact=contract_artifact,
            topology_request=topology_request,
            source_asset=source_asset,
            policy=policy,
            policy_artifact=policy_artifact,
            policy_sha256=policy_sha256,
        )
        del stage

    _require_source_identity(
        source_path,
        expected=source_asset,
        mismatch_code="source_asset_mutated",
    )
    return request


def _produce_from_bound_stage(
    stage: Any,
    *,
    contract: ArticulationContractV1,
    contract_artifact: ArtifactIdentityV1,
    topology_request: JointRiggerInputV1,
    source_asset: ArtifactIdentityV1,
    policy: Gate3PhysicsPolicyV1,
    policy_artifact: ArtifactIdentityV1,
    policy_sha256: str,
) -> JointRiggerInputV1:
    links = tuple(
        sorted(
            (record for record in contract.records if isinstance(record, LinkRecordV1)),
            key=lambda record: record.link_id,
        )
    )
    memberships = _memberships_by_link(contract)
    meters_per_unit = _stage_meters_per_unit(stage)
    binding = _PolicyBinding(
        policy=policy,
        policy_artifact=policy_artifact,
        policy_sha256=policy_sha256,
        source_asset=source_asset,
    )
    geometry = _resolve_link_geometry(
        stage,
        links=links,
        memberships=memberships,
        allowed_collider_prim_types=policy.allowed_collider_prim_types,
    )
    bodies = tuple(
        _build_body_plan(
            link,
            geometry=geometry[link.link_id],
            meters_per_unit=meters_per_unit,
            binding=binding,
        )
        for link in links
    )
    joints = tuple(
        _build_joint_plan(joint, binding=binding)
        for joint in topology_request.plan.joints
    )
    physics_plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=joints,
        rigid_bodies=bodies,
    )
    physics_plan_envelope = Gate3PhysicsPlanEnvelopeV1(
        schema_version=GATE3_PHYSICS_PLAN_ENVELOPE_SCHEMA_VERSION,
        source_asset=source_asset,
        contract_artifact=contract_artifact,
        plan=physics_plan,
    )
    physics_plan_sha256 = canonical_sha256(physics_plan_envelope)
    physics_plan_artifact = ArtifactIdentityV1(
        uri=f"{_PHYSICS_PLAN_URN_PREFIX}{physics_plan_sha256}",
        root_sha256=physics_plan_sha256,
    )
    request: JointRiggerInputV1 = build_gate3_joint_rigger_input_from_contract(
        contract,
        contract_artifact=contract_artifact,
        source_asset=source_asset,
        physics_plan=physics_plan_envelope,
        physics_plan_artifact=physics_plan_artifact,
    )
    return request


@dataclass(frozen=True)
class _PolicyBinding:
    """Canonical identities included in every derived physics provenance."""

    policy: Gate3PhysicsPolicyV1
    policy_artifact: ArtifactIdentityV1
    policy_sha256: str
    source_asset: ArtifactIdentityV1

    def provenance(
        self,
        *,
        prim_path: str,
        properties: tuple[str, ...],
        fact: str,
    ) -> FieldProvenanceV1:
        identity_text = canonical_json(
            {
                "approval_identity": self.policy.approval_identity.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "policy_identity": self.policy_artifact.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "source_identity": self.source_asset.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
            }
        )
        return FieldProvenanceV1(
            source="owner_approved_plan",
            artifact=self.source_asset,
            prim_path=prim_path,
            properties=properties,
            derivation=(
                f"{_PRODUCER_DERIVATION};policy_sha256={self.policy_sha256};"
                f"approval_sha256={self.policy.approval_identity.root_sha256};"
                f"source_sha256={self.source_asset.root_sha256}"
            ),
            evidence=(
                f"{self.policy.approval_evidence} Identities={identity_text}. {fact}"
            ),
        )


@contextmanager
def _identity_bound_source_stage(
    path: Path,
    *,
    expected: ArtifactIdentityV1,
) -> Iterator[Any]:
    """Compose only from a sealed snapshot of the exact requested USD closure."""

    binding: SealedSourceBinding | None = None
    directory: BoundInputDirectory | None = None
    primary_error: BaseException | None = None
    try:
        try:
            binding = create_sealed_source_binding(path, expected=expected)
            projected_path, directory, _ = materialize_bound_input(
                descriptor=binding.descriptor,
                expected_sha256=binding.sha256,
                logical_input_path=path,
                dependencies=bound_input_dependency_snapshots(binding),
                editable_root=False,
            )
            require_sealed_source_binding(binding)
        except Exception as exc:
            raise JointRiggerContractError(
                "source_asset_binding_failed",
                f"could not create the immutable source projection: {exc}",
            ) from exc

        yield _open_source_stage(projected_path)

        try:
            require_sealed_source_binding(binding)
        except Exception as exc:
            raise JointRiggerContractError(
                "source_asset_binding_failed",
                f"immutable source descriptors changed during production: {exc}",
            ) from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if directory is not None:
            try:
                remove_bound_input_directory(directory)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if binding is not None:
            try:
                cleanup_errors.extend(close_source_binding(binding))
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            detail = "; ".join(str(error) for error in cleanup_errors)
            if primary_error is not None:
                primary_error.add_note(f"immutable source cleanup failed: {detail}")
            else:
                raise JointRiggerContractError(
                    "source_binding_cleanup_failed",
                    detail,
                )


def _open_source_stage(path: Path) -> Any:
    from pxr import Usd

    try:
        stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    except Exception as exc:
        raise JointRiggerContractError(
            "source_stage_open_failed",
            f"could not open the identity-bound source stage: {exc}",
        ) from exc
    if stage is None:
        raise JointRiggerContractError(
            "source_stage_open_failed",
            "could not open the identity-bound source stage",
        )
    return stage


def _require_source_identity(
    path: Path,
    *,
    expected: ArtifactIdentityV1,
    mismatch_code: str,
) -> None:
    try:
        observed = identify_usd_artifact(path, uri=expected.uri)
    except JointRiggerContractError as exc:
        if mismatch_code != "source_asset_mutated":
            raise
        raise JointRiggerContractError(
            mismatch_code,
            f"source identity could not be re-established after production: {exc.code}",
        ) from exc
    if observed != expected:
        detail = (
            "source_usd_path does not match source_asset"
            if mismatch_code == "source_asset_identity_mismatch"
            else "source identity changed while the Gate 3 plan was produced"
        )
        raise JointRiggerContractError(mismatch_code, detail)


def _stage_meters_per_unit(stage: Any) -> float:
    from pxr import UsdGeom

    value = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(value) or value <= 0.0:
        raise JointRiggerContractError(
            "invalid_stage_linear_units",
            "OpenUSD metersPerUnit must resolve to a finite positive value",
        )
    return value


def _memberships_by_link(
    contract: ArticulationContractV1,
) -> Mapping[str, tuple[PrimRecordV1, ...]]:
    grouped: dict[str, list[PrimRecordV1]] = {}
    for record in contract.records:
        if isinstance(record, PrimRecordV1):
            grouped.setdefault(record.link_id, []).append(record)
    return {
        link_id: tuple(sorted(records, key=lambda record: record.prim_path))
        for link_id, records in sorted(grouped.items())
    }


def _resolve_link_geometry(
    stage: Any,
    *,
    links: tuple[LinkRecordV1, ...],
    memberships: Mapping[str, tuple[PrimRecordV1, ...]],
    allowed_collider_prim_types: tuple[ColliderPrimType, ...],
) -> Mapping[str, _LinkGeometry]:
    from pxr import Usd, UsdGeom

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    geometry_by_link: dict[str, _LinkGeometry] = {}
    all_body_paths = tuple(link.body_prim_path for link in links)

    for link in links:
        if link.body_authoring == "existing":
            body_frame = stage.GetPrimAtPath(link.body_prim_path)
            _require_usable_prim(
                body_frame,
                path=link.body_prim_path,
                missing_code="link_body_missing",
                ambiguous_code="link_body_ambiguous",
            )
            if not bool(UsdGeom.Xformable(body_frame)):
                raise JointRiggerContractError(
                    "link_body_not_xformable",
                    f"link {link.link_id!r} body {link.body_prim_path} is not "
                    "Xformable",
                )
        else:
            if stage.GetPrimAtPath(link.body_prim_path).IsValid():
                raise JointRiggerContractError(
                    "aggregate_body_collision",
                    f"aggregate body target already exists: {link.body_prim_path}",
                )
            parent_path = link.body_prim_path.rpartition("/")[0] or "/"
            body_frame = stage.GetPrimAtPath(parent_path)
            _require_usable_prim(
                body_frame,
                path=parent_path,
                missing_code="aggregate_parent_missing",
                ambiguous_code="aggregate_parent_ambiguous",
            )
        _require_static_noninstanced_ancestry(
            body_frame,
            label=f"link {link.link_id!r} body frame",
        )
        body_world = xform_cache.GetLocalToWorldTransform(body_frame)

        member_records = memberships.get(link.link_id, ())
        if not member_records:
            raise JointRiggerContractError(
                "link_membership_missing",
                f"link {link.link_id!r} has no explicit PrimRecordV1 membership",
            )
        excluded_roots = tuple(
            path
            for path in all_body_paths
            if path != link.body_prim_path
            and _is_same_or_descendant_path(path, link.body_prim_path)
        )
        colliders_by_path: dict[str, tuple[str, ColliderPrimType]] = {}
        for member in member_records:
            member_prim = stage.GetPrimAtPath(member.prim_path)
            _require_usable_prim(
                member_prim,
                path=member.prim_path,
                missing_code="member_prim_missing",
                ambiguous_code="member_prim_ambiguous",
            )
            _collect_member_colliders(
                member_prim,
                link=link,
                excluded_roots=excluded_roots,
                colliders_by_path=colliders_by_path,
                allowed_collider_prim_types=allowed_collider_prim_types,
            )

        if not colliders_by_path:
            raise JointRiggerContractError(
                "link_geometry_missing",
                f"link {link.link_id!r} memberships contain no owned Gprim geometry",
            )
        colliders = tuple(
            _ColliderGeometry(
                source_prim_path=path,
                authored_prim_path=_authored_collider_path(
                    link,
                    member_path=colliders_by_path[path][0],
                    source_path=path,
                ),
                prim_type=colliders_by_path[path][1],
            )
            for path in sorted(colliders_by_path)
        )
        geometry_by_link[link.link_id] = _compute_body_local_bounds(
            stage,
            body_world=body_world,
            body_owner=link.body_prim_path,
            colliders=colliders,
            xform_cache=xform_cache,
            link_id=link.link_id,
        )
    return geometry_by_link


def _authored_collider_path(
    link: LinkRecordV1,
    *,
    member_path: str,
    source_path: str,
) -> str:
    if link.body_authoring == "existing":
        return source_path
    authored_member = f"{link.body_prim_path}/{member_path.rsplit('/', 1)[-1]}"
    relative_path = source_path.removeprefix(member_path)
    return f"{authored_member}{relative_path}"


def _collect_member_colliders(
    member_prim: Any,
    *,
    link: LinkRecordV1,
    excluded_roots: tuple[str, ...],
    colliders_by_path: dict[str, tuple[str, ColliderPrimType]],
    allowed_collider_prim_types: tuple[ColliderPrimType, ...],
) -> None:
    from pxr import Usd, UsdGeom

    member_path = str(member_prim.GetPath())
    prim_iterator = iter(Usd.PrimRange(member_prim))
    for prim in prim_iterator:
        path = str(prim.GetPath())
        excluded = next(
            (
                root
                for root in excluded_roots
                if _is_same_or_descendant_path(path, root)
            ),
            None,
        )
        if excluded is not None:
            prim_iterator.PruneChildren()
            continue

        _require_noninstanced_prim(prim, label=f"member {member_path}")
        _require_static_transform(prim, label=f"member {member_path}")
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if _source_collider_is_explicitly_disabled(prim, link_id=link.link_id):
            continue
        prim_type = _supported_collider_prim_type(str(prim.GetTypeName()))
        if prim_type is None:
            raise JointRiggerContractError(
                "unsupported_collider_prim_type",
                f"link {link.link_id!r} selected Gprim {path} with type "
                f"{prim.GetTypeName()!r}; supported collider types are "
                f"{SUPPORTED_COLLIDER_PRIM_TYPES}",
            )
        if prim_type not in allowed_collider_prim_types:
            raise JointRiggerContractError(
                "unsupported_collider_prim_type",
                f"link {link.link_id!r} selected {prim_type} {path}; policy allows "
                f"only {allowed_collider_prim_types}",
            )
        if prim_type == "Mesh":
            _require_static_mesh(UsdGeom.Mesh(prim), link_id=link.link_id)
        else:
            _require_static_native_shape(
                prim,
                prim_type=prim_type,
                link_id=link.link_id,
            )
        previous = colliders_by_path.get(path)
        if previous is not None and previous[0] != member_path:
            raise JointRiggerContractError(
                "geometry_membership_overlap",
                f"{prim_type} {path} is selected by explicit members {previous[0]} "
                f"and {member_path} of link {link.link_id!r}",
            )
        colliders_by_path[path] = (member_path, prim_type)


def _source_collider_is_explicitly_disabled(prim: Any, *, link_id: str) -> bool:
    """Return one deterministic, source-authored collider exclusion."""

    from pxr import Sdf, Usd

    path = str(prim.GetPath())
    attribute = prim.GetAttribute("physics:collisionEnabled")
    if not attribute:
        return False
    if attribute.GetTypeName() != Sdf.ValueTypeNames.Bool:
        raise JointRiggerContractError(
            "collider_enablement_ambiguous",
            f"link {link_id!r} Gprim {path} has non-Boolean physics:collisionEnabled",
        )
    if attribute.GetConnections():
        raise JointRiggerContractError(
            "collider_enablement_ambiguous",
            f"link {link_id!r} Gprim {path} has connected physics:collisionEnabled",
        )
    if attribute.GetNumTimeSamples() != 0:
        raise JointRiggerContractError(
            "collider_enablement_ambiguous",
            f"link {link_id!r} Gprim {path} has time-varying physics:collisionEnabled",
        )
    if not attribute.HasAuthoredValueOpinion():
        return False

    value = attribute.Get(Usd.TimeCode.Default())
    if type(value) is not bool:
        raise JointRiggerContractError(
            "collider_enablement_ambiguous",
            f"link {link_id!r} Gprim {path} has no resolved static Boolean "
            "physics:collisionEnabled value",
        )
    return not value


def _supported_collider_prim_type(value: str) -> ColliderPrimType | None:
    if value == "Mesh":
        return "Mesh"
    if value == "Cube":
        return "Cube"
    if value == "Sphere":
        return "Sphere"
    if value == "Capsule":
        return "Capsule"
    if value == "Cylinder":
        return "Cylinder"
    if value == "Cone":
        return "Cone"
    return None


def _require_usable_prim(
    prim: Any,
    *,
    path: str,
    missing_code: str,
    ambiguous_code: str,
) -> None:
    if not prim or not prim.IsValid():
        raise JointRiggerContractError(
            missing_code, f"source prim does not exist: {path}"
        )
    if not prim.IsActive() or not prim.IsDefined() or prim.IsAbstract():
        raise JointRiggerContractError(
            ambiguous_code,
            f"source prim is inactive, undefined, or abstract: {path}",
        )


def _require_static_noninstanced_ancestry(prim: Any, *, label: str) -> None:
    current = prim
    while current and not current.IsPseudoRoot():
        _require_noninstanced_prim(current, label=label)
        _require_static_transform(current, label=label)
        current = current.GetParent()


def _require_noninstanced_prim(prim: Any, *, label: str) -> None:
    path = str(prim.GetPath())
    if prim.IsPrototype() or prim.IsInPrototype():
        raise JointRiggerContractError(
            "prototype_geometry_unsupported",
            f"{label} resolves prototype geometry at {path}",
        )
    if prim.IsInstance() or prim.IsInstanceProxy() or prim.IsInstanceable():
        raise JointRiggerContractError(
            "instance_geometry_unsupported",
            f"{label} resolves instance or instanceable geometry at {path}",
        )


def _require_static_transform(prim: Any, *, label: str) -> None:
    for attribute in prim.GetAttributes():
        name = attribute.GetName()
        if name != "xformOpOrder" and not name.startswith("xformOp:"):
            continue
        samples = tuple(float(value) for value in attribute.GetTimeSamples())
        if samples:
            raise JointRiggerContractError(
                "time_varying_geometry_unsupported",
                f"{label} has time-sampled transform {_path_and_property(prim, name)} "
                f"at {samples}",
            )


def _require_static_mesh(mesh: Any, *, link_id: str) -> None:
    path = str(mesh.GetPath())
    for name in _MESH_GEOMETRY_ATTRIBUTES:
        attribute = mesh.GetPrim().GetAttribute(name)
        samples = tuple(float(value) for value in attribute.GetTimeSamples())
        if samples:
            raise JointRiggerContractError(
                "time_varying_geometry_unsupported",
                f"link {link_id!r} Mesh {path} has time-sampled geometry "
                f"attribute {name} at {samples}",
            )


def _require_static_native_shape(
    prim: Any,
    *,
    prim_type: ColliderPrimType,
    link_id: str,
) -> None:
    path = str(prim.GetPath())
    attributes = _NATIVE_SHAPE_GEOMETRY_ATTRIBUTES[prim_type]
    for name in attributes:
        attribute = prim.GetAttribute(name)
        samples = tuple(float(value) for value in attribute.GetTimeSamples())
        if samples:
            raise JointRiggerContractError(
                "time_varying_geometry_unsupported",
                f"link {link_id!r} {prim_type} {path} has time-sampled geometry "
                f"attribute {name} at {samples}",
            )

    if prim_type == "Cube":
        _require_positive_shape_attribute(prim, name="size", prim_type=prim_type)
        return
    if prim_type == "Sphere":
        _require_positive_shape_attribute(prim, name="radius", prim_type=prim_type)
        return
    _require_positive_shape_attribute(prim, name="height", prim_type=prim_type)
    _require_positive_shape_attribute(prim, name="radius", prim_type=prim_type)
    axis = prim.GetAttribute("axis").Get()
    if axis not in {"X", "Y", "Z"}:
        raise JointRiggerContractError(
            "geometry_bounds_ambiguous",
            f"{prim_type} {path} has unsupported axis {axis!r}",
        )


def _require_positive_shape_attribute(
    prim: Any,
    *,
    name: str,
    prim_type: ColliderPrimType,
) -> None:
    value = prim.GetAttribute(name).Get()
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise JointRiggerContractError(
            "geometry_bounds_ambiguous",
            f"{prim_type} {prim.GetPath()} has no numeric {name}",
        ) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise JointRiggerContractError(
            "geometry_bounds_ambiguous",
            f"{prim_type} {prim.GetPath()} has nonpositive or non-finite "
            f"{name}={number}",
        )


def _compute_body_local_bounds(
    stage: Any,
    *,
    body_world: Any,
    body_owner: str,
    colliders: tuple[_ColliderGeometry, ...],
    xform_cache: Any,
    link_id: str,
) -> _LinkGeometry:
    from pxr import Gf

    try:
        _require_invertible_matrix(body_world, owner=body_owner)
        body_world_inverse = body_world.GetInverse()
        body_axis_scales = _body_axis_scales(
            body_world,
            owner=body_owner,
        )
        minimum = [math.inf, math.inf, math.inf]
        maximum = [-math.inf, -math.inf, -math.inf]
        for collider in colliders:
            prim = stage.GetPrimAtPath(collider.source_prim_path)
            points = _collider_local_bound_points(
                prim,
                prim_type=collider.prim_type,
            )
            collider_world = xform_cache.GetLocalToWorldTransform(prim)
            _require_finite_matrix(collider_world, owner=collider.source_prim_path)
            for point in points:
                world_point = collider_world.Transform(Gf.Vec3d(*point))
                local_point = body_world_inverse.Transform(world_point)
                values = tuple(
                    float(local_point[index]) * body_axis_scales[index]
                    for index in range(3)
                )
                if not all(math.isfinite(value) for value in values):
                    raise JointRiggerContractError(
                        "geometry_bounds_ambiguous",
                        f"{collider.prim_type} {collider.source_prim_path} produced "
                        "non-finite body-local bounds",
                    )
                for index, value in enumerate(values):
                    minimum[index] = min(minimum[index], value)
                    maximum[index] = max(maximum[index], value)
    except JointRiggerContractError:
        raise
    except Exception as exc:
        raise JointRiggerContractError(
            "geometry_bounds_ambiguous",
            f"could not resolve body-local bounds for link {link_id!r}: {exc}",
        ) from exc

    return _LinkGeometry(
        colliders=colliders,
        minimum=(minimum[0], minimum[1], minimum[2]),
        maximum=(maximum[0], maximum[1], maximum[2]),
    )


def _collider_local_bound_points(
    prim: Any,
    *,
    prim_type: ColliderPrimType,
) -> tuple[Any, ...]:
    from pxr import Usd, UsdGeom

    if prim_type == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
        counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
        indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
        if points is None or len(points) < 3 or not counts or not indices:
            raise JointRiggerContractError(
                "geometry_bounds_ambiguous",
                f"Mesh {prim.GetPath()} requires points and nonempty face topology",
            )
        valid_topology, reason = UsdGeom.Mesh.ValidateTopology(
            indices,
            counts,
            len(points),
        )
        if not valid_topology:
            raise JointRiggerContractError(
                "geometry_bounds_ambiguous",
                f"Mesh {prim.GetPath()} topology is invalid: {reason}",
            )
        return tuple(points)

    boundable = UsdGeom.Boundable(prim)
    extent = UsdGeom.Boundable.ComputeExtentFromPlugins(
        boundable,
        Usd.TimeCode.Default(),
    )
    if extent is None or len(extent) != 2:
        raise JointRiggerContractError(
            "geometry_bounds_ambiguous",
            f"OpenUSD could not compute an extent for {prim_type} {prim.GetPath()}",
        )
    minimum = tuple(float(extent[0][index]) for index in range(3))
    maximum = tuple(float(extent[1][index]) for index in range(3))
    if not all(
        math.isfinite(minimum[index])
        and math.isfinite(maximum[index])
        and minimum[index] < maximum[index]
        for index in range(3)
    ):
        raise JointRiggerContractError(
            "geometry_bounds_ambiguous",
            f"{prim_type} {prim.GetPath()} produced an invalid OpenUSD extent",
        )
    return tuple(
        (x, y, z)
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    )


def _require_invertible_matrix(matrix: Any, *, owner: str) -> None:
    _require_finite_matrix(matrix, owner=owner)
    determinant = float(matrix.GetDeterminant())
    if not math.isfinite(determinant) or determinant == 0.0:
        raise JointRiggerContractError(
            "geometry_bounds_ambiguous",
            f"body transform is singular at {owner}",
        )


def _require_finite_matrix(matrix: Any, *, owner: str) -> None:
    values = (float(matrix[row][column]) for row in range(4) for column in range(4))
    if not all(math.isfinite(value) for value in values):
        raise JointRiggerContractError(
            "geometry_bounds_ambiguous",
            f"transform contains non-finite values at {owner}",
        )


def _body_axis_scales(
    matrix: Any,
    *,
    owner: str,
) -> tuple[float, float, float]:
    """Return physical body-axis scale while rejecting shear and reflection."""

    from pxr import Gf

    basis = tuple(
        matrix.TransformDir(direction)
        for direction in (
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 1.0, 0.0),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
    )
    scales = tuple(float(direction.GetLength()) for direction in basis)
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise JointRiggerContractError(
            "geometry_bounds_ambiguous",
            f"body transform has invalid axis scale at {owner}: {scales}",
        )
    normalized = tuple(
        direction / scale for direction, scale in zip(basis, scales, strict=True)
    )
    dots = (
        float(Gf.Dot(normalized[0], normalized[1])),
        float(Gf.Dot(normalized[0], normalized[2])),
        float(Gf.Dot(normalized[1], normalized[2])),
    )
    handedness = float(Gf.Dot(Gf.Cross(normalized[0], normalized[1]), normalized[2]))
    if any(abs(value) > 1e-6 for value in dots) or handedness < (1.0 - 1e-6):
        raise JointRiggerContractError(
            "geometry_bounds_ambiguous",
            f"body transform has shear or reflection at {owner}",
        )
    return (scales[0], scales[1], scales[2])


def _build_body_plan(
    link: LinkRecordV1,
    *,
    geometry: _LinkGeometry,
    meters_per_unit: float,
    binding: _PolicyBinding,
) -> RigidBodyPlanV1:
    dimensions_stage = tuple(
        geometry.maximum[index] - geometry.minimum[index] for index in range(3)
    )
    dimensions_m = tuple(value * meters_per_unit for value in dimensions_stage)
    if not all(math.isfinite(value) and value > 0.0 for value in dimensions_m):
        raise JointRiggerContractError(
            "nonpositive_link_volume",
            f"link {link.link_id!r} body-local bounds have nonpositive dimensions "
            f"{dimensions_stage}",
        )
    box_volume_m3 = math.prod(dimensions_m)
    mass_kg = (
        binding.policy.density_kg_m3
        * binding.policy.volume_fill_fraction
        * box_volume_m3
    )
    if not math.isfinite(box_volume_m3) or not math.isfinite(mass_kg):
        raise JointRiggerContractError(
            "geometry_mass_overflow",
            f"link {link.link_id!r} bounds produce non-finite volume or mass",
        )
    x_size, y_size, z_size = dimensions_m
    inertia = (
        mass_kg * (y_size * y_size + z_size * z_size) / 12.0,
        mass_kg * (x_size * x_size + z_size * z_size) / 12.0,
        mass_kg * (x_size * x_size + y_size * y_size) / 12.0,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in inertia):
        raise JointRiggerContractError(
            "geometry_mass_overflow",
            f"link {link.link_id!r} bounds produce invalid box inertia",
        )

    center_stage = (
        (geometry.minimum[0] + geometry.maximum[0]) / 2.0,
        (geometry.minimum[1] + geometry.maximum[1]) / 2.0,
        (geometry.minimum[2] + geometry.maximum[2]) / 2.0,
    )
    center_of_mass_m = (
        center_stage[0] * meters_per_unit,
        center_stage[1] * meters_per_unit,
        center_stage[2] * meters_per_unit,
    )
    mass_fact = canonical_json(
        {
            "body_local_bbox_center_stage": center_stage,
            "body_local_bbox_dimensions_stage": dimensions_stage,
            "box_volume_m3": box_volume_m3,
            "density_kg_m3": binding.policy.density_kg_m3,
            "mass_kg": mass_kg,
            "meters_per_unit": meters_per_unit,
            "volume_fill_fraction": binding.policy.volume_fill_fraction,
        }
    )
    mass = MassPropertiesV1(
        mass_kg=mass_kg,
        center_of_mass_m=center_of_mass_m,
        diagonal_inertia_kg_m2=inertia,
        principal_axes=(1.0, 0.0, 0.0, 0.0),
        provenance=binding.provenance(
            prim_path=link.body_prim_path,
            properties=(
                "body_local_axis_aligned_bounds",
                "density_kg_m3",
                "volume_fill_fraction",
            ),
            fact=(
                "Mass is density times fill fraction times the positive SI box "
                f"volume; diagonal box inertia is around the local bbox center. "
                f"Facts={mass_fact}"
            ),
        ),
    )
    colliders = tuple(
        _build_collider_plan(collider, binding=binding)
        for collider in geometry.colliders
    )
    return RigidBodyPlanV1(
        prim_path=link.body_prim_path,
        mass=mass,
        colliders=colliders,
        provenance=binding.provenance(
            prim_path=link.body_prim_path,
            properties=("rigid_body", "explicit_prim_memberships"),
            fact=(
                f"Link {link.link_id!r} has exact rigid-body coverage from "
                f"supported Gprims "
                f"{[(item.source_prim_path, item.authored_prim_path, item.prim_type) for item in geometry.colliders]}."
            ),
        ),
    )


def _build_collider_plan(
    collider: _ColliderGeometry,
    *,
    binding: _PolicyBinding,
) -> ColliderPlanV1:
    properties: tuple[str, ...]
    if collider.prim_type == "Mesh":
        approximation: MeshColliderApproximation | None = (
            binding.policy.mesh_collider_approximation
        )
        properties = (
            "allowed_collider_prim_types",
            "faceVertexCounts",
            "faceVertexIndices",
            "points",
        )
        fact = (
            f"Exact owned Mesh {collider.source_prim_path} is selected as collider "
            f"{collider.authored_prim_path} with explicit approximation "
            f"{approximation!r}."
        )
    else:
        approximation = None
        properties = (
            "allowed_collider_prim_types",
            *_NATIVE_SHAPE_GEOMETRY_ATTRIBUTES[collider.prim_type],
        )
        fact = (
            f"Exact owned native {collider.prim_type} "
            f"{collider.source_prim_path} is selected as collider "
            f"{collider.authored_prim_path} without mesh-only API evidence."
        )
    return ColliderPlanV1(
        prim_path=collider.authored_prim_path,
        mesh_approximation=approximation,
        provenance=binding.provenance(
            prim_path=collider.source_prim_path,
            properties=properties,
            fact=fact,
        ),
    )


def _build_joint_plan(joint: JointPlanV1, *, binding: _PolicyBinding) -> JointPlanV1:
    topology = joint.topology
    if topology.joint_type == "spherical":
        return JointPlanV1(topology=topology)
    if joint.limit is not None:
        lower = joint.limit.lower
        upper = joint.limit.upper
        if (lower is not None and lower > 0.0) or (upper is not None and upper < 0.0):
            raise JointRiggerContractError(
                "zero_rest_state_outside_limit",
                f"joint {topology.joint_id!r} has an explicit limit excluding zero",
            )

    state = JointStateV1(
        position=binding.policy.passive_drive_target_position,
        velocity=binding.policy.passive_drive_target_velocity,
        provenance=binding.provenance(
            prim_path=topology.body1,
            properties=("joint_rest_position", "joint_rest_velocity"),
            fact=(
                f"Non-spherical joint {topology.joint_id!r} uses the explicit "
                "zero position and zero velocity policy rest state."
            ),
        ),
    )
    drive = JointDriveV1(
        drive_type=binding.policy.passive_drive_type,
        stiffness=binding.policy.passive_drive_stiffness,
        damping=binding.policy.passive_drive_damping,
        max_force=binding.policy.passive_drive_max_force,
        target_position=binding.policy.passive_drive_target_position,
        target_velocity=binding.policy.passive_drive_target_velocity,
        max_joint_velocity=binding.policy.passive_drive_max_velocity,
        provenance=binding.provenance(
            prim_path=topology.body1,
            properties=(
                "passive_drive_damping",
                "passive_drive_max_force",
                "passive_drive_max_velocity",
            ),
            fact=(
                f"Non-spherical joint {topology.joint_id!r} uses an explicit "
                "zero-stiffness, zero-target passive force drive with damping="
                f"{binding.policy.passive_drive_damping}, max_force="
                f"{binding.policy.passive_drive_max_force}, and max_velocity="
                f"{binding.policy.passive_drive_max_velocity}."
            ),
        ),
    )
    # Contract limits remain solely topology-owned and are merged by final Gate
    # 3 admission. This producer never creates or copies an independent opinion.
    return JointPlanV1(topology=topology, state=state, drive=drive)


def _is_same_or_descendant_path(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _path_and_property(prim: Any, property_name: str) -> str:
    """Return a stable diagnostic locator without consulting local filenames."""

    return f"{prim.GetPath()}.{property_name}"


__all__ = [
    "ColliderPrimType",
    "GATE3_PHYSICS_POLICY_SCHEMA_VERSION",
    "Gate3PhysicsPolicyV1",
    "MeshColliderApproximation",
    "SUPPORTED_COLLIDER_PRIM_TYPES",
    "produce_owner_approved_gate3_joint_rigger_input",
]
