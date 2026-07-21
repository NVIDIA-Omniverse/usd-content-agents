# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the shared Joint Rigger v1 contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys

import pytest
from pydantic import TypeAdapter, ValidationError

from world_understanding.functions.physics.joint_rigger.models import (
    DIAGNOSTICS_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION_V2,
    PLAN_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION_V2,
    RESULT_SCHEMA_VERSION,
    ArticulationRootPlanV1,
    ArtifactIdentityV1,
    ColliderPlanV1,
    FieldDecisionV1,
    FieldProvenanceV1,
    JointAnchorV1,
    JointDiagnosticV1,
    JointDriveV1,
    JointFrictionV1,
    JointLimitV1,
    JointMimicV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerDiagnosticsV1,
    JointRiggerInputV1,
    JointRiggerInputV2,
    JointRiggerPlanV1,
    JointRiggerPlanV2,
    JointRiggerResultV1,
    JointStateV1,
    JointTopologyV1,
    LegacyComponentAssignmentV1,
    LegacyComponentNameCompatibilityV1,
    MassPropertiesV1,
    RigidBodyPlanV1,
    RigidLinkMemberPlanV1,
    RigidLinkPlanV1,
    canonical_json,
    canonical_sha256,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _artifact(
    *,
    uri: str = "s3://example/assets/source.usdz",
    root_sha256: str = SHA_A,
) -> ArtifactIdentityV1:
    return ArtifactIdentityV1(uri=uri, root_sha256=root_sha256)


def _provenance(
    field: str,
    *,
    prim_path: str | None = None,
    properties: tuple[str, ...] = (),
) -> FieldProvenanceV1:
    prim_path = prim_path or "/World/reference"
    properties = properties or (field,)
    return FieldProvenanceV1(
        source="authored_reference",
        artifact=_artifact(),
        prim_path=prim_path,
        properties=properties,
        evidence=f"reference evidence for {field}",
    )


def _topology(
    joint_id: str = "joint_a",
    *,
    joint_type: str = "revolute",
    body0: str = "/World/base",
    body1: str = "/World/link_a",
    axis_stage: tuple[float, float, float] | None = None,
) -> JointTopologyV1:
    provenance = {
        field: _provenance(field, prim_path=body0 if field == "body0" else body1)
        for field in ("joint_type", "body0", "body1")
    }
    if joint_type in {"revolute", "prismatic"}:
        axis_stage = axis_stage if axis_stage is not None else (0.0, 0.0, 1.0)
        provenance["axis_stage"] = _provenance(
            "axis_stage",
            properties=("physics:localRot1", "physics:axis"),
        )
    return JointTopologyV1(
        joint_id=joint_id,
        joint_type=joint_type,
        body0=body0,
        body1=body1,
        axis_stage=axis_stage,
        field_provenance=provenance,
    )


def _joint(
    joint_id: str = "joint_a",
    *,
    joint_type: str = "revolute",
    body0: str = "/World/base",
    body1: str = "/World/link_a",
    **facts: object,
) -> JointPlanV1:
    return JointPlanV1(
        topology=_topology(
            joint_id,
            joint_type=joint_type,
            body0=body0,
            body1=body1,
        ),
        **facts,
    )


def _mimic(reference_joint_id: str) -> JointMimicV1:
    return JointMimicV1(
        reference_joint_id=reference_joint_id,
        gearing=1.0,
        offset=0.0,
        natural_frequency=5.0,
        damping_ratio=0.7,
        provenance=_provenance("mimic"),
    )


def _friction(coefficient: float = 0.15) -> JointFrictionV1:
    return JointFrictionV1(
        coefficient=coefficient,
        provenance=_provenance(
            "joint friction",
            properties=("physxJoint:jointFriction",),
        ),
    )


def _diagnostics() -> JointRiggerDiagnosticsV1:
    return JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="owned-core",
        backend_version="1.0.0",
        field_decisions=(
            FieldDecisionV1(
                field="plan",
                disposition="accepted",
                provenance=_provenance("plan"),
            ),
        ),
    )


def _v2_request() -> JointRiggerInputV2:
    plan = JointRiggerPlanV2(
        schema_version=PLAN_SCHEMA_VERSION_V2,
        joints=(
            _joint(
                "drawer_slide",
                joint_type="prismatic",
                body0="/World/base",
                body1="/World/drawer",
            ),
        ),
        articulation_roots=(
            ArticulationRootPlanV1(
                prim_path="/World/base",
                provenance=_provenance(
                    "articulation_root",
                    prim_path="/World/base",
                ),
            ),
        ),
    )
    return JointRiggerInputV2(
        schema_version=INPUT_SCHEMA_VERSION_V2,
        source_asset=_artifact(),
        plan=plan,
        rigid_links=(
            RigidLinkPlanV1(
                link_id="drawer",
                body_authoring="aggregate",
                body_prim_path="/World/drawer",
                members=(
                    RigidLinkMemberPlanV1(
                        source_prim_path="/World/panel_b",
                        authored_prim_path="/World/drawer/panel_b",
                    ),
                    RigidLinkMemberPlanV1(
                        source_prim_path="/World/panel_a",
                        authored_prim_path="/World/drawer/panel_a",
                    ),
                ),
            ),
            RigidLinkPlanV1(
                link_id="base",
                body_authoring="existing",
                body_prim_path="/World/base",
                members=(
                    RigidLinkMemberPlanV1(
                        source_prim_path="/World/base",
                        authored_prim_path="/World/base",
                    ),
                ),
            ),
        ),
    )


def test_all_contract_models_are_strict_frozen_and_forbid_extra_fields() -> None:
    model_types = (
        ArtifactIdentityV1,
        FieldProvenanceV1,
        JointTopologyV1,
        JointLimitV1,
        JointAnchorV1,
        JointDriveV1,
        JointFrictionV1,
        JointStateV1,
        JointMimicV1,
        MassPropertiesV1,
        ColliderPlanV1,
        RigidBodyPlanV1,
        ArticulationRootPlanV1,
        JointPlanV1,
        JointRiggerPlanV1,
        JointRiggerPlanV2,
        LegacyComponentAssignmentV1,
        LegacyComponentNameCompatibilityV1,
        JointRiggerInputV1,
        RigidLinkMemberPlanV1,
        RigidLinkPlanV1,
        JointRiggerInputV2,
        FieldDecisionV1,
        JointDiagnosticV1,
        JointRiggerDiagnosticsV1,
        JointRiggerResultV1,
    )

    for model_type in model_types:
        assert model_type.model_config["extra"] == "forbid"
        assert model_type.model_config["frozen"] is True
        assert model_type.model_config["strict"] is True

    artifact = _artifact()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ArtifactIdentityV1(
            uri=artifact.uri,
            root_sha256=artifact.root_sha256,
            unexpected=True,
        )
    with pytest.raises(ValidationError, match="Instance is frozen"):
        artifact.uri = "s3://example/changed.usdz"


def test_strict_contract_numerics_reject_python_and_json_coercion() -> None:
    provenance = _provenance("strict numeric")
    valid_mass = MassPropertiesV1(
        mass_kg=1.0,
        diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
        provenance=provenance,
    )
    valid_state = JointStateV1(
        position=0.0,
        velocity=2.5,
        provenance=provenance,
    )
    valid_friction = JointFrictionV1(
        coefficient=0.15,
        provenance=provenance,
    )
    cases = (
        (MassPropertiesV1, valid_mass, "mass_kg", True),
        (JointStateV1, valid_state, "velocity", "2.5"),
        (JointFrictionV1, valid_friction, "coefficient", "0.15"),
    )

    for model_type, valid, field, invalid_value in cases:
        python_payload = valid.model_dump(mode="python")
        python_payload[field] = invalid_value
        with pytest.raises(ValidationError) as python_error:
            model_type.model_validate(python_payload)
        assert python_error.value.errors()[0]["loc"] == (field,)
        assert python_error.value.errors()[0]["type"] == "float_type"

        json_payload = valid.model_dump(mode="json")
        json_payload[field] = invalid_value
        with pytest.raises(ValidationError) as json_error:
            model_type.model_validate_json(json.dumps(json_payload))
        assert json_error.value.errors()[0]["loc"] == (field,)
        assert json_error.value.errors()[0]["type"] == "float_type"

        assert model_type.model_validate_json(canonical_json(valid)) == valid


@pytest.mark.parametrize(
    "digest",
    ["a" * 63, "a" * 65, "A" * 64, "g" * 64, " a" + "a" * 63],
)
def test_artifact_identity_requires_normalized_lowercase_sha256(digest: str) -> None:
    with pytest.raises(ValidationError, match="lowercase 64-character SHA-256"):
        ArtifactIdentityV1(uri="source.usda", root_sha256=digest)


def test_artifact_identity_and_provenance_normalize_deterministic_properties() -> None:
    artifact = ArtifactIdentityV1(
        uri="s3://example/source.usdz",
        root_sha256=SHA_A,
        dependency_bundle_sha256=SHA_B,
    )
    provenance = FieldProvenanceV1(
        source="authored_reference",
        artifact=artifact,
        prim_path="/World/link",
        properties=("physics:mass", "physics:axis", "physics:mass"),
        derivation="copied without inference",
        evidence="paired rigged reference",
    )

    assert provenance.properties == ("physics:axis", "physics:mass")
    assert provenance.artifact == artifact

    with pytest.raises(ValidationError, match="must not be blank"):
        FieldProvenanceV1(source="authored_reference", evidence="  ")
    with pytest.raises(ValidationError, match="absolute non-root USD prim path"):
        FieldProvenanceV1(
            source="authored_reference",
            artifact=artifact,
            prim_path="World/link",
            properties=("physics:axis",),
            evidence="evidence",
        )


@pytest.mark.parametrize(
    "source",
    [
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
    ],
)
def test_artifact_backed_provenance_requires_exact_source_locator(
    source: str,
) -> None:
    with pytest.raises(ValidationError, match="requires an artifact identity"):
        FieldProvenanceV1(
            source=source,
            prim_path="/World/link",
            properties=("physics:axis",),
            evidence="source-backed evidence",
        )
    with pytest.raises(ValidationError, match="requires a prim_path"):
        FieldProvenanceV1(
            source=source,
            artifact=_artifact(),
            properties=("physics:axis",),
            evidence="source-backed evidence",
        )
    with pytest.raises(ValidationError, match="requires at least one property"):
        FieldProvenanceV1(
            source=source,
            artifact=_artifact(),
            prim_path="/World/link",
            evidence="source-backed evidence",
        )


def test_owner_plan_and_template_default_provenance_need_no_artifact_locator() -> None:
    for source in ("owner_approved_plan", "template_default"):
        provenance = FieldProvenanceV1(
            source=source,
            prim_path=None,
            evidence="Explicit non-artifact-backed decision.",
        )
        assert provenance.artifact is None
        assert provenance.prim_path is None
        assert provenance.properties == ()


@pytest.mark.parametrize("source", ["owner_approved_plan", "template_default"])
def test_optional_provenance_locator_is_complete_or_absent(source: str) -> None:
    with pytest.raises(ValidationError, match="must provide artifact, prim_path"):
        FieldProvenanceV1(
            source=source,
            artifact=_artifact(),
            evidence="Incomplete optional locator.",
        )
    with pytest.raises(ValidationError, match="must provide artifact, prim_path"):
        FieldProvenanceV1(
            source=source,
            prim_path="/World/link",
            properties=("physics:axis",),
            evidence="Incomplete optional locator.",
        )

    complete = FieldProvenanceV1(
        source=source,
        artifact=_artifact(),
        prim_path="/World/link",
        properties=("physics:axis",),
        evidence="Complete optional locator.",
    )
    assert FieldProvenanceV1.model_validate_json(canonical_json(complete)) == complete


@pytest.mark.parametrize(
    "path",
    ["World/link", "/", "/World//link", "/World/link/", "/World/../link"],
)
def test_topology_requires_absolute_non_root_usd_prim_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="absolute non-root|invalid path segment"):
        _topology(body1=path)


@pytest.mark.parametrize(
    "path",
    [
        "/World/body.attr",
        "/World/A{model=rigged}",
        "/World/bad name",
        "/World/123body",
        "/World/body-name",
        "/World.rel[/Other]",
        "/World/body\n",
        "/World/body\r\n",
        "World/body",
    ],
)
def test_every_prim_path_contract_rejects_noncanonical_or_non_prim_sdf_paths(
    path: str,
) -> None:
    provenance = _provenance("path target")
    builders = (
        lambda: _provenance("field", prim_path=path),
        lambda: _topology(body1=path),
        lambda: ColliderPlanV1(prim_path=path, provenance=provenance),
        lambda: RigidBodyPlanV1(prim_path=path, provenance=provenance),
        lambda: ArticulationRootPlanV1(prim_path=path, provenance=provenance),
        lambda: LegacyComponentAssignmentV1(
            prim_path=path,
            component_name="component",
            source_field="component_name",
        ),
    )

    for build in builders:
        with pytest.raises(ValidationError, match="valid absolute non-root USD prim"):
            build()


@pytest.mark.parametrize("path", ["/World", "/World/door_01", "/世界/门_1"])
def test_every_prim_path_contract_preserves_valid_absolute_prim_paths(
    path: str,
) -> None:
    provenance = _provenance("path target")

    assert _provenance("field", prim_path=path).prim_path == path
    assert _topology(body1=path).body1 == path
    assert ColliderPlanV1(prim_path=path, provenance=provenance).prim_path == path
    assert RigidBodyPlanV1(prim_path=path, provenance=provenance).prim_path == path
    assert (
        ArticulationRootPlanV1(prim_path=path, provenance=provenance).prim_path == path
    )
    assignment = LegacyComponentAssignmentV1(
        prim_path=path,
        component_name="component",
        source_field="component_name",
    )
    assert assignment.prim_path == path


def test_prim_path_contract_has_strict_fallback_without_pxr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pxr", None)

    assert _topology(body1="/世界/部件_1").body1 == "/世界/部件_1"
    for invalid in (
        "World/body",
        "/World/body.attr",
        "/World/A{v=x}",
        "/World/body\n",
        "/World/body\r\n",
    ):
        with pytest.raises(ValidationError, match="valid absolute non-root USD prim"):
            _topology(body1=invalid)


def test_topology_requires_distinct_endpoints() -> None:
    with pytest.raises(ValidationError, match="same_body_endpoints"):
        _topology(body0="/World/link", body1="/World/link")


def test_axis_rules_preserve_a_normalized_signed_stage_frame_vector() -> None:
    topology = _topology(axis_stage=(-1.0, 0.0, 0.0))
    assert topology.axis_stage == (-1.0, 0.0, 0.0)

    with pytest.raises(ValidationError, match="require axis_stage"):
        JointTopologyV1(
            joint_id="missing_axis",
            joint_type="prismatic",
            body0="/World/base",
            body1="/World/slider",
            field_provenance={
                field: _provenance(field)
                for field in ("joint_type", "body0", "body1", "axis_stage")
            },
        )
    with pytest.raises(ValidationError, match="normalized vector"):
        _topology(axis_stage=(2.0, 0.0, 0.0))
    with pytest.raises(ValidationError, match="must be finite"):
        _topology(axis_stage=(math.inf, 0.0, 0.0))

    spherical = _topology(joint_type="spherical", axis_stage=None)
    assert spherical.axis_stage is None
    with pytest.raises(ValidationError, match="must not carry axis_stage"):
        JointTopologyV1(
            joint_id="ball",
            joint_type="spherical",
            body0="/World/base",
            body1="/World/ball",
            axis_stage=(0.0, 1.0, 0.0),
            field_provenance={
                field: _provenance(field) for field in ("joint_type", "body0", "body1")
            },
        )


def test_topology_requires_exact_field_provenance_keys() -> None:
    base = {
        field: _provenance(field)
        for field in ("joint_type", "body0", "body1", "axis_stage")
    }
    for invalid in (
        {key: value for key, value in base.items() if key != "body1"},
        {**base, "anchor": _provenance("anchor")},
    ):
        with pytest.raises(ValidationError, match="keys must exactly match"):
            JointTopologyV1(
                joint_id="hinge",
                joint_type="revolute",
                body0="/World/base",
                body1="/World/door",
                axis_stage=(0.0, 1.0, 0.0),
                field_provenance=invalid,
            )


def test_topology_field_provenance_is_deeply_immutable_and_hash_stable() -> None:
    provenance = {
        field: _provenance(field)
        for field in ("joint_type", "body0", "body1", "axis_stage")
    }
    topology = JointTopologyV1(
        joint_id="hinge",
        joint_type="revolute",
        body0="/World/base",
        body1="/World/door",
        axis_stage=(0.0, 1.0, 0.0),
        field_provenance=provenance,
    )
    expected_hash = canonical_sha256(topology)
    expected_json = canonical_json(topology)

    provenance["joint_type"] = _provenance("changed caller mapping")
    with pytest.raises(TypeError, match="does not support item assignment"):
        topology.field_provenance["joint_type"] = _provenance(  # type: ignore[index]
            "attempted mutation"
        )

    assert canonical_sha256(topology) == expected_hash
    assert canonical_json(topology) == expected_json
    reverse_order = JointTopologyV1(
        joint_id="hinge",
        joint_type="revolute",
        body0="/World/base",
        body1="/World/door",
        axis_stage=(0.0, 1.0, 0.0),
        field_provenance={
            field: _provenance(field)
            for field in ("axis_stage", "body1", "body0", "joint_type")
        },
    )
    assert reverse_order.model_dump_json() == topology.model_dump_json()
    assert list(json.loads(topology.model_dump_json())["field_provenance"]) == sorted(
        provenance
    )
    round_trip = JointTopologyV1.model_validate_json(expected_json)
    assert round_trip == topology
    with pytest.raises(TypeError, match="does not support item assignment"):
        round_trip.field_provenance["body0"] = _provenance(  # type: ignore[index]
            "round-trip mutation"
        )

    for copied in (copy.deepcopy(topology), topology.model_copy(deep=True)):
        assert copied is not topology
        assert copied == topology
        assert copied.field_provenance is not topology.field_provenance
        assert canonical_json(copied) == expected_json
        with pytest.raises(TypeError, match="does not support item assignment"):
            copied.field_provenance["body1"] = _provenance(  # type: ignore[index]
                "deep-copy mutation"
            )

    spherical = JointTopologyV1(
        joint_id="ball",
        joint_type="spherical",
        body0="/World/base",
        body1="/World/ball",
        field_provenance={
            field: _provenance(field) for field in ("joint_type", "body0", "body1")
        },
    )
    expected_fields_set = spherical.model_fields_set
    expected_unset_dump = spherical.model_dump(exclude_unset=True)
    for copied in (copy.deepcopy(spherical), spherical.model_copy(deep=True)):
        assert copied.model_fields_set == expected_fields_set
        assert copied.model_dump(exclude_unset=True) == expected_unset_dump
        for field, item in copied.field_provenance.items():
            assert (
                item.model_fields_set
                == spherical.field_provenance[field].model_fields_set
            )


def test_joint_limit_is_source_backed_finite_and_type_appropriate() -> None:
    provenance = _provenance("limit")
    limit = JointLimitV1(lower=-45.0, upper=90.0, unit="degrees", provenance=provenance)
    assert _joint(limit=limit).limit == limit

    with pytest.raises(ValidationError, match="at least one authored limit"):
        JointLimitV1(unit="degrees", provenance=provenance)
    with pytest.raises(ValidationError, match="lower must not exceed upper"):
        JointLimitV1(lower=2.0, upper=1.0, unit="meters", provenance=provenance)
    with pytest.raises(ValidationError, match="must be finite"):
        JointLimitV1(lower=-math.inf, unit="meters", provenance=provenance)
    with pytest.raises(ValidationError, match="prismatic limits must use meters"):
        _joint(
            joint_type="prismatic",
            limit=JointLimitV1(lower=0.0, unit="degrees", provenance=provenance),
        )
    with pytest.raises(ValidationError, match="spherical limits are unsupported"):
        _joint(
            joint_type="spherical",
            limit=JointLimitV1(lower=-1.0, unit="degrees", provenance=provenance),
        )


def test_anchor_drive_state_and_mimic_reject_unsupported_numeric_values() -> None:
    provenance = _provenance("physics fact")
    anchor = JointAnchorV1(position_stage=(1.0, 2.0, 3.0), provenance=provenance)
    state = JointStateV1(position=0.0, velocity=0.0, provenance=provenance)
    drive = JointDriveV1(
        drive_type="force",
        stiffness=10.0,
        damping=1.0,
        max_force=100.0,
        target_position=0.0,
        target_velocity=0.0,
        provenance=provenance,
    )
    assert _joint(anchor=anchor, state=state, drive=drive).drive == drive

    with pytest.raises(ValidationError, match="position_stage.*must be finite"):
        JointAnchorV1(position_stage=(0.0, math.nan, 0.0), provenance=provenance)
    with pytest.raises(ValidationError, match="velocity must be finite"):
        JointStateV1(position=0.0, velocity=math.inf, provenance=provenance)
    with pytest.raises(ValidationError, match="stiffness must be nonnegative"):
        JointDriveV1(
            drive_type="force",
            stiffness=-1.0,
            damping=0.0,
            max_force=1.0,
            target_position=0.0,
            target_velocity=0.0,
            provenance=provenance,
        )

    mimic = JointMimicV1(
        reference_joint_id="joint_a",
        gearing=-1.0,
        offset=0.0,
        natural_frequency=10.0,
        damping_ratio=1.0,
        provenance=provenance,
    )
    with pytest.raises(ValidationError, match="both drive and mimic"):
        _joint(joint_id="joint_b", drive=drive, mimic=mimic)
    with pytest.raises(ValidationError, match="gearing must be nonzero"):
        JointMimicV1(
            reference_joint_id="joint_a",
            gearing=0.0,
            offset=0.0,
            natural_frequency=10.0,
            damping_ratio=1.0,
            provenance=provenance,
        )


@pytest.mark.parametrize("coefficient", [-1.0, -math.inf, math.inf, math.nan])
def test_joint_friction_requires_a_finite_nonnegative_coefficient(
    coefficient: float,
) -> None:
    with pytest.raises(ValidationError, match="coefficient must be"):
        _friction(coefficient)


def test_spherical_joint_rejects_every_unaddressable_scalar_control_fact() -> None:
    provenance = _provenance("spherical scalar control")
    controls = (
        ("joint_friction", _friction()),
        (
            "drive",
            JointDriveV1(
                drive_type="force",
                stiffness=10.0,
                damping=1.0,
                max_force=100.0,
                target_position=0.0,
                target_velocity=0.0,
                provenance=provenance,
            ),
        ),
        (
            "state",
            JointStateV1(
                position=0.0,
                velocity=0.0,
                provenance=provenance,
            ),
        ),
        ("mimic", _mimic("source_joint")),
    )

    for field, value in controls:
        with pytest.raises(
            ValidationError,
            match=rf"scalar spherical control facts.*{field}",
        ):
            _joint(joint_type="spherical", **{field: value})


def test_spherical_anchor_remains_valid_and_round_trips() -> None:
    anchor = JointAnchorV1(
        position_stage=(1.0, 2.0, 3.0),
        provenance=_provenance("spherical anchor"),
    )
    joint = _joint(joint_type="spherical", anchor=anchor)

    assert joint.anchor == anchor
    assert JointPlanV1.model_validate_json(canonical_json(joint)) == joint


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
def test_scalar_control_facts_remain_valid_for_single_axis_joints(
    joint_type: str,
) -> None:
    provenance = _provenance(f"{joint_type} scalar control")
    drive = JointDriveV1(
        drive_type="force",
        stiffness=10.0,
        damping=1.0,
        max_force=100.0,
        target_position=0.0,
        target_velocity=0.0,
        provenance=provenance,
    )
    state = JointStateV1(
        position=0.0,
        velocity=0.0,
        provenance=provenance,
    )
    friction = _friction(0.25)

    passive = _joint(joint_type=joint_type, joint_friction=friction)
    driven = _joint(joint_type=joint_type, joint_friction=friction, drive=drive)
    assert passive.joint_friction == friction
    assert passive.drive is None
    assert driven.joint_friction == friction
    assert driven.drive == drive
    assert "joint_friction" not in driven.drive.model_dump(mode="json")
    assert _joint(joint_type=joint_type, state=state).state == state
    assert _joint(joint_type=joint_type, mimic=_mimic("source_joint")).mimic


def test_joint_friction_does_not_weaken_mimic_conflict() -> None:
    with pytest.raises(ValidationError, match="friction with a mimic"):
        _joint(
            joint_id="mimic_joint",
            joint_friction=_friction(),
            mimic=_mimic("source_joint"),
        )


def test_mass_properties_require_physical_finite_values() -> None:
    provenance = _provenance("mass")
    mass = MassPropertiesV1(
        mass_kg=2.0,
        center_of_mass_m=(0.25, -0.5, 1.0),
        diagonal_inertia_kg_m2=(1.0, 1.5, 2.0),
        principal_axes=(1.0, 0.0, 0.0, 0.0),
        provenance=provenance,
    )
    assert mass.mass_kg == 2.0
    assert mass.center_of_mass_m == (0.25, -0.5, 1.0)
    assert MassPropertiesV1(
        mass_kg=2.0,
        diagonal_inertia_kg_m2=(1.0, 1.5, 2.0),
        provenance=provenance,
    ).model_dump(mode="json", exclude_none=True) == {
        "mass_kg": 2.0,
        "diagonal_inertia_kg_m2": [1.0, 1.5, 2.0],
        "provenance": provenance.model_dump(mode="json", exclude_none=True),
    }

    with pytest.raises(ValidationError, match="mass_kg must be positive"):
        MassPropertiesV1(
            mass_kg=0.0,
            diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
            provenance=provenance,
        )
    with pytest.raises(ValidationError, match="inertia triangle"):
        MassPropertiesV1(
            mass_kg=1.0,
            diagonal_inertia_kg_m2=(1.0, 1.0, 3.0),
            provenance=provenance,
        )
    with pytest.raises(ValidationError, match="normalized quaternion"):
        MassPropertiesV1(
            mass_kg=1.0,
            diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
            principal_axes=(2.0, 0.0, 0.0, 0.0),
            provenance=provenance,
        )
    with pytest.raises(ValidationError, match=r"center_of_mass_m\[1\] must be finite"):
        MassPropertiesV1(
            mass_kg=1.0,
            center_of_mass_m=(0.0, math.nan, 0.0),
            diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
            provenance=provenance,
        )


@pytest.mark.parametrize(
    "collider_path",
    [
        "/World/Body",
        "/World/Body/Collider",
        "/World/Body/Geometry/NestedCollider",
    ],
)
def test_rigid_body_accepts_colliders_in_its_usd_prim_subtree(
    collider_path: str,
) -> None:
    provenance = _provenance("collider ownership")

    body = RigidBodyPlanV1(
        prim_path="/World/Body",
        colliders=(
            ColliderPlanV1(
                prim_path=collider_path,
                provenance=provenance,
            ),
        ),
        provenance=provenance,
    )

    assert body.colliders[0].prim_path == collider_path


@pytest.mark.parametrize(
    "collider_path",
    [
        "/World/Body2",
        "/World/Body2/Collider",
        "/World/Other/Body",
    ],
)
def test_rigid_body_rejects_colliders_outside_its_usd_prim_subtree(
    collider_path: str,
) -> None:
    provenance = _provenance("collider ownership")

    with pytest.raises(ValidationError, match="descendant.*body's subtree"):
        RigidBodyPlanV1(
            prim_path="/World/Body",
            colliders=(
                ColliderPlanV1(
                    prim_path=collider_path,
                    provenance=provenance,
                ),
            ),
            provenance=provenance,
        )


def test_rigid_body_collider_subtree_uses_component_semantics_without_pxr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pxr", None)
    provenance = _provenance("collider ownership")

    for collider_path in (
        "/World/Body",
        "/World/Body/Collider",
    ):
        body = RigidBodyPlanV1(
            prim_path="/World/Body",
            colliders=(
                ColliderPlanV1(
                    prim_path=collider_path,
                    provenance=provenance,
                ),
            ),
            provenance=provenance,
        )
        assert body.colliders[0].prim_path == collider_path

    with pytest.raises(ValidationError, match="descendant.*body's subtree"):
        RigidBodyPlanV1(
            prim_path="/World/Body",
            colliders=(
                ColliderPlanV1(
                    prim_path="/World/Body2/Collider",
                    provenance=provenance,
                ),
            ),
            provenance=provenance,
        )


def test_collider_mesh_schema_presence_is_explicit_and_backward_compatible() -> None:
    provenance = _provenance("mesh collider")
    bare_schema = ColliderPlanV1(
        prim_path="/World/body/bareMeshCollider",
        mesh_collision_api=True,
        provenance=provenance,
    )
    legacy_approximation = ColliderPlanV1(
        prim_path="/World/body/legacyMeshCollider",
        mesh_approximation="convexHull",
        provenance=provenance,
    )
    redundant_explicit_approximation = ColliderPlanV1(
        prim_path="/World/body/legacyMeshCollider",
        mesh_collision_api=True,
        mesh_approximation="convexHull",
        provenance=provenance,
    )
    no_approximation = ColliderPlanV1(
        prim_path="/World/body/instanceRootCollider",
        mesh_approximation="none",
        provenance=provenance,
    )

    assert bare_schema.mesh_collision_api is True
    assert bare_schema.has_mesh_collision_api is True
    assert bare_schema.mesh_approximation is None
    assert legacy_approximation.mesh_collision_api is None
    assert legacy_approximation.has_mesh_collision_api is True
    assert redundant_explicit_approximation.mesh_collision_api is None
    assert redundant_explicit_approximation.has_mesh_collision_api is True
    assert no_approximation.mesh_approximation == "none"
    assert no_approximation.has_mesh_collision_api is True
    assert "mesh_collision_api" not in json.loads(canonical_json(legacy_approximation))
    backward_compatible_hash = (
        "cb10b40afb01c3e99b4d56340709c70103b01918324e0c3041a08a5f2b9b33b7"
    )
    assert canonical_sha256(legacy_approximation) == backward_compatible_hash
    assert canonical_json(redundant_explicit_approximation) == canonical_json(
        legacy_approximation
    )
    assert (
        canonical_sha256(redundant_explicit_approximation) == backward_compatible_hash
    )
    assert "mesh_collision_api" in json.loads(canonical_json(bare_schema))
    bare_schema_round_trip = ColliderPlanV1.model_validate_json(
        canonical_json(bare_schema)
    )
    assert bare_schema_round_trip == bare_schema
    assert canonical_sha256(bare_schema_round_trip) == canonical_sha256(bare_schema)

    with pytest.raises(
        ValidationError,
        match="mesh_collision_api must be exactly true or null",
    ):
        ColliderPlanV1(
            prim_path="/World/body/notPresent",
            mesh_collision_api=False,  # type: ignore[arg-type]
            provenance=provenance,
        )


@pytest.mark.parametrize("invalid_value", [False, 0, 1, 1.0, "true"])
def test_collider_mesh_schema_presence_rejects_non_boolean_wire_values(
    invalid_value: object,
) -> None:
    payload = {
        "prim_path": "/World/body/meshCollider",
        "mesh_collision_api": invalid_value,
        "provenance": _provenance("mesh collider").model_dump(mode="json"),
    }

    with pytest.raises(
        ValidationError,
        match="mesh_collision_api must be exactly true or null",
    ):
        ColliderPlanV1.model_validate(payload)
    with pytest.raises(
        ValidationError,
        match="mesh_collision_api must be exactly true or null",
    ):
        ColliderPlanV1.model_validate_json(json.dumps(payload))


def test_plan_canonicalizes_nested_collections_and_rejects_duplicate_targets() -> None:
    provenance = _provenance("schema")
    collider_a = ColliderPlanV1(
        prim_path="/World/a/visual",
        mesh_approximation="convexHull",
        provenance=provenance,
    )
    collider_b = ColliderPlanV1(
        prim_path="/World/a/collision",
        provenance=provenance,
    )
    body_a = RigidBodyPlanV1(
        prim_path="/World/a",
        colliders=(collider_a, collider_b),
        provenance=provenance,
    )
    body_b = RigidBodyPlanV1(prim_path="/World/b", provenance=provenance)
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(
            _joint("joint_b", body0="/World/a", body1="/World/b"),
            _joint("joint_a", body0="/World/base", body1="/World/a"),
        ),
        rigid_bodies=(body_b, body_a),
        articulation_root=ArticulationRootPlanV1(
            prim_path="/World/a",
            provenance=provenance,
        ),
    )

    assert [item.topology.joint_id for item in plan.joints] == ["joint_a", "joint_b"]
    assert [item.prim_path for item in plan.rigid_bodies] == ["/World/a", "/World/b"]
    assert [item.prim_path for item in plan.rigid_bodies[0].colliders] == [
        "/World/a/collision",
        "/World/a/visual",
    ]

    ancestor_root_plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(),
        rigid_bodies=(body_a, body_b),
        articulation_root=ArticulationRootPlanV1(
            prim_path="/World",
            provenance=provenance,
        ),
    )
    assert ancestor_root_plan.articulation_root is not None
    assert ancestor_root_plan.articulation_root.prim_path == "/World"

    with pytest.raises(ValidationError, match="joint_id values must be unique"):
        JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(_joint(), _joint()),
        )
    with pytest.raises(ValidationError, match="may belong to only one body"):
        JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(),
            rigid_bodies=(
                body_a,
                RigidBodyPlanV1(
                    prim_path="/World/a/visual",
                    colliders=(collider_a,),
                    provenance=provenance,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="nearest planned rigid body"):
        JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(),
            rigid_bodies=(
                RigidBodyPlanV1(
                    prim_path="/World/parent",
                    colliders=(
                        ColliderPlanV1(
                            prim_path="/World/parent/child/collision",
                            provenance=provenance,
                        ),
                    ),
                    provenance=provenance,
                ),
                RigidBodyPlanV1(
                    prim_path="/World/parent/child",
                    provenance=provenance,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="root must name a planned joint body"):
        JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(),
            rigid_bodies=(body_a,),
            articulation_root=ArticulationRootPlanV1(
                prim_path="/World/missing",
                provenance=provenance,
            ),
        )


@pytest.mark.parametrize("root_path", ["/World/base", "/World"])
def test_articulation_root_accepts_joint_endpoint_or_ancestor_without_body_plans(
    root_path: str,
) -> None:
    provenance = _provenance("articulation root")
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(
            _joint(
                "/Other/Joints/hinge",
                body0="/World/base",
                body1="/World/link",
            ),
        ),
        articulation_root=ArticulationRootPlanV1(
            prim_path=root_path,
            provenance=provenance,
        ),
    )

    assert plan.articulation_root is not None
    assert plan.articulation_root.prim_path == root_path
    assert JointRiggerPlanV1.model_validate_json(canonical_json(plan)) == plan


@pytest.mark.parametrize(
    ("joint_id", "body0", "body1", "root_path"),
    [
        ("joint", "/World/base", "/World/link", "/Other"),
        ("joint", "/World/base", "/World/link", "/World/base/child"),
        ("joint", "/World/body2", "/World/link", "/World/body"),
        ("/Other/Joints/hinge", "/World/base", "/World/link", "/Other"),
    ],
)
def test_articulation_root_rejects_paths_unrelated_to_joint_endpoints(
    joint_id: str,
    body0: str,
    body1: str,
    root_path: str,
) -> None:
    with pytest.raises(ValidationError, match="root must name a planned joint body"):
        JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(_joint(joint_id, body0=body0, body1=body1),),
            articulation_root=ArticulationRootPlanV1(
                prim_path=root_path,
                provenance=_provenance("unrelated articulation root"),
            ),
        )


def test_articulation_root_requires_at_least_one_planned_body_association() -> None:
    with pytest.raises(ValidationError, match="root must name a planned joint body"):
        JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(),
            articulation_root=ArticulationRootPlanV1(
                prim_path="/World",
                provenance=_provenance("orphan articulation root"),
            ),
        )


def test_articulation_root_association_uses_component_semantics_without_pxr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pxr", None)
    provenance = _provenance("articulation root")

    valid = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(_joint(body0="/World/Body/Root", body1="/World/link"),),
        articulation_root=ArticulationRootPlanV1(
            prim_path="/World/Body",
            provenance=provenance,
        ),
    )
    assert valid.articulation_root is not None

    with pytest.raises(ValidationError, match="root must name a planned joint body"):
        JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(_joint(body0="/World/Body2", body1="/World/link"),),
            articulation_root=ArticulationRootPlanV1(
                prim_path="/World/Body",
                provenance=provenance,
            ),
        )


def test_mimic_references_are_resolved_against_the_complete_plan() -> None:
    mimic = JointMimicV1(
        reference_joint_id="joint_a",
        gearing=1.0,
        offset=0.0,
        natural_frequency=5.0,
        damping_ratio=0.7,
        provenance=_provenance("mimic"),
    )
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(
            _joint("joint_b", body1="/World/link_b", mimic=mimic),
            _joint("joint_a"),
        ),
    )
    assert plan.joints[1].mimic == mimic

    with pytest.raises(ValidationError, match="cannot reference itself"):
        JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(_joint("joint_a", mimic=mimic),),
        )
    with pytest.raises(ValidationError, match="must name another plan joint"):
        JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(
                _joint(
                    "joint_b",
                    mimic=mimic.model_copy(
                        update={"reference_joint_id": "missing_joint"}
                    ),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="cannot name a spherical joint"):
        JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(
                _joint(
                    "joint_b",
                    body1="/World/link_b",
                    mimic=_mimic("joint_a"),
                ),
                _joint("joint_a", joint_type="spherical"),
            ),
        )


def test_mimic_reference_graph_rejects_cycles_and_preserves_acyclic_chains() -> None:
    with pytest.raises(
        ValidationError,
        match="joint_a -> joint_b -> joint_a",
    ):
        JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(
                _joint("joint_b", mimic=_mimic("joint_a")),
                _joint("joint_a", mimic=_mimic("joint_b")),
            ),
        )

    three_node_cycle = (
        _joint("joint_c", mimic=_mimic("joint_a")),
        _joint("joint_a", mimic=_mimic("joint_b")),
        _joint("joint_b", mimic=_mimic("joint_c")),
    )
    cycle_payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "joints": [item.model_dump(mode="json") for item in three_node_cycle],
    }
    with pytest.raises(
        ValidationError,
        match="joint_a -> joint_b -> joint_c -> joint_a",
    ):
        JointRiggerPlanV1.model_validate_json(json.dumps(cycle_payload))

    acyclic = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(
            _joint("joint_c", mimic=_mimic("joint_b")),
            _joint("joint_a"),
            _joint("joint_b", mimic=_mimic("joint_a")),
        ),
    )
    assert [item.topology.joint_id for item in acyclic.joints] == [
        "joint_a",
        "joint_b",
        "joint_c",
    ]
    assert JointRiggerPlanV1.model_validate_json(canonical_json(acyclic)) == acyclic


def test_legacy_compatibility_assignments_are_explicit_sorted_and_unique() -> None:
    compatibility = LegacyComponentNameCompatibilityV1(
        assignments=(
            LegacyComponentAssignmentV1(
                prim_path="/World/z",
                component_name="wheel",
                source_field="role",
            ),
            LegacyComponentAssignmentV1(
                prim_path="/World/a",
                component_name="base",
                source_field="component_name",
            ),
        )
    )
    assert [item.prim_path for item in compatibility.assignments] == [
        "/World/a",
        "/World/z",
    ]

    with pytest.raises(ValidationError, match="prim_path values must be unique"):
        LegacyComponentNameCompatibilityV1(
            assignments=(compatibility.assignments[0], compatibility.assignments[0])
        )

    with pytest.raises(ValidationError, match="at least 1 item"):
        LegacyComponentNameCompatibilityV1(assignments=())


@pytest.mark.parametrize(
    "disposition",
    ["ignored", "defaulted", "rejected", "unresolved"],
)
def test_nonaccepted_field_decisions_require_reason_codes(disposition: str) -> None:
    with pytest.raises(ValidationError, match="require reason_code"):
        FieldDecisionV1(field="axis_stage", disposition=disposition)

    decision = FieldDecisionV1(
        field="axis_stage",
        disposition=disposition,
        reason_code="axis_unavailable",
    )
    assert decision.reason_code == "axis_unavailable"


def test_accepted_field_decisions_require_provenance() -> None:
    with pytest.raises(ValidationError, match="require provenance"):
        FieldDecisionV1(field="body0", disposition="accepted")

    decision = FieldDecisionV1(
        field="body0",
        disposition="accepted",
        provenance=_provenance("body0"),
    )
    assert decision.reason_code is None


def test_diagnostics_are_canonical_and_reject_duplicate_decision_identities() -> None:
    accepted_z = FieldDecisionV1(
        field="z_field",
        disposition="accepted",
        provenance=_provenance("z"),
    )
    rejected_a = FieldDecisionV1(
        field="a_field",
        disposition="rejected",
        reason_code="unsupported",
    )
    path_decision = FieldDecisionV1(
        field="usd.joint_prim_path",
        disposition="defaulted",
        reason_code="deterministic_joint_path",
        detail="/World/Joints/joint_b",
    )
    diagnostic_b = JointDiagnosticV1(
        joint_id="joint_b",
        field_decisions=(accepted_z, rejected_a, path_decision),
        reason_codes=("z_reason", "a_reason", "z_reason"),
    )
    diagnostic_a = JointDiagnosticV1(joint_id="joint_a")
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="owned-core",
        field_decisions=(accepted_z, rejected_a),
        joint_diagnostics=(diagnostic_b, diagnostic_a),
        errors=("z error", "a error", "z error"),
        warnings=("z warning", "a warning"),
    )

    assert [item.field for item in diagnostics.field_decisions] == [
        "a_field",
        "z_field",
    ]
    assert [item.joint_id for item in diagnostics.joint_diagnostics] == [
        "joint_a",
        "joint_b",
    ]
    assert diagnostic_a.authored_prim_path is None
    assert diagnostic_b.reason_codes == ("a_reason", "z_reason")
    assert diagnostic_b.authored_prim_path == "/World/Joints/joint_b"
    diagnostic_payload = diagnostic_b.model_dump(mode="json", exclude_none=True)
    assert "authored_prim_path" not in diagnostic_payload
    assert (
        JointDiagnosticV1.model_validate_json(
            json.dumps(diagnostic_payload)
        ).authored_prim_path
        == "/World/Joints/joint_b"
    )
    assert diagnostics.errors == ("a error", "z error")
    assert diagnostics.warnings == ("a warning", "z warning")

    with pytest.raises(ValidationError, match="top-level field decisions"):
        JointRiggerDiagnosticsV1(
            schema_version=DIAGNOSTICS_SCHEMA_VERSION,
            backend_name="owned-core",
            field_decisions=(accepted_z, accepted_z),
        )
    with pytest.raises(ValidationError, match="unique joint_id"):
        JointRiggerDiagnosticsV1(
            schema_version=DIAGNOSTICS_SCHEMA_VERSION,
            backend_name="owned-core",
            joint_diagnostics=(diagnostic_a, diagnostic_a),
        )
    invalid_path = JointDiagnosticV1(
        joint_id="invalid_path",
        field_decisions=(
            path_decision.model_copy(update={"detail": "World/Joints/invalid"}),
        ),
    )
    assert invalid_path.authored_prim_path is None


def test_schema_versions_and_result_output_identity_are_fail_closed() -> None:
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(_joint(),),
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=_artifact(),
        plan=plan,
    )
    diagnostics = _diagnostics()
    output = _artifact(uri="s3://example/assets/rigged.usdz", root_sha256=SHA_C)
    result = JointRiggerResultV1(
        schema_version=RESULT_SCHEMA_VERSION,
        status="succeeded",
        input_sha256=canonical_sha256(request),
        plan_sha256=canonical_sha256(plan),
        output_artifact=output,
        diagnostics=diagnostics,
    )

    assert request.schema_version == INPUT_SCHEMA_VERSION
    assert plan.schema_version == PLAN_SCHEMA_VERSION
    assert diagnostics.schema_version == DIAGNOSTICS_SCHEMA_VERSION
    assert result.schema_version == RESULT_SCHEMA_VERSION
    assert result.input_sha256 == canonical_sha256(request)
    assert (
        JointRiggerResultV1.model_json_schema()["properties"]["input_sha256"][
            "description"
        ]
        == "SHA-256 of the canonical Joint Rigger input JSON payload."
    )

    with pytest.raises(ValidationError, match="require output_artifact"):
        JointRiggerResultV1(
            schema_version=RESULT_SCHEMA_VERSION,
            status="succeeded",
            input_sha256=SHA_A,
            plan_sha256=SHA_B,
            diagnostics=diagnostics,
        )
    with pytest.raises(ValidationError, match="must not claim output_artifact"):
        JointRiggerResultV1(
            schema_version=RESULT_SCHEMA_VERSION,
            status="failed",
            input_sha256=SHA_A,
            plan_sha256=SHA_B,
            output_artifact=output,
            diagnostics=diagnostics,
        )
    with pytest.raises(ValidationError, match="must not contain diagnostics errors"):
        JointRiggerResultV1(
            schema_version=RESULT_SCHEMA_VERSION,
            status="succeeded",
            input_sha256=SHA_A,
            plan_sha256=SHA_B,
            output_artifact=output,
            diagnostics=diagnostics.model_copy(update={"errors": ("fatal",)}),
        )
    with pytest.raises(ValidationError, match="lowercase 64-character SHA-256"):
        JointRiggerResultV1(
            schema_version=RESULT_SCHEMA_VERSION,
            status="failed",
            input_sha256="A" * 64,
            plan_sha256=SHA_B,
            diagnostics=diagnostics,
        )


def test_v2_rigid_link_request_is_canonical_versioned_and_v1_compatible() -> None:
    request = _v2_request()

    assert isinstance(request, JointRiggerInputV1)
    assert request.schema_version == INPUT_SCHEMA_VERSION_V2
    assert [link.link_id for link in request.rigid_links] == ["base", "drawer"]
    assert [member.source_prim_path for member in request.rigid_links[1].members] == [
        "/World/panel_a",
        "/World/panel_b",
    ]
    assert JointRiggerInputV2.model_validate_json(canonical_json(request)) == request
    assert "schema_version" in JointRiggerInputV2.model_json_schema()["required"]

    payload = request.model_dump(mode="json")
    payload.pop("schema_version")
    with pytest.raises(ValidationError) as missing_version:
        JointRiggerInputV2.model_validate(payload)
    assert missing_version.value.errors()[0]["loc"] == ("schema_version",)


def test_v1_wire_json_and_hashes_are_unchanged_by_v2_models() -> None:
    provenance = FieldProvenanceV1(
        source="owner_approved_plan",
        evidence="v1 compatibility evidence",
    )
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(
            JointPlanV1(
                topology=JointTopologyV1(
                    joint_id="hinge",
                    joint_type="revolute",
                    body0="/World/base",
                    body1="/World/link",
                    axis_stage=(0.0, 0.0, 1.0),
                    field_provenance=dict.fromkeys(
                        ("joint_type", "body0", "body1", "axis_stage"),
                        provenance,
                    ),
                )
            ),
        ),
        articulation_root=ArticulationRootPlanV1(
            prim_path="/World/base",
            provenance=provenance,
        ),
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=ArtifactIdentityV1(
            uri="s3://example/source.usda",
            root_sha256=SHA_A,
        ),
        plan=plan,
    )

    assert "articulation_roots" not in json.loads(canonical_json(plan))
    assert "rigid_links" not in json.loads(canonical_json(request))
    assert canonical_sha256(plan) == (
        "2dc28bf7ee9c8be818a7f2b19c9e689573715bb057857bc607f5c57dffc166aa"
    )
    assert canonical_sha256(request) == (
        "ceff66124811ec03d1a3ce73ff52843eee44242d33416640888f8f77735ce04e"
    )
    assert (
        "articulation_roots" not in JointRiggerPlanV1.model_json_schema()["properties"]
    )


def test_v2_plan_canonicalizes_exact_disconnected_component_roots() -> None:
    roots = (
        ArticulationRootPlanV1(
            prim_path="/World/component_b",
            provenance=_provenance("root_b"),
        ),
        ArticulationRootPlanV1(
            prim_path="/World/component_a",
            provenance=_provenance("root_a"),
        ),
    )
    plan = JointRiggerPlanV2(
        schema_version=PLAN_SCHEMA_VERSION_V2,
        joints=(
            _joint(
                "joint_b",
                body0="/World/component_b",
                body1="/World/component_b/link",
            ),
            _joint(
                "joint_a",
                body0="/World/component_a",
                body1="/World/component_a/link",
            ),
        ),
        articulation_roots=roots,
    )

    assert tuple(root.prim_path for root in plan.articulation_roots) == (
        "/World/component_a",
        "/World/component_b",
    )
    assert JointRiggerPlanV2.model_validate_json(canonical_json(plan)) == plan
    assert "articulation_root" not in json.loads(canonical_json(plan))


@pytest.mark.parametrize(
    ("roots", "message"),
    (
        (("/World/component_a",), "exactly match"),
        (
            ("/World/component_a", "/World/component_b", "/World/extra"),
            "exactly match",
        ),
        (
            ("/World/component_a", "/World/component_a"),
            "must be unique",
        ),
        (
            ("/World/component_a", "/World/component_a/link"),
            "must not overlap",
        ),
        (("/World",), "exactly match"),
    ),
)
def test_v2_plan_rejects_missing_extra_duplicate_nested_or_ancestor_roots(
    roots: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        JointRiggerPlanV2(
            schema_version=PLAN_SCHEMA_VERSION_V2,
            joints=(
                _joint(
                    "joint_a",
                    body0="/World/component_a",
                    body1="/World/component_a/link",
                ),
                _joint(
                    "joint_b",
                    body0="/World/component_b",
                    body1="/World/component_b/link",
                ),
            ),
            articulation_roots=tuple(
                ArticulationRootPlanV1(
                    prim_path=path,
                    provenance=_provenance("root", prim_path=path),
                )
                for path in roots
            ),
        )


def test_v2_plan_rejects_cycles_multiple_incoming_and_non_topology_empty_roots() -> (
    None
):
    with pytest.raises(ValidationError, match="directed cycle"):
        JointRiggerPlanV2(
            schema_version=PLAN_SCHEMA_VERSION_V2,
            joints=(
                _joint("forward", body0="/World/a", body1="/World/b"),
                _joint("reverse", body0="/World/b", body1="/World/a"),
            ),
            articulation_roots=(
                ArticulationRootPlanV1(
                    prim_path="/World/a",
                    provenance=_provenance("root"),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="multiple incoming"):
        JointRiggerPlanV2(
            schema_version=PLAN_SCHEMA_VERSION_V2,
            joints=(
                _joint("left", body0="/World/a", body1="/World/c"),
                _joint("right", body0="/World/b", body1="/World/c"),
            ),
            articulation_roots=(
                ArticulationRootPlanV1(
                    prim_path="/World/a",
                    provenance=_provenance("root_a"),
                ),
                ArticulationRootPlanV1(
                    prim_path="/World/b",
                    provenance=_provenance("root_b"),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="topology-only internal projection"):
        JointRiggerPlanV2(
            schema_version=PLAN_SCHEMA_VERSION_V2,
            joints=(
                _joint(
                    "controlled",
                    state=JointStateV1(
                        position=0.0,
                        velocity=0.0,
                        provenance=_provenance("state"),
                    ),
                ),
            ),
            articulation_roots=(),
        )


def test_v2_nested_existing_links_follow_transitive_joint_ancestry() -> None:
    joints = (
        _joint(
            "first",
            body0="/World/base",
            body1="/World/base/link",
        ),
        _joint(
            "second",
            body0="/World/base/link",
            body1="/World/base/link/tool",
        ),
    )
    plan = JointRiggerPlanV2(
        schema_version=PLAN_SCHEMA_VERSION_V2,
        joints=joints,
        articulation_roots=(
            ArticulationRootPlanV1(
                prim_path="/World/base",
                provenance=_provenance("root"),
            ),
        ),
    )
    links = tuple(
        RigidLinkPlanV1(
            link_id=path.rsplit("/", maxsplit=1)[-1],
            body_authoring="existing",
            body_prim_path=path,
            members=(
                RigidLinkMemberPlanV1(
                    source_prim_path=path,
                    authored_prim_path=path,
                ),
            ),
        )
        for path in (
            "/World/base",
            "/World/base/link",
            "/World/base/link/tool",
        )
    )

    request = JointRiggerInputV2(
        schema_version=INPUT_SCHEMA_VERSION_V2,
        source_asset=_artifact(),
        plan=plan,
        rigid_links=links,
    )
    assert len(request.rigid_links) == 3

    reversed_plan = JointRiggerPlanV2(
        schema_version=PLAN_SCHEMA_VERSION_V2,
        joints=(
            _joint(
                "reverse",
                body0="/World/base/link",
                body1="/World/base",
            ),
        ),
        articulation_roots=(
            ArticulationRootPlanV1(
                prim_path="/World/base/link",
                provenance=_provenance("reverse_root"),
            ),
        ),
    )
    with pytest.raises(ValidationError, match="transitive joint ancestry"):
        JointRiggerInputV2(
            schema_version=INPUT_SCHEMA_VERSION_V2,
            source_asset=_artifact(),
            plan=reversed_plan,
            rigid_links=links[:2],
        )


@pytest.mark.parametrize(
    ("payload_update", "message"),
    (
        (
            {
                "body_authoring": "existing",
                "body_prim_path": "/World/base",
                "members": [
                    {
                        "source_prim_path": "/World/base",
                        "authored_prim_path": "/World/not_base",
                    }
                ],
            },
            "identity member equal to body_prim_path",
        ),
        (
            {
                "body_authoring": "aggregate",
                "body_prim_path": "/drawer",
                "members": [
                    {
                        "source_prim_path": "/panel_a",
                        "authored_prim_path": "/drawer/panel_a",
                    },
                    {
                        "source_prim_path": "/panel_b",
                        "authored_prim_path": "/drawer/panel_b",
                    },
                ],
            },
            "below one authored parent",
        ),
        (
            {
                "body_authoring": "aggregate",
                "body_prim_path": "/World/drawer",
                "members": [
                    {
                        "source_prim_path": "/World/panel_a",
                        "authored_prim_path": "/World/drawer/panel_a",
                    }
                ],
            },
            "at least two members",
        ),
        (
            {
                "body_authoring": "aggregate",
                "body_prim_path": "/World/drawer",
                "members": [
                    {
                        "source_prim_path": "/World/panel_a",
                        "authored_prim_path": "/World/drawer/panel_a",
                    },
                    {
                        "source_prim_path": "/Other/panel_b",
                        "authored_prim_path": "/World/drawer/panel_b",
                    },
                ],
            },
            "share one parent",
        ),
        (
            {
                "body_authoring": "aggregate",
                "body_prim_path": "/World/drawer",
                "members": [
                    {
                        "source_prim_path": "/World/panel_a",
                        "authored_prim_path": "/World/drawer/renamed_a",
                    },
                    {
                        "source_prim_path": "/World/panel_b",
                        "authored_prim_path": "/World/drawer/panel_b",
                    },
                ],
            },
            "deterministic direct children",
        ),
    ),
)
def test_rigid_link_models_reject_non_identity_or_ambiguous_mappings(
    payload_update: dict[str, object],
    message: str,
) -> None:
    payload = {"link_id": "test", **payload_update}
    with pytest.raises(ValidationError, match=message):
        RigidLinkPlanV1.model_validate_json(json.dumps(payload))


def test_v2_request_requires_exact_nonoverlapping_plan_body_coverage() -> None:
    request = _v2_request()
    payload = request.model_dump(mode="json")
    payload["rigid_links"] = payload["rigid_links"][1:]
    with pytest.raises(ValidationError, match="exactly cover every planned body"):
        JointRiggerInputV2.model_validate_json(json.dumps(payload))

    payload = request.model_dump(mode="json")
    aggregate = payload["rigid_links"][1]
    aggregate["members"][0]["source_prim_path"] = "/World/base"
    aggregate["members"][0]["authored_prim_path"] = "/World/drawer/base"
    with pytest.raises(ValidationError, match="may belong to only one rigid link"):
        JointRiggerInputV2.model_validate_json(json.dumps(payload))


def test_wire_schema_versions_are_required_by_every_validation_entrypoint() -> None:
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(),
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=_artifact(),
        plan=plan,
    )
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="owned-core",
    )
    result = JointRiggerResultV1(
        schema_version=RESULT_SCHEMA_VERSION,
        status="failed",
        input_sha256=canonical_sha256(request),
        plan_sha256=canonical_sha256(plan),
        diagnostics=diagnostics,
    )
    contracts = (
        (JointRiggerPlanV1, plan, PLAN_SCHEMA_VERSION),
        (JointRiggerInputV1, request, INPUT_SCHEMA_VERSION),
        (
            JointRiggerDiagnosticsV1,
            diagnostics,
            DIAGNOSTICS_SCHEMA_VERSION,
        ),
        (JointRiggerResultV1, result, RESULT_SCHEMA_VERSION),
    )

    for model_type, instance, expected_version in contracts:
        payload = instance.model_dump(mode="python")
        json_payload = instance.model_dump(mode="json")
        assert payload["schema_version"] == expected_version
        assert "schema_version" in model_type.model_json_schema()["required"]
        assert model_type.model_validate(payload) == instance
        assert model_type.model_validate_json(canonical_json(instance)) == instance
        assert canonical_sha256(model_type.model_validate(payload)) == canonical_sha256(
            instance
        )

        missing_version = dict(payload)
        missing_version.pop("schema_version")
        missing_json_version = dict(json_payload)
        missing_json_version.pop("schema_version")
        validators = (
            lambda model_type=model_type, payload=missing_version: model_type(
                **payload
            ),
            lambda model_type=model_type, payload=missing_version: (
                model_type.model_validate(payload)
            ),
            lambda model_type=model_type, payload=missing_json_version: (
                model_type.model_validate_json(json.dumps(payload))
            ),
            lambda model_type=model_type, payload=missing_version: TypeAdapter(
                model_type
            ).validate_python(payload),
        )
        for validate in validators:
            with pytest.raises(ValidationError) as caught:
                validate()
            assert caught.value.errors()[0]["loc"] == ("schema_version",)
            assert caught.value.errors()[0]["type"] == "missing"

        wrong_version = {**payload, "schema_version": "unknown-version"}
        with pytest.raises(ValidationError) as caught:
            model_type.model_validate(wrong_version)
        assert caught.value.errors()[0]["loc"] == ("schema_version",)
        assert caught.value.errors()[0]["type"] == "literal_error"


def test_nested_wire_contracts_require_their_own_schema_versions() -> None:
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(),
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=_artifact(),
        plan=plan,
    )
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="owned-core",
    )
    result = JointRiggerResultV1(
        schema_version=RESULT_SCHEMA_VERSION,
        status="failed",
        input_sha256=canonical_sha256(request),
        plan_sha256=canonical_sha256(plan),
        diagnostics=diagnostics,
    )

    request_payload = request.model_dump(mode="json")
    request_payload["plan"].pop("schema_version")
    with pytest.raises(ValidationError) as caught:
        JointRiggerInputV1.model_validate(request_payload)
    assert caught.value.errors()[0]["loc"] == ("plan", "schema_version")
    assert caught.value.errors()[0]["type"] == "missing"

    result_payload = result.model_dump(mode="json")
    result_payload["diagnostics"].pop("schema_version")
    with pytest.raises(ValidationError) as caught:
        JointRiggerResultV1.model_validate_json(json.dumps(result_payload))
    assert caught.value.errors()[0]["loc"] == ("diagnostics", "schema_version")
    assert caught.value.errors()[0]["type"] == "missing"


def test_canonical_json_is_compact_ordered_and_omits_unsupported_optional_fields() -> (
    None
):
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(_joint(),),
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=_artifact(),
        plan=plan,
    )
    payload = canonical_json(request)
    parsed = json.loads(payload)

    assert ": " not in payload
    assert ", " not in payload
    assert "legacy_component_names" not in parsed
    assert "dependency_bundle_sha256" not in parsed["source_asset"]
    joint_payload = parsed["plan"]["joints"][0]
    assert set(joint_payload) == {"topology"}
    assert tuple(parsed) == tuple(sorted(parsed))

    mapping_a = {"z": 1, "a": {"y": 2, "x": 3}}
    mapping_b = {"a": {"x": 3, "y": 2}, "z": 1}
    assert canonical_json(mapping_a) == canonical_json(mapping_b)
    expected = hashlib.sha256(canonical_json(mapping_a).encode("utf-8")).hexdigest()
    assert canonical_sha256(mapping_a) == expected


def test_contract_round_trips_preserve_graph_schema_facts_and_hashes() -> None:
    provenance = _provenance("physics")
    plan = JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(
            _joint(
                "slider",
                joint_type="prismatic",
                body0="/World/base",
                body1="/World/drawer",
                limit=JointLimitV1(
                    lower=0.0,
                    upper=0.4,
                    unit="meters",
                    provenance=provenance,
                ),
                anchor=JointAnchorV1(
                    position_stage=(0.0, 0.0, 0.0),
                    provenance=provenance,
                ),
                joint_friction=JointFrictionV1(
                    coefficient=0.15,
                    provenance=provenance,
                ),
                state=JointStateV1(
                    position=0.0,
                    velocity=0.0,
                    provenance=provenance,
                ),
            ),
        ),
        rigid_bodies=(
            RigidBodyPlanV1(
                prim_path="/World/base",
                provenance=provenance,
            ),
            RigidBodyPlanV1(
                prim_path="/World/drawer",
                mass=MassPropertiesV1(
                    mass_kg=1.0,
                    diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
                    provenance=provenance,
                ),
                colliders=(
                    ColliderPlanV1(
                        prim_path="/World/drawer/visual",
                        mesh_collision_api=True,
                        mesh_approximation="convexHull",
                        provenance=provenance,
                    ),
                ),
                provenance=provenance,
            ),
        ),
        articulation_root=ArticulationRootPlanV1(
            prim_path="/World/base",
            provenance=provenance,
        ),
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=_artifact(),
        plan=plan,
    )
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="owned-core",
        joint_diagnostics=(
            JointDiagnosticV1(
                joint_id="slider",
                field_decisions=(
                    FieldDecisionV1(
                        field="axis_stage",
                        disposition="accepted",
                        provenance=_provenance("axis_stage"),
                    ),
                ),
            ),
        ),
    )
    result = JointRiggerResultV1(
        schema_version=RESULT_SCHEMA_VERSION,
        status="succeeded",
        input_sha256=canonical_sha256(request),
        plan_sha256=canonical_sha256(plan),
        output_artifact=_artifact(
            uri="s3://example/assets/rigged.usdz",
            root_sha256=SHA_C,
        ),
        diagnostics=diagnostics,
    )

    request_round_trip = JointRiggerInputV1.model_validate_json(canonical_json(request))
    diagnostics_round_trip = JointRiggerDiagnosticsV1.model_validate_json(
        canonical_json(diagnostics)
    )
    result_round_trip = JointRiggerResultV1.model_validate_json(canonical_json(result))

    assert request_round_trip == request
    assert diagnostics_round_trip == diagnostics
    assert result_round_trip == result
    assert canonical_sha256(request_round_trip) == canonical_sha256(request)
    round_trip_friction = request_round_trip.plan.joints[0].joint_friction
    assert round_trip_friction is not None
    assert round_trip_friction.coefficient == 0.15
    assert round_trip_friction.provenance == provenance
    round_trip_collider = request_round_trip.plan.rigid_bodies[1].colliders[0]
    assert round_trip_collider.mesh_collision_api is None
    assert round_trip_collider.has_mesh_collision_api is True
    topology = request_round_trip.plan.joints[0].topology
    assert (
        topology.joint_type,
        topology.body0,
        topology.body1,
        topology.axis_stage,
    ) == (
        "prismatic",
        "/World/base",
        "/World/drawer",
        (0.0, 0.0, 1.0),
    )


def test_contract_error_exposes_machine_readable_code_and_detail() -> None:
    error = JointRiggerContractError("axis_unresolved", "joint_a has no axis")

    assert error.code == "axis_unresolved"
    assert error.detail == "joint_a has no axis"
    assert str(error) == "axis_unresolved: joint_a has no axis"
