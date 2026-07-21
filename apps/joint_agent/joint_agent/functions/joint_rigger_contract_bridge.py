# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate first-class articulation records into the owned Joint Rigger core."""

from __future__ import annotations

import json
from typing import Any

from world_understanding.functions.physics.joint_rigger import (
    INPUT_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION_V2,
    PLAN_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION_V2,
    ArticulationRootPlanV1,
    ArtifactIdentityV1,
    FieldProvenanceV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerInputV1,
    JointRiggerInputV2,
    JointRiggerPlanV1,
    JointRiggerPlanV2,
    JointTopologyV1,
    RigidLinkMemberPlanV1,
    RigidLinkPlanV1,
    canonical_sha256,
)

from joint_agent.functions.articulation_contract import (
    ArticulationContractV1,
    JointRecordV1,
    LinkRecordV1,
    PrimRecordV1,
)

_DERIVATION = "articulation_contract_v1_to_joint_rigger_input_v1"
_CONTENT_ARTIFACT_URI_PREFIX = "memory://joint-agent/content-sha256"


def canonical_content_artifact_identity(
    artifact: ArtifactIdentityV1,
) -> ArtifactIdentityV1:
    """Replace one path-bearing URI with its stable closure-content URI."""

    if not isinstance(artifact, ArtifactIdentityV1):
        raise TypeError("artifact must be an ArtifactIdentityV1")
    dependency = artifact.dependency_bundle_sha256 or "no-dependencies"
    return artifact.model_copy(
        update={
            "uri": (
                f"{_CONTENT_ARTIFACT_URI_PREFIX}/{artifact.root_sha256}/{dependency}"
            )
        }
    )


def canonicalize_articulation_contract_artifacts(
    contract: ArticulationContractV1,
) -> ArticulationContractV1:
    """Canonicalize every declared and referenced contract artifact identity.

    Stage/session paths are transport details. Replacing them with identities
    derived from the already-validated root and dependency-closure hashes makes
    the exact Joint Rigger request portable across service and verifier roots.
    The rebuilt typed contract re-runs the source/provenance consistency checks.
    """

    if not isinstance(contract, ArticulationContractV1):
        raise TypeError("contract must be an ArticulationContractV1")

    def canonicalize(value: Any) -> Any:
        if isinstance(value, list):
            return [canonicalize(item) for item in value]
        if not isinstance(value, dict):
            return value
        if (
            "uri" in value
            and "root_sha256" in value
            and set(value).issubset({"uri", "root_sha256", "dependency_bundle_sha256"})
        ):
            identity = ArtifactIdentityV1.model_validate(value)
            return canonical_content_artifact_identity(identity).model_dump(mode="json")
        return {key: canonicalize(item) for key, item in value.items()}

    payload = canonicalize(contract.model_dump(mode="json"))
    return ArticulationContractV1.model_validate_json(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )


def build_canonical_joint_rigger_input_from_contract(
    contract: ArticulationContractV1,
    *,
    source_asset: ArtifactIdentityV1,
) -> JointRiggerInputV1 | JointRiggerInputV2:
    """Build the exact request using directory-independent artifact identities."""

    canonical_contract = canonicalize_articulation_contract_artifacts(contract)
    canonical_source = canonical_content_artifact_identity(source_asset)
    contract_sha256 = canonical_sha256(canonical_contract)
    return build_joint_rigger_input_from_contract(
        canonical_contract,
        contract_artifact=ArtifactIdentityV1(
            uri=f"memory://joint-agent/articulation-contract/{contract_sha256}",
            root_sha256=contract_sha256,
        ),
        source_asset=canonical_source,
    )


def build_joint_rigger_input_from_contract(
    contract: ArticulationContractV1,
    *,
    contract_artifact: ArtifactIdentityV1,
    source_asset: ArtifactIdentityV1,
) -> JointRiggerInputV1 | JointRiggerInputV2:
    """Build one topology-only #579 request from a ready first-class contract.

    ``contract_artifact`` identifies the canonical JSON form of ``contract``.
    The bridge rejects mismatched identities instead of accepting an unbound
    model assembled from different bytes.

    One-root existing-body contracts retain the exact V1 request. Contracts
    with multiple roots or aggregate links use V2 with complete contract-derived
    articulation roots and exact source-to-authored rigid-link mappings. Isolated
    links fail instead of losing membership semantics. Link roles remain
    validation facts in the bound contract and are never converted into a legacy
    ``component_name`` assignment.
    """

    if not isinstance(contract, ArticulationContractV1):
        raise TypeError("contract must be an ArticulationContractV1")
    if not isinstance(contract_artifact, ArtifactIdentityV1):
        raise TypeError("contract_artifact must be an ArtifactIdentityV1")
    if not isinstance(source_asset, ArtifactIdentityV1):
        raise TypeError("source_asset must be an ArtifactIdentityV1")

    if contract.status != "ready_for_rigger_input":
        raise JointRiggerContractError(
            "articulation_contract_not_ready",
            "review_required contracts cannot produce owned rigger input",
        )
    expected_contract_sha256 = canonical_sha256(contract)
    if contract_artifact.root_sha256 != expected_contract_sha256:
        raise JointRiggerContractError(
            "articulation_contract_identity_mismatch",
            "contract_artifact root_sha256 does not match canonical contract JSON",
        )
    if source_asset not in contract.source_identities:
        raise JointRiggerContractError(
            "source_asset_not_declared",
            "source_asset must be one of the contract source_identities",
        )

    links = {
        record.link_id: record
        for record in contract.records
        if isinstance(record, LinkRecordV1)
    }
    prim_paths_by_link = {
        link_id: tuple(
            sorted(
                record.prim_path
                for record in contract.records
                if isinstance(record, PrimRecordV1) and record.link_id == link_id
            )
        )
        for link_id in links
    }
    joints = tuple(
        record for record in contract.records if isinstance(record, JointRecordV1)
    )
    if not joints:
        raise JointRiggerContractError(
            "articulation_contract_has_no_joints",
            "a ready owned-rigger request requires at least one joint",
        )
    endpoint_link_ids = {
        link_id for joint in joints for link_id in (joint.body0_link, joint.body1_link)
    }
    isolated_link_ids = sorted(set(links) - endpoint_link_ids)
    if isolated_link_ids:
        raise JointRiggerContractError(
            "articulation_contract_has_isolated_links",
            "every owned-rigger link must be an endpoint of at least one joint: "
            + ", ".join(isolated_link_ids),
        )

    joint_plans = tuple(
        _joint_plan(
            joint,
            links=links,
            contract_artifact=contract_artifact,
        )
        for joint in joints
    )
    requires_v2 = len(contract.articulation_roots) > 1 or any(
        link.body_authoring == "aggregate" for link in links.values()
    )
    if not requires_v2:
        return JointRiggerInputV1(
            schema_version=INPUT_SCHEMA_VERSION,
            source_asset=source_asset,
            plan=JointRiggerPlanV1(
                schema_version=PLAN_SCHEMA_VERSION,
                joints=joint_plans,
            ),
            legacy_component_names=None,
        )

    rigid_links = tuple(
        _rigid_link_plan(
            link,
            member_paths=prim_paths_by_link[link.link_id],
        )
        for link in links.values()
    )
    articulation_roots = tuple(
        _articulation_root_plan(
            links[link_id],
            contract_artifact=contract_artifact,
        )
        for link_id in contract.articulation_roots
    )
    return JointRiggerInputV2(
        schema_version=INPUT_SCHEMA_VERSION_V2,
        source_asset=source_asset,
        plan=JointRiggerPlanV2(
            schema_version=PLAN_SCHEMA_VERSION_V2,
            joints=joint_plans,
            articulation_roots=articulation_roots,
        ),
        rigid_links=rigid_links,
        legacy_component_names=None,
    )


def _rigid_link_plan(
    link: LinkRecordV1,
    *,
    member_paths: tuple[str, ...],
) -> RigidLinkPlanV1:
    members: tuple[RigidLinkMemberPlanV1, ...]
    if link.body_authoring == "existing":
        if member_paths != (link.body_prim_path,):
            raise JointRiggerContractError(
                "articulation_contract_existing_membership_mismatch",
                f"existing link {link.link_id!r} must have one identity member",
            )
        members = (
            RigidLinkMemberPlanV1(
                source_prim_path=link.body_prim_path,
                authored_prim_path=link.body_prim_path,
            ),
        )
    else:
        if len(member_paths) < 2:
            raise JointRiggerContractError(
                "articulation_contract_aggregate_membership_incomplete",
                f"aggregate link {link.link_id!r} requires at least two members",
            )
        members = tuple(
            RigidLinkMemberPlanV1(
                source_prim_path=member_path,
                authored_prim_path=(
                    f"{link.body_prim_path}/{member_path.rsplit('/', 1)[-1]}"
                ),
            )
            for member_path in member_paths
        )
    return RigidLinkPlanV1(
        link_id=link.link_id,
        body_authoring=link.body_authoring,
        body_prim_path=link.body_prim_path,
        members=members,
    )


def _articulation_root_plan(
    link: LinkRecordV1,
    *,
    contract_artifact: ArtifactIdentityV1,
) -> ArticulationRootPlanV1:
    return ArticulationRootPlanV1(
        prim_path=link.body_prim_path,
        provenance=_contract_provenance(
            contract_artifact,
            prim_path=link.body_prim_path,
            properties=(
                f"link:{link.link_id}.body_prim_path",
                "joint_graph.component_root",
            ),
            evidence=(
                f"First-class link {link.link_id!r} is a declared articulation "
                "component root."
            ),
        ),
    )


def _joint_plan(
    joint: JointRecordV1,
    *,
    links: dict[str, LinkRecordV1],
    contract_artifact: ArtifactIdentityV1,
) -> JointPlanV1:
    body0 = links[joint.body0_link]
    body1 = links[joint.body1_link]
    provenance = {
        "joint_type": _contract_provenance(
            contract_artifact,
            prim_path=body1.body_prim_path,
            properties=(f"joint:{joint.joint_id}.motion_type",),
            evidence=(
                f"First-class joint {joint.joint_id!r} resolves motion_type "
                f"to {joint.motion_type!r}."
            ),
        ),
        "body0": _contract_provenance(
            contract_artifact,
            prim_path=body0.body_prim_path,
            properties=(
                f"joint:{joint.joint_id}.body0_link",
                f"link:{body0.link_id}.body_prim_path",
            ),
            evidence=(
                f"First-class joint {joint.joint_id!r} resolves body0 link "
                f"{body0.link_id!r} to {body0.body_prim_path}."
            ),
        ),
        "body1": _contract_provenance(
            contract_artifact,
            prim_path=body1.body_prim_path,
            properties=(
                f"joint:{joint.joint_id}.body1_link",
                f"link:{body1.link_id}.body_prim_path",
            ),
            evidence=(
                f"First-class joint {joint.joint_id!r} resolves body1 link "
                f"{body1.link_id!r} to {body1.body_prim_path}."
            ),
        ),
    }
    if joint.axis_stage is not None:
        provenance["axis_stage"] = _contract_provenance(
            contract_artifact,
            prim_path=body1.body_prim_path,
            properties=(f"joint:{joint.joint_id}.axis_stage",),
            evidence=(
                f"First-class joint {joint.joint_id!r} supplies one resolved "
                "stage-frame axis."
            ),
        )

    return JointPlanV1(
        topology=JointTopologyV1(
            joint_id=joint.joint_id,
            joint_type=joint.motion_type,
            body0=body0.body_prim_path,
            body1=body1.body_prim_path,
            axis_stage=joint.axis_stage,
            field_provenance=provenance,
        ),
        limit=joint.limit,
    )


def _contract_provenance(
    artifact: ArtifactIdentityV1,
    *,
    prim_path: str,
    properties: tuple[str, ...],
    evidence: str,
) -> FieldProvenanceV1:
    return FieldProvenanceV1(
        source="accepted_manifest",
        artifact=artifact,
        prim_path=prim_path,
        properties=properties,
        derivation=_DERIVATION,
        evidence=evidence,
    )


__all__ = [
    "build_canonical_joint_rigger_input_from_contract",
    "build_joint_rigger_input_from_contract",
    "canonical_content_artifact_identity",
    "canonicalize_articulation_contract_artifacts",
]
