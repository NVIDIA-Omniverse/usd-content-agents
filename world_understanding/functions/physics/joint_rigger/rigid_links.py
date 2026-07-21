# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed aggregate rigid-link namespace authoring for V2 requests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Never

from world_understanding.functions.physics.joint_rigger.models import (
    JointRiggerContractError,
    JointRiggerInputV1,
    JointRiggerInputV2,
    RigidLinkMemberPlanV1,
    RigidLinkPlanV1,
    canonical_sha256,
)

RIGID_LINK_AUTHORING_VERSION = "world-understanding-rigid-link-author-v1"

_RIGID_LINK_SCAN_MAX_PRIM_VISITS = 1_000_000
_TRANSFORM_TOLERANCE = 1e-9


@dataclass(frozen=True)
class _PrimSnapshot:
    relative_path: str
    type_name: str
    active: bool
    defined: bool
    instanceable: bool
    world_transform: tuple[float, ...] | None


@dataclass(frozen=True)
class _MemberSnapshot:
    member: RigidLinkMemberPlanV1
    prims: tuple[_PrimSnapshot, ...]


@dataclass(frozen=True)
class _AggregatePreflight:
    link: RigidLinkPlanV1
    members: tuple[_MemberSnapshot, ...]


def request_has_aggregate_links(request: JointRiggerInputV1) -> bool:
    """Return whether a V2 request requires namespace authoring."""

    return isinstance(request, JointRiggerInputV2) and any(
        link.body_authoring == "aggregate" for link in request.rigid_links
    )


def author_aggregate_rigid_links(stage: Any, request: JointRiggerInputV1) -> None:
    """Create and populate every V2 aggregate in one private editable stage.

    V1 is an intentional no-op. V2 existing links are validated as exact identity
    mappings but never receive namespace edits or metadata.
    """

    if not isinstance(request, JointRiggerInputV2):
        return
    aggregates = _preflight_source_links(stage, request)
    if not aggregates:
        return

    Sdf, UsdGeom = _pxr_modules()
    root_layer = stage.GetRootLayer()
    rollback_layer = Sdf.Layer.CreateAnonymous("joint-rigger-rigid-link-rollback.usda")
    if rollback_layer is None:  # pragma: no cover - OpenUSD allocation guard
        _fail(
            "aggregate_rollback_snapshot_failed",
            "could not allocate the aggregate rollback layer",
        )
    rollback_layer.TransferContent(root_layer)
    try:
        for aggregate in aggregates:
            body_path = aggregate.link.body_prim_path
            body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
            if not body or not body.IsValid():
                _fail(
                    "aggregate_body_creation_failed",
                    f"could not define aggregate Xform at {body_path}",
                )
            _author_aggregate_metadata(
                body,
                aggregate.link,
                member_snapshot_sha256=_member_snapshot_sha256(aggregate.members),
            )

        edits = Sdf.BatchNamespaceEdit()
        for aggregate in aggregates:
            for member in aggregate.link.members:
                edits.Add(member.source_prim_path, member.authored_prim_path)
        if not root_layer.CanApply(edits):
            _fail(
                "aggregate_namespace_edit_rejected",
                "the root layer rejected the complete aggregate namespace batch",
            )
        if not root_layer.Apply(edits):
            _fail(
                "aggregate_namespace_edit_failed",
                "the root layer could not apply the aggregate namespace batch",
            )
        _validate_authored_aggregates(stage, aggregates)
    except BaseException as primary_error:
        rollback_error = _rollback_aggregate_authoring(
            stage,
            rollback_layer,
        )
        if rollback_error is not None:
            primary_error.add_note(
                "Aggregate private-stage rollback also failed: "
                f"{type(rollback_error).__name__}: {rollback_error}"
            )
        raise


def validate_authored_rigid_links(stage: Any, request: JointRiggerInputV1) -> None:
    """Validate exact V2 identity and aggregate mappings on a saved output."""

    if not isinstance(request, JointRiggerInputV2):
        return
    for link in request.rigid_links:
        if link.body_authoring == "existing":
            _require_existing_identity_link(stage, link)
            continue
        _validate_authored_aggregate(stage, link, expected_members=None)


def _preflight_source_links(
    stage: Any,
    request: JointRiggerInputV2,
) -> tuple[_AggregatePreflight, ...]:
    if stage is None:
        _fail("invalid_stage", "stage must not be None")
    root_layer = stage.GetRootLayer()
    edit_layer = stage.GetEditTarget().GetLayer()
    if edit_layer.identifier != root_layer.identifier:
        _fail(
            "aggregate_edit_target_mismatch",
            "aggregate authoring requires the stage root layer as edit target",
        )
    if not root_layer.permissionToEdit:
        _fail("aggregate_root_not_editable", "stage root layer is not editable")

    aggregates: list[_AggregatePreflight] = []
    for link in request.rigid_links:
        if link.body_authoring == "existing":
            _require_existing_identity_link(stage, link)
            continue
        parent_path = _parent_path(link.body_prim_path)
        if parent_path == "/":  # model permits top-level existing links only
            _fail(
                "aggregate_top_level_unsupported",
                "aggregate source members must live below one authored parent",
            )
        parent = stage.GetPrimAtPath(parent_path)
        _require_root_owned_prim(parent, root_layer, label="aggregate parent")
        if _valid_prim(stage.GetPrimAtPath(link.body_prim_path)) or (
            root_layer.GetPrimAtPath(link.body_prim_path) is not None
        ):
            _fail(
                "aggregate_body_collision",
                f"aggregate body target already exists: {link.body_prim_path}",
            )

        member_snapshots: list[_MemberSnapshot] = []
        for member in link.members:
            source = stage.GetPrimAtPath(member.source_prim_path)
            _require_root_owned_prim(source, root_layer, label="aggregate member")
            if str(source.GetParent().GetPath()) != parent_path:
                _fail(
                    "aggregate_source_parent_mismatch",
                    f"aggregate member is no longer a direct sibling: "
                    f"{member.source_prim_path}",
                )
            if _valid_prim(stage.GetPrimAtPath(member.authored_prim_path)) or (
                root_layer.GetPrimAtPath(member.authored_prim_path) is not None
            ):
                _fail(
                    "aggregate_authored_path_collision",
                    f"aggregate member target already exists: "
                    f"{member.authored_prim_path}",
                )
            _reject_unsupported_subtree(stage, source, root_layer)
            member_snapshots.append(
                _MemberSnapshot(
                    member=member,
                    prims=_capture_subtree(stage, source),
                )
            )
        aggregates.append(
            _AggregatePreflight(link=link, members=tuple(member_snapshots))
        )
    return tuple(aggregates)


def _require_existing_identity_link(stage: Any, link: RigidLinkPlanV1) -> None:
    prim = stage.GetPrimAtPath(link.body_prim_path)
    if not _valid_prim(prim) or not prim.IsActive() or not prim.IsDefined():
        _fail(
            "existing_link_missing",
            f"existing rigid-link body is missing or inactive: {link.body_prim_path}",
        )


def _require_root_owned_prim(prim: Any, root_layer: Any, *, label: str) -> None:
    if not _valid_prim(prim) or not prim.IsActive() or not prim.IsDefined():
        path = "<invalid>" if not _valid_prim(prim) else str(prim.GetPath())
        _fail(
            "aggregate_source_missing",
            f"{label} must be active and defined in the source: {path}",
        )
    path = str(prim.GetPath())
    if (
        prim.IsInstance()
        or prim.IsInstanceProxy()
        or prim.IsPrototype()
        or prim.IsInPrototype()
        or prim.IsInstanceable()
    ):
        _fail(
            "aggregate_instance_unsupported",
            f"{label} cannot be an instance, prototype, or instanceable prim: {path}",
        )
    stack = tuple(prim.GetPrimStack())
    if len(stack) != 1 or stack[0].layer.identifier != root_layer.identifier:
        _fail(
            "aggregate_dependency_authorship_unsupported",
            f"{label} must have exactly one root-layer PrimSpec: {path}",
        )
    if stack[0].path.ContainsPrimVariantSelection():
        _fail(
            "aggregate_variant_unsupported",
            f"{label} cannot be authored inside a variant: {path}",
        )


def _reject_unsupported_subtree(stage: Any, root: Any, root_layer: Any) -> None:
    root_path = root.GetPath()
    visits = 0
    for prim in stage.TraverseAll():
        if not prim.GetPath().HasPrefix(root_path):
            continue
        visits += 1
        if visits > _RIGID_LINK_SCAN_MAX_PRIM_VISITS:
            _fail(
                "aggregate_subtree_scan_limit_exceeded",
                "aggregate member subtree exceeded the fixed safety scan limit",
            )
        _require_root_owned_prim(prim, root_layer, label="aggregate member subtree")
        path = str(prim.GetPath())
        if (
            prim.HasAuthoredReferences()
            or prim.HasAuthoredPayloads()
            or prim.HasAuthoredInherits()
            or prim.HasAuthoredSpecializes()
        ):
            _fail(
                "aggregate_composition_arc_unsupported",
                f"aggregate member subtree has an authored composition arc: {path}",
            )
        if prim.GetVariantSets().GetNames():
            _fail(
                "aggregate_variant_unsupported",
                f"aggregate member subtree has variant authoring: {path}",
            )


def _capture_subtree(stage: Any, root: Any) -> tuple[_PrimSnapshot, ...]:
    _, UsdGeom = _pxr_modules()
    root_path = root.GetPath()
    xform_cache = UsdGeom.XformCache()
    rows: list[_PrimSnapshot] = []
    visits = 0
    for prim in stage.TraverseAll():
        if not prim.GetPath().HasPrefix(root_path):
            continue
        visits += 1
        if visits > _RIGID_LINK_SCAN_MAX_PRIM_VISITS:
            _fail(
                "aggregate_subtree_scan_limit_exceeded",
                "aggregate member subtree exceeded the fixed safety scan limit",
            )
        relative_path = str(prim.GetPath().MakeRelativePath(root_path))
        xformable = UsdGeom.Xformable(prim)
        world_transform = None
        if xformable:
            matrix = xform_cache.GetLocalToWorldTransform(prim)
            world_transform = tuple(
                float(matrix[row][column]) for row in range(4) for column in range(4)
            )
        rows.append(
            _PrimSnapshot(
                relative_path=relative_path,
                type_name=str(prim.GetTypeName()),
                active=bool(prim.IsActive()),
                defined=bool(prim.IsDefined()),
                instanceable=bool(prim.IsInstanceable()),
                world_transform=world_transform,
            )
        )
    if not rows:
        _fail(
            "aggregate_source_missing",
            f"aggregate member subtree is empty: {root_path}",
        )
    return tuple(sorted(rows, key=lambda row: row.relative_path))


def _validate_authored_aggregates(
    stage: Any,
    aggregates: tuple[_AggregatePreflight, ...],
) -> None:
    for aggregate in aggregates:
        _validate_authored_aggregate(
            stage,
            aggregate.link,
            expected_members=aggregate.members,
        )


def _validate_authored_aggregate(
    stage: Any,
    link: RigidLinkPlanV1,
    *,
    expected_members: tuple[_MemberSnapshot, ...] | None,
) -> None:
    _, UsdGeom = _pxr_modules()
    body = stage.GetPrimAtPath(link.body_prim_path)
    if (
        not _valid_prim(body)
        or not body.IsActive()
        or not body.IsDefined()
        or not body.IsA(UsdGeom.Xform)
        or body.IsInstance()
        or body.IsInstanceProxy()
        or body.IsInstanceable()
    ):
        _fail(
            "authored_aggregate_mismatch",
            f"aggregate body is not one active, defined, non-instance Xform: "
            f"{link.body_prim_path}",
        )
    xformable = UsdGeom.Xformable(body)
    if xformable.GetOrderedXformOps() or xformable.GetResetXformStack():
        _fail(
            "authored_aggregate_mismatch",
            f"aggregate body must retain an identity local transform: "
            f"{link.body_prim_path}",
        )
    sealed_snapshot_sha256 = _validate_aggregate_metadata(body, link)
    if expected_members is not None and sealed_snapshot_sha256 != (
        _member_snapshot_sha256(expected_members)
    ):
        _fail(
            "authored_aggregate_mismatch",
            f"aggregate member snapshot seal does not match preflight at "
            f"{link.body_prim_path}",
        )

    expected_children = {member.authored_prim_path for member in link.members}
    observed_children = {str(child.GetPath()) for child in body.GetAllChildren()}
    if observed_children != expected_children:
        _fail(
            "authored_aggregate_mismatch",
            "aggregate body children do not exactly match the member plan: "
            f"observed={sorted(observed_children)}, "
            f"expected={sorted(expected_children)}",
        )

    snapshots_by_path = {
        snapshot.member.authored_prim_path: snapshot
        for snapshot in expected_members or ()
    }
    observed_members: list[_MemberSnapshot] = []
    for member in link.members:
        if _valid_prim(stage.GetPrimAtPath(member.source_prim_path)):
            _fail(
                "authored_aggregate_mismatch",
                f"source member still exists after aggregate authoring: "
                f"{member.source_prim_path}",
            )
        authored = stage.GetPrimAtPath(member.authored_prim_path)
        if not _valid_prim(authored) or str(authored.GetParent().GetPath()) != (
            link.body_prim_path
        ):
            _fail(
                "authored_aggregate_mismatch",
                f"authored member is missing or has the wrong parent: "
                f"{member.authored_prim_path}",
            )
        expected = snapshots_by_path.get(member.authored_prim_path)
        observed = _capture_subtree(stage, authored)
        if expected is not None:
            _require_preserved_member_snapshot(expected.prims, observed, member)
        observed_members.append(_MemberSnapshot(member=member, prims=observed))
    if sealed_snapshot_sha256 != _member_snapshot_sha256(tuple(observed_members)):
        _fail(
            "aggregate_member_changed",
            f"aggregate member structure or transform differs from the sealed "
            f"source snapshot at {link.body_prim_path}",
        )


def _author_aggregate_metadata(
    body: Any,
    link: RigidLinkPlanV1,
    *,
    member_snapshot_sha256: str,
) -> None:
    values = {
        "rigidLinkAuthoring": "aggregate",
        "rigidLinkAuthoringVersion": RIGID_LINK_AUTHORING_VERSION,
        "rigidLinkId": link.link_id,
        "rigidLinkMemberSnapshotSha256": member_snapshot_sha256,
        "rigidLinkPlanSha256": canonical_sha256(link),
    }
    for key, value in values.items():
        try:
            body.SetCustomDataByKey(f"jointRigger:{key}", value)
        except Exception as exc:
            _fail(
                "aggregate_metadata_authoring_failed",
                f"could not author {key} metadata at {link.body_prim_path}: {exc}",
            )
    observed_snapshot_sha256 = _validate_aggregate_metadata(body, link)
    if observed_snapshot_sha256 != member_snapshot_sha256:
        _fail(
            "aggregate_metadata_authoring_failed",
            f"member snapshot seal did not round-trip at {link.body_prim_path}",
        )


def _validate_aggregate_metadata(body: Any, link: RigidLinkPlanV1) -> str:
    observed = body.GetCustomData().get("jointRigger")
    if not isinstance(observed, dict):
        _fail(
            "authored_aggregate_mismatch",
            f"aggregate metadata is missing at {link.body_prim_path}",
        )
    snapshot_sha256 = observed.get("rigidLinkMemberSnapshotSha256")
    if (
        not isinstance(snapshot_sha256, str)
        or len(snapshot_sha256) != 64
        or any(character not in "0123456789abcdef" for character in snapshot_sha256)
    ):
        _fail(
            "authored_aggregate_mismatch",
            f"aggregate member snapshot seal is invalid at {link.body_prim_path}",
        )
    expected: dict[str, str] = {
        "rigidLinkAuthoring": "aggregate",
        "rigidLinkAuthoringVersion": RIGID_LINK_AUTHORING_VERSION,
        "rigidLinkId": link.link_id,
        "rigidLinkMemberSnapshotSha256": snapshot_sha256,
        "rigidLinkPlanSha256": canonical_sha256(link),
    }
    if observed != expected:
        _fail(
            "authored_aggregate_mismatch",
            f"aggregate metadata does not match the request at {link.body_prim_path}",
        )
    return snapshot_sha256


def _member_snapshot_sha256(members: tuple[_MemberSnapshot, ...]) -> str:
    return canonical_sha256(
        {
            "members": [
                {
                    "authored_prim_path": snapshot.member.authored_prim_path,
                    "prims": [
                        {
                            "active": prim.active,
                            "defined": prim.defined,
                            "instanceable": prim.instanceable,
                            "relative_path": prim.relative_path,
                            "type_name": prim.type_name,
                            "world_transform": prim.world_transform,
                        }
                        for prim in snapshot.prims
                    ],
                    "source_prim_path": snapshot.member.source_prim_path,
                }
                for snapshot in members
            ]
        }
    )


def _require_preserved_member_snapshot(
    expected: tuple[_PrimSnapshot, ...],
    observed: tuple[_PrimSnapshot, ...],
    member: RigidLinkMemberPlanV1,
) -> None:
    if len(expected) != len(observed):
        _fail(
            "aggregate_member_changed",
            f"aggregate member prim count changed while moving "
            f"{member.source_prim_path}",
        )
    for before, after in zip(expected, observed, strict=True):
        if (
            before.relative_path != after.relative_path
            or before.type_name != after.type_name
            or before.active != after.active
            or before.defined != after.defined
            or before.instanceable != after.instanceable
        ):
            _fail(
                "aggregate_member_changed",
                f"aggregate member structure changed while moving "
                f"{member.source_prim_path}",
            )
        if (before.world_transform is None) != (after.world_transform is None):
            _fail(
                "aggregate_member_changed",
                f"aggregate member transformability changed while moving "
                f"{member.source_prim_path}",
            )
        if before.world_transform is not None and after.world_transform is not None:
            if not all(
                math.isclose(
                    left,
                    right,
                    rel_tol=_TRANSFORM_TOLERANCE,
                    abs_tol=_TRANSFORM_TOLERANCE,
                )
                for left, right in zip(
                    before.world_transform,
                    after.world_transform,
                    strict=True,
                )
            ):
                _fail(
                    "aggregate_member_world_transform_changed",
                    f"aggregate member world transform changed while moving "
                    f"{member.source_prim_path}",
                )


def _rollback_aggregate_authoring(
    stage: Any,
    rollback_layer: Any,
) -> BaseException | None:
    try:
        stage.GetRootLayer().TransferContent(rollback_layer)
    except BaseException as exc:
        return exc
    return None


def _parent_path(path: str) -> str:
    parent, _, _ = path.rpartition("/")
    return parent or "/"


def _valid_prim(prim: Any) -> bool:
    return bool(prim and prim.IsValid())


def _pxr_modules() -> tuple[Any, Any]:
    try:
        from pxr import Sdf, UsdGeom
    except ImportError as exc:  # pragma: no cover - optional-runtime guard
        raise JointRiggerContractError(
            "openusd_unavailable",
            "OpenUSD bindings are required for aggregate rigid-link authoring",
        ) from exc
    return Sdf, UsdGeom


def _fail(code: str, detail: str) -> Never:
    raise JointRiggerContractError(code, detail)


__all__ = [
    "RIGID_LINK_AUTHORING_VERSION",
    "author_aggregate_rigid_links",
    "request_has_aggregate_links",
    "validate_authored_rigid_links",
]
