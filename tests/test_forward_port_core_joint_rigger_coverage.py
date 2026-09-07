# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release-gate coverage for fail-closed Joint Rigger core branches."""

from __future__ import annotations

import importlib
import io
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom

from world_understanding.functions.physics.joint_rigger.models import (
    JointRiggerContractError,
    JointRiggerInputV2,
    JointRiggerPlanV2,
    RigidLinkMemberPlanV1,
    RigidLinkPlanV1,
)

author = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.author"
)
combined = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.combined"
)
mass_properties = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.mass_properties"
)
models = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.models"
)
opaque_dependencies = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.opaque_dependencies"
)
reference = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.reference"
)
rigid_links = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.rigid_links"
)
schemas = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.schemas"
)
source_binding = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.source_binding"
)
validation = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.validation"
)


def _contract_error(code: str):
    return pytest.raises(JointRiggerContractError, match=rf"^{code}:")


def _joint(body0: str, body1: str, *, joint_id: str = "/World/joint") -> Any:
    return SimpleNamespace(
        topology=SimpleNamespace(body0=body0, body1=body1, joint_id=joint_id)
    )


def _member() -> RigidLinkMemberPlanV1:
    return RigidLinkMemberPlanV1(
        source_prim_path="/World/source",
        authored_prim_path="/World/body/source",
    )


def _aggregate_link() -> RigidLinkPlanV1:
    # ``model_construct`` is intentional: low-level authoring preflight must
    # still fail closed if a caller bypasses the public model boundary.
    return RigidLinkPlanV1.model_construct(
        link_id="body",
        body_authoring="aggregate",
        body_prim_path="/World/body",
        members=(_member(),),
    )


def test_opaque_mdl_import_parser_rejects_malformed_absolute_targets() -> None:
    assert opaque_dependencies._mdl_runtime_import_module("relative::symbol") is None
    assert opaque_dependencies._mdl_runtime_import_module("::module::9bad") is None
    assert opaque_dependencies._mdl_runtime_import_module("::bad-name::*") is None
    assert opaque_dependencies._mdl_runtime_import_module("::good::module::*") == (
        "good::module"
    )

    source = "  first, second   "
    opaque_dependencies._validate_mdl_using_selectors(
        source,
        start=0,
        end=len(source),
        document=Path("material.mdl"),
    )
    malformed = "first, 9bad"
    with pytest.raises(
        opaque_dependencies.OpaqueDependencyError,
        match=(
            r"^Opaque MDL dependency has an unsupported using import list: "
            r"material\.mdl$"
        ),
    ):
        opaque_dependencies._validate_mdl_using_selectors(
            malformed,
            start=0,
            end=len(malformed),
            document=Path("material.mdl"),
        )


def test_model_helpers_reject_wrong_collider_owner_and_cyclic_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = SimpleNamespace(
        prim_path="/World",
        colliders=(SimpleNamespace(prim_path="/World/link/shape"),),
    )
    inner = SimpleNamespace(prim_path="/World/link", colliders=())
    with pytest.raises(ValueError, match="nearest planned rigid body /World/link"):
        JointRiggerPlanV2._canonical_bodies((outer, inner))

    # Duplicate a queued child to exercise the defensive visited-set guard.
    real_sorted = sorted
    empty_adjacency_sorts = 0

    def duplicate_child(value: Any, *args: Any, **kwargs: Any) -> list[Any]:
        nonlocal empty_adjacency_sorts
        result = real_sorted(value, *args, **kwargs)
        if not result:
            empty_adjacency_sorts += 1
        return result * 2 if result == ["/child"] else result

    with monkeypatch.context() as patch:
        patch.setitem(models.__dict__, "sorted", duplicate_child)
        assert models._directed_joint_graph_roots((_joint("/root", "/child"),)) == {
            "/root"
        }
    assert empty_adjacency_sorts == 1

    cycle = (
        _joint("/World", "/Elsewhere"),
        _joint("/Elsewhere", "/World", joint_id="/World/joint2"),
    )
    with pytest.raises(ValueError, match="matching transitive joint ancestry"):
        models._require_nested_existing_graph_ancestry(
            "/World/nested",
            "/World",
            cycle,
        )


def test_mass_property_scalar_guards_are_deterministic() -> None:
    with _contract_error("invalid_mass_properties"):
        mass_properties._canonicalize_quaternion(
            (math.nan, 0.0, 0.0, 1.0), label="axes"
        )
    with _contract_error("invalid_mass_properties"):
        mass_properties._canonicalize_quaternion((0.0, 0.0, 0.0, 0.0), label="axes")
    assert mass_properties._canonicalize_quaternion(
        (0.0, 0.0, 0.0, -2.0), label="axes"
    ) == (0.0, 0.0, 0.0, 1.0)
    with _contract_error("descendant_mass_transform_invalid"):
        mass_properties._finite_float(math.inf, label="translation")


def test_mass_frame_path_unit_and_chain_guards() -> None:
    prim = SimpleNamespace(GetPath=lambda: Sdf.Path("/World/owner/part"))
    owner = SimpleNamespace(GetPath=lambda: Sdf.Path("/World/owner"))
    with _contract_error("invalid_stage_units"):
        mass_properties._lift_descendant_mass_frame(
            prim,
            owner,
            center_of_mass_m=(0.0, 0.0, 0.0),
            principal_axes=(1.0, 0.0, 0.0, 0.0),
            meters_per_unit=0.0,
        )

    same = SimpleNamespace(GetPath=lambda: Sdf.Path("/World/owner"))
    with _contract_error("descendant_mass_owner_mismatch"):
        mass_properties._require_descendant_path(
            same,
            owner,
            contributor_path="/World/owner",
            owner_path="/World/owner",
        )

    connected = SimpleNamespace(
        HasAuthoredConnections=lambda: True,
        GetName=lambda: "xformOp:transform",
    )
    current = MagicMock()
    current.IsValid.return_value = True
    current.IsPseudoRoot.return_value = False
    current.IsInstance.return_value = False
    current.IsInstanceProxy.return_value = False
    current.IsInstanceable.return_value = False
    current.GetPath.return_value = Sdf.Path("/World/owner/part")
    fake_xformable = SimpleNamespace(
        GetResetXformStack=lambda: False,
        GetXformOpOrderAttr=lambda: connected,
        GetOrderedXformOps=lambda: (),
    )
    fake_usd_geom = SimpleNamespace(Xformable=lambda _prim: fake_xformable)
    with _contract_error("descendant_mass_transform_connected"):
        mass_properties._require_static_noninstance_chain(
            current,
            owner,
            contributor_path="/World/owner/part",
            owner_path="/World/owner",
            UsdGeom=fake_usd_geom,
        )

    pseudo_root = MagicMock()
    pseudo_root.IsValid.return_value = True
    pseudo_root.IsPseudoRoot.return_value = True
    detached = MagicMock()
    detached.IsValid.return_value = True
    detached.IsPseudoRoot.return_value = False
    detached.IsInstance.return_value = False
    detached.IsInstanceProxy.return_value = False
    detached.IsInstanceable.return_value = False
    detached.GetPath.return_value = Sdf.Path("/Other/part")
    detached.GetParent.return_value = pseudo_root
    with _contract_error("descendant_mass_owner_mismatch"):
        mass_properties._require_static_noninstance_chain(
            detached,
            owner,
            contributor_path="/Other/part",
            owner_path="/World/owner",
            UsdGeom=SimpleNamespace(Xformable=lambda _prim: None),
        )


@pytest.mark.parametrize(
    ("matrix", "code"),
    [
        ([[math.nan] * 4 for _ in range(4)], "descendant_mass_transform_invalid"),
        (
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "descendant_mass_transform_not_rigid",
        ),
    ],
)
def test_mass_frame_matrix_rejects_nonfinite_and_projective_values(
    matrix: list[list[float]], code: str
) -> None:
    with _contract_error(code):
        mass_properties._require_rigid_matrix(
            matrix,
            contributor_path="/World/part",
            owner_path="/World",
            Gf=SimpleNamespace(),
        )


def test_source_binding_cleanup_mountinfo_and_state_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_bytes(b"#usda 1.0\n")
    snapshot = tempfile.TemporaryFile(dir=tmp_path)
    monkeypatch.setattr(
        source_binding,
        "_create_disk_backed_snapshot_file",
        lambda _source: snapshot,
    )
    with pytest.raises(
        source_binding.JointRiggerArtifactError, match="expected identity"
    ):
        source_binding._create_sealed_file_binding(
            source,
            expected_sha256="0" * 64,
            prefer_disk_snapshot=True,
        )
    assert snapshot.closed

    device = os.makedev(8, 1)
    monkeypatch.setattr(
        source_binding.os,
        "fstat",
        lambda _descriptor: SimpleNamespace(st_dev=device),
    )
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: io.StringIO("1 2 8:1 / / rw shared:1 no-separator\n"),
    )
    with pytest.raises(OSError, match="Could not identify filesystem type"):
        source_binding._descriptor_filesystem_types(7)


def test_sealed_disk_binding_detects_descriptor_state_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "bound.usda"
    path.write_bytes(b"#usda 1.0\n")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        observed = os.fstat(descriptor)
        captured = list(source_binding._descriptor_state(observed))
        captured[-1] += 1
        binding = source_binding.SealedDependencyBinding(
            path=path,
            descriptor=descriptor,
            sha256="a" * 64,
            storage_kind="pinned_file",
            descriptor_state=tuple(captured),
        )
        monkeypatch.setattr(
            source_binding,
            "_stable_descriptor_sha256",
            lambda *_args, **_kwargs: "a" * 64,
        )
        with pytest.raises(
            source_binding.JointRiggerArtifactError,
            match="Pinned input file changed",
        ):
            source_binding._require_sealed_file_binding(binding)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("aggregate", [False, True])
def test_topology_probe_routes_v2_rigid_link_authoring(
    aggregate: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = JointRiggerInputV2.model_construct(
        plan=SimpleNamespace(rigid_bodies=()), rigid_links=()
    )
    stage = object()
    calls: list[str] = []

    @contextmanager
    def projection(*_args: Any, **_kwargs: Any):
        yield object(), Path("bound.usda"), object(), object()

    monkeypatch.setattr(author, "_validate_supported_request", lambda _request: None)
    monkeypatch.setattr(author, "_validate_raw_usd_path", lambda *_a, **_k: None)
    monkeypatch.setattr(author, "_topology_phase_plan", lambda plan: plan)
    monkeypatch.setattr(
        author, "request_has_aggregate_links", lambda _request: aggregate
    )
    monkeypatch.setattr(author, "_bound_source_projection", projection)
    monkeypatch.setattr(author, "_open_stage", lambda *_a, **_k: stage)
    monkeypatch.setattr(
        author,
        "author_aggregate_rigid_links",
        lambda *_a: calls.append("aggregate"),
    )
    monkeypatch.setattr(
        author,
        "validate_authored_rigid_links",
        lambda *_a: calls.append("existing"),
    )
    monkeypatch.setattr(
        author,
        "_author_v2_articulation_roots",
        lambda *_a: calls.append("roots"),
    )
    monkeypatch.setattr(author, "_preflight_topology_authoring", lambda *_a: None)
    monkeypatch.setattr(author, "require_sealed_source_binding", lambda _binding: None)

    author.OwnedTopologyBackend(Path("source.usda")).probe(request)

    assert calls == (["aggregate", "roots"] if aggregate else ["existing", "roots"])


@pytest.mark.parametrize("aggregate", [False, True])
def test_combined_probe_routes_v2_rigid_link_authoring(
    aggregate: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = JointRiggerInputV2.model_construct(plan=object(), rigid_links=())
    stage = object()
    calls: list[str] = []

    @contextmanager
    def projection(*_args: Any, **_kwargs: Any):
        yield object(), Path("bound.usda"), object(), object()

    for name in (
        "_validate_supported_request",
        "validate_physics_plan_evidence",
        "_validate_raw_usd_path",
    ):
        monkeypatch.setattr(combined, name, lambda *_a, **_k: None)
    monkeypatch.setattr(combined, "_topology_phase_plan", lambda plan: plan)
    monkeypatch.setattr(
        combined, "request_has_aggregate_links", lambda _request: aggregate
    )
    monkeypatch.setattr(combined, "_bound_source_projection", projection)
    monkeypatch.setattr(combined, "_open_stage", lambda *_a, **_k: stage)
    monkeypatch.setattr(
        combined,
        "author_aggregate_rigid_links",
        lambda *_a: calls.append("aggregate"),
    )
    monkeypatch.setattr(
        combined,
        "validate_authored_rigid_links",
        lambda *_a: calls.append("existing"),
    )
    monkeypatch.setattr(combined, "_preflight_topology_authoring", lambda *_a: None)
    monkeypatch.setattr(
        combined, "require_sealed_source_binding", lambda _binding: None
    )

    combined.OwnedTopologyAndPhysicsBackend(Path("source.usda")).probe(request)

    assert calls == (["aggregate"] if aggregate else ["existing"])


def test_reference_static_property_rejects_raw_connections() -> None:
    layer = SimpleNamespace(
        identifier="source.usda",
        ListTimeSamplesForPath=lambda _path: (),
    )
    property_spec = SimpleNamespace(
        layer=layer,
        path=Sdf.Path("/World.body.physics:value"),
        ListInfoKeys=lambda: ("connectionPaths",),
    )
    attribute = SimpleNamespace(
        GetName=lambda: "physics:value",
        HasAuthoredConnections=lambda: False,
        GetTimeSamples=lambda: (),
        GetPropertyStack=lambda: (property_spec,),
    )
    with _contract_error("unsupported_attribute_connection"):
        reference._require_static_attribute(attribute, owner_path="/World/body")


def test_reference_physx_max_velocity_requires_drive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prop = SimpleNamespace(GetName=lambda: reference._PHYSX_MAX_JOINT_VELOCITY)
    prim = SimpleNamespace(GetAuthoredProperties=lambda: (prop,))
    monkeypatch.setattr(
        reference, "_applied_schema_tokens", lambda _prim: {"PhysxJointAPI"}
    )
    with _contract_error("unsupported_optional_schema"):
        reference._extract_physx_joint_opinions(
            prim,
            joint_type="revolute",
            joint_path="/World/joint",
            has_drive=False,
            reference_identity=object(),
        )


def test_reference_matching_mass_requires_active_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference, "_is_active_defined_prim", lambda _prim: False)
    assert not reference._matching_preexisting_mass_facts(
        object(), object(), UsdPhysics=object()
    )


def test_reference_mass_center_conversion_and_vector_shape_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mass_attr = SimpleNamespace(Get=lambda: 1.0, GetName=lambda: "physics:mass")
    inertia_attr = SimpleNamespace(
        Get=lambda: (1.0, 1.0, 1.0),
        GetName=lambda: "physics:diagonalInertia",
    )
    center_attr = SimpleNamespace(
        Get=lambda: (1e308, 0.0, 0.0),
        GetName=lambda: "physics:centerOfMass",
    )
    principal_attr = SimpleNamespace(Get=lambda: None)
    attributes = {
        "physics:centerOfMass": center_attr,
    }
    prim = SimpleNamespace(
        HasAPI=lambda _schema: True,
        GetAttribute=lambda name: attributes.get(name, SimpleNamespace(authored=False)),
    )
    api = SimpleNamespace(
        GetMassAttr=lambda: mass_attr,
        GetDiagonalInertiaAttr=lambda: inertia_attr,
        GetPrincipalAxesAttr=lambda: principal_attr,
    )
    fake_usd_physics = SimpleNamespace(
        MassAPI=lambda _prim: api,
    )
    authored_ids = {id(mass_attr), id(inertia_attr), id(center_attr)}
    monkeypatch.setattr(
        reference,
        "_is_authored_value_only_attribute",
        lambda attribute, **_kwargs: id(attribute) in authored_ids,
    )
    monkeypatch.setattr(reference, "_require_static_attribute", lambda *_a, **_k: None)
    with _contract_error("invalid_mass_properties"):
        reference._extract_mass(
            prim,
            body_path="/World/body",
            reference_identity=object(),
            kilograms_per_unit=1.0,
            meters_per_unit=2.0,
            UsdPhysics=fake_usd_physics,
        )

    with _contract_error("invalid_mass_properties"):
        reference._mass_vector3(
            7,
            body_path="/World/body",
            label="center of mass",
        )
    with _contract_error("invalid_mass_properties"):
        reference._mass_vector3(
            (1.0, 2.0),
            body_path="/World/body",
            label="center of mass",
        )


def test_validation_rejects_spherical_friction() -> None:
    plan = SimpleNamespace(
        joints=(
            SimpleNamespace(
                topology=SimpleNamespace(
                    joint_id="/World/joint", joint_type="spherical"
                ),
                state=None,
                mimic=None,
                drive=None,
                joint_friction=object(),
                limit=None,
                anchor=None,
            ),
        ),
        rigid_bodies=(),
        articulation_root=None,
    )
    with _contract_error("joint_friction_not_applicable"):
        validation._validate_supported_plan_shape(plan)


def test_anchor_reconciliation_skips_explicit_drift_candidates() -> None:
    class Transform:
        def Transform(self, value: Any) -> Any:
            return value

        def GetInverse(self) -> Transform:
            return self

    transform = Transform()
    local_pos0 = (5e-6, 0.0, 0.0)
    local_pos1 = (6e-6, 0.0, 0.0)
    result = validation._reconcile_float32_local_anchors(
        transform,
        transform,
        requested_anchor=(0.0, 0.0, 0.0),
        local_pos0=local_pos0,
        local_pos1=local_pos1,
        explicit_anchor=True,
        label="joint",
        Gf=Gf,
    )
    assert result == (local_pos0, local_pos1)


def test_rigid_link_v1_paths_are_intentional_noops() -> None:
    request = SimpleNamespace(rigid_links=())
    assert rigid_links.author_aggregate_rigid_links(object(), request) is None
    assert rigid_links.validate_authored_rigid_links(object(), request) is None


@pytest.mark.parametrize(
    ("can_apply", "apply", "code"),
    [
        (False, True, "aggregate_namespace_edit_rejected"),
        (True, False, "aggregate_namespace_edit_failed"),
    ],
)
def test_aggregate_authoring_rejects_atomic_namespace_edit_failures(
    can_apply: bool,
    apply: bool,
    code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    link = _aggregate_link()
    aggregate = rigid_links._AggregatePreflight(link=link, members=())
    request = JointRiggerInputV2.model_construct(rigid_links=(link,))
    body = MagicMock()
    body.IsValid.return_value = True
    root = SimpleNamespace(
        CanApply=lambda _edits: can_apply,
        Apply=lambda _edits: apply,
        TransferContent=lambda _layer: None,
    )

    edits: list[tuple[str, str]] = []

    class Batch:
        def Add(self, source: str, target: str) -> None:
            edits.append((source, target))

    fake_sdf = SimpleNamespace(
        Layer=SimpleNamespace(
            CreateAnonymous=lambda _name: SimpleNamespace(
                TransferContent=lambda _layer: None
            )
        ),
        BatchNamespaceEdit=Batch,
    )
    fake_usd_geom = SimpleNamespace(
        Xform=SimpleNamespace(
            Define=lambda _stage, _path: SimpleNamespace(GetPrim=lambda: body)
        )
    )
    stage = SimpleNamespace(GetRootLayer=lambda: root)
    monkeypatch.setattr(
        rigid_links, "_preflight_source_links", lambda *_args: (aggregate,)
    )
    monkeypatch.setattr(rigid_links, "_pxr_modules", lambda: (fake_sdf, fake_usd_geom))
    monkeypatch.setattr(
        rigid_links, "_author_aggregate_metadata", lambda *_a, **_k: None
    )

    with _contract_error(code):
        rigid_links.author_aggregate_rigid_links(stage, request)
    assert edits == [
        (link.members[0].source_prim_path, link.members[0].authored_prim_path)
    ]


def test_aggregate_authoring_rejects_invalid_body_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    link = _aggregate_link()
    aggregate = rigid_links._AggregatePreflight(link=link, members=())
    request = JointRiggerInputV2.model_construct(rigid_links=(link,))
    fake_sdf = SimpleNamespace(
        Layer=SimpleNamespace(
            CreateAnonymous=lambda _name: SimpleNamespace(
                TransferContent=lambda _layer: None
            )
        )
    )
    fake_usd_geom = SimpleNamespace(
        Xform=SimpleNamespace(
            Define=lambda _stage, _path: SimpleNamespace(GetPrim=lambda: None)
        )
    )
    stage = SimpleNamespace(
        GetRootLayer=lambda: SimpleNamespace(TransferContent=lambda _layer: None)
    )
    monkeypatch.setattr(
        rigid_links, "_preflight_source_links", lambda *_args: (aggregate,)
    )
    monkeypatch.setattr(rigid_links, "_pxr_modules", lambda: (fake_sdf, fake_usd_geom))
    with _contract_error("aggregate_body_creation_failed"):
        rigid_links.author_aggregate_rigid_links(stage, request)


@pytest.mark.parametrize(
    "seam",
    [
        rigid_links._author_aggregate_rigid_link_plans,
        rigid_links._validate_authored_rigid_link_plans,
    ],
)
def test_exact_plan_seams_reject_cross_link_aggregate_overlap(seam: Any) -> None:
    aggregate = RigidLinkPlanV1(
        link_id="aggregate",
        body_authoring="aggregate",
        body_prim_path="/World/AssemblyA/BaseAggregate",
        members=(
            RigidLinkMemberPlanV1(
                source_prim_path="/World/AssemblyA/Base",
                authored_prim_path="/World/AssemblyA/BaseAggregate/Base",
            ),
            RigidLinkMemberPlanV1(
                source_prim_path="/World/AssemblyA/BaseTrim",
                authored_prim_path="/World/AssemblyA/BaseAggregate/BaseTrim",
            ),
        ),
    )
    overlapping_existing = RigidLinkPlanV1(
        link_id="assembly",
        body_authoring="existing",
        body_prim_path="/World/AssemblyA",
        members=(
            RigidLinkMemberPlanV1(
                source_prim_path="/World/AssemblyA",
                authored_prim_path="/World/AssemblyA",
            ),
        ),
    )

    with _contract_error("rigid_link_cross_link_invalid"):
        seam(object(), (aggregate, overlapping_existing))


@pytest.mark.parametrize(
    "seam",
    [
        rigid_links._author_aggregate_rigid_link_plans,
        rigid_links._validate_authored_rigid_link_plans,
    ],
)
def test_exact_plan_seams_require_exact_plan_tuple(seam: Any) -> None:
    with pytest.raises(
        TypeError,
        match="rigid_links must contain exact RigidLinkPlanV1 values",
    ):
        seam(object(), [])


def test_aggregate_preflight_stage_and_namespace_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    link = _aggregate_link()
    request = JointRiggerInputV2.model_construct(rigid_links=(link,))
    with _contract_error("invalid_stage"):
        rigid_links._preflight_source_links(None, request)

    def stage(root: Any, edit: Any) -> Any:
        return SimpleNamespace(
            GetRootLayer=lambda: root,
            GetEditTarget=lambda: SimpleNamespace(GetLayer=lambda: edit),
        )

    root = SimpleNamespace(identifier="root", permissionToEdit=True)
    with _contract_error("aggregate_edit_target_mismatch"):
        rigid_links._preflight_source_links(
            stage(root, SimpleNamespace(identifier="other")), request
        )

    readonly = SimpleNamespace(identifier="root", permissionToEdit=False)
    with _contract_error("aggregate_root_not_editable"):
        rigid_links._preflight_source_links(stage(readonly, readonly), request)

    top_level_link = RigidLinkPlanV1.model_construct(
        link_id="body",
        body_authoring="aggregate",
        body_prim_path="/body",
        members=(_member(),),
    )
    top_level_request = JointRiggerInputV2.model_construct(
        rigid_links=(top_level_link,)
    )
    editable = SimpleNamespace(identifier="root", permissionToEdit=True)
    with _contract_error("aggregate_top_level_unsupported"):
        rigid_links._preflight_source_links(
            stage(editable, editable), top_level_request
        )

    monkeypatch.setattr(rigid_links, "_require_root_owned_prim", lambda *_a, **_k: None)
    root_layer = SimpleNamespace(
        identifier="root",
        permissionToEdit=True,
        GetPrimAtPath=lambda _path: None,
    )
    wrong_parent = SimpleNamespace(
        GetParent=lambda: SimpleNamespace(GetPath=lambda: Sdf.Path("/Other"))
    )
    mismatch_stage = SimpleNamespace(
        GetRootLayer=lambda: root_layer,
        GetEditTarget=lambda: SimpleNamespace(GetLayer=lambda: root_layer),
        GetPrimAtPath=lambda path: wrong_parent if path == "/World/source" else None,
    )
    with _contract_error("aggregate_source_parent_mismatch"):
        rigid_links._preflight_source_links(mismatch_stage, request)

    correct_parent = SimpleNamespace(
        GetParent=lambda: SimpleNamespace(GetPath=lambda: Sdf.Path("/World"))
    )
    collision = MagicMock()
    collision.IsValid.return_value = True
    collision_stage = SimpleNamespace(
        GetRootLayer=lambda: root_layer,
        GetEditTarget=lambda: SimpleNamespace(GetLayer=lambda: root_layer),
        GetPrimAtPath=lambda path: correct_parent
        if path == "/World/source"
        else collision
        if path == "/World/body/source"
        else None,
    )
    with _contract_error("aggregate_authored_path_collision"):
        rigid_links._preflight_source_links(collision_stage, request)


def test_rigid_link_prim_ownership_and_subtree_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_stage = SimpleNamespace(GetPrimAtPath=lambda _path: None)
    existing = RigidLinkPlanV1.model_construct(
        body_prim_path="/World/missing",
        body_authoring="existing",
        link_id="missing",
        members=(),
    )
    with _contract_error("existing_link_missing"):
        rigid_links._require_existing_identity_link(missing_stage, existing)

    spec = SimpleNamespace(
        layer=SimpleNamespace(identifier="root"),
        path=SimpleNamespace(ContainsPrimVariantSelection=lambda: True),
    )
    prim = MagicMock()
    prim.IsValid.return_value = True
    prim.IsActive.return_value = True
    prim.IsDefined.return_value = True
    prim.IsInstance.return_value = False
    prim.IsInstanceProxy.return_value = False
    prim.IsPrototype.return_value = False
    prim.IsInPrototype.return_value = False
    prim.IsInstanceable.return_value = False
    prim.GetPrimStack.return_value = (spec,)
    prim.GetPath.return_value = Sdf.Path("/World/source")
    with _contract_error("aggregate_variant_unsupported"):
        rigid_links._require_root_owned_prim(
            prim, SimpleNamespace(identifier="root"), label="member"
        )

    root = SimpleNamespace(GetPath=lambda: Sdf.Path("/World/source"))
    child = MagicMock()
    child.GetPath.return_value = Sdf.Path("/World/source/child")
    stage = SimpleNamespace(TraverseAll=lambda: (child,))
    monkeypatch.setattr(rigid_links, "_RIGID_LINK_SCAN_MAX_PRIM_VISITS", 0)
    with _contract_error("aggregate_subtree_scan_limit_exceeded"):
        rigid_links._reject_unsupported_subtree(stage, root, object())
    fake_usd_geom = SimpleNamespace(XformCache=lambda: object())
    monkeypatch.setattr(
        rigid_links,
        "_pxr_modules",
        lambda: (SimpleNamespace(), fake_usd_geom),
    )
    with _contract_error("aggregate_subtree_scan_limit_exceeded"):
        rigid_links._capture_subtree(stage, root)

    monkeypatch.setattr(rigid_links, "_RIGID_LINK_SCAN_MAX_PRIM_VISITS", 10)
    monkeypatch.setattr(rigid_links, "_require_root_owned_prim", lambda *_a, **_k: None)
    child.HasAuthoredReferences.return_value = True
    with _contract_error("aggregate_composition_arc_unsupported"):
        rigid_links._reject_unsupported_subtree(stage, root, object())
    child.HasAuthoredReferences.return_value = False
    child.HasAuthoredPayloads.return_value = False
    child.HasAuthoredInherits.return_value = False
    child.HasAuthoredSpecializes.return_value = False
    child.GetVariantSets.return_value.GetNames.return_value = ("choice",)
    with _contract_error("aggregate_variant_unsupported"):
        rigid_links._reject_unsupported_subtree(stage, root, object())

    empty_stage = SimpleNamespace(TraverseAll=lambda: ())
    with _contract_error("aggregate_source_missing"):
        rigid_links._capture_subtree(empty_stage, root)


def _valid_aggregate_body() -> MagicMock:
    body = MagicMock()
    body.IsValid.return_value = True
    body.IsActive.return_value = True
    body.IsDefined.return_value = True
    body.IsA.return_value = True
    body.IsInstance.return_value = False
    body.IsInstanceProxy.return_value = False
    body.IsInstanceable.return_value = False
    return body


def test_authored_aggregate_shape_and_snapshot_seal_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    link = _aggregate_link()
    body = _valid_aggregate_body()
    xformable = SimpleNamespace(
        GetOrderedXformOps=lambda: (object(),),
        GetResetXformStack=lambda: False,
    )
    fake_usd_geom = SimpleNamespace(Xform=object(), Xformable=lambda _body: xformable)
    stage = SimpleNamespace(GetPrimAtPath=lambda _path: body)
    monkeypatch.setattr(
        rigid_links,
        "_pxr_modules",
        lambda: (SimpleNamespace(), fake_usd_geom),
    )
    with _contract_error("authored_aggregate_mismatch"):
        rigid_links._validate_authored_aggregate(stage, link, expected_members=None)

    xformable.GetOrderedXformOps = lambda: ()
    monkeypatch.setattr(
        rigid_links, "_validate_aggregate_metadata", lambda *_a: "a" * 64
    )
    monkeypatch.setattr(rigid_links, "_member_snapshot_sha256", lambda _a: "b" * 64)
    with _contract_error("authored_aggregate_mismatch"):
        rigid_links._validate_authored_aggregate(stage, link, expected_members=())

    body.GetAllChildren.return_value = ()
    with _contract_error("authored_aggregate_mismatch"):
        rigid_links._validate_authored_aggregate(stage, link, expected_members=None)

    expected_child = SimpleNamespace(
        GetPath=lambda: Sdf.Path(link.members[0].authored_prim_path)
    )
    body.GetAllChildren.return_value = (expected_child,)
    source = MagicMock()
    source.IsValid.return_value = True
    stage.GetPrimAtPath = lambda path: body if path == link.body_prim_path else source
    with _contract_error("authored_aggregate_mismatch"):
        rigid_links._validate_authored_aggregate(stage, link, expected_members=None)

    stage.GetPrimAtPath = lambda path: body if path == link.body_prim_path else None
    with _contract_error("authored_aggregate_mismatch"):
        rigid_links._validate_authored_aggregate(stage, link, expected_members=None)


def test_aggregate_metadata_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    link = _aggregate_link()
    body = MagicMock()
    with monkeypatch.context() as patch:
        patch.setattr(rigid_links, "_validate_aggregate_metadata", lambda *_a: "b" * 64)
        with _contract_error("aggregate_metadata_authoring_failed"):
            rigid_links._author_aggregate_metadata(
                body, link, member_snapshot_sha256="a" * 64
            )

    body.GetCustomData.return_value = {}
    with _contract_error("authored_aggregate_mismatch"):
        rigid_links._validate_aggregate_metadata(body, link)
    body.GetCustomData.return_value = {
        "jointRigger": {"rigidLinkMemberSnapshotSha256": "not-a-digest"}
    }
    with _contract_error("authored_aggregate_mismatch"):
        rigid_links._validate_aggregate_metadata(body, link)


def test_aggregate_member_snapshot_guards() -> None:
    member = _member()

    def snapshot(
        *,
        path: str = ".",
        type_name: str = "Mesh",
        transform: tuple[float, ...] | None = (1.0,) * 16,
    ) -> Any:
        return rigid_links._PrimSnapshot(
            relative_path=path,
            type_name=type_name,
            active=True,
            defined=True,
            instanceable=False,
            world_transform=transform,
        )

    before = snapshot()
    with _contract_error("aggregate_member_changed"):
        rigid_links._require_preserved_member_snapshot((before,), (), member)
    with _contract_error("aggregate_member_changed"):
        rigid_links._require_preserved_member_snapshot(
            (before,), (snapshot(type_name="Xform"),), member
        )
    with _contract_error("aggregate_member_changed"):
        rigid_links._require_preserved_member_snapshot(
            (before,), (snapshot(transform=None),), member
        )
    changed = (1.0,) * 15 + (2.0,)
    with _contract_error("aggregate_member_world_transform_changed"):
        rigid_links._require_preserved_member_snapshot(
            (before,), (snapshot(transform=changed),), member
        )


def test_v2_physics_evidence_requires_exact_component_roots() -> None:
    joint = _joint("/World/a", "/World/b")
    bodies = (
        SimpleNamespace(prim_path="/World/a"),
        SimpleNamespace(prim_path="/World/b"),
    )
    missing = JointRiggerPlanV2.model_construct(
        joints=(joint,), rigid_bodies=bodies, articulation_roots=()
    )
    with _contract_error("articulation_roots_missing"):
        schemas.validate_physics_plan_evidence(missing)

    mismatch = JointRiggerPlanV2.model_construct(
        joints=(joint,),
        rigid_bodies=bodies,
        articulation_roots=(SimpleNamespace(prim_path="/World/b"),),
    )
    with _contract_error("articulation_roots_mismatch"):
        schemas.validate_physics_plan_evidence(mismatch)


@pytest.mark.parametrize(
    ("owner", "tokens", "code"),
    [
        ("/World/body", (), "physics_schema_conflict"),
        ("/Other/body", ("PhysicsMassAPI",), "physics_schema_conflict"),
    ],
)
def test_source_collider_mass_contract_requires_schema_and_descendant_owner(
    owner: str,
    tokens: tuple[str, ...],
    code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_path = "/World/body/collider"
    if owner == "/Other/body":
        scene_path = "/World/collider"
    collider = SimpleNamespace(prim_path=scene_path)
    plan = SimpleNamespace(
        rigid_bodies=(SimpleNamespace(prim_path=owner, colliders=(collider,)),)
    )
    contract = schemas._R3RawAuthorshipContract(
        schema_order=(),
        attribute_specs={},
        attribute_defaults={},
        relationship_targets={},
    )
    density = object()
    prim = SimpleNamespace(
        GetAttribute=lambda name: density if name == "physics:density" else None
    )
    stage = SimpleNamespace(GetPrimAtPath=lambda _path: prim)
    monkeypatch.setattr(schemas, "_composed_raw_api_schema_items", lambda _prim: tokens)
    monkeypatch.setattr(schemas, "_has_authored_value", lambda *_a, **_k: True)
    with _contract_error(code):
        schemas._with_valid_source_physics(stage, plan, {scene_path: contract})


def _mass_collider_stage() -> tuple[Any, Any, Any]:
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/collider").GetPrim()
    spec = stage.GetRootLayer().GetPrimAtPath("/World/collider")
    assert spec is not None
    return stage, prim, spec


def _set_mass_api(spec: Any, *, appended: tuple[str, ...] = ()) -> None:
    if appended:
        op = Sdf.TokenListOp()
        op.prependedItems = ["PhysicsMassAPI"]
        op.appendedItems = list(appended)
    else:
        op = Sdf.TokenListOp.CreateExplicit(["PhysicsMassAPI"])
    spec.SetInfo("apiSchemas", op)


def test_source_mass_contract_rejects_missing_or_ambiguous_ownership() -> None:
    source_stage, prim, spec = _mass_collider_stage()
    _set_mass_api(spec)
    empty_stage = Usd.Stage.CreateInMemory()
    with _contract_error("physics_schema_conflict"):
        schemas._source_collider_mass_contract(
            empty_stage, prim, scene_path="/World/collider"
        )

    plain_stage, plain, _plain_spec = _mass_collider_stage()
    with _contract_error("physics_schema_conflict"):
        schemas._source_collider_mass_contract(
            plain_stage, plain, scene_path="/World/collider"
        )

    _set_mass_api(spec, appended=("MaterialBindingAPI",))
    with _contract_error("physics_schema_conflict"):
        schemas._source_collider_mass_contract(
            source_stage, prim, scene_path="/World/collider"
        )


def test_source_mass_contract_rejects_non_list_schema_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pxr = importlib.import_module("pxr")

    class PrimSpec:
        def ListInfoKeys(self) -> tuple[str, ...]:
            return ("apiSchemas",)

        def GetInfo(self, _name: str) -> tuple[str, ...]:
            return ("PhysicsMassAPI",)

    spec = PrimSpec()
    layer = SimpleNamespace(identifier="layer", GetPrimAtPath=lambda _path: spec)
    edit_target = SimpleNamespace(
        GetLayer=lambda: layer,
        MapToSpecPath=lambda path: path,
    )
    prim = SimpleNamespace(GetPrimStack=lambda: (spec,))
    stage = SimpleNamespace(GetEditTarget=lambda: edit_target)
    sdf_shim = SimpleNamespace(
        Path=Sdf.Path,
        PrimSpec=PrimSpec,
        TokenListOp=Sdf.TokenListOp,
    )
    with monkeypatch.context() as patch:
        patch.setattr(real_pxr, "Sdf", sdf_shim)
        with _contract_error("physics_schema_conflict"):
            schemas._source_collider_mass_contract(
                stage, prim, scene_path="/World/collider"
            )


def test_source_mass_contract_property_shape_guards() -> None:
    stage, prim, spec = _mass_collider_stage()
    _set_mass_api(spec)
    malformed_prim = SimpleNamespace(
        GetPrimStack=prim.GetPrimStack,
        GetRelationship=lambda name: object() if name == "physics:mass" else None,
        GetAttribute=prim.GetAttribute,
    )
    with _contract_error("physics_schema_conflict"):
        schemas._source_collider_mass_contract(
            stage, malformed_prim, scene_path="/World/collider"
        )

    stage, prim, spec = _mass_collider_stage()
    _set_mass_api(spec)
    prim.CreateAttribute("physics:density", Sdf.ValueTypeNames.Float).Set(5.0)
    specs, defaults = schemas._source_collider_mass_contract(
        stage, prim, scene_path="/World/collider"
    )
    assert specs == {"physics:density": ("float", "varying")}
    assert defaults["physics:density"] == pytest.approx(5.0)

    stage, prim, spec = _mass_collider_stage()
    prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float, custom=True).Set(1.0)
    _set_mass_api(spec)
    with _contract_error("physics_schema_conflict"):
        schemas._source_collider_mass_contract(
            stage, prim, scene_path="/World/collider"
        )

    stage, prim, spec = _mass_collider_stage()
    _set_mass_api(spec)
    with _contract_error("physics_schema_conflict"):
        schemas._source_collider_mass_contract(
            stage, prim, scene_path="/World/collider"
        )


def test_schema_normalizer_skips_empty_explicit_opinion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pxr = importlib.import_module("pxr")

    class TokenListOp:
        explicitItems: tuple[str, ...] = ()
        addedItems: tuple[str, ...] = ()
        prependedItems: tuple[str, ...] = ()
        appendedItems: tuple[str, ...] = ()
        deletedItems: tuple[str, ...] = ()
        orderedItems: tuple[str, ...] = ()
        isExplicit = True

    class PrimSpec:
        layer = SimpleNamespace(identifier="layer")
        path = Sdf.Path("/World/body")

        def ListInfoKeys(self) -> tuple[str, ...]:
            return ("apiSchemas",)

        def GetInfo(self, _name: str) -> TokenListOp:
            return TokenListOp()

    spec = PrimSpec()
    layer = SimpleNamespace(identifier="layer", GetPrimAtPath=lambda _path: spec)
    edit_target = SimpleNamespace(
        GetLayer=lambda: layer,
        MapToSpecPath=lambda path: path,
    )
    prim = SimpleNamespace(GetPrimStack=lambda: (spec,))
    stage = SimpleNamespace(
        GetEditTarget=lambda: edit_target,
        GetPrimAtPath=lambda _path: prim,
    )
    contract = schemas._R3RawAuthorshipContract(
        schema_order=("PhysicsRigidBodyAPI",),
        attribute_specs={},
        attribute_defaults={},
        relationship_targets={},
    )
    sdf_shim = SimpleNamespace(
        Path=Sdf.Path,
        PrimSpec=PrimSpec,
        TokenListOp=TokenListOp,
    )
    with monkeypatch.context() as patch:
        patch.setattr(real_pxr, "Sdf", sdf_shim)
        schemas._normalize_compatible_explicit_api_schemas(
            stage, {"/World/body": contract}
        )


def test_schema_normalizer_rejects_incompatible_owned_token_order() -> None:
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/body").GetPrim()
    spec = stage.GetRootLayer().GetPrimAtPath("/World/body")
    assert spec is not None
    spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(["PhysicsMassAPI", "PhysicsRigidBodyAPI"]),
    )
    contract = schemas._R3RawAuthorshipContract(
        schema_order=("PhysicsRigidBodyAPI", "PhysicsMassAPI"),
        attribute_specs={},
        attribute_defaults={},
        relationship_targets={},
    )
    assert prim.GetPrimStack()
    with _contract_error("physics_schema_list_op_ambiguous"):
        schemas._normalize_compatible_explicit_api_schemas(
            stage, {"/World/body": contract}
        )


def test_schema_preflight_rejects_nonfinite_nested_body_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pxr = importlib.import_module("pxr")
    plan = SimpleNamespace(
        joints=(_joint("/World/base", "/World/base/child"),),
        rigid_bodies=(
            SimpleNamespace(prim_path="/World/base"),
            SimpleNamespace(prim_path="/World/base/child"),
        ),
    )
    prim = SimpleNamespace(
        IsInstanceable=lambda: False,
        GetAttribute=lambda _name: None,
        GetRelationship=lambda _name: None,
    )
    xformable = SimpleNamespace(GetResetXformStack=lambda: False)
    matrix = [[math.nan] * 4 for _ in range(4)]
    usd_geom_shim = SimpleNamespace(
        GetStageMetersPerUnit=lambda _stage: 1.0,
        XformCache=lambda: SimpleNamespace(
            GetLocalToWorldTransform=lambda _prim: matrix
        ),
        Xformable=lambda _prim: xformable,
    )
    usd_physics_shim = SimpleNamespace(GetStageKilogramsPerUnit=lambda _stage: 1.0)
    monkeypatch.setattr(schemas, "validate_physics_plan_evidence", lambda _plan: None)
    monkeypatch.setattr(schemas, "_graph_roots", lambda *_a: ("/World/base",))
    monkeypatch.setattr(
        schemas,
        "_articulation_roots",
        lambda _plan: (SimpleNamespace(prim_path="/World/base"),),
    )
    monkeypatch.setattr(schemas, "_require_target_prim", lambda *_a, **_k: prim)
    monkeypatch.setattr(schemas, "_require_static_transform_chain", lambda _prim: None)
    with monkeypatch.context() as patch:
        patch.setattr(real_pxr, "UsdGeom", usd_geom_shim)
        patch.setattr(real_pxr, "UsdPhysics", usd_physics_shim)
        with _contract_error("nested_body_transform_invalid"):
            schemas._preflight(object(), plan)


def test_schema_graph_traversal_guard_and_v2_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sorted = sorted
    empty_adjacency_sorts = 0

    def duplicate_child(value: Any, *args: Any, **kwargs: Any) -> list[Any]:
        nonlocal empty_adjacency_sorts
        result = real_sorted(value, *args, **kwargs)
        if not result:
            empty_adjacency_sorts += 1
        return result * 2 if result == ["/child"] else result

    plan = JointRiggerPlanV2.model_construct(joints=(_joint("/root", "/child"),))
    with monkeypatch.context() as patch:
        patch.setitem(schemas.__dict__, "sorted", duplicate_child)
        assert schemas._graph_roots(plan, {"/root", "/child"}) == ("/root",)
    assert empty_adjacency_sorts == 1

    cyclic = JointRiggerPlanV2.model_construct(
        joints=(
            _joint("/a", "/b"),
            _joint("/b", "/a", joint_id="/joint2"),
        )
    )
    with _contract_error("cyclic_joint_graph"):
        schemas._graph_roots(cyclic, {"/a", "/b"})


def test_body_preflight_rejects_unrepresented_density(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    density = object()
    prim = SimpleNamespace(
        GetAttribute=lambda name: density if name == "physics:density" else None,
        GetPath=lambda: Sdf.Path("/World/body"),
    )
    stage = SimpleNamespace(GetPrimAtPath=lambda _path: prim)
    body = SimpleNamespace(
        prim_path="/World/body", mass=object(), colliders=(object(),)
    )
    static: list[Any] = []
    monkeypatch.setattr(schemas, "_preflight_attr", lambda *_a, **_k: None)
    monkeypatch.setattr(schemas, "_has_authored_value", lambda *_a, **_k: True)
    monkeypatch.setattr(
        schemas, "_require_static", lambda attr, **_k: static.append(attr)
    )
    with _contract_error("mass_schema_conflict"):
        schemas._preflight_body(
            stage,
            body,
            body_paths={"/World/body"},
            meters_per_unit=1.0,
            kilograms_per_unit=1.0,
            UsdGeom=object(),
        )
    assert static == [density]


def test_mimic_preflight_rejects_joint_friction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = SimpleNamespace(
        topology=SimpleNamespace(joint_id="/World/joint", joint_type="revolute"),
        drive=None,
        mimic=SimpleNamespace(reference_joint_id="/World/reference"),
        joint_friction=object(),
    )
    prim = SimpleNamespace(GetAuthoredProperties=lambda: ())
    context = SimpleNamespace(prim=prim, plan=plan, motion="angular")
    monkeypatch.setattr(schemas, "_applied_schema_tokens", lambda _prim: set())
    with _contract_error("mimic_schema_conflict"):
        schemas._preflight_joint_control(context, {})


def test_nested_body_apply_and_validation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pxr = importlib.import_module("pxr")
    matrix = tuple(
        1.0 if row == column else 0.0 for row in range(4) for column in range(4)
    )
    preflight = SimpleNamespace(nested_body_world_matrices={"/World/child": matrix})

    matrix_op = SimpleNamespace(Set=lambda _matrix: False)
    apply_xformable = SimpleNamespace(
        AddTransformOp=lambda *_a, **_k: matrix_op,
        SetXformOpOrder=lambda *_a, **_k: True,
    )
    apply_usd_geom = SimpleNamespace(
        Xformable=lambda _prim: apply_xformable,
        XformOp=SimpleNamespace(PrecisionDouble="double"),
    )
    stage = SimpleNamespace(GetPrimAtPath=lambda _path: object())
    with monkeypatch.context() as patch:
        patch.setattr(real_pxr, "UsdGeom", apply_usd_geom)
        with _contract_error("nested_body_reset_failed"):
            schemas._apply(stage, SimpleNamespace(rigid_bodies=()), preflight)

    cases = (
        ((), (), "canonical reset stack"),
        (("!resetXformStack!", schemas._NESTED_BODY_RESET_OP_NAME), (), "ambiguous"),
        (
            ("!resetXformStack!", schemas._NESTED_BODY_RESET_OP_NAME),
            (SimpleNamespace(Get=lambda: [[0.0] * 4 for _ in range(4)]),),
            "does not preserve",
        ),
    )
    for order, ops, detail in cases:
        xformable = SimpleNamespace(
            GetXformOpOrderAttr=lambda order=order: SimpleNamespace(Get=lambda: order),
            GetOrderedXformOps=lambda ops=ops: ops,
        )
        validate_usd_geom = SimpleNamespace(
            Xformable=lambda _prim, current=xformable: current
        )
        with monkeypatch.context() as patch:
            patch.setattr(real_pxr, "UsdGeom", validate_usd_geom)
            with pytest.raises(JointRiggerContractError, match=detail):
                schemas._validate_authored(
                    stage, SimpleNamespace(rigid_bodies=()), preflight
                )
