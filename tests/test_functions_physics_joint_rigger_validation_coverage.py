# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial branch coverage for the owned Joint Rigger validator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pxr")
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics

from world_understanding.functions.physics.joint_rigger import (
    PLAN_SCHEMA_VERSION,
    ArticulationRootPlanV1,
    ArtifactIdentityV1,
    FieldProvenanceV1,
    JointDriveV1,
    JointLimitV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerPlanV1,
    JointTopologyV1,
    validate_authored_joint_topology,
    validate_joint_topology_plan,
)
from world_understanding.functions.physics.joint_rigger import author as author_module
from world_understanding.functions.physics.joint_rigger import (
    validation as validation_module,
)

_BODY0 = "/World/Base"
_BODY1 = "/World/Link"


def _identity() -> ArtifactIdentityV1:
    return ArtifactIdentityV1(
        uri="fixture://joint-rigger-validation/source.usda",
        root_sha256="0" * 64,
    )


def _provenance(field: str, *, prim_path: str = _BODY1) -> FieldProvenanceV1:
    return FieldProvenanceV1(
        source="source_metadata",
        artifact=_identity(),
        prim_path=prim_path,
        properties=(field,),
        evidence=f"Synthetic source evidence for {field}.",
    )


def _drive() -> JointDriveV1:
    return JointDriveV1(
        drive_type="force",
        stiffness=12.0,
        damping=3.0,
        max_force=40.0,
        target_position=5.0,
        target_velocity=0.5,
        provenance=_provenance("drive"),
    )


def _limit() -> JointLimitV1:
    return JointLimitV1(
        lower=-10.0,
        upper=100.0,
        unit="degrees",
        provenance=_provenance("limit"),
    )


def _plan(
    *,
    joint_type: str = "revolute",
    body0: str = _BODY0,
    body1: str = _BODY1,
    drive: JointDriveV1 | None = None,
    limit: JointLimitV1 | None = None,
) -> JointRiggerPlanV1:
    axis = None if joint_type == "spherical" else (0.0, 0.0, 1.0)
    fields = ("joint_type", "body0", "body1")
    if axis is not None:
        fields += ("axis_stage",)
    topology = JointTopologyV1(
        joint_id="coverage joint",
        joint_type=joint_type,
        body0=body0,
        body1=body1,
        axis_stage=axis,
        field_provenance={
            field: _provenance(field, prim_path=body1) for field in fields
        },
    )
    return JointRiggerPlanV1(
        schema_version=PLAN_SCHEMA_VERSION,
        joints=(JointPlanV1(topology=topology, drive=drive, limit=limit),),
    )


def _stage(*, set_default_prim: bool = True) -> Any:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    if set_default_prim:
        stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, _BODY0)
    UsdGeom.Xform.Define(stage, _BODY1)
    return stage


def _stage_with_composed_source_prims() -> Any:
    source_layer = Sdf.Layer.CreateAnonymous("joint-rigger-source.usda")
    source_stage = Usd.Stage.Open(source_layer)
    assert source_stage is not None
    UsdGeom.Xform.Define(source_stage, "/World")
    UsdGeom.Xform.Define(source_stage, _BODY0)
    UsdGeom.Xform.Define(source_stage, _BODY1)
    UsdGeom.Xform.Define(source_stage, "/World/Unrelated")
    UsdGeom.Scope.Define(source_stage, "/World/Joints")

    stage = Usd.Stage.CreateInMemory()
    root_layer = stage.GetRootLayer()
    root_layer.subLayerPaths = [source_layer.identifier]
    root_layer.defaultPrim = "World"
    assert stage.GetDefaultPrim().IsValid()
    return stage


def _authored_stage(
    *,
    joint_type: str = "revolute",
    drive: JointDriveV1 | None = None,
    limit: JointLimitV1 | None = None,
) -> tuple[Any, JointRiggerPlanV1, Any, Any]:
    stage = _stage()
    plan = _plan(joint_type=joint_type, drive=drive, limit=limit)
    preflight = validation_module._preflight_topology_authoring(stage, plan)
    diagnostics = author_module._build_diagnostics(plan, preflight)
    author_module._author_preflight(stage, preflight, diagnostics)
    validation_module._validate_authored_preflight(
        stage,
        preflight,
        diagnostics=diagnostics,
    )
    return stage, plan, preflight, diagnostics


def _tamper_authored_joint_metadata(stage: Any, path: str, tamper: str) -> None:
    """Author one raw metadata opinion outside the exact WP-R2 shape."""

    prim = stage.GetPrimAtPath(path)
    spec = stage.GetRootLayer().GetPrimAtPath(path)
    assert spec is not None
    if tamper in {"raw_api_deleted", "raw_api_ordered"}:
        schemas = spec.GetInfo("apiSchemas")
        assert isinstance(schemas, Sdf.TokenListOp)
        if tamper == "raw_api_deleted":
            schemas.deletedItems = ["UnknownAdversarialAPI"]
        else:
            schemas.orderedItems = ["UnknownAdversarialAPI"]
        spec.SetInfo("apiSchemas", schemas)
    elif tamper == "unexpected_custom_data":
        prim.SetCustomDataByKey("unexpected:payload", "survived")
        assert prim.GetCustomDataByKey("unexpected:payload") == "survived"
    elif tamper == "unexpected_metadata":
        assert prim.SetMetadata("documentation", "unexpected joint metadata")
    elif tamper == "attribute_custom_data":
        attribute = spec.properties["physics:localPos0"]
        attribute.SetInfo("customData", {"unexpected": "survived"})
    elif tamper == "relationship_documentation":
        relationship = spec.properties["physics:body0"]
        relationship.SetInfo("documentation", "unexpected relationship metadata")
    else:  # pragma: no cover - callers and helper must stay in lockstep
        raise AssertionError(f"unhandled authored metadata tamper: {tamper}")


def _assert_code(
    error: pytest.ExceptionInfo[JointRiggerContractError], code: str
) -> None:
    assert error.value.code == code


class _RootPathDefaultPrim:
    """Expose an impossible default-prim path through the public stage contract."""

    def __init__(self, prim: Any) -> None:
        self._prim = prim

    def __bool__(self) -> bool:
        return bool(self._prim)

    def IsValid(self) -> bool:
        return self._prim.IsValid()

    def IsActive(self) -> bool:
        return self._prim.IsActive()

    def IsDefined(self) -> bool:
        return self._prim.IsDefined()

    def GetPath(self) -> Any:
        return Sdf.Path.absoluteRootPath


class _RootPathDefaultStage:
    def __init__(self, stage: Any) -> None:
        self._stage = stage

    def GetDefaultPrim(self) -> _RootPathDefaultPrim:
        return _RootPathDefaultPrim(self._stage.GetDefaultPrim())


def test_preflight_rejects_missing_and_impossible_default_prims() -> None:
    with pytest.raises(JointRiggerContractError) as missing:
        validate_joint_topology_plan(_stage(set_default_prim=False), _plan())
    _assert_code(missing, "invalid_default_prim")

    with pytest.raises(JointRiggerContractError) as root_path:
        validate_joint_topology_plan(_RootPathDefaultStage(_stage()), _plan())
    _assert_code(root_path, "invalid_default_prim")
    assert "defaultPrim path is invalid" in str(root_path.value)


def test_preflight_rejects_instance_default_prim() -> None:
    stage = Usd.Stage.CreateInMemory()
    model = UsdGeom.Xform.Define(stage, "/Model")
    UsdGeom.Xform.Define(stage, "/Model/Base")
    UsdGeom.Xform.Define(stage, "/Model/Link")
    world = stage.OverridePrim("/World")
    assert world.GetReferences().AddInternalReference(model.GetPath())
    assert world.SetInstanceable(True)
    stage.SetDefaultPrim(world)
    assert world.IsInstance()

    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(stage, _plan())

    _assert_code(error, "invalid_default_prim")
    assert "cannot be an instance" in str(error.value)


def test_preflight_rejects_duplicate_deterministic_joint_paths() -> None:
    plan = _plan()
    duplicate_plan = plan.model_copy(update={"joints": (plan.joints[0],) * 2})

    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(_stage(), duplicate_plan)

    _assert_code(error, "joint_target_collision")
    assert "multiple topology entries" in str(error.value)


def test_snapshot_identifier_normalization_handles_relative_mapped_and_absolute(
    tmp_path: Path,
) -> None:
    projected = tmp_path / "projection" / "source.usda"
    logical = tmp_path / "logical" / "source.usda"
    untouched = tmp_path / "untouched.usda"
    prim = validation_module._PrimSnapshot(
        type_name="Xform",
        active=True,
        defined=True,
        instanceable=False,
        applied_schemas=(),
        raw_api_schemas=None,
        authored_prim_specs=(
            ("relative.usda", "/World", ()),
            (str(projected), "/World", ()),
            (str(untouched), "/World", ()),
        ),
        authored_property_specs=(
            (
                "visibility",
                ((str(projected), "/World.visibility", "inherited"),),
            ),
        ),
        world_transform=None,
    )
    snapshot = validation_module._StageSnapshot(
        default_prim_path="/World",
        meters_per_unit=1.0,
        up_axis="Y",
        prims={"/World": prim},
    )

    normalized = validation_module._normalized_snapshot_layer_identifiers(
        snapshot,
        layer_identifier_remap={projected: logical},
    )

    normalized_prim = normalized.prims["/World"]
    assert [item[0] for item in normalized_prim.authored_prim_specs] == [
        "relative.usda",
        str(logical),
        str(untouched.resolve(strict=False)),
    ]
    assert normalized_prim.authored_property_specs[0][1][0][0] == str(logical)


def test_stable_usd_info_value_serializes_nested_sequences() -> None:
    value = ["outer", (1, 2)]

    rendered = validation_module._stable_usd_info_value(value)

    assert rendered.startswith("builtins.list:[")
    assert "builtins.tuple:[builtins.int:1,builtins.int:2]" in rendered


def test_preflight_rejects_unsupported_plan_shapes() -> None:
    stage = _stage()
    plan = _plan()
    cases: list[tuple[JointRiggerPlanV1, str]] = [
        (plan.model_copy(update={"joints": ()}), "empty_topology"),
        (
            plan.model_copy(
                update={
                    "articulation_root": ArticulationRootPlanV1(
                        prim_path="/World",
                        provenance=_provenance("articulation", prim_path="/World"),
                    )
                }
            ),
            "physics_schema_fields_unsupported",
        ),
    ]
    unsupported_topology = plan.joints[0].topology.model_copy(
        update={"joint_type": "fixed"}
    )
    cases.append(
        (
            plan.model_copy(
                update={
                    "joints": (
                        plan.joints[0].model_copy(
                            update={"topology": unsupported_topology}
                        ),
                    )
                }
            ),
            "unsupported_joint_type",
        )
    )
    spherical = _plan(joint_type="spherical")
    cases.append(
        (
            spherical.model_copy(
                update={
                    "joints": (
                        spherical.joints[0].model_copy(update={"drive": _drive()}),
                    )
                }
            ),
            "unsupported_drive_instance",
        )
    )

    for invalid_plan, expected_code in cases:
        with pytest.raises(JointRiggerContractError) as error:
            validate_joint_topology_plan(stage, invalid_plan)
        _assert_code(error, expected_code)


def test_nested_joint_target_guard_rejects_parent_child_paths() -> None:
    with pytest.raises(JointRiggerContractError) as error:
        validation_module._reject_nested_joint_paths(
            (
                Sdf.Path("/World/Joints/parent"),
                Sdf.Path("/World/Joints/parent/child"),
            ),
            Sdf=Sdf,
        )

    _assert_code(error, "joint_target_collision")
    assert "must not be nested" in str(error.value)


def test_preflight_rejects_existing_non_joint_at_deterministic_target() -> None:
    stage = _stage()
    plan = _plan()
    UsdGeom.Scope.Define(stage, "/World/Joints")
    joint_path = validation_module._deterministic_joint_path(
        default_path=stage.GetDefaultPrim().GetPath(),
        joint=plan.joints[0],
        Sdf=Sdf,
        Tf=Tf,
    )
    UsdGeom.Xform.Define(stage, joint_path)

    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(stage, plan)

    _assert_code(error, "joint_target_collision")
    assert "refusing to overwrite" in str(error.value)


def test_preflight_rejects_joint_hidden_below_inactive_ancestor() -> None:
    stage = _stage()
    inactive = UsdGeom.Xform.Define(stage, "/World/Inactive")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Inactive/HiddenJoint")
    assert inactive.GetPrim().SetActive(False)
    assert not stage.GetPrimAtPath("/World/Inactive/HiddenJoint").IsValid()

    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(stage, _plan())

    _assert_code(error, "source_already_rigged")
    assert "HiddenJoint" in str(error.value)


def test_hidden_joint_spec_scan_uses_only_selected_variants() -> None:
    stage = _stage()
    variants = stage.GetDefaultPrim().GetVariantSets().AddVariantSet("Physics")
    variants.AddVariant("None")
    variants.AddVariant("Physics")
    variants.SetVariantSelection("Physics")
    with variants.GetVariantEditContext():
        UsdPhysics.RevoluteJoint.Define(stage, "/World/VariantJoint")

    variants.SetVariantSelection("None")
    validate_joint_topology_plan(stage, _plan())

    variants.SetVariantSelection("Physics")
    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(stage, _plan())

    _assert_code(error, "source_already_rigged")


def test_hidden_joint_scan_ignores_unreferenced_layer_specs() -> None:
    asset_stage = Usd.Stage.CreateInMemory()
    asset = UsdGeom.Xform.Define(asset_stage, "/Asset")
    asset_stage.SetDefaultPrim(asset.GetPrim())
    UsdGeom.Xform.Define(asset_stage, "/Asset/Used")
    UsdPhysics.RevoluteJoint.Define(asset_stage, "/UnusedJoint")

    stage = _stage()
    stage.GetDefaultPrim().GetReferences().AddReference(
        asset_stage.GetRootLayer().identifier,
        "/Asset",
    )
    root_before = stage.GetRootLayer().ExportToString()
    session_before = stage.GetSessionLayer().ExportToString()

    validate_joint_topology_plan(stage, _plan())
    assert stage.GetRootLayer().ExportToString() == root_before
    assert stage.GetSessionLayer().ExportToString() == session_before


def test_hidden_joint_scan_respects_referenced_variant_selection() -> None:
    asset_stage = Usd.Stage.CreateInMemory()
    asset = UsdGeom.Xform.Define(asset_stage, "/Asset")
    asset_stage.SetDefaultPrim(asset.GetPrim())
    variants = asset.GetPrim().GetVariantSets().AddVariantSet("Physics")
    variants.AddVariant("None")
    variants.AddVariant("Physics")
    variants.SetVariantSelection("Physics")
    with variants.GetVariantEditContext():
        UsdPhysics.RevoluteJoint.Define(asset_stage, "/Asset/VariantJoint")
    variants.SetVariantSelection("None")

    stage = _stage()
    stage.GetDefaultPrim().GetReferences().AddReference(
        asset_stage.GetRootLayer().identifier,
        "/Asset",
    )

    validate_joint_topology_plan(stage, _plan())

    composed_variants = stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics")
    assert composed_variants.SetVariantSelection("Physics")
    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(stage, _plan())

    _assert_code(error, "source_already_rigged")
    assert "/World/VariantJoint" in str(error.value)


def test_hidden_joint_scan_expands_inactive_subtree_in_native_instance() -> None:
    asset_stage = Usd.Stage.CreateInMemory()
    asset = UsdGeom.Xform.Define(asset_stage, "/Asset")
    asset_stage.SetDefaultPrim(asset.GetPrim())
    inactive = UsdGeom.Xform.Define(asset_stage, "/Asset/Inactive")
    UsdPhysics.RevoluteJoint.Define(asset_stage, "/Asset/Inactive/HiddenJoint")
    assert inactive.GetPrim().SetActive(False)

    stage = _stage()
    instance = UsdGeom.Xform.Define(stage, "/World/Instance").GetPrim()
    assert instance.GetReferences().AddReference(
        asset_stage.GetRootLayer().identifier,
        "/Asset",
    )
    assert instance.SetInstanceable(True)
    assert instance.IsInstance()

    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(stage, _plan())

    _assert_code(error, "source_already_rigged")
    assert "/World/Instance/Inactive/HiddenJoint" in str(error.value)


def test_hidden_joint_scan_preserves_session_variant_selection() -> None:
    stage = _stage()
    variants = stage.GetDefaultPrim().GetVariantSets().AddVariantSet("Physics")
    variants.AddVariant("None")
    variants.AddVariant("Physics")
    variants.SetVariantSelection("Physics")
    with variants.GetVariantEditContext():
        UsdPhysics.RevoluteJoint.Define(stage, "/World/SessionVariantJoint")
    stage.SetEditTarget(stage.GetSessionLayer())
    variants.SetVariantSelection("None")
    stage.SetEditTarget(stage.GetRootLayer())
    session_before = stage.GetSessionLayer().ExportToString()

    validate_joint_topology_plan(stage, _plan())

    assert variants.GetVariantSelection() == "None"
    assert stage.GetSessionLayer().ExportToString() == session_before


def test_hidden_joint_scan_respects_unloaded_payload() -> None:
    asset_stage = Usd.Stage.CreateInMemory()
    asset = UsdGeom.Xform.Define(asset_stage, "/Asset")
    asset_stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.RevoluteJoint.Define(asset_stage, "/Asset/PayloadJoint")

    stage = _stage()
    payload = UsdGeom.Xform.Define(stage, "/World/Payload").GetPrim()
    assert payload.GetPayloads().AddPayload(
        asset_stage.GetRootLayer().identifier,
        "/Asset",
    )
    assert stage.GetPrimAtPath("/World/Payload/PayloadJoint").IsValid()
    stage.Unload("/World/Payload")
    load_rules_before = str(stage.GetLoadRules())

    validate_joint_topology_plan(stage, _plan())

    assert not stage.GetPrimAtPath("/World/Payload/PayloadJoint").IsValid()
    assert str(stage.GetLoadRules()) == load_rules_before


def test_hidden_joint_scan_respects_population_mask() -> None:
    stage = _stage()
    UsdPhysics.RevoluteJoint.Define(stage, "/World/MaskedJoint")
    mask = Usd.StagePopulationMask()
    mask.Add(_BODY0)
    mask.Add(_BODY1)
    stage.SetPopulationMask(mask)
    assert not stage.GetPrimAtPath("/World/MaskedJoint").IsValid()

    validate_joint_topology_plan(stage, _plan())

    assert stage.GetPopulationMask() == mask


def test_hidden_joint_scan_never_opens_muted_layer(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    stage = _stage()
    missing = tmp_path / "muted-missing.usda"
    stage.GetRootLayer().subLayerPaths.append(str(missing))
    stage.MuteLayer(str(missing))
    inactive = UsdGeom.Xform.Define(stage, "/World/UnrelatedInactive")
    assert inactive.GetPrim().SetActive(False)
    muted_before = stage.GetMutedLayers()
    capfd.readouterr()

    validate_joint_topology_plan(stage, _plan())

    captured = capfd.readouterr()
    assert str(missing) not in captured.err
    assert stage.GetMutedLayers() == muted_before


def test_hidden_joint_scan_bounds_activation_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage()
    outer = UsdGeom.Xform.Define(stage, "/World/Outer")
    inner = UsdGeom.Xform.Define(stage, "/World/Outer/Inner")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Outer/Inner/HiddenJoint")
    assert inner.GetPrim().SetActive(False)
    assert outer.GetPrim().SetActive(False)
    monkeypatch.setattr(
        validation_module,
        "_INACTIVE_SCAN_MAX_ACTIVATION_ROUNDS",
        1,
    )

    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(stage, _plan())

    _assert_code(error, "source_joint_scan_limit_exceeded")
    assert "activation-round budget" in str(error.value)


def test_hidden_joint_scan_bounds_initial_prim_visits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage()
    inactive = UsdGeom.Xform.Define(stage, "/World/Inactive")
    assert inactive.GetPrim().SetActive(False)
    monkeypatch.setattr(validation_module, "_INACTIVE_SCAN_MAX_PRIM_VISITS", 1)

    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(stage, _plan())

    _assert_code(error, "source_joint_scan_limit_exceeded")
    assert "initial composed-prim work budget" in str(error.value)


def test_hidden_joint_scan_bounds_composed_prim_visits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage()
    inactive = UsdGeom.Xform.Define(stage, "/World/AInactive")
    assert inactive.GetPrim().SetActive(False)
    monkeypatch.setattr(validation_module, "_INACTIVE_SCAN_MAX_PRIM_VISITS", 3)

    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(stage, _plan())

    _assert_code(error, "source_joint_scan_limit_exceeded")
    assert "composed-prim work budget" in str(error.value)


def test_hidden_joint_scan_bounds_instance_expansion_visits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_stage = Usd.Stage.CreateInMemory()
    asset = UsdGeom.Xform.Define(asset_stage, "/Asset")
    asset_stage.SetDefaultPrim(asset.GetPrim())
    inactive = UsdGeom.Xform.Define(asset_stage, "/Asset/Inactive")
    assert inactive.GetPrim().SetActive(False)

    stage = _stage()
    instance = UsdGeom.Xform.Define(stage, "/World/Instance").GetPrim()
    assert instance.GetReferences().AddReference(
        asset_stage.GetRootLayer().identifier,
        "/Asset",
    )
    assert instance.SetInstanceable(True)
    monkeypatch.setattr(validation_module, "_INACTIVE_SCAN_MAX_PRIM_VISITS", 10)

    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(stage, _plan())

    _assert_code(error, "source_joint_scan_limit_exceeded")
    assert "instance-expansion work budget" in str(error.value)


class _RejectedScanOverride:
    def SetInstanceable(self, value: bool) -> bool:
        assert value is False
        return False

    def SetActive(self, value: bool) -> bool:
        assert value is True
        return False


class _RejectedScanStage:
    def OverridePrim(self, path: str) -> _RejectedScanOverride:
        assert path == "/World/Rejected"
        return _RejectedScanOverride()


@pytest.mark.parametrize(
    ("operation", "detail"),
    [
        (validation_module._require_scan_instance_expansion, "instance subtree"),
        (validation_module._require_scan_activation, "inactive composed subtree"),
    ],
)
def test_hidden_joint_scan_override_failures_are_stable(
    operation: Any,
    detail: str,
) -> None:
    with pytest.raises(JointRiggerContractError) as error:
        operation(_RejectedScanStage(), "/World/Rejected")

    _assert_code(error, "source_joint_scan_failed")
    assert detail in str(error.value)


def test_preflight_rejects_inexact_endpoint_path() -> None:
    plan = _plan()
    invalid_topology = plan.joints[0].topology.model_copy(
        update={"body1": "World/Link"}
    )
    invalid_plan = plan.model_copy(
        update={
            "joints": (
                plan.joints[0].model_copy(update={"topology": invalid_topology}),
            )
        }
    )

    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(_stage(), invalid_plan)

    _assert_code(error, "endpoint_missing")
    assert "exact absolute prim path" in str(error.value)


def test_preflight_rejects_real_instance_proxy_endpoint() -> None:
    stage = _stage()
    model = UsdGeom.Xform.Define(stage, "/Model")
    UsdGeom.Xform.Define(stage, "/Model/ProxyLink")
    instance = stage.OverridePrim("/World/Instance")
    assert instance.GetReferences().AddInternalReference(model.GetPath())
    assert instance.SetInstanceable(True)
    proxy_path = "/World/Instance/ProxyLink"
    assert stage.GetPrimAtPath(proxy_path).IsInstanceProxy()

    with pytest.raises(JointRiggerContractError) as error:
        validate_joint_topology_plan(stage, _plan(body1=proxy_path))

    _assert_code(error, "endpoint_instance_proxy")


def test_joint_frame_rejects_nonorthogonal_real_openusd_transform() -> None:
    transform = Gf.Matrix4d(
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    component = 1.0 / math.sqrt(2.0)
    stage_frame = validation_module._stage_joint_frame(
        Gf.Vec3d(component, component, 0.0),
        Gf=Gf,
    )

    with pytest.raises(JointRiggerContractError) as error:
        validation_module._local_joint_frame_rotation(
            transform,
            stage_frame=stage_frame,
            axis_token="X",
            label="scaled endpoint",
            Gf=Gf,
        )

    _assert_code(error, "unsupported_endpoint_joint_frame")
    assert "orthonormal joint frame" in str(error.value)


class _ExtractionDriftMatrix3d:
    """Return a clearly wrong rotation after receiving an exact identity frame."""

    def __init__(self, diagonal: float) -> None:
        assert diagonal == 1.0
        self._rows: dict[int, Any] = {}

    def SetRow(self, index: int, direction: Any) -> None:
        self._rows[index] = direction

    def ExtractRotation(self) -> Any:
        assert tuple(self._rows) == (0, 1, 2)
        assert tuple(tuple(self._rows[index]) for index in range(3)) == (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        return Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), 90.0)


class _ExtractionDriftGf:
    Matrix3d = _ExtractionDriftMatrix3d

    @staticmethod
    def Vec3d(*values: float) -> Any:
        return Gf.Vec3d(*values)

    @staticmethod
    def Dot(left: Any, right: Any) -> float:
        return float(Gf.Dot(left, right))

    @staticmethod
    def Cross(left: Any, right: Any) -> Any:
        return Gf.Cross(left, right)


def test_joint_frame_rejects_rotation_extraction_drift() -> None:
    stage_frame = (
        Gf.Vec3d(1.0, 0.0, 0.0),
        Gf.Vec3d(0.0, 1.0, 0.0),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    pair_dots = tuple(
        abs(float(Gf.Dot(stage_frame[left], stage_frame[right])))
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    handedness = float(Gf.Dot(Gf.Cross(stage_frame[0], stage_frame[1]), stage_frame[2]))
    assert pair_dots == (0.0, 0.0, 0.0)
    assert handedness == 1.0

    drifted_axis = Gf.Rotation(
        Gf.Vec3d(0.0, 0.0, 1.0),
        90.0,
    ).TransformDir(stage_frame[0])
    assert abs(float(Gf.Dot(drifted_axis, stage_frame[0]))) < 1e-12
    assert (drifted_axis - stage_frame[0]).GetLength() > 1.0

    with pytest.raises(JointRiggerContractError) as error:
        validation_module._local_joint_frame_rotation(
            Gf.Matrix4d(1.0),
            stage_frame=stage_frame,
            axis_token="X",
            label="extraction-drift endpoint",
            Gf=_ExtractionDriftGf,
        )

    _assert_code(error, "unsupported_endpoint_joint_frame")
    assert "could not represent shared joint frame direction 0" in str(error.value)


@pytest.mark.parametrize(
    "tamper",
    [
        "diagnostics_coverage",
        "wrong_joint_schema",
        "spherical_axis",
        "axis_token",
        "missing_local_rotation",
        "blocked_local_rotation",
        "inactive_joint",
        "undefined_joint",
        "joint_id",
        "plan_sha256",
        "diagnostics_path",
        "field_decisions",
        "unexpected_drive",
        "drive_instances",
        "drive_attribute",
        "blocked_drive_attribute",
        "limit_attribute",
        "blocked_limit_attribute",
        "raw_api_deleted",
        "raw_api_ordered",
        "unexpected_custom_data",
        "unexpected_metadata",
        "attribute_custom_data",
        "relationship_documentation",
    ],
)
def test_authored_graph_tampering_fails_closed(tamper: str) -> None:
    joint_type = "spherical" if tamper == "spherical_axis" else "revolute"
    drive = (
        _drive()
        if tamper
        in {
            "drive_instances",
            "drive_attribute",
            "blocked_drive_attribute",
            "raw_api_deleted",
            "raw_api_ordered",
        }
        else None
    )
    limit = (
        _limit() if tamper in {"limit_attribute", "blocked_limit_attribute"} else None
    )
    stage, plan, preflight, diagnostics = _authored_stage(
        joint_type=joint_type,
        drive=drive,
        limit=limit,
    )
    path = preflight.joints[0].joint_path
    prim = stage.GetPrimAtPath(path)
    joint = {
        "revolute": UsdPhysics.RevoluteJoint,
        "spherical": UsdPhysics.SphericalJoint,
    }[joint_type](prim)
    tampered_diagnostics = diagnostics

    if tamper == "diagnostics_coverage":
        tampered_diagnostics = diagnostics.model_copy(update={"joint_diagnostics": ()})
    elif tamper == "wrong_joint_schema":
        stage.RemovePrim(path)
        UsdPhysics.PrismaticJoint.Define(stage, path)
    elif tamper == "spherical_axis":
        assert joint.CreateAxisAttr().Set("X")
    elif tamper == "axis_token":
        assert joint.GetAxisAttr().Set("Y")
    elif tamper == "missing_local_rotation":
        assert prim.RemoveProperty(joint.GetLocalRot0Attr().GetName())
    elif tamper == "blocked_local_rotation":
        assert joint.GetLocalRot0Attr().Set(Sdf.ValueBlock())
        assert joint.GetLocalRot0Attr().HasAuthoredValueOpinion()
        assert joint.GetLocalRot0Attr().Get() is None
    elif tamper == "inactive_joint":
        assert prim.SetActive(False)
    elif tamper == "undefined_joint":
        spec = stage.GetRootLayer().GetPrimAtPath(path)
        assert spec is not None
        spec.specifier = Sdf.SpecifierOver
        assert not prim.IsDefined()
    elif tamper == "joint_id":
        prim.SetCustomDataByKey("jointRigger:jointId", "tampered")
    elif tamper == "plan_sha256":
        prim.SetCustomDataByKey("jointRigger:planSha256", "f" * 64)
    elif tamper == "diagnostics_path":
        original_diagnostic = diagnostics.joint_diagnostics[0]
        diagnostic = original_diagnostic.model_copy(
            update={
                "field_decisions": tuple(
                    decision.model_copy(update={"detail": "/World/Joints/tampered"})
                    if decision.field == "usd.joint_prim_path"
                    else decision
                    for decision in original_diagnostic.field_decisions
                )
            }
        )
        tampered_diagnostics = diagnostics.model_copy(
            update={"joint_diagnostics": (diagnostic,)}
        )
    elif tamper == "field_decisions":
        prim.SetCustomDataByKey("jointRigger:fieldDecisions", "[]")
    elif tamper == "unexpected_drive":
        assert UsdPhysics.DriveAPI.Apply(prim, "angular")
    elif tamper == "drive_instances":
        assert UsdPhysics.DriveAPI.Apply(prim, "linear")
    elif tamper == "drive_attribute":
        authored_drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        assert prim.RemoveProperty(authored_drive.GetStiffnessAttr().GetName())
    elif tamper == "blocked_drive_attribute":
        authored_drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        assert authored_drive.GetStiffnessAttr().Set(Sdf.ValueBlock())
        assert authored_drive.GetStiffnessAttr().Get() is None
    elif tamper == "limit_attribute":
        assert prim.RemoveProperty(joint.GetLowerLimitAttr().GetName())
    elif tamper == "blocked_limit_attribute":
        assert joint.GetLowerLimitAttr().Set(Sdf.ValueBlock())
        assert joint.GetLowerLimitAttr().Get() is None
    elif tamper in {
        "raw_api_deleted",
        "raw_api_ordered",
        "unexpected_custom_data",
        "unexpected_metadata",
        "attribute_custom_data",
        "relationship_documentation",
    }:
        _tamper_authored_joint_metadata(stage, path, tamper)
    else:  # pragma: no cover - parametrization and branch must stay in lockstep
        raise AssertionError(f"unhandled authored-graph tamper case: {tamper}")

    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_topology(stage, plan, tampered_diagnostics)

    _assert_code(error, "authored_graph_mismatch")


@pytest.mark.parametrize(
    ("joint_type", "drive"),
    [
        ("revolute", None),
        ("revolute", _drive()),
        (
            "revolute",
            _drive().model_copy(update={"max_joint_velocity": 4.0}),
        ),
        ("prismatic", _drive()),
        ("spherical", None),
    ],
)
def test_authored_graph_accepts_exact_author_owned_prim_metadata(
    joint_type: str,
    drive: JointDriveV1 | None,
) -> None:
    stage, plan, _, diagnostics = _authored_stage(
        joint_type=joint_type,
        drive=drive,
    )

    validate_authored_joint_topology(stage, plan, diagnostics)
    validate_authored_joint_topology(stage, plan)


def test_schema_fallback_custom_data_is_not_treated_as_authored_metadata() -> None:
    stage, plan, preflight, diagnostics = _authored_stage()
    joint_path = preflight.joints[0].joint_path
    prim = stage.GetPrimAtPath(joint_path)
    root_spec = stage.GetRootLayer().GetPrimAtPath(joint_path)
    assert root_spec is not None
    authored_custom_data = root_spec.GetInfo("customData")
    composed_custom_data = prim.GetCustomData()

    assert set(authored_custom_data) == {"jointRigger"}
    if "userDocBrief" not in composed_custom_data:
        pytest.skip("OpenUSD build does not expose joint schema fallback customData")
    assert set(composed_custom_data) - set(authored_custom_data) == {"userDocBrief"}

    validate_authored_joint_topology(stage, plan, diagnostics)

    prim.SetCustomDataByKey("unexpected:payload", "survived")
    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_topology(stage, plan, diagnostics)
    _assert_code(error, "authored_graph_mismatch")


@pytest.mark.parametrize(
    "tamper",
    [
        "raw_api_deleted",
        "raw_api_ordered",
        "unexpected_custom_data",
        "unexpected_metadata",
        "attribute_custom_data",
        "relationship_documentation",
    ],
)
def test_saved_authored_graph_rejects_unauthorized_raw_metadata(
    tmp_path: Path,
    tamper: str,
) -> None:
    stage, plan, preflight, diagnostics = _authored_stage(drive=_drive())
    _tamper_authored_joint_metadata(stage, preflight.joints[0].joint_path, tamper)
    output = tmp_path / f"raw-metadata-{tamper}.usda"
    assert stage.GetRootLayer().Export(str(output))
    del stage
    reopened = Usd.Stage.Open(str(output))
    assert reopened is not None

    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_topology(reopened, plan, diagnostics)

    _assert_code(error, "authored_graph_mismatch")


def test_authored_graph_rejects_multiple_joint_prim_specs() -> None:
    stage, plan, preflight, diagnostics = _authored_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    override = stage.OverridePrim(preflight.joints[0].joint_path)
    assert override.SetMetadata("documentation", "unexpected stronger PrimSpec")
    stage.SetEditTarget(stage.GetRootLayer())
    assert len(stage.GetPrimAtPath(preflight.joints[0].joint_path).GetPrimStack()) == 2

    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_topology(stage, plan, diagnostics)

    _assert_code(error, "authored_graph_mismatch")
    assert "exactly one authored PrimSpec" in str(error.value)


def test_raw_joint_type_and_property_list_ops_fail_closed() -> None:
    stage, _, preflight, _ = _authored_stage()
    expected = preflight.joints[0]
    prim = stage.GetPrimAtPath(expected.joint_path)
    spec = stage.GetRootLayer().GetPrimAtPath(expected.joint_path)
    assert spec is not None
    spec.typeName = "PhysicsJoint"

    with pytest.raises(JointRiggerContractError) as raw_type:
        validation_module._validate_joint_applied_schemas(
            prim,
            expected,
            Sdf=Sdf,
        )
    _assert_code(raw_type, "authored_graph_mismatch")
    assert "raw PrimSpec type" in str(raw_type.value)

    stage, _, preflight, _ = _authored_stage()
    expected = preflight.joints[0]
    prim = stage.GetPrimAtPath(expected.joint_path)
    joint = UsdPhysics.RevoluteJoint(prim)
    allowed = validation_module._joint_allowed_authored_properties(
        joint,
        prim,
        expected,
        UsdPhysics=UsdPhysics,
    )
    relationship = stage.GetRootLayer().GetPropertyAtPath(
        Sdf.Path(f"{expected.joint_path}.physics:body0")
    )
    assert isinstance(relationship, Sdf.RelationshipSpec)
    relationship.targetPathList.orderedItems = [Sdf.Path(_BODY0)]
    with pytest.raises(JointRiggerContractError) as raw_targets:
        validation_module._validate_joint_raw_property_specs(
            joint,
            prim,
            expected,
            allowed=allowed,
            Sdf=Sdf,
            UsdPhysics=UsdPhysics,
        )
    _assert_code(raw_targets, "authored_graph_mismatch")
    assert "raw target list-op" in str(raw_targets.value)

    extra = prim.CreateAttribute("fixture:extra", Sdf.ValueTypeNames.String)
    assert extra.Set("unexpected")
    with pytest.raises(JointRiggerContractError) as raw_property:
        validation_module._validate_joint_raw_property_specs(
            joint,
            prim,
            expected,
            allowed=allowed,
            Sdf=Sdf,
            UsdPhysics=UsdPhysics,
        )
    _assert_code(raw_property, "authored_graph_mismatch")
    assert "raw property specs" in str(raw_property.value)


@pytest.mark.parametrize(
    "value",
    [None, "not-json", "{}", '[{"field":"missing-required-fields"}]'],
)
def test_reportless_field_decision_metadata_rejects_invalid_shapes(value: Any) -> None:
    with pytest.raises(JointRiggerContractError) as error:
        validation_module._require_canonical_field_decisions(
            value,
            joint_path="/World/Joints/Test",
        )

    _assert_code(error, "authored_graph_mismatch")


def test_reportless_field_decision_metadata_requires_canonical_json() -> None:
    stage, _, preflight, _ = _authored_stage()
    value = stage.GetPrimAtPath(preflight.joints[0].joint_path).GetCustomDataByKey(
        "jointRigger:fieldDecisions"
    )
    assert isinstance(value, str)
    noncanonical = json.dumps(json.loads(value), indent=2, ensure_ascii=False)

    with pytest.raises(JointRiggerContractError) as error:
        validation_module._require_canonical_field_decisions(
            noncanonical,
            joint_path=preflight.joints[0].joint_path,
        )

    _assert_code(error, "authored_graph_mismatch")
    assert "not canonical" in str(error.value)


@pytest.mark.parametrize(
    "attribute_name",
    [
        "local_pos0",
        "local_pos1",
        "axis",
        "local_rot0",
        "local_rot1",
        "lower_limit",
        "upper_limit",
        "drive_type",
        "drive_stiffness",
        "drive_damping",
        "drive_max_force",
        "drive_target_position",
        "drive_target_velocity",
        "max_joint_velocity",
    ],
)
def test_authored_graph_rejects_time_samples_on_every_owned_attribute(
    attribute_name: str,
) -> None:
    drive = _drive().model_copy(update={"max_joint_velocity": 4.0})
    stage, plan, preflight, diagnostics = _authored_stage(
        drive=drive,
        limit=_limit(),
    )
    prim = stage.GetPrimAtPath(preflight.joints[0].joint_path)
    joint = UsdPhysics.RevoluteJoint(prim)
    authored_drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    attributes = {
        "local_pos0": joint.GetLocalPos0Attr(),
        "local_pos1": joint.GetLocalPos1Attr(),
        "axis": joint.GetAxisAttr(),
        "local_rot0": joint.GetLocalRot0Attr(),
        "local_rot1": joint.GetLocalRot1Attr(),
        "lower_limit": joint.GetLowerLimitAttr(),
        "upper_limit": joint.GetUpperLimitAttr(),
        "drive_type": authored_drive.GetTypeAttr(),
        "drive_stiffness": authored_drive.GetStiffnessAttr(),
        "drive_damping": authored_drive.GetDampingAttr(),
        "drive_max_force": authored_drive.GetMaxForceAttr(),
        "drive_target_position": authored_drive.GetTargetPositionAttr(),
        "drive_target_velocity": authored_drive.GetTargetVelocityAttr(),
        "max_joint_velocity": prim.GetAttribute("physxJoint:maxJointVelocity"),
    }
    attribute = attributes[attribute_name]
    default_value = attribute.Get()
    assert default_value is not None
    if attribute_name == "drive_stiffness":
        attribute.Clear()
        assert not attribute.HasAuthoredValueOpinion()
    assert attribute.Set(default_value, Usd.TimeCode(1.0))

    with pytest.raises(JointRiggerContractError) as error:
        validate_authored_joint_topology(stage, plan, diagnostics)

    _assert_code(error, "authored_graph_mismatch")
    assert str(attribute.GetName()) in str(error.value)
    assert "time samples" in str(error.value)


@pytest.mark.parametrize(
    "tamper",
    ["metadata", "prim_paths", "prim_facts", "raw_api_schemas"],
)
def test_no_reshape_detects_each_source_mutation_class(tamper: str) -> None:
    stage = _stage()
    preflight = validation_module._preflight_topology_authoring(stage, _plan())

    if tamper == "metadata":
        UsdGeom.SetStageMetersPerUnit(stage, 2.0)
    elif tamper == "prim_paths":
        UsdGeom.Xform.Define(stage, "/World/Unexpected")
    elif tamper == "prim_facts":
        attribute = stage.GetPrimAtPath(_BODY0).CreateAttribute(
            "fixture:changed",
            Sdf.ValueTypeNames.String,
        )
        assert attribute.Set("yes")
    elif tamper == "raw_api_schemas":
        prim = stage.GetPrimAtPath(_BODY0)
        assert prim.GetAppliedSchemas() == []
        assert prim.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.CreateExplicit(["VendorUnknownAPI"]),
        )
        assert prim.GetAppliedSchemas() == []
    else:  # pragma: no cover - parametrization and branch must stay in lockstep
        raise AssertionError(f"unhandled no-reshape tamper case: {tamper}")

    with pytest.raises(JointRiggerContractError) as error:
        validation_module._validate_no_reshape(stage, preflight)

    _assert_code(error, "no_reshape_violation")


@pytest.mark.parametrize("tamper", ["custom_data", "documentation"])
def test_no_reshape_detects_existing_prim_metadata_mutations(tamper: str) -> None:
    stage = _stage()
    body = stage.GetPrimAtPath(_BODY0)
    if tamper == "custom_data":
        body.SetCustomDataByKey("fixture:state", "before")
    else:
        assert body.SetMetadata("documentation", "before")
    preflight = validation_module._preflight_topology_authoring(stage, _plan())

    if tamper == "custom_data":
        body.SetCustomDataByKey("fixture:state", "after")
    else:
        assert body.SetMetadata("documentation", "after")

    with pytest.raises(JointRiggerContractError) as error:
        validation_module._validate_no_reshape(stage, preflight)

    _assert_code(error, "no_reshape_violation")


def test_no_reshape_allows_expected_new_joint_prims() -> None:
    stage = _stage()
    preflight = validation_module._preflight_topology_authoring(stage, _plan())

    UsdGeom.Scope.Define(stage, preflight.joints_scope_path)
    UsdPhysics.RevoluteJoint.Define(stage, preflight.joints[0].joint_path)

    validation_module._validate_no_reshape(stage, preflight)


def test_authored_validation_accepts_exact_new_joints_scope() -> None:
    stage = _stage()
    plan = _plan()
    preflight = validation_module._preflight_topology_authoring(stage, plan)
    assert preflight.create_joints_scope
    diagnostics = author_module._build_diagnostics(plan, preflight)

    # The original absent scope remains the correct post-rollback shape.
    validation_module._validate_no_reshape(stage, preflight)

    author_module._author_preflight(stage, preflight, diagnostics)
    validation_module._validate_authored_preflight(
        stage,
        preflight,
        diagnostics=diagnostics,
    )
    validation_module._validate_no_reshape(stage, preflight)


def test_saved_stage_validation_rejects_corrupted_new_joints_scope() -> None:
    stage = _stage()
    plan = _plan()
    source_preflight = validation_module._preflight_topology_authoring(stage, plan)
    diagnostics = author_module._build_diagnostics(plan, source_preflight)
    author_module._author_preflight(stage, source_preflight, diagnostics)
    validation_module._validate_authored_preflight(
        stage,
        source_preflight,
        diagnostics=diagnostics,
    )
    stage.GetPrimAtPath(source_preflight.joints_scope_path).SetCustomDataByKey(
        "unexpected:postSavePayload",
        "survived",
    )

    with pytest.raises(JointRiggerContractError) as error:
        author_module._validate_authored_saved_stage(
            stage,
            plan,
            diagnostics,
            source_preflight=source_preflight,
        )

    _assert_code(error, "no_reshape_violation")
    assert "exact author-owned UsdGeom.Scope" in str(error.value)


@pytest.mark.parametrize(
    "tamper",
    [
        "wrong_type",
        "custom_data",
        "property",
        "time_sample",
        "metadata",
        "api_schema",
        "instanceable",
    ],
)
def test_authored_validation_rejects_corrupted_new_joints_scope(
    tamper: str,
) -> None:
    stage = _stage()
    plan = _plan()
    preflight = validation_module._preflight_topology_authoring(stage, plan)
    assert preflight.create_joints_scope
    diagnostics = author_module._build_diagnostics(plan, preflight)
    author_module._author_preflight(stage, preflight, diagnostics)
    prim = stage.GetPrimAtPath(preflight.joints_scope_path)

    if tamper == "wrong_type":
        assert prim.SetTypeName("Xform")
    elif tamper == "custom_data":
        prim.SetCustomDataByKey("unexpected:payload", "survived")
    elif tamper in {"property", "time_sample"}:
        attribute = prim.CreateAttribute(
            "unexpected:value",
            Sdf.ValueTypeNames.String,
        )
        if tamper == "time_sample":
            assert attribute.Set("survived", Usd.TimeCode(1.0))
        else:
            assert attribute.Set("survived")
    elif tamper == "metadata":
        assert prim.SetMetadata("documentation", "unexpected scope metadata")
    elif tamper == "api_schema":
        assert prim.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.CreateExplicit(["VendorUnknownAPI"]),
        )
    elif tamper == "instanceable":
        assert prim.SetInstanceable(True)
    else:  # pragma: no cover - parametrization and branch stay in lockstep
        raise AssertionError(tamper)

    with pytest.raises(JointRiggerContractError) as authored_error:
        validation_module._validate_authored_preflight(
            stage,
            preflight,
            diagnostics=diagnostics,
        )

    _assert_code(authored_error, "authored_graph_mismatch")
    assert preflight.joints_scope_path in str(authored_error.value)
    assert "exact author-owned UsdGeom.Scope" in str(authored_error.value)

    with pytest.raises(JointRiggerContractError) as reshape_error:
        validation_module._validate_no_reshape(stage, preflight)

    _assert_code(reshape_error, "no_reshape_violation")
    assert preflight.joints_scope_path in str(reshape_error.value)
    assert "exact author-owned UsdGeom.Scope" in str(reshape_error.value)


def test_no_reshape_allows_only_planned_ancestor_inert_overs() -> None:
    stage = _stage_with_composed_source_prims()
    preflight = validation_module._preflight_topology_authoring(stage, _plan())
    assert not preflight.create_joints_scope

    UsdPhysics.RevoluteJoint.Define(stage, preflight.joints[0].joint_path)
    world_spec = stage.GetRootLayer().GetPrimAtPath("/World")
    assert world_spec is not None
    assert world_spec.ListInfoKeys() == ["specifier"]
    assert world_spec.specifier == Sdf.SpecifierOver
    scope_spec = stage.GetRootLayer().GetPrimAtPath(preflight.joints_scope_path)
    assert scope_spec is not None
    assert scope_spec.ListInfoKeys() == ["specifier"]
    assert scope_spec.specifier == Sdf.SpecifierOver

    validation_module._validate_no_reshape(stage, preflight)

    unrelated_spec = Sdf.CreatePrimInLayer(
        stage.GetRootLayer(),
        "/World/Unrelated",
    )
    assert unrelated_spec.ListInfoKeys() == ["specifier"]
    assert unrelated_spec.specifier == Sdf.SpecifierOver

    with pytest.raises(JointRiggerContractError) as error:
        validation_module._validate_no_reshape(stage, preflight)
    _assert_code(error, "no_reshape_violation")
    assert "/World/Unrelated" in str(error.value)


@pytest.mark.parametrize(
    "tamper",
    [
        "visibility_value",
        "display_color_value",
        "attribute_time_sample",
        "attribute_metadata",
        "material_binding_target",
        "material_binding_list_op",
    ],
)
def test_no_reshape_detects_existing_property_opinion_mutations(tamper: str) -> None:
    stage = _stage()
    body = stage.GetPrimAtPath(_BODY0)
    mutate: Any
    if tamper == "visibility_value":
        attribute = UsdGeom.Imageable(body).CreateVisibilityAttr()
        assert attribute.Set(UsdGeom.Tokens.inherited)

        def mutate() -> None:
            assert attribute.Set(UsdGeom.Tokens.invisible)

    elif tamper == "display_color_value":
        cube = UsdGeom.Cube.Define(stage, f"{_BODY0}/Visual")
        attribute = UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr()
        assert attribute.Set([Gf.Vec3f(0.1, 0.2, 0.3)])

        def mutate() -> None:
            assert attribute.Set([Gf.Vec3f(0.7, 0.8, 0.9)])

    elif tamper == "attribute_time_sample":
        attribute = body.CreateAttribute("fixture:animated", Sdf.ValueTypeNames.Float)
        assert attribute.Set(1.0, Usd.TimeCode(1.0))

        def mutate() -> None:
            assert attribute.Set(2.0, Usd.TimeCode(1.0))

    elif tamper == "attribute_metadata":
        attribute = body.CreateAttribute("fixture:metadata", Sdf.ValueTypeNames.String)
        assert attribute.Set("stable")
        assert attribute.SetMetadata("documentation", "before")

        def mutate() -> None:
            assert attribute.SetMetadata("documentation", "after")

    else:
        UsdGeom.Scope.Define(stage, "/World/Looks")
        UsdGeom.Scope.Define(stage, "/World/Looks/One")
        UsdGeom.Scope.Define(stage, "/World/Looks/Two")
        relationship = body.CreateRelationship("material:binding")
        assert relationship.SetTargets([Sdf.Path("/World/Looks/One")])
        if tamper == "material_binding_target":

            def mutate() -> None:
                assert relationship.SetTargets([Sdf.Path("/World/Looks/Two")])

        elif tamper == "material_binding_list_op":

            def mutate() -> None:
                spec = stage.GetRootLayer().GetPropertyAtPath(
                    Sdf.Path(f"{_BODY0}.material:binding")
                )
                assert isinstance(spec, Sdf.RelationshipSpec)
                spec.targetPathList.orderedItems = [Sdf.Path("/World/Looks/One")]

        else:  # pragma: no cover - parametrization invariant
            raise AssertionError(tamper)

    preflight = validation_module._preflight_topology_authoring(stage, _plan())
    mutate()

    with pytest.raises(JointRiggerContractError) as error:
        validation_module._validate_no_reshape(stage, preflight)

    _assert_code(error, "no_reshape_violation")


def test_vector_guards_reject_degenerate_and_mismatched_values() -> None:
    with pytest.raises(JointRiggerContractError) as mismatch:
        validation_module._require_close_vector(
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 1.0, 0.0),
            label="orthogonal directions",
            normalized=True,
        )
    _assert_code(mismatch, "authored_graph_mismatch")

    with pytest.raises(JointRiggerContractError) as direction:
        validation_module._normalized_direction(
            Gf.Vec3d(0.0, 0.0, 0.0),
            label="zero direction",
        )
    _assert_code(direction, "singular_endpoint_transform")

    with pytest.raises(JointRiggerContractError) as vector_tuple:
        validation_module._normalized_tuple(
            (0.0, 0.0, 0.0),
            label="zero tuple",
        )
    _assert_code(vector_tuple, "authored_graph_mismatch")


def test_optional_prismatic_conversion_none_and_overflow_paths() -> None:
    assert validation_module._optional_divide(None, 0.01) is None
    assert not validation_module._anchor_vectors_close(
        (float("inf"), 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    for value, divisor in (
        (10**400, 0.01),
        (1e300, 1e-300),
        (1e-300, 1e300),
    ):
        with pytest.raises(JointRiggerContractError) as error:
            validation_module._optional_divide(value, divisor)

        _assert_code(error, "authored_value_out_of_range")
        assert "not representable" in str(error.value)

    with pytest.raises(JointRiggerContractError) as float32_error:
        validation_module._float32_round_trip(1e40, label="oversized drive value")
    _assert_code(float32_error, "authored_value_out_of_range")


def test_additional_schema_contract_requires_complete_canonical_order() -> None:
    stage, _, preflight, _ = _authored_stage()
    expected = preflight.joints[0]
    prim = stage.GetPrimAtPath(expected.joint_path)
    additional = frozenset({"PhysicsArticulationRootAPI"})

    with pytest.raises(JointRiggerContractError) as missing_order:
        validation_module._validate_joint_applied_schemas(
            prim,
            expected,
            Sdf=Sdf,
            additional_allowed=additional,
        )
    _assert_code(missing_order, "authored_graph_mismatch")
    assert "lack a canonical raw ordering contract" in str(missing_order.value)

    with pytest.raises(JointRiggerContractError) as incomplete_order:
        validation_module._validate_joint_applied_schemas(
            prim,
            expected,
            Sdf=Sdf,
            additional_allowed=additional,
            additional_expected_order=(),
        )
    _assert_code(incomplete_order, "authored_graph_mismatch")
    assert "does not exactly cover the allowed schemas" in str(incomplete_order.value)


def test_additional_raw_property_contracts_fail_closed() -> None:
    stage, _, preflight, _ = _authored_stage()
    expected = preflight.joints[0]
    prim = stage.GetPrimAtPath(expected.joint_path)
    joint = UsdPhysics.RevoluteJoint(prim)
    allowed = validation_module._joint_allowed_authored_properties(
        joint,
        prim,
        expected,
        UsdPhysics=UsdPhysics,
    )

    with pytest.raises(JointRiggerContractError) as relationship_overlap:
        validation_module._validate_joint_raw_property_specs(
            joint,
            prim,
            expected,
            allowed=allowed,
            Sdf=Sdf,
            UsdPhysics=UsdPhysics,
            additional_expected_relationship_targets={
                "physics:body0": expected.source.topology.body0
            },
        )
    _assert_code(relationship_overlap, "authored_graph_mismatch")
    assert "overlaps topology-owned relationships" in str(relationship_overlap.value)

    with pytest.raises(JointRiggerContractError) as attribute_overlap:
        validation_module._validate_joint_raw_property_specs(
            joint,
            prim,
            expected,
            allowed=allowed,
            Sdf=Sdf,
            UsdPhysics=UsdPhysics,
            additional_expected_attribute_specs={
                "physics:localPos0": ("float", "varying")
            },
        )
    _assert_code(attribute_overlap, "authored_graph_mismatch")
    assert "overlaps topology-owned attributes" in str(attribute_overlap.value)

    extra_name = "fixture:extra"
    assert prim.CreateAttribute(extra_name, Sdf.ValueTypeNames.String).Set("value")
    allowed_with_extra = allowed | {extra_name}
    with pytest.raises(JointRiggerContractError) as unsupported_spec:
        validation_module._validate_joint_raw_property_specs(
            joint,
            prim,
            expected,
            allowed=allowed_with_extra,
            Sdf=Sdf,
            UsdPhysics=UsdPhysics,
            additional_expected_attribute_specs={
                extra_name: ("unsupported", "varying")
            },
        )
    _assert_code(unsupported_spec, "authored_graph_mismatch")
    assert "unsupported raw specification contract" in str(unsupported_spec.value)

    with pytest.raises(JointRiggerContractError) as incomplete_allowlist:
        validation_module._validate_joint_raw_property_specs(
            joint,
            prim,
            expected,
            allowed=allowed_with_extra,
            Sdf=Sdf,
            UsdPhysics=UsdPhysics,
        )
    _assert_code(incomplete_allowlist, "authored_graph_mismatch")
    assert "does not exactly cover the authored allowlist" in str(
        incomplete_allowlist.value
    )


def test_optional_metadata_and_snapshot_fallbacks() -> None:
    class _SequenceMetadataPrim:
        def GetAppliedSchemas(self) -> tuple[str, ...]:
            return ("PhysicsRigidBodyAPI",)

        def GetMetadata(self, name: str) -> list[str]:
            assert name == "apiSchemas"
            return ["PhysxRigidBodyAPI"]

    class _MissingRelationshipPrim:
        def GetRelationship(self, name: str) -> None:
            assert name == "fixture:missing"
            return None

    class _OpaqueValue:
        def __str__(self) -> str:
            return "opaque-value"

    assert validation_module._applied_schema_tokens(_SequenceMetadataPrim()) == {
        "PhysicsRigidBodyAPI",
        "PhysxRigidBodyAPI",
    }
    assert (
        validation_module._relationship_targets(
            _MissingRelationshipPrim(), "fixture:missing"
        )
        == ()
    )
    assert validation_module._snapshot_value(_OpaqueValue()) == "opaque-value"


@pytest.mark.parametrize(
    ("operation", "expected_code"),
    [
        (
            validation_module.capture_joint_rigger_stage_snapshot,
            "stage_snapshot_scan_limit_exceeded",
        ),
        (
            validation_module.capture_joint_rigger_physics_schema_snapshot,
            "stage_snapshot_scan_limit_exceeded",
        ),
        (
            validation_module.physics_schema_counts,
            "physics_schema_count_scan_limit_exceeded",
        ),
    ],
    ids=["stage-snapshot", "physics-snapshot", "physics-counts"],
)
def test_public_stage_scans_have_fixed_streaming_visit_limit(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
    expected_code: str,
) -> None:
    stage = _stage()
    before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(
        validation_module,
        "_STAGE_SNAPSHOT_MAX_PRIM_VISITS",
        1,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        operation(stage)

    _assert_code(caught, expected_code)
    assert "fixed 1-prim visit limit" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_private_topology_snapshot_has_same_fixed_visit_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage()
    before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(
        validation_module,
        "_STAGE_SNAPSHOT_MAX_PRIM_VISITS",
        1,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        validate_joint_topology_plan(stage, _plan())

    _assert_code(caught, "stage_snapshot_scan_limit_exceeded")
    assert stage.GetRootLayer().ExportToString() == before


def test_existing_joint_scan_shares_visit_limit_with_prototypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage()
    prototype_source = UsdGeom.Xform.Define(stage, "/Library")
    UsdGeom.Xform.Define(stage, "/Library/Child")
    assert prototype_source.GetPrim().SetInstanceable(True)
    instance = stage.OverridePrim("/World/Instance")
    assert instance.GetReferences().AddInternalReference("/Library")
    assert stage.GetPrototypes()
    stage_visit_count = len(
        list(
            Usd.PrimRange.Stage(
                stage,
                Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
            )
        )
    )
    before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(
        validation_module,
        "_SOURCE_JOINT_SCAN_MAX_PRIM_VISITS",
        stage_visit_count,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        validate_joint_topology_plan(stage, _plan())

    _assert_code(caught, "source_joint_scan_limit_exceeded")
    assert "prototype scan" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_existing_joint_inactive_scan_uses_same_path_retention_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage()
    for index in range(2):
        inactive = UsdGeom.Xform.Define(stage, f"/World/Inactive_{index}")
        UsdPhysics.FixedJoint.Define(
            stage,
            f"/World/Inactive_{index}/HiddenJoint",
        )
        assert inactive.GetPrim().SetActive(False)
    before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(
        validation_module,
        "_SOURCE_JOINT_SCAN_MAX_PATHS",
        1,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        validate_joint_topology_plan(stage, _plan())

    _assert_code(caught, "source_joint_path_limit_exceeded")
    assert "fixed 1-path retention limit" in caught.value.detail
    assert stage.GetRootLayer().ExportToString() == before


def test_existing_joint_scan_preserves_unrelated_contract_error_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage()
    injected = JointRiggerContractError(
        "unrelated_scan_failure",
        "injected inactive joint scan failure",
    )

    def fail_scan(*_args: Any, **_kwargs: Any) -> set[str]:
        raise injected

    monkeypatch.setattr(
        validation_module,
        "_paths_with_inactive_ancestors_enabled",
        fail_scan,
    )

    with pytest.raises(JointRiggerContractError) as caught:
        validate_joint_topology_plan(stage, _plan())

    assert caught.value is injected
