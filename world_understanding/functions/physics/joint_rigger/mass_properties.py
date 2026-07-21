# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rigid-frame helpers for evidence-backed Joint Rigger mass properties."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

from world_understanding.functions.physics.joint_rigger.models import (
    JointRiggerContractError,
)

type _Vector3 = tuple[float, float, float]
type _Quaternion = tuple[float, float, float, float]

_RIGID_FRAME_TOLERANCE = 1e-8
_QUATERNION_ZERO_TOLERANCE = 1e-15


@dataclass(frozen=True)
class _LiftedMassFrame:
    """One descendant mass frame expressed in its rigid-body owner frame."""

    center_of_mass_m: _Vector3
    principal_axes: _Quaternion
    derivation_receipt: str


def _canonicalize_quaternion(
    value: tuple[float, float, float, float],
    *,
    label: str,
) -> _Quaternion:
    """Normalize one quaternion and choose a deterministic equivalent sign."""

    components = tuple(float(component) for component in value)
    if any(not math.isfinite(component) for component in components):
        _fail("invalid_mass_properties", f"{label} contains non-finite values")
    norm = math.sqrt(sum(component * component for component in components))
    if not math.isfinite(norm) or norm <= _QUATERNION_ZERO_TOLERANCE:
        _fail("invalid_mass_properties", f"{label} has zero quaternion norm")
    normalized = tuple(component / norm for component in components)
    sign = 1.0
    for component in normalized:
        if abs(component) <= _QUATERNION_ZERO_TOLERANCE:
            continue
        sign = 1.0 if component > 0.0 else -1.0
        break
    canonical = tuple(_canonical_zero(sign * component) for component in normalized)
    return cast(_Quaternion, canonical)


def _lift_descendant_mass_frame(
    contributor_prim: Any,
    owner_prim: Any,
    *,
    center_of_mass_m: _Vector3,
    principal_axes: _Quaternion,
    meters_per_unit: float,
) -> _LiftedMassFrame:
    """Express descendant SI center of mass and inertia axes in the owner frame."""

    from pxr import Gf, Usd, UsdGeom

    contributor_path = str(contributor_prim.GetPath())
    owner_path = str(owner_prim.GetPath())
    length_scale = float(meters_per_unit)
    if not math.isfinite(length_scale) or length_scale <= 0.0:
        _fail(
            "invalid_stage_units",
            f"mass frame lift for {contributor_path} requires a finite positive "
            f"metersPerUnit; got {meters_per_unit!r}",
        )
    _require_descendant_path(
        contributor_prim,
        owner_prim,
        contributor_path=contributor_path,
        owner_path=owner_path,
    )
    _require_static_noninstance_chain(
        contributor_prim,
        owner_prim,
        contributor_path=contributor_path,
        owner_path=owner_path,
        UsdGeom=UsdGeom,
    )

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    contributor_world = cache.GetLocalToWorldTransform(contributor_prim)
    owner_world = cache.GetLocalToWorldTransform(owner_prim)
    # Gf uses row vectors, so child world is child-local * parent world.
    relative = contributor_world * owner_world.GetInverse()
    basis = _require_rigid_matrix(
        relative,
        contributor_path=contributor_path,
        owner_path=owner_path,
        Gf=Gf,
    )
    relative_rotation = _rotation_from_basis(basis, Gf=Gf)
    relative_quaternion = _rotation_quaternion(
        relative_rotation,
        label=f"{contributor_path} to {owner_path} rotation",
    )

    translation_stage = cast(
        _Vector3,
        tuple(
            _finite_float(
                item,
                label=f"{contributor_path} to {owner_path} translation_stage[{index}]",
            )
            for index, item in enumerate(relative.Transform(Gf.Vec3d(0.0, 0.0, 0.0)))
        ),
    )
    translation_m = cast(
        _Vector3,
        tuple(
            _finite_float(
                value * length_scale,
                label=f"{contributor_path} to {owner_path} translation_m[{index}]",
            )
            for index, value in enumerate(translation_stage)
        ),
    )
    rotated_center_m = relative.TransformDir(Gf.Vec3d(*center_of_mass_m))
    lifted_center_m = cast(
        _Vector3,
        tuple(
            _finite_float(
                float(rotated_center_m[index]) + translation_m[index],
                label=f"{contributor_path} lifted center_of_mass_m[{index}]",
            )
            for index in range(3)
        ),
    )

    canonical_principal = _canonicalize_quaternion(
        principal_axes,
        label=f"{contributor_path} physics:principalAxes",
    )
    principal_rotation = Gf.Rotation(
        Gf.Quatd(
            canonical_principal[0],
            Gf.Vec3d(*canonical_principal[1:]),
        )
    )
    lifted_basis = tuple(
        relative.TransformDir(principal_rotation.TransformDir(direction))
        for direction in _basis_vectors(Gf=Gf)
    )
    lifted_rotation = _rotation_from_basis(lifted_basis, Gf=Gf)
    lifted_principal = _rotation_quaternion(
        lifted_rotation,
        label=f"{contributor_path} lifted physics:principalAxes",
    )
    receipt = (
        "rigid_body_frame_lift("
        f"contributor={contributor_path}, owner={owner_path}, "
        f"translation_stage={_format_tuple(translation_stage)}, "
        f"translation_m={_format_tuple(translation_m)}, "
        f"rotation_wxyz={_format_tuple(relative_quaternion)}; "
        "centerOfMass_owner_m=R*centerOfMass_contributor_m+"
        "translation_stage*metersPerUnit; "
        "principalAxes_owner=R*principalAxes_contributor; "
        "diagonalInertia_eigenvalues_unchanged)"
    )
    return _LiftedMassFrame(
        center_of_mass_m=lifted_center_m,
        principal_axes=lifted_principal,
        derivation_receipt=receipt,
    )


def _require_descendant_path(
    contributor_prim: Any,
    owner_prim: Any,
    *,
    contributor_path: str,
    owner_path: str,
) -> None:
    contributor = contributor_prim.GetPath()
    owner = owner_prim.GetPath()
    if contributor == owner or not contributor.HasPrefix(owner):
        _fail(
            "descendant_mass_owner_mismatch",
            f"mass contributor {contributor_path} is not below owner {owner_path}",
        )


def _require_static_noninstance_chain(
    contributor_prim: Any,
    owner_prim: Any,
    *,
    contributor_path: str,
    owner_path: str,
    UsdGeom: Any,
) -> None:
    current = contributor_prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        current_path = str(current.GetPath())
        if (
            current.IsInstance()
            or current.IsInstanceProxy()
            or current.IsInstanceable()
        ):
            _fail(
                "descendant_mass_instance_unsupported",
                f"mass contributor {contributor_path} uses an instance at "
                f"{current_path} below owner {owner_path}",
            )
        xformable = UsdGeom.Xformable(current)
        if xformable:
            if current != owner_prim and xformable.GetResetXformStack():
                _fail(
                    "descendant_mass_reset_xform_unsupported",
                    f"mass contributor {contributor_path} resets its transform "
                    f"stack at {current_path} below owner {owner_path}",
                )
            attributes = (
                xformable.GetXformOpOrderAttr(),
                *(operation.GetAttr() for operation in xformable.GetOrderedXformOps()),
            )
            for attribute in attributes:
                if attribute.HasAuthoredConnections():
                    _fail(
                        "descendant_mass_transform_connected",
                        f"mass contributor {contributor_path} has connected "
                        f"transform {attribute.GetName()} at {current_path}",
                    )
                samples, has_spline, might_vary = _transform_time_evidence(
                    attribute,
                    contributor_path=contributor_path,
                    current_path=current_path,
                )
                if samples or has_spline or might_vary:
                    _fail(
                        "descendant_mass_transform_time_sampled",
                        f"mass contributor {contributor_path} has time-varying "
                        f"transform {attribute.GetName()} at {current_path}: "
                        f"samples={samples}, spline={has_spline}, "
                        f"value_might_vary={might_vary}",
                    )
        if current == owner_prim:
            return
        current = current.GetParent()
    _fail(
        "descendant_mass_owner_mismatch",
        f"mass contributor {contributor_path} is not below owner {owner_path}",
    )


def _transform_time_evidence(
    attribute: Any,
    *,
    contributor_path: str,
    current_path: str,
) -> tuple[tuple[float, ...], bool, bool]:
    """Inspect composed and raw attribute opinions for temporal behavior."""

    try:
        samples = {float(value) for value in attribute.GetTimeSamples()}
        has_spline = False
        composed_has_spline = getattr(attribute, "HasSpline", None)
        if callable(composed_has_spline):
            has_spline = bool(composed_has_spline())
        for property_spec in attribute.GetPropertyStack():
            list_info_keys = getattr(property_spec, "ListInfoKeys", None)
            if callable(list_info_keys):
                has_spline = has_spline or "spline" in {
                    str(key) for key in list_info_keys()
                }
            layer = getattr(property_spec, "layer", None)
            path = getattr(property_spec, "path", None)
            if layer is not None and path is not None:
                samples.update(
                    float(value) for value in layer.ListTimeSamplesForPath(path)
                )
        value_might_vary = getattr(attribute, "ValueMightBeTimeVarying", None)
        might_vary = bool(value_might_vary()) if callable(value_might_vary) else False
    except Exception as exc:
        raise JointRiggerContractError(
            "descendant_mass_transform_time_unresolved",
            f"mass contributor {contributor_path} cannot prove transform "
            f"{attribute.GetName()} at {current_path} is static: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    return tuple(sorted(samples)), has_spline, might_vary


def _require_rigid_matrix(
    matrix: Any,
    *,
    contributor_path: str,
    owner_path: str,
    Gf: Any,
) -> tuple[Any, Any, Any]:
    values = tuple(
        float(matrix[row][column]) for row in range(4) for column in range(4)
    )
    if any(not math.isfinite(value) for value in values):
        _fail(
            "descendant_mass_transform_invalid",
            f"mass contributor {contributor_path} has a non-finite transform "
            f"relative to {owner_path}",
        )
    if any(
        not math.isclose(
            float(matrix[row][3]),
            0.0,
            rel_tol=0.0,
            abs_tol=_RIGID_FRAME_TOLERANCE,
        )
        for row in range(3)
    ) or not math.isclose(
        float(matrix[3][3]),
        1.0,
        rel_tol=0.0,
        abs_tol=_RIGID_FRAME_TOLERANCE,
    ):
        _fail(
            "descendant_mass_transform_not_rigid",
            f"mass contributor {contributor_path} has a projective transform "
            f"relative to {owner_path}",
        )
    basis = tuple(matrix.TransformDir(item) for item in _basis_vectors(Gf=Gf))
    lengths = tuple(float(item.GetLength()) for item in basis)
    dots = (
        float(Gf.Dot(basis[0], basis[1])),
        float(Gf.Dot(basis[0], basis[2])),
        float(Gf.Dot(basis[1], basis[2])),
    )
    handedness = float(Gf.Dot(Gf.Cross(basis[0], basis[1]), basis[2]))
    if (
        any(
            not math.isclose(
                length,
                1.0,
                rel_tol=0.0,
                abs_tol=_RIGID_FRAME_TOLERANCE,
            )
            for length in lengths
        )
        or any(
            not math.isclose(
                dot,
                0.0,
                rel_tol=0.0,
                abs_tol=_RIGID_FRAME_TOLERANCE,
            )
            for dot in dots
        )
        or not math.isclose(
            handedness,
            1.0,
            rel_tol=0.0,
            abs_tol=_RIGID_FRAME_TOLERANCE,
        )
    ):
        _fail(
            "descendant_mass_transform_not_rigid",
            f"mass contributor {contributor_path} has scale, shear, or reflection "
            f"relative to {owner_path}; lengths={lengths!r}, dots={dots!r}, "
            f"handedness={handedness!r}",
        )
    return cast(tuple[Any, Any, Any], basis)


def _basis_vectors(*, Gf: Any) -> tuple[Any, Any, Any]:
    return (
        Gf.Vec3d(1.0, 0.0, 0.0),
        Gf.Vec3d(0.0, 1.0, 0.0),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )


def _rotation_from_basis(basis: tuple[Any, Any, Any], *, Gf: Any) -> Any:
    matrix = Gf.Matrix3d(1.0)
    for index, direction in enumerate(basis):
        matrix.SetRow(index, direction)
    return matrix.ExtractRotation()


def _rotation_quaternion(rotation: Any, *, label: str) -> _Quaternion:
    quaternion = rotation.GetQuat()
    imaginary = quaternion.GetImaginary()
    return _canonicalize_quaternion(
        (
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
        label=label,
    )


def _finite_float(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        _fail(
            "descendant_mass_transform_invalid",
            f"mass frame conversion produced a non-finite value for {label}: "
            f"{result!r}",
        )
    return _canonical_zero(result)


def _canonical_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _format_tuple(value: tuple[float, ...]) -> str:
    return "(" + ", ".join(format(item, ".17g") for item in value) + ")"


def _fail(code: str, detail: str) -> None:
    raise JointRiggerContractError(code, detail)


__all__: list[str] = []
