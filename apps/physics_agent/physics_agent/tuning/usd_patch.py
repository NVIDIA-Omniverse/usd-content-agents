# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply tuned parameter values onto a simulation-ready USD.

The output of ``apply_physics`` carries authored ``UsdPhysics`` schemas and
backend-specific attributes:

* :class:`UsdPhysics.MassAPI` — mass / density per rigid body.
* :class:`UsdPhysics.MaterialAPI` — static / dynamic friction + restitution
  per physics material.
* :class:`UsdPhysics.CollisionAPI` — optional backend contact attributes such
  as ``newton:contact_ke`` / ``newton:contact_kd`` on collision prims.

Resolved tuning bindings drive the exact USD prims and attributes to update.
The legacy no-bindings path still supports mass scaling and material
friction/restitution overrides. The result is exported as a flattened
``.usda`` so downstream consumers see one self-contained file.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def patch_physics_usd(
    input_usd: Path,
    output_usd: Path,
    tuned_params: dict[str, float],
    *,
    bindings: list[dict[str, Any]] | None = None,
) -> Path:
    """Apply tuned parameters to ``input_usd`` and write a derivative USD.

    Args:
        input_usd: Path to a USD file authored by ``apply_physics``. Must
            already contain :class:`UsdPhysics.MassAPI` and/or
            :class:`UsdPhysics.MaterialAPI` schemas — this function does NOT
            create new schemas, it only overrides authored values.
        output_usd: Output ``.usda`` path. Parent directories are created.
        tuned_params: Mapping of supported parameter name → value. Without
            ``bindings``, recognised keys are ``mass_scale``,
            ``static_friction``, ``dynamic_friction``, ``restitution``. Backend
            contact params such as ``contact_ke`` and ``contact_kd`` require
            resolved bindings.
        bindings: Optional resolved parameter bindings produced by
            ``scenario_resolution``. When present, values are applied through
            those bindings instead of the legacy param-name switch.

    Returns:
        Absolute path to the written USD.

    Raises:
        FileNotFoundError: when ``input_usd`` does not exist.
        RuntimeError: when ``input_usd`` cannot be opened as a USD stage.
    """
    input_path = Path(input_usd)
    output_path = Path(output_usd).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input USD not found: {input_path}")

    # Local import — pxr is heavy and only needed at patch time.
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(input_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {input_path}")

    n_mass = 0
    n_attribute = 0

    def _iter_writable_prims(
        prim_paths: list[str] | None,
        *,
        required_api: Any,
    ) -> Iterator[Any]:
        prims = (
            [stage.GetPrimAtPath(path) for path in prim_paths]
            if prim_paths is not None
            else list(stage.Traverse())
        )
        for prim in prims:
            if not prim or not prim.IsValid():
                continue
            if prim.IsInstanceProxy():
                continue
            if not prim.HasAPI(required_api):
                continue
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
            yield prim

    def _scale_mass(prim_paths: list[str] | None, mass_scale: float) -> None:
        nonlocal n_mass
        ms = float(mass_scale)
        if ms < 0.0:
            raise ValueError(f"mass_scale must be non-negative, got {ms}")
        for prim in _iter_writable_prims(
            prim_paths,
            required_api=UsdPhysics.MassAPI,
        ):
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr()
            if not mass_attr.IsValid():  # pragma: no cover - MassAPI attr is defined
                continue
            has_authored = getattr(mass_attr, "HasAuthoredValue", None)
            authored = (
                has_authored()
                if callable(has_authored)
                else mass_attr.HasAuthoredValueOpinion()
            )
            if authored:
                current = mass_attr.Get()
                if current is not None:
                    mass_attr.Set(float(current) * ms)
                    n_mass += 1

    def _set_material_attr(
        prim_paths: list[str] | None,
        *,
        attribute: str,
        value: float,
    ) -> None:
        nonlocal n_attribute
        for prim in _iter_writable_prims(
            prim_paths,
            required_api=UsdPhysics.MaterialAPI,
        ):
            mat_api = UsdPhysics.MaterialAPI(prim)
            if attribute == "physics:staticFriction":
                mat_api.CreateStaticFrictionAttr(float(value), writeSparsely=False)
            elif attribute == "physics:dynamicFriction":
                mat_api.CreateDynamicFrictionAttr(float(value), writeSparsely=False)
            elif attribute == "physics:restitution":
                mat_api.CreateRestitutionAttr(float(value), writeSparsely=False)
            else:
                raise ValueError(f"Unsupported USD material attribute {attribute!r}")
            n_attribute += 1

    def _set_collision_attr(
        prim_paths: list[str] | None,
        *,
        attribute: str,
        value: float,
    ) -> None:
        nonlocal n_attribute
        if attribute not in {"newton:contact_ke", "newton:contact_kd"}:
            raise ValueError(f"Unsupported USD collision attribute {attribute!r}")
        for prim in _iter_writable_prims(
            prim_paths,
            required_api=UsdPhysics.CollisionAPI,
        ):
            attr = prim.GetAttribute(attribute)
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute(attribute, Sdf.ValueTypeNames.Float)
            attr.Set(float(value))
            n_attribute += 1

    if bindings is None:
        mass_scale = tuned_params.get("mass_scale")
        static_friction = tuned_params.get("static_friction")
        dynamic_friction = tuned_params.get("dynamic_friction")
        restitution = tuned_params.get("restitution")

        if mass_scale is not None:
            _scale_mass(None, float(mass_scale))

        if any(v is not None for v in (static_friction, dynamic_friction, restitution)):
            if static_friction is not None:
                _set_material_attr(
                    None,
                    attribute="physics:staticFriction",
                    value=float(static_friction),
                )
            if dynamic_friction is not None:
                _set_material_attr(
                    None,
                    attribute="physics:dynamicFriction",
                    value=float(dynamic_friction),
                )
            if restitution is not None:
                _set_material_attr(
                    None,
                    attribute="physics:restitution",
                    value=float(restitution),
                )
    else:
        for binding in bindings:
            param_name = str(binding.get("param") or "")
            if param_name not in tuned_params:
                continue
            before_mass = n_mass
            before_attribute = n_attribute
            value = float(tuned_params[param_name])
            raw_paths = binding.get("prim_paths")
            prim_paths = (
                [str(path) for path in raw_paths]
                if isinstance(raw_paths, list)
                else None
            )
            kind = str(binding.get("kind") or "")
            if kind == "usd_mass_scale":
                _scale_mass(prim_paths, value)
            elif kind == "usd_attribute":
                attribute = binding.get("attribute")
                if not isinstance(attribute, str):
                    raise ValueError(
                        f"USD attribute binding missing attribute: {binding}"
                    )
                schema = str(binding.get("schema") or "")
                if schema == "UsdPhysics.CollisionAPI":
                    _set_collision_attr(prim_paths, attribute=attribute, value=value)
                else:
                    _set_material_attr(prim_paths, attribute=attribute, value=value)
            else:
                raise ValueError(f"Unsupported tuning binding kind {kind!r}")
            if n_mass == before_mass and n_attribute == before_attribute:
                raise ValueError(
                    f"Resolved tuning binding for parameter {param_name!r} did "
                    "not update any USD prim. The binding may be stale or "
                    f"point at prims without the required schema: {binding}"
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    flattened_layer = stage.Flatten()
    flattened_layer.Export(str(output_path))

    logger.info(
        "patch_physics_usd: scaled %d mass attrs, updated %d attrs → %s",
        n_mass,
        n_attribute,
        output_path,
    )
    return output_path


def make_tuned_usd_path(output_dir: Path) -> Path:
    """Return the canonical artifact path for the tuned USD."""
    return Path(output_dir) / "tuned_physics.usd"


__all__ = ["patch_physics_usd", "make_tuned_usd_path"]


# silence unused-import warning when typing is consumer-only
_ = Any
