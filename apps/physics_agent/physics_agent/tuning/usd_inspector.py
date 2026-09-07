# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD inspection helpers for tuning resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UsdTuningCandidate:
    """A concrete USD location that can back a resolved tuning binding."""

    schema: str
    attribute: str
    prim_path: str
    has_authored_value: bool
    current_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "attribute": self.attribute,
            "prim": self.prim_path,
            "has_authored_value": self.has_authored_value,
            "current_value": self.current_value,
        }


@dataclass(frozen=True)
class UsdTuningReport:
    """Tuning-relevant facts discovered from one USD asset."""

    usd_path: Path
    candidates: tuple[UsdTuningCandidate, ...]

    def find(
        self,
        *,
        schema: str,
        attribute: str,
        require_authored_value: bool = False,
    ) -> tuple[UsdTuningCandidate, ...]:
        matches = [
            c
            for c in self.candidates
            if c.schema == schema and c.attribute == attribute
        ]
        if require_authored_value:
            matches = [c for c in matches if c.has_authored_value]
        return tuple(matches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "usd_path": str(self.usd_path),
            "candidates": [c.to_dict() for c in self.candidates],
        }


def _attr_value(attr: Any) -> Any:
    try:
        return attr.Get()
    except Exception:
        return None


def _has_authored_value(attr: Any) -> bool:
    try:
        has_authored = getattr(attr, "HasAuthoredValue", None)
        if callable(has_authored):
            return bool(attr.IsValid() and has_authored())
        return bool(attr.IsValid() and attr.HasAuthoredValueOpinion())
    except Exception:
        return False


def inspect_usd_for_tuning(usd_path: Path | str) -> UsdTuningReport:
    """Inspect a USD stage for physics attributes the tuner can bind to."""

    path = Path(usd_path)
    if not path.exists():
        raise FileNotFoundError(f"Input USD not found: {path}")

    from pxr import Usd, UsdPhysics

    try:
        stage = Usd.Stage.Open(str(path))
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to open USD stage: {path}") from exc
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {path}")

    candidates: list[UsdTuningCandidate] = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            for attribute in ("newton:contact_ke", "newton:contact_kd"):
                attr = prim.GetAttribute(attribute)
                candidates.append(
                    UsdTuningCandidate(
                        schema="UsdPhysics.CollisionAPI",
                        attribute=attribute,
                        prim_path=prim_path,
                        has_authored_value=_has_authored_value(attr),
                        current_value=_attr_value(attr),
                    )
                )
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            material_api = UsdPhysics.MaterialAPI(prim)
            for attribute, attr in (
                ("physics:staticFriction", material_api.GetStaticFrictionAttr()),
                ("physics:dynamicFriction", material_api.GetDynamicFrictionAttr()),
                ("physics:restitution", material_api.GetRestitutionAttr()),
            ):
                candidates.append(
                    UsdTuningCandidate(
                        schema="UsdPhysics.MaterialAPI",
                        attribute=attribute,
                        prim_path=prim_path,
                        has_authored_value=_has_authored_value(attr),
                        current_value=_attr_value(attr),
                    )
                )
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_attr = UsdPhysics.MassAPI(prim).GetMassAttr()
            candidates.append(
                UsdTuningCandidate(
                    schema="UsdPhysics.MassAPI",
                    attribute="physics:mass",
                    prim_path=prim_path,
                    has_authored_value=_has_authored_value(mass_attr),
                    current_value=_attr_value(mass_attr),
                )
            )

    return UsdTuningReport(usd_path=path.resolve(), candidates=tuple(candidates))


__all__ = [
    "UsdTuningCandidate",
    "UsdTuningReport",
    "inspect_usd_for_tuning",
]
