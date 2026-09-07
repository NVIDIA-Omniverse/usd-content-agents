# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UsdGeomSubset support — per-face material assignment on meshes.

A GeomSubset is a named set of face indices under a mesh; subsets in the `materialBind`
family carry their own material bindings, letting one mesh render with several
materials. These helpers list/inspect subsets, validate face coverage, and keep the
family metadata (`familyName`, family type) correct when bindings are authored.
"""

from __future__ import annotations

MATERIAL_BIND_FAMILY = "materialBind"


def _mesh_face_count(prim) -> int | None:
    """Number of faces on a mesh (len of faceVertexCounts), or None if unauthored."""
    attr = prim.GetAttribute("faceVertexCounts")
    counts = attr.Get() if attr else None
    return len(counts) if counts is not None else None


def subset_info(stage, subset_prim) -> dict:
    """JSON-safe description of one GeomSubset: family, faces, binding, problems."""
    from pxr import UsdGeom, UsdShade

    sub = UsdGeom.Subset(subset_prim)
    indices = sub.GetIndicesAttr().Get() or []
    family = sub.GetFamilyNameAttr().Get() or ""
    parent = subset_prim.GetParent()
    face_count = _mesh_face_count(parent)

    api = UsdShade.MaterialBindingAPI(subset_prim)
    mat, _ = api.ComputeBoundMaterial()
    direct = api.GetDirectBindingRel()
    targets = [t.pathString for t in direct.GetTargets()] if direct else []

    problems = []
    if targets and family != MATERIAL_BIND_FAMILY:
        problems.append(f"bound subset lacks familyName='{MATERIAL_BIND_FAMILY}' "
                        f"(has '{family or '(none)'}')")
    if str(sub.GetElementTypeAttr().Get() or "face") != "face":
        problems.append(f"elementType '{sub.GetElementTypeAttr().Get()}' — material "
                        "subsets must be per-face")
    if face_count is not None:
        bad = [i for i in indices if i < 0 or i >= face_count]
        if bad:
            problems.append(f"{len(bad)} face indices out of range 0..{face_count - 1} "
                            f"(e.g. {bad[:3]})")
    return {
        "path": subset_prim.GetPath().pathString,
        "name": subset_prim.GetName(),
        "family": family or None,
        "element_type": str(sub.GetElementTypeAttr().Get() or "face"),
        "face_count": len(indices),
        "mesh_face_count": face_count,
        "bound_material_path": mat.GetPath().pathString if mat else None,
        "direct_targets": targets,
        "problems": problems,
    }


def list_subsets(stage, mesh_path: str, *, family: str | None = None) -> list[dict]:
    """All GeomSubsets under a mesh (optionally one family) as subset_info dicts."""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(mesh_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"no prim at {mesh_path}")
    subs = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim))
    infos = [subset_info(stage, s.GetPrim()) for s in subs]
    if family:
        infos = [i for i in infos if i["family"] == family]
    return infos


def validate_family(stage, mesh_path: str, family: str = MATERIAL_BIND_FAMILY) -> dict:
    """Coverage + validity of one subset family on a mesh.

    Reports USD's own family validation (overlaps vs the authored family type, index
    bounds) plus face coverage — how many of the mesh's faces any subset claims.
    """
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(mesh_path)
    img = UsdGeom.Imageable(prim)
    subs = [s for s in UsdGeom.Subset.GetAllGeomSubsets(img)
            if (UsdGeom.Subset(s).GetFamilyNameAttr().Get() or "") == family]
    face_count = _mesh_face_count(prim)
    covered: set[int] = set()
    overlapping = 0
    for s in subs:
        idx = set(UsdGeom.Subset(s).GetIndicesAttr().Get() or [])
        overlapping += len(covered & idx)
        covered |= idx
    family_type = str(UsdGeom.Subset.GetFamilyType(img, family))
    valid, reason = UsdGeom.Subset.ValidateFamily(img, UsdGeom.Tokens.face, family)
    out = {
        "mesh": mesh_path,
        "family": family,
        "family_type": family_type,
        "subsets": len(subs),
        "valid": bool(valid),
        "faces_covered": len(covered),
        "mesh_face_count": face_count,
        "overlapping_face_claims": overlapping,
    }
    if face_count is not None:
        out["faces_uncovered"] = max(0, face_count - len(covered))
    if not valid:
        out["reason"] = str(reason)
    return out


def ensure_material_bind_family(stage, subset_path: str) -> list[str]:
    """Make a subset legal for material binding; returns the fixes applied.

    Sets `familyName = materialBind` and, if the family has no authored type yet on the
    parent mesh, marks it `nonOverlapping` (the conventional type for material subsets —
    partition would demand full coverage, which partial assignments legitimately lack).
    """
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(subset_path)
    sub = UsdGeom.Subset(prim)
    if not sub:
        raise ValueError(f"{subset_path} is not a UsdGeomSubset")
    fixed = []
    if (sub.GetFamilyNameAttr().Get() or "") != MATERIAL_BIND_FAMILY:
        sub.GetFamilyNameAttr().Set(MATERIAL_BIND_FAMILY)
        fixed.append(f"familyName={MATERIAL_BIND_FAMILY}")
    if not sub.GetElementTypeAttr().HasAuthoredValue():
        sub.GetElementTypeAttr().Set(UsdGeom.Tokens.face)
        fixed.append("elementType=face")
    img = UsdGeom.Imageable(prim.GetParent())
    mesh_prim = prim.GetParent()
    token_attr = mesh_prim.GetAttribute(f"subsetFamily:{MATERIAL_BIND_FAMILY}:familyType")
    if not (token_attr and token_attr.HasAuthoredValue()):
        UsdGeom.Subset.SetFamilyType(img, MATERIAL_BIND_FAMILY,
                                     UsdGeom.Tokens.nonOverlapping)
        fixed.append("familyType=nonOverlapping")
    return fixed
