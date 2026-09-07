# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General USD validation (`usd-cli validate [--fix]`).

Deterministic checks over the live stage: composition health, material-binding
integrity, GeomSubset families, unresolved asset references, malformed shaders, and
renderer compatibility (which surface shaders each usd-cli backend can actually render).

`fix=True` applies the safe, mechanical repairs — removing binding targets that point
at nothing and authoring the `materialBind` family metadata bound subsets must carry.
Anything judgement-shaped (out-of-range face indices, missing textures, an unconnected
material) is only reported.
"""

from __future__ import annotations

from usd_core.subsets import MATERIAL_BIND_FAMILY, list_subsets, validate_family

# UsdPreviewSurface's input vocabulary — anything else authored on a preview shader is
# silently ignored by every renderer, which is worth surfacing.
PREVIEW_SURFACE_INPUTS = {
    "diffuseColor", "emissiveColor", "useSpecularWorkflow", "specularColor",
    "metallic", "roughness", "clearcoat", "clearcoatRoughness", "opacity",
    "opacityMode", "opacityThreshold", "ior", "normal", "displacement", "occlusion",
}

# What the supported OVRTX render backends can do with a surface shader kind.
RENDERER_SUPPORT = {
    "UsdPreviewSurface": {"ovrtx": "supported", "remote": "supported"},
    "mdl": {"ovrtx": "supported", "remote": "supported"},
    "unknown": {
        "ovrtx": "unknown shader id — verify rendered appearance",
        "remote": "unknown shader id — verify rendered appearance",
    },
}


def _issue(check: str, severity: str, path: str, message: str) -> dict:
    return {"check": check, "severity": severity, "path": path, "message": message}


def _check_composition(stage, issues: list) -> None:
    errors = stage.GetCompositionErrors() if hasattr(stage, "GetCompositionErrors") else []
    for err in errors:
        site = getattr(err, "rootSite", None)
        path = str(getattr(site, "path", "")) if site else ""
        issues.append(_issue("composition", "error", path, str(err)))


def _check_assets(stage, issues: list) -> None:
    """Asset-valued attributes whose path doesn't resolve (skips UDIM tiles)."""
    from pxr import Usd

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        for attr in prim.GetAttributes():
            tn = str(attr.GetTypeName())
            if tn not in ("asset", "asset[]") or not attr.HasAuthoredValue():
                continue
            vals = attr.Get()
            if vals is None:
                continue
            for val in (vals if tn == "asset[]" else [vals]):
                raw = getattr(val, "path", "") or ""
                if not raw or "<UDIM>" in raw or "<udim>" in raw:
                    continue
                if not (getattr(val, "resolvedPath", "") or ""):
                    issues.append(_issue(
                        "assets", "error", prim.GetPath().pathString,
                        f"{attr.GetName()}: unresolved asset '{raw}'"))


def _check_bindings(stage, issues: list, fixed: list, *, fix: bool) -> None:
    """Direct material bindings must target existing UsdShade.Material prims."""
    from pxr import Usd, UsdShade

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
        if not rel:
            continue
        for t in rel.GetTargets():
            tp = stage.GetPrimAtPath(t)
            problem = ("target prim does not exist" if not tp or not tp.IsValid()
                       else None if UsdShade.Material(tp)
                       else "target is not a UsdShade.Material")
            if not problem:
                continue
            path = prim.GetPath().pathString
            if fix and not prim.IsInstanceProxy():
                rel.RemoveTarget(t)
                fixed.append(f"{path}: removed invalid binding target {t} ({problem})")
            else:
                issues.append(_issue("bindings", "error", path,
                                     f"binding target {t}: {problem}"))


def _check_subsets(stage, issues: list, fixed: list, *, fix: bool) -> None:
    from pxr import Usd, UsdGeom

    from usd_core.subsets import ensure_material_bind_family

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_path = prim.GetPath().pathString
        subs = list_subsets(stage, mesh_path)
        touched = False
        for s in subs:
            fixable = (s["direct_targets"] and s["family"] != MATERIAL_BIND_FAMILY)
            if fixable and fix and not prim.IsInstanceProxy():
                applied = ensure_material_bind_family(stage, s["path"])
                fixed.append(f"{s['path']}: {', '.join(applied)}")
                touched = True
                s = {**s, "problems": [p for p in s["problems"] if "familyName" not in p]}
            for p in s["problems"]:
                issues.append(_issue("subsets", "error", s["path"], p))
        if any(s["family"] == MATERIAL_BIND_FAMILY for s in subs) or touched:
            fam = validate_family(stage, mesh_path)
            if not fam["valid"]:
                issues.append(_issue("subsets", "error", mesh_path,
                                     f"materialBind family invalid: {fam.get('reason')}"))


def _material_shader(stage, prim):
    """(kind, shader, unknown_inputs) for a Material prim; kind in preview/mdl/unknown/None."""
    from pxr import UsdShade

    material = UsdShade.Material(prim)
    source = material.ComputeSurfaceSource("mdl") or material.ComputeSurfaceSource()
    if not source or not source[0]:
        return None, None, []
    shader = source[0]
    node_def = UsdShade.NodeDefAPI(shader.GetPrim())
    if node_def.GetSourceAsset("mdl"):
        return "mdl", shader, []
    sid = shader.GetShaderId()
    if sid == "UsdPreviewSurface":
        unknown = [i.GetBaseName() for i in shader.GetInputs()
                   if i.GetBaseName() not in PREVIEW_SURFACE_INPUTS]
        return "UsdPreviewSurface", shader, unknown
    return ("unknown" if not sid else sid), shader, []


def _check_shaders(stage, issues: list, compat: list) -> None:
    from pxr import Usd, UsdShade

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if UsdShade.Shader(prim):
            node_def = UsdShade.NodeDefAPI(prim)
            has_source = (prim.GetAttribute("info:id").Get()
                          or node_def.GetSourceAsset("mdl")
                          or node_def.GetSourceAsset())
            if not has_source:
                issues.append(_issue("shaders", "error", prim.GetPath().pathString,
                                     "shader has no info:id or source asset"))
        if not UsdShade.Material(prim):
            continue
        kind, shader, unknown = _material_shader(stage, prim)
        path = prim.GetPath().pathString
        if kind is None:
            issues.append(_issue("shaders", "warn", path,
                                 "material has no connected surface shader"))
            continue
        support = RENDERER_SUPPORT.get(kind, RENDERER_SUPPORT["unknown"])
        row = {"material": path, "shader_kind": kind, "backends": dict(support)}
        if kind == "mdl":
            asset = UsdShade.NodeDefAPI(shader.GetPrim()).GetSourceAsset("mdl")
            row["mdl_module"] = asset.path if asset else None
        if unknown:
            row["ignored_inputs"] = unknown
            issues.append(_issue(
                "shaders", "warn", path,
                f"UsdPreviewSurface inputs no renderer reads: {', '.join(sorted(unknown))}"))
        compat.append(row)


def validate_stage(stage, *, fix: bool = False) -> dict:
    """Run all checks; with `fix`, apply the safe repairs first and report them."""
    issues: list[dict] = []
    fixed: list[str] = []
    compat: list[dict] = []

    _check_composition(stage, issues)
    _check_assets(stage, issues)
    _check_bindings(stage, issues, fixed, fix=fix)
    _check_subsets(stage, issues, fixed, fix=fix)
    _check_shaders(stage, issues, compat)

    by_check: dict[str, int] = {}
    for i in issues:
        by_check[i["check"]] = by_check.get(i["check"], 0) + 1
    return {
        "ok": not any(i["severity"] == "error" for i in issues),
        "issues": issues,
        "fixed": fixed,
        "counts": {"errors": sum(1 for i in issues if i["severity"] == "error"),
                   "warnings": sum(1 for i in issues if i["severity"] == "warn"),
                   **by_check},
        "renderer_compat": compat,
    }
