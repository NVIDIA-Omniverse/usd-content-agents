# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete material-binding audit (`usd-cli material audit`).

One pass over the stage answering the questions a material workflow keeps asking:
which renderables are bound (directly / by inheritance / not at all), what does each
one *effectively* resolve to, which subsets carry face-level bindings and how much of
their mesh do they cover, which binding relationships point at garbage, and which
authored materials nothing uses.
"""

from __future__ import annotations

from usd_core.subsets import MATERIAL_BIND_FAMILY, list_subsets, validate_family

# JSON-payload cap: a 36k-mesh `--json material audit` emitted ~2.2M tokens of per-prim
# paths and the consumer truncated it anyway. Aggregate counts stay exact; any per-prim
# listing longer than _CAP_OVER is sliced to its first _CAP_KEEP entries. Lists stay
# HOMOGENEOUS (no marker strings mixed into dict lists — schema-v1 consumers iterate
# these): the omitted count is reported separately as `<key>_omitted`, plus a top-level
# `truncated: true` flag.
_CAP_OVER = 200
_CAP_KEEP = 50


def _capped(items: list) -> tuple[list, int]:
    """Slice a per-prim listing to its first _CAP_KEEP entries; return (list, omitted)."""
    if len(items) <= _CAP_OVER:
        return items, 0
    return items[:_CAP_KEEP], len(items) - _CAP_KEEP


def _binding_state(stage, prim) -> dict:
    from pxr import UsdShade

    api = UsdShade.MaterialBindingAPI(prim)
    mat, _ = api.ComputeBoundMaterial()
    direct = api.GetDirectBindingRel()
    targets = [t.pathString for t in direct.GetTargets()] if direct else []
    bound = mat.GetPath().pathString if mat and mat.GetPrim().IsValid() else None
    invalid = []
    for t in targets:
        tp = stage.GetPrimAtPath(t)
        if not tp or not tp.IsValid():
            invalid.append({"target": t, "problem": "target prim does not exist"})
        elif not UsdShade.Material(tp):
            invalid.append({"target": t, "problem": "target is not a UsdShade.Material"})
    kind = "none"
    if bound:
        kind = "direct" if targets else "inherited"
    return {"binding": kind, "material": bound, "direct_targets": targets,
            "invalid_targets": invalid}


def material_audit(stage, *, effective: bool = False,
                   include_subsets: bool = False, under: str | None = None) -> dict:
    """Audit every renderable's material state; see module docstring for the contract.

    `effective` adds the per-renderable resolved material listing (otherwise only the
    aggregate counts are returned); `include_subsets` adds GeomSubset coverage and
    family validation for meshes using face-level assignment.

    Per-prim listings longer than 200 entries are sliced to their first 50, with the
    omitted count in `<key>_omitted` and `truncated: true` on the report, so the
    payload stays bounded on large scenes while lists stay schema-homogeneous; the
    `counts` block is always exact.
    """
    from pxr import Usd, UsdGeom, UsdShade

    renderables = []
    bound_direct = bound_inherited = unbound = 0
    invalid_bindings: list[dict] = []
    used_materials: set[str] = set()
    subset_reports: list[dict] = []
    subset_problems = 0

    under_prefix = (under.rstrip("/") + "/") if under else None
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if under_prefix and not (prim.GetPath().pathString + "/").startswith(under_prefix):
            continue
        if not prim.IsActive():
            continue
        path = prim.GetPath().pathString
        # invalid direct targets matter on *any* prim (bindings inherit down)
        state = None
        if prim.HasRelationship(UsdShade.Tokens.materialBinding):
            state = _binding_state(stage, prim)
            for bad in state["invalid_targets"]:
                invalid_bindings.append({"path": path, **bad})
        if not prim.IsA(UsdGeom.Gprim):
            continue
        state = state or _binding_state(stage, prim)
        if state["material"]:
            used_materials.add(state["material"])
            if state["binding"] == "direct":
                bound_direct += 1
            else:
                bound_inherited += 1
        else:
            unbound += 1
        row = {"path": path, **{k: state[k] for k in ("binding", "material")}}
        if prim.IsInstanceProxy():
            row["instance_proxy"] = True
        renderables.append(row)

        if include_subsets and prim.IsA(UsdGeom.Mesh):
            subs = list_subsets(stage, path, family=None)
            bind_subs = [s for s in subs if s["bound_material_path"] or
                         s["family"] == MATERIAL_BIND_FAMILY]
            if not bind_subs:
                continue
            for s in bind_subs:
                if s["bound_material_path"]:
                    used_materials.add(s["bound_material_path"])
                subset_problems += len(s["problems"])
            report = validate_family(stage, path)
            report["subsets_detail"] = bind_subs
            if not report["valid"]:
                subset_problems += 1
            subset_reports.append(report)

    # unused materials: authored Material prims nothing (prim or subset) resolves to
    all_materials = [p.GetPath().pathString for p in stage.Traverse()
                     if UsdShade.Material(p)]
    unused = sorted(set(all_materials) - used_materials)

    out = {
        "counts": {
            "renderables": len(renderables),
            "bound_direct": bound_direct,
            "bound_inherited": bound_inherited,
            "unbound": unbound,
            "invalid_binding_targets": len(invalid_bindings),
            "materials_authored": len(all_materials),
            "materials_unused": len(unused),
        },
        "unbound_paths": [r["path"] for r in renderables if not r["material"]],
        "invalid_bindings": invalid_bindings,
        "unused_materials": unused,
    }
    if effective:
        out["renderables"] = renderables
    if include_subsets:
        out["counts"]["meshes_with_material_subsets"] = len(subset_reports)
        out["counts"]["subset_problems"] = subset_problems
        out["subset_families"] = subset_reports
    out["ok"] = not invalid_bindings and subset_problems == 0
    truncated = False
    for key in ("unbound_paths", "invalid_bindings", "unused_materials", "renderables",
                "subset_families"):
        if key in out:
            out[key], omitted = _capped(out[key])
            if omitted:
                out[f"{key}_omitted"] = omitted
                truncated = True
    # subset reports carry a nested per-subset listing; cap those too or the
    # payload stays unbounded on subset-heavy scenes.
    for rep in out.get("subset_families", []):
        detail = rep.get("subsets_detail")
        if isinstance(detail, list):
            rep["subsets_detail"], omitted = _capped(detail)
            if omitted:
                rep["subsets_detail_omitted"] = omitted
                truncated = True
    if truncated:
        out["truncated"] = True
    return out
