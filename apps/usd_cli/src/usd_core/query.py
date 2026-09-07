# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only scene queries — adapted from world_understanding's `agentic .../usd_queries.py`
and `utils/usd/*`. Pure functions over a live stage; the Session wraps them in Responses.

Covers prim properties (attributes/relationships/bounds/material), stage info, search
(by name/type/material/visibility/box/sphere), a rule-based describe, and mesh candidates
(a low-level physics inspection input).
"""

from __future__ import annotations

from usd_core.edit import _jsonsafe


def iter_prims(stage):
    """All active prims *including* native-instance proxies.

    `stage.Traverse()` deliberately skips the contents of instanceable prims, which makes
    an instanced asset (e.g. a flattened PCB with thousands of meshes behind internal
    `instanceable` references) look like it has zero geometry. Read-only inspection must
    descend into instance proxies to see the real scene.
    """
    from pxr import Usd

    return stage.Traverse(Usd.TraverseInstanceProxies())


_ELIDE_AFTER = 32


def _round_floats(value, sig: int = 6):
    """Round floats for agent output — 17-digit doubles carry no agent-usable
    information and cost real tokens (168 kB across the round-7 traces).
    Non-finite floats pass through as strings (JSON-strict encoders 500 on
    inf/nan; physics schema defaults are legitimately inf)."""
    import math
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return float(f"{value:.{sig}g}")
    if isinstance(value, (list, tuple)):
        return [_round_floats(v, sig) for v in value]
    return value


def _elide_large(value, limit: int = _ELIDE_AFTER):
    """Summarize huge array values (mesh points/indices) in properties output.

    Dense-mesh attributes were dumping multi-million-token arrays into
    `properties --json`; even the 8-value preview + elision marker was ~100 kB
    of noise per run. Long sequences now collapse to a single descriptor —
    agents that need the data read the USD, not the CLI.
    """
    if isinstance(value, (list, tuple)) and len(value) > limit:
        kinds = {type(v).__name__ for v in value[:4]}
        return f"<{len(value)} values ({'/'.join(sorted(kinds))}) — elided>"
    return _round_floats(value)


def prim_attribute(stage, path: str, attr: str, *, limit: int = 256) -> dict:
    """One attribute's value in full(er) detail — the escape hatch from elision.

    `properties @ref` elides long arrays; `properties @ref ATTR` is how an agent
    asks for the actual numbers of a single attribute (round 8: 22 failed
    attempts at exactly this call shape before it existed).
    """
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"no prim at {path}")
    a = prim.GetAttribute(attr)
    if not a:
        names = [x.GetName() for x in prim.GetAttributes()]
        close = [n for n in names if attr.lower() in n.lower()][:8]
        hint = f" — close matches: {', '.join(close)}" if close else ""
        raise ValueError(f"no attribute '{attr}' on {path}{hint}")
    return {"path": path, "attr": attr, "type": str(a.GetTypeName()),
            "authored": a.HasAuthoredValue(),
            "value": _elide_large(_jsonsafe(a.Get()), limit=limit)}


def prim_properties(stage, path: str, *, max_attrs: int = 100) -> dict:
    """Compact, JSON-safe properties for a prim (attrs, relationships, bounds, material)."""
    from pxr import UsdGeom

    from usd_core.materials import bound_material
    from usd_core.spatial import get_world_bbox

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"no prim at {path}")
    attrs = {}
    all_attrs = prim.GetAttributes()
    for a in all_attrs[:max_attrs]:
        if not a.HasAuthoredValue():
            continue
        attrs[a.GetName()] = {"type": str(a.GetTypeName()),
                              "value": _elide_large(_jsonsafe(a.Get()))}
    rels = {}
    for r in prim.GetRelationships():
        targets = [t.pathString for t in r.GetTargets()]
        if targets:
            rels[r.GetName()] = targets
    out = {
        "path": path,
        "name": prim.GetName(),
        "type": str(prim.GetTypeName()),
        "active": prim.IsActive(),
        "kind": _kind(prim),
        "visible": _visible(prim),
        "attributes": attrs,
        "relationships": rels,
        "attribute_count": len(all_attrs),
        "truncated": len(all_attrs) > max_attrs,
    }
    bbox = get_world_bbox(stage, path)
    if bbox:
        out["bounds"] = {k: bbox[k] for k in ("min", "max", "center", "size")}
    mat = bound_material(stage, path)
    if mat["bound_material_path"]:
        out["material"] = mat
    return out


def _kind(prim) -> str | None:
    from pxr import Usd

    k = Usd.ModelAPI(prim).GetKind()
    return k or None


def _visible(prim) -> bool:
    from pxr import UsdGeom

    if not prim.IsA(UsdGeom.Imageable):
        return True
    return UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible


def stage_info(stage, stage_path: str | None) -> dict:
    """Stage metadata: up axis, units, frame range, counts, layers, default prim, extent."""
    from pxr import Usd, UsdGeom

    from usd_core.spatial import scene_bbox

    total = meshes = instances = proxies = 0
    for p in iter_prims(stage):
        total += 1
        if p.IsInstance():
            instances += 1
        if p.IsInstanceProxy():
            proxies += 1
        if p.IsA(UsdGeom.Mesh):
            meshes += 1
    dp = stage.GetDefaultPrim()
    used = stage.GetUsedLayers()
    info = {
        "stage": stage_path,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "start_time": stage.GetStartTimeCode(),
        "end_time": stage.GetEndTimeCode(),
        "fps": stage.GetTimeCodesPerSecond(),
        "prim_count": total,
        "mesh_count": meshes,
        "default_prim": dp.GetPath().pathString if dp and dp.IsValid() else None,
        "layer_count": len(used),
    }
    if instances:
        info["instance_count"] = instances
        info["instance_proxy_prim_count"] = proxies
    rng = scene_bbox(stage)
    if rng is not None:
        info["extent"] = {"min": list(rng.GetMin()), "max": list(rng.GetMax())}
    return info


def decode_identifier(name: str) -> str:
    """Human label behind a USD-encoded prim name.

    CAD/omni importers encode illegal identifier characters as `_UXX_` hex runs
    (space -> `_U20_`, é -> `_UE9_`) behind a `tn__` prefix — round-8 agents'
    plain-label patterns bounced off `tn__Steel_20Painted...` names through
    several retry rounds. Best-effort: undecodable runs stay as-is."""
    import re as _re

    if name.startswith("tn__"):
        name = name[4:]
    def _sub(m):
        try:
            return chr(int(m.group(1), 16))
        except ValueError:
            return m.group(0)
    return _re.sub(r"_U([0-9A-Fa-f]{2,6})_", _sub, name)


def _name_matcher(name):
    """Case-insensitive matcher for --name: `|`-separated alternates, each a glob
    (fnmatch) or substring. Round-7 agents dumped the whole scene 14x and grepped
    client-side (`resolve | awk '/at10|belt|correa/'`) for want of alternation.
    Patterns match the raw prim name OR its decoded human label (see
    decode_identifier), so `--name 'Steel Painted*'` finds `tn__Steel_20Painted…`."""
    import fnmatch

    alts = [a.strip().lower() for a in name.split("|") if a.strip()]

    def match(prim_name: str) -> bool:
        candidates = {prim_name.lower()}
        decoded = decode_identifier(prim_name)
        if decoded != prim_name:
            candidates.add(decoded.lower())
        for low in candidates:
            for a in alts:
                if any(c in a for c in "*?["):
                    if fnmatch.fnmatch(low, a):
                        return True
                elif a in low:
                    return True
        return False

    return match


def find_prims(stage, *, name=None, type_=None, material=None, has_attr=None,
               in_box=None, in_sphere=None, visible=False, hidden=False,
               under=None, where=None) -> list[dict]:
    """Search prims by filters; returns [{path, type, name}] (caller maps to @refs).

    `where` takes the same rule expressions as `material --where`
    (name~=Glob*, attr:primvars:displayColor==..., ANDed) — round 8's agents
    reached for `find --where` first and read installed sources when it
    wasn't there."""
    from pxr import UsdGeom

    from usd_core import selector
    from usd_core.materials import bound_material
    from usd_core.spatial import get_world_bbox

    rules = [selector.parse_rule(w) for w in (where or [])]
    match_name = _name_matcher(name) if name else None
    type_l = type_.lower() if type_ else None
    under_prefix = (under.rstrip("/") + "/") if under else None
    results = []
    for prim in iter_prims(stage):
        if not prim.IsActive():
            continue
        p = prim.GetPath().pathString
        if p == "/":
            continue
        if under_prefix and not (p + "/").startswith(under_prefix):
            continue
        if match_name and not match_name(prim.GetName()):
            continue
        if type_l and type_l != str(prim.GetTypeName()).lower():
            continue
        if has_attr and not prim.HasAttribute(has_attr):
            continue
        if visible and not _visible(prim):
            continue
        if hidden and _visible(prim):
            continue
        if material:
            bm = bound_material(stage, p)["bound_material_path"] or ""
            if material.lstrip("@").lower() not in bm.lower():
                continue
        if rules and not all(selector._rule_matches(stage, prim, r) for r in rules):
            continue
        if in_box or in_sphere:
            bbox = get_world_bbox(stage, p)
            if not bbox:
                continue
            c = bbox["center"]
            if in_box and not _in_box(c, in_box):
                continue
            if in_sphere and not _in_sphere(c, in_sphere):
                continue
        row = {"path": p, "type": str(prim.GetTypeName()), "name": prim.GetName()}
        if prim.IsInstanceProxy():
            row["instance_proxy"] = True  # read/inspect-only: USD forbids editing inside an instance
        results.append(row)
    return results


def _in_box(c, box) -> bool:
    (x1, y1, z1), (x2, y2, z2) = box
    return (min(x1, x2) <= c[0] <= max(x1, x2) and min(y1, y2) <= c[1] <= max(y1, y2)
            and min(z1, z2) <= c[2] <= max(z1, z2))


def _in_sphere(c, sph) -> bool:
    center, radius = sph
    d2 = sum((c[i] - center[i]) ** 2 for i in range(3))
    return d2 <= radius * radius


def mesh_candidates(stage, *, root: str | None = None) -> list[dict]:
    """Mesh prims with bounds, material, and existing physics schemas — a caller's
    inspection input (adapted from physics_ops candidate inspection).

    Descends into native-instance proxies so instanced assets (e.g. SimReady payloads behind
    `instanceable` references) report their real mesh candidates instead of none. Proxy rows
    carry `instance_proxy: True` plus `instance_root` — the editable prim that physics
    authoring (`physics apply`) must target, since USD forbids authoring inside an instance.
    """
    from pxr import Usd, UsdGeom

    from usd_core.materials import bound_material
    from usd_core.spatial import get_world_bbox

    start = stage.GetPrimAtPath(root) if root else stage.GetPseudoRoot()
    out = []
    for prim in Usd.PrimRange(start, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or not prim.IsActive():
            continue
        p = prim.GetPath().pathString
        schemas = [s for s in prim.GetAppliedSchemas() if "Physics" in s]
        row = {
            "path": p, "name": prim.GetName(),
            "bounds": get_world_bbox(stage, p),
            "material": bound_material(stage, p)["bound_material_path"],
            "existing_physics_schemas": schemas,
        }
        if prim.IsInstanceProxy():
            row["instance_proxy"] = True
            row["instance_root"] = instance_root_path(prim)
        out.append(row)
    return out


def instance_root_path(prim) -> str | None:
    """Path of the nearest enclosing *editable* instance root of an instance proxy.

    Walks ancestors to the first prim that is an instance root and not itself a proxy
    (nested instancing makes inner roots proxies too). This is where USD allows authoring
    overrides that affect the proxy — e.g. material bindings or physics schemas.
    """
    anc = prim.GetParent()
    while anc and anc.IsValid():
        if anc.IsInstance() and not anc.IsInstanceProxy():
            return anc.GetPath().pathString
        anc = anc.GetParent()
    return None


def describe_scene(stage, stage_path: str | None) -> str:
    """Rule-based natural-language summary of the scene (no VLM)."""
    from pxr import UsdGeom

    info = stage_info(stage, stage_path)
    type_counts: dict[str, int] = {}
    for prim in iter_prims(stage):
        if prim.IsActive():
            type_counts[str(prim.GetTypeName())] = type_counts.get(str(prim.GetTypeName()), 0) + 1
    top = sorted(type_counts.items(), key=lambda kv: -kv[1])[:6]
    parts = [f"Scene '{info['stage']}' — {info['prim_count']} prims "
             f"({info['mesh_count']} meshes), up axis {info['up_axis']}, "
             f"{info['meters_per_unit']} m/unit."]
    if info.get("instance_count"):
        parts.append(f"{info['instance_count']} native instances "
                     f"({info['instance_proxy_prim_count']} prims counted via instance proxies).")
    if "extent" in info:
        mn, mx = info["extent"]["min"], info["extent"]["max"]
        size = [round(mx[i] - mn[i], 3) for i in range(3)]
        parts.append(f"World extent size ~ {size} (scene units).")
    if top:
        parts.append("Prim types: " + ", ".join(f"{n}×{t}" for t, n in top) + ".")
    return " ".join(parts)


def describe_prim(stage, path: str, ref: str | None = None) -> str:
    """Rule-based description of a single prim, keyed on its @ref (not its SdfPath)."""
    props = prim_properties(stage, path, max_attrs=20)
    head = f'{ref} ' if ref else ""
    bits = [f"{head}is a {props['type']} named '{props['name']}'",
            "visible" if props["visible"] else "hidden"]
    if props.get("bounds"):
        bits.append(f"size {[round(s, 3) for s in props['bounds']['size']]}")
    if props.get("material"):
        bits.append(f"material {props['material']['bound_material_path'].rsplit('/', 1)[-1]}")
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid() and prim.IsInstance():
        # a collapsed instance still reports the effective materials inside it
        from usd_core.materials import effective_materials_under
        agg = effective_materials_under(stage, path)
        if agg["materials"]:
            inner = ", ".join(f"{p.rsplit('/', 1)[-1]}×{n}" for p, n in sorted(
                agg["materials"].items(), key=lambda kv: -kv[1]))
            bits.append(f"an instance containing {agg['gprims']} gprims with "
                        f"materials {inner}"
                        + (f" ({agg['unbound']} unbound)" if agg["unbound"] else ""))
    return ", ".join(bits) + "."
