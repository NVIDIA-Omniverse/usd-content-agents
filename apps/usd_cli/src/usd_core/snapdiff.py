# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural scene-tree diff for `snapshot --diff` (PR-2.5).

The counterpart to the image-based render-diff (`imaging.render_diff`): where that compares
*pixels* of two renders, this compares the *scene graph* — which prims were added, removed,
or changed (transform / visibility / material / type / name). Diffs are keyed by SdfPath,
which is deterministic and stable within a session, so the result
shows only real edits, not incidental re-orderings.

Three pieces, split so the comparison is stage-free and unit-testable:

- `capture_state(stage, root)` walks a stage into a plain, JSON-safe dict (one entry per
  meaningful prim). The daemon keeps the previous capture as the `-D` baseline and can
  capture a checkpoint's stage for `--since`. Diff captures are always *full*: rooted at
  the stage pseudo-root, descending into native-instance proxies, and including Material
  and Shader prims — so new Looks content and bindings inside instances are diffable.
- `diff_states(before, after)` compares two captures → added / removed / changed. Scope /
  type / depth / visibility filters are applied here, symmetrically to both sides — a
  baseline banked by a scoped or filtered snapshot must never make a later whole-scene
  `-D` report membership churn (the 2026-07-09 task-12 failure).
- `format_diff(diff)` renders it as text: a compact per-prim summary by default, or an
  attribute-level line (`~ @n5 -t(2,0,0) +t(5,0,0)`) with `structural=True`.
"""

from __future__ import annotations

_TRS_TOL = 1e-4  # ignore sub-thousandth transform jitter from float round-trips
_ABBR = {"translate": "t", "rotate": "r", "scale": "s"}


def _stable_path(path):
    """Rewrite a `/__Prototype_N/...` path to a numbering-independent form.

    Prototype numbering is assigned by traversal order, so a material computed *inside*
    a prototype would flip the diff on every re-composition even though nothing changed.
    Keyed per-prim, the prototype-relative remainder is still a faithful signature.
    """
    if path and path.startswith("/__"):
        return "proto:/" + path.split("/", 2)[-1]
    return path


def capture_state(stage, root_path: str, *, refs=None, type_filter=None,
                  depth: int | None = None, visible_only: bool = False) -> dict:
    """Snapshot a stage's meaningful prims into a plain dict keyed by SdfPath.

    Membership mirrors `Session._index_prims`, so a capture of the live stage and a
    capture of a checkpoint stage are directly comparable. `refs` (the live ref table) is
    used only to attach a display ref where a path still has one — matching by path is
    what makes the diff meaningful, not the ref (refs renumber; paths don't).

    A capture includes Material/Shader prims (new Looks content must be diffable —
    task-12: a full material pass reported "(no changes)" without this), and collapsed
    instance roots carry an aggregate `materials_within` signature so prototype-source
    binds inside instances surface too. Diff captures should be *unfiltered* (root "/",
    no type/depth/visible) — pass the display filters to `diff_states` instead, which
    applies them to both sides symmetrically. The filter parameters here are kept for
    direct callers that want a single filtered view.
    """
    from pxr import Usd, UsdGeom, UsdShade

    from usd_core.edit import current_trs
    from usd_core.materials import bound_material, effective_materials_under

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return {}
    base = root_path.rstrip("/").count("/")
    tf = {t.lower() for t in type_filter} if type_filter else None

    # Mirror the snapshot scoping rule: a whole-scene capture stays compact (instances are
    # units — their root's material/xform/visibility is what diffs), but a capture scoped
    # *to* an instance opts into its proxy contents, like `snapshot <instance-ref>` does.
    into_instance = root.IsInstance() or root.IsInstanceProxy()
    prim_iter = Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if into_instance \
        else Usd.PrimRange(root)

    state: dict = {}
    for prim in prim_iter:
        path = prim.GetPath().pathString
        if path == "/" or not prim.IsActive():
            continue
        type_name = str(prim.GetTypeName())
        is_cam = prim.IsA(UsdGeom.Camera)
        is_light = "Light" in type_name
        is_img = prim.IsA(UsdGeom.Imageable)
        is_shading = prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader)
        if not (is_cam or is_light or is_img or is_shading):
            continue
        d = path.rstrip("/").count("/") - base
        if depth is not None and d > depth:
            continue
        if tf and type_name.lower() not in tf:
            continue
        visible = True
        if is_img:
            visible = UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
        if visible_only and not visible:
            continue
        trs = None
        if prim.IsA(UsdGeom.Xformable):
            try:
                trs = current_trs(stage, path)
            except Exception:  # noqa: BLE001 — non-canonical xforms shouldn't break capture
                trs = None
        material, binding = None, []
        if not is_shading:  # a material's own "binding" is meaningless
            try:
                m = bound_material(stage, path)
                material = _stable_path(m.get("bound_material_path"))
                # the direct material:binding targets are part of the signature — an
                # edit that swaps the direct bind must diff even if the computed
                # (inherited) material happens to match
                binding = [_stable_path(t) for t in m.get("direct_targets") or []]
            except Exception:  # noqa: BLE001
                material, binding = None, []
        entry = {
            "ref": refs.ref_for_path(path) if refs else None,
            "type": type_name,
            "name": prim.GetName(),
            "visible": visible,
            "trs": trs,
            "material": material,
            "binding": binding,
        }
        if prim.IsInstance() and not prim.IsInstanceProxy():
            # a collapsed instance diffs as a unit, but a binding edit *inside* it
            # (a prototype-source bind every instance inherits) must still surface —
            # sign the aggregate of effective materials under the root
            try:
                agg = effective_materials_under(stage, path)["materials"]
                entry["materials_within"] = {
                    _stable_path(p): n for p, n in sorted(agg.items())}
            except Exception:  # noqa: BLE001
                pass
        state[path] = entry
    return state


def _vec_changed(a, b, tol: float = _TRS_TOL) -> bool:
    return any(abs(float(x) - float(y)) > tol for x, y in zip(a, b))


def _trs_delta(before, after) -> dict | list | None:
    """Per-component transform delta, or None if unchanged within tolerance."""
    if before is None or after is None:
        return None if before == after else [before, after]
    comps: dict = {}
    for comp in ("translate", "rotate", "scale"):
        if _vec_changed(before[comp], after[comp]):
            comps[comp] = [before[comp], after[comp]]
    return comps or None


def filter_state(state: dict, *, scope: str | None = None, type_filter=None,
                 depth: int | None = None, visible_only: bool = False) -> dict:
    """Restrict a capture to a subtree / types / depth / visibility.

    Diff filters must be applied here — identically to BOTH sides — never baked into the
    captures: a baseline captured through one filter and a current state captured through
    another differ in *membership*, which reads as wholesale added/removed churn (task-12
    reported added: 148 | removed: 143 after a single bind because of exactly that).
    """
    if not (scope or type_filter or depth is not None or visible_only):
        return state
    prefix = scope.rstrip("/") if scope else None
    tf = {t.lower() for t in type_filter} if type_filter else None
    base = prefix.count("/") if prefix else 0
    out: dict = {}
    for path, entry in state.items():
        if prefix and path != prefix and not path.startswith(prefix + "/"):
            continue
        if depth is not None and path.rstrip("/").count("/") - base > depth:
            continue
        if tf and entry["type"].lower() not in tf:
            continue
        if visible_only and not entry.get("visible", True):
            continue
        out[path] = entry
    return out


def diff_states(before: dict | None, after: dict | None, *, scope: str | None = None,
                type_filter=None, depth: int | None = None,
                visible_only: bool = False) -> dict:
    """Compare two captures. Returns added / removed / changed lists plus counts.

    Entries are keyed and matched by prim path — the one identity that is stable across
    snapshots. Optional filters narrow the comparison and apply to both sides.
    """
    flt = dict(scope=scope, type_filter=type_filter, depth=depth, visible_only=visible_only)
    before = filter_state(before or {}, **flt)
    after = filter_state(after or {}, **flt)

    def stub(path, state):
        s = after.get(path) if path in after else before.get(path)
        return {"path": path, "ref": s["ref"], "type": s["type"], "name": s["name"]}

    added = [stub(p, after) for p in after if p not in before]
    removed = [stub(p, before) for p in before if p not in after]

    changed = []
    for path, a in after.items():
        b = before.get(path)
        if b is None:
            continue
        fields: dict = {}
        for key in ("type", "name", "visible", "material", "binding", "materials_within"):
            if b.get(key) != a.get(key):
                fields[key] = [b.get(key), a.get(key)]
        trs = _trs_delta(b["trs"], a["trs"])
        if trs:
            fields["trs"] = trs
        if fields:
            changed.append({"path": path, "ref": a["ref"], "fields": fields})

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "counts": {"added": len(added), "removed": len(removed), "changed": len(changed)},
    }


# -- text rendering --------------------------------------------------------------
def _label(entry: dict) -> str:
    return entry.get("ref") or entry["path"]


def _num(x) -> str:
    return f"{float(x):g}"


def _vec(v) -> str:
    return "(" + ",".join(_num(x) for x in v) + ")"


def _short(path) -> str:
    return path.rsplit("/", 1)[-1] if path else "none"


def _mats(agg) -> str:
    """Compact `leaf×count` rendering of a materials_within aggregate."""
    return ",".join(f"{_short(p)}×{n}" for p, n in sorted((agg or {}).items()))


def _structural_fields(fields: dict) -> str:
    parts = []
    for key, val in fields.items():
        if key == "trs":
            if isinstance(val, dict):
                for comp, (bv, av) in val.items():
                    ab = _ABBR[comp]
                    parts.append(f"-{ab}{_vec(bv)} +{ab}{_vec(av)}")
            else:
                parts.append("-trs +trs")
        elif key == "visible":
            parts.append(f"-vis({str(val[0]).lower()}) +vis({str(val[1]).lower()})")
        elif key == "material":
            parts.append(f"-mat({_short(val[0])}) +mat({_short(val[1])})")
        elif key == "binding":  # direct material:binding targets
            bv = _short(val[0][-1]) if val[0] else "none"
            av = _short(val[1][-1]) if val[1] else "none"
            parts.append(f"-bind({bv}) +bind({av})")
        elif key == "materials_within":  # aggregate inside a collapsed instance
            parts.append(f"-mats{{{_mats(val[0])}}} +mats{{{_mats(val[1])}}}")
        else:  # type, name
            parts.append(f"-{key}({val[0]}) +{key}({val[1]})")
    return " ".join(parts)


def _summary_fields(fields: dict) -> str:
    parts = []
    if "trs" in fields:
        comps = fields["trs"].keys() if isinstance(fields["trs"], dict) else ()
        moved = []
        if "translate" in comps:
            moved.append("moved")
        if "rotate" in comps:
            moved.append("rotated")
        if "scale" in comps:
            moved.append("scaled")
        parts += moved or ["transformed"]
    if "visible" in fields:
        parts.append("shown" if fields["visible"][1] else "hidden")
    if "material" in fields:
        parts.append("material changed")
    elif "binding" in fields:  # direct bind swapped, computed material happened to match
        parts.append("binding changed")
    if "materials_within" in fields:
        parts.append("materials within changed")
    if "type" in fields:
        parts.append(f"type {fields['type'][0]}→{fields['type'][1]}")
    if "name" in fields:
        parts.append(f"renamed {fields['name'][0]}→{fields['name'][1]}")
    return ", ".join(parts)


def format_diff(diff: dict, *, structural: bool = False) -> str:
    """Render a diff dict as text. `+` added, `-` removed, `~` changed."""
    lines = []
    for e in diff["added"]:
        lines.append(f'+ {_label(e)} [{e["type"]}] "{e["name"]}"')
    for e in diff["removed"]:
        lines.append(f'- {_label(e)} [{e["type"]}] "{e["name"]}"')
    for e in diff["changed"]:
        lab = e.get("ref") or e["path"]
        body = _structural_fields(e["fields"]) if structural else _summary_fields(e["fields"])
        sep = " " if structural else ": "
        lines.append(f"~ {lab}{sep}{body}")
    return "\n".join(lines) if lines else "(no changes)"
