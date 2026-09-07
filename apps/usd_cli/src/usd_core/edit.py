# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene-manipulation primitives — pure USD authoring, adapted from world_understanding's
`utils/usd/prim.py`/`material.py` and kept to usd-cli's conventions (operate on a live stage +
prim paths; the Session layer resolves @refs, records history, and wraps the Response).

Reversibility choices (so undo is cheap and correct):
  * delete   → deactivate (SetActive False); undo re-activates. Non-destructive, robust for
               referenced/payload prims where RemovePrim can't fully erase a composed spec.
  * duplicate→ a new prim with an *internal reference* to the source: inherits all opinions
               cheaply; undo just removes the new prim.
  * transform→ canonical xformOp:{translate,rotateXYZ,scale}; undo re-authors the prior TRS.
  * rename/reparent → Sdf namespace edit on the editable layer; refs go stale by design
               (usd-cli convention: path changed → re-snapshot).
"""

from __future__ import annotations

import math


# ── transforms ────────────────────────────────────────────────────────────────────
_TRS_ORDER = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]


def _quat_to_euler_xyz(quat) -> list[float]:
    """Quaternion → intrinsic XYZ Euler degrees (matches UsdGeom rotateXYZ)."""
    from pxr import Gf

    if isinstance(quat, Gf.Quatd):
        w = quat.GetReal()
        i = quat.GetImaginary()
        x, y, z = i[0], i[1], i[2]
    else:  # Gf.Quaternion
        w = quat.GetReal()
        x, y, z = quat.GetImaginary()
    # standard XYZ extraction
    sx = 2 * (w * x + y * z)
    cx = 1 - 2 * (x * x + y * y)
    rx = math.atan2(sx, cx)
    sy = 2 * (w * y - z * x)
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    sz = 2 * (w * z + x * y)
    cz = 1 - 2 * (y * y + z * z)
    rz = math.atan2(sz, cz)
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def current_trs(stage, path: str) -> dict:
    """Read a prim's local transform as translate/rotateXYZ(deg)/scale.

    Uses the named TRS ops directly when present (clean, lossless); otherwise decomposes
    the composed local-to-parent matrix.
    """
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    by_name = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    t = list(by_name["xformOp:translate"].Get()) if "xformOp:translate" in by_name and \
        by_name["xformOp:translate"].Get() is not None else None
    r = list(by_name["xformOp:rotateXYZ"].Get()) if "xformOp:rotateXYZ" in by_name and \
        by_name["xformOp:rotateXYZ"].Get() is not None else None
    s = list(by_name["xformOp:scale"].Get()) if "xformOp:scale" in by_name and \
        by_name["xformOp:scale"].Get() is not None else None
    if t is not None and r is not None and s is not None:
        return {"translate": [float(v) for v in t], "rotate": [float(v) for v in r],
                "scale": [float(v) for v in s]}
    # decompose the composed local transform for anything non-canonical (matrix/orient/etc.)
    m = xf.GetLocalTransformation()
    xform = Gf.Transform(m)
    tr = xform.GetTranslation()
    sc = xform.GetScale()
    eul = _quat_to_euler_xyz(xform.GetRotation().GetQuat())
    return {"translate": [tr[0], tr[1], tr[2]] if t is None else [float(v) for v in t],
            "rotate": eul if r is None else [float(v) for v in r],
            "scale": [sc[0], sc[1], sc[2]] if s is None else [float(v) for v in s]}


def set_trs(stage, path: str, translate, rotate, scale) -> None:
    """Author a canonical xformOp:{translate,rotateXYZ,scale} stack on the prim."""
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    op_t = xf.AddTranslateOp()
    op_t.Set(Gf.Vec3d(*[float(v) for v in translate]))
    op_r = xf.AddRotateXYZOp()
    op_r.Set(Gf.Vec3f(*[float(v) for v in rotate]))
    op_s = xf.AddScaleOp()
    op_s.Set(Gf.Vec3f(*[float(v) for v in scale]))


def _apply_axis(cur: float, delta) -> float:
    """delta is {'mode': 'relative'|'absolute', 'value': float} or None."""
    if not delta:
        return cur
    v = float(delta["value"])
    return cur + v if delta.get("mode") == "relative" else v


def apply_transform(stage, path: str, *, tx=None, ty=None, tz=None, rx=None, ry=None,
                    rz=None, sx=None, sy=None, sz=None, translate=None, rotate=None,
                    scale=None) -> tuple[dict, dict]:
    """Apply TRS deltas; returns (before_trs, after_trs) for history/inverse.

    Per-axis args are signed {mode,value} dicts (relative/absolute). The vector args
    (translate/rotate/scale) are absolute overrides applied first.
    """
    before = current_trs(stage, path)
    t = list(translate) if translate else list(before["translate"])
    r = list(rotate) if rotate else list(before["rotate"])
    s = list(scale) if scale else list(before["scale"])
    t = [_apply_axis(t[0], tx), _apply_axis(t[1], ty), _apply_axis(t[2], tz)]
    r = [_apply_axis(r[0], rx), _apply_axis(r[1], ry), _apply_axis(r[2], rz)]
    s = [_apply_axis(s[0], sx), _apply_axis(s[1], sy), _apply_axis(s[2], sz)]
    set_trs(stage, path, t, r, s)
    return before, {"translate": t, "rotate": r, "scale": s}


# ── create ──────────────────────────────────────────────────────────────────────────
_SHAPE_TYPES = {
    "cube": "Cube", "sphere": "Sphere", "cylinder": "Cylinder",
    "cone": "Cone", "capsule": "Capsule", "plane": "Plane",
}
_LIGHT_TYPES = {
    "distant": "DistantLight", "dome": "DomeLight", "sphere": "SphereLight",
    "rect": "RectLight", "disk": "DiskLight", "cylinder": "CylinderLight",
}


def unique_child_path(stage, parent: str, name: str) -> str:
    """A child path under `parent` that doesn't collide (append _2, _3, …)."""
    from pxr import Sdf

    parent = parent.rstrip("/") or ""
    base = f"{parent}/{name}"
    if not stage.GetPrimAtPath(Sdf.Path(base)).IsValid():
        return base
    i = 2
    while stage.GetPrimAtPath(Sdf.Path(f"{base}_{i}")).IsValid():
        i += 1
    return f"{base}_{i}"


def set_translate(stage, path: str, vec) -> None:
    """Set a prim's absolute translate, reusing an existing translate op if present
    (avoids the 'xformOp:translate already exists' error on referenced/copied prims)."""
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            op.Set(Gf.Vec3d(*[float(v) for v in vec]))
            return
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in vec]))


def create_prim(stage, *, type_: str, name: str | None, parent: str, shape: str | None,
                at=None) -> str:
    """Define a new prim; returns its path. `type_` is xform|mesh|light|camera|<shape>."""
    from pxr import Gf, UsdGeom

    t = type_.lower()
    if t in ("xform", "group"):
        type_name = "Xform"
    elif t == "mesh" or t in _SHAPE_TYPES:
        type_name = _SHAPE_TYPES.get(shape or t, "Cube") if (shape or t) in _SHAPE_TYPES else "Cube"
    elif t == "light":
        type_name = _LIGHT_TYPES.get((shape or "distant").lower(), "DistantLight")
    elif t == "camera":
        type_name = "Camera"
    else:
        type_name = type_  # trust a literal USD type name
    base = name or type_name
    path = unique_child_path(stage, parent or "", base)
    prim = stage.DefinePrim(path, type_name)
    if at and prim.IsValid():
        set_translate(stage, path, at)
    return path


# ── delete / restore (reversible via active state) ───────────────────────────────────
def set_active(stage, path: str, active: bool) -> None:
    stage.GetPrimAtPath(path).SetActive(active)


# ── duplicate (internal reference) ───────────────────────────────────────────────────
def duplicate_prim(stage, src_path: str, *, name: str | None = None, at=None) -> str:
    """Deep-copy a prim's composed spec into a new sibling (a true, independent duplicate).

    We flatten then Sdf.CopySpec so the copy carries the source's resolved opinions but is
    NOT a live mirror — later edits to the source don't leak into the copy.
    """
    from pxr import Sdf

    src = stage.GetPrimAtPath(src_path)
    parent = src.GetParent().GetPath().pathString
    base = name or (src.GetName() + "_copy")
    dst_path = unique_child_path(stage, parent, base)
    flat = stage.Flatten()
    edit_layer = stage.GetEditTarget().GetLayer()
    Sdf.CreatePrimInLayer(edit_layer, Sdf.Path(dst_path))
    Sdf.CopySpec(flat, Sdf.Path(src_path), edit_layer, Sdf.Path(dst_path))
    if at:
        set_translate(stage, dst_path, at)
    return dst_path


# ── rename / reparent (Sdf namespace edit on the editable layer) ─────────────────────
def _namespace_edit(stage, edit) -> bool:
    from pxr import Sdf

    layer = stage.GetEditTarget().GetLayer()
    batch = Sdf.BatchNamespaceEdit()
    batch.Add(edit)
    return layer.CanApply(batch) and layer.Apply(batch)


def rename_prim(stage, path: str, new_name: str) -> str:
    from pxr import Sdf

    parent = path.rsplit("/", 1)[0] or ""
    new_path = f"{parent}/{new_name}"
    edit = Sdf.NamespaceEdit.Rename(path, new_name)
    if not _namespace_edit(stage, edit):
        raise RuntimeError(
            f"cannot rename {path}: prim is not defined in the editable layer "
            "(it likely comes from a reference/payload)")
    return new_path


def reparent_prim(stage, path: str, new_parent: str) -> str:
    from pxr import Sdf

    name = path.rsplit("/", 1)[-1]
    new_path = f"{new_parent.rstrip('/')}/{name}"
    edit = Sdf.NamespaceEdit.Reparent(path, new_parent, -1)
    if not _namespace_edit(stage, edit):
        raise RuntimeError(
            f"cannot reparent {path} under {new_parent}: prim is not in the editable layer "
            "(it likely comes from a reference/payload)")
    return new_path


# ── visibility ───────────────────────────────────────────────────────────────────────
def set_visibility(stage, path: str, visible: bool) -> None:
    from pxr import UsdGeom

    img = UsdGeom.Imageable(stage.GetPrimAtPath(path))
    if visible:
        img.MakeVisible()
    else:
        img.MakeInvisible()


# ── generic attribute set ─────────────────────────────────────────────────────────────
def set_relationship(stage, path: str, rel_name: str, value: str):
    """Set a relationship's targets from a comma-separated prim-path list.

    Returns (old_targets, new_targets) for history. Targets must exist on the stage —
    a typo'd path would otherwise author a dangling target that only fails much later.
    """
    prim = stage.GetPrimAtPath(path)
    rel = prim.GetRelationship(rel_name)
    old = [t.pathString for t in rel.GetTargets()] if rel else []
    targets = [t.strip() for t in str(value).split(",") if t.strip()]
    if not targets:
        raise ValueError(f"relationship {rel_name} needs at least one target prim path")
    for t in targets:
        if not t.startswith("/"):
            raise ValueError(f"relationship target must be an absolute prim path: {t}")
        if not stage.GetPrimAtPath(t).IsValid():
            raise ValueError(f"relationship target does not exist: {t}")
    if not rel:
        rel = prim.CreateRelationship(rel_name, custom=True)
    rel.SetTargets(targets)
    return old, targets


def set_attribute(stage, path: str, attr: str, value: str):
    """Set an attribute, coercing the string `value` to the attribute's type.

    Returns (old_value, new_value) for history. Creates the attribute if absent (best-effort
    typed from the literal).
    """
    from pxr import Sdf, Usd

    prim = stage.GetPrimAtPath(path)
    if prim.GetRelationship(attr):
        # e.g. physics:body0 on a joint — authoring an attribute here would shadow the
        # schema relationship, "succeed", and then blow up composition on the next read.
        raise ValueError(
            f"{attr} is a relationship, not an attribute — pass prim path(s) as the value "
            "and it will author relationship targets")
    a = prim.GetAttribute(attr)
    old = a.Get() if a and a.HasAuthoredValue() else None
    coerced = _coerce(value, a.GetTypeName() if a else None)
    created = not (a and a.GetTypeName())
    if created:
        a = prim.CreateAttribute(attr, _infer_type(coerced), custom=True)
    a.Set(coerced)
    return (None if old is None else _jsonsafe(old)), _jsonsafe(a.Get()), created


def _infer_type(value):
    """Best-effort Sdf value type for a Python value (bool before int — bool subclasses int)."""
    from pxr import Sdf

    if isinstance(value, bool):
        return Sdf.ValueTypeNames.Bool
    if isinstance(value, float):
        return Sdf.ValueTypeNames.Double
    if isinstance(value, int):
        return Sdf.ValueTypeNames.Int
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return Sdf.ValueTypeNames.Float3
    return Sdf.ValueTypeNames.String


def set_raw_attr(stage, path: str, attr: str, value) -> None:
    """Set an already-coerced value, re-creating the attribute if it lacks a declared type.

    Used by undo/redo: a prior undo may have removed a custom attribute's spec entirely, so
    redo must re-declare it (inferring the type from the stored value) before setting.
    """
    prim = stage.GetPrimAtPath(path)
    a = prim.GetAttribute(attr)
    if not a or not a.GetTypeName():
        a = prim.CreateAttribute(attr, _infer_type(value), custom=True)
    a.Set(value)


def _coerce(value: str, type_name=None):
    """Coerce a CLI string into a Python/USD value (best-effort, type-hinted)."""
    from pxr import Gf

    s = value.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if "," in s:
        # tolerate wrapping parens/brackets: `set @n1 xformOp:orient "(1,0,0,0)"`
        parts = [float(x) for x in s.strip("()[] ").split(",")]
        tn = str(type_name).lower() if type_name is not None else ""
        if "quat" in tn and len(parts) == 4:
            # USD order: real (w) first — matches usdview/usda display
            w, x, y, z = parts
            if "quatd" in tn:
                return Gf.Quatd(w, Gf.Vec3d(x, y, z))
            if "quath" in tn:
                return Gf.Quath(w, Gf.Vec3h(x, y, z))
            return Gf.Quatf(w, Gf.Vec3f(x, y, z))
        if type_name is not None and "2" in str(type_name) and len(parts) == 2:
            return Gf.Vec2f(*parts)
        if type_name is not None and "3" in str(type_name):
            return Gf.Vec3f(*parts[:3])
        if type_name is not None and "4" in str(type_name) and len(parts) == 4:
            return Gf.Vec4f(*parts)
        return parts
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        return s


# ── applied API schema removal ────────────────────────────────────────────────────────
def _registered_api_schema_names() -> list[str]:
    """All applied-API schema type names this USD build's registry knows about."""
    from pxr import Tf, Usd

    base = Tf.Type.Find(Usd.APISchemaBase)
    return sorted({name for name in (Usd.SchemaRegistry.GetSchemaTypeName(t)
                                     for t in base.GetAllDerivedTypes()) if name})


def _resolve_api_name(prim, name: str) -> tuple[str, str | None]:
    """Resolve a user-supplied API name to (schema_type_name, instance|None) on `prim`.

    Accepts the registered type name ("PhysicsRigidBodyAPI"), a short suffix
    ("RigidBodyAPI", "MassAPI"), any casing of either, and "SchemaAPI:instance" for
    multiple-apply schemas. Matches against the prim's *applied* schemas — the only
    thing removable — so short names stay unambiguous per prim; the full registry is
    consulted only to tell "not applied here" apart from "no such schema".
    """
    want, want_inst = name, None
    if ":" in name:
        want, want_inst = name.split(":", 1)
    low = want.lower()
    matches: set[tuple[str, str | None]] = set()
    for entry in prim.GetAppliedSchemas():
        base, inst = entry, None
        if ":" in entry:
            base, inst = entry.split(":", 1)
        if want_inst is not None and inst != want_inst:
            continue
        if base.lower() == low or base.lower().endswith(low):
            matches.add((base, inst))
    if len(matches) == 1:
        return next(iter(matches))
    applied = list(prim.GetAppliedSchemas())
    if len(matches) > 1:
        opts = sorted(f"{b}:{i}" if i else b for b, i in matches)
        raise ValueError(f"API name '{name}' is ambiguous on {prim.GetPath()}: "
                         f"matches {opts} — pass the full schema name")
    known = [n for n in _registered_api_schema_names()
             if n.lower() == low or n.lower().endswith(low)]
    if known:
        raise ValueError(f"{' / '.join(known)} is not applied on {prim.GetPath()} "
                         f"(applied: {applied or 'none'})")
    raise ValueError(f"unknown API schema '{name}' "
                     f"(applied on {prim.GetPath()}: {applied or 'none'})")


def _capture_property(prim, prop_name: str) -> dict:
    """Snapshot one authored property (typed value, or relationship targets) so the
    Session can register a REAL remove-api undo: re-apply the schema, re-author these."""
    rel = prim.GetRelationship(prop_name)
    if rel:
        return {"name": prop_name, "kind": "relationship",
                "targets": [t.pathString for t in rel.GetTargets()]}
    attr = prim.GetAttribute(prop_name)
    had = attr.HasAuthoredValue()
    return {"name": prop_name, "kind": "attribute", "type": str(attr.GetTypeName()),
            "custom": attr.IsCustom(), "had_value": had,
            "value": attr.Get() if had else None}


def _surviving_opinion_layer(prim, prop_name: str) -> str | None:
    """After RemoveProperty, the strongest layer still contributing an opinion for
    `prop_name` — or None when the property is really gone. RemoveProperty only deletes
    the edit target's spec; weaker opinions (from references/payloads) survive and keep
    resolving, so reporting such a property as 'removed' would be a lie."""
    from pxr import Usd

    prop = prim.GetProperty(prop_name)
    if not prop or not prop.IsValid():
        return None
    specs = prop.GetPropertyStack(Usd.TimeCode.Default())
    if not specs:
        return None
    return specs[0].layer.identifier


def remove_api(stage, path: str, api_name: str) -> dict:
    """Remove an applied API schema from a prim, including the properties it owns.

    `Usd.Prim.RemoveAPI` alone only edits the apiSchemas listing — the schema's authored
    attributes/relationships would linger as stale opinions the next consumer still
    composes — so the schema's declared properties (and their namespaced children, e.g.
    material:binding:physics under MaterialBindingAPI) are deleted too. `api_name`
    accepts short names ("RigidBodyAPI") and any casing; multiple-apply instances use
    the "SchemaAPI:instance" form.

    Returns {path, api, removed, properties_removed, properties_masked, captured}:
    `properties_masked` lists properties whose opinion still RESOLVES after removal (a
    weaker referenced opinion survives — the edit-target spec is gone but the value is
    only masked, not removed); `captured` is the pre-removal state (schema, instance,
    property snapshots) the Session records for a real undo, or None if capture failed.
    """
    from pxr import Usd

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"cannot remove API: no prim at {path}")
    schema, instance = _resolve_api_name(prim, api_name.strip())
    tf_type = Usd.SchemaRegistry.GetTypeFromSchemaTypeName(schema)
    # the property names this schema owns (instantiated for multiple-apply instances)
    prim_def = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema)
    owned = list(prim_def.GetPropertyNames()) if prim_def else []
    if instance:
        owned = [Usd.SchemaRegistry.MakeMultipleApplyNameInstance(n, instance)
                 for n in owned]
    if schema == "MaterialBindingAPI":
        # binding properties are per-purpose and dynamic — not in the prim definition
        owned.append("material:binding")
    to_remove = [prop.GetName() for prop in prim.GetAuthoredProperties()
                 if any(prop.GetName() == o or prop.GetName().startswith(o + ":")
                        for o in owned)]
    # capture the pre-removal state BEFORE mutating anything, so undo can re-apply the
    # schema and its property opinions for real (best-effort: None disables real undo
    # and the Session records an honest non-undoable marker instead)
    captured: dict | None
    try:
        captured = {"schema": schema, "instance": instance,
                    "properties": [_capture_property(prim, n) for n in to_remove]}
    except Exception:  # noqa: BLE001 — capture must never block the removal itself
        captured = None
    removed = prim.RemoveAPI(tf_type, instance) if instance else prim.RemoveAPI(tf_type)
    deleted: list[str] = []
    masked: list[dict] = []
    for prop_name in to_remove:
        dropped = prim.RemoveProperty(prop_name)
        layer = _surviving_opinion_layer(prim, prop_name)
        if layer is not None:
            # the value still resolves — a weaker opinion survives in `layer`
            masked.append({"name": prop_name, "layer": layer})
        elif dropped:
            deleted.append(prop_name)
    return {"path": path, "api": f"{schema}:{instance}" if instance else schema,
            "removed": bool(removed), "properties_removed": sorted(deleted),
            "properties_masked": sorted(masked, key=lambda m: m["name"]),
            "captured": captured}


def _jsonsafe(v):
    """Convert a USD/Gf value to a JSON-serializable form for Response payloads.

    Non-finite floats become strings: physics schema defaults like
    physics:breakForce = inf crashed FastAPI's strict JSON encoder with a 500
    the client then mislabeled "unreachable" (round 9, task-13 — 11 daemon
    tracebacks from one multi-ATTR properties sweep)."""
    import math
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        np = None
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)  # "inf" / "-inf" / "nan"
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if np is not None and isinstance(v, np.generic):
        return _jsonsafe(v.item())
    if hasattr(v, "__len__") and not isinstance(v, str):
        try:
            return [_jsonsafe(x) for x in v]
        except TypeError:
            return str(v)
    return str(v)
