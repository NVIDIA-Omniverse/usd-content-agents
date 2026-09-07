# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bulk prim selection for rule-based commands (`material --all --type Mesh --where …`).

A selection is the intersection of:
  * a hierarchy scope   (`--under @ref` — STRICT descendants of that root; the anchor
                         prim itself is never a candidate, so `--under @asm --where
                         'name~=P*'` can't bulk-bind the assembly root. default: whole
                         stage),
  * prim-type filters   (`--type Mesh`, repeatable, case-insensitive),
  * rule expressions    (`--where 'name~=Conductor*'`, repeatable, ANDed).

Rule syntax is `key OP value` with keys
    name | path | type | kind | purpose | material | attr:<attrName>
and operators
    ~=  glob match (fnmatchcase: case-sensitive, matches the full value)
    ==  equals (case-insensitive; `=` accepted)
    !=  not equal
    *=  contains (case-insensitive)

Traverses native-instance proxies so instanced geometry is selectable; callers decide
where the edit is actually authored (instance root vs prototype source).
"""

from __future__ import annotations

import fnmatch
import re

# Keep repeated character classes disjoint. The previous ``\s*(.*?)\s*$``
# tail let whitespace be partitioned among three repetitions and was flagged
# for polynomial backtracking. Capture the remainder once and trim it in code.
_RULE = re.compile(r"^\s*([A-Za-z_][\w:.]*)\s*(~=|==|!=|\*=|=)([^\r\n]*)$")


def parse_rule(expr: str):
    """Parse one `--where` expression into (key, op, value); raises on bad syntax."""
    m = _RULE.match(expr)
    if not m:
        raise ValueError(
            f"bad --where expression '{expr}' — expected key~=Glob*, key==value, "
            "key!=value or key*=substring (keys: name, path, type, kind, purpose, "
            "material, attr:<name>)")
    key, op, value = m.group(1), m.group(2), m.group(3).strip()
    return key, ("==" if op == "=" else op), value


def _rule_value(stage, prim, key: str):
    """The prim's raw string value for a rule key, or None."""
    if key == "name":
        return prim.GetName()
    if key == "path":
        return prim.GetPath().pathString
    if key == "type":
        return str(prim.GetTypeName())
    if key == "kind":
        from pxr import Usd
        return Usd.ModelAPI(prim).GetKind() or ""
    if key == "purpose":
        from pxr import UsdGeom
        img = UsdGeom.Imageable(prim)
        return str(img.ComputePurpose()) if img else ""
    if key == "material":
        from usd_core.materials import bound_material
        return (bound_material(stage, prim.GetPath().pathString)["bound_material_path"]
                or "")
    if key.startswith("attr:"):
        attr = prim.GetAttribute(key[5:])
        if not attr or not attr.HasValue():
            return None
        return str(attr.Get())
    raise ValueError(f"unknown --where key '{key}' (use name, path, type, kind, purpose, "
                     "material, or attr:<name>)")


_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _numbers_of(s: str) -> list[float] | None:
    """Every number in the string, or None when it has none."""
    hits = _NUM.findall(s)
    return [float(h) for h in hits] if hits else None


def _numeric_equal(got: str, want: str) -> bool | None:
    """Tolerant number-sequence equality for attr: rules, or None (not numeric).

    `attr:primvars:displayColor==(0.01, 0.01, 0.01)` must not hinge on float
    formatting or bracket style (round 8: "floating-point string equality is
    too strict for black/orange") — compare the numbers, not the repr."""
    import math

    a, b = _numbers_of(got), _numbers_of(want)
    if not a or not b or len(a) != len(b):
        return None
    return all(math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-6) for x, y in zip(a, b))


def _rule_matches(stage, prim, rule) -> bool:
    key, op, want = rule
    got = _rule_value(stage, prim, key)
    if key == "material":
        want = want.lstrip("@")
    if got is None:
        return op == "!="
    if op == "~=":
        # Strict glob: case-sensitive, anchored to the full value. A lowercase
        # `T*` must not sweep up `tn__…` prims (wu task-03 over-match).
        return fnmatch.fnmatchcase(got, want)
    if key.startswith("attr:") and op in ("==", "!="):
        num = _numeric_equal(got, want)
        if num is not None:
            return num if op == "==" else not num
    got, want = got.lower(), want.lower()
    if op == "==":
        return got == want
    if op == "!=":
        return got != want
    if op == "*=":
        return want in got
    return False


def select_prims(stage, *, types: list[str] | None = None, where: list[str] | None = None,
                 under: str | None = None) -> list:
    """All active prims matching the type/rule filters inside the `under` subtree.

    With no type filter, candidates are actual geometry (Gprims) — the sensible target
    set for bulk material work; an explicit `--type` widens/narrows to that prim type.
    The `under` root itself is excluded (subtree = strict descendants), so a name rule
    that happens to match the anchor never edits it. Returns the matching Usd.Prim
    objects (proxies included, in stage order).
    """
    from pxr import Usd, UsdGeom

    rules = [parse_rule(w) for w in (where or [])]
    type_filter = {t.lower() for t in types} if types else None

    root = stage.GetPrimAtPath(under) if under else stage.GetPseudoRoot()
    if under and not root.IsValid():
        raise ValueError(f"--under prim {under} does not exist")

    out = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or prim.GetPath().pathString == "/":
            continue
        if under and prim.GetPath() == root.GetPath():
            continue  # the anchor is the scope, not a candidate (strict descendants)
        if type_filter:
            if str(prim.GetTypeName()).lower() not in type_filter:
                continue
        elif not prim.IsA(UsdGeom.Gprim):
            continue
        if all(_rule_matches(stage, prim, r) for r in rules):
            out.append(prim)
    return out


def instance_root_of(prim):
    """The enclosing instance root of a proxy (the first non-proxy ancestor); the prim
    itself when it is not an instance proxy."""
    while prim.IsInstanceProxy():
        prim = prim.GetParent()
    return prim


def matches_rules(stage, prim, where: list[str] | None) -> bool:
    """True when `prim` itself satisfies every --where expression."""
    return all(_rule_matches(stage, prim, parse_rule(w)) for w in (where or []))
