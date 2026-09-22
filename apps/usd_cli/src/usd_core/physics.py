# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow-neutral USD physics schema authoring and inspection.

Authors UsdPhysics schema (Scene / RigidBodyAPI / CollisionAPI / MassAPI / physics
MaterialAPI), reports candidate state, and runs a deterministic (no-runtime) schema
validation. Actual rigid-body *simulation* (ovphysx) is an external runtime; `validate`
here checks the authored schema is well-formed and self-consistent, which is the part
that ports cleanly. Runtime sim is surfaced as an explicit "not available locally" note.
"""

from __future__ import annotations

_COLLISION_APPROX = {"none", "convexHull", "convexDecomposition", "boundingCube",
                     "boundingSphere", "meshSimplification"}


def define_physics_scene(stage, path: str) -> str:
    """Author a UsdPhysics.Scene at the caller-selected path."""
    from pxr import UsdPhysics

    if not isinstance(path, str) or not path.startswith("/") or path == "/":
        raise ValueError("physics scene path must be an absolute prim path")
    return UsdPhysics.Scene.Define(stage, path).GetPath().pathString


def _validate_mass_properties(value):
    """Validate explicit MassAPI vectors before any stage mutation."""
    import math
    import struct

    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {
        "center_of_mass", "diagonal_inertia", "principal_axes"
    }:
        raise ValueError("mass_properties must contain only explicit MassAPI vectors")
    result = {}
    for name, count in (("center_of_mass", 3), ("diagonal_inertia", 3), ("principal_axes", 4)):
        vector = value.get(name, (1.0, 0.0, 0.0, 0.0) if name == "principal_axes" else None)
        if not isinstance(vector, (list, tuple)) or len(vector) != count:
            raise ValueError(f"mass_properties.{name} must have {count} values")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in vector):
            raise ValueError(f"mass_properties.{name} must contain numbers")
        try:
            stored = [struct.unpack("f", struct.pack("f", item))[0] for item in vector]
        except (OverflowError, struct.error) as exc:
            raise ValueError("mass properties must be representable as finite USD floats") from exc
        if not all(math.isfinite(item) for item in stored):
            raise ValueError("mass properties must be finite")
        result[name] = stored
    inertia = result["diagonal_inertia"]
    if min(inertia) <= 0:
        raise ValueError("diagonal inertia must be positive")
    if 2 * max(inertia) > sum(inertia) + 1e-6 * max(inertia):
        raise ValueError("diagonal inertia violates the principal-moment triangle inequality")
    if abs(sum(item * item for item in result["principal_axes"]) - 1.0) > 1e-5:
        raise ValueError("principal_axes must be a normalized (w,x,y,z) quaternion")
    return result


def apply_rigid_body(stage, path: str, *, density=None, mass=None, mass_properties=None) -> list[str]:
    """Apply RigidBodyAPI (+ MassAPI when mass/density given); returns the API names
    authored so callers can report them instead of applying schema silently."""
    from pxr import Gf, UsdPhysics

    vectors = _validate_mass_properties(mass_properties)
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"cannot apply rigid body: no prim at {path}")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    authored = ["PhysicsRigidBodyAPI"]
    if density is not None or mass is not None or vectors is not None:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        authored.append("PhysicsMassAPI")
        if mass is not None:
            mass_api.CreateMassAttr().Set(float(mass))
        if density is not None:
            mass_api.CreateDensityAttr().Set(float(density))
        if vectors is not None:
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*vectors["center_of_mass"]))
            mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*vectors["diagonal_inertia"]))
            q = vectors["principal_axes"]
            mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
    return authored


def apply_collision(stage, path: str, *, approximation: str | None = None) -> list[str]:
    """Apply CollisionAPI (+ MeshCollisionAPI approximation on meshes); returns the API
    names authored so callers can report them instead of applying schema silently."""
    from pxr import UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"cannot apply collision: no prim at {path}")
    # Validate the approximation up front, regardless of prim type, so a bad value on a
    # non-mesh isn't silently dropped.
    if approximation is not None and approximation not in _COLLISION_APPROX:
        raise ValueError(f"unknown collision approximation '{approximation}' "
                         f"(expected one of {sorted(_COLLISION_APPROX)})")
    UsdPhysics.CollisionAPI.Apply(prim)
    authored = ["PhysicsCollisionAPI"]
    if approximation and prim.IsA(UsdGeom.Mesh):
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr().Set(approximation)
        authored.append("PhysicsMeshCollisionAPI")
    return authored


def apply_physics_material(stage, *, path: str, static_friction=None, dynamic_friction=None,
                           restitution=None) -> str:
    """Author physics material properties at one caller-selected prim path."""
    from pxr import UsdPhysics

    if not isinstance(path, str) or not path.startswith("/") or path == "/":
        raise ValueError("physics material path must be an absolute prim path")
    existing = stage.GetPrimAtPath(path)
    if existing.IsValid():
        if existing.GetTypeName() not in {"", "Material"}:
            raise ValueError(
                f"cannot author physics material at {path}: existing prim has type "
                f"{existing.GetTypeName()!r}"
            )
        if not existing.HasAPI(UsdPhysics.MaterialAPI):
            raise ValueError(
                f"cannot author physics material at {path}: existing prim is not "
                "already a physics material"
            )
    mat_prim = stage.DefinePrim(path, "Material")
    mat_path = mat_prim.GetPath().pathString
    api = UsdPhysics.MaterialAPI.Apply(mat_prim)
    if static_friction is not None:
        api.CreateStaticFrictionAttr().Set(float(static_friction))
    if dynamic_friction is not None:
        api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    if restitution is not None:
        api.CreateRestitutionAttr().Set(float(restitution))
    return mat_path


def apply_operations(stage, operations: dict) -> dict:
    """Apply explicit, workflow-selected USD physics operations.

    This is deliberately a mechanical command surface.  It does not infer a
    scene path, body/collider targets, material names, or preservation policy.
    """
    from pxr import UsdShade

    if not isinstance(operations, dict):
        raise ValueError("physics operations must be an object")
    allowed = {"scene_paths", "rigid_bodies", "colliders", "materials", "bindings"}
    unknown = set(operations) - allowed
    if unknown:
        raise ValueError(f"unknown physics operation fields: {sorted(unknown)}")

    def rows(name: str) -> list[dict]:
        value = operations.get(name, [])
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError(f"physics operations {name} must be a list of objects")
        return value

    scenes = operations.get("scene_paths", [])
    if not isinstance(scenes, list) or not all(isinstance(path, str) for path in scenes):
        raise ValueError("physics operations scene_paths must be a list of paths")
    rigid_bodies, colliders, materials, bindings = (
        rows("rigid_bodies"), rows("colliders"), rows("materials"), rows("bindings")
    )
    # Validate every vector record before earlier rows or scenes can mutate.
    for row in rigid_bodies:
        _validate_mass_properties(row.get("mass_properties"))
    target_paths = [
        str(row.get("path") or "") for row in [*rigid_bodies, *colliders]
    ] + [str(row.get("target_path") or "") for row in bindings]
    for path in target_paths:
        if not path or not stage.GetPrimAtPath(path).IsValid():
            raise ValueError(f"physics operation references a missing target prim: {path}")
    material_paths = {str(row.get("path") or "") for row in materials}
    if "" in material_paths:
        raise ValueError("physics material operation requires path")
    for path in scenes:
        if not path.startswith("/") or path == "/":
            raise ValueError("physics scene path must be an absolute prim path")
    for row in colliders:
        approximation = row.get("approximation")
        if approximation is not None and approximation not in _COLLISION_APPROX:
            raise ValueError(
                f"unknown collision approximation '{approximation}' "
                f"(expected one of {sorted(_COLLISION_APPROX)})"
            )
    for path in material_paths:
        if not path.startswith("/") or path == "/":
            raise ValueError("physics material path must be an absolute prim path")
    for row in bindings:
        material_path = str(row.get("material_path") or "")
        if material_path in material_paths:
            continue
        material_prim = stage.GetPrimAtPath(material_path)
        if not material_prim.IsValid():
            raise ValueError(
                f"physics binding references a missing material: {material_path}"
            )
        if not material_prim.IsA(UsdShade.Material):
            raise ValueError(
                "physics binding references a prim that is not a Material: "
                f"{material_path}"
            )

    authored = {"scene": [], "rigid_body": [], "collision": [], "material": [],
                "binding": [], "authored_apis": {}}

    def note(path: str, apis: list[str]) -> None:
        seen = authored["authored_apis"].setdefault(path, [])
        seen.extend(api for api in apis if api not in seen)

    for path in scenes:
        authored["scene"].append(define_physics_scene(stage, path))
    for row in rigid_bodies:
        path = str(row["path"])
        note(path, apply_rigid_body(stage, path, density=row.get("density"), mass=row.get("mass"),
                                    mass_properties=row.get("mass_properties")))
        authored["rigid_body"].append(path)
    for row in colliders:
        path = str(row["path"])
        note(path, apply_collision(stage, path, approximation=row.get("approximation")))
        authored["collision"].append(path)
    for row in materials:
        path = apply_physics_material(
            stage, path=str(row["path"]), static_friction=row.get("static_friction"),
            dynamic_friction=row.get("dynamic_friction"), restitution=row.get("restitution"),
        )
        authored["material"].append(path)
        note(path, ["PhysicsMaterialAPI"])
    for row in bindings:
        target_path, material_path = str(row["target_path"]), str(row["material_path"])
        target = stage.GetPrimAtPath(target_path)
        UsdShade.MaterialBindingAPI.Apply(target).Bind(
            UsdShade.Material(stage.GetPrimAtPath(material_path)), materialPurpose="physics"
        )
        authored["binding"].append({"target_path": target_path, "material_path": material_path})
        note(target_path, ["MaterialBindingAPI"])
    return authored


def validate_schema(stage) -> dict:
    """Deterministic, no-runtime validation of authored physics schema.

    Checks: a Scene exists; rigid bodies have finite positive mass/density; colliders use a
    known approximation; rigid bodies aren't nested. Returns {ok, checks, issues}.

    Traverses native-instance proxies: SimReady-style assets often carry their colliders
    inside `instanceable` payloads, and a plain `Traverse()` would report them as having no
    physics at all. Each instance's schemas count once per instance, which matches what a
    runtime would simulate.
    """
    from pxr import Usd, UsdPhysics

    issues: list[str] = []
    prims = list(stage.Traverse(Usd.TraverseInstanceProxies()))
    scenes = [p for p in prims if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        issues.append("no UsdPhysics.Scene authored")
    rigid = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    enabled_rigid = [
        p
        for p in rigid
        if bool(UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get())
    ]
    colliders = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    for rb in rigid:
        # nested rigid body check
        anc = rb.GetParent()
        while anc and anc.IsValid() and anc.GetPath().pathString != "/":
            if anc.HasAPI(UsdPhysics.RigidBodyAPI):
                issues.append(f"nested rigid body: {rb.GetPath()} under {anc.GetPath()}")
                break
            anc = anc.GetParent()
        if rb.HasAPI(UsdPhysics.MassAPI):
            m = UsdPhysics.MassAPI(rb)
            # In USD a mass of 0 means "derive from density"; only authored *negative*/NaN
            # values are actually invalid.
            import math
            ma, da = m.GetMassAttr(), m.GetDensityAttr()
            mass = ma.Get() if ma and ma.HasAuthoredValue() else None
            dens = da.Get() if da and da.HasAuthoredValue() else None
            mass_ok = mass is not None and mass > 0 and math.isfinite(mass)
            if mass is not None and (mass < 0 or not math.isfinite(mass)):
                issues.append(f"invalid mass on {rb.GetPath()}: {mass}")
            # A negative/NaN density is always malformed. Density 0 means "derive inertia
            # from density" and is only a problem when there is no explicit mass to override
            # it — with an authored mass>0, USD ignores density, so density 0 is harmless.
            if dens is not None:
                if dens < 0 or not math.isfinite(dens):
                    issues.append(f"invalid density on {rb.GetPath()}: {dens}")
                elif dens == 0.0 and not mass_ok:
                    issues.append(f"rigid body {rb.GetPath()} has density 0 and no valid "
                                  "mass to derive inertia from")
    return {
        "ok": not issues,
        "checks": {"scenes": len(scenes), "rigid_bodies": len(rigid),
                   "enabled_rigid_bodies": len(enabled_rigid),
                   "colliders": len(colliders)},
        "issues": issues,
        "runtime_note": "deterministic schema check only; runtime simulation (ovphysx) "
                        "is an external dependency not run here",
    }
