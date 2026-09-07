# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material authoring + binding — adapted from world_understanding's `utils/usd/material.py`,
trimmed to the low-level operations exposed by usd-cli's `material` verb:

  * create a UsdPreviewSurface material (diffuse/metallic/roughness) and bind it,
  * create an MDL material (OmniPBR by default) with constants + texture maps and bind it,
  * bind an existing material prim (a @m ref / path) to a prim,
  * read a prim's bound-material state (for queries / verification),
  * describe a material's surface shader (preview vs MDL, module, inputs),
  * nullify (unbind) a prim's material.

UsdShade requires applying MaterialBindingAPI before Bind() — we do that consistently.
"""

from __future__ import annotations

import hashlib


def _looks_scope(stage) -> str:
    """Ensure a `/Looks` (or <defaultPrim>/Looks) Scope exists; return its path."""
    from pxr import UsdGeom

    dp = stage.GetDefaultPrim()
    base = dp.GetPath().pathString if dp and dp.IsValid() and dp.GetPath().pathString != "/" else ""
    looks = f"{base}/Looks" if base else "/Looks"
    if not stage.GetPrimAtPath(looks).IsValid():
        UsdGeom.Scope.Define(stage, looks)
    return looks


def _material_define_path(stage, parent: str, name: str) -> str:
    """The exact path a material named `name` is authored at under `parent`.

    Binding relationships frequently pre-exist their material — scene repair means
    authoring the material AT the path the dangling bindings already target. Authoring
    at a suffixed sibling (`<Name>_2`) instead would leave those bindings dangling
    forever, silently. So the exact path is used when it is free, holds an untyped
    prim (an `over` / bare `def` — converted in place), or already IS a Material
    (updated in place). Any other prim type is an error — never a silent suffix.
    """
    from pxr import Sdf, Tf, UsdShade

    path = f"{parent.rstrip('/')}/{name}"
    if not Sdf.Path.IsValidPathString(path):
        raise ValueError(f"'{name}' is not a valid material name — try "
                         f"'{Tf.MakeValidIdentifier(name)}'")
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid() or UsdShade.Material(prim) or not str(prim.GetTypeName()):
        return path
    raise ValueError(
        f"{path} already exists as a {prim.GetTypeName()} prim — cannot author a "
        "material there. Pick a different --name (materials are never silently "
        "renamed to a suffixed path).")


def _st_output(stage, mat_path: str, uv_set: str | None, tex_scale=None,
               tex_rotate=None, tex_translate=None):
    """UV network for textures: a primvar reader (which UV set) plus an optional
    UsdTransform2d when a texture transform is requested. Returns the Float2 output
    the textures' `st` inputs connect to."""
    from pxr import Gf, Sdf, UsdShade

    reader = UsdShade.Shader.Define(stage, f"{mat_path}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(uv_set or "st")
    out = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    if tex_scale is None and tex_rotate is None and tex_translate is None:
        return out
    xform = UsdShade.Shader.Define(stage, f"{mat_path}/uvTransform")
    xform.CreateIdAttr("UsdTransform2d")
    xform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(out)
    if tex_scale is not None:
        xform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(*[float(v) for v in tex_scale]))
    if tex_rotate is not None:
        xform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(float(tex_rotate))
    if tex_translate is not None:
        xform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(*[float(v) for v in tex_translate]))
    return xform.CreateOutput("result", Sdf.ValueTypeNames.Float2)


def _uv_texture(stage, mat_path: str, slot: str, file_path: str, st_out, *,
                raw: bool = False, normal_map: bool = False):
    """One UsdUVTexture node wired to the shared st network. Scalar maps (roughness /
    metallic) should read the `r` output; color maps `rgb`. Normal maps are decoded from
    [0,1] to [-1,1] tangent space (raw colorspace, scale/bias)."""
    from pxr import Gf, Sdf, UsdShade

    tex = UsdShade.Shader.Define(stage, f"{mat_path}/{slot}Tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(file_path)))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    if raw or normal_map:
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    if normal_map:
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 2))
        tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, -1))
    return tex


def create_preview_material(stage, *, name: str, color=None, metallic: float | None = None,
                            roughness: float | None = None, opacity: float | None = None,
                            clearcoat: float | None = None,
                            clearcoat_roughness: float | None = None,
                            ior: float | None = None, emissive=None,
                            diffuse_texture: str | None = None,
                            normal_texture: str | None = None,
                            roughness_texture: str | None = None,
                            metallic_texture: str | None = None,
                            uv_set: str | None = None, tex_scale=None, tex_rotate=None,
                            tex_translate=None, inputs: dict | None = None) -> str:
    """Define a UsdPreviewSurface material under Looks; return the material prim path.

    Covers the full physically-based surface: constants (diffuse/metallic/roughness/
    opacity/clearcoat/IOR/emissive) and texture maps (albedo, normal, roughness,
    metallic) wired through a primvar reader for `uv_set` with an optional
    UsdTransform2d (tex_scale / tex_rotate degrees / tex_translate). `inputs` passes
    raw UsdPreviewSurface attributes through (e.g. {"opacityThreshold": 0.5}).
    """
    from pxr import Gf, Sdf, UsdShade

    looks = _looks_scope(stage)
    mat_path = _material_define_path(stage, looks, name)
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    if color is not None:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*[float(c) for c in color]))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
        float(metallic) if metallic is not None else 0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        float(roughness) if roughness is not None else 0.5)
    for attr, value in (("opacity", opacity), ("clearcoat", clearcoat),
                        ("clearcoatRoughness", clearcoat_roughness), ("ior", ior)):
        if value is not None:
            shader.CreateInput(attr, Sdf.ValueTypeNames.Float).Set(float(value))
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*[float(c) for c in emissive]))
    for attr, value in (inputs or {}).items():
        if isinstance(value, (list, tuple)):
            shader.CreateInput(attr, Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*[float(c) for c in value]))
        elif isinstance(value, bool):
            shader.CreateInput(attr, Sdf.ValueTypeNames.Bool).Set(value)
        else:
            shader.CreateInput(attr, Sdf.ValueTypeNames.Float).Set(float(value))

    if any(t for t in (diffuse_texture, normal_texture, roughness_texture, metallic_texture)):
        st_out = _st_output(stage, mat_path, uv_set, tex_scale, tex_rotate, tex_translate)
        if diffuse_texture:
            tex = _uv_texture(stage, mat_path, "diffuse", diffuse_texture, st_out)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
        if normal_texture:
            tex = _uv_texture(stage, mat_path, "normal", normal_texture, st_out, normal_map=True)
            shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
        if roughness_texture:
            tex = _uv_texture(stage, mat_path, "roughness", roughness_texture, st_out, raw=True)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
                tex.CreateOutput("r", Sdf.ValueTypeNames.Float))
        if metallic_texture:
            tex = _uv_texture(stage, mat_path, "metallic", metallic_texture, st_out, raw=True)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
                tex.CreateOutput("r", Sdf.ValueTypeNames.Float))

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat_path


# OmniPBR's constant-input names (the de-facto MDL material on the NVIDIA stack). The named
# kwargs of create_mdl_material map onto these; other modules can still be authored via
# `inputs=` with raw attribute names.
_OMNIPBR_INPUTS = {
    "color": ("diffuse_color_constant", "color"),
    "metallic": ("metallic_constant", "float"),
    "roughness": ("reflection_roughness_constant", "float"),
    "opacity": ("opacity_constant", "float"),
    "emissive": ("emissive_color", "color"),
    "diffuse_texture": ("diffuse_texture", "asset"),
    "normal_texture": ("normalmap_texture", "asset"),
    "orm_texture": ("ORM_texture", "asset"),
}


def _set_mdl_input(shader, name: str, kind: str, value) -> None:
    """Create + set one MDL shader input, coercing `value` to the USD type for `kind`."""
    from pxr import Gf, Sdf, UsdShade

    types = {"color": Sdf.ValueTypeNames.Color3f, "float": Sdf.ValueTypeNames.Float,
             "asset": Sdf.ValueTypeNames.Asset, "bool": Sdf.ValueTypeNames.Bool,
             "int": Sdf.ValueTypeNames.Int}
    if kind == "color":
        value = Gf.Vec3f(*[float(c) for c in value])
    elif kind == "float":
        value = float(value)
    elif kind == "asset":
        value = Sdf.AssetPath(str(value))
    elif kind == "bool":
        value = bool(value)
    elif kind == "int":
        value = int(value)
    shader.CreateInput(name, types[kind]).Set(value)


def create_mdl_material(stage, *, name: str, module: str = "OmniPBR.mdl",
                        subidentifier: str | None = None, color=None,
                        metallic: float | None = None, roughness: float | None = None,
                        opacity: float | None = None, emissive=None,
                        diffuse_texture: str | None = None,
                        normal_texture: str | None = None, orm_texture: str | None = None,
                        inputs: dict | None = None) -> str:
    """Define an MDL material (OmniPBR by default) under Looks; return the material prim path.

    `module` is the MDL asset (e.g. "OmniPBR.mdl", "OmniGlass.mdl"); `subidentifier` is the
    material name inside it (defaults to the module stem). The named kwargs map onto OmniPBR's
    constant inputs (diffuse_color_constant / reflection_roughness_constant / metallic_constant
    / opacity_constant) and its texture inputs (diffuse_texture / normalmap_texture /
    ORM_texture). For other modules, pass raw `inputs={name: value}` — colors as 3-seqs, scalars
    as floats, texture paths as strings ending in an image extension.
    """
    from pxr import Sdf, UsdShade

    looks = _looks_scope(stage)
    mat_path = _material_define_path(stage, looks, name)
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    # MDL is sourced from an asset module + subIdentifier, not a built-in shader id.
    node_def = UsdShade.NodeDefAPI.Apply(shader.GetPrim())
    node_def.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    node_def.SetSourceAsset(Sdf.AssetPath(module), "mdl")
    sub = subidentifier or module.rsplit("/", 1)[-1].removesuffix(".mdl")
    node_def.SetSourceAssetSubIdentifier(sub, "mdl")

    named = {"color": color, "metallic": metallic, "roughness": roughness,
             "opacity": opacity, "emissive": emissive, "diffuse_texture": diffuse_texture,
             "normal_texture": normal_texture, "orm_texture": orm_texture}
    for key, value in named.items():
        if value is None:
            continue
        attr, kind = _OMNIPBR_INPUTS[key]
        _set_mdl_input(shader, attr, kind, value)
    # OmniPBR gates opacity / emission / ORM behind enable flags — flip them on when the
    # user supplies the corresponding value.
    if opacity is not None:
        _set_mdl_input(shader, "enable_opacity", "bool", True)
    if emissive is not None:
        _set_mdl_input(shader, "enable_emission", "bool", True)
    if orm_texture is not None:
        _set_mdl_input(shader, "enable_ORM_texture", "bool", True)
    for attr, value in (inputs or {}).items():
        kind = ("color" if isinstance(value, (list, tuple)) else
                "asset" if isinstance(value, str) else
                "bool" if isinstance(value, bool) else "float")
        _set_mdl_input(shader, attr, kind, value)

    # MDL connects through the "mdl" render context; surface/displacement/volume all -> "out".
    for make in ("CreateSurfaceOutput", "CreateDisplacementOutput", "CreateVolumeOutput"):
        getattr(material, make)("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return mat_path


def update_material(stage, material_path: str, *, color=None, metallic: float | None = None,
                    roughness: float | None = None, opacity: float | None = None,
                    clearcoat: float | None = None,
                    clearcoat_roughness: float | None = None, ior: float | None = None,
                    emissive=None, inputs: dict | None = None) -> dict:
    """Edit an EXISTING material's surface shader in place — no create, no bind.

    This is what `material <material-ref> --color …` means: the ref IS the material,
    so its shader params are updated rather than a new material being authored and
    bound onto the Material prim. Works for both UsdPreviewSurface and MDL shaders
    (the named kwargs map onto OmniPBR's constant inputs, as in create_mdl_material);
    only the params actually supplied are touched. `inputs` passes raw shader
    attributes through. Returns {"material_path", "kind", "updated": [input names]}.
    """
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not material:
        raise ValueError(f"{material_path} is not a UsdShade.Material")
    # Prefer the MDL surface output; fall back to the universal (UsdPreviewSurface) one.
    source = material.ComputeSurfaceSource("mdl") or material.ComputeSurfaceSource()
    shader = source[0] if source else None
    if not shader:
        raise ValueError(f"{material_path} has no surface shader to update")
    node_def = UsdShade.NodeDefAPI(shader.GetPrim())
    is_mdl = bool(node_def.GetSourceAsset("mdl"))
    updated: list[str] = []

    if is_mdl:
        for key, value in (("clearcoat", clearcoat),
                           ("clearcoat_roughness", clearcoat_roughness), ("ior", ior)):
            if value is not None:
                raise ValueError(f"--{key.replace('_', '-')} has no OmniPBR mapping — "
                                 "set the module's raw input via --input instead")
        named = {"color": color, "metallic": metallic, "roughness": roughness,
                 "opacity": opacity, "emissive": emissive}
        for key, value in named.items():
            if value is None:
                continue
            attr, kind = _OMNIPBR_INPUTS[key]
            _set_mdl_input(shader, attr, kind, value)
            updated.append(attr)
        if opacity is not None:
            _set_mdl_input(shader, "enable_opacity", "bool", True)
        if emissive is not None:
            _set_mdl_input(shader, "enable_emission", "bool", True)
        for attr, value in (inputs or {}).items():
            kind = ("color" if isinstance(value, (list, tuple)) else
                    "asset" if isinstance(value, str) else
                    "bool" if isinstance(value, bool) else "float")
            _set_mdl_input(shader, attr, kind, value)
            updated.append(attr)
        return {"material_path": material_path, "kind": "mdl", "updated": updated}

    def _set_preview(attr: str, type_name, value) -> None:
        inp = shader.CreateInput(attr, type_name)
        # A connected source (e.g. a UsdUVTexture) beats any constant — updating the
        # constant under it would be a silent no-op, so the constant edit wins.
        if inp.HasConnectedSource():
            inp.DisconnectSource()
        inp.Set(value)
        updated.append(attr)

    if color is not None:
        _set_preview("diffuseColor", Sdf.ValueTypeNames.Color3f,
                     Gf.Vec3f(*[float(c) for c in color]))
    for attr, value in (("metallic", metallic), ("roughness", roughness),
                        ("opacity", opacity), ("clearcoat", clearcoat),
                        ("clearcoatRoughness", clearcoat_roughness), ("ior", ior)):
        if value is not None:
            _set_preview(attr, Sdf.ValueTypeNames.Float, float(value))
    if emissive is not None:
        _set_preview("emissiveColor", Sdf.ValueTypeNames.Color3f,
                     Gf.Vec3f(*[float(c) for c in emissive]))
    for attr, value in (inputs or {}).items():
        if isinstance(value, (list, tuple)):
            _set_preview(attr, Sdf.ValueTypeNames.Color3f,
                         Gf.Vec3f(*[float(c) for c in value]))
        elif isinstance(value, bool):
            _set_preview(attr, Sdf.ValueTypeNames.Bool, value)
        else:
            _set_preview(attr, Sdf.ValueTypeNames.Float, float(value))
    return {"material_path": material_path, "kind": "UsdPreviewSurface",
            "updated": updated}


def _editable_binding_prim(stage, prim_path: str):
    """The prim we can actually author a binding on for `prim_path`, plus whether it was
    redirected. USD forbids authoring on instance proxies (and prototypes), so for a proxy
    we walk up to the nearest editable ancestor — the instanceable root — whose binding the
    proxy inherits. Returns (prim, redirected_from | None)."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsInstanceProxy():
        return prim, None
    anc = prim
    while anc and anc.IsInstanceProxy():
        anc = anc.GetParent()
    if not anc or not anc.IsValid():
        raise ValueError(f"{prim_path} is an instance proxy with no editable ancestor")
    return anc, prim_path


def prototype_source_path(stage, prim_path: str) -> tuple[str, str]:
    """Map a prim at/inside a native instance to the editable *source* prim its prototype
    composes from — where a binding is authored once and inherited by every instance.

    Works for internal references (`instanceable` prims referencing a subtree of this
    stage, the flattened-asset pattern). Returns (source_path, instance_root_path).
    Raises when the prim isn't instanced or the prototype comes from an external layer
    (not editable from this stage — bind on the instance root instead).
    """
    from pxr import Usd

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"no prim at {prim_path}")
    root = prim
    while root.IsInstanceProxy():
        root = root.GetParent()
    if not root.IsInstance():
        raise ValueError(f"{prim_path} is not inside a native instance — --prototype "
                         "only applies to instanced prims")
    rel = prim.GetPath().MakeRelativePath(root.GetPath())

    root_layer = stage.GetRootLayer()
    query = Usd.PrimCompositionQuery.GetDirectReferences(root)
    for arc in query.GetCompositionArcs():
        target = arc.GetTargetNode()
        if target.layerStack.identifier.rootLayer != root_layer:
            continue  # external asset — its prims aren't authorable from this stage
        src = target.path if prim == root else target.path.AppendPath(rel)
        src_prim = stage.GetPrimAtPath(src)
        if src_prim and src_prim.IsValid() and not src_prim.IsInstanceProxy():
            return src.pathString, root.GetPath().pathString
    raise ValueError(
        f"{prim_path}: its prototype comes from an external layer, so it has no editable "
        "source prim in this stage — bind on the instance root instead (omit --prototype)")


def bind_material(stage, material_path: str, prim_path: str, *,
                  on_prototype: bool = False) -> str:
    """Bind an existing material prim to a geometry prim, a GeomSubset, or (with
    `on_prototype`) the instanced prim's prototype source. Returns the path the binding
    was actually authored on.

    * GeomSubset target: bound directly on the subset; the `materialBind` family
      metadata (familyName / elementType / familyType) is authored automatically.
    * Instance proxy: by default the binding is authored on the editable instanceable
      root as `strongerThanDescendants` (colors the whole instance). With
      `on_prototype=True` it is instead authored once on the prototype's source prim,
      so *every* instance sharing that prototype picks it up at the right granularity.
    """
    from pxr import UsdShade

    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not material:
        raise ValueError(f"{material_path} is not a UsdShade.Material")
    if on_prototype:
        prim_path, _ = prototype_source_path(stage, prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"{prim_path} is not a valid prim")
    if UsdShade.Material(prim):
        # Binding a material ONTO a Material prim is never what anyone means — the
        # classic mistake is `material <material-ref> --color …`, which should edit
        # that material in place (update_material), not author a new one over it.
        raise ValueError(
            f"{prim_path} is itself a UsdShade.Material — refusing to bind a material "
            "onto a material. Edit its shader inputs in place instead, or bind onto "
            "the geometry that should use it.")
    if str(prim.GetTypeName()) == "GeomSubset":
        if prim.IsInstanceProxy():
            raise ValueError(
                f"{prim_path} is a GeomSubset inside a native instance — USD cannot "
                "author there. Re-run with --prototype to bind on the prototype source.")
        from usd_core.subsets import ensure_material_bind_family
        ensure_material_bind_family(stage, prim_path)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        return prim_path
    prim, redirected = _editable_binding_prim(stage, prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"{prim_path} is not a valid prim")
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    if redirected:
        api.Bind(material, UsdShade.Tokens.strongerThanDescendants)
    else:
        api.Bind(material)
    return prim.GetPath().pathString


def effective_materials_under(stage, root_path: str) -> dict:
    """Aggregate the *effective* (computed) material of every gprim under a prim,
    descending into native-instance proxies — how `snapshot -m` / `describe` expose
    materials inside collapsed instances without prototype-path knowledge.

    Returns {"gprims": n, "unbound": n, "materials": {material_path: gprim_count}}.
    """
    from pxr import Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise ValueError(f"no prim at {root_path}")
    counts: dict[str, int] = {}
    unbound = total = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or not prim.IsA(UsdGeom.Gprim):
            continue
        total += 1
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat and mat.GetPrim().IsValid():
            p = mat.GetPath().pathString
            counts[p] = counts.get(p, 0) + 1
        else:
            unbound += 1
    return {"gprims": total, "unbound": unbound, "materials": counts}


_MATERIAL_TERMINAL_ATTRIBUTES = (
    "outputs:surface",
    "outputs:displacement",
    "outputs:volume",
)


def _is_valid_sdf_path_string(value: str) -> bool:
    from pxr import Sdf

    result = Sdf.Path.IsValidPathString(value)
    return bool(result[0] if isinstance(result, tuple) else result)


def is_library_material_prim(prim) -> bool:
    """Return whether a library prim can be imported as a USD material.

    Proper ``UsdShade.Material`` prims remain the canonical form. Some DCC
    exporters instead author a generic Scope/Xform carrying the standard
    material terminal attributes. Those prims contain enough unambiguous shade
    data to normalize at the local reference site without modifying the source
    library. A generic prim with only a material-like name is not accepted.
    """
    from pxr import UsdShade

    if not prim or not prim.IsValid():
        return False
    if UsdShade.Material(prim):
        return True
    for attribute_name in _MATERIAL_TERMINAL_ATTRIBUTES:
        terminal = prim.GetAttribute(attribute_name)
        if terminal.IsValid() and terminal.HasAuthoredConnections():
            return True
    return False


def _find_library_material(lib, material_name: str) -> str | None:
    """Path of the library material matching `material_name` (exact, then normalized)."""
    from pxr import Tf

    # Guard GetPrimAtPath: a display name with spaces ("Stainless Steel") is not a valid
    # SdfPath and each probe spews an "Ill-formed SdfPath" warning (dozens per bind). Only
    # probe candidates that are actually valid path strings; the normalized match below
    # handles spaced names.
    candidates = [f"/Looks/{material_name}", f"/{material_name}",
                  f"/World/Looks/{material_name}"]
    src = next((c for c in candidates
                if _is_valid_sdf_path_string(c) and lib.GetPrimAtPath(c).IsValid()), None)
    if src is not None:
        return src
    # Normalized match: display names like "Aluminum Brushed" → prim Aluminum_Brushed.
    want = Tf.MakeValidIdentifier(material_name).lower()
    for p in lib.Traverse():
        if is_library_material_prim(p) and p.GetName().lower() == want:
            return p.GetPath().pathString
    return None


def reference_library_material(
    stage,
    *,
    library_path: str,
    material_name: str,
    source_prim_path: str | None = None,
    prim_path: str | None = None,
) -> str:
    """Bring a material from an external library USD into Looks (via reference); bind it
    when `prim_path` is given, else just import it (returns the local material path).

    Importing without binding is the scene-repair path: pull the named library materials
    into `/Looks`, then re-bind existing prims to them by name.
    """
    local = import_library_material(
        stage,
        library_path=library_path,
        material_name=material_name,
        source_prim_path=source_prim_path,
    )
    if prim_path is not None:
        bind_material(stage, local, prim_path)
    return local


def import_library_material(
    stage,
    *,
    library_path: str,
    material_name: str,
    source_prim_path: str | None = None,
) -> str:
    """Reference a material from an external library USD into Looks (no bind). Returns the
    local material path. See reference_library_material for the bind-too variant."""
    import os

    from pxr import Sdf, Tf, Usd, UsdShade

    # A caller-relative path like ../shared/lib.usd resolves here against the process
    # CWD, but an authored reference re-anchors to the layer receiving the opinion.
    # Keep file-backed edit layers portable; anonymous session layers have no stable
    # anchor, so they must carry an absolute path. Clean-slate session edits are saved
    # flattened, which composes this reference away from the derivative output.
    library_path = os.path.abspath(library_path)
    canonical_library_identity = os.path.realpath(library_path)
    edit_layer = stage.GetEditTarget().GetLayer()
    edit_real = getattr(edit_layer, "realPath", "") or ""
    authored_path = (
        os.path.relpath(library_path, os.path.dirname(edit_real))
        if edit_real
        else library_path
    )
    lib = Usd.Stage.Open(library_path)
    if not lib:
        raise FileNotFoundError(f"cannot open material library: {library_path}")
    if source_prim_path is not None:
        if not _is_valid_sdf_path_string(source_prim_path):
            raise ValueError(
                "library material source path must be an exact absolute prim path"
            )
        source_path = Sdf.Path(source_prim_path)
        if (
            not source_path.IsAbsolutePath()
            or not source_path.IsPrimPath()
            or source_path == Sdf.Path.absoluteRootPath
        ):
            raise ValueError(
                "library material source path must be an exact absolute prim path"
            )
        src = source_path.pathString
    else:
        src = _find_library_material(lib, material_name)
    if (
        src is None
        and source_prim_path is None
        and material_name == "Material"
    ):  # no --name given: first material
        src = next((p.GetPath().pathString for p in lib.Traverse()
                    if is_library_material_prim(p)), None)
    if src is None:
        available = sorted(
            p.GetName() for p in lib.Traverse() if is_library_material_prim(p)
        )
        listing = ", ".join(available[:20]) + (" …" if len(available) > 20 else "")
        raise ValueError(f"no material '{material_name}' in {library_path} "
                         f"({len(available)} available: {listing})")
    source_prim = lib.GetPrimAtPath(src)
    if not is_library_material_prim(source_prim):
        raise ValueError(
            f"referenced prim {src} in {library_path} is not a UsdShade.Material "
            "or a legacy prim with an authored material terminal"
        )
    normalize_legacy_type = not bool(UsdShade.Material(source_prim))
    looks = _looks_scope(stage)
    # Local prim name must be a valid identifier even when the lookup name was a
    # display name ("Aluminum Brushed") — otherwise DefinePrim leaks a raw pxr error.
    exact_source = source_prim_path is not None
    if exact_source:
        # Exact imports are identified by their complete source prim path, not
        # only its leaf.  Reusing /Looks/Metal for both /A/Looks/Metal and
        # /B/Looks/Metal would compose two unrelated references into one local
        # Material; an original scene-owned /Looks/Metal is equally unsafe to
        # reuse.  Keep the leaf readable and add a stable path-derived suffix.
        source_digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
        local_name = f"{Sdf.Path(src).name}_{source_digest}"
        local = f"{looks.rstrip('/')}/{local_name}"
        existing = stage.GetPrimAtPath(local)
        if existing.IsValid():
            stored_asset = existing.GetCustomDataByKey("usdCliLibraryAsset")
            stored_identity = ""
            if isinstance(stored_asset, str) and stored_asset:
                if os.path.isabs(stored_asset):
                    stored_identity = os.path.realpath(stored_asset)
                else:
                    owning_layer_real_path = ""
                    for prim_spec in existing.GetPrimStack():
                        custom_data = (
                            prim_spec.GetInfo("customData")
                            if prim_spec.HasInfo("customData")
                            else {}
                        )
                        if (
                            isinstance(custom_data, dict)
                            and custom_data.get("usdCliLibraryAsset") == stored_asset
                        ):
                            owning_layer_real_path = str(
                                getattr(prim_spec.layer, "realPath", "") or ""
                            )
                            break
                    anchor = owning_layer_real_path or edit_real
                    if anchor:
                        stored_identity = os.path.realpath(
                            os.path.join(os.path.dirname(anchor), stored_asset)
                        )
            same_import = (
                bool(UsdShade.Material(existing))
                and stored_identity == canonical_library_identity
                and existing.GetCustomDataByKey("usdCliLibrarySourcePrim") == src
            )
            if not same_import:
                raise ValueError(
                    f"{local} is already occupied by a different material; "
                    f"cannot import exact library prim {src}"
                )
            if not existing.IsActive():
                existing.SetActive(True)
            return local
    else:
        local_name = Tf.MakeValidIdentifier(material_name)
        local = _material_define_path(stage, looks, local_name)
    # Canonical libraries supply the type through the reference. A legacy DCC
    # library may carry valid material terminals on a generic Scope/Xform; type
    # only that local reference site as Material so binding remains valid while
    # the source layer stays immutable.
    prim = (
        UsdShade.Material.Define(stage, local).GetPrim()
        if normalize_legacy_type
        else stage.DefinePrim(local)
    )
    prim.GetReferences().AddReference(authored_path, Sdf.Path(src))
    if exact_source:
        prim.SetCustomDataByKey(
            "usdCliLibraryAsset",
            # This field is provenance, not a USD asset-valued composition arc.
            # Keep the reference itself relative for portability, but record the
            # canonical source identity absolutely.  A later ``save --flatten``
            # can move the composed prim to another directory; copying a relative
            # provenance string verbatim would silently re-anchor it there.
            canonical_library_identity,
        )
        prim.SetCustomDataByKey("usdCliLibrarySourcePrim", src)
    if not UsdShade.Material(prim):
        raise ValueError(f"referenced prim {src} in {library_path} is not a UsdShade.Material")
    return local


def unbind_material(stage, prim_path: str) -> None:
    """Remove a direct material binding from a prim (reversible-ish: clears the rel).

    For an instance proxy, clears the binding on the editable instanceable root where
    `bind_material` would have authored it (you can't author on the proxy itself)."""
    from pxr import UsdShade

    prim, _ = _editable_binding_prim(stage, prim_path)
    if not prim or not prim.IsValid():
        return
    api = UsdShade.MaterialBindingAPI(prim)
    rel = api.GetDirectBindingRel()
    if rel:
        rel.ClearTargets(True)


def bound_material(stage, prim_path: str) -> dict:
    """Report a prim's material binding (for queries/verification)."""
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        # ComputeBoundMaterial on an invalid prim dies with the cryptic pxr
        # "Accessed schema on invalid prim" — say what actually went wrong.
        raise ValueError(f"no prim at {prim_path}")
    api = UsdShade.MaterialBindingAPI(prim)
    mat, rel = api.ComputeBoundMaterial()
    direct = api.GetDirectBindingRel()
    targets = [t.pathString for t in direct.GetTargets()] if direct else []
    bound_path = mat.GetPath().pathString if mat and mat.GetPrim().IsValid() else None
    binding_type = "none"
    if bound_path:
        binding_type = "direct" if targets else "inherited"
    return {"binding_type": binding_type, "bound_material_path": bound_path,
            "direct_targets": targets}


def describe_material(stage, material_path: str) -> dict:
    """Report a material's surface shader: kind (UsdPreviewSurface | mdl), MDL module +
    subIdentifier when MDL, and its authored inputs. For queries / verification."""
    from pxr import UsdShade

    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not material:
        raise ValueError(f"{material_path} is not a UsdShade.Material")
    # Prefer the MDL surface output; fall back to the universal (UsdPreviewSurface) one.
    source = material.ComputeSurfaceSource("mdl") or material.ComputeSurfaceSource()
    shader = source[0] if source else None
    info: dict = {"material_path": material_path, "kind": None, "mdl_module": None,
                  "mdl_subidentifier": None, "shader_id": None, "inputs": {}}
    if not shader:
        return info
    node_def = UsdShade.NodeDefAPI(shader.GetPrim())
    asset, sub = node_def.GetSourceAsset("mdl"), node_def.GetSourceAssetSubIdentifier("mdl")
    if asset:
        info.update(kind="mdl", mdl_module=asset.path, mdl_subidentifier=sub or None)
    else:
        shader_id = shader.GetShaderId()
        info.update(kind=shader_id or "unknown", shader_id=shader_id or None)
    for inp in shader.GetInputs():
        val = inp.Get()
        info["inputs"][inp.GetBaseName()] = (
            val.path if hasattr(val, "path") else
            list(val) if hasattr(val, "__len__") and not isinstance(val, str) else val)
        # a connected input (e.g. a UsdUVTexture) matters more than its constant fallback
        if inp.HasConnectedSource():
            src = inp.GetConnectedSource()[0]
            src_prim = src.GetPrim()
            file_attr = src_prim.GetAttribute("inputs:file")
            file_val = file_attr.Get() if file_attr else None
            info["inputs"][inp.GetBaseName()] = (
                f"<- {src_prim.GetName()}" + (f" ({file_val.path})" if file_val else ""))
    return info
