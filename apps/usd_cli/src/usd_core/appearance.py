# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clean-slate appearance masking in the strongest USD session layer.

``appearance clear`` is intentionally a composition overlay, not a destructive
rewrite.  It blocks every composed material-binding relationship and the display
primvars that Hydra can use as fallback appearance.  The source/root layer is never
edited; callers explicitly flatten the composed stage to a *different* file when they
want to persist a clean-slate derivative.

Native instances are de-instanced only in the session layer.  That makes prototype
properties addressable without touching their source layers and lets one atomic layer
snapshot restore both appearance and instancing on undo.
"""

from __future__ import annotations

from dataclasses import dataclass

APPEARANCE_CLEAR_MARKER = "usdCliAppearanceClear"
DISPLAY_APPEARANCE_ATTRIBUTES = (
    "primvars:displayColor",
    "primvars:displayColor:indices",
    "primvars:displayOpacity",
)
DIRECT_SHADER_OUTPUT_TERMS = frozenset({"surface", "mdl", "displacement", "volume"})
_AUDIT_LIST_LIMIT = 50


@dataclass(frozen=True)
class AppearanceClearResult:
    """The reversible state transition produced by :func:`clear_appearance`."""

    before_layer: str
    after_layer: str
    previous_edit_target: str
    blocked_bindings: int
    blocked_display_attributes: int
    blocked_shader_outputs: int
    deinstanced_roots: tuple[str, ...]
    before_audit: dict
    after_audit: dict


def _is_material_binding(name: str) -> bool:
    return name == "material:binding" or name.startswith("material:binding:")


def _is_direct_shader_output(name: str) -> bool:
    components = name.split(":")
    return components[0] == "outputs" and bool(
        DIRECT_SHADER_OUTPUT_TERMS.intersection(components[1:])
    )


def _active_prims_with_instance_proxies(stage):
    from pxr import Usd

    return list(stage.Traverse(Usd.TraverseInstanceProxies()))


def _deinstance_in_session(stage) -> list[str]:
    """Make every native instance editable, outermost first, in the session layer.

    Nested instance roots are proxies until their enclosing instance is opened. Each
    pass therefore disables only real, editable roots. A repeated root or a pass with
    no editable root is an unsupported composition and fails before appearance masking
    continues; the caller restores the complete layer snapshot.
    """

    changed: list[str] = []
    seen: set[str] = set()
    while True:
        prims = _active_prims_with_instance_proxies(stage)
        proxies_before = sum(1 for prim in prims if prim.IsInstanceProxy())
        roots = [
            prim for prim in prims if prim.IsInstance() and not prim.IsInstanceProxy()
        ]
        if not roots:
            if proxies_before:
                raise RuntimeError(
                    f"cannot make {proxies_before} instance proxy prim(s) editable "
                    "without modifying a prototype source"
                )
            return changed

        progress = 0
        for prim in roots:
            path = prim.GetPath().pathString
            if path in seen:
                raise RuntimeError(f"de-instancing made no progress at {path}")
            if prim.SetInstanceable(False) is False:
                raise RuntimeError(f"failed to de-instance {path} in the session layer")
            if stage.GetPrimAtPath(path).IsInstance():
                raise RuntimeError(
                    f"{path} remained an instance after the session-layer override"
                )
            seen.add(path)
            changed.append(path)
            progress += 1

        if not progress:
            raise RuntimeError(
                "de-instancing made no progress; clean-slate appearance cannot be "
                "guaranteed for this composition"
            )


def _binding_relationships(stage) -> list[tuple[str, str]]:
    relationships: list[tuple[str, str]] = []
    for prim in stage.TraverseAll():
        for rel in prim.GetRelationships():
            if _is_material_binding(rel.GetName()):
                relationships.append((prim.GetPath().pathString, rel.GetName()))
    return relationships


def _display_attributes(stage) -> list[tuple[str, str]]:
    attributes: list[tuple[str, str]] = []
    for prim in stage.TraverseAll():
        for name in DISPLAY_APPEARANCE_ATTRIBUTES:
            attr = prim.GetAttribute(name)
            if attr and (
                attr.HasAuthoredValueOpinion()
                or attr.GetNumTimeSamples() > 0
                or attr.Get() is not None
            ):
                attributes.append((prim.GetPath().pathString, name))
    return attributes


def _direct_shader_outputs(stage) -> list[tuple[str, str]]:
    """Return effective shader-style outputs authored on renderable prims."""

    from pxr import UsdGeom

    outputs: list[tuple[str, str]] = []
    for prim in stage.TraverseAll():
        if not (prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset)):
            continue
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if not _is_direct_shader_output(name):
                continue
            if attr.GetConnections() or attr.Get() is not None or attr.GetTimeSamples():
                outputs.append((prim.GetPath().pathString, name))
    return outputs


def _block_binding_relationship(stage, prim_path: str, name: str) -> None:
    """Author an explicit-empty target list in the current (session) edit target."""

    prim = stage.OverridePrim(prim_path)
    rel = prim.GetRelationship(name)
    if not rel:
        rel = prim.CreateRelationship(name, custom=False)
    if rel.SetTargets([]) is False or rel.GetTargets():
        raise RuntimeError(f"failed to block {prim_path}.{name}")


def _block_display_attribute(stage, prim_path: str, name: str) -> None:
    """Author an Sdf value block for one composed display primvar."""

    from pxr import Sdf

    prim = stage.OverridePrim(prim_path)
    attr = prim.GetAttribute(name)
    if not attr:
        raise RuntimeError(
            f"cannot block {prim_path}.{name}: composed attribute disappeared"
        )
    attr.Block()
    spec_path = Sdf.Path(prim_path).AppendProperty(name)
    spec = stage.GetSessionLayer().GetPropertyAtPath(spec_path)
    if spec is None or not isinstance(spec.default, Sdf.ValueBlock):
        raise RuntimeError(
            f"failed to author a session-layer value block at {spec_path}"
        )


def _block_direct_shader_output(stage, prim_path: str, name: str) -> None:
    """Block one renderable prim's direct shader output and its connections."""

    from pxr import Sdf

    prim = stage.OverridePrim(prim_path)
    attr = prim.GetAttribute(name)
    if not attr:
        raise RuntimeError(
            f"cannot block {prim_path}.{name}: composed attribute disappeared"
        )
    if attr.SetConnections([]) is False or attr.GetConnections():
        raise RuntimeError(f"failed to clear connections on {prim_path}.{name}")
    attr.Block()
    spec_path = Sdf.Path(prim_path).AppendProperty(name)
    spec = stage.GetSessionLayer().GetPropertyAtPath(spec_path)
    if spec is None or not isinstance(spec.default, Sdf.ValueBlock):
        raise RuntimeError(
            f"failed to author a session-layer shader-output block at {spec_path}"
        )
    if attr.Get() is not None or attr.GetTimeSamples():
        raise RuntimeError(f"failed to block shader output values at {spec_path}")


def _material_purposes():
    from pxr import UsdShade

    return (
        UsdShade.Tokens.allPurpose,
        UsdShade.Tokens.preview,
        UsdShade.Tokens.full,
    )


def _surface_shader_kind(material) -> str | None:
    """Return the effective surface shader kind for audit output, if one exists."""

    for context in ("", "mdl", "mtlx"):
        try:
            source = (
                material.ComputeSurfaceSource(context)
                if context
                else material.ComputeSurfaceSource()
            )
        except Exception:  # noqa: BLE001 - renderer contexts are extensible
            continue
        shader = source[0] if source else None
        if not shader:
            continue
        shader_id = shader.GetShaderId()
        return str(shader_id or context or "connected")
    return None


def _cap(items: list) -> tuple[list, int]:
    if len(items) <= _AUDIT_LIST_LIMIT:
        return items, 0
    return items[:_AUDIT_LIST_LIMIT], len(items) - _AUDIT_LIST_LIMIT


def audit_appearance(stage) -> dict:
    """Audit effective material/shader/display appearance for the composed stage.

    ``clear`` is true only when no material-binding relationship still has targets,
    no renderable or material subset resolves a material for any standard purpose,
    no such material supplies an effective shader, no direct renderable-prim
    shader output resolves, no display appearance value resolves, and no
    uneditable instance proxy remains.
    """

    from pxr import UsdGeom, UsdShade

    live_relationships: list[dict] = []
    display_values: list[dict] = []
    effective_bindings: list[dict] = []
    effective_shaders: list[dict] = []
    direct_shader_outputs: list[dict] = []
    instance_proxies: list[str] = []

    for prim in _active_prims_with_instance_proxies(stage):
        path = prim.GetPath().pathString
        if prim.IsInstanceProxy():
            instance_proxies.append(path)
        for rel in prim.GetRelationships():
            if not _is_material_binding(rel.GetName()):
                continue
            targets = [target.pathString for target in rel.GetTargets()]
            if targets:
                live_relationships.append(
                    {"path": path, "relationship": rel.GetName(), "targets": targets}
                )
        for name in DISPLAY_APPEARANCE_ATTRIBUTES:
            attr = prim.GetAttribute(name)
            if not attr:
                continue
            value = attr.Get()
            samples = list(attr.GetTimeSamples())
            if value is not None or samples:
                display_values.append(
                    {
                        "path": path,
                        "attribute": name,
                        "has_default": value is not None,
                        "time_samples": len(samples),
                    }
                )

        if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset):
            for attr in prim.GetAttributes():
                name = attr.GetName()
                if not _is_direct_shader_output(name):
                    continue
                connections = [item.pathString for item in attr.GetConnections()]
                value = attr.Get()
                samples = list(attr.GetTimeSamples())
                if connections or value is not None or samples:
                    direct_shader_outputs.append(
                        {
                            "path": path,
                            "attribute": name,
                            "connections": connections,
                            "has_default": value is not None,
                            "time_samples": len(samples),
                        }
                    )

        if not (prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset)):
            continue
        api = UsdShade.MaterialBindingAPI(prim)
        seen: set[tuple[str, str]] = set()
        for purpose in _material_purposes():
            material, rel = api.ComputeBoundMaterial(materialPurpose=purpose)
            if not material or not material.GetPrim().IsValid():
                continue
            material_path = material.GetPath().pathString
            relationship = rel.GetPath().pathString if rel else ""
            # preview/full queries fall back to the same all-purpose relationship.
            # Count that composed winner once, but retain genuinely distinct
            # purpose-specific relationships.
            key = (material_path, relationship)
            if key in seen:
                continue
            seen.add(key)
            row = {
                "path": path,
                "purpose": str(purpose) or "allPurpose",
                "material": material_path,
                "relationship": relationship,
            }
            effective_bindings.append(row)
            shader = _surface_shader_kind(material)
            if shader:
                effective_shaders.append({**row, "shader": shader})

    counts = {
        "binding_relationships_with_targets": len(live_relationships),
        "effective_material_bindings": len(effective_bindings),
        "effective_shader_appearances": len(effective_shaders),
        "direct_shader_outputs": len(direct_shader_outputs),
        "display_values": len(display_values),
        "instance_proxies": len(instance_proxies),
    }
    report = {
        "clear": not any(counts.values()),
        "overlay_active": appearance_clear_active(stage),
        "counts": counts,
    }
    for key, items in (
        ("binding_relationships", live_relationships),
        ("effective_bindings", effective_bindings),
        ("effective_shaders", effective_shaders),
        ("direct_shader_outputs", direct_shader_outputs),
        ("display_values", display_values),
        ("instance_proxies", instance_proxies),
    ):
        capped, omitted = _cap(items)
        report[key] = capped
        if omitted:
            report[f"{key}_omitted"] = omitted
            report["truncated"] = True
    return report


def appearance_clear_active(stage) -> bool:
    """Whether the session layer carries the clean-slate overlay marker."""

    marker = stage.GetSessionLayer().customLayerData.get(APPEARANCE_CLEAR_MARKER)
    return isinstance(marker, dict) and marker.get("version") == 1


def _edit_target_name(stage) -> str:
    target = stage.GetEditTarget().GetLayer()
    if target == stage.GetRootLayer():
        return "root"
    if target == stage.GetSessionLayer():
        return "session"
    raise RuntimeError(
        "appearance clear supports only the root or session edit target; "
        f"current target is {target.identifier}"
    )


def restore_session_layer(stage, layer_text: str, edit_target: str) -> None:
    """Restore an exact session-layer snapshot and the associated edit target."""

    layer = stage.GetSessionLayer()
    if layer.ImportFromString(layer_text) is False:
        raise RuntimeError("failed to restore the appearance session-layer snapshot")
    if edit_target == "root":
        stage.SetEditTarget(stage.GetRootLayer())
    elif edit_target == "session":
        stage.SetEditTarget(layer)
    else:
        raise RuntimeError(f"unknown saved edit target '{edit_target}'")


def clear_appearance(stage) -> AppearanceClearResult:
    """Mask all composed appearance in one reversible session-layer transaction."""

    session_layer = stage.GetSessionLayer()
    if not session_layer.permissionToEdit:
        raise RuntimeError("the USD session layer is not editable")
    previous_target = _edit_target_name(stage)
    before_layer = session_layer.ExportToString()
    before_audit = audit_appearance(stage)

    try:
        stage.SetEditTarget(session_layer)
        deinstanced = _deinstance_in_session(stage)
        relationships = _binding_relationships(stage)
        attributes = _display_attributes(stage)
        shader_outputs = _direct_shader_outputs(stage)

        for prim_path, name in relationships:
            _block_binding_relationship(stage, prim_path, name)
        for prim_path, name in attributes:
            _block_display_attribute(stage, prim_path, name)
        for prim_path, name in shader_outputs:
            _block_direct_shader_output(stage, prim_path, name)

        custom_data = dict(session_layer.customLayerData)
        custom_data[APPEARANCE_CLEAR_MARKER] = {
            "version": 1,
            "mode": "bindings-shader-outputs-and-display-primvars",
        }
        session_layer.customLayerData = custom_data

        after_audit = audit_appearance(stage)
        if not after_audit["clear"]:
            counts = after_audit["counts"]
            raise RuntimeError(
                "effective appearance remains after masking: "
                + ", ".join(f"{key}={value}" for key, value in counts.items() if value)
            )
        after_layer = session_layer.ExportToString()
    except Exception:
        restore_session_layer(stage, before_layer, previous_target)
        raise

    return AppearanceClearResult(
        before_layer=before_layer,
        after_layer=after_layer,
        previous_edit_target=previous_target,
        blocked_bindings=len(relationships),
        blocked_display_attributes=len(attributes),
        blocked_shader_outputs=len(shader_outputs),
        deinstanced_roots=tuple(deinstanced),
        before_audit=before_audit,
        after_audit=after_audit,
    )
