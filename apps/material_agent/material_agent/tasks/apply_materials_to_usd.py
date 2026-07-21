# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for applying resolved materials to USD prims."""

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.usd.asset_paths import (
    is_absolute_asset_path,
    is_relative_to,
    is_unsafe_resolver_asset_path,
    is_uri_asset_path,
    resolve_relative_asset_path_under_base,
)
from world_understanding.utils.usd.material import (
    add_mdl_material,
    add_ovrtx_preview_fallbacks_for_materialx_openpbr,
    ensure_looks_scope_spec,
)
from world_understanding.utils.usd.prim import nullify_material

from material_agent.material_profiles import (
    MaterialProfile,
    normalize_material_profile,
)
from material_agent.materials import (
    FALLBACK_MATERIAL_NAME,
    PREDICTION_CONTAINER_KEYS,
    PREDICTION_ID_KEYS,
    PREDICTION_MATERIAL_KEYS,
    PREDICTION_VALIDATION_STATUS_KEYS,
    UNKNOWN_MATERIAL_SENTINEL,
    USE_DEFAULT_LIBRARY_SENTINEL,
    is_actionable_material_name,
    is_default_library_fallback_name,
    is_disallowed_unknown_validation_status,
    is_fallback_material_name,
    is_unknown_material_name,
    material_mapping_with_fallback,
    normalize_material_name,
)

logger = logging.getLogger(__name__)


def _warn_asset_remap(listener: Any | None, message: str) -> None:
    if listener is not None:
        listener.warning(message)
    else:
        logger.warning(message)


def remap_single_asset_path(
    path_str: str,
    source_dir: Path,
    target_dir: Path,
    listener: Any | None = None,
) -> str:
    """Remap one material asset path from source_dir to target_dir."""
    if not path_str:
        return path_str

    if is_uri_asset_path(path_str):
        _warn_asset_remap(
            listener,
            "Clearing resolver URI asset path from copied material so native "
            f"USD renderers cannot fetch it directly: {path_str}",
        )
        return ""

    try:
        if is_absolute_asset_path(path_str):
            abs_path = Path(path_str).resolve()
            if not is_relative_to(abs_path, source_dir.resolve()):
                _warn_asset_remap(
                    listener,
                    "Clearing absolute asset path outside the material library "
                    f"directory: {path_str}",
                )
                return ""
        else:
            abs_path = resolve_relative_asset_path_under_base(path_str, source_dir)
    except ValueError as e:
        _warn_asset_remap(listener, f"Clearing unsafe material asset path: {e}")
        return ""

    try:
        new_rel = os.path.relpath(abs_path, target_dir)
    except ValueError:
        _warn_asset_remap(
            listener,
            "Clearing material asset path that cannot be made relative to "
            f"output: {path_str}",
        )
        return ""

    return new_rel.replace("\\", "/")


def remap_asset_paths_in_prim(
    layer: Sdf.Layer,
    prim_path: Sdf.Path,
    source_dir: Path,
    target_dir: Path,
    listener: Any | None = None,
) -> None:
    """Remap all asset paths in a prim and its descendants."""
    prim_spec = layer.GetPrimAtPath(prim_path)
    if not prim_spec:
        return

    for attr_name in list(prim_spec.attributes.keys()):
        attr_spec = prim_spec.attributes[attr_name]
        value = attr_spec.default
        if isinstance(value, Sdf.AssetPath):
            new_path = remap_single_asset_path(
                value.path,
                source_dir,
                target_dir,
                listener,
            )
            if new_path != value.path:
                attr_spec.default = Sdf.AssetPath(new_path)
        elif isinstance(value, Sdf.AssetPathArray):
            new_arr = Sdf.AssetPathArray(
                [
                    Sdf.AssetPath(
                        remap_single_asset_path(
                            ap.path,
                            source_dir,
                            target_dir,
                            listener,
                        )
                    )
                    for ap in value
                ]
            )
            if new_arr != value:
                attr_spec.default = new_arr

    for child_spec in prim_spec.nameChildren:
        remap_asset_paths_in_prim(
            layer,
            prim_path.AppendChild(child_spec.name),
            source_dir,
            target_dir,
            listener,
        )


def clear_color_space_on_empty_asset_inputs(
    layer: Sdf.Layer,
    prim_path: Sdf.Path,
) -> int:
    """Clear colorSpace metadata from empty asset-valued material inputs."""
    prim_spec = layer.GetPrimAtPath(prim_path)
    if not prim_spec:
        return 0

    cleared_count = 0
    for attr_name in list(prim_spec.attributes.keys()):
        attr_spec = prim_spec.attributes[attr_name]
        value = attr_spec.default
        if (
            attr_name.startswith("inputs:")
            and isinstance(value, Sdf.AssetPath)
            and not value.path
            and not attr_spec.connectionPathList.GetAddedOrExplicitItems()
            and attr_spec.HasInfo("colorSpace")
        ):
            attr_spec.ClearInfo("colorSpace")
            cleared_count += 1

    for child_spec in prim_spec.nameChildren:
        cleared_count += clear_color_space_on_empty_asset_inputs(
            layer,
            prim_path.AppendChild(child_spec.name),
        )

    return cleared_count


class ApplyMaterialsToUSDTask(Task):
    """Task to apply resolved materials to USD prims.

    This task takes the resolved material files and applies them to USD prims
    based on the predictions. It supports two modes:
    1. Default: Create a new USD stage with everything (geometry + materials)
    2. Layer mode: Create only a sublayer with material bindings

    Input context keys:
        - input_usd_path: Path to the input USD file
        - output_usd_path: Path to save the output USD file
        - predictions_path: Path to the predictions file
        - resolved_materials: Dictionary mapping material names to local file paths
        - layer_only: Boolean flag to create only a material binding layer (default: False)
        - flatten_output: Boolean flag to flatten output (default: False)
                         When False, preserves references to material libraries
                         When True, creates self-contained flattened USD
        - skip_instance_check: Boolean flag to skip instance material traversal
                              (default: False). Set True for payload pipelines
                              where instances inherit materials via composition.

    Output context keys:
        - output_usd_path: Path where the USD file was saved
        - materials_applied: Dictionary of materials that were applied
        - assignment_stats: Statistics about material assignments
    """

    def __init__(self):
        """Initialize the apply materials to USD task."""
        self.name = "ApplyMaterialsToUSD"
        self.description = "Apply resolved materials to USD prims"

    def _requested_material_profile(self, context: dict[str, Any]) -> MaterialProfile:
        """Return the requested material authoring profile from context aliases."""
        for key in (
            "material_profile",
            "shader_target",
            "material_authoring_target",
            "material_output_target",
        ):
            if key in context and context[key] is not None:
                return normalize_material_profile(context[key])
        return normalize_material_profile(None)

    def _profile_event(
        self,
        *,
        code: str,
        message: str,
        severity: str = "warning",
        material_name: str | None = None,
        requested_profile: str | None = None,
        resolved_profile: str | None = None,
        fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        """Build a structured material-profile warning/error payload."""
        event = {
            "code": code,
            "severity": severity,
            "message": message,
        }
        if material_name is not None:
            event["material_name"] = material_name
        if requested_profile is not None:
            event["requested_profile"] = requested_profile
        if resolved_profile is not None:
            event["resolved_profile"] = resolved_profile
        if fallback_reason is not None:
            event["fallback_reason"] = fallback_reason
        return event

    def _material_profile_result(
        self,
        requested_profile: MaterialProfile,
    ) -> dict[str, Any]:
        """Create the material profile result structure returned to callers."""
        return {
            "requested_profile": requested_profile,
            "resolved_profile": requested_profile
            if requested_profile != "auto"
            else "auto",
            "resolved_profiles": [],
            "fallback_reason": None,
            "warnings": [],
            "errors": [],
            "materials": {},
            "authoring_backends": [],
        }

    @staticmethod
    def _store_material_profile_result(
        context: dict[str, Any],
        material_profile_result: dict[str, Any],
    ) -> None:
        """Store the public material-profile result contract in context."""
        context["material_profile_result"] = material_profile_result
        context["resolved_material_profile"] = material_profile_result.get(
            "resolved_profile"
        )
        context["material_profile_warnings"] = material_profile_result.get(
            "warnings",
            [],
        )
        context["material_profile_errors"] = material_profile_result.get(
            "errors",
            [],
        )

    def _append_profile_event(
        self,
        result: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        bucket = "errors" if event.get("severity") == "error" else "warnings"
        result[bucket].append(event)
        message = event["message"]
        if bucket == "errors":
            self.listener.error(message)
        else:
            self.listener.warning(message)

    def _connected_shader_ids(self, output: UsdShade.Output) -> set[str]:
        """Return shader IDs connected to a USD shade output."""
        shader_ids: set[str] = set()
        try:
            sources, _ = output.GetConnectedSources()
        except Exception:
            return shader_ids

        for source_info in sources:
            source = source_info.source
            if not source:
                continue
            shader = UsdShade.Shader(source.GetPrim())
            if not shader:
                continue
            shader_id_attr = shader.GetIdAttr()
            shader_id = shader_id_attr.Get() if shader_id_attr else None
            if shader_id:
                shader_ids.add(str(shader_id))
        return shader_ids

    def _material_has_mdl_shader(self, material_prim: Usd.Prim) -> bool:
        """Return whether a material contains an MDL source shader."""
        for prim in Usd.PrimRange(material_prim, Usd.PrimAllPrimsPredicate):
            if not prim.IsActive() or not prim.IsA(UsdShade.Shader):
                continue
            mdl_attr = prim.GetAttribute("info:mdl:sourceAsset")
            if mdl_attr and mdl_attr.IsValid() and mdl_attr.Get() is not None:
                return True
        return False

    def _detect_material_profile_on_stage(
        self,
        stage: Usd.Stage,
        material_path: str,
    ) -> tuple[MaterialProfile | None, dict[str, bool]]:
        """Detect the authored profile of a material prim on a stage."""
        prim = stage.GetPrimAtPath(material_path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdShade.Material):
            return None, {
                "has_openpbr_materialx": False,
                "has_preview_surface": False,
                "has_mdl": False,
            }

        material = UsdShade.Material(prim)
        has_openpbr_materialx = False
        mtlx_output = material.GetSurfaceOutput("mtlx")
        if mtlx_output:
            has_openpbr_materialx = (
                "ND_open_pbr_surface_surfaceshader"
                in self._connected_shader_ids(mtlx_output)
            )

        has_preview_surface = False
        surface_output = material.GetSurfaceOutput()
        if surface_output:
            has_preview_surface = "UsdPreviewSurface" in self._connected_shader_ids(
                surface_output
            )

        has_mdl = bool(
            material.GetSurfaceOutput("mdl")
        ) or self._material_has_mdl_shader(prim)

        details = {
            "has_openpbr_materialx": has_openpbr_materialx,
            "has_preview_surface": has_preview_surface,
            "has_mdl": has_mdl,
        }
        if has_openpbr_materialx:
            return "openpbr_materialx", details
        if has_mdl:
            return "omnipbr_mdl", details
        if has_preview_surface:
            return "preview_surface", details
        return None, details

    def _source_profile_for_file_material(
        self,
        material_name: str,
        material_path: str,
    ) -> tuple[MaterialProfile, str | None, dict[str, bool]]:
        """Return the source/effective profile for a direct file material."""
        if is_fallback_material_name(material_name):
            return (
                "preview_surface",
                "canonical_fallback_material",
                {
                    "has_openpbr_materialx": False,
                    "has_preview_surface": True,
                    "has_mdl": False,
                },
            )
        if material_path.endswith(".mdl"):
            return (
                "omnipbr_mdl",
                None,
                {
                    "has_openpbr_materialx": False,
                    "has_preview_surface": False,
                    "has_mdl": True,
                },
            )
        return (
            "preview_surface",
            "non_mdl_file_material_uses_preview_surface",
            {
                "has_openpbr_materialx": False,
                "has_preview_surface": True,
                "has_mdl": False,
            },
        )

    def _resolve_material_profile_request(
        self,
        *,
        requested_profile: MaterialProfile,
        resolved_materials: dict[str, str],
        is_library_based: bool,
        material_library_path: str | None,
    ) -> dict[str, Any]:
        """Resolve a requested material profile against available materials."""
        result = self._material_profile_result(requested_profile)

        library_stage = None
        if is_library_based and material_library_path:
            try:
                library_stage = Usd.Stage.Open(str(material_library_path))
            except Exception:
                library_stage = None
            if library_stage is None:
                severity = "error" if requested_profile != "auto" else "warning"
                event = self._profile_event(
                    code="material_profile.material_library_unreadable",
                    severity=severity,
                    message=(
                        "Unable to inspect material profiles because the material "
                        f"library could not be opened: {material_library_path}"
                    ),
                    requested_profile=requested_profile,
                )
                self._append_profile_event(result, event)
                return result

        for material_name, material_path in resolved_materials.items():
            source_profile: MaterialProfile | None
            source_fallback_reason: str | None = None
            source_details: dict[str, bool]

            if is_fallback_material_name(material_name):
                source_profile, source_fallback_reason, source_details = (
                    self._source_profile_for_file_material(material_name, material_path)
                )
            elif library_stage is not None:
                source_profile, source_details = self._detect_material_profile_on_stage(
                    library_stage,
                    material_path,
                )
            else:
                source_profile, source_fallback_reason, source_details = (
                    self._source_profile_for_file_material(material_name, material_path)
                )

            resolved_profile = source_profile
            fallback_reason = source_fallback_reason
            authoring_backend = self._authoring_backend_for_material_profile(
                requested_profile=requested_profile,
                resolved_profile=resolved_profile,
                source_profile=source_profile,
                fallback_reason=fallback_reason,
                is_library_based=is_library_based,
            )

            if requested_profile == "auto":
                if source_profile is None:
                    resolved_profile = "preview_surface"
                    fallback_reason = "unrecognized_material_graph_uses_preview_surface"
                    authoring_backend = self._authoring_backend_for_material_profile(
                        requested_profile=requested_profile,
                        resolved_profile=resolved_profile,
                        source_profile=source_profile,
                        fallback_reason=fallback_reason,
                        is_library_based=is_library_based,
                    )
                    self._append_profile_event(
                        result,
                        self._profile_event(
                            code="material_profile.unrecognized_graph_auto_preview",
                            message=(
                                f"Material '{material_name}' has an unrecognized "
                                "shader graph; auto mode will treat it as "
                                "PreviewSurface-compatible fallback output."
                            ),
                            material_name=material_name,
                            requested_profile=requested_profile,
                            resolved_profile=resolved_profile,
                            fallback_reason=fallback_reason,
                        ),
                    )
            elif requested_profile == "display_color":
                resolved_profile = "display_color"
                fallback_reason = None
                authoring_backend = "usd.primvars.displayColor"
            elif requested_profile == "preview_surface":
                if source_profile == "preview_surface":
                    resolved_profile = "preview_surface"
                    authoring_backend = self._authoring_backend_for_material_profile(
                        requested_profile=requested_profile,
                        resolved_profile=resolved_profile,
                        source_profile=source_profile,
                        fallback_reason=fallback_reason,
                        is_library_based=is_library_based,
                    )
                elif source_profile == "openpbr_materialx":
                    resolved_profile = "preview_surface"
                    fallback_reason = (
                        "explicit_preview_surface_requested_for_openpbr_materialx"
                    )
                    authoring_backend = self._authoring_backend_for_material_profile(
                        requested_profile=requested_profile,
                        resolved_profile=resolved_profile,
                        source_profile=source_profile,
                        fallback_reason=fallback_reason,
                        is_library_based=is_library_based,
                    )
                    self._append_profile_event(
                        result,
                        self._profile_event(
                            code="material_profile.preview_surface_overlay",
                            message=(
                                f"Material '{material_name}' is OpenPBR/MaterialX; "
                                "explicit preview_surface output will author a "
                                "UsdPreviewSurface overlay and suppress the "
                                "MaterialX surface connection."
                            ),
                            material_name=material_name,
                            requested_profile=requested_profile,
                            resolved_profile=resolved_profile,
                            fallback_reason=fallback_reason,
                        ),
                    )
                elif source_profile == "omnipbr_mdl":
                    resolved_profile = "preview_surface"
                    fallback_reason = "explicit_preview_surface_requested_for_mdl"
                    authoring_backend = self._authoring_backend_for_material_profile(
                        requested_profile=requested_profile,
                        resolved_profile=resolved_profile,
                        source_profile=source_profile,
                        fallback_reason=fallback_reason,
                        is_library_based=is_library_based,
                    )
                    self._append_profile_event(
                        result,
                        self._profile_event(
                            code="material_profile.mdl_to_preview_surface",
                            message=(
                                f"Material '{material_name}' is MDL; explicit "
                                "preview_surface output will author a portable "
                                "UsdPreviewSurface approximation."
                            ),
                            material_name=material_name,
                            requested_profile=requested_profile,
                            resolved_profile=resolved_profile,
                            fallback_reason=fallback_reason,
                        ),
                    )
                else:
                    self._append_profile_event(
                        result,
                        self._profile_event(
                            code="material_profile.unsupported_preview_surface_source",
                            severity="error",
                            message=(
                                f"Material '{material_name}' cannot be converted "
                                "to preview_surface by the apply task."
                            ),
                            material_name=material_name,
                            requested_profile=requested_profile,
                            resolved_profile=str(source_profile),
                        ),
                    )
            elif source_profile == requested_profile:
                resolved_profile = requested_profile
                authoring_backend = self._authoring_backend_for_material_profile(
                    requested_profile=requested_profile,
                    resolved_profile=resolved_profile,
                    source_profile=source_profile,
                    fallback_reason=fallback_reason,
                    is_library_based=is_library_based,
                )
            elif is_fallback_material_name(material_name):
                resolved_profile = "preview_surface"
                fallback_reason = "canonical_fallback_material"
                authoring_backend = self._authoring_backend_for_material_profile(
                    requested_profile=requested_profile,
                    resolved_profile=resolved_profile,
                    source_profile=source_profile,
                    fallback_reason=fallback_reason,
                    is_library_based=is_library_based,
                )
                self._append_profile_event(
                    result,
                    self._profile_event(
                        code="material_profile.canonical_fallback_material",
                        message=(
                            f"Material '{material_name}' is the canonical fallback "
                            "for unknown predictions; it must be authored as "
                            "UsdPreviewSurface even though "
                            f"{requested_profile!r} was requested."
                        ),
                        material_name=material_name,
                        requested_profile=requested_profile,
                        resolved_profile=resolved_profile,
                        fallback_reason=fallback_reason,
                    ),
                )
            else:
                self._append_profile_event(
                    result,
                    self._profile_event(
                        code="material_profile.unsupported_source_profile",
                        severity="error",
                        message=(
                            f"Material '{material_name}' resolves to "
                            f"{source_profile or 'unknown'} but "
                            f"{requested_profile} was requested."
                        ),
                        material_name=material_name,
                        requested_profile=requested_profile,
                        resolved_profile=str(source_profile),
                    ),
                )

            material_result = {
                "requested_profile": requested_profile,
                "source_profile": source_profile,
                "resolved_profile": resolved_profile,
                "fallback_reason": fallback_reason,
                "authoring_backend": authoring_backend,
                **source_details,
            }
            result["materials"][material_name] = material_result

        resolved_profiles = sorted(
            {
                str(info.get("resolved_profile"))
                for info in result["materials"].values()
                if info.get("resolved_profile")
            }
        )
        result["resolved_profiles"] = resolved_profiles
        if len(resolved_profiles) == 1:
            result["resolved_profile"] = resolved_profiles[0]
        elif resolved_profiles:
            result["resolved_profile"] = "mixed"

        fallback_reasons = sorted(
            {
                str(info.get("fallback_reason"))
                for info in result["materials"].values()
                if info.get("fallback_reason")
            }
        )
        if len(fallback_reasons) == 1:
            result["fallback_reason"] = fallback_reasons[0]
        elif fallback_reasons:
            result["fallback_reason"] = "mixed_material_profile_fallbacks"

        authoring_backends = sorted(
            {
                str(info.get("authoring_backend"))
                for info in result["materials"].values()
                if info.get("authoring_backend")
            }
        )
        result["authoring_backends"] = authoring_backends

        return result

    def _authoring_backend_for_material_profile(
        self,
        *,
        requested_profile: MaterialProfile,
        resolved_profile: MaterialProfile | str | None,
        source_profile: MaterialProfile | None,
        fallback_reason: str | None,
        is_library_based: bool,
    ) -> str:
        """Return an audit string naming the material authoring backend."""
        if requested_profile == "display_color" or resolved_profile == "display_color":
            return "usd.primvars.displayColor"
        if (
            fallback_reason
            == "explicit_preview_surface_requested_for_openpbr_materialx"
        ):
            return "world_understanding.openpbr_preview_overlay"
        if fallback_reason == "explicit_preview_surface_requested_for_mdl":
            return "usdshade.UsdPreviewSurface"
        if resolved_profile == "preview_surface" and not is_library_based:
            return "usdshade.UsdPreviewSurface"
        if resolved_profile == "omnipbr_mdl" and not is_library_based:
            return "world_understanding.add_mdl_material"
        if source_profile == resolved_profile and is_library_based:
            return "preserved_library_material_graph"
        if fallback_reason == "canonical_fallback_material":
            return "usdshade.UsdPreviewSurface"
        if resolved_profile:
            return "preserved_or_existing_material_graph"
        return "unresolved"

    def _raise_for_material_profile_errors(
        self,
        material_profile_result: dict[str, Any],
    ) -> None:
        errors = material_profile_result.get("errors") or []
        if not errors:
            return
        messages = "; ".join(error["message"] for error in errors)
        raise ValueError(f"Unsupported material profile request: {messages}")

    def _create_preview_surface_material(
        self,
        stage: Usd.Stage,
        material_name: str,
        material_prim_path: str,
        color: tuple[float, float, float] | Gf.Vec3f | None = None,
        roughness: float = 0.5,
        metallic: float = 0.0,
    ) -> str:
        """Author a simple UsdPreviewSurface material on the output stage."""
        path = Sdf.Path(material_prim_path)
        parent = path.GetParentPath()
        while parent != Sdf.Path.absoluteRootPath:
            if not stage.GetPrimAtPath(parent):
                UsdGeom.Scope.Define(stage, parent)
            parent = parent.GetParentPath()

        preview_color = color or self._get_material_color(material_name)
        if not isinstance(preview_color, Gf.Vec3f):
            preview_color = Gf.Vec3f(*preview_color)

        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path.AppendChild("PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            preview_color
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        )
        return material_prim_path

    def _create_material_on_stage(
        self,
        stage: Usd.Stage,
        material_name: str,
        material_path: str,
        output_usd_path: Path,
        path_prefix: str | None = None,
        material_profile: MaterialProfile = "auto",
    ) -> tuple[str | None, bool, dict[str, Any]]:
        """Create a material on the USD stage.

        Args:
            stage: USD stage to add material to
            material_name: Name of the material
            material_path: Path to the material file
            output_usd_path: Path to the output USD file (for relative path calculation)
            path_prefix: Optional path prefix for material (None for default, "" for root)
            material_profile: Requested material authoring profile

        Returns:
            Tuple of (material_prim_path, success, material_profile_metadata)
        """
        profile_metadata: dict[str, Any] = {
            "requested_profile": material_profile,
            "resolved_profile": None,
            "fallback_reason": None,
        }
        try:
            # Sanitize material name for USD path
            sanitized_name = self._sanitize_material_name(material_name)

            if is_fallback_material_name(material_name):
                material_prim_path = self._create_canonical_fallback_material(
                    stage,
                    self._fallback_material_path(path_prefix, sanitized_name),
                )
                self.listener.info(
                    f"Created fallback material '{material_name}' at {material_prim_path}"
                )
                profile_metadata.update(
                    {
                        "resolved_profile": "preview_surface",
                        "fallback_reason": "canonical_fallback_material",
                    }
                )
                return material_prim_path, True, profile_metadata

            if material_profile == "preview_surface":
                material_prim_path = self._fallback_material_path(
                    path_prefix,
                    sanitized_name,
                )
                self._create_preview_surface_material(
                    stage,
                    material_name,
                    material_prim_path,
                )
                profile_metadata["resolved_profile"] = "preview_surface"
                if material_path.endswith(".mdl"):
                    profile_metadata["fallback_reason"] = (
                        "explicit_preview_surface_requested_for_mdl"
                    )
                self.listener.info(
                    f"Created preview material '{material_name}' at {material_prim_path}"
                )
                return material_prim_path, True, profile_metadata

            # For MDL materials, use the proven add_mdl_material function
            if material_path.endswith(".mdl"):
                # Extract subIdentifier from the MDL filename (without .mdl extension)
                mdl_filename = Path(material_path).stem

                # Convert material path to be relative to the output USD file
                relative_material_path = self._make_path_relative_to_usd(
                    material_path, output_usd_path
                )

                # Add MDL material using proven utility
                stage, material_prim_path = add_mdl_material(
                    stage=stage,
                    material_name=sanitized_name,
                    source_asset_path=relative_material_path,
                    sub_identifier=mdl_filename,
                    path_prefix=path_prefix,
                    color=None,
                )

                self.listener.info(
                    f"Created material '{material_name}' at {material_prim_path}"
                )
                profile_metadata["resolved_profile"] = "omnipbr_mdl"
                return material_prim_path, True, profile_metadata
            else:
                # For non-MDL materials, create a basic UsdPreviewSurface
                # This is a fallback and should rarely be used
                self.listener.warning(
                    f"Material '{material_name}' is not an MDL file. "
                    f"Creating fallback UsdPreviewSurface material."
                )
                material_prim_path = self._fallback_material_path(
                    path_prefix,
                    sanitized_name,
                )
                self._create_preview_surface_material(
                    stage,
                    material_name,
                    material_prim_path,
                )

                self.listener.info(
                    f"Created fallback material '{material_name}' at {material_prim_path}"
                )
                profile_metadata.update(
                    {
                        "resolved_profile": "preview_surface",
                        "fallback_reason": "non_mdl_file_material_uses_preview_surface",
                    }
                )
                return material_prim_path, True, profile_metadata

        except Exception as e:
            self.listener.error(f"Failed to create material '{material_name}': {e}")
            return None, False, profile_metadata

    def _fallback_material_path(
        self, path_prefix: str | None, sanitized_name: str
    ) -> str:
        if path_prefix == "":
            return f"/Materials/{sanitized_name}"
        if path_prefix:
            return f"{path_prefix}/Looks/{sanitized_name}"
        return f"/Materials/{sanitized_name}"

    def _create_canonical_fallback_material(
        self, stage: Usd.Stage, material_prim_path: str
    ) -> str:
        """Author the neutral fallback material directly on the output stage."""
        return self._create_preview_surface_material(
            stage,
            FALLBACK_MATERIAL_NAME,
            material_prim_path,
            color=Gf.Vec3f(0.5, 0.5, 0.5),
            roughness=0.65,
            metallic=0.0,
        )

    def _author_material_binding_in_layer(
        self,
        root_layer: Sdf.Layer,
        prim_path: str,
        material_prim_path: str,
    ) -> None:
        """Author a material binding relationship directly on a layer."""
        prim_spec = root_layer.GetPrimAtPath(prim_path)
        if prim_spec is None:
            prim_spec = Sdf.CreatePrimInLayer(root_layer, prim_path)
            prim_spec.specifier = Sdf.SpecifierOver

        api_name = "MaterialBindingAPI"
        api_schemas = prim_spec.GetInfo("apiSchemas")
        api_schema_items = set()
        if api_schemas:
            api_schema_items.update(api_schemas.explicitItems)
            api_schema_items.update(api_schemas.prependedItems)
            api_schema_items.update(api_schemas.appendedItems)
            api_schema_items.update(api_schemas.addedItems)
        if not api_schemas:
            prim_spec.SetInfo(
                "apiSchemas",
                Sdf.TokenListOp.Create(prependedItems=[api_name]),
            )
        elif api_name not in api_schema_items:
            updated_api_schemas = Sdf.TokenListOp()
            updated_api_schemas.explicitItems = list(api_schemas.explicitItems)
            updated_api_schemas.prependedItems = list(api_schemas.prependedItems)
            updated_api_schemas.appendedItems = list(api_schemas.appendedItems)
            updated_api_schemas.addedItems = list(api_schemas.addedItems)
            updated_api_schemas.deletedItems = list(api_schemas.deletedItems)
            updated_api_schemas.orderedItems = list(api_schemas.orderedItems)
            if api_schemas.isExplicit:
                updated_api_schemas.explicitItems = [
                    *updated_api_schemas.explicitItems,
                    api_name,
                ]
            else:
                updated_api_schemas.prependedItems = [
                    api_name,
                    *updated_api_schemas.prependedItems,
                ]
            prim_spec.SetInfo("apiSchemas", updated_api_schemas)

        binding_rel = prim_spec.relationships.get(
            "material:binding"
        ) or Sdf.RelationshipSpec(prim_spec, "material:binding")
        binding_rel.custom = False
        binding_rel.targetPathList.ClearEditsAndMakeExplicit()
        binding_rel.targetPathList.explicitItems = [Sdf.Path(material_prim_path)]

    def _author_material_binding_on_stage(
        self,
        stage: Usd.Stage,
        prim_path: str,
        material_prim_path: str,
    ) -> None:
        """Author a material binding on the stage root layer."""
        self._author_material_binding_in_layer(
            stage.GetRootLayer(),
            prim_path,
            material_prim_path,
        )

    def _add_preview_surface_overlays_for_mdl_materials(
        self,
        stage: Usd.Stage,
        materials_applied: dict[str, str],
    ) -> int:
        """Author PreviewSurface overlays for copied library MDL materials."""
        updated = 0
        for material_name, material_path in materials_applied.items():
            source_profile, _details = self._detect_material_profile_on_stage(
                stage,
                material_path,
            )
            if source_profile != "omnipbr_mdl":
                continue

            self._create_preview_surface_material(
                stage,
                material_name,
                material_path,
            )
            material = UsdShade.Material(stage.GetPrimAtPath(material_path))
            material.CreateSurfaceOutput("mdl").GetAttr().SetConnections([])
            material.CreateDisplacementOutput("mdl").GetAttr().SetConnections([])
            material.CreateVolumeOutput("mdl").GetAttr().SetConnections([])
            updated += 1
        return updated

    def _display_color_for_material(self, material_name: str) -> Gf.Vec3f:
        color = self._get_material_color(material_name)
        return Gf.Vec3f(*color)

    def _author_display_color_on_prim(
        self,
        prim: Usd.Prim,
        material_name: str,
        opacity: float = 1.0,
    ) -> None:
        """Author displayColor/displayOpacity primvars on a composed prim."""
        color = self._display_color_for_material(material_name)
        primvars_api = UsdGeom.PrimvarsAPI(prim)
        display_color = primvars_api.CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.constant,
        )
        display_color.Set(Vt.Vec3fArray([color]))
        display_opacity = primvars_api.CreatePrimvar(
            "displayOpacity",
            Sdf.ValueTypeNames.FloatArray,
            UsdGeom.Tokens.constant,
        )
        display_opacity.Set(Vt.FloatArray([float(opacity)]))

    def _author_display_color_on_prim_spec(
        self,
        root_layer: Sdf.Layer,
        prim_path: str,
        material_name: str,
        opacity: float = 1.0,
        stage: Usd.Stage | None = None,
    ) -> Sdf.PrimSpec:
        """Author displayColor/displayOpacity primvars in a layer over spec."""

        def author_spec(path: str) -> Sdf.PrimSpec:
            prim_spec = Sdf.CreatePrimInLayer(root_layer, path)
            prim_spec.specifier = Sdf.SpecifierOver

            color_attr = prim_spec.attributes.get(
                "primvars:displayColor"
            ) or Sdf.AttributeSpec(
                prim_spec,
                "primvars:displayColor",
                Sdf.ValueTypeNames.Color3fArray,
                Sdf.VariabilityVarying,
            )
            color_attr.default = Vt.Vec3fArray(
                [self._display_color_for_material(material_name)]
            )
            color_attr.SetInfo("interpolation", UsdGeom.Tokens.constant)

            opacity_attr = prim_spec.attributes.get(
                "primvars:displayOpacity"
            ) or Sdf.AttributeSpec(
                prim_spec,
                "primvars:displayOpacity",
                Sdf.ValueTypeNames.FloatArray,
                Sdf.VariabilityVarying,
            )
            opacity_attr.default = Vt.FloatArray([float(opacity)])
            opacity_attr.SetInfo("interpolation", UsdGeom.Tokens.constant)

            binding_rel = prim_spec.relationships.get(
                "material:binding"
            ) or Sdf.RelationshipSpec(prim_spec, "material:binding")
            binding_rel.custom = False
            binding_rel.targetPathList.ClearEditsAndMakeExplicit()
            binding_rel.targetPathList.explicitItems = []
            return prim_spec

        prim_spec = author_spec(prim_path)

        if stage is not None:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                for child in Usd.PrimRange(prim):
                    if child.GetPath() == prim.GetPath():
                        continue
                    if child.IsA(UsdGeom.Subset):
                        author_spec(str(child.GetPath()))

        return prim_spec

    def _sanitize_material_name(self, material_name: str) -> str:
        """Sanitize material name for use as USD prim name.

        Replaces problematic characters (spaces, slashes, dashes) with underscores
        to ensure the name is valid for USD material/shader names.

        Args:
            material_name: Original material name from predictions

        Returns:
            Sanitized material name safe for USD
        """
        # Replace spaces, forward slashes, backslashes, and dashes with underscores
        sanitized = material_name.replace(" ", "_")
        sanitized = sanitized.replace("/", "_")
        sanitized = sanitized.replace("\\", "_")
        sanitized = sanitized.replace("-", "_")
        return sanitized

    def _load_prim_material_mapping(self, predictions_path: Path) -> dict[str, str]:
        """Load prim-to-material mapping from predictions file.

        Args:
            predictions_path: Path to the predictions JSONL file

        Returns:
            Dictionary mapping prim paths to material names
        """
        prim_to_material, _prediction_target_ids = self._load_prim_material_evidence(
            predictions_path
        )
        return prim_to_material

    def _load_prim_material_evidence(
        self,
        predictions_path: Path,
    ) -> tuple[dict[str, str], set[str]]:
        """Load actionable assignments and every prediction target ID."""
        prim_to_material: dict[str, str] = {}
        prediction_target_ids: set[str] = set()

        if not predictions_path or not Path(predictions_path).exists():
            self.listener.warning(f"Predictions file not found: {predictions_path}")
            return prim_to_material, prediction_target_ids

        try:
            with open(predictions_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        prediction = json.loads(line)
                        for prim_id, material in self._iter_prediction_mapping_records(
                            prediction
                        ):
                            if prim_id:
                                prediction_target_ids.add(prim_id)
                            if is_unknown_material_name(material):
                                self.listener.warning(
                                    f"Using fallback material for {prim_id}: "
                                    f"'{UNKNOWN_MATERIAL_SENTINEL}' -> "
                                    f"'{FALLBACK_MATERIAL_NAME}'"
                                )
                                material = FALLBACK_MATERIAL_NAME
                            if is_default_library_fallback_name(material):
                                self.listener.warning(
                                    f"Skipping unresolved default-library fallback "
                                    f"for {prim_id}: '{USE_DEFAULT_LIBRARY_SENTINEL}'"
                                )
                                continue

                            if prim_id and is_actionable_material_name(material):
                                normalized_material = normalize_material_name(material)
                                prim_to_material[prim_id] = normalized_material
                                self.listener.debug(
                                    f"Mapped {prim_id} -> {normalized_material}"
                                )

                    except json.JSONDecodeError as e:
                        self.listener.warning(f"Failed to parse prediction line: {e}")
                        continue

        except Exception as e:
            self.listener.error(f"Failed to load predictions file: {e}")

        return prim_to_material, prediction_target_ids

    def _iter_prediction_mapping_records(
        self,
        prediction: Any,
        fallback_id: str | None = None,
    ):
        """Yield ``(prim_id, material)`` pairs from flexible prediction payloads."""
        if isinstance(prediction, list):
            for item in prediction:
                yield from self._iter_prediction_mapping_records(item, fallback_id)
            return

        if isinstance(prediction, str):
            if fallback_id:
                yield fallback_id, prediction
            return

        if not isinstance(prediction, dict):
            return

        explicit_prim_id = self._prediction_prim_id(prediction)
        prim_id = explicit_prim_id or fallback_id
        has_material, material = self._prediction_material_value(prediction)
        if has_material or prim_id is not None:
            yield prim_id, material

        for container_key in PREDICTION_CONTAINER_KEYS:
            container = prediction.get(container_key)
            if isinstance(container, dict | list):
                yield from self._iter_prediction_mapping_records(container, prim_id)

        for key, value in prediction.items():
            if (
                key in PREDICTION_CONTAINER_KEYS
                or key in PREDICTION_ID_KEYS
                or key in PREDICTION_MATERIAL_KEYS
                or key in PREDICTION_VALIDATION_STATUS_KEYS
                or key == "materials"
            ):
                continue
            fallback = key if isinstance(key, str) and key.startswith("/") else None
            if fallback:
                # A slash-keyed record names a prediction target even when its
                # payload is empty, malformed, or otherwise not actionable.
                # The caller de-duplicates target IDs while retaining only
                # usable material assignments.
                yield fallback, None
            if isinstance(value, dict | list):
                yield from self._iter_prediction_mapping_records(value, fallback)
            elif isinstance(value, str) and fallback:
                yield fallback, value

    @staticmethod
    def _prediction_prim_id(prediction: dict[str, Any]) -> str | None:
        """Return the first supported prim identifier from a prediction record."""
        for key in PREDICTION_ID_KEYS:
            value = prediction.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _prediction_has_disallowed_unknown_status(
        prediction: dict[str, Any],
    ) -> bool:
        """Return True when validation durably marked a cleared unknown."""
        for key in PREDICTION_VALIDATION_STATUS_KEYS:
            if is_disallowed_unknown_validation_status(prediction.get(key)):
                return True

        materials = prediction.get("materials")
        if isinstance(materials, dict):
            for key in PREDICTION_VALIDATION_STATUS_KEYS:
                if is_disallowed_unknown_validation_status(materials.get(key)):
                    return True

        return False

    @staticmethod
    def _status_material_value(
        prediction: dict[str, Any], material: Any | None
    ) -> tuple[bool, Any | None]:
        """Preserve disallowed-unknown status when material was cleared."""
        if ApplyMaterialsToUSDTask._prediction_has_disallowed_unknown_status(
            prediction
        ) and not is_actionable_material_name(material):
            return True, UNKNOWN_MATERIAL_SENTINEL
        return True, material

    @staticmethod
    def _prediction_material_value(
        prediction: dict[str, Any],
    ) -> tuple[bool, Any | None]:
        """Return whether a prediction has a selected material and its value."""
        materials = prediction.get("materials")
        if isinstance(materials, dict):
            return ApplyMaterialsToUSDTask._status_material_value(
                prediction, materials.get("material")
            )
        if isinstance(materials, str):
            return ApplyMaterialsToUSDTask._status_material_value(prediction, materials)
        for key in PREDICTION_MATERIAL_KEYS:
            if key in prediction:
                return ApplyMaterialsToUSDTask._status_material_value(
                    prediction, prediction.get(key)
                )
        if ApplyMaterialsToUSDTask._prediction_has_disallowed_unknown_status(
            prediction
        ):
            return True, UNKNOWN_MATERIAL_SENTINEL
        return False, None

    def _count_prediction_materials(
        self, predictions_path: str | Path | None
    ) -> dict[str, int]:
        """Count actionable and unknown material predictions."""
        counts = {"total": 0, "actionable": 0, "unknown": 0, "missing": 0}
        if not predictions_path:
            return counts

        path = Path(predictions_path)
        if not path.exists():
            return counts

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prediction = json.loads(line)
                except json.JSONDecodeError as e:
                    listener = getattr(self, "listener", logger)
                    listener.warning(
                        f"Failed to parse prediction line while counting materials: {e}"
                    )
                    continue

                for material in self._iter_prediction_material_values(prediction):
                    counts["total"] += 1
                    if is_unknown_material_name(
                        material
                    ) or is_default_library_fallback_name(material):
                        counts["unknown"] += 1
                    elif is_actionable_material_name(material):
                        counts["actionable"] += 1
                    else:
                        counts["missing"] += 1

        return counts

    def _iter_prediction_material_values(
        self, prediction: Any, fallback_id: str | None = None
    ):
        """Yield material values from flexible prediction payload shapes."""
        if isinstance(prediction, list):
            for index, item in enumerate(prediction):
                child_fallback = (
                    f"{fallback_id}.{index}" if fallback_id is not None else None
                )
                yield from self._iter_prediction_material_values(
                    item,
                    fallback_id=child_fallback,
                )
            return

        if isinstance(prediction, str):
            yield prediction
            return

        if not isinstance(prediction, dict):
            return

        explicit_prim_id = self._prediction_prim_id(prediction)
        has_material, material = self._prediction_material_value(prediction)
        if has_material:
            yield material

        found_nested = has_material
        for container_key in PREDICTION_CONTAINER_KEYS:
            container = prediction.get(container_key)
            if isinstance(container, dict | list):
                found_nested = True
                yield from self._iter_prediction_material_values(container)

        for key, value in prediction.items():
            if (
                key in PREDICTION_CONTAINER_KEYS
                or key in PREDICTION_ID_KEYS
                or key in PREDICTION_MATERIAL_KEYS
                or key in PREDICTION_VALIDATION_STATUS_KEYS
                or key == "materials"
            ):
                continue
            if isinstance(value, dict):
                fallback = key if isinstance(key, str) and key.startswith("/") else None
                nested = list(self._iter_prediction_material_values(value, fallback))
                if nested:
                    found_nested = True
                    yield from nested
            elif (
                isinstance(value, str) and isinstance(key, str) and key.startswith("/")
            ):
                found_nested = True
                yield value

        if not found_nested and (
            explicit_prim_id is not None
            or "materials" in prediction
            or any(key in prediction for key in PREDICTION_MATERIAL_KEYS)
        ):
            yield None

    def _make_path_relative_to_usd(
        self, material_path: str, output_usd_path: Path
    ) -> str:
        """Convert a local material path to be relative to the output USD file.

        Args:
            material_path: Local path to the material file
            output_usd_path: Path to the output USD file

        Returns:
            Path to material relative to the USD file
        """
        if is_uri_asset_path(material_path):
            raise ValueError(
                "Refusing to author resolver URI material path into generated USD: "
                f"{material_path}"
            )

        try:
            # Convert both to absolute paths first
            material_abs = Path(material_path).resolve()
            usd_abs = Path(output_usd_path).resolve()

            # Get the directory containing the USD file
            usd_dir = usd_abs.parent

            # Compute relative path from USD directory to material file
            rel_path = os.path.relpath(material_abs, usd_dir)

            # Convert to forward slashes for USD
            rel_path = rel_path.replace("\\", "/")

            self.listener.info(
                f"Path relativization: material={material_path} -> abs={material_abs}, "
                f"usd={output_usd_path} -> abs={usd_abs}, usd_dir={usd_dir}, relative={rel_path}"
            )
            return rel_path
        except Exception as e:
            self.listener.warning(
                f"Failed to make path relative, using original: {material_path}. Error: {e}"
            )
            if is_unsafe_resolver_asset_path(material_path):
                raise ValueError(
                    "Refusing to author unsafe material path into generated USD: "
                    f"{material_path}"
                ) from e
            # Fall back to original path with forward slashes
            return material_path.replace("\\", "/")

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Apply materials to USD prims.

        Args:
            context: Workflow context
            object_store: Optional object store (not used)

        Returns:
            Updated context with applied materials
        """
        # Get event listener (or logger fallback) and store as instance variable
        # so helper methods can access it via self.listener
        listener = get_listener(context, logger_name=__name__)
        self.listener = listener

        input_usd_path = context.get("input_usd_path")
        output_usd_path = context.get("output_usd_path")
        resolved_materials = context.get("resolved_materials", {})
        layer_only = context.get("layer_only", False)
        flatten_output = context.get("flatten_output", True)
        predictions_path = context.get("predictions_path")
        is_library_based = context.get("is_library_based_mapping", False)
        material_library_path = context.get("material_library_path")
        skip_instance_check = context.get("skip_instance_check", False)
        requested_material_profile = self._requested_material_profile(context)
        context["material_profile"] = requested_material_profile
        material_profile_result = self._material_profile_result(
            requested_material_profile,
        )
        self._store_material_profile_result(context, material_profile_result)
        # Unified apply configs use the apply-specific key while legacy apply configs
        # use allow_empty_predictions directly.
        allow_empty_predictions = context.get(
            "apply_allow_empty_predictions",
            context.get("allow_empty_predictions", False),
        )
        if not isinstance(allow_empty_predictions, bool):
            raise ValueError(
                "allow_empty_predictions must be a boolean, got "
                f"{type(allow_empty_predictions).__name__}"
            )
        fail_on_unknown_material = context.get(
            "apply_fail_on_unknown_material",
            context.get("fail_on_unknown_material", False),
        )
        if not isinstance(fail_on_unknown_material, bool):
            raise ValueError(
                "fail_on_unknown_material must be a boolean, got "
                f"{type(fail_on_unknown_material).__name__}"
            )
        prediction_counts = self._count_prediction_materials(predictions_path)
        # Keep the exact prediction-derived target IDs alongside aggregate
        # assignment statistics.  Service callers need these IDs to distinguish
        # a unique material definition from an authored prim binding.
        prim_to_material, prediction_target_ids = self._load_prim_material_evidence(
            predictions_path
        )
        existing_unknown_count = context.get("unknown_material_predictions", 0)
        if not isinstance(existing_unknown_count, int):
            existing_unknown_count = 0
        context["unknown_material_predictions"] = max(
            existing_unknown_count,
            prediction_counts["unknown"],
        )
        unknown_material_predictions = context["unknown_material_predictions"]
        if prediction_counts["unknown"] > 0 and FALLBACK_MATERIAL_NAME not in (
            resolved_materials or {}
        ):
            resolved_materials = material_mapping_with_fallback(
                resolved_materials or {}
            )
            if set(resolved_materials) == {FALLBACK_MATERIAL_NAME}:
                is_library_based = False
                material_library_path = None
            self.listener.info(
                "Added canonical fallback material for "
                f"{prediction_counts['unknown']} unknown prediction(s)"
            )

        if not input_usd_path:
            raise ValueError("No input USD path provided")

        if not output_usd_path:
            raise ValueError("No output USD path provided")

        if fail_on_unknown_material and unknown_material_predictions > 0:
            unknown_count_for_message = max(
                prediction_counts["unknown"],
                unknown_material_predictions,
            )
            error_msg = (
                f"{unknown_count_for_message} material prediction(s) were "
                f"classified as '{UNKNOWN_MATERIAL_SENTINEL}'. "
                "fail_on_unknown_material=true requires every prediction to "
                "have an actionable material. This includes unknown predictions "
                "recorded by earlier validation steps, even if the sentinel was "
                "later cleared from the predictions file."
            )
            self.listener.error(error_msg)
            raise ValueError(error_msg)

        if not resolved_materials:
            # Check if we have predictions but no resolved materials (material resolution failure)
            if (
                prediction_counts["total"] > 0
                and prediction_counts["actionable"] == 0
                and prediction_counts["unknown"] > 0
            ):
                if allow_empty_predictions:
                    self.listener.warning(
                        f"{prediction_counts['unknown']}/"
                        f"{prediction_counts['total']} material prediction(s) "
                        f"were classified as '{UNKNOWN_MATERIAL_SENTINEL}'. "
                        "Skipping material application because empty prediction "
                        "application is explicitly allowed."
                    )
                    context["materials_applied"] = {}
                    context["assignment_stats"] = {
                        "total_prims": 0,
                        "materials_applied": 0,
                        "materials_created": 0,
                        "failed": 0,
                        "unknown": prediction_counts["unknown"],
                        "bound_prim_ids": [],
                        "unbound_prim_ids": sorted(prediction_target_ids),
                    }
                    return context

                error_msg = (
                    f"{prediction_counts['unknown']}/"
                    f"{prediction_counts['total']} material prediction(s) were "
                    f"classified as '{UNKNOWN_MATERIAL_SENTINEL}'. No material "
                    "bindings can be applied because the VLM reported no usable "
                    "visual evidence for those prims."
                )
                self.listener.error(error_msg)
                raise ValueError(error_msg)

            if (
                prediction_counts["total"] > 0
                and prediction_counts["actionable"] == 0
                and prediction_counts["missing"] > 0
            ):
                if allow_empty_predictions:
                    self.listener.warning(
                        f"{prediction_counts['missing']}/"
                        f"{prediction_counts['total']} material prediction(s) "
                        "did not contain actionable materials. Skipping material "
                        "application because empty prediction application is "
                        "explicitly allowed."
                    )
                    context["materials_applied"] = {}
                    context["assignment_stats"] = {
                        "total_prims": 0,
                        "materials_applied": 0,
                        "materials_created": 0,
                        "failed": 0,
                        "missing": prediction_counts["missing"],
                        "bound_prim_ids": [],
                        "unbound_prim_ids": sorted(prediction_target_ids),
                    }
                    return context

                error_msg = (
                    f"{prediction_counts['missing']}/"
                    f"{prediction_counts['total']} material prediction(s) did "
                    "not contain actionable material values. This can happen "
                    "when validation clears disallowed unknown sentinels or "
                    "when predictions omit material names."
                )
                self.listener.error(error_msg)
                raise ValueError(error_msg)

            predictions_exist = (
                predictions_path
                and Path(predictions_path).exists()
                and Path(predictions_path).stat().st_size > 0
            )
            if predictions_exist:
                # This is a critical failure - we have predictions but couldn't resolve any materials
                error_msg = (
                    "Critical error: Material resolution failed. "
                    "VLM predicted materials but none could be resolved from the material library. "
                    "This usually means:\n"
                    "  1. VLM returned material names that don't match the library (check system prompt)\n"
                    "  2. Material library is missing or incorrectly configured\n"
                    "  3. Material names in predictions don't match available materials\n"
                    "Check the MaterialRetrieval task logs for details."
                )
                self.listener.error(error_msg)
                raise ValueError(error_msg)
            else:
                if not allow_empty_predictions:
                    error_msg = (
                        "No material predictions were found; refusing to apply "
                        "materials with zero prediction-derived bindings. Set "
                        "allow_empty_predictions=true only for workflows that "
                        "intentionally permit empty material application."
                    )
                    self.listener.error(error_msg)
                    raise ValueError(error_msg)

                self.listener.warning(
                    "No resolved materials to apply (no predictions found)"
                )
                context["materials_applied"] = {}
                context["assignment_stats"] = {
                    "total_prims": 0,
                    "materials_applied": 0,
                    "materials_created": 0,
                    "failed": 0,
                    "bound_prim_ids": [],
                    "unbound_prim_ids": sorted(prediction_target_ids),
                }
                return context

        material_profile_result = self._resolve_material_profile_request(
            requested_profile=requested_material_profile,
            resolved_materials=resolved_materials,
            is_library_based=is_library_based,
            material_library_path=material_library_path,
        )
        self._store_material_profile_result(context, material_profile_result)
        self._raise_for_material_profile_errors(material_profile_result)

        self.listener.info(f"Applying {len(resolved_materials)} materials to USD")
        self.listener.info(
            "Material profile: "
            f"requested={requested_material_profile}, "
            f"resolved={material_profile_result.get('resolved_profile')}"
        )
        self.listener.info(f"Mode: {'Layer only' if layer_only else 'Full stage'}")
        if not layer_only:
            self.listener.info(
                f"Flatten: {'Yes (self-contained)' if flatten_output else 'No (preserves references)'}"
            )

        # Prediction evidence was loaded before material resolution so even an
        # explicit allow-empty return reports every unbound prediction target.
        self.listener.info(f"Loaded {len(prim_to_material)} prim-to-material mappings")
        if (
            isinstance(unknown_material_predictions, int)
            and unknown_material_predictions
        ):
            self.listener.warning(
                f"{unknown_material_predictions} prim(s) were classified as "
                f"'{UNKNOWN_MATERIAL_SENTINEL}' and will receive "
                f"'{FALLBACK_MATERIAL_NAME}'."
            )
        if prediction_counts["missing"] > 0:
            self.listener.warning(
                f"{prediction_counts['missing']}/"
                f"{prediction_counts['total']} material prediction(s) do not "
                "contain an actionable material and will not receive material "
                "bindings."
            )
        if not prim_to_material and not allow_empty_predictions:
            if (
                isinstance(unknown_material_predictions, int)
                and unknown_material_predictions
            ):
                error_msg = (
                    f"{unknown_material_predictions} material prediction(s) "
                    f"were classified as '{UNKNOWN_MATERIAL_SENTINEL}'. No "
                    "material bindings can be applied because the VLM reported "
                    "no usable visual evidence for those prims."
                )
            else:
                error_msg = (
                    "No material predictions were loaded from "
                    f"{predictions_path}; refusing to apply materials with zero "
                    "prediction-derived bindings. Set allow_empty_predictions=true "
                    "only for workflows that intentionally permit empty material "
                    "application."
                )
            self.listener.error(error_msg)
            raise ValueError(error_msg)

        # Statistics tracking
        materials_applied = {}
        materials_created_count = 0
        prims_with_materials = 0
        failed_count = 0

        try:
            if layer_only:
                # Create only a material binding layer
                stage, materials_applied, stats = self._create_material_layer(
                    input_usd_path,
                    output_usd_path,
                    resolved_materials,
                    prim_to_material,
                    is_library_based,
                    material_library_path,
                    skip_instance_check=skip_instance_check,
                    material_profile=requested_material_profile,
                    prediction_target_ids=prediction_target_ids,
                )
            else:
                # Create a complete new stage with materials
                stage, materials_applied, stats = self._create_full_stage(
                    input_usd_path,
                    output_usd_path,
                    resolved_materials,
                    prim_to_material,
                    is_library_based,
                    material_library_path,
                    flatten_output,
                    skip_instance_check=skip_instance_check,
                    material_profile=requested_material_profile,
                    prediction_target_ids=prediction_target_ids,
                )

            materials_created_count = stats["materials_created"]
            prims_with_materials = stats["prims_with_materials"]
            failed_count = stats.get("failed", 0)

            # Save the stage (skip if already saved during flattening)
            if not (not layer_only and flatten_output):
                self.listener.info(f"Saving USD to {output_usd_path}")
                stage.GetRootLayer().Export(str(output_usd_path))
            else:
                self.listener.info(
                    f"USD already saved during flattening: {output_usd_path}"
                )

        except Exception as e:
            # A partially authored/exported USD is not reliable output. Fail closed
            # so callers see the authoring/export failure instead of a false success.
            self.listener.error(f"Failed to apply materials to USD: {e}")
            try:
                output_path = Path(output_usd_path)
                input_path = Path(input_usd_path)
                if (
                    output_path.exists()
                    and output_path.resolve() != input_path.resolve()
                ):
                    output_path.unlink()
                    self.listener.warning(
                        f"Removed partial material output USD: {output_path}"
                    )
            except Exception as cleanup_error:
                self.listener.warning(
                    "Failed to remove partial material output USD "
                    f"{output_usd_path}: {cleanup_error}"
                )
            raise

        # Calculate statistics
        assignment_stats = {
            "total_prims": prims_with_materials,
            "materials_applied": len(materials_applied),
            "materials_created": materials_created_count,
            "failed": failed_count,
            "unknown": unknown_material_predictions
            if isinstance(unknown_material_predictions, int)
            else 0,
            "bound_prim_ids": stats.get("bound_prim_ids", []),
            "unbound_prim_ids": stats.get("unbound_prim_ids", []),
        }

        self.listener.info(
            f"Material application completed: "
            f"{prims_with_materials} prims updated, "
            f"{materials_created_count} materials created, "
            f"{failed_count} failed"
        )

        # Update context
        context["output_usd_path"] = output_usd_path
        context["materials_applied"] = materials_applied
        context["assignment_stats"] = assignment_stats
        self._store_material_profile_result(context, material_profile_result)

        return context

    def _apply_materials_to_instances(
        self,
        stage: Usd.Stage,
        prim_to_material: dict[str, str],
        materials_applied: dict[str, str],
    ) -> dict[str, int]:
        """Apply materials to instance prims by looking up their master's material.

        This handles instances that don't have direct predictions by finding
        their master/prototype prim and applying the master's predicted material.

        Args:
            stage: USD stage
            prim_to_material: Dictionary mapping prim paths to material names from predictions
            materials_applied: Dictionary mapping material names to material prim paths

        Returns:
            Statistics dictionary with counts
        """
        self.listener.info("Checking for instances without predictions...")
        instances_found = 0
        instances_applied = 0
        instances_skipped = 0

        # Traverse ALL prims in the stage to find instances
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())

            # Skip if already has a material assignment from predictions
            if prim_path in prim_to_material:
                continue

            # Check if this is an instance
            if not prim.IsInstance():
                continue

            instances_found += 1

            # Get the instance's master/prototype
            master = prim.GetPrototype()
            if not master or not master.IsValid():
                self.listener.debug(f"Instance {prim_path} has no valid prototype")
                instances_skipped += 1
                continue

            master_path = str(master.GetPath())

            # Check if master has a material prediction
            master_material = prim_to_material.get(master_path)
            if not master_material:
                self.listener.debug(
                    f"Master {master_path} has no prediction for instance {prim_path}"
                )
                instances_skipped += 1
                continue

            # Get material prim path
            material_prim_path = materials_applied.get(master_material)
            if not material_prim_path:
                self.listener.warning(
                    f"Material '{master_material}' not available for instance {prim_path}"
                )
                instances_skipped += 1
                continue

            # Nullify existing material and apply master's material
            try:
                nullify_material(prim)
                self._author_material_binding_on_stage(
                    stage,
                    prim_path,
                    material_prim_path,
                )
                instances_applied += 1
                self.listener.debug(
                    f"Applied {master_material} to instance {prim_path} "
                    f"(from master {master_path})"
                )
            except Exception as e:
                self.listener.warning(
                    f"Failed to apply material to instance {prim_path}: {e}"
                )
                instances_skipped += 1
                continue

        if instances_found > 0:
            self.listener.info(
                f"Instance materials: {instances_applied} applied, "
                f"{instances_skipped} skipped ({instances_found} total instances)"
            )

        return {
            "instances_found": instances_found,
            "instances_applied": instances_applied,
            "instances_skipped": instances_skipped,
        }

    def _get_local_instance_reference_map(
        self, stage: Usd.Stage
    ) -> dict[str, str | None]:
        """Map instance roots to their local referenced source prim path.

        USD instance proxies are read-only.  For same-layer internal
        references, authoring a stronger opinion at the referenced source prim
        path lets all instances inherit the override through composition.
        External asset references cannot be overridden this way, so they map
        to None.
        """
        instance_root_to_ref_prim: dict[str, str | None] = {}
        for prim in stage.Traverse():
            if not prim.IsInstance():
                continue

            ref_path: str | None = None
            for spec in prim.GetPrimStack():
                added = spec.referenceList.GetAddedOrExplicitItems()
                if added:
                    ref = added[0]
                    if not ref.assetPath and ref.primPath:
                        ref_path = str(ref.primPath)
                    break

            instance_root_to_ref_prim[str(prim.GetPath())] = ref_path

        return instance_root_to_ref_prim

    def _remap_instance_binding_target(
        self,
        prim_path: str,
        instance_root_to_ref_prim: dict[str, str | None],
    ) -> tuple[str, bool, bool]:
        """Return the authorable binding path for a predicted prim path.

        Returns:
            Tuple of (binding_target_path, remapped, skip).  ``skip`` is true
            when the prim is under an instance backed by an external reference.
        """
        for instance_root in sorted(instance_root_to_ref_prim, key=len, reverse=True):
            if prim_path != instance_root and not prim_path.startswith(
                instance_root + "/"
            ):
                continue

            ref_prim = instance_root_to_ref_prim[instance_root]
            if not ref_prim:
                return prim_path, False, True

            suffix = prim_path[len(instance_root) :]
            return ref_prim + suffix, True, False

        return prim_path, False, False

    def _group_binding_assignments(
        self,
        prim_to_material: dict[str, str],
        instance_root_to_ref_prim: dict[str, str | None],
    ) -> tuple[list[tuple[str, str, list[str], bool]], int, int]:
        """Group predictions by the USD path where an opinion is authored.

        Several instance/prototype paths can resolve to one shared authoring
        target. Identical assignments are authored once and credit every source
        prediction. Conflicting materials are not authored because choosing one
        would silently overwrite the other while falsely reporting both bound.
        """
        grouped: dict[str, dict[str, list[tuple[str, bool]]]] = {}
        remapped_count = 0
        skipped_count = 0
        for prim_path, material_name in prim_to_material.items():
            binding_target_path, was_remapped, skip = (
                self._remap_instance_binding_target(
                    prim_path,
                    instance_root_to_ref_prim,
                )
            )
            if skip:
                skipped_count += 1
                self.listener.debug(
                    f"Skipping {prim_path}: instance references external asset"
                )
                continue
            if was_remapped:
                remapped_count += 1
            grouped.setdefault(binding_target_path, {}).setdefault(
                material_name, []
            ).append((prim_path, was_remapped))

        assignments: list[tuple[str, str, list[str], bool]] = []
        for binding_target_path, by_material in grouped.items():
            if len(by_material) != 1:
                source_ids = sorted(
                    source_id
                    for sources in by_material.values()
                    for source_id, _was_remapped in sources
                )
                self.listener.warning(
                    "Conflicting material assignments target shared USD path "
                    f"{binding_target_path}: materials={sorted(by_material)}, "
                    f"source_prims={source_ids}. Leaving all source prims unbound."
                )
                continue
            material_name, sources = next(iter(by_material.items()))
            assignments.append(
                (
                    binding_target_path,
                    material_name,
                    [source_id for source_id, _was_remapped in sources],
                    any(was_remapped for _source_id, was_remapped in sources),
                )
            )

        return assignments, remapped_count, skipped_count

    def _copy_library_materials(
        self,
        stage: Usd.Stage,
        library_path: str,
        output_usd_path: Path,
        resolved_materials: dict[str, str],
        default_prim_name: str = "",
    ) -> tuple[Usd.Stage, dict]:
        """Copy only the used materials from a library into the output stage.

        Instead of sublayering the entire library (which includes all materials
        and breaks texture paths when flattened), this copies only the materials
        that are actually used and remaps their asset paths (textures, MDL files)
        to be relative to the output USD.

        Material paths from the library (e.g. /World/Looks/Iron) are remapped
        to sit under the asset's default prim (e.g. /MyAsset/Looks/Iron) so
        the output doesn't introduce extra root prims like /World.

        Args:
            stage: Output USD stage to copy materials into
            library_path: Path to the material library USD file
            output_usd_path: Path to the output USD file
            resolved_materials: Dict mapping material name -> prim path in library
            default_prim_name: Name of the default prim to place materials under

        Returns:
            Tuple of (stage, materials_applied dict)
        """
        materials_applied = {}

        try:
            # Validate library path exists before attempting to open
            if not Path(library_path).exists():
                self.listener.error(f"Material library file not found: {library_path}")
                return stage, materials_applied

            # Open library layer (read-only, no sublayering)
            library_layer = Sdf.Layer.FindOrOpen(str(library_path))
            if not library_layer:
                self.listener.error(f"Failed to open material library: {library_path}")
                return stage, materials_applied

            output_layer = stage.GetRootLayer()

            # Compute directory paths for asset path remapping
            library_dir = Path(library_path).resolve().parent
            output_dir = Path(output_usd_path).resolve().parent

            # Remap library material paths to sit under the asset's
            # default prim instead of the library's own root (e.g.
            # /World/Looks/Iron -> /MyAsset/Looks/Iron).  This avoids
            # creating extra root prims like /World in the output.
            def _remap_target(lib_path: str) -> str:
                if not default_prim_name:
                    return lib_path
                parts = lib_path.strip("/").split("/")
                # Library paths typically look like /World/Looks/MatName
                # or /RootPrim/Looks/MatName.  Replace the first component
                # (the library's root) with the asset's default prim.
                if len(parts) >= 2:
                    parts[0] = default_prim_name
                return "/" + "/".join(parts)

            # Build remapped target paths
            target_materials: dict[str, tuple[str, str]] = {}
            for material_name, lib_path in resolved_materials.items():
                target_path = _remap_target(lib_path)
                target_materials[material_name] = (lib_path, target_path)
                if target_path != lib_path:
                    self.listener.debug(
                        f"Remapped material path: {lib_path} -> {target_path}"
                    )

            # Ensure parent prim hierarchy exists for remapped targets
            parent_paths: set[str] = set()
            for _, target_path in target_materials.values():
                path = Sdf.Path(target_path)
                parent = path.GetParentPath()
                while parent != Sdf.Path.absoluteRootPath:
                    parent_paths.add(str(parent))
                    parent = parent.GetParentPath()

            for parent_path in sorted(parent_paths):
                if not output_layer.GetPrimAtPath(parent_path):
                    Sdf.CreatePrimInLayer(output_layer, parent_path)
                # Ensure parent prims are 'def' (not 'over') so they are
                # traversable by stage.Traverse() and visible to renderers
                prim_spec = output_layer.GetPrimAtPath(parent_path)
                if prim_spec and prim_spec.specifier != Sdf.SpecifierDef:
                    prim_spec.specifier = Sdf.SpecifierDef
                    self.listener.debug(f"Created parent prim: {parent_path}")
                ensure_looks_scope_spec(output_layer, parent_path)

            # Copy each used material from library to output
            for material_name, (lib_path, target_path) in target_materials.items():
                source_spec = library_layer.GetPrimAtPath(lib_path)
                if not source_spec:
                    if is_fallback_material_name(material_name):
                        self._create_canonical_fallback_material(stage, target_path)
                        materials_applied[material_name] = target_path
                        self.listener.info(
                            f"Created fallback material '{material_name}' at "
                            f"{target_path}"
                        )
                        continue
                    self.listener.error(
                        f"Material prim not found in library for "
                        f"'{material_name}': {lib_path}"
                    )
                    continue

                success = Sdf.CopySpec(
                    library_layer,
                    Sdf.Path(lib_path),
                    output_layer,
                    Sdf.Path(target_path),
                )

                if success:
                    self._remap_asset_paths_in_prim(
                        output_layer,
                        Sdf.Path(target_path),
                        library_dir,
                        output_dir,
                    )
                    cleared_color_spaces = (
                        self._clear_color_space_on_empty_asset_inputs(
                            output_layer,
                            Sdf.Path(target_path),
                        )
                    )
                    if cleared_color_spaces:
                        self.listener.debug(
                            "Cleared colorSpace metadata from "
                            f"{cleared_color_spaces} empty asset input(s) on "
                            f"material '{material_name}'"
                        )
                    materials_applied[material_name] = target_path
                    self.listener.info(
                        f"Copied material '{material_name}' to {target_path}"
                    )
                else:
                    self.listener.error(
                        f"Failed to copy material '{material_name}' from library"
                    )

            # Save and reopen: Sdf.CopySpec operates at the layer level,
            # so the Usd.Stage's composition cache is stale. Reopening
            # ensures material bindings and prim traversal reflect the
            # newly copied specs.
            stage.Save()
            stage = Usd.Stage.Open(str(output_usd_path))
            if not stage:
                raise RuntimeError(
                    f"Failed to reopen stage after saving: {output_usd_path}"
                )

            self.listener.info(
                f"✓ Copied {len(materials_applied)} materials from library "
                f"(out of {len(resolved_materials)} requested)"
            )

        except Exception as e:
            self.listener.error(f"Failed to copy library materials: {e}")
            self.listener.debug(traceback.format_exc())

        return stage, materials_applied

    def _fix_stale_default_prim(self, stage: Usd.Stage, original_name: str) -> None:
        """Detect and correct a stale defaultPrim after composition.

        The NVCF optimizer may wrap content under a new root (e.g. /World),
        making the original defaultPrim name stale. This detects the mismatch
        and updates defaultPrim to the actual root prim.

        Args:
            stage: USD stage to check
            original_name: The original defaultPrim name from the input
        """
        dp = stage.GetDefaultPrim()
        if not dp.IsValid():
            root_children = list(stage.GetPseudoRoot().GetChildren())
            if root_children:
                actual_root_name = root_children[0].GetName()
                stage.GetRootLayer().defaultPrim = actual_root_name
                self.listener.warning(
                    f"Default prim '{original_name}' not found in composed "
                    f"stage. Updated to actual root prim: /{actual_root_name}"
                )

    def _clear_color_space_on_empty_asset_inputs(
        self,
        layer: Sdf.Layer,
        prim_path: Sdf.Path,
    ) -> int:
        """Clear colorSpace metadata from empty asset-valued material inputs."""
        return clear_color_space_on_empty_asset_inputs(layer, prim_path)

    def _collect_bound_material_paths(self, stage: Usd.Stage) -> set[str]:
        """Collect material prim paths resolved by authored material bindings."""
        material_paths: set[str] = set()
        material_purposes = (
            UsdShade.Tokens.allPurpose,
            UsdShade.Tokens.preview,
            UsdShade.Tokens.full,
        )
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            binding_api = UsdShade.MaterialBindingAPI(prim)
            for material_purpose in material_purposes:
                material, _binding_rel = binding_api.ComputeBoundMaterial(
                    material_purpose
                )
                material_prim = material.GetPrim() if material else Usd.Prim()
                if material_prim and material_prim.IsValid():
                    material_paths.add(str(material_prim.GetPath()))

        return material_paths

    def _is_uri_asset_path(self, path: str) -> bool:
        """Return whether an asset path string is a URI."""
        return is_uri_asset_path(path)

    def _is_absolute_asset_path(self, path: str) -> bool:
        """Return whether an asset path is absolute on POSIX or Windows."""
        return is_absolute_asset_path(path)

    def _asset_path_to_string(self, value: object) -> str:
        """Return the authored string for an Sdf.AssetPath-like value."""
        try:
            return value.path if hasattr(value, "path") else str(value)
        except Exception:
            return str(value)

    def _asset_base_dirs_for_attr(
        self,
        stage: Usd.Stage,
        attr: Usd.Attribute,
    ) -> list[Path]:
        """Return candidate layer directories for a composed asset attribute."""
        base_dirs: list[Path] = []
        for spec in attr.GetPropertyStack():
            layer = getattr(spec, "layer", None)
            real_path = getattr(layer, "realPath", "") if layer else ""
            if real_path:
                base_dirs.append(Path(real_path).parent)

        root_layer = stage.GetRootLayer()
        base_dirs.append(
            Path(root_layer.realPath).parent if root_layer.realPath else Path.cwd()
        )

        unique_base_dirs: list[Path] = []
        seen: set[Path] = set()
        for base_dir in base_dirs:
            if base_dir in seen:
                continue
            seen.add(base_dir)
            unique_base_dirs.append(base_dir)
        return unique_base_dirs

    def _is_unresolved_local_asset_path(
        self,
        asset_value: object,
        authored_path: str,
        base_dirs: Path | list[Path],
    ) -> bool:
        """Return true when an asset path is unsafe or cannot be resolved locally."""
        if not authored_path:
            return False
        if self._is_uri_asset_path(authored_path):
            return True

        resolved_path = str(getattr(asset_value, "resolvedPath", ""))
        if resolved_path:
            return False

        search_dirs = [base_dirs] if isinstance(base_dirs, Path) else list(base_dirs)
        candidates = (
            [Path(authored_path)]
            if self._is_absolute_asset_path(authored_path)
            else [base_dir / authored_path for base_dir in search_dirs]
        )
        for candidate in candidates:
            try:
                if candidate.exists():
                    return False
            except (OSError, ValueError):
                continue
        return True

    def _collect_unresolved_mdl_shader_paths(
        self,
        stage: Usd.Stage,
        material_prim: Usd.Prim,
    ) -> list[str]:
        """Return shader prim paths under a material with unresolved MDL assets."""
        shader_paths: list[str] = []
        material_path = str(material_prim.GetPath())

        for prim in Usd.PrimRange(material_prim, Usd.PrimAllPrimsPredicate):
            if not prim.IsActive():
                continue
            if not prim.IsA(UsdShade.Shader):
                continue
            shader_material_path = self._composition_target_material_path(
                stage,
                prim.GetPath(),
            )
            if shader_material_path != material_path:
                continue

            mdl_attr = prim.GetAttribute("info:mdl:sourceAsset")
            if not mdl_attr or not mdl_attr.IsValid():
                continue

            asset_value = mdl_attr.Get()
            if asset_value is None:
                continue

            authored_path = self._asset_path_to_string(asset_value)
            if self._is_unresolved_local_asset_path(
                asset_value,
                authored_path,
                self._asset_base_dirs_for_attr(stage, mdl_attr),
            ):
                shader_paths.append(str(prim.GetPath()))

        return shader_paths

    def _collect_unresolved_mdl_shader_paths_from_prims(
        self,
        stage: Usd.Stage,
        prim_paths: set[str],
    ) -> list[str]:
        """Return unresolved MDL shader paths from the given prim paths."""
        shader_paths: list[str] = []

        for prim_path in sorted(prim_paths):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid() or not prim.IsActive():
                continue
            if not prim.IsA(UsdShade.Shader):
                continue

            mdl_attr = prim.GetAttribute("info:mdl:sourceAsset")
            if not mdl_attr or not mdl_attr.IsValid():
                continue

            asset_value = mdl_attr.Get()
            if asset_value is None:
                continue

            authored_path = self._asset_path_to_string(asset_value)
            if self._is_unresolved_local_asset_path(
                asset_value,
                authored_path,
                self._asset_base_dirs_for_attr(stage, mdl_attr),
            ):
                shader_paths.append(prim_path)

        return shader_paths

    def _composition_target_material_path(
        self,
        stage: Usd.Stage,
        target_path: Sdf.Path,
    ) -> str | None:
        """Return the owning material path for a composition target, if any."""
        target_prim = stage.GetPrimAtPath(target_path)
        while target_prim:
            if target_prim.IsA(UsdShade.Material):
                return str(target_prim.GetPath())
            target_prim = target_prim.GetParent()
        return None

    def _composition_target_paths(self, prim: Usd.Prim) -> list[Sdf.Path]:
        """Return direct composition target paths authored on a prim."""
        target_paths = list(prim.GetInherits().GetAllDirectInherits())

        specializes = prim.GetMetadata("specializes")
        if specializes is not None:
            target_paths.extend(specializes.GetAppliedItems())

        references = prim.GetMetadata("references")
        if references is not None:
            for reference in references.GetAppliedItems():
                if getattr(reference, "assetPath", ""):
                    # External prim paths are scoped to the referenced layer.
                    # Resolving them in this stage can protect unrelated local
                    # material paths with the same name.
                    continue
                target_path = getattr(reference, "primPath", Sdf.Path.emptyPath)
                if target_path:
                    target_paths.append(target_path)

        payloads = prim.GetMetadata("payload")
        if payloads is not None:
            for payload in payloads.GetAppliedItems():
                if getattr(payload, "assetPath", ""):
                    # External prim paths are scoped to the referenced layer.
                    # Resolving them in this stage can protect unrelated local
                    # material paths with the same name.
                    continue
                target_path = getattr(payload, "primPath", Sdf.Path.emptyPath)
                if target_path:
                    target_paths.append(target_path)

        return target_paths

    def _collect_composition_target_material_paths(
        self,
        stage: Usd.Stage,
        root_material_paths: set[str],
        protected_material_paths: set[str] | None = None,
    ) -> set[str]:
        """Collect composition-target material prims reachable from material roots."""
        protected_material_paths = protected_material_paths or set()
        material_paths: set[str] = set()
        material_queue = list(root_material_paths)
        queued_material_paths = set(root_material_paths)
        visited_material_paths: set[str] = set()

        while material_queue:
            root_material_path = material_queue.pop()
            if root_material_path in visited_material_paths:
                continue
            visited_material_paths.add(root_material_path)

            root_material_prim = stage.GetPrimAtPath(root_material_path)
            if not root_material_prim:
                continue

            if root_material_path in protected_material_paths:
                prim_paths = self._collect_material_graph_prim_paths(
                    stage,
                    root_material_prim,
                )
            else:
                prim_paths = {
                    str(prim.GetPath())
                    for prim in Usd.PrimRange(
                        root_material_prim,
                        Usd.PrimAllPrimsPredicate,
                    )
                    if prim.IsActive()
                }
                prim_paths.update(
                    self._collect_material_graph_prim_paths(stage, root_material_prim)
                )

            prims = [stage.GetPrimAtPath(prim_path) for prim_path in prim_paths]
            for prim in prims:
                if not prim or not prim.IsValid():
                    continue
                for target_path in self._composition_target_paths(prim):
                    material_path = self._composition_target_material_path(
                        stage,
                        target_path,
                    )
                    if material_path:
                        if (
                            material_path not in queued_material_paths
                            and material_path not in visited_material_paths
                        ):
                            material_queue.append(material_path)
                            queued_material_paths.add(material_path)
                        material_paths.add(material_path)

        return material_paths

    def _collect_material_graph_prim_paths(
        self,
        stage: Usd.Stage,
        material_prim: Usd.Prim,
    ) -> set[str]:
        """Collect prim paths reachable from a material through its graph."""
        prim_paths: set[str] = set()
        prim_queue: list[Usd.Prim] = []
        queued_prim_paths: set[str] = set()

        def queue_prim(prim: Usd.Prim) -> None:
            if not prim or not prim.IsValid() or not prim.IsActive():
                return
            prim_path = str(prim.GetPath())
            if prim_path in prim_paths or prim_path in queued_prim_paths:
                return
            queued_prim_paths.add(prim_path)
            prim_queue.append(prim)

        queue_prim(material_prim)

        while prim_queue:
            prim = prim_queue.pop()
            if not prim or not prim.IsValid() or not prim.IsActive():
                continue

            prim_path = str(prim.GetPath())
            if prim_path in prim_paths:
                continue
            prim_paths.add(prim_path)

            for target_path in self._composition_target_paths(prim):
                target_prim = stage.GetPrimAtPath(target_path)
                queue_prim(target_prim)

            for attr in prim.GetAttributes():
                for connection_path in attr.GetConnections():
                    target_prim_path = connection_path.GetPrimPath()
                    if not target_prim_path:
                        continue
                    target_prim = stage.GetPrimAtPath(target_prim_path)
                    queue_prim(target_prim)

        return prim_paths

    def _collect_connected_material_paths(
        self,
        stage: Usd.Stage,
        root_material_paths: set[str],
        protected_material_paths: set[str] | None = None,
    ) -> set[str]:
        """Collect material prims reached through connections from bound materials."""
        protected_material_paths = protected_material_paths or set()
        material_paths: set[str] = set()
        prim_queue: list[Usd.Prim] = []
        visited_prim_paths: set[str] = set()
        queued_prim_paths: set[str] = set()

        def queue_prim(prim: Usd.Prim) -> None:
            if not prim or not prim.IsValid() or not prim.IsActive():
                return
            prim_path = str(prim.GetPath())
            if prim_path in visited_prim_paths or prim_path in queued_prim_paths:
                return
            queued_prim_paths.add(prim_path)
            prim_queue.append(prim)

        for material_path in root_material_paths:
            material_prim = stage.GetPrimAtPath(material_path)
            if not material_prim:
                continue
            if material_path in protected_material_paths:
                for prim_path in self._collect_material_graph_prim_paths(
                    stage,
                    material_prim,
                ):
                    queue_prim(stage.GetPrimAtPath(prim_path))
            else:
                for prim_path in self._collect_material_graph_prim_paths(
                    stage,
                    material_prim,
                ):
                    queue_prim(stage.GetPrimAtPath(prim_path))
                for prim in Usd.PrimRange(
                    material_prim,
                    Usd.PrimAllPrimsPredicate,
                ):
                    if prim.IsActive():
                        queue_prim(prim)

        while prim_queue:
            prim = prim_queue.pop()
            if not prim or not prim.IsValid() or not prim.IsActive():
                continue
            prim_path = str(prim.GetPath())
            if prim_path in visited_prim_paths:
                continue
            visited_prim_paths.add(prim_path)

            for attr in prim.GetAttributes():
                for connection_path in attr.GetConnections():
                    target_prim_path = connection_path.GetPrimPath()
                    if not target_prim_path:
                        continue
                    material_path = self._composition_target_material_path(
                        stage,
                        target_prim_path,
                    )
                    if material_path:
                        material_paths.add(material_path)

                    target_prim = stage.GetPrimAtPath(target_prim_path)
                    queue_prim(target_prim)

        return material_paths

    def _collect_reachable_shader_prim_paths(
        self,
        stage: Usd.Stage,
        reachable_material_paths: set[str],
        protected_material_paths: set[str],
    ) -> set[str]:
        """Collect shader prims already protected by reachable material roots."""
        shader_paths: set[str] = set()

        for material_path in reachable_material_paths:
            material_prim = stage.GetPrimAtPath(material_path)
            if not material_prim:
                continue

            if material_path in protected_material_paths:
                prim_paths = self._collect_material_graph_prim_paths(
                    stage,
                    material_prim,
                )
            else:
                prim_paths = {
                    str(prim.GetPath())
                    for prim in Usd.PrimRange(
                        material_prim,
                        Usd.PrimAllPrimsPredicate,
                    )
                    if prim.IsActive()
                }
                prim_paths.update(
                    self._collect_material_graph_prim_paths(stage, material_prim)
                )

            for prim_path in prim_paths:
                prim = stage.GetPrimAtPath(prim_path)
                if (
                    prim
                    and prim.IsValid()
                    and prim.IsActive()
                    and prim.IsA(UsdShade.Shader)
                ):
                    shader_paths.add(prim_path)

        return shader_paths

    def _deactivate_unbound_unresolved_mdl_shaders(
        self,
        stage: Usd.Stage,
        protected_material_paths: set[str] | None = None,
    ) -> list[str]:
        """Hide stale unbound MDL shaders that would preserve invalid asset paths.

        Materialized outputs are authored as an overlay on top of the input USD.
        Once new material bindings are written, old input material definitions can
        remain in the composed stage even when nothing binds to them anymore. Some
        validators still inspect those stale material prims, so unresolved MDL
        source assets such as ``OmniPBR.mdl`` must be blocked from the output.

        The cleanup deactivates only the unresolved Shader prims, not their parent
        Materials, so valid fallback contexts and composition bases remain usable.

        Args:
            stage: USD stage to inspect and edit.
            protected_material_paths: Material roots freshly authored by this task;
                only their live shader graph is protected from cleanup.
        """
        protected_material_paths = protected_material_paths or set()
        bound_material_paths = self._collect_bound_material_paths(stage)
        reachable_material_paths = set(protected_material_paths) | bound_material_paths
        frontier_material_paths = set(reachable_material_paths)
        material_path_upper_bound = {
            str(prim.GetPath())
            for prim in stage.TraverseAll()
            if prim.IsA(UsdShade.Material)
        }
        material_path_upper_bound.update(reachable_material_paths)
        max_reachability_iterations = max(1, len(material_path_upper_bound) + 1)
        reachability_iteration = 0
        while frontier_material_paths:
            reachability_iteration += 1
            if reachability_iteration > max_reachability_iterations:
                self.listener.warning(
                    "Stopping stale MDL material reachability traversal after "
                    f"{max_reachability_iterations} iterations; remaining frontier: "
                    f"{', '.join(sorted(frontier_material_paths))}"
                )
                break

            discovered_material_paths = set()
            discovered_material_paths.update(
                self._collect_composition_target_material_paths(
                    stage,
                    frontier_material_paths,
                    protected_material_paths=protected_material_paths,
                )
            )
            discovered_material_paths.update(
                self._collect_connected_material_paths(
                    stage,
                    frontier_material_paths,
                    protected_material_paths=protected_material_paths,
                )
            )
            new_material_paths = discovered_material_paths - reachable_material_paths
            reachable_material_paths.update(new_material_paths)
            frontier_material_paths = new_material_paths
        deactivated_paths: list[str] = []
        deactivated_path_set: set[str] = set()
        reachable_shader_paths = self._collect_reachable_shader_prim_paths(
            stage,
            reachable_material_paths,
            protected_material_paths,
        )

        def deactivate_shader_paths(shader_paths: list[str]) -> None:
            for shader_path in shader_paths:
                if shader_path in deactivated_path_set:
                    continue
                shader_prim = stage.GetPrimAtPath(shader_path)
                if shader_prim and shader_prim.IsValid() and shader_prim.IsActive():
                    shader_prim.SetActive(False)
                    deactivated_paths.append(shader_path)
                    deactivated_path_set.add(shader_path)

        # Instance proxy prims are read-only. Binding discovery includes them above,
        # but cleanup edits only real material prims in the composed stage.
        material_prims = [
            prim
            for prim in stage.TraverseAll()
            if prim.IsActive() and prim.IsA(UsdShade.Material)
        ]
        for prim in material_prims:
            material_path = str(prim.GetPath())
            if material_path in reachable_material_paths:
                if material_path not in protected_material_paths:
                    continue

                material_graph_prim_paths = self._collect_material_graph_prim_paths(
                    stage,
                    prim,
                )
                shader_paths = [
                    shader_path
                    for shader_path in self._collect_unresolved_mdl_shader_paths(
                        stage,
                        prim,
                    )
                    if shader_path not in material_graph_prim_paths
                ]
            else:
                shader_paths = []
                seen_shader_paths: set[str] = set()
                material_graph_prim_paths = self._collect_material_graph_prim_paths(
                    stage,
                    prim,
                )
                for shader_path in self._collect_unresolved_mdl_shader_paths(
                    stage, prim
                ) + self._collect_unresolved_mdl_shader_paths_from_prims(
                    stage,
                    material_graph_prim_paths,
                ):
                    if (
                        shader_path in seen_shader_paths
                        or shader_path in reachable_shader_paths
                    ):
                        continue
                    seen_shader_paths.add(shader_path)
                    shader_paths.append(shader_path)

            deactivate_shader_paths(shader_paths)

        all_shader_paths = {
            str(prim.GetPath())
            for prim in stage.TraverseAll()
            if prim.IsActive() and prim.IsA(UsdShade.Shader)
        }
        loose_shader_paths = [
            shader_path
            for shader_path in self._collect_unresolved_mdl_shader_paths_from_prims(
                stage,
                all_shader_paths,
            )
            if shader_path not in reachable_shader_paths
            and shader_path not in deactivated_path_set
        ]
        deactivate_shader_paths(loose_shader_paths)

        if deactivated_paths:
            self.listener.warning(
                "Deactivated stale unbound MDL shader prims with unresolved "
                f"source assets: {', '.join(deactivated_paths)}"
            )

        return deactivated_paths

    def _remap_asset_paths_in_prim(
        self,
        layer: Sdf.Layer,
        prim_path: Sdf.Path,
        source_dir: Path,
        target_dir: Path,
    ) -> None:
        """Remap all SdfAssetPath values in a prim and its descendants.

        Converts relative paths that were relative to source_dir to be
        relative to target_dir instead.

        Args:
            layer: The layer containing the prim specs
            prim_path: Path to the prim to process
            source_dir: Directory the original paths were relative to
            target_dir: Directory the new paths should be relative to
        """
        remap_asset_paths_in_prim(
            layer,
            prim_path,
            source_dir,
            target_dir,
            self.listener,
        )

    def _remap_single_asset_path(
        self,
        path_str: str,
        source_dir: Path,
        target_dir: Path,
    ) -> str:
        """Remap a single asset path from source_dir-relative to target_dir-relative.

        Args:
            path_str: The original path string
            source_dir: Directory the path was relative to
            target_dir: Directory the path should be relative to

        Returns:
            The remapped path string
        """
        return remap_single_asset_path(path_str, source_dir, target_dir, self.listener)

    def _create_full_stage(
        self,
        input_usd_path: Path,
        output_usd_path: Path,
        resolved_materials: dict,
        prim_to_material: dict,
        is_library_based: bool = False,
        material_library_path: str | None = None,
        flatten_output: bool = False,
        skip_instance_check: bool = False,
        material_profile: MaterialProfile = "auto",
        prediction_target_ids: set[str] | None = None,
    ) -> tuple[Usd.Stage, dict, dict]:
        """Create a complete new USD stage with materials applied.

        Args:
            input_usd_path: Path to input USD file
            output_usd_path: Path for output USD file
            resolved_materials: Dictionary of resolved material paths
            prim_to_material: Dictionary mapping prim paths to material names
            prediction_target_ids: Every prim ID represented in prediction evidence
            is_library_based: Whether materials are from a library (default: False)
            material_library_path: Path to material library USD file (optional)
            flatten_output: Whether to flatten the output stage (default: False)
                           When False, preserves references to material libraries
                           When True, creates a self-contained flattened USD

        Returns:
            Tuple of (stage, materials_applied, statistics)
        """
        self.listener.info(f"Creating full stage from {input_usd_path}")

        # Always start by creating a new stage that references the input
        # We'll flatten at the END if requested, after all materials are applied
        self.listener.info(
            "Creating new stage with reference to input"
            + (
                " (will flatten after materials applied)"
                if flatten_output
                else " (preserving composition)"
            )
        )

        # Open input stage to get up axis and default prim before creating output
        input_stage = Usd.Stage.Open(str(input_usd_path))
        if not input_stage:
            raise RuntimeError(f"Failed to open input USD: {input_usd_path}")
        original_up_axis = UsdGeom.GetStageUpAxis(input_stage)
        original_meters_per_unit = UsdGeom.GetStageMetersPerUnit(input_stage)
        self.listener.info(f"Original USD up axis: {original_up_axis}")
        self.listener.info(f"Original USD metersPerUnit: {original_meters_per_unit}")

        # Read defaultPrim from the input's root layer (non-composable metadata)
        input_default_prim = input_stage.GetRootLayer().defaultPrim
        self.listener.info(
            f"Original USD default prim: {input_default_prim or '(none)'}"
        )

        output_stage = Usd.Stage.CreateNew(str(output_usd_path))

        # Set stage metrics to match the input before adding sublayers
        UsdGeom.SetStageUpAxis(output_stage, original_up_axis)
        UsdGeom.SetStageMetersPerUnit(output_stage, original_meters_per_unit)
        self.listener.info(
            f"Set output USD up axis to: {original_up_axis}, "
            f"metersPerUnit to: {original_meters_per_unit}"
        )

        # Preserve defaultPrim from input (non-composable, must be on root layer)
        if input_default_prim:
            output_stage.GetRootLayer().defaultPrim = input_default_prim
            self.listener.info(f"Set output USD default prim to: {input_default_prim}")

        # Add the input USD as a sublayer to preserve all content
        output_stage.GetRootLayer().subLayerPaths.append(str(input_usd_path))

        # Save and reload to ensure composition is complete
        output_stage.Save()
        output_stage = Usd.Stage.Open(str(output_usd_path))

        # After composition, verify the default prim is valid.
        self._fix_stale_default_prim(output_stage, input_default_prim)

        # Apply materials to the new stage
        materials_applied = {}
        materials_created_count = 0
        prims_with_materials = 0
        bound_prim_ids: set[str] = set()

        if material_profile == "display_color":
            materials_applied = dict.fromkeys(resolved_materials, "display_color")
        elif is_library_based and material_library_path:
            # Library-based: Copy only used materials (not the entire library)
            self.listener.info(
                "Using library-based materials - copying used materials only"
            )
            output_stage, materials_applied = self._copy_library_materials(
                output_stage,
                material_library_path,
                output_usd_path,
                resolved_materials,
                default_prim_name=input_default_prim or "",
            )
            if material_profile == "preview_surface":
                updated = self._add_preview_surface_overlays_for_mdl_materials(
                    output_stage,
                    materials_applied,
                )
                if updated:
                    self.listener.warning(
                        "Authored PreviewSurface overlays for "
                        f"{updated} MDL material(s) because "
                        "material_profile='preview_surface' was requested."
                    )
                updated = add_ovrtx_preview_fallbacks_for_materialx_openpbr(
                    output_stage,
                    suppress_materialx_surface=True,
                    target_material_paths=materials_applied.values(),
                )
                if updated:
                    self.listener.warning(
                        "Authored PreviewSurface overlays for "
                        f"{updated} OpenPBR/MaterialX material(s) because "
                        "material_profile='preview_surface' was requested."
                    )
            materials_created_count = len(materials_applied)
        else:
            # File-based: Create materials for each resolved material using proven utility functions
            for material_name, material_path in resolved_materials.items():
                material_prim_path, success, _profile_metadata = (
                    self._create_material_on_stage(
                        stage=output_stage,
                        material_name=material_name,
                        material_path=material_path,
                        output_usd_path=output_usd_path,
                        path_prefix=None,  # Will use DefaultPrim/Looks
                        material_profile=material_profile,
                    )
                )

                if success:
                    materials_applied[material_name] = material_prim_path
                    materials_created_count += 1

        # Apply materials to prims based on predictions mapping
        instance_proxies_skipped = 0
        instance_root_to_ref_prim = self._get_local_instance_reference_map(output_stage)
        eligible_assignments: dict[str, str] = {}
        for prim_path, material_name in prim_to_material.items():
            if material_name not in materials_applied:
                self.listener.warning(
                    f"Material '{material_name}' not found in applied materials "
                    f"for prim {prim_path}"
                )
            else:
                eligible_assignments[prim_path] = material_name
        (
            grouped_assignments,
            remapped_instance_prims,
            skipped_instance_prims,
        ) = self._group_binding_assignments(
            eligible_assignments,
            instance_root_to_ref_prim,
        )
        for (
            binding_target_path,
            material_name,
            source_prim_paths,
            was_remapped,
        ) in grouped_assignments:
            source_description = ", ".join(source_prim_paths)

            # Get the prim from the stage
            prim = output_stage.GetPrimAtPath(binding_target_path)
            if not prim.IsValid():
                self.listener.warning(
                    f"Prim not found in stage: {binding_target_path}"
                    + (
                        f" (remapped from instance proxy {source_description})"
                        if was_remapped
                        else ""
                    )
                )
                continue

            # Instance proxies are READ-ONLY in USD - cannot author properties to them
            # Skip them here; they may be handled later via prototype material propagation
            if prim.IsInstanceProxy():
                instance_proxies_skipped += len(source_prim_paths)
                self.listener.debug(
                    f"Skipping instance proxy {source_description} - will inherit "
                    "material from prototype"
                )
                continue

            # Nullify existing material bindings and display colors
            try:
                nullify_material(prim)
            except Exception as e:
                self.listener.warning(
                    f"Failed to nullify material on prim {binding_target_path}: {e}"
                )

            # Clear material bindings on GeomSubset children so they don't override
            # the parent binding (USD uses weakerThanDescendants by default which
            # means descendant GeomSubset bindings would otherwise win).
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Subset):
                    UsdShade.MaterialBindingAPI(child).UnbindAllBindings()

            if material_profile == "display_color":
                self._author_display_color_on_prim(prim, material_name)
                prims_with_materials += len(source_prim_paths)
                bound_prim_ids.update(source_prim_paths)
                self.listener.info(
                    f"Authored displayColor for '{material_name}' on "
                    f"{binding_target_path}"
                    + (
                        f" (remapped from instance proxy {source_description})"
                        if was_remapped
                        else ""
                    )
                )
                continue

            # Find the corresponding material in our applied materials
            material_prim_path = materials_applied.get(material_name)
            if not material_prim_path:
                self.listener.warning(
                    f"Material '{material_name}' not found in applied materials "
                    f"for prim(s) {source_description}"
                )
                continue

            # Bind the new material directly on the output layer. This avoids
            # the native usdex binding helper, which can abort the process
            # before Python can catch an exception.
            try:
                self._author_material_binding_on_stage(
                    output_stage,
                    binding_target_path,
                    material_prim_path,
                )
                prims_with_materials += len(source_prim_paths)
                bound_prim_ids.update(source_prim_paths)
                self.listener.info(
                    f"Bound material '{material_name}' to prim {binding_target_path}"
                    + (
                        f" (remapped from instance proxy {source_description})"
                        if was_remapped
                        else ""
                    )
                )
            except Exception as e:
                self.listener.warning(
                    f"Failed to bind material '{material_name}' to prim "
                    f"{binding_target_path}: {e}"
                )

        if instance_proxies_skipped > 0:
            self.listener.info(
                f"Skipped {instance_proxies_skipped} instance proxy prims (read-only)"
            )
        if remapped_instance_prims:
            self.listener.info(
                f"Remapped {remapped_instance_prims} instance prim paths "
                f"to prototype paths"
            )
        if skipped_instance_prims:
            self.listener.info(
                f"Skipped {skipped_instance_prims} instance prims "
                f"(external-asset references cannot be overridden)"
            )

        # Apply materials to instances by looking up their master's material
        if material_profile == "display_color":
            instance_stats = {
                "instances_found": 0,
                "instances_applied": 0,
                "instances_skipped": 0,
            }
        elif skip_instance_check:
            self.listener.info("Skipping instance material check (payload mode)")
            instance_stats = {
                "instances_found": 0,
                "instances_applied": 0,
                "instances_skipped": 0,
            }
        else:
            instance_stats = self._apply_materials_to_instances(
                output_stage, prim_to_material, materials_applied
            )

        # If flatten_output is requested, flatten the entire composed stage now
        # This happens AFTER all materials and libraries are composed
        if flatten_output:
            try:
                self._deactivate_unbound_unresolved_mdl_shaders(
                    output_stage,
                    protected_material_paths=set(materials_applied.values()),
                )
            except Exception as e:
                self.listener.warning(
                    "Failed to deactivate stale unresolved MDL shaders before "
                    f"flattening; continuing with flatten: {e}"
                )
                self.listener.debug(traceback.format_exc())

            self.listener.info(
                "Flattening composed stage (resolving all sublayers and references)"
            )
            # Preserve stage metrics before flattening (Flatten() doesn't keep these)
            original_up_axis = UsdGeom.GetStageUpAxis(output_stage)
            original_meters_per_unit = UsdGeom.GetStageMetersPerUnit(output_stage)

            # Save the stage first to ensure all edits are written
            output_stage.Save()

            # Flatten the fully composed stage
            flattened_layer = output_stage.Flatten()

            try:
                flattened_stage = Usd.Stage.Open(flattened_layer)
                if flattened_stage:
                    self._deactivate_unbound_unresolved_mdl_shaders(
                        flattened_stage,
                        protected_material_paths=set(materials_applied.values()),
                    )
                    flattened_layer = flattened_stage.Flatten()
                else:
                    self.listener.warning(
                        "Failed to open flattened layer for stale MDL cleanup; "
                        "continuing with export"
                    )
            except Exception as e:
                self.listener.warning(
                    "Failed to deactivate stale unresolved MDL shaders after "
                    f"flattening; continuing with export: {e}"
                )
                self.listener.debug(traceback.format_exc())

            # Export the flattened layer, overwriting the output file
            flattened_layer.Export(str(output_usd_path))

            # Reload and restore stage metrics
            output_stage = Usd.Stage.Open(str(output_usd_path))
            UsdGeom.SetStageUpAxis(output_stage, original_up_axis)
            UsdGeom.SetStageMetersPerUnit(output_stage, original_meters_per_unit)
            output_stage.Save()
            self.listener.info(
                "✓ Stage flattened - output is now self-contained with no external references"
            )

        unbound_scope = (
            set(prediction_target_ids)
            if prediction_target_ids is not None
            else set(prim_to_material)
        )
        stats = {
            "materials_created": materials_created_count,
            "prims_with_materials": prims_with_materials
            + instance_stats["instances_applied"],
            "instances_applied": instance_stats["instances_applied"],
            "instances_skipped": instance_stats["instances_skipped"],
            "bound_prim_ids": sorted(bound_prim_ids),
            "unbound_prim_ids": sorted(unbound_scope - bound_prim_ids),
        }

        return output_stage, materials_applied, stats

    def _create_material_layer(
        self,
        input_usd_path: Path,
        output_usd_path: Path,
        resolved_materials: dict,
        prim_to_material: dict,
        is_library_based: bool = False,
        material_library_path: str | None = None,
        skip_instance_check: bool = False,
        material_profile: MaterialProfile = "auto",
        prediction_target_ids: set[str] | None = None,
    ) -> tuple[Usd.Stage, dict, dict]:
        """Create only a material binding layer that can be composed over the input.

        Args:
            input_usd_path: Path to input USD file
            output_usd_path: Path for output USD layer file
            resolved_materials: Dictionary of resolved material paths
            prim_to_material: Dictionary mapping prim paths to material names
            prediction_target_ids: Every prim ID represented in prediction evidence

        Returns:
            Tuple of (stage, materials_applied, statistics)
        """
        self.listener.info(f"Creating material binding layer for {input_usd_path}")

        # Open input stage to get up axis and default prim before creating output
        input_stage = Usd.Stage.Open(str(input_usd_path))
        if not input_stage:
            raise RuntimeError(f"Failed to open input USD: {input_usd_path}")
        original_up_axis = UsdGeom.GetStageUpAxis(input_stage)
        original_meters_per_unit = UsdGeom.GetStageMetersPerUnit(input_stage)
        self.listener.info(f"Original USD up axis: {original_up_axis}")
        self.listener.info(f"Original USD metersPerUnit: {original_meters_per_unit}")

        # Read defaultPrim from the input's root layer (non-composable metadata)
        input_default_prim = input_stage.GetRootLayer().defaultPrim
        self.listener.info(
            f"Original USD default prim: {input_default_prim or '(none)'}"
        )

        # Create a new stage for the material layer
        stage = Usd.Stage.CreateNew(str(output_usd_path))

        # Set stage metrics to match the input before adding sublayers
        UsdGeom.SetStageUpAxis(stage, original_up_axis)
        UsdGeom.SetStageMetersPerUnit(stage, original_meters_per_unit)
        self.listener.info(
            f"Set output USD up axis to: {original_up_axis}, "
            f"metersPerUnit to: {original_meters_per_unit}"
        )

        # Preserve defaultPrim from input (non-composable, must be on root layer)
        if input_default_prim:
            stage.GetRootLayer().defaultPrim = input_default_prim
            self.listener.info(f"Set output USD default prim to: {input_default_prim}")

        # Use sublayer to compose over the input USD file
        # This creates a non-destructive layer that can be composed over the original
        stage.GetRootLayer().subLayerPaths.append(str(input_usd_path))

        # After adding sublayer, verify the default prim is valid.
        self._fix_stale_default_prim(stage, input_default_prim)

        # Create materials and bindings in the overlay layer
        materials_applied = {}
        materials_created_count = 0
        prims_with_materials = 0
        bound_prim_ids: set[str] = set()

        if material_profile == "display_color":
            materials_applied = dict.fromkeys(resolved_materials, "display_color")
        elif is_library_based and material_library_path:
            # Library-based: Copy only used materials (not the entire library)
            self.listener.info(
                "Using library-based materials - copying used materials only"
            )
            stage, materials_applied = self._copy_library_materials(
                stage,
                material_library_path,
                output_usd_path,
                resolved_materials,
                default_prim_name=input_default_prim or "",
            )
            if material_profile == "preview_surface":
                updated = self._add_preview_surface_overlays_for_mdl_materials(
                    stage,
                    materials_applied,
                )
                if updated:
                    self.listener.warning(
                        "Authored PreviewSurface overlays for "
                        f"{updated} MDL material(s) because "
                        "material_profile='preview_surface' was requested."
                    )
                updated = add_ovrtx_preview_fallbacks_for_materialx_openpbr(
                    stage,
                    suppress_materialx_surface=True,
                    target_material_paths=materials_applied.values(),
                )
                if updated:
                    self.listener.warning(
                        "Authored PreviewSurface overlays for "
                        f"{updated} OpenPBR/MaterialX material(s) because "
                        "material_profile='preview_surface' was requested."
                    )
            materials_created_count = len(materials_applied)
        else:
            # File-based: Create materials using proven utility functions
            # For layer-only mode, we'll use "/Materials" as the path prefix
            for material_name, material_path in resolved_materials.items():
                material_prim_path, success, _profile_metadata = (
                    self._create_material_on_stage(
                        stage=stage,
                        material_name=material_name,
                        material_path=material_path,
                        output_usd_path=output_usd_path,
                        path_prefix="",  # Root level for layer mode
                        material_profile=material_profile,
                    )
                )

                if success:
                    materials_applied[material_name] = material_prim_path
                    materials_created_count += 1

        # Apply material bindings as "over" opinions in the layer based on
        # predictions.  We use the Sdf API directly because:
        #   - stage.OverridePrim() silently skips spec creation when the prim
        #     already exists via a sublayer.
        #   - Stage-level binding APIs refuse to author on instance proxies.
        #
        # Instance handling: we only bind to non-instance prims (prototypes /
        # reference sources).  USD instances inherit material bindings from
        # their prototype via composition — a stronger sublayer opinion on
        # the reference source overrides the referenced content's local
        # bindings, and instances see the override through the shared
        # prototype.  No de-instancing is needed.
        root_layer = stage.GetRootLayer()

        # Build instance root → local referenced prim path mapping.
        # For USD instances that reference a local prim (same-file, empty
        # assetPath), we write material overrides at the referenced prim paths
        # in our output layer.  The instance prototype inherits the stronger
        # sublayer opinion, so all instances sharing that prototype get the
        # material override.  Instances referencing external files (non-empty
        # assetPath) cannot be overridden this way and are skipped.
        instance_root_to_ref_prim = self._get_local_instance_reference_map(stage)
        eligible_assignments: dict[str, str] = {}
        for prim_path, material_name in prim_to_material.items():
            if material_name not in materials_applied:
                self.listener.warning(
                    f"Material '{material_name}' not found in applied materials for prim {prim_path}"
                )
            else:
                eligible_assignments[prim_path] = material_name

        # For prims under an instance root, group predictions by the referenced
        # prototype path where the binding opinion will actually be authored.
        (
            grouped_assignments,
            remapped_instance_prims,
            skipped_instance_prims,
        ) = self._group_binding_assignments(
            eligible_assignments,
            instance_root_to_ref_prim,
        )
        for (
            binding_target_path,
            material_name,
            source_prim_paths,
            was_remapped,
        ) in grouped_assignments:
            source_description = ", ".join(source_prim_paths)

            if material_profile == "display_color":
                self._author_display_color_on_prim_spec(
                    root_layer,
                    binding_target_path,
                    material_name,
                    stage=stage,
                )
                prims_with_materials += len(source_prim_paths)
                bound_prim_ids.update(source_prim_paths)
                self.listener.info(
                    f"Authored displayColor for '{material_name}' on "
                    f"{binding_target_path}"
                    + (
                        f" (remapped from instance proxy {source_description})"
                        if was_remapped
                        else ""
                    )
                )
                continue

            material_prim_path = materials_applied.get(material_name)
            if not material_prim_path:
                self.listener.warning(
                    f"Material '{material_name}' not found in applied materials "
                    f"for prim(s) {source_description}"
                )
                continue

            self._author_material_binding_in_layer(
                root_layer,
                binding_target_path,
                material_prim_path,
            )

            prims_with_materials += len(source_prim_paths)
            bound_prim_ids.update(source_prim_paths)
            self.listener.info(
                f"Bound material '{material_name}' to prim {binding_target_path}"
                + (
                    f" (remapped from instance proxy {source_description})"
                    if was_remapped
                    else ""
                )
            )

        if remapped_instance_prims:
            self.listener.info(
                f"Remapped {remapped_instance_prims} instance prim paths "
                f"to prototype paths"
            )
        if skipped_instance_prims:
            self.listener.info(
                f"Skipped {skipped_instance_prims} instance prims "
                f"(external-asset references cannot be overridden)"
            )

        # Apply materials to instances by looking up their master's material
        if material_profile == "display_color":
            instance_stats = {
                "instances_found": 0,
                "instances_applied": 0,
                "instances_skipped": 0,
            }
        elif skip_instance_check:
            self.listener.info("Skipping instance material check (payload mode)")
            instance_stats = {
                "instances_found": 0,
                "instances_applied": 0,
                "instances_skipped": 0,
            }
        else:
            instance_stats = self._apply_materials_to_instances(
                stage, prim_to_material, materials_applied
            )

        unbound_scope = (
            set(prediction_target_ids)
            if prediction_target_ids is not None
            else set(prim_to_material)
        )
        stats = {
            "materials_created": materials_created_count,
            "prims_with_materials": prims_with_materials
            + instance_stats["instances_applied"],
            "instances_applied": instance_stats["instances_applied"],
            "instances_skipped": instance_stats["instances_skipped"],
            "bound_prim_ids": sorted(bound_prim_ids),
            "unbound_prim_ids": sorted(unbound_scope - bound_prim_ids),
        }

        return stage, materials_applied, stats

    def _get_material_color(self, material_name: str) -> tuple[float, float, float]:
        """Get a default color based on material name.

        Args:
            material_name: Name of the material

        Returns:
            RGB color tuple
        """
        # Simple color mapping based on common material names
        color_map = {
            "metal": (0.7, 0.7, 0.8),
            "aluminum": (0.8, 0.8, 0.85),
            "steel": (0.6, 0.6, 0.65),
            "iron": (0.5, 0.5, 0.5),
            "plastic": (0.9, 0.9, 0.9),
            "rubber": (0.2, 0.2, 0.2),
            "glass": (0.95, 0.95, 1.0),
            "wood": (0.5, 0.3, 0.2),
            "concrete": (0.5, 0.5, 0.5),
            "brick": (0.6, 0.3, 0.2),
            "fabric": (0.7, 0.7, 0.6),
            "leather": (0.4, 0.2, 0.1),
        }

        # Check if any key is in the material name
        material_lower = material_name.lower()
        for key, color in color_map.items():
            if key in material_lower:
                return color

        # Default gray color
        return (0.5, 0.5, 0.5)
