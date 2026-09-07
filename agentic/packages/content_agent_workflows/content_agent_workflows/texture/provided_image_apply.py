# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministically apply outer-generated images to an exact Texture scope."""

from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from texture_agent.functions.material_discovery import (
    TEXTURE_SEMANTIC_MATERIAL_ALIASES_RELATIONSHIP,
)
from world_understanding.utils.usd.package import (
    extract_usdz_package_for_edit,
    parse_package_member_asset_path,
    safe_usdz_member_name,
)

from content_agent_workflows.asset_composition import (
    bind_usd_dependency_closure,
    verify_usd_dependency_closure,
)
from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256
from content_agent_workflows.common.usd_package_localizer import (
    create_localized_usdz_package,
)

from .capabilities import TextureGeneratorLeafRequest
from .models import TextureExecutionResult, TextureUnitArtifact
from .scope_validation import (
    _authorable_instance_path,
    validate_texture_scope_invariants,
)


class _ProvidedImageScopeUnit(BaseModel):
    """Typed exact scope fields carried by the compatibility plan envelope."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    material_prim_paths: tuple[str, ...] = Field(min_length=1)
    member_prim_paths: tuple[str, ...] = ()
    member_subset_paths: tuple[str, ...] = ()


class ProvidedImageTextureApplyLeaf:
    """Apply exact outer-provided albedo images without invoking a provider."""

    provider_id = "outer_provided_image_apply"
    capability_id = "texture.apply-provided.v1"
    _MAX_PACKAGED_IMAGE_BYTES = 512 * 1024 * 1024
    _PACKAGE_READ_CHUNK_BYTES = 1024 * 1024

    @staticmethod
    def _verify_regular_binding(binding: Any, *, label: str) -> Path:
        raw_path = Path(binding.path).expanduser()
        if raw_path.is_symlink():
            raise ValueError(f"{label} must not be a symlink: {raw_path}")
        path = raw_path.resolve()
        if not path.is_file():
            raise ValueError(f"{label} is not a regular file: {path}")
        if (
            path.stat().st_size != binding.size_bytes
            or file_sha256(path) != binding.sha256
        ):
            raise ValueError(f"{label} bytes changed: {path}")
        return path

    @classmethod
    def _verify_png(cls, binding: Any, *, texture_size: int) -> Path:
        path = cls._verify_regular_binding(binding, label="Texture provided image")
        try:
            with path.open("rb") as stream:
                with Image.open(stream) as image:
                    if image.format != "PNG":
                        raise ValueError("Texture provided image must be a PNG")
                    if image.size != (texture_size, texture_size):
                        raise ValueError(
                            "Texture provided image dimensions differ from the outer "
                            f"texture_size: {image.size} != {(texture_size, texture_size)}"
                        )
                    image.verify()
                stream.seek(0)
                with Image.open(stream) as image:
                    image.load()
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                f"Texture provided image is not a safe PNG: {path}"
            ) from exc
        return path

    @staticmethod
    def _copy_exact(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        replacement: Path | None = None
        if destination.is_symlink():
            raise ValueError(
                f"existing Texture package member must not be a symlink: {destination}"
            )
        if destination.exists():
            if not destination.is_file():
                raise ValueError(
                    "existing Texture package member must be a regular file: "
                    f"{destination}"
                )
            replacement = destination.with_name(f".{destination.name}.replacement")
        write_path = replacement or destination
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(write_path, flags, 0o600)
        try:
            with (
                source.open("rb") as source_stream,
                os.fdopen(descriptor, "wb") as destination_stream,
            ):
                descriptor = -1
                shutil.copyfileobj(source_stream, destination_stream)
            if file_sha256(source) != file_sha256(write_path):
                raise ValueError("copied Texture provided image bytes changed")
            if replacement is not None:
                os.replace(replacement, destination)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if replacement is not None and replacement.exists():
                replacement.unlink()
        if file_sha256(source) != file_sha256(destination):
            raise ValueError("copied Texture provided image bytes changed")

    @staticmethod
    def _author_material(
        stage: Any,
        *,
        material_path: str,
        texture_asset_path: str,
        name_suffix: str,
    ) -> str:
        from pxr import Sdf, UsdShade

        material_prim = stage.GetPrimAtPath(material_path)
        if not material_prim or not material_prim.IsA(UsdShade.Material):
            raise ValueError(
                f"Texture prepared material is unavailable in the source: {material_path}"
            )
        material = UsdShade.Material(material_prim)
        for output in material.GetOutputs():
            if output.GetBaseName().split(":")[-1] == "surface":
                output.DisconnectSource()
        primvar = UsdShade.Shader.Define(
            stage,
            f"{material_path}/OuterProvidedPrimvar_{name_suffix}",
        )
        primvar.CreateIdAttr("UsdPrimvarReader_float2")
        primvar.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        primvar_output = primvar.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        texture_path = f"{material_path}/OuterProvidedAlbedo_{name_suffix}"
        texture = UsdShade.Shader.Define(
            stage,
            texture_path,
        )
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_asset_path)
        )
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            primvar_output
        )
        texture_output = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        surface = UsdShade.Shader.Define(
            stage,
            f"{material_path}/OuterProvidedSurface_{name_suffix}",
        )
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture_output
        )
        surface_output = surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(surface_output)
        return texture_path

    @staticmethod
    def _clone_material(stage: Any, *, source_path: str, clone_path: str) -> None:
        """Clone one flattened material before exact-member rebinding."""

        from pxr import Sdf, UsdShade

        source = stage.GetPrimAtPath(source_path)
        if not source or not source.IsA(UsdShade.Material):
            raise ValueError(
                f"Texture prepared material is unavailable in the source: {source_path}"
            )
        if stage.GetPrimAtPath(clone_path).IsValid():
            raise ValueError(
                f"provided-image material clone already exists: {clone_path}"
            )
        if not Sdf.CopySpec(
            stage.GetRootLayer(),
            source_path,
            stage.GetRootLayer(),
            clone_path,
        ):
            raise RuntimeError(
                f"could not clone provided-image material: {source_path}"
            )

    @staticmethod
    def _author_semantic_material_aliases(
        stage: Any,
        *,
        source_path: str,
        material_path: str,
    ) -> None:
        """Carry stable source aliases onto a replacement bound material."""

        from pxr import Sdf, UsdShade

        source = stage.GetPrimAtPath(source_path)
        material = stage.GetPrimAtPath(material_path)
        if not source or not source.IsA(UsdShade.Material):
            raise ValueError(
                f"Texture semantic source material is unavailable: {source_path}"
            )
        if not material or not material.IsA(UsdShade.Material):
            raise ValueError(
                f"Texture semantic replacement material is unavailable: {material_path}"
            )
        aliases = {Sdf.Path(source_path)}
        inherited = source.GetRelationship(
            TEXTURE_SEMANTIC_MATERIAL_ALIASES_RELATIONSHIP
        )
        if inherited and inherited.IsValid():
            aliases.update(inherited.GetTargets())
        for alias in aliases:
            alias_prim = stage.GetPrimAtPath(alias)
            if (
                not alias.IsAbsolutePath()
                or not alias.IsPrimPath()
                or not alias_prim
                or not alias_prim.IsA(UsdShade.Material)
            ):
                raise ValueError(
                    "Texture semantic material alias is not an existing material: "
                    f"{alias}"
                )
        material.CreateRelationship(
            TEXTURE_SEMANTIC_MATERIAL_ALIASES_RELATIONSHIP,
            custom=True,
        ).SetTargets(sorted(aliases, key=str))

    @staticmethod
    def _bound_material_path(stage: Any, member_path: str) -> str:
        """Resolve one exact member's effective material before authoring."""

        from pxr import UsdShade

        member = stage.GetPrimAtPath(member_path)
        if not member or not member.IsValid():
            raise ValueError(
                f"provided-image Texture member is unavailable: {member_path}"
            )
        if member.IsInstanceProxy():
            raise ValueError(
                "provided-image Texture member is a read-only instance proxy: "
                f"{member_path}"
            )
        material, _relationship = UsdShade.MaterialBindingAPI(
            member
        ).ComputeBoundMaterial()
        if not material:
            raise ValueError(
                f"provided-image Texture member has no effective material: {member_path}"
            )
        return str(material.GetPath())

    @classmethod
    def _verify_packaged_image_member(
        cls,
        *,
        candidate: Path,
        asset_path: Any,
        supplied_binding: Any,
        shader_path: str,
    ) -> dict[str, Any]:
        """Bind one authored shader asset to exact bytes in ``candidate``."""

        from pxr import Sdf

        if not isinstance(asset_path, Sdf.AssetPath) or not asset_path.path:
            raise ValueError(
                f"provided-image shader has no authored Sdf asset: {shader_path}"
            )
        authored_member = safe_usdz_member_name(asset_path.path)
        if authored_member is None or authored_member != asset_path.path:
            raise ValueError(
                "provided-image shader has a noncanonical package member path: "
                f"{shader_path}"
            )
        if not asset_path.resolvedPath:
            raise ValueError(
                f"provided-image shader asset did not resolve: {shader_path}"
            )
        parsed = parse_package_member_asset_path(asset_path.resolvedPath)
        if parsed is None:
            raise ValueError(
                "provided-image shader asset did not resolve to a package member: "
                f"{shader_path}"
            )
        resolved_package, resolved_member = parsed
        try:
            expected_package = candidate.resolve(strict=True)
            observed_package = resolved_package.resolve(strict=True)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"provided-image package is unavailable for {shader_path}"
            ) from exc
        if observed_package != expected_package:
            raise ValueError(
                "provided-image shader resolved outside its exact candidate package: "
                f"{shader_path}"
            )
        if resolved_member != authored_member:
            raise ValueError(
                "provided-image shader resolved to a substituted package member: "
                f"{shader_path}"
            )
        expected_size = supplied_binding.size_bytes
        if expected_size > cls._MAX_PACKAGED_IMAGE_BYTES:
            raise ValueError(
                f"provided-image package member exceeds the read bound: {shader_path}"
            )

        digest = hashlib.sha256()
        bytes_read = 0
        try:
            with zipfile.ZipFile(expected_package) as archive:
                try:
                    info = archive.getinfo(resolved_member)
                except KeyError as exc:
                    raise ValueError(
                        f"provided-image package member is missing: {resolved_member}"
                    ) from exc
                file_type = (info.external_attr >> 16) & 0o170000
                if info.is_dir() or file_type == 0o120000:
                    raise ValueError(
                        "provided-image package member is not a regular file: "
                        f"{resolved_member}"
                    )
                if info.flag_bits & 0x1:
                    raise ValueError(
                        "provided-image package member must not be encrypted: "
                        f"{resolved_member}"
                    )
                if info.file_size != expected_size:
                    raise ValueError(
                        "provided-image package member size differs from its supplied "
                        f"binding: {resolved_member}"
                    )
                with archive.open(info) as stream:
                    while True:
                        chunk = stream.read(cls._PACKAGE_READ_CHUNK_BYTES)
                        if not chunk:
                            break
                        bytes_read += len(chunk)
                        if (
                            bytes_read > expected_size
                            or bytes_read > cls._MAX_PACKAGED_IMAGE_BYTES
                        ):
                            raise ValueError(
                                "provided-image package member exceeded its bounded "
                                f"binding: {resolved_member}"
                            )
                        digest.update(chunk)
        except zipfile.BadZipFile as exc:
            raise ValueError(
                f"provided-image candidate package is unreadable: {candidate}"
            ) from exc
        if bytes_read != expected_size or digest.hexdigest() != supplied_binding.sha256:
            raise ValueError(
                "provided-image package member bytes differ from its supplied binding: "
                f"{resolved_member}"
            )
        return {
            "package_member_path": resolved_member,
            "package_member_sha256": digest.hexdigest(),
            "package_member_size_bytes": bytes_read,
        }

    @classmethod
    def _verify_packaged_texture_readbacks(
        cls,
        *,
        candidate: Path,
        authored_shader_paths_by_unit: dict[str, tuple[str, ...]],
        provided_by_unit: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Reopen the saved USDZ and verify every newly authored image asset."""

        from pxr import Sdf, Usd, UsdShade

        candidate_stage = Usd.Stage.Open(str(candidate))
        if candidate_stage is None:
            raise RuntimeError(
                f"Could not reopen packaged provided-image candidate: {candidate}"
            )
        readbacks: dict[str, dict[str, Any]] = {}
        for unit_id, shader_paths in authored_shader_paths_by_unit.items():
            provided = provided_by_unit[unit_id]
            unit_binding: dict[str, Any] | None = None
            for shader_path in shader_paths:
                shader_prim = candidate_stage.GetPrimAtPath(shader_path)
                if not shader_prim or not shader_prim.IsA(UsdShade.Shader):
                    raise ValueError(
                        "provided-image shader is missing from packaged candidate: "
                        f"{shader_path}"
                    )
                file_input = UsdShade.Shader(shader_prim).GetInput("file")
                asset_path = file_input.Get() if file_input else None
                if not isinstance(asset_path, Sdf.AssetPath):
                    raise ValueError(
                        f"provided-image shader file is not an Sdf asset: {shader_path}"
                    )
                observed = cls._verify_packaged_image_member(
                    candidate=candidate,
                    asset_path=asset_path,
                    supplied_binding=provided.artifact,
                    shader_path=shader_path,
                )
                if unit_binding is not None and observed != unit_binding:
                    raise ValueError(
                        "provided-image shaders for one unit resolved to different "
                        f"package members: {unit_id}"
                    )
                unit_binding = observed
            if unit_binding is None:
                raise ValueError(
                    f"provided-image Texture unit has no authored shader: {unit_id}"
                )
            readbacks[unit_id] = {
                **unit_binding,
                "authored_shader_paths": list(shader_paths),
            }
        return readbacks

    def generate(self, request: TextureGeneratorLeafRequest) -> TextureExecutionResult:
        """Create one byte-self-contained USDZ from supplied candidate images."""

        from pxr import Sdf, Usd, UsdShade

        if tuple(unit.unit_id for unit in request.scope_plan.selected_units) != (
            request.target_unit_ids
        ):
            raise ValueError("provided-image apply scope differs from preparation")
        if len(request.generator_inputs) != len(request.target_unit_ids):
            raise ValueError("provided-image apply inputs differ from target order")
        self._verify_regular_binding(request.source, label="Texture source")
        verify_usd_dependency_closure(
            request.source.path,
            request.source_dependencies,
        )
        for reference in request.reference_artifacts:
            self._verify_regular_binding(
                reference.artifact,
                label=f"Texture reference {reference.role}",
            )

        output_dir = Path(request.output_dir).expanduser().resolve()
        if output_dir.exists():
            raise FileExistsError(
                f"provided-image Texture output already exists: {output_dir}"
            )
        output_dir.mkdir(parents=True, mode=0o700)
        workspace = output_dir / "provided_apply"
        workspace.mkdir(mode=0o700)

        source_path = Path(request.source.path).expanduser().resolve()
        package_workspace = workspace
        editable_source = source_path
        if source_path.suffix.lower() == ".usdz":
            # Re-applying to a prior deterministic candidate must retain its
            # texture members as ordinary dependencies of the new root layer.
            # Flattening a stage directly from a package causes USD's package
            # builder to embed that whole package as a nested USDZ; the authored
            # asset paths then change and exact non-target material validation
            # correctly rejects the result.  Extract the already digest-bound
            # package into this private workspace first so both existing and new
            # textures are repackaged as direct, content-equivalent members.
            package_workspace = workspace / "source_package"
            editable_source = extract_usdz_package_for_edit(
                source_path,
                package_workspace,
            )

        scope_by_id = {
            unit.unit_id: _ProvidedImageScopeUnit.model_validate(
                unit.model_dump(mode="json")
            )
            for unit in request.scope_plan.selected_units
        }
        copied_by_unit: dict[str, Path] = {}
        original_by_unit: dict[str, Path] = {}
        provided_by_unit: dict[str, Any] = {}
        for unit_id, inputs in zip(
            request.target_unit_ids,
            request.generator_inputs,
            strict=True,
        ):
            if inputs.execution_mode != "apply_provided":
                raise ValueError("apply-provided leaf requires apply_provided mode")
            if inputs.backend != self.provider_id:
                raise ValueError(
                    "apply-provided backend differs from the selected leaf"
                )
            if (
                inputs.engine is not None
                or inputs.seed is not None
                or inputs.parameters
            ):
                raise ValueError(
                    "apply-provided leaf rejects generator engine, seed, and parameters"
                )
            if len(inputs.provided_images) != 1:
                raise ValueError(
                    "apply-provided leaf requires exactly one image per unit"
                )
            provided = inputs.provided_images[0]
            if provided.unit_id != unit_id or provided.channel != "albedo":
                raise ValueError(
                    "provided image channel or unit differs from prepared scope"
                )
            original = self._verify_png(
                provided.artifact,
                texture_size=inputs.texture_size,
            )
            copied = package_workspace / "textures" / unit_id / "albedo.png"
            self._copy_exact(original, copied)
            original_by_unit[unit_id] = original
            copied_by_unit[unit_id] = copied
            provided_by_unit[unit_id] = provided

        source_stage = Usd.Stage.Open(str(editable_source))
        if source_stage is None:
            raise RuntimeError(f"Could not open Texture source: {editable_source}")
        flattened = source_stage.Flatten()
        scope_source_layer = package_workspace / "provided_apply_source.usda"
        root_layer = package_workspace / "provided_apply_root.usda"
        if not flattened.Export(str(root_layer)):
            raise RuntimeError(
                "Could not flatten Texture source for deterministic apply"
            )
        candidate_stage = Usd.Stage.Open(str(root_layer))
        if candidate_stage is None:
            raise RuntimeError("Could not reopen deterministic Texture apply stage")
        source_documentation = source_stage.GetPseudoRoot().GetMetadata("documentation")
        candidate_root = candidate_stage.GetPseudoRoot()
        candidate_root.ClearMetadata("documentation")
        if source_documentation is not None:
            candidate_root.SetMetadata("documentation", source_documentation)
        if not candidate_stage.GetRootLayer().Export(str(scope_source_layer)):
            raise RuntimeError(
                "Could not freeze normalized Texture source scope for validation"
            )

        claimed_material_paths: set[str] = set()
        claimed_authorable_material_paths: set[str] = set()
        claimed_member_paths: set[str] = set()
        claimed_clone_paths: set[str] = set()
        uses_authorable_instance_mapping = False
        material_targets_by_unit: dict[
            str,
            tuple[tuple[str, str, str, tuple[str, ...]], ...],
        ] = {}
        for unit_id in request.target_unit_ids:
            unit = scope_by_id[unit_id]
            source_material_paths = tuple(
                dict.fromkeys(str(path) for path in unit.material_prim_paths)
            )
            overlap = claimed_material_paths.intersection(source_material_paths)
            if overlap:
                raise ValueError(
                    "provided-image Texture units overlap material path: "
                    f"{sorted(overlap)[0]}"
                )
            claimed_material_paths.update(source_material_paths)
            source_material_by_authorable_path: dict[str, str] = {}
            for source_material_path in source_material_paths:
                authorable_material_path = _authorable_instance_path(
                    candidate_stage,
                    source_material_path,
                )
                uses_authorable_instance_mapping = (
                    uses_authorable_instance_mapping
                    or authorable_material_path != source_material_path
                )
                prior_source_path = source_material_by_authorable_path.setdefault(
                    authorable_material_path,
                    source_material_path,
                )
                if prior_source_path != source_material_path:
                    raise ValueError(
                        "provided-image Texture material aliases collapse onto one "
                        f"authorable material: {authorable_material_path}"
                    )
            authorable_overlap = claimed_authorable_material_paths.intersection(
                source_material_by_authorable_path
            )
            if authorable_overlap:
                raise ValueError(
                    "provided-image Texture units overlap authorable material path: "
                    f"{sorted(authorable_overlap)[0]}"
                )
            claimed_authorable_material_paths.update(source_material_by_authorable_path)
            member_paths = tuple(
                dict.fromkeys((*unit.member_prim_paths, *unit.member_subset_paths))
            )
            member_overlap = claimed_member_paths.intersection(member_paths)
            if member_overlap:
                raise ValueError(
                    "provided-image Texture units overlap member path: "
                    f"{sorted(member_overlap)[0]}"
                )
            claimed_member_paths.update(member_paths)

            if not member_paths:
                material_targets_by_unit[unit_id] = tuple(
                    (
                        authorable_path,
                        source_path,
                        authorable_path,
                        (),
                    )
                    for authorable_path, source_path in (
                        source_material_by_authorable_path.items()
                    )
                )
                continue

            members_by_material: dict[str, list[str]] = {
                path: [] for path in source_material_by_authorable_path
            }
            for member_path in member_paths:
                authorable_member_path = _authorable_instance_path(
                    candidate_stage,
                    member_path,
                )
                bound_material_path = self._bound_material_path(
                    candidate_stage,
                    authorable_member_path,
                )
                if bound_material_path not in members_by_material:
                    raise ValueError(
                        "provided-image Texture member is bound outside its exact "
                        f"material scope: {member_path}"
                    )
                members_by_material[bound_material_path].append(authorable_member_path)
            missing_members = sorted(
                path for path, members in members_by_material.items() if not members
            )
            if missing_members:
                raise ValueError(
                    "provided-image Texture material has no exact member target: "
                    f"{missing_members[0]}"
                )
            targets: list[tuple[str, str, str, tuple[str, ...]]] = []
            for material_source_path, members in members_by_material.items():
                semantic_source_path = source_material_by_authorable_path[
                    material_source_path
                ]
                clone_path = str(
                    Sdf.Path(material_source_path).GetParentPath().AppendChild(unit_id)
                )
                if (
                    clone_path in claimed_clone_paths
                    or candidate_stage.GetPrimAtPath(clone_path).IsValid()
                ):
                    raise ValueError(
                        f"provided-image material clone path overlaps: {clone_path}"
                    )
                claimed_clone_paths.add(clone_path)
                targets.append(
                    (
                        material_source_path,
                        semantic_source_path,
                        clone_path,
                        tuple(members),
                    )
                )
            material_targets_by_unit[unit_id] = tuple(targets)

        authored_shader_paths_by_unit: dict[str, tuple[str, ...]] = {}
        for unit_id in request.target_unit_ids:
            provided = provided_by_unit[unit_id]
            relative_texture = (
                copied_by_unit[unit_id].relative_to(package_workspace).as_posix()
            )
            unit_shader_paths: list[str] = []
            for (
                material_source_path,
                _semantic_source_path,
                material_path,
                member_paths,
            ) in material_targets_by_unit[unit_id]:
                if material_path != material_source_path:
                    self._clone_material(
                        candidate_stage,
                        source_path=material_source_path,
                        clone_path=material_path,
                    )
                unit_shader_paths.append(
                    self._author_material(
                        candidate_stage,
                        material_path=material_path,
                        texture_asset_path=relative_texture,
                        name_suffix=provided.artifact.sha256[:12],
                    )
                )
                if member_paths:
                    material = UsdShade.Material(
                        candidate_stage.GetPrimAtPath(material_path)
                    )
                    for member_path in member_paths:
                        member = candidate_stage.GetPrimAtPath(member_path)
                        if (
                            not member
                            or not member.IsValid()
                            or member.IsInstanceProxy()
                        ):
                            raise ValueError(
                                "provided-image Texture member is not writable before "
                                f"authoring: {member_path}"
                            )
                        UsdShade.MaterialBindingAPI.Apply(member).Bind(material)
            authored_shader_paths_by_unit[unit_id] = tuple(unit_shader_paths)
        from .scene_validation import _source_bound_surfaces

        bound_material_paths = set(_source_bound_surfaces(candidate_stage).values())
        for unit_id in request.target_unit_ids:
            for (
                source_path,
                semantic_source_path,
                material_path,
                _member_paths,
            ) in material_targets_by_unit[unit_id]:
                if (
                    material_path != source_path
                    and source_path not in bound_material_paths
                ):
                    self._author_semantic_material_aliases(
                        candidate_stage,
                        source_path=semantic_source_path,
                        material_path=material_path,
                    )
        candidate_stage.GetRootLayer().Save()

        candidate = output_dir / "provided-texture-candidate.usdz"
        create_localized_usdz_package(
            root_layer,
            candidate,
            root_layer.name,
        )
        if bind_usd_dependency_closure(candidate):
            raise RuntimeError("provided-image Texture candidate is not self-contained")
        packaged_readbacks = self._verify_packaged_texture_readbacks(
            candidate=candidate,
            authored_shader_paths_by_unit=authored_shader_paths_by_unit,
            provided_by_unit=provided_by_unit,
        )
        scope_report = validate_texture_scope_invariants(
            source_asset_path=request.source.path,
            output_asset_path=candidate,
            plan=request.scope_plan,
            # A unique instance is authored in USD's flattened source
            # namespace. Keep the original path as the dependency-resolution
            # anchor while comparing composed state in that exact namespace.
            normalized_source_asset_path=(
                scope_source_layer if uses_authorable_instance_mapping else None
            ),
        )
        if not scope_report.passed:
            details = "; ".join(
                f"{item.code} at {item.prim_path}: {item.summary}"
                for item in scope_report.violations
            )
            raise ValueError(
                f"provided-image apply changed content outside Texture scope: {details}"
            )

        candidate_sha256 = file_sha256(candidate)
        unit_artifacts: list[TextureUnitArtifact] = []
        for unit_id in request.target_unit_ids:
            provided = provided_by_unit[unit_id]
            packaged_readback = packaged_readbacks[unit_id]
            receipt = atomic_write_json(
                output_dir / f"{unit_id}_provided_apply_receipt.json",
                {
                    "schema_version": "content-agent-workflows.texture-provided-apply-unit.v1",
                    "unit_id": unit_id,
                    "channel": provided.channel,
                    "role": provided.role,
                    "provided_image": provided.artifact.model_dump(mode="json"),
                    "producer": provided.producer.model_dump(mode="json"),
                    "source": request.source.model_dump(mode="json"),
                    "source_dependencies": [
                        item.model_dump(mode="json")
                        for item in request.source_dependencies
                    ],
                    "material_prim_paths": list(
                        scope_by_id[unit_id].material_prim_paths
                    ),
                    "member_prim_paths": list(scope_by_id[unit_id].member_prim_paths),
                    "member_subset_paths": list(
                        scope_by_id[unit_id].member_subset_paths
                    ),
                    "candidate_path": str(candidate),
                    "candidate_sha256": candidate_sha256,
                    **packaged_readback,
                    "provider_invoked": False,
                    "model_invoked": False,
                },
            )
            unit_artifacts.append(
                TextureUnitArtifact(
                    unit_id=unit_id,
                    artifact_paths=(str(original_by_unit[unit_id]), str(receipt)),
                    metadata={
                        "backend": self.provider_id,
                        "capability": self.capability_id,
                        "execution_mode": "apply_provided",
                        "provided_image_sha256": provided.artifact.sha256,
                        **packaged_readback,
                        "output_asset_path": str(candidate),
                    },
                )
            )

        # Close the mutation window by rebinding every external input once more.
        self._verify_regular_binding(request.source, label="Texture source")
        verify_usd_dependency_closure(
            request.source.path,
            request.source_dependencies,
        )
        for reference in request.reference_artifacts:
            self._verify_regular_binding(
                reference.artifact,
                label=f"Texture reference {reference.role}",
            )
        for provided in provided_by_unit.values():
            self._verify_regular_binding(
                provided.artifact,
                label=f"Texture provided image {provided.unit_id}",
            )

        return TextureExecutionResult(
            requested_unit_ids=request.target_unit_ids,
            unit_artifacts=tuple(unit_artifacts),
            output_asset_path=str(candidate),
            metadata={
                "backend": self.provider_id,
                "capability": self.capability_id,
                "execution_mode": "apply_provided",
                "provider_invoked": False,
                "model_invoked": False,
                "live_backend_invoked": False,
                "candidate_sha256": candidate_sha256,
            },
        )


__all__ = ["ProvidedImageTextureApplyLeaf"]
