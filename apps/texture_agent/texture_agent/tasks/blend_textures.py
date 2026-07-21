# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Blend generated textures onto material base values."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from world_understanding.agentic.tasks import Task

from texture_agent.functions.material_discovery import PrimTextureUnit
from texture_agent.functions.texture_blending import blend_texture_onto_constant
from texture_agent.functions.texture_generation import GeneratedTextures
from texture_agent.tasks.thresholds import validate_failure_threshold

logger = logging.getLogger(__name__)


@dataclass
class BlendedTextures:
    """Paths to blended PBR texture files for a material."""

    albedo: str
    normal: str
    orm: str


class BlendTexturesTask(Task):
    """Composite generated PBR textures onto material constant values.

    For albedo: blends generated texture onto the material's constant
    base_color at the configured opacity.

    For normal: uses the generated normal map directly (no blending --
    normal maps are additive, not multiplicative with a base value).

    For ORM: blends each channel independently onto the material's
    constant roughness/metalness values at the configured opacity.

    Context keys read:
        prim_texture_units (list[PrimTextureUnit]): From DiscoverMaterialsTask.
        generated_textures (dict[str, GeneratedTextures]): From GenerateTexturesTask.
        blend_config (dict): Default opacity, output size.
        working_dir (str): Working directory.

    Context keys written:
        blended_textures (dict[str, BlendedTextures]):
            Unit key -> BlendedTextures (albedo, normal, orm paths).
    """

    def __init__(self) -> None:
        self.name = "BlendTextures"
        self.description = "Blend PBR textures onto material base values"

    def _blend_orm(
        self,
        orm_path: str,
        roughness: float,
        metalness: float,
        output_size: tuple[int, int],
        opacity: float,
    ) -> Image.Image:
        """Blend ORM texture channels onto material constant values.

        ORM packing: R=Occlusion, G=Roughness, B=Metallic.

        - Occlusion: generated value used directly (no base constant)
        - Roughness: blended with material's specular_roughness
        - Metalness: blended with material's base_metalness
        """
        with Image.open(orm_path) as orm_source:
            orm_img = orm_source.resize(output_size, Image.Resampling.LANCZOS)
            orm_arr = np.array(orm_img, dtype=np.float32)

        # Create base ORM from material constants
        # NumPy uses (height, width, channels); PIL size is (width, height)
        h, w = output_size[1], output_size[0]
        base_arr = np.zeros((h, w, 3), dtype=np.float32)
        base_arr[:, :, 0] = 255.0  # Occlusion = 1.0 (no occlusion)
        base_arr[:, :, 1] = roughness * 255.0
        base_arr[:, :, 2] = metalness * 255.0

        # Blend: result = base * (1 - opacity) + generated * opacity
        result = base_arr * (1.0 - opacity) + orm_arr[:, :, :3] * opacity
        return Image.fromarray(result.clip(0, 255).astype(np.uint8))

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        units: list[PrimTextureUnit] = context.get("prim_texture_units", [])
        generated: dict[str, GeneratedTextures] = context.get("generated_textures", {})
        blend_config: dict = context.get("blend_config", {})
        working_dir = Path(context["working_dir"])

        # Validate threshold before any per-unit work so a typo fails fast.
        failure_threshold = validate_failure_threshold(
            blend_config.get("failure_threshold", 1.0),
            config_key="blend_config.failure_threshold",
        )

        output_size_val = blend_config.get("output_size", 1024)
        output_size = (output_size_val, output_size_val)

        out_dir = working_dir / "textures"
        out_dir.mkdir(parents=True, exist_ok=True)

        unit_by_key = {u.key: u for u in units}
        blended: dict[str, BlendedTextures] = {}
        errors: list[dict[str, Any]] = []

        # Wire context to the live ``errors`` and ``blended`` containers
        # up front so that if a hard exception propagates mid-loop, the
        # executor's per-step except block (which calls ``_extract_step_
        # stats`` from context) still sees any soft errors recorded
        # before the exception.
        context["blended_textures"] = blended
        context["blend_textures_errors"] = errors
        context["blend_textures_attempted_count"] = len(generated)

        for key, gen_textures in generated.items():
            unit = unit_by_key.get(key)
            if not unit:
                logger.warning("Generated texture for unknown key: %s", key)
                errors.append(
                    {
                        "material": key,
                        "type": "UnknownUnit",
                        "status": None,
                        "message": "No prim_texture_unit matches this key",
                    }
                )
                continue

            mat = unit.material_info
            opacity = unit.opacity

            # --- Albedo: blend onto base_color ---
            if not Path(gen_textures.albedo).exists():
                logger.warning("Albedo texture missing for %s, skipping", key)
                errors.append(
                    {
                        "material": key,
                        "type": "MissingAlbedo",
                        "status": None,
                        "message": (
                            f"Generated albedo path does not exist on disk: "
                            f"{gen_textures.albedo!r}"
                        ),
                    }
                )
                continue
            # Hard exceptions (Image.open on corrupt PNG, save OSError, blend
            # math) propagate -- pre-MR behavior. Soft per-unit cases above
            # (missing prim_unit, missing albedo file) are the only paths
            # threshold-gated; surfacing those preserves the original
            # warning+continue while letting the executor's per-step except
            # block surface a hard exception with a structured failed-step
            # stats payload (the `except` block extracts step stats from
            # context, so any soft errors recorded so far are still surfaced
            # alongside the propagated exception).
            with Image.open(gen_textures.albedo) as albedo_img:
                blended_albedo = blend_texture_onto_constant(
                    base_color=mat.base_color,
                    texture=albedo_img,
                    output_size=output_size,
                    opacity=opacity,
                )
            albedo_path = out_dir / f"{key}_albedo.png"
            blended_albedo.save(str(albedo_path))

            # --- Normal: use directly (no blending) ---
            normal_path = out_dir / f"{key}_normal.png"
            if gen_textures.normal and Path(gen_textures.normal).exists():
                with Image.open(gen_textures.normal) as normal_source:
                    normal_img = normal_source.resize(
                        output_size, Image.Resampling.LANCZOS
                    )
                normal_img.save(str(normal_path))
            else:
                Image.new("RGB", output_size, (128, 128, 255)).save(str(normal_path))

            # --- ORM: blend roughness/metalness channels ---
            orm_path = out_dir / f"{key}_orm.png"
            roughness = (
                mat.specular_roughness if mat.specular_roughness is not None else 0.5
            )
            metalness = mat.base_metalness if mat.base_metalness is not None else 0.0
            if gen_textures.orm and Path(gen_textures.orm).exists():
                blended_orm = self._blend_orm(
                    gen_textures.orm,
                    roughness=roughness,
                    metalness=metalness,
                    output_size=output_size,
                    opacity=opacity,
                )
                blended_orm.save(str(orm_path))
            else:
                h, w = output_size[1], output_size[0]
                orm_arr = np.zeros((h, w, 3), dtype=np.uint8)
                orm_arr[:, :, 0] = 255
                orm_arr[:, :, 1] = int(roughness * 255)
                orm_arr[:, :, 2] = int(metalness * 255)
                Image.fromarray(orm_arr).save(str(orm_path))

            blended[key] = BlendedTextures(
                albedo=str(albedo_path),
                normal=str(normal_path),
                orm=str(orm_path),
            )

            logger.info(
                "Blended %s: albedo + normal + orm (opacity=%.2f) -> %s",
                key,
                opacity,
                out_dir,
            )

        # ``blended_textures``, ``blend_textures_errors``, and
        # ``blend_textures_attempted_count`` are already on context (wired
        # to the live containers up front). Only the failed-count snapshot
        # lands here so executor sees the same shape on success and on a
        # mid-loop hard-exception propagation.
        context["blend_textures_failed_count"] = len(errors)
        logger.info(
            "Blended %d PBR texture sets (%d failures across %d attempts)",
            len(blended),
            len(errors),
            len(generated),
        )

        # Match generate_textures' threshold semantics (default 1.0 = raise
        # only when 100% of attempted blends fail; configurable down to 0.0
        # for fail-fast). Evaluate whenever there are errors -- gating on
        # `not blended` would silently swallow lowered thresholds the moment
        # any single blend succeeds, which is the bug a partial-failure
        # threshold is meant to catch.
        attempted = len(generated)
        if attempted and errors:
            failure_rate = len(errors) / attempted
            if failure_rate >= failure_threshold:
                sample_lines = [
                    f"{e['material']}: [{e['type']}] {e['message']}" for e in errors[:3]
                ]
                sample = "\n  - ".join(sample_lines)
                more = f"\n  ... ({len(errors) - 3} more)" if len(errors) > 3 else ""
                threshold_pct = int(failure_threshold * 100)
                raise RuntimeError(
                    f"{len(errors)}/{attempted} blend operations failed "
                    f"(failure rate {failure_rate:.0%} >= "
                    f"threshold {threshold_pct}%). "
                    f"First errors:\n  - {sample}{more}"
                )
        return context
