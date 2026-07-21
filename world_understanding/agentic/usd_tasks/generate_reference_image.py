# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate photorealistic reference images from scene previews and a text prompt.

This is a self-contained, reusable task that:
- Takes rendered preview images (from ``render_preview``) and a user prompt
- Provisions an image-generation model via ``create_image_generation_model``
- Generates reference images conditioned on the preview + prompt
- Saves images to ``output_dir/generated_ref_{i}.png``
- Outputs ``generated_reference_image_paths`` in context
"""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.models.image_generation_models import (
    create_image_generation_model,
)
from world_understanding.utils.credentials import (
    API_KEY_ENV_VAR_MAP,
    get_env_api_key_for_backend,
    get_nim_api_key_for_base_url,
    get_openai_api_key_for_base_url,
    is_placeholder_api_key,
    resolve_endpoint_api_key,
)
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class GenerateReferenceImageTask(Task):
    """Generate photorealistic reference images using an image-generation model.

    This task is **self-contained**: it provisions its own image-generation
    model so that it can be dropped into any workflow without external model
    provisioning.

    Input context keys:
        - rendered_preview_paths: list[str] — preview images from render_preview
        - image_gen_config: dict — e.g.
            ``{"backend": "gemini", "model": "gemini-3-pro-image-preview"}``
        - image_gen_prompt: str — user description of desired look
        - output_dir: str — directory to save generated images
        - num_images: int — how many images to generate (default 1)
        - reference_images: list[str] — optional existing reference images to
          condition the generation on (e.g. product photos, mood boards)

    Output context keys:
        - generated_reference_image_paths: list[str] — paths to generated images
    """

    def __init__(self) -> None:
        self.name = "GenerateReferenceImage"
        self.description = "Generate reference images from preview renders + prompt"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        # ── Resolve inputs ────────────────────────────────────────────
        preview_paths: list[str] = context.get("rendered_preview_paths", [])
        if not preview_paths:
            raise ValueError(
                "rendered_preview_paths not found in context. "
                "Run the render_preview step first or provide paths explicitly."
            )

        image_gen_config: dict[str, Any] = context.get("image_gen_config", {})
        prompt: str = context.get("image_gen_prompt", "")
        if not prompt:
            # Auto-build prompt from identification (set by IdentifyAssetTask)
            identification = context.get("identification", {})
            if identification and identification.get("asset_type") != "unknown":
                prompt = self._prompt_from_identification(identification)
                listener.info(
                    f"Auto-generated prompt from identification: {prompt[:150]}..."
                )
            else:
                raise ValueError(
                    "image_gen_prompt is required — describe the desired look. "
                    "Alternatively, run the identify_asset step first to "
                    "auto-generate the prompt."
                )

        # Append additional instructions if provided (composable with auto-generated prompt)
        additional_prompt = context.get("additional_prompt", "")
        if additional_prompt:
            prompt = f"{prompt}\n\nAdditional instructions: {additional_prompt}"
            listener.info(f"Appended additional prompt: {additional_prompt[:100]}...")

        output_dir = Path(context.get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Default: generate one reference image per preview angle.
        num_images_val = context.get("num_images")
        num_images: int = (
            num_images_val if num_images_val is not None else len(preview_paths)
        )
        reference_images: list[str] = context.get("reference_images", [])

        backend = image_gen_config.get("backend", "gemini")
        model_kwargs: dict[str, Any] = {}
        for key in ("model", "base_url", "timeout"):
            if key in image_gen_config:
                model_kwargs[key] = image_gen_config[key]
        explicit_api_key = resolve_endpoint_api_key(
            image_gen_config.get("api_key"),
            image_gen_config.get("api_key_env"),
        )
        api_key: str | None = None
        if backend == "nim":
            api_key = get_nim_api_key_for_base_url(
                model_kwargs.get("base_url"),
                explicit_api_key,
            )
        elif backend == "openai":
            api_key = get_openai_api_key_for_base_url(
                model_kwargs.get("base_url"),
                explicit_api_key,
            )
        elif backend in API_KEY_ENV_VAR_MAP:
            api_key = get_env_api_key_for_backend(backend, explicit_api_key)
        elif explicit_api_key is not None:
            explicit_api_key_str = str(explicit_api_key).strip()
            if explicit_api_key_str and not is_placeholder_api_key(
                explicit_api_key_str
            ):
                api_key = explicit_api_key_str
        if api_key:
            model_kwargs["api_key"] = api_key

        listener.info(
            f"Generating {num_images} reference image(s) via {backend} "
            f"using {len(preview_paths)} preview(s)"
            + (
                f" and {len(reference_images)} reference image(s)"
                if reference_images
                else ""
            )
        )

        # ── Provision image-generation model ──────────────────────────
        model = create_image_generation_model(backend, **model_kwargs)
        listener.info(f"Model provisioned: {model.model_name} ({model.backend_name})")

        # ── Generate images ───────────────────────────────────────────
        # Chained generation for consistency across angles:
        # - Image 1: uses all previews + scene references to establish the look
        # - Image 2+: uses the first generated image as style reference +
        #   one specific preview angle to match
        generated_paths: list[str] = []

        for i in range(num_images):
            pairs: list[tuple[str, str]] = []

            # Scene reference images (user-provided)
            for idx, ref_path in enumerate(reference_images):
                desc = (
                    f"This is reference image {idx + 1} showing the "
                    "desired look / style to match."
                )
                pairs.append((desc, ref_path))

            if i == 0:
                # First image: use ALL previews to establish canonical look
                for idx, preview_path in enumerate(preview_paths):
                    desc = (
                        f"This is preview image {idx + 1} of a 3D scene "
                        "rendered from a USD file."
                    )
                    pairs.append((desc, preview_path))

                final_prompt = (
                    "Based on the image(s) above, generate a single "
                    f"photorealistic reference image that shows: {prompt}"
                )
            else:
                # Subsequent images: chain from first generated image +
                # one specific preview angle for consistency
                pairs.append(
                    (
                        "This is the previously generated reference image. "
                        "Match its colors, materials, and style EXACTLY.",
                        generated_paths[0],
                    )
                )

                preview_idx = i % len(preview_paths)
                pairs.append(
                    (
                        "This is a 3D preview from a different camera angle. "
                        "Generate a reference image from THIS angle while "
                        "keeping the exact same colors and materials as the "
                        "previous reference.",
                        preview_paths[preview_idx],
                    )
                )

                final_prompt = (
                    "Generate a photorealistic reference image matching "
                    "the camera angle of the 3D preview above, while "
                    "maintaining the EXACT SAME colors and materials as "
                    f"the previous reference image. Details: {prompt}"
                )

            listener.info(f"Generating image {i + 1}/{num_images}...")
            try:
                generated_image = model.generate_with_image_prompt_pairs(
                    image_prompt_pairs=pairs,
                    final_prompt=final_prompt,
                    max_tokens=4096,
                )

                save_path = output_dir / f"generated_ref_{i}.png"
                generated_image.save(str(save_path))
                generated_paths.append(str(save_path))
                listener.info(f"Saved generated reference: {save_path.name}")

            except Exception as e:
                listener.error(f"Failed to generate image {i + 1}: {e}")
                raise

        listener.info(
            f"Generated {len(generated_paths)} reference image(s) to {output_dir}"
        )

        # ── Update context ────────────────────────────────────────────
        context["generated_reference_image_paths"] = generated_paths
        return context

    @staticmethod
    def _prompt_from_identification(identification: dict[str, Any]) -> str:
        """Build a generation prompt from asset identification results."""
        asset_type = identification.get("asset_type", "object")
        subtype = identification.get("asset_subtype", "")
        description = identification.get("asset_description", "")
        colors = identification.get("expected_colors", "")

        parts = ["A photorealistic product photograph of"]
        if subtype and subtype != "unknown":
            parts.append(f"a {subtype}")
        elif asset_type and asset_type != "unknown":
            parts.append(f"a {asset_type}")
        else:
            parts.append("the object shown in the 3D preview")

        if description:
            parts.append(f"({description})")
        parts.append(".")

        if colors:
            parts.append(f"Color scheme: {colors}.")

        parts.append(
            "Match the geometry and proportions from the 3D preview. "
            "Professional product photography, clean neutral background."
        )
        return " ".join(parts)
