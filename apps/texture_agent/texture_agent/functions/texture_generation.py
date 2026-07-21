# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Texture variation generation following the Texture Variation API spec.

Implements the data models and client from the Visual & Physical USD
Texture Variation Generation API. The local engine implementation uses
world_understanding image generation models as the backend.

See: docs/texture_variation_api.md for the full API specification.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apps.texture_gen_service_common.artifacts import local_path_from_file_uri
from apps.texture_gen_service_common.prompting import (
    NIM_MAX_PROMPT_CHARS,
    PromptBudgetError,
    append_bounded_instruction,
)
from PIL import Image
from world_understanding.utils.credentials import (
    get_env_api_key_for_backend,
    get_nim_api_key_for_base_url,
    get_openai_api_key_for_base_url,
    resolve_endpoint_api_key,
)

logger = logging.getLogger(__name__)

# Default prompt suffix for PBR texture generation
_TEXTURE_PROMPT_SUFFIX = (
    "The image should be a flat texture map suitable for use as a "
    "PBR albedo/base color map. No 3D objects, no perspective, "
    "no lighting effects -- just a flat, front-facing material "
    "texture that tiles seamlessly."
)
_NORMAL_PROMPT_SUFFIX = (
    "Generate a tangent-space normal map texture. "
    "The image should be predominantly blue-purple (RGB ~128,128,255) "
    "with subtle red/green variations encoding surface bumps, "
    "scratches, and surface detail. "
    "No 3D objects, no perspective -- just a flat normal map texture."
)
_ROUGHNESS_PROMPT_SUFFIX = (
    "Generate a PBR roughness texture map as a grayscale image. "
    "White = rough/matte areas, black = smooth/glossy areas. "
    "Worn, scratched, or corroded areas should be brighter (rougher). "
    "Clean, polished areas should be darker (smoother). "
    "No 3D objects, no perspective -- just a flat grayscale texture."
)
_NORMAL_PROMPT_PREFIX = "Normal map for: "
_ROUGHNESS_PROMPT_PREFIX = "Roughness map for: "
# Keep this alias for compatibility with internal diagnostics/tests while the
# provider contract itself lives in the shared service helper.
_NIM_MAX_PROMPT_CHARS = NIM_MAX_PROMPT_CHARS


# ---------------------------------------------------------------------------
# API Data Models (from texture_variation_api.md)
# ---------------------------------------------------------------------------


@dataclass
class Conditioning:
    """Conditioning inputs for texture generation.

    At least one field must be non-empty.
    """

    text_prompt: str | None = None
    """Text describing the target texture appearance."""

    reference_image_uris: list[str] = field(default_factory=list)
    """Style/material reference images (file paths or URIs)."""

    turntable_video_uri: str | None = None
    """Video of the asset for multi-view conditioning."""

    multiview_image_uris: list[str] = field(default_factory=list)
    """Ordered still images for multi-view conditioning."""

    def validate(self) -> None:
        """Raise ValueError if no conditioning input is provided."""
        has_prompt = bool(self.text_prompt and self.text_prompt.strip())
        has_refs = bool(self.reference_image_uris)
        has_video = bool(self.turntable_video_uri and self.turntable_video_uri.strip())
        has_multiview = bool(self.multiview_image_uris)
        if not (has_prompt or has_refs or has_video or has_multiview):
            raise ValueError(
                "At least one non-empty conditioning input is required "
                "(text_prompt, reference_image_uris, turntable_video_uri, "
                "or multiview_image_uris)."
            )


@dataclass
class TextureTarget:
    """Selected Texture Agent material/prim scope for projection backends."""

    material_name: str | None = None
    """Texture Agent material key, for example ``Aluminum_Matte``."""

    material_path: str | None = None
    """USD material prim path selected by Texture Agent."""

    prim_paths: list[str] = field(default_factory=list)
    """Geometry prim paths bound to this generation unit."""

    mode: str = "per_material"
    """Generation mode: ``per_material`` or ``per_prim``."""

    strict_scope: bool = True
    """If true, the backend must not broaden beyond this target."""


@dataclass
class TextureVariationConfig:
    """Configuration for texture variation generation."""

    strength: float = 0.8
    """Edit strength (0.0 = no change, 1.0 = full regeneration)."""

    seed: int | None = None
    """Seed for reproducibility. None = random."""

    variant_name: str | None = None
    """Human-readable name for the variant. None = auto-generated."""

    engine: str | None = None
    """Route to a specific backend engine. None = server default."""

    texture_size: int | None = None
    """Requested square texture resolution. None = backend/source default."""

    custom_parameters: dict[str, Any] = field(default_factory=dict)
    """Engine-specific overrides."""


@dataclass
class BackendCapabilities:
    """Projection backend capability hints or resolved response capabilities."""

    image_conditioning: bool | None = None
    multiview: bool | None = None
    normal_map: bool | None = None
    orm: bool | None = None
    masks: bool | None = None
    coverage: bool | None = None
    geometry_output: str | None = None


@dataclass
class MapArtifact:
    """Normalized backend map artifact metadata."""

    uri: str
    width: int | None = None
    height: int | None = None
    mime_type: str = "image/png"
    colorspace: str | None = None
    packing: str | None = None


@dataclass
class GeneratedTextures:
    """Paths to the generated texture set.

    Projection backends may return degraded albedo-only results. Missing maps
    are represented as ``None`` and should be paired with diagnostics.
    """

    albedo: str | None = None
    """Path/URI to the albedo (base color) texture."""

    normal: str | None = None
    """Path/URI to the normal map texture."""

    orm: str | None = None
    """Path/URI to the packed ORM texture (Occlusion=R, Roughness=G, Metallic=B)."""


@dataclass
class GenerationResult:
    """Result of a completed texture variation job."""

    variant_asset_uri: str
    """Path/URI to the output USD file with texture overrides."""

    variant_name: str
    """Human-readable name for this variant."""

    generated_textures: GeneratedTextures
    """Paths to the generated PBR texture files."""

    maps: dict[str, MapArtifact] = field(default_factory=dict)
    """Canonical map artifacts keyed by channel."""

    auxiliary_artifacts: dict[str, Any] = field(default_factory=dict)
    """Optional masks, debug previews, and ignored geometry artifacts."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Backend/model/capability/seed/size/timing/coverage metadata."""

    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    """Structured texture-agent-diagnostic.v1 diagnostics."""


@dataclass
class JobStatus:
    """Status of a texture variation job."""

    job_id: str
    """Unique job identifier."""

    status: str
    """Job status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled'."""

    progress: int = 0
    """Progress percentage (0-100)."""

    message: str | None = None
    """Human-readable status message."""

    result: GenerationResult | None = None
    """Populated when status == 'completed'."""

    error_message: str | None = None
    """Populated when status == 'failed'."""


# ---------------------------------------------------------------------------
# Engine abstraction (internal)
# ---------------------------------------------------------------------------


class BaseTextureEngine(ABC):
    """Abstract base for texture generation engines.

    An engine produces PBR texture files (albedo, normal, ORM) given
    conditioning inputs. Engines are swappable without changing the
    client API contract.
    """

    @abstractmethod
    def generate(
        self,
        conditioning: Conditioning,
        config: TextureVariationConfig,
        output_dir: Path,
        source_resolution: tuple[int, int] | None = None,
    ) -> GeneratedTextures:
        """Generate PBR textures.

        Args:
            conditioning: Text prompt, reference images, etc.
            config: Strength, seed, engine-specific params.
            output_dir: Directory to write output texture files.
            source_resolution: Resolution of the source textures
                (output should match). None = use default 1024x1024.

        Returns:
            GeneratedTextures with paths to the written files.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier."""
        ...


class ImageGenEngine(BaseTextureEngine):
    """Engine using world_understanding image generation models.

    Generates albedo textures through the configured image generation backend.
    Normal and ORM are placeholder identity maps in v1 (no PBR extraction model
    available yet).
    """

    def __init__(
        self,
        backend: str = "nim",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        self._backend = backend
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._model_instance: Any = None
        self._conditioning_warning_emitted = False
        self._conditioning_warning_lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"image_gen ({self._backend})"

    def _ensure_model(self) -> Any:
        """Lazily create the image generation model."""
        if self._model_instance is None:
            from world_understanding.functions.models.image_generation_models import (
                create_image_generation_model,
            )

            kwargs: dict[str, Any] = {}
            if self._model:
                kwargs["model"] = self._model
            if self._base_url:
                kwargs["base_url"] = self._base_url
            explicit_api_key = resolve_endpoint_api_key(
                self._api_key,
                self._api_key_env,
            )
            if self._backend == "nim":
                api_key = get_nim_api_key_for_base_url(
                    self._base_url,
                    explicit_api_key,
                )
            elif self._backend == "openai":
                api_key = get_openai_api_key_for_base_url(
                    self._base_url,
                    explicit_api_key,
                )
            else:
                api_key = get_env_api_key_for_backend(
                    self._backend,
                    explicit_api_key,
                )
            if api_key:
                kwargs["api_key"] = api_key

            logger.info(
                "Initializing image gen model: backend=%s, model=%s, base_url=%s",
                self._backend,
                self._model or "(default)",
                self._base_url or "(default)",
            )
            self._model_instance = create_image_generation_model(
                self._backend, **kwargs
            )
        return self._model_instance

    def _generate_image(
        self,
        model: Any,
        prompt: str,
        size: tuple[int, int],
        ref_images: list[Image.Image] | None = None,
    ) -> Image.Image:
        """Generate a single image and resize to target."""
        image = model.generate(prompt, images=ref_images)
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return image

    def _warn_if_conditioning_unsupported(self, model: Any) -> None:
        """Log unsupported image-conditioning behavior once per engine instance."""
        if self._conditioning_warning_emitted:
            return
        with self._conditioning_warning_lock:
            if self._conditioning_warning_emitted:
                return
            self._conditioning_warning_emitted = True
            logger.warning(
                "Backend %s does not support image conditioning; normal "
                "and roughness maps will be generated from text only for "
                "this run.",
                getattr(model, "backend_name", "unknown"),
            )

    def generate(
        self,
        conditioning: Conditioning,
        config: TextureVariationConfig,
        output_dir: Path,
        source_resolution: tuple[int, int] | None = None,
    ) -> GeneratedTextures:
        size = source_resolution or (1024, 1024)
        job_prefix = config.variant_name or "texture"
        base_prompt = conditioning.text_prompt or ""

        # Build every channel prompt before initializing or calling the model.
        # A channel-specific prefix can make the normal/roughness prompt exceed
        # a provider limit even when the albedo prompt still fits.  Preflighting
        # the complete set prevents a partial PBR result after an albedo request
        # has already been launched.
        max_prompt_chars = (
            _NIM_MAX_PROMPT_CHARS if self._backend.strip().lower() == "nim" else None
        )
        channel_specs = (
            (
                base_prompt,
                _TEXTURE_PROMPT_SUFFIX,
                "Flat PBR albedo/base-color texture map.",
                0,
            ),
            (
                f"{_NORMAL_PROMPT_PREFIX}{base_prompt}",
                _NORMAL_PROMPT_SUFFIX,
                "Tangent-space normal texture map.",
                len(_NORMAL_PROMPT_PREFIX),
            ),
            (
                f"{_ROUGHNESS_PROMPT_PREFIX}{base_prompt}",
                _ROUGHNESS_PROMPT_SUFFIX,
                "Grayscale PBR roughness texture map.",
                len(_ROUGHNESS_PROMPT_PREFIX),
            ),
        )
        channel_prompts: list[str] = []
        prompt_errors: list[PromptBudgetError] = []
        for (
            prompt,
            instruction,
            minimum_instruction,
            service_prefix_chars,
        ) in channel_specs:
            try:
                channel_prompts.append(
                    append_bounded_instruction(
                        prompt,
                        instruction,
                        max_chars=max_prompt_chars,
                        minimum_instruction=minimum_instruction,
                        service_prefix_chars=service_prefix_chars,
                    )
                )
            except PromptBudgetError as exc:
                prompt_errors.append(exc)
                channel_prompts.append("")

        if prompt_errors:
            raise min(prompt_errors, key=lambda exc: exc.max_text_prompt_chars)

        albedo_prompt, normal_prompt, roughness_prompt = channel_prompts

        model = self._ensure_model()

        # If the backend can't accept img2img conditioning (e.g. the cloud
        # NIM GenAI endpoint), don't pass the albedo as a reference for
        # the normal/roughness passes -- the endpoint silently drops it
        # and we'd just burn a round-trip. The generated maps will be
        # text-conditioned only.
        supports_conditioning = getattr(model, "supports_image_conditioning", True)
        if not supports_conditioning:
            self._warn_if_conditioning_unsupported(model)

        # Load reference images for conditioning
        ref_images: list[Image.Image] | None = None
        if conditioning.reference_image_uris:
            ref_images = []
            for uri in conditioning.reference_image_uris:
                path = local_path_from_file_uri(uri)
                if path is None:
                    raise ValueError(
                        f"Unsupported URI scheme in reference_image_uris: {uri}"
                    )
                with Image.open(path) as image:
                    ref_images.append(image.convert("RGB").copy())

        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Albedo ---
        logger.info("Generating albedo texture (size=%s)", size)
        albedo_img = self._generate_image(model, albedo_prompt, size, ref_images)
        albedo_path = output_dir / f"{job_prefix}_albedo.png"
        albedo_img.save(str(albedo_path))

        # --- Normal map ---
        # Use the albedo as conditioning so the normal matches the same
        # surface features (scratches, dents, etc.) -- skipped when the
        # backend can't accept reference images (see supports_conditioning
        # check above).
        logger.info("Generating normal map")
        normal_ref = [albedo_img] if supports_conditioning else None
        normal_img = self._generate_image(model, normal_prompt, size, normal_ref)
        normal_path = output_dir / f"{job_prefix}_normal.png"
        normal_img.save(str(normal_path))

        # --- ORM (Occlusion, Roughness, Metallic) ---
        # Generate a roughness map, then pack into ORM
        logger.info("Generating roughness map")
        roughness_ref = [albedo_img] if supports_conditioning else None
        roughness_img = self._generate_image(
            model, roughness_prompt, size, roughness_ref
        )

        # Pack into ORM: R=Occlusion(white), G=Roughness, B=Metallic(black)
        import numpy as np

        roughness_gray = np.array(roughness_img.convert("L"))
        orm_arr = np.zeros((*roughness_gray.shape, 3), dtype=np.uint8)
        orm_arr[:, :, 0] = 255  # Occlusion = 1.0 (no occlusion)
        orm_arr[:, :, 1] = roughness_gray  # Roughness from generated map
        orm_arr[:, :, 2] = 0  # Metallic = 0 (keep from material constant)
        orm_path = output_dir / f"{job_prefix}_orm.png"
        Image.fromarray(orm_arr).save(str(orm_path))

        logger.info(
            "Generated PBR set: albedo=%s, normal=%s, orm=%s",
            albedo_path,
            normal_path,
            orm_path,
        )

        return GeneratedTextures(
            albedo=str(albedo_path),
            normal=str(normal_path),
            orm=str(orm_path),
        )


# ---------------------------------------------------------------------------
# Client (from texture_variation_api.md Section 2.1)
# ---------------------------------------------------------------------------


class TextureVariationClient:
    """Client for the texture variation API.

    This local implementation runs the generation engine in-process.
    When a remote service is deployed, this can be replaced with a
    REST client that calls POST/GET/DELETE on /v1/texture-variations.

    Example:
        client = TextureVariationClient()
        status = client.generate(
            source_asset_uri="file:///path/to/asset.usd",
            conditioning=Conditioning(
                text_prompt="heavily rusted metal with chipped paint"
            ),
            config=TextureVariationConfig(strength=0.9),
        )
        if status.status == "completed" and status.result:
            print(status.result.generated_textures.albedo)
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        engine: BaseTextureEngine | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Initialize the texture variation client.

        Args:
            endpoint_url: Not used in local mode. Reserved for REST client.
            api_key: Not used in local mode. Reserved for REST client.
            engine: Texture generation engine. Defaults to ImageGenEngine.
            output_dir: Base directory for output files. Defaults to temp dir.
        """
        self._endpoint_url = endpoint_url
        self._api_key = api_key
        self._engine = engine or ImageGenEngine()
        self._output_dir = Path(output_dir) if output_dir else None

    def generate(
        self,
        source_asset_uri: str,
        conditioning: Conditioning,
        config: TextureVariationConfig | None = None,
        wait: bool = True,
        timeout_sec: int = 600,
        target: TextureTarget | None = None,
    ) -> JobStatus:
        """Submit a texture variation job.

        In local mode, runs synchronously and returns completed status.

        Args:
            source_asset_uri: URI to the source USD asset.
            conditioning: Text prompt, reference images, etc.
            config: Generation configuration.
            wait: Ignored in local mode (always synchronous).
            timeout_sec: Ignored in local mode.
            target: Optional selected material/prim scope. Ignored in local mode.

        Returns:
            JobStatus with status='completed' and result, or 'failed'.
        """
        config = config or TextureVariationConfig()
        conditioning.validate()

        job_id = f"vj-{uuid.uuid4().hex[:12]}"
        variant_name = config.variant_name or f"variant_{job_id}"

        # Determine output directory
        if self._output_dir:
            output_dir = self._output_dir / variant_name
        else:
            import tempfile

            # NOTE: The caller is responsible for cleaning up the temp directory
            # returned via the result paths. The directory must persist because
            # the caller reads the generated texture files after this method returns.
            output_dir = Path(tempfile.mkdtemp()) / variant_name

        logger.info(
            "Starting texture variation job %s (engine=%s, strength=%.2f)",
            job_id,
            config.engine or self._engine.name,
            config.strength,
        )

        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Run the engine
            textures = self._engine.generate(
                conditioning=conditioning,
                config=config,
                output_dir=output_dir,
                source_resolution=(config.texture_size, config.texture_size)
                if config.texture_size
                else None,
            )

            result = GenerationResult(
                variant_asset_uri=source_asset_uri,  # In v1, no USD rewrite
                variant_name=variant_name,
                generated_textures=textures,
                metadata={
                    "engine": self._engine.name,
                    "texture_size": config.texture_size,
                    "target": target.__dict__ if target else {},
                },
            )

            return JobStatus(
                job_id=job_id,
                status="completed",
                progress=100,
                result=result,
            )

        except Exception as e:
            logger.exception("Texture variation job %s failed", job_id)
            return JobStatus(
                job_id=job_id,
                status="failed",
                progress=0,
                error_message=str(e),
            )

    def get_status(self, job_id: str) -> JobStatus:
        """Query job status. In local mode, jobs complete synchronously."""
        raise NotImplementedError(
            "Local client runs synchronously. "
            "Use the REST client for async job tracking."
        )

    def cancel(self, job_id: str) -> None:
        """Cancel a job. Not applicable in local mode."""
        raise NotImplementedError(
            "Local client runs synchronously. "
            "Use the REST client for async job cancellation."
        )


# ---------------------------------------------------------------------------
# Convenience: Legacy interfaces (backward compat with existing tasks)
# ---------------------------------------------------------------------------


@dataclass
class TextureRequest:
    """Simplified input for direct texture generation (legacy)."""

    prompt: str
    material_name: str
    base_color: tuple[float, float, float]
    size: tuple[int, int] = (1024, 1024)
    reference_image: Image.Image | None = None


@dataclass
class TextureResult:
    """Simplified output from texture generation (legacy)."""

    image: Image.Image
    prompt_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTextureGenerator(ABC):
    """Legacy generator interface. Prefer TextureVariationClient."""

    @abstractmethod
    def generate(self, request: TextureRequest) -> TextureResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class ImageGenTextureGenerator(BaseTextureGenerator):
    """Legacy wrapper. Delegates to ImageGenEngine internally."""

    def __init__(
        self,
        backend: str = "nim",
        model: str | None = None,
    ) -> None:
        self._engine = ImageGenEngine(backend=backend, model=model)

    @property
    def name(self) -> str:
        return self._engine.name

    def _ensure_model(self) -> Any:
        """Pre-initialize the engine's model (for thread safety)."""
        return self._engine._ensure_model()

    def generate(self, request: TextureRequest) -> TextureResult:
        conditioning = Conditioning(text_prompt=request.prompt)

        config = TextureVariationConfig(strength=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            if request.reference_image:
                # Save reference image into the same temp directory so it is
                # cleaned up automatically when the context manager exits.
                ref_path = Path(tmpdir) / "reference.png"
                request.reference_image.save(str(ref_path))
                conditioning.reference_image_uris = [str(ref_path)]

            textures = self._engine.generate(
                conditioning=conditioning,
                config=config,
                output_dir=Path(tmpdir),
                source_resolution=request.size,
            )
            with Image.open(textures.albedo) as generated:
                # Load into memory before temp cleanup.
                image = generated.copy()

        return TextureResult(
            image=image,
            prompt_used=request.prompt,
            metadata={"engine": self._engine.name},
        )


def create_texture_generator(
    backend: str = "nim",
    model: str | None = None,
) -> ImageGenTextureGenerator:
    """Create a legacy texture generator."""
    return ImageGenTextureGenerator(backend=backend, model=model)
