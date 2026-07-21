# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple Texture Variation API service using image gen models.

A lightweight implementation of the Texture Variation API spec
(texture_variation_api.md) using Gemini image generation for albedo,
normal, and roughness maps. No GPU or conda required — runs anywhere
the world_understanding package is installed.

Usage:
    source .venv/bin/activate
    uvicorn apps.texture_gen_simple_service.app:app --port 8000

    # Or directly:
    python apps/texture_gen_simple_service/app.py --port 8000
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from apps.texture_gen_service_common import (
    NIM_MAX_PROMPT_CHARS,
    BackendCapabilities,
    BackendHealth,
    CreateJobRequest,
    GeneratedTextures,
    GenerationResult,
    HealthResponse,
    JobStatus,
    MapArtifact,
    PromptBudgetError,
    TextureGenerationBackend,
    TextureGenerationBackendError,
    append_bounded_instruction,
    create_app,
)

__all__ = [
    "BackendCapabilities",
    "CreateJobRequest",
    "GeneratedTextures",
    "GenerationResult",
    "HealthResponse",
    "JobStatus",
    "MapArtifact",
    "SimpleImageGenerationBackend",
    "app",
]

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path(
    os.environ.get(
        "TEXTURE_OUTPUT_DIR",
        os.path.join(tempfile.gettempdir(), "texture_gen_simple_service"),
    )
)
_BACKEND = os.environ.get("TEXTURE_GEN_BACKEND", "nim")
_MODEL = os.environ.get("TEXTURE_GEN_MODEL", None)
_BASE_URL = os.environ.get("TEXTURE_GEN_BASE_URL", None)
_API_KEY = os.environ.get("TEXTURE_GEN_API_KEY", None)
_RETRYABLE_IMAGE_GEN_STATUS_CODES = {408, 429, 500, 502, 503, 504}
# Retain the private alias used by existing callers and tests while sourcing
# the provider contract from the shared prompt-budget module.
_NIM_MAX_PROMPT_CHARS = NIM_MAX_PROMPT_CHARS

# Lazy model instance (protected by _model_lock for thread safety)
_model_instance: Any = None
_model_lock = threading.Lock()

# Prompt templates
_ALBEDO_SUFFIX = (
    "The image should be a flat texture map suitable for use as a "
    "PBR albedo/base color map. No 3D objects, no perspective, "
    "no lighting effects -- just a flat, front-facing material "
    "texture that tiles seamlessly."
)

_NORMAL_SUFFIX = (
    "Generate a tangent-space normal map texture. "
    "The image should be predominantly blue-purple (RGB ~128,128,255) "
    "with subtle red/green variations encoding surface bumps, "
    "scratches, and surface detail. "
    "No 3D objects, no perspective -- just a flat normal map texture."
)

_ROUGHNESS_SUFFIX = (
    "Generate a PBR roughness texture map as a grayscale image. "
    "White = rough/matte areas, black = smooth/glossy areas. "
    "Worn, scratched, or corroded areas should be brighter (rougher). "
    "Clean, polished areas should be darker (smoother). "
    "No 3D objects, no perspective -- just a flat grayscale texture."
)
_NORMAL_PROMPT_PREFIX = "Normal map for: "
_ROUGHNESS_PROMPT_PREFIX = "Roughness map for: "

_MINIMUM_CHANNEL_INSTRUCTIONS = {
    _ALBEDO_SUFFIX: "Flat PBR albedo/base-color texture map.",
    _NORMAL_SUFFIX: "Tangent-space normal texture map.",
    _ROUGHNESS_SUFFIX: "Grayscale PBR roughness texture map.",
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _get_model() -> Any:
    """Lazy-init the image generation model (thread-safe)."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    with _model_lock:
        if _model_instance is None:
            from world_understanding.functions.models.image_generation_models import (
                create_image_generation_model,
            )

            kwargs: dict[str, Any] = {}
            if _MODEL:
                kwargs["model"] = _MODEL
            if _BASE_URL:
                kwargs["base_url"] = _BASE_URL
            if _API_KEY:
                kwargs["api_key"] = _API_KEY

            logger.info(
                "Initializing image gen model: backend=%s, model=%s, base_url=%s",
                _BACKEND,
                _MODEL or "(default)",
                "(configured)" if _BASE_URL else "(default)",
            )
            _model_instance = create_image_generation_model(_BACKEND, **kwargs)
        return _model_instance


def _max_workers() -> int:
    value = os.environ.get("TEXTURE_GEN_MAX_WORKERS", "2")
    try:
        return max(1, int(value))
    except ValueError:
        return 2


def _retry_attempts() -> int:
    value = os.environ.get("TEXTURE_GEN_RETRY_ATTEMPTS", "3")
    try:
        return max(1, int(value))
    except ValueError:
        return 3


def _retry_backoff_sec() -> float:
    value = os.environ.get("TEXTURE_GEN_RETRY_BACKOFF_SEC", "1.0")
    try:
        return max(0.0, float(value))
    except ValueError:
        return 1.0


def _exception_status_code(exc: BaseException) -> int | None:
    candidates = [
        getattr(exc, "status", None),
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(exc, "code", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _is_retryable_image_gen_error(exc: BaseException) -> bool:
    status_code = _exception_status_code(exc)
    if status_code in _RETRYABLE_IMAGE_GEN_STATUS_CODES:
        return True
    exc_name = exc.__class__.__name__.lower()
    if "ratelimit" in exc_name or "rate_limit" in exc_name:
        return True
    message = str(exc).lower()
    return "rate limit" in message or "too many requests" in message


def _generate_image(
    model: Any,
    prompt: str,
    size: tuple[int, int],
    *,
    cancel_event: threading.Event,
    ref: Image.Image | None = None,
) -> Image.Image:
    """Generate a single image."""
    images = [ref] if ref else None
    attempts = _retry_attempts()
    backoff_sec = _retry_backoff_sec()
    for attempt in range(1, attempts + 1):
        try:
            img = model.generate(prompt, images=images)
            if img.size != size:
                img = img.resize(size, Image.Resampling.LANCZOS)
            return img
        except Exception as exc:
            if attempt >= attempts or not _is_retryable_image_gen_error(exc):
                raise
            logger.warning(
                "Image generation attempt %s/%s failed with retryable error: %s",
                attempt,
                attempts,
                exc,
            )
            if cancel_event.wait(backoff_sec):
                _raise_if_cancelled(cancel_event)
    raise RuntimeError("Image generation retry loop exited unexpectedly.")


def _map_artifact(
    path: Path,
    *,
    width: int,
    height: int,
    colorspace: str,
    packing: str | None = None,
) -> MapArtifact:
    return MapArtifact(
        uri=path.as_uri(),
        width=width,
        height=height,
        colorspace=colorspace,
        packing=packing,
    )


def _raise_if_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise RuntimeError("Texture generation job was cancelled.")


def _image_prompt(
    text_prompt: str,
    instruction: str,
    *,
    service_prefix_chars: int = 0,
) -> str:
    """Append service instructions without exceeding provider prompt limits.

    The hosted NIM GenAI schema accepts at most 800 characters. The Texture
    Variation API prompt can be longer than the remaining budget after the
    service adds its channel-specific prefix and suffix. Preserve the caller's
    prompt, account for the prefix in diagnostics, and trim only the
    service-owned instruction.
    """
    max_chars = _NIM_MAX_PROMPT_CHARS if _BACKEND.strip().lower() == "nim" else None
    return append_bounded_instruction(
        text_prompt,
        instruction,
        max_chars=max_chars,
        minimum_instruction=_MINIMUM_CHANNEL_INSTRUCTIONS.get(
            instruction,
            instruction,
        ),
        service_prefix_chars=service_prefix_chars,
    )


class SimpleImageGenerationBackend(TextureGenerationBackend):
    """Image generation backend for the lightweight service."""

    @property
    def name(self) -> str:
        return f"texture_gen_simple_service:{_BACKEND}"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            image_conditioning=False,
            multiview=False,
            normal_map=True,
            orm=True,
            masks=False,
            coverage=False,
            geometry_output="none",
        )

    def health(self) -> BackendHealth:
        return BackendHealth(
            status="healthy",
            ready=True,
            warmup_complete=_model_instance is not None,
            gpu_available=None,
            capabilities={
                **self.capabilities().model_dump(exclude_none=True),
                "texture_size_max": 4096,
                "texture_size_default": 1024,
                "backend": _BACKEND,
                "model": _MODEL or "(default)",
                "base_url_configured": bool(_BASE_URL),
            },
        )

    def _failure_result(
        self,
        request: CreateJobRequest,
        *,
        variant_name: str,
        diagnostic: dict[str, Any],
        extra_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """Build a normalized result for failures before model launch."""
        capabilities = self.capabilities().model_dump(exclude_none=True)
        target = request.target.model_dump(exclude_none=True) if request.target else {}
        return GenerationResult(
            variant_asset_uri=request.source_asset_uri,
            variant_name=variant_name,
            generated_textures=GeneratedTextures(),
            metadata={
                "backend_name": self.name,
                "model": _MODEL or None,
                "endpoint_type": "local_simple_service",
                "seed": request.configuration.seed,
                "texture_size": request.configuration.texture_size or 1024,
                "target": target,
                "capabilities": capabilities,
                "requested_capabilities": (
                    request.capabilities.model_dump(exclude_none=True)
                    if request.capabilities
                    else {}
                ),
                "skipped_before_backend_launch": True,
                **(extra_metadata or {}),
            },
            diagnostics=[diagnostic],
        )

    def _reject_unsupported_conditioning(
        self,
        request: CreateJobRequest,
        *,
        variant_name: str,
    ) -> None:
        unsupported_fields: list[str] = []
        if request.conditioning.reference_image_uris:
            unsupported_fields.append("reference_image_uris")
        if request.conditioning.turntable_video_uri:
            unsupported_fields.append("turntable_video_uri")
        if request.conditioning.multiview_image_uris:
            unsupported_fields.append("multiview_image_uris")
        if not unsupported_fields:
            return

        capabilities = self.capabilities().model_dump(exclude_none=True)
        diagnostic = {
            "schema_version": "texture-agent-diagnostic.v1",
            "code": "BACKEND_CONDITIONING_UNSUPPORTED",
            "severity": "error",
            "stage": "generate_textures",
            "prim_path": (
                request.target.prim_paths[0]
                if request.target and request.target.prim_paths
                else None
            ),
            "material_name": (request.target.material_name if request.target else None),
            "message": (
                "simple_image_gen is a text-only backend and cannot use the "
                "requested reference, turntable, or multiview conditioning."
            ),
            "recommended_action": (
                "Remove the unsupported conditioning fields or select a backend "
                "that advertises the required capability."
            ),
            "details": {
                "unsupported_fields": unsupported_fields,
                "capabilities": capabilities,
            },
        }
        result = self._failure_result(
            request,
            variant_name=variant_name,
            diagnostic=diagnostic,
        )
        fields = ", ".join(unsupported_fields)
        raise TextureGenerationBackendError(
            f"BACKEND_CONDITIONING_UNSUPPORTED: simple_image_gen does not "
            f"support {fields}.",
            result=result,
        )

    def generate(
        self,
        request: CreateJobRequest,
        *,
        job_id: str,
        output_dir: Path,
        cancel_event: threading.Event,
    ) -> GenerationResult:
        """Run the full PBR generation pipeline (albedo + normal + roughness -> ORM)."""
        variant_name = request.configuration.variant_name or f"variant_{job_id}"
        self._reject_unsupported_conditioning(
            request,
            variant_name=variant_name,
        )

        prompt = request.conditioning.text_prompt
        if not prompt or not prompt.strip():
            raise TextureGenerationBackendError(
                "SIMPLE_SERVICE_TEXT_PROMPT_REQUIRED: "
                "texture_gen_simple_service requires conditioning.text_prompt."
            )
        prompt = prompt.strip()
        size_value = request.configuration.texture_size or 1024
        size = (size_value, size_value)

        channel_specs = {
            "albedo": (prompt, _ALBEDO_SUFFIX, 0),
            "normal": (
                f"{_NORMAL_PROMPT_PREFIX}{prompt}",
                _NORMAL_SUFFIX,
                len(_NORMAL_PROMPT_PREFIX),
            ),
            "roughness": (
                f"{_ROUGHNESS_PROMPT_PREFIX}{prompt}",
                _ROUGHNESS_SUFFIX,
                len(_ROUGHNESS_PROMPT_PREFIX),
            ),
        }
        prompts: dict[str, str] = {}
        prompt_errors: list[tuple[str, PromptBudgetError]] = []
        for channel, (
            channel_prompt,
            instruction,
            service_prefix_chars,
        ) in channel_specs.items():
            try:
                prompts[channel] = _image_prompt(
                    channel_prompt,
                    instruction,
                    service_prefix_chars=service_prefix_chars,
                )
            except PromptBudgetError as exc:
                prompt_errors.append((channel, exc))

        if prompt_errors:
            channel, exc = min(
                prompt_errors,
                key=lambda item: item[1].max_text_prompt_chars,
            )
            message = (
                "The text prompt is too long to retain the required "
                f"{channel} generation instruction within the configured "
                f"{exc.max_chars}-character backend limit."
            )
            diagnostic = {
                "schema_version": "texture-agent-diagnostic.v1",
                "code": "BACKEND_PROMPT_TOO_LONG",
                "severity": "error",
                "stage": "generate_textures",
                "prim_path": (
                    request.target.prim_paths[0]
                    if request.target and request.target.prim_paths
                    else None
                ),
                "material_name": (
                    request.target.material_name if request.target else None
                ),
                "message": message,
                "recommended_action": "Shorten conditioning.text_prompt and retry.",
                "details": {
                    "backend": _BACKEND,
                    "channel": channel,
                    "limit": exc.max_chars,
                    "prompt_length": exc.prompt_chars,
                    "maximum_text_prompt_length": exc.max_text_prompt_chars,
                },
            }
            result = self._failure_result(
                request,
                variant_name=variant_name,
                diagnostic=diagnostic,
                extra_metadata={"prompt_rejected_channel": channel},
            )
            raise TextureGenerationBackendError(
                f"BACKEND_PROMPT_TOO_LONG: {message}",
                result=result,
            ) from exc

        albedo_prompt = prompts["albedo"]
        normal_prompt = prompts["normal"]
        roughness_prompt = prompts["roughness"]

        model = _get_model()
        output_dir.mkdir(parents=True, exist_ok=True)
        supports_image_conditioning = bool(
            getattr(model, "supports_image_conditioning", True)
        )

        _raise_if_cancelled(cancel_event)
        logger.info("[%s] Generating albedo", job_id)
        albedo_img = _generate_image(
            model,
            albedo_prompt,
            size,
            cancel_event=cancel_event,
        )
        albedo_path = output_dir / f"{variant_name}_albedo.png"
        albedo_img.save(str(albedo_path))

        _raise_if_cancelled(cancel_event)
        logger.info("[%s] Generating normal map", job_id)
        normal_img = _generate_image(
            model,
            normal_prompt,
            size,
            cancel_event=cancel_event,
            ref=albedo_img if supports_image_conditioning else None,
        )
        normal_path = output_dir / f"{variant_name}_normal.png"
        normal_img.save(str(normal_path))

        _raise_if_cancelled(cancel_event)
        logger.info("[%s] Generating roughness map", job_id)
        roughness_img = _generate_image(
            model,
            roughness_prompt,
            size,
            cancel_event=cancel_event,
            ref=albedo_img if supports_image_conditioning else None,
        )

        # Pack into ORM: R=Occlusion(white), G=Roughness, B=Metallic(black)
        roughness_gray = np.array(roughness_img.convert("L"))
        orm_arr = np.zeros((*roughness_gray.shape, 3), dtype=np.uint8)
        orm_arr[:, :, 0] = 255  # Occlusion = 1.0
        orm_arr[:, :, 1] = roughness_gray
        orm_arr[:, :, 2] = 0  # Metallic = 0.0
        orm_path = output_dir / f"{variant_name}_orm.png"
        Image.fromarray(orm_arr).save(str(orm_path))

        logger.info("[%s] Complete: %s", job_id, output_dir)
        return GenerationResult(
            variant_asset_uri=request.source_asset_uri,
            variant_name=variant_name,
            generated_textures=GeneratedTextures(
                albedo=albedo_path.as_uri(),
                normal=normal_path.as_uri(),
                orm=orm_path.as_uri(),
            ),
            maps={
                "albedo": _map_artifact(
                    albedo_path,
                    width=size[0],
                    height=size[1],
                    colorspace="srgb",
                ),
                "normal": _map_artifact(
                    normal_path,
                    width=size[0],
                    height=size[1],
                    colorspace="linear",
                ),
                "orm": _map_artifact(
                    orm_path,
                    width=size[0],
                    height=size[1],
                    colorspace="linear",
                    packing="occlusion:R roughness:G metallic:B",
                ),
            },
            metadata={
                "backend_name": self.name,
                "model": getattr(model, "model_name", None) or _MODEL,
                "endpoint_type": "local_simple_service",
                "texture_size": size[0],
                "seed": request.configuration.seed,
                "target": (
                    request.target.model_dump(exclude_none=True)
                    if request.target
                    else {}
                ),
                "requested_capabilities": (
                    request.capabilities.model_dump(exclude_none=True)
                    if request.capabilities
                    else {}
                ),
                "capabilities": self.capabilities().model_dump(exclude_none=True),
                "supports_image_conditioning": supports_image_conditioning,
                "reference_image_count": len(request.conditioning.reference_image_uris),
                "multiview_image_count": len(request.conditioning.multiview_image_uris),
                "turntable_video_uri": request.conditioning.turntable_video_uri,
            },
        )


_backend = SimpleImageGenerationBackend()
app = create_app(
    backend=_backend,
    output_dir=_OUTPUT_DIR,
    title="Texture Variation API (Simple)",
    version="1.0.0",
    description=(
        "Generate PBR texture variations using image generation models. "
        "Lightweight implementation of the Texture Variation API spec."
    ),
    service_name="texture-gen-simple-service",
    max_workers=_max_workers(),
)


@app.get("/livez")
async def livez() -> dict[str, str]:
    """Return cheap process liveness without model warmup."""
    return {"status": "healthy", "service": "texture-gen-simple-service"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the development uvicorn server."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Simple Texture Variation API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
