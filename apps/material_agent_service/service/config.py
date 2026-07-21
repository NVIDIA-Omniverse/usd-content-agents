# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend configuration for Material Agent Service.

NOTE: This config now delegates VLM/LLM/rendering defaults to material_agent.api.defaults.
Only FastAPI-specific service settings are defined here.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import yaml
from material_agent import __version__
from material_agent.api.defaults import (
    DEFAULT_CLUSTER_BATCH_SIZE,
    DEFAULT_CLUSTER_EMBEDDING_BACKEND,
    DEFAULT_CLUSTER_EMBEDDING_MODEL,
    DEFAULT_CLUSTER_MAX_SIZE,
    DEFAULT_CLUSTER_MAX_WORKERS,
    DEFAULT_CLUSTER_MIN_PRIMS_TO_ACTIVATE,
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_MAX_TOKENS,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_TEMPERATURE,
)
from material_agent.simready import (
    DEFAULT_SIMREADY_RELEASE_TAG,
    SimReadyCatalogError,
    build_material_entries,
    is_simready_library_id,
    load_manifest,
)
from pydantic import Field
from pydantic_settings import BaseSettings
from world_understanding.utils.credentials import (
    get_env_api_key_for_backend,
    get_nim_api_key_for_base_url,
    get_openai_api_key_for_base_url,
    is_nvidia_provider_base_url,
    is_placeholder_api_key,
    resolve_endpoint_api_key,
)

logger = logging.getLogger(__name__)

_LOCAL_RENDER_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "ovrtx-rendering-api",
}


def _has_real_api_key(value: str | None) -> bool:
    """Return True when a service credential is non-empty and not a placeholder."""
    return bool(value and not is_placeholder_api_key(value))


def _is_local_render_endpoint(endpoint: str | None) -> bool:
    """Return True when the render endpoint targets a local renderer."""
    if not endpoint:
        return False

    host = urlparse(endpoint).hostname or endpoint
    return host.lower() in _LOCAL_RENDER_HOSTS


def _display_name_from_library_id(library_id: str) -> str:
    if library_id.startswith("simready-category:"):
        return f"simready category {library_id.removeprefix('simready-category:')}"
    return library_id.replace("_", " ").replace("-", " ")


def _backend_has_credentials(
    backend: str | None,
    *,
    nvidia_api_key: str | None,
    nim_base_url: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> bool:
    """Check whether the active backend has the credential it needs."""
    backend_name = (backend or "").lower()

    if not backend_name:
        return True
    import world_understanding.functions.models.backends  # noqa: F401
    from world_understanding.functions.models.backends.registry import (
        vlm_backend_requires_api_key,
    )

    try:
        requires_api_key = vlm_backend_requires_api_key(backend_name)
    except ValueError:
        return bool(get_env_api_key_for_backend(backend_name, api_key))
    if not requires_api_key:
        return True
    if backend_name == "nim":
        base_url = nim_base_url or base_url
        explicit_key = api_key
        if explicit_key is None and is_nvidia_provider_base_url(base_url):
            explicit_key = nvidia_api_key
        return bool(get_nim_api_key_for_base_url(base_url, explicit_key))
    if backend_name == "openai":
        return bool(get_openai_api_key_for_base_url(base_url, api_key))
    if backend_name in ("anthropic", "gemini"):
        return bool(get_env_api_key_for_backend(backend_name, api_key))
    return bool(get_env_api_key_for_backend(backend_name, api_key))


def _image_gen_backend_has_credentials(
    backend: str | None,
    *,
    api_key: str | None = None,
    nvidia_api_key: str | None,
    base_url: str | None = None,
) -> bool:
    """Check image-generation credentials using constructor-equivalent rules."""
    backend_name = (backend or "").lower()

    if not backend_name:
        return True
    import world_understanding.functions.models.backends  # noqa: F401
    from world_understanding.functions.models.backends.registry import (
        image_gen_backend_requires_api_key,
    )

    try:
        requires_api_key = image_gen_backend_requires_api_key(backend_name)
    except ValueError:
        return bool(get_env_api_key_for_backend(backend_name, api_key))
    if not requires_api_key:
        return True
    if backend_name == "openai":
        return bool(get_openai_api_key_for_base_url(base_url, api_key))
    if backend_name == "nim":
        explicit_key = api_key
        if (
            explicit_key is None
            and nvidia_api_key
            and is_nvidia_provider_base_url(base_url)
        ):
            explicit_key = nvidia_api_key
        return bool(get_nim_api_key_for_base_url(base_url, explicit_key))
    return bool(get_env_api_key_for_backend(backend_name, api_key))


@dataclass
class MaterialLibrary:
    """Data for a single material library discovered on disk."""

    id: str
    name: str
    yaml_path: str
    library_path: str  # Absolute path to the USD library file
    entries: list[dict[str, str]] = field(default_factory=list)
    icons: dict[str, str] = field(
        default_factory=dict
    )  # material name -> icon rel path
    base_dir: str = ""  # Directory containing the YAML (for serving icons)


class ServiceConfig(BaseSettings):
    """Service configuration - FastAPI-specific settings only.

    All VLM/LLM/rendering defaults come from material_agent.api.defaults.
    This class only contains settings specific to the FastAPI service layer.
    """

    # Service info
    service_name: str = "Material Agent Service"
    service_version: str = __version__
    api_version: str = "v1"
    description: str | None = None

    # Materials configuration (service-specific for HTTP icon serving)
    materials_library_path: str = "/app/materials/materials_libs_v2.usd"
    materials_config_path: str = "materials/default/materials.yaml"
    materials: list[dict[str, str]] = []  # Loaded from materials_config_path
    material_icons: dict[str, str] = {}  # Map material name -> icon path

    # Multi-library support
    material_libraries: dict[str, MaterialLibrary] = {}
    default_library_id: str = "default"

    # SimReady material libraries
    simready_enabled: bool = False
    simready_release_tag: str = DEFAULT_SIMREADY_RELEASE_TAG
    simready_manifest_path: str | None = None
    simready_cache_dir: str = "/var/material-agent/simready-cache"
    simready_allowed_categories: str | None = None
    simready_split_archives_enabled: bool = False

    # Session settings (FastAPI-specific)
    session_storage_path: str = "/var/material-agent/sessions"
    session_ttl_hours: int = 24

    # Session cleanup settings
    cleanup_interval_hours: float = 1.0  # How often to run cleanup (hours)
    cleanup_max_age_hours: float = 24.0  # Max age before local cache cleanup
    cleanup_enabled: bool = True  # Enable/disable background cleanup

    # File upload settings (FastAPI-specific)
    max_upload_size_mb: int = 500
    allowed_extensions: set[str] = {".usd", ".usda", ".usdc", ".usdz"}
    max_render_num_workers: int = Field(
        default=32,
        ge=1,
        description="Maximum accepted render_num_workers override per pipeline",
    )
    max_scene_workers: int = Field(
        default=4,
        ge=1,
        description="Maximum accepted large-scene asset worker count per pipeline",
    )
    max_scene_vlm_concurrency: int = Field(
        default=64,
        ge=1,
        description=(
            "Maximum accepted scene_workers * predict.max_workers for "
            "large-scene pipelines"
        ),
    )
    default_user_email: str = Field(
        default="anonymous@nvidia.com",
        description=(
            "Telemetry user email to use when a pipeline request omits "
            "user_email or sends it blank"
        ),
    )
    scene_render_batch_size: int = Field(
        default=16,
        ge=1,
        description=(
            "Maximum build_dataset_usd render batch size for large-scene "
            "pipelines. Smaller batches avoid long NVCF render requests for "
            "complex scenes."
        ),
    )

    # API Keys (from environment)
    nvidia_api_key: str | None = None
    nvcf_api_key: str | None = None

    # VLM/LLM settings
    vlm_backend: str = Field(
        default=DEFAULT_VLM_BACKEND, description="VLM backend to use"
    )
    vlm_model: str = Field(default=DEFAULT_VLM_MODEL, description="VLM model to use")
    vlm_base_url: str | None = Field(
        default=None,
        description="Optional VLM API base URL for non-NIM endpoint routing",
    )
    vlm_api_key: str | None = Field(
        default=None,
        description="Optional endpoint-scoped VLM API key",
    )
    vlm_api_key_env: str | None = Field(
        default=None,
        description="Environment variable containing the endpoint-scoped VLM API key",
    )
    vlm_temperature: float = Field(
        default=DEFAULT_VLM_TEMPERATURE, description="VLM temperature to use"
    )
    vlm_max_tokens: int = Field(
        default=DEFAULT_VLM_MAX_TOKENS,
        ge=1,
        description="Maximum VLM completion tokens to request",
    )
    llm_backend: str = Field(
        default=DEFAULT_LLM_BACKEND, description="LLM backend to use"
    )
    llm_model: str = Field(default=DEFAULT_LLM_MODEL, description="LLM model to use")
    llm_base_url: str | None = Field(
        default=None,
        description="Optional LLM API base URL for non-NIM endpoint routing",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="Optional endpoint-scoped LLM API key",
    )
    llm_api_key_env: str | None = Field(
        default=None,
        description="Environment variable containing the endpoint-scoped LLM API key",
    )
    llm_temperature: float = Field(
        default=DEFAULT_LLM_TEMPERATURE, description="LLM temperature to use"
    )
    llm_max_tokens: int = Field(
        default=DEFAULT_LLM_MAX_TOKENS,
        ge=1,
        description="Maximum LLM completion tokens to request",
    )
    vlm_backend_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional provider-specific VLM constructor options",
    )
    image_gen_backend: str = Field(
        default="gemini",
        description=(
            "Image generation backend for interactive reference images and "
            "generated material textures"
        ),
    )
    image_gen_model: str | None = Field(
        default=None,
        description=(
            "Optional image generation model for interactive reference images "
            "and generated material textures. If unset, the selected backend "
            "default is used."
        ),
    )
    image_gen_base_url: str | None = Field(
        default=None,
        description="Optional image generation API base URL",
    )
    image_gen_api_key: str | None = Field(
        default=None,
        description=(
            "Optional image generation API key. Use 'not-used' only for explicit "
            "no-auth local endpoints."
        ),
    )
    image_gen_api_key_env: str | None = Field(
        default=None,
        description="Environment variable containing the image generation API key",
    )
    cluster_embedding_backend: str = Field(
        default=DEFAULT_CLUSTER_EMBEDDING_BACKEND,
        description="Default embedding backend for opt-in prim clustering",
    )
    cluster_embedding_model: str = Field(
        default=DEFAULT_CLUSTER_EMBEDDING_MODEL,
        description="Default embedding model for opt-in prim clustering",
    )
    cluster_embedding_base_url: str | None = Field(
        default=None,
        description="Optional embedding API base URL for opt-in prim clustering",
    )
    cluster_embedding_api_key: str | None = Field(
        default=None,
        description=(
            "Optional API key for the prim clustering embedding endpoint. "
            "Use 'not-used' only for explicit no-auth local endpoints."
        ),
    )
    cluster_embedding_api_key_env: str | None = Field(
        default=None,
        description="Environment variable containing the clustering embedding API key",
    )
    cluster_embedding_max_workers: int = Field(
        default=DEFAULT_CLUSTER_MAX_WORKERS,
        ge=1,
        description="Default parallel embedding workers for prim clustering",
    )
    cluster_embedding_batch_size: int = Field(
        default=DEFAULT_CLUSTER_BATCH_SIZE,
        ge=1,
        description="Default embedding batch size for prim clustering",
    )
    cluster_min_prims: int = Field(
        default=DEFAULT_CLUSTER_MIN_PRIMS_TO_ACTIVATE,
        ge=1,
        description="Default minimum prim count before prim clustering runs",
    )
    cluster_max_size: int | None = Field(
        default=DEFAULT_CLUSTER_MAX_SIZE,
        ge=1,
        description="Default maximum prims propagated from one cluster representative",
    )

    class Config:
        env_prefix = "MA_"  # Environment variables prefix: MA_*
        case_sensitive = False

    def __init__(self, **kwargs: Any) -> None:
        """Initialize config and load materials/API keys."""
        super().__init__(**kwargs)

        # Load API keys from environment - try both prefixed and unprefixed
        if not self.nvidia_api_key:
            self.nvidia_api_key = os.getenv(
                "MA_NVIDIA_API_KEY", os.getenv("NVIDIA_API_KEY")
            )
        if not self.nvcf_api_key:
            self.nvcf_api_key = os.getenv("NGC_API_KEY")

        # Use local sessions directory for development if /var/ doesn't exist
        if not Path(self.session_storage_path).exists():
            local_sessions = Path(__file__).parent.parent / "sessions"
            self.session_storage_path = str(local_sessions)

        # load description from README.md file
        self.description = self._load_description()

        # Discover and load all material libraries
        self.material_libraries = self._discover_libraries()

        # Backward-compat: populate materials/material_icons/materials_library_path
        # from the default library
        default_lib = self.material_libraries.get(self.default_library_id)
        if default_lib:
            self.materials = default_lib.entries
            self.material_icons = dict(default_lib.icons)
            self.materials_library_path = default_lib.library_path
        else:
            # Fallback: use legacy single-library loading
            if not Path(self.materials_config_path).is_absolute():
                materials_config = (
                    Path(__file__).parent.parent / self.materials_config_path
                )
                self.materials_config_path = str(materials_config)

            if not Path(self.materials_library_path).exists():
                local_lib_path = (
                    Path(__file__).parent.parent
                    / "materials"
                    / "default"
                    / "materials_libs_v2.usd"
                )
                if local_lib_path.exists():
                    self.materials_library_path = str(local_lib_path)

            self.materials = self._load_materials_from_yaml(
                Path(self.materials_config_path)
            )

    def _discover_libraries(self) -> dict[str, MaterialLibrary]:
        """Scan materials/ directory for subdirectories containing materials.yaml.

        Returns:
            Dict mapping library_id -> MaterialLibrary
        """
        materials_root = Path(__file__).parent.parent / "materials"

        if not materials_root.is_dir():
            logger.warning("Materials root not found: %s", materials_root)
            return {}

        libraries: dict[str, MaterialLibrary] = {}

        for subdir in sorted(materials_root.iterdir()):
            if not subdir.is_dir():
                continue

            yaml_path = subdir / "materials.yaml"
            if not yaml_path.exists():
                continue

            library_id = subdir.name

            try:
                with open(yaml_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not isinstance(data, dict):
                    logger.warning(
                        "Invalid materials.yaml in %s: not a dict", library_id
                    )
                    continue

                # Handle both nested (materials.entries) and flat (entries) formats
                materials_section = data.get("materials", data)
                entries = materials_section.get("entries", [])
                library_path_relative = materials_section.get("library_path", "")

                if not entries or not library_path_relative:
                    logger.warning(
                        "Skipping %s: entries=%d, library_path=%s",
                        library_id,
                        len(entries) if entries else 0,
                        "set" if library_path_relative else "missing",
                    )
                    continue

                # Resolve library_path relative to the YAML file
                library_path_abs = (subdir / library_path_relative).resolve()

                # Build icon mapping
                icons: dict[str, str] = {}
                for entry in entries:
                    if "name" in entry and "icon" in entry:
                        icons[entry["name"]] = entry["icon"]

                # Generate a display name from directory name
                display_name = _display_name_from_library_id(library_id)

                lib = MaterialLibrary(
                    id=library_id,
                    name=display_name,
                    yaml_path=str(yaml_path),
                    library_path=str(library_path_abs),
                    entries=entries,
                    icons=icons,
                    base_dir=str(subdir),
                )
                libraries[library_id] = lib

                logger.info(
                    "Discovered library '%s': %d materials, USD=%s",
                    library_id,
                    len(entries),
                    library_path_abs.name,
                )

            except Exception as e:
                logger.warning("Failed to load library '%s': %s", library_id, e)

        logger.info("Total libraries discovered: %d", len(libraries))
        return libraries

    @staticmethod
    def _load_description() -> str:
        """Load description from README.md file."""
        readme_path = Path(__file__).parent / "README.md"
        if not readme_path.exists():
            return "Material Agent Service"
        with open(readme_path, encoding="utf-8") as f:
            return f.read()

    def _load_materials_from_yaml(self, materials_path: Path) -> list[dict[str, str]]:
        """Load materials from a specific YAML config file (legacy fallback).

        Returns:
            List of material dicts with name, description, binding, icon
        """
        if not materials_path.exists():
            logger.warning("Materials config not found: %s", materials_path)
            return []

        try:
            with open(materials_path, encoding="utf-8") as f:
                materials_data = yaml.safe_load(f)

            materials_section = materials_data.get("materials", {})
            materials = materials_section.get("entries", [])

            for material in materials:
                if "name" in material and "icon" in material:
                    self.material_icons[material["name"]] = material["icon"]

            logger.info(
                "Loaded %d materials with %d icons from %s",
                len(materials),
                len(self.material_icons),
                materials_path,
            )
            return cast(list[dict[str, str]], materials)

        except Exception as e:
            logger.warning("Failed to load materials: %s", e)
            return []

    def get_library(self, library_id: str) -> MaterialLibrary | None:
        """Get a material library by ID."""
        return self.material_libraries.get(library_id)

    def resolve_material_library(self, library_id: str) -> MaterialLibrary | None:
        """Resolve a material library ID, including SimReady namespace IDs."""
        if is_simready_library_id(library_id):
            return self._resolve_simready_library(library_id)
        return self.get_library(library_id)

    def _resolve_simready_library(self, library_id: str) -> MaterialLibrary:
        """Resolve a metadata-only SimReady library view."""
        if not self.simready_enabled:
            raise ValueError(
                "SimReady material libraries are disabled in this deployment."
            )

        try:
            manifest = load_manifest(self.simready_manifest_path)
            release_tag = str(manifest.get("release_tag") or "")
            if release_tag != self.simready_release_tag:
                raise SimReadyCatalogError(
                    f"SimReady manifest release tag {release_tag!r} does not "
                    f"match configured tag {self.simready_release_tag!r}"
                )
            entries = build_material_entries(
                manifest,
                library_id,
                allowed_categories=self._simready_allowed_categories(),
                split_archives_enabled=self.simready_split_archives_enabled,
            )
        except (
            SimReadyCatalogError,
            OSError,
            yaml.YAMLError,
        ) as exc:
            raise ValueError(str(exc)) from exc

        if not entries:
            raise ValueError(
                f"SimReady material library has no enabled entries: {library_id}"
            )

        display_name = _display_name_from_library_id(library_id)
        return MaterialLibrary(
            id=library_id,
            name=display_name,
            yaml_path=self.simready_manifest_path or "",
            # This is metadata-only until the lazy hydrator materializes a
            # session-local library USD before apply.
            library_path="",
            entries=entries,
            icons={},
            base_dir="",
        )

    def _simready_allowed_categories(self) -> set[str] | None:
        """Return configured SimReady category allowlist, if any."""
        raw = self.simready_allowed_categories
        if not raw:
            return None
        categories = {item.strip() for item in raw.split(",") if item.strip()}
        return categories or None

    @property
    def has_required_api_keys(self) -> bool:
        """Check if the active backend and render settings are configured."""
        vlm_nim_base_url = os.getenv("MA_VLM_NIM_BASE_URL")
        llm_nim_base_url = os.getenv("MA_LLM_NIM_BASE_URL") or vlm_nim_base_url
        vlm_backend = "nim" if vlm_nim_base_url else self.vlm_backend
        llm_backend = "nim" if llm_nim_base_url else self.llm_backend
        vlm_api_key = (
            None
            if vlm_nim_base_url
            else resolve_endpoint_api_key(
                self.vlm_api_key, self.vlm_api_key_env, prefer_env=True
            )
        )
        llm_api_key = (
            None
            if llm_nim_base_url
            else resolve_endpoint_api_key(
                self.llm_api_key, self.llm_api_key_env, prefer_env=True
            )
        )

        vlm_ready = _backend_has_credentials(
            vlm_backend,
            nvidia_api_key=self.nvidia_api_key,
            nim_base_url=vlm_nim_base_url,
            base_url=self.vlm_base_url,
            api_key=vlm_api_key,
        )
        llm_ready = _backend_has_credentials(
            llm_backend,
            nvidia_api_key=self.nvidia_api_key,
            nim_base_url=llm_nim_base_url,
            base_url=self.llm_base_url,
            api_key=llm_api_key,
        )

        render_endpoint = os.getenv("RENDER_ENDPOINT")
        render_ready = True
        if render_endpoint:
            render_ready = _is_local_render_endpoint(
                render_endpoint
            ) or _has_real_api_key(self.nvcf_api_key)
        elif os.getenv("NVCF_RENDER_FUNCTION_ID"):
            render_ready = _has_real_api_key(self.nvcf_api_key)

        return vlm_ready and llm_ready and render_ready

    @property
    def image_gen_ready(self) -> bool:
        """Check if the interactive image-generation backend is configured."""
        api_key = resolve_endpoint_api_key(
            self.image_gen_api_key,
            self.image_gen_api_key_env,
            prefer_env=True,
        )
        return _image_gen_backend_has_credentials(
            self.image_gen_backend,
            api_key=api_key,
            nvidia_api_key=self.nvidia_api_key,
            base_url=self.image_gen_base_url,
        )


# Global config instance
config = ServiceConfig()
