# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend configuration for Texture Agent Service."""

import os
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings
from texture_agent.api.defaults import DEFAULT_LLM_BACKEND, DEFAULT_LLM_MODEL
from world_understanding.utils.credentials import (
    get_env_api_key_for_backend,
    get_nim_api_key_for_base_url,
    get_openai_api_key_for_base_url,
    is_nvidia_provider_base_url,
    resolve_endpoint_api_key,
)

from .utils import get_version


def _llm_backend_has_credentials(
    backend: str | None,
    *,
    nvidia_api_key: str | None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> bool:
    backend_name = (backend or "").lower()

    if not backend_name:
        return True
    import world_understanding.functions.models.backends  # noqa: F401
    from world_understanding.functions.models.backends.registry import (
        chat_backend_requires_api_key,
    )

    try:
        requires_api_key = chat_backend_requires_api_key(backend_name)
    except ValueError:
        return bool(get_env_api_key_for_backend(backend_name, api_key))
    if not requires_api_key:
        return True
    if backend_name == "nim":
        explicit_key = api_key
        if explicit_key is None and is_nvidia_provider_base_url(base_url):
            explicit_key = nvidia_api_key
        return bool(get_nim_api_key_for_base_url(base_url, explicit_key))
    if backend_name == "openai":
        return bool(get_openai_api_key_for_base_url(base_url, api_key))
    return bool(get_env_api_key_for_backend(backend_name, api_key))


class ServiceConfig(BaseSettings):
    """Service configuration - FastAPI-specific settings only."""

    # Service info
    service_name: str = "Texture Agent Service"
    service_version: str = get_version()
    api_version: str = "v1"
    description: str | None = None

    # Session settings
    session_storage_path: str = "/var/texture-agent/sessions"
    session_ttl_hours: int = 24

    # Shared session storage settings
    storage_kind: str = "local"
    storage_s3_bucket: str | None = None
    storage_s3_prefix: str = ""
    storage_s3_region: str | None = None
    storage_s3_profile: str | None = None
    storage_s3_endpoint_url: str | None = None
    storage_s3_access_key_id: SecretStr | None = None
    storage_s3_secret_access_key: SecretStr | None = None
    storage_s3_session_token: SecretStr | None = None
    storage_s3_use_path_style: bool = True
    storage_s3_create_bucket: bool = False
    storage_s3_presign: bool = True
    storage_s3_sessions_cache_ttl: int = 5
    storage_s3_max_pool_connections: int = 64

    # File upload settings
    max_upload_size_mb: int = 500
    allowed_extensions: set[str] = {
        ".usd",
        ".usda",
        ".usdc",
        ".usdz",
        ".yaml",
        ".yml",
    }

    # Exact bucket names allowed for client-supplied s3_uri inputs. Empty is a
    # deliberate fail-closed default; configure TA_S3_ALLOWED_BUCKETS to opt in.
    s3_allowed_buckets: str = ""

    cancel_drain_timeout_seconds: float = Field(
        default=30.0,
        description=(
            "Maximum seconds to keep a cancelled asyncio task waiting for its "
            "synchronous worker thread to return before marking the session "
            "failed and preserving a stalled-worker deletion guard."
        ),
    )

    # API Keys
    nvidia_api_key: str | None = None

    # Texture generation defaults
    texture_backend: str = Field(
        default="simple_image_gen", description="Texture generation backend"
    )
    texture_endpoint: str | None = Field(
        default=None,
        description=(
            "Default Texture Variation API endpoint when texture_backend is "
            "`service`. Requests can still override this per run."
        ),
    )
    backend_engine: str | None = Field(
        default=None,
        description=(
            "Default Texture Variation API engine/model hint when "
            "texture_backend is `service`."
        ),
    )
    simple_texture_endpoint: str | None = Field(
        default=None,
        description=(
            "Optional Texture Variation API endpoint for request-time "
            "`texture_backend=simple_image_gen` routing. When set, simple "
            "requests are served by this sidecar endpoint instead of the "
            "in-process image generator path."
        ),
    )
    simple_backend_engine: str | None = Field(
        default="simple_image_gen",
        description="Engine/model hint used for the simple image-gen sidecar.",
    )
    simple_texture_workers: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Texture-agent worker count for simple sidecar requests. None "
            "keeps the global texture_workers value."
        ),
    )
    simple_texture_job_timeout_sec: int | None = Field(
        default=3600,
        ge=1,
        description=(
            "Texture generation job timeout for simple sidecar requests. None "
            "keeps the global texture_job_timeout_sec value."
        ),
    )
    simple_uv_scope: str = Field(
        default="stage",
        description=(
            "UV projection scope used when a request selects the simple "
            "image-gen backend and does not provide uv_scope."
        ),
    )
    simple_uv_rebake_source_albedo: bool = Field(
        default=False,
        description=(
            "Scoped source albedo rebake default used when a request selects "
            "the simple image-gen backend."
        ),
    )
    simple_uv_rebake_size: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional source texture rebake size for simple image-gen requests."
        ),
    )
    image_gen_backend: str = Field(
        default="nim",
        description=(
            "Image generation backend. Default `nim` points at NVIDIA's "
            "hosted FLUX.2 Klein 4B at build.nvidia.com and uses "
            "NVIDIA_API_KEY. The docker-compose image-gen overlay flips "
            "this to `openai` with a base_url override to route through "
            "a locally-hosted FLUX.2 NIM container."
        ),
    )
    image_gen_model: str | None = Field(
        default=None, description="Image generation model"
    )
    image_gen_base_url: str | None = Field(
        default=None,
        description=(
            "Override base URL for the image-gen backend. Used to point at "
            "a locally-hosted NIM container (OpenAI-compatible endpoint, "
            "e.g. http://image-gen-nim:8000/v1). None = use the backend's "
            "default."
        ),
    )
    image_gen_api_key: str | None = Field(
        default=None,
        description="Optional endpoint-scoped image generation API key",
    )
    image_gen_api_key_env: str | None = Field(
        default=None,
        description=(
            "Environment variable containing the endpoint-scoped image generation API key"
        ),
    )
    llm_backend: str = Field(
        default=DEFAULT_LLM_BACKEND,
        description=(
            "Chat LLM backend used by auto-prompt generation for materials "
            "without an explicit prompt. Falls back to a templated "
            "user_prompt + material name when the backend is unavailable."
        ),
    )
    llm_model: str | None = Field(
        default=DEFAULT_LLM_MODEL,
        description="Chat LLM model name (backend-specific).",
    )
    llm_base_url: str | None = Field(
        default=None,
        description=(
            "Override base URL for the chat LLM backend. Used to route "
            "auto-prompt generation through a locally-hosted NIM container "
            "(e.g. http://llm-nim:8000/v1). None = use the backend's default."
        ),
    )
    llm_api_key: str | None = Field(
        default=None,
        description="Optional endpoint-scoped LLM API key",
    )
    llm_api_key_env: str | None = Field(
        default=None,
        description="Environment variable containing the endpoint-scoped LLM API key",
    )
    auto_prompt_max_generated_materials: int = Field(
        default=64,
        ge=0,
        description=(
            "Maximum number of missing discovered materials the service may "
            "auto-prompt in one unscoped request. 0 disables the guard."
        ),
    )
    max_texture_units: int = Field(
        default=64,
        ge=0,
        description=(
            "Maximum number of expanded texture generation units in one run. "
            "This guards backend job fan-out after per-material/per-prim "
            "expansion. 0 disables the guard."
        ),
    )
    texture_plan_default_cap: int = Field(
        default=32,
        ge=32,
        le=32,
        description="Immutable v1 default selected-unit cap for generic backends.",
    )
    texture_plan_uv_aware_default_cap: int = Field(
        default=16,
        ge=16,
        le=16,
        description="Immutable v1 default selected-unit cap for UV-aware backends.",
    )
    texture_plan_hard_cap: int = Field(
        default=64,
        ge=64,
        le=64,
        description="Immutable v1 hard maximum for a bounded texture plan.",
    )
    texture_size: int = Field(default=1024, description="Texture resolution")
    texture_workers: int = Field(
        default=4, description="Parallel texture generation workers"
    )
    texture_job_timeout_sec: int = Field(
        default=3600,
        description=(
            "Maximum seconds to wait for each service-backed texture generation "
            "job before marking that material failed."
        ),
    )
    blend_opacity: float = Field(
        default=0.85, description="Default blend opacity (0-1)"
    )
    uv_policy: str = Field(
        default="generate_missing",
        description="Default UV policy for service-created texture pipelines",
    )
    uv_scope: str = Field(
        default="stage",
        description=(
            "Default UV projection scope for service-created texture pipelines "
            "('stage' or 'target_prims')."
        ),
    )
    uv_backend: str = Field(
        default="python",
        description="Default UV preparation backend for service-created pipelines",
    )
    uv_projection: str = Field(
        default="box",
        description="Default UV projection mode for service-created pipelines",
    )
    uv_overwrite_existing: bool = Field(
        default=False,
        description="Overwrite existing UVs during service-created UV projection",
    )
    uv_rebake_source_albedo: bool = Field(
        default=False,
        description=(
            "Rebake source albedo/normal/ORM maps after scoped UV projection so "
            "service backends receive texture maps in the generated UV layout."
        ),
    )
    uv_rebake_size: int | None = Field(
        default=None,
        description="Optional source texture rebake resolution for scoped UV projection",
    )
    uv_normalize_out_of_range: bool = Field(
        default=False,
        description="Normalize out-of-range UVs during service-created UV prep",
    )
    render_previews_enabled: bool = Field(
        default=False,
        description=(
            "Enable material preview rendering in service-created pipelines. "
            "Default is false because the base service compose does not start "
            "a render backend."
        ),
    )
    render_enabled: bool = Field(
        default=False,
        description=(
            "Enable final USD rendering in service-created pipelines. Sidecar "
            "compose packages set this true when RENDER_ENDPOINT is wired."
        ),
    )
    render_preview_image_width: int = Field(
        default=512, description="Material preview render width"
    )
    render_preview_image_height: int = Field(
        default=512, description="Material preview render height"
    )
    render_image_width: int = Field(default=1024, description="Final render width")
    render_image_height: int = Field(default=1024, description="Final render height")
    render_timeout_sec: int | None = Field(
        default=None,
        gt=0,
        description="Optional final render request timeout in seconds",
    )

    class Config:
        env_prefix = "TA_"
        case_sensitive = False

    def __init__(self, **kwargs):
        """Initialize config and load API keys."""
        super().__init__(**kwargs)

        # Load API keys from environment - try both prefixed and unprefixed
        if not self.nvidia_api_key:
            self.nvidia_api_key = os.getenv(
                "TA_NVIDIA_API_KEY", os.getenv("NVIDIA_API_KEY")
            )
        if not self.storage_s3_bucket:
            self.storage_s3_bucket = os.getenv("WU_S3_BUCKET")
        if not self.storage_s3_region:
            self.storage_s3_region = os.getenv("WU_S3_REGION")
        if not self.storage_s3_profile:
            self.storage_s3_profile = os.getenv("WU_S3_PROFILE")

        # Use local sessions directory for development if /var/ doesn't exist
        if not Path(self.session_storage_path).exists():
            local_sessions = Path(__file__).parent.parent / "sessions"
            self.session_storage_path = str(local_sessions)

        # Load description from README.md
        self.description = self._load_description()

    @staticmethod
    def _load_description() -> str:
        """Load description from README.md file."""
        readme_path = Path(__file__).parent.parent / "README.md"
        if readme_path.exists():
            with open(readme_path, encoding="utf-8") as f:
                return f.read()
        return "Texture Agent REST API Service"

    @staticmethod
    def _secret_value(secret: SecretStr | None) -> str | None:
        return secret.get_secret_value() if secret is not None else None

    @property
    def llm_ready(self) -> bool:
        """Check if the auto-prompt LLM backend has required credentials."""
        api_key = resolve_endpoint_api_key(
            self.llm_api_key,
            self.llm_api_key_env,
            prefer_env=True,
        )
        return _llm_backend_has_credentials(
            self.llm_backend,
            nvidia_api_key=self.nvidia_api_key,
            base_url=self.llm_base_url,
            api_key=api_key,
        )

    @property
    def has_required_api_keys(self) -> bool:
        """Check if service-owned model defaults are ready for use."""
        return self.llm_ready

    def build_session_store(self):
        """Build the configured session storage backend."""
        from .storage import LocalSessionStore, S3SessionStore, StorageConfig

        if self.storage_kind == "s3":
            storage_cfg = StorageConfig(
                kind=self.storage_kind,
                s3_bucket=self.storage_s3_bucket,
                s3_prefix=self.storage_s3_prefix,
                s3_region=self.storage_s3_region,
                s3_profile=self.storage_s3_profile,
                s3_endpoint_url=self.storage_s3_endpoint_url,
                s3_access_key_id=self._secret_value(self.storage_s3_access_key_id),
                s3_secret_access_key=self._secret_value(
                    self.storage_s3_secret_access_key
                ),
                s3_session_token=self._secret_value(self.storage_s3_session_token),
                s3_use_path_style=self.storage_s3_use_path_style,
                s3_create_bucket=self.storage_s3_create_bucket,
                s3_presign=self.storage_s3_presign,
                s3_sessions_cache_ttl=self.storage_s3_sessions_cache_ttl,
                s3_max_pool_connections=self.storage_s3_max_pool_connections,
            )
            return S3SessionStore.from_config(storage_cfg)

        return LocalSessionStore(root_dir=self.session_storage_path)


# Global config instance
config = ServiceConfig()
