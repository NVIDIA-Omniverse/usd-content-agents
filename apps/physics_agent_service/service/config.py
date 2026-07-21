# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend configuration for Physics Agent Service.

Delegates VLM/LLM defaults to physics_agent.api.defaults.
Only FastAPI-specific service settings are defined here.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

from physics_agent import __version__
from physics_agent.api.defaults import (
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_TEMPERATURE,
)
from pydantic import Field
from pydantic_settings import BaseSettings
from world_understanding.utils.credentials import (
    get_env_api_key_for_backend,
    get_nim_api_key_for_base_url,
    get_openai_api_key_for_base_url,
    get_vlm_nim_env_base_url_override,
    is_nvidia_provider_base_url,
    resolve_endpoint_api_key,
)

_LOCAL_RENDER_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "ovrtx-rendering-api",
    "physics-ovrtx-rendering-api",
}


def _is_local_render_endpoint(endpoint: str | None) -> bool:
    """Return True when the render endpoint targets a local renderer."""
    if not endpoint:
        return False

    host = urlparse(endpoint).hostname or endpoint
    return host.lower() in _LOCAL_RENDER_HOSTS


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
        # An unregistered provider is not service-ready unless an explicit
        # credential was supplied.  The eventual model construction reports
        # the more specific registration error.
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
    if backend_name == "anthropic":
        return bool(get_env_api_key_for_backend(backend_name, api_key))
    if backend_name == "gemini":
        return bool(get_env_api_key_for_backend(backend_name, api_key))
    return bool(get_env_api_key_for_backend(backend_name, api_key))


class ServiceConfig(BaseSettings):
    """Service configuration - FastAPI-specific settings only.

    All VLM/LLM/rendering defaults come from physics_agent.api.defaults.
    """

    # Service info
    service_name: str = "Physics Agent Service"
    service_version: str = __version__
    api_version: str = "v1"
    description: str | None = None

    # Session settings
    session_storage_path: str = "/var/physics-agent/sessions"
    session_ttl_hours: int = 24
    cleanup_interval_hours: float = 1.0
    cleanup_max_age_hours: float = 24.0
    cleanup_enabled: bool = True

    # File upload settings
    max_upload_size_mb: int = 500
    allowed_extensions: set[str] = {".usd", ".usda", ".usdc", ".usdz", ".yaml", ".yml"}

    # Mode-A dataset_path allowlist. The /predict route treats dataset_path as
    # a privileged server-side import; an explicit allowlist of roots prevents
    # arbitrary local-file reads. The session_storage_path is always allowed.
    # Colon-separated string via env: PA_DATASET_ALLOWED_ROOTS.
    dataset_allowed_roots: str = ""

    # Exact bucket names allowed for client-supplied s3_uri inputs. Empty is a
    # deliberate fail-closed default; configure PA_S3_ALLOWED_BUCKETS to opt in.
    s3_allowed_buckets: str = ""

    # API Keys
    nvidia_api_key: str | None = None
    nvcf_api_key: str | None = None

    # Storage backend (local or s3 for multi-instance)
    storage_kind: str = "local"
    storage_s3_bucket: str = ""
    storage_s3_prefix: str = ""
    storage_s3_region: str = "us-east-2"
    storage_s3_endpoint_url: str | None = None
    storage_s3_access_key_id: str | None = None
    storage_s3_secret_access_key: str | None = None
    storage_s3_session_token: str | None = None
    storage_s3_use_path_style: bool = True
    storage_s3_create_bucket: bool = False
    storage_s3_presign: bool = True
    storage_s3_sessions_cache_ttl: int = 5

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
    vlm_backend_options: dict[str, object] = Field(
        default_factory=dict,
        description="Optional provider-specific VLM constructor options",
    )

    class Config:
        env_prefix = "PA_"
        case_sensitive = False

    def __init__(self, **kwargs):
        """Initialize config and load API keys."""
        super().__init__(**kwargs)

        # Load API keys from environment - try both prefixed and unprefixed
        if not self.nvidia_api_key:
            self.nvidia_api_key = os.getenv(
                "PA_NVIDIA_API_KEY", os.getenv("NVIDIA_API_KEY")
            )
        if not self.nvcf_api_key:
            self.nvcf_api_key = os.getenv("NGC_API_KEY")

        # Use local sessions directory for development if /var/ doesn't exist
        if not Path(self.session_storage_path).exists():
            local_sessions = Path(__file__).parent.parent / "sessions"
            self.session_storage_path = str(local_sessions)

        # Load description from README.md
        self.description = self._load_description()

    @staticmethod
    def _load_description() -> str:
        """Load description from README.md file."""
        readme_path = Path(__file__).parent / "README.md"
        if readme_path.exists():
            with open(readme_path, encoding="utf-8") as f:
                return f.read()
        return "Physics Agent REST API Service"

    @property
    def has_required_api_keys(self) -> bool:
        """Check if the active backend and render settings are configured."""
        vlm_nim_base_url = get_vlm_nim_env_base_url_override()
        effective_vlm_backend = "nim" if vlm_nim_base_url else self.vlm_backend
        vlm_api_key = (
            None
            if vlm_nim_base_url
            else resolve_endpoint_api_key(
                self.vlm_api_key, self.vlm_api_key_env, prefer_env=True
            )
        )
        vlm_ready = _backend_has_credentials(
            effective_vlm_backend,
            nvidia_api_key=self.nvidia_api_key,
            nim_base_url=vlm_nim_base_url,
            base_url=self.vlm_base_url,
            api_key=vlm_api_key,
        )

        render_ready = True
        if os.getenv("PA_RENDER_BACKEND", "remote").lower() == "remote":
            render_endpoint = os.getenv("RENDER_ENDPOINT")
            if render_endpoint:
                render_ready = _is_local_render_endpoint(render_endpoint) or bool(
                    self.nvcf_api_key
                )
            elif os.getenv("NVCF_RENDER_FUNCTION_ID"):
                render_ready = bool(self.nvcf_api_key)

        return vlm_ready and render_ready

    def build_session_store(self):
        """Build a SessionStore from config."""
        from .storage import LocalSessionStore, S3SessionStore, StorageConfig

        if self.storage_kind == "s3":
            storage_cfg = StorageConfig(
                kind="s3",
                s3_bucket=self.storage_s3_bucket,
                s3_prefix=self.storage_s3_prefix,
                s3_region=self.storage_s3_region,
                s3_endpoint_url=self.storage_s3_endpoint_url,
                s3_access_key_id=self.storage_s3_access_key_id,
                s3_secret_access_key=self.storage_s3_secret_access_key,
                s3_session_token=self.storage_s3_session_token,
                s3_use_path_style=self.storage_s3_use_path_style,
                s3_create_bucket=self.storage_s3_create_bucket,
                s3_presign=self.storage_s3_presign,
                s3_sessions_cache_ttl=self.storage_s3_sessions_cache_ttl,
            )
            return S3SessionStore.from_config(storage_cfg)

        return LocalSessionStore(root_dir=self.session_storage_path)


# Global config instance
config = ServiceConfig()
