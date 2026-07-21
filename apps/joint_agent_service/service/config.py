# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend configuration for Joint Agent Service.

Delegates VLM/LLM defaults to joint_agent.api.defaults.
Only FastAPI-specific service settings are defined here.
"""

import os
from pathlib import Path

from joint_agent.api.defaults import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MODEL,
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_MAX_WORKERS,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_TEMPERATURE,
)
from pydantic import Field
from pydantic_settings import BaseSettings

from . import __version__


class ServiceConfig(BaseSettings):
    """Service configuration - FastAPI-specific settings only.

    All VLM/LLM/rendering defaults come from joint_agent.api.defaults.
    """

    # Service info
    service_name: str = "Joint Agent Service"
    service_version: str = __version__
    api_version: str = "v1"
    description: str | None = None

    # Session settings
    session_storage_path: str = "/var/joint-agent/sessions"
    session_ttl_hours: int = 24
    run_claim_lease_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Cross-instance pipeline ownership lease duration",
    )
    run_claim_heartbeat_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Pipeline ownership lease renewal interval",
    )

    # File upload settings
    max_upload_size_mb: int = Field(
        default=0,
        ge=0,
        description="Optional upload limit in MiB; 0 disables the limit",
    )
    allowed_extensions: set[str] = {".usd", ".usda", ".usdc", ".usdz", ".yaml", ".yml"}

    # Exact bucket names allowed for client-supplied s3_uri inputs. Empty is a
    # deliberate fail-closed default; configure JA_S3_ALLOWED_BUCKETS to opt in.
    s3_allowed_buckets: str = ""

    # API Keys
    nvidia_api_key: str | None = None

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
    vlm_temperature: float = Field(
        default=DEFAULT_VLM_TEMPERATURE, description="VLM temperature to use"
    )
    vlm_max_workers: int = Field(
        default=DEFAULT_VLM_MAX_WORKERS,
        ge=1,
        description="Maximum parallel prediction VLM requests per pipeline",
    )
    llm_backend: str = Field(
        default=DEFAULT_LLM_BACKEND,
        description="LLM backend used by analyze_structure (hierarchy naming)",
    )
    llm_model: str = Field(
        default=DEFAULT_LLM_MODEL, description="LLM model used by analyze_structure"
    )

    class Config:
        env_prefix = "JA_"
        case_sensitive = False

    def __init__(self, **kwargs):
        """Initialize config and load API keys."""
        super().__init__(**kwargs)

        # Load API keys from environment - try both prefixed and unprefixed
        if not self.nvidia_api_key:
            self.nvidia_api_key = os.getenv(
                "JA_NVIDIA_API_KEY", os.getenv("NVIDIA_API_KEY")
            )
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
        return "Joint Agent REST API Service"

    @property
    def has_required_api_keys(self) -> bool:
        """Check whether the default hosted model credential is present."""
        return bool(self.nvidia_api_key)

    def build_session_store(self):
        """Build a SessionStore from config."""
        from .storage import LocalSessionStore, S3SessionStore, StorageConfig

        if self.storage_kind == "s3":
            if not self.storage_s3_bucket.strip():
                raise ValueError(
                    "JA_STORAGE_S3_BUCKET is required when JA_STORAGE_KIND=s3"
                )
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
