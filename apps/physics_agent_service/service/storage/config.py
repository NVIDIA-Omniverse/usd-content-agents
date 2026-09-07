# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment value, treating an empty value as absent."""
    return int(os.getenv(name) or str(default))


@dataclass
class StorageConfig:
    kind: str = os.getenv("PA_STORAGE_KIND", "local")  # local | s3
    # Local
    local_root: str = os.getenv("PA_STORAGE_LOCAL_ROOT", "sessions")
    # S3
    s3_bucket: str | None = os.getenv("PA_STORAGE_S3_BUCKET")
    s3_prefix: str = os.getenv("PA_STORAGE_S3_PREFIX", "")
    s3_region: str | None = os.getenv("PA_STORAGE_S3_REGION")
    s3_endpoint_url: str | None = os.getenv("PA_STORAGE_S3_ENDPOINT_URL")
    s3_access_key_id: str | None = os.getenv("PA_STORAGE_S3_ACCESS_KEY_ID")
    s3_secret_access_key: str | None = os.getenv("PA_STORAGE_S3_SECRET_ACCESS_KEY")
    s3_session_token: str | None = os.getenv("PA_STORAGE_S3_SESSION_TOKEN")
    s3_use_path_style: bool = (
        os.getenv("PA_STORAGE_S3_USE_PATH_STYLE", "true").lower() == "true"
    )
    s3_create_bucket: bool = (
        os.getenv("PA_STORAGE_S3_CREATE_BUCKET", "false").lower() == "true"
    )
    s3_presign: bool = os.getenv("PA_STORAGE_S3_PRESIGN", "true").lower() == "true"
    s3_sessions_cache_ttl: int = _env_int("PA_STORAGE_S3_SESSIONS_CACHE_TTL", 5)
    s3_generation_retention: int = _env_int("PA_STORAGE_S3_GENERATION_RETENTION", 5)
