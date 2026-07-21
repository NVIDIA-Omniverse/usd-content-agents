# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@dataclass
class StorageConfig:
    kind: str = field(
        default_factory=lambda: os.getenv("TA_STORAGE_KIND", "local")
    )  # local | s3

    local_root: str = field(
        default_factory=lambda: os.getenv(
            "TA_STORAGE_LOCAL_ROOT",
            "_run/texture_sessions",
        )
    )

    s3_bucket: str | None = field(
        default_factory=lambda: _env_first("TA_STORAGE_S3_BUCKET", "WU_S3_BUCKET")
    )
    s3_prefix: str = field(
        default_factory=lambda: os.getenv("TA_STORAGE_S3_PREFIX", "")
    )
    s3_region: str | None = field(
        default_factory=lambda: _env_first("TA_STORAGE_S3_REGION", "WU_S3_REGION")
    )
    s3_profile: str | None = field(
        default_factory=lambda: _env_first("TA_STORAGE_S3_PROFILE", "WU_S3_PROFILE")
    )
    s3_endpoint_url: str | None = field(
        default_factory=lambda: os.getenv("TA_STORAGE_S3_ENDPOINT_URL")
    )
    s3_access_key_id: str | None = field(
        default_factory=lambda: os.getenv("TA_STORAGE_S3_ACCESS_KEY_ID"),
        repr=False,
    )
    s3_secret_access_key: str | None = field(
        default_factory=lambda: os.getenv("TA_STORAGE_S3_SECRET_ACCESS_KEY"),
        repr=False,
    )
    s3_session_token: str | None = field(
        default_factory=lambda: os.getenv("TA_STORAGE_S3_SESSION_TOKEN") or None,
        repr=False,
    )
    s3_use_path_style: bool = field(
        default_factory=lambda: os.getenv(
            "TA_STORAGE_S3_USE_PATH_STYLE",
            "true",
        ).lower()
        == "true"
    )
    s3_create_bucket: bool = field(
        default_factory=lambda: os.getenv(
            "TA_STORAGE_S3_CREATE_BUCKET",
            "false",
        ).lower()
        == "true"
    )
    s3_presign: bool = field(
        default_factory=lambda: os.getenv("TA_STORAGE_S3_PRESIGN", "true").lower()
        == "true"
    )
    s3_sessions_cache_ttl: int = field(
        default_factory=lambda: int(os.getenv("TA_STORAGE_S3_SESSIONS_CACHE_TTL", "5"))
    )
    s3_max_pool_connections: int = field(
        default_factory=lambda: int(
            os.getenv("TA_STORAGE_S3_MAX_POOL_CONNECTIONS", "64")
        )
    )
