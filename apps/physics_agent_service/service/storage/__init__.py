# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable session storage backends (local filesystem and S3)."""

from .base import (
    METADATA_KEY,
    CompletedSessionSnapshot,
    SessionGeneration,
    SessionGenerationConflictError,
    SessionGenerationOwnershipError,
    SessionNotCompletedError,
    SessionStore,
)
from .config import StorageConfig
from .local_store import LocalSessionStore
from .s3_store import S3SessionStore

__all__ = [
    "CompletedSessionSnapshot",
    "METADATA_KEY",
    "LocalSessionStore",
    "S3SessionStore",
    "SessionGeneration",
    "SessionGenerationConflictError",
    "SessionGenerationOwnershipError",
    "SessionNotCompletedError",
    "SessionStore",
    "StorageConfig",
]
