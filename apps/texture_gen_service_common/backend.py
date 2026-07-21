# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend interface for Texture Variation API services."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .models import BackendCapabilities, CreateJobRequest, GenerationResult


class TextureGenerationBackendError(RuntimeError):
    """Backend failure that still carries a normalized partial result."""

    def __init__(self, message: str, *, result: GenerationResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class BackendHealth(BaseModel):
    """Backend runtime health."""

    status: str = "healthy"
    ready: bool = True
    warmup_complete: bool | None = None
    gpu_available: bool | None = None
    capabilities: BackendCapabilities | dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class TextureGenerationBackend(ABC):
    """Backend plugin contract for texture variation generation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...  # pragma: no cover - abstract contract placeholder

    def health(self) -> BackendHealth:
        """Return current backend health.

        Backends with lazy model loading should report ``ready=False`` until
        warm-up has completed successfully.
        """
        return BackendHealth(capabilities=self.capabilities())

    def capabilities(self) -> BackendCapabilities | dict[str, Any]:
        """Return backend capability metadata."""
        return {}

    @abstractmethod
    def generate(
        self,
        request: CreateJobRequest,
        *,
        job_id: str,
        output_dir: Path,
        cancel_event: threading.Event,
    ) -> GenerationResult:
        """Generate texture maps for one Texture Variation API request."""
        ...  # pragma: no cover - abstract contract placeholder
