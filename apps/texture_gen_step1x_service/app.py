# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI entrypoint for the optional Step1X Texture Variation API service.

Usage:
    uvicorn apps.texture_gen_step1x_service.app:app --port 8000
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from apps.texture_gen_service_common import create_app

from .backend import Step1XBackend, Step1XBackendConfig

load_dotenv()


def _output_dir() -> Path:
    configured = os.environ.get("TEXTURE_OUTPUT_DIR")
    if configured and configured.strip():
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / "texture_gen_step1x_service"


def _max_workers() -> int:
    value = os.environ.get("TEXTURE_STEP1X_MAX_WORKERS", "1")
    try:
        return max(1, int(value))
    except ValueError:
        return 1


backend = Step1XBackend(config=Step1XBackendConfig.from_env())
app = create_app(
    backend=backend,
    output_dir=_output_dir(),
    title="Texture Variation API (Step1X)",
    version="1.0.0",
    description=(
        "Optional Step1X-backed Texture Variation API service. The service "
        "expects Step1X runtime assets to be supplied externally."
    ),
    service_name="texture-gen-step1x-service",
    max_workers=_max_workers(),
)


@app.get("/livez")
async def livez() -> dict[str, str]:
    """Return cheap process liveness without runtime readiness probes."""
    return {"status": "healthy", "service": "texture-gen-step1x-service"}
