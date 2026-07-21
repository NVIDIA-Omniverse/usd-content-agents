# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Texture Agent FastAPI Service - Main Application."""

import asyncio
import io
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from world_understanding.utils.logging import setup_logging
from world_understanding.utils.public_response import (
    PublicJsonResponseSanitizationMiddleware,
)

from .utils import AccessLogFilter

# Add parent directories to Python path for texture_agent imports
service_dir = Path(__file__).parent.parent
apps_dir = service_dir.parent
repo_root = apps_dir.parent

for path in [str(apps_dir), str(repo_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Load .env file BEFORE importing config
load_dotenv()

from .config import config  # noqa: E402
from .routers import (  # noqa: E402
    artifacts_router,
    pipeline_router,
    sessions_router,
)
from .runtime.registry import get_job_registry  # noqa: E402
from .session.manager import SessionManager  # noqa: E402

# Setup logging from config
setup_logging()

if sys.platform == "win32":  # pragma: no cover - Windows-only import-time setup
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def _get_max_active_sessions() -> int:
    """Return the capacity enforced by the process-wide job registry."""
    return get_job_registry().max_concurrent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting Texture Agent Service...")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(AccessLogFilter())

    # Check for NVIDIA API key
    if not config.nvidia_api_key:
        logger.warning(
            "NVIDIA_API_KEY not set. Image generation and NVCF rendering "
            "may be unavailable."
        )
    if not config.has_required_api_keys:
        logger.warning(
            "Required API keys are not configured for the active texture LLM "
            "backend. Auto-prompt generation may fall back or fail."
        )

    store = config.build_session_store()
    logger.info(
        "Initializing session storage: kind=%s, local_path=%s",
        store.kind,
        config.session_storage_path,
    )
    if store.kind == "s3":
        logger.info(
            "S3 storage: bucket=%s, prefix=%s",
            config.storage_s3_bucket,
            config.storage_s3_prefix,
        )

    # Initialize session manager
    session_mgr = SessionManager(
        storage_path=config.session_storage_path,
        ttl_hours=config.session_ttl_hours,
        store=store,
    )

    # Set global session manager in all routers
    pipeline_router.set_session_manager(session_mgr)
    artifacts_router.set_session_manager(session_mgr)
    sessions_router.set_session_manager(session_mgr)

    # Initialize event bus with the shared session manager
    from .runtime.bus import init_event_bus

    init_event_bus(session_mgr)

    # Start periodic session cleanup (every 30 minutes)
    from .runtime.bus import get_event_bus

    async def _cleanup_loop():  # pragma: no cover - periodic service loop
        while True:
            await asyncio.sleep(30 * 60)
            try:
                cleaned = await asyncio.to_thread(session_mgr.cleanup_expired_sessions)
                # Release the in-memory bus snapshot/queue for each session
                # whose disk dir we just removed. Without this, per-session
                # bus state accumulates indefinitely in long-running services
                # because TTL deletion bypasses the HTTP DELETE handler.
                for session_id in cleaned:
                    await get_event_bus().cleanup_session(session_id)
                orphaned = await get_event_bus().cleanup_orphaned_sessions()
                if cleaned:
                    logger.info(
                        "Periodic cleanup removed %d expired sessions", len(cleaned)
                    )
                if orphaned:
                    logger.info(
                        "Periodic cleanup pruned %d orphaned bus sessions",
                        len(orphaned),
                    )
            except Exception as e:
                logger.warning("Periodic session cleanup failed: %s", e)

    cleanup_task = asyncio.create_task(_cleanup_loop())

    logger.info("Service started: %s v%s", config.service_name, config.service_version)
    logger.info(
        "Texture backend: %s (image_gen: %s)",
        config.texture_backend,
        config.image_gen_backend,
    )
    logger.info(
        "Texture size: %d, workers: %d, blend opacity: %.2f",
        config.texture_size,
        config.texture_workers,
        config.blend_opacity,
    )

    nvidia_status = "configured" if config.nvidia_api_key else "NOT SET"
    logger.info("NVIDIA API key: %s", nvidia_status)

    yield

    # Shutdown
    cleanup_task.cancel()
    logger.info("Shutting down Texture Agent Service...")


# Create FastAPI app
app = FastAPI(
    title=config.service_name,
    description=config.description,
    version=config.service_version,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    PublicJsonResponseSanitizationMiddleware,
    session_roots=(config.session_storage_path,),
)

# Include routers
app.include_router(pipeline_router.router)
app.include_router(artifacts_router.router)
app.include_router(sessions_router.router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Report the key that matches the currently-configured image_gen_backend
    # so callers can tell at a glance whether the active backend is wired
    # up. We still expose the raw nvidia flag for backward compatibility.
    backend_key_env = {
        "nim": "NVIDIA_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }.get(config.image_gen_backend)
    active_backend_key_configured = (
        bool(os.environ.get(backend_key_env)) if backend_key_env else None
    )
    return {
        "status": "healthy",
        "service": config.service_name,
        "version": config.service_version,
        "image_gen_backend": config.image_gen_backend,
        "active_backend_key_configured": active_backend_key_configured,
        "llm_backend": config.llm_backend,
        "llm_ready": config.llm_ready,
        "llm_api_keys_configured": config.has_required_api_keys,
        "nvidia_api_key_configured": bool(config.nvidia_api_key),
        "max_active_sessions": _get_max_active_sessions(),
    }


@app.get("/api")
async def root_api_info():
    """Root endpoint with service info."""
    return {
        "service": config.service_name,
        "version": config.service_version,
        "docs": "/docs",
        "health": "/health",
        "api": {
            "pipeline": {
                "upload_usd": "POST /pipeline/upload-usd",
                "create": "POST /pipeline",
                "plan": "GET /pipeline/{session_id}/plan",
                "status": "GET /pipeline/{session_id}/status",
                "results": "GET /pipeline/{session_id}/results",
                "cancel": "POST /pipeline/{session_id}/cancel",
                "events": "GET /pipeline/{session_id}/events",
                "regenerate": "POST /pipeline/{session_id}/regenerate",
                "event_log": "GET /pipeline/{session_id}/event-log",
            },
            "artifacts": {
                "materials": "GET /artifacts/{session_id}/materials",
                "textures": "GET /artifacts/{session_id}/textures",
                "textures_file": "GET /artifacts/{session_id}/textures/{filename}",
                "output": "GET /artifacts/{session_id}/output",
                "renders": "GET /artifacts/{session_id}/renders",
                "renders_file": "GET /artifacts/{session_id}/renders/{filename}",
            },
            "sessions": {
                "list": "GET /sessions",
                "get": "GET /sessions/{session_id}",
                "delete": "DELETE /sessions/{session_id}",
            },
        },
    }


@app.get("/")
async def root():
    """Root endpoint redirects to API info."""
    return await root_api_info()


def main():
    """Entry point for running the service."""
    import uvicorn

    uvicorn.run(
        "service.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
