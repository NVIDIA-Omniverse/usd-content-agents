# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material Agent FastAPI Service - Main Application."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Import telemetry initialization functions
from world_understanding.telemetry import (
    TelemetryConfig,
    initialize_telemetry,
    shutdown_telemetry,
)
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)
from world_understanding.utils.logging import setup_logging
from world_understanding.utils.public_response import (
    PublicJsonResponseSanitizationMiddleware,
)

from .utils import AccessLogFilter

# Check if OpenTelemetry instrumentation is available
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor

    OTEL_INSTRUMENTATION_AVAILABLE = True
except ImportError:  # pragma: no cover
    OTEL_INSTRUMENTATION_AVAILABLE = False

# Add parent directories to Python path for material_agent imports
service_dir = Path(__file__).parent.parent
apps_dir = service_dir.parent
repo_root = apps_dir.parent

# Add paths if not already present
for path in [str(apps_dir), str(repo_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import io  # noqa: E402
import os  # noqa: E402

from dotenv import dotenv_values, load_dotenv  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

# Load .env file BEFORE importing config (which reads env vars at module load)
load_dotenv()

from world_understanding.utils.nvcf_utils import get_base_url  # noqa: E402

from .config import config  # noqa: E402
from .routers import (  # noqa: E402
    artifacts_router,
    assets_router,
    materials_router,
    pipeline_router,
    sessions_router,
)
from .runtime.bus import get_event_bus  # noqa: E402
from .runtime.registry import get_job_registry  # noqa: E402
from .session.manager import SessionManager  # noqa: E402
from .storage.config import StorageConfig  # noqa: E402
from .storage.s3_store import S3SessionStore  # noqa: E402

# Load environment variables
load_dotenv()

# setup logging from config
setup_logging()

if sys.platform == "win32":  # pragma: no cover
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


def _load_aws_config_file_into_env(*, log: logging.Logger) -> None:
    """
    Expand AWS_CONFIG_FILE contents into process environment variables.

    Supported format:
    - dotenv-style: KEY=VALUE lines (optionally prefixed with "export ")

    Only the explicitly supported AWS-related keys are considered:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_DEFAULT_REGION / AWS_REGION / region

    Existing environment variables win (we do not overwrite).
    """

    config_path_raw = os.getenv("AWS_CONFIG_FILE")
    if not config_path_raw:
        return

    config_path = Path(config_path_raw).expanduser()

    if not config_path.exists():
        log.warning("AWS_CONFIG_FILE set but file does not exist: %s", config_path)
        return

    try:
        # Use python-dotenv's parser so we don't maintain our own parsing.
        # This supports comments, quotes, and `export KEY=VALUE` lines.
        values = dotenv_values(config_path)
    except Exception:
        log.exception("Failed reading AWS_CONFIG_FILE: %s", config_path)
        return

    def _set_env_if_missing(*, env_key: str, value: str | None) -> bool:
        if not value:
            return False
        # Existing environment variables win (never overwrite).
        if os.getenv(env_key):
            return False
        os.environ[env_key] = value
        return True

    # Explicitly handle each supported variable (no loops).
    access_key_id = values.get("AWS_ACCESS_KEY_ID") or values.get("aws_access_key_id")
    secret_access_key = values.get("AWS_SECRET_ACCESS_KEY") or values.get(
        "aws_secret_access_key"
    )
    # Region can be provided in multiple common forms; normalize to
    # AWS_DEFAULT_REGION.
    region = (
        values.get("AWS_DEFAULT_REGION")
        or values.get("aws_default_region")
        or values.get("AWS_REGION")
        or values.get("aws_region")
        or values.get("region")
    )

    any_set = False
    any_set = (
        _set_env_if_missing(env_key="AWS_ACCESS_KEY_ID", value=access_key_id) or any_set
    )
    any_set = (
        _set_env_if_missing(env_key="AWS_SECRET_ACCESS_KEY", value=secret_access_key)
        or any_set
    )
    if _set_env_if_missing(env_key="AWS_DEFAULT_REGION", value=region):
        any_set = True
        # Keep AWS_REGION in sync if not already set.
        _set_env_if_missing(env_key="AWS_REGION", value=region)

    if any_set:
        log.info("AWS_CONFIG_FILE loaded.")

    # Never log secret values; only indicate presence.
    if "AWS_ACCESS_KEY_ID" in os.environ and os.environ["AWS_ACCESS_KEY_ID"]:
        log.info("AWS Access Key ID: present")
    if "AWS_SECRET_ACCESS_KEY" in os.environ and os.environ["AWS_SECRET_ACCESS_KEY"]:
        log.info("AWS Secret Access Key: present")
    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
    if region:
        log.info("AWS Region: %s", region)


async def _periodic_cleanup_task(
    manager: SessionManager,
    interval_hours: float,
    max_age_hours: float,
) -> None:
    """Background task that periodically cleans up stale sessions.

    Args:
        manager: SessionManager instance
        interval_hours: How often to run cleanup (in hours)
        max_age_hours: Max age before local cache cleanup
    """
    logger.info(
        f"Starting periodic cleanup task (interval={interval_hours}h, max_age={max_age_hours}h)"
    )

    while True:
        try:
            # Wait for the interval before first cleanup
            await asyncio.sleep(interval_hours * 3600)

            logger.info("Running periodic session cleanup...")

            # Clean up stale local cache (sync to S3 and remove)
            cleaned_cache = await manager.cleanup_stale_local_cache(
                max_age_hours=max_age_hours
            )

            # Clean up expired sessions from remote storage
            expired_sessions = await manager.cleanup_expired_sessions()

            if cleaned_cache > 0 or expired_sessions > 0:
                logger.info(
                    f"Cleanup complete: {cleaned_cache} stale cache entries, "
                    f"{expired_sessions} expired sessions"
                )

        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            break
        except Exception:
            log_durable_failure(
                logger,
                "periodic_session_cleanup_failed",
                phase=FailurePhase.ROLLBACK,
                retryable=True,
            )
            # Continue running despite errors


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting Material Agent Service...")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(AccessLogFilter())

    # Initialize telemetry (reads from env vars via TelemetryConfig)
    # Telemetry is optional - failures are logged but don't crash the app
    telemetry_config = TelemetryConfig()
    tracer_provider = initialize_telemetry(telemetry_config)
    if tracer_provider is not None:
        logger.info(
            f"Telemetry initialized: enabled={telemetry_config.enabled}, "
            f"service={telemetry_config.service_name}, "
            f"exporters={telemetry_config.exporters}"
        )
    elif telemetry_config.enabled:
        logger.warning("Telemetry enabled but failed to initialize (check logs above)")
    else:
        logger.debug("Telemetry disabled via OTEL_ENABLED=false")

    # Auto-instrument libraries if available
    if OTEL_INSTRUMENTATION_AVAILABLE:
        RequestsInstrumentor().instrument()
        logger.info("OpenTelemetry auto-instrumentation enabled for requests library")

    # Check for required API keys for the active backend/render settings
    if not config.has_required_api_keys:
        logger.warning(
            "Required API keys are not configured for the current backend or "
            "render settings. VLM=%s LLM=%s RENDER_ENDPOINT=%s",
            config.vlm_backend,
            config.llm_backend,
            os.getenv("RENDER_ENDPOINT", ""),
        )

    # Expand AWS_CONFIG_FILE into AWS* env vars (if provided)
    _load_aws_config_file_into_env(log=logger)

    # Initialize session store (local or S3) based on configuration
    storage_cfg = StorageConfig()
    session_store = None
    if storage_cfg.kind == "s3":
        logger.info(
            "Using S3 session store: bucket=%s prefix=%s endpoint=%s",
            storage_cfg.s3_bucket,
            storage_cfg.s3_prefix,
            storage_cfg.s3_endpoint_url,
        )
        session_store = S3SessionStore.from_config(storage_cfg)
    else:
        logger.info("Using local session store at: %s", storage_cfg.local_root)
        # Local store is optional - SessionManager handles local disk directly
        # Only create LocalSessionStore if we want to use its presigned URL logic
        # For now, we don't need it since local files are served via FileResponse

    # Initialize session manager with optional store backend
    logger.info("Initializing session storage at: %s", config.session_storage_path)
    session_mgr = SessionManager(
        storage_path=config.session_storage_path,
        ttl_hours=config.session_ttl_hours,
        store=session_store,
    )

    # Set global session manager in all routers
    pipeline_router.set_session_manager(session_mgr)
    artifacts_router.set_session_manager(session_mgr)
    assets_router.set_session_manager(session_mgr)
    sessions_router.set_session_manager(session_mgr)

    # Set session manager on event bus for persistence with correct storage backend
    event_bus = get_event_bus()
    event_bus.set_session_manager(session_mgr)

    # Start background cleanup task if enabled
    cleanup_task = None
    if config.cleanup_enabled:
        cleanup_task = asyncio.create_task(
            _periodic_cleanup_task(
                manager=session_mgr,
                interval_hours=config.cleanup_interval_hours,
                max_age_hours=config.cleanup_max_age_hours,
            )
        )
        logger.info(
            f"Background cleanup enabled: interval={config.cleanup_interval_hours}h, "
            f"max_age={config.cleanup_max_age_hours}h"
        )
    else:
        logger.info("Background cleanup disabled (MA_CLEANUP_ENABLED=false)")

    logger.info("Service started: %s v%s", config.service_name, config.service_version)
    logger.info(f"Materials library: {config.materials_library_path}")
    logger.info(
        f"VLM: {config.vlm_backend} / {config.vlm_model} "
        f"(temp={config.vlm_temperature})"
    )
    backend_key_status = "configured" if config.has_required_api_keys else "NOT SET"
    logger.info(f"Backend/render credentials: {backend_key_status}")

    # Log NVCF function IDs for rendering and scene optimization
    try:
        render_url = get_base_url(None, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")
        # Extract function ID from URL
        render_fn_id = render_url.replace("https://", "").replace(
            ".invocation.api.nvcf.nvidia.com", ""
        )
        logger.info(f"NVCF Render function: {render_fn_id}")
    except ValueError:
        logger.info("NVCF Render function: NOT CONFIGURED")

    try:
        optimizer_url = get_base_url(
            None, "OPTIMIZER_ENDPOINT", "NVCF_OPTIMIZER_FUNCTION_ID"
        )
        optimizer_fn_id = optimizer_url.replace("https://", "").replace(
            ".invocation.api.nvcf.nvidia.com", ""
        )
        logger.info(f"NVCF Optimizer function: {optimizer_fn_id}")
    except ValueError:
        logger.info("NVCF Optimizer function: NOT CONFIGURED")

    yield

    # Shutdown
    logger.info("Shutting down Material Agent Service...")

    # Cancel background cleanup task
    if cleanup_task is not None:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Cleanup task stopped")

    # Cleanup telemetry
    shutdown_telemetry()
    logger.info("Telemetry shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=config.service_name,
    description=config.description,
    version=config.service_version,
    lifespan=lifespan,
)

# Instrument FastAPI app with OpenTelemetry if available
if OTEL_INSTRUMENTATION_AVAILABLE:
    FastAPIInstrumentor.instrument_app(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    PublicJsonResponseSanitizationMiddleware,
    session_roots=(config.session_storage_path,),
)


# SessionManager raises InvalidSessionIdError (ValueError subclass) for
# malformed session_id path params (e.g. `../`, non-UUID); translate to 400
# so malicious traffic doesn't hit the default 500 handler. We target the
# subclass (not all ValueError) so unrelated ValueError bugs still surface as
# 500 — otherwise pydantic / type-conversion errors would silently map to 400.
from .session.manager import InvalidSessionIdError  # noqa: E402
from .storage.base import SessionMetadataContentionError  # noqa: E402


@app.exception_handler(InvalidSessionIdError)
async def _invalid_session_id_handler(
    request: Request, exc: InvalidSessionIdError
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(SessionMetadataContentionError)
async def _session_metadata_contention_handler(
    request: Request,
    exc: SessionMetadataContentionError,
) -> JSONResponse:
    """Return a retryable response without exposing storage internals."""
    del request, exc
    return JSONResponse(
        status_code=503,
        content={"detail": "Session metadata is temporarily busy; retry the request."},
        headers={"Retry-After": "1"},
    )


# Include routers
app.include_router(pipeline_router.router)
app.include_router(artifacts_router.router)
app.include_router(assets_router.router)
app.include_router(sessions_router.router)
app.include_router(materials_router.router)

# Mount docs/images as static files for user manual
docs_images_path = Path(__file__).parent.parent / "docs" / "images"
if docs_images_path.exists():  # pragma: no cover
    app.mount(
        "/docs/images", StaticFiles(directory=docs_images_path), name="docs-images"
    )


# Mount React build static assets (if available)
react_static_path = Path(__file__).parent.parent / "web" / "dist" / "_static"
if react_static_path.exists():  # pragma: no cover
    app.mount("/_static", StaticFiles(directory=react_static_path), name="react-static")


# Serve static files (index.html)
@app.get("/")
async def serve_index():
    """Serve the web UI (React build)."""
    react_index = Path(__file__).parent.parent / "web" / "dist" / "index.html"
    if react_index.exists():
        return FileResponse(react_index)
    return await root_api_info()


@app.get("/manual")
async def serve_manual():
    """Serve the user manual."""
    manual_path = Path(__file__).parent.parent / "manual.html"
    if manual_path.exists():
        return FileResponse(manual_path)
    else:
        return {"error": "User manual not found"}


@app.get("/3rd_party_licenses.html")
async def serve_third_party_licenses():
    """Serve the third-party licenses page."""
    licenses_path = Path(__file__).parent.parent / "3rd_party_licenses.html"
    if licenses_path.exists():
        return FileResponse(licenses_path, media_type="text/html")
    else:
        return {"error": "Third-party licenses file not found"}


@app.get("/license-body.html")
async def serve_license_body():
    """Serve the license agreement body HTML fragment."""
    path = Path(__file__).parent.parent / "license_body.html"
    if path.exists():
        return FileResponse(path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="License body file not found")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": config.service_name,
        "version": config.service_version,
        "api_keys_configured": config.has_required_api_keys,
        "image_gen_configured": config.image_gen_ready,
        "max_active_sessions": _get_max_active_sessions(),
    }


@app.get("/config/vlm-models")
async def get_vlm_models():
    """Return available VLM models for the UI dropdown."""
    models = [
        {
            "value": "nim/qwen/qwen3.5-397b-a17b",
            "label": "Qwen 3.5 397B (Default)",
            "is_default": True,
        },
        {
            "value": "nim/meta/llama-4-maverick-17b-128e-instruct",
            "label": "Llama 4 Maverick 17B",
            "is_default": False,
        },
        {
            "value": "nim/__custom__",
            "label": "NIM (Custom Model)",
            "is_default": False,
        },
    ]
    if os.getenv("ENABLE_COSMOS_VLM", "").strip().lower() in ("1", "true", "yes"):
        models.append(
            {
                "value": "nim/nvidia/cosmos-reason2-8b",
                "label": "Cosmos Reason 2 (8B)",
                "is_default": False,
            }
        )
    return {"models": models}


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
                "create": "POST /pipeline",
                "upload_usd": "POST /pipeline/upload-usd",
                "status": "GET /pipeline/{session_id}/status",
                "results": "GET /pipeline/{session_id}/results",
                "cancel": "POST /pipeline/{session_id}/cancel",
                "events": "GET /pipeline/{session_id}/events",
                "regenerate": "POST /pipeline/{session_id}/regenerate",
                "event_log": "GET /pipeline/{session_id}/event-log",
            },
            "artifacts": {
                "output_usd": "GET /artifacts/{session_id}/output",
                "final_render": "GET /artifacts/{session_id}/final-render",
                "predictions": "GET /artifacts/{session_id}/predictions",
                "report": "GET /artifacts/{session_id}/report",
            },
            "assets": {
                "input_render": "GET /assets/{session_id}/input-render",
                "previews": "GET /assets/{session_id}/previews",
                "references": "GET /assets/{session_id}/references",
            },
            "sessions": {
                "list": "GET /sessions",
                "get": "GET /sessions/{session_id}",
                "delete": "DELETE /sessions/{session_id}",
            },
            "materials": {
                "list": "GET /materials",
                "icon": "GET /materials/icon/{material_name}",
            },
        },
    }


def main():
    """Entry point for running the service."""
    import uvicorn

    uvicorn.run(
        "service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
