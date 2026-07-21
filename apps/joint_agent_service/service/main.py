# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint Agent FastAPI Service - Main Application."""

import logging
import sys
from contextlib import asynccontextmanager
from functools import cache
from pathlib import Path

from world_understanding.utils.logging import setup_logging
from world_understanding.utils.public_response import (
    PublicJsonResponseSanitizationMiddleware,
)

from .utils import AccessLogFilter

# Add parent directories to Python path for joint_agent imports
service_dir = Path(__file__).parent.parent
apps_dir = service_dir.parent
repo_root = apps_dir.parent

for path in [str(apps_dir), str(repo_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import importlib  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

# Load .env file BEFORE importing config
load_dotenv()

from .config import config  # noqa: E402
from .routers import (  # noqa: E402
    artifacts_router,
    pipeline_router,
    sessions_router,
)
from .runtime.registry import get_job_registry  # noqa: E402
from .session.manager import (  # noqa: E402
    PREFERRED_JOINT_RIGGER_OUTPUT_FILENAME,
    SessionManager,
)

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


def _load_aws_config_file_into_env(*, log: logging.Logger) -> None:
    """Expand AWS_CONFIG_FILE contents into process environment variables."""
    from dotenv import dotenv_values

    config_path_raw = os.getenv("AWS_CONFIG_FILE")
    if not config_path_raw:
        return

    config_path = Path(config_path_raw).expanduser()

    if not config_path.exists():
        log.warning("AWS_CONFIG_FILE set but file does not exist: %s", config_path)
        return

    try:
        values = dotenv_values(config_path)
    except Exception:
        log.exception("Failed reading AWS_CONFIG_FILE: %s", config_path)
        return

    def _set_env_if_missing(*, env_key: str, value: str | None) -> bool:
        if not value:
            return False
        if os.getenv(env_key):
            return False
        os.environ[env_key] = value
        return True

    access_key_id = values.get("AWS_ACCESS_KEY_ID") or values.get("aws_access_key_id")
    secret_access_key = values.get("AWS_SECRET_ACCESS_KEY") or values.get(
        "aws_secret_access_key"
    )
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
        _set_env_if_missing(env_key="AWS_REGION", value=region)

    if any_set:
        log.info("AWS_CONFIG_FILE loaded.")

    if "AWS_ACCESS_KEY_ID" in os.environ and os.environ["AWS_ACCESS_KEY_ID"]:
        log.info("AWS Access Key ID: present")
    if "AWS_SECRET_ACCESS_KEY" in os.environ and os.environ["AWS_SECRET_ACCESS_KEY"]:
        log.info("AWS Secret Access Key: present")
    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
    if region:
        log.info("AWS Region: %s", region)


@cache
def _joint_rigger_capabilities() -> dict[str, object]:
    capabilities: dict[str, object] = {}
    try:
        importlib.import_module("joint_agent.functions.joint_rigger_core_bridge")
    except Exception as exc:
        capabilities.update(
            {
                "owned_core_available": False,
                "owned_core_import_error_type": type(exc).__name__,
            }
        )
    else:
        capabilities["owned_core_available"] = True

    try:
        module = importlib.import_module("usd_joint_rigger")
    except Exception as exc:
        capabilities.update(
            {
                "usd_joint_rigger_available": False,
                "import_error_type": type(exc).__name__,
            }
        )
        return capabilities

    capabilities.update(
        {
            "usd_joint_rigger_available": True,
            "has_apply_joint_rigger": callable(
                getattr(module, "apply_joint_rigger", None)
            ),
            "has_handoff_create_joints": callable(
                getattr(module, "create_joints", None)
            ),
        }
    )
    return capabilities


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting Joint Agent Service...")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(AccessLogFilter())

    # Check for required API keys
    if not config.has_required_api_keys:
        logger.warning(
            "NVIDIA_API_KEY is not set. The default hosted NIM model backend "
            "is unavailable until a credential or another public backend is "
            "configured."
        )

    # Expand AWS_CONFIG_FILE into AWS* env vars
    _load_aws_config_file_into_env(log=logger)

    # Capabilities are process-static; load once during startup so /health does
    # not perform the first optional package import on the asyncio event loop.
    _joint_rigger_capabilities()

    # Initialize session store and manager
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
    session_mgr = SessionManager(
        storage_path=config.session_storage_path,
        ttl_hours=config.session_ttl_hours,
        store=store,
        run_claim_lease_seconds=config.run_claim_lease_seconds,
        run_claim_heartbeat_seconds=config.run_claim_heartbeat_seconds,
    )

    # Set global session manager in all routers
    pipeline_router.set_session_manager(session_mgr)
    artifacts_router.set_session_manager(session_mgr)
    sessions_router.set_session_manager(session_mgr)

    # Pre-initialise warp + compile kernels in the main thread so that the
    # first pipeline run doesn't hang when asyncio.to_thread delegates to a
    # worker thread.  Kernel compilation spawns subprocesses that can deadlock
    # inside the default executor.
    try:
        import warp as wp

        wp.init()
        logger.info(
            "Warp initialised: %s (devices: %d)",
            wp.__version__,
            wp.get_cuda_device_count(),
        )
        # Pre-import newton raytrace so kernel compilation happens now (main
        # thread) rather than inside asyncio.to_thread where subprocess
        # spawning can deadlock.
        try:
            from newton._src.sensors.warp_raytrace import (  # noqa: F401
                RenderContext,
            )

            logger.info("Newton warp_raytrace loaded — kernels will be cached")
        except ImportError:  # pragma: no cover - optional dependency branch
            logger.info("Newton not installed — Warp rendering backend unavailable")
    except ImportError:
        logger.info("warp-lang not installed — Warp rendering backend unavailable")
    except Exception as e:
        logger.warning("Warp initialisation failed: %s", e)

    logger.info("Service started: %s v%s", config.service_name, config.service_version)
    logger.info(
        f"VLM: {config.vlm_backend} / {config.vlm_model} "
        f"(temp={config.vlm_temperature})"
    )
    provider_status = "configured" if config.has_required_api_keys else "NOT SET"
    logger.info(f"Default model credential: {provider_status}")

    yield

    # Shutdown
    logger.info("Shutting down Joint Agent Service...")


# Create FastAPI app
app = FastAPI(
    title=config.service_name,
    description=config.description or "",
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


# SessionManager raises InvalidSessionIdError (ValueError subclass) for
# malformed session_id path params (e.g. `../`, non-UUID); translate to 400
# so malicious traffic doesn't hit the default 500 handler. We target the
# subclass (not all ValueError) so unrelated ValueError bugs still surface as
# 500 — otherwise pydantic / type-conversion errors would silently map to 400.
from .session.manager import InvalidSessionIdError  # noqa: E402


@app.exception_handler(InvalidSessionIdError)
async def _invalid_session_id_handler(
    request: Request, exc: InvalidSessionIdError
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# Include routers
app.include_router(pipeline_router.router)
app.include_router(artifacts_router.router)
app.include_router(sessions_router.router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": config.service_name,
        "version": config.service_version,
        "api_keys_configured": config.has_required_api_keys,
        "max_active_sessions": _get_max_active_sessions(),
        "capabilities": {
            "joint_rigger": _joint_rigger_capabilities(),
        },
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
                "create": "POST /pipeline",
                "status": "GET /pipeline/{session_id}/status",
                "results": "GET /pipeline/{session_id}/results",
                "cancel": "POST /pipeline/{session_id}/cancel?run_id={run_id}",
                "events": "GET /pipeline/{session_id}/events",
                "regenerate": "POST /pipeline/{session_id}/regenerate",
            },
            "artifacts": {
                "predictions": "GET /artifacts/{session_id}/predictions",
                "report": "GET /artifacts/{session_id}/report",
                "dataset": "GET /artifacts/{session_id}/dataset",
                "joint_rigger_output": (
                    "GET /artifacts/{session_id}/joint-rigger-output"
                ),
                "joint_rigger_output_filename": (
                    PREFERRED_JOINT_RIGGER_OUTPUT_FILENAME
                ),
                "joint_rigger_diagnostics": (
                    "GET /artifacts/{session_id}/joint-rigger-diagnostics"
                ),
                "joint_rigger_validation": (
                    "GET /artifacts/{session_id}/joint-rigger-validation"
                ),
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
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
