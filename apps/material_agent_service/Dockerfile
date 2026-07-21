# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Material Agent Service Dockerfile
FROM ubuntu:24.04

ARG APP_VERSION=unknown
ARG GIT_COMMIT=unknown
ARG WU_REQUIRED_OPTIONAL_BACKEND

# Prevent interactive prompts during apt-get.
ENV DEBIAN_FRONTEND=noninteractive

# CVE-2025-13836: Ubuntu 24.04 applies security backports to Python 3.12.
# Use Python 3.12 because usd-exchange currently publishes Linux ARM64 wheels
# for cp310-cp312, not cp313.
# CVE-2026-4519: webbrowser.open() command injection — not exploitable
# (webbrowser module unused by this service). Ubuntu 24.04 patch pending.

# Install system dependencies. --no-install-recommends keeps the image lean
# and avoids pulling in unexpected packages (docker:S6500).
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python -c "import sys; v=sys.version_info; assert v[:2] == (3, 12), \
    f'USD wheel compatibility: Python {sys.version} is unsupported, need 3.12'"

# Install uv for faster package management. Keep pip above the image-scan floor.
RUN python -m pip install --no-cache-dir --upgrade "pip>=26.1" uv
ENV UV_NO_CACHE=1

# Set working directory
WORKDIR /app

# Copy only the source trees needed to build and run this service. Keep
# `.build-resources` present so an optional Scene Optimizer bundle can be
# staged there before `docker compose build` without copying unrelated local
# files from the repository root.
COPY pyproject.toml README.md VERSION.md /app/
COPY world_understanding /app/world_understanding
COPY apps/material_agent /app/apps/material_agent
COPY apps/material_agent_service /app/apps/material_agent_service
COPY .build-resources /app/.build-resources

# Local Scene Optimizer backend. Populate `.build-resources/scene_optimizer_core`
# on the host by running `scripts/fetch_build_resources.sh` before
# `docker compose build`. When absent, the `optimize_usd` step falls back to
# NVCF at runtime.
#
# The SO bundle's compiled extensions are cpython-312-only. Verify the distro
# interpreter before accepting a staged bundle.
RUN BUILD_RESOURCES=/app/.build-resources; \
    SO=$BUILD_RESOURCES/scene_optimizer_core; \
    if [ -d "$BUILD_RESOURCES" ]; then \
      chmod a+X "$BUILD_RESOURCES"; \
    fi; \
    if [ -d "$SO" ]; then \
      chmod -R a+rX "$SO"; \
    fi; \
    if [ -d "$SO/python" ] && [ -d "$SO/lib" ] && \
       [ -d "$SO/extraLibs" ] && [ -d "$SO/usdpy" ]; then \
      if python3 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"; then \
        echo "Scene Optimizer Core present at $SO; using base Python 3.12 for SO subprocess"; \
      else \
        echo "Scene Optimizer Core present at $SO but distro Python 3.12 is unavailable"; \
        exit 1; \
      fi; \
    else \
      echo "Scene Optimizer Core not provided — optimize_usd will fall back to NVCF"; \
    fi
ENV WU_SO_PACKAGE_DIR=/app/.build-resources/scene_optimizer_core

# Install world_understanding package and dependencies.
# --no-config + --no-sources give us a clean, reproducible install path:
# ignore any host-level uv config and tool.uv.sources indirection.
# Keep every editable package on the repository release version so local
# dependency floors resolve against the installed world-understanding package.
RUN cd /app && SETUPTOOLS_SCM_PRETEND_VERSION="$(cat /app/VERSION.md)" UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
    uv pip install -e ".[telemetry]" --no-config --no-sources

# Install the optional backend wheel when release automation stages it in the
# Docker build context. Public builds do not provide this wheel and continue
# with public backends only.
RUN if ls /app/.build-resources/optional-backend/*.whl >/dev/null 2>&1; then \
      uv pip install --no-deps --no-config --no-sources \
        /app/.build-resources/optional-backend/*.whl && \
      WU_REQUIRED_OPTIONAL_BACKEND="${WU_REQUIRED_OPTIONAL_BACKEND}" \
      python -c "import os; import world_understanding.functions.models.backends; from world_understanding.functions.models.backends.registry import list_chat_backends, list_image_gen_backends, list_loaded_backend_plugins, list_text_embedding_backends, list_vlm_backends; required = os.environ.get('WU_REQUIRED_OPTIONAL_BACKEND', ''); registered = set(list_chat_backends()) | set(list_vlm_backends()) | set(list_image_gen_backends()) | set(list_text_embedding_backends()); assert list_loaded_backend_plugins(), 'Optional backend wheel did not register a plugin'; assert not required or required in registered, f'Required optional backend {required!r} was not registered'"; \
    elif [ -n "${WU_REQUIRED_OPTIONAL_BACKEND}" ]; then \
      echo "Required optional backend ${WU_REQUIRED_OPTIONAL_BACKEND} has no staged wheel" >&2; \
      exit 1; \
    else \
      echo "Optional backend wheel not provided; using public backends only"; \
    fi

# Install material_agent package if it exists
RUN if [ -d "/app/apps/material_agent" ]; then \
      cd /app/apps/material_agent && UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
      uv pip install -e . --no-config --no-sources; \
    fi

# Install material_agent_service package (versions from pyproject.toml)
RUN cd /app/apps/material_agent_service && UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" uv pip install -e . --no-config --no-sources

# Do not ship uv source caches or their nested Git metadata in the final image.
RUN rm -rf /root/.cache/uv

# Create directories for materials and sessions
RUN mkdir -p /var/material-agent/sessions /app/materials

# Keep copied source importable when the host checkout has restrictive modes.
RUN chmod -R a+rX \
    /app/world_understanding \
    /app/apps/material_agent \
    /app/apps/material_agent_service

# Create non-root user for security
RUN useradd -m -u 10000 material-agent && \
    chown -R material-agent:material-agent /var/material-agent

# Keep /app read-only for the service user without Python trying to create
# __pycache__ beside editable source files at runtime.
ENV PYTHONDONTWRITEBYTECODE=1

# Add Python paths for material_agent imports
ENV PYTHONPATH="/app:/app/apps:/app/apps/material_agent_service"

# Switch to non-root user
USER material-agent

# Set working directory to service
WORKDIR /app/apps/material_agent_service

# Environment variables with defaults
ENV MA_SESSION_STORAGE_PATH=/var/material-agent/sessions \
    MA_SESSION_TTL_HOURS=24 \
    MA_MAX_ACTIVE_SESSIONS=8 \
    MA_MAX_UPLOAD_SIZE_MB=500 \
    MA_GEOM_LIMIT=0 \
    MA_DEBUG=false \
    MA_PROGRESS_POLL_INTERVAL_SECONDS=2 \
    MA_IMAGE_WIDTH=512 \
    MA_IMAGE_HEIGHT=512 \
    MA_RENDERER_BACKEND=remote \
    MA_MAX_RENDER_NUM_WORKERS=32

LABEL org.opencontainers.image.version="${APP_VERSION}"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"

ENV APP_VERSION=${APP_VERSION}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run service with proper module path
CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
