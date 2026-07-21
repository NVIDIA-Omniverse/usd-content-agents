# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Physics Agent Service Dockerfile
# The main service is CPU-only. Rendering is delegated to the sidecar
# ovrtx-rendering-api container via RENDER_ENDPOINT.
FROM ubuntu:24.04

ARG APP_VERSION=unknown
ARG GIT_COMMIT=unknown
ARG TARGETARCH
ARG WU_REQUIRED_OPTIONAL_BACKEND

# Prevent interactive prompts during apt-get.
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies. The gnupg/libgpgme packages pull the
# CVE-2025-68973 fix; keep them listed explicitly so the security test
# in tests/test_dockerfile_security.py sees them.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    gnupg \
    gpg \
    gpg-agent \
    gpgconf \
    gpgsm \
    gpg-wks-client \
    dirmngr \
    libgpgme11t64 \
    libgl1 \
    libgomp1 \
    libopengl0 \
    libx11-6 \
    libxt6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CVE-2026-4519: webbrowser.open() command injection — not exploitable
# (webbrowser module unused by this service). Ubuntu 24.04 patch pending.

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
COPY apps/physics_agent /app/apps/physics_agent
COPY apps/physics_agent_service /app/apps/physics_agent_service
COPY .build-resources /app/.build-resources

# Local Scene Optimizer backend. Populate `.build-resources/scene_optimizer_core`
# on the host by running `scripts/fetch_build_resources.sh` before
# `docker compose build`. When absent, the `optimize_usd` step falls back to
# NVCF at runtime.
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
      echo "Scene Optimizer Core present at $SO"; \
    else \
      echo "Scene Optimizer Core not provided — optimize_usd will fall back to NVCF"; \
    fi
ENV WU_SO_PACKAGE_DIR=/app/.build-resources/scene_optimizer_core

# Install world_understanding package and dependencies.
# --no-config + --no-sources keep the install self-contained inside the repo.
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

# Install first-party Physics Agent code without resolving the optional tuning
# graph. The CPU parent runtime is locked separately below.
RUN if [ -d "/app/apps/physics_agent" ]; then \
    cd /app/apps/physics_agent && UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
    uv pip install -e . --no-deps --no-config --no-sources; \
    fi
RUN case "${TARGETARCH:-amd64}" in \
      amd64) tuning_lock=pylock.physics-tuning-runtime.toml ;; \
      arm64) tuning_lock=pylock.physics-tuning-runtime.aarch64.toml ;; \
      *) echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac && \
    uv pip install \
      -r "/app/apps/physics_agent/runtime/${tuning_lock}" \
      --no-config --no-sources
RUN python -c "from physics_agent.tuning.optimizers import is_botorch_available; assert is_botorch_available(), 'physics_agent[tuning] install missing BoTorch support'"

# Refine uses OvPhysX through a separate daemon venv because ovphysx bundles an
# OpenUSD build that conflicts with the parent process' usd-core.
ENV WU_OVPHYSX_VENV_DIR=/opt/ovphysx_venv
RUN case "${TARGETARCH:-amd64}" in \
      amd64) ovphysx_lock=pylock.ovphysx-runtime.toml ;; \
      arm64) ovphysx_lock=pylock.ovphysx-runtime.aarch64.toml ;; \
      *) echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac && \
    uv venv "${WU_OVPHYSX_VENV_DIR}" && \
    uv pip install \
      --python "${WU_OVPHYSX_VENV_DIR}/bin/python" \
      -r "/app/apps/physics_agent/runtime/${ovphysx_lock}" \
      --require-hashes --no-deps --no-config --no-sources && \
    chmod -R a+rX "${WU_OVPHYSX_VENV_DIR}"
RUN env -u PYTHONPATH "${WU_OVPHYSX_VENV_DIR}/bin/python" -c \
    "from ovphysx import PhysX; physics = PhysX(device='cpu'); physics.release()" && \
    touch "${WU_OVPHYSX_VENV_DIR}/.wu-ovphysx-runtime-ready"

# Install physics_agent_service package
RUN cd /app/apps/physics_agent_service && UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
    uv pip install -e . --no-config --no-sources

# Do not ship uv source caches or their nested Git metadata in the final image.
RUN rm -rf /root/.cache/uv

# Create directories for sessions
RUN mkdir -p /var/physics-agent/sessions

# Keep copied source importable when the host checkout has restrictive modes.
RUN chmod -R a+rX \
    /app/world_understanding \
    /app/apps/physics_agent \
    /app/apps/physics_agent_service

# Create non-root user for security
RUN useradd -m -u 10000 physics-agent && \
    chown -R physics-agent:physics-agent /var/physics-agent

# Keep /app read-only for the service user without Python trying to create
# __pycache__ beside editable source files at runtime.
ENV PYTHONDONTWRITEBYTECODE=1

# Add Python paths for physics_agent imports
ENV PYTHONPATH="/app:/app/apps:/app/apps/physics_agent_service"

# Switch to non-root user
USER physics-agent

# Set working directory to service
WORKDIR /app/apps/physics_agent_service

# Environment variables with defaults
ENV PA_SESSION_STORAGE_PATH=/var/physics-agent/sessions \
    PA_SESSION_TTL_HOURS=24 \
    PA_MAX_ACTIVE_SESSIONS=1 \
    PA_MAX_UPLOAD_SIZE_MB=500 \
    PA_DEBUG=false \
    PA_RENDER_BACKEND=remote \
    PA_MAX_WORKERS=16

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
