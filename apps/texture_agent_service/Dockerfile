# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Texture Agent Service Dockerfile
# No GPU/CUDA needed — texture generation uses API calls (Gemini/NVIDIA Inference)
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ARG WU_REQUIRED_OPTIONAL_BACKEND

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
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

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CVE-2026-4519: webbrowser.open() command injection - not exploitable
# (webbrowser module unused by this service). Ubuntu 24.04 patch pending.

RUN python -c "import sys; v=sys.version_info; assert v[:2] == (3, 12), \
    f'USD wheel compatibility: Python {sys.version} is unsupported, need 3.12'"

# Install uv for faster package management. Keep pip above the image-scan floor.
RUN python -m pip install --no-cache-dir --upgrade "pip>=26.1" uv
ENV UV_NO_CACHE=1

# Set working directory
WORKDIR /app

# Copy only the source trees needed to build and run this service rather than
# recursively copying the full repository build context.
COPY pyproject.toml README.md VERSION.md /app/
COPY world_understanding /app/world_understanding
COPY apps/__init__.py /app/apps/__init__.py
COPY apps/texture_agent /app/apps/texture_agent
COPY apps/texture_gen_service_common /app/apps/texture_gen_service_common
COPY apps/texture_gen_simple_service /app/apps/texture_gen_simple_service
COPY apps/texture_agent_service /app/apps/texture_agent_service

# Release automation stages optional runtime resources here. Copy the complete
# directory so Texture Agent can use the same local Scene Optimizer Core bundle
# as Material and Physics Agent when UV projection is requested.
COPY .build-resources /app/.build-resources

# Local Scene Optimizer backend. Populate `.build-resources/scene_optimizer_core`
# on the host by running `scripts/fetch_build_resources.sh` before
# `docker compose build`. The compiled extensions are cpython-312-only, so fail
# the image build when a staged bundle is malformed or the interpreter drifts.
RUN BUILD_RESOURCES=/app/.build-resources; \
    SO=$BUILD_RESOURCES/scene_optimizer_core; \
    if [ -d "$BUILD_RESOURCES" ]; then \
      chmod a+X "$BUILD_RESOURCES"; \
    fi; \
    if [ -d "$SO" ]; then \
      chmod -R a+rX "$SO"; \
    fi; \
    if [ ! -d "$SO" ]; then \
      echo "Scene Optimizer Core not provided - scene_optimizer UV projection will be unavailable"; \
    elif [ -d "$SO/python" ] && [ -d "$SO/lib" ] && \
         [ -d "$SO/extraLibs" ] && [ -d "$SO/usdpy" ]; then \
      python3 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" && \
      echo "Scene Optimizer Core present at $SO"; \
    else \
      echo "Scene Optimizer Core bundle is incomplete at $SO" >&2; \
      exit 1; \
    fi
ENV WU_SO_PACKAGE_DIR=/app/.build-resources/scene_optimizer_core

# Install world_understanding package and dependencies.
# --no-config + --no-sources keep the install self-contained inside the repo
# instead of resolving repo-local packages from an external index.
RUN cd /app && git init \
    && git config user.email "build@docker" \
    && git config user.name "build" \
    && git add -A \
    && git commit -m "docker build" \
    && SETUPTOOLS_SCM_PRETEND_VERSION="$(cat /app/VERSION.md)" UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
       uv pip install -e "." --no-config --no-sources \
    && rm -rf /app/.git

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

# Install texture_agent package if it exists. It imports the shared texture
# generation service helpers at runtime, so install that package first.
RUN cd /app/apps/texture_gen_service_common && UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
    uv pip install -e . --no-config --no-sources

RUN cd /app/apps/texture_gen_simple_service && UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
    uv pip install -e . --no-config --no-sources

RUN if [ -d "/app/apps/texture_agent" ]; then \
    cd /app/apps/texture_agent && UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
    uv pip install -e . --no-config --no-sources; \
    fi

# Install texture_agent_service package
RUN cd /app/apps/texture_agent_service && UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
    uv pip install -e . --no-config --no-sources

# Do not ship uv source caches or their nested Git metadata in the final image.
RUN rm -rf /root/.cache/uv

# Create directories for sessions
RUN mkdir -p /var/texture-agent/sessions

# Normalize copied source permissions for the non-root runtime user. Some host
# checkouts carry 0600/0700 modes in generated app folders; keep /app root-owned
# but readable/importable instead of recursively changing ownership.
RUN chmod -R a+rX /app/world_understanding \
    /app/apps \
    /app/apps/texture_agent \
    /app/apps/texture_gen_service_common \
    /app/apps/texture_gen_simple_service \
    /app/apps/texture_agent_service

# Create non-root user for security
RUN useradd -m -u 10000 texture-agent && \
    chown -R texture-agent:texture-agent /var/texture-agent

# Keep /app read-only for the service user without Python trying to create
# __pycache__ beside editable source files at runtime.
ENV PYTHONDONTWRITEBYTECODE=1

# Add Python paths for texture_agent imports
ENV PYTHONPATH="/app:/app/apps:/app/apps/texture_agent_service"

# Switch to non-root user
USER texture-agent

# Set working directory to service
WORKDIR /app/apps/texture_agent_service

# Environment variables with defaults
ENV TA_SESSION_STORAGE_PATH=/var/texture-agent/sessions \
    TA_SESSION_TTL_HOURS=24 \
    TA_MAX_ACTIVE_SESSIONS=4 \
    TA_MAX_UPLOAD_SIZE_MB=500 \
    TA_TEXTURE_BACKEND=simple_image_gen

# Build provenance, reported by /health so a running deployment can be
# identified from the outside. GIT_COMMIT and APP_VERSION are already supplied
# by the NGC Upload workflow; they were simply never carried into the image.
# IMAGE_VERSION is the resolved NGC tag, which is what distinguishes two builds
# of the same VERSION.md string. Declared last so a provenance change only
# invalidates this layer. Empty rather than "unknown" so a source checkout
# reports null instead of a placeholder that reads like a real value.
ARG GIT_COMMIT=""
ARG APP_VERSION=""
ARG IMAGE_VERSION=""
ARG BUILD_TIMESTAMP=""
ENV TA_BUILD_COMMIT_SHA=${GIT_COMMIT} \
    TA_BUILD_IMAGE_TAG=${IMAGE_VERSION} \
    TA_BUILD_TIMESTAMP=${BUILD_TIMESTAMP}
LABEL org.opencontainers.image.revision=${GIT_COMMIT} \
      org.opencontainers.image.version=${APP_VERSION} \
      org.opencontainers.image.created=${BUILD_TIMESTAMP}

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Run service with proper module path
CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8001"]
