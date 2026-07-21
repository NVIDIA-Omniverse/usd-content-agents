# Joint Agent Service Dockerfile
# Uses NVIDIA CUDA base for GPU-accelerated rendering (OvRTX, Warp)
# Ubuntu 24.04 ships Python 3.12 natively — no deadsnakes PPA needed
FROM nvcr.io/nvidia/cuda:12.6.3-runtime-ubuntu24.04

ARG WU_REQUIRED_OPTIONAL_BACKEND

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Refresh the CUDA base image's Ubuntu packages before installing service
# dependencies. NGC CUDA runtime tags can lag Ubuntu security pockets, and the
# image scan gates fail on stale base OS packages such as OpenSSL/PAM/GnuTLS.
# This trades deterministic OS package contents for release scan compliance;
# prefer a newer CUDA base tag when NGC publishes one with these fixes.
# Install Python 3.12 (Ubuntu 24.04 default), system deps, and Vulkan/GL libs required by OvRTX
# CVE-2025-13836: Ubuntu 24.04 applies security backports to its python3.12 package
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    git \
    curl \
    libgl1 \
    libegl1 \
    libglu1-mesa \
    libgomp1 \
    libvulkan1 \
    libxrandr2 \
    libxi6 \
    libsm6 \
    gnupg \
    gpg \
    gpg-agent \
    gpgconf \
    gpgsm \
    gpg-wks-client \
    dirmngr \
    libgpgme11t64 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# CVE-2026-4519: webbrowser.open() command injection — not exploitable
# (webbrowser module unused by this service). Ubuntu 24.04 patch pending.

# Set up Python 3.12 as default
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Create a venv for the app (avoids PEP 668 "externally managed" error)
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install uv for faster package management
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy only the source trees needed to build and run this service rather than
# recursively copying the full repository build context.
COPY pyproject.toml README.md VERSION.md /app/
COPY world_understanding /app/world_understanding
COPY apps/joint_agent /app/apps/joint_agent
COPY apps/joint_agent_service /app/apps/joint_agent_service

# Initialize a git repo so uv-dynamic-versioning can resolve the version
# (.git is excluded by .dockerignore)
RUN cd /app && git init && git config user.email "build@docker" && git config user.name "build" && git add -A && git commit -m "docker build"

# Release automation may stage an optional backend wheel here.
COPY .build-resources/optional-backend /app/.build-resources/optional-backend

# Install world_understanding package and dependencies.
# --no-config + --no-sources keep the install self-contained inside the repo
# instead of resolving repo-local packages from an external index. Keep every
# editable package on the release version so local dependency floors resolve
# against the already-installed world-understanding package.
RUN cd /app && SETUPTOOLS_SCM_PRETEND_VERSION="$(cat /app/VERSION.md)" UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
    uv pip install -e ".[warp]" --no-config --no-sources

# Install the optional backend wheel when release automation stages it in the
# Docker build context. Public builds do not provide this wheel and continue
# with the bundled public backends only.
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

# Install joint_agent package if it exists
RUN if [ -d "/app/apps/joint_agent" ]; then \
    cd /app/apps/joint_agent && UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
    uv pip install -e . --no-config --no-sources; \
    fi

# Install joint_agent_service package
RUN cd /app/apps/joint_agent_service && UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
    uv pip install -e . --no-config --no-sources

# Create isolated ovrtx venv (separate from app venv to avoid USD native lib
# conflicts). Use the package provisioning CLI so Docker builds and runtime
# validation share one explicit OVRTX version/dependency source.
ENV WU_OVRTX_VENV_DIR=/opt/ovrtx_venv
RUN UV_CACHE_DIR=/tmp/wu-ovrtx-uv-cache \
    WU_OVRTX_LOCK_DIR=/tmp/wu-ovrtx-locks \
    python -m world_understanding.functions.graphics.render_ovrtx --provision-only && \
    rm -rf /tmp/wu-ovrtx-locks /tmp/wu-ovrtx-uv-cache /root/.cache/uv
ENV WU_OVRTX_AUTO_PROVISION=0

# Vulkan ICD — tells Vulkan where to find NVIDIA's driver (headless, no X11 needed)
COPY apps/joint_agent_service/docker/nvidia_icd.json /usr/share/vulkan/icd.d/
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# NVIDIA container runtime env vars
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

# Create directories for sessions
RUN mkdir -p /var/joint-agent/sessions

# Create non-root user for security (ubuntu user with UID 1000 doesn't exist in this base)
RUN useradd -m -u 10000 joint-agent && \
    chown -R joint-agent:joint-agent /var/joint-agent /opt/ovrtx_venv
RUN chmod -R a+rX /app && \
    chmod -R go-w /app

# Fail the build if restrictive source modes make the editable service package
# unreadable after the runtime user switch.
RUN su -s /bin/sh -c \
    'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app:/app/apps:/app/apps/joint_agent_service /opt/venv/bin/python -c "import service.main"' \
    joint-agent

# Keep /app read-only for the service user without Python trying to create
# __pycache__ beside editable source files at runtime.
ENV PYTHONDONTWRITEBYTECODE=1

# Add Python paths for joint_agent imports
ENV PYTHONPATH="/app:/app/apps:/app/apps/joint_agent_service"

# Switch to non-root user
USER joint-agent

# Set working directory to service
WORKDIR /app/apps/joint_agent_service

# Environment variables with defaults
ENV JA_SESSION_STORAGE_PATH=/var/joint-agent/sessions \
    JA_SESSION_TTL_HOURS=24 \
    JA_MAX_ACTIVE_SESSIONS=1 \
    JA_MAX_UPLOAD_SIZE_MB=0 \
    JA_RENDER_BACKEND=remote \
    JA_DEBUG=false

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run service with proper module path
CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
