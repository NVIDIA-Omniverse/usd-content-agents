# syntax=docker/dockerfile:1.19
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 AS geometry-agent-build

ARG APP_VERSION=unknown
ARG GIT_COMMIT=unknown
ARG UV_VERSION=0.11.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    binutils \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libjpeg-dev \
    libpng-dev \
    ninja-build \
    pkg-config \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN python3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" \
    && python3.12 -m venv /opt/build-tools
ENV PATH=/opt/build-tools/bin:$PATH \
    UV_NO_CACHE=1

RUN python -m pip install --no-cache-dir --upgrade "pip>=26.1" "uv==${UV_VERSION}"

WORKDIR /app

# Build from the public staging root. External authoring implementations are
# separate services and are deliberately absent from this image.
COPY VERSION.md README.md pyproject.toml /app/
COPY world_understanding /app/world_understanding
COPY agentic/packages/cad_verifier /app/agentic/packages/cad_verifier
COPY agentic/packages/content_agent_workflows /app/agentic/packages/content_agent_workflows
COPY agentic/packages/geometry_authoring_contracts /app/agentic/packages/geometry_authoring_contracts
COPY agentic/packages/geometry_authoring_connectors /app/agentic/packages/geometry_authoring_connectors
COPY --exclude=geometry_repair/[s]mlib_promotion.py \
    --exclude=geometry_repair/[s]mlib_repair.py \
    --exclude=geometry_repair/workers/[s]mlib_brep_heal.py \
    --exclude=tests/test_[s]mlib_promotion.py \
    --exclude=tests/test_[s]mlib_repair.py \
    agentic/packages/geometry_repair /app/agentic/packages/geometry_repair
COPY agentic/packages/sdf_tools /app/agentic/packages/sdf_tools
COPY apps/joint_agent /app/apps/joint_agent
COPY apps/physics_agent /app/apps/physics_agent
COPY apps/texture_agent /app/apps/texture_agent
COPY apps/texture_gen_service_common /app/apps/texture_gen_service_common
COPY apps/usd_cli /app/apps/usd_cli
COPY apps/geometry_agent_service /app/apps/geometry_agent_service
COPY .build-resources /app/.build-resources

# The public Scene Optimizer bundle is optional at build time but explicit at
# runtime. Geometry validation reports it as unavailable instead of claiming an
# optimized handoff when the resource was not staged.
RUN BUILD_RESOURCES=/app/.build-resources; \
    SO=$BUILD_RESOURCES/scene_optimizer_core; \
    chmod a+X "$BUILD_RESOURCES"; \
    if [ -d "$SO/python" ] && [ -d "$SO/lib" ] && \
       [ -d "$SO/extraLibs" ] && [ -d "$SO/usdpy" ]; then \
      chmod -R a+rX "$SO"; \
      echo "Scene Optimizer Core present at $SO"; \
    else \
      echo "Scene Optimizer Core not provided; optimization will be unavailable"; \
    fi

# Public staging intentionally omits repository lockfiles. Resolve from the
# sanitized build context, then keep installation frozen to that resolution.
RUN UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
    uv lock \
      --project /app/apps/geometry_agent_service \
      --python /usr/bin/python3.12 \
      --no-cache

RUN UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)" \
    uv sync \
      --project /app/apps/geometry_agent_service \
      --python /usr/bin/python3.12 \
      --frozen \
      --no-dev \
      --no-editable \
      --no-cache

# Public 0.6 deliberately defers CAD/mesh conversion. Fail the build if the
# unqualified converter or its embedded Python runtime re-enters transitively.
RUN --mount=type=bind,source=scripts/verify_geometry_converter_absent.py,target=/tmp/verify_geometry_converter_absent.py,ro \
    /app/apps/geometry_agent_service/.venv/bin/python \
      /tmp/verify_geometry_converter_absent.py

# Export only installed public runtime artifacts. Source copied into the build
# stage is intentionally unreachable from the published image layers.
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 AS geometry-agent-runtime

ARG APP_VERSION=unknown
ARG GIT_COMMIT=unknown

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libgomp1 \
    libglu1-mesa \
    libjpeg-turbo8 \
    libpng16-16t64 \
    libpython3.12t64 \
    libsm6 \
    libstdc++6 \
    libxext6 \
    libxi6 \
    libxrender1 \
    python3.12 \
    zlib1g \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && python3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)"

WORKDIR /app

COPY --from=geometry-agent-build \
    /app/apps/geometry_agent_service/.venv \
    /app/apps/geometry_agent_service/.venv
COPY --from=geometry-agent-build \
    /app/.build-resources \
    /app/.build-resources

RUN --mount=type=bind,source=scripts/verify_geometry_converter_absent.py,target=/tmp/verify_geometry_converter_absent.py,ro \
    /app/apps/geometry_agent_service/.venv/bin/python \
      /tmp/verify_geometry_converter_absent.py

RUN useradd --create-home --uid 10000 geometry-agent-service \
    && mkdir -p \
      /app/apps/geometry_agent_service \
      /var/geometry-agent-service/workspace \
    && chown -R geometry-agent-service:geometry-agent-service \
      /var/geometry-agent-service

ENV PATH=/app/apps/geometry_agent_service/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GEOMETRY_AGENT_SERVICE_HOST=0.0.0.0 \
    GEOMETRY_AGENT_SERVICE_PORT=8776 \
    GEOMETRY_AGENT_SERVICE_WORKSPACE_ROOT=/var/geometry-agent-service/workspace \
    WU_SO_PACKAGE_DIR=/app/.build-resources/scene_optimizer_core

LABEL org.opencontainers.image.version="${APP_VERSION}"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"

USER geometry-agent-service
WORKDIR /app/apps/geometry_agent_service

EXPOSE 8776

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; port = os.environ.get('GEOMETRY_AGENT_SERVICE_PORT', '8776'); urllib.request.urlopen(f'http://localhost:{port}/api/health', timeout=5).read()"]

CMD ["sh", "-c", "exec python -m uvicorn geometry_agent_service.main:app --host \"${GEOMETRY_AGENT_SERVICE_HOST:-0.0.0.0}\" --port \"${GEOMETRY_AGENT_SERVICE_PORT:-8776}\""]
