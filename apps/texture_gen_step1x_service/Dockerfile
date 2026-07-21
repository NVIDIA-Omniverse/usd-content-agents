# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:24.04

ARG APP_VERSION=unknown
ARG GIT_COMMIT=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/world-understanding \
    TEXTURE_STEP1X_REQUIRED_EXECUTABLES=uv

# Install Ubuntu-patched Python instead of copying the official Python image's
# /usr/local interpreter into the scanned final image.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        ca-certificates \
        libgl1 \
        libstdc++6 \
        zlib1g \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace/world-understanding

COPY --chown=root:root --chmod=0555 \
    apps/texture_gen_service_common ./apps/texture_gen_service_common
COPY --chown=root:root --chmod=0555 \
    apps/texture_gen_step1x_service ./apps/texture_gen_step1x_service
COPY --chown=root:root --chmod=0555 \
    apps/texture_gen_step1x_service/healthcheck.py /healthcheck.py

# Keep this reduced image dependency list in sync with imports from the copied
# Step1X service and common service packages. It intentionally avoids installing
# the full world-understanding project and its transitive source-scan surface.
# Use the same public usd-exchange/OpenUSD binding contract as the root project
# on every supported architecture so image and runtime notice resolution agree.
# Keep pip above the container image-scan floor used by the public services.
RUN python -m pip install --no-cache-dir --upgrade \
        "pip>=26.1" \
        "uv>=0.9.17,<0.12.0" \
    && python -m pip install --no-cache-dir \
        "fastapi>=0.115.8,<1.0.0" \
        "pydantic>=2.11,<3.0.0" \
        "uvicorn[standard]>=0.34.0,<1.0.0" \
        "python-dotenv==1.2.2" \
        "Pillow>=12.1.1,<13.0.0" \
        "usd-exchange>=2.3,<3"

RUN mkdir -p \
        /var/texture-agent/sessions \
        /work/cache/cuda \
        /work/cache/cupy \
        /work/cache/matplotlib \
        /work/cache/numba \
        /work/cache/torch \
        /work/cache/triton \
        /work/cache/uv \
        /work/cache/xdg \
        /work/texture_sessions \
    && groupadd --gid 10000 texture \
    && useradd --create-home --uid 10000 --gid 10000 texture \
    && chown -R texture:texture /var/texture-agent /work

# CVE-2026-4519: webbrowser.open() command injection is not exploitable here;
# the Step1X service does not import or invoke the webbrowser module.

ENV CUDA_CACHE_PATH=/work/cache/cuda \
    CUPY_CACHE_DIR=/work/cache/cupy \
    MPLCONFIGDIR=/work/cache/matplotlib \
    NUMBA_CACHE_DIR=/work/cache/numba \
    TORCH_HOME=/work/cache/torch \
    TRITON_CACHE_DIR=/work/cache/triton \
    UV_CACHE_DIR=/work/cache/uv \
    XDG_CACHE_HOME=/work/cache/xdg

USER texture

EXPOSE 8000

LABEL org.opencontainers.image.version="${APP_VERSION}"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"

# The public image does not bundle Step1X weights or third-party runtime
# checkouts. Keep Docker health scoped to API liveness; /health still reports
# ready=false with diagnostics until an operator mounts a complete runtime.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD ["/opt/venv/bin/python", "/healthcheck.py", "--liveness"]

CMD ["/opt/venv/bin/python", "-m", "uvicorn", "apps.texture_gen_step1x_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
