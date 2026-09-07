# Managed Content Agents OVRTX adapter. Needs an NVIDIA RTX GPU at runtime.
# Supply an immutable CUDA base, for example
# nvidia/cuda:12.6.3-runtime-ubuntu24.04@sha256:<approved digest>.
ARG CUDA_BASE_IMAGE
FROM ${CUDA_BASE_IMAGE}

# Populated by buildkit; needed to pick the ovrtx wheel for the built image.
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    # The venv baked below is the service runtime. Readiness is now the
    # contents of that marker rather than its existence, so a persisted volume
    # from an older image, or a remapped WU_OVRTX_VENV_DIR, would otherwise
    # reach the replace path: rmtree ~2.5 GB and refetch from pypi.nvidia.com,
    # which a hardened deployment blocks. No apostrophes in this block -- it is
    # inside a continuation, and tests/test_copy_to_staging.py shlex-splits the
    # joined instruction to find COPY/ADD sources.
    WU_OVRTX_AUTO_PROVISION=0 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # the container's system Python is the service environment; Ubuntu 24.04's
    # PEP 668 marker would otherwise refuse the image's own pip installs
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DISPLAY=:0 \
    WU_OVRTX_VENV_DIR=/opt/ovrtx_venv \
    WU_OVPHYSX_VENV_DIR=/opt/ovphysx_venv

# The pinned CUDA base lags Ubuntu security updates, so patch the OS packages
# it ships before installing anything else. Without this the image inherits
# known high/critical advisories in libssl, libc6, perl and friends, which the
# container vulnerability gate rejects.
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        xvfb libvulkan1 vulkan-tools libgl1 libglib2.0-0 libgomp1 \
        libopengl0 libglu1-mesa libegl1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Both built distributions and the reviewed native locks support Python 3.11
# and 3.12. Fail during the image build for any other distro Python.
RUN python3 -c \
    'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), sys.version'

# Older nvidia-container-toolkits (<=1.14) mount the driver libraries but not the
# Vulkan ICD / GLVND EGL vendor manifests; without all three of libegl1, these two
# JSON files, and the driver libs, vkCreateInstance fails with ERROR_INCOMPATIBLE_DRIVER
# (the 550-series ICD initializes through the GLVND EGL path). Newer toolkits mount
# identical files over these — baking them in is harmless there.
RUN mkdir -p /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d \
    && printf '%s\n' \
       '{ "file_format_version": "1.0.0",' \
       '  "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3.277" } }' \
       > /etc/vulkan/icd.d/nvidia_icd.json \
    && printf '%s\n' \
       '{ "file_format_version": "1.0.0",' \
       '  "ICD": { "library_path": "libEGL_nvidia.so.0" } }' \
       > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /app

# Install the in-tree usd-cli component (usd_core) + its managed OVRTX adapter.
COPY pyproject.toml README.md LICENSE /app/repo/
COPY src /app/repo/src
COPY requirements/constraints.txt /app/repo/requirements/constraints.txt
COPY apps/ovrtx_rendering_api /app/service_app
RUN python3 -m pip wheel --no-cache-dir -c /app/repo/requirements/constraints.txt \
       --wheel-dir /tmp/wheels '/app/repo[cli,server]' /app/service_app \
    && python3 -m pip install --no-cache-dir --no-index --find-links /tmp/wheels \
       'usd-cli[cli,server]==0.0.1' 'ovrtx-rendering-api==0.1.0' \
    && python3 -m pip check \
    && python3 -m venv /opt/ovrtx_venv \
    # ovrtx is installed from the exact artifacts the runtime locks record, so
    # pip verifies each #sha256= and no index resolution can substitute a
    # different wheel. The wheel is chosen by TARGETARCH: the previous
    # index-based install picked the architecture automatically, and hard-coding
    # x86_64 broke arm64 image builds, which are an official target.
    # numpy/pillow use the lock's exact wheel for both supported Python minors
    # and architectures. A version-only index install cannot justify stamping
    # the whole-lock identity into the readiness marker below.
    # TARGETARCH is a BuildKit automatic ARG: under the legacy builder it
    # expands empty, the arm64 branch never fires, and the image would bake
    # the x86_64 wheel while still writing the readiness marker -- a venv
    # _venv_matches_pin accepts and whose ovrtx cannot import. Fail closed.
    && case "${TARGETARCH}" in amd64|arm64) ;; *) echo "TARGETARCH must be amd64 or arm64 (got: ${TARGETARCH:-unset}); build with BuildKit" >&2; exit 1 ;; esac \
    && OVRTX_WHEEL="https://pypi.nvidia.com/ovrtx/ovrtx-0.4.1.364340-py3-none-manylinux_2_35_x86_64.whl#sha256=597e09f34819f31a71fb919c9b0024c3b652588bdfa4d1858b2222f219d0ec09" \
    && OVSTAGE_WHEEL="https://pypi.nvidia.com/ovstage/ovstage-0.1.1.355824-py3-none-manylinux_2_35_x86_64.whl#sha256=b4affd9e5b23620ec71901a164df4ae373d25de2c76b9fae4d193d97aa2df0ec" \
    && WARP_WHEEL="https://pypi.nvidia.com/warp-lang/warp_lang-1.16.0-py3-none-manylinux_2_28_x86_64.whl#sha256=96449fc1e3b354185e2f09434fb794b5953ab2e8673b104d7e7f48d5d418bb35" \
    && if [ "${TARGETARCH}" = "arm64" ]; then \
         OVRTX_WHEEL="https://pypi.nvidia.com/ovrtx/ovrtx-0.4.1.364340-py3-none-manylinux_2_35_aarch64.whl#sha256=38b33e5052b81572c11af317fd9f214b873a631ae727e448f6a6b990597c0320"; \
         OVSTAGE_WHEEL="https://pypi.nvidia.com/ovstage/ovstage-0.1.1.355824-py3-none-manylinux_2_35_aarch64.whl#sha256=046aa360959bf5344d912470aaa11d3ec1fc750e28816e5239ce0438864c9863"; \
         WARP_WHEEL="https://pypi.nvidia.com/warp-lang/warp_lang-1.16.0-py3-none-manylinux_2_34_aarch64.whl#sha256=d4715171bd6821436b82293d774c9a082ace08215c189572b4348d1e48fb8ee8"; \
       fi \
    && PYTHON_MINOR="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" \
    && case "${PYTHON_MINOR}/${TARGETARCH}" in \
         3.11/amd64) NUMPY_WHEEL="https://files.pythonhosted.org/packages/02/03/74fe2a4cb3817d94d86402f2506554130a2f01414e299b5a843e5a8a957f/numpy-2.4.6-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=89cd468399cfd2504718f0ba50e410dca55a170b61a02ad92bb18c8a65186e93"; PILLOW_WHEEL="https://files.pythonhosted.org/packages/3b/2d/ede717bc1144f63886c21fd349bb95860b0d1a21149ff16f2bb362b612b6/pillow-12.3.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=23d27a3e0307ec2244cc51e7287b919aa68d097504ebe19df4e76a98a3eea5bd" ;; \
         3.11/arm64) NUMPY_WHEEL="https://files.pythonhosted.org/packages/33/a8/6fa8c1a345a8c85dbb21932c447bee07c30a2c2a3f31e369c0a84b300147/numpy-2.4.6-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl#sha256=0ab0a9c4ffb1a6d95ef519fe4247dba8eb6b18ad93999f76b7f657039acabd47"; PILLOW_WHEEL="https://files.pythonhosted.org/packages/c8/98/766667a4be768150a202836acd9fad19c06824ca86c4286d3cf6b274964e/pillow-12.3.0-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl#sha256=bcb46e2f9feff8d06323983bd83ed00c201fdcab3d74973e7072a889b3979fcd" ;; \
         3.12/amd64) NUMPY_WHEEL="https://files.pythonhosted.org/packages/99/60/14115e6364fa676c5397c2ad3004e527e9aa487abf5d0706ec81bbd08529/numpy-2.4.6-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=90f9849678c75fe7afa2d348ac842c168b0a4d3d61919687216dfc547976d853"; PILLOW_WHEEL="https://files.pythonhosted.org/packages/84/21/a35af28dcc61f37ed850a2d64c65c701321dfbf25085e469d5559360cbbf/pillow-12.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=78cb2c6865a35ab8ff8b75fd122f6033b92a62c82801110e48ddd6c936a45d91" ;; \
         3.12/arm64) NUMPY_WHEEL="https://files.pythonhosted.org/packages/c9/c6/50a46a6205feba2343f1d6d17438107c5dc491ed1c736e6ea68689fd906b/numpy-2.4.6-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl#sha256=5f9fb9157b4ce2971008323afe46053787b526ef624fea915b261468a8421a0f"; PILLOW_WHEEL="https://files.pythonhosted.org/packages/25/27/ac8f99618ffd3dde21db0f4d4b1d2ab00c0880595bfd17df103f7f39fd0c/pillow-12.3.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl#sha256=d9c7f76c0673154f044e9d78c8655fb4213f6ca31a836df48b40fe5d187717b9" ;; \
         *) echo "unsupported OVRTX runtime target: Python ${PYTHON_MINOR}, ${TARGETARCH}" >&2; exit 1 ;; \
       esac \
    && OVRTX_LOCK=/app/repo/src/usd_core/render/pylock.ovrtx-runtime.toml \
    && python3 -c 'import sys,tomllib; data=tomllib.load(open(sys.argv[1], "rb")); allowed={artifact["url"]+"#sha256="+artifact["hashes"]["sha256"] for package in data["packages"] for artifact in package.get("wheels", [])}; missing=set(sys.argv[2:])-allowed; assert not missing, f"OVRTX wheel references absent from runtime lock: {sorted(missing)}"' \
       "$OVRTX_LOCK" "$OVRTX_WHEEL" "$NUMPY_WHEEL" "$PILLOW_WHEEL" \
    && /opt/ovrtx_venv/bin/pip install --no-cache-dir --no-deps \
       "ovrtx @ ${OVRTX_WHEEL}" \
       "ovstage @ ${OVSTAGE_WHEEL}" \
       "warp-lang @ ${WARP_WHEEL}" \
       "numpy @ ${NUMPY_WHEEL}" \
       "pillow @ ${PILLOW_WHEEL}" \
    # Prove every worker import and exact package version before recording
    # readiness. The marker binds this qualified native profile plus the lock
    # digest, which also covers numpy and pillow.
    && /opt/ovrtx_venv/bin/python -c "import importlib.metadata as m; import ovrtx, ovstage, warp, numpy; from PIL import Image; assert {name: m.version(name) for name in ('ovrtx', 'ovstage', 'warp-lang', 'numpy', 'pillow')} == {'ovrtx': '0.4.1.364340', 'ovstage': '0.1.1.355824', 'warp-lang': '1.16.0', 'numpy': '2.4.6', 'pillow': '12.3.0'}" \
    # Generate the exact canonical profile-and-lock marker usd_core validates.
    # OVRTX_RUNTIME_LOCK resolves to the installed package's immutable copy.
    && python3 -c "from usd_core.render.ovrtx import OVRTX_RUNTIME_LOCK, _readiness_marker_body; print(_readiness_marker_body(OVRTX_RUNTIME_LOCK), end='')" \
       > /opt/ovrtx_venv/.usd-cli-ovrtx-ready \
    # ovphysx (POST /physics/simulate) lives in its own venv: it bundles its own OpenUSD
    # and cannot coexist with usd-core in one process. Install the exact artifacts in
    # usd_core's shipped, architecture-specific physics-agent lock, probe a real CPU
    # solver, then bind the v2 readiness marker to both that lock and this venv's Python.
    && python3 -m venv /opt/ovphysx_venv \
    && case "${PYTHON_MINOR}/${TARGETARCH}" in \
         3.11/amd64) PHYSX_LOCK=/app/repo/src/usd_core/pylock.ovphysx-runtime.py311.toml; OVPHYSX_WHEEL="https://files.pythonhosted.org/packages/0d/34/0052e734c9cfdf1c8d67772a7fa605d1984e4b7e732abd1acd205e371755/ovphysx-0.4.13-py3-none-manylinux_2_35_x86_64.whl#sha256=4bc5bd9976529f86248826916a12243813833607131ed17149f976262f04dec6"; NUMPY_PHYSX_WHEEL="https://files.pythonhosted.org/packages/cf/c5/9fcb7e0e69cef59cf10c746b84f7d58b08bc66a6b7d459783c5a4f6101a6/numpy-2.4.4-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=df3775294accfdd75f32c74ae39fcba920c9a378a2fc18a12b6820aa8c1fb502" ;; \
         3.11/arm64) PHYSX_LOCK=/app/repo/src/usd_core/pylock.ovphysx-runtime.py311.aarch64.toml; OVPHYSX_WHEEL="https://files.pythonhosted.org/packages/b8/46/cb1eea38ab93554d2fc32a9209c70b16327fcabc2125236d70e910c6a630/ovphysx-0.4.13-py3-none-manylinux_2_35_aarch64.whl#sha256=b65be30c0bb535b830a3215bc89367b2abd0bfa78b45a3acd82270ea57b39d1e"; NUMPY_PHYSX_WHEEL="https://files.pythonhosted.org/packages/68/62/63417c13aa35d57bee1337c67446761dc25ea6543130cf868eace6e8157b/numpy-2.4.4-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl#sha256=8a87ec22c87be071b6bdbd27920b129b94f2fc964358ce38f3822635a3e2e03d" ;; \
         3.12/amd64) PHYSX_LOCK=/app/repo/src/usd_core/pylock.ovphysx-runtime.toml; OVPHYSX_WHEEL="https://files.pythonhosted.org/packages/0d/34/0052e734c9cfdf1c8d67772a7fa605d1984e4b7e732abd1acd205e371755/ovphysx-0.4.13-py3-none-manylinux_2_35_x86_64.whl#sha256=4bc5bd9976529f86248826916a12243813833607131ed17149f976262f04dec6"; NUMPY_PHYSX_WHEEL="https://files.pythonhosted.org/packages/0a/0d/0e3ecece05b7a7e87ab9fb587855548da437a061326fff64a223b6dcb78a/numpy-2.4.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=81f4a14bee47aec54f883e0cad2d73986640c1590eb9bfaaba7ad17394481e6e" ;; \
         3.12/arm64) PHYSX_LOCK=/app/repo/src/usd_core/pylock.ovphysx-runtime.aarch64.toml; OVPHYSX_WHEEL="https://files.pythonhosted.org/packages/b8/46/cb1eea38ab93554d2fc32a9209c70b16327fcabc2125236d70e910c6a630/ovphysx-0.4.13-py3-none-manylinux_2_35_aarch64.whl#sha256=b65be30c0bb535b830a3215bc89367b2abd0bfa78b45a3acd82270ea57b39d1e"; NUMPY_PHYSX_WHEEL="https://files.pythonhosted.org/packages/7f/37/eed308a8f56cba4d1fdf467a4fc67ef4ff4bf1c888f5fc980481890104b1/numpy-2.4.4-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl#sha256=f9e75681b59ddaa5e659898085ae0eaea229d054f2ac0c7e563a62205a700121" ;; \
       esac \
    && PACKAGING_WHEEL="https://files.pythonhosted.org/packages/ec/1a/610693ac4ee14fcdf2d9bf3c493370e4f2ef7ae2e19217d7a237ff42367d/packaging-23.2-py3-none-any.whl#sha256=8c491190033a9af7e1d931d0b5dacc2ef47509b34dd0de67ed209b5203fc88c7" \
    && python3 -c 'import sys,tomllib; data=tomllib.load(open(sys.argv[1], "rb")); artifacts=[package.get("archive") for package in data["packages"]]+[artifact for package in data["packages"] for artifact in package.get("wheels", [])]; allowed={artifact["url"]+"#sha256="+artifact["hashes"]["sha256"] for artifact in artifacts if artifact}; missing=set(sys.argv[2:])-allowed; assert not missing, f"OvPhysX wheel references absent from runtime lock: {sorted(missing)}"' \
       "$PHYSX_LOCK" "$OVPHYSX_WHEEL" "$NUMPY_PHYSX_WHEEL" "$PACKAGING_WHEEL" \
    && /opt/ovphysx_venv/bin/pip install --no-cache-dir --no-deps \
       "ovphysx @ ${OVPHYSX_WHEEL}" \
       "numpy @ ${NUMPY_PHYSX_WHEEL}" \
       "packaging @ ${PACKAGING_WHEEL}" \
    && /opt/ovphysx_venv/bin/python -c \
       "from ovphysx import PhysX; physics = PhysX(device='cpu'); physics.release()" \
    && python3 -c "from pathlib import Path; from usd_core.physics_runtime import _ovphysx_readiness_marker_body; print(_ovphysx_readiness_marker_body(Path('$PHYSX_LOCK'), Path('/opt/ovphysx_venv/bin/python')), end='')" \
       > /opt/ovphysx_venv/.usd-cli-ovphysx-ready \
    && python3 -m pip list --format=json > /app/python-inventory.json \
    && rm -rf /tmp/wheels

WORKDIR /app/service_app
COPY apps/ovrtx_rendering_api/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Keep application code and prebuilt environments root-owned. OVRTX resolves
# its NVIDIA shader cache through a package-relative path, so redirect only that
# mutable subtree into the service user's home instead of changing ownership of
# the complete native environment.
RUN useradd --create-home --uid 10001 dsc \
    && install -d -o dsc -g dsc /home/dsc/.cache/ovrtx-nv-shadercache \
    && install -d -m 1777 /tmp/.X11-unix \
    && OVRTX_BIN="$(/opt/ovrtx_venv/bin/python -c \
        'from pathlib import Path; import ovrtx; print(Path(ovrtx.__file__).resolve().parent / "bin")')" \
    && mkdir -p "$OVRTX_BIN/cache" \
    && rm -rf "$OVRTX_BIN/cache/nv_shadercache" \
    && chown -R dsc:dsc "$OVRTX_BIN/cache" \
    && find "$OVRTX_BIN/cache" -type d -exec chmod 700 {} + \
    && find "$OVRTX_BIN/cache" -type f -exec chmod 600 {} + \
    && ln -s /home/dsc/.cache/ovrtx-nv-shadercache \
        "$OVRTX_BIN/cache/nv_shadercache"

ENV HOME=/home/dsc \
    XDG_CACHE_HOME=/home/dsc/.cache

# Prove the non-root runtime user can import both native environments and write
# through OVRTX's redirected cache before switching users. The readiness markers
# alone only attest that provisioning finished.
RUN su -s /bin/sh -c \
      '/opt/ovrtx_venv/bin/python -c "import ovrtx, ovstage, warp, numpy; from PIL import Image"; cache=$(/opt/ovrtx_venv/bin/python -c "from pathlib import Path; import ovrtx; print(Path(ovrtx.__file__).resolve().parent / \"bin\" / \"cache\" / \"nv_shadercache\")"); test -w "$cache"; touch "$cache/.write-probe"; rm "$cache/.write-probe"' dsc \
    && su -s /bin/sh -c \
      '/opt/ovphysx_venv/bin/python -c "from ovphysx import PhysX"' dsc

USER 10001:10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=12 \
  CMD python3 -c "import os, urllib.request; request = urllib.request.Request('http://127.0.0.1:8000/ready', headers={'Authorization': 'Bearer ' + os.environ['OVRTX_API_KEY']}); urllib.request.urlopen(request, timeout=5).read()" || exit 1
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
