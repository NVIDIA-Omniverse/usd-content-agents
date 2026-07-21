# Texture Gen Step1X Service

Optional Texture Variation API service for Step1X-backed texture generation.

This package uses the shared `apps.texture_gen_service_common` FastAPI harness
and exposes:

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/texture-variations` | Submit a texture generation job |
| `GET` | `/v1/texture-variations/{job_id}` | Query job status |
| `DELETE` | `/v1/texture-variations/{job_id}` | Cancel a job |
| `GET` | `/health` | Health check |

## Source And Asset Boundary

The service boundary is `Step1XRunner` in `backend.py`. The default
`ExternalStep1XRunner` invokes an `edit_texture.py`-style CLI in a separate
runtime process; the FastAPI service does not import or initialize Step1X at
startup.

The public 0.5 source release ships the API adapter and service container, but
does not ship a managed Step1X runtime template, downloader/setup package,
model checkpoints, or runtime cache layout. Mount an operator-managed runtime
with `TEXTURE_STEP1X_RUNTIME_DIR` after the runtime and assets have been
reviewed and deployed in your environment.

Model weights, checkpoints, generated model caches, credentials, and run
artifacts are not committed to this repository. They must be downloaded into
ignored cache/model paths or mounted from an operator-managed volume after the
applicable security and legal review.

## Operational Ownership

This repository owns the Texture Variation API service: FastAPI entrypoint,
Dockerfile, compose files, request/response schema, Step1X adapter, health
checks, smoke script, and unit coverage. The public source artifact expects the
runtime to be operator-managed: a deployment mounts an external runtime and sets
`TEXTURE_STEP1X_RUNTIME_DIR` / `TEXTURE_STEP1X_EDIT_SCRIPT`, or replaces the
default command with `TEXTURE_STEP1X_COMMAND_TEMPLATE`.

`GET /health` reports the management model in
`capabilities.external_runtime`, including the configured runtime directory,
edit script, command-template mode, model/cache paths, asset-validation mode,
and `weights_policy=downloadable_not_committed`.

## Runtime Setup

For standalone/custom deployments, configure the service container with:

| Variable | Description |
|---|---|
| `TEXTURE_STEP1X_RUNTIME_DIR` | Path to an external Step1X checkout or runtime environment mounted into the service container |
| `TEXTURE_STEP1X_EDIT_SCRIPT` | Optional path to the external edit script; defaults to `$TEXTURE_STEP1X_RUNTIME_DIR/edit_texture.py` |
| `TEXTURE_STEP1X_PYTHON` | Optional Python executable for the runtime; the compose package defaults to `/opt/texture-editing/.venv_gen/bin/python` |
| `TEXTURE_STEP1X_MODEL_DIR` | Optional path to Step1X model weights or model cache; checked when configured |
| `TEXTURE_STEP1X_CACHE_DIR` | Optional runtime cache path |
| `TEXTURE_OUTPUT_DIR` | Optional generated artifact output directory |
| `TEXTURE_STEP1X_MAX_WORKERS` | Optional worker count, default `1` |
| `TEXTURE_STEP1X_TIMEOUT_SEC` | Optional external command timeout, default `3600` |
| `TEXTURE_STEP1X_SKIP_MA` | Whether to pass `--skip-ma` by default; default `true` |
| `TEXTURE_STEP1X_REQUIRE_UPSCALER` | Require upscaler readiness in `/health`; default `false` |
| `TEXTURE_STEP1X_VALIDATE_ASSETS` | Whether to validate local USD target/material/albedo scope before launch, default `true` |
| `TEXTURE_STEP1X_REQUIRED_EXECUTABLES` | Space- or comma-separated executables that must be available on `PATH` before the service accepts jobs; defaults to `uv` for env-based service startup |
| `TEXTURE_STEP1X_GPU_DEVICE` | GPU device ID for the standalone Step1X compose files, default `0` |
| `TEXTURE_STEP1X_EXTRA_ARGS` | Optional extra CLI args appended to the default command |
| `TEXTURE_STEP1X_COMMAND_TEMPLATE` | Optional full command template replacing the default edit script command |
| `TEXTURE_STEP1X_LD_LIBRARY_PATH` | Optional CUDA library path for operator-mounted runtimes; Compose defaults to the reference `.venv_gen` torch/CUDA wheel layout |
| `TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS` | Enable readiness-healthcheck runtime import probe for torch/CuPy/NVRTC when `healthcheck.py` checks `/health`; package Docker health uses `/livez` |
| `TEXTURE_STEP1X_HEALTHCHECK_TIMEOUT` | Healthcheck timeout, default `180s`; relevant to cold runtime imports only for readiness healthchecks |
| `TEXTURE_UPSCALER_BACKEND` | Optional upscaler backend for `custom_parameters.upscale=true`; default `swin2sr`, legacy `ncnn-vulkan` is also supported |
| `TEXTURE_SWIN2SR_DEVICE` | Swin2SR device policy, default `auto` (`cuda` when available, otherwise CPU) |
| `TEXTURE_SWIN2SR_TILE_SIZE` | Swin2SR input tile size for large textures, default `512`; set `0` to disable tiling |
| `TEXTURE_SWIN2SR_TILE_OVERLAP` | Swin2SR input tile overlap in pixels, default `16` |
| `TEXTURE_REALESRGAN_VK_ICD` | Vulkan ICD file for the legacy Real-ESRGAN ncnn-vulkan backend |
| `TEXTURE_REALESRGAN_GPU_ID` | Container-visible legacy ncnn-vulkan GPU id, default `0` |
| `TEXTURE_REALESRGAN_ALLOW_LLVMPIPE_FALLBACK` | Allow CPU llvmpipe fallback for legacy ncnn-vulkan when GPU Vulkan fails, default `false` |
| `TEXTURE_REALESRGAN_TIMEOUT_SECONDS` | Per legacy ncnn-vulkan subprocess timeout, default `300` |
| `NVIDIA_DRIVER_CAPABILITIES` | NVIDIA container driver capabilities for GPU runtimes; compose defaults to `compute,utility,graphics` |

When required runtime paths are unset or do not exist, `/health` returns
`ready=false` with a not-ready status instead of crashing. `TEXTURE_STEP1X_MODEL_DIR`
is optional because some runtimes resolve models through their own cache. The
health payload also reports `capabilities.external_runtime.required_executables`;
missing entries make the service not ready before a request can fail inside the
external runtime.

For real Step1X evidence, do not treat API liveness alone as proof that the
runtime can execute a job. Docker Compose health uses `/livez` for cheap
container liveness; query `/health` for readiness. That readiness payload
reports the mounted runtime path, configured Python executable, required
executables, optional Material Anything/upscaler probes, CUDA library paths, and
asset diagnostics. Make sure the Python executable and CUDA library paths
resolve inside the service container.

The service image includes `uv` so command-template and operator-mounted
runtimes can use it when extracting or exporting USD assets. If a custom
runtime uses a different helper, mount it into the service container and
include it in `TEXTURE_STEP1X_REQUIRED_EXECUTABLES`, or provide a
`TEXTURE_STEP1X_COMMAND_TEMPLATE` that does not require the helper.

For operator-managed runtimes, the container must see more than
`edit_texture.py`: the runtime Python interpreter/environment, any auxiliary
source trees used by that environment, and the model/cache directories it
imports at runtime. Prefer packaging those into the mounted runtime directory or
a purpose-built runtime image. If that is not yet possible, mount the additional
read-only paths explicitly and append them with `TEXTURE_STEP1X_EXTRA_PYTHONPATH`;
mount model caches under a stable container path and set `HF_HOME` /
runtime-specific cache variables. Writable outputs and cache directories should
live on a volume owned by the service user, such as the shared session volume
used by the texture-agent compose overlay.

When the mounted runtime, conda environment, or model cache lives under
host-user-only directories, run the container as the host user, for example
`--user "$(id -u):$(id -g)"`. The service source in the image is world-readable,
while writable caches default to `/work/cache` for CUDA, CuPy, Matplotlib,
Numba, PyTorch, Triton, `uv`, and XDG. Keep those defaults or override them to
another writable volume when the mounted runtime imports packages that compile
kernels or cache code at import time.

The default command passes:

```bash
python edit_texture.py \
  --usd <source_asset> \
  --prompt <text_prompt> \
  --output <job_output_dir> \
  --strength <strength> \
  --seed <seed> \
  --resolution <texture_size>
```

The public Compose package uses the fast albedo-only path by default.
`custom_parameters` may override/add `steps`, `guidance`, `ma_steps`, `gpu`,
`skip_material_anything`, `upscale`, and `debug`.

## Material Anything And Upscaling

The public Compose package runs the fast albedo-only path by default:
`TEXTURE_STEP1X_SKIP_MA=true` disables Material Anything and
`TEXTURE_STEP1X_REQUIRE_UPSCALER=false` keeps Swin2SR out of readiness. Enable
the full PBR path only when the request needs ORM output:

```json
{
  "configuration": {
    "custom_parameters": {
      "skip_material_anything": false,
      "ma_steps": 10,
      "gpu": 0,
      "upscale": true
    }
  }
}
```

With Material Anything enabled, `/health` requires the source and model assets
before reporting `ready=true`:

- `third_party/MaterialAnything/scripts/generate_texture_pbr_3d.py`
- `third_party/MaterialAnything/pretrained_models/material_estimator`
- `third_party/MaterialAnything/pretrained_models/material_refiner`
- `third_party/MaterialAnything/models/ControlNet/models/control_sd15_depth.pth`

The health response always includes `capabilities.material_anything` with
`enabled_by_default`, `ready`, `missing`, and `paths`, so operators can
distinguish "disabled for fast albedo-only validation" from "enabled but
missing model assets."

When Material Anything succeeds, `edit_texture.py` keeps the Step1X albedo and
uses Material Anything roughness/metallic outputs to write `final_orm.png`.
The service exposes that as `generated_textures.orm` and `maps.orm` with
`packing=occlusion_roughness_metallic`. Because Material Anything emits bump
intermediates rather than a tangent-space normal map, scoped edits preserve the
source material's normal as `final_normal.png` when one is available. Albedo-only
responses remain valid and continue to report
`metadata.degraded_channels=["normal", "orm"]` plus a `STEP1X_MAPS_DEGRADED`
diagnostic.

The default `upscale` implementation uses Apache-2.0 CAIDAS Swin2SR checkpoints
through Transformers/PyTorch: `caidas/swin2SR-classical-sr-x2-64` for 2x
restoration and `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` for 4x
restoration. The legacy Real-ESRGAN `ncnn-vulkan` binary remains available via
`TEXTURE_UPSCALER_BACKEND=ncnn-vulkan`, but it requires a Vulkan-capable
container runtime and host driver stack. For service/API smoke tests that only
need to validate UV-aware Step1X texture generation, set
`custom_parameters.upscale=false`.
Set `TEXTURE_STEP1X_REQUIRE_UPSCALER=true` only when requests are expected to
use `custom_parameters.upscale=true`; otherwise missing optional upscaler
dependencies are reported in `capabilities.upscaler` without blocking
albedo-only readiness.

## Run

From the repository root with the project environment activated:

```bash
uvicorn apps.texture_gen_step1x_service.app:app --port 8000
```

Example health check:

```bash
curl http://localhost:8000/health
```

## Docker Compose

The standalone Step1X compose file starts only the API adapter. It requires a
custom runtime and shared asset root that you provide:

```bash
TEXTURE_STEP1X_HOST_RUNTIME=/path/to/texture-editing-runtime \
TEXTURE_STEP1X_PYTHON=/opt/texture-editing/.venv_gen/bin/python \
TEXTURE_SHARED_HOST_ROOT=/path/to/texture_gen_shared \
docker compose -f apps/texture_gen_step1x_service/docker-compose.yml up --build
```

Run the smoke script against a running service:

```bash
python apps/texture_gen_step1x_service/scripts/smoke.py \
  --endpoint http://localhost:8000 \
  --request apps/texture_agent/tests/fixtures/step1x_service_requests/request_cleaning_bucket_opaque_metal.json
```

Run the MA/full-PBR smoke against an MA-enabled runtime:

```bash
python apps/texture_gen_step1x_service/scripts/smoke.py \
  --endpoint http://localhost:8000 \
  --request apps/texture_agent/tests/fixtures/step1x_service_requests/request_cleaning_bucket_opaque_metal_ma.json \
  --require-orm
```

When validating through texture-agent, prefer the integrated overlay at
`apps/texture_agent_service/docker-compose.step1x.yml`. It pins Step1X with
`TEXTURE_STEP1X_GPU_DEVICE` and OVRTX with `OVRTX_GPU_DEVICE` (`0` and `1` by
default), mounts the texture-agent session volume into Step1X and OVRTX at the
same container path so `file://` request assets and generated USDs are visible
across services, and routes texture-agent to `http://texture-gen-step1x:8000`
through Docker DNS. The overlay does not create or download the runtime.

## Adapter Contract

`Step1XBackend.generate()` normalizes the shared Texture Variation API request
into `Step1XRunRequest`, including:

- `prompt`
- `seed`
- `strength`
- `texture_size`
- `source_asset_uri`
- local `source_asset_path` and selected `source_albedo_path` when validation is enabled
- `target`
- configured runtime/model/cache paths

Before launch, the default backend validates local `file://` assets, resolves
`target.material_path` / `target.material_name` / `target.prim_paths`, rejects
unsupported `per_prim` scope, and extracts the selected material's albedo path.
It also validates scoped mesh UV primvars and fails before model launch with
`STEP1X_UV_INVALID` when existing UVs have invalid topology or non-finite
coordinates. This prevents the service from silently accepting first-mesh-only
behavior or handing known-bad UV data to Step1X.

Runner implementations return `Step1XRunResult`. Albedo-only output is valid:
missing normal and ORM maps are represented with `null` generated texture fields,
`metadata.degraded_channels`, and a warning diagnostic. For scoped UV-preserving
edits, the default backend reuses the source normal map when the runner omits a
normal map and the source material has one.
