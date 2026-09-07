# OVRTX Rendering API

USD rendering service using the [OVRTX](https://github.com/NVIDIA-Omniverse/ovrtx) local RTX renderer. Provides a drop-in REST API compatible with the Kit-based rendering service, so upstream agents (material, physics, texture) can use either backend interchangeably.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/live` | Package-owned usd-cli protocol-v3 identity and upload limits |
| `GET` | `/health` | Health status including GPU initialization state |
| `POST` | `/render` | Render a USD file and return base64-encoded images |
| `POST` | `/render/upload` | Render the bounded multipart USDZ contract used by package-owned usd-cli |

The render endpoint accepts the same request body as the Kit-based `rendering-api` and returns the same V1 response format (`images[frame][camera][sensor] = base64`). See [`openapi.yaml`](openapi.yaml) for the full schema.

The image also installs `usd-cli` from the same checkout and exposes that
package's exact remote protocol in standalone and multi-GPU dispatcher modes.
`/live` reports protocol v3; `/render/upload` accepts the protocol's gzip/none
multipart floor and returns the renderer-reported mode, update count, and
active AOV with every image.

## Quick Start

```bash
# From repository root
uv pip install -e ".[telemetry]" -e apps/usd_cli -e apps/ovrtx_rendering_api

# Run the service
uvicorn service.main:app --host 0.0.0.0 --port 8001
```

The service listens on port 8001 by default. Visit `/docs` for the Swagger UI.
On a cold start, GPU warm-up commonly takes around 5 minutes; `/health`
continues to return `gpu_initialized=false` until that warm-up completes.

## Docker

The service is packaged as a GPU-enabled container and has a standalone Compose
file for shared render-service deployments:

```bash
OVRTX_RENDER_MODE=pt docker compose -f apps/ovrtx_rendering_api/docker-compose.yml up --build
```

The compose file publishes host port `8001` to container port `8000` by default.
Override the host port with `OVRTX_HOST_PORT`.

Raw Docker is also supported:

```bash
# Build the image
docker build -f apps/ovrtx_rendering_api/Dockerfile -t ovrtx-rendering-api .

# Run with GPU access
docker run --rm --gpus all \
  -e OVRTX_RENDER_MODE=pt \
  -e OVRTX_DAEMON_RENDER_TIMEOUT=900 \
  -p 8001:8000 \
  ovrtx-rendering-api
```

The container health check now waits for `/health` to report
`gpu_initialized=true`, so a fresh `docker run` may stay in `starting` state
for several minutes before it becomes healthy.

### Service Configuration

General service knobs are available for both single-worker and dispatcher mode:

| Variable | Default | Description |
|---|---:|---|
| `OVRTX_MAX_BODY_BYTES` | `73400320` | Maximum raw or compressed file-part bytes accepted by `/render/upload`. The service enforces a separate receive-layer ceiling before multipart parsing, with 1 MiB reserved for bounded form metadata and framing. |
| `OVRTX_ZIP_MAX_UNCOMPRESSED_BYTES` | `2147483648` | Maximum uncompressed ZIP bundle size accepted by the service. Raise only for trusted large-scene deployments. |
| `OVRTX_HTTP_MAX_DOWNLOAD_BYTES` | `2147483648` | Maximum decoded bytes streamed from an HTTP(S) USD or bundle response. Content-Length and actual streamed bytes are both enforced. |
| `OVRTX_S3_ALLOWED_BUCKETS` | empty | Exact bucket names allowed for client-supplied S3 URLs, separated by commas or whitespace. Empty rejects all S3 intake before the service uses AWS credentials. |
| `OVRTX_DAEMON_MAX_RENDERS` | `64` | Recycle the isolated renderer process before the next request after this many completed render commands. `0` disables the count guard. |
| `OVRTX_DAEMON_MAX_RSS_BYTES` | `25769803776` | Recycle the isolated renderer process before the next request after its RSS reaches this many bytes (24 GiB by default). `0` disables the RSS guard; unsupported platforms continue to use the count guard. |

Submitted USD and ZIP-bundle layers are inspected before composed-stage open.
Every authored sublayer, reference, payload, texture, and other asset path must
be relative, resolve inside the request's temporary intake root, and identify
an existing file or USDZ member. Bounded bare MDL module tokens such as
`OmniPBR.mdl` remain renderer-resolved rather than host-file dependencies.
Resolver URIs, absolute paths, nested package identifiers, traversal, missing
dependencies, and symlink escapes are rejected rather than allowing the USD
resolver to read host-local content or silently render an incomplete scene.

The renderer remains single-flight. Recycling happens under the same render
lock and never interrupts an in-flight request: the request that crosses a
limit completes normally, then the following request pays the cold-start cost
for a fresh daemon. The count guard gives every deployment a deterministic
bound; the RSS guard catches one unusually expensive scene sooner. `/health`
reports the current daemon PID, completed-render count, RSS, recycle count, and
last/pending recycle reasons so operators can tune the thresholds from evidence.

The image runs as the non-root `renderer` user (`10001:10001`). Application
code and the prebuilt OVRTX virtual environment remain root-owned; runtime
state is limited to `/tmp` and the renderer user's home/cache directory,
including OVRTX's writable NVIDIA shader cache.

### Non-root GPU Smoke

Use the non-root smoke script on a GPU host to validate Xvfb startup, NVIDIA
device access, OVRTX daemon warm-up, and a minimal render request:

```bash
apps/ovrtx_rendering_api/scripts/smoke_non_root_container.sh
```

The script builds the image, runs it with `no-new-privileges` and all Linux
capabilities dropped, waits for `gpu_initialized=true`, then posts a 64x64
data-URI render request using the bundled smoke cube scene.

### Multi-GPU Dispatcher

On a multi-GPU host, one container can expose a single public `/render`
endpoint while launching one private OVRTX worker process per GPU:

```bash
OVRTX_GPU_WORKERS=2 docker compose -f apps/ovrtx_rendering_api/docker-compose.yml up --build
```

`OVRTX_GPU_WORKERS` accepts either a worker count (`2` -> GPUs `0` and `1`) or
an explicit comma-separated GPU id list (`0,1`). Leave it unset for the legacy
single-worker service; set it to `0` to disable dispatcher mode explicitly.

The compose file reserves all visible GPUs by default so `OVRTX_GPU_WORKERS=2`
works without a second variable. Set `OVRTX_GPU_COUNT=1` if you need the legacy
single-GPU reservation behavior.

Dispatcher workers bind only to private localhost ports inside the container,
starting at `OVRTX_WORKER_PORT_BASE` (`8100` by default). Worker warm-up runs in
parallel, and `/render` routes only to workers whose health reports
`gpu_initialized=true`. If all workers are busy, requests wait up to
`OVRTX_WORKER_QUEUE_TIMEOUT` seconds (`60` by default) before returning an
exception response. Each worker is single-flight, so long renders can saturate
all healthy workers for minutes; tune the queue timeout to match the expected
render duration and client retry behavior.

For a single standalone OVRTX REST endpoint, configure dataset builders with
`max_concurrent_requests: 1` and set
`WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS=1` when multiple pipelines or rendering
paths share the same client process. Raise these values only when dispatcher
`/health` reports the corresponding number of ready GPU workers. An HTTP 200
health response or `gpu_initialized=true` does not prove render integrity;
qualification must include a non-empty render and a bounded multi-request run.

Other dispatcher knobs are available for operations:

| Variable | Default | Description |
|---|---:|---|
| `OVRTX_WORKER_HEALTH_INTERVAL` | `5` | Seconds between private worker health polls. |
| `OVRTX_WORKER_REQUEST_TIMEOUT` | `3600` | HTTP timeout for parent-to-worker render calls. |
| `OVRTX_WORKER_WARMUP_STAGGER_SECONDS` | `0` | Optional delay between worker process starts. |
| `OVRTX_WORKER_RESTART_COOLDOWN` | `10` | Seconds before restarting exited or unhealthy workers. |

Dispatcher mode assumes a single parent uvicorn process. Do not combine
`OVRTX_GPU_WORKERS` with uvicorn's `--workers N`; each parent process would try
to bind the same private worker ports. The validated Docker path starts one
parent process and lets that parent supervise the per-GPU worker processes.
The workers share the container's Xvfb display (`:99`), which was validated on
the two-L40 `horde-content-agents` host.

In dispatcher mode, `/health` includes aggregate and per-worker capacity:

```json
{
  "status": "healthy",
  "protocol_version": 3,
  "gpu_initialized": true,
  "daemon_pid": 123,
  "daemon_completed_renders": 12,
  "daemon_rss_bytes": 8589934592,
  "daemon_recycle_count": 1,
  "daemon_last_recycle_reason": "completed_render_limit",
  "daemon_pending_recycle_reason": null,
  "ready_workers": 2,
  "total_workers": 2,
  "workers": [
    {"gpu": "0", "port": 8100, "ready": true, "busy": false},
    {"gpu": "1", "port": 8101, "ready": true, "busy": false}
  ]
}
```

## Example Request

```bash
curl -X POST http://localhost:8001/render \
  -H "Content-Type: application/json" \
  -d '{
    "url": "file:///data/scene.usd",
    "render_settings": {
      "camera_paths": ["/Camera"],
      "frame_range": {"start": 0, "end": 0},
      "camera_parameters": {"width": 1024, "height": 1024}
    }
  }'
```

## Requirements

- NVIDIA GPU with driver 535+
- CUDA runtime (provided by the container base image)
- USD asset accessible via URL (file, http, or s3)

## Project Structure

```
ovrtx_rendering_api/
├── service/        # FastAPI app, render pipeline, models
├── client/         # Optional Python client for programmatic use
├── tests/          # Unit and integration tests
├── Dockerfile      # Container build
├── openapi.yaml    # API specification
└── pyproject.toml  # Install metadata
```
