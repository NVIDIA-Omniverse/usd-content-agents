# Material Agent Service - Docker Compose Deployment

## Quick Start

### Prerequisites

- Docker Compose **v2.24.4+** (required for the `!override` tags and
  `env_file: required: false` long-form syntax used by the Compose files)
- Default OVRTX topology: RTX-capable NVIDIA GPU with 48GB+ VRAM (e.g., L40, L40S, RTX6000 Ada)
- WSL2 WARP topology: CUDA-capable NVIDIA GPU exposed to WSL2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- VLM provider API key (OpenAI, Anthropic, Gemini, or NVIDIA)

### Setup

```bash
# From the repo root -- create .env with your VLM provider key.
# The compose file reads this .env via env_file: ../../.env.
echo 'OPENAI_API_KEY=sk-...' > .env

# Build and start (main service + OVRTX rendering)
docker compose -f apps/material_agent_service/docker-compose.yml up --build
```

That's it. No NGC login needed -- the OVRTX rendering API builds from source.
Expect the bundled `ovrtx-rendering-api` sidecar to spend roughly 5 minutes in
GPU warm-up on a cold start. During that time `/health` on port `8001` returns
`gpu_initialized=false`, and `material-agent-service` will wait to start.

### WSL2 with in-process WARP

Local OVRTX is not supported inside WSL2. Use the WARP overlay, which removes
the OVRTX startup dependency, keeps that sidecar inactive, and assigns one GPU
to `material-agent-service`:

```bash
docker compose --env-file .env \
  -f apps/material_agent_service/docker-compose.yml \
  -f apps/material_agent_service/docker-compose.warp.yml up --build
```

The service image already installs the qualified WARP/Newton dependency set.
Set `MA_WARP_GPU_DEVICE_ID` before invoking Compose to select a GPU other than
device 0. The overlay works either with Docker Engine and Compose installed
directly inside WSL2 (without Docker Desktop), or with Docker Desktop's WSL2
backend.

### Access

- **Health**: http://localhost:8000/health
- **API Docs** (Swagger UI): http://localhost:8000/docs
- **Rendering API Health**: http://localhost:8001/health

For Brev deployment planning, see [`brev.md`](brev.md).

## Services

| Service | Port | GPU | Build | Description |
|---|---|---|---|---|
| material-agent-service | 8000 | No; 1x with WARP overlay | From source | Main service (pipeline + REST API) |
| ovrtx-rendering-api | 8001 | 1x | From source | OVRTX-based USD rendering |
| vlm-nim (optional) | 8003 | 1x | NGC image | Local Cosmos Reason2 8B VLM |
| cluster-embedding-nim (optional) | 8004 | 1x | NGC image | Local Llama Nemotron VLM embeddings for prim clustering |

## Usage

```bash
# Main + rendering (default, no NGC needed)
docker compose -f apps/material_agent_service/docker-compose.yml up --build

# WSL2 + in-process WARP (no local OVRTX sidecar)
docker compose --env-file .env \
  -f apps/material_agent_service/docker-compose.yml \
  -f apps/material_agent_service/docker-compose.warp.yml up --build

# Add local VLM NIM (requires NGC login and 2+ GPUs)
printf '%s' "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' --password-stdin
docker compose \
  -f apps/material_agent_service/docker-compose.yml \
  -f apps/material_agent_service/docker-compose.multi-gpu.yml \
  --profile vlm up --build

# Add local embedding NIM for prim clustering (requires NGC login and 2+ GPUs)
printf '%s' "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' --password-stdin
docker compose \
  -f apps/material_agent_service/docker-compose.yml \
  -f apps/material_agent_service/docker-compose.cluster-embedding.yml \
  --profile cluster-embedding up --build

# Detached mode
docker compose -f apps/material_agent_service/docker-compose.yml up --build -d

# View logs
docker compose -f apps/material_agent_service/docker-compose.yml logs -f

# Stop
docker compose -f apps/material_agent_service/docker-compose.yml down

# Stop and remove session data
docker compose -f apps/material_agent_service/docker-compose.yml down -v
```

## VLM Provider Configuration

Set one of these in `.env` (or export before running):

| Provider | Environment Variable |
|---|---|
| NVIDIA (build.nvidia.com) | `NVIDIA_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |

Generated reference images use `MA_IMAGE_GEN_BACKEND`, which defaults to
`gemini` and requires `GOOGLE_API_KEY` or `GEMINI_API_KEY`. Set
`MA_IMAGE_GEN_BACKEND=openai` with `OPENAI_API_KEY`, or set
`MA_IMAGE_GEN_BACKEND=nim` with `NVIDIA_API_KEY`. For a no-auth local
OpenAI-compatible image endpoint, set `MA_IMAGE_GEN_BASE_URL` and explicitly set
`MA_IMAGE_GEN_API_KEY=not-used`.

When using the `vlm` profile (local Cosmos VLM NIM), set `NGC_API_KEY` for model weight download. The VLM NIM takes ~15 minutes to start on first run (model compilation).

## Prim Clustering Configuration

Image-based prim clustering is available through the service API but remains
off by default. Local Docker Compose uses NVIDIA image embeddings when callers
enable clustering. Use hosted NVIDIA embeddings with `NVIDIA_API_KEY`, or start
the optional local embedding sidecar below.

```bash
curl -X POST http://localhost:8000/pipeline \
  -F "usd_file=@scene.usd" \
  -F "user_email=user@example.com" \
  -F "enable_prim_clustering=true"
```

Compose defaults:

| Variable | Default | Description |
|---|---|---|
| `MA_CLUSTER_EMBEDDING_BACKEND` | `nim` | NVIDIA image embedding backend. |
| `MA_CLUSTER_EMBEDDING_MODEL` | `nvidia/llama-nemotron-embed-vl-1b-v2` | Hosted NVIDIA image embedding model. |
| `MA_CLUSTER_EMBEDDING_BASE_URL` | | Optional embedding endpoint URL. |
| `MA_CLUSTER_EMBEDDING_MAX_WORKERS` | `4` | Parallel embedding workers. |
| `MA_CLUSTER_EMBEDDING_BATCH_SIZE` | `50` | Embedding batch size. |
| `MA_CLUSTER_MIN_PRIMS` | `50` | Minimum prim count before clustering runs. |
| `MA_CLUSTER_MAX_SIZE` | `25` | Maximum prims per propagated representative prediction. |
| `MA_CLUSTER_EMBEDDING_API_KEY` | | Optional endpoint-scoped embedding API key. |

For hosted NVIDIA embeddings, add these to `.env`:

```bash
NVIDIA_API_KEY=nvapi-...
MA_CLUSTER_EMBEDDING_BACKEND=nim
MA_CLUSTER_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2
MA_CLUSTER_MAX_SIZE=25
```

The base compose topology remains one GPU for OVRTX rendering. Hosted `nim`
embeddings use network calls and the configured API key.

To self-host the embedding model as an optional sidecar, use the
`docker-compose.cluster-embedding.yml` overlay:

```bash
echo 'NGC_API_KEY=...' >> .env
printf '%s' "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' --password-stdin

docker compose \
  -f apps/material_agent_service/docker-compose.yml \
  -f apps/material_agent_service/docker-compose.cluster-embedding.yml \
  --profile cluster-embedding up --build
```

The overlay runs `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0`, routes
`MA_CLUSTER_EMBEDDING_BASE_URL` to
`http://cluster-embedding-nim:8000/v1`, and uses
`MA_CLUSTER_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2`. It pins rendering to GPU 0
and the embedding NIM to GPU 1 by default; set `MA_CLUSTER_NIM_GPU_DEVICE_ID=2` before
`docker compose` if you also enable the local VLM sidecar on GPU 1. The overlay
sets `shm_size` to `16gb` by default for Triton; override it with
`MA_CLUSTER_NIM_SHM_SIZE` if your host requires a different value.

## Resource Requirements

| Configuration | GPUs | CPU | Memory |
|---|---|---|---|
| Main + rendering (default) | 1 | 10 | 20G |
| WSL2 + WARP | 1 | 8 | 16G |
| + VLM NIM | 2 | 16 | 56G |
| + embedding NIM | 2 | 16 | 32G |

### GPU Notes

- OVRTX rendering needs 1 RTX-capable GPU with 48GB VRAM
- Local OVRTX is unsupported inside WSL2; use `docker-compose.warp.yml` there
- Local compose starts one OVRTX renderer, so the main service defaults
  `MA_MAX_ACTIVE_SESSIONS`, `MA_MAX_RENDER_NUM_WORKERS`, and
  `WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS` to `1`
- Prim clustering with the embedding sidecar needs 1 additional GPU
- VLM NIM needs 1 additional GPU with 48GB VRAM
- A100/H100-class GPUs are useful for VLM serving but should not be used as
  OVRTX render nodes because they lack RTX rendering support
- On 48GB GPUs (L40/L40S), VLM NIM uses `NIM_MAX_MODEL_LEN=131072` by default
- With the multi-GPU overlay, `docker exec vlm-nim nvidia-smi --query-gpu=count --format=csv,noheader`
  should print `1`. Any other value means the sidecar was not pinned to a
  single GPU.

## Startup Timeline

| Service | Ready |
|---|---|
| material-agent-service | ~1 min |
| ovrtx-rendering-api | ~5 min on cold start before `gpu_initialized=true` |
| vlm-nim | ~15 min (model compilation) |
| cluster-embedding-nim | Several minutes on first start while NIM downloads and compiles |

## Troubleshooting

### OVRTX rendering fails

```bash
# Check rendering API logs
docker logs ovrtx-rendering-api

# Verify GPU is visible
docker exec ovrtx-rendering-api nvidia-smi
```

If you see shader compilation messages, wait -- first-run compilation takes ~5 minutes.
The main service will not become reachable until the OVRTX sidecar health check
passes.

### VLM NIM KV cache error

If VLM NIM crashes with KV cache memory error on 48GB GPUs:

```bash
# Already set to 131072 by default in docker-compose.yml
# To override:
NIM_MAX_MODEL_LEN=65536 docker compose \
  -f apps/material_agent_service/docker-compose.yml \
  -f apps/material_agent_service/docker-compose.multi-gpu.yml \
  --profile vlm up
```

### Out of memory

Reduce resource limits in `docker-compose.yml` or allocate more GPUs:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # more GPUs
          capabilities: [gpu]
```

### Rebuild after code changes

```bash
docker compose -f apps/material_agent_service/docker-compose.yml up --build

# Force full rebuild (no cache)
docker compose -f apps/material_agent_service/docker-compose.yml build --no-cache
docker compose -f apps/material_agent_service/docker-compose.yml up
```
