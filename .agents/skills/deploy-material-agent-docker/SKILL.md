---
name: deploy-material-agent-docker
description: Deploy the material-agent-service locally using Docker Compose with OVRTX rendering. Use when user wants to run material agent with docker, docker compose, set up local deployment, run the service locally with GPU rendering, start material agent containers, or configure VLM provider for docker deployment. Trigger phrases include "docker compose", "docker deploy", "run locally with docker", "start material agent docker", "docker up", "local deployment", "compose up".
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - material-agent
  - docker
  - ovrtx
  - deployment
tools:
  - Shell
  - Docker
  - curl
  - Python
  - Filesystem
compatibility: Requires Docker daemon, Docker Compose v2.24+, NVIDIA Container Toolkit, an RTX-class GPU with 48GB+ VRAM for local OVRTX rendering, repo-root .env provider credentials, and free host ports 8000/8001.
---

# Deploy Material Agent Service with Docker Compose

Deploy the material-agent-service and OVRTX rendering API locally using Docker Compose. No NGC login or Kubernetes required for the default setup.

## When to Use

- Use when the user wants to run `material-agent-service` locally with Docker Compose.
- Use when the user needs the bundled OVRTX rendering API sidecar on a GPU host.
- Use when the user wants to configure cloud VLM providers or an optional local VLM NIM sidecar for Material Agent.
- Use `quickstart` for a shorter first local POC, and use `deploy-collection` when running multiple Content Agents together with shared dependency endpoints.

## Limitations

- The default stack owns host ports 8000 and 8001. Stop overlapping Material, Physics, Texture, or standalone OVRTX stacks before startup.
- Local OVRTX rendering needs an RTX-class GPU with about 48GB+ VRAM; A100/H100/H200/V100 are not supported for this path.
- First build and first render are long-running operations. Start Compose normally, then return logs and health commands instead of holding an agent session open indefinitely.
- Keep secrets out of chat and commits. Tell the user to edit `.env`; do not ask them to paste keys.

## Prerequisites

Check before deploying:

1. **Docker Compose v2.24+**: `docker compose version` -- required for `env_file: required: false` long-form syntax
2. **RTX-capable NVIDIA GPU** with 48GB+ VRAM for local OVRTX rendering
   (L40, L40S, RTX PRO 6000, etc.; not A100/H100/H200/V100): `nvidia-smi`
3. **NVIDIA Container Toolkit** installed: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
4. **VLM provider API key** (at least one): OpenAI, Anthropic, Gemini, or NVIDIA
5. **Image generation key** (optional): public default is Gemini, so set `GOOGLE_API_KEY` if the user plans to use generated reference images or generated material-library textures. Otherwise set `MA_IMAGE_GEN_BACKEND` to another configured provider.

## Instructions

1. Confirm Docker, Compose, GPU, NVIDIA Container Toolkit, and port availability before starting the stack.
2. Create or update the repo-root `.env` with exactly the provider credentials the selected backend needs.
3. Start the Material Agent compose stack from the repo root.
4. Wait for both the main service and OVRTX readiness checks before reporting the service ready.
5. Return service URLs, health state, log commands, and stop commands using the output format below.

### Set VLM API Key

Create `.env` at the **repo root** (the compose file reads it via `env_file: ../../.env`):

```bash
# Pick ONE provider:
echo 'OPENAI_API_KEY=sk-...' > .env
# OR
echo 'NVIDIA_API_KEY=nvapi-...' > .env
# OR
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
# OR
echo 'GOOGLE_API_KEY=...' > .env
```

### Start Services

```bash
docker compose -f apps/material_agent_service/docker-compose.yml up --build
```

This starts:
- **material-agent-service** on port 8000 (REST API)
- **ovrtx-rendering-api** on port 8001 (GPU rendering, built from source)

First build takes ~10 minutes. First render takes ~5 minutes (shader compilation, cached after).

### Access

- **Health**: http://localhost:8000/health
- **API Docs** (Swagger UI): http://localhost:8000/docs
- **OpenAPI spec**: http://localhost:8000/openapi.json

## Services

| Service | Port | GPU | Builds From | Always Starts |
|---|---|---|---|---|
| material-agent-service | 8000 | No | Source | Yes |
| ovrtx-rendering-api | 8001 | 1x (48GB) | Source | Yes |
| vlm-nim | 8003 | 1x (48GB) | NGC image | No (profile) |

## Adding Local VLM NIM

To run Cosmos Reason2 8B locally instead of using a cloud VLM API:

```bash
# NGC login required for VLM NIM image
printf '%s' "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' --password-stdin

# Add NGC_API_KEY to .env for model weight download
echo 'NGC_API_KEY=...' >> .env

# Start with vlm profile + multi-gpu overlay
# The overlay pins ovrtx-rendering-api to GPU 0 and vlm-nim to GPU 1.
# It is REQUIRED here: NIM's profile selector reports "Free GPUs: <None>"
# and refuses to launch when ovrtx is also holding a GPU, even with tens
# of GB free. See docker-compose.multi-gpu.yml for the pinning details.
docker compose \
  -f apps/material_agent_service/docker-compose.yml \
  -f apps/material_agent_service/docker-compose.multi-gpu.yml \
  --profile vlm up --build
```

Requires 2 GPUs (48GB each). VLM NIM takes ~15 minutes to start (model compilation).

The multi-gpu overlay also routes `MA_VLM_*` and `MA_LLM_*` to the local
vlm-nim sidecar (`nvidia/cosmos-reason2-8b`), so no extra `.env` edits
are needed for `--profile vlm`.

## Operations

### View Logs

```bash
# All services
docker compose -f apps/material_agent_service/docker-compose.yml logs -f

# Specific service
docker logs material-agent-service
docker logs ovrtx-rendering-api
```

### Stop

```bash
# Stop all services
docker compose -f apps/material_agent_service/docker-compose.yml down

# Stop and remove session data
docker compose -f apps/material_agent_service/docker-compose.yml down -v
```

### Rebuild After Code Changes

```bash
docker compose -f apps/material_agent_service/docker-compose.yml up --build

# Force full rebuild (no cache)
docker compose -f apps/material_agent_service/docker-compose.yml build --no-cache
docker compose -f apps/material_agent_service/docker-compose.yml up
```

### Check Health

```bash
curl http://localhost:8000/health   # main service
python - <<'PY'
import json
from urllib.request import urlopen

try:
    with urlopen("http://localhost:8001/health", timeout=10) as response:
        health = json.load(response)
except Exception as exc:
    print(f"rendering API unreachable: {exc}")
    raise SystemExit(1)

print(json.dumps(health))
if health.get("status") == "unhealthy":
    print("rendering API unhealthy")
    raise SystemExit(1)
if health.get("gpu_initialized") is True:
    print("rendering API ready")
else:
    print("rendering API warming")
PY
```

## Resource Requirements

| Configuration | GPUs | Compose CPU limit | Compose memory limit |
|---|---|---|---|
| Default (main service + OVRTX) | 1 | 8 | 16G |
| + VLM NIM profile | 2 | 8 | 16G |

The CPU and memory limits above are the explicit `material-agent-service` limits in `apps/material_agent_service/docker-compose.yml`. `ovrtx-rendering-api` and `vlm-nim` reserve GPUs but do not define CPU or memory limits in the compose files, so size the host with additional headroom for those sidecars.

## GPU Configuration

To assign more GPUs to the rendering API, edit `docker-compose.yml`:

```yaml
ovrtx-rendering-api:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 2          # number of GPUs
            capabilities: [gpu]
```

Or assign specific GPU IDs:

```yaml
            device_ids: ['0', '1']
```

## Environment Variables

All configurable via `.env` file at the repo root:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | | OpenAI VLM provider |
| `ANTHROPIC_API_KEY` | | Anthropic VLM provider |
| `GOOGLE_API_KEY` | | Google Gemini VLM provider |
| `NVIDIA_API_KEY` | | NVIDIA (build.nvidia.com) VLM provider |
| `NGC_API_KEY` | | NGC auth (only for VLM NIM profile) |
| `MA_IMAGE_GEN_BACKEND` | `gemini` | Shared backend for generated reference images and generated material-library textures (`gemini`, `openai`, `nim`) |
| `MA_IMAGE_GEN_MODEL` | | Optional shared image-generation model override |
| `MA_IMAGE_GEN_BASE_URL` | | Optional shared image-gen API base URL override |
| `MA_MAX_ACTIVE_SESSIONS` | 1 | Max concurrent pipelines for local OVRTX compose |
| `MA_MAX_RENDER_NUM_WORKERS` | 1 | Max accepted render worker override for local OVRTX compose |
| `WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS` | 1 | Process-wide outbound render request cap |
| `MA_SESSION_TTL_HOURS` | 24 | Session expiry time |
| `MA_MAX_UPLOAD_SIZE_MB` | 500 | Max USD upload size |
| `OVRTX_NUM_SENSOR_UPDATES` | 500 | Sensor update count before capture |
| `NIM_MAX_MODEL_LEN` | 131072 | VLM NIM context length (for 48GB GPUs) |

Prediction concurrency is set per request with `vlm_max_workers`; it is not a
service-wide environment variable.

Generated reference images and generated material-library mode are optional. If
enabled from the client or UI, the service uses `MA_IMAGE_GEN_BACKEND=gemini`
by default.

## Output Format

When handing control back to the user, report:

- `SERVICE_URL`: `http://localhost:8000`
- `DOCS_URL`: `http://localhost:8000/docs`
- `SERVICE_HEALTH`: `healthy`, `starting`, or `unhealthy`
- `OVRTX_HEALTH`: `healthy` only when `/health` contains `"gpu_initialized":true`; otherwise `warming` or `unhealthy`
- `LOGS`: `docker compose -f apps/material_agent_service/docker-compose.yml logs -f`
- `STOP`: `docker compose -f apps/material_agent_service/docker-compose.yml down`
- Any missing credentials, port conflicts, or GPU/toolkit blockers.

## Troubleshooting

### OVRTX rendering API not starting

Check GPU access:
```bash
docker logs ovrtx-rendering-api
docker exec ovrtx-rendering-api nvidia-smi
```

If shader compilation messages appear, wait ~5 minutes (first-time only).

### Main service unhealthy before rendering API ready

The main service `depends_on` the rendering API health check. If rendering takes long to start (shader compilation), increase `start_period` in `docker-compose.yml`.

### VLM NIM KV cache error on 48GB GPUs

Already configured with `NIM_MAX_MODEL_LEN=131072` by default. To reduce further:

```bash
NIM_MAX_MODEL_LEN=65536 docker compose \
  -f apps/material_agent_service/docker-compose.yml \
  -f apps/material_agent_service/docker-compose.multi-gpu.yml \
  --profile vlm up
```
