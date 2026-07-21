---
name: deploy-physics-agent-docker
description: Deploy the physics-agent-service locally using Docker Compose with the bundled OVRTX GPU rendering sidecar. Use when user wants to run physics agent with docker, docker compose, set up local deployment of the physics service, run it on a GPU box, start physics agent containers, configure the VLM provider for physics docker deployment, or check whether tune/refine service routes are usable. Trigger phrases include "deploy physics agent", "docker compose physics", "run physics agent locally", "start physics service docker", "physics compose up", "physics agent docker".
version: "0.1.1"
author: NVIDIA Content Agents
tags:
  - content-agents
  - physics-agent
  - docker
  - ovrtx
  - deployment
tools:
  - Shell
  - Docker
  - curl
  - Python
  - Filesystem
compatibility: Requires Docker daemon, Docker Compose v2.24+, NVIDIA Container Toolkit, an NVIDIA GPU with about 16GB+ VRAM for the OVRTX sidecar, repo-root .env provider credentials, free host ports 8000/8001, and a tuning-enabled image when validating service-side refine.
---

# Deploy Physics Agent Service with Docker Compose

Deploy the `physics-agent-service` and the bundled OVRTX rendering API locally using Docker Compose. The physics service is CPU-only; the rendering sidecar uses the GPU.

## When to Use

- Use when the user wants to run `physics-agent-service` locally with Docker Compose.
- Use when the user needs the bundled OVRTX rendering sidecar for physics classification.
- Use when the user wants to configure VLM provider credentials or run local smoke requests with optimizer flags.
- Use when the user asks whether `/tune` or `/refine` is available in a
  Docker deployment, including the server-configured refine provider requirements.
- Use `quickstart` for a shorter first local POC, and use `deploy-collection` when running multiple Content Agents together.

## Limitations

- The default stack owns host ports 8000 and 8001. Stop overlapping Material, Physics, Texture, or standalone OVRTX stacks before startup.
- The main service waits on OVRTX readiness; OVRTX is ready only when `/health` reports `gpu_initialized: true`.
- First build and first render are long-running operations. Return logs and health commands rather than holding an agent session open indefinitely.
- Keep secrets out of chat and commits. Tell the user to edit `.env`; do not ask them to paste keys.
- Service `/refine` requires an image with the `tuning` dependencies, an
  OvPhysX runtime, and a registered chat/VLM provider selected with
  `PA_REFINE_BACKEND` and `PA_REFINE_MODEL`.

## Prerequisites

Check before deploying:

1. **Docker Compose v2.24+**: `docker compose version` -- required for `env_file: required: false` long-form syntax
2. **NVIDIA GPU** with ~16 GB+ VRAM: `nvidia-smi`
3. **NVIDIA Container Toolkit** installed: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
4. **VLM provider API key** (at least one): NVIDIA NIM, OpenAI, Anthropic, or Gemini
5. **Scene Optimizer backend**: either run `./scripts/fetch_build_resources.sh`
   for the local optimizer bundle, or configure a remote NVCF optimizer with
   `NGC_API_KEY` plus `NVCF_OPTIMIZER_FUNCTION_ID` / `OPTIMIZER_ENDPOINT`
6. **Refine runtime/config** only when validating `/refine`: tuning
   dependencies, an OvPhysX runtime, `PA_REFINE_BACKEND`, `PA_REFINE_MODEL`,
   and the selected provider's credential

## Instructions

1. Confirm Docker, Compose, GPU, NVIDIA Container Toolkit, and port availability before starting the stack.
2. Create or update the repo-root `.env` with exactly the VLM provider credentials the selected backend needs.
3. Prepare one Scene Optimizer backend before building the image: run
   `./scripts/fetch_build_resources.sh` for the local bundle, or skip that
   local fetch and set `NGC_API_KEY` plus `NVCF_OPTIMIZER_FUNCTION_ID` /
   `OPTIMIZER_ENDPOINT` for a remote NVCF optimizer.
4. Start the Physics Agent compose stack from the repo root.
5. Wait for both the main service and OVRTX readiness checks before reporting the service ready.
6. For optimizer-sensitive smoke assets, use the optimizer form fields below.
7. For tune/refine validation, confirm the container image includes the tuning
   extra. For `/refine`, also confirm `PA_REFINE_BACKEND`, `PA_REFINE_MODEL`,
   and the selected provider's credential before submitting a job.
8. Return service URLs, health state, log commands, and stop commands using the output format below.

### Set VLM API Key

Create `.env` at the **repo root** (the compose file reads it via `env_file: ../../.env`):

```bash
# Pick ONE provider:
echo 'NVIDIA_API_KEY=nvapi-...' > .env
# OR
echo 'OPENAI_API_KEY=sk-...' > .env
# OR
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
# OR
echo 'GOOGLE_API_KEY=...' > .env
```

### Start Services

Choose one Scene Optimizer backend before starting the stack.

#### Local Scene Optimizer Core

```bash
./scripts/fetch_build_resources.sh
docker compose -f apps/physics_agent_service/docker-compose.yml up --build
```

`scripts/fetch_build_resources.sh` stages the local Scene Optimizer Core bundle
under `.build-resources/scene_optimizer_core` so the default `optimize_usd`
pipeline path works inside the service image. If the default package is not
usable for the host architecture, set `SO_CORE_URL` to an explicit Scene
Optimizer Core zip.

#### Remote NVCF Scene Optimizer

Skip `./scripts/fetch_build_resources.sh` for remote-only deployments,
especially on host architectures without a local Scene Optimizer package.

```bash
cat >> .env <<'EOF'
NGC_API_KEY=ngc-...
NVCF_OPTIMIZER_FUNCTION_ID=...
# or OPTIMIZER_ENDPOINT=https://...
EOF

docker compose -f apps/physics_agent_service/docker-compose.yml up --build
```

This starts:
- **physics-agent-service** on port 8000 (REST API)
- **ovrtx-rendering-api** on port 8001 (GPU rendering, built from source)

First build takes ~10 minutes. First render takes ~5 minutes (shader compilation; cached after).

### Access

- **Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **OpenAPI spec**: `apps/physics_agent_service/openapi.yaml`

## Services

| Service | Port | GPU | Builds From | Always Starts |
|---|---|---|---|---|
| physics-agent-service | 8000 | No | Source | Yes |
| ovrtx-rendering-api | 8001 | 1x | Source | Yes |

The main service `depends_on` the rendering API's health check passing (which flips `gpu_initialized` to `true`). On cold start expect the physics-agent container to sit in "waiting" state for ~5 minutes before it comes up.

## Operations

### View Logs

```bash
# All services
docker compose -f apps/physics_agent_service/docker-compose.yml logs -f

# Specific service
docker logs physics-agent-service
docker logs physics-ovrtx-rendering-api
```

### Stop

```bash
# Stop all services
docker compose -f apps/physics_agent_service/docker-compose.yml down

# Stop and remove session data
docker compose -f apps/physics_agent_service/docker-compose.yml down -v
```

### Rebuild After Code Changes

```bash
docker compose -f apps/physics_agent_service/docker-compose.yml up --build

# Force full rebuild (no cache)
docker compose -f apps/physics_agent_service/docker-compose.yml build --no-cache
docker compose -f apps/physics_agent_service/docker-compose.yml up
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

### REST Smoke with Optimizer Flags

For ordinary smoke assets, run the pipeline without optimizer flags. For
instanced USDs, assets that fail `apply_physics` with an instance-proxy authoring
error, or one combined mesh that needs split-by-component predictions, pass the
optimizer form fields to `POST /pipeline`:

```bash
# Instance-proxy authoring fix.
curl -X POST "http://localhost:8000/pipeline" \
  -F "usd_file=@scene.usd" \
  -F "optimize_usd=true" \
  -F "enable_deinstance=true"

# Also split one combined disjoint mesh into separate component predictions.
curl -X POST "http://localhost:8000/pipeline" \
  -F "usd_file=@scene.usd" \
  -F "optimize_usd=true" \
  -F "enable_deinstance=true" \
  -F "enable_split=true"
```

Use `enable_deduplicate=true` only when repeated identical geometry should be
collapsed. At least one optimizer operation must be enabled when
`optimize_usd=true`.

### Refine Route Smoke

Only run this on a tuning-enabled image. Production `/refine` execution
requires the tuning extra, an OvPhysX daemon environment, and a registered
chat/VLM provider with matching credentials.

```bash
# First produce an apply_physics output USD with /pipeline.
PIPELINE_SESSION=$(curl -fsS -X POST "http://localhost:8000/pipeline" \
  -F "usd_file=@scene.usd" | jq -r .session_id)

# Wait for /pipeline/$PIPELINE_SESSION/status to become completed, then refine.
REFINE_SESSION=$(curl -fsS -X POST "http://localhost:8000/refine" \
  -F "source_session_id=$PIPELINE_SESSION" \
  -F "scenario_yaml=<apps/physics_agent/configs/tuning/drop_settle.yaml" \
  -F "user_prompt=make the object settle on the target surface" \
  -F "optimizer=botorch" \
  -F "score_threshold=0.9" \
  -F "seed=42" | jq -r .session_id)

curl -fsS "http://localhost:8000/refine/$REFINE_SESSION/status" | jq .
curl -fsS "http://localhost:8000/refine/$REFINE_SESSION/results" | jq .
```

## Resource Requirements

| Configuration | GPUs | CPU | Memory |
|---|---|---|---|
| Default (main + rendering) | 1 | 10 | 20 G |

## Environment Variables

Configurable via `.env` at the repo root. Key settings:

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | | NVIDIA (build.nvidia.com) VLM provider |
| `OPENAI_API_KEY` | | OpenAI VLM provider |
| `ANTHROPIC_API_KEY` | | Anthropic VLM provider |
| `GOOGLE_API_KEY` | | Google Gemini VLM provider |
| `PA_VLM_BACKEND` | `nim` | Which VLM backend to use |
| `PA_VLM_MODEL` | `qwen/qwen3.5-397b-a17b` | Model id for the selected backend |
| `PA_VLM_TEMPERATURE` | `1.0` | Sampling temperature |
| `PA_REFINE_BACKEND` | `gemini` | `/refine` judge/refiner backend; must be registered for both chat and VLM |
| `PA_REFINE_MODEL` | `gemini-3-pro-preview` | `/refine` judge/refiner model override |
| `PA_MAX_ACTIVE_SESSIONS` | `1` | Max concurrent pipelines |
| `PA_SESSION_TTL_HOURS` | `24` | Session expiry time |
| `PA_MAX_UPLOAD_SIZE_MB` | `500` | Max USD upload size |
| `WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS` | `1` | Process-wide outbound render request cap for the local OVRTX sidecar |
| `OVRTX_NUM_SENSOR_UPDATES` | `500` | Sensor update count before capture (rendering sidecar) |

## GPU Configuration

To assign specific GPUs to the rendering API, edit
`apps/physics_agent_service/docker-compose.yml`:

```yaml
ovrtx-rendering-api:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1          # number of GPUs
            capabilities: [gpu]
```

Or pin a specific GPU ID:

```yaml
            device_ids: ['0']
```

## Output Format

When handing control back to the user, report:

- `SERVICE_URL`: `http://localhost:8000`
- `DOCS_URL`: `http://localhost:8000/docs`
- `SERVICE_HEALTH`: `healthy`, `starting`, or `unhealthy`
- `OVRTX_HEALTH`: `healthy` only when `/health` contains `"gpu_initialized":true`; otherwise `warming` or `unhealthy`
- `LOGS`: `docker compose -f apps/physics_agent_service/docker-compose.yml logs -f`
- `STOP`: `docker compose -f apps/physics_agent_service/docker-compose.yml down`
- Any missing credentials, port conflicts, GPU/toolkit blockers, or optimizer flags used for smoke validation.
- For `/refine`, whether the image has tuning/OvPhysX dependencies and whether
  the selected `PA_REFINE_BACKEND`, `PA_REFINE_MODEL`, and credential are
  configured.

## Troubleshooting

### OVRTX rendering API not starting

Check GPU access:

```bash
docker logs physics-ovrtx-rendering-api
docker exec physics-ovrtx-rendering-api nvidia-smi
```

Shader compilation on first boot takes ~5 minutes; wait it out.

### Main service unhealthy before rendering API ready

The main service `depends_on` the rendering API's health check. If rendering takes long to start, the physics-agent-service container will stay in "waiting" state. Check `docker compose ps` to see which container is blocking.

### 503 / VLM failures under load

`PA_MAX_ACTIVE_SESSIONS` defaults to 1 because rendering plus a VLM call per prim is the main throughput bottleneck. Raising this requires headroom on both CPU memory and VLM provider quota.

### Refine fails before first iteration

The service `/refine` route builds judge/refiner models inside the deployment.
If logs mention missing dependencies, backend registration, or an API key, use
an image with the `tuning` extra and OvPhysX runtime, then select a provider
registered for both chat and VLM and configure its credential.
