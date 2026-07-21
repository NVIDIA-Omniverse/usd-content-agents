---
name: deploy-texture-agent-docker
description: Deploy the texture-agent-service locally using Docker Compose, including hosted/simple image generation, optional local FLUX.2 and LLM NIM sidecars, and an operator-mounted Texture Variation API / Step1X-compatible overlay with OVRTX. Use when user wants to run texture agent with docker, docker compose, set up local deployment, run UV-aware texture projection locally, validate an existing Step1X-compatible backend, start the texture-agent service containers, or route image generation to local sidecars. Trigger phrases include "docker compose texture", "docker deploy texture", "run texture agent locally", "start texture agent docker", "texture agent up", "local texture deployment", "image-gen sidecar", "step1x overlay", "uv-aware texture".
version: "0.2.1"
author: NVIDIA Content Agents
tags:
  - content-agents
  - texture-agent
  - docker
  - image-generation
  - step1x
  - texture-variation-api
  - ovrtx
  - deployment
tools:
  - Shell
  - Docker
  - curl
  - Python
  - Filesystem
compatibility: Requires Docker daemon, Docker Compose v2.24+, compose-interpolated TA/TEXTURE/OVRTX settings in the shell or repo-root .env passed with --env-file, provider keys for hosted backends, free host port 8001 for the main service, optional ports 8005 and 8006 for local NIM sidecars, and optional ports 8019, 8018, and 8002 plus NVIDIA GPU support for the operator-mounted Step1X-compatible overlay. Step1X overlay runs require NVIDIA Container Toolkit and a reviewed runtime mounted with TEXTURE_STEP1X_HOST_RUNTIME.
---

# Deploy Texture Agent Service with Docker Compose

Deploy the Texture Agent service locally with Docker Compose. Pick the smallest
mode that matches the user's goal:

1. **Default service:** `texture-agent-service` on port `8001`, using the
   configured hosted/simple image-generation backend.
2. **Local NIM sidecars:** optional FLUX.2 image generation and LLM sidecars
   for local image-gen or auto-prompt calls.
3. **Operator-mounted Step1X-compatible overlay:** Texture Agent plus the simple
   Texture Variation API sidecar, a Step1X-compatible API adapter, and OVRTX
   rendering. The public release does not install or download the Step1X runtime.

## When to Use

- Use when the user wants to run `texture-agent-service` locally with Docker
  Compose, with hosted image-gen or optional local FLUX.2 / LLM NIM sidecars.
- Use when the user wants UV-aware texture generation through a Texture Variation
  API backend, including a Step1X-compatible runtime they already mounted.
- Use when the user wants a fake-runner smoke test for container wiring without
  Step1X model weights.
- Use `quickstart` for a shorter first local POC, and use `deploy-collection`
  when running multiple Content Agents together.

## Limitations

- The main service owns host port `8001`, overlapping other local Content Agent
  stacks that expose a primary service.
- The default hosted `nim` image backend is text-only and cannot accept
  reference images for tightly matched PBR sets; use a conditioning-capable
  backend or an operator-mounted Texture Variation API backend for UV-aware
  texture evidence.
- Local NIM sidecars require idle GPUs, NGC auth, and model warm-up time.
- The Step1X runtime, model weights, third-party checkouts, and caches are not
  committed to this repository. The public Compose overlay requires a runtime
  directory that has already been prepared and reviewed by the operator.
- Full Step1X-compatible validation needs GPU memory and runtime assets. Two
  GPUs are preferred: one for the texture backend, one for OVRTX.
- The Step1X-compatible overlay validates service topology and compatible USD
  assets; it is not a blanket guarantee that every arbitrary USD can be edited
  without target-specific material scope, UV, or backend configuration.
- The overlay defaults to the Step1X-compatible backend for normal requests but
  also starts the simple Texture Variation sidecar. Requests can select
  `texture_backend=simple_image_gen` without restarting or redeploying the stack.
- OVRTX mounts the session volume read-only and host `/tmp` at `/host_tmp` so
  render requests can resolve generated USDs and host temp paths.
- One-GPU shared validation has no runtime GPU contention guard after startup.
  Defaults serialize texture workers and render requests, but operators should
  run one pipeline job at a time or use separate GPUs for final evidence.
- Keep secrets out of chat and commits. Tell the user to edit `.env`; do not
  ask them to paste keys.

## Prerequisites

1. Docker 20.10+ with Compose v2.24+: `docker compose version`
2. A free main-service port, normally `8001`
3. Provider creds for hosted backends/Step1X auto-prompt, usually
   `NVIDIA_API_KEY`; send `auto_prompt_enabled=false` to avoid hosted LLM calls
4. For local NIM sidecars:
   - NVIDIA GPU capacity for each enabled sidecar
   - NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
   - NGC auth for pulling `nvcr.io/nim/*` images
   - `NGC_API_KEY` and, for FLUX.2, `HF_TOKEN` in repo-root `.env`
5. For the Step1X-compatible overlay:
   - NVIDIA GPU and NVIDIA Container Toolkit
   - Free ports `8001`, `8019`, `8018`, and `8002`, or overrides for
     `TA_HOST_PORT`, `TEXTURE_SIMPLE_PORT`, `TEXTURE_STEP1X_PORT`, and
     `OVRTX_HOST_PORT`
   - A reviewed runtime directory available on the host and passed as
     `TEXTURE_STEP1X_HOST_RUNTIME`
   - A container-visible Python/runtime command configured with
     `TEXTURE_STEP1X_PYTHON`, `TEXTURE_STEP1X_EDIT_SCRIPT`, or
     `TEXTURE_STEP1X_COMMAND_TEMPLATE`
   - Reproducible evidence tools: `curl`, `jq`, and `unzip`

## Instructions

1. Choose the deployment mode: default service, local NIM sidecars, or
   operator-mounted Step1X-compatible overlay.
2. From the repo root, create or update `.env`. Compose reads repo-root `.env`
   when invoked with `--env-file .env`; put Compose-interpolated
   `TA_*`/`TEXTURE_*`/`OVRTX_*` overrides there or in the shell. If all
   overrides come from the shell, run `touch .env` first because
   `--env-file .env` still requires the file to exist.
3. Start the chosen Compose stack from the repository root.
4. Check health endpoints before reporting the service ready. For
   operator-mounted Step1X-compatible runs, query `/health` on the backend and
   require runtime readiness before treating output as real model evidence.
5. Use `apps/texture_gen_step1x_service/README.md` for the runtime contract and
   `apps/texture_agent/examples/simple_image_gen_bucket/README.md` for the
   public simple-image-gen baseline.
6. Return service URLs, active backend, sidecar state, log commands, and stop
   commands using the output format below.

### Default Service

The default service starts only `texture-agent-service`. By default it uses
`TA_TEXTURE_BACKEND=simple_image_gen`, `TA_IMAGE_GEN_BACKEND=nim`, and the
hosted NVIDIA FLUX.2 endpoint at build.nvidia.com.

```bash
cat >> .env <<'EOF'
NVIDIA_API_KEY=YOUR_NVIDIA_API_KEY_HERE
EOF

docker compose --env-file .env -f apps/texture_agent_service/docker-compose.yml up --build
```

Check:

```bash
curl -fsS http://localhost:8001/health
```

Main endpoints are `http://localhost:8001/health` and
`http://localhost:8001/docs`.

#### Hosted Backend Overrides

Use shell variables or repo-root `.env` passed with `--env-file .env` for
Compose-interpolated `TA_*` settings. Precedence is explicit Compose
`environment:` entries first, then shell or repo-root `--env-file .env`
interpolation, then service-local `apps/texture_agent_service/.env`;
service-local `TA_*` values do not override explicit Compose `environment:`
entries.

Set `TA_IMAGE_GEN_BACKEND=gemini` with `GOOGLE_API_KEY`, or
`TA_IMAGE_GEN_BACKEND=openai` with `OPENAI_API_KEY`. Leave
`TA_IMAGE_GEN_BACKEND=nim` for the hosted NVIDIA default.

### Local NIM Sidecars

Two optional sidecars can replace hosted calls:

- `image-gen-nim`: FLUX.2 Klein 4B. Profile: `image-gen`. Port `8005`.
- `llm-nim`: Llama 3.1 Nemotron Nano 8B. Profile: `llm`. Port `8006`.

Put `NGC_API_KEY` and `HF_TOKEN` in repo-root `.env` because the NIM services
read the repo-root env file. The multi-GPU overlay routes the main service to
the in-network sidecar endpoints and pins sidecars to specific GPUs.

```bash
printf '%s' "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' --password-stdin

cat >> .env <<'EOF'
NGC_API_KEY=YOUR_NGC_API_KEY_HERE
HF_TOKEN=YOUR_HF_TOKEN_HERE
EOF

# Image-gen sidecar; add `--profile llm` for the LLM sidecar too
docker compose --env-file .env -f apps/texture_agent_service/docker-compose.yml -f apps/texture_agent_service/docker-compose.multi-gpu.yml --profile image-gen up --build
```

Edit both `NVIDIA_VISIBLE_DEVICES` and `device_ids` in
`apps/texture_agent_service/docker-compose.multi-gpu.yml` when GPU `0` or `1`
is not free. That overlay pins the local NIM sidecars directly in YAML rather
than through shell variables.

Check:

```bash
curl -fsS http://localhost:8001/health
curl -fsS http://localhost:8005/v1/health/ready  # if --profile image-gen is enabled
curl -fsS http://localhost:8006/v1/health/ready  # if --profile llm is enabled
```

### Operator-Mounted Step1X + OVRTX Overlay

Use this mode only after the Step1X-compatible runtime has been reviewed,
prepared, and mounted by the operator. The overlay starts Texture Agent, the
simple Texture Variation API, the Step1X-compatible API adapter, and OVRTX.

```bash
export TEXTURE_STEP1X_HOST_RUNTIME=/path/to/reviewed/texture-editing-runtime
export TEXTURE_STEP1X_PYTHON=/opt/texture-editing/.venv_gen/bin/python

docker compose --env-file .env \
  -f apps/texture_agent_service/docker-compose.yml \
  -f apps/texture_agent_service/docker-compose.step1x.yml \
  up --build
```

The overlay requires `TEXTURE_STEP1X_HOST_RUNTIME` and does not create, clone,
install, download, or cache Step1X, Material Anything, Swin2SR, Kaolin,
nvdiffrast, or model checkpoints. Use
`apps/texture_gen_step1x_service/README.md` for the runtime contract and health
payload expectations.

Before reporting Step1X-compatible evidence, make sure the backend `/health`
endpoint is ready. Also check the simple sidecar `/livez` or `/health` endpoint
when the user asks for full sidecar proof. On one GPU, run only one pipeline job
at a time.

For container-wiring smoke without model weights, use the bundled fake runner
through the same overlay. This proves API, sidecar DNS, shared session volume,
texture application, and render plumbing, but must not be presented as
model-quality Step1X output. The command template below assumes the standard
service image layout where the repo is mounted at `/workspace/world-understanding`;
adjust the script path if you use a custom image or mount point:

```bash
TEXTURE_STEP1X_HOST_RUNTIME="$PWD" \
TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS=false \
TEXTURE_STEP1X_VALIDATE_ASSETS=false \
TEXTURE_STEP1X_REQUIRED_EXECUTABLES= \
TEXTURE_STEP1X_COMMAND_TEMPLATE='python /workspace/world-understanding/apps/texture_gen_step1x_service/smoke/fake_step1x_runner.py --source-asset {source_asset} --prompt {prompt} --output-dir {output_dir} --texture-size {texture_size}' \
docker compose --env-file .env \
  -f apps/texture_agent_service/docker-compose.yml \
  -f apps/texture_agent_service/docker-compose.step1x.yml \
  up --build
```

## Operations

#### View Compose Logs

```bash
# Default service
docker compose --env-file .env -f apps/texture_agent_service/docker-compose.yml logs -f

# Step1X-compatible overlay: add -f apps/texture_agent_service/docker-compose.step1x.yml.
```

#### Stop

```bash
# Default service
docker compose --env-file .env -f apps/texture_agent_service/docker-compose.yml down

# Step1X-compatible overlay: stop with the same compose file list used for `up`.
# Add `-v` only when you intentionally want a clean session-volume reset.
```

## Resource Requirements

| Configuration | GPUs | CPU | Memory |
|---|---:|---:|---:|
| Default service with hosted image-gen and hosted LLM | 0 | 4 limit | 8 GB limit |
| Image-gen NIM sidecar | 1, 24 GB+ | 8 recommended | 24 GB recommended |
| LLM NIM sidecar | 1, 48 GB | 8 recommended | 24 GB recommended |
| Both NIM sidecars | 2 | 12 recommended | 40 GB recommended |
| Step1X-compatible overlay, texture backend and OVRTX on separate GPUs | 2 | 12 recommended | 40 GB recommended |
| Step1X-compatible overlay, one-GPU serial or shared-GPU validation | 1 | 12 recommended | 40 GB recommended |

Only `texture-agent-service` defines CPU and memory limits in Compose; GPU
sidecar rows are host sizing guidance. The simple Texture Variation sidecar is
CPU-only and does not add a GPU request.

## Environment Variables

This list only covers high-signal wiring. See
`apps/texture_agent_service/docs/api.md#configuration` for the full Texture
Agent service table and `apps/texture_gen_step1x_service/README.md` for the
Step1X runtime contract.

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | - | NVIDIA inference / cloud NIM backend key |
| `GOOGLE_API_KEY` | - | Gemini provider key |
| `OPENAI_API_KEY` | - | Hosted OpenAI provider key |
| `NGC_API_KEY` | - | NGC auth for local NIM images |
| `HF_TOKEN` | - | Hugging Face token for FLUX.2 NIM weight download |
| `TA_IMAGE_GEN_BACKEND` | service default `nim` | `nim`, `gemini`, or `openai` for simple image-gen path |
| `TA_IMAGE_GEN_BASE_URL` | service default backend-specific | Override for hosted or local image-gen endpoint |
| `TA_TEXTURE_PLAN_DEFAULT_CAP` | `32` | Generic/simple-image-gen planning default |
| `TA_TEXTURE_PLAN_UV_AWARE_DEFAULT_CAP` | `16` | UV-aware/service and Step1X planning default |
| `TA_TEXTURE_PLAN_HARD_CAP` | `64` | Immutable per-plan hard maximum before backend work |
| `TA_MAX_TEXTURE_UNITS` | `64` | Compatibility executor guard; keep aligned with the planning hard cap |
| `TEXTURE_GEN_BACKEND` | simple sidecar default | Simple Texture Variation sidecar provider (`nim`, `gemini`, or `openai`; installed plugins may add providers) |
| `TEXTURE_GEN_API_KEY` | unset | Optional endpoint-scoped simple sidecar key. Leave unset for hosted providers so `NVIDIA_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`, or `OPENAI_API_KEY` is used; set `not-used` only for no-auth local endpoints |
| `TEXTURE_GEN_SIMPLE_ENV_FILE` | `/dev/null` | Optional extra env file for the simple sidecar when provider keys live outside this checkout; prefer this over relying on Compose `--env-file` for container secrets |
| `TEXTURE_STEP1X_HOST_RUNTIME` | required for overlay | Host runtime directory mounted into the Step1X-compatible API adapter |
| `TEXTURE_STEP1X_GPU_DEVICE` | `0` | GPU device ID for the Step1X sidecar |
| `OVRTX_GPU_DEVICE` | `1` | GPU device ID for the OVRTX sidecar; set both GPU vars to `0` on one-GPU hosts |
| `TEXTURE_STEP1X_PYTHON` | runtime default | Advanced override for custom mounted runtimes |
| `TEXTURE_STEP1X_LD_LIBRARY_PATH` | reference `.venv_gen` torch/CUDA libs | Advanced override when a custom mounted runtime stores CUDA libraries elsewhere |
| `TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS` | readiness healthcheck `true` | Runs cached torch/CuPy/NVRTC probe when `healthcheck.py` checks `/health` |
| `TEXTURE_STEP1X_HEALTHCHECK_TIMEOUT` | `180s` | Healthcheck timeout; relevant to cold runtime imports only for readiness healthchecks |
| `OVRTX_NUM_SENSOR_UPDATES` | OVRTX default | Raise for higher quality final renders |
| `WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS` | `1` in overlay | Limits concurrent OVRTX render requests from the main service |

## Output Format

When handing control back to the user, report:

- `SERVICE_URL` / `DOCS_URL`: usually `http://localhost:8001` and `/docs`
- `COMPOSE_FILE`: the file or files used
- `TEXTURE_BACKEND`: active `TA_TEXTURE_BACKEND`, endpoint, and engine
- `IMAGE_GEN_BACKEND`: active simple image-gen backend when relevant
- `SERVICE_HEALTH`: `healthy`, `starting`, or `unhealthy`
- `SIDECAR_HEALTH`: readiness for enabled local NIM sidecars
- `STEP1X_HEALTH`: readiness, missing-runtime diagnostics, and runtime import
  preflight state
- `OVRTX_HEALTH`: render service readiness and `gpu_initialized` state
- `EXAMPLE_DOC`: reproducible example path used for evidence, when applicable
- `LOGS` / `STOP`: exact `docker compose ... logs -f` and `down` commands
- Any missing credentials, port conflicts, GPU/toolkit blockers, runtime
  mount issues, or backend limitations.

## Troubleshooting

### Step1X-Compatible Overlay Fails Before Startup

Check `docker logs texture-gen-step1x` first. Make sure
`TEXTURE_STEP1X_HOST_RUNTIME` points at an existing reviewed runtime directory
and that the container can read the runtime's edit script, Python executable,
auxiliary source trees, CUDA libraries, and model/cache paths.

### Step1X Health Reports Missing Runtime Assets

Inspect the Step1X health payload. Missing `edit_texture.py`, model paths,
Material Anything assets, or required executables make `ready=false`. A
`libnvrtc.so.12` or CuPy/NVRTC error usually means
`TEXTURE_STEP1X_LD_LIBRARY_PATH` does not match the mounted runtime layout. Use
fake-runner smoke only for wiring.

### Texture Agent Cannot Reach Step1X

The overlay must use the Docker DNS endpoint from inside the service:

```text
TA_TEXTURE_ENDPOINT=http://texture-gen-step1x:8000
```

Do not set it to `http://localhost:8018` inside Compose. `localhost` would
resolve to the Texture Agent container itself.

### Step1X Output Paths Are Not Visible To Texture Agent

Use the overlay's shared session volume. Avoid host-only paths in API
responses unless the same path is mounted into every container that needs it.
The overlay mounts `/var/texture-agent/sessions` into both Texture Agent and
Step1X for this reason.

### Inspect Individual Container Logs

Use individual container logs when the aggregate Compose logs are too noisy:

```bash
for c in texture-agent-service texture-gen-step1x ovrtx-rendering-api image-gen-nim llm-nim; do docker logs "$c"; done
```

### OVRTX Is Healthy But Renders Look Too Noisy

Raise `OVRTX_NUM_SENSOR_UPDATES` before collecting shareable evidence when the
default is too noisy for the validation you need.

### `image-gen-nim` Reports `Free GPUs: <None>`

NIM's profile selector considers a GPU occupied if any process holds VRAM on
it. Pin the sidecar to a truly idle GPU by editing both `NVIDIA_VISIBLE_DEVICES`
and `device_ids` in `apps/texture_agent_service/docker-compose.multi-gpu.yml`.

### Texture Agent Ignores A Local NIM Sidecar

Check the container environment:

```bash
docker exec texture-agent-service env | grep TA_IMAGE_GEN
```

When the service is routed to `image-gen-nim`, expect
`TA_IMAGE_GEN_BACKEND=openai`,
`TA_IMAGE_GEN_BASE_URL=http://image-gen-nim:8000/v1`, and
`TA_IMAGE_GEN_API_KEY=not-used`.

### Slow First Request After Restart

Local NIM sidecars and operator-mounted Step1X-compatible runtimes may warm
model weights on first start. Persist cache volumes for repeated development
runs and check sidecar logs before treating the deployment as broken.
