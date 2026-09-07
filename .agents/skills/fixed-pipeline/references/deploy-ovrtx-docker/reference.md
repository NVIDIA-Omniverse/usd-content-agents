---
name: deploy-ovrtx-docker
description: "Fixed pipeline reference: Deploy just the OVRTX rendering API standalone via Docker Compose so one or more agent CLIs / services can point at it as a shared rendering endpoint. Use when user wants to run OVRTX standalone, deploy a shared rendering service, separate the rendering endpoint from an agent service, expose OVRTX to multiple callers, or set RENDER_ENDPOINT to a dedicated render host. Trigger phrases include \"deploy ovrtx\", \"run ovrtx standalone\", \"shared rendering service\", \"ovrtx rendering docker\", \"render endpoint deploy\", \"ovrtx docker compose\", \"standalone rendering\"."
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - ovrtx
  - docker
  - rendering
  - deployment
tools:
  - Shell
  - Docker
  - curl
  - Python
  - Filesystem
compatibility: Requires Docker daemon, Docker Compose, NVIDIA Container Toolkit, NVIDIA GPU with about 16GB+ VRAM or a validated AWS GPU instance, free host port 8001 by default, and OVRTX readiness with gpu_initialized=true.
---

# Fixed Pipeline Reference: Deploy OVRTX Rendering API Standalone

Bring up just the OVRTX rendering API without the rest of an agent service stack. Useful when:

- Multiple agent CLIs (or services) need to share a single rendering host.
- The rendering GPU box is separate from the machine running the CLI / agent service.
- You want to iterate on the agent pipeline without rebooting the rendering container each time.

`apps/ovrtx_rendering_api/` ships the Dockerfile, service code, and standalone
`docker-compose.yml`. Use that compose file for shared render-service
deployments so the command is not semantically tied to material or physics.

## When to Use

- Use when the user wants a standalone OVRTX rendering API without a Material or Physics service stack.
- Use when multiple agent CLIs or services should share one rendering endpoint.
- Use when the render GPU host is separate from the machine running the agent CLI or service.
- Use service-specific Docker deploy skills when the user wants the bundled sidecar managed with that service stack.

## Limitations

- The default standalone stack owns host port 8001. Stop overlapping Texture, Material, Physics, or prior OVRTX stacks before startup.
- OVRTX is not ready on HTTP 200 alone; report ready only when `/health` contains `gpu_initialized: true`.
- First build, shader warm-up, and first render can take several minutes. Return logs and health commands instead of holding an agent session open indefinitely.
- For Brev/AWS validation smoke paths, keep `OVRTX_RENDER_MODE=pt`; do not switch to `rt2` as a workaround for hangs.

## Prerequisites

1. **Docker Compose**: `docker compose version`
2. **NVIDIA GPU** with ~16 GB+ VRAM: `nvidia-smi`
3. **NVIDIA Container Toolkit**: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`

No VLM provider key is required — OVRTX itself does not call a VLM.

## Instructions

1. Confirm Docker, Compose, GPU, NVIDIA Container Toolkit, and port availability before starting OVRTX.
2. Start the standalone OVRTX compose stack from the repo root.
3. Poll `/health` and report readiness only when `gpu_initialized` is `true`.
4. Export `RENDER_ENDPOINT` for any caller that should use the shared endpoint.
5. Return endpoint, health state, log commands, stop commands, and caller environment variables using the output format below.

### Start Standalone OVRTX

Run the standalone OVRTX compose file from the repo root:

```bash
OVRTX_RENDER_MODE=pt docker compose -f apps/ovrtx_rendering_api/docker-compose.yml up --build
```

If you need a different host port:

```bash
OVRTX_HOST_PORT=8010 OVRTX_RENDER_MODE=pt docker compose -f apps/ovrtx_rendering_api/docker-compose.yml up --build
```

First build takes ~10 minutes (OVRTX builds from source). First render takes
several minutes because shader work is cached only after the first render. For
Brev smoke tests, keep `OVRTX_RENDER_MODE=pt` and use
`OVRTX_DAEMON_RENDER_TIMEOUT=900`.

Validated AWS render candidates:

- `g7e.2xlarge` / RTX PRO Server 6000: preferred fast validation path when
  budget allows. `/health` and `/render` passed; cold warm-up completed within
  the 900 second timeout.
- `g6e.xlarge` / L40S: cheaper validated fallback. `/health` and `/render`
  passed with `OVRTX_NUM_SENSOR_UPDATES=1`, but cold warm-up can be much slower
  than RTX PRO Server 6000.

You may temporarily set `OVRTX_NUM_SENSOR_UPDATES=1` to reduce cold-start work,
then restore the service default before quality-sensitive agent runs.

By default the compose file publishes host port **8001** to container port
**8000**. Health check succeeds once the GPU has warmed up
(`gpu_initialized: true`).

## Raw Docker Alternative

Use this when you do not want Compose:

```bash
docker build -f apps/ovrtx_rendering_api/Dockerfile -t ovrtx-rendering-api .
docker run --rm --gpus all \
  --name ovrtx-rendering-api \
  -e OVRTX_RENDER_MODE=pt \
  -e OVRTX_NUM_SENSOR_UPDATES=1 \
  -e OVRTX_DAEMON_RENDER_TIMEOUT=900 \
  -p 8001:8000 \
  ovrtx-rendering-api
```

This is functionally equivalent for smoke testing, but Compose is preferred for
repeatable shared-service deployments because it carries the health check,
restart policy, logging policy, debug volume, and GPU reservation.

## Point agent CLIs / services at it

Once the health check passes, export `RENDER_ENDPOINT` in the environment of the machine running the agent CLI or service:

```bash
export RENDER_ENDPOINT=http://localhost:8001            # same host
export RENDER_ENDPOINT=http://render-host.lan:8001      # remote host
```

For Material Agent using `backend: remote` against OVRTX, also set:

```bash
export MA_RENDERING_USE_DATA_URI=true
```

That keeps the NVCF-compatible renderer client on the direct data-URI path
instead of uploading prepared stages through S3.

Then invoke agent CLIs / services that use `backend: remote` rendering:

```bash
material-agent run apps/material_agent/configs/unified_example.yaml
physics-agent run apps/physics_agent/configs/lightbulb.yaml   # set render.backend: remote before using a shared endpoint
texture-agent run apps/texture_agent/configs/texture_example.yaml
```

`apps/physics_agent/configs/lightbulb.yaml` is public and defaults to local OVRTX rendering. Copy it or edit it for your run, then set the render backend to `remote` before pointing it at a shared `RENDER_ENDPOINT`.

For an agent service deployed via Docker Compose alongside the rendering host, set `RENDER_ENDPOINT=http://render-host.lan:8001` in the agent service's `.env`.

## Check Health

```bash
python - <<'PY'
import json
from urllib.request import urlopen

try:
    with urlopen("http://localhost:8001/health", timeout=10) as response:
        health = json.load(response)
except Exception as exc:
    print(f"OVRTX unreachable: {exc}")
    raise SystemExit(1)

print(json.dumps(health))
if health.get("status") == "unhealthy":
    print("OVRTX unhealthy")
    raise SystemExit(1)
if health.get("gpu_initialized") is True:
    print("OVRTX ready")
else:
    print("OVRTX warming")
PY
```

Until GPU warm-up completes, `gpu_initialized` is `false`. Agent CLIs / services that depend on the endpoint will block on their own retry logic until warm-up finishes.

## Operations

### Logs

```bash
docker compose -f apps/ovrtx_rendering_api/docker-compose.yml logs -f ovrtx-rendering-api
```

### Stop

```bash
docker compose -f apps/ovrtx_rendering_api/docker-compose.yml stop ovrtx-rendering-api
docker compose -f apps/ovrtx_rendering_api/docker-compose.yml rm -f ovrtx-rendering-api
```

For Brev-hosted runs, delete the workspace after the smoke or pipeline run
unless the operator explicitly keeps it. Brev auth can expire during long
build, warm-up, or smoke-test windows; if deletion is blocked, re-login and
repeat `brev delete`. If the instance is still reachable by SSH, power it off
first to stop compute spend while Brev workspace cleanup catches up.

### Rebuild

```bash
docker compose -f apps/ovrtx_rendering_api/docker-compose.yml up --build --force-recreate
```

## GPU Assignment

To pin OVRTX to a specific GPU, edit the relevant agent service's `docker-compose.yml`:

```yaml
ovrtx-rendering-api:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['0']      # specific GPU by ID
            capabilities: [gpu]
```

Or raise `count` to reserve multiple GPUs (OVRTX uses the first one in the assigned set).

## Environment Variables

Configurable via shell environment variables before `docker compose up`:

| Variable | Default | Description |
|---|---|---|
| `OVRTX_LOG_LEVEL` | `warn` | Log verbosity (`trace`, `debug`, `info`, `warn`, `error`) |
| `OVRTX_NUM_SENSOR_UPDATES` | `500` | Sensor update count before capture; lower only for smoke tests |
| `OVRTX_RENDER_MODE` | `pt` | Render mode. Brev/AWS validation smoke paths must stay on `pt`; do not use `rt2` as a workaround for hangs |

## Output Format

When handing control back to the user, report:

- `RENDER_ENDPOINT`: `http://localhost:8001` or the configured remote host/port.
- `OVRTX_HEALTH`: `healthy` only when `/health` contains `"gpu_initialized":true`; otherwise `warming` or `unhealthy`.
- `CALLER_ENV`: `RENDER_ENDPOINT=...` and `MA_RENDERING_USE_DATA_URI=true` when relevant for Material Agent remote rendering.
- `LOGS`: `docker compose -f apps/ovrtx_rendering_api/docker-compose.yml logs -f ovrtx-rendering-api`
- `STOP`: `docker compose -f apps/ovrtx_rendering_api/docker-compose.yml stop ovrtx-rendering-api`
- Any GPU/toolkit blockers, port conflicts, or host-driver issues.

## Troubleshooting

### `gpu_initialized: false` after several minutes

Cause: Shader compilation is slow on first boot (cold cache) or the GPU is being shared with another process. Check `nvidia-smi` for VRAM pressure.
Solution: Wait up to ~5 minutes on cold start. If it persists, make sure no other CUDA process is holding VRAM on the same GPU.

Some Brev L40S provider images can have NVIDIA compute libraries but not the
NVIDIA GL/Vulkan libraries required by OVRTX. Install the matching
`libnvidia-gl` package and regenerate CDI before recreating the container:

```bash
sudo apt-get install -y libnvidia-gl-580=<driver-version> vulkan-tools
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo cp /etc/cdi/nvidia.yaml /var/run/cdi/nvidia.yaml
```

If the daemon reaches readiness but the first Brev/AWS validation render still
hangs, do not switch to `rt2` as a workaround. Keep `OVRTX_RENDER_MODE=pt`, use
a 900 second render timeout for smoke testing, and capture daemon logs. AWS `g7e.2xlarge` / RTX PRO Server
6000 and AWS `g6e.xlarge` / L40S both completed full `/health` plus `/render`
smoke. Prefer RTX PRO Server 6000 for fast validation loops; use L40S when
cost matters more than cold-start latency.

### Port 8001 already in use

Cause: Another process (e.g., another agent service's OVRTX sidecar, or a prior run) is using 8001.
Solution: Stop the other container, or set `OVRTX_HOST_PORT` before starting
the standalone compose file. OVRTX listens on container port 8000; the host
port is the left side of the mapping.

```bash
OVRTX_HOST_PORT=8010 OVRTX_RENDER_MODE=pt docker compose -f apps/ovrtx_rendering_api/docker-compose.yml up --build
```

### Agent CLI fails with connection refused

Cause: OVRTX container still warming up, or `RENDER_ENDPOINT` is unset / pointing at the wrong host.
Solution: Poll `/health` first and confirm `gpu_initialized: true` before invoking the CLI; verify `RENDER_ENDPOINT` with `echo $RENDER_ENDPOINT`.
